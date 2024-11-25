import itertools  # type: ignore

import dask  # type: ignore
import dask.array as da  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import xarray as xr  # climate data manipulation library  # type: ignore
from config import config
from data_preparation import (
    align_daily_data_xr,
    convert_6hr_to_original_xr,
    daily_to_6hourly_xr,
)

# from mrmbc import constants as cons  # type: ignore
from sdmbc_bc_function import (
    bc_correction_future,
    bc_correction_hist,
    bc_correction_with_rescaling,
    empirical_quantile_mapping_future_xarray,
    rescale_to_sum_one,
)

var_list_w = (
    ["w", "ta", "hus"]
    if config.bc_boundary == "lateral"
    else config.target_variable_sst
)
var_list = (
    config.input_vargcm_3d
    if config.bc_boundary == "lateral"
    else config.target_variable_sst
)


def correction_wrapper(gcm_data, obs_data):
    grid_cell_gcm = np.stack(gcm_data, axis=0)  # Shape should be (3, 31, 12, 31)
    grid_cell_obs = np.stack(obs_data, axis=0)  # Shape should be (3, 31, 12, 31)
    result_dict = bc_correction_hist(grid_cell_gcm, grid_cell_obs)
    return result_dict["gcmc"], result_dict["bc_params"]


def correction_wrapper_future(gcm_data, bc_params_array):
    grid_cell_gcm = np.stack(gcm_data, axis=0)  # Shape should be (3, 31, 12, 31)
    result_dict = bc_correction_future(
        grid_cell_gcm, bc_params_array, config.startyear_f, config.endyear_f
    )
    return result_dict["gcmc"]


def process_grid_cell(lat, lon, reshaped_gcm, reshaped_obs):
    gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    obs_data = [reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w]

    # Apply the correction
    gcmc_corrected, bc_params = correction_wrapper(gcm_data, obs_data)

    return {
        "lat": lat,
        "lon": lon,
        "gcmc_corrected": gcmc_corrected,
        "bc_params": bc_params,
    }


def process_batch_of_grid_cells(batch, reshaped_gcm, reshaped_obs):
    batch_results = []

    for lat, lon in batch:
        # Extract GCM and observation data for the current lat/lon
        # print("lat, lon", lat, lon)
        # print("reshaped_gcm", reshaped_gcm)
        # print("reshaped_obs", reshaped_obs)
        # print("var_list_w", var_list_w)
        # print("reshaped_gcm[var]", reshaped_gcm["tos"])
        # print(
        #     "reshaped_gcm[var].sel(lat=lat, lon=lon)",
        #     reshaped_gcm["tos"].sel(lat=lat, lon=lon),
        # )
        # print(
        #     "reshaped_gcm[var].sel(lat=lat, lon=lon).values",
        #     reshaped_gcm["tos"].sel(lat=lat, lon=lon).values,
        # )
        # print("reshaped_obs[var]", reshaped_obs["tos"])
        # print(
        #     "reshaped_obs[var].sel(lat=lat, lon=lon)",
        #     reshaped_obs["tos"].sel(lat=lat, lon=lon),
        # )
        # print(
        #     "reshaped_obs[var].sel(lat=lat, lon=lon).values",
        #     reshaped_obs["tos"].sel(lat=lat, lon=lon).values,
        # )
        try:
            gcm_data = [
                reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w
            ]
        except AttributeError:
            gcm_data = [np.nan] * len(var_list_w)
        try:
            obs_data = [
                reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w
            ]
        except AttributeError:
            obs_data = [np.nan] * len(var_list_w)
        # obs_data = [
        #     reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w
        # ]

        # Check if any NaN values are present in the GCM or observation data
        if np.isnan(gcm_data).any() or np.isnan(obs_data).any():
            # If NaN exists, append NaN to the results
            gcmc_corrected = np.nan
            bc_params = np.nan
        else:
            # If no NaN, perform the bias correction
            gcmc_corrected, bc_params = correction_wrapper(gcm_data, obs_data)

        # Append the result for the current lat/lon
        batch_results.append(
            {
                "lat": lat,
                "lon": lon,
                "gcmc_corrected": gcmc_corrected,
                "bc_params": bc_params,
            }
        )

    return batch_results


def process_batch_of_grid_cells_future(batch, reshaped_gcm, bc_params_array):
    batch_results = []
    for lat, lon in batch:
        try:
            gcm_data = [
                reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w
            ]
        except AttributeError:
            gcm_data = [np.nan] * len(var_list_w)

        # Check if any NaN values are present in the GCM or observation data
        if any(np.isnan(gcm_data)):
            # If NaN exists, append NaN to the results
            gcmc_corrected = np.nan
        else:
            gcmc_corrected = correction_wrapper_future(gcm_data, bc_params_array)

        batch_results.append(
            {
                "lat": lat,
                "lon": lon,
                "gcmc_corrected": gcmc_corrected,
            }
        )
    return batch_results


def process_grid_cell_future(lat, lon, reshaped_gcm, bc_params_array):
    gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    params_data = bc_params_array.sel(lat=lat, lon=lon)
    # Apply the correction
    gcmc_corrected = correction_wrapper_future(gcm_data, params_data)

    return {
        "lat": lat,
        "lon": lon,
        "gcmc_corrected": gcmc_corrected,
    }


# Delay the rescaling and reformatting steps until after the initial corrections are computed
def rescale_and_reformat(gcmc_corrected, ff_gcm, ff_obs):
    # Rescale and Reformat using Fraction Factors and Sliced GCM Data
    if config.sub_daily_correction:
        gcm_corrected = bc_correction_with_rescaling(
            ff_obs, ff_gcm, var_list_w, n_quantiles=100, extrapolation="constant"
        )
        six_hourly_data_hist = daily_to_6hourly_xr(
            gcmc_corrected,
            gcm_corrected,
            var_list_w,
            config.startyear_h,
            config.endyear_h,
        )
    else:
        six_hourly_data_hist = daily_to_6hourly_xr(
            gcmc_corrected, ff_gcm, var_list_w, config.startyear_h, config.endyear_h
        )

    return six_hourly_data_hist


def rescale_and_reformat_future(gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future):
    # Rescale and Reformat using Fraction Factors and Sliced GCM Data
    if config.sub_daily_correction:
        gcm_corrected = empirical_quantile_mapping_future_xarray(
            ff_obs,
            ff_gcm,
            ff_gcm_future,
            var_list_w,
            n_quantiles=100,
            extrapolation="constant",
        )

        fraction_factors_gcm_corrected_future = xr.Dataset()

        for var in var_list:
            sim_data = gcm_corrected[var].values
            corrected_gcm_data_rescaled = rescale_to_sum_one(sim_data)
            # Create a new DataArray and append to the corrected Dataset
            fraction_factors_gcm_corrected_future[var] = xr.DataArray(
                corrected_gcm_data_rescaled,
                dims=gcm_corrected[var].dims,
                coords=gcm_corrected[var].coords,
            )

        six_hourly_data_hist = daily_to_6hourly_xr(
            gcmc_corrected,
            gcm_corrected,
            var_list_w,
            config.startyear_h,
            config.endyear_h,
        )
    else:
        six_hourly_data_hist = daily_to_6hourly_xr(
            gcmc_corrected, ff_gcm, var_list_w, config.startyear_h, config.endyear_h
        )

    return six_hourly_data_hist


# Delay the boundary condition correction if needed
def apply_boundary_correction(six_hourly_data_hist, sliced_gcm):

    if config.bc_boundary == "lateral":
        # Select the u and v wind components for the grid cells
        g_u = sliced_gcm.ua
        g_v = sliced_gcm.va

        six_hourly_data = convert_6hr_to_original_xr(six_hourly_data_hist, g_u, g_v)
        return six_hourly_data
    else:
        return six_hourly_data_hist


def bc_correction_grid_cell_hist_dask(
    reshaped_gcm,
    reshaped_obs,
    ff_gcm,
    ff_obs,
    var_list_w,
    sliced_gcm,
    startyear_h,
    endyear_h,
):
    """
    Perform bias correction across all grid cells in parallel using Dask.

    Parameters:
    - reshaped_gcm: Xarray Dataset containing GCM data with dimensions (year, month, day, lat, lon)
    - reshaped_obs: Xarray Dataset containing observational data with dimensions (year, month, day, lat, lon)
    - ff_gcm: Xarray Dataset containing GCM fraction factors with dimensions (time, lat, lon)
    - ff_obs: Xarray Dataset containing observational fraction factors with dimensions (time, lat, lon)
    - var_list_w: List of variables to be corrected
    - sliced_gcm: Xarray Dataset containing sliced GCM data (ua and va components) with dimensions (time, lat, lon)
    - startyear_h, endyear_h: Start and end years for the historical period

    Returns:
    - ds_corrected: Xarray Dataset with corrected data and bias correction parameters
    """

    # # Generate tasks for each grid cell
    # tasks = [
    #     dask.delayed(process_grid_cell)(lat, lon, reshaped_gcm, reshaped_obs)
    #     for lat, lon in itertools.product(
    #         reshaped_gcm.lat.values, reshaped_gcm.lon.values
    #     )
    # ]

    grid_cells = list(
        itertools.product(reshaped_gcm.lat.values, reshaped_gcm.lon.values)
    )
    batch_size = 5  # Process 20 grid cells at a time
    tasks = [
        dask.delayed(process_batch_of_grid_cells)(
            grid_cells[i : i + batch_size], reshaped_gcm, reshaped_obs
        )
        for i in range(0, len(grid_cells), batch_size)
    ]

    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]

    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty((31, 12, 31, len(reshaped_gcm.lat), len(reshaped_gcm.lon)))
        for var in var_list_w
    }
    bc_params_array = np.empty(
        (len(reshaped_gcm.lat), len(reshaped_gcm.lon)), dtype=object
    )

    # Fill the arrays with data from results
    for result in flattened_results:
        lat_idx = np.where(reshaped_gcm.lat.values == result["lat"])[0][0]
        lon_idx = np.where(reshaped_gcm.lon.values == result["lon"])[0][0]
        for i, var in enumerate(var_list_w):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]
        bc_params_array[lat_idx, lon_idx] = result["bc_params"].to_dict()

    # Convert to Xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in var_list_w
        },
        coords={
            "year": reshaped_gcm.year,
            "month": reshaped_gcm.month,
            "day": reshaped_gcm.day,
            "lat": reshaped_gcm.lat,
            "lon": reshaped_gcm.lon,
        },
    )

    six_hourly_data_hist = rescale_and_reformat(gcmc_corrected, ff_gcm, ff_obs)

    bc_corrected_6hourly_data = apply_boundary_correction(
        six_hourly_data_hist, sliced_gcm
    )

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(bc_corrected_6hourly_data, bc_params_array)

    return final_corrected_data[0], final_corrected_data[1]


def bc_correction_grid_cell_hist_dask_2d(
    reshaped_gcm,
    reshaped_obs,
    var_list_w,
    startyear_h,
):
    """
    Perform bias correction across all grid cells in parallel using Dask.

    Parameters:
    - reshaped_gcm: Xarray Dataset containing GCM data with dimensions (year, month, day, lat, lon)
    - reshaped_obs: Xarray Dataset containing observational data with dimensions (year, month, day, lat, lon)
    - ff_gcm: Xarray Dataset containing GCM fraction factors with dimensions (time, lat, lon)
    - ff_obs: Xarray Dataset containing observational fraction factors with dimensions (time, lat, lon)
    - var_list_w: List of variables to be corrected
    - sliced_gcm: Xarray Dataset containing sliced GCM data (ua and va components) with dimensions (time, lat, lon)
    - startyear_h, endyear_h: Start and end years for the historical period

    Returns:
    - ds_corrected: Xarray Dataset with corrected data and bias correction parameters
    """

    # # Generate tasks for each grid cell
    # tasks = [
    #     dask.delayed(process_grid_cell)(lat, lon, reshaped_gcm, reshaped_obs)
    #     for lat, lon in itertools.product(
    #         reshaped_gcm.lat.values, reshaped_gcm.lon.values
    #     )
    # ]

    grid_cells = list(
        itertools.product(reshaped_gcm.lat.values, reshaped_gcm.lon.values)
    )
    batch_size = 10  # Process 20 grid cells at a time
    tasks = [
        dask.delayed(process_batch_of_grid_cells)(
            grid_cells[i : i + batch_size], reshaped_gcm, reshaped_obs
        )
        for i in range(0, len(grid_cells), batch_size)
    ]

    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]

    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty((31, 12, 31, len(reshaped_gcm.lat), len(reshaped_gcm.lon)))
        for var in var_list_w
    }
    bc_params_array = np.empty(
        (len(reshaped_gcm.lat), len(reshaped_gcm.lon)), dtype=object
    )

    # Fill the arrays with data from results
    for result in flattened_results:
        lat_idx = np.where(reshaped_gcm.lat.values == result["lat"])[0][0]
        lon_idx = np.where(reshaped_gcm.lon.values == result["lon"])[0][0]
        for i, var in enumerate(var_list_w):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]
        bc_params_array[lat_idx, lon_idx] = result["bc_params"].to_dict()

    # Convert to Xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in var_list_w
        },
        coords={
            "year": reshaped_gcm.year,
            "month": reshaped_gcm.month,
            "day": reshaped_gcm.day,
            "lat": reshaped_gcm.lat,
            "lon": reshaped_gcm.lon,
        },
    )

    # Align the daily data with the generated dates
    daily_data_aligned = align_daily_data_xr(gcmc_corrected, startyear_h)

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(daily_data_aligned, bc_params_array)

    return final_corrected_data[0], final_corrected_data[1]


def bc_correction_grid_cell_future_subdaily_dask(
    ff_gcm,
    ff_obs,
    gcm_future,
    ff_gcm_future,
    bc_params_array,
    startyear_f,
    endyear_f,
    var_list_w,
    sliced_gcm_future,
):

    # Generate tasks for each grid cell
    grid_cells = list(itertools.product(gcm_future.lat.values, gcm_future.lon.values))
    batch_size = 20  # Process 20 grid cells at a time
    tasks = [
        dask.delayed(process_batch_of_grid_cells_future)(
            grid_cells[i : i + batch_size], gcm_future, bc_params_array
        )
        for i in range(0, len(grid_cells), batch_size)
    ]
    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)

    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty((31, 12, 31, len(gcm_future.lat), len(gcm_future.lon)))
        for var in var_list_w
    }

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]

    # Fill the arrays with data from results
    for result in flattened_results:
        lat_idx = np.where(gcm_future.lat.values == result["lat"])[0][0]
        lon_idx = np.where(gcm_future.lon.values == result["lon"])[0][0]
        for i, var in enumerate(var_list_w):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

    # Convert to Xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in var_list_w
        },
        coords={
            "year": gcm_future.year,
            "month": gcm_future.month,
            "day": gcm_future.day,
            "lat": gcm_future.lat,
            "lon": gcm_future.lon,
        },
    )

    six_hourly_data = rescale_and_reformat_future(
        gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future
    )

    # g_u = sliced_gcm_future.ua.sel(time=slice(str(startyear_f), str(endyear_f)))
    # g_v = sliced_gcm_future.va.sel(time=slice(str(startyear_f), str(endyear_f)))
    # six_hourly_data = six_hourly_data.sel(time=slice(str(startyear_f), str(endyear_f)))

    bc_corrected_6hourly_data = apply_boundary_correction(
        six_hourly_data, sliced_gcm_future
    )

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(bc_corrected_6hourly_data)

    return final_corrected_data[0]


def bc_correction_grid_cell_future_daily_dask(
    ff_gcm,
    ff_obs,
    gcm_future,
    ff_gcm_future,
    bc_params_array,
    startyear_f,
    endyear_f,
    var_list_w,
    sliced_gcm_future,
):

    grid_cells = list(itertools.product(gcm_future.lat.values, gcm_future.lon.values))
    batch_size = 20  # Process 20 grid cells at a time
    tasks = [
        dask.delayed(process_batch_of_grid_cells_future)(
            grid_cells[i : i + batch_size], gcm_future, bc_params_array
        )
        for i in range(0, len(grid_cells), batch_size)
    ]
    # # Generate tasks for each grid cell
    # tasks = [
    #     process_grid_cell_future(lat, lon, gcm_future, bc_params_array)
    #     for lat, lon in itertools.product(gcm_future.lat.values, gcm_future.lon.values)
    # ]

    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]

    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty((31, 12, 31, len(gcm_future.lat), len(gcm_future.lon)))
        for var in var_list_w
    }

    # Fill the arrays with data from results
    for result in flattened_results:
        lat_idx = np.where(gcm_future.lat.values == result["lat"])[0][0]
        lon_idx = np.where(gcm_future.lon.values == result["lon"])[0][0]
        for i, var in enumerate(var_list_w):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

    # Convert to Xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in var_list_w
        },
        coords={
            "year": gcm_future.year,
            "month": gcm_future.month,
            "day": gcm_future.day,
            "lat": gcm_future.lat,
            "lon": gcm_future.lon,
        },
    )

    six_hourly_data = rescale_and_reformat_future(
        gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future
    )

    # g_u = sliced_gcm_future.ua.sel(time=slice(str(startyear_f), str(endyear_f)))
    # g_v = sliced_gcm_future.va.sel(time=slice(str(startyear_f), str(endyear_f)))
    # six_hourly_data = six_hourly_data.sel(time=slice(str(startyear_f), str(endyear_f)))

    bc_corrected_6hourly_data = apply_boundary_correction(
        six_hourly_data, sliced_gcm_future
    )

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(bc_corrected_6hourly_data)

    return final_corrected_data[0]


def is_leap_year(year):
    """Check if a year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def adjust_dates_to_target_year(corrected_data, target_year):

    # Create 'hourofyear' directly, excluding February 29 if necessary
    if is_leap_year(target_year):
        corrected_data["hourofyear"] = xr.DataArray(
            corrected_data.indexes["time"].strftime("%m-%d %H"),
            coords=corrected_data.time.coords,
        )
    else:
        # Exclude February 29 by creating a mask for leap day and filtering it out
        corrected_data = corrected_data.sel(
            time=~(
                (corrected_data.time.dt.month == 2) & (corrected_data.time.dt.day == 29)
            )
        )
        corrected_data["hourofyear"] = xr.DataArray(
            corrected_data.indexes["time"].strftime("%m-%d %H"),
            coords=corrected_data.time.coords,
        )

    # Group by 'hourofyear' and calculate the mean
    result_filtered = corrected_data.groupby("hourofyear").mean("time")

    # Determine the end date correctly based on leap year
    new_time_series = pd.date_range(
        start=f"{target_year}-01-01", end=f"{target_year}-12-31 23:59", freq="6h"
    )

    # Assuming the length matches because we directly managed leap years
    result_filtered = result_filtered.assign_coords(hourofyear=new_time_series)
    result_filtered = result_filtered.rename({"hourofyear": "time"})

    return result_filtered


def apply_moving_window_bias_correction(
    lat,
    lon,
    gcm_future,
    gcm_ref,
    fraction_factors,
    bc_params_array,
    start_year,
    end_year,
    window_size=30,
):
    corrected_dataset = xr.Dataset()

    for target_year in range(start_year, end_year + 1):
        pre_window = min(window_size // 2, target_year - start_year)
        post_window = min(window_size // 2, end_year - target_year)

        # If not enough data for a 30-year window, adjust to use as much as possible
        if pre_window + post_window + 1 < window_size:
            if target_year - start_year < window_size // 2:
                # Closer to start, prioritize adding years after the target year
                actual_window_start = min(start_year, target_year)
                actual_window_end = min(
                    end_year, target_year + 15
                )  # Use 15 years after, if available
            else:
                # Closer to end, prioritize adding years before the target year
                actual_window_start = max(
                    start_year, target_year - 15
                )  # Use 15 years before, if available
                actual_window_end = max(target_year, end_year)
        else:
            actual_window_start = target_year - pre_window
            actual_window_end = target_year + post_window

        window_data = gcm_future.sel(
            year=slice(str(actual_window_start), str(actual_window_end))
        )
        fraction_factor = fraction_factors.sel(
            time=slice(str(actual_window_start), str(actual_window_end))
        )

        if config.sub_daily_correction:
            corrected_data = bc_correction_grid_cell_future_subdaily_dask(
                lat,
                lon,
                fraction_factor,
                fraction_factor,
                window_data,
                fraction_factor,
                bc_params_array,
                actual_window_start,
                actual_window_end,
                var_list,
                gcm_future,
            )
        else:
            corrected_data = bc_correction_grid_cell_future_daily_dask(
                lat,
                lon,
                gcm_ref,
                window_data,
                fraction_factor,
                bc_params_array,
                actual_window_start,
                actual_window_end,
                var_list_w,
            )

        # Adjusting datetime to target year and handling February 29th
        # corrected_data = adjust_dates_to_target_year(corrected_data, target_year)
        corrected_data = corrected_data.sel(
            time=slice(f"{target_year}-01-01", f"{target_year}-12-31")
        )

        if not pd.Index(corrected_data["time"].values).is_monotonic_increasing:
            corrected_data = corrected_data.sortby("time")
        if target_year == start_year:
            corrected_dataset = xr.merge(
                [corrected_dataset, corrected_data], compat="override"
            )
        else:
            corrected_dataset = xr.concat(
                [corrected_dataset, corrected_data], dim="time"
            )

        print(f"{target_year} completed")

    return corrected_dataset


# def bc_correction_grid_cell_hist(
#     lat, lon, reshaped_gcm, reshaped_obs, ff_gcm, ff_obs, var_list, sliced_gcm
# ):
#     # Select individual grid cells
#     # These part should be outside of the function if dask array is used
#     # ---------------------------------------------------------------------------
#     # selected_gcm = gcm_3d_input.sel(lon=lon, lat=lat)
#     # selected_obs = obs_3d_input.sel(lon=lon, lat=lat)

#     # assign_gcm = assign_w_6hr(selected_gcm, bc_boundary)
#     # assign_obs = assign_w_6hr(selected_obs, bc_boundary)

#     # # Convert 6hourly to daily and calculate the fraction
#     # daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(assign_gcm)
#     # daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(assign_obs)
#     # ---------------------------------------------------------------------------
#     grid_cell_gcm = reshaped_gcm.sel(lat=lat, lon=lon).to_array().values
#     grid_cell_obs = reshaped_obs.sel(lat=lat, lon=lon).to_array().values
#     fraction_factors_gcm = ff_gcm.sel(lat=lat, lon=lon)
#     fraction_factors_obs = ff_obs.sel(lat=lat, lon=lon)
#     # print('grid_cell_gcm shape is: ', grid_cell_gcm.shape)
#     # print('grid_cell_gcm[0,0,0,:], is: ', grid_cell_gcm[0,0,0,:])
#     if config.sub_daily_correction:
#         # Apply empirical quantile mapping to the xarray Dataset
#         # Other option for bias correction will be added later
#         gcm_corrected = empirical_quantile_mapping_xarray(
#             fraction_factors_obs,
#             fraction_factors_gcm,
#             var_list,
#             n_quantiles=100,
#             extrapolation="constant",
#         )

#         # fraction_factors_gcm_corrected = xr.Dataset()

#         # for var in var_list:
#         # sim_data = gcm_corrected[var].values
#         corrected_gcm_data_rescaled = gcm_corrected
#         # print(corrected_gcm_data_rescaled)
#         # Create a new DataArray and append to the corrected Dataset
#         # fraction_factors_gcm_corrected[var] = xr.DataArray(
#         #     corrected_gcm_data_rescaled,
#         #     dims=gcm_corrected[var].dims,
#         #     coords=gcm_corrected[var].coords,
#         # )

#     # print(" ----------------------------------------------------------- ")
#     # print(" Bias correction for current climate ")
#     # print(" ----------------------------------------------------------- ")
#     result_dict = bc_correction_hist(grid_cell_gcm, grid_cell_obs)
#     gcmc_hist = result_dict["gcmc"]
#     bc_params = result_dict["bc_params"]
#     # print(gcmc_hist[0,0,0,:])

#     if config.sub_daily_correction:
#         # print(" ----------------------------------------------------------- ")
#         # print(" Bias correction for sub-daily data ")
#         # print(" ----------------------------------------------------------- ")
#         six_hourly_data_hist = daily_to_6hourly_xr(
#             gcmc_hist,
#             corrected_gcm_data_rescaled,
#             var_list,
#             31,
#             12,
#             31,
#             config.startyear_h,
#         )
#     else:
#         six_hourly_data_hist = daily_to_6hourly_xr(
#             gcmc_hist,
#             fraction_factors_gcm,
#             var_list,
#             31,
#             12,
#             31,
#             config.startyear_h,
#         )

#     # g_u = selected_gcm.u.values.flatten()
#     # g_v = selected_gcm.v.values.flatten()
#     g_u = sliced_gcm.ua.sel(lat=lat, lon=lon)
#     g_v = sliced_gcm.va.sel(lat=lat, lon=lon)

#     # Convert to original data form
#     if config.bc_boundary == "lateral":
#         bc_corrected_6hourly_data = convert_6hr_to_original_xr(
#             six_hourly_data_hist, g_u, g_v
#         )

#         return bc_corrected_6hourly_data, bc_params
#     elif config.bc_boundary == "surface":
#         return six_hourly_data_hist, bc_params


# def apply_moving_window_bias_correction(
#     lat,
#     lon,
#     gcm_future,
#     gcm_ref,
#     fraction_factors,
#     bc_params_array,
#     start_year,
#     end_year,
#     window_size=30,
# ):
#     corrected_dataset = None

#     for target_year in range(start_year, end_year + 1):
#         # Adjust window bounds
#         window_start = max(start_year, target_year - window_size // 2)
#         window_end = min(end_year, window_start + window_size - 1)
#         window_start = max(start_year, window_end - window_size + 1)

#         window_data = gcm_future.sel(year=slice(str(window_start), str(window_end)))
#         fraction_factor = fraction_factors.sel(
#             time=slice(str(window_start), str(window_end))
#         )

#         # Apply bias correction
#         corrected_data = bc_correction_grid_cell_future_daily(
#             lat,
#             lon,
#             gcm_ref,
#             window_data,
#             fraction_factor,
#             bc_params_array,
#             window_start,
#             window_end,
#             var_list_w,
#         )

#         # Ensure time dimension is monotonically increasing
#         if not pd.Index(corrected_data["time"].values).is_monotonic_increasing:
#             corrected_data = corrected_data.sortby("time")

#         # Align the time dimension of corrected_data with corrected_dataset
#         if corrected_dataset is not None:
#             corrected_dataset, corrected_data = xr.align(
#                 corrected_dataset, corrected_data, join="outer"
#             )

#         # Combine the datasets
#         if corrected_dataset is not None:
#             corrected_dataset.update(corrected_data)
#         else:
#             corrected_dataset = corrected_data

#         print(f"{target_year} completed")

#     return corrected_dataset
