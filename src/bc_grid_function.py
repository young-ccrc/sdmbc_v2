"""
Bias Correction Grid Function Module for SDMBCv2

This script provides core functionality for performing bias correction on climate model data using the SDMBCv2 framework.
It includes various utilities for applying bias correction to historical data, rescaling future climate projections,
and handling sub-daily correction with boundary adjustments.

Main features of this script:
- Uses Dask for distributed parallel processing.
- Implements historical and future bias correction methods.
- Supports splitting spatial domains into grid cells for large-scale data correction.
- Incorporates functionalities for boundary conditions and rescaling.

This module is intended to be used as part of the SDMBCv2 package for correcting bias in GCM (Global Climate Model) and observational datasets.
"""

import itertools  # type: ignore
import logging
import os
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

import dask  # type: ignore
import dask.array as da  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import xarray as xr  # climate data manipulation library  # type: ignore

# from config import config
from data_preparation import (
    align_daily_data_xr,
    assign_w_6hr,
    convert_6hr_to_original_xr,
    convert_to_daily_with_fraction,
    daily_to_6hourly_xr,
    extract_and_reshape_delayed,
    load_and_combine_variables,
)

# from mrmbc import constants as cons  # type: ignore
from sdmbc_bc_function import (
    bc_correction_future,
    bc_correction_hist,
    quantile_mapping_rescale_all,
    quantile_mapping_rescale_future,
    rescale_to_sum_one,
)

# Suppress INFO and lower-level logs
logging.getLogger("flox").setLevel(logging.WARNING)
logging.getLogger("xarray").setLevel(logging.WARNING)
logging.getLogger("dask").setLevel(logging.WARNING)

var_list_w = (
    ["w", "ta", "hus"] if config.bc_boundary == "lateral" else config.target_variable
)
var_list = (
    config.target_variable
    if config.bc_boundary == "lateral"
    else config.target_variable
)


def correction_wrapper(config, gcm_data, obs_data):
    """
    Wrapper function to apply bias correction for historical GCM data.

    Args:
        gcm_data (array-like): Stacked GCM (Global Climate Model) data for each variable.
        obs_data (array-like): Stacked observational data for each variable.

    Returns:
        tuple: Corrected GCM data and bias correction parameters.
    """
    grid_cell_gcm = np.stack(gcm_data, axis=0)  # Shape should be (3, 31, 12, 31)
    grid_cell_obs = np.stack(obs_data, axis=0)  # Shape should be (3, 31, 12, 31)
    result_dict = bc_correction_hist(config, grid_cell_gcm, grid_cell_obs)
    return result_dict["gcmc"], result_dict["bc_params"]


def correction_wrapper_future(config, gcm_data, bc_params_array):
    """
    Wrapper function to apply bias correction for future GCM data.

    Args:
        gcm_data (array-like): Stacked GCM data for each variable.
        bc_params_array (array-like): Bias correction parameters from historical bias correction.

    Returns:
        array-like: Bias-corrected future GCM data.
    """
    grid_cell_gcm = np.stack(gcm_data, axis=0)  # Shape should be (3, 31, 12, 31)
    result_dict = bc_correction_future(config, grid_cell_gcm, bc_params_array)
    return result_dict


def process_grid_cell(config, lat, lon, reshaped_gcm, reshaped_obs):
    """
    Perform bias correction for a single grid cell.

    Args:
        lat (float): Latitude of the grid cell.
        lon (float): Longitude of the grid cell.
        reshaped_gcm (xarray.Dataset): GCM data reshaped for correction.
        reshaped_obs (xarray.Dataset): Observational data reshaped for correction.

    Returns:
        dict: Contains corrected GCM data and bias correction parameters.
    """
    gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    obs_data = [reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w]

    # Apply the correction
    gcmc_corrected, bc_params = correction_wrapper(config, gcm_data, obs_data)

    return {
        "lat": lat,
        "lon": lon,
        "gcmc_corrected": gcmc_corrected,
        "bc_params": bc_params,
    }


def process_grid_cell_future(config, lat, lon, reshaped_gcm, bc_params_array):
    """
    Perform bias correction for a single grid cell for future data.

    Args:
        lat (float): Latitude of the grid cell.
        lon (float): Longitude of the grid cell.
        reshaped_gcm (xarray.Dataset): GCM data reshaped for correction.
        bc_params_array (xarray.Dataset): Bias correction parameters.

    Returns:
        dict: Contains bias-corrected GCM data for the future period.
    """
    gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]

    # Find the nearest latitude and longitude indices in reshaped_gcm_delayed_f
    lat_array = reshaped_gcm.lat.values
    lon_array = reshaped_gcm.lon.values

    lat_idx = (np.abs(lat_array - lat)).argmin()  # Index of the nearest latitude
    lon_idx = (np.abs(lon_array - lon)).argmin()  # Index of the nearest longitude

    # Select the corresponding value in bc_params_array_loaded
    params_data = bc_params_array[lat_idx, lon_idx]
    # Apply the correction
    gcmc_corrected = correction_wrapper_future(config, gcm_data, params_data)

    return {
        "lat": lat,
        "lon": lon,
        "gcmc_corrected": gcmc_corrected,
    }


# Delay the rescaling and reformatting steps until after the initial corrections are computed
def rescale_and_reformat(config, gcmc_corrected, ff_gcm, ff_obs):
    """
    Rescale and reformat corrected GCM data using fraction factors for sub-daily adjustments.

    Args:
        gcmc_corrected (xarray.Dataset): Bias-corrected GCM data.
        ff_gcm (xarray.Dataset): Fraction factors for GCM data.
        ff_obs (xarray.Dataset): Fraction factors for observational data.

    Returns:
        xarray.Dataset: Six-hourly rescaled bias-corrected GCM data.
    """
    # Rescale and Reformat using Fraction Factors and Sliced GCM Data
    if config.sub_daily_correction:
        ff_corrected = quantile_mapping_rescale_all(config, ff_obs, ff_gcm)
        six_hourly_data_hist = daily_to_6hourly_xr(
            config,
            gcmc_corrected,
            ff_corrected,
            var_list_w,
            config.startyear_h,
            config.endyear_h,
        )
    else:
        six_hourly_data_hist = daily_to_6hourly_xr(
            config,
            gcmc_corrected,
            ff_gcm,
            var_list_w,
            config.startyear_h,
            config.endyear_h,
        )

    return six_hourly_data_hist


# def rescale_and_reformat_future(gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future):
#     """
#     Rescale and reformat corrected GCM data for future scenarios using fraction factors.

#     Args:
#         gcmc_corrected (xarray.Dataset): Corrected GCM data.
#         ff_gcm (xarray.Dataset): Fraction factors for GCM data.
#         ff_obs (xarray.Dataset): Fraction factors for observational data.
#         ff_gcm_future (xarray.Dataset): Fraction factors for future GCM data.

#     Returns:
#         xarray.Dataset: Rescaled future six-hourly bias-corrected GCM data.
#     """
#     # Rescale and Reformat using Fraction Factors and Sliced GCM Data
#     if config.sub_daily_correction:
#         gcm_corrected = empirical_quantile_mapping_future_xarray(
#             ff_obs,
#             ff_gcm,
#             ff_gcm_future,
#             var_list_w,
#             n_quantiles=100,
#             extrapolation="constant",
#         )

#         fraction_factors_gcm_corrected_future = xr.Dataset()

#         for var in var_list:
#             sim_data = gcm_corrected[var].values
#             corrected_gcm_data_rescaled = rescale_to_sum_one(sim_data)
#             # Create a new DataArray and append to the corrected Dataset
#             fraction_factors_gcm_corrected_future[var] = xr.DataArray(
#                 corrected_gcm_data_rescaled,
#                 dims=gcm_corrected[var].dims,
#                 coords=gcm_corrected[var].coords,
#             )

#         six_hourly_data_hist = daily_to_6hourly_xr(
#             gcmc_corrected,
#             gcm_corrected,
#             var_list_w,
#             config.startyear_h,
#             config.endyear_h,
#         )
#     else:
#         six_hourly_data_hist = daily_to_6hourly_xr(
#             gcmc_corrected, ff_gcm, var_list_w, config.startyear_h, config.endyear_h
#         )

#     return six_hourly_data_hist


# Delay the boundary condition correction if needed
def apply_boundary_correction(config, six_hourly_data_hist, sliced_gcm):
    """
    Apply boundary correction to the six-hourly bias-corrected GCM data.

    Args:
        six_hourly_data_hist (xarray.Dataset): Six-hourly rescaled bias-corrected GCM data.
        sliced_gcm (xarray.Dataset): GCM data with selected boundary components.

    Returns:
        xarray.Dataset: Bias-corrected GCM data with boundaries applied.
    """

    if config.bc_boundary == "lateral":
        # Select the u and v wind components for the grid cells
        g_u = sliced_gcm.ua
        g_v = sliced_gcm.va

        six_hourly_data = convert_6hr_to_original_xr(
            config, six_hourly_data_hist, g_u, g_v
        )
        return six_hourly_data
    else:
        return six_hourly_data_hist


# def process_tile(
#     tile,
#     variables,
#     file_paths_by_variable_gcm,
#     level,
#     config,
#     temp_dir,
# ):
#     """
#     Process a single tile for bias correction, involving loading GCM and observational data,
#     performing bias correction, and saving the outputs.

#     Args:
#         tile (dict): Dictionary defining the spatial bounds of the tile.
#         variables (list): List of variable names to process.
#         file_paths_by_variable_gcm (dict): File paths for GCM data by variable.
#         file_paths_by_variable_obs (dict): File paths for observational data by variable.
#         level (int): Processing level for multilevel data.
#         config (module): Configuration object for bias correction parameters.
#         temp_dir (str): Path to the temporary directory for intermediate files.

#     Returns:
#         None
#     """

#     """Process a single tile for bias correction."""
#     lat_range = (tile["lat_min"], tile["lat_max"])
#     lon_range = (tile["lon_min"], tile["lon_max"])

#     # ------------------ Load GCM Data ------------------
#     sliced_gcm = xr.Dataset()
#     for var_name, file_paths in file_paths_by_variable_gcm.items():
#         data_var = load_preprocess_variable(
#             file_paths,
#             var_name,
#             level,
#             lat_range,
#             lon_range,
#             config.startyear_h,
#             config.endyear_h,
#         )

#         # Check if the variable is one of the wind components with different lon
#         if var_name in ["ua", "va"]:
#             # Let's assume hus and ta have the target longitude values, and they are already loaded
#             target_lon = sliced_gcm.lon if "lon" in sliced_gcm else data_var.lon
#             target_lat = sliced_gcm.lat if "lat" in sliced_gcm else data_var.lat
#             target_lev = sliced_gcm.lev if "lev" in sliced_gcm else data_var.lev

#             # Interpolate va to match the target latitude grid
#             if not data_var.lat.equals(target_lat):
#                 data_var = data_var.interp(
#                     lat=target_lat,
#                     method="linear",
#                     kwargs={"fill_value": "extrapolate"},
#                 )
#             if not data_var.lon.equals(target_lon):
#                 data_var = data_var.interp(
#                     lon=target_lon,
#                     method="linear",
#                     kwargs={"fill_value": "extrapolate"},
#                 )
#             # Assign the adjusted longitude values to ua or va
#             data_var = data_var.assign_coords(
#                 lon=target_lon, lat=target_lat, lev=target_lev
#             )
#         if isinstance(data_var, xr.Dataset):  # Ensure we extract the correct DataArray
#             data_var = data_var[var_name]
#         sliced_gcm[var_name] = data_var

#     if config.bc_boundary == "lateral":
#         # sliced_gcm = sliced_gcm_all[0]
#         assign_gcm = assign_w_6hr(sliced_gcm, config.bc_boundary)
#         daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(assign_gcm)
#         # daily_gcm = daily_gcm.chunk({"time": 1000, "lat": "auto", "lon": "auto"})
#         # daily_gcm_rechunk = daily_gcm.chunk({"time": 1000, "lat": -1, "lon": -1})
#         # fraction_factors_gcm = fraction_factors_gcm.chunk(
#         #     {"time": 1000, "lat": "auto", "lon": "auto"}
#         # )
#     else:
#         daily_gcm = sliced_gcm

#     # print("start delayed process gcm")
#     # print("daily_gcm", daily_gcm)
#     reshaped_gcm_delayed = extract_and_reshape_delayed(
#         daily_gcm,
#         config.no_of_variables,
#         config.startyear_h,
#         config.endyear_h,
#         config.bc_boundary,
#     )
#     # print("reshaped_gcm_delayed", reshaped_gcm_delayed)
#     if config.bc_boundary != "lateral":
#         reshaped_gcm_delayed += 273.15

#     # ------------------ Load Observational Data ------------------
#     sliced_obs = xr.Dataset()
#     # for var_name, file_paths in file_paths_by_variable_obs.items():
#     #     obs_var = load_preprocess_variable(file_paths, var_name, level, lat_range, lon_range, config.startyear_h, config.endyear_h)
#     #     if isinstance(obs_var, xr.Dataset):  # Ensure we extract the correct DataArray
#     #         obs_var = obs_var[var_name]
#     #     sliced_obs[var_name] = obs_var
#     for var_name in variables:
#         # Construct the path to the preprocessed file for this variable
#         obs_file = os.path.join(
#             temp_dir,
#             f"preprocessed_obs_{var_name}_lev_{level}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc",
#         )
#         obs_var = xr.open_dataset(obs_file)[var_name]  # Load the variable from the file
#         sliced_obs[var_name] = obs_var  # Add it to the observational dataset

#     sliced_obs = sliced_obs.sel(
#         lat=slice(sliced_gcm.lat.min().item(), sliced_gcm.lat.max().item()),
#         lon=slice(sliced_gcm.lon.min().item(), sliced_gcm.lon.max().item()),
#     )

#     if config.bc_boundary == "lateral":
#         # sliced_obs = sliced_obs_all[0]
#         assign_obs = assign_w_6hr(sliced_obs, config.bc_boundary)
#         daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(assign_obs)
#         # daily_obs = daily_obs.chunk({"time": 1000, "lat": "auto", "lon": "auto"})
#         # daily_obs_rechunk = daily_obs.chunk({"time": 1000, "lat": -1, "lon": -1})
#         # fraction_factors_obs = fraction_factors_obs.chunk(
#         #     {"time": 1000, "lat": "auto", "lon": "auto"}
#         # )
#     else:
#         daily_obs = sliced_obs

#     # print("start delayed process obs")
#     # print("daily_obs", daily_obs)
#     reshaped_obs_delayed = extract_and_reshape_delayed(
#         daily_obs,
#         config.no_of_variables,
#         config.startyear_h,
#         config.endyear_h,
#         config.bc_boundary,
#     )
#     # print("reshaped_obs_delayed", reshaped_obs_delayed)

#     # Perform bias correction across the tile
#     if config.bc_boundary == "lateral":
#         bc_corrected_gcm_hist_tile, bc_params_tile = bc_correction_grid_cell_hist_dask(
#             reshaped_gcm_delayed,
#             reshaped_obs_delayed,
#             fraction_factors_gcm,
#             fraction_factors_obs,
#             var_list_w,
#             sliced_gcm,
#             config.startyear_h,
#             config.endyear_h,
#         )
#     else:
#         bc_corrected_gcm_hist_tile, bc_params_tile = (
#             bc_correction_grid_cell_hist_dask_2d(
#                 reshaped_gcm_delayed,
#                 reshaped_obs_delayed,
#                 var_list_w,
#                 config.startyear_h,
#             )
#         )

#     return bc_corrected_gcm_hist_tile, bc_params_tile


def process_tile(
    config,
    sliced_gcm,
    sliced_obs,
    config,
):
    """
    Process a single tile for bias correction, involving loading GCM and observational data,
    performing bias correction, and saving the outputs.

    Args:
        sliced_gcm (xarray.Dataset): Sliced GCM data for the tile.
        sliced_obs (xarray.Dataset): Sliced observational data for the tile.
        config (module): Configuration object for bias correction parameters.

    Returns:
        tuple: Corrected GCM data and bias correction parameters.
    """

    # ------------------ Load GCM Data ------------------
    if config.bc_boundary == "lateral":
        assign_gcm = assign_w_6hr(config, sliced_gcm, config.bc_boundary)
        daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(
            config, assign_gcm
        )
    else:
        daily_gcm = sliced_gcm

    reshaped_gcm_delayed = extract_and_reshape_delayed(
        config,
        daily_gcm,
        config.no_of_variables,
        config.startyear_h,
        config.endyear_h,
        config.bc_boundary,
    )

    if config.bc_boundary != "lateral":
        reshaped_gcm_delayed += 273.15

    # ------------------ Load Observational Data ------------------
    if config.bc_boundary == "lateral":
        assign_obs = assign_w_6hr(config, sliced_obs, config.bc_boundary)
        daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(
            config, assign_obs
        )
    else:
        daily_obs = sliced_obs

    reshaped_obs_delayed = extract_and_reshape_delayed(
        config,
        daily_obs,
        config.no_of_variables,
        config.startyear_h,
        config.endyear_h,
        config.bc_boundary,
    )

    # Perform bias correction across the tile
    if config.bc_boundary == "lateral":
        bc_corrected_gcm_hist_tile, bc_params_tile = bc_correction_grid_cell_hist_dask(
            config,
            reshaped_gcm_delayed,
            reshaped_obs_delayed,
            fraction_factors_gcm,
            fraction_factors_obs,
            var_list_w,
            sliced_gcm,
        )

    else:
        bc_corrected_gcm_hist_tile, bc_params_tile = (
            bc_correction_grid_cell_hist_dask_2d(
                config,
                reshaped_gcm_delayed,
                reshaped_obs_delayed,
                var_list_w,
                config.startyear_h,
            )
        )

    return bc_corrected_gcm_hist_tile, bc_params_tile


def future_subdaily_correction(
    config,
    sliced_gcm,
    sliced_obs,
    ff_future,
    bc_corrected_gcm_future_tile,
    config,
):
    """
    Process a single tile for bias correction for future sub-daily data.

    Args:
        tile (dict): Dictionary defining the spatial bounds of the tile.
        sliced_gcm (xarray.Dataset): Sliced GCM data for the tile.
        sliced_obs (xarray.Dataset): Sliced observational data for the tile.
        ff_future (xarray.Dataset): Fraction factors for future GCM data.
        bc_corrected_gcm_future_tile (xarray.Dataset): Bias-corrected GCM data for the future period.
        config (module): Configuration object for bias correction parameters.

    Returns:
        xarray.Dataset: Bias-corrected future six-hourly GCM data.
    """

    """Process a single tile for bias correction."""
    # lat_range = (tile["lat_min"], tile["lat_max"])
    # lon_range = (tile["lon_min"], tile["lon_max"])

    # # ------------------ Load GCM historical Data ------------------
    # sliced_gcm = xr.Dataset()
    # for var_name, file_paths in file_paths_by_variable_gcm.items():
    #     data_var = load_preprocess_variable(
    #         file_paths,
    #         var_name,
    #         level,
    #         lat_range,
    #         lon_range,
    #         config.startyear_h,
    #         config.endyear_h,
    #     )

    #     # Check if the variable is one of the wind components with different lon
    #     if var_name in ["ua", "va"]:
    #         # Let's assume hus and ta have the target longitude values, and they are already loaded
    #         target_lon = sliced_gcm.lon if "lon" in sliced_gcm else data_var.lon
    #         target_lat = sliced_gcm.lat if "lat" in sliced_gcm else data_var.lat
    #         target_lev = sliced_gcm.lev if "lev" in sliced_gcm else data_var.lev

    #         # Interpolate va to match the target latitude grid
    #         if not data_var.lat.equals(target_lat):
    #             data_var = data_var.interp(
    #                 lat=target_lat,
    #                 method="linear",
    #                 kwargs={"fill_value": "extrapolate"},
    #             )
    #         if not data_var.lon.equals(target_lon):
    #             data_var = data_var.interp(
    #                 lon=target_lon,
    #                 method="linear",
    #                 kwargs={"fill_value": "extrapolate"},
    #             )
    #         # Assign the adjusted longitude values to ua or va
    #         data_var = data_var.assign_coords(
    #             lon=target_lon, lat=target_lat, lev=target_lev
    #         )
    #     if isinstance(data_var, xr.Dataset):  # Ensure we extract the correct DataArray
    #         data_var = data_var[var_name]
    #     sliced_gcm[var_name] = data_var

    if config.bc_boundary == "lateral":
        assign_gcm = assign_w_6hr(config, sliced_gcm, config.bc_boundary)
        _, ff_gcm = convert_to_daily_with_fraction(config, assign_gcm)

    # # ------------------ Load Observational Data ------------------
    # sliced_obs = xr.Dataset()

    # for var_name in variables:
    #     # Construct the path to the preprocessed file for this variable
    #     obs_file = os.path.join(
    #         file_paths_by_variable_obs,
    #         f"preprocessed_obs_{var_name}_lev_{level}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc",
    #     )
    #     obs_var = xr.open_dataset(obs_file)[var_name]  # Load the variable from the file
    #     sliced_obs[var_name] = obs_var  # Add it to the observational dataset

    sliced_obs = sliced_obs.sel(
        lat=slice(sliced_gcm.lat.min().item(), sliced_gcm.lat.max().item()),
        lon=slice(sliced_gcm.lon.min().item(), sliced_gcm.lon.max().item()),
    )

    assign_obs = assign_w_6hr(config, sliced_obs, config.bc_boundary)
    _, ff_obs = convert_to_daily_with_fraction(config, assign_obs)

    ff_obs = ff_obs.compute()
    ff_gcm = ff_gcm.compute()
    ff_future = ff_future.compute()

    ff_future_corrected = quantile_mapping_rescale_future(
        config, ff_obs, ff_gcm, ff_future
    )

    six_hourly_data_future = daily_to_6hourly_xr(
        config,
        bc_corrected_gcm_future_tile,
        ff_future_corrected,
        var_list_w,
        config.startyear_f,
        config.endyear_f,
    )

    if (
        "lev" in six_hourly_data_future.coords
        and "lev" not in six_hourly_data_future.dims
    ):
        six_hourly_data_future = six_hourly_data_future.drop_vars(
            "lev"
        )  # Remove as a coordinate
        six_hourly_data_future = six_hourly_data_future.expand_dims(
            lev=[ff_gcm.lev.values]
        )  # Add as a dimension
    return six_hourly_data_future


def dict_to_simplenamespace(config, d):
    return SimpleNamespace(**d)


def load_bc_params_subset(
    config, file_path, lat_values, lon_values, level, lat_min, lat_max, lon_min, lon_max
):
    """
    Load only the required subset of bc_params from a large npy file, mapped to the correct tile's grid cell indices.

    Args:
        file_path (str): Path to the `.npy` file.
        lat_values (np.ndarray): Global latitude grid corresponding to bc_params indices.
        lon_values (np.ndarray): Global longitude grid corresponding to bc_params indices.
        level (int): Vertical level being processed.
        lat_min (float): Minimum latitude for the tile.
        lat_max (float): Maximum latitude for the tile.
        lon_min (float): Minimum longitude for the tile.
        lon_max (float): Maximum longitude for the tile.

    Returns:
        np.ndarray: Subset of bc_params with selected lat/lon indices.
    """

    # Load the full file (necessary since dtype=object)
    bc_params_array = np.load(
        f"{file_path}/bc_params_3d_historical_lev_{level}_{config.gname}_to_{config.input_model}_{config.startyear_h}_{config.endyear_h}.npy",
        allow_pickle=True,
    )  # Remove mmap_mode

    # Convert lat/lon values to corresponding grid indices
    lat_min_idx = max(0, np.searchsorted(lat_values, lat_min, side="left"))
    lat_max_idx = min(
        len(lat_values), np.searchsorted(lat_values, lat_max, side="right")
    )

    lon_min_idx = max(0, np.searchsorted(lon_values, lon_min, side="left"))
    lon_max_idx = min(
        len(lon_values), np.searchsorted(lon_values, lon_max, side="right")
    )

    # Extract the correct subset based on computed indices
    bc_params_subset = bc_params_array[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]

    # Convert dictionaries to SimpleNamespace only if not already converted
    for i in range(bc_params_subset.shape[0]):
        for j in range(bc_params_subset.shape[1]):
            if isinstance(
                bc_params_subset[i, j], dict
            ):  # Convert only if it's a dictionary
                bc_params_subset[i, j] = SimpleNamespace(**bc_params_subset[i, j])

    return bc_params_subset


# def load_bc_params_subset(
#     file_path, lat_values, lon_values, level, lat_min, lat_max, lon_min, lon_max
# ):
#     """
#     Load only the required subset of bc_params from a large npy file, based on latitude and longitude range.

#     Args:
#         file_path (str): Path to the `.npy` file.
#         lat_values (np.ndarray): Latitude values corresponding to bc_params indices.
#         lon_values (np.ndarray): Longitude values corresponding to bc_params indices.
#         lat_min (float): Minimum latitude to select.
#         lat_max (float): Maximum latitude to select.
#         lon_min (float): Minimum longitude to select.
#         lon_max (float): Maximum longitude to select.

#     Returns:
#         np.ndarray: Subset of bc_params with selected lat/lon indices.
#     """

#     # Load the full file (necessary since dtype=object)
#     bc_params_array = np.load(
#         f"{file_path}/bc_params_3d_historical_lev_{level}_{config.gname}_to_{config.input_model}_{config.startyear_h}_{config.endyear_h}.npy",
#         allow_pickle=True,
#     )  # Remove mmap_mode

#     # Find indices for the latitude and longitude range
#     lat_indices = np.where((lat_values >= lat_min) & (lat_values <= lat_max))[0]
#     lon_indices = np.where((lonValues >= lon_min) & (lonValues <= lon_max))[0]

#     # Select only the needed subset
#     bc_params_subset = bc_params_array[np.ix_(lat_indices, lon_indices)]

#     # Convert dictionaries to SimpleNamespace for easy access
#     for i in range(bc_params_subset.shape[0]):
#         for j in range(bc_params_subset.shape[1]):
#             bc_params_subset[i, j] = SimpleNamespace(**bc_params_subset[i, j])

#     return bc_params_subset


def process_tile_future(
    config,
    tile,
    variables,
    level,
    config,
    gcm_hist,
    obs_hist,
    temp_dir,
    idx,
):
    """
    Process a single tile for bias correction for future data.

    Args:
        tile (dict): Dictionary defining the spatial bounds of the tile.
        variables (list): List of variable names to process.
        level (int): Processing level for multilevel data.
        config (module): Configuration object for bias correction parameters.
        gcm_hist (xarray.Dataset): Historical GCM data.
        obs_hist (xarray.Dataset): Historical observational data.

    Returns:
        xarray.Dataset: Bias-corrected future GCM data.
    """

    """Process a single tile for bias correction."""

    # =============== Load GCM ===============
    sliced_gcm_future = load_and_combine_variables(
        config,
        tile,
        variables,
        level,
        config.startyear_f,
        config.endyear_f,
        data_type=config.period_f,
    )
    sliced_gcm = sliced_gcm_future.sel(
        time=slice(f"{config.startyear_f}-01-01", f"{config.endyear_f}-12-31")
    )
    assign_gcm = assign_w_6hr(config, sliced_gcm, config.bc_boundary)
    daily_gcm, ff_future = convert_to_daily_with_fraction(config, assign_gcm)

    reshaped_gcm_delayed = extract_and_reshape_delayed(
        config,
        daily_gcm,
        config.no_of_variables,
        config.startyear_f,
        config.endyear_f,
        config.bc_boundary,
    )
    # bc_params_array_loaded = load_bc_params_subset(
    #     config.out_path,
    #     gcm.lat.values,
    #     gcm.lon.values,
    #     level,
    #     tile["lat_min"],
    #     tile["lat_max"],
    #     tile["lon_min"],
    #     tile["lon_max"],
    # )
    # bc_params_array_loaded = np.load(
    #     f"{config.out_path}/bc_params_3d_{config.period}_lev_{level}_{config.gname}_to_{config.input_model}_{config.startyear_h}_{config.endyear_h}.npy",
    #     allow_pickle=True,
    # )
    # bc_params_array_loaded = np.load(
    #     f"{temp_dir}/bc_params_tile_3d_{config.period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.npy",
    #     allow_pickle=True,
    # )
    output_params = f"{temp_dir}/bc_params_tile_3d_{config.period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
    bc_params_array_loaded = xr.load_dataset(output_params)

    # # Convert dictionaries to SimpleNamespace only if not already converted
    # for i in range(bc_params_array_loaded.shape[0]):
    #     for j in range(bc_params_array_loaded.shape[1]):
    #         if isinstance(
    #             bc_params_array_loaded[i, j], dict
    #         ):  # Convert only if it's a dictionary
    #             bc_params_array_loaded[i, j] = SimpleNamespace(
    #                 **bc_params_array_loaded[i, j]
    #             )

    # Perform bias correction across the tile
    # bc_corrected_gcm_future_tile = bc_correction_grid_cell_future_dask(
    #     reshaped_gcm_delayed,
    #     bc_params_array_loaded,
    #     var_list_w,
    # )
    bc_corrected_gcm_future_tile = bc_correction_grid_cell_future_multiprocess(
        config,
        reshaped_gcm_delayed,
        bc_params_array_loaded,
        var_list_w,
    )

    if config.sub_daily_correction:
        bc_corrected_gcm_future = future_subdaily_correction(
            config,
            gcm_hist,
            obs_hist,
            ff_future,
            bc_corrected_gcm_future_tile,
            config,
        )
    else:
        bc_corrected_gcm_future = daily_to_6hourly_xr(
            config,
            bc_corrected_gcm_future_tile,
            ff_future,
            var_list_w,
            config.startyear_f,
            config.endyear_f,
        )
    # print(bc_corrected_gcm_future)
    bc_corrected_6hourly_data = apply_boundary_correction(
        config, bc_corrected_gcm_future, sliced_gcm
    )

    return bc_corrected_6hourly_data


def bc_correction_grid_cell_future_dask(
    config,
    gcm_future,
    bc_params_array,
    var_list_w,
):
    """
    Perform daily bias correction for future GCM data across all grid cells in parallel using Dask.

    Args:
        ff_gcm, ff_obs: Fraction factors for GCM and observational data.
        gcm_future (xarray.Dataset): GCM data for future period.
        ff_gcm_future: Fraction factors for future GCM data.
        bc_params_array: Bias correction parameters.
        startyear_f (int): Start year of the future period.
        endyear_f (int): End year of the future period.
        var_list_w (list of str): List of variable names to be corrected.
        sliced_gcm_future (xarray.Dataset): Sliced GCM data.

    Returns:
        xarray.Dataset: Bias-corrected daily GCM data for the future.
    """

    # grid_cells = list(itertools.product(gcm_future.lat.values, gcm_future.lon.values))
    # batch_size = 20  # Process 20 grid cells at a time
    # tasks = [
    #     dask.delayed(process_batch_of_grid_cells_future)(
    #         grid_cells[i : i + batch_size], gcm_future, bc_params_array
    #     )
    #     for i in range(0, len(grid_cells), batch_size)
    # ]
    # # Generate tasks for each grid cell
    tasks = [
        process_grid_cell_future(config, lat, lon, gcm_future, bc_params_array)
        for lat, lon in itertools.product(gcm_future.lat.values, gcm_future.lon.values)
    ]

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
    for result in results:
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

    # six_hourly_data = rescale_and_reformat_future(
    #     gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future
    # )

    # # g_u = sliced_gcm_future.ua.sel(time=slice(str(startyear_f), str(endyear_f)))
    # # g_v = sliced_gcm_future.va.sel(time=slice(str(startyear_f), str(endyear_f)))
    # # six_hourly_data = six_hourly_data.sel(time=slice(str(startyear_f), str(endyear_f)))

    # bc_corrected_6hourly_data = apply_boundary_correction(
    #     six_hourly_data, sliced_gcm_future
    # )

    # # Compute all delayed tasks in parallel
    # final_corrected_data = dask.compute(bc_corrected_6hourly_data)

    return gcmc_corrected


def preprocess_and_save_gcm(
    config,
    tile,
    file_paths_by_variable_gcm,
    temp_dir,
    level,
):
    """
    Preprocess and save GCM data for each tile.

    Args:
        tile (dict): Dictionary specifying the spatial bounds of the tile.
        file_paths_by_variable_gcm (dict): File paths for GCM data by variable.
        temp_dir (str): Directory to save preprocessed data.
        var_name (str): Name of the variable.
        level (int): Processing level for multilevel data.

    Returns:
        str: Path to the saved preprocessed data file.
    """
    lat_range = (tile[0]["lat_min"], tile[0]["lat_max"])
    lon_range = (tile[0]["lon_min"], tile[0]["lon_max"])

    temp_file = os.path.join(
        temp_dir,
        f"preprocessed_{config.gname}_lev_{level}_{tile[0]['lat_min']}_{tile[0]['lat_max']}_{tile[0]['lon_min']}_{tile[0]['lon_max']}.nc",
    )
    if not os.path.exists(temp_file):
        sliced_gcm = xr.Dataset()
        for var_name, file_paths in file_paths_by_variable_gcm.items():
            data_var = load_preprocess_variable(
                config,
                file_paths,
                var_name,
                level,
                lat_range,
                lon_range,
                config.startyear_h,
                config.endyear_h,
            )

            # Check if the variable is one of the wind components with different lon
            if var_name in ["ua", "va"]:
                # Let's assume hus and ta have the target longitude values, and they are already loaded
                target_lon = sliced_gcm.lon if "lon" in sliced_gcm else data_var.lon
                target_lat = sliced_gcm.lat if "lat" in sliced_gcm else data_var.lat
                target_lev = sliced_gcm.lev if "lev" in sliced_gcm else data_var.lev

                # Interpolate va to match the target latitude grid
                if not data_var.lat.equals(target_lat):
                    data_var = data_var.interp(
                        lat=target_lat,
                        method="linear",
                        kwargs={"fill_value": "extrapolate"},
                    )
                if not data_var.lon.equals(target_lon):
                    data_var = data_var.interp(
                        lon=target_lon,
                        method="linear",
                        kwargs={"fill_value": "extrapolate"},
                    )
                # Assign the adjusted longitude values to ua or va
                data_var = data_var.assign_coords(
                    lon=target_lon, lat=target_lat, lev=target_lev
                )
            if isinstance(
                data_var, xr.Dataset
            ):  # Ensure we extract the correct DataArray
                data_var = data_var[var_name]
            sliced_gcm[var_name] = data_var

        sliced_gcm_sel = sliced_gcm.astype(np.float32).persist()
        sliced_gcm_sel.to_netcdf(temp_file)
    else:
        print(f"File {temp_file} already exists. Skipping.")

    return temp_file


def preprocess_and_save_obs(
    config,
    tile,
    file_paths,
    temp_dir,
    var_name,
    level_index,
    startyear_h,
    endyear_h,
):
    """
    Preprocess and save observational data for each tile.

    Args:
        tile (dict): Dictionary specifying the spatial bounds of the tile.
        file_paths (list): Paths to observation files.
        temp_dir (str): Directory to save preprocessed data.
        var_name (str): Name of the variable.
        level_index (int): Index for specific levels in the data.
        lat_min, lat_max, lon_min, lon_max (float): Spatial extent for subsetting.
        startyear_h, endyear_h (int): Temporal range for subsetting.

    Returns:
        str: Path to the saved preprocessed data file.
    """

    # Create the temporary folder if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    lat_range = (tile[0]["lat_min"], tile[0]["lat_max"])
    lon_range = (tile[0]["lon_min"], tile[0]["lon_max"])
    # Save the preprocessed data to a temporary file
    temp_file = os.path.join(
        temp_dir,
        f"preprocessed_obs_{var_name}_lev_{level_index}_{tile[0]['lat_min']}_{tile[0]['lat_max']}_{tile[0]['lon_min']}_{tile[0]['lon_max']}_{startyear_h}_{endyear_h}.nc",
    )

    if not os.path.exists(temp_file):
        # Preprocess the observational data
        obs_ds = xr.open_mfdataset(
            file_paths,
            combine="by_coords",
            chunks={"time": "auto", "lat": "auto", "lon": "auto"},
        )
        # Apply nearest logic to longitude
        # lon_values = obs_ds.lon.values
        # lat_values = obs_ds.lat.values
        # nearest_lat_min = lat_values[np.abs(lat_values - lat_range[0]).argmin()]
        # nearest_lat_max = lat_values[np.abs(latValues - lat_range[1]).argmin()]
        # nearest_lon_min = lon_values[np.abs(lon_values - lon_range[0]).argmin()]
        # nearest_lon_max = lon_values[np.abs(lon_values - lon_range[1]).argmin()]

        # obs_ds_sel = (
        #     obs_ds[var_name]
        #     .isel(lev=level_index)
        #     .sel(
        #         lat=slice(nearest_lat_min, nearest_lat_max),
        #         lon=slice(nearest_lon_min, nearest_lon_max),
        #         time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31"),
        #     )
        # )
        obs_ds_sel = (
            obs_ds[var_name]
            .isel(lev=level_index)
            .sel(
                lat=slice(*lat_range),
                lon=slice(*lon_range),
                time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31"),
            )
        )
        obs_ds_sel = obs_ds_sel.astype("float32").persist()
        obs_ds_sel.to_netcdf(temp_file)
    else:
        print(f"File {temp_file} already exists. Skipping.")

    return temp_file


def preprocess_dataset(
    config,
    ds,
    var_name,
    level_index,
    lat_range,
    lon_range,
    startyear_h,
    endyear_h,
    bc_boundary,
):
    """
    Preprocess each dataset by selecting relevant variables, levels, and ranges.

    Args:
        ds (xarray.Dataset): Input dataset.
        var_name (str): Name of the variable to extract.
        level_index (int): Index of the level to select (if applicable).
        lat_range (tuple): Latitude range to subset (min, max).
        lon_range (tuple): Longitude range to subset (min, max).
        startyear_h (int): Start year for time range.
        endyear_h (int): End year for time range.
        bc_boundary (str): Boundary type ('lateral' or otherwise).

    Returns:
        xarray.Dataset: Preprocessed dataset.
    """
    if bc_boundary == "lateral":
        return (
            ds[var_name]
            .isel(lev=level_index)
            .sel(
                lat=slice(*lat_range),
                lon=slice(*lon_range),
                time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31"),
            )
        )
    else:
        return ds[var_name].sel(
            lat=slice(*lat_range),
            lon=slice(*lon_range),
            time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31"),
        )


def load_preprocess_variable(
    config,
    file_paths,
    var_name,
    level_index,
    lat_range,
    lon_range,
    startyear_h,
    endyear_h,
):
    """
    Load and preprocess a variable from multiple files.

    Args:
        file_paths (list of str): Paths to the data files.
        var_name (str): Name of the variable.
        level_index (int): Index for selecting specific levels.
        lat_range (tuple): Latitude range to subset.
        lon_range (tuple): Longitude range to subset.
        startyear_h (int): Start year for the time range.
        endyear_h (int): End year for the time range.

    Returns:
        xarray.DataArray: Preprocessed dataset for the variable.
    """

    ds_sel = xr.open_mfdataset(
        file_paths,
        combine="by_coords",
        chunks={"time": 1000, "lat": "auto", "lon": "auto"},
        preprocess=lambda ds: preprocess_dataset(
            config,
            ds,
            var_name,
            level_index,
            lat_range,
            lon_range,
            startyear_h,
            endyear_h,
            config.bc_boundary,
        ),
    )

    return ds_sel


def correction_wrapper_pool(args):
    """
    Wrapper function to apply bias correction for a single (lat, lon) grid cell.
    Runs in parallel using multiprocessing.
    """
    config, lat, lon, reshaped_gcm, reshaped_obs = args
    # Print the process ID (PID) to check parallel execution
    # print(f"Processing lat: {lat}, lon: {lon} on process ID: {os.getpid()}")

    # Extract GCM data for this (lat, lon)
    gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    obs_data = [reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w]

    # Convert GCM data to NumPy (Fortran needs NumPy)
    gcm_data_np = np.stack(gcm_data, axis=0).astype(np.float32)
    obs_data_np = np.stack(obs_data, axis=0).astype(np.float32)

    # Apply bias correction using Fortran function
    result_dict = bc_correction_hist(config, gcm_data_np, obs_data_np)

    return {
        "lat": lat,
        "lon": lon,
        "gcmc_corrected": result_dict["gcmc"],
        "bc_params": result_dict["bc_params"],
    }


def bc_correction_grid_cell_multiprocess(config, gcm, obs, variable):
    """
    Apply bias correction for future data using multiprocessing (no tiling required).
    """
    # Extract lat/lon values
    lat_values = gcm.lat.values
    lon_values = gcm.lon.values

    # Prepare arguments for multiprocessing (list of tuples)
    args_list = [
        (config, lat, lon, gcm, obs) for lat in lat_values for lon in lon_values
    ]

    # Use multiprocessing Pool
    num_workers = cpu_count()  # Get number of CPU cores
    n = num_workers
    # n = min(len(os.sched_getaffinity(0)), num_workers, 96)
    print(f"Using {n} CPU cores for parallel processing...")

    with Pool(processes=n) as pool:
        results = pool.map(correction_wrapper_pool, args_list)

    # Convert results back to xarray.Dataset
    year = config.endyear_h - config.startyear_h + 1
    corrected_data = {
        var: np.empty((year, 12, 31, len(lat_values), len(lon_values)))
        for var in variable
    }
    bc_params_array = np.empty((len(gcm.lat), len(gcm.lon)), dtype=object)
    # Fill the arrays
    for result in results:
        lat_idx = np.where(lat_values == result["lat"])[0][0]
        lon_idx = np.where(lon_values == result["lon"])[0][0]
        for i, var in enumerate(variable):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]
        bc_params_array[lat_idx, lon_idx] = result["bc_params"].to_dict()

    # Convert to xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in variable
        },
        coords={
            "year": gcm.year,
            "month": gcm.month,
            "day": gcm.day,
            "lat": gcm.lat,
            "lon": gcm.lon,
        },
    )
    if len(variable) == 1:
        daily_data_aligned = align_daily_data_xr(
            config, gcmc_corrected, config.startyear_h
        )
        return daily_data_aligned
    else:
        return gcmc_corrected, bc_params_array


def bc_correction_grid_cell_hist_dask(
    config,
    reshaped_gcm,
    reshaped_obs,
    ff_gcm,
    ff_obs,
    var_list_w,
    sliced_gcm,
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

    Returns:
    - ds_corrected: Xarray Dataset with corrected data and bias correction parameters
    """

    # # # Generate tasks for each grid cell
    # tasks = [
    #     dask.delayed(process_grid_cell)(lat, lon, reshaped_gcm, reshaped_obs)
    #     for lat, lon in itertools.product(
    #         reshaped_gcm.lat.values, reshaped_gcm.lon.values
    #     )
    # ]

    # # grid_cells = list(
    # #     itertools.product(reshaped_gcm.lat.values, reshaped_gcm.lon.values)
    # # )
    # # batch_size = 1  # Process 20 grid cells at a time
    # # tasks = [
    # #     dask.delayed(process_batch_of_grid_cells)(
    # #         grid_cells[i : i + batch_size], reshaped_gcm, reshaped_obs
    # #     )
    # #     for i in range(0, len(grid_cells), batch_size)
    # # ]

    # # Compute all tasks in parallel at the end
    # results = dask.compute(*tasks)

    # # Flatten the list of results
    # flattened_results = [item for sublist in results for item in sublist]

    # # Initialize arrays to hold the final data
    # corrected_data = {
    #     var: np.empty((31, 12, 31, len(reshaped_gcm.lat), len(reshaped_gcm.lon)))
    #     for var in var_list_w
    # }
    # bc_params_array = np.empty(
    #     (len(reshaped_gcm.lat), len(reshaped_gcm.lon)), dtype=object
    # )

    # # Fill the arrays with data from results
    # for result in results:
    #     lat_idx = np.where(reshaped_gcm.lat.values == result["lat"])[0][0]
    #     lon_idx = np.where(reshaped_gcm.lon.values == result["lon"])[0][0]
    #     for i, var in enumerate(var_list_w):
    #         corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]
    #     bc_params_array[lat_idx, lon_idx] = result["bc_params"].to_dict()

    # # Convert to Xarray Dataset
    # gcmc_corrected = xr.Dataset(
    #     {
    #         var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
    #         for var in var_list_w
    #     },
    #     coords={
    #         "year": reshaped_gcm.year,
    #         "month": reshaped_gcm.month,
    #         "day": reshaped_gcm.day,
    #         "lat": reshaped_gcm.lat,
    #         "lon": reshaped_gcm.lon,
    #     },
    # )
    gcmc_corrected, bc_params_array = bc_correction_grid_cell_multiprocess(
        config, reshaped_gcm, reshaped_obs, var_list_w
    )
    six_hourly_data_hist = rescale_and_reformat(config, gcmc_corrected, ff_gcm, ff_obs)

    bc_corrected_6hourly_data = apply_boundary_correction(
        config, six_hourly_data_hist, sliced_gcm
    )

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(bc_corrected_6hourly_data, bc_params_array)

    return final_corrected_data[0], final_corrected_data[1]


def bc_correction_grid_cell_hist_dask_2d(
    config,
    reshaped_gcm,
    reshaped_obs,
    var_list_w,
    startyear_h,
):
    """
    Perform bias correction for historical GCM data across all grid cells in parallel using Dask.

    Args:
        reshaped_gcm (xarray.Dataset): Reshaped GCM data.
        reshaped_obs (xarray.Dataset): Reshaped observational data.
        var_list_w (list of str): List of variables to be corrected.
        startyear_h (int): Start year for historical correction.

    Returns:
        tuple: Corrected GCM data and bias correction parameters.
    """

    # Generate tasks for each grid cell
    tasks = [
        dask.delayed(process_grid_cell)(config, lat, lon, reshaped_gcm, reshaped_obs)
        for lat, lon in itertools.product(
            reshaped_gcm.lat.values, reshaped_gcm.lon.values
        )
    ]

    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)

    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty((31, 12, 31, len(reshaped_gcm.lat), len(reshaped_gcm.lon)))
        for var in var_list_w
    }
    bc_params_array = np.empty(
        (len(reshaped_gcm.lat), len(reshaped_gcm.lon)), dtype=object
    )

    # Fill the arrays with data from results
    for result in results:
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
    daily_data_aligned = align_daily_data_xr(config, gcmc_corrected, startyear_h)

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(daily_data_aligned, bc_params_array)

    return final_corrected_data[0], final_corrected_data[1]


# def bc_correction_grid_cell_future_subdaily_dask(
#     ff_gcm,
#     ff_obs,
#     gcm_future,
#     ff_gcm_future,
#     bc_params_array,
#     startyear_f,
#     endyear_f,
#     var_list_w,
#     sliced_gcm_future,
# ):
#     """
#     Perform sub-daily bias correction for future GCM data across all grid cells in parallel using Dask.

#     Args:
#         ff_gcm, ff_obs: Fraction factors for GCM and observational data.
#         gcm_future (xarray.Dataset): GCM data for future period.
#         ff_gcm_future: Fraction factors for future GCM data.
#         bc_params_array: Bias correction parameters.
#         startyear_f (int): Start year of the future period.
#         endyear_f (int): End year of the future period.
#         var_list_w (list of str): List of variable names to be corrected.
#         sliced_gcm_future (xarray.Dataset): Sliced GCM data.

#     Returns:
#         xarray.Dataset: Bias-corrected six-hourly GCM data for the future.
#     """

#     # Generate tasks for each grid cell
#     tasks = [
#         process_grid_cell_future(lat, lon, gcm_future, bc_params_array)
#         for lat, lon in itertools.product(gcm_future.lat.values, gcm_future.lon.values)
#     ]

#     # Compute all tasks in parallel at the end
#     results = dask.compute(*tasks)

#     # Initialize arrays to hold the final data
#     corrected_data = {
#         var: np.empty((31, 12, 31, len(gcm_future.lat), len(gcm_future.lon)))
#         for var in var_list_w
#     }

#     # Flatten the list of results
#     flattened_results = [item for sublist in results for item in sublist]

#     # Fill the arrays with data from results
#     for result in flattened_results:
#         lat_idx = np.where(gcm_future.lat.values == result["lat"])[0][0]
#         lon_idx = np.where(gcm_future.lon.values == result["lon"])[0][0]
#         for i, var in enumerate(var_list_w):
#             corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

#     # Convert to Xarray Dataset
#     gcmc_corrected = xr.Dataset(
#         {
#             var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
#             for var in var_list_w
#         },
#         coords={
#             "year": gcm_future.year,
#             "month": gcm_future.month,
#             "day": gcm_future.day,
#             "lat": gcm_future.lat,
#             "lon": gcm_future.lon,
#         },
#     )

#     six_hourly_data = rescale_and_reformat_future(
#         gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future
#     )

#     # g_u = sliced_gcm_future.ua.sel(time=slice(str(startyear_f), str(endyear_f)))
#     # g_v = sliced_gcm_future.va.sel(time=slice(str(startyear_f), str(endyear_f)))
#     # six_hourly_data = six_hourly_data.sel(time=slice(str(startyear_f), str(endyear_f)))

#     bc_corrected_6hourly_data = apply_boundary_correction(
#         six_hourly_data, sliced_gcm_future
#     )

#     # Compute all delayed tasks in parallel
#     final_corrected_data = dask.compute(bc_corrected_6hourly_data)

#     return final_corrected_data[0]


def correction_wrapper_future_pool(args):
    """
    Wrapper function to apply bias correction for a single (lat, lon) grid cell.
    Runs in parallel using multiprocessing.
    """
    config, lat, lon, reshaped_gcm, bc_params_array = args
    # Print the process ID (PID) to check parallel execution
    # print(f"Processing lat: {lat}, lon: {lon} on process ID: {os.getpid()}")

    # Extract GCM data for this (lat, lon)
    gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]

    # # Find nearest indices
    # lat_array = reshaped_gcm.lat.values
    # lon_array = reshaped_gcm.lon.values

    # lat_idx = (np.abs(lat_array - lat)).argmin()
    # lon_idx = (np.abs(lon_array - lon)).argmin()

    # Select bias correction parameters
    # params_data = bc_params_array[lat_idx, lon_idx]
    # ds_cell = bc_params_array.sel(lat=lat, lon=lon, method="nearest")
    ds_cell = bc_params_array.sel(lat=lat, lon=lon)

    # For each variable, reshape the flattened data back to its original shape.
    params = {}
    for var in ds_cell.data_vars:
        # The flattened data is stored in a variable with a unique flattened dimension.
        flat_data = ds_cell[var].values
        # Retrieve the original shape from the variable's attributes.
        original_shape = ds_cell[var].attrs.get("original_shape", None)
        if original_shape is None:
            raise ValueError(f"Original shape not found for variable '{var}'.")
        # Reshape the 1D (flattened) data back to its original shape.
        reshaped_data = flat_data.reshape(original_shape)
        params[var] = reshaped_data

    # Convert the dictionary to a SimpleNamespace so that we can access attributes like params_data.avdc_iter
    params_data = SimpleNamespace(**params)

    # Convert GCM dat a to NumPy (Fortran needs NumPy)
    gcm_data_np = np.stack(gcm_data, axis=0).astype(np.float32)

    # Apply bias correction using Fortran function
    result_dict = bc_correction_future(config, gcm_data_np, params_data)

    return {"lat": lat, "lon": lon, "gcmc_corrected": result_dict}


def bc_correction_grid_cell_future_multiprocess(
    config, gcm_future, bc_params_array, variable
):
    """
    Apply bias correction for future data using multiprocessing (no tiling required).
    """
    # Extract lat/lon values
    lat_values = gcm_future.lat.values
    lon_values = gcm_future.lon.values

    # Prepare arguments for multiprocessing (list of tuples)
    args_list = [
        (config, lat, lon, gcm_future, bc_params_array)
        for lat in lat_values
        for lon in lon_values
    ]

    # Use multiprocessing Pool
    num_workers = cpu_count()  # Get number of CPU cores
    n = num_workers
    n = min(len(os.sched_getaffinity(0)), num_workers, 96)
    print(f"Using {n} CPU cores for parallel processing...")

    with Pool(processes=n) as pool:
        results = pool.map(correction_wrapper_future_pool, args_list)

    # Convert results back to xarray.Dataset
    corrected_data = {
        var: np.empty((31, 12, 31, len(lat_values), len(lon_values)))
        for var in variable
    }

    # Fill the arrays
    for result in results:
        lat_idx = np.where(lat_values == result["lat"])[0][0]
        lon_idx = np.where(lon_values == result["lon"])[0][0]
        for i, var in enumerate(variable):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

    # Convert to xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in variable
        },
        coords={
            "year": gcm_future.year,
            "month": gcm_future.month,
            "day": gcm_future.day,
            "lat": gcm_future.lat,
            "lon": gcm_future.lon,
        },
    )
    if len(variable) == 1:
        daily_data_aligned = align_daily_data_xr(
            config, gcmc_corrected, config.startyear_f
        )
        return daily_data_aligned
    else:
        return gcmc_corrected


# def bc_correction_grid_cell_future_daily_dask(
#     ff_gcm,
#     ff_obs,
#     gcm_future,
#     ff_gcm_future,
#     bc_params_array,
#     startyear_f,
#     endyear_f,
#     var_list_w,
#     sliced_gcm_future,
# ):
#     """
#     Perform daily bias correction for future GCM data across all grid cells in parallel using Dask.

#     Args:
#         ff_gcm, ff_obs: Fraction factors for GCM and observational data.
#         gcm_future (xarray.Dataset): GCM data for future period.
#         ff_gcm_future: Fraction factors for future GCM data.
#         bc_params_array: Bias correction parameters.
#         startyear_f (int): Start year of the future period.
#         endyear_f (int): End year of the future period.
#         var_list_w (list of str): List of variable names to be corrected.
#         sliced_gcm_future (xarray.Dataset): Sliced GCM data.

#     Returns:
#         xarray.Dataset: Bias-corrected daily GCM data for the future.
#     """

#     # grid_cells = list(itertools.product(gcm_future.lat.values, gcm_future.lon.values))
#     # batch_size = 20  # Process 20 grid cells at a time
#     # tasks = [
#     #     dask.delayed(process_batch_of_grid_cells_future)(
#     #         grid_cells[i : i + batch_size], gcm_future, bc_params_array
#     #     )
#     #     for i in range(0, len(grid_cells), batch_size)
#     # ]
#     # # Generate tasks for each grid cell
#     tasks = [
#         process_grid_cell_future(lat, lon, gcm_future, bc_params_array)
#         for lat, lon in itertools.product(gcm_future.lat.values, gcm_future.lon.values)
#     ]

#     # Compute all tasks in parallel at the end
#     results = dask.compute(*tasks)

#     # Flatten the list of results
#     flattened_results = [item for sublist in results for item in sublist]

#     # Initialize arrays to hold the final data
#     corrected_data = {
#         var: np.empty((31, 12, 31, len(gcm_future.lat), len(gcm_future.lon)))
#         for var in var_list_w
#     }

#     # Fill the arrays with data from results
#     for result in results:
#         lat_idx = np.where(gcm_future.lat.values == result["lat"])[0][0]
#         lon_idx = np.where(gcm_future.lon.values == result["lon"])[0][0]
#         for i, var in enumerate(var_list_w):
#             corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

#     # Convert to Xarray Dataset
#     gcmc_corrected = xr.Dataset(
#         {
#             var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
#             for var in var_list_w
#         },
#         coords={
#             "year": gcm_future.year,
#             "month": gcm_future.month,
#             "day": gcm_future.day,
#             "lat": gcm_future.lat,
#             "lon": gcm_future.lon,
#         },
#     )

#     six_hourly_data = rescale_and_reformat_future(
#         gcmc_corrected, ff_gcm, ff_obs, ff_gcm_future
#     )

#     # g_u = sliced_gcm_future.ua.sel(time=slice(str(startyear_f), str(endyear_f)))
#     # g_v = sliced_gcm_future.va.sel(time=slice(str(startyear_f), str(endyear_f)))
#     # six_hourly_data = six_hourly_data.sel(time=slice(str(startyear_f), str(endyear_f)))

#     bc_corrected_6hourly_data = apply_boundary_correction(
#         six_hourly_data, sliced_gcm_future
#     )

#     # Compute all delayed tasks in parallel
#     final_corrected_data = dask.compute(bc_corrected_6hourly_data)

#     return final_corrected_data[0]


def bc_correction_grid_cell_future_dask_2d(
    config,
    gcm_future,
    bc_params_array,
    startyear_f,
    endyear_f,
    variable,
):

    tasks = [
        dask.delayed(process_grid_cell_future)(
            config, lat, lon, gcm_future, bc_params_array
        )
        for lat, lon in itertools.product(gcm_future.lat.values, gcm_future.lon.values)
    ]
    print("compute tasks")
    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)
    print("tasks done")
    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty((31, 12, 31, len(gcm_future.lat), len(gcm_future.lon)))
        for var in variable
    }

    # Fill the arrays with data from results
    for result in results:
        lat_idx = np.where(gcm_future.lat.values == result["lat"])[0][0]
        lon_idx = np.where(gcm_future.lon.values == result["lon"])[0][0]
        for i, var in enumerate(variable):
            corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

    # Convert to Xarray Dataset
    gcmc_corrected = xr.Dataset(
        {
            var: (["year", "month", "day", "lat", "lon"], corrected_data[var])
            for var in variable
        },
        coords={
            "year": gcm_future.year,
            "month": gcm_future.month,
            "day": gcm_future.day,
            "lat": gcm_future.lat,
            "lon": gcm_future.lon,
        },
    )
    print("align")
    # Align the daily data with the generated dates
    daily_data_aligned = align_daily_data_xr(config, gcmc_corrected, startyear_f)

    # Compute all delayed tasks in parallel
    final_corrected_data = dask.compute(daily_data_aligned)

    return final_corrected_data[0]


def is_leap_year(config, year):
    """
    Check if a given year is a leap year.

    Args:
        year (int): Year to be checked.

    Returns:
        bool: True if the year is a leap year, otherwise False.
    """

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def adjust_dates_to_target_year(config, corrected_data, target_year):
    """
    Adjust the dates of corrected data to align with a target year, accounting for leap years.

    Args:
        corrected_data (xarray.Dataset): Corrected dataset to adjust.
        target_year (int): Target year for the adjustment.

    Returns:
        xarray.Dataset: Dataset with adjusted dates.
    """

    # Create 'hourofyear' directly, excluding February 29 if necessary
    if is_leap_year(config, target_year):
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
    config,
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
    """
    Apply moving window bias correction over a given time frame for a grid cell.

    Args:
        lat (float): Latitude of the grid cell.
        lon (float): Longitude of the grid cell.
        gcm_future (xarray.Dataset): Future GCM data.
        gcm_ref (xarray.Dataset): Reference GCM data.
        fraction_factors (xarray.Dataset): Fraction factors for correction.
        bc_params_array (xarray.Dataset): Bias correction parameters.
        start_year (int): Start year for bias correction.
        end_year (int): End year for bias correction.
        window_size (int, optional): Size of the moving window for correction. Defaults to 30.

    Returns:
        xarray.Dataset: Bias-corrected dataset for the specified grid cell and time window.
    """

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
            corrected_data = bc_correction_grid_cell_future_daily_dask(
                config,
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
                config,
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


def convert_bc_params_to_xarray(config, bc_params_array, lat_values, lon_values):
    nlat, nlon = bc_params_array.shape

    # --- Extract Variable Keys and Pre-allocate Arrays ---
    # Find the variable keys from the first non-None dictionary in bc_params_array.
    var_keys = None
    for i in range(nlat):
        for j in range(nlon):
            if bc_params_array[i, j] is not None:
                var_keys = list(bc_params_array[i, j].keys())
                break
        if var_keys is not None:
            break

    if var_keys is None:
        raise ValueError("No valid dictionary found in bc_params_array.")

    # For each key, determine the flattened size from the first sample,
    # converting scalars to numpy arrays so that they have a shape.
    data_vars = {}
    original_shapes = {}  # to store each variable's original shape for later reshaping
    for key in var_keys:
        sample_val = bc_params_array[i, j][key]
        # Convert scalars (or objects without shape) to numpy arrays.
        if np.isscalar(sample_val) or not hasattr(sample_val, "shape"):
            sample_array = np.array([sample_val])
        else:
            sample_array = sample_val
        original_shapes[key] = sample_array.shape
        flat_len = sample_array.size
        data_vars[key] = np.empty((flat_len, nlat, nlon), dtype=sample_array.dtype)
        # print(f"{key}: original shape = {sample_array.shape}, flat length = {flat_len}")

    # --- Fill the Data Arrays ---
    # Loop through each grid cell, flatten the array from each dictionary, and store it.
    for i in range(nlat):
        for j in range(nlon):
            cell_dict = bc_params_array[i, j]
            if cell_dict is not None:
                for key in var_keys:
                    val = cell_dict[key]
                    if np.isscalar(val) or not hasattr(val, "shape"):
                        arr = np.array([val])
                    else:
                        arr = val
                    data_vars[key][:, i, j] = arr.flatten()
            else:
                for key in var_keys:
                    data_vars[key][:, i, j] = np.nan

    # --- Create xarray Dataset ---
    # Build an xarray Dataset with coordinates for lat and lon.
    ds = xr.Dataset(
        coords={"lat": (("lat",), lat_values), "lon": (("lon",), lon_values)}
    )

    # Add each variable to the Dataset with a unique flattened dimension for each variable.
    for key, arr in data_vars.items():
        flat_dim_name = f"{key}_flat"
        ds[key] = ((flat_dim_name, "lat", "lon"), arr)
        ds[key].attrs["original_shape"] = original_shapes[key]

    # (Optional) Save the Dataset to a NetCDF file instead of npy for more efficient I/O.
    # ds.to_netcdf("bc_params.nc")
    return ds
