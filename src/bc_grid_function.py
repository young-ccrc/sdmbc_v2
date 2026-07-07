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
import uuid, tempfile

import dask  # type: ignore
import dask.array as da  # type: ignore
from dask.distributed import Lock  # type: ignore

import numpy as np  # type: ignore
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
)

# Suppress INFO and lower-level logs
logging.getLogger("flox").setLevel(logging.WARNING)
logging.getLogger("xarray").setLevel(logging.WARNING)
logging.getLogger("dask").setLevel(logging.WARNING)


def _interp_to_point_eager(data_var, target_lat, target_lon):
    """
    Interpolate a small lat/lon stencil to a single point eagerly.

    This keeps the one-grid BC smoke test on the production intent for wind
    variables while avoiding xarray/scipy/dask edge cases on tiny chunks.
    """
    if "lat" not in data_var.dims or "lon" not in data_var.dims:
        return data_var

    lat_vals = np.asarray(data_var.lat.values, dtype=float)
    lon_vals = np.asarray(data_var.lon.values, dtype=float)
    if lat_vals.size == 0 or lon_vals.size == 0:
        raise ValueError("Cannot interpolate point from an empty lat/lon stencil.")

    target_lat_val = float(np.asarray(target_lat).squeeze())
    target_lon_val = float(np.asarray(target_lon).squeeze())

    lat_order = np.argsort(lat_vals)
    lon_order = np.argsort(lon_vals)
    lat_sorted = lat_vals[lat_order]
    lon_sorted = lon_vals[lon_order]

    da = data_var.transpose(..., "lat", "lon")
    values = np.asarray(da.values)
    lead_dims = da.dims[:-2]
    lead_shape = values.shape[:-2]
    flat = values.reshape((-1, values.shape[-2], values.shape[-1]))

    out = np.empty((flat.shape[0],), dtype=values.dtype)
    for idx, slice2d in enumerate(flat):
        slice_sorted = slice2d[np.ix_(lat_order, lon_order)]
        lon_interp = np.array(
            [np.interp(target_lon_val, lon_sorted, row) for row in slice_sorted]
        )
        out[idx] = np.interp(target_lat_val, lat_sorted, lon_interp)

    out = out.reshape(lead_shape + (1, 1))
    coords = {dim: da.coords[dim] for dim in lead_dims if dim in da.coords}
    coords["lat"] = [target_lat_val]
    coords["lon"] = [target_lon_val]
    return xr.DataArray(
        out, dims=lead_dims + ("lat", "lon"), coords=coords, attrs=da.attrs
    )


def _single_grid_wind_bounds(file_paths, target_lat, target_lon):
    """
    Return a small 3-point lat/lon stencil around the target point.

    This is only used for the single-grid BC smoke test on staggered wind
    variables so that the preprocessing step never degenerates to an empty
    slice.
    """
    sample_path = file_paths[0] if isinstance(file_paths, (list, tuple)) else file_paths
    with xr.open_dataset(sample_path) as sample:
        lat_vals = np.asarray(sample.lat.values, dtype=float)
        lon_vals = np.asarray(sample.lon.values, dtype=float)

    if lat_vals.size == 0 or lon_vals.size == 0:
        raise ValueError("Cannot determine wind stencil from empty coordinates.")

    def _stencil(vals, target):
        if vals.size <= 3:
            return float(vals.min()), float(vals.max())
        pos = int(np.searchsorted(vals, float(target)))
        start = max(0, min(pos - 1, vals.size - 3))
        stop = start + 3
        return float(vals[start]), float(vals[stop - 1])

    lat_min, lat_max = _stencil(lat_vals, target_lat)
    lon_min, lon_max = _stencil(lon_vals, target_lon)
    return (lat_min, lat_max), (lon_min, lon_max)


# Helper: robust coordinate-to-index lookup (avoids nearest overlap and float mismatch)
def _grid_index(coord_vals, target, tol=None):
    arr = np.asarray(coord_vals)
    idx = int(np.argmin(np.abs(arr - target)))
    if tol is None:
        return idx
    if abs(arr[idx] - target) <= tol:
        return idx
    raise KeyError(f"Target {target} not found within tolerance {tol}")


def _default_tol_from(da_or_ds):
    lat = da_or_ds.lat.values
    lon = da_or_ds.lon.values
    dy = float(np.min(np.abs(np.diff(lat)))) if lat.size > 1 else 0.0
    dx = float(np.min(np.abs(np.diff(lon)))) if lon.size > 1 else 0.0
    # strict tolerance: smaller than half-grid so two tiles can't claim the same edge
    base = min(dx if dx > 0 else 1.0, dy if dy > 0 else 1.0)
    return 0.25 * base


def _isel_point(obj, lat, lon, tol=None):
    tol = _default_tol_from(obj) if tol is None else tol
    lat_idx = _grid_index(obj.lat.values, lat, tol)
    lon_idx = _grid_index(obj.lon.values, lon, tol)
    return obj.isel(lat=lat_idx, lon=lon_idx)


def _atomic_to_netcdf(ds, final_path, engine="netcdf4"):
    """
    Single-writer, atomic NetCDF write guarded by a distributed lock.
    Loads ds to memory to avoid Dask's store graph during write.
    """
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    lock = Lock(name=f"nc-write::{final_path}")  # lock on final path
    tmp_path = os.path.join(os.path.dirname(final_path), f".{uuid.uuid4().hex}.tmp.nc")

    with lock:
        if os.path.exists(final_path):
            return final_path
        try:
            # Ensure no dask arrays remain; avoid dask.store() path
            if isinstance(ds, xr.Dataset):
                ds = ds.unify_chunks()  # safe even if not chunked
                ds = ds.load()
            else:
                # DataArray or other
                ds = ds.load()

            # Plain netCDF4 write (no compute=), then atomic rename
            ds.to_netcdf(tmp_path, engine=engine)
            os.replace(tmp_path, final_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    return final_path


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


def _materialize_hist_cell_inputs(reshaped_gcm, reshaped_obs, var_names):
    """Load one tile once and return contiguous arrays with spatial axes last."""
    required_dims = ("year", "month", "day", "lat", "lon")
    gcm_arrays = []
    obs_arrays = []

    for var_name in var_names:
        if var_name not in reshaped_gcm or var_name not in reshaped_obs:
            raise KeyError(f"Missing required bias-correction variable: {var_name}")

        gcm_var = reshaped_gcm[var_name].transpose(*required_dims)
        obs_var = reshaped_obs[var_name].transpose(*required_dims)
        if gcm_var.shape != obs_var.shape:
            raise ValueError(
                f"Shape mismatch for {var_name}: "
                f"GCM {gcm_var.shape} != reference {obs_var.shape}"
            )

        gcm_arrays.append(np.asarray(gcm_var.values, dtype=np.float32))
        obs_arrays.append(np.asarray(obs_var.values, dtype=np.float32))

    gcm_array = np.ascontiguousarray(np.stack(gcm_arrays, axis=0))
    obs_array = np.ascontiguousarray(np.stack(obs_arrays, axis=0))
    if not np.isfinite(gcm_array).all():
        raise ValueError("GCM tile contains NaN or infinite values before correction.")
    if not np.isfinite(obs_array).all():
        raise ValueError("Reference tile contains NaN or infinite values before correction.")

    return gcm_array, obs_array


def _correct_hist_cell_numpy(config, lat_index, lon_index, gcm_cell, obs_cell):
    """Correct one complete grid-cell history using isolated NumPy inputs."""
    gcm_cell = np.ascontiguousarray(gcm_cell, dtype=np.float32)
    obs_cell = np.ascontiguousarray(obs_cell, dtype=np.float32)
    if gcm_cell.ndim != 4 or obs_cell.ndim != 4:
        raise ValueError(
            "Fortran cell inputs must have shape (variable, year, month, day)."
        )
    if gcm_cell.shape != obs_cell.shape:
        raise ValueError(
            f"Cell input shape mismatch: GCM {gcm_cell.shape} != "
            f"reference {obs_cell.shape}"
        )
    if not np.isfinite(gcm_cell).all() or not np.isfinite(obs_cell).all():
        raise ValueError(
            f"Non-finite cell input at lat index {lat_index}, lon index {lon_index}."
        )

    corrected, bc_params = correction_wrapper(config, gcm_cell, obs_cell)
    corrected = np.asarray(corrected)
    if corrected.shape != gcm_cell.shape:
        raise ValueError(
            f"Unexpected corrected shape at lat index {lat_index}, "
            f"lon index {lon_index}: {corrected.shape} != {gcm_cell.shape}"
        )
    if not np.isfinite(corrected).all():
        raise FloatingPointError(
            f"Non-finite corrected values at lat index {lat_index}, "
            f"lon index {lon_index}."
        )

    return lat_index, lon_index, corrected, bc_params


def _correct_hist_grid_cells(config, reshaped_gcm, reshaped_obs, var_names):
    """
    Correct a tile in bounded horizontal batches.

    The complete time history remains intact in every task. Only the horizontal
    cell dimension is partitioned, so each native call receives an array shaped
    (variable, year, month, day).
    """
    gcm_array, obs_array = _materialize_hist_cell_inputs(
        reshaped_gcm, reshaped_obs, var_names
    )
    _, n_year, n_month, n_day, n_lat, n_lon = gcm_array.shape
    output_shape = (n_year, n_month, n_day, n_lat, n_lon)
    corrected_data = {
        var_name: np.full(output_shape, np.nan, dtype=np.float32)
        for var_name in var_names
    }
    bc_params_array = np.full((n_lat, n_lon), None, dtype=object)

    cells = list(itertools.product(range(n_lat), range(n_lon)))
    batch_size = max(1, int(getattr(config, "dask_cell_batch_size", 32)))
    for batch_start in range(0, len(cells), batch_size):
        batch = cells[batch_start : batch_start + batch_size]
        tasks = []
        for lat_index, lon_index in batch:
            # Copies are deliberate: workers receive only this cell, never the
            # complete xarray tile or a lazy graph referencing it.
            gcm_cell = np.ascontiguousarray(
                gcm_array[:, :, :, :, lat_index, lon_index]
            )
            obs_cell = np.ascontiguousarray(
                obs_array[:, :, :, :, lat_index, lon_index]
            )
            tasks.append(
                dask.delayed(_correct_hist_cell_numpy, pure=False)(
                    config,
                    lat_index,
                    lon_index,
                    gcm_cell,
                    obs_cell,
                )
            )

        for lat_index, lon_index, corrected, bc_params in dask.compute(*tasks):
            for var_index, var_name in enumerate(var_names):
                corrected_data[var_name][:, :, :, lat_index, lon_index] = corrected[
                    var_index
                ]
            bc_params_array[lat_index, lon_index] = bc_params.to_dict()

    if any(param is None for param in bc_params_array.flat):
        raise RuntimeError("Bias correction did not return parameters for every grid cell.")
    for var_name, values in corrected_data.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(
                f"Bias correction left non-finite values in {var_name}."
            )

    corrected_ds = xr.Dataset(
        {
            var_name: (
                ["year", "month", "day", "lat", "lon"],
                corrected_data[var_name],
            )
            for var_name in var_names
        },
        coords={
            "year": reshaped_gcm.year,
            "month": reshaped_gcm.month,
            "day": reshaped_gcm.day,
            "lat": reshaped_gcm.lat,
            "lon": reshaped_gcm.lon,
        },
    )
    return corrected_ds, bc_params_array


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
    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )
    # gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    # obs_data = [reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    # Use index-based selection to avoid .sel() KeyErrors on tile edges
    gcm_data = [_isel_point(reshaped_gcm[var], lat, lon).values for var in var_list_w]
    obs_data = [_isel_point(reshaped_obs[var], lat, lon).values for var in var_list_w]

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
    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )

    # gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    # Use index-based selection to avoid .sel() KeyErrors on tile edges
    gcm_data = [_isel_point(reshaped_gcm[var], lat, lon).values for var in var_list_w]

    # # Find the nearest latitude and longitude indices in reshaped_gcm_delayed_f
    # lat_array = reshaped_gcm.lat.values
    # lon_array = reshaped_gcm.lon.values

    # lat_idx = (np.abs(lat_array - lat)).argmin()  # Index of the nearest latitude
    # lon_idx = (np.abs(lon_array - lon)).argmin()  # Index of the nearest longitude

    # Select the corresponding value in bc_params_array_loaded
    # params_data = bc_params_array[lat_idx, lon_idx]
    params_ds = _isel_point(bc_params_array, lat, lon)
    # For each variable, reshape the flattened data back to its original shape.
    params = {}
    for var in params_ds.data_vars:
        flat_data = params_ds[var].values
        original_shape = params_ds[var].attrs.get("original_shape", None)
        if original_shape is None:
            raise ValueError(f"Original shape not found for variable '{var}'.")
        params[var] = flat_data.reshape(original_shape)
    # Convert the dictionary to a SimpleNamespace so that we can access attributes like params_data.avdc_iter
    params_data = SimpleNamespace(**params)

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
    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )

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

    if getattr(config, "single_grid_test", False):
        # The one-grid smoke test is only meant to exercise the BC core.
        # Lateral boundary reconstruction expects a spatial field and is not
        # meaningful for a single grid cell.
        return six_hourly_data_hist

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
#     config,
#     sliced_gcm,
#     sliced_obs,
# ):
#     """
#     Process a single tile for bias correction, involving loading GCM and observational data,
#     performing bias correction, and saving the outputs.

#     Args:
#         sliced_gcm (xarray.Dataset): Sliced GCM data for the tile.
#         sliced_obs (xarray.Dataset): Sliced observational data for the tile.
#         config (module): Configuration object for bias correction parameters.

#     Returns:
#         tuple: Corrected GCM data and bias correction parameters.
#     """

#     # ------------------ Load GCM Data ------------------
#     if config.bc_boundary == "lateral":
#         assign_gcm = assign_w_6hr(config, sliced_gcm, config.bc_boundary)
#         daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(
#             config, assign_gcm
#         )
#     else:
#         daily_gcm = sliced_gcm

#     var_list_w = (
#         ["w", "ta", "hus"]
#         if config.bc_boundary == "lateral"
#         else config.target_variable
#     )

#     reshaped_gcm_delayed = extract_and_reshape_delayed(
#         config,
#         daily_gcm,
#         config.no_of_variables,
#         config.startyear_h,
#         config.endyear_h,
#         config.bc_boundary,
#     )

#     if config.bc_boundary != "lateral":
#         reshaped_gcm_delayed += 273.15

#     # ------------------ Load Observational Data ------------------
#     if config.bc_boundary == "lateral":
#         assign_obs = assign_w_6hr(config, sliced_obs, config.bc_boundary)
#         daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(
#             config, assign_obs
#         )
#     else:
#         daily_obs = sliced_obs

#     reshaped_obs_delayed = extract_and_reshape_delayed(
#         config,
#         daily_obs,
#         config.no_of_variables,
#         config.startyear_h,
#         config.endyear_h,
#         config.bc_boundary,
#     )

#     # Perform bias correction across the tile
#     if config.bc_boundary == "lateral":
#         bc_corrected_gcm_hist_tile, bc_params_tile = bc_correction_grid_cell_hist_dask(
#             config,
#             reshaped_gcm_delayed,
#             reshaped_obs_delayed,
#             fraction_factors_gcm,
#             fraction_factors_obs,
#             var_list_w,
#             sliced_gcm,
#         )

#     else:
#         bc_corrected_gcm_hist_tile, bc_params_tile = (
#             bc_correction_grid_cell_hist_dask_2d(
#                 config,
#                 reshaped_gcm_delayed,
#                 reshaped_obs_delayed,
#                 var_list_w,
#                 config.startyear_h,
#             )
#         )

#     return bc_corrected_gcm_hist_tile, bc_params_tile


def _prepare_hist_args(config, sliced_gcm, sliced_obs):
    if config.bc_boundary == "lateral":
        assign_gcm = assign_w_6hr(config, sliced_gcm, config.bc_boundary)
        daily_gcm, ff_gcm = convert_to_daily_with_fraction(config, assign_gcm)
        assign_obs = assign_w_6hr(config, sliced_obs, config.bc_boundary)
        daily_obs, ff_obs = convert_to_daily_with_fraction(config, assign_obs)
        var_list_w = ["w", "ta", "hus"]
    else:
        daily_gcm, daily_obs = sliced_gcm, sliced_obs
        ff_gcm = ff_obs = None
        var_list_w = config.target_variable

    rg = extract_and_reshape_delayed(
        config,
        daily_gcm,
        config.no_of_variables,
        config.startyear_h,
        config.endyear_h,
        config.bc_boundary,
    )
    ro = extract_and_reshape_delayed(
        config,
        daily_obs,
        config.no_of_variables,
        config.startyear_h,
        config.endyear_h,
        config.bc_boundary,
    )
    if config.bc_boundary != "lateral":
        rg = rg + 273.15

    return rg, ro, ff_gcm, ff_obs, var_list_w, sliced_gcm


def process_tile(
    config,
    sliced_gcm,
    sliced_obs,
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

    rg, ro, ffg, ffo, var_list_w, gcm_pass = _prepare_hist_args(
        config, sliced_gcm, sliced_obs
    )

    # Perform bias correction across the tile
    if config.bc_boundary == "lateral":
        ds, bc_params = bc_correction_grid_cell_hist_dask(
            config, rg, ro, ffg, ffo, var_list_w, gcm_pass.copy(deep=True)
        )
    else:
        ds, bc_params = bc_correction_grid_cell_hist_dask_2d(
            config,
            rg,
            ro,
            var_list_w,
            config.startyear_h,
        )

    return ds, bc_params


def future_subdaily_correction(
    config,
    sliced_gcm,
    sliced_obs,
    ff_future,
    bc_corrected_gcm_future_tile,
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

    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )
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


def process_tile_future(
    config,
    tile,
    variables,
    level,
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

    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )

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
    bc_corrected_gcm_future_tile = bc_correction_grid_cell_future_dask(
        config,
        reshaped_gcm_delayed,
        bc_params_array_loaded,
        var_list_w,
    )
    # bc_corrected_gcm_future_tile = bc_correction_grid_cell_future_multiprocess(
    #     config,
    #     reshaped_gcm_delayed,
    #     bc_params_array_loaded,
    #     var_list_w,
    # )

    if config.sub_daily_correction:
        bc_corrected_gcm_future = future_subdaily_correction(
            config,
            gcm_hist,
            obs_hist,
            ff_future,
            bc_corrected_gcm_future_tile,
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
    variable,
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
        variable (list of str): List of variable names to be corrected.
        sliced_gcm_future (xarray.Dataset): Sliced GCM data.

    Returns:
        xarray.Dataset: Bias-corrected daily GCM data for the future.
    """

    # Extract lat/lon values
    lat_values = gcm_future.lat.values
    lon_values = gcm_future.lon.values

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
        dask.delayed(process_grid_cell_future)(
            config, lat, lon, gcm_future, bc_params_array
        )
        for lat, lon in itertools.product(lat_values, lon_values)
    ]

    # Compute all tasks in parallel at the end
    results = dask.compute(*tasks)

    # Flatten the list of results
    flattened_results = [item for sublist in results for item in sublist]

    # Initialize arrays to hold the final data
    corrected_data = {
        var: np.empty(
            (
                len(gcm_future.year),
                len(gcm_future.month),
                len(gcm_future.day),
                len(lat_values),
                len(lon_values),
            )
        )
        for var in variable
    }

    # # Fill the arrays with data from results
    # for result in results:
    #     lat_idx = np.where(gcm_future.lat.values == result["lat"])[0][0]
    #     lon_idx = np.where(gcm_future.lon.values == result["lon"])[0][0]
    #     for i, var in enumerate(var_list_w):
    #         corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]

    for result in results:
        li = _grid_index(lat_values, result["lat"])
        lj = _grid_index(lon_values, result["lon"])
        for i, var in enumerate(variable):
            corrected_data[var][:, :, :, li, lj] = result["gcmc_corrected"][i]

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

    if len(variable) == 1:
        daily_data_aligned = align_daily_data_xr(
            config, gcmc_corrected, config.startyear_f
        )
        return daily_data_aligned
    else:
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
            var_lat_range = lat_range
            var_lon_range = lon_range
            if getattr(config, "single_grid_test", False) and var_name in ["ua", "va"]:
                var_lat_range, var_lon_range = _single_grid_wind_bounds(
                    file_paths,
                    tile[0]["lat_min"],
                    tile[0]["lon_min"],
                )
                print(
                    f"[INFO] single_grid_test wind stencil for {var_name}: "
                    f"lat_range={var_lat_range}, lon_range={var_lon_range}"
                )
            data_var = load_preprocess_variable(
                config,
                file_paths,
                var_name,
                level,
                var_lat_range,
                var_lon_range,
                config.startyear_h,
                config.endyear_h,
            )
            if isinstance(data_var, xr.Dataset):
                data_var = data_var[var_name]

            # Check if the variable is one of the wind components with different lon
            if var_name in ["ua", "va"]:
                # Let's assume hus and ta have the target longitude values, and they are already loaded
                target_lon = sliced_gcm.lon if "lon" in sliced_gcm else data_var.lon
                target_lat = sliced_gcm.lat if "lat" in sliced_gcm else data_var.lat
                target_lev = sliced_gcm.lev if "lev" in sliced_gcm else data_var.lev

                if getattr(config, "single_grid_test", False):
                    # One-grid tests keep a small local source stencil for
                    # staggered wind grids, then interpolate to the scalar
                    # cell. This preserves the production intent while
                    # avoiding empty zero-width slices. Interpolate eagerly on
                    # the tiny loaded stencil instead of using xarray/scipy
                    # for this smoke test, because the full-stack path can hit
                    # empty-array edge cases on very small Dask chunks.
                    data_var = data_var.load()
                    data_var = _interp_to_point_eager(
                        data_var, target_lat, target_lon
                    )
                    data_var = data_var.assign_coords(
                        lon=target_lon, lat=target_lat, lev=target_lev
                    )
                else:
                    # Interpolate staggered wind grids to match scalar fields.
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

        if str(getattr(config, "preprocess_scope", "domain")).lower() == "tile":
            sliced_gcm_sel = sliced_gcm.astype(np.float32).load()
        else:
            sliced_gcm_sel = sliced_gcm.astype(np.float32).persist()
        lock = Lock(name=temp_file)  # one-writer-per-path
        with lock:
            # Double-check if the file was created while waiting for the lock
            if not os.path.exists(temp_file):
                _atomic_to_netcdf(sliced_gcm_sel, temp_file)
        # Ensure the dataset is closed after writing
        sliced_gcm_sel.close()
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
        if str(getattr(config, "preprocess_scope", "domain")).lower() == "tile":
            obs_ds_sel = load_preprocess_variable(
                config,
                file_paths,
                var_name,
                level_index,
                lat_range,
                lon_range,
                startyear_h,
                endyear_h,
            )
        else:
            obs_ds = xr.open_mfdataset(
                file_paths,
                combine="by_coords",
                chunks={"time": "auto", "lat": "auto", "lon": "auto"},
            )
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
        # Use a lock to ensure only one process writes to the file at a time
        lock = Lock(name=temp_file)  # one-writer-per-path
        with lock:
            # Double-check if the file was created while waiting for the lock
            if not os.path.exists(temp_file):
                _atomic_to_netcdf(obs_ds_sel, temp_file)
        # Ensure the dataset is closed after writing
        obs_ds_sel.close()
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


def _load_preprocess_variable_serial(
    config,
    file_paths,
    var_name,
    level_index,
    lat_range,
    lon_range,
    startyear_h,
    endyear_h,
):
    """Load one tile with serial NetCDF I/O, avoiding Dask worker HDF5 access."""
    arrays = []
    for file_path in file_paths:
        with xr.open_dataset(file_path, chunks=None) as ds:
            data = preprocess_dataset(
                config,
                ds,
                var_name,
                level_index,
                lat_range,
                lon_range,
                startyear_h,
                endyear_h,
                config.bc_boundary,
            )
            if data.sizes.get("time", 1) == 0:
                continue
            arrays.append(data.astype("float32").load())

    if not arrays:
        raise ValueError(
            f"No data found for {var_name} in lat={lat_range}, lon={lon_range}, "
            f"years={startyear_h}-{endyear_h}."
        )
    if len(arrays) == 1:
        combined = arrays[0]
    else:
        combined = xr.concat(arrays, dim="time")
    if "time" in combined.coords:
        combined = combined.sortby("time")
    return combined


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

    if str(getattr(config, "preprocess_scope", "domain")).lower() == "tile":
        return _load_preprocess_variable_serial(
            config,
            file_paths,
            var_name,
            level_index,
            lat_range,
            lon_range,
            startyear_h,
            endyear_h,
        )

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

    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )

    # Print the process ID (PID) to check parallel execution
    # print(f"Processing lat: {lat}, lon: {lon} on process ID: {os.getpid()}")

    # Extract GCM data for this (lat, lon)
    # gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    # obs_data = [reshaped_obs[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    # Index-based selection
    gcm_data = [_isel_point(reshaped_gcm[var], lat, lon).values for var in var_list_w]
    obs_data = [_isel_point(reshaped_obs[var], lat, lon).values for var in var_list_w]
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
    n = int(num_workers / 2)
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
    # for result in results:
    #     lat_idx = np.where(lat_values == result["lat"])[0][0]
    #     lon_idx = np.where(lon_values == result["lon"])[0][0]
    #     for i, var in enumerate(variable):
    #         corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]
    #     bc_params_array[lat_idx, lon_idx] = result["bc_params"].to_dict()
    # Use index lookup instead of float equality
    for result in results:
        li = _grid_index(lat_values, result["lat"])
        lj = _grid_index(lon_values, result["lon"])
        for i, var in enumerate(variable):
            corrected_data[var][:, :, :, li, lj] = result["gcmc_corrected"][i]
        bc_params_array[li, lj] = result["bc_params"].to_dict()

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

    gcmc_corrected, bc_params_array = _correct_hist_grid_cells(
        config,
        reshaped_gcm,
        reshaped_obs,
        var_list_w,
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
        var: np.empty(
            (
                len(reshaped_gcm.year),
                len(reshaped_gcm.month),
                len(reshaped_gcm.day),
                len(reshaped_gcm.lat),
                len(reshaped_gcm.lon),
            )
        )
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
    var_list_w = (
        ["w", "ta", "hus"]
        if config.bc_boundary == "lateral"
        else config.target_variable
    )
    # Extract GCM data for this (lat, lon)
    # gcm_data = [reshaped_gcm[var].sel(lat=lat, lon=lon).values for var in var_list_w]
    # Index-based selection for GCM data
    gcm_data = [_isel_point(reshaped_gcm[var], lat, lon).values for var in var_list_w]
    # Index-based selection for params (dataset)
    params_ds = _isel_point(bc_params_array, lat, lon)
    # # Find nearest indices
    # lat_array = reshaped_gcm.lat.values
    # lon_array = reshaped_gcm.lon.values

    # lat_idx = (np.abs(lat_array - lat)).argmin()
    # lon_idx = (np.abs(lon_array - lon)).argmin()

    # Select bias correction parameters
    # params_data = bc_params_array[lat_idx, lon_idx]
    # ds_cell = bc_params_array.sel(lat=lat, lon=lon, method="nearest")
    # ds_cell = bc_params_array.sel(lat=lat, lon=lon)

    # For each variable, reshape the flattened data back to its original shape.
    params = {}
    # for var in ds_cell.data_vars:
    #     # The flattened data is stored in a variable with a unique flattened dimension.
    #     flat_data = ds_cell[var].values
    #     # Retrieve the original shape from the variable's attributes.
    #     original_shape = ds_cell[var].attrs.get("original_shape", None)
    #     if original_shape is None:
    #         raise ValueError(f"Original shape not found for variable '{var}'.")
    #     # Reshape the 1D (flattened) data back to its original shape.
    #     reshaped_data = flat_data.reshape(original_shape)
    #     params[var] = reshaped_data
    for var in params_ds.data_vars:
        flat_data = params_ds[var].values
        original_shape = params_ds[var].attrs.get("original_shape", None)
        if original_shape is None:
            raise ValueError(f"Original shape not found for variable '{var}'.")
        params[var] = flat_data.reshape(original_shape)
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
    n = int(num_workers / 2)
    # n = min(len(os.sched_getaffinity(0)), num_workers, 96)
    print(f"Using {n} CPU cores for parallel processing...")

    with Pool(processes=n) as pool:
        results = pool.map(correction_wrapper_future_pool, args_list)

    # Convert results back to xarray.Dataset
    corrected_data = {
        var: np.empty(
            (
                len(gcm_future.year),
                len(gcm_future.month),
                len(gcm_future.day),
                len(lat_values),
                len(lon_values),
            )
        )
        for var in variable
    }

    # Fill the arrays
    # for result in results:
    #     lat_idx = np.where(lat_values == result["lat"])[0][0]
    #     lon_idx = np.where(lon_values == result["lon"])[0][0]
    #     for i, var in enumerate(variable):
    #         corrected_data[var][:, :, :, lat_idx, lon_idx] = result["gcmc_corrected"][i]
    for result in results:
        li = _grid_index(lat_values, result["lat"])
        lj = _grid_index(lon_values, result["lon"])
        for i, var in enumerate(variable):
            corrected_data[var][:, :, :, li, lj] = result["gcmc_corrected"][i]
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
        var: np.empty(
            (
                len(gcm_future.year),
                len(gcm_future.month),
                len(gcm_future.day),
                len(gcm_future.lat),
                len(gcm_future.lon),
            )
        )
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


# def apply_moving_window_bias_correction(
#     config,
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
#     """
#     Apply moving window bias correction over a given time frame for a grid cell.

#     Args:
#         lat (float): Latitude of the grid cell.
#         lon (float): Longitude of the grid cell.
#         gcm_future (xarray.Dataset): Future GCM data.
#         gcm_ref (xarray.Dataset): Reference GCM data.
#         fraction_factors (xarray.Dataset): Fraction factors for correction.
#         bc_params_array (xarray.Dataset): Bias correction parameters.
#         start_year (int): Start year for bias correction.
#         end_year (int): End year for bias correction.
#         window_size (int, optional): Size of the moving window for correction. Defaults to 30.

#     Returns:
#         xarray.Dataset: Bias-corrected dataset for the specified grid cell and time window.
#     """
#     var_list_w = (
#         ["w", "ta", "hus"] if config.bc_boundary == "lateral" else config.target_variable
#     )

#     corrected_dataset = xr.Dataset()

#     for target_year in range(start_year, end_year + 1):
#         pre_window = min(window_size // 2, target_year - start_year)
#         post_window = min(window_size // 2, end_year - target_year)

#         # If not enough data for a 30-year window, adjust to use as much as possible
#         if pre_window + post_window + 1 < window_size:
#             if target_year - start_year < window_size // 2:
#                 # Closer to start, prioritize adding years after the target year
#                 actual_window_start = min(start_year, target_year)
#                 actual_window_end = min(
#                     end_year, target_year + 15
#                 )  # Use 15 years after, if available
#             else:
#                 # Closer to end, prioritize adding years before the target year
#                 actual_window_start = max(
#                     start_year, target_year - 15
#                 )  # Use 15 years before, if available
#                 actual_window_end = max(target_year, end_year)
#         else:
#             actual_window_start = target_year - pre_window
#             actual_window_end = target_year + post_window

#         window_data = gcm_future.sel(
#             year=slice(str(actual_window_start), str(actual_window_end))
#         )
#         fraction_factor = fraction_factors.sel(
#             time=slice(str(actual_window_start), str(actual_window_end))
#         )

#         if config.sub_daily_correction:
#             corrected_data = bc_correction_grid_cell_future_multiprocess(
#                 config,
#                 lat,
#                 lon,
#                 fraction_factor,
#                 window_data,
#                 fraction_factor,
#                 bc_params_array,
#                 actual_window_start,
#                 actual_window_end,
#                 var_list_w,
#                 gcm_future,
#             )
#         else:
#             corrected_data = bc_correction_grid_cell_future_dask(
#                 config,
#                 lat,
#                 lon,
#                 gcm_ref,
#                 window_data,
#                 fraction_factor,
#                 bc_params_array,
#                 actual_window_start,
#                 actual_window_end,
#                 var_list_w,
#             )

#         # Adjusting datetime to target year and handling February 29th
#         # corrected_data = adjust_dates_to_target_year(corrected_data, target_year)
#         corrected_data = corrected_data.sel(
#             time=slice(f"{target_year}-01-01", f"{target_year}-12-31")
#         )

#         if not pd.Index(corrected_data["time"].values).is_monotonic_increasing:
#             corrected_data = corrected_data.sortby("time")
#         if target_year == start_year:
#             corrected_dataset = xr.merge(
#                 [corrected_dataset, corrected_data], compat="override"
#             )
#         else:
#             corrected_dataset = xr.concat(
#                 [corrected_dataset, corrected_data], dim="time"
#             )

#         print(f"{target_year} completed")

#     return corrected_dataset


# def convert_bc_params_to_xarray(config, bc_params_array, lat_values, lon_values):
#     nlat, nlon = bc_params_array.shape

#     # --- Extract Variable Keys and Pre-allocate Arrays ---
#     # Find the variable keys from the first non-None dictionary in bc_params_array.
#     var_keys = None
#     for i in range(nlat):
#         for j in range(nlon):
#             if bc_params_array[i, j] is not None:
#                 var_keys = list(bc_params_array[i, j].keys())
#                 break
#         if var_keys is not None:
#             break

#     if var_keys is None:
#         raise ValueError("No valid dictionary found in bc_params_array.")

#     # For each key, determine the flattened size from the first sample,
#     # converting scalars to numpy arrays so that they have a shape.
#     data_vars = {}
#     original_shapes = {}  # to store each variable's original shape for later reshaping
#     for key in var_keys:
#         sample_val = bc_params_array[i, j][key]
#         # Convert scalars (or objects without shape) to numpy arrays.
#         if np.isscalar(sample_val) or not hasattr(sample_val, "shape"):
#             sample_array = np.array([sample_val])
#         else:
#             sample_array = sample_val
#         original_shapes[key] = sample_array.shape
#         flat_len = sample_array.size
#         data_vars[key] = np.empty((flat_len, nlat, nlon), dtype=sample_array.dtype)
#         # print(f"{key}: original shape = {sample_array.shape}, flat length = {flat_len}")

#     # --- Fill the Data Arrays ---
#     # Loop through each grid cell, flatten the array from each dictionary, and store it.
#     for i in range(nlat):
#         for j in range(nlon):
#             cell_dict = bc_params_array[i, j]
#             if cell_dict is not None:
#                 for key in var_keys:
#                     val = cell_dict[key]
#                     if np.isscalar(val) or not hasattr(val, "shape"):
#                         arr = np.array([val])
#                     else:
#                         arr = val
#                     data_vars[key][:, i, j] = arr.flatten()
#             else:
#                 for key in var_keys:
#                     data_vars[key][:, i, j] = np.nan

#     # --- Create xarray Dataset ---
#     # Build an xarray Dataset with coordinates for lat and lon.
#     ds = xr.Dataset(
#         coords={"lat": (("lat",), lat_values), "lon": (("lon",), lon_values)}
#     )

#     # Add each variable to the Dataset with a unique flattened dimension for each variable.
#     for key, arr in data_vars.items():
#         flat_dim_name = f"{key}_flat"
#         ds[key] = ((flat_dim_name, "lat", "lon"), arr)
#         ds[key].attrs["original_shape"] = original_shapes[key]

#     # (Optional) Save the Dataset to a NetCDF file instead of npy for more efficient I/O.
#     # ds.to_netcdf("bc_params.nc")
#     return ds


def convert_bc_params_to_xarray(config, bc_params_array, lat_values, lon_values):
    nlat, nlon = bc_params_array.shape

    # ---- discover keys from first non-None cell ----
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

    def to_array(v):
        if np.isscalar(v) or not hasattr(v, "shape"):
            return np.array([v])
        return np.asarray(v)

    # ---- scan all cells to find the MAX (global) shape per key ----
    # We pick, for each key, the shape with the largest total size seen across the grid.
    ref_shape_by_key = {}
    max_flat_len = {}

    for k in var_keys:
        ref_shape_by_key[k] = None
        max_flat_len[k] = 0

    for i in range(nlat):
        for j in range(nlon):
            d = bc_params_array[i, j]
            if d is None:
                continue
            for k in var_keys:
                a = to_array(d[k])
                size = a.size
                # update to the largest shape encountered
                if size > max_flat_len[k]:
                    max_flat_len[k] = size
                    ref_shape_by_key[k] = tuple(a.shape)

    # safety: ensure we found at least one shape per key
    missing = [k for k, shp in ref_shape_by_key.items() if shp is None]
    if missing:
        raise ValueError(f"No sample shape found for keys: {missing}")

    # ---- allocate flat arrays sized to the MAX flat length per key ----
    data_vars = {}
    for k in var_keys:
        L = max_flat_len[k]
        # use float to allow NaN padding (xarray will write doubles by default)
        data_vars[k] = np.full((L, nlat, nlon), np.nan, dtype=float)

    # ---- fill (pad with NaN up to the MAX length) ----
    for i in range(nlat):
        for j in range(nlon):
            d = bc_params_array[i, j]
            if d is None:
                continue
            for k in var_keys:
                flat = to_array(d[k]).astype(float, copy=False).ravel()
                L = flat.size
                data_vars[k][:L, i, j] = flat  # remainder stays NaN

    # ---- build Dataset with one flat dim per variable, as in your level-0 file ----
    ds = xr.Dataset(coords={"lat": ("lat", np.asarray(lat_values, dtype=float)),
                            "lon": ("lon", np.asarray(lon_values, dtype=float))})

    for k, arr in data_vars.items():
        flat_dim = f"{k}_flat"
        # dimension naming + attach variable with (flat, lat, lon)
        ds[k] = ((flat_dim, "lat", "lon"), arr)

        # store the EXACT "original_shape" attribute as an INT array
        # so it appears as "3LL, 12LL, ..." in ncdump and is easily reshape-able
        ds[k].attrs["original_shape"] = np.asarray(ref_shape_by_key[k], dtype=np.int64)

    return ds
