"""
This script performs horizontal and vertical interpolation of atmospheric variables from ERA5 reanalysis data to a target Global Climate Model (GCM) dataset. The interpolated data is saved as NetCDF files.

The script takes input variables from the ERA5 reanalysis data, such as specific humidity (q), temperature (t), zonal wind (u), and meridional wind (v), and maps them to the target variables in the GCM dataset, which are specific humidity (hus), air temperature (ta), zonal wind (ua), and meridional wind (va).

The horizontal interpolation is performed using the xESMF library, which provides conservative or bilinear methods depending on the input variable. The vertical interpolation is performed by converting pressure levels to hybrid height coordinates (and hybrid sigma pressure level) using the geopotential height from ERA5 and the coefficients from the GCM dataset.

The script utilizes Dask for parallel processing and xarray for data manipulation. The dask.distributed.Client is used to create a local cluster for parallel processing.

The script is designed to be run on the NCI's Gadi supercomputer, but can be modified to run on other systems.

Note: This script assumes that the necessary input files and directories are available and properly formatted.

Author: Youngil (Young) Kim, CCRC, UNSW
Contact: youngil.kim@unsw.edu.au
"""

import argparse
import gc
import glob
import os
import warnings

# from multiprocessing import Pool
from pathlib import Path

import dask  # type: ignore
import dask.array as da  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# import dask.array as da  # type: ignore
import xarray as xr  # type: ignore
import xesmf as xe  # type: ignore
import yaml  # type: ignore
from dask.diagnostics import ProgressBar  # type: ignore
from dask.distributed import Client  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

warnings.simplefilter("ignore", UserWarning)


# Start of the script -----------------------------------------------------
def load_config(yaml_path):
    """
    Load configuration from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        Config: Configuration object with loaded settings.
    """
    with open(yaml_path, "r") as file:
        config_data = yaml.safe_load(file)
    # return Config(**config_data)
    return config_data


# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)


def setup_client(ncpus, mem_gb):
    """
    Dynamically set up a Dask client based on available CPUs and memory.
    """
    # Set threads per worker (try to keep under 8 for memory balance)
    threads_per_worker = min(ncpus, 8)
    n_workers = max(1, ncpus // threads_per_worker)
    mem_per_worker = int(mem_gb / n_workers)

    print(
        f"[INFO] Starting Dask client: {n_workers} workers × {threads_per_worker} threads"
    )
    print(f"[INFO] Each worker memory limit: {mem_per_worker}GB")

    return Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f"{mem_per_worker}GB",
    )


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run atmospheric data interpolation.")
    parser.add_argument(
        "--yp",
        type=str,
        default="./user_input_test.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--var",
        type=str,
        nargs="+",
        default=["hus", "ta", "ua", "va"],
        help="Variable(s) to interpolate (e.g., --var ta or --var ta hus).",
    )

    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument("--mem", type=int, default=None)

    # parser.add_argument(
    #     "--sy", type=int, default=config.startyear_h, help="Start year."
    # )
    # parser.add_argument("--ey", type=int, default=config.endyear_h, help="End year.")

    return parser.parse_args()


def normalize_target_variables(selected_variables):
    """
    Normalize a target-variable selection to a non-empty list of variable names.

    Parameters:
        selected_variables (str | list[str] | tuple[str, ...]): Variable selection.

    Returns:
        list[str]: Normalized variable names.
    """
    if isinstance(selected_variables, str):
        selected_variables = [selected_variables]
    else:
        selected_variables = list(selected_variables)

    if not selected_variables:
        raise ValueError("At least one target variable must be provided.")

    return selected_variables


# Start of the script -----------------------------------------------------
def standardize_dims(ds):
    """
    Standardize the dimension names of an xarray Dataset.

    This function attempts to rename the dimensions of the given xarray Dataset
    to a set of standard names: "time", "lat", "lon", and "lev". It uses the
    following heuristics to guess the appropriate standard name for each dimension:

    - "time": if the dimension's data type is datetime64[ns]
    - "lat": if the dimension name contains "lat" (case insensitive) or if the
      dimension's values are within the range [-90, 90]
    - "lon": if the dimension name contains "lon" (case insensitive) or if the
      dimension's values are within the range [-180, 180]
    - "lev": if the dimension name contains "lev", "level", "height", or "depth"
      (case insensitive)

    Parameters:
    ds (xarray.Dataset): The input xarray Dataset whose dimensions need to be standardized.

    Returns:
    xarray.Dataset: A new xarray Dataset with standardized dimension names.
    """
    standard_names = ["time", "lat", "lon", "lev"]
    used_names = set(ds.dims) & set(
        standard_names
    )  # Find intersection of existing and standard names
    dim_map = {}

    # Function to guess dimension type based on data
    def guess_dim_type(dim):
        if ds[dim].dtype == "datetime64[ns]":
            return "time"
        elif "lat" in dim.lower() or (ds[dim].max() <= 90 and ds[dim].min() >= -90):
            return "lat"
        elif "lon" in dim.lower() or (ds[dim].max() <= 180 and ds[dim].min() >= -180):
            return "lon"
        elif (
            "lev" in dim.lower()
            or "level" in dim.lower()
            or "height" in dim.lower()
            or "depth" in dim.lower()
        ):
            return "lev"
        return None

    # Attempt to map each dimension to a standard name if applicable
    for dim in ds.dims:
        if dim not in used_names:
            suggested_name = guess_dim_type(dim)
            if suggested_name and suggested_name not in used_names:
                dim_map[dim] = suggested_name
                used_names.add(suggested_name)  # Mark this standard name as used

    return ds.rename_dims(dim_map)


# Define functions for the horizontal and vertical interpolation ----------------
def geopotential_to_geopotential_height(g):
    """
    Calculate geopotential height from the geopotential variable.

    Parameters:
    g (float): The geopotential variable from ERA5, in m^2 s^-2.

    Returns:
    float: The geopotential height Z, in meters.

    Formula:
    The geopotential height Z is calculated as Z = g / g0, where g0 = 9.80665 m/s^2.
    """
    g0 = 9.80665
    Z = g / g0
    return Z


def calculate_hybrid_height(a, b, orog):
    """
    Calculates the hybrid height using coefficients from the GCM dataset and the surface orography.

    Parameters:
    a (float): Coefficient 'a' from the GCM dataset.
    b (float): Coefficient 'b' from the GCM dataset.
    orog (float): Surface orography.

    Returns:
    float: The calculated hybrid height.

    """
    z = a + b * orog
    return z


def calculate_pressure_levels(ap, b, ps):
    """
    Calculate pressure at each model level by automatically broadcasting 'ap' and 'b'
    against 'ps' across their shared dimensions ('lat', 'lon') and aligning them with
    'ps' time dimension.

    Parameters:
    - ap: Xarray DataArray of 'ap' coefficient, shaped [lev]
    - b: Xarray DataArray of 'b' coefficient, shaped [lev]
    - ps: Xarray DataArray of surface pressure, shaped [time, lat, lon]

    Returns:
    - p_levels: Xarray DataArray of pressure at each model level, shaped [time, lev, lat, lon]
    """

    # Ensure 'ap' and 'b' are broadcasted and aligned along 'ps' dimensions
    # This uses Xarray's automatic alignment and broadcasting
    ap_expanded = ap * xr.ones_like(
        ps
    )  # This automatically broadcasts 'ap' across 'ps' dimensions
    b_expanded = b * xr.ones_like(ps)  # Similarly for 'b'

    # Calculate pressure at each model level
    p_levels = ap_expanded + b_expanded * ps
    return p_levels


# def compute_geopotential_height(p_levels, T_levels, q_levels=None):
#     Rd = 287.05  # J/kg/K, specific gas constant for dry air
#     g = 9.80665  # m/s^2, acceleration due to gravity

#     # Ensure T_levels and q_levels are DataArrays for compatibility with xarray operations
#     T_levels = (
#         T_levels if isinstance(T_levels, xr.DataArray) else xr.DataArray(T_levels)
#     )
#     if q_levels is not None:
#         q_levels = (
#             q_levels if isinstance(q_levels, xr.DataArray) else xr.DataArray(q_levels)
#         )

#     # Compute the pressure ratio without aligning by levels
#     upper_p = p_levels.isel(
#         lev=slice(None, -1)
#     ).data  # Pressure at the lower boundary of each layer
#     lower_p = p_levels.isel(
#         lev=slice(1, None)
#     ).data  # Pressure at the upper boundary of each layer
#     p_ratio = upper_p / lower_p
#     log_p_ratio = da.log(p_ratio)

#     # Re-create the DataArray for log_p_ratio with adjusted coordinates
#     log_p_ratio_da = xr.DataArray(
#         log_p_ratio,
#         dims=["time", "lev", "lat", "lon"],
#         coords={
#             "time": p_levels.time,
#             "lev": p_levels.lev[:-1],  # Use coordinates from the upper slice
#             "lat": p_levels.lat,
#             "lon": p_levels.lon,
#         },
#     )

#     # Compute mean temperature between consecutive levels without alignment
#     upper_T = T_levels.isel(lev=slice(None, -1)).data
#     lower_T = T_levels.isel(lev=slice(1, None)).data
#     mean_T = (upper_T + lower_T) / 2

#     # Re-create the DataArray for mean_T with adjusted coordinates
#     mean_T_da = xr.DataArray(
#         mean_T,
#         dims=["time", "lev", "lat", "lon"],
#         coords=log_p_ratio_da.coords,  # Match coordinates with log_p_ratio_da
#     )

#     if q_levels is not None:
#         # Compute moist temperature
#         upper_q = q_levels.isel(lev=slice(None, -1)).data
#         lower_q = q_levels.isel(lev=slice(1, None)).data
#         mean_q = (upper_q + lower_q) / 2

#         # Re-create the DataArray for mean_q with adjusted coordinates
#         mean_q_da = xr.DataArray(
#             mean_q,
#             dims=["time", "lev", "lat", "lon"],
#             coords=log_p_ratio_da.coords,  # Match coordinates with log_p_ratio_da
#         )
#         mean_T_da = mean_T_da * (1.0 + 0.609133 * mean_q_da)

#     # Calculate thickness of each layer (delta Z)
#     delta_Z = (Rd / g) * mean_T_da * log_p_ratio_da

#     # Integrate delta_Z from the top to obtain geopotential heights
#     Z_levels_cumsum = delta_Z.cumsum(dim="lev")

#     # Add an extra level at the top with extrapolated geopotential height
#     gradient_top = Z_levels_cumsum.isel(lev=-1) - Z_levels_cumsum.isel(lev=-2)
#     top_extrapolated = Z_levels_cumsum.isel(lev=-1) + gradient_top
#     Z_levels = xr.concat(
#         [Z_levels_cumsum, top_extrapolated.expand_dims(lev=[p_levels.lev[-1]])],
#         dim="lev",
#     )

#     # Update level coordinates to include the top level
#     Z_levels = xr.DataArray(
#         Z_levels,
#         dims=["time", "lev", "lat", "lon"],
#         coords={
#             "time": p_levels.time,
#             "lev": p_levels.lev,  # Use original levels, assuming the extra level is added at the top
#             "lat": p_levels.lat,
#             "lon": p_levels.lon,
#         },
#         name="zfull",
#     )

#     return Z_levels


def compute_geopotential_height(p_levels, T_levels, orog, q_levels=None):
    """
    Compute the geopotential height at various pressure levels.

    Parameters:
    -----------
    p_levels : xarray.DataArray
        Pressure levels (Pa) with dimensions (time, lev, lat, lon).
    T_levels : xarray.DataArray or array-like
        Temperature levels (K) with dimensions (time, lev, lat, lon).
    orog : xarray.DataArray or array-like
        Surface orography (m) with dimensions (time, lat, lon).
    q_levels : xarray.DataArray or array-like, optional
        Specific humidity levels (kg/kg) with dimensions (time, lev, lat, lon).

    Returns:
    --------
    xarray.DataArray
        Geopotential height (m) with dimensions (time, lev, lat, lon).
    """
    Rd = 287.05  # J/kg/K, specific gas constant for dry air
    g0 = 9.80665  # m/s^2, acceleration due to gravity

    # Ensure input variables are xarray DataArrays
    T_levels = (
        T_levels if isinstance(T_levels, xr.DataArray) else xr.DataArray(T_levels)
    )

    if q_levels is not None:
        q_levels = (
            q_levels if isinstance(q_levels, xr.DataArray) else xr.DataArray(q_levels)
        )

    # Pressure levels
    upper_p = p_levels.isel(lev=slice(None, -1)).data
    lower_p = p_levels.isel(lev=slice(1, None)).data
    p_ratio = upper_p / lower_p
    log_p_ratio = da.log(p_ratio)

    log_p_ratio_da = xr.DataArray(
        log_p_ratio,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": p_levels.time,
            "lev": p_levels.lev[:-1],
            "lat": p_levels.lat,
            "lon": p_levels.lon,
        },
    )

    # Compute mean temperature
    upper_T = T_levels.isel(lev=slice(None, -1)).data
    lower_T = T_levels.isel(lev=slice(1, None)).data
    mean_T = (upper_T + lower_T) / 2

    mean_T_da = xr.DataArray(
        mean_T,
        dims=["time", "lev", "lat", "lon"],
        coords=log_p_ratio_da.coords,
    )

    if q_levels is not None:
        upper_q = q_levels.isel(lev=slice(None, -1)).data
        lower_q = q_levels.isel(lev=slice(1, None)).data
        mean_q = (upper_q + lower_q) / 2

        mean_q_da = xr.DataArray(
            mean_q,
            dims=["time", "lev", "lat", "lon"],
            coords=log_p_ratio_da.coords,
        )
        mean_T_da = mean_T_da * (1.0 + 0.609133 * mean_q_da)

    # Compute layer thickness
    delta_Z = (Rd / g0) * mean_T_da * log_p_ratio_da
    delta_Phi = g0 * delta_Z  # Convert height differences to geopotential differences

    # Convert surface altitude to surface geopotential (Z_g = orog + delta_Z.cumsum(dim="lev") can be used as the orog already in meters)
    # Phi_s = g0 * orog  # Convert orog (meters) to geopotential (m²/s²)
    Phi_s = (g0 * orog).broadcast_like(
        delta_Phi.isel(lev=0)
    )  # (time, lat, lon) via auto align

    # Integrate from surface geopotential height
    Phi_levels = Phi_s + delta_Phi.cumsum(dim="lev")

    Z_g_mid = Phi_levels / g0  # Mid-layer geopotential height

    # Surface level height
    Z_s = orog  # Surface geopotential height in meters

    # Expand Z_s to match dimensions and concatenate it as the first level
    Z_s_expanded = Z_s.expand_dims(
        dim={"lev": [p_levels.lev[0]]}, axis=1
    )  # Shape: (time, 1, lat, lon)

    # Concatenate
    Z_g = xr.concat([Z_s_expanded, Z_g_mid], dim="lev")
    Z_g = Z_g.transpose("time", "lev", "lat", "lon")

    # Ensure the 'lev' coordinate matches p_levels
    Z_g = xr.DataArray(
        Z_g,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": p_levels.time,
            "lev": p_levels.lev,  # Assign the full coordinate
            "lat": p_levels.lat,
            "lon": p_levels.lon,
        },
        name="zfull",
    )
    return Z_g


def clean_data(data):
    """
    Clean the data by removing NaN or infinity values.

    Args:
        data (array-like): Input data to clean.

    Returns:
        array-like: Cleaned data with NaN and infinity values handled.
    """
    if isinstance(data, xr.DataArray):
        return data.where(~xr.ufuncs.isnan(data) & ~xr.ufuncs.isinf(data), drop=True)
    else:
        return data[np.isfinite(data)]


def custom_interp(x_new, interp_func, x_min, x_max):
    """
    Custom interpolation function that extrapolates for lower bounds and fills NaN for upper bounds.

    Args:
        x_new (array-like): The target levels to interpolate the source profile to.
        interp_func (callable): The interpolation function.
        x_min (float): The minimum value of the original x data.
        x_max (float): The maximum value of the original x data.

    Returns:
        array-like: The interpolated values with custom handling for extrapolation.
    """
    y_new = interp_func(x_new)
    lower_bound = x_new < x_min
    upper_bound = x_new > x_max
    y_new[upper_bound] = np.nan
    return y_new


# def interpolate_profile(source_profile, source_levels, target_levels):
#     """
#     Interpolates a source profile to match target levels with custom handling for extrapolation.

#     Args:
#         source_profile (array-like): The source profile to be interpolated.
#         source_levels (array-like): The levels corresponding to the source profile.
#         target_levels (array-like): The target levels to interpolate the source profile to.

#     Returns:
#         array-like: The interpolated profile matching the target levels.
#     """
#     # Clean the data to remove NaN or infinity values
#     source_profile = clean_data(source_profile)
#     source_levels = clean_data(source_levels)
#     target_levels = clean_data(target_levels)

#     # Ensure unique and sorted source levels
#     # unique_source_levels = np.unique(source_levels)
#     # if len(unique_source_levels) < 2:
#     #     # If there are not enough unique levels, return NaNs of the same shape as target_levels
#     #     return np.full_like(target_levels, np.nan, dtype=source_profile.dtype)

#     # Create the interpolation function with 'extrapolate' mode
#     f_interp = interp1d(
#         source_levels, source_profile, bounds_error=False, fill_value="extrapolate"
#     )

#     # Get the min and max of source levels
#     x_min = np.min(source_levels)
#     x_max = np.max(source_levels)

#     return custom_interp(target_levels, f_interp, x_min, x_max)


# def interpolate_profile(
#     source_profile, source_levels, target_levels, gcm_profile, gcm_levels
# ):
#     """
#     Interpolates a source profile to match target levels with custom handling for extrapolation.

#     Args:
#         source_profile (array-like): The source profile to be interpolated.
#         source_levels (array-like): The levels corresponding to the source profile.
#         target_levels (array-like): The target levels to interpolate the source profile to.

#     Returns:
#         array-like: The interpolated profile matching the target levels.
#     """
#     # Check for NaN values
#     if np.any(np.isnan(source_profile)) or np.any(np.isnan(source_levels)):
#         print("NaN values found in source_profile or source_levels.")
#         return np.full_like(
#             target_levels, np.nan
#         )  # Return NaN array of target levels' shape

#     # Check for duplicate values in source_levels
#     if len(np.unique(source_levels)) != len(source_levels):
#         print("Duplicate values found in source_levels.")
#         return np.full_like(
#             target_levels, np.nan
#         )  # Return NaN array of target levels' shape

#     # Create the interpolation function with 'extrapolate' mode
#     f_interp = interp1d(
#         source_levels, source_profile, bounds_error=False, fill_value="extrapolate"
#     )

#     # Interpolate values at target levels
#     interpolated_profile = f_interp(target_levels)

#     # Identify the highest ERA5 level
#     max_era5_level = np.max(source_levels)

#     # Replace values above ERA5 max level with GCM values
#     above_era5_mask = target_levels > max_era5_level
#     interpolated_profile[above_era5_mask] = np.interp(
#         target_levels[above_era5_mask], gcm_levels, gcm_profile
#     )

#     return interpolated_profile


# def vertical_interpolation(
#     source_da, source_levels_da, target_levels, gcm_profile, gcm_levels
# ):
#     """
#     Perform vertical interpolation of a source data array to target levels using xarray's apply_ufunc.

#     Parameters:
#         source_da (xarray.DataArray): The source data array to be interpolated.
#         source_levels_da (xarray.DataArray): The source data array's levels.
#         target_levels (array-like): The target levels to interpolate to.

#     Returns:
#         xarray.DataArray: The interpolated data array.
#     """
#     # Wrapper to apply interpolation using xarray's apply_ufunc to handle Dask arrays efficiently
#     # interpolated_da = xr.apply_ufunc(
#     #     interpolate_profile,
#     #     source_da,
#     #     source_levels,
#     #     target_levels,
#     #     gcm_profile,
#     #     gcm_levels,
#     #     vectorize=True,  # Enable vectorized execution
#     #     input_core_dims=[["level"], ["level"], ["lev"]],  # Define core dimensions
#     #     output_core_dims=[["lev"]],  # Define output dimensions
#     #     dask="parallelized",  # Enable Dask parallelization
#     #     output_dtypes=[source_da.dtype],
#     # )
#     print("Performing vertical interpolation...")
#     print("Source_da:", source_da)
#     print("Source_levels_da:", source_levels_da)
#     print("Target_levels:", target_levels)
#     print("GCM_profile:", gcm_profile)
#     print("GCM_levels:", gcm_levels)

#     interpolated_da = xr.apply_ufunc(
#         interpolate_profile,
#         source_da,  # ERA5 temperature
#         source_levels_da,  # ERA5 geopotential height
#         target_levels,  # Target geopotential height
#         gcm_profile,  # GCM temperature
#         gcm_levels,  # GCM geopotential height
#         vectorize=True,
#         input_core_dims=[
#             ["level"],
#             ["level"],
#             ["lev"],
#             ["lev"],
#             ["lev"],
#         ],  # Add GCM dims
#         output_core_dims=[["lev"]],
#         dask="parallelized",
#         output_dtypes=[source_da.dtype],
#     )
#     return interpolated_da.assign_coords(lev=target_levels.lev)


def get_vdim(da: xr.DataArray) -> str:
    for cand in ("lev", "level", "plev", "height"):
        if cand in da.dims:
            return cand
    raise ValueError(
        f"No vertical dim among ('lev','level','plev','height') in {da.dims}"
    )


def _interpolate_profile_1d(
    source_profile, source_levels, target_levels, gcm_profile, gcm_levels
):
    # finite & enough pts
    m = np.isfinite(source_profile) & np.isfinite(source_levels)
    if m.sum() < 2:
        return np.full_like(target_levels, np.nan, dtype=np.float64)

    sp = source_profile[m].astype(np.float64, copy=False)
    sl = source_levels[m].astype(np.float64, copy=False)

    # strictly increasing
    if not np.all(np.diff(sl) > 0):
        order = np.argsort(sl)
        sl = sl[order]
        sp = sp[order]

    f = interp1d(
        sl, sp, bounds_error=False, fill_value="extrapolate", assume_sorted=True
    )
    tgt = np.asarray(target_levels, dtype=np.float64)
    out = f(tgt)

    # replace above ERA5 top with GCM column
    gm = np.isfinite(gcm_profile) & np.isfinite(gcm_levels)
    if gm.sum() >= 2:
        gl = np.asarray(gcm_levels[gm], dtype=np.float64)
        gp = np.asarray(gcm_profile[gm], dtype=np.float64)
        if not np.all(np.diff(gl) > 0):
            oo = np.argsort(gl)
            gl = gl[oo]
            gp = gp[oo]
        above = tgt > sl.max()
        if np.any(above):
            out[above] = np.interp(tgt[above], gl, gp)

    return out


def vertical_interpolation(
    source_da, source_levels_da, target_levels, gcm_profile, gcm_levels
):
    vdim_src = get_vdim(source_da)
    vdim_slev = get_vdim(source_levels_da)
    vdim_tgt = get_vdim(target_levels)
    vdim_gcm1 = get_vdim(gcm_profile)
    vdim_gcm2 = get_vdim(gcm_levels)

    out = xr.apply_ufunc(
        _interpolate_profile_1d,
        source_da,
        source_levels_da,
        target_levels,
        gcm_profile,
        gcm_levels,
        vectorize=True,
        input_core_dims=[[vdim_src], [vdim_slev], [vdim_tgt], [vdim_gcm1], [vdim_gcm2]],
        output_core_dims=[[vdim_tgt]],
        dask="parallelized",
        output_dtypes=[np.float64],
        join="override",
    )
    return out.assign_coords({vdim_tgt: target_levels[vdim_tgt]})


def standardize_coords_from(ds):
    """
    Standardizes the coordinate names in the given dataset.

    This function renames the coordinates in the dataset to a standard naming convention.
    Specifically, it renames 'longitude' to 'lon' and 'latitude' to 'lat' if they exist in the dataset.

    Parameters:
    ds (xarray.Dataset): The input dataset with coordinates to be standardized.

    Returns:
    xarray.Dataset: A new dataset with standardized coordinate names.
    """
    coord_map = {"longitude": "lon", "latitude": "lat"}
    new_ds = ds.copy()
    for old_name, new_name in coord_map.items():
        if old_name in new_ds.coords:
            new_ds = new_ds.rename({old_name: new_name})
    return new_ds


def correct_latitudes(ds):
    """Ensure latitudes are within bounds."""
    ds["lat"] = ds["lat"].clip(-90, 90)
    return ds


def ensure_bounds(ds):
    """Ensure that latitude and longitude bounds are present in the dataset."""
    if "lon_bnds" not in ds.variables and "lon" in ds.coords:
        lon = ds["lon"]
        lon_bnds = np.zeros((len(lon), 2))
        lon_bnds[:, 0] = lon - (lon[1] - lon[0]) / 2
        lon_bnds[:, 1] = lon + (lon[1] - lon[0]) / 2
        ds["lon_bnds"] = (("lon", "bnds"), lon_bnds)
        ds["lon"].attrs["bounds"] = "lon_bnds"

    if "lat_bnds" not in ds.variables and "lat" in ds.coords:
        lat = ds["lat"]
        lat_bnds = np.zeros((len(lat), 2))
        lat_bnds[:, 0] = lat - (lat[1] - lat[0]) / 2
        lat_bnds[:, 1] = lat + (lat[1] - lat[0]) / 2
        ds["lat_bnds"] = (("lat", "bnds"), lat_bnds)
        ds["lat"].attrs["bounds"] = "lat_bnds"

    return ds


def reduce_bounds(ds):
    """Ensure lat_bnds and lon_bnds are 2D (lat, bnds) and (lon, bnds)."""
    if "lat_bnds" in ds.variables and "time" in ds["lat_bnds"].dims:
        ds["lat_bnds"] = ds["lat_bnds"].isel(time=0)  # drop time dimension
    if "lon_bnds" in ds.variables and "time" in ds["lon_bnds"].dims:
        ds["lon_bnds"] = ds["lon_bnds"].isel(time=0)
    return ds


def regrid(source_ds, target_ds, method, weights_path, rename_dict):
    """
    Perform interpolation/regridding of a source dataset to a target dataset using the specified method.

    Parameters:
    -----------
    source_ds : xr.Dataset or xr.DataArray
        The source dataset or data array to be regridded.
    target_ds : xr.Dataset or xr.DataArray
        The target dataset or data array to which the source dataset will be regridded.
    method : str
        The regridding method to be used (e.g., 'bilinear', 'nearest_s2d', 'patch', etc.).
    weights_path : str
        The file path where the regridding weights will be saved or loaded from.
    rename_dict : dict
        A dictionary mapping original variable names in the source dataset to new names in the regridded dataset.

    Returns:
    --------
    xr.Dataset or xr.DataArray
        The regridded dataset or data array with variables renamed according to rename_dict.

    Raises:
    -------
    TypeError
        If the input source_ds is not an xarray DataArray or Dataset.
    """
    # source_ds = standardize_coords(source_ds)
    # target_ds = standardize_coords(target_ds)

    if isinstance(target_ds, xr.Dataset):
        if (
            "lon_bnds" not in target_ds.variables
            or "lat_bnds" not in target_ds.variables
        ):
            target_ds = ensure_bounds(target_ds)
    elif isinstance(target_ds, xr.DataArray):
        if "lon_bnds" not in target_ds.coords or "lat_bnds" not in target_ds.coords:
            target_ds = ensure_bounds(target_ds)

    if isinstance(source_ds, xr.Dataset):
        if (
            "lon_bnds" not in source_ds.variables
            or "lat_bnds" not in source_ds.variables
        ):
            source_ds = ensure_bounds(source_ds)
    elif isinstance(source_ds, xr.DataArray):
        if "lon_bnds" not in source_ds.coords or "lat_bnds" not in source_ds.coords:
            source_ds = ensure_bounds(source_ds)

    source_ds = reduce_bounds(source_ds)
    target_ds = reduce_bounds(target_ds)

    weights_file = Path(weights_path)
    reuse_weights = weights_file.is_file()
    regridder = xe.Regridder(
        source_ds,
        target_ds,
        method=method,
        filename=weights_path,
        reuse_weights=reuse_weights,
    )
    try:
        if isinstance(source_ds, xr.DataArray):
            regridded_data = regridder(source_ds)
            if source_ds.name in rename_dict:
                regridded_data.name = rename_dict[source_ds.name]
            return regridded_data

        elif isinstance(source_ds, xr.Dataset):
            regridded_data = regridder(source_ds)
            for var in list(regridded_data.data_vars):
                if var in rename_dict:
                    regridded_data = regridded_data.rename({var: rename_dict[var]})
            return regridded_data

    finally:
        # Avoid temporary in-memory structures leaking; keep the on-disk weights
        regridder._grid_in = None
        regridder._grid_out = None


def standardize_coords(ds):
    """Standardize coordinate names."""
    coord_map = {"lon": "longitude", "lat": "latitude", "lev": "level"}
    new_ds = ds.copy()

    for old_name, new_name in coord_map.items():
        if old_name in new_ds.coords:
            if new_name in new_ds.coords:
                if xr.all(new_ds[old_name] == new_ds[new_name]):
                    new_ds = new_ds.drop_vars(old_name)
            else:
                new_ds = new_ds.rename({old_name: new_name})

    return new_ds


def prepend_last_hist_if_needed(future_files, hist_files, chunks=None):
    # Open the future dataset
    future_ds = xr.open_mfdataset(future_files, combine="by_coords", chunks=chunks)
    first_time = future_ds.time.values[0]
    # Check if first time is not 00:00
    if str(first_time)[11:16] != "00:00":
        # Open the historical dataset
        hist_ds = xr.open_mfdataset(hist_files, combine="by_coords", chunks=chunks)
        last_hist = hist_ds.sel(time=hist_ds.time == hist_ds.time.max())
        # Concatenate along time
        future_ds = xr.concat([last_hist, future_ds], dim="time")
    return future_ds


# prepare last_hist_ds and future_ds for concat
def prepare_for_concat(last_hist_ds, future_ds):
    # normalize lev name
    # if "level" in last_hist_ds.dims and "lev" not in last_hist_ds.dims:
    #     last_hist_ds = last_hist_ds.rename({"level": "lev"})
    # if "level" in future_ds.dims and "lev" not in future_ds.dims:
    #     future_ds = future_ds.rename({"level": "lev"})

    # If lev arrays differ, drop vertical-metadata from the hist slice (cheap & safe if you don't need lev_bnds/b later)
    # Note: drop them from both datasets to avoid MergeError due to differing attributes/values.
    drop_vertical_meta = ["lev_bnds", "b", "b_bnds"]
    last_hist_ds = last_hist_ds.drop_vars(drop_vertical_meta, errors="ignore")
    future_ds = future_ds.drop_vars(drop_vertical_meta, errors="ignore")

    # drop other bound vars that often conflict
    last_hist_ds = last_hist_ds.drop_vars(
        ["lat_bnds", "lon_bnds", "orog"], errors="ignore"
    )
    future_ds = future_ds.drop_vars(["lat_bnds", "lon_bnds", "orog"], errors="ignore")

    # drop scalar coords/attrs that are not present in the other ds
    for c in list(last_hist_ds.coords):
        if c not in future_ds.coords and c not in future_ds.dims:
            last_hist_ds = last_hist_ds.drop_vars(c, errors="ignore")

    return last_hist_ds, future_ds


# clear
def load_target_files(target_path, infor, sinfor, version, var, year):
    """
    Find CMIP6 target files like:
    {target_path}/{infor}/{var}/{grid_label}/{version}/{var}_{infor}_*_{grid_label}_*.nc
    """
    # Recursive, but constrained by {infor}/{var}/ to avoid Amon/day pulling in.
    pattern = f"{target_path}/{infor}/{var}/g*/v*/{var}_*{year}*.nc"
    files = glob.glob(pattern, recursive=True)
    print(files)
    # Safeguard: if no files match, return empty list early
    if not files:
        raise FileNotFoundError(f"No files found for {pattern}. ")
    # Optional: filter to grid labels and subtable inside the filename to avoid cross-table hits
    # e.g. keep only filenames containing f"_{infor}_" and one of grid labels
    keep = []
    for f in files:
        base = os.path.basename(f)
        if f"_{infor}_" not in base:
            continue
        # If sinfor is provided (gn/gr/etc) prefer matches; otherwise accept any
        if sinfor and f"_{sinfor}_" not in base:
            continue
        keep.append(f)

    return sorted(keep)


# clear
def load_datasets_with_history(year, month, files, chunks, config, var, target_path):
    """
    Load datasets (future or historical) and, if the future starts at a non-zero hour,
    attempt to prepend the final timestep from the historical run specified by
    config['hist_target_path'].

    Logic:
     - Open future files (files).
     - If the first timestep hour != 0 and config contains 'hist_target_path' and
       hist path differs from target_path, search hist files for the needed time
       (first_time - 6h). If found, prepend that single timestep and return the
       concatenated dataset (then select year/month).
    """
    if not files:
        raise FileNotFoundError(f"No files found for {year}-{month:02d}. ")

    # open future dataset
    future_ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    if "time" not in future_ds.coords:
        # nothing to do
        return future_ds.sel(
            time=(future_ds["time"].dt.year == year)
            & (future_ds["time"].dt.month == month)
        )

    first_time = pd.to_datetime(future_ds.time.values[0])
    # if starts at midnight -> no action
    if first_time.hour == 0:
        # harmonize chunks
        if "time" in future_ds.dims:
            future_ds = future_ds.chunk(
                {"time": min(32, max(8, future_ds.sizes["time"]))}
            )
        return future_ds.sel(
            time=(future_ds["time"].dt.year == year)
            & (future_ds["time"].dt.month == month)
        )

    # need to prepend previous 6-hour step
    hist_root = config.get("hist_target_path")
    if not hist_root or hist_root == target_path:
        # no historical root provided or same path -> return future as-is
        if "time" in future_ds.dims:
            future_ds = future_ds.chunk(
                {"time": min(32, max(8, future_ds.sizes["time"]))}
            )
        return future_ds.sel(
            time=(future_ds["time"].dt.year == year)
            & (future_ds["time"].dt.month == month)
        )

    needed_time = np.datetime64(first_time - pd.Timedelta(hours=6))

    # look for candidate hist files in year and year-1 (covers files that span year-boundaries)
    cand_files = []
    for y in (year, year - 1):
        try:
            cand_files += load_target_files(
                hist_root,
                config["infor"],
                config.get("sinfor", None),
                config.get("version", None),
                var,
                y,
            )
        except Exception:
            continue

    cand_files = sorted(set(cand_files))
    last_hist_slice = None
    for hf in cand_files:
        try:
            # open minimal dataset (doesn't load data arrays until needed)
            ds_tmp = xr.open_dataset(hf)
            if "time" in ds_tmp.coords:
                # check if the file contains needed_time
                if np.any(ds_tmp["time"].values == needed_time):
                    # last_hist_slice = ds_tmp.sel(time=needed_time)
                    last_hist_slice = ds_tmp.sel(time=[needed_time])

                    # keep only coordinates and needed variables; ds_tmp will be closed by GC
                    break
        except Exception:
            continue

    if last_hist_slice is None:
        # nothing found; return future as-is
        if "time" in future_ds.dims:
            future_ds = future_ds.chunk(
                {"time": min(32, max(8, future_ds.sizes["time"]))}
            )
        return future_ds.sel(
            time=(future_ds["time"].dt.year == year)
            & (future_ds["time"].dt.month == month)
        )

    # ensure last_hist_slice has same variable names / coords as future_ds for concatenation
    # convert to dataset if DataArray
    if isinstance(last_hist_slice, xr.DataArray):
        last_hist_ds = last_hist_slice.to_dataset()
    else:
        last_hist_ds = last_hist_slice

    # --- NEW: normalize vertical coordinate names if necessary ---
    # Commonly files use 'lev' or 'level' — prefer 'lev' in this codebase
    def normalize_lev(ds):
        if "level" in ds.dims and "lev" not in ds.dims:
            ds = ds.rename({"level": "lev"})
        if "level" in ds.coords and "lev" not in ds.coords:
            ds = ds.rename({"level": "lev"})
        return ds

    last_hist_ds = normalize_lev(last_hist_ds)
    future_ds = normalize_lev(future_ds)

    last_hist_ds, future_ds = prepare_for_concat(last_hist_ds, future_ds)

    # 3) Make lev *identical* to avoid a union on concat
    if "lev" in last_hist_ds.dims and "lev" in future_ds.dims:
        last_hist_ds = last_hist_ds.assign_coords(lev=future_ds.lev)

    # 4) Match variable dim order (prevents xarray from reordering on concat)
    for v in set(last_hist_ds.data_vars) & set(future_ds.data_vars):
        last_hist_ds[v] = last_hist_ds[v].transpose(*future_ds[v].dims)

    xr.testing.assert_allclose(last_hist_ds.lev, future_ds.lev)
    combined = xr.concat(
        [last_hist_ds, future_ds],
        dim="time",
        data_vars="minimal",
        coords="minimal",
        join="exact",
        compat="equals",
    )
    # print("combined", combined)

    if "time" in combined.dims:
        combined = combined.chunk({"time": min(32, max(8, combined.sizes["time"]))})
    return combined.sel(
        time=(combined["time"].dt.year == year) & (combined["time"].dt.month == month)
    )


# clear
def load_datasets(year, month, files, chunks):
    """
    Load and filter datasets based on the specified year and month.

    Parameters:
    year (int): The year to filter the dataset.
    month (int): The month to filter the dataset.
    files (list of str): List of file paths to be loaded.
    chunks (dict): Dictionary specifying the chunk sizes for dask.

    Returns:
    xarray.Dataset: The filtered dataset containing data only for the specified year and month.
    """
    # print(files)
    if not files:
        raise FileNotFoundError(f"No files found for {year}-{month:02d}. ")
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    # Harmonize chunks a little
    if "time" in ds.dims:
        ds = ds.chunk({"time": min(32, max(8, ds.sizes["time"]))})
    return ds.sel(time=(ds["time"].dt.year == year) & (ds["time"].dt.month == month))


# clear
def process_year_month(config, year, month, selected_variables):
    """
    Processes data for a given year and month based on the provided configuration.

    This function loads datasets for selected variables and temperature/humidity variables,
    clips the latitude values to the range [-90, 90], and checks if there is any data available
    for the specified year and month. If no data is available, it raises a ValueError.
    Finally, it processes the geopotential height using the loaded datasets.

    Parameters:
    config (dict): Configuration dictionary containing paths and other settings.
    year (int): The year for which data is to be processed.
    month (int): The month for which data is to be processed.

    Returns:
    Any: The result of the `process_geopotential_height` function.

    Raises:
    ValueError: If no data is available for the specified year, month, and variable.
    """
    selected_variables = normalize_target_variables(selected_variables)
    print("selected_variables", selected_variables)
    tq_variables = ["ta", "hus"]

    # target_grids = {
    #     var: load_datasets(
    #         year,
    #         month,
    #         load_target_files(
    #             config["target_path"],
    #             config["infor"],
    #             config["sinfor"],
    #             config["version"],
    #             var,
    #             year,
    #         ),
    #         {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"},
    #     )
    #     for var in selected_variables
    # }
    # tq_grids = {
    #     var: load_datasets(
    #         year,
    #         month,
    #         load_target_files(
    #             config["target_path"],
    #             config["infor"],
    #             config["sinfor"],
    #             config["version"],
    #             var,
    #             year,
    #         ),
    #         {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"},
    #     )
    #     for var in tq_variables
    # }

    target_grids = {}
    for var in selected_variables:
        files = load_target_files(
            config["target_path"],
            config["infor"],
            config["sinfor"],
            config["version"],
            var,
            year,
        )
        target_grids[var] = load_datasets_with_history(
            year,
            month,
            files,
            {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"},
            config,
            var,
            config["target_path"],
        )

    tq_grids = {}
    for var in tq_variables:
        files = load_target_files(
            config["target_path"],
            config["infor"],
            config["sinfor"],
            config["version"],
            var,
            year,
        )
        tq_grids[var] = load_datasets_with_history(
            year,
            month,
            files,
            {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"},
            config,
            var,
            config["target_path"],
        )

    for var in selected_variables:
        target_grids[var]["lat"] = target_grids[var]["lat"].clip(-90, 90)
        if target_grids[var].sizes["time"] == 0:
            raise ValueError(f"No data available for {year}-{month} in variable {var}.")

    for var in tq_variables:
        tq_grids[var]["lat"] = tq_grids[var]["lat"].clip(-90, 90)
        if tq_grids[var].sizes["time"] == 0:
            raise ValueError(f"No data available for {year}-{month} in variable {var}.")

    return process_geopotential_height(
        config, target_grids, tq_grids, year, month, selected_variables[0]
    )


# clear
def process_geopotential_height(config, target_grids, tq_grids, year, month, variable):
    """
    Processes the geopotential height based on the configuration and target grids.

    Parameters:
    config (dict): Configuration dictionary containing various settings and paths.
    target_grids (dict): Dictionary containing target grid datasets.
    tq_grids (dict): Dictionary containing temperature and specific humidity grids.
    year (int): The year for which the processing is being done.
    month (int): The month for which the processing is being done.

    Returns:
    tuple: A tuple containing the processed geopotential height data, target grids, and tq grids.
    """
    if (
        target_grids[variable].lev.standard_name
        == "atmosphere_hybrid_height_coordinate"
    ):
        target_z_files = glob.glob(
            f"{config['target_path']}/fx/zfull/{config['sinfor']}/v*/zfull_fx_*.nc"
        )
        target_zfull = (
            xr.open_dataset(
                target_z_files[0], chunks={"lev": -1, "lat": -1, "lon": -1}
            )["zfull"].transpose("lev", "lat", "lon")
            # .persist()
        )
        return target_zfull, target_grids, tq_grids
    elif (
        target_grids[variable].lev.standard_name
        == "atmosphere_hybrid_sigma_pressure_coordinate"
    ):
        return compute_or_load_geopotential(
            config, target_grids, tq_grids, year, month, variable
        )


# clear
def compute_or_load_geopotential(config, target_grids, tq_grids, year, month, variable):
    """
    Compute or load geopotential height for a given year and month.

    Parameters:
    config (dict): Configuration dictionary containing paths and settings.
    target_grids (dict): Dictionary containing target grid data.
    tq_grids (dict): Dictionary containing temperature and specific humidity grids.
    year (int): Year for which to compute or load geopotential height.
    month (int): Month for which to compute or load geopotential height.

    Returns:
    tuple: A tuple containing:
        - target_zfull (xarray.DataArray): Geopotential height data.
        - target_grids (dict): Updated target grid data.
        - tq_grids (dict): Updated temperature and specific humidity grids.

    Raises:
    FileNotFoundError: If no orography files are found at the specified path.
    """
    if config["target_g_path"] == "None":
        target_t = tq_grids["ta"]
        # target_q = tq_grids["hus"]
        target_p = calculate_pressure_levels(target_t.ap, target_t.b, target_t.ps)
        # orog = xr.open_dataset(
        #     f"{config['target_path']}/fx/orog/{config['sinfor']}/v*/orog_fx_*.nc"
        # )["orog"]
        # Load orography from the target dataset
        orog_path = glob.glob(
            f"{config['target_orog_path']}/fx/orog/g*/v*/orog_fx_*.nc"
        )
        # Check if orog_path is empty
        if not orog_path:
            raise FileNotFoundError(
                f"No orography files found at {config['target_orog_path']}"
            )

        orog = xr.open_dataset(orog_path[0], chunks={"lat": -1, "lon": -1})["orog"]
        target_zfull = (
            compute_geopotential_height(target_p, target_t.ta, orog)
            .chunk({"time": 10, "lev": -1, "lat": -1, "lon": -1})
            .transpose("time", "lev", "lat", "lon")
            # .persist()
        )
    else:
        target_zg_files = [
            f
            for y in range(config["start_year"], config["end_year"] + 1)
            for f in sorted(glob.glob(f"{config['target_g_path']}/*_*{y}*.nc"))
        ]
        target_z = xr.open_mfdataset(
            target_zg_files, chunks={"time": 10, "lev": -1, "lat": -1, "lon": -1}
        )
        selected_target_z = target_z.sel(
            time=(target_grids[variable]["time"].dt.year == year)
            & (target_grids[variable]["time"].dt.month == month)
        )
        target_zfull = selected_target_z.rename({"z": "zfull"})  # .persist()

    return target_zfull, target_grids, tq_grids


# clear
def load_input_files(input_path, var, year, month):
    """
    Load input files matching a specific pattern.

    This function generates a file pattern based on the provided input path, variable name,
    year, and month, and returns a sorted list of file paths that match the pattern.

    Parameters:
    input_path (str): The base directory path where the files are located.
    var (str): The variable name to include in the file pattern.
    year (int): The year to include in the file pattern.
    month (int): The month to include in the file pattern (1-12).

    Returns:
    list: A sorted list of file paths that match the generated pattern.
    """
    pattern = f"{input_path}/{var}/{year}/{var}_*_{year}{month:02d}*.nc"
    return sorted(glob.glob(pattern))


# clear
def load_input_data(input_files):
    """
    Load input data from a list of NetCDF files and standardize coordinates.

    This function opens multiple NetCDF files specified in the input_files list,
    determines an appropriate chunking strategy based on the dimensions of the
    first file, and combines the datasets by coordinates. The resulting dataset
    is then standardized in terms of its coordinates.

    Parameters:
    -----------
    input_files : list of str
        List of file paths to the NetCDF files to be loaded.

    Returns:
    --------
    xarray.Dataset
        The combined and standardized dataset.
    """
    # with xr.open_dataset(input_files[0]) as temp_ds:
    #     # Determine chunking strategy based on dimensions
    #     chunks = {dim: "auto" for dim in temp_ds.dims}
    chunks = {dim: "auto" for dim in xr.open_dataset(input_files[0]).dims}
    ds = xr.open_mfdataset(input_files, combine="by_coords", chunks=chunks)
    return standardize_coords(ds)


def get_vertical_dim_name(da):
    for dim in da.dims:
        if dim in ("lev", "level", "height", "plev"):
            return dim
    raise ValueError("Vertical dimension not found.")


def regrid_and_interpolate(
    config, input_var, target_var, year, month, target_grids, target_zfull_per, tq_grids
):
    """
    Regrids and interpolates input data to match the target grid and vertical levels.

    Parameters:
    -----------
    config : dict
        Configuration dictionary containing paths, model information, and other settings.
    input_var : str
        The variable name in the input data (e.g., 'u', 'v', 'q').
    target_var : str
        The variable name in the target data (e.g., 'ua', 'va').
    year : int
        The year of the data to process.
    month : int
        The month of the data to process.
    target_grids : dict
        Dictionary containing target grid datasets.
    target_zfull_per : xarray.Dataset
        Dataset containing the target vertical levels.
    tq_grids : dict
        Dictionary containing temperature and specific humidity grids.

    Returns:
    --------
    xarray.Dataset
        Interpolated dataset with the same structure as the target grid.
    """
    method = (
        "bilinear"
        if input_var in ["u", "v"]
        else "conservative" if input_var == "q" else "bilinear"
    )
    print(method)
    if config["input_model"] == "reanalysis":
        input_files_pattern = glob.glob(
            f"{config['input_path']}/{input_var}/{year}/{input_var}_*_{year}{month:02d}*.nc"
        )
        weights_path = f"{config['output_path']}/weights_{config['input_model']}_to_{config['gname']}_{target_var}.nc"
        weights_path_z = f"{config['output_path']}/weights_zfull_{config['input_model']}_to_{config['gname']}_{method}.nc"
        weights_path_va_g_era5 = f"{config['output_path']}/weights_{target_var}_{config['input_model']}_to_{config['gname']}_g_obs.nc"
        weights_path_va_target_var = f"{config['output_path']}/weights_{target_var}_{config['input_model']}_to_{config['gname']}_target_var.nc"
        # weights_path_wind = f"{config['output_path']}/weights_{target_var}_{config['input_model']}_to_{config['gname']}_wind_{method}.nc"
    else:
        input_files_pattern = glob.glob(
            f"{config['input_path']}/{config['input_infor']}/{target_var}/{config['input_sinfor']}/{config['input_version']}/{target_var}_*{year}*.nc"
        )
        weights_path = f"{config['output_path']}/weights_{config['input_gname']}_to_{config['gname']}_{target_var}.nc"
        weights_path_z = f"{config['output_path']}/weights_zfull_{config['input_gname']}_to_{config['gname']}_{method}.nc"
        weights_path_va_g_era5 = f"{config['output_path']}/weights_{target_var}_{config['input_gname']}_to_{config['gname']}_g_obs.nc"
        weights_path_va_target_var = f"{config['output_path']}/weights_{target_var}_{config['input_gname']}_to_{config['gname']}_target_var.nc"
        # weights_path_wind = f"{config['output_path']}/weights_{target_var}_{config['input_gname']}_to_{config['gname']}_wind_{method}.nc"

    target_ds = standardize_coords_from(target_grids[target_var])
    target_ds = correct_latitudes(target_ds)
    target_ds = ensure_bounds(target_ds)
    # print("target_ds in line 1359", target_ds)
    rename_dict = {input_var: target_var}

    do = standardize_coords_from(load_input_data(input_files_pattern))
    do_corrected = correct_latitudes(do).sel(
        time=do["time"].dt.hour.isin([0, 6, 12, 18])
    )
    # print('do_Corrected')
    # print(do_corrected)
    # print('zfull_name', target_zfull_per.lev.standard_name)
    # print('target_ds',target_ds)
    # target_ds.va[0,:,140:,150].plot(vmin= -40, vmax=40)
    # plt.show()
    # if target_zfull_per.lev.standard_name == "atmosphere_hybrid_height_coordinate":
    # do_regridded = standardize_coords_from(do_regridded)
    # do_corrected = ensure_bounds(do_corrected)
    # print('do_corrected_ensure_bounds', do_corrected)

    do_regridded = regrid(
        do_corrected, tq_grids["ta"], method, weights_path, rename_dict
    )
    # print('do_regridded')
    # print(do_regridded)
    if do_regridded is None:
        raise RuntimeError(
            "Regridding failed: do_regridded is None. Check input data and regridder configuration."
        )

    sliced_do = (
        do_regridded.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({"level": -1})
        # .persist()
    )
    # print('sliced_ds', sliced_do)
    # sliced_do.va[0,:,140:,150].plot(vmin= -40, vmax=40)
    # plt.show()

    # geopotential
    if config["input_z_path"] not in [None, "None"]:
        input_z_files = glob.glob(
            f"{config['input_z_path']}/{year}/*{year}{month:02d}*.nc"
        )
        g_era5 = standardize_coords_from(load_input_data(input_z_files))
        g_era5 = g_era5.sel(
            time=(g_era5["time"].dt.year == year) & (g_era5["time"].dt.month == month)
        )
        g_era5_resampled = correct_latitudes(g_era5).sel(
            time=g_era5["time"].dt.hour.isin([0, 6, 12, 18])
        )
    else:
        result = process_year_month(config, year, month, target_var)
        if result is not None:
            g_era5_resampled = standardize_coords_from(result[0])

    target_zfull_per = standardize_coords_from(target_zfull_per)

    if isinstance(g_era5_resampled, xr.DataArray):
        g_era5_resampled = g_era5_resampled.to_dataset(name="zfull")

    old_var_name = list(g_era5_resampled.data_vars)[0]
    rename_dict_z = {old_var_name: "zfull"}

    if target_var == "va":
        g_era5_regridded = regrid(
            g_era5_resampled,
            tq_grids["ta"],
            method,
            weights_path_va_g_era5,
            rename_dict_z,
        )
    else:
        g_era5_regridded = regrid(
            g_era5_resampled,
            tq_grids["ta"],
            method,
            weights_path_z,
            rename_dict_z,
        )

    if not target_zfull_per.lat.equals(
        target_ds.lat
    ) or not target_zfull_per.lon.equals(target_ds.lon):

        target_ds = regrid(
            target_ds,
            tq_grids["ta"],
            method,
            weights_path_va_target_var,
            rename_dict_z,
        )

    # if target_zfull_per.lev.standard_name == "atmosphere_hybrid_height_coordinate":
    g_era5_regridded = standardize_coords_from(g_era5_regridded)
    # Selected area
    sliced_g_era5_regridded = g_era5_regridded.sel(
        lat=slice(config["lat_min"], config["lat_max"]),
        lon=slice(config["lon_min"], config["lon_max"]),
    )

    # Perform calculations
    g_era5 = sliced_g_era5_regridded["zfull"]  # Assuming 'zfull' is geopotential height

    vdim = get_vertical_dim_name(g_era5)

    if config["input_model"] == "reanalysis":
        Z_era5 = geopotential_to_geopotential_height(g_era5)
        Z_era5_per = Z_era5.chunk({vdim: -1})
    else:
        Z_era5_per = g_era5.chunk({vdim: -1})

    if target_ds is None:
        raise RuntimeError(
            "Interpolation failed: target_ds is None. Check input data and interpolation configuration."
        )

    if target_var in ["ua", "va"]:
        # print('target_zfull_per_ua_va', target_zfull_per)
        # print('target_ds_ua_va', target_ds)
        # gcm_z_data_interp = regrid(target_zfull_per, target_ds, method, weights_path_wind, {"zfull": "zfull"})
        # gcm_z_data_interp = standardize_coords_from(gcm_z_data_interp)
        gcm_z_data_interp = target_zfull_per.copy()
        # print(gcm_z_data_interp)
        # print(target_ds)
        if not gcm_z_data_interp.lon.equals(target_ds.lon):
            # Assume hus and ta have the target longitude values, and they are already loaded
            # target_lon = target_ds.lon if "lon" in target_ds else gcm_z_data_interp.lon
            gcm_z_data_interp = gcm_z_data_interp.interp(
                lon=target_ds.lon, method="linear", kwargs={"fill_value": "extrapolate"}
            )
            # Assign the adjusted longitude values to ua or va
            gcm_z_data_interp = gcm_z_data_interp.assign_coords(lon=target_ds.lon)

        if not gcm_z_data_interp.lat.equals(target_ds.lat):
            # target_lat = target_ds.lat if "lat" in target_ds else gcm_z_data_interp.lat
            gcm_z_data_interp = gcm_z_data_interp.interp(
                lat=target_ds.lat, method="linear", kwargs={"fill_value": "extrapolate"}
            )
            gcm_z_data_interp = gcm_z_data_interp.assign_coords(lat=target_ds.lat)

        vdim = get_vertical_dim_name(gcm_z_data_interp)
        sliced_gcm_z_data_interp = gcm_z_data_interp.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({vdim: -1})
        sliced_target_ds = target_ds.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({vdim: -1})
        # print('sliced_do[target_var]', sliced_do[target_var])
        # print('Z_era5_per', Z_era5_per)
        # print('sliced_gcm_z_data_interp', sliced_gcm_z_data_interp)

        interpolated_ds = vertical_interpolation(
            sliced_do[target_var],
            Z_era5_per,
            sliced_gcm_z_data_interp,
            sliced_target_ds[target_var],
            sliced_gcm_z_data_interp,
        )
    else:
        vdim = get_vertical_dim_name(target_zfull_per)
        sliced_target_zfull = target_zfull_per.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({vdim: -1})
        sliced_target_ds = target_ds.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({vdim: -1})
        # print(sliced_do[target_var])
        # print(Z_era5_per)
        # print(sliced_target_zfull)
        # print(sliced_target_ds[target_var])
        interpolated_ds = vertical_interpolation(
            sliced_do[target_var],
            Z_era5_per,
            sliced_target_zfull,
            sliced_target_ds[target_var],
            sliced_target_zfull,
        )

    vdim = get_vertical_dim_name(interpolated_ds)
    interpolated_ds = interpolated_ds.transpose("time", vdim, "lat", "lon").chunk(
        {vdim: -1}
    )

    if target_var in ["ua", "va"]:
        if not interpolated_ds.lev.equals(sliced_target_ds[target_var].lev):
            # target_lev = (
            #     interpolated_ds.lev if "lev" in target_ds else sliced_target_ds.lev
            # )
            target_lev = sliced_target_ds[target_var].lev
            sliced_target_ds = sliced_target_ds.interp(lev=target_lev)
            sliced_target_ds = sliced_target_ds.assign_coords(lev=target_lev)
    # if year == 2015 and month == 1:
    #     interpolated_ds_adjusted = interpolated_ds.persiste()
    # else:
    #     interpolated_ds_adjusted = xr.where(
    #         interpolated_ds.isnull(), sliced_target_ds[target_var], interpolated_ds
    #     ).persist()

    return interpolated_ds


# clear
def process_input_variable(
    config, input_var, target_var, year, month, target_grids, target_zfull, tq_grids
):
    """
    Processes the input variable by regridding and interpolating it to the target grids and saves the result.

    Parameters:
    config (dict): Configuration dictionary containing various settings and paths.
    input_var (str): Name of the input variable to be processed.
    target_var (str): Name of the target variable after processing.
    year (int): Year of the data to be processed.
    month (int): Month of the data to be processed.
    target_grids (xarray.Dataset): Target grids for regridding.
    target_zfull (xarray.DataArray): Full vertical coordinate for the target grids.
    tq_grids (xarray.Dataset): Grids for temperature and specific humidity.

    Returns:
    xarray.DataArray: Interpolated data array with the target variable.
    """
    interpolated_ds = regrid_and_interpolate(
        config, input_var, target_var, year, month, target_grids, target_zfull, tq_grids
    )
    interpolated_era5_da = xr.DataArray(
        interpolated_ds,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": interpolated_ds.time,
            "lev": interpolated_ds.lev,
            "lat": interpolated_ds.lat,
            "lon": interpolated_ds.lon,
        },
        name=target_var,
    )

    interpolated_era5_da.attrs["interpolation_to"] = config["gname"]
    output_file = f"{config['output_path']}/{target_var}_{config['input_model'] if config['input_model'] == 'reanalysis' else config['input_gname']}_to_{config['gname']}_{year}-{month:02}.nc"
    save_interpolated_data(interpolated_era5_da, output_file)
    print(f"Saved regridded data for {target_var} {year}-{month:02} to {output_file}")
    return interpolated_era5_da


# clear
def save_interpolated_data(interpolated_era5_da, output_file):
    """
    Save the interpolated ERA5 data to a NetCDF file.

    Parameters:
    interpolated_era5_da (xarray.DataArray): The interpolated ERA5 data array to be saved.
    output_file (str): The path to the output NetCDF file.

    Returns:
    None
    """
    with ProgressBar():
        print(f"Writing to {output_file}")
        interpolated_era5_da.to_netcdf(output_file, compute=False).compute()


def map_obs_name(target_var):
    return {"hus": "q", "ta": "t", "ua": "u", "va": "v"}[target_var]


# End of the functions -------------------------------------------------------


def main(config_path, var_interp, override_ncpus=None, override_mem=None):
    """
    Main function to perform interpolation of variables from observational data to GCM (General Circulation Model) grid.

    Args:
        config (dict): Configuration dictionary containing the following keys:
            - "var_interp" (str): Variable to interpolate (e.g., "ua", "va", "ta", "hus").
            - "startyear_h" (int): Start year for the interpolation.
            - "endyear_h" (int): End year for the interpolation.
            - "output_path" (str): Path to the directory where output files will be saved.
            - "input_model" (str): Model type of the input data (e.g., "reanalysis").

    Returns:
        None
    """
    config = load_config(config_path)
    target_names = normalize_target_variables(var_interp)
    if len(target_names) != 1:
        raise ValueError(
            "This script currently processes one target variable per invocation."
        )

    target_var = target_names[0]

    # Extract resources from config
    ncpus = config.get("resources", {}).get("ncpus", 4)
    mem_gb = config.get("resources", {}).get("mem_gb", 16)

    # Override from command-line if provided
    if override_ncpus:
        ncpus = override_ncpus
    if override_mem:
        mem_gb = override_mem

    # Dask performance settings
    dask.config.set(
        {
            "array.slicing.split_large_chunks": True,
            "array.chunk-size": "64MiB",
            "optimization.fuse.active": True,
        }
    )

    # Start Dask client
    client = setup_client(ncpus, mem_gb)
    try:
        print("[INFO] Starting interpolation for variable:", target_var)
        print(f"[INFO] CMIP6 root: {config['target_path']} (table={config['gname']})")

        if target_var not in ["ua", "va", "ta", "hus"]:
            raise ValueError(
                f"Unsupported target variable '{target_var}'. "
                "Expected one of: ua, va, ta, hus."
            )

        # Define start and end years
        start_year = config["startyear_h"]
        end_year = config["endyear_h"]

        # Create output directory if it doesn't exist
        os.makedirs(config["output_path"], exist_ok=True)

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                output_file = f"{config['output_path']}/{target_var}_{config['input_model'] if config['input_model'] == 'reanalysis' else config['input_gname']}_to_{config['gname']}_{year}-{month:02}.nc"
                if os.path.exists(output_file):
                    print(
                        f"[INFO] Output file already exists: {output_file}. Skipping..."
                    )
                    continue

                result = process_year_month(config, year, month, target_names)
                if config["input_model"] == "reanalysis":
                    var_obs = map_obs_name(target_var)
                else:
                    var_obs = target_var

                process_input_variable(
                    config,
                    var_obs,
                    target_var,
                    year,
                    month,
                    result[1] if result else None,
                    result[0] if result else None,
                    result[2] if result else None,
                )
    finally:
        client.close()
        gc.collect()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.yp, args.var, args.ncpus, args.mem)

    print("All done!")
