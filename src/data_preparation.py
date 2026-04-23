"""
Data Preparation Module for SDMBCv2

This script provides a comprehensive set of functions used for preparing climate data
for bias correction within the SDMBCv2 framework. It includes functionalities to load,
preprocess, and reshape GCM and observational data to ensure the correct format and
structure required for the bias correction process. This module is intended to support
both historical and future climate model data as well as observational datasets.

Main Features of the Script:
- **Configuration Management**: Load configuration settings from a YAML file to customize
  parameters for different experiments.
- **Data Loading and Preprocessing**: Load GCM, observational, and future climate data,
  and preprocess them for bias correction, including operations like slicing, combining
  multiple variables, and rescaling.
- **Temporal Adjustments**: Convert between different temporal resolutions (daily to
  6-hourly) using fractional scaling to preserve physical characteristics.
- **Spatial Adjustments**: Split the spatial domain into tiles, apply boundary corrections,
  and handle rescaling of GCM data to better match observational data.
- **Validation and Verification**: Validate data integrity, such as ensuring the absence
  of NaN values and verifying consistency between input files and expected parameters.

The module is designed to facilitate efficient data preparation, including handling large-scale
datasets with Dask for parallel processing, and accommodating different spatial and temporal
resolutions.

Usage:
This script is not intended to be run directly but rather imported into other scripts
as part of the SDMBCv2 workflow. The functions provide modular capabilities to handle
a variety of preprocessing steps for GCM and observational data.

Modules Imported:
- **xarray, numpy, pandas**: For numerical and array manipulations commonly used in climate data.
- **Dask**: To enable parallel computation, particularly useful for large-scale climate model datasets.
- **SDMBCv2 Specific Functions**: To support bias correction, variable extraction, and other key processes.
"""

import glob
import logging
import math
import re

import dask  # type: ignore

# import dask.array as da  # type: ignore
import numpy as np  # arrays and matrix math # type: ignore
import pandas as pd  # type: ignore
import xarray as xr  # type: ignore

# import yaml  # type: ignore

# from config import config

# Suppress INFO and lower-level logs
logging.getLogger("flox").setLevel(logging.WARNING)
logging.getLogger("xarray").setLevel(logging.WARNING)
logging.getLogger("dask").setLevel(logging.WARNING)

# import os
# import sys


# ==========================================================================#
# Set current folder as working directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# with open(f"../user_input_test.yaml", "r") as file:
#     config_data = yaml.safe_load(file)


# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)


# config = Config(**config_data)
# sys.path.append("/scratch/dm6/yk8692/sdmbc/")


def validate_inputs(lat_min, lat_max):
    """
    Validate latitude inputs to ensure they fall within the correct range.

    Args:
        lat_min (float): Minimum latitude value.
        lat_max (float): Maximum latitude value.

    Raises:
        ValueError: If latitude values are not within the range -90 to 90.
    """

    if not (-90 <= lat_min <= 90) or not (-90 <= lat_max <= 90):
        raise ValueError("Latitude values must be between -90 and 90.")


def verify_output(file_path):
    """
    Verify if the output file exists and can be opened.

    Args:
        file_path (str): Path to the file to be verified.

    Prints:
        Verification status message.
    """

    try:
        xr.open_dataset(file_path)
        print(f"Successfully verified the output file: {file_path}")
    except IOError:
        print(f"Verification failed for the file: {file_path}")


def initialize_data_arrays(dims_time, dims_lat, dims_lon, var_list):
    """
    Initialize empty data arrays for the given dimensions and variables.

    Args:
        dims_time (int): Number of time steps.
        dims_lat (int): Number of latitude points.
        dims_lon (int): Number of longitude points.
        var_list (list): List of variable names.

    Returns:
        dict: Dictionary of initialized arrays with variable names as keys.
    """

    return {var: np.zeros((dims_time, dims_lat, dims_lon)) for var in var_list}


# Function to extract the year range from the filename
def extract_years(filename):
    """
    Extract the start and end years from the filename.

    Args:
        filename (str): The filename containing the year information.

    Returns:
        tuple: Start and end year extracted from the filename, or None if not found.
    """

    match = re.search(r"_(\d{8})-(\d{8})\.nc", filename)
    if match:
        start = int(match.group(1)[:4])
        end = int(match.group(2)[:4])
        return start, end
    return None, None


# Function to check if the file overlaps with the desired period
def is_within_period(file_start, file_end, target_start, target_end):
    """
    Check if the file's period overlaps with the target period.

    Args:
        file_start (int): Start year of the file.
        file_end (int): End year of the file.
        target_start (int): Target start year.
        target_end (int): Target end year.

    Returns:
        bool: True if the file period overlaps with the target period, False otherwise.
    """

    # True if file period overlaps with the target period
    if file_start is None or file_end is None:
        return False
    return not (file_end < target_start or file_start > target_end)


# Function to extract the year range from the filename
def extract_years_remap(filename):
    """
    Extract the start and end years from filenames that include a date range.

    This matches YYYYMMDD-YYYYMMDD anywhere in the filename, regardless of extra suffixes
    like `_remapbil`, `_remapped`, etc.

    Args:
        filename (str): The filename containing the year information.

    Returns:
        tuple: Start and end year extracted from the filename, or (None, None) if not found.
    """

    # Match ..._YYYYMMDD-YYYYMMDD_... anywhere in the name
    m = re.search(r"_(\d{8})-(\d{8})(?:_|\.|$)", filename)
    if m:
        start = int(m.group(1)[:4])
        end = int(m.group(2)[:4])
        return start, end
    return None, None


# def generate_file_paths(
#     base_path,
#     variable,
#     infor,
#     gname,
#     period,
#     cinfor,
#     sinfor,
#     version,
#     start_year,
#     end_year,
# ):
#     """
#     Generate file paths for given parameters within a specified year range.

#     Args:
#         base_path (str): Base directory path.
#         variable (str): Variable name.
#         infor, gname, period, cinfor, sinfor, version (str): Metadata strings.
#         start_year (int): Start year.
#         end_year (int): End year.

#     Returns:
#         list: List of generated file paths.
#     """

#     file_paths = []
#     if config.bc_boundary == "lateral":
#         for year in range(start_year, end_year + 1):
#             # For models that start in January at 06:00 and require the previous December
#             if gname in ["ACCESS-ESM1-5"]:
#                 if year == start_year:
#                     # Include December of the year before start_year if focusing on a period starting from 1982 or later
#                     prev_dec_path = (
#                         f"{base_path}/{variable}/{sinfor}/v{version}/"
#                         f"{variable}_{infor}_{gname}_*_{cinfor}_{sinfor}_{year-1}*.nc"
#                     )
#                     file_paths.extend(glob.glob(prev_dec_path))

#             # Pattern for the main year of interest, adjusted to include the broadest possible match
#             file_path_pattern = (
#                 f"{base_path}/{variable}/{sinfor}/v{version}/"
#                 f"{variable}_{infor}_{gname}_*_{cinfor}_{sinfor}_{year}*.nc"
#             )
#             matching_file_paths = glob.glob(file_path_pattern)
#             file_paths.extend(matching_file_paths)
#     else:
#         file_path_pattern = glob.glob(f"{base_path}/{variable}_*_remapped.nc")
#         # Filter files that overlap with the desired year range
#         filtered_files = [
#             f
#             for f in file_path_pattern
#             if is_within_period(*extract_years_remap(f), start_year, end_year)
#         ]

#         file_paths.extend(filtered_files)

#     return file_paths


def parse_start_hour_from_filename(fname):
    match = re.search(r"_(\d{12})-(\d{12})\.nc$", fname)
    if not match:
        return None
    start_str = match.group(1)
    return int(start_str[8:10])  # hh from yyyymmddhhmm


def covers_december(file_path, year_minus_1):
    match = re.search(r"_(\d{12})-(\d{12})\.nc$", file_path)
    if not match:
        return False
    start_str, end_str = match.groups()
    start_date = pd.to_datetime(start_str, format="%Y%m%d%H%M")
    end_date = pd.to_datetime(end_str, format="%Y%m%d%H%M")
    return start_date.year <= year_minus_1 and end_date >= pd.Timestamp(
        f"{year_minus_1}-12-01"
    )


def is_shifted_gcm(file_list):
    """Return True if GCM uses shifted hours like 03/09/15/21 (not 00/06/12/18)."""
    for f in file_list:
        try:
            ds = xr.open_dataset(f, decode_times=True)
            hours = sorted(set(ds.time.dt.hour.values))
            ds.close()
            return set(hours).issubset({3, 9, 15, 21})  # common in NorESM2-MM
        except Exception:
            continue
    return False  # default: not shifted


def needs_prev_december(file_list, target_year):
    """
    Return True if model follows 00/06/12/18 and first file lacks 00:00.
    """
    if is_shifted_gcm(file_list):
        return False  # shift-based GCMs like NorESM2-MM don't need Dec files

    for f in file_list:
        if str(target_year) in f:
            hour = parse_start_hour_from_filename(f)
            if hour == 0:
                return False  # starts at 00:00
    return True


def generate_file_paths(
    config,
    base_path,
    variable,
    infor,
    gname,
    period,
    cinfor,
    sinfor,
    version,
    start_year,
    end_year,
):
    """
    Generate CMIP6 GCM input file paths and optionally include the previous year's
    December file if needed (for 00-based GCMs only).

    Only models that operate on the 00/06/12/18 UTC cycle are considered.
    """

    file_paths = []

    if config.bc_boundary == "lateral":
        for year in range(start_year, end_year + 1):
            year_pattern = (
                f"{base_path}/{variable}/g?/v*/"
                f"{variable}_{infor}_{gname}_*_{year}*.nc"
            )
            this_year_files = sorted(glob.glob(year_pattern))

            if year == start_year and needs_prev_december(this_year_files, year):
                prev_year_pattern = (
                    f"{base_path}/{variable}/g?/v*/"
                    f"{variable}_{infor}_{gname}_*_{year-1}*.nc"
                )
                prev_year_files = glob.glob(prev_year_pattern)
                december_files = [
                    f for f in prev_year_files if covers_december(f, year - 1)
                ]
                file_paths.extend(december_files)

            file_paths.extend(this_year_files)

    else:
        # file_path_pattern = glob.glob(f"{base_path}/{variable}_*_remap*.nc")
        # file_path_pattern = glob.glob(f"{base_path}/{variable}_*.nc")
        file_path_pattern = glob.glob(f"{base_path}/{variable}_*_*[rR]emap*.nc")
        filtered_files = [
            f
            for f in file_path_pattern
            if is_within_period(*extract_years_remap(f), start_year, end_year)
        ]
        file_paths.extend(filtered_files)

    return sorted(file_paths)


def generate_file_paths_obs(
    config,
    base_path,
    variable,
    gname,
    start_year,
    end_year,
):
    """
    Generate observational file paths for given parameters within a specified year range.

    Args:
        base_path (str): Base directory path.
        variable (str): Variable name.
        gname (str): GCM name.
        start_year (int): Start year.
        end_year (int): End year.

    Returns:
        list: List of generated file paths for observation data.
    """

    # output_file = f"{output_path}/{target_var}_{gname}_{year}-{month:02}.nc"
    file_paths = []
    for year in range(start_year, end_year + 1):
        # Pattern for the main year of interest, adjusted to include the broadest possible match
        file_path_pattern = f"{base_path}/{variable}_*_to_{gname}_{year}*.nc"
        matching_file_paths = glob.glob(file_path_pattern)
        file_paths.extend(matching_file_paths)

    return sorted(file_paths)


def slice_data(config, ds):
    """
    Slice the dataset to a specific latitude and longitude range.

    Args:
        ds (xarray.Dataset): Dataset to be sliced.

    Returns:
        xarray.Dataset: Sliced dataset.
    """

    try:
        sliced_dataset = ds.transpose("time", "lat", "lon").sel(
            lat=slice(config.lat_min, config.lat_max),
            lon=slice(config.lon_min, config.lon_max),
        )
        return sliced_dataset
    except IOError as e:
        print(f"An error occurred when loading the dataset {print(ds)}: {e}")
        raise


def load_and_slice_data(file_paths):
    """
    Load and slice data from multiple files.

    Args:
        file_paths (list of str): List of file paths.

    Returns:
        xarray.Dataset: Loaded and sliced dataset.
    """

    # First, check each file individually
    for file_path in file_paths:
        try:
            _ = xr.open_dataset(file_path)
        except IOError as e:
            print(f"Error loading file: {file_path}")
            raise e  # Re-raise the exception to stop the process

    # If all files are fine, proceed to open all with mfdataset
    try:
        dataset = xr.open_mfdataset(
            file_paths,
            concat_dim="time",
            combine="nested",
            parallel=True,
            preprocess=slice_data,
            chunks={"time": 1000, "lat": -1, "lon": -1},
        )
        return dataset
    except IOError as e:
        print(f"An error occurred when loading the dataset with open_mfdataset: {e}")
        raise


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
    # print(file_paths)
    ds = xr.open_mfdataset(
        file_paths,
        combine="by_coords",
        chunks={"time": "auto", "lat": "auto", "lon": "auto"},
    )
    # print(ds)
    if config.bc_boundary == "lateral":
        ds_sel = ds[var_name].isel(lev=level_index)
    else:
        ds_sel = ds[var_name]

    # A one-grid-cell test often uses zero-width bounds. Some ACCESS variables
    # are on staggered grids, so exact slicing can return an empty dimension.
    # Scalar variables use the nearest source point. Wind variables keep a
    # small local stencil so they can still be linearly interpolated onto the
    # scalar grid in preprocess_and_save_gcm().
    if float(lat_range[0]) == float(lat_range[1]) and float(lon_range[0]) == float(lon_range[1]):
        target_lat = float(lat_range[0])
        target_lon = float(lon_range[0])
        if var_name in ["ua", "va"]:
            lat_values = ds_sel["lat"].values
            lon_values = ds_sel["lon"].values
            lat_order = np.argsort(np.abs(lat_values - target_lat))[: min(3, len(lat_values))]
            lon_order = np.argsort(np.abs(lon_values - target_lon))[: min(3, len(lon_values))]
            lat_indexers = sorted(lat_order.tolist())
            lon_indexers = sorted(lon_order.tolist())
            ds_sel = ds_sel.isel(lat=lat_indexers, lon=lon_indexers)
        else:
            ds_sel = ds_sel.sel(
                lat=[target_lat],
                lon=[target_lon],
                method="nearest",
            )
    else:
        ds_sel = ds_sel.sel(lat=slice(*lat_range), lon=slice(*lon_range))

    ds_sel = ds_sel.sel(time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31"))
    # ds_sel = ds[var_name].sel(lat=slice(*lat_range), lon=slice(*lon_range))
    return ds_sel


def contains_nan(data):
    """
    Check if an xarray Dataset or DataArray contains any NaN values.

    Parameters:
    - data: xarray.Dataset or xarray.DataArray to be checked for NaN values.

    Returns:
    - bool: True if there are any NaN values, False otherwise.
    """
    # Check if the input is a Dataset
    if isinstance(data, xr.Dataset):
        # Iterate through all DataArrays in the Dataset
        for var in data:
            # If any value in the DataArray is NaN, return True
            if data[var].isnull().any():
                raise TypeError("Input Datasets must not contains any NaN values")
    # If the input is a DataArray, directly check for NaN values
    elif isinstance(data, xr.DataArray):
        if data.isnull().any():
            raise TypeError("Input DataArray must not contains any NaN values")
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray.")

    # If no NaN values were found, return False
    return "No NaN values"


def is_leap_year(year):
    """
    Determine whether a given year is a leap year.

    Args:
        year (int): Year to check.

    Returns:
        bool: True if the year is a leap year, otherwise False.
    """

    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def generate_dates(start_year, end_year):
    """
    Generate a complete datetime index for the given year range.

    Args:
        start_year (int): Start year.
        end_year (int): End year.

    Returns:
        pandas.DatetimeIndex: Complete date range for the given years.
    """

    return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")


def extract_and_reshape_xr(ds_selected, start_year, end_year):
    """
    Extract and reshape the given dataset to a structured year-month-day format.

    Args:
        ds_selected (xarray.Dataset): Dataset to be reshaped.
        start_year (int): Start year for reshaping.
        end_year (int): End year for reshaping.

    Returns:
        xarray.Dataset: Reshaped dataset.
    """

    date_index = generate_dates(start_year, end_year)
    reshaped_data = {}

    for var in ds_selected.data_vars:
        # Convert to a numpy array
        # var_np = ds_selected[var].reindex(time=date_index, fill_value=np.nan).values
        var_np = ds_selected[var].reindex(time=date_index, fill_value=0.0).values

        # Preallocate reshaped array
        # reshaped_var = np.full((end_year - start_year + 1, 12, 31, *var_np.shape[1:]), np.nan)
        reshaped_var = np.full(
            (end_year - start_year + 1, 12, 31, *var_np.shape[1:]), 0.0
        )

        # Fill in the reshaped array
        for year_idx, year in enumerate(range(start_year, end_year + 1)):
            for month in range(1, 13):
                days_in_month = (
                    29
                    if month == 2 and is_leap_year(year)
                    else (
                        28
                        if month == 2
                        else 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30
                    )
                )
                loc = date_index.get_loc(f"{year}-{month:02d}-01")
                if isinstance(loc, slice):
                    if loc.start is None:
                        raise TypeError("Slice start is None, cannot determine index")
                    start_idx = loc.start
                elif isinstance(loc, np.ndarray):
                    # Find the first True index if it's a boolean array
                    true_indices = np.where(loc)[0]
                    if true_indices.size == 0:
                        raise TypeError("No True index found in boolean array")
                    start_idx = true_indices[0]
                elif isinstance(loc, (int, np.integer)):
                    start_idx = int(loc)
                else:
                    raise TypeError(f"Unexpected type returned by get_loc: {type(loc)}")

                reshaped_var[year_idx, month - 1, :days_in_month, :, :] = var_np[
                    start_idx : start_idx + days_in_month
                ]

        # Create a new DataArray with appropriate dimensions and coordinates
        reshaped_data[var] = xr.DataArray(
            reshaped_var,
            dims=["year", "month", "day", "lat", "lon"],
            coords={
                "year": range(start_year, end_year + 1),
                "month": range(1, 13),
                "day": range(1, 32),
                "lat": ds_selected.lat,
                "lon": ds_selected.lon,
            },
        )

    return xr.Dataset(reshaped_data)


def assign_w_day(ds, bc_boundary):
    """
    Adds daily specific humidity and wind speed to an xarray dataset based on boundary conditions.

    Args:
        ds (xarray.Dataset): Original dataset.
        bc_boundary (str): Boundary condition type ('lateral' or 'surface').

    Returns:
        xarray.Dataset: Modified dataset with additional variables.
    """

    # Resample dataset to daily frequency, summing over the day.
    # ds_daily = ds.resample(time="D").sum()
    ds_daily = ds.resample(time="D").mean()

    if bc_boundary == "lateral":
        # Convert specific humidity from kg/kg to g/kg
        ds_daily["hus"] *= 10000

        # Calculate wind speed and add as a new variable.
        # ds_daily["w"] = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2).resample(time="D").sum()
        ds_daily["w"] = np.sqrt(ds["ua"] ** 2 + ds["va"] ** 2).resample(time="D").mean()

        ds_daily["w"] *= 10

        # Drop 'u' and 'v' as they are no longer needed.
        ds_daily = ds_daily.drop_vars(["ua", "va"])

        # Transpose the order of variables from q t w to w t q
        # ds_daily = ds_daily.transpose("w", "t", "q")
        ds_daily_ordered = ds_daily[["w", "ta", "hus"]]

        return ds_daily_ordered
    else:
        return ds_daily


def calculate_w(u, v):
    """
    Calculate wind speed from the u and v wind components.

    Args:
        u (array-like): Zonal wind component.
        v (array-like): Meridional wind component.

    Returns:
        array-like: Wind speed.
    """

    return np.sqrt(u**2 + v**2)


def assign_w_6hr(config, do, bc_boundary):
    """
    Add specific humidity and wind speed to an xarray dataset for 6-hourly data based on boundary conditions.

    Args:
        config (dict): Configuration dictionary (unused here, kept for compatibility).
        do (xarray.Dataset): Original dataset.
        bc_boundary (str): Boundary condition type ('lateral' or 'surface').

    Returns:
        xarray.Dataset: Modified dataset with 'hus' scaled, wind speed 'w' computed, 
                        and ordered variables, without modifying the input dataset.
    """
    if bc_boundary == "lateral":
        # Work on a copy to prevent modifying the input
        ds = do.copy(deep=True)

        # Scale 'hus' safely without altering 'do'
        ds = ds.assign(hus=ds["hus"] * 10000)

        # Compute wind speed
        ds = ds.assign(
            w=xr.apply_ufunc(calculate_w, ds["ua"], ds["va"], dask="parallelized") * 10
        )

        # Drop unnecessary vars
        ds = ds.drop_vars(["ua", "va"])

        # Reorder variables
        ds_ordered = ds[["w", "ta", "hus"]]
        return ds_ordered
    else:
        return do


def extract_and_reshape_sam_origin(ds_selected, nvar, start_year, end_year):
    """
    Extract and reshape the given dataset to its original format for specific variables.

    Args:
        ds_selected (xarray.Dataset): Dataset to be reshaped.
        nvar (int): Number of variables.
        start_year (int): Start year for reshaping.
        end_year (int): End year for reshaping.

    Returns:
        xarray.DataArray: Reshaped dataset.
    """

    # Select data for the desired years
    ds_selected = ds_selected.sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )

    # Initialize an empty array to hold the reshaped data
    reshaped_data = xr.DataArray(
        np.zeros(
            (
                nvar,
                end_year - start_year + 1,
                12,
                31,
                len(ds_selected.lat),
                len(ds_selected.lon),
            )
        ),
        dims=["var", "year", "month", "day", "lat", "lon"],
        coords={
            "var": list(ds_selected.data_vars),
            "year": range(start_year, end_year + 1),
            "month": range(1, 13),
            "day": range(1, 32),
            "lat": ds_selected.lat,
            "lon": ds_selected.lon,
        },
    )

    for i, var in enumerate(ds_selected.data_vars):
        reshaped_data.loc[
            {
                "var": var,
                "year": ds_selected["time.year"],
                "month": ds_selected["time.month"],
                "day": ds_selected["time.day"],
                "lat": ds_selected.lat,
                "lon": ds_selected.lon,
            }
        ] = ds_selected[var]

    return reshaped_data


def fill_reshaped_data(var, ds_selected, reshaped_data):
    """
    Fill reshaped data with values from the selected dataset for the given variable.

    Args:
        var (str): Variable name.
        ds_selected (xarray.Dataset): Dataset containing variable values.
        reshaped_data (xarray.Dataset): Dataset to be filled.

    Returns:
        xarray.Dataset: Filled dataset.
    """

    reshaped_data[var].loc[
        {
            "year": ds_selected["time.year"],
            "month": ds_selected["time.month"],
            "day": ds_selected["time.day"],
            "lat": ds_selected.lat,
            "lon": ds_selected.lon,
        }
    ] = ds_selected[var]
    return reshaped_data


def extract_and_reshape_delayed(
    config, ds_selected, nvar, start_year, end_year, bc_boundary
):
    """
    Extract and reshape dataset asynchronously, including applying bias correction boundaries.

    Args:
        ds_selected (xarray.Dataset): Dataset to extract and reshape.
        nvar (int): Number of variables.
        start_year (int): Start year.
        end_year (int): End year.
        bc_boundary (str): Boundary condition ('lateral' or 'surface').

    Returns:
        xarray.Dataset: Reshaped and merged dataset.
    """

    ds_selected = ds_selected.sel(
        time=slice(f"{start_year}-01-01", f"{end_year}-12-31")
    )

    reshaped_data = xr.Dataset(
        {
            var: xr.DataArray(
                np.zeros(
                    (
                        end_year - start_year + 1,
                        12,
                        31,
                        len(ds_selected.lat),
                        len(ds_selected.lon),
                    )
                ),
                dims=["year", "month", "day", "lat", "lon"],
                coords={
                    "year": range(start_year, end_year + 1),
                    "month": range(1, 13),
                    "day": range(1, 32),
                    "lat": ds_selected.lat,
                    "lon": ds_selected.lon,
                },
            )
            for var in ds_selected.data_vars
        }
    )

    delayed_tasks = [
        dask.delayed(fill_reshaped_data)(var, ds_selected, reshaped_data)
        for var in ds_selected.data_vars
    ]

    results = dask.compute(*delayed_tasks)
    if bc_boundary == "lateral":
        merged_data = xr.merge([results[0].w, results[1].ta, results[2].hus])
    else:
        merged_data = xr.merge([results[0].tos])

    return merged_data


def convert_to_daily_with_fraction(config, ds):
    """
    Convert 6-hourly data to daily data and calculate fraction factors.

    Args:
        ds (xarray.Dataset): Original dataset.

    Returns:
        tuple: Daily dataset and the corresponding fraction factors.
    """

    # Resample the dataset to daily frequency by summing 6-hourly data
    # daily_ds = ds.resample(time="1D").sum()
    # Replace to daily mean
    daily_ds = ds.resample(time="1D").mean()

    # Expand the daily dataset back to the original 6-hourly time dimension
    # by repeating daily values for each 6-hourly interval
    # daily_expanded = daily_ds.reindex(time=ds.time, method="pad")

    # Calculate the fraction factor for each 6-hourly time step

    return daily_ds, ds / daily_ds.reindex(time=ds.time, method="pad").compute()


def generate_dates_xr(start_year, end_year):
    """
    Generate a complete date range considering leap years.

    Args:
        start_year (int): Start year.
        end_year (int): End year.

    Returns:
        pandas.DatetimeIndex: Generated date range.
    """

    dates = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D"
    )
    return dates


def generate_dates_6hr(start_year, nyrmax, monmax, ndmax):
    """
    Generate a date range considering leap years.

    Args:
        start_year (int): The starting year.
        nyrmax (int): Number of years.
        monmax (int): Number of months (typically 12).
        ndmax (int): Number of days in the longest month (typically 31).

    Returns:
        pandas.DatetimeIndex: The generated date range.
    """

    dates = []
    for year in range(start_year, start_year + nyrmax):
        for month in range(1, monmax + 1):
            # Determine the number of days in the month
            if month == 2:  # February
                days_in_month = 29 if is_leap_year(year) else 28
            elif month in [4, 6, 9, 11]:  # April, June, September, November
                days_in_month = 30
            else:
                days_in_month = 31

            for day in range(1, days_in_month + 1):
                dates.append(pd.Timestamp(year=year, month=month, day=day))

    return pd.to_datetime(dates)


def align_daily_data_xr(config, daily_data_xr, start_year):
    """
    Align daily data considering leap years and return an xarray Dataset
    with a time dimension replacing the year, month, and day dimensions.

    Args:
        daily_data_xr (xarray.Dataset): Dataset of daily data with dimensions (year, month, day, lat, lon).
        start_year (int): The starting year.

    Returns:
        xarray.Dataset: Dataset aligned with a time dimension.
    """

    # Generate the full range of dates accounting for leap years
    end_year = start_year + daily_data_xr.sizes["year"] - 1
    dates = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D"
    )

    # Filter out the days that do not exist in the calendar (e.g., February 30th)
    valid_days = pd.to_datetime(dates)

    # Reshape the data to have a single 'time' dimension instead of year, month, and day
    flattened_data = daily_data_xr.stack(time=("year", "month", "day"))

    # Filter out invalid dates from the MultiIndex
    valid_mask = ~pd.isna(
        pd.to_datetime(
            {
                "year": flattened_data["year"].values,
                "month": flattened_data["month"].values,
                "day": flattened_data["day"].values,
            },
            errors="coerce",
        )
    )

    # Apply the mask to filter out invalid dates
    flattened_data = flattened_data.isel(time=valid_mask)

    # Construct a datetime64 index from the filtered MultiIndex
    time_index = pd.to_datetime(
        {
            "year": flattened_data["year"].values,
            "month": flattened_data["month"].values,
            "day": flattened_data["day"].values,
        }
    )

    flattened_data = flattened_data.drop_vars(["time", "year", "month", "day"])
    # Assign the new time index to the flattened data
    flattened_data = flattened_data.assign_coords(time=time_index).reset_coords(
        drop=True
    )
    # flattened_data = flattened_data.assign_coords('time', time_index).reset_coords(
    #     drop=True
    # )
    # Align the flattened data with the valid dates
    aligned_data_xr = flattened_data.sel(time=valid_days)

    # Reindex the aligned data to ensure all dates are covered, filling in any missing dates
    aligned_data_xr = aligned_data_xr.reindex({"time": valid_days}, method="ffill")

    return aligned_data_xr


def daily_to_6hourly_xr(
    config, daily_data_np, fraction_factors_xr, var_names, starty, endy
):
    """
    Convert daily data to 6-hourly data using fraction factors.

    Parameters:
    - daily_data_np: numpy array of shape (nvar, nyrmax, monmax, ndmax)
    - fraction_factors_xr: xarray Dataset with 6-hourly fraction factors
    - var_names: list of variable names
    - nyrmax, monmax, ndmax, nhmax, starty: int, dimensions and start year

    Returns:
    - six_hourly_data_xr: xarray Dataset of 6-hourly data
    """

    # Align the daily data with the generated dates
    daily_data_aligned = align_daily_data_xr(config, daily_data_np, starty)

    # Repeat the daily data to match the 6-hourly fraction factors time dimension
    daily_repeated = daily_data_aligned.reindex_like(
        fraction_factors_xr, method="ffill"
    )

    # Multiply by fraction factors for each variable
    six_hourly_data_xr = xr.Dataset()
    for var in var_names:
        six_hourly_data_xr[var] = daily_repeated[var] * fraction_factors_xr[var]

    return six_hourly_data_xr


def convert_6hr_to_original_xr(config, bias_corrected_data_xr, g_u_xr, g_v_xr):
    """
    Convert 6-hourly bias-corrected output back to its original form.

    Parameters:
    bias_corrected_data_xr: xarray Dataset with variables 'hus', 'ta', 'w'
        The 6-hourly bias-corrected data
    g_u_xr: xarray DataArray
        Original u component of wind
    g_v_xr: xarray DataArray
        Original v component of wind

    Returns:
    xarray Dataset
        The 6-hourly data in its original form (with bcu and bcv added as variables)
    """
    # Keep singleton lat/lon dimensions for 1x1 tile tests. Dropping them turns
    # the corrected field into a time series and breaks spatial reconstruction.
    squeeze_dims = [
        dim
        for dim, size in bias_corrected_data_xr.sizes.items()
        if size == 1 and dim not in {"time", "lat", "lon"}
    ]
    if squeeze_dims:
        bias_corrected_data_xr = bias_corrected_data_xr.squeeze(
            dim=squeeze_dims, drop=False
        )

    # Pre-chunk the input datasets to optimize computation
    bias_corrected_data_xr = bias_corrected_data_xr.transpose(
        "time", "lat", "lon"
    ).chunk({"time": 1000, "lat": -1, "lon": -1})
    g_u_xr = g_u_xr.transpose("lat", "lon", "time").chunk(
        {"time": 1000, "lat": -1, "lon": -1}
    )
    g_v_xr = g_v_xr.transpose("lat", "lon", "time").chunk(
        {"time": 1000, "lat": -1, "lon": -1}
    )

    # Convert hus by dividing to 1000, back to original units
    q_converted = (
        bias_corrected_data_xr["hus"] / 10000
    )  # change 1000 to 10000 to avoid small values

    # Convert ta by multiplying negative values by -1 using apply_ufunc for better parallelization
    t_converted = xr.apply_ufunc(
        lambda x: xr.where(x < 0, x * -1, x),
        bias_corrected_data_xr["ta"],
        dask="parallelized",
        output_dtypes=[float],
    )

    # Convert w to wind components u and v using optimized operations
    w = bias_corrected_data_xr["w"] / 10
    bcv = xr.apply_ufunc(
        np.sqrt,
        w**2 / ((g_u_xr / g_v_xr) ** 2 + 1),
        dask="parallelized",
        output_dtypes=[float],
    )
    bcu = xr.apply_ufunc(
        np.sqrt,
        w**2 - bcv**2,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Adjust the sign of bcu and bcv based on the original wind components g_u and g_v
    bcu = xr.where(g_u_xr < 0, bcu * -1, bcu)
    bcv = xr.where(g_v_xr < 0, bcv * -1, bcv)

    # Combine the converted data into the final dataset
    converted_data_xr = xr.Dataset(
        {"hus": q_converted, "ta": t_converted, "ua": bcu, "va": bcv}
    )

    return converted_data_xr


def expand_config_bounds_from_data(lat_min, lat_max, lon_min, lon_max, sample_file):
    """
    Dynamically expands latitude and longitude bounds by (about) one grid cell,
    but clamps to the dataset extent for global requests.

    - If user requests (near) global in lat or lon, do not expand past the grid.
    - Handle lon convention differences ([-180,180] vs [0,360]).
    """
    sample_data = xr.open_dataset(sample_file)

    lat_values = sample_data.lat.values
    lon_values = sample_data.lon.values

    # Grid spacing
    dy = float(np.abs(lat_values[1] - lat_values[0])) if len(lat_values) > 1 else 0.0
    dx = float(np.abs(lon_values[1] - lon_values[0])) if len(lon_values) > 1 else 0.0

    data_lat_min = float(np.min(lat_values))
    data_lat_max = float(np.max(lat_values))
    data_lon_min = float(np.min(lon_values))
    data_lon_max = float(np.max(lon_values))

    # Normalize requested lon range to dataset convention
    req_lon_min, req_lon_max = float(lon_min), float(lon_max)

    def to_0360(v):
        return v % 360.0

    def to_m180_180(v):
        v = ((v + 180.0) % 360.0) - 180.0
        # Fix -180 exact to 180 conventionally if dataset min is ~-180
        return v

    if data_lon_min >= 0.0:  # dataset in [0,360)
        req_lon_min = to_0360(req_lon_min)
        req_lon_max = to_0360(req_lon_max)
    else:  # dataset in [-180,180]
        req_lon_min = to_m180_180(req_lon_min)
        req_lon_max = to_m180_180(req_lon_max)

    # Determine if user asked for global coverage
    # Requested lon span, accounting for wrap
    span_req = (req_lon_max - req_lon_min) % 360.0
    total_span = (data_lon_max - data_lon_min) + (dx if dx > 0 else 0.0)

    # Treat span 0 (e.g., 0..360) as full circle
    is_global_lon = dx > 0 and (
        abs(span_req - total_span) <= 0.5 * dx or span_req == 0.0 or abs(span_req - 360.0) <= 0.5 * dx
    )

    # For latitude, check near-global request (within half a cell from ends)
    is_global_lat = (
        lat_min <= data_lat_min + 0.5 * dy if dy > 0 else True
    ) and (
        lat_max >= data_lat_max - 0.5 * dy if dy > 0 else True
    )

    # Expand or clamp latitude
    if is_global_lat or dy == 0.0:
        latmin = data_lat_min
        latmax = data_lat_max
    else:
        latmin = max(data_lat_min, float(lat_min) - dy)
        latmax = min(data_lat_max, float(lat_max) + dy)

    # Expand or clamp longitude (no cross-seam slice here; clamp to dataset span)
    if is_global_lon or dx == 0.0:
        lonmin = data_lon_min
        lonmax = data_lon_max
    else:
        # Ensure ordering consistent with dataset (assumes monotonic increasing lon)
        if req_lon_min <= req_lon_max:
            lonmin = max(data_lon_min, req_lon_min - dx)
            lonmax = min(data_lon_max, req_lon_max + dx)
        else:
            # Rare case: user provided wrapped range (e.g., 350..10) — clamp to dataset and do not expand across seam
            lonmin = data_lon_min
            lonmax = data_lon_max

    # Subset once to determine sizes/values after clamped expansion
    subset = sample_data.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
    lat_size, lon_size = subset.sizes["lat"], subset.sizes["lon"]
    lat_values = subset.lat.values
    lon_values = subset.lon.values

    sample_data.close()
    return latmin, latmax, lonmin, lonmax, lat_size, lon_size, lat_values, lon_values


def determine_tiles(config, lat_size, lon_size, max_tile_size=None):
    """
    Determines the number of tiles based on the total number of grid points using an index-based tiling approach.

    Args:
        file_paths (dict): Dictionary of file paths for each variable.
        var (str): Variable name used to extract the dataset.
        lat_size (int): Total number of latitude grid cells.
        lon_size (int): Total number of longitude grid cells.
        max_tile_size (int): Maximum allowed tile size (default: 1500 grid cells).

    Returns:
        (int, int): Number of tiles for latitude and longitude.
    """
    max_tile_size = config.max_tile_size if max_tile_size is None else max_tile_size

    # Compute total number of grid points
    total_points = lat_size * lon_size

    # Estimate number of tiles needed
    approx_tiles = math.ceil(total_points / max_tile_size)
    ratio = lat_size / lon_size  # Maintain lat/lon aspect ratio

    # Determine the number of tiles along latitude and longitude
    n_lat_tiles = max(1, round(math.sqrt(approx_tiles * ratio)))
    n_lon_tiles = max(1, round(approx_tiles / n_lat_tiles))

    # Ensure tile counts do not exceed actual grid size
    n_lat_tiles = min(n_lat_tiles, lat_size)
    n_lon_tiles = min(n_lon_tiles, lon_size)

    return n_lat_tiles, n_lon_tiles


def split_domain(n_lat_tiles, n_lon_tiles, lat_values, lon_values):
    """
    Splits the domain into index-based tiles, expanding only interior tiles by half a grid cell.
    Edge tiles are clamped to the dataset extent (important for global domains).
    """
    lat_size = len(lat_values)
    lon_size = len(lon_values)

    lat_indices = np.linspace(0, lat_size, n_lat_tiles + 1, dtype=int)
    lon_indices = np.linspace(0, lon_size, n_lon_tiles + 1, dtype=int)

    dy = float(np.abs(lat_values[1] - lat_values[0])) if lat_size > 1 else 0.0
    dx = float(np.abs(lon_values[1] - lon_values[0])) if lon_size > 1 else 0.0

    data_lat_min = float(lat_values.min())
    data_lat_max = float(lat_values.max())
    data_lon_min = float(lon_values.min())
    data_lon_max = float(lon_values.max())

    tiles = []
    for i in range(n_lat_tiles):
        for j in range(n_lon_tiles):
            lat_min_idx, lat_max_idx = lat_indices[i], lat_indices[i + 1]
            lon_min_idx, lon_max_idx = lon_indices[j], lon_indices[j + 1]

            # Expand only interior tile edges by half-cell
            lat_min = float(lat_values[lat_min_idx]) - (0.5 * dy if i > 0 else 0.0)
            lat_max = float(lat_values[lat_max_idx - 1]) + (0.5 * dy if i < n_lat_tiles - 1 else 0.0)

            lon_min = float(lon_values[lon_min_idx]) - (0.5 * dx if j > 0 else 0.0)
            lon_max = float(lon_values[lon_max_idx - 1]) + (0.5 * dx if j < n_lon_tiles - 1 else 0.0)

            # Clamp to dataset extent (prevents going beyond global edges)
            lat_min = max(lat_min, data_lat_min)
            lat_max = min(lat_max, data_lat_max)
            lon_min = max(lon_min, data_lon_min)
            lon_max = min(lon_max, data_lon_max)

            tiles.append(
                {
                    "lat_min_idx": lat_min_idx,
                    "lat_max_idx": lat_max_idx,
                    "lon_min_idx": lon_min_idx,
                    "lon_max_idx": lon_max_idx,
                    "lat_min": lat_min,
                    "lat_max": lat_max,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                }
            )

    return tiles


def determine_base_path(config, year):
    if year < 2015:
        return (
            config.bc_hist_path,
            config.infor,
            config.gname,
            config.period,
            config.cinfor,
            config.sinfor,
            config.version,
        )
    else:
        return (
            config.bc_future_path,
            config.infor,
            config.gname,
            config.scenario,
            config.cinfor_f,
            config.sinfor,
            config.version_f,
        )


# def generate_file_paths_future(variable, start_year, end_year):
#     file_paths = []
#     for year in range(start_year, end_year + 1):
#         base_path, info, gna, peri, cinf, sinf, vers = determine_base_path(year)
#         if year == start_year and gna in ["ACCESS-ESM1-5"]:
#             prev_dec_path = f"{base_path}/{variable}/{sinf}/v{vers}/{variable}_{info}_{gna}_{peri}_{cinf}_{sinf}_{year-1}12*.nc"
#             file_paths.extend(glob.glob(prev_dec_path))

#         file_path_pattern = f"{base_path}/{variable}/{sinf}/v{vers}/{variable}_{info}_{gna}_{peri}_{cinf}_{sinf}_{year}*.nc"
#         file_paths.extend(glob.glob(file_path_pattern))


#     return file_paths
def generate_file_paths_future(config, variable, start_year, end_year, data_type):
    """
    Generates file paths for different types of data (future GCM, validation GCM, or validation OBS)
    based on the variable and year range.

    Parameters:
        variable (str): The variable name (e.g., 'tas', 'pr').
        start_year (int): Start year of the data.
        end_year (int): End year of the data.
        data_type (str): Type of data ('future', 'validation_gcm', 'validation_obs').

    Returns:
        list: List of file paths matching the given criteria.
    """
    file_paths = []
    if config.bc_boundary == "lateral":
        for year in range(start_year, end_year + 1):
            if data_type == "future":
                base_path, info, gna, peri, cinf, sinf, vers = determine_base_path(
                    config, year
                )
                file_path_pattern_gcm = (
                    f"{base_path}/{variable}/g*/v*/"
                    f"{variable}_{info}_{gna}_*_{year}*.nc"
                )
                file_paths.extend(glob.glob(file_path_pattern_gcm))
            elif data_type == "validation":
                base_path, info, gna, peri, cinf, sinf, vers = determine_base_path(
                    config, year
                )
                file_path_pattern_gcm = (
                    f"{base_path}/{variable}/g*/v*/"
                    f"{variable}_{info}_{gna}_*_{year}*.nc"
                )
                file_paths.extend(glob.glob(file_path_pattern_gcm))
            # Include data from the previous December for specific cases (future GCM only)
            if (
                data_type == "future"
                and year == start_year
                and gna in ["ACCESS-ESM1-5"]
            ):
                prev_dec_path = (
                    f"{base_path}/{variable}/g*/v*/"
                    f"{variable}_{info}_{gna}_*_{year-1}12*.nc"
                )
                file_paths.extend(glob.glob(prev_dec_path))

    elif config.bc_boundary == "surface":
        # file_paths = glob.glob(
        #     f"{config.obs_path}/{variable}_{config.infor}_{config.gname}_*_{config.cinfor}_{config.sinfor}_*_remapped.nc"
        # )
        file_paths = glob.glob(
            f"{config.obs_path}/{variable}_{config.infor}_{config.gname}_*"
        )
        if config.bc_future:
            # file_paths += glob.glob(
            #     f"{config.obs_path}/{data_type}/{variable}_{config.infor}_{config.gname}_*_{config.cinfor}_{config.sinfor}_*_remapped.nc"
            # )
            file_paths += glob.glob(
                f"{config.obs_path}/{data_type}/{variable}_{config.infor}_{config.gname}_*"
            )
        return sorted(file_paths)  # Early return to avoid double globbing

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    file_paths.extend(sorted(glob.glob(file_path_pattern_gcm)))

    return file_paths


def load_and_combine_variables(
    config, tile, variables, level, start_year, end_year, data_type
):
    """
    Load and combine historical and future GCM data for given variables
    over specified year ranges.
    """
    lat_range = (tile["lat_min"], tile["lat_max"])  # Adjust as needed
    lon_range = (tile["lon_min"], tile["lon_max"])  # Adjust as needed

    sliced_ds_hist = xr.Dataset()
    sliced_ds_future = xr.Dataset()

    # # Handle historical data if start_year is before 2015
    # if config.bc_boundary == "lateral":
    #     if start_year < 2015:
    #         hist_end_year = min(end_year, 2014)
    #         for variable in variables:
    #             file_paths = generate_file_paths_future(
    #                 variable, start_year - 1, hist_end_year, data_type=data_type
    #             )
    #             data_var = load_preprocess_variable(
    #                 file_paths,
    #                 variable,
    #                 level,
    #                 lat_range,
    #                 lon_range,
    #                 start_year,
    #                 hist_end_year,
    #             )
    #             if variable in ["ua", "va"]:
    #                 target_lon = (
    #                     sliced_ds_hist.lon if "lon" in sliced_ds_hist else data_var.lon
    #                 )
    #                 target_lat = (
    #                     sliced_ds_hist.lat if "lat" in sliced_ds_hist else data_var.lat
    #                 )
    #                 target_lev = (
    #                     sliced_ds_hist.lev if "lev" in sliced_ds_hist else data_var.lev
    #                 )
    #                 data_var = data_var.interp(
    #                     lat=target_lat,
    #                     lon=target_lon,
    #                     method="linear",
    #                     kwargs={"fill_value": "extrapolate"},
    #                 ).assign_coords(lon=target_lon, lat=target_lat, lev=target_lev)
    #             sliced_ds_hist[variable] = data_var

    #     # Handle future data if end_year is 2015 or later
    #     if end_year >= 2015:
    #         future_start_year = max(start_year, 2015)
    #         for variable in variables:
    #             file_paths = generate_file_paths_future(
    #                 variable, future_start_year - 1, end_year, data_type=data_type
    #             )
    #             data_var_h = load_preprocess_variable(
    #                 file_paths[0],
    #                 variable,
    #                 level,
    #                 lat_range,
    #                 lon_range,
    #                 future_start_year,
    #                 end_year,
    #             )
    #             data_var_f = load_preprocess_variable(
    #                 file_paths[1:],
    #                 variable,
    #                 level,
    #                 lat_range,
    #                 lon_range,
    #                 future_start_year,
    #                 end_year,
    #             )
    #             if variable in ["ua", "va"]:
    #                 target_lon = (
    #                     sliced_ds_future.lon
    #                     if "lon" in sliced_ds_future
    #                     else data_var.lon
    #                 )
    #                 target_lat = (
    #                     sliced_ds_future.lat
    #                     if "lat" in sliced_ds_future
    #                     else data_var.lat
    #                 )
    #                 target_lev = (
    #                     sliced_ds_future.lev
    #                     if "lev" in sliced_ds_future
    #                     else data_var.lev
    #                 )
    #                 data_var_h = data_var_h.interp(
    #                     lat=target_lat,
    #                     lon=target_lon,
    #                     method="linear",
    #                     kwargs={"fill_value": "extrapolate"},
    #                 ).assign_coords(lon=target_lon, lat=target_lat, lev=target_lev)
    #                 data_var_f = data_var_f.interp(
    #                     lat=target_lat,
    #                     lon=target_lon,
    #                     method="linear",
    #                     kwargs={"fill_value": "extrapolate"},
    #                 ).assign_coords(lon=target_lon, lat=target_lat, lev=target_lev)
    #             data_var = xr.combine_by_coords([data_var_h, data_var_f])
    #             sliced_ds_future[variable] = data_var[variable]

    # elif config.bc_boundary == "surface":
    #     for variable in variables:
    #         file_paths = generate_file_paths_future(
    #             variable, start_year, end_year, data_type=data_type
    #         )
    #         data_var = load_preprocess_variable(
    #             file_paths,
    #             variable,
    #             level,
    #             lat_range,
    #             lon_range,
    #             start_year,
    #             end_year,
    #         )
    #         sliced_ds_hist[variable] = data_var

    # # Combine historical and future datasets if both exist
    # combined_ds = xr.combine_by_coords(
    #     [sliced_ds_hist, sliced_ds_future]
    #     if sliced_ds_hist and sliced_ds_future
    #     else ([sliced_ds_hist] if sliced_ds_hist else [sliced_ds_future])
    # )
    if config.bc_boundary == "lateral":
        if start_year < 2015:
            hist_end_year = min(end_year, 2014)
            for variable in variables:
                file_paths = generate_file_paths_future(
                    config, variable, start_year - 1, hist_end_year, data_type=data_type
                )
                # print(file_paths)
                data_var = load_preprocess_variable(
                    config,
                    file_paths,
                    variable,
                    level,
                    lat_range,
                    lon_range,
                    start_year,
                    hist_end_year,
                )
                if variable in ["ua", "va"]:
                    target_lon = (
                        sliced_ds_hist.lon if "lon" in sliced_ds_hist else data_var.lon
                    )
                    target_lat = (
                        sliced_ds_hist.lat if "lat" in sliced_ds_hist else data_var.lat
                    )
                    target_lev = (
                        sliced_ds_hist.lev if "lev" in sliced_ds_hist else data_var.lev
                    )
                    data_var = data_var.interp(
                        lat=target_lat,
                        lon=target_lon,
                        method="linear",
                        kwargs={"fill_value": "extrapolate"},
                    ).assign_coords(lon=target_lon, lat=target_lat, lev=target_lev)
                sliced_ds_hist[variable] = data_var

        # Handle future data if end_year is 2015 or later
        if end_year >= 2015:
            future_start_year = max(start_year, 2015)
            for variable in variables:
                file_paths = generate_file_paths_future(
                    config,
                    variable,
                    future_start_year - 1,
                    end_year,
                    data_type=data_type,
                )
                # print('>2015', file_paths)
                data_var_h = load_preprocess_variable(
                    config,
                    file_paths[0],
                    variable,
                    level,
                    lat_range,
                    lon_range,
                    future_start_year,
                    end_year,
                )
                data_var_f = load_preprocess_variable(
                    config,
                    file_paths[1:],
                    variable,
                    level,
                    lat_range,
                    lon_range,
                    future_start_year,
                    end_year,
                )
                # print(data_var_h)
                # print(data_var_f)
                if variable in ["ua", "va"]:
                    target_lon = (
                        sliced_ds_future.lon
                        if "lon" in sliced_ds_future
                        else data_var.lon
                    )
                    target_lat = (
                        sliced_ds_future.lat
                        if "lat" in sliced_ds_future
                        else data_var.lat
                    )
                    target_lev = (
                        sliced_ds_future.lev
                        if "lev" in sliced_ds_future
                        else data_var.lev
                    )
                    data_var_h = data_var_h.interp(
                        lat=target_lat,
                        lon=target_lon,
                        method="linear",
                        kwargs={"fill_value": "extrapolate"},
                    ).assign_coords(lon=target_lon, lat=target_lat, lev=target_lev)
                    data_var_f = data_var_f.interp(
                        lat=target_lat,
                        lon=target_lon,
                        method="linear",
                        kwargs={"fill_value": "extrapolate"},
                    ).assign_coords(lon=target_lon, lat=target_lat, lev=target_lev)
                data_var = xr.combine_by_coords([data_var_h, data_var_f])
                sliced_ds_future[variable] = data_var[variable]
    elif config.bc_boundary == "surface":
        for variable in variables:
            file_paths = generate_file_paths_future(
                config, variable, start_year, end_year, data_type=data_type
            )
            data_var = load_preprocess_variable(
                config,
                file_paths,
                variable,
                level,
                lat_range,
                lon_range,
                start_year,
                end_year,
            )
            sliced_ds_hist[variable] = data_var

    for var in sliced_ds_hist.data_vars:
        sliced_ds_hist[var].attrs = {}
    for var in sliced_ds_future.data_vars:
        sliced_ds_future[var].attrs = {}

    # Combine historical and future datasets if both exist
    combined_ds = xr.combine_by_coords(
        [sliced_ds_hist, sliced_ds_future]
        if sliced_ds_hist and sliced_ds_future
        else ([sliced_ds_hist] if sliced_ds_hist else [sliced_ds_future])
    )
    return combined_ds


# def load_and_combine_variables(
#     variables, level, lat_range, lon_range, start_year, end_year
# ):

#     # Handle historical data if start_year is before 2015
#     if start_year < 2015:
#         sliced_ds_hist = xr.Dataset()
#         hist_end_year = min(
#             end_year, 2014
#         )  # Ensure historical data goes up to 2014 at most
#         for variable in variables:
#             file_paths = generate_file_paths_future(
#                 variable,
#                 start_year,
#                 hist_end_year,
#             )
#             data_var = load_preprocess_variable(
#                 file_paths, variable, level - 1, lat_range, lon_range
#             )
#             if variable in ["ua", "va"]:
#                 # Let's assume hus and ta have the target longitude values, and they are already loaded
#                 target_lon = (
#                     sliced_ds_hist.lon if "lon" in sliced_ds_hist else data_var.lon
#                 )
#                 target_lat = (
#                     sliced_ds_hist.lat if "lat" in sliced_ds_hist else data_var.lat
#                 )
#                 target_lev = (
#                     sliced_ds_hist.lev if "lev" in sliced_ds_hist else data_var.lev
#                 )
#                 # Check if va coordinates are different before interpolation
#                 if not data_var.lat.equals(target_lat):
#                     data_var = data_var.interp(
#                         lat=target_lat,
#                         method="linear",
#                         kwargs={"fill_value": "extrapolate"},
#                     )
#                 if not data_var.lon.equals(target_lon):
#                     data_var = data_var.interp(
#                         lon=target_lon,
#                         method="linear",
#                         kwargs={"fill_value": "extrapolate"},
#                     )
#                 # Assign the adjusted longitude values to ua or va
#                 data_var = data_var.assign_coords(
#                     lon=target_lon, lat=target_lat, lev=target_lev
#                 )
#             # datasets_list.append(data_var)
#             sliced_ds_hist[variable] = data_var

#     # Handle future data if end_year is 2015 or later
#     if end_year >= 2015:
#         sliced_ds_future = xr.Dataset()
#         future_start_year = max(
#             start_year, 2015
#         )  # Start from 2015 or the start_year if it's later
#         for variable in variables:
#             file_paths = generate_file_paths_future(
#                 variable,
#                 future_start_year,
#                 end_year,
#             )
#             data_var = load_preprocess_variable(
#                 file_paths, variable, level - 1, lat_range, lon_range
#             )
#             if variable in ["ua", "va"]:
#                 # Let's assume hus and ta have the target longitude values, and they are already loaded
#                 target_lon = (
#                     sliced_ds_future.lon if "lon" in sliced_ds_future else data_var.lon
#                 )
#                 target_lat = (
#                     sliced_ds_future.lat if "lat" in sliced_ds_future else data_var.lat
#                 )
#                 target_lev = (
#                     sliced_ds_future.lev if "lev" in sliced_ds_future else data_var.lev
#                 )
#                 # Check if va coordinates are different before interpolation
#                 if not data_var.lat.equals(target_lat):
#                     data_var = data_var.interp(
#                         lat=target_lat,
#                         method="linear",
#                         kwargs={"fill_value": "extrapolate"},
#                     )
#                 if not data_var.lon.equals(target_lon):
#                     data_var = data_var.interp(
#                         lon=target_lon,
#                         method="linear",
#                         kwargs={"fill_value": "extrapolate"},
#                     )
#                 # Assign the adjusted longitude values to ua or va
#                 data_var = data_var.assign_coords(
#                     lon=target_lon, lat=target_lat, lev=target_lev
#                 )
#             # Check for variable adjustments (e.g., for 'ua', 'va') and apply if necessary
#             # datasets_list.append(data_var)
#             sliced_ds_future[variable] = data_var

#     # Ensure datasets are pre-chunked before combining
#     sliced_ds_hist = sliced_ds_hist.chunk({"time": 1000, "lat": -1, "lon": -1})
#     sliced_ds_future = sliced_ds_future.chunk({"time": 1000, "lat": -1, "lon": -1})

#     # Combine all loaded datasets into one, handling level, latitude, and longitude adjustments as needed
#     combined_ds = xr.combine_by_coords([sliced_ds_hist, sliced_ds_future])

#     # Optionally, rename variables if needed
#     # rename_dict = {"hus": "q", "ta": "t", "ua": "u", "va": "v"}
#     # combined_ds = combined_ds.rename(rename_dict)

#     return combined_ds


# def generate_file_paths(
#     base_path, variable, infor, gname, period, cinfor, sinfor, version, start_year, end_year
# ):
#     file_paths = []
#     for year in range(start_year, end_year + 1):
#         file_path = (
#             #f"{base_path}/{variable}/{str(year)[:-1]}/"
#             f"{base_path}/{variable}/{sinfor}/v{version}/"
#             # f"{variable}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{year}01010000-{year}12311800.nc"
#             f"{variable}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{year}0101*.nc"
#         )
#         matching_file_paths = glob.glob(file_path)
#         file_paths.extend(matching_file_paths)
#     return file_paths


# def generate_dates(start_year, nyrmax, monmax, ndmax):
#     """
#     Generate a date range considering leap years.

#     Parameters:
#     - start_year: int, the starting year
#     - nyrmax: int, number of years
#     - monmax: int, number of months (typically 12)
#     - ndmax: int, number of days in the longest month (typically 31)

#     Returns:
#     - dates: pandas DatetimeIndex, the generated date range
#     """
#     dates = []
#     for year in range(start_year, start_year + nyrmax):
#         for month in range(1, monmax + 1):
#             # Determine the number of days in the month
#             if month == 2:  # February
#                 days_in_month = 29 if is_leap_year(year) else 28
#             elif month in [4, 6, 9, 11]:  # April, June, September, November
#                 days_in_month = 30
#             else:
#                 days_in_month = 31

#             for day in range(1, days_in_month + 1):
#                 dates.append(pd.Timestamp(year=year, month=month, day=day))

#     return pd.to_datetime(dates)


# def align_daily_data(daily_data_np, start_year, nyrmax, monmax, ndmax):
#     """Align daily data considering leap years"""
#     aligned_data = []
#     for i in range(nyrmax):
#         year = start_year + i
#         for month in range(1, monmax + 1):
#             if month == 2:  # February
#                 days_in_month = 29 if is_leap_year(year) else 28
#             elif month in [4, 6, 9, 11]:  # April, June, September, November
#                 days_in_month = 30
#             else:
#                 days_in_month = 31

#             month_index = month - 1
#             month_data = daily_data_np[:, i, month_index, :days_in_month]
#             aligned_data.append(month_data)

#     # Combine the data from all months and years
#     aligned_data = np.concatenate(aligned_data, axis=1)
#     return aligned_data


# def daily_to_6hourly_xr(
#     daily_data_np, fraction_factors_xr, var_names, nyrmax, monmax, ndmax, starty
# ):
#     """
#     Convert daily data to 6-hourly data using fraction factors.

#     Parameters:
#     - daily_data_np: numpy array of shape (nvar, nyrmax, monmax, ndmax)
#     - fraction_factors_xr: xarray Dataset with 6-hourly fraction factors
#     - var_names: list of variable names
#     - nyrmax, monmax, ndmax, nhmax, starty: int, dimensions and start year

#     Returns:
#     - six_hourly_data_xr: xarray Dataset of 6-hourly data
#     """
#     # Generate the correct date range considering leap years
#     dates = generate_dates(starty, nyrmax, monmax, ndmax)

#     # Align the daily data with the generated dates
#     daily_data_aligned = align_daily_data(daily_data_np, starty, nyrmax, monmax, ndmax)

#     # Create an xarray Dataset from the aligned daily data
#     daily_data_ds = xr.Dataset(
#         {
#             var_names[i]: xr.DataArray(
#                 daily_data_aligned[i], dims=["time"], coords={"time": dates}
#             )
#             for i in range(len(var_names))
#         }
#     )

#     # Repeat the daily data to match the 6-hourly fraction factors time dimension
#     daily_repeated = daily_data_ds.reindex_like(fraction_factors_xr, method="ffill")

#     # Multiply by fraction factors for each variable
#     six_hourly_data_xr = xr.Dataset()
#     for var in var_names:
#         six_hourly_data_xr[var] = daily_repeated[var] * fraction_factors_xr[var]

#     return six_hourly_data_xr


# def convert_6hr_to_original_xr(bias_corrected_data_xr, g_u_xr, g_v_xr):
#     """
#     Convert 6-hourly bias-corrected output back to its original form.

#     Parameters:
#     bias_corrected_data_xr: xarray Dataset with variables 'q', 't', 'w'
#         The 6-hourly bias-corrected data
#     g_u_xr: xarray DataArray
#         Original u component of wind
#     g_v_xr: xarray DataArray
#         Original v component of wind

#     Returns:
#     xarray Dataset
#         The 6-hourly data in its original form (with bcu and bcv added as variables)
#     """
#     # Convert q by dividing by 1000, back to original units
#     q_converted = bias_corrected_data_xr["hus"] / 1000.0

#     # Convert t by multiplying negative values by -1
#     t_converted = xr.where(
#         bias_corrected_data_xr["ta"] < 0,
#         bias_corrected_data_xr["ta"] * -1,
#         bias_corrected_data_xr["ta"],
#     )

#     # Convert w to wind components u and v
#     w = bias_corrected_data_xr["w"] / 10
#     bcv = np.sqrt(w**2 / ((g_u_xr / g_v_xr) ** 2 + 1))
#     bcu = np.sqrt(w**2 - bcv**2)

#     # Adjust the sign of bcu and bcv based on the original wind components g_u and g_v
#     bcu = xr.where(g_u_xr < 0, bcu * -1, bcu)
#     bcv = xr.where(g_v_xr < 0, bcv * -1, bcv)

#     # Combine the converted data
#     converted_data_xr = xr.Dataset(
#         {"hus": q_converted, "ta": t_converted, "ua": bcu, "va": bcv}
#     )

#     return converted_data_xr


# def align_daily_data(daily_data_np, start_year, nyrmax, monmax, ndmax):
#     """Align daily data considering leap years"""
#     aligned_data = []
#     for i in range(nyrmax):
#         year = start_year + i
#         for month in range(1, monmax + 1):
#             if month == 2:  # February
#                 days_in_month = 29 if is_leap_year(year) else 28
#             elif month in [4, 6, 9, 11]:  # April, June, September, November
#                 days_in_month = 30
#             else:
#                 days_in_month = 31

#             month_index = month - 1
#             month_data = daily_data_np[:, i, month_index, :days_in_month]
#             aligned_data.append(month_data)

#     # Combine the data from all months and years
#     aligned_data = np.concatenate(aligned_data, axis=1)
#     return aligned_data
