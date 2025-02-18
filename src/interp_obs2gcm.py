"""
This script performs horizontal and vertical interpolation of atmospheric variables from ERA5 reanalysis data to a target Global Climate Model (GCM) dataset. The interpolated data is saved as NetCDF files.

The script takes input variables from the ERA5 reanalysis data, such as specific humidity (q), temperature (t), zonal wind (u), and meridional wind (v), and maps them to the target variables in the GCM dataset, which are specific humidity (hus), air temperature (ta), zonal wind (ua), and meridional wind (va).

The horizontal interpolation is performed using the xESMF library, which provides conservative or bilinear methods depending on the input variable. The vertical interpolation is performed by converting pressure levels to hybrid height coordinates (and hybrid sigma pressure level) using the geopotential height from ERA5 and the coefficients from the GCM dataset.

The script utilizes Dask for parallel processing and xarray for data manipulation. The dask.distributed.Client is used to create a local cluster for parallel processing.

The script is designed to be run on the NCI's Gadi supercomputer, but can be modified to run on other systems.

Note: This script assumes that the necessary input files and directories are available and properly formatted.

Author: Youngil (Young) Kim, CLEX, CCRC, UNSW
Contact: youngil.kim@unsw.edu.au
"""

import argparse
import glob
import os
import warnings
# from multiprocessing import Pool
from pathlib import Path

import dask.array as da  # type: ignore
import numpy as np  # type: ignore
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


def setup_client(n_workers=None):
    """
    Set up Dask client for parallel processing, allowing customization of workers and threads.

    Args:
        n_workers (int): Number of workers to use.
        threads_per_worker (int): Number of threads per worker.

    Returns:
        Client: A Dask distributed client instance.
    """
    if n_workers is None:
        c = Client()
    else:
        c = Client(n_workers=n_workers)
    print("Dask client setup complete.")
    return c


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
        default=["hus", "ta", "ua", "va"],
        help="Path to the YAML configuration file.",
    )

    # parser.add_argument(
    #     "--sy", type=int, default=config.startyear_h, help="Start year."
    # )
    # parser.add_argument("--ey", type=int, default=config.endyear_h, help="End year.")

    return parser.parse_args()


# Start of the script -----------------------------------------------------
import matplotlib.pyplot as plt


def standardize_dims(ds):
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


def compute_geopotential_height(p_levels, T_levels, q_levels=None):
    Rd = 287.05  # J/kg/K, specific gas constant for dry air
    g = 9.80665  # m/s^2, acceleration due to gravity

    # Ensure T_levels and q_levels are DataArrays for compatibility with xarray operations
    T_levels = (
        T_levels if isinstance(T_levels, xr.DataArray) else xr.DataArray(T_levels)
    )
    if q_levels is not None:
        q_levels = (
            q_levels if isinstance(q_levels, xr.DataArray) else xr.DataArray(q_levels)
        )

    # Compute the pressure ratio without aligning by levels
    upper_p = p_levels.isel(
        lev=slice(None, -1)
    ).data  # Pressure at the lower boundary of each layer
    lower_p = p_levels.isel(
        lev=slice(1, None)
    ).data  # Pressure at the upper boundary of each layer
    p_ratio = upper_p / lower_p
    log_p_ratio = da.log(p_ratio)

    # Re-create the DataArray for log_p_ratio with adjusted coordinates
    log_p_ratio_da = xr.DataArray(
        log_p_ratio,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": p_levels.time,
            "lev": p_levels.lev[:-1],  # Use coordinates from the upper slice
            "lat": p_levels.lat,
            "lon": p_levels.lon,
        },
    )

    # Compute mean temperature between consecutive levels without alignment
    upper_T = T_levels.isel(lev=slice(None, -1)).data
    lower_T = T_levels.isel(lev=slice(1, None)).data
    mean_T = (upper_T + lower_T) / 2

    # Re-create the DataArray for mean_T with adjusted coordinates
    mean_T_da = xr.DataArray(
        mean_T,
        dims=["time", "lev", "lat", "lon"],
        coords=log_p_ratio_da.coords,  # Match coordinates with log_p_ratio_da
    )

    if q_levels is not None:
        # Compute moist temperature
        upper_q = q_levels.isel(lev=slice(None, -1)).data
        lower_q = q_levels.isel(lev=slice(1, None)).data
        mean_q = (upper_q + lower_q) / 2

        # Re-create the DataArray for mean_q with adjusted coordinates
        mean_q_da = xr.DataArray(
            mean_q,
            dims=["time", "lev", "lat", "lon"],
            coords=log_p_ratio_da.coords,  # Match coordinates with log_p_ratio_da
        )
        mean_T_da = mean_T_da * (1.0 + 0.609133 * mean_q_da)

    # Calculate thickness of each layer (delta Z)
    delta_Z = (Rd / g) * mean_T_da * log_p_ratio_da

    # Integrate delta_Z from the top to obtain geopotential heights
    Z_levels_cumsum = delta_Z.cumsum(dim="lev")

    # Add an extra level at the top with extrapolated geopotential height
    gradient_top = Z_levels_cumsum.isel(lev=-1) - Z_levels_cumsum.isel(lev=-2)
    top_extrapolated = Z_levels_cumsum.isel(lev=-1) + gradient_top
    Z_levels = xr.concat(
        [Z_levels_cumsum, top_extrapolated.expand_dims(lev=[p_levels.lev[-1]])],
        dim="lev",
    )

    # Update level coordinates to include the top level
    Z_levels = xr.DataArray(
        Z_levels,
        dims=["time", "lev", "lat", "lon"],
        coords={
            "time": p_levels.time,
            "lev": p_levels.lev,  # Use original levels, assuming the extra level is added at the top
            "lat": p_levels.lat,
            "lon": p_levels.lon,
        },
        name="zfull",
    )

    return Z_levels


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


def interpolate_profile(source_profile, source_levels, target_levels):
    """
    Interpolates a source profile to match target levels with custom handling for extrapolation.

    Args:
        source_profile (array-like): The source profile to be interpolated.
        source_levels (array-like): The levels corresponding to the source profile.
        target_levels (array-like): The target levels to interpolate the source profile to.

    Returns:
        array-like: The interpolated profile matching the target levels.
    """
    # Check for NaN values
    if np.any(np.isnan(source_profile)) or np.any(np.isnan(source_levels)):
        print("NaN values found in source_profile or source_levels.")
        return np.full_like(
            target_levels, np.nan
        )  # Return NaN array of target levels' shape

    # Check for duplicate values in source_levels
    if len(np.unique(source_levels)) != len(source_levels):
        print("Duplicate values found in source_levels.")
        return np.full_like(
            target_levels, np.nan
        )  # Return NaN array of target levels' shape

    # Create the interpolation function with 'extrapolate' mode
    f_interp = interp1d(
        source_levels, source_profile, bounds_error=False, fill_value="extrapolate"
    )

    # Get the min and max of source levels
    x_min = np.min(source_levels)
    x_max = np.max(source_levels)

    return custom_interp(target_levels, f_interp, x_min, x_max)


def vertical_interpolation(source_da, source_levels_da, target_levels):
    """
    Perform vertical interpolation of a source data array to target levels using xarray's apply_ufunc.

    Parameters:
        source_da (xarray.DataArray): The source data array to be interpolated.
        source_levels_da (xarray.DataArray): The source data array's levels.
        target_levels (array-like): The target levels to interpolate to.

    Returns:
        xarray.DataArray: The interpolated data array.
    """
    source_levels = source_levels_da  # Ensure we are using the correct levels

    # Wrapper to apply interpolation using xarray's apply_ufunc to handle Dask arrays efficiently
    interpolated_da = xr.apply_ufunc(
        interpolate_profile,
        source_da,
        source_levels,
        target_levels,
        vectorize=True,  # Enable vectorized execution
        input_core_dims=[["level"], ["level"], ["lev"]],  # Define core dimensions
        output_core_dims=[["lev"]],  # Define output dimensions
        dask="parallelized",  # Enable Dask parallelization
        output_dtypes=[source_da.dtype],
    )

    return interpolated_da.assign_coords(lev=target_levels.lev)


def standardize_coords_from(ds):
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


def regrid(source_ds, target_ds, method, weights_path, rename_dict):
    """Perform interpolation/regridding of a source dataset to a target dataset using the specified method."""
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

    weights_file = Path(weights_path)
    reuse_weights = weights_file.is_file()
    regridder = xe.Regridder(
        source_ds,
        target_ds,
        method=method,
        filename=weights_path,
        reuse_weights=reuse_weights,
    )

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

    else:
        raise TypeError("Input must be xarray DataArray or Dataset")


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


# clear
def load_target_files(target_path, infor, sinfor, version, var, year):
    pattern = f"{target_path}/{infor}/{var}/{sinfor}/{version}/{var}_*{year}*.nc"
    return sorted(glob.glob(pattern))


# clear
def load_datasets(year, month, files, chunks):
    ds = xr.open_mfdataset(files, combine="by_coords", chunks=chunks)
    return ds.sel(time=(ds["time"].dt.year == year) & (ds["time"].dt.month == month))


# clear
def process_year_month(config, year, month):
    selected_variables = [config["var_interp"]]
    tq_variables = ["ta", "hus"]

    target_grids = {
        var: load_datasets(
            year,
            month,
            load_target_files(
                config["target_path"],
                config["infor"],
                config["sinfor"],
                config["version"],
                var,
                year,
            ),
            {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"},
        )
        for var in selected_variables
    }
    tq_grids = {
        var: load_datasets(
            year,
            month,
            load_target_files(
                config["target_path"],
                config["infor"],
                config["sinfor"],
                config["version"],
                var,
                year,
            ),
            {"time": "auto", "lev": "auto", "lat": "auto", "lon": "auto"},
        )
        for var in tq_variables
    }

    for var in selected_variables:
        target_grids[var]["lat"] = target_grids[var]["lat"].clip(-90, 90)
        if target_grids[var].sizes["time"] == 0:
            raise ValueError(f"No data available for {year}-{month} in variable {var}.")

    for var in tq_variables:
        tq_grids[var]["lat"] = tq_grids[var]["lat"].clip(-90, 90)
        if tq_grids[var].sizes["time"] == 0:
            raise ValueError(f"No data available for {year}-{month} in variable {var}.")

    return process_geopotential_height(config, target_grids, tq_grids, year, month)


# clear
def process_geopotential_height(config, target_grids, tq_grids, year, month):
    if (
        target_grids[config["var_interp"]].lev.standard_name
        == "atmosphere_hybrid_height_coordinate"
    ):
        target_z_files = glob.glob(
            f"{config['target_path']}/fx/zfull/{config['sinfor']}/v*/zfull_fx_*.nc"
        )
        target_zfull = (
            xr.open_dataset(
                target_z_files[0], chunks={"lev": -1, "lat": -1, "lon": -1}
            )["zfull"]
            .transpose("lev", "lat", "lon")
            .persist()
        )
        return target_zfull, target_grids, tq_grids
    elif (
        target_grids[config["var_interp"]].lev.standard_name
        == "atmosphere_hybrid_sigma_pressure_coordinate"
    ):
        compute_or_load_geopotential(config, target_grids, tq_grids, year, month)


# clear
def compute_or_load_geopotential(config, target_grids, tq_grids, year, month):
    if config["target_g_path"] == "None":
        target_t = tq_grids["ta"]
        target_q = tq_grids["hus"]
        target_p = calculate_pressure_levels(target_t.ap, target_t.b, target_t.ps)
        target_zfull = (
            compute_geopotential_height(target_p, target_t.ta)
            .chunk({"time": 10, "lev": -1, "lat": -1, "lon": -1})
            .transpose("time", "lev", "lat", "lon")
            .persist()
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
            time=(target_grids[config["var_interp"]]["time"].dt.year == year)
            & (target_grids[config["var_interp"]]["time"].dt.month == month)
        )
        target_zfull = selected_target_z.rename({"z": "zfull"}).persist()

    return target_zfull, target_grids


# clear
def load_input_files(input_path, var, year, month):
    pattern = f"{input_path}/{var}/{year}/{var}_*_{year}{month:02d}*.nc"
    return sorted(glob.glob(pattern))


# clear
def load_input_data(input_files):
    # with xr.open_dataset(input_files[0]) as temp_ds:
    #     # Determine chunking strategy based on dimensions
    #     chunks = {dim: "auto" for dim in temp_ds.dims}
    chunks = {dim: "auto" for dim in xr.open_dataset(input_files[0]).dims}
    ds = xr.open_mfdataset(input_files, combine="by_coords", chunks=chunks)
    return standardize_coords(ds)


def regrid_and_interpolate(
    config, input_var, target_var, year, month, target_grids, target_zfull_per, tq_grids
):

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

    sliced_do = (
        do_regridded.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        )
        .chunk({"level": -1})
        .persist()
    )
    # print('sliced_ds', sliced_do)
    # sliced_do.va[0,:,140:,150].plot(vmin= -40, vmax=40)
    # plt.show()

    # geopotential
    if config["input_z_path"]:
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
        g_era5_resampled, input_grids, tq_input_grids = process_year_month(
            config, config["input_path"], year, month
        )
        g_era5_resampled = standardize_coords_from(g_era5_resampled)

    target_zfull_per = standardize_coords_from(target_zfull_per)
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

    if config["input_model"] == "reanalysis":
        Z_era5 = geopotential_to_geopotential_height(g_era5)
        Z_era5_per = Z_era5.persist().chunk({"level": -1})
    else:
        Z_era5_per = g_era5.persist().chunk({"level": -1})

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

        sliced_gcm_z_data_interp = gcm_z_data_interp.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({"lev": -1})
        # print('sliced_do[target_var]', sliced_do[target_var])
        # print('Z_era5_per', Z_era5_per)
        # print('sliced_gcm_z_data_interp', sliced_gcm_z_data_interp)
        interpolated_ds = vertical_interpolation(
            sliced_do[target_var], Z_era5_per, sliced_gcm_z_data_interp
        )
    else:
        sliced_target_zfull = target_zfull_per.sel(
            lat=slice(config["lat_min"], config["lat_max"]),
            lon=slice(config["lon_min"], config["lon_max"]),
        ).chunk({"lev": -1})
        interpolated_ds = vertical_interpolation(
            sliced_do[target_var], Z_era5_per, sliced_target_zfull
        )

    interpolated_ds = interpolated_ds.transpose("time", "lev", "lat", "lon").chunk(
        {"lev": -1}
    )
    sliced_target_ds = target_ds.sel(
        lat=slice(config["lat_min"], config["lat_max"]),
        lon=slice(config["lon_min"], config["lon_max"]),
    ).chunk({"lev": -1})

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
    output_file = f"{config['output_path']}/{target_var}_{config['input_model'] if config["input_model"] == 'reanalysis' else config["input_gname"]}_to_{config['gname']}_{year}-{month:02}.nc"
    save_interpolated_data(interpolated_era5_da, output_file)
    print(f"Saved regridded data for {target_var} {year}-{month:02} to {output_file}")
    return interpolated_era5_da


# clear
def save_interpolated_data(interpolated_era5_da, output_file):
    with ProgressBar():
        print(f"Writing to {output_file}")
        interpolated_era5_da.to_netcdf(output_file, compute=False).compute()


# End of the functions -------------------------------------------------------


def main(config):

    client = setup_client()

    # target_model = config.target_model
    # Define variables to interpolate
    var_interp = config["var_interp"]
    # Define start and end years
    start_year = config["startyear_h"]
    end_year = config["endyear_h"]

    # Create output directory if it doesn't exist
    os.makedirs(config["output_path"], exist_ok=True)

    if var_interp in ["ua", "va", "ta", "hus"]:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                target_zfull, target_grids, tq_grids = process_year_month(
                    config, year, month
                )
                # var_interp = "ua"
                # print('press_year_month_output_target_zfull', target_zfull)
                if config["input_model"] == "reanalysis":
                    if var_interp == "hus":
                        var_obs = "q"
                    elif var_interp == "ta":
                        var_obs = "t"
                    elif var_interp == "ua":
                        var_obs = "u"
                    elif var_interp == "va":
                        var_obs = "v"
                else:
                    var_obs = var_interp
                process_input_variable(
                    config,
                    var_obs,
                    var_interp,
                    year,
                    month,
                    target_grids,
                    target_zfull,
                    tq_grids,
                )


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.yp)
    config["var_interp"] = args.var
    # config.startyear_h = args.sy
    # config.endyear_h = args.ey
    main(config)
    print("All done!")
