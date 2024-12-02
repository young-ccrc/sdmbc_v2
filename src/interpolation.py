import warnings

# from multiprocessing import Pool
from pathlib import Path

import dask.array as da  # type: ignore
import numpy as np  # type: ignore

# import dask.array as da  # type: ignore
import xarray as xr  # type: ignore
import xesmf as xe  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

warnings.simplefilter("ignore", UserWarning)


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
    # Identify indices where extrapolation happens
    lower_bound = x_new < x_min
    upper_bound = x_new > x_max

    # Fill NaN for upper bound
    y_new[upper_bound] = np.nan

    return y_new


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
    # Create the interpolation function with 'extrapolate' mode
    f_interp = interp1d(
        source_levels, source_profile, bounds_error=False, fill_value="extrapolate"
    )

    # Get the min and max of source levels
    x_min = np.min(source_levels)
    x_max = np.max(source_levels)

    return custom_interp(target_levels, f_interp, x_min, x_max)


# Assuming ds_regridded, Z_era5, and gcm_z_data are xarray DataArrays/Datasets with Dask arrays
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
    # Wrapper to apply interpolation using xarray's apply_ufunc to handle Dask arrays efficiently
    interpolated_da = xr.apply_ufunc(
        interpolate_profile,
        source_da,
        source_levels_da,
        target_levels,
        vectorize=True,  # Enable vectorized execution
        input_core_dims=[["level"], ["level"], ["lev"]],  # Define core dimensions
        output_core_dims=[["lev"]],  # Define output dimensions
        dask="parallelized",  # Enable Dask parallelization
        output_dtypes=[source_da.dtype],
    )

    # return interpolated_da
    return interpolated_da.assign_coords(lev=target_levels.lev)


def correct_latitudes(ds):
    # Ensure latitudes are within bounds
    ds["latitude"] = ds["latitude"].clip(-90, 90)
    return ds


def ensure_bounds(ds):
    if "longitude_bnds" not in ds.variables and "longitude" in ds.coords:
        lon = ds["longitude"]
        lon_bnds = np.zeros((len(lon), 2))
        lon_bnds[:, 0] = lon - (lon[1] - lon[0]) / 2
        lon_bnds[:, 1] = lon + (lon[1] - lon[0]) / 2
        ds["longitude_bnds"] = (("longitude", "bnds"), lon_bnds)
        ds["longitude"].attrs["bounds"] = "longitude_bnds"

    if "latitude_bnds" not in ds.variables and "latitude" in ds.coords:
        lat = ds["latitude"]
        lat_bnds = np.zeros((len(lat), 2))
        lat_bnds[:, 0] = lat - (lat[1] - lat[0]) / 2
        lat_bnds[:, 1] = lat + (lat[1] - lat[0]) / 2
        ds["latitude_bnds"] = (("latitude", "bnds"), lat_bnds)
        ds["latitude"].attrs["bounds"] = "latitude_bnds"

    return ds


# Horizontal interpolation
def regrid(source_ds, target_ds, method, weights_path, rename_dict, standard_name=None):
    """
    Perform interpolation/regridding of a source dataset to a target dataset using the specified method.

    Args:
        source_ds (xarray.DataArray or xarray.Dataset): The source dataset to be regridded.
        target_ds (xarray.Dataset): The target dataset with the desired grid.
        method (str): The interpolation method to be used. Supported methods: 'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch', 'nearest_s2d', 'nearest_d2s', 'patch'.
        weights_path (str): The path to the weights file used for regridding. If the file exists, the weights will be reused. If not, the weights will be computed and saved to this path.
        rename_dict (dict): A dictionary mapping variable names in the source dataset to the desired variable names in the regridded dataset.

    Returns:
        xarray.Dataset: The regridded dataset with variables renamed according to the provided rename_dict.

    """
    if standard_name == "atmosphere_hybrid_height_coordinate":
        if method == "conservative":
            # Standardize coordinate names if needed
            source_ds = standardize_coords(source_ds)
            target_ds = standardize_coords(target_ds)

            # Check if target_ds needs bounds to be ensured
            if (
                "longitude_bnds" not in target_ds.variables
                or "latitude_bnds" not in target_ds.variables
            ):
                target_ds = ensure_bounds(target_ds)

            # Check if source_ds needs bounds to be ensured
            if (
                "longitude_bnds" not in source_ds.variables
                or "latitude_bnds" not in source_ds.variables
            ):
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
        # Handling for DataArray
        if source_ds.name != "zfull":
            regridded_data = regridder(source_ds)
            # Rename the DataArray if necessary
            if source_ds.name in rename_dict:
                regridded_data.name = rename_dict[source_ds.name]
            return regridded_data
        return source_ds
    elif isinstance(source_ds, xr.Dataset):
        # Handling for Dataset
        regridded_data = regridder(source_ds)
        # Check and rename variables in the Dataset if necessary
        for var in list(regridded_data.data_vars):
            if var in rename_dict:
                regridded_data = regridded_data.rename({var: rename_dict[var]})
        return regridded_data
    else:
        raise TypeError("Input must be xarray DataArray or Dataset")


def standardize_coords_from(ds):
    coord_map = {"longitude": "lon", "latitude": "lat"}
    new_ds = ds.copy()
    for old_name, new_name in coord_map.items():
        if old_name in new_ds.coords:
            new_ds = new_ds.rename({old_name: new_name})
    return new_ds


def standardize_coords(ds):
    # Define the desired name mappings
    coord_map = {"lon": "longitude", "lat": "latitude", "lev": "level"}
    # Prepare to modify the dataset
    new_ds = ds.copy()

    # Check and handle each coordinate in the map
    for old_name, new_name in coord_map.items():
        if old_name in new_ds.coords:
            if new_name in new_ds.coords:
                # Both old and new coordinates exist
                # print(f"Both '{old_name}' and '{new_name}' exist.")
                # Check if they are identical to decide on merging or replacing
                if xr.all(new_ds[old_name] == new_ds[new_name]):
                    # If identical, drop the old coordinate
                    new_ds = new_ds.drop_vars(old_name)
                    # print(f"Dropped '{old_name}' as it is identical to '{new_name}'.")
                # else:
                # If not identical, consider how to merge or rename to avoid conflict
                # print(f"Coordinates '{old_name}' and '{new_name}' are not identical. Consider merging or manually handling.")
            else:
                # Safe to rename as the new name does not exist
                new_ds = new_ds.rename({old_name: new_name})
                # print(f"Renamed '{old_name}' to '{new_name}'.")
        # else:
        # print(f"'{old_name}' not found in the coordinates; no changes made.")

    return new_ds
