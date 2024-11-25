# ==========================================================================#

# import sys
import warnings
from dataclasses import asdict, dataclass  # type: ignore

import dask.array as da  # type: ignore
import numpy as np  # arrays and matrix math # type: ignore
import xarray as xr  # type: ignore
from config import config
# import yaml  # type: ignore
from data_preparation import is_leap_year
# import pandas as pd  # dataFrames
# ==========================================================================#
# calling fortran subroutines
# from mrmbc import constants as cons  # type: ignore
from mrmbc import mbc_subroutines as mbc  # type: ignore

# sys.path.append("/scratch/dm6/yk8692/sdmbc/")

# ==========================================================================#

# with open("/scratch/dm6/yk8692/sdmbc/user_input_test.yaml", "r") as file:
#     config_data = yaml.safe_load(file)

# from user_input import (
#    correction_model,
#    endyear_h,
#    lower_limit,
#    missing_value,
#    moving_window,
#    no_of_iterations,
#    no_of_variables,
#    startyear_h,
#    time_scale,
#    upper_limit,
# )

# Ignore a specific category of warning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# class Config:
#     def __init__(self, **entries):
#         self.__dict__.update(entries)


# config = Config(**config_data)
correction_model = config.correction_model
startyear_h = config.startyear_h
endyear_h = config.endyear_h
lower_limit = config.lower_limit
# missing_value = config.missing_value
# moving_window = config.moving_window
no_of_iterations = config.no_of_iterations
no_of_variables = config.no_of_variables
startyear_h = config.startyear_h
# time_scale = config.time_scale
upper_limit = config.upper_limit
bc_boundary = config.bc_boundary

time_scale = 0 # 0: daily, 1: monthly, default: 0
missing_value = 0.00001 # missing value in the input data
moving_window = 15 # centred moving window if input data is at daily time scale

# Thresholds for bias correction
# upper_limit[2] = upper_limit[2] * 10
# lower_limit[2] = lower_limit[2] * 10
# For wind speed, the thresholds are multiplied by 10, for bias correction purpose
upper_limit[0] = upper_limit[0] * 10
lower_limit[0] = lower_limit[0] * 10

# hist: daily mean, future: - avdc + avdh
# hist: daily sd, future: (x - avd)*sddh/sddc + avd
# hist: daily corl, future: cmod, gmod, cobs, gobs

# hist: monthly mean, future: - avmc + avmh
# hist: monthly sd, future: (x - avm)*sdmh/sdmc + avm
# hist: monthly corl, future: cmodm, gmodm, cobsm, gobsm

# hist: seasonal mean, future: - avsc + avsh
# hist: seasonal sd, future: (x - avs)*sdsh/sdsc + avs
# hist: seasonal corl, future: cmods, gmods, cobss, gobss

# hist: annual mean, future: - avyc + avyh
# hist: annual sd, future: (x - avy)*sdyh/sdyc + avy
# hist: annual corl, future: cmody, gmody, cobsy, gobsy


def compute_quantiles_along_axis(arr, axis, quantiles):
    # Compute percentiles along a specific axis
    return np.percentile(arr, quantiles, axis=axis)


def empirical_quantile_mapping(o, s, n_quantiles=100, extrapolation="constant"):
    """
    Empirical Quantile Mapping for a single variable using Dask.

    Parameters:
        o (dask.array or np.ndarray): Observational data.
        s (dask.array or np.ndarray): Simulation data to be corrected.
        n_quantiles (int): Number of quantiles.
        extrapolation (str): Method for extrapolation ("constant" or any other string).

    Returns:
        dask.array: Corrected simulation data.
    """
    # Ensure input arrays are Dask arrays
    o = da.asarray(o) if isinstance(o, np.ndarray) else o
    s = da.asarray(s) if isinstance(s, np.ndarray) else s

    # Define the quantiles to be computed
    quantiles = np.linspace(0, 100, n_quantiles)

    # Compute the quantiles along the appropriate axis using map_blocks
    obs_quantiles = da.map_blocks(
        compute_quantiles_along_axis, o, axis=0, quantiles=quantiles, dtype=o.dtype
    )
    sim_quantiles = da.map_blocks(
        compute_quantiles_along_axis, s, axis=0, quantiles=quantiles, dtype=s.dtype
    )

    # Interpolate the simulation data to match the observation quantiles
    corrected_s = da.map_blocks(
        np.interp, s, sim_quantiles, obs_quantiles, dtype=s.dtype
    )

    # Handle extrapolation
    if extrapolation == "constant":
        corrected_s = da.where(
            s > da.max(sim_quantiles),
            s + (da.max(obs_quantiles) - da.max(sim_quantiles)),
            corrected_s,
        )
        corrected_s = da.where(
            s < da.min(sim_quantiles),
            s + (da.min(obs_quantiles) - da.min(sim_quantiles)),
            corrected_s,
        )
    else:
        corrected_s = da.where(
            s > da.max(sim_quantiles), da.max(obs_quantiles), corrected_s
        )
        corrected_s = da.where(
            s < da.min(sim_quantiles), da.min(obs_quantiles), corrected_s
        )

    return corrected_s


def empirical_quantile_mapping_future(
    o_hist, s_hist, s_future, n_quantiles=100, extrapolation="constant"
):
    """
    Empirical Quantile Mapping for future period using historical observational
    and simulation data for calibration, adapted for Dask and xarray.

    Parameters:
        o_hist (dask.array or np.ndarray): Historical observational data.
        s_hist (dask.array or np.ndarray): Historical simulation data used for calibration.
        s_future (dask.array or np.ndarray): Future simulation data to be corrected.
        n_quantiles (int): Number of quantiles for the quantile mapping.
        extrapolation (str): Method for extrapolation ("constant" or "linear").

    Returns:
        dask.array: Corrected future simulation data.
    """

    # Ensure input arrays are Dask arrays
    o_hist = da.asarray(o_hist) if isinstance(o_hist, np.ndarray) else o_hist
    s_hist = da.asarray(s_hist) if isinstance(s_hist, np.ndarray) else s_hist
    s_future = da.asarray(s_future) if isinstance(s_future, np.ndarray) else s_future

    # Define the quantiles to be computed
    quantiles = np.linspace(0, 100, n_quantiles)

    # Compute the quantiles along the appropriate axis using map_blocks
    obs_quantiles = da.map_blocks(
        compute_quantiles_along_axis,
        o_hist,
        axis=0,
        quantiles=quantiles,
        dtype=o_hist.dtype,
    )
    sim_quantiles = da.map_blocks(
        compute_quantiles_along_axis,
        s_hist,
        axis=0,
        quantiles=quantiles,
        dtype=s_hist.dtype,
    )

    # Interpolate to map future simulations to observational quantiles
    corrected_s_future = da.map_blocks(
        np.interp, s_future, sim_quantiles, obs_quantiles, dtype=s_future.dtype
    )

    # Handle extrapolation for values outside the range of historical simulation data
    if extrapolation == "constant":
        upper_extrap = obs_quantiles[-1] - sim_quantiles[-1]
        lower_extrap = obs_quantiles[0] - sim_quantiles[0]

        corrected_s_future = da.where(
            s_future > sim_quantiles[-1], s_future + upper_extrap, corrected_s_future
        )
        corrected_s_future = da.where(
            s_future < sim_quantiles[0], s_future + lower_extrap, corrected_s_future
        )
    else:
        corrected_s_future = da.where(
            s_future > sim_quantiles[-1], obs_quantiles[-1], corrected_s_future
        )
        corrected_s_future = da.where(
            s_future < sim_quantiles[0], obs_quantiles[0], corrected_s_future
        )

    return corrected_s_future


def rescale_to_sum_one(ds):
    """
    Rescale the 6-hourly data in an xarray Dataset so that the sum across each day equals 1.

    Parameters:
        ds (xr.Dataset): The xarray Dataset with 6-hourly Dask arrays.

    Returns:
        xr.Dataset: Dataset with rescaled 6-hourly data.
    """
    rescaled_ds = xr.Dataset(coords=ds.coords)

    for var in ds.data_vars:
        data_6hr = ds[var].data

        # Check if the time dimension is divisible by 4 (for 6-hourly data)
        if data_6hr.shape[0] % 4 != 0:
            raise ValueError(
                "The time dimension must be divisible by 4 for 6-hourly data."
            )

        # Reshape and sum across each day (4 time steps per day)
        reshaped_data = data_6hr.reshape((-1, 4, data_6hr.shape[1], data_6hr.shape[2]))
        daily_sum = da.sum(reshaped_data, axis=1)  # Sum across the 4 time steps

        # Reshape daily_sum to match the original 6-hourly data shape
        daily_sum_repeated = daily_sum.repeat(4, axis=0)

        # Rescale the 6-hourly data
        rescaled_data_6hr = data_6hr / daily_sum_repeated

        # Assign the rescaled data back to the Dataset
        rescaled_ds[var] = xr.DataArray(
            rescaled_data_6hr, dims=ds[var].dims, coords=ds[var].coords
        )

    return rescaled_ds.chunk({"time": 1000, "lat": -1, "lon": -1})


def bc_correction_with_rescaling(
    ff_obs, ff_gcm, var_list_w, n_quantiles=100, extrapolation="constant"
):
    """
    Apply bias correction with empirical quantile mapping, followed by rescaling of fraction factors.

    Parameters:
    - ff_obs: Observational fraction factors as an xarray Dataset.
    - ff_gcm: GCM fraction factors as an xarray Dataset.
    - var_list_w: List of variables to be corrected.
    - n_quantiles: Number of quantiles for EQM.
    - extrapolation: Extrapolation method for EQM.

    Returns:
    - rescaled_corrected_gcm: Bias-corrected and rescaled fraction factors as an xarray Dataset.
    """
    corrected_ff_gcm = xr.Dataset()

    # Apply EQM for each variable
    for var in var_list_w:
        corrected_ff_gcm[var] = xr.apply_ufunc(
            empirical_quantile_mapping,
            ff_obs[var],
            ff_gcm[var],
            kwargs={"n_quantiles": n_quantiles, "extrapolation": extrapolation},
            dask="parallelized",
            output_dtypes=[ff_gcm[var].dtype],
        )

    # Rescale the corrected fraction factors so that their sum equals one across each day
    rescaled_corrected_gcm = rescale_to_sum_one(corrected_ff_gcm)

    return rescaled_corrected_gcm


def bc_correction_with_rescaling_future(
    ff_obs,
    ff_gcm_hist,
    ff_gcm_future,
    var_list_w,
    n_quantiles=100,
    extrapolation="constant",
):
    """
    Apply bias correction with empirical quantile mapping, followed by rescaling of fraction factors.

    Parameters:
    - ff_obs: Observational fraction factors as an xarray Dataset.
    - ff_gcm: GCM fraction factors as an xarray Dataset.
    - var_list_w: List of variables to be corrected.
    - n_quantiles: Number of quantiles for EQM.
    - extrapolation: Extrapolation method for EQM.

    Returns:
    - rescaled_corrected_gcm: Bias-corrected and rescaled fraction factors as an xarray Dataset.
    """
    corrected_ff_future = xr.Dataset()

    # Apply EQM for each variable
    for var in var_list_w:
        corrected_ff_future[var] = xr.apply_ufunc(
            empirical_quantile_mapping_future,
            o_hist=ff_obs[var],  # Historical observational data
            s_hist=ff_gcm_hist[var],  # Historical simulation data
            s_future=ff_gcm_future[var],  # Future simulation data
            kwargs={"n_quantiles": n_quantiles, "extrapolation": extrapolation},
            dask="parallelized",
            output_dtypes=[ff_gcm_future[var].dtype],
        )

    # Rescale the corrected fraction factors so that their sum equals one across each day
    rescaled_corrected_gcm = rescale_to_sum_one(corrected_ff_future)

    return rescaled_corrected_gcm


# def empirical_quantile_mapping(o, s, n_quantiles=100, extrapolation="constant"):
#     """
#     Empirical Quantile Mapping for a single variable.

#     Parameters:
#         o (np.ndarray or da.Array): Observational data.
#         s (np.ndarray or da.Array): Simulation data to be corrected.
#         n_quantiles (int): Number of quantiles.
#         extrapolation (str): Method for extrapolation ("constant" or other).

#     Returns:
#         np.ndarray or da.Array: Corrected simulation data.
#     """
#     # Ensure that inputs are either numpy arrays or dask arrays
#     o = da.asarray(o) if isinstance(o, np.ndarray) else o
#     s = da.asarray(s) if isinstance(s, np.ndarray) else s

#     # Calculate quantiles for observation and simulation
#     obs_quantiles = da.percentile(o, np.linspace(0, 100, n_quantiles))
#     sim_quantiles = da.percentile(s, np.linspace(0, 100, n_quantiles))

#     # Mapping function from simulation quantiles to observation quantiles
#     corrected_s = da.interp(s, sim_quantiles, obs_quantiles)

#     if extrapolation == "constant":
#         max_obs_diff = obs_quantiles[-1] - sim_quantiles[-1]
#         min_obs_diff = obs_quantiles[0] - sim_quantiles[0]

#         corrected_s = da.where(
#             s > sim_quantiles[-1],
#             s + max_obs_diff,
#             da.where(s < sim_quantiles[0], s + min_obs_diff, corrected_s),
#         )
#     else:
#         corrected_s = da.where(
#             s > sim_quantiles[-1],
#             obs_quantiles[-1],
#             da.where(s < sim_quantiles[0], obs_quantiles[0], corrected_s),
#         )

#     return corrected_s


# def empirical_quantile_mapping(o, s, n_quantiles=100, extrapolation="constant"):
#     """
#     Empirical Quantile Mapping for a single variable.

#     Parameters:
#         o (np.ndarray): Observational data.
#         s (np.ndarray): Simulation data to be corrected.
#         n_quantiles (int): Number of quantiles.
#         extrapolation (str): Method for extrapolation ("constant" or any other string).

#     Returns:
#         np.ndarray: Corrected simulation data.
#     """
#     # Calculate quantiles for observation and simulation
#     obs_quantiles = np.percentile(o, np.linspace(0, 100, n_quantiles))
#     sim_quantiles = np.percentile(s, np.linspace(0, 100, n_quantiles))

#     # Mapping function from simulation quantiles to observation quantiles
#     corrected_s = np.interp(s, sim_quantiles, obs_quantiles)

#     if extrapolation == "constant":
#         corrected_s[s > np.nanmax(sim_quantiles)] = s[s > np.nanmax(sim_quantiles)] + (
#             obs_quantiles[-1] - sim_quantiles[-1]
#         )
#         corrected_s[s < np.nanmin(sim_quantiles)] = s[s < np.nanmin(sim_quantiles)] + (
#             obs_quantiles[0] - sim_quantiles[0]
#         )
#     else:
#         corrected_s[s > np.nanmax(sim_quantiles)] = obs_quantiles[-1]
#         corrected_s[s < np.nanmin(sim_quantiles)] = obs_quantiles[0]

#     return corrected_s

# def empirical_quantile_mapping_xarray(
#     obs_ds, sim_ds, var_list, n_quantiles=100, extrapolation="constant"
# ):
#     """
#     Apply Empirical Quantile Mapping to an xarray Dataset.

#     Parameters:
#         obs_ds (xr.Dataset): Observational xarray Dataset.
#         sim_ds (xr.Dataset): Simulation xarray Dataset.
#         var_list (list): List of variable names to apply EQM to.
#         n_quantiles (int): Number of quantiles for EQM.
#         extrapolation (str): Method for extrapolation.

#     Returns:
#         xr.Dataset: Bias-corrected xarray Dataset.
#     """
#     corrected_ds = xr.Dataset()

#     for var in var_list:
#         obs_data = obs_ds[var]  # DataArray
#         sim_data = sim_ds[var]

#         # # Ensure data is loaded as Dask arrays
#         # if isinstance(obs_data.data, np.ndarray):
#         #     obs_data = obs_data.chunk()
#         # if isinstance(sim_data.data, np.ndarray):
#         #     sim_data = sim_data.chunk()

#         # Apply EQM using map_blocks
#         corrected_data = xr.apply_ufunc(
#             empirical_quantile_mapping,
#             obs_data,
#             sim_data,
#             kwargs={"n_quantiles": n_quantiles, "extrapolation": extrapolation},
#             dask="parallelized",
#             output_dtypes=[sim_data.dtype],
#         )

#         corrected_ds[var] = corrected_data

#     return corrected_ds


# def empirical_quantile_mapping_future(
#     o_hist, s_hist, s_future, n_quantiles=100, extrapolation="constant"
# ):
#     """
#     Empirical Quantile Mapping for future period using historical observational
#     and simulation data for calibration.

#     Parameters:
#         o_hist (np.ndarray): Historical observational data.
#         s_hist (np.ndarray): Historical simulation data used for calibration.
#         s_future (np.ndarray): Future simulation data to be corrected.
#         n_quantiles (int): Number of quantiles for the quantile mapping.
#         extrapolation (str): Method for extrapolation ("constant" or any other string).

#     Returns:
#         np.ndarray: Corrected future simulation data.
#     """
#     # Calculate quantiles for historical observation and simulation
#     obs_quantiles = np.percentile(o_hist, np.linspace(0, 100, n_quantiles))
#     sim_quantiles = np.percentile(s_hist, np.linspace(0, 100, n_quantiles))

#     # Interpolate to find corresponding future simulation quantiles
#     corrected_s_future = np.interp(s_future, sim_quantiles, obs_quantiles)

#     # Handle extrapolation for values outside the range of historical simulation data
#     if extrapolation == "constant":
#         corrected_s_future[s_future > np.nanmax(sim_quantiles)] = s_future[
#             s_future > np.nanmax(sim_quantiles)
#         ] + (obs_quantiles[-1] - sim_quantiles[-1])
#         corrected_s_future[s_future < np.nanmin(sim_quantiles)] = s_future[
#             s_future < np.nanmin(sim_quantiles)
#         ] + (obs_quantiles[0] - sim_quantiles[0])
#     else:
#         corrected_s_future[s_future > np.nanmax(sim_quantiles)] = obs_quantiles[-1]
#         corrected_s_future[s_future < np.nanmin(sim_quantiles)] = obs_quantiles[0]

#     return corrected_s_future


# def empirical_quantile_mapping_future_xarray(
#     obs_ds,
#     sim_ds,
#     sim_ds_future,
#     var_list,
#     n_quantiles=None,
#     extrapolation="constant",
# ):
#     """
#     Apply Empirical Quantile Mapping to an xarray Dataset.

#     Parameters:
#         obs_ds (xr.Dataset): Observational xarray Dataset for hist.
#         sim_ds (xr.Dataset): Simulation xarray Dataset for hist.
#         sim_ds_future (xr.Dataset): Simulation xarray Dataset for future.
#         var_list (list): List of variable names to apply EQM to.
#         n_quantiles (int): Number of quantiles for EQM.
#         extrapolation (str): Method for extrapolation, either "constant" or any other string.

#     Returns:
#         xr.Dataset: Bias-corrected xarray Dataset.
#     """
#     corrected_ds = xr.Dataset()

#     for var in var_list:
#         obs_data = obs_ds[var].values  # Extract numpy array from DataArray
#         sim_data = sim_ds[var].values
#         sim_data_future = sim_ds_future[var].values

#         # Assuming the first dimension is time
#         reshaped_obs_data = obs_data.reshape(
#             obs_data.shape[0], -1
#         )  # Reshape to 2D array
#         reshaped_sim_data = sim_data.reshape(sim_data.shape[0], -1)
#         reshaped_sim_data_future = sim_data_future.reshape(sim_data.shape[0], -1)

#         # corrected_data = np.empty(reshaped_sim_data.shape)
#         corrected_data = np.empty(reshaped_sim_data_future.shape)
#         corrected_data.fill(np.nan)

#         for i in range(reshaped_sim_data.shape[1]):
#             # EQM function is applied for each time series (each column in reshaped_data)
#             o, s, s_f = (
#                 reshaped_obs_data[:, i],
#                 reshaped_sim_data[:, i],
#                 reshaped_sim_data_future[:, i],
#             )

#             # Apply EQM
#             corrected_data[:, i] = empirical_quantile_mapping_future(
#                 o, s, s_f, n_quantiles=n_quantiles, extrapolation=extrapolation
#             )

#         # Reshape corrected_data back to original shape
#         corrected_data = corrected_data.reshape(sim_data_future.shape)

#         # Create a new DataArray and append to the corrected Dataset
#         corrected_ds[var] = xr.DataArray(
#             corrected_data,
#             dims=sim_ds_future[var].dims,
#             coords=sim_ds_future[var].coords,
#         )

#     return corrected_ds


def empirical_quantile_mapping_future_xarray(
    obs_ds,
    sim_ds,
    sim_ds_future,
    var_list,
    n_quantiles=100,
    extrapolation="constant",
):
    """
    Apply Empirical Quantile Mapping to an xarray Dataset.

    Parameters:
        obs_ds (xr.Dataset): Observational xarray Dataset for hist.
        sim_ds (xr.Dataset): Simulation xarray Dataset for hist.
        sim_ds_future (xr.Dataset): Simulation xarray Dataset for future.
        var_list (list): List of variable names to apply EQM to.
        n_quantiles (int): Number of quantiles for EQM.
        extrapolation (str): Method for extrapolation, either "constant" or any other string.

    Returns:
        xr.Dataset: Bias-corrected xarray Dataset.
    """
    corrected_ds = xr.Dataset()

    for var in var_list:
        obs_data = obs_ds[var]  # Extract numpy array from DataArray
        sim_data = sim_ds[var]
        sim_data_future = sim_ds_future[var]

        # Ensure data is loaded as Dask arrays
        if isinstance(obs_data.data, np.ndarray):
            obs_data = obs_data.chunk()
        if isinstance(sim_data.data, np.ndarray):
            sim_data = sim_data.chunk()
        if isinstance(sim_data_future.data, np.ndarray):
            sim_data_future = sim_data_future.chunk()

        # Apply EQM using map_blocks
        corrected_data = xr.apply_ufunc(
            empirical_quantile_mapping_future,
            obs_data,
            sim_data,
            sim_data_future,
            kwargs={"n_quantiles": n_quantiles, "extrapolation": extrapolation},
            dask="parallelized",
            output_dtypes=[sim_data.dtype],
        )

        corrected_ds[var] = corrected_data

    return corrected_ds


# def rescale_to_sum_one(ds):
#     """
#     Rescale the 6-hourly data in an xarray Dataset so that the sum across each day equals 1.

#     Parameters:
#         ds (xr.Dataset): The xarray Dataset with 6-hourly Dask arrays.

#     Returns:
#         xr.Dataset: Dataset with rescaled 6-hourly data.
#     """
#     rescaled_ds = xr.Dataset(coords=ds.coords)

#     for var in ds.data_vars:
#         data_6hr = ds[var].data

#         # Check if the time dimension is divisible by 4
#         if data_6hr.shape[0] % 4 != 0:
#             raise ValueError(
#                 "The time dimension must be divisible by 4 for 6-hourly data."
#             )

#         # Reshape and sum across each day
#         reshaped_data = data_6hr.rechunk({0: -1}).reshape(-1, 4)
#         daily_sum = da.sum(reshaped_data, axis=1)

#         # Rescale the 6-hourly data
#         rescaled_data_6hr = data_6hr / daily_sum.rechunk({0: -1}).repeat(4, axis=0)

#         # Assign the rescaled data back to the Dataset
#         rescaled_ds[var] = xr.DataArray(
#             rescaled_data_6hr, dims=ds[var].dims, coords=ds[var].coords
#         )

#     return rescaled_ds


def set_variables():
    # set variables
    nout = 12  # number of months
    leap = [0, 0, 0, 0]

    if time_scale == 0:
        if moving_window > 15:
            print("Maximum moving window value allowed is 15")
            exit()

    if time_scale == 0:  # daily: 0 or monthly: 1
        itmp = 0
    else:
        itmp = 1

    irho = np.zeros((5, 5), dtype=int)
    for i in range(
        itmp, 5, 1
    ):  # set irho matrix (bias correction options: 1-included and 0-excluded)
        if correction_model == 1:
            irho[i, :] = [1, 0, 0, 0, 0]
        if correction_model == 2:
            irho[i, :] = [1, 1, 0, 0, 0]
        if correction_model == 3:
            irho[i, :] = [1, 1, 1, 0, 0]
        if correction_model == 4:
            irho[i, :] = [1, 1, 1, 1, 0]

    isn = 4  # number of seasons in a year
    ij = [3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]  # Number of months in each season
    a = [
        [12, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # isj=np.zeros((1,ij[0]),dtype=int)
    isj = np.zeros(
        (1, 12), dtype=int
    )  # Month numbering assigned to each season (1-Jan, 2-Feb......, 12-Dec)
    for i in range(0, isn, 1):
        # b=[int(a[i][0]),int(a[i][1]),int(a[i][2])]
        b = [int(a[i][j]) for j in range(0, 12)]
        # isjs=np.zeros((1,ij[i]),dtype=int)
        isj = np.r_[isj, [b]]
    isj = np.delete(isj, (0), axis=0)

    mx_th = np.zeros((1, 6), dtype=float)
    for i in range(0, no_of_variables, 1):
        b = [0, lower_limit[i], upper_limit[i], 0, 0, 0]
        mx_th = np.r_[mx_th, [b]]
    mx_th = np.delete(mx_th, (0), axis=0)
    phlwr, phupr, ilimit, thres = (
        mx_th[:, 1],
        mx_th[:, 2],
        mx_th[:, 4],
        mx_th[:, 5],
    )

    for i in range(0, no_of_variables, 1):
        if thres[i] == 0.0:
            thres[i] = 0.00000001

    # if (
    #     time_scale == 0
    # ):  # read additional information about number of days in a year for each dataset
    #     isum = sum(leap)

    a1 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    idays = np.zeros([nout, 4])
    for i in range(0, 4):
        for j in range(0, nout):
            idays[j, i] = a1[j]

    return nout, leap, irho, isn, ij, isj, phlwr, phupr, ilimit, thres, idays


# Define a custom class to hold the bias correction parameters
@dataclass
class BiasCorrectionParams:
    avdc_iter: np.ndarray
    sddc_iter: np.ndarray
    avmc_iter: np.ndarray
    sdmc_iter: np.ndarray
    avsc_iter: np.ndarray
    sdsc_iter: np.ndarray
    avyc_iter: np.ndarray
    sdyc_iter: np.ndarray
    cmod_iter: np.ndarray
    gmod_iter: np.ndarray
    cmodm_iter: np.ndarray
    gmodm_iter: np.ndarray
    cmods_iter: np.ndarray
    gmods_iter: np.ndarray
    cmody_iter: np.ndarray
    gmody_iter: np.ndarray
    avdh: np.ndarray
    sddh: np.ndarray
    avmh: np.ndarray
    sdmh: np.ndarray
    avsh: np.ndarray
    sdsh: np.ndarray
    avyh: np.ndarray
    sdyh: np.ndarray
    cobs: np.ndarray
    dobs: np.ndarray
    cobsm: np.ndarray
    dobsm: np.ndarray
    cobss: np.ndarray
    dobss: np.ndarray
    cobsy: np.ndarray
    dobsy: np.ndarray
    phll: np.ndarray
    phul: np.ndarray
    missing_value: float

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return BiasCorrectionParams(**data)


# @dataclass
# class BiasCorrectionParams:
#     def __init__(
#         self,
#         avdc_iter,
#         sddc_iter,
#         avmc_iter,
#         sdmc_iter,
#         avsc_iter,
#         sdsc_iter,
#         avyc_iter,
#         sdyc_iter,
#         cmod_iter,
#         gmod_iter,
#         cmodm_iter,
#         gmodm_iter,
#         cmods_iter,
#         gmods_iter,
#         cmody_iter,
#         gmody_iter,
#         avdh,
#         sddh,
#         avmh,
#         sdmh,
#         avsh,
#         sdsh,
#         avyh,
#         sdyh,
#         cobs,
#         dobs,
#         cobsm,
#         dobsm,
#         cobss,
#         dobss,
#         cobsy,
#         dobsy,
#         phll,
#         phul,
#         missing_value,
#     ):
#         self.avdc_iter = avdc_iter
#         self.sddc_iter = sddc_iter
#         self.avmc_iter = avmc_iter
#         self.sdmc_iter = sdmc_iter
#         self.avsc_iter = avsc_iter
#         self.sdsc_iter = sdsc_iter
#         self.avyc_iter = avyc_iter
#         self.sdyc_iter = sdyc_iter
#         self.cmod_iter = cmod_iter
#         self.gmod_iter = gmod_iter
#         self.cmodm_iter = cmodm_iter
#         self.gmodm_iter = gmodm_iter
#         self.cmods_iter = cmods_iter
#         self.gmods_iter = gmods_iter
#         self.cmody_iter = cmody_iter
#         self.gmody_iter = gmody_iter
#         self.avdh = avdh
#         self.sddh = sddh
#         self.avmh = avmh
#         self.sdmh = sdmh
#         self.avsh = avsh
#         self.sdsh = sdsh
#         self.avyh = avyh
#         self.sdyh = sdyh
#         self.cobs = cobs
#         self.dobs = dobs
#         self.cobsm = cobsm
#         self.dobsm = dobsm
#         self.cobss = cobss
#         self.dobss = dobss
#         self.cobsy = cobsy
#         self.dobsy = dobsy
#         self.phll = phll
#         self.phul = phul
#         self.missing_value = missing_value
# # Method to convert object to a serializable format
# def to_dict(self):
#     return {
#         "avdc_iter": self.avdc_iter,
#         "sddc_iter": self.sddc_iter,
#         "avmc_iter": self.avmc_iter,
#         "sdmc_iter": self.sdmc_iter,
#         "avsc_iter": self.avsc_iter,
#         "sdsc_iter": self.sdsc_iter,
#         "avyc_iter": self.avyc_iter,
#         "sdyc_iter": self.sdyc_iter,
#         "cmod_iter": self.cmod_iter,
#         "gmod_iter": self.gmod_iter,
#         "cmodm_iter": self.cmodm_iter,
#         "gmodm_iter": self.gmodm_iter,
#         "cmods_iter": self.cmods_iter,
#         "gmods_iter": self.gmods_iter,
#         "cmody_iter": self.cmody_iter,
#         "gmody_iter": self.gmody_iter,
#         "avdh": self.avdh,
#         "sddh": self.sddh,
#         "avmh": self.avmh,
#         "sdmh": self.sdmh,
#         "avsh": self.avsh,
#         "sdsh": self.sdsh,
#         "avyh": self.avyh,
#         "sdyh": self.sdyh,
#         "cobs": self.cobs,
#         "dobs": self.dobs,
#         "cobsm": self.cobsm,
#         "dobsm": self.dobsm,
#         "cobss": self.cobss,
#         "dobss": self.dobss,
#         "cobsy": self.cobsy,
#         "dobsy": self.dobsy,
#         "phll": self.phll,
#         "phul": self.phul,
#         "missing_value": self.missing_value,
#     }

# # Static method to create an object from a dictionary
# @staticmethod
# def from_dict(data):
#     return BiasCorrectionParams(
#         data["avdc_iter"],
#         data["sddc_iter"],
#         data["avmc_iter"],
#         data["sdmc_iter"],
#         data["avsc_iter"],
#         data["sdsc_iter"],
#         data["avyc_iter"],
#         data["sdyc_iter"],
#         data["cmod_iter"],
#         data["gmod_iter"],
#         data["cmodm_iter"],
#         data["gmodm_iter"],
#         data["cmods_iter"],
#         data["gmods_iter"],
#         data["cmody_iter"],
#         data["gmody_iter"],
#         data["avdh"],
#         data["sddh"],
#         data["avmh"],
#         data["sdmh"],
#         data["avsh"],
#         data["sdsh"],
#         data["avyh"],
#         data["sdyh"],
#         data["cobs"],
#         data["dobs"],
#         data["cobsm"],
#         data["dobsm"],
#         data["cobss"],
#         data["dobss"],
#         data["cobsy"],
#         data["dobsy"],
#         data["phll"],
#         data["phul"],
#         data["missing_value"],
#     )
# def to_dict(self):
#     # Automatically include all instance attributes
#     return {attr: getattr(self, attr) for attr in vars(self)}

# @staticmethod
# def from_dict(data):
#     # Automatically unpack dictionary to attributes
#     return BiasCorrectionParams(**data)


def bc_correction_hist(gcm_reshape, obs_reshape):
    # Description: Performs bias correction on GCM data using observational data.
    # Input:
    # gcm_reshape (np.ndarray): The reshaped GCM data.
    # obs_reshape (np.ndarray): The reshaped observational data.
    # Parameters:
    # idays (int): Day information.
    # leap (bool): Indicator for leap year.
    # isn, ij, isj (int): Seasonal information variables.
    # phlwr, phupr (float): Lower and upper correction bounds.
    # irho (int): Correction options.
    # nout, i1, ilimit, thres (int, float): Various other variables for
    # the correction process.
    # Output:
    # Implicitly modifies data in-place or outputs data to external medium;
    # no explicit return value.

    # Workflow:
    # The function is designed to be invoked after assign_w and
    # extract_and_reshape functions have processed the original 3D and 2D climate data
    # from GCM and observational sources.

    # The idays, isn, ij, and isj parameters are used for subsetting the data
    # according to different seasons and months.

    # The phlwr and phupr parameters set the lower and upper limits for bias correction.
    # Any values outside these limits might be adjusted or flagged.

    # The irho matrix controls the type of bias correction to be applied,
    # supporting multiple models.

    # The function performs the bias correction on gcm_reshape using obs_reshape
    # as a reference and returns the corrected data,
    # usually for further analysis or to be written back to a NetCDF file.

    # The function is intended to operate within a framework where its settings and
    # prerequisites are provided by modules like user_input and others.

    # ==========================================================================#

    nout, leap, irho, isn, ij, isj, phlwr, phupr, ilimit, thres, idays = set_variables()

    # Exclude specific humidity if the first element of the GCM data is less than 0.1
    # to avoid unexpected values during the bias correction process
    if bc_boundary == "lateral":
        if correction_model == 4 and np.max(gcm_reshape[2, :, :, :]) < 0.1:
            print("Exclude specific humidity below the lower threshold (1*10-4).")
            if time_scale == 0:  # daily: 0 or monthly: 1
                itmp = 0
            else:
                itmp = 1

            irho = np.zeros((5, 5), dtype=int)
            for i in range(
                itmp, 5, 1
            ):  # set irho matrix (bias correction options: 1-included and 0-excluded)
                irho[i, :] = [1, 1, 1, 0, 0]

    nycur = endyear_h - startyear_h + 1
    nsc = startyear_h - 1
    nvar = no_of_variables

    # nday=np.zeros([constants.monmax,])
    nday = mbc.day()
    rem = np.zeros([no_of_variables, nycur, 12])
    res = np.zeros([no_of_variables, nycur, 12])
    rey = np.zeros([no_of_variables, nycur])

    # 	calculate annual, seasonal and monthly series of reanalysis data
    # for k in range(0, no_of_variables):
    #     for i in range(0, nycur):
    #         ss1 = np.zeros(
    #             [
    #                 4,
    #             ]
    #         )
    #         for j in range(0, 12):
    #             iss = mbc.iseas(j + 1, isn, ij, isj)
    #             if time_scale != 0:
    #                 jd = 1
    #             else:
    #                 jd = int(idays[j, 2])
    #                 if leap[2] == 0:
    #                     jd = int(nday[j])
    #                     if j == 1:
    #                         jd = int(mbc.daycount(nsc, i + 1))
    #             sm1 = 0.0
    #             i1 = 0
    #             for day_index in range(0, jd):
    #                 if obs_reshape[k, i, j, day_index] > missing_value:
    #                     i1 += 1
    #                     sm1 += obs_reshape[k, i, j, day_index]

    #             rem[k, i, j] = sm1 / float(i1)
    #             ss1[iss - 1] += rem[k, i, j]

    #         sy1 = 0.0
    #         for iss in range(0, isn):
    #             res[k, i, iss] = ss1[iss] / ij[iss]
    #             sy1 += res[k, i, iss]

    #         rey[k, i] = sy1 / float(isn)

    # Calculate annual, seasonal, and monthly series of reanalysis data
    for k in range(nvar):
        for i in range(nycur):
            ss1 = np.zeros(4)  # Seasonal sum for each year
            for j in range(12):
                iss = mbc.iseas(j + 1, isn, ij, isj)

                # Set jd based on time_scale and leap year
                if time_scale != 0:
                    jd = 1
                else:
                    jd = int(idays[j, 2]) if leap[2] != 0 else int(nday[j])
                    if j == 1:
                        jd = int(mbc.daycount(nsc, i + 1))

                # Calculate monthly sum (sm1) and count valid days (i1)
                valid_obs = obs_reshape[k, i, j, :jd] > missing_value
                i1 = np.sum(valid_obs)  # Count valid days
                sm1 = np.sum(
                    obs_reshape[k, i, j, :jd][valid_obs]
                )  # Sum valid observations

                rem[k, i, j] = (
                    sm1 / float(i1) if i1 > 0 else missing_value
                )  # Avoid division by zero
                ss1[iss - 1] += rem[k, i, j]  # Accumulate seasonal sum

            # Calculate seasonal averages
            for iss in range(isn):
                res[k, i, iss] = ss1[iss] / ij[iss]

            # Calculate annual average
            rey[k, i] = np.mean(res[k, i, :isn])

    # ==========================================================================#
    # variables to be used
    # changed
    phll = np.zeros([10, no_of_variables])
    phul = np.zeros([10, no_of_variables])
    phll[phll == 0] = 100000.0
    phul[phul == 0] = -100000.0

    inx = 0

    # ==========================================================================#
    ######## calculate daily, monthly, seasonal, annual stats of obs ########
    # avdh, sddh, cordh = mbc.sdsmooth(obs_reshape, moving_window, nday,
    #                                 nsc, 3, leap, idays, missing_value)
    avdh, sddh, cordh = mbc.sdsmooth(
        obs_reshape, moving_window, nday, nsc, 3, leap, idays, missing_value
    )
    # cobs,dobs = c_g_corl_daily(atm,iband,nday,ns,ilpg,leap,idays,inx,miss,[nvar,ny,nout])
    # c_g_corl_daily(atm,cobs,dobs,iband,nvar,nday,ny,ns,ilpg,leap,idays,inx,miss,nout
    cobs, dobs = mbc.c_g_corl_daily(
        obs_reshape, moving_window, nday, nsc, 3, leap, idays, inx, missing_value
    )

    # Find monthly means, sds and corl of reanalysis data
    # avsds(atm,nvar,ny,av,sd,rho,nout)
    avmh, sdmh, cormh = mbc.avsds(rem)

    # Calculate matrices C and G
    # c_g_corl_season(atm,nvar,ny,cobs,dobs,inx,nout)
    cobsm, dobsm = mbc.c_g_corl_season(rem, inx)

    # Find seasonal, sds and corl of reanalysis data
    avsh, sdsh, corsh = mbc.avsds(res)

    # Calculate matrices C and G
    cobss, dobss = mbc.c_g_corl_season(res, inx)

    # Find annual, sds and corl of reanalysis data
    # avsdy(atm,nvar,ny,avy,sdy,rhoy)
    avyh, sdyh, coryh = mbc.avsdy(rey)

    # Calculate matrices C and G
    # c_g_corl_year(atm,nvar,ny,cobs,dobs,inx)
    cobsy, dobsy = mbc.c_g_corl_year(rey, inx)

    inx = 2
    # ==========================================================================#
    #
    # Loop structures
    nntr = no_of_iterations + 1
    nxt = 0
    for i in range(5):  # Loops from 0 to 4 inclusive
        for j in range(3, 5):  # Loops from 3 to 4 inclusive
            nxt += irho[i, j]

    if nxt > 0:
        nntr = no_of_iterations + 2

    ngcur = endyear_h - startyear_h + 1
    nsgc = startyear_h - 1
    nsmax = 4
    nss = 4
    tprint = 0
    iss = 0

    if tprint == 1:
        print("Start loop, first: boundary limits, next: correction")

    # Initialize arrays to avoid unbound local error
    # day
    avdc = np.zeros((nvar, 12, 31))
    sddc = np.zeros((nvar, 12, 31))
    cordc = np.zeros((nvar, 12, 31))
    # month
    avmc = np.zeros((nvar, 12))
    sdmc = np.zeros((nvar, 12))
    cormc = np.zeros((nvar, 12))
    # season
    avsc = np.zeros((nvar, 12))
    sdsc = np.zeros((nvar, 12))
    corsc = np.zeros((nvar, 12))
    # year
    avyc = np.zeros((nvar))
    sdyc = np.zeros((nvar))
    coryc = np.zeros((nvar))

    # day
    avdc_iter = np.zeros((nntr, nvar, 12, 31))
    sddc_iter = np.zeros((nntr, nvar, 12, 31))
    cmod_iter = np.zeros((nntr, 12, 31, nvar, nvar))
    gmod_iter = np.zeros((nntr, 12, 31, nvar, nvar))
    # month
    avmc_iter = np.zeros((nntr, nvar, 12))
    sdmc_iter = np.zeros((nntr, nvar, 12))
    cmodm_iter = np.zeros((nntr, 12, nvar, nvar))
    gmodm_iter = np.zeros((nntr, 12, nvar, nvar))
    # season
    avsc_iter = np.zeros((nntr, nvar, 12))
    sdsc_iter = np.zeros((nntr, nvar, 12))
    cmods_iter = np.zeros((nntr, 12, nvar, nvar))
    gmods_iter = np.zeros((nntr, 12, nvar, nvar))
    # year
    avyc_iter = np.zeros((nntr, nvar))
    sdyc_iter = np.zeros((nntr, nvar))
    cmody_iter = np.zeros((nntr, nvar, nvar))
    gmody_iter = np.zeros((nntr, nvar, nvar))

    # Initialize input arrays
    gcmc = gcm_reshape.copy()
    # gcmf = gcm_reshape.copy()

    gm = np.zeros((nvar, 31, 12))
    gs = np.zeros((nvar, 31, 12))
    gy = np.zeros((nvar, 31))

    # Account for days in February in leap years
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for itr in range(nntr):
        jj = 0

        iyr = ngcur
        ns1 = nsgc
        gcm = gcmc.copy()

        if tprint == 1 and itr > 1:
            print(" Applying bias correction for current climate ", itr)

        gd = gcm.copy()
        ggd = gd.copy()
        gdct = gd.copy()

        # find smoothened mean and sd of daily gcm series
        if time_scale == 0 and jj == 0:  # historical and daily
            avdc, sddc, cordc = mbc.sdsmooth(
                gd, moving_window, nday, nsc, 3, leap, idays, missing_value
            )

        if itr == 0:
            # -------------------------------------------------------------------------
            # Start loop, first: boundary limits, next: correction
            # -------------------------------------------------------------------------
            if time_scale != 0:
                # changed
                for k in range(nvar):
                    for i in range(iyr):
                        for j in range(nout):
                            if j == 1 and is_leap_year(iyr + startyear_h):
                                days = 29
                            else:
                                days = days_in_month[j]

                            gm[k, i, j] = np.sum(gdct[k, i, j, :days]) / days

                ggm = gm.copy()
                gmct = gm.copy()

                # 2. Second Segment
                avmc, sdmc, cormc = mbc.avsds(gm)  # Adjust the function call if needed

                # Save the current avmc and sdmc arrays to the larger array
                avmc_iter[itr, :nvar, :] = avmc.copy()
                sdmc_iter[itr, :nvar, :] = sdmc.copy()

                # 3. Third Segment
                ggm = (gm - avmc[:, np.newaxis, :]) * sdmh[:, np.newaxis, :] / sdmc[
                    :, np.newaxis, :
                ] + avmh[:, np.newaxis, :]
                gmct = ggm.copy()

                # 4. Fourth Segment
                potential_phul = gmct + 3 * sdmh[:, np.newaxis, :]
                potential_phll = gmct - 3 * sdmh[:, np.newaxis, :]

                phul[jj, :] = np.maximum(potential_phul.max(axis=(1, 2)), phul[jj, :])
                phll[jj, :] = np.minimum(potential_phll.min(axis=(1, 2)), phll[jj, :])

            else:
                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]
                        # Adjust ggd
                        ggd[:, i, j, :jd] = (gd[:, i, j, :jd] - avdc[:, j, :jd]) * sddh[
                            :, j, :jd
                        ] / sddc[:, j, :jd] + avdh[:, j, :jd]

                # Copy the computed values to gdct
                gdct = ggd.copy()

                non_zero_mask = (gd > missing_value) & (
                    gd != 0.0
                )  # Adjust the threshold as needed
                ac = np.ones_like(gd)
                ac[non_zero_mask] = gdct[non_zero_mask] / gd[non_zero_mask]
                # ac[(gdct < 0.1) & (gd < 0.1)] = 1.0

                # Calculate the potential new maximum values for bc_params.phul
                potential_phul = gcm * ac + 3 * sddh[:, np.newaxis, :, :]

                # Update the bc_params.phul values if the condition is met
                phul[jj, :] = np.where(
                    potential_phul.max(axis=(1, 2, 3)) > phul[jj, :],
                    potential_phul.max(axis=(1, 2, 3)),
                    phul[jj, :],
                )

                # Calculate the potential new minimum values for bc_params.phll
                potential_phll = gcm * ac - 3 * sddh[:, np.newaxis, :, :]

                # Update the bc_params.phll values if the condition is met
                phll[jj, :] = np.where(
                    potential_phll.min(axis=(1, 2, 3)) < phll[jj, :],
                    potential_phll.min(axis=(1, 2, 3)),
                    phll[jj, :],
                )

                # Returning the first few values of 'bc_params.phul' and 'bc_params.phll' for verification
                # bc_params.phul[0, :5], bc_params.phll[0, :5]

            # for j in range(no_of_variables):
            #     if phll[jj, j] < phlwr[j] * 4:
            #         phll[jj, j] = phlwr[j] * 4
            #     if phul[jj, j] > phupr[j] * 4:
            #         phul[jj, j] = phupr[j] * 4
            # changed threshold considering assign_6hr function that times 10 for wind speed
            for j in range(no_of_variables):
                if config.boundary == "lateral" and j == 0:
                    if phll[jj, j] < phlwr[j] * 10:
                        phll[jj, j] = phlwr[j] * 10
                    if phul[jj, j] > phupr[j] * 10:
                        phul[jj, j] = phupr[j] * 10
                else:
                    if phll[jj, j] < phlwr[j]:
                        phll[jj, j] = phlwr[j]
                    if phul[jj, j] > phupr[j]:
                        phul[jj, j] = phupr[j]
            # for j in range(no_of_variables):
            #     if phll[jj, j] < phlwr[j]:
            #         phll[jj, j] = phlwr[j]
            #     if phul[jj, j] > phupr[j]:
            #         phul[jj, j] = phupr[j]

            # -------------------------------------------------------------------------
            # End loop, first: boundary limits, next: correction
            # -------------------------------------------------------------------------

        elif itr > 0:
            if irho[0, 0] != 0:  # hist: daily mean, future: - avdc + avdh
                if tprint == 1:
                    print("Correcting for daily mean", jj)

                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]

                        # Calculate new values for ggd
                        ggd[:, i, j, :jd] = (gd[:, i, j, :jd] - avdc[:, j, :jd]) + avdh[
                            :, j, :jd
                        ]

                avdc_iter[itr, :, :, :] = avdc.copy()

                # Modify ggd based on bc_params.phul and bc_params.phll conditions
                ggd = np.where(
                    ggd > phul[jj, :, None, None, None],
                    phul[jj, :, None, None, None],
                    ggd,
                )
                ggd = np.where(
                    ggd < phll[jj, :, None, None, None],
                    phll[jj, :, None, None, None],
                    ggd,
                )

                # Copy ggd values to gdct
                gdct = ggd.copy()
                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )
                    print("a4 mean gcmc avd (:,4,:3)", avd[:, 4, :3])
                    print("a4 mean obs avdh (:,4,:3)", avdh[:, 4, :3])
            if irho[0, 1] != 0:  # hist: daily sd, future: (x - avd)*sddh/sddc + avd
                if tprint == 1:
                    print("Correcting for daily mean and standard deviation", jj)

                # historical
                avd, sddc, cordc = mbc.sdsmooth(
                    ggd,
                    moving_window,
                    nday,
                    nsc,
                    3,
                    leap,
                    idays,
                    missing_value,
                )

                sddc_iter[itr, :, :, :] = sddc.copy()

                # Calculate fact
                non_zero_mask = sddc > 1e-10
                fact = np.ones_like(sddc)
                fact[non_zero_mask] = sddh[non_zero_mask] / sddc[non_zero_mask]
                fact[(sddh < 0.1) & (sddc < 0.1)] = 1.0

                # correct for bias in daily sd
                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]

                        # Adjust ggd
                        ggd[:, i, j, :jd] = (ggd[:, i, j, :jd] - avd[:, j, :jd]) * fact[
                            :, j, :jd
                        ] + avd[:, j, :jd]

                ggd = np.where(
                    ggd > phul[jj, :, None, None, None],
                    phul[jj, :, None, None, None],
                    ggd,
                )
                ggd = np.where(
                    ggd < phll[jj, :, None, None, None],
                    phll[jj, :, None, None, None],
                    ggd,
                )
                # Copy ggd values to gdct
                gdct = ggd.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )
                    print("a4 std gcmc sdd (:,4,:3)", sdd[:, 4, :3])
                    print("a4 std obs sddh (:,4,:3)", sddh[:, 4, :3])

            if (
                irho[0, 2] != 0 or irho[0, 3] != 0
            ):  # hist: daily corl, future: cmod, gmod, cobs, gobs
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit daily corrl itr, jj", itr, jj)
                    pass
                else:
                    if tprint == 1 and itr > 1:
                        print("Correcting for daily corrl", jj)
                    # correct for bias in corl
                    # find smoothened mean and sd of gcm mean and sd corrected series
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )

                    # Conditional block for additional function call
                    cmod, gmod = mbc.c_g_corl_daily(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        2,
                        missing_value,
                    )

                    cmod_iter[itr, :, :, :, :] = cmod.copy()
                    gmod_iter[itr, :, :, :, :] = gmod.copy()

                    # Calculate fact
                    non_zero_mask = sdd > 1e-10
                    fact = np.ones_like(sdd)
                    fact[non_zero_mask] = sdd[non_zero_mask]
                    fact[(sdd < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for j in range(nout):
                            if j == 1 and is_leap_year(iyr + startyear_h):
                                jd = 29
                            else:
                                jd = days_in_month[j]

                            for day_index in range(jd):
                                gmg = gmod[j, day_index, :nvar, :nvar]
                                cmg = cmod[j, day_index, :nvar, :nvar]
                                go = dobs[j, day_index, :nvar, :nvar]
                                co = cobs[j, day_index, :nvar, :nvar]

                                bt[:nvar] = (
                                    ggd[:nvar, i, j, day_index]
                                    - avd[:nvar, j, day_index]
                                ) / sdd[:nvar, j, day_index]

                                if i == 0 and j == 0 and day_index == 0:
                                    gprev[:nvar] = bt[:nvar]
                                    btprev[:nvar] = bt[:nvar]

                                temp = np.matmul(go, gmg)
                                temp1 = np.matmul(temp, bt)
                                temp4 = np.matmul(co, gprev)
                                temp2 = np.matmul(temp, cmg)
                                temp3 = np.matmul(temp2, btprev)

                                gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                                gprev[:nvar] = gcur[:nvar]
                                btprev[:nvar] = bt[:nvar]
                                gdct[:nvar, i, j, day_index] = (
                                    gcur[:nvar] * sdd[:nvar, j, day_index]
                                    + avd[:nvar, j, day_index]
                                )
                                gdct[:nvar, i, j, day_index] = np.clip(
                                    gdct[:nvar, i, j, day_index],
                                    phll[jj, :nvar],
                                    phul[jj, :nvar],
                                )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avd, sdd, cord = mbc.sdsmooth(
                            gdct,
                            moving_window,
                            nday,
                            ns1,
                            jj + 1,
                            leap,
                            idays,
                            missing_value,
                        )

                        print("a4 std gcmc cord (:,4,:3)", cord[:, 4, :3])
                        print("a4 std obs cordh (:,4,:3)", cordh[:, 4, :3])

            # ------- end calculate daily series of gcm series and store -------

            # ------- calculate monthly series of gcm series and store --------

            for k in range(nvar):
                for i in range(iyr):
                    for j in range(nout):
                        if time_scale != 0:
                            jd = 1
                        elif j == 1 and is_leap_year(iyr + startyear_h):
                            days = 29
                        else:
                            days = days_in_month[j]

                        gm[k, i, j] = np.sum(gdct[k, i, j, :days]) / days

            ggm = gm.copy()
            gmct = gm.copy()

            if irho[1, 0] != 0:  # hist: monthly mean, future: - avmc + avmh
                if tprint == 1 and itr > 0:
                    print("Correcting for monthly mean", jj)

                # find mean and sd of gcm monthly series, historical
                avmc, sdmc, cormc = mbc.avsds(gm)

                avmc_iter[itr, :, :] = avmc.copy()

                # Calculate new values for ggm
                for i in range(iyr):
                    ggm[:, i, :] = (gm[:, i, :] - avmc) + avmh

                # Modify ggd based on bc_params.phul and bc_params.phll conditions
                ggm = np.where(
                    ggm > phul[jj, :, None, None],
                    phul[jj, :, None, None],
                    ggm,
                )
                ggm = np.where(
                    ggm < phll[jj, :, None, None],
                    phll[jj, :, None, None],
                    ggm,
                )

                # Copy ggd values to gdct
                gmct = ggm.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avm, sdm, corm = mbc.avsds(ggm)

                    print("a4 mean gcmc avm (:,:)", avm[:, :])
                    print("a4 mean obs avmh (:,:)", avmh[:, :])

            if irho[1, 1] != 0:  # hist: monthly sd, future: (x - avm)*sdmh/sdmc + avm
                if tprint == 1:
                    print("Correcting for monthly mean and standard deviation", jj)

                # find mean and sd of gcm mean corrected monthly series, historical
                avm, sdmc, cormc = mbc.avsds(ggm)

                sdmc_iter[itr, :, :] = sdmc.copy()

                # Calculate fact
                non_zero_mask = sdmc > 1e-10
                fact = np.ones_like(sdmc)
                fact[non_zero_mask] = sdmh[non_zero_mask] / sdmc[non_zero_mask]
                fact[(sdmh < 0.1) & (sdmc < 0.1)] = 1.0

                # correct for bias in monthly sd
                for i in range(iyr):
                    # Adjust ggm
                    ggm[:, i, :] = (ggm[:, i, :] - avm) * fact + avm

                ggm = np.where(
                    ggm > phul[jj, :, None, None],
                    phul[jj, :, None, None],
                    ggm,
                )
                ggm = np.where(
                    ggm < phll[jj, :, None, None],
                    phll[jj, :, None, None],
                    ggm,
                )

                # Copy ggd values to gdct
                gmct = ggm.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avm, sdm, corm = mbc.avsds(ggm)

                    print("a4 mean gcmc sdm (:,:)", sdm[:, :])
                    print("a4 mean obs sdmh (:,:)", sdmh[:, :])

            if (
                irho[1, 2] != 0 or irho[1, 3] != 0
            ):  # hist: monthly corl, future: cmodm, gmodm, cobsm, gobsm
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("exit monthly corrl itr, jj", itr, jj)
                    pass

                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for monthly corrl", jj)

                    avm, sdm, corm = mbc.avsds(ggm)

                    # Conditional block for additional function call
                    cmodm, gmodm = mbc.c_g_corl_season(ggm, inx)

                    cmodm_iter[itr, :, :, :] = cmodm.copy()
                    gmodm_iter[itr, :, :, :] = gmodm.copy()

                    # Calculate fact
                    non_zero_mask = sdm > 1e-10
                    fact = np.ones_like(sdm)
                    fact[non_zero_mask] = sdm[non_zero_mask]
                    fact[(sdm < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for j in range(nout):
                            gmg = gmodm[j, :nvar, :nvar]
                            cmg = cmodm[j, :nvar, :nvar]
                            go = dobsm[j, :nvar, :nvar]
                            co = cobsm[j, :nvar, :nvar]

                            bt[:nvar] = (ggm[:nvar, i, j] - avm[:nvar, j]) / sdm[
                                :nvar, j
                            ]

                            if i == 0 and j == 0:
                                gprev[:nvar] = bt[:nvar]
                                btprev[:nvar] = bt[:nvar]

                            temp = np.matmul(go, gmg)
                            temp1 = np.matmul(temp, bt)
                            temp4 = np.matmul(co, gprev)
                            temp = np.matmul(go, gmg)
                            temp2 = np.matmul(temp, cmg)
                            temp3 = np.matmul(temp2, btprev)

                            gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                            gprev[:nvar] = gcur[:nvar]
                            btprev[:nvar] = bt[:nvar]
                            gmct[:nvar, i, j] = (
                                gcur[:nvar] * sdm[:nvar, j] + avm[:nvar, j]
                            )
                            gmct[:nvar, i, j] = np.clip(
                                gmct[:nvar, i, j],
                                phll[jj, :nvar],
                                phul[jj, :nvar],
                            )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avm, sdm, corm = mbc.avsds(gmct)

                        print("a4 mean gcmc corm (:,:4)", corm[:, :4])
                        print("a4 mean obs cormh (:,:4)", cormh[:, :4])

            # ------- end calculate monthly series of gcm series and store --------

            # ------- form seasonal series of corrected monthly gcm series and store --

            for k in range(nvar):
                for i in range(iyr):
                    ss1 = np.zeros(nsmax)
                    for j in range(nout):
                        iss = mbc.iseas(j + 1, nss, ij, isj)
                        ss1[iss - 1] += gmct[k, i, j]
                    for iss in range(nss):
                        gs[k, i, iss] = ss1[iss] / ij[iss]

            ggs = gs.copy()
            gsct = gs.copy()

            if irho[2, 0] != 0:  # hist: seasonal mean, future: - avsc + avsh
                if tprint == 1 and itr > 0:
                    print("Correcting for seasonal mean", jj)

                avsc, sdsc, corsc = mbc.avsds(gs)

                avsc_iter[itr, :, :] = avsc.copy()

                # Calculate new values for ggm
                for i in range(iyr):
                    ggs[:, i, :] = (gs[:, i, :] - avsc) + avsh

                # Copy ggd values to gdct
                gsct = ggs.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avs, sds, cors = mbc.avsds(ggs)

                    print("a4 mean gcmc avs (:,:)", avs[:, :])
                    print("a4 mean obs avsh (:,:)", avsh[:, :])

            if irho[2, 1] != 0:  # hist: seasonal sd, future: (x - avs)*sdsh/sdsc + avs
                if tprint == 1:
                    print("Correcting for seasonal mean and standard deviation", jj)

                avs, sdsc, corsc = mbc.avsds(ggs)

                sdsc_iter[itr, :, :] = sdsc.copy()

                # Calculate fact
                non_zero_mask = sdsc > 1e-10
                fact = np.ones_like(sdsc)
                fact[non_zero_mask] = sdsh[non_zero_mask] / sdsc[non_zero_mask]
                fact[(sdsh < 0.1) & (sdsc < 0.1)] = 1.0

                for i in range(iyr):
                    # Adjust ggm
                    ggs[:, i, :] = (ggs[:, i, :] - avs) * fact + avs

                # Copy ggd values to gdct
                gsct = ggs.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avs, sds, cors = mbc.avsds(ggs)

                    print("a4 mean gcmc sds (:,:)", sds[:, :])
                    print("a4 mean obs sdsh (:,:)", sdsh[:, :])

            if (
                irho[2, 2] != 0 or irho[2, 3] != 0
            ):  # hist: seasonal corl, future: cmods, gmods, cobss, gobss
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit seasonal corrl itr, jj", itr, jj)
                    pass
                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for seasonal corrl", jj)

                    avs, sds, cors = mbc.avsds(ggs)

                    # Conditional block for additional function call
                    cmods, gmods = mbc.c_g_corl_season(ggs, inx)

                    cmods_iter[itr, :, :, :] = cmods.copy()
                    gmods_iter[itr, :, :, :] = gmods.copy()

                    # Calculate fact
                    non_zero_mask = sds > 1e-10
                    fact = np.ones_like(sds)
                    fact[non_zero_mask] = sds[non_zero_mask]
                    fact[(sds < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for iss in range(nss):
                            gmg = gmods[iss, :nvar, :nvar]
                            cmg = cmods[iss, :nvar, :nvar]
                            go = dobss[iss, :nvar, :nvar]
                            co = cobss[iss, :nvar, :nvar]

                            bt[:nvar] = (ggs[:nvar, i, iss] - avs[:nvar, iss]) / sds[
                                :nvar, iss
                            ]

                            if i == 0 and iss == 0:
                                gprev[:nvar] = bt[:nvar]
                                btprev[:nvar] = bt[:nvar]

                            temp = np.matmul(go, gmg)
                            temp1 = np.matmul(temp, bt)
                            temp4 = np.matmul(co, gprev)
                            temp = np.matmul(go, gmg)
                            temp2 = np.matmul(temp, cmg)
                            temp3 = np.matmul(temp2, btprev)

                            gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                            gprev[:nvar] = gcur[:nvar]
                            btprev[:nvar] = bt[:nvar]
                            gsct[:nvar, i, iss] = (
                                gcur[:nvar] * sds[:nvar, iss] + avs[:nvar, iss]
                            )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avs, sds, cors = mbc.avsds(gsct)

                        print("a4 mean gcmc cors (:,:)", cors[:, :])
                        print("a4 mean obs corsh (:,:)", corsh[:, :])

            # ------- end calculate seasonal series of gcm series and store --------

            # ------- form annual series of corrected seasonal gcm series and store ----

            for k in range(nvar):
                for i in range(iyr):
                    sy1 = 0.0
                    for j in range(nss):
                        sy1 += gsct[k, i, j]

                    gy[k, i] = sy1 / nss

            ggy = gy.copy()
            gyct = gy.copy()

            if irho[3, 0] != 0:  # hist: annual mean, future: - avyc + avyh
                if tprint == 1 and itr > 0:
                    print("Correcting for annual mean", jj)

                avyc, sdsy, coryc = mbc.avsdy(gy)

                avyc_iter[itr, :] = avyc.copy()

                # Calculate new values for ggy
                for i in range(iyr):
                    ggy[:, i] = (gy[:, i] - avyc) + avyh

                # Copy ggd values to gdct
                gyct = ggy.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avy, sdy, cory = mbc.avsdy(ggy)

                    print("a4 mean gcmc avy (:)", avy[:])
                    print("a4 mean obs avyh (:)", avyh[:])

            if irho[3, 1] != 0:  # hist: annual sd, future: (x - avy)*sdyh/sdyc + avy
                if tprint == 1:
                    print("Correcting for annual mean and standard deviation", jj)

                avy, sdyc, coryc = mbc.avsdy(ggy)

                sdyc_iter[itr, :] = sdyc.copy()

                # Calculate fact
                non_zero_mask = sdyc > 1e-10
                fact = np.ones_like(sdyc)
                fact[non_zero_mask] = sdyh[non_zero_mask] / sdyc[non_zero_mask]
                fact[(sdyh < 0.1) & (sdyc < 0.1)] = 1.0

                for i in range(iyr):
                    # Adjust ggy
                    ggy[:, i] = (ggy[:, i] - avy) * fact + avy

                # Copy ggd values to gdct
                gyct = ggy.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avy, sdy, cory = mbc.avsdy(ggy)

                    print("a4 mean gcmc sdy (:)", sdy[:])
                    print("a4 mean obs sdyh (:)", sdyh[:])

            if (
                irho[3, 2] != 0 or irho[3, 3] != 0
            ):  # hist: annual corl, future: cmody, gmody, cobsy, gobsy
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit annual corrl itr, jj", itr, jj)
                    pass

                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for annual corrl", jj)

                    avy, sdy, cory = mbc.avsdy(ggy)

                    # Conditional block for additional function call
                    cmody, gmody = mbc.c_g_corl_year(ggy, 2)

                    cmody_iter[itr, :, :] = cmody.copy()
                    gmody_iter[itr, :, :] = gmody.copy()

                    # Calculate fact
                    non_zero_mask = sdy > 1e-10
                    fact = np.ones_like(sdy)
                    fact[non_zero_mask] = sdy[non_zero_mask]
                    fact[(sdy < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        gmg = gmody[:nvar, :nvar]
                        cmg = cmody[:nvar, :nvar]
                        go = dobsy[:nvar, :nvar]
                        co = cobsy[:nvar, :nvar]

                        bt[:nvar] = (ggy[:nvar, i] - avy[:nvar]) / sdy[:nvar]

                        if i == 0:
                            gprev[:nvar] = bt[:nvar]
                            btprev[:nvar] = bt[:nvar]

                        temp = np.matmul(go, gmg)
                        temp1 = np.matmul(temp, bt)
                        temp4 = np.matmul(co, gprev)
                        temp = np.matmul(go, gmg)
                        temp2 = np.matmul(temp, cmg)
                        temp3 = np.matmul(temp2, btprev)

                        gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                        gprev[:nvar] = gcur[:nvar]
                        btprev[:nvar] = bt[:nvar]
                        gyct[:nvar, i] = gcur[:nvar] * sdy[:nvar] + avy[:nvar]

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avy, sdy, cory = mbc.avsdy(gyct)

                        print("a4 mean gcmc cory (:)", cory[:])
                        print("a4 mean obs coryh (:)", coryh[:])

            # -------------- end calculate the gcm series and store ----------------

            for i in range(iyr):
                for j in range(nout):
                    iss = mbc.iseas(j + 1, isn, ij, isj)
                    if time_scale != 0:
                        jd = 1
                    elif j == 1 and is_leap_year(iyr + startyear_h):
                        jd = 29
                    else:
                        jd = days_in_month[j]

                    for day_index in range(jd):
                        for k in range(nvar):
                            aa = gcmc[k, i, j, day_index]

                            if aa > missing_value:
                                af1 = 1.0
                                af2 = 1.0
                                af3 = 1.0
                                af4 = 1.0
                                # af5 = 1.0

                                if time_scale == 0:
                                    if gd[k, i, j, day_index] != 0.0:
                                        af1 = (
                                            gdct[k, i, j, day_index]
                                            / gd[k, i, j, day_index]
                                        )
                                if gm[k, i, j] != 0.0:
                                    af2 = gmct[k, i, j] / gm[k, i, j]
                                if gs[k, i, iss - 1] != 0.0:
                                    af3 = gsct[k, i, iss - 1] / gs[k, i, iss - 1]
                                if gy[k, i] != 0.0:
                                    af4 = gyct[k, i] / gy[k, i]

                                ac = af1 * af2 * af3 * af4 * 1

                                if ilimit[k] > 0:
                                    if aa < thres[k] and aa >= 0.0:
                                        ac = 1.0
                                    if aa >= thres[k] and (aa * ac) < thres[k]:
                                        ac = 1.001 * (thres[k] / aa)

                                ac1 = aa * ac

                                if ac1 > phul[jj, k]:
                                    # print(
                                    #     "Constrained value exceeds upper limit",
                                    #     jj,
                                    #     k,
                                    # )
                                    ac1 = phul[jj, k]
                                if ac1 < phll[jj, k]:
                                    ac1 = phll[jj, k]
                                    # print(
                                    #     "Constrained value exceeds lower limit",
                                    #     jj,
                                    #     k,
                                    # )

                                gcmc[k, i, j, day_index] = ac1

                            else:
                                gcmc[k, i, j, day_index] = missing_value

    # print("Iteration", itr, "completed")
    # Create an instance of BiasCorrectionParams with the computed values
    bc_params = BiasCorrectionParams(
        avdc_iter=avdc_iter,
        sddc_iter=sddc_iter,
        avmc_iter=avmc_iter,
        sdmc_iter=sdmc_iter,
        avsc_iter=avsc_iter,
        sdsc_iter=sdsc_iter,
        avyc_iter=avyc_iter,
        sdyc_iter=sdyc_iter,
        cmod_iter=cmod_iter,
        gmod_iter=gmod_iter,
        cmodm_iter=cmodm_iter,
        gmodm_iter=gmodm_iter,
        cmods_iter=cmods_iter,
        gmods_iter=gmods_iter,
        cmody_iter=cmody_iter,
        gmody_iter=gmody_iter,
        avdh=avdh,
        sddh=sddh,
        avmh=avmh,
        sdmh=sdmh,
        avsh=avsh,
        sdsh=sdsh,
        avyh=avyh,
        sdyh=sdyh,
        cobs=cobs,
        dobs=dobs,
        cobsm=cobsm,
        dobsm=dobsm,
        cobss=cobss,
        dobss=dobss,
        cobsy=cobsy,
        dobsy=dobsy,
        phll=phll,
        phul=phul,
        missing_value=missing_value,
    )

    return {"gcmc": gcmc, "bc_params": bc_params}


def bc_correction_future(gcm_reshape, bc_params, startyear_f, endyear_f):
    """
    Perform bias correction for future climate data.

    Args:
        gcm_reshape (ndarray): Reshaped GCM data.
        bc_params (object): Object containing bias correction parameters.
        startyear_f (int): Start year of the future climate data.
        endyear_f (int): End year of the future climate data.

    Returns:
        None

    Workflow:
        1. Set variables using the `set_variables` function.
        2. Initialize some variables and arrays.
        3. Loop over iterations.
            a. If it's the first iteration, perform boundary limits and correction.
            b. If it's not the first iteration, perform correction based on daily mean and standard deviation.
        4. Perform bias correction for daily mean and standard deviation.
    """

    # ==========================================================================#
    nout, leap, irho, isn, ij, isj, phlwr, phupr, ilimit, thres, idays = set_variables()
    # ==========================================================================#
    # Exclude specific humidity if the first element of the GCM data is less than 0.1
    # to avoid unexpected values during the bias correction process
    if bc_boundary == "lateral":
        if correction_model == 4 and np.max(gcm_reshape[2, :, :, :]) < 0.1:
            print("Exclude specific humidity below the lower threshold (1*10-4).")
            if time_scale == 0:  # daily: 0 or monthly: 1
                itmp = 0
            else:
                itmp = 1

            irho = np.zeros((5, 5), dtype=int)
            for i in range(
                itmp, 5, 1
            ):  # set irho matrix (bias correction options: 1-included and 0-excluded)
                irho[i, :] = [1, 1, 1, 0, 0]

    nsc = startyear_f - 1
    nday = mbc.day()

    # ==========================================================================#
    # inx = 2
    # ==========================================================================#
    #
    # Loop structures
    nntr = no_of_iterations + 1
    nxt = 0
    for i in range(5):  # Loops from 0 to 4 inclusive
        for j in range(3, 5):  # Loops from 3 to 4 inclusive
            nxt += irho[i, j]

    if nxt > 0:
        nntr = no_of_iterations + 2

    ngcur = endyear_f - startyear_f + 1
    nsgc = startyear_f - 1
    nvar = no_of_variables
    # if gcm_reshape[2, 0, 0, 0] <= 0.1:
    #     nvar = no_of_variables - 1
    nsmax = 4
    nss = 4
    tprint = 0
    iss = 0

    if tprint == 1:
        print("Start loop, first: boundary limits, next: correction")

    # Initialize input arrays
    gcmc = gcm_reshape.copy()
    jj = 1
    gs = np.zeros((nvar, ngcur, 12))
    gm = np.zeros((nvar, ngcur, 12))
    gy = np.zeros((nvar, ngcur))

    # Account for days in February in leap years
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for itr in range(nntr):
        iyr = ngcur
        ns1 = nsgc
        gcm = gcmc.copy()

        if tprint == 1 and itr > 1:
            print(" Applying bias correction for future climate ", itr)

        gd = gcm.copy()
        ggd = gd.copy()
        gdct = gd.copy()

        if itr == 0:
            # -------------------------------------------------------------------------
            # Start loop, first: boundary limits, next: correction
            # -------------------------------------------------------------------------
            if time_scale != 0:
                for k in range(nvar):
                    for i in range(iyr):
                        for j in range(nout):
                            if j == 1 and is_leap_year(iyr + startyear_h):
                                days = 29
                            else:
                                days = days_in_month[j]

                            gm[k, i, j] = np.sum(gdct[k, i, j, :days]) / days

                ggm = gm.copy()
                gmct = gm.copy()

                ggm = (gm - bc_params.avmc_iter[1, :, np.newaxis, :]) * bc_params.sdmh[
                    :, np.newaxis, :
                ] / bc_params.sdmc_iter[1, :, np.newaxis, :] + bc_params.avmh[
                    :, np.newaxis, :
                ]
                gmct = ggm.copy()

                # 4. Fourth Segment
                potential_phul = gmct + 3 * bc_params.sdmh[:, np.newaxis, :]
                potential_phll = gmct - 3 * bc_params.sdmh[:, np.newaxis, :]

                bc_params.phul[jj, :] = np.maximum(
                    potential_phul.max(axis=(1, 2)),
                    bc_params.phul[jj, :],
                )
                bc_params.phll[jj, :] = np.minimum(
                    potential_phll.min(axis=(1, 2)),
                    bc_params.phll[jj, :],
                )

            else:
                # print("Shape of bc_params.avdc_iter:", bc_params.avdc_iter.shape)

                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]
                        # Adjust ggd
                        ggd[:, i, j, :jd] = (
                            gd[:, i, j, :jd] - bc_params.avdc_iter[1, :, j, :jd]
                        ) * bc_params.sddh[:, j, :jd] / bc_params.sddc_iter[
                            1, :, j, :jd
                        ] + bc_params.avdh[
                            :, j, :jd
                        ]

                # Copy the computed values to gdct
                gdct = ggd.copy()

                non_zero_mask = (gd > missing_value) & (
                    gd != 0.0
                )  # Adjust the threshold as needed
                ac = np.ones_like(gd)
                ac[non_zero_mask] = gdct[non_zero_mask] / gd[non_zero_mask]

                # Calculate the potential new maximum values for bc_params.phul
                potential_phul = gcm * ac + 3 * bc_params.sddh[:, np.newaxis, :, :]

                # Update the bc_params.phul values if the condition is met
                bc_params.phul[jj, :] = np.where(
                    potential_phul.max(axis=(1, 2, 3)) > bc_params.phul[jj, :],
                    potential_phul.max(axis=(1, 2, 3)),
                    bc_params.phul[jj, :],
                )

                # Calculate the potential new minimum values for bc_params.phll
                potential_phll = gcm * ac - 3 * bc_params.sddh[:, np.newaxis, :, :]

                # Update the bc_params.phll values if the condition is met
                bc_params.phll[jj, :] = np.where(
                    potential_phll.min(axis=(1, 2, 3)) < bc_params.phll[jj, :],
                    potential_phll.min(axis=(1, 2, 3)),
                    bc_params.phll[jj, :],
                )

                # Returning the first few values of 'bc_params.phul' and 'bc_params.phll' for verification
                # bc_params.phul[0, :5], bc_params.phll[0, :5]

            # for j in range(no_of_variables):
            #     if bc_params.phll[jj, j] < phlwr[j] * 4:
            #         bc_params.phll[jj, j] = phlwr[j] * 4
            #     if bc_params.phul[jj, j] > phupr[j] * 4:
            #         bc_params.phul[jj, j] = phupr[j] * 4
            for j in range(no_of_variables):
                if bc_params.phll[jj, j] < phlwr[j]:
                    bc_params.phll[jj, j] = phlwr[j]
                if bc_params.phul[jj, j] > phupr[j]:
                    bc_params.phul[jj, j] = phupr[j]
            # -------------------------------------------------------------------------
            # End loop, first: boundary limits, next: correction
            # -------------------------------------------------------------------------

        elif itr > 0:
            if irho[0, 0] != 0:  # hist: daily mean, future: - avdc + avdh
                if tprint == 1:
                    print("Correcting for daily mean", jj)

                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_f):
                            jd = 29
                        else:
                            jd = days_in_month[j]

                        # Calculate new values for ggd
                        ggd[:, i, j, :jd] = (
                            gd[:, i, j, :jd] - bc_params.avdc_iter[itr, :, j, :jd]
                        ) + bc_params.avdh[:, j, :jd]

                # Modify ggd based on bc_params.phul and bc_params.phll conditions
                ggd = np.where(
                    ggd > bc_params.phul[jj, :, None, None, None],
                    bc_params.phul[jj, :, None, None, None],
                    ggd,
                )
                ggd = np.where(
                    ggd < bc_params.phll[jj, :, None, None, None],
                    bc_params.phll[jj, :, None, None, None],
                    ggd,
                )

                # Copy ggd values to gdct
                gdct = ggd.copy()
                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )
                    print("a4 mean gcmc avd (:,4,:3)", avd[:, 4, :3])
                    print("a4 mean obs avdh (:,4,:3)", bc_params.avdh[:, 4, :3])

            if irho[0, 1] != 0:  # hist: daily sd, future: (x - avd)*sddh/sddc + avd
                if tprint == 1:
                    print("Correcting for daily mean and standard deviation", jj)

                # future
                avd, sdd, cord = mbc.sdsmooth(
                    ggd, moving_window, nday, nsc, 3, leap, idays, missing_value
                )

                # Calculate fact
                non_zero_mask = bc_params.sddc_iter[itr, :, :, :] > 1e-10
                fact = np.ones_like(bc_params.sddc_iter[itr, :, :, :])
                fact[non_zero_mask] = (
                    bc_params.sddh[non_zero_mask]
                    / bc_params.sddc_iter[itr, :, :, :][non_zero_mask]
                )
                fact[
                    (bc_params.sddh < 0.1) & (bc_params.sddc_iter[itr, :, :, :] < 0.1)
                ] = 1.0

                # correct for bias in daily sd
                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]

                        # Adjust ggd
                        ggd[:, i, j, :jd] = (ggd[:, i, j, :jd] - avd[:, j, :jd]) * fact[
                            :, j, :jd
                        ] + avd[:, j, :jd]

                ggd = np.where(
                    ggd > bc_params.phul[jj, :, None, None, None],
                    bc_params.phul[jj, :, None, None, None],
                    ggd,
                )
                ggd = np.where(
                    ggd < bc_params.phll[jj, :, None, None, None],
                    bc_params.phll[jj, :, None, None, None],
                    ggd,
                )
                # Copy ggd values to gdct
                gdct = ggd.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )
                    print("a4 std gcmc sdd (1,4,:3)", sdd[:, 4, :3])
                    print("a4 std obs sddh (1,4,:3)", bc_params.sddh[:, 4, :3])

            if (
                irho[0, 2] != 0 or irho[0, 3] != 0
            ):  # hist: daily corl, future: cmod, gmod, cobs, gobs
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit daily corrl itr, jj", itr, jj)
                    pass
                else:
                    if tprint == 1 and itr > 1:
                        print("Correcting for daily corrl", jj)
                    # correct for bias in corl
                    # find smoothened mean and sd of gcm mean and sd corrected series
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )

                    # Calculate fact
                    non_zero_mask = sdd > 1e-10
                    fact = np.ones_like(sdd)
                    fact[non_zero_mask] = sdd[non_zero_mask]
                    fact[(sdd < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for j in range(nout):
                            if j == 1 and is_leap_year(iyr + startyear_h):
                                jd = 29
                            else:
                                jd = days_in_month[j]

                            for day_index in range(jd):
                                gmg = bc_params.gmod_iter[
                                    itr, j, day_index, :nvar, :nvar
                                ]
                                cmg = bc_params.cmod_iter[
                                    itr, j, day_index, :nvar, :nvar
                                ]
                                go = bc_params.dobs[j, day_index, :nvar, :nvar]
                                co = bc_params.cobs[j, day_index, :nvar, :nvar]

                                bt[:nvar] = (
                                    ggd[:nvar, i, j, day_index]
                                    - avd[:nvar, j, day_index]
                                ) / sdd[:nvar, j, day_index]

                                if i == 0 and j == 0 and day_index == 0:
                                    gprev[:nvar] = bt[:nvar]
                                    btprev[:nvar] = bt[:nvar]

                                temp = np.matmul(go, gmg)
                                temp1 = np.matmul(temp, bt)
                                temp4 = np.matmul(co, gprev)
                                temp2 = np.matmul(temp, cmg)
                                temp3 = np.matmul(temp2, btprev)

                                gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                                gprev[:nvar] = gcur[:nvar]
                                btprev[:nvar] = bt[:nvar]
                                gdct[:nvar, i, j, day_index] = (
                                    gcur[:nvar] * sdd[:nvar, j, day_index]
                                    + avd[:nvar, j, day_index]
                                )
                                gdct[:nvar, i, j, day_index] = np.clip(
                                    gdct[:nvar, i, j, day_index],
                                    bc_params.phll[jj, :nvar],
                                    bc_params.phul[jj, :nvar],
                                )

                    # Copy ggd values to gdct
                    # gcmc = gdct.copy()

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avd, sdd, cord = mbc.sdsmooth(
                            gdct,
                            moving_window,
                            nday,
                            ns1,
                            jj + 1,
                            leap,
                            idays,
                            missing_value,
                        )

                        print("a4 std gcmc cord (1,4,:3)", cord[:, 4, :3])
                        # print("a4 std obs cordh (1,4,:3)", cordh[2, 4, :3])

            # ------- end calculate daily series of gcm series and store -------

            # ------- calculate monthly series of gcm series and store --------

            for k in range(nvar):
                for i in range(iyr):
                    for j in range(nout):
                        if time_scale != 0:
                            jd = 1
                        elif j == 1 and is_leap_year(iyr + startyear_h):
                            days = 29
                        else:
                            days = days_in_month[j]

                        gm[k, i, j] = np.sum(gdct[k, i, j, :days]) / days

            ggm = gm.copy()
            gmct = gm.copy()

            if irho[1, 0] != 0:  # hist: monthly mean, future: - avmc + avmh
                if tprint == 1 and itr > 0:
                    print("Correcting for monthly mean", jj)

                # Calculate new values for ggm
                for i in range(iyr):
                    ggm[:, i, :] = (
                        gm[:, i, :] - bc_params.avmc_iter[itr, :, :]
                    ) + bc_params.avmh

                # Modify ggd based on bc_params.phul and bc_params.phll conditions
                ggm = np.where(
                    ggm > bc_params.phul[jj, :, None, None],
                    bc_params.phul[jj, :, None, None],
                    ggm,
                )
                ggm = np.where(
                    ggm < bc_params.phll[jj, :, None, None],
                    bc_params.phll[jj, :, None, None],
                    ggm,
                )

                # Copy ggd values to gdct
                gmct = ggm.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avm, sdm, corm = mbc.avsds(ggm)

                    print("a4 mean gcmc avm (:,:)", avm[:, :])
                    print("a4 mean obs avmh (:,:)", bc_params.avmh[:, :])

            if irho[1, 1] != 0:  # hist: monthly sd, future: (x - avm)*sdmh/sdmc + avm
                if tprint == 1:
                    print("Correcting for monthly mean and standard deviation", jj)

                # future
                avm, sdm, corm = mbc.avsds(ggm)

                # Calculate fact
                non_zero_mask = bc_params.sdmc_iter[itr, :, :] > 1e-10
                fact = np.ones_like(bc_params.sdmc_iter[itr, :, :])
                fact[non_zero_mask] = (
                    bc_params.sdmh[non_zero_mask]
                    / bc_params.sdmc_iter[itr, :, :][non_zero_mask]
                )
                fact[
                    (bc_params.sdmh < 0.1) & (bc_params.sdmc_iter[itr, :, :] < 0.1)
                ] = 1.0

                # correct for bias in monthly sd
                for i in range(iyr):
                    # Adjust ggm
                    ggm[:, i, :] = (ggm[:, i, :] - avm) * fact + avm

                ggm = np.where(
                    ggm > bc_params.phul[jj, :, None, None],
                    bc_params.phul[jj, :, None, None],
                    ggm,
                )
                ggm = np.where(
                    ggm < bc_params.phll[jj, :, None, None],
                    bc_params.phll[jj, :, None, None],
                    ggm,
                )

                # Copy ggd values to gdct
                gmct = ggm.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avm, sdm, corm = mbc.avsds(ggm)

                    print("a4 mean gcmc sdm (:,:)", sdm[:, :])
                    print("a4 mean obs sdmh (:,:)", bc_params.sdmh[:, :])

            if (
                irho[1, 2] != 0 or irho[1, 3] != 0
            ):  # hist: monthly corl, future: cmodm, gmodm, cobsm, gobsm
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("exit monthly corrl itr, jj", itr, jj)
                    pass

                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for monthly corrl", jj)

                    avm, sdm, corm = mbc.avsds(ggm)

                    # Calculate fact
                    non_zero_mask = sdm > 1e-10
                    fact = np.ones_like(sdm)
                    fact[non_zero_mask] = sdm[non_zero_mask]
                    fact[(sdm < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for j in range(nout):
                            gmg = bc_params.gmodm_iter[itr, j, :nvar, :nvar]
                            cmg = bc_params.cmodm_iter[itr, j, :nvar, :nvar]
                            go = bc_params.dobsm[j, :nvar, :nvar]
                            co = bc_params.cobsm[j, :nvar, :nvar]

                            bt[:nvar] = (ggm[:nvar, i, j] - avm[:nvar, j]) / sdm[
                                :nvar, j
                            ]

                            if i == 0 and j == 0:
                                gprev[:nvar] = bt[:nvar]
                                btprev[:nvar] = bt[:nvar]

                            temp = np.matmul(go, gmg)
                            temp1 = np.matmul(temp, bt)
                            temp4 = np.matmul(co, gprev)
                            temp = np.matmul(go, gmg)
                            temp2 = np.matmul(temp, cmg)
                            temp3 = np.matmul(temp2, btprev)

                            gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                            gprev[:nvar] = gcur[:nvar]
                            btprev[:nvar] = bt[:nvar]
                            gmct[:nvar, i, j] = (
                                gcur[:nvar] * sdm[:nvar, j] + avm[:nvar, j]
                            )
                            gmct[:nvar, i, j] = np.clip(
                                gmct[:nvar, i, j],
                                bc_params.phll[jj, :nvar],
                                bc_params.phul[jj, :nvar],
                            )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avm, sdm, corm = mbc.avsds(gmct)

                        print("a4 mean gcmc corm (:,:4)", corm[:, :4])
                    # print("a4 mean obs cormh (2,:4)", bc_params.cormh[2, :4])

            # ------- end calculate monthly series of gcm series and store --------

            # ------- form seasonal series of corrected monthly gcm series and store --

            for k in range(nvar):
                for i in range(iyr):
                    ss1 = np.zeros(nsmax)
                    for j in range(nout):
                        iss = mbc.iseas(j + 1, nss, ij, isj)
                        ss1[iss - 1] += gmct[k, i, j]
                    for iss in range(nss):
                        gs[k, i, iss] = ss1[iss] / ij[iss]

            ggs = gs.copy()
            gsct = gs.copy()

            if irho[2, 0] != 0:  # hist: seasonal mean, future: - avsc + avsh
                if tprint == 1 and itr > 0:
                    print("Correcting for seasonal mean", jj)

                # Calculate new values for ggm
                for i in range(iyr):
                    ggs[:, i, :] = (
                        gs[:, i, :] - bc_params.avsc_iter[itr, :, :]
                    ) + bc_params.avsh

                # Copy ggd values to gdct
                gsct = ggs.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avs, sds, cors = mbc.avsds(ggs)

                    print("a4 mean gcmc avs (:,:4)", avs[:, :4])
                    print("a4 mean obs avsh (:,:4)", bc_params.avsh[:, :4])

            if irho[2, 1] != 0:  # hist: seasonal sd, future: (x - avs)*sdsh/sdsc + avs
                if tprint == 1:
                    print("Correcting for seasonal mean and standard deviation", jj)

                avs, sds, cors = mbc.avsds(ggs)

                # Calculate fact
                non_zero_mask = bc_params.sdsc_iter[itr, :, :] > 1e-10
                fact = np.ones_like(bc_params.sdsc_iter[itr, :, :])
                fact[non_zero_mask] = (
                    bc_params.sdsh[non_zero_mask]
                    / bc_params.sdsc_iter[itr, :, :][non_zero_mask]
                )
                fact[
                    (bc_params.sdsh < 0.1) & (bc_params.sdsc_iter[itr, :, :] < 0.1)
                ] = 1.0

                for i in range(iyr):
                    # Adjust ggm
                    ggs[:, i, :] = (ggs[:, i, :] - avs) * fact + avs

                # Copy ggd values to gdct
                gsct = ggs.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avs, sds, cors = mbc.avsds(ggs)

                    print("a4 mean gcmc sds (:,:4)", sds[:, :4])
                    print("a4 mean obs sdsh (:,:4)", bc_params.sdsh[:, :4])

            if (
                irho[2, 2] != 0 or irho[2, 3] != 0
            ):  # hist: seasonal corl, future: cmods, gmods, cobss, gobss
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit seasonal corrl itr, jj", itr, jj)
                    pass
                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for seasonal corrl", jj)

                    avs, sds, cors = mbc.avsds(ggs)

                    # Calculate fact
                    non_zero_mask = sds > 1e-10
                    fact = np.ones_like(sds)
                    fact[non_zero_mask] = sds[non_zero_mask]
                    fact[(sds < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for iss in range(nss):
                            gmg = bc_params.gmods_iter[itr, iss, :nvar, :nvar]
                            cmg = bc_params.cmods_iter[itr, iss, :nvar, :nvar]
                            go = bc_params.dobss[iss, :nvar, :nvar]
                            co = bc_params.cobss[iss, :nvar, :nvar]

                            bt[:nvar] = (ggs[:nvar, i, iss] - avs[:nvar, iss]) / sds[
                                :nvar, iss
                            ]

                            if i == 0 and iss == 0:
                                gprev[:nvar] = bt[:nvar]
                                btprev[:nvar] = bt[:nvar]

                            temp = np.matmul(go, gmg)
                            temp1 = np.matmul(temp, bt)
                            temp4 = np.matmul(co, gprev)
                            temp = np.matmul(go, gmg)
                            temp2 = np.matmul(temp, cmg)
                            temp3 = np.matmul(temp2, btprev)

                            gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                            gprev[:nvar] = gcur[:nvar]
                            btprev[:nvar] = bt[:nvar]
                            gsct[:nvar, i, iss] = (
                                gcur[:nvar] * sds[:nvar, iss] + avs[:nvar, iss]
                            )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avs, sds, cors = mbc.avsds(gsct)

                        print("a4 mean gcmc cors (:4,:4)", cors[:4, :4])
                    # print("a4 mean obs corsh (2,:)", bc_params.corsh[2, :])

            # ------- end calculate seasonal series of gcm series and store --------

            # ------- form annual series of corrected seasonal gcm series and store ----

            for k in range(nvar):
                for i in range(iyr):
                    sy1 = 0.0
                    for j in range(nss):
                        sy1 += gsct[k, i, j]

                    gy[k, i] = sy1 / iss

            ggy = gy.copy()
            gyct = gy.copy()

            if irho[3, 0] != 0:  # hist: annual mean, future: - avyc + avyh
                if tprint == 1 and itr > 0:
                    print("Correcting for annual mean", jj)

                # Calculate new values for ggy
                for i in range(iyr):
                    ggy[:, i] = (
                        gy[:, i] - bc_params.avyc_iter[itr, :]
                    ) + bc_params.avyh

                # Copy ggd values to gdct
                gyct = ggy.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avy, sdy, cory = mbc.avsdy(ggy)

                    print("a4 mean gcmc avy (:)", avy[:])
                    print("a4 mean obs avyh (:)", bc_params.avyh[:])

            if irho[3, 1] != 0:  # hist: annual sd, future: (x - avy)*sdyh/sdyc + avy
                if tprint == 1:
                    print("Correcting for annual mean and standard deviation", jj)

                avy, sdy, cory = mbc.avsdy(ggy)

                # Calculate fact
                non_zero_mask = bc_params.sdyc_iter[itr, :] > 1e-10
                fact = np.ones_like(bc_params.sdyc_iter[itr, :])
                fact[non_zero_mask] = (
                    bc_params.sdyh[non_zero_mask]
                    / bc_params.sdyc_iter[itr, :][non_zero_mask]
                )
                fact[(bc_params.sdyh < 0.1) & (bc_params.sdyc_iter[itr, :] < 0.1)] = 1.0

                for i in range(iyr):
                    # Adjust ggy
                    ggy[:, i] = (ggy[:, i] - avy) * fact + avy

                # Copy ggd values to gdct
                gyct = ggy.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avy, sdy, cory = mbc.avsdy(ggy)

                    print("a4 mean gcmc sdy (:)", sdy[:])
                    print("a4 mean obs sdyh (:)", bc_params.sdyh[:])

            if (
                irho[3, 2] != 0 or irho[3, 3] != 0
            ):  # hist: annual corl, future: cmody, gmody, cobsy, gobsy
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit annual corrl itr, jj", itr, jj)
                    pass

                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for annual corrl", jj)

                    avy, sdy, cory = mbc.avsdy(ggy)

                    # Calculate fact
                    non_zero_mask = sdy > 1e-10
                    fact = np.ones_like(sdy)
                    fact[non_zero_mask] = sdy[non_zero_mask]
                    fact[(sdy < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        gmg = bc_params.gmody_iter[itr, :nvar, :nvar]
                        cmg = bc_params.cmody_iter[itr, :nvar, :nvar]
                        go = bc_params.dobsy[:nvar, :nvar]
                        co = bc_params.cobsy[:nvar, :nvar]

                        bt[:nvar] = (ggy[:nvar, i] - avy[:nvar]) / sdy[:nvar]

                        if i == 0:
                            gprev[:nvar] = bt[:nvar]
                            btprev[:nvar] = bt[:nvar]

                        temp = np.matmul(go, gmg)
                        temp1 = np.matmul(temp, bt)
                        temp4 = np.matmul(co, gprev)
                        temp = np.matmul(go, gmg)
                        temp2 = np.matmul(temp, cmg)
                        temp3 = np.matmul(temp2, btprev)

                        gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                        gprev[:nvar] = gcur[:nvar]
                        btprev[:nvar] = bt[:nvar]
                        gyct[:nvar, i] = gcur[:nvar] * sdy[:nvar] + avy[:nvar]

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avy, sdy, cory = mbc.avsdy(gyct)

                        print("a4 mean gcmc cory (:)", cory[:])
                        # print("a4 mean obs coryh (:)", coryh[:])

            # -------------- end calculate the gcm series and store ----------------

            for i in range(iyr):
                for j in range(nout):
                    iss = mbc.iseas(j + 1, isn, ij, isj)
                    if time_scale != 0:
                        jd = 1
                    elif j == 1 and is_leap_year(iyr + startyear_h):
                        jd = 29
                    else:
                        jd = days_in_month[j]

                    for day_index in range(jd):
                        for k in range(nvar):
                            aa = gcmc[k, i, j, day_index]

                            if aa > missing_value:
                                af1 = 1.0
                                af2 = 1.0
                                af3 = 1.0
                                af4 = 1.0
                                # af5 = 1.0

                                if time_scale == 0:
                                    if gd[k, i, j, day_index] != 0.0:
                                        af1 = (
                                            gdct[k, i, j, day_index]
                                            / gd[k, i, j, day_index]
                                        )
                                if gm[k, i, j] != 0.0:
                                    af2 = gmct[k, i, j] / gm[k, i, j]
                                if gs[k, i, iss - 1] != 0.0:
                                    af3 = gsct[k, i, iss - 1] / gs[k, i, iss - 1]
                                if gy[k, i] != 0.0:
                                    af4 = gyct[k, i] / gy[k, i]

                                ac = af1 * af2 * af3 * af4 * 1

                                if ilimit[k] > 0:
                                    if aa < thres[k] and aa >= 0.0:
                                        ac = 1.0
                                    if aa >= thres[k] and (aa * ac) < thres[k]:
                                        ac = 1.001 * (thres[k] / aa)

                                ac1 = aa * ac

                                if ac1 > bc_params.phul[jj, k]:
                                    # print(
                                    #     "Constrained value exceeds upper limit",
                                    #     jj,
                                    #     k,
                                    #     ac1,
                                    #     bc_params.phul[jj, k],
                                    # )
                                    ac1 = bc_params.phul[jj, k]
                                if ac1 < bc_params.phll[jj, k]:
                                    # print(
                                    #     "Constrained value exceeds lower limit",
                                    #     jj,
                                    #     k,
                                    #     ac1,
                                    #     bc_params.phll[jj, k],
                                    # )
                                    ac1 = bc_params.phll[jj, k]

                                gcmc[k, i, j, day_index] = ac1

                            else:
                                gcmc[k, i, j, day_index] = missing_value

    print("Iteration", itr, "completed")

    return gcmc


def bc_correction_future_old(gcm_reshape, bc_params, startyear_f, endyear_f):
    # ==========================================================================#
    nout, leap, irho, isn, ij, isj, phlwr, phupr, ilimit, thres, idays = set_variables()
    # ==========================================================================#

    nsc = startyear_f - 1
    nday = mbc.day()

    # ==========================================================================#
    # inx = 2
    # ==========================================================================#
    #
    # Loop structures
    nntr = no_of_iterations + 1
    nxt = 0
    for i in range(5):  # Loops from 0 to 4 inclusive
        for j in range(3, 5):  # Loops from 3 to 4 inclusive
            nxt += irho[i, j]

    if nxt > 0:
        nntr = no_of_iterations + 2

    ngcur = endyear_f - startyear_f + 1
    nsgc = startyear_f - 1
    nvar = no_of_variables
    nsmax = 4
    nss = 4
    tprint = 0
    iss = 0

    if tprint == 1:
        print("Start loop, first: boundary limits, next: correction")

    # Initialize input arrays
    gcmc = gcm_reshape.copy()
    jj = 1
    gs = np.zeros((nvar, ngcur, 12))
    gm = np.zeros((nvar, ngcur, 12))
    gy = np.zeros((nvar, ngcur))

    # Account for days in February in leap years
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for itr in range(nntr):
        iyr = ngcur
        ns1 = nsgc
        gcm = gcmc.copy()

        if tprint == 1 and itr > 1:
            print(" Applying bias correction for future climate ", itr)

        gd = gcm.copy()
        ggd = gd.copy()
        gdct = gd.copy()

        if itr == 0:
            # -------------------------------------------------------------------------
            # Start loop, first: boundary limits, next: correction
            # -------------------------------------------------------------------------
            if time_scale != 0:
                for k in range(nvar):
                    for i in range(iyr):
                        for j in range(nout):
                            if j == 1 and is_leap_year(iyr + startyear_h):
                                days = 29
                            else:
                                days = days_in_month[j]

                            gm[k, i, j] = np.sum(gdct[k, i, j, :days]) / days

                ggm = gm.copy()
                gmct = gm.copy()

                ggm = (
                    gm - bc_params.avmc_iter[itr, :, np.newaxis, :]
                ) * bc_params.sdmh[:, np.newaxis, :] / bc_params.sdmc_iter[
                    itr, :, np.newaxis, :
                ] + bc_params.avmh[
                    :, np.newaxis, :
                ]
                gmct = ggm.copy()

                # 4. Fourth Segment
                potential_phul = gmct + 3 * bc_params.sdmh[:, np.newaxis, :]
                potential_phll = gmct - 3 * bc_params.sdmh[:, np.newaxis, :]

                bc_params.phul[jj, :] = np.maximum(
                    potential_phul.max(axis=(1, 2)),
                    bc_params.phul[jj, :],
                )
                bc_params.phll[jj, :] = np.minimum(
                    potential_phll.min(axis=(1, 2)),
                    bc_params.phll[jj, :],
                )

            else:
                # print("Shape of bc_params.avdc_iter:", bc_params.avdc_iter.shape)

                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]
                        # Adjust ggd
                        ggd[:, i, j, :jd] = (
                            gd[:, i, j, :jd] - bc_params.avdc_iter[itr, :, j, :jd]
                        ) * bc_params.sddh[:, j, :jd] / bc_params.sddc_iter[
                            itr, :, j, :jd
                        ] + bc_params.avdc_iter[
                            itr, :, j, :jd
                        ]

                # Copy the computed values to gdct
                gdct = ggd.copy()

                non_zero_mask = (gd > missing_value) & (
                    gd != 0.0
                )  # Adjust the threshold as needed
                ac = np.ones_like(gd)
                ac[non_zero_mask] = gdct[non_zero_mask] / gd[non_zero_mask]

                # Calculate the potential new maximum values for bc_params.phul
                potential_phul = gcm * ac + 3 * bc_params.sddh[:, np.newaxis, :, :]

                # Update the bc_params.phul values if the condition is met
                bc_params.phul[jj, :] = np.where(
                    potential_phul.max(axis=(1, 2, 3)) > bc_params.phul[jj, :],
                    potential_phul.max(axis=(1, 2, 3)),
                    bc_params.phul[jj, :],
                )

                # Calculate the potential new minimum values for bc_params.phll
                potential_phll = gcm * ac - 3 * bc_params.sddh[:, np.newaxis, :, :]

                # Update the bc_params.phll values if the condition is met
                bc_params.phll[jj, :] = np.where(
                    potential_phll.min(axis=(1, 2, 3)) < bc_params.phll[jj, :],
                    potential_phll.min(axis=(1, 2, 3)),
                    bc_params.phll[jj, :],
                )

                # Returning the first few values of 'bc_params.phul' and 'bc_params.phll' for verification
                # bc_params.phul[0, :5], bc_params.phll[0, :5]

            for j in range(no_of_variables):
                if bc_params.phll[jj, j] < phlwr[j] * 4:
                    bc_params.phll[jj, j] = phlwr[j] * 4
                if bc_params.phul[jj, j] > phupr[j] * 4:
                    bc_params.phul[jj, j] = phupr[j] * 4

            # -------------------------------------------------------------------------
            # End loop, first: boundary limits, next: correction
            # -------------------------------------------------------------------------

        elif itr > 0:
            if irho[0, 0] != 0:  # hist: daily mean, future: - avdc + avdh
                if tprint == 1:
                    print("Correcting for daily mean", jj)

                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_f):
                            jd = 29
                        else:
                            jd = days_in_month[j]

                        # Calculate new values for ggd
                        ggd[:, i, j, :jd] = (
                            gd[:, i, j, :jd] - bc_params.avdc_iter[itr, :, j, :jd]
                        ) + bc_params.avdh[:, j, :jd]

                # Modify ggd based on bc_params.phul and bc_params.phll conditions
                ggd = np.where(
                    ggd > bc_params.phul[jj, :, None, None, None],
                    bc_params.phul[jj, :, None, None, None],
                    ggd,
                )
                ggd = np.where(
                    ggd < bc_params.phll[jj, :, None, None, None],
                    bc_params.phll[jj, :, None, None, None],
                    ggd,
                )

                # Copy ggd values to gdct
                gdct = ggd.copy()
                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )
                    print("a4 mean gcmc avd (:,4,:3)", avd[:, 4, :3])
                    print("a4 mean obs avdh (:,4,:3)", bc_params.avdh[:, 4, :3])

            if irho[0, 1] != 0:  # hist: daily sd, future: (x - avd)*sddh/sddc + avd
                if tprint == 1:
                    print("Correcting for daily mean and standard deviation", jj)

                # future
                avd, sdd, cord = mbc.sdsmooth(
                    ggd, moving_window, nday, nsc, 3, leap, idays, missing_value
                )

                # Calculate fact
                non_zero_mask = bc_params.sddc_iter[itr, :, :, :] > 1e-10
                fact = np.ones_like(bc_params.sddc_iter[itr, :, :, :])
                fact[non_zero_mask] = (
                    bc_params.sddh[non_zero_mask]
                    / bc_params.sddc_iter[itr, :, :, :][non_zero_mask]
                )
                fact[
                    (bc_params.sddh < 0.1) & (bc_params.sddc_iter[itr, :, :, :] < 0.1)
                ] = 1.0

                # correct for bias in daily sd
                for i in range(iyr):
                    for j in range(nout):
                        if j == 1 and is_leap_year(iyr + startyear_h):
                            jd = 29
                        else:
                            jd = days_in_month[j]

                        # Adjust ggd
                        ggd[:, i, j, :jd] = (ggd[:, i, j, :jd] - avd[:, j, :jd]) * fact[
                            :, j, :jd
                        ] + avd[:, j, :jd]

                ggd = np.where(
                    ggd > bc_params.phul[jj, :, None, None, None],
                    bc_params.phul[jj, :, None, None, None],
                    ggd,
                )
                ggd = np.where(
                    ggd < bc_params.phll[jj, :, None, None, None],
                    bc_params.phll[jj, :, None, None, None],
                    ggd,
                )
                # Copy ggd values to gdct
                gdct = ggd.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )
                    print("a4 std gcmc sdd (1,4,:3)", sdd[:, 4, :3])
                    print("a4 std obs sddh (1,4,:3)", bc_params.sddh[:, 4, :3])

            if (
                irho[0, 2] != 0 or irho[0, 3] != 0
            ):  # hist: daily corl, future: cmod, gmod, cobs, gobs
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit daily corrl itr, jj", itr, jj)
                    pass
                else:
                    if tprint == 1 and itr > 1:
                        print("Correcting for daily corrl", jj)
                    # correct for bias in corl
                    # find smoothened mean and sd of gcm mean and sd corrected series
                    # Translating the function call to sdsmooth
                    avd, sdd, cord = mbc.sdsmooth(
                        ggd,
                        moving_window,
                        nday,
                        ns1,
                        jj + 1,
                        leap,
                        idays,
                        missing_value,
                    )

                    # Calculate fact
                    non_zero_mask = sdd > 1e-10
                    fact = np.ones_like(sdd)
                    fact[non_zero_mask] = sdd[non_zero_mask]
                    fact[(sdd < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for j in range(nout):
                            if j == 1 and is_leap_year(iyr + startyear_h):
                                jd = 29
                            else:
                                jd = days_in_month[j]

                            for day_index in range(jd):
                                gmg = bc_params.gmod_iter[
                                    itr, j, day_index, :nvar, :nvar
                                ]
                                cmg = bc_params.cmod_iter[
                                    itr, j, day_index, :nvar, :nvar
                                ]
                                go = bc_params.dobs[j, day_index, :nvar, :nvar]
                                co = bc_params.cobs[j, day_index, :nvar, :nvar]

                                bt[:nvar] = (
                                    ggd[:nvar, i, j, day_index]
                                    - avd[:nvar, j, day_index]
                                ) / sdd[:nvar, j, day_index]

                                if i == 0 and j == 0 and day_index == 0:
                                    gprev[:nvar] = bt[:nvar]
                                    btprev[:nvar] = bt[:nvar]

                                temp = np.matmul(go, gmg)
                                temp1 = np.matmul(temp, bt)
                                temp4 = np.matmul(co, gprev)
                                temp2 = np.matmul(temp, cmg)
                                temp3 = np.matmul(temp2, btprev)

                                gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                                gprev[:nvar] = gcur[:nvar]
                                btprev[:nvar] = bt[:nvar]
                                gdct[:nvar, i, j, day_index] = (
                                    gcur[:nvar] * sdd[:nvar, j, day_index]
                                    + avd[:nvar, j, day_index]
                                )
                                gdct[:nvar, i, j, day_index] = np.clip(
                                    gdct[:nvar, i, j, day_index],
                                    bc_params.phll[jj, :nvar],
                                    bc_params.phul[jj, :nvar],
                                )

                    # Copy ggd values to gdct
                    # gcmc = gdct.copy()

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avd, sdd, cord = mbc.sdsmooth(
                            gdct,
                            moving_window,
                            nday,
                            ns1,
                            jj + 1,
                            leap,
                            idays,
                            missing_value,
                        )

                        print("a4 std gcmc cord (1,4,:3)", cord[:, 4, :3])
                        # print("a4 std obs cordh (1,4,:3)", cordh[2, 4, :3])

            # ------- end calculate daily series of gcm series and store -------

            # ------- calculate monthly series of gcm series and store --------

            for k in range(nvar):
                for i in range(iyr):
                    for j in range(nout):
                        if time_scale != 0:
                            jd = 1
                        elif j == 1 and is_leap_year(iyr + startyear_h):
                            days = 29
                        else:
                            days = days_in_month[j]

                        gm[k, i, j] = np.sum(gdct[k, i, j, :days]) / days

            ggm = gm.copy()
            gmct = gm.copy()

            if irho[1, 0] != 0:  # hist: monthly mean, future: - avmc + avmh
                if tprint == 1 and itr > 0:
                    print("Correcting for monthly mean", jj)

                # Calculate new values for ggm
                for i in range(iyr):
                    ggm[:, i, :] = (
                        gm[:, i, :] - bc_params.avmc_iter[itr, :, :]
                    ) + bc_params.avmh

                # Modify ggd based on bc_params.phul and bc_params.phll conditions
                ggm = np.where(
                    ggm > bc_params.phul[jj, :, None, None],
                    bc_params.phul[jj, :, None, None],
                    ggm,
                )
                ggm = np.where(
                    ggm < bc_params.phll[jj, :, None, None],
                    bc_params.phll[jj, :, None, None],
                    ggm,
                )

                # Copy ggd values to gdct
                gmct = ggm.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avm, sdm, corm = mbc.avsds(ggm)

                    print("a4 mean gcmc avm (:,:)", avm[:, :])
                    print("a4 mean obs avmh (:,:)", bc_params.avmh[:, :])

            if irho[1, 1] != 0:  # hist: monthly sd, future: (x - avm)*sdmh/sdmc + avm
                if tprint == 1:
                    print("Correcting for monthly mean and standard deviation", jj)

                # future
                avm, sdm, corm = mbc.avsds(ggm)

                # Calculate fact
                non_zero_mask = bc_params.sdmc_iter[itr, :, :] > 1e-10
                fact = np.ones_like(bc_params.sdmc_iter[itr, :, :])
                fact[non_zero_mask] = (
                    bc_params.sdmh[non_zero_mask]
                    / bc_params.sdmc_iter[itr, :, :][non_zero_mask]
                )
                fact[
                    (bc_params.sdmh < 0.1) & (bc_params.sdmc_iter[itr, :, :] < 0.1)
                ] = 1.0

                # correct for bias in monthly sd
                for i in range(iyr):
                    # Adjust ggm
                    ggm[:, i, :] = (ggm[:, i, :] - avm) * fact + avm

                ggm = np.where(
                    ggm > bc_params.phul[jj, :, None, None],
                    bc_params.phul[jj, :, None, None],
                    ggm,
                )
                ggm = np.where(
                    ggm < bc_params.phll[jj, :, None, None],
                    bc_params.phll[jj, :, None, None],
                    ggm,
                )

                # Copy ggd values to gdct
                gmct = ggm.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avm, sdm, corm = mbc.avsds(ggm)

                    print("a4 mean gcmc sdm (:,:)", sdm[:, :])
                    print("a4 mean obs sdmh (:,:)", bc_params.sdmh[:, :])

            if (
                irho[1, 2] != 0 or irho[1, 3] != 0
            ):  # hist: monthly corl, future: cmodm, gmodm, cobsm, gobsm
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("exit monthly corrl itr, jj", itr, jj)
                    pass

                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for monthly corrl", jj)

                    avm, sdm, corm = mbc.avsds(ggm)

                    # Calculate fact
                    non_zero_mask = sdm > 1e-10
                    fact = np.ones_like(sdm)
                    fact[non_zero_mask] = sdm[non_zero_mask]
                    fact[(sdm < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for j in range(nout):
                            gmg = bc_params.gmodm_iter[itr, j, :nvar, :nvar]
                            cmg = bc_params.cmodm_iter[itr, j, :nvar, :nvar]
                            go = bc_params.dobsm[j, :nvar, :nvar]
                            co = bc_params.cobsm[j, :nvar, :nvar]

                            bt[:nvar] = (ggm[:nvar, i, j] - avm[:nvar, j]) / sdm[
                                :nvar, j
                            ]

                            if i == 0 and j == 0:
                                gprev[:nvar] = bt[:nvar]
                                btprev[:nvar] = bt[:nvar]

                            temp = np.matmul(go, gmg)
                            temp1 = np.matmul(temp, bt)
                            temp4 = np.matmul(co, gprev)
                            temp = np.matmul(go, gmg)
                            temp2 = np.matmul(temp, cmg)
                            temp3 = np.matmul(temp2, btprev)

                            gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                            gprev[:nvar] = gcur[:nvar]
                            btprev[:nvar] = bt[:nvar]
                            gmct[:nvar, i, j] = (
                                gcur[:nvar] * sdm[:nvar, j] + avm[:nvar, j]
                            )
                            gmct[:nvar, i, j] = np.clip(
                                gmct[:nvar, i, j],
                                bc_params.phll[jj, :nvar],
                                bc_params.phul[jj, :nvar],
                            )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avm, sdm, corm = mbc.avsds(gmct)

                        print("a4 mean gcmc corm (:,:4)", corm[:, :4])
                    # print("a4 mean obs cormh (2,:4)", bc_params.cormh[2, :4])

            # ------- end calculate monthly series of gcm series and store --------

            # ------- form seasonal series of corrected monthly gcm series and store --

            for k in range(nvar):
                for i in range(iyr):
                    ss1 = np.zeros(nsmax)
                    for j in range(nout):
                        iss = mbc.iseas(j + 1, nss, ij, isj)
                        ss1[iss - 1] += gmct[k, i, j]
                    for iss in range(nss):
                        gs[k, i, iss] = ss1[iss] / ij[iss]

            ggs = gs.copy()
            gsct = gs.copy()

            if irho[2, 0] != 0:  # hist: seasonal mean, future: - avsc + avsh
                if tprint == 1 and itr > 0:
                    print("Correcting for seasonal mean", jj)

                # Calculate new values for ggm
                for i in range(iyr):
                    ggs[:, i, :] = (
                        gs[:, i, :] - bc_params.avsc_iter[itr, :, :]
                    ) + bc_params.avsh

                # Copy ggd values to gdct
                gsct = ggs.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avs, sds, cors = mbc.avsds(ggs)

                    print("a4 mean gcmc avs (:,:4)", avs[:, :4])
                    print("a4 mean obs avsh (:,:4)", bc_params.avsh[:, :4])

            if irho[2, 1] != 0:  # hist: seasonal sd, future: (x - avs)*sdsh/sdsc + avs
                if tprint == 1:
                    print("Correcting for seasonal mean and standard deviation", jj)

                avs, sds, cors = mbc.avsds(ggs)

                # Calculate fact
                non_zero_mask = bc_params.sdsc_iter[itr, :, :] > 1e-10
                fact = np.ones_like(bc_params.sdsc_iter[itr, :, :])
                fact[non_zero_mask] = (
                    bc_params.sdsh[non_zero_mask]
                    / bc_params.sdsc_iter[itr, :, :][non_zero_mask]
                )
                fact[
                    (bc_params.sdsh < 0.1) & (bc_params.sdsc_iter[itr, :, :] < 0.1)
                ] = 1.0

                for i in range(iyr):
                    # Adjust ggm
                    ggs[:, i, :] = (ggs[:, i, :] - avs) * fact + avs

                # Copy ggd values to gdct
                gsct = ggs.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avs, sds, cors = mbc.avsds(ggs)

                    print("a4 mean gcmc sds (:,:4)", sds[:, :4])
                    print("a4 mean obs sdsh (:,:4)", bc_params.sdsh[:, :4])

            if (
                irho[2, 2] != 0 or irho[2, 3] != 0
            ):  # hist: seasonal corl, future: cmods, gmods, cobss, gobss
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit seasonal corrl itr, jj", itr, jj)
                    pass
                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for seasonal corrl", jj)

                    avs, sds, cors = mbc.avsds(ggs)

                    # Calculate fact
                    non_zero_mask = sds > 1e-10
                    fact = np.ones_like(sds)
                    fact[non_zero_mask] = sds[non_zero_mask]
                    fact[(sds < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        for iss in range(nss):
                            gmg = bc_params.gmods_iter[itr, iss, :nvar, :nvar]
                            cmg = bc_params.cmods_iter[itr, iss, :nvar, :nvar]
                            go = bc_params.dobss[iss, :nvar, :nvar]
                            co = bc_params.cobss[iss, :nvar, :nvar]

                            bt[:nvar] = (ggs[:nvar, i, iss] - avs[:nvar, iss]) / sds[
                                :nvar, iss
                            ]

                            if i == 0 and iss == 0:
                                gprev[:nvar] = bt[:nvar]
                                btprev[:nvar] = bt[:nvar]

                            temp = np.matmul(go, gmg)
                            temp1 = np.matmul(temp, bt)
                            temp4 = np.matmul(co, gprev)
                            temp = np.matmul(go, gmg)
                            temp2 = np.matmul(temp, cmg)
                            temp3 = np.matmul(temp2, btprev)

                            gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                            gprev[:nvar] = gcur[:nvar]
                            btprev[:nvar] = bt[:nvar]
                            gsct[:nvar, i, iss] = (
                                gcur[:nvar] * sds[:nvar, iss] + avs[:nvar, iss]
                            )

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avs, sds, cors = mbc.avsds(gsct)

                        print("a4 mean gcmc cors (:4,:4)", cors[:4, :4])
                    # print("a4 mean obs corsh (2,:)", bc_params.corsh[2, :])

            # ------- end calculate seasonal series of gcm series and store --------

            # ------- form annual series of corrected seasonal gcm series and store ----

            for k in range(nvar):
                for i in range(iyr):
                    sy1 = 0.0
                    for j in range(nss):
                        sy1 += gsct[k, i, j]

                    gy[k, i] = sy1 / iss

            ggy = gy.copy()
            gyct = gy.copy()

            if irho[3, 0] != 0:  # hist: annual mean, future: - avyc + avyh
                if tprint == 1 and itr > 0:
                    print("Correcting for annual mean", jj)

                # Calculate new values for ggy
                for i in range(iyr):
                    ggy[:, i] = (
                        gy[:, i] - bc_params.avyc_iter[itr, :]
                    ) + bc_params.avyh

                # Copy ggd values to gdct
                gyct = ggy.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avy, sdy, cory = mbc.avsdy(ggy)

                    print("a4 mean gcmc avy (:)", avy[:])
                    print("a4 mean obs avyh (:)", bc_params.avyh[:])

            if irho[3, 1] != 0:  # hist: annual sd, future: (x - avy)*sdyh/sdyc + avy
                if tprint == 1:
                    print("Correcting for annual mean and standard deviation", jj)

                avy, sdy, cory = mbc.avsdy(ggy)

                # Calculate fact
                non_zero_mask = bc_params.sdyc_iter[itr, :] > 1e-10
                fact = np.ones_like(bc_params.sdyc_iter[itr, :])
                fact[non_zero_mask] = (
                    bc_params.sdyh[non_zero_mask]
                    / bc_params.sdyc_iter[itr, :][non_zero_mask]
                )
                fact[(bc_params.sdyh < 0.1) & (bc_params.sdyc_iter[itr, :] < 0.1)] = 1.0

                for i in range(iyr):
                    # Adjust ggy
                    ggy[:, i] = (ggy[:, i] - avy) * fact + avy

                # Copy ggd values to gdct
                gyct = ggy.copy()

                if tprint == 1:
                    # Translating the function call to sdsmooth
                    avy, sdy, cory = mbc.avsdy(ggy)

                    print("a4 mean gcmc sdy (:)", sdy[:])
                    print("a4 mean obs sdyh (:)", bc_params.sdyh[:])

            if (
                irho[3, 2] != 0 or irho[3, 3] != 0
            ):  # hist: annual corl, future: cmody, gmody, cobsy, gobsy
                if itr == nntr - 1 and nxt > 0:
                    if tprint == 1 and itr > 0:
                        print("Exit annual corrl itr, jj", itr, jj)
                    pass

                else:
                    if tprint == 1 and itr > 0:
                        print("Correcting for annual corrl", jj)

                    avy, sdy, cory = mbc.avsdy(ggy)

                    # Calculate fact
                    non_zero_mask = sdy > 1e-10
                    fact = np.ones_like(sdy)
                    fact[non_zero_mask] = sdy[non_zero_mask]
                    fact[(sdy < 0.1)] = 1.0

                    bt = np.zeros(nvar)
                    gprev = np.zeros(nvar)
                    btprev = np.zeros(nvar)
                    gcur = np.zeros(nvar)

                    for i in range(iyr):
                        gmg = bc_params.gmody_iter[itr, :nvar, :nvar]
                        cmg = bc_params.cmody_iter[itr, :nvar, :nvar]
                        go = bc_params.dobsy[:nvar, :nvar]
                        co = bc_params.cobsy[:nvar, :nvar]

                        bt[:nvar] = (ggy[:nvar, i] - avy[:nvar]) / sdy[:nvar]

                        if i == 0:
                            gprev[:nvar] = bt[:nvar]
                            btprev[:nvar] = bt[:nvar]

                        temp = np.matmul(go, gmg)
                        temp1 = np.matmul(temp, bt)
                        temp4 = np.matmul(co, gprev)
                        temp = np.matmul(go, gmg)
                        temp2 = np.matmul(temp, cmg)
                        temp3 = np.matmul(temp2, btprev)

                        gcur[:nvar] = temp4[:nvar] + temp1[:nvar] - temp3[:nvar]
                        gprev[:nvar] = gcur[:nvar]
                        btprev[:nvar] = bt[:nvar]
                        gyct[:nvar, i] = gcur[:nvar] * sdy[:nvar] + avy[:nvar]

                    if tprint == 1:
                        # Translating the function call to sdsmooth
                        avy, sdy, cory = mbc.avsdy(gyct)

                        print("a4 mean gcmc cory (:)", cory[:])
                        # print("a4 mean obs coryh (:)", coryh[:])

            # -------------- end calculate the gcm series and store ----------------

            for i in range(iyr):
                for j in range(nout):
                    iss = mbc.iseas(j + 1, isn, ij, isj)
                    if time_scale != 0:
                        jd = 1
                    elif j == 1 and is_leap_year(iyr + startyear_h):
                        jd = 29
                    else:
                        jd = days_in_month[j]

                    for day_index in range(jd):
                        for k in range(nvar):
                            aa = gcmc[k, i, j, day_index]

                            if aa > missing_value:
                                af1 = 1.0
                                af2 = 1.0
                                af3 = 1.0
                                af4 = 1.0
                                # af5 = 1.0

                                if time_scale == 0:
                                    if gd[k, i, j, day_index] != 0.0:
                                        af1 = (
                                            gdct[k, i, j, day_index]
                                            / gd[k, i, j, day_index]
                                        )
                                if gm[k, i, j] != 0.0:
                                    af2 = gmct[k, i, j] / gm[k, i, j]
                                if gs[k, i, iss - 1] != 0.0:
                                    af3 = gsct[k, i, iss - 1] / gs[k, i, iss - 1]
                                if gy[k, i] != 0.0:
                                    af4 = gyct[k, i] / gy[k, i]

                                ac = af1 * af2 * af3 * af4 * 1

                                if ilimit[k] > 0:
                                    if aa < thres[k] and aa >= 0.0:
                                        ac = 1.0
                                    if aa >= thres[k] and (aa * ac) < thres[k]:
                                        ac = 1.001 * (thres[k] / aa)

                                ac1 = aa * ac

                                if ac1 > bc_params.phul[jj, k]:
                                    print(
                                        "Constrained value exceeds upper limit",
                                        jj,
                                        k,
                                    )
                                    ac1 = bc_params.phul[jj, k]
                                if ac1 < bc_params.phll[jj, k]:
                                    ac1 = bc_params.phll[jj, k]
                                    print(
                                        "Constrained value exceeds lower limit",
                                        jj,
                                        k,
                                    )

                                gcmc[k, i, j, day_index] = ac1

                            else:
                                gcmc[k, i, j, day_index] = missing_value

    print("Iteration", itr, "completed")

    return gcmc
