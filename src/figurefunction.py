#! usr/bin/python
# Funtions for main_figure.py
# -----------------------------------------------------------------------------------------------------------------
# This is a script for a comparison of the models used.
# It starts by importing several necessary packages
# including NumPy, pandas, xarray, and matplotlib.
# It also imports some user-defined variables and functions from another Python file called "user_input".

# After importing the packages, the file defines several functions,
# including "cal_mean" and "cal_std" for calculating statistics
# and "assign_w", "cal_acor", and "cal_ccor" for performing specific calculations in 3D.
# These functions calculate the mean, standard deviation, auto-, and cross- correlation over different time periods
# for each of the atmospheric variables. The resulting figures are saved in the specified output path.

# The plotting functions use the matplotlib and Basemap packages.

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-04-10
# -----------------------------------------------------------------------------------------------------------------

import glob
import os
import shutil

# Silence warnings
import warnings

import cartopy  # type: ignore

# import matplotlib.colors as colors
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs  # type: ignore
import cartopy.feature as cfeature  # type: ignore

# Plotting
import matplotlib.pyplot as plt  # Plotting

# Load pacakges ===================================
import numpy as np  # Arrays and matrix math
import pandas as pd

# import pandas as pd  # DataFrames
import xarray as xr  # Xarray # type: ignore

# from matplotlib.colors import BoundaryNorm
# from matplotlib.colors import ListedColormap
# from matplotlib.cm import get_cmap
from matplotlib.offsetbox import AnchoredText
from scipy.stats import ks_2samp

from config import config
from data_preparation import (
    assign_w_6hr,
    convert_to_daily_with_fraction,
    generate_file_paths,
    generate_file_paths_obs,
    load_preprocess_variable,
)

# from mpl_toolkits.basemap import Basemap
# from mpl_toolkits.basemap import cm


warnings.simplefilter("ignore")

# Load pacakges end ================================


# Functions for main_figure =========================================
# functions for 3D --------------------------------------------------

# startyear_h = config.startyear_h
# endyear_h = config.endyear_h
no_of_variables = config.no_of_variables
bc_boundary = config.bc_boundary
bc_hist_path = config.bc_hist_path
lat_max = config.lat_max
lat_min = config.lat_min
lon_max = config.lon_max
lon_min = config.lon_min
tlevel = config.tlevel
obs_path = config.obs_path
out_path = config.out_path
input_model = config.input_model
infor = config.infor
gname = config.gname
period = config.period
cinfor = config.cinfor
sinfor = config.sinfor
version = config.version
variables = config.target_variable
out_figure_path = config.out_figure_path

if config.period == "historical":
    startyear = config.startyear_h
    endyear = config.endyear_h
else:
    startyear = config.startyear_f
    endyear = config.endyear_f


# Set variable hus, ta, and w
def assign_w(ds):
    """
    Add the w variable (wind speed) to a given dataset 'ds'.

    """
    dsd = ds.resample(time="D").sum("time")
    ds_w = xr.apply_ufunc(np.hypot, ds["u"], ds["v"], output_dtypes=[ds["u"].dtype])
    dsd_w = ds_w.resample(time="D").sum("time")
    dsd = dsd.assign(w=dsd_w)
    dsd = dsd.drop(["u", "v"])
    return dsd


# Climatological mean
def cal_mean(ds):
    """
    Calculate the climatological mean of a given dataset ds over different time periods.

    """
    dsd = ds.groupby("time.dayofyear").mean("time").mean("dayofyear")

    return dsd


# Standard deviation
def cal_std(ds):
    """
    Calculate the standard deviation of a given dataset ds over different time periods.

    """
    dsd = ds.std("time")

    return dsd


# Climatological mean
def cal_mean_2d(ds):
    """
    Calculate the climatological mean of a given dataset ds over different time periods.

    """
    dss = ds.groupby("time.season").mean("time").mean("season")

    return dss


# Standard deviation
def cal_std_2d(ds):
    """
    Calculate the standard deviation of a given dataset ds over different time periods.

    """
    dss = ds.resample(time="QS-DEC").mean("time").std("time")

    return dss


# Lag1 auto-correlation
# def cal_acor(ds, period, variable):  # period: M, QS-DEC, Y
#     """
#     Calculate the lag1 auto-correlation of given variables 'variable'
#     in a given dataset 'ds' over a given time period 'period'.


#     """
#     if period != "D":
#         ds = ds.resample(time=period).mean("time")
#     temp = []
#     cor_matrix = np.zeros([len(variable), len(ds["lat"]), len(ds["lon"])])
#     for v in range(0, len(variable)):
#         var = variable[v]
#         for i in range(0, len(ds["lat"])):
#             for j in range(0, len(ds["lon"])):
#                 temp = np.corrcoef(
#                     ds[var][1:, i, j], ds[var][0 : (len(ds.time) - 1), i, j]
#                 )
#                 cor_matrix[v, i, j] = temp[0, 1]
#     return cor_matrix
def cal_acor(ds, period, var):
    """
    Calculate the lag1 auto-correlation of given variables in a dataset over a specific time period.

    Args:
        ds (xarray.Dataset): The dataset containing variables.
        period (str): Resampling period (e.g., 'D', 'M', 'Y').
        variables (list of str): List of variable names to calculate autocorrelation for.

    Returns:
        dict: A dictionary with autocorrelation values for each variable.
    """
    if period != "D":
        ds = ds.resample(time=period).mean()

    ds = ds.chunk({"time": -1})

    shifted_ds = ds[var].shift(time=1)
    autocorr = xr.apply_ufunc(
        lambda x, y: (
            np.corrcoef(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)])[
                0, 1
            ]
            if len(x[~np.isnan(x) & ~np.isnan(y)]) > 1
            else np.nan
        ),
        ds[var],
        shifted_ds,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return autocorr


# Lag0 cross-correlation
# def cal_ccor(ds, period, v):
#     """
#     Calculate the lag0 cross-correlation of given variables 'variable'
#     in a given dataset 'ds' over a given time period 'period'.


#     """
#     if period != "D":
#         ds = ds.resample(time=period).mean("time")
#     temp = []
#     cor_matrix = np.zeros([len(v), len(ds["lat"]), len(ds["lon"])])
#     for i in range(0, len(ds["lat"])):
#         for j in range(0, len(ds["lon"])):
#             temp = np.corrcoef(ds[v[0]][:, i, j], ds[v[1]][:, i, j])
#             cor_matrix[0, i, j] = temp[0, 1]
#             temp = np.corrcoef(ds[v[0]][:, i, j], ds[v[2]][:, i, j])
#             cor_matrix[1, i, j] = temp[0, 1]
#             temp = np.corrcoef(ds[v[1]][:, i, j], ds[v[2]][:, i, j])
#             cor_matrix[2, i, j] = temp[0, 1]
#     return cor_matrix
def cal_ccor(ds, period, vars_to_correlate):
    """
    Calculate the lag-0 cross-correlation for given pairs of variables in a dataset.

    Args:
        ds (xarray.Dataset): The dataset containing variables.
        period (str): Resampling period (e.g., 'D', 'M', 'Y').
        vars_to_correlate (list of tuple): List of variable pairs to calculate cross-correlation for.

    Returns:
        dict: A dictionary with cross-correlation values for each pair of variables.
    """
    if period != "D":
        ds = ds.resample(time=period).mean()

    ds = ds.chunk({"time": -1})  # Ensure time is a single chunk
    cross_correlations = {}

    for var1, var2 in vars_to_correlate:
        cross_corr = xr.apply_ufunc(
            lambda x, y: (
                np.corrcoef(
                    x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)]
                )[0, 1]
                if len(x[~np.isnan(x) & ~np.isnan(y)]) > 1
                else np.nan
            ),
            ds[var1],
            ds[var2],
            input_core_dims=[["time"], ["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        cross_correlations[(var1, var2)] = cross_corr

    return cross_correlations


def ks_pvalue(obs, sim):
    """Compute the KS test p-value for two 1D arrays."""
    if len(obs) == 0 or len(sim) == 0:
        return np.nan
    return ks_2samp(obs, sim).pvalue


def calculate_ks_agreement(obs_ds, sim_ds, variables, intervals):
    ks_agreement = {
        interval: {var: {"p>=0.05": 0, "p>=0.01": 0} for var in variables}
        for interval in intervals
    }

    for interval in intervals:
        obs_interval = obs_ds.sel(time=obs_ds.time.dt.hour == interval)
        sim_interval = sim_ds.sel(time=sim_ds.time.dt.hour == interval)

        for var in variables:
            p_values = xr.apply_ufunc(
                ks_pvalue,
                obs_interval[var],
                sim_interval[var],
                input_core_dims=[["time"], ["time"]],
                output_core_dims=[[]],
                vectorize=True,
            )

            ks_agreement[interval][var]["p>=0.05"] = (
                100 * (p_values >= 0.05).mean().item()
            )
            ks_agreement[interval][var]["p>=0.01"] = (
                100 * (p_values >= 0.01).mean().item()
            )

    return ks_agreement


# Function to extract p-values from dictionary format
def extract_p_values(ks_dict, threshold):
    if isinstance(ks_dict, dict):
        return ks_dict.get(threshold, float("nan"))
    elif isinstance(ks_dict, str):
        try:
            ks_dict = eval(ks_dict)
            return ks_dict.get(threshold, float("nan"))
        except:
            return float("nan")
    return float("nan")


def compute_ks_comparison(
    obs_ds,
    sim_ds_gcm,
    sim_ds_qm,
    variables,
    level,
    output_file_path,
    intervals=(0, 6, 12, 18),
):
    """
    Compute the KS test agreement for both GCM and QM-corrected datasets and return a formatted DataFrame.

    Parameters:
        obs_ds (xarray.Dataset): Observational dataset.
        sim_ds_gcm (xarray.Dataset): Original GCM dataset.
        sim_ds_qm (xarray.Dataset): QM-corrected dataset.
        variables (list): List of variables to analyze.
        intervals (tuple): 6-hour intervals to analyze.

    Returns:
        pd.DataFrame: KS test agreement results formatted as a comparison table.
    """

    # Compute KS agreement for GCM and QM datasets
    ks_results_gcm = calculate_ks_agreement(obs_ds, sim_ds_gcm, variables, intervals)
    ks_results_qm = calculate_ks_agreement(obs_ds, sim_ds_qm, variables, intervals)

    # Prepare data for comparison table
    data = []
    for interval in intervals:
        for var in variables:
            data.append(
                [
                    interval,
                    var,
                    ks_results_gcm[interval][var],  # KS agreement for GCM
                    ks_results_qm[interval][var],  # KS agreement for QM-corrected
                ]
            )

    # Create a DataFrame for comparison
    df_ks_comparison = pd.DataFrame(
        data,
        columns=["Time Interval", "Variable", "GCM KS Agreement", "SDMBC KS Agreement"],
    )

    # Extract "p≥0.05" and "p≥0.01" values for GCM and QM KS agreement
    ks_comparison = df_ks_comparison.copy()
    ks_comparison["GCM KS Agreement (p≥0.05)"] = df_ks_comparison[
        "GCM KS Agreement"
    ].apply(lambda x: extract_p_values(x, "p>=0.05"))
    ks_comparison["GCM KS Agreement (p≥0.01)"] = df_ks_comparison[
        "GCM KS Agreement"
    ].apply(lambda x: extract_p_values(x, "p>=0.01"))
    ks_comparison["SDMBC KS Agreement (p≥0.05)"] = df_ks_comparison[
        "SDMBC KS Agreement"
    ].apply(lambda x: extract_p_values(x, "p>=0.05"))
    ks_comparison["SDMBC KS Agreement (p≥0.01)"] = df_ks_comparison[
        "SDMBC KS Agreement"
    ].apply(lambda x: extract_p_values(x, "p>=0.01"))

    # Convert values to formatted percentage strings
    for col in [
        "GCM KS Agreement (p≥0.05)",
        "GCM KS Agreement (p≥0.01)",
        "SDMBC KS Agreement (p≥0.05)",
        "SDMBC KS Agreement (p≥0.01)",
    ]:
        ks_comparison[col] = ks_comparison[col].apply(
            lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A"
        )

    # Reorder the columns
    ks_comparison = ks_comparison[
        [
            "Time Interval",
            "Variable",
            "GCM KS Agreement (p≥0.05)",
            "GCM KS Agreement (p≥0.01)",
            "SDMBC KS Agreement (p≥0.05)",
            "SDMBC KS Agreement (p≥0.01)",
        ]
    ]
    ks_comparison.to_csv(
        f"{output_file_path}/ks_{level}_comparison_results.txt", sep="\t", index=False
    )

    return ks_comparison


def rounder(x):
    if x - int(x) >= 0.5:
        return int(np.ceil(x))
    else:
        return int(np.floor(x))


# def calculate_ks_matrix(de, ds, variable):
#     """
#     Calculate the Kolmogorov-Smirnov (KS) test matrix and percentage of p-values above a given threshold (0.05) for each variable.

#     Parameters:
#     de (xarray.Dataset): Observed xarray dataset containing variables to be compared.
#     ds (xarray.Dataset): Modelled xarray dataset containing variables to be compared.
#     variable (list): List of variable names to perform the KS test on.

#     Returns:
#     ks_result (numpy.ndarray): An array containing the percentage of p-values greater than or equal to 0.05 for each variable.
#     """

#     ks_matrix = np.empty((len(variable), len(ds["lat"]), len(ds["lon"])))

#     for k, var in enumerate(variable):
#         g_statistic, g_pvalue = xr.apply_ufunc(
#             ks_pvalue,
#             de[var],
#             ds[var],
#             input_core_dims=[["time"], ["time"]],
#             output_core_dims=[[], []],
#             vectorize=True,
#         )

#         ks_matrix[k, :, :] = g_pvalue

#     ks_result = np.empty([len(variable)])

#     for k in range(0, len(variable)):
#         ks_out = xr.where(ks_matrix[k] >= 0.05, 1, 0)
#         ks_result[k] = 100 * np.sum(ks_out) / (len(ds.lat) * len(ds.lon))

#     return ks_result


# Plot functions
# For lag1 auto-correlation
def scatter_auto(ax, ds, de, markers, alpha, label, sz, v, limlist):
    """
    Draw scatter plots for lag1 auto-correlation

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v, :],
            ds[m][v, :],
            s=sz,
            marker=markers[m],
            label=label[m],
            alpha=alpha,
        )
    xpoints, ypoints = plt.xlim(limlist), plt.ylim(limlist)
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# For climatological mean
def scatter_mean(ax, ds, de, markers, alpha, label, sz, v):
    """
    Draw scatter plots for mean

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v], ds[m][v], s=sz, marker=markers[m], label=label[m], alpha=alpha
        )
    xpoints, ypoints = plt.xlim(), plt.ylim()
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# For standard deviation
def scatter_sd(ax, ds, de, markers, alpha, label, sz, v):
    """
    Draw scatter plots for standard deviation

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v], ds[m][v], s=sz, marker=markers[m], label=label[m], alpha=alpha
        )
    xpoints, ypoints = plt.xlim(), plt.ylim()
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# For lag0 cross-correlation
def scatter_dot(ax, ds, de, markers, alpha, label, sz, v):
    """
    Draw scatter plots for lag0 cross-correlation

    """
    for m in range(0, 4):
        ax.scatter(
            de[m][v, :],
            ds[m][v, :],
            s=sz,
            marker=markers[m],
            label=label[m],
            alpha=alpha,
        )
    xpoints, ypoints = plt.xlim(), plt.ylim()
    ax.plot(
        xpoints, ypoints, linestyle="--", color="k", lw=1, scalex=False, scaley=False
    )


# Function to adjust longitude from 0-360 to -180 to 180
def adjust_longitude(ds):
    ds = ds.copy()  # Create a copy of the dataset to avoid modifying the original
    ds["lon"] = ((ds["lon"] + 180) % 360) - 180
    return ds.sortby("lon")


# Function to plot bias maps for multiple variables in a grid layout
def plot_bias_grid(
    data_gcm,
    data_mbc,
    data_era5,
    variables,
    level,
    statistic,
    title_prefix,
    cmap="RdBu_r",
    variable_limits=None,
    central_longitude=180,
):
    """
    Plot bias maps for multiple variables in a 3-row x 2-column grid layout.

    Parameters:
    data_gcm (xarray.Dataset): GCM dataset to plot.
    data_mbc (xarray.Dataset): MBC bias-corrected dataset to plot.
    data_era5 (xarray.Dataset): Reference ERA5 dataset.
    variables (list): List of variables to plot.
    title_prefix (str): Prefix for the plot titles.
    cmap (str): Colormap to use (default is 'RdBu_r').
    variable_limits (dict): Dictionary containing vmin and vmax for each variable.
    central_longitude (float): The central longitude for the map projection.
    """
    # Adjust longitude for all datasets
    data_gcm = data_gcm.sel(level=level).squeeze()
    data_mbc = data_mbc.sel(level=level).squeeze()
    data_era5 = data_era5.sel(level=level).squeeze()

    data_gcm = adjust_longitude(data_gcm)
    data_mbc = adjust_longitude(data_mbc)
    data_era5 = adjust_longitude(data_era5)
    units = ["m/s", "K", "g/kg"]
    # Set up the figure with 3 rows and 2 columns
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(20, 15),
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=central_longitude)
        },
    )

    for i, var in enumerate(variables):
        # GCM Bias vs ERA5
        ax_gcm = axes[i, 0]
        bias_gcm = data_gcm[var] - data_era5[var]
        vmin, vmax = variable_limits[var]["vmin"], variable_limits[var]["vmax"]
        bias_plot_gcm = ax_gcm.pcolormesh(
            bias_gcm["lon"],
            bias_gcm["lat"],
            bias_gcm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax_gcm.coastlines(resolution="50m", linewidth=1)
        ax_gcm.add_feature(cfeature.BORDERS, linestyle=":")
        ax_gcm.add_feature(cfeature.LAND, edgecolor="black", zorder=-1, alpha=0.3)
        ax_gcm.add_feature(cfeature.OCEAN, zorder=-1, alpha=0.3)
        ax_gcm.set_title(f"{title_prefix} GCM - ERA5, {var}", fontsize=12)
        cbar = fig.colorbar(
            bias_plot_gcm,
            ax=ax_gcm,
            orientation="horizontal",
            pad=0.05,
            extend="both",
            shrink=0.67,
            aspect=25,
        )  # Adjusted colorbar size
        cbar.set_label(f"{var} ({units[i]})")

        # MBC Bias vs ERA5
        ax_mbc = axes[i, 1]
        bias_mbc = data_mbc[var] - data_era5[var]
        bias_plot_mbc = ax_mbc.pcolormesh(
            bias_mbc["lon"],
            bias_mbc["lat"],
            bias_mbc,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax_mbc.coastlines(resolution="50m", linewidth=1)
        ax_mbc.add_feature(cfeature.BORDERS, linestyle=":")
        ax_mbc.add_feature(cfeature.LAND, edgecolor="black", zorder=-1, alpha=0.3)
        ax_mbc.add_feature(cfeature.OCEAN, zorder=-1, alpha=0.3)
        ax_mbc.set_title(f"{title_prefix} BC - ERA5, {var}", fontsize=12)
        cbar = fig.colorbar(
            bias_plot_mbc,
            ax=ax_mbc,
            orientation="horizontal",
            pad=0.05,
            extend="both",
            shrink=0.67,
            aspect=25,
        )  # Adjusted colorbar size
        cbar.set_label(f"{var} ({units[i]})")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(
        f"{out_figure_path}/sdmbc_{level}_{statistic}_{startyear}_{endyear}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


# Function to plot bias maps for multiple variables in a grid layout
def plot_bias_grid_auto(
    data_gcm,
    data_mbc,
    data_era5,
    ref,
    variables,
    level,
    statistic,
    title_prefix,
    cmap="RdBu_r",
    variable_limits=None,
    central_longitude=180,
):
    """
    Plot bias maps for multiple variables in a 3-row x 2-column grid layout.

    Parameters:
    data_gcm (xarray.Dataset): GCM dataset to plot.
    data_mbc (xarray.Dataset): MBC bias-corrected dataset to plot.
    data_era5 (xarray.Dataset): Reference ERA5 dataset.
    variables (list): List of variables to plot.
    title_prefix (str): Prefix for the plot titles.
    cmap (str): Colormap to use (default is 'RdBu_r').
    variable_limits (dict): Dictionary containing vmin and vmax for each variable.
    central_longitude (float): The central longitude for the map projection.
    """

    data_gcm = data_gcm.sel(level=level).squeeze()
    data_mbc = data_mbc.sel(level=level).squeeze()
    data_era5 = data_era5.sel(level=level).squeeze()

    # Adjust longitude for all datasets
    ref = adjust_longitude(ref)
    # Set up the figure with 3 rows and 2 columns
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(20, 15),
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=central_longitude)
        },
    )

    for i, var in enumerate(variables):
        # GCM Bias vs ERA5
        ax_gcm = axes[i, 0]
        bias_gcm = data_gcm[i] - data_era5[i]
        vmin, vmax = variable_limits[var]["vmin"], variable_limits[var]["vmax"]
        bias_plot_gcm = ax_gcm.pcolormesh(
            ref["lon"],
            ref["lat"],
            bias_gcm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax_gcm.coastlines(resolution="50m", linewidth=1)
        ax_gcm.add_feature(cfeature.BORDERS, linestyle=":")
        ax_gcm.add_feature(cfeature.LAND, edgecolor="black", zorder=-1, alpha=0.3)
        ax_gcm.add_feature(cfeature.OCEAN, zorder=-1, alpha=0.3)
        ax_gcm.set_title(f"{title_prefix} GCM - ERA5, {var}", fontsize=12)
        cbar = fig.colorbar(
            bias_plot_gcm,
            ax=ax_gcm,
            orientation="horizontal",
            pad=0.05,
            extend="both",
            shrink=0.67,
            aspect=25,
        )  # Adjusted colorbar size
        cbar.set_label(f"{var}")

        # MBC Bias vs ERA5
        ax_mbc = axes[i, 1]
        bias_mbc = data_mbc[i] - data_era5[i]
        vmin, vmax = variable_limits[var]["vmin"], variable_limits[var]["vmax"]
        bias_plot_mbc = ax_mbc.pcolormesh(
            ref["lon"],
            ref["lat"],
            bias_mbc,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax_mbc.coastlines(resolution="50m", linewidth=1)
        ax_mbc.add_feature(cfeature.BORDERS, linestyle=":")
        ax_mbc.add_feature(cfeature.LAND, edgecolor="black", zorder=-1, alpha=0.3)
        ax_mbc.add_feature(cfeature.OCEAN, zorder=-1, alpha=0.3)
        ax_mbc.set_title(f"{title_prefix} BC - ERA5, {var}", fontsize=12)
        cbar = fig.colorbar(
            bias_plot_mbc,
            ax=ax_mbc,
            orientation="horizontal",
            pad=0.05,
            extend="both",
            shrink=0.67,
            aspect=25,
        )  # Adjusted colorbar size
        cbar.set_label(f"{var}")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(
        f"{out_figure_path}/sdmbc_{level}_{statistic}_{startyear}_{endyear}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


# Function to plot bias maps for multiple variables in a grid layout
def plot_bias_grid_cross(
    data_gcm,
    data_mbc,
    data_era5,
    ref,
    variables,
    level,
    statistic,
    title_prefix,
    cmap="RdBu_r",
    variable_limits=None,
    central_longitude=180,
):
    """
    Plot bias maps for multiple variables in a 3-row x 2-column grid layout.

    Parameters:
    data_gcm (xarray.Dataset): GCM dataset to plot.
    data_mbc (xarray.Dataset): MBC bias-corrected dataset to plot.
    data_era5 (xarray.Dataset): Reference ERA5 dataset.
    variables (list): List of variables to plot.
    title_prefix (str): Prefix for the plot titles.
    cmap (str): Colormap to use (default is 'RdBu_r').
    variable_limits (dict): Dictionary containing vmin and vmax for each variable.
    central_longitude (float): The central longitude for the map projection.
    """

    data_gcm = data_gcm.sel(level=level).squeeze()
    data_mbc = data_mbc.sel(level=level).squeeze()
    data_era5 = data_era5.sel(level=level).squeeze()

    # Adjust longitude for all datasets
    ref = adjust_longitude(ref)
    ctitle = ["w & T", "w & q", "T & q"]
    # Set up the figure with 3 rows and 2 columns
    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(20, 15),
        subplot_kw={
            "projection": ccrs.PlateCarree(central_longitude=central_longitude)
        },
    )

    for i, var in enumerate(variables):
        # GCM Bias vs ERA5
        ax_gcm = axes[i, 0]
        bias_gcm = data_gcm[variables[i]] - data_era5[variables[i]]
        vmin, vmax = variable_limits[var]["vmin"], variable_limits[var]["vmax"]
        bias_plot_gcm = ax_gcm.pcolormesh(
            ref["lon"],
            ref["lat"],
            bias_gcm,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax_gcm.coastlines(resolution="50m", linewidth=1)
        ax_gcm.add_feature(cfeature.BORDERS, linestyle=":")
        ax_gcm.add_feature(cfeature.LAND, edgecolor="black", zorder=-1, alpha=0.3)
        ax_gcm.add_feature(cfeature.OCEAN, zorder=-1, alpha=0.3)
        ax_gcm.set_title(f"{title_prefix} GCM - ERA5, {ctitle[i]}", fontsize=12)
        cbar = fig.colorbar(
            bias_plot_gcm,
            ax=ax_gcm,
            orientation="horizontal",
            pad=0.05,
            extend="both",
            shrink=0.67,
            aspect=25,
        )  # Adjusted colorbar size
        cbar.set_label(f"{var}")

        # MBC Bias vs ERA5
        ax_mbc = axes[i, 1]
        bias_mbc = data_mbc[variables[i]] - data_era5[variables[i]]
        vmin, vmax = variable_limits[var]["vmin"], variable_limits[var]["vmax"]
        bias_plot_mbc = ax_mbc.pcolormesh(
            ref["lon"],
            ref["lat"],
            bias_mbc,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax_mbc.coastlines(resolution="50m", linewidth=1)
        ax_mbc.add_feature(cfeature.BORDERS, linestyle=":")
        ax_mbc.add_feature(cfeature.LAND, edgecolor="black", zorder=-1, alpha=0.3)
        ax_mbc.add_feature(cfeature.OCEAN, zorder=-1, alpha=0.3)
        ax_mbc.set_title(f"{title_prefix} BC - ERA5, {ctitle[i]}", fontsize=12)
        cbar = fig.colorbar(
            bias_plot_mbc,
            ax=ax_mbc,
            orientation="horizontal",
            pad=0.05,
            extend="both",
            shrink=0.67,
            aspect=25,
        )  # Adjusted colorbar size
        cbar.set_label(f"{var}")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(
        f"{out_figure_path}/sdmbc_{level}_{statistic}_{startyear}_{endyear}.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


# Scatter plot of 3d atmospheric variables
def save_figure_3d(file_paths_by_variable_gcm, file_paths_by_variable_obs, level):
    lat_range = (lat_min, lat_max)
    lon_range = (lon_min, lon_max)
    level = 0
    # Create a temporary folder for saving intermediate files
    temp_dir = os.path.join(out_figure_path, "temp_figure")
    os.makedirs(temp_dir, exist_ok=True)

    # =============== Load GCM ===============
    sliced_gcm = xr.Dataset()
    for var_name, file_paths in file_paths_by_variable_gcm.items():
        data_var = load_preprocess_variable(
            file_paths,
            var_name,
            level,
            lat_range,
            lon_range,
            startyear,
            endyear,
        )
        data_var = data_var.astype("float32")
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
        if isinstance(data_var, xr.Dataset):  # Ensure we extract the correct DataArray
            data_var = data_var[var_name]
        # print(data_var)
        sliced_gcm[var_name] = data_var

    if config.bc_boundary == "lateral":
        assign_gcm = assign_w_6hr(sliced_gcm, config.bc_boundary)
        dgcm_3d, fraction_factors_gcm = convert_to_daily_with_fraction(assign_gcm)
        # dgcm_3d = dgcm_3d.chunk({"time": 1000, "lat": 'auto', "lon": 'auto'})
        dgcm_3d.load().to_netcdf(
            f"{temp_dir}/temp_gcm_daily_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear}_{endyear}.nc"
        )
    # =============== Load GCM end ===============

    # =============== Load Obs ===============
    sliced_obs = xr.Dataset()

    # Load each variable and adjust longitude for ua and va if necessary
    for var_name, file_paths in file_paths_by_variable_obs.items():
        obs_var = load_preprocess_variable(
            file_paths,
            var_name,
            level,
            lat_range,
            lon_range,
            startyear,
            endyear,
        )
        obs_var = obs_var.astype("float32")
        # Add the processed variable to the dataset
        # print(obs_var)
        sliced_obs[var_name] = obs_var
    # sliced_obs = sliced_obs.chunk({"time": 500, "lat": -1, "lon": -1})
    # Rename variables
    if bc_boundary == "lateral":
        assign_obs = assign_w_6hr(sliced_obs, bc_boundary)
        dobs_3d, fraction_factors_obs = convert_to_daily_with_fraction(assign_obs)
        # dobs_3d = dobs_3d.chunk({"time": 1000, "lat": 'auto', "lon": 'auto'})
        dobs_3d.load().to_netcdf(f"{temp_dir}/temp_obs_daily_{startyear}_{endyear}.nc")
    # =============== Load Obs end ===============

    # Load the processed GCM and Obs data
    gcm = xr.open_dataset(
        f"{temp_dir}/temp_gcm_daily_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear}_{endyear}.nc"
    )
    era = xr.open_dataset(f"{temp_dir}/temp_obs_daily_{startyear}_{endyear}.nc")
    bcd = xr.open_dataset(
        f"{out_path}/bc_corrected_3d_lev_{level}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear}_{endyear}.nc"
    )
    assign_bcd = assign_w_6hr(bcd, bc_boundary)
    bcd_3d, fraction_factors_bcd = convert_to_daily_with_fraction(assign_bcd)
    bcd_3d = bcd_3d.compute()

    # calculate mean over whole periods
    g_d = cal_mean(gcm)
    e_d = cal_mean(era)
    d_d = cal_mean(bcd_3d)

    # Dictionary containing vmin and vmax for each variable
    variable_limits = {
        "w": {"vmin": -10, "vmax": 10},
        "ta": {"vmin": -3, "vmax": 3},
        "hus": {"vmin": -2, "vmax": 2},
    }

    # Plotting bias maps for each variable in a 3-row by 2-column grid
    variables = ["w", "ta", "hus"]
    plot_bias_grid(
        g_d,
        d_d,
        e_d,
        variables,
        level,
        "mean",
        "Daily Mean",
        cmap="RdBu_r",
        variable_limits=variable_limits,
        central_longitude=180,
    )

    # calculate mean over whole periods
    g_ds = cal_std(gcm)
    e_ds = cal_std(era)
    d_ds = cal_std(bcd_3d)

    variable_limits = {
        "w": {"vmin": -10, "vmax": 10},
        "ta": {"vmin": -1, "vmax": 1},
        "hus": {"vmin": -5, "vmax": 5},
    }

    plot_bias_grid(
        g_ds,
        d_ds,
        e_ds,
        variables,
        level,
        "std",
        "Daily Std.",
        cmap="RdBu_r",
        variable_limits=variable_limits,
        central_longitude=180,
    )

    # calculate auto-corr over whole periods
    d_da = cal_acor(adjust_longitude(bcd_3d), "D", variables)
    g_da = cal_acor(adjust_longitude(gcm), "D", variables)
    e_da = cal_acor(adjust_longitude(era), "D", variables)

    variable_limits = {
        "w": {"vmin": -0.5, "vmax": 0.5},
        "ta": {"vmin": -0.5, "vmax": 0.5},
        "hus": {"vmin": -0.15, "vmax": 0.15},
    }

    plot_bias_grid_auto(
        g_da,
        d_da,
        e_da,
        e_d,
        variables,
        level,
        "lag1",
        "Daily Auto-correlation.",
        cmap="RdBu_r",
        variable_limits=variable_limits,
        central_longitude=180,
    )

    variable_pairs = [("w", "ta"), ("ta", "hus"), ("hus", "w")]
    d_dc = cal_ccor(adjust_longitude(bcd_3d), "D", variable_pairs)
    g_dc = cal_ccor(adjust_longitude(gcm), "D", variable_pairs)
    e_dc = cal_ccor(adjust_longitude(era), "D", variable_pairs)

    variable_limits = {
        ("w", "ta"): {"vmin": -0.7, "vmax": 0.7},
        ("ta", "hus"): {"vmin": -0.7, "vmax": 0.7},
        ("hus", "w"): {"vmin": -0.7, "vmax": 0.7},
    }

    plot_bias_grid_cross(
        g_dc,
        d_dc,
        e_dc,
        e_d,
        variable_pairs,
        level,
        "cross",
        "Daily Cross-correlation.",
        cmap="RdBu_r",
        variable_limits=variable_limits,
        central_longitude=180,
    )

    if config.sub_daily_correction:
        print("K-S test has been included")
        compute_ks_comparison(
            fraction_factors_obs.compute(),
            fraction_factors_gcm.compute(),
            fraction_factors_bcd.compute(),
            variables,
            level,
            config.out_figure_path,
            intervals=(0, 6, 12, 18),
        )
    # Remove the temporary directory and its contents
    shutil.rmtree(temp_dir)
    print("Intermediate files deleted.")


# scatter plot should be here

# functions for sst -------------------------------------------------


# Auto-correlation for sst
def cal_acor_sst(ds, period):  # period: M, QS-DEC, Y
    """
    Calculate the lag1 auto-correlation of given surface variable 'variable'
    in a given dataset 'ds' over a given time period 'period'.

    """
    if period != "D":
        ds = ds.resample(time=period).mean("time")
    temp = []
    cor_matrix = np.zeros([len(ds.lat), len(ds.lon)])
    for i in range(0, len(ds.lat)):
        for j in range(0, len(ds.lon)):
            temp = np.corrcoef(ds[1:, i, j], ds[:-1, i, j])
            cor_matrix[i, j] = temp[0, 1]
    return cor_matrix


# Contour plot of sst
def save_figure_surface(g_sst, d_sst, e_sst, out_figure_path):
    """
    Save the contourf plots for two dimensional surface variable (sst)
    including mean absolute error.
    g_sst: an array containing data from the raw GCM data
    d_sst: an array containing data from the bias-corrected GCM data
    e_sst: an array containing data from the observed data
    out_figure_path: a string representing the path where the output figure will be saved

    """

    # Statistics calculation =======================
    print("Calculate statistics")
    # calculate mean over whole periods
    d_s = cal_mean_2d(d_sst)
    e_s = cal_mean_2d(e_sst)
    g_s = cal_mean_2d(g_sst)

    # calculate std over whole periods
    d_ss = cal_std_2d(d_sst)
    e_ss = cal_std_2d(e_sst)
    g_ss = cal_std_2d(g_sst)

    # calculate auto-corr over whole periods
    # d_da, d_ma, d_sa, d_ya = (
    #     cal_acor_sst(d_sst, "D"),
    #     cal_acor_sst(d_sst, "M"),
    #     cal_acor_sst(d_sst, "QS-DEC"),
    #     cal_acor_sst(d_sst, "Y"),
    # )
    # e_da, e_ma, e_sa, e_ya = (
    #     cal_acor_sst(e_sst, "D"),
    #     cal_acor_sst(e_sst, "M"),
    #     cal_acor_sst(e_sst, "QS-DEC"),
    #     cal_acor_sst(e_sst, "Y"),
    # )
    # g_da, g_ma, g_sa, g_ya = (
    #     cal_acor_sst(g_sst, "D"),
    #     cal_acor_sst(g_sst, "M"),
    #     cal_acor_sst(g_sst, "QS-DEC"),
    #     cal_acor_sst(g_sst, "Y"),
    # )

    d_sa = cal_acor_sst(d_sst, "QS-DEC")
    e_sa = cal_acor_sst(e_sst, "QS-DEC")
    g_sa = cal_acor_sst(g_sst, "QS-DEC")

    # nan where e is zero
    g_ss = xr.where(e_ss == 0, np.nan, g_ss)
    d_ss = xr.where(e_ss == 0, np.nan, d_ss)
    e_ss = xr.where(g_ss == 0, np.nan, e_ss)

    g_s = xr.where(e_s == 0, np.nan, g_s)
    d_s = xr.where(e_s == 0, np.nan, d_s)
    e_s = xr.where(g_s == 0, np.nan, e_s)

    gss_bias = g_ss - e_ss
    dss_bias = d_ss - e_ss

    gsm_bias = g_s - e_s
    dsm_bias = d_s - e_s

    gsa_bias = g_sa - e_sa
    dsa_bias = d_sa - e_sa

    gss_biasm = np.nanmean(abs(gss_bias))
    dss_biasm = np.nanmean(abs(dss_bias))

    gsm_biasm = np.nanmean(abs(gsm_bias))
    dsm_biasm = np.nanmean(abs(dsm_bias))

    gsa_biasm = np.nanmean(abs(gsa_bias))
    dsa_biasm = np.nanmean(abs(dsa_bias))

    # inputs
    datasets = [gsm_bias, gss_bias, gsa_bias, dsm_bias, dss_bias, dsa_bias]

    model_bias = [gsm_biasm, gss_biasm, gsa_biasm, dsm_biasm, dss_biasm, dsa_biasm]

    # col, row
    columns = 3
    rows = 2

    # contour levels and ticks
    mlevels = np.linspace(-2.0, 2.0, 61, endpoint=True, dtype=float)

    slevels = np.linspace(-0.5, 0.5, 61, endpoint=True, dtype=float)

    alevels = np.linspace(-0.2, 0.2, 61, endpoint=True, dtype=float)

    ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    fsize = 10
    tsize = 20

    # coords
    lat = e_s["lat"].values
    lon = e_s["lon"].values
    titles = ["GCM", "Bias-corrected GCM"]
    title = ["Seasonal M", "Seasonal Std.", "Seasonal Lag1"]

    print("Save figures")

    # draw figure
    fig = plt.figure(figsize=(14, 8))
    for i in range(1, columns * rows + 1):
        ax = fig.add_subplot(rows, columns, i, projection=ccrs.PlateCarree(180))
        if i in [1, 4]:
            levels = mlevels
        elif i in [2, 5]:
            levels = slevels
        else:
            levels = alevels

        # filled contours
        cf = ax.contourf(
            lon,
            lat,
            datasets[i - 1],
            levels=levels,
            cmap="RdBu",
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        plt.gca().set_facecolor("silver")

        # set backgrounds
        ax.coastlines()
        ax.add_feature(cartopy.feature.LAND, edgecolor="black")

        ax.set_xlabel("Longitude", fontsize=5)
        ax.set_ylabel("Latitude", fontsize=5)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.2,
            color="gray",
            alpha=0.7,
            linestyle="--",
            dms=True,
        )

        # add mean value
        at = AnchoredText(
            np.round(model_bias[i - 1], 1),
            prop=dict(size=15),
            frameon=True,
            loc="lower right",
        )

        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)

        if i < 4:
            if i == 1:
                # plt.ylabel(titles[0]+'\n ',fontsize=20)
                plt.ylabel(
                    titles[0] + "\n\n\n ", fontsize=fsize, multialignment="center"
                )
                plt.title(title[i - 1] + "\n ", fontsize=tsize)
                gl.top_labels = False
                gl.right_labels = False
                gl.bottom_labels = False
                # plt.gca().axes.get_xaxis().set_visible(True)
                plt.gca().axes.get_yaxis().set_visible(True)
                ax.set_yticklabels([], fontsize=5)
            else:
                plt.title(title[i - 1] + "\n ", fontsize=tsize)
                gl.top_labels = False
                gl.left_labels = False
                gl.right_labels = False
                gl.bottom_labels = False
        if i > 3 and i < 7:
            if i == 4:
                plt.ylabel(
                    titles[1] + "\n\n\n ", fontsize=fsize, multialignment="center"
                )
                gl.top_labels = False
                gl.right_labels = False
                # gl.bottom_labels = False
                plt.gca().axes.get_yaxis().set_visible(True)
                ax.set_yticklabels([], fontsize=5)
            else:
                gl.top_labels = False
                gl.right_labels = False
                gl.left_labels = False
                gl.bottom_labels = False

        if i == 4:
            # color bar location [left, bottom, width, height]
            cbaxes = fig.add_axes([0.1255, 0.05, 0.255, 0.01])
            cbar = fig.colorbar(
                cf,
                cax=cbaxes,
                ticks=[-2.0, -1.0, 0, 1.0, 2.0],
                orientation="horizontal",
                extend="both",
            )
            cbar.set_ticklabels([-2.0, -1.0, 0, 1.0, 2.0])
        elif i == 5:
            cbaxes = fig.add_axes([0.3855, 0.05, 0.255, 0.01])
            cbar = fig.colorbar(
                cf,
                cax=cbaxes,
                ticks=[-0.5, -0.25, 0, 0.25, 0.5],
                orientation="horizontal",
                extend="both",
            )
            cbar.set_ticklabels([-0.5, -0.25, 0, 0.25, 0.5])

        elif i == 6:
            cbaxes = fig.add_axes([0.6455, 0.05, 0.255, 0.01])
            cbar = fig.colorbar(
                cf,
                cax=cbaxes,
                ticks=[-0.2, -0.1, 0, 0.1, 0.2],
                orientation="horizontal",
                extend="both",
            )
            cbar.set_ticklabels([-0.2, -0.1, 0, 0.1, 0.2])

    plt.subplots_adjust(hspace=0.08, wspace=0.02)

    var_surface = "tos"
    f_surface_name = "seasonal_stats_all"
    plt.savefig(
        f"{out_figure_path}{var_surface}_{f_surface_name}.png",
        dpi=300,
        bbox_inches="tight",
    )


def figure_surface(
    out_path,
    obs_path,
    startyear,
    endyear,
    lat_range,
    lon_range,
    save_figure_surface,
    out_figure_path,
):
    """
    Analyzes surface variables and generates plots for bias-corrected data.
    Saves contour plots of sea surface temperature (SST) for
    raw GCM, bias-corrected GCM, and observation data.

    Args:
        out_path (str): Path to the folder containing bias-corrected data.
        obs_path (str): Path to the folder containing observation and GCM data.
        startyear (int): Start year for the analysis.
        lat_range (tuple): Latitude range (min, max) for the analysis.
        lon_range (tuple): Longitude range (min, max) for the analysis.
        save_figure_surface (function): Function to save SST contour plots.
        out_figure_path (str): Path to save the output figure.
    """

    # File paths
    gcm_sst_pattern = sorted(glob.glob(f"{obs_path}/tos_Oday_*_remapped.nc"))
    obs_sst_pattern = sorted(glob.glob(f"{obs_path}/tos_reanalysis_to_*.nc"))
    bcd_sst_file = sorted(
        glob.glob(f"{out_path}/bc_corrected_2d_Oday_*_{startyear}_{endyear}.nc")
    )

    # Load datasets
    dgcm_sst = xr.open_mfdataset(gcm_sst_pattern, combine="by_coords").sel(
        time=slice(f"{startyear}-01-01", f"{endyear}-12-31")
    )
    dobs_sst = xr.open_mfdataset(obs_sst_pattern, combine="by_coords").sel(
        time=slice(f"{startyear}-01-01", f"{endyear}-12-31")
    )
    dbcd_sst = xr.open_dataset(bcd_sst_file).sel(
        time=slice(f"{startyear}-01-01", f"{endyear}-12-31")
    )

    # Reformat time for consistency
    times = pd.date_range(
        start=f"{startyear}-01-01", freq="D", periods=len(dgcm_sst.time)
    )
    dgcm_sst["time"] = times
    dobs_sst["time"] = times
    dbcd_sst["time"] = times

    # Add coordinates to bias-corrected data if missing
    dbcd_sst = dbcd_sst.assign_coords(lat=dgcm_sst.lat, lon=dgcm_sst.lon)

    # Select spatial range for the analysis
    g_sst = dgcm_sst.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    e_sst = dobs_sst.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    d_sst = dbcd_sst.sel(
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    # Ensure the SST datasets are DataArrays or Datasets and loaded
    g_sst = g_sst.load() if hasattr(g_sst, "load") else g_sst
    e_sst = e_sst.load() if hasattr(e_sst, "load") else e_sst
    d_sst = d_sst.load() if hasattr(d_sst, "load") else d_sst

    # Replace SST values greater than 1000 or equal to 0 with NaN
    g_sst = xr.where((g_sst.tos > 1000) | (g_sst.tos == 0), np.nan, g_sst.tos)
    e_sst = xr.where((e_sst.tos > 1000) | (e_sst.tos == 0), np.nan, e_sst.tos)
    d_sst = xr.where((d_sst.tos > 1000) | (d_sst.tos == 0), np.nan, d_sst.tos)
    g_sst += 273.15
    # Resample to daily data
    g_sst = g_sst.resample(time="D").mean("time")
    e_sst = e_sst.resample(time="D").mean("time")
    d_sst = d_sst.resample(time="D").mean("time")
    # print(g_sst)
    # Save contour plot of SST
    save_figure_surface(g_sst, d_sst, e_sst, out_figure_path)


# functions for sst end ---------------------------------------------

# functions for ks-test ---------------------------------------------
# Added in the main script

# Functions for main_figure end =====================================
