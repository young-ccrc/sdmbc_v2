#! usr/bin/python
# =======================================================================================
# Description
#
# Input file formats
# - Atmospheric fields: 3D.%model%.%idx%.input.nc
# - Surface fields: sfc.%model%.input.nc

# This script is to draw the output figures of bias-corrected GCM with regard to Obs.
# This includes
# 1. Scatter plot of three statistics: mean, standard deviation, lag1 auto-correlation
# 2. Scatter plot of cross-correlation between atmospheric variables
# 3. Contour plot of surface vairable: sea surface temperature (SST)
# 4. The Kolmogorov-Smirnov (K-S) test result for each variable: hus, ta, w

# One of the grid cells within a speficied domain will be chosen to test the output.

import glob

import numpy as np
import pandas as pd

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-04-17
# =======================================================================================
import xarray as xr  # type: ignore
from config import config
from scipy.stats import ks_2samp

from sdmbc_v2.data_preparation import load_config
from sdmbc_v2.figurefunction import (
    assign_w,
    calculate_ks_matrix,
    rounder,
    save_figure_3d,
    save_figure_3d_cross,
    save_figure_surface,
)


class AnalysisBC:
    """
    A class to analyze bias-corrected climate model data.

    Attributes:
    -----------
    bc_path : str
        Path to the folder containing the bias-corrected data.
    lat_range : tuple
        A tuple containing the range of latitude indices to analyze.
    lon_range : tuple
        A tuple containing the range of longitude indices to analyze.
    startyear : int
        The starting year of the data.
    out_figure_path : str
        The path to save output figures.
    kstest : bool, optional
        If True, the Kolmogorov-Smirnov (K-S) test will be performed,
        otherwise not. Default is False.
    """

    def __init__(
        self,
        gcm,
        obs,
        bcd,
        variable,
        out_figure_path,
        kstest=False,
    ):
        self.dgcm_3d = gcm
        self.dobs_3d = obs
        self.dbcd_3d = bcd
        self.variable = variable
        self.out_figure_path = out_figure_path
        self.kstest = kstest

    def figure_atmos(self):
        """
        Analyzes atmospheric variables and generates plots for bias-corrected data.
        Saves scatter plots of mean, standard deviation, and lag1 auto-correlation
        for the atmospheric variables, as well as cross-correlation between them.
        Optionally, it can also perform the Kolmogorov-Smirnov (K-S) test
        and print the results.
        """

        # Save 1. Scatter plot of three statistics: mean, standard deviation, lag1 auto-correlation
        save_figure_3d(self.dgcm_3d, self.dobs_3d, self.dbcd_3d, self.out_figure_path)

        # Save 2. Scatter plot of cross-correlation between atmospheric variables
        save_figure_3d_cross(
            self.dgcm_3d, self.dobs_3d, self.dbcd_3d, self.out_figure_path
        )

        if self.kstest == True:
            # Save 4. The Kolmogorov-Smirnov (K-S) test result for each variable: hus, ta, w
            # Perform KS test on the two samples
            self.gks_result = calculate_ks_matrix(
                self.dobs_3d, self.dgcm_3d, self.variable
            )
            self.dks_result = calculate_ks_matrix(
                self.dobs_3d, self.dbcd_3d, self.variable
            )

            for j in range(0, len(self.variable)):
                var = self.variable[j]
                print("")
                print(
                    f"Kolmogorov-Smirnov (K-S) test over the domain for the variable: {var}"
                )
                # print(f"For variable {var}")
                print("")
                print(f"• KS test for raw GCM")
                print(
                    f" - {rounder(self.gks_result[j])}% of the raw GCM are likely to be drawn from the same distribution."
                )
                print("")
                print(f"• KS test for bias-corrected GCM")
                print(
                    f" - {rounder(self.dks_result[j])}% of the bias-corrected GCM are likely to be drawn from the same distribution."
                )
                print("")
                print("• Result")
                print(
                    f" - The bias-corrected GCM presents {rounder(self.dks_result[j])-rounder(self.gks_result[j])}% improvement compared to the raw GCM."
                )
                print("")

    def figure_surface(self):
        """
        Analyzes surface variables and generates plots for bias-corrected data.
        Saves contour plots of sea surface temperature (SST) for
        raw GCM, bias-corrected GCM, and observation data.
        """

        input_gcm_sst = "sfc.gcm.input.nc"
        input_obs_sst = "sfc.obs.input.nc"
        input_bcd_sst = "sfc.bcd.output.nc"

        self.dgcm_sst = xr.open_dataset(f"{self.bc_path}{input_gcm_sst}")
        self.dobs_sst = xr.open_dataset(f"{self.bc_path}{input_obs_sst}")
        self.dbcd_sst = xr.open_dataset(f"{self.bc_path}{input_bcd_sst}")

        # Reformat time
        times = pd.date_range(
            "%s-01-01" % (self.startyear), freq="6H", periods=len(self.dgcm_sst.time)
        )

        self.dgcm_sst = self.dgcm_sst.update({"time": times})
        self.dobs_sst = self.dobs_sst.update({"time": times})
        self.dbcd_sst = self.dbcd_sst.update({"time": times})

        # Add coordinates
        self.dbcd_sst["lat"], self.dbcd_sst["lon"] = (
            self.dgcm_sst.lat,
            self.dgcm_sst.lon,
        )

        # Select a first grid cell to test sst output
        self.g_sst = self.dgcm_sst.isel(
            lat=slice(self.lat_range[0], self.lat_range[1]),
            lon=slice(self.lon_range[0], self.lon_range[1]),
        )
        self.e_sst = self.dobs_sst.isel(
            lat=slice(self.lat_range[0], self.lat_range[1]),
            lon=slice(self.lon_range[0], self.lon_range[1]),
        )
        self.d_sst = self.dbcd_sst.isel(
            lat=slice(self.lat_range[0], self.lat_range[1]),
            lon=slice(self.lon_range[0], self.lon_range[1]),
        )

        # Replace sst values greater than 1000 or equal to 0 with NaN
        self.g_sst = xr.where(
            (self.g_sst > 1000) | (self.g_sst == 0), np.nan, self.g_sst
        )
        self.e_sst = xr.where(
            (self.e_sst > 1000) | (self.e_sst == 0), np.nan, self.e_sst
        )
        self.d_sst = xr.where(
            (self.d_sst > 1000) | (self.d_sst == 0), np.nan, self.d_sst
        )

        self.g_sst = self.g_sst.resample(time="D").sum("time")
        self.e_sst = self.e_sst.resample(time="D").sum("time")
        self.d_sst = self.d_sst.resample(time="D").sum("time")

        # Save 3. Contour plot of surface vairable: sea surface temperature (SST)
        save_figure_surface(self.g_sst, self.d_sst, self.e_sst, self.out_figure_path)


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
    bcd_sst_file = f"{out_path}/bc_corrected_2d_Oday_*_{startyear}_{endyear}.nc"

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
