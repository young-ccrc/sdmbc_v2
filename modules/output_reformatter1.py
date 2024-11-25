#! usr/bin/python
# Funtions for main_reformat.py
# -----------------------------------------------------------------------------------------------------------------
# This is a Python script named "inputfunction.py" that contains a set of functions
# to be used in another script named "main_reformat.py".

# The script imports the following modules:
# "user_input", "numpy", "pandas", "xarray", and "glob".
# It defines several functions that will be used in the "main_reformat.py" script
# to perform some data processing tasks.
#
# The functions are:
#
# add_lev_dim(new): This function adds a new dimension named "lev" to a netCDF file.
#
# copyenv(new, old, vn, vo): This function copies the environment of 3D variables (latitude, longitude, and time)
# from an old netCDF file to a new one, as well as the attributes and encoding of the variable of interest.
#
# reformatsave(bcf, var_new, var_old, y): This function is the main function for reformatting and saving output
# as netCDF format. It opens raw GCM datasets, renames variable names, concatenates if needed,
# transposes dimensions, creates a new dataset, copies attributes, changes values, and encodes the output.

# Written by Youngil(Young) Kim
# PhD Candidate
# Water Research Centre
# Climate Change Research Centre
# University of New South Wales
# 2023-04-17
# -----------------------------------------------------------------------------------------------------------------
# Load pacakges ===================================
import glob
import os
import re
from functools import partial

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import xarray as xr  # type: ignore
import xesmf as xe  # type: ignore
from config import config
from interpolation import regrid

# Load pacakges end ================================
# ---------------------------------------------------------------------------------------------------
# Functions to be used for main_reformat


def add_lev_dim(new):
    """
    Add levels to the bias-corrected netcdf files

    """
    if "lev" not in new.dims:
        new = new.expand_dims(lev=1)
    return new


def copyenv(new, old, vn, vo):
    """
    Copies the environment of 3D variables from an old netCDF file to a new one,
    along with variable attributes and encoding.
    Also adds level dimension if the variable is not sea surface temperature.
    """
    # Copy dimensions and their attributes and encoding
    for dim in ["lat", "lon", "time"]:
        new[dim] = old[dim]
        new[dim].attrs = old[dim].attrs
        new[dim].encoding = {**old[dim].encoding, "_FillValue": None}

    # Copy variable attributes and encoding
    new[vn] = old[vo]
    new[vn].attrs = old[vo].attrs
    new[vn].encoding = {**old[vo].encoding, "_FillValue": None}

    # Add level dimension if not sea surface temperature
    if vn != "tos" or "sst":
        new["lev"] = old["lev"]
        new["lev"].attrs = old["lev"].attrs
        new["lev"].encoding = {**old["lev"].encoding, "_FillValue": None}

    # Copy global attributes
    new.attrs = old.attrs

    return new


def extract_time_range(filename):
    """
    Extracts the start and end time from the given filename.
    Assumes the filename contains the time range in the format 'YYYYMMDDHHMM-YYYYMMDDHHMM'.
    """
    match = re.search(r"(\d{12})-(\d{12})", filename)
    if match:
        start_time = match.group(1)
        end_time = match.group(2)
        return start_time, end_time
    else:
        raise ValueError(f"Filename {filename} does not contain a valid time range.")


def reformatsave_3d(bcf, var_new, var_old, y, input_files, out_path):
    """
    Reformat and save a 3D variable from bias-corrected data.

    Parameters:
    bcf (xarray.Dataset): An array containing bias-corrected GCM data over the research periods.
    var_new (str): Bias-corrected variable's name.
    var_old (str): Original GCM variable's name.
    y (int): The current year being processed.
    input_files (list): List of input file paths for the given year.
    out_path (str): Output path to save the netCDF files.

    Returns:
    list of tuples: Each tuple contains the reformatted dataset and its corresponding output filename.
    """
    # Open and concatenate datasets
    esmf = xr.open_mfdataset(input_files, combine="by_coords")

    # Extract time ranges
    time_ranges = [extract_time_range(fp) for fp in input_files]

    # Rename variable's name if necessary
    bcf_rn = bcf.rename({var_new: var_old})

    # Initialize list to store the results
    results = []

    for start_time, end_time in time_ranges:
        # Determine the appropriate slice
        if y == config.endyear_h:  # Last year concatenate
            xr_conf = xr.concat(
                [bcf_rn[var_old], esmf[var_old].isel(time=slice(-1, None))], dim="time"
            )
            start_time_full = f"{start_time[:4]}-{start_time[4:6]}-{start_time[6:8]} {start_time[8:10]}:{start_time[10:12]}"
            end_time_full = f"{end_time[:4]}-{end_time[4:6]}-{end_time[6:8]} {end_time[8:10]}:{end_time[10:12]}"
            bcf_sel = xr_conf.sel(time=slice(start_time_full, end_time_full))

        elif y == config.startyear_h - 1:  # Previous year concatenate
            bcf_1 = bcf.isel(time=slice(None, 1))  # Exclude first time step, mbcf
            bcf_rn = bcf_1.rename({var_new: var_old})
            start_time_full = f"{start_time[:4]}-{start_time[4:6]}-{start_time[6:8]} {start_time[8:10]}:{start_time[10:12]}"
            end_time_full = f"{end_time[:4]}-{end_time[4:6]}-{end_time[6:8]} {end_time[8:10]}:{end_time[10:12]}"
            bcf_sel = xr.concat(
                [
                    esmf[var_old].sel(time=slice(start_time_full, end_time_full)),
                    bcf_rn[var_old],
                ],
                dim="time",
            )

        else:
            start_time_full = f"{start_time[:4]}-{start_time[4:6]}-{start_time[6:8]} {start_time[8:10]}:{start_time[10:12]}"
            end_time_full = f"{end_time[:4]}-{end_time[4:6]}-{end_time[6:8]} {end_time[8:10]}:{end_time[10:12]}"
            bcf_sel = bcf_rn.sel(time=slice(start_time_full, end_time_full))

        # Transpose dimensions
        bcf_sel = bcf_sel.transpose("time", "lev", "lat", "lon")

        # Regrid to consider if
        sliced_esmf = (
            esmf[var_old]
            .sel(
                lat=slice(config.lat_min, config.lat_max),
                lon=slice(config.lon_min, config.lon_max),
            )
            .chunk({"time": -1, "lev": -1, "lat": -1, "lon": -1})
        )

        # Regrid to consider if
        if var_old in ["ua", "va"]:
            if not bcf_sel.lat.equals(sliced_esmf.lat) or not bcf_sel.lon.equals(
                sliced_esmf.lon
            ):
                bcf_sel["lev"] = sliced_esmf.lev
                print(
                    f"{var_old} has cooridnates not the same with the original needed to be regridded"
                )

                method = "bilinear"
                rename_dict_reformat = {var_new: var_old}
                weight_path_reformat = (
                    f"{output_path}/weight_{config.gname}_{var_old}_reformat.nc"
                )
                bcf_sel = regrid(
                    bcf_sel,
                    sliced_esmf,
                    method,
                    weight_path_reformat,
                    rename_dict_reformat,
                )

        # Convert the entire variable to a NumPy array
        esmf_data = esmf[var_old].load()

        # Update the values in the original dataset for the subset region
        esmf_data.loc[
            dict(
                time=bcf_sel.time,
                lev=esmf_data.lev,
                lat=bcf_sel.lat,
                lon=bcf_sel.lon,
            )
        ] = bcf_sel[var_old].values

        # Set attributes
        esmf_data.attrs["history"] = "Bias-corrected and reformatted data"

        # Assign the updated data back to the original dataset
        esmf[var_old] = esmf_data

        # None encoding for specific variables
        encoding_vars = ["lev_bnds", "b", "orog", "b_bnds"]
        if var_old != "va":
            encoding_vars.extend(["lat_bnds", "lon_bnds"])

        for var in encoding_vars:
            if var in esmf.variables:
                esmf[var].encoding["_FillValue"] = None

        # Determine the output filename
        output_filename = os.path.basename(input_files[0])
        output_path = os.path.join(out_path, output_filename)

        # Append the dataset and filename to results
        results.append((esmf, output_path))

    return results


def reformat_and_save_3d(
    bc_path,
    tlevel,
    startyear,
    endyear,
    input_vargcm,
    origin_vargcm,
):
    """
    Reformat and save 3D bias-corrected data to netCDF files.

    Parameters:
    bc_path (str): Path to the bias-corrected data files.
    tlevel (int): Total number of vertical levels.
    startyear (int): The first year of the data.
    endyear (int): The last year of the data.
    input_vargcm (list): List of input variable names from the GCM.
    origin_vargcm (list): List of original variable names from the GCM.
    out_path (str): Path to save the output netCDF files.
    infor (str): Additional information for the output file names.
    gname (str): Name of the GCM model.
    period (str): Time period information for the output file names.
    cinfor (str): Additional information for the output file names.
    sinfor (str): Additional information for the output file names.

    Returns:
    None: Saves the reformatted data to netCDF files in the specified output path.
    """

    # Load bias-corrected data
    list_3D = [
        sorted(glob.glob(os.path.join(bc_path, f"bc_corrected_3d_level_{idx}_*.nc")))
        for idx in range(0, tlevel + 1)
    ]
    ifile_3D = ["".join(list_3D[i]) for i in range(tlevel)]

    # Open the bias-corrected data with xarray
    partial_func = partial(add_lev_dim)
    bcf = xr.open_mfdataset(
        ifile_3D,
        preprocess=partial_func,
        concat_dim="lev",
        chunks={"time": 1000},
        data_vars="minimal",
        coords="minimal",
        compat="override",
        parallel=True,
        combine="nested",
    )

    # Update time dimension
    time = pd.date_range(f"{startyear}-01-01", freq="6h", periods=len(bcf.time))
    bcf = bcf.assign_coords({"time": time})

    # Loop through the input variables and years to reformat and save the data
    # Multiprocessing has not been used due to memory issue.
    for k in range(len(input_vargcm)):
        for y in range(startyear, endyear + 1):
            year = str(y)
            nyear = y + 1
            nyear = str(nyear)

            # Get the input file paths for the current year
            input_files = sorted(
                glob.glob(
                    f"{config.bc_hist_path}/{origin_vargcm[k]}/{config.sinfor}/v{config.version}/{origin_vargcm[k]}_*_{year}*.nc"
                )
            )

            # Reformat and save the 3D variables
            results = reformatsave_3d(
                bcf, input_vargcm[k], origin_vargcm[k], y, input_files, config.out_path
            )

            # Save data
            for dataset, output_filename in results:
                print(f"Save 3d to netcdf {output_filename}")
                dataset.load().to_netcdf(output_filename)
                print(f"Completed {output_filename}")


def reformatsave_2d(bcf, var_new, var_old, y):
    """
    Reformat and save a 2D variable from bias-corrected data.

    Parameters:
    bcf (xarray.Dataset): An array containing bias-corrected GCM data over the research periods.
    var_new (str): Bias-corrected variable's name.
    var_old (str): Original GCM variable's name.
    y (int): The current year being processed.

    Returns:
    xarray.Dataset: Reformatted dataset with the updated 2D variable.
    """
    # Set years
    year = str(y)
    nyear = y + 9
    if nyear > 2009:
        nyear = y + 4
    nyear = str(nyear)

    # Open raw GCM datasets
    file_path = f"{config.bc_hist_path}{var_old}_Oday_{config.gname}_{config.period}_{config.cinfor}_{config.sinfor}_{year}0101-{nyear}1231.nc"
    esmf = xr.open_dataset(file_path, keep_attrs=True)

    # Extract time ranges
    start_time, end_time = extract_time_range(file_path)

    # Rename variable's name
    sfcfd_rn = bcf.rename({var_new: var_old})

    # Concatenate if needed
    if y == config.starty_sfc_h:
        xr_conf = xr.concat(
            [
                esmf[var_old].sel(
                    time=slice(
                        f"{config.starty_sfc_h}-01-01",
                        f"{config.starty_sfc_h + 1}-12-31",
                    )
                ),
                sfcfd_rn[var_old],
            ],
            dim="time",
        )
        bcf_sel = xr_conf.sel(time=slice(f"{year}-01-01", f"{nyear}-12-31"))
    elif y == 2010:
        xr_conf = xr.concat(
            [
                sfcfd_rn[var_old],
                esmf[var_old].sel(time=slice("2013-01-01", "2014-12-31")),
            ],
            dim="time",
        )
        bcf_sel = xr_conf.sel(time=slice(f"{year}-01-01", f"{nyear}-12-31"))
    else:
        bcf_sel = sfcfd_rn.sel(time=slice(f"{year}-01-01", f"{nyear}-12-31"))

    # Create new dataset
    bcf_sel = xr.Dataset({var_old: bcf_sel})
    # Copy attributes
    bcf_sel = copyenv(bcf_sel, esmf, var_old, var_old)
    # Change values
    esmf[var_old] = bcf_sel[var_old]

    # None encoding
    esmf.time_bnds.encoding["_FillValue"] = None

    # Return reformatted dataset
    return esmf


def reformat_and_save_2d(
    bc_path,
    startyear,
    endyear,
    input_vargcm,
    origin_vargcm,
    out_path,
    infor,
    gname,
    period,
    cinfor,
    sinfor,
    starty_sfc,
    endy_sfc,
):
    """
    Reformat and save 2D bias-corrected data to netCDF files.

    Parameters:
    bc_path (str): Path to the bias-corrected data files.
    startyear (int): The first year of the data.
    endyear (int): The last year of the data.
    input_vargcm (list): List of input variable names from the GCM.
    origin_vargcm (list): List of original variable names from the GCM.
    out_path (str): Path to save the output netCDF files.
    infor (str): Additional information for the output file names.
    gname (str): Name of the GCM model.
    period (str): Time period information for the output file names.
    cinfor (str): Additional information for the output file names.
    sinfor (str): Additional information for the output file names.
    starty_sfc (int): The first year indicated in the name of the original surface data file.
    endy_sfc (int): The last year indicated in  the name of of the original surface data file.

    Returns:
    None: Saves the reformatted data to netCDF files in the specified output path.
    """

    # ================ Start creating input files, Surface fields ======================#
    with xr.open_dataset(f"{bc_path}sfc.bcd.output.nc") as sfcf:
        # update time dimension
        time = pd.date_range(
            "%s-01-01" % (startyear), freq="6H", periods=len(sfcf.time)
        )
        sfcf = sfcf.update({"time": time})
        # resample to daily
        sfcfd = sfcf.resample(time="D").mean("time")

    # Loop through the 2d input variable and years to reformat and save the data
    # Multiprocessing has not been used due to memory issue.
    for y in range(starty_sfc, endy_sfc + 1, 10):
        esmf = reformatsave_2d(sfcfd, input_vargcm[-1], origin_vargcm[-1], y)
        year = str(y)
        nyear = y + 9
        if nyear > 2009:
            nyear = y + 4
        nyear = str(nyear)

        # save data
        print("Save 2d to netcdf ", y)
        esmf.load().to_netcdf(
            f"{out_path}{origin_vargcm[-1]}_Oday_{gname}_{period}_{cinfor}_{sinfor}_{year}0101-{nyear}1231.nc"
        )
        print("Completed ", y)

    # Load appropriate time periods


# ---------------------------------------------------------------------------------------------------
