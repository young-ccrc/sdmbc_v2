#! usr/bin/python
"""
Funtions for main_reformat.py
-----------------------------------------------------------------------------------------------------------------
This is a Python script named "inputfunction.py" that contains a set of functions
to be used in another script named "main_reformat.py".

The script imports the following modules:
"user_input", "numpy", "pandas", "xarray", and "glob".
It defines several functions that will be used in the "main_reformat.py" script
to perform some data processing tasks.

The functions are:

add_lev_dim(new): This function adds a new dimension named "lev" to a netCDF file.

copyenv(new, old, vn, vo): This function copies the environment of 3D variables (latitude, longitude, and time)
from an old netCDF file to a new one, as well as the attributes and encoding of the variable of interest.

reformatsave(bcf, var_new, var_old, y): This function is the main function for reformatting and saving output
as netCDF format. It opens raw GCM datasets, renames variable names, concatenates if needed,
transposes dimensions, creates a new dataset, copies attributes, changes values, and encodes the output.
"""
import argparse

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

import pandas as pd  # type: ignore
import xarray as xr  # type: ignore
from cdo import Cdo  # type: ignore
from config import config
from dask.distributed import Client  # type: ignore

from sdmbc_v2.interpolation import regrid

cdo = Cdo()
# Load pacakges end ================================
# ---------------------------------------------------------------------------------------------------
# Functions to be used for main_reformat


# Start of the script -----------------------------------------------------
def setup_client(n_workers=None, threads_per_worker=None):
    """
    Set up Dask client for parallel processing, allowing customization of workers and threads.

    Args:
        n_workers (int): Number of workers to use.
        threads_per_worker (int): Number of threads per worker.

    Returns:
        Client: A Dask distributed client instance.
    """

    if n_workers is None or threads_per_worker is None:
        c = Client()
    else:
        c = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
    print("Dask client setup complete.")
    return c


def parse_arguments():
    """
    Parse command-line arguments for running atmospheric data interpolation.

    Returns:
        argparse.Namespace: Parsed arguments including configuration file path, variable names,
        start year, and end year.
    """

    parser = argparse.ArgumentParser(description="Run atmospheric data interpolation.")
    parser.add_argument(
        "--config",
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
    #     "--nc",
    #     type=int,
    #     default=48,
    #     help="Number of cores to use.",
    # )

    parser.add_argument("--sy", type=int, default=1982, help="Start year.")
    parser.add_argument("--ey", type=int, default=2012, help="End year.")

    return parser.parse_args()


def add_lev_dim(new):
    """
    Add a level dimension ("lev") to the bias-corrected netCDF file if not present.

    Args:
        new (xarray.Dataset): The dataset to which the level dimension should be added.

    Returns:
        xarray.Dataset: Modified dataset with the level dimension added.
    """

    if "lev" not in new.dims:
        new = new.expand_dims(lev=1)
    return new


def copyenv(new, old, vn, vo):
    """
    Copy the environment of 3D variables (latitude, longitude, time) from an old netCDF file to a new one.
    Also copies the attributes, encoding, and level dimension if applicable.

    Args:
        new (xarray.Dataset): The new dataset to which information will be copied.
        old (xarray.Dataset): The old dataset providing the information.
        vn (str): Name of the variable in the new dataset.
        vo (str): Name of the variable in the old dataset.

    Returns:
        xarray.Dataset: Updated new dataset with copied environment and attributes.
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
    Extract the start and end time from the given filename.
    Assumes the filename contains the time range in the format 'YYYYMMDDHHMM-YYYYMMDDHHMM'.

    Args:
        filename (str): The filename from which to extract the time range.

    Returns:
        tuple: A tuple containing the start and end times as strings.

    Raises:
        ValueError: If the filename does not contain a valid time range.
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

    Args:
        bcf (xarray.Dataset): Bias-corrected GCM data over the research periods.
        var_new (str): New variable name for bias-corrected data.
        var_old (str): Original GCM variable name.
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

    Args:
        bc_path (str): Path to the bias-corrected data files.
        tlevel (int): Total number of vertical levels.
        startyear (int): The first year of the data.
        endyear (int): The last year of the data.
        input_vargcm (list): List of input variable names from the GCM.
        origin_vargcm (list): List of original variable names from the GCM.

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


def add_lat_lon_bnds(input_file, output_file):
    """
    Add latitude and longitude bounds if they are missing in the input file.

    Args:
        input_file (str): Path to the input netCDF file.
        output_file (str): Path to the output netCDF file where bounds are added.

    Returns:
        None: Saves the updated netCDF file.
    """

    ds = xr.open_dataset(input_file)

    if "lat_bnds" not in ds.variables or "lon_bnds" not in ds.variables:
        print("Adding lat_bnds and lon_bnds to the input file...")
        lat = ds["lat"].values
        lon = ds["lon"].values

        lat_bnds = xr.DataArray(
            [[lat[i] - 0.5, lat[i] + 0.5] for i in range(len(lat))],
            dims=["lat", "bnds"],
        )
        lon_bnds = xr.DataArray(
            [[lon[i] - 0.5, lon[i] + 0.5] for i in range(len(lon))],
            dims=["lon", "bnds"],
        )

        ds["lat_bnds"] = lat_bnds
        ds["lon_bnds"] = lon_bnds

        ds.to_netcdf(output_file)
    else:
        print("lat_bnds and lon_bnds already exist in the file.")

    ds.close()


def reformat_and_save_2d(input_path, original_file, output_file, remap_weights_file):
    """
    Regrid the bias-corrected input data to the original grid using CDO.

    Args:
        input_path (str): Path to the bias-corrected input netCDF file.
        original_file (str): Path to the original netCDF file to use as a grid reference.
        output_file (str): Path to the output netCDF file after regridding.
        remap_weights_file (str): Path to store remapping weights generated during regridding.

    Returns:
        None: Saves the regridded netCDF file.
    """

    # Add lat_bnds and lon_bnds to the input file if necessary
    input_with_bnds = input_path.replace(".nc", "_with_bnds.nc")
    add_lat_lon_bnds(input_path, input_with_bnds)

    # Generate remapping weights
    print("Generating remap weights...")
    cdo.genbil(original_file, input=input_with_bnds, output=remap_weights_file)

    # Regrid the bias-corrected data to match the grid of the original file
    print("Regridding the input data to the original grid...")
    cdo.remap(
        f"{original_file},{remap_weights_file}",
        input=input_with_bnds,
        output=output_file,
    )

    print(f"Regridding completed: {input_with_bnds} -> {output_file}")


# ---------------------------------------------------------------------------------------------------
def main(config):
    """
    Main function to initiate the reformatting process for bias-corrected GCM data.

    Args:
        config (module): Configuration object that contains user-defined parameters such as
        start year, end year, output paths, etc.

    Returns:
        None: Initiates the reformatting for both 2D and 3D data based on the given configuration.
    """

    setup_client()

    startyear_h = config.startyear_h
    endyear_h = config.endyear_h
    startyear_h = config.startyear_h
    bc_boundary = config.bc_boundary
    tlevel = config.tlevel
    out_path = config.out_path
    target_variable = config.target_variable

    print("Start reformatting")
    if bc_boundary == "lateral":
        reformat_and_save_3d(
            out_path,
            tlevel,
            startyear_h,
            endyear_h,
            target_variable,
            target_variable,
        )
        print("Finish 3D reformatting")

    else:
        target_files = sorted(
            glob.glob(
                f"{config.bc_hist_path}/{target_variable[0]}/{config.sinfor}/v{config.version}/{config.target_variable[0]}_*.nc"
            )
        )
        input_files = sorted(
            glob.glob(os.path.join(out_path, f"bc_corrected_2d_*_{startyear_h}*.nc"))
        )
        # obs_files = sorted(glob.glob(f"{config.obs_path}/{config.target_variable_sst[0]}_*.nc"))
        remap_weights_file = f"{out_path}/remap_weights_{target_variable[0]}_{startyear_h}_{endyear_h}.nc"
        output_file = f"{out_path}/{target_variable[0]}_{config.infor}_{config.gname}_{config.period}_{config.cinfor}_{config.sinfor}_{startyear_h}0101-{endyear_h}1231.nc"

        reformat_and_save_2d(
            input_files[0], target_files[0], output_file, remap_weights_file
        )
        print("Finish 2D reformatting")
