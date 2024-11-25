import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import dask  # type: ignore
import numpy as np
import pandas as pd
import xarray as xr  # type: ignore

# import yaml  # type: ignore
from bc_grid_function import bc_correction_grid_cell_hist
from config import config  # type: ignore
from dask.distributed import Client  # type: ignore
from data_preparation import (
    assign_w_6hr,
    assign_w_day,
    convert_to_daily_with_fraction,
    extract_and_reshape_delayed,
    generate_file_paths,
    generate_file_paths_obs,
    load_config,
    load_preprocess_variable,
    validate_inputs,
)

from sdmbc_v2.analysis_plot import AnalysisBC  # type: ignore
from sdmbc_v2.output_reformatter import reformat_and_save_2d  # type: ignore
from sdmbc_v2.output_reformatter import reformat_and_save_3d  # type: ignore

# import sys
# sys.path.append('/scratch/dm6/yk8692/sdmbc/')


warnings.simplefilter("ignore", UserWarning)


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
    parser.add_argument(
        "--nc",
        type=int,
        default=28,
        help="Number of cores to use.",
    )

    parser.add_argument(
        "--mp",
        type=int,
        default=64,
        help="Amount of RAM (in GB) to allocate.",
    )

    parser.add_argument(
        "--np",
        type=int,
        default=4,
        help="Number of multiprocessing processes to use.",
    )
    parser.add_argument(
        "--mpp",
        type=int,
        default=32,
        help="Amount of RAM (in GB) to allocate per process.",
    )
    parser.add_argument("--sy", type=int, default=1982, help="Start year.")
    parser.add_argument("--ey", type=int, default=2012, help="End year.")

    return parser.parse_args()


def main(config):

    setup_client()

    startyear_h = config.startyear_h
    endyear_h = config.endyear_h
    no_of_variables = config.no_of_variables
    startyear_h = config.startyear_h
    bc_boundary = config.bc_boundary
    bc_hist_path = config.bc_hist_path
    input_vargcm_2d = config.input_vargcm_2d
    input_vargcm_3d = config.input_vargcm_3d
    lat_max = config.lat_max
    lat_min = config.lat_min
    lon_max = config.lon_max
    lon_min = config.lon_min
    level = config.level
    obs_path = config.obs_path
    out_path = config.out_path
    infor = config.infor
    gname = config.gname
    period = config.period
    cinfor = config.cinfor
    sinfor = config.sinfor
    version = config.version
    origin_vargcm_3d = config.origin_vargcm_3d

    # Validate inputs
    validate_inputs(lat_min, lat_max)
    var_list_w = ["w", "ta", "hus"] if bc_boundary == "lateral" else input_vargcm_2d
    var_list = input_vargcm_3d if bc_boundary == "lateral" else input_vargcm_2d

    # if bc_hist:  # historical bias correction

    # List of variables to generate file paths for
    variables = origin_vargcm_3d

    # Define parameters for selection
    lat_range = (lat_min, lat_max)  # Adjust as needed
    lon_range = (lon_min, lon_max)  # Adjust as needed

    # =============== Load GCM ===============
    # Generate file paths for each variable
    file_paths_by_variable = {}
    for variable in variables:
        file_paths_by_variable[variable] = generate_file_paths(
            bc_hist_path,
            variable,
            infor,
            gname,
            period,
            cinfor,
            sinfor,
            version,
            startyear_h,
            endyear_h,
        )

    sliced_gcm = xr.Dataset()

    # Load each variable and adjust longitude for ua and va if necessary
    for var_name, file_paths in file_paths_by_variable.items():
        data_var = load_preprocess_variable(
            file_paths, var_name, level, lat_range, lon_range
        )
        # Check if the variable is one of the wind components with different lon
        if var_name in ["ua", "va"]:
            # Let's assume hus and ta have the target longitude values, and they are already loaded
            target_lon = sliced_gcm.lon
            target_lev = sliced_gcm.lev
            target_lat = sliced_gcm.lat

            # Assign the adjusted longitude values to ua or va
            data_var = data_var.assign_coords(
                lon=target_lon, lat=target_lat, lev=target_lev
            )
        # Add the processed variable to the dataset
        sliced_gcm[var_name] = data_var

    # Rename variables
    # rename_dict = {"hus": "q", "ta": "t", "ua": "u", "va": "v"}
    sliced_gcm = sliced_gcm.sel(
        time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31")
    )
    assign_gcm = assign_w_6hr(sliced_gcm, config.bc_boundary)
    assign_gcm = assign_gcm[["w", "ta", "hus"]]
    daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(assign_gcm)
    daily_gcm_rechunk = daily_gcm.chunk({"time": -1, "lat": "auto", "lon": "auto"})
    print("start delayed process")
    reshaped_gcm_delayed = extract_and_reshape_delayed(
        daily_gcm_rechunk, 3, config.startyear_h, config.endyear_h
    )

    # =============== Load GCM end ===============

    # =============== Load Obs ===============

    file_paths_by_variable = {}
    for variable in variables:
        file_paths_by_variable[variable] = generate_file_paths_obs(
            obs_path,
            variable,
            gname,
            startyear_h,
            endyear_h,
        )

    sliced_obs = xr.Dataset()

    # Load each variable and adjust longitude for ua and va if necessary
    for var_name, file_paths in file_paths_by_variable.items():
        obs_var = load_preprocess_variable(
            file_paths, var_name, level, lat_range, lon_range
        )
        # Check if the variable is one of the wind components with different lon
        if var_name in ["ua", "va"]:
            # Let's assume hus and ta have the target longitude values, and they are already loaded
            target_lon = sliced_obs.lon
            target_lev = sliced_obs.lev
            target_lat = sliced_obs.lat

            # Assign the adjusted longitude values to ua or va
            obs_var = obs_var.assign_coords(
                lon=target_lon, lat=target_lat, lev=target_lev
            )
        # Add the processed variable to the dataset
        sliced_obs[var_name] = obs_var

    # Rename variables
    sliced_obs = sliced_obs.sel(
        time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31")
    )
    assign_obs = assign_w_6hr(sliced_obs, config.bc_boundary)
    assign_obs = assign_obs[["w", "ta", "hus"]]
    daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(assign_obs)
    daily_obs_rechunk = daily_obs.chunk({"time": -1, "lat": "auto", "lon": "auto"})
    print("start delayed process")
    reshaped_obs_delayed = extract_and_reshape_delayed(
        daily_obs_rechunk, 3, config.startyear_h, config.endyear_h
    )
    # =============== Load obs end ===============

    # Initialize empty arrays to hold the results.
    dims_time, dims_lat, dims_lon = (
        daily_gcm["time"].shape[0],
        len(daily_gcm.lat),
        len(daily_gcm.lon),
    )
    bc_params_array = np.empty((dims_lat, dims_lon), dtype=object)

    # List to store the results for each grid cell
    corrected_data_dict = {}

    def process_cell(lat_lon):
        i, j = lat_lon
        lat, lon = (
            reshaped_gcm_delayed.lat.values[i],
            reshaped_gcm_delayed.lon.values[j],
        )
        bc_corrected_gcm_hist, bc_params = bc_correction_grid_cell_hist(
            lat,
            lon,
            reshaped_gcm_delayed,
            reshaped_obs_delayed,
            fraction_factors_gcm,
            fraction_factors_obs,
            var_list_w,
            sliced_gcm,
        )
        return (
            i,
            j,
            bc_corrected_gcm_hist,
            bc_params.to_dict(),
        )  # Assuming to_dict() is implemented

    # Prepare lat/lon pairs for processing
    lat_lon_pairs = [(i, j) for i in range(dims_lat) for j in range(dims_lon)]

    # Use a process pool to parallelize
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_cell, lat_lon_pairs))

    # Store results
    for i, j, bc_corrected_gcm_hist, bc_params_dict in results:
        bc_params_array[i, j] = bc_params_dict  # Storing as dict for serialization
        corrected_data_dict[
            (reshaped_gcm_delayed.lat.values[i], reshaped_gcm_delayed.lon.values[j])
        ] = bc_corrected_gcm_hist.assign_coords(
            lat=reshaped_gcm_delayed.lat.values[i],
            lon=reshaped_gcm_delayed.lon.values[j],
        ).expand_dims(
            ["lat", "lon"]
        )

    # Save the BC model
    np.save(f"{out_path}/bc_params_{gname}.npy", bc_params_array)

    corrected_data_list = list(corrected_data_dict.values())

    # Combine the DataArrays into a single Dataset
    ds_corrected = xr.combine_by_coords(corrected_data_list)

    # Set attributes
    ds_corrected.attrs["history"] = "Bias-corrected data"

    if config.save_bc_output:
        # save the bias corrected data
        ds_corrected.to_netcdf(
            f"{out_path}/bc_corrected_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{version}.nc"
        )

    if config.draw_figure:
        ds_corrected_day = assign_w_day(ds_corrected, bc_boundary)

        print("Drawing figures")
        if config.sub_daily_correction:
            print("K-S test has been included")
            statistics = AnalysisBC(
                daily_gcm,
                daily_obs,
                ds_corrected_day,
                config.origin_vargcm_3d,
                config.out_figure_path,
                kstest=True,
            )
        if config.sub_daily_correction == False:
            print("K-S test has not been included")
            statistics = AnalysisBC(
                daily_gcm,
                daily_obs,
                ds_corrected_day,
                config.origin_vargcm_3d,
                config.out_figure_path,
                kstest=False,
            )
        statistics.figure_atmos()
        print("Finish 3d field")
        if (level == 1) and (no_of_variables >= 4):
            statistics.figure_surface()
            print("Finish 2d field")

    if config.reformat_to_original:
        print("Start reformatting")

        reformat_and_save_3d(
            bc_path,
            tlevel,
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
        )
        print("Finish 3D reformatting")

        if (level == 1) and (no_of_variables >= 4):
            reformat_and_save_2d(
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
            )
            print("Finish 2D reformatting")


if __name__ == "__main__":
    args = parse_arguments()
    # config = load_config(args.yp)
    config.var_interp = args.var
    config.num_cores = args.nc
    config.memory = args.mp
    config.startyear_h = args.sy
    config.endyear_h = args.ey
    # config.num_processes = args.np
    # config.memory_per_process = args.mpp
    # config.num_cores_per_process = args.nc // args.np  # Distribute cores evenly
    main(config)
    print("All done!")
