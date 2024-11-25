import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

sys.path.append("/g/data/w28/yk8692/sdmbc_v2/modules")

# import dask  # type: ignore
import numpy as np  # type: ignore

# import pandas as pd # type: ignore
import xarray as xr  # type: ignore
from analysis_plot import AnalysisBC  # type: ignore

# import yaml  # type: ignore
from bc_grid_function import bc_correction_grid_cell_hist  # type: ignore
from config import config  # type: ignore
from dask.distributed import Client  # type: ignore
from data_preparation import (
    assign_w_6hr,
    assign_w_day,
    convert_to_daily_with_fraction,
    extract_and_reshape_delayed,
    generate_file_paths,
    generate_file_paths_obs,
    load_preprocess_variable,
    validate_inputs,
)
from output_reformatter import reformat_and_save_2d  # type: ignore
from output_reformatter import reformat_and_save_3d  # type: ignore

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


def process_cell(
    lat_lon,
    reshaped_gcm_delayed,
    reshaped_obs_delayed,
    fraction_factors_gcm,
    fraction_factors_obs,
    var_list_w,
    sliced_gcm,
):
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


def process_cell_wrapper(
    lat_lon,
    reshaped_gcm_delayed,
    reshaped_obs_delayed,
    fraction_factors_gcm,
    fraction_factors_obs,
    var_list_w,
    sliced_gcm,
):
    return process_cell(
        lat_lon,
        reshaped_gcm_delayed,
        reshaped_obs_delayed,
        fraction_factors_gcm,
        fraction_factors_obs,
        var_list_w,
        sliced_gcm,
    )


def main(config):

    setup_client()

    startyear_h = config.startyear_h
    endyear_h = config.endyear_h
    no_of_variables = config.no_of_variables
    startyear_h = config.startyear_h
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
    input_variables = config.target_variable

    # Validate inputs
    validate_inputs(lat_min, lat_max)
    var_list_w = ["w", "ta", "hus"] if bc_boundary == "lateral" else ["tos"]

    # if bc_hist:  # historical bias correction

    # List of variables to generate file paths for
    variables = input_variables

    # Define parameters for selection
    lat_range = (lat_min, lat_max)  # Adjust as needed
    lon_range = (lon_min, lon_max)  # Adjust as needed

    # =============== Load GCM ===============
    # Generate file paths for each variable
    file_paths_by_variable_gcm = {}
    for variable in variables:
        file_paths_by_variable_gcm[variable] = generate_file_paths(
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
    # =============== Load GCM end ===============

    # =============== Load Obs ===============
    # config should also consider the input and target path
    file_paths_by_variable_obs = {}
    if input_model == "reanalysis":
        for variable in variables:
            file_paths_by_variable_obs[variable] = generate_file_paths_obs(
                obs_path,
                variable,
                gname,
                startyear_h,
                endyear_h,
            )
    else:
        for variable in variables:
            file_paths_by_variable_obs[variable] = generate_file_paths(
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
    # =============== Load obs end ===============

    for level in range(0, tlevel):

        # =============== Load GCM ===============
        sliced_gcm_all = []
        sliced_gcm = xr.Dataset()

        # Load each variable and adjust longitude for ua and va if necessary
        for var_name, file_paths in file_paths_by_variable_gcm.items():
            data_var = load_preprocess_variable(
                file_paths, var_name, level, lat_range, lon_range
            )
            # Check if the variable is one of the wind components with different lon
            if var_name in ["ua", "va"]:
                # Let's assume hus and ta have the target longitude values, and they are already loaded
                target_lon = sliced_gcm.lon if "lon" in sliced_gcm else data_var.lon
                target_lat = sliced_gcm.lat if "lat" in sliced_gcm else data_var.lat
                target_lev = sliced_gcm.lev if "lev" in sliced_gcm else data_var.lev

                # Check if va coordinates are different before interpolation
                # if var_name == 'va':
                #     if not data_var.lat.equals(target_lat):
                #         # data_var = data_var.interp(lat=target_lat)
                #         # Interpolate using nearest method
                #         data_var = data_var.interp(lat=target_lat, method='linear', kwargs={"fill_value": "extrapolate"})
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
            # Add the processed variable to the dataset
            sliced_gcm[var_name] = data_var

        sliced_gcm_all.append(sliced_gcm)
        # Rename variables
        # rename_dict = {"hus": "q", "ta": "t", "ua": "u", "va": "v"}
        sliced_gcm = sliced_gcm_all[0].sel(
            time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31")
        )
        assign_gcm = assign_w_6hr(sliced_gcm, bc_boundary)
        daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(assign_gcm)
        daily_gcm_rechunk = daily_gcm.chunk({"time": -1, "lat": "auto", "lon": "auto"})
        print("start delayed process gcm")
        print("daily_gcm_rechunk", daily_gcm_rechunk)
        reshaped_gcm_delayed = extract_and_reshape_delayed(
            daily_gcm_rechunk, no_of_variables, startyear_h, endyear_h
        )

        # =============== Load GCM end ===============

        # =============== Load Obs ===============
        sliced_obs_all = []
        sliced_obs = xr.Dataset()

        # Load each variable and adjust longitude for ua and va if necessary
        for var_name, file_paths in file_paths_by_variable_obs.items():
            obs_var = load_preprocess_variable(
                file_paths, var_name, level, lat_range, lon_range
            )
            # # Check if the variable is one of the wind components with different lon
            # if var_name in ["ua", "va"]:
            #     # Let's assume hus and ta have the target longitude values, and they are already loaded
            #     target_lon = sliced_obs.lon if "lon" in sliced_obs else obs_var.lon
            #     target_lat = sliced_obs.lat if "lat" in sliced_obs else obs_var.lat
            #     target_lev = sliced_obs.lev if "lev" in sliced_obs else obs_var.lev

            #     # Interpolate va to match the target latitude grid
            #     if (
            #         var_name == "va"
            #         and "lat" in data_var.coords
            #         and "lat" in target_lat.coords
            #     ):
            #         obs_var = obs_var.interp(lat=target_lat)

            #         # Assign the adjusted longitude values to ua or va
            #         obs_var = obs_var.assign_coords(
            #             lon=target_lon, lat=target_lat, lev=target_lev
            #         )
            # Add the processed variable to the dataset
            sliced_obs[var_name] = obs_var

        sliced_obs_all.append(sliced_obs)
        # Rename variables
        sliced_obs = sliced_obs_all[0].sel(
            time=slice(f"{startyear_h}-01-01", f"{endyear_h}-12-31")
        )
        assign_obs = assign_w_6hr(sliced_obs, bc_boundary)
        daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(assign_obs)
        daily_obs_rechunk = daily_obs.chunk({"time": -1, "lat": "auto", "lon": "auto"})
        print("start delayed process obs")
        print("daily_obs_rechunk", daily_obs_rechunk)
        reshaped_obs_delayed = extract_and_reshape_delayed(
            daily_obs_rechunk, no_of_variables, startyear_h, endyear_h
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

        # Prepare lat/lon pairs for processing
        lat_lon_pairs = [(i, j) for i in range(dims_lat) for j in range(dims_lon)]

        # Use a process pool to parallelize
        with ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    process_cell_wrapper,
                    lat_lon_pairs,
                    [reshaped_gcm_delayed] * len(lat_lon_pairs),
                    [reshaped_obs_delayed] * len(lat_lon_pairs),
                    [fraction_factors_gcm] * len(lat_lon_pairs),
                    [fraction_factors_obs] * len(lat_lon_pairs),
                    [var_list_w] * len(lat_lon_pairs),
                    [sliced_gcm] * len(lat_lon_pairs),
                )
            )

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
        if len(input_variables) > 1:
            np.save(
                f"{out_path}/bc_params_3d_level_{level}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
                bc_params_array,
            )
        else:
            np.save(
                f"{out_path}/bc_params_2d_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
                bc_params_array,
            )

        corrected_data_list = list(corrected_data_dict.values())

        # Combine the DataArrays into a single Dataset
        ds_corrected = xr.combine_by_coords(corrected_data_list)
        ds_corrected = ds_corrected.chunk({"time": -1, "lat": "auto", "lon": "auto"})
        # Set attributes
        ds_corrected.attrs["history"] = "Bias-corrected data, SDMBCv2"
        ds_corrected.attrs["source"] = f"{gname} bias-corrected to {input_model}"

        if config.save_bc_output:
            # save the bias corrected data # from input gcm or obs to target gcm
            if len(input_variables) > 1:
                ds_corrected.to_netcdf(
                    f"{out_path}/bc_corrected_3d_level_{level}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
                )
            else:
                ds_corrected.to_netcdf(
                    f"{out_path}/bc_corrected_2d_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
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
                    variables,
                    config.out_figure_path,
                    kstest=True,
                )
            if config.sub_daily_correction == False:
                print("K-S test has not been included")
                statistics = AnalysisBC(
                    daily_gcm,
                    daily_obs,
                    ds_corrected_day,
                    variables,
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
        if len(input_variables) > 1:
            reformat_and_save_3d(
                out_path,
                tlevel,
                startyear_h,
                endyear_h,
                variables,
                variables,
                out_path,
                infor,
                gname,
                period,
                cinfor,
                sinfor,
            )
            print("Finish 3D reformatting")

        else:
            reformat_and_save_2d(
                out_path,
                startyear_h,
                endyear_h,
                variables,
                variables,
                out_path,
                infor,
                gname,
                period,
                cinfor,
                sinfor,
                startyear_h,
                endyear_h,
            )
            print("Finish 2D reformatting")


if __name__ == "__main__":
    # args = parse_arguments()
    # config = load_config(args.yp)
    # config.var_interp = args.var
    # config.startyear_h = args.sy
    # config.endyear_h = args.ey
    # config.num_processes = args.np
    # config.memory_per_process = args.mpp
    # config.num_cores_per_process = args.nc // args.np  # Distribute cores evenly
    main(config)
    print("All done!")
