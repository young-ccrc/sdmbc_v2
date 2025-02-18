import argparse
import logging
import math
import os
import shutil
import sys
import time  # Import the time module
import warnings
from types import SimpleNamespace

# import dask  # type: ignore
import numpy as np  # type: ignore

# import pandas as pd # type: ignore
import xarray as xr  # type: ignore
from dask.distributed import Client  # type: ignore
from tqdm import tqdm  # type: ignore

# import yaml  # type: ignore
from bc_grid_function import load_preprocess_variable  # type: ignore
from bc_grid_function import (
    preprocess_and_save_gcm,
    preprocess_and_save_obs,
    process_tile,
    process_tile_future,
)
from config import config  # type: ignore

# from data_preparation import   # type: ignore
from data_preparation import (
    determine_tiles,
    generate_file_paths,
    generate_file_paths_future,
    generate_file_paths_obs,
    validate_inputs,
)
from figurefunction import save_figure_3d

# Suppress INFO and lower-level logs
logging.getLogger("flox").setLevel(logging.WARNING)
logging.getLogger("xarray").setLevel(logging.WARNING)
logging.getLogger("dask").setLevel(logging.WARNING)

# sys.path.append("/g/data/w28/yk8692/sdmbc_v2/sdmbc_v2")


# from analysis_plot import AnalysisBC  # type: ignore


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


def split_domain(lat_min, lat_max, lon_min, lon_max, n_lat_tiles, n_lon_tiles):
    # Create ranges for latitude and longitude tiles
    lat_ranges = np.linspace(lat_min, lat_max, n_lat_tiles + 1, endpoint=True)
    lon_ranges = np.linspace(lon_min, lon_max, n_lon_tiles + 1, endpoint=True)

    tiles = []
    for i in range(n_lat_tiles):
        for j in range(n_lon_tiles):
            tiles.append(
                {
                    "lat_min": lat_ranges[i],
                    "lat_max": lat_ranges[i + 1],
                    "lon_min": lon_ranges[j],
                    "lon_max": lon_ranges[j + 1],
                }
            )
    return tiles


def dict_to_simplenamespace(d):
    return SimpleNamespace(**d)


def main(config):

    setup_client()

    startyear_h = config.startyear_h
    endyear_h = config.endyear_h
    startyear_h = config.startyear_h
    bc_boundary = config.bc_boundary
    bc_hist_path = config.bc_hist_path
    lat_max = config.lat_max
    lat_min = config.lat_min
    lon_max = config.lon_max
    lon_min = config.lon_min
    slevel = config.slevel
    elevel = config.elevel
    obs_path = config.obs_path
    out_path = config.out_path
    input_model = config.input_model
    infor = config.infor
    gname = config.gname
    period = config.period
    cinfor = config.cinfor
    sinfor = config.sinfor
    version = config.version
    upper_limit = config.upper_limit
    lower_limit = config.lower_limit

    # Validate inputs
    if config.bc_future:
        bc_future_path = config.bc_future_path
        period_f = config.period_f
        startyear_f = config.startyear_f
        endyear_f = config.endyear_f

    # Validate inputs
    validate_inputs(lat_min, lat_max)

    # List of variables to generate file paths for
    variables = config.target_variable

    # =============== Load GCM ===============
    # Generate file paths for each variable
    file_paths_by_variable_gcm = {
        var: generate_file_paths(
            bc_hist_path,
            var,
            infor,
            gname,
            period,
            cinfor,
            sinfor,
            version,
            startyear_h,
            endyear_h,
        )
        for var in variables
    }

    # =============== Load GCM end ===============

    # =============== Load Obs ===============
    # config should also consider the input and target path
    file_paths_by_variable_obs = {
        var: (
            generate_file_paths_obs(obs_path, var, gname, startyear_h, endyear_h)
            if config.input_model == "reanalysis"
            else generate_file_paths(
                bc_hist_path,
                var,
                infor,
                gname,
                period,
                cinfor,
                sinfor,
                version,
                startyear_h,
                endyear_h,
            )
        )
        for var in variables
    }

    # =============== Load obs end ===============
    # Dynamically determine number of tiles

    n_lat_tiles, n_lon_tiles = determine_tiles(
        file_paths_by_variable_obs, variables[0], lat_min, lat_max, lon_min, lon_max
    )

    print("start dask bc correction")

    tiles = split_domain(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        n_lat_tiles=n_lat_tiles,
        n_lon_tiles=n_lon_tiles,
    )

    domain = split_domain(
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        n_lat_tiles=1,
        n_lon_tiles=1,
    )

    for level in range(slevel, elevel + 1):
        # Record the start time for this level
        start_time = time.time()
        lat_range = (lat_min, lat_max)
        lon_range = (lon_min, lon_max)
        print(f"Starting processing for level {level}")

        temp_dir = os.path.join(out_path, f"temp_tiles_{gname}_{level}")
        os.makedirs(temp_dir, exist_ok=True)
        # ------------------ Load GCM Data ------------------
        temp_gcm = preprocess_and_save_gcm(
            domain,
            file_paths_by_variable_gcm,
            temp_dir,
            level,
        )
        sliced_gcm = xr.open_dataset(
            f"{temp_dir}/preprocessed_{gname}_lev_{level}_{lat_min}_{lat_max}_{lon_min}_{lon_max}.nc"
        )

        # ------------------ Load GCM Data end ------------------

        # ------------------ Load Obs Data ------------------
        for var_name, file_paths in file_paths_by_variable_obs.items():
            temp_netcdf = preprocess_and_save_obs(
                domain,
                file_paths,
                temp_dir,
                var_name,
                level,
                startyear_h,
                endyear_h,
            )
        sliced_obs = xr.Dataset()
        for var_name in variables:
            # Construct the path to the preprocessed file for this variable
            obs_file = os.path.join(
                temp_dir,
                f"preprocessed_obs_{var_name}_lev_{level}_{float(lat_min)}_{float(lat_max)}_{float(lon_min)}_{float(lon_max)}.nc",
            )
            sliced_obs[var_name] = xr.open_dataset(obs_file)[
                var_name
            ]  # Load the variable from the file

        sliced_obs = sliced_obs.astype("float32")
        # ------------------ Load Obs Data end ------------------

        if config.bc_hist:

            # To store all bc_params for later concatenation
            all_bc_params = []
            # # for idx, tile in enumerate(tiles):
            for idx, tile in enumerate(tqdm(tiles, desc="Processing tiles")):
                #     for var_name, file_paths in file_paths_by_variable_obs.items():
                #         temp_netcdf = preprocess_and_save_obs(
                #             tile,
                #             file_paths,
                #             temp_dir,
                #             var_name,
                #             level,
                #             lat_min,
                #             lat_max,
                #             lon_min,
                #             lon_max,
                #             startyear_h,
                #             endyear_h,
                #         )
                lat_range = (tile["lat_min"], tile["lat_max"])
                lon_range = (tile["lon_min"], tile["lon_max"])
                try:
                    # Define output file paths
                    output_file = f"{temp_dir}/bc_corrected_tile_3d_{period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
                    output_params = f"{temp_dir}/bc_params_tile_3d_{period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.npy"

                    obs_tile = sliced_obs.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )
                    gcm_tile = sliced_gcm.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )
                    # # Check if both output files already exist
                    # if os.path.exists(output_file) and os.path.exists(output_params):
                    #     print(
                    #         f"Both output file and params for tile {idx}, level {level} already exist. Skipping..."
                    #     )
                    #     continue  # Skip processing this tile

                    # Process the tile if either output is missing
                    print(f"Processing tile {idx}, level {level}...")
                    bc_corrected_gcm_hist_tile, bc_params_tile = process_tile(
                        gcm_tile,
                        obs_tile,
                        config,
                    )

                    # Save the bias-corrected output
                    if not os.path.exists(output_file):
                        print(
                            f"Saving 3D output for tile {idx}, level {level} to {output_file}"
                        )
                        bc_corrected_gcm_hist_tile.compute().to_netcdf(output_file)
                    else:
                        print(f"File {output_file} already exists. Skipping...")

                    # Save the bias-correction parameters
                    if not os.path.exists(output_params):
                        print(
                            f"Saving 3D params for tile {idx}, level {level} to {output_params}"
                        )
                        np.save(output_params, bc_params_tile)
                    else:
                        print(f"File {output_params} already exists. Skipping...")
                        # continue

                    # Accumulate bc_params_tile for later concatenation
                    all_bc_params.append(bc_params_tile)

                    # Free memory after saving each tile
                    del bc_corrected_gcm_hist_tile
                    # print(f"Processed and saved tile {idx}, lat range: {lat_range}, lon range: {lon_range}")

                except Exception as e:
                    print(f"Error processing tile {idx}: {e}")
                    # continue

            # Concatenate all bc_params along the latitude and longitude
            # Initialize an empty list to hold rows of tiles for each latitude band
            lat_band_tiles = []

            # Step 2: Iterate over the tiles and organize by rows
            for i in range(0, len(all_bc_params), n_lon_tiles):
                # Extract a row of tiles (all tiles in the same latitude band)
                row_tiles = all_bc_params[i : i + n_lon_tiles]

                # Concatenate the row of tiles along the longitude (axis=1)
                lat_band = np.concatenate(row_tiles, axis=1)

                # Add the concatenated latitude band to the list
                lat_band_tiles.append(lat_band)

            # Concatenate all latitude bands along the latitude (axis=0)
            full_param_array = np.concatenate(lat_band_tiles, axis=0)

            # Load the bias-corrected tiles and combine them into a single dataset
            tile_files = [
                f"{temp_dir}/bc_corrected_tile_3d_{period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
                for idx, tile in enumerate(tiles)
            ]
            full_bc_corrected = xr.open_mfdataset(
                tile_files, combine="by_coords"
            )  # Combine by matching coordinates

            # Extract the original latitude and longitude values (with duplicates)
            original_lat_values = full_bc_corrected["lat"].values
            original_lon_values = full_bc_corrected["lon"].values

            # Identify indices of duplicate values and determine which to remove
            # Identify duplicated latitudes and keep only the first occurrence
            _, lat_unique_indices = np.unique(original_lat_values, return_index=True)
            # Get all indices, and identify which ones are to be removed (i.e., not in the unique set)
            lat_indices_to_remove = np.setdiff1d(
                np.arange(len(original_lat_values)), lat_unique_indices
            )

            # Identify duplicated longitudes and keep only the first occurrence
            _, lon_unique_indices = np.unique(original_lon_values, return_index=True)
            # Get all indices, and identify which ones are to be removed (i.e., not in the unique set)
            lon_indices_to_remove = np.setdiff1d(
                np.arange(len(original_lon_values)), lon_unique_indices
            )

            # Remove duplicate rows and columns from the full parameter array
            # Assuming full_param_array has shape (146, 193) that includes duplicated values
            full_param_array_corrected = np.delete(
                full_param_array, lat_indices_to_remove, axis=0
            )  # Remove the duplicate latitude rows
            full_param_array_corrected = np.delete(
                full_param_array_corrected, lon_indices_to_remove, axis=1
            )  # Remove the duplicate longitude columns

            # Remove duplicate rows and columns from the full_bc_corrected dataset
            full_bc_corrected = full_bc_corrected.drop_duplicates(
                "lat"
            ).drop_duplicates("lon")

            # Define the list of variables to be adjusted based on the limits
            # Assume order corresponds to limits in `config.yaml`
            variables_to_limit = ["w", "ta", "hus", "tos"]

            for var_name, lower, upper in zip(
                variables_to_limit, lower_limit, upper_limit
            ):
                if var_name in full_bc_corrected:
                    if var_name == "hus":
                        lower = lower / 1000
                        upper = upper / 1000
                    # Apply the limits to the variable by masking values outside of the range
                    full_bc_corrected[var_name] = full_bc_corrected[var_name].where(
                        (full_bc_corrected[var_name] > lower),
                        lower,
                    )
                    full_bc_corrected[var_name] = full_bc_corrected[var_name].where(
                        (full_bc_corrected[var_name] < upper),
                        upper,
                    )

            full_bc_corrected["time"].attrs.update(
                {
                    "standard_name": "time",
                    "long_name": "time",
                }
            )

            # Check and correct the attributes of lat and lon
            full_bc_corrected["lat"].attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                }
            )

            full_bc_corrected["lon"].attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                }
            )

            # Ensure these attributes are properly set for each variable
            m_names = ["M", "MSD", "NBC", "MBC"]
            for var in full_bc_corrected.data_vars:
                full_bc_corrected[var].attrs.update(
                    {
                        "description": f"bias-corrected data, {m_names[int(config.correction_model)-1]}, SDMBCv2",
                        "history": "Created by applying SDMBCv2 package",
                    }
                )

            if "lev" not in full_bc_corrected.dims:
                full_bc_corrected = full_bc_corrected.expand_dims(
                    lev=[full_bc_corrected.lev.values]
                )
            full_bc_corrected = full_bc_corrected.transpose("time", "lev", "lat", "lon")

            print("save the bc model")
            full_bc_corrected = full_bc_corrected.astype("float32")  # save as float32

            # Save the BC model
            np.save(
                f"{out_path}/bc_params_3d_{period}_lev_{level}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
                full_param_array_corrected,
            )

            if config.save_bc_output:
                full_bc_corrected.load().to_netcdf(
                    f"{out_path}/bc_corrected_3d_lev_{level}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
                )

            del full_bc_corrected, full_param_array_corrected

        if config.bc_future:

            for idx, tile in enumerate(tqdm(tiles, desc="Processing tiles")):
                # try:
                # if config.bc_hist == False:
                #     for var_name, file_paths in file_paths_by_variable_obs.items():
                #         temp_netcdf = preprocess_and_save_obs(
                #             tile,
                #             file_paths,
                #             temp_dir,
                #             var_name,
                #             level,
                #             lat_min,
                #             lat_max,
                #             lon_min,
                #             lon_max,
                #             startyear_h,
                #             endyear_h,
                #         )
                # Define output file paths
                output_file = f"{temp_dir}/bc_corrected_tile_3d_{period_f}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
                lat_range = (tile["lat_min"], tile["lat_max"])
                lon_range = (tile["lon_min"], tile["lon_max"])

                if os.path.exists(output_file):
                    print(f"File {output_file} already exists. Skipping...")
                else:
                    obs_tile = sliced_obs.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )
                    gcm_tile = sliced_gcm.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )
                    bc_corrected_gcm_future_tile = process_tile_future(
                        tile,
                        variables,
                        level,
                        config,
                        gcm_tile,
                        obs_tile,
                    )

                # Save the bias-corrected output
                if not os.path.exists(output_file):
                    print(
                        f"Saving 3D output for tile {idx}, level {level} to {output_file}"
                    )
                    bc_corrected_gcm_future_tile.compute().to_netcdf(output_file)
                    del bc_corrected_gcm_future_tile
                # else:
                #     print(f"File {output_file} already exists. Skipping...")

                # Free memory after saving each tile

                # print(f"Processed and saved tile {idx}, lat range: {lat_range}, lon range: {lon_range}")

                # except Exception as e:
                #     print(f"Error processing tile {idx}: {e}")
                #     # continue

            # Load the bias-corrected tiles and combine them into a single dataset
            tile_files = [
                f"{temp_dir}/bc_corrected_tile_3d_{period_f}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
                for idx, tile in enumerate(tiles)
            ]
            full_bc_corrected = xr.open_mfdataset(
                tile_files, combine="by_coords"
            )  # Combine by matching coordinates

            # Remove duplicate rows and columns from the full_bc_corrected dataset
            full_bc_corrected = full_bc_corrected.drop_duplicates(
                "lat"
            ).drop_duplicates("lon")

            # Define the list of variables to be adjusted based on the limits
            # Assume order corresponds to limits in `config.yaml`
            variables_to_limit = ["w", "ta", "hus", "tos"]

            for var_name, lower, upper in zip(
                variables_to_limit, lower_limit, upper_limit
            ):
                if var_name in full_bc_corrected:
                    if var_name == "hus":
                        lower = lower / 1000
                        upper = upper / 1000
                    # Apply the limits to the variable by masking values outside of the range
                    full_bc_corrected[var_name] = full_bc_corrected[var_name].where(
                        (full_bc_corrected[var_name] > lower),
                        lower,
                    )
                    full_bc_corrected[var_name] = full_bc_corrected[var_name].where(
                        (full_bc_corrected[var_name] < upper),
                        upper,
                    )

            full_bc_corrected["time"].attrs.update(
                {
                    "standard_name": "time",
                    "long_name": "time",
                }
            )

            # Check and correct the attributes of lat and lon
            full_bc_corrected["lat"].attrs.update(
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "axis": "Y",
                }
            )

            full_bc_corrected["lon"].attrs.update(
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "axis": "X",
                }
            )

            # Ensure these attributes are properly set for each variable
            m_names = ["M", "MSD", "NBC", "MBC"]
            for var in full_bc_corrected.data_vars:
                full_bc_corrected[var].attrs.update(
                    {
                        "description": f"bias-corrected data, {m_names[int(config.correction_model)-1]}, SDMBCv2",
                        "history": "Created by applying SDMBCv2 package",
                    }
                )

            if "lev" not in full_bc_corrected.dims:
                full_bc_corrected = full_bc_corrected.expand_dims(
                    lev=[full_bc_corrected.lev[0].values]
                )
            full_bc_corrected = full_bc_corrected.transpose("time", "lev", "lat", "lon")

            print("save the bc model")
            full_bc_corrected = full_bc_corrected.astype("float32")  # save as float32

        if config.save_bc_output:
            # save the bias corrected data # from input gcm or obs to target gcm
            full_bc_corrected.load().to_netcdf(
                f"{out_path}/bc_corrected_3d_lev_{level}_{infor}_{gname}_{period_f}_{cinfor}_{sinfor}_{startyear_f}_{endyear_f}.nc"
            )

        # Remove the temporary directory and its contents
        shutil.rmtree(temp_dir)
        print("Intermediate files deleted.")

        # Log the time taken for this level
        end_time = time.time()
        elapsed_time_minutes = (
            end_time - start_time
        ) / 60  # Convert seconds to minutes
        print(f"Completed processing in {elapsed_time_minutes:.2f} minutes")

    if config.draw_figure:
        if bc_boundary == "lateral":
            print("Drawing figures")
            # print("K-S test has been included")
            # bottom level test for 3d field
            level = 0
            if config.bc_hist:
                save_figure_3d(
                    file_paths_by_variable_gcm, file_paths_by_variable_obs, level
                )
            else:
                save_figure_3d(file_paths_by_variable_gcm, out_path, level)
            print("Finish 3d field")


if __name__ == "__main__":
    main(config)
    print("All done!")
