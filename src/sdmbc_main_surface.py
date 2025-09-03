"""
SDMBCv2 Historical Bias Correction Main Script for Surface Variables

This script is used to perform bias correction on historical surface climate data
using the SDMBCv2 package. It uses Dask for parallel processing and handles
large-scale data processing, including loading, preprocessing, splitting domains
into tiles, and bias correction.

Key features of the script:
- Set up Dask distributed client for parallel computation.
- Load and preprocess GCM and observational datasets.
- Perform bias correction for each lat/lon tile.
- Save intermediate files and bias-corrected outputs.
- Optional drawing of figures and reformatting to original formats.

Usage:
This script can be executed by providing a configuration YAML file that defines
various input parameters, such as variable names, start and end years, paths, etc.

Modules imported:
- argparse: for handling command line arguments.
- numpy, xarray, Dask: for numerical computations and distributed processing.
- SDMBCv2 specific modules: for bias correction and data preparation functions.
"""

import argparse
import importlib
import os
import shutil
import time  # Import the time module
import warnings
from multiprocessing import Pool, cpu_count
from types import SimpleNamespace

import dask  # type: ignore
import numpy as np  # type: ignore

# import pandas as pd # type: ignore
import xarray as xr  # type: ignore
import yaml  # type: ignore
from dask.distributed import Client  # type: ignore
from tqdm import tqdm  # type: ignore

from bc_grid_function import (
    bc_correction_grid_cell_future_multiprocess,
    bc_correction_grid_cell_hist_dask_2d,
    convert_bc_params_to_xarray,
)

# from data_preparation import   # type: ignore
from data_preparation import (
    extract_and_reshape_delayed,
    generate_file_paths,
    generate_file_paths_obs,
    load_and_combine_variables,
    load_preprocess_variable,
    validate_inputs,
)

# from analysis_plot import AnalysisBC  # type: ignore


warnings.simplefilter("ignore", UserWarning)


# Start of the script -----------------------------------------------------
def setup_client(ncpus, mem_gb):
    """
    Dynamically set up a Dask client based on available CPUs and memory.
    """
    threads_per_worker = min(ncpus, 8)
    n_workers = max(1, ncpus // threads_per_worker)
    mem_per_worker = int(mem_gb / n_workers)

    print(f"[INFO] Starting Dask client: {n_workers} workers × {threads_per_worker} threads")
    print(f"[INFO] Each worker memory limit: {mem_per_worker}GB")

    return Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f"{mem_per_worker}GB",
    )



def parse_arguments():
    """
    Parse command-line arguments for configuration setup.

    Returns:
        argparse.Namespace: Parsed arguments including configuration file path and variables.
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
    """
    Split the spatial domain into smaller latitude and longitude tiles.

    Args:
        lat_min (float): Minimum latitude of the domain.
        lat_max (float): Maximum latitude of the domain.
        lon_min (float): Minimum longitude of the domain.
        lon_max (float): Maximum longitude of the domain.
        n_lat_tiles (int): Number of latitude tiles.
        n_lon_tiles (int): Number of longitude tiles.

    Returns:
        list: List of dictionaries containing the bounds of each tile.
    """
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


def main(config_path, override_ncpus=None, override_mem=None):
    """
    Main function to run the SDMBCv2 bias correction process.

    Args:
        config (module): Configuration object that contains all user-defined parameters for the process.
    """
    # setup_client()
    # Load YAML config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = SimpleNamespace(**config_dict)

    # Extract resources from config
    ncpus = getattr(config, "resources", {}).get("ncpus", 4)
    mem_gb = getattr(config, "resources", {}).get("mem_gb", 16)

    # Override from command-line if provided
    if override_ncpus:
        ncpus = override_ncpus
    if override_mem:
        mem_gb = override_mem

    # Dask performance settings
    dask.config.set(
        {"array.slicing.split_large_chunks": True, "array.chunk-size": "50MiB"}
    )

    # Start Dask client
    client = setup_client(ncpus, mem_gb)
    print(f"Using {ncpus} workers with {mem_gb} GB memory each.")

    startyear_h = config.startyear_h
    endyear_h = config.endyear_h
    no_of_variables = config.no_of_variables
    bc_boundary = config.bc_boundary
    lat_max = config.lat_max
    lat_min = config.lat_min
    lon_max = config.lon_max
    lon_min = config.lon_min
    obs_path = config.obs_path
    out_path = config.out_path
    input_model = config.input_model
    infor = config.infor
    gname = config.gname
    period = config.period
    cinfor = config.cinfor
    sinfor = config.sinfor
    version = config.version
    target_variable = config.target_variable
    upper_limit = config.upper_limit
    lower_limit = config.lower_limit
    out_figure_path = config.out_figure_path

    # Validate inputs
    if config.bc_future:
        startyear_f = config.startyear_f
        endyear_f = config.endyear_f
        period_f = config.period_f
        cinfor_f = config.cinfor_f
        scenario = config.scenario
        cinfor_f = config.cinfor_f
        target_variable = config.target_variable

    # Validate inputs
    validate_inputs(lat_min, lat_max)
    var_list_w = target_variable

    # if bc_hist:  # historical bias correction

    # List of variables to generate file paths for
    variables = target_variable

    # Define parameters for selection
    lat_range = (lat_min, lat_max)  # Adjust as needed
    lon_range = (lon_min, lon_max)  # Adjust as needed

    if config.bc_hist:

        b_path = obs_path

        # =============== Load GCM ===============
        # Generate file paths for each variable
        file_paths_by_variable_gcm = {}
        for variable in variables:
            file_paths_by_variable_gcm[variable] = generate_file_paths(
                config,
                b_path,
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
                    config,
                    obs_path,
                    variable,
                    gname,
                    startyear_h,
                    endyear_h,
                )
        else:
            for variable in variables:
                file_paths_by_variable_obs[variable] = generate_file_paths(
                    config,
                    b_path,
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

        # Record the start time for this level
        start_time = time.time()

        print(f"Starting processing for tos")
        level = 0
        # =============== Load GCM ===============
        sliced_gcm_all = []
        sliced_gcm = xr.Dataset()

        # Load each variable and adjust longitude for ua and va if necessary
        for var_name, file_paths in file_paths_by_variable_gcm.items():
            data_var = load_preprocess_variable(
                config,
                file_paths,
                var_name,
                level,
                lat_range,
                lon_range,
                startyear_h,
                endyear_h,
            )
            # Add the processed variable to the dataset
            sliced_gcm[var_name] = data_var
        # Chunk the data
        sliced_gcm = sliced_gcm.chunk({"time": 1000, "lat": -1, "lon": -1})
        sliced_gcm_all.append(sliced_gcm)
        daily_gcm = sliced_gcm_all[0]
        reshaped_gcm_delayed = extract_and_reshape_delayed(
            config, daily_gcm, no_of_variables, startyear_h, endyear_h, bc_boundary
        )
        reshaped_gcm_delayed += 273.15

        # =============== Load GCM end ===============

        # =============== Load Obs ===============
        sliced_obs_all = []
        sliced_obs = xr.Dataset()

        # Load each variable and adjust longitude for ua and va if necessary
        for var_name, file_paths in file_paths_by_variable_obs.items():
            obs_var = load_preprocess_variable(
                config,
                file_paths,
                var_name,
                level,
                lat_range,
                lon_range,
                startyear_h,
                endyear_h,
            )

            # Add the processed variable to the dataset
            sliced_obs[var_name] = obs_var
        sliced_obs = sliced_obs.chunk({"time": 1000, "lat": -1, "lon": -1})
        sliced_obs_all.append(sliced_obs)
        daily_obs = sliced_obs_all[0]
        reshaped_obs_delayed = extract_and_reshape_delayed(
            config, daily_obs, no_of_variables, startyear_h, endyear_h, bc_boundary
        )
        # =============== Load obs end ===============

        # Create a temporary folder for saving intermediate files
        temp_dir = os.path.join(out_path, "temp_tiles_surface")
        os.makedirs(temp_dir, exist_ok=True)

        print("start dask bc correction")
        # Perform bias correction across the entire grid (all lat/lon pairs)
        n_lat_tiles = 10
        n_lon_tiles = 10

        tiles = split_domain(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            n_lat_tiles=n_lat_tiles,
            n_lon_tiles=n_lon_tiles,
        )

        # To store all bc_params for later concatenation
        all_bc_params = []

        # for idx, tile in enumerate(tiles):
        for idx, tile in enumerate(tqdm(tiles, desc="Processing tiles")):
            try:
                lat_range = (tile["lat_min"], tile["lat_max"])
                lon_range = (tile["lon_min"], tile["lon_max"])

                # Slice GCM and Obs data for the tile
                reshaped_gcm_delayed_tile = reshaped_gcm_delayed.sel(
                    lat=slice(*lat_range), lon=slice(*lon_range)
                )
                reshaped_obs_delayed_tile = reshaped_obs_delayed.sel(
                    lat=slice(*lat_range), lon=slice(*lon_range)
                )

                bc_corrected_gcm_hist_tile, bc_params_tile = (
                    bc_correction_grid_cell_hist_dask_2d(
                        config,
                        reshaped_gcm_delayed_tile,
                        reshaped_obs_delayed_tile,
                        var_list_w,
                        config.startyear_h,
                    )
                )

                # Save each tile's bias-corrected output immediately to disk
                output_file = f"{temp_dir}/bc_corrected_tile_2d_{idx}_{lat_range[0]}_{lat_range[1]}_{lon_range[0]}_{lon_range[1]}.nc"
                output_params = f"{temp_dir}/bc_params_tile_2d_{idx}_{lat_range[0]}_{lat_range[1]}_{lon_range[0]}_{lon_range[1]}.nc"
                # Save the bias-corrected data for the tile
                if not os.path.exists(output_file):
                    print(f"Saving 2D bias-corrected for tile {idx} to {output_file}")
                    bc_corrected_gcm_hist_tile.compute().to_netcdf(
                        output_file
                    )  # Save tile result
                else:
                    print(f"File {output_file} already exists. Skipping...")

                # Save the bc_params for each tile
                obs_tile = sliced_obs.sel(lat=slice(*lat_range), lon=slice(*lon_range))
                tile_lat = obs_tile["lat"].values
                tile_lon = obs_tile["lon"].values
                ds_params = convert_bc_params_to_xarray(
                    config, bc_params_tile, tile_lat, tile_lon
                )

                if not os.path.exists(output_params):
                    print(f"Saving 2D params for tile {idx} to {output_params}")
                    ds_params.to_netcdf(output_params)
                else:
                    print(f"File {output_params} already exists. Skipping...")

                # Accumulate bc_params_tile for later concatenation
                all_bc_params.append(output_params)

                # Free memory after saving each tile
                del bc_corrected_gcm_hist_tile, ds_params
                # print(f"Processed and saved tile {idx}, lat range: {lat_range}, lon range: {lon_range}")

            except Exception as e:
                print(f"Error processing tile {idx}: {e}")
                continue

        # # Concatenate all bc_params along the latitude and longitude
        # # Initialize an empty list to hold rows of tiles for each latitude band
        # lat_band_tiles = []

        # # Step 2: Iterate over the tiles and organize by rows
        # for i in range(0, len(all_bc_params), n_lon_tiles):
        #     # Extract a row of tiles (all tiles in the same latitude band)
        #     row_tiles = all_bc_params[i : i + n_lon_tiles]

        #     # Concatenate the row of tiles along the longitude (axis=1)
        #     lat_band = np.concatenate(row_tiles, axis=1)

        #     # Add the concatenated latitude band to the list
        #     lat_band_tiles.append(lat_band)

        # # Concatenate all latitude bands along the latitude (axis=0)
        # full_param_array = np.concatenate(lat_band_tiles, axis=0)

        # Load the bias-corrected tiles and combine them into a single dataset
        tile_files = [
            f"{temp_dir}/bc_corrected_tile_2d_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
            for idx, tile in enumerate(tiles)
        ]
        full_bc_corrected = xr.open_mfdataset(
            tile_files, combine="by_coords"
        )  # Combine by matching coordinates

        # # Extract the original latitude and longitude values (with duplicates)
        # original_lat_values = full_bc_corrected["lat"].values
        # original_lon_values = full_bc_corrected["lon"].values

        # # Identify indices of duplicate values and determine which to remove
        # # Identify duplicated latitudes and keep only the first occurrence
        # _, lat_unique_indices = np.unique(original_lat_values, return_index=True)
        # # Get all indices, and identify which ones are to be removed (i.e., not in the unique set)
        # lat_indices_to_remove = np.setdiff1d(
        #     np.arange(len(original_lat_values)), lat_unique_indices
        # )

        # # Identify duplicated longitudes and keep only the first occurrence
        # _, lon_unique_indices = np.unique(original_lon_values, return_index=True)
        # # Get all indices, and identify which ones are to be removed (i.e., not in the unique set)
        # lon_indices_to_remove = np.setdiff1d(
        #     np.arange(len(original_lon_values)), lon_unique_indices
        # )

        # # Remove duplicate rows and columns from the full parameter array
        # # Assuming full_param_array has shape (146, 193) that includes duplicated values
        # full_param_array_corrected = np.delete(
        #     full_param_array, lat_indices_to_remove, axis=0
        # )  # Remove the duplicate latitude rows
        # full_param_array_corrected = np.delete(
        #     full_param_array_corrected, lon_indices_to_remove, axis=1
        # )  # Remove the duplicate longitude columns

        # Remove duplicate rows and columns from the full_bc_corrected dataset
        full_bc_corrected = full_bc_corrected.drop_duplicates("lat").drop_duplicates(
            "lon"
        )

        # Define the list of variables to be adjusted based on the limits
        # Assume order corresponds to limits in `config.yaml`
        variables_to_limit = ["w", "ta", "hus", "tos"]

        for var_name, lower, upper in zip(variables_to_limit, lower_limit, upper_limit):
            if var_name in full_bc_corrected:
                # Apply the limits to the variable by masking values outside of the range
                full_bc_corrected[var_name] = full_bc_corrected[var_name].where(
                    (full_bc_corrected[var_name] >= lower)
                    & (full_bc_corrected[var_name] <= upper),
                    np.nan,
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

        full_bc_corrected = full_bc_corrected.transpose("time", "lat", "lon")

        print("save the bc model")
        full_bc_corrected = full_bc_corrected.astype("float32")
        # # Save the BC model
        # np.save(
        #     f"{out_path}/bc_params_2d_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
        #     full_param_array_corrected,
        # )
        # Merge all tile bc_params NetCDF files into a single xarray Dataset for the whole domain.
        # This uses xarray's open_mfdataset which combines datasets by matching coordinate values.
        ds_full_params = xr.open_mfdataset(all_bc_params, combine="by_coords")

        # If for any reason duplicate lat or lon values exist, remove duplicates.
        # xarray's merge usually handles non-overlapping coordinates, but if needed:
        ds_full_params = ds_full_params.drop_duplicates("lat").drop_duplicates("lon")

        # Save the merged full-domain bias-correction parameters as a single NetCDF file.
        full_params_output = f"{out_path}/bc_params_2d_{period}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.nc"
        ds_full_params.to_netcdf(full_params_output)
        print("Saved full domain bc_params to", full_params_output)

        if config.save_bc_output:
            # save the bias corrected data # from input gcm or obs to target gcm
            full_bc_corrected.load().to_netcdf(
                f"{out_path}/bc_corrected_2d_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
            )

        del full_bc_corrected, ds_full_params
        # # Remove the temporary directory and its contents
        # shutil.rmtree(temp_dir)
        # print("Intermediate files deleted.")

        # # Log the time taken for this level
        # end_time = time.time()
        # elapsed_time_minutes = (
        #     end_time - start_time
        # ) / 60  # Convert seconds to minutes
        # print(f"Completed processing in {elapsed_time_minutes:.2f} minutes")

        # if config.draw_figure:
        #     print("Figure 2d field")
        #     figure_surface(
        #         out_path,
        #         obs_path,
        #         startyear_h,
        #         endyear_h,
        #         lat_range,
        #         lon_range,
        #         save_figure_surface,
        #         out_figure_path,
        #     )
        #     print("Finish 2d field")

        if config.reformat_to_original:
            print("Start reformatting")
            # Dynamically construct the module folder path relative to the main script
            # module_folder = os.path.join(os.path.dirname(__file__), "modules")
            module_name = (
                "reformat_gcm2origin"  # The Python file name without the .py extension
            )

            # Add the module folder to sys.path if it's not already there
            # if module_folder not in os.sys.path:
            #     os.sys.path.append(module_folder)

            # Import the module dynamically and run
            try:
                reformat_module = importlib.import_module(module_name)
                reformat_module.main()  # Assuming the .py file has a main() function
            except Exception as e:
                print(f"Failed to run the module '{module_name}': {e}")

    if (
        config.bc_future
    ):  # ========================= future bias correction =========================
        # Record the start time for this level
        start_time = time.time()
        level = 0
        domain = split_domain(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            n_lat_tiles=1,
            n_lon_tiles=1,
        )

        sliced_gcm_future = load_and_combine_variables(
            config,
            domain[0],
            variables,
            level,
            startyear_f,
            endyear_f,
            data_type=config.period_f,
        )
        sliced_gcm = sliced_gcm_future.sel(
            time=slice(f"{config.startyear_f}-01-01", f"{config.endyear_f}-12-31")
        )
        reshaped_gcm_delayed_f = extract_and_reshape_delayed(
            config, sliced_gcm, 1, startyear_f, endyear_f, bc_boundary
        )
        reshaped_gcm_delayed_f += 273.15

        # bc_params_array_loaded = np.load(
        #     f"{config.out_path}/bc_params_2d_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
        #     allow_pickle=True,
        # )

        # # Convert a dictionary to SimpleNamespace for easier attribute access

        # # Assuming the shape is (1, 3) and contains dictionaries
        # for i in range(bc_params_array_loaded.shape[0]):
        #     for j in range(bc_params_array_loaded.shape[1]):
        #         bc_params_array_loaded[i, j] = dict_to_simplenamespace(
        #             bc_params_array_loaded[i, j]
        #         )
        # Create a temporary folder for saving intermediate files
        temp_dir = os.path.join(out_path, "temp_tiles_surface")
        os.makedirs(temp_dir, exist_ok=True)

        output_params = f"{out_path}/bc_params_2d_{period}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.nc"
        bc_params_array_loaded = xr.load_dataset(output_params)

        # reshaped_gcm_delayed_f = reshaped_gcm_delayed_f.astype("float32")

        # n_lat_tiles, n_lon_tiles = determine_tiles(
        #     reshaped_gcm_delayed_f, lat_min, lat_max, lon_min, lon_max
        # )
        # print(f"Generated {n_lat_tiles} x {n_lon_tiles} tiles.")

        # print("start dask bc correction")

        # tiles = split_domain(
        #     lat_min,
        #     lat_max,
        #     lon_min,
        #     lon_max,
        #     n_lat_tiles=n_lat_tiles,
        #     n_lon_tiles=n_lon_tiles,
        # )

        # # for idx, tile in enumerate(tiles):
        # for idx, tile in enumerate(tqdm(tiles, desc="Processing tiles")):
        #     try:
        #         lat_range = (tile["lat_min"], tile["lat_max"])
        #         lon_range = (tile["lon_min"], tile["lon_max"])

        #         # Slice GCM and Obs data for the tile
        #         reshaped_gcm_delayed_tile = reshaped_gcm_delayed_f.sel(
        #             lat=slice(*lat_range), lon=slice(*lon_range)
        #         )
        #         # print("reshaped_tile")
        #         # print(reshaped_gcm_delayed_tile)

        #         bc_corrected_gcm_future_tile = bc_correction_grid_cell_future_dask_2d(
        #             reshaped_gcm_delayed_tile,
        #             bc_params_array_loaded,
        #             startyear_f,
        #             endyear_f,
        #             var_list_w,
        #         )
        #         # print("bc done and ready to save")
        #         # Save each tile's bias-corrected output immediately to disk
        #         output_file = f"{temp_dir}/bc_corrected_future_tile_2d_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
        #         bc_corrected_gcm_future_tile.compute().to_netcdf(
        #             output_file
        #         )  # Save tile result

        #         # Free memory after saving each tile
        #         del bc_corrected_gcm_future_tile
        #         # print(f"Processed and saved tile {idx}, lat range: {lat_range}, lon range: {lon_range}")

        #     except Exception as e:
        #         print(f"Error processing tile {idx}: {e}")
        #         continue
        print("Start multiprocessing for bias correction...")

        bc_corrected_gcm_future = bc_correction_grid_cell_future_multiprocess(
            config,
            reshaped_gcm_delayed_f,
            bc_params_array_loaded,
            var_list_w,
        )

        # Save results to NetCDF
        output_file = f"{temp_dir}/bc_corrected_{config.period_f}_2d_{gname}_{startyear_f}_{endyear_f}.nc"
        bc_corrected_gcm_future.to_netcdf(output_file)

        # Concatenate all bc_params along the latitude and longitude
        # Initialize an empty list to hold rows of tiles for each latitude band
        # lat_band_tiles = []

        # Load the bias-corrected tiles and combine them into a single dataset
        # tile_files = [
        #     f"{temp_dir}/bc_corrected_future_tile_2d_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
        #     for idx, tile in enumerate(tiles)
        # ]
        # full_bc_corrected = xr.open_mfdataset(
        #     tile_files, combine="by_coords"
        # )  # Combine by matching coordinates
        full_bc_corrected = xr.open_dataset(output_file)

        # # Extract the original latitude and longitude values (with duplicates)
        # original_lat_values = full_bc_corrected["lat"].values
        # original_lon_values = full_bc_corrected["lon"].values

        # # Identify indices of duplicate values and determine which to remove
        # # Identify duplicated latitudes and keep only the first occurrence
        # _, lat_unique_indices = np.unique(original_lat_values, return_index=True)
        # # Get all indices, and identify which ones are to be removed (i.e., not in the unique set)
        # lat_indices_to_remove = np.setdiff1d(
        #     np.arange(len(original_lat_values)), lat_unique_indices
        # )

        # # Identify duplicated longitudes and keep only the first occurrence
        # _, lon_unique_indices = np.unique(original_lon_values, return_index=True)
        # # Get all indices, and identify which ones are to be removed (i.e., not in the unique set)
        # lon_indices_to_remove = np.setdiff1d(
        #     np.arange(len(original_lon_values)), lon_unique_indices
        # )

        # Remove duplicate rows and columns from the full_bc_corrected dataset
        full_bc_corrected = full_bc_corrected.drop_duplicates("lat").drop_duplicates(
            "lon"
        )

        # Define the list of variables to be adjusted based on the limits
        # Assume order corresponds to limits in `config.yaml`
        variables_to_limit = ["w", "ta", "hus", "tos"]

        for var_name, lower, upper in zip(variables_to_limit, lower_limit, upper_limit):
            if var_name in full_bc_corrected:
                # Apply the limits to the variable by masking values outside of the range
                full_bc_corrected[var_name] = full_bc_corrected[var_name].where(
                    (full_bc_corrected[var_name] >= lower)
                    & (full_bc_corrected[var_name] <= upper),
                    np.nan,
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

        full_bc_corrected = full_bc_corrected.transpose("time", "lat", "lon")

        print("save the bc model")
        full_bc_corrected = full_bc_corrected.astype("float32")

        if config.save_bc_output:
            full_bc_corrected.load().to_netcdf(
                f"{out_path}/bc_corrected_2d_{infor}_{gname}_{period_f}_{cinfor}_{sinfor}_{startyear_f}_{endyear_f}.nc"
            )

            # save the bias corrected data # from input gcm or obs to target gcm
            # if config.period == "validation":
            #     full_bc_corrected.load().to_netcdf(
            #         f"{out_path}/bc_corrected_2d_{config.period_f}}_{infor}_{gname}_{period_f}_{cinfor}_{sinfor}_{startyear_f}_{endyear_f}.nc"
            #     )
            # elif config.period == "future":
            #     full_bc_corrected.load().to_netcdf(
            #         f"{out_path}/bc_corrected_2d_{config.period_f}_{infor}_{gname}_{period_f}_{cinfor_f}_{scenario}_{startyear_f}_{endyear_f}.nc"
            #     )

        # Remove the temporary directory and its contents
        shutil.rmtree(temp_dir)
        print("Intermediate files deleted.")

        # Log the time taken for this level
        end_time = time.time()
        elapsed_time_minutes = (
            end_time - start_time
        ) / 60  # Convert seconds to minutes
        print(f"Completed processing in {elapsed_time_minutes:.2f} minutes")

        # if config.draw_figure:
        #     print("Figure 2d field")
        #     figure_surface(
        #         out_path,
        #         obs_path,
        #         startyear_f,
        #         endyear_f,
        #         lat_range,
        #         lon_range,
        #         save_figure_surface,
        #         out_figure_path,
        #     )
        #     print("Finish 2d field")

        if config.reformat_to_original:
            print("Start reformatting")
            # Dynamically construct the module folder path relative to the main script
            # module_folder = os.path.join(os.path.dirname(__file__), "modules")
            module_name = (
                "reformat_gcm2origin"  # The Python file name without the .py extension
            )

            # Add the module folder to sys.path if it's not already there
            # if module_folder not in os.sys.path:
            #     os.sys.path.append(module_folder)

            # Import the module dynamically and run
            try:
                reformat_module = importlib.import_module(module_name)
                reformat_module.main()  # Assuming the .py file has a main() function
            except Exception as e:
                print(f"Failed to run the module '{module_name}': {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--yp", required=True, help="Path to YAML config")
    parser.add_argument(
        "--ncpus", type=int, default=None, help="Override number of CPUs"
    )
    parser.add_argument("--mem", type=int, default=None, help="Override memory in GB")

    args = parser.parse_args()
    main(args.yp, args.ncpus, args.mem)

    print("All done!")
