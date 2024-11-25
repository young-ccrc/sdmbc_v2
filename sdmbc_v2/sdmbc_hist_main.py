import argparse
import os
import shutil
import sys
import time  # Import the time module
import warnings

# import dask  # type: ignore
import numpy as np  # type: ignore

# import pandas as pd # type: ignore
import xarray as xr  # type: ignore
from config import config  # type: ignore
from dask.distributed import Client  # type: ignore
from tqdm import tqdm  # type: ignore

# import yaml  # type: ignore
from sdmbc_v2.bc_grid_function import (  # type: ignore
    bc_correction_grid_cell_hist_dask,
    bc_correction_grid_cell_hist_dask_2d,
)

# from data_preparation import   # type: ignore
from sdmbc_v2.data_preparation import (
    assign_w_6hr,
    convert_to_daily_with_fraction,
    extract_and_reshape_delayed,
    generate_file_paths,
    generate_file_paths_obs,
    load_preprocess_variable,
    validate_inputs,
)

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
    # input_variables = config.target_variable
    upper_limit = config.upper_limit
    lower_limit = config.lower_limit

    # Validate inputs
    validate_inputs(lat_min, lat_max)
    var_list_w = ["w", "ta", "hus"] if bc_boundary == "lateral" else ["tos"]

    # if bc_hist:  # historical bias correction

    # List of variables to generate file paths for
    variables = config.target_variable

    # Define parameters for selection
    lat_range = (lat_min, lat_max)  # Adjust as needed
    lon_range = (lon_min, lon_max)  # Adjust as needed

    if bc_boundary == "lateral":
        b_path = bc_hist_path
    else:
        b_path = obs_path

    # =============== Load GCM ===============
    # Generate file paths for each variable
    file_paths_by_variable_gcm = {}
    for variable in variables:
        file_paths_by_variable_gcm[variable] = generate_file_paths(
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
                obs_path,
                variable,
                gname,
                startyear_h,
                endyear_h,
            )
    else:
        for variable in variables:
            file_paths_by_variable_obs[variable] = generate_file_paths(
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

    for level in range(0, tlevel):
        # Record the start time for this level
        start_time = time.time()

        print(f"Starting processing for level {level}")

        # =============== Load GCM ===============
        sliced_gcm_all = []
        sliced_gcm = xr.Dataset()

        # Load each variable and adjust longitude for ua and va if necessary
        for var_name, file_paths in file_paths_by_variable_gcm.items():
            data_var = load_preprocess_variable(
                file_paths,
                var_name,
                level,
                lat_range,
                lon_range,
                startyear_h,
                endyear_h,
            )
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
            # Add the processed variable to the dataset
            sliced_gcm[var_name] = data_var
            # Chunk the data
            sliced_gcm = sliced_gcm.chunk({"time": 1000, "lat": -1, "lon": -1})
        sliced_gcm_all.append(sliced_gcm)
        # Rename variables
        # rename_dict = {"hus": "q", "ta": "t", "ua": "u", "va": "v"}
        if bc_boundary == "lateral":
            sliced_gcm = sliced_gcm_all[0]
            assign_gcm = assign_w_6hr(sliced_gcm, bc_boundary)
            daily_gcm, fraction_factors_gcm = convert_to_daily_with_fraction(assign_gcm)
            daily_gcm = daily_gcm.chunk({"time": 1000, "lat": -1, "lon": -1})
            # daily_gcm_rechunk = daily_gcm.chunk({"time": 1000, "lat": -1, "lon": -1})
            fraction_factors_gcm = fraction_factors_gcm.chunk(
                {"time": 1000, "lat": -1, "lon": -1}
            )
        else:
            daily_gcm = sliced_gcm_all[0]

        # print("start delayed process gcm")
        # print("daily_gcm", daily_gcm)
        reshaped_gcm_delayed = extract_and_reshape_delayed(
            daily_gcm, no_of_variables, startyear_h, endyear_h, bc_boundary
        )
        # print("reshaped_gcm_delayed", reshaped_gcm_delayed)
        if bc_boundary != "lateral":
            reshaped_gcm_delayed += 273.15

        # =============== Load GCM end ===============

        # =============== Load Obs ===============
        sliced_obs_all = []
        sliced_obs = xr.Dataset()

        # Load each variable and adjust longitude for ua and va if necessary
        for var_name, file_paths in file_paths_by_variable_obs.items():
            obs_var = load_preprocess_variable(
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
        # Rename variables
        if bc_boundary == "lateral":
            sliced_obs = sliced_obs_all[0]
            assign_obs = assign_w_6hr(sliced_obs, bc_boundary)
            daily_obs, fraction_factors_obs = convert_to_daily_with_fraction(assign_obs)
            daily_obs = daily_obs.chunk({"time": 1000, "lat": -1, "lon": -1})
            # daily_obs_rechunk = daily_obs.chunk({"time": 1000, "lat": -1, "lon": -1})
            fraction_factors_obs = fraction_factors_obs.chunk(
                {"time": 1000, "lat": -1, "lon": -1}
            )
        else:
            daily_obs = sliced_obs_all[0]

        print("start delayed process obs")
        # print("daily_obs", daily_obs)
        reshaped_obs_delayed = extract_and_reshape_delayed(
            daily_obs, no_of_variables, startyear_h, endyear_h, bc_boundary
        )
        # print("reshaped_obs_delayed", reshaped_obs_delayed)
        # =============== Load obs end ===============

        # Create a temporary folder for saving intermediate files
        temp_dir = os.path.join(out_path, "temp_tiles")
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

                # Perform bias correction across the tile
                if bc_boundary == "lateral":
                    fraction_factors_gcm_tile = fraction_factors_gcm.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )
                    fraction_factors_obs_tile = fraction_factors_obs.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )
                    sliced_gcm_tile = sliced_gcm.sel(
                        lat=slice(*lat_range), lon=slice(*lon_range)
                    )

                    bc_corrected_gcm_hist_tile, bc_params_tile = (
                        bc_correction_grid_cell_hist_dask(
                            reshaped_gcm_delayed_tile,
                            reshaped_obs_delayed_tile,
                            fraction_factors_gcm_tile,
                            fraction_factors_obs_tile,
                            var_list_w,
                            sliced_gcm_tile,
                            config.startyear_h,
                            config.endyear_h,
                        )
                    )
                else:
                    bc_corrected_gcm_hist_tile, bc_params_tile = (
                        bc_correction_grid_cell_hist_dask_2d(
                            reshaped_gcm_delayed_tile,
                            reshaped_obs_delayed_tile,
                            var_list_w,
                            config.startyear_h,
                        )
                    )

                # Save each tile's bias-corrected output immediately to disk
                output_file = f"{temp_dir}/bc_corrected_tile_{idx}_{lat_range[0]}_{lat_range[1]}_{lon_range[0]}_{lon_range[1]}.nc"
                bc_corrected_gcm_hist_tile.compute().to_netcdf(
                    output_file
                )  # Save tile result

                # Accumulate bc_params_tile for later concatenation
                all_bc_params.append(bc_params_tile)

                # Free memory after saving each tile
                del bc_corrected_gcm_hist_tile
                # print(f"Processed and saved tile {idx}, lat range: {lat_range}, lon range: {lon_range}")

            except Exception as e:
                print(f"Error processing tile {idx}: {e}")
                continue

        # if bc_boundary == "lateral":
        #     full_bc_corrected, bc_params_array = bc_correction_grid_cell_hist_dask(
        #         reshaped_gcm_delayed,
        #         reshaped_obs_delayed,
        #         fraction_factors_gcm,
        #         fraction_factors_obs,
        #         var_list_w,
        #         sliced_gcm,
        #         config.startyear_h,
        #         config.endyear_h,
        #     )
        # else:
        #     print(var_list_w)
        #     full_bc_corrected, bc_params_array = (
        #         bc_correction_grid_cell_hist_dask_2d(
        #             reshaped_gcm_delayed,
        #             reshaped_obs_delayed,
        #             var_list_w,
        #             config.startyear_h,
        #         )
        #     )

        # print("end dask bc correction")
        # # full_bc_corrected = full_bc_corrected.transpose("time", "lat", "lon")

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
            f"{temp_dir}/bc_corrected_tile_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
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
        full_bc_corrected = full_bc_corrected.drop_duplicates("lat").drop_duplicates(
            "lon"
        )

        # # Assuming `full_bc_corrected` has variables like 'wind_speed', 'temperature', 'specific_humidity', 'sst'
        # variable_names = list(full_bc_corrected.data_vars)  # Get the names of the variables in the dataset

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
            if bc_boundary == "lateral":
                full_bc_corrected[var].attrs.update(
                    {
                        "coordinates": "time lev lat lon",
                        "description": f"bias-corrected data, {m_names[int(config.correction_model)-1]}, SDMBCv2",
                        "history": "Created by applying SDMBCv2 package",
                    }
                )
            else:
                full_bc_corrected[var].attrs.update(
                    {
                        "coordinates": "time lat lon",
                        "description": f"bias-corrected data, {m_names[int(config.correction_model)-1]}, SDMBCv2",
                        "history": "Created by applying SDMBCv2 package",
                    }
                )
        if bc_boundary == "lateral":
            full_bc_corrected = full_bc_corrected.transpose("time", "lev", "lat", "lon")
        else:
            full_bc_corrected = full_bc_corrected.transpose("time", "lat", "lon")

        print("save the bc model")
        # Save the BC model
        if bc_boundary == "lateral":
            np.save(
                f"{out_path}/bc_params_3d_level_{level}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
                full_param_array_corrected,
            )
        else:
            np.save(
                f"{out_path}/bc_params_2d_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
                full_param_array_corrected,
            )

        if config.save_bc_output:
            # save the bias corrected data # from input gcm or obs to target gcm
            if bc_boundary == "lateral":
                full_bc_corrected.load().to_netcdf(
                    f"{out_path}/bc_corrected_3d_level_{level}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
                )
            else:
                full_bc_corrected.load().to_netcdf(
                    f"{out_path}/bc_corrected_2d_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
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

        # if config.draw_figure:
        #     ds_corrected_day = assign_w_day(full_bc_corrected, bc_boundary)
        #     if bc_boundary == "lateral":
        #         print("Drawing figures")
        #         if config.sub_daily_correction:
        #             print("K-S test has been included")
        #             statistics = AnalysisBC(
        #                 daily_gcm,
        #                 daily_obs,
        #                 ds_corrected_day,
        #                 variables,
        #                 config.out_figure_path,
        #                 kstest=True,
        #             )
        #         if config.sub_daily_correction == False:
        #             print("K-S test has not been included")
        #             statistics = AnalysisBC(
        #                 daily_gcm,
        #                 daily_obs,
        #                 ds_corrected_day,
        #                 variables,
        #                 config.out_figure_path,
        #                 kstest=False,
        #             )
        #         statistics.figure_atmos()
        #         print("Finish 3d field")

        #     else:
        #         statistics.figure_surface()
        #         print("Finish 2d field")

    # if config.reformat_to_original:
    #     print("Start reformatting")
    #     if bc_boundary == "lateral":
    #         reformat_and_save_3d(
    #             out_path,
    #             tlevel,
    #             startyear_h,
    #             endyear_h,
    #             variables,
    #             variables,
    #             out_path,
    #             infor,
    #             gname,
    #             period,
    #             cinfor,
    #             sinfor,
    #         )
    #         print("Finish 3D reformatting")

    #     else:
    #         reformat_and_save_2d(
    #             out_path,
    #             startyear_h,
    #             endyear_h,
    #             variables,
    #             variables,
    #             out_path,
    #             infor,
    #             gname,
    #             period,
    #             cinfor,
    #             sinfor,
    #             startyear_h,
    #             endyear_h,
    #         )
    #         print("Finish 2D reformatting")


if __name__ == "__main__":
    main(config)
    print("All done!")
