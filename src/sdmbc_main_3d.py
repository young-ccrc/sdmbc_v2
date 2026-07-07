import argparse
import logging
import os
import shutil
import time  # Import the time module
import warnings
from types import SimpleNamespace

import dask  # type: ignore

# import pandas as pd # type: ignore
import xarray as xr  # type: ignore
import yaml  # type: ignore
from dask.distributed import Client  # type: ignore
from tqdm import tqdm  # type: ignore

# from bc_grid_function import load_preprocess_variable  # type: ignore
from bc_grid_function import (
    convert_bc_params_to_xarray,
    preprocess_and_save_gcm,
    preprocess_and_save_obs,
    process_tile,
    process_tile_future,
    _atomic_to_netcdf,
)

# from data_preparation import   # type: ignore
from data_preparation import (
    determine_tiles,
    expand_config_bounds_from_data,
    generate_file_paths,
    generate_file_paths_obs,
    split_domain,
    validate_inputs,
)

# from figurefunction import save_figure_3d

# Suppress INFO and lower-level logs
logging.getLogger("flox").setLevel(logging.WARNING)
logging.getLogger("xarray").setLevel(logging.WARNING)
logging.getLogger("dask").setLevel(logging.WARNING)

# sys.path.append("/g/data/w28/yk8692/sdmbc_v2/sdmbc_v2")


# from analysis_plot import AnalysisBC  # type: ignore


warnings.simplefilter("ignore", UserWarning)


# Start of the script -----------------------------------------------------
# def setup_client(ncpus, mem_gb):
#     """
#     Dynamically set up a Dask client based on available CPUs and memory.
#     """
#     threads_per_worker = min(ncpus, 8)
#     n_workers = max(1, ncpus // threads_per_worker)
#     mem_per_worker = int(mem_gb / n_workers)

#     print(
#         f"[INFO] Starting Dask client: {n_workers} workers × {threads_per_worker} threads"
#     )
#     print(f"[INFO] Each worker memory limit: {mem_per_worker}GB")

#     return Client(
#         n_workers=n_workers,
#         threads_per_worker=threads_per_worker,
#         memory_limit=f"{mem_per_worker}GB",
#     )


def setup_client(ncpus, mem_gb):
    """Dynamically set up a Dask client based on available CPUs and memory."""
    ncpus = ncpus or int(os.environ.get("PBS_NCPUS", os.cpu_count() or 1))

    threads_per_worker = int(os.environ.get("DASK_THREADS_PER_WORKER") or "1")
    n_workers_default = max(1, ncpus // threads_per_worker)
    n_workers = int(os.environ.get("DASK_N_WORKERS") or str(n_workers_default))
    processes = (os.environ.get("DASK_PROCESSES") or "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mem_per_worker = max(1, int(mem_gb / n_workers))
    local_directory = os.environ.get("DASK_LOCAL_DIRECTORY") or os.environ.get(
        "DASK_TEMPORARY_DIRECTORY"
    )
    if local_directory:
        os.makedirs(local_directory, exist_ok=True)

    print(
        f"[INFO] Starting Dask client: {n_workers} workers × {threads_per_worker} threads "
        f"(processes={processes})"
    )
    print(f"[INFO] Each worker memory limit: {mem_per_worker}GB")
    if local_directory:
        print(f"[INFO] Dask local directory: {local_directory}")

    return Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        memory_limit=f"{mem_per_worker}GB",
        local_directory=local_directory,
    )


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

    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument("--mem", type=int, default=None)

    return parser.parse_args()


# def split_domain(lat_min, lat_max, lon_min, lon_max, n_lat_tiles, n_lon_tiles):
#     # Create ranges for latitude and longitude tiles
#     lat_ranges = np.linspace(lat_min, lat_max, n_lat_tiles + 1, endpoint=True)
#     lon_ranges = np.linspace(lon_min, lon_max, n_lon_tiles + 1, endpoint=True)

#     tiles = []
#     for i in range(n_lat_tiles):
#         for j in range(n_lon_tiles):
#             tiles.append(
#                 {
#                     "lat_min": lat_ranges[i],
#                     "lat_max": lat_ranges[i + 1],
#                     "lon_min": lon_ranges[j],
#                     "lon_max": lon_ranges[j + 1],
#                 }
#             )
#     return tiles


# def expand_config_bounds_from_data(config, sample_file):
#     """
#     Dynamically expands latitude and longitude bounds based on the grid step
#     detected from a sample NetCDF file.

#     Args:
#         config: Configuration object containing lat/lon boundaries.
#         sample_file (str): Path to a sample NetCDF file to detect lat/lon resolution.

#     Returns:
#         Updated configuration with expanded lat/lon bounds.
#     """

#     # Load a small dataset to detect lat/lon spacing
#     sample_data = xr.open_dataset(sample_file)

#     # Compute grid spacing
#     lat_step = np.abs(sample_data.lat[1] - sample_data.lat[0])
#     lon_step = np.abs(sample_data.lon[1] - sample_data.lon[0])

#     # Expand the domain before loading full dataset
#     config.lat_min -= lat_step
#     config.lat_max += lat_step
#     config.lon_min -= lon_step
#     config.lon_max += lon_step

#     return config.lat_min, config.lat_max, config.lon_min, config.lon_max


def dict_to_simplenamespace(d):
    return SimpleNamespace(**d)


def use_flat_truth_paths(config):
    return getattr(config, "input_model", "") in ("reanalysis", "model_as_truth")


def main(config_path, override_ncpus=None, override_mem=None):

    # setup_client()
    # Cap math/BLAS threads to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
    print("Dask client started with resources:", ncpus, "CPUs and", mem_gb, "GB memory")

    startyear_h = config.startyear_h
    endyear_h = config.endyear_h
    bc_boundary = config.bc_boundary
    bc_hist_path = config.bc_hist_path
    lat_max = config.lat_max
    lat_min = config.lat_min
    lon_max = config.lon_max
    lon_min = config.lon_min
    requested_lat_min = float(lat_min)
    requested_lat_max = float(lat_max)
    requested_lon_min = float(lon_min)
    requested_lon_max = float(lon_max)
    slevel = config.slevel
    elevel = config.elevel
    obs_path = config.obs_path
    out_path = config.out_path
    temp_root = getattr(config, "temp_root", out_path)
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
            config,
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
            generate_file_paths_obs(
                config, obs_path, var, gname, startyear_h, endyear_h
            )
            if use_flat_truth_paths(config)
            else generate_file_paths(
                config,
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
    # obs_path_f = "/g/data/w28/yk8692/input/era5/validation"
    if config.bc_future:
        file_paths_by_variable_obs_f = {
            var: (
                generate_file_paths_obs(
                    config, obs_path, var, gname, startyear_f, endyear_f
                )
                if use_flat_truth_paths(config)
                else generate_file_paths(
                    config,
                    bc_hist_path,
                    var,
                    infor,
                    gname,
                    period,
                    cinfor,
                    sinfor,
                    version,
                    startyear_f,
                    endyear_f,
                )
            )
            for var in variables
        }

    # =============== Load obs end ===============
    # Dynamically determine number of tiles
    if not file_paths_by_variable_obs[variables[0]]:
        raise ValueError(f"No observation files found for variable: {variables[0]}")

    lat_min, lat_max, lon_min, lon_max, lat_size, lon_size, lat_values, lon_values = (
        expand_config_bounds_from_data(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            file_paths_by_variable_obs[variables[0]][0],
        )
    )

    # expand_config_bounds_from_data snaps to the available grid and may widen
    # the domain. Clip back to explicitly requested bounds so tests can exclude
    # invalid pole rows such as lat=-90/90 after interpolation.
    lat_mask = (lat_values >= requested_lat_min) & (lat_values <= requested_lat_max)
    lon_mask = (lon_values >= requested_lon_min) & (lon_values <= requested_lon_max)
    lat_values = lat_values[lat_mask]
    lon_values = lon_values[lon_mask]
    if lat_values.size == 0 or lon_values.size == 0:
        raise ValueError(
            "Requested domain has no grid cells after clipping: "
            f"lat={requested_lat_min}:{requested_lat_max}, "
            f"lon={requested_lon_min}:{requested_lon_max}"
        )
    lat_min = float(lat_values[0])
    lat_max = float(lat_values[-1])
    lon_min = float(lon_values[0])
    lon_max = float(lon_values[-1])
    lat_size = lat_values.size
    lon_size = lon_values.size
    print(
        f"[INFO] effective domain after requested-bound clipping: "
        f"lat={lat_min}:{lat_max} ({lat_size}), "
        f"lon={lon_min}:{lon_max} ({lon_size})"
    )

    if getattr(config, "single_grid_test", False):
        single_lat = float(getattr(config, "single_lat", lat_min))
        single_lon = float(getattr(config, "single_lon", lon_min))
        lat_idx = int(abs(lat_values - single_lat).argmin())
        lon_idx = int(abs(lon_values - single_lon).argmin())
        lat_values = lat_values[lat_idx : lat_idx + 1]
        lon_values = lon_values[lon_idx : lon_idx + 1]
        lat_min = lat_max = float(lat_values[0])
        lon_min = lon_max = float(lon_values[0])
        lat_size = lon_size = 1
        print(
            f"[INFO] single_grid_test enabled at nearest grid cell: "
            f"lat={lat_min}, lon={lon_min}"
        )

    n_lat_tiles, n_lon_tiles = determine_tiles(config, lat_size, lon_size)

    print("start dask bc correction")

    tiles = split_domain(
        n_lat_tiles,
        n_lon_tiles,
        lat_values,
        lon_values,
    )
    tile_items = list(enumerate(tiles))

    tile_ids_env = os.environ.get("SDMBC_TILE_IDS")
    tile_start_env = os.environ.get("SDMBC_TILE_START")
    tile_end_env = os.environ.get("SDMBC_TILE_END")
    if tile_ids_env:
        selected_ids = {
            int(value.strip())
            for value in tile_ids_env.split(",")
            if value.strip()
        }
        tile_items = [(idx, tile) for idx, tile in tile_items if idx in selected_ids]
        print(f"[INFO] selected tile ids: {sorted(selected_ids)}")
    elif tile_start_env is not None or tile_end_env is not None:
        tile_start = int(tile_start_env or 0)
        tile_end = int(tile_end_env or len(tiles))
        tile_start = max(0, tile_start)
        tile_end = min(len(tiles), tile_end)
        tile_items = [
            (idx, tile)
            for idx, tile in tile_items
            if tile_start <= idx < tile_end
        ]
        print(f"[INFO] selected tile range: [{tile_start}, {tile_end})")

    if not tile_items:
        raise ValueError("No tiles selected for processing.")
    print(f"[INFO] processing {len(tile_items)} of {len(tiles)} tiles")
    skip_merge = (os.environ.get("SDMBC_SKIP_MERGE") or "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if skip_merge:
        print("[INFO] SDMBC_SKIP_MERGE=true: tile outputs will not be merged in this job")
    # print('tiles', tiles)
    domain = split_domain(
        n_lat_tiles=1,
        n_lon_tiles=1,
        lat_values=lat_values,
        lon_values=lon_values,
    )
    preprocess_scope = str(getattr(config, "preprocess_scope", "domain")).lower()
    if preprocess_scope not in {"domain", "tile"}:
        raise ValueError(
            "preprocess_scope must be 'domain' or 'tile', "
            f"got {preprocess_scope!r}"
        )
    print(f"[INFO] preprocess_scope={preprocess_scope}")

    for level in range(slevel, elevel + 1):
        # Record the start time for this level
        start_time = time.time()
        # lat_range = (lat_min, lat_max)
        # lon_range = (lon_min, lon_max)
        print(f"Starting processing for level {level}")

        temp_dir = os.path.join(temp_root, f"temp_tiles_{gname}_{level}")

        os.makedirs(temp_dir, exist_ok=True)
        if preprocess_scope == "domain":
            # ------------------ Load GCM Data ------------------
            temp_gcm = preprocess_and_save_gcm(
                config,
                domain,
                file_paths_by_variable_gcm,
                temp_dir,
                level,
            )
            sliced_gcm = xr.open_dataset(
                f"{temp_dir}/preprocessed_{gname}_lev_{level}_{domain[0]['lat_min']}_{domain[0]['lat_max']}_{domain[0]['lon_min']}_{domain[0]['lon_max']}.nc"
            )

            # ------------------ Load GCM Data end ------------------

            # ------------------ Load Obs Data ------------------
            for var_name, file_paths in file_paths_by_variable_obs.items():
                temp_netcdf = preprocess_and_save_obs(
                    config,
                    domain,
                    file_paths,
                    temp_dir,
                    var_name,
                    level,
                    startyear_h,
                    endyear_h,
                )
            sliced_obs = xr.Dataset()
            obs_domain = domain[0]
            for var_name in variables:
                # Construct the path to the preprocessed file for this variable
                obs_file = os.path.join(
                    temp_dir,
                    f"preprocessed_obs_{var_name}_lev_{level}_{obs_domain['lat_min']}_{obs_domain['lat_max']}_{obs_domain['lon_min']}_{obs_domain['lon_max']}_{startyear_h}_{endyear_h}.nc",
                )
                sliced_obs[var_name] = xr.open_dataset(obs_file)[
                    var_name
                ]  # Load the variable from the file

            sliced_obs = sliced_obs.astype("float32")
            # ------------------ Load Obs Data end ------------------

        if config.bc_hist:

            # To store all bc_params for later concatenation
            all_bc_params = []
            single_tile_fast_path = False
            single_tile_corrected = None
            single_tile_params = None
            if single_tile_fast_path:
                print(
                    "[INFO] Single-tile fast path enabled: "
                    "skipping temporary tile NetCDF writes."
                )
            # # for idx, tile in enumerate(tiles):
            for idx, tile in tqdm(tile_items, desc="Processing tiles"):
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
                # lat_range = (tile["lat_min"], tile["lat_max"])
                # lon_range = (tile["lon_min"], tile["lon_max"])
                lat_i0, lat_i1 = tile["lat_min_idx"], tile["lat_max_idx"]
                lon_j0, lon_j1 = tile["lon_min_idx"], tile["lon_max_idx"]
                # print('lat_i0, lat_i1, lon_j0, lon_j1', lat_i0, lat_i1, lon_j0, lon_j1)
                lat_range = (tile["lat_min"], tile["lat_max"])
                lon_range = (tile["lon_min"], tile["lon_max"])
                # print(f"Tile {idx}: lat {lat_range}, lon {lon_range}")
                # try:
                # Define output file paths
                output_file = f"{temp_dir}/bc_corrected_tile_3d_{period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
                output_params = f"{temp_dir}/bc_params_tile_3d_{period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"

                if preprocess_scope == "tile":
                    if os.path.exists(output_file) and os.path.exists(output_params):
                        print(
                            f"Both output files for tile {idx}, level {level} exist. Skipping..."
                        )
                        all_bc_params.append(output_params)
                        continue

                    tile_domain = [tile]
                    temp_gcm = preprocess_and_save_gcm(
                        config,
                        tile_domain,
                        file_paths_by_variable_gcm,
                        temp_dir,
                        level,
                    )
                    gcm_tile = xr.open_dataset(temp_gcm)

                    for var_name, file_paths in file_paths_by_variable_obs.items():
                        temp_netcdf = preprocess_and_save_obs(
                            config,
                            tile_domain,
                            file_paths,
                            temp_dir,
                            var_name,
                            level,
                            startyear_h,
                            endyear_h,
                        )
                    obs_tile = xr.Dataset()
                    for var_name in variables:
                        obs_file = os.path.join(
                            temp_dir,
                            f"preprocessed_obs_{var_name}_lev_{level}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}_{startyear_h}_{endyear_h}.nc",
                        )
                        obs_tile[var_name] = xr.open_dataset(obs_file)[var_name]
                    obs_tile = obs_tile.astype("float32")
                else:
                    obs_tile = sliced_obs.isel(
                        lat=slice(lat_i0, lat_i1), lon=slice(lon_j0, lon_j1)
                    )
                    gcm_tile = sliced_gcm.isel(
                        lat=slice(lat_i0, lat_i1), lon=slice(lon_j0, lon_j1)
                    )
                # # Check if both output files already exist
                # if os.path.exists(output_file) and os.path.exists(output_params):
                #     print(
                #         f"Both output file and params for tile {idx}, level {level} already exist. Skipping..."
                #     )
                #     continue  # Skip processing this tile
                # print('obs_tile', obs_tile)
                # print('gcm_tile', gcm_tile)
                # Process the tile if either output is missing
                print(f"Processing tile {idx}, level {level}...")
                bc_corrected_gcm_hist_tile, bc_params_tile = process_tile(
                    config,
                    gcm_tile,
                    obs_tile,
                )

                # Save the bias-corrected output
                # if not os.path.exists(output_file):
                #     print(
                #         f"Saving 3D output for tile {idx}, level {level} to {output_file}"
                #     )
                #     bc_corrected_gcm_hist_tile.compute().to_netcdf(output_file)
                # else:
                #     print(f"File {output_file} already exists. Skipping...")
                if single_tile_fast_path:
                    print(
                        f"[INFO] Keeping corrected output for tile {idx}, level {level} in memory."
                    )
                elif not os.path.exists(output_file):
                    print(
                        f"Saving 3D output for tile {idx}, level {level} to {output_file}"
                    )
                    ds_tile = (
                        bc_corrected_gcm_hist_tile.compute()
                    )  # materialize to numpy
                    _atomic_to_netcdf(ds_tile, output_file)
                else:
                    print(f"File {output_file} already exists. Skipping...")

                # if isinstance(bc_params_tile, dict):
                # Assume each tile hn be extracted from obs_tile or gcm_tile.
                tile_lat = obs_tile["lat"].values
                tile_lon = obs_tile["lon"].values

                # (Optional safety check)
                if (
                    tile_lat.shape[0] != bc_params_tile.shape[0]
                    or tile_lon.shape[0] != bc_params_tile.shape[1]
                ):
                    raise ValueError(
                        f"Tile {idx} coord size mismatch: "
                        f"lat coords {tile_lat.shape[0]} vs params {bc_params_tile.shape[0]}, "
                        f"lon coords {tile_lon.shape[0]} vs params {bc_params_tile.shape[1]}"
                    )

                ds_params = convert_bc_params_to_xarray(
                    config, bc_params_tile, tile_lat, tile_lon
                )
                # else:
                #     ds_params = bc_params_tile

                # Save the bias-correction parameters
                # if not os.path.exists(output_params):
                #     print(
                #         f"Saving 3D params for tile {idx}, level {level} to {output_params}"
                #     )
                #     np.save(output_params, bc_params_tile)
                # else:
                #     print(f"File {output_params} already exists. Skipping...")
                #     # continue
                # if not os.path.exists(output_params):
                #     print(
                #         f"Saving 3D params for tile {idx}, level {level} to {output_params}"
                #     )
                #     ds_params.to_netcdf(output_params)
                # else:
                #     print(f"File {output_params} already exists. Skipping...")
                if single_tile_fast_path:
                    print(
                        f"[INFO] Keeping BC params for tile {idx}, level {level} in memory."
                    )
                    single_tile_corrected = bc_corrected_gcm_hist_tile
                    single_tile_params = ds_params.load()
                elif not os.path.exists(output_params):
                    print(
                        f"Saving 3D params for tile {idx}, level {level} to {output_params}"
                    )
                    _atomic_to_netcdf(ds_params.load(), output_params)
                else:
                    print(f"File {output_params} already exists. Skipping...")
                if not single_tile_fast_path:
                    # Append the bc_params file path for later merging.
                    all_bc_params.append(output_params)

                # Free memory after saving each tile
                del bc_corrected_gcm_hist_tile, ds_params, gcm_tile, obs_tile
                # print(f"Processed and saved tile {idx}, lat range: {lat_range}, lon range: {lon_range}")

                # except Exception as e:
                #     print(f"Error processing tile {idx}: {e}")
                #     # continue

            # # Concatenate all bc_params along the latitude and longitude
            # # Initialize an empty list to hold rows of tiles for each latitude band
            # lat_band_tiles = []as its own lat/lon coordinates that ca

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

            if skip_merge:
                print(
                    f"[INFO] Skipping final merge for level {level}; "
                    f"processed {len(tile_items)} tile(s)."
                )
                elapsed_time = time.time() - start_time
                print(f"Completed tile-group processing in {elapsed_time / 60:.2f} minutes")
                continue

            if single_tile_fast_path:
                if single_tile_corrected is None or single_tile_params is None:
                    raise RuntimeError("Single-tile fast path did not produce outputs.")
                full_bc_corrected = single_tile_corrected
            else:
                # Load the bias-corrected tiles and combine them into a single dataset
                tile_files = [
                    f"{temp_dir}/bc_corrected_tile_3d_{period}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"
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
                    # Clip finite out-of-range values while preserving NaNs for diagnostics.
                    full_bc_corrected[var_name] = full_bc_corrected[var_name].clip(
                        min=lower,
                        max=upper,
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
            # np.save(
            #     f"{out_path}/bc_params_3d_{period}_lev_{level}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.npy",
            #     full_param_array_corrected,
            # )
            if single_tile_fast_path:
                ds_full_params = single_tile_params
            else:
                # Merge all tile bc_params NetCDF files into a single xarray Dataset for the whole domain.
                # This uses xarray's open_mfdataset which combines datasets by matching coordinate values.
                ds_full_params = xr.open_mfdataset(all_bc_params, combine="by_coords")

            # If for any reason duplicate lat or lon values exist, remove duplicates.
            # xarray's merge usually handles non-overlapping coordinates, but if needed:
            ds_full_params = ds_full_params.drop_duplicates("lat").drop_duplicates(
                "lon"
            )

            # Save the merged full-domain bias-correction parameters as a single NetCDF file.
            full_params_output = f"{out_path}/bc_params_3d_{period}_lev_{level}_{gname}_to_{input_model}_{startyear_h}_{endyear_h}.nc"
            # ds_full_params.to_netcdf(full_params_output)
            _atomic_to_netcdf(ds_full_params.load(), full_params_output)
            print("Saved full domain bc_params to", full_params_output)

            if config.save_bc_output:
                full_bc_corrected.load().to_netcdf(
                    f"{out_path}/bc_corrected_3d_lev_{level}_{infor}_{gname}_{period}_{cinfor}_{sinfor}_{startyear_h}_{endyear_h}.nc"
                )

            del full_bc_corrected, ds_full_params

        if config.bc_future:

            for idx, tile in enumerate(tqdm(tiles, desc="Processing tiles")):
                # try:
                # test
                for var_name, file_paths in file_paths_by_variable_obs_f.items():
                    temp_netcdf = preprocess_and_save_obs(
                        config,
                        domain,
                        file_paths,
                        temp_dir,
                        var_name,
                        level,
                        startyear_f,
                        endyear_f,
                    )
                # Define output file paths
                output_file = f"{temp_dir}/bc_corrected_tile_3d_{period_f}_lev_{level}_{idx}_{tile['lat_min']}_{tile['lat_max']}_{tile['lon_min']}_{tile['lon_max']}.nc"

                lat_i0, lat_i1 = tile["lat_min_idx"], tile["lat_max_idx"]
                lon_j0, lon_j1 = tile["lon_min_idx"], tile["lon_max_idx"]

                lat_range = (tile["lat_min"], tile["lat_max"])
                lon_range = (tile["lon_min"], tile["lon_max"])

                if os.path.exists(output_file):
                    print(f"File {output_file} already exists. Skipping...")
                else:
                    obs_tile = sliced_obs.isel(
                        lat=slice(lat_i0, lat_i1), lon=slice(lon_j0, lon_j1)
                    )
                    gcm_tile = sliced_gcm.isel(
                        lat=slice(lat_i0, lat_i1), lon=slice(lon_j0, lon_j1)
                    )
                    bc_corrected_gcm_future_tile = process_tile_future(
                        config,
                        tile,
                        variables,
                        level,
                        gcm_tile,
                        obs_tile,
                        temp_dir,
                        idx,
                    )

                # Save the bias-corrected output
                # if not os.path.exists(output_file):
                #     print(
                #         f"Saving 3D output for tile {idx}, level {level} to {output_file}"
                #     )
                #     bc_corrected_gcm_future_tile.compute().to_netcdf(output_file)
                #     del bc_corrected_gcm_future_tile
                if not os.path.exists(output_file):
                    print(
                        f"Saving 3D output for tile {idx}, level {level} to {output_file}"
                    )
                    ds_tile_f = bc_corrected_gcm_future_tile.compute()
                    _atomic_to_netcdf(ds_tile_f, output_file)
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
                    # Clip finite out-of-range values while preserving NaNs for diagnostics.
                    full_bc_corrected[var_name] = full_bc_corrected[var_name].clip(
                        min=lower,
                        max=upper,
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

            # if "lev" not in full_bc_corrected.dims:
            #     full_bc_corrected = full_bc_corrected.expand_dims(
            #         lev=[full_bc_corrected.lev[0].values]
            #     )

            if (
                "lev" in full_bc_corrected.coords
                and "lev" not in full_bc_corrected.dims
            ):
                full_bc_corrected = full_bc_corrected.drop_vars(
                    "lev"
                )  # Remove as a coordinate
                full_bc_corrected = full_bc_corrected.expand_dims(
                    lev=[sliced_gcm.lev.values]
                )  # Add as a dimension

            full_bc_corrected = full_bc_corrected.transpose("time", "lev", "lat", "lon")

            print("save the bc model")
            full_bc_corrected = full_bc_corrected.astype("float32")  # save as float32

            # if config.save_bc_output:
            #     # save the bias corrected data # from input gcm or obs to target gcm
            #     full_bc_corrected.load().to_netcdf(
            #         f"{out_path}/bc_corrected_3d_lev_{level}_{infor}_{gname}_{period_f}_{cinfor}_{sinfor}_{startyear_f}_{endyear_f}.nc"
            #     )
            if config.save_bc_output:
                _atomic_to_netcdf(
                    full_bc_corrected.load(),
                    f"{out_path}/bc_corrected_3d_lev_{level}_{infor}_{gname}_{period_f}_{cinfor}_{sinfor}_{startyear_f}_{endyear_f}.nc",
                )
        # Remove temporary files only when explicitly requested. Tile-group
        # workflows need these files for QC and later full-domain merging.
        if getattr(config, "cleanup_bc_inputs", True):
            shutil.rmtree(temp_dir)
            print("Intermediate files deleted.")
        else:
            print(f"Intermediate files retained at {temp_dir}")

        # Log the time taken for this level
        end_time = time.time()
        elapsed_time_minutes = (
            end_time - start_time
        ) / 60  # Convert seconds to minutes
        print(f"Completed processing in {elapsed_time_minutes:.2f} minutes")


if __name__ == "__main__":
    # args = parse_arguments()
    # cfg = Config(path=args.config)
    # main(cfg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--yp", required=True, help="Path to YAML config")
    parser.add_argument(
        "--ncpus", type=int, default=None, help="Override number of CPUs"
    )
    parser.add_argument("--mem", type=int, default=None, help="Override memory in GB")

    args = parser.parse_args()
    main(args.yp, args.ncpus, args.mem)

    print("All done!")
