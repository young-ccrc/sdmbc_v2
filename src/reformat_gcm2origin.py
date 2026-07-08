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

# Written by Youngil Kim
# Climate Change Research Centre
# University of New South Wales
# 2025-10-01
# -----------------------------------------------------------------------------------------------------------------
# Load pacakges ===================================
import glob
import json
import os
import re
from functools import partial
from pathlib import Path  # added

import dask  # type: ignore  # added
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import xarray as xr  # type: ignore
import yaml  # type: ignore  # added
from cdo import Cdo  # type: ignore
from dask.distributed import Client  # type: ignore

# from config import config
from interpolation import regrid

cdo = Cdo()
# Mitigate HDF5 file locking issues on shared filesystems before any IO libs import
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Load pacakges end ================================
# ---------------------------------------------------------------------------------------------------
# Functions to be used for main_reformat


# Start of the script -----------------------------------------------------
def load_config(yaml_path):
    """
    Load configuration from a YAML file.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        Config: Configuration object with loaded settings.
    """
    with open(yaml_path, "r") as file:
        config_data = yaml.safe_load(file)
    # return Config(**config_data)
    return config_data


def setup_client(ncpus, mem_gb):
    """
    Dynamically set up a Dask client based on available CPUs and memory.
    """
    # Set threads per worker (try to keep under 8 for memory balance)
    threads_per_worker = min(ncpus, 8)
    n_workers = max(1, ncpus // threads_per_worker)
    mem_per_worker = int(mem_gb / n_workers)

    print(
        f"[INFO] Starting Dask client: {n_workers} workers × {threads_per_worker} threads"
    )
    print(f"[INFO] Each worker memory limit: {mem_per_worker}GB")

    return Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f"{mem_per_worker}GB",
    )


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run reformatting.")
    parser.add_argument(
        "--yp",
        type=str,
        default="./user_input_test.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument("--mem", type=int, default=None)

    return parser.parse_args()


def infer_cmip6_ids_from_path(path):
    """
    Infer (activity, institution, source_id) from a CMIP6 path.
    Example: /g/data/oi10/replicas/CMIP6/CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/...
    """
    parts = Path(path).parts
    try:
        i = parts.index("CMIP6")
    except ValueError:
        return None, None, None
    # Need .../CMIP6/<activity>/<institution>/<source_id>/...
    if len(parts) > i + 3:
        return parts[i + 1], parts[i + 2], parts[i + 3]
    return None, None, None


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

    # Add level dimension if not sea surface temperature variables
    if vn not in ["tos", "sst"] and "lev" in old:
        new["lev"] = old["lev"]
        new["lev"].attrs = old["lev"].attrs
        new["lev"].encoding = {**old["lev"].encoding, "_FillValue": None}

    # Copy global attributes
    new.attrs = old.attrs

    return new


def get_vertical_dim_name(obj):
    """Return the name of the vertical dimension, supporting a few common aliases.
    Falls back to 'lev' if present, else returns None when not found.
    """
    candidates = ("lev", "level", "plev", "pressure", "height")
    dims = getattr(obj, "dims", {})
    for name in candidates:
        if name in dims:
            return name
    # also check coords if needed
    coords = getattr(obj, "coords", {})
    for name in candidates:
        if name in coords:
            return name
    return None


def align_vertical_to_target(da_bc, target_levels, vdim):
    """Interpolate or assign bc dataarray vertical coordinate to match target_levels.

    - If da_bc already has matching vdim and equal values, just return.
    - Else, if vdim present, use linear interpolation to target_levels.
    - Else, simply assign the target_levels as a new coordinate (expects broadcasting to work).
    """
    if vdim is None:
        # cannot detect a vertical dimension, just return as-is
        return da_bc
    if vdim in da_bc.dims:
        try:
            src = da_bc[vdim]
            # if equal, just reassign coords to ensure identity
            if src.size == target_levels.size and xr.DataArray(src).identical(
                target_levels
            ):
                return da_bc.assign_coords({vdim: target_levels})
        except Exception:
            pass
        # interpolate along vdim
        try:
            return da_bc.interp({vdim: target_levels}, method="linear")
        except Exception:
            # as a fallback, just assign coords (may raise on shape mismatch later)
            return da_bc.assign_coords({vdim: target_levels})
    else:
        # no such dim in data, just assign coords
        return da_bc.assign_coords({vdim: target_levels})


# -------------------------
# Discovery helpers for CMIP6
# -------------------------

# Default search roots: most GCMs in oi10, ACCESS in fs38
DEFAULT_ARCHIVE_ROOTS = [
    "/g/data/oi10/replicas/CMIP6",
    "/g/data/fs38/publications/CMIP6",
]

# Minimal variable->table mapping; can be extended/overridden by user config
VAR_TO_TABLE_DEFAULT = {
    # 3D 6-hourly
    "hus": "6hrLev",
    "ta": "6hrLev",
    "ua": "6hrLev",
    "va": "6hrLev",
    # common 2D
    "tos": "Oday",  # daily SST often under Oday (ocean daily)
}


def discover_cmip6_files(
    variable,
    activity,
    institution,
    source_id,
    experiment,
    variant,
    table_id=None,
    grid_label=None,
    version=None,
    year=None,
    roots=None,
):
    """Discover CMIP6 file paths by pattern under one or more archive roots.

    Path pattern (LLNL style):
      <root>/<activity>/<institution>/<source_id>/<experiment>/<variant>/<table>/<variable>/<grid_label>/<version>/<filename>.nc

    Only required arguments are the CMIP6 identifiers. 'table_id' will be
    guessed from a minimal mapping if omitted.
    """
    roots = roots or DEFAULT_ARCHIVE_ROOTS
    table = table_id or VAR_TO_TABLE_DEFAULT.get(variable, "*")
    grid_part = grid_label or "*"
    ver_part = version or "v*"

    results = []
    for root in roots:
        base = os.path.join(
            root,
            activity,
            institution,
            source_id,
            experiment,
            variant,
            table,
            variable,
            grid_part,
            ver_part,
        )
        # Filename pattern, include year if provided to limit matches
        fname = f"{variable}_{table}_{source_id}_{experiment}_{variant}_{grid_part}_"
        if year is not None:
            pattern = os.path.join(base, f"{fname}{year}*.nc")
        else:
            pattern = os.path.join(base, f"{fname}*.nc")
        matches = sorted(glob.glob(pattern))
        results.extend(matches)
    return sorted(set(results))


def get_archive_roots_user(config_module=None):
    """Return archive roots with user overrides.

    Order of precedence:
      1) config.archive_roots (list[str]) if present
      2) env CMIP6_ARCHIVE_ROOTS (comma-separated)
      3) DEFAULT_ARCHIVE_ROOTS
    """
    # config override
    roots = None
    if config_module is not None and hasattr(config_module, "archive_roots"):
        roots = getattr(config_module, "archive_roots")
    if roots:
        return list(roots)
    # env override
    env_val = os.environ.get("CMIP6_ARCHIVE_ROOTS")
    if env_val:
        parts = [p.strip() for p in env_val.split(",") if p.strip()]
        if parts:
            return parts
    return DEFAULT_ARCHIVE_ROOTS


def get_var_to_table_user(config_module=None):
    """Return variable->table mapping with user overrides.

    Order of precedence:
      1) config.var_to_table (dict)
      2) env CMIP6_VAR_TO_TABLE_JSON (JSON string)
      3) VAR_TO_TABLE_DEFAULT
    """
    mapping = None
    if config_module is not None and hasattr(config_module, "var_to_table"):
        try:
            user_map = dict(getattr(config_module, "var_to_table"))
            return {**VAR_TO_TABLE_DEFAULT, **user_map}
        except Exception:
            pass
    env_val = os.environ.get("CMIP6_VAR_TO_TABLE_JSON")
    if env_val:
        try:
            user_map = json.loads(env_val)
            if isinstance(user_map, dict):
                return {**VAR_TO_TABLE_DEFAULT, **user_map}
        except Exception:
            pass
    return VAR_TO_TABLE_DEFAULT


def build_manifest(
    variables,
    activity,
    institution,
    source_by_exp,
    variant,
    grid_label=None,
    version=None,
    years=None,
    roots=None,
    var_to_table=None,
):
    """Build a manifest dict of discovered files per variable and year.

    source_by_exp: dict mapping experiment -> source_id (e.g., {"historical":"ACCESS-ESM1-5","ssp370":"ACCESS-ESM1-5"}
    years: iterable of years to search (optional). If None, searches all years.
    var_to_table: optional mapping to override VAR_TO_TABLE_DEFAULT.
    """
    # If mapping not given, pick up user overrides from config module
    if var_to_table is None:
        try:
            from config import config as _cfg  # local import to avoid cycles
        except Exception:
            _cfg = None
        var_to_table = get_var_to_table_user(_cfg)
    else:
        var_to_table = {**VAR_TO_TABLE_DEFAULT, **(var_to_table or {})}
    # If roots not given, get user overrides
    if roots is None:
        try:
            from config import config as _cfg2
        except Exception:
            _cfg2 = None
        roots = get_archive_roots_user(_cfg2)
    manifest = {}
    for var in variables:
        table = var_to_table.get(var, None)
        manifest[var] = {}
        for experiment, source_id in source_by_exp.items():
            key = f"{experiment}:{source_id}"
            manifest[var][key] = {}
            if years is None:
                files = discover_cmip6_files(
                    var,
                    activity,
                    institution,
                    source_id,
                    experiment,
                    variant,
                    table_id=table,
                    grid_label=grid_label,
                    version=version,
                    year=None,
                    roots=roots,
                )
                manifest[var][key]["all"] = files
            else:
                for y in years:
                    files = discover_cmip6_files(
                        var,
                        activity,
                        institution,
                        source_id,
                        experiment,
                        variant,
                        table_id=table,
                        grid_label=grid_label,
                        version=version,
                        year=y,
                        roots=roots,
                    )
                    manifest[var][key][str(y)] = files
    return manifest


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
    Reformat and save a 3D variable from bias-corrected data, per original file.

    Strategy: for each original GCM file, open it as the template, find the
    intersection of its time coordinate with the bias-corrected dataset, align
    vertical coordinates and lat/lon (regrid for winds if needed), and replace
    the values for the overlapping times and spatial subset. Save one output per
    original input file, preserving the original filename inside out_path.

    Returns: list[(xarray.Dataset, output_path)] for audit; files are not written here.
    """

    results = []
    # Rename variable in bc dataset to match original, if needed
    bcf_rn = bcf if var_new == var_old else bcf.rename({var_new: var_old})

    for fp in input_files:
        # Open template (single file) to avoid cross-file concat issues
        ds_tmpl = xr.open_dataset(fp)

        # Identify vertical dimension name
        vdim = get_vertical_dim_name(ds_tmpl[var_old])

        # Determine overlapping times
        if "time" not in ds_tmpl.coords or "time" not in bcf_rn.coords:
            ds_tmpl.close()
            continue
        times_in = pd.to_datetime(ds_tmpl.time.values)
        times_bc = pd.to_datetime(bcf_rn.time.values)
        common = np.intersect1d(times_in, times_bc)
        if common.size == 0:
            # Nothing to replace in this file
            ds_tmpl.close()
            continue

        # Select overlapping time slices
        bc_slice = bcf_rn[var_old].sel(time=common)

        # Align vertical levels to template
        if vdim is not None and vdim in ds_tmpl:
            bc_slice = align_vertical_to_target(bc_slice, ds_tmpl[vdim], vdim)

        # Ensure canonical order
        desired_order = [
            d
            for d in ("time", vdim, "lat", "lon")
            if d is not None and d in bc_slice.dims
        ]
        bc_slice = bc_slice.transpose(*desired_order)

        # Spatial subset and regrid if necessary (winds often require exact grid match)
        tmpl_var = ds_tmpl[var_old]
        # If lat/lon mismatch, regrid bias-corrected slice to template grid for winds
        if var_old in ["ua", "va"] and (
            not bc_slice.lat.identical(tmpl_var.lat)
            or not bc_slice.lon.identical(tmpl_var.lon)
        ):
            print(
                f"Regridding {var_old} slice to match original grid for file {os.path.basename(fp)}"
            )
            rename_dict_reformat = {var_old: var_old}
            weight_path_reformat = os.path.join(
                out_path, f"weight_{config.gname}_{var_old}_reformat.nc"
            )
            bc_slice = regrid(
                bc_slice.to_dataset(name=var_old),
                tmpl_var.to_dataset(name=var_old),
                "bilinear",
                weight_path_reformat,
                rename_dict_reformat,
            )[var_old]

        # ------------------------------
        # Partial-domain safe replacement
        # ------------------------------
        # Bias-corrected file may cover only a sub-region (lat/lon subset). We must
        # restrict the assignment to intersecting coordinates to avoid shape mismatch.

        def _match_subset_coords(src_vals, target_vals, tol=1e-6):
            """Return coordinate values from src_vals that exist in target_vals within tol.

            Uses vectorized absolute difference; falls back to exact matching if tol == 0.
            Returns a numpy array of matched coordinate values preserving the order in src_vals.
            """
            src_vals = np.asarray(src_vals)
            target_vals = np.asarray(target_vals)
            matches = []
            for v in src_vals:
                # Find closest target coordinate
                j = int(np.argmin(np.abs(target_vals - v)))
                if abs(target_vals[j] - v) <= tol:
                    matches.append(target_vals[j])
                else:
                    # Coordinate outside tolerance – skip (not present in template)
                    pass
            return np.array(matches)

        # Determine intersecting lat/lon with a conservative tolerance (fraction of grid spacing)
        if bc_slice.lat.size > 1:
            dlat = float(np.min(np.abs(np.diff(bc_slice.lat.values))))
        else:
            dlat = float(np.min(np.abs(np.diff(tmpl_var.lat.values)))) if tmpl_var.lat.size > 1 else 1.0
        if bc_slice.lon.size > 1:
            dlon = float(np.min(np.abs(np.diff(bc_slice.lon.values))))
        else:
            dlon = float(np.min(np.abs(np.diff(tmpl_var.lon.values)))) if tmpl_var.lon.size > 1 else 1.0
        tol_lat = 0.05 * dlat
        tol_lon = 0.05 * dlon

        common_lat = _match_subset_coords(bc_slice.lat.values, tmpl_var.lat.values, tol=tol_lat)
        common_lon = _match_subset_coords(bc_slice.lon.values, tmpl_var.lon.values, tol=tol_lon)

        if common_lat.size == 0 or common_lon.size == 0:
            print(
                f"[WARN] No overlapping spatial coordinates between bias-corrected slice and template for file {os.path.basename(fp)}. Skipping spatial assignment."
            )
            ds_tmpl.close()
            continue

        # Reduce bc_slice to intersection only (preserving order relative to template)
        # We build an indexer dict for selecting from bc_slice by value.
        bc_slice_sub = bc_slice.sel(lat=common_lat, lon=common_lon)

        # Now perform replacement only on that sub-region
        data_var = tmpl_var.load()  # load to allow in-place numpy assignment

        # Build indexer for data_var.loc using overlapping coordinates & times
        loc_indexer = {"time": common, "lat": common_lat, "lon": common_lon}
        if vdim is not None and vdim in data_var.dims:
            # Align vertical coordinate already handled; require full level coverage match
            loc_indexer[vdim] = data_var[vdim]
            # Ensure bc_slice_sub has vdim; if missing (e.g., SST), expand
            if vdim not in bc_slice_sub.dims:
                bc_slice_sub = bc_slice_sub.expand_dims({vdim: data_var[vdim]})

        # Ensure ordering consistent with loc assignment (time, vdim?, lat, lon)
        ordered_dims = [d for d in ("time", vdim, "lat", "lon") if d and d in bc_slice_sub.dims]
        bc_slice_sub = bc_slice_sub.transpose(*ordered_dims)

        # Safety: shapes must align except for excluded template regions
        expected_shape = tuple(len(loc_indexer[d]) for d in ordered_dims)
        if bc_slice_sub.values.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch after intersection for {var_old} in file {os.path.basename(fp)}: "
                f"bc_slice_sub {bc_slice_sub.values.shape} vs expected {expected_shape}."
            )

        # Assignment (only subset replaced)
        data_var.loc[loc_indexer] = bc_slice_sub.values
        data_var.attrs["history"] = "Bias-corrected and reformatted data"
        ds_tmpl[var_old] = data_var

        # Clean encodings for known bound/meta variables to avoid fill value issues
        encoding_vars = ["lev_bnds", "b", "orog", "b_bnds", "lat_bnds", "lon_bnds"]
        for v in encoding_vars:
            if v in ds_tmpl.variables:
                ds_tmpl[v].encoding["_FillValue"] = None

        # Output path mirrors original filename
        output_path = os.path.join(out_path, os.path.basename(fp))
        results.append((ds_tmpl, output_path))

    return results


def reformat_and_save_3d(
    config,
    bc_path,
    tlevel,
    startyear,
    endyear,
    input_vargcm,
    origin_vargcm,
    manifest=None,  # added
    manifest_key=None,  # added (e.g., "historical:ACCESS-ESM1-5")
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
        sorted(glob.glob(os.path.join(bc_path, f"bc_corrected_3d_lev_{idx}_*.nc")))
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
    # Track successful outputs for optional cleanup
    produced_files = []
    for k in range(len(input_vargcm)):
        for y in range(startyear, endyear + 1):
            year = str(y)
            # Prefer manifest if supplied
            if manifest is not None and manifest_key is not None:
                # Try origin name first, then input name
                var_key = (
                    origin_vargcm[k]
                    if origin_vargcm[k] in manifest
                    else input_vargcm[k]
                )
                files_by_group = manifest.get(var_key, {}).get(manifest_key, {})
                input_files = files_by_group.get(year, [])
            else:
                # Fallback to previous glob behaviour
                input_files = sorted(
                    glob.glob(
                        f"{config['bc_hist_path']}/{origin_vargcm[k]}/{config['sinfor']}/v{config['version']}/{origin_vargcm[k]}_*_{year}*.nc"
                    )
                )

            if not input_files:
                print(f"[WARN] No template files for {origin_vargcm[k]} {year}")
                continue

            results = reformatsave_3d(
                bcf,
                input_vargcm[k],
                origin_vargcm[k],
                y,
                input_files,
                config["out_path"],
            )
            # Save data
            for dataset, output_filename in results:
                print(f"Save 3d to netcdf {output_filename}")
                # Apply compression (default level 5 or override via config['compression_level'])
                # comp_level = config.get("compression_level", 5)
                comp_level = 5
                encoding = {}
                for v in dataset.data_vars:
                    # Preserve existing encoding but enforce compression
                    enc_prev = dict(dataset[v].encoding) if hasattr(dataset[v], "encoding") else {}
                    # Remove automatic chunk size hints that can conflict
                    enc_prev.pop("chunksizes", None)
                    encoding[v] = {**enc_prev, "zlib": True, "complevel": comp_level}
                # Coordinates: avoid attempting to compress scalar/index coords unnecessarily
                dataset.load().to_netcdf(output_filename, encoding=encoding)
                print(f"Completed {output_filename}")
                produced_files.append(output_filename)

    # Optional cleanup: remove bias-corrected input tiles if configured
    if config.get("cleanup_bc_inputs", False):
        try:
            # Remove only the lev files we actually opened (ignore missing quietly)
            for idx in range(0, tlevel + 1):
                pattern = os.path.join(bc_path, f"bc_corrected_3d_lev_{idx}_*.nc")
                for f in glob.glob(pattern):
                    if os.path.isfile(f):
                        try:
                            os.remove(f)
                            print(f"[CLEANUP] Removed {f}")
                        except Exception as e_rm:
                            print(f"[CLEANUP WARN] Could not remove {f}: {e_rm}")
        except Exception as e:
            print(f"[CLEANUP ERROR] Cleanup encountered an issue: {e}")


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


def reformat_and_save_2d(config, input_path, original_file, output_file, remap_weights_file):
    """Integrate a (possibly partial-domain) 2D bias-corrected field into the
    original file's grid.

    Logic:
      1. Open bias-corrected dataset (bc_ds) & template (tmpl_ds).
      2. If bc lat/lon already subset of template and identical points, copy only subset.
      3. Else (grids differ / need interpolation), perform CDO regrid on the full field.
      4. Write final merged dataset to output_file with compression.

    Args:
        input_path: path to bias-corrected 2D nc (may be subset spatially)
        original_file: path to original reference nc (full domain)
        output_file: final merged output path
        remap_weights_file: path for CDO weights (if regridding is required)
    """
    varname = None
    with xr.open_dataset(input_path) as bc_ds, xr.open_dataset(original_file) as tmpl_ds:
        # Guess variable (first non-coordinate var)
        cand_vars = [v for v in bc_ds.data_vars if v not in bc_ds.coords]
        if not cand_vars:
            raise ValueError(f"No data variables found in {input_path}")
        varname = cand_vars[0]

        # Coordinate name detection
        def _detect_coord(ds, candidates):
            for name in candidates:
                if name in ds.coords:
                    return name
            # try case-insensitive match
            lower_map = {c.lower(): c for c in ds.coords}
            for name in candidates:
                if name.lower() in lower_map:
                    return lower_map[name.lower()]
            return None

        lat_candidates = ["lat", "latitude", "Latitude", "nav_lat", "y"]
        lon_candidates = ["lon", "longitude", "Longitude", "nav_lon", "x"]
        bc_lat_name = _detect_coord(bc_ds, lat_candidates)
        bc_lon_name = _detect_coord(bc_ds, lon_candidates)
        tmpl_lat_name = _detect_coord(tmpl_ds, lat_candidates)
        tmpl_lon_name = _detect_coord(tmpl_ds, lon_candidates)
        if not all([bc_lat_name, bc_lon_name, tmpl_lat_name, tmpl_lon_name]):
            raise ValueError(
                "Could not detect latitude/longitude coordinate names in one of the datasets. "
                f"Detected -> bc: ({bc_lat_name}, {bc_lon_name}), tmpl: ({tmpl_lat_name}, {tmpl_lon_name})"
            )

        # Quick path: identical full grid (simple replace)
        same_lat = bc_ds[bc_lat_name].identical(tmpl_ds[tmpl_lat_name])
        same_lon = bc_ds[bc_lon_name].identical(tmpl_ds[tmpl_lon_name])
        partial_subset = False
        if not (same_lat and same_lon):
            # Check if bc is a strict subset with matching coordinate values
            bc_lat = bc_ds[bc_lat_name].values
            bc_lon = bc_ds[bc_lon_name].values
            tmpl_lat = tmpl_ds[tmpl_lat_name].values
            tmpl_lon = tmpl_ds[tmpl_lon_name].values
            subset_lat = np.isin(bc_lat, tmpl_lat)
            subset_lon = np.isin(bc_lon, tmpl_lon)
            if subset_lat.all() and subset_lon.all():
                partial_subset = True
            else:
                # Need regridding
                print("[2D] Grids differ; performing regrid via CDO ...")
                input_with_bnds = input_path.replace(".nc", "_with_bnds.nc")
                add_lat_lon_bnds(input_path, input_with_bnds)
                cdo.genbil(original_file, input=input_with_bnds, output=remap_weights_file)
                cdo.remap(
                    f"{original_file},{remap_weights_file}",
                    input=input_with_bnds,
                    output=output_file,
                )
                print(f"[2D] Regridding completed: {input_with_bnds} -> {output_file}")
                return

        # Build output dataset starting from template
        out = tmpl_ds.load()  # ensure writeable
        if same_lat and same_lon:
            print("[2D] Full-domain replacement.")
            out[varname].values = bc_ds[varname].values
        elif partial_subset:
            print("[2D] Partial subset merge.")
            # Determine indices of bc coords inside template
            lat_map = {val: i for i, val in enumerate(out[tmpl_lat_name].values)}
            lon_map = {val: j for j, val in enumerate(out[tmpl_lon_name].values)}
            lat_idx = [lat_map[v] for v in bc_ds[bc_lat_name].values]
            lon_idx = [lon_map[v] for v in bc_ds[bc_lon_name].values]
            # Broadcast assignment respecting time if present
            if "time" in out[varname].dims and "time" in bc_ds[varname].dims:
                out_vals = out[varname].values
                bc_vals = bc_ds[varname].values
                # Expect shape (time, lat_subset, lon_subset) or (lat_subset, lon_subset)
                if out_vals.ndim == 3:  # (time, lat, lon)
                    for ii, li in enumerate(lat_idx):
                        for jj, lj in enumerate(lon_idx):
                            out_vals[:, li, lj] = bc_vals[:, ii, jj]
                elif out_vals.ndim == 2:  # (lat, lon)
                    for ii, li in enumerate(lat_idx):
                        for jj, lj in enumerate(lon_idx):
                            out_vals[li, lj] = bc_vals[ii, jj]
                else:
                    raise ValueError(
                        f"Unexpected dimensionality for variable {varname}: {out_vals.shape}"
                    )
                out[varname].values = out_vals
            else:
                # No time dimension present
                for ii, li in enumerate(lat_idx):
                    for jj, lj in enumerate(lon_idx):
                        out[varname].values[li, lj] = bc_ds[varname].values[ii, jj]
        else:
            # Should not reach here
            raise RuntimeError("Unhandled 2D merging path")

        # Compression on write
        comp_level = 5
        encoding = {
            varname: {"zlib": True, "complevel": comp_level}
        }
        out.to_netcdf(output_file, encoding=encoding)
        print(f"[2D] Merged output written: {output_file}")

    # Optional cleanup of the bias-corrected 2D input file & intermediate with_bnds
    if config.get("cleanup_bc_inputs", False):
        try:
            if os.path.isfile(input_path):
                os.remove(input_path)
                print(f"[CLEANUP] Removed 2D bias-corrected input {input_path}")
            with_bnds = input_path.replace(".nc","_with_bnds.nc")
            if os.path.isfile(with_bnds):
                os.remove(with_bnds)
                print(f"[CLEANUP] Removed temporary {with_bnds}")
        except Exception as e:
            print(f"[CLEANUP WARN] 2D cleanup issue: {e}")


# ---------------------------------------------------------------------------------------------------
def main(config_path, override_ncpus=None, override_mem=None):
    """
    Main function to initiate the reformatting process for bias-corrected GCM data.

    Args:
        config (module): Configuration object that contains user-defined parameters such as
        start year, end year, output paths, etc.

    Returns:
        None: Initiates the reformatting for both 2D and 3D data based on the given configuration.
    """

    # Load YAML config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resources
    ncpus = override_ncpus or cfg.get("resources", {}).get("ncpus") or 4
    mem_gb = override_mem or cfg.get("resources", {}).get("mem_gb") or 16

    # Dask performance settings
    dask.config.set(
        {
            "array.slicing.split_large_chunks": True,
            "array.chunk-size": "64MiB",
            "optimization.fuse.active": True,
        }
    )

    # Start Dask client
    client = setup_client(ncpus, mem_gb)

    print(f"[INFO] CMIP6 root: {cfg['target_path']} (table={cfg['gname']})")

    # Inputs from YAML
    out_path = cfg["out_path"]
    tlevel = cfg.get("tlevel", len(cfg.get("target_variable", [])))  # fallback
    target_variable = cfg["target_variable"]
    bc_boundary = cfg["bc_boundary"]

    # User overrides for archive search
    roots = cfg.get("archive_roots", DEFAULT_ARCHIVE_ROOTS)
    vmap = {**VAR_TO_TABLE_DEFAULT, **cfg.get("var_to_table", {})}

    # Historical identifiers
    startyear_h = cfg["startyear_h"]
    endyear_h = cfg["endyear_h"]
    activity_h, institution_h, source_id_h = infer_cmip6_ids_from_path(
        cfg["bc_hist_path"]
    )
    if not all([activity_h, institution_h, source_id_h]):
        raise ValueError(
            f"Could not infer CMIP6 IDs from bc_hist_path: {cfg['bc_hist_path']}"
        )

    # Build manifest for historical only (extend similarly for future if needed)
    manifest_hist = build_manifest(
        variables=target_variable,
        activity=activity_h,
        institution=institution_h,
        source_by_exp={"historical": source_id_h},
        variant=cfg["cinfor"],
        grid_label=cfg["sinfor"],
        version=f"v{cfg['version']}",
        years=range(startyear_h, endyear_h + 1),
        roots=roots,
        var_to_table=vmap,
    )
    manifest_key_hist = f"historical:{source_id_h}"

    print("Start reformatting")
    if bc_boundary == "lateral":
        reformat_and_save_3d(
            cfg,
            bc_path=out_path,
            tlevel=tlevel,
            startyear=startyear_h,
            endyear=endyear_h,
            input_vargcm=target_variable,
            origin_vargcm=target_variable,
            manifest=manifest_hist,
            manifest_key=manifest_key_hist,
        )
        print("Finish 3D reformatting")
    else:
        # 2D path remains as-is (can be wired to manifest similarly if desired)
        target_files = sorted(
            glob.glob(
                f"{cfg['bc_hist_path']}/{target_variable[0]}/{cfg['sinfor']}/v{cfg['version']}/{target_variable[0]}_*.nc"
            )
        )
        input_files = sorted(
            glob.glob(
                os.path.join(
                    out_path,
                    f"bc_corrected_2d_{cfg['infor']}_{cfg['gname']}_{cfg['period']}_{cfg['cinfor']}_{cfg['sinfor']}_{startyear_h}_{endyear_h}.nc",
                )
            )
        )
        remap_weights_file = f"{out_path}/remap_weights_{target_variable[0]}_{startyear_h}_{endyear_h}.nc"
        output_file = f"{out_path}/{target_variable[0]}_{cfg['infor']}_{cfg['gname']}_{cfg['period']}_{cfg['cinfor']}_{cfg['sinfor']}_{startyear_h}0101-{endyear_h}1231.nc"
        reformat_and_save_2d(
            cfg, input_files[0], target_files[0], output_file, remap_weights_file
        )
        print("Finish 2D reformatting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yp", required=True, help="Path to YAML config")
    parser.add_argument(
        "--ncpus", type=int, default=None, help="Override number of CPUs"
    )
    parser.add_argument("--mem", type=int, default=None, help="Override memory in GB")

    args = parser.parse_args()
    main(args.yp, args.ncpus, args.mem)
