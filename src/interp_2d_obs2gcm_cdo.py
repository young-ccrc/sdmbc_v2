"""
This script performs horizontal interpolation of two dimensional atmospheric variables from ERA5 reanalysis data to a target Global Climate Model (GCM) dataset. The interpolated data is saved as NetCDF files.

The script takes input variables from the ERA5 reanalysis data, sea surface temperature (tos), and maps it to the target variable in the GCM dataset, which is sea surface temperature (sst).

The horizontal interpolation is performed using xESMF library, which uses conservative or bilinear methods depending on the input variable.

The script loops through each year and month specified in the input parameters and performs the interpolation for each variable. The interpolated data is then saved as NetCDF files.

The script uses Dask for parallel processing and xarray for data manipulation. The dask.distributed.Client is used to create a local cluster for parallel processing.

The script is designed to be run on the NCI's Gadi supercomputer, but can be modified to run on other systems.

Note: This script assumes that the necessary input files and directories are available and properly formatted.

Written by: Youngil (Young) Kim, CLEX, CCRC, UNSW
Contact: youngil.kim@unsw.edu.au
"""

import argparse
import glob
import os
import re

# Initialize a Dask client, optimally configured for your environment
import warnings
from pathlib import Path

import xarray as xr  # type: ignore
import xesmf as xe  # type: ignore
import yaml  # type: ignore
from cdo import Cdo
from dask.diagnostics import ProgressBar  # type: ignore
from dask.distributed import Client  # type: ignore

cdo = Cdo()
warnings.simplefilter("ignore", UserWarning)


# Define functions for the horizontal and vertical interpolation ----------------
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
    return parser.parse_args()


def setup_client(n_workers=None):
    """
    Set up Dask client for parallel processing, allowing customization of workers and threads.

    Args:
        n_workers (int): Number of workers to use.
        threads_per_worker (int): Number of threads per worker.

    Returns:
        Client: A Dask distributed client instance.
    """
    if n_workers is None:
        c = Client()
    else:
        c = Client(n_workers=n_workers)
    print("Dask client setup complete.")
    return c


def correct_latitudes(ds):
    # Ensure latitudes are within bounds
    ds["latitude"] = ds["latitude"].clip(-90, 90)
    return ds


# Horizontal interpolation
def regrid(source_ds, target_ds, method, weights_path, rename_dict):
    """
    Perform interpolation/regridding of a source dataset to a target dataset using the specified method.

    Args:
        source_ds (xarray.Dataset): The source dataset to be regridded.
        target_ds (xarray.Dataset): The target dataset with the desired grid.
        method (str): The interpolation method to be used. Supported methods: 'bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch', 'nearest_s2d', 'nearest_d2s', 'patch'.
        weights_path (str): The path to the weights file used for regridding. If the file exists, the weights will be reused. If not, the weights will be computed and saved to this path.
        rename_dict (dict): A dictionary mapping variable names in the source dataset to the desired variable names in the regridded dataset.

    Returns:
        xarray.Dataset: The regridded dataset with variables renamed according to the provided rename_dict.

    """
    weights_file = Path(weights_path)
    reuse_weights = weights_file.is_file()
    regridder = xe.Regridder(
        source_ds,
        target_ds,
        method=method,
        filename=weights_path,
        reuse_weights=reuse_weights,
    )
    regridded_ds = regridder(source_ds).rename(rename_dict)
    return regridded_ds


def standardize_coords_from(ds):
    coord_map = {"longitude": "lon", "latitude": "lat"}
    new_ds = ds.copy()
    for old_name, new_name in coord_map.items():
        if old_name in new_ds.coords:
            new_ds = new_ds.rename({old_name: new_name})
    return new_ds


# Function to extract the year range from the filename
def extract_years(filename):
    match = re.search(r"_(\d{8})-(\d{8})\.nc", filename)
    if match:
        start = int(match.group(1)[:4])
        end = int(match.group(2)[:4])
        return start, end
    return None, None


# Function to check if the file overlaps with the desired period
def is_within_period(file_start, file_end, target_start, target_end):
    # True if file period overlaps with the target period
    return not (file_end < target_start or file_start > target_end)


# End of the functions -------------------------------------------------------
def main(config):

    client = setup_client()

    # Define start and end years
    start_year = config["startyear_h"]
    end_year = config["endyear_h"]
    target_path = config["target_path_sst"]
    input_path = config["input_path_sst"]
    output_path = config["output_path"]
    input_model = config["input_model"]
    sinfor = config["sinfor"]
    version = config["version"]
    gname = config["gname"]
    if input_model != "reanalysis":
        input_gname = config["input_gname"]
        input_sinfor = config["input_sinfor"]
        input_version = config["input_version"]
    target_2d = config["var_interp"]

    if input_model == "reanalysis":
        input_2d = "sst"
    else:
        input_2d = target_2d

    # Ensure output directory exists
    os.makedirs(config["output_path"], exist_ok=True)

    # sst
    # ds = xr.open_mfdataset(target_files, combine="by_coords")

    # Filter files that overlap with the desired year range
    # GCM
    # Load the target dataset
    target_files = sorted(
        glob.glob(f"{target_path}/Oday/{target_2d}/{sinfor}/v*/{target_2d}_*")
    )

    filtered_files = [
        f
        for f in target_files
        if is_within_period(*extract_years(f), start_year, end_year)
    ]
    # print(filtered_files)
    target_latlon_grid = glob.glob(f"{target_path}/fx/orog/{sinfor}/v*/orog_*.nc")

    for file in filtered_files:
        output_file = os.path.join(
            output_path, os.path.basename(file).replace(".nc", "_remapped.nc")
        )
        # Regrid the dataset to lat/lon grid
        print(f"Regridding variable {target_2d} to lat/lon...")
        cdo.remapbil(target_latlon_grid[0], input=file, output=output_file)
        print(f"Remapped: {file} -> {output_file}")

    # Observation
    if input_model == "reanalysis":
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                with xr.open_dataset(target_latlon_grid[0]) as temp_ds:
                    chunks = {dim: "auto" for dim in temp_ds.dims}
                ds = xr.open_mfdataset(
                    target_latlon_grid[0], combine="by_coords", chunks=chunks
                )
                # ds = xr.decode_cf(ds)
                # selected_ds = ds.sel(
                #     time=(ds["time"].dt.year == year) & (ds["time"].dt.month == month)
                # )

                # if selected_ds.sizes["time"] == 0:
                #     raise ValueError(
                #         f"No data available for {year}-{month} in variable {target_var}."
                #     )

                for input_var, target_var in zip([input_2d], [target_2d]):
                    method = "bilinear"
                    input_files_pattern = glob.glob(
                        f"{input_path}/{input_var}/{year}/{input_var}_*_{year}{month:02d}*.nc"
                    )

                    weights_path = f"{output_path}/weights_{input_model if input_model == 'reanalysis' else input_gname}_to_{gname}_{target_var}_{method}.nc"
                    output_file = f"{output_path}/{target_var}_{input_model}_to_{gname}_{year}-{month:02}.nc"

                    with xr.open_dataset(input_files_pattern[0]) as temp_ds:
                        chunks = {dim: "auto" for dim in temp_ds.dims}
                    do = xr.open_mfdataset(
                        input_files_pattern, combine="by_coords", chunks=chunks
                    )
                    do = standardize_coords_from(do)
                    do_corrected = do.sel(
                        time=(do["time"].dt.year == year)
                        & (do["time"].dt.month == month)
                    )
                    do_corrected = do_corrected.resample(time="D").mean()

                    rename_dict = {input_var: target_var}
                    do_regridded = regrid(
                        do_corrected, ds, method, weights_path, rename_dict
                    )
                    sliced_do_per = do_regridded.persist()
                    print(
                        f"Horizontal interpolation for {input_var} {year}-{month:02} complete."
                    )

                    # coord_map = {"i": ["j", "i"], "x": ["y", "x"]}
                    # data_vars = sliced_do_per.coords
                    # dim_order = coord_map["i"] if "i" in data_vars else coord_map["x"]

                    interpolated_da = xr.DataArray(
                        sliced_do_per[target_var],
                        dims=["time", "lat", "lon"],
                        coords={
                            dim: sliced_do_per[dim] for dim in ["time", "lat", "lon"]
                        },
                        name=target_var,
                    )

                    interpolated_da.attrs = ds.attrs.copy()
                    interpolated_da.attrs["interpolation_to"] = f"{gname}"
                    write_job = interpolated_da.to_netcdf(output_file, compute=False)
                    with ProgressBar():
                        print(f"Writing to {output_file}")
                        write_job.compute()
                    print(
                        f"Saved regridded data for {target_var} {year}-{month:02} to {output_file}"
                    )
                print(f"Interpolation for year: {year} month: {month:02} complete")
    else:
        # Filter files that overlap with the desired year range
        # GCM
        # Load the target dataset
        input_files = sorted(
            glob.glob(
                f"{input_path}/Oday/{input_2d}/{input_sinfor}/{input_version}/{input_2d}_*"
            )
        )
        filtered_files = [
            f
            for f in input_files
            if is_within_period(*extract_years(f), start_year, end_year)
        ]

        for file in filtered_files:
            output_file = os.path.join(
                output_path, os.path.basename(file).replace(".nc", "_remapped.nc")
            )
            # Regrid the dataset to lat/lon grid
            print(f"Regridding variable {input_2d} to lat/lon...")
            cdo.remapbil(target_latlon_grid[0], input=file, output=output_file)
            print(f"Remapped: {file} -> {output_file}")


print("Interpolation complete.")

if __name__ == "__main__":
    args = parse_arguments()
    config = load_config(args.yp)
    config["var_interp"] = args.var
    main(config)
    print("All done!")
