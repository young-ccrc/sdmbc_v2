#!/usr/bin/env python3
"""Quick validation for interpolated outputs before SDMBCv2 runs."""

import argparse
import glob
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

GROUPED_SECTIONS = ("paths", "source", "target", "domain", "period", "resources", "future")


def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config_data = yaml.safe_load(file) or {}

    merged = dict(config_data)
    for section in GROUPED_SECTIONS:
        section_data = config_data.get(section)
        if isinstance(section_data, dict):
            merged.update(section_data)
    return merged


def month_range(start_year, end_year):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def expected_output_path(config, var, year, month):
    source_name = config["input_model"] if config["input_model"] == "reanalysis" else config["input_gname"]
    name = f"{var}_{source_name}_to_{config['gname']}_{year}-{month:02}.nc"
    return Path(config["output_path"]) / name


def find_target_files(config, var, year):
    pattern = f"{config['target_path']}/{config['infor']}/{var}/g*/v*/{var}_*{year}*.nc"
    files = []
    for f in glob.glob(pattern):
        base = Path(f).name
        if f"_{config['infor']}_" not in base:
            continue
        if config.get("sinfor") and f"_{config['sinfor']}_" not in base:
            continue
        files.append(f)
    return sorted(files)


def open_target_month(config, var, year, month):
    files = find_target_files(config, var, year)
    if not files:
        raise FileNotFoundError(f"No target files found for {var} {year}")
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
    if "time" in ds.coords:
        ds = ds.sel(time=(ds["time"].dt.year == year) & (ds["time"].dt.month == month))
    return ds


def same_coord(a, b):
    return a.equals(b)


def validate_file(config, var, year, month, output_path):
    issues = []
    ds = xr.open_dataset(output_path, engine="netcdf4")
    try:
        if var not in ds.data_vars:
            issues.append(f"missing variable {var}")
            return issues

        da = ds[var]
        if "time" not in da.dims:
            issues.append("missing time dimension")
            return issues
        if da.sizes["time"] == 0:
            issues.append("empty time dimension")
            return issues

        times = pd.DatetimeIndex(da.time.values)
        if not times.is_monotonic_increasing:
            issues.append("time not monotonic increasing")
        if times.has_duplicates:
            issues.append("duplicate timestamps present")

        allowed_hours = {0, 6, 12, 18}
        hours = set(times.hour.tolist())
        if not hours.issubset(allowed_hours):
            issues.append(f"unexpected timestep hours: {sorted(hours)}")

        target_ds = open_target_month(config, var, year, month)
        try:
            target_da = target_ds[var]

            if da.sizes["time"] != target_da.sizes["time"]:
                issues.append(
                    f"time length mismatch: output={da.sizes['time']} target={target_da.sizes['time']}"
                )
            elif not same_coord(da.time, target_da.time):
                issues.append("time coordinate mismatch with target")

            for coord in ("lat", "lon"):
                if coord not in da.coords or coord not in target_da.coords:
                    issues.append(f"missing {coord} coordinate")
                    continue
                if not same_coord(da[coord], target_da[coord]):
                    issues.append(f"{coord} coordinate mismatch with target")

            if "lev" in da.coords and "lev" in target_da.coords:
                if not same_coord(da["lev"], target_da["lev"]):
                    issues.append("lev coordinate mismatch with target")
        finally:
            target_ds.close()

        if bool(da.isel(time=0).isnull().all()):
            issues.append("first timestep is fully NaN")
    finally:
        ds.close()

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate interpolated monthly outputs against target structure.")
    parser.add_argument("--config", required=True, help="Path to interpolation YAML config")
    parser.add_argument("--var", nargs="+", required=True, help="Variables to validate")
    args = parser.parse_args()

    config = load_config(args.config)
    start_year = config["startyear_h"]
    end_year = config["endyear_h"]

    any_issue = False
    for var in args.var:
        missing = []
        checked = 0
        print(f"[INFO] Validating {var} from {start_year} to {end_year}")
        for year, month in month_range(start_year, end_year):
            output_path = expected_output_path(config, var, year, month)
            if not output_path.exists():
                missing.append(output_path.name)
                continue

            issues = validate_file(config, var, year, month, output_path)
            checked += 1
            if issues:
                any_issue = True
                print(f"[FAIL] {output_path.name}")
                for issue in issues:
                    print(f"  - {issue}")

        if missing:
            any_issue = True
            print(f"[FAIL] Missing {len(missing)} files for {var}")
            for name in missing[:12]:
                print(f"  - {name}")
            if len(missing) > 12:
                print(f"  - ... {len(missing) - 12} more")
        else:
            print(f"[PASS] All expected monthly files exist for {var}")

        print(f"[INFO] Checked {checked} monthly files for {var}")

    if any_issue:
        raise SystemExit(1)
    print("[PASS] Validation completed without issues.")


if __name__ == "__main__":
    main()
