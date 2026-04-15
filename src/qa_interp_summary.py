#!/usr/bin/env python3
"""Fast QA summary for interpolated monthly outputs."""

import argparse
from pathlib import Path

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


def source_name(config):
    return config["input_model"] if config["input_model"] == "reanalysis" else config["input_gname"]


def expected_output_path(config, var, year, month):
    name = f"{var}_{source_name(config)}_to_{config['gname']}_{year}-{month:02}.nc"
    return Path(config["output_path"]) / name


def inspect_file(path, var):
    issues = []
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if var not in ds.data_vars:
                issues.append(f"missing variable {var}")
                return issues
            da = ds[var]
            if "time" not in da.dims:
                issues.append("missing time dimension")
                return issues
            if da.sizes.get("time", 0) == 0:
                issues.append("empty time dimension")
                return issues
            if bool(da.isel(time=0).isnull().all()):
                issues.append("first timestep fully NaN")
    except Exception as exc:
        issues.append(f"open/read failed: {exc}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Fast QA summary for interpolated monthly outputs")
    parser.add_argument("--config", required=True, help="Path to interpolation YAML config")
    parser.add_argument("--var", nargs="+", required=True, help="Variables to inspect")
    parser.add_argument("--sy", type=int, default=None, help="Optional start year override")
    parser.add_argument("--ey", type=int, default=None, help="Optional end year override")
    args = parser.parse_args()

    config = load_config(args.config)
    start_year = args.sy if args.sy is not None else config["startyear_h"]
    end_year = args.ey if args.ey is not None else config["endyear_h"]

    any_issue = False
    for var in args.var:
        expected = 0
        existing = 0
        healthy = 0
        missing = []
        bad = []

        for year, month in month_range(start_year, end_year):
            expected += 1
            output_path = expected_output_path(config, var, year, month)
            if not output_path.exists():
                missing.append(output_path.name)
                continue
            existing += 1
            issues = inspect_file(output_path, var)
            if issues:
                bad.append((output_path.name, issues))
            else:
                healthy += 1

        print(f"[INFO] {var}: expected={expected} existing={existing} healthy={healthy}")
        if missing:
            any_issue = True
            print(f"[FAIL] Missing {len(missing)} files for {var}")
            for name in missing[:12]:
                print(f"  - {name}")
            if len(missing) > 12:
                print(f"  - ... {len(missing) - 12} more")
        if bad:
            any_issue = True
            print(f"[FAIL] Found {len(bad)} invalid files for {var}")
            for name, issues in bad[:12]:
                print(f"  - {name}: {'; '.join(issues)}")
            if len(bad) > 12:
                print(f"  - ... {len(bad) - 12} more")
        if not missing and not bad:
            print(f"[PASS] Quick QA passed for {var}")

    raise SystemExit(1 if any_issue else 0)


if __name__ == "__main__":
    main()
