#!/usr/bin/env python3
"""Generate base 3D BC model-as-truth configs for all truth GCMs x periods.

Fills in src/config_templates/bc_3d_model_as_truth_template.yaml once per
(truth GCM, period) combination, using the same MODELS/TARGET metadata as
generate_model_as_truth_configs.py (see model_as_truth_registry.py) so the
two generators can't drift apart.

Output is a *base* config per experiment -- still whole-domain, single
level (slevel=elevel=0). Fan it out per vertical level with
generate_bc_level_configs.py before submitting.
"""

import argparse
from pathlib import Path

import yaml

from model_as_truth_registry import (
    HIST_PERIOD,
    FUTURE_PERIOD,
    MODELS,
    ROOT,
    TARGET,
    interp_output_path,
)

TEMPLATE_PATH = ROOT / "config_templates" / "bc_3d_model_as_truth_template.yaml"

OUTPUT_ROOT = Path("/g/data/w28/yk8692/output/model_as_truth")
TEMP_ROOT = Path("/scratch/n81/yk8692/sdmbc_cache/model_as_truth")

# BC resources are tuned separately from interpolation's (see
# submit_bc_3d.sh defaults); do not reuse model_as_truth_registry.RESOURCES,
# which is interpolation-specific (48 CPU/190GB).
BC_RESOURCES = {"ncpus": 8, "mem_gb": 64}

# Validated no-poles domain from TROUBLESHOOTING_BC_MODEL_AS_TRUTH.md --
# exact poles (lat=-90/90) contain invalid interpolated reference values.
DOMAIN = {
    "lat_min": -88.75,
    "lat_max": 88.75,
    "lon_min": 0.0,
    "lon_max": 358.125,
}

CHANGE_ME_PREFIX = "CHANGE_ME"


def load_template() -> dict:
    with TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_bc_config(model_name: str, meta: dict, scenario: str) -> dict:
    period_dir = "hist" if scenario == "historical" else "ssp126_2080_2100"
    label = meta["label"]
    target_hist = TARGET["historical"]
    target_future = TARGET["ssp126"]

    config = load_template()
    config.update(
        {
            "bc_hist_path": target_hist["target_path"],
            "bc_future_path": target_future["target_path"],
            "obs_path": str(interp_output_path(label, "historical", "3d")),
            "out_path": str(OUTPUT_ROOT / f"{label}_to_access" / f"bc_{period_dir}_3d_model_as_truth"),
            "temp_root": str(TEMP_ROOT / f"{label}_to_access" / f"bc_{period_dir}_3d_model_as_truth"),
            "bc_hist": True,
            "bc_future": scenario != "historical",
            "startyear_h": HIST_PERIOD["startyear_h"],
            "endyear_h": HIST_PERIOD["endyear_h"],
            "startyear_f": FUTURE_PERIOD["startyear_h"],
            "endyear_f": FUTURE_PERIOD["endyear_h"],
            "infor": target_hist["infor"],
            "gname": target_hist["gname"],
            "period": target_hist["period"],
            "cinfor": target_hist["cinfor"],
            "sinfor": target_hist["sinfor"],
            "version": target_hist["version"],
            "scenario": target_future["period"],
            "cinfor_f": target_future["cinfor"],
            "version_f": target_future["version"],
            "period_f": target_future["period"],
            "truth_gname": model_name,
            "single_lat": None,
            "single_lon": None,
            "lat_min": DOMAIN["lat_min"],
            "lat_max": DOMAIN["lat_max"],
            "lon_min": DOMAIN["lon_min"],
            "lon_max": DOMAIN["lon_max"],
            "resources": dict(BC_RESOURCES),
        }
    )

    unresolved = [
        key
        for key, value in config.items()
        if isinstance(value, str) and value.startswith(CHANGE_ME_PREFIX)
    ]
    if unresolved:
        raise ValueError(
            f"Unresolved {CHANGE_ME_PREFIX}_* placeholder(s) in generated config "
            f"for {model_name}/{scenario}: {unresolved}"
        )
    return config


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, default_flow_style=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output paths and counts without writing files.",
    )
    parser.add_argument(
        "--truth-gcm",
        choices=sorted(MODELS),
        default=None,
        help="Generate only this truth GCM's config(s) instead of all three.",
    )
    parser.add_argument(
        "--period",
        choices=["historical", "ssp126"],
        default=None,
        help="Generate only this period's config instead of both.",
    )
    args = parser.parse_args()

    models = {args.truth_gcm: MODELS[args.truth_gcm]} if args.truth_gcm else MODELS
    periods = (
        [(args.period, "hist" if args.period == "historical" else "ssp126_2080_2100")]
        if args.period
        else [("historical", "hist"), ("ssp126", "ssp126_2080_2100")]
    )

    planned = []
    for model_name, meta in models.items():
        label = meta["label"]
        for scenario, period_dir in periods:
            out_file = ROOT / f"config_bc_3d_{label}_to_access_{period_dir}_model_as_truth.yaml"
            planned.append((model_name, scenario, out_file))

    print(f"[INFO] {len(planned)} base BC config(s) planned "
          f"({len(models)} truth GCM(s) x {len(periods)} period(s))")
    for model_name, scenario, out_file in planned:
        if args.dry_run:
            print(f"DRY_RUN: would write {out_file} ({model_name}, {scenario})")
            continue
        config = build_bc_config(model_name, MODELS[model_name], scenario)
        write_yaml(out_file, config)
        print(out_file)


if __name__ == "__main__":
    main()
