#!/usr/bin/env python3
"""Generate per-level 3D BC configs from one validated base config."""

import argparse
import copy
from pathlib import Path
from typing import List, Optional

import yaml


def parse_levels(value: str) -> List[int]:
    levels = []  # type: List[int]
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            levels.extend(range(start, end + step, step))
        else:
            levels.append(int(part))

    if not levels:
        raise argparse.ArgumentTypeError("at least one level is required")
    return levels


def build_config(
    base_config: dict,
    *,
    level: int,
    output_root: Path,
    temp_root: Path,
    suffix: str,
    run_prefix: str,
    dask_cell_batch_size: int,
    ncpus: Optional[int],
    mem_gb: Optional[int],
    lat_min: Optional[float],
    lat_max: Optional[float],
    lon_min: Optional[float],
    lon_max: Optional[float],
    max_tile_size: Optional[int],
    preprocess_scope: Optional[str],
) -> dict:
    config = copy.deepcopy(base_config)
    run_name = f"{run_prefix}_l{level}_hist_3d_{suffix}"

    config["slevel"] = level
    config["elevel"] = level
    config["out_path"] = str(output_root / run_name)
    config["temp_root"] = str(temp_root / run_name)
    config["dask_cell_batch_size"] = dask_cell_batch_size

    if lat_min is not None:
        config["lat_min"] = lat_min
    if lat_max is not None:
        config["lat_max"] = lat_max
    if lon_min is not None:
        config["lon_min"] = lon_min
    if lon_max is not None:
        config["lon_max"] = lon_max
    if max_tile_size is not None:
        config["max_tile_size"] = max_tile_size
    if preprocess_scope is not None:
        config["preprocess_scope"] = preprocess_scope

    resources = dict(config.get("resources") or {})
    if ncpus is not None:
        resources["ncpus"] = ncpus
    if mem_gb is not None:
        resources["mem_gb"] = mem_gb
    if resources:
        config["resources"] = resources

    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate level-specific SDMBCv2 3D model-as-truth configs."
    )
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--levels", required=True, type=parse_levels)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--temp-root", required=True, type=Path)
    parser.add_argument("--suffix", default="cell_arrays_1w")
    parser.add_argument("--run-prefix", default="bc_tile")
    parser.add_argument("--dask-cell-batch-size", default=16, type=int)
    parser.add_argument("--ncpus", type=int)
    parser.add_argument("--mem-gb", type=int)
    parser.add_argument("--lat-min", type=float)
    parser.add_argument("--lat-max", type=float)
    parser.add_argument("--lon-min", type=float)
    parser.add_argument("--lon-max", type=float)
    parser.add_argument("--max-tile-size", type=int)
    parser.add_argument("--preprocess-scope", choices=["domain", "tile"])
    args = parser.parse_args()

    with args.base_config.open("r", encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    if not isinstance(base_config, dict):
        raise ValueError(f"{args.base_config} did not contain a YAML mapping")

    configs_dir = args.run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    for level in args.levels:
        config = build_config(
            base_config,
            level=level,
            output_root=args.output_root,
            temp_root=args.temp_root,
            suffix=args.suffix,
            run_prefix=args.run_prefix,
            dask_cell_batch_size=args.dask_cell_batch_size,
            ncpus=args.ncpus,
            mem_gb=args.mem_gb,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            max_tile_size=args.max_tile_size,
            preprocess_scope=args.preprocess_scope,
        )
        config_name = f"config_bc_3d_{args.label}_hist_{args.run_prefix}_l{level}_{args.suffix}.yaml"
        out_file = configs_dir / config_name
        with out_file.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        print(out_file)


if __name__ == "__main__":
    main()
