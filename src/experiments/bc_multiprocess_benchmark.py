#!/usr/bin/env python3
"""Isolated benchmark: does the real cell-array BC path corrupt under a
multi-worker dask.distributed.Client?

Background (see TROUBLESHOOTING_BC_MODEL_AS_TRUTH.md and the project plan's
Phase 2 correction): a documented 4-worker run using
bc_grid_function._correct_hist_grid_cells -- the SAME eager-copy cell-array
path already validated safe at 1 worker -- still produced wrong lower-bound
columns. That function does no file I/O inside worker calls (workers only
receive pre-sliced, contiguous NumPy blocks and call the Fortran `mrmbc`
extension), so this benchmark's job is to help localize which regime
reproduces the corruption, not to test a "new" isolation strategy.

This script imports and calls the REAL, unmodified
_materialize_hist_cell_inputs / _correct_hist_grid_cells from
bc_grid_function.py -- it does not reimplement dispatch logic. It's a
separate script only so production code isn't touched while investigating.

Two axes are crossed explicitly, since the bad run's exact settings aren't
recorded in the manifest:
  - worker count: 1 (baseline), 2, 4, optionally 8
  - DASK_PROCESSES: true (separate OS processes) and false (threads)

Usage (real data, from a completed 1-worker validated run's tile inputs):
  python bc_multiprocess_benchmark.py \\
    --gcm-file /path/to/reshaped_gcm_tile.nc \\
    --obs-file /path/to/reshaped_obs_tile.nc \\
    --variables hus,ta,ua,va

Usage (fast synthetic self-test of the harness itself, no real data needed):
  python bc_multiprocess_benchmark.py --synthetic-self-test
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from dask.distributed import Client  # noqa: E402

import bc_grid_function  # noqa: E402


def make_synthetic_tile(n_year=2, n_month=2, n_day=3, n_lat=4, n_lon=4, seed=0):
    """Small deterministic synthetic tile for a fast harness self-test.

    Not a substitute for the real-data run -- this only exercises the
    benchmark's own dispatch/comparison logic quickly and deterministically.
    """
    rng = np.random.default_rng(seed)
    shape = (n_year, n_month, n_day, n_lat, n_lon)
    coords = {
        "year": np.arange(2000, 2000 + n_year),
        "month": np.arange(1, n_month + 1),
        "day": np.arange(1, n_day + 1),
        "lat": np.linspace(-10.0, 10.0, n_lat),
        "lon": np.linspace(100.0, 120.0, n_lon),
    }
    var_names = ["hus", "ta", "ua", "va"]

    def make_ds(offset):
        return xr.Dataset(
            {
                name: (("year", "month", "day", "lat", "lon"), (rng.random(shape) + offset).astype(np.float32))
                for name in var_names
            },
            coords=coords,
        )

    return make_ds(0.0), make_ds(10.0), var_names


def run_one(config, reshaped_gcm, reshaped_obs, var_names, n_workers, processes):
    """Run the REAL _correct_hist_grid_cells under a real distributed Client."""
    client = Client(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=processes,
        memory_limit=None,
    )
    try:
        corrected_ds, bc_params_array = bc_grid_function._correct_hist_grid_cells(
            config, reshaped_gcm, reshaped_obs, var_names
        )
    finally:
        client.close()
    return corrected_ds, bc_params_array


def lower_bound_pin_counts(corrected_ds, var_names, lower_limit: Optional[Sequence[float]], atol=1e-6):
    """Count cells pinned at the configured physical lower bound per variable.

    Assumes lower_limit[i] corresponds to var_names[i] (the config's
    target_variable order) -- adjust if your config orders variables
    differently.
    """
    counts = {}
    if lower_limit is None:
        return counts
    for i, var_name in enumerate(var_names):
        if i >= len(lower_limit):
            continue
        bound = lower_limit[i]
        values = corrected_ds[var_name].values
        counts[var_name] = int(np.isclose(values, bound, atol=atol).sum())
    return counts


def compare_to_reference(reference_ds, reference_params, candidate_ds, candidate_params,
                          var_names, lower_limit, tolerance):
    """Return a dict of check -> (pass: bool, detail: str)."""
    checks = {}

    ref_shapes = {v: reference_ds[v].shape for v in var_names}
    cand_shapes = {v: candidate_ds[v].shape for v in var_names}
    checks["shapes_match"] = (ref_shapes == cand_shapes, f"{ref_shapes} vs {cand_shapes}")

    any_nan_inf = any(not np.isfinite(candidate_ds[v].values).all() for v in var_names)
    checks["no_nan_inf"] = (not any_nan_inf, "NaN/Inf present" if any_nan_inf else "clean")

    max_abs_diffs = {
        v: float(np.max(np.abs(candidate_ds[v].values - reference_ds[v].values)))
        for v in var_names
    }
    max_diff = max(max_abs_diffs.values()) if max_abs_diffs else 0.0
    checks["max_abs_diff"] = (max_diff <= tolerance, f"{max_abs_diffs} (tolerance={tolerance})")

    ref_pins = lower_bound_pin_counts(reference_ds, var_names, lower_limit)
    cand_pins = lower_bound_pin_counts(candidate_ds, var_names, lower_limit)
    checks["lower_bound_pin_count_matches"] = (
        ref_pins == cand_pins, f"reference={ref_pins} candidate={cand_pins}"
    )

    ref_params_flat = [p for p in reference_params.flat]
    cand_params_flat = [p for p in candidate_params.flat]
    checks["params_match"] = (
        ref_params_flat == cand_params_flat,
        "differs" if ref_params_flat != cand_params_flat else "identical",
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gcm-file", type=Path, help="Reshaped GCM tile NetCDF (year/month/day/lat/lon dims)")
    parser.add_argument("--obs-file", type=Path, help="Reshaped truth/obs tile NetCDF, same dims")
    parser.add_argument("--variables", default="hus,ta,ua,va", help="Comma-separated variable names")
    parser.add_argument("--worker-counts", default="1,2,4", help="Comma-separated worker counts to test")
    parser.add_argument(
        "--processes", default="true,false",
        help="Comma-separated DASK_PROCESSES values to test (true=OS processes, false=threads)",
    )
    parser.add_argument("--dask-cell-batch-size", type=int, default=16)
    parser.add_argument("--lower-limit", default=None,
                         help="Comma-separated physical lower bounds, ordered to match --variables")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--synthetic-self-test", action="store_true",
        help="Use small synthetic data instead of --gcm-file/--obs-file (fast harness self-check only).",
    )
    args = parser.parse_args()

    if args.synthetic_self_test:
        reshaped_gcm, reshaped_obs, var_names = make_synthetic_tile()
    else:
        if not args.gcm_file or not args.obs_file:
            parser.error("--gcm-file and --obs-file are required unless --synthetic-self-test is set")
        reshaped_gcm = xr.open_dataset(args.gcm_file)
        reshaped_obs = xr.open_dataset(args.obs_file)
        var_names = [v.strip() for v in args.variables.split(",") if v.strip()]

    lower_limit = (
        [float(v) for v in args.lower_limit.split(",")] if args.lower_limit else None
    )

    from types import SimpleNamespace
    config = SimpleNamespace(dask_cell_batch_size=args.dask_cell_batch_size)

    worker_counts = [int(v) for v in args.worker_counts.split(",")]
    processes_options = [v.strip().lower() in {"1", "true", "yes"} for v in args.processes.split(",")]

    print("[INFO] Establishing 1-worker/processes=true baseline...")
    reference_ds, reference_params = run_one(config, reshaped_gcm, reshaped_obs, var_names, 1, True)

    results: Dict[str, dict] = {}
    all_pass = True
    for n_workers in worker_counts:
        for processes in processes_options:
            label = f"workers={n_workers},processes={processes}"
            if n_workers == 1 and processes:
                results[label] = {"skipped": "same as baseline"}
                continue
            print(f"[INFO] Running {label}...")
            candidate_ds, candidate_params = run_one(
                config, reshaped_gcm, reshaped_obs, var_names, n_workers, processes
            )
            checks = compare_to_reference(
                reference_ds, reference_params, candidate_ds, candidate_params,
                var_names, lower_limit, args.tolerance,
            )
            passed = all(ok for ok, _ in checks.values())
            all_pass = all_pass and passed
            results[label] = {"passed": passed, "checks": checks}

    print("\n=== Benchmark report ===")
    for label, result in results.items():
        if "skipped" in result:
            print(f"{label}: SKIPPED ({result['skipped']})")
            continue
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{label}: {status}")
        for check_name, (ok, detail) in result["checks"].items():
            print(f"    {'ok' if ok else 'FAIL'}  {check_name}: {detail}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
