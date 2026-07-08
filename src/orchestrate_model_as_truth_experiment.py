#!/usr/bin/env python3
"""Orchestrate one model-as-truth experiment (one truth GCM x period).

Thin driver over existing scripts: submits interpolation, generates the BC
base config plus per-level fan-out, submits BC via the tile-group PBS array
script chained after interpolation, then chains reformat after BC -- and
tracks job IDs/status per stage in case_root/manifest.yaml (see
manifest_schema.py). Re-running is safe: manifest fields are updated, not
recreated, so a partially-completed experiment resumes rather than restarts.

This does not reimplement submission logic -- if you change how interpolation
or BC submission works, change the underlying scripts and this driver picks
it up.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from manifest_schema import load_manifest, new_manifest, save_manifest, update_stage
from model_as_truth_registry import MODELS, STAGED_3D_MODELS, check_3d_model_staged

SCRIPT_DIR = Path(__file__).resolve().parent
VARS_3D = ["ta", "ua", "va", "hus"]


def run(cmd, dry_run: bool, capture: bool = False):
    print(f"[CMD] {' '.join(str(c) for c in cmd)}")
    if dry_run:
        return "" if capture else None
    result = subprocess.run(
        [str(c) for c in cmd], cwd=SCRIPT_DIR, capture_output=True, text=True, check=True
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return result.stdout if capture else result.stdout.strip().splitlines()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-gcm", required=True, choices=sorted(MODELS))
    parser.add_argument("--period", required=True, choices=["historical", "ssp126"])
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument(
        "--total-tiles",
        type=int,
        required=True,
        help="Total tile count for the BC step; see submit_bc_3d_tile_groups.sh docs.",
    )
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--levels", default="0-37")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    label = MODELS[args.truth_gcm]["label"]
    period_dir = "hist" if args.period == "historical" else "ssp126_2080_2100"
    experiment_id = f"{label}_to_access_{period_dir}"
    case_root = args.case_root
    manifest_path = case_root / "manifest.yaml"

    manifest = load_manifest(manifest_path)
    if not manifest:
        manifest = new_manifest(experiment_id, args.truth_gcm, args.period, case_root)
    save_manifest(manifest_path, manifest)

    if args.truth_gcm in STAGED_3D_MODELS:
        check_3d_model_staged(args.truth_gcm, args.period)

    # 1. Interpolation: 3D vars + surface tos.
    interp_config_3d = SCRIPT_DIR / f"config_interp_3d_{label}_to_access_{period_dir}.yaml"
    interp_config_surface = SCRIPT_DIR / f"config_interp_surface_{label}_to_access_{period_dir}.yaml"
    interp_job_ids = []
    for var in VARS_3D:
        job_id = run(
            [SCRIPT_DIR / "submit_interp.sh", "3d", interp_config_3d, var,
             f"interp_{label}_{var}_{period_dir}"],
            args.dry_run,
        )
        if job_id:
            interp_job_ids.append(job_id)
    job_id = run(
        [SCRIPT_DIR / "submit_interp.sh", "surface", interp_config_surface, "tos",
         f"interp_{label}_tos_{period_dir}"],
        args.dry_run,
    )
    if job_id:
        interp_job_ids.append(job_id)
    update_stage(
        manifest, "interp",
        job_ids=interp_job_ids,
        status="dry_run" if args.dry_run else "submitted",
    )
    save_manifest(manifest_path, manifest)

    # 2. Generate BC base config, then fan out per level.
    run(
        [sys.executable, SCRIPT_DIR / "generate_bc_model_as_truth_configs.py",
         "--truth-gcm", args.truth_gcm, "--period", args.period],
        args.dry_run,
    )
    bc_base_config = SCRIPT_DIR / f"config_bc_3d_{label}_to_access_{period_dir}_model_as_truth.yaml"
    level_configs_output = run(
        [sys.executable, SCRIPT_DIR / "generate_bc_level_configs.py",
         "--base-config", bc_base_config,
         "--run-dir", case_root,
         "--levels", args.levels,
         "--label", label,
         "--output-root", case_root / "output",
         "--temp-root", case_root / "temp"],
        args.dry_run,
        capture=True,
    )
    level_configs = (
        [] if args.dry_run else [Path(line) for line in level_configs_output.splitlines() if line]
    )
    update_stage(
        manifest, "bc_configs",
        base_config=str(bc_base_config),
        level_configs=[str(p) for p in level_configs],
        status="dry_run" if args.dry_run else "generated",
    )
    save_manifest(manifest_path, manifest)

    # 3. Submit BC per level via the tile-group PBS array, chained after
    #    all interpolation jobs.
    depend = "afterok:" + ":".join(interp_job_ids) if interp_job_ids else None
    bc_job_ids = []
    depend_env = {"JOB_DEPENDENCY_OVERRIDE": depend} if depend else {}
    for level_config in level_configs or [None]:
        if level_config is None and not args.dry_run:
            break
        job_name = f"bc_{label}_{period_dir}" if level_config is None else f"bc_{label}_{period_dir}_{level_config.stem}"
        env = {**os.environ, **depend_env}
        print(f"[CMD] submit_bc_3d_tile_groups.sh {level_config} {args.total_tiles} {args.group_size} {job_name}"
              f" (JOB_DEPENDENCY_OVERRIDE={depend})")
        if args.dry_run:
            continue
        result = subprocess.run(
            [str(SCRIPT_DIR / "submit_bc_3d_tile_groups.sh"), str(level_config),
             str(args.total_tiles), str(args.group_size), job_name],
            cwd=SCRIPT_DIR, capture_output=True, text=True, check=True, env=env,
        )
        print(result.stdout, end="")
        job_id = result.stdout.strip().splitlines()[-1]
        bc_job_ids.append(job_id)
    update_stage(
        manifest, "bc",
        job_ids=bc_job_ids,
        status="dry_run" if args.dry_run else "submitted",
    )
    save_manifest(manifest_path, manifest)

    # 4. Chain reformat after all BC (array) jobs.
    reformat_depend = "afterok:" + ":".join(bc_job_ids) if bc_job_ids else None
    reformat_env = {**os.environ}
    if reformat_depend:
        reformat_env["JOB_DEPENDENCY_OVERRIDE"] = reformat_depend
    print(f"[CMD] submit_reformat.sh {bc_base_config} reformat_{label}_{period_dir}"
          f" (JOB_DEPENDENCY_OVERRIDE={reformat_depend})")
    reformat_job_id = None
    if not args.dry_run:
        result = subprocess.run(
            [str(SCRIPT_DIR / "submit_reformat.sh"), str(bc_base_config),
             f"reformat_{label}_{period_dir}"],
            cwd=SCRIPT_DIR, capture_output=True, text=True, check=True, env=reformat_env,
        )
        print(result.stdout, end="")
        reformat_job_id = result.stdout.strip().splitlines()[-1]
    update_stage(
        manifest, "reformat",
        job_id=reformat_job_id,
        status="dry_run" if args.dry_run else "submitted",
    )
    save_manifest(manifest_path, manifest)

    print(f"[INFO] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
