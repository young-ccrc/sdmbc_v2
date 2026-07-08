#!/usr/bin/env python3
"""Manifest tracking for model-as-truth experiments.

Formalizes the case_root/manifest.yaml/qa pattern the user has used ad hoc
in scratch run-trees (e.g. sdmbc_runs/model_as_truth/.../manifest.yaml) into
reusable tooling. Deliberately a thin YAML read/update/write helper, not a
schema-validation library -- consistent with the rest of the repo's style.
"""

from datetime import datetime, timezone
from pathlib import Path

import yaml


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_manifest(experiment_id: str, truth_gcm: str, period: str, case_root: Path) -> dict:
    return {
        "experiment_id": experiment_id,
        "truth_gcm": truth_gcm,
        "period": period,
        "case_root": str(case_root),
        "created": now(),
        "stages": {},
    }


def load_manifest(path: Path) -> dict:
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, default_flow_style=False, sort_keys=False), encoding="utf-8")


def update_stage(manifest: dict, stage: str, **fields) -> None:
    """Update one stage's status/fields in place; call save_manifest after."""
    manifest.setdefault("stages", {})
    manifest["stages"].setdefault(stage, {})
    manifest["stages"][stage].update(fields)
    manifest["stages"][stage]["updated"] = now()
