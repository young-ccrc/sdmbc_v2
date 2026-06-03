#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_bc_3d_levels.sh <base_config> <start_level> <end_level> <job_prefix>

Creates one generated config per level with slevel == elevel, then submits one
PBS job per level through submit_bc_3d.sh.

Optional environment overrides are passed through to submit_bc_3d.sh:
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE
  GENERATED_CONFIG_DIR
  CONFIG_PYTHON
  DRY_RUN=1

Example:
  WALLTIME_OVERRIDE=06:00:00 MEM_GB_OVERRIDE=64 \
  ./submit_bc_3d_levels.sh \
    config_bc_3d_ecearth3veg_to_access_hist_tile_l0_4.yaml \
    0 4 bc_ecearth_tile

Dry run:
  DRY_RUN=1 ./submit_bc_3d_levels.sh base.yaml 0 4 bc_ecearth_tile
EOF
}

if [ "$#" -ne 4 ]; then
    usage
    exit 1
fi

BASE_CONFIG=$(realpath "$1")
START_LEVEL=$2
END_LEVEL=$3
JOB_PREFIX=$4
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_bc_3d.sh"
GENERATED_CONFIG_DIR=${GENERATED_CONFIG_DIR:-"/scratch/n81/$USER/sdmbc_v2_generated_configs/bc_3d_levels"}
CONFIG_PYTHON=${CONFIG_PYTHON:-python3}

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Base config not found: $BASE_CONFIG" >&2
    exit 1
fi

case "$START_LEVEL" in
    ''|*[!0-9]*) echo "start_level must be a non-negative integer" >&2; exit 1 ;;
esac

case "$END_LEVEL" in
    ''|*[!0-9]*) echo "end_level must be a non-negative integer" >&2; exit 1 ;;
esac

if [ "$START_LEVEL" -gt "$END_LEVEL" ]; then
    echo "start_level must be <= end_level" >&2
    exit 1
fi

mkdir -p "$GENERATED_CONFIG_DIR"

if ! "$CONFIG_PYTHON" - <<'PY' >/dev/null 2>&1
import yaml
PY
then
    if [ -x /g/data/xp65/public/apps/med_conda/envs/analysis3-26.02/bin/python3 ]; then
        CONFIG_PYTHON=/g/data/xp65/public/apps/med_conda/envs/analysis3-26.02/bin/python3
    else
        echo "Could not import PyYAML with CONFIG_PYTHON=$CONFIG_PYTHON" >&2
        echo "Set CONFIG_PYTHON to a Python that has PyYAML, or run: module load conda/analysis3-26.02" >&2
        exit 1
    fi
fi

for LEVEL in $(seq "$START_LEVEL" "$END_LEVEL"); do
    GENERATED_CONFIG="$GENERATED_CONFIG_DIR/$(basename "$BASE_CONFIG" .yaml)_lev_${LEVEL}.yaml"
    "$CONFIG_PYTHON" - "$BASE_CONFIG" "$GENERATED_CONFIG" "$LEVEL" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, out_path, level = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(base_path) as f:
    config = yaml.safe_load(f)

config["slevel"] = level
config["elevel"] = level

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    yaml.safe_dump(config, f, default_flow_style=False)
PY

    JOB_NAME="${JOB_PREFIX}_l${LEVEL}"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN level ${LEVEL}: $SUBMIT_SCRIPT $GENERATED_CONFIG $JOB_NAME"
    else
        echo "Submitting level ${LEVEL}: ${GENERATED_CONFIG} as ${JOB_NAME}"
        "$SUBMIT_SCRIPT" "$GENERATED_CONFIG" "$JOB_NAME"
    fi
done
