#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_bc_climatology_pipeline.sh <bc_nc> [config_yaml] [job_name_prefix]

This submits the climatology/bias compute job, then submits a follow-up
bias-map plot job that uses the generated bias NetCDF.

Optional environment overrides:
  COMPUTE_OUTPUT_PATH_OVERRIDE
  COMPUTE_OUTDIR_OVERRIDE
  COMPUTE_BIAS_OUTPUT_PATH_OVERRIDE
  COMPUTE_REFERENCE_INPUT_OVERRIDE
  COMPUTE_CHUNKS_OVERRIDE
  PLOT_OUTDIR_OVERRIDE
  PLOT_PATTERN_OVERRIDE
  PLOT_FACET_DIM_OVERRIDE
  PLOT_CMAP_OVERRIDE
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE
  PROJECT_OVERRIDE

Example:
  ./submit_bc_climatology_pipeline.sh bc_corrected.nc config_bc_3d.yaml bc_climo
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    usage
    exit 1
fi

BC_INPUT=$(realpath "$1")
CONFIG_PATH=""
if [ "$#" -ge 2 ] && [ -n "${2:-}" ] && [ "$2" != "-" ]; then
    CONFIG_PATH=$(realpath "$2")
fi
JOB_PREFIX=${3:-bc_climo}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

COMPUTE_JOB="$JOB_PREFIX.compute"
PLOT_JOB="$JOB_PREFIX.plot"

COMPUTE_OUTPUT_PATH=${COMPUTE_OUTPUT_PATH_OVERRIDE:-}
COMPUTE_OUTDIR=${COMPUTE_OUTDIR_OVERRIDE:-}
COMPUTE_BIAS_OUTPUT_PATH=${COMPUTE_BIAS_OUTPUT_PATH_OVERRIDE:-}
COMPUTE_REFERENCE_INPUT=${COMPUTE_REFERENCE_INPUT_OVERRIDE:-}
COMPUTE_CHUNKS=${COMPUTE_CHUNKS_OVERRIDE:-}
PROJECT=${PROJECT_OVERRIDE:-n81}

COMPUTE_CMD=("$SCRIPT_DIR/submit_compute_bc_climatology.sh" "$BC_INPUT")
if [ -n "$CONFIG_PATH" ]; then
    COMPUTE_CMD+=("$CONFIG_PATH")
fi
COMPUTE_CMD+=("$COMPUTE_JOB")

if [ -n "$COMPUTE_OUTPUT_PATH" ]; then
    export OUTPUT_PATH_OVERRIDE="$COMPUTE_OUTPUT_PATH"
fi
if [ -n "$COMPUTE_OUTDIR" ]; then
    export OUTDIR_OVERRIDE="$COMPUTE_OUTDIR"
fi
if [ -n "$COMPUTE_BIAS_OUTPUT_PATH" ]; then
    export BIAS_OUTPUT_PATH_OVERRIDE="$COMPUTE_BIAS_OUTPUT_PATH"
fi
if [ -n "$COMPUTE_REFERENCE_INPUT" ]; then
    export REFERENCE_INPUT_OVERRIDE="$COMPUTE_REFERENCE_INPUT"
fi
if [ -n "$COMPUTE_CHUNKS" ]; then
    export CHUNKS_OVERRIDE="$COMPUTE_CHUNKS"
fi
export PROJECT_OVERRIDE="$PROJECT"

if [ -z "$COMPUTE_OUTPUT_PATH" ]; then
    COMPUTE_OUTPUT_PATH=$(python3 - <<'PY' "$BC_INPUT"
from pathlib import Path
import sys
p = Path(sys.argv[1])
name = p.name
if name.startswith("bc_corrected_"):
    out_name = name.replace("bc_corrected_", "bc_climatology_", 1)
elif name.startswith("bc_"):
    out_name = name.replace("bc_", "bc_climatology_", 1)
else:
    out_name = f"{p.stem}_climatology.nc"
print(p.parent / out_name)
PY
)
fi

COMPUTE_JID=$("${COMPUTE_CMD[@]}")
echo "$COMPUTE_JID"
COMPUTE_JID_NUM=${COMPUTE_JID%%.*}

BIAS_FILE=${COMPUTE_BIAS_OUTPUT_PATH:-}
if [ -z "$BIAS_FILE" ]; then
    BIAS_FILE=$(python3 - <<'PY' "$COMPUTE_OUTPUT_PATH"
from pathlib import Path
import sys
p = Path(sys.argv[1])
name = p.name
if name.startswith("bc_climatology_"):
    name = name.replace("bc_climatology_", "bc_climatology_bias_", 1)
else:
    name = f"{p.stem}_bias.nc"
print(p.parent / name)
PY
)
fi

if [ -z "$BIAS_FILE" ]; then
    echo "Bias output path could not be inferred. Set COMPUTE_BIAS_OUTPUT_PATH_OVERRIDE." >&2
    exit 1
fi

PLOT_OUTDIR=${PLOT_OUTDIR_OVERRIDE:-}
PLOT_PATTERN=${PLOT_PATTERN_OVERRIDE:-}
PLOT_FACET_DIM=${PLOT_FACET_DIM_OVERRIDE:-}
PLOT_CMAP=${PLOT_CMAP_OVERRIDE:-}

PLOT_CMD=("$SCRIPT_DIR/submit_plot_bc_bias_map.sh" "$BIAS_FILE")
if [ -n "$PLOT_OUTDIR" ]; then
    PLOT_CMD+=("$PLOT_OUTDIR")
fi
PLOT_CMD+=("$PLOT_JOB")

if [ -n "$PLOT_PATTERN" ]; then
    export PATTERN_OVERRIDE="$PLOT_PATTERN"
fi
if [ -n "$PLOT_FACET_DIM" ]; then
    export FACET_DIM_OVERRIDE="$PLOT_FACET_DIM"
fi
if [ -n "$PLOT_CMAP" ]; then
    export CMAP_OVERRIDE="$PLOT_CMAP"
fi
export JOB_DEPENDENCY_OVERRIDE="afterok:${COMPUTE_JID_NUM}"
export ALLOW_MISSING_INPUT_OVERRIDE=1

PLOT_JID=$("${PLOT_CMD[@]}")
echo "$PLOT_JID"
