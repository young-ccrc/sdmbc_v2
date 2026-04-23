#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_plot_bc_bias_map.sh <bias_nc> [outdir] [job_name]

Optional environment overrides:
  PROJECT_OVERRIDE
  JOB_DEPENDENCY_OVERRIDE
  ALLOW_MISSING_INPUT_OVERRIDE=1
  PATTERN_OVERRIDE
  FACET_DIM_OVERRIDE
  CMAP_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  NCPUS_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE

Example:
  ./submit_plot_bc_bias_map.sh bc_climatology_bias.nc plots bc_bias_plot
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    usage
    exit 1
fi

INPUT_PATH=$(realpath "$1")
OUTDIR_PATH=${2:-}
JOB_NAME=${3:-bc_bias_plot}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PBS_SCRIPT="$SCRIPT_DIR/plot_bc_bias_map.pbs"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_bc_bias_map.py"

NCPUS=${NCPUS_OVERRIDE:-4}
MEM_GB=${MEM_GB_OVERRIDE:-16}
JOBFS_GB=${JOBFS_GB_OVERRIDE:-5}
WALLTIME=${WALLTIME_OVERRIDE:-02:00:00}
MODULE_ANALYSIS=${MODULE_ANALYSIS_OVERRIDE:-analysis3-26.02}
PROJECT=${PROJECT_OVERRIDE:-n81}

if [ "${ALLOW_MISSING_INPUT_OVERRIDE:-0}" != "1" ] && [ ! -f "$INPUT_PATH" ]; then
    echo "Input not found: $INPUT_PATH" >&2
    exit 1
fi

qsub \
    -P "$PROJECT" \
    -N "$JOB_NAME" \
    -l walltime="$WALLTIME" \
    -l mem="${MEM_GB}GB" \
    -l ncpus="$NCPUS" \
    -l jobfs="${JOBFS_GB}GB" \
    ${JOB_DEPENDENCY_OVERRIDE:+-W depend="$JOB_DEPENDENCY_OVERRIDE"} \
    -v INPUT_PATH="$INPUT_PATH",OUTDIR_PATH="$OUTDIR_PATH",PATTERN_VALUE="${PATTERN_OVERRIDE:-}",FACET_DIM_VALUE="${FACET_DIM_OVERRIDE:-}",CMAP_VALUE="${CMAP_OVERRIDE:-RdBu_r}",MODULE_ANALYSIS="$MODULE_ANALYSIS",SCRIPT_PATH="$PYTHON_SCRIPT" \
    "$PBS_SCRIPT"
