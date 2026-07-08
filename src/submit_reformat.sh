#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_reformat.sh <config> [job_name]

Submits reformat_gcm2origin.py, which reconstructs bias-corrected output back
into the original GCM's NetCDF structure. Typically run after a BC job
completes; chain it with JOB_DEPENDENCY_OVERRIDE=afterok:<bc_job_id>.

Optional environment overrides:
  PROJECT_OVERRIDE
  JOB_DEPENDENCY_OVERRIDE   (e.g. afterok:12345.gadi-pbs)
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE

Example:
  JOB_DEPENDENCY_OVERRIDE=afterok:172700316.gadi-pbs \
  ./submit_reformat.sh config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml bc_l20_reformat
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 1
fi

CONFIG=$(realpath "$1")
JOB_NAME=${2:-bc_reformat}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PBS_SCRIPT="$SCRIPT_DIR/reformat_gcm2origin.pbs"
PYTHON_SCRIPT="$SCRIPT_DIR/reformat_gcm2origin.py"

NCPUS=${NCPUS_OVERRIDE:-8}
MEM_GB=${MEM_GB_OVERRIDE:-64}
JOBFS_GB=${JOBFS_GB_OVERRIDE:-20}
WALLTIME=${WALLTIME_OVERRIDE:-06:00:00}
MODULE_ANALYSIS=${MODULE_ANALYSIS_OVERRIDE:-analysis3-26.02}
PROJECT=${PROJECT_OVERRIDE:-n81}

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
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
    -v CONFIG="$CONFIG",NCPUS="$NCPUS",MEM_GB="$MEM_GB",MODULE_ANALYSIS="$MODULE_ANALYSIS",SCRIPT_PATH="$PYTHON_SCRIPT" \
    "$PBS_SCRIPT"
