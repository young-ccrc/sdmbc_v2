#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_bc_3d.sh <config> [job_name]

Optional environment overrides:
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE
  DASK_N_WORKERS_OVERRIDE
  DASK_THREADS_PER_WORKER_OVERRIDE
  DASK_PROCESSES_OVERRIDE

Example:
  ./submit_bc_3d.sh config_bc_3d_ecearth3veg_to_access_hist_onegrid.yaml bc_ecearth_onegrid_hist
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 1
fi

CONFIG=$(realpath "$1")
JOB_NAME=${2:-bc_3d_test}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PBS_SCRIPT="$SCRIPT_DIR/sdmbc_3d_job.pbs"
PYTHON_SCRIPT="$SCRIPT_DIR/sdmbc_main_3d.py"

NCPUS=${NCPUS_OVERRIDE:-8}
MEM_GB=${MEM_GB_OVERRIDE:-64}
JOBFS_GB=${JOBFS_GB_OVERRIDE:-20}
WALLTIME=${WALLTIME_OVERRIDE:-06:00:00}
# Keep BC jobs on Python 3.11 until mrmbc is rebuilt for Python 3.12.
MODULE_ANALYSIS=${MODULE_ANALYSIS_OVERRIDE:-analysis3-26.02}
DASK_N_WORKERS=${DASK_N_WORKERS_OVERRIDE:-}
DASK_THREADS_PER_WORKER=${DASK_THREADS_PER_WORKER_OVERRIDE:-}
DASK_PROCESSES=${DASK_PROCESSES_OVERRIDE:-}

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

qsub \
    -N "$JOB_NAME" \
    -l walltime="$WALLTIME" \
    -l mem="${MEM_GB}GB" \
    -l ncpus="$NCPUS" \
    -l jobfs="${JOBFS_GB}GB" \
    -v CONFIG="$CONFIG",NCPUS="$NCPUS",MEM_GB="$MEM_GB",MODULE_ANALYSIS="$MODULE_ANALYSIS",SCRIPT_PATH="$PYTHON_SCRIPT",DASK_N_WORKERS="$DASK_N_WORKERS",DASK_THREADS_PER_WORKER="$DASK_THREADS_PER_WORKER",DASK_PROCESSES="$DASK_PROCESSES" \
    "$PBS_SCRIPT"
