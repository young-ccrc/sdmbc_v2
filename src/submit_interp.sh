#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_interp.sh <kind> <config> <var> [job_name]

Arguments:
  kind     3d | surface
  config   Path to YAML config
  var      Variable name, e.g. ta, ua, va, hus, tos
  job_name Optional PBS job name

Optional environment overrides:
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE

Examples:
  ./submit_interp.sh 3d src/config_interp_3d_era5_to_access_hist.yaml ta
  ./submit_interp.sh 3d src/config_interp_3d_cnrm_to_access_hist.yaml ta
  ./submit_interp.sh surface src/config_interp_surface_era5_to_access_hist.yaml tos
EOF
}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    usage
    exit 1
fi

KIND=$1
CONFIG=$(realpath "$2")
VAR=$3
JOB_NAME=${4:-interp_${KIND}_${VAR}}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

case "$KIND" in
    3d)
        PBS_SCRIPT="$SCRIPT_DIR/interp_3d_job.pbs"
        PYTHON_SCRIPT="$SCRIPT_DIR/interp_obs2gcm.py"
        NCPUS=${NCPUS_OVERRIDE:-48}
        MEM_GB=${MEM_GB_OVERRIDE:-190}
        WALLTIME=${WALLTIME_OVERRIDE:-24:00:00}
        MODULE_ANALYSIS=${MODULE_ANALYSIS_OVERRIDE:-analysis3-24.01}
        ;;
    surface)
        PBS_SCRIPT="$SCRIPT_DIR/interp_surface_job.pbs"
        PYTHON_SCRIPT="$SCRIPT_DIR/interp_2d_obs2gcm_cdo.py"
        NCPUS=${NCPUS_OVERRIDE:-48}
        MEM_GB=${MEM_GB_OVERRIDE:-190}
        WALLTIME=${WALLTIME_OVERRIDE:-06:00:00}
        MODULE_ANALYSIS=${MODULE_ANALYSIS_OVERRIDE:-analysis3-24.04}
        ;;
    *)
        echo "Unknown kind: $KIND" >&2
        usage
        exit 1
        ;;
esac

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

qsub     -N "$JOB_NAME"     -l walltime="$WALLTIME"     -l mem="${MEM_GB}GB"     -l ncpus="$NCPUS"     -v CONFIG="$CONFIG",VAR="$VAR",NCPUS="$NCPUS",MEM_GB="$MEM_GB",MODULE_ANALYSIS="$MODULE_ANALYSIS",SCRIPT_PATH="$PYTHON_SCRIPT"     "$PBS_SCRIPT"
