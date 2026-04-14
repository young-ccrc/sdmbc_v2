#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_interp_chunks.sh <kind> <config> <vars> <chunks> [job_prefix]

Arguments:
  kind       3d | surface
  config     Path to YAML config
  vars       Comma-separated variables, e.g. ta,ua,va,hus or tos
  chunks     Comma-separated year ranges, e.g. 1984-1994,1995-2004,2005-2014
  job_prefix Optional job name prefix

Optional environment overrides:
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE

Examples:
  ./submit_interp_chunks.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta,ua,va,hus 1984-1994,1995-2004,2005-2014 ecearth_hist
  ./submit_interp_chunks.sh surface config_interp_surface_cnrm_to_access_hist.yaml tos 1984-1994,1995-2004,2005-2014 cnrm_sst_hist
EOF
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    usage
    exit 1
fi

KIND=$1
CONFIG=$2
VARS_CSV=$3
CHUNKS_CSV=$4
JOB_PREFIX=${5:-$(basename "$CONFIG" .yaml)}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_interp.sh"

if [ ! -x "$SUBMIT_SCRIPT" ]; then
    echo "submit_interp.sh not found or not executable: $SUBMIT_SCRIPT" >&2
    exit 1
fi

IFS=',' read -r -a VARS <<< "$VARS_CSV"
IFS=',' read -r -a CHUNKS <<< "$CHUNKS_CSV"

for VAR in "${VARS[@]}"; do
    for CHUNK in "${CHUNKS[@]}"; do
        STARTYEAR=${CHUNK%-*}
        ENDYEAR=${CHUNK#*-}

        if [ -z "$STARTYEAR" ] || [ -z "$ENDYEAR" ] || [ "$STARTYEAR" = "$ENDYEAR" ] && ! [[ "$CHUNK" =~ ^[0-9]{4}-[0-9]{4}$ ]]; then
            echo "Invalid chunk format: $CHUNK (expected YYYY-YYYY)" >&2
            exit 1
        fi

        JOB_NAME="${JOB_PREFIX}_${VAR}_${STARTYEAR}_${ENDYEAR}"
        echo "Submitting $JOB_NAME"
        STARTYEAR_OVERRIDE="$STARTYEAR"         ENDYEAR_OVERRIDE="$ENDYEAR"         "$SUBMIT_SCRIPT" "$KIND" "$CONFIG" "$VAR" "$JOB_NAME"
    done
done
