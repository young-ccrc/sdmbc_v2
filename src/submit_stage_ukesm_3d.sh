#!/bin/bash
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <period> [login|pbs]" >&2
  echo "Example: $0 historical" >&2
  echo "Example: $0 historical login" >&2
  echo "Example: $0 ssp126 pbs" >&2
  exit 1
fi

PERIOD=$1
RUN_MODE=${2:-login}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

case "$RUN_MODE" in
  login)
    echo "Running UKESM staging on the login node for period: $PERIOD"
    PERIOD="$PERIOD" "$SCRIPT_DIR/stage_ukesm_3d_to_gadi.sh"
    ;;
  pbs)
    qsub -N "ukesm_stage_${PERIOD}" -v PERIOD="$PERIOD",SCRIPT_DIR="$SCRIPT_DIR" "$SCRIPT_DIR/stage_ukesm_3d_to_gadi.pbs"
    ;;
  *)
    echo "Unknown run mode: $RUN_MODE" >&2
    echo "Use 'login' or 'pbs'." >&2
    exit 1
    ;;
esac
