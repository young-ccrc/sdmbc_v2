#!/bin/bash
set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <period>" >&2
  echo "Example: $0 historical" >&2
  echo "Example: $0 ssp126" >&2
  exit 1
fi

PERIOD=$1
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
qsub -N "ukesm_stage_${PERIOD}" -v PERIOD="$PERIOD" "$SCRIPT_DIR/stage_ukesm_3d_to_gadi.pbs"
