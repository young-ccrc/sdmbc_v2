#!/bin/bash
set -eu

PERIOD=${PERIOD:-historical}
REMOTE_USER=${REMOTE_USER:-z5239661}
REMOTE_HOST=${REMOTE_HOST:-squall.ccrc.unsw.edu.au}
REMOTE_BASE=${REMOTE_BASE:-/srv/ccrc/NARCliM2/bdy/GCM/UKESM1-0-LL}
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-/g/data/w28/yk8692/input/model_as_truth/ukesm1_0_ll_6hrlev_stage}
LOCAL_VERSION=${LOCAL_VERSION:-v20260401}
MEMBER=${MEMBER:-r1i1p1f2}
GRID_LABEL=${GRID_LABEL:-gn}
VARS=${VARS:-"hus ta ua va"}

RAW_BASE="$LOCAL_STAGE_ROOT/raw/$PERIOD"
CMIP_BASE="$LOCAL_STAGE_ROOT/$PERIOD/$MEMBER/6hrLev"
mkdir -p "$RAW_BASE"

for var in $VARS; do
  raw_dest="$RAW_BASE/$var"
  final_dest="$CMIP_BASE/$var/$GRID_LABEL/$LOCAL_VERSION"
  mkdir -p "$raw_dest" "$final_dest"

  echo "Staging $PERIOD $var from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/$PERIOD/$var/"
  rsync -av "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/$PERIOD/$var/" "$raw_dest/"

  find "$raw_dest" -type f -name "${var}_6hrLev_UKESM1-0-LL_${PERIOD}_*.nc" -print0 | while IFS= read -r -d '' file; do
    ln -sfn "$file" "$final_dest/$(basename "$file")"
  done

done

echo "UKESM 3D staging complete for period: $PERIOD"
echo "CMIP-like files available under: $CMIP_BASE"
