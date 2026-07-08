#!/bin/bash
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODELS_3D=(ecearth3veg mpi_esm1_2_hr ukesm1_0_ll)
MODELS_SURFACE=(ecearth3veg mpi_esm1_2_hr ukesm1_0_ll)
VARS_3D=(ta ua va hus)

for model in "${MODELS_3D[@]}"; do
  if [ "$model" = "ukesm1_0_ll" ]; then
    # UKESM 3D input requires the staging step (stage_ukesm_3d_to_gadi.sh)
    # first -- fail fast here rather than deep inside interpolation.
    ( cd "$SCRIPT_DIR" && python3 -c "
from model_as_truth_registry import check_3d_model_staged
check_3d_model_staged('UKESM1-0-LL', 'historical')
" )
  fi
  config="$SCRIPT_DIR/config_interp_3d_${model}_to_access_hist.yaml"
  for var in "${VARS_3D[@]}"; do
    "$SCRIPT_DIR/submit_interp.sh" 3d "$config" "$var" "interp_${model}_${var}_hist"
  done
done

for model in "${MODELS_SURFACE[@]}"; do
  config="$SCRIPT_DIR/config_interp_surface_${model}_to_access_hist.yaml"
  "$SCRIPT_DIR/submit_interp.sh" surface "$config" tos "interp_${model}_tos_hist"
done
