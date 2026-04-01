#!/bin/bash
set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MODELS_3D=(ecearth3veg mpi_esm1_2_hr)
MODELS_SURFACE=(ecearth3veg mpi_esm1_2_hr ukesm1_0_ll)
VARS_3D=(ta ua va hus)

for model in "${MODELS_3D[@]}"; do
  config="$SCRIPT_DIR/config_interp_3d_${model}_to_access_hist.yaml"
  for var in "${VARS_3D[@]}"; do
    "$SCRIPT_DIR/submit_interp.sh" 3d "$config" "$var" "interp_${model}_${var}_hist"
  done
done

for model in "${MODELS_SURFACE[@]}"; do
  config="$SCRIPT_DIR/config_interp_surface_${model}_to_access_hist.yaml"
  "$SCRIPT_DIR/submit_interp.sh" surface "$config" tos "interp_${model}_tos_hist"
done
