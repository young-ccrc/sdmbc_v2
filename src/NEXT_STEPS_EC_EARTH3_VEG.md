# Next Steps: EC-Earth3-Veg

## Goal

Use `EC-Earth3-Veg` as the first working model-as-truth case on the `ACCESS-ESM1-5` grid.

## Why This Model Next

- Already included in generated model-as-truth configs
- Does not have the UKESM `360_day` calendar problem in the current workflow path
- Better next validation target before scaling to other GCMs

## Configs To Use

3D:

- `config_interp_3d_ecearth3veg_to_access_hist.yaml`
- `config_interp_3d_ecearth3veg_to_access_ssp126_2080_2100.yaml`

Surface:

- `config_interp_surface_ecearth3veg_to_access_hist.yaml`
- `config_interp_surface_ecearth3veg_to_access_ssp126_2080_2100.yaml`

## Suggested Order

1. Run 3D interpolation for:
   - `ta`
   - `ua`
   - `va`
   - `hus`
2. Validate 3D outputs with `validate_interp_outputs.py`
3. Run surface interpolation for `tos`
4. Check surface files manually
5. Prepare SDMBCv2 one-grid-cell test using:
   - all required variables
   - full historical period
   - one grid cell only

## Example Commands

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src

for var in ta ua va hus; do
  ./submit_interp.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml "$var"
done

for var in ta ua va hus; do
  ./submit_interp.sh 3d config_interp_3d_ecearth3veg_to_access_ssp126_2080_2100.yaml "$var"
done
```

Surface:

```bash
./submit_interp.sh surface config_interp_surface_ecearth3veg_to_access_hist.yaml tos
./submit_interp.sh surface config_interp_surface_ecearth3veg_to_access_ssp126_2080_2100.yaml tos
```

Validation:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3

python validate_interp_outputs.py \
  --config config_interp_3d_ecearth3veg_to_access_hist.yaml \
  --var ta ua va hus
```

## Bias-Correction Reminder

For SDMBCv2, the meaningful first test is not a single variable.
The first real correction test should use:

- all variables together
- full historical period
- one grid cell only

This keeps the runtime small while preserving the multivariate climatological behavior.
