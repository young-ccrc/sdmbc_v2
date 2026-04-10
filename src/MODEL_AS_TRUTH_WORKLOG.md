# Model-as-Truth Work Log

## Branch

- Working branch: `interp-obs2gcm-dev`
- Stable target grid for model-as-truth work: `ACCESS-ESM1-5`

## Workflow Changes Already Made

- Interpolation configs were reorganized into grouped YAML sections:
  - `paths`
  - `source`
  - `target`
  - `domain`
  - `period`
  - `resources`
  - `future`
- Generic PBS submission was added:
  - `interp_3d_job.pbs`
  - `interp_surface_job.pbs`
  - `submit_interp.sh`
- Model-as-truth config generation was added:
  - `generate_model_as_truth_configs.py`
- Batch submission helpers were added:
  - `submit_model_as_truth_hist.sh`
  - `submit_model_as_truth_future.sh`

## Environment Changes Already Made

- PBS module path updated from deprecated `hh5` to `xp65`
- Generic interpolation jobs now use:
  - `module use /g/data/xp65/public/modules`
  - `module load conda/analysis3`
- PBS email alarms were removed from generic interpolation and staging jobs

## UKESM1-0-LL Findings

- UKESM 3D files were staged to Gadi successfully from the login node
- PBS compute nodes could not reach Squall over SSH, so staging now defaults to login-node execution
- 3D interpolation jobs for UKESM appeared to finish, but outputs are not usable

### Why UKESM 3D Failed

- UKESM source files use `360_day` calendar
- ACCESS target files use `proleptic_gregorian`
- The interpolation code aligned time-dependent inputs by exact timestamps
- Exact time intersection became empty
- Resulting NetCDF outputs were written with:
  - `time = UNLIMITED ; // (0 currently)`

### Consequence

- UKESM 3D model-as-truth outputs should not be used for SDMBCv2
- UKESM would need a separate calendar-conversion design before it can be used safely

## UKESM Decision

- Do not use UKESM as the first model-as-truth case
- Keep UKESM surface and staging notes for later follow-up
- Move to `EC-Earth3-Veg` next

## Other Model Decisions

- `NorESM2-MM` is excluded for now because its 6-hour time steps are `03 09 15 21`
- Preferred next model-as-truth candidates:
  - `EC-Earth3-Veg`
  - `MPI-ESM1-2-HR`

## Validation Work

- A quick validation script was added:
  - `validate_interp_outputs.py`
- Purpose:
  - check monthly output existence
  - check variable presence
  - check non-empty time dimension
  - check `time`, `lat`, `lon`, `lev` consistency against target files

## Recommended Immediate Next Step

1. Use `EC-Earth3-Veg` as the first full model-as-truth case
2. Run 3D interpolation
3. Run surface SST interpolation
4. Validate outputs
5. Run one-grid-cell SDMBCv2 test using all variables and the full intended historical period
