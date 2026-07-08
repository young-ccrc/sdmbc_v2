# SDMBCv2 Workflow Index

This file lists the current entry points for the model-as-truth workflow.
Generated configs, PBS logs, and large outputs should live outside the git
tree whenever possible.

Detailed runbooks:

- Interpolation: `runbooks/INTERPOLATION_RUNBOOK.md`
- Bias correction: `runbooks/BC_RUNBOOK.md`
- `mrmbc` build/import issues: `runbooks/MRMBC_BUILD.md`

## Interpolation

Use these for EC-Earth3-Veg to ACCESS-ESM1-5 model-as-truth inputs.

- 3D config: `config_interp_3d_ecearth3veg_to_access_hist.yaml`
- Surface config: `config_interp_surface_ecearth3veg_to_access_hist.yaml`
- Submit one variable: `submit_interp.sh`
- Submit chunks/resume: `submit_interp_chunks.sh`, `submit_interp_resume.sh`
- QA: `validate_interp_outputs.py`

Example:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_interp.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta
```

## 3D Bias Correction

Current model-as-truth configs:

- One-grid smoke test: `config_bc_3d_ecearth3veg_to_access_hist_onegrid.yaml`
- Small tile, one level: `config_bc_3d_ecearth3veg_to_access_hist_tile_l0.yaml`
- Small tile, levels 0-4: `config_bc_3d_ecearth3veg_to_access_hist_tile_l0_4.yaml`

Submit one config:

```bash
./submit_bc_3d.sh config_bc_3d_ecearth3veg_to_access_hist_tile_l0.yaml bc_ecearth_tile_l0_hist
```

Submit independent per-level jobs from one base config:

```bash
DRY_RUN=1 ./submit_bc_3d_levels.sh \
  config_bc_3d_ecearth3veg_to_access_hist_tile_l0_4.yaml \
  0 4 bc_ecearth_tile

WALLTIME_OVERRIDE=06:00:00 MEM_GB_OVERRIDE=64 ./submit_bc_3d_levels.sh \
  config_bc_3d_ecearth3veg_to_access_hist_tile_l0_4.yaml \
  0 4 bc_ecearth_tile
```

Generated per-level configs default to:

```text
/scratch/n81/$USER/sdmbc_v2_generated_configs/bc_3d_levels
```

Override with `GENERATED_CONFIG_DIR=/path/to/dir` if needed.

Submit one PBS array job covering all tile groups for a single config (the
only numerically validated way to scale BC throughput -- see
`runbooks/BC_RUNBOOK.md#scaling-up-with-pbs-tile-groups`):

```bash
DRY_RUN=1 ./submit_bc_3d_tile_groups.sh \
  config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml \
  140 5 bc_l20_group
```

## Reformat

Reconstruct BC output back into the target GCM's original NetCDF structure,
chained after a BC job:

```bash
JOB_DEPENDENCY_OVERRIDE=afterok:<bc_job_id> \
./submit_reformat.sh config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml bc_l20_reformat
```

## Climatology And Plots

Use these only after BC outputs are verified.

- Compute climatology/bias: `submit_compute_bc_climatology.sh`
- Plot bias maps: `submit_plot_bc_bias_map.sh`
- Combined compute plus plot dependency: `submit_bc_climatology_pipeline.sh`

Example:

```bash
./submit_bc_climatology_pipeline.sh \
  /path/to/bc_corrected.nc \
  config_bc_3d_ecearth3veg_to_access_hist_tile_l0.yaml \
  bc_climo
```

## Keep Git Clean

Keep in git:

- source code
- reusable submit helpers
- reusable base configs
- runbooks and workflow notes

Keep outside git:

- generated per-level/per-tile configs
- PBS `.o*` and `.e*` logs
- scratch preprocessing files
- final NetCDF outputs
- exploratory one-off dumps

Suggested locations:

- generated configs: `/scratch/n81/$USER/sdmbc_v2_generated_configs`
- temporary BC inputs: `/scratch/n81/$USER/sdmbc_v2_temp`
- persistent outputs: `/g/data/w28/yk8692/output`
