# Interpolation Workflow

## Why this file exists

The interpolation workflow now supports two common use cases:

1. ERA5 to GCM
2. GCM to another GCM for model-as-truth experiments

The goal is to avoid editing PBS scripts and YAML files in place for every run.

## Recommended practice

Do not keep one "working" YAML file that you edit repeatedly.
Create one YAML per experiment and keep the filename descriptive.

Examples:

- `config_interp_3d_era5_to_access_hist.yaml`: ERA5 to ACCESS-ESM1-5 historical
- `config_interp_3d_era5_to_access_ssp370.yaml`: ERA5 to ACCESS-ESM1-5 future
- `config_interp_3d_cnrm_to_access_hist.yaml`: CNRM-CM6-1 to ACCESS-ESM1-5 historical
- `config_interp_surface_cnrm_to_access_hist.yaml`: CNRM-CM6-1 to ACCESS-ESM1-5 surface historical

## YAML structure

Each config uses grouped sections.

- `paths`: filesystem locations for source data, target data, and outputs
- `source`: the input dataset being interpolated
- `target`: the destination GCM grid and metadata
- `domain`: latitude and longitude subset
- `period`: years to process
- `resources`: default runtime resources
- `future`: only for future runs that need a historical bridge such as `hist_target_path`

## Which keys matter most

### For ERA5 to GCM

Set:

- `source.input_model: reanalysis`
- `paths.input_path`
- `paths.input_path_sst`
- `paths.input_z_path`
- `target.*`

### For GCM to GCM

Set:

- `source.input_model: gcm`
- `source.input_gname`
- `source.input_infor`
- `source.input_period`
- `source.input_cinfor`
- `source.input_sinfor`
- `source.input_version`
- `paths.input_path` for 3D or `paths.input_path_sst` for surface
- `target.*`

For 3D GCM-to-GCM runs, `source.input_z_path` can be `null` when the source model geopotential is obtained from the source files or computed path logic.

## Submission

Use the wrapper script instead of editing PBS files directly.

### 3D

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_interp.sh 3d config_interp_3d_era5_to_access_hist.yaml ta
./submit_interp.sh 3d config_interp_3d_cnrm_to_access_hist.yaml ta
```

### Surface

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_interp.sh surface config_interp_surface_era5_to_access_hist.yaml tos
./submit_interp.sh surface config_interp_surface_cnrm_to_access_hist.yaml tos
```

## Comments in YAML vs documentation

Do not comment every line in the YAML.
That usually makes the file harder to scan.

Use this split instead:

- keep YAML comments short and only where users may hesitate
- keep the fuller explanation in this Markdown file
- keep one config file per experiment so users mostly do not need to edit internals

## Suggested naming convention

Use filenames that describe source, target, dimensionality, and period.

Examples:

- `config_interp_3d_era5_to_access_hist.yaml`
- `config_interp_3d_era5_to_access_ssp370.yaml`
- `config_interp_3d_cnrm_to_access_hist.yaml`
- `config_interp_surface_cnrm_to_access_hist.yaml`

That keeps the workflow explicit and reduces mistakes.

## Legacy filenames

The older config filenames are still kept in the repository for compatibility, but new work should prefer the descriptive names above.

## Model-as-truth batch workflow

ACCESS-ESM1-5 is used as the fixed target grid.
The source models currently configured for production-style use are:

- EC-Earth3-Veg
- MPI-ESM1-2-HR
- UKESM1-0-LL surface

Historical batch period:

- 1984 to 2014

Future batch period:

- 2080 to 2100 using `ssp126`

Files added for this workflow:

- `generate_model_as_truth_configs.py`
- `submit_model_as_truth_hist.sh`
- `submit_model_as_truth_future.sh`
- `stage_ukesm_3d_to_gadi.sh`
- `stage_ukesm_3d_to_gadi.pbs`
- `submit_stage_ukesm_3d.sh`

Generate or refresh the configs:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./generate_model_as_truth_configs.py
```

Submit historical batch jobs:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_model_as_truth_hist.sh
```

Submit future batch jobs:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_model_as_truth_future.sh
```

Stage UKESM1-0-LL 3D data from Squall to Gadi:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_stage_ukesm_3d.sh historical
./submit_stage_ukesm_3d.sh ssp126
```

By default this runs directly on the login node, because SSH from PBS compute jobs to Squall may be unreachable. If you need the old behavior explicitly, use:

```bash
./submit_stage_ukesm_3d.sh historical pbs
./submit_stage_ukesm_3d.sh ssp126 pbs
```

The staging workflow copies files from Squall into a local raw directory and then creates a CMIP-like tree on Gadi so the existing interpolation code can use it directly.

Current limitations and decisions:

- `NorESM2-MM` is excluded from the batch workflow because its 6-hour time steps are offset to `03 09 15 21`, which does not align with the standard `00 06 12 18` bias-correction workflow.
- `UKESM1-0-LL` 3D configs are generated to use the local staged path on Gadi, not the remote Squall path.
- `UKESM1-0-LL` 3D interpolation should be tested with one variable first after staging because the source files begin at `0600` rather than `0000`.

