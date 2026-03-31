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
