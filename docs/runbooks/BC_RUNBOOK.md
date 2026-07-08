# SDMBCv2 Bias-Correction Runbook

## Goal

Use this workflow after interpolation outputs have passed QA.

This document has two workflows:

- typical BC: reanalysis such as ERA5 is the observation/reference and a GCM is corrected
- model-as-truth BC: one GCM is treated as the truth/reference and another GCM is corrected

For the first test, run 3D atmospheric variables only:

- `hus`
- `ta`
- `ua`
- `va`

SST (`tos`) is corrected later with the surface workflow.

## Typical Reanalysis-As-Observation Workflow

This is the standard SDMBCv2 use case.

Role mapping:

- GCM to correct: target GCM historical data, for example `ACCESS-ESM1-5`
- observation/reference: ERA5 already interpolated to the target GCM grid and levels

Required historical inputs:

- target GCM historical 3D files
- ERA5 3D files horizontally and vertically interpolated to the target GCM
- matching variables, grid, vertical levels, and time period

Example paths for ACCESS-ESM1-5:

```text
GCM:
/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1/6hrLev

ERA5 interpolated to ACCESS:
/g/data/w28/yk8692/input/era5/access_esm
```

Typical config intent:

```yaml
input_model: reanalysis
bc_hist_path: /path/to/target/GCM/historical/6hrLev
obs_path: /path/to/era5_interpolated_to_target_grid
gname: ACCESS-ESM1-5
target_variable: [hus, ta, ua, va]
bc_hist: true
bc_future: false
```

Start with:

1. one grid cell
2. one vertical level
3. all 3D variables together
4. full calibration period, ideally about 30 years

After that passes, scale to:

1. more levels for the same grid cell
2. a small spatial tile
3. full historical domain
4. future application using saved historical BC parameters

## Model-As-Truth Workflow

Use this workflow when evaluating the method by treating one GCM as the
truth/reference.

Role mapping:

- GCM to correct: the target model, for example `ACCESS-ESM1-5`
- truth/reference: another GCM interpolated to the target grid and vertical levels, for example `EC-Earth3-Veg`

Required historical inputs:

- target GCM historical 3D files
- source/truth GCM interpolated to the target GCM grid and vertical levels
- matching variables, grid, vertical levels, and time period

For EC-Earth3-Veg to ACCESS-ESM1-5:

```text
GCM to correct:
/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1/6hrLev

Truth/reference:
/g/data/w28/yk8692/input/model_as_truth/ecearth3veg_to_access/hist/3d
```

Typical config intent:

```yaml
input_model: model_as_truth
bc_hist_path: /path/to/target/GCM/historical/6hrLev
obs_path: /path/to/source_GCM_interpolated_to_target_grid
gname: ACCESS-ESM1-5
truth_gname: EC-Earth3-Veg
target_variable: [hus, ta, ua, va]
bc_hist: true
bc_future: false
```

## Required Historical Checks

Before running BC, confirm:

- 3D interpolation QA passed for `hus`, `ta`, `ua`, `va`
- all historical monthly files exist for the calibration period
- files have non-empty `time`
- files are on the same target grid and vertical levels

## Config Templates

Start new BC experiments from a template instead of copying old run configs.
This avoids accidentally reusing stale paths, years, domains, or output
directories from a previous test.

Templates are stored in:

```text
src/config_templates
```

Use:

```bash
cp src/config_templates/bc_3d_model_as_truth_template.yaml \
  src/config_bc_3d_<truth>_to_<target>_<period>_<purpose>.yaml
```

or:

```bash
cp src/config_templates/bc_3d_reanalysis_template.yaml \
  src/config_bc_3d_<reanalysis>_to_<target>_<period>_<purpose>.yaml
```

Before submitting, replace every `CHANGE_ME` value and check:

- `bc_hist_path`
- `bc_future_path`
- `obs_path`
- `out_path`
- `temp_root`
- `input_model`
- `gname`
- `truth_gname`
- historical and future years
- domain bounds or one-grid point
- `max_tile_size`

`max_tile_size` means maximum horizontal grid cells per tile, not number of
tiles. A value of `1500` is a reasonable starting point for tile tests; lower it
if memory is too high, or raise it if tiles are too small and overhead dominates.

Use `temp_root` on scratch for preprocessing and intermediate tile files. Keep
`out_path` on `/g/data` for final BC outputs and parameters.

## Python And `mrmbc`

SDMBCv2 uses the compiled Fortran extension `mrmbc`.
Compiled Python extensions are Python-version specific.

The current prebuilt extension is:

```text
mrmbc.cpython-311-x86_64-linux-gnu.so
```

So BC jobs should currently use Python 3.11.
On NCI:

```bash
module use /g/data/xp65/public/modules
module load conda/analysis3-26.02
```

Check import:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
python3 - <<'PY'
from mrmbc import mbc_subroutines as mbc
print("mrmbc import ok")
PY
```

If Python changes, rebuild `mrmbc`:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./build_mrmbc.sh
```

See `MRMBC_BUILD.md` for details.

## One-Grid 3D Historical Test

Start with one grid cell and one vertical level.
This validates the multivariate 3D workflow without paying the full-domain cost.

For a typical ERA5-as-observation workflow, create an equivalent one-grid
config using:

- `input_model: reanalysis`
- `obs_path` pointing to ERA5 interpolated to the target GCM
- `bc_hist_path` pointing to the target GCM historical data

For the current model-as-truth EC-Earth3-Veg test, use:

Config:

```text
config_bc_3d_ecearth3veg_to_access_hist_onegrid.yaml
```

Submit:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_bc_3d.sh config_bc_3d_ecearth3veg_to_access_hist_onegrid.yaml bc_ecearth_onegrid_hist
```

The config uses:

- ACCESS-ESM1-5 historical 6hrLev as the GCM to correct
- interpolated EC-Earth3-Veg historical 3D outputs as truth/reference
- `1984-2014`
- one ACCESS grid cell near `lat=12.5`, `lon=116.25`
- `slevel=0`, `elevel=0`
- `bc_future: false`

Watch the log:

```bash
tail -f bc_ecearth_onegrid_hist.o<jobid>
```

Expected persistent outputs are written under:

```text
/g/data/w28/yk8692/output/model_as_truth/ecearth3veg_to_access/bc_onegrid_hist_3d
```

## What To Check After The Test

Confirm that the job:

- imports `mrmbc`
- finds ACCESS GCM files
- finds interpolated EC-Earth truth files
- preprocesses all four variables
- writes BC parameters
- writes BC output if `save_bc_output: true`
- exits with status `0`

Check output files:

```bash
ls -lh /g/data/w28/yk8692/output/model_as_truth/ecearth3veg_to_access/bc_onegrid_hist_3d
```

At minimum, retain:

- BC parameter NetCDF files
- BC corrected output files if enabled
- config used
- PBS log

## If The One-Grid Test Passes

Next steps:

1. Run more vertical levels, still one grid cell.
2. Run a small spatial tile for all vertical levels.
3. Run the full historical domain only after the small tile is stable.
4. Save BC parameters and required historical BC outputs.
5. Confirm future correction can use the saved historical BC parameters.
6. Remove interpolated historical NetCDF intermediates only after BC outputs are verified.

## Scaling Up With PBS Tile Groups

Only 1 Dask worker x 1 thread is numerically validated for BC (see
`TROUBLESHOOTING_BC_MODEL_AS_TRUTH.md`); every multi-worker attempt tested so
far has silently corrupted output. Throughput therefore comes from
PBS-level parallelism across tile groups, not from raising
`DASK_N_WORKERS`/`DASK_THREADS_PER_WORKER`.

`submit_bc_3d_tile_groups.sh` submits one PBS job array covering all tile
groups for a config, with each array task processing a disjoint tile range
at 1 worker x 1 thread:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_bc_3d_tile_groups.sh \
  config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml \
  140 5 bc_l20_group
```

The first argument after the config is `total_tiles`, which must be known
ahead of submission -- check a prior run's log for the line
`[INFO] processing X of Y tiles`, where `Y` is `total_tiles`. The second is
`group_size` (tiles per array task); pick it relative to `max_tile_size` in
the config, trading off array-task count against per-task memory/walltime.

Dry-run first to sanity-check the array size:

```bash
DRY_RUN=1 ./submit_bc_3d_tile_groups.sh \
  config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml 140 5 bc_l20_group
```

Only the array task covering the final tile group attempts to merge tile
outputs; earlier tasks run with `SDMBC_SKIP_MERGE=true` automatically.

## If The One-Grid Test Fails

Common failure classes:

- `ModuleNotFoundError: No module named 'mrmbc'`
  - wrong Python ABI for the compiled extension
  - use `analysis3-26.02` or rebuild `mrmbc`
- missing input files
  - check `bc_hist_path`, `obs_path`, years, and variable names
- time/calendar mismatch
  - check that the interpolated truth files and ACCESS GCM files cover the same period
- all-NaN data
  - rerun interpolation QA and repair bad monthly files
- memory or walltime
  - keep one-grid or tile size small until stable

## Storage Policy

Treat interpolated NetCDF files as temporary BC inputs.

Keep:

- BC parameter files
- final BC outputs
- configs
- PBS logs
- QA summaries

Delete interpolated historical files only after:

- one-grid and production BC outputs are verified
- BC parameters are saved and reusable
- future correction does not need to reread the historical interpolation files

Future interpolation should be delayed until historical BC is proven.

## Reformatting Bias-Corrected Output

After BC output is verified, `reformat_gcm2origin.py` reconstructs it back
into the target GCM's original NetCDF/CMIP6 structure. Submit it chained
`afterok` the BC job so it only runs once BC succeeds:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
JOB_DEPENDENCY_OVERRIDE=afterok:<bc_job_id> \
./submit_reformat.sh config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml bc_l20_reformat
```

Omit `JOB_DEPENDENCY_OVERRIDE` to run it standalone against already-verified
BC output.

## Climatology And Bias Maps

Run climatology and bias statistics in a separate PBS job after the BC output
has been verified.

Submit the compute job with:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_compute_bc_climatology.sh /path/to/bc_corrected.nc config_bc_3d_ecearth3veg_to_access_hist_onegrid.yaml bc_climo
```

The compute job writes:

- a climatology NetCDF file
- a bias NetCDF file when `--config` or `--reference-input` is supplied

Generate quick-look bias maps from the bias NetCDF with:

```bash
python plot_bc_bias_map.py /path/to/bc_climatology_bias.nc -o /path/to/plots
```

For PBS use, submit the plot job with:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_plot_bc_bias_map.sh /path/to/bc_climatology_bias.nc plots bc_bias_plot
```

If you want a single command to submit compute and then plotting, use:

```bash
./submit_bc_climatology_pipeline.sh /path/to/bc_corrected.nc config_bc_3d_ecearth3veg_to_access_hist_onegrid.yaml bc_climo
```

That pipeline submits the plot job with an `afterok` dependency on the compute
job, so the plot step only starts if the compute step succeeds.

For monthly or seasonal bias panels, pass a glob and a facet dimension:

```bash
python plot_bc_bias_map.py /path/to/bc_climatology_bias.nc \
  --pattern '*_mean_monthly_bias' \
  --facet-dim month
```
