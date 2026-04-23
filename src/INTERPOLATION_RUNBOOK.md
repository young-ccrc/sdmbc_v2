# Interpolation Runbook

## Goal

Use this workflow for efficient, restartable interpolation runs on NCI.
It is intended for both:

- ERA5 to GCM
- GCM to GCM model-as-truth runs

The key principles are:

- run one variable per job
- split long periods into chunks
- keep restartable outputs: monthly for 3D variables, annual for daily SST
- rerun safely by skipping valid files
- run quick QA before moving to the next variable

## Recommended order

1. Test one variable first.
2. Run quick QA.
3. If clean, run the remaining variables.
4. Run quick QA on all variables.
5. Run the full validator on the final set before SDMBCv2.

For EC-Earth3-Veg to ACCESS-ESM1-5 historical, the order is:

- `ta`
- `ua`
- `va`
- `hus`
- `tos` separately for surface

## Recommended chunking

Do not run one full historical or future period in one job.

Use:

- historical: `1984-1994`, `1995-2004`, `2005-2014`
- future: `2080-2090`, `2091-2100`

This reduces walltime risk and makes restart much cheaper.

## Submit one chunk

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
STARTYEAR_OVERRIDE=1984 ENDYEAR_OVERRIDE=1994 ./submit_interp.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta interp_ecearth3veg_ta_1984_1994
```

## Submit multiple chunks and variables

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_interp_chunks.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta,ua,va,hus 1984-1994,1995-2004,2005-2014 ecearth_hist
```

For surface:

```bash
./submit_interp_chunks.sh surface config_interp_surface_ecearth3veg_to_access_hist.yaml tos 1984-1994,1995-2004,2005-2014 ecearth_sst_hist
```

Daily SST inputs are usually stored as annual files, so surface outputs are
annual by default, for example `tos_EC-Earth3-Veg_to_ACCESS-ESM1-5_1984.nc`.

If a reference or target GCM stores SST on a native `i/j` grid and you also
need that GCM's SST on its lat/lon orography grid, enable the optional
diagnostic remap in the surface config:

```yaml
write_reference_sst_latlon: true
paths:
  reference_sst_latlon_path: /path/to/reference_sst_latlon
```

Leave this disabled unless the extra reference-GCM SST files are needed.

## Resume interrupted runs

The interpolation scripts skip existing outputs only if they are valid.
For 3D variables this is monthly. For daily SST this is annual.
If a file is missing or incomplete, it will be recomputed.

Resume automatically from the first missing month:

```bash
cd /g/data/w28/yk8692/sdmbc_v2/src
./submit_interp_resume.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta
```

Resume only within a narrower year window:

```bash
STARTYEAR_OVERRIDE=2013 ENDYEAR_OVERRIDE=2014 ./submit_interp_resume.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta
```

## Quick QA

Run quick QA after each variable finishes.
This checks:

- expected output files exist
- file opens
- expected variable exists
- `time` exists and is non-empty
- first timestep is not fully NaN

```bash
cd /g/data/w28/yk8692
module use /g/data/xp65/public/modules
module load conda/analysis3

python3 /g/data/w28/yk8692/sdmbc_v2/src/qa_interp_summary.py   --config /g/data/w28/yk8692/sdmbc_v2/src/config_interp_3d_ecearth3veg_to_access_hist.yaml   --var ta
```

If quick QA passes, move to the next variable.
If quick QA fails, delete only the bad output files and rerun the affected year or chunk.

## Full validation

Run the slower full validator only after the whole variable set is ready.

```bash
python3 /g/data/w28/yk8692/sdmbc_v2/src/validate_interp_outputs.py   --config /g/data/w28/yk8692/sdmbc_v2/src/config_interp_3d_ecearth3veg_to_access_hist.yaml   --var ta ua va hus
```

## Storage-aware workflow

Treat interpolated NetCDF files as temporary working data, not as the
long-term product. For storage management, run historical first and do not
interpolate future data until the historical SDMBCv2 step is proven.

Recommended lifecycle:

1. Interpolate historical variables only.
2. Run quick QA and full validation on historical interpolation outputs.
3. Run SDMBCv2 on historical using all required variables.
4. Save persistent products:
   - SDMBCv2 parameter or transfer files
   - final bias-corrected historical outputs that are needed later
   - configs, job logs, and QA summaries
5. Confirm SDMBCv2 future correction can run from the saved parameter files.
6. Remove historical interpolated NetCDF files.
7. Interpolate future variables.
8. Apply the saved SDMBCv2 parameters to future.
9. Save final bias-corrected future outputs.
10. Remove future interpolated NetCDF files when no longer needed.

Before deleting any interpolated files, confirm:

- the relevant quick QA and full validation passed
- SDMBCv2 parameter files are complete and readable
- future correction does not need to reread historical interpolated files
- final outputs and logs are stored outside the temporary interpolation tree

Keep small traceability artifacts:

- exact YAML configs used
- PBS logs for successful runs
- QA summary outputs
- one or two representative interpolated files, if storage allows

Do not delete files from a source-model experiment until the next downstream
step has successfully consumed them.

## Suggested production workflow

For one source model:

1. Run historical `ta` chunked.
2. Run quick QA for historical `ta`.
3. Run historical `ua`, `va`, `hus` chunked.
4. Run quick QA for all historical 3D variables.
5. Run historical `tos` chunked.
6. Run quick QA for historical `tos`.
7. Run the full validator on the historical set.
8. Move to `BC_RUNBOOK.md` for SDMBCv2 testing and production correction.

## Watchdog timeout

Interpolation jobs now include a watchdog.
If no new output file appears for a configured period, the PBS job terminates the Python process and exits.
This prevents long-running stalled jobs from consuming service units indefinitely.

Default timeout:

- `60` minutes without a new output file

Override it when submitting:

```bash
WATCHDOG_MINUTES_OVERRIDE=90 ./submit_interp.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta
```

When the watchdog triggers, the log prints a resume command using `submit_interp_resume.sh`.

## Resource guidance

Current defaults are conservative and suitable for heavy 3D jobs:

- `ncpus=48`
- `mem=190GB`
- `jobfs=50GB`

Override only after benchmarking:

```bash
NCPUS_OVERRIDE=24 MEM_GB_OVERRIDE=120 JOBFS_GB_OVERRIDE=30 ./submit_interp.sh 3d config_interp_3d_ecearth3veg_to_access_hist.yaml ta
```

## Practical notes

- Prefer restartability over maximum parallel submission.
- Do not submit all variables and all chunks at once until one full variable has passed QA.
- Keep failed and suspicious files small in scope by rerunning only the affected chunk.
- For model-as-truth experiments, complete interpolation and QA for all variables before SDMBCv2.
