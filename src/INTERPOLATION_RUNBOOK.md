# Interpolation Runbook

## Goal

Use this workflow for efficient, restartable interpolation runs on NCI.
It is intended for both:

- ERA5 to GCM
- GCM to GCM model-as-truth runs

The key principles are:

- run one variable per job
- split long periods into chunks
- keep monthly outputs
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

## Resume interrupted runs

The interpolation script skips monthly files only if they are valid.
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

- expected monthly files exist
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
If quick QA fails, delete only the bad monthly files and rerun the affected year or chunk.

## Full validation

Run the slower full validator only after the whole variable set is ready.

```bash
python3 /g/data/w28/yk8692/sdmbc_v2/src/validate_interp_outputs.py   --config /g/data/w28/yk8692/sdmbc_v2/src/config_interp_3d_ecearth3veg_to_access_hist.yaml   --var ta ua va hus
```

## Suggested production workflow

For one source model:

1. Run `ta` chunked.
2. Run quick QA for `ta`.
3. Run `ua`, `va`, `hus` chunked.
4. Run quick QA for all 3D variables.
5. Run `tos` chunked.
6. Run quick QA for `tos`.
7. Run the full validator.
8. Move to SDMBCv2 testing.

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
