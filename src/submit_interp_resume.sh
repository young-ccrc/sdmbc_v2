#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_interp_resume.sh <kind> <config> <var> [job_name]

Arguments:
  kind     3d | surface
  config   Path to YAML config
  var      Variable name, e.g. ta, ua, va, hus, tos
  job_name Optional job name

Optional environment overrides:
  STARTYEAR_OVERRIDE
  ENDYEAR_OVERRIDE
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE

Behavior:
  - Scans monthly outputs in the configured output directory
  - Finds the first missing YYYY-MM file in the requested year range
  - Submits from that year to the requested end year
  - Existing files are still skipped by interp_obs2gcm.py
EOF
}

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    usage
    exit 1
fi

KIND=$1
CONFIG=$(realpath "$2")
VAR=$3
JOB_NAME=${4:-}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUBMIT_SCRIPT="$SCRIPT_DIR/submit_interp.sh"

if [ ! -f "$CONFIG" ]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

if [ ! -x "$SUBMIT_SCRIPT" ]; then
    echo "submit_interp.sh not found or not executable: $SUBMIT_SCRIPT" >&2
    exit 1
fi

mapfile -t INFO < <(python3 - "$CONFIG" "$VAR" <<'PYINFO'
import sys, yaml
from pathlib import Path
cfg_path = Path(sys.argv[1])
var = sys.argv[2]
with cfg_path.open() as f:
    data = yaml.safe_load(f) or {}
merged = dict(data)
for section in ("paths", "source", "target", "domain", "period", "resources", "future"):
    if isinstance(data.get(section), dict):
        merged.update(data[section])
source_name = merged["input_model"] if merged["input_model"] == "reanalysis" else merged["input_gname"]
print(merged["output_path"])
print(source_name)
print(merged["gname"])
print(merged["startyear_h"])
print(merged["endyear_h"])
PYINFO
)

OUTPUT_PATH=${INFO[0]}
SOURCE_NAME=${INFO[1]}
TARGET_NAME=${INFO[2]}
CONFIG_START=${INFO[3]}
CONFIG_END=${INFO[4]}

STARTYEAR=${STARTYEAR_OVERRIDE:-$CONFIG_START}
ENDYEAR=${ENDYEAR_OVERRIDE:-$CONFIG_END}

FIRST_MISSING=""
for YEAR in $(seq "$STARTYEAR" "$ENDYEAR"); do
    for MONTH in $(seq 1 12); do
        FILE=$(printf '%s/%s_%s_to_%s_%04d-%02d.nc' "$OUTPUT_PATH" "$VAR" "$SOURCE_NAME" "$TARGET_NAME" "$YEAR" "$MONTH")
        if [ ! -f "$FILE" ]; then
            FIRST_MISSING=$(printf '%04d-%02d' "$YEAR" "$MONTH")
            break 2
        fi
    done
done

if [ -z "$FIRST_MISSING" ]; then
    echo "All monthly outputs already exist for $VAR in $STARTYEAR-$ENDYEAR"
    exit 0
fi

RESUME_YEAR=${FIRST_MISSING%-*}
RESUME_MONTH=${FIRST_MISSING#*-}
if [ -z "$JOB_NAME" ]; then
    JOB_NAME="resume_${KIND}_${VAR}_${RESUME_YEAR}_${RESUME_MONTH}"
fi

echo "First missing output for $VAR: $FIRST_MISSING"
echo "Submitting from year $RESUME_YEAR to $ENDYEAR with job name $JOB_NAME"

STARTYEAR_OVERRIDE="$RESUME_YEAR" ENDYEAR_OVERRIDE="$ENDYEAR" "$SUBMIT_SCRIPT" "$KIND" "$CONFIG" "$VAR" "$JOB_NAME"
