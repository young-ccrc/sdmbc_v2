#!/bin/bash
set -eu

usage() {
    cat <<'EOF'
Usage:
  submit_bc_3d_tile_groups.sh <base_config> <total_tiles> <group_size> <job_prefix>

Submits ONE PBS job array covering all tile groups for a single BC config.
Each array task processes tiles [index*group_size, min((index+1)*group_size,
total_tiles)) via SDMBC_TILE_START/SDMBC_TILE_END, keeping Dask at 1 worker x
1 thread per task (the only numerically validated setting -- see
TROUBLESHOOTING_BC_MODEL_AS_TRUTH.md). Throughput comes from PBS-level
fan-out across array tasks, not intra-job Dask parallelism.

total_tiles must be known ahead of submission (e.g. from a prior run's log
line "[INFO] processing X of Y tiles", where Y is total_tiles).

Optional environment overrides:
  NCPUS_OVERRIDE
  MEM_GB_OVERRIDE
  JOBFS_GB_OVERRIDE
  WALLTIME_OVERRIDE
  MODULE_ANALYSIS_OVERRIDE
  DASK_N_WORKERS_OVERRIDE       (default 1; do not raise without re-running
                                 the isolated multiprocessing benchmark first)
  DASK_THREADS_PER_WORKER_OVERRIDE (default 1)
  DASK_PROCESSES_OVERRIDE
  JOB_DEPENDENCY_OVERRIDE       (e.g. afterok:12345.gadi-pbs:12346.gadi-pbs)
  DRY_RUN=1

Example:
  WALLTIME_OVERRIDE=12:00:00 MEM_GB_OVERRIDE=128 \
  ./submit_bc_3d_tile_groups.sh \
    config_bc_3d_ecearth3veg_to_access_hist_tile_l20.yaml \
    140 5 bc_l20_group

Dry run:
  DRY_RUN=1 ./submit_bc_3d_tile_groups.sh base.yaml 140 5 bc_l20_group
EOF
}

if [ "$#" -ne 4 ]; then
    usage
    exit 1
fi

BASE_CONFIG=$(realpath "$1")
TOTAL_TILES=$2
GROUP_SIZE=$3
JOB_PREFIX=$4
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PBS_SCRIPT="$SCRIPT_DIR/sdmbc_3d_tile_group_job.pbs"
PYTHON_SCRIPT="$SCRIPT_DIR/sdmbc_main_3d.py"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Base config not found: $BASE_CONFIG" >&2
    exit 1
fi

case "$TOTAL_TILES" in
    ''|*[!0-9]*) echo "total_tiles must be a positive integer" >&2; exit 1 ;;
esac

case "$GROUP_SIZE" in
    ''|*[!0-9]*) echo "group_size must be a positive integer" >&2; exit 1 ;;
esac

if [ "$TOTAL_TILES" -lt 1 ] || [ "$GROUP_SIZE" -lt 1 ]; then
    echo "total_tiles and group_size must be >= 1" >&2
    exit 1
fi

NUM_GROUPS=$(( (TOTAL_TILES + GROUP_SIZE - 1) / GROUP_SIZE ))
LAST_INDEX=$((NUM_GROUPS - 1))

NCPUS=${NCPUS_OVERRIDE:-8}
MEM_GB=${MEM_GB_OVERRIDE:-64}
JOBFS_GB=${JOBFS_GB_OVERRIDE:-20}
WALLTIME=${WALLTIME_OVERRIDE:-12:00:00}
# Keep BC jobs on Python 3.11 until mrmbc is rebuilt for Python 3.12.
MODULE_ANALYSIS=${MODULE_ANALYSIS_OVERRIDE:-analysis3-26.02}
DASK_N_WORKERS=${DASK_N_WORKERS_OVERRIDE:-1}
DASK_THREADS_PER_WORKER=${DASK_THREADS_PER_WORKER_OVERRIDE:-1}
DASK_PROCESSES=${DASK_PROCESSES_OVERRIDE:-true}

echo "Total tiles: $TOTAL_TILES, group size: $GROUP_SIZE, array tasks: $NUM_GROUPS (0-$LAST_INDEX)"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN: qsub -N ${JOB_PREFIX} -J 0-${LAST_INDEX} -l walltime=${WALLTIME} -l mem=${MEM_GB}GB -l ncpus=${NCPUS} -l jobfs=${JOBFS_GB}GB ${JOB_DEPENDENCY_OVERRIDE:+-W depend=${JOB_DEPENDENCY_OVERRIDE}} -v CONFIG=${BASE_CONFIG},GROUP_SIZE=${GROUP_SIZE},TOTAL_TILES=${TOTAL_TILES},NCPUS=${NCPUS},MEM_GB=${MEM_GB},MODULE_ANALYSIS=${MODULE_ANALYSIS},SCRIPT_PATH=${PYTHON_SCRIPT},DASK_N_WORKERS=${DASK_N_WORKERS},DASK_THREADS_PER_WORKER=${DASK_THREADS_PER_WORKER},DASK_PROCESSES=${DASK_PROCESSES} ${PBS_SCRIPT}"
    exit 0
fi

qsub \
    -N "$JOB_PREFIX" \
    -J "0-${LAST_INDEX}" \
    -l walltime="$WALLTIME" \
    -l mem="${MEM_GB}GB" \
    -l ncpus="$NCPUS" \
    -l jobfs="${JOBFS_GB}GB" \
    ${JOB_DEPENDENCY_OVERRIDE:+-W depend="$JOB_DEPENDENCY_OVERRIDE"} \
    -v CONFIG="$BASE_CONFIG",GROUP_SIZE="$GROUP_SIZE",TOTAL_TILES="$TOTAL_TILES",NCPUS="$NCPUS",MEM_GB="$MEM_GB",MODULE_ANALYSIS="$MODULE_ANALYSIS",SCRIPT_PATH="$PYTHON_SCRIPT",DASK_N_WORKERS="$DASK_N_WORKERS",DASK_THREADS_PER_WORKER="$DASK_THREADS_PER_WORKER",DASK_PROCESSES="$DASK_PROCESSES" \
    "$PBS_SCRIPT"
