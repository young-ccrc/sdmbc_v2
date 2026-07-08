#!/usr/bin/env python3
"""Shared model-as-truth metadata for config generators.

Both the interpolation-config generator (generate_model_as_truth_configs.py)
and the BC-config generator (generate_bc_model_as_truth_configs.py) need the
same truth-GCM and target-GCM metadata. Keeping one copy here avoids the two
generators drifting apart.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path('/g/data/w28/yk8692/input/model_as_truth')
UKESM_STAGE_ROOT = Path('/g/data/w28/yk8692/input/model_as_truth/ukesm1_0_ll_6hrlev_stage')
UKESM_STAGE_VERSION = 'v20260401'

TARGET = {
    'historical': {
        'target_path': '/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1',
        'target_path_sst': '/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1',
        'target_g_path': None,
        'target_orog_path': '/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1',
        'infor': '6hrLev',
        'gname': 'ACCESS-ESM1-5',
        'period': 'historical',
        'cinfor': 'r6i1p1f1',
        'sinfor': 'gn',
        'version': 'v20200529',
    },
    'ssp126': {
        'target_path': '/g/data/fs38/publications/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp126/r6i1p1f1',
        'target_path_sst': '/g/data/fs38/publications/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp126/r6i1p1f1',
        'target_g_path': None,
        'target_orog_path': '/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1',
        'infor': '6hrLev',
        'gname': 'ACCESS-ESM1-5',
        'period': 'ssp126',
        'cinfor': 'r6i1p1f1',
        'sinfor': 'gn',
        'version': 'v20200529',
        'future': {
            'hist_target_path': '/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r6i1p1f1',
        },
    },
}

MODELS = {
    'EC-Earth3-Veg': {
        'label': 'ecearth3veg',
        'member_hist': 'r1i1p1f1',
        'member_future': 'r1i1p1f1',
        'three_d': {
            'hist_root': '/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3-Veg/historical/r1i1p1f1',
            'future_root': '/g/data/oi10/replicas/CMIP6/ScenarioMIP/EC-Earth-Consortium/EC-Earth3-Veg/ssp126/r1i1p1f1',
            'sinfor': 'gr',
            'version_hist': 'v20210601',
            'version_future': 'v20210601',
        },
        'surface': {
            'hist_root': '/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3-Veg/historical/r1i1p1f1',
            'future_root': '/g/data/oi10/replicas/CMIP6/ScenarioMIP/EC-Earth-Consortium/EC-Earth3-Veg/ssp126/r1i1p1f1',
            'sinfor': 'gn',
            'version_hist': 'v20211207',
            'version_future': 'v20200919',
        },
    },
    'MPI-ESM1-2-HR': {
        'label': 'mpi_esm1_2_hr',
        'member_hist': 'r1i1p1f1',
        'member_future': 'r1i1p1f1',
        'three_d': {
            'hist_root': '/g/data/oi10/replicas/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1',
            'future_root': '/g/data/oi10/replicas/CMIP6/ScenarioMIP/DKRZ/MPI-ESM1-2-HR/ssp126/r1i1p1f1',
            'sinfor': 'gn',
            'version_hist': 'v20190710',
            'version_future': 'v20190710',
        },
        'surface': {
            'hist_root': '/g/data/oi10/replicas/CMIP6/CMIP/MPI-M/MPI-ESM1-2-HR/historical/r1i1p1f1',
            'future_root': '/g/data/oi10/replicas/CMIP6/ScenarioMIP/DKRZ/MPI-ESM1-2-HR/ssp126/r1i1p1f1',
            'sinfor': 'gn',
            'version_hist': 'v20190710',
            'version_future': 'v20190710',
        },
    },
    'UKESM1-0-LL': {
        'label': 'ukesm1_0_ll',
        'member_hist': 'r1i1p1f2',
        'member_future': 'r1i1p1f2',
        'three_d': {
            'hist_root': str(UKESM_STAGE_ROOT / 'historical' / 'r1i1p1f2'),
            'future_root': str(UKESM_STAGE_ROOT / 'ssp126' / 'r1i1p1f2'),
            'sinfor': 'gn',
            'version_hist': UKESM_STAGE_VERSION,
            'version_future': UKESM_STAGE_VERSION,
        },
        'surface': {
            'hist_root': '/g/data/oi10/replicas/CMIP6/CMIP/MOHC/UKESM1-0-LL/historical/r1i1p1f2',
            'future_root': '/g/data/oi10/replicas/CMIP6/ScenarioMIP/MOHC/UKESM1-0-LL/ssp126/r1i1p1f2',
            'sinfor': 'gn',
            'version_hist': 'v20190627',
            'version_future': 'v20190726',
        },
    },
}

# Truth GCMs whose 3D input requires a staging step (src/stage_ukesm_3d_to_gadi.sh)
# before interpolation can run, because the source archive isn't directly
# mounted in a CMIP6-DRS-compatible layout. Extend this set if a future truth
# GCM has the same requirement.
STAGED_3D_MODELS = {'UKESM1-0-LL'}

HIST_PERIOD = {'startyear_h': 1984, 'endyear_h': 2014}
FUTURE_PERIOD = {'startyear_h': 2080, 'endyear_h': 2100}
DOMAIN = {'lat_min': -90, 'lat_max': 90, 'lon_min': 0, 'lon_max': 360}
RESOURCES = {'ncpus': 48, 'mem_gb': 190}


def interp_output_path(label: str, scenario: str, table: str) -> Path:
    """Interpolated-truth-GCM output path, matching generate_model_as_truth_configs.py's convention."""
    period_dir = 'hist' if scenario == 'historical' else 'ssp126_2080_2100'
    return OUTPUT_ROOT / f"{label}_to_access" / period_dir / table


def check_3d_model_staged(model_name: str, scenario: str) -> None:
    """Raise if a truth GCM that requires staging hasn't been staged for this scenario yet.

    Call this before submitting interpolation jobs for a truth GCM in
    STAGED_3D_MODELS, so a missing staging step fails fast with a clear
    message instead of deep inside the interpolation pipeline.
    """
    if model_name not in STAGED_3D_MODELS:
        return
    meta = MODELS[model_name]['three_d']
    root = Path(meta['hist_root'] if scenario == 'historical' else meta['future_root'])
    if not root.is_dir() or not any(root.rglob('*.nc')):
        raise FileNotFoundError(
            f"{model_name} 3D input for scenario={scenario!r} not found under {root}. "
            f"Run stage_ukesm_3d_to_gadi.sh (PERIOD={scenario} for the historical/ssp126 "
            "staging convention) before submitting interpolation for this model."
        )
