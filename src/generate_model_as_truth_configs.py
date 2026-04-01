#!/usr/bin/env python3
"""Generate model-as-truth interpolation configs targeting ACCESS-ESM1-5."""

from pathlib import Path
import yaml

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

HIST_PERIOD = {'startyear_h': 1984, 'endyear_h': 2014}
FUTURE_PERIOD = {'startyear_h': 2080, 'endyear_h': 2100}
DOMAIN = {'lat_min': -90, 'lat_max': 90, 'lon_min': 0, 'lon_max': 360}
RESOURCES = {'ncpus': 48, 'mem_gb': 190}


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, default_flow_style=False))


def build_3d_config(model_name: str, meta: dict, scenario: str) -> dict:
    phase = 'historical' if scenario == 'historical' else 'ssp126'
    target = dict(TARGET[phase])
    future_block = target.pop('future', None)
    source_meta = meta['three_d']
    data = {
        'schema_version': 2,
        'workflow': 'interp_3d',
        'experiment': f"{meta['label']}_to_access_esm1_5_{'hist' if scenario == 'historical' else 'ssp126_2080_2100'}",
        'paths': {
            'input_path': source_meta['hist_root'] if scenario == 'historical' else source_meta['future_root'],
            'output_path': str(OUTPUT_ROOT / f"{meta['label']}_to_access" / ('hist' if scenario == 'historical' else 'ssp126_2080_2100') / '3d'),
        },
        'source': {
            'input_model': 'gcm',
            'input_infor': '6hrLev',
            'input_gname': model_name,
            'input_period': scenario,
            'input_cinfor': meta['member_hist'] if scenario == 'historical' else meta['member_future'],
            'input_sinfor': source_meta['sinfor'],
            'input_version': source_meta['version_hist'] if scenario == 'historical' else source_meta['version_future'],
            'input_z_path': None,
        },
        'target': target,
        'domain': DOMAIN,
        'period': HIST_PERIOD if scenario == 'historical' else FUTURE_PERIOD,
        'resources': RESOURCES,
    }
    if future_block:
        data['future'] = future_block
    return data


def build_surface_config(model_name: str, meta: dict, scenario: str) -> dict:
    phase = 'historical' if scenario == 'historical' else 'ssp126'
    target = dict(TARGET[phase])
    future_block = target.pop('future', None)
    source_meta = meta['surface']
    data = {
        'schema_version': 2,
        'workflow': 'interp_surface',
        'experiment': f"{meta['label']}_to_access_esm1_5_surface_{'hist' if scenario == 'historical' else 'ssp126_2080_2100'}",
        'paths': {
            'input_path_sst': source_meta['hist_root'] if scenario == 'historical' else source_meta['future_root'],
            'output_path': str(OUTPUT_ROOT / f"{meta['label']}_to_access" / ('hist' if scenario == 'historical' else 'ssp126_2080_2100') / 'surface'),
        },
        'source': {
            'input_model': 'gcm',
            'input_gname': model_name,
            'input_infor': 'Oday',
            'input_period': scenario,
            'input_cinfor': meta['member_hist'] if scenario == 'historical' else meta['member_future'],
            'input_sinfor': source_meta['sinfor'],
            'input_version': source_meta['version_hist'] if scenario == 'historical' else source_meta['version_future'],
        },
        'target': target,
        'domain': DOMAIN,
        'period': HIST_PERIOD if scenario == 'historical' else FUTURE_PERIOD,
        'resources': RESOURCES,
    }
    if future_block:
        data['future'] = future_block
    return data


def main() -> None:
    for model_name, meta in MODELS.items():
        label = meta['label']
        write_yaml(ROOT / f'config_interp_3d_{label}_to_access_hist.yaml', build_3d_config(model_name, meta, 'historical'))
        write_yaml(ROOT / f'config_interp_3d_{label}_to_access_ssp126_2080_2100.yaml', build_3d_config(model_name, meta, 'ssp126'))
        write_yaml(ROOT / f'config_interp_surface_{label}_to_access_hist.yaml', build_surface_config(model_name, meta, 'historical'))
        write_yaml(ROOT / f'config_interp_surface_{label}_to_access_ssp126_2080_2100.yaml', build_surface_config(model_name, meta, 'ssp126'))


if __name__ == '__main__':
    main()
