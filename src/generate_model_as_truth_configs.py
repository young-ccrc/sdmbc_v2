#!/usr/bin/env python3
"""Generate model-as-truth interpolation configs targeting ACCESS-ESM1-5."""

from pathlib import Path

import yaml

from model_as_truth_registry import (
    DOMAIN,
    FUTURE_PERIOD,
    HIST_PERIOD,
    MODELS,
    OUTPUT_ROOT,
    ROOT,
    RESOURCES,
    TARGET,
)


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
