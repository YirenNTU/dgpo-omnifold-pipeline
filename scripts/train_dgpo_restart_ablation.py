#!/usr/bin/env python3
"""Launch step260 DGPO branches, with preflight before allocating Ray workers."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

import torch
from train_dgpo_iteration_one import state_digest
from train_neutrino_backend import build_runtime_config, read_yaml

ROOT = Path(__file__).resolve().parents[1]


def config_path(arm):
    return ROOT / f'config/dgpo_omnifold_ztautau_10pct_step260_{arm}.yaml'


def check_pair(inherit, restart):
    """Permit only inheritance and output/provenance differences."""
    normalized = []
    for arm, config in [('inherit', inherit), ('restart', restart)]:
        c = copy.deepcopy(config)
        d = c['dgpo']
        a = d['adaptive_omnifold']
        r = a['recalibration']
        expected = [1] if arm == 'inherit' else []
        if r['warm_start_iterations'] != expected or r.get('warm_start_from_iteration_one'):
            raise ValueError(f'{arm}: incorrect inheritance protocol')
        if r['crossfit_partition'] != 'identity' or not a['single_pool_train_validation']:
            raise ValueError('Both arms require identical identity-based splits')
        if d['checkpoint_load_mode'] != 'weights_only' or d['auto_resume_from_last']:
            raise ValueError('Fresh live-policy initialization required')
        if not r['bootstrap_on_start'] or not a['baseline_probe_on_start']:
            raise ValueError('Both arms must fit an initial reward and monitor')
        if r['refit_once_on_resume'] or d['global_best_checkpoint_search_dirs']:
            raise ValueError('Do not import historical refit/global-best state')
        if not c['logger']['wandb']['fresh_run'] or c['logger']['wandb']['resume'] != 'never':
            raise ValueError('Each arm requires a new W&B run')
        r['warm_start_iterations'] = []
        del c['logger'], c['nersc']
        del c['options']['Training']['model_checkpoint_save_path']
        normalized.append(c)
    if normalized[0] != normalized[1]:
        raise ValueError('The two arms differ beyond reward inheritance and output metadata')
    if inherit['options']['Training']['model_checkpoint_save_path'] == restart['options']['Training']['model_checkpoint_save_path']:
        raise ValueError('Arms must use different checkpoint directories')


def verify_source(config):
    d = config['dgpo']
    a = d['adaptive_omnifold']
    pinned = d.get('pinned_classifier_restart', False)
    if d['checkpoint_load_mode'] != ('resume' if pinned else 'weights_only') or d['auto_resume_from_last']:
        raise ValueError('This launcher requires a fresh step260 policy branch')
    if d.get('best_source_start_new_experiment') or d.get('auto_resume_best_source_checkpoint_dir'):
        raise ValueError('Use the pinned step260 source, not automatic best selection')
    if a['recalibration']['bootstrap_on_start'] == pinned or not a['baseline_probe_on_start']:
        raise ValueError('Invalid initial stack/baseline settings for this restart mode')
    training = config['options']['Training']
    source = Path(training['model_checkpoint_load_path'])
    output = Path(training['model_checkpoint_save_path'])
    if output.resolve() == source.parent.resolve():
        raise ValueError('Output must not overwrite the source')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Fresh-run output is not empty: {output}')
    if any(training['EMA'].get(k, False) for k in ('replace_model_after_load', 'use_for_generation')):
        raise ValueError('This experiment requires live state_dict, not EMA')
    try:
        payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as exc:
        if 'mmap can only be used' not in str(exc):
            raise
        payload = torch.load(source, map_location='cpu', weights_only=False)
    if (payload.get('epoch'), payload.get('global_step'), payload.get('dgpo_next_epoch')) != (25, 260, 26):
        raise ValueError('Expected epoch25 / completed step260 / next_epoch26 source')
    digest = state_digest(payload['state_dict'])
    if pinned:
        sys.path.insert(0, str(ROOT/'evenet_dgpo'))
        from RL.DGPO_neutrino.dgpo_trainer import _prepare_pinned_classifier_restart
        _prepare_pinned_classifier_restart(payload, d)
        print('Verified saved OmniFold iteration-1 fold cache and raw monitor; retaining installed stack.', flush=True)
    for value in (config['reward_config']['omnifold']['backbone_checkpoint'],
                  config['platform']['data_parquet_dir'], config['platform']['data_parquet_val_dir']):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    print(f'Verified source: {source}\nLive policy SHA256: {digest}\n'
          f'Fresh DGPO step=0 epoch=0; classifier inheritance={pinned}; new optimizer.\nOutput: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--arm', choices=['inherit', 'restart'])
    selection.add_argument('--config', type=Path, help='Standalone fresh step260 overlay (no paired-arm comparison).')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.config is not None:
        selected_config = args.config.resolve()
        config = read_yaml(selected_config)
    else:
        configs = {arm: read_yaml(config_path(arm)) for arm in ('inherit', 'restart')}
        check_pair(configs['inherit'], configs['restart'])
        selected_config = config_path(args.arm)
        config = configs[args.arm]
    verify_source(config)
    runtime = build_runtime_config(base_config=ROOT/'config/train_diffusion_nersc.yaml',
                                   overlay_config=selected_config, backend='dgpo-evenet')
    for section in read_yaml(runtime).values():
        if isinstance(section, dict) and isinstance(section.get('default'), str):
            if not Path(section['default']).is_file():
                raise FileNotFoundError(f"Missing config dependency: {section['default']}")
    print(f'Preflight passed: {selected_config.name}', flush=True)
    if args.check_only:
        return
    subprocess.run([sys.executable, str(ROOT/'scripts/train_neutrino_backend.py'),
                    '--backend', 'dgpo-evenet', '--base-config', str(ROOT/'config/train_diffusion_nersc.yaml'),
                    '--overlay-config', str(selected_config), '--', '--ray-dir',
                    config['nersc']['ray']['results_dir']], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
