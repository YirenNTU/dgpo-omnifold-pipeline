"""Launch the sample, preflight, or fit phase from one 10% DGPO-matched YAML.

The user starts the 16-GPU Ray allocation and invokes each phase personally.
This launcher never submits a compute job or changes the DGPO policy.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError('Experiment YAML must be a mapping')
    for section in ('reference', 'platform', 'experiment', 'sampling', 'classifier', 'logger'):
        if not isinstance(data.get(section), dict):
            raise ValueError(f'Missing YAML section: {section}')
    reference_path = ROOT / data['reference']['dgpo_10pct_config']
    with reference_path.open() as stream:
        reference = yaml.safe_load(stream)
    for key in ('data_parquet_dir', 'data_parquet_val_dir'):
        configured = Path(data['platform'][key])
        dgpo = Path(reference['platform'][key])
        if configured != dgpo:
            raise ValueError(f'{key} differs from 10% DGPO: {configured} != {dgpo}')
    if data['experiment']['representation'] != 'tau':
        raise ValueError('This launcher requires the tau candidate representation')
    wandb = data['logger']['wandb']
    for key in ('sampling_run_name','classifier_run_name'):
        name = wandb[key]
        if not isinstance(name, str) or len(name) > 96 or len(name.split(' | ')) < 2:
            raise ValueError(f'Invalid W&B display name: {key}')
    return data


def command(config, phase, ray_address=None):
    if phase not in ('sample','prepare','train'):
        raise ValueError(f'Unknown phase: {phase}')
    experiment, platform = config['experiment'], config['platform']
    wandb = config['logger']['wandb']
    if phase == 'sample':
        sample = config['sampling']
        args = [sys.executable, str(ROOT / 'scripts/sample_conditional_tau_train.py'),
            '--test-source', experiment['test_source'],
            '--events', platform['data_parquet_dir'],
            '--output', experiment['train_sample_root'],
            '--workers', str(sample['workers']), '--batch-size', str(sample['batch_size']),
            '--seed', str(sample['seed']),
            '--run-name', wandb['sampling_run_name'],
            '--wandb-group', wandb['group']]
    else:
        training = config['classifier']
        args = [sys.executable, str(ROOT / 'scripts/train_conditional_tau_ratio.py'),
            '--representation', experiment['representation'],
            '--condition-normalization', training.get('condition_normalization', 'legacy_slot'),
            '--train-source', experiment['train_sample_root'],
            '--test-source', experiment['test_source'],
            '--train-events', platform['data_parquet_dir'],
            '--test-events', platform['data_parquet_val_dir'],
            '--output', experiment['output'],
            '--workers', str(training['workers']),
            '--batch-size', str(training['batch_size']),
            '--epochs', str(training['epochs']),
            '--patience', str(training['patience']),
            '--min-delta', str(training['min_delta']),
            '--min-steps', str(training['min_steps']),
            '--lr', str(training['lr']),
            '--min-lr', str(training['min_lr']),
            '--weight-decay', str(training['weight_decay']),
            '--dropout', str(training['dropout']),
            '--hidden', str(training['hidden']),
            '--bootstrap', str(training['bootstrap']),
            '--seed', str(training['seed']),
            '--run-name', wandb['classifier_run_name'],
            '--wandb-group', wandb['group']]
        if phase == 'prepare':
            args.append('--prepare-only')
        if training.get('backbone_cache'):
            args.extend(['--backbone-cache', training['backbone_cache'],
                         '--mmd-coefficient', str(training.get('mmd_coefficient', 0.))])
        if training.get('relative_angle_input', False):
            args.append('--relative-angle-input')
        if experiment.get('baseline_directory'):
            args.extend(['--baseline-directory', experiment['baseline_directory']])
        if experiment.get('baseline_run'):
            args.extend(['--baseline-run', experiment['baseline_run']])
        if experiment.get('allow_baseline_batch_size_mismatch', False):
            args.append('--allow-baseline-batch-size-mismatch')
    if ray_address and phase != 'prepare':
        args.extend(['--ray-address', ray_address])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('phase', choices=('sample','prepare','train'))
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.phase == 'sample':
        root = Path(cfg['experiment']['train_sample_root'])
        completed = [p for p in root.glob('sample-*') if (p/'COMPLETE').is_file()]
        if (root/'COMPLETE').is_file() and (root/'candidates.npz').is_file():
            completed.append(root)
        if completed:
            raise ValueError(f'{len(completed)} completed sample set(s) already exist under {root}; '
                'reuse one for prepare/train instead of resampling')
    cmd = command(cfg, args.phase, os.environ.get('RAY_ADDRESS'))
    print('PHASE:', args.phase, flush=True)
    print('TRAIN DATA:', cfg['platform']['data_parquet_dir'], flush=True)
    print('VALIDATION DATA:', cfg['platform']['data_parquet_val_dir'], flush=True)
    print('COMMAND:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
