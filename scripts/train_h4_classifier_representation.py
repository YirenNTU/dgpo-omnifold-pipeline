#!/usr/bin/env python3
"""One cold H4 classifier, clean validation, detached representation measurements."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_neutrino_backend import read_overlay_yaml
from train_h4_classifier_lr_stability import validated_config as validate_parent, DIAGNOSTIC_CONFIG

CONFIG = ROOT / 'config/dgpo_h4_classifier_representation.yaml'
CLEAN = '/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val'
OUTPUT = '/pscratch/sd/y/yiren/Ztautau/h4_classifier_representation'
SOURCE = '/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt'
SOURCE_RUN = 'ytchou97-university-of-washington/nu2flow-RL/c4a91e07'


def validated_config(path_experiment=False):
    parent = validate_parent(DIAGNOSTIC_CONFIG)
    path = ROOT / 'config/dgpo_h4_classifier_path.yaml' if path_experiment else CONFIG
    output = OUTPUT + '_path' if path_experiment else OUTPUT
    config = read_overlay_yaml(path)
    expected = dict(parent['dgpo']['adaptive_omnifold']['audit_fit'])
    expected.update(lr_scheduler='constant', representation_diagnostic_enabled=True,
                    representation_probe_rows=128, representation_probe_interval_steps=100,
                    diagnostic_snapshot_dir=output + '/diagnostic_snapshots')
    if path_experiment:
        expected['representation_path_enabled'] = True
    dgpo = dict(config['dgpo'])
    adaptive = dict(dgpo['adaptive_omnifold'])
    if adaptive['audit_fit'] != expected:
        raise ValueError('Only measurement settings may change the classifier fit')
    adaptive['audit_fit'] = parent['dgpo']['adaptive_omnifold']['audit_fit']
    dgpo['adaptive_omnifold'] = adaptive
    if dgpo != parent['dgpo']:
        raise ValueError('Policy/reward/fit settings must match the parent')
    expected_platform = {**parent['platform'], 'data_parquet_val_dir': CLEAN}
    if config['platform'] != expected_platform:
        raise ValueError('Requires original train data, clean validation and 16 GPUs')
    for key in ('classifier_only', 'classifier_fit_count', 'reward_fit_count', 'rounds',
                'policy_updates_per_round',
                'classifier_design_arm'):
        if config['experiment'][key] != parent['experiment'][key]:
            raise ValueError(f'Experiment contract changed: {key}')
    provenance = config['nersc']['reproducibility']
    if (config['experiment']['source_policy_step'] != 1110
            or config['experiment']['source_wandb_run'] != SOURCE_RUN
            or provenance['source_dgpo_global_step'] != 1110
            or provenance['source_wandb_run'] != SOURCE_RUN
            or provenance['source_checkpoint'] != SOURCE):
        raise ValueError('Requires pinned old-DGPO step-1110 source provenance')
    train = dict(config['options']['Training'])
    if train.pop('model_checkpoint_save_path') != output + '/checkpoints':
        raise ValueError('Requires independent checkpoint directory')
    old_train = dict(parent['options']['Training'])
    old_train.pop('model_checkpoint_save_path')
    old_train['model_checkpoint_load_path'] = SOURCE
    if train != old_train or config['reward_config'] != parent['reward_config']:
        raise ValueError('Requires step-1110 source; classifier backbone and normalizer must be unchanged')
    if config['logger']['wandb']['id'] != ('h4clfpath1' if path_experiment else 'h4clfrep1'):
        raise ValueError('Requires independent W&B run')
    if path_experiment and config['logger']['wandb'].get('resume') != 'never':
        raise ValueError('Path experiment must not merge restarted fits into an existing run')
    return config


def validate_manifest(path):
    manifest = json.loads(path.read_text())
    if (manifest.get('complete') is not True or manifest.get('rows_in') != 119004
            or manifest.get('rows_out') != 119002 or manifest.get('events_removed') != 2
            or manifest.get('output') != CLEAN
            or {str(r['source_event_key']) for r in manifest.get('removed_events', [])}
            != {'9628165', '3333501'}):
        raise ValueError('Clean validation manifest does not match the verified two-event exclusion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--ray-dir', type=Path)
    parser.add_argument('--path-readouts', action='store_true', help='Raw/normalized Fourier and fit-only ridge CV experiment')
    args = parser.parse_args()
    config = validated_config(args.path_readouts)
    overlay = ROOT / 'config/dgpo_h4_classifier_path.yaml' if args.path_readouts else CONFIG
    print('Contract OK: c4a91e07 step-1110 policy; one cold H4; constant LR; 16 GPUs; measurement-only readouts.')
    if args.validate_only:
        print('Configuration only; NERSC files are checked at launch.')
        return 0
    checkpoint = Path(config['options']['Training']['model_checkpoint_load_path'])
    if not checkpoint.is_file() or not any(Path(CLEAN).glob('*.parquet')):
        parser.error('Missing pinned policy checkpoint or clean validation Parquet')
    validate_manifest(Path(CLEAN).parent / 'filter_manifest.json')
    return subprocess.run([
        sys.executable, str(ROOT / 'scripts/train_neutrino_backend.py'),
        '--backend', 'dgpo-evenet', '--base-config', str(ROOT / 'config/train_diffusion_nersc.yaml'),
        '--overlay-config', str(overlay), '--', '--ray-dir',
        str(args.ray_dir or config['nersc']['ray']['results_dir']),
    ], cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
