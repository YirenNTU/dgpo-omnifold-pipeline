#!/usr/bin/env python3
"""Capture early fixed representations using the unchanged 16-GPU H4 fit."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_h4_classifier_representation import validated_config as parent_config, validate_manifest, CLEAN
from train_neutrino_backend import read_overlay_yaml

CONFIG = ROOT / 'config/dgpo_h4_frozen_readout_capture.yaml'
OUTPUT = '/pscratch/sd/y/yiren/Ztautau/h4_frozen_readout_capture'
QUICK_OUTPUT = '/pscratch/sd/y/yiren/Ztautau/h4_frozen_readout_quick300'
QUICK_CONFIG = ROOT / 'config/dgpo_h4_frozen_readout_quick.yaml'


def validated_config(quick=False):
    parent = parent_config(True)
    cfg = read_overlay_yaml(QUICK_CONFIG if quick else CONFIG)
    output = QUICK_OUTPUT if quick else OUTPUT
    compared = deepcopy(cfg)
    fit = compared['dgpo']['adaptive_omnifold']['audit_fit']
    if fit.pop('representation_export_dir') != output + '/panels':
        raise ValueError('Requires isolated panel export directory')
    if fit['diagnostic_snapshot_dir'] != output + '/diagnostic_snapshots':
        raise ValueError('Requires isolated diagnostics')
    if cfg['options']['Training']['model_checkpoint_save_path'] != output + '/checkpoints':
        raise ValueError('Requires isolated checkpoint directory')
    if cfg['nersc']['ray']['results_dir'] != output + '/ray_results':
        raise ValueError('Requires isolated Ray directory')
    if quick:
        for key, value in dict(steps=300, min_steps=300, require_saturation=False).items():
            if fit[key] != value:
                raise ValueError('Quick capture must use exactly 300 updates')
            fit[key] = parent['dgpo']['adaptive_omnifold']['audit_fit'][key]
    fit['diagnostic_snapshot_dir'] = parent['dgpo']['adaptive_omnifold']['audit_fit']['diagnostic_snapshot_dir']
    compared['options']['Training']['model_checkpoint_save_path'] = parent['options']['Training']['model_checkpoint_save_path']
    for key in ('dgpo', 'platform', 'options', 'reward_config'):
        if compared[key] != parent[key]:
            raise ValueError(f'Capture must preserve path experiment: {key}')
    for key in ('classifier_only', 'classifier_fit_count', 'reward_fit_count', 'policy_updates_per_round', 'source_policy_step', 'source_wandb_run'):
        if cfg['experiment'][key] != parent['experiment'][key]:
            raise ValueError(f'Changed experiment contract: {key}')
    if cfg['logger']['wandb']['id'] != ('h4frzq1' if quick else 'h4frzcap1') or cfg['logger']['wandb']['resume'] != 'never':
        raise ValueError('Requires fresh capture W&B run')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--quick', action='store_true', help='300-update plateau screen, separate run/output')
    args = parser.parse_args()
    cfg = validated_config(args.quick)
    print('Contract OK: step-1110 H4, 16 GPUs, exports at 0/50/100; zero policy updates; ' +
          ('300-update early-plateau screen only.' if args.quick else 'original full-fit stopping.'))
    if args.validate_only:
        return 0
    if not Path(cfg['options']['Training']['model_checkpoint_load_path']).is_file():
        parser.error('Missing pinned step-1110 checkpoint')
    validate_manifest(Path(CLEAN).parent / 'filter_manifest.json')
    if not any(Path(CLEAN).glob('*.parquet')):
        parser.error('Missing clean validation Parquet')
    panels = Path(QUICK_OUTPUT if args.quick else OUTPUT) / 'panels'
    if panels.exists() and any(panels.iterdir()):
        parser.error('Panel directory already populated; do not mix independent captures')
    code = subprocess.run([sys.executable, str(ROOT / 'scripts/train_neutrino_backend.py'),
        '--backend', 'dgpo-evenet', '--base-config', str(ROOT / 'config/train_diffusion_nersc.yaml'),
        '--overlay-config', str(QUICK_CONFIG if args.quick else CONFIG), '--', '--ray-dir', cfg['nersc']['ray']['results_dir']],
        cwd=ROOT, check=False).returncode
    if code == 0 and not all((panels / f'panel_step{s:04d}.pt').is_file() for s in (0, 50, 100)):
        raise RuntimeError('Capture ended without all three panels; do not interpret replay')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
