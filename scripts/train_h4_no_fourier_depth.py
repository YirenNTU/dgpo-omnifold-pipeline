"""Cold no-Fourier attention-depth pilot; no policy updates."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys

from train_h4_classifier_representation import ROOT, CLEAN, validate_manifest, validated_config as parent_config
from train_neutrino_backend import read_overlay_yaml


def validated_config(layers):
    if layers not in (1, 2):
        raise ValueError('Requires one or two decoder layers')
    path = ROOT / f'config/dgpo_h4_no_fourier_depth{layers}.yaml'
    cfg = read_overlay_yaml(path)
    parent = parent_config()
    expected = copy.deepcopy(parent['dgpo'])
    a = expected['adaptive_omnifold']
    output = f'/pscratch/sd/y/yiren/Ztautau/h4_no_fourier_depth{layers}'
    for block in (a['audit_fit'], a['recalibration']):
        block.update(periodic_pair_features=False, topology_fourier_embedding=False,
                     topology_direct_logit=False, topology_conditioning=False,
                     visible_pair_rest_frame=False, decoder_layers=layers)
    a['audit_fit']['representation_path_enabled'] = False
    a['audit_fit']['diagnostic_snapshot_dir'] = output + '/diagnostic_snapshots'
    if cfg['dgpo'] != expected:
        raise ValueError('Only no-Fourier flags, depth and diagnostic location may change')
    for key in ('platform', 'reward_config'):
        if cfg[key] != parent[key]:
            raise ValueError(f'Changed fixed contract: {key}')
    options = copy.deepcopy(parent['options'])
    options['Training']['model_checkpoint_save_path'] = output + '/checkpoints'
    if cfg['options'] != options:
        raise ValueError('Source checkpoint, normalization and training options must match')
    for key in ('classifier_only', 'classifier_fit_count', 'reward_fit_count', 'rounds',
                'policy_updates_per_round', 'source_policy_step', 'source_wandb_run', 'classifier_design_arm'):
        if cfg['experiment'][key] != parent['experiment'][key]:
            raise ValueError(f'Changed experiment contract: {key}')
    if (cfg['logger']['wandb']['id'] != f'h4nfd{layers}'
            or cfg['logger']['wandb']['resume'] != 'never'):
        raise ValueError('Requires independent non-resuming W&B run')
    return path, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, choices=(1, 2), required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    path, cfg = validated_config(args.layers)
    print(f'Contract OK: step-1110, no Fourier/pair features, {args.layers} decoder layers, 16 GPUs, constant LR.')
    if args.validate_only:
        print('Configuration only; NERSC files are checked at launch.')
        return 0
    if not Path(cfg['options']['Training']['model_checkpoint_load_path']).is_file():
        parser.error('Missing step-1110 checkpoint')
    if not any(Path(CLEAN).glob('*.parquet')):
        parser.error('Missing clean validation Parquet')
    validate_manifest(Path(CLEAN).parent / 'filter_manifest.json')
    return subprocess.run([sys.executable, str(ROOT / 'scripts/train_neutrino_backend.py'),
        '--backend', 'dgpo-evenet', '--base-config', str(ROOT / 'config/train_diffusion_nersc.yaml'),
        '--overlay-config', str(path), '--', '--ray-dir', cfg['nersc']['ray']['results_dir']],
        cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
