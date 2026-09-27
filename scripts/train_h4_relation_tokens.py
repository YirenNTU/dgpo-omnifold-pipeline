#!/usr/bin/env python3
"""Cold EveNet relation-token classifier pilot from the pinned step-1110 policy."""
import argparse
from copy import deepcopy
from pathlib import Path
import re
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_neutrino_backend import read_overlay_yaml
from train_h4_pair_token import validated_config as parent_config
from train_h4_classifier_representation import CLEAN, SOURCE, validate_manifest

OUTPUT = '/pscratch/sd/y/yiren/Ztautau/h4_relation_tokens'
CONFIG = ROOT / 'config/dgpo_h4_relation_tokens.yaml'
PRETRAIN = '/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt'


def validated_config():
    cfg = read_overlay_yaml(CONFIG)
    parent = parent_config()
    expected = deepcopy(parent['dgpo'])
    a = expected['adaptive_omnifold']
    for block in (a['audit_fit'], a['recalibration']):
        block.update(topology_pair_token=False, topology_fourier_embedding=False,
            periodic_pair_features=False, relation_token_count=4, decoder_layers=2,
            topology_max_harmonic=1)
    a['audit_fit'].update(fourier_output_standardization=False,
        representation_path_enabled=False, representation_export_dir=None,
        diagnostic_snapshot_dir=OUTPUT + '/diagnostic_snapshots',
        ratio_audit_export_dir=OUTPUT + '/ratio_audit')
    if cfg['dgpo'] != expected:
        raise ValueError('Only relation-token architecture and diagnostic destinations may differ from h4pair01')
    for key in ('platform', 'reward_config'):
        if cfg[key] != parent[key]:
            raise ValueError(f'Changed fixed data/normalizer contract: {key}')
    options = deepcopy(parent['options'])
    options['Training']['model_checkpoint_save_path'] = OUTPUT + '/checkpoints'
    if cfg['options'] != options:
        raise ValueError('Requires pinned step-1110 checkpoint and original training options')
    for key in ('classifier_only', 'classifier_fit_count', 'reward_fit_count', 'rounds',
                'policy_updates_per_round', 'source_policy_step', 'source_wandb_run',
                'classifier_design_arm'):
        if cfg['experiment'][key] != parent['experiment'][key]:
            raise ValueError(f'Changed classifier-only contract: {key}')
    if (cfg['experiment']['protocol'] != 'h4-relation-tokens-v1'
            or cfg['nersc']['reproducibility']['source_checkpoint'] != SOURCE
            or cfg['platform']['data_parquet_val_dir'] != CLEAN
            or cfg['nersc']['ray']['results_dir'] != OUTPUT + '/ray_results'
            or cfg['logger']['wandb']['id'] != 'h4rel01'
            or cfg['logger']['wandb']['resume'] != 'never'):
        raise ValueError('Invalid relation-token provenance or output isolation')
    if cfg['reward_config']['omnifold']['backbone_checkpoint'] != PRETRAIN:
        raise ValueError('Classifier must initialize from 10pct pretrained EveNet, not step1110')
    if cfg['platform']['number_of_workers'] != 16:
        raise ValueError('Requires 16 GPU workers')
    return cfg


def with_suffix(cfg, suffix):
    if not re.fullmatch(r'[a-z0-9-]{1,24}', suffix):
        raise ValueError('run-suffix requires 1-24 lowercase letters, digits or hyphens')
    cfg = deepcopy(cfg)
    output = OUTPUT + '-' + suffix
    cfg['logger']['wandb']['id'] += '-' + suffix
    for key in ('name', 'version'):
        cfg['logger']['local'][key] += '-' + suffix
    cfg['logger']['local']['save_dir'] = output + '/logs'
    cfg['nersc']['ray']['results_dir'] = output + '/ray_results'
    cfg['options']['Training']['model_checkpoint_save_path'] = output + '/checkpoints'
    fit = cfg['dgpo']['adaptive_omnifold']['audit_fit']
    fit['diagnostic_snapshot_dir'] = output + '/diagnostic_snapshots'
    fit['ratio_audit_export_dir'] = output + '/ratio_audit'
    return cfg


def preflight(cfg):
    if not Path(SOURCE).is_file() or not any(Path(CLEAN).glob('*.parquet')):
        raise ValueError('Missing pinned step-1110 checkpoint or clean validation Parquet')
    if not Path(PRETRAIN).is_file():
        raise ValueError('Missing classifier pretrained checkpoint')
    validate_manifest(Path(CLEAN).parent / 'filter_manifest.json')
    output = Path(cfg['options']['Training']['model_checkpoint_save_path']).parent
    if output.exists():
        raise ValueError('Output already exists; use a fresh --run-suffix')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--run-suffix')
    args = parser.parse_args()
    cfg = validated_config()
    if args.run_suffix:
        cfg = with_suffix(cfg, args.run_suffix)
    print('Contract OK: EveNet four relation tokens / two rounds; step1110; clean validation truth; 16 GPUs; 1000/3000 fit updates; zero policy updates.')
    if args.validate_only:
        return 0
    output = preflight(cfg)
    if args.check_only:
        print('Source/data preflight OK; no training started.')
        return 0
    output.mkdir(parents=True, exist_ok=False)
    resolved = output / 'resolved_ratio_experiment.yaml'
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return subprocess.run([sys.executable, str(ROOT / 'scripts/train_neutrino_backend.py'),
        '--backend', 'dgpo-evenet', '--base-config', str(ROOT / 'config/train_diffusion_nersc.yaml'),
        '--overlay-config', str(resolved), '--', '--ray-dir', cfg['nersc']['ray']['results_dir']],
        cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
