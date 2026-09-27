#!/usr/bin/env python3
"""Matched complete-classifier Fourier output scaling pilot on 16 GPUs."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import re
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_h4_frozen_readout import validated_config as parent_config
from train_h4_classifier_representation import CLEAN, validate_manifest
from train_neutrino_backend import read_overlay_yaml


def validated_config(arm):
    if arm == 'ratio-health':
        cfg = retry_config(validated_config('standardized-long'), 'standardized-long', 'ratio-health')
        output = str(Path(cfg['options']['Training']['model_checkpoint_save_path']).parent)
        cfg['dgpo']['adaptive_omnifold']['audit_fit']['ratio_audit_export_dir'] = output + '/ratio_audit'
        cfg['dgpo']['adaptive_omnifold']['audit_fit']['validation_min_delta'] = 1e-3
        cfg['experiment']['diagnostic_parent_run'] = 'h4scllong1'
        cfg['experiment']['intervention'] = 'best_classifier_ratio_audit_export_with_bce_patience_delta_0.001'
        cfg['experiment']['primary_decision_rule'] = (
            'After best-BCE restore, report raw-ratio weighted minus unweighted '
            'held-out joint cos(delta_phi)/opening-cosine JSD with paired event '
            'bootstrap interval, alongside ESS and weight concentration. '
            'Fixed r^0.75 is sensitivity only; never choose temperature on test. '
            'No DGPO update or complete distribution-closure claim.'
        )
        if not cfg['dgpo']['adaptive_omnifold']['audit_fit'].get('disjoint_final_audit'):
            raise ValueError('Ratio health requires an independent final audit split')
        cfg['logger']['wandb']['id'] = 'h4ratio1'
        cfg['logger']['wandb']['run_name'] = 'Are the learned ratios usable? | H4 held-out reweighting | best BCE'
        cfg['nersc']['execution']['command'] = 'shifter python3 scripts/train_h4_output_scaling.py --arm ratio-health'
        return cfg
    if arm == 'standardized-long':
        parent = validated_config('standardized')
        cfg = read_overlay_yaml(ROOT / 'config/dgpo_h4_scale_standardized_long.yaml')
        compared = deepcopy(cfg)
        fit = compared['dgpo']['adaptive_omnifold']['audit_fit']
        expected = dict(steps=3000, min_steps=1000, validation_patience_epochs=10,
                        require_saturation=False)
        for key, value in expected.items():
            if fit[key] != value:
                raise ValueError(f'Wrong long-fit setting: {key}')
            fit[key] = parent['dgpo']['adaptive_omnifold']['audit_fit'][key]
        output = '/pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long'
        for key, suffix in [('representation_export_dir', 'panels'), ('diagnostic_snapshot_dir', 'diagnostic_snapshots')]:
            if fit[key] != output + '/' + suffix:
                raise ValueError('Requires isolated long-fit output')
            fit[key] = parent['dgpo']['adaptive_omnifold']['audit_fit'][key]
        for key in ('nersc', 'logger', 'experiment'):
            compared[key] = parent[key]
        if cfg['options']['Training']['model_checkpoint_save_path'] != output + '/checkpoints':
            raise ValueError('Requires isolated long-fit checkpoints')
        compared['options']['Training']['model_checkpoint_save_path'] = parent['options']['Training']['model_checkpoint_save_path']
        if compared != parent:
            raise ValueError('Long fit may change budget only; preserve standardized model and seeds')
        if (cfg['experiment']['protocol'] != 'h4-output-scale-long-v1'
                or cfg['logger']['wandb']['id'] != 'h4scllong1'
                or cfg['logger']['wandb']['resume'] != 'never'
                or cfg['nersc']['ray']['results_dir'] != output + '/ray_results'):
            raise ValueError('Wrong long-fit protocol or output isolation')
        for key in ('classifier_only', 'classifier_fit_count', 'reward_fit_count', 'policy_updates_per_round', 'source_policy_step', 'source_wandb_run', 'classifier_design_arm'):
            if cfg['experiment'][key] != parent['experiment'][key]:
                raise ValueError('Changed long-fit experiment contract')
        return cfg
    if arm not in ('control', 'standardized'):
        raise ValueError('Unknown arm')
    parent = parent_config(True)
    cfg = read_overlay_yaml(ROOT / f'config/dgpo_h4_scale_{arm}.yaml')
    compared = deepcopy(cfg)
    fit = compared['dgpo']['adaptive_omnifold']['audit_fit']
    if fit.pop('fourier_output_standardization') is not (arm == 'standardized'):
        raise ValueError('Wrong scale intervention')
    output = f'/pscratch/sd/y/yiren/Ztautau/h4_scale_{arm}300'
    for key, suffix in [('representation_export_dir','panels'), ('diagnostic_snapshot_dir','diagnostic_snapshots')]:
        if fit[key] != output + '/' + suffix:
            raise ValueError('Requires isolated output')
        fit[key] = parent['dgpo']['adaptive_omnifold']['audit_fit'][key]
    if cfg['options']['Training']['model_checkpoint_save_path'] != output + '/checkpoints':
        raise ValueError('Requires isolated checkpoints')
    compared['options']['Training']['model_checkpoint_save_path'] = parent['options']['Training']['model_checkpoint_save_path']
    for key in ('dgpo','platform','options','reward_config'):
        if compared[key] != parent[key]:
            raise ValueError(f'Only output standardization may change: {key}')
    for key in ('classifier_only','classifier_fit_count','reward_fit_count','policy_updates_per_round','source_policy_step','source_wandb_run'):
        if cfg['experiment'][key] != parent['experiment'][key]:
            raise ValueError('Changed experiment contract')
    if cfg['logger']['wandb']['id'] != ('h4sclc1' if arm == 'control' else 'h4scls1') or cfg['logger']['wandb']['resume'] != 'never':
        raise ValueError('Requires isolated fresh W&B arm')
    if cfg['nersc']['ray']['results_dir'] != output + '/ray_results':
        raise ValueError('Requires isolated Ray directory')
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['control','standardized','standardized-long','ratio-health'], required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--run-suffix', help='Fresh attempt suffix; isolates W&B ID and every output directory')
    args = parser.parse_args()
    cfg = validated_config(args.arm)
    if args.run_suffix:
        if not re.fullmatch(r'[a-z0-9-]{1,24}', args.run_suffix):
            parser.error('run-suffix must contain 1-24 lowercase letters, digits or hyphens')
        cfg = retry_config(cfg, args.arm, args.run_suffix)
    budget = '1000 minimum / 3000 maximum updates, patience 10 epochs' if args.arm in ('standardized-long','ratio-health') else '300 updates'
    print(f'Contract OK: {args.arm}; fresh classifier, step-1110 policy, 16 GPUs, {budget}, BCE primary, zero policy updates.')
    if args.validate_only:
        return 0
    validate_manifest(Path(CLEAN).parent / 'filter_manifest.json')
    if not Path(cfg['options']['Training']['model_checkpoint_load_path']).is_file() or not any(Path(CLEAN).glob('*.parquet')):
        parser.error('Missing pinned source checkpoint or clean data')
    panels = Path(cfg['dgpo']['adaptive_omnifold']['audit_fit']['representation_export_dir'])
    if panels.exists() and any(panels.iterdir()):
        parser.error('Existing arm output; do not merge/restart fits')
    if args.arm == 'ratio-health':
        provenance = Path(cfg['options']['Training']['model_checkpoint_save_path']).parent / 'resolved_ratio_experiment.yaml'
        provenance.parent.mkdir(parents=True, exist_ok=True)
        with provenance.open('x') as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
    with tempfile.TemporaryDirectory(prefix='h4-scale-attempt-') as temporary:
        overlay = Path(temporary) / 'overlay.yaml'
        overlay.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return subprocess.run([sys.executable, str(ROOT / 'scripts/train_neutrino_backend.py'),
            '--backend','dgpo-evenet','--base-config',str(ROOT / 'config/train_diffusion_nersc.yaml'),
            '--overlay-config',str(overlay),
            '--','--ray-dir',cfg['nersc']['ray']['results_dir']], cwd=ROOT, check=False).returncode


def retry_config(config, arm, suffix):
    cfg = deepcopy(config)
    old = str(Path(cfg['options']['Training']['model_checkpoint_save_path']).parent)
    new = old + '-' + suffix
    cfg['logger']['wandb']['id'] += '-' + suffix
    cfg['logger']['local']['name'] += '-' + suffix
    cfg['logger']['local']['version'] += '-' + suffix
    cfg['logger']['local']['save_dir'] = new + '/logs'
    cfg['nersc']['ray']['results_dir'] = new + '/ray_results'
    cfg['options']['Training']['model_checkpoint_save_path'] = new + '/checkpoints'
    fit = cfg['dgpo']['adaptive_omnifold']['audit_fit']
    fit['representation_export_dir'] = new + '/panels'
    fit['diagnostic_snapshot_dir'] = new + '/diagnostic_snapshots'
    if fit.get('ratio_audit_export_dir'):
        fit['ratio_audit_export_dir'] = new + '/ratio_audit'
    cfg['experiment']['attempt_suffix'] = suffix
    return cfg


if __name__ == '__main__':
    raise SystemExit(main())
