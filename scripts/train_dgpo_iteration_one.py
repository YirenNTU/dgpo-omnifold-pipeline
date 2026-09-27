#!/usr/bin/env python3
"""Verify the pinned diagnostic policy, then launch the production DGPO ablation."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'config/dgpo_omnifold_ztautau_10pct_iteration1_only.yaml'


def state_digest(state):
    # Same complete tensor digest as diagnose_raw_monitor_replay.state_digest.
    if not state:
        raise ValueError('Empty live policy state_dict')
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid live policy tensor: {name}')
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def verify(config):
    dgpo = config['dgpo']
    recal = dgpo['adaptive_omnifold']['recalibration']
    if recal.get('iteration_one_only') is not True or (recal['min_iterations'], recal['max_iterations']) != (1,1):
        raise ValueError('Not an iteration-one-only configuration')
    if dgpo['checkpoint_load_mode'] != 'weights_only' or dgpo['auto_resume_from_last']:
        raise ValueError('This ablation must start with weights only and a fresh clock')
    training = config['options']['Training']
    if any(training['EMA'].get(k, False) for k in ('replace_model_after_load','use_for_generation')):
        raise ValueError('Expected live policy, not EMA')
    source = Path(training['model_checkpoint_load_path'])
    output = Path(training['model_checkpoint_save_path'])
    if output.resolve() == source.parent.resolve():
        raise ValueError('Output must not overwrite the source checkpoint directory')
    if output.exists() and any(output.glob('*.ckpt')):
        raise FileExistsError(f'Fresh-run output already contains checkpoints: {output}; use a new output directory')
    provenance = config['nersc']['reproducibility']
    try:
        payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError as exc:
        if 'mmap can only be used' not in str(exc):
            raise
        payload = torch.load(source, map_location='cpu', weights_only=False)
    step = payload.get('global_step')
    if step != provenance['source_dgpo_global_step']:
        raise ValueError(f'Source checkpoint step mismatch: {step}')
    digest = state_digest(payload['state_dict'])
    if digest != provenance['source_policy_sha256']:
        raise ValueError(f'Source live state_dict SHA256 mismatch: {digest}')
    for label, value in (
        ('classifier backbone', config['reward_config']['omnifold']['backbone_checkpoint']),
        ('training pool', config['platform']['data_parquet_dir']),
        ('validation pool', config['platform']['data_parquet_val_dir']),
    ):
        if not Path(value).exists():
            raise FileNotFoundError(f'{label}: {value}')
    print(f'Verified live policy: {source}\nSource step: {step}\nSHA256: {digest}\n'
          f'New DGPO clock: step=0 epoch=0; initial reward/monitor trained fresh.\n'
          f'Output: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    verify(config)
    if args.check_only:
        return
    subprocess.run([sys.executable, str(ROOT/'scripts/train_neutrino_backend.py'),
        '--backend','dgpo-evenet','--base-config',str(ROOT/'config/train_diffusion_nersc.yaml'),
        '--overlay-config',str(args.config.resolve()),'--','--ray-dir',
        config['nersc']['ray']['results_dir']], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
