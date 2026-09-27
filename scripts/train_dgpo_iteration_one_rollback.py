#!/usr/bin/env python3
"""Preflight and full-state resume of run122b9d84's recorded step155 best."""
import argparse
import math
from pathlib import Path
import subprocess
import sys

from train_neutrino_backend import build_runtime_config, read_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'config/dgpo_omnifold_ztautau_10pct_iteration1_rollback_resume.yaml'
sys.path.insert(0, str(ROOT/'evenet_dgpo'))


def validate_payload(payload, dgpo, *, initial_branch):
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
        AdaptiveOmniFoldState, resolve_adaptive_config, adaptive_audit_protocol_signature,
        migrate_unstarted_policy_warmup_after_resume,
    )
    required = {'state_dict', 'dgpo_checkpoint_version', 'dgpo_next_epoch',
                'dgpo_optimizer_state_dict', 'dgpo_ref_state_dict',
                'dgpo_round_ref_state_dict', 'dgpo_round_ref_sha256',
                'dgpo_omnifold_reward_stack', 'dgpo_omnifold_reward_metadata',
                'dgpo_adaptive_omnifold_state'}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f'Incomplete full-state checkpoint: {missing}')
    cfg = resolve_adaptive_config(dgpo)
    if dgpo['checkpoint_load_mode'] != 'resume' or not cfg.raw_rollback_to_best_on_plateau:
        raise ValueError('Expected full resume with global-best rollback enabled')
    if cfg.raw_best_scope != 'global' or not cfg.iteration_one_only:
        raise ValueError('Preserve global-best scope and iteration-one-only reward')
    state = AdaptiveOmniFoldState.from_dict(payload['dgpo_adaptive_omnifold_state'])
    signature = adaptive_audit_protocol_signature(cfg)
    if state.audit_protocol_signature != signature:
        raise ValueError(f'Raw audit protocol mismatch: saved={state.audit_protocol_signature}, current={signature}')
    if not state.raw_monitor_state or not payload['dgpo_omnifold_reward_stack']:
        raise ValueError('Missing saved monitor or installed reward stack')
    if migrate_unstarted_policy_warmup_after_resume(state, cfg=cfg):
        raise ValueError('Unexpected warmup migration; preserve the saved protocol')
    optimizer = payload['dgpo_optimizer_state_dict']
    if not {'optimizer', 'scheduler', 'lr_schedule'}.issubset(optimizer):
        raise ValueError('Missing AdamW/cosine scheduler state')
    for key in ('total_steps', 'min_lr_ratio'):
        if optimizer['lr_schedule'].get(key) != dgpo['lr_schedule'][key]:
            raise ValueError(f'Cosine protocol mismatch: {key}')
    if initial_branch:
        actual = tuple(payload.get(k) for k in ('epoch', 'global_step', 'dgpo_next_epoch', 'dgpo_epoch_step'))
        if actual != (15, 155, 15, 5):
            raise ValueError(f'Expected mid-epoch step155 snapshot; got {actual}')
        if state.raw_best_global_step != 155 or not math.isclose(
                state.raw_best_auc_gap, .40751809708503384, rel_tol=0, abs_tol=1e-10):
            raise ValueError('Source does not contain the W&B-recorded global best')
    return state


def preflight(runtime):
    from RL.DGPO_neutrino.model_utils import (
        resolve_dgpo_auto_resume_checkpoint, _load_checkpoint_metadata,
    )
    from RL.DGPO_neutrino.dgpo_trainer import _resolve_raw_best_policy_checkpoint
    from train_dgpo_iteration_one import state_digest

    c = read_yaml(runtime)
    d = c['dgpo']; t = c['options']['Training']
    output = Path(t['model_checkpoint_save_path'])
    fallback = Path(d['auto_resume_fallback_checkpoint_path'])
    if output.resolve() == fallback.parent.resolve():
        raise ValueError('Rollback branch must not overwrite the parent run')
    selected = resolve_dgpo_auto_resume_checkpoint(output, enabled=True, fallback_checkpoint_path=fallback)
    payload = _load_checkpoint_metadata(selected)
    state = validate_payload(payload, d, initial_branch=selected.resolve() == fallback.resolve())
    best = _resolve_raw_best_policy_checkpoint(
        state, output, global_scope=True,
        source_dirs=[selected.parent, *d['global_best_checkpoint_search_dirs']],
    )
    digest = state_digest(payload['state_dict'])
    for value in (c['reward_config']['omnifold']['backbone_checkpoint'],
                  c['platform']['data_parquet_dir'], c['platform']['data_parquet_val_dir']):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    for section in c.values():
        if isinstance(section, dict) and isinstance(section.get('default'), str):
            if not Path(section['default']).is_file():
                raise FileNotFoundError(section['default'])
    print(f'Preflight passed\nResume: {selected}\nLive policy SHA256: {digest}\n'
          f'Completed DGPO steps: {payload["global_step"]}; next epoch: {payload["dgpo_next_epoch"]}; '
          f'within-epoch progress: {payload.get("dgpo_epoch_step", 0)}\n'
          f'Global best: {best}\nReward round: {state.reward_round_id}\n'
          f'Output: {output}\nSaved stack/monitor/optimizer/cosine clocks preserved; no initial refit.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    runtime = build_runtime_config(base_config=ROOT/'config/train_diffusion_nersc.yaml',
                                   overlay_config=CONFIG, backend='dgpo-evenet')
    preflight(runtime)
    if not args.check_only:
        subprocess.run([sys.executable, str(ROOT/'scripts/train_neutrino_backend.py'),
                        '--backend', 'dgpo-evenet', '--base-config', str(ROOT/'config/train_diffusion_nersc.yaml'),
                        '--overlay-config', str(CONFIG), '--', '--ray-dir',
                        read_yaml(CONFIG)['nersc']['ray']['results_dir']], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
