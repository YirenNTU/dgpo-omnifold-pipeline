"""Fit balanced truth/anchor BCE once, then freeze it for final finetuning.

No new allocation or nested trainer: generation and head fitting use all existing
workers. The classifier and velocity reference both anchor at source step1920.
Actor parameters, AdamW and scheduler are never updated by this startup refit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.tau_trajectory_reward_refit import validate_reward_refit


def prepare_trajectory_reward(cycle, options, folder, *, epoch, step, min_selected_steps=0):
    from RL.DGPO_neutrino.checkpoint_transfer import isolated_evaluation
    from RL.DGPO_neutrino.conditional_tau_reward import validate_head
    from RL.DGPO_neutrino.conditional_tau_cycle import barrier

    cfg = validate_reward_refit(options)
    if cfg is None or step != cfg['anchor_step']:
        raise ValueError('Startup reward refit must precede all updates at step1920')
    folder = Path(folder)
    cycle.train = cycle.read_panel(cycle.cfg['train_panel'])
    if (len(cycle.train['source_ids']) != 416701
            or len(cycle.validation['source_ids']) != 119002
            or np.intersect1d(cycle.train['source_ids'], cycle.validation['source_ids']).size):
        raise ValueError('Reward refit requires complete disjoint filtered train/validation panels')
    with isolated_evaluation(cycle.actor, int(cycle.cfg['refit_seed']) + cycle.rank):
        generated = cycle.generate(cycle.train, folder / 'anchor-candidates', 1,
            cycle.cfg['refit_seed'] + step, False)
    if generated['features'].shape != cycle.train['candidate_truth'].shape:
        raise ValueError('Require exactly one anchor candidate paired with each truth event')
    _, saved, status, arrays = cycle.fit(cycle.train, generated, folder / 'classifier',
        audit=False, step=step, reward_ratio_bound=cfg['ratio_bound'], log_phase='trajectory_reward_refit',
        min_selected_steps=min_selected_steps)
    if (saved.get('ratio_bound') != cfg['ratio_bound']
            or saved.get('source_policy_step') != step
            or saved.get('fitting_current_policy') is not True
            or not status.get('minimum_fit_steps_met')
            or status['total_steps'] < max(1000, saved['min_steps'])):
        raise ValueError('Fresh reward must preserve its bound/denominator and reach the classifier fit budget')
    if min_selected_steps and status['best_steps'] < min_selected_steps:
        raise ValueError('Selected startup reward did not reach the classifier fit budget')
    validate_head(saved)
    if any(not torch.isfinite(value).all() for value in saved['state_dict'].values()):
        raise FloatingPointError('Nonfinite fresh reward checkpoint')
    saved['training_candidates'] = 1
    saved['startup_final_finetuning_refit'] = True
    cycle.reward.install(saved, policy_step=step, epoch=epoch)
    cycle.reference.load_state_dict(cycle.actor.state_dict(), strict=True)
    cycle.reference.eval().requires_grad_(False)
    counts = {str(split): int(np.count_nonzero(arrays['split'] == split)) for split in (0, 1, 2)}
    report = dict(complete=True, anchor_step=step, ratio_bound=cfg['ratio_bound'],
        reward='unbounded_log_ratio' if cfg['ratio_bound'] is None else 'bounded_log_ratio',
        truth_events=416701, anchor_events=416701, training_candidates=1,
        per_class_split_events=counts, paired_class_weights=True,
        normalization='unchanged pinned checkpoint normalization',
        head_checkpoint=str(folder / 'classifier' / 'best.pt'), fit=status,
        reference_recentered=True, actor_optimizer_reset=False,
        scope='One startup refit, then fixed teacher/reference; not periodic refitting, exact KL or a hard trust region')
    if cycle.rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'reward_refit.json').write_text(json.dumps(report, indent=2) + '\n')
        cycle.emit({**{f'tau/trajectory_reward_refit/{k}': v for k, v in status.items()},
            'tau/trajectory_reward_refit/unbounded': int(cfg['ratio_bound'] is None),
            'tau/trajectory_reward_refit/anchor_step': step,
            'tau/trajectory_reward_refit/reference_recentered': 1,
            'tau/trajectory_reward_refit/truth_events': 416701,
            'tau/trajectory_reward_refit/anchor_events': 416701}, step)
    barrier(cycle.world)
    return report
