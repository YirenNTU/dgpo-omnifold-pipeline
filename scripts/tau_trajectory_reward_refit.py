"""Explicit startup refit contract for the final step1920 trajectory ablation."""
from __future__ import annotations

import copy


def validate_reward_refit(value):
    if value is None:
        return None
    required = {'anchor_step', 'ratio_bound', 'reference_recenter'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('reward_refit requires explicit anchor_step, ratio_bound and reference_recenter')
    if type(value['anchor_step']) is not int or value['anchor_step'] != 1920:
        raise ValueError('Fresh final-finetuning reward requires the source step1920 anchor')
    if isinstance(value['ratio_bound'], bool) or value['ratio_bound'] not in (30, None):
        raise ValueError('reward_refit.ratio_bound must be 30 or None (unbounded BCE logit)')
    if value['reference_recenter'] is not True:
        raise ValueError('Fresh reward and velocity reference must use the same step1920 anchor')
    return copy.deepcopy(value)
