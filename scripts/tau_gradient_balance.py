"""Pure, checkpointable EMA scale controller; no autograd through coefficients."""
from __future__ import annotations

import copy
import math


def validate_reference_balance(cfg):
    if cfg is None:
        return None
    required = {'mode', 'target_ratio', 'ema_decay', 'min_coefficient',
                'max_coefficient', 'initial_coefficient', 'max_change_factor', 'epsilon'}
    if not isinstance(cfg, dict) or set(cfg) != required or cfg['mode'] != 'ema_norm':
        raise ValueError('reference_balance requires the complete ema_norm configuration')
    result = dict(cfg)
    for key in required-{'mode'}:
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'reference_balance.{key} must be finite numeric')
        result[key] = float(value)
    if (not 0 <= result['ema_decay'] < 1 or result['target_ratio'] <= 0
            or result['epsilon'] <= 0 or result['min_coefficient'] <= 0
            or not result['min_coefficient'] <= result['initial_coefficient'] <= result['max_coefficient']
            or result['max_change_factor'] < 1):
        raise ValueError('Invalid reference_balance bounds, decay, target or epsilon')
    return result


class EMAGradientBalance:
    def __init__(self, cfg):
        self.cfg = validate_reference_balance(cfg)
        if self.cfg is None:
            raise ValueError('EMA balance requires an explicit configuration')
        self.coefficient = self.cfg['initial_coefficient']
        self.ema_reward = self.ema_reference = None
        self.steps = 0

    def state_dict(self):
        return dict(schema_version=1, config=copy.deepcopy(self.cfg),
                    coefficient=self.coefficient, ema_reward=self.ema_reward,
                    ema_reference=self.ema_reference, steps=self.steps)

    def load_state_dict(self, state):
        if (state.get('schema_version') != 1 or state.get('config') != self.cfg
                or type(state.get('steps')) is not int or state['steps'] < 0):
            raise ValueError('Incompatible EMA balance checkpoint')
        coefficient = state.get('coefficient')
        if (isinstance(coefficient, bool) or not isinstance(coefficient, (int, float))
                or not math.isfinite(coefficient)
                or not self.cfg['min_coefficient'] <= coefficient <= self.cfg['max_coefficient']):
            raise ValueError('Invalid EMA balance checkpoint coefficient')
        ema = [state.get('ema_reward'), state.get('ema_reference')]
        if ((ema[0] is None) != (ema[1] is None) or any(
                x is not None and (isinstance(x, bool) or not isinstance(x, (int, float))
                                   or not math.isfinite(x) or x <= self.cfg['epsilon']) for x in ema)):
            raise ValueError('Invalid EMA balance checkpoint averages')
        self.coefficient, self.ema_reward, self.ema_reference, self.steps = (
            float(coefficient), *ema, state['steps'])

    def propose(self, reward_norm, reference_norm):
        if any(not math.isfinite(x) or x < 0 for x in (reward_norm, reference_norm)):
            raise FloatingPointError('Nonfinite/negative global component gradient norm')
        state = self.state_dict()
        state['steps'] += 1
        hold = min(reward_norm, reference_norm) <= self.cfg['epsilon']
        target = self.coefficient
        bounded = rate_limited = False
        if not hold:
            decay = self.cfg['ema_decay']
            state['ema_reward'] = reward_norm if self.ema_reward is None else decay*self.ema_reward+(1-decay)*reward_norm
            state['ema_reference'] = reference_norm if self.ema_reference is None else decay*self.ema_reference+(1-decay)*reference_norm
            target = self.cfg['target_ratio']*state['ema_reward']/(state['ema_reference']+self.cfg['epsilon'])
            value = min(self.cfg['max_coefficient'], max(self.cfg['min_coefficient'], target))
            bounded = value != target
            factor = self.cfg['max_change_factor']
            state['coefficient'] = min(self.coefficient*factor, max(self.coefficient/factor, value))
            rate_limited = state['coefficient'] != value
        metrics = dict(coefficient=state['coefficient'], target_coefficient=target,
                       held_for_zero_gradient=float(hold), bound_active=float(bounded),
                       rate_limit_active=float(rate_limited), steps=state['steps'],
                       ema_reward_norm=state['ema_reward'] or 0.0,
                       ema_reference_norm=state['ema_reference'] or 0.0)
        return state, metrics
