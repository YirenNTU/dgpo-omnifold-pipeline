"""Explicit coefficient-one current/anchor classifier-KL protocol."""
import copy
import math


def validate_classifier_kl(value):
    if value is None:
        return None
    required = {'anchor_step', 'coefficient', 'refit_every_updates'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('classifier_kl requires anchor_step, coefficient and refit_every_updates')
    if type(value['anchor_step']) is not int or value['anchor_step'] != 1920:
        raise ValueError('Classifier KL requires the fixed step1920 anchor')
    if isinstance(value['coefficient'], bool) or value['coefficient'] != 1:
        raise ValueError('Classifier KL preserves coefficient 1, without gradient balancing')
    if type(value['refit_every_updates']) is not int or value['refit_every_updates'] != 1:
        raise ValueError('Classifier KL must be freshly fitted before every actor update')
    return copy.deepcopy(value)


def validate_classifier_trust(value):
    if value is None:
        return None
    required = {'anchor_step', 'max_kl', 'backtrack_factor', 'max_backtracks',
                'confidence', 'validation_seed', 'evaluation_candidates'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('hard_trust_region requires the explicit classifier-KL gate protocol')
    if type(value['anchor_step']) is not int or value['anchor_step'] != 1920:
        raise ValueError('Hard classifier trust requires the fixed step1920 anchor')
    for key in ('max_kl', 'backtrack_factor', 'confidence'):
        x = value[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f'hard_trust_region.{key} must be finite and positive')
    if value['backtrack_factor'] >= 1 or not .5 < value['confidence'] < 1:
        raise ValueError('Require backtrack_factor < 1 and .5 < confidence < 1')
    if type(value['max_backtracks']) is not int or not 0 <= value['max_backtracks'] <= 10:
        raise ValueError('Declare zero through ten hard-trust backtracks')
    if type(value['validation_seed']) is not int or value['validation_seed'] < 0:
        raise ValueError('Declare a nonnegative hard-trust validation seed')
    if type(value['evaluation_candidates']) is not int or value['evaluation_candidates'] != 1:
        raise ValueError('Classifier trust uses one independent draw per complete held-out conditioning row')
    return copy.deepcopy(value)
