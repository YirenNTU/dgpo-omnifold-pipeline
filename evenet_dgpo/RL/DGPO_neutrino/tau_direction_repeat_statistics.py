"""Descriptive replication summaries for read-only step1920 direction probes.

Event-bootstrap intervals are conditional on a particular update draw and
evaluation-noise seed. They must not be pooled into an interval for training
noise, nor subtracted to make a confidence interval for a direction contrast.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _describe(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return dict(count=0, mean=None, minimum=None, maximum=None, sample_sd=None,
                    positive_count=0, negative_count=0, zero_count=0)
    return dict(count=int(len(array)), mean=float(array.mean()), minimum=float(array.min()),
                maximum=float(array.max()), sample_sd=float(array.std(ddof=1)) if len(array) > 1 else None,
                positive_count=int((array > 0).sum()), negative_count=int((array < 0).sum()),
                zero_count=int((array == 0).sum()))


def summarize_direction_repeats(draws):
    """Summarize a possibly incomplete, indexed collection of native draws.

Each intervention has direction/fraction/sign and an evaluations mapping.
Each evaluation contains paired_statistics' reward delta_mean/lo95/hi95.
The two evaluation seeds are averaged *within* each draw. Across-draw ranges
and SDs are descriptive (four draws are a screening experiment, not a power
calculation). Positive/negative CI counts describe only conditional event
bootstrap intervals. No candidate is treated as an independent event.
"""
    if not isinstance(draws, Mapping):
        raise ValueError('draws must be an indexed mapping')
    groups, cells, seed_contract = {}, {}, None
    for draw_id, draw in draws.items():
        interventions = draw.get('interventions')
        if not isinstance(interventions, Mapping):
            raise ValueError('Each draw requires indexed interventions')
        seen = set()
        for intervention in interventions.values():
            direction = intervention.get('direction')
            if direction not in ('raw_reward', 'raw_total', 'native_adamw'):
                raise ValueError('Unknown local direction')
            fraction = _finite(intervention.get('fraction'), 'fraction')
            sign = intervention.get('sign')
            if fraction <= 0 or type(sign) is not int or sign not in (-1, 1):
                raise ValueError('Positive radius and an integer signed direction required')
            key = (direction, fraction, sign)
            if key in seen:
                raise ValueError('Duplicate direction/radius/sign in a draw')
            seen.add(key)
            evaluations = intervention.get('evaluations')
            if not isinstance(evaluations, Mapping) or len(evaluations) < 2:
                raise ValueError('At least two separately seeded evaluations are required')
            seeds = tuple(sorted(map(str, evaluations)))
            if seed_contract is None:
                seed_contract = seeds
            if seeds != seed_contract:
                raise ValueError('Evaluation seeds must match across all paired directions and draws')
            values, intervals, validities = {}, {}, []
            for seed, evaluated in evaluations.items():
                validity = evaluated.get('measurement_valid', True)
                if type(validity) is not bool:
                    raise ValueError('measurement_valid must be a boolean')
                validities.append(validity)
                reward = evaluated['reward']
                mean = _finite(reward.get('delta_mean'), 'reward delta_mean')
                lo = _finite(reward.get('delta_lo95'), 'reward delta_lo95')
                hi = _finite(reward.get('delta_hi95'), 'reward delta_hi95')
                if lo > hi:
                    raise ValueError('Inverted event-bootstrap interval')
                values[str(seed)] = mean
                intervals[str(seed)] = dict(mean=mean, lo95=lo, hi95=hi)
            values_array = list(values.values())
            cell = dict(evaluation_noise_mean=float(np.mean(values_array)),
                        measurement_valid=all(validities),
                        evaluation_noise_range=float(max(values_array) - min(values_array)),
                        all_evaluation_point_estimates_positive=all(x > 0 for x in values_array),
                        all_evaluation_point_estimates_negative=all(x < 0 for x in values_array),
                        all_conditional_event_intervals_positive=all(x['lo95'] > 0 for x in intervals.values()),
                        all_conditional_event_intervals_negative=all(x['hi95'] < 0 for x in intervals.values()),
                        conditional_event_intervals=intervals)
            groups.setdefault(key, {})[str(draw_id)] = cell
            cells[str(draw_id), direction, fraction, sign] = values
    interventions = {}
    for (direction, fraction, sign), group in groups.items():
        key = f'{direction}_rms{fraction:g}_sign{sign:+d}'
        valid_group = [row for row in group.values() if row['measurement_valid']]
        interventions[key] = dict(direction=direction, fraction=fraction, sign=sign, draws=group,
            update_draw_means=_describe([row['evaluation_noise_mean'] for row in valid_group]),
            raw_including_invalid_update_draw_means=_describe([row['evaluation_noise_mean'] for row in group.values()]),
            invalid_draw_count=len(group)-len(valid_group),
            evaluation_noise_ranges=_describe([row['evaluation_noise_range'] for row in group.values()]),
            all_evaluation_sign_positive_draws=sum(row['all_evaluation_point_estimates_positive'] for row in valid_group),
            all_evaluation_sign_negative_draws=sum(row['all_evaluation_point_estimates_negative'] for row in valid_group),
            all_conditional_ci_positive_draws=sum(row['all_conditional_event_intervals_positive'] for row in valid_group),
            all_conditional_ci_negative_draws=sum(row['all_conditional_event_intervals_negative'] for row in valid_group))

    signed_response = {}
    for direction, fraction, sign in groups:
        if sign != 1 or (direction, fraction, -1) not in groups:
            continue
        rows = {}
        for draw_id in groups[direction, fraction, 1]:
            if (not groups[direction, fraction, 1][draw_id]['measurement_valid']
                    or not groups[direction, fraction, -1].get(draw_id, {}).get('measurement_valid', False)):
                continue
            plus = cells[draw_id, direction, fraction, 1]
            minus = cells.get((draw_id, direction, fraction, -1))
            if minus is None:
                continue
            odd = np.asarray([(plus[seed] - minus[seed]) / 2 for seed in plus])
            even = np.asarray([(plus[seed] + minus[seed]) / 2 for seed in plus])
            rows[draw_id] = dict(odd_mean=float(odd.mean()), even_mean=float(even.mean()),
                central_slope_mean=float(odd.mean() / fraction),
                absolute_even_to_odd=(float(abs(even.mean()) / abs(odd.mean())) if abs(odd.mean()) > 1e-15 else None),
                odd_evaluation_noise_range=float(np.ptp(odd)), even_evaluation_noise_range=float(np.ptp(even)))
        if rows:
            signed_response[f'{direction}_rms{fraction:g}'] = dict(draws=rows,
                odd_response=_describe([row['odd_mean'] for row in rows.values()]),
                even_response=_describe([row['even_mean'] for row in rows.values()]),
                central_slope=_describe([row['central_slope_mean'] for row in rows.values()]),
                scope='Descriptive central difference; no joint bootstrap CI. Even response flags curvature or finite-precision effects, not their cause.')

    contrasts = {}
    for first, second in (('raw_total', 'raw_reward'), ('native_adamw', 'raw_total')):
        for direction, fraction, sign in groups:
            if direction != first:
                continue
            rows = {}
            for draw_id in groups[direction, fraction, sign]:
                if (not groups[direction, fraction, sign][draw_id]['measurement_valid']
                        or not groups.get((second, fraction, sign), {}).get(draw_id, {}).get('measurement_valid', False)):
                    continue
                a = cells[draw_id, first, fraction, sign]
                b = cells.get((draw_id, second, fraction, sign))
                if b is not None:
                    rows[draw_id] = float(np.mean([a[seed] - b[seed] for seed in a]))
            if rows:
                contrasts[f'{first}_minus_{second}_rms{fraction:g}_sign{sign:+d}'] = dict(
                    draw_means=rows, across_draws=_describe(list(rows.values())), paired_ci_available=False,
                    scope='Same evaluation identities/noise; point-estimate contrast only. Do not subtract endpoint CIs. Parameter-RMS matched, NOT output-motion matched.')
    return dict(draw_count=len(draws), evaluation_seed_count=len(seed_contract or ()),
        evaluation_seeds=list(seed_contract or ()), interventions=interventions,
        signed_response=signed_response, direction_contrasts=contrasts,
        uncertainty='Update-draw dispersion, evaluation-noise dispersion and conditional event-bootstrap intervals are separate. No pooled CI; four native shuffled batches are not an IID training-seed study.',
        limitations='A null or mixed result does not rule out a mechanism. Invalid perturbations are excluded from sign counts, signed responses and direction contrasts (raw values retained). Check realized parameter and tau-direction motion before interpreting tiny radii. Fixed-head reward is uptake, not fresh-classifier or physics closure.')
