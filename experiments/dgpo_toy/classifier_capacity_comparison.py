"""Toy-only width comparison against saved adequately trained classifiers.

No truth changes, no learned-policy updates and no selection using test labels.
Used after the paired narrow fits; no repeated width search is implemented.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from .reward_transfer_comparison import assert_same_panel, paired_intervals


def context_bce(scores):
    positive, negative = (scores[key].double() for key in ('positive', 'negative'))
    if positive.ndim != 1 or positive.shape != negative.shape:
        raise ValueError('Need paired one-dimensional classifier scores')
    value = .5 * (F.softplus(-positive) + F.softplus(negative))
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite classifier scores')
    return value


def capacity_contrasts(scores):
    losses = {key: context_bce(value) for key, value in scores.items()}
    if len({len(v) for v in losses.values()}) != 1:
        raise ValueError('Paired contrasts need identical sample lengths')
    narrow = losses['narrow_both'] - losses['narrow_plain']
    wide = losses['wide_both'] - losses['wide_plain']
    return {'narrow_fourier_minus_plain': narrow,
            'wide_fourier_minus_plain': wide,
            'width_interaction': narrow - wide,
            'narrow_plain_minus_chance': losses['narrow_plain'] - math.log(2.)}


def decision(contrasts, *, valid, margin):
    if not valid:
        return 'inconclusive_fit_budget'
    if (contrasts['width_interaction']['hi95'] < -margin
            and contrasts['narrow_fourier_minus_plain']['hi95'] < -margin):
        return 'supports_larger_fourier_advantage_at_narrow_width'
    if contrasts['narrow_plain_minus_chance']['hi95'] < -margin:
        return 'narrow_plain_still_detects_no_resolved_material_width_interaction'
    return 'unresolved_not_evidence_of_absolute_plain_inability'


def compare_saved_width(plan, panels, fits, scores):
    # Import locally: the lab calls this after fitting and owns the path checks.
    from .closed_loop_lab import toy_path, score_metrics
    if len(panels) != 1 or set(plan['features']) != {'plain', 'both'}:
        raise ValueError('Width comparison requires one population and a matched feature pair')
    dataset = next(iter(panels))
    merged = {f'narrow_{mode}': scores[f'{dataset}__{mode}'] for mode in ('plain', 'both')}
    controls = plan['classifier_width_controls']
    if set(controls) != {'plain', 'both'}:
        raise ValueError('Both wide feature controls required')
    metadata = {}
    for mode, spec in controls.items():
        directory = toy_path(spec['directory'])
        report = json.loads((directory/'report.json').read_text())
        fit, old_plan = report['fit'], report['plan']
        key = spec['key']
        old_dataset, old_mode = key.rsplit('__', 1)
        if old_mode != mode or old_plan['classifier_seed'] != plan['classifier_seed']:
            raise ValueError('Control feature/seed mismatch')
        if fit.get('width', 128) != 128 or plan['classifier_width'] != 32:
            raise ValueError('This declared experiment is width32 vs saved128 only')
        if not fit['cold_start'] or not fit['test'][key]['valid']:
            raise ValueError('Saved wide control was not an adequate cold fit')
        for field in ('check_every', 'patience_checks', 'min_delta'):
            if fit[field] != fits[field]:
                raise ValueError('Stopping rule mismatch: '+field)
        saved_panel = torch.load(directory/f'{old_dataset}_panels.pt',
                                 map_location='cpu', weights_only=True)
        for split in ('train', 'validation', 'test'):
            assert_same_panel(panels[dataset][split], saved_panel[split], include_negative=True)
        saved_scores = torch.load(directory/'classifiers/test_scores.pt',
                                  map_location='cpu', weights_only=True)[key]
        metrics = score_metrics(saved_scores)
        if not math.isclose(metrics['bce'], fit['test'][key]['bce'], abs_tol=1e-10):
            raise ValueError('Saved control scores disagree with selected report')
        merged[f'wide_{mode}'] = saved_scores
        metadata[mode] = {'directory': str(directory), 'key': key, **fit['test'][key],
                          'minimum_fit_steps': fit['minimum_steps'], 'maximum_fit_steps': fit['max_steps']}
    contrasts = paired_intervals(capacity_contrasts(merged), repeats=plan['bootstrap_repeats'])
    valid = all(value['valid'] for value in fits['test'].values())
    return {'valid': valid, 'all_panel_fields_exact': True, 'wide_controls': metadata,
            'metrics': {name: score_metrics(value) for name, value in merged.items()},
            'bce_contrasts': contrasts,
            'decision': decision(contrasts, valid=valid, margin=plan['material_bce_margin']),
            'limitation': 'Single-seed adaptive width/stopping-protocol contrast; changed width also changes initialization geometry. '
                'Wide controls reuse adequate selected fits with recorded unequal minimum budgets. '
                'Not equal FLOPs, a global expressivity bound, or an EveNet failure reproduction. '
                'No oracle features; truth, samples and policy unchanged.'}
