"""Fit-only, mask-aware preprocessing for the small conditional ratio MLP.

This is not the generator normalizer: policy inputs and saved samples stay fixed.
Only the MLP's x statistics are pooled over valid particles instead of slots.
No extra log, CDF, Fourier, or physics-feature transformation is introduced.
"""
from __future__ import annotations

import numpy as np


def layout(spec):
    spans, offset = {}, 0
    for key, shape in spec['shapes'].items():
        width = int(np.prod(shape, dtype=int))
        spans[key] = (slice(offset, offset + width), tuple(shape))
        offset += width
    return spans, offset


def particle_view(condition, spec):
    spans, width = layout(spec)
    if condition.shape[1] != width + 16:
        raise ValueError('Expected packed condition followed by 16 decay indicators')
    xs, shape = spans['x']
    ms, _ = spans['x_mask']
    if len(shape) != 2 or ms.stop - ms.start != shape[0]:
        raise ValueError('Expected x=[particles,features] and matching x_mask')
    mask = condition[:, ms]
    if not np.isin(mask, (0, 1)).all():
        raise ValueError('Particle masks must be binary')
    return condition[:, xs].reshape(len(condition), *shape), mask.astype(bool)


def _stats(values):
    if len(values) == 0:
        raise ValueError('No valid fitting values for normalization')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite valid normalization inputs')
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    binary = np.isin(values, (0, 1)).all(axis=0)
    mean[binary], std[binary] = 0., 1.
    std = np.where(std > 1e-6, std, 1.)
    return mean, std


def fit_masked_feature(condition, fit, spec):
    spans, width = layout(spec)
    x, mask = particle_view(condition, spec)
    fit = np.asarray(fit, bool)
    mean = np.zeros(condition.shape[1], dtype=np.float64)
    scale = np.ones_like(mean)
    xs, _ = spans['x']
    # Pool the same raw feature across valid particle slots. Padding never
    # contributes to the mean, scale, or binary-feature decision.
    m, s = _stats(x[fit][mask[fit]])
    mean[xs], scale[xs] = np.tile(m, x.shape[1]), np.tile(s, x.shape[1])
    for key, (span, _) in spans.items():
        if key in ('x', 'x_mask') or key.endswith('_mask'):
            continue
        valid = fit.copy()
        if key == 'conditions' and 'conditions_mask' in spans:
            ms, _ = spans['conditions_mask']
            cm = condition[:, ms]
            if cm.shape[1] != 1 or not np.isin(cm, (0, 1)).all():
                raise ValueError('Expected one binary global-condition mask per event')
            valid &= cm[:, 0].astype(bool)
            if not valid.any():
                continue
        mean[span], scale[span] = _stats(condition[valid, span])
    # The appended channel one-hot and all masks retain their original values.
    return mean.astype('float32'), scale.astype('float32')


def apply_masked_feature(condition, mean, scale, spec):
    x, mask = particle_view(condition, spec)
    spans, _ = layout(spec)
    values = condition.copy()
    xs, _ = spans['x']
    values[:, xs] = np.where(mask[..., None], x, 0).reshape(len(values), -1)
    values = np.clip((values - mean) / scale, -20, 20).astype('float32')
    x_new = values[:, xs].reshape(x.shape)
    x_new[~mask] = 0
    if 'conditions' in spans and 'conditions_mask' in spans:
        cs, _ = spans['conditions']
        ms, _ = spans['conditions_mask']
        values[:, cs] = np.where(condition[:, ms].astype(bool), values[:, cs], 0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite preprocessed condition')
    return values


def preprocessing_report(condition, split, spec):
    x, mask = particle_view(condition, spec)
    rows = []
    for index, name in enumerate(('train', 'validation', 'test')):
        take = split == index
        valid = mask[take]
        values = x[take]
        clipped = (np.abs(values) >= 20) & valid[..., None]
        denom = max(1, int(valid.sum()) * x.shape[-1])
        rows.append(dict(split=name, events=int(take.sum()),
            valid_feature_clip_fraction=float(clipped.sum()/denom),
            events_with_clipped_valid_features=int(clipped.any(axis=(1, 2)).sum()),
            padding_nonzero_values=int(np.count_nonzero(values[~valid])),
            slots=[dict(slot=j, valid_particles=int(valid[:, j].sum()),
                clipped_features=int(clipped[:, j].sum())) for j in range(x.shape[1])]))
    return dict(splits=rows, scope='Normalized MLP inputs; generator normalization unchanged')


def ratio_group_report(base, logits, positive_logits, category, multiplicity):
    from scripts.diagnose_conditional_tau_ratio_tail import normalized_weights, health
    base, logits = np.asarray(base, float), np.asarray(logits, float)
    positive_logits = np.asarray(positive_logits, float)
    weights, _ = normalized_weights(base, logits)
    rows = []
    groups = [('all', np.ones(len(base), bool))]
    groups += [(f'category_{k}', category == k) for k in np.unique(category)]
    groups += [(f'multiplicity_{k}', multiplicity == k) for k in np.unique(multiplicity)]
    groups += [(f'category_{k}_multiplicity_{m}', (category == k) & (multiplicity == m))
        for k in np.unique(category) for m in np.unique(multiplicity)]
    for name, take in groups:
        if not take.any() or base[take].sum() <= 0:
            continue
        rows.append(dict(group=name, **health(base[take], logits[take]),
            base_fraction=float(base[take].sum()/base.sum()),
            reweighted_fraction=float(weights[take].sum()),
            truth_logit_mean=float(np.average(positive_logits[take], weights=base[take])),
            generated_logit_mean=float(np.average(logits[take], weights=base[take]))))
    return dict(groups=rows,
        identity='For the true conditional ratio, E_q[r|c]=1; coarse-group log means should approach 0.',
        limitation='K=1 group estimates have sampling error; no per-event normalization or calibration is applied.')
