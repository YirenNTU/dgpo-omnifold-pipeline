"""Observed-visible/candidate relations, with fit-only shared scaling.

These are input coordinates, not Cij constraints. Both classes use the same
reconstructed tau map. No truth value enters a generated event's features.
"""
import numpy as np


def relative_angles(tau_a, tau_b, visible_a, visible_b):
    def leg(tau, visible):
        tau, visible = (np.asarray(x, dtype=np.float64) for x in (tau, visible))
        if tau.shape != visible.shape or tau.ndim != 2 or tau.shape[1] != 4:
            raise ValueError('Expected aligned tau and visible [N,4]')
        if not np.isfinite(tau).all() or not np.isfinite(visible).all():
            raise ValueError('Nonfinite relative-angle input')
        if (np.linalg.norm(tau[:, 1:], axis=1) == 0).any() or (np.linalg.norm(visible[:, 1:], axis=1) == 0).any():
            raise ValueError('Undefined zero-momentum direction')
        theta = lambda p: np.arctan2(np.hypot(p[:, 1], p[:, 2]), p[:, 3])
        phi = lambda p: np.arctan2(p[:, 2], p[:, 1])
        dphi = phi(tau)-phi(visible)
        # Sin/cos remove the phi branch cut; theta uses canonical p4 angles,
        # not raw offsets that might wrap/reflect beyond a pole.
        return np.stack((theta(tau)-theta(visible), np.sin(dphi), np.cos(dphi)), axis=1)
    return np.concatenate((leg(tau_a, visible_a), leg(tau_b, visible_b)), axis=1)


def attach_relative_inputs(arrays):
    fit = arrays['split'] == 0
    joined = np.concatenate((arrays['relative_truth'][fit], arrays['relative_generated'][fit]))
    if joined.ndim != 2 or joined.shape[1] != 6 or not np.isfinite(joined).all():
        raise ValueError('Require finite six-dimensional relative inputs')
    mean = joined.mean(axis=0, dtype=np.float64)
    std = joined.std(axis=0, dtype=np.float64)
    scale = np.maximum(std, 1e-6)
    for target in ('truth', 'generated'):
        relative = ((arrays[f'relative_{target}']-mean)/scale).astype('float32')
        if not np.isfinite(relative).all():
            raise ValueError('Nonfinite normalized relative input')
        old = arrays[f'candidate_{target}']
        # Keep the original explicit 15 tau features LAST: diagnostic MMD
        # must not change geometry when changing the classifier's inputs.
        arrays[f'candidate_{target}'] = np.concatenate((old[:, :-15], relative, old[:, -15:]), axis=1)
    return dict(feature_order=['a_delta_theta','a_sin_delta_phi','a_cos_delta_phi',
        'b_delta_theta','b_sin_delta_phi','b_cos_delta_phi'], mean=mean.tolist(), scale=scale.tolist(),
        scale_floor=1e-6, floored_features=int((std < 1e-6).sum()),
        fitted_on='fit split only; pooled truth/generated equally', clipping=False,
        new_input_weights='zero; common head initialization and RNG preserved')


def paired_input_contract(arrays, packing_spec):
    """Structural condition-only check, NOT a separately trained audit.

    The trainer constructs both classes from one condition tensor and one
    event-weight tensor. For any deterministic c-only score, the two weighted
    score distributions must be identical. Verify this with a fixed projection.
    """
    from scripts.train_conditional_spin_ratio import pair_metrics
    forbidden = {'source_ids', 'source_event_key', 'source_event_index',
                 'source_sample_index', 'source_file_index', 'truth', 'truth_deltas', 'label'}
    if forbidden.intersection(packing_spec['shapes']):
        raise ValueError('Identity or target field in classifier condition packing')
    c = arrays['condition']
    direction = np.random.default_rng(910).normal(size=c.shape[1]) / np.sqrt(c.shape[1])
    scores = c @ direction
    rows = []
    for value, name in enumerate(('fit', 'validation', 'test')):
        take = arrays['split'] == value
        w = arrays['event_weight'][take]
        metrics = pair_metrics(scores[take], scores[take], w)
        if not np.isclose(metrics['auc'], .5, atol=1e-12):
            raise ValueError('Condition-only paired score distributions differ')
        rows.append(dict(split=name, events=int(take.sum()), condition_only_auc=metrics['auc'],
                         class_weight_ratio=1., optimal_constant_bce=float(np.log(2.))))
    return dict(splits=rows, candidates_per_condition=1,
        condition_shared_by_construction=True, event_weight_shared_by_construction=True,
        condition_only_test='deterministic projection on identical weighted pairs; not a trained control',
        classifier_call='one condition + one candidate; never paired candidates in the same input',
        identity_usage='alignment and split only; excluded from packed fields',
        limitation='Does not replace tau-only training or a fresh weighted joint audit')
