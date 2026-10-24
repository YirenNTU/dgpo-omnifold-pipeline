"""Explicit evaluation weights; never changes classifier fitting or DGPO loss."""
import numpy as np

from scripts.tau_tail_attribution import candidate_weights


def validate_weighting(mode, event_weighting):
    if mode not in ('global', 'within_event'):
        raise ValueError('Reweighting mode must be global or within_event')
    if event_weighting not in ('dataset', 'uniform'):
        raise ValueError('Event weighting must be dataset or uniform')


def evaluation_panel(panel, event_weighting):
    validate_weighting('within_event', event_weighting)
    w = np.asarray(panel['event_weight'], dtype=np.float64)
    if (w.shape != (len(panel['source_ids']),) or not np.isfinite(w).all()
            or np.any(w < 0) or w.sum() <= 0):
        raise ValueError('Invalid evaluation event weights')
    return dict(panel, event_weight=np.ones_like(w) if event_weighting == 'uniform' else w)


def reweight_candidates(base, logits, mode):
    """Within-event: q[e,k] = normalized_base[e] * softmax(log_ratio[e])[k]."""
    validate_weighting(mode, 'dataset')
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError('Need at least one candidate per event')
    global_weights, log_mean = candidate_weights(base, logits)
    if mode == 'global':
        return global_weights, log_mean
    base = np.asarray(base, dtype=np.float64)
    mass = base / base.sum()
    relative = np.exp(logits - logits.max(axis=1, keepdims=True))
    relative /= relative.sum(axis=1, keepdims=True)
    weights = mass[:, None] * relative
    np.testing.assert_allclose(weights.sum(axis=1), mass, rtol=1e-12, atol=1e-15)
    return weights, log_mean
