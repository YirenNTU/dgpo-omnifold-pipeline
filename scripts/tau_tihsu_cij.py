"""tihsu Cij reconstruction/histogram convention, without training dependencies.

Reference: tihsu99/LEP_tree_ana cmo_update (also master as inspected),
quantum/observables_builder.py, quantum/unfold.py, utils/tau_decay.py;
ml_pipeline/util/export_evenet_prediction_to_qi.py and post_calibration.py.

These are PRE-UNFOLDING monitors. Final Cij must run the original QIProcessor
with its response, backgrounds, region cuts and normalization. A filtered
candidate panel cannot supply that inclusive response or its misses/fakes.
Historical raw-moment monitors deliberately remain unchanged.
"""
from __future__ import annotations

import numpy as np
from scripts.diagnose_ztautau_cij import angles, tau_from_deltas, unit

PROTOCOL = 'tihsu-calibrated-channel-histogram-v1'
AXES = ('n', 'r', 'k')
EDGES = np.linspace(-1., 1., 11)
ANALYZING_POWERS = np.array([0., 1., .41, -.33, -.34, 0.])


def analyzing_powers(category):
    category = np.asarray(category)
    if (not np.isfinite(category).all() or not np.equal(category, category.astype(int)).all()):
        raise ValueError('Invalid decay category')
    a, b = category.astype(int) // 10, category.astype(int) % 10
    if np.any((a < 1) | (a > 4) | (b < 1) | (b > 4)):
        raise ValueError('Cij requires a supported decay channel with nonzero analyzing power')
    return np.stack((-ANALYZING_POWERS[a], ANALYZING_POWERS[b]), axis=-1)


def post_calibrate(tau_a, tau_b):
    """Same map as util/post_calibration: normalize EACH leg before subtraction."""
    direction = unit(unit(np.asarray(tau_a, float)[..., 1:])
                     - unit(np.asarray(tau_b, float)[..., 1:]))
    p = np.sqrt(45.6**2 - 1.777**2) * direction
    energy = np.full(p.shape[:-1] + (1,), 45.6)
    return np.concatenate((energy, p), -1), np.concatenate((energy, -p), -1)


def observables(tau_a, tau_b, visible_a, visible_b):
    """No implicit calibration: truth p4 must retain their actual energies."""
    a, b = angles(tau_a, tau_b, visible_a, visible_b)
    # Existing local angles uses k,r,n; upstream uses n,r,k.
    a, b = a[..., [2, 1, 0]], b[..., [2, 1, 0]]
    from scripts.diagnose_ztautau_cij import boost
    pair = np.asarray(tau_a, float) + np.asarray(tau_b, float)
    acm = boost(np.asarray(tau_a, float), -pair[..., 1:] / pair[..., :1])
    theta = np.arccos(np.clip(np.abs(unit(acm[..., 1:])[..., 2]), 0, 1)) * 2 / np.pi
    mass2 = pair[..., 0]**2 - np.sum(pair[..., 1:]**2, axis=-1)
    if np.any(mass2 <= 0):
        raise ValueError('Non-timelike tau pair')
    result = dict(theta_cm=theta, mtautau=np.sqrt(mass2))
    for i, axis in enumerate(AXES):
        result[f'cos_theta_A_{axis}'] = a[..., i]
        result[f'cos_theta_B_{axis}'] = b[..., i]
        for j, other in enumerate(AXES):
            result[f'cos_theta_A_{axis}_times_cos_theta_B_{other}'] = a[..., i] * b[..., j]
    return result


def from_deltas(visible_a, visible_b, deltas):
    """Offsets are radians, in canonical a=tau+, b=tau- order, ending (2,2)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.shape[-2:] != (2, 2) or not np.isfinite(deltas).all():
        raise ValueError('Expected finite (...,leg,theta/phi) offsets')
    a = tau_from_deltas(visible_a, deltas[..., 0, :])
    b = tau_from_deltas(visible_b, deltas[..., 1, :])
    return observables(*post_calibrate(a, b), visible_a, visible_b)


def bin_indices(values):
    """ROOT convention: -1 included; +1 is overflow, NOT the last bin."""
    return np.digitize(np.asarray(values), EDGES) - 1


def histogram_moment(values, weights, coefficient):
    values, weights = np.broadcast_arrays(np.asarray(values, float), np.asarray(weights, float))
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError('Nonfinite observable/weight; select valid rows explicitly')
    bins = bin_indices(values).ravel()
    weights = weights.ravel()
    keep = (bins >= 0) & (bins < 10)
    counts = np.bincount(bins[keep], weights=weights[keep], minlength=10)
    sumw2 = np.bincount(bins[keep], weights=weights[keep]**2, minlength=10)
    if counts.sum() <= 0:
        raise ValueError('Nonpositive in-range histogram normalization')
    centers = (EDGES[:-1] + EDGES[1:]) / 2
    mean = float(counts @ centers / counts.sum())
    error = np.sqrt(np.sum((centers - mean)**2 * sumw2)) / counts.sum()
    return dict(value=float(coefficient * mean), error=float(abs(coefficient) * error),
                values=counts.tolist(), sumw2=sumw2.tolist(),
                underflow_weight=float(weights[bins < 0].sum()),
                overflow_weight=float(weights[bins >= 10].sum()))


def channel_matrices(obs, weights, kappa):
    """Call per channel, after explicit selection. No pooled kappa averaging."""
    kappa = np.asarray(kappa, float)
    if kappa.shape != (2,) or not np.isfinite(kappa).all() or np.any(kappa == 0):
        raise ValueError('Need one nonzero analyzing-power pair per channel')
    matrix = np.empty((3, 3)); histograms = {}
    for i, left in enumerate(AXES):
        for j, right in enumerate(AXES):
            key = f'cos_theta_A_{left}_times_cos_theta_B_{right}'
            h = histogram_moment(obs[key], weights, 9 / np.prod(kappa))
            matrix[i, j] = h['value']; histograms[key] = h
    return dict(Cij=matrix.tolist(), histograms=histograms)


def panel_report(panel, deltas):
    """All saved events, no invented central region cuts or truth information.

    K candidates contribute w/K separately to histograms BEFORE taking moments.
    Candidate-wise Sumw2 is not an event-bootstrap uncertainty for K>1, so only
    central Cij and bin sums are reported here.
    """
    d = np.asarray(deltas)
    n = len(panel['source_ids'])
    if d.ndim != 4 or d.shape[0] != n or d.shape[1] < 1 or d.shape[2:] != (2, 2):
        raise ValueError('Expected aligned [events,K,2,2] samples')
    if len(np.unique(panel['source_ids'])) != n:
        raise ValueError('Duplicate panel identities')
    expected = analyzing_powers(panel['category'])
    if not np.allclose(panel['kappas'], expected, rtol=0, atol=1e-6):
        raise ValueError('Stored analyzing powers do not match tihsu charge/channel convention')
    w = np.asarray(panel['event_weight'], float)
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid panel weights')
    from scripts.tau_root_export_geometry import panel_protocol, target_spin_observables, prediction_spin_observables
    protocol = panel_protocol(panel)
    target = target_spin_observables(panel)
    pred = prediction_spin_observables(panel, d)
    channels = {}
    for category in np.unique(panel['category']):
        take = panel['category'] == category
        if w[take].sum() <= 0:
            channels[str(int(category))] = dict(status='zero_weight')
            continue
        kap = expected[take][0]
        t = channel_matrices({k: v[take] for k, v in target.items()}, w[take], kap)
        p = channel_matrices({k: v[take] for k, v in pred.items()}, w[take, None] / d.shape[1], kap)
        for result in (t, p):
            for h in result['histograms'].values():
                h.pop('error'); h.pop('sumw2')
        diff = np.array(p['Cij']) - t['Cij']
        channels[str(int(category))] = dict(events=int(take.sum()), analyzing_powers=kap.tolist(),
            matched_target=t, generated=p, delta=diff.tolist(), error=float(np.linalg.norm(diff)))
    scope = ('Calibrated, per-channel 10-bin Cij on the complete saved filtered panel. '
             'Reference is true offsets plus reconstructed visible, not MC tau/MC visible. '
             'No central region selection, background subtraction or acceptance unfolding.')
    if protocol != PROTOCOL:
        scope = ('Root angle exporter target/evenet maps; target is full MC tau plus reco visible; '
                 'prediction uses common.py calibration. Per-channel 10-bin Cij on saved filtered events; '
                 'no central region selection or unfolding. Different target/prediction maps can leave '
                 'a reference gap even for perfect angular offsets.')
    return dict(protocol=protocol, axes=list(AXES), bin_edges=EDGES.tolist(), channels=channels,
        candidates=int(d.shape[1]), unfolded=False, final_analysis_match=False,
        scope=scope,
        uncertainty='Point estimates only; K candidates share an event. No independent-candidate error claimed.')


def monitoring_metrics(report, prefix):
    out = {}
    for category, result in report['channels'].items():
        if 'error' not in result:
            continue
        base = f'{prefix}/channel_{category}'
        out[base + '/matched_error'] = result['error']
        for family in ('generated', 'matched_target'):
            for i, left in enumerate(AXES):
                for j, right in enumerate(AXES):
                    out[f'{base}/{family}/{left}{right}'] = result[family]['Cij'][i][j]
    return out
