"""Saved-panel factorial diagnostic; hybrid masses are NOT learned density ratios."""
import numpy as np

from scripts.tau_tail_attribution import AXES, candidate_weights, paired_bootstrap
from scripts.tau_cij_components import simultaneous_intervals

ARMS = ('AA', 'AB', 'BA', 'BB', 'unweighted')
# Columns follow ARMS. Negative error contrasts mean improvement.
CONTRASTS = {
    'within_at_mass_A': [-1, 1, 0, 0, 0],
    'mass_at_within_A': [-1, 0, 1, 0, 0],
    'within_at_mass_B': [0, 0, -1, 1, 0],
    'mass_at_within_B': [0, -1, 0, 1, 0],
    'interaction': [1, -1, -1, 1, 0],
    'total_B_minus_A': [-1, 0, 0, 1, 0],
}


def factor(base, logits):
    weights, log_mean = candidate_weights(base, logits)
    shifted = np.asarray(logits, float) - np.max(logits, axis=1, keepdims=True)
    pi = np.exp(shifted)
    pi /= pi.sum(1, keepdims=True)
    mass = weights.sum(1)
    if not np.allclose(mass[:, None]*pi, weights, atol=1e-15, rtol=1e-10):
        raise ValueError('Factorization did not reproduce original candidate weights')
    return mass, pi, log_mean


def factor_weights(base, a, b):
    if np.shape(a) != np.shape(b):
        raise ValueError('A/B candidate shapes differ')
    ma, pa, la = factor(base, a)
    mb, pb, lb = factor(base, b)
    weights = dict(AA=ma[:, None]*pa, AB=ma[:, None]*pb,
                   BA=mb[:, None]*pa, BB=mb[:, None]*pb)
    weights['unweighted'], _ = candidate_weights(base, np.zeros_like(a))
    return weights, dict(AA=la, BB=lb, AB=None, BA=None, unweighted=0.)


def status(interval):
    lo, hi = interval
    return 'improved' if hi < 0 else 'worsened' if lo > 0 else 'unresolved'


def analyze(inputs, generated, a, b, cfg, progress=None):
    base = np.asarray(inputs['weight'], float)
    truth = np.asarray(inputs['truth_cij'], float)
    n, k = np.shape(a)
    pt = np.asarray(inputs['visible_pt_sum'])
    edges = np.asarray(cfg['condition_pt_edges'])
    if (generated.shape != (n, k, 9) or truth.shape != (n, 9)
            or not np.isfinite(generated).all() or not np.isfinite(truth).all()
            or pt.shape != (n,) or not np.isfinite(pt).all()
            or np.asarray(inputs['category']).shape != (n,)
            or edges.ndim != 1 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all()
            or np.asarray(inputs['source_ids']).shape != (n,)
            or len(np.unique(inputs['source_ids'])) != n):
        raise ValueError('Invalid observables or duplicate event identities')
    weights, logmeans = factor_weights(base, a, b)
    target = np.average(truth, weights=base, axis=0)
    nums, dens, rows, groups = [], [], {}, []
    bins = np.searchsorted(cfg['condition_pt_edges'], inputs['visible_pt_sum'], side='right')
    for name in ARMS:
        w = weights[name]
        den = w.sum(1)
        num = np.einsum('nk,nkd->nd', w, generated)
        c = num.sum(0)
        nums.append(num); dens.append(den)
        rows[name] = dict(C=c.reshape(3, 3).tolist(), error=float(np.linalg.norm(c-target)),
            absolute_component_error=np.abs(c-target).tolist(),
            candidate_ess=float(1/np.square(w).sum()), event_ess=float(1/np.square(den).sum()),
            max_candidate_mass=float(w.max()), log_mean_ratio=logmeans[name])
        for cat in np.unique(inputs['category']):
            for bucket in range(len(cfg['condition_pt_edges'])+1):
                take = (inputs['category'] == cat) & (bins == bucket)
                bm = base[take].sum()/base.sum()
                if bm <= 0:
                    continue
                mass = den[take].sum()
                if mass <= 0:
                    raise ValueError('Group has no positive candidate mass')
                gt = np.average(truth[take], weights=base[take], axis=0)
                gc = num[take].sum(0)/mass
                groups.append(dict(arm=name, category=int(cat), pt_bin=bucket, events=int(take.sum()),
                    base_mass=float(bm), weighted_mass=float(mass), C=gc.tolist(), truth_C=gt.tolist(),
                    error=float(np.linalg.norm(gc-gt)), signed_population_residual=(num[take].sum(0)-bm*gt).tolist()))
        gr = [g for g in groups if g['arm'] == name]
        rows[name]['group_mass_tv'] = .5*sum(abs(g['weighted_mass']-g['base_mass']) for g in gr)
        rows[name]['base_weighted_group_error'] = sum(g['base_mass']*g['error'] for g in gr)
    if progress:
        progress('paired_event_bootstrap')
    draws, targets = paired_bootstrap(inputs, nums, dens, cfg['bootstrap'], cfg['bootstrap_seed'])
    errors = np.array([rows[name]['error'] for name in ARMS])
    error_draws = np.linalg.norm(draws-targets[:, None, :], axis=-1)
    components = np.array([rows[name]['absolute_component_error'] for name in ARMS])
    component_draws = np.abs(draws-targets[:, None, :])
    matrix = np.asarray(list(CONTRASTS.values()))
    points = matrix @ errors
    deltas = error_draws @ matrix.T
    comp_points = matrix @ components
    comp_deltas = np.einsum('ca,baj->bcj', matrix, component_draws)
    # Two primary contrasts are declared before this diagnostic is run.
    primary_bands, _ = simultaneous_intervals(points[:2], deltas[:, :2])
    all_bands, _ = simultaneous_intervals(points, deltas)
    comp_bands, _ = simultaneous_intervals(comp_points, comp_deltas)
    contrasts = {}
    for j, name in enumerate(CONTRASTS):
        band = primary_bands[:, j] if j < 2 else all_bands[:, j]
        contrasts[name] = dict(error_change=float(points[j]),
            pointwise_ci95=np.quantile(deltas[:, j], [.025, .975]).tolist(),
            simultaneous_ci95=band.tolist(), family='primary2' if j < 2 else 'all6',
            status=status(band) if name != 'interaction' else
                ('resolved_interaction' if band[0] > 0 or band[1] < 0 else 'unresolved'),
            signed_C_change=(matrix[j] @ np.stack(nums).sum(1)).tolist(),
            components=[dict(component=axis, absolute_error_change=float(comp_points[j, i]),
                simultaneous_ci95=comp_bands[:, j, i].tolist(),
                status=status(comp_bands[:, j, i])) for i, axis in enumerate(AXES)])
    for j, name in enumerate(ARMS):
        delta = error_draws[:, j]-error_draws[:, -1]
        rows[name]['minus_unweighted'] = dict(error_change=float(errors[j]-errors[-1]),
            pointwise_ci95=np.quantile(delta, [.025, .975]).tolist())
    report = dict(events=n, candidates=k, truth_C=target.reshape(3, 3).tolist(), arms=rows,
        contrasts=contrasts, groups=groups,
        interpretation=dict(within_at_A=contrasts['within_at_mass_A']['status'],
            mass_at_A=contrasts['mass_at_within_A']['status'],
            interaction=contrasts['interaction']['status'], unique_root_cause_identified=False),
        uncertainty='Paired whole-event bootstrap; all candidates and truth stay together. Fixed trained models and K64 panel; no training uncertainty. Exploratory after prior test-set inspection.',
        component_family='Approximate max-deviation simultaneous band over all 6 contrasts x 9 components.',
        scope='First letter supplies event mass; second supplies candidate proportions. Hybrids are diagnostic masses, not calibrated density ratios; no extra cap applied. No proof of conditional closure or of the source of ratio error.')
    arrays = dict(source_ids=inputs['source_ids'], event_numerators=np.stack(nums, 1),
        event_masses=np.stack(dens, 1), arm_names=np.array(ARMS),
        bootstrap_C=draws, bootstrap_truth=targets)
    return report, arrays


def verify_endpoint(row, truth, saved):
    if not np.allclose(truth, saved['truth_C'], atol=1e-9, rtol=1e-8):
        raise ValueError('Saved truth endpoint differs')
    for key in ('C', 'error', 'event_ess', 'candidate_ess', 'max_candidate_mass', 'log_mean_ratio'):
        if not np.allclose(row[key], saved['arms']['bounded'][key], atol=1e-9, rtol=1e-8):
            raise ValueError('Saved bounded endpoint differs: '+key)
