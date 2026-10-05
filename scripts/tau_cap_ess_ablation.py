"""Fixed-model clipping versus weights-only ESS-matched shrinkage.

Neither operation estimates a new exact density ratio. All normalization is
global; events retain all K candidates and the original truth target.
"""
import numpy as np

from scripts.tau_tail_attribution import AXES, candidate_weights, paired_bootstrap
from scripts.tau_cij_components import simultaneous_intervals
from scripts.tau_weight_factor_swap import status


def match_ess(original, unweighted, target_ess):
    """Return the smallest mixing lambda in [0,1], or None if unreachable.

    Solve the scalar quadratic on its monotone segments. This also handles
    nonuniform base weights, for which ESS need not be monotone in lambda.
    No observable, truth or Cij enters this calculation.
    """
    w, u = np.asarray(original, float), np.asarray(unweighted, float)
    if (w.shape != u.shape or not np.isfinite(w).all() or not np.isfinite(u).all()
            or (w < 0).any() or (u < 0).any()
            or not np.isclose(w.sum(), 1) or not np.isclose(u.sum(), 1)
            or not np.isfinite(target_ess) or not 1 <= target_ess <= w.size*(1+1e-12)):
        raise ValueError('Invalid normalized weights or ESS target')
    d = u-w
    a, b, c = float(np.square(d).sum()), float(2*np.sum(w*d)), float(np.square(w).sum())
    target = 1/target_ess
    tol = 1e-12*max(c, target)
    if abs(c-target) <= tol:
        return 0.
    if a == 0:
        return None
    turning = float(np.clip(-b/(2*a), 0, 1))
    knots = sorted(set((0., turning, 1.)))
    def f(x):
        return (a*x+b)*x+c-target
    for lo, hi in zip(knots[:-1], knots[1:]):
        fl, fh = f(lo), f(hi)
        if abs(fl) <= tol: return lo
        if abs(fh) <= tol: return hi
        if fl*fh > 0: continue
        for _ in range(70):
            mid = (lo+hi)/2
            fm = f(mid)
            if fl*fm <= 0:
                hi = mid
            else:
                lo, fl = mid, fm
        return (lo+hi)/2
    return None


def weight_arms(base, logits, caps):
    s = np.asarray(logits, float)
    if (len(caps) < 2 or caps[0] != 30 or len(set(caps)) != len(caps)
            or any(not np.isfinite(t) or t <= 0 or t > 30 for t in caps)
            or not np.isfinite(s).all() or s.max() > np.log(30)+1e-6):
        raise ValueError('Requires finite bound30 scores and unique caps starting at30')
    w, logmean = candidate_weights(base, s)
    u, _ = candidate_weights(base, np.zeros_like(s))
    yield 'unweighted', u, dict(log_mean_ratio=0., kind='unweighted')
    for cap in caps:
        # cap30 is the original trained bound30 endpoint, not an uncapped head.
        clipped = s if cap == 30 else np.minimum(s, np.log(cap))
        wc, lc = candidate_weights(base, clipped)
        ess = float(1/np.square(wc).sum())
        lam = match_ess(w, u, ess)
        cut = s > np.log(cap)
        yield f'cap{cap:g}', wc, dict(cap=cap, kind='cap', log_mean_ratio=lc,
            candidate_cut_fraction=float(cut.mean()),
            base_weighted_cut_fraction=float(np.sum(u*cut)),
            original_mass_above_cap=float(np.sum(w*cut)),
            removed_unnormalized_ratio_mass_fraction=float(-np.expm1(lc-logmean)),
            matched_mix_lambda=lam,
            matching_status='matched' if lam is not None else 'unreachable_ESS')
        if cap != 30 and lam is not None:
            mix = (1-lam)*w+lam*u
            actual = float(1/np.square(mix).sum())
            if not np.isclose(actual, ess, rtol=1e-9, atol=1e-9):
                raise ValueError('Mixture did not match candidate ESS')
            yield f'mix{cap:g}', mix, dict(kind='ESS_matched_mix', mix_lambda=lam,
                matched_cap=cap, target_candidate_ess=ess, log_mean_ratio=None)


def analyze(inputs, generated, logits, cfg, progress=None):
    base, truth = np.asarray(inputs['weight'], float), np.asarray(inputs['truth_cij'], float)
    g, s = np.asarray(generated), np.asarray(logits)
    n, k = s.shape
    pt, cats, ids = (np.asarray(inputs[key]) for key in ('visible_pt_sum', 'category', 'source_ids'))
    edges = np.asarray(cfg['condition_pt_edges'])
    if (g.shape != (n, k, 9) or truth.shape != (n, 9)
            or not np.isfinite(g).all() or not np.isfinite(truth).all()
            or pt.shape != (n,) or cats.shape != (n,) or ids.shape != (n,)
            or not np.isfinite(pt).all() or len(np.unique(ids)) != n
            or edges.ndim != 1 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all()):
        raise ValueError('Invalid or unaligned observables')
    target = np.average(truth, weights=base, axis=0)
    bins = np.searchsorted(edges, pt, side='right')
    rows, groups, nums, dens = {}, [], [], []
    for name, w, metadata in weight_arms(base, s, cfg['caps']):
        den = w.sum(1)
        num = np.einsum('nk,nkd->nd', w, g)
        c = num.sum(0)
        residual = c-target
        nums.append(num); dens.append(den)
        rows[name] = dict(metadata, C=c.reshape(3, 3).tolist(),
            error=float(np.linalg.norm(residual)), absolute_component_error=np.abs(residual).tolist(),
            diagonal_error=float(np.linalg.norm(residual[[0, 4, 8]])),
            offdiagonal_error=float(np.linalg.norm(residual[[1, 2, 3, 5, 6, 7]])),
            candidate_ess=float(1/np.square(w).sum()), candidate_ess_fraction=float(1/np.square(w).sum()/(n*k)),
            event_ess=float(1/np.square(den).sum()), event_ess_fraction=float(1/np.square(den).sum()/n),
            max_candidate_mass=float(w.max()))
        for cat in np.unique(cats):
            for bucket in [-1, *range(len(edges)+1)]:
                take = (cats == cat) & ((bins == bucket) if bucket >= 0 else True)
                bm = base[take].sum()/base.sum()
                if bm <= 0: continue
                mass = den[take].sum()
                if mass <= 0: raise ValueError('Group has zero candidate mass')
                gt = np.average(truth[take], weights=base[take], axis=0)
                gc = num[take].sum(0)/mass
                groups.append(dict(arm=name, category=int(cat), pt_bin=bucket,
                    grouping='decay' if bucket == -1 else 'decay_x_pt', events=int(take.sum()),
                    base_mass=float(bm), weighted_mass=float(mass), C=gc.tolist(), truth_C=gt.tolist(),
                    error=float(np.linalg.norm(gc-gt))))
        gr = [r for r in groups if r['arm'] == name and r['grouping'] == 'decay_x_pt']
        rows[name]['group_mass_tv'] = .5*sum(abs(r['weighted_mass']-r['base_mass']) for r in gr)
        rows[name]['base_weighted_group_error'] = sum(r['base_mass']*r['error'] for r in gr)
        if progress: progress('computed_'+name)
    if progress: progress('paired_event_bootstrap')
    draws, targets = paired_bootstrap(inputs, nums, dens, cfg['bootstrap'], cfg['bootstrap_seed'])
    names = list(rows)
    errors = np.array([rows[a]['error'] for a in names])
    components = np.array([rows[a]['absolute_component_error'] for a in names])
    de = np.linalg.norm(draws-targets[:, None], axis=-1)
    dc = np.abs(draws-targets[:, None])
    pairs = []
    for cap in cfg['caps'][1:]:
        for control in ('cap30', f'mix{cap:g}'):
            if control in rows: pairs.append((f'cap{cap:g}', control))
    points = np.array([errors[names.index(a)]-errors[names.index(b)] for a, b in pairs])
    changes = np.stack([de[:, names.index(a)]-de[:, names.index(b)] for a, b in pairs], axis=1)
    cp = np.stack([components[names.index(a)]-components[names.index(b)] for a, b in pairs])
    cd = np.stack([dc[:, names.index(a)]-dc[:, names.index(b)] for a, b in pairs], axis=1)
    bands, _ = simultaneous_intervals(points, changes)
    cbands, _ = simultaneous_intervals(cp, cd)
    contrasts = {}
    for j, (a, b) in enumerate(pairs):
        contrasts[a+'_minus_'+b] = dict(error_change=float(points[j]),
            pointwise_ci95=np.quantile(changes[:, j], [.025, .975]).tolist(),
            simultaneous_ci95=bands[:, j].tolist(), status=status(bands[:, j]),
            components=[dict(component=axis, absolute_error_change=float(cp[j, i]),
                simultaneous_ci95=cbands[:, j, i].tolist(), status=status(cbands[:, j, i]))
                for i, axis in enumerate(AXES)])
    for j, name in enumerate(names):
        rows[name]['minus_unweighted'] = dict(error_change=float(errors[j]-errors[0]),
            pointwise_ci95=np.quantile(de[:, j]-de[:, 0], [.025, .975]).tolist())
    report = dict(events=n, candidates=k, truth_C=target.reshape(3, 3).tolist(), arms=rows,
        contrasts=contrasts, groups=groups, primary='All cap20/10/5 versus cap30 and their ESS-matched mixtures',
        contrast_family=len(pairs), component_family=len(pairs)*9,
        matching='Candidate ESS only; event ESS need not match. Mix lambda chosen from weights only and fixed during bootstrap.',
        uncertainty='Paired whole-event bootstrap; fixed fitted model, transforms and K64 draws. Approximate simultaneous max-deviation bands, not corrected for prior test-set inspection.',
        interpretation='Cap versus mixture tests selective tail suppression versus generic shrinkage at equal ESS; does not certify ratio correctness or conditional closure.',
        selection='Exploratory sensitivity analysis; no automatic best-cap selection or deployment; no event or truth removal.')
    arrays = dict(source_ids=ids, arm_names=np.array(names), event_numerators=np.stack(nums, 1),
        event_masses=np.stack(dens, 1), bootstrap_C=draws, bootstrap_truth=targets)
    return report, arrays
