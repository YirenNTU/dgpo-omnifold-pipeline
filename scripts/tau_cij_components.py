"""Exact observed event accounting, NOT causal attribution or a weight optimizer."""
import numpy as np

from scripts.tau_tail_attribution import AXES, candidate_weights, paired_bootstrap


def simultaneous_intervals(point, draws):
    """Approximate bootstrap max-deviation band over the complete declared family.

    All entries use the same physical units. No unstable studentization. Absolute
    errors are nonsmooth near zero; this is an exploratory bootstrap band, not an
    exact finite-sample guarantee or a correction for prior test-set inspection.
    """
    point, draws = np.asarray(point), np.asarray(draws)
    deviation = np.max(np.abs(draws-point).reshape(len(draws), -1), axis=1)
    radius = float(np.quantile(deviation, .95))
    return np.stack((point-radius, point+radius)), radius


def event_accounting(num, den, uniform_num, uniform_den, target):
    """Origin-invariant split of the globally normalized reweighting shift.

    within_i = sum_k w_ik (g_ik - uniform_candidate_mean_i)
    between_i = (event_mass_i - base_mass_i) * (uniform_mean_i - global_U)
    sum_i(within_i+between_i) = C_weighted-C_U.
    Multiplying by a fixed secant slope gives exact signed accounting of the
    absolute-error change, including overshoot. Shared weights mean these terms
    are not independent causal effects of individual events.
    """
    u, a = uniform_num.sum(0), num.sum(0)
    local = np.divide(uniform_num, uniform_den[:, None],
                      out=np.zeros_like(uniform_num), where=uniform_den[:, None] > 0)
    within = num-den[:, None]*local
    between = (den-uniform_den)[:, None]*(local-u)
    delta = within+between
    ru, ra = u-target, a-target
    slope = np.divide(ra+ru, np.abs(ra)+np.abs(ru), out=np.zeros_like(ra),
                      where=(np.abs(ra)+np.abs(ru)) > 0)
    return dict(delta=delta, within=within, between=between,
                absolute_error_contribution=delta*slope,
                within_absolute_error=within*slope, between_absolute_error=between*slope)


def component_attribution(inputs, accounting, j, cfg):
    c = accounting['absolute_error_contribution'][:, j]
    harmful, helpful = np.maximum(c, 0), np.maximum(-c, 0)
    order = np.argsort(-harmful, kind='stable')
    counts = sorted(set([1, 10, 100, max(1, int(np.ceil(len(c)*.01)))]))
    concentration = []
    for count in counts:
        idx = order[:min(count, len(c))]
        concentration.append(dict(events=len(idx), harmful_sum=float(harmful[idx].sum()),
            fraction_of_all_harm=None if harmful.sum() == 0 else float(harmful[idx].sum()/harmful.sum()),
            descriptive_change_without_these_terms=float(c.sum()-c[idx].sum())))
    # These are arithmetic contributions, NOT recomputed deletion estimators.
    top = {}
    for label, value in [('harmful', harmful), ('helpful', helpful)]:
        idx = np.argsort(-value, kind='stable')[:cfg['top_count']]
        top[label] = [dict(source_id=str(inputs['source_ids'][i]), category=int(inputs['category'][i]),
            contribution=float(c[i]), all_components=accounting['absolute_error_contribution'][i].tolist(),
            within=float(accounting['within_absolute_error'][i, j]),
            between=float(accounting['between_absolute_error'][i, j])) for i in idx if value[i] > 0]
    return dict(component=AXES[j], harmful_sum=float(harmful.sum()), helpful_sum=float(helpful.sum()),
        net=float(c.sum()), harmful_event_fraction=float(np.mean(c > 0)),
        helpful_event_fraction=float(np.mean(c < 0)),
        harmful_base_mass_fraction=float(inputs['weight'][c > 0].sum()/inputs['weight'].sum()),
        within_net=float(accounting['within_absolute_error'][:, j].sum()),
        between_net=float(accounting['between_absolute_error'][:, j].sum()),
        concentration=concentration, top_events=top)


def analyze(inputs, data, cfg, progress=None):
    g, logits, base = data['cij'], data['logits'], inputs['weight']
    n, k = logits.shape
    if (g.shape != (n, k, 9) or inputs['truth_cij'].shape != (n, 9)
            or base.shape != (n,) or inputs['category'].shape != (n,)
            or len(inputs['source_ids']) != n or len(np.unique(inputs['source_ids'])) != n
            or not all(np.isfinite(v).all() for v in (g, logits, base, inputs['truth_cij'], inputs['category']))):
        raise ValueError('Invalid finite aligned component panel')
    target = np.average(inputs['truth_cij'], weights=base, axis=0)
    nums, dens, rows = [], [], []
    for name, score in [('unweighted', np.zeros_like(logits)), ('raw', logits),
                        ('cap30', np.minimum(logits, np.log(cfg['cap'])))]:
        w, _ = candidate_weights(base, score)
        num, den = np.einsum('nk,nkd->nd', w, g), w.sum(1)
        cij = num.sum(0)
        nums.append(num); dens.append(den)
        rows.append(dict(arm=name, cij=cij.tolist(), error=float(np.linalg.norm(cij-target)),
                         event_ess=float(1/np.square(den).sum())))
        if progress: progress('aggregated_'+name)
    estimates, targets = paired_bootstrap(inputs, nums, dens, cfg['bootstrap'], cfg['bootstrap_seed'])
    boot_residual = estimates-targets[:, None]
    boot_delta = np.abs(boot_residual[:, 1:])-np.abs(boot_residual[:, :1])
    residual = np.array([r['cij'] for r in rows])-target
    point = np.abs(residual[1:])-np.abs(residual[:1])
    bands, radius = simultaneous_intervals(point, boot_delta)
    report = dict(truth=target.tolist(), arms=rows, components=[], attribution={},
        family=dict(entries=18, definition='raw-minus-unweighted and cap30-minus-unweighted, all nine absolute errors',
                    method='paired event bootstrap centered max-absolute-deviation, approximate simultaneous95%',
                    radius=radius), tradeoffs={}, categories={})
    saved = dict(source_ids=inputs['source_ids'])
    for a, name in enumerate(('raw', 'cap30'), start=1):
        acc = event_accounting(nums[a], dens[a], nums[0], dens[0], target)
        if not np.allclose(acc['absolute_error_contribution'].sum(0), point[a-1], atol=1e-10, rtol=1e-8):
            raise ValueError('Event accounting does not reconstruct endpoint')
        for key, value in acc.items(): saved[name+'_'+key] = value
        report['attribution'][name] = [component_attribution(inputs, acc, j, cfg) for j in range(9)]
        c = acc['absolute_error_contribution']
        # Row j: share of total helpful contribution to j arising from events
        # which simultaneously harm column l. Not a causal conflict probability.
        helpful = np.maximum(-c, 0)
        overlap = np.einsum('nj,nl->jl', helpful, (c > 0).astype(float))
        total = helpful.sum(0)
        report['tradeoffs'][name] = [[None if total[j] == 0 else float(overlap[j,l]/total[j])
                                     for l in range(9)] for j in range(9)]
        groups = []
        for category in np.unique(inputs['category']):
            mask = inputs['category'] == category
            groups.append(dict(category=int(category), events=int(mask.sum()),
                base_mass=float(dens[0][mask].sum()), weighted_mass=float(dens[a][mask].sum()),
                absolute_error_contribution=c[mask].sum(0).tolist(),
                within=acc['within_absolute_error'][mask].sum(0).tolist(),
                between=acc['between_absolute_error'][mask].sum(0).tolist()))
        report['categories'][name] = groups
        for j, axis in enumerate(AXES):
            lo, hi = bands[:, a-1, j]
            report['components'].append(dict(arm=name, component=axis, truth=float(target[j]),
                unweighted=float(rows[0]['cij'][j]), estimate=float(rows[a]['cij'][j]),
                absolute_error_change=float(point[a-1,j]),
                pointwise_ci95=np.quantile(boot_delta[:,a-1,j], [.025,.975]).tolist(),
                simultaneous_ci95=[float(lo),float(hi)],
                status='improved' if hi < 0 else 'worsened' if lo > 0 else 'unresolved'))
    report['limitations'] = [
        'Observed fixed-model accounting, not causal attribution or proof of ratio miscalibration.',
        'Ranking and group comparisons are post hoc on previously inspected conditions; no automatic removals or tuning.',
        'Approximate simultaneous bands cover the declared18 component contrasts only, not post-hoc group/rank searches.',
        'Bootstrap keeps each condition and all64 candidates/truth together; excludes unseen tails and model-fit uncertainty.',
        'Absolute-error bootstrap can be nonregular near zero residual; intervals are exploratory, not exact guarantees.',
        'Within/between terms use unweighted per-condition means; this is one exact algebraic decomposition, not unique causal effects.',
        'The inherited Cij/analyzing-power convention is unchanged, not independently recertified.']
    return report, saved
