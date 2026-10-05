"""Fixed-score sensitivity analysis; not a learned or unbiased ratio correction."""
import numpy as np

from scripts.tau_tail_attribution import AXES, candidate_weights, paired_bootstrap
from scripts.tau_cij_components import simultaneous_intervals


def logmeanexp(s):
    s = np.asarray(s, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] == 0 or not np.isfinite(s).all():
        raise ValueError('Expected finite nonempty candidate logits')
    top = s.max(1)
    return top + np.log(np.exp(s-top[:, None]).mean(1))


def normalized_scores(s):
    s = np.asarray(s, dtype=np.float64)
    k = s.shape[1]
    if k < 2 or k % 2:
        raise ValueError('Disjoint normalization requires positive even K')
    h = k//2
    za, zb = logmeanexp(s[:, :h]), logmeanexp(s[:, h:])
    same = s-logmeanexp(s)[:, None]
    cross = np.concatenate((s[:, :h]-zb[:, None], s[:, h:]-za[:, None]), axis=1)
    if not np.isfinite(same).all() or not np.isfinite(cross).all():
        raise ValueError('Normalization overflow')
    return same, cross, za, zb


def denominator_diagnostics(s, base, za, zb):
    p = base/base.sum()
    positive = base > 0
    da, db = za-np.dot(p, za), zb-np.dot(p, zb)
    variance = np.dot(p, da*da)*np.dot(p, db*db)
    # Softmax only within each condition; ESS does not depend on its total scale.
    local = np.exp(s-s.max(1)[:, None]); local /= local.sum(1)[:, None]
    ess = 1/np.square(local).sum(1)
    return dict(denominator_candidates=len(s[0])//2,
        log_z_half_correlation=None if variance <= 1e-24 else float(np.dot(p, da*db)/np.sqrt(variance)),
        log_z_half_rms_difference=float(np.sqrt(np.dot(p, (za-zb)**2))),
        log_z_half_difference_quantiles=np.quantile((za-zb)[positive], [.01,.5,.99]).tolist(),
        log_z_quantiles=np.quantile(logmeanexp(s)[positive], [.01,.5,.99]).tolist(),
        within_condition_ess_quantiles=np.quantile(ess[positive], [.01,.5,.99]).tolist(),
        base_mass_with_ess_below_2=float(p[ess < 2].sum()),
        quantile_scope='Unweighted quantiles over positive-base-weight conditions')


def analyze(inputs, data, cfg, progress=None):
    s, g = np.asarray(data['logits'], float), np.asarray(data['cij'])
    base, truth = np.asarray(inputs['weight'], float), np.asarray(inputs['truth_cij'], float)
    n, total_k = s.shape
    if (g.shape != (n,total_k,9) or truth.shape != (n,9) or base.shape != (n,)
            or np.asarray(inputs['category']).shape != (n,)
            or len(inputs['source_ids']) != n or len(np.unique(inputs['source_ids'])) != n
            or not all(np.isfinite(v).all() for v in (s,g,base,truth,inputs['category']))
            or (base < 0).any() or base.sum() <= 0):
        raise ValueError('Invalid aligned finite panel')
    prefixes = cfg['prefixes']
    if (not prefixes or len(set(prefixes)) != len(prefixes)
            or any(k < 2 or k % 2 or k > total_k for k in prefixes)):
        raise ValueError('Invalid even candidate prefixes')
    # Preserve the prior report's target and cap rounding for exact replay.
    # Newly introduced log-normalization arithmetic is float64.
    target = np.average(inputs['truth_cij'], weights=inputs['weight'], axis=0)
    p = base/base.sum()
    report = dict(truth=target.tolist(), axes=list(AXES), prefixes={}, limitations=[
        'Same-pool event mass closure is imposed, not evidence of a correct density ratio.',
        'Disjoint denominators use other candidates, NOT independent conditions or a newly fitted classifier.',
        'Inverse estimated Z has finite-K bias; the symmetric disjoint estimator can amplify noisy denominators.',
        'With equal half sizes, disjoint event mass is proportional to cosh(log Z_A - log Z_B).',
        'Combining the disjoint halves changes their relative mixture; inspect each direction against its matched raw half.',
        'Conditional normalization cannot repair wrong relative scores within a condition or missing proposal support.',
        'Bootstrap resamples whole conditions with paired truth and all candidates; fixed fits/noise, no unseen-tail guarantee.',
        'All conditions were previously inspected; this is sensitivity analysis, not fresh generalization validation.',
        'Cij and signed analyzing-power convention are inherited unchanged, not independently recertified.',
        'Component bands are approximate simultaneous over both normalization arms and all K prefixes; norm CIs are pointwise.'
    ])
    family_points, family_draws = [], []
    arrays = dict(source_ids=inputs['source_ids'])
    for k in prefixes:
        sk, gg = s[:, :k], g[:, :k]
        same, cross, za, zb = normalized_scores(sk)
        arrays[f'K{k}_log_z_a'], arrays[f'K{k}_log_z_b'] = za, zb
        rows, nums, dens = [], [], []
        h=k//2
        arms=[('unweighted',np.zeros_like(sk),gg), ('raw',sk,gg),
              ('cap30',np.minimum(data['logits'][:,:k],np.log(cfg['cap'])),gg),
              ('event_normalized',same,gg), ('disjoint_normalized',cross,gg),
              ('raw_A',sk[:,:h],gg[:,:h]), ('raw_B',sk[:,h:],gg[:,h:]),
              ('A_using_Z_B',cross[:,:h],gg[:,:h]), ('B_using_Z_A',cross[:,h:],gg[:,h:])]
        for name, score, candidate_g in arms:
            w, lm = candidate_weights(base, score)
            num, den = np.einsum('nk,nkd->nd',w,candidate_g), w.sum(1)
            cij = num.sum(0)
            category_tv = .5*sum(abs(den[inputs['category']==c].sum()-p[inputs['category']==c].sum())
                                  for c in np.unique(inputs['category']))
            rows.append(dict(arm=name,cij=cij.tolist(),error=float(np.linalg.norm(cij-target)),
                event_ess=float(1/np.square(den).sum()),candidate_ess=float(1/np.square(w).sum()),
                max_candidate_mass=float(w.max()),event_mass_tv=float(.5*np.abs(den-p).sum()),
                category_mass_tv=float(category_tv),log_mean_transformed_ratio=lm,
                zero_weight_candidates_positive_base=int(np.count_nonzero(w[base > 0] == 0)),
                mass_closure_imposed=name in ('unweighted','event_normalized')))
            nums.append(num); dens.append(den)
        if progress: progress(f'K{k}: paired event bootstrap')
        estimates, targets = paired_bootstrap(inputs,nums,dens,cfg['bootstrap'],cfg['bootstrap_seed'])
        residual = np.array([r['cij'] for r in rows])-target
        boot_residual = estimates-targets[:,None]
        errors = np.linalg.norm(boot_residual,axis=-1)
        for i,row in enumerate(rows):
            row['matrix_ci95'] = np.quantile(estimates[:,i],[.025,.975],axis=0).tolist()
            row['comparisons'] = {}
            for j in ((0,1,2,5) if i==7 else (0,1,2,6) if i==8 else (0,1,2)):
                row['comparisons'][rows[j]['arm']] = dict(error_change=row['error']-rows[j]['error'],
                    error_change_ci95=np.quantile(errors[:,i]-errors[:,j],[.025,.975]).tolist(),
                    component_absolute_error_change=(np.abs(residual[i])-np.abs(residual[j])).tolist())
        point = np.abs(residual[3:5])-np.abs(residual[1])
        draws = np.abs(boot_residual[:,3:5])-np.abs(boot_residual[:,1,None])
        family_points.append(point); family_draws.append(draws)
        report['prefixes'][str(k)] = dict(arms=rows,denominator=denominator_diagnostics(sk,base,za,zb))
    points, draws = np.stack(family_points), np.stack(family_draws,axis=1)
    bands,radius = simultaneous_intervals(points,draws)
    report['component_family'] = dict(entries=int(points.size),radius=radius,
        contrast='Absolute error minus raw, both normalization arms, all prefixes and nine components')
    for ki,k in enumerate(prefixes):
        components=[]
        for ai,name in enumerate(('event_normalized','disjoint_normalized')):
            for j,axis in enumerate(AXES):
                lo,hi = bands[:,ki,ai,j]
                components.append(dict(arm=name,component=axis,absolute_error_change=float(points[ki,ai,j]),
                    pointwise_ci95=np.quantile(draws[:,ki,ai,j],[.025,.975]).tolist(),
                    simultaneous_ci95=[float(lo),float(hi)],
                    status='improved' if hi < 0 else 'worsened' if lo > 0 else 'unresolved'))
        report['prefixes'][str(k)]['components_vs_raw'] = components
    return report,arrays
