"""Held-out conditional moment checks; no per-event normalization at K=1."""
import numpy as np


def fit_pt_edges(pt, split):
    values = np.asarray(pt)[np.asarray(split) == 0]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Require finite fitting pT values')
    return np.quantile(values, [.25, .5, .75]).tolist()


def conditional_report(truth, generated, weight, logits, category, pt, edges):
    truth, generated = np.asarray(truth, float), np.asarray(generated, float)
    weight, logits = np.asarray(weight, float), np.asarray(logits, float)
    category, pt = np.asarray(category), np.asarray(pt)
    n = len(weight)
    if (truth.shape != generated.shape or truth.shape != (n,15) or
        any(x.shape != (n,) for x in (logits,category,pt)) or
        not all(np.isfinite(x).all() for x in (truth,generated,weight,logits,pt)) or
        (weight < 0).any() or weight.sum() <= 0):
        raise ValueError('Invalid conditional diagnostic arrays')
    logw = np.full(n, -np.inf)
    np.log(weight, out=logw, where=weight>0)
    total = np.logaddexp.reduce(logw)
    total_ratio = np.logaddexp.reduce(logw+logits)
    bw, rw = weight/weight.sum(), np.exp(logw+logits-total_ratio)
    bins = np.searchsorted(np.asarray(edges), pt, side='right')
    groups = [('all', np.ones(n,dtype=bool))]
    cats = sorted(np.unique(category))
    groups += [(f'category_{cat}',category==cat) for cat in cats]
    groups += [(f'pt_{i}',bins==i) for i in range(4)]
    groups += [(f'category_{cat}_pt_{i}',(category==cat)&(bins==i)) for cat in cats for i in range(4)]
    rows = []
    for name, take in groups:
        if not take.any() or weight[take].sum() <= 0:
            continue
        w, r = bw[take], rw[take]
        bm, rm = w.sum(), r.sum()
        if rm <= 0:
            raise FloatingPointError('Underflowed group weight mass')
        target = w @ truth[take]/bm
        raw = w @ generated[take]/bm
        estimate = r @ generated[take]/rm
        rows.append(dict(group=name,events=int(take.sum()),base_mass=float(bm),
            weighted_mass=float(rm),mass_drift=float(rm-bm),
            log_mean_ratio=float(np.logaddexp.reduce((logw+logits)[take])-
                                 np.logaddexp.reduce(logw[take])),
            ess=float(rm**2/(r@r)),
            tau_error_unweighted=float(np.linalg.norm(raw-target)),
            tau_error_reweighted=float(np.linalg.norm(estimate-target))))
    cat_rows = [r for r in rows if r['group'].startswith('category_') and '_pt_' not in r['group']]
    joint_rows = [r for r in rows if '_pt_' in r['group']]
    summary = dict(log_mean_ratio=float(total_ratio-total),
        category_mass_tv=float(.5*sum(abs(r['mass_drift']) for r in cat_rows)),
        group_log_mean_ratio_rms=float(np.sqrt(sum(r['base_mass']*r['log_mean_ratio']**2 for r in joint_rows))),
        group_tau_error_unweighted=float(sum(r['base_mass']*r['tau_error_unweighted'] for r in joint_rows)),
        group_tau_error_reweighted=float(sum(r['base_mass']*r['tau_error_reweighted'] for r in joint_rows)))
    return dict(summary=summary,groups=rows,pt_edges=list(edges),
        scope='Coarse category x fit-only visible-pT strata; not pointwise conditional closure. '
              'Raw mean ratio before normalization; group moments self-normalized for diagnostics only.')
