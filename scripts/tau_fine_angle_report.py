"""Predeclared fine-angle closure, independent joint checks and ratio diagnostics."""
import itertools
import json
from pathlib import Path
import numpy as np
from scripts.tau_tail_attribution import candidate_weights
from scripts.tau_fine_angle_residual import PRIMARY

NAMES = ('a_theta', 'a_phi', 'b_theta', 'b_phi')


def coordinates(values):
    result = np.asarray(values, dtype=float).copy()
    result[..., 1] = np.arctan2(np.sin(result[..., 1]), np.cos(result[..., 1]))
    return result.reshape(*result.shape[:-2], 4)


def core_comparison(truth, generated, weight, scores, bootstrap, seed):
    """Truth and all K candidates resampled together by conditioning event."""
    t, g, w = coordinates(truth), coordinates(generated), np.asarray(weight, float)
    if (bootstrap < 20 or t.shape != (len(w), 4) or g.shape[0] != len(w)
            or g.ndim != 3 or not np.isfinite(t).all() or not np.isfinite(g).all()):
        raise ValueError('Invalid fine-angle comparison')
    weights = {a: candidate_weights(w, s)[0] for a, s in
               {'unweighted': np.zeros(g.shape[:2]), **scores}.items()}
    # Two phi indicators, independent of any model-specific histogram edge.
    ti, gi = np.abs(t[:, (1, 3)]) < .01, np.abs(g[:, :, (1, 3)]) < .01
    target = np.average(ti, weights=w, axis=0)
    columns = [np.c_[ti*w[:, None], w]]
    arms = {}
    for name, q in weights.items():
        num, den = np.einsum('nk,nkd->nd', q, gi), q.sum(1)
        mass = num.sum(0)/den.sum()
        arms[name] = dict(core_mass=mass.tolist(), signed_bias=(mass-target).tolist(),
                          error=float(np.abs(mass-target).mean()))
        columns.append(np.c_[num, den])
    values = np.concatenate(columns, axis=1)
    names = list(weights)
    pairs = [('residual', b) for b in ('general', 'continued', 'unweighted')]
    draws = np.empty((bootstrap, len(pairs)))
    rng = np.random.default_rng(seed)
    for j in range(bootstrap):
        counts = np.bincount(rng.integers(0, len(w), len(w)), minlength=len(w))
        sums = np.einsum('i,ij->j', counts, values, optimize=False).reshape(-1, 3)
        if (sums[:, 2] <= 0).any():
            raise ValueError('Empty bootstrap denominator')
        means = sums[:, :2]/sums[:, 2, None]
        error = dict(zip(names, np.abs(means[1:]-means[0]).mean(1)))
        draws[j] = [error[a]-error[b] for a, b in pairs]
    comparisons = {a+'_minus_'+b: dict(delta=arms[a]['error']-arms[b]['error'],
                       ci95=np.quantile(draws[:, j], [.025, .975]).tolist()) for j, (a, b) in enumerate(pairs)}
    return dict(primary=PRIMARY, core_radius_rad=.01, truth_core_mass=target.tolist(), arms=arms,
        comparisons=comparisons, primary_pass=all(v['ci95'][1] < 0 for v in comparisons.values()),
        uncertainty='Paired whole-event percentile95%, truth and all K resampled; conditional on fitted heads and saved candidates. Exploratory reused test panel; no fit/systematic uncertainty.'), weights


def probability_distance(p, q):
    p, q = np.asarray(p), np.asarray(q)
    mean = (p+q)/2
    def kl(x):
        keep = x > 0
        return float(np.sum(x[keep]*np.log(x[keep]/mean[keep])))
    return dict(tv=float(np.abs(p-q).sum()/2), jsd=(kl(p)+kl(q))/2)


def distribution_checks(truth, generated, weight, weights):
    t, g = coordinates(truth), coordinates(generated)
    w = np.asarray(weight, float)/np.sum(weight)
    edges = np.linspace(-.2, .2, 81)
    padded = np.r_[-np.inf, edges, np.inf]
    records = dict(marginals={}, joints={})
    for j, name in enumerate(NAMES):
        p = np.histogram(t[:, j], padded, weights=w)[0]
        rows = {}
        for a, q in weights.items():
            h = np.histogram(g[:, :, j].ravel(), padded, weights=q.ravel())[0]
            rows[a] = dict(probability=h.tolist(), outside=float(h[0]+h[-1]), **probability_distance(h,p))
        records['marginals'][name] = dict(edges=edges.tolist(), truth=p.tolist(), arms=rows)
    # Six 2D projections: both within-leg joints and all cross-leg coordinate pairs.
    # These are supporting projections, not a proof of complete 4D closure.
    for i, j in itertools.combinations(range(4), 2):
        p = np.histogram2d(t[:, i], t[:, j], (padded,padded), weights=w)[0]
        rows = {}
        for a, q in weights.items():
            h = np.histogram2d(g[:,:,i].ravel(), g[:,:,j].ravel(), (padded,padded), weights=q.ravel())[0]
            rows[a] = dict(probability=h.tolist(), **probability_distance(h,p),
                           outside=float(1-h[1:-1,1:-1].sum()))
        records['joints'][NAMES[i]+'__'+NAMES[j]] = dict(edges=edges.tolist(), truth=p.tolist(), arms=rows)
    differences = {name: {b: row['arms']['residual']['tv']-row['arms'][b]['tv']
                          for b in ('general','continued','unweighted')}
                   for name,row in records['joints'].items()}
    guard = dict(point_estimate_only=True, criterion='No increase in any of six joint TVs versus General, continued or unweighted',
                 residual_minus_control_tv=differences,
                 passed=all(v <= 0 for row in differences.values() for v in row.values()),
                 limitation='No joint-TV uncertainty interval; not full 4D or spin closure, not a deployment gate.')
    return records, guard


def calibration(positive, negative, weight):
    w = np.asarray(weight, float)
    logit = np.r_[positive, negative]
    if not np.isfinite(logit).all() or w.sum() <= 0:
        raise ValueError('Invalid calibration inputs')
    probability = 1/(1+np.exp(-logit))
    target = np.r_[np.ones(len(w)), np.zeros(len(w))]
    mass = np.r_[w,w]/(2*w.sum())
    bin_ids = np.minimum((probability*15).astype(int),14)
    rows, ece = [], 0.
    for i in range(15):
        take = bin_ids == i
        total = mass[take].sum()
        if total <= 0:
            continue
        pred, observed = [float(np.average(v[take], weights=mass[take])) for v in (probability,target)]
        rows.append(dict(bin=i, mass=float(total), predicted=pred, observed=observed))
        ece += total*abs(pred-observed)
    return dict(ece=float(ece), brier=float(np.sum(mass*(probability-target)**2)), reliability=rows,
                negative_mean_ratio=float(np.average(np.exp(negative),weights=w)),
                scope='Diagnostic only; equal truth/generated priors; no fitted temperature or score change.')


def write_fine_report(output, panel, generated, scores, positives, bootstrap, seed, process_labels):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    out = Path(output)
    core, weights = core_comparison(panel['truth_deltas'], generated, panel['event_weight'], scores, bootstrap, seed)
    distributions, guard = distribution_checks(panel['truth_deltas'], generated, panel['event_weight'], weights)
    calibration_rows = {a: calibration(positives[a], scores[a][:,0], panel['event_weight']) for a in scores}
    groups = {}
    w = np.asarray(panel['event_weight'], float)
    t, g = coordinates(panel['truth_deltas']), coordinates(generated)
    for grouping, labels in (('process', process_labels), ('decay_category',panel['category'])):
        groups[grouping] = {}
        for label in np.unique(labels):
            take = labels == label
            if w[take].sum() <= 0:
                continue
            truth_mass = float(w[take].sum()/w.sum())
            target = np.average(np.abs(t[take][:,(1,3)]) < .01, weights=w[take], axis=0)
            rows = {}
            for a,q in weights.items():
                mass = q[take].sum()
                event_mass = q[take].sum(1)/mass
                core_mass = np.einsum('nk,nkd->d',q[take],np.abs(g[take][:,:,(1,3)]) < .01)/mass
                rows[a] = dict(mass=float(mass), mass_minus_truth=float(mass-truth_mass),
                    conditional_core_mass=core_mass.tolist(), conditional_core_error=float(np.abs(core_mass-target).mean()),
                    event_ess=float(1/np.square(event_mass).sum()))
                if a in scores:
                    rows[a]['calibration'] = calibration(positives[a][take],scores[a][take,0],w[take])
            groups[grouping][str(label)] = dict(events=int(take.sum()), truth_mass=truth_mass,
                truth_core_mass=target.tolist(), arms=rows)
    report = dict(**core, joint_guard=guard, calibration=calibration_rows, groups=groups,
                  policy_updates=0, candidate_generation=False, bin_width_rad=.005)
    (out/'fine_angle_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'fine_angle_histograms.json').write_text(json.dumps(distributions,allow_nan=False)+'\n')
    plots = []
    fig = Figure(figsize=(12,8),layout='constrained'); FigureCanvasAgg(fig)
    for ax,(name,row) in zip(fig.subplots(2,2).ravel(), distributions['marginals'].items()):
        e = np.array(row['edges'])
        ax.stairs(np.array(row['truth'])[1:-1]/np.diff(e),e,label='truth',color='black',linestyle='--')
        for a,r in row['arms'].items():
            ax.stairs(np.array(r['probability'])[1:-1]/np.diff(e),e,label=a)
        ax.set(title=name,xlabel='Angular offset [rad]',ylabel='Density [1/rad]'); ax.legend(fontsize=8)
    fig.suptitle('Fixed step500 K8 | global ratio weights | fine-angle closure')
    path=out/'fine_angle_marginals.png';fig.savefig(path,dpi=140);plots.append(path)
    # Plot every joint with one symmetric difference scale per joint across arms.
    fig=Figure(figsize=(18,12),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(4,6)
    for col,(name,row) in enumerate(distributions['joints'].items()):
        e=np.array(row['edges']);target=np.array(row['truth'])[1:-1,1:-1]
        ds={a:np.array(r['probability'])[1:-1,1:-1]-target for a,r in row['arms'].items()}
        scale=max(np.abs(d).max() for d in ds.values()) or 1.
        for ax,(a,d) in zip(axes[:,col],ds.items()):
            im=ax.pcolormesh(e,e,d.T,cmap='RdBu_r',vmin=-scale,vmax=scale,shading='flat')
            ax.set(title=a+'\n'+name,xlabel='First offset [rad]',ylabel='Second offset [rad]')
            fig.colorbar(im,ax=ax,shrink=.65)
    fig.suptitle('Prediction - truth probability/bin | tails included in normalization and TV')
    path=out/'fine_angle_joint_differences.png';fig.savefig(path,dpi=120);plots.append(path)
    return report, plots
