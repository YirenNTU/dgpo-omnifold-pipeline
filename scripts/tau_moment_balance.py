"""Split-held-out, physics-targeted calibration of a frozen bounded ratio.

This is regularized moment balancing, not exact entropy balancing, a new
classifier, or a certified conditional density ratio. No test truth enters fit
or selection. The nine calibration coefficients are functions of a candidate's
own saved Cij contributions, never its paired truth's contributions.
"""
import hashlib

import numpy as np
from scipy.optimize import minimize

from scripts.tau_tail_attribution import AXES, candidate_weights, paired_bootstrap
from scripts.tau_cij_components import simultaneous_intervals

DIAG = np.array([0, 4, 8])
OFFDIAG = np.array([1, 2, 3, 5, 6, 7])


def event_split(ids, seed, fractions):
    """Row-order-invariant event split; all K candidates inherit their event."""
    ids = np.asarray(ids).astype(str)
    f = np.asarray(fractions, float)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or f.shape != (3,)
            or not np.isfinite(f).all() or (f <= 0).any() or not np.isclose(f.sum(), 1)):
        raise ValueError('Unique IDs and three positive fractions summing to one required')
    u = np.array([int.from_bytes(hashlib.blake2b(
        f'moment-balance:{seed}:{x}'.encode(), digest_size=8).digest(), 'big') / 2**64 for x in ids])
    return np.searchsorted(np.cumsum(f)[:-1], u, side='right')


def validate(inputs, generated, logits, cfg):
    n, k = np.shape(logits)
    base = np.asarray(inputs['weight'], float)
    if (generated.shape != (n, k, 9) or np.shape(inputs['truth_cij']) != (n, 9)
            or base.shape != (n,) or not np.isfinite(base).all() or (base <= 0).any()
            or not np.isfinite(generated).all() or not np.isfinite(inputs['truth_cij']).all()
            or not np.isfinite(logits).all() or np.max(logits) > np.log(cfg['cap']) + 1e-6
            or np.shape(inputs['category']) != (n,) or np.shape(inputs['visible_pt_sum']) != (n,)
            or not np.isfinite(inputs['visible_pt_sum']).all()):
        raise ValueError('Invalid positive-weight, finite, bounded paired panel')
    edges = np.asarray(cfg['condition_pt_edges'], float)
    if edges.ndim != 1 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError('Invalid fixed condition pT edges')


def groups(inputs, edges):
    # Fixed upstream fit-only pT boundaries; never optimize boundaries with Cij.
    pt = np.searchsorted(edges, inputs['visible_pt_sum'], side='right')
    keys = np.stack([inputs['category'], pt], axis=1)
    return np.unique(keys, axis=0, return_inverse=True)


def subset(inputs, indices):
    return {key: np.asarray(value)[indices] for key, value in inputs.items()}


def candidate_features(generated, base):
    weights = np.asarray(base, float) / np.sum(base)
    center = np.einsum('n,nkd->d', weights, generated) / generated.shape[1]
    variance = np.einsum('n,nkd->d', weights, np.square(generated-center)) / generated.shape[1]
    scale = np.sqrt(np.maximum(variance, 1e-12))
    return center, scale


def transform(logits, generated, model):
    """Deployable frozen correction: this API intentionally has no truth input."""
    phi = (np.asarray(generated, float)-model['center']) / model['scale']
    limit = float(model['max_log_correction'])
    delta = limit*np.tanh(np.einsum('nkd,d->nk', phi, model['theta']) / limit)
    return np.minimum(np.asarray(logits, float)+delta, np.log(model['cap']))


class CalibrationObjective:
    """Analytic derivative of moment error + weight KL + condition-mass penalty.

    L = strength/2 * ||C(w)-C_truth||^2 + KL(w||w0)
        + mass_strength/2 * sum_g (mass_g-base_g)^2/base_g
        + ridge/2 * ||theta||^2.
    w is globally normalized base/K*exp(corrected log ratio). The cap is on
    the unnormalized ratio, exactly as in the fixed baseline; it is not a cap
    of 30 on the globally self-normalized weights.
    """
    def __init__(self, inputs, generated, logits, cfg, strength):
        self.g = np.asarray(generated, float)
        self.s = np.asarray(logits, float)
        self.base = np.asarray(inputs['weight'], float)
        self.target = np.average(inputs['truth_cij'], weights=self.base, axis=0)
        self.center, self.scale = candidate_features(self.g, self.base)
        self.phi = ((self.g-self.center)/self.scale).reshape(-1, 9)
        self.gf = self.g.reshape(-1, 9)
        _, self.logmean0 = candidate_weights(self.base, self.s)
        _, self.group = groups(inputs, cfg['condition_pt_edges'])
        self.bg = np.bincount(self.group, weights=self.base/self.base.sum())
        self.cfg, self.strength = cfg, float(strength)
        self.calls = 0

    def __call__(self, theta):
        self.calls += 1
        limit = np.log(self.cfg['max_correction_factor'])
        t = np.tanh(np.einsum('nd,d->n', self.phi, theta)/limit).reshape(self.s.shape)
        raw = self.s + limit*t
        corrected = np.minimum(raw, np.log(self.cfg['cap']))
        w, logmean = candidate_weights(self.base, corrected)
        c = np.einsum('n,nd->d', w.ravel(), self.gf)
        residual = c-self.target
        log_relative = corrected-self.s-(logmean-self.logmean0)
        kl = float(np.sum(w*log_relative))
        mass = np.bincount(self.group, weights=w.sum(1), minlength=len(self.bg))
        imbalance = (mass-self.bg)/self.bg
        mass_loss = .5*self.cfg['mass_strength']*np.dot(mass-self.bg, imbalance)
        ridge = self.cfg['coefficient_ridge']
        loss = .5*self.strength*np.dot(residual, residual) + kl + mass_loss + .5*ridge*np.dot(theta, theta)
        sensitivity = (self.strength*np.einsum('nd,d->n', self.gf, residual)).reshape(w.shape) + log_relative
        sensitivity += self.cfg['mass_strength']*imbalance[self.group, None]
        dlog = w*(sensitivity-np.sum(w*sensitivity))
        dlog *= (1-t*t)*(raw < np.log(self.cfg['cap']))
        grad = np.einsum('nd,n->d', self.phi, dlog.ravel()) + ridge*theta
        if not np.isfinite(loss) or not np.isfinite(grad).all():
            raise ValueError('Nonfinite calibration objective')
        return float(loss), grad


def fit(inputs, generated, logits, cfg, strength, progress=None):
    objective = CalibrationObjective(inputs, generated, logits, cfg, strength)
    history = []

    def callback(theta):
        loss, grad = objective(theta)
        row = dict(iteration=len(history)+1, strength=float(strength), loss=loss,
                   gradient_norm=float(np.linalg.norm(grad)))
        history.append(row)
        if progress:
            progress('calibration_fit', row)

    result = minimize(objective, np.zeros(9), jac=True, method='L-BFGS-B',
        bounds=[(-cfg['coefficient_limit'], cfg['coefficient_limit'])]*9,
        callback=callback, options=dict(maxiter=cfg['max_iterations'], ftol=1e-12, gtol=1e-7, maxls=40))
    return dict(theta=result.x.tolist(), center=objective.center.tolist(), scale=objective.scale.tolist(),
        max_log_correction=float(np.log(cfg['max_correction_factor'])), cap=cfg['cap'], strength=float(strength),
        converged=bool(result.success), optimizer_message=str(result.message), iterations=int(result.nit),
        function_evaluations=objective.calls, final_loss=float(result.fun), history=history,
        feature_order=list(AXES), feature_definition='Saved candidate 9*a_i*b_j/(kappa_a*kappa_b)',
        fit_truth_C=objective.target.tolist())


def metrics(inputs, generated, logits, edges):
    base = np.asarray(inputs['weight'], float)
    w, logmean = candidate_weights(base, logits)
    mass = w.sum(1)
    num = np.einsum('nk,nkd->nd', w, generated)
    c = num.sum(0)
    truth = np.average(inputs['truth_cij'], weights=base, axis=0)
    residual = c-truth
    keys, gi = groups(inputs, edges)
    bg = np.bincount(gi, weights=base/base.sum())
    gm = np.bincount(gi, weights=mass)
    group_rows = []
    for j, (category, pt_bin) in enumerate(keys):
        take = gi == j
        gt = np.average(inputs['truth_cij'][take], weights=base[take], axis=0)
        gc = num[take].sum(0)/gm[j]
        group_rows.append(dict(category=int(category), pt_bin=int(pt_bin), events=int(take.sum()),
            base_mass=float(bg[j]), weighted_mass=float(gm[j]), C=gc.tolist(), truth_C=gt.tolist(),
            error=float(np.linalg.norm(gc-gt))))
    return dict(C=c.reshape(3, 3).tolist(), truth_C=truth.reshape(3, 3).tolist(),
        error=float(np.linalg.norm(residual)), diagonal_error=float(np.linalg.norm(residual[DIAG])),
        offdiagonal_error=float(np.linalg.norm(residual[OFFDIAG])), absolute_component_error=np.abs(residual).tolist(),
        candidate_ess_fraction=float(1/np.square(w).sum()/w.size),
        event_ess_fraction=float(1/np.square(mass).sum()/len(mass)),
        max_candidate_mass=float(w.max()), log_mean_ratio=logmean,
        raw_ratio_max=float(np.exp(np.max(logits))),
        near_cap_fraction=float(np.mean(np.asarray(logits) >= np.log(27))),
        group_mass_tv=float(.5*np.abs(gm-bg).sum()), groups=group_rows), num, mass


def guardrails(row, baseline, cfg):
    guards = cfg['guardrails']
    return dict(
        candidate_ess=row['candidate_ess_fraction'] >= guards['ess_retention']*baseline['candidate_ess_fraction'],
        event_ess=row['event_ess_fraction'] >= guards['ess_retention']*baseline['event_ess_fraction'],
        condition_mass=row['group_mass_tv'] <= baseline['group_mass_tv']+guards['mass_tv_increase'],
        diagonal=row['diagonal_error'] <= baseline['diagonal_error']+guards['diagonal_error_increase'],
        total=row['error'] <= baseline['error']+guards['total_error_increase'])


def select(models, validation, baseline, cfg):
    """Choose by selection-split off-diagonal error, never by final-test truth."""
    eligible = []
    checks = {}
    for name, model in models.items():
        checks[name] = dict(converged=model['converged'], **guardrails(validation[name], baseline, cfg))
        checks[name]['offdiagonal_improves'] = validation[name]['offdiagonal_error'] < baseline['offdiagonal_error']
        if all(checks[name].values()):
            eligible.append(name)
    choice = min(eligible, key=lambda name: (validation[name]['offdiagonal_error'], models[name]['strength'])) if eligible else 'baseline'
    return dict(selected=choice, candidates=checks, used_test=False,
        rule='Converged; all fixed guardrails; selection offdiagonal error improves. Lowest such error; otherwise baseline.')


def paired_report(inputs, generated, baseline_logits, corrected_logits, cfg):
    arm_logits = dict(unweighted=np.zeros_like(baseline_logits), baseline=baseline_logits, calibrated=corrected_logits)
    rows, nums, dens = {}, [], []
    for name, scores in arm_logits.items():
        row, num, den = metrics(inputs, generated, scores, cfg['condition_pt_edges'])
        rows[name] = row
        nums.append(num); dens.append(den)
    draws, targets = paired_bootstrap(inputs, nums, dens, cfg['bootstrap'], cfg['bootstrap_seed'])
    residual = draws-targets[:, None, :]
    comparisons = {}
    for name, axes in [('offdiagonal', OFFDIAG), ('diagonal', DIAG), ('total', np.arange(9))]:
        key = 'error' if name == 'total' else name+'_error'
        values = np.linalg.norm(residual[..., axes], axis=-1)
        delta = values[:, 2]-values[:, 1]
        interval = np.quantile(delta, [.025, .975]).tolist()
        comparisons[name] = dict(change=rows['calibrated'][key]-rows['baseline'][key], ci95=interval,
            status='improved' if interval[1] < 0 else 'worsened' if interval[0] > 0 else 'unresolved')
    cp = np.array(rows['calibrated']['absolute_component_error'])-rows['baseline']['absolute_component_error']
    cd = np.abs(residual[:, 2])-np.abs(residual[:, 1])
    bands, _ = simultaneous_intervals(cp, cd)
    comparisons['components'] = [dict(component=name, change=float(cp[j]),
        pointwise_ci95=np.quantile(cd[:, j], [.025, .975]).tolist(), simultaneous_ci95=bands[:, j].tolist())
        for j, name in enumerate(AXES)]
    guards = guardrails(rows['calibrated'], rows['baseline'], cfg)
    w, logmean = candidate_weights(inputs['weight'], corrected_logits)
    delta = corrected_logits-baseline_logits
    correction = dict(weight_kl_to_baseline=float(np.sum(w*(delta-(logmean-rows['baseline']['log_mean_ratio'])))),
        minimum_raw_ratio_factor=float(np.exp(np.min(delta))),
        maximum_raw_ratio_factor=float(np.exp(np.max(delta))),
        changed_candidate_fraction=float(np.mean(np.abs(delta) > 1e-6)))
    # Primary statistical gate is cross-term improvement; guardrails are reported separately.
    qualification = (comparisons['offdiagonal']['ci95'][1] < 0 and all(guards.values())
        and comparisons['diagonal']['ci95'][1] <= cfg['guardrails']['diagonal_error_increase']
        and comparisons['total']['ci95'][1] <= cfg['guardrails']['total_error_increase'])
    return dict(arms=rows, comparisons=comparisons, observed_guardrails=guards, correction=correction,
        qualified_improvement=bool(qualification),
        uncertainty='Paired whole-event bootstrap; fixed selected coefficients/head/candidates. Pointwise primary/guardrail intervals; simultaneous component family9. No fit/selection uncertainty. Previously inspected panel; exploratory.'), dict(
        source_ids=inputs['source_ids'], event_numerators=np.stack(nums, 1), event_masses=np.stack(dens, 1),
        bootstrap_C=draws, bootstrap_truth=targets, arm_names=np.array(list(arm_logits)))


def analyze(inputs, generated, logits, cfg, progress=None, seal=None):
    validate(inputs, generated, logits, cfg)
    split = event_split(inputs['source_ids'], cfg['split_seed'], cfg['split_fractions'])
    idx = [np.flatnonzero(split == i) for i in range(3)]
    if min(map(len, idx)) < cfg['minimum_split_events']:
        raise ValueError('Insufficient events in a calibration/selection/test split')
    fi, vi, ti = idx
    fit_inputs, val_inputs = subset(inputs, fi), subset(inputs, vi)
    fg, vg, fs, vs = generated[fi], generated[vi], logits[fi], logits[vi]
    models, validation, calibration = {}, {}, {}
    baseline = metrics(val_inputs, vg, vs, cfg['condition_pt_edges'])[0]
    calibration['baseline'] = metrics(fit_inputs, fg, fs, cfg['condition_pt_edges'])[0]
    for strength in cfg['moment_strengths']:
        name = f'moment_{strength:g}'
        model = fit(fit_inputs, fg, fs, cfg, strength, progress)
        models[name] = model
        calibration[name] = metrics(fit_inputs, fg, transform(fs, fg, model), cfg['condition_pt_edges'])[0]
        validation[name] = metrics(val_inputs, vg, transform(vs, vg, model), cfg['condition_pt_edges'])[0]
        if progress:
            progress('selection_candidate', dict(name=name, converged=model['converged'],
                **{k: v for k, v in validation[name].items() if isinstance(v, (float, int))}))
    decision = select(models, validation, baseline, cfg)
    frozen = dict(selection=decision, models=models, split_counts=dict(zip(('calibration', 'selection', 'test'), map(len, idx))),
        split_seed=cfg['split_seed'], split_fractions=cfg['split_fractions'],
        calibration=calibration, validation=dict(baseline=baseline, **validation))
    # Save coefficients, split and decision BEFORE any final-test analysis.
    if seal:
        seal(frozen, split)
    if progress:
        progress('heldout_test', dict(selected=decision['selected']))
    tg, ts = generated[ti], logits[ti]
    corrected = ts if decision['selected'] == 'baseline' else transform(ts, tg, models[decision['selected']])
    test, arrays = paired_report(subset(inputs, ti), tg, ts, corrected, cfg)
    report = dict(**frozen, test=test, events=len(split), candidates=generated.shape[1],
        primary='Selected calibration minus same-event baseline: held-out offdiagonal Frobenius error',
        scope='Physics-targeted calibration of existing candidate weights, not a model-agnostic density-ratio estimate. Global/group moments do not prove full conditional tau closure.',
        baseline_retrained=False, policy_updates=0, classifier_fits=0, generated_samples=0,
        calibration_fits=len(models), pristine_test=False)
    arrays.update(all_source_ids=inputs['source_ids'], event_split=split)
    return report, arrays
