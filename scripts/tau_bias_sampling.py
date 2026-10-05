"""Paired event resampling: no classifier ratios and no prediction-dependent cuts."""
import numpy as np

AXES = [a+b for a in 'krn' for b in 'krn']


def target_probabilities(z, weights, target):
    """Stable exponential tilt of nonnegative MC mass; no generated inputs."""
    z, w = np.asarray(z,float), np.asarray(weights,float)
    if z.shape != w.shape or not np.isfinite(z).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0 or not np.isfinite(target):
        raise ValueError('Invalid tilt inputs')
    keep = w>0
    if not z[keep].min() < target < z[keep].max():
        raise ValueError('Target outside the interior of supported truth moments')
    scale=max(float(np.std(z[keep])),1e-12)
    x=(z[keep]-target)/scale
    logw=np.log(w[keep])
    def evaluate(t):
        logits=logw+t*x; p=np.exp(logits-logits.max()); p/=p.sum()
        return p,float(p@z[keep])
    lo,hi=-1.,1.
    for _ in range(60):
        if evaluate(lo)[1]<=target<=evaluate(hi)[1]: break
        lo*=2; hi*=2
    else: raise ValueError('Could not bracket target')
    for _ in range(100):
        mid=(lo+hi)/2; p,mean=evaluate(mid)
        if abs(mean-target)<1e-10: break
        if mean<target: lo=mid
        else: hi=mid
    prob=np.zeros_like(w); prob[keep]=p
    if abs(mean-target)>1e-8: raise ValueError('Target solve did not converge')
    return prob,float(mid/scale)


def bin_indices(u):
    u = np.asarray(u)
    if not np.isfinite(u).all() or (np.abs(u) > 1+1e-8).any():
        raise ValueError('Truth products must be in [-1,1]')
    return np.searchsorted(np.linspace(-1, 1, 7)[1:-1], u, side='right')


def polarization_moments(a, b, kappas):
    """Signed-kappa convention: 3<a_i/kappa_a>, 3<b_i/kappa_b>.

    Selected-sample angular moments, not acceptance-corrected polarization.
    No additional tau-charge sign is assumed.
    """
    k = np.asarray(kappas, float)
    if k.shape != (len(a),2) or not np.isfinite(k).all() or (k == 0).any():
        raise ValueError('Invalid analyzing powers')
    return 3*np.concatenate((a/k[:,0,None], b/k[:,1,None]), axis=1)


def sampling_scan(products, truth, predictions, weights, seed=421001, repeats=100,
                  amplitude=.5, moments=None, targets=None, min_ess_fraction=.1):
    """Truth refs and predictions: dictionaries of [N,9] event contributions.

    Draw N events with replacement according ONLY to stress multipliers;
    retain original MC weights in means. Never multiply the stress twice.
    Repeats estimate resampling variability conditional on this fixed panel,
    not unconditional coverage or training uncertainty.
    """
    weights = np.asarray(weights, float)
    n = len(weights)
    if repeats < 2 or n < 2 or products.shape != (n, 9):
        raise ValueError('Need [N,9] products, N>=2 and >=2 repeats')
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Require finite nonnegative MC weights')
    arrays = {**{'truth/'+k: v for k, v in truth.items()}, **predictions}
    if any(v.shape != (n,9) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Invalid Cij event contributions')
    if not 0 < amplitude < 1:
        raise ValueError('Amplitude must be between 0 and 1')
    if moments is not None:
        if set(moments) != set(arrays) or any(v.shape != (n,6) or not np.isfinite(v).all() for v in moments.values()):
            raise ValueError('Need six finite B moments for every model/reference')
        arrays = {k:np.concatenate((v,moments[k]),axis=1) for k,v in arrays.items()}
    bins = bin_indices(products)
    cases = [('nominal', None, 0)] + [(f'{axis}/{direction:+d}', j, direction)
                                    for j, axis in enumerate(AXES) for direction in (-1, 1)]
    targeted = targets is not None
    if targeted:
        if 'full' not in truth or not targets or not np.isfinite(targets).all() or len(set(targets))!=len(targets):
            raise ValueError('Need full truth and unique finite targets')
        cases=[('nominal',None,None)]+[(f'{axis}/target={target:+.2f}',j,float(target)) for j,axis in enumerate(AXES) for target in targets]
    nominal_probability=weights/weights.sum() if targeted else np.ones(n)/n
    nominal_cdf=np.cumsum(nominal_probability); nominal_cdf[-1]=1
    baseline = {k: np.average(v, axis=0, weights=weights) for k,v in arrays.items()}
    reports = []
    for name, j, direction in cases:
        meta={}
        if targeted:
            factors=None
            try:
                probability,tilt=(nominal_probability,0.) if j is None else target_probabilities(truth['full'][:,j],weights,direction)
            except ValueError as exc:
                reports.append(dict(case=name,status='unsupported',target_component=AXES[j],requested_target=direction,reason=str(exc)))
                continue
            ess=float(1/(n*np.sum(probability**2)))
            meta=dict(target_component=None if j is None else AXES[j],requested_target=direction,tilt_lambda=tilt,
                expected_target=None if j is None else float(probability@truth['full'][:,j]),
                sampling_ess_fraction=ess,max_probability=float(probability.max()),
                top1pct_probability=float(np.sort(probability)[-max(1,int(np.ceil(n*.01))):].sum()),
                expected_unique_fraction=float(np.sum(-np.expm1(n*np.log1p(-np.minimum(probability,1-1e-16))))/n),
                concentration_warning=ess<min_ess_fraction,
                status='low_support' if ess<min_ess_fraction else 'ok')
        else:
            factors = np.ones(6) if j is None else 1+direction*np.linspace(-amplitude,amplitude,6)
            probability = np.ones(n) if j is None else factors[bins[:, j]]
            probability /= probability.sum()
        cdf = np.cumsum(probability); cdf[-1] = 1
        draws = {k: [] for k in arrays}
        changes = {k: [] for k in arrays}
        counts = np.zeros(6, dtype=float)
        # Identical uniforms for both models, all refs and injection variants.
        rng = np.random.default_rng(seed)
        for _ in range(repeats):
            uniform = rng.random(n)
            idx = np.minimum(np.searchsorted(cdf, uniform, side='right'), n-1)
            nominal_idx = np.minimum(np.searchsorted(nominal_cdf,uniform,side='right'),n-1) if targeted else np.minimum((uniform*n).astype(int), n-1)
            if weights[idx].sum() <= 0 or weights[nominal_idx].sum() <= 0:
                raise ValueError('Resampling yielded zero total MC weight')
            for key, values in arrays.items():
                value = np.mean(values[idx],axis=0) if targeted else np.average(values[idx], axis=0, weights=weights[idx])
                nominal = np.mean(values[nominal_idx],axis=0) if targeted else np.average(values[nominal_idx], axis=0, weights=weights[nominal_idx])
                draws[key].append(value); changes[key].append(value-nominal)
            if j is not None:
                counts += np.bincount(bins[idx,j], minlength=6)/repeats
        draws = {k: np.asarray(v) for k,v in draws.items()}
        changes = {k: np.asarray(v) for k,v in changes.items()}
        b_report = {}
        if moments is not None:
            b_report = dict(B={k:v[:,9:].mean(0).tolist() for k,v in draws.items()},
                B_interval95={k:np.quantile(v[:,9:],[.025,.975],axis=0).tolist() for k,v in draws.items()},
                B_bias={f'{ref}/{model}':dict(mean=(draws[model][:,9:]-draws['truth/'+ref][:,9:]).mean(0).tolist(),
                    interval95=np.quantile(draws[model][:,9:]-draws['truth/'+ref][:,9:],[.025,.975],axis=0).tolist())
                    for ref in truth for model in predictions})
        draws = {k:v[:,:9] for k,v in draws.items()}
        changes = {k:v[:,:9] for k,v in changes.items()}
        row = dict(case=name, sampling_multipliers=None if factors is None else factors.tolist(),
                   mean_sampled_bin_counts=counts.tolist() if j is not None else None,
                   C={k:v.mean(0).reshape(3,3).tolist() for k,v in draws.items()},
                   delta_C={k:v.mean(0).reshape(3,3).tolist() for k,v in changes.items()}, metrics={})
        row.update(b_report)
        row.update(meta)
        row['C_interval95'] = {k:np.quantile(v,[.025,.975],axis=0).tolist() for k,v in draws.items()}
        row['C_bias'] = {f'{ref}/{model}':dict(mean=(draws[model]-draws['truth/'+ref]).mean(0).tolist(),
            interval95=np.quantile(draws[model]-draws['truth/'+ref],[.025,.975],axis=0).tolist())
            for ref in truth for model in predictions}
        errors = {}
        for ref in truth:
            target = 'truth/'+ref
            for model in predictions:
                residual = draws[model]-draws[target]
                response = changes[model]-changes[target]
                for metric, values in [('absolute', residual), ('response', response)]:
                    norms = np.linalg.norm(values, axis=1)
                    key = f'{ref}/{model}/{metric}'
                    errors[key] = norms
                    row['metrics'][key] = dict(mean=float(norms.mean()),
                        resampling_interval95=np.quantile(norms,[.025,.975]).tolist(),
                        signed_matrix_mean=values.mean(0).reshape(3,3).tolist())
            if 'dgpo' in predictions and 'pretrain' in predictions:
                for metric in ('absolute','response'):
                    gain = errors[f'{ref}/dgpo/{metric}']-errors[f'{ref}/pretrain/{metric}']
                    row['metrics'][f'{ref}/dgpo_minus_pretrain/{metric}'] = dict(
                        mean=float(gain.mean()), resampling_interval95=np.quantile(gain,[.025,.975]).tolist())
        reports.append(row)
        if targeted:
            print(f'[Cij resampling] {len(reports)}/{len(cases)} {name}: '
                  f'status={meta["status"]} ESS/N={meta["sampling_ess_fraction"]:.4f}',flush=True)
    return dict(events=n, repeats=repeats, seed=seed, amplitude=amplitude, targets=targets,
        sampling_mode='truth_moment_exponential_tilt_equal_weight_draws' if targeted else 'bin_multipliers_mc_weighted_draws',
        min_ess_fraction=min_ess_fraction, edges=np.linspace(-1,1,7).tolist(),
        nominal_full_panel_C={k:v[:9].reshape(3,3).tolist() for k,v in baseline.items()},
        nominal_full_panel_B={k:v[9:].tolist() for k,v in baseline.items()} if moments is not None else {},
        B_convention='B angular moments: +3<a_i/kappa_a>, +3<b_i/kappa_b>; signed parquet kappas, common k/r/n axes; no acceptance correction or extra charge sign; not certified physical polarization',
        bin_counts={axis:np.bincount(bins[:,j],minlength=6).tolist() for j,axis in enumerate(AXES)},
        scope='Development frozen-model stress test. Intervals describe repeated resampling of this fixed panel '
              'and fixed predictions, not full coverage, unseen-data or training uncertainty. No stress weights '
              'in Cij estimator. Target mode includes original MC weights in sampling probability and uses unit analysis weights; '
              'bin mode retains original MC analysis weights. Angular-moment stress only, not physical spin-state injection. '
              'Negative DGPO-minus-pretrain is improvement.',
        cases=reports)
