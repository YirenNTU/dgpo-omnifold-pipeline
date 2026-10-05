"""Nested, raw-ratio moment estimates; candidates are clustered by condition."""
import numpy as np


def validate_inputs(truth, generated, logits, weight, prefixes):
    truth, generated, logits, weight = [np.asarray(x, dtype=np.float64)
                                        for x in (truth, generated, logits, weight)]
    n = len(weight)
    if (not prefixes or prefixes[0]!=1 or n < 2 or truth.ndim != 2 or truth.shape[0]!=n or generated.shape != (n, max(prefixes), truth.shape[1])
        or logits.shape != generated.shape[:2] or weight.shape != (n,)
        or any(k < 1 for k in prefixes) or sorted(set(prefixes)) != list(prefixes)
        or not all(np.isfinite(x).all() for x in (truth, generated, logits, weight))
        or (weight < 0).any() or weight.sum() <= 0):
        raise ValueError('Invalid nested candidate arrays or prefixes')
    return truth, generated, logits, weight


def aggregates(generated, logits, weight, k):
    """Return per-event numerator/denominator for U and R, not event-normalized ratios."""
    g, s = generated[:, :k], logits[:, :k]
    shift = float(s[weight>0].max())
    r = np.zeros_like(s)
    r[weight>0] = np.exp(s[weight>0]-shift)
    den = np.stack((weight, weight*r.mean(1)), 1)
    num = np.stack((weight[:, None]*g.mean(1),
                    weight[:, None]*(r[..., None]*g).mean(1)), 1)
    return num, den, r, shift


def conditional_mc_se(generated, r, weight, matrices):
    """Delta-method conditional sampling SE, fixed events/truth/model.

    Uses independent draws WITHIN each condition; k=1 cannot estimate this.
    Does not include uncertainty from the finite condition/truth population.
    Heavy tails can invalidate the finite-sample approximation.
    """
    k = generated.shape[1]
    if k < 2:
        return None
    u = weight[:, None, None]*(generated-matrices[0])/weight.sum()
    den = (weight[:, None]*r).sum()/k
    v = weight[:, None, None]*r[..., None]*(generated-matrices[1])/den
    return np.sqrt(np.stack((u, v, v-u), 2).var(axis=1, ddof=1).sum(0)/k)


def convergence_report(truth, generated, logits, weight, prefixes=(1,2,4,8), bootstrap=500, seed=42):
    truth, generated, logits, weight = validate_inputs(truth, generated, logits, weight, prefixes)
    if bootstrap < 20:
        raise ValueError('At least 20 condition bootstrap draws required')
    n, _, d = generated.shape
    target = weight @ truth/weight.sum()
    nums, dens, rows = [], [], {}
    for k in prefixes:
        num, den, r, shift = aggregates(generated, logits, weight, k)
        matrices = num.sum(0)/den.sum(0)[:, None]
        error = np.linalg.norm(matrices-target, axis=1)
        w = weight[:, None]*r
        w /= w.sum()
        ew = w.sum(1)
        influence = (num-den[..., None]*matrices[None])/den.sum(0)[None, :, None]
        mass = np.abs(influence).sum(0)
        count = max(1, int(np.ceil(.01*n)))
        share = np.divide(np.sort(np.abs(influence), axis=0)[-count:].sum(0), mass,
                          out=np.zeros_like(mass), where=mass>0)
        mc = conditional_mc_se(generated[:, :k], r, weight, matrices)
        rows[str(k)] = dict(unweighted=matrices[0].tolist(), reweighted=matrices[1].tolist(),
            unweighted_error=float(error[0]), reweighted_error=float(error[1]),
            error_change=float(error[1]-error[0]),
            candidate_ess=float(1/np.square(w).sum()),
            candidate_ess_fraction=float(1/(n*k*np.square(w).sum())),
            event_ess=float(1/np.square(ew).sum()),
            event_ess_fraction=float(1/(n*np.square(ew).sum())),
            max_candidate_mass=float(w.max()), max_event_mass=float(ew.max()),
            top1pct_candidate_mass=float(np.sort(w.ravel())[-max(1,int(np.ceil(.01*n*k))):].sum()),
            log_mean_ratio=float(shift+np.log(den[:,1].sum()/weight.sum())),
            top1pct_event_absolute_influence=share.tolist(),
            conditional_mc_se=None if mc is None else mc.tolist())
        nums.append(num); dens.append(den)
    # One shared event resample for every K and arm. Never resample NK as independent events.
    nums, dens = np.stack(nums,1), np.stack(dens,1)
    rng = np.random.default_rng(seed)
    samples, targets = [], []
    for _ in range(bootstrap):
        count = rng.multinomial(n, np.full(n,1/n)).astype(float)
        base = count*weight
        if base.sum() <= 0:
            raise ValueError('Bootstrap has zero base mass')
        den = np.einsum('n,nka->ka',count,dens)
        if (den <= 0).any():
            raise ValueError('Bootstrap has zero ratio mass')
        samples.append(np.einsum('n,nkad->kad',count,nums)/den[...,None])
        targets.append(base @ truth/base.sum())
    samples, targets = np.asarray(samples), np.asarray(targets)
    residual = samples-targets[:,None,None,:]
    errors = np.linalg.norm(residual,axis=-1)
    for j,k in enumerate(prefixes):
        row=rows[str(k)]
        row['matrix_ci95']=np.quantile(samples[:,j],[.025,.975],axis=0).tolist()
        row['residual_ci95']=np.quantile(residual[:,j],[.025,.975],axis=0).tolist()
        row['error_change_ci95']=np.quantile(errors[:,j,1]-errors[:,j,0],[.025,.975]).tolist()
        row['reweighted_minus_K1']= (np.asarray(row['reweighted'])-rows['1']['reweighted']).tolist()
        row['reweighted_minus_K1_ci95']=np.quantile(samples[:,j,1]-samples[:,0,1],[.025,.975],axis=0).tolist()
    # Disjoint candidate panels expose sampling instability without retraining seeds.
    panels={}
    for k in prefixes:
        panels[str(k)]=[]
        for start in range(0,generated.shape[1],k):
            num,den,_,_=aggregates(generated[:,start:start+k],logits[:,start:start+k],weight,k)
            m=num.sum(0)/den.sum(0)[:,None]
            panels[str(k)].append(dict(start=start,unweighted=m[0].tolist(),reweighted=m[1].tolist(),
                error_change=float(np.linalg.norm(m[1]-target)-np.linalg.norm(m[0]-target))))
    return dict(events=n,candidates=generated.shape[1],truth=target.tolist(),prefixes=rows,
        disjoint_panels=panels,bootstrap=bootstrap,
        uncertainty='Paired event-cluster bootstrap, fixed models/candidates; pointwise intervals. '
        'Separate conditional Monte Carlo SE is a within-event delta-method approximation (U,R,R-U); '
        'K1 unavailable. No classifier-refit uncertainty. Heavy tails may remain unresolved.',
        interpretation='Raw exp(logit), base_weight/K; only global self-normalization. '
        'No cap, tempering, event normalization, candidate selection or fitted physics target.')


def group_report(truth, generated, logits, weight, category, pt, edges, prefixes):
    """Same fit-only category x pT strata as classifier experiment, now with K draws."""
    truth, generated, logits, weight=validate_inputs(truth,generated,logits,weight,prefixes)
    bins=np.searchsorted(edges,pt,side='right')
    cats=np.unique(category)
    groups=[('all',np.ones(len(weight),bool))]
    groups += [(f'category_{c}',category==c) for c in cats]
    groups += [(f'pt_{i}',bins==i) for i in range(4)]
    groups += [(f'category_{c}_pt_{i}',(category==c)&(bins==i)) for c in cats for i in range(4)]
    result={}
    for k in prefixes:
        num,den,_,shift=aggregates(generated,logits,weight,k)
        rows=[]
        for name,take in groups:
            if weight[take].sum()<=0: continue
            d=den[take].sum(0)
            if (d<=0).any(): raise ValueError('Underflowed group')
            target=weight[take]@truth[take]/weight[take].sum()
            m=num[take].sum(0)/d[:,None]
            rows.append(dict(group=name,events=int(take.sum()),base_mass=float(d[0]/weight.sum()),
                weighted_mass=float(d[1]/den[:,1].sum()),
                log_mean_ratio=float(shift+np.log(d[1]/d[0])),
                unweighted_error=float(np.linalg.norm(m[0]-target)),
                reweighted_error=float(np.linalg.norm(m[1]-target))))
        result[str(k)]=rows
    return result
