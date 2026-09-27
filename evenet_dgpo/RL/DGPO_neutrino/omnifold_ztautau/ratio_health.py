"""Read-only K=1, equal-prior, independent-test density-ratio diagnostics."""
import math
import numpy as np
import torch
from torch.nn import functional as F


def report(truth_logits, gen_logits, truth_topology, gen_topology):
    t, g = [x.detach().cpu().double().reshape(-1) for x in (truth_logits, gen_logits)]
    a, b = [x.detach().cpu().double().numpy().reshape(-1, 3) for x in (truth_topology, gen_topology)]
    if len(t) != len(g) or len(t) < 2 or len(a) != len(t) or len(b) != len(g):
        raise ValueError('Requires paired K=1 held-out events')
    if not (torch.isfinite(t).all() and torch.isfinite(g).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite held-out values')
    out = {'events': len(g), 'test_bce': float(.5*(F.softplus(-t).mean()+F.softplus(g).mean()))}
    # Fixed physical bounds, no test-fitted bins. First two channels are tau
    # delta-phi sin/cos; third is the physical opening-angle cosine.
    edges = np.linspace(-1.000001, 1.000001, 17)
    def distances(weights, indices=None):
        aa, bb = (a,b) if indices is None else (a[indices],b[indices])
        result = {}
        for name, axes in [('delta_phi_cos', [1]), ('opening_cos', [2]), ('joint_delta_phi_opening', [1,2])]:
            p = np.histogramdd(aa[:,axes], bins=[edges]*len(axes))[0]
            q = np.histogramdd(bb[:,axes], bins=[edges]*len(axes), weights=weights)[0]
            if p.sum() <= 0 or q.sum() <= 0:
                raise ValueError('Empty topology histogram')
            p, q = p/p.sum(), q/q.sum()
            m = (p+q)/2
            maskp, maskq = p>0, q>0
            result[name] = float(.5*(np.sum(p[maskp]*np.log(p[maskp]/m[maskp]))+np.sum(q[maskq]*np.log(q[maskq]/m[maskq]))))
        return result
    baseline = distances(np.ones(len(g))/len(g))
    out.update({f'unweighted/{k}_jsd':v for k,v in baseline.items()})
    # Prespecified sensitivity only: never choose temperature using this test.
    for name, temperature in [('raw',1.), ('tempered075',.75)]:
        s = temperature*g
        w = torch.softmax(s,0)
        lm = float(torch.logsumexp(s,0)-math.log(len(s)))
        values = dict(log_mean_ratio=lm, ess_fraction=float(1/(len(w)*w.square().sum())),
                      top1pct_mass=float(w.topk(max(1,math.ceil(.01*len(w)))).values.sum()),
                      max_weight_mass=float(w.max()), log_ratio_max=float(s.max()))
        for q in [.01,.5,.95,.99,.999]:
            values[f'log_ratio_q{q:g}'] = float(torch.quantile(s,q))
        # Stable relative SE of the UNNORMALIZED mean, for iid event sampling.
        values['mean_ratio_relative_se'] = math.sqrt(max(0.,float(len(w)*w.square().sum()-1))/(len(w)-1))
        for key,value in values.items(): out[f'{name}/{key}'] = value
        for key,value in distances(w.numpy()).items():
            out[f'{name}/{key}_jsd'] = value
            out[f'{name}/{key}_jsd_change'] = value-baseline[key]
        # Paired event bootstrap: truth/generated candidates from the same
        # event always move together. Conditional on this fixed fitted model;
        # it does not capture retraining uncertainty or unseen extreme tails.
        rng = np.random.default_rng(417)
        changes, log_means = [], []
        scores = s.numpy()
        for _ in range(200):
            idx = rng.integers(0,len(g),len(g))
            selected = scores[idx]
            peak = selected.max()
            ww = np.exp(selected-peak)
            log_means.append(float(peak+np.log(ww.mean())))
            ww /= ww.sum()
            old = distances(np.ones(len(g))/len(g), idx)
            new = distances(ww, idx)
            changes.append(new['joint_delta_phi_opening']-old['joint_delta_phi_opening'])
        for label,values in [('joint_jsd_change',changes),('log_mean_ratio',log_means)]:
            lo, hi = np.quantile(values,[.025,.975])
            out[f'{name}/{label}_bootstrap_lo95'] = float(lo)
            out[f'{name}/{label}_bootstrap_hi95'] = float(hi)
    return out


def export_bundle(directory, model, fit_config, diagnostics, protocol, seed,
                  condition, truth, generated, truth_logits, gen_logits,
                  fit_idx, early_idx, test_idx, identity_condition):
    """Called on rank zero after restored-best scoring; no optimizer mutation."""
    import json
    from dataclasses import asdict
    from pathlib import Path
    from .evenet_ratio import periodic_tau_pair_features
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    if any(set(x.cpu().tolist()) & set(y.cpu().tolist()) for x,y in [(fit_idx,early_idx),(fit_idx,test_idx),(early_idx,test_idx)]):
        raise ValueError('Overlapping split indices')
    topology = [periodic_tau_pair_features(condition[test_idx], x[test_idx], model.packing_spec).detach().cpu() for x in (truth,generated)]
    metrics = report(truth_logits, gen_logits, *topology)
    bundle = dict(schema='h4-ratio-health-v1', seed=seed, split_protocol=protocol,
        fit_config=asdict(fit_config), fit_diagnostics=asdict(diagnostics),
        packing_spec=model.packing_spec.to_dict(),
        model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
        split_indices={k:v.cpu() for k,v in [('fit',fit_idx),('early_stop',early_idx),('test',test_idx)]},
        identity_condition=identity_condition.detach().cpu(),
        test_condition=condition[test_idx].detach().cpu(),
        test_truth=truth[test_idx].detach().cpu(), test_generated=generated[test_idx].detach().cpu(),
        truth_logits=truth_logits.detach().cpu(), gen_logits=gen_logits.detach().cpu(),
        truth_topology=topology[0], gen_topology=topology[1])
    torch.save(bundle, destination/'best_classifier_and_test.pt')
    with (destination/'report.json').open('x') as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
    # Written last: incomplete export must never be treated as a valid artifact.
    (destination/'COMPLETE').write_text('h4-ratio-health-v1\n')
    return metrics
