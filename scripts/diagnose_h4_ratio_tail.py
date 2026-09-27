#!/usr/bin/env python3
"""Offline tail attribution for an exported H4 classifier; no fitting or clipping."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def jsd(a, b, weights=None):
    edges = np.linspace(-1.000001, 1.000001, 17)
    p = np.histogramdd(a[:, [1, 2]], bins=[edges, edges])[0]
    q = np.histogramdd(b[:, [1, 2]], bins=[edges, edges], weights=weights)[0]
    if p.sum() != len(a) or (weights is None and q.sum() != len(b)):
        raise ValueError('Topology outside fixed physical bins')
    p, q = p/p.sum(), q/q.sum()
    m = (p+q)/2
    mp, mq = p>0, q>0
    return float(.5*((p[mp]*np.log(p[mp]/m[mp])).sum()+(q[mq]*np.log(q[mq]/m[mq])).sum()))


def weight_metrics(logits):
    s = logits.double()
    w = torch.softmax(s, 0)
    lm = float(torch.logsumexp(s,0)-math.log(len(s)))
    return dict(events=len(s), ess=float(1/w.square().sum()),
                ess_fraction=float(1/(len(s)*w.square().sum())),
                log_mean_ratio=lm, mean_ratio=math.exp(lm) if lm<700 else None,
                top1pct_mass=float(w.topk(max(1,math.ceil(.01*len(s)))).values.sum()),
                max_weight_mass=float(w.max()))


def summary(values):
    x = values.detach().double().reshape(-1)
    finite = x[torch.isfinite(x)]
    return dict(count=len(x), nonfinite=int((~torch.isfinite(x)).sum()),
                absmax=float(finite.abs().max()) if len(finite) else None,
                abs_p99=float(torch.quantile(finite.abs(),.99)) if len(finite) else None)


def unpack(bundle):
    x = bundle['test_condition']
    result, offset = {}, 0
    for name, shape in bundle['packing_spec']['shapes'].items():
        width = math.prod(shape)
        result[name] = x[:,offset:offset+width].reshape(len(x),*shape)
        offset += width
    if offset != x.shape[-1]:
        raise ValueError('Packing spec does not match test_condition')
    return result


def analyze(bundle):
    if bundle.get('schema') != 'h4-ratio-health-v1':
        raise ValueError('Unsupported artifact schema')
    g, t = [bundle[k].double().reshape(-1) for k in ['gen_logits','truth_logits']]
    n = len(g)
    if n <= 20 or len(t) != n:
        raise ValueError('Requires more than 20 paired K=1 test events')
    a, b = [bundle[k].double().reshape(n,3).numpy() for k in ['truth_topology','gen_topology']]
    if not (torch.isfinite(g).all() and torch.isfinite(t).all() and np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Nonfinite logits/topology; inspect artifact before ratio analysis')
    if np.max(np.abs(a))>1.000001 or np.max(np.abs(b))>1.000001:
        raise ValueError('Topology outside physical range')
    indices = bundle['split_indices']
    if len(indices['test']) != n or len(set(indices['test'].tolist())) != n:
        raise ValueError('Invalid test index mapping')
    if any(set(indices['test'].tolist()) & set(indices[k].tolist()) for k in ['fit','early_stop']):
        raise ValueError('Test overlaps model fitting or selection')
    order = torch.argsort(g,descending=True,stable=True)
    w = torch.softmax(g,0)
    gen_bce, truth_bce = F.softplus(g), F.softplus(-t)
    sensitivity=[]
    for count in [0,1,5,20]:
        keep = torch.ones(n,dtype=torch.bool); keep[order[:count]]=False
        for name, temperature in [('raw',1.),('tempered075',.75)]:
            s = temperature*g
            ww = torch.softmax(s[keep],0).numpy()
            # Both variants are descriptive interventions on the test sample.
            # Fixed truth isolates weight leverage; paired removal additionally
            # changes the target event population and is reported separately.
            baseline = jsd(a,b)
            fixed = jsd(a,b[keep.numpy()],ww)
            paired_baseline = jsd(a[keep.numpy()],b[keep.numpy()])
            paired = jsd(a[keep.numpy()],b[keep.numpy()],ww)
            sensitivity.append(dict(arm=name, removed=count, **weight_metrics(s[keep]),
                removed_original_weight_mass=float(torch.softmax(s,0)[~keep].sum()),
                removed_gen_bce_share=float(gen_bce[~keep].sum()/gen_bce.sum()),
                retained_raw_gen_bce=float(gen_bce[keep].mean()),
                fixed_truth_unweighted_jsd=baseline, fixed_truth_weighted_jsd=fixed,
                fixed_truth_jsd_change=fixed-baseline,
                paired_unweighted_jsd=paired_baseline, paired_weighted_jsd=paired,
                paired_jsd_change=paired-paired_baseline))
    fields = unpack(bundle)
    # Match checkpoint affine normalization only. inv_cdf_index is not in the
    # state dict, so never call these final network-normalized inputs.
    state = bundle['model_state']
    norm_prefixes = {'x':'backbone.sequential_normalizer', 'conditions':'backbone.global_normalizer'}
    input_ranges=[]
    valid_masks={}
    selected = torch.zeros(n,dtype=torch.bool); selected[order[:20]]=True
    for field, value in fields.items():
        if field.endswith('mask'): continue
        mask = fields.get(field+'_mask')
        valid = torch.ones_like(value,dtype=torch.bool)
        if mask is not None:
            mask = mask>.5
            while mask.ndim<value.ndim: mask=mask.unsqueeze(-1)
            valid = torch.broadcast_to(mask,value.shape)
        valid_masks[field]=valid
        transformed=None
        prefix=norm_prefixes.get(field,'')
        mean,std=state.get(prefix+'.mean'),state.get(prefix+'.std')
        if mean is not None and std is not None and mean.numel()==value.shape[-1]:
            transformed=(value.double()-mean.double())/std.double()
        for label, selection in [('top20',selected),('rest',~selected)]:
            entry=dict(field=field,group=label,raw=summary(value[selection][valid[selection]]))
            if transformed is not None:
                entry['affine_pre_icdf']=summary(transformed[selection][valid[selection]])
            input_ranges.append(entry)
    candidates = {k:bundle[k].reshape(n,2,2).double() for k in ['test_truth','test_generated']}
    if any(not torch.isfinite(v).all() for v in candidates.values()):
        raise ValueError('Nonfinite physical candidates')
    inv_mean,inv_std=[state.get('backbone.invisible_normalizer.'+k) for k in ['mean','std']]
    normalized={}
    if inv_mean is not None and inv_std is not None and inv_mean.numel()>=2:
        normalized={k:(v-inv_mean.reshape(-1)[:2])/inv_std.reshape(-1)[:2] for k,v in candidates.items()}
    edges=np.linspace(-1.000001,1.000001,17)
    bins=lambda z: np.clip(np.searchsorted(edges,z[:,[1,2]],side='right')-1,0,15)
    ab,bb=bins(a),bins(b)
    top=[]
    for rank,i in enumerate(order[:20].tolist(),1):
        cell=bb[i]
        ta=np.all(ab==cell,axis=1); gb=np.all(bb==cell,axis=1)
        row=dict(rank=rank,test_row=i,pool_row=int(indices['test'][i]),
                 gen_logit=float(g[i]),truth_logit=float(t[i]),weight_mass=float(w[i]),
                 cumulative_weight_mass=float(w[order[:rank]].sum()),
                 gen_bce=float(gen_bce[i]),gen_bce_share=float(gen_bce[i]/gen_bce.sum()),
                 truth_bce=float(truth_bce[i]),
                 generated_delta_theta_phi=candidates['test_generated'][i].tolist(),
                 truth_delta_theta_phi=candidates['test_truth'][i].tolist(),
                 generated_topology=b[i].tolist(),truth_topology=a[i].tolist(),
                 topology_cell=cell.tolist(),cell_truth_events=int(ta.sum()),cell_gen_events=int(gb.sum()),
                 cell_count_ratio=float(ta.sum()/gb.sum()),cell_raw_weight_mass=float(w[torch.from_numpy(gb)].sum()))
        if normalized:
            row['generated_affine_pre_icdf']=normalized['test_generated'][i].tolist()
            row['truth_affine_pre_icdf']=normalized['test_truth'][i].tolist()
        row['input_ranges']={k:summary(v[i][valid_masks[k][i]]) for k,v in fields.items() if not k.endswith('mask')}
        top.append(row)
    return dict(schema='h4-ratio-tail-v1',events=n,source_seed=bundle['seed'],
        best_step=bundle['fit_diagnostics'].get('best_step'),
        baseline_test_bce=float(.5*(truth_bce.mean()+gen_bce.mean())),
        normalization_note='Affine pre-ICDF only; missing transforms are not reconstructed. No test-fitted normalization.',
        identity_note='pool_row indexes saved identity_condition, NOT original parquet source_event_index.',
        interpretation='Descriptive post-hoc sensitivity, not a clipping policy or proof of conditional support/calibration.',
        top20=top,input_ranges=input_ranges,sensitivity=sensitivity)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not (args.directory/'COMPLETE').is_file(): parser.error('Incomplete ratio export')
    if args.output.exists(): parser.error('Output exists; choose a new report path')
    bundle=torch.load(args.directory/'best_classifier_and_test.pt',map_location='cpu',weights_only=True)
    result=analyze(bundle)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    console={k:v for k,v in result.items() if k!='top20'}
    console['top20']=[{k:v for k,v in row.items() if k in (
        'rank','test_row','pool_row','gen_logit','truth_logit','weight_mass',
        'cumulative_weight_mass','gen_bce_share','generated_delta_theta_phi',
        'cell_truth_events','cell_gen_events','cell_count_ratio','cell_raw_weight_mass')}
        for row in result['top20']]
    print(json.dumps(console,indent=2,allow_nan=False))
    print('TOP20 DETAILS:',args.output)


if __name__=='__main__': main()
