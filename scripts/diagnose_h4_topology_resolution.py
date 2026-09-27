#!/usr/bin/env python3
"""Read-only coordinate, unit-contract and fixed-resolution topology audit."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from diagnose_h4_ratio_tail import unpack, weight_metrics


def quantiles(x):
    return dict(zip(['min','q01','q10','q50','q90','q99','max'],
                    np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist()))


def reconstruct(fields, candidate):
    n=len(candidate)
    z=candidate.double().reshape(n,2,2).numpy()
    if not np.isfinite(z).all(): raise ValueError('Nonfinite candidates')
    directions=[]; invalid={}; angles=[]
    for j,leg in enumerate(['a','b']):
        p=np.stack([fields[f'lead_{leg}_visible_{axis}'].reshape(n).double().numpy() for axis in ['px','py','pz']],1)
        if not np.isfinite(p).all(): raise ValueError('Nonfinite visible momentum')
        invalid[leg+'_zero_momentum']=int((np.linalg.norm(p,axis=1)==0).sum())
        theta=np.arctan2(np.hypot(p[:,0],p[:,1]),p[:,2])+z[:,j,0]
        phi=np.arctan2(p[:,1],p[:,0])+z[:,j,1]
        invalid[leg+'_theta_outside_0_pi']=int(((theta<0)|(theta>math.pi)).sum())
        directions.append(np.stack([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)],1))
        angles.append((theta,phi))
    delta=angles[0][1]-angles[1][1]
    wrapped=np.arctan2(np.sin(delta),np.cos(delta))
    dot=np.sum(directions[0]*directions[1],1)
    # atan2 avoids acos precision loss near +/-1.
    opening=np.arctan2(np.linalg.norm(np.cross(*directions),axis=1),dot)
    return dict(topology=np.stack([np.sin(delta),np.cos(delta),dot],1),
                acoplanarity=math.pi-np.abs(wrapped),opening=opening,
                acollinearity=math.pi-opening,invalid=invalid,
                candidate=z.reshape(n,4))


def distance(p,q):
    p=p/p.sum(); q=q/q.sum(); m=(p+q)/2
    ip,iq=p>0,q>0
    return float(.5*((p[ip]*np.log(p[ip]/m[ip])).sum()+(q[iq]*np.log(q[iq]/m[iq])).sum()))


def w1(a,b,w):
    aa=np.sort(a); order=np.argsort(b); bb=b[order]; cum=np.r_[0,np.cumsum(w[order]/w.sum())]
    grid=np.unique(np.r_[aa,bb])
    fa=np.searchsorted(aa,grid[:-1],side='right')/len(aa)
    fb=cum[np.searchsorted(bb,grid[:-1],side='right')]
    return float(np.sum(np.abs(fa-fb)*np.diff(grid)))


def analyze(bundle):
    if bundle.get('schema')!='h4-ratio-health-v1': raise ValueError('Unknown artifact schema')
    fields=unpack(bundle)
    t=reconstruct(fields,bundle['test_truth']); g=reconstruct(fields,bundle['test_generated'])
    logits=bundle['gen_logits'].double().reshape(-1)
    n=len(logits)
    if n!=len(t['topology']) or not torch.isfinite(logits).all(): raise ValueError('Invalid score alignment')
    indices=bundle['split_indices']
    if len(indices['test'])!=n or any(set(indices['test'].tolist()) & set(indices[k].tolist()) for k in ['fit','early_stop']):
        raise ValueError('Invalid or overlapping test split')
    top=torch.argsort(logits,descending=True,stable=True)[:20].numpy()
    weights={'unweighted':np.ones(n)/n,'raw':torch.softmax(logits,0).numpy(),
             'tempered075':torch.softmax(.75*logits,0).numpy()}
    checks={}
    for label,values in [('truth',t),('generated',g)]:
        saved=bundle['truth_topology' if label=='truth' else 'gen_topology'].double().reshape(n,3).numpy()
        if not np.isfinite(saved).all(): raise ValueError('Nonfinite saved topology')
        checks[label]=dict(max_abs_reconstruction_error=float(np.abs(saved-values['topology']).max()),
                           matches_saved=bool(np.allclose(saved,values['topology'],atol=2e-5,rtol=0)),
                           invalid=values['invalid'],
                           coordinates={k:quantiles(values[k]) for k in ['acoplanarity','opening','acollinearity']},
                           candidate_radian_contract={name:quantiles(values['candidate'][:,j]) for j,name in enumerate(['a_delta_theta','a_delta_phi','b_delta_theta','b_delta_phi'])})
    # Both endpoints are resolved; all resolutions/edges are fixed before data.
    angular_log=np.r_[0,np.geomspace(1e-6,math.pi,64)]
    specifications=[('cosine16','cosine',np.linspace(-1.000001,1.000001,17)),
                    ('cosine64','cosine',np.linspace(-1.000001,1.000001,65)),
                    ('cosine128','cosine',np.linspace(-1.000001,1.000001,129)),
                    ('angle64','angles',np.linspace(0,math.pi+1e-9,65)),
                    ('endpoint_log64','angles',angular_log)]
    histograms=[]
    for name,kind,edges in specifications:
        a=t['topology'][:,[1,2]] if kind=='cosine' else np.stack([t['acoplanarity'],t['acollinearity']],1)
        b=g['topology'][:,[1,2]] if kind=='cosine' else np.stack([g['acoplanarity'],g['acollinearity']],1)
        p=np.histogramdd(a,bins=[edges,edges])[0]
        q=np.histogramdd(b,bins=[edges,edges])[0]
        if p.sum()!=n or q.sum()!=n: raise ValueError('Histogram dropped events')
        cells=np.clip(np.searchsorted(edges,b,side='right')-1,0,len(edges)-2)
        occupied=(p+q)>0
        result=dict(name=name,edges=edges.tolist(),truth_dominant_cell_fraction=float(p.max()/n),
                    generated_dominant_cell_fraction=float(q.max()/n),occupied_cells=int(occupied.sum()),
                    fraction_occupied_cells_with_lt5_each=float(((p[occupied]<5)|(q[occupied]<5)).mean()),
                    unweighted_jsd=distance(p,q),arms={},top20_cells=[])
        for arm,w in weights.items():
            qw=np.histogramdd(b,bins=[edges,edges],weights=w)[0]
            result['arms'][arm]=dict(jsd=distance(p,qw),jsd_change=distance(p,qw)-distance(p,q))
        for i in top:
            cell=tuple(cells[i]); hits=np.all(cells==cells[i],axis=1)
            result['top20_cells'].append(dict(test_row=int(i),cell=list(map(int,cell)),
                truth_count=int(p[cell]),generated_count=int(q[cell]),
                count_ratio=float(p[cell]/q[cell]),raw_weight_mass=float(weights['raw'][hits].sum())))
        histograms.append(result)
    continuous={key:{arm:w1(t[key],g[key],w) for arm,w in weights.items()}
                for key in ['acoplanarity','opening','acollinearity']}
    return dict(schema='h4-topology-resolution-v1',events=n,checks=checks,histograms=histograms,
                wasserstein1_radians=continuous,raw_weights=weight_metrics(logits),
                units='Production contract assumes candidate deltas in radians; visible angles from atan2(momentum). Agreement checks implementation, not upstream unit correctness.',
                scope='Descriptive reused-test audit; no bin/temperature selection or support/calibration proof. Fine sparse-bin JSD has sampling bias.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not (args.directory/'COMPLETE').is_file(): parser.error('Incomplete artifact')
    if args.output.exists(): parser.error('Output exists; choose a fresh path')
    bundle=torch.load(args.directory/'best_classifier_and_test.pt',map_location='cpu',weights_only=True)
    result=analyze(bundle)
    with args.output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
    console={k:v for k,v in result.items() if k!='histograms'}
    console['histograms']=[{k:v for k,v in h.items() if k not in ('edges','top20_cells')} for h in result['histograms']]
    console['endpoint_log64_top20_cells']=result['histograms'][-1]['top20_cells']
    print(json.dumps(console,indent=2,allow_nan=False)); print('FULL REPORT:',args.output)


if __name__=='__main__': main()
