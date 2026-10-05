"""Paired angular-moment closure; no generation, fitting, cuts or unfolding.

See docs/reweighted_cij.md for the mandatory physics/input contract.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def features(a, b, kappas):
    kappas=np.asarray(kappas)
    product=np.prod(kappas,axis=-1)
    while product.ndim < a.ndim-1:
        product=product[...,None]
    return (9*a[..., :, None]*b[..., None, :]/product[...,None,None]).reshape(*a.shape[:-1], 9)


def prepare(data, kappas, mode):
    required = ('event_id', 'truth_a', 'truth_b', 'sample_a', 'sample_b', 'log_ratio', 'event_weight')
    missing = set(required)-set(data)
    if missing:
        raise ValueError(f'Missing input arrays: {sorted(missing)}')
    ids=np.asarray(data['event_id'])
    n=len(ids)
    if ids.ndim != 1 or n < 2 or len(np.unique(ids)) != n:
        raise ValueError('Need >=2 unique event IDs; store K candidates within each event')
    ta,tb,sa,sb = [np.asarray(data[k],dtype=float) for k in required[1:5]]
    if ta.shape != (n,3) or tb.shape != ta.shape or sa.ndim != 3 or sa.shape[0] != n or sa.shape[2] != 3 or sb.shape != sa.shape or sa.shape[1] < 1:
        raise ValueError('Expected truth [N,3], sample [N,K,3], axes k,r,n')
    for x in (ta,tb,sa,sb):
        if not np.isfinite(x).all() or (np.abs(x)>1+1e-8).any() or not np.allclose((x*x).sum(-1),1,atol=1e-5):
            raise ValueError('Cosines must be finite unit directions in a common orthonormal basis')
    lw=np.asarray(data['log_ratio'],dtype=float)
    ew=np.asarray(data['event_weight'],dtype=float)
    if lw.shape != sa.shape[:2] or np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any(1).all():
        raise ValueError('log_ratio must be [N,K], no NaN/+inf, >=1 positive-weight candidate per event')
    if ew.shape != (n,) or not np.isfinite(ew).all() or (ew<0).any() or ew.sum()<=0:
        raise ValueError('event_weight must be finite nonnegative [N], positive total; signed MC unsupported')
    # Numerically stable exponentiation; no clipping or tempering.
    w=np.exp(lw-(lw.max(axis=1,keepdims=True) if mode=='conditional' else lw.max()))
    if mode=='conditional':
        w /= w.sum(1,keepdims=True)
    else:
        w /= sa.shape[1]
    weighted=ew[:,None]*w
    f=features(sa,sb,kappas)
    numerator=np.stack((ew[:,None]*features(ta,tb,kappas), ew[:,None]*f.mean(1), (weighted[...,None]*f).sum(1)),axis=1)
    denominator=np.stack((ew,ew,weighted.sum(1)),axis=1)
    health={
        'candidate_ess_fraction':float(weighted.sum()**2/(weighted.size*(weighted**2).sum())),
        'event_ess_fraction':float(weighted.sum()**2/(n*(weighted.sum(1)**2).sum())),
        'max_candidate_mass':float(weighted.max()/weighted.sum()),
        'max_event_mass':float(weighted.sum(1).max()/weighted.sum()),
    }
    return numerator,denominator,health


def contrasts(m):
    return dict(C_frobenius_error_change=float(np.linalg.norm(m[2]-m[0])-np.linalg.norm(m[1]-m[0])))


def _comparison_results(m, matrices):
    """Summarize three arms with a common event bootstrap, in truth/U/R order."""
    labels=('truth','unweighted','reweighted')
    results={label:dict(C=v.reshape(3,3).tolist(),C_minus_truth=(v-m[0]).reshape(3,3).tolist(),
                        frobenius_error=float(np.linalg.norm(v-m[0]))) for label,v in zip(labels,m)}
    changes=np.linalg.norm(matrices[:,2]-matrices[:,0],axis=1)-np.linalg.norm(matrices[:,1]-matrices[:,0],axis=1)
    comparison={'C_frobenius_error_change':dict(value=contrasts(m)['C_frobenius_error_change'],
                                               ci95=np.quantile(changes,[.025,.975]).tolist())}
    for i,label in enumerate(labels):
        results[label]['C_ci95']=np.quantile(matrices[:,i],[.025,.975],axis=0).reshape(2,3,3).tolist()
        results[label]['C_minus_truth_ci95']=np.quantile(matrices[:,i]-matrices[:,0],[.025,.975],axis=0).reshape(2,3,3).tolist()
    comparison['absolute_error_change_matrix']=(np.abs(m[2]-m[0])-np.abs(m[1]-m[0])).reshape(3,3).tolist()
    comparison['absolute_error_change_matrix_ci95']=np.quantile(np.abs(matrices[:,2]-matrices[:,0])-np.abs(matrices[:,1]-matrices[:,0]),[.025,.975],axis=0).reshape(2,3,3).tolist()
    return dict(results=results,comparison=comparison)


def analyze(data,kappas,mode,bootstrap=1000,seed=42,*,references=None):
    kappas=np.asarray(kappas,dtype=float)
    if kappas.shape not in ((2,),(len(data['event_id']),2)) or not np.isfinite(kappas).all() or (np.abs(kappas)>1).any() or (kappas==0).any():
        raise ValueError('Require finite nonzero analyzing powers [2] or [N,2] in [-1,1]')
    if mode not in ('joint','conditional') or bootstrap<20:
        raise ValueError('Specify weight mode and >=20 bootstrap replicates')
    num,den,health=prepare(data,kappas,mode)
    # Optional alternative truth maps share the SAME events, base weights and
    # bootstrap indices. They never receive the candidate density-ratio weight.
    references=references or {}
    for name,(a,b) in references.items():
        if name in ('truth','unweighted','reweighted'): raise ValueError('Reserved reference name')
        for value in (a,b):
            value=np.asarray(value)
            if value.shape!=(len(num),3) or not np.isfinite(value).all() or not np.allclose((value*value).sum(-1),1,atol=1e-5):
                raise ValueError(f'Invalid reference directions: {name}')
        ew=np.asarray(data['event_weight'],dtype=float)
        ref_num=ew[:,None]*features(np.asarray(a),np.asarray(b),kappas)
        num=np.concatenate((num,ref_num[:,None]),axis=1)
        den=np.concatenate((den,ew[:,None]),axis=1)
    m=num.sum(0)/den.sum(0)[:,None]
    rng=np.random.default_rng(seed)
    matrices=[]
    for _ in range(bootstrap):
        idx=rng.integers(0,len(num),len(num))  # Paired event clusters, never independent candidate resampling.
        d=den[idx].sum(0)
        if (d<=0).any():
            raise ValueError('Bootstrap has zero total weight; effective sample support insufficient')
        bm=num[idx].sum(0)/d[:,None]
        matrices.append(bm)
    matrices=np.asarray(matrices)
    primary=_comparison_results(m[:3],matrices[:,:3])
    reference_reports={}
    for i,name in enumerate(references,3):
        reference_reports[name]=_comparison_results(m[[i,1,2]],matrices[:,[i,1,2]])
        reference_reports[name]['reference_minus_stored_truth']=dict(
            C=(m[i]-m[0]).reshape(3,3).tolist(),
            ci95=np.quantile(matrices[:,i]-matrices[:,0],[.025,.975],axis=0).reshape(2,3,3).tolist(),
            frobenius=float(np.linalg.norm(m[i]-m[0])))
    return dict(events=len(num),candidates=data['sample_a'].shape[1],axes=['k','r','n'],kappas=(kappas.tolist() if kappas.ndim==1 else {'mode':'per_event_inverse_product','min':kappas.min(0).tolist(),'max':kappas.max(0).tolist()}),weight_mode=mode,
        **primary,reference_comparisons=reference_reports,weight_health=health,bootstrap=bootstrap,seed=seed,
        interpretation='Negative error change means closer to fixed truth. CIs condition on fixed fitted weights and generated candidates; no classifier-refit uncertainty. Matrix intervals are pointwise, not simultaneous.',
        limitations=['Angular moments require valid analyzing powers and acceptance treatment.', 'No detector unfolding or entanglement claim.'])


def plot_report(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['truth','unweighted','reweighted']
    arrays=[np.asarray(report['results'][s]['C']) for s in labels]
    limit=max(float(np.abs(a).max()) for a in arrays) or 1
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    def panel(ax,a,title,lim):
        im=ax.imshow(a,vmin=-lim,vmax=lim,cmap='RdBu_r')
        for i in range(3):
            for j in range(3): ax.text(j,i,f'{a[i,j]:+.3f}',ha='center',va='center',color='white' if abs(a[i,j])>.6*lim else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['k','r','n'],yticklabels=['k','r','n'],title=title,xlabel='B axis',ylabel='A axis')
        fig.colorbar(im,ax=ax,shrink=.75)
    display_labels=[report.get('reference_label','truth'),'unweighted','reweighted']
    for ax,a,label in zip(axes[0],arrays,display_labels): panel(ax,a,label,limit)
    diffs=[arrays[1]-arrays[0],arrays[2]-arrays[0],np.abs(arrays[2]-arrays[0])-np.abs(arrays[1]-arrays[0])]
    lim=max(float(np.abs(a).max()) for a in diffs) or 1
    for ax,a,title in zip(axes[1],diffs,['Unweighted − reference','Reweighted − reference','Absolute error change (negative better)']): panel(ax,a,title,lim)
    fig.suptitle(report.get('plot_title','Paired Cij moment comparison — no concurrence / no unfolding'))
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kappas',type=float,nargs=2,required=True)
    p.add_argument('--weight-mode',choices=['joint','conditional'],required=True)
    p.add_argument('--physics-definition',required=True,help='Process, charge/axis convention, decay polarimeter and acceptance treatment')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    with np.load(args.input,allow_pickle=False) as data:
        report=analyze(data,args.kappas,args.weight_mode,args.bootstrap,args.seed)
    report.update(input=str(args.input.resolve()),physics_definition=args.physics_definition)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot_report(report,args.output.with_suffix('.png'))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
