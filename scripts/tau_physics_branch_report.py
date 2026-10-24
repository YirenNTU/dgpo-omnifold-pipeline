"""Held-out fixed-candidate physics comparison with joint ratio weights."""
from pathlib import Path
import json
import numpy as np
from scripts.tau_tail_attribution import candidate_weights
from scripts.diagnose_ztautau_cij import angles,tau_from_deltas
from evenet_dgpo.RL.DGPO_neutrino.diagnostics.tau_marginals import observables,probability,distances

AXES=('kk','kr','kn','rk','rr','rn','nk','nr','nn')
DIAG=np.array([0,4,8])
OFF=np.array([1,2,3,5,6,7])


def terms(panel,delta):
    a,b=angles(tau_from_deltas(panel['visible_a'],delta[:,0]),
               tau_from_deltas(panel['visible_b'],delta[:,1]),panel['visible_a'],panel['visible_b'])
    k=np.asarray(panel['kappas'],float).prod(1)
    if not np.isfinite(k).all() or (k==0).any(): raise ValueError('Invalid analyzing powers')
    return 9*(a[:,:,None]*b[:,None,:]).reshape(-1,9)/k[:,None]


def errors(value,target):
    d=value-target
    return dict(error=float(np.linalg.norm(d)),diagonal_error=float(np.linalg.norm(d[DIAG])),
                offdiagonal_error=float(np.linalg.norm(d[OFF])),absolute_errors=np.abs(d).tolist())


def summarize(panel,generated,logits,bootstrap=1000,seed=42,*,comparison_pairs=None,primary=None):
    """All candidates share each whole-event bootstrap draw; no per-event softmax."""
    w=np.asarray(panel['event_weight'],float); t=terms(panel,panel['truth_deltas'])
    n,k=generated.shape[:2]
    if generated.shape!=(len(w),k,2,2) or k<1: raise ValueError('Invalid candidate layout')
    if bootstrap<20: raise ValueError('At least20 paired bootstrap draws required')
    g=np.stack([terms(panel,generated[:,j]) for j in range(k)],1)
    if not np.isfinite(t).all() or not np.isfinite(g).all(): raise ValueError('Nonfinite moments')
    if 'unweighted' in logits:raise ValueError('Unweighted baseline is reserved')
    scores={'unweighted':np.zeros((n,k)),**logits}
    pairs=comparison_pairs if comparison_pairs is not None else [
        ('physics','control'),('physics','unweighted'),('control','unweighted')]
    if not pairs or any(a not in scores or b not in scores or a==b for a,b in pairs):
        raise ValueError('Invalid comparison arms')
    weights={};results={}; nums={};dens={}
    target=np.average(t,weights=w,axis=0)
    for name,s in scores.items():
        if np.shape(s)!=(n,k): raise ValueError('Score/candidate alignment differs')
        if name!='unweighted' and np.max(s)>np.log(30)+1e-5: raise ValueError('Reward cap changed')
        q,log_mean=candidate_weights(w,s);weights[name]=q
        value=np.einsum('nk,nkd->d',q,g); event=q.sum(1)
        results[name]=dict(C=value.reshape(3,3).tolist(),**errors(value,target),
            event_ess=float(1/np.square(event).sum()),candidate_ess=float(1/np.square(q).sum()),
            max_event_mass=float(event.max()),top1pct_event_mass=float(np.sort(event)[-max(1,int(np.ceil(n*.01))):].sum()),
            log_mean_ratio=log_mean,cap_fraction=float((np.exp(s)>29.7).mean()))
        # Multiplicative scaling cancels in each bootstrap arm's global ratio.
        nums[name]=np.einsum('nk,nkd->nd',q,g);dens[name]=event
    rng=np.random.default_rng(seed)
    contrasts={a+'_minus_'+b:[] for a,b in pairs}
    for _ in range(bootstrap):
        idx=rng.integers(0,n,n)
        if w[idx].sum()<=0: raise ValueError('Bootstrap draw has zero truth weight')
        target_b=np.average(t[idx],weights=w[idx],axis=0)
        e={}
        for name in scores:
            value=nums[name][idx].sum(0)/dens[name][idx].sum()
            row=errors(value,target_b)
            e[name]=np.r_[row['error'],row['diagonal_error'],row['offdiagonal_error'],row['absolute_errors']]
        for left,right in pairs:
            contrasts[left+'_minus_'+right].append(e[left]-e[right])
    comparisons={}
    fields=['error','diagonal_error','offdiagonal_error']+['abs_'+x for x in AXES]
    for name,draws in contrasts.items():
        left,right=name.split('_minus_')
        la,ra=results[left],results[right]
        delta=np.r_[*[la[x]-ra[x] for x in fields[:3]],np.array(la['absolute_errors'])-ra['absolute_errors']]
        ci=np.quantile(draws,[.025,.975],axis=0)
        comparisons[name]={key:dict(delta=float(delta[i]),ci95=ci[:,i].tolist()) for i,key in enumerate(fields)}
    groups={}
    for category in np.unique(panel['category']):
        take=panel['category']==category
        if w[take].sum()<=0: continue
        tar=np.average(t[take],weights=w[take],axis=0)
        group={'events':int(take.sum()),'truth_mass':float(w[take].sum()/w.sum()),'truth_C':tar.reshape(3,3).tolist()}
        for name,q in weights.items():
            mass=q[take].sum();value=np.einsum('nk,nkd->d',q[take],g[take])/mass
            group[name]=dict(mass=float(mass),C=value.reshape(3,3).tolist(),**errors(value,tar))
        groups[str(category)]=group
    return dict(truth_C=target.reshape(3,3).tolist(),arms=results,comparisons=comparisons,
        categories=groups,events=n,candidates=k,bootstrap=bootstrap,
        primary=primary or 'physics_minus_control/error; improvement requires upper CI95 < 0',
        secondary='all nine errors, offdiagonal norm, distributions, channel mass and ESS',
        scope='Matched fixed-energy analyzer moments; not certified full polarimeter/entanglement closure.',
        uncertainty='Paired whole-event pointwise bootstrap, fixed fitted heads/candidates; excludes fit uncertainty. Previously inspected external population.'),weights


def distribution_report(panel,generated,weights):
    t=observables(panel,panel['truth_deltas']); k=generated.shape[1]
    series=[observables(panel,generated[:,j]) for j in range(k)]
    # Include the spin-sensitive directions' nine analysis contributions.
    truth_terms=terms(panel,panel['truth_deltas'])
    gen_terms=np.stack([terms(panel,generated[:,j]) for j in range(k)],1)
    for i,key in enumerate(AXES):
        t['moments/'+key]=truth_terms[:,i]
        for j in range(k):series[j]['moments/'+key]=gen_terms[:,j,i]
    records={}
    w=panel['event_weight']
    for name,target in t.items():
        values=np.stack([s[name] for s in series],1)
        fixed=np.r_[target,values.ravel()]
        if name.endswith('_phi'):lo,hi=-np.pi,np.pi
        elif name.startswith('reco/'):lo,hi=0.,np.pi
        else:
            lo,hi=np.quantile(fixed,[.005,.995]);pad=max((hi-lo)*.05,1e-6);lo-=pad;hi+=pad
        edges=np.linspace(lo,hi,61)
        p=probability(target,w,edges)
        record=dict(edges=edges.tolist(),truth=p.tolist(),arms={})
        for arm,q in weights.items():
            prob=probability(values.ravel(),q.ravel(),edges)
            record['arms'][arm]=dict(probability=prob.tolist(),**distances(prob,p),outside=float(prob[0]+prob[-1]))
        records[name]=record
    # Resolve the narrow topology core, which broad original bins can hide.
    for name,transform in [('topology/cos_opening',lambda x:np.log10(np.maximum(1+x,1e-12))),
                           ('topology/delta_phi_to_pi',lambda x:np.log10(np.maximum(x,1e-12)))]:
        if name not in t:continue
        edges=np.linspace(-12,1,105);target=transform(t[name]);values=transform(np.stack([s[name] for s in series],1))
        p=probability(target,w,edges)
        records[name+'_log10']=dict(edges=edges.tolist(),truth=p.tolist(),arms={
            arm:dict(probability=(prob:=probability(values.ravel(),q.ravel(),edges)).tolist(),
                     **distances(prob,p),outside=float(prob[0]+prob[-1])) for arm,q in weights.items()})
    return records


def write_report(output,panel,generated,logits,bootstrap,seed,*,comparison_pairs=None,primary=None):
    output=Path(output)
    report,weights=summarize(panel,generated,logits,bootstrap,seed,comparison_pairs=comparison_pairs,primary=primary)
    distributions=distribution_report(panel,generated,weights)
    report['distributions']={name:{arm:{k:v for k,v in data.items() if k!='probability'}
        for arm,data in record['arms'].items()} for name,record in distributions.items()}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'histograms.json').write_text(json.dumps(distributions,allow_nan=False)+'\n')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    paths=[]
    for group in ('target','reco','topology','moments'):
        names=[x for x in distributions if x.startswith(group+'/')]
        if not names:continue
        fig=Figure(figsize=(12,3.4*((len(names)+1)//2)),layout='constrained');FigureCanvasAgg(fig)
        axes=fig.subplots((len(names)+1)//2,2,squeeze=False).ravel()
        for ax,name in zip(axes,names):
            item=distributions[name];edges=np.array(item['edges'])
            ax.stairs(np.array(item['truth'])[1:-1]/np.diff(edges),edges,label='Truth',color='black')
            for arm,data in item['arms'].items():
                ax.stairs(np.array(data['probability'])[1:-1]/np.diff(edges),edges,label=arm)
            ax.set_title(name);ax.set_ylabel('Probability density');ax.legend()
        for ax in axes[len(names):]:ax.set_visible(False)
        path=output/(group+'.png');fig.savefig(path,dpi=140);paths.append(path)
    arm_names=list(report['arms'])
    cols=len(arm_names) if len(arm_names)<=4 else 3
    rows=(len(arm_names)+cols-1)//cols
    fig=Figure(figsize=(4*cols,4*rows),layout='constrained');FigureCanvasAgg(fig)
    ds=[np.array(report['arms'][arm]['C'])-report['truth_C'] for arm in arm_names]
    scale=max(float(np.abs(d).max()) for d in ds) or 1.
    axes=fig.subplots(rows,cols,squeeze=False).ravel()
    for ax,arm,d in zip(axes,arm_names,ds):
        im=ax.imshow(d,vmin=-scale,vmax=scale,cmap='coolwarm');ax.set_title(arm+' minus truth')
        ax.set_xticks(range(3),list('krn'));ax.set_yticks(range(3),list('krn'))
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{d[i,j]:.3f}',ha='center',va='center')
        fig.colorbar(im,ax=ax)
    for ax in axes[len(arm_names):]:ax.set_visible(False)
    path=output/'cij_errors.png';fig.savefig(path,dpi=140);paths.append(path)
    return report,paths
