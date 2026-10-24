"""Raw-policy diagnostics on the saved K8 cache; unfolding is a separate CPU stage."""
from pathlib import Path
import numpy as np
import torch
from experiments.ttbar_general_classifier.run import parts
from experiments.ttbar_general_classifier.core import nu_p4,reconstructed_visible,mass,wrap
from experiments.ttbar_dgpo_general.reward import log_mixture


def summarize(path,reward):
    sums={};counts={};hist={};bin_edges={};n=0;selected=0
    def mean(key,values):
        x=np.asarray(values,dtype=float);sums[key]=sums.get(key,0.)+x.sum();counts[key]=counts.get(key,0)+x.size
    def distribution(key,t,p,edges):
        # Both edge bins include tails; event weights sum to one for each K8 pool.
        ht=np.histogram(t,edges)[0];hp=np.histogram(p,edges)[0]/p.shape[1]
        bin_edges[key]=edges
        if key not in hist:hist[key]=[ht.astype(float),hp.astype(float)]
        else:hist[key][0]+=ht;hist[key][1]+=hp
    for _,_,a in parts(path,'validation'):
        b=len(a['truth']);n+=b;dev=reward.device
        co=torch.tensor(np.array(a['condition']),device=dev);la=torch.tensor(np.array(a['labels']),device=dev)
        f=torch.tensor(np.array(a['generated_features']),device=dev);k=f.shape[1]
        with torch.no_grad():
            logits=reward.head(co.repeat_interleave(k,0),((f-reward.mean)/reward.scale).reshape(b*k,-1),la.repeat_interleave(k)).reshape(b,k).T
            logw,logq,_=log_mixture(logits,torch.tensor(np.array(a['candidates']),device=dev).permute(1,0,2,3),
                torch.tensor(np.array(a['truth']),device=dev),reward.stats['oracle_scale'],reward.alpha,reward.temperature)
        mean('val/reward/mean',logw.cpu().numpy())
        mean('val/mixture/ess',(1/logw.exp().square().sum(0)).cpu().numpy())
        mean('val/general/ess',(1/logq.exp().square().sum(0)).cpu().numpy())
        keep=np.asarray(a['physics_keep']).astype(bool)&(a['partition']!=0);selected+=int(keep.sum())
        if not keep.any():continue
        truth=np.array(a['truth'][keep]);pred=np.array(a['candidates'][keep])
        for leg,name in enumerate(('nu','nubar')):
            for j,obs in enumerate(('pt','eta','phi')):
                t=truth[:,leg,j];p=pred[:,:,leg,j]
                if j==0:t,p=np.expm1(t),np.expm1(p)
                if j==2:t,p=wrap(t),wrap(p)
                lo,hi,nb=(0,150,30) if j==0 else ((-4,4,20) if j==1 else (-np.pi,np.pi,16))
                edges=np.linspace(lo,hi,nb+1);edges[0],edges[-1]=-np.inf,np.inf
                distribution(f'{name}_{obs}',t,p,edges)
                spread=1-np.abs(np.exp(1j*p).mean(1)) if j==2 else p.std(1)
                mean(f'val/conditional_spread/{name}_{obs}',spread)
            visible=reconstructed_visible({key:np.asarray(v)[keep] for key,v in a.items() if key.startswith('EXTRA/')})
            t=mass(visible+nu_p4(truth))[:,leg];p=mass(visible[:,None]+nu_p4(pred))[:,:,leg]
            edges=np.linspace(100,240,29);edges[0],edges[-1]=-np.inf,np.inf
            distribution(f'top{leg+1}_mass_reco_truthnu',t,p,edges)
    out={key:float(value/counts[key]) for key,value in sums.items()}
    for key,(t,p) in hist.items():
        out[f'val/distribution/{key}/TV']=float(.5*np.abs(t/t.sum()-p/p.sum()).sum())
    out.update({'val/events':n,'val/physics_events':selected,'val/winrate':0.,'val/winrate_evaluated':0.,'val/oracle_fraction':reward.alpha})
    np.savez(Path(path)/'raw-distributions.npz',**{f'{key}_{which}':values[i] for key,values in hist.items() for i,which in enumerate(('truth','prediction'))},**{key+'_edges':edges for key,edges in bin_edges.items()})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    for ax,(key,(t,p)) in zip(axes.flat,hist.items()):
        edges=bin_edges[key].copy();edges[0]=edges[1]-(edges[2]-edges[1]);edges[-1]=edges[-2]+(edges[-2]-edges[-3])
        ax.stairs(t/t.sum(),edges,label='Reference');ax.stairs(p/p.sum(),edges,label='DGPO')
        ax.set_title(key.replace('_',' '));ax.set_ylabel('Probability');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(Path(path)/'distributions.png',dpi=150);plt.close(fig)
    return out
