"""Conditional mode/shape swaps. No diffusion updates; optional fresh audit fits."""
import argparse
import copy
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from . import conditional as native
from .cube_lockdown import ConditionDenoiser
from .nonperiodic_cube import Critic,Data,classification
from .truth_pretrain import atomic_json,atomic_checkpoint


def modes(y):
    s=(y>=0).long()
    return s[...,0]*4+s[...,1]*2+s[...,2]


def truth_shape(ids,centers,rng):
    """Gaussian shape restricted to its mode; crossing mass <1e-10 in this toy."""
    center=centers[ids];y=center+.15*torch.randn(center.shape,generator=rng)
    bad=y*center<=0
    while bad.any():
        y[bad]=center[bad]+.15*torch.randn(int(bad.sum()),generator=rng)
        bad=y*center<=0
    return y


@torch.no_grad()
def swaps(model,data,grid,n,donors,seed,*,return_pool=False):
    if not 1 <= n <= donors:
        raise ValueError('Require 1 <= samples per condition <= donors')
    rng=native.generator(seed)
    c=((torch.arange(grid)+.5)/grid*2-1)[:,None]
    z=torch.randn(grid,donors,3,generator=rng)
    pool=torch.cat([native.ddim(model,cc[:,None],zz,50)
                   for cc,zz in zip(c.split(8),z.split(8))])
    if not torch.isfinite(pool).all():raise FloatingPointError('Nonfinite donor pool')
    ids=modes(pool)
    counts=torch.stack([torch.bincount(row,minlength=8) for row in ids])
    if int(counts.min())<16:
        raise ValueError(f'Sparse condition/mode cell: minimum={int(counts.min())}; no fallback mixing')
    # Separate, common RNG makes A, truth positives, and desired C modes identical across policies.
    trng=native.generator(seed+100000)
    p=data.probabilities(c[:,0],.9)
    target_ids=torch.multinomial(p,n,replacement=True,generator=trng)
    positive_ids=torch.multinomial(p,n,replacement=True,generator=trng)
    a=truth_shape(target_ids,data.centers,trng)
    positive=truth_shape(positive_ids,data.centers,trng)
    d=pool[:,:n].clone()
    b=truth_shape(ids[:,:n],data.centers,native.generator(seed+200000))
    shaped=torch.empty_like(a)
    for j in range(grid):
        for mode in range(8):
            slots=(target_ids[j]==mode).nonzero().flatten()
            choices=(ids[j]==mode).nonzero().flatten()
            draws=torch.randint(len(choices),(len(slots),),generator=rng)
            shaped[j,slots]=pool[j,choices[draws]]
    assert torch.equal(modes(a),modes(shaped)) and torch.equal(modes(b),modes(d))
    cc=c[:,None].expand(-1,n,-1).reshape(-1,1)
    panels={name:{'c':cc,'positive':positive.reshape(-1,3),'negative':y.reshape(-1,3)}
            for name,y in [('A',a),('B',b),('C',shaped),('D',d)]}
    health={'min_cell_count':int(counts.min()),'counts':counts.tolist(),
            'grid':grid,'samples_per_condition':n,'donors_per_condition':donors}
    if return_pool:
        return panels,health,{'c':c,'y':pool,'ids':ids,'counts':counts,'p':p}
    return panels,health


@torch.no_grad()
def reward_values(critic,p,grid):
    return torch.cat([critic(y,c) for y,c in zip(p['negative'].split(2048),p['c'].split(2048))]).reshape(grid,-1)


def fit_audits(panels,output,emit,max_steps=16000):
    # One common A sanity control; three B/C/D families. Shared cold initialization.
    keys=['baseline/A']+[f'{policy}/{swap}' for policy in ('baseline','raw','fourier') for swap in ('B','C','D')]
    torch.manual_seed(73);initial=Critic(True)
    models={k:copy.deepcopy(initial) for k in keys}
    opts={k:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.001) for k,m in models.items()}
    best={k:math.inf for k in keys};anchor=best.copy();stale={k:0 for k in keys};selected={};weights={}
    rng=native.generator(77317)
    for step in range(1,max_steps+1):
        ids=torch.randint(len(panels['baseline']['train']['A']['c']),(256,),generator=rng)
        for key,m in models.items():
            policy,swap=key.split('/');p=panels[policy]['train'][swap]
            c=torch.cat([p['c'][ids]]*2);y=torch.cat([p['positive'][ids],p['negative'][ids]])
            loss=F.binary_cross_entropy_with_logits(m(y,c),torch.cat([torch.ones(256),torch.zeros(256)]))
            opts[key].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opts[key].step()
        if step%100==0:
            stats={}
            for key,m in models.items():
                policy,swap=key.split('/');s,_=classification(m,panels[policy]['validation'][swap]);v=s['bce']
                if v<best[key]:best[key]=v;weights[key]=copy.deepcopy(m.state_dict());selected[key]=step
                if v<anchor[key]-1e-4:anchor[key]=v;stale[key]=0
                else:stale[key]+=1
                stats[key]={**s,'stale_checks':stale[key],'selected_step':selected[key]}
            emit({'phase':'fresh_swap_audits','step':step,'validation':stats})
            if step>=2000 and min(stale.values())>=20:break
    results={}
    for key,m in models.items():
        m.load_state_dict(weights[key]);policy,swap=key.split('/')
        results[key]=classification(m,panels[policy]['test'][swap])[0]
        results[key].update(selected_step=selected[key],plateau=stale[key]>=20,fit_steps=step)
        atomic_checkpoint(output/(key.replace('/','_')+'_audit.pt'),{'model':m.state_dict(),'selected_step':selected[key]})
    return results


def run(source,audit,output,grid=128,fresh=False):
    meta=json.loads((source/'report.json').read_text())
    if meta['state']!='completed':raise ValueError('DGPO source incomplete')
    lineage=Path(meta['source']);s=torch.load(lineage/'reference.pt',weights_only=True,map_location='cpu')
    cfg=native.Config(**s['config']);data=Data(cfg)
    initial=native.Denoiser(cfg);initial.load_state_dict(s['model']);initial.eval()
    policies={'baseline':initial}
    for basis in ('raw','fourier'):
        m=ConditionDenoiser(cfg,basis,initial.state_dict())
        ck=torch.load(source/f'{basis}_last.pt',weights_only=True,map_location='cpu')
        if ck['step']!=meta['arms'][basis]['steps']:raise ValueError('Endpoint mismatch')
        m.load_state_dict(ck['model']);policies[basis]=m.eval()
    fixed=Critic(True);fixed.load_state_dict(torch.load(lineage/'actual/fourier_classifier.pt',weights_only=True,map_location='cpu')['model']);fixed.eval()
    judges={}
    for name in policies:
        m=Critic(True);m.load_state_dict(torch.load(audit/f'{name}_best.pt',weights_only=True,map_location='cpu')['model']);judges[name]=m.eval()
    output.mkdir(parents=True,exist_ok=False)
    report={'state':'constructing_swaps','grid':grid,'source':str(source.resolve()),'audit_source':str(audit.resolve()),
            'scope':'Fixed-condition-grid empirical conditional shape swaps; existing judges are NOT fresh swap fits',
            'cells':{},'scores':{},'fresh_audits_requested':fresh}
    def emit(row):
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        report['active']=row;atomic_json(output/'report.json',report)
        print(json.dumps(row),flush=True)
    panels={};rewards={}
    splits=[('test',256,1024,161017)]
    if fresh:splits=[('train',256,1024,141017),('validation',128,1024,151017)]+splits
    for name,m in policies.items():
        panels[name]={}
        for split,n,donors,seed in splits:
            p,health=swaps(m,data,grid,n,donors,seed);panels[name][split]=p
            report['cells'][f'{name}/{split}']=health
            atomic_checkpoint(output/f'{name}_{split}_swaps.pt',p)
            emit({'phase':'swaps_ready','policy':name,'split':split,'min_cell_count':health['min_cell_count']})
        rewards[name]={}
        report['scores'][name]={}
        for swap,p in panels[name]['test'].items():
            r=reward_values(fixed,p,grid);rewards[name][swap]=r
            stats,_=classification(judges[name],p)
            report['scores'][name][swap]={'fixed_reward_mean':float(r.mean()),'existing_audit_judge':stats}
        r=rewards[name]
        report['scores'][name]['reward_decomposition']={
            'mode_effect_at_truth_shape':float((r['B']-r['A']).mean()),
            'shape_effect_at_model_modes':float((r['D']-r['B']).mean()),
            'shape_effect_at_truth_modes':float((r['C']-r['A']).mean()),
            'interaction':float((r['D']-r['B']-r['C']+r['A']).mean())}
    report['reward_change_decomposition']={}
    for name in ('raw','fourier'):
        a,b=rewards[name],rewards['baseline']
        total=float((a['D']-b['D']).mean());mode=float((a['B']-b['B']).mean())
        shape=float(((a['D']-a['B'])-(b['D']-b['B'])).mean())
        report['reward_change_decomposition'][name]={'total':total,'mode_at_truth_shape':mode,'shape_at_model_modes':shape,
            'identity_residual':total-mode-shape,'not_causal_training_attribution':True}
    atomic_checkpoint(output/'fixed_reward_arrays.pt',rewards)
    if fresh:
        report['state']='fitting_fresh_swap_audits'
        report['fresh_audits']=fit_audits(panels,output,emit)
    report['state']='completed';atomic_json(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid',type=int,choices=(64,128),default=128)
    p.add_argument('--fit-audits',action='store_true')
    a=p.parse_args();torch.set_num_threads(2)
    r=run(a.source,a.audit,a.output,a.grid,a.fit_audits)
    print(json.dumps({'state':r['state'],'reward_change_decomposition':r['reward_change_decomposition']}),flush=True)
