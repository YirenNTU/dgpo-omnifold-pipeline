"""Matched cold Fourier audits of source and both fixed-DGPO endpoints."""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import conditional as native
from .cube_lockdown import ConditionDenoiser
from .nonperiodic_cube import Critic,Data,panel,classification
from .truth_pretrain import atomic_json,atomic_checkpoint


@torch.no_grad()
def scores(model,p):
    return {key:torch.cat([model(y,c) for y,c in zip(p[key].split(1024),p['c'].split(1024))])
            for key in ('positive','negative')}


def auc_contrasts(predictions,repeats=300):
    """Paired-context bootstrap, preserving truth/generated dependence and arm pairing."""
    arrays={k:{s:v.numpy() for s,v in d.items()} for k,d in predictions.items()}
    n=len(arrays['baseline']['positive']);labels=np.r_[np.ones(n),np.zeros(n)]
    def auc(d,ids):return roc_auc_score(labels,np.r_[d['positive'][ids],d['negative'][ids]])
    full=np.arange(n);original={k:auc(d,full) for k,d in arrays.items()}
    rng=np.random.default_rng(98117)
    draws={k:[] for k in ('raw','fourier')}
    for _ in range(repeats):
        ids=rng.integers(0,n,n);base=auc(arrays['baseline'],ids)
        for k in draws:draws[k].append(auc(arrays[k],ids)-base)
    return {k:{'delta_auc':original[k]-original['baseline'],
               'lo95':float(np.quantile(v,.025)),'hi95':float(np.quantile(v,.975)),
               'scope':'paired-context percentile bootstrap; conditional on fitted classifiers'}
            for k,v in draws.items()}


def run(source,output,max_steps=16000):
    metadata=json.loads((source/'report.json').read_text())
    if metadata['state']!='completed':raise ValueError('Need completed DGPO pair')
    original=Path(metadata['source'])
    saved=torch.load(original/'reference.pt',map_location='cpu',weights_only=True)
    cfg=native.Config(**saved['config']);data=Data(cfg)
    base=native.Denoiser(cfg);base.load_state_dict(saved['model']);base.eval()
    policies={'baseline':base}
    for basis in ('raw','fourier'):
        saved_arm=torch.load(source/f'{basis}_last.pt',map_location='cpu',weights_only=True)
        if saved_arm['step']!=metadata['arms'][basis]['steps']:raise ValueError('Endpoint step mismatch')
        m=ConditionDenoiser(cfg,basis,base.state_dict());m.load_state_dict(saved_arm['model']);m.eval()
        policies[basis]=m
    output.mkdir(parents=True,exist_ok=False)
    report={'state':'generating_panels','source':str(source.resolve()),'max_steps':max_steps,
            'seed':41,'history':[],'cold_start':True,'primary':'fresh test AUC and balanced BCE',
            'panel_seeds':[910041,920041,930041]}
    atomic_json(output/'report.json',report)
    panels={}
    for name,m in policies.items():
        panels[name]={split:panel(data,n,seed,m) for split,n,seed in
            [('train',32768,910041),('validation',8192,920041),('test',16384,930041)]}
        atomic_checkpoint(output/f'{name}_panels.pt',panels[name])
    for name in ('raw','fourier'):
        for split in panels[name]:
            assert torch.equal(panels[name][split]['c'],panels['baseline'][split]['c'])
            assert torch.equal(panels[name][split]['positive'],panels['baseline'][split]['positive'])
    torch.manual_seed(41);initial=Critic(True)
    models={k:copy.deepcopy(initial) for k in policies}
    opts={k:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.001) for k,m in models.items()}
    best={k:math.inf for k in models};anchor=best.copy();stale={k:0 for k in models}
    weights={};selected={};rng=native.generator(940041)
    report['state']='training_cold_audits'
    for step in range(1,max_steps+1):
        ids=torch.randint(32768,(256,),generator=rng)
        train_losses={}
        for name,m in models.items():
            p=panels[name]['train'];c=torch.cat([p['c'][ids]]*2)
            y=torch.cat([p['positive'][ids],p['negative'][ids]])
            loss=F.binary_cross_entropy_with_logits(m(y,c),torch.cat([torch.ones(256),torch.zeros(256)]))
            opts[name].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opts[name].step()
            train_losses[name]=float(loss.detach())
        if step%100==0 or step==max_steps:
            row={'step':step,'arms':{}}
            for name,m in models.items():
                stats,_=classification(m,panels[name]['validation']);score=stats['bce']
                if score<best[name]:best[name]=score;selected[name]=step;weights[name]=copy.deepcopy(m.state_dict())
                if score<anchor[name]-1e-4:anchor[name]=score;stale[name]=0
                else:stale[name]+=1
                row['arms'][name]={**stats,'train_minibatch_bce':train_losses[name],
                    'selected_step':selected[name],'stale_checks':stale[name]}
            report['history'].append(row);report['step']=step
            atomic_json(output/'report.json',report)
            print(json.dumps(row),flush=True)
            if step>=2000 and min(stale.values())>=20:break
    report['test']={};predictions={}
    for name,m in models.items():
        atomic_checkpoint(output/f'{name}_training_state.pt',{'model':m.state_dict(),
            'optimizer':opts[name].state_dict(),'rng':rng.get_state(),'step':step,
            'best_model':weights[name],'selected_step':selected[name]})
        m.load_state_dict(weights[name]);m.eval()
        stats,_=classification(m,panels[name]['test']);predictions[name]=scores(m,panels[name]['test'])
        report['test'][name]={**stats,'selected_step':selected[name],'plateau':stale[name]>=20}
        atomic_checkpoint(output/f'{name}_best.pt',{'model':m.state_dict(),'selected_step':selected[name]})
    atomic_checkpoint(output/'test_scores.pt',predictions)
    report['auc_changes']=auc_contrasts(predictions)
    report['state']='completed' if min(stale.values())>=20 else 'budget_exhausted_inconclusive'
    atomic_json(output/'report.json',report)
    print(json.dumps({'state':report['state'],'test':report['test'],'auc_changes':report['auc_changes']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);run(args.source,args.output)
