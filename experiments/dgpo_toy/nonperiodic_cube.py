"""Gated local cube experiment: learned Fourier critic then native DGPO.

No oracle parity/target features enter either learned model.
"""
import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from . import conditional as native
from .parity_cube import CubeDistribution
from .cube_lockdown import ConditionDenoiser, verify_initial
from .truth_pretrain import atomic_json, atomic_checkpoint, noisy_target, make_panel, validation


class Data(CubeDistribution):
    def __init__(self, cfg):
        super().__init__(cfg, mass=.5, width=.15, continuous=True)
        self.condition_bins = 32

    def condition_signal(self, c):
        centers = c.new_tensor([-.77, -.31, .12, .63])
        amplitudes = c.new_tensor([1., -.85, .95, -1.])
        bumps = torch.exp(-.5*((c[...,None]-centers)/.06).square())
        return torch.tanh(2*(bumps*amplitudes).sum(-1))


class Critic(nn.Module):
    def __init__(self, fourier, width=128):
        super().__init__()
        self.fourier = fourier
        # Identical nominal architecture and initialization; plain slots inactive.
        self.net = nn.Sequential(nn.Linear(36,width), nn.GELU(), nn.Linear(width,width),
                                 nn.GELU(), nn.Linear(width,width), nn.GELU(), nn.Linear(width,1))

    def features(self, y, c):
        raw = torch.cat((c.expand(*y.shape[:-1],1), y),-1)
        phase = raw*raw.new_tensor([math.pi,math.pi/2,math.pi/2,math.pi/2])
        extras = torch.cat([f(k*phase) for k in range(1,5) for f in (torch.sin,torch.cos)],-1)
        if not self.fourier:
            extras = torch.zeros_like(extras)
        return torch.cat((raw,extras),-1)

    def forward(self, y, c, data=None):
        return self.net(self.features(y,c)).squeeze(-1)


def panel(data,n,seed,model=None):
    rng=native.generator(seed); c=data.contexts(n,rng)
    positive=data.sample(c,rng,truth=True)
    if model is None:
        negative=data.sample(c,rng)
    else:
        with torch.no_grad():
            z=torch.randn(n,3,generator=rng)
            negative=torch.cat([native.ddim(model,cc,zz,50) for cc,zz in zip(c.split(512),z.split(512))])
    return {'c':c,'positive':positive,'negative':negative}


@torch.no_grad()
def classification(model,p):
    scores={k:torch.cat([model(y,c) for y,c in zip(p[k].split(1024),p['c'].split(1024))])
            for k in ('positive','negative')}
    losses=.5*(F.softplus(-scores['positive'])+F.softplus(scores['negative']))
    return {'bce':float(losses.double().mean()),
            'auc':native.auc(scores['positive'],scores['negative']),
            'weight':native.weight_health(scores['negative'])},losses


def fit_pair(output,data,emit,seed=17,model=None,steps=6000):
    panels={name:panel(data,n,seed+offset,model) for name,n,offset in
            (('train',32768,10000),('validation',8192,20000),('test',16384,30000))}
    atomic_checkpoint(output/'classifier_panels.pt',panels)
    with torch.random.fork_rng():
        torch.manual_seed(seed); plain=Critic(False)
    models={'plain':plain,'fourier':copy.deepcopy(plain)};models['fourier'].fourier=True
    optimizers={k:torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.001) for k,m in models.items()}
    best={k:math.inf for k in models}; weights={}; selected={}; stale={k:0 for k in models}
    anchor=best.copy(); history=[]; rng=native.generator(seed+40000)
    train=panels['train']
    for step in range(1,steps+1):
        ids=torch.randint(len(train['c']),(256,),generator=rng)
        c=torch.cat([train['c'][ids]]*2)
        y=torch.cat([train['positive'][ids],train['negative'][ids]])
        labels=torch.cat([torch.ones(256),torch.zeros(256)])
        for name,m in models.items():
            loss=F.binary_cross_entropy_with_logits(m(y,c),labels)
            optimizers[name].zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizers[name].step()
        if step%100==0 or step==steps:
            row={'phase':'classifier','step':step,'arms':{}}
            for name,m in models.items():
                stats,_=classification(m,panels['validation']);score=stats['bce']
                if score<best[name]:
                    best[name]=score;weights[name]=copy.deepcopy(m.state_dict());selected[name]=step
                if score<anchor[name]-1e-4:anchor[name]=score;stale[name]=0
                else:stale[name]+=1
                row['arms'][name]={**stats,'best_step':selected[name],'stale_checks':stale[name]}
            history.append(row);emit(row)
            if step>=2000 and min(stale.values())>=20:break
    results={};losses={}
    for name,m in models.items():
        m.load_state_dict(weights[name]);m.eval().requires_grad_(False)
        results[name],losses[name]=classification(m,panels['test'])
        results[name].update(selected_step=selected[name],stale_checks=stale[name])
        atomic_checkpoint(output/f'{name}_classifier.pt',{'model':m.state_dict(),'fourier':m.fourier,'selected_step':selected[name]})
    d=(losses['fourier']-losses['plain']).double()
    mean=float(d.mean());se=float(d.std()/math.sqrt(len(d)))
    adequate=min(stale.values())>=20
    gate={'adequate_plateau_both':adequate,
          'plain_near_chance':results['plain']['bce']>=math.log(2)-.005 and results['plain']['auc']<=.55,
          'fourier_discriminates':results['fourier']['bce']<math.log(2)-.015 and results['fourier']['auc']>=.60,
          'material_bce_advantage':mean+1.96*se<-.01}
    gate['passed']=all(gate.values())
    return models['fourier'],{'test':results,'paired_bce_difference':{'mean':mean,'lo95':mean-1.96*se,'hi95':mean+1.96*se},
        'gate':gate,'steps':step,'history':history,'negative_source':'ideal_uniform_cube' if model is None else 'actual_diffusion'}


def pretrain(output,data,cfg,emit):
    splits={}
    for name,n,seed in [('train',32768,51017),('validation',8192,52017)]:
        rng=native.generator(seed);c=data.contexts(n,rng)
        splits[name]={'condition':c,'target':data.sample(c,rng)}
    atomic_checkpoint(output/'reference_dataset.pt',splits)
    torch.manual_seed(17);model=native.Denoiser(cfg)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.001)
    rng=native.generator(53017); val=make_panel(splits['validation'],54017)
    best=anchor=math.inf;stale=epoch=0;weights=None
    while stale<20:
        epoch+=1
        for ids in torch.randperm(32768,generator=rng).split(512):
            x,t,target=noisy_target(splits['train']['target'][ids],rng)
            loss=(model(x,t,splits['train']['condition'][ids])-target).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        score=validation(model,val)['velocity_mse']
        if score<best:best=score;weights=copy.deepcopy(model.state_dict())
        if score<anchor-1e-4:anchor=score;stale=0
        else:stale+=1
        emit({'phase':'pretrain','epoch':epoch,'val_mse':score,'stale_epochs':stale})
    model.load_state_dict(weights);model.eval()
    atomic_checkpoint(output/'reference.pt',{'model':weights,'config':asdict(cfg),'epochs':epoch,
        'trained_on':'uniform cube reference, not full truth','val_mse':best})
    return model


@torch.no_grad()
def evaluate(model,critic,data,seed,n=4096,k=32):
    rng=native.generator(seed);c=data.contexts(n,rng);z=torch.randn(n,k,3,generator=rng)
    y=torch.cat([native.ddim(model,cc[:,None],zz,50) for cc,zz in zip(c.split(128),z.split(128))])
    signs=y.sign();parity=signs.prod(-1);g=data.condition_signal(c[:,0])
    r=critic(y,c[:,None]); bins=((c[:,0]+1)*16).long().clamp(0,31)
    moment_error=[];low=[]
    for b in range(32):
        s=signs[bins==b];p=parity[bins==b]
        moment_error.append((p.mean()-.8*g[bins==b].mean()).abs())
        low.extend(s.mean((0,1)).abs().tolist())
        low.extend((s*s.roll(1,-1)).mean((0,1)).abs().tolist())
    stats={'reward_mean':float(r.mean()),'parity_bin_mae':float(torch.stack(moment_error).mean()),
           'low_order_sign_moment_max':max(low),
           'corner_fraction':float(((y-signs).norm(dim=-1)<.5).float().mean())}
    return r,stats


def run(output,steps=1000,classifier_steps=6000,continue_actual_from=None):
    output.mkdir(parents=True,exist_ok=False)
    report={'state':'ideal_classifier_gate','policy_steps':steps,'arms':{},
            'scope':'Exploratory local nonperiodic cube; no production H4 claim'}
    cfg=native.Config(dimensions=3,context_dim=1,hidden=128,ddim_steps=50,policy_steps=steps,eval_every=100)
    data=Data(cfg)
    def emit(row):
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        report['active']=row;atomic_json(output/'report.json',report)
        print(json.dumps(row),flush=True)
    if continue_actual_from is None:
        ideal=output/'ideal';ideal.mkdir()
        critic,fit=fit_pair(ideal,data,emit,steps=classifier_steps)
        report['ideal_classifier']=fit
        if not fit['gate']['passed']:
            report['state']='stopped_ideal_classifier_gate';atomic_json(output/'report.json',report);return report
        report['state']='pretraining_reference';source=pretrain(output,data,cfg,emit)
        _,health=evaluate(source,critic,data,60017)
        report['reference_health']=health
        if health['low_order_sign_moment_max']>.06 or health['corner_fraction']<.90:
            report['state']='stopped_reference_quality';atomic_json(output/'report.json',report);return report
    else:
        previous=json.loads((continue_actual_from/'report.json').read_text())
        if not previous['ideal_classifier']['gate']['passed']:
            raise ValueError('Source must pass ideal classifier gate')
        health=previous['reference_health']
        if health['low_order_sign_moment_max']>.06 or health['corner_fraction']<.90:
            raise ValueError('Source must pass reference quality')
        saved=torch.load(continue_actual_from/'reference.pt',map_location='cpu',weights_only=True)
        if saved['config']!=asdict(cfg):raise ValueError('Source config mismatch')
        source=native.Denoiser(cfg);source.load_state_dict(saved['model']);source.eval()
        atomic_checkpoint(output/'reference.pt',saved)
        report.update(ideal_classifier=previous['ideal_classifier'],reference_health=health,
            reused_source=str(continue_actual_from),
            classifier_replay='Same initial seed, samples and optimizer; replay from step0 with larger budget, not checkpoint resume')
    actual=output/'actual';actual.mkdir();report['state']='actual_classifier_gate'
    critic,fit=fit_pair(actual,data,emit,seed=23,model=source,steps=classifier_steps)
    report['actual_classifier']=fit
    if not fit['gate']['passed']:
        report['state']='stopped_actual_classifier_gate';atomic_json(output/'report.json',report);return report
    models={basis:ConditionDenoiser(cfg,basis,source.state_dict()) for basis in ('raw','fourier')}
    report['initial_matching']=verify_initial(source,models,cfg)
    before,base=evaluate(source,critic,data,70017);report['baseline']=base
    report['state']='dgpo';results={}
    for name,initial in models.items():
        def checkpoint(step,m,opt,rng,history):
            if step%100==0 or step==steps:
                atomic_checkpoint(output/f'{name}_last.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),
                    'rng':rng.get_state(),'step':step,'config':asdict(cfg)})
        m,_=native.policy_train('dgpo',initial,critic,data,cfg,17,71017,
            lambda row:emit({**row,'basis':name}),checkpoint,velocity_coefficient=1.)
        r,stats=evaluate(m,critic,data,70017);results[name]=r
        report['arms'][name]={'metrics':stats,'reward_gain':native.paired_gain(r,before)}
    report['fourier_minus_raw']=native.paired_gain(results['fourier'],results['raw'])
    report['state']='completed';atomic_json(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--classifier-steps',type=int,default=6000)
    parser.add_argument('--continue-actual-from',type=Path)
    args=parser.parse_args()
    if args.steps<1 or args.classifier_steps<2000:parser.error('Need positive policy budget and >=2000 classifier updates')
    torch.set_num_threads(2)
    print(json.dumps({'final_state':run(args.output,args.steps,args.classifier_steps,args.continue_actual_from)['state']}),flush=True)
