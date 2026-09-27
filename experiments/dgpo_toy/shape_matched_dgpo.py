"""Frozen shape-matched learned reward, original native DGPO, no production edits."""
import argparse
import copy
import json
import math
from dataclasses import replace,asdict
from pathlib import Path
import torch
import torch.nn.functional as F

from . import conditional as native
from .nonperiodic_cube import Critic,Data,classification,evaluate
from .cube_swap import swaps,modes
from .cube_lockdown import ConditionDenoiser,verify_initial
from .truth_pretrain import atomic_json,atomic_checkpoint


def make_fit_panel(model,data,grid,n,seed):
    p,health=swaps(model,data,grid,n,1024,seed)
    # C: truth mode weights, reference shape; D: unchanged reference samples.
    return {'c':p['C']['c'],'positive':p['C']['negative'],'negative':p['D']['negative']},health


def fit(output,source,data,emit,max_steps=16000):
    panels={};health={}
    for name,grid,n,seed in [('train',128,256,241017),('validation',128,64,251017),('test',256,128,261017)]:
        panels[name],health[name]=make_fit_panel(source,data,grid,n,seed)
    atomic_checkpoint(output/'classifier_panels.pt',panels)
    torch.manual_seed(23);m=Critic(True)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.001)
    rng=native.generator(40023);best=anchor=math.inf;stale=0;selected=0;weights=None
    history=[];p=panels['train']
    for step in range(1,max_steps+1):
        ids=torch.randint(len(p['c']),(256,),generator=rng)
        c=torch.cat([p['c'][ids]]*2);y=torch.cat([p['positive'][ids],p['negative'][ids]])
        loss=F.binary_cross_entropy_with_logits(m(y,c),torch.cat([torch.ones(256),torch.zeros(256)]))
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%100==0 or step==max_steps:
            stats,_=classification(m,panels['validation']);v=stats['bce']
            if v<best:best=v;selected=step;weights=copy.deepcopy(m.state_dict())
            if v<anchor-1e-4:anchor=v;stale=0
            else:stale+=1
            row={'phase':'classifier','step':step,'train_bce':float(loss.detach()),
                 'validation':stats,'selected_step':selected,'stale_checks':stale}
            history.append(row);emit(row)
            if step>=2000 and stale>=20:break
    atomic_checkpoint(output/'classifier_training_state.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),
        'rng':rng.get_state(),'step':step,'best_model':weights,'selected_step':selected})
    m.load_state_dict(weights);m.eval().requires_grad_(False)
    test,_=classification(m,panels['test'])
    gate={'plateau':stale>=20,'heldout_condition_discrimination':test['auc']>.55 and test['bce']<math.log(2)-.01}
    gate['passed']=all(gate.values())
    atomic_checkpoint(output/'classifier.pt',{'model':m.state_dict(),'selected_step':selected,'test':test})
    return m,{'test':test,'gate':gate,'steps':step,'selected_step':selected,'health':health,'history':history}


@torch.no_grad()
def mode_shape_panel(model,critic,data,seed=281017,grid=128,k=1024):
    c=((torch.arange(grid)+.5)/grid*2-1)[:,None]
    z=torch.randn(grid,k,3,generator=native.generator(seed))
    y=torch.cat([native.ddim(model,cc[:,None],zz,50) for cc,zz in zip(c.split(8),z.split(8))])
    if not torch.isfinite(y).all():raise FloatingPointError('Nonfinite evaluation samples')
    ids=modes(y);r=critic(y,c[:,None]);count=torch.stack([torch.bincount(row,minlength=8) for row in ids])
    if count.min()<16:raise ValueError('Sparse evaluation mode; no hidden fallback')
    mean=torch.stack([torch.stack([r[j,ids[j]==a].mean() for a in range(8)]) for j in range(grid)])
    q=count.double()/k;p=data.probabilities(c[:,0],.9).double()
    stats={'reward_mean':float(r.mean()),'mode_tv':float((q-p).abs().sum(-1).mean()/2),
           'mode_rmse':float((q-p).square().mean().sqrt()),'min_cell_count':int(count.min())}
    return {'q':q,'p':p,'mode_mean_reward':mean,'reward':r,'metrics':stats}


def decomposition(current,baseline):
    # Mode-only counterfactual retains baseline conditional shape score in every mode.
    ref_scores=baseline['mode_mean_reward'].double()
    mode=(current['q']*ref_scores).sum(-1).mean()-(baseline['q']*ref_scores).sum(-1).mean()
    total=current['reward'].double().mean()-baseline['reward'].double().mean()
    return {'total_reward_gain':float(total),'mode_probability_contribution':float(mode),
            'within_mode_shape_contribution':float(total-mode),
            'scope':'empirical grid decomposition, not optimizer/KL causal attribution'}


def run(source,output,steps=1000,velocity_coefficient=1.,reuse_classifier=None):
    if velocity_coefficient not in (0.,1.):
        raise ValueError('This matched ablation supports coefficient0 or1')
    saved=torch.load(source/'reference.pt',map_location='cpu',weights_only=True)
    cfg=replace(native.Config(**saved['config']),policy_steps=steps,eval_every=100,eval_events=512)
    initial=native.Denoiser(cfg);initial.load_state_dict(saved['model']);initial.eval()
    output.mkdir(parents=True,exist_ok=False);data=Data(cfg)
    report={'state':'building_shape_matched_classifier','source':str(source.resolve()),
            'config':asdict(cfg),'target':'truth conditional mode probabilities with initial generator conditional shape',
            'velocity_coefficient':velocity_coefficient,'arms':{}}
    def emit(row):
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        report['active']={k:row[k] for k in ('phase','step','basis') if k in row}
        atomic_json(output/'report.json',report);print(json.dumps(row),flush=True)
    atomic_json(output/'report.json',report)
    try:
        if reuse_classifier is None:
            critic,fit_report=fit(output,initial,data,emit)
        else:
            prior=json.loads((reuse_classifier/'report.json').read_text())
            if prior['state']!='completed' or Path(prior['source']).resolve()!=source.resolve() or prior['config']!=asdict(cfg):
                raise ValueError('Matched comparison requires same completed source/config/budget')
            fit_report=prior['classifier']
            ck=torch.load(reuse_classifier/'classifier.pt',map_location='cpu',weights_only=True)
            if ck['selected_step']!=fit_report['selected_step']:
                raise ValueError('Classifier selection mismatch')
            critic=Critic(True);critic.load_state_dict(ck['model']);critic.eval().requires_grad_(False)
            atomic_checkpoint(output/'classifier.pt',ck)
            report['reused_classifier']=str(reuse_classifier.resolve())
        report['classifier']=fit_report
        if not fit_report['gate']['passed']:
            report['state']='stopped_classifier_gate';return report
        base=mode_shape_panel(initial,critic,data)
        report['baseline']=base['metrics'];atomic_checkpoint(output/'baseline_decomposition.pt',base)
        if reuse_classifier is not None:
            old=torch.load(reuse_classifier/'baseline_decomposition.pt',map_location='cpu',weights_only=True)
            report['matched_baseline_exact']=all(torch.equal(base[k],old[k]) for k in ('q','p','mode_mean_reward','reward'))
            if not report['matched_baseline_exact']:raise ValueError('Initial evaluation differs from matched control')
        oracle=(base['p'].clamp_min(1e-10).log()-base['q'].clamp_min(1e-10).log())
        a=base['mode_mean_reward'].double();a=a-a.mean(-1,keepdim=True)
        b=oracle-oracle.mean(-1,keepdim=True)
        report['mode_ordering_cosine']=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-20))
        # Stop if the learned mode preferences are not demonstrably aligned even globally.
        if report['mode_ordering_cosine']<.5:
            report['state']='stopped_mode_ordering';return report
        models={b:ConditionDenoiser(cfg,b,initial.state_dict()).eval() for b in ('raw','fourier')}
        report['initial_matching']=verify_initial(initial,models,cfg)
        before,stats=evaluate(initial,critic,data,282017);report['continuous_baseline']=stats
        report['state']='training_dgpo';final_rewards={}
        for name,m0 in models.items():
            report['arms'][name]={'state':'running','history':[]}
            def checkpoint(step,m,opt,rng,history):
                if step%100==0 or step==steps:
                    _,s=evaluate(m,critic,data,283017,n=1024)
                    report['arms'][name]['history'].append({'step':step,**s})
                    atomic_checkpoint(output/f'{name}_last.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),
                        'rng':rng.get_state(),'step':step,'history':history,'config':asdict(cfg),'basis':name,
                        'velocity_coefficient':velocity_coefficient})
                    emit({'phase':'structure','basis':name,'step':step,**s})
            m,_=native.policy_train('dgpo',m0,critic,data,cfg,17,284017,
                lambda row:emit({**row,'basis':name}),checkpoint,velocity_coefficient=velocity_coefficient)
            current=mode_shape_panel(m,critic,data);r,s=evaluate(m,critic,data,282017)
            final_rewards[name]=r
            report['arms'][name].update(state='completed',endpoint=s,reward_gain=native.paired_gain(r,before),
                mode_metrics=current['metrics'],mode_tv_change=current['metrics']['mode_tv']-base['metrics']['mode_tv'],
                decomposition=decomposition(current,base))
            atomic_checkpoint(output/f'{name}_decomposition.pt',current)
            atomic_checkpoint(output/f'{name}_continuous_reward.pt',r)
            if reuse_classifier is not None:
                control=ConditionDenoiser(cfg,name,initial.state_dict())
                ck=torch.load(reuse_classifier/f'{name}_last.pt',map_location='cpu',weights_only=True)
                if ck['step']!=steps:raise ValueError('Control endpoint budget mismatch')
                control.load_state_dict(ck['model']);control.eval()
                control_reward,control_metrics=evaluate(control,critic,data,282017)
                report['arms'][name]['versus_coefficient_control']={
                    'control_coefficient':prior['velocity_coefficient'],
                    'reward_difference':native.paired_gain(r,control_reward),
                    'parity_mae_difference':s['parity_bin_mae']-control_metrics['parity_bin_mae'],
                    'mode_tv_difference':current['metrics']['mode_tv']-prior['arms'][name]['mode_metrics']['mode_tv']}
        report['fourier_minus_raw']=native.paired_gain(final_rewards['fourier'],final_rewards['raw'])
        report['state']='completed'
    except BaseException as exc:
        report.update(state='failed_or_interrupted',error=repr(exc));raise
    finally:
        atomic_json(output/'report.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--velocity-coefficient',type=float,choices=(0.,1.),default=1.)
    p.add_argument('--reuse-classifier',type=Path)
    args=p.parse_args()
    if args.steps<1:p.error('Need positive steps')
    torch.set_num_threads(2)
    r=run(args.source,args.output,args.steps,args.velocity_coefficient,args.reuse_classifier)
    print(json.dumps({'state':r['state']}),flush=True)
