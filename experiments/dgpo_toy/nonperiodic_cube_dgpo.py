"""Second ablation: fixed learned critic, matched raw/Fourier diffusion DGPO."""
import argparse
import json
import time
from dataclasses import asdict,replace
from pathlib import Path

import torch

from . import conditional as native
from .nonperiodic_cube import Data,Critic,evaluate
from .cube_lockdown import ConditionDenoiser,verify_initial
from .truth_pretrain import atomic_json,atomic_checkpoint


def load(source,steps):
    previous=json.loads((source/'report.json').read_text())
    fit=previous['actual_classifier']
    if not (fit['gate']['fourier_discriminates'] and fit['gate']['material_bce_advantage']
            and fit['test']['fourier']['stale_checks']>=20):
        raise ValueError('Need an adequately trained, validated Fourier critic')
    saved=torch.load(source/'reference.pt',map_location='cpu',weights_only=True)
    cfg=replace(native.Config(**saved['config']),policy_steps=steps,eval_every=100,eval_events=512)
    model=native.Denoiser(cfg);model.load_state_dict(saved['model']);model.eval()
    cs=torch.load(source/'actual/fourier_classifier.pt',map_location='cpu',weights_only=True)
    if cs['fourier'] is not True or cs['selected_step']!=fit['test']['fourier']['selected_step']:
        raise ValueError('Critic provenance mismatch')
    critic=Critic(True);critic.load_state_dict(cs['model']);critic.eval().requires_grad_(False)
    return cfg,model,critic,fit,cs['selected_step']


def run(source,output,steps=1000):
    cfg,initial,critic,fit,critic_step=load(source,steps)
    output.mkdir(parents=True,exist_ok=False)
    data=Data(cfg)
    models={b:ConditionDenoiser(cfg,b,initial.state_dict()).eval() for b in ('raw','fourier')}
    report={'state':'initializing','source':str(source.resolve()),'config':asdict(cfg),
        'critic_selected_step':critic_step,'classifier_test':fit['test'],
        'plain_plateau_gate_waived_by_user':True,
        'question':'Does Fourier diffusion conditioning improve fixed learned reward absorption?',
        'not_claimed':'Plain classifier can never learn; exact endpoint KL; production EveNet result',
        'velocity_coefficient':1.,'policy_seed':17,'monitor_seed':81017,'endpoint_seed':82017,
        'structure_monitor_seed':83017,'reward_margin':.01,'arms':{},
        'initial_matching':verify_initial(initial,models,cfg)}
    start=time.monotonic()
    def emit(row):
        with (output/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        report['active']={'phase':row['phase'],'basis':row.get('basis'),'step':row.get('step')}
        report['elapsed_seconds']=time.monotonic()-start
        atomic_json(output/'report.json',report)
        print(json.dumps(row,allow_nan=False),flush=True)
    critic_state={k:v.clone() for k,v in critic.state_dict().items()}
    source_state={k:v.clone() for k,v in initial.state_dict().items()}
    before,base=evaluate(initial,critic,data,82017)
    _,monitor_base=evaluate(initial,critic,data,83017,n=1024)
    report.update(state='training_dgpo',baseline=base,monitor_baseline=monitor_base)
    atomic_checkpoint(output/'initial_endpoint.pt',{'reward':before,'metrics':base})
    results={}
    try:
        for basis,m0 in models.items():
            report['arms'][basis]={'state':'running','structure_history':[]}
            def checkpoint(step,m,opt,rng,history):
                if step%100==0 or step==steps:
                    _,metrics=evaluate(m,critic,data,83017,n=1024)
                    report['arms'][basis]['structure_history'].append({'step':step,**metrics})
                    atomic_checkpoint(output/f'{basis}_last.pt',{'model':m.state_dict(),
                        'optimizer':opt.state_dict(),'rng':rng.get_state(),'history':history,
                        'step':step,'config':asdict(cfg),'velocity_coefficient':1.,'basis':basis,
                        'source':str(source.resolve()),'critic_selected_step':critic_step})
                    emit({'phase':'structure','basis':basis,'step':step,**metrics})
            final,history=native.policy_train('dgpo',m0,critic,data,cfg,17,81017,
                lambda row:emit({**row,'basis':basis}),checkpoint,velocity_coefficient=1.)
            r,stats=evaluate(final,critic,data,82017)
            results[basis]=r
            report['arms'][basis].update(state='completed',steps=len(history),endpoint=stats,
                reward_gain=native.paired_gain(r,before),
                parity_mae_change=stats['parity_bin_mae']-base['parity_bin_mae'],
                low_order_moment_change=stats['low_order_sign_moment_max']-base['low_order_sign_moment_max'],
                corner_fraction_change=stats['corner_fraction']-base['corner_fraction'])
            atomic_checkpoint(output/f'{basis}_endpoint.pt',{'reward':r,'metrics':stats})
            emit({'phase':'arm_completed','basis':basis,'step':steps,**report['arms'][basis]['reward_gain']})
        report['fourier_minus_raw']=native.paired_gain(results['fourier'],results['raw'])
        report['decision']={
            'raw_material_reward_gain':report['arms']['raw']['reward_gain']['lo95']>.01,
            'fourier_material_reward_gain':report['arms']['fourier']['reward_gain']['lo95']>.01,
            'fourier_material_advantage':report['fourier_minus_raw']['lo95']>.01,
            'raw_small_gain_at_budget':report['arms']['raw']['reward_gain']['hi95']<.01,
            'scope':'single seed, finite budget; context-paired test intervals, not training-seed uncertainty'}
        report['decision']['failure_and_rescue']=(report['decision']['raw_small_gain_at_budget'] and
            report['decision']['fourier_material_reward_gain'] and report['decision']['fourier_material_advantage'])
        report['source_unchanged']=all(torch.equal(v,source_state[k]) for k,v in initial.state_dict().items())
        report['critic_unchanged']=all(torch.equal(v,critic_state[k]) for k,v in critic.state_dict().items())
        report['state']='completed'
    except BaseException as exc:
        report.update(state='failed_or_interrupted',error=repr(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        atomic_json(output/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    torch.set_num_threads(2)
    result=run(args.source,args.output,args.steps)
    print(json.dumps({'state':result['state'],'decision':result.get('decision')}),flush=True)
