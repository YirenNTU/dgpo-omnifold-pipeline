"""Exact saved-step replay versus averaging complete native DGPO calls.

Toy only. Same fixed reward, reference, AdamW state, LR and primary minibatch
stream. Additional calls have their OWN detached nonlinear gates, so averaging
preserves the expected objective (unlike transforming/scaling advantages).
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import torch
from . import closed_loop_lab as lab
from . import conditional as native
from . import relational_experiment as rel
from .classifier_signal_attribution import simultaneous_mean_intervals
from .nonperiodic_cube import Data
from .relational_conditioning import RelationalData
from .relational_retention import make_models, installed_reward, retention_contrasts, tracker_for
from .truth_pretrain import atomic_json,atomic_checkpoint


def inputs(data,cfg,rng):
    c=data.contexts(cfg.batch,rng)
    z=torch.randn(cfg.batch,cfg.candidates,cfg.dimensions,generator=rng)
    t=.7*torch.rand(cfg.timesteps,cfg.batch,generator=rng)
    eps=torch.randn(cfg.timesteps,cfg.batch,cfg.dimensions,generator=rng)
    return c,z,t,eps


def run(directory,output):
    directory,output=lab.toy_path(directory),lab.toy_path(output)
    old=json.loads((directory/'plan.json').read_text())
    cfg,source=lab.load_source();cfg=replace(cfg,policy_steps=20,eval_every=1,eval_events=128)
    initial=make_models(source,cfg,old)['raw']
    state=torch.load(directory/'raw_step5.pt',map_location='cpu',weights_only=True)
    expected=torch.load(directory/'raw_step20.pt',map_location='cpu',weights_only=True)
    reward,_=installed_reward(old);data=RelationalData(Data(cfg))
    output.mkdir(parents=True,exist_ok=False)
    plan={'question':'Does averaging native losses prevent the confirmed transient loss?',
        'run_name':'Does gradient averaging retain reward? | exact step 5 replay | native DGPO mean of 8',
        'source':str(directory),'resume_step':5,'end_step':20,'replicas':[1,8],
        'primary_stream':'Same saved native RNG in both arms; extra calls use a separate explicit RNG',
        'reference':'Exact original reference from round2 start, not step5 recentering',
        'primary_endpoint':'Paired average8-minus-native reward at20 on previously declared large panel; lower95>.01',
        'failure_rule':'Original >50 percent retained-gain criterion unchanged; absence of growth is not rescue',
        'scope':'Same expected objective; mean of individually gated losses. No changed reward, KL or LR. '
                'More samples/compute, not an equal-compute comparison. One training seed.'}
    atomic_json(output/'plan.json',plan)
    tracker=tracker_for(output,plan);started=time.monotonic()
    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        from .coverage_budget import flatten_metrics
        tracker.log(flatten_metrics(row,f"{row.get('arm','shared')}/"))
        print(json.dumps(row,allow_nan=False),flush=True)
    try:
        control,history=native.policy_train('dgpo',initial,reward,data,cfg,
            old['policy_seed'],old['monitor_seed'],lambda row:emit({**row,'arm':'native'}),
            resume_state=state,velocity_coefficient=1.)
        replay_equal=all(torch.equal(v,expected['model'][k]) for k,v in control.state_dict().items())
        if not replay_equal:raise ValueError('Native replay must match saved step20 bitwise')
        atomic_checkpoint(output/'native_step20.pt',expected)
        model=copy.deepcopy(initial).requires_grad_(True);model.load_state_dict(state['model'])
        reference=copy.deepcopy(initial).requires_grad_(False)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg.policy_lr,weight_decay=cfg.weight_decay)
        opt.load_state_dict(copy.deepcopy(state['optimizer']))
        rng=native.generator(0);rng.set_state(state['rng']);extra=native.generator(929963)
        for step in range(6,21):
            opt.zero_grad(set_to_none=True);loss_value=0.
            for replica in range(8):
                batch=inputs(data,cfg,rng if replica==0 else extra)
                loss,_=native.dgpo_objective(model,reference,reward,data,cfg,*batch,
                                            velocity_coefficient=1.,trace_gradients=False)
                (loss/8).backward();loss_value+=float(loss.detach())/8
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            opt.step();emit({'phase':'policy','arm':'average8','step':step,'total_loss':loss_value,
                'gradient_norm':norm,'gradient_calls':8,'velocity_coefficient':1.})
        atomic_checkpoint(output/'average8_step20.pt',{'model':model.state_dict(),
            'optimizer':opt.state_dict(),'rng':rng.get_state(),'extra_rng':extra.get_state(),
            'step':20,'config':asdict(cfg),'velocity_coefficient':1.,'representation':'raw',
            'scope':'Branch experiment with extra-call RNG, not a native single-call resume file'})
        eval_plan={**old,'evaluation_seed':929927,'eval_contexts':16384,'eval_candidates':32}
        baseline=torch.load(directory/'confirmation/arrays.pt',map_location='cpu',weights_only=True)
        arrays={'baseline_0':baseline['baseline_0'],'native_5':baseline['raw_5'],
                'average8_5':baseline['raw_5'],'native_20':baseline['raw_20']}
        arrays['average8_20']=rel.reward_panel(model,reward,data,cfg,eval_plan)
        atomic_checkpoint(output/'arrays.pt',arrays)
        result={'state':'completed','native_replay_bitwise':replay_equal,
            'retention':retention_contrasts(arrays,['native','average8']),
            'comparison':simultaneous_mean_intervals({'average8_minus_native':
                (arrays['average8_20']-arrays['native_20']).double().mean(-1)},2000,929964),
            'gain20':{arm:float((arrays[f'{arm}_20']-arrays['baseline_0']).double().mean())
                      for arm in ('native','average8')},
            'elapsed_seconds':time.monotonic()-started,
            'scope':'Same fitted policies, paired independent contexts; not training-seed robustness'}
        atomic_json(output/'report.json',result);emit({'phase':'decision',**result})
    finally:tracker.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);run(a.directory,a.output)


if __name__=='__main__':main()
