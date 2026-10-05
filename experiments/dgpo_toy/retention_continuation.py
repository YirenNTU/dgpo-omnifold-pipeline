"""20-update toy-only native resume with a large paired reward-retention panel."""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch
from . import closed_loop_lab as lab
from . import conditional as native
from .nonperiodic_cube import Critic, Data
from .reward_retention import analyze, intervals
from .truth_pretrain import atomic_checkpoint, atomic_json


@torch.no_grad()
def evaluate(model, critic, c, z, cfg):
    samples=[];rewards=[]
    for cc,zz in zip(c.split(32),z.split(32)):
        y=native.ddim(model,cc[:,None],zz,cfg.ddim_steps)
        samples.append(y);rewards.append(critic(y,cc[:,None]))
    return torch.cat(samples),torch.cat(rewards)


def run(source,output,tracker=None):
    output=lab.toy_path(output);output.mkdir(parents=True,exist_ok=False)
    cfg,base=lab.load_source();data=Data(cfg)
    plan=json.loads((source/'plan.json').read_text())
    critic=Critic(True).eval().requires_grad_(False)
    critic.load_state_dict(torch.load(lab.HISTORY/'rewards/full_reward.pt',weights_only=True,map_location='cpu')['model'])
    rng=native.generator(929003)
    c=data.contexts(1024,rng);z=torch.randn(1024,64,cfg.dimensions,generator=rng)
    atomic_checkpoint(output/'panel.pt',{'contexts':c,'noise':z,'seed':929003})
    started=time.monotonic()
    report={'state':'running','source':str(source),'anchors':{},'policy_updates_per_arm':20,
            'reward':str(lab.HISTORY/'rewards/full_reward.pt'),
            'scope':'Toy native full-state resume; same critic/reference/optimizer/RNG; no fresh audit or closure claim'}
    for arm in ('raw','fourier'):
        with torch.random.fork_rng():
            torch.manual_seed(plan['initialization_seed'])
            initial=lab.ConditioningAblation(cfg,base.state_dict(),**plan['conditioning'][arm]).eval()
        state=torch.load(source/f'{arm}_step1000.pt',weights_only=True,map_location='cpu')
        model=lab.ConditioningAblation(cfg,base.state_dict(),**plan['conditioning'][arm]).eval()
        model.load_state_dict(state['model'])
        samples,r0=evaluate(model,critic,c,z,cfg)
        samples_again,r0_again=evaluate(model,critic,c,z,cfg)
        if not torch.equal(samples,samples_again) or not torch.equal(r0,r0_again):
            raise ValueError('No-update replay must be exact')
        atomic_checkpoint(output/f'{arm}_endpoint0.pt',{'contexts':c,'samples':samples,'rewards':r0})
        values={0:r0.numpy()}
        def emit(row):
            row={**row,'objective_arm':row.get('arm','dgpo'),'arm':arm}
            with (output/f'{arm}_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            if tracker is not None:
                tracker.log({f'{arm}/{key}':value for key,value in row.items()
                             if isinstance(value,(int,float))})
            print(json.dumps(row,allow_nan=False),flush=True)
        def checkpoint(step,m,opt,train_rng,history):
            relative=step-1000
            if relative not in (1,5,20):return
            samples,rewards=evaluate(m,critic,c,z,cfg)
            values[relative]=rewards.numpy()
            atomic_checkpoint(output/f'{arm}_endpoint{relative}.pt',{'contexts':c,'samples':samples,'rewards':rewards})
            atomic_checkpoint(output/f'{arm}_step{step}.pt',{'model':m.state_dict(),'optimizer':opt.state_dict(),
                'rng':train_rng.get_state(),'step':step,'history':history,'velocity_coefficient':1.,
                'config':asdict(replace(cfg,policy_steps=1020,eval_every=5,eval_events=512)),
                'conditioning':state['conditioning']})
            emit({'phase':'paired_endpoint','relative_step':relative,
                  'reward':float(rewards.mean()),'gain':float((rewards-r0).mean())})
        native.policy_train('dgpo',initial,critic,data,replace(cfg,policy_steps=1020,eval_every=5,eval_events=512),
                            plan['policy_seed'],plan['monitor_seed'],emit,checkpoint,
                            resume_state=state,velocity_coefficient=1.)
        result=analyze(values[0],values[5],values[20])
        result['endpoint_gains']={str(step):intervals((value-values[0]).mean(1)[:,None])[0]
                                  for step,value in values.items()}
        result['condition_bins']=[]
        bins=np.minimum(((c.numpy().ravel()+1)*4).astype(int),7)
        for b in range(8):
            ids=bins==b
            result['condition_bins'].append({'bounds':[-1+b/4,-1+(b+1)/4],
                'result':analyze(values[0][ids],values[5][ids],values[20][ids])})
        result['no_update_replay_max_error']=0.
        report['anchors'][arm]=result
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'report.json',report)
    report['state']='completed';atomic_json(output/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=lab.ARTIFACTS/'conditioning_closed_loop_20260926/round04')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    # Offline-only: never publish toy artifacts to the production project.
    import wandb
    args.output=lab.toy_path(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with wandb.init(project='dgpo-toy-local',mode='offline',dir=str(args.output.parent),
                    name='Do late updates retain reward? | conditional toy | 20 native updates',
                    group='Conditional reward retention',tags=['toy-only','fixed-H4','raw-no-EMA','velocity-MSE-1'],
                    config={'source':str(args.source),'output':str(args.output),'logging':'live-local-offline'},
                    settings=wandb.Settings(disable_git=True)) as tracker:
        run(args.source,args.output,tracker)


if __name__=='__main__':main()
