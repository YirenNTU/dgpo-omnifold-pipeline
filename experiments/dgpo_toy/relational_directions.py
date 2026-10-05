"""Read-only native-surrogate versus sampled DDIM-reward derivative diagnosis."""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from . import closed_loop_lab as lab
from . import conditional as native
from .conditional_interference import flat_grad, cosine, displaced
from .nonperiodic_cube import Data
from .relational_conditioning import RelationalData
from .relational_retention import make_models, installed_reward
from .truth_pretrain import atomic_json, atomic_checkpoint


def reward_gradient(model, reward, contexts, noise, cfg):
    gradients=[]
    for split in (slice(0,None,2),slice(1,None,2)):
        cs,zs=contexts[split],noise[split]
        total=None
        for c,z in zip(cs.split(16),zs.split(16)):
            y=native.ddim(model,c[:,None],z,cfg.ddim_steps)
            g=flat_grad(reward(y,c[:,None]).mean(),model)*(len(c)/len(cs))
            total=g if total is None else total+g
        gradients.append(total)
    return torch.stack(gradients)


@torch.no_grad()
def evaluate(model,reward,c,z,cfg):
    return torch.cat([reward(native.ddim(model,cc[:,None],zz,cfg.ddim_steps),cc[:,None])
                      for cc,zz in zip(c.split(32),z.split(32))]).double().mean()


def probe(model,reference,reward,data,cfg,state):
    rng=native.generator(929951)
    main,regularizer=[],[]
    for _ in range(16):
        c=data.contexts(cfg.batch,rng)
        z=torch.randn(cfg.batch,cfg.candidates,3,generator=rng)
        t=.7*torch.rand(cfg.timesteps,cfg.batch,generator=rng)
        eps=torch.randn(cfg.timesteps,cfg.batch,3,generator=rng)
        parts={}
        def capture(a,b):
            parts['main']=flat_grad(a,model,retain_graph=True)
            parts['reference']=flat_grad(b,model,retain_graph=True)
        loss,_=native.dgpo_objective(model,reference,reward,data,cfg,c,z,t,eps,
            velocity_coefficient=1.,trace_gradients=False,component_callback=capture)
        full=flat_grad(loss,model)
        if not torch.allclose(full,parts['main']+parts['reference'],rtol=2e-4,atol=2e-7):
            raise ValueError('Native gradient components do not reconstruct the loss')
        main.append(parts['main']);regularizer.append(parts['reference'])
    gm,gr=torch.stack(main),torch.stack(regularizer)
    total=(gm+gr).mean(0)
    rng=native.generator(929952)
    c=data.contexts(1024,rng);z=torch.randn(1024,8,3,generator=rng)
    halves=reward_gradient(model,reward,c,z,cfg);exact=halves.mean(0)
    proposal=copy.deepcopy(model)
    opt=torch.optim.AdamW(proposal.parameters(),lr=cfg.policy_lr,weight_decay=cfg.weight_decay)
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    i=0
    for p in proposal.parameters():
        p.grad=total[i:i+p.numel()].view_as(p).clone();i+=p.numel()
    torch.nn.utils.clip_grad_norm_(proposal.parameters(),1.,error_if_nonfinite=True)
    opt.step()
    delta=torch.cat([(p-q).detach().flatten() for p,q in zip(proposal.parameters(),model.parameters())])
    scale=.1
    finite=float((evaluate(displaced(model,delta,scale),reward,c,z,cfg)-
                  evaluate(displaced(model,delta,-scale),reward,c,z,cfg))/(2*scale))
    predicted=float(exact.double()@delta.double())
    relative_error=abs(finite-predicted)/max(abs(predicted),1e-10)
    diagnostics={
        'reward_gradient_half_cosine':cosine(*halves),
        'native_gradient_half_cosine':cosine((gm+gr)[::2].mean(0),(gm+gr)[1::2].mean(0)),
        'main_reference_cosine':cosine(gm.mean(0),gr.mean(0)),
        'reference_to_main_norm':float(gr.mean(0).norm()/gm.mean(0).norm()),
        'main_on_actual_reward_cosine':cosine(-gm.mean(0),exact),
        'reference_on_actual_reward_cosine':cosine(-gr.mean(0),exact),
        'total_on_actual_reward_cosine':cosine(-total,exact),
        'native_adamw_predicted_gain':predicted,'native_adamw_finite_difference':finite,
        'finite_difference_relative_error':relative_error,'derivative_check_valid':relative_error<.1,
        'half_predictions':[float(g.double()@delta.double()) for g in halves],
        'scope':'Local derivative of sampled actual DDIM reward; mean16 exact native losses. '
                'Counterfactual native AdamW proposal, not applied training and not expected reward proof.'}
    return diagnostics,{'main':gm,'reference':gr,'actual_reward_halves':halves,'displacement':delta}


def run(directory,output):
    directory,output=lab.toy_path(directory),lab.toy_path(output)
    plan=json.loads((directory/'plan.json').read_text())
    cfg,source=lab.load_source();cfg=replace(cfg,eval_every=25,eval_events=512)
    reference=make_models(source,cfg,plan)['raw'].eval().requires_grad_(False)
    reward,_=installed_reward(plan);data=RelationalData(Data(cfg))
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',{'source':str(directory),'checkpoints':[5,20],
        'question':'Is the surrogate/native displacement aligned with actual reward during the transient?',
        'train_calls':16,'train_contexts_per_call':64,'eval_contexts':1024,'eval_k':8,
        'validity':'Component identity and finite difference error<.1; inspect independent-half consistency',
        'scope':'Read-only; no reward/objective/model update persisted; no training seed search'})
    report={'state':'running','points':{}};start=time.monotonic()
    for step in (5,20):
        state=torch.load(directory/f'raw_step{step}.pt',map_location='cpu',weights_only=True)
        model=copy.deepcopy(reference).requires_grad_(True)
        model.load_state_dict(state['model'])
        result,arrays=probe(model,reference,reward,data,cfg,state)
        report['points'][str(step)]=result
        atomic_checkpoint(output/f'gradients_step{step}.pt',arrays)
        atomic_json(output/'report.json',report)
        print(step,json.dumps(result),flush=True)
    report.update(state='completed',elapsed_seconds=time.monotonic()-start)
    atomic_json(output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(args.directory,args.output)


if __name__=='__main__':main()
