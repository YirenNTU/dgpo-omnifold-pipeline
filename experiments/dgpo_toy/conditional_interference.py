"""Read-only actual-reward / native-surrogate conditional transfer on saved toys."""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from . import conditional as native
from . import closed_loop_lab as lab
from .nonperiodic_cube import Critic, Data
from .truth_pretrain import atomic_json


def flat_grad(loss, model, *, retain_graph=False):
    parameters = tuple(model.parameters())
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, values)])


def cosine(a, b):
    norm = a.norm()*b.norm()
    return float(a.double()@b.double()/norm) if norm > 0 else None


def bin_contexts(index, bins, n, rng):
    return -1 + (index+torch.rand(n, 1, generator=rng))*2/bins


def derivative_matrix(reward_gradients, loss_gradients):
    """Row = evaluated condition, column = updated condition, positive = benefit."""
    return -(reward_gradients.double() @ loss_gradients.double().T)


@torch.no_grad()
def panel_rewards(model, critic, contexts, noises, cfg):
    output=[]
    for c, z in zip(contexts, noises):
        y=native.ddim(model,c[:,None],z,cfg.ddim_steps)
        output.append(critic(y,c[:,None]).mean())
    return torch.stack(output)


def displaced(model, displacement, scale):
    model=copy.deepcopy(model)
    offset=0
    with torch.no_grad():
        for p in model.parameters():
            p.add_(displacement[offset:offset+p.numel()].view_as(p),alpha=scale)
            offset+=p.numel()
    return model


def probe(model, reference, critic, data, cfg, checkpoint, *, bins=8, train_per_bin=32, eval_per_bin=16, k=16):
    train_rng=native.generator(929001)
    eval_rng=native.generator(929002)
    main, regularizer, exact_halves, eval_c, eval_z=[],[],[],[],[]
    for b in range(bins):
        c=bin_contexts(b,bins,train_per_bin,train_rng)
        z=torch.randn(train_per_bin,cfg.candidates,cfg.dimensions,generator=train_rng)
        t=.7*torch.rand(cfg.timesteps,train_per_bin,generator=train_rng)
        eps=torch.randn(cfg.timesteps,train_per_bin,cfg.dimensions,generator=train_rng)
        components={}
        def capture(m,r):
            components['main']=flat_grad(m,model,retain_graph=True)
            components['ref']=flat_grad(r,model,retain_graph=True)
        loss,_=native.dgpo_objective(model,reference,critic,data,cfg,c,z,t,eps,
                                    velocity_coefficient=1.,component_callback=capture)
        # The components must reconstruct the unchanged objective gradient.
        total=flat_grad(loss,model)
        if not torch.allclose(total,components['main']+components['ref'],rtol=2e-4,atol=2e-7):
            raise ValueError('Surrogate gradient reconstruction failed')
        main.append(components['main']);regularizer.append(components['ref'])
        c=bin_contexts(b,bins,eval_per_bin,eval_rng)
        z=torch.randn(eval_per_bin,k,cfg.dimensions,generator=eval_rng)
        eval_c.append(c);eval_z.append(z)
        halves=[]
        for noise in (z[:,::2],z[:,1::2]):
            y=native.ddim(model,c[:,None],noise,cfg.ddim_steps)
            halves.append(flat_grad(critic(y,c[:,None]).mean(),model))
        exact_halves.append(torch.stack(halves))
    main,regularizer=torch.stack(main),torch.stack(regularizer)
    exact_halves=torch.stack(exact_halves)
    exact=exact_halves.mean(1);total=main+regularizer
    matrices={name:derivative_matrix(exact,g) for name,g in
              (('main',main),('reference',regularizer),('total',total))}
    half_matrices=torch.stack([derivative_matrix(exact_halves[:,i],total) for i in range(2)])
    stable_negative=(half_matrices[0]<0)&(half_matrices[1]<0)
    mask=~torch.eye(bins,dtype=torch.bool)
    g=total.mean(0)
    proposal=copy.deepcopy(model)
    optimizer=torch.optim.AdamW(proposal.parameters(),lr=cfg.policy_lr,weight_decay=cfg.weight_decay)
    optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    offset=0
    for p in proposal.parameters():
        p.grad=g[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
    torch.nn.utils.clip_grad_norm_(proposal.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    displacement=torch.cat([(p.detach()-q.detach()).flatten() for p,q in zip(proposal.parameters(),model.parameters())])
    predicted=exact.double()@displacement.double()
    predicted_halves=torch.stack([exact_halves[:,i].double()@displacement.double() for i in range(2)])
    # Small finite displacement checks the derivative, not a changed training LR.
    scale=.01
    plus=panel_rewards(displaced(model,displacement,scale),critic,eval_c,eval_z,cfg)
    minus=panel_rewards(displaced(model,displacement,-scale),critic,eval_c,eval_z,cfg)
    finite=(plus.double()-minus.double())/(2*scale)
    relative=float((finite-predicted).norm()/predicted.norm().clamp_min(1e-12))
    diag=torch.diagonal(matrices['total'])/bins**2
    off=(matrices['total'].sum()-torch.diagonal(matrices['total']).sum())/bins**2
    return {
        'bins':bins,'train_conditions':bins*train_per_bin,'eval_conditions':bins*eval_per_bin,
        'eval_candidates':k,'gradient_parameters':len(g),
        'matrices':{name:value.tolist() for name,value in matrices.items()},
        'split_total_matrices':half_matrices.tolist(),
        'surrogate_main_bin_cosines':[[cosine(a,b) for b in main] for a in main],
        'reward_derivative_split_cosines':[cosine(x[0],x[1]) for x in exact_halves],
        'stable_negative_offdiagonal_fraction':float(stable_negative[mask].double().mean()),
        'stable_negative_diagonal_count':int(torch.diagonal(stable_negative).sum()),
        'raw_sgd_decomposition':{'own_bin_contribution':float(diag.sum()),'cross_bin_contribution':float(off),
                                 'total_reward_derivative':float(matrices['total'].mean())},
        'adamw':{'predicted_per_bin':predicted.tolist(),'split_predictions':predicted_halves.tolist(),
                 'finite_difference_per_bin':finite.tolist(),'finite_difference_relative_error':relative,
                 'derivative_valid':relative<.1,'predicted_mean':float(predicted.mean()),
                 'stable_negative_bins':int(((predicted_halves[0]<0)&(predicted_halves[1]<0)).sum()),
                 'parameter_rms':float(displacement.square().mean().sqrt()),'probe_fraction':scale},
        'scope':'Local derivatives on independent sampled c/noise; not a training result, confidence interval, or production attribution'}


def run(directory, output):
    output=lab.toy_path(output);output.mkdir(parents=True,exist_ok=False)
    cfg,source=lab.load_source();data=Data(cfg)
    plan=json.loads((directory/'plan.json').read_text())
    reward=Critic(True).eval().requires_grad_(False)
    reward.load_state_dict(torch.load(lab.HISTORY/'rewards/full_reward.pt',weights_only=True,map_location='cpu')['model'])
    report={'state':'running','source':str(directory),'anchors':{}}
    started=time.monotonic()
    for step in (25,1000):
        for arm in ('raw','fourier'):
            with torch.random.fork_rng():
                torch.manual_seed(plan['initialization_seed'])
                model=lab.ConditioningAblation(cfg,source.state_dict(),**plan['conditioning'][arm]).eval()
            reference=copy.deepcopy(model).requires_grad_(False)
            checkpoint=torch.load(directory/f'{arm}_step{step}.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            result=probe(model,reference,reward,data,cfg,checkpoint)
            report['anchors'][f'{arm}_{step}']=result
            report['elapsed_seconds']=time.monotonic()-started
            atomic_json(output/'report.json',report)
            print(json.dumps({'anchor':f'{arm}_{step}', 'adamw':result['adamw'],
                              'raw_sgd':result['raw_sgd_decomposition']},allow_nan=False),flush=True)
    report['state']='completed';atomic_json(output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=lab.ARTIFACTS/'conditioning_closed_loop_20260926/round04')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1);run(args.source,args.output)


if __name__=='__main__':main()
