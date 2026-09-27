from dataclasses import replace
import torch
from experiments.dgpo_toy.conditional import Config,Distribution,generator,Denoiser
from experiments.dgpo_toy.broad_coverage import coverage,region_hits


def test_broad_target_contains_joint_signal_and_normal_marginals():
    cfg=replace(Config(),kappa=2.)
    d=Distribution(cfg);r=generator(98);c=d.contexts(40000,r)
    y=d.sample(c,r,truth=True);z=y-d.mean(c)
    assert float(d.joint_signal(y,c).mean())>.5
    assert z.mean(0).abs().max()<.025
    assert (z.var(0)-1).abs().max()<.04
    assert region_hits(y,c,d).float().mean()>.1


def test_coverage_is_nested_and_reproducible():
    torch.set_num_threads(1)
    cfg=replace(Config(),eval_events=8,hidden=8,ddim_steps=2)
    d=Distribution(cfg);model=Denoiser(cfg)
    a=coverage(model,d,cfg);b=coverage(model,d,cfg)
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert (a['any_k128']>=a['any_k8']).all()
