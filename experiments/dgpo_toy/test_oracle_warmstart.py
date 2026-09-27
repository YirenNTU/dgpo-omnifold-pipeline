import torch
from dataclasses import replace
from experiments.dgpo_toy.conditional import Config,Distribution,Denoiser,generator
from experiments.dgpo_toy.structure_metrics import structure_target
from experiments.dgpo_toy.oracle_warmstart import oracle_loss,candidate_gate


def test_oracle_auxiliary_reaches_denoiser_parameters():
    torch.set_num_threads(1)
    cfg=replace(Config(),dimensions=6,hidden=8,ddim_steps=2)
    d=Distribution(cfg);r=generator(17);c=d.contexts(64,r)
    m=Denoiser(cfg);y=d.sample(c,r,truth=True)
    loss,_=oracle_loss(m,c,y,torch.randn(64,6,generator=r),d,r,torch.tensor(structure_target(d),dtype=torch.float32)*.3)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert sum(float(p.grad.square().sum()) for p in m.parameters())>0


def test_gate_rejects_marginal_damage_and_complete_alignment():
    good=dict(joint_mean=.2,hit_fraction=.1,mean_absmax=.02,variance_error=.03,pair_covariance=.02)
    assert candidate_gate(good,dict(hit_fraction=.01))
    assert not candidate_gate({**good,'variance_error':2.},dict(hit_fraction=.01))
    assert not candidate_gate({**good,'joint_mean':.8},dict(hit_fraction=.01))
