import torch
import pytest
from dataclasses import replace
from experiments.dgpo_toy.conditional import Config,Distribution,generator,Denoiser
from experiments.dgpo_toy.oracle_warmstart import sample_teacher,sample_teacher_loss
from experiments.dgpo_toy.broad_coverage import region_hits


def test_teacher_marginals_joint_and_repeatability():
    torch.set_num_threads(1)
    d=Distribution(Config());r=generator(49)
    c=d.contexts(50000,r);z=torch.randn(50000,12,generator=r)
    y=sample_teacher(c,z,d)
    assert torch.equal(y,sample_teacher(c,z,d))
    torch.testing.assert_close(sample_teacher(c,z,d,0.),d.mean(c)+z)
    residual=y-d.mean(c);cov=torch.cov(residual.T)
    assert residual.mean(0).abs().max()<.025
    assert (cov.diag()-1).abs().max()<.04
    assert (cov-torch.diag(cov.diag())).abs().max()<.025
    assert abs(float(d.joint_signal(y,c).mean())-.3)<.015
    assert region_hits(y,c,d).float().mean()>.04
    with pytest.raises(ValueError):sample_teacher(c,z,d,1.)


def test_teacher_loss_backpropagates_only_through_model():
    cfg=replace(Config(),dimensions=6,hidden=8,ddim_steps=2)
    d=Distribution(cfg);r=generator(51);c=d.contexts(32,r)
    z=torch.randn(32,6,generator=r);m=Denoiser(cfg)
    target=sample_teacher(c,z,d)
    assert not target.requires_grad
    loss,_=sample_teacher_loss(m,c,z,d);loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
