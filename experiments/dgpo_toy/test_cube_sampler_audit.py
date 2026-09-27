import torch
from dataclasses import replace
from experiments.dgpo_toy.conditional import Config,alpha_sigma
from experiments.dgpo_toy.parity_cube import CubeDistribution
from experiments.dgpo_toy.cube_sampler_audit import ExactCubeVelocity


def test_analytic_velocity_matches_density_score_identity():
    data=CubeDistribution(replace(Config(),dimensions=3,context_dim=1),continuous=True)
    oracle=ExactCubeVelocity(data)
    x=torch.tensor([[.3,-.8,1.2],[-.4,.2,.9]],dtype=torch.float64,requires_grad=True)
    c=torch.tensor([[-.7],[.4]],dtype=torch.float64)
    t=torch.tensor([.23,.81],dtype=torch.float64)
    a,s=alpha_sigma(t);var=a*a*data.width**2+s*s
    logp=torch.logsumexp(data.probabilities(c[:,0]).log()-
        (x[:,None]-a[:,None,None]*data.centers).square().sum(-1)/(2*var[:,None]),dim=-1)
    score=torch.autograd.grad(logp.sum(),x)[0]
    expected=-s[:,None]/a[:,None]*(score+x)
    torch.testing.assert_close(oracle(x,t,c),expected,rtol=1e-10,atol=1e-10)


def test_analytic_velocity_endpoint_and_shapes():
    data=CubeDistribution(replace(Config(),dimensions=3,context_dim=1),continuous=True)
    model=ExactCubeVelocity(data);x=torch.randn(2,8,3);c=torch.tensor([[-.5],[.7]])[:,None]
    for t in (0.,1.,.5):
        y=model(x,torch.tensor(t),c)
        assert y.shape==x.shape and torch.isfinite(y).all()
    # Existing finite-logSNR schedule has nonzero sigma even at t=0.
    _,sigma=alpha_sigma(torch.tensor(0.))
    assert sigma>0
    assert model(x,torch.tensor(0.),c).abs().max()<.1
