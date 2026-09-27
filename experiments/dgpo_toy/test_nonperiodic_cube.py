import torch
from experiments.dgpo_toy.nonperiodic_cube import Data,Critic
from experiments.dgpo_toy.conditional import Config, Denoiser
from experiments.dgpo_toy.cube_lockdown import ConditionDenoiser, verify_initial


def test_only_third_order_distribution_changes():
    data=Data(Config(dimensions=3,context_dim=1))
    c=torch.linspace(-1,1,100)
    p=data.probabilities(c,.9);q=data.probabilities(c)
    torch.testing.assert_close(p.sum(-1),torch.ones(100))
    signs=data.centers
    torch.testing.assert_close(p@signs,torch.zeros(100,3),atol=1e-7,rtol=0)
    torch.testing.assert_close(p@(signs*signs.roll(1,-1)),torch.zeros(100,3),atol=1e-7,rtol=0)
    torch.testing.assert_close(p@signs.prod(-1),.8*data.condition_signal(c),atol=1e-7,rtol=1e-5)
    assert (p-q).abs().max()>.08


def test_critic_features_do_not_consult_truth_or_parity():
    plain,fourier=Critic(False),Critic(True)
    y=torch.randn(7,3);c=torch.rand(7,1)
    assert plain.features(y,c).shape==(7,36)
    assert torch.count_nonzero(plain.features(y,c)[:,4:])==0
    assert torch.equal(plain.features(y,c)[:,:4],fourier.features(y,c)[:,:4])
    assert fourier(y,c,object()).shape==(7,)
    assert sum(p.numel() for p in plain.parameters())==sum(p.numel() for p in fourier.parameters())


def test_diffusion_fork_identical_before_update():
    cfg=Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=2)
    source=Denoiser(cfg)
    arms={b:ConditionDenoiser(cfg,b,source.state_dict()) for b in ('raw','fourier')}
    result=verify_initial(source,arms,cfg)
    assert all(x['samples_exact'] and x['velocity_exact'] for x in result.values())
