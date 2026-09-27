from dataclasses import replace
import math
import torch
from experiments.dgpo_toy.conditional import Config,Denoiser,generator,dgpo_objective
from experiments.dgpo_toy.parity_cube import CubeDistribution,ModeReward,cube_metrics,baseline_gate,generation_panel
from experiments.dgpo_toy.parity_cube import prepare_splits


def setup():
    return replace(Config(),dimensions=3,context_dim=1,hidden=16,ddim_steps=3,eval_events=32,batch=4,timesteps=2)


def test_conditional_joint_only_distribution():
    d=CubeDistribution(setup())
    for c in (-1,1):
        for mass in (.001,.1,.9):
            p=d.probabilities(c,mass)
            torch.testing.assert_close(p.sum(),torch.tensor(1.))
            torch.testing.assert_close((p[:,None]*d.centers).sum(0),torch.zeros(3),atol=1e-7,rtol=0)
            for i,j in ((0,1),(0,2),(1,2)):
                for a in (-1,1):
                    for b in (-1,1):
                        torch.testing.assert_close(p[(d.centers[:,i]==a)&(d.centers[:,j]==b)].sum(),torch.tensor(.25))
            torch.testing.assert_close((p*d.centers.prod(-1)).sum(),torch.tensor(c*(2*mass-1)),atol=1e-7,rtol=0)


def test_sample_gate_and_oracle_reward():
    d=CubeDistribution(setup());rng=generator(41)
    c=d.contexts(60000,rng);y=d.sample(c,rng)
    metrics=cube_metrics(y,c,d)
    assert baseline_gate(metrics)
    r=ModeReward()(y,c,d);hit=d.joint_signal(y,c)
    torch.testing.assert_close(r,math.log(9)*(2*hit-1))
    truth=cube_metrics(d.sample(c,rng,truth=True),c,d)
    assert .88<truth["preferred_mass"]<.92
    assert not baseline_gate(truth)


def test_dgpo_and_generation_shapes():
    torch.set_num_threads(1)
    cfg=setup();d=CubeDistribution(cfg);m=Denoiser(cfg)
    ref=Denoiser(cfg);ref.load_state_dict(m.state_dict());ref.requires_grad_(False)
    rng=generator(14);c=d.contexts(cfg.batch,rng)
    loss,row=dgpo_objective(m,ref,ModeReward(),d,cfg,c,
        torch.randn(cfg.batch,8,3,generator=rng),.7*torch.rand(2,cfg.batch,generator=rng),
        torch.randn(2,cfg.batch,3,generator=rng),velocity_coefficient=1.)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    r,h,stats=generation_panel(m,d,cfg,123)
    assert r.shape==h.shape==(32,8) and stats["events"]==256


def test_larger_dataset_preserves_prefix_and_holdouts():
    cfg=setup();d=CubeDistribution(cfg)
    small=prepare_splits(d,cfg,17)
    large=prepare_splits(d,replace(cfg,train_events=131072),17)
    rng=generator(17+610000)
    c=d.contexts(32768,rng);y=d.sample(c,rng)
    assert torch.equal(small["train"]["condition"],c)
    assert torch.equal(small["train"]["target"],y)
    for field in ("condition","target"):
        assert torch.equal(small["train"][field],large["train"][field][:32768])
        for split in ("validation","test"):
            assert torch.equal(small[split][field],large[split][field])


def test_continuous_condition_changes_only_three_way_structure():
    d=CubeDistribution(setup(),continuous=True)
    for c in (-1.,-.83,-.2,0.,.37,1.):
        for mass in (.1,.9):
            p=d.probabilities(c,mass)
            torch.testing.assert_close(p.sum(),torch.tensor(1.))
            torch.testing.assert_close((p[:,None]*d.centers).sum(0),torch.zeros(3),rtol=0,atol=1e-7)
            for i,j in ((0,1),(0,2),(1,2)):
                for a in (-1,1):
                    for b in (-1,1):
                        torch.testing.assert_close(p[(d.centers[:,i]==a)&(d.centers[:,j]==b)].sum(),torch.tensor(.25))
            torch.testing.assert_close((p*d.centers.prod(-1)).sum(),torch.tensor((2*mass-1)*c),rtol=0,atol=1e-7)
    c=torch.tensor([[-.7],[0.],[.4]])
    y=torch.tensor([[1.,1.,1.],[1.,-1.,1.],[-1.,-1.,1.]])
    r=ModeReward()(y,c,d)
    parity=y.prod(-1)
    torch.testing.assert_close(r,((.5+.4*c[:,0]*parity)/(.5-.4*c[:,0]*parity)).log())
    assert r[1]==0 and r[0]<0 and r[2]>0


def test_continuous_samples_and_binned_gate():
    d=CubeDistribution(setup(),continuous=True);rng=generator(142)
    c=d.contexts(120000,rng)
    assert (c.abs()<1).all() and c.unique().numel()>10000
    metrics=cube_metrics(d.sample(c,rng),c,d)
    assert len(metrics["conditions"])==8 and baseline_gate(metrics)
    assert -.29<metrics["condition_weighted_parity"]<-.24
    cfg=setup();m=Denoiser(cfg)
    r,h,stats=generation_panel(m,d,cfg,99,n=64)
    assert r.shape==h.shape==(64,8) and len(stats["conditions"])==8


def test_reference_sharpness_preserves_truth_and_lower_order_marginals():
    cfg=setup();easy=CubeDistribution(cfg,continuous=True)
    hard=CubeDistribution(cfg,continuous=True,reference_sharpness=4.)
    c=torch.linspace(-1,1,1001)
    torch.testing.assert_close(easy.probabilities(c,.9),hard.probabilities(c,.9))
    q=hard.probabilities(c)
    torch.testing.assert_close(q.sum(-1),torch.ones_like(c))
    torch.testing.assert_close(q@hard.centers,torch.zeros(1001,3),atol=1e-7,rtol=0)
    for i,j in ((0,1),(0,2),(1,2)):
        torch.testing.assert_close(q@(hard.centers[:,i]*hard.centers[:,j]),torch.zeros_like(c),atol=1e-7,rtol=0)
    assert float(hard.positive_probability(torch.tensor(1.)))<.001
    y=hard.centers[None].expand(len(c),-1,-1);cc=c[:,None,None].expand(-1,8,1)
    r=ModeReward()(y,cc,hard)
    torch.testing.assert_close(r,(hard.probabilities(c,.9)/q).log(),atol=2e-5,rtol=2e-5)
