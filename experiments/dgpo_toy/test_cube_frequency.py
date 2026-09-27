import math
import pytest
import torch
from experiments.dgpo_toy.conditional import Config, generator
from experiments.dgpo_toy.parity_cube import CubeDistribution, ModeReward, cube_metrics
from experiments.dgpo_toy.cube_frequency import ideal_metrics, summarize


def distribution(k):
    return CubeDistribution(Config(dimensions=3,context_dim=1),continuous=True,
        reference_sharpness=4.,condition_frequency=k)


def test_integer_frequency_preserves_ideal_difficulty():
    base=ideal_metrics(1)
    for k in (2,4,8):
        for key,value in ideal_metrics(k).items():
            assert value==pytest.approx(base[key],rel=2e-6,abs=2e-7)


def test_reward_and_preferred_corner_use_sine_not_condition_sign():
    data=distribution(2);c=torch.tensor([[.25],[.75]],dtype=torch.float64)
    y=torch.ones(2,3,dtype=torch.float64)
    assert data.joint_signal(y,c).tolist()==[1.,0.]
    p=data.positive_probability(c[:,0],truth=True)
    q=data.positive_probability(c[:,0])
    torch.testing.assert_close(ModeReward()(y,c,data),(p/q).log())
    for truth in (False,True):
        probs=data.probabilities(c[:,0],.9 if truth else None)
        torch.testing.assert_close(probs.sum(-1),torch.ones(2,dtype=torch.float64))
        for i in range(3):
            torch.testing.assert_close(probs[:,data.centers[:,i]>0].sum(-1),torch.full((2,),.5,dtype=torch.float64))


def test_fine_bins_resolve_alternating_truth():
    data=distribution(8)
    c=((torch.arange(32768,dtype=torch.float32)+.5)/32768*2-1)[:,None]
    y=data.sample(c,generator(13),truth=True)
    stats=cube_metrics(y,c,data)
    assert len(stats['conditions'])==256
    assert stats['preferred_mass']>.7
    assert stats['condition_weighted_parity']==pytest.approx(.4,abs=.025)
    for b in stats['conditions'].values():
        assert sum(b['counts'])==128
    result=summarize(stats|{'group_hit_fraction':1.,'reward_mean':0.},torch.zeros(32,8))
    assert result['mean_target_tv']<.16


@pytest.mark.parametrize('k',[0,-1,1.5,True])
def test_invalid_frequency(k):
    with pytest.raises(ValueError):distribution(k)


def test_frequency_policy_smoke(tmp_path,monkeypatch):
    from dataclasses import asdict,replace
    from experiments.dgpo_toy import cube_frequency as runner
    from experiments.dgpo_toy.conditional import Denoiser
    torch.set_num_threads(1)
    cfg=Config(dimensions=3,context_dim=1,hidden=8,ddim_steps=2,batch=4,candidates=4,timesteps=2)
    source=tmp_path/'source';source.mkdir()
    torch.save({'model':Denoiser(cfg).state_dict(),'config':asdict(cfg),
        'condition_frequency':2,'step':1},source/'best_pretrain.pt')
    real_panel=runner.generation_panel
    monkeypatch.setattr(runner,'generation_panel',lambda m,d,c,s:real_panel(m,d,replace(c,eval_events=512),s))
    real_train=runner.policy_train
    def short_train(arm,m,r,d,c,*args,**kwargs):
        return real_train(arm,m,r,d,replace(c,policy_steps=2,eval_every=1,eval_events=64),*args,**kwargs)
    monkeypatch.setattr(runner,'policy_train',short_train)
    result=runner.run_arm(source,tmp_path/'out',2,'disabled')
    assert result['state']=='completed'
    assert result['completed_steps']==2
    assert result['source_unchanged']
    assert len(result['reward_gain_by_condition'])==64
    assert (tmp_path/'out'/'policy.pt').exists()
