import pytest
import torch

from experiments.dgpo_toy.reward_transfer_comparison import (
    assert_same_panel, check_factor, curve_average, gain_distribution, paired_intervals, score,
)
from experiments.dgpo_toy.nonperiodic_cube import Critic


def test_curve_average_linear_and_invalid_clock():
    assert torch.allclose(curve_average([0,25,100], [torch.ones(5)*t for t in [0,25,100]]),
                          torch.full((5,),50.,dtype=torch.float64))
    with pytest.raises(ValueError):
        curve_average([0,25,20],[torch.zeros(5)]*3)


def test_pairing_checks_data_not_only_shape():
    p={'c':torch.zeros(4,1),'positive':torch.ones(4,3),'negative':torch.zeros(4,3)}
    q={k:v.clone() for k,v in p.items()}
    assert_same_panel(p,q,include_negative=True)
    q['negative'][0,0]=1
    assert_same_panel(p,q)
    with pytest.raises(ValueError):assert_same_panel(p,q,include_negative=True)
    q['c'][0]=1
    with pytest.raises(ValueError):assert_same_panel(p,q)


def test_declared_factor_rejects_hidden_changes():
    check_factor({'layers':3},{'layers':1},'layers')
    with pytest.raises(ValueError):
        check_factor({'layers':3,'normalization':True},{'layers':1,'normalization':False},'layers')
    with pytest.raises(ValueError):check_factor({'layers':3},{'layers':3},'layers')


def test_bootstrap_uses_paired_differences():
    x=torch.arange(20).float()
    out=paired_intervals({'same':x-x,'offset':x+2-x},repeats=100)
    assert out['same']=={'mean':0.,'lo95':0.,'hi95':0.}
    assert out['offset']=={'mean':2.,'lo95':2.,'hi95':2.}
    with pytest.raises(ValueError):paired_intervals({'bad':torch.tensor([float('nan'),1.])},100)


def test_gain_concentration_and_empty_positive_gain():
    s=gain_distribution(torch.tensor([-1.,1.,3.,0.]))
    assert s['fraction_positive']==.5 and s['top1pct_share_of_positive_gain']==.75
    assert gain_distribution(-torch.ones(10))['top1pct_share_of_positive_gain'] is None


def test_scoring_does_not_modify_model_or_rng():
    model=Critic(True).eval().requires_grad_(False)
    p={'c':torch.zeros(7,1),'negative':torch.ones(7,3)}
    state={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.get_rng_state().clone()
    a=score(model,p);b=score(model,p)
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(state[k],v) for k,v in model.state_dict().items())
