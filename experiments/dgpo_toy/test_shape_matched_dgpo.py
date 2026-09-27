import torch
from experiments.dgpo_toy.shape_matched_dgpo import decomposition,make_fit_panel
from experiments.dgpo_toy.nonperiodic_cube import Data
from experiments.dgpo_toy import conditional as native


def test_shape_matched_samples_both_use_generated_shape(monkeypatch):
    monkeypatch.setattr(native,'ddim',lambda model,c,z,steps:z.sign()*(1+.02*z.abs()))
    data=Data(native.Config(dimensions=3,context_dim=1))
    p,h=make_fit_panel(None,data,8,64,17)
    assert h['min_cell_count']>=16
    assert (p['positive'].abs()>=1).all() and (p['negative'].abs()>=1).all()
    assert p['c'].shape==(512,1)


def test_decomposition_separates_mode_and_shape():
    base={'q':torch.tensor([[.5,.5]]),'mode_mean_reward':torch.tensor([[0.,2.]]),
          'reward':torch.tensor([[0.,2.]])}
    changed={'q':torch.tensor([[.25,.75]]),'reward':torch.tensor([[1.,2.]])}
    r=decomposition(changed,base)
    assert r['total_reward_gain']==r['mode_probability_contribution']==.5
    assert r['within_mode_shape_contribution']==0
    changed['q']=base['q']
    r=decomposition(changed,base)
    assert r['mode_probability_contribution']==0
    assert r['within_mode_shape_contribution']==.5
