import itertools
from unittest.mock import patch

import numpy as np
import pytest
import torch

from experiments.dgpo_toy import classifier_signal_attribution as a
from experiments.dgpo_toy import cube_swap
from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy.nonperiodic_cube import Data


CENTERS = torch.tensor(list(itertools.product((-1.,1.),repeat=3)))


def test_walsh_orthogonality_and_mode_order():
    h,order = a.walsh(CENTERS)
    torch.testing.assert_close(h.T@h,torch.eye(8,dtype=torch.float64)*8)
    assert order.tolist()==[0,1,1,1,2,2,2,3]


def test_each_signal_order_is_isolated_and_identity_holds():
    h,orders = a.walsh(CENTERS)
    q = torch.full((2,8),.125,dtype=torch.float64)
    for column in range(1,8):
        p = q+.04*h[:,column]
        mean = h[:,column].expand(2,-1)
        out = a.score_decomposition(p,q,mean,mean,CENTERS)
        for k in (1,2,3):
            expected = .32 if k==orders[column] else 0.
            torch.testing.assert_close(out[f'order{k}_mode_logit_gap'],torch.full((2,),expected,dtype=torch.float64))
        assert out['within_mode_shape_logit_gap'].abs().max()==0
        assert out['identity_residual'].abs().max()<1e-12


def test_pure_shape_and_constant_condition_offset():
    q=torch.full((3,8),.125,dtype=torch.float64)
    truth=torch.arange(3,dtype=torch.float64)[:,None].expand(-1,8)
    out=a.score_decomposition(q,q,truth,truth-.3,CENTERS)
    torch.testing.assert_close(out['within_mode_shape_logit_gap'],torch.full((3,),.3,dtype=torch.float64))
    for k in (1,2,3):assert out[f'order{k}_mode_logit_gap'].abs().max()==0


def test_nonfinite_and_invalid_probability_fail_closed():
    q=torch.full((2,8),.125,dtype=torch.float64)
    with pytest.raises(FloatingPointError):
        a.score_decomposition(q,q,q*float('nan'),q,CENTERS)
    with pytest.raises(ValueError):
        a.score_decomposition(q*2,q,q,q,CENTERS)


def test_structure_parseval_and_independent_half_estimator():
    ids=torch.arange(8).repeat(32)[None].expand(3,-1)
    y=CENTERS[ids]
    h,_=a.walsh(CENTERS)
    p=torch.full((3,8),.125,dtype=torch.float64)+.03*h[:,-1]
    result=a.structure({'y':y,'ids':ids,'p':p},CENTERS)
    torch.testing.assert_close(result['mode_l2'],torch.full((3,),8*.03**2,dtype=torch.float64))
    torch.testing.assert_close(result['order3_mode_l2_debiased'],result['mode_l2'])
    assert result['order1_mode_l2'].abs().max()==0
    assert result['order2_mode_l2'].abs().max()==0


def test_bce_aggregates_by_condition_and_paired_intervals_cancel():
    scores={'positive':torch.tensor([0.,1.,2.,3.]),'negative':torch.tensor([-1.,-2.,-3.,-4.])}
    values=a.bce_by_condition(scores,2)
    assert values.shape==(2,)
    out=a.simultaneous_mean_intervals({'zero':values-values},100,17)
    assert out['zero']=={'mean':0.,'lo95':0.,'hi95':0.}


def test_swaps_preserve_identities_and_old_api():
    torch.set_num_threads(1)
    data=Data(native.Config(dimensions=3,context_dim=1))
    def fake_ddim(model,c,z,steps):
        ids=torch.arange(z.shape[1])%8
        return CENTERS[ids][None].expand(z.shape[0],-1,-1).clone()
    with patch.object(cube_swap.native,'ddim',side_effect=fake_ddim):
        panels,health,pool=cube_swap.swaps(None,data,2,64,128,17,return_pool=True)
        legacy=cube_swap.swaps(None,data,2,64,128,17)
    assert len(legacy)==2 and health['min_cell_count']==16
    assert torch.equal(cube_swap.modes(panels['A']['negative']),cube_swap.modes(panels['C']['negative']))
    assert torch.equal(cube_swap.modes(panels['B']['negative']),cube_swap.modes(panels['D']['negative']))
    for name,p in panels.items():
        assert torch.equal(p['positive'],panels['A']['positive'])
        assert torch.equal(p['negative'],legacy[0][name]['negative'])
    assert pool['y'].shape==(2,128,3)


def test_cell_means_preserve_mode_and_condition_axes():
    class Parity(torch.nn.Module):
        def forward(self,y,c):return y.prod(-1)+c[:,0]
    ids=torch.arange(8)[None].expand(2,-1).repeat(1,16)
    c=torch.tensor([[-.5],[.5]])
    pool={'y':CENTERS[ids],'ids':ids,'c':c}
    truth=CENTERS[None,:,None].expand(2,-1,5,-1)
    t,g=a.cell_means(Parity(),pool,truth)
    expected=(CENTERS.prod(-1)[None]+c).double()
    torch.testing.assert_close(t,expected)
    torch.testing.assert_close(g,expected)
