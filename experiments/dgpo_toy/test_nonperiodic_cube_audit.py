import torch
from experiments.dgpo_toy.nonperiodic_cube_audit import auc_contrasts


def test_identical_scores_have_exact_zero_paired_difference():
    p={'positive':torch.tensor([1.,2.,3.,4.]),'negative':torch.tensor([-1.,0.,1.,2.])}
    result=auc_contrasts({'baseline':p,'raw':p,'fourier':p},repeats=10)
    assert all(v['delta_auc']==v['lo95']==v['hi95']==0 for v in result.values())


def test_auc_contrast_orientation_and_ties():
    p={'positive':torch.ones(10),'negative':torch.zeros(10)}
    q={'positive':torch.zeros(10),'negative':torch.zeros(10)}
    r=auc_contrasts({'baseline':p,'raw':q,'fourier':q},repeats=10)
    assert all(v['delta_auc']==v['lo95']==v['hi95']==-.5 for v in r.values())
