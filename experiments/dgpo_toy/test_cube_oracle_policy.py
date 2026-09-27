import torch
from experiments.dgpo_toy.cube_oracle_policy import paired_bins


def test_paired_condition_bins_keep_opposite_responses():
    before=torch.zeros(64,8);after=torch.ones(64,8)
    after[:32]*=-1
    bins=paired_bins(after,before)
    assert len(bins)==8
    assert all(v["gain"]==(-1 if i<4 else 1) for i,v in enumerate(bins.values()))
