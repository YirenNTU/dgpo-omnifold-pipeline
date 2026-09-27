from dataclasses import replace
import torch
from experiments.dgpo_toy.conditional import Config,Distribution,generator
from experiments.dgpo_toy.check_classifier_direction import nominal_ratio,candidate_metrics
from experiments.dgpo_toy.structure_metrics import structure_features
from experiments.dgpo_toy.broad_coverage import region_hits


def test_nominal_identity_and_perfect_ranking():
    torch.set_num_threads(1)
    d=Distribution(replace(Config(),dimensions=6));r=generator(7)
    c=d.contexts(64,r)[:,None].expand(-1,8,-1)
    y=d.mean(c)+torch.randn(64,8,6,generator=r)
    score=nominal_ratio(y,c,d,0.)
    torch.testing.assert_close(score,d.nominal_log_ratio(y,c))
    result=candidate_metrics(score,score,region_hits(y,c,d).float(),structure_features(y,c,d),d,8)
    assert abs(result['mean_within_context_spearman']-1)<1e-10
    assert result['top_score_nominal_logratio_gain']['gain']>0
