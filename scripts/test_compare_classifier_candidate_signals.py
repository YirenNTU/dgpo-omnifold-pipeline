import numpy as np
import pytest
import torch

from compare_classifier_candidate_signals import aligned_rows, signal_report


def test_opposite_ranking_and_constant_offset():
    a = np.array([[0., 1., 2.], [3., 5., 4.]])
    same = signal_report(a, a + 99)
    assert same['within_event_spearman']['mean'] == pytest.approx(1)
    assert same['unique_winner_agreement'] == 1
    opposite = signal_report(a, -a)
    assert opposite['centered_score_cosine']['mean'] == pytest.approx(-1)
    assert opposite['pairwise_order_agreement'] == 0
    assert opposite['unique_winner_agreement'] == 0


def test_constant_scores_do_not_invent_rank_agreement():
    r = signal_report(np.ones((2, 8)), np.ones((2, 8)))
    assert r['within_event_spearman'] is None
    assert r['unique_winner_agreement'] is None
    assert r['pairwise_order_agreement'] is None
    assert r['arms']['old']['softmax_score_ess_fraction']['mean'] == 1


def test_holdout_identity_mapping_and_rejection():
    b = dict(packing_spec={'x': 1}, split_indices=dict(test=torch.tensor([3, 7]), fit=torch.tensor([1]), early_stop=torch.tensor([2])),
             test_truth=torch.tensor([[1.], [2.]]), test_condition=torch.tensor([[3.], [4.]]))
    panel = dict(packing_spec={'x': 1}, pool_rows=torch.tensor([7, 3]), truth=torch.tensor([[2.], [1.]]), condition=torch.tensor([[4.], [3.]]))
    assert aligned_rows(b, panel).tolist() == [1, 0]
    panel['condition'][0, 0] = 0
    with pytest.raises(ValueError, match='not entirely held out'):
        aligned_rows(b, panel)
    panel['condition'][0, 0] = 4
    panel['pool_rows'][0] = 999
    assert aligned_rows(b, panel).tolist() == [1, 0]
    panel['truth'][0, 0] = 0
    with pytest.raises(ValueError, match='truth mismatch'):
        aligned_rows(b, panel)
