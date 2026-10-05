import numpy as np
import pytest
from experiments.dgpo_toy.reward_retention import analyze, native_panel


def test_accounting_and_reproduction():
    z = np.zeros((20, 8))
    result = analyze(z, z+1, z+.1, repeats=100)
    assert result['strong_aggregate_regression']
    assert result['accounting']['erased_early_gains'] == pytest.approx(.9)
    assert result['accounting']['additional_damage'] == pytest.approx(0)
    assert result['accounting']['reconstruction_error'] < 1e-12


def test_aggregate_can_hide_condition_losses():
    z = np.zeros((20, 8)); e = z+1; last = z.copy()
    last[:10] = 3
    result = analyze(z, e, last, repeats=100)
    assert not result['strong_aggregate_regression']
    assert result['metrics']['later_change']['mean'] == .5
    assert result['fractions']['later_negative'] == .5


def test_split_does_not_reuse_winner_noise():
    z = np.zeros((20, 8)); e = z.copy(); e[:, ::2] = 1; e[:, 1::2] = -1
    result = analyze(z, e, z, repeats=100)
    group = result['crossfit_groups']['early_beneficiaries']
    assert group['fraction'] == .5
    assert group['conditional_means']['early_gain'] == -1
    for metric in ('early_gain','later_change','net_gain'):
        total = sum(g['population_contributions'][metric]['mean'] for g in result['crossfit_groups'].values())
        assert total == pytest.approx(result['metrics'][metric]['mean'])


def test_native_identity_alignment(tmp_path):
    for step, ids, value in ((0, ['a','b'], [1,2]), (5, ['b','a'], [3,2]), (20, ['a','b'], [4,5])):
        np.savez(tmp_path/f'heldout_step{step:02d}_rank00.npz',event_ids=ids,rewards=np.repeat(np.array(value)[:,None],4,1))
    panels = native_panel(tmp_path, expected_ranks=1)
    np.testing.assert_array_equal(panels[1][:,0], [2,3])
    np.savez(tmp_path/'heldout_step20_rank00.npz',event_ids=['a','a'],rewards=np.zeros((2,4)))
    with pytest.raises(ValueError,match='Duplicate'):
        native_panel(tmp_path, expected_ranks=1)


def test_validation():
    z=np.zeros((10,8))
    with pytest.raises(ValueError,match='identical'):
        analyze(z,z[:2],z)
    with pytest.raises(ValueError,match='finite'):
        analyze(z,z,z+np.nan)
    with pytest.raises(ValueError,match='even'):
        analyze(z[:,:3],z[:,:3],z[:,:3])


def test_native_rejects_partial_distributed_panel(tmp_path):
    for step in (0,5,20):
        np.savez(tmp_path/f'heldout_step{step:02d}_rank00.npz',event_ids=['a','b'],rewards=np.zeros((2,4)))
    with pytest.raises(ValueError,match='Missing native ranks'):
        native_panel(tmp_path)
