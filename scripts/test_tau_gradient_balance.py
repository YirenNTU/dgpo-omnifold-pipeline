import copy
import math

import pytest
import torch
from torch import nn

from scripts.tau_gradient_balance import EMAGradientBalance, validate_reference_balance
from scripts.test_tau_full_trajectory import controller


def balance_cfg(**kwargs):
    cfg = dict(mode='ema_norm', target_ratio=1., ema_decay=.9, min_coefficient=1.,
                max_coefficient=30., initial_coefficient=5., max_change_factor=2.,
                epsilon=1e-8)
    cfg.update(kwargs)
    return cfg


def test_ema_bounds_rate_limit_and_zero_gradients_are_transactional():
    balance = EMAGradientBalance(balance_cfg())
    original = balance.state_dict()
    state, metrics = balance.propose(100., 1.)
    assert state['coefficient'] == 10.
    assert metrics['bound_active'] and metrics['rate_limit_active']
    assert balance.state_dict() == original
    balance.load_state_dict(state)
    state, metrics = balance.propose(0., 0.)
    assert state['coefficient'] == 10.
    assert metrics['held_for_zero_gradient']
    assert state['ema_reward'] == 100.
    balance.load_state_dict(state)
    state, _ = balance.propose(10., 2.)
    assert state['ema_reward'] == pytest.approx(91.)
    assert state['ema_reference'] == pytest.approx(1.1)
    assert state['coefficient'] == 20.


def test_checkpoint_roundtrip_continues_identically_and_rejects_changed_recipe():
    a = EMAGradientBalance(balance_cfg())
    a.load_state_dict(a.propose(20., 4.)[0])
    b = EMAGradientBalance(balance_cfg())
    b.load_state_dict(a.state_dict())
    assert a.propose(12., 6.) == b.propose(12., 6.)
    bad = copy.deepcopy(a.state_dict()); bad['config']['target_ratio'] = .5
    with pytest.raises(ValueError, match='Incompatible'):
        b.load_state_dict(bad)
    for key, value in [('coefficient', float('nan')), ('ema_reference', -1), ('steps', -1)]:
        bad = copy.deepcopy(a.state_dict()); bad[key] = value
        with pytest.raises(ValueError):
            b.load_state_dict(bad)


@pytest.mark.parametrize('field,value', [('ema_decay', 1.), ('min_coefficient', 0.),
    ('initial_coefficient', 31.), ('max_change_factor', .5), ('target_ratio', True),
    ('epsilon', float('nan')), ('mode', 'unknown')])
def test_bad_configuration_rejected(field, value):
    cfg = balance_cfg(); cfg[field] = value
    with pytest.raises(ValueError):
        validate_reference_balance(cfg)


def test_gradient_combination_only_scales_reference_and_waits_for_commit():
    model = nn.Linear(2, 1, bias=False).double()
    c = controller(); c.balance = EMAGradientBalance(balance_cfg(max_change_factor=100.))
    reference = torch.tensor([[1., 2.]], dtype=torch.float64)
    reward = torch.tensor([[3., -4.]], dtype=torch.float64)
    c._reference_grads = [reference.clone()]
    model.weight.grad = reference+reward
    metrics = c.finalize_gradients(model, world_size=1, device=torch.device('cpu'))
    coefficient = metrics['trajectory/reference_coefficient']
    assert coefficient == pytest.approx(5/math.sqrt(5), rel=1e-8)
    torch.testing.assert_close(model.weight.grad, reward+coefficient*reference)
    assert metrics['trajectory/global_reward_gradient_norm'] == 5.
    assert metrics['trajectory/global_unweighted_reference_gradient_norm'] == pytest.approx(math.sqrt(5))
    assert c.balance.coefficient == 5.  # Not committed until optimizer success.
    with pytest.raises(RuntimeError, match='uncommitted'):
        c.balance_state_dict()
    c.commit_balance()
    assert c.balance_state_dict()['steps'] == 1
    assert c.balance.coefficient == coefficient
    assert c._reference_grads is None


def test_zero_reference_keeps_previous_coefficient_without_creating_gradient():
    model = nn.Linear(2, 1, bias=False)
    c = controller(); c.balance = EMAGradientBalance(balance_cfg())
    c._reference_grads = [None]
    model.weight.grad = torch.tensor([[2., 3.]])
    before = model.weight.grad.clone()
    metrics = c.finalize_gradients(model, world_size=1, device=torch.device('cpu'))
    torch.testing.assert_close(model.weight.grad, before)
    assert metrics['trajectory/reference_coefficient'] == 5.
    assert metrics['trajectory/balance/held_for_zero_gradient'] == 1.
    c.commit_balance()


def test_dynamic_launch_preserves_controls_and_comparison_labels(tmp_path):
    from pathlib import Path
    import numpy as np
    from scripts import diagnose_tau_reward_mechanisms as launch
    from scripts.test_tau_reward_transfer_launch import source_runtime, saved_state
    root = Path(__file__).resolve().parents[1]
    settings = launch.read_mapping(root/'config/tau_full_trajectory_1920_mb64_dynamic.yaml')
    original = source_runtime()
    resolved = launch.configure(original, settings, launch.source_metadata(saved_state()),
        pinned=Path('/test/source.ckpt'), output=Path('/test/out'), stage='trajectory', method='pathwise')
    assert resolved['dgpo']['reference_trust'] == original['dgpo']['reference_trust']
    assert resolved['experiment']['reference_balance'] == settings['full_trajectory']['reference_balance']
    assert 'EMA norms' in resolved['logger']['wandb']['run_name']
    with pytest.raises(ValueError, match='requires --method pathwise'):
        launch.configure(original, settings, launch.source_metadata(saved_state()),
            pinned=Path('/test/source.ckpt'), output=Path('/test/out'), stage='trajectory', method='native')
    records = []
    for dynamic in (False, True):
        path = tmp_path/f'{dynamic}.npz'
        np.savez(path, source_ids=np.arange(3), weight=np.ones(3), truth_cij=np.zeros((3,9)),
                 generated_cij=np.ones((3,9)), reward=np.ones((3,8)))
        contract = copy.deepcopy(settings['full_trajectory'])
        if not dynamic:
            contract.pop('reference_balance'); contract.pop('reference_coefficient')
        records.append(dict(stage='trajectory',reward_arm='inherited',method='pathwise',
            native_training_directory='/same',source_checkpoint='/source',validation_seed=42,
            candidates=8,native_replay_valid=True,native_replay_fingerprint='same',
            last_relative_step=50,output=str(path),endpoint_measurements=str(path),
            ensemble_directory=None,trajectory_contract=contract))
    comparison = launch._paired_cross_arm(settings, records)
    assert len(comparison) == 1 and comparison[0]['intervention'] == 'reference_balance'
    assert comparison[0]['after_reference_balance']['mode'] == 'ema_norm'
    records[0]['method'] = 'native'
    assert launch._paired_cross_arm(settings, records) == []


def _distributed_balance_worker(rank, rendezvous, output):
    import json
    from datetime import timedelta
    from pathlib import Path
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        model = nn.Linear(2, 1, bias=False).double()
        c = controller(); c.balance = EMAGradientBalance(balance_cfg(max_change_factor=100.))
        reference = torch.tensor([[1., 1. if rank == 0 else -1.]], dtype=torch.float64)
        reward = torch.tensor([[10., 0.]] if rank == 0 else [[-8., 2.]], dtype=torch.float64)
        c._reference_grads = [reference.clone()]
        model.weight.grad = reference+reward
        dist.all_reduce(model.weight.grad); model.weight.grad.div_(2)
        metrics = c.finalize_gradients(model, world_size=2, device=torch.device('cpu'))
        expected_coefficient = math.sqrt(2)/(1+1e-8)
        torch.testing.assert_close(model.weight.grad, torch.tensor([[1+expected_coefficient,1.]], dtype=torch.float64))
        assert metrics['trajectory/global_reward_gradient_norm'] == pytest.approx(math.sqrt(2))
        assert metrics['trajectory/global_unweighted_reference_gradient_norm'] == 1.
        assert metrics['trajectory/extra_reference_allreduces'] == 1.
        c.commit_balance()
        Path(output, f'rank{rank}.json').write_text(json.dumps(c.balance_state_dict()))
    finally:
        dist.destroy_process_group()


def test_distributed_cancellation_uses_norm_of_average_and_shared_state(tmp_path):
    import json
    torch.multiprocessing.spawn(_distributed_balance_worker,
        args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    states = [json.loads((tmp_path/f'rank{rank}.json').read_text()) for rank in range(2)]
    assert states[0] == states[1]
    assert states[0]['coefficient'] == pytest.approx(math.sqrt(2)/(1+1e-8))
