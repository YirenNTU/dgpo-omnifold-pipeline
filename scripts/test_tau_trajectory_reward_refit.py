"""Unbounded BCE reward, startup anchor transaction and isolated launch checks."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from scripts.test_tau_full_trajectory import Trunk
from scripts.test_dgpo_tau_ratio import example_bundle, live_batch
from scripts.test_tau_reward_transfer_launch import source_runtime, saved_state
from scripts import diagnose_tau_reward_mechanisms as launch
from scripts.tau_trajectory_reward_refit import validate_reward_refit
from scripts.train_conditional_spin_ratio import build_classifier
from RL.DGPO_neutrino.conditional_tau_reward import ConditionalTauReward, validate_head
from RL.DGPO_neutrino.conditional_tau_cycle import ConditionalTauCycle, fit_head
from RL.DGPO_neutrino.tau_pathwise_reward import compute_reward_grad
from RL.DGPO_neutrino.tau_trajectory_reward_refit import prepare_trajectory_reward

ROOT = Path(__file__).resolve().parents[1]
OPTIONS = dict(anchor_step=1920, ratio_bound=None, reference_recenter=True)


def unbounded_bundle():
    bundle = example_bundle()
    bundle['head'].update(ratio_bound=None, fitting_current_policy=True, source_policy_step=1920)
    model = build_classifier(bundle['head'])
    with torch.no_grad():
        model.readout.bias.fill_(10)
    bundle['head']['state_dict'] = copy.deepcopy(model.state_dict())
    return bundle


def test_unbounded_real_reward_parity_frozen_input_gradient_and_checkpoint_restore():
    reward = ConditionalTauReward(example_bundle(), Trunk(), 'cpu')
    reward.install(unbounded_bundle()['head'], policy_step=1920, epoch=191)
    batch = live_batch(3)
    candidate = torch.randn(2, 3, 2, 2, requires_grad=True)
    oracle = reward.compute(candidate.detach(), batch)
    actual = compute_reward_grad(reward, candidate, batch)
    torch.testing.assert_close(actual, oracle, atol=3e-5, rtol=3e-5)
    assert (actual > np.log(30)).all()
    actual.sum().backward()
    assert candidate.grad.norm() > 0 and torch.isfinite(candidate.grad).all()
    assert all(p.grad is None for p in reward.head.parameters())
    assert all(p.grad is None for p in reward.backbone.parameters())
    payload = reward.stack_payload()
    assert payload['head']['ratio_bound'] is None
    other = ConditionalTauReward(example_bundle(), Trunk(), 'cpu')
    other.load_stack_payload(payload)
    assert other.checkpoint_metadata()['reward'] == 'unbounded_log_ratio'
    assert other.checkpoint_metadata()['ratio_bound'] is None
    assert other.denominator_step == 1920
    # Same feature trunk is restored independently; compare head predictions.
    c = torch.randn(3, reward.head.condition_encoder[0].in_features)
    x = torch.randn(3, reward.installed_head['candidate_dim'])
    torch.testing.assert_close(other.head(c, x), reward.head(c, x), rtol=0, atol=0)


def test_removing_bound_from_old_weights_requires_fresh_fit_provenance():
    saved = example_bundle()['head']
    saved['ratio_bound'] = None
    with pytest.raises(ValueError, match='freshly BCE-fitted'):
        validate_head(saved)


@pytest.mark.parametrize('change', [dict(anchor_step=1880), dict(reference_recenter=False),
    dict(ratio_bound=1), dict(ratio_bound=True), dict(ratio_bound='none')])
def test_refit_contract_rejects_ambiguous_or_changed_anchor(change):
    with pytest.raises(ValueError):
        validate_reward_refit(OPTIONS | change)


@pytest.mark.parametrize('bound', [None, 30])
def test_fit_uses_same_paired_training_split_and_seed_with_explicit_bound(tmp_path, bound):
    cycle = object.__new__(ConditionalTauCycle)
    cycle.reward = SimpleNamespace(bundle=example_bundle(), round_id=20)
    cycle.cfg = dict(fit={'min_steps':1000}, audit_seed=43, refit_seed=42, production_cross_attention=True)
    cycle.device, cycle.rank, cycle.world = torch.device('cpu'), 0, 1
    cycle.emit = lambda *_: None
    panel = dict(source_ids=np.arange(12), condition=np.zeros((12, 2)), candidate_truth=np.ones((12, 3)),
        event_weight=np.ones(12), split=np.arange(12)%3)
    generated = dict(features=-panel['candidate_truth'])
    with patch('RL.DGPO_neutrino.conditional_tau_cycle.fit_head') as fit:
        fit.side_effect = lambda cfg, data, *args: (None, cfg, {'total_steps':1000})
        _, saved, _, data = cycle.fit(panel, generated, tmp_path, audit=False, step=1920, reward_ratio_bound=bound)
    assert saved['ratio_bound'] is bound
    assert saved['seed'] == 42 and saved['fitting_current_policy'] is True
    assert saved['source_policy_step'] == 1920
    assert data['split'] is panel['split']
    assert data['candidate_truth'].shape == data['candidate_generated'].shape
    np.testing.assert_equal(data['event_weight'], panel['event_weight'])
    assert cycle.reward.bundle['head']['ratio_bound'] == 30


def test_actual_unbounded_balanced_bce_fit_has_finite_trainable_head(tmp_path):
    cfg = example_bundle()['head'] | dict(ratio_bound=None, seed=123, batch_size=8, epochs=3,
        lr=.002, min_lr=.0001, weight_decay=0., patience=10, min_delta=0., min_steps=2,
        fitting_current_policy=True, source_policy_step=1920)
    n = 32
    rng = np.random.default_rng(1)
    arrays = dict(condition=rng.normal(size=(n,cfg['condition_dim'])).astype('float32'),
        candidate_truth=rng.normal(1,.2,size=(n,cfg['candidate_dim'])).astype('float32'),
        candidate_generated=rng.normal(-1,.2,size=(n,cfg['candidate_dim'])).astype('float32'),
        event_weight=np.ones(n), split=np.r_[np.zeros(24,int),np.ones(8,int)])
    head, saved, info = fit_head(cfg, arrays, tmp_path/'best.pt', torch.device('cpu'), 0, 1, lambda _:None)
    assert info['minimum_fit_steps_met'] and info['total_steps'] == 9
    assert saved['ratio_bound'] is None and head.ratio_bound is None
    assert all(torch.isfinite(p).all() and not p.requires_grad for p in head.parameters())


def startup_fixture():
    actor, reference = torch.nn.Linear(2,2), torch.nn.Linear(2,2).eval().requires_grad_(False)
    reward = ConditionalTauReward(example_bundle(), Trunk(), 'cpu')
    panel = dict(source_ids=np.arange(416701), candidate_truth=np.empty((416701,0)), split=np.arange(416701)%3)
    saved = unbounded_bundle()['head'] | dict(min_steps=1000)
    def fit(panel, generated, *args, **kwargs):
        assert kwargs['audit'] is False and kwargs['step'] == 1920 and kwargs['reward_ratio_bound'] is None
        return None, saved, dict(total_steps=1000, minimum_fit_steps_met=True), panel
    cycle = SimpleNamespace(actor=actor, reference=reference, reward=reward,
        validation=dict(source_ids=np.arange(119002)+1000000), rank=0, world=1, device=torch.device('cpu'),
        cfg=dict(train_panel='filtered-train', refit_seed=42), emit=lambda *_:None,
        read_panel=lambda _:panel, fit=fit)
    cycle.generate = lambda p,*args: dict(features=p['candidate_truth'].copy())
    return cycle, saved


def test_startup_installs_matching_1920_anchor_without_actor_optimizer_update(tmp_path):
    cycle, _ = startup_fixture()
    optimizer = torch.optim.AdamW(cycle.actor.parameters(), lr=.03)
    cycle.actor(torch.ones(3,2)).sum().backward(); optimizer.step()
    before_actor = copy.deepcopy(cycle.actor.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    report = prepare_trajectory_reward(cycle, OPTIONS, tmp_path, epoch=191, step=1920)
    assert report['truth_events'] == report['anchor_events'] == 416701
    assert report['per_class_split_events'] == {'0':138901, '1':138900, '2':138900}
    assert cycle.reward.denominator_step == 1920 and cycle.reward.installed_head['ratio_bound'] is None
    for key,value in before_actor.items():
        torch.testing.assert_close(cycle.actor.state_dict()[key],value,rtol=0,atol=0)
        torch.testing.assert_close(cycle.reference.state_dict()[key],value,rtol=0,atol=0)
    after_optimizer = optimizer.state_dict()
    assert before_optimizer['param_groups'] == after_optimizer['param_groups']
    for key,state in before_optimizer['state'].items():
        for field,value in state.items():
            torch.testing.assert_close(after_optimizer['state'][key][field],value,rtol=0,atol=0)


def test_undertrained_startup_fails_before_install_or_recenter(tmp_path):
    cycle, saved = startup_fixture()
    old_head = cycle.reward.head
    before = copy.deepcopy(cycle.reference.state_dict())
    cycle.fit = lambda *args,**kwargs:(None,saved,dict(total_steps=999,minimum_fit_steps_met=False),cycle.train)
    with pytest.raises(ValueError,match='fit budget'):
        prepare_trajectory_reward(cycle, OPTIONS, tmp_path, epoch=191, step=1920)
    assert cycle.reward.head is old_head and cycle.reward.denominator_step == 0
    for key,value in before.items():
        torch.testing.assert_close(cycle.reference.state_dict()[key],value,rtol=0,atol=0)


def test_configs_match_except_ratio_bound_and_launcher_keeps_original_source(tmp_path):
    a,b = [yaml.safe_load((ROOT/f'config/tau_full_trajectory_1920_mb64_fresh_{label}.yaml').read_text())
           for label in ('unbounded','bound30')]
    aa,bb = copy.deepcopy(a),copy.deepcopy(b)
    assert aa['full_trajectory']['reward_refit'].pop('ratio_bound') is None
    assert bb['full_trajectory']['reward_refit'].pop('ratio_bound') == 30
    assert aa == bb
    original = source_runtime(); before = copy.deepcopy(original)
    cfg = launch.configure(original,a,launch.source_metadata(saved_state()),pinned=tmp_path/'source.ckpt',
        output=tmp_path,stage='trajectory',method='pathwise')
    assert original == before
    assert cfg['dgpo']['reference_trust']['coefficient'] == 1
    assert cfg['experiment']['reward_refit'] == OPTIONS
    assert 'unbounded' in cfg['logger']['wandb']['run_name']
    assert cfg['dgpo']['tau_ratio']['mechanism_probe']['periodic_refits'] is False
    with pytest.raises(ValueError):
        launch.configure(original,a,launch.source_metadata(saved_state()),pinned=tmp_path/'source.ckpt',
            output=tmp_path,stage='trajectory',method='native')
