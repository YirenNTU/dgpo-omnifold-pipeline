"""Small CPU fixtures for the step-1920 diagnostic transaction, no real data."""
import copy
import ast
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evenet_dgpo'))
# Exercise the real class methods without importing optional Lightning/vision
# dependencies. The fixtures deliberately replace sampling and head fitting.
source = Path(__file__).resolve().parents[1]/'evenet_dgpo/RL/DGPO_neutrino/conditional_tau_cycle.py'
class_node = next(node for node in ast.parse(source.read_text()).body
                  if isinstance(node, ast.ClassDef) and node.name == 'ConditionalTauCycle')
namespace = dict(np=np, torch=torch, copy=copy, json=json, Path=Path,
    log=logging.getLogger(__name__), feature_precision=nullcontext, barrier=lambda world: None)
exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(source), 'exec'), namespace)
ConditionalTauCycle = namespace['ConditionalTauCycle']


def make_transaction(tmp_path):
    cycle = object.__new__(ConditionalTauCycle)
    actor = torch.nn.Linear(2, 1)
    reference = copy.deepcopy(actor)
    with torch.no_grad():
        reference.weight.add_(1)
    inherited = torch.nn.Linear(2, 1)
    fresh = torch.nn.Linear(2, 1)
    installed = []
    reward = SimpleNamespace(head=inherited, round_id=20, denominator_step=1880,
        installed_head={'cross_attention': True})
    def install(best, policy_step, epoch):
        installed.append((best, policy_step, epoch))
        reward.head = fresh
        reward.round_id += 1
        reward.denominator_step = policy_step
    reward.install = install
    cycle.actor, cycle.reference, cycle.reward = actor, reference, reward
    cycle.rank, cycle.world, cycle.device, cycle.output = 0, 1, torch.device('cpu'), tmp_path
    cycle.cfg = dict(train_panel='unused', validation_candidates=8, validation_seed=42, refit_seed=42,
        transfer_probe=dict(enabled=True, source_step=1920, expected_source_reward_round=20))
    cycle.validation = {'source_ids': np.arange(416701, 535703)}
    cycle.read_panel = lambda _: {'source_ids': np.arange(416701)}
    calls = []
    def generate(panel, folder, k, seed, physics):
        folder.mkdir(parents=True, exist_ok=True)
        calls.append(('generate', k, physics))
        return {'features': np.zeros((2, 2))}
    cycle.generate = generate
    cycle._score_transfer_head = lambda g, candidates, folder, step, phase: (
        calls.append(('score', phase, id(g), str(candidates))) or {'bce': .68})
    cycle.reward_cij_probe = lambda step, g, phase: calls.append(('reweight', phase, id(g)))
    cycle.fit = lambda *args, **kwargs: (fresh, {'epoch': 12}, {'best_steps': 1020}, {})
    cycle.emit = lambda payload, step: calls.append(('emit', step))
    return cycle, calls, installed, inherited


def test_startup_compares_same_candidates_then_installs_once_without_actor_update(tmp_path):
    cycle, calls, installed, inherited = make_transaction(tmp_path)
    optimizer = torch.optim.AdamW(cycle.actor.parameters(), lr=5e-5)
    cycle.actor(torch.ones(2, 2)).sum().backward(); optimizer.step()
    before = copy.deepcopy(optimizer.state_dict())
    actor_before = copy.deepcopy(cycle.actor.state_dict())
    cycle.prepare_reward_transfer(191, 1920)
    assert [(x[1], x[2]) for x in calls if x[0] == 'generate'] == [(8, True), (1, False)]
    scores = [x for x in calls if x[0] == 'score']
    assert scores[0][2:] == scores[1][2:]
    assert len(installed) == 1 and installed[0][1:] == (1920, 191)
    assert cycle.reward.round_id == 21 and cycle.reward.denominator_step == 1920
    for key, value in actor_before.items():
        torch.testing.assert_close(cycle.actor.state_dict()[key], value)
        torch.testing.assert_close(cycle.reference.state_dict()[key], value)
    assert not any(p.requires_grad for p in cycle.reference.parameters())
    assert optimizer.state_dict()['param_groups'] == before['param_groups']
    for key, entry in before['state'].items():
        for name, value in entry.items():
            torch.testing.assert_close(optimizer.state_dict()['state'][key][name], value)
    assert (tmp_path/'transfer-startup/report.json').is_file()


def test_failed_fresh_comparison_restores_installed_head_and_never_recenters(tmp_path):
    cycle, calls, installed, inherited = make_transaction(tmp_path)
    reference_before = copy.deepcopy(cycle.reference.state_dict())
    def score(*args):
        if args[-1] == 'fresh':
            raise ValueError('fixture scoring failure')
        return {'bce': .68}
    cycle._score_transfer_head = score
    with pytest.raises(ValueError, match='fixture scoring'):
        cycle.prepare_reward_transfer(191, 1920)
    assert cycle.reward.head is inherited and not installed
    assert cycle.reward.round_id == 20 and cycle.reward.denominator_step == 1880
    for key, value in reference_before.items():
        torch.testing.assert_close(cycle.reference.state_dict()[key], value)


def test_transfer_suppresses_normal_epoch_refit_and_audit(tmp_path):
    cycle, calls, _, _ = make_transaction(tmp_path)
    cycle.evaluate = lambda *args: pytest.fail('No routine audit during fixed-head transfer')
    cycle.refit = lambda *args: pytest.fail('No routine refit during fixed-head transfer')
    for epoch in range(192, 197):
        assert not cycle.epoch_end(epoch, (epoch+1)*10)


def test_wrong_source_or_changed_round_fails_before_generation(tmp_path):
    cycle, calls, _, _ = make_transaction(tmp_path)
    with pytest.raises(ValueError, match='pinned source step'):
        cycle.prepare_reward_transfer(191, 1930)
    cycle.reward.round_id = 21
    with pytest.raises(ValueError, match='source classifier round'):
        cycle.prepare_reward_transfer(191, 1920)
    assert not calls


def test_fresh_audit_is_not_installed_and_does_not_change_normal_clocks(tmp_path):
    cycle = object.__new__(ConditionalTauCycle)
    cycle.rank, cycle.world, cycle.device = 0, 1, torch.device('cpu')
    cycle.cfg = {'fit': {'batch_size': 3}}
    head = torch.nn.Linear(1, 1)
    cycle.reward = SimpleNamespace(head=head, round_id=21, last_refit_epoch=191, last_evaluation_epoch=189)
    cycle.validation = {}
    arrays = dict(split=np.array([0, 1, 2, 2, 2]), event_weight=np.ones(5),
                  condition=np.ones((5, 1)), candidate_truth=np.ones((5, 1)),
                  candidate_generated=np.zeros((5, 1)))
    model = torch.nn.Linear(1, 1)
    status = {'minimum_fit_steps_met': True, 'best_steps': 1000, 'total_steps': 1100}
    tmp_path.mkdir(exist_ok=True)
    cycle.fit = lambda *args, **kwargs: (model, {}, status, arrays)
    with patch('scripts.train_conditional_spin_ratio.score_pair', return_value=(np.ones(3), np.zeros(3))):
        report = cycle.transfer_audit({}, tmp_path, 1970)
    assert report['fit_total_steps'] == 1100 and report['test_events'] == 3
    assert cycle.reward.head is head and cycle.reward.round_id == 21
    assert cycle.reward.last_refit_epoch == 191 and cycle.reward.last_evaluation_epoch == 189
