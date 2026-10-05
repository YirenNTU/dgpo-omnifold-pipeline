"""CPU contracts for the ensemble experiment; no real data or compute jobs."""
from __future__ import annotations

import copy
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evenet_dgpo'))
from RL.DGPO_neutrino.tau_classifier_ensemble_probe import (
    MeanBoundedLogRatio, TauClassifierEnsemble, ensemble_disagreement,
    fit_tau_ensemble, score_saved_ensemble_panel,
)
from scripts.train_conditional_spin_ratio import build_classifier


def make_cycle(tmp_path):
    cfg = dict(condition_dim=36, candidate_dim=21, relative_dim=6, hidden=32,
        dropout=.05, head_kind='film', head_depth=3, condition_hidden=256,
        condition_width=256, ratio_bound=30, ratio_objective='bce',
        cross_attention=True, attention_heads=4,
        packing_spec={'shapes': {'x': [4, 4], 'x_mask': [4]}},
        relative_preprocessing={'mean': [0.]*6, 'scale': [1.]*6})
    reward = SimpleNamespace(installed_head=cfg, head=build_classifier(cfg),
        device=torch.device('cpu'), spec=None, round_id=20, denominator_step=1880,
        last_refit_epoch=187, last_evaluation_epoch=189,
        bundle=dict(source_run='raw1110', source_checkpoint='/pinned/raw1110.ckpt',
            normalization_file='/pinned/normalization.json', condition_mean=[0.]*36,
            condition_scale=[1.]*36, head=copy.deepcopy(cfg)))
    rng = np.random.default_rng(42)
    train = dict(source_ids=np.arange(6), condition=rng.normal(size=(6, 36)).astype('float32'),
        candidate_truth=rng.normal(size=(6, 21)).astype('float32'), event_weight=np.arange(1, 7.),
        split=np.array([0, 0, 0, 0, 1, 1], dtype='uint8'))
    validation = dict(source_ids=np.arange(10, 13), condition=rng.normal(size=(3, 36)).astype('float32'),
        candidate_truth=rng.normal(size=(3, 21)).astype('float32'), event_weight=np.array([1., 3., 2.]))
    cycle = SimpleNamespace(reward=reward, train=train, validation=validation,
        world=1, rank=0, device=torch.device('cpu'), output=tmp_path,
        cfg=dict(fit=dict(lr=2e-4, min_lr=1e-6, weight_decay=1e-4, batch_size=4,
                         epochs=2, patience=2, min_delta=0., min_steps=1000),
                 generation_batch_size=2))
    cycle.messages = []
    cycle.emit = lambda row, step: cycle.messages.append((row, step))
    cycle.actor, cycle.reference = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    generated = dict(features=rng.normal(size=(6, 21)).astype('float32'), deltas=np.zeros((6, 1, 2, 2)))
    return cycle, generated


def fake_dependencies(calls, *, total_steps=1001):
    cycle_module = ModuleType('RL.DGPO_neutrino.conditional_tau_cycle')
    reward_module = ModuleType('RL.DGPO_neutrino.conditional_tau_reward')
    reward_module.feature_precision = nullcontext
    cycle_module.barrier = lambda world: None
    cycle_module.panel_batch = lambda panel, take, spec, device: dict(
        x=torch.ones(len(take), 4, 4), event_weight=torch.as_tensor(panel['event_weight'][take]))
    def fit(cfg, arrays, path, device, rank, world, emit):
        calls.append((copy.deepcopy(cfg), {k: np.array(v, copy=True) for k, v in arrays.items()}))
        model = build_classifier(cfg).to(device).eval().requires_grad_(False)
        # Every member's best validation loss differs: member0 must still be
        # the control, regardless of the other members' validation ranking.
        saved = dict(cfg, state_dict=copy.deepcopy(model.state_dict()), epoch=2,
                     optimizer_steps=total_steps, val_bce=1/cfg['seed'])
        status = dict(total_steps=total_steps, epochs=2, best_epoch=2,
            best_steps=total_steps, best_val_bce=saved['val_bce'],
            minimum_fit_steps_met=total_steps >= cfg['min_steps'], early_stopped=False)
        torch.save(saved, path)
        emit(dict(epoch=2, optimizer_steps=total_steps, val_bce=saved['val_bce']))
        return model, saved, status
    cycle_module.fit_head = fit
    return patch.dict(sys.modules, {cycle_module.__name__: cycle_module,
                                   reward_module.__name__: reward_module})


def prepared_ensemble(tmp_path):
    cycle, generated = make_cycle(tmp_path)
    calls = []
    with fake_dependencies(calls):
        ensemble = fit_tau_ensemble(cycle, generated, tmp_path/'fit', seeds=(11, 22, 33, 44),
            judge_seed=55, source_checkpoint='/pinned/policy1920.ckpt', allow_cpu_fixture=True)
    return cycle, generated, ensemble, calls


def test_mean_uses_bounded_log_outputs_not_probability_or_best_member():
    class Constant(torch.nn.Module):
        def __init__(self, value):
            super().__init__(); self.value = value
        def forward(self, c, f):
            return c.new_full((len(c),), self.value)
    scores = [-6., -2., .2, 3.]
    head = MeanBoundedLogRatio([Constant(v) for v in scores])
    result = head(torch.ones(2, 1), torch.ones(2, 1))
    torch.testing.assert_close(result, torch.full((2,), np.mean(scores)))
    assert not torch.allclose(result, torch.full((2,), np.log(np.exp(scores).mean())))
    with pytest.raises(ValueError):
        MeanBoundedLogRatio([Constant(0.)])


def test_disagreement_is_condition_centered_weighted_and_records_undefined_direction():
    base = np.array([[-1., -2.], [0., 0.], [1., 2.]])
    scores = np.stack([base, base+.2, -base, -base+.3])
    result = ensemble_disagreement(scores, [3., 1.])
    assert result['pairwise']['0-1']['ranking_disagreement'] == 0
    assert result['pairwise']['0-2']['ranking_disagreement'] == 1
    assert result['pairwise']['0-2']['advantage_cosine'] == -1
    assert result['ensemble_advantage_rms'] < 1e-15
    # K/(K-1) leave-one-out scaling, retaining reward units.
    assert result['single_advantage_rms'] == pytest.approx(np.sqrt(.75*1.5+.25*6.))
    zero = ensemble_disagreement(np.zeros((4, 8, 2)), [1., 1.])
    assert zero['pairwise']['0-1']['advantage_cosine'] is None
    assert zero['pairwise']['0-1']['ranking_disagreement'] is None
    assert zero['ensemble_to_single_advantage_rms'] is None
    with pytest.raises(ValueError):
        ensemble_disagreement(scores, [1., -1.])
    with pytest.raises(ValueError):
        ensemble_disagreement(scores[:, :1], [1., 1.])
    invalid = scores.copy(); invalid[0, 0, 0] = 10
    with pytest.raises(ValueError):
        ensemble_disagreement(invalid, [1., 1.])


def test_fit_shares_one_negative_panel_split_and_preserves_policy_reference_reward_rng(tmp_path):
    cycle, generated = make_cycle(tmp_path)
    original_head, installed = cycle.reward.head, cycle.reward.installed_head
    actor, reference = copy.deepcopy(cycle.actor.state_dict()), copy.deepcopy(cycle.reference.state_dict())
    torch_state = torch.get_rng_state().clone()
    calls = []
    with fake_dependencies(calls):
        ensemble = fit_tau_ensemble(cycle, generated, tmp_path/'fit', seeds=(11, 22, 33, 44),
            judge_seed=55, source_checkpoint='/pinned/policy1920.ckpt', allow_cpu_fixture=True)
    assert len(calls) == 5 and [cfg['seed'] for cfg, _ in calls] == [11, 22, 33, 44, 55]
    for cfg, arrays in calls:
        np.testing.assert_array_equal(arrays['candidate_generated'], generated['features'])
        np.testing.assert_array_equal(arrays['split'], cycle.train['split'])
        assert cfg['cross_attention'] and cfg['ratio_bound'] == 30 and cfg['source_policy_step'] == 1920
    assert cycle.reward.head is original_head and cycle.reward.installed_head is installed
    assert cycle.reward.round_id == 20 and cycle.reward.denominator_step == 1880
    for key, value in actor.items():
        torch.testing.assert_close(cycle.actor.state_dict()[key], value)
    for key, value in reference.items():
        torch.testing.assert_close(cycle.reference.state_dict()[key], value)
    torch.testing.assert_close(torch.get_rng_state(), torch_state)
    assert ensemble.head_for('single') is ensemble.heads[0]
    assert ensemble.judge_head not in ensemble.heads
    assert (tmp_path/'fit/judge.pt').is_file()
    report = json.loads((tmp_path/'fit/report.json').read_text())
    assert report['fresh_denominator_step'] == 1920 and report['inherited_denominator_step'] == 1880
    assert not report['production_head_installed'] and not report['reference_recentered']


def test_artifact_reloads_identical_members_and_rejects_changed_source_or_underfit(tmp_path):
    cycle, _, ensemble, _ = prepared_ensemble(tmp_path)
    loaded = TauClassifierEnsemble.load(tmp_path/'fit', cycle.reward, 'cpu')
    c, f = torch.randn(2, 36), torch.randn(2, 21)
    for left, right in zip((*ensemble.heads, ensemble.judge_head), (*loaded.heads, loaded.judge_head)):
        torch.testing.assert_close(left(c, f), right(c, f), rtol=0, atol=0)
        assert not any(p.requires_grad for p in right.parameters())
    cycle.reward.bundle['normalization_file'] = '/wrong.json'
    with pytest.raises(ValueError, match='normalization'):
        TauClassifierEnsemble.load(tmp_path/'fit', cycle.reward, 'cpu')
    cycle.reward.bundle['normalization_file'] = '/pinned/normalization.json'
    payload = copy.deepcopy(ensemble.payload)
    payload['judge']['fit']['total_steps'] = 999
    with pytest.raises(ValueError, match='undertrained'):
        TauClassifierEnsemble(payload, cycle.reward, 'cpu')
    payload = copy.deepcopy(ensemble.payload); payload['complete'] = False
    with pytest.raises(ValueError, match='artifact'):
        TauClassifierEnsemble(payload, cycle.reward, 'cpu')
    payload = copy.deepcopy(ensemble.payload); payload['judge']['seed'] = 11
    with pytest.raises(ValueError, match='independent'):
        TauClassifierEnsemble(payload, cycle.reward, 'cpu')


def test_temporary_context_restores_head_metadata_clocks_on_exception(tmp_path):
    cycle, _, ensemble, _ = prepared_ensemble(tmp_path)
    old, installed = cycle.reward.head, cycle.reward.installed_head
    with pytest.raises(ValueError, match='fixture failure'):
        with ensemble.temporary_reward(cycle.reward, 'ensemble'):
            assert cycle.reward.head is ensemble.mean_head
            assert cycle.reward.denominator_step == 1880
            raise ValueError('fixture failure')
    assert cycle.reward.head is old and cycle.reward.installed_head is installed
    with ensemble.temporary_reward(cycle.reward, 'judge'):
        assert cycle.reward.head is ensemble.judge_head
    assert cycle.reward.head is old
    with pytest.raises(RuntimeError, match='changed reward clocks'):
        with ensemble.temporary_reward(cycle.reward):
            cycle.reward.round_id = 99
    assert cycle.reward.round_id == 20 and cycle.reward.head is old


def test_shared_feature_scoring_and_saved_k8_panel_keep_judge_outside_ensemble(tmp_path):
    cycle, _, ensemble, _ = prepared_ensemble(tmp_path)
    feature_calls = []
    def features(batch, delta):
        feature_calls.append(delta.clone())
        n = len(delta)
        c = np.zeros((n, 36), dtype='float32'); c[:, 16:20] = 1.
        f = np.zeros((n, 21), dtype='float32'); f[:, :4] = delta.cpu().numpy().reshape(n, 4)
        return c, f
    cycle.reward.features = features
    candidate_folder = tmp_path/'candidates'; candidate_folder.mkdir()
    delta = np.arange(3*8*2*2, dtype='float32').reshape(3, 8, 2, 2)/100
    np.savez(candidate_folder/'rank-00.npz', positions=np.arange(3), deltas=delta)
    generated = dict(features=np.zeros((3, 21), dtype='float32'))
    with fake_dependencies([]):
        result = ensemble.score_candidates(cycle.reward,
            dict(event_weight=torch.tensor([1., 0., 1.])), torch.from_numpy(delta).permute(1, 0, 2, 3),
            include_judge=True)
        assert result.shape == (5, 8, 3) and len(feature_calls) == 8
        assert torch.count_nonzero(result[:, :, 1]) == 0
        report = score_saved_ensemble_panel(cycle, ensemble, generated, candidate_folder, tmp_path/'scores')
    assert report['members'] == 4 and 'external_judge' in report
    with np.load(tmp_path/'scores/measurements.npz') as saved:
        assert saved['member_log_ratio'].shape == (4, 3, 8)
        assert saved['judge_log_ratio'].shape == (3, 8)
        np.testing.assert_allclose(saved['ensemble_log_ratio'], saved['member_log_ratio'].mean(0))
    # Eight feature calls per event minibatch, not five times that amount.
    assert len(feature_calls) == 24


def test_guardrails_fail_before_fitting(tmp_path):
    cycle, generated = make_cycle(tmp_path)
    calls = []
    with fake_dependencies(calls):
        with pytest.raises(ValueError, match='16-GPU'):
            fit_tau_ensemble(cycle, generated, tmp_path)
        with pytest.raises(ValueError, match='four unique'):
            fit_tau_ensemble(cycle, generated, tmp_path, seeds=(1, 1, 2, 3), allow_cpu_fixture=True)
        with pytest.raises(ValueError, match='independent'):
            fit_tau_ensemble(cycle, generated, tmp_path, seeds=(1, 2, 3, 4), judge_seed=1, allow_cpu_fixture=True)
        with pytest.raises(ValueError, match='architecture'):
            fit_tau_ensemble(cycle, generated, tmp_path, fit_overrides={'head_depth': 4}, allow_cpu_fixture=True)
        with pytest.raises(ValueError, match='1000'):
            fit_tau_ensemble(cycle, generated, tmp_path, fit_overrides={'min_steps': 10}, allow_cpu_fixture=True)
    assert not calls


def reward_with_real_serialization(fixture):
    # Import the production class without optional EveNet/Lightning imports,
    # exactly as the existing cycle tests do. Exercise its real serialization.
    path = Path(__file__).resolve().parents[1]/'evenet_dgpo/RL/DGPO_neutrino/conditional_tau_reward.py'
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == 'ConditionalTauReward')
    namespace = dict(BaseReward=object, copy=copy, torch=torch, np=np,
        KIND='conditional_tau_bound30', POLICY_CONDITIONING_CONTRACT='saved_tau_denominator_label0_v1')
    namespace['validate_head'] = lambda cfg: None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    reward = object.__new__(namespace['ConditionalTauReward'])
    reward.__dict__.update(copy.copy(fixture.__dict__))
    return reward


@pytest.mark.parametrize('selection', ['single', 'ensemble'])
def test_persistent_teacher_install_and_full_reward_resume_keep_reference_untouched(tmp_path, selection):
    cycle, _, ensemble, _ = prepared_ensemble(tmp_path)
    reward = reward_with_real_serialization(cycle.reward)
    reference = copy.deepcopy(cycle.reference.state_dict())
    legacy = reward.stack_payload()
    assert 'diagnostic_ensemble' not in legacy
    ensemble.install_into_reward(reward, selection, policy_step=1920, epoch=191)
    assert reward.round_id == 21 and reward.denominator_step == 1920 and reward.last_refit_epoch == 191
    assert reward.head is ensemble.head_for(selection)
    metadata = reward.checkpoint_metadata()
    assert metadata['diagnostic_teacher'] == selection
    assert metadata['diagnostic_reference_recentered'] is False
    saved = reward.stack_payload()
    assert saved['head']['diagnostic_ensemble_selection'] == selection
    assert saved['diagnostic_ensemble']['selection'] == selection
    checkpoint = tmp_path/'reward-stack.pt'; torch.save(saved, checkpoint)
    resumed = reward_with_real_serialization(cycle.reward)
    resumed.load_stack_payload(torch.load(checkpoint, map_location='cpu', weights_only=True))
    c, f = torch.randn(2, 36), torch.randn(2, 21)
    torch.testing.assert_close(reward.head(c, f), resumed.head(c, f), rtol=0, atol=0)
    assert resumed.round_id == 21 and resumed.denominator_step == 1920
    # Observer loads the common fitted judge after the teacher is installed.
    observer_ensemble = TauClassifierEnsemble.load(tmp_path/'fit', resumed, 'cpu')
    with observer_ensemble.temporary_reward(resumed, 'judge'):
        assert resumed.head is observer_ensemble.judge_head
        with pytest.raises(ValueError, match='temporary'):
            resumed.stack_payload()
    assert resumed.head is resumed._tau_diagnostic_ensemble_owner.head_for(selection)
    with pytest.raises(ValueError, match='only be installed once'):
        observer_ensemble.install_into_reward(resumed, selection)
    for key, value in reference.items():
        torch.testing.assert_close(cycle.reference.state_dict()[key], value)
    malformed = copy.deepcopy(saved); malformed['diagnostic_ensemble']['selection'] = 'judge'
    with pytest.raises(ValueError, match='training teacher'):
        resumed.load_stack_payload(malformed)
    assert resumed.head is resumed._tau_diagnostic_ensemble_owner.head_for(selection)
    # A normal saved single-head state still follows the legacy path and clears
    # the optional diagnostic metadata when explicitly restored.
    resumed.load_stack_payload(legacy)
    assert resumed.round_id == 20 and resumed.denominator_step == 1880
    assert not hasattr(resumed, '_tau_diagnostic_ensemble')
    assert 'diagnostic_ensemble' not in resumed.stack_payload()
