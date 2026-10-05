"""Numerical full-chain/feature gradients and isolated step1920 launch guards."""
import copy
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

# Profiling decoration is unrelated to numerical sampling; avoid loading the
# optional Lightning/torchvision logging stack in this CPU unit test.
_debug = ModuleType("evenet.utilities.debug_tool")
_debug.time_decorator = lambda **kwargs: lambda function: function
sys.modules.setdefault("evenet.utilities.debug_tool", _debug)

from RL.DGPO_neutrino.conditional_tau_reward import ConditionalTauReward, candidate_coordinates
from RL.DGPO_neutrino.tau_pathwise_reward import candidate_coordinates_grad, candidate_hidden_grad, compute_reward_grad
from RL.DGPO_neutrino.tau_full_trajectory import FullTrajectoryController, replay_rng, slice_batch
from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
from evenet.network.body.normalizer import Normalizer
from evenet.utilities.diffusion_sampler import DDIMSampler
from scripts.test_dgpo_tau_ratio import example_bundle, live_batch
from scripts.train_conditional_spin_ratio import build_classifier
from scripts.test_tau_reward_transfer_launch import saved_state, source_runtime
from scripts import diagnose_tau_reward_mechanisms as launch


def normalizer(dtype=torch.float32):
    return Normalizer(torch.zeros(2, dtype=dtype), torch.ones(2, dtype=dtype), torch.ones(2, dtype=torch.bool))


class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.invisible_normalizer = normalizer()
        self.TruthGeneration = nn.Module()
        self.TruthGeneration.generator = nn.Linear(2, 2)
        self.hidden_scale = nn.Parameter(torch.tensor([.7, 1.3]))

    def invisible_coordinate_normalizer(self, batch):
        return self.invisible_normalizer

    def predict_diffusion_vector(self, noise_x, cond_x, time, mode, noise_mask):
        return self.TruthGeneration.generator(noise_x*self.hidden_scale)


@pytest.mark.parametrize('attention', [False, True])
def test_actual_reward_parity_and_frozen_trunk_input_gradient(attention):
    torch.manual_seed(21)
    bundle = example_bundle()
    if attention:
        bundle['head']['cross_attention'] = True
        bundle['head']['attention_heads'] = 4
        head = build_classifier(bundle['head'])
        nn.init.normal_(head.attention_output.weight, std=.1)
        bundle['head']['state_dict'] = head.state_dict()
    source = ConditionalTauReward(bundle, Trunk(), 'cpu', microbatch=2)
    batch = live_batch(5)
    # Branch crossings/pole reflections, not just small positive offsets.
    candidate = (torch.randn(2, 5, 2, 2)*2).requires_grad_()
    before = copy.deepcopy(source.backbone.state_dict())
    expected = source.compute(candidate.detach(), batch)
    actual = compute_reward_grad(source, candidate, batch)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-6)
    actual.sum().backward()
    assert torch.isfinite(candidate.grad).all() and candidate.grad.norm() > 0
    assert all(p.grad is None for p in source.head.parameters())
    assert all(p.grad is None for p in source.backbone.parameters())
    for key, value in before.items():
        torch.testing.assert_close(value, source.backbone.state_dict()[key], rtol=0, atol=0)
    # Directly require the frozen hidden branch to carry candidate derivatives.
    x = candidate[0].detach().requires_grad_()
    h = candidate_hidden_grad(source.backbone, batch, x)
    grad, = torch.autograd.grad(h.sum(), x)
    torch.testing.assert_close(grad, source.backbone.hidden_scale.expand_as(x))
    leaked = {**batch, 'truth_cij': torch.randn(5, 9), 'x_invisible': torch.randn(5, 2, 2),
              'classification': torch.ones(5, dtype=torch.long)}
    torch.testing.assert_close(compute_reward_grad(source, candidate.detach(), leaked), expected, rtol=3e-5, atol=2e-6)


def test_coordinates_match_numpy_with_periodicity_and_directional_derivative():
    torch.manual_seed(1)
    visible = torch.tensor([[10., 3., 1., 7.], [12., -4., 2., -8.]], dtype=torch.float64)
    delta = torch.tensor([[[.2, 3.2], [-3.5, -4.]], [[3.7, -2.5], [-.3, .1]]], dtype=torch.float64, requires_grad=True)
    stats = dict(mean=[.1]*6, scale=[.7]*6)
    f = candidate_coordinates_grad(visible, visible.flip(0), delta, stats)
    expected = candidate_coordinates(visible.numpy(), visible.flip(0).numpy(), delta.detach().numpy(), stats)
    np.testing.assert_allclose(f.detach().numpy(), expected, rtol=2e-6, atol=2e-7)
    weights = torch.randn_like(f)
    grad, = torch.autograd.grad((f*weights).sum(), delta)
    direction = torch.randn_like(delta)
    eps = 1e-3
    def value(x):
        return (candidate_coordinates_grad(visible, visible.flip(0), x, stats)*weights).sum()
    finite = (value(delta.detach()+eps*direction)-value(delta.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close(finite.double(), (grad*direction).sum(), rtol=2e-3, atol=2e-3)


class ChainPolicy(nn.Module):
    invisible_input_dim = 2
    def __init__(self, dropout=0.):
        super().__init__()
        self.gains = nn.Parameter(torch.linspace(.01, .05, 20, dtype=torch.float64))
        self.invisible_normalizer = normalizer(torch.float64)
        self.dropout = nn.Dropout(dropout)
    def predict_diffusion_vector(self, noise_x, cond_x, time, mode, noise_mask):
        i = (time*20).round().long().clamp(1, 20)-1
        return .15*noise_x + self.gains[i, None, None]*self.dropout(torch.ones_like(noise_x))


def batch(n=3):
    return dict(x=torch.ones(n, 2, 2, dtype=torch.float64), x_invisible=torch.zeros(n, 2, 2, dtype=torch.float64),
                x_invisible_mask=torch.ones(n, 2, dtype=torch.bool))


def score(_source, x, _batch):
    return -((x-.2)**2).mean(dim=(-2,-1))


def controller(method='pathwise'):
    return FullTrajectoryController(dict(method=method, event_microbatch=2, parity_atol=1e-9, parity_rtol=1e-9))


@pytest.mark.parametrize('mode', ['legacy', 'stable_v'])
def test_all_twenty_steps_and_checkpoint_gradient_match_finite_difference(mode):
    model = ChainPolicy()
    sampler = DDIMSampler(torch.device('cpu'), x0_mode=mode, dtype=torch.float64)
    b = batch(2)
    state = (torch.get_rng_state(), None)
    def loss(checkpoint):
        with replay_rng(state, torch.device('cpu')):
            x = generate_neutrino_candidates(model, b, sampler, K=1, num_ddim_steps=20,
                device=torch.device('cpu'), differentiable=True, checkpoint_steps=checkpoint)
            return -score(None, x, b).mean()
    grad, = torch.autograd.grad(loss(True), model.gains)
    direct, = torch.autograd.grad(loss(False), model.gains)
    torch.testing.assert_close(grad, direct, rtol=1e-9, atol=1e-9)
    assert torch.isfinite(grad).all() and (grad.abs() > 1e-8).all()
    initial = model.gains.detach().clone()
    direction = torch.linspace(-.8, .7, 20, dtype=torch.float64)
    eps = 1e-5
    with torch.no_grad():
        model.gains.copy_(initial+eps*direction)
    plus = loss(False).detach()
    with torch.no_grad():
        model.gains.copy_(initial-eps*direction)
    minus = loss(False).detach()
    with torch.no_grad():
        model.gains.copy_(initial)
    torch.testing.assert_close((plus-minus)/(2*eps), (grad*direction).sum(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('reference_coefficient', [1.0, 5.0])
def test_chunked_k8_replay_dropout_tail_gradient_and_reference_addition(reference_coefficient):
    model = ChainPolicy(.2).train()
    sampler = DDIMSampler(torch.device('cpu'), dtype=torch.float64, x0_mode='stable_v')
    b, c = batch(3), controller()
    c.reference_coefficient = reference_coefficient
    x = c.rollout(model, b, sampler, K=8, num_ddim_steps=20, device=torch.device('cpu'))
    # Independent uncheckpointed full objective on the exact stored MC panel.
    losses = []
    for k, start, stop, state in c.records:
        with replay_rng(state, torch.device('cpu')):
            sub = slice_batch(b, start, stop)
            generated = generate_neutrino_candidates(model, sub, sampler, K=1, num_ddim_steps=20,
                device=torch.device('cpu'), differentiable=True, checkpoint_steps=False)
            losses.append(-score(None, generated, sub).sum()/(8*3))
    expected_loss = torch.stack(losses).sum()
    expected, = torch.autograd.grad(expected_loss, model.gains)
    reference = torch.linspace(-.1, .2, 20, dtype=torch.float64)
    model.gains.grad = reference_coefficient*reference.clone()
    state_after = torch.get_rng_state().clone()
    with patch('RL.DGPO_neutrino.tau_full_trajectory.compute_reward_grad', side_effect=score):
        metrics = c.backward(model, b, sampler, None, x, score(None, x, b), device=torch.device('cpu'))
    torch.testing.assert_close(model.gains.grad, reference_coefficient*reference+expected, rtol=1e-9, atol=1e-9)
    assert metrics['trajectory/reward_loss'] == pytest.approx(float(expected_loss.detach()))
    assert metrics['trajectory/local_reference_gradient_norm'] == pytest.approx(float(reference.norm())*reference_coefficient)
    assert metrics['trajectory/local_unweighted_reference_gradient_norm'] == pytest.approx(float(reference.norm()))
    assert metrics['trajectory/local_reward_to_reference_gradient_ratio'] == pytest.approx(float(expected.norm()/reference.norm())/reference_coefficient)
    assert metrics['trajectory/local_total_on_reward_projection_ratio'] == pytest.approx(float(((expected+reference_coefficient*reference)*expected).sum()/expected.square().sum()))
    assert torch.equal(state_after, torch.get_rng_state())
    assert c.records == []


def test_native_and_pathwise_rollouts_are_identical_and_parity_fail_closed():
    model = ChainPolicy(.2).train()
    sampler = DDIMSampler(torch.device('cpu'), dtype=torch.float64, x0_mode='stable_v')
    outputs = []
    for method in ('native', 'pathwise'):
        torch.manual_seed(17)
        c = controller(method)
        outputs.append(c.rollout(model, batch(), sampler, K=8, num_ddim_steps=20, device=torch.device('cpu')))
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    with patch('RL.DGPO_neutrino.tau_full_trajectory.compute_reward_grad', side_effect=score):
        with pytest.raises(ValueError, match='reward differs'):
            c.backward(model, batch(), sampler, SimpleNamespace(compute=lambda x, b: score(None, x, b)), outputs[-1], score(None, outputs[-1], batch())+1, device=torch.device('cpu'))


def test_source1920_launch_preserves_objective_controls_and_uses_inherited_head():
    settings = launch.read_mapping(Path(__file__).resolve().parents[1]/'config/tau_full_trajectory_1920.yaml')
    original = source_runtime()
    metadata = launch.source_metadata(saved_state())
    for method in ('native', 'pathwise'):
        cfg = launch.configure(original, settings, metadata, pinned=Path('/test/source.ckpt'), output=Path('/test/out'),
            stage='trajectory', method=method)
        assert cfg['dgpo']['tau_ratio']['mechanism_probe']['full_trajectory']['method'] == method
        for key in ('reference_trust', 'lr_schedule', 'K', 'num_train_timesteps'):
            assert cfg['dgpo'][key] == original['dgpo'][key]
        assert cfg['dgpo']['checkpoint_load_mode'] == 'resume'
        assert cfg['experiment']['source_policy_step'] == 1920
        assert cfg['logger']['wandb']['fresh_run'] and method in cfg['logger']['wandb']['run_name']
        assert len(cfg['logger']['wandb']['run_name']) < 96
        assert cfg['platform']['number_of_workers'] == 16
        assert cfg['dgpo']['tau_ratio']['mechanism_probe']['relative_steps'] == [0, 1, 5, 10, 20, 35, 50]
    for kwargs in ({}, dict(stage='trajectory', ensemble_directory=Path('/test/ensemble')), dict(stage='trajectory', arm='ensemble')):
        with pytest.raises(ValueError):
            launch.configure(original, settings, metadata, pinned=Path('/test/source.ckpt'), output=Path('/test/out'), **kwargs)


def test_full_trajectory_ensemble_override_explains_correct_command_before_artifact_search():
    settings = launch.read_mapping(Path(__file__).resolve().parents[1]/'config/tau_full_trajectory_1920.yaml')
    assert launch.resolve_ensemble_directory(settings, None, arm='inherited', stage='trajectory') is None
    with patch.object(launch, 'validate_native_training_manifest') as scan:
        for override in ('unique', '/test/ensemble'):
            with pytest.raises(ValueError, match='remove --ensemble-directory'):
                launch.resolve_ensemble_directory(settings, override, arm='inherited', stage='trajectory')
        with pytest.raises(ValueError, match='inherited single head'):
            launch.resolve_ensemble_directory(settings, None, arm='ensemble', stage='trajectory')
        scan.assert_not_called()


def test_reference_scale_launch_changes_only_the_declared_coefficient():
    root = Path(__file__).resolve().parents[1]
    base = launch.read_mapping(root/'config/tau_full_trajectory_1920_mb64.yaml')
    trial = launch.read_mapping(root/'config/tau_full_trajectory_1920_mb64_ref5.yaml')
    stripped = copy.deepcopy(trial)
    assert stripped['full_trajectory'].pop('reference_coefficient') == 5.0
    assert stripped == base
    original = source_runtime()
    before = copy.deepcopy(original)
    cfg = launch.configure(original, trial, launch.source_metadata(saved_state()),
        pinned=Path('/test/source.ckpt'), output=Path('/test/out'), stage='trajectory', method='pathwise')
    assert original == before
    expected = copy.deepcopy(before['dgpo']['reference_trust'])
    expected['coefficient'] = 5.0
    assert cfg['dgpo']['reference_trust'] == expected
    assert cfg['dgpo']['tau_ratio']['mechanism_probe']['full_trajectory']['reference_coefficient'] == 5.0
    assert 'ref=5' in cfg['logger']['wandb']['run_name']
    assert cfg['dgpo']['lr_schedule'] == before['dgpo']['lr_schedule']
    with pytest.raises(ValueError, match='requires --method pathwise'):
        launch.configure(original, trial, launch.source_metadata(saved_state()),
            pinned=Path('/test/source.ckpt'), output=Path('/test/out'), stage='trajectory', method='native')


@pytest.mark.parametrize('value', [0, -1, True, float('nan'), float('inf'), '5'])
def test_reference_scale_rejects_invalid_coefficient(value):
    settings = launch.read_mapping(Path(__file__).resolve().parents[1]/'config/tau_full_trajectory_1920.yaml')
    settings['full_trajectory']['reference_coefficient'] = value
    with pytest.raises(ValueError, match='finite and positive'):
        launch.validate_settings(settings)
    with pytest.raises(ValueError, match='reference coefficient'):
        FullTrajectoryController(dict(settings['full_trajectory'], method='pathwise'))


def test_prepare_only_both_methods_write_distinct_manifests_without_starting_workers(tmp_path):
    import json
    import yaml
    settings = launch.read_mapping(Path(__file__).resolve().parents[1]/'config/tau_full_trajectory_1920.yaml')
    source = tmp_path/'source.ckpt'; source.touch()
    runtime = tmp_path/'source.yaml'; runtime.write_text(yaml.safe_dump(source_runtime()))
    shared = tmp_path/'shared-replay'
    settings.update(checkpoint=str(source), source_runtime=str(runtime), series_root=str(tmp_path/'new-series'),
                    native_training_directory=str(shared))
    config = tmp_path/'settings.yaml'; config.write_text(yaml.safe_dump(settings))
    filtered = ModuleType('evenet.dataset.filtered_data')
    filtered.validate_filtered_dataset = lambda path: {'rows':416701 if path == launch.TRAIN_DIRECTORY else 119002}
    with patch.dict(sys.modules, {'evenet.dataset.filtered_data':filtered}), \
         patch.object(torch, 'load', return_value=saved_state()), \
         patch.object(launch, 'validate_ensemble_metadata') as ensemble, \
         patch.object(launch.subprocess, 'run') as run:
        for method in ('native','pathwise'):
            launch.main([str(config),'prepare','--method',method])
        run.assert_not_called()
        ensemble.assert_not_called()
    assert not shared.exists()
    manifests = [json.loads(p.read_text()) for p in (tmp_path/'new-series').glob('*/invocation_manifest.json')]
    assert len(manifests)==2
    assert {m['method'] for m in manifests}=={'native','pathwise'}
    for m in manifests:
        assert m['prepared_only'] and m['relative_updates']==50
        assert m['method'] in m['wandb_name']
        cfg=launch.read_mapping(m['runtime'])
        assert cfg['dgpo']['tau_ratio']['mechanism_probe']['full_trajectory']['method']==m['method']
        command=launch.command_for_runtime(Path(m['runtime']),Path(m['runtime']).parent,stage='trajectory')
        assert command[command.index('--max-steps')+1]=='1970'


def test_cross_method_comparison_requires_matching_rollout_contract(tmp_path):
    records=[]
    for method in ('native','pathwise'):
        path=tmp_path/f'{method}.npz'
        np.savez(path,source_ids=np.arange(4),weight=np.ones(4),truth_cij=np.zeros((4,9)),
                 generated_cij=np.ones((4,9)),reward=np.ones((4,8))*(method=='pathwise'))
        records.append(dict(stage='trajectory',reward_arm='inherited',method=method,ensemble_directory=None,native_training_directory='/same-replay',
            endpoint_measurements=str(path),source_checkpoint='/source',validation_seed=42,candidates=8,
            native_replay_valid=True,native_replay_fingerprint='same',last_relative_step=50,output=str(path),
            trajectory_contract=dict(event_microbatch=16,bootstrap_seed=42)))
    settings=dict(bootstrap_replicates=10,bootstrap_seed=42)
    comparison=launch._paired_cross_arm(settings,records)
    assert len(comparison)==1 and comparison[0]['before_method']=='native'
    assert comparison[0]['after_method']=='pathwise'
    assert comparison[0]['reward_evaluator']=='same_inherited_training_head'
    assert 'inherited_head_reward' in comparison[0]
    records[-1]['trajectory_contract']=dict(event_microbatch=32,bootstrap_seed=42)
    assert launch._paired_cross_arm(settings,records)==[]
    records[0]['method'] = 'pathwise'
    records[-1]['trajectory_contract'] = dict(event_microbatch=16, bootstrap_seed=42, reference_coefficient=5.0)
    comparison = launch._paired_cross_arm(settings, records)
    assert len(comparison) == 1
    assert comparison[0]['intervention'] == 'reference_coefficient'
    assert comparison[0]['before_reference_coefficient'] == 1.0
    assert comparison[0]['after_reference_coefficient'] == 5.0
    records[0]['method'] = 'native'
    assert launch._paired_cross_arm(settings, records) == []


def test_matched_reward_layout_and_weighted_breakdown_with_partial_tail():
    calls = []
    class Aggregator:
        def compute(self, x, b):
            calls.append((tuple(x.shape), len(b['x'])))
            value = x.sum(dim=(-1, -2))
            return 2*value, {'conditional_tau':value}
    c = controller('pathwise')
    candidates = torch.randn(8, 5, 2, 2)
    b = {'x':torch.zeros(5, 2, 2)}
    total, components = c.score_rollout(Aggregator(), candidates, b)
    torch.testing.assert_close(total, 2*candidates.sum(dim=(-1,-2)))
    torch.testing.assert_close(components['conditional_tau'], total/2)
    assert all(shape[0] == 8 and rows <= c.microbatch for shape, rows in calls)
    assert sum(rows for _, rows in calls) == 5


def test_reward_failure_separates_endpoint_amplification_and_saves_evidence(tmp_path):
    c = controller('pathwise'); c.failure_directory = tmp_path
    target = torch.zeros(1, 2, 2, 2)
    replay = target + 1e-6
    def steep(x, b):
        return x.sum((-1,-2))*1000
    source = SimpleNamespace(compute=steep)
    with patch('RL.DGPO_neutrino.tau_full_trajectory.compute_reward_grad', side_effect=lambda s,x,b:steep(x,b)):
        error = c.reward_failure(source, target, replay, {'x':torch.zeros(2,2)},
            steep(target, None), steep(replay, None), k=0, start=0)
    assert 'fixed_endpoint_implementation_max_abs' in str(error)
    saved = torch.load(next(tmp_path.glob('*.pt')), weights_only=True)
    assert saved['details']['fixed_endpoint_implementation_max_abs'] == 0
    assert saved['details']['original_batch_layout_max_abs'] == 0
    assert saved['details']['replay_endpoint_amplification_max_abs'] > .001


def test_classifier_backward_runs_once_per_chunk():
    calls = []
    class CountBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.sum((-1,-2))
        @staticmethod
        def backward(ctx, grad):
            calls.append(1)
            return grad[..., None, None].expand(*grad.shape, 2, 2)
    model = ChainPolicy()
    c = controller()
    b = batch(3)
    sampler = DDIMSampler(torch.device('cpu'), dtype=torch.float64, x0_mode='stable_v')
    x = c.rollout(model,b,sampler,K=8,num_ddim_steps=20,device=torch.device('cpu'))
    chunks = len(c.records)
    with patch('RL.DGPO_neutrino.tau_full_trajectory.compute_reward_grad', side_effect=lambda s,x,b:CountBackward.apply(x)):
        c.backward(model,b,sampler,None,x,x.sum((-1,-2)),device=torch.device('cpu'))
    assert len(calls)==chunks
    assert model.gains.grad is not None and torch.isfinite(model.gains.grad).all()
    assert c.metrics['trajectory/time/pathwise_backward_and_diagnostics_seconds'] > 0


@pytest.mark.parametrize('attention', [False, True])
def test_prepared_reward_inputs_reuse_matches_original_k1_values_and_gradients(attention):
    bundle = example_bundle()
    if attention:
        bundle['head']['cross_attention'] = True
        bundle['head']['attention_heads'] = 4
        bundle['head']['state_dict'] = build_classifier(bundle['head']).state_dict()
    source = ConditionalTauReward(bundle, Trunk(), 'cpu', microbatch=2)
    b = live_batch(5)
    x = torch.randn(8,5,2,2,requires_grad=True)
    # Each independent K1 call prepares fresh observations, the old behavior.
    expected = torch.cat([source.compute(delta[None].detach(),b) for delta in x])
    with patch.object(source, 'prepare_inputs', wraps=source.prepare_inputs) as prepare:
        actual = source.compute(x.detach(),b)
        assert prepare.call_count == 1
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    prepared = source.prepare_inputs(b)
    cached = compute_reward_grad(source,x,b,prepared=prepared)
    fresh = compute_reward_grad(source,x,b)
    torch.testing.assert_close(cached,fresh,rtol=0,atol=0)
    g1, = torch.autograd.grad(cached.sum(),x)
    g2, = torch.autograd.grad(fresh.sum(),x)
    torch.testing.assert_close(g1,g2,rtol=0,atol=0)
    from RL.DGPO_neutrino.rewards import RewardAggregator
    agg = RewardAggregator(); agg.add(source,1)
    got,_ = controller().score_rollout(agg,x.detach(),b)
    torch.testing.assert_close(got,expected,rtol=3e-5,atol=2e-6)


def test_outer_checkpoint_scope_restores_flags_even_on_error():
    from RL.DGPO_neutrino.tau_full_trajectory import sampler_checkpoint_only
    model = ChainPolicy(); model.gradient_checkpointing = True
    with pytest.raises(RuntimeError):
        with sampler_checkpoint_only(model):
            assert model.gradient_checkpointing is False
            raise RuntimeError('fixture')
    assert model.gradient_checkpointing is True


def test_rng_isolation_only_seeds_declared_gpu_and_restores_cpu():
    from contextlib import nullcontext
    from RL.DGPO_neutrino.local_rng import seeded_torch_rng
    from unittest.mock import Mock
    initial = torch.get_rng_state().clone()
    with seeded_torch_rng(97, []):
        first = torch.rand(4)
    torch.testing.assert_close(torch.get_rng_state(), initial, rtol=0, atol=0)
    with seeded_torch_rng(97, []):
        torch.testing.assert_close(torch.rand(4),first,rtol=0,atol=0)
    generators = [Mock() for _ in range(4)]
    try:
        with patch('torch.random.fork_rng', return_value=nullcontext()) as fork, \
             patch('torch.cuda.default_generators',generators), \
             patch('torch.cuda.manual_seed_all') as all_devices:
            with seeded_torch_rng(31, [2]):
                pass
            fork.assert_called_once_with(devices=[2])
            all_devices.assert_not_called()
        for index,generator in enumerate(generators):
            if index==2: generator.manual_seed.assert_called_once_with(31)
            else: generator.manual_seed.assert_not_called()
    finally:
        torch.set_rng_state(initial)
