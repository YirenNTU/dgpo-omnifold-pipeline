"""Legacy penalty semantics, matched comparisons, and evaluation-only moments."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from experiments.dgpo_toy.conditional import (
    Classifier, Denoiser, Distribution, component_gradient_trace, dgpo_objective,
    generator, initialize, policy_train)
from experiments.dgpo_toy.run import build_reference_trust_loss
from experiments.dgpo_toy.fixed_reward import run as fixed_run, load_source
from experiments.dgpo_toy.velocity_kl import run
from experiments.dgpo_toy.structure_metrics import (
    structure_features, structure_target, paired_structure_change)
from experiments.dgpo_toy.test_conditional import small_config
from experiments.dgpo_toy.test_fixed_reward import saved, assert_nested_equal


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_production_half_mean_penalty_and_reference_stop_gradient():
    v = torch.randn(2, 4, 3, 6, requires_grad=True)
    ref = torch.randn_like(v, requires_grad=True)
    loss, diag = build_reference_trust_loss(v, ref, torch.ones_like(v))
    torch.testing.assert_close(loss, .5*(v-ref).square().mean())
    loss.backward()
    torch.testing.assert_close(v.grad, (v-ref.detach())/v.numel())
    assert ref.grad is None
    assert float(diag["reference_trust/velocity_mse"]) == pytest.approx(float(2*loss.detach()))


def objective_inputs():
    cfg = small_config()
    data = Distribution(cfg)
    model = initialize(Denoiser, cfg, 17)
    reference = copy.deepcopy(model)
    critic = initialize(Classifier, cfg, 29, True)
    rng = generator(91)
    c = data.contexts(cfg.batch, rng)
    noise = torch.randn(cfg.batch, cfg.candidates, cfg.dimensions, generator=rng, requires_grad=True)
    t = .7*torch.rand(cfg.timesteps, cfg.batch, generator=rng)
    eps = torch.randn(cfg.timesteps, cfg.batch, cfg.dimensions, generator=rng)
    return model, reference, critic, data, cfg, c, noise, t, eps


def test_zero_coefficient_and_initial_zero_penalty_exact():
    args = objective_inputs()
    default, diag = dgpo_objective(*args)
    off, offdiag = dgpo_objective(*args, velocity_coefficient=0.)
    on, ondiag = dgpo_objective(*args, velocity_coefficient=1., trace_gradients=True)
    assert torch.equal(default, off) and torch.equal(default, on)
    assert diag == offdiag
    assert ondiag["velocity_penalty"] == 0.
    assert ondiag["velocity_gradient_norm"] == 0.
    gradients = [torch.autograd.grad(loss, tuple(args[0].parameters())) for loss in (default, off, on)]
    for candidate in gradients[1:]:
        for a, b in zip(gradients[0], candidate):
            assert torch.equal(a, b)


def test_nonzero_penalty_detaches_rollout_reference_critic():
    args = objective_inputs()
    with torch.no_grad():
        next(args[0].parameters()).add_(.1)
    loss, diag = dgpo_objective(*args, velocity_coefficient=1., trace_gradients=True)
    assert diag["velocity_penalty"] > 0 and diag["velocity_gradient_norm"] > 0
    loss.backward()
    assert args[6].grad is None
    assert all(p.grad is None for model in args[1:3] for p in model.parameters())


def test_component_gradient_geometry_and_no_grad_mutation():
    p = torch.tensor([2., 3.], requires_grad=True)
    result = component_gradient_trace(2*p[0], -p[0]+p[1], (p,))
    assert p.grad is None
    assert result["main_gradient_norm"] == 2.
    assert result["velocity_gradient_norm"] == pytest.approx(2**.5)
    assert result["main_velocity_gradient_cosine"] == pytest.approx(-1/2**.5)
    assert result["total_on_main_projection"] == .5


def test_diagnostics_do_not_change_updates_and_first_step_matches_off():
    model, _, critic, data, cfg, *_ = objective_inputs()
    snapshots = []
    for coefficient, trace in ((0., True), (1., False), (1., True)):
        states = []
        def save(step, model, optimizer, rng, history):
            states.append(copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                         "rng": rng.get_state()}))
        policy_train("dgpo", model, critic, data, cfg, 17, 90017, lambda row: None, save,
                     velocity_coefficient=coefficient, trace_gradients=trace)
        snapshots.append(states)
    assert_nested_equal(snapshots[0][0], snapshots[1][0])
    assert_nested_equal(snapshots[1], snapshots[2])


def test_resume_rejects_silent_coefficient_change():
    model, _, critic, data, cfg, *_ = objective_inputs()
    with pytest.raises(ValueError, match="preserve velocity"):
        policy_train("dgpo", model, critic, data, cfg, 17, 90017, lambda _: None,
                     resume_state={"velocity_coefficient": 1.}, velocity_coefficient=0.)


def test_truth_higher_order_moments_match_instrument():
    cfg = small_config()
    data = Distribution(cfg)
    rng = generator(719)
    c = data.contexts(60000, rng)
    truth = data.sample(c, rng, truth=True)
    generated = data.sample(c, rng, truth=False)
    features = structure_features(truth, c, data).double().numpy()
    target = structure_target(data)
    assert features.shape[-1] == 20  # 2 triples * 4 harmonics * sin/cos + 4 cross moments
    assert np.max(np.abs(features.mean(0)-target)) < .013
    baseline = structure_features(generated, c, data).double().numpy()
    change = paired_structure_change(features[:3000], baseline[:3000], target, 987, replicates=100)
    assert change["hi95"] < 0
    identical = paired_structure_change(features[:50], features[:50], target, 987, replicates=10)
    assert identical["mse_change"] == identical["lo95"] == identical["hi95"] == 0.


@pytest.mark.parametrize("coefficient", [1., .1])
def test_runner_preserves_source_matches_first_step_and_saves_state(saved, tmp_path, coefficient):
    before = saved.read_bytes()
    control, output = tmp_path/"control", tmp_path/"velocity"
    fixed_run(saved, control)
    result = run(saved, control, output, policy_steps=2, coefficient=coefficient)
    assert result["first_update_matches_no_kl"] and result["frozen_source_unchanged"]
    assert saved.read_bytes() == before
    state = torch.load(output/"dgpo_state.pt", weights_only=True)
    assert state["step"] == 2 and state["velocity_coefficient"] == coefficient
    assert all(float(s["step"]) == 2 for s in state["optimizer"]["state"].values())
    assert result["endpoints"]["velocity_kl"]["structure"]["fourier_moment_rmse"] > 0
    for arm in ("no_kl", "velocity_kl"):
        assert len(result["policy_histories"][arm]) == 2
    with pytest.raises(ValueError, match="matched"):
        run(saved, control, tmp_path/"wrong", policy_steps=3)
    with pytest.raises(ValueError, match="separate"):
        run(saved, control, control, policy_steps=2)
    with pytest.raises(ValueError, match="stream"):
        run(saved, control, tmp_path/"overlap", policy_steps=2, endpoint_seed=90017)
