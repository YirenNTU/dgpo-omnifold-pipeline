"""Fresh-data and numerical-density instruments; never force low ESS to pass."""
import copy
from dataclasses import replace
import math

import pytest
import torch

from experiments.dgpo_toy.conditional import (
    Config, Distribution, Denoiser, Classifier, initialize, generator, make_panel,
    ddim, alpha_sigma,
)
from experiments.dgpo_toy.run import endpoint_variance, velocity_coefficient
from experiments.dgpo_toy.low_ess_recovery import (
    fresh_batch, fit_streaming, inverse_ddim, neural_log_density,
    density_diagnostic, density_gates, confirmation_metrics,
    JointFourierClassifier, invertibility_certificate,
)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def cfg_small():
    return replace(Config(), dimensions=6, hidden=12, classifier_hidden=12,
                   fit_batch=16, classifier_steps=3, standardize_step=2)


class AnalyticGaussian(torch.nn.Module):
    def forward(self, x, t, c):
        theta = x.new_tensor([.1])
        return velocity_coefficient(theta, t).squeeze(-1)[..., None]*x


def test_numerical_density_matches_known_finite_ddim_density():
    model = AnalyticGaussian()
    rng = generator(17)
    c = torch.randn(12, 3, generator=rng, dtype=torch.float64)
    noise = torch.randn(12, 6, generator=rng, dtype=torch.float64)
    y = ddim(model, c, noise, 20)
    mean = torch.zeros_like(y)
    logq, diagnostic = neural_log_density(model, y, c, mean, 20, chunk=4)
    variance = endpoint_variance(torch.tensor([.1], dtype=torch.float64), 20)
    exact = -.5*(y.square()/variance+variance.log()+math.log(2*math.pi)).sum(-1)
    torch.testing.assert_close(logq, exact, atol=1e-9, rtol=1e-9)
    assert diagnostic["inverse_max_residual"] < 1e-10
    assert diagnostic["inverse_start_disagreement"] < 1e-10
    assert diagnostic["min_singular_value"] == pytest.approx(float(variance.sqrt()), abs=1e-10)
    assert diagnostic["global_injectivity_certified"] is False


def test_full_neural_inverse_roundtrip_and_jacobian_finite_difference():
    cfg = cfg_small()
    model = initialize(Denoiser, cfg, 17).double().requires_grad_(False)
    rng = generator(12)
    c = torch.randn(3, cfg.context_dim, generator=rng, dtype=torch.float64)
    noise = torch.randn(3, cfg.dimensions, generator=rng, dtype=torch.float64)
    y = ddim(model, c, noise, cfg.ddim_steps)
    inverse, error = inverse_ddim(model, y, c, torch.zeros_like(noise), cfg.ddim_steps)
    assert float(error.max()) < 1e-8
    torch.testing.assert_close(inverse, noise, atol=1e-8, rtol=1e-8)
    z, context = noise[0], c[0]
    jac = torch.func.jacrev(lambda x: ddim(model, context, x, cfg.ddim_steps))(z)
    columns = []
    for j in range(cfg.dimensions):
        delta = torch.zeros_like(z)
        delta[j] = 1e-5
        columns.append((ddim(model, context, z+delta, cfg.ddim_steps)
                        - ddim(model, context, z-delta, cfg.ddim_steps))/(2e-5))
    torch.testing.assert_close(jac, torch.stack(columns, -1), atol=1e-8, rtol=1e-7)


def test_fresh_batches_have_paired_contexts_and_actual_generator_negatives():
    cfg = cfg_small()
    data = Distribution(cfg)
    model = initialize(Denoiser, cfg, 17).requires_grad_(False)
    rng = generator(111)
    y, c, target = fresh_batch(model, data, cfg, rng)
    half = cfg.fit_batch//2
    torch.testing.assert_close(c[:half], c[half:])
    assert torch.equal(target, torch.cat([torch.ones(half), torch.zeros(half)]))
    replay = generator(111)
    cc = data.contexts(half, replay)
    truth = data.sample(cc, replay, truth=True)
    zz = torch.randn(half, cfg.dimensions, generator=replay)
    torch.testing.assert_close(y[:half], truth)
    torch.testing.assert_close(y[half:], ddim(model, cc, zz, cfg.ddim_steps))
    _, second_c, _ = fresh_batch(model, data, cfg, rng)
    assert not torch.equal(c, second_c)


def test_streaming_fit_does_not_update_source_and_is_deterministic():
    cfg = cfg_small()
    data = Distribution(cfg)
    initial = initialize(Denoiser, cfg, 17).requires_grad_(False)
    state = copy.deepcopy(initial.state_dict())
    train = make_panel(initial, data, cfg, 32, 10017)
    validation = make_panel(initial, data, cfg, 32, 20017)
    first, fit1 = fit_streaming(initial, data, cfg, train, validation, 17, lambda _: None)
    second, fit2 = fit_streaming(initial, data, cfg, train, validation, 17, lambda _: None)
    assert fit1 == fit2
    assert fit1["fresh_contexts_seen"] == cfg.classifier_steps*cfg.fit_batch//2
    assert fit1["selected_step"] == cfg.classifier_steps
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    assert all(torch.equal(value, state[key]) for key, value in initial.state_dict().items())
    assert all(p.grad is None for p in initial.parameters())


def test_confirmation_reports_tail_uncertainty_not_only_point_ess():
    cfg = cfg_small()
    data = Distribution(cfg)
    initial = initialize(Denoiser, cfg, 17).requires_grad_(False)
    critic = initialize(Classifier, cfg, 17, True).requires_grad_(False)
    panel = make_panel(initial, data, cfg, 64, 73017)
    metrics = confirmation_metrics(critic, panel, data)
    assert len(metrics["chunks"]) == 8
    expected = math.sqrt((1/metrics["ess_fraction"]-1)/63)
    assert metrics["mean_ratio_relative_se"] == pytest.approx(expected)


def test_density_validation_rejects_instrument_and_tail_errors():
    good = {"inverse_max_residual": 1e-12, "inverse_start_disagreement": 1e-12,
            "min_singular_value": .95, "logit_error_rms": .2}
    diagnostic = {"classes": {"truth": good.copy(), "generated": good.copy()}, "bce_excess": .01}
    assert all(density_gates(diagnostic).values())
    diagnostic["classes"]["truth"]["logit_error_rms"] = 2.
    diagnostic["classes"]["generated"]["inverse_max_residual"] = .1
    gates = density_gates(diagnostic)
    assert not gates["truth_logit_rms"]
    assert not gates["inverse_converged"]


def test_diagnostic_never_changes_critic_or_generator():
    cfg = cfg_small()
    data = Distribution(cfg)
    initial = initialize(Denoiser, cfg, 17).requires_grad_(False)
    critic = initialize(Classifier, cfg, 29, True).requires_grad_(False)
    a, b = copy.deepcopy(initial.state_dict()), copy.deepcopy(critic.state_dict())
    result = density_diagnostic(initial, critic, data, cfg, 4, 74017)
    assert result["density_kind"] == "numerical_certified_bijection"
    assert result["global_injectivity_certified"]
    assert all(torch.equal(value, a[key]) for key, value in initial.state_dict().items())
    assert all(torch.equal(value, b[key]) for key, value in critic.state_dict().items())


def test_generic_joint_dictionary_is_not_truth_selected_and_has_input_gradients():
    cfg = cfg_small()
    data = Distribution(cfg)
    critic = initialize(JointFourierClassifier, cfg, 17)
    assert len(critic.triples) == math.comb(cfg.dimensions, 3)
    assert len(critic.signs) == 4
    c = data.contexts(4, generator(17))
    y = torch.randn(4, cfg.dimensions, requires_grad=True)
    data.phase = lambda *_: (_ for _ in ()).throw(AssertionError("truth phase leaked"))
    data.joint_signal = data.phase
    features = critic.features(y, c, data)
    base = cfg.context_dim+cfg.dimensions*(1+2*cfg.harmonics)
    assert features.shape[-1] == base+8*math.comb(cfg.dimensions, 3)
    logits = critic(y, c, data)
    gradient = torch.autograd.grad(logits.sum(), y)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0
    before = logits.detach()
    critic.standardize(y.detach(), c, data)
    torch.testing.assert_close(before, critic(y, c, data), atol=1e-6, rtol=1e-5)


def test_global_certificate_is_sufficient_not_assumed_for_every_model():
    cfg = cfg_small()
    model = initialize(Denoiser, cfg, 17).double()
    bound = invertibility_certificate(model, 20)
    assert bound["certified"]
    assert bound["minimum_step_margin"] > 0
    with torch.no_grad():
        model.network[-1].weight.mul_(10000)
    assert not invertibility_certificate(model, 20)["certified"]
