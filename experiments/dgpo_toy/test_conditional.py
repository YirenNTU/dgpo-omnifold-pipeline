"""Test instruments and isolation, never require the desired training failure."""
import copy
from dataclasses import replace
import math

import pytest
import torch

from experiments.dgpo_toy.conditional import (
    Config, Distribution, Classifier, Denoiser, alpha_sigma, baseline_health,
    build_dgpo_loss, decide, ddim, dgpo_objective, generator, head_group_gradients,
    initialize, paired_gain, policy_train, weight_health,
)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def small_config():
    return replace(Config(), dimensions=6, hidden=12, classifier_hidden=12,
                   batch=3, candidates=4, timesteps=2, eval_events=8,
                   policy_steps=2, eval_every=2)


def test_truth_preserves_lower_order_but_changes_third_order():
    cfg = Config()
    data = Distribution(cfg)
    rng = generator(17)
    c = data.contexts(80000, rng)
    y = data.sample(c, rng, truth=True)
    health = baseline_health(y, c, data)
    assert health["max_marginal_ks"] < .012
    assert health["max_variance_error"] < .025
    assert health["max_offdiag_covariance"] < .018
    k = torch.tensor(cfg.kappa)
    expected = cfg.structured_mass*float(torch.special.i1e(k)/torch.special.i0e(k))
    assert float(data.joint_signal(y, c).mean()) == pytest.approx(expected, abs=.01)
    base = data.sample(c, rng, truth=False)
    assert abs(float(data.joint_signal(base, c).mean())) < .01
    # Removing the true context destroys part of the context-dependent phase.
    assert float(data.joint_signal(y, c.roll(1, 0)).mean()) < expected-.1


def test_nominal_ratio_normalization_and_ess_quadrature():
    cfg = Config()
    data = Distribution(cfg)
    phase = (torch.arange(65536, dtype=torch.float64)+.5)*(2*math.pi/65536)
    k = torch.tensor(cfg.kappa, dtype=torch.float64)
    h = (k*phase.cos()-(torch.special.i0e(k).log()+k)).exp()
    assert float(h.mean()) == pytest.approx(1., abs=1e-12)
    m = cfg.structured_mass
    exact = 1/(1-m*m+m*m*float(h.square().mean())**(cfg.dimensions//3))
    assert data.nominal_ess() == pytest.approx(exact, rel=1e-12)
    assert exact < .01
    rng = generator(123)
    c = data.contexts(160000, rng)
    y = data.sample(c, rng, truth=False)
    ratio = data.nominal_log_ratio(y, c).double().exp()
    assert abs(float(ratio.mean())-1) < 5*float(ratio.std()/math.sqrt(len(ratio)))


def test_classifier_standardization_preserves_function_and_is_per_coordinate():
    cfg = small_config()
    data = Distribution(cfg)
    rng = generator(91)
    c = data.contexts(64, rng)
    y = data.sample(c, rng, truth=True)
    for fourier in [False, True]:
        model = initialize(Classifier, cfg, 17, fourier)
        before = model(y, c, data).detach()
        model.standardize(y, c, data)
        torch.testing.assert_close(model(y, c, data), before, rtol=1e-5, atol=1e-6)
        features = model.features(y, c, data)
        assert features.shape[-1] == cfg.context_dim + cfg.dimensions*(1+2*cfg.harmonics*fourier)
        # No joint sum channels. Changing coordinate 0 changes only its channels.
        altered = y.clone()
        altered[:, 0] += .2
        difference = (model.features(altered, c, data)-features).abs().sum(0)
        assert int((difference>0).sum()) == 1+2*cfg.harmonics*fourier


def test_neural_ddim_is_conditional_many_parameter_and_differentiable():
    cfg = small_config()
    model = initialize(Denoiser, cfg, 17)
    assert sum(p.numel() for p in model.parameters()) > 100
    noise = torch.randn(4, cfg.dimensions, generator=generator(19))
    c = torch.zeros(4, cfg.context_dim)
    y = ddim(model, c, noise, cfg.ddim_steps)
    assert not torch.allclose(y, ddim(model, c+1, noise, cfg.ddim_steps))
    gradients = torch.autograd.grad(y.square().mean(), tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(float(g.norm()) > 0 for g in gradients)
    # Independent reference implementation of stable-v DDIM update.
    x = noise
    for i in range(cfg.ddim_steps, 0, -1):
        t = x.new_tensor(i/cfg.ddim_steps)
        a, s = alpha_sigma(t)
        ap, sp = alpha_sigma(t-1/cfg.ddim_steps)
        x = (ap*a+sp*s)*x+(sp*a-ap*s)*model(x, t, c)
    torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)


def test_dgpo_detaches_rollout_reference_and_critic():
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
    loss, diagnostics = dgpo_objective(model, reference, critic, data, cfg, c, noise, t, eps)
    loss.backward()
    assert noise.grad is None
    assert all(p.grad is None for p in reference.parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert diagnostics["gate_mean"] == pytest.approx(.5)


def test_head_contributions_match_full_autograd():
    cfg = small_config()
    model = initialize(Denoiser, cfg, 29)
    rng = generator(123)
    m, k, b, d = cfg.timesteps, cfg.candidates, cfg.batch, cfg.dimensions
    x = torch.randn(m, k, b, d, generator=rng)
    t = .7*torch.rand(m, b, generator=rng)
    c = torch.randn(b, cfg.context_dim, generator=rng)
    output, hidden = model.predict_features(x, t[:, None], c[None, None])
    target = torch.randn(output.shape, generator=rng)
    advantage = torch.randn(k, b, generator=rng)
    current = (output-target).square().mean(-1)
    reference = torch.randn(m, k, b, generator=rng)
    gate = torch.sigmoid((advantage[None]*(current.detach()-reference)).mean(1))
    loss = torch.stack([build_dgpo_loss(current[i], reference[i], advantage, 1., k)[0]
                        for i in range(m)]).mean()
    grad_w, grad_b = torch.autograd.grad(loss, tuple(model.network[-1].parameters()))
    group = head_group_gradients(output-target, hidden, advantage, gate, alpha_sigma(t)[1])
    torch.testing.assert_close(group.mean(0), torch.cat([grad_w.flatten(), grad_b]), atol=1e-7, rtol=1e-5)


def test_context_cluster_uncertainty_and_weight_health():
    after = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
    result = paired_gain(after, torch.zeros_like(after))
    assert result["gain"] == 2.
    assert result["se"] == pytest.approx(1/math.sqrt(3))
    assert weight_health(torch.zeros(100))["ess_fraction"] == pytest.approx(1.)
    assert weight_health(torch.tensor([0., 0., 20.]))["ess_fraction"] < .34


def test_decision_does_not_force_failure():
    yes = {"passed": True}
    good = {"gain": .2, "lo95": .1}
    bad = {"gain": 0., "lo95": -.1}
    assert decide({"passed": False}, {}) == "inconclusive_setup"
    assert decide(yes, {"dgpo": good, "pathwise": good}) == "reward_improves"
    assert decide(yes, {"dgpo": bad, "pathwise": good}) == "dgpo_transfer_deficit"
    assert decide(yes, {"dgpo": bad, "pathwise": bad}) == "inconclusive_actionability"
    assert decide({}, {}, True) == "smoke_only"


def test_evaluation_rng_cannot_change_training_updates():
    cfg = small_config()
    data = Distribution(cfg)
    initial = initialize(Denoiser, cfg, 17)
    critic = initialize(Classifier, cfg, 29, True).requires_grad_(False)
    before = copy.deepcopy(initial.state_dict())
    for arm in ["dgpo", "pathwise"]:
        first, _ = policy_train(arm, initial, critic, data, cfg, 17, 60000, lambda _: None)
        second, _ = policy_train(arm, initial, critic, data,
                                 replace(cfg, eval_events=12, eval_every=1), 17, 70000, lambda _: None)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
