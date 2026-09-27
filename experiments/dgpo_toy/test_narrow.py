"""Validate narrow-ridge densities, integration, and production gradient fidelity."""
from dataclasses import replace
import math

import pytest
import torch

from experiments.dgpo_toy.narrow import (
    Config, concentration, decide, fixed_reward, observe, population_moments,
    population_weight_ess, quadrature, sample_objective, train,
)
from experiments.dgpo_toy.run import endpoint_variance, invert_endpoint_variance


@pytest.mark.parametrize("tau", [.25, .0002])
def test_normalized_ratio_and_analytic_ess(tau):
    mass = .8
    one = torch.tensor(1., dtype=torch.float64)
    ess, mean = population_weight_ess(one, tau, mass)
    assert float(mean) == pytest.approx(1., abs=1e-14)
    expected = 1/(1-mass**2+mass**2/(tau*math.sqrt(2-tau**2)))
    assert float(ess) == pytest.approx(expected, rel=1e-14)
    u, weights = quadrature(512)
    z = tau*u
    measure = weights*tau*torch.exp(-z.square()/2)/math.sqrt(2*math.pi)
    ratios = fixed_reward(z, one, tau, mass).exp()
    # Analytically integrate the constant background, numerically its excess.
    first = 1-mass+(measure*(ratios-(1-mass))).sum()
    second = (1-mass)**2+(measure*(ratios.square()-(1-mass)**2)).sum()
    assert float(first) == pytest.approx(1., abs=1e-11)
    assert float(first.square()/second) == pytest.approx(expected, rel=1e-10)


def test_observation_marginals_uniform_for_every_residual():
    context = ((torch.arange(4096, dtype=torch.float64)+.5)/4096)[:, None].expand(-1, 4)
    residual = torch.tensor([-3., -.0001, 0., .9], dtype=torch.float64)[None, :].expand(4096, -1)
    x, y = observe(context, residual, torch.tensor(.8838, dtype=torch.float64))
    for value in [x, y]:
        counts = torch.histc(value.flatten(), bins=64, min=0, max=1)
        torch.testing.assert_close(counts, torch.full_like(counts, value.numel()/64))


@pytest.mark.parametrize("tau", [.25, .0002])
def test_population_integral_gradient_and_resolution(tau):
    ref = endpoint_variance(torch.zeros(1, dtype=torch.float64), 20).squeeze()
    for value in [0., -2., -7., -11.]:
        theta = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        def objective(th, order=256):
            return population_moments(endpoint_variance(th, 20).squeeze()/ref, tau, .8, order)[0]
        reward = objective(theta)
        grad = torch.autograd.grad(reward, theta)[0]
        fd = (objective(theta.detach()+1e-5)-objective(theta.detach()-1e-5))/2e-5
        torch.testing.assert_close(grad.squeeze(), fd, rtol=2e-5, atol=2e-8)
        torch.testing.assert_close(reward, objective(theta, 512), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("arm", ["dgpo", "score_mc"])
def test_event_gradient_decomposition_equals_autograd(arm):
    cfg = Config(batch=64, timesteps=4)
    rng = torch.Generator().manual_seed(113)
    ref = endpoint_variance(torch.zeros(1, dtype=torch.float64), 20).squeeze()
    for tau in [.25, .0002]:
        for value in [0., -2., -8.]:
            theta = torch.tensor([value], dtype=torch.float64, requires_grad=True)
            normals = torch.randn(8, 64, dtype=torch.float64, generator=rng)
            t = .7*torch.rand(4, 64, dtype=torch.float64, generator=rng)
            eps = torch.randn(4, 64, dtype=torch.float64, generator=rng)
            loss, _, groups = sample_objective(arm, theta, ref, normals, t, eps, tau, cfg)
            actual = torch.autograd.grad(loss, theta)[0].squeeze()
            torch.testing.assert_close(groups.mean(), actual, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("tau", [.25, .0002])
def test_population_score_identity(tau):
    variance = torch.tensor(.01, dtype=torch.float64, requires_grad=True)
    reward = population_moments(variance, tau, .8)[0]
    derivative = torch.autograd.grad(reward, variance)[0]
    nodes, weights = quadrature(512)
    scale = min(math.sqrt(float(variance.detach())), tau)
    z = scale*nodes
    density_measure = weights*scale*torch.exp(-z.square()/(2*variance.detach()))/(2*math.pi*variance.detach()).sqrt()
    extra = fixed_reward(z, torch.tensor(1.), tau, .8)-math.log(.2)
    # E[score]=0, so subtracting the constant reward baseline is exact.
    score = .5*(z.square()/variance.detach()-1)/variance.detach()
    expected = (density_measure*extra*score).sum()
    torch.testing.assert_close(derivative, expected, rtol=1e-9, atol=1e-9)


def test_zero_gradient_concentration_is_undefined():
    values = concentration(torch.zeros(8), "gradient")
    assert values["gradient_ess_fraction"] is None
    assert values["gradient_total"] == 0
    assert concentration(torch.ones(8), "gradient")["gradient_ess_fraction"] == 1


def test_capability_witness_respects_finite_ddim_noise_floor():
    ref = endpoint_variance(torch.zeros(1, dtype=torch.float64), 20).squeeze()
    for tau in [.25, .0002]:
        desired = (ref*(.5*tau)**2).reshape(1)
        theta = invert_endpoint_variance(desired, 20)
        actual = endpoint_variance(theta, 20)
        torch.testing.assert_close(actual, desired, rtol=1e-8, atol=1e-16)
        initial = population_moments(torch.ones_like(ref), tau, .8)[0]
        witness = population_moments(actual.squeeze()/ref, tau, .8)[0]
        assert float((witness-initial)/(math.log(.2+.8/tau)-initial)) > .9


def test_training_deterministic_and_eval_stream_separate():
    torch.set_num_threads(1)
    cfg = Config(steps=5, batch=16, timesteps=2, eval_samples=128, eval_every=2)
    a = train("dgpo", "narrow", .0002, 17, cfg)
    assert a == train("dgpo", "narrow", .0002, 17, cfg)
    b = train("dgpo", "narrow", .0002, 17, replace(cfg, eval_samples=256, eval_every=3))
    assert [r["theta"] for r in a] == [r["theta"] for r in b]


def test_decision_does_not_force_failure_or_accept_invalid_control():
    histories = []
    for width, ess in [("broad", .4), ("narrow", .0004)]:
        for arm in ["dgpo", "score_mc", "population"]:
            histories.append([{"width": width, "arm": arm, "seed": 17,
                               "fixed_weight_population_ess_fraction": ess,
                               "reward_gain_fraction": gain} for gain in [0., .9]])
    assert decide(histories, Config())["status"] == "not_reproduced"
    for h in histories:
        if h[0]["arm"] == "population":
            h[-1]["reward_gain_fraction"] = .01
    assert decide(histories, Config())["status"] == "inconclusive_control_budget"
