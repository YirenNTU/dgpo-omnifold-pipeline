"""Validity checks; these test the measuring instrument, not a desired failure."""
import ast
from contextlib import nullcontext
import typing

import pytest
import torch

from experiments.dgpo_toy.run import (
    ROOT, LOSS_SOURCE, Settings, alpha_sigma, build_dgpo_loss, compute_advantage,
    endpoint_variance, gaussian_kl, invert_endpoint_variance, log_ratio,
    production_objective, reward_moments, stepwise_ddim, train,
)


def test_exact_reward_kl_identity_and_gradient():
    theta = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
    q = endpoint_variance(theta, 20)
    ref = endpoint_variance(torch.zeros_like(theta), 20)
    p = ref * torch.tensor([1.8, .2], dtype=torch.float64)
    objective = -reward_moments(q, p, ref)[0] + gaussian_kl(q, ref)
    direct = gaussian_kl(q, p)
    torch.testing.assert_close(objective, direct, atol=1e-13, rtol=1e-13)
    first = torch.autograd.grad(objective, theta, retain_graph=True)[0]
    second = torch.autograd.grad(direct, theta)[0]
    torch.testing.assert_close(first, second, atol=1e-13, rtol=1e-13)


def test_target_representable_and_initial_marginals_identical():
    ref = endpoint_variance(torch.zeros(2, dtype=torch.float64), 20)
    p = ref * torch.tensor([1.8, .2], dtype=torch.float64)
    inverse = invert_endpoint_variance(p, 20)
    torch.testing.assert_close(endpoint_variance(inverse, 20), p, atol=1e-12, rtol=1e-12)
    # A 45-degree rotation converts eigenvalues into equal marginal variances.
    torch.testing.assert_close(p.mean(), ref.mean())
    assert float((p[0] - p[1]) / p.sum()) == pytest.approx(.8)


def test_collapsed_ddim_matches_full_chain_and_finite_difference():
    theta = torch.tensor([.4, -.9], dtype=torch.float64, requires_grad=True)
    noise = torch.tensor([[1.2, -.5], [-.3, .7]], dtype=torch.float64)
    collapsed = noise * endpoint_variance(theta, 20).sqrt()
    chain = stepwise_ddim(theta, noise, 20)
    torch.testing.assert_close(collapsed, chain, atol=1e-13, rtol=1e-13)
    jac = torch.autograd.functional.jacobian(lambda x: endpoint_variance(x, 20), theta)
    for i in range(2):
        delta = torch.zeros_like(theta)
        delta[i] = 1e-5
        fd = (endpoint_variance(theta + delta, 20) - endpoint_variance(theta - delta, 20)) / 2e-5
        torch.testing.assert_close(jac[:, i], fd, atol=1e-9, rtol=1e-8)


def test_against_actual_production_ddim_sampler():
    path = ROOT / "evenet_dgpo/evenet/utilities/diffusion_sampler.py"
    wanted = {"logsnr_schedule_cosine", "get_logsnr_alpha_sigma", "DDIMSampler"}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    scope = {"torch": torch, "Tensor": torch.Tensor, "Callable": typing.Callable,
             "Optional": typing.Optional, "nullcontext": nullcontext,
             "time_decorator": lambda **_: lambda fn: fn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    theta = torch.tensor([.4, -.9], dtype=torch.float64)
    noise = torch.tensor([[1.2, -.5], [-.3, .7]], dtype=torch.float64)
    for mode in ["legacy", "stable_v"]:
        sampler = scope["DDIMSampler"]("cpu", x0_mode=mode)
        sampler.prior_sde = lambda shape: noise.clone()
        def predictor(noise_x, time):
            from experiments.dgpo_toy.run import velocity_coefficient
            return velocity_coefficient(theta, time.double()) * noise_x
        result = sampler.sample(noise.shape, predictor, num_steps=20)
        # The production schedule forces float32; the toy evaluates it in float64.
        torch.testing.assert_close(result, stepwise_ddim(theta, noise, 20), atol=3e-6, rtol=3e-6)


def test_kernel_is_production_and_gate_gradient_matches_softplus():
    assert build_dgpo_loss.__code__.co_filename == str(LOSS_SOURCE)
    generator = torch.Generator().manual_seed(17)
    current = torch.randn(8, 10, dtype=torch.float64, generator=generator, requires_grad=True)
    reference = torch.randn(8, 10, dtype=torch.float64, generator=generator)
    rewards = torch.randn(8, 10, dtype=torch.float64, generator=generator)
    advantage, _ = compute_advantage(rewards, estimator="leave_one_out_unscaled")
    torch.testing.assert_close(advantage.mean(0), torch.zeros(10, dtype=torch.float64), atol=1e-15, rtol=0)
    main, _ = build_dgpo_loss(current, reference, advantage, 1., 8)
    exact_potential = torch.nn.functional.softplus((advantage * (current-reference)).mean(0)).mean()
    a = torch.autograd.grad(main, current, retain_graph=True)[0]
    b = torch.autograd.grad(exact_potential, current)[0]
    torch.testing.assert_close(a, b, atol=1e-15, rtol=1e-13)


def test_oracle_ratio_normalization_and_moments():
    # Population moments are checked by deterministic Gaussian quadrature.
    import numpy as np
    points, weights = np.polynomial.hermite.hermgauss(80)
    x, y = np.meshgrid(points * np.sqrt(2), points * np.sqrt(2))
    z = torch.tensor(np.stack([x.ravel(), y.ravel()], axis=-1))
    w = torch.tensor(np.outer(weights, weights).ravel() / np.pi)
    ref = torch.ones(2, dtype=torch.float64)
    p = torch.tensor([1.5, .5], dtype=torch.float64)
    score = log_ratio(z, p, ref)
    assert float((w * score.exp()).sum()) == pytest.approx(1., abs=1e-10)
    mean, std = reward_moments(ref, p, ref)
    torch.testing.assert_close((w * score).sum(), mean)
    torch.testing.assert_close((w * (score-mean).square()).sum().sqrt(), std)


def test_short_runs_finite_deterministic_and_use_independent_evaluation():
    torch.set_num_threads(1)
    cfg = Settings(steps=20, batch=8, timesteps=2, eval_samples=128, eval_every=10)
    a = train("dgpo_endpoint_kl", 17, cfg)
    b = train("dgpo_endpoint_kl", 17, cfg)
    assert a == b
    from dataclasses import replace
    c = train("dgpo_endpoint_kl", 17, replace(cfg, eval_samples=256, eval_every=7))
    for first, second in zip(a, c):
        assert first["theta_plus"] == second["theta_plus"]
        assert first["theta_minus"] == second["theta_minus"]
    assert a[1]["kl_q_truth"] < a[0]["kl_q_truth"]


def test_external_scale_does_not_change_initial_gate_or_oracle_control():
    from dataclasses import replace
    torch.set_num_threads(1)
    cfg = Settings(steps=2, batch=16, timesteps=2, eval_samples=64)
    scaled = replace(cfg, main_scale=7.)
    base = train("dgpo_endpoint_kl", 17, cfg)
    changed = train("dgpo_endpoint_kl", 17, scaled)
    assert base[1]["gate_mean"] == changed[1]["gate_mean"]
    assert changed[1]["main_gradient_norm"] == pytest.approx(7 * base[1]["main_gradient_norm"])
    assert train("exact_reward_kl", 17, cfg) == train("exact_reward_kl", 17, scaled)


def test_invalid_positive_control_is_inconclusive():
    from experiments.dgpo_toy.run import decide
    cfg = Settings(steps=100)
    histories = [[{"arm": arm, "seed": 17, "step": i, "kl_q_truth": 1.}
                  for i in range(101)] for arm in ["exact_reward_kl", "dgpo_endpoint_kl"]]
    assert decide(histories, cfg)["status"] == "inconclusive"
