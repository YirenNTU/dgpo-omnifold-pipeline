"""Measure the toy vector field without taking optimizer steps."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

try:
    from .run import (Settings, endpoint_variance, gaussian_kl, invert_endpoint_variance,
                      production_objective, reward_moments, source_fingerprints)
except ImportError:
    from run import (Settings, endpoint_variance, gaussian_kl, invert_endpoint_variance,
                     production_objective, reward_moments, source_fingerprints)


def field(theta_value, cfg, seeds):
    ref_theta = torch.zeros(2, dtype=torch.float64)
    ref = endpoint_variance(ref_theta, cfg.ddim_steps)
    p = ref.mean() * torch.tensor([1 + cfg.rho, 1 - cfg.rho], dtype=torch.float64)
    gradients, gate_means = [], []
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed)
        theta = theta_value.detach().clone().requires_grad_(True)
        noise = torch.randn(cfg.candidates, cfg.batch, 2, generator=rng, dtype=torch.float64)
        t = cfg.t_max * torch.rand(cfg.timesteps, cfg.batch, generator=rng, dtype=torch.float64)
        eps = torch.randn(cfg.timesteps, cfg.batch, 2, generator=rng, dtype=torch.float64)
        main, diag = production_objective(theta, ref_theta, p, ref, noise, t, eps, cfg)
        gradients.append(torch.autograd.grad(main, theta)[0])
        gate_means.append(float(diag["gate_mean"]))
    theta = theta_value.detach().clone().requires_grad_(True)
    q = endpoint_variance(theta, cfg.ddim_steps)
    exact_reward = -reward_moments(q, p, ref)[0]
    g_reward = torch.autograd.grad(exact_reward, theta, retain_graph=True)[0]
    g_kl = torch.autograd.grad(gaussian_kl(q, ref), theta)[0]
    stacked = torch.stack(gradients)
    g_main = stacked.mean(0)
    se = stacked.std(dim=0, unbiased=True) / len(seeds)**.5
    least_squares_scale = float(torch.dot(g_main, -g_kl) / g_main.square().sum())
    best_positive = max(0., least_squares_scale)
    # At a fixed anchor the KL gradient is analytic/deterministic. Its
    # perpendicular component cannot be canceled by changing its coefficient.
    if g_kl.norm() > 1e-12:
        perpendicular = torch.stack([-g_kl[1], g_kl[0]]) / g_kl.norm()
        orthogonal_samples = stacked @ perpendicular
    else:
        orthogonal_samples = torch.zeros(len(seeds), dtype=torch.float64)
    return {"theta": theta_value.tolist(), "q_endpoint": q.detach().tolist(),
            "main_gradient_panels": stacked.tolist(), "main_gradient_mean": g_main.tolist(),
            "main_gradient_se": se.tolist(), "exact_negative_reward_gradient": g_reward.tolist(),
            "exact_kl_gradient": g_kl.tolist(), "exact_composite_gradient": (g_reward+g_kl).tolist(),
            "dgpo_composite_gradient_scale1": (g_main+g_kl).tolist(),
            "best_positive_scale_at_anchor_diagnostic_only": best_positive,
            "best_scalar_composite_residual_norm": float((best_positive*g_main+g_kl).norm()),
            "main_perpendicular_to_kl_panels": orthogonal_samples.tolist(),
            "main_perpendicular_to_kl_mean": float(orthogonal_samples.mean()),
            "main_perpendicular_to_kl_se": float(orthogonal_samples.std(unbiased=True)/len(seeds)**.5),
            "gate_mean": sum(gate_means)/len(gate_means)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--baseline-report", type=Path)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    torch.set_num_threads(1)
    cfg = Settings(batch=args.batch)
    seeds = [101, 211, 307, 401]
    started = time.perf_counter()
    ref = endpoint_variance(torch.zeros(2, dtype=torch.float64), cfg.ddim_steps)
    p = ref.mean()*torch.tensor([1+cfg.rho, 1-cfg.rho], dtype=torch.float64)
    anchors = {"initial": torch.zeros(2, dtype=torch.float64),
               "truth": invert_endpoint_variance(p, cfg.ddim_steps)}
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text())
        for k in ["rho", "ddim_steps", "t_max", "candidates", "timesteps"]:
            if baseline["settings"][k] != getattr(cfg, k):
                parser.error(f"Baseline settings differ for {k}")
        for e in baseline["endpoints"]:
            if e["arm"] == "dgpo_endpoint_kl":
                anchors[f"baseline_endpoint_seed{e['seed']}"] = torch.tensor([e["theta_plus"],e["theta_minus"]], dtype=torch.float64)
    fields = {name: field(theta, cfg, seeds) for name, theta in anchors.items()}
    initial = fields["initial"]
    gm = torch.tensor(initial["main_gradient_mean"], dtype=torch.float64)
    gr = torch.tensor(initial["exact_negative_reward_gradient"], dtype=torch.float64)
    scale = float(torch.dot(gm, gr) / gm.square().sum())
    report = {"protocol": "dgpo-toy-gradient-units-v1", "settings": asdict(cfg),
              "source_sha256": source_fingerprints(),
              "gradient_panel_seeds": seeds, "fields": fields,
              "initial_calibrated_main_scale": scale, "wall_seconds": time.perf_counter()-started,
              "scope": "Scale fitted only at initial policy; truth anchor is diagnostic, not tuning."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
