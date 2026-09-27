"""Read-only, large-panel gradient probe at saved narrow-toy anchors."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

try:
    from .narrow import Config, endpoint_variance, population_moments, sample_objective
except ImportError:
    from narrow import Config, endpoint_variance, population_moments, sample_objective


def probe(theta_value: float, tau: float, cfg: Config, seeds: list[int]) -> dict:
    ref = endpoint_variance(torch.zeros(1, dtype=torch.float64), cfg.ddim_steps).squeeze()
    theta = torch.tensor([theta_value], dtype=torch.float64, requires_grad=True)
    q = endpoint_variance(theta, cfg.ddim_steps).squeeze()
    reward = population_moments(q/ref, tau, cfg.mass, cfg.quadrature_order)[0]
    oracle = float(torch.autograd.grad(-reward, theta)[0].item())
    panels = {arm: [] for arm in ["dgpo", "score_mc"]}
    for seed in seeds:
        rng = torch.Generator().manual_seed(seed)
        normals = torch.randn(cfg.candidates, cfg.batch, dtype=torch.float64, generator=rng)
        t = cfg.t_max*torch.rand(cfg.timesteps, cfg.batch, dtype=torch.float64, generator=rng)
        eps = torch.randn(cfg.timesteps, cfg.batch, dtype=torch.float64, generator=rng)
        for arm in panels:
            th = torch.tensor([theta_value], dtype=torch.float64, requires_grad=True)
            loss, diag, _ = sample_objective(arm, th, ref, normals, t, eps, tau, cfg)
            gradient = float(torch.autograd.grad(loss, th)[0].item())
            panels[arm].append({"seed": seed, "gradient": gradient, **diag})
    summary = {}
    for arm, rows in panels.items():
        values = torch.tensor([r["gradient"] for r in rows], dtype=torch.float64)
        summary[arm] = {"gradient_mean": float(values.mean()),
                        "gradient_se": float(values.std(unbiased=True)/len(values)**.5),
                        "mean_over_population_gradient": float(values.mean())/oracle,
                        "sign_matches_population_count": sum(r["gradient"]*oracle>0 for r in rows)}
    return {"theta": theta_value, "population_negative_reward_gradient": oracle,
            "summary": summary, "panels": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8192)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    report = json.loads((args.directory/"report.json").read_text())
    cfg = replace(Config(**report["settings"]), batch=args.batch)
    with (args.directory/"history.csv").open() as stream:
        anchors = [r for r in csv.DictReader(stream) if r["width"]=="narrow"
                   and r["arm"]=="dgpo" and r["seed"]=="17" and int(r["step"]) in [0, 300, 600]]
    if {int(r["step"]) for r in anchors} != {0, 300, 600}:
        parser.error("Requires seed17 DGPO anchors at0/300/600")
    torch.set_num_threads(1)
    start = time.perf_counter()
    seeds = [101, 211, 307, 401]
    results = {r["step"]: probe(float(r["theta"]), cfg.tau_narrow, cfg, seeds) for r in anchors}
    output = {"protocol": "narrow-toy-post-screen-gradient-probe-v1",
              "source": str(args.directory), "batch": cfg.batch, "seeds": seeds,
              "optimizer_updates": 0, "anchors": results, "wall_seconds": time.perf_counter()-start,
              "scope": "Exploratory gradient measurement after screen; no change to training decision."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"anchors": {k:v["summary"] for k,v in results.items()},
                      "wall_seconds": output["wall_seconds"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
