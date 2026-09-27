"""Two-parameter DDIM toy; exact ratios, production DGPO kernel, CPU only."""
from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
LOSS_SOURCE = ROOT / "evenet_dgpo/RL/DGPO_neutrino/dgpo_utils.py"


def source_fingerprints() -> dict[str, str]:
    paths = [LOSS_SOURCE, Path(__file__),
             ROOT / "evenet_dgpo/evenet/utilities/diffusion_sampler.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def load_production_kernel():
    """Execute the original pure functions, with no Ray/Lightning imports."""
    names = {"compute_per_event_advantage", "build_dgpo_loss"}
    constants = {"ADVANTAGE_ESTIMATOR_ZSCORE", "ADVANTAGE_ESTIMATOR_LOO_UNSCALED",
                 "VALID_ADVANTAGE_ESTIMATORS"}
    tree = ast.parse(LOSS_SOURCE.read_text(), filename=str(LOSS_SOURCE))
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in names
    ) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                        for t in n.targets)
    )]
    scope = {"torch": torch, "Tensor": Tensor}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LOSS_SOURCE), "exec"), scope)
    return scope["compute_per_event_advantage"], scope["build_dgpo_loss"]


compute_advantage, build_dgpo_loss = load_production_kernel()


def load_reference_trust_kernel():
    """Use the production velocity-MSE definition without training dependencies."""
    constants = {"REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE",
                 "REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL", "VALID_REFERENCE_TRUST_OBJECTIVES"}
    tree = ast.parse(LOSS_SOURCE.read_text(), filename=str(LOSS_SOURCE))
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name == "build_reference_trust_loss"
    ) or (
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                        for t in n.targets)
    )]
    scope = {"torch": torch, "Tensor": Tensor}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LOSS_SOURCE), "exec"), scope)
    return scope["build_reference_trust_loss"]


build_reference_trust_loss = load_reference_trust_kernel()


@dataclass(frozen=True)
class Settings:
    steps: int = 600
    batch: int = 256
    candidates: int = 8
    timesteps: int = 8
    ddim_steps: int = 20
    rho: float = 0.8
    lr: float = 0.01
    weight_decay: float = 0.001
    coefficient: float = 1.0
    main_scale: float = 1.0
    t_max: float = 0.7
    eval_every: int = 25
    eval_samples: int = 16384


def alpha_sigma(t: Tensor) -> tuple[Tensor, Tensor]:
    # Float64 version of the production finite-logSNR cosine VP schedule.
    start = math.atan(math.exp(-10.0))
    end = math.atan(math.exp(10.0))
    angle = start + (end - start) * t
    return angle.cos(), angle.sin()


def velocity_coefficient(log_variance: Tensor, t: Tensor) -> Tensor:
    variance = log_variance.exp()
    a, s = alpha_sigma(t.unsqueeze(-1))
    return a * s * (1 - variance) / (a.square() * variance + s.square())


def endpoint_variance(log_variance: Tensor, steps: int) -> Tensor:
    t = torch.arange(steps, 0, -1, dtype=log_variance.dtype) / steps
    a, s = alpha_sigma(t.unsqueeze(-1))
    ap, sp = alpha_sigma((t - 1 / steps).unsqueeze(-1))
    m = velocity_coefficient(log_variance, t)
    factors = ap * a + sp * s + (sp * a - ap * s) * m
    return factors.prod(dim=0).square()


def stepwise_ddim(log_variance: Tensor, noise: Tensor, steps: int) -> Tensor:
    x = noise
    for i in range(steps, 0, -1):
        t = log_variance.new_tensor(i / steps)
        a, s = alpha_sigma(t)
        ap, sp = alpha_sigma(t - 1 / steps)
        v = velocity_coefficient(log_variance, t) * x
        x = ap * (a * x - s * v) + sp * (s * x + a * v)
    return x


def invert_endpoint_variance(desired: Tensor, steps: int) -> Tensor:
    """Demonstrate that the target is in the actual finite-DDIM model family."""
    lo, hi = torch.full_like(desired, -20), torch.full_like(desired, 20)
    for _ in range(100):
        mid = (lo + hi) / 2
        below = endpoint_variance(mid, steps) < desired
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    return (lo + hi) / 2


def gaussian_kl(q: Tensor, p: Tensor) -> Tensor:
    return 0.5 * (q / p - 1 + p.log() - q.log()).sum()


def log_ratio(z: Tensor, numerator: Tensor, denominator: Tensor) -> Tensor:
    return 0.5 * (denominator.log() - numerator.log()
                  - z.square() * (1 / numerator - 1 / denominator)).sum(dim=-1)


def reward_moments(q: Tensor, p: Tensor, ref: Tensor) -> tuple[Tensor, Tensor]:
    difference = 1 / p - 1 / ref
    mean = 0.5 * (ref.log() - p.log() - q * difference).sum()
    std = (0.5 * (q * difference).square().sum()).sqrt()
    return mean, std


def production_objective(theta: Tensor, ref_theta: Tensor, p: Tensor, ref: Tensor,
                         normals: Tensor, t: Tensor, eps: Tensor,
                         cfg: Settings) -> tuple[Tensor, dict[str, Tensor]]:
    # Rollout stop-gradient is crucial: differentiating it changes the algorithm.
    with torch.no_grad():
        x = normals * endpoint_variance(theta, cfg.ddim_steps).sqrt()
        rewards = log_ratio(x, p, ref)
        advantage, _ = compute_advantage(rewards, estimator="leave_one_out_unscaled")
        a, s = alpha_sigma(t[:, None, :, None])
        noisy = a * x.unsqueeze(0) + s * eps[:, None, :, :]
        target_v = a * eps[:, None, :, :] - s * x.unsqueeze(0)
        ref_v = velocity_coefficient(ref_theta, t)[:, None, :, :] * noisy
        loss_ref = (ref_v - target_v).square().mean(dim=-1)
    current_v = velocity_coefficient(theta, t)[:, None, :, :] * noisy
    loss_cur = (current_v - target_v).square().mean(dim=-1)
    terms = []
    for i in range(cfg.timesteps):
        term, _ = build_dgpo_loss(loss_cur[i], loss_ref[i], advantage, 1.0, cfg.candidates)
        terms.append(term)
    gate = torch.sigmoid((advantage.unsqueeze(0) * (loss_cur.detach() - loss_ref)).mean(dim=1))
    return torch.stack(terms).mean(), {
        "gate_mean": gate.mean(),
        "gate_saturated_fraction": ((gate < .01) | (gate > .99)).double().mean(),
        "advantage_std": advantage.std(unbiased=False),
    }


def auc(truth: Tensor, generated: Tensor) -> float:
    positives, negatives = truth.detach().numpy(), np.sort(generated.detach().numpy())
    return float(((np.searchsorted(negatives, positives, side="left")
                   + np.searchsorted(negatives, positives, side="right")) * .5).mean()
                 / len(negatives))


def metrics(theta: Tensor, p: Tensor, ref: Tensor, cfg: Settings,
            eval_noise: tuple[Tensor, Tensor] | None = None) -> dict[str, float]:
    with torch.no_grad():
        q = endpoint_variance(theta, cfg.ddim_steps)
        reward, reward_std = reward_moments(q, p, ref)
        result = {
            "kl_q_truth": float(gaussian_kl(q, p)),
            "kl_truth_q": float(gaussian_kl(p, q)),
            "kl_q_reference": float(gaussian_kl(q, ref)),
            "reward_mean": float(reward), "reward_std": float(reward_std),
            "correlation": float((q[0] - q[1]) / q.sum()),
            "marginal_variance": float(q.mean()),
            "marginal_variance_relative_error": float(q.mean() / p.mean() - 1),
            "variance_plus": float(q[0]), "variance_minus": float(q[1]),
            "theta_plus": float(theta[0]), "theta_minus": float(theta[1]),
        }
        if eval_noise is not None:
            truth, generated = eval_noise[0] * p.sqrt(), eval_noise[1] * q.sqrt()
            for name, denominator in [("fixed_oracle", ref), ("fresh_bayes", q)]:
                st, sg = log_ratio(truth, p, denominator), log_ratio(generated, p, denominator)
                result[name + "_auc"] = auc(st, sg)
                result[name + "_bce"] = float(.5 * (torch.nn.functional.softplus(-st).mean()
                                                     + torch.nn.functional.softplus(sg).mean()))
        return result


def cosine(a: Tensor, b: Tensor) -> float:
    denom = a.norm() * b.norm()
    return float(torch.dot(a, b) / denom) if denom > 1e-15 else 0.0


def train(arm: str, seed: int, cfg: Settings) -> list[dict]:
    rng = torch.Generator().manual_seed(seed)
    evaluation_rng = torch.Generator().manual_seed(100000 + seed)
    eval_noise = tuple(torch.randn(cfg.eval_samples, 2, dtype=torch.float64,
                                  generator=evaluation_rng) for _ in range(2))
    theta = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    ref_theta = theta.detach().clone()
    ref = endpoint_variance(ref_theta, cfg.ddim_steps)
    p = ref.mean() * torch.tensor([1 + cfg.rho, 1 - cfg.rho], dtype=torch.float64)
    optimizer = torch.optim.AdamW([theta], lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = [{"step": 0, "arm": arm, "seed": seed, **metrics(theta, p, ref, cfg, eval_noise)}]
    for step in range(1, cfg.steps + 1):
        normals = torch.randn(cfg.candidates, cfg.batch, 2, generator=rng, dtype=torch.float64)
        t = torch.rand(cfg.timesteps, cfg.batch, generator=rng, dtype=torch.float64) * cfg.t_max
        eps = torch.randn(cfg.timesteps, cfg.batch, 2, generator=rng, dtype=torch.float64)
        q = endpoint_variance(theta, cfg.ddim_steps)
        kl = gaussian_kl(q, ref)
        true_kl = gaussian_kl(q, p)
        if arm == "exact_reward_kl":
            main = -reward_moments(q, p, ref)[0]
            diag = {}
        else:
            main, diag = production_objective(theta, ref_theta, p, ref, normals, t, eps, cfg)
            main = cfg.main_scale * main
        coefficient = 0.0 if arm == "dgpo_only" else cfg.coefficient
        main_grad = torch.autograd.grad(main, theta, retain_graph=True)[0]
        kl_grad = coefficient * torch.autograd.grad(kl, theta, retain_graph=True)[0]
        truth_grad = torch.autograd.grad(true_kl, theta, retain_graph=True)[0]
        total_grad = main_grad + kl_grad
        if not torch.isfinite(total_grad).all():
            raise FloatingPointError(f"Nonfinite gradient: {arm}, seed={seed}, step={step}")
        old_theta, old_kl = theta.detach().clone(), float(true_kl.detach())
        optimizer.zero_grad(set_to_none=True)
        theta.grad = total_grad.detach().clone()
        preclip = float(torch.nn.utils.clip_grad_norm_([theta], 1.0))
        optimizer.step()
        displacement = theta.detach() - old_theta
        row = {"step": step, "arm": arm, "seed": seed,
               **metrics(theta, p, ref, cfg, eval_noise if step % cfg.eval_every == 0 or step == cfg.steps else None),
               "main_loss": float(main.detach()), "weighted_kl_loss": float((coefficient * kl).detach()),
               "gradient_norm": preclip, "clip_active": int(preclip > 1),
               "main_gradient_norm": float(main_grad.norm()), "kl_gradient_norm": float(kl_grad.norm()),
               "main_kl_cosine": cosine(main_grad, kl_grad),
               "main_truth_cosine": cosine(main_grad, truth_grad),
               "total_truth_cosine": cosine(total_grad, truth_grad),
               "gradient_cancellation": float(total_grad.norm() / (main_grad.norm() + kl_grad.norm() + 1e-30)),
               "update_norm": float(displacement.norm()),
               "predicted_true_kl_change": float(torch.dot(truth_grad, displacement)),
               "actual_true_kl_change": 0.0,
               **{k: float(v.detach()) for k, v in diag.items()}}
        row["actual_true_kl_change"] = row["kl_q_truth"] - old_kl
        for index, suffix in enumerate(["plus", "minus"]):
            row["main_gradient_" + suffix] = float(main_grad[index])
            row["kl_gradient_" + suffix] = float(kl_grad[index])
            row["truth_gradient_" + suffix] = float(truth_grad[index])
        if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError(f"Nonfinite measurement: {arm}, seed={seed}, step={step}")
        history.append(row)
    return history


def decide(histories: list[list[dict]], cfg: Settings) -> dict:
    by_run = {(h[0]["arm"], h[0]["seed"]): h for h in histories}
    seeds = sorted({h[0]["seed"] for h in histories})
    decisions = []
    for seed in seeds:
        exact, dgpo = by_run[("exact_reward_kl", seed)], by_run[("dgpo_endpoint_kl", seed)]
        initial = exact[0]["kl_q_truth"]
        late = dgpo[max(1, int(.8 * cfg.steps)):]
        middle = len(late) // 2
        first = np.mean([r["kl_q_truth"] for r in late[:middle]])
        last = np.mean([r["kl_q_truth"] for r in late[middle:]])
        checks = {
            "positive_control_closes": exact[-1]["kl_q_truth"] <= .01 * initial,
            "initial_dgpo_improves": dgpo[min(10, cfg.steps)]["kl_q_truth"] <= .999 * initial,
            "substantial_residual": np.mean([r["kl_q_truth"] for r in late]) >= .1 * initial,
            "late_plateau": first - last <= .01 * initial,
        }
        decisions.append({"seed": seed, "checks": {k: bool(v) for k, v in checks.items()},
                          "reproduced": all(checks.values()), "initial_kl": initial,
                          "control_final_kl": exact[-1]["kl_q_truth"],
                          "dgpo_final_kl": dgpo[-1]["kl_q_truth"],
                          "dgpo_late_mean_kl": float(np.mean([r["kl_q_truth"] for r in late]))})
    reproduced = all(d["reproduced"] for d in decisions)
    control_valid = all(d["checks"]["positive_control_closes"] for d in decisions)
    status = "inconclusive" if not control_valid else "not_reproduced"
    if reproduced:
        status = "replicated_toy_failure" if len(seeds) >= 3 else "exploratory_toy_failure"
    return {"status": status,
            "seeds": decisions,
            "scope": "Oracle two-dimensional Gaussian toy, not a causal diagnosis of production H4."}


def plot(histories: list[list[dict]], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    names = [("kl_q_truth", "Exact KL(q || truth)"), ("reward_mean", "Exact fixed reward mean"),
             ("correlation", "Generated joint correlation"), ("reward_std", "Exact fixed reward std"),
             ("fresh_bayes_auc", "Current Bayes AUC (independent MC)"),
             ("marginal_variance_relative_error", "Relative marginal variance drift")]
    colors = {"exact_reward_kl": "#277da1", "dgpo_endpoint_kl": "#e76f51", "dgpo_only": "#6a994e"}
    for ax, (key, label) in zip(axes.flat, names):
        labeled = set()
        for h in histories:
            arm = h[0]["arm"]
            rows = [r for r in h if key in r]
            ax.plot([r["step"] for r in rows], [r[key] for r in rows], color=colors[arm], alpha=.7,
                    label=arm if arm not in labeled else None)
            labeled.add(arm)
        ax.set_title(label)
        ax.set_xlabel("Policy updates")
        ax.grid(alpha=.2)
    axes[0, 0].set_yscale("symlog", linthresh=1e-4)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exact-ratio DDIM toy — not a production causal diagnosis")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dgpo_toy/baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    for name, field in Settings.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args()
    cfg = Settings(**{k: getattr(args, k) for k in Settings.__dataclass_fields__})
    if cfg.steps < 20 or cfg.batch < 1 or cfg.candidates < 2 or cfg.timesteps < 1 or cfg.ddim_steps < 2:
        parser.error("Need steps>=20, batch>=1, K>=2, timesteps>=1, DDIM steps>=2")
    if (not 0 < cfg.rho < 1 or not 0 < cfg.t_max <= 1 or cfg.lr <= 0
            or cfg.coefficient < 0 or cfg.main_scale <= 0 or cfg.weight_decay < 0):
        parser.error("Invalid correlation, t_max, LR or coefficient")
    if cfg.eval_every < 1 or cfg.eval_samples < 2 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Invalid evaluation settings or duplicate seeds")
    torch.set_num_threads(1)
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    histories = []
    for seed in args.seeds:
        for arm in ["exact_reward_kl", "dgpo_endpoint_kl", "dgpo_only"]:
            h = train(arm, seed, cfg)
            histories.append(h)
            print(json.dumps({"arm": arm, "seed": seed, "initial_kl": h[0]["kl_q_truth"],
                              "final_kl": h[-1]["kl_q_truth"], "final_rho": h[-1]["correlation"]}), flush=True)
    rows = [r for h in histories for r in h]
    with (args.output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    theta = torch.zeros(2, dtype=torch.float64)
    ref = endpoint_variance(theta, cfg.ddim_steps)
    truth = ref.mean() * torch.tensor([1 + cfg.rho, 1 - cfg.rho], dtype=torch.float64)
    target_theta = invert_endpoint_variance(truth, cfg.ddim_steps)
    report = {"protocol": "dgpo-toy-v1", "settings": asdict(cfg), "seeds": args.seeds,
              "production_kernel_source": str(LOSS_SOURCE),
              "source_sha256": source_fingerprints(),
              "reference_endpoint_eigenvalues": ref.tolist(), "truth_eigenvalues": truth.tolist(),
              "target_theta": target_theta.tolist(),
              "representability_error": float((endpoint_variance(target_theta, cfg.ddim_steps) - truth).abs().max()),
              "decision": decide(histories, cfg),
              "endpoints": [h[-1] for h in histories], "wall_seconds": time.perf_counter() - started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot(histories, args.output)
    print(json.dumps({"decision": report["decision"], "wall_seconds": report["wall_seconds"],
                      "report": str(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
