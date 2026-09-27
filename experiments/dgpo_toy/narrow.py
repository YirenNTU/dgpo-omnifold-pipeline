"""Pure-reward DDIM toy: matched broad/narrow joint ridges and exact ratios."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor

try:
    from .run import (ROOT, LOSS_SOURCE, alpha_sigma, build_dgpo_loss,
                      compute_advantage, endpoint_variance, invert_endpoint_variance,
                      velocity_coefficient)
except ImportError:
    from run import (ROOT, LOSS_SOURCE, alpha_sigma, build_dgpo_loss,
                     compute_advantage, endpoint_variance, invert_endpoint_variance,
                     velocity_coefficient)

ARMS = ("dgpo", "score_mc", "population")


@dataclass(frozen=True)
class Config:
    steps: int = 600
    batch: int = 256
    candidates: int = 8
    timesteps: int = 8
    ddim_steps: int = 20
    tau_broad: float = 0.25
    tau_narrow: float = 0.0002
    mass: float = 0.8
    lr: float = 0.03
    weight_decay: float = 0.001
    t_max: float = 0.7
    quadrature_order: int = 256
    eval_every: int = 25
    eval_samples: int = 23927


def fixed_reward(z: Tensor, ref_variance: Tensor, tau: float, mass: float) -> Tensor:
    log_spike_ratio = -math.log(tau) - 0.5 * z.square() / ref_variance * (1/tau**2 - 1)
    return torch.logaddexp(log_spike_ratio + math.log(mass),
                          torch.full_like(z, math.log1p(-mass)))


def observe(context: Tensor, z: Tensor, ref_variance: Tensor) -> tuple[Tensor, Tensor]:
    uniform_residual = .5 * (1 + torch.erf(z / (2 * ref_variance).sqrt()))
    return context, (context + uniform_residual - .5).remainder(1)


@lru_cache(maxsize=8)
def quadrature(order: int) -> tuple[Tensor, Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (torch.tensor(14 * nodes, dtype=torch.float64),
            torch.tensor(14 * weights, dtype=torch.float64))


def population_moments(relative_variance: Tensor, tau: float, mass: float,
                       order: int = 256) -> tuple[Tensor, Tensor]:
    """Scale-adapted quadrature: integrate reward above its constant background.

    The detached coordinate scale changes the integration grid, not the density
    derivative. Full-domain integrals are invariant to that coordinate choice;
    |u|>14 tails are negligible for either min(policy_sigma, spike_sigma).
    """
    nodes, weights = quadrature(order)
    scale = torch.minimum(relative_variance.detach().sqrt(), relative_variance.new_tensor(tau))
    z = scale * nodes
    measure = weights * scale * torch.exp(-.5 * z.square() / relative_variance)
    measure = measure / (2 * math.pi * relative_variance).sqrt()
    extra = torch.nn.functional.softplus(math.log(mass/(1-mass)/tau)
                                         - .5*z.square()*(1/tau**2-1))
    first = (measure * extra).sum()
    second = (measure * extra.square()).sum()
    return math.log1p(-mass) + first, (second-first.square()).clamp_min(0)


def population_weight_ess(relative_variance: Tensor, tau: float, mass: float) -> tuple[Tensor, Tensor]:
    eh = 1/tau / (1 + relative_variance*(1/tau**2-1)).sqrt()
    eh2 = 1/tau**2 / (1 + 2*relative_variance*(1/tau**2-1)).sqrt()
    mean = 1-mass + mass*eh
    second = (1-mass)**2 + 2*mass*(1-mass)*eh + mass**2*eh2
    return mean.square()/second, mean


def concentration(values: Tensor, prefix: str) -> dict:
    weights = values.detach().abs().flatten()
    total = float(weights.sum())
    if total == 0:
        return {prefix+"_ess_fraction": None, prefix+"_top1pct_mass": None,
                prefix+"_max_mass": None, prefix+"_total": 0.}
    normalized = weights / total
    return {prefix+"_ess_fraction": float(1 / (len(weights)*normalized.square().sum())),
            prefix+"_top1pct_mass": float(normalized.topk(max(1, math.ceil(.01*len(weights)))).values.sum()),
            prefix+"_max_mass": float(normalized.max()), prefix+"_total": total}


def sample_objective(arm: str, theta: Tensor, ref: Tensor, normals: Tensor,
                     t: Tensor, eps: Tensor, tau: float, cfg: Config):
    q = endpoint_variance(theta, cfg.ddim_steps).squeeze()
    dq = torch.autograd.grad(q, theta, retain_graph=True)[0].detach().squeeze()
    z = (normals*q.detach().sqrt()).detach()  # [K,B]; never backprop through rollout.
    rewards = fixed_reward(z, ref, tau, cfg.mass)
    advantage, _ = compute_advantage(rewards, estimator="leave_one_out_unscaled")
    diag = {
        "rollout_reward_mean": float(rewards.mean()),
        "rollout_reward_std": float(rewards.std(unbiased=False)),
        "group_hit_fraction": float((z.abs() < 3*tau*ref.sqrt()).any(dim=0).double().mean()),
        "group_informative_fraction": float(((rewards.max(0).values-rewards.min(0).values)>1e-3).double().mean()),
        "advantage_abs_mean": float(advantage.abs().mean()),
        "within_k_fixed_weight_ess_mean": float((1/(cfg.candidates*rewards.softmax(0).square().sum(0))).mean()),
    }
    diag.update(concentration((rewards-rewards.max()).exp(), "rollout_fixed_weight"))
    if arm == "score_mc":
        log_q = -.5 * (math.log(2*math.pi) + q.log() + z.square()/q)
        loss = -(advantage*log_q).mean()
        group_grad = -(advantage * .5*(z.square()/q.detach()-1) * dq/q.detach()).mean(0)
    elif arm == "dgpo":
        with torch.no_grad():
            a, s = alpha_sigma(t)
            noisy = a[:, None, :]*z[None, :, :] + s[:, None, :]*eps[:, None, :]
            target = a[:, None, :]*eps[:, None, :] - s[:, None, :]*z[None, :, :]
            ref_m = velocity_coefficient(torch.zeros_like(theta), t).squeeze(-1)
            loss_ref = (ref_m[:, None, :]*noisy-target).square()
        m = velocity_coefficient(theta, t).squeeze(-1)
        error = m[:, None, :]*noisy-target
        loss_cur = error.square()  # mean over a single coordinate is identity.
        terms = [build_dgpo_loss(loss_cur[i], loss_ref[i], advantage, 1., cfg.candidates)[0]
                 for i in range(cfg.timesteps)]
        loss = torch.stack(terms).mean()
        with torch.no_grad():
            gate = torch.sigmoid((advantage[None, :, :]*(loss_cur-loss_ref)).mean(1))
            variance = theta.exp().squeeze()
            dm = -a*s*variance / (a.square()*variance+s.square()).square()
            group_grad = (gate[:, None, :]*advantage[None, :, :]
                          *2*error*dm[:, None, :]*noisy).mean(dim=(0, 1))
            diag["gate_mean"] = float(gate.mean())
            diag["gate_saturated_fraction"] = float(((gate<.01)|(gate>.99)).double().mean())
    else:
        raise ValueError(f"Unknown sampled arm: {arm}")
    diag.update(concentration(group_grad, "gradient_contribution"))
    diag["group_gradient_mean"] = float(group_grad.mean())
    diag["gradient_signed_to_abs_ratio"] = float(group_grad.sum().abs()/(group_grad.abs().sum()+1e-300))
    return loss, diag, group_grad


def metrics(theta: Tensor, ref: Tensor, tau: float, cfg: Config,
            initial_reward: float, eval_noise: Tensor | None = None) -> dict:
    with torch.no_grad():
        relative = endpoint_variance(theta, cfg.ddim_steps).squeeze()/ref
        reward, variance = population_moments(relative, tau, cfg.mass, cfg.quadrature_order)
        ceiling = math.log(1-cfg.mass+cfg.mass/tau)
        ess, mean_ratio = population_weight_ess(relative, tau, cfg.mass)
        result = {"theta": float(theta.item()), "relative_sigma": float(relative.sqrt()),
                  "sigma_to_spike": float(relative.sqrt()/tau),
                  "reward_mean": float(reward), "reward_std": float(variance.sqrt()),
                  "reward_gain": float(reward)-initial_reward,
                  "reward_gain_fraction": (float(reward)-initial_reward)/(ceiling-initial_reward),
                  "population_hit_fraction": float(torch.erf(3*tau/(2*relative).sqrt())),
                  "fixed_weight_population_ess_fraction": float(ess),
                  "fixed_weight_population_mean": float(mean_ratio)}
        if eval_noise is not None:
            scores = fixed_reward(eval_noise*(relative*ref).sqrt(), ref, tau, cfg.mass)
            result.update(concentration((scores-scores.max()).exp(), "heldout_fixed_weight"))
            result["heldout_reward_mean"] = float(scores.mean())
            result["heldout_reward_se"] = float(scores.std(unbiased=True)/math.sqrt(len(scores)))
        return result


def train(arm: str, width: str, tau: float, seed: int, cfg: Config) -> list[dict]:
    if arm not in ARMS:
        raise ValueError(arm)
    rng = torch.Generator().manual_seed(seed)
    evaluation = torch.Generator().manual_seed(100000+seed)
    eval_noise = torch.randn(cfg.eval_samples, dtype=torch.float64, generator=evaluation)
    theta = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    ref = endpoint_variance(theta.detach(), cfg.ddim_steps).squeeze()
    initial_reward = float(population_moments(torch.ones_like(ref), tau, cfg.mass, cfg.quadrature_order)[0])
    optimizer = torch.optim.AdamW([theta], lr=cfg.lr, weight_decay=cfg.weight_decay)
    common = {"arm": arm, "width": width, "tau": tau, "seed": seed}
    history = [{**common, "step": 0, **metrics(theta, ref, tau, cfg, initial_reward, eval_noise)}]
    for step in range(1, cfg.steps+1):
        normals = torch.randn(cfg.candidates, cfg.batch, generator=rng, dtype=torch.float64)
        t = cfg.t_max*torch.rand(cfg.timesteps, cfg.batch, generator=rng, dtype=torch.float64)
        eps = torch.randn(cfg.timesteps, cfg.batch, generator=rng, dtype=torch.float64)
        q = endpoint_variance(theta, cfg.ddim_steps).squeeze()
        reward = population_moments(q/ref, tau, cfg.mass, cfg.quadrature_order)[0]
        oracle_gradient = torch.autograd.grad(-reward, theta, retain_graph=True)[0]
        if arm == "population":
            loss, diag = -reward, {}
        else:
            loss, diag, _ = sample_objective(arm, theta, ref, normals, t, eps, tau, cfg)
        gradient = torch.autograd.grad(loss, theta)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite gradient: {common}, step={step}")
        old_theta, old_reward = theta.detach().clone(), float(reward.detach())
        optimizer.zero_grad(set_to_none=True)
        theta.grad = gradient.detach().clone()
        preclip = float(torch.nn.utils.clip_grad_norm_([theta], 1.))
        optimizer.step()
        displacement = float((theta.detach()-old_theta).item())
        row = {**common, "step": step,
               **metrics(theta, ref, tau, cfg, initial_reward,
                         eval_noise if step % cfg.eval_every == 0 or step == cfg.steps else None),
               **diag, "gradient": float(gradient.item()),
               "population_negative_reward_gradient": float(oracle_gradient.item()),
               "gradient_norm": preclip, "clip_active": int(preclip>1),
               "gradient_sign_matches_population": int(float((gradient*oracle_gradient).item())>0),
               "update": displacement,
               "predicted_reward_change": -float(oracle_gradient.item())*displacement}
        row["actual_reward_change"] = row["reward_mean"]-old_reward
        if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise FloatingPointError(f"Nonfinite metric: {common}, step={step}")
        history.append(row)
    return history


def decide(histories: list[list[dict]], cfg: Config) -> dict:
    runs = {(h[0]["width"], h[0]["arm"], h[0]["seed"]): h for h in histories}
    decisions = []
    for seed in sorted({h[0]["seed"] for h in histories}):
        broad, narrow = (runs[(width, "dgpo", seed)] for width in ["broad", "narrow"])
        ob, on = (runs[(width, "population", seed)][-1]["reward_gain_fraction"] for width in ["broad", "narrow"])
        db, dn = broad[-1]["reward_gain_fraction"], narrow[-1]["reward_gain_fraction"]
        checks = {"low_narrow_ess": narrow[0]["fixed_weight_population_ess_fraction"]<=.001,
                  "broad_ess_valid": broad[0]["fixed_weight_population_ess_fraction"]>=.1,
                  "population_controls_valid": min(ob, on)>=.5,
                  "broad_dgpo_learns": db>=.5,
                  "narrow_slower_than_broad": dn<=.25*db,
                  "narrow_deficit_vs_oracle": dn<=.25*on}
        decisions.append({"seed": seed, "checks": checks, "reproduced": all(checks.values()),
                          "broad_dgpo_gain_fraction": db, "narrow_dgpo_gain_fraction": dn,
                          "broad_population_gain_fraction": ob, "narrow_population_gain_fraction": on,
                          "narrow_score_mc_gain_fraction": runs[("narrow", "score_mc", seed)][-1]["reward_gain_fraction"]})
    if not all(d["checks"]["population_controls_valid"] for d in decisions):
        status = "inconclusive_control_budget"
    elif all(d["reproduced"] for d in decisions):
        status = "replicated_pure_reward_slowdown" if len(decisions)>=3 else "exploratory_pure_reward_slowdown"
    elif any(d["reproduced"] for d in decisions):
        status = "mixed"
    else:
        status = "not_reproduced"
    return {"status": status, "seeds": decisions,
            "scope": "Relative finite-budget slowdown, not proof of a stationary plateau or production cause."}


def plot(histories: list[list[dict]], output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    colors = {"dgpo": "#e76f51", "score_mc": "#4f8a42", "population": "#277da1"}
    for col, width in enumerate(["broad", "narrow"]):
        for row, (key, label) in enumerate([
            ("reward_gain_fraction", "Fraction of available fixed reward gained"),
            ("sigma_to_spike", "Policy sigma / spike sigma"),
            ("gradient_contribution_ess_fraction", "Per-group absolute gradient ESS / B")]):
            ax = axes[row, col]
            labels = set()
            for h in histories:
                if h[0]["width"] != width:
                    continue
                arm = h[0]["arm"]
                rows = [r for r in h if r.get(key) is not None]
                ax.plot([r["step"] for r in rows], [r[key] for r in rows],
                        color=colors[arm], alpha=.7, linewidth=1,
                        label=arm if arm not in labels else None)
                labels.add(arm)
            ax.set_title(f"{width}: {label}")
            ax.set_xlabel("Policy updates")
            ax.grid(alpha=.2)
            if row:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-.03, 1.03)
                ax.legend(fontsize=8)
    fig.suptitle("Pure reward, no additive KL — narrow-joint toy, not production proof")
    fig.tight_layout()
    fig.savefig(output/"curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/dgpo_toy/narrow_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument("--"+name.replace("_", "-"), type=type(field.default), default=field.default)
    args = parser.parse_args()
    cfg = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    if not all(math.isfinite(v) for v in asdict(cfg).values()):
        parser.error("All settings must be finite")
    if (cfg.steps<1 or cfg.batch<1 or cfg.candidates<2 or cfg.timesteps<1 or cfg.ddim_steps<2
            or not 0<cfg.tau_narrow<cfg.tau_broad<1 or not 0<cfg.mass<1
            or cfg.lr<=0 or cfg.weight_decay<0 or not 0<cfg.t_max<=1
            or cfg.quadrature_order<64 or cfg.eval_every<1 or cfg.eval_samples<2
            or len(set(args.seeds))!=len(args.seeds)):
        parser.error("Invalid toy settings")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    histories, validity = [], {}
    ref = endpoint_variance(torch.zeros(1, dtype=torch.float64), cfg.ddim_steps).squeeze()
    for width, tau in [("broad", cfg.tau_broad), ("narrow", cfg.tau_narrow)]:
        desired = (ref*(.5*tau)**2).reshape(1)
        witness = invert_endpoint_variance(desired, cfg.ddim_steps)
        witness_q = endpoint_variance(witness, cfg.ddim_steps).squeeze()/ref
        initial = population_moments(torch.ones_like(ref), tau, cfg.mass, cfg.quadrature_order)[0]
        witness_r = population_moments(witness_q, tau, cfg.mass, cfg.quadrature_order)[0]
        errors, grad_errors = [], []
        for sigma in [1., math.sqrt(tau), tau, .1*tau]:
            v = torch.tensor(sigma**2, dtype=torch.float64, requires_grad=True)
            r1 = population_moments(v, tau, cfg.mass, cfg.quadrature_order)[0]
            r2 = population_moments(v, tau, cfg.mass, 2*cfg.quadrature_order)[0]
            errors.append(float((r1-r2).abs().detach()))
            g1, g2 = (torch.autograd.grad(r, v)[0] for r in [r1, r2])
            grad_errors.append(float((g1-g2).abs()/(g2.abs()+1e-30)))
        validity[width] = {"witness_theta": witness.tolist(), "witness_reward": float(witness_r),
                           "witness_gain_fraction": float((witness_r-initial)/(math.log(1-cfg.mass+cfg.mass/tau)-initial)),
                           "max_quadrature_doubling_error": max(errors),
                           "max_quadrature_gradient_relative_error": max(grad_errors),
                           "witness_variance_relative_error": float((witness_q*ref/desired.squeeze()-1).abs())}
        if max(errors)>1e-7 or max(grad_errors)>1e-5 or validity[width]["witness_variance_relative_error"]>1e-6:
            raise RuntimeError(f"Instrument validity failed: {validity[width]}")
        for seed in args.seeds:
            for arm in ARMS:
                h = train(arm, width, tau, seed, cfg)
                histories.append(h)
                print(json.dumps({k: h[-1][k] for k in ["width", "arm", "seed", "reward_mean", "reward_gain_fraction", "sigma_to_spike"]}), flush=True)
    rows = [r for h in histories for r in h]
    with (args.output/"history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for r in rows for key in r}))
        writer.writeheader()
        writer.writerows(rows)
    report = {"protocol": "pure-reward-narrow-joint-v1", "settings": asdict(cfg),
              "seeds": args.seeds, "production_kernel_source": str(LOSS_SOURCE),
              "additive_kl": False, "reward_frozen": True, "instrument_validity": validity,
              "decision": decide(histories, cfg), "initials": [h[0] for h in histories],
              "endpoints": [h[-1] for h in histories],
              "wall_seconds_training_and_measurement": time.perf_counter()-start}
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    plot(histories, args.output)
    print(json.dumps({"decision": report["decision"], "report": str(args.output/"report.json"),
                      "wall_seconds": report["wall_seconds_training_and_measurement"]}, indent=2))


if __name__ == "__main__":
    main()
