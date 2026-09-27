"""Native DGPO after Fourier truth pretraining, with a frozen measured mode ratio.

The numerator is known binned truth. The denominator is independently sampled
from the ACTUAL frozen source diffusion. This is not an exact neural density
ratio, and the velocity-MSE penalty is not endpoint KL.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .conditional import Config, Denoiser, ddim, generator, paired_gain, policy_train, weight_health
from .coverage_budget import flatten_metrics
from .cube_lockdown import ConditionDenoiser
from .cube_truth_fourier import evaluate as sample_panel, truth_data
from .truth_pretrain import atomic_checkpoint, atomic_json

RUN_NAME = "Can DGPO refine pretraining? | Fourier cube | frozen mode ratio | V MSE 1"


def mode_ids(y):
    return (y[..., 0] >= 0).long() * 4 + (y[..., 1] >= 0).long() * 2 + (y[..., 2] >= 0).long()


def bin_ids(c, bins=256):
    return ((c[..., 0] + 1) * bins / 2).floor().long().clamp(0, bins - 1)


def target_table(bins=256, frequency=8):
    edges = torch.linspace(-1, 1, bins + 1, dtype=torch.float64)
    mean_sine = ((frequency * math.pi * edges[:-1]).cos() -
                 (frequency * math.pi * edges[1:]).cos()) / (frequency * math.pi * 2 / bins)
    positive = .5 + .4 * mean_sine
    ids = torch.arange(8)
    parity = ((2 * ((ids >> 2) & 1) - 1) * (2 * ((ids >> 1) & 1) - 1) * (2 * (ids & 1) - 1))
    return torch.where(parity[None] > 0, positive[:, None], 1 - positive[:, None]) / 4


class FrozenModeRatio(torch.nn.Module):
    """Fixed log[p_truth(mode|condition-bin)/q_source(mode|condition-bin)]."""
    def __init__(self, counts, pseudocount=.5):
        super().__init__()
        counts = counts.detach().double()
        if (counts.ndim != 2 or counts.shape != (256, 8) or not torch.isfinite(counts).all()
                or (counts < 0).any() or (counts.sum(-1) <= 0).any()
                or not math.isfinite(pseudocount) or pseudocount <= 0):
            raise ValueError("Need finite, nonnegative 256x8 counts and a positive pseudocount")
        q = (counts + pseudocount) / (counts.sum(-1, keepdim=True) + 8 * pseudocount)
        p = target_table()
        self.register_buffer("counts", counts.clone())
        self.register_buffer("reference_probabilities", q)
        self.register_buffer("target_probabilities", p)
        self.register_buffer("log_ratios", (p.log() - q.log()).float())
        self.pseudocount = pseudocount

    def forward(self, y, c, data=None):
        c = c.expand(*y.shape[:-1], 1)
        return self.log_ratios[bin_ids(c), mode_ids(y)]


def load_source(source, seed, steps, eval_every):
    path = source / f"seed{seed}" / "early_stop_fourier_model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved.get("basis") != "fourier" or saved.get("trained_on") != "complete_truth"
            or saved.get("weights") != "raw_no_ema" or saved.get("condition_frequency") != 8):
        raise ValueError("Use a selected raw-weight Fourier complete-truth k8 checkpoint")
    cfg = replace(Config(**saved["config"]), policy_steps=steps, eval_every=eval_every,
        eval_events=512, batch=64, candidates=8, timesteps=4, policy_lr=1e-4, weight_decay=.001)
    base = Denoiser(cfg)
    initial = ConditionDenoiser(cfg, "fourier", base.state_dict())
    initial.load_state_dict(saved["model"], strict=True)
    initial.eval().requires_grad_(False)
    if not all(torch.isfinite(v).all() for v in initial.state_dict().values()):
        raise FloatingPointError("Nonfinite source checkpoint")
    return cfg, initial, saved, path


@torch.no_grad()
def calibration_counts(model, cfg, seed, contexts=8192, candidates=64, emit=lambda row: None):
    """Stream mode counts only; do not reuse training/selection/endpoint panels."""
    if contexts < 256 or contexts % 256 or candidates < 1:
        raise ValueError("Calibration needs balanced conditions and positive candidates")
    c = ((torch.arange(contexts, dtype=torch.float32) + .5) / contexts * 2 - 1)[:, None]
    rng = generator(seed); counts = torch.zeros(256, 8, dtype=torch.int64)
    model.eval()
    for start in range(0, contexts, 128):
        cc = c[start:start + 128]
        z = torch.randn(len(cc), candidates, 3, generator=rng)
        y = ddim(model, cc[:, None], z, cfg.ddim_steps)
        if not torch.isfinite(y).all():
            raise FloatingPointError("Nonfinite calibration generation")
        ids = bin_ids(cc)[:, None] * 8 + mode_ids(y)
        counts += torch.bincount(ids.flatten(), minlength=256 * 8).reshape(256, 8)
        if start == 0 or (start + len(cc)) % 1024 == 0 or start + len(cc) == contexts:
            emit({"phase": "calibration", "step": 0, "contexts": start + len(cc),
                  "total_contexts": contexts, "samples": int(counts.sum())})
    return counts


def binned_probabilities(panel):
    y = panel["samples"]
    return F.one_hot(mode_ids(y), 8).double().reshape(256, -1, y.shape[1], 8).mean((1, 2))


def split_joint_squared_error(panel):
    """Independent candidate halves remove the plug-in squared-error noise floor.

Both halves cover the same condition grid. Their residual cross-product is
unbiased for the squared error of the grid-averaged mode probabilities; it
can be negative in a finite panel and is never clamped or square-rooted.
"""
    y = panel["samples"]
    if y.shape[1] < 2 or y.shape[1] % 2:
        raise ValueError("Use an even number of evaluation candidates")
    probs = F.one_hot(mode_ids(y), 8).double().reshape(256, -1, y.shape[1], 8)
    c = panel["condition"][:, 0].double()
    positive = .5 + .4 * torch.sin(8 * math.pi * c)
    modes = torch.arange(8)
    parity = (2 * ((modes >> 2) & 1) - 1) * (2 * ((modes >> 1) & 1) - 1) * (2 * (modes & 1) - 1)
    target = (torch.where(parity[None] > 0, positive[:, None], 1 - positive[:, None]) / 4).reshape(256, -1, 8).mean(1)
    half = y.shape[1] // 2
    a, b = probs[:, :, :half].mean((1, 2)), probs[:, :, half:].mean((1, 2))
    return float(((a - target) * (b - target)).sum(-1).mean())


@torch.no_grad()
def evaluate(model, critic, data, cfg, seed, contexts=8192, candidates=32):
    panel, metrics = sample_panel(model, data, replace(cfg, eval_events=contexts, candidates=candidates), seed)
    y, c = panel["samples"], panel["condition"]
    rewards = critic(y, c[:, None], data)
    parity = torch.where(y >= 0, 1., -1.).prod(-1)
    moment = data.condition_signal(c[:, 0])[:, None] * parity
    panel.update(reward=rewards, moment=moment)
    metrics.update(reward_mean=float(rewards.double().mean()), reward_std=float(rewards.double().std(unbiased=False)),
        weight_health=weight_health(rewards), joint_squared_error_cross=split_joint_squared_error(panel))
    return panel, metrics


def reward_preflight(panel, critic, counts_a, counts_b):
    """Independent fixed-support counterfactual, NOT a policy update or fit gate."""
    q = binned_probabilities(panel)
    p = critic.target_probabilities
    tilted = q * (p / critic.reference_probabilities)
    normalization = tilted.sum(-1)
    tilted /= normalization[:, None]
    baseline_tv = float((q - p).abs().sum(-1).mean() / 2)
    ideal_tv = float((tilted - p).abs().sum(-1).mean() / 2)
    r = critic.log_ratios.double()
    baseline_reward = float((q * r).sum(-1).mean())
    ideal_reward = float((tilted * r).sum(-1).mean())
    y, c = panel["samples"], panel["condition"]
    a = FrozenModeRatio(counts_a)(y, c[:, None]).double()
    b = FrozenModeRatio(counts_b)(y, c[:, None]).double()
    a -= a.mean(-1, keepdim=True); b -= b.mean(-1, keepdim=True)
    norm = float(a.norm() * b.norm())
    return {"baseline_mode_tv": baseline_tv, "ideal_reweighted_mode_tv": ideal_tv,
        "ideal_mode_tv_improvement": baseline_tv - ideal_tv,
        "baseline_reward": baseline_reward, "ideal_reweighted_reward": ideal_reward,
        "ideal_reward_headroom": ideal_reward - baseline_reward,
        "mean_ratio": float(normalization.mean()), "max_bin_normalization_error": float((normalization - 1).abs().max()),
        "split_calibration_centered_reward_cosine": float((a * b).sum()) / norm if norm > 0 else None,
        "split_calibration_reference_tv": float((FrozenModeRatio(counts_a).reference_probabilities -
            FrozenModeRatio(counts_b).reference_probabilities).abs().sum(-1).mean() / 2),
        "min_reference_mode_count": float(critic.counts.min()),
        "interpretation": "Diagnostic only: independent binned reweighting, not K=8 resampling or attainable policy optimum"}


def endpoint_changes(panel, metrics, baseline, initial_metrics, preflight):
    reward = paired_gain(panel["reward"], baseline["reward"])
    headroom = preflight["ideal_reward_headroom"]
    return {"reward_gain": reward, "moment_gain": paired_gain(panel["moment"], baseline["moment"]),
        "reward_gain_over_ideal_headroom": reward["gain"] / headroom if headroom > 0 else None,
        "conditional_mode_tv_change": metrics["conditional_mode_tv"] - initial_metrics["conditional_mode_tv"],
        "joint_squared_error_cross_change": metrics["joint_squared_error_cross"] - initial_metrics["joint_squared_error_cross"],
        "moment_abs_error_change": metrics["condition_moment_abs_error"] - initial_metrics["condition_moment_abs_error"],
        "near_corner_change": metrics["near_corner_fraction"] - initial_metrics["near_corner_fraction"]}


def decisions(arms, seeds, steps):
    if len(arms) != len(seeds) or any(a["state"] != "completed" for a in arms.values()):
        return {"state": "pending"}
    endpoints = [a["evaluations"][str(steps)] for a in arms.values()]
    return {"state": "completed",
        "reward_absorption_all_seeds": all(x["changes"]["reward_gain"]["lo95"] > 0 for x in endpoints),
        "joint_squared_error_improves_all_seeds": all(x["changes"]["joint_squared_error_cross_change"] < 0 for x in endpoints),
        "joint_tv_improves_all_seeds": all(x["changes"]["conditional_mode_tv_change"] < 0 for x in endpoints),
        "moment_error_improves_all_seeds": all(x["changes"]["moment_abs_error_change"] < 0 for x in endpoints),
        "corner_guardrail_all_seeds": all(x["changes"]["near_corner_change"] >= -.02 for x in endpoints),
        "preflight_direction_all_seeds": all(a["preflight"]["ideal_mode_tv_improvement"] > 0 for a in arms.values()),
        "scope": "Estimated fixed categorical ratio and velocity-MSE surrogate; no exact endpoint-KL or production claim"}


def run(source, output, *, seeds=(17, 23, 41), steps=3000, eval_every=100,
        calibration_contexts=8192, calibration_candidates=64, endpoint_contexts=8192,
        endpoint_candidates=32, wandb_mode="offline", resume=False, run_id=None):
    if (not seeds or len(set(seeds)) != len(seeds) or min(steps, eval_every) < 1
        or calibration_contexts < 256 or calibration_contexts % 256 or calibration_candidates < 1
        or endpoint_contexts < 512 or endpoint_contexts % 256 or endpoint_candidates < 2 or endpoint_candidates % 2):
        raise ValueError("Invalid seeds, budgets, balanced grids, or candidate counts")
    if source.resolve() == output.resolve():
        raise ValueError("DGPO output must be separate from the pretrained source")
    source_report = json.loads((source / "report.json").read_text())
    if source_report["state"] != "completed":
        raise ValueError("Pretraining source must be completed")
    for seed in seeds:
        if source_report["pairs"][str(seed)]["state"] != "completed":
            raise ValueError("Requested source seed did not complete")
    contract = {"schema": "cube_pretrained_dgpo_v1", "source": str(source.resolve()), "seeds": list(seeds),
        "steps": steps, "eval_every": eval_every, "velocity_coefficient": 1.,
        "calibration_contexts": calibration_contexts, "calibration_candidates_per_half": calibration_candidates,
        "endpoint_contexts": endpoint_contexts, "endpoint_candidates": endpoint_candidates,
        "reward": "Frozen log exact binned truth / independently sampled initial-model bin probability; pseudocount .5",
        "reference": "Selected source checkpoint; frozen for complete DGPO run",
        "primary": "Paired held-out fixed reward gain at prescribed final step; lo95>0 in every seed",
        "secondary": "Conditional joint TV, independent-half squared error, moment error, corner and marginal diagnostics",
        "new_pretrain_steps": 0, "classifier": None, "weights": "raw_no_ema",
        "scope": "Mode-level approximate ratio, not full neural density ratio; native DGPO with velocity-MSE surrogate"}
    report_path = output / "report.json"
    if report_path.exists():
        if not resume:
            raise ValueError("Experiment already exists; use --resume")
        report = json.loads(report_path.read_text())
        if report["contract"] != contract:
            raise ValueError("Resume must preserve the experiment contract")
        if report["state"] == "completed":
            return report
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = {"state": "prepared", "contract": contract, "arms": {}}
    if "error" in report:
        report.setdefault("previous_failures", []).append(report.pop("error"))
    atomic_json(report_path, report)
    import wandb
    wb = wandb.init(project="dgpo-toy", mode=wandb_mode, dir=str(output.resolve()),
        id=run_id or report.get("wandb", {}).get("id"), resume="allow" if resume else None,
        name=RUN_NAME, group="Conditional cube post-pretraining DGPO", config=contract,
        tags=["toy", "k8", "Fourier-condition", "frozen-mode-ratio", "V-MSE-1", "raw-no-ema"])
    report["wandb"] = {"id": wb.id, "mode": wandb_mode, "directory": wb.dir}
    for seed in seeds:
        wb.define_metric(f"seed{seed}/*", step_metric=f"seed{seed}/step")
    start = time.monotonic(); previous = report.get("seconds", 0.)
    try:
        with (output / "progress.jsonl").open("a" if resume else "w") as log:
            def emit(seed, row):
                log.write(json.dumps({"seed": seed, **row}, allow_nan=False) + "\n"); log.flush()
                print(json.dumps({"seed": seed, **row}, allow_nan=False), flush=True)
                wb.log({f"seed{seed}/step": row["step"],
                    **flatten_metrics(row, f"seed{seed}/" + row["phase"] + "/")})
            for seed in seeds:
                key = str(seed)
                if report["arms"].get(key, {}).get("state") == "completed":
                    continue
                cfg, initial, saved, path = load_source(source, seed, steps, eval_every)
                data = truth_data(cfg)
                directory = output / f"seed{seed}"; directory.mkdir(exist_ok=True)
                arm = report["arms"].setdefault(key, {"state": "calibrating_reward", "source": str(path.resolve()),
                    "source_epoch": saved["epoch"], "source_step": saved["step"], "config": asdict(cfg),
                    "evaluations": {}, "streams": {"calibration_a": 940017 + seed, "calibration_b": 950017 + seed,
                        "native_monitor": 960017 + seed, "structure_monitor": 970017 + seed, "endpoint": 980017 + seed,
                        "policy_seed": seed}})
                report.update(state="calibrating_reward", active_seed=seed); atomic_json(report_path, report)
                reward_path = directory / "frozen_ratio.pt"
                if reward_path.exists():
                    calibration = torch.load(reward_path, map_location="cpu", weights_only=True)
                    if calibration["contract"] != contract or calibration["source"] != arm["source"]:
                        raise ValueError("Frozen ratio provenance mismatch")
                    a, b = calibration["counts_a"], calibration["counts_b"]
                else:
                    a, b = [calibration_counts(initial, cfg, arm["streams"][f"calibration_{half}"],
                        calibration_contexts, calibration_candidates,
                        lambda row, h=half: emit(seed, {**row, "phase": "calibration_" + h, "half": h})) for half in ("a", "b")]
                    atomic_checkpoint(reward_path, {"counts_a": a, "counts_b": b, "source": arm["source"],
                        "source_step": saved["step"], "contract": contract, "pseudocount": .5})
                critic = FrozenModeRatio(a + b).eval().requires_grad_(False)
                before_critic = copy.deepcopy(critic.state_dict())
                base_path = directory / "initial_evaluation.pt"
                if base_path.exists():
                    original = torch.load(base_path, map_location="cpu", weights_only=True)
                    if original["contract"] != contract:
                        raise ValueError("Baseline evaluation provenance mismatch")
                    baseline, initial_metrics = original["panel"], original["metrics"]
                else:
                    baseline, initial_metrics = evaluate(initial, critic, data, cfg, arm["streams"]["endpoint"],
                        endpoint_contexts, endpoint_candidates)
                    atomic_checkpoint(base_path, {"panel": baseline, "metrics": initial_metrics, "contract": contract})
                arm["initial"] = initial_metrics
                arm["preflight"] = reward_preflight(baseline, critic, a, b)
                emit(seed, {"phase": "initial", "step": 0, "metrics": initial_metrics, "preflight": arm["preflight"]})
                report["state"] = arm["state"] = "training_dgpo"
                atomic_json(report_path, report)

                def record_endpoint(step, model):
                    panel, stats = evaluate(model, critic, data, cfg, arm["streams"]["endpoint"],
                        endpoint_contexts, endpoint_candidates)
                    changes = endpoint_changes(panel, stats, baseline, initial_metrics, arm["preflight"])
                    arm["evaluations"][str(step)] = {"metrics": stats, "changes": changes}
                    atomic_checkpoint(directory / f"evaluation_{step}.pt", panel)
                    emit(seed, {"phase": "endpoint", "step": step, "metrics": stats, "changes": changes})

                milestones = {s for s in (300, 1000, steps) if s <= steps}
                state_path = directory / "policy_state.pt"
                def save(step, model, opt, rng, history):
                    if step == 1 or step % eval_every == 0 or step in milestones:
                        # Persist the optimizer transaction BEFORE expensive diagnostics.
                        atomic_checkpoint(state_path, {"model": model.state_dict(), "optimizer": opt.state_dict(),
                            "rng": rng.get_state(), "history": history, "step": step, "config": asdict(cfg),
                            "velocity_coefficient": 1., "source": arm["source"], "seed": seed, "contract": contract})
                        arm["completed_steps"] = step
                        if step in milestones:
                            record_endpoint(step, model)
                        else:
                            _, stats = evaluate(model, critic, data, cfg, arm["streams"]["structure_monitor"], 1024, 8)
                            emit(seed, {"phase": "structure_monitor", "step": step, "metrics": stats})
                        report["seconds"] = previous + time.monotonic() - start
                        atomic_json(report_path, report)
                state = torch.load(state_path, map_location="cpu", weights_only=True) if resume and state_path.exists() else None
                if state is not None:
                    if state["contract"] != contract or state["source"] != arm["source"]:
                        raise ValueError("Policy checkpoint provenance mismatch")
                    if not all(int(v["step"]) == state["step"] for v in state["optimizer"]["state"].values()):
                        raise ValueError("Policy optimizer clock mismatch")
                if state is not None and state["step"] == steps:
                    final = copy.deepcopy(initial).requires_grad_(True); final.load_state_dict(state["model"])
                    history = state["history"]
                else:
                    final, history = policy_train("dgpo", initial, critic, data, cfg, seed,
                        arm["streams"]["native_monitor"], lambda row: emit(seed, row),
                        checkpoint_callback=save, resume_state=state, velocity_coefficient=1.)
                if str(steps) not in arm["evaluations"]:
                    record_endpoint(steps, final)
                arm["source_unchanged"] = all(torch.equal(v, saved["model"][k]) for k, v in initial.state_dict().items())
                arm["reward_unchanged"] = all(torch.equal(v, before_critic[k]) for k, v in critic.state_dict().items())
                if not arm["source_unchanged"] or not arm["reward_unchanged"]:
                    raise RuntimeError("Frozen source or reward mutated")
                arm.update(state="completed", completed_steps=len(history))
                atomic_json(report_path, report)
        report.update(state="completed", decision=decisions(report["arms"], seeds, steps))
    except BaseException as exc:
        report.update(state="failed_or_interrupted", error=repr(exc)); raise
    finally:
        report["seconds"] = previous + time.monotonic() - start
        atomic_json(report_path, report); wb.summary.update(flatten_metrics(report)); wb.summary["state"] = report["state"]
        wb.finish(exit_code=0 if report["state"] == "completed" else 1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/dgpo_toy/cube_truth_fourier_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--wandb-mode", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    run(args.source, args.output, seeds=tuple(args.seeds), steps=args.steps, eval_every=args.eval_every,
        wandb_mode=args.wandb_mode, resume=args.resume, run_id=args.run_id)


if __name__ == "__main__":
    main()
