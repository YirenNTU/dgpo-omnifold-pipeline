"""Matched raw/Fourier condition pretraining on one fixed complete-truth cube dataset.

No classifier, reward, policy update, reference penalty, or EMA. See
CUBE_TRUTH_FOURIER_PROTOCOL.md before interpreting this deliberately k8-aligned toy.
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

from .conditional import Config, Denoiser, ddim, generator
from .coverage_budget import flatten_metrics
from .cube_lockdown import ConditionDenoiser
from .parity_cube import CubeDistribution
from .truth_pretrain import (
    atomic_checkpoint, atomic_json, early_stop_update, make_panel, noisy_target, validation,
)

BASES = ("raw", "fourier")
RUN_NAME = "Can Fourier improve pretraining? | truth cube k=8 | raw vs Fourier"
DEFAULT_CONFIG = replace(
    Config(), dimensions=3, context_dim=1, hidden=128, train_events=131072,
    validation_events=8192, test_events=8192, eval_events=8192,
    candidates=8, ddim_steps=50,
)


def truth_data(cfg):
    # mass=.9 also makes all legacy reference probabilities equal truth here.
    return CubeDistribution(cfg, mass=.9, width=.15, continuous=True, condition_frequency=8)


def prepare_truth_dataset(path, cfg, seed):
    spec = {k: getattr(cfg, k) for k in ("train_events", "validation_events", "test_events")}
    metadata = {
        "schema": "cube_fixed_truth_fourier_v1", "dataset_seed": seed, **spec,
        "condition": "Uniform[-1,1]", "positive_parity_probability": "0.5+0.4*sin(8*pi*c)",
        "width": .15, "truth": True, "condition_frequency": 8,
        "split_seeds": {name: seed + 610000 + i for i, name in enumerate(("train", "validation", "test"))},
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Existing dataset differs from this experiment contract")
        return saved
    data = truth_data(cfg)
    saved = {"metadata": metadata}
    for name, count in zip(("train", "validation", "test"), spec.values()):
        rng = generator(metadata["split_seeds"][name])
        c = data.contexts(count, rng)
        saved[name] = {"condition": c, "target": data.sample(c, rng, truth=True)}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(path, saved)
    atomic_json(path.with_suffix(".json"), metadata)
    return saved


def initial_pair(cfg, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        base = Denoiser(cfg)
        models = {b: ConditionDenoiser(cfg, b, base.state_dict()) for b in BASES}
    rng = generator(seed + 700000)
    c = 2 * torch.rand(16, 1, generator=rng) - 1
    z = torch.randn(16, 2, 3, generator=rng)
    t = torch.rand(16, 2, generator=rng)
    with torch.no_grad():
        velocity_exact = torch.equal(models["raw"](z, t, c[:, None]), models["fourier"](z, t, c[:, None]))
        samples_exact = torch.equal(ddim(models["raw"], c[:, None], z, 2), ddim(models["fourier"], c[:, None], z, 2))
    if not velocity_exact or not samples_exact:
        raise RuntimeError("Initial raw/Fourier functions do not match")
    matching = {"velocity_exact": velocity_exact, "samples_exact": samples_exact,
        "parameter_count": {b: sum(p.numel() for p in m.parameters()) for b, m in models.items()},
        "raw_adapter": "inert zero features; same stored size, not equal effective capacity"}
    return models, matching


def mode_rows(y, c, data):
    """One row per independent condition, averaging its K candidates."""
    if y.ndim != 3 or y.shape[-1] != 3 or c.shape != (len(y), 1):
        raise ValueError("Expected samples [conditions,K,3] and conditions [conditions,1]")
    if not torch.isfinite(y).all():
        raise FloatingPointError("Nonfinite generation")
    ids = ((y[..., 0] >= 0).long() * 4 + (y[..., 1] >= 0).long() * 2 + (y[..., 2] >= 0).long())
    return F.one_hot(ids, 8).double().mean(1), data.probabilities(c[:, 0], .9).double()


def generation_metrics(y, c, data):
    observed, expected = mode_rows(y, c, data)
    bins = data.condition_bins
    if len(c) % bins:
        raise ValueError("Balanced evaluation needs a multiple of 256 conditions")
    # evaluate() supplies an ordered midpoint grid, so each bin has equal contexts.
    grid = ((torch.arange(len(c), dtype=c.dtype) + .5) / len(c) * 2 - 1)[:, None]
    if not torch.equal(c.cpu(), grid):
        raise ValueError("Use the declared balanced midpoint condition grid")
    empirical = observed.reshape(bins, -1, 8).mean(1)
    target = expected.reshape(bins, -1, 8).mean(1)
    positive_modes = data.centers.prod(-1) > 0
    parity_error = (empirical - target)[:, positive_modes].sum(-1)
    signs = torch.where(y >= 0, 1., -1.)
    parity = signs.prod(-1)
    moment = (data.condition_signal(c[:, 0])[:, None] * parity).double().mean()
    flat = y.flatten(0, 1).double()
    centered = flat - flat.mean(0)
    covariance = centered.T @ centered / len(flat)
    offdiag = covariance - covariance.diag().diag()
    residual = flat - signs.flatten(0, 1).double()
    preferred = (parity == torch.where(data.condition_signal(c[:, 0]) >= 0, 1., -1.)[:, None]).double().mean()
    return {
        "conditional_mode_tv": float((empirical - target).abs().sum(-1).mean() / 2),
        "conditional_parity_rmse": float(parity_error.square().mean().sqrt()),
        "condition_weighted_parity": float(moment), "target_condition_weighted_parity": .4,
        "condition_moment_abs_error": abs(float(moment) - .4),
        "preferred_mass": float(preferred),
        "near_corner_fraction": float((residual.norm(dim=-1) < .5).double().mean()),
        "marginal_mean_abs_max": float(flat.mean(0).abs().max()),
        "marginal_variance_abs_error_max": float((covariance.diag() - (1 + data.width ** 2)).abs().max()),
        "offdiagonal_covariance_abs_max": float(offdiag.abs().max()),
        "corner_residual_rms": float(residual.square().mean().sqrt()),
        "max_abs_coordinate": float(flat.abs().max()),
        "conditions": len(c), "samples": flat.shape[0], "condition_bins": bins,
    }


@torch.no_grad()
def evaluate(model, data, cfg, seed):
    c = ((torch.arange(cfg.eval_events, dtype=torch.float32) + .5) / cfg.eval_events * 2 - 1)[:, None]
    rng = generator(seed)
    if model is None:
        # Monte Carlo floor / instrument control, never an input to training.
        y = data.sample(c[:, None].expand(-1, cfg.candidates, -1).reshape(-1, 1), rng, truth=True)
        y = y.reshape(cfg.eval_events, cfg.candidates, 3)
    else:
        model.eval()
        z = torch.randn(cfg.eval_events, cfg.candidates, 3, generator=rng)
        y = torch.cat([ddim(model, cc[:, None], zz, cfg.ddim_steps)
            for cc, zz in zip(c.split(128), z.split(128))])
    return {"samples": y, "condition": c}, generation_metrics(y, c, data)


def paired_tv_comparison(raw, fourier, data, *, replicates=256, seed=930017):
    """Paired, within-bin context bootstrap. Never count K candidates as contexts."""
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    if not torch.equal(raw["condition"], fourier["condition"]):
        raise ValueError("Paired evaluation must use identical conditions")
    r, target = mode_rows(raw["samples"], raw["condition"], data)
    f, _ = mode_rows(fourier["samples"], fourier["condition"], data)
    r, f, target = (x.reshape(data.condition_bins, -1, 8) for x in (r, f, target))
    def gain(rr, ff, tt):
        return ((rr.mean(-2) - tt.mean(-2)).abs().sum(-1) -
                (ff.mean(-2) - tt.mean(-2)).abs().sum(-1)).mean(-1) / 2
    point = float(gain(r, f, target))
    rng = generator(seed)
    draws = []
    bin_ids = torch.arange(data.condition_bins)[None, :, None]
    for start in range(0, replicates, 16):
        ids = torch.randint(r.shape[1], (min(16, replicates - start), data.condition_bins, r.shape[1]), generator=rng)
        draws.append(gain(r[bin_ids, ids], f[bin_ids, ids], target[bin_ids, ids]))
    draws = torch.cat(draws)
    lo, hi = draws.quantile(torch.tensor([.025, .975], dtype=draws.dtype)).tolist()
    return {"raw_minus_fourier_tv": point, "bootstrap_lo95": lo, "bootstrap_hi95": hi,
        "bootstrap_replicates": replicates, "unit": "paired conditions within fixed bins",
        "scope": "sampling uncertainty conditional on these trained models, not training-seed uncertainty"}


def train_epoch(models, optimizers, active, split, cfg, rng):
    totals = {b: {"loss_sum": 0., "max_grad_norm": 0., "clipped_steps": 0, "steps": 0} for b in active}
    for b in active:
        models[b].train()
    order = torch.randperm(len(split["target"]), generator=rng)
    for ids in order.split(cfg.fit_batch):
        c, y = split["condition"][ids], split["target"][ids]
        x, t, target = noisy_target(y, rng)  # Draw once; both arms see EXACTLY this batch.
        for b in active:
            optimizer = optimizers[b]
            loss = (models[b](x, t, c) - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {b} training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(models[b].parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            totals[b]["loss_sum"] += float(loss.detach()) * len(ids)
            totals[b]["max_grad_norm"] = max(totals[b]["max_grad_norm"], norm)
            totals[b]["clipped_steps"] += int(norm > 1.)
            totals[b]["steps"] += 1
    return {b: {**v, "train_mse": v["loss_sum"] / len(order),
        "grad_clip_fraction": v["clipped_steps"] / v["steps"]} for b, v in totals.items()}


def fit_pair(directory, dataset, cfg, seed, *, patience, min_delta, monitor_every,
             emit, resume=False, stop_after_epoch=None):
    directory.mkdir(parents=True, exist_ok=True)
    models, matching = initial_pair(cfg, seed)
    optimizers = {b: torch.optim.AdamW(m.parameters(), lr=cfg.fit_lr, weight_decay=cfg.weight_decay)
                  for b, m in models.items()}
    rng = generator(seed + 620000)
    val = make_panel(dataset["validation"], 630017)
    data = truth_data(cfg)
    arms = {b: {"steps": 0, "epochs": 0, "stale": 0, "anchor": None, "best_mse": None,
                "best_epoch": 0, "best_step": 0, "stopped": False} for b in BASES}
    best = {b: copy.deepcopy(m.state_dict()) for b, m in models.items()}
    epoch = 0
    report = {"seed": seed, "state": "training", "initial_matching": matching, "arms": arms}
    state_path = directory / "last_state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        epoch, best, report = state["epoch"], state["best"], state["report"]
        arms = report["arms"]
        for b in BASES:
            models[b].load_state_dict(state["models"][b])
            optimizers[b].load_state_dict(state["optimizers"][b])
        rng.set_state(state["rng"])
        if report["state"] == "completed":
            return report
        report["state"] = "training"

    def save():
        atomic_checkpoint(state_path, {"epoch": epoch, "best": best, "report": report,
            "models": {b: m.state_dict() for b, m in models.items()},
            "optimizers": {b: o.state_dict() for b, o in optimizers.items()}, "rng": rng.get_state()})
        atomic_json(directory / "report.json", report)

    def endpoints(label):
        panels, result = {}, {}
        for b in BASES:
            selected = copy.deepcopy(models[b]); selected.load_state_dict(best[b]); selected.eval()
            panels[b], stats = evaluate(selected, data, cfg, 880017)
            result[b] = {"generation": stats, "best_epoch": arms[b]["best_epoch"], "best_step": arms[b]["best_step"],
                "available_epochs": arms[b]["epochs"], "available_steps": arms[b]["steps"]}
            atomic_checkpoint(directory / f"{label}_{b}_evaluation.pt", panels[b])
            atomic_checkpoint(directory / f"{label}_{b}_model.pt", {
                "model": best[b], "config": asdict(cfg), "basis": b, "condition_frequency": 8,
                "trained_on": "complete_truth", "seed": seed, "weights": "raw_no_ema",
                "step": arms[b]["best_step"], "epoch": arms[b]["best_epoch"],
                "dataset_metadata": dataset["metadata"], "validation_mse": arms[b]["best_mse"],
            })
            emit({"seed": seed, "basis": b, "phase": label, "epoch": epoch, **result[b]})
        result["comparison"] = paired_tv_comparison(panels["raw"], panels["fourier"], data)
        result["available_epochs"] = epoch
        report[label] = result
        emit({"seed": seed, "basis": "comparison", "phase": label, "epoch": epoch, **result["comparison"]})

    save()  # Resume transaction starts at a complete epoch boundary.
    while not all(a["stopped"] for a in arms.values()):
        epoch += 1
        active = [b for b in BASES if not arms[b]["stopped"]]
        started = time.monotonic()
        training = train_epoch(models, optimizers, active, dataset["train"], cfg, rng)
        train_seconds = time.monotonic() - started
        for b in active:
            models[b].eval()
            v = validation(models[b], val)
            score = v["velocity_mse"]
            arm = arms[b]
            arm["steps"] += training[b]["steps"]; arm["epochs"] = epoch
            if arm["best_mse"] is None or score < arm["best_mse"]:
                arm.update(best_mse=score, best_epoch=epoch, best_step=arm["steps"])
                best[b] = copy.deepcopy(models[b].state_dict())
            anchor = float("inf") if arm["anchor"] is None else arm["anchor"]
            arm["anchor"], arm["stale"] = early_stop_update(score, anchor, arm["stale"], min_delta)
            arm["stopped"] = arm["stale"] >= patience
            row = {"seed": seed, "basis": b, "phase": "pretrain", "epoch": epoch, "step": arm["steps"],
                **training[b], "validation": v, "best_val_mse": arm["best_mse"],
                "validation_time_bins": {str(i): values for i, values in enumerate(v["time_bins"])},
                "stale_epochs": arm["stale"], "early_stopped": arm["stopped"],
                "lr": optimizers[b].param_groups[0]["lr"], "pair_train_seconds": train_seconds}
            if epoch == 1 or epoch % monitor_every == 0:
                tick = time.monotonic()
                row["generation"] = evaluate(models[b], data, replace(cfg, eval_events=1024), 870017)[1]
                row["generation_seconds"] = time.monotonic() - tick
            emit(row)
        if "matched_budget" not in report and any(a["stopped"] for a in arms.values()):
            # At first early stop both arms have exactly the same available update budget.
            endpoints("matched_budget")
        save()
        if stop_after_epoch is not None and epoch >= stop_after_epoch and not all(a["stopped"] for a in arms.values()):
            report["state"] = "paused_test_only"; save(); return report
    endpoints("early_stop")
    for b in BASES:
        selected = copy.deepcopy(models[b]); selected.load_state_dict(best[b]); selected.eval()
        report["early_stop"][b]["test_validation"] = validation(selected, make_panel(dataset["test"], 630018))
    report.update(state="completed", stop_reason="independent_validation_early_stopping"); save()
    return report


def decisions(pairs, required_seeds):
    if len(pairs) != len(required_seeds) or any(p["state"] != "completed" for p in pairs.values()):
        return {"state": "pending", "interpretation": "No completed primary comparison yet"}
    primary = [p["early_stop"]["comparison"] for p in pairs.values()]
    passed = all(x["raw_minus_fourier_tv"] > .05 and x["bootstrap_lo95"] > 0 for x in primary)
    guardrails = all(
        p["early_stop"]["fourier"]["generation"]["near_corner_fraction"] >=
        p["early_stop"]["raw"]["generation"]["near_corner_fraction"] - .02
        for p in pairs.values())
    return {"state": "completed", "primary_material_gain_all_seeds": passed,
        "corner_shape_guardrail_all_seeds": guardrails,
        "matched_budget_material_gain_all_seeds": all(
            p["matched_budget"]["comparison"]["raw_minus_fourier_tv"] > .05 and
            p["matched_budget"]["comparison"]["bootstrap_lo95"] > 0 for p in pairs.values()),
        "decision": "supports" if passed and guardrails else "unresolved",
        "seed_replication": "three_or_more" if len(pairs) >= 3 else "pilot_only",
        "scope": "Known k8 condition basis, one fixed truth dataset; no production or DGPO conclusion"}


def run(output, *, seeds=(17, 23, 41), dataset_seed=17, cfg=DEFAULT_CONFIG,
        patience=20, min_delta=1e-4, monitor_every=10, wandb_mode="offline", run_id=None,
        resume=False, stop_after_epoch=None):
    if (not seeds or len(set(seeds)) != len(seeds) or min(patience, monitor_every, cfg.fit_batch,
        cfg.train_events, cfg.validation_events, cfg.test_events, cfg.candidates, cfg.ddim_steps) < 1
        or cfg.eval_events < 512 or cfg.eval_events % 256 or not math.isfinite(min_delta) or min_delta < 0):
        raise ValueError("Invalid seeds, budgets, evaluation grid, or early-stop settings")
    contract = {"schema": "cube_truth_fourier_pretrain_v1", "seeds": list(seeds), "dataset_seed": dataset_seed,
        "config": asdict(cfg), "patience": patience, "min_delta": min_delta, "monitor_every": monitor_every,
        "max_steps": None, "max_epochs": None, "checkpoint_selection": "minimum validation velocity MSE",
        "primary": "raw minus Fourier held-out conditional eight-mode TV at selected early-stop checkpoints",
        "margin": .05, "shape_guardrail": "near-corner fraction loss <= .02",
        "truth": "p(parity+|c)=.5+.4*sin(8*pi*c); uniform within each parity; Gaussian width .15",
        "training_input": "noisy y, time, raw c, optional sin/cos c at frequencies 1,2,4,8",
        "training_loss": "ordinary velocity MSE; no oracle structure supplied to loss",
        "unused_config_fields": ["pretrain_steps", "policy_steps", "classifier_steps"],
        "source": "random initialization; no previous diffusion checkpoint",
        "weights": "raw_no_ema", "scope": "toy pretraining only; no classifier, DGPO, or KL"}
    existing = output / "report.json"
    if existing.exists():
        if not resume:
            raise ValueError("Output already contains this experiment; use --resume to continue it")
        report = json.loads(existing.read_text())
        if report["contract"] != contract:
            raise ValueError("Resume requires the same experiment contract")
        if report["state"] == "completed":
            return report
    else:
        output.mkdir(parents=True, exist_ok=True)
        report = {"contract": contract, "state": "prepared", "pairs": {}}
    dataset = prepare_truth_dataset(output / "dataset.pt", cfg, dataset_seed)
    report["dataset_metadata"] = dataset["metadata"]
    # Write provenance before initializing optional logging or running a fit.
    atomic_json(existing, report)
    import wandb
    wb = wandb.init(project="dgpo-toy", mode=wandb_mode, dir=str(output.resolve()),
        id=run_id or report.get("wandb", {}).get("id"), resume="allow" if resume else None,
        name=RUN_NAME, group="Conditional cube truth pretraining", config=contract,
        tags=["toy", "k8", "pretraining-only", "raw-no-ema", "shared-truth-data"])
    report["wandb"] = {"id": wb.id, "mode": wandb_mode, "directory": wb.dir}
    for seed in seeds:
        for basis in (*BASES, "comparison"):
            prefix = f"seed{seed}/{basis}"
            wb.define_metric(prefix + "/*", step_metric=prefix + "/epoch")
    started = time.monotonic()
    previous_seconds = report.get("seconds", 0.)
    try:
        report["state"] = "training"
        if "truth_sampling_floor" not in report:
            _, report["truth_sampling_floor"] = evaluate(None, truth_data(cfg), cfg, 880018)
            wb.log(flatten_metrics(report["truth_sampling_floor"], "truth_sampling_floor/"))
        with (output / "progress.jsonl").open("a" if resume else "w") as log:
            def emit(row):
                log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                prefix = f"seed{row['seed']}/{row['basis']}"
                wb.log({prefix + "/epoch": row["epoch"], **flatten_metrics(row, prefix + "/" + row["phase"] + "/")})
            for seed in seeds:
                report["active_seed"] = seed; atomic_json(existing, report)
                report["pairs"][str(seed)] = fit_pair(output / f"seed{seed}", dataset, cfg, seed,
                    patience=patience, min_delta=min_delta, monitor_every=monitor_every, emit=emit,
                    resume=resume, stop_after_epoch=stop_after_epoch)
                atomic_json(existing, report)
                if report["pairs"][str(seed)]["state"] != "completed":
                    report["state"] = "paused_test_only"; break
            else:
                report["state"] = "completed"
            report["decision"] = decisions(report["pairs"], seeds)
    except BaseException as exc:
        report.update(state="failed_or_interrupted", error=repr(exc)); raise
    finally:
        report["seconds"] = previous_seconds + time.monotonic() - started
        atomic_json(existing, report)
        wb.summary.update(flatten_metrics(report))
        wb.summary["state"] = report["state"]
        wb.summary["decision"] = report.get("decision", {}).get("decision", "pending")
        wb.finish(exit_code=1 if report["state"] == "failed_or_interrupted" else 0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--monitor-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--wandb-mode", choices=["offline", "online", "disabled"], default="offline")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    result = run(args.output, seeds=tuple(args.seeds), dataset_seed=args.dataset_seed,
        patience=args.patience, min_delta=args.min_delta, monitor_every=args.monitor_every,
        wandb_mode=args.wandb_mode, run_id=args.run_id, resume=args.resume)
    print(json.dumps({"state": result["state"], "decision": result.get("decision"),
        "report": str((args.output / "report.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
