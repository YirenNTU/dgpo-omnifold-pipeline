"""Pretrained moving-Gaussian diffusion: no Fourier vs early vs late adapter.

Local CPU only. See CONDITIONING_PLACEMENT_PROTOCOL.md. No scientific run is
started on import. Analytic velocity is used only for readiness/evaluation.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import torch
from torch import nn

from . import conditioning_mse as core

ARMS = ("none", "early", "late")


@dataclass(frozen=True)
class Config:
    training: core.Config = field(default_factory=core.Config)
    case: str = "periodic"
    headroom_min: float = .01
    headroom_max: float = .05
    required_val_reduction: float = .5


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, width))

    def forward(self, x):
        return x+self.net(x)


class VelocityModel(nn.Module):
    def __init__(self, width, placement="none"):
        super().__init__()
        if placement not in ARMS:
            raise ValueError(placement)
        self.placement = placement
        self.input = nn.Sequential(nn.Linear(7, width), nn.GELU())
        self.backbone = nn.Sequential(Block(width), Block(width))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                  nn.SiLU(), nn.Linear(width, 1))
        self.adapter = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.LayerNorm(width),
                                     nn.Linear(width, width, bias=False))
        # Only the final adapter projection is zero. Internal layers can learn
        # after the first step; double-zero initialization would kill gradients.
        nn.init.zeros_(self.adapter[-1].weight)
        if placement == "none":
            self.adapter.requires_grad_(False)

    def representations(self, x, t, c):
        time = torch.stack((t, (math.pi*t).sin(), (math.pi*t).cos(),
                            (2*math.pi*t).sin(), (2*math.pi*t).cos()), -1)
        token = self.input(torch.cat((x, math.sqrt(3)*c, time), -1))
        delta = self.adapter(core.condition_features(c, "fourier"))
        early = token + delta if self.placement == "early" else token
        body = self.backbone(early)
        head_input = body + delta if self.placement == "late" else body
        injection_base = token if self.placement == "early" else body
        if self.placement == "none":
            delta = torch.zeros_like(delta)
        return body, head_input, delta, injection_base

    def forward(self, x, t, c):
        _, h, _, _ = self.representations(x, t, c)
        return core.alpha_sigma(t)[1][:, None]*self.head(h)


def initialize(cfg, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return VelocityModel(cfg.training.hidden)


def fork_models(source):
    result = {arm: copy.deepcopy(source) for arm in ARMS}
    for arm, model in result.items():
        model.placement = arm
        model.adapter.requires_grad_(arm != "none")
    return result


def readiness(initial_mse, metrics, cfg):
    excess = metrics["oracle_excess_mse"]
    return {"learned_signal": metrics["velocity_mse"] <= initial_mse*(1-cfg.required_val_reduction),
            "headroom_remaining": excess >= cfg.headroom_min,
            "baseline_accurate_enough": excess <= cfg.headroom_max}


@torch.no_grad()
def representation_metrics(model, source, panel):
    totals = dict(body_delta=0., body_ref=0., residual=0., base=0.)
    for ids in torch.arange(len(panel["t"])).split(1024):
        args = (panel["x"][ids], panel["t"][ids], panel["condition"][ids])
        body, _, residual, base = model.representations(*args)
        reference = source.representations(*args)[0]
        totals["body_delta"] += float((body-reference).double().square().sum())
        totals["body_ref"] += float(reference.double().square().sum())
        totals["residual"] += float(residual.double().square().sum())
        totals["base"] += float(base.double().square().sum())
    return {"body_relative_rms_change": math.sqrt(totals["body_delta"]/(totals["body_ref"]+1e-20)),
            "residual_to_token_rms": math.sqrt(totals["residual"]/(totals["base"]+1e-20))}


def make_optimizer(model, cfg):
    # All arms reset AdamW identically, with the same LR for all active weights.
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=cfg.lr, weight_decay=cfg.weight_decay)


def pretrain(output, data, cfg, seed, emit):
    model = initialize(cfg, seed)
    opt = make_optimizer(model, cfg.training)
    panel = core.make_panel(data["validation"], 72017, cfg.case, cfg.training)
    initial = core.evaluate(model, panel)[0]
    anchor, best_score = initial["velocity_mse"], initial["velocity_mse"]
    best_state = copy.deepcopy(model.state_dict())
    best_metrics, best_epoch, stale, epoch = initial, 0, 0, 0
    rng = core.generator(seed+81000)
    while True:
        checks = readiness(initial["velocity_mse"], best_metrics, cfg)
        if all(checks.values()):
            status = "ready"
            break
        if stale >= cfg.training.patience or (
            cfg.training.max_epochs is not None and epoch >= cfg.training.max_epochs
        ):
            status = "source_not_ready"
            break
        epoch += 1
        online = core.train_epoch({"none": model}, {"none": opt}, data["train"], cfg.training, rng)
        metrics = core.evaluate(model, panel)[0]
        score = metrics["velocity_mse"]
        if score < best_score:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            best_state = copy.deepcopy(model.state_dict())
        if score < anchor-cfg.training.min_delta:
            anchor, stale = score, 0
        else:
            stale += 1
        emit({"phase": "pretrain", "arm": "none", "seed": seed, "epoch": epoch,
              "validation": metrics, **online["none"]})
    model.load_state_dict(best_state)
    report = {"status": status, "trained_epochs": epoch, "selected_epoch": best_epoch,
              "initial_validation": initial, "selected_validation": best_metrics,
              "readiness": checks, "source_case": cfg.case}
    core.atomic_checkpoint(output/"source.pt", {"model": best_state, "config": asdict(cfg),
        "seed": seed, "weights": "raw_no_ema", "report": report})
    return model, report


def comparison(baseline, candidate, margin):
    result = core.paired_comparison(baseline, candidate, margin)
    result["candidate_minus_baseline_mse"] = result.pop("fourier_minus_raw_mse")
    result["decision"] = result["decision"].replace("fourier_", "candidate_")
    return result


def continue_matched(output, data, cfg, seed, source, emit):
    models = fork_models(source)
    opts = {a: make_optimizer(m, cfg.training) for a, m in models.items()}
    probe = {k: v[:cfg.training.train_probe_events] for k, v in data["train"].items()}
    panels = {"train_probe": core.make_panel(probe, 71017, cfg.case, cfg.training),
              "validation": core.make_panel(data["validation"], 72017, cfg.case, cfg.training)}
    p = panels["validation"]
    with torch.no_grad():
        predictions = [m(p["x"], p["t"], p["condition"]) for m in models.values()]
    if not all(torch.equal(predictions[0], x) for x in predictions[1:]):
        raise RuntimeError("Matched source outputs differ")
    initial = core.evaluate(source, p)[0]
    best = {a: copy.deepcopy(m.state_dict()) for a, m in models.items()}
    best_scores = dict.fromkeys(ARMS, initial["velocity_mse"])
    anchors, stale, best_epochs = best_scores.copy(), dict.fromkeys(ARMS, 0), dict.fromkeys(ARMS, 0)
    rng = core.generator(seed+82000)
    history, epoch = [], 0
    while True:
        online = core.train_epoch(models, opts, data["train"], cfg.training, rng) if epoch else {}
        for arm, model in models.items():
            metrics = {key: core.evaluate(model, panel)[0] for key, panel in panels.items()}
            score = metrics["validation"]["velocity_mse"]
            if epoch:
                if score < best_scores[arm]:
                    best_scores[arm], best_epochs[arm] = score, epoch
                    best[arm] = copy.deepcopy(model.state_dict())
                if score < anchors[arm]-cfg.training.min_delta:
                    anchors[arm], stale[arm] = score, 0
                else:
                    stale[arm] += 1
            row = {"phase": "continue", "seed": seed, "arm": arm, "epoch": epoch,
                "step": epoch*math.ceil(cfg.training.train_events/cfg.training.batch_size),
                "best_epoch": best_epochs[arm], "stale_epochs": stale[arm],
                **online.get(arm, {}), **metrics,
                "representation": representation_metrics(model, source, panels["train_probe"])}
            history.append(row)
            emit(row)
        if all(v >= cfg.training.patience for v in stale.values()):
            reason = "joint_validation_early_stop"
            break
        if cfg.training.max_epochs is not None and epoch >= cfg.training.max_epochs:
            reason = "budget_limit_not_convergence"
            break
        epoch += 1
    test = core.make_panel(data["test"], 74017, cfg.case, cfg.training)
    source_test, _ = core.evaluate(source, test)
    endpoints, errors = {}, {}
    for arm, model in models.items():
        core.atomic_checkpoint(output/f"last_{arm}.pt", {"model": model.state_dict(),
            "optimizer": opts[arm].state_dict(), "epoch": epoch, "placement": arm,
            "config": asdict(cfg), "seed": seed, "weights": "raw_no_ema"})
        model.load_state_dict(best[arm])
        metrics, errors[arm] = core.evaluate(model, test)
        endpoints[arm] = {"selected_epoch": best_epochs[arm], "validation_mse": best_scores[arm],
            "test": metrics, "train_probe": core.evaluate(model, panels["train_probe"])[0],
            "representation": representation_metrics(model, source, panels["train_probe"])}
        core.atomic_checkpoint(output/f"best_{arm}.pt", {"model": best[arm], "epoch": best_epochs[arm],
            "placement": arm, "config": asdict(cfg), "seed": seed, "weights": "raw_no_ema"})
    contrasts = {f"{candidate}_minus_{baseline}": comparison(errors[baseline], errors[candidate], cfg.training.material_mse)
                for baseline, candidate in (("none", "early"), ("none", "late"), ("early", "late"))}
    # Never call a nonsignificant difference evidence of preservation.
    margin = cfg.training.material_mse
    checks = {"early_materially_worse_than_none": contrasts["early_minus_none"]["lo95"] > margin,
              "late_noninferior_to_none": contrasts["late_minus_none"]["hi95"] < margin,
              "late_materially_better_than_early": contrasts["late_minus_early"]["hi95"] < -margin}
    return {"initial_velocity_exact": True, "initial_validation": initial, "source_test": source_test,
            "epochs": epoch, "stop_reason": reason, "endpoint": endpoints, "comparisons": contrasts,
            "interference_checks": checks, "supports_early_interference_at_tested_protocol": all(checks.values()),
            "history": history}


def run(output, cfg, seeds=(17,)):
    core.validate_config(cfg.training)
    if cfg.case not in core.CASES+core.FREQUENCY_CASES:
        raise ValueError("Unsupported function")
    if not (0 < cfg.headroom_min < cfg.headroom_max and 0 < cfg.required_val_reduction < 1):
        raise ValueError("Invalid source readiness thresholds")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Distinct seeds required")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"report.json").exists():
        raise ValueError("Existing experiment output; choose another --output")
    data = core.make_dataset(cfg.training, cfg.case)
    core.atomic_checkpoint(output/"dataset.pt", data)
    report = {"state": "running", "config": asdict(cfg), "seeds": list(seeds), "results": {},
        "primary": "late minus early independent test velocity MSE, validated against continued no-Fourier control",
        "frequency_bank": [1, 2, 3, 4], "same_distribution_both_stages": True,
        "optimizer_at_fork": "fresh AdamW in all arms; same LR for every active parameter",
        "oracle_usage": "validation readiness gate and diagnostics only; never training inputs or targets",
        "scope": "MLP placement toy, not attention-specific proof; one seed is a pilot"}
    core.atomic_json(output/"report.json", report)
    with (output/"progress.jsonl").open("w") as stream:
        def emit(row):
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            print(f"{row['phase']} seed={row['seed']} {row['arm']} epoch={row['epoch']} "
                  f"val={row['validation']['velocity_mse']:.6f} "
                  f"excess={row['validation']['oracle_excess_mse']:.6f}", flush=True)
        for seed in seeds:
            directory = output/str(seed)
            directory.mkdir(exist_ok=True)
            source, pre = pretrain(directory, data, cfg, seed, emit)
            result = {"pretrain": pre}
            report["results"][str(seed)] = result
            core.atomic_json(output/"report.json", report)
            if pre["status"] == "ready":
                result["continuation"] = continue_matched(directory, data, cfg, seed, source, emit)
            core.atomic_json(output/"report.json", report)
    report["state"] = ("completed" if all("continuation" in v for v in report["results"].values())
                       else "inconclusive_source_not_ready")
    core.atomic_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dgpo_toy/conditioning_placement_pilot1"))
    parser.add_argument("--case", choices=core.CASES+core.FREQUENCY_CASES, default="periodic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--max-epochs", type=int, help="Optional per-stage safety cap; not convergence")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = Config(training=core.Config(max_epochs=args.max_epochs), case=args.case)
    result = run(args.output, cfg, tuple(args.seeds))
    print(json.dumps({"state": result["state"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
