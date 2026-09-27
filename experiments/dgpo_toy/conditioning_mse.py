"""A moving Gaussian: paired raw/Fourier diffusion, velocity MSE only.

Run as a module. See CONDITIONING_MSE_PROTOCOL.md. The analytic teacher is
evaluation-only: training receives sampled (c, y), never the conditional mean.
"""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from .conditional import Config as LegacyConfig, Denoiser, alpha_sigma, generator
from .truth_pretrain import atomic_checkpoint, atomic_json, noisy_target


CASES = ("linear", "periodic", "bump")
ARMS = ("raw", "fourier")
FREQUENCY_CASES = ("periodic_high", "chirp")
FREQUENCIES = {"raw": (1., 2., 3., 4.), "fourier": (1., 2., 3., 4.),
               "fourier_mid": (2., 4., 6., 8.), "fourier_high": (3., 6., 9., 12.)}
FREQUENCY_ARMS = tuple(FREQUENCIES)


@lru_cache(maxsize=1)
def chirp_normalization():
    # Fixed deterministic quadrature, not a statistic fitted on any data split.
    c = (torch.arange(65536, dtype=torch.float64)+.5)/65536*2-1
    value = torch.sin(math.pi*(8*c+3*c.square()))
    return float(value.mean()), float(value.std(unbiased=False))


@dataclass(frozen=True)
class Config:
    train_events: int = 32768
    validation_events: int = 8192
    test_events: int = 16384
    train_probe_events: int = 8192
    hidden: int = 64
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = .001
    noise_width: float = .35
    patience: int = 20
    min_delta: float = 1e-4
    max_epochs: int | None = None
    dataset_seed: int = 42017
    material_mse: float = .005


def mean_function(c, case):
    """Unit-scale functions under Uniform[-1,1]; chirp uses fixed quadrature."""
    if case == "linear":
        return math.sqrt(3) * c
    if case == "periodic":
        return math.sqrt(2) * torch.sin(4 * math.pi * c)
    if case == "periodic_high":
        return math.sqrt(2) * torch.sin(12 * math.pi * c)
    if case == "chirp":
        center, scale = chirp_normalization()
        return (torch.sin(math.pi*(8*c+3*c.square()))-center)/scale
    if case not in ("bump", "narrow_bump"):
        raise ValueError(f"Unknown conditioning function: {case}")
    center, width = .25, (.05 if case == "narrow_bump" else .15)
    # Analytic first/second moments of exp(-(c-center)^2/(2*width^2)).
    def integral(scale):
        return scale * math.sqrt(math.pi / 2) / 2 * (
            math.erf((1-center) / (math.sqrt(2)*scale))
            - math.erf((-1-center) / (math.sqrt(2)*scale)))
    mu, second = integral(width), integral(width / math.sqrt(2))
    return (torch.exp(-.5*((c-center)/width).square()) - mu) / math.sqrt(second-mu*mu)


def make_dataset(cfg, case):
    """All cases share conditions/noise draws; only f(c) changes."""
    splits = {}
    for i, (name, count) in enumerate((
        ("train", cfg.train_events), ("validation", cfg.validation_events), ("test", cfg.test_events)
    )):
        rng = generator(cfg.dataset_seed + i * 10000)
        c = 2 * torch.rand(count, 1, generator=rng) - 1
        y = mean_function(c, case) + cfg.noise_width * torch.randn(count, 1, generator=rng)
        splits[name] = {"condition": c, "target": y}
    return splits


def condition_features(c, arm):
    if arm not in FREQUENCIES:
        raise ValueError(arm)
    phase = math.pi * c * c.new_tensor(FREQUENCIES[arm])
    if arm == "raw":
        return torch.zeros_like(torch.cat((phase, phase), -1))
    return math.sqrt(2) * torch.cat((phase.sin(), phase.cos()), -1)


class ConditionDenoiser(Denoiser):
    def __init__(self, cfg, arm, base_state):
        super().__init__(LegacyConfig(dimensions=1, context_dim=1, hidden=cfg.hidden))
        self.load_state_dict(base_state)
        self.arm = arm
        self.condition_adapter = nn.Linear(8, cfg.hidden, bias=False)
        nn.init.zeros_(self.condition_adapter.weight)

    def predict_features(self, x, t, c):
        t = t.expand(x.shape[:-1])
        c = c.expand(*x.shape[:-1], 1)
        time_features = torch.stack([t, (math.pi*t).sin(), (math.pi*t).cos(),
                                    (2*math.pi*t).sin(), (2*math.pi*t).cos()], -1)
        # Both raw c and each extra Fourier feature have population variance 1.
        hidden = self.network[0](torch.cat((x, math.sqrt(3)*c, time_features), -1))
        hidden = hidden + self.condition_adapter(condition_features(c, self.arm))
        hidden = self.network[1:-1](hidden)
        return alpha_sigma(t)[1][..., None] * self.network[-1](hidden), hidden


def initial_pair(cfg, seed, arms=ARMS):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        base = Denoiser(LegacyConfig(dimensions=1, context_dim=1, hidden=cfg.hidden))
        models = {arm: ConditionDenoiser(cfg, arm, base.state_dict()) for arm in arms}
    return models


def oracle_velocity(x, t, c, case, noise_width):
    """E[alpha*eps-sigma*y | x_t,c] and conditional irreducible MSE."""
    a, s = alpha_sigma(t[:, None])
    variance = noise_width ** 2
    denominator = a.square()*variance + s.square()
    mu = mean_function(c, case)
    best = -s*mu + a*s*(1-variance)/denominator * (x-a*mu)
    return best, variance / denominator


def make_panel(split, seed, case, cfg):
    x, t, target = noisy_target(split["target"], generator(seed))
    best, floor = oracle_velocity(x, t, split["condition"], case, cfg.noise_width)
    return {"condition": split["condition"], "x": x, "t": t,
            "target": target, "oracle": best, "floor": floor}


@torch.no_grad()
def evaluate(model, panel):
    model.eval()
    predictions = torch.cat([model(panel["x"][ids], panel["t"][ids], panel["condition"][ids])
                             for ids in torch.arange(len(panel["t"])).split(1024)])
    errors = (predictions-panel["target"]).double().square().mean(-1)
    excess = (predictions-panel["oracle"]).double().square().mean(-1)
    if not torch.isfinite(errors).all() or not torch.isfinite(excess).all():
        raise FloatingPointError("Nonfinite evaluation MSE")
    result = {"velocity_mse": float(errors.mean()), "oracle_excess_mse": float(excess.mean()),
              "oracle_empirical_mse": float((panel["oracle"]-panel["target"]).double().square().mean()),
              "irreducible_mse": float(panel["floor"].double().mean())}
    for name, values, lo, hi in (("time", panel["t"], 0., 1.),
                                 ("condition", panel["condition"][:, 0], -1., 1.)):
        result[f"{name}_bins"] = []
        for i in range(10):
            left, right = lo+(hi-lo)*i/10, lo+(hi-lo)*(i+1)/10
            mask = (values >= left) & (values < right)
            result[f"{name}_bins"].append({"lo": left, "hi": right, "count": int(mask.sum()),
                "velocity_mse": float(errors[mask].mean()) if mask.any() else None,
                "oracle_excess_mse": float(excess[mask].mean()) if mask.any() else None})
    return result, errors


def train_epoch(models, optimizers, split, cfg, rng):
    totals = {arm: {"sum": 0., "steps": 0, "clipped": 0, "max_grad_norm": 0.} for arm in models}
    for model in models.values():
        model.train()
    for ids in torch.randperm(len(split["target"]), generator=rng).split(cfg.batch_size):
        c, y = split["condition"][ids], split["target"][ids]
        x, t, target = noisy_target(y, rng)  # Draw once for both arms.
        for arm, model in models.items():
            optimizer = optimizers[arm]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x, t, c)-target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {arm} training loss")
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            entry = totals[arm]
            entry["sum"] += float(loss.detach()) * len(ids)
            entry["steps"] += 1
            entry["clipped"] += int(norm > 1.)
            entry["max_grad_norm"] = max(entry["max_grad_norm"], norm)
    return {arm: {"online_train_mse": value["sum"]/len(split["target"]),
                  "grad_clip_fraction": value["clipped"]/value["steps"],
                  "max_grad_norm": value["max_grad_norm"]} for arm, value in totals.items()}


def paired_comparison(raw, fourier, margin):
    delta = fourier-raw
    mean = float(delta.mean())
    se = float(delta.std(unbiased=True)/math.sqrt(len(delta)))
    lo, hi = mean-1.96*se, mean+1.96*se
    label = "fourier_better" if hi < -margin else "fourier_worse" if lo > margin else "unresolved"
    return {"fourier_minus_raw_mse": mean, "paired_se": se, "lo95": lo, "hi95": hi,
            "material_margin": margin, "decision": label,
            "scope": "pointwise paired test-event normal interval; conditional on trained models"}


def fit_pair(output, dataset, cfg, case, seed, emit, arms=ARMS):
    output.mkdir(parents=True, exist_ok=True)
    models = initial_pair(cfg, seed, arms)
    opts = {a: torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for a, m in models.items()}
    train_probe = {k: v[:cfg.train_probe_events] for k, v in dataset["train"].items()}
    panels = {"train_probe": make_panel(train_probe, 71017, case, cfg),
              "validation": make_panel(dataset["validation"], 72017, case, cfg)}
    with torch.no_grad():
        p = panels["validation"]
        predictions = [m(p["x"], p["t"], p["condition"]) for m in models.values()]
    if not all(torch.equal(predictions[0], other) for other in predictions[1:]):
        raise RuntimeError("Step-zero velocities differ; paired experiment invalid")
    best = {a: copy.deepcopy(m.state_dict()) for a, m in models.items()}
    best_metrics = {a: evaluate(m, panels["validation"])[0] for a, m in models.items()}
    best_losses = {a: metric["velocity_mse"] for a, metric in best_metrics.items()}
    anchors, stale, best_epochs = best_losses.copy(), dict.fromkeys(arms, 0), dict.fromkeys(arms, 0)
    first_plateau = dict.fromkeys(arms, None)
    history = []
    rng = generator(seed+73000)
    epoch = 0
    started = time.perf_counter()
    while True:
        training = train_epoch(models, opts, dataset["train"], cfg, rng) if epoch else {}
        for arm, model in models.items():
            metrics = {key: evaluate(model, panel)[0] for key, panel in panels.items()}
            score = metrics["validation"]["velocity_mse"]
            if epoch:
                if score < best_losses[arm]:
                    best_losses[arm], best_epochs[arm] = score, epoch
                    best[arm] = copy.deepcopy(model.state_dict())
                    best_metrics[arm] = metrics["validation"]
                if score < anchors[arm]-cfg.min_delta:
                    anchors[arm], stale[arm] = score, 0
                else:
                    stale[arm] += 1
                if stale[arm] >= cfg.patience and first_plateau[arm] is None:
                    first_plateau[arm] = {"epoch": epoch, "validation_mse": score,
                        "best_validation_mse": best_losses[arm],
                        "best_oracle_excess_mse": best_metrics[arm]["oracle_excess_mse"],
                        "oracle_excess_mse": metrics["validation"]["oracle_excess_mse"]}
            row = {"case": case, "seed": seed, "arm": arm, "epoch": epoch,
                   "step": epoch*math.ceil(cfg.train_events/cfg.batch_size), "lr": cfg.lr,
                   "stale_epochs": stale[arm], "best_epoch": best_epochs[arm],
                   **training.get(arm, {}), **metrics}
            history.append(row)
            emit(row)
        if all(value >= cfg.patience for value in stale.values()):
            stop_reason = "paired_validation_early_stop"
            break
        if cfg.max_epochs is not None and epoch >= cfg.max_epochs:
            stop_reason = "budget_limit_not_convergence"
            break
        epoch += 1
    endpoint = {}
    test = make_panel(dataset["test"], 74017, case, cfg)  # Never consulted by selection/stopping.
    errors = {}
    for arm, model in models.items():
        atomic_checkpoint(output/f"last_{arm}.pt", {"model": model.state_dict(), "epoch": epoch,
            "optimizer": opts[arm].state_dict(), "config": asdict(cfg), "case": case, "seed": seed,
            "weights": "raw_no_ema", "arm": arm,
            "frequencies": list(FREQUENCIES[arm]) if arm != "raw" else []})
        model.load_state_dict(best[arm])
        metrics, errors[arm] = evaluate(model, test)
        endpoint[arm] = {"selected_epoch": best_epochs[arm], "validation_mse": best_losses[arm],
                         "train_probe": evaluate(model, panels["train_probe"])[0], "test": metrics}
        atomic_checkpoint(output/f"best_{arm}.pt", {"model": best[arm], "config": asdict(cfg),
            "case": case, "seed": seed, "epoch": best_epochs[arm], "weights": "raw_no_ema",
            "arm": arm, "frequencies": list(FREQUENCIES[arm]) if arm != "raw" else [],
            "selection": "minimum validation velocity MSE, including step zero"})
    comparisons = {}
    for baseline in ("raw", "fourier"):
        for candidate in arms:
            if candidate in ("raw", baseline):
                continue
            comparison = paired_comparison(errors[baseline], errors[candidate], cfg.material_mse)
            delta = comparison.pop("fourier_minus_raw_mse")
            comparison["decision"] = comparison["decision"].replace("fourier_", "candidate_")
            comparisons[f"{candidate}_minus_{baseline}"] = {
                **comparison, "candidate_minus_baseline_mse": delta,
                "candidate": candidate, "baseline": baseline}
    report = {"case": case, "seed": seed, "epochs": epoch, "stop_reason": stop_reason,
              "initial_velocity_exact": True, "endpoint": endpoint, "history": history,
              "first_patience_plateau": first_plateau, "comparisons": comparisons,
              "comparison": paired_comparison(errors["raw"], errors["fourier"], cfg.material_mse),
              "seconds": time.perf_counter()-started}
    atomic_json(output/"report.json", report)
    return report


def validate_config(cfg):
    for name in ("train_events", "validation_events", "test_events", "train_probe_events"):
        if getattr(cfg, name) < 2:
            raise ValueError(f"{name} must be at least 2")
    for name in ("hidden", "batch_size", "patience"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg.max_epochs is not None and cfg.max_epochs < 1:
        raise ValueError("max_epochs must be positive or omitted")
    for name in ("lr", "noise_width", "material_mse"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("min_delta", "weight_decay"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def plot_design(path, cfg, cases=CASES):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = torch.linspace(-1, 1, 1000)
    fig, axes = plt.subplots(1, len(cases), figsize=(4*len(cases), 3.2), sharey=True,
                             squeeze=False, constrained_layout=True)
    axes = axes[0]
    titles = {"linear": "Linear: center follows the dial", "periodic": "Periodic: center oscillates",
              "bump": "Local bump: one special region", "periodic_high": "Fast wave: k=12",
              "chirp": "Chirp: oscillations become faster"}
    for ax, case in zip(axes, cases):
        mu = mean_function(c, case)
        ax.plot(c.numpy(), mu.numpy(), color="#137c8b", lw=2)
        ax.fill_between(c.numpy(), (mu-cfg.noise_width).numpy(), (mu+cfg.noise_width).numpy(),
                        color="#137c8b", alpha=.2)
        ax.set(title=titles[case], xlabel="Observed condition c", xlim=(-1, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Target y: mean and +/- one std")
    fig.suptitle(f"Design only — y | c ~ Normal(f(c), {cfg.noise_width}²); no training results")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_results(path, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases, seeds = report["cases"], report["seeds"]
    fig, axes = plt.subplots(2, len(cases), figsize=(4*len(cases), 6), squeeze=False, constrained_layout=True)
    for col, case in enumerate(cases):
        for seed in seeds:
            pair = report["pairs"][f"{case}/{seed}"]
            colors = {"raw": "#555555", "fourier": "#137c8b",
                      "fourier_mid": "#d98520", "fourier_high": "#974fb5"}
            for arm in report.get("arms", ARMS):
                color = colors[arm]
                rows = [r for r in pair["history"] if r["arm"] == arm]
                for ax, split in zip(axes[:, col], ("train_probe", "validation")):
                    ax.plot([r["step"] for r in rows], [r[split]["velocity_mse"] for r in rows],
                            color=color, alpha=.75, label=arm if seed == seeds[0] else None)
                    ax.set(xlabel="Optimizer steps", ylabel=f"{split} velocity MSE", title=case)
                    ax.grid(alpha=.2)
        axes[0, col].legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(output, cfg, cases=CASES, seeds=(17, 23, 41), arms=ARMS):
    validate_config(cfg)
    if not cases or not set(cases) <= set(CASES+FREQUENCY_CASES) or len(set(cases)) != len(cases):
        raise ValueError("Choose distinct supported cases")
    if tuple(arms) not in (ARMS, FREQUENCY_ARMS):
        raise ValueError("Use the raw/Fourier pair or the declared frequency ablation")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct seeds")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"report.json").exists() or (output/"progress.jsonl").exists():
        raise ValueError("Existing experiment output; choose a new --output (no data deleted)")
    report = {"state": "prepared", "config": asdict(cfg), "cases": list(cases), "seeds": list(seeds),
        "arms": list(arms), "frequency_banks": {a: list(FREQUENCIES[a]) if a != "raw" else [] for a in arms},
        "primary_contrast": "fourier_high_minus_fourier" if tuple(arms) == FREQUENCY_ARMS else "fourier_minus_raw",
        "primary": "independent test velocity MSE at best validation checkpoint; common training budget",
        "features": "raw c retained; sqrt(2)*sin/cos(k*pi*c); see frequency_banks; zero-init additive projection",
        "truth_input": False, "oracle_used_in_training": False, "ema": False,
        "classifier": False, "dgpo": False, "scheduler": "constant learning rate in both arms",
        "capacity_caveat": "same stored parameter count, but raw adapter is inert; active capacity differs",
        "pairs": {}}
    atomic_json(output/"report.json", report)
    plot_design(output/"design.png", cfg, cases)
    started = time.perf_counter()
    with (output/"progress.jsonl").open("w") as stream:
        def emit(row):
            stream.write(json.dumps(row, allow_nan=False)+"\n")
            stream.flush()
            print(f"{row['case']} seed={row['seed']} {row['arm']:7s} epoch={row['epoch']} "
                  f"train_probe={row['train_probe']['velocity_mse']:.6f} "
                  f"val={row['validation']['velocity_mse']:.6f} "
                  f"val_excess={row['validation']['oracle_excess_mse']:.6f}", flush=True)
        for case in cases:
            dataset = make_dataset(cfg, case)
            atomic_checkpoint(output/f"{case}_dataset.pt", dataset)
            for seed in seeds:
                report["state"] = "training"
                atomic_json(output/"report.json", report)
                pair = fit_pair(output/case/str(seed), dataset, cfg, case, seed, emit, arms)
                report["pairs"][f"{case}/{seed}"] = pair
                atomic_json(output/"report.json", report)
    report["state"] = "completed"
    report["seconds"] = time.perf_counter()-started
    report["decisions"] = {}
    for case in cases:
        pairs = [report["pairs"][f"{case}/{seed}"] for seed in seeds]
        primary_key = report["primary_contrast"]
        labels = ([p["comparisons"][primary_key]["decision"] for p in pairs]
                  if tuple(arms) == FREQUENCY_ARMS else [p["comparison"]["decision"] for p in pairs])
        report["decisions"][case] = {
            "contrast": primary_key,
            "at_matched_budget": labels[0] if len(set(labels)) == 1 else "seed_dependent",
            "replication": "multi_seed_same_dataset" if len(seeds) >= 3 else "pilot_only",
            "both_arms_plateaued": all(p["stop_reason"] == "paired_validation_early_stop" for p in pairs),
            "scope": "This one-dimensional toy only; no unique bottleneck or EveNet attribution"}
        report["decisions"][case]["frequency_comparisons"] = {
            key: {"at_matched_budget": values[0] if len(set(values)) == 1 else "seed_dependent"}
            for key in pairs[0]["comparisons"]
            for values in [[p["comparisons"][key]["decision"] for p in pairs]]}
        if tuple(arms) == FREQUENCY_ARMS:
            report["decisions"][case]["rescue_checks_by_seed"] = {}
            for pair in pairs:
                plateau = pair["first_patience_plateau"]["fourier"]
                endpoint = pair["endpoint"]
                checks = {
                    "low_bank_patience_plateau_with_excess_gt_001": bool(
                        plateau and plateau["best_oracle_excess_mse"] > .01),
                    "high_bank_material_test_gain": pair["comparisons"][primary_key]["decision"] == "candidate_better",
                    "high_bank_lower_train_probe_mse": endpoint["fourier_high"]["train_probe"]["velocity_mse"]
                        < endpoint["fourier"]["train_probe"]["velocity_mse"]}
                checks["all_checks_pass"] = all(checks.values())
                report["decisions"][case]["rescue_checks_by_seed"][str(pair["seed"])] = checks
    atomic_json(output/"report.json", report)
    plot_results(output/"mse_curves.png", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dgpo_toy/conditioning_mse_v1"))
    parser.add_argument("--cases", nargs="+", choices=CASES+FREQUENCY_CASES)
    parser.add_argument("--frequency-ablation", action="store_true",
                        help="Raw and three equal-size frequency banks; defaults to fast wave and chirp")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 41])
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--min-delta", type=float, default=Config.min_delta)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--preview-only", action="store_true", help="Draw the data-generating design; no training")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cfg = Config(lr=args.lr, patience=args.patience, min_delta=args.min_delta, max_epochs=args.max_epochs)
    validate_config(cfg)
    cases = tuple(args.cases or (FREQUENCY_CASES if args.frequency_ablation else CASES))
    arms = FREQUENCY_ARMS if args.frequency_ablation else ARMS
    if args.preview_only:
        plot_design(args.output/"design.png", cfg, cases)
        print(args.output/"design.png")
        return
    report = run(args.output, cfg, cases, tuple(args.seeds), arms)
    print(json.dumps(report["decisions"], indent=2))


if __name__ == "__main__":
    main()
