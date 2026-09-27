"""Fixed full-truth dataset, ordinary velocity targets, validation early stopping."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from .conditional import Config, Distribution, Denoiser, alpha_sigma, generator, initialize
from .structure_metrics import structure_panel, structure_target, paired_structure_change


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def prepare_dataset(path, cfg, seed):
    """Truth formulas are used ONLY here to produce observed (condition, target)."""
    spec = {k: getattr(cfg, k) for k in ("dimensions", "context_dim", "kappa", "structured_mass",
                                      "train_events", "validation_events", "test_events")}
    metadata = {"schema": "fixed_complete_truth_v1", "seed": seed, "distribution": spec,
                "split_seeds": {"train": seed+40000, "validation": seed+50000, "test": seed+60000}}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError("Dataset configuration differs; use a different dataset path")
        return saved
    data = Distribution(cfg)
    saved = {"metadata": metadata}
    for name, count in (("train", cfg.train_events), ("validation", cfg.validation_events), ("test", cfg.test_events)):
        if count < 2:
            raise ValueError("Each split needs at least two events")
        rng = generator(metadata["split_seeds"][name])
        c = data.contexts(count, rng)
        y = data.sample(c, rng, truth=True)
        saved[name] = {"condition": c, "target": y}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint(path, saved)
    atomic_json(path.with_suffix(".json"), {**metadata,
        "columns": ["condition", "target"], "target_distribution": "complete_truth",
        "splits": {name: len(saved[name]["target"]) for name in ("train", "validation", "test")}})
    return saved


def noisy_target(y, rng):
    t = torch.rand(len(y), generator=rng)
    eps = torch.randn(y.shape, generator=rng)
    a, s = alpha_sigma(t[:, None])
    return a*y+s*eps, t, a*eps-s*y


def make_panel(split, seed):
    x, t, target = noisy_target(split["target"], generator(seed))
    return split["condition"], x, t, target


def sample_loss(model, batch):
    c, x, t, target = batch
    return (model(x, t, c)-target).square().mean()


@torch.no_grad()
def validation(model, panel):
    c, x, t, target = panel
    errors = torch.cat([(model(xx, tt, cc)-vv).square().mean(-1)
        for cc, xx, tt, vv in zip(c.split(512), x.split(512), t.split(512), target.split(512))])
    if not torch.isfinite(errors).all():
        raise FloatingPointError("Nonfinite validation error")
    result = {"velocity_mse": float(errors.double().mean()), "events": len(c), "time_bins": []}
    for i in range(10):
        values = errors[(t >= i/10) & (t < (i+1)/10)]
        result["time_bins"].append({"lo": i/10, "hi": (i+1)/10, "count": len(values),
            "mse": float(values.double().mean()) if len(values) else None})
    return result


def early_stop_update(value, anchor, stale, min_delta):
    """Significant improvements can accumulate; best checkpoint tracked separately."""
    if value < anchor-min_delta:
        return value, 0
    return anchor, stale+1


def run(output, *, dataset_path, cfg=None, seed=17, patience=20, min_delta=1e-4,
        structure_every=10, resume_from=None, stop_after_epoch=None):
    cfg = cfg or Config()
    if patience < 1 or not math.isfinite(min_delta) or min_delta < 0 or min(cfg.fit_batch, cfg.eval_events, structure_every) < 1:
        raise ValueError("Positive batch/patience/interval and finite nonnegative min_delta required")
    if resume_from is not None and output.resolve() == resume_from.parent.resolve():
        raise ValueError("Resume into a separate output to preserve the previous run")
    dataset = prepare_dataset(dataset_path, cfg, seed)
    data = Distribution(cfg)  # Only generation diagnostics use known truth structure.
    model = initialize(Denoiser, cfg, seed)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.fit_lr, weight_decay=cfg.weight_decay)
    rng = generator(seed+1000)
    contract = {"experiment": "fixed_truth_sample_velocity_early_stop_v1", "seed": seed,
        "patience_epochs": patience, "min_delta_absolute": min_delta, "structure_every": structure_every,
        "weights": "raw_no_ema", "selection": "minimum validation velocity_mse",
        "dataset": str(dataset_path.resolve())}
    report = {**contract, "config": asdict(cfg), "state": "training", "completed_steps": 0,
        "completed_epochs": 0, "max_steps": None, "max_epochs": None,
        "unused_legacy_config_budgets": ["pretrain_steps", "classifier_steps", "policy_steps"],
        "teacher": None, "training_target": "alpha(t)*epsilon-sigma(t)*truth_sample",
        "input": "raw noisy target, condition, existing time embedding only",
        "dataset_metadata": dataset["metadata"],
        "stream_seeds": {"train_noise_and_shuffle": seed+1000, "validation_noise": seed+21000,
            "generation_monitor": seed+91000, "endpoint": seed+92000, "test_noise": seed+22000},
        "history": [], "best_epoch": 0, "best_step": 0, "best_validation_mse": None,
        "initial_state_origin": "random; no Gaussian checkpoint"}
    epoch, step, stale, anchor = 0, 0, 0, float("inf")
    best_model, best_loss = copy.deepcopy(initial_state), float("inf")
    if resume_from is not None:
        state = torch.load(resume_from, map_location="cpu", weights_only=True)
        if state["config"] != asdict(cfg) or state["contract"] != contract:
            raise ValueError("Resume must preserve dataset, optimizer, architecture and stopping contract")
        if state["report"]["state"] == "completed":
            raise ValueError("This checkpoint already reached early stopping")
        epoch, step, stale, anchor = state["epoch"], state["step"], state["stale_epochs"], state["anchor"]
        if any(int(x["step"]) != step for x in state["optimizer"]["state"].values()):
            raise ValueError("Optimizer clock mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["rng"])
        initial_state, best_model, best_loss = state["initial"], state["best_model"], state["best_loss"]
        report = state["report"]
        report.update(state="training", resumed_from=str(resume_from.resolve()))
    output.mkdir(parents=True, exist_ok=True)
    panel = make_panel(dataset["validation"], seed+21000)
    if not epoch:
        report["initial_validation"] = validation(model, panel)
        report["initial_generation"] = structure_panel(model, data, cfg, seed+91000)[1]
    started, previous_seconds = time.perf_counter(), report.get("seconds", 0.)

    def save():
        report.update(completed_steps=step, completed_epochs=epoch, stale_epochs=stale,
                      seconds=previous_seconds+time.perf_counter()-started)
        atomic_checkpoint(output/"last_state.pt", {"contract": contract, "config": asdict(cfg),
            "epoch": epoch, "step": step, "stale_epochs": stale, "anchor": anchor,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.get_state(),
            "initial": initial_state, "best_model": best_model, "best_loss": best_loss, "report": report})
        atomic_json(output/"report.json", report)

    def export_best():
        atomic_checkpoint(output/"best_model.pt", {"config": asdict(cfg), "model": best_model,
            "step": report["best_step"], "epoch": report["best_epoch"], "seed": seed,
            "weights": "raw", "trained_on": "fixed_complete_truth_dataset",
            "selection": contract["selection"], "validation_mse": best_loss})

    with (output/"progress.jsonl").open("w") as log:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            print(line, flush=True)
            log.write(line+"\n")
            log.flush()
        emit({"phase": "start", "epoch": epoch, "step": step,
              "parameters": sum(p.numel() for p in model.parameters()), **contract})
        save()  # Resumable even if interrupted during the first epoch.
        try:
            while stale < patience:
                epoch += 1
                order = torch.randperm(len(dataset["train"]["target"]), generator=rng)
                total_loss, max_grad, clipped = 0., 0., 0
                for ids in order.split(cfg.fit_batch):
                    c, y = dataset["train"]["condition"][ids], dataset["train"]["target"][ids]
                    x, t, target = noisy_target(y, rng)
                    loss = sample_loss(model, (c, x, t, target))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                    optimizer.step()
                    step += 1
                    total_loss += float(loss.detach())*len(ids)
                    max_grad, clipped = max(max_grad, norm), clipped+int(norm > 1.)
                val = validation(model, panel)
                value = val["velocity_mse"]
                if value < best_loss:
                    best_loss, best_model = value, copy.deepcopy(model.state_dict())
                    report.update(best_epoch=epoch, best_step=step, best_validation_mse=value)
                    export_best()
                anchor, stale = early_stop_update(value, anchor, stale, min_delta)
                row = {"phase": "epoch", "epoch": epoch, "step": step,
                    "train_velocity_mse": total_loss/len(order), "validation": val,
                    "max_gradient_norm": max_grad, "clipped_updates": clipped, "lr": cfg.fit_lr,
                    "stale_epochs": stale, "best_epoch": report["best_epoch"]}
                report["history"].append(row)
                emit(row)
                if epoch % structure_every == 0 or stale >= patience:
                    summary = structure_panel(model, data, cfg, seed+91000)[1]
                    row = {"phase": "generation_monitor", "epoch": epoch, "step": step,
                        **{k:v for k,v in summary.items() if k not in ("moment_mean", "truth_moment")}}
                    report["history"].append(row)
                    emit(row)
                save()
                if epoch == stop_after_epoch and stale < patience:
                    report["state"] = "paused"
                    save()
                    return report
            report.update(state="evaluating", stop_reason="validation_early_stopping")
            save()
            initial = initialize(Denoiser, cfg, seed).eval()
            initial.load_state_dict(initial_state)
            selected = initialize(Denoiser, cfg, seed).eval()
            selected.load_state_dict(best_model)
            # Test split is first used here, AFTER stopping and model selection.
            test = make_panel(dataset["test"], seed+22000)
            panels, endpoints = {}, {}
            for name, m in (("initial", initial), ("last", model), ("best_validation", selected)):
                panels[name], structure = structure_panel(m, data, cfg, seed+92000)
                endpoints[name] = {"step": 0 if name == "initial" else step if name == "last" else report["best_step"],
                    "test_loss": validation(m, test), "structure": structure}
            for name in ("last", "best_validation"):
                endpoints[name]["structure_change"] = paired_structure_change(
                    panels[name], panels["initial"], structure_target(data), seed+92001)
            report.update(state="completed", endpoints=endpoints,
                scope="Validation-loss early stopping only. No classifier, DGPO, teacher or forced residual.")
            atomic_checkpoint(output/"last_model.pt", {"config": asdict(cfg), "model": model.state_dict(),
                "step": step, "epoch": epoch, "seed": seed, "weights": "raw",
                "trained_on": "fixed_complete_truth_dataset"})
            export_best()
            save()
            emit({"phase": "complete", "epoch": epoch, "step": step, "best_epoch": report["best_epoch"],
                  "seconds": report["seconds"], "selected_structure": endpoints["best_validation"]["structure_change"]})
            return report
        except BaseException as exc:
            # Keep last complete epoch's model/optimizer/RNG, not a partial update.
            atomic_json(output/"error.json", {"type": type(exc).__name__, "message": str(exc),
                "attempted_epoch": epoch, "step": step, "resume": "last_state.pt: last complete saved epoch"})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--structure-every", type=int, default=10)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output, dataset_path=args.dataset, seed=args.seed, patience=args.patience,
        min_delta=args.min_delta, structure_every=args.structure_every, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
