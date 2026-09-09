#!/usr/bin/env python3
"""Classifier-only initialization ablation on a saved, immutable replay pool.

No policy loading/generation, DGPO, OmniFold unfolding, W&B, or production writes.
Uses the production classifier, balanced loss, sampler, gradient synchronization,
clipping and AdamW. Only initialization differs between the three arms.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_raw_monitor_replay as replay

ARMS = ("A_cold", "B_old_features", "C_full_warm")
FEATURE_PREFIXES = {
    "grouped_sequential_embedding": "_backbone.GroupedSequentialEmbedding.",
    "invisible_projector": "_backbone.InvisibleInputProjector.",
    "pet_adapters": "_backbone.PET.adapters.",
}


def file_digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_read(path):
    with Path(path).open() as f:
        return json.load(f)


def tensor_read(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def clone_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def checked_state_pair(cold, old):
    if set(cold) != set(old):
        raise ValueError("Old monitor and cold classifier state keys differ")
    for k, v in cold.items():
        if v.shape != old[k].shape or v.dtype != old[k].dtype:
            raise ValueError(f"Old monitor shape/dtype mismatch: {k}")
        if not torch.isfinite(v).all() or not torch.isfinite(old[k]).all():
            raise ValueError(f"Nonfinite initialization: {k}")
    for group, prefix in FEATURE_PREFIXES.items():
        if not any(k.startswith(prefix) for k in cold):
            raise ValueError(f"No trainable saved tensors for {group}")


def arm_initial_state(arm, cold, old):
    """B inherits ONLY the three named body modules; decoder/position/readout stay cold."""
    checked_state_pair(cold, old)
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    prefixes = tuple(FEATURE_PREFIXES.values())
    return clone_state({k: old[k] if arm == "C_full_warm" or (
        arm == "B_old_features" and k.startswith(prefixes)) else v
        for k, v in cold.items()})


def split_pool(pool, provenance, saved_validation, *, minimum):
    """Train on the OLD training fold, not the complement of the common validation."""
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, _identity_crossfit_splits
    conditions, protocols = [], []
    for label in ("old", "new"):
        meta = provenance[label]
        protocol = meta["split_protocol"]
        if protocol.get("schema") != "raw-monitor-condition-split-v1" or protocol.get("folds") != 5:
            raise ValueError("Unsupported source split protocol")
        if type(protocol.get("seed")) is not int:
            raise ValueError("Invalid source split seed")
        conditions.append(replay.repack_pool(pool, EventPackingSpec.from_dict(meta["packing_spec"])))
        protocols.append(protocol)
    validation, counts = replay.shared_validation_indices(conditions, protocols, minimum=minimum)
    if saved_validation.dtype != torch.int64 or not torch.equal(validation.cpu(), saved_validation.cpu()):
        raise ValueError("Saved validation identities differ from recomputed intersection")
    train, _ = _identity_crossfit_splits(conditions[0], folds=5, seed=protocols[0]["seed"])[0]
    if torch.isin(train, validation).any():
        raise ValueError("Training/validation overlap")
    return {"train": train.cpu(), "validation": validation.cpu()}, counts


def validate_settings(settings):
    cfg = dict(settings)
    for key in ("workers", "steps", "seed", "batch_size", "microbatch_size", "score_batch_size", "minimum_validation_events"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["batch_size"] % cfg["workers"]:
        raise ValueError("Global batch_size must be divisible by workers")
    for key in ("learning_rate", "gradient_clip_norm", "positive_control_auc_tolerance"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(float(cfg["weight_decay"])) or float(cfg["weight_decay"]) < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    points = cfg["evaluation_steps"]
    if (not isinstance(points, list) or not points or any(type(p) is not int for p in points)
            or points != sorted(set(points)) or points[0] != 0 or points[-1] != cfg["steps"]
            or any(p < 0 or p > cfg["steps"] for p in points)):
        raise ValueError("evaluation_steps must be unique/sorted and include 0 and steps")
    return cfg


def prepare(settings):
    """Read-only CPU preflight; --check-only neither creates output nor starts Ray."""
    import yaml
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldPool
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec

    cfg = validate_settings(settings)
    source = Path(cfg["replay_dir"]).expanduser().resolve(strict=True)
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a NEW output directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output must be separate from the replay directory")
    paths = {name: str((source / filename).resolve(strict=True)) for name, filename in {
        "manifest": "manifest.json", "report": "report.json", "pool": "pool.pt",
        "scores": "scores.pt", "runtime": "old_runtime.yaml",
    }.items()}
    manifest, report = json_read(paths["manifest"]), json_read(paths["report"])
    if not report.get("policy_loaded_verified") or not report.get("policy_unchanged"):
        raise ValueError("Source replay did not verify its fixed policy")
    if report.get("policy_updates") != 0 or report.get("classifier_fits") != 0:
        raise ValueError("Expected an evaluation-only replay")
    provenance = manifest["provenance"]
    paths["old_checkpoint"] = str(Path(provenance["old"]["checkpoint"]).resolve(strict=True))
    # Hash protected inputs BEFORE reading them, and check again after preflight.
    fingerprints = {p: file_digest(p) for p in paths.values()}
    checkpoint = replay.read_checkpoint(paths["old_checkpoint"])
    if checkpoint.get("global_step") != 260:
        raise ValueError("Expected the verified step-260 old checkpoint")
    policy_hash = replay.state_digest(checkpoint["state_dict"])
    if policy_hash != manifest["expected_policy_state_sha256"]:
        raise ValueError("Old checkpoint policy differs from the verified replay")
    cache, old_spec, reward = replay.monitor_payload(checkpoint)
    old_hash = replay.state_digest(cache["state"])
    if (old_hash != provenance["old"]["monitor_state_sha256"]
            or old_hash != report["metrics"]["old"]["monitor_state_sha256"]
            or old_spec != provenance["old"]["packing_spec"]
            or cache["protocol"] != provenance["old"]["split_protocol"]
            or reward["base_digest"] != provenance["old"]["legacy_base_digest"]
            or checkpoint["dgpo_omnifold_reward_stack"]["source_bundle_sha256"] != provenance["old"]["source_bundle_sha256"]):
        raise ValueError("Old monitor or source provenance changed after replay")
    del checkpoint, cache, reward
    raw_pool = tensor_read(paths["pool"])
    if raw_pool.get("schema_version") != 1 or raw_pool["policy_state_sha256"] != report["loaded_policy_sha256"]:
        raise ValueError("Cached pool policy identity differs from verified replay")
    for key in ("packed_event", "truth", "raw"):
        t = raw_pool[key]
        if not isinstance(t, torch.Tensor) or not torch.isfinite(t).all():
            raise ValueError(f"Invalid/nonfinite pool {key}")
    n = len(raw_pool["packed_event"])
    if (n != report["pool_events"] or n != manifest["pool_events"]
            or raw_pool["truth"].ndim != 2 or raw_pool["raw"].shape != (n, 1, raw_pool["truth"].shape[-1])):
        raise ValueError("Pool must contain aligned truth and exactly one raw candidate per event")
    pool = AdaptiveOmniFoldPool(packed_event=raw_pool["packed_event"], truth=raw_pool["truth"],
                               candidates=raw_pool["raw"], packing_spec=EventPackingSpec.from_dict(raw_pool["packing_spec"]))
    splits, counts = split_pool(pool, provenance, tensor_read(paths["scores"])["validation_indices"],
                               minimum=cfg["minimum_validation_events"])
    if len(splits["train"]) < cfg["batch_size"]:
        raise ValueError("Training fold is smaller than one drop-last global batch")
    if len(splits["validation"]) != report["common_validation_events"]:
        raise ValueError("Replay validation count changed")
    with Path(paths["runtime"]).open() as f:
        runtime = yaml.safe_load(f)
    paths["backbone"] = str(Path(runtime["reward_config"]["omnifold"]["backbone_checkpoint"]).resolve(strict=True))
    paths["normalization"] = str(Path(runtime["options"]["Dataset"]["normalization_file"]).resolve(strict=True))
    for key in ("backbone", "normalization"):
        fingerprints[paths[key]] = file_digest(paths[key])
    if fingerprints[paths["backbone"]] != report["metrics"]["old"]["backbone_file_sha256"]:
        raise ValueError("Classifier backbone file changed after replay")
    ema = runtime["options"]["Training"].get("EMA", {})
    if ema.get("replace_model_after_load", False):
        raise ValueError("Expected live classifier backbone state_dict, not EMA substitution")
    for p in (Path(paths["old_checkpoint"]).parent, Path(paths["backbone"]).parent):
        if output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError("Output cannot overlap checkpoint directories")
    cfg.update(replay_dir=str(source), output_dir=str(output), paths=paths,
               source_fingerprints=fingerprints, provenance=provenance,
               expected_old_auc=report["metrics"]["old"]["auc"], policy_state_sha256=policy_hash,
               loaded_policy_sha256=report["loaded_policy_sha256"], pool_events=n,
               training_events=len(splits["train"]), validation_events=len(splits["validation"]),
               original_validation_counts=dict(zip(("old", "new"), counts)))
    verify_sources(cfg)
    return cfg, splits


def verify_sources(cfg):
    for path, expected in cfg["source_fingerprints"].items():
        if file_digest(path) != expected:
            raise ValueError(f"Protected source changed: {path}")


def create_output(cfg, splits):
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    replay._exclusive_torch_save(output / "split.pt", splits)
    cfg["split_file_sha256"] = file_digest(output / "split.pt")
    replay._exclusive_json(output / "manifest.json", cfg)


@contextmanager
def seeded(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def parameter_group(name):
    for group, prefix in FEATURE_PREFIXES.items():
        if name.startswith(prefix.replace("_backbone.", "backbone.", 1)):
            return group
    if name.startswith("bank."):
        return "decoder_position_readout"
    raise ValueError(f"Unexpected trainable parameter: {name}")


class UpdateTracker:
    """Observe gradients AFTER distributed averaging/clipping; never change them."""
    def __init__(self, model):
        self.parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
        self.groups = {k: parameter_group(k) for k in self.parameters}
        if not set(FEATURE_PREFIXES).issubset(set(self.groups.values())):
            raise ValueError("All three feature groups must be trainable and exposed to AdamW")
        self.previous = {k: p.detach().clone() for k, p in self.parameters.items()}

    @torch.no_grad()
    def observe(self, *, record):
        out = {}
        if record:
            for group in sorted(set(self.groups.values())):
                names = [k for k in self.parameters if self.groups[k] == group]
                before2 = sum(self.previous[k].double().square().sum() for k in names)
                delta2 = sum((self.parameters[k].detach().double() - self.previous[k].double()).square().sum() for k in names)
                grads = [self.parameters[k].grad for k in names if self.parameters[k].grad is not None]
                grad2 = sum((g.detach().double().square().sum() for g in grads), before2.new_zeros(()))
                before, delta = float(before2.sqrt()), float(delta2.sqrt())
                out[group] = {"parameter_count": sum(self.parameters[k].numel() for k in names),
                              "gradient_present_count": sum(g.numel() for g in grads),
                              "gradient_nonzero_count": sum(int(torch.count_nonzero(g)) for g in grads),
                              "gradient_l2_after_clip": float(grad2.sqrt()),
                              "parameter_l2_before": before, "single_step_update_l2": delta,
                              "single_step_relative_update": delta / max(before, 1e-12)}
        for k, p in self.parameters.items():
            self.previous[k].copy_(p.detach())
        return out


def frozen_state(model):
    return {**{f"parameter:{k}": p for k, p in model.backbone.named_parameters() if not p.requires_grad},
            **{f"buffer:{k}": b for k, b in model.backbone.named_buffers()}}


def run_arm(model, arm, data, cfg, *, rank=0):
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig, fit_density_ratio, _broadcast_module
    _broadcast_module(model)
    initial = clone_state(model.state_dict())
    frozen_before = replay.state_digest(frozen_state(model))
    tracker = UpdateTracker(model)
    out = Path(cfg["output_dir"]) / arm
    if rank == 0:
        out.mkdir(exist_ok=False)
        replay._exclusive_torch_save(out / "initial_monitor.pt", initial)
    evaluation_steps = set(cfg["evaluation_steps"])
    rows = []

    def evaluate(step, progress=None, updates=None):
        was_training = model.training
        try:
            metrics, scores = replay.score_monitor(model, data["validation_condition"], data["validation_truth"],
                                                   data["validation_raw"], cfg["score_batch_size"])
        finally:
            model.train(was_training)
        row = {"arm": arm, "step": step, **metrics, "training": progress or {}, "module_updates": updates or {}}
        rows.append(row)
        if rank == 0:
            replay._exclusive_json(out / f"step_{step:04d}.json", row)
            replay._exclusive_torch_save(out / f"scores_{step:04d}.pt", scores)
            print(f"[monitor-ablation] {arm} step={step}/{cfg['steps']} AUC={metrics['auc']:.6f} "
                  f"BCE={metrics['bce']:.6f} balanced_acc={metrics['balanced_accuracy']:.6f}", flush=True)

    evaluate(0)

    def progress(row):
        step = int(row["step"])
        updates = tracker.observe(record=step in evaluation_steps)
        if step in evaluation_steps:
            evaluate(step, row, updates)
        elif rank == 0 and step % 10 == 0:
            print(f"[monitor-ablation] {arm} step={step}/{cfg['steps']} train_loss={row['training_loss']:.6f}", flush=True)

    fit_config = RatioFitConfig(steps=cfg["steps"], batch_size=cfg["batch_size"],
                               train_microbatch_size_per_rank=cfg["microbatch_size"], drop_last_batch=True,
                               learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                               gradient_clip_norm=cfg["gradient_clip_norm"], sampling="independent_epoch_shuffle",
                               restore_best=False, require_saturation=False, progress_interval_steps=1)
    # Explicit reset at fit entry: model construction/transplant and previous arms
    # cannot consume the training dropout RNG. The production sampler has its own seed.
    with seeded(cfg["seed"], next(model.parameters()).device):
        diagnostics = fit_density_ratio(model, data["train_condition"], data["train_truth"],
                                        torch.ones_like(data["train_truth"][:, 0]),
                                        data["train_condition"], data["train_raw"],
                                        torch.ones_like(data["train_raw"][:, :, 0]),
                                        fit_config, cfg["seed"], progress_callback=progress)
    if diagnostics.steps_completed != cfg["steps"] or diagnostics.saturated or diagnostics.threshold_reached:
        raise RuntimeError("Arm did not complete the identical fixed update budget")
    if replay.state_digest(frozen_state(model)) != frozen_before:
        raise RuntimeError("Frozen classifier backbone changed during the ablation")
    result = {"arm": arm, "initial": rows[0], "final": rows[-1], "evaluations": rows,
              "steps_completed": diagnostics.steps_completed, "fit_config": asdict(fit_config),
              "best_sampled_auc_step": max(rows, key=lambda r: r["auc"])["step"],
              "best_sampled_bce_step": min(rows, key=lambda r: r["bce"])["step"],
              "frozen_backbone_unchanged": True}
    if rank == 0:
        replay._exclusive_torch_save(out / "final_monitor.pt", clone_state(model.state_dict()))
        replay._exclusive_json(out / "report.json", result)
    return result


def compact_summary(results, cfg):
    return {
        "steps_per_arm": cfg["steps"], "training_events": cfg["training_events"],
        "validation_events": cfg["validation_events"], "policy_updates": 0, "policy_generations": 0,
        "metric_direction": "Higher AUC = stronger classifier on the same fixed policy, not worse DGPO performance",
        "arms": {arm: {"initial_auc": r["initial"]["auc"], "final_auc": r["final"]["auc"],
                       "final_bce": r["final"]["bce"], "final_balanced_accuracy": r["final"]["balanced_accuracy"],
                       "best_sampled_auc": max(v["auc"] for v in r["evaluations"]),
                       "best_sampled_auc_step": r["best_sampled_auc_step"]} for arm, r in results.items()},
        "matched_step_comparison": [
            {"step": a["step"], "A_auc": a["auc"], "B_auc": b["auc"], "C_auc": c["auc"],
             "B_minus_A_auc": b["auc"] - a["auc"], "C_minus_A_auc": c["auc"] - a["auc"]}
            for a, b, c in zip(*(results[arm]["evaluations"] for arm in ARMS), strict=True)],
    }


def ablation_worker(cfg):
    import ray.train
    import ray.train.torch
    from RL.DGPO_neutrino.model_utils import load_training_config, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterModelBuilder, EventPackingSpec
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldPool
    from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import payload_sha256
    ctx = ray.train.get_context()
    rank, device = ctx.get_world_rank(), ray.train.torch.get_device()
    if ctx.get_world_size() != cfg["workers"]:
        raise ValueError("Ray worker count differs from configured global batch geometry")
    if rank == 0:
        verify_sources(cfg)
    split_path = Path(cfg["output_dir"]) / "split.pt"
    if file_digest(split_path) != cfg["split_file_sha256"]:
        raise ValueError("Preflight split changed")
    splits = tensor_read(split_path)
    raw = tensor_read(cfg["paths"]["pool"])
    pool = AdaptiveOmniFoldPool(packed_event=raw["packed_event"], truth=raw["truth"], candidates=raw["raw"],
                               packing_spec=EventPackingSpec.from_dict(raw["packing_spec"]))
    spec = EventPackingSpec.from_dict(cfg["provenance"]["old"]["packing_spec"])
    condition = replay.repack_pool(pool, spec)
    train, val = splits["train"], splits["validation"]
    data = {"train_condition": condition[train].to(device), "train_truth": pool.truth[train].to(device),
            "train_raw": pool.candidates[train].to(device), "validation_condition": condition[val].to(device),
            "validation_truth": pool.truth[val].to(device), "validation_raw": pool.candidates[val].to(device)}
    del pool, raw, condition
    config = load_training_config(cfg["paths"]["runtime"])
    config.reward_config.omnifold.backbone_checkpoint = cfg["paths"]["backbone"]
    config.options.Dataset.normalization_file = cfg["paths"]["normalization"]
    torch.set_float32_matmul_precision(str(config.dgpo.get("float32_matmul_precision", "medium")))
    kwargs = {k: v for k, v in dict(config.dgpo.adaptive_omnifold.recalibration).items()
              if k in inspect.signature(EvenetAdapterModelBuilder).parameters}
    kwargs.update({k: v for k, v in dict(config.dgpo.adaptive_omnifold.audit_fit).items() if k in replay.ARCH_KEYS})
    if any(kwargs.get(k, False) for k in ("periodic_pair_features", "topology_fourier_embedding", "topology_conditioning")):
        raise ValueError("Initialization ablation requires the clean raw monitor architecture")
    if not kwargs.get("train_grouped_sequential_embedding") or not kwargs.get("train_invisible_projector") or kwargs.get("train_backbone"):
        raise ValueError("Expected trainable input projectors/adapters and frozen PET core")
    with seeded(cfg["seed"], device):
        builder = EvenetAdapterModelBuilder(config=config, normalization_dict=load_normalization_dict(config),
                                            checkpoint_path=cfg["paths"]["backbone"], device=device, **kwargs)
    meta = cfg["provenance"]["old"]
    identity = {"schema_version": 2, "kind": "ztautau_in_dgpo_omnifold_bootstrap",
                "base_digest": builder.base_digest, "policy_reference_sha256": cfg["source_fingerprints"][cfg["paths"]["backbone"]]}
    if builder.base_digest != meta["legacy_base_digest"] or payload_sha256(identity) != meta["source_bundle_sha256"]:
        raise ValueError("Classifier backbone does not match saved monitor provenance")
    template_hash = replay.state_digest(builder._pretrained_body)
    checkpoint = replay.read_checkpoint(cfg["paths"]["old_checkpoint"])
    old = clone_state(replay.monitor_payload(checkpoint)[0]["state"])
    del checkpoint
    with seeded(cfg["seed"], device):
        model = builder.make_classifier(spec, reset=True)
    cold = clone_state(model.state_dict())
    checked_state_pair(cold, old)
    replay.strict_monitor_load(model, old)
    control, _ = replay.score_monitor(model, data["validation_condition"], data["validation_truth"],
                                      data["validation_raw"], cfg["score_batch_size"])
    if abs(control["auc"] - cfg["expected_old_auc"]) > cfg["positive_control_auc_tolerance"]:
        raise ValueError("Old monitor no longer reproduces replay AUC; stop before training")
    if rank == 0:
        replay._exclusive_json(Path(cfg["output_dir"]) / "positive_control.json", control)
        print(f"[monitor-ablation] positive control AUC={control['auc']:.6f}; "
              f"train={len(train)} validation={len(val)}; no policy generation/updates", flush=True)
    del model
    results = {}
    for arm in ARMS:
        with seeded(cfg["seed"], device):
            model = builder.make_classifier(spec, reset=True)
        replay.strict_monitor_load(model, arm_initial_state(arm, cold, old))
        results[arm] = run_arm(model, arm, data, cfg, rank=rank)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if replay.state_digest(builder._pretrained_body) != template_hash:
        raise RuntimeError("Cold classifier template was mutated")
    if rank == 0:
        verify_sources(cfg)
        report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "policy_updates": 0, "policy_generations": 0, "classifier_fits": 3,
                  "protected_sources_unchanged": True, "training_events": len(train), "validation_events": len(val),
                  "positive_control": control, "arms": results,
                  "limitations": [f"Single seed and a fixed {cfg['steps']}-update quick screen are not proof of convergence.",
                                  "Validation identities were used historically for model selection, not an untouched test.",
                                  "B resets decoder, slot-position encoder and readout; failure does not isolate which of those matters.",
                                  "Module gradients are measured after distributed averaging and clipping; updates also include AdamW decay."]}
        replay._exclusive_json(Path(cfg["output_dir"]) / "report.json", report)
        replay._exclusive_json(Path(cfg["output_dir"]) / "summary.json", compact_summary(results, cfg))
        print(f"[monitor-ablation] report: {cfg['output_dir']}/report.json", flush=True)
        print(f"[monitor-ablation] compact results: {cfg['output_dir']}/summary.json", flush=True)
    ray.train.report({f"{arm}/final_auc": result["final"]["auc"] for arm, result in results.items()})


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Use another NEW output directory")
    parser.add_argument("--check-only", action="store_true", help="Read-only CPU preflight; no output/Ray/GPU")
    args = parser.parse_args()
    with args.config.open() as f:
        settings = yaml.safe_load(f)
    if args.output_dir:
        settings["output_dir"] = str(args.output_dir)
    cfg, splits = prepare(settings)
    print(f"[monitor-ablation] preflight OK: train={cfg['training_events']} validation={cfg['validation_events']} "
          f"steps={cfg['steps']} workers={cfg['workers']}", flush=True)
    if args.check_only:
        return 0
    create_output(cfg, splits)
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", runtime_env={"env_vars": {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]),
    }})
    TorchTrainer(train_loop_per_worker=ablation_worker, train_loop_config=cfg,
                 scaling_config=ScalingConfig(num_workers=cfg["workers"], use_gpu=True,
                                             resources_per_worker={"CPU": 2, "GPU": 1}),
                 run_config=RunConfig(name="raw-monitor-init-ablation", storage_path=str(Path(cfg["output_dir"]) / "ray_results"),
                                      failure_config=FailureConfig(max_failures=0))).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
