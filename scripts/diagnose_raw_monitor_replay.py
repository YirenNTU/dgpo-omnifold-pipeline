#!/usr/bin/env python3
"""Read-only, paired replay of two saved raw monitors on one fixed policy pool.

No DGPO loop, optimizer, classifier fitting, EMA substitution, or W&B run.
Uses production Parquet preprocessing, DDIM generation, event hashing and scoring.
Only writes a new diagnostic directory; existing checkpoints are never written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

ARCH_KEYS = (
    "head_dropout", "decoder_hidden_dim", "decoder_layers", "decoder_heads",
    "periodic_pair_features", "topology_fourier_embedding", "topology_conditioning",
    "topology_dropout",
)


def state_digest(state):
    """Same complete tensor digest used in the user's step-260 comparison."""
    if not state:
        raise ValueError("Empty tensor state")
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor state entry: {name}")
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(repr(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read_checkpoint(path):
    # Only explicitly supplied, trusted local training checkpoints are accepted.
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def monitor_payload(checkpoint):
    cache = (checkpoint.get("dgpo_adaptive_omnifold_state") or {}).get("raw_monitor_state") or {}
    state, protocol = cache.get("state"), cache.get("protocol") or {}
    if not state:
        raise ValueError("Checkpoint has no saved raw monitor; a pre-monitor bootstrap is insufficient")
    if (protocol.get("schema") != "raw-monitor-condition-split-v1"
            or type(protocol.get("seed")) is not int or protocol.get("folds") != 5):
        raise ValueError("Unknown monitor split protocol; refusing a potentially contaminated validation split")
    reward = (checkpoint.get("dgpo_omnifold_reward_stack") or {}).get("reward") or {}
    increments = reward.get("increments") or []
    if not increments or not increments[0].get("packing_spec") or not reward.get("base_digest"):
        raise ValueError("Missing reward packing/base provenance needed to reconstruct monitor inputs")
    spec = increments[0]["packing_spec"]
    readiness = cache.get("training_policy") or {}
    width = sum(math.prod(shape) for shape in spec["shapes"].values())
    if readiness and readiness.get("condition_width") != width:
        raise ValueError("Saved monitor width disagrees with reward packing; do not infer its split")
    return cache, spec, reward


def verify_policy_loaded(model, saved_state):
    """Catch silent safe_load_state skips before sampling the supposed policy."""
    expected = {k.removeprefix("model."): v for k, v in saved_state.items()}
    actual = model.state_dict()
    missing = sorted(set(actual) - set(expected))
    extra = sorted(k for k in set(expected) - set(actual) if not k.startswith("famo.w."))
    different = [k for k in actual if k in expected and (
        actual[k].shape != expected[k].shape or actual[k].dtype != expected[k].dtype
        or not torch.equal(actual[k].detach().cpu(), expected[k].detach().cpu()))]
    if missing or extra or different:
        raise ValueError(f"Loaded policy differs from state_dict: missing={missing[:8]} "
                         f"extra={extra[:8]} different={different[:8]}")
    return state_digest(actual)


def strict_monitor_load(model, state):
    current = model.state_dict()
    if set(current) != set(state):
        raise ValueError(f"Monitor key mismatch: missing={sorted(set(current)-set(state))[:8]} "
                         f"extra={sorted(set(state)-set(current))[:8]}")
    for k, v in state.items():
        if current[k].shape != v.shape or current[k].dtype != v.dtype:
            raise ValueError(f"Monitor shape/dtype mismatch: {k}")
        if not torch.isfinite(v).all():
            raise ValueError(f"Nonfinite saved monitor tensor: {k}")
    model.load_state_dict(state, strict=True)
    model.eval()
    if state_digest(model.state_dict()) != state_digest(state):
        raise ValueError("Monitor tensors changed during load")


def repack_pool(pool, spec):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs, unpack_event_inputs
    fields = unpack_event_inputs(pool.packed_event, pool.packing_spec)
    packed, _ = pack_event_inputs(fields, spec=spec)
    return packed


def shared_validation_indices(conditions, protocols, *, minimum):
    """Intersection, NOT union: neither saved classifier trained on these rows."""
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _identity_crossfit_splits
    n = len(conditions[0])
    keep = torch.ones(n, dtype=torch.bool, device=conditions[0].device)
    counts = []
    for condition, protocol in zip(conditions, protocols, strict=True):
        if len(condition) != n:
            raise ValueError("Monitor populations are not aligned")
        _, val = _identity_crossfit_splits(condition, folds=protocol["folds"], seed=protocol["seed"])[0]
        mask = torch.zeros_like(keep)
        mask[val] = True
        keep &= mask
        counts.append(len(val))
    indices = torch.where(keep)[0]
    if len(indices) < minimum:
        raise ValueError(f"Only {len(indices)} common validation identities; require {minimum}. "
                         "Increase pool_events, never broaden the validation split.")
    return indices, counts


def score_monitor(model, condition, truth, candidates, batch_size):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        _score_population, _weighted_binary_score_metrics,
    )
    model.eval()
    before = state_digest(model.state_dict())
    data_score = _score_population(model, condition, truth, batch_size).reshape(-1)
    gen_score = _score_population(model, condition, candidates, batch_size).reshape(-1)
    loss, acc, auc = _weighted_binary_score_metrics(data_score, gen_score, torch.ones_like(gen_score))
    if state_digest(model.state_dict()) != before:
        raise RuntimeError("Monitor state mutated during evaluation")
    return {"auc": auc, "auc_gap": abs(auc - 0.5), "balanced_accuracy": acc,
            "bce": loss, "events_per_class": len(data_score), "monitor_state_sha256": before}, {
                "truth_logits": data_score.detach().cpu(), "raw_logits": gen_score.detach().cpu(),
            }


def _exclusive_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _exclusive_torch_save(path, value):
    with Path(path).open("xb") as f:
        torch.save(value, f)


def prepare(settings, *, root=ROOT):
    """Preflight on CPU before allocating workers or generating any data."""
    import yaml
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    cfg = dict(settings)
    for key in ("policy_checkpoint", "old_checkpoint", "new_checkpoint", "base_config", "old_overlay", "new_overlay"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((p if p.is_absolute() else root / p).resolve(strict=True))
    for key in ("workers", "pool_events", "generation_batch_size", "score_batch_size", "minimum_validation_events"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    payload = read_checkpoint(cfg["policy_checkpoint"])
    policy_hash = state_digest(payload["state_dict"])
    if policy_hash != cfg["expected_policy_state_sha256"]:
        raise ValueError("Policy checkpoint is not the verified step-260 state_dict")
    if payload.get("global_step") != 260:
        raise ValueError("This experiment requires source step 260")
    del payload
    runtimes, provenance = {}, {}
    base = Path(cfg["base_config"])
    for label in ("old", "new"):
        raw = absolutize_default_paths(deep_update(read_yaml(base), read_yaml(Path(cfg[label + "_overlay"]))), base.parent)
        raw["options"]["Training"]["EMA"].update(
            replace_model_after_load=False, use_for_generation=False,
            use_ema_during_training_eval=False,
        )
        checkpoint = read_checkpoint(cfg[label + "_checkpoint"])
        if state_digest(checkpoint["state_dict"]) != policy_hash:
            raise ValueError(f"{label} monitor checkpoint policy differs from the fixed source; "
                             "use the new run's step-0 baseline, NOT last.ckpt after training")
        cache, spec, reward = monitor_payload(checkpoint)
        adaptive = raw["dgpo"]["adaptive_omnifold"]
        if adaptive["single_pool_split_seed"] != cache["protocol"]["seed"]:
            raise ValueError(f"{label} overlay does not match saved monitor split seed")
        readiness = cache.get("training_policy")
        if readiness and not readiness.get("condition_width"):
            raise ValueError("Incomplete new monitor training provenance")
        provenance[label] = {
            "checkpoint": cfg[label + "_checkpoint"], "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("global_step"), "state_dict_sha256": policy_hash,
            "monitor_state_sha256": state_digest(cache["state"]),
            "monitor_tensor_count": len(cache["state"]), "split_protocol": cache["protocol"],
            "packing_spec": spec, "legacy_base_digest": reward["base_digest"],
            "source_bundle_sha256": checkpoint["dgpo_omnifold_reward_stack"]["source_bundle_sha256"],
        }
        runtimes[label] = raw
        del checkpoint, cache, reward
    # This is an architecture/initialization comparison, not a different dataset or sampler test.
    for path in (("platform", "data_parquet_dir"), ("platform", "data_parquet_val_dir"),
                 ("options", "Dataset", "normalization_file"),
                 ("reward_config", "omnifold", "backbone_checkpoint")):
        values = []
        for raw in runtimes.values():
            v = raw
            for k in path:
                v = v[k]
            values.append(v)
        if values[0] != values[1]:
            raise ValueError(f"Runtime mismatch at {'.'.join(path)}")
    steps = [raw["dgpo"].get("validation_num_ddim_steps", raw["dgpo"]["num_ddim_steps"]) for raw in runtimes.values()]
    if steps[0] != steps[1]:
        raise ValueError("Old/new DDIM step settings differ")
    cfg["ddim_steps"] = int(steps[0])
    cfg["provenance"] = provenance
    output = Path(cfg["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg["output_dir"] = str(output)
    for label, raw in runtimes.items():
        path = output / f"{label}_runtime.yaml"
        with path.open("x") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        cfg[label + "_runtime"] = str(path)
    _exclusive_json(output / "manifest.json", cfg)
    return cfg


def replay_worker(cfg):
    import ray.train
    import ray.train.torch
    from RL.DGPO_neutrino.model_utils import load_evenet_model_for_dgpo, load_training_config, load_normalization_dict
    from RL.DGPO_neutrino.dgpo_trainer import _materialize_adaptive_omnifold_pool
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterModelBuilder, EventPackingSpec
    from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import payload_sha256
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler

    logging.basicConfig(level=logging.INFO)
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    device = ray.train.torch.get_device()
    output = Path(cfg["output_dir"])
    global_config.load_yaml(cfg["new_runtime"])
    torch.set_float32_matmul_precision(str(global_config.dgpo.get("float32_matmul_precision", "medium")))
    source = read_checkpoint(cfg["policy_checkpoint"])
    if state_digest(source["state_dict"]) != cfg["expected_policy_state_sha256"]:
        raise ValueError("Policy checkpoint changed after preflight")
    bundle = load_evenet_model_for_dgpo(config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"])
    policy = bundle.model.eval()
    policy.requires_grad_(False)
    before = verify_policy_loaded(policy, source["state_dict"])
    del source
    pool = _materialize_adaptive_omnifold_pool(
        ray.train.get_dataset_shard("pool"),
        {"batch_size": cfg["generation_batch_size"], "prefetch_batches": 1},
        model=policy, sampler=DDIMSampler(device=device), device=device,
        world_size=world, rank=rank, quota_events=cfg["pool_events"],
        num_ddim_steps=cfg["ddim_steps"], seed=cfg["seed"], include_pairwise_context=True,
    )
    if pool.n_events != cfg["pool_events"]:
        raise ValueError(f"Only generated {pool.n_events} / {cfg['pool_events']} requested events")
    if state_digest(policy.state_dict()) != before:
        raise RuntimeError("Fixed policy changed during generation")
    if rank == 0:
        _exclusive_torch_save(output / "pool.pt", {
            "schema_version": 1, "policy_state_sha256": before,
            "packing_spec": pool.packing_spec.to_dict(), "packed_event": pool.packed_event.cpu(),
            "truth": pool.truth.cpu(), "raw": pool.candidates.cpu(),
        })
    del policy, bundle
    gc.collect()
    torch.cuda.empty_cache()

    conditions, protocols = [], []
    for label in ("old", "new"):
        meta = cfg["provenance"][label]
        conditions.append(repack_pool(pool, EventPackingSpec.from_dict(meta["packing_spec"])).to(device))
        protocols.append(meta["split_protocol"])
    val, individual_counts = shared_validation_indices(
        conditions, protocols, minimum=cfg["minimum_validation_events"],
    )
    truth, raw = pool.truth.to(device)[val], pool.candidates.to(device)[val]
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "policy_loaded_verified": True, "policy_unchanged": True,
              "loaded_policy_sha256": before, "pool_events": pool.n_events,
              "validation_rule": "intersection of saved old/new identity-hash validation folds",
              "individual_validation_counts": dict(zip(("old", "new"), individual_counts)),
              "common_validation_events": len(val), "classifier_fits": 0, "policy_updates": 0,
              "metrics": {}, "limitations": [
                  "Fresh common-noise pool, not a bitwise replay of historical Ray row order or DDIM noise.",
                  "Validation was used for earlier model selection; it is not an untouched test set.",
                  "Legacy monitor caches have no self-contained frozen backbone or full architecture manifest; supplied runtime overlays are recorded.",
              ]}
    saved_scores = {"validation_indices": val.cpu()}
    for label, condition in zip(("old", "new"), conditions):
        config = load_training_config(cfg[label + "_runtime"])
        recal = dict(config.dgpo.adaptive_omnifold.recalibration)
        audit = dict(config.dgpo.adaptive_omnifold.audit_fit)
        kwargs = {k: v for k, v in recal.items() if k in inspect.signature(EvenetAdapterModelBuilder).parameters}
        kwargs.update({k: audit[k] for k in ARCH_KEYS if k in audit})
        if any(kwargs.get(k, False) for k in ("periodic_pair_features", "topology_fourier_embedding", "topology_conditioning")):
            raise ValueError("This diagnostic compares two CLEAN raw monitors, not Fourier reward classifiers")
        builder = EvenetAdapterModelBuilder(
            config=config, normalization_dict=load_normalization_dict(config),
            checkpoint_path=config.reward_config.omnifold.backbone_checkpoint,
            device=device, **kwargs,
        )
        meta = cfg["provenance"][label]
        if builder.base_digest != meta["legacy_base_digest"]:
            raise ValueError(f"{label} pretrained body fingerprint changed; stop before interpreting AUC")
        with Path(config.reward_config.omnifold.backbone_checkpoint).open("rb") as f:
            backbone_hash = hashlib.file_digest(f, "sha256").hexdigest()
        source_identity = {"schema_version": 2, "kind": "ztautau_in_dgpo_omnifold_bootstrap",
                           "base_digest": builder.base_digest, "policy_reference_sha256": backbone_hash}
        if payload_sha256(source_identity) != meta["source_bundle_sha256"]:
            raise ValueError(f"{label} backbone file does not match saved source provenance")
        model = builder.make_classifier(EventPackingSpec.from_dict(meta["packing_spec"]), reset=True)
        checkpoint = read_checkpoint(cfg[label + "_checkpoint"])
        cache, _, _ = monitor_payload(checkpoint)
        if state_digest(cache["state"]) != meta["monitor_state_sha256"]:
            raise ValueError(f"{label} monitor checkpoint changed after preflight")
        strict_monitor_load(model, cache["state"])
        metrics, logits = score_monitor(model, condition[val], truth, raw, cfg["score_batch_size"])
        metrics["backbone_file_sha256"] = backbone_hash
        report["metrics"][label] = metrics
        saved_scores[label] = logits
        if rank == 0:
            _exclusive_json(output / f"{label}_metrics.json", metrics)
            print(f"[raw-monitor-replay] {label}: {json.dumps(metrics)}", flush=True)
        del model, builder, cache, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    report["auc_old_minus_new"] = report["metrics"]["old"]["auc"] - report["metrics"]["new"]["auc"]
    if rank == 0:
        _exclusive_torch_save(output / "scores.pt", saved_scores)
        _exclusive_json(output / "report.json", report)
        print(f"[raw-monitor-replay] report: {output / 'report.json'}", flush=True)
    ray.train.report({"old_auc": report["metrics"]["old"]["auc"],
                      "new_auc": report["metrics"]["new"]["auc"], "validation_events": len(val)})


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Override with another NEW diagnostic directory")
    parser.add_argument("--prepare-only", action="store_true", help="CPU checkpoint preflight; no Ray/GPU work")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.output_dir:
        settings["output_dir"] = str(args.output_dir)
    cfg = prepare(settings)
    print(f"Diagnostic output: {cfg['output_dir']}", flush=True)
    if args.prepare_only:
        return 0
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from evenet.control.global_config import global_config
    from evenet.shared import make_process_fn, register_dataset
    global_config.load_yaml(cfg["new_runtime"])
    path = Path(global_config.platform.data_parquet_dir)
    files = sorted(map(str, path.glob("*.parquet")))
    if not files:
        raise ValueError(f"No processed parquet files in {path}")
    ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", runtime_env={"env_vars": {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]),
    }})
    dataset, _ = register_dataset(files, make_process_fn(path), global_config.platform,
                                  dataset_limit=1.0, file_shuffling=False)
    trainer = TorchTrainer(
        train_loop_per_worker=replay_worker, train_loop_config=cfg, datasets={"pool": dataset},
        scaling_config=ScalingConfig(num_workers=cfg["workers"], use_gpu=True,
                                     resources_per_worker={"CPU": 2, "GPU": 1}),
        run_config=RunConfig(name="raw-monitor-replay", storage_path=str(Path(cfg["output_dir"]) / "ray_results"),
                             failure_config=FailureConfig(max_failures=0)),
    )
    trainer.fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
