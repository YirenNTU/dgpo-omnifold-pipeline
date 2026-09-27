#!/usr/bin/env python3
"""Confirm the h4time01 low-noise ordering across fresh rollout seeds.

This read-only diagnostic loads the saved, VP-distance-matched time-stratum
directions from h4time01.  It evaluates common-noise signed probes on the union
of the h4grad01 probe and h4natg01 judge identities.  It does not recompute a
gradient, fit a classifier, update the policy, or change the DGPO objective.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_block_natural_gradient as geometry
import diagnose_production_gradient_reproducibility as gradient_repro
import diagnose_raw_monitor_replay as replay
import diagnose_reward_interface as interface
from ablate_raw_monitor_initialization import clone_state


REWARD_ARTIFACTS = (
    "runtime.yaml",
    "manifest.json",
    "report.json",
    "fixed_k1_pool.pt",
    "independent_h4_judge.pt",
)
TIME_ARTIFACTS = (
    "manifest.json",
    "report.json",
    "time_stratum_directions_fp16.pt",
)
ARMS = ("full", "low", "rest", "high")
DIRECTION_NAMES = ("full", "low", "rest", "mid_low", "mid_high", "high")
PANELS = ("union", "h4grad01_probe", "h4natg01_judge")
REPORT_SCHEMA = "c4a91e07-h4-low-noise-replay-v1"


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _artifact_paths(root: Path, names: Sequence[str]) -> dict[str, Path]:
    return {name: (root / name).resolve(strict=True) for name in names}


def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(settings)
    required = {
        "reward_source_dir",
        "time_source_dir",
        "output_dir",
        "expected_reward_schema",
        "expected_time_schema",
        "expected_direction_schema",
        "expected_source_wandb_run",
        "expected_time_wandb_id",
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "K",
        "score_batch_size",
        "rollout_seeds",
        "bootstrap_replicates",
        "bootstrap_seed",
        "confidence_level",
        "minimum_low_over_full_gain",
        "minimum_consistency_fraction",
        "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing low-noise replay settings: {missing}")

    for key in (
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "K",
        "score_batch_size",
        "bootstrap_replicates",
        "bootstrap_seed",
    ):
        if type(cfg[key]) is not int or int(cfg[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if int(cfg["workers"]) != 16:
        raise ValueError("this experiment requires workers=16")
    if int(cfg["K"]) != 8:
        raise ValueError("this experiment requires K=8")
    if int(cfg["bootstrap_replicates"]) < 1000:
        raise ValueError("bootstrap_replicates must be at least 1000")

    seeds = list(cfg["rollout_seeds"] or ())
    if len(seeds) != 8:
        raise ValueError("this experiment requires exactly eight rollout seeds")
    if any(type(seed) is not int or seed < 1 for seed in seeds):
        raise ValueError("rollout seeds must be positive integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("rollout seeds must be unique")
    cfg["rollout_seeds"] = seeds

    for key in (
        "confidence_level",
        "minimum_low_over_full_gain",
        "minimum_consistency_fraction",
    ):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if not 0.0 < cfg["confidence_level"] < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if cfg["minimum_low_over_full_gain"] <= 1.0:
        raise ValueError("minimum_low_over_full_gain must exceed one")
    if not 0.5 <= cfg["minimum_consistency_fraction"] <= 1.0:
        raise ValueError("minimum_consistency_fraction must be in [0.5, 1]")

    wandb = dict(cfg["wandb"] or {})
    wandb.setdefault("enabled", False)
    if type(wandb["enabled"]) is not bool:
        raise ValueError("wandb.enabled must be boolean")
    if wandb.get("required") and not wandb["enabled"]:
        raise ValueError("wandb.required=true requires wandb.enabled=true")
    cfg["wandb"] = wandb
    return cfg


def select_confirmation_indices(
    final_indices: Tensor,
    gradient_manifest: Mapping[str, Any],
    geometry_manifest: Mapping[str, Any],
    *,
    direction_events: int,
) -> dict[str, Tensor]:
    """Reconstruct two old evaluation panels disjoint from time-direction fit."""

    old_gradient, old_probe = gradient_repro.select_disjoint_event_indices(
        final_indices, gradient_manifest
    )
    _, old_geometry_judge, _ = geometry.select_unused_event_indices(
        final_indices, gradient_manifest, geometry_manifest
    )
    if direction_events < 1 or direction_events > int(old_gradient.numel()):
        raise ValueError("invalid h4time01 direction-event count")
    direction = old_gradient[:direction_events]
    if int(old_probe.numel()) != 2048 or int(old_geometry_judge.numel()) != 2048:
        raise ValueError("confirmation requires two 2,048-event source panels")
    union = torch.cat((old_probe, old_geometry_judge))
    for name, values in {
        "direction": direction,
        "h4grad01_probe": old_probe,
        "h4natg01_judge": old_geometry_judge,
        "union": union,
    }.items():
        if int(torch.unique(values).numel()) != int(values.numel()):
            raise ValueError(f"{name} contains duplicate event identities")
    direction_set = set(direction.tolist())
    if not direction_set.isdisjoint(union.tolist()):
        raise ValueError("confirmation union overlaps h4time01 direction identities")
    if not set(old_probe.tolist()).isdisjoint(old_geometry_judge.tolist()):
        raise ValueError("the two confirmation panels overlap")
    return {
        "direction": direction,
        "h4grad01_probe": old_probe,
        "h4natg01_judge": old_geometry_judge,
        "union": union,
    }


def validate_direction_payload(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_policy_step: int,
    expected_layout: Sequence[Mapping[str, Any]],
    matched_distances: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the saved directions and their h4time01 distance calibration."""

    if payload.get("schema") != expected_schema:
        raise ValueError("time-direction schema mismatch")
    if int(payload.get("policy_global_step", -1)) != int(expected_policy_step):
        raise ValueError("time-direction policy step mismatch")
    layout = payload.get("parameter_layout")
    if not expected_layout or layout != list(expected_layout):
        raise ValueError("time-direction parameter layout differs from h4time01")
    directions = payload.get("directions")
    expected_names = set(DIRECTION_NAMES)
    if not isinstance(directions, Mapping) or set(directions) != expected_names:
        raise ValueError("time-direction payload does not contain all six directions")
    sizes = set()
    result_directions: dict[str, Tensor] = {}
    for name, value in directions.items():
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise ValueError(f"saved {name} direction is not a flat tensor")
        value = value.float()
        if not bool(torch.isfinite(value).all()) or float(value.double().norm()) <= 0.0:
            raise ValueError(f"saved {name} direction is zero or non-finite")
        sizes.add(int(value.numel()))
        result_directions[name] = value
    if len(sizes) != 1 or int(expected_layout[-1]["stop"]) != next(iter(sizes)):
        raise ValueError("time directions and parameter layout have different sizes")

    saved_rms = payload.get("matched_parameter_rms")
    if not isinstance(saved_rms, Mapping) or set(saved_rms) != expected_names:
        raise ValueError("time-direction payload lacks matched parameter RMS values")
    result_rms: dict[str, float] = {}
    for name in expected_names:
        value = float(saved_rms[name])
        report_row = matched_distances.get(name, {})
        report_value = float(report_row.get("parameter_rms", float("nan")))
        if (
            not math.isfinite(value)
            or value <= 0.0
            or not bool(report_row.get("passed"))
            or not math.isclose(value, report_value, rel_tol=1.0e-9, abs_tol=0.0)
        ):
            raise ValueError(f"invalid or inconsistent matched RMS for {name}")
        result_rms[name] = value
    return {
        "directions": result_directions,
        "matched_parameter_rms": result_rms,
        "parameter_layout": list(layout),
    }


def percentile_mean_interval(
    values: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """Deterministic percentile interval for the mean over paired seeds."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    if replicates < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(replicates), len(array)))
    means = array[indices].mean(axis=1)
    alpha = 0.5 * (1.0 - float(confidence))
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean": float(array.mean()),
        "lower": float(low),
        "upper": float(high),
        "confidence": float(confidence),
        "replicates": int(replicates),
    }


def classify_replication(summary: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    if float(summary.get("candidate_nonfinite_fraction_max", 0.0)) > 0.0:
        return "invalid_nonfinite_candidates"
    threshold = float(cfg["minimum_consistency_fraction"])
    signs_and_consistency = (
        float(summary["low_improvement"]) > 0.0
        and float(summary["mean_low_delta"]) < float(summary["mean_full_delta"])
        and float(summary["mean_high_delta"]) > 0.0
        and float(summary["low_beats_full_fraction"]) >= threshold
        and float(summary["high_harmful_fraction"]) >= threshold
    )
    if not signs_and_consistency:
        return "h4time01_ordering_not_replicated"
    strong = (
        math.isfinite(float(summary["low_over_full_gain"]))
        and float(summary["low_over_full_gain"])
        >= float(cfg["minimum_low_over_full_gain"])
        and float(summary["low_minus_full_interval"]["upper"]) < 0.0
        and float(summary["high_delta_interval"]["lower"]) > 0.0
    )
    if strong:
        return "noise_dependent_credit_ordering_replicates"
    return "pattern_persists_but_uncertain"


def summarize_replications(
    rows: Sequence[Mapping[str, Any]],
    *,
    panel: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize per-seed, common-noise AUC-gap deltas for one panel."""

    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}")
    if len(rows) != len(cfg["rollout_seeds"]):
        raise ValueError("replication row count differs from rollout seed count")
    if [int(row["seed"]) for row in rows] != list(cfg["rollout_seeds"]):
        raise ValueError("replication rows do not match the declared rollout seeds")
    deltas = {
        arm: np.asarray([
            float(row["panels"][panel]["arms"][arm]["plus_gap_delta"])
            for row in rows
        ], dtype=np.float64)
        for arm in ARMS
    }
    if any(not np.isfinite(values).all() for values in deltas.values()):
        raise ValueError("replication deltas contain non-finite values")
    low_minus_full = deltas["low"] - deltas["full"]
    full_improvement = -float(deltas["full"].mean())
    low_improvement = -float(deltas["low"].mean())
    gain = (
        low_improvement / full_improvement
        if full_improvement > 0.0
        else float("nan")
    )
    bootstrap = dict(
        replicates=int(cfg["bootstrap_replicates"]),
        confidence=float(cfg["confidence_level"]),
    )
    summary: dict[str, Any] = {
        "seeds": [int(row["seed"]) for row in rows],
        "mean_full_delta": float(deltas["full"].mean()),
        "mean_low_delta": float(deltas["low"].mean()),
        "mean_rest_delta": float(deltas["rest"].mean()),
        "mean_high_delta": float(deltas["high"].mean()),
        "full_improvement": full_improvement,
        "low_improvement": low_improvement,
        "low_over_full_gain": gain,
        "low_beats_full_fraction": float(np.mean(low_minus_full < 0.0)),
        "high_harmful_fraction": float(np.mean(deltas["high"] > 0.0)),
        "candidate_nonfinite_fraction_max": float(max(
            float(row.get("candidate_nonfinite_fraction_max", 0.0)) for row in rows
        )),
        "low_plus_beats_minus_fraction": float(np.mean([
            bool(row["panels"][panel]["arms"]["low"]["plus_beats_minus"])
            for row in rows
        ])),
        "low_minus_full_interval": percentile_mean_interval(
            low_minus_full,
            seed=int(cfg["bootstrap_seed"]),
            **bootstrap,
        ),
        "high_delta_interval": percentile_mean_interval(
            deltas["high"],
            seed=int(cfg["bootstrap_seed"]) + 1,
            **bootstrap,
        ),
        "arm_deltas": {name: values.tolist() for name, values in deltas.items()},
    }
    summary["decision"] = classify_replication(summary, cfg)
    return summary


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = _validated_settings(settings)
    reward_root = Path(cfg["reward_source_dir"]).expanduser().resolve(strict=True)
    time_root = Path(cfg["time_source_dir"]).expanduser().resolve(strict=True)
    reward_paths = _artifact_paths(reward_root, REWARD_ARTIFACTS)
    time_paths = _artifact_paths(time_root, TIME_ARTIFACTS)
    reward_report = json.loads(reward_paths["report.json"].read_text())
    time_manifest = json.loads(time_paths["manifest.json"].read_text())
    time_report = json.loads(time_paths["report.json"].read_text())

    if reward_report.get("schema") != cfg["expected_reward_schema"]:
        raise ValueError("reward-interface schema mismatch")
    if reward_report.get("source_wandb_run") != cfg["expected_source_wandb_run"]:
        raise ValueError("reward-interface W&B provenance mismatch")
    if time_report.get("schema") != cfg["expected_time_schema"]:
        raise ValueError("h4time01 report schema mismatch")
    if time_report.get("source_wandb_run") != cfg["expected_source_wandb_run"]:
        raise ValueError("h4time01 source run mismatch")
    if time_manifest.get("wandb", {}).get("id") != cfg["expected_time_wandb_id"]:
        raise ValueError("h4time01 W&B identity mismatch")
    if time_report.get("result", {}).get("decision") != "low_noise_helps_but_is_not_decisive":
        raise ValueError("h4time01 does not have the result this replay confirms")
    for report in (reward_report, time_report):
        if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
            raise ValueError("source policy step mismatch")
    if int(time_report.get("policy_updates", -1)) != 0:
        raise ValueError("h4time01 unexpectedly updated the policy")
    if int(time_report.get("classifier_fits", -1)) != 0:
        raise ValueError("h4time01 unexpectedly fitted a classifier")
    if bool(time_report.get("objective_changed")):
        raise ValueError("h4time01 unexpectedly changed the objective")
    if int(time_manifest.get("K", -1)) != int(cfg["K"]):
        raise ValueError("h4time01 K differs from replay K")

    policy_checkpoint = Path(time_report["policy_checkpoint"]).resolve(strict=True)
    if policy_checkpoint != Path(reward_report["policy_checkpoint"]).resolve(strict=True):
        raise ValueError("reward source and h4time01 use different policy checkpoints")
    checkpoint = replay.read_checkpoint(policy_checkpoint)
    if int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("live policy checkpoint step differs from source reports")
    del checkpoint

    runtime = __import__("yaml").safe_load(reward_paths["runtime.yaml"].read_text())
    if runtime["dgpo"].get("checkpoint_load_mode") != "weights_only":
        raise ValueError("reward runtime must use weights-only policy loading")
    if int(runtime["dgpo"]["K"]) != int(cfg["K"]):
        raise ValueError("reward runtime K differs from replay K")
    training = runtime["options"]["Training"]
    if any(training.get("EMA", {}).get(key, False) for key in (
        "replace_model_after_load",
        "use_for_generation",
        "use_ema_during_training_eval",
    )):
        raise ValueError("reward runtime unexpectedly selects EMA weights")

    gradient_manifest = time_manifest.get("gradient_manifest")
    geometry_manifest = time_manifest.get("geometry_manifest")
    parameter_layout = time_manifest.get("parameter_layout")
    if not isinstance(gradient_manifest, Mapping):
        raise ValueError("h4time01 manifest lacks h4grad01 provenance")
    if not isinstance(geometry_manifest, Mapping):
        raise ValueError("h4time01 manifest lacks h4natg01 provenance")
    if not isinstance(parameter_layout, list) or not parameter_layout:
        raise ValueError("h4time01 manifest lacks its parameter layout")
    if gradient_manifest.get("wandb", {}).get("id") != "h4grad01":
        raise ValueError("h4time01 gradient provenance is not h4grad01")
    if geometry_manifest.get("wandb", {}).get("id") != "h4natg01":
        raise ValueError("h4time01 geometry provenance is not h4natg01")

    direction_payload = _load(time_paths["time_stratum_directions_fp16.pt"])
    validated_directions = validate_direction_payload(
        direction_payload,
        expected_schema=str(cfg["expected_direction_schema"]),
        expected_policy_step=int(cfg["expected_policy_step"]),
        expected_layout=parameter_layout,
        matched_distances=time_report["matched_distances"],
    )
    del direction_payload, validated_directions

    pool = _load(reward_paths["fixed_k1_pool.pt"])
    final_indices = pool.get("partitions", {}).get("final_audit")
    if not isinstance(final_indices, Tensor) or final_indices.ndim != 1:
        raise ValueError("reward source lacks final_audit identities")
    panels = select_confirmation_indices(
        final_indices.long(),
        gradient_manifest,
        geometry_manifest,
        direction_events=int(time_report["direction_events"]),
    )
    del pool

    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    for protected in (reward_root, time_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output directory overlaps an immutable source")
    all_paths = [*reward_paths.values(), *time_paths.values(), policy_checkpoint]
    cfg.update(
        reward_source_dir=str(reward_root),
        time_source_dir=str(time_root),
        output_dir=str(output),
        runtime_path=str(reward_paths["runtime.yaml"]),
        policy_checkpoint=str(policy_checkpoint),
        gradient_manifest=dict(gradient_manifest),
        geometry_manifest=dict(geometry_manifest),
        parameter_layout=parameter_layout,
        direction_events=int(panels["direction"].numel()),
        primary_events=int(panels["union"].numel()),
        h4grad01_probe_events=int(panels["h4grad01_probe"].numel()),
        h4natg01_judge_events=int(panels["h4natg01_judge"].numel()),
        matched_parameter_rms={
            name: float(time_report["matched_distances"][name]["parameter_rms"])
            for name in DIRECTION_NAMES
        },
        source_stats={
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in all_paths
        },
    )
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"source artifact changed during replay: {path}")


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import batch_to_device
    from RL.DGPO_neutrino.model_utils import (
        assert_dgpo_neutrino_policy_deterministic,
        load_evenet_model_for_dgpo,
        load_normalization_dict,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EvenetAdapterModelBuilder,
        EventPackingSpec,
        unpack_event_inputs,
    )
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    if world != int(cfg["workers"]):
        raise ValueError(f"expected {cfg['workers']} workers, found {world}")
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg["runtime_path"])
    torch.set_float32_matmul_precision(
        str(global_config.dgpo.get("float32_matmul_precision", "medium"))
    )
    output_root = Path(cfg["output_dir"])
    reward_root = Path(cfg["reward_source_dir"])
    time_root = Path(cfg["time_source_dir"])

    wandb_run = None
    if rank == 0 and cfg["wandb"].get("enabled"):
        import wandb

        wandb_run = wandb.init(
            entity=cfg["wandb"].get("entity"),
            project=cfg["wandb"].get("project"),
            id=cfg["wandb"].get("id"),
            resume=cfg["wandb"].get("resume", "never"),
            name=cfg["wandb"].get("name"),
            group=cfg["wandb"].get("group"),
            tags=cfg["wandb"].get("tags"),
            job_type="diagnostic",
            config=interface._jsonable({
                key: value
                for key, value in cfg.items()
                if key not in {
                    "source_stats",
                    "gradient_manifest",
                    "geometry_manifest",
                    "parameter_layout",
                }
            }),
        )
        wandb_run.summary.update({
            "phase": "load_sources",
            "classifier_fits": 0,
            "gradient_computations": 0,
            "policy_updates": 0,
        })
        wandb_run.log({
            "low_noise_replay/progress/seeds_complete": 0.0,
            "low_noise_replay/contract/K": float(cfg["K"]),
            "low_noise_replay/contract/workers": float(cfg["workers"]),
            "low_noise_replay/contract/primary_events": float(cfg["primary_events"]),
            "low_noise_replay/contract/policy_updates": 0.0,
            "low_noise_replay/contract/classifier_fits": 0.0,
        })

    pool = _load(reward_root / "fixed_k1_pool.pt")
    packing_spec = EventPackingSpec.from_dict(pool["packing_spec"])
    panels = select_confirmation_indices(
        pool["partitions"]["final_audit"].long(),
        cfg["gradient_manifest"],
        cfg["geometry_manifest"],
        direction_events=int(cfg["direction_events"]),
    )
    primary = panels["union"]
    positions = torch.arange(len(primary), dtype=torch.long)
    local_indices = primary[rank::world]
    local_positions = positions[rank::world]
    if len(local_indices) != int(cfg["primary_events"]) // world:
        raise RuntimeError("primary panel sharding produced an unexpected size")
    panel_boundary = int(cfg["h4grad01_probe_events"])
    local_panel_ids = (local_positions >= panel_boundary).long()

    checkpoint = replay.read_checkpoint(cfg["policy_checkpoint"])
    bundle = load_evenet_model_for_dgpo(
        config=global_config,
        device=device,
        checkpoint_path=cfg["policy_checkpoint"],
    )
    policy = bundle.model
    interface.verify_policy_loaded(policy, checkpoint["state_dict"])
    assert_dgpo_neutrino_policy_deterministic(policy)
    anchor_state = clone_state(policy.state_dict())
    del checkpoint
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    direction_payload = _load(time_root / "time_stratum_directions_fp16.pt")
    validated = validate_direction_payload(
        direction_payload,
        expected_schema=str(cfg["expected_direction_schema"]),
        expected_policy_step=int(cfg["expected_policy_step"]),
        expected_layout=cfg["parameter_layout"],
        matched_distances={
            name: {
                "parameter_rms": cfg["matched_parameter_rms"][name],
                "passed": True,
            }
            for name in direction_payload["matched_parameter_rms"]
        },
    )
    directions = {
        name: validated["directions"][name] for name in ARMS
    }
    matched_rms = {
        name: float(validated["matched_parameter_rms"][name]) for name in ARMS
    }
    vector_size = int(next(iter(directions.values())).numel())
    geometry.validate_parameter_layout(
        cfg["parameter_layout"], list(policy.named_parameters()), vector_size
    )
    anchor_parameters = tuple(
        parameter.detach().cpu().clone() for parameter in parameters
    )
    del direction_payload, validated

    packed = pool["packed_event"].index_select(0, local_indices)
    truth = pool["truth"].index_select(0, local_indices)
    noise_mask = pool["policy_noise_mask"].index_select(0, local_indices)
    batch = unpack_event_inputs(packed, packing_spec)
    batch["x_invisible"] = truth.reshape(-1, 2, 2)
    batch["x_invisible_mask"] = noise_mask
    batch = batch_to_device(batch, device)

    recalibration = dict(global_config.dgpo.adaptive_omnifold.recalibration)
    builder_keys = set(inspect.signature(EvenetAdapterModelBuilder).parameters)
    builder = EvenetAdapterModelBuilder(
        config=global_config,
        normalization_dict=load_normalization_dict(global_config),
        checkpoint_path=global_config.reward_config.omnifold.backbone_checkpoint,
        device=device,
        **{
            key: value
            for key, value in recalibration.items()
            if key in builder_keys
        },
    )
    builder.restore_pretrained_body()
    judge = builder.make_classifier(
        packing_spec, "low_noise_replay_judge", reset=True
    ).to(device)
    judge.load_state_dict(
        _load(reward_root / "independent_h4_judge.pt"), strict=True
    )
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    truth_logits_local = interface._score_local_on_device(
        judge, packed, truth, int(cfg["score_batch_size"])
    ).cpu()
    sampler = DDIMSampler(device=device)

    def restore_anchor() -> None:
        policy.load_state_dict(anchor_state, strict=True)
        policy.eval()
        policy.zero_grad(set_to_none=True)

    def evaluate(
        direction: Tensor | None,
        *,
        descent_sign: int,
        parameter_rms: float,
        rollout_seed: int,
    ) -> dict[str, Any]:
        restore_anchor()
        if direction is not None:
            geometry._set_parameter_offset(
                parameters,
                anchor_parameters,
                direction,
                signed_rms=-float(descent_sign) * float(parameter_rms),
            )
        torch.manual_seed(int(rollout_seed) + rank)
        with torch.no_grad():
            candidates = generate_neutrino_candidates(
                policy,
                batch,
                sampler,
                K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
                device=device,
                parallel_chains=int(
                    global_config.dgpo.get("rollout_parallel_chains", 1)
                ),
            )
        nonfinite = float((~torch.isfinite(candidates)).float().mean().cpu())
        candidates = torch.nan_to_num(candidates)
        sample = candidates.permute(1, 0, 2, 3).reshape(
            len(local_indices), int(cfg["K"]), 4
        ).cpu()
        generated_logits_local = interface._score_local_on_device(
            judge, packed, sample, int(cfg["score_batch_size"])
        ).cpu()
        gathered = interface._all_gather_object({
            "truth": truth_logits_local,
            "generated": generated_logits_local,
            "panel_ids": local_panel_ids,
            "nonfinite": nonfinite,
        })
        if rank != 0:
            return {}
        all_truth = torch.cat([row["truth"] for row in gathered])
        all_generated = torch.cat([row["generated"] for row in gathered])
        all_panel_ids = torch.cat([row["panel_ids"] for row in gathered])
        result: dict[str, Any] = {
            "candidate_nonfinite_fraction": float(np.mean([
                row["nonfinite"] for row in gathered
            ])),
            "candidate_nonfinite_fraction_max_rank": float(np.max([
                row["nonfinite"] for row in gathered
            ])),
            "panels": {},
        }
        masks = {
            "union": torch.ones(len(all_panel_ids), dtype=torch.bool),
            "h4grad01_probe": all_panel_ids == 0,
            "h4natg01_judge": all_panel_ids == 1,
        }
        for name, mask in masks.items():
            result["panels"][name] = interface._classification_metrics(
                all_truth[mask], all_generated[mask].reshape(-1)
            )
        return result

    rows: list[dict[str, Any]] = []
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "multi_seed_signed_probes"
    for seed_index, rollout_seed in enumerate(cfg["rollout_seeds"]):
        zero = evaluate(
            None,
            descent_sign=0,
            parameter_rms=0.0,
            rollout_seed=int(rollout_seed),
        )
        arm_results: dict[str, Any] = {}
        for arm in ARMS:
            minus = evaluate(
                directions[arm],
                descent_sign=-1,
                parameter_rms=matched_rms[arm],
                rollout_seed=int(rollout_seed),
            )
            plus = evaluate(
                directions[arm],
                descent_sign=1,
                parameter_rms=matched_rms[arm],
                rollout_seed=int(rollout_seed),
            )
            if rank == 0:
                arm_results[arm] = {"minus": minus, "plus": plus}
        if rank == 0:
            panel_rows: dict[str, Any] = {}
            for panel in PANELS:
                panel_arms: dict[str, Any] = {}
                for arm in ARMS:
                    minus_metrics = arm_results[arm]["minus"]["panels"][panel]
                    plus_metrics = arm_results[arm]["plus"]["panels"][panel]
                    zero_metrics = zero["panels"][panel]
                    panel_arms[arm] = {
                        "parameter_rms": matched_rms[arm],
                        "minus": minus_metrics,
                        "plus": plus_metrics,
                        "minus_gap_delta": (
                            minus_metrics["auc_gap"] - zero_metrics["auc_gap"]
                        ),
                        "plus_gap_delta": (
                            plus_metrics["auc_gap"] - zero_metrics["auc_gap"]
                        ),
                        "plus_beats_minus": (
                            plus_metrics["auc_gap"] < minus_metrics["auc_gap"]
                        ),
                    }
                panel_rows[panel] = {"zero": zero["panels"][panel], "arms": panel_arms}
            row = {
                "seed_index": seed_index,
                "seed": int(rollout_seed),
                "panels": panel_rows,
                "candidate_nonfinite_fraction_max": float(max(
                    [zero["candidate_nonfinite_fraction_max_rank"]]
                    + [
                        arm_results[arm][sign]["candidate_nonfinite_fraction_max_rank"]
                        for arm in ARMS for sign in ("minus", "plus")
                    ]
                )),
            }
            rows.append(row)
            union = panel_rows["union"]
            print(
                f"[low-noise-replay] seed={rollout_seed} "
                f"zero={union['zero']['auc_gap']:.6f} "
                f"full_delta={union['arms']['full']['plus_gap_delta']:+.6g} "
                f"low_delta={union['arms']['low']['plus_gap_delta']:+.6g} "
                f"rest_delta={union['arms']['rest']['plus_gap_delta']:+.6g} "
                f"high_delta={union['arms']['high']['plus_gap_delta']:+.6g}",
                flush=True,
            )
            if wandb_run is not None:
                metrics: dict[str, Any] = {
                    "low_noise_replay/progress/seeds_complete": float(seed_index + 1),
                    "low_noise_replay/seed/index": float(seed_index),
                    "low_noise_replay/seed/value": float(rollout_seed),
                    "low_noise_replay/union/zero_auc_gap": union["zero"]["auc_gap"],
                    "low_noise_replay/candidate_nonfinite_fraction_max": row[
                        "candidate_nonfinite_fraction_max"
                    ],
                }
                for panel in PANELS:
                    for arm in ARMS:
                        arm_row = panel_rows[panel]["arms"][arm]
                        metrics.update({
                            f"low_noise_replay/{panel}/{arm}/plus_auc_gap": arm_row[
                                "plus"
                            ]["auc_gap"],
                            f"low_noise_replay/{panel}/{arm}/minus_auc_gap": arm_row[
                                "minus"
                            ]["auc_gap"],
                            f"low_noise_replay/{panel}/{arm}/plus_gap_delta": arm_row[
                                "plus_gap_delta"
                            ],
                            f"low_noise_replay/{panel}/{arm}/plus_beats_minus": float(
                                arm_row["plus_beats_minus"]
                            ),
                        })
                wandb_run.log(metrics)

    restore_anchor()
    if rank == 0:
        interface.verify_policy_loaded(policy, anchor_state)
        summaries = {
            panel: summarize_replications(rows, panel=panel, cfg=cfg)
            for panel in PANELS
        }
        primary = summaries["union"]
        report = {
            "schema": REPORT_SCHEMA,
            "source_wandb_run": cfg["expected_source_wandb_run"],
            "source_time_wandb_run": cfg["expected_time_wandb_id"],
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0,
            "gradient_computations": 0,
            "classifier_fits": 0,
            "objective_changed": False,
            "time_reweighted_policy_update": False,
            "K": int(cfg["K"]),
            "direction_events": int(cfg["direction_events"]),
            "primary_events": int(cfg["primary_events"]),
            "primary_panel": "union",
            "primary_panel_components": {
                "h4grad01_probe": int(cfg["h4grad01_probe_events"]),
                "h4natg01_judge": int(cfg["h4natg01_judge_events"]),
            },
            "primary_disjoint_from_direction_estimation": True,
            "h4time01_judge_excluded_from_confirmation": True,
            "common_random_numbers_within_seed": True,
            "rollout_seeds": list(cfg["rollout_seeds"]),
            "matched_parameter_rms": matched_rms,
            "per_seed": rows,
            "summaries": summaries,
            "result": {
                "decision": primary["decision"],
                "authorizes_short_time_weighted_training": (
                    primary["decision"]
                    == "noise_dependent_credit_ordering_replicates"
                ),
            },
            "scope": (
                "Read-only multi-seed replay of saved h4time01 directions. "
                "No gradient computation, classifier fit, optimizer step, objective "
                "change, time-weighted training, or cold-H4 audit is performed."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(
            output_root / "report.json", interface._jsonable(report)
        )
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                "c4a91e07-h4-low-noise-replay", type="diagnostic"
            )
            artifact.add_file(str(output_root / "manifest.json"))
            artifact.add_file(str(output_root / "report.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete",
                "classifier_fits": 0,
                "gradient_computations": 0,
                "policy_updates": 0,
                "low_noise_replay/primary/decision": primary["decision"],
                "low_noise_replay/primary/low_over_full_gain": primary[
                    "low_over_full_gain"
                ],
                "low_noise_replay/primary/low_beats_full_fraction": primary[
                    "low_beats_full_fraction"
                ],
                "low_noise_replay/primary/high_harmful_fraction": primary[
                    "high_harmful_fraction"
                ],
                "low_noise_replay/primary/mean_full_delta": primary[
                    "mean_full_delta"
                ],
                "low_noise_replay/primary/mean_low_delta": primary[
                    "mean_low_delta"
                ],
                "low_noise_replay/primary/mean_rest_delta": primary[
                    "mean_rest_delta"
                ],
                "low_noise_replay/primary/mean_high_delta": primary[
                    "mean_high_delta"
                ],
                "low_noise_replay/primary/low_minus_full_ci90_upper": primary[
                    "low_minus_full_interval"
                ]["upper"],
                "low_noise_replay/primary/high_delta_ci90_lower": primary[
                    "high_delta_interval"
                ]["lower"],
                "low_noise_replay/primary/authorizes_training": float(
                    report["result"]["authorizes_short_time_weighted_training"]
                ),
            })
            wandb_run.finish()
        print(
            f"Low-noise replay report: {output_root / 'report.json'}",
            flush=True,
        )
    ray.train.report({
        "completed": 1,
        "classifier_fits": 0,
        "gradient_computations": 0,
        "policy_updates": 0,
    })


def main() -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.output_dir is not None:
        settings["output_dir"] = str(args.output_dir)
    cfg = prepare(settings)
    print(
        "Verified low-noise replay: "
        f"{cfg['primary_events']} events, {len(cfg['rollout_seeds'])} seeds, "
        "four signed directions, zero policy updates.",
        flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"].get("required"):
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in {
            "1", "true", "yes"
        }
        offline = os.environ.get("WANDB_MODE", "").lower() in {
            "offline", "disabled", "dryrun"
        }
        if disabled or offline:
            raise RuntimeError("this experiment requires live online W&B logging")
    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is unset; start/source the 16-GPU Ray cluster")

    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ray.init(
        address=ray_address,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join([
            str(ROOT / "evenet_dgpo"),
            str(ROOT / "scripts"),
            os.environ.get("PYTHONPATH", ""),
        ])}},
    )
    available_gpus = float(ray.cluster_resources().get("GPU", 0) or 0)
    if available_gpus < int(cfg["workers"]):
        raise RuntimeError(
            f"Ray cluster has {available_gpus:g} GPUs; requires {cfg['workers']}"
        )
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=False)
    replay._exclusive_json(output_root / "manifest.json", interface._jsonable(cfg))
    TorchTrainer(
        train_loop_per_worker=_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]),
            use_gpu=True,
            resources_per_worker={"CPU": int(cfg["cpus_per_worker"]), "GPU": 1},
        ),
        run_config=RunConfig(
            name="c4a91e07-h4-low-noise-replay-v1",
            storage_path=str(output_root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
