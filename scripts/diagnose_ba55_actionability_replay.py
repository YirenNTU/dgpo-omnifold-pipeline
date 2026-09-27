#!/usr/bin/env python3
"""Replay the saved two-member BA55 H4 gradient over several policy radii.

The source classifiers, fixed K=8 logits, independent H4 judge, and policy
event panel come from the completed aqbszk1r trajectory diagnostic.  This
program performs no classifier fit and no optimizer update.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_raw_monitor_replay as replay
import diagnose_reward_interface as interface
from ablate_raw_monitor_initialization import clone_state

from RL.DGPO_neutrino.reward_interface import reward_advantage_arms


REQUIRED_SOURCE_ARTIFACTS = (
    "runtime.yaml",
    "manifest.json",
    "trajectory_report.json",
    "fixed_k1_pool.pt",
    "trajectory_fixed_k8_panel.pt",
    "independent_h4_judge.pt",
)
DEFAULT_TARGET_COMPONENTS = (
    "tau_a_delta_theta",
    "tau_a_delta_phi",
    "tau_b_delta_theta",
    "tau_b_delta_phi",
)
PHYSICS_REPLAY_SCHEMA = "c4a91e07-h4-ba55-physics-actionability-replay-v1"


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _member_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["seed"]), int(row["fold"])


def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(settings)
    required = {
        "source_dir", "output_dir", "expected_source_schema",
        "expected_policy_step", "reward_stage", "selected_members", "workers",
        "cpus_per_worker", "probe_events", "K", "gradient_events",
        "gradient_blocks", "gradient_timesteps", "gradient_seed", "beta",
        "policy_eval_t_min", "policy_eval_t_max", "raw_tempering",
        "step_rms_values", "rollout_seeds", "score_batch_size",
        "minimum_plus_beats_zero_fraction",
        "minimum_plus_beats_minus_fraction", "bootstrap_seed",
        "bootstrap_replicates", "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing BA55 replay settings: {missing}")
    for key in (
        "expected_policy_step", "workers", "cpus_per_worker", "probe_events",
        "K", "gradient_events", "gradient_blocks", "gradient_timesteps",
        "gradient_seed", "score_batch_size", "bootstrap_seed",
        "bootstrap_replicates",
    ):
        if type(cfg[key]) is not int or int(cfg[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if int(cfg["workers"]) != 16 or int(cfg["K"]) != 8:
        raise ValueError("the saved BA55 replay is pinned to 16 workers and K=8")
    if int(cfg["gradient_events"]) > int(cfg["probe_events"]):
        raise ValueError("gradient_events cannot exceed probe_events")
    if (
        int(cfg["gradient_blocks"]) < 2
        or int(cfg["gradient_events"]) < int(cfg["gradient_blocks"])
    ):
        raise ValueError("gradient panel needs at least two nonempty blocks")
    if int(cfg["bootstrap_replicates"]) < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    for key in (
        "beta", "raw_tempering", "minimum_plus_beats_zero_fraction",
        "minimum_plus_beats_minus_fraction",
    ):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if cfg["beta"] <= 0.0 or cfg["raw_tempering"] <= 0.0:
        raise ValueError("beta and raw_tempering must be positive")
    for key in (
        "minimum_plus_beats_zero_fraction",
        "minimum_plus_beats_minus_fraction",
    ):
        if not 0.0 <= cfg[key] <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1]")
    radii = [float(value) for value in cfg["step_rms_values"]]
    if (
        not radii
        or radii != sorted(set(radii))
        or any(not math.isfinite(value) or value <= 0.0 for value in radii)
    ):
        raise ValueError(
            "step_rms_values must be finite, positive, unique, and increasing"
        )
    cfg["step_rms_values"] = radii
    seeds = cfg["rollout_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 1 for seed in seeds)
    ):
        raise ValueError("rollout_seeds must contain distinct positive integers")
    members = cfg["selected_members"]
    if not isinstance(members, list) or len(members) != 2:
        raise ValueError("selected_members must contain exactly two OOF folds")
    keys = [_member_key(row) for row in members]
    if keys != [(20260913, 1), (20260913, 2)]:
        raise ValueError(
            "the one-repeat replay must select seed 20260913 folds 1 and 2"
        )
    if str(cfg["reward_stage"]) != "ba_lcb_55":
        raise ValueError("this replay is pinned to the saved ba_lcb_55 stage")
    physics = cfg.get("physics_endpoint")
    if physics is not None:
        if not isinstance(physics, Mapping):
            raise ValueError("physics_endpoint must be a mapping")
        physics = dict(physics)
        enabled = physics.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("physics_endpoint.enabled must be boolean")
        components = physics.get(
            "target_components", list(DEFAULT_TARGET_COMPONENTS)
        )
        if (
            not isinstance(components, list)
            or not components
            or len(set(components)) != len(components)
            or any(component not in DEFAULT_TARGET_COMPONENTS for component in components)
        ):
            raise ValueError(
                "physics_endpoint.target_components must be unique Ztautau targets"
            )
        physics_bins = int(physics.get("physics_bins", 60))
        response_bins = int(physics.get("response_bins", 12))
        if physics_bins < 10 or response_bins < 2:
            raise ValueError("physics/response bins are too small")
        if enabled and len(radii) != 1:
            raise ValueError(
                "physics endpoint replay requires exactly one predeclared RMS"
            )
        physics.update(
            enabled=enabled,
            target_components=list(components),
            physics_bins=physics_bins,
            response_bins=response_bins,
        )
        cfg["physics_endpoint"] = physics
    wandb = dict(cfg["wandb"] or {})
    if wandb.get("required") and not wandb.get("enabled"):
        raise ValueError("wandb.required=true requires wandb.enabled=true")
    if wandb.get("enabled") and not wandb.get("id"):
        raise ValueError("live W&B replay requires a fixed run id")
    cfg["wandb"] = wandb
    return cfg


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable aqbszk1r artifacts before allocating workers."""

    cfg = _validated_settings(settings)
    source = Path(cfg["source_dir"]).expanduser().resolve(strict=True)
    paths = {
        name: (source / name).resolve(strict=True)
        for name in REQUIRED_SOURCE_ARTIFACTS
    }
    report = json.loads(paths["trajectory_report.json"].read_text())
    if report.get("schema") != cfg["expected_source_schema"]:
        raise ValueError(
            f"source schema mismatch: expected {cfg['expected_source_schema']!r}, "
            f"found {report.get('schema')!r}"
        )
    if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("source report policy step does not match c4a91e07")
    if not report.get("fixed_policy") or not report.get("policy_unchanged"):
        raise ValueError("source trajectory did not certify an unchanged policy")
    if int(report.get("policy_updates", -1)) != 0:
        raise ValueError("source trajectory unexpectedly updated the policy")
    protocol = report.get("trajectory_protocol") or {}
    if protocol.get("reward_arm") != "raw_loo":
        raise ValueError("source trajectory did not use the production raw_loo arm")
    judge_fit = (report.get("independent_judge") or {}).get("fit_diagnostics") or {}
    if not bool(judge_fit.get("saturated")):
        raise ValueError("saved independent H4 judge was not saturation certified")
    if int(judge_fit.get("steps_completed", 0)) < 1000:
        raise ValueError("saved independent H4 judge is under the 1000-step floor")

    stage = str(cfg["reward_stage"])
    report_members = (report.get("trajectory_members") or {}).get(stage) or []
    report_member_keys = [_member_key(row) for row in report_members]
    selected_keys = [_member_key(row) for row in cfg["selected_members"]]
    missing_members = [key for key in selected_keys if key not in report_member_keys]
    if missing_members:
        raise ValueError(f"source report lacks selected BA55 members: {missing_members}")

    panel = _load(paths["trajectory_fixed_k8_panel.pt"])
    panel_members = panel.get("trajectory_members", {}).get(stage) or []
    panel_member_keys = [_member_key(row) for row in panel_members]
    if panel_member_keys != report_member_keys:
        raise ValueError("saved panel member order differs from the source report")
    logits = panel.get("stage_member_logits", {}).get(stage)
    if logits is None or logits.ndim != 3:
        raise ValueError("saved panel has no (member,K,event) BA55 logits")
    if int(logits.shape[0]) != len(panel_member_keys):
        raise ValueError("saved BA55 member-logit count is inconsistent")
    if int(logits.shape[1]) != int(cfg["K"]):
        raise ValueError("saved BA55 logits use a different K")
    if int(logits.shape[2]) < int(cfg["probe_events"]):
        raise ValueError("saved panel has fewer events than probe_events")
    member_indices = [panel_member_keys.index(key) for key in selected_keys]
    del panel, logits

    policy_checkpoint = Path(report["policy_checkpoint"]).resolve(strict=True)
    checkpoint = replay.read_checkpoint(policy_checkpoint)
    if int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("live policy checkpoint step changed since aqbszk1r")
    del checkpoint
    runtime = __import__("yaml").safe_load(paths["runtime.yaml"].read_text())
    if runtime["dgpo"].get("checkpoint_load_mode") != "weights_only":
        raise ValueError("source runtime did not load policy weights only")
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("BA55 replay output overlaps its source artifact directory")
    source_stats = {
        str(path): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in paths.values()
    }
    source_stats[str(policy_checkpoint)] = [
        policy_checkpoint.stat().st_size, policy_checkpoint.stat().st_mtime_ns
    ]
    cfg.update(
        source_dir=str(source),
        output_dir=str(output),
        runtime_path=str(paths["runtime.yaml"]),
        policy_checkpoint=str(policy_checkpoint),
        member_indices=member_indices,
        source_artifact_wandb_run=(
            "ytchou97-university-of-washington/nu2flow-RL/aqbszk1r"
        ),
        policy_source_wandb_run=report.get("source_wandb_run"),
        source_judge_steps=int(judge_fit["steps_completed"]),
        source_stats=source_stats,
    )
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"source artifact changed during BA55 replay: {path}")


def diagnose_sweep(
    sweep: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the predeclared paired multi-seed radius decision rule."""

    radius_rows: dict[str, Any] = {}
    for radius_index, radius in enumerate(settings["step_rms_values"]):
        key = f"{float(radius):.12g}"
        seed_rows = list(sweep[key]["seeds"])
        plus_zero = np.asarray([
            float(row["plus_judge_auc_gap"])
            - float(row["zero_judge_auc_gap"])
            for row in seed_rows
        ])
        plus_minus = np.asarray([
            float(row["plus_judge_auc_gap"])
            - float(row["minus_judge_auc_gap"])
            for row in seed_rows
        ])
        finite = bool(np.isfinite(plus_zero).all() and np.isfinite(plus_minus).all())
        no_nonfinite_candidates = all(
            float(row["candidate_nonfinite_fraction_max_rank"]) == 0.0
            for row in seed_rows
        )
        if finite:
            plus_zero_stats = interface._bootstrap_mean_interval(
                plus_zero.tolist(),
                seed=int(settings["bootstrap_seed"]) + radius_index,
                replicates=int(settings["bootstrap_replicates"]),
            )
            plus_minus_stats = interface._bootstrap_mean_interval(
                plus_minus.tolist(),
                seed=int(settings["bootstrap_seed"]) + 10000 + radius_index,
                replicates=int(settings["bootstrap_replicates"]),
            )
            plus_beats_zero = float(np.mean(plus_zero < 0.0))
            plus_beats_minus = float(np.mean(plus_minus < 0.0))
        else:
            plus_zero_stats = {"mean": float("nan"), "ci90_low": float("nan"), "ci90_high": float("nan")}
            plus_minus_stats = dict(plus_zero_stats)
            plus_beats_zero = plus_beats_minus = 0.0
        reliable = bool(
            finite
            and no_nonfinite_candidates
            and plus_zero_stats["mean"] < 0.0
            and plus_zero_stats["ci90_high"] < 0.0
            and plus_beats_zero
            >= float(settings["minimum_plus_beats_zero_fraction"])
            and plus_beats_minus
            >= float(settings["minimum_plus_beats_minus_fraction"])
        )
        radius_rows[key] = {
            "step_rms": float(radius),
            "rollout_seeds": len(seed_rows),
            "plus_minus_zero": plus_zero_stats,
            "plus_minus_minus": plus_minus_stats,
            "plus_beats_zero_fraction": plus_beats_zero,
            "plus_beats_minus_fraction": plus_beats_minus,
            "no_nonfinite_candidates": no_nonfinite_candidates,
            "reliable": reliable,
        }
    reliable_rows = [row for row in radius_rows.values() if row["reliable"]]
    selected = min(
        reliable_rows,
        key=lambda row: row["plus_minus_zero"]["mean"],
        default=None,
    )
    base = min(radius_rows.values(), key=lambda row: row["step_rms"])
    if selected is None:
        finding = "ba55_actionability_not_replicated"
    elif selected["step_rms"] > base["step_rms"]:
        finding = "larger_ba55_radius_supported"
    else:
        finding = "base_ba55_radius_preferred"
    return {
        "finding": finding,
        "selected_step_rms": None if selected is None else selected["step_rms"],
        "base_step_rms": base["step_rms"],
        "radii": radius_rows,
        "selection_rule": (
            "Select the reliable radius with the most negative mean paired "
            "plus-minus-zero H4 judge-gap change. Reliability requires a "
            "negative 90% bootstrap upper bound, the declared seed consistency, "
            "plus beating minus, and zero nonfinite generated candidates."
        ),
    }


def diagnose_physics_endpoint(
    sweep: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Test whether the BA55 plus direction improves mean target JSD."""

    physics = settings.get("physics_endpoint") or {}
    if not bool(physics.get("enabled", False)):
        raise ValueError("physics endpoint is disabled")
    radius = float(settings["step_rms_values"][0])
    radius_key = f"{radius:.12g}"
    rows = list(sweep[radius_key]["seeds"])
    plus_zero = np.asarray([
        float(row["plus_target_jsd_mean"])
        - float(row["zero_target_jsd_mean"])
        for row in rows
    ])
    plus_minus = np.asarray([
        float(row["plus_target_jsd_mean"])
        - float(row["minus_target_jsd_mean"])
        for row in rows
    ])
    finite = bool(np.isfinite(plus_zero).all() and np.isfinite(plus_minus).all())
    no_nonfinite_candidates = all(
        float(row["candidate_nonfinite_fraction_max_rank"]) == 0.0
        for row in rows
    )
    if finite:
        plus_zero_stats = interface._bootstrap_mean_interval(
            plus_zero.tolist(),
            seed=int(settings["bootstrap_seed"]),
            replicates=int(settings["bootstrap_replicates"]),
        )
        plus_minus_stats = interface._bootstrap_mean_interval(
            plus_minus.tolist(),
            seed=int(settings["bootstrap_seed"]) + 1,
            replicates=int(settings["bootstrap_replicates"]),
        )
        plus_beats_zero = float(np.mean(plus_zero < 0.0))
        plus_beats_minus = float(np.mean(plus_minus < 0.0))
    else:
        plus_zero_stats = {
            "mean": float("nan"),
            "ci90_low": float("nan"),
            "ci90_high": float("nan"),
        }
        plus_minus_stats = dict(plus_zero_stats)
        plus_beats_zero = plus_beats_minus = 0.0
    components: dict[str, Any] = {}
    for component_index, component in enumerate(physics["target_components"]):
        component_plus_zero = np.asarray([
            float(row["plus_target_jsd"][component])
            - float(row["zero_target_jsd"][component])
            for row in rows
        ])
        component_plus_minus = np.asarray([
            float(row["plus_target_jsd"][component])
            - float(row["minus_target_jsd"][component])
            for row in rows
        ])
        component_finite = bool(
            np.isfinite(component_plus_zero).all()
            and np.isfinite(component_plus_minus).all()
        )
        if component_finite:
            component_plus_zero_stats = interface._bootstrap_mean_interval(
                component_plus_zero.tolist(),
                seed=int(settings["bootstrap_seed"]) + 10 + component_index,
                replicates=int(settings["bootstrap_replicates"]),
            )
            component_plus_minus_stats = interface._bootstrap_mean_interval(
                component_plus_minus.tolist(),
                seed=int(settings["bootstrap_seed"]) + 20 + component_index,
                replicates=int(settings["bootstrap_replicates"]),
            )
        else:
            component_plus_zero_stats = {
                "mean": float("nan"),
                "ci90_low": float("nan"),
                "ci90_high": float("nan"),
            }
            component_plus_minus_stats = dict(component_plus_zero_stats)
        components[component] = {
            "plus_minus_zero": component_plus_zero_stats,
            "plus_minus_minus": component_plus_minus_stats,
            "finite": component_finite,
        }
    reliable = bool(
        finite
        and no_nonfinite_candidates
        and plus_zero_stats["mean"] < 0.0
        and plus_zero_stats["ci90_high"] < 0.0
        and plus_beats_zero
        >= float(settings["minimum_plus_beats_zero_fraction"])
        and plus_beats_minus
        >= float(settings["minimum_plus_beats_minus_fraction"])
    )
    return {
        "finding": (
            "ba55_h4_physics_improvement_supported"
            if reliable
            else "ba55_h4_physics_improvement_not_replicated"
        ),
        "step_rms": radius,
        "primary_endpoint": "mean four-target JSD plus-minus-zero",
        "plus_minus_zero": plus_zero_stats,
        "plus_minus_minus": plus_minus_stats,
        "plus_beats_zero_fraction": plus_beats_zero,
        "plus_beats_minus_fraction": plus_beats_minus,
        "component_deltas": components,
        "no_nonfinite_candidates": no_nonfinite_candidates,
        "reliable": reliable,
        "classifier_gap_is_secondary": True,
    }


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import batch_to_device
    from RL.DGPO_neutrino.model_utils import (
        load_evenet_model_for_dgpo,
        load_normalization_dict,
    )
    from RL.DGPO_neutrino.diagnostics.ztautau_validation import (
        _histogram_jsd,
        _observable_edges,
        build_ztautau_validation_metrics,
        collect_ztautau_validation_arrays,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EventPackingSpec,
        EvenetAdapterModelBuilder,
        unpack_event_inputs,
    )
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    if world != int(cfg["workers"]):
        raise ValueError(f"expected {cfg['workers']} Ray workers, found {world}")
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg["runtime_path"])
    torch.set_float32_matmul_precision(
        str(global_config.dgpo.get("float32_matmul_precision", "medium"))
    )
    root = Path(cfg["output_dir"])
    source = Path(cfg["source_dir"])
    wandb_run = None
    if rank == 0 and cfg["wandb"].get("enabled"):
        import wandb

        wandb_run = wandb.init(
            entity=cfg["wandb"].get("entity"),
            project=cfg["wandb"].get("project"),
            name=cfg["wandb"].get("name"),
            id=cfg["wandb"].get("id"),
            resume=cfg["wandb"].get("resume"),
            group=cfg["wandb"].get("group"),
            tags=cfg["wandb"].get("tags"),
            job_type="diagnostic",
            config=interface._jsonable({
                key: value for key, value in cfg.items()
                if key != "source_stats"
            }),
        )
        wandb_run.define_metric("actionability/rollout_index", hidden=True)
        wandb_run.define_metric(
            "actionability/*",
            step_metric="actionability/rollout_index",
            step_sync=False,
        )
        if bool((cfg.get("physics_endpoint") or {}).get("enabled", False)):
            wandb_run.define_metric("physics_actionability/rollout_index", hidden=True)
            wandb_run.define_metric(
                "physics_actionability/*",
                step_metric="physics_actionability/rollout_index",
                step_sync=False,
            )
        wandb_run.summary.update({
            "phase": "load_aqbszk1r_artifacts",
            "classifier_fits": 0,
            "policy_updates": 0,
        })

    panel = _load(source / "trajectory_fixed_k8_panel.pt")
    pool = _load(source / "fixed_k1_pool.pt")
    packing_spec = EventPackingSpec.from_dict(pool["packing_spec"])
    limit = int(cfg["probe_events"])
    global_positions = torch.arange(limit)[rank::world]
    packed = panel["packed_event"][:limit][rank::world]
    truth = panel["truth"][:limit][rank::world]
    noise_mask = panel["policy_noise_mask"][:limit][rank::world]
    fixed_candidates = panel["candidates_kb22"][:, :limit][:, rank::world].to(device)
    batch = unpack_event_inputs(packed, packing_spec)
    batch["x_invisible"] = truth.reshape(-1, 2, 2)
    batch["x_invisible_mask"] = noise_mask
    batch = batch_to_device(batch, device)

    checkpoint = replay.read_checkpoint(cfg["policy_checkpoint"])
    bundle = load_evenet_model_for_dgpo(
        config=global_config,
        device=device,
        checkpoint_path=cfg["policy_checkpoint"],
    )
    policy = bundle.model.eval()
    interface.verify_policy_loaded(policy, checkpoint["state_dict"])
    anchor_model_state = clone_state(policy.state_dict())
    reference_bundle = load_evenet_model_for_dgpo(
        config=global_config,
        device=device,
        checkpoint_path=cfg["policy_checkpoint"],
    )
    reference = reference_bundle.model.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    interface.verify_policy_loaded(reference, checkpoint["state_dict"])

    all_logits = panel["stage_member_logits"][cfg["reward_stage"]]
    selected_logits = all_logits[cfg["member_indices"], :, :limit]
    reward_logits = selected_logits.mean(0)[:, rank::world].to(device)
    advantage = reward_advantage_arms(
        reward_logits,
        temperature=1.0,
        raw_tempering=float(cfg["raw_tempering"]),
    )["raw_loo"]
    gradient_report, gradients = interface._gradient_audit(
        cfg={**cfg, "training_seeds": [20260913]},
        policy=policy,
        reference=reference,
        batch=batch,
        candidates=fixed_candidates,
        advantage_sets={"primary/raw_loo": advantage},
        global_positions=global_positions,
        device=device,
        dtype=next(policy.parameters()).dtype,
    )
    direction = gradients["primary/raw_loo"]
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    anchor = tuple(parameter.detach().cpu().clone() for parameter in parameters)

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
        packing_spec, "ba55_replay_independent_judge", reset=True
    ).to(device)
    judge.load_state_dict(
        _load(source / "independent_h4_judge.pt"), strict=True
    )
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    truth_logits_local = interface._score_local_on_device(
        judge, packed, truth, int(cfg["score_batch_size"])
    ).cpu()
    sampler = DDIMSampler(device=device)
    physics_enabled = bool(
        (cfg.get("physics_endpoint") or {}).get("enabled", False)
    )

    def restore_anchor() -> None:
        with torch.no_grad():
            for parameter, base in zip(parameters, anchor):
                parameter.copy_(base.to(parameter.device, parameter.dtype))

    def evaluate(
        direction_vector: torch.Tensor | None,
        *,
        sign: int,
        radius: float,
        seed: int,
    ) -> dict[str, Any]:
        restore_anchor()
        scale = 0.0
        if direction_vector is not None:
            scale = interface._assign_direction(
                parameters,
                anchor,
                direction_vector,
                sign=sign,
                epsilon_rms=float(radius),
            )
        torch.manual_seed(int(seed) + rank)
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
        local_nonfinite = float((~torch.isfinite(candidates)).float().mean().cpu())
        gathered_nonfinite = interface._all_gather_object(local_nonfinite)
        candidates = torch.nan_to_num(candidates)
        arrays: dict[str, np.ndarray] = {}
        if physics_enabled:
            arrays = collect_ztautau_validation_arrays(
                candidates,
                fixed_candidates,
                batch,
                torch.ones(len(truth), device=device, dtype=torch.bool),
            )
            arrays = interface._gather_numpy_dict(arrays)
        generated = candidates.permute(1, 0, 2, 3).reshape(
            len(truth), int(cfg["K"]), 4
        ).cpu()
        generated_logits = interface._score_local_on_device(
            judge, packed, generated, int(cfg["score_batch_size"])
        ).cpu()
        gathered_truth = interface._all_gather_object(truth_logits_local)
        gathered_generated = interface._all_gather_object(generated_logits)
        if rank != 0:
            return {}
        metrics = interface._classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_generated).reshape(-1)
        )
        result = {
            "judge": metrics,
            "judge_auc_gap": float(metrics["auc_gap"]),
            "flat_gradient_scale": float(scale),
            "candidate_nonfinite_fraction_mean": float(np.mean(gathered_nonfinite)),
            "candidate_nonfinite_fraction_max_rank": float(np.max(gathered_nonfinite)),
        }
        if physics_enabled:
            physics = build_ztautau_validation_metrics(
                arrays,
                val_k=int(cfg["K"]),
                tarp_config={"enabled": False},
                metrics_config={
                    "enabled": True,
                    "bins": int(cfg["physics_endpoint"]["physics_bins"]),
                    "candidate_index": 0,
                },
                include_images=False,
            )
            response, _ = interface.response_matrix_metrics(
                arrays["_tarp_truth"],
                arrays["_tarp_candidates"],
                bins=int(cfg["physics_endpoint"]["response_bins"]),
            )
            target_jsd = {}
            for component in cfg["physics_endpoint"]["target_components"]:
                name = f"target/{component}"
                truth_values = arrays[f"{name}/truth"]
                current_values = arrays[f"{name}/current"]
                reference_values = arrays[f"{name}/ref"]
                # Keep the binning fixed across zero/minus/plus and rollout
                # seeds. Including the arm being evaluated in the range would
                # make a paired JSD change partly a moving-bin artifact.
                edges = _observable_edges(
                    name,
                    (truth_values, reference_values),
                    int(cfg["physics_endpoint"]["physics_bins"]),
                )
                target_jsd[component] = float(
                    _histogram_jsd(truth_values, current_values, edges)
                )
            result.update({
                "physics": physics,
                "target_jsd": target_jsd,
                "target_jsd_mean": float(np.mean(list(target_jsd.values()))),
                "response": response,
                "response_mean_abs_bin_offset": float(np.mean([
                    values["mean_abs_bin_offset"]
                    for values in response.values()
                ])),
            })
        return result

    sweep = {
        f"{float(radius):.12g}": {"step_rms": float(radius), "seeds": []}
        for radius in cfg["step_rms_values"]
    }
    for seed_index, rollout_seed in enumerate(cfg["rollout_seeds"]):
        zero = evaluate(None, sign=0, radius=0.0, seed=int(rollout_seed))
        for radius in cfg["step_rms_values"]:
            radius_key = f"{float(radius):.12g}"
            minus = evaluate(
                direction,
                sign=-1,
                radius=float(radius),
                seed=int(rollout_seed),
            )
            plus = evaluate(
                direction,
                sign=1,
                radius=float(radius),
                seed=int(rollout_seed),
            )
            if rank == 0:
                row = {
                    "rollout_seed": int(rollout_seed),
                    "zero_judge_auc_gap": zero["judge_auc_gap"],
                    "minus_judge_auc_gap": minus["judge_auc_gap"],
                    "plus_judge_auc_gap": plus["judge_auc_gap"],
                    "candidate_nonfinite_fraction_max_rank": max(
                        zero["candidate_nonfinite_fraction_max_rank"],
                        minus["candidate_nonfinite_fraction_max_rank"],
                        plus["candidate_nonfinite_fraction_max_rank"],
                    ),
                }
                if physics_enabled:
                    row.update({
                        "zero_target_jsd": zero["target_jsd"],
                        "minus_target_jsd": minus["target_jsd"],
                        "plus_target_jsd": plus["target_jsd"],
                        "zero_target_jsd_mean": zero["target_jsd_mean"],
                        "minus_target_jsd_mean": minus["target_jsd_mean"],
                        "plus_target_jsd_mean": plus["target_jsd_mean"],
                        "zero_response_mean_abs_bin_offset": zero[
                            "response_mean_abs_bin_offset"
                        ],
                        "minus_response_mean_abs_bin_offset": minus[
                            "response_mean_abs_bin_offset"
                        ],
                        "plus_response_mean_abs_bin_offset": plus[
                            "response_mean_abs_bin_offset"
                        ],
                        "zero_physics": zero["physics"],
                        "minus_physics": minus["physics"],
                        "plus_physics": plus["physics"],
                    })
                sweep[radius_key]["seeds"].append(row)
                plus_delta = (
                    row["plus_judge_auc_gap"] - row["zero_judge_auc_gap"]
                )
                physics_message = (
                    "target_jsd_delta="
                    f"{row['plus_target_jsd_mean'] - row['zero_target_jsd_mean']:+.6g} "
                    if physics_enabled
                    else ""
                )
                print(
                    f"[ba55-replay] seed={rollout_seed} rms={radius:.1e} "
                    f"delta={plus_delta:+.6g} "
                    f"{physics_message}"
                    f"plus_beats_minus={row['plus_judge_auc_gap'] < row['minus_judge_auc_gap']}",
                    flush=True,
                )
                if wandb_run is not None:
                    metric_root = f"actionability/rms_{radius_key}"
                    wandb_run.log({
                        "actionability/rollout_index": seed_index,
                        f"{metric_root}/zero_gap": row["zero_judge_auc_gap"],
                        f"{metric_root}/minus_gap": row["minus_judge_auc_gap"],
                        f"{metric_root}/plus_gap": row["plus_judge_auc_gap"],
                        f"{metric_root}/plus_delta": plus_delta,
                        f"{metric_root}/nonfinite_fraction_max_rank": row[
                            "candidate_nonfinite_fraction_max_rank"
                        ],
                    })
                    if physics_enabled:
                        physics_root = "physics_actionability/ba_lcb_55"
                        physics_metrics = {
                            "physics_actionability/rollout_index": seed_index,
                            f"{physics_root}/zero_target_jsd_mean": row[
                                "zero_target_jsd_mean"
                            ],
                            f"{physics_root}/minus_target_jsd_mean": row[
                                "minus_target_jsd_mean"
                            ],
                            f"{physics_root}/plus_target_jsd_mean": row[
                                "plus_target_jsd_mean"
                            ],
                            f"{physics_root}/plus_target_jsd_delta": (
                                row["plus_target_jsd_mean"]
                                - row["zero_target_jsd_mean"]
                            ),
                            f"{physics_root}/zero_response_offset": row[
                                "zero_response_mean_abs_bin_offset"
                            ],
                            f"{physics_root}/minus_response_offset": row[
                                "minus_response_mean_abs_bin_offset"
                            ],
                            f"{physics_root}/plus_response_offset": row[
                                "plus_response_mean_abs_bin_offset"
                            ],
                        }
                        for component in cfg["physics_endpoint"][
                            "target_components"
                        ]:
                            physics_metrics[
                                f"{physics_root}/zero_target_jsd/{component}"
                            ] = row["zero_target_jsd"][component]
                            physics_metrics[
                                f"{physics_root}/minus_target_jsd/{component}"
                            ] = row["minus_target_jsd"][component]
                            physics_metrics[
                                f"{physics_root}/plus_target_jsd/{component}"
                            ] = row["plus_target_jsd"][component]
                        wandb_run.log(physics_metrics)

    restore_anchor()
    if rank == 0:
        decision = diagnose_sweep(sweep, cfg)
        physics_decision = (
            diagnose_physics_endpoint(sweep, cfg) if physics_enabled else None
        )
        interface.verify_policy_loaded(policy, anchor_model_state)
        report = {
            "schema": (
                PHYSICS_REPLAY_SCHEMA
                if physics_enabled
                else "c4a91e07-h4-ba55-m2-actionability-replay-v1"
            ),
            "source_artifact_wandb_run": cfg["source_artifact_wandb_run"],
            "policy_source_wandb_run": cfg["policy_source_wandb_run"],
            "source_dir": cfg["source_dir"],
            "source_schema": cfg["expected_source_schema"],
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "reward_stage": cfg["reward_stage"],
            "selected_members": cfg["selected_members"],
            "member_indices": cfg["member_indices"],
            "reward_arm": "raw_loo",
            "raw_tempering": float(cfg["raw_tempering"]),
            "classifier_fits": 0,
            "policy_updates": 0,
            "optimizer_state_resumed": False,
            "saved_candidate_logits_reused": True,
            "saved_independent_judge_reused": True,
            "common_random_numbers_within_seed": True,
            "target_jsd_binning": (
                "edges from truth plus frozen saved reference; fixed across "
                "zero/minus/plus arms and rollout seeds"
            ) if physics_enabled else None,
            "gradient": gradient_report["primary/raw_loo"],
            "sweep": sweep,
            "decision": decision,
            "physics_decision": physics_decision,
            "scope": (
                "Read-only replay of the saved two-member BA55 production "
                "gradient. It tests local policy-radius actionability"
                + (
                    " with mean target JSD as the primary endpoint"
                    if physics_enabled
                    else ""
                )
                + " and does not establish closed-loop fresh-classifier closure."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(root / "report.json", interface._jsonable(report))
        if wandb_run is not None:
            import wandb

            numeric: dict[str, float] = {}
            for radius_key, row in decision["radii"].items():
                metric_root = f"actionability_summary/rms_{radius_key}"
                numeric.update({
                    f"{metric_root}/mean_plus_delta": row[
                        "plus_minus_zero"
                    ]["mean"],
                    f"{metric_root}/ci90_low": row[
                        "plus_minus_zero"
                    ]["ci90_low"],
                    f"{metric_root}/ci90_high": row[
                        "plus_minus_zero"
                    ]["ci90_high"],
                    f"{metric_root}/plus_beats_zero_fraction": row[
                        "plus_beats_zero_fraction"
                    ],
                    f"{metric_root}/plus_beats_minus_fraction": row[
                        "plus_beats_minus_fraction"
                    ],
                    f"{metric_root}/reliable": float(row["reliable"]),
                })
            if physics_decision is not None:
                physics_root = "physics_actionability_summary/ba_lcb_55"
                numeric.update({
                    f"{physics_root}/target_jsd_delta_mean": physics_decision[
                        "plus_minus_zero"
                    ]["mean"],
                    f"{physics_root}/target_jsd_delta_ci90_low": physics_decision[
                        "plus_minus_zero"
                    ]["ci90_low"],
                    f"{physics_root}/target_jsd_delta_ci90_high": physics_decision[
                        "plus_minus_zero"
                    ]["ci90_high"],
                    f"{physics_root}/plus_beats_zero_fraction": physics_decision[
                        "plus_beats_zero_fraction"
                    ],
                    f"{physics_root}/plus_beats_minus_fraction": physics_decision[
                        "plus_beats_minus_fraction"
                    ],
                    f"{physics_root}/reliable": float(
                        physics_decision["reliable"]
                    ),
                })
                for component, component_row in physics_decision[
                    "component_deltas"
                ].items():
                    numeric[
                        f"{physics_root}/component_delta/{component}"
                    ] = component_row["plus_minus_zero"]["mean"]
            wandb_run.log(numeric)
            artifact = wandb.Artifact(
                (
                    "c4a91e07-h4-ba55-physics-actionability-replay"
                    if physics_enabled
                    else "c4a91e07-h4-ba55-m2-actionability-replay"
                ),
                type="diagnostic",
            )
            artifact.add_file(str(root / "manifest.json"))
            artifact.add_file(str(root / "report.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete",
                "classifier_fits": 0,
                "policy_updates": 0,
                "decision": (
                    physics_decision["finding"]
                    if physics_decision is not None
                    else decision["finding"]
                ),
                "selected_step_rms": (
                    physics_decision["step_rms"]
                    if physics_decision is not None
                    else decision["selected_step_rms"]
                ),
            })
            wandb_run.finish()
        final_finding = (
            physics_decision["finding"]
            if physics_decision is not None
            else decision["finding"]
        )
        final_rms = (
            physics_decision["step_rms"]
            if physics_decision is not None
            else decision["selected_step_rms"]
        )
        print(
            f"[ba55-replay] decision={final_finding} selected_rms={final_rms}",
            flush=True,
        )
        print(f"[ba55-replay] report={root / 'report.json'}", flush=True)
    ray.train.report({"completed": 1, "classifier_fits": 0, "policy_updates": 0})


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
        f"Verified aqbszk1r artifacts: BA55 members={cfg['member_indices']} "
        f"judge_steps={cfg['source_judge_steps']} "
        f"radii={len(cfg['step_rms_values'])} seeds={len(cfg['rollout_seeds'])}; "
        "classifier_fits=0 policy_updates=0; "
        + (
            f"physics_schema={PHYSICS_REPLAY_SCHEMA}"
            if bool((cfg.get("physics_endpoint") or {}).get("enabled", False))
            else "physics_endpoint=disabled"
        ),
        flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"].get("required"):
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in {
            "1", "true", "yes",
        }
        offline = os.environ.get("WANDB_MODE", "").lower() in {
            "offline", "disabled", "dryrun",
        }
        if disabled or offline:
            raise RuntimeError("this replay requires live W&B logging")
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is unset; start/source the 16-GPU Ray cluster")
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
            f"Ray cluster has {available_gpus:g} GPUs; replay requires "
            f"{cfg['workers']}"
        )
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    replay._exclusive_json(root / "manifest.json", interface._jsonable(cfg))
    TorchTrainer(
        train_loop_per_worker=_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]),
            use_gpu=True,
            resources_per_worker={
                "CPU": int(cfg["cpus_per_worker"]),
                "GPU": 1,
            },
        ),
        run_config=RunConfig(
            name=(
                "h4-ba55-physics-actionability-replay"
                if bool((cfg.get("physics_endpoint") or {}).get("enabled", False))
                else "h4-ba55-radius-actionability-replay"
            ),
            storage_path=str(root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
