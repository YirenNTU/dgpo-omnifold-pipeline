#!/usr/bin/env python3
"""Decompose the step-20 H4 gradient-to-AdamW direction, without training.

The experiment restores the completed h4xfer01 step-20 policy, its paired
round reference, installed four-member H4 reward stack, and AdamW state.
It computes one production-sized gradient and compares five descent
directions on disjoint panels at matched VP-path distance.  No optimizer step
is committed, no classifier is fitted, and no checkpoint is modified.
"""

from __future__ import annotations

import argparse
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

from RL.DGPO_neutrino.reward_interface import vector_cosine


ARM_ORDER = (
    "native_adamw",
    "native_adamw_no_weight_decay",
    "zero_first_moment_keep_second",
    "fresh_adamw",
    "raw_gradient",
)


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(settings)
    required = {
        "base_config", "source_checkpoint", "source_runtime", "event_pool", "output_dir",
        "expected_source_wandb_id", "expected_policy_step",
        "expected_reward_round_id", "workers", "cpus_per_worker", "K",
        "events_per_worker", "event_microbatch_size", "gradient_timesteps",
        "gradient_candidate_seed", "gradient_seed", "panel_selection_seed",
        "curvature_events", "curvature_candidate_seed", "curvature_timesteps",
        "curvature_path_seed", "native_parameter_rms",
        "distance_match_tolerance", "distance_match_iterations",
        "minimum_parameter_rms", "maximum_parameter_rms", "judge_events",
        "judge_rollout_seeds", "score_batch_size", "save_direction_vectors",
        "upload_direction_vectors", "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing AdamW diagnostic settings: {missing}")
    integer_keys = (
        "expected_policy_step", "expected_reward_round_id", "workers",
        "cpus_per_worker", "K", "events_per_worker",
        "event_microbatch_size", "gradient_timesteps",
        "gradient_candidate_seed", "gradient_seed", "panel_selection_seed",
        "curvature_events", "curvature_candidate_seed",
        "curvature_timesteps", "curvature_path_seed", "judge_events",
        "score_batch_size",
    )
    for key in integer_keys:
        if type(cfg[key]) is not int or int(cfg[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    exact = {
        "expected_policy_step": 20,
        "expected_reward_round_id": 1,
        "workers": 16,
        "K": 8,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "gradient_timesteps": 8,
    }
    for key, expected in exact.items():
        if cfg[key] != expected:
            raise ValueError(
                f"h4opt01 requires {key}={expected}, found {cfg[key]}"
            )
    if cfg["events_per_worker"] % cfg["event_microbatch_size"]:
        raise ValueError("events_per_worker must divide into exact microbatches")
    if cfg["curvature_events"] % cfg["workers"] or cfg["judge_events"] % cfg["workers"]:
        raise ValueError("curvature_events and judge_events must divide across workers")
    seeds = cfg["judge_rollout_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 8:
        raise ValueError("h4opt01 requires exactly eight judge rollout seeds")
    if any(type(seed) is not int or seed < 1 for seed in seeds) or len(set(seeds)) != 8:
        raise ValueError("judge rollout seeds must be eight unique positive integers")
    for key in (
        "native_parameter_rms", "distance_match_tolerance",
        "minimum_parameter_rms", "maximum_parameter_rms",
    ):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0.0 < cfg["distance_match_tolerance"] <= 0.25:
        raise ValueError("distance_match_tolerance must lie in (0, 0.25]")
    if type(cfg["distance_match_iterations"]) is not int or not 1 <= cfg[
        "distance_match_iterations"
    ] <= 8:
        raise ValueError("distance_match_iterations must be in [1, 8]")
    if not (
        cfg["minimum_parameter_rms"] <= cfg["native_parameter_rms"]
        <= cfg["maximum_parameter_rms"]
    ):
        raise ValueError("native parameter RMS lies outside matching bounds")
    for key in ("save_direction_vectors", "upload_direction_vectors"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg["upload_direction_vectors"] and not cfg["save_direction_vectors"]:
        raise ValueError("upload_direction_vectors requires saved vectors")
    wandb = dict(cfg["wandb"] or {})
    if wandb.get("required") and not wandb.get("enabled"):
        raise ValueError("wandb.required=true requires wandb.enabled=true")
    if wandb.get("enabled") and wandb.get("id") != "h4opt01":
        raise ValueError("this predeclared experiment uses W&B id h4opt01")
    if wandb.get("enabled") and wandb.get("resume") != "allow":
        raise ValueError("h4opt01 must use resume=allow after a pre-result retry")
    cfg["wandb"] = wandb
    return cfg


def _runtime_contract(runtime: Mapping[str, Any]) -> dict[str, Any]:
    dgpo = runtime["dgpo"]
    trust = dict(dgpo.get("reference_trust") or {})
    variance = dict(dgpo.get("variance_regularization") or {})
    projection = dict(dgpo.get("projection_constraint") or {})
    contract = {
        "K": int(dgpo["K"]),
        "gradient_timesteps": int(dgpo["num_train_timesteps"]),
        "events_per_worker": int(runtime["platform"]["batch_size"]),
        "event_microbatch_size": int(dgpo["policy_eval_event_microbatch_size"]),
        "beta": float(dgpo["beta"]),
        "policy_eval_t_min": float(dgpo["policy_eval_t_min"]),
        "policy_eval_t_max": float(dgpo["policy_eval_t_max"]),
        "advantage_estimator": str(dgpo["advantage_estimator"]),
        "grad_clip_norm": float(dgpo.get("grad_clip_norm", 1.0)),
        "reference_trust_enabled": bool(trust.get("enabled", False)),
        "reference_trust_coefficient": float(trust.get("coefficient", 0.0)),
        "reference_trust_objective": str(trust.get("objective", "velocity_mse")),
        "hard_trust_enabled": bool((trust.get("adaptive_boundary") or {}).get("enabled", False)),
        "projection_type": str(projection.get("type", "none")),
        "variance_regularization_enabled": bool(variance.get("enabled", False)),
        "beta_kl": float(dgpo.get("beta_kl", 0.0)),
        "adv_clip_max": dgpo.get("adv_clip_max"),
        "steps_per_epoch": int(dgpo["steps_per_epoch"]),
    }
    expected = {
        "K": 8,
        "gradient_timesteps": 8,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "advantage_estimator": "leave_one_out_unscaled",
        "reference_trust_enabled": True,
        "reference_trust_coefficient": 1.0,
        "reference_trust_objective": "velocity_mse",
        "hard_trust_enabled": False,
        "projection_type": "none",
        "variance_regularization_enabled": False,
        "beta_kl": 0.0,
        "adv_clip_max": None,
        "steps_per_epoch": 10,
    }
    mismatch = {
        key: (contract[key], value)
        for key, value in expected.items()
        if contract[key] != value
    }
    if mismatch:
        raise ValueError(f"source production objective changed: {mismatch}")
    if not math.isclose(contract["beta"], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("source beta must remain 1.0")
    if not (
        math.isclose(contract["policy_eval_t_min"], 0.0, abs_tol=1e-12)
        and math.isclose(contract["policy_eval_t_max"], 0.7, abs_tol=1e-12)
    ):
        raise ValueError("source policy-evaluation time window changed")
    return contract


def _last_completed_raw_audit(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    history = list(
        (checkpoint.get("dgpo_adaptive_omnifold_state") or {}).get(
            "probe_history", []
        )
        or []
    )
    rows = [
        row for row in history
        if isinstance(row, Mapping)
        and math.isfinite(float(row.get("raw_auc_gap", float("nan"))))
    ]
    if not rows:
        raise ValueError("step-20 checkpoint has no completed cold H4 audit")
    return rows[-1]


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed before allocating the Ray workers."""

    from train_neutrino_backend import (
        absolutize_default_paths,
        deep_update,
        read_yaml,
        read_overlay_yaml,
    )

    cfg = _validated_settings(settings)
    for key in ("base_config", "source_checkpoint", "source_runtime", "event_pool"):
        path = Path(cfg[key]).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        cfg[key] = str(path.resolve(strict=True))
    checkpoint = replay.read_checkpoint(cfg["source_checkpoint"])
    if int(checkpoint.get("global_step", -1)) != cfg["expected_policy_step"]:
        raise ValueError("source checkpoint is not h4xfer01 step 20")
    if int(checkpoint.get("dgpo_reward_round_id", -1)) != cfg[
        "expected_reward_round_id"
    ]:
        raise ValueError("source checkpoint does not contain fixed reward round 1")
    required_checkpoint = {
        "state_dict", "dgpo_optimizer_state_dict", "dgpo_round_ref_state_dict",
        "dgpo_omnifold_reward_stack", "dgpo_adaptive_omnifold_state",
    }
    missing = sorted(required_checkpoint - set(checkpoint))
    if missing:
        raise ValueError(f"source full-state checkpoint is incomplete: {missing}")
    reward_payload = (
        checkpoint["dgpo_omnifold_reward_stack"].get("reward") or {}
    )
    if not reward_payload.get("increments") or not reward_payload.get("base_digest"):
        raise ValueError("source checkpoint has no self-contained installed H4 reward")
    audit = _last_completed_raw_audit(checkpoint)
    if int(audit.get("global_step", -1)) != cfg["expected_policy_step"]:
        raise ValueError("latest completed cold H4 audit is not the step-20 audit")
    if int(audit.get("raw_audit_training_steps", 0)) < 1000:
        raise ValueError("step-20 cold H4 audit did not complete 1000 updates")
    if float(audit.get("raw_audit_saturated", 1.0)) < 0.5:
        raise ValueError("saved step-20 H4 audit was not saturated")
    base_path = Path(cfg["base_config"])
    runtime = absolutize_default_paths(
        deep_update(
            read_yaml(base_path),
            read_overlay_yaml(Path(cfg["source_runtime"])),
        ),
        base_path.parent,
    )
    missing_runtime = sorted({"event_info", "resonance"} - set(runtime))
    if missing_runtime:
        raise ValueError(
            "merged source runtime still lacks required section(s): "
            + ", ".join(missing_runtime)
        )
    if runtime["logger"]["wandb"].get("id") != cfg["expected_source_wandb_id"]:
        raise ValueError("source runtime is not h4xfer01")
    contract = _runtime_contract(runtime)
    for key in ("K", "gradient_timesteps", "events_per_worker", "event_microbatch_size"):
        if contract[key] != cfg[key]:
            raise ValueError(f"diagnostic {key} differs from source production")
    pool = _load(Path(cfg["event_pool"]))
    needed_pool_keys = {"packing_spec", "packed_event", "truth", "policy_noise_mask", "partitions"}
    if not needed_pool_keys.issubset(pool):
        raise ValueError("event pool lacks required fixed event tensors")
    final = (pool.get("partitions") or {}).get("final_audit")
    minimum = (
        cfg["workers"] * cfg["events_per_worker"]
        + cfg["curvature_events"] + cfg["judge_events"]
    )
    if not isinstance(final, Tensor) or final.ndim != 1 or final.numel() < minimum:
        raise ValueError(f"fixed final-audit pool needs at least {minimum} identities")
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    source_stats = {
        str(Path(cfg[key])): [Path(cfg[key]).stat().st_size, Path(cfg[key]).stat().st_mtime_ns]
        for key in ("base_config", "source_checkpoint", "source_runtime", "event_pool")
    }
    cfg.update(
        output_dir=str(output),
        production_contract=contract,
        source_raw_audit=interface._jsonable(dict(audit)),
        source_stats=source_stats,
        total_gradient_events=cfg["workers"] * cfg["events_per_worker"],
        _runtime_payload=runtime,
    )
    del checkpoint, pool
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for raw_path, expected in cfg["source_stats"].items():
        path = Path(raw_path)
        observed = [path.stat().st_size, path.stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"immutable source changed during run: {path}")


def select_panels(
    final_indices: Tensor,
    cfg: Mapping[str, Any],
) -> tuple[Tensor, Tensor, Tensor]:
    """Select disjoint gradient, curvature, and frozen-judge identities."""

    final_indices = final_indices.long().cpu()
    if final_indices.ndim != 1:
        raise ValueError("panel indices must be one-dimensional")
    if int(torch.unique(final_indices).numel()) != int(final_indices.numel()):
        raise ValueError("final-audit event identities must be unique")
    generator = torch.Generator().manual_seed(int(cfg["panel_selection_seed"]))
    remaining = final_indices[torch.randperm(len(final_indices), generator=generator)]
    gradient_count = int(cfg["total_gradient_events"])
    curvature_count = int(cfg["curvature_events"])
    judge_count = int(cfg["judge_events"])
    if remaining.numel() < gradient_count + curvature_count + judge_count:
        raise ValueError("fixed pool is too small for three disjoint panels")
    gradient = remaining[:gradient_count]
    curvature = remaining[gradient_count:gradient_count + curvature_count]
    judge = remaining[
        gradient_count + curvature_count:
        gradient_count + curvature_count + judge_count
    ]
    return gradient, curvature, judge


def _state_step(state: Mapping[str, Any]) -> int:
    value = state.get("step", 0)
    if isinstance(value, Tensor):
        value = value.detach().cpu().item()
    step = int(value)
    if step < 0:
        raise ValueError("AdamW state has a negative step")
    return step


def _adamw_parameter_descent(
    parameter: Tensor,
    gradient: Tensor,
    state: Mapping[str, Any],
    group: Mapping[str, Any],
    *,
    keep_first_moment: bool,
    keep_second_moment: bool,
    keep_clock: bool,
    use_weight_decay: bool,
) -> Tensor:
    """Analytic PyTorch AdamW one-step displacement ``theta_old-theta_new``."""

    if bool(group.get("maximize", False)):
        gradient = -gradient
    beta1, beta2 = (float(value) for value in group.get("betas", (0.9, 0.999)))
    eps = float(group.get("eps", 1.0e-8))
    lr = float(group["lr"])
    wd = float(group.get("weight_decay", 0.0)) if use_weight_decay else 0.0
    old_step = _state_step(state) if keep_clock else 0
    step = old_step + 1
    old_m = state.get("exp_avg") if keep_first_moment else None
    old_v = state.get("exp_avg_sq") if keep_second_moment else None
    if old_m is None:
        old_m = torch.zeros_like(gradient)
    else:
        old_m = old_m.to(device=gradient.device, dtype=gradient.dtype)
    if old_v is None:
        old_v = torch.zeros_like(gradient)
    else:
        old_v = old_v.to(device=gradient.device, dtype=gradient.dtype)
    m_new = old_m.mul(beta1).add(gradient, alpha=1.0 - beta1)
    v_new = old_v.mul(beta2).addcmul(gradient, gradient, value=1.0 - beta2)
    if bool(group.get("amsgrad", False)):
        old_max = state.get("max_exp_avg_sq") if keep_second_moment else None
        if old_max is None:
            old_max = torch.zeros_like(v_new)
        else:
            old_max = old_max.to(device=gradient.device, dtype=gradient.dtype)
        denominator_state = torch.maximum(old_max, v_new)
    else:
        denominator_state = v_new
    m_hat = m_new / (1.0 - beta1 ** step)
    v_hat = denominator_state / (1.0 - beta2 ** step)
    adaptive = m_hat / (v_hat.sqrt() + eps)
    return lr * adaptive + lr * wd * parameter.detach()


def adamw_counterfactual_directions(
    parameters: Sequence[Tensor],
    optimizer: Any,
    flat_gradient: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Build five directions without mutating parameters or optimizer state."""

    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    group_by_parameter: dict[int, Mapping[str, Any]] = {}
    for group in base_optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in group_by_parameter:
                raise ValueError("parameter occurs in more than one optimizer group")
            group_by_parameter[id(parameter)] = group
    if sum(parameter.numel() for parameter in parameters) != flat_gradient.numel():
        raise ValueError("flat gradient length differs from trainable parameters")
    chunks: dict[str, list[Tensor]] = {name: [] for name in ARM_ORDER}
    state_steps: list[int] = []
    inactive_without_state = 0
    newly_active_without_state = 0
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        gradient = flat_gradient[offset:offset + count].reshape_as(parameter).to(
            parameter.device, parameter.dtype
        )
        offset += count
        try:
            group = group_by_parameter[id(parameter)]
        except KeyError as exc:
            raise ValueError("trainable parameter is absent from AdamW groups") from exc
        state = base_optimizer.state.get(parameter, {})
        if not state:
            if not bool(torch.count_nonzero(gradient).item()):
                inactive_without_state += 1
                for name in ARM_ORDER:
                    chunks[name].append(
                        torch.zeros_like(gradient).float().cpu().reshape(-1)
                    )
                continue
            newly_active_without_state += 1
            state = {}
        elif "exp_avg" not in state or "exp_avg_sq" not in state:
            raise ValueError("step-20 checkpoint has incomplete AdamW moments")
        state_steps.append(_state_step(state))
        variants = {
            "native_adamw": (True, True, True, True),
            "native_adamw_no_weight_decay": (True, True, True, False),
            "zero_first_moment_keep_second": (False, True, True, True),
            "fresh_adamw": (False, False, False, True),
        }
        for name, (keep_m, keep_v, keep_clock, keep_wd) in variants.items():
            chunks[name].append(_adamw_parameter_descent(
                parameter,
                gradient,
                state,
                group,
                keep_first_moment=keep_m,
                keep_second_moment=keep_v,
                keep_clock=keep_clock,
                use_weight_decay=keep_wd,
            ).detach().float().cpu().reshape(-1))
        chunks["raw_gradient"].append(gradient.detach().float().cpu().reshape(-1))
    directions = {name: torch.cat(chunks[name]) for name in ARM_ORDER}
    if any(not torch.isfinite(value).all() or geometry.vector_rms(value) <= 0.0
           for value in directions.values()):
        raise FloatingPointError("optimizer counterfactual produced a nonfinite/zero direction")
    diagnostics = {
        "optimizer_state_parameter_count": len(state_steps),
        "inactive_parameters_without_optimizer_state": inactive_without_state,
        "newly_active_parameters_without_optimizer_state": newly_active_without_state,
        "optimizer_state_step_min": min(state_steps) if state_steps else 0,
        "optimizer_state_step_max": max(state_steps) if state_steps else 0,
        "direction_rms": {
            name: geometry.vector_rms(value) for name, value in directions.items()
        },
        "cosine_to_raw_gradient": {
            name: vector_cosine(value, directions["raw_gradient"])
            for name, value in directions.items()
        },
        "pairwise_cosine": {
            f"{left}__{right}": vector_cosine(directions[left], directions[right])
            for left_index, left in enumerate(ARM_ORDER)
            for right in ARM_ORDER[left_index + 1:]
        },
    }
    return directions, diagnostics


def paired_mean_interval(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired interval needs at least two finite values")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(len(array)))
    # Eight predeclared seeds: t_(0.975, 7).
    half = 2.364624251 * se
    return {"mean": mean, "standard_error": se, "ci95_low": mean - half,
            "ci95_high": mean + half, "n": float(len(array))}


def classify_optimizer_result(
    arm_summaries: Mapping[str, Mapping[str, Any]],
    raw_minus_native: Mapping[str, float],
    *,
    all_distance_matches: bool,
) -> str:
    if not all_distance_matches:
        return "invalid_vp_distance_match"
    raw = arm_summaries["raw_gradient"]
    if (
        int(raw["plus_beats_minus_count"]) < 6
        or float(raw["improvement_zero_minus_plus"]["mean"]) <= 0.0
    ):
        return "raw_total_gradient_has_no_stable_local_h4_signal"
    if float(raw_minus_native["ci95_low"]) <= 0.0:
        return "optimizer_transform_not_established"
    native_gap = float(arm_summaries["native_adamw"]["plus_gap_mean"])
    raw_gap = float(arm_summaries["raw_gradient"]["plus_gap_mean"])
    threshold = native_gap - 0.5 * (native_gap - raw_gap)
    if float(arm_summaries["native_adamw_no_weight_decay"]["plus_gap_mean"]) <= threshold:
        return "weight_decay_is_material_optimizer_bottleneck"
    if float(arm_summaries["zero_first_moment_keep_second"]["plus_gap_mean"]) <= threshold:
        return "stale_first_moment_is_material_optimizer_bottleneck"
    if float(arm_summaries["fresh_adamw"]["plus_gap_mean"]) <= threshold:
        return "full_adam_state_is_material_optimizer_bottleneck"
    return "adaptive_preconditioning_or_group_scaling_is_material_bottleneck"


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import (
        batch_to_device,
        build_optimizer,
        build_reward_aggregator,
        policy_evaluation_step,
    )
    from RL.DGPO_neutrino.dgpo_utils import (
        compute_per_event_advantage,
        sample_cosine_vp_path_kl_timesteps,
    )
    from RL.DGPO_neutrino.model_utils import (
        apply_component_freezes,
        assert_dgpo_neutrino_policy_deterministic,
        load_evenet_model_for_dgpo,
        load_normalization_dict,
        make_round_reference_model,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EventPackingSpec,
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
                key: value for key, value in cfg.items()
                if key not in {"source_stats", "runtime_path"}
            }),
        )
        wandb_run.summary.update({
            "phase": "restore_step20_full_state",
            "classifier_fits": 0,
            "policy_updates": 0,
            "source_wandb_id": cfg["expected_source_wandb_id"],
            "source_policy_step": int(cfg["expected_policy_step"]),
        })
        wandb_run.log({
            "optimizer_counterfactual/progress/stage": 0.0,
            "optimizer_counterfactual/policy_updates": 0.0,
            "optimizer_counterfactual/classifier_fits": 0.0,
        })

    checkpoint = replay.read_checkpoint(cfg["source_checkpoint"])
    if (
        int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"])
        or int(checkpoint.get("dgpo_reward_round_id", -1))
        != int(cfg["expected_reward_round_id"])
    ):
        raise RuntimeError("h4xfer01 step-20 checkpoint provenance changed")
    pool = _load(Path(cfg["event_pool"]))
    pool_spec = EventPackingSpec.from_dict(pool["packing_spec"])
    gradient_indices, curvature_indices, judge_indices = select_panels(
        pool["partitions"]["final_audit"], cfg
    )
    local_gradient = gradient_indices[rank::world]
    local_curvature = curvature_indices[rank::world]
    local_judge = judge_indices[rank::world]
    if (
        len(local_gradient) != int(cfg["events_per_worker"])
        or len(local_curvature) != int(cfg["curvature_events"]) // world
        or len(local_judge) != int(cfg["judge_events"]) // world
    ):
        raise RuntimeError("panel sharding is not equal across 16 workers")

    normalization = load_normalization_dict(global_config)
    bundle = load_evenet_model_for_dgpo(
        config=global_config, device=device, checkpoint_path=cfg["source_checkpoint"]
    )
    policy = bundle.model
    # Match the source trainer's order exactly: it applies component freezes
    # after loading weights and before constructing AdamW.  Without this, the
    # configured full-frozen ObjectEncoder creates an extra parameter group.
    policy.train()
    apply_component_freezes(policy, global_config)
    policy.eval()
    interface.verify_policy_loaded(policy, checkpoint["state_dict"])
    assert_dgpo_neutrino_policy_deterministic(policy)
    anchor_state = clone_state(policy.state_dict())
    reference = make_round_reference_model(
        policy, global_config, normalization, device, checkpoint=checkpoint
    )
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    anchor_parameters = tuple(parameter.detach().cpu().clone() for parameter in parameters)

    def restore_anchor() -> None:
        policy.load_state_dict(anchor_state, strict=True)
        policy.eval()
        policy.zero_grad(set_to_none=True)

    def make_batch(indices: Tensor) -> tuple[dict[str, Any], Tensor, Tensor]:
        packed = pool["packed_event"].index_select(0, indices)
        truth = pool["truth"].index_select(0, indices)
        noise_mask = pool["policy_noise_mask"].index_select(0, indices)
        batch = unpack_event_inputs(packed, pool_spec)
        batch["x_invisible"] = truth.reshape(-1, 2, 2)
        batch["x_invisible_mask"] = noise_mask
        return batch_to_device(batch, device), packed, truth

    gradient_batch, _, _ = make_batch(local_gradient)
    sampler = DDIMSampler(device=device)
    restore_anchor()
    torch.manual_seed(int(cfg["gradient_candidate_seed"]) + rank)
    with torch.no_grad():
        gradient_candidates = generate_neutrino_candidates(
            policy, gradient_batch, sampler, K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    gradient_nonfinite = interface._all_gather_object(float(
        (~torch.isfinite(gradient_candidates)).float().mean().cpu()
    ))
    gradient_candidates = torch.nan_to_num(gradient_candidates)
    reward_agg = build_reward_aggregator(policy, device, normalization_dict=normalization)
    reward_source = reward_agg.omnifold_source
    if reward_source is None:
        raise RuntimeError("source runtime did not construct an OmniFold reward")
    reward_source.load_stack_payload(checkpoint["dgpo_omnifold_reward_stack"])
    reward_batch = dict(gradient_batch)
    reward_batch["_dgpo_reward_context"] = "policy_update"
    with torch.no_grad():
        rewards, _ = reward_agg.compute(gradient_candidates, reward_batch)
        advantages, _ = compute_per_event_advantage(
            rewards, estimator=str(cfg["production_contract"]["advantage_estimator"])
        )
    gradient_cfg = {
        **dict(cfg),
        "beta": float(cfg["production_contract"]["beta"]),
        "policy_eval_t_min": float(cfg["production_contract"]["policy_eval_t_min"]),
        "policy_eval_t_max": float(cfg["production_contract"]["policy_eval_t_max"]),
        "time_diagnostic_events_per_worker": 0,
    }
    policy.train()
    total_gradient, gradient_report = gradient_repro._production_gradient(
        cfg=gradient_cfg, policy=policy, reference=reference,
        batch=gradient_batch, candidates=gradient_candidates,
        advantages=advantages, realization=0, rank=rank, world=world,
        device=device, dtype=next(policy.parameters()).dtype,
    )
    raw_gradient_norm = float(total_gradient.double().norm())
    clip_norm = float(cfg["production_contract"]["grad_clip_norm"])
    clip_scale = min(1.0, clip_norm / (raw_gradient_norm + 1.0e-6))
    applied_gradient = total_gradient * clip_scale
    steps_per_epoch = int(cfg["production_contract"]["steps_per_epoch"])
    warmup_factor = float(global_config.options.Training.get(
        "learning_rate_warm_up_factor", 1.0
    ))
    optimizer = build_optimizer(
        policy,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=max(1, math.ceil(warmup_factor * steps_per_epoch)),
        is_rank0=rank == 0,
        lr_schedule=global_config.dgpo.get("lr_schedule"),
    )
    saved_optimizer = checkpoint["dgpo_optimizer_state_dict"]
    saved_native = (
        saved_optimizer.get("optimizer", saved_optimizer)
        if isinstance(saved_optimizer, Mapping)
        else saved_optimizer
    )
    if not isinstance(saved_native, Mapping):
        raise ValueError("checkpoint AdamW payload is not a mapping")
    saved_groups = list(saved_native.get("param_groups", []))
    current_group_sizes = [len(group["params"]) for group in optimizer.param_groups]
    saved_group_sizes = [len(group.get("params", [])) for group in saved_groups]
    if current_group_sizes != saved_group_sizes:
        raise ValueError(
            "source-trainer optimizer structure was not reproduced after component "
            "freezes: current group sizes="
            f"{current_group_sizes}, checkpoint group sizes={saved_group_sizes}"
        )
    optimizer.load_state_dict(saved_optimizer)
    directions, optimizer_report = adamw_counterfactual_directions(
        parameters, optimizer, applied_gradient
    )
    optimizer_report.update({
        "raw_gradient_l2_norm": raw_gradient_norm,
        "grad_clip_norm": clip_norm,
        "grad_clip_scale": clip_scale,
        "grad_clip_active": clip_scale < 1.0,
        "effective_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "effective_group_weight_decay": [
            float(group["weight_decay"]) for group in optimizer.param_groups
        ],
    })
    if rank == 0:
        print(
            "[h4opt01] exact step-21 gradient complete; "
            f"norm={raw_gradient_norm:.6g} clip_scale={clip_scale:.6g}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.summary["phase"] = "match_vp_path_distance"
            metrics = {
                "optimizer_counterfactual/progress/stage": 1.0,
                "optimizer_counterfactual/gradient/l2_norm": raw_gradient_norm,
                "optimizer_counterfactual/gradient/clip_scale": clip_scale,
            }
            for name in ARM_ORDER:
                metrics[f"optimizer_counterfactual/direction/{name}/actual_rms"] = optimizer_report["direction_rms"][name]
                metrics[f"optimizer_counterfactual/direction/{name}/cosine_to_raw"] = optimizer_report["cosine_to_raw_gradient"][name]
            wandb_run.log(metrics)
    del optimizer, gradient_candidates, rewards, advantages
    if device.type == "cuda":
        torch.cuda.empty_cache()

    curvature_batch, _, _ = make_batch(local_curvature)
    restore_anchor()
    torch.manual_seed(int(cfg["curvature_candidate_seed"]) + rank)
    with torch.no_grad():
        curvature_candidates = generate_neutrino_candidates(
            policy, curvature_batch, sampler, K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    curvature_candidates = torch.nan_to_num(curvature_candidates)
    local_b = len(local_curvature)
    timesteps = int(cfg["curvature_timesteps"])
    path_generator = torch.Generator(device=device)
    path_generator.manual_seed(int(cfg["curvature_path_seed"]) + rank)
    path_t, path_normalizer = sample_cosine_vp_path_kl_timesteps(
        timesteps, local_b, total_strata=timesteps, stratum_offset=0,
        device=device, dtype=torch.float32, generator=path_generator,
    )
    n_nu, n_features = curvature_candidates.shape[2:]
    base_eps = torch.randn(
        timesteps, local_b, n_nu, n_features, device=device,
        dtype=next(policy.parameters()).dtype, generator=path_generator,
    )
    path_eps = base_eps.unsqueeze(1).expand(
        timesteps, int(cfg["K"]), local_b, n_nu, n_features
    ).reshape(timesteps * int(cfg["K"]) * local_b, n_nu, n_features)

    def vp_output() -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            values = policy_evaluation_step(
                policy, reference, curvature_batch, curvature_candidates,
                K=int(cfg["K"]), shared_noise=True, device=device,
                dtype=next(policy.parameters()).dtype, t=path_t, eps_rep=path_eps,
                t_min=0.0, t_max=1.0, num_timesteps=timesteps,
            )
        return values[3].detach(), values[5].detach()

    restore_anchor()
    baseline_v, path_mask = vp_output()
    expanded_mask = path_mask.expand_as(baseline_v).to(baseline_v.dtype)
    def vp_distance(output: Tensor) -> float:
        local_sum = ((output - baseline_v).double().square() * expanded_mask.double()).sum()
        totals = torch.stack((local_sum, baseline_v.new_tensor(float(output.shape[0]), dtype=torch.float64)))
        if world > 1:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        return float(0.5 * float(path_normalizer) * totals[0] / totals[1].clamp_min(1.0))

    def symmetric_distance(direction: Tensor, parameter_rms: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for signed, label in ((-parameter_rms, "descent"), (parameter_rms, "reverse")):
            restore_anchor()
            geometry._set_parameter_offset(
                parameters, anchor_parameters, direction, signed_rms=float(signed)
            )
            output, observed_mask = vp_output()
            if not torch.equal(path_mask, observed_mask):
                raise RuntimeError("VP path mask changed during distance matching")
            result[label] = vp_distance(output)
        result["symmetric_mean"] = 0.5 * (result["descent"] + result["reverse"])
        return result

    matched_rms: dict[str, float] = {
        "native_adamw": float(cfg["native_parameter_rms"])
    }
    vp_distances: dict[str, dict[str, float]] = {
        "native_adamw": symmetric_distance(
            directions["native_adamw"], matched_rms["native_adamw"]
        )
    }
    target_distance = vp_distances["native_adamw"]["symmetric_mean"]
    if not math.isfinite(target_distance) or target_distance <= 0.0:
        raise ValueError("native AdamW direction has zero/nonfinite VP distance")
    for name in ARM_ORDER[1:]:
        parameter_rms = float(cfg["native_parameter_rms"])
        for _ in range(int(cfg["distance_match_iterations"])):
            observed = symmetric_distance(directions[name], parameter_rms)
            ratio = observed["symmetric_mean"] / target_distance
            if abs(ratio - 1.0) <= float(cfg["distance_match_tolerance"]):
                break
            parameter_rms *= math.sqrt(1.0 / max(ratio, 1.0e-30))
            parameter_rms = min(
                float(cfg["maximum_parameter_rms"]),
                max(float(cfg["minimum_parameter_rms"]), parameter_rms),
            )
        matched_rms[name] = parameter_rms
        vp_distances[name] = symmetric_distance(directions[name], parameter_rms)
    vp_ratios = {
        name: row["symmetric_mean"] / target_distance
        for name, row in vp_distances.items()
    }
    distance_pass = {
        name: abs(ratio - 1.0) <= float(cfg["distance_match_tolerance"])
        for name, ratio in vp_ratios.items()
    }
    if rank == 0:
        print(
            "[h4opt01] VP matching: "
            + ", ".join(f"{name}={vp_ratios[name]:.4f}" for name in ARM_ORDER),
            flush=True,
        )
        if wandb_run is not None:
            metrics = {"optimizer_counterfactual/progress/stage": 2.0}
            for name in ARM_ORDER:
                metrics[f"optimizer_counterfactual/vp/{name}/parameter_rms"] = matched_rms[name]
                metrics[f"optimizer_counterfactual/vp/{name}/distance"] = vp_distances[name]["symmetric_mean"]
                metrics[f"optimizer_counterfactual/vp/{name}/ratio"] = vp_ratios[name]
                metrics[f"optimizer_counterfactual/vp/{name}/match_passed"] = float(distance_pass[name])
            wandb_run.log(metrics)
            wandb_run.summary["phase"] = "eight_seed_frozen_h4_judge"

    judge_batch, _, judge_truth = make_batch(local_judge)
    truth_candidates = judge_truth.to(device).reshape(1, len(local_judge), 2, 2)
    frozen_judge_batch = dict(judge_batch)
    frozen_judge_batch["_dgpo_reward_context"] = "diagnostic"
    with torch.no_grad():
        truth_logits_local = reward_agg.compute(
            truth_candidates, frozen_judge_batch
        )[0].reshape(-1).cpu()

    def evaluate_judge(
        direction: Tensor | None,
        *,
        descent_sign: int,
        parameter_rms: float,
        seed: int,
    ) -> dict[str, Any]:
        restore_anchor()
        if direction is not None:
            geometry._set_parameter_offset(
                parameters, anchor_parameters, direction,
                signed_rms=-float(descent_sign) * float(parameter_rms),
            )
        torch.manual_seed(int(seed) + rank)
        with torch.no_grad():
            candidates = generate_neutrino_candidates(
                policy, judge_batch, sampler, K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
                parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
            )
        local_nonfinite = float((~torch.isfinite(candidates)).float().mean().cpu())
        nonfinite = interface._all_gather_object(local_nonfinite)
        candidates = torch.nan_to_num(candidates)
        with torch.no_grad():
            generated_logits_local = reward_agg.compute(
                candidates, frozen_judge_batch
            )[0].reshape(-1).cpu()
        gathered_truth = interface._all_gather_object(truth_logits_local)
        gathered_generated = interface._all_gather_object(generated_logits_local)
        if rank != 0:
            return {}
        metrics = interface._classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_generated).reshape(-1)
        )
        return {
            "judge": metrics,
            "judge_auc_gap": float(metrics["auc_gap"]),
            "candidate_nonfinite_fraction": float(np.mean(nonfinite)),
            "candidate_nonfinite_fraction_max_rank": float(np.max(nonfinite)),
        }

    seed_rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(cfg["judge_rollout_seeds"]):
        zero = evaluate_judge(None, descent_sign=0, parameter_rms=0.0, seed=int(seed))
        arms: dict[str, Any] = {}
        for arm_index, name in enumerate(ARM_ORDER):
            minus = evaluate_judge(
                directions[name], descent_sign=-1,
                parameter_rms=matched_rms[name], seed=int(seed),
            )
            plus = evaluate_judge(
                directions[name], descent_sign=1,
                parameter_rms=matched_rms[name], seed=int(seed),
            )
            if rank == 0:
                arms[name] = {
                    "minus": minus,
                    "zero": zero,
                    "plus": plus,
                    "plus_gap_delta": plus["judge_auc_gap"] - zero["judge_auc_gap"],
                    "minus_gap_delta": minus["judge_auc_gap"] - zero["judge_auc_gap"],
                    "plus_beats_minus": plus["judge_auc_gap"] < minus["judge_auc_gap"],
                }
                if wandb_run is not None:
                    wandb_run.log({
                        "optimizer_counterfactual/progress/stage": 3.0,
                        "optimizer_counterfactual/probe/seed_index": seed_index,
                        "optimizer_counterfactual/probe/arm_index": arm_index,
                        f"optimizer_counterfactual/probe/{name}/zero_gap": zero["judge_auc_gap"],
                        f"optimizer_counterfactual/probe/{name}/minus_gap": minus["judge_auc_gap"],
                        f"optimizer_counterfactual/probe/{name}/plus_gap": plus["judge_auc_gap"],
                        f"optimizer_counterfactual/probe/{name}/plus_gap_delta": arms[name]["plus_gap_delta"],
                        f"optimizer_counterfactual/probe/{name}/plus_beats_minus": float(arms[name]["plus_beats_minus"]),
                    })
        if rank == 0:
            seed_rows.append({"seed": int(seed), "zero": zero, "arms": arms})
            print(
                f"[h4opt01] seed {seed_index + 1}/8: zero={zero['judge_auc_gap']:.6f}; "
                + ", ".join(
                    f"{name}=({arms[name]['minus']['judge_auc_gap']:.6f},"
                    f"{arms[name]['plus']['judge_auc_gap']:.6f})"
                    for name in ARM_ORDER
                ),
                flush=True,
            )

    restore_anchor()
    if rank == 0:
        arm_summaries: dict[str, Any] = {}
        for name in ARM_ORDER:
            plus_gaps = [row["arms"][name]["plus"]["judge_auc_gap"] for row in seed_rows]
            minus_gaps = [row["arms"][name]["minus"]["judge_auc_gap"] for row in seed_rows]
            zero_gaps = [row["zero"]["judge_auc_gap"] for row in seed_rows]
            improvements = [zero - plus for zero, plus in zip(zero_gaps, plus_gaps, strict=True)]
            arm_summaries[name] = {
                "plus_gap": paired_mean_interval(plus_gaps),
                "minus_gap": paired_mean_interval(minus_gaps),
                "zero_gap": paired_mean_interval(zero_gaps),
                "improvement_zero_minus_plus": paired_mean_interval(improvements),
                "plus_gap_mean": float(np.mean(plus_gaps)),
                "plus_beats_minus_count": sum(
                    bool(row["arms"][name]["plus_beats_minus"]) for row in seed_rows
                ),
            }
        raw_minus_native_values = [
            row["arms"]["native_adamw"]["plus"]["judge_auc_gap"]
            - row["arms"]["raw_gradient"]["plus"]["judge_auc_gap"]
            for row in seed_rows
        ]
        raw_minus_native = paired_mean_interval(raw_minus_native_values)
        diagnosis = classify_optimizer_result(
            arm_summaries,
            raw_minus_native,
            all_distance_matches=all(distance_pass.values()),
        )
        interface.verify_policy_loaded(policy, checkpoint["state_dict"])
        report = {
            "schema": "c4a91e07-h4-adamw-direction-transfer-v1",
            "source_wandb_run": "ytchou97-university-of-washington/nu2flow-RL/h4xfer01",
            "source_checkpoint": cfg["source_checkpoint"],
            "source_policy_step": int(cfg["expected_policy_step"]),
            "source_reward_round_id": int(cfg["expected_reward_round_id"]),
            "source_raw_audit": cfg["source_raw_audit"],
            "production_contract": cfg["production_contract"],
            "policy_updates": 0,
            "classifier_fits": 0,
            "objective_changed": False,
            "optimizer_state_mutated": False,
            "checkpoint_mutated": False,
            "gradient_events": int(cfg["total_gradient_events"]),
            "gradient": gradient_report,
            "gradient_candidate_nonfinite_fraction": float(np.mean(gradient_nonfinite)),
            "optimizer_directions": optimizer_report,
            "arm_order": list(ARM_ORDER),
            "vp_target": "native_adamw_at_fixed_parameter_rms",
            "vp_target_distance": target_distance,
            "matched_parameter_rms": matched_rms,
            "vp_distances": vp_distances,
            "vp_distance_ratios": vp_ratios,
            "vp_distance_match_passed": distance_pass,
            "judge_is_frozen_step20_installed_h4_ensemble": True,
            "fresh_step20_judge_weights_available_in_checkpoint": False,
            "fresh_to_installed_gradient_cosine_from_h4xfer01": 0.896,
            "gradient_curvature_judge_identities_disjoint": True,
            "common_random_numbers_within_each_seed": True,
            "judge_rollout_seeds": list(cfg["judge_rollout_seeds"]),
            "signed_seed_rows": seed_rows,
            "arm_summaries": arm_summaries,
            "primary_raw_gradient_advantage_over_native": raw_minus_native,
            "diagnosis": diagnosis,
            "scope": (
                "Read-only decomposition of one exact production-sized step-21 "
                "gradient through the restored step-20 AdamW state. Every signed "
                "arm is evaluated at matched VP-path distance by the frozen, saved "
                "step-20 installed four-member H4 ensemble."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(root / "report.json", interface._jsonable(report))
        direction_path = root / "optimizer_directions_fp16.pt"
        if cfg["save_direction_vectors"]:
            replay._exclusive_torch_save(direction_path, {
                "schema": "c4a91e07-h4-adamw-direction-vectors-v1",
                "source_policy_step": int(cfg["expected_policy_step"]),
                "arm_order": list(ARM_ORDER),
                "directions": {name: value.half() for name, value in directions.items()},
            })
        if wandb_run is not None:
            import wandb

            metrics = {
                "optimizer_counterfactual/progress/stage": 4.0,
                "optimizer_counterfactual/primary/raw_minus_native_gap": raw_minus_native["mean"],
                "optimizer_counterfactual/primary/raw_minus_native_ci95_low": raw_minus_native["ci95_low"],
                "optimizer_counterfactual/primary/raw_minus_native_ci95_high": raw_minus_native["ci95_high"],
            }
            for name in ARM_ORDER:
                summary = arm_summaries[name]
                metrics[f"optimizer_counterfactual/summary/{name}/plus_gap_mean"] = summary["plus_gap_mean"]
                metrics[f"optimizer_counterfactual/summary/{name}/improvement_mean"] = summary["improvement_zero_minus_plus"]["mean"]
                metrics[f"optimizer_counterfactual/summary/{name}/plus_beats_minus_count"] = summary["plus_beats_minus_count"]
            wandb_run.log(metrics)
            artifact = wandb.Artifact(
                "c4a91e07-h4-adamw-direction-transfer", type="diagnostic"
            )
            artifact.add_file(str(root / "manifest.json"))
            artifact.add_file(str(root / "report.json"))
            if cfg["upload_direction_vectors"]:
                artifact.add_file(str(direction_path))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete",
                "diagnosis": diagnosis,
                "classifier_fits": 0,
                "policy_updates": 0,
                "optimizer_counterfactual/primary/raw_minus_native_gap": raw_minus_native["mean"],
                "optimizer_counterfactual/primary/raw_minus_native_ci95_low": raw_minus_native["ci95_low"],
                "optimizer_counterfactual/all_vp_matches_passed": float(all(distance_pass.values())),
            })
            wandb_run.finish()
        print(f"[h4opt01] diagnosis={diagnosis}", flush=True)
        print(f"[h4opt01] report={root / 'report.json'}", flush=True)
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
        "h4opt01 preflight passed: h4xfer01 step 20, reward round 1, restored "
        "AdamW/round-reference/installed H4 ensemble; 8192-event exact gradient, five "
        "VP-matched directions, 8 CRN signed judge seeds, zero policy updates "
        "and zero classifier fits.",
        flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"].get("required"):
        if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            raise RuntimeError("h4opt01 requires live W&B")
        if os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled", "dryrun"}:
            raise RuntimeError("WANDB_MODE must allow online logging")
    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is unset; start/source the 16-GPU Ray cluster")
    root = Path(cfg["output_dir"])
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ray.init(
        address=ray_address,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join([
            str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"),
            os.environ.get("PYTHONPATH", ""),
        ])}},
    )
    available_gpus = float(ray.cluster_resources().get("GPU", 0) or 0)
    if available_gpus < int(cfg["workers"]):
        raise RuntimeError(
            f"Ray cluster has {available_gpus:g} GPUs; h4opt01 requires {cfg['workers']}"
        )
    runtime_payload = cfg.pop("_runtime_payload")
    root.mkdir(parents=True, exist_ok=False)
    runtime_path = root / "runtime.yaml"
    with runtime_path.open("x") as handle:
        yaml.safe_dump(runtime_payload, handle, sort_keys=False)
    cfg["runtime_path"] = str(runtime_path)
    replay._exclusive_json(root / "manifest.json", interface._jsonable(cfg))
    TorchTrainer(
        train_loop_per_worker=_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]), use_gpu=True,
            resources_per_worker={"CPU": int(cfg["cpus_per_worker"]), "GPU": 1},
        ),
        run_config=RunConfig(
            name="c4a91e07-h4-adamw-direction-transfer-v1",
            storage_path=str(root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
