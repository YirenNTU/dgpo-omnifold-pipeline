#!/usr/bin/env python3
"""Fixed-policy classifier-to-DGPO reward-interface experiments.

This experiment loads the c4a91e07 ``last.ckpt`` as a fixed policy anchor.
It fits one cold density-ratio increment with one or two seeds and two folds, then
uses the same K=8 logits to compare raw LOO, calibrated LOO, per-event z-score,
and centered-rank advantages.  With ``classifier_trajectory`` configured, it
instead compares BA-LCB threshold checkpoints from the same classifier training
trajectory against the fully trained checkpoint.  It never installs a reward,
steps a policy optimizer, or resumes the checkpoint's training state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_raw_monitor_replay as replay
from ablate_raw_monitor_initialization import clone_state, seeded
from diagnose_residual_weights import score_metrics, score_on_device

from RL.DGPO_neutrino.reward_interface import (
    REWARD_ARMS,
    advantage_metrics,
    fit_scalar_temperature,
    member_ordering_metrics,
    reward_advantage_arms,
    reward_interface_diagnosis,
    vector_concentration,
    vector_cosine,
)


PARTITIONS = ("reward_fit", "early_stop", "calibration", "judge_fit", "final_audit")
CLASSIFIER_ARCHITECTURE_KEYS = frozenset({
    "head_dropout",
    "periodic_pair_features",
    "topology_fourier_embedding",
    "topology_conditioning",
    "visible_pair_rest_frame",
    "topology_max_harmonic",
    "topology_include_theta_pair",
    "topology_hidden_dim",
    "topology_embedding_dim",
    "topology_fusion_hidden_dim",
    "topology_dropout",
    "topology_direct_logit",
    "topology_context_residual_scale",
    "conditional_residual_rank",
})


def _dist_context() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _broadcast_tensor(tensor: Tensor) -> None:
    rank, world = _dist_context()
    if world <= 1:
        return
    if tensor.device.type == "cpu" and torch.distributed.get_backend() == "nccl":
        device = torch.device("cuda", torch.cuda.current_device())
        staged = tensor.to(device)
        torch.distributed.broadcast(staged, src=0)
        tensor.copy_(staged.cpu())
    else:
        torch.distributed.broadcast(tensor, src=0)


def identity_partitions(
    identity: Tensor, *, fractions: Mapping[str, float], seed: int
) -> dict[str, Tensor]:
    """Stable, identity-disjoint partitions whose names and fractions are pinned."""

    if tuple(fractions) != PARTITIONS:
        raise ValueError(f"partition fractions must be ordered as {PARTITIONS}")
    weights = np.asarray([float(fractions[name]) for name in PARTITIONS])
    if np.any(weights <= 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9):
        raise ValueError("partition fractions must be positive and sum to one")
    labels = torch.zeros(len(identity), dtype=torch.int64, device=identity.device)
    rank, world = _dist_context()
    if rank == 0:
        cuts = np.cumsum(weights)
        rows = identity.detach().cpu().float().contiguous().numpy().astype("<f4", copy=True)
        rows[rows == 0] = 0
        _, inverse = np.unique(rows, axis=0, return_inverse=True)
        generator = np.random.default_rng(int(seed))
        unique_assignments = np.searchsorted(cuts, generator.random(int(inverse.max()) + 1))
        assignments = np.minimum(len(PARTITIONS) - 1, unique_assignments[inverse])
        labels.copy_(torch.from_numpy(assignments).to(labels.device))
    if world > 1:
        _broadcast_tensor(labels)
    result = {
        name: torch.nonzero(labels == index, as_tuple=True)[0]
        for index, name in enumerate(PARTITIONS)
    }
    if any(len(indices) < 2 for indices in result.values()):
        raise ValueError("identity partition produced a population with fewer than two events")
    # Duplicate identities must always receive one label.
    if rank == 0:
        seen: dict[bytes, int] = {}
        for index, row in enumerate(rows):
            key = row.tobytes()
            label = int(labels[index])
            if key in seen and seen[key] != label:
                raise RuntimeError("duplicate event identity crossed partitions")
            seen[key] = label
    return result


def _validated_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(settings)
    required = {
        "base_config", "overlay_config", "source_wandb_run",
        "expected_policy_checkpoint", "expected_policy_step", "output_dir",
        "workers", "generation_batch_size", "score_batch_size", "generation_seed",
        "partition_seed", "partition_fractions", "training_seeds", "folds",
        "classifier_fit", "K", "raw_tempering", "gradient_events",
        "gradient_blocks", "gradient_timesteps", "event_gradient_events",
        "signed_probe_events", "signed_step_rms", "response_bins",
        "response_bootstrap_replicates", "wandb", "fold_seed", "judge_seed",
        "candidate_seed", "probe_selection_seed", "gradient_seed", "beta",
        "policy_eval_t_min", "policy_eval_t_max", "signed_rollout_seed",
        "physics_bins", "response_bootstrap_seed",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing reward-interface settings: {missing}")
    if int(cfg["expected_policy_step"]) < 0:
        raise ValueError("expected_policy_step must be nonnegative")
    if str(cfg["source_wandb_run"]).rstrip("/").split("/")[-1] != "c4a91e07":
        raise ValueError("this experiment is pinned to W&B source run c4a91e07")
    for key in (
        "workers", "generation_batch_size", "score_batch_size", "folds", "K",
        "gradient_events", "gradient_blocks", "gradient_timesteps",
        "event_gradient_events", "signed_probe_events", "response_bins",
        "response_bootstrap_replicates",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["folds"] != 2:
        raise ValueError("the predeclared experiment uses exactly two folds")
    if cfg["K"] < 2:
        raise ValueError("DGPO reward-interface diagnostics require K >= 2")
    if not math.isclose(float(cfg["raw_tempering"]), 0.75, abs_tol=1e-12):
        raise ValueError("the matched c4a91e07 raw arm requires tempering=0.75")
    if cfg["gradient_blocks"] < 2 or cfg["gradient_events"] < cfg["gradient_blocks"]:
        raise ValueError("gradient_events must cover at least two nonempty blocks")
    if cfg["event_gradient_events"] > cfg["gradient_events"]:
        raise ValueError("event_gradient_events cannot exceed gradient_events")
    if cfg["gradient_events"] > cfg["signed_probe_events"]:
        raise ValueError("signed_probe_events must include the gradient panel")
    if not math.isfinite(float(cfg["signed_step_rms"])) or float(cfg["signed_step_rms"]) <= 0:
        raise ValueError("signed_step_rms must be finite and positive")
    if not math.isfinite(float(cfg["raw_tempering"])) or float(cfg["raw_tempering"]) <= 0:
        raise ValueError("raw_tempering must be finite and positive")
    seeds = cfg["training_seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(seeds) > 2
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("training_seeds must contain one or two distinct seeds")
    if len(seeds) == 1 and not (
        cfg.get("classifier_trajectory") is not None
        and cfg.get("actionability_sweep") is not None
    ):
        raise ValueError(
            "one classifier repeat is allowed only for the two-fold "
            "classifier-trajectory actionability sweep"
        )
    fractions = cfg["partition_fractions"]
    if not isinstance(fractions, Mapping) or tuple(fractions) != PARTITIONS:
        raise ValueError(f"partition_fractions must be ordered as {PARTITIONS}")
    total = sum(float(fractions[name]) for name in PARTITIONS)
    if any(float(fractions[name]) <= 0 for name in PARTITIONS) or not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("partition fractions must be positive and sum to one")
    if cfg.get("pool_events") is not None and int(cfg["pool_events"]) < cfg["signed_probe_events"]:
        raise ValueError("pool_events is too small for the signed probe")
    fit = dict(cfg["classifier_fit"])
    if int(fit.get("min_steps", 0)) < 1 or int(fit.get("steps", 0)) < int(fit["min_steps"]):
        raise ValueError("classifier_fit requires steps >= min_steps >= 1")
    nested_series = cfg.get("series_arm") is not None
    expected_selection = "loss" if nested_series else "balanced_accuracy"
    if fit.get("checkpoint_selection_metric") != expected_selection:
        raise ValueError(
            "nested residual series must select by held-out validation loss"
            if nested_series
            else (
                "this reward-interface experiment selects classifiers by held-out "
                "balanced_accuracy before separate temperature calibration"
            )
        )
    conditional_fit = cfg.get("conditional_residual_fit")
    if conditional_fit is not None:
        if not isinstance(conditional_fit, Mapping):
            raise ValueError("conditional_residual_fit must be a mapping")
        conditional_fit = {**fit, **dict(conditional_fit)}
        if (
            int(conditional_fit.get("min_steps", 0)) < 1
            or int(conditional_fit.get("steps", 0))
            < int(conditional_fit["min_steps"])
        ):
            raise ValueError(
                "conditional_residual_fit requires steps >= min_steps >= 1"
            )
        if conditional_fit.get("checkpoint_selection_metric") != expected_selection:
            raise ValueError(
                "conditional residual selection must match the base checkpoint metric"
            )
        cfg["conditional_residual_fit"] = conditional_fit
    conditional_residual_only = cfg.get("conditional_residual_only", False)
    if type(conditional_residual_only) is not bool:
        raise ValueError("conditional_residual_only must be boolean")
    cfg["conditional_residual_only"] = conditional_residual_only
    wandb = dict(cfg["wandb"] or {})
    wandb.setdefault("enabled", False)
    if type(wandb["enabled"]) is not bool:
        raise ValueError("wandb.enabled must be boolean")
    cfg["wandb"] = wandb
    for key in ("reward_classifier_overrides", "judge_classifier_overrides"):
        overrides = cfg.get(key, {})
        if not isinstance(overrides, Mapping):
            raise ValueError(f"{key} must be a mapping")
        overrides = dict(overrides)
        unknown = sorted(set(overrides) - CLASSIFIER_ARCHITECTURE_KEYS)
        if unknown:
            raise ValueError(f"{key} contains unsupported keys: {unknown}")
        if (
            "topology_max_harmonic" in overrides
            and int(overrides["topology_max_harmonic"]) < 1
        ):
            raise ValueError(f"{key}.topology_max_harmonic must be >= 1")
        if (
            "head_dropout" in overrides
            and (
                isinstance(overrides["head_dropout"], bool)
                or not isinstance(overrides["head_dropout"], (int, float))
                or not math.isfinite(float(overrides["head_dropout"]))
                or not 0.0 <= float(overrides["head_dropout"]) < 1.0
            )
        ):
            raise ValueError(f"{key}.head_dropout must lie in [0, 1)")
        if (
            "conditional_residual_rank" in overrides
            and (
                type(overrides["conditional_residual_rank"]) is not int
                or int(overrides["conditional_residual_rank"]) < 0
            )
        ):
            raise ValueError(
                f"{key}.conditional_residual_rank must be a nonnegative integer"
            )
        cfg[key] = overrides
    reward_overrides = cfg["reward_classifier_overrides"]
    residual_rank = int(reward_overrides.get("conditional_residual_rank", 0))
    if residual_rank == 0 and (
        conditional_fit is not None or conditional_residual_only
    ):
        raise ValueError(
            "conditional residual training requires a positive reward "
            "conditional_residual_rank"
        )
    if residual_rank > 0 and (
        (conditional_fit is None) == (not conditional_residual_only)
    ):
        raise ValueError(
            "a positive reward conditional_residual_rank requires exactly one of "
            "conditional_residual_fit or conditional_residual_only=true"
        )
    if residual_rank > 0:
        if not bool(reward_overrides.get("periodic_pair_features", False)):
            raise ValueError("conditional residual requires periodic_pair_features=true")
        if any(
            bool(reward_overrides.get(key, False))
            for key in (
                "topology_fourier_embedding",
                "topology_direct_logit",
                "topology_conditioning",
            )
        ):
            raise ValueError(
                "conditional residual must replace Fourier fusion/direct conditioning"
            )
    report_schema = str(
        cfg.get("report_schema", "c4a91e07-h4-weak-classifier-trajectory-v1")
    )
    if not report_schema or any(character.isspace() for character in report_schema):
        raise ValueError("report_schema must be a nonempty whitespace-free string")
    cfg["report_schema"] = report_schema
    trajectory = cfg.get("classifier_trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, Mapping):
            raise ValueError("classifier_trajectory must be a mapping")
        trajectory = dict(trajectory)
        targets = trajectory.get("balanced_accuracy_lcb_targets")
        if (
            not isinstance(targets, list)
            or any(
                not math.isfinite(float(target))
                or not 0.5 < float(target) < 1.0
                for target in targets
            )
        ):
            raise ValueError(
                "classifier_trajectory.balanced_accuracy_lcb_targets must be "
                "a list with values in (0.5, 1)"
            )
        normalized_targets = [float(target) for target in targets]
        if normalized_targets != sorted(set(normalized_targets)):
            raise ValueError(
                "classifier trajectory targets must be unique and increasing"
            )
        confidence_z = float(trajectory.get("confidence_z", 1.96))
        if not math.isfinite(confidence_z) or confidence_z < 0.0:
            raise ValueError("classifier trajectory confidence_z must be nonnegative")
        if trajectory.get("reward_arm", "raw_loo") != "raw_loo":
            raise ValueError(
                "the classifier-depth ablation is pinned to the production raw_loo arm"
            )
        validation_interval = int(fit.get("validation_interval_steps", 0))
        progress_interval = int(fit.get("progress_every_n_steps", 0))
        if (
            progress_interval > 0
            and (
                progress_interval > validation_interval
                or validation_interval % progress_interval != 0
            )
        ):
            raise ValueError(
                "classifier trajectory capture requires every validation step "
                "to invoke progress_callback"
            )
        trajectory.update(
            balanced_accuracy_lcb_targets=normalized_targets,
            confidence_z=confidence_z,
            reward_arm="raw_loo",
        )
        cfg["classifier_trajectory"] = trajectory
    actionability = cfg.get("actionability_sweep")
    if actionability is not None:
        if trajectory is None:
            raise ValueError(
                "actionability_sweep requires classifier_trajectory checkpoints"
            )
        if not isinstance(actionability, Mapping):
            raise ValueError("actionability_sweep must be a mapping")
        actionability = dict(actionability)
        stages = actionability.get("stages")
        available_stages = {
            *(
                _trajectory_stage_label(target)
                for target in trajectory["balanced_accuracy_lcb_targets"]
            ),
            "fully_trained",
        }
        if (
            not isinstance(stages, list)
            or not stages
            or len(set(stages)) != len(stages)
            or any(stage not in available_stages for stage in stages)
        ):
            raise ValueError(
                "actionability_sweep.stages must be unique trajectory stage names"
            )
        radii = actionability.get("step_rms_values")
        if (
            not isinstance(radii, list)
            or not radii
            or any(
                not math.isfinite(float(radius)) or float(radius) <= 0.0
                for radius in radii
            )
        ):
            raise ValueError(
                "actionability_sweep.step_rms_values must be positive finite values"
            )
        normalized_radii = [float(radius) for radius in radii]
        if normalized_radii != sorted(set(normalized_radii)):
            raise ValueError("actionability sweep radii must be unique and increasing")
        rollout_seeds = actionability.get("rollout_seeds")
        if (
            not isinstance(rollout_seeds, list)
            or len(rollout_seeds) < 2
            or len(set(rollout_seeds)) != len(rollout_seeds)
            or any(type(seed) is not int for seed in rollout_seeds)
        ):
            raise ValueError(
                "actionability_sweep.rollout_seeds must contain at least two "
                "distinct integer seeds"
            )
        primary_stage = actionability.get("primary_stage", stages[0])
        if primary_stage not in stages:
            raise ValueError("actionability primary_stage must be one of stages")
        min_plus_zero = float(
            actionability.get("minimum_plus_beats_zero_fraction", 0.75)
        )
        min_plus_minus = float(
            actionability.get("minimum_plus_beats_minus_fraction", 1.0)
        )
        if not (0.0 <= min_plus_zero <= 1.0 and 0.0 <= min_plus_minus <= 1.0):
            raise ValueError("actionability consistency fractions must lie in [0, 1]")
        actionability.update(
            stages=list(stages),
            step_rms_values=normalized_radii,
            rollout_seeds=list(rollout_seeds),
            primary_stage=str(primary_stage),
            minimum_plus_beats_zero_fraction=min_plus_zero,
            minimum_plus_beats_minus_fraction=min_plus_minus,
            bootstrap_seed=int(actionability.get("bootstrap_seed", 20260917)),
            bootstrap_replicates=int(
                actionability.get("bootstrap_replicates", 10000)
            ),
        )
        if actionability["bootstrap_replicates"] < 100:
            raise ValueError("actionability bootstrap_replicates must be at least 100")
        cfg["actionability_sweep"] = actionability
    historical = cfg.get("historical_actionability_reference")
    if historical is not None:
        if not isinstance(historical, Mapping):
            raise ValueError("historical_actionability_reference must be a mapping")
        historical = dict(historical)
        if int(historical.get("source_policy_step", -1)) != int(
            cfg["expected_policy_step"]
        ):
            raise ValueError("historical actionability source step does not match")
        runs = historical.get("saturated_h4_runs")
        if not isinstance(runs, list) or not runs or any(
            not isinstance(run, str) or not run for run in runs
        ):
            raise ValueError("historical saturated_h4_runs must be nonempty run IDs")
        gate_radius = float(historical.get("curriculum_gate_radius", float("nan")))
        if not math.isfinite(gate_radius) or gate_radius <= 0.0:
            raise ValueError("historical curriculum_gate_radius must be positive")
        if actionability is None or gate_radius not in actionability["step_rms_values"]:
            raise ValueError(
                "historical curriculum_gate_radius must be in the actionability grid"
            )
        historical["curriculum_gate_radius"] = gate_radius
        cfg["historical_actionability_reference"] = historical
    return cfg


def _trajectory_stage_label(target: float) -> str:
    """Stable label for a held-out balanced-accuracy LCB threshold."""

    percentage = 100.0 * float(target)
    if not math.isfinite(percentage) or not 50.0 < percentage < 100.0:
        raise ValueError("trajectory target must lie in (0.5, 1)")
    rounded = round(percentage)
    if not math.isclose(percentage, rounded, abs_tol=1.0e-9):
        return f"ba_lcb_{str(float(target)).replace('.', 'p')}"
    return f"ba_lcb_{int(rounded):02d}"


def _balanced_accuracy_lcb(
    balanced_accuracy: float, *, events_per_class: int, confidence_z: float
) -> tuple[float, float]:
    """Conservative normal lower bound for balanced accuracy.

    Each validation population has the same number of truth and generated
    events.  At the maximum-variance Bernoulli point the standard error of the
    mean of the two class recalls is ``sqrt(1 / (8 n))``.
    """

    value = float(balanced_accuracy)
    count = int(events_per_class)
    z = float(confidence_z)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("balanced_accuracy must be finite and in [0, 1]")
    if count < 1:
        raise ValueError("events_per_class must be positive")
    if not math.isfinite(z) or z < 0.0:
        raise ValueError("confidence_z must be finite and nonnegative")
    standard_error = math.sqrt(0.125 / float(count))
    return value - z * standard_error, standard_error


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only preflight for the declared c4a91e07 checkpoint."""

    from train_neutrino_backend import absolutize_default_paths, deep_update, read_yaml

    cfg = _validated_settings(settings)
    base = (ROOT / cfg["base_config"]).resolve(strict=True)
    overlay = (ROOT / cfg["overlay_config"]).resolve(strict=True)
    runtime = absolutize_default_paths(deep_update(read_yaml(base), read_yaml(overlay)), base.parent)
    training = runtime["options"]["Training"]
    training.setdefault("EMA", {}).update(
        replace_model_after_load=False,
        use_for_generation=False,
        use_ema_during_training_eval=False,
    )
    runtime["dgpo"]["auto_resume_from_last"] = False
    runtime["dgpo"]["checkpoint_load_mode"] = "weights_only"
    declared_checkpoint = Path(training["model_checkpoint_load_path"]).expanduser()
    expected_declared = Path(cfg["expected_policy_checkpoint"]).expanduser()
    if (
        declared_checkpoint.absolute() != expected_declared.absolute()
        or declared_checkpoint.name != "last.ckpt"
    ):
        raise ValueError(
            "overlay does not point to the declared c4a91e07 last.ckpt: "
            f"runtime={declared_checkpoint}, expected={expected_declared}"
        )
    checkpoint = declared_checkpoint.resolve(strict=True)
    expected_checkpoint = expected_declared.resolve(strict=True)
    if checkpoint != expected_checkpoint:
        raise ValueError("declared c4a91e07 last.ckpt resolves to an unexpected target")
    recalibration = runtime["dgpo"]["adaptive_omnifold"]["recalibration"]
    h4_contract = {
        "periodic_pair_features": True,
        "topology_fourier_embedding": True,
        "topology_conditioning": False,
        "visible_pair_rest_frame": False,
        "topology_max_harmonic": 4,
        "topology_include_theta_pair": False,
        "topology_direct_logit": False,
    }
    mismatched = {
        key: (recalibration.get(key), expected)
        for key, expected in h4_contract.items()
        if recalibration.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"overlay does not implement the pinned nonlinear H4 classifier: {mismatched}")
    if (
        int(
            cfg["reward_classifier_overrides"].get(
                "conditional_residual_rank", 0
            )
        )
        > 0
        and recalibration.get("asymmetric_attention") is not True
    ):
        raise ValueError(
            "conditional residual requires asymmetric_attention=true so its "
            "visible event factor is candidate-independent"
        )
    source = replay.read_checkpoint(checkpoint)
    if int(source.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError(
            f"wrong c4a91e07 checkpoint step: expected {cfg['expected_policy_step']}, "
            f"found {source.get('global_step')}"
        )
    del source
    backbone = Path(runtime["reward_config"]["omnifold"]["backbone_checkpoint"]).resolve(strict=True)
    normalization = Path(runtime["options"]["Dataset"]["normalization_file"]).resolve(strict=True)
    data_dir = Path(runtime["platform"]["data_parquet_dir"]).resolve(strict=True)
    files = sorted(str(path.resolve()) for path in data_dir.glob("*.parquet"))
    if not files:
        raise ValueError("no processed parquet files for the 10% population")
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    for protected in (checkpoint.parent, backbone.parent, normalization.parent, data_dir):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("diagnostic output overlaps a source directory")
    cfg.update(
        runtime=runtime,
        policy_checkpoint=str(checkpoint),
        backbone=str(backbone),
        files=files,
        output_dir=str(output),
        source_stats={
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in (base, overlay, checkpoint, backbone, normalization)
        },
        data_stats={
            path: [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
            for path in files
        },
    )
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"source metadata changed during experiment: {path}")
    for path, expected in cfg["data_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"dataset metadata changed during experiment: {path}")


def verify_policy_loaded(model: torch.nn.Module, saved_state: Mapping[str, Tensor]) -> None:
    """Verify checkpoint loading directly, without a persistent hash contract."""

    expected = {key.removeprefix("model."): value for key, value in saved_state.items()}
    actual = model.state_dict()
    missing = sorted(set(actual) - set(expected))
    extra = sorted(key for key in set(expected) - set(actual) if not key.startswith("famo.w."))
    different = [
        key for key in actual if key in expected and (
            actual[key].shape != expected[key].shape
            or actual[key].dtype != expected[key].dtype
            or not torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu())
        )
    ]
    if missing or extra or different:
        raise ValueError(
            "loaded policy differs from checkpoint: "
            f"missing={missing[:8]} extra={extra[:8]} different={different[:8]}"
        )


def _unit_weighted_populations(
    condition: Tensor, positive_sample: Tensor, negative_sample: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build a balanced ratio-fit tuple with weights matching every sample axis."""

    if positive_sample.ndim < 2 or negative_sample.ndim < 2:
        raise ValueError("density-ratio samples must include a feature dimension")
    positive_weight = torch.ones(
        positive_sample.shape[:-1], dtype=torch.float32, device=positive_sample.device
    )
    negative_weight = torch.ones(
        negative_sample.shape[:-1], dtype=torch.float32, device=negative_sample.device
    )
    return (
        condition,
        positive_sample,
        positive_weight,
        condition,
        negative_sample,
        negative_weight,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        return _jsonable(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _score_state(builder, packing_spec, state, name, condition, sample, batch_size, device):
    builder.restore_pretrained_body()
    model = builder.make_classifier(packing_spec, name, reset=True).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    scores = score_on_device(model, condition, sample, batch_size).cpu()
    builder.discard_bank(name)
    del model
    return scores


def _score_local_on_device(model, condition, sample, batch_size):
    """Score a process-local panel without invoking distributed population sharding."""

    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _score_in_batches

    device = next(model.parameters()).device

    class Staged(torch.nn.Module):
        def forward(self, event, candidate):
            return model(event.to(device), candidate.to(device))

    return _score_in_batches(Staged(), condition, sample, int(batch_size))


def _score_local_state(builder, packing_spec, state, name, condition, sample, batch_size, device):
    builder.restore_pretrained_body()
    model = builder.make_classifier(packing_spec, name, reset=True).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    scores = _score_local_on_device(model, condition, sample, batch_size).cpu()
    builder.discard_bank(name)
    del model
    return scores


def _classification_metrics(truth_logits: Tensor, generated_logits: Tensor) -> dict[str, float]:
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _weighted_binary_score_metrics

    generated = generated_logits.reshape(-1)
    bce, accuracy, auc = _weighted_binary_score_metrics(
        truth_logits.reshape(-1), generated, torch.ones_like(generated)
    )
    return {
        "bce": bce,
        "balanced_accuracy": accuracy,
        "auc": auc,
        "auc_gap": abs(auc - 0.5),
    }


def _all_gather_object(value: Any) -> list[Any]:
    _, world = _dist_context()
    if world == 1:
        return [value]
    result = [None] * world
    torch.distributed.all_gather_object(result, value)
    return result


def _gather_numpy_dict(local: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    gathered = _all_gather_object({key: np.asarray(value) for key, value in local.items()})
    if _dist_context()[0] != 0:
        return {}
    keys = sorted(set().union(*(item.keys() for item in gathered)))
    return {
        key: np.concatenate([np.asarray(item[key]) for item in gathered if key in item], axis=0)
        for key in keys
    }


def response_matrix_metrics(
    truth: np.ndarray, candidates: np.ndarray, *, bins: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Truth-normalized response summaries for the four generated target coordinates."""

    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 3 or candidates.shape[0] != len(truth) or candidates.shape[2] != 4:
        raise ValueError("response candidates must have shape (B,K,4)")
    labels = ("tau_a_delta_theta", "tau_a_delta_phi", "tau_b_delta_theta", "tau_b_delta_phi")
    report: dict[str, Any] = {}
    event_scores: dict[str, np.ndarray] = {}
    for component, label in enumerate(labels):
        target = truth[:, component].copy()
        predicted = candidates[:, :, component].copy()
        if label.endswith("phi"):
            target = np.arctan2(np.sin(target), np.cos(target))
            predicted = np.arctan2(np.sin(predicted), np.cos(predicted))
            edges = np.linspace(-math.pi, math.pi, bins + 1)
        else:
            finite = target[np.isfinite(target)]
            edges = np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))
            if len(np.unique(edges)) != len(edges):
                lo, hi = np.nanpercentile(finite, [0.1, 99.9])
                edges = np.linspace(lo, hi if hi > lo else lo + 1.0e-6, bins + 1)
        truth_bin = np.clip(np.digitize(target, edges[1:-1]), 0, bins - 1)
        candidate_bin = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
        matrix = np.zeros((bins, bins), dtype=np.float64)
        for column in range(bins):
            rows = candidate_bin[truth_bin == column].reshape(-1)
            if rows.size:
                matrix[:, column] = np.bincount(rows, minlength=bins) / rows.size
        offsets = candidate_bin - truth_bin[:, None]
        event_score = np.mean(np.abs(offsets), axis=1)
        valid_columns = matrix.sum(axis=0) > 0
        diagonal = float(np.mean(np.diag(matrix)[valid_columns]))
        near = []
        for column in np.flatnonzero(valid_columns):
            near.append(matrix[max(0, column - 1):min(bins, column + 2), column].sum())
        per_bin_bias, per_bin_rms = [], []
        for column in np.flatnonzero(valid_columns):
            values = offsets[truth_bin == column].reshape(-1)
            per_bin_bias.append(float(np.mean(values)))
            per_bin_rms.append(float(np.sqrt(np.mean(values**2))))
        report[label] = {
            "truth_normalized_matrix": matrix.tolist(),
            "diagonal_fraction": diagonal,
            "near_diagonal_fraction": float(np.mean(near)),
            "mean_abs_bin_offset": float(np.mean(event_score)),
            "rms_bin_offset": float(np.sqrt(np.mean(offsets.astype(np.float64) ** 2))),
            "equal_truth_bin_abs_bias": float(np.mean(np.abs(per_bin_bias))),
            "equal_truth_bin_resolution": float(np.mean(per_bin_rms)),
            "populated_truth_bins": int(np.sum(valid_columns)),
        }
        event_scores[label] = event_score
    return report, event_scores


def _paired_bootstrap_delta(
    current: np.ndarray, baseline: np.ndarray, *, replicates: int, seed: int
) -> dict[str, float]:
    delta = np.asarray(current, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.ndim != 1 or not len(delta):
        raise ValueError("paired bootstrap needs non-empty one-dimensional event values")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        draws[index] = float(np.mean(delta[rng.integers(0, len(delta), len(delta))]))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"mean_delta": float(np.mean(delta)), "ci95_low": float(lo), "ci95_high": float(hi)}


def _select_batch(batch: Mapping[str, Any], indices: Tensor) -> dict[str, Any]:
    size = int(next(value for value in batch.values() if isinstance(value, Tensor)).shape[0])
    result = {}
    for key, value in batch.items():
        if isinstance(value, Tensor) and value.ndim and int(value.shape[0]) == size:
            result[key] = value.index_select(0, indices.to(value.device))
        else:
            result[key] = value
    return result


def _flat_gradient(loss: Tensor, parameters: tuple[Tensor, ...], *, retain_graph: bool) -> Tensor:
    values = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient.detach()).float().reshape(-1)
        for parameter, gradient in zip(parameters, values)
    ])


def _loss_for_advantage(L_cur: Tensor, L_ref: Tensor, advantage: Tensor, beta: float) -> Tensor:
    from RL.DGPO_neutrino.dgpo_utils import build_dgpo_loss

    if L_cur.ndim == 2:
        return build_dgpo_loss(L_cur, L_ref, advantage, beta_dgpo=beta, K=len(advantage))[0]
    return torch.stack([
        build_dgpo_loss(current, reference, advantage, beta_dgpo=beta, K=len(advantage))[0]
        for current, reference in zip(L_cur, L_ref)
    ]).mean()


def _gradient_audit(
    *, cfg, policy, reference, batch, candidates, advantage_sets, global_positions, device, dtype
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    from RL.DGPO_neutrino.dgpo_trainer import policy_evaluation_step

    rank, world = _dist_context()
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("source policy exposes no trainable DGPO parameters")
    total_parameters = sum(parameter.numel() for parameter in parameters)
    keys = list(advantage_sets)
    aggregate = {
        key: torch.zeros(total_parameters, dtype=torch.float32)
        for key in keys if rank == 0 or key.startswith("primary/")
    }
    half = {
        key: [torch.zeros(total_parameters), torch.zeros(total_parameters)]
        for key in keys if rank == 0
    }
    block_norms = {key: [] for key in keys} if rank == 0 else {}
    panel_events = int(cfg["gradient_events"])
    blocks = int(cfg["gradient_blocks"])
    local_keep = global_positions < panel_events
    positions = global_positions[local_keep]
    gradient_batch = _select_batch(batch, torch.nonzero(local_keep, as_tuple=True)[0])
    gradient_candidates = candidates[:, local_keep.to(candidates.device)]
    gradient_advantages = {
        key: value[:, local_keep.to(value.device)] for key, value in advantage_sets.items()
    }
    for block in range(blocks):
        start = (panel_events * block) // blocks
        stop = (panel_events * (block + 1)) // blocks
        local_index = torch.nonzero((positions >= start) & (positions < stop), as_tuple=True)[0]
        block_size = stop - start
        vectors: dict[str, Tensor] = {}
        if len(local_index):
            local_batch = _select_batch(gradient_batch, local_index)
            local_candidates = gradient_candidates[:, local_index.to(gradient_candidates.device)]
            torch.manual_seed(int(cfg["gradient_seed"]) + block * 100003 + rank)
            values = policy_evaluation_step(
                policy, reference, local_batch, local_candidates,
                K=int(cfg["K"]), shared_noise=True, device=device, dtype=dtype,
                t_min=float(cfg["policy_eval_t_min"]),
                t_max=float(cfg["policy_eval_t_max"]),
                num_timesteps=int(cfg["gradient_timesteps"]),
            )
            L_cur, L_ref = values[:2]
            for key_index, key in enumerate(keys):
                loss = _loss_for_advantage(
                    L_cur,
                    L_ref,
                    gradient_advantages[key][
                        :, local_index.to(gradient_advantages[key].device)
                    ],
                    float(cfg["beta"]),
                )
                vectors[key] = _flat_gradient(
                    loss, parameters, retain_graph=key_index < len(keys) - 1
                ) * (len(local_index) / float(block_size))
            del values, L_cur, L_ref
        for key in keys:
            vector = vectors.get(key)
            if vector is None:
                vector = torch.zeros(total_parameters, device=device, dtype=torch.float32)
            if world > 1:
                torch.distributed.all_reduce(vector, op=torch.distributed.ReduceOp.SUM)
            cpu = vector.cpu()
            if key in aggregate:
                aggregate[key].add_(cpu, alpha=block_size / float(panel_events))
            if rank == 0:
                half[key][0 if block < blocks // 2 else 1].add_(cpu, alpha=block_size / float(panel_events))
                block_norms[key].append(float(cpu.norm()))
        del vectors
    report: dict[str, Any] = {}
    if rank == 0:
        for key in keys:
            full = aggregate[key]
            norms = np.asarray(block_norms[key], dtype=np.float64)
            report[key] = {
                "gradient_norm": float(full.norm()),
                "split_half_cosine": vector_cosine(half[key][0], half[key][1]),
                "block_norm_mean": float(norms.mean()),
                "block_norm_cv": float(norms.std() / max(norms.mean(), 1.0e-30)),
                "largest_block_norm_fraction": float(norms.max() / max(norms.sum(), 1.0e-30)),
            }
        if len(cfg["training_seeds"]) >= 2:
            for arm in REWARD_ARMS:
                left_key = f"seed_{cfg['training_seeds'][0]}/{arm}"
                right_key = f"seed_{cfg['training_seeds'][1]}/{arm}"
                if left_key in aggregate and right_key in aggregate:
                    report[f"seed_agreement/{arm}"] = {
                        "gradient_cosine": vector_cosine(
                            aggregate[left_key], aggregate[right_key]
                        )
                    }
        primary_arms = [arm for arm in REWARD_ARMS if f"primary/{arm}" in aggregate]
        for left_index, left in enumerate(primary_arms):
            for right in primary_arms[left_index + 1:]:
                report[f"arm_agreement/{left}_vs_{right}"] = {
                    "gradient_cosine": vector_cosine(
                        aggregate[f"primary/{left}"], aggregate[f"primary/{right}"]
                    )
                }
    return report, {key: value for key, value in aggregate.items() if key.startswith("primary/")}


def _event_gradient_concentration(
    *, cfg, policy, reference, batch, candidates, advantages, global_positions, device, dtype
) -> dict[str, Any]:
    from RL.DGPO_neutrino.dgpo_trainer import policy_evaluation_step

    rank, _ = _dist_context()
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    limit = int(cfg["event_gradient_events"])
    local_indices = torch.nonzero(global_positions < limit, as_tuple=True)[0]
    local_norms = {arm: [] for arm in REWARD_ARMS}
    for local_index in local_indices.tolist():
        ix = torch.tensor([local_index], device=device, dtype=torch.long)
        event_batch = _select_batch(batch, ix)
        event_candidates = candidates[:, ix]
        torch.manual_seed(int(cfg["gradient_seed"]) + int(global_positions[local_index]) * 100003)
        values = policy_evaluation_step(
            policy, reference, event_batch, event_candidates,
            K=int(cfg["K"]), shared_noise=True, device=device, dtype=dtype,
            t_min=float(cfg["policy_eval_t_min"]), t_max=float(cfg["policy_eval_t_max"]),
            num_timesteps=int(cfg["gradient_timesteps"]),
        )
        L_cur, L_ref = values[:2]
        for arm_index, arm in enumerate(REWARD_ARMS):
            loss = _loss_for_advantage(L_cur, L_ref, advantages[arm][:, ix], float(cfg["beta"]))
            gradient = _flat_gradient(loss, parameters, retain_graph=arm_index < len(REWARD_ARMS) - 1)
            local_norms[arm].append(float(gradient.norm()))
        del values
    gathered = _all_gather_object(local_norms)
    if rank != 0:
        return {}
    return {
        arm: vector_concentration(torch.tensor([
            value for shard in gathered for value in shard[arm]
        ], dtype=torch.float64))
        for arm in REWARD_ARMS
    }


def _assign_direction(
    parameters: tuple[Tensor, ...], anchor: tuple[Tensor, ...], direction: Tensor,
    *, sign: int, epsilon_rms: float
) -> float:
    if sign not in (-1, 0, 1):
        raise ValueError("signed direction sign must be -1, 0, or +1")
    rms = float(direction.double().square().mean().sqrt())
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("cannot normalize a zero/nonfinite policy gradient")
    scale = float(epsilon_rms) / rms
    offset = 0
    with torch.no_grad():
        for parameter, base in zip(parameters, anchor):
            count = parameter.numel()
            update = direction[offset:offset + count].reshape_as(parameter).to(parameter.device, parameter.dtype)
            # +epsilon is the optimizer/descent direction for the DGPO loss.
            parameter.copy_(base.to(parameter.device, parameter.dtype) - float(sign) * scale * update)
            offset += count
    if offset != direction.numel():
        raise RuntimeError("flat direction does not match trainable policy parameters")
    return scale


def _paired_signal_metrics(left_logits: Tensor, right_logits: Tensor) -> dict[str, float]:
    """Compare two classifiers on an identical ``(K, B)`` candidate panel."""

    from RL.DGPO_neutrino.dgpo_utils import compute_per_event_advantage
    from RL.DGPO_neutrino.reward_interface import centered_candidate_rank

    if left_logits.shape != right_logits.shape or left_logits.ndim != 2:
        raise ValueError("paired candidate logits must have the same (K, B) shape")
    if int(left_logits.shape[0]) < 2:
        raise ValueError("paired candidate metrics require K >= 2")
    left_advantage = compute_per_event_advantage(
        left_logits, estimator="leave_one_out_unscaled"
    )[0]
    right_advantage = compute_per_event_advantage(
        right_logits, estimator="leave_one_out_unscaled"
    )[0]
    left_rank = centered_candidate_rank(left_logits)
    right_rank = centered_candidate_rank(right_logits)
    comparable = (left_advantage != 0) & (right_advantage != 0)
    if bool(comparable.any()):
        sign_agreement = float(
            (
                torch.sign(left_advantage[comparable])
                == torch.sign(right_advantage[comparable])
            ).double().mean()
        )
    else:
        sign_agreement = float("nan")
    return {
        "advantage_cosine": vector_cosine(left_advantage, right_advantage),
        "rank_cosine": vector_cosine(left_rank, right_rank),
        "candidate_sign_agreement": sign_agreement,
        "winner_agreement": float(
            (left_logits.argmax(0) == right_logits.argmax(0)).double().mean()
        ),
    }


def _trajectory_diagnosis(
    stages: list[str],
    *,
    signal_alignment: Mapping[str, Mapping[str, float]],
    gradient_alignment: Mapping[str, float],
    signed_probes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Predeclared interpretation of weak versus fully trained reward directions."""

    rows: dict[str, dict[str, Any]] = {}
    for stage in stages:
        signed = signed_probes[stage]
        zero_gap = float(signed["zero"]["judge_auc_gap"])
        plus_gap = float(signed["plus"]["judge_auc_gap"])
        minus_gap = float(signed["minus"]["judge_auc_gap"])
        advantage_cosine = float(signal_alignment[stage]["advantage_cosine"])
        gradient_cosine = float(gradient_alignment[stage])
        rows[stage] = {
            "advantage_cosine_with_judge": advantage_cosine,
            "gradient_cosine_with_judge": gradient_cosine,
            "plus_minus_zero_judge_auc_gap": {
                "minus": minus_gap,
                "zero": zero_gap,
                "plus": plus_gap,
            },
            "plus_delta_judge_auc_gap": plus_gap - zero_gap,
            "minus_delta_judge_auc_gap": minus_gap - zero_gap,
            "direction_pass": bool(
                advantage_cosine > 0.0
                and gradient_cosine > 0.0
                and plus_gap < zero_gap
                and plus_gap < minus_gap
            ),
        }
    early = [stage for stage in stages if stage != "fully_trained"]
    passing_early = [stage for stage in early if rows[stage]["direction_pass"]]
    full_pass = rows["fully_trained"]["direction_pass"]
    if passing_early and not full_pass:
        finding = "early_stopping_repairs_classifier_to_dgpo_direction"
    elif passing_early and full_pass:
        finding = "h4_direction_works_but_early_stopping_is_not_required_locally"
    elif not passing_early and full_pass:
        finding = "fully_trained_h4_outperforms_weak_classifier_direction"
    else:
        finding = "weak_classifier_early_stopping_does_not_repair_direction"
    best_stage = min(
        stages, key=lambda stage: rows[stage]["plus_delta_judge_auc_gap"]
    )
    return {
        "finding": finding,
        "best_local_stage": best_stage,
        "stages": rows,
        "scope": (
            "One normalized local policy step on a fixed event/candidate/noise panel. "
            "A passing result licenses a short frozen-reward pilot; it is not a "
            "claim of long-run DGPO closure."
        ),
    }


def _bootstrap_mean_interval(
    values: list[float], *, seed: int, replicates: int, confidence: float = 0.90
) -> dict[str, float]:
    """Deterministic percentile interval over paired rollout-seed effects."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("actionability bootstrap needs at least two finite values")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(replicates), len(array)))
    draws = array[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "mean": float(array.mean()),
        "ci90_low": float(low),
        "ci90_high": float(high),
    }


def _actionability_sweep_diagnosis(
    sweep: Mapping[str, Any], settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Select a reproducible local radius without using physics metrics."""

    stage_rows: dict[str, dict[str, Any]] = {}
    bootstrap_seed = int(settings["bootstrap_seed"])
    replicates = int(settings["bootstrap_replicates"])
    for stage_index, stage in enumerate(settings["stages"]):
        radius_rows: dict[str, Any] = {}
        for radius_index, radius in enumerate(settings["step_rms_values"]):
            key = f"{float(radius):.12g}"
            seed_rows = sweep[stage][key]["seeds"]
            plus_zero = [
                float(row["plus_judge_auc_gap"])
                - float(row["zero_judge_auc_gap"])
                for row in seed_rows
            ]
            plus_minus = [
                float(row["plus_judge_auc_gap"])
                - float(row["minus_judge_auc_gap"])
                for row in seed_rows
            ]
            plus_zero_stats = _bootstrap_mean_interval(
                plus_zero,
                seed=bootstrap_seed + 100 * stage_index + radius_index,
                replicates=replicates,
            )
            plus_minus_stats = _bootstrap_mean_interval(
                plus_minus,
                seed=bootstrap_seed + 10000 + 100 * stage_index + radius_index,
                replicates=replicates,
            )
            plus_beats_zero = float(np.mean(np.asarray(plus_zero) < 0.0))
            plus_beats_minus = float(np.mean(np.asarray(plus_minus) < 0.0))
            reliable = bool(
                plus_zero_stats["mean"] < 0.0
                and plus_zero_stats["ci90_high"] < 0.0
                and plus_beats_zero
                >= float(settings["minimum_plus_beats_zero_fraction"])
                and plus_beats_minus
                >= float(settings["minimum_plus_beats_minus_fraction"])
            )
            radius_rows[key] = {
                "step_rms": float(radius),
                "rollout_seeds": len(seed_rows),
                "plus_minus_zero": {
                    "plus_minus_zero": plus_zero_stats,
                    "plus_minus_minus": plus_minus_stats,
                    "plus_beats_zero_fraction": plus_beats_zero,
                    "plus_beats_minus_fraction": plus_beats_minus,
                },
                "reliable": reliable,
            }
        stage_rows[stage] = radius_rows

    primary_stage = str(settings["primary_stage"])
    primary_rows = list(stage_rows[primary_stage].values())
    reliable = [row for row in primary_rows if row["reliable"]]
    selected = min(
        reliable,
        key=lambda row: row["plus_minus_zero"]["plus_minus_zero"]["mean"],
        default=None,
    )
    base = min(primary_rows, key=lambda row: row["step_rms"])
    if primary_stage == "ba_lcb_55":
        findings = (
            "ba55_actionability_not_replicated",
            "larger_ba55_radius_supported",
            "base_ba55_radius_preferred",
        )
    else:
        findings = (
            f"{primary_stage}_actionability_not_replicated",
            f"larger_{primary_stage}_radius_supported",
            f"base_{primary_stage}_radius_preferred",
        )
    if selected is None:
        finding = findings[0]
    elif selected["step_rms"] > base["step_rms"]:
        finding = findings[1]
    else:
        finding = findings[2]
    return {
        "finding": finding,
        "primary_stage": primary_stage,
        "selected_step_rms": (
            None if selected is None else float(selected["step_rms"])
        ),
        "base_step_rms": float(base["step_rms"]),
        "stages": stage_rows,
        "selection_rule": (
            "Among radii whose 90% bootstrap upper bound for plus-minus-zero is "
            "negative and whose paired seed consistency passes both declared "
            "fractions, select the largest mean H4-gap improvement."
        ),
        "scope": (
            "Read-only matched-distance local probes. A passing larger radius "
            "licenses a short two-fold one-repeat DGPO pilot; it is not a "
            "closed-loop classifier-closure claim."
        ),
    }


def _load_classifier_state(path: str | Path) -> dict[str, Tensor]:
    """Load a classifier-only state file without accepting arbitrary objects."""

    try:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older cluster torch
        state = torch.load(Path(path), map_location="cpu")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"invalid classifier state: {path}")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in state.items()):
        raise ValueError(f"classifier state contains non-tensor entries: {path}")
    return dict(state)


def _run_classifier_trajectory_analysis(
    *, cfg, rank, world, root, wandb_run, builder, pool, populations,
    partitions, member_records, trajectory_member_states, judge, judge_config,
    judge_diagnostics, policy, anchor_model_state, sampler, local_pool,
    local_batch, local_positions, order, fixed_candidates, sample_bk4, device,
    global_config,
) -> None:
    """Compare weak and fully trained H4 checkpoints on one fixed policy panel."""

    from RL.DGPO_neutrino.diagnostics.ztautau_validation import (
        build_ztautau_validation_metrics,
        collect_ztautau_validation_arrays,
    )
    from RL.DGPO_neutrino.model_utils import load_evenet_model_for_dgpo
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    trajectory_cfg = cfg["classifier_trajectory"]
    stage_order = [
        _trajectory_stage_label(target)
        for target in trajectory_cfg["balanced_accuracy_lcb_targets"]
    ] + ["fully_trained"]
    expected_members = len(cfg["training_seeds"]) * int(cfg["folds"])
    for stage in stage_order:
        if len(trajectory_member_states[stage]) != expected_members:
            raise RuntimeError(
                f"trajectory stage {stage} has {len(trajectory_member_states[stage])} "
                f"members; expected {expected_members}"
            )
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "score_classifier_trajectory"

    calibration = populations["calibration"]
    audit = populations["final_audit"]
    stage_member_logits: dict[str, Tensor] = {}
    classification_report: dict[str, Any] = {}
    stage_temperatures: dict[str, float] = {}
    for stage in stage_order:
        calibration_truth, calibration_generated = [], []
        audit_truth, audit_generated = [], []
        candidate_logits = []
        for member_index, item in enumerate(trajectory_member_states[stage]):
            state = _load_classifier_state(item["path"])
            model_name = f"trajectory_{stage}_{member_index}"
            builder.restore_pretrained_body()
            model = builder.make_classifier(
                pool.packing_spec, model_name, reset=True
            ).to(device)
            model.load_state_dict(state, strict=True)
            model.eval()
            calibration_truth.append(
                score_on_device(
                    model, calibration.packed_event, calibration.truth,
                    int(cfg["score_batch_size"]),
                ).cpu()
            )
            calibration_generated.append(
                score_on_device(
                    model, calibration.packed_event, calibration.candidates,
                    int(cfg["score_batch_size"]),
                ).cpu()
            )
            audit_truth.append(
                score_on_device(
                    model, audit.packed_event, audit.truth,
                    int(cfg["score_batch_size"]),
                ).cpu()
            )
            audit_generated.append(
                score_on_device(
                    model, audit.packed_event, audit.candidates,
                    int(cfg["score_batch_size"]),
                ).cpu()
            )
            candidate_logits.append(
                _score_local_on_device(
                    model, local_pool.packed_event, sample_bk4,
                    int(cfg["score_batch_size"]),
                ).reshape(len(local_pool.truth), int(cfg["K"])).T.cpu()
            )
            builder.discard_bank(model_name)
            del state, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        calibration_truth_ensemble = torch.stack(calibration_truth).mean(0)
        calibration_generated_ensemble = torch.stack(calibration_generated).mean(0)
        temperature = fit_scalar_temperature(
            calibration_truth_ensemble, calibration_generated_ensemble
        )
        stage_temperatures[stage] = float(temperature["temperature"])
        audit_truth_ensemble = torch.stack(audit_truth).mean(0)
        audit_generated_ensemble = torch.stack(audit_generated).mean(0)
        classification_report[stage] = {
            "temperature": temperature,
            "calibration_uncalibrated": _classification_metrics(
                calibration_truth_ensemble, calibration_generated_ensemble
            ),
            "final_audit_uncalibrated": _classification_metrics(
                audit_truth_ensemble, audit_generated_ensemble
            ),
            "final_audit_calibrated": _classification_metrics(
                audit_truth_ensemble / temperature["temperature"],
                audit_generated_ensemble / temperature["temperature"],
            ),
        }
        stage_member_logits[stage] = torch.stack(candidate_logits)
        if rank == 0 and wandb_run is not None:
            audit_metrics = classification_report[stage][
                "final_audit_uncalibrated"
            ]
            wandb_run.log({
                f"trajectory/{stage}/audit_auc": audit_metrics["auc"],
                f"trajectory/{stage}/audit_balanced_accuracy": audit_metrics[
                    "balanced_accuracy"
                ],
                f"trajectory/{stage}/calibration_temperature": (
                    stage_temperatures[stage]
                ),
            })

    judge_logits_local = _score_local_on_device(
        judge, local_pool.packed_event, sample_bk4, int(cfg["score_batch_size"])
    ).reshape(len(local_pool.truth), int(cfg["K"])).T.cpu()
    judge_final_audit = _classification_metrics(
        score_on_device(
            judge, audit.packed_event, audit.truth, int(cfg["score_batch_size"])
        ).cpu(),
        score_on_device(
            judge, audit.packed_event, audit.candidates,
            int(cfg["score_batch_size"]),
        ).cpu(),
    )
    stage_advantages: dict[str, Tensor] = {}
    stage_ordering_report: dict[str, Any] = {}
    signal_alignment: dict[str, Any] = {}
    complete_stage_members: dict[str, Tensor] = {}
    complete_stage_ensembles: dict[str, Tensor] = {}
    gathered_judge_logits = _all_gather_object(judge_logits_local)
    complete_judge_logits = (
        torch.cat(gathered_judge_logits, dim=1) if rank == 0 else None
    )
    for stage in stage_order:
        local_members = stage_member_logits[stage]
        local_ensemble = local_members.mean(0)
        stage_advantages[stage] = reward_advantage_arms(
            local_ensemble,
            temperature=stage_temperatures[stage],
            raw_tempering=float(cfg["raw_tempering"]),
        )["raw_loo"].to(device)
        gathered_members = _all_gather_object(local_members)
        if rank == 0:
            complete_members = torch.cat(gathered_members, dim=2)
            complete_ensemble = complete_members.mean(0)
            complete_stage_members[stage] = complete_members
            complete_stage_ensembles[stage] = complete_ensemble
            stage_ordering_report[stage] = member_ordering_metrics(
                complete_members
            )
            signal_alignment[stage] = _paired_signal_metrics(
                complete_ensemble, complete_judge_logits
            )
            if wandb_run is not None:
                wandb_run.log({
                    f"trajectory/{stage}/candidate_advantage_cosine_judge": (
                        signal_alignment[stage]["advantage_cosine"]
                    ),
                    f"trajectory/{stage}/candidate_rank_cosine_judge": (
                        signal_alignment[stage]["rank_cosine"]
                    ),
                    f"trajectory/{stage}/candidate_winner_agreement_judge": (
                        signal_alignment[stage]["winner_agreement"]
                    ),
                })

    cross_stage_alignment: dict[str, Any] = {}
    same_member_alignment_with_full: dict[str, Any] = {}
    if rank == 0:
        for left_index, left in enumerate(stage_order):
            for right in stage_order[left_index + 1:]:
                cross_stage_alignment[f"{left}_vs_{right}"] = (
                    _paired_signal_metrics(
                        complete_stage_ensembles[left],
                        complete_stage_ensembles[right],
                    )
                )
        full_members = complete_stage_members["fully_trained"]
        for stage in stage_order[:-1]:
            member_rows = [
                _paired_signal_metrics(
                    complete_stage_members[stage][member_index],
                    full_members[member_index],
                )
                for member_index in range(expected_members)
            ]
            same_member_alignment_with_full[stage] = {
                metric: {
                    "mean": float(np.nanmean([row[metric] for row in member_rows])),
                    "min": float(np.nanmin([row[metric] for row in member_rows])),
                    "members": [float(row[metric]) for row in member_rows],
                }
                for metric in member_rows[0]
            }

    judge_advantage_local = reward_advantage_arms(
        judge_logits_local,
        temperature=1.0,
        raw_tempering=float(cfg["raw_tempering"]),
    )["raw_loo"].to(device)
    advantage_sets = {
        **{f"primary/{stage}": value for stage, value in stage_advantages.items()},
        "primary/independent_judge": judge_advantage_local,
    }
    advantage_report: dict[str, Any] = {}
    for key, local_advantage in advantage_sets.items():
        gathered = _all_gather_object(local_advantage.cpu())
        if rank == 0:
            advantage_report[key] = advantage_metrics(torch.cat(gathered, dim=1))

    # Save the exact logits used for every stage comparison.  This makes the
    # candidate-ordering claim independently reproducible without regenerating
    # any diffusion samples.
    panel_shards = _all_gather_object({
        "position": local_positions.cpu(),
        "audit_index": order[rank::world].cpu(),
        "packed_event": local_pool.packed_event.cpu(),
        "truth": local_pool.truth.cpu(),
        "policy_noise_mask": local_pool.policy_noise_mask.cpu(),
        "candidates_kb22": fixed_candidates.cpu(),
        "stage_member_logits": {
            stage: value.cpu() for stage, value in stage_member_logits.items()
        },
        "judge_logits": judge_logits_local,
    })
    if rank == 0:
        positions_complete = torch.cat([item["position"] for item in panel_shards])
        permutation = torch.argsort(positions_complete)
        replay._exclusive_torch_save(root / "trajectory_fixed_k8_panel.pt", {
            "position": torch.arange(len(permutation)),
            "final_audit_index": torch.cat([
                item["audit_index"] for item in panel_shards
            ])[permutation],
            "packed_event": torch.cat([
                item["packed_event"] for item in panel_shards
            ])[permutation],
            "truth": torch.cat([item["truth"] for item in panel_shards])[permutation],
            "policy_noise_mask": torch.cat([
                item["policy_noise_mask"] for item in panel_shards
            ])[permutation],
            "candidates_kb22": torch.cat([
                item["candidates_kb22"] for item in panel_shards
            ], dim=1)[:, permutation],
            "stage_member_logits": {
                stage: torch.cat([
                    item["stage_member_logits"][stage] for item in panel_shards
                ], dim=2)[:, :, permutation]
                for stage in stage_order
            },
            "judge_logits": torch.cat([
                item["judge_logits"] for item in panel_shards
            ], dim=1)[:, permutation],
            "trajectory_members": trajectory_member_states,
            "candidate_seed": int(cfg["candidate_seed"]),
        })

    reference_bundle = load_evenet_model_for_dgpo(
        config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"]
    )
    reference = reference_bundle.model.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    verify_policy_loaded(reference, anchor_model_state)
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "classifier_trajectory_gradient_audit"
    gradient_report, gradients = _gradient_audit(
        cfg=cfg, policy=policy, reference=reference, batch=local_batch,
        candidates=fixed_candidates, advantage_sets=advantage_sets,
        global_positions=local_positions, device=device,
        dtype=next(policy.parameters()).dtype,
    )
    gradient_alignment: dict[str, float] = {}
    cross_stage_gradient_alignment: dict[str, float] = {}
    if rank == 0:
        judge_gradient = gradients["primary/independent_judge"]
        for stage in stage_order:
            alignment = vector_cosine(
                gradients[f"primary/{stage}"], judge_gradient
            )
            gradient_alignment[stage] = alignment
            gradient_report[f"alignment/{stage}_vs_independent_judge"] = {
                "gradient_cosine": alignment
            }
            if wandb_run is not None:
                wandb_run.log({
                    f"trajectory/{stage}/policy_gradient_cosine_judge": alignment,
                    f"trajectory/{stage}/policy_gradient_norm": gradient_report[
                        f"primary/{stage}"
                    ]["gradient_norm"],
                    f"trajectory/{stage}/policy_gradient_split_half_cosine": (
                        gradient_report[f"primary/{stage}"]["split_half_cosine"]
                    ),
                })
        for left_index, left in enumerate(stage_order):
            for right in stage_order[left_index + 1:]:
                cross_stage_gradient_alignment[f"{left}_vs_{right}"] = (
                    vector_cosine(
                        gradients[f"primary/{left}"],
                        gradients[f"primary/{right}"],
                    )
                )

    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    anchor = tuple(parameter.detach().cpu().clone() for parameter in parameters)
    truth_judge_local = _score_local_on_device(
        judge, local_pool.packed_event, local_pool.truth,
        int(cfg["score_batch_size"]),
    ).cpu()

    baseline_candidates = fixed_candidates

    def evaluate_signed(
        label: str,
        direction: Tensor | None,
        sign: int,
        *,
        epsilon_rms: float | None = None,
        rollout_seed: int | None = None,
        classifier_gap_only: bool = False,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        epsilon = (
            float(cfg["signed_step_rms"])
            if epsilon_rms is None
            else float(epsilon_rms)
        )
        seed = (
            int(cfg["signed_rollout_seed"])
            if rollout_seed is None
            else int(rollout_seed)
        )
        if direction is None:
            scale = 0.0
            with torch.no_grad():
                for parameter, base in zip(parameters, anchor):
                    parameter.copy_(base.to(parameter.device, parameter.dtype))
        else:
            scale = _assign_direction(
                parameters, anchor, direction, sign=sign,
                epsilon_rms=epsilon,
            )
        torch.manual_seed(seed + rank)
        with torch.no_grad():
            candidates = generate_neutrino_candidates(
                policy, local_batch, sampler, K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
                device=device,
                parallel_chains=int(
                    global_config.dgpo.get("rollout_parallel_chains", 1)
                ),
            )
        generated = candidates.permute(1, 0, 2, 3).reshape(
            len(local_pool.truth), int(cfg["K"]), 4
        ).cpu()
        judge_logits = _score_local_on_device(
            judge, local_pool.packed_event, generated,
            int(cfg["score_batch_size"]),
        ).cpu()
        gathered_truth = _all_gather_object(truth_judge_local)
        gathered_generated = _all_gather_object(judge_logits)
        arrays = None
        if not classifier_gap_only:
            event_valid = torch.ones(
                len(local_pool.truth), device=device, dtype=torch.bool
            )
            arrays = collect_ztautau_validation_arrays(
                candidates, baseline_candidates, local_batch, event_valid
            )
            arrays = _gather_numpy_dict(arrays)
        if rank != 0:
            return {}, {}
        judge_metrics = _classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_generated).reshape(-1)
        )
        result = {
            "label": label,
            "direction_sign": sign,
            "signed_step_rms": 0.0 if sign == 0 else epsilon,
            "rollout_seed": seed,
            "flat_gradient_scale": scale,
            "judge": judge_metrics,
            "judge_auc_gap": judge_metrics["auc_gap"],
        }
        if classifier_gap_only:
            return result, {}
        assert arrays is not None
        physics = build_ztautau_validation_metrics(
            arrays, val_k=int(cfg["K"]), tarp_config={"enabled": False},
            metrics_config={
                "enabled": True, "bins": int(cfg["physics_bins"]),
                "candidate_index": 0,
            },
            include_images=False,
        )
        response, event_scores = response_matrix_metrics(
            arrays["_tarp_truth"], arrays["_tarp_candidates"],
            bins=int(cfg["response_bins"]),
        )
        result.update({
            "physics": physics,
            "response": response,
            "response_mean_abs_bin_offset": float(np.mean([
                values["mean_abs_bin_offset"] for values in response.values()
            ])),
        })
        return result, event_scores

    signed_report: dict[str, Any] = {}
    actionability_decision: dict[str, Any] | None = None
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "classifier_trajectory_signed_probes"
    actionability = cfg.get("actionability_sweep")
    if actionability is not None:
        sweep: dict[str, dict[str, Any]] = {
            stage: {
                f"{float(radius):.12g}": {
                    "step_rms": float(radius), "seeds": []
                }
                for radius in actionability["step_rms_values"]
            }
            for stage in actionability["stages"]
        }
        for seed_index, rollout_seed in enumerate(actionability["rollout_seeds"]):
            zero_result, _ = evaluate_signed(
                f"seed_{rollout_seed}/zero",
                None,
                0,
                rollout_seed=int(rollout_seed),
                classifier_gap_only=True,
            )
            for stage in actionability["stages"]:
                direction = gradients[f"primary/{stage}"]
                for radius in actionability["step_rms_values"]:
                    radius_key = f"{float(radius):.12g}"
                    minus_result, _ = evaluate_signed(
                        f"seed_{rollout_seed}/{stage}/rms_{radius_key}/minus",
                        direction,
                        -1,
                        epsilon_rms=float(radius),
                        rollout_seed=int(rollout_seed),
                        classifier_gap_only=True,
                    )
                    plus_result, _ = evaluate_signed(
                        f"seed_{rollout_seed}/{stage}/rms_{radius_key}/plus",
                        direction,
                        +1,
                        epsilon_rms=float(radius),
                        rollout_seed=int(rollout_seed),
                        classifier_gap_only=True,
                    )
                    if rank == 0:
                        row = {
                            "rollout_seed": int(rollout_seed),
                            "zero_judge_auc_gap": float(
                                zero_result["judge_auc_gap"]
                            ),
                            "minus_judge_auc_gap": float(
                                minus_result["judge_auc_gap"]
                            ),
                            "plus_judge_auc_gap": float(
                                plus_result["judge_auc_gap"]
                            ),
                        }
                        sweep[stage][radius_key]["seeds"].append(row)
                        if wandb_run is not None:
                            metric_root = (
                                f"actionability/{stage}/rms_{radius_key}"
                            )
                            wandb_run.log({
                                "actionability/rollout_index": seed_index,
                                f"{metric_root}/zero_gap": row[
                                    "zero_judge_auc_gap"
                                ],
                                f"{metric_root}/minus_gap": row[
                                    "minus_judge_auc_gap"
                                ],
                                f"{metric_root}/plus_gap": row[
                                    "plus_judge_auc_gap"
                                ],
                                f"{metric_root}/plus_delta": (
                                    row["plus_judge_auc_gap"]
                                    - row["zero_judge_auc_gap"]
                                ),
                            })
        if rank == 0:
            actionability_decision = _actionability_sweep_diagnosis(
                sweep, actionability
            )
            signed_report = {
                "protocol": "multi_seed_matched_radius_classifier_gap_only",
                "sweep": sweep,
                "decision": actionability_decision,
            }
    else:
        with torch.no_grad():
            for parameter, base in zip(parameters, anchor):
                parameter.copy_(base.to(parameter.device, parameter.dtype))
        torch.manual_seed(int(cfg["signed_rollout_seed"]) + rank)
        with torch.no_grad():
            baseline_candidates = generate_neutrino_candidates(
                policy, local_batch, sampler, K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
                device=device,
                parallel_chains=int(
                    global_config.dgpo.get("rollout_parallel_chains", 1)
                ),
            )
        zero_result, zero_events = evaluate_signed("zero", None, 0)
        for stage_index, stage in enumerate(stage_order):
            direction = gradients[f"primary/{stage}"]
            minus_result, minus_events = evaluate_signed(
                f"{stage}/minus", direction, -1
            )
            plus_result, plus_events = evaluate_signed(
                f"{stage}/plus", direction, +1
            )
            if rank == 0:
                bootstrap = {}
                for component_index, component in enumerate(zero_events):
                    bootstrap[f"plus_vs_zero/{component}"] = (
                        _paired_bootstrap_delta(
                            plus_events[component], zero_events[component],
                            replicates=int(cfg["response_bootstrap_replicates"]),
                            seed=(
                                int(cfg["response_bootstrap_seed"])
                                + 10 * stage_index + component_index
                            ),
                        )
                    )
                    bootstrap[f"minus_vs_zero/{component}"] = (
                        _paired_bootstrap_delta(
                            minus_events[component], zero_events[component],
                            replicates=int(cfg["response_bootstrap_replicates"]),
                            seed=(
                                int(cfg["response_bootstrap_seed"]) + 1000
                                + 10 * stage_index + component_index
                            ),
                        )
                    )
                signed_report[stage] = {
                    "minus": minus_result,
                    "zero": zero_result,
                    "plus": plus_result,
                    "paired_response_bootstrap": bootstrap,
                }
                if wandb_run is not None:
                    wandb_run.log({
                        f"trajectory/{stage}/minus_judge_auc_gap": (
                            minus_result["judge_auc_gap"]
                        ),
                        f"trajectory/{stage}/zero_judge_auc_gap": (
                            zero_result["judge_auc_gap"]
                        ),
                        f"trajectory/{stage}/plus_judge_auc_gap": (
                            plus_result["judge_auc_gap"]
                        ),
                        f"trajectory/{stage}/minus_response_offset": (
                            minus_result["response_mean_abs_bin_offset"]
                        ),
                        f"trajectory/{stage}/zero_response_offset": (
                            zero_result["response_mean_abs_bin_offset"]
                        ),
                        f"trajectory/{stage}/plus_response_offset": (
                            plus_result["response_mean_abs_bin_offset"]
                        ),
                    })
    with torch.no_grad():
        for parameter, base in zip(parameters, anchor):
            parameter.copy_(base.to(parameter.device, parameter.dtype))

    if rank == 0:
        decision = (
            actionability_decision
            if actionability_decision is not None
            else _trajectory_diagnosis(
                stage_order,
                signal_alignment=signal_alignment,
                gradient_alignment=gradient_alignment,
                signed_probes=signed_report,
            )
        )
        historical_reference = cfg.get("historical_actionability_reference")
        if actionability_decision is not None and historical_reference:
            gate_radius = float(historical_reference["curriculum_gate_radius"])
            gate_key = f"{gate_radius:.12g}"
            primary_stage = str(actionability_decision["primary_stage"])
            gate_row = actionability_decision["stages"][primary_stage].get(gate_key)
            if gate_row is None:
                raise ValueError(
                    "historical curriculum gate radius is absent from the "
                    "actionability sweep"
                )
            decision = {
                **decision,
                "historical_comparison": historical_reference,
                "curriculum_gate_radius": gate_radius,
                "curriculum_gate_passed": bool(gate_row["reliable"]),
            }
        verify_policy_loaded(policy, anchor_model_state)
        report = {
            "schema": cfg["report_schema"],
            "source_wandb_run": cfg.get("source_wandb_run"),
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0,
            "reward_installs": 0,
            "optimizer_state_resumed": False,
            "fixed_policy": True,
            "policy_unchanged": True,
            "fixed_candidate_panel": True,
            "reward_classifier_overrides": cfg["reward_classifier_overrides"],
            "judge_classifier_overrides": cfg["judge_classifier_overrides"],
            "conditional_residual_fit": cfg.get("conditional_residual_fit"),
            "conditional_residual_only": bool(
                cfg.get("conditional_residual_only", False)
            ),
            "series_arm": cfg.get("series_arm"),
            "historical_actionability_reference": cfg.get(
                "historical_actionability_reference"
            ),
            "trajectory_protocol": {
                **trajectory_cfg,
                "selection": (
                    "first checkpoint whose raw held-out balanced-accuracy 95% "
                    "lower confidence bound reaches the declared target"
                ),
                "same_training_trajectory": True,
                "reward_arm": "raw_loo",
            },
            "partitions": {
                name: int(len(indices)) for name, indices in partitions.items()
            },
            "classifier_members": member_records,
            "trajectory_members": trajectory_member_states,
            "classification": classification_report,
            "independent_judge": {
                "seed": int(cfg["judge_seed"]),
                "fit_config": asdict(judge_config),
                "fit_diagnostics": _jsonable(asdict(judge_diagnostics)),
                "final_audit": judge_final_audit,
            },
            "member_ordering": stage_ordering_report,
            "signal_alignment_with_independent_judge": signal_alignment,
            "cross_stage_signal_alignment": cross_stage_alignment,
            "same_member_alignment_with_fully_trained": (
                same_member_alignment_with_full
            ),
            "advantages": advantage_report,
            "gradients": gradient_report,
            "gradient_alignment_with_independent_judge": gradient_alignment,
            "cross_stage_gradient_alignment": cross_stage_gradient_alignment,
            "signed_probes": signed_report,
            "decision": decision,
        }
        verify_sources(cfg)
        replay._exclusive_json(root / "trajectory_report.json", _jsonable(report))
        if wandb_run is not None:
            import wandb

            numeric: dict[str, float] = {}
            for stage in stage_order:
                audit_metrics = classification_report[stage][
                    "final_audit_uncalibrated"
                ]
                numeric.update({
                    f"trajectory/{stage}/audit_auc": audit_metrics["auc"],
                    f"trajectory/{stage}/audit_balanced_accuracy": audit_metrics[
                        "balanced_accuracy"
                    ],
                    f"trajectory/{stage}/candidate_advantage_cosine_judge": (
                        signal_alignment[stage]["advantage_cosine"]
                    ),
                    f"trajectory/{stage}/candidate_rank_cosine_judge": (
                        signal_alignment[stage]["rank_cosine"]
                    ),
                    f"trajectory/{stage}/policy_gradient_cosine_judge": (
                        gradient_alignment[stage]
                    ),
                })
                if actionability is None:
                    numeric.update({
                        f"trajectory/{stage}/minus_judge_auc_gap": (
                            signed_report[stage]["minus"]["judge_auc_gap"]
                        ),
                        f"trajectory/{stage}/zero_judge_auc_gap": (
                            signed_report[stage]["zero"]["judge_auc_gap"]
                        ),
                        f"trajectory/{stage}/plus_judge_auc_gap": (
                            signed_report[stage]["plus"]["judge_auc_gap"]
                        ),
                        f"trajectory/{stage}/minus_response_offset": (
                            signed_report[stage]["minus"][
                                "response_mean_abs_bin_offset"
                            ]
                        ),
                        f"trajectory/{stage}/zero_response_offset": (
                            signed_report[stage]["zero"][
                                "response_mean_abs_bin_offset"
                            ]
                        ),
                        f"trajectory/{stage}/plus_response_offset": (
                            signed_report[stage]["plus"][
                                "response_mean_abs_bin_offset"
                            ]
                        ),
                    })
                if stage != "fully_trained":
                    numeric.update({
                        f"trajectory/{stage}/ensemble_advantage_cosine_full": (
                            cross_stage_alignment[
                                f"{stage}_vs_fully_trained"
                            ]["advantage_cosine"]
                        ),
                        f"trajectory/{stage}/same_member_advantage_cosine_full_mean": (
                            same_member_alignment_with_full[stage][
                                "advantage_cosine"
                            ]["mean"]
                        ),
                        f"trajectory/{stage}/policy_gradient_cosine_full": (
                            cross_stage_gradient_alignment[
                                f"{stage}_vs_fully_trained"
                            ]
                        ),
                    })
            if actionability_decision is not None:
                for stage, radius_rows in actionability_decision["stages"].items():
                    for radius_key, row in radius_rows.items():
                        stats = row["plus_minus_zero"]
                        root_key = f"actionability_summary/{stage}/rms_{radius_key}"
                        numeric.update({
                            f"{root_key}/mean_plus_delta": stats[
                                "plus_minus_zero"
                            ]["mean"],
                            f"{root_key}/ci90_low": stats[
                                "plus_minus_zero"
                            ]["ci90_low"],
                            f"{root_key}/ci90_high": stats[
                                "plus_minus_zero"
                            ]["ci90_high"],
                            f"{root_key}/plus_beats_zero_fraction": stats[
                                "plus_beats_zero_fraction"
                            ],
                            f"{root_key}/plus_beats_minus_fraction": stats[
                                "plus_beats_minus_fraction"
                            ],
                            f"{root_key}/reliable": float(row["reliable"]),
                        })
            wandb_run.log(numeric)
            artifact = wandb.Artifact(
                f"{cfg['report_schema']}-report",
                type="diagnostic",
            )
            artifact.add_file(str(root / "trajectory_report.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.summary["phase"] = "complete"
            wandb_run.summary["decision"] = decision["finding"]
            if "curriculum_gate_passed" in decision:
                wandb_run.summary["curriculum_gate_passed"] = decision[
                    "curriculum_gate_passed"
                ]
            if "best_local_stage" in decision:
                wandb_run.summary["best_local_stage"] = decision[
                    "best_local_stage"
                ]
            if "selected_step_rms" in decision:
                wandb_run.summary["selected_step_rms"] = decision[
                    "selected_step_rms"
                ]
            wandb_run.finish()
        print(
            f"Classifier-trajectory report: {root / 'trajectory_report.json'}",
            flush=True,
        )


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import (
        _materialize_adaptive_omnifold_pool,
        batch_to_device,
    )
    from RL.DGPO_neutrino.diagnostics.ztautau_validation import (
        build_ztautau_validation_metrics,
        collect_ztautau_validation_arrays,
    )
    from RL.DGPO_neutrino.model_utils import load_evenet_model_for_dgpo, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EvenetAdapterModelBuilder,
        _identity_crossfit_splits,
        unpack_event_inputs,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio
    from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    device = ray.train.torch.get_device()
    root = Path(cfg["output_dir"])
    if world != int(cfg["workers"]):
        raise ValueError("unexpected Ray worker count")
    global_config.load_yaml(cfg["runtime_path"])
    wandb_run = None
    if rank == 0 and cfg["wandb"].get("enabled"):
        import wandb

        wandb_run = wandb.init(
            entity=cfg["wandb"].get("entity"),
            project=cfg["wandb"].get("project"),
            name=cfg["wandb"].get("name"),
            id=cfg["wandb"].get("id"),
            group=cfg["wandb"].get("group"),
            resume=cfg["wandb"].get("resume"),
            config=_jsonable({
                key: value for key, value in cfg.items()
                if key not in {"runtime", "source_stats", "data_stats"}
            }),
            tags=cfg["wandb"].get("tags"),
            job_type="diagnostic",
        )
        # W&B's transport step also advances for summaries, trajectory markers,
        # and signed probes.  Give every plotted series its own coordinate so
        # those unrelated writes cannot distort the x axis.
        if cfg.get("actionability_sweep") is not None:
            wandb_run.define_metric("actionability/rollout_index", hidden=True)
            wandb_run.define_metric(
                "actionability/*",
                step_metric="actionability/rollout_index",
                step_sync=False,
            )
        wandb_run.summary["phase"] = "materialize_fixed_k1_pool"
    if rank == 0:
        print("[reward-interface] materializing fixed K=1 pool", flush=True)
    torch.set_float32_matmul_precision(str(global_config.dgpo.get("float32_matmul_precision", "medium")))
    source = replay.read_checkpoint(cfg["policy_checkpoint"])
    if int(source.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("c4a91e07 checkpoint step changed after preflight")
    bundle = load_evenet_model_for_dgpo(
        config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"]
    )
    policy = bundle.model.eval()
    verify_policy_loaded(policy, source["state_dict"])
    anchor_model_state = clone_state(policy.state_dict())
    del source
    sampler = DDIMSampler(device=device)
    adaptive = resolve_adaptive_config(global_config.dgpo)
    pool = _materialize_adaptive_omnifold_pool(
        ray.train.get_dataset_shard("pool"),
        {"batch_size": int(cfg["generation_batch_size"]), "prefetch_batches": 1},
        model=policy,
        sampler=sampler,
        device=device,
        world_size=world,
        rank=rank,
        quota_events=cfg.get("pool_events"),
        num_ddim_steps=int(global_config.dgpo.validation_num_ddim_steps),
        seed=int(cfg["generation_seed"]),
        include_pairwise_context=adaptive.periodic_pair_features_enabled,
        include_visible_pair_rest_frame=adaptive.visible_pair_rest_frame_enabled,
        collect_policy_noise_mask=True,
    )
    if rank == 0:
        print(f"[reward-interface] fixed K=1 pool ready: events={pool.n_events}", flush=True)
        if wandb_run is not None:
            wandb_run.summary["phase"] = "fit_reward_members"
    if pool.policy_noise_mask is None:
        raise RuntimeError("fixed policy pool is missing the policy noise mask")
    partitions = identity_partitions(
        pool.identity_inputs,
        fractions=cfg["partition_fractions"],
        seed=int(cfg["partition_seed"]),
    )
    populations = {name: pool.select(indices) for name, indices in partitions.items()}
    if rank == 0:
        replay._exclusive_torch_save(root / "fixed_k1_pool.pt", {
            "packing_spec": pool.packing_spec.to_dict(),
            "packed_event": pool.packed_event,
            "truth": pool.truth,
            "candidate": pool.candidates,
            "policy_noise_mask": pool.policy_noise_mask,
            "partitions": {key: value.cpu() for key, value in partitions.items()},
            "generation_seed": int(cfg["generation_seed"]),
        })
    rec = dict(global_config.dgpo.adaptive_omnifold.recalibration)
    rec.update(cfg["reward_classifier_overrides"])
    builder_keys = set(__import__("inspect").signature(EvenetAdapterModelBuilder).parameters)
    builder = EvenetAdapterModelBuilder(
        config=global_config,
        normalization_dict=load_normalization_dict(global_config),
        checkpoint_path=cfg["backbone"],
        device=device,
        **{key: value for key, value in rec.items() if key in builder_keys},
    )

    fit_pool = populations["reward_fit"]
    early = populations["early_stop"]
    calibration = populations["calibration"]
    audit = populations["final_audit"]
    member_records, member_states = [], []
    trajectory_cfg = cfg.get("classifier_trajectory")
    trajectory_member_states: dict[str, list[dict[str, Any]]] = {}
    if trajectory_cfg is not None:
        trajectory_member_states = {
            _trajectory_stage_label(target): []
            for target in trajectory_cfg["balanced_accuracy_lcb_targets"]
        }
        trajectory_member_states["fully_trained"] = []
    validation = _unit_weighted_populations(
        early.packed_event, early.truth, early.candidates
    )
    early_validation_independent_events = early.n_events
    if trajectory_cfg is not None:
        early_validation_independent_events = int(np.unique(
            early.identity_inputs.detach().cpu().float().contiguous().numpy(),
            axis=0,
        ).shape[0])
    for seed in cfg["training_seeds"]:
        folds = _identity_crossfit_splits(
            fit_pool.identity_inputs, folds=2, seed=int(cfg["fold_seed"])
        )
        for fold_index, (train_index, oof_index) in enumerate(folds, start=1):
            train = fit_pool.select(train_index)
            oof = fit_pool.select(oof_index)
            fit_config = build_fit_config(
                cfg["classifier_fit"], n_train=train.n_events,
                n_validation=early.n_events, max_batch_population=train.n_events,
            )
            conditional_fit_config = None
            direct_residual_only = bool(
                cfg.get("conditional_residual_only", False)
            )
            conditional_rank = int(
                cfg["reward_classifier_overrides"].get(
                    "conditional_residual_rank", 0
                )
            )
            if cfg.get("conditional_residual_fit") is not None:
                conditional_fit_config = build_fit_config(
                    cfg["conditional_residual_fit"],
                    n_train=train.n_events,
                    n_validation=early.n_events,
                    max_batch_population=train.n_events,
                )
            name = f"reward_s{seed}_f{fold_index}"
            live_curve_name = (
                "reward"
                if len(cfg["training_seeds"]) == 1
                else f"reward_s{seed}"
            )
            log_live_curve = fold_index == 1
            matched_old_arm = bool(
                conditional_fit_config is None
                and not direct_residual_only
                and dict(cfg.get("series_arm") or {}).get("name") == "old"
            )
            if rank == 0 and wandb_run is not None and log_live_curve:
                curve_names = (
                    [f"{live_curve_name}_rank{conditional_rank}_residual"]
                    if direct_residual_only
                    else
                    [
                        f"{live_curve_name}_old_base"
                        if matched_old_arm
                        else live_curve_name
                    ]
                    if conditional_fit_config is None
                    else [
                        f"{live_curve_name}_old_base",
                        f"{live_curve_name}_rank{conditional_rank}_residual",
                    ]
                )
                for curve_name in curve_names:
                    live_step = f"classifier_fit/{curve_name}/step"
                    wandb_run.define_metric(live_step, hidden=True)
                    wandb_run.define_metric(
                        f"classifier_fit/{curve_name}/*",
                        step_metric=live_step,
                        step_sync=False,
                    )
            builder.restore_pretrained_body()
            with seeded(int(seed) + fold_index, device):
                model = builder.make_classifier(pool.packing_spec, name, reset=True).to(device)
            captured_trajectory: dict[str, dict[str, Any]] = {}
            fit_phase = {
                "name": (
                    f"rank{conditional_rank}_residual"
                    if direct_residual_only
                    else (
                        "old_base"
                        if conditional_fit_config is not None or matched_old_arm
                        else "fit"
                    )
                ),
                "capture": conditional_fit_config is None,
            }

            def progress(
                row, *, _name=name, _model=model,
                _captured=captured_trajectory,
                _log_live_curve=log_live_curve,
                _live_curve_name=live_curve_name,
            ):
                captured_now: list[tuple[str, dict[str, Any]]] = []
                if (
                    trajectory_cfg is not None
                    and bool(fit_phase["capture"])
                    and bool(row.get("validation_evaluated", 0.0))
                ):
                    lcb, standard_error = _balanced_accuracy_lcb(
                        row["validation_balanced_accuracy"],
                        events_per_class=early_validation_independent_events,
                        confidence_z=trajectory_cfg["confidence_z"],
                    )
                    for target in trajectory_cfg["balanced_accuracy_lcb_targets"]:
                        stage = _trajectory_stage_label(target)
                        if stage in _captured or lcb < float(target):
                            continue
                        path = root / f"classifier_{_name}_{stage}.pt"
                        metadata = {
                            "stage": stage,
                            "target": float(target),
                            "step": int(row["step"]),
                            "validation_balanced_accuracy": float(
                                row["validation_balanced_accuracy"]
                            ),
                            "validation_balanced_accuracy_lcb": float(lcb),
                            "validation_balanced_accuracy_standard_error": float(
                                standard_error
                            ),
                            "validation_independent_events_per_class": (
                                early_validation_independent_events
                            ),
                            "path": str(path),
                        }
                        # The distributed fit keeps every replica synchronized.
                        # Save one rank-zero copy immediately so the weak state
                        # cannot be overwritten by later classifier updates.
                        if rank == 0:
                            replay._exclusive_torch_save(
                                path, clone_state(_model.state_dict())
                            )
                        _captured[stage] = metadata
                        captured_now.append((stage, metadata))
                if rank == 0:
                    curve_suffix = (
                        "" if fit_phase["name"] == "fit"
                        else f"_{fit_phase['name']}"
                    )
                    curve = root / f"classifier_{_name}{curve_suffix}.jsonl"
                    with curve.open("a") as handle:
                        handle.write(json.dumps(_jsonable(row), allow_nan=False) + "\n")
                    if wandb_run is not None and _log_live_curve:
                        curve_name = (
                            _live_curve_name
                            if fit_phase["name"] == "fit"
                            else f"{_live_curve_name}_{fit_phase['name']}"
                        )
                        metric_root = f"classifier_fit/{curve_name}"
                        live_metrics = {
                            f"{metric_root}/step": row["step"],
                            f"{metric_root}/training_loss": row["training_loss"],
                            f"{metric_root}/training_balanced_accuracy": row[
                                "training_balanced_accuracy"
                            ],
                        }
                        for metric in (
                            "training_auc", "gradient_norm", "gradient_clipped"
                        ):
                            if metric in row:
                                live_metrics[f"{metric_root}/{metric}"] = row[metric]
                        if bool(row.get("validation_evaluated", 0.0)):
                            live_metrics.update({
                                f"{metric_root}/validation_loss": row[
                                    "validation_loss"
                                ],
                                f"{metric_root}/validation_balanced_accuracy": row[
                                    "validation_balanced_accuracy"
                                ],
                            })
                            if "validation_auc" in row:
                                live_metrics[f"{metric_root}/validation_auc"] = row[
                                    "validation_auc"
                                ]
                        wandb_run.log(live_metrics)
                        for stage, metadata in captured_now:
                            wandb_run.log({
                                f"trajectory/{_name}/{stage}/captured": 1,
                                f"trajectory/{_name}/{stage}/step": metadata["step"],
                                f"trajectory/{_name}/{stage}/validation_ba": metadata[
                                    "validation_balanced_accuracy"
                                ],
                                f"trajectory/{_name}/{stage}/validation_ba_lcb": metadata[
                                    "validation_balanced_accuracy_lcb"
                                ],
                            })
                    if bool(row.get("validation_evaluated", 0.0)):
                        print(
                            f"[reward-interface] {name}/{fit_phase['name']} "
                            f"step={int(row['step'])} "
                            f"train_ba={row['training_balanced_accuracy']:.4f} "
                            f"val_ba={row['validation_balanced_accuracy']:.4f} "
                            f"val_bce={row['validation_loss']:.6f}",
                            flush=True,
                        )
                    for stage, metadata in captured_now:
                        print(
                            f"[reward-interface] captured {_name} {stage} "
                            f"at step={metadata['step']} "
                            f"val_ba={metadata['validation_balanced_accuracy']:.4f} "
                            f"lcb={metadata['validation_balanced_accuracy_lcb']:.4f}",
                            flush=True,
                        )

            fit_populations = _unit_weighted_populations(
                train.packed_event, train.truth, train.candidates
            )
            stage_diagnostics: dict[str, Any] = {}
            if conditional_fit_config is not None:
                model.configure_conditional_residual_training_stage(1)
            elif direct_residual_only:
                # ``reset=True`` creates a zero-output legacy path. Freeze it
                # immediately and fit only the bilinear conditional H2 term.
                # The resulting score is a direct low-rank density-ratio model,
                # with no learned old-base nuisance stage.
                model.configure_conditional_residual_training_stage(2)
            with seeded(int(seed) + fold_index, device):
                base_diagnostics = fit_density_ratio(
                    model,
                    *fit_populations,
                    fit_config, int(seed) + fold_index,
                    validation=validation,
                    progress_callback=progress,
                )
            diagnostics = base_diagnostics
            stage_diagnostics[fit_phase["name"]] = _jsonable(
                asdict(base_diagnostics)
            )
            if conditional_fit_config is not None:
                base_path = root / f"classifier_{name}_old_base.pt"
                if rank == 0:
                    replay._exclusive_torch_save(
                        base_path, clone_state(model.state_dict())
                    )
                fit_phase.update(
                    name=f"rank{conditional_rank}_residual", capture=True
                )
                model.configure_conditional_residual_training_stage(2)
                with seeded(int(seed) + 10000 + fold_index, device):
                    diagnostics = fit_density_ratio(
                        model,
                        *fit_populations,
                        conditional_fit_config,
                        int(seed) + 10000 + fold_index,
                        validation=validation,
                        progress_callback=progress,
                    )
                stage_diagnostics[fit_phase["name"]] = _jsonable(
                    asdict(diagnostics)
                )
            state = clone_state(model.state_dict())
            full_path = root / f"classifier_{name}.pt"
            if rank == 0:
                # Persist the expensive fit before downstream scoring/reporting.
                replay._exclusive_torch_save(full_path, state)
            if trajectory_cfg is not None:
                missing_stages = sorted(
                    set(trajectory_member_states) - {"fully_trained"}
                    - set(captured_trajectory)
                )
                if missing_stages:
                    raise RuntimeError(
                        f"{name} did not reach classifier trajectory stages "
                        f"{missing_stages}; no post-hoc substitution is allowed"
                    )
                for stage, metadata in captured_trajectory.items():
                    trajectory_member_states[stage].append({
                        "seed": int(seed), "fold": fold_index, **metadata,
                    })
                trajectory_member_states["fully_trained"].append({
                    "seed": int(seed),
                    "fold": fold_index,
                    "stage": "fully_trained",
                    "target": None,
                    "step": int(diagnostics.best_step),
                    "validation_balanced_accuracy": float(
                        diagnostics.validation_balanced_accuracy
                    ),
                    "validation_balanced_accuracy_lcb": None,
                    "path": str(full_path),
                })
            scores = {}
            for population_name, population in (
                ("oof", oof), ("early_stop", early),
                ("calibration", calibration), ("final_audit", audit),
            ):
                truth_logits = score_on_device(model, population.packed_event, population.truth, int(cfg["score_batch_size"])).cpu()
                gen_logits = score_on_device(model, population.packed_event, population.candidates, int(cfg["score_batch_size"])).cpu()
                scores[population_name] = score_metrics(
                    truth_logits, gen_logits, torch.zeros(population.n_events)
                )
            record = {
                "seed": int(seed), "fold": fold_index,
                "fit_config": asdict(fit_config),
                "fit_diagnostics": _jsonable(asdict(diagnostics)),
                "conditional_residual_fit_config": (
                    None
                    if conditional_fit_config is None
                    else asdict(conditional_fit_config)
                ),
                "stage_diagnostics": stage_diagnostics,
                "conditional_residual_rank": conditional_rank,
                "scores": scores,
                "trajectory_checkpoints": _jsonable(captured_trajectory),
            }
            member_records.append(record)
            member_states.append({"seed": int(seed), "fold": fold_index, "state": state})
            if rank == 0:
                selected_metric = (
                    conditional_fit_config.checkpoint_selection_metric
                    if conditional_fit_config is not None
                    else fit_config.checkpoint_selection_metric
                )
                selected_value = (
                    diagnostics.validation_loss
                    if selected_metric == "loss"
                    else diagnostics.validation_balanced_accuracy
                )
                print(
                    f"[reward-interface] {name} selected step={diagnostics.best_step} "
                    f"by {selected_metric}={selected_value:.6f}",
                    flush=True,
                )
                replay._exclusive_json(
                    root / f"classifier_{name}.json", _jsonable(record)
                )
            builder.discard_bank(name)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if trajectory_cfg is not None and world > 1:
        # Rank zero wrote each threshold state while the fit was synchronized.
        # Do not let another worker begin shared-panel scoring before those
        # files are visible on the shared filesystem.
        torch.distributed.barrier()

    # Calibration is fit only after every member is frozen.
    calibration_member_logits = []
    audit_member_logits = []
    for item in member_states:
        suffix = f"score_s{item['seed']}_f{item['fold']}"
        calibration_member_logits.append((
            _score_state(builder, pool.packing_spec, item["state"], suffix + "c1",
                         calibration.packed_event, calibration.truth, int(cfg["score_batch_size"]), device),
            _score_state(builder, pool.packing_spec, item["state"], suffix + "c0",
                         calibration.packed_event, calibration.candidates, int(cfg["score_batch_size"]), device),
        ))
        audit_member_logits.append((
            _score_state(builder, pool.packing_spec, item["state"], suffix + "a1",
                         audit.packed_event, audit.truth, int(cfg["score_batch_size"]), device),
            _score_state(builder, pool.packing_spec, item["state"], suffix + "a0",
                         audit.packed_event, audit.candidates, int(cfg["score_batch_size"]), device),
        ))
    by_seed: dict[int, list[int]] = {
        int(seed): [index for index, item in enumerate(member_states) if item["seed"] == int(seed)]
        for seed in cfg["training_seeds"]
    }
    calibration_report: dict[str, Any] = {}
    temperatures: dict[str, float] = {}
    for label, indices in {
        "primary": list(range(len(member_states))),
        **{f"seed_{seed}": indices for seed, indices in by_seed.items()},
    }.items():
        truth_logits = torch.stack([calibration_member_logits[index][0] for index in indices]).mean(0)
        gen_logits = torch.stack([calibration_member_logits[index][1] for index in indices]).mean(0)
        temperature = fit_scalar_temperature(truth_logits, gen_logits)
        temperatures[label] = temperature["temperature"]
        audit_truth = torch.stack([audit_member_logits[index][0] for index in indices]).mean(0)
        audit_gen = torch.stack([audit_member_logits[index][1] for index in indices]).mean(0)
        calibration_report[label] = {
            **temperature,
            "final_audit_uncalibrated": _classification_metrics(audit_truth, audit_gen),
            "final_audit_calibrated": _classification_metrics(
                audit_truth / temperature["temperature"], audit_gen / temperature["temperature"]
            ),
        }

    # Fit a disjoint H4 judge.  It shares calibration identities only for early
    # stopping; its fit identities never enter a reward member.
    judge_fit = populations["judge_fit"]
    judge_config = build_fit_config(
        cfg.get("judge_classifier_fit", cfg["classifier_fit"]),
        n_train=judge_fit.n_events, n_validation=calibration.n_events,
        max_batch_population=judge_fit.n_events,
    )
    judge_name = "independent_h4_judge"
    if rank == 0:
        print("[reward-interface] fitting independent H4 judge", flush=True)
        if wandb_run is not None:
            wandb_run.summary["phase"] = "fit_independent_judge"
            judge_step = "classifier_fit/independent_judge/step"
            wandb_run.define_metric(judge_step, hidden=True)
            wandb_run.define_metric(
                "classifier_fit/independent_judge/*",
                step_metric=judge_step,
                step_sync=False,
            )
    builder.restore_pretrained_body()
    with seeded(int(cfg["judge_seed"]), device):
        judge = builder.make_classifier(
            pool.packing_spec,
            judge_name,
            reset=True,
            **cfg["judge_classifier_overrides"],
        ).to(device)
    judge_validation = _unit_weighted_populations(
        calibration.packed_event, calibration.truth, calibration.candidates
    )
    def judge_progress(row):
        if rank == 0:
            if wandb_run is not None:
                metric_root = "classifier_fit/independent_judge"
                live_metrics = {
                    f"{metric_root}/step": row["step"],
                    f"{metric_root}/training_loss": row["training_loss"],
                    f"{metric_root}/training_balanced_accuracy": row[
                        "training_balanced_accuracy"
                    ],
                }
                for metric in (
                    "training_auc", "gradient_norm", "gradient_clipped"
                ):
                    if metric in row:
                        live_metrics[f"{metric_root}/{metric}"] = row[metric]
                if bool(row.get("validation_evaluated", 0.0)):
                    live_metrics.update({
                        f"{metric_root}/validation_loss": row["validation_loss"],
                        f"{metric_root}/validation_balanced_accuracy": row[
                            "validation_balanced_accuracy"
                        ],
                    })
                    if "validation_auc" in row:
                        live_metrics[f"{metric_root}/validation_auc"] = row[
                            "validation_auc"
                        ]
                wandb_run.log(live_metrics)
            if bool(row.get("validation_evaluated", 0.0)):
                print(
                    f"[reward-interface] {judge_name} step={int(row['step'])} "
                    f"train_ba={row['training_balanced_accuracy']:.4f} "
                    f"val_ba={row['validation_balanced_accuracy']:.4f} "
                    f"val_bce={row['validation_loss']:.6f}",
                    flush=True,
                )

    with seeded(int(cfg["judge_seed"]), device):
        judge_diagnostics = fit_density_ratio(
            judge,
            *_unit_weighted_populations(
                judge_fit.packed_event, judge_fit.truth, judge_fit.candidates
            ),
            judge_config, int(cfg["judge_seed"]), validation=judge_validation,
            progress_callback=judge_progress,
        )
    judge_state = clone_state(judge.state_dict())
    if rank == 0:
        replay._exclusive_torch_save(root / "independent_h4_judge.pt", judge_state)
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)

    # One shared K=8 candidate panel feeds all reward arms and gradient audits.
    if audit.n_events < int(cfg["signed_probe_events"]):
        raise ValueError(
            f"final audit has {audit.n_events} events, fewer than signed_probe_events="
            f"{cfg['signed_probe_events']}"
        )
    order = torch.randperm(
        audit.n_events, generator=torch.Generator().manual_seed(int(cfg["probe_selection_seed"]))
    )[: int(cfg["signed_probe_events"])]
    global_positions_all = torch.arange(len(order))
    local_positions = global_positions_all[rank::world]
    local_pool = audit.select(order[rank::world])
    local_batch = unpack_event_inputs(local_pool.packed_event, local_pool.packing_spec)
    local_batch["x_invisible"] = local_pool.truth.reshape(-1, 2, 2)
    local_batch["x_invisible_mask"] = local_pool.policy_noise_mask
    local_batch = batch_to_device(local_batch, device)
    torch.manual_seed(int(cfg["candidate_seed"]) + rank)
    with torch.no_grad():
        fixed_candidates = generate_neutrino_candidates(
            policy, local_batch, sampler, K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    sample_bk4 = fixed_candidates.permute(1, 0, 2, 3).reshape(len(local_pool.truth), int(cfg["K"]), 4).cpu()
    if trajectory_cfg is not None:
        _run_classifier_trajectory_analysis(
            cfg=cfg,
            rank=rank,
            world=world,
            root=root,
            wandb_run=wandb_run,
            builder=builder,
            pool=pool,
            populations=populations,
            partitions=partitions,
            member_records=member_records,
            trajectory_member_states=trajectory_member_states,
            judge=judge,
            judge_config=judge_config,
            judge_diagnostics=judge_diagnostics,
            policy=policy,
            anchor_model_state=anchor_model_state,
            sampler=sampler,
            local_pool=local_pool,
            local_batch=local_batch,
            local_positions=local_positions,
            order=order,
            fixed_candidates=fixed_candidates,
            sample_bk4=sample_bk4,
            device=device,
            global_config=global_config,
        )
        ray.train.report({
            "policy_updates": 0,
            "reward_installs": 0,
            "completed": 1,
            "classifier_trajectory": 1,
        })
        return
    local_member_logits = []
    for item in member_states:
        local_member_logits.append(
            _score_local_state(
                builder, pool.packing_spec, item["state"],
                f"candidate_s{item['seed']}_f{item['fold']}",
                local_pool.packed_event, sample_bk4, int(cfg["score_batch_size"]), device,
            ).reshape(len(local_pool.truth), int(cfg["K"])).T
        )
    local_member_logits_t = torch.stack(local_member_logits)
    logits_by_ensemble = {
        "primary": local_member_logits_t.mean(0),
        **{
            f"seed_{seed}": local_member_logits_t[indices].mean(0)
            for seed, indices in by_seed.items()
        },
    }
    advantage_sets: dict[str, Tensor] = {}
    advantage_report: dict[str, Any] = {}
    for label, logits in logits_by_ensemble.items():
        arms = reward_advantage_arms(
            logits, temperature=temperatures[label], raw_tempering=float(cfg["raw_tempering"]),
        )
        for arm, advantage in arms.items():
            advantage_sets[f"{label}/{arm}"] = advantage.to(device)
            gathered_advantage = _all_gather_object(advantage.cpu())
            if rank == 0:
                complete = torch.cat(gathered_advantage, dim=1)
                advantage_report[f"{label}/{arm}"] = advantage_metrics(complete)
    gathered_member_logits = _all_gather_object(local_member_logits_t.cpu())
    ordering_report = {}
    if rank == 0:
        complete_member_logits = torch.cat(gathered_member_logits, dim=2)
        ordering_report = member_ordering_metrics(complete_member_logits)
    panel_shards = _all_gather_object({
        "position": local_positions.cpu(),
        "audit_index": order[rank::world].cpu(),
        "packed_event": local_pool.packed_event.cpu(),
        "truth": local_pool.truth.cpu(),
        "policy_noise_mask": local_pool.policy_noise_mask.cpu(),
        "candidates_kb22": fixed_candidates.cpu(),
    })
    if rank == 0:
        positions_complete = torch.cat([item["position"] for item in panel_shards])
        permutation = torch.argsort(positions_complete)
        candidates_complete = torch.cat(
            [item["candidates_kb22"] for item in panel_shards], dim=1
        )[:, permutation]
        packed_complete = torch.cat([item["packed_event"] for item in panel_shards])[permutation]
        truth_complete = torch.cat([item["truth"] for item in panel_shards])[permutation]
        mask_complete = torch.cat([item["policy_noise_mask"] for item in panel_shards])[permutation]
        audit_index_complete = torch.cat([item["audit_index"] for item in panel_shards])[permutation]
        complete_member_logits = torch.cat(gathered_member_logits, dim=2)[:, :, permutation]
        replay._exclusive_torch_save(root / "fixed_k8_panel.pt", {
            "position": torch.arange(len(permutation)),
            "final_audit_index": audit_index_complete,
            "packed_event": packed_complete,
            "truth": truth_complete,
            "policy_noise_mask": mask_complete,
            "candidates_kb22": candidates_complete,
            "member_logits_mkb": complete_member_logits,
            "member_order": [
                {"seed": item["seed"], "fold": item["fold"]}
                for item in member_states
            ],
            "candidate_seed": int(cfg["candidate_seed"]),
        })

    # The reference is an exact clone of the c4a91e07 anchor.  Hence the
    # detached nonlinear DGPO gate starts at 0.5 for every reward mapping.
    reference_bundle = load_evenet_model_for_dgpo(
        config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"]
    )
    reference = reference_bundle.model.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    verify_policy_loaded(reference, anchor_model_state)
    gradient_report, primary_gradients = _gradient_audit(
        cfg=cfg, policy=policy, reference=reference, batch=local_batch,
        candidates=fixed_candidates, advantage_sets=advantage_sets,
        global_positions=local_positions, device=device,
        dtype=next(policy.parameters()).dtype,
    )
    event_concentration = _event_gradient_concentration(
        cfg=cfg, policy=policy, reference=reference, batch=local_batch,
        candidates=fixed_candidates, advantages={
            arm: advantage_sets[f"primary/{arm}"] for arm in REWARD_ARMS
        }, global_positions=local_positions, device=device,
        dtype=next(policy.parameters()).dtype,
    )

    # Common-noise normalized signed probes. +epsilon is gradient descent on
    # the indicated DGPO loss; -epsilon is the symmetry control.
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    anchor = tuple(parameter.detach().cpu().clone() for parameter in parameters)
    raw_gradient = primary_gradients["primary/raw_loo"]
    calibrated_gradient = primary_gradients["primary/calibrated_loo"]
    expected_scale_ratio = 1.0 / (
        float(cfg["raw_tempering"]) * float(temperatures["primary"])
    )
    scale_equivalence = {
        "anchor_gate": 0.5,
        "raw_vs_calibrated_gradient_cosine": vector_cosine(
            raw_gradient, calibrated_gradient
        ),
        "calibrated_over_raw_gradient_norm": float(
            calibrated_gradient.norm() / raw_gradient.norm().clamp_min(1.0e-30)
        ),
        "expected_calibrated_over_raw_scale": expected_scale_ratio,
        "meaning": (
            "At the exact anchor, scalar temperature cannot change the normalized "
            "LOO direction. The finite-step relinearization below tests its effect "
            "through the nonlinear DGPO gate."
        ),
    }
    _assign_direction(
        parameters, anchor, raw_gradient, sign=1,
        epsilon_rms=float(cfg["signed_step_rms"]),
    )
    scale_relinearization, _ = _gradient_audit(
        cfg=cfg, policy=policy, reference=reference, batch=local_batch,
        candidates=fixed_candidates,
        advantage_sets={
            "primary/raw_loo": advantage_sets["primary/raw_loo"],
            "primary/calibrated_loo": advantage_sets["primary/calibrated_loo"],
        },
        global_positions=local_positions, device=device,
        dtype=next(policy.parameters()).dtype,
    )
    with torch.no_grad():
        for parameter, base in zip(parameters, anchor):
            parameter.copy_(base.to(parameter.device, parameter.dtype))
    truth_judge_local = _score_local_on_device(
        judge, local_pool.packed_event, local_pool.truth, int(cfg["score_batch_size"])
    ).cpu()

    def evaluate_signed(label: str, direction: Tensor | None, sign: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        scale = 0.0
        if direction is not None:
            scale = _assign_direction(
                parameters, anchor, direction, sign=sign,
                epsilon_rms=float(cfg["signed_step_rms"]),
            )
        else:
            with torch.no_grad():
                for parameter, base in zip(parameters, anchor):
                    parameter.copy_(base.to(parameter.device, parameter.dtype))
        torch.manual_seed(int(cfg["signed_rollout_seed"]) + rank)
        with torch.no_grad():
            candidates = generate_neutrino_candidates(
                policy, local_batch, sampler, K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
                parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
            )
        gen_bk4 = candidates.permute(1, 0, 2, 3).reshape(len(local_pool.truth), int(cfg["K"]), 4).cpu()
        judge_logits = _score_local_on_device(
            judge, local_pool.packed_event, gen_bk4, int(cfg["score_batch_size"])
        ).cpu()
        gathered_truth = _all_gather_object(truth_judge_local)
        gathered_gen = _all_gather_object(judge_logits)
        event_valid = torch.ones(len(local_pool.truth), device=device, dtype=torch.bool)
        arrays = collect_ztautau_validation_arrays(candidates, baseline_candidates, local_batch, event_valid)
        arrays = _gather_numpy_dict(arrays)
        if rank != 0:
            return {}, {}
        judge_metrics = _classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_gen).reshape(-1)
        )
        physics = build_ztautau_validation_metrics(
            arrays, val_k=int(cfg["K"]), tarp_config={"enabled": False},
            metrics_config={"enabled": True, "bins": int(cfg["physics_bins"]), "candidate_index": 0},
            include_images=False,
        )
        response, event_scores = response_matrix_metrics(
            arrays["_tarp_truth"], arrays["_tarp_candidates"], bins=int(cfg["response_bins"])
        )
        mean_offset = float(np.mean([
            values["mean_abs_bin_offset"] for values in response.values()
        ]))
        result = {
            "label": label, "direction_sign": sign,
            "signed_step_rms": 0.0 if sign == 0 else float(cfg["signed_step_rms"]),
            "flat_gradient_scale": scale,
            "judge": judge_metrics,
            "judge_auc_gap": judge_metrics["auc_gap"],
            "physics": physics,
            "response": response,
            "response_mean_abs_bin_offset": mean_offset,
        }
        return result, event_scores

    baseline_candidates = fixed_candidates
    # Regenerate the zero arm using the signed-rollout seed; it becomes the
    # common reference for every +/- candidate panel.
    with torch.no_grad():
        for parameter, base in zip(parameters, anchor):
            parameter.copy_(base.to(parameter.device, parameter.dtype))
    torch.manual_seed(int(cfg["signed_rollout_seed"]) + rank)
    with torch.no_grad():
        baseline_candidates = generate_neutrino_candidates(
            policy, local_batch, sampler, K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps), device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    zero_result, zero_events = evaluate_signed("zero", None, 0)
    signed_report: dict[str, Any] = {}
    for arm_index, arm in enumerate(REWARD_ARMS):
        direction = primary_gradients[f"primary/{arm}"]
        minus_result, minus_events = evaluate_signed(f"{arm}/minus", direction, -1)
        plus_result, plus_events = evaluate_signed(f"{arm}/plus", direction, +1)
        if rank == 0:
            bootstrap = {}
            for component_index, component in enumerate(zero_events):
                bootstrap[f"plus_vs_zero/{component}"] = _paired_bootstrap_delta(
                    plus_events[component], zero_events[component],
                    replicates=int(cfg["response_bootstrap_replicates"]),
                    seed=int(cfg["response_bootstrap_seed"]) + 10 * arm_index + component_index,
                )
                bootstrap[f"minus_vs_zero/{component}"] = _paired_bootstrap_delta(
                    minus_events[component], zero_events[component],
                    replicates=int(cfg["response_bootstrap_replicates"]),
                    seed=int(cfg["response_bootstrap_seed"]) + 1000 + 10 * arm_index + component_index,
                )
            signed_report[arm] = {
                "minus": minus_result, "zero": zero_result, "plus": plus_result,
                "paired_response_bootstrap": bootstrap,
            }
    with torch.no_grad():
        for parameter, base in zip(parameters, anchor):
            parameter.copy_(base.to(parameter.device, parameter.dtype))

    if rank == 0:
        decision_input = {
            arm: {
                "minus_judge_auc_gap": row["minus"]["judge_auc_gap"],
                "zero_judge_auc_gap": row["zero"]["judge_auc_gap"],
                "plus_judge_auc_gap": row["plus"]["judge_auc_gap"],
                "minus_response_mean_abs_bin_offset": row["minus"]["response_mean_abs_bin_offset"],
                "zero_response_mean_abs_bin_offset": row["zero"]["response_mean_abs_bin_offset"],
                "plus_response_mean_abs_bin_offset": row["plus"]["response_mean_abs_bin_offset"],
            }
            for arm, row in signed_report.items()
        }
        verify_policy_loaded(policy, anchor_model_state)
        report = {
            "schema": "c4a91e07-h4-reward-interface-v2",
            "source_wandb_run": cfg.get("source_wandb_run"),
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0,
            "reward_installs": 0,
            "optimizer_state_resumed": False,
            "fixed_policy": True,
            "policy_unchanged": True,
            "fixed_candidate_panel": True,
            "signed_probe_events": int(cfg["signed_probe_events"]),
            "gradient_events": int(cfg["gradient_events"]),
            "partitions": {name: int(len(indices)) for name, indices in partitions.items()},
            "classifier_members": member_records,
            "classifier_calibration": calibration_report,
            "independent_judge": {
                "seed": int(cfg["judge_seed"]),
                "fit_config": asdict(judge_config),
                "fit_diagnostics": _jsonable(asdict(judge_diagnostics)),
            },
            "member_ordering": ordering_report,
            "advantages": advantage_report,
            "gradients": gradient_report,
            "raw_calibrated_scale_equivalence": scale_equivalence,
            "finite_step_scale_relinearization": scale_relinearization,
            "event_gradient_concentration": event_concentration,
            "signed_probes": signed_report,
            "decision": reward_interface_diagnosis(decision_input),
            "decision_scope": (
                "Local fixed-policy direction only. A passing arm is eligible for a later "
                "50-step frozen-reward pilot; this run does not claim long-run DGPO closure."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(root / "report.json", _jsonable(report))
        if wandb_run is not None:
            numeric = {}
            for arm, row in signed_report.items():
                for sign in ("minus", "zero", "plus"):
                    numeric[f"signed/{arm}/{sign}/judge_auc_gap"] = row[sign]["judge_auc_gap"]
                    numeric[f"signed/{arm}/{sign}/response_mean_abs_bin_offset"] = row[sign]["response_mean_abs_bin_offset"]
                numeric[f"gradient/{arm}/norm"] = gradient_report[f"primary/{arm}"]["gradient_norm"]
                numeric[f"gradient/{arm}/split_half_cosine"] = gradient_report[f"primary/{arm}"]["split_half_cosine"]
            wandb_run.log(numeric)
            artifact = wandb.Artifact("c4a91e07-h4-reward-interface-report", type="diagnostic")
            artifact.add_file(str(root / "report.json"))
            wandb_run.log_artifact(artifact)
            wandb_run.summary["phase"] = "complete"
            wandb_run.summary["decision"] = report["decision"]["finding"]
            wandb_run.finish()
        print(f"Reward-interface report: {root / 'report.json'}", flush=True)
    ray.train.report({"policy_updates": 0, "reward_installs": 0, "completed": 1})


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
        f"Verified c4a91e07 last.ckpt step={cfg['expected_policy_step']} "
        f"output={cfg['output_dir']}", flush=True,
    )
    if args.check_only:
        return 0
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    cfg["runtime_path"] = str(root / "runtime.yaml")
    with Path(cfg["runtime_path"]).open("x") as handle:
        yaml.safe_dump(cfg["runtime"], handle, sort_keys=False)
    replay._exclusive_json(root / "manifest.json", _jsonable(cfg))
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from evenet.control.global_config import global_config
    from evenet.shared import make_process_fn, register_dataset

    global_config.load_yaml(cfg["runtime_path"])
    ray.init(
        address=os.environ.get("RAY_ADDRESS") or "auto",
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join([
            str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")
        ])}},
    )
    dataset, _ = register_dataset(
        cfg["files"], make_process_fn(Path(global_config.platform.data_parquet_dir)),
        global_config.platform, dataset_limit=1.0, file_shuffling=False,
    )
    TorchTrainer(
        train_loop_per_worker=_worker,
        train_loop_config=cfg,
        datasets={"pool": dataset},
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]), use_gpu=True,
            resources_per_worker={"CPU": int(cfg.get("cpus_per_worker", 2)), "GPU": 1},
        ),
        run_config=RunConfig(
            name=str(
                cfg.get("ray_run_name", "c4a91e07-h4-reward-interface-v5")
            ),
            storage_path=str(root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
