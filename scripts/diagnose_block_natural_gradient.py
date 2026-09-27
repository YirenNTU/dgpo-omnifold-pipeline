#!/usr/bin/env python3
"""Compare vanilla and block-natural H4 DGPO directions at matched VP distance.

The experiment consumes the completed h4grad01 mean gradient.  It does not fit
a classifier, recompute the reward gradient, or perform an optimizer update.
An empirical cosine-VP velocity metric is estimated on identities unused by
h4grad01, then an independent panel measures common-noise signed judge probes.
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

import diagnose_production_gradient_reproducibility as gradient_repro
import diagnose_raw_monitor_replay as replay
import diagnose_reward_interface as interface
from ablate_raw_monitor_initialization import clone_state

from RL.DGPO_neutrino.reward_interface import vector_cosine


REWARD_ARTIFACTS = (
    "runtime.yaml",
    "manifest.json",
    "report.json",
    "fixed_k1_pool.pt",
    "independent_h4_judge.pt",
)
GRADIENT_ARTIFACTS = (
    "manifest.json",
    "report.json",
    "gradient_directions_fp16.pt",
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
        "reward_source_dir",
        "gradient_source_dir",
        "output_dir",
        "expected_reward_schema",
        "expected_gradient_schema",
        "expected_direction_schema",
        "expected_source_wandb_run",
        "expected_gradient_wandb_id",
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "K",
        "curvature_events",
        "curvature_selection_seed",
        "curvature_candidate_seed",
        "curvature_timesteps",
        "curvature_path_seed",
        "basis_probe_rms",
        "maximum_metric_condition",
        "ridge_floor_fraction",
        "zero_block_norm",
        "vanilla_parameter_rms",
        "distance_match_tolerance",
        "distance_match_iterations",
        "minimum_parameter_rms",
        "maximum_parameter_rms",
        "minimum_decisive_efficiency_gain",
        "judge_events",
        "judge_selection_seed",
        "judge_rollout_seed",
        "score_batch_size",
        "save_direction_vectors",
        "upload_direction_vectors",
        "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing block-natural-gradient settings: {missing}")
    integer_keys = (
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "K",
        "curvature_events",
        "curvature_selection_seed",
        "curvature_candidate_seed",
        "curvature_timesteps",
        "curvature_path_seed",
        "distance_match_iterations",
        "judge_events",
        "judge_selection_seed",
        "judge_rollout_seed",
        "score_batch_size",
    )
    for key in integer_keys:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    exact = {"workers": 16, "K": 8, "curvature_timesteps": 8}
    for key, expected in exact.items():
        if cfg[key] != expected:
            raise ValueError(f"this experiment requires {key}={expected}")
    for key in ("curvature_events", "judge_events"):
        if cfg[key] % cfg["workers"]:
            raise ValueError(f"{key} must divide evenly over workers")
    float_keys = (
        "basis_probe_rms",
        "maximum_metric_condition",
        "ridge_floor_fraction",
        "zero_block_norm",
        "vanilla_parameter_rms",
        "distance_match_tolerance",
        "minimum_parameter_rms",
        "maximum_parameter_rms",
        "minimum_decisive_efficiency_gain",
    )
    for key in float_keys:
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["maximum_metric_condition"] <= 1.0:
        raise ValueError("maximum_metric_condition must exceed one")
    if cfg["ridge_floor_fraction"] >= 1.0:
        raise ValueError("ridge_floor_fraction must be smaller than one")
    if cfg["distance_match_tolerance"] >= 1.0:
        raise ValueError("distance_match_tolerance must be smaller than one")
    if cfg["minimum_parameter_rms"] >= cfg["maximum_parameter_rms"]:
        raise ValueError("parameter RMS bounds are reversed")
    if not (
        cfg["minimum_parameter_rms"]
        <= cfg["vanilla_parameter_rms"]
        <= cfg["maximum_parameter_rms"]
    ):
        raise ValueError("vanilla_parameter_rms lies outside the matching bounds")
    for key in ("save_direction_vectors", "upload_direction_vectors"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg["upload_direction_vectors"] and not cfg["save_direction_vectors"]:
        raise ValueError("upload_direction_vectors requires save_direction_vectors")
    wandb = dict(cfg["wandb"] or {})
    wandb.setdefault("enabled", False)
    if type(wandb["enabled"]) is not bool:
        raise ValueError("wandb.enabled must be boolean")
    if wandb.get("required") and not wandb["enabled"]:
        raise ValueError("wandb.required=true requires wandb.enabled=true")
    cfg["wandb"] = wandb
    return cfg


def _artifact_paths(root: Path, names: Sequence[str]) -> dict[str, Path]:
    return {name: (root / name).resolve(strict=True) for name in names}


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate reward, gradient, policy, and parameter-layout provenance."""

    cfg = _validated_settings(settings)
    reward_root = Path(cfg["reward_source_dir"]).expanduser().resolve(strict=True)
    gradient_root = Path(cfg["gradient_source_dir"]).expanduser().resolve(strict=True)
    reward_paths = _artifact_paths(reward_root, REWARD_ARTIFACTS)
    gradient_paths = _artifact_paths(gradient_root, GRADIENT_ARTIFACTS)
    reward_report = json.loads(reward_paths["report.json"].read_text())
    gradient_report = json.loads(gradient_paths["report.json"].read_text())
    gradient_manifest = json.loads(gradient_paths["manifest.json"].read_text())
    if reward_report.get("schema") != cfg["expected_reward_schema"]:
        raise ValueError("reward-interface schema mismatch")
    if reward_report.get("source_wandb_run") != cfg["expected_source_wandb_run"]:
        raise ValueError("reward-interface W&B provenance mismatch")
    if gradient_report.get("schema") != cfg["expected_gradient_schema"]:
        raise ValueError("production-gradient schema mismatch")
    if gradient_manifest.get("wandb", {}).get("id") != cfg["expected_gradient_wandb_id"]:
        raise ValueError("production-gradient W&B identity mismatch")
    if gradient_report.get("production_variance_status") != "production_gradient_reproducible":
        raise ValueError("h4grad01 did not certify a reproducible production gradient")
    reproducibility = gradient_report.get("reproducibility", {})
    if (
        float(reproducibility.get("mean_pairwise_cosine", -1.0)) < 0.90
        or float(reproducibility.get("min_pairwise_cosine", -1.0)) < 0.80
    ):
        raise ValueError("production gradient no longer passes its declared threshold")
    for report in (reward_report, gradient_report):
        if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
            raise ValueError("source policy step mismatch")
    if int(gradient_report.get("policy_updates", -1)) != 0:
        raise ValueError("gradient source unexpectedly performed policy updates")
    if int(gradient_report.get("classifier_fits", -1)) != 0:
        raise ValueError("gradient source unexpectedly fitted classifiers")
    policy_checkpoint = Path(gradient_report["policy_checkpoint"]).resolve(strict=True)
    if policy_checkpoint != Path(reward_report["policy_checkpoint"]).resolve(strict=True):
        raise ValueError("reward and gradient sources use different policy checkpoints")
    checkpoint = replay.read_checkpoint(policy_checkpoint)
    if int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("live policy checkpoint step differs from source reports")
    del checkpoint

    direction_payload = _load(gradient_paths["gradient_directions_fp16.pt"])
    if direction_payload.get("schema") != cfg["expected_direction_schema"]:
        raise ValueError("gradient-direction schema mismatch")
    if int(direction_payload.get("policy_global_step", -1)) != int(
        cfg["expected_policy_step"]
    ):
        raise ValueError("gradient-direction policy step mismatch")
    directions = direction_payload.get("directions", {})
    if set(directions) != {"g1", "g2", "g3", "g4", "gbar"}:
        raise ValueError("gradient source does not contain all four gradients and gbar")
    gbar = directions["gbar"].float()
    reconstructed = torch.stack([
        directions[f"g{index}"].float() for index in range(1, 5)
    ]).mean(0)
    reconstruction_cosine = vector_cosine(gbar, reconstructed)
    if reconstruction_cosine < 0.999:
        raise ValueError("stored gbar is inconsistent with its four gradient realizations")
    if not bool(torch.isfinite(gbar).all()) or float(gbar.double().norm()) <= 0.0:
        raise ValueError("stored gbar is zero or non-finite")
    layout = direction_payload.get("parameter_layout")
    if not isinstance(layout, list) or not layout:
        raise ValueError("gradient source lacks a parameter layout")
    if int(layout[-1]["stop"]) != int(gbar.numel()):
        raise ValueError("gradient layout does not cover gbar")
    del direction_payload, directions, reconstructed, gbar

    runtime = __import__("yaml").safe_load(reward_paths["runtime.yaml"].read_text())
    if runtime["dgpo"].get("checkpoint_load_mode") != "weights_only":
        raise ValueError("reward runtime must use weights-only policy loading")
    training = runtime["options"]["Training"]
    if any(training.get("EMA", {}).get(key, False) for key in (
        "replace_model_after_load",
        "use_for_generation",
        "use_ema_during_training_eval",
    )):
        raise ValueError("reward runtime unexpectedly selects EMA weights")
    if int(runtime["dgpo"]["K"]) != int(cfg["K"]):
        raise ValueError("K differs from the source production objective")

    pool = _load(reward_paths["fixed_k1_pool.pt"])
    final_indices = pool.get("partitions", {}).get("final_audit")
    if not isinstance(final_indices, Tensor) or final_indices.ndim != 1:
        raise ValueError("reward source lacks final_audit identities")
    curvature, judge, unused = select_unused_event_indices(
        final_indices.long(), gradient_manifest, cfg
    )
    del pool, curvature, judge

    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    for protected in (reward_root, gradient_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output directory overlaps an immutable source directory")
    all_paths = [*reward_paths.values(), *gradient_paths.values()]
    cfg.update(
        reward_source_dir=str(reward_root),
        gradient_source_dir=str(gradient_root),
        output_dir=str(output),
        runtime_path=str(reward_paths["runtime.yaml"]),
        policy_checkpoint=str(policy_checkpoint),
        parameter_layout=layout,
        gradient_reconstruction_cosine=reconstruction_cosine,
        h4grad01_manifest=gradient_manifest,
        unused_final_audit_events=int(unused),
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
            raise RuntimeError(f"source artifact changed during experiment: {path}")


def select_unused_event_indices(
    final_indices: Tensor,
    previous_manifest: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[Tensor, Tensor, int]:
    """Exclude every h4grad01 event, then create disjoint metric/judge panels."""

    previous_gradient, previous_probe = gradient_repro.select_disjoint_event_indices(
        final_indices, previous_manifest
    )
    previously_used = torch.cat((previous_gradient, previous_probe))
    remaining = final_indices[~torch.isin(final_indices, previously_used)]
    needed = int(cfg["curvature_events"]) + int(cfg["judge_events"])
    if int(remaining.numel()) < needed:
        raise ValueError(
            f"only {remaining.numel()} unused final_audit events remain; need {needed}"
        )
    curvature_order = torch.randperm(
        len(remaining),
        generator=torch.Generator().manual_seed(int(cfg["curvature_selection_seed"])),
    )
    curvature = remaining[curvature_order[:int(cfg["curvature_events"])]]
    after_curvature = remaining[curvature_order[int(cfg["curvature_events"]):]]
    judge_order = torch.randperm(
        len(after_curvature),
        generator=torch.Generator().manual_seed(int(cfg["judge_selection_seed"])),
    )
    judge = after_curvature[judge_order[:int(cfg["judge_events"])]]
    return curvature, judge, int(remaining.numel())


def validate_parameter_layout(
    layout: Sequence[Mapping[str, Any]],
    named_parameters: Sequence[tuple[str, Tensor]],
    vector_size: int,
) -> dict[str, list[tuple[int, int]]]:
    trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    if len(layout) != len(trainable):
        raise ValueError("saved parameter layout length differs from live policy")
    blocks: dict[str, list[tuple[int, int]]] = {}
    offset = 0
    for row, (name, parameter) in zip(layout, trainable, strict=True):
        start, stop = int(row["start"]), int(row["stop"])
        if (
            row["name"] != name
            or list(row["shape"]) != list(parameter.shape)
            or start != offset
            or stop - start != parameter.numel()
        ):
            raise ValueError(f"saved parameter layout differs at {name}")
        blocks.setdefault(str(row["block"]), []).append((start, stop))
        offset = stop
    if offset != int(vector_size):
        raise ValueError("saved parameter layout does not cover the live policy")
    return blocks


def vector_rms(vector: Tensor) -> float:
    return float(vector.detach().double().square().mean().sqrt())


def build_block_basis(
    gradient: Tensor,
    blocks: Mapping[str, Sequence[tuple[int, int]]],
    *,
    norm_floor: float,
) -> tuple[list[str], list[Tensor], Tensor]:
    """Return disjoint unit-full-RMS block vectors and B^T gradient."""

    names: list[str] = []
    basis: list[Tensor] = []
    q: list[float] = []
    for name, ranges in blocks.items():
        vector = torch.zeros_like(gradient, dtype=torch.float32)
        for start, stop in ranges:
            vector[start:stop] = gradient[start:stop].float()
        norm = float(vector.double().norm())
        if not math.isfinite(norm) or norm <= float(norm_floor):
            continue
        rms = vector_rms(vector)
        unit = vector / rms
        names.append(name)
        basis.append(unit)
        q.append(float(torch.dot(gradient.double(), unit.double())))
    if not basis:
        raise ValueError("no nonzero gradient blocks remain")
    return names, basis, torch.tensor(q, dtype=torch.float64)


def ridge_for_condition(
    metric: Tensor,
    *,
    maximum_condition: float,
    floor_fraction: float,
) -> tuple[float, dict[str, float]]:
    """Smallest scalar ridge satisfying a fixed condition-number ceiling."""

    matrix = 0.5 * (metric.double() + metric.double().T)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    largest = float(eigenvalues[-1])
    smallest = float(eigenvalues[0])
    if not math.isfinite(largest) or largest <= 0.0:
        raise ValueError("empirical VP metric has no positive eigenvalue")
    floor = float(floor_fraction) * largest
    condition_ridge = max(
        0.0,
        (largest - float(maximum_condition) * smallest)
        / (float(maximum_condition) - 1.0),
    )
    ridge = max(floor, condition_ridge)
    regularized_condition = (largest + ridge) / max(smallest + ridge, 1.0e-30)
    return ridge, {
        "minimum_eigenvalue": smallest,
        "maximum_eigenvalue": largest,
        "unregularized_condition": (
            largest / smallest if smallest > 0.0 else float("inf")
        ),
        "ridge": ridge,
        "regularized_condition": regularized_condition,
    }


def solve_block_natural_direction(
    metric: Tensor,
    q: Tensor,
    basis: Sequence[Tensor],
    *,
    maximum_condition: float,
    floor_fraction: float,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    if metric.shape != (len(basis), len(basis)) or q.shape != (len(basis),):
        raise ValueError("metric, q, and basis dimensions differ")
    ridge, diagnostics = ridge_for_condition(
        metric,
        maximum_condition=maximum_condition,
        floor_fraction=floor_fraction,
    )
    regularized = 0.5 * (metric.double() + metric.double().T)
    regularized = regularized + ridge * torch.eye(len(basis), dtype=torch.float64)
    coefficients = torch.linalg.solve(regularized, q.double())
    direction = torch.zeros_like(basis[0], dtype=torch.float64)
    for coefficient, vector in zip(coefficients, basis, strict=True):
        direction.add_(vector.double(), alpha=float(coefficient))
    descent_derivative = float(torch.dot(q.double(), coefficients.double()))
    if descent_derivative <= 0.0 or not bool(torch.isfinite(direction).all()):
        raise ValueError("regularized block-natural direction is not a descent direction")
    diagnostics["linear_descent_derivative"] = descent_derivative
    diagnostics["direction_rms_before_matching"] = vector_rms(direction)
    return direction.float(), coefficients, diagnostics


def classify_result(
    *,
    zero_gap: float,
    vanilla_plus_gap: float,
    natural_plus_gap: float,
    natural_minus_gap: float,
    vp_distance_ratio: float,
    distance_tolerance: float,
    decisive_gain: float,
) -> dict[str, Any]:
    vanilla_improvement = float(zero_gap) - float(vanilla_plus_gap)
    natural_improvement = float(zero_gap) - float(natural_plus_gap)
    efficiency_gain = (
        natural_improvement / vanilla_improvement
        if vanilla_improvement > 0.0
        else float("nan")
    )
    match_passed = abs(float(vp_distance_ratio) - 1.0) <= float(distance_tolerance)
    plus_beats_minus = float(natural_plus_gap) < float(natural_minus_gap)
    if not match_passed or not plus_beats_minus:
        decision = "invalid_distance_match_or_natural_sign"
    elif (
        natural_improvement > 0.0
        and natural_plus_gap < vanilla_plus_gap
        and efficiency_gain >= decisive_gain
    ):
        decision = "block_policy_conditioning_is_material_bottleneck"
    elif natural_improvement > 0.0 and natural_plus_gap < vanilla_plus_gap:
        decision = "block_geometry_helps_but_is_not_decisive"
    else:
        decision = "block_geometry_is_not_sufficient"
    return {
        "decision": decision,
        "vanilla_improvement": vanilla_improvement,
        "natural_improvement": natural_improvement,
        "efficiency_gain": efficiency_gain,
        "distance_match_passed": match_passed,
        "natural_plus_beats_minus": plus_beats_minus,
    }


def _set_parameter_offset(
    parameters: Sequence[Tensor],
    anchor: Sequence[Tensor],
    direction: Tensor,
    *,
    signed_rms: float,
) -> float:
    rms = vector_rms(direction)
    if not math.isfinite(rms) or rms <= 0.0:
        raise ValueError("cannot apply a zero or non-finite direction")
    scale = float(signed_rms) / rms
    offset = 0
    with torch.no_grad():
        for parameter, base in zip(parameters, anchor, strict=True):
            count = int(parameter.numel())
            update = direction[offset:offset + count].reshape_as(parameter)
            parameter.copy_(
                base.to(parameter.device, parameter.dtype)
                + scale * update.to(parameter.device, parameter.dtype)
            )
            offset += count
    if offset != int(direction.numel()):
        raise RuntimeError("flat direction does not match policy parameters")
    return scale


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import batch_to_device, policy_evaluation_step
    from RL.DGPO_neutrino.dgpo_utils import sample_cosine_vp_path_kl_timesteps
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
        raise ValueError(f"expected {cfg['workers']} Ray workers, found {world}")
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg["runtime_path"])
    torch.set_float32_matmul_precision(
        str(global_config.dgpo.get("float32_matmul_precision", "medium"))
    )
    output_root = Path(cfg["output_dir"])
    reward_root = Path(cfg["reward_source_dir"])
    gradient_root = Path(cfg["gradient_source_dir"])

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
                    "h4grad01_manifest",
                    "parameter_layout",
                }
            }),
        )
        wandb_run.summary.update({
            "phase": "load_sources",
            "classifier_fits": 0,
            "policy_updates": 0,
        })
        wandb_run.log({
            "geometry/progress/stage": 0.0,
            "geometry/objective_contract/K": float(cfg["K"]),
            "geometry/objective_contract/workers": float(cfg["workers"]),
            "geometry/objective_contract/policy_updates": 0.0,
            "geometry/objective_contract/classifier_fits": 0.0,
        })

    pool = _load(reward_root / "fixed_k1_pool.pt")
    packing_spec = EventPackingSpec.from_dict(pool["packing_spec"])
    final_indices = pool["partitions"]["final_audit"].long()
    curvature_indices, judge_indices, unused_count = select_unused_event_indices(
        final_indices, cfg["h4grad01_manifest"], cfg
    )
    local_curvature = curvature_indices[rank::world]
    local_judge = judge_indices[rank::world]
    if (
        len(local_curvature) != int(cfg["curvature_events"]) // world
        or len(local_judge) != int(cfg["judge_events"]) // world
    ):
        raise RuntimeError("identity sharding produced unequal local panels")

    checkpoint = replay.read_checkpoint(cfg["policy_checkpoint"])
    bundle = load_evenet_model_for_dgpo(
        config=global_config,
        device=device,
        checkpoint_path=cfg["policy_checkpoint"],
    )
    policy = bundle.model.eval()
    interface.verify_policy_loaded(policy, checkpoint["state_dict"])
    assert_dgpo_neutrino_policy_deterministic(policy)
    anchor_state = clone_state(policy.state_dict())
    reference_bundle = load_evenet_model_for_dgpo(
        config=global_config,
        device=device,
        checkpoint_path=cfg["policy_checkpoint"],
    )
    reference = reference_bundle.model.eval()
    interface.verify_policy_loaded(reference, checkpoint["state_dict"])
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    del checkpoint

    direction_payload = _load(gradient_root / "gradient_directions_fp16.pt")
    gbar = direction_payload["directions"]["gbar"].float()
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    anchor_parameters = tuple(parameter.detach().cpu().clone() for parameter in parameters)
    blocks = validate_parameter_layout(
        cfg["parameter_layout"],
        list(policy.named_parameters()),
        int(gbar.numel()),
    )
    block_names, basis, q = build_block_basis(
        gbar,
        blocks,
        norm_floor=float(cfg["zero_block_norm"]),
    )
    del direction_payload

    def restore_anchor() -> None:
        policy.load_state_dict(anchor_state, strict=True)
        policy.eval()
        policy.zero_grad(set_to_none=True)

    def make_batch(indices: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor]:
        packed = pool["packed_event"].index_select(0, indices)
        truth = pool["truth"].index_select(0, indices)
        noise_mask = pool["policy_noise_mask"].index_select(0, indices)
        batch = unpack_event_inputs(packed, packing_spec)
        batch["x_invisible"] = truth.reshape(-1, 2, 2)
        batch["x_invisible_mask"] = noise_mask
        return batch_to_device(batch, device), packed, truth, noise_mask

    curvature_batch, curvature_packed, _, _ = make_batch(local_curvature)
    restore_anchor()
    torch.manual_seed(int(cfg["curvature_candidate_seed"]) + rank)
    sampler = DDIMSampler(device=device)
    with torch.no_grad():
        curvature_candidates = generate_neutrino_candidates(
            policy,
            curvature_batch,
            sampler,
            K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
            device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    candidate_nonfinite = float(
        (~torch.isfinite(curvature_candidates)).float().mean().cpu()
    )
    candidate_nonfinite_all = interface._all_gather_object(candidate_nonfinite)
    curvature_candidates = torch.nan_to_num(curvature_candidates)

    local_B = len(local_curvature)
    timesteps = int(cfg["curvature_timesteps"])
    path_generator = torch.Generator(device=device)
    path_generator.manual_seed(int(cfg["curvature_path_seed"]) + rank)
    path_t, path_normalizer = sample_cosine_vp_path_kl_timesteps(
        timesteps,
        local_B,
        total_strata=timesteps,
        stratum_offset=0,
        device=device,
        dtype=torch.float32,
        generator=path_generator,
    )
    n_nu, n_features = int(curvature_candidates.shape[2]), int(
        curvature_candidates.shape[3]
    )
    base_eps = torch.randn(
        timesteps,
        local_B,
        n_nu,
        n_features,
        device=device,
        dtype=next(policy.parameters()).dtype,
        generator=path_generator,
    )
    path_eps = (
        base_eps.unsqueeze(1)
        .expand(timesteps, int(cfg["K"]), local_B, n_nu, n_features)
        .reshape(timesteps * int(cfg["K"]) * local_B, n_nu, n_features)
    )

    def vp_output() -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            values = policy_evaluation_step(
                policy,
                reference,
                curvature_batch,
                curvature_candidates,
                K=int(cfg["K"]),
                shared_noise=True,
                device=device,
                dtype=next(policy.parameters()).dtype,
                t=path_t,
                eps_rep=path_eps,
                t_min=0.0,
                t_max=1.0,
                num_timesteps=timesteps,
            )
        return values[3].detach(), values[5].detach()

    restore_anchor()
    baseline_output, path_mask = vp_output()
    responses: list[Tensor] = []
    basis_diagnostics: dict[str, Any] = {}
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "estimate_block_vp_metric"
    for index, (name, direction) in enumerate(zip(block_names, basis, strict=True)):
        restore_anchor()
        _set_parameter_offset(
            parameters,
            anchor_parameters,
            direction,
            signed_rms=float(cfg["basis_probe_rms"]),
        )
        positive, positive_mask = vp_output()
        restore_anchor()
        _set_parameter_offset(
            parameters,
            anchor_parameters,
            direction,
            signed_rms=-float(cfg["basis_probe_rms"]),
        )
        negative, negative_mask = vp_output()
        if not torch.equal(path_mask, positive_mask) or not torch.equal(
            path_mask, negative_mask
        ):
            raise RuntimeError("VP path mask changed across central differences")
        response = (positive - negative) / (2.0 * float(cfg["basis_probe_rms"]))
        midpoint = 0.5 * (positive + negative) - baseline_output
        response_norm = float(response.detach().double().norm().cpu())
        nonlinear_ratio = float(
            midpoint.detach().double().norm().cpu()
            / max((0.5 * (positive - negative)).detach().double().norm().cpu(), 1e-30)
        )
        responses.append(response)
        if rank == 0:
            basis_diagnostics[name] = {
                "response_norm_rank0": response_norm,
                "central_nonlinearity_ratio_rank0": nonlinear_ratio,
            }
            print(
                f"[geometry] metric basis {index + 1}/{len(basis)} {name}: "
                f"response_norm={response_norm:.6g}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "geometry/progress/stage": 1.0,
                    "geometry/metric/basis_index": index,
                    f"geometry/metric/block/{name}/response_norm_rank0": response_norm,
                    f"geometry/metric/block/{name}/central_nonlinearity_ratio_rank0": nonlinear_ratio,
                })
        del positive, negative, positive_mask, negative_mask, midpoint

    expanded_mask = path_mask.expand_as(responses[0]).to(responses[0].dtype)
    metric_sum = torch.zeros(
        len(responses), len(responses), device=device, dtype=torch.float64
    )
    for left, left_response in enumerate(responses):
        for right in range(left, len(responses)):
            value = (
                left_response.double()
                * responses[right].double()
                * expanded_mask.double()
            ).sum()
            metric_sum[left, right] = value
            metric_sum[right, left] = value
    row_count = torch.tensor(
        float(baseline_output.shape[0]), device=device, dtype=torch.float64
    )
    if world > 1:
        torch.distributed.all_reduce(metric_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(row_count, op=torch.distributed.ReduceOp.SUM)
    metric = 0.5 * float(path_normalizer) * metric_sum.cpu() / float(row_count.cpu())
    natural_direction, coefficients, natural_diagnostics = solve_block_natural_direction(
        metric,
        q,
        basis,
        maximum_condition=float(cfg["maximum_metric_condition"]),
        floor_fraction=float(cfg["ridge_floor_fraction"]),
    )
    direction_cosine = vector_cosine(gbar, natural_direction)

    def vp_distance(output: Tensor) -> float:
        difference = output - baseline_output
        local_sum = (
            difference.double().square() * expanded_mask.double()
        ).sum()
        totals = torch.stack((local_sum, row_count.new_tensor(float(output.shape[0]))))
        if world > 1:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        return float(
            0.5 * float(path_normalizer) * totals[0] / totals[1].clamp_min(1.0)
        )

    def symmetric_vp_distance(direction: Tensor, parameter_rms: float) -> dict[str, float]:
        distances: dict[str, float] = {}
        for sign, label in ((-1, "plus"), (1, "minus")):
            restore_anchor()
            _set_parameter_offset(
                parameters,
                anchor_parameters,
                direction,
                signed_rms=sign * float(parameter_rms),
            )
            output, output_mask = vp_output()
            if not torch.equal(path_mask, output_mask):
                raise RuntimeError("VP path mask changed during distance matching")
            distances[label] = vp_distance(output)
            del output, output_mask
        distances["symmetric_mean"] = 0.5 * (
            distances["plus"] + distances["minus"]
        )
        return distances

    vanilla_rms = float(cfg["vanilla_parameter_rms"])
    vanilla_distance = symmetric_vp_distance(gbar, vanilla_rms)
    target_distance = vanilla_distance["symmetric_mean"]
    if not math.isfinite(target_distance) or target_distance <= 0.0:
        raise ValueError("vanilla VP path distance is zero or non-finite")
    natural_rms = vanilla_rms
    natural_distance: dict[str, float] = {}
    for _ in range(int(cfg["distance_match_iterations"])):
        natural_distance = symmetric_vp_distance(natural_direction, natural_rms)
        observed = natural_distance["symmetric_mean"]
        if not math.isfinite(observed) or observed <= 0.0:
            raise ValueError("natural VP path distance is zero or non-finite")
        ratio = observed / target_distance
        if abs(ratio - 1.0) <= float(cfg["distance_match_tolerance"]):
            break
        natural_rms *= math.sqrt(target_distance / observed)
        natural_rms = min(
            float(cfg["maximum_parameter_rms"]),
            max(float(cfg["minimum_parameter_rms"]), natural_rms),
        )
    natural_distance = symmetric_vp_distance(natural_direction, natural_rms)
    distance_ratio = natural_distance["symmetric_mean"] / target_distance
    distance_match_passed = (
        abs(distance_ratio - 1.0) <= float(cfg["distance_match_tolerance"])
    )

    vanilla_step = gbar.double() * (vanilla_rms / vector_rms(gbar))
    natural_step = natural_direction.double() * (
        natural_rms / vector_rms(natural_direction)
    )
    vanilla_linear_descent = float(torch.dot(gbar.double(), vanilla_step))
    natural_linear_descent = float(torch.dot(gbar.double(), natural_step))
    linear_efficiency_gain = natural_linear_descent / max(
        vanilla_linear_descent, 1e-30
    )
    if rank == 0:
        print(
            f"[geometry] direction cosine={direction_cosine:.6f} "
            f"matched VP ratio={distance_ratio:.6f} natural_rms={natural_rms:.6g}",
            flush=True,
        )
        if wandb_run is not None:
            metric_values = {
                "geometry/metric/minimum_eigenvalue": natural_diagnostics[
                    "minimum_eigenvalue"
                ],
                "geometry/metric/maximum_eigenvalue": natural_diagnostics[
                    "maximum_eigenvalue"
                ],
                "geometry/metric/ridge": natural_diagnostics["ridge"],
                "geometry/metric/regularized_condition": natural_diagnostics[
                    "regularized_condition"
                ],
                "geometry/direction/cosine": direction_cosine,
                "geometry/direction/linear_efficiency_gain": linear_efficiency_gain,
                "geometry/distance/vanilla_symmetric_vp": target_distance,
                "geometry/distance/natural_symmetric_vp": natural_distance[
                    "symmetric_mean"
                ],
                "geometry/distance/natural_to_vanilla_ratio": distance_ratio,
                "geometry/distance/natural_parameter_rms": natural_rms,
                "geometry/distance/match_passed": float(distance_match_passed),
            }
            wandb_run.log(metric_values)
            wandb_run.summary["phase"] = "fixed_judge_signed_probes"

    judge_batch, judge_packed, judge_truth, _ = make_batch(local_judge)
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
    judge = builder.make_classifier(packing_spec, "block_natural_judge", reset=True).to(
        device
    )
    judge.load_state_dict(_load(reward_root / "independent_h4_judge.pt"), strict=True)
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    truth_logits_local = interface._score_local_on_device(
        judge, judge_packed, judge_truth, int(cfg["score_batch_size"])
    ).cpu()

    def evaluate_judge(
        label: str,
        direction: Tensor | None,
        *,
        descent_sign: int,
        parameter_rms: float,
    ) -> dict[str, Any]:
        restore_anchor()
        if direction is not None:
            if descent_sign not in (-1, 1):
                raise ValueError("signed judge probe must use -1 or +1")
            # descent_sign=+1 applies theta - normalized_direction.
            _set_parameter_offset(
                parameters,
                anchor_parameters,
                direction,
                signed_rms=-float(descent_sign) * float(parameter_rms),
            )
        torch.manual_seed(int(cfg["judge_rollout_seed"]) + rank)
        with torch.no_grad():
            candidates = generate_neutrino_candidates(
                policy,
                judge_batch,
                sampler,
                K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
                device=device,
                parallel_chains=int(
                    global_config.dgpo.get("rollout_parallel_chains", 1)
                ),
            )
        local_nonfinite = float((~torch.isfinite(candidates)).float().mean().cpu())
        nonfinite = interface._all_gather_object(local_nonfinite)
        candidates = torch.nan_to_num(candidates)
        sample = candidates.permute(1, 0, 2, 3).reshape(
            len(local_judge), int(cfg["K"]), 4
        ).cpu()
        generated_logits_local = interface._score_local_on_device(
            judge, judge_packed, sample, int(cfg["score_batch_size"])
        ).cpu()
        gathered_truth = interface._all_gather_object(truth_logits_local)
        gathered_generated = interface._all_gather_object(generated_logits_local)
        if rank != 0:
            return {}
        metrics = interface._classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_generated).reshape(-1)
        )
        return {
            "label": label,
            "descent_sign": descent_sign,
            "parameter_rms": float(parameter_rms) if direction is not None else 0.0,
            "judge": metrics,
            "judge_auc_gap": metrics["auc_gap"],
            "candidate_nonfinite_fraction": float(np.mean(nonfinite)),
            "candidate_nonfinite_fraction_max_rank": float(np.max(nonfinite)),
        }

    probes: dict[str, Any] = {}
    zero = evaluate_judge("zero", None, descent_sign=0, parameter_rms=0.0)
    arms = (
        ("vanilla", gbar, vanilla_rms),
        ("natural", natural_direction, natural_rms),
    )
    for arm_index, (name, direction, parameter_rms) in enumerate(arms):
        minus = evaluate_judge(
            f"{name}/minus",
            direction,
            descent_sign=-1,
            parameter_rms=parameter_rms,
        )
        plus = evaluate_judge(
            f"{name}/plus",
            direction,
            descent_sign=1,
            parameter_rms=parameter_rms,
        )
        if rank == 0:
            row = {
                "minus": minus,
                "zero": zero,
                "plus": plus,
                "minus_gap_delta": minus["judge_auc_gap"] - zero["judge_auc_gap"],
                "plus_gap_delta": plus["judge_auc_gap"] - zero["judge_auc_gap"],
                "plus_beats_minus": plus["judge_auc_gap"] < minus["judge_auc_gap"],
            }
            probes[name] = row
            print(
                f"[geometry] {name}: minus={minus['judge_auc_gap']:.6f} "
                f"zero={zero['judge_auc_gap']:.6f} plus={plus['judge_auc_gap']:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "geometry/progress/stage": 2.0 + float(arm_index),
                    "geometry/probe/index": arm_index,
                    f"geometry/probe/{name}/minus_judge_auc_gap": minus[
                        "judge_auc_gap"
                    ],
                    f"geometry/probe/{name}/zero_judge_auc_gap": zero[
                        "judge_auc_gap"
                    ],
                    f"geometry/probe/{name}/plus_judge_auc_gap": plus[
                        "judge_auc_gap"
                    ],
                    f"geometry/probe/{name}/plus_gap_delta": row["plus_gap_delta"],
                    f"geometry/probe/{name}/plus_beats_minus": float(
                        row["plus_beats_minus"]
                    ),
                })
    restore_anchor()

    if rank == 0:
        interface.verify_policy_loaded(policy, anchor_state)
        result = classify_result(
            zero_gap=zero["judge_auc_gap"],
            vanilla_plus_gap=probes["vanilla"]["plus"]["judge_auc_gap"],
            natural_plus_gap=probes["natural"]["plus"]["judge_auc_gap"],
            natural_minus_gap=probes["natural"]["minus"]["judge_auc_gap"],
            vp_distance_ratio=distance_ratio,
            distance_tolerance=float(cfg["distance_match_tolerance"]),
            decisive_gain=float(cfg["minimum_decisive_efficiency_gain"]),
        )
        report = {
            "schema": "c4a91e07-h4-block-natural-gradient-v1",
            "source_wandb_run": cfg["expected_source_wandb_run"],
            "source_gradient_wandb_run": cfg["expected_gradient_wandb_id"],
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0,
            "classifier_fits": 0,
            "objective_changed": False,
            "gradient_recomputed": False,
            "gradient_reconstruction_cosine": float(
                cfg["gradient_reconstruction_cosine"]
            ),
            "previous_events_excluded": True,
            "unused_final_audit_events_before_split": unused_count,
            "curvature_events": int(cfg["curvature_events"]),
            "judge_events": int(cfg["judge_events"]),
            "panels_disjoint": True,
            "K": int(cfg["K"]),
            "curvature_timesteps": timesteps,
            "path_normalizer": float(path_normalizer),
            "curvature_candidate_nonfinite_fraction": float(
                np.mean(candidate_nonfinite_all)
            ),
            "block_names": block_names,
            "basis_diagnostics": basis_diagnostics,
            "metric": metric.tolist(),
            "q": q.tolist(),
            "natural_coefficients": coefficients.tolist(),
            "natural_diagnostics": natural_diagnostics,
            "direction_cosine": direction_cosine,
            "linear_efficiency_gain": linear_efficiency_gain,
            "vanilla_parameter_rms": vanilla_rms,
            "natural_parameter_rms": natural_rms,
            "vanilla_vp_distance": vanilla_distance,
            "natural_vp_distance": natural_distance,
            "vp_distance_ratio": distance_ratio,
            "distance_match_passed": distance_match_passed,
            "signed_probes": probes,
            "result": result,
            "scope": (
                "Local target-preserving geometry audit. The H4 DGPO gradient is "
                "fixed from h4grad01; only the parameter metric changes. No optimizer "
                "step, classifier fit, reward transform, or cold audit is performed."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(output_root / "report.json", interface._jsonable(report))
        direction_path = output_root / "geometry_directions_fp16.pt"
        if cfg["save_direction_vectors"]:
            replay._exclusive_torch_save(direction_path, {
                "schema": "c4a91e07-h4-block-natural-directions-v1",
                "policy_global_step": int(cfg["expected_policy_step"]),
                "parameter_layout": cfg["parameter_layout"],
                "directions": {
                    "vanilla_gbar": gbar.half(),
                    "block_natural": natural_direction.half(),
                },
                "vanilla_parameter_rms": vanilla_rms,
                "natural_parameter_rms": natural_rms,
            })
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                "c4a91e07-h4-block-natural-gradient",
                type="diagnostic",
            )
            artifact.add_file(str(output_root / "manifest.json"))
            artifact.add_file(str(output_root / "report.json"))
            if cfg["upload_direction_vectors"]:
                artifact.add_file(str(direction_path))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete",
                "classifier_fits": 0,
                "policy_updates": 0,
                "geometry/primary/vanilla_plus_gap_delta": probes["vanilla"][
                    "plus_gap_delta"
                ],
                "geometry/primary/natural_plus_gap_delta": probes["natural"][
                    "plus_gap_delta"
                ],
                "geometry/primary/efficiency_gain": result["efficiency_gain"],
                "geometry/primary/vp_distance_ratio": distance_ratio,
                "geometry/primary/direction_cosine": direction_cosine,
                "geometry/primary/decision": result["decision"],
            })
            wandb_run.finish()
        print(f"Block-natural-gradient report: {output_root / 'report.json'}", flush=True)
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
        "Verified h4grad01 and geometry protocol: "
        f"{cfg['curvature_events']} curvature + {cfg['judge_events']} judge events, "
        "16 GPUs, zero optimizer steps.",
        flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"].get("required"):
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
        offline = os.environ.get("WANDB_MODE", "").lower() in {
            "offline",
            "disabled",
            "dryrun",
        }
        if disabled or offline:
            raise RuntimeError("this experiment requires live online W&B logging")
    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is unset; start/source the 16-GPU Ray cluster")
    output_root = Path(cfg["output_dir"])
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
            f"Ray cluster has {available_gpus:g} GPUs; experiment requires {cfg['workers']}"
        )
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
            name="c4a91e07-h4-block-natural-gradient-v1",
            storage_path=str(output_root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
