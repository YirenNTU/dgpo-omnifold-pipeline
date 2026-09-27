#!/usr/bin/env python3
"""Test whether low-noise DGPO time credit is diluted by time averaging.

The four equal time-band gradients decompose the unchanged production
expectation.  Band directions are diagnostic only: no classifier is fitted,
no optimizer step is taken, and no time-reweighted training objective is used.
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

from RL.DGPO_neutrino.reward_interface import reward_advantage_arms, vector_cosine


GRADIENT_ARTIFACTS = (
    "manifest.json",
    "report.json",
    "gradient_directions_fp16.pt",
)
GEOMETRY_ARTIFACTS = ("manifest.json", "report.json")
EXPECTED_BANDS = ("low", "mid_low", "mid_high", "high")


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
        "geometry_source_dir",
        "output_dir",
        "expected_reward_schema",
        "expected_gradient_schema",
        "expected_direction_schema",
        "expected_geometry_schema",
        "expected_source_wandb_run",
        "expected_gradient_wandb_id",
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "events_per_worker",
        "event_microbatch_size",
        "K",
        "beta",
        "timesteps_per_band",
        "time_bands",
        "candidate_seed",
        "gradient_seed",
        "minimum_split_cosine",
        "distance_candidate_seed",
        "distance_timesteps",
        "distance_path_seed",
        "full_parameter_rms",
        "distance_match_tolerance",
        "distance_match_iterations",
        "minimum_parameter_rms",
        "maximum_parameter_rms",
        "judge_events",
        "judge_selection_seed",
        "judge_rollout_seed",
        "score_batch_size",
        "minimum_decisive_efficiency_gain",
        "save_direction_vectors",
        "upload_direction_vectors",
        "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing diffusion-time settings: {missing}")
    integer_keys = (
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "events_per_worker",
        "event_microbatch_size",
        "K",
        "timesteps_per_band",
        "candidate_seed",
        "gradient_seed",
        "distance_candidate_seed",
        "distance_timesteps",
        "distance_path_seed",
        "distance_match_iterations",
        "judge_events",
        "judge_selection_seed",
        "judge_rollout_seed",
        "score_batch_size",
    )
    for key in integer_keys:
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    exact = {
        "workers": 16,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "K": 8,
        "timesteps_per_band": 8,
        "distance_timesteps": 8,
    }
    for key, expected in exact.items():
        if cfg[key] != expected:
            raise ValueError(f"this experiment requires {key}={expected}")
    if cfg["events_per_worker"] % 2:
        raise ValueError("events_per_worker must split into two equal halves")
    if (cfg["events_per_worker"] // 2) % cfg["event_microbatch_size"]:
        raise ValueError("each event half must divide by event_microbatch_size")
    if cfg["judge_events"] % cfg["workers"]:
        raise ValueError("judge_events must divide evenly over workers")

    float_keys = (
        "beta",
        "minimum_split_cosine",
        "full_parameter_rms",
        "distance_match_tolerance",
        "minimum_parameter_rms",
        "maximum_parameter_rms",
        "minimum_decisive_efficiency_gain",
    )
    for key in float_keys:
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]) or cfg[key] <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
    if cfg["minimum_split_cosine"] > 1.0:
        raise ValueError("minimum_split_cosine cannot exceed one")
    if cfg["distance_match_tolerance"] >= 1.0:
        raise ValueError("distance_match_tolerance must be smaller than one")
    if cfg["minimum_parameter_rms"] >= cfg["maximum_parameter_rms"]:
        raise ValueError("parameter RMS bounds are reversed")
    if not (
        cfg["minimum_parameter_rms"]
        <= cfg["full_parameter_rms"]
        <= cfg["maximum_parameter_rms"]
    ):
        raise ValueError("full_parameter_rms lies outside matching bounds")

    bands = [dict(row) for row in cfg["time_bands"]]
    if tuple(row.get("name") for row in bands) != EXPECTED_BANDS:
        raise ValueError(f"time bands must be ordered as {EXPECTED_BANDS}")
    widths: list[float] = []
    previous = None
    for row in bands:
        lo, hi = float(row.get("min", float("nan"))), float(
            row.get("max", float("nan"))
        )
        if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
            raise ValueError(f"invalid time band {row}")
        if previous is not None and not math.isclose(
            lo, previous, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("time bands must be contiguous")
        row["min"], row["max"] = lo, hi
        widths.append(hi - lo)
        previous = hi
    if not math.isclose(bands[0]["min"], 0.0, abs_tol=1.0e-12):
        raise ValueError("time bands must start at production t=0")
    if not math.isclose(bands[-1]["max"], 0.7, abs_tol=1.0e-12):
        raise ValueError("time bands must end at production t=0.7")
    if max(widths) - min(widths) > 1.0e-12:
        raise ValueError("time bands must have equal width")
    cfg["time_bands"] = bands

    for key in ("save_direction_vectors", "upload_direction_vectors"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg["upload_direction_vectors"] and not cfg["save_direction_vectors"]:
        raise ValueError("upload_direction_vectors requires saved vectors")
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


def select_protocol_indices(
    final_indices: Tensor,
    gradient_manifest: Mapping[str, Any],
    geometry_manifest: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Tensor]:
    """Reuse direction/curvature panels and reserve a never-before-used judge."""

    old_gradient, old_probe = gradient_repro.select_disjoint_event_indices(
        final_indices, gradient_manifest
    )
    old_curvature, old_geometry_judge, _ = geometry.select_unused_event_indices(
        final_indices, gradient_manifest, geometry_manifest
    )
    direction_count = int(cfg["workers"]) * int(cfg["events_per_worker"])
    direction = old_gradient[:direction_count]
    previously_used = torch.cat(
        (old_gradient, old_probe, old_curvature, old_geometry_judge)
    )
    remaining = final_indices[~torch.isin(final_indices, previously_used)]
    if int(remaining.numel()) < int(cfg["judge_events"]):
        raise ValueError(
            f"only {remaining.numel()} never-used events remain; "
            f"need {cfg['judge_events']}"
        )
    order = torch.randperm(
        len(remaining),
        generator=torch.Generator().manual_seed(int(cfg["judge_selection_seed"])),
    )
    judge = remaining[order[: int(cfg["judge_events"])]]
    panels = {
        "direction": direction,
        "distance": old_curvature,
        "judge": judge,
    }
    if any(
        int(torch.unique(value).numel()) != int(value.numel())
        for value in panels.values()
    ):
        raise ValueError("one of the selected panels contains duplicate identities")
    if not set(direction.tolist()).isdisjoint(judge.tolist()):
        raise RuntimeError("direction and judge identities overlap")
    if not set(old_curvature.tolist()).isdisjoint(judge.tolist()):
        raise RuntimeError("distance and judge identities overlap")
    panels["never_used_before_judge"] = torch.tensor(
        [len(remaining)], dtype=torch.long
    )
    return panels


def summarize_time_directions(
    half_gradients: Mapping[str, Sequence[Tensor]],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if tuple(half_gradients) != EXPECTED_BANDS:
        raise ValueError(f"half gradients must be ordered as {EXPECTED_BANDS}")
    means: dict[str, Tensor] = {}
    split_cosines: dict[str, float] = {}
    half_full = [None, None]
    for name, pair in half_gradients.items():
        if len(pair) != 2 or pair[0].shape != pair[1].shape:
            raise ValueError(f"{name} needs two equal-shaped split gradients")
        means[name] = 0.5 * (pair[0].double() + pair[1].double())
        split_cosines[name] = vector_cosine(pair[0], pair[1])
        for index in range(2):
            contribution = pair[index].double() / float(len(EXPECTED_BANDS))
            half_full[index] = (
                contribution
                if half_full[index] is None
                else half_full[index] + contribution
            )
    full = torch.stack([means[name] for name in EXPECTED_BANDS]).mean(0)
    rest = torch.stack([means[name] for name in EXPECTED_BANDS[1:]]).mean(0)
    directions = {name: means[name].float() for name in EXPECTED_BANDS}
    directions["rest"] = rest.float()
    directions["full"] = full.float()
    norms = {name: float(value.norm()) for name, value in means.items()}
    denominator = float(np.mean(list(norms.values())))
    cancellation_ratio = float(full.norm()) / max(denominator, 1.0e-30)
    pairwise = {
        f"{left}_{right}": vector_cosine(means[left], means[right])
        for left_index, left in enumerate(EXPECTED_BANDS)
        for right in EXPECTED_BANDS[left_index + 1 :]
    }
    diagnostics = {
        "split_cosines": split_cosines,
        "full_split_cosine": vector_cosine(half_full[0], half_full[1]),
        "band_norms": norms,
        "full_norm": float(full.norm()),
        "rest_norm": float(rest.norm()),
        "cancellation_ratio": cancellation_ratio,
        "pairwise_cosines": pairwise,
        "low_to_rest_cosine": vector_cosine(means["low"], rest),
    }
    return directions, diagnostics


def classify_result(
    *,
    low_split_cosine: float,
    full_split_cosine: float,
    minimum_split_cosine: float,
    full_to_saved_cosine: float,
    low_distance_matched: bool,
    rest_distance_matched: bool,
    full_plus_gap_delta: float,
    low_plus_gap_delta: float,
    rest_plus_gap_delta: float,
    low_plus_beats_minus: bool,
    decisive_gain: float,
) -> dict[str, Any]:
    full_improvement = -float(full_plus_gap_delta)
    low_improvement = -float(low_plus_gap_delta)
    rest_improvement = -float(rest_plus_gap_delta)
    gain = (
        low_improvement / full_improvement
        if full_improvement > 0.0
        else float("nan")
    )
    reproducible = (
        float(low_split_cosine) >= float(minimum_split_cosine)
        and float(full_split_cosine) >= float(minimum_split_cosine)
        and float(full_to_saved_cosine) >= float(minimum_split_cosine)
    )
    if (
        not reproducible
        or not low_distance_matched
        or not rest_distance_matched
        or not low_plus_beats_minus
    ):
        decision = "invalid_or_underpowered_time_decomposition"
    elif full_improvement <= 0.0:
        decision = "full_direction_failed_positive_control"
    elif (
        low_improvement > full_improvement
        and low_improvement > rest_improvement
        and gain >= float(decisive_gain)
    ):
        decision = "low_noise_credit_is_materially_diluted"
    elif low_improvement > full_improvement:
        decision = "low_noise_helps_but_is_not_decisive"
    else:
        decision = "low_noise_hypothesis_not_supported"
    return {
        "decision": decision,
        "full_improvement": full_improvement,
        "low_improvement": low_improvement,
        "rest_improvement": rest_improvement,
        "low_noise_efficiency_gain": gain,
        "split_reproducibility_passed": reproducible,
    }


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = _validated_settings(settings)
    reward_root = Path(cfg["reward_source_dir"]).expanduser().resolve(strict=True)
    gradient_root = Path(cfg["gradient_source_dir"]).expanduser().resolve(strict=True)
    geometry_root = Path(cfg["geometry_source_dir"]).expanduser().resolve(strict=True)
    gradient_paths = _artifact_paths(gradient_root, GRADIENT_ARTIFACTS)
    geometry_paths = _artifact_paths(geometry_root, GEOMETRY_ARTIFACTS)
    gradient_manifest = json.loads(gradient_paths["manifest.json"].read_text())
    gradient_report = json.loads(gradient_paths["report.json"].read_text())
    geometry_manifest = json.loads(geometry_paths["manifest.json"].read_text())
    geometry_report = json.loads(geometry_paths["report.json"].read_text())
    if gradient_report.get("schema") != cfg["expected_gradient_schema"]:
        raise ValueError("production-gradient schema mismatch")
    if gradient_manifest.get("wandb", {}).get("id") != cfg["expected_gradient_wandb_id"]:
        raise ValueError("production-gradient W&B identity mismatch")
    if gradient_report.get("production_variance_status") != "production_gradient_reproducible":
        raise ValueError("source production gradient did not pass reproducibility")
    if geometry_report.get("schema") != cfg["expected_geometry_schema"]:
        raise ValueError("geometry report schema mismatch")
    if geometry_report.get("source_gradient_wandb_run") != cfg["expected_gradient_wandb_id"]:
        raise ValueError("geometry source does not descend from h4grad01")
    for report in (gradient_report, geometry_report):
        if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
            raise ValueError("source policy step mismatch")
        if int(report.get("policy_updates", -1)) != 0:
            raise ValueError("source diagnostic unexpectedly updated the policy")
        if int(report.get("classifier_fits", -1)) != 0:
            raise ValueError("source diagnostic unexpectedly fitted a classifier")

    # Reuse the strict production-objective and reward-artifact validation from
    # h4grad01. Seeds here only reconstruct its immutable identity split.
    source_cfg = {
        "source_dir": str(reward_root),
        "output_dir": cfg["output_dir"],
        "expected_source_schema": cfg["expected_reward_schema"],
        "expected_source_wandb_run": cfg["expected_source_wandb_run"],
        "expected_policy_step": cfg["expected_policy_step"],
        "workers": cfg["workers"],
        "cpus_per_worker": cfg["cpus_per_worker"],
        "realizations": 4,
        "events_per_worker": cfg["events_per_worker"],
        "event_microbatch_size": cfg["event_microbatch_size"],
        "K": cfg["K"],
        "gradient_timesteps": cfg["timesteps_per_band"],
        "beta": cfg["beta"],
        "policy_eval_t_min": 0.0,
        "policy_eval_t_max": 0.7,
        "event_selection_seed": int(gradient_manifest["event_selection_seed"]),
        "candidate_seed": int(gradient_manifest["candidate_seed"]),
        "gradient_seed": int(gradient_manifest["gradient_seed"]),
        "probe_events": int(gradient_manifest["probe_events"]),
        "probe_selection_seed": int(gradient_manifest["probe_selection_seed"]),
        "probe_rollout_seed": int(gradient_manifest["probe_rollout_seed"]),
        "probe_epsilon_rms": float(gradient_manifest["probe_epsilon_rms"]),
        "score_batch_size": cfg["score_batch_size"],
        "time_diagnostic_events_per_worker": 0,
        "save_direction_vectors": False,
        "upload_direction_vectors": False,
        "wandb": {"enabled": False},
    }
    source = gradient_repro.prepare(source_cfg)
    if Path(source["policy_checkpoint"]).resolve() != Path(
        gradient_report["policy_checkpoint"]
    ).resolve():
        raise ValueError("reward and gradient diagnostics use different policies")

    direction_payload = _load(gradient_paths["gradient_directions_fp16.pt"])
    if direction_payload.get("schema") != cfg["expected_direction_schema"]:
        raise ValueError("saved production-direction schema mismatch")
    saved_gbar = direction_payload.get("directions", {}).get("gbar")
    if not isinstance(saved_gbar, Tensor) or not bool(torch.isfinite(saved_gbar).all()):
        raise ValueError("saved production gbar is missing or non-finite")
    del direction_payload, saved_gbar

    pool = _load(reward_root / "fixed_k1_pool.pt")
    final_indices = pool["partitions"]["final_audit"].long()
    panels = select_protocol_indices(
        final_indices, gradient_manifest, geometry_manifest, cfg
    )
    del pool

    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    for protected in (reward_root, gradient_root, geometry_root):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("output directory overlaps an immutable source")
    all_extra_paths = [*gradient_paths.values(), *geometry_paths.values()]
    source_stats = dict(source["source_stats"])
    source_stats.update({
        str(path): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in all_extra_paths
    })
    cfg.update(
        reward_source_dir=str(reward_root),
        gradient_source_dir=str(gradient_root),
        geometry_source_dir=str(geometry_root),
        output_dir=str(output),
        runtime_path=source["runtime_path"],
        policy_checkpoint=source["policy_checkpoint"],
        temperature=source["temperature"],
        member_order=source["member_order"],
        production_contract=source["production_contract"],
        gradient_manifest=gradient_manifest,
        geometry_manifest=geometry_manifest,
        parameter_layout=geometry_manifest["parameter_layout"],
        source_stats=source_stats,
        direction_events=int(panels["direction"].numel()),
        distance_events=int(panels["distance"].numel()),
        never_used_before_judge=int(panels["never_used_before_judge"].item()),
    )
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"source artifact changed during experiment: {path}")


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
        raise ValueError(f"expected {cfg['workers']} workers, found {world}")
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
                    "gradient_manifest",
                    "geometry_manifest",
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
            "time_alignment/progress/stage": 0.0,
            "time_alignment/contract/K": float(cfg["K"]),
            "time_alignment/contract/workers": float(cfg["workers"]),
            "time_alignment/contract/policy_updates": 0.0,
            "time_alignment/contract/classifier_fits": 0.0,
        })

    pool = _load(reward_root / "fixed_k1_pool.pt")
    packing_spec = EventPackingSpec.from_dict(pool["packing_spec"])
    final_indices = pool["partitions"]["final_audit"].long()
    panels = select_protocol_indices(
        final_indices, cfg["gradient_manifest"], cfg["geometry_manifest"], cfg
    )
    local_direction = panels["direction"][rank::world]
    local_distance = panels["distance"][rank::world]
    local_judge = panels["judge"][rank::world]
    expected_counts = (
        int(cfg["events_per_worker"]),
        int(cfg["distance_events"]) // world,
        int(cfg["judge_events"]) // world,
    )
    if tuple(map(len, (local_direction, local_distance, local_judge))) != expected_counts:
        raise RuntimeError("event sharding produced unexpected local panel sizes")

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
    parameters = tuple(
        parameter for parameter in policy.parameters() if parameter.requires_grad
    )
    blocks = geometry.validate_parameter_layout(
        cfg["parameter_layout"], list(policy.named_parameters()),
        int(cfg["parameter_layout"][-1]["stop"]),
    )
    del blocks

    def restore_anchor(*, training: bool) -> None:
        policy.load_state_dict(anchor_state, strict=True)
        policy.train(training)
        policy.zero_grad(set_to_none=True)

    def make_batch(indices: Tensor) -> tuple[dict[str, Any], Tensor, Tensor, Tensor]:
        packed = pool["packed_event"].index_select(0, indices)
        truth = pool["truth"].index_select(0, indices)
        noise_mask = pool["policy_noise_mask"].index_select(0, indices)
        batch = unpack_event_inputs(packed, packing_spec)
        batch["x_invisible"] = truth.reshape(-1, 2, 2)
        batch["x_invisible_mask"] = noise_mask
        return batch_to_device(batch, device), packed, truth, noise_mask

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
    member_states = [
        _load(reward_root / name) for name in gradient_repro.MEMBER_FILES
    ]
    sampler = DDIMSampler(device=device)

    # All time bands see the same anchor-policy candidates and event identities.
    direction_batch, direction_packed, _, _ = make_batch(local_direction)
    restore_anchor(training=True)
    torch.manual_seed(int(cfg["candidate_seed"]) + rank)
    with torch.no_grad():
        direction_candidates = generate_neutrino_candidates(
            policy,
            direction_batch,
            sampler,
            K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
            device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    direction_nonfinite = interface._all_gather_object(float(
        (~torch.isfinite(direction_candidates)).float().mean().cpu()
    ))
    direction_candidates = torch.nan_to_num(direction_candidates)
    sample_bk4 = direction_candidates.permute(1, 0, 2, 3).reshape(
        len(local_direction), int(cfg["K"]), 4
    ).cpu()
    logits: list[Tensor] = []
    for member_index, state in enumerate(member_states):
        logits.append(interface._score_local_state(
            builder,
            packing_spec,
            state,
            f"time_alignment_member{member_index + 1}",
            direction_packed,
            sample_bk4,
            int(cfg["score_batch_size"]),
            device,
        ).reshape(len(local_direction), int(cfg["K"])).T)
    ensemble_logits = torch.stack(logits).mean(0).to(device)
    advantage = reward_advantage_arms(
        ensemble_logits,
        temperature=float(cfg["temperature"]),
        raw_tempering=0.75,
    )["calibrated_loo"]
    gathered_advantage = interface._all_gather_object(advantage.cpu())
    if rank == 0:
        advantage_report = interface.advantage_metrics(
            torch.cat(gathered_advantage, dim=1)
        )
    else:
        advantage_report = {}

    half_events = int(cfg["events_per_worker"]) // 2
    half_gradients: dict[str, list[Tensor]] = {}
    band_reports: dict[str, Any] = {}
    half_full_accumulators: list[Tensor | None] = [None, None]
    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "time_band_gradients"
    for band_index, band in enumerate(cfg["time_bands"]):
        name = str(band["name"])
        halves: list[Tensor] = []
        reports: list[dict[str, Any]] = []
        for half in range(2):
            start, stop = half * half_events, (half + 1) * half_events
            half_cfg = dict(cfg)
            half_cfg.update(
                events_per_worker=half_events,
                gradient_timesteps=int(cfg["timesteps_per_band"]),
                policy_eval_t_min=float(band["min"]),
                policy_eval_t_max=float(band["max"]),
                gradient_seed=(
                    int(cfg["gradient_seed"])
                    + band_index * 100_000_007
                    + half * 10_000_019
                ),
                time_diagnostic_events_per_worker=0,
            )
            restore_anchor(training=True)
            gradient, half_report = gradient_repro._production_gradient(
                cfg=half_cfg,
                policy=policy,
                reference=reference,
                batch=gradient_repro._select_batch(
                    direction_batch, start, stop
                ),
                candidates=gradient_repro._select_candidates(
                    direction_candidates, start, stop
                ),
                advantages=advantage[:, start:stop],
                realization=half,
                rank=rank,
                world=world,
                device=device,
                dtype=next(policy.parameters()).dtype,
            )
            halves.append(gradient)
            reports.append(half_report)
            contribution = gradient.double() / float(len(EXPECTED_BANDS))
            half_full_accumulators[half] = (
                contribution
                if half_full_accumulators[half] is None
                else half_full_accumulators[half] + contribution
            )
        half_gradients[name] = halves
        if rank == 0:
            split_cosine = vector_cosine(halves[0], halves[1])
            band_reports[name] = {
                "interval": [float(band["min"]), float(band["max"])],
                "split_cosine": split_cosine,
                "halves": reports,
            }
            print(
                f"[time-alignment] {name} complete: split cosine={split_cosine:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "time_alignment/progress/stage": 1.0,
                    "time_alignment/band/index": float(band_index),
                    f"time_alignment/band/{name}/split_cosine": split_cosine,
                    f"time_alignment/band/{name}/half1_norm": reports[0]["gradient_norm"],
                    f"time_alignment/band/{name}/half2_norm": reports[1]["gradient_norm"],
                })

    directions, time_diagnostics = summarize_time_directions(half_gradients)
    if half_full_accumulators[0] is None or half_full_accumulators[1] is None:
        raise RuntimeError("failed to construct full split controls")
    # Cross-check the standalone summary implementation before interpretation.
    if not math.isclose(
        time_diagnostics["full_split_cosine"],
        vector_cosine(half_full_accumulators[0], half_full_accumulators[1]),
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise RuntimeError("full split-cosine reconstruction mismatch")
    saved_payload = _load(gradient_root / "gradient_directions_fp16.pt")
    saved_gbar = saved_payload["directions"]["gbar"].float()
    full_to_saved = vector_cosine(directions["full"], saved_gbar)
    del saved_payload
    if rank == 0:
        print(
            "[time-alignment] cancellation ratio="
            f"{time_diagnostics['cancellation_ratio']:.6f} "
            f"full-to-h4grad01={full_to_saved:.6f}",
            flush=True,
        )
        if wandb_run is not None:
            summary_metrics = {
                "time_alignment/gradient/cancellation_ratio": time_diagnostics[
                    "cancellation_ratio"
                ],
                "time_alignment/gradient/full_split_cosine": time_diagnostics[
                    "full_split_cosine"
                ],
                "time_alignment/gradient/low_to_rest_cosine": time_diagnostics[
                    "low_to_rest_cosine"
                ],
                "time_alignment/gradient/full_to_h4grad01_cosine": full_to_saved,
            }
            summary_metrics.update({
                f"time_alignment/gradient/pairwise/{key}": value
                for key, value in time_diagnostics["pairwise_cosines"].items()
            })
            wandb_run.log(summary_metrics)
            wandb_run.summary["phase"] = "matched_vp_distances"

    # Match all diagnostic directions to the full direction's measured VP path
    # movement on the curvature-only panel inherited from h4natg01.
    distance_batch, _, _, _ = make_batch(local_distance)
    restore_anchor(training=False)
    torch.manual_seed(int(cfg["distance_candidate_seed"]) + rank)
    with torch.no_grad():
        distance_candidates = generate_neutrino_candidates(
            policy,
            distance_batch,
            sampler,
            K=int(cfg["K"]),
            num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
            device=device,
            parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
        )
    distance_nonfinite = interface._all_gather_object(float(
        (~torch.isfinite(distance_candidates)).float().mean().cpu()
    ))
    distance_candidates = torch.nan_to_num(distance_candidates)
    local_distance_events = len(local_distance)
    path_generator = torch.Generator(device=device)
    path_generator.manual_seed(int(cfg["distance_path_seed"]) + rank)
    path_t, path_normalizer = sample_cosine_vp_path_kl_timesteps(
        int(cfg["distance_timesteps"]),
        local_distance_events,
        total_strata=int(cfg["distance_timesteps"]),
        stratum_offset=0,
        device=device,
        dtype=torch.float32,
        generator=path_generator,
    )
    n_nu, n_features = distance_candidates.shape[2:]
    base_eps = torch.randn(
        int(cfg["distance_timesteps"]),
        local_distance_events,
        int(n_nu),
        int(n_features),
        device=device,
        dtype=next(policy.parameters()).dtype,
        generator=path_generator,
    )
    path_eps = (
        base_eps.unsqueeze(1)
        .expand(
            int(cfg["distance_timesteps"]),
            int(cfg["K"]),
            local_distance_events,
            int(n_nu),
            int(n_features),
        )
        .reshape(
            int(cfg["distance_timesteps"]) * int(cfg["K"]) * local_distance_events,
            int(n_nu),
            int(n_features),
        )
    )
    anchor_parameters = tuple(
        parameter.detach().cpu().clone() for parameter in parameters
    )

    def vp_output() -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            values = policy_evaluation_step(
                policy,
                reference,
                distance_batch,
                distance_candidates,
                K=int(cfg["K"]),
                shared_noise=True,
                device=device,
                dtype=next(policy.parameters()).dtype,
                t=path_t,
                eps_rep=path_eps,
                t_min=0.0,
                t_max=1.0,
                num_timesteps=int(cfg["distance_timesteps"]),
            )
        return values[3].detach(), values[5].detach()

    restore_anchor(training=False)
    baseline_output, path_mask = vp_output()
    expanded_mask = path_mask.expand_as(baseline_output).to(baseline_output.dtype)

    def vp_distance(output: Tensor) -> float:
        local_sum = (
            (output - baseline_output).double().square() * expanded_mask.double()
        ).sum()
        totals = torch.tensor(
            [float(local_sum), float(output.shape[0])],
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        return float(
            0.5 * float(path_normalizer) * totals[0] / totals[1].clamp_min(1.0)
        )

    def symmetric_distance(direction: Tensor, rms: float) -> dict[str, float]:
        values: dict[str, float] = {}
        for signed_rms, label in ((-rms, "plus"), (rms, "minus")):
            restore_anchor(training=False)
            geometry._set_parameter_offset(
                parameters,
                anchor_parameters,
                direction,
                signed_rms=float(signed_rms),
            )
            output, mask = vp_output()
            if not torch.equal(mask, path_mask):
                raise RuntimeError("VP mask changed while matching directions")
            values[label] = vp_distance(output)
        values["symmetric_mean"] = 0.5 * (values["plus"] + values["minus"])
        return values

    full_rms = float(cfg["full_parameter_rms"])
    full_distance = symmetric_distance(directions["full"], full_rms)
    target_distance = full_distance["symmetric_mean"]
    if not math.isfinite(target_distance) or target_distance <= 0.0:
        raise ValueError("full-gradient VP distance is zero or non-finite")
    matched: dict[str, Any] = {
        "full": {
            "parameter_rms": full_rms,
            "distance": full_distance,
            "ratio": 1.0,
            "passed": True,
        }
    }
    arm_order = ("full", "low", "rest", "mid_low", "mid_high", "high")
    for name in arm_order[1:]:
        rms = full_rms
        observed: dict[str, float] = {}
        for _ in range(int(cfg["distance_match_iterations"])):
            observed = symmetric_distance(directions[name], rms)
            ratio = observed["symmetric_mean"] / target_distance
            if abs(ratio - 1.0) <= float(cfg["distance_match_tolerance"]):
                break
            rms *= math.sqrt(target_distance / max(observed["symmetric_mean"], 1.0e-30))
            rms = min(
                float(cfg["maximum_parameter_rms"]),
                max(float(cfg["minimum_parameter_rms"]), rms),
            )
        observed = symmetric_distance(directions[name], rms)
        ratio = observed["symmetric_mean"] / target_distance
        matched[name] = {
            "parameter_rms": rms,
            "distance": observed,
            "ratio": ratio,
            "passed": abs(ratio - 1.0) <= float(cfg["distance_match_tolerance"]),
        }
        if rank == 0 and wandb_run is not None:
            wandb_run.log({
                "time_alignment/progress/stage": 2.0,
                f"time_alignment/distance/{name}/parameter_rms": rms,
                f"time_alignment/distance/{name}/ratio": ratio,
                f"time_alignment/distance/{name}/passed": float(matched[name]["passed"]),
            })

    # The primary fixed judge is disjoint from every previous diagnostic panel.
    judge_batch, judge_packed, judge_truth, _ = make_batch(local_judge)
    builder.restore_pretrained_body()
    judge = builder.make_classifier(
        packing_spec, "time_alignment_judge", reset=True
    ).to(device)
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
        restore_anchor(training=False)
        if direction is not None:
            geometry._set_parameter_offset(
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
        nonfinite = interface._all_gather_object(float(
            (~torch.isfinite(candidates)).float().mean().cpu()
        ))
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
            "parameter_rms": 0.0 if direction is None else float(parameter_rms),
            "judge": metrics,
            "judge_auc_gap": metrics["auc_gap"],
            "candidate_nonfinite_fraction": float(np.mean(nonfinite)),
            "candidate_nonfinite_fraction_max_rank": float(np.max(nonfinite)),
        }

    if rank == 0 and wandb_run is not None:
        wandb_run.summary["phase"] = "signed_fixed_judge_probes"
    zero = evaluate_judge("zero", None, descent_sign=0, parameter_rms=0.0)
    probes: dict[str, Any] = {}
    for arm_index, name in enumerate(arm_order):
        minus = evaluate_judge(
            f"{name}/minus",
            directions[name],
            descent_sign=-1,
            parameter_rms=float(matched[name]["parameter_rms"]),
        )
        plus = evaluate_judge(
            f"{name}/plus",
            directions[name],
            descent_sign=1,
            parameter_rms=float(matched[name]["parameter_rms"]),
        )
        if rank == 0:
            probes[name] = {
                "minus": minus,
                "zero": zero,
                "plus": plus,
                "minus_gap_delta": minus["judge_auc_gap"] - zero["judge_auc_gap"],
                "plus_gap_delta": plus["judge_auc_gap"] - zero["judge_auc_gap"],
                "plus_beats_minus": plus["judge_auc_gap"] < minus["judge_auc_gap"],
            }
            print(
                f"[time-alignment] {name}: minus={minus['judge_auc_gap']:.6f} "
                f"zero={zero['judge_auc_gap']:.6f} plus={plus['judge_auc_gap']:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "time_alignment/progress/stage": 3.0,
                    "time_alignment/probe/index": float(arm_index),
                    f"time_alignment/probe/{name}/minus_judge_auc_gap": minus[
                        "judge_auc_gap"
                    ],
                    f"time_alignment/probe/{name}/zero_judge_auc_gap": zero[
                        "judge_auc_gap"
                    ],
                    f"time_alignment/probe/{name}/plus_judge_auc_gap": plus[
                        "judge_auc_gap"
                    ],
                    f"time_alignment/probe/{name}/plus_gap_delta": probes[name][
                        "plus_gap_delta"
                    ],
                    f"time_alignment/probe/{name}/plus_beats_minus": float(
                        probes[name]["plus_beats_minus"]
                    ),
                })
    restore_anchor(training=False)

    if rank == 0:
        interface.verify_policy_loaded(policy, anchor_state)
        result = classify_result(
            low_split_cosine=time_diagnostics["split_cosines"]["low"],
            full_split_cosine=time_diagnostics["full_split_cosine"],
            minimum_split_cosine=float(cfg["minimum_split_cosine"]),
            full_to_saved_cosine=full_to_saved,
            low_distance_matched=bool(matched["low"]["passed"]),
            rest_distance_matched=bool(matched["rest"]["passed"]),
            full_plus_gap_delta=probes["full"]["plus_gap_delta"],
            low_plus_gap_delta=probes["low"]["plus_gap_delta"],
            rest_plus_gap_delta=probes["rest"]["plus_gap_delta"],
            low_plus_beats_minus=bool(probes["low"]["plus_beats_minus"]),
            decisive_gain=float(cfg["minimum_decisive_efficiency_gain"]),
        )
        report = {
            "schema": "c4a91e07-h4-time-stratum-alignment-v1",
            "source_wandb_run": cfg["expected_source_wandb_run"],
            "source_gradient_wandb_run": cfg["expected_gradient_wandb_id"],
            "source_geometry_wandb_run": "h4natg01",
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0,
            "classifier_fits": 0,
            "objective_changed": False,
            "time_reweighted_policy_update": False,
            "direction_events": int(cfg["direction_events"]),
            "direction_event_halves": 2,
            "direction_reuses_h4grad01_g1_identities": True,
            "distance_events": int(cfg["distance_events"]),
            "distance_reuses_h4natg01_curvature_identities": True,
            "judge_events": int(cfg["judge_events"]),
            "judge_was_never_used_by_h4grad01_or_h4natg01": True,
            "never_used_events_before_judge_selection": int(
                cfg["never_used_before_judge"]
            ),
            "panels_disjoint_from_judge": True,
            "K": int(cfg["K"]),
            "time_bands": cfg["time_bands"],
            "timesteps_per_band": int(cfg["timesteps_per_band"]),
            "advantage": advantage_report,
            "direction_candidate_nonfinite_fraction": float(
                np.mean(direction_nonfinite)
            ),
            "distance_candidate_nonfinite_fraction": float(
                np.mean(distance_nonfinite)
            ),
            "band_reports": band_reports,
            "time_diagnostics": time_diagnostics,
            "full_to_saved_h4grad01_cosine": full_to_saved,
            "path_normalizer": float(path_normalizer),
            "matched_distances": matched,
            "signed_probes": probes,
            "result": result,
            "scope": (
                "Read-only decomposition of the unchanged production time expectation. "
                "Band directions are diagnostic only; no time-reweighted objective, "
                "optimizer step, classifier fit, or cold audit is performed."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(
            output_root / "report.json", interface._jsonable(report)
        )
        direction_path = output_root / "time_stratum_directions_fp16.pt"
        if cfg["save_direction_vectors"]:
            replay._exclusive_torch_save(direction_path, {
                "schema": "c4a91e07-h4-time-stratum-directions-v1",
                "policy_global_step": int(cfg["expected_policy_step"]),
                "parameter_layout": cfg["parameter_layout"],
                "directions": {
                    name: value.half() for name, value in directions.items()
                },
                "matched_parameter_rms": {
                    name: float(row["parameter_rms"])
                    for name, row in matched.items()
                },
            })
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                "c4a91e07-h4-time-stratum-alignment", type="diagnostic"
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
                "time_alignment/primary/decision": result["decision"],
                "time_alignment/primary/low_noise_efficiency_gain": result[
                    "low_noise_efficiency_gain"
                ],
                "time_alignment/primary/full_plus_gap_delta": probes["full"][
                    "plus_gap_delta"
                ],
                "time_alignment/primary/low_plus_gap_delta": probes["low"][
                    "plus_gap_delta"
                ],
                "time_alignment/primary/rest_plus_gap_delta": probes["rest"][
                    "plus_gap_delta"
                ],
                "time_alignment/primary/low_split_cosine": time_diagnostics[
                    "split_cosines"
                ]["low"],
                "time_alignment/primary/full_split_cosine": time_diagnostics[
                    "full_split_cosine"
                ],
                "time_alignment/primary/low_to_rest_cosine": time_diagnostics[
                    "low_to_rest_cosine"
                ],
                "time_alignment/primary/cancellation_ratio": time_diagnostics[
                    "cancellation_ratio"
                ],
                "time_alignment/primary/full_to_h4grad01_cosine": full_to_saved,
                "time_alignment/primary/low_vp_distance_ratio": matched["low"][
                    "ratio"
                ],
            })
            wandb_run.finish()
        print(
            f"Time-stratum alignment report: {output_root / 'report.json'}",
            flush=True,
        )
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
        "Verified low-noise time decomposition: "
        f"{cfg['direction_events']} direction events, "
        f"{cfg['distance_events']} distance events, "
        f"{cfg['judge_events']} new judge events, zero policy updates.",
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
            f"Ray cluster has {available_gpus:g} GPUs; requires {cfg['workers']}"
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
            name="c4a91e07-h4-low-noise-time-alignment-v1",
            storage_path=str(output_root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
