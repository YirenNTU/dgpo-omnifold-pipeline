#!/usr/bin/env python3
"""Audit four independent production-sized gradients of the exact DGPO loss.

The experiment reuses the completed H4 reward-interface artifacts.  It fits no
classifier and performs no optimizer step.  Each gradient uses a disjoint event
batch, fresh K-candidate rollout, eight independently sampled diffusion times,
and one 16-rank reduction.  The four gradients are averaged only after each
full production estimate has been completed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_raw_monitor_replay as replay
import diagnose_reward_interface as interface
from ablate_raw_monitor_initialization import clone_state

from RL.DGPO_neutrino.dgpo_utils import (
    build_dgpo_loss,
    build_reference_trust_loss,
)
from RL.DGPO_neutrino.reward_interface import reward_advantage_arms, vector_cosine


MEMBER_FILES = (
    "classifier_reward_s20260913_f1.pt",
    "classifier_reward_s20260913_f2.pt",
    "classifier_reward_s20260914_f1.pt",
    "classifier_reward_s20260914_f2.pt",
)
REQUIRED_ARTIFACTS = (
    "runtime.yaml",
    "manifest.json",
    "report.json",
    "fixed_k1_pool.pt",
    "fixed_k8_panel.pt",
    *MEMBER_FILES,
    "independent_h4_judge.pt",
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
        "source_dir",
        "output_dir",
        "expected_source_schema",
        "expected_source_wandb_run",
        "expected_policy_step",
        "workers",
        "cpus_per_worker",
        "realizations",
        "events_per_worker",
        "event_microbatch_size",
        "K",
        "gradient_timesteps",
        "beta",
        "policy_eval_t_min",
        "policy_eval_t_max",
        "event_selection_seed",
        "candidate_seed",
        "gradient_seed",
        "probe_events",
        "probe_selection_seed",
        "probe_rollout_seed",
        "probe_epsilon_rms",
        "score_batch_size",
        "time_diagnostic_events_per_worker",
        "save_direction_vectors",
        "upload_direction_vectors",
        "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing production-gradient settings: {missing}")
    integer_keys = (
        "workers",
        "cpus_per_worker",
        "realizations",
        "events_per_worker",
        "event_microbatch_size",
        "K",
        "gradient_timesteps",
        "event_selection_seed",
        "candidate_seed",
        "gradient_seed",
        "probe_events",
        "probe_selection_seed",
        "probe_rollout_seed",
        "score_batch_size",
        "time_diagnostic_events_per_worker",
    )
    for key in integer_keys:
        minimum = 0 if key == "time_diagnostic_events_per_worker" else 1
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if cfg["realizations"] != 4:
        raise ValueError("this predeclared experiment requires exactly four realizations")
    exact_integer_contract = {
        "workers": 16,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "K": 8,
        "gradient_timesteps": 8,
    }
    for key, expected in exact_integer_contract.items():
        if cfg[key] != expected:
            raise ValueError(
                f"this predeclared production audit requires {key}={expected}, "
                f"found {cfg[key]}"
            )
    if cfg["K"] < 2:
        raise ValueError("K must be at least two")
    if cfg["events_per_worker"] % cfg["event_microbatch_size"]:
        raise ValueError("events_per_worker must be divisible by event_microbatch_size")
    if cfg["time_diagnostic_events_per_worker"] > cfg["events_per_worker"]:
        raise ValueError("time diagnostic cannot exceed the production event batch")
    if cfg["probe_events"] % cfg["workers"]:
        raise ValueError("probe_events must be divisible by workers")
    for key in ("beta", "policy_eval_t_min", "policy_eval_t_max", "probe_epsilon_rms"):
        cfg[key] = float(cfg[key])
        if not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")
    if not 0.0 <= cfg["policy_eval_t_min"] < cfg["policy_eval_t_max"] <= 1.0:
        raise ValueError("policy-evaluation time range must satisfy 0 <= min < max <= 1")
    if cfg["probe_epsilon_rms"] <= 0.0:
        raise ValueError("probe_epsilon_rms must be positive")
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


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable v5 artifacts and production-size contract."""

    cfg = _validated_settings(settings)
    source = Path(cfg["source_dir"]).expanduser().resolve(strict=True)
    paths = {name: (source / name).resolve(strict=True) for name in REQUIRED_ARTIFACTS}
    report = json.loads(paths["report.json"].read_text())
    manifest = json.loads(paths["manifest.json"].read_text())
    if report.get("schema") != cfg["expected_source_schema"]:
        raise ValueError(
            f"source schema mismatch: expected {cfg['expected_source_schema']!r}, "
            f"found {report.get('schema')!r}"
        )
    if report.get("source_wandb_run") != cfg["expected_source_wandb_run"]:
        raise ValueError(
            "source W&B provenance mismatch: "
            f"expected {cfg['expected_source_wandb_run']!r}, "
            f"found {report.get('source_wandb_run')!r}"
        )
    if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("source report policy step does not match the declared anchor")
    if not report.get("fixed_policy") or not report.get("policy_unchanged"):
        raise ValueError("source diagnostic did not certify an unchanged fixed policy")
    member_order = [
        {"seed": int(row["seed"]), "fold": int(row["fold"])}
        for row in report.get("classifier_members", [])
    ]
    expected_order = [
        {"seed": 20260913, "fold": 1},
        {"seed": 20260913, "fold": 2},
        {"seed": 20260914, "fold": 1},
        {"seed": 20260914, "fold": 2},
    ]
    if member_order != expected_order:
        raise ValueError(f"unexpected source ensemble: {member_order}")
    temperature = float(report["classifier_calibration"]["primary"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("source primary calibration temperature is invalid")
    policy_checkpoint = Path(report["policy_checkpoint"]).resolve(strict=True)
    checkpoint = replay.read_checkpoint(policy_checkpoint)
    if int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("live policy checkpoint step differs from the declared anchor")
    del checkpoint

    runtime = __import__("yaml").safe_load(paths["runtime.yaml"].read_text())
    dgpo = runtime["dgpo"]
    if dgpo.get("checkpoint_load_mode") != "weights_only":
        raise ValueError("source runtime must use weights_only policy loading")
    reference_trust = dict(dgpo.get("reference_trust") or {})
    adaptive_boundary = dict(reference_trust.get("adaptive_boundary") or {})
    contract = {
        "K": int(dgpo["K"]),
        "gradient_timesteps": int(dgpo["num_train_timesteps"]),
        "events_per_worker": int(runtime["platform"]["batch_size"]),
        "event_microbatch_size": int(dgpo["policy_eval_event_microbatch_size"]),
        "beta": float(dgpo["beta"]),
        "policy_eval_t_min": float(dgpo["policy_eval_t_min"]),
        "policy_eval_t_max": float(dgpo["policy_eval_t_max"]),
        "advantage_estimator": str(dgpo["advantage_estimator"]),
        "beta_kl": float(dgpo.get("beta_kl", 0.0)),
        "adv_clip_max": dgpo.get("adv_clip_max"),
        "variance_regularization_enabled": bool(
            (dgpo.get("variance_regularization") or {}).get("enabled", False)
        ),
        "projection_constraint_type": str(
            (dgpo.get("projection_constraint") or {}).get("type", "none")
        ),
        "reference_trust_enabled": bool(reference_trust.get("enabled", False)),
        "reference_trust_coefficient": float(
            reference_trust.get("coefficient", 0.0)
        ),
        "reference_trust_objective": str(
            reference_trust.get("objective", "velocity_mse")
        ),
        "reference_trust_adaptive_boundary_enabled": bool(
            adaptive_boundary.get("enabled", False)
        ),
        "sequential_vp_trust_backward": bool(
            dgpo.get("sequential_vp_trust_backward", False)
        ),
    }
    requested = {
        "K": cfg["K"],
        "gradient_timesteps": cfg["gradient_timesteps"],
        "events_per_worker": cfg["events_per_worker"],
        "event_microbatch_size": cfg["event_microbatch_size"],
        "beta": cfg["beta"],
        "policy_eval_t_min": cfg["policy_eval_t_min"],
        "policy_eval_t_max": cfg["policy_eval_t_max"],
    }
    for key, value in requested.items():
        if isinstance(value, float):
            matches = math.isclose(float(contract[key]), value, rel_tol=0.0, abs_tol=1e-12)
        else:
            matches = contract[key] == value
        if not matches:
            raise ValueError(
                f"requested {key}={value!r} does not match production {contract[key]!r}"
            )
    if contract["advantage_estimator"] != "leave_one_out_unscaled":
        raise ValueError("production reward no longer uses leave_one_out_unscaled")
    if contract["beta_kl"] != 0.0 or contract["adv_clip_max"] is not None:
        raise ValueError("the source production objective has unexpected KL or clipping")
    if contract["variance_regularization_enabled"]:
        raise ValueError("variance regularization would change the audited gradient")
    if contract["projection_constraint_type"].lower() != "none":
        raise ValueError("projection repair is outside the pre-AdamW gradient contract")
    if (
        not contract["reference_trust_enabled"]
        or not math.isclose(
            contract["reference_trust_coefficient"],
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or contract["reference_trust_objective"] != "velocity_mse"
    ):
        raise ValueError(
            "source production objective must use coefficient-1 velocity_mse "
            "reference trust"
        )
    if contract["reference_trust_adaptive_boundary_enabled"]:
        raise ValueError("hard reference-trust boundary is outside this audit")
    if contract["sequential_vp_trust_backward"]:
        raise ValueError("unexpected sequential VP-trust backward in source runtime")

    pool = _load(paths["fixed_k1_pool.pt"])
    final_audit = pool.get("partitions", {}).get("final_audit")
    if not isinstance(final_audit, Tensor) or final_audit.ndim != 1:
        raise ValueError("fixed K=1 pool lacks a one-dimensional final_audit partition")
    needed = (
        cfg["realizations"] * cfg["events_per_worker"] * cfg["workers"]
        + cfg["probe_events"]
    )
    if int(final_audit.numel()) < needed:
        raise ValueError(
            f"final_audit has {final_audit.numel()} events but the disjoint protocol needs {needed}"
        )
    del pool

    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output directory overlaps the immutable source artifacts")
    cfg.update(
        source_dir=str(source),
        output_dir=str(output),
        runtime_path=str(paths["runtime.yaml"]),
        policy_checkpoint=str(policy_checkpoint),
        temperature=temperature,
        member_order=member_order,
        source_manifest=manifest,
        production_contract=contract,
        total_events_per_gradient=cfg["workers"] * cfg["events_per_worker"],
        source_stats={
            str(path): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in paths.values()
        },
    )
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"source artifact changed during experiment: {path}")


def select_disjoint_event_indices(
    final_indices: Tensor, cfg: Mapping[str, Any]
) -> tuple[Tensor, Tensor]:
    """Draw disjoint production-gradient and signed-probe identities."""

    gradient_total = (
        int(cfg["realizations"])
        * int(cfg["events_per_worker"])
        * int(cfg["workers"])
    )
    needed = gradient_total + int(cfg["probe_events"])
    if final_indices.ndim != 1 or int(final_indices.numel()) < needed:
        raise ValueError(
            f"final_audit needs at least {needed} event identities for this protocol"
        )
    if int(torch.unique(final_indices).numel()) != int(final_indices.numel()):
        raise ValueError("final_audit event identities must be unique")
    selection = final_indices[torch.randperm(
        len(final_indices),
        generator=torch.Generator().manual_seed(int(cfg["event_selection_seed"])),
    )]
    gradient_indices = selection[:gradient_total]
    remaining = selection[gradient_total:]
    probe_order = torch.randperm(
        len(remaining),
        generator=torch.Generator().manual_seed(int(cfg["probe_selection_seed"])),
    )
    probe_indices = remaining[probe_order[:int(cfg["probe_events"])]]
    return gradient_indices, probe_indices


def _flat_parameter_gradient(parameters: Sequence[Tensor]) -> Tensor:
    return torch.cat([
        (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach()
        ).float().reshape(-1)
        for parameter in parameters
    ])


def _flat_autograd_gradient(loss: Tensor, parameters: Sequence[Tensor]) -> Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([
        (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient.detach()
        ).float().reshape(-1)
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ])


def _component_name(parameter_name: str) -> str:
    lowered = parameter_name.lower()
    patterns = (
        ("InvisibleInputProjector", ("invisibleinputprojector", "invisible_input_projector")),
        ("GroupedSequentialEmbedding", ("groupedsequentialembedding", "grouped_sequential")),
        ("GlobalEmbedding", ("globalembedding", "global_embedding")),
        ("ObjectEncoder", ("objectencoder", "object_encoder")),
        ("TruthGeneration", ("truthgeneration", "truth_generation")),
        ("PET", ("pet.", "pet.body", "petbody")),
    )
    for label, needles in patterns:
        if any(needle in lowered for needle in needles):
            return label
    pieces = parameter_name.removeprefix("module.").split(".")
    return pieces[0] if pieces else "other"


def parameter_layout(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    layout: list[dict[str, Any]] = []
    blocks: dict[str, list[tuple[int, int]]] = {}
    offset = 0
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        stop = offset + parameter.numel()
        block = _component_name(name)
        layout.append({
            "name": name,
            "shape": list(parameter.shape),
            "start": offset,
            "stop": stop,
            "block": block,
        })
        blocks.setdefault(block, []).append((offset, stop))
        offset = stop
    return layout, blocks


def _sliced_cosine(left: Tensor, right: Tensor, ranges: Sequence[tuple[int, int]]) -> float:
    dot = left.new_zeros((), dtype=torch.float64)
    left_sq = dot.clone()
    right_sq = dot.clone()
    for start, stop in ranges:
        a = left[start:stop].double()
        b = right[start:stop].double()
        dot += torch.dot(a, b)
        left_sq += torch.dot(a, a)
        right_sq += torch.dot(b, b)
    denominator = torch.sqrt(left_sq * right_sq)
    if not torch.isfinite(denominator) or float(denominator) <= 1.0e-30:
        return float("nan")
    return float(dot / denominator)


def _sliced_norm(vector: Tensor, ranges: Sequence[tuple[int, int]]) -> float:
    total = vector.new_zeros((), dtype=torch.float64)
    for start, stop in ranges:
        values = vector[start:stop].double()
        total += torch.dot(values, values)
    return float(torch.sqrt(total))


def _finite_cosines(vectors: Sequence[Tensor]) -> list[float]:
    return [
        value
        for left_index, left in enumerate(vectors)
        for right in vectors[left_index + 1 :]
        if math.isfinite(value := vector_cosine(left, right))
    ]


def summarize_reproducibility(
    gradients: Sequence[Tensor],
    blocks: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    if len(gradients) != 4:
        raise ValueError("reproducibility summary requires four gradients")
    pairwise: dict[str, float] = {}
    values: list[float] = []
    for left_index, left in enumerate(gradients):
        for right_index in range(left_index + 1, len(gradients)):
            value = vector_cosine(left, gradients[right_index])
            pairwise[f"g{left_index + 1}_g{right_index + 1}"] = value
            if math.isfinite(value):
                values.append(value)
    if not values:
        raise ValueError("all production-gradient cosine values are undefined")
    half_left = (gradients[0].double() + gradients[1].double()) / 2.0
    half_right = (gradients[2].double() + gradients[3].double()) / 2.0
    mean64 = torch.zeros_like(gradients[0], dtype=torch.float64)
    for gradient in gradients:
        mean64.add_(gradient.double(), alpha=0.25)
    mean = mean64.float()
    report: dict[str, Any] = {
        "pairwise": pairwise,
        "mean_pairwise_cosine": float(np.mean(values)),
        "min_pairwise_cosine": float(np.min(values)),
        "half_mean_cosine": vector_cosine(half_left, half_right),
        "gradient_norms": [float(gradient.double().norm()) for gradient in gradients],
        "gbar_norm": float(mean64.norm()),
        "cosine_to_gbar": [vector_cosine(gradient, mean) for gradient in gradients],
    }
    block_report: dict[str, Any] = {}
    for name, ranges in (blocks or {}).items():
        block_values = [
            _sliced_cosine(gradients[left], gradients[right], ranges)
            for left in range(4)
            for right in range(left + 1, 4)
        ]
        finite = [value for value in block_values if math.isfinite(value)]
        block_report[name] = {
            "mean_pairwise_cosine": (
                float(np.mean(finite)) if finite else float("nan")
            ),
            "min_pairwise_cosine": (
                float(np.min(finite)) if finite else float("nan")
            ),
            "gradient_norms": [
                _sliced_norm(gradient, ranges) for gradient in gradients
            ],
            "gbar_norm": _sliced_norm(mean, ranges),
        }
    report["blocks"] = block_report
    report["gbar"] = mean
    return report


def production_variance_status(mean_cosine: float, min_cosine: float) -> str:
    if mean_cosine >= 0.90 and min_cosine >= 0.80:
        return "production_gradient_reproducible"
    if mean_cosine >= 0.70:
        return "material_residual_variance"
    return "production_gradient_unstable"


def diagnose(
    reproducibility: Mapping[str, Any], probes: Mapping[str, Mapping[str, Any]]
) -> str:
    individual_deltas = [
        float(probes[f"g{index}"]["plus_gap_delta"]) for index in range(1, 5)
    ]
    plus_beats_minus = [
        float(probes[f"g{index}"]["plus"]["judge_auc_gap"])
        < float(probes[f"g{index}"]["minus"]["judge_auc_gap"])
        for index in range(1, 5)
    ]
    gbar = probes["gbar"]
    if not any(plus_beats_minus) and (
        float(gbar["plus"]["judge_auc_gap"])
        >= float(gbar["minus"]["judge_auc_gap"])
    ):
        return "signed_backward_contract_inconsistent"
    status = production_variance_status(
        float(reproducibility["mean_pairwise_cosine"]),
        float(reproducibility["min_pairwise_cosine"]),
    )
    gbar_delta = float(gbar["plus_gap_delta"])
    median_delta = float(np.median(individual_deltas))
    if status != "production_gradient_reproducible" and gbar_delta < median_delta:
        return "monte_carlo_variance_materially_loses_h4_signal"
    if status == "production_gradient_reproducible":
        return "expected_dgpo_direction_is_intrinsically_weak_or_poorly_conditioned"
    return "averaging_alone_does_not_recover_policy_transfer"


def _tensor_summary(values: Tensor) -> dict[str, float]:
    flat = values.detach().double().reshape(-1).cpu()
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "p05": float(torch.quantile(flat, 0.05)),
        "p50": float(torch.quantile(flat, 0.50)),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(flat.max()),
    }


def _select_batch(batch: Mapping[str, Any], start: int, stop: int) -> dict[str, Any]:
    size = int(next(value for value in batch.values() if isinstance(value, Tensor)).shape[0])
    return {
        key: (
            value[start:stop]
            if isinstance(value, Tensor) and value.ndim and int(value.shape[0]) == size
            else value
        )
        for key, value in batch.items()
    }


def _select_candidates(candidates: Tensor, start: int, stop: int) -> Tensor:
    return candidates[:, start:stop]


def _time_stratum_metrics(
    *,
    cfg: Mapping[str, Any],
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    batch: Mapping[str, Any],
    candidates: Tensor,
    advantages: Tensor,
    parameters: Sequence[Tensor],
    realization: int,
    rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    from RL.DGPO_neutrino.dgpo_trainer import policy_evaluation_step

    events = int(cfg["time_diagnostic_events_per_worker"])
    if events == 0:
        return {"mean_rank_local_cosine": float("nan"), "min_rank_local_cosine": float("nan")}
    small_batch = _select_batch(batch, 0, events)
    small_candidates = _select_candidates(candidates, 0, events)
    small_advantage = advantages[:, :events]
    vectors: list[Tensor] = []
    for timestep in range(int(cfg["gradient_timesteps"])):
        torch.manual_seed(
            int(cfg["gradient_seed"])
            + realization * 10_000_019
            + rank * 100_003
            + timestep * 1_009
            + 700_000_001
        )
        values = policy_evaluation_step(
            policy,
            reference,
            small_batch,
            small_candidates,
            K=int(cfg["K"]),
            shared_noise=True,
            device=device,
            dtype=dtype,
            t_min=float(cfg["policy_eval_t_min"]),
            t_max=float(cfg["policy_eval_t_max"]),
            num_timesteps=1,
        )
        loss_dgpo = build_dgpo_loss(
            values[0], values[1], small_advantage,
            beta_dgpo=float(cfg["beta"]), K=int(cfg["K"]),
        )[0]
        trust_loss = build_reference_trust_loss(
            values[3],
            values[4],
            values[5],
            L_ref_2d=values[1],
            objective=str(cfg["production_contract"]["reference_trust_objective"]),
        )[0]
        loss = loss_dgpo + (
            float(cfg["production_contract"]["reference_trust_coefficient"])
            * trust_loss
        )
        vectors.append(_flat_autograd_gradient(loss, parameters).cpu())
        del values, loss_dgpo, trust_loss, loss
    cosines = _finite_cosines(vectors)
    local = {
        "mean_rank_local_cosine": float(np.mean(cosines)) if cosines else float("nan"),
        "min_rank_local_cosine": float(np.min(cosines)) if cosines else float("nan"),
    }
    gathered = interface._all_gather_object(local)
    if rank != 0:
        return {}
    return {
        "mean_rank_local_cosine": float(np.nanmean([
            row["mean_rank_local_cosine"] for row in gathered
        ])),
        "min_rank_local_cosine": float(np.nanmin([
            row["min_rank_local_cosine"] for row in gathered
        ])),
        "events_per_worker": float(events),
    }


def _production_gradient(
    *,
    cfg: Mapping[str, Any],
    policy: torch.nn.Module,
    reference: torch.nn.Module,
    batch: Mapping[str, Any],
    candidates: Tensor,
    advantages: Tensor,
    realization: int,
    rank: int,
    world: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, dict[str, Any]]:
    """Reproduce one pre-AdamW production gradient and reduce it once."""

    from RL.DGPO_neutrino.dgpo_trainer import policy_evaluation_step

    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    local_events = int(cfg["events_per_worker"])
    if int(advantages.shape[1]) != local_events:
        raise ValueError("local advantage batch does not match events_per_worker")
    policy.zero_grad(set_to_none=True)
    gate_values: list[Tensor] = []
    loss_values: list[float] = []
    dgpo_loss_values: list[float] = []
    trust_loss_values: list[float] = []
    microbatch = int(cfg["event_microbatch_size"])
    timesteps = int(cfg["gradient_timesteps"])
    for start in range(0, local_events, microbatch):
        stop = start + microbatch
        chunk_batch = _select_batch(batch, start, stop)
        chunk_candidates = _select_candidates(candidates, start, stop)
        chunk_advantage = advantages[:, start:stop]
        chunk_weight = float(stop - start) / float(local_events)
        for timestep in range(timesteps):
            torch.manual_seed(
                int(cfg["gradient_seed"])
                + realization * 10_000_019
                + rank * 100_003
                + (start // microbatch) * 10_007
                + timestep * 1_009
            )
            values = policy_evaluation_step(
                policy,
                reference,
                chunk_batch,
                chunk_candidates,
                K=int(cfg["K"]),
                shared_noise=True,
                device=device,
                dtype=dtype,
                t_min=float(cfg["policy_eval_t_min"]),
                t_max=float(cfg["policy_eval_t_max"]),
                num_timesteps=1,
            )
            L_cur, L_ref = values[:2]
            loss_dgpo, _ = build_dgpo_loss(
                L_cur,
                L_ref,
                chunk_advantage,
                beta_dgpo=float(cfg["beta"]),
                K=int(cfg["K"]),
            )
            trust_loss, _ = build_reference_trust_loss(
                values[3],
                values[4],
                values[5],
                L_ref_2d=L_ref,
                objective=str(
                    cfg["production_contract"]["reference_trust_objective"]
                ),
            )
            loss = loss_dgpo + (
                float(cfg["production_contract"]["reference_trust_coefficient"])
                * trust_loss
            )
            weighted_loss = loss * chunk_weight / float(timesteps)
            weighted_loss.backward()
            with torch.no_grad():
                delta = L_cur.detach() - L_ref.detach()
                statistic = (
                    float(cfg["beta"]) / float(cfg["K"])
                ) * (chunk_advantage * delta).sum(dim=0)
                gate_values.append(torch.sigmoid(statistic).cpu())
                loss_values.append(float(loss.detach().cpu()))
                dgpo_loss_values.append(float(loss_dgpo.detach().cpu()))
                trust_loss_values.append(float(trust_loss.detach().cpu()))
            del values, L_cur, L_ref, loss_dgpo, trust_loss, loss, weighted_loss

    local_flat = _flat_parameter_gradient(parameters)
    if world > 1:
        torch.distributed.all_reduce(local_flat, op=torch.distributed.ReduceOp.SUM)
        local_flat.div_(float(world))
    gradient = local_flat.cpu()
    local_gates = torch.cat(gate_values)
    gathered_gates = interface._all_gather_object(local_gates)
    gathered_losses = interface._all_gather_object(loss_values)
    gathered_dgpo_losses = interface._all_gather_object(dgpo_loss_values)
    gathered_trust_losses = interface._all_gather_object(trust_loss_values)
    time_report = _time_stratum_metrics(
        cfg=cfg,
        policy=policy,
        reference=reference,
        batch=batch,
        candidates=candidates,
        advantages=advantages,
        parameters=parameters,
        realization=realization,
        rank=rank,
        device=device,
        dtype=dtype,
    )
    report: dict[str, Any] = {}
    if rank == 0:
        report = {
            "gradient_norm": float(gradient.double().norm()),
            "event_gate": _tensor_summary(torch.cat(gathered_gates)),
            "loss_mean_over_global_substeps": float(np.mean([
                value for shard in gathered_losses for value in shard
            ])),
            "dgpo_loss_mean_over_global_substeps": float(np.mean([
                value for shard in gathered_dgpo_losses for value in shard
            ])),
            "reference_trust_loss_mean_over_global_substeps": float(np.mean([
                value for shard in gathered_trust_losses for value in shard
            ])),
            "time_strata": time_report,
        }
    policy.zero_grad(set_to_none=True)
    return gradient, report


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
        raise ValueError(f"expected {cfg['workers']} Ray workers, found {world}")
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg["runtime_path"])
    torch.set_float32_matmul_precision(
        str(global_config.dgpo.get("float32_matmul_precision", "medium"))
    )
    root = Path(cfg["output_dir"])
    source_root = Path(cfg["source_dir"])
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
                if key not in {"source_manifest", "source_stats"}
            }),
        )
        wandb_run.summary.update({
            "phase": "load_immutable_h4_artifacts",
            "classifier_fits": 0,
            "policy_updates": 0,
        })
        wandb_run.log({
            "gradient_repro/objective_contract/workers": float(cfg["workers"]),
            "gradient_repro/objective_contract/events_per_worker": float(
                cfg["events_per_worker"]
            ),
            "gradient_repro/objective_contract/events_per_gradient": float(
                cfg["total_events_per_gradient"]
            ),
            "gradient_repro/objective_contract/K": float(cfg["K"]),
            "gradient_repro/objective_contract/gradient_timesteps": float(
                cfg["gradient_timesteps"]
            ),
            "gradient_repro/objective_contract/beta": float(cfg["beta"]),
            "gradient_repro/objective_contract/reference_trust_coefficient": float(
                cfg["production_contract"]["reference_trust_coefficient"]
            ),
            "gradient_repro/objective_contract/policy_updates": 0.0,
            "gradient_repro/objective_contract/classifier_fits": 0.0,
        })

    pool_payload = _load(source_root / "fixed_k1_pool.pt")
    fixed_panel = _load(source_root / "fixed_k8_panel.pt")
    packing_spec = EventPackingSpec.from_dict(pool_payload["packing_spec"])
    if list(fixed_panel["member_order"]) != cfg["member_order"]:
        raise ValueError("fixed K=8 panel member order differs from source report")
    del fixed_panel
    final_indices = pool_payload["partitions"]["final_audit"].long()
    gradient_indices, probe_indices = select_disjoint_event_indices(
        final_indices, cfg
    )

    checkpoint = replay.read_checkpoint(cfg["policy_checkpoint"])
    bundle = load_evenet_model_for_dgpo(
        config=global_config,
        device=device,
        checkpoint_path=cfg["policy_checkpoint"],
    )
    policy = bundle.model
    interface.verify_policy_loaded(policy, checkpoint["state_dict"])
    anchor_model_state = clone_state(policy.state_dict())
    assert_dgpo_neutrino_policy_deterministic(policy)
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

    recalibration = dict(global_config.dgpo.adaptive_omnifold.recalibration)
    builder_keys = set(inspect.signature(EvenetAdapterModelBuilder).parameters)
    builder = EvenetAdapterModelBuilder(
        config=global_config,
        normalization_dict=load_normalization_dict(global_config),
        checkpoint_path=global_config.reward_config.omnifold.backbone_checkpoint,
        device=device,
        **{key: value for key, value in recalibration.items() if key in builder_keys},
    )
    member_states = [_load(source_root / name) for name in MEMBER_FILES]
    sampler = DDIMSampler(device=device)
    gradients: list[Tensor] = []
    realization_reports: list[dict[str, Any]] = []
    policy.train()
    parameters = tuple(parameter for parameter in policy.parameters() if parameter.requires_grad)
    layout, blocks = parameter_layout(policy.named_parameters())

    for realization in range(int(cfg["realizations"])):
        if rank == 0:
            print(
                f"[gradient-repro] starting g{realization + 1}/4: "
                f"{cfg['total_events_per_gradient']} events, K={cfg['K']}, "
                f"timesteps={cfg['gradient_timesteps']}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "gradient_repro/stage": realization,
                    f"gradient_repro/g{realization + 1}/started": 1.0,
                })
                wandb_run.summary["phase"] = f"gradient_{realization + 1}_running"
        policy.load_state_dict(anchor_model_state, strict=True)
        policy.zero_grad(set_to_none=True)
        policy.train()
        start = realization * int(cfg["events_per_worker"]) * world
        stop = start + int(cfg["events_per_worker"]) * world
        global_chunk = gradient_indices[start:stop]
        local_indices = global_chunk[rank::world]
        if len(local_indices) != int(cfg["events_per_worker"]):
            raise RuntimeError("event sharding did not produce a production local batch")
        packed = pool_payload["packed_event"].index_select(0, local_indices)
        truth = pool_payload["truth"].index_select(0, local_indices)
        noise_mask = pool_payload["policy_noise_mask"].index_select(0, local_indices)
        batch = unpack_event_inputs(packed, packing_spec)
        batch["x_invisible"] = truth.reshape(-1, 2, 2)
        batch["x_invisible_mask"] = noise_mask
        batch = batch_to_device(batch, device)

        torch.manual_seed(
            int(cfg["candidate_seed"])
            + realization * 10_000_019
            + rank * 100_003
        )
        with torch.no_grad():
            candidates = generate_neutrino_candidates(
                policy,
                batch,
                sampler,
                K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
                device=device,
                parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
            )
        local_nonfinite_fraction = float(
            (~torch.isfinite(candidates)).float().mean().cpu()
        )
        nonfinite_fractions = interface._all_gather_object(local_nonfinite_fraction)
        candidates = torch.nan_to_num(candidates)
        sample_bk4 = candidates.permute(1, 0, 2, 3).reshape(
            len(local_indices), int(cfg["K"]), 4
        ).cpu()
        logits = []
        for member_index, state in enumerate(member_states):
            logits.append(interface._score_local_state(
                builder,
                packing_spec,
                state,
                f"gradient_r{realization + 1}_member{member_index + 1}",
                packed,
                sample_bk4,
                int(cfg["score_batch_size"]),
                device,
            ).reshape(len(local_indices), int(cfg["K"])).T)
        ensemble_logits = torch.stack(logits).mean(0).to(device)
        advantage = reward_advantage_arms(
            ensemble_logits,
            temperature=float(cfg["temperature"]),
            raw_tempering=0.75,
        )["calibrated_loo"]
        gathered_advantage = interface._all_gather_object(advantage.cpu())

        gradient, row = _production_gradient(
            cfg=cfg,
            policy=policy,
            reference=reference,
            batch=batch,
            candidates=candidates,
            advantages=advantage,
            realization=realization,
            rank=rank,
            world=world,
            device=device,
            dtype=next(policy.parameters()).dtype,
        )
        gradients.append(gradient)
        if rank == 0:
            complete_advantage = torch.cat(gathered_advantage, dim=1)
            row.update({
                "name": f"g{realization + 1}",
                "realization": realization + 1,
                "events": int(cfg["total_events_per_gradient"]),
                "K": int(cfg["K"]),
                "gradient_timesteps": int(cfg["gradient_timesteps"]),
                "candidate_nonfinite_fraction": float(np.mean(nonfinite_fractions)),
                "candidate_nonfinite_fraction_max_rank": float(
                    np.max(nonfinite_fractions)
                ),
                "advantage": interface.advantage_metrics(complete_advantage),
                "block_norms": {
                    name: _sliced_norm(gradient, ranges)
                    for name, ranges in blocks.items()
                },
            })
            realization_reports.append(row)
            print(
                f"[gradient-repro] g{realization + 1}/4 complete: "
                f"norm={row['gradient_norm']:.6g} "
                f"gate_mean={row['event_gate']['mean']:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                metrics = {
                    "gradient_repro/stage": realization,
                    f"gradient_repro/g{realization + 1}/complete": 1.0,
                    f"gradient_repro/g{realization + 1}/norm": row["gradient_norm"],
                    f"gradient_repro/g{realization + 1}/event_gate_mean": row["event_gate"]["mean"],
                    f"gradient_repro/g{realization + 1}/event_gate_p05": row["event_gate"]["p05"],
                    f"gradient_repro/g{realization + 1}/event_gate_p95": row["event_gate"]["p95"],
                    f"gradient_repro/g{realization + 1}/reference_trust_loss_mean": row["reference_trust_loss_mean_over_global_substeps"],
                    f"gradient_repro/g{realization + 1}/candidate_nonfinite_fraction": row["candidate_nonfinite_fraction"],
                    f"gradient_repro/g{realization + 1}/candidate_nonfinite_fraction_max_rank": row["candidate_nonfinite_fraction_max_rank"],
                    f"gradient_repro/g{realization + 1}/time_strata_mean_local_cosine": row["time_strata"].get("mean_rank_local_cosine"),
                }
                metrics.update({
                    f"gradient_repro/g{realization + 1}/block/{name}/norm": value
                    for name, value in row["block_norms"].items()
                })
                wandb_run.log(metrics)
                wandb_run.summary["phase"] = f"gradient_{realization + 1}_complete"
        del batch, candidates, sample_bk4, logits, ensemble_logits, advantage
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reproducibility = summarize_reproducibility(gradients, blocks)
    gbar = reproducibility.pop("gbar")
    status = production_variance_status(
        reproducibility["mean_pairwise_cosine"],
        reproducibility["min_pairwise_cosine"],
    )
    if rank == 0 and wandb_run is not None:
        metrics = {
            "gradient_repro/primary/mean_pairwise_cosine": reproducibility["mean_pairwise_cosine"],
            "gradient_repro/primary/min_pairwise_cosine": reproducibility["min_pairwise_cosine"],
            "gradient_repro/primary/half_mean_cosine": reproducibility["half_mean_cosine"],
            "gradient_repro/gbar/norm": reproducibility["gbar_norm"],
        }
        metrics.update({
            f"gradient_repro/pairwise/{name}/cosine": value
            for name, value in reproducibility["pairwise"].items()
        })
        metrics.update({
            f"gradient_repro/g{index + 1}/cosine_to_gbar": value
            for index, value in enumerate(reproducibility["cosine_to_gbar"])
        })
        for block, row in reproducibility["blocks"].items():
            metrics[
                f"gradient_repro/primary/block/{block}/mean_pairwise_cosine"
            ] = row["mean_pairwise_cosine"]
            metrics[
                f"gradient_repro/primary/block/{block}/min_pairwise_cosine"
            ] = row["min_pairwise_cosine"]
            metrics[f"gradient_repro/gbar/block/{block}/norm"] = row["gbar_norm"]
        wandb_run.log(metrics)
        wandb_run.summary["phase"] = "signed_fixed_judge_probes"

    # The probe events are disjoint from all four production-gradient batches.
    local_probe_indices = probe_indices[rank::world]
    probe_packed = pool_payload["packed_event"].index_select(0, local_probe_indices)
    probe_truth = pool_payload["truth"].index_select(0, local_probe_indices)
    probe_mask = pool_payload["policy_noise_mask"].index_select(0, local_probe_indices)
    probe_batch = unpack_event_inputs(probe_packed, packing_spec)
    probe_batch["x_invisible"] = probe_truth.reshape(-1, 2, 2)
    probe_batch["x_invisible_mask"] = probe_mask
    probe_batch = batch_to_device(probe_batch, device)

    policy.load_state_dict(anchor_model_state, strict=True)
    policy.zero_grad(set_to_none=True)
    builder.restore_pretrained_body()
    judge = builder.make_classifier(packing_spec, "gradient_repro_judge", reset=True).to(device)
    judge.load_state_dict(_load(source_root / "independent_h4_judge.pt"), strict=True)
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    truth_judge_local = interface._score_local_on_device(
        judge, probe_packed, probe_truth, int(cfg["score_batch_size"])
    ).cpu()
    anchor = tuple(parameter.detach().cpu().clone() for parameter in parameters)

    def restore_anchor() -> None:
        with torch.no_grad():
            for parameter, base in zip(parameters, anchor, strict=True):
                parameter.copy_(base.to(parameter.device, parameter.dtype))

    def probe_rollout() -> Tensor:
        torch.manual_seed(int(cfg["probe_rollout_seed"]) + rank)
        policy.eval()
        with torch.no_grad():
            result = generate_neutrino_candidates(
                policy,
                probe_batch,
                sampler,
                K=int(cfg["K"]),
                num_ddim_steps=int(global_config.dgpo.num_ddim_steps),
                device=device,
                parallel_chains=int(global_config.dgpo.get("rollout_parallel_chains", 1)),
            )
        policy.train()
        return result

    def evaluate_probe(label: str, direction: Tensor | None, sign: int) -> dict[str, Any]:
        restore_anchor()
        scale = 0.0
        if direction is not None:
            scale = interface._assign_direction(
                parameters,
                anchor,
                direction,
                sign=sign,
                epsilon_rms=float(cfg["probe_epsilon_rms"]),
            )
        candidates = probe_rollout()
        local_nonfinite_fraction = float(
            (~torch.isfinite(candidates)).float().mean().cpu()
        )
        nonfinite_fractions = interface._all_gather_object(local_nonfinite_fraction)
        candidates = torch.nan_to_num(candidates)
        sample = candidates.permute(1, 0, 2, 3).reshape(
            len(local_probe_indices), int(cfg["K"]), 4
        ).cpu()
        generated_logits = interface._score_local_on_device(
            judge, probe_packed, sample, int(cfg["score_batch_size"])
        ).cpu()
        gathered_truth = interface._all_gather_object(truth_judge_local)
        gathered_generated = interface._all_gather_object(generated_logits)
        if rank != 0:
            return {}
        metrics = interface._classification_metrics(
            torch.cat(gathered_truth), torch.cat(gathered_generated).reshape(-1)
        )
        return {
            "label": label,
            "direction_sign": sign,
            "epsilon_rms": 0.0 if direction is None else float(cfg["probe_epsilon_rms"]),
            "flat_gradient_scale": scale,
            "candidate_nonfinite_fraction": float(np.mean(nonfinite_fractions)),
            "candidate_nonfinite_fraction_max_rank": float(
                np.max(nonfinite_fractions)
            ),
            "judge": metrics,
            "judge_auc_gap": metrics["auc_gap"],
        }

    zero = evaluate_probe("zero", None, 0)
    probes: dict[str, Any] = {}
    directions = {f"g{index + 1}": value for index, value in enumerate(gradients)}
    directions["gbar"] = gbar
    for probe_index, (name, direction) in enumerate(directions.items()):
        minus = evaluate_probe(f"{name}/minus", direction, -1)
        plus = evaluate_probe(f"{name}/plus", direction, 1)
        if rank == 0:
            row = {
                "minus": minus,
                "zero": zero,
                "plus": plus,
                "plus_gap_delta": plus["judge_auc_gap"] - zero["judge_auc_gap"],
                "minus_gap_delta": minus["judge_auc_gap"] - zero["judge_auc_gap"],
                "plus_beats_minus": plus["judge_auc_gap"] < minus["judge_auc_gap"],
            }
            probes[name] = row
            print(
                f"[gradient-repro] {name} probe: "
                f"minus={minus['judge_auc_gap']:.6f} "
                f"zero={zero['judge_auc_gap']:.6f} "
                f"plus={plus['judge_auc_gap']:.6f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    "gradient_repro/probe/index": probe_index,
                    f"gradient_repro/probe/{name}/minus_judge_auc_gap": minus["judge_auc_gap"],
                    f"gradient_repro/probe/{name}/zero_judge_auc_gap": zero["judge_auc_gap"],
                    f"gradient_repro/probe/{name}/plus_judge_auc_gap": plus["judge_auc_gap"],
                    f"gradient_repro/probe/{name}/plus_gap_delta": row["plus_gap_delta"],
                    f"gradient_repro/probe/{name}/plus_beats_minus": float(row["plus_beats_minus"]),
                    f"gradient_repro/probe/{name}/plus_candidate_nonfinite_fraction": plus["candidate_nonfinite_fraction"],
                })
    restore_anchor()

    if rank == 0:
        interface.verify_policy_loaded(policy, anchor_model_state)
        finding = diagnose(reproducibility, probes)
        report = {
            "schema": "c4a91e07-h4-production-gradient-reproducibility-v1",
            "source_schema": cfg["expected_source_schema"],
            "source_dir": cfg["source_dir"],
            "source_wandb_run": cfg["source_manifest"].get("source_wandb_run"),
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "policy_updates": 0,
            "classifier_fits": 0,
            "optimizer_state_resumed": False,
            "objective_changed": False,
            "production_contract": cfg["production_contract"],
            "events_per_gradient": int(cfg["total_events_per_gradient"]),
            "realizations": realization_reports,
            "reproducibility": reproducibility,
            "production_variance_status": status,
            "probe_events": int(cfg["probe_events"]),
            "probe_epsilon_rms": float(cfg["probe_epsilon_rms"]),
            "probe_events_disjoint_from_gradient_events": True,
            "common_probe_events_and_rollout_noise": True,
            "signed_probes": probes,
            "diagnosis": finding,
            "parameter_layout": layout,
            "scope": (
                "Read-only audit of the exact existing DGPO objective. Four complete "
                "pre-AdamW production gradients are compared; no reward transform, "
                "optimizer step, classifier fit, or cold audit is performed."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_json(root / "report.json", interface._jsonable(report))
        direction_path = root / "gradient_directions_fp16.pt"
        if cfg["save_direction_vectors"]:
            replay._exclusive_torch_save(direction_path, {
                "schema": "c4a91e07-h4-production-gradient-directions-v1",
                "policy_global_step": int(cfg["expected_policy_step"]),
                "parameter_layout": layout,
                "directions": {
                    **{f"g{index + 1}": value.half() for index, value in enumerate(gradients)},
                    "gbar": gbar.half(),
                },
            })
        if wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                "c4a91e07-h4-production-gradient-reproducibility",
                type="diagnostic",
            )
            artifact.add_file(str(root / "manifest.json"))
            artifact.add_file(str(root / "report.json"))
            if cfg["upload_direction_vectors"]:
                artifact.add_file(str(direction_path))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete",
                "classifier_fits": 0,
                "policy_updates": 0,
                "gradient_repro/primary/mean_pairwise_cosine": reproducibility["mean_pairwise_cosine"],
                "gradient_repro/primary/min_pairwise_cosine": reproducibility["min_pairwise_cosine"],
                "gradient_repro/primary/half_mean_cosine": reproducibility["half_mean_cosine"],
                "gradient_repro/primary/gbar_fixed_judge_gap_delta": probes["gbar"]["plus_gap_delta"],
                "gradient_repro/primary/production_variance_status": status,
                "gradient_repro/diagnosis": finding,
            })
            wandb_run.finish()
        print(f"Production-gradient report: {root / 'report.json'}", flush=True)
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
        "Verified H4 artifacts and exact production objective: "
        f"4 gradients x {cfg['total_events_per_gradient']} events x K={cfg['K']} "
        f"x {cfg['gradient_timesteps']} timesteps; zero optimizer steps.",
        flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"].get("required"):
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
        offline = os.environ.get("WANDB_MODE", "").lower() in {
            "offline", "disabled", "dryrun",
        }
        if disabled or offline:
            raise RuntimeError(
                "this experiment requires live W&B; WANDB_DISABLED/WANDB_MODE "
                "currently disables online logging"
            )
    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError(
            "RAY_ADDRESS is unset; start/source the 16-GPU Ray cluster before launch"
        )
    root = Path(cfg["output_dir"])
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
    root.mkdir(parents=True, exist_ok=False)
    replay._exclusive_json(root / "manifest.json", interface._jsonable(cfg))
    TorchTrainer(
        train_loop_per_worker=_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]),
            use_gpu=True,
            resources_per_worker={"CPU": int(cfg["cpus_per_worker"]), "GPU": 1},
        ),
        run_config=RunConfig(
            name="c4a91e07-h4-production-gradient-reproducibility-v1",
            storage_path=str(root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
