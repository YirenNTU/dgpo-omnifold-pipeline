#!/usr/bin/env python3
"""Fixed-policy counterfactual test of the classifier-to-candidate direction.

This diagnostic reuses the immutable K=8 panel and H4 classifiers produced by
``diagnose_reward_interface.py``.  It never regenerates candidates, fits a
classifier, or updates the diffusion policy.  For each conditioning event it
self-normalizes ``exp(alpha * classifier_logit)`` across the same eight
candidates, then evaluates the resulting counterfactual distribution with an
independent H4 judge and physics metrics.  The installed legacy reward from the
c4a91e07 checkpoint is scored on the same panel as a matched control.
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
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))

import diagnose_raw_monitor_replay as replay
import diagnose_reward_interface as interface


REQUIRED_SOURCE_ARTIFACTS = (
    "runtime.yaml",
    "manifest.json",
    "report.json",
    "fixed_k1_pool.pt",
    "fixed_k8_panel.pt",
    "independent_h4_judge.pt",
)
TARGET_COMPONENTS = (
    "tau_a_delta_theta",
    "tau_a_delta_phi",
    "tau_b_delta_theta",
    "tau_b_delta_phi",
)
CORRELATION_FEATURES = (
    "tau_a_delta_theta",
    "sin_tau_a_delta_phi",
    "cos_tau_a_delta_phi",
    "tau_b_delta_theta",
    "sin_tau_b_delta_phi",
    "cos_tau_b_delta_phi",
)
PRODUCTION_EXPONENT = 0.75
RESPONSE_METRIC_SCOPE = "four_delta_coordinate_proxy_not_root_roounfold"


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
        "source_dir", "output_dir", "expected_source_schema",
        "expected_policy_step", "workers", "cpus_per_worker", "K",
        "score_batch_size", "tilt_exponents", "physics_bins", "response_bins",
        "response_bootstrap_replicates", "response_bootstrap_seed",
        "primary", "legacy_control", "wandb",
    }
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"missing candidate-reweighting settings: {missing}")
    for key in (
        "workers", "cpus_per_worker", "K", "score_batch_size", "physics_bins",
        "response_bins", "response_bootstrap_replicates",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["K"] < 2:
        raise ValueError("counterfactual candidate reweighting requires K >= 2")
    exponents = [float(value) for value in cfg["tilt_exponents"]]
    if (
        not exponents
        or any(not math.isfinite(value) or value <= 0.0 for value in exponents)
        or exponents != sorted(set(exponents))
    ):
        raise ValueError("tilt_exponents must be finite, positive, unique, and increasing")
    if not any(math.isclose(value, PRODUCTION_EXPONENT, abs_tol=1.0e-12) for value in exponents):
        raise ValueError("tilt_exponents must include the production exponent 0.75")
    cfg["tilt_exponents"] = exponents
    primary = dict(cfg["primary"] or {})
    expected_primary = {
        "source": "h4_ensemble",
        "exponent": PRODUCTION_EXPONENT,
        "require_independent_judge_improvement": True,
        "require_target_jsd_mean_improvement": True,
        "minimum_target_components_improved": 3,
    }
    if primary != expected_primary:
        raise ValueError(
            "primary must pin the reviewed H4 ensemble alpha=0.75 contract: "
            f"{expected_primary}"
        )
    cfg["primary"] = primary
    legacy = dict(cfg["legacy_control"] or {})
    legacy.setdefault("enabled", True)
    if type(legacy["enabled"]) is not bool:
        raise ValueError("legacy_control.enabled must be boolean")
    if legacy["enabled"]:
        for key in ("base_config", "overlay_config"):
            if not legacy.get(key):
                raise ValueError(f"legacy_control.{key} is required")
    cfg["legacy_control"] = legacy
    wandb = dict(cfg["wandb"] or {})
    wandb.setdefault("enabled", False)
    wandb.setdefault("required", False)
    if type(wandb["enabled"]) is not bool or type(wandb["required"]) is not bool:
        raise ValueError("wandb enabled/required flags must be boolean")
    if wandb["required"] and not wandb["enabled"]:
        raise ValueError("wandb.required=true requires wandb.enabled=true")
    if wandb["enabled"] and not wandb.get("id"):
        raise ValueError("live diagnostic W&B logging requires a fixed run id")
    cfg["wandb"] = wandb
    return cfg


def prepare(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source artifacts and the immutable c4a91e07 reward stack."""

    from train_neutrino_backend import absolutize_default_paths, deep_update, read_yaml

    cfg = _validated_settings(settings)
    source = Path(cfg["source_dir"]).expanduser().resolve(strict=True)
    paths = {name: (source / name).resolve(strict=True) for name in REQUIRED_SOURCE_ARTIFACTS}
    report = json.loads(paths["report.json"].read_text())
    manifest = json.loads(paths["manifest.json"].read_text())
    if report.get("schema") != cfg["expected_source_schema"]:
        raise ValueError(
            f"source schema mismatch: expected {cfg['expected_source_schema']!r}, "
            f"found {report.get('schema')!r}"
        )
    if int(report.get("policy_global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("source policy step does not match the declared anchor")
    if not report.get("fixed_policy") or not report.get("policy_unchanged"):
        raise ValueError("source experiment did not certify an unchanged fixed policy")
    if int(report.get("policy_updates", -1)) != 0:
        raise ValueError("source experiment unexpectedly updated the policy")
    panel = _load(paths["fixed_k8_panel.pt"])
    if int(panel["candidates_kb22"].shape[0]) != int(cfg["K"]):
        raise ValueError("saved candidate panel K differs from the requested K")
    if int(panel["member_logits_mkb"].shape[0]) != 4:
        raise ValueError("the H4 control must contain its declared four members")
    if int(panel["member_logits_mkb"].shape[1]) != int(cfg["K"]):
        raise ValueError("saved H4 member logits disagree with candidate K")
    policy_checkpoint = Path(report["policy_checkpoint"]).resolve(strict=True)
    checkpoint = replay.read_checkpoint(policy_checkpoint)
    if int(checkpoint.get("global_step", -1)) != int(cfg["expected_policy_step"]):
        raise ValueError("live c4a91e07 checkpoint no longer matches the source report")
    legacy_stack = checkpoint.get("dgpo_omnifold_reward_stack")
    if cfg["legacy_control"]["enabled"]:
        if not isinstance(legacy_stack, Mapping) or not legacy_stack.get("reward"):
            raise ValueError("c4a91e07 checkpoint has no installed legacy reward stack")
        increments = legacy_stack["reward"].get("increments") or []
        if not increments:
            raise ValueError("installed legacy reward stack has no ratio increments")
        if any(
            bool((row.get("classifier_config") or {}).get("topology_fourier_embedding", False))
            for row in increments
        ):
            raise ValueError("legacy control unexpectedly contains a Fourier classifier")
    del checkpoint, panel

    legacy_runtime = None
    source_stats = {
        str(path): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in paths.values()
    }
    source_stats[str(policy_checkpoint)] = [
        policy_checkpoint.stat().st_size, policy_checkpoint.stat().st_mtime_ns
    ]
    if cfg["legacy_control"]["enabled"]:
        base = (ROOT / cfg["legacy_control"]["base_config"]).resolve(strict=True)
        overlay = (ROOT / cfg["legacy_control"]["overlay_config"]).resolve(strict=True)
        legacy_runtime = absolutize_default_paths(
            deep_update(read_yaml(base), read_yaml(overlay)), base.parent
        )
        source_stats[str(base)] = [base.stat().st_size, base.stat().st_mtime_ns]
        source_stats[str(overlay)] = [overlay.stat().st_size, overlay.stat().st_mtime_ns]
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("counterfactual output overlaps its source artifacts")
    temperature = float(report["classifier_calibration"]["primary"]["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("source H4 calibration temperature is invalid")
    cfg.update(
        source_dir=str(source), output_dir=str(output),
        source_report=report, source_manifest=manifest,
        source_stats=source_stats, policy_checkpoint=str(policy_checkpoint),
        h4_temperature=temperature, legacy_runtime=legacy_runtime,
        h4_runtime_path=str(paths["runtime.yaml"]),
    )
    return cfg


def verify_sources(cfg: Mapping[str, Any]) -> None:
    for path, expected in cfg["source_stats"].items():
        observed = [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        if observed != expected:
            raise RuntimeError(f"protected source changed during diagnostic: {path}")


def candidate_tilt_weights(logits_kb: Tensor, exponent: float) -> Tensor:
    """Per-event self-normalized importance weights with shape ``(B,K)``."""

    if logits_kb.ndim != 2 or int(logits_kb.shape[0]) < 2:
        raise ValueError("candidate logits must have shape (K,B), K>=2")
    if not bool(torch.isfinite(logits_kb).all()):
        raise ValueError("candidate logits contain NaN/Inf")
    if not math.isfinite(float(exponent)) or float(exponent) < 0.0:
        raise ValueError("tilt exponent must be finite and nonnegative")
    return torch.softmax(logits_kb.double() * float(exponent), dim=0).T.contiguous()


def winner_weights(logits_kb: Tensor) -> Tensor:
    result = torch.zeros(
        (int(logits_kb.shape[1]), int(logits_kb.shape[0])), dtype=torch.float64
    )
    result.scatter_(1, logits_kb.argmax(dim=0).reshape(-1, 1).cpu(), 1.0)
    return result


def weight_diagnostics(weights_bk: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights_bk, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] < 2:
        raise ValueError("candidate weights must have shape (B,K), K>=2")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("candidate weights contain invalid values")
    row_sum = weights.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("candidate weights must sum to one within every event")
    k = weights.shape[1]
    ess_fraction = 1.0 / np.maximum(np.sum(weights**2, axis=1) * k, 1.0e-300)
    positive = weights > 0.0
    entropy = -np.sum(np.where(positive, weights * np.log(np.maximum(weights, 1e-300)), 0.0), axis=1)
    return {
        "event_ess_fraction_mean": float(np.mean(ess_fraction)),
        "event_ess_fraction_p05": float(np.quantile(ess_fraction, 0.05)),
        "event_ess_fraction_p50": float(np.quantile(ess_fraction, 0.50)),
        "event_ess_fraction_p95": float(np.quantile(ess_fraction, 0.95)),
        "normalized_entropy_mean": float(np.mean(entropy) / math.log(k)),
        "winner_mass_mean": float(np.mean(np.max(weights, axis=1))),
        "winner_mass_p95": float(np.quantile(np.max(weights, axis=1), 0.95)),
    }


def _weighted_histogram_jsd(
    truth: np.ndarray, generated: np.ndarray, generated_weight: np.ndarray,
    edges: np.ndarray,
) -> float:
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    generated = np.asarray(generated, dtype=np.float64).reshape(-1)
    weight = np.asarray(generated_weight, dtype=np.float64).reshape(-1)
    if generated.shape != weight.shape:
        raise ValueError("generated values and weights differ in length")
    truth = truth[np.isfinite(truth)]
    keep = np.isfinite(generated) & np.isfinite(weight) & (weight >= 0.0)
    generated, weight = generated[keep], weight[keep]
    if not truth.size or not generated.size:
        return float("nan")
    p = np.histogram(truth, bins=edges)[0].astype(np.float64)
    q = np.histogram(generated, bins=edges, weights=weight)[0].astype(np.float64)
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        positive = left > 0.0
        return float(np.sum(left[positive] * np.log2(left[positive] / right[positive])))

    return float(math.sqrt(max(0.0, 0.5 * (kl(p, midpoint) + kl(q, midpoint)))))


def weighted_physics_metrics(
    observables: Mapping[str, tuple[np.ndarray, np.ndarray]],
    weights_bk: np.ndarray, *, bins: int,
) -> dict[str, Any]:
    from RL.DGPO_neutrino.diagnostics.ztautau_validation import _observable_edges

    weights = np.asarray(weights_bk, dtype=np.float64)
    output: dict[str, Any] = {"observables": {}}
    groups: dict[str, list[float]] = {"target": [], "reco": [], "topology": []}
    for name, (truth, candidates) in sorted(observables.items()):
        truth = np.asarray(truth, dtype=np.float64).reshape(-1)
        candidates = np.asarray(candidates, dtype=np.float64)
        if candidates.shape != weights.shape or len(truth) != len(weights):
            raise ValueError(f"observable {name} does not match candidate weights")
        edges = _observable_edges(name, (truth, candidates.reshape(-1)), int(bins))
        jsd = _weighted_histogram_jsd(
            truth, candidates.reshape(-1), weights.reshape(-1), edges
        )
        residual = candidates - truth[:, None]
        if name.endswith("_phi"):
            residual = np.arctan2(np.sin(residual), np.cos(residual))
        weighted_abs = np.sum(weights * np.abs(residual), axis=1)
        output["observables"][name] = {
            "jsd": jsd,
            "residual_abs_mean": float(np.mean(weighted_abs)),
            "residual_mean": float(np.mean(np.sum(weights * residual, axis=1))),
        }
        group = name.split("/", 1)[0]
        if group in groups and math.isfinite(jsd):
            groups[group].append(jsd)
    for group, values in groups.items():
        output[f"{group}_jsd_mean"] = float(np.mean(values)) if values else float("nan")
        output[f"{group}_jsd_geomean"] = (
            float(np.exp(np.mean(np.log(np.maximum(values, 1.0e-12)))))
            if values else float("nan")
        )
    all_values = [value for values in groups.values() for value in values]
    output["all_jsd_mean"] = float(np.mean(all_values)) if all_values else float("nan")
    return output


def _weighted_pearson(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    keep = np.isfinite(first) & np.isfinite(second) & np.isfinite(weight) & (weight >= 0.0)
    first, second, weight = first[keep], second[keep], weight[keep]
    total = weight.sum()
    if len(first) < 2 or total <= 0.0:
        return float("nan")
    mean_first, mean_second = np.sum(weight * first) / total, np.sum(weight * second) / total
    left, right = first - mean_first, second - mean_second
    denominator = math.sqrt(
        float(np.sum(weight * left**2) * np.sum(weight * right**2))
    )
    return float(np.sum(weight * left * right) / denominator) if denominator > 0.0 else float("nan")


def correlation_metrics(
    truth_b4: np.ndarray, candidates_bk4: np.ndarray, weights_bk: np.ndarray
) -> dict[str, Any]:
    truth = np.asarray(truth_b4, dtype=np.float64).reshape(-1, 4)
    candidates = np.asarray(candidates_bk4, dtype=np.float64)
    weights = np.asarray(weights_bk, dtype=np.float64)
    if candidates.shape != (len(truth), weights.shape[1], 4) or weights.shape[0] != len(truth):
        raise ValueError("target arrays and candidate weights disagree")
    truth_features = np.stack((
        truth[:, 0], np.sin(truth[:, 1]), np.cos(truth[:, 1]),
        truth[:, 2], np.sin(truth[:, 3]), np.cos(truth[:, 3]),
    ), axis=-1)
    candidate_features = np.stack((
        candidates[:, :, 0], np.sin(candidates[:, :, 1]), np.cos(candidates[:, :, 1]),
        candidates[:, :, 2], np.sin(candidates[:, :, 3]), np.cos(candidates[:, :, 3]),
    ), axis=-1)
    dimensions = len(CORRELATION_FEATURES)
    truth_matrix = np.eye(dimensions, dtype=np.float64)
    generated_matrix = np.eye(dimensions, dtype=np.float64)
    pair_errors: dict[str, float] = {}
    flat_weight = weights.reshape(-1)
    for left in range(dimensions):
        for right in range(left + 1, dimensions):
            truth_corr = _weighted_pearson(
                truth_features[:, left], truth_features[:, right], np.ones(len(truth))
            )
            generated_corr = _weighted_pearson(
                candidate_features[:, :, left].reshape(-1),
                candidate_features[:, :, right].reshape(-1), flat_weight,
            )
            truth_matrix[left, right] = truth_matrix[right, left] = truth_corr
            generated_matrix[left, right] = generated_matrix[right, left] = generated_corr
            pair_errors[f"{CORRELATION_FEATURES[left]}__{CORRELATION_FEATURES[right]}"] = abs(
                generated_corr - truth_corr
            )
    scale_errors: dict[str, float] = {}
    for component, name in enumerate(CORRELATION_FEATURES):
        truth_std = float(np.std(truth_features[:, component]))
        values = candidate_features[:, :, component].reshape(-1)
        total = float(flat_weight.sum())
        mean = float(np.sum(flat_weight * values) / total)
        generated_std = math.sqrt(float(np.sum(flat_weight * (values - mean) ** 2) / total))
        scale_errors[name] = abs(math.log(max(generated_std, 1e-12) / max(truth_std, 1e-12)))
    return {
        "truth_matrix": truth_matrix.tolist(),
        "generated_matrix": generated_matrix.tolist(),
        "pair_abs_errors": pair_errors,
        "mean_abs_pair_error": float(np.mean(list(pair_errors.values()))),
        "max_abs_pair_error": float(np.max(list(pair_errors.values()))),
        "component_abs_log_std_ratio": scale_errors,
        "mean_abs_log_std_ratio": float(np.mean(list(scale_errors.values()))),
    }


def weighted_response_matrix_metrics(
    truth_b4: np.ndarray, candidates_bk4: np.ndarray, weights_bk: np.ndarray,
    *, bins: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    truth = np.asarray(truth_b4, dtype=np.float64).reshape(-1, 4)
    candidates = np.asarray(candidates_bk4, dtype=np.float64)
    weights = np.asarray(weights_bk, dtype=np.float64)
    if candidates.shape != (len(truth), weights.shape[1], 4):
        raise ValueError("response candidates and weights disagree")
    report: dict[str, Any] = {}
    event_scores: dict[str, np.ndarray] = {}
    for component, label in enumerate(TARGET_COMPONENTS):
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
                edges = np.linspace(lo, hi if hi > lo else lo + 1e-6, bins + 1)
        truth_bin = np.clip(np.digitize(target, edges[1:-1]), 0, bins - 1)
        candidate_bin = np.clip(np.digitize(predicted, edges[1:-1]), 0, bins - 1)
        matrix = np.zeros((bins, bins), dtype=np.float64)
        offsets = candidate_bin - truth_bin[:, None]
        event_abs = np.sum(weights * np.abs(offsets), axis=1)
        for column in range(bins):
            selected = truth_bin == column
            if not np.any(selected):
                continue
            rows = candidate_bin[selected].reshape(-1)
            row_weights = weights[selected].reshape(-1)
            total = row_weights.sum()
            if total > 0.0:
                matrix[:, column] = np.bincount(
                    rows, weights=row_weights, minlength=bins
                ) / total
        valid_columns = matrix.sum(axis=0) > 0.0
        near = [
            matrix[max(0, col - 1):min(bins, col + 2), col].sum()
            for col in np.flatnonzero(valid_columns)
        ]
        per_bin_bias, per_bin_rms = [], []
        for column in np.flatnonzero(valid_columns):
            selected = truth_bin == column
            local_weights = weights[selected]
            local_offsets = offsets[selected]
            total = local_weights.sum()
            per_bin_bias.append(float(np.sum(local_weights * local_offsets) / total))
            per_bin_rms.append(float(np.sqrt(np.sum(local_weights * local_offsets**2) / total)))
        report[label] = {
            "truth_normalized_matrix": matrix.tolist(),
            "diagonal_fraction": float(np.mean(np.diag(matrix)[valid_columns])),
            "near_diagonal_fraction": float(np.mean(near)),
            "mean_abs_bin_offset": float(np.mean(event_abs)),
            "rms_bin_offset": float(np.sqrt(np.mean(np.sum(weights * offsets**2, axis=1)))),
            "equal_truth_bin_abs_bias": float(np.mean(np.abs(per_bin_bias))),
            "equal_truth_bin_resolution": float(np.mean(per_bin_rms)),
            "populated_truth_bins": int(np.sum(valid_columns)),
        }
        event_scores[label] = event_abs
    return report, event_scores


def evaluate_arm(
    *, label: str, weights_bk: np.ndarray, truth_judge: Tensor,
    candidate_judge_bk: Tensor, observables: Mapping[str, tuple[np.ndarray, np.ndarray]],
    truth_b4: np.ndarray, candidates_bk4: np.ndarray, physics_bins: int,
    response_bins: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _weighted_binary_score_metrics

    weights = np.asarray(weights_bk, dtype=np.float64)
    generated_weight = torch.as_tensor(weights.reshape(-1), dtype=torch.float64)
    bce, accuracy, auc = _weighted_binary_score_metrics(
        truth_judge.double().reshape(-1), candidate_judge_bk.double().reshape(-1),
        generated_weight,
    )
    physics = weighted_physics_metrics(observables, weights, bins=physics_bins)
    response, event_scores = weighted_response_matrix_metrics(
        truth_b4, candidates_bk4, weights, bins=response_bins
    )
    result = {
        "label": label,
        "weights": weight_diagnostics(weights),
        "judge": {
            "bce": bce, "balanced_accuracy": accuracy, "auc": auc,
            "auc_gap": abs(auc - 0.5),
        },
        "physics": physics,
        "correlation": correlation_metrics(truth_b4, candidates_bk4, weights),
        "response": response,
        "response_mean_abs_bin_offset": float(np.mean([
            row["mean_abs_bin_offset"] for row in response.values()
        ])),
    }
    return result, event_scores


def compare_to_uniform(
    baseline: Mapping[str, Any], arm: Mapping[str, Any], *,
    minimum_target_components_improved: int = 3,
) -> dict[str, Any]:
    judge_delta = float(arm["judge"]["auc_gap"]) - float(baseline["judge"]["auc_gap"])
    target_delta = float(arm["physics"]["target_jsd_mean"]) - float(
        baseline["physics"]["target_jsd_mean"]
    )
    topology_delta = float(arm["physics"]["topology_jsd_mean"]) - float(
        baseline["physics"]["topology_jsd_mean"]
    )
    correlation_delta = float(arm["correlation"]["mean_abs_pair_error"]) - float(
        baseline["correlation"]["mean_abs_pair_error"]
    )
    response_delta = float(arm["response_mean_abs_bin_offset"]) - float(
        baseline["response_mean_abs_bin_offset"]
    )
    target_changes = {
        name: float(arm["physics"]["observables"][f"target/{name}"]["jsd"])
        - float(baseline["physics"]["observables"][f"target/{name}"]["jsd"])
        for name in TARGET_COMPONENTS
    }
    target_improved = sum(delta < 0.0 for delta in target_changes.values())
    if judge_delta >= 0.0:
        finding = "classifier_candidate_direction_not_supported"
    elif (
        target_delta < 0.0
        and target_improved >= int(minimum_target_components_improved)
    ):
        finding = "candidate_direction_useful_dgpo_transfer_is_next_suspect"
    elif target_delta >= 0.0:
        finding = "classifier_proxy_improves_without_target_closure"
    else:
        finding = "mixed_candidate_direction"
    return {
        "finding": finding,
        "judge_auc_gap_delta": judge_delta,
        "target_jsd_mean_delta": target_delta,
        "topology_jsd_mean_delta": topology_delta,
        "correlation_mean_abs_pair_error_delta": correlation_delta,
        "response_mean_abs_bin_offset_delta": response_delta,
        "target_jsd_deltas": target_changes,
        "target_components_improved": int(target_improved),
        "target_components_total": len(TARGET_COMPONENTS),
    }


def _physics_observables(
    truth_b22: Tensor, candidates_kb22: Tensor, batch
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from RL.DGPO_neutrino.domains.ztautau import (
        reconstruct_tau_angles_from_deltas,
        tau_back_to_back_metrics,
        tau_calibration_magnitude_metrics,
    )

    truth = truth_b22.detach().double().clone()
    candidates = candidates_kb22.detach().double().clone()
    truth[..., 1] = torch.atan2(torch.sin(truth[..., 1]), torch.cos(truth[..., 1]))
    candidates[..., 1] = torch.atan2(torch.sin(candidates[..., 1]), torch.cos(candidates[..., 1]))
    truth_a, truth_b = reconstruct_tau_angles_from_deltas(truth.unsqueeze(0), batch)
    cand_a, cand_b = reconstruct_tau_angles_from_deltas(candidates, batch)
    truth_topology = {
        **tau_back_to_back_metrics(truth.unsqueeze(0), batch),
        **tau_calibration_magnitude_metrics(truth.unsqueeze(0), batch),
    }
    cand_topology = {
        **tau_back_to_back_metrics(candidates, batch),
        **tau_calibration_magnitude_metrics(candidates, batch),
    }

    def pair(truth_values: Tensor, candidate_values: Tensor) -> tuple[np.ndarray, np.ndarray]:
        return (
            truth_values.detach().cpu().numpy().reshape(-1),
            candidate_values.detach().cpu().numpy().T,
        )

    result = {
        "target/tau_a_delta_theta": pair(truth[:, 0, 0], candidates[:, :, 0, 0]),
        "target/tau_a_delta_phi": pair(truth[:, 0, 1], candidates[:, :, 0, 1]),
        "target/tau_b_delta_theta": pair(truth[:, 1, 0], candidates[:, :, 1, 0]),
        "target/tau_b_delta_phi": pair(truth[:, 1, 1], candidates[:, :, 1, 1]),
        "reco/tau_a_theta": pair(truth_a[0, :, 0], cand_a[:, :, 0]),
        "reco/tau_a_phi": pair(truth_a[0, :, 1], cand_a[:, :, 1]),
        "reco/tau_b_theta": pair(truth_b[0, :, 0], cand_b[:, :, 0]),
        "reco/tau_b_phi": pair(truth_b[0, :, 1], cand_b[:, :, 1]),
    }
    for name in truth_topology:
        result[f"topology/{name}"] = pair(truth_topology[name][0], cand_topology[name])
    return result


def _ordered_gather(positions: Tensor, values: Tensor) -> Tensor | None:
    rows = interface._all_gather_object({
        "positions": positions.detach().cpu(), "values": values.detach().cpu()
    })
    if interface._dist_context()[0] != 0:
        return None
    order = torch.argsort(torch.cat([row["positions"] for row in rows]))
    return torch.cat([row["values"] for row in rows], dim=0)[order]


def _build_legacy_reward(cfg: Mapping[str, Any], device: torch.device):
    from evenet.control.global_config import Config
    from RL.DGPO_neutrino.model_utils import load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import (
        build_uninstalled_ztautau_omnifold_reward,
    )

    legacy_path = Path(cfg["legacy_runtime_path"])
    legacy_config = Config()
    legacy_config.load_yaml(legacy_path)
    recal = legacy_config.dgpo.adaptive_omnifold.recalibration
    checkpoint = replay.read_checkpoint(cfg["policy_checkpoint"])
    stack = checkpoint["dgpo_omnifold_reward_stack"]
    reward = build_uninstalled_ztautau_omnifold_reward(
        backbone_checkpoint=legacy_config.reward_config.omnifold.backbone_checkpoint,
        training_config=legacy_config,
        normalization_dict=load_normalization_dict(legacy_config),
        device=device,
        classifier_config=dict(recal),
        candidate_consensus=stack.get("candidate_consensus"),
    )
    reward.load_stack_payload(stack, allow_source_bundle_migration=False)
    del checkpoint
    return reward


def _worker(cfg: Mapping[str, Any]) -> None:
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.dgpo_trainer import batch_to_device
    from RL.DGPO_neutrino.model_utils import load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EventPackingSpec, EvenetAdapterModelBuilder, unpack_event_inputs,
    )

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    if world != int(cfg["workers"]):
        raise ValueError(f"expected {cfg['workers']} Ray workers, found {world}")
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg["h4_runtime_path"])
    torch.set_float32_matmul_precision(
        str(global_config.dgpo.get("float32_matmul_precision", "medium"))
    )
    root = Path(cfg["output_dir"])
    wandb_run = None
    if rank == 0 and cfg["wandb"]["enabled"]:
        import wandb
        wandb_run = wandb.init(
            entity=cfg["wandb"].get("entity"), project=cfg["wandb"].get("project"),
            id=cfg["wandb"]["id"], resume=cfg["wandb"].get("resume", "allow"),
            name=cfg["wandb"].get("name"), tags=cfg["wandb"].get("tags"),
            job_type="diagnostic",
            config=interface._jsonable({
                key: value for key, value in cfg.items()
                if key not in {"source_report", "source_manifest", "source_stats", "legacy_runtime"}
            }),
        )
        wandb_run.summary.update({
            "phase": "score_independent_h4_judge", "policy_updates": 0,
            "classifier_fits": 0, "candidate_regenerations": 0,
        })
    source = Path(cfg["source_dir"])
    panel = _load(source / "fixed_k8_panel.pt")
    pool = _load(source / "fixed_k1_pool.pt")
    packing_spec = EventPackingSpec.from_dict(pool["packing_spec"])
    total = int(panel["packed_event"].shape[0])
    positions = torch.arange(total)[rank::world]
    packed = panel["packed_event"][rank::world]
    truth = panel["truth"][rank::world]
    noise_mask = panel["policy_noise_mask"][rank::world]
    candidates = panel["candidates_kb22"][:, rank::world].to(device)
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
        **{key: value for key, value in recalibration.items() if key in builder_keys},
    )
    builder.restore_pretrained_body()
    judge = builder.make_classifier(packing_spec, "candidate_reweighting_judge", reset=True).to(device)
    judge.load_state_dict(_load(source / "independent_h4_judge.pt"), strict=True)
    judge.eval()
    for parameter in judge.parameters():
        parameter.requires_grad_(False)
    sample_bk4 = candidates.permute(1, 0, 2, 3).reshape(len(truth), int(cfg["K"]), 4)
    truth_logits_local = interface._score_local_on_device(
        judge, packed, truth, int(cfg["score_batch_size"])
    )
    candidate_logits_local = interface._score_local_on_device(
        judge, packed, sample_bk4.cpu(), int(cfg["score_batch_size"])
    )
    truth_logits = _ordered_gather(positions, truth_logits_local)
    candidate_logits = _ordered_gather(positions, candidate_logits_local)
    del judge, builder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    legacy_logits = None
    if cfg["legacy_control"]["enabled"]:
        if wandb_run is not None:
            wandb_run.summary["phase"] = "score_installed_legacy_reward"
        legacy_reward = _build_legacy_reward(cfg, device)
        with torch.no_grad():
            legacy_local = legacy_reward.compute(candidates, batch).T.contiguous()
        legacy_logits = _ordered_gather(positions, legacy_local)
        del legacy_reward
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0:
        if wandb_run is not None:
            wandb_run.summary.update({
                "phase": "evaluate_fixed_candidate_panel", "policy_updates": 0,
                "classifier_fits": 0, "candidate_regenerations": 0,
            })

        full_truth = panel["truth"].reshape(-1, 2, 2)
        full_candidates = panel["candidates_kb22"]
        full_mask = panel["policy_noise_mask"]
        valid = (full_mask.reshape(len(full_mask), -1)[:, :2] > 0).all(dim=1)
        full_batch = unpack_event_inputs(panel["packed_event"], packing_spec)
        full_batch["x_invisible"] = full_truth
        full_batch["x_invisible_mask"] = full_mask
        observables_all = _physics_observables(full_truth, full_candidates, full_batch)
        observables = {
            name: (truth_values[valid.numpy()], candidate_values[valid.numpy()])
            for name, (truth_values, candidate_values) in observables_all.items()
        }
        truth_b4 = full_truth.reshape(-1, 4)[valid].double().numpy()
        candidates_bk4 = full_candidates.permute(1, 0, 2, 3).reshape(total, int(cfg["K"]), 4)[valid].double().numpy()
        truth_judge = truth_logits[valid]
        candidate_judge = candidate_logits[valid]
        h4_members = panel["member_logits_mkb"][:, :, valid]
        h4_ensemble = h4_members.mean(dim=0)

        baseline_weights = np.full((int(valid.sum()), int(cfg["K"])), 1.0 / int(cfg["K"]))
        baseline, baseline_events = evaluate_arm(
            label="uniform", weights_bk=baseline_weights,
            truth_judge=truth_judge, candidate_judge_bk=candidate_judge,
            observables=observables, truth_b4=truth_b4,
            candidates_bk4=candidates_bk4, physics_bins=int(cfg["physics_bins"]),
            response_bins=int(cfg["response_bins"]),
        )
        rows: list[dict[str, Any]] = []
        saved_weights: dict[str, Tensor] = {
            "valid_event_mask": valid,
            "uniform": torch.from_numpy(baseline_weights),
        }

        def add_arm(label: str, weights: Tensor, *, source_name: str, exponent: float | None) -> None:
            weights_np = weights.cpu().numpy()
            result, event_scores = evaluate_arm(
                label=label, weights_bk=weights_np,
                truth_judge=truth_judge, candidate_judge_bk=candidate_judge,
                observables=observables, truth_b4=truth_b4,
                candidates_bk4=candidates_bk4, physics_bins=int(cfg["physics_bins"]),
                response_bins=int(cfg["response_bins"]),
            )
            comparison = compare_to_uniform(
                baseline,
                result,
                minimum_target_components_improved=int(
                    cfg["primary"]["minimum_target_components_improved"]
                ),
            )
            bootstrap = {
                component: interface._paired_bootstrap_delta(
                    event_scores[component], baseline_events[component],
                    replicates=int(cfg["response_bootstrap_replicates"]),
                    seed=int(cfg["response_bootstrap_seed"]) + 100 * len(rows) + index,
                )
                for index, component in enumerate(TARGET_COMPONENTS)
            }
            result.update({
                "source": source_name, "exponent": exponent,
                "vs_uniform": comparison,
                "paired_response_bootstrap": bootstrap,
            })
            rows.append(result)
            saved_weights[label] = weights.cpu()
            print(
                f"[candidate-reweight] {label} "
                f"ESS/K={result['weights']['event_ess_fraction_mean']:.4f} "
                f"dAUCgap={comparison['judge_auc_gap_delta']:+.6g} "
                f"dTargetJSD={comparison['target_jsd_mean_delta']:+.6g} "
                f"dTopologyJSD={comparison['topology_jsd_mean_delta']:+.6g} "
                f"dCorr={comparison['correlation_mean_abs_pair_error_delta']:+.6g} "
                f"dResponse={comparison['response_mean_abs_bin_offset_delta']:+.6g}",
                flush=True,
            )
            if wandb_run is not None:
                metrics = {
                    "counterfactual/index": len(rows),
                    "counterfactual/arm": label,
                    "counterfactual/source": source_name,
                    "counterfactual/exponent": -1.0 if exponent is None else exponent,
                    "counterfactual/judge_auc_gap": result["judge"]["auc_gap"],
                    "counterfactual/target_jsd_mean": result["physics"]["target_jsd_mean"],
                    "counterfactual/topology_jsd_mean": result["physics"]["topology_jsd_mean"],
                    "counterfactual/correlation_error": result["correlation"]["mean_abs_pair_error"],
                    "counterfactual/response_offset": result["response_mean_abs_bin_offset"],
                    "counterfactual/event_ess_fraction": result["weights"]["event_ess_fraction_mean"],
                    "counterfactual/delta_judge_auc_gap": comparison["judge_auc_gap_delta"],
                    "counterfactual/delta_target_jsd_mean": comparison["target_jsd_mean_delta"],
                    "counterfactual/delta_topology_jsd_mean": comparison["topology_jsd_mean_delta"],
                    "counterfactual/delta_correlation_error": comparison["correlation_mean_abs_pair_error_delta"],
                    "counterfactual/delta_response_offset": comparison["response_mean_abs_bin_offset_delta"],
                }
                for name, values in result["physics"]["observables"].items():
                    metrics[f"counterfactual/jsd/{name}"] = values["jsd"]
                wandb_run.log(metrics, step=len(rows))

        if wandb_run is not None:
            wandb_run.log({
                "counterfactual/index": 0,
                "counterfactual/arm": "uniform",
                "counterfactual/source": "fixed_policy",
                "counterfactual/exponent": 0.0,
                "counterfactual/judge_auc_gap": baseline["judge"]["auc_gap"],
                "counterfactual/target_jsd_mean": baseline["physics"]["target_jsd_mean"],
                "counterfactual/topology_jsd_mean": baseline["physics"]["topology_jsd_mean"],
                "counterfactual/correlation_error": baseline["correlation"]["mean_abs_pair_error"],
                "counterfactual/response_offset": baseline["response_mean_abs_bin_offset"],
                "counterfactual/event_ess_fraction": 1.0,
            }, step=0)

        for exponent in cfg["tilt_exponents"]:
            add_arm(
                f"h4_ensemble_alpha_{exponent:g}",
                candidate_tilt_weights(h4_ensemble, exponent),
                source_name="h4_ensemble", exponent=exponent,
            )
        add_arm(
            "h4_ensemble_calibrated",
            candidate_tilt_weights(h4_ensemble, 1.0 / float(cfg["h4_temperature"])),
            source_name="h4_ensemble_calibrated", exponent=1.0 / float(cfg["h4_temperature"]),
        )
        add_arm(
            "h4_ensemble_winner",
            winner_weights(h4_ensemble), source_name="h4_ensemble", exponent=None,
        )
        for member in range(int(h4_members.shape[0])):
            add_arm(
                f"h4_member_{member + 1}_alpha_0.75",
                candidate_tilt_weights(h4_members[member], PRODUCTION_EXPONENT),
                source_name=f"h4_member_{member + 1}", exponent=PRODUCTION_EXPONENT,
            )
        if legacy_logits is not None:
            legacy_kb = legacy_logits[valid].T.contiguous()
            for multiplier in cfg["tilt_exponents"]:
                add_arm(
                    f"legacy_installed_scale_{multiplier:g}",
                    candidate_tilt_weights(legacy_kb, multiplier),
                    source_name="legacy_installed_stack", exponent=multiplier,
                )
            add_arm(
                "legacy_ensemble_winner", winner_weights(legacy_kb),
                source_name="legacy_installed_stack", exponent=None,
            )

        by_label = {row["label"]: row for row in rows}
        primary_label = (
            f"{cfg['primary']['source']}_alpha_{float(cfg['primary']['exponent']):g}"
        )
        h4_production = by_label[primary_label]
        # ``legacy_logits`` are the installed stack's final scores and already
        # include every saved iteration coefficient and tempering factor.
        legacy_production = by_label.get("legacy_installed_scale_1")
        h4_member_rows = [
            row for row in rows if row["label"].startswith("h4_member_")
        ]
        robustness = {
            "h4_members_improving_judge": sum(
                row["vs_uniform"]["judge_auc_gap_delta"] < 0.0 for row in h4_member_rows
            ),
            "h4_members_improving_target_jsd": sum(
                row["vs_uniform"]["target_jsd_mean_delta"] < 0.0 for row in h4_member_rows
            ),
            "h4_members_total": len(h4_member_rows),
        }
        report = {
            "schema": "c4a91e07-fixed-candidate-counterfactual-v1",
            "source_schema": cfg["expected_source_schema"],
            "source_dir": cfg["source_dir"],
            "source_wandb_run": cfg["source_report"].get("source_wandb_run"),
            "policy_checkpoint": cfg["policy_checkpoint"],
            "policy_global_step": int(cfg["expected_policy_step"]),
            "events": int(valid.sum()), "K": int(cfg["K"]),
            "policy_updates": 0, "candidate_regenerations": 0, "classifier_fits": 0,
            "fixed_policy": True, "fixed_candidate_panel": True,
            "counterfactual_definition": (
                "Within each event, weights are proportional to exp(alpha * classifier_logit). "
                "For raw H4 logits alpha=0.75 is the predeclared production-matched endpoint. "
                "The legacy stack scores already include saved tempering, so its matched "
                "endpoint is scale=1. alpha=0 is uniform."
            ),
            "response_metric_scope": RESPONSE_METRIC_SCOPE,
            "primary_contract": {
                **cfg["primary"],
                "arm": primary_label,
                "comparison": "uniform",
                "policy_updates": 0,
                "classifier_fits": 0,
                "candidate_regenerations": 0,
            },
            "baseline": baseline, "arms": rows,
            "h4_production_diagnosis": h4_production["vs_uniform"],
            "legacy_production_diagnosis": (
                None if legacy_production is None else legacy_production["vs_uniform"]
            ),
            "member_robustness": robustness,
            "interpretation": {
                "candidate_direction_useful_dgpo_transfer_is_next_suspect": (
                    "H4 reweighting improves the independent judge and at least three of four "
                    "target JSDs without any policy update; inspect advantage-to-gradient and "
                    "gradient-to-policy transfer next."
                ),
                "classifier_proxy_improves_without_target_closure": (
                    "The independent H4 judge follows the reward, but the target distribution "
                    "does not; the H4 representation/objective is a mismatched physics proxy."
                ),
                "classifier_candidate_direction_not_supported": (
                    "Even an independent H4 judge does not improve under the proposed ordering; "
                    "the conditional candidate direction is not reproducible."
                ),
                "mixed_candidate_direction": (
                    "The fixed-candidate evidence is mixed across target components; increase K "
                    "or audit event regions before changing DGPO optimization."
                ),
            },
            "scope": (
                "This identifies whether the reward ordering can improve a finite K=8 candidate "
                "mixture. It does not test whether a neural policy can represent that mixture or "
                "whether repeated DGPO updates remain stable. Response matrices cover the four "
                "generated tau-delta coordinates, not the downstream ROOT response pipeline."
            ),
        }
        verify_sources(cfg)
        replay._exclusive_torch_save(root / "counterfactual_weights.pt", saved_weights)
        replay._exclusive_json(root / "report.json", interface._jsonable(report))
        if wandb_run is not None:
            import wandb
            artifact = wandb.Artifact(
                "c4a91e07-fixed-candidate-counterfactual", type="diagnostic"
            )
            artifact.add_file(str(root / "report.json"))
            artifact.add_file(str(root / "counterfactual_weights.pt"))
            wandb_run.log_artifact(artifact)
            wandb_run.summary.update({
                "phase": "complete",
                "finding": report["h4_production_diagnosis"]["finding"],
                "primary_arm": primary_label,
                "response_metric_scope": RESPONSE_METRIC_SCOPE,
                "h4_production_delta_judge_auc_gap": report["h4_production_diagnosis"]["judge_auc_gap_delta"],
                "h4_production_delta_target_jsd": report["h4_production_diagnosis"]["target_jsd_mean_delta"],
                "h4_production_delta_topology_jsd": report["h4_production_diagnosis"]["topology_jsd_mean_delta"],
                "h4_production_target_components_improved": report[
                    "h4_production_diagnosis"
                ]["target_components_improved"],
                "h4_production_event_ess_fraction_mean": h4_production[
                    "weights"
                ]["event_ess_fraction_mean"],
                "h4_production_event_ess_fraction_p05": h4_production[
                    "weights"
                ]["event_ess_fraction_p05"],
                "h4_production_winner_mass_mean": h4_production[
                    "weights"
                ]["winner_mass_mean"],
                "h4_production_delta_correlation": report[
                    "h4_production_diagnosis"
                ]["correlation_mean_abs_pair_error_delta"],
                "h4_production_delta_response": report[
                    "h4_production_diagnosis"
                ]["response_mean_abs_bin_offset_delta"],
            })
            wandb_run.finish()
        print(f"Candidate counterfactual report: {root / 'report.json'}", flush=True)
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
        f"Verified fixed K={cfg['K']} panel at c4a91e07 step "
        f"{cfg['expected_policy_step']}; classifier fits=0, policy updates=0.",
        flush=True,
    )
    if args.check_only:
        return 0
    if cfg["wandb"]["required"]:
        disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
        offline = os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled", "dryrun"}
        if disabled or offline:
            raise RuntimeError("this experiment requires live W&B logging")
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        raise RuntimeError("RAY_ADDRESS is unset; source the active NERSC Ray cluster first")
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
            f"Ray cluster has {available_gpus:g} GPUs; diagnostic requires {cfg['workers']}"
        )
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    if cfg["legacy_runtime"] is not None:
        legacy_runtime_path = root / "legacy_runtime.yaml"
        with legacy_runtime_path.open("x") as handle:
            yaml.safe_dump(cfg["legacy_runtime"], handle, sort_keys=False)
        cfg["legacy_runtime_path"] = str(legacy_runtime_path)
    replay._exclusive_json(
        root / "manifest.json",
        interface._jsonable({
            key: value for key, value in cfg.items()
            if key not in {"source_report", "source_manifest", "legacy_runtime"}
        }),
    )
    TorchTrainer(
        train_loop_per_worker=_worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(
            num_workers=int(cfg["workers"]), use_gpu=True,
            resources_per_worker={"CPU": int(cfg["cpus_per_worker"]), "GPU": 1},
        ),
        run_config=RunConfig(
            name="c4a91e07-fixed-candidate-counterfactual-v1",
            storage_path=str(root / "ray_results"),
            failure_config=FailureConfig(max_failures=0),
        ),
    ).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
