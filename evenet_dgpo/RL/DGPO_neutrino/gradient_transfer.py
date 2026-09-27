"""Exact, read-only diagnostics for the component gradients of a DGPO update.

The main trajectory still applies the configured loss and native optimizer
step.  This module only summarizes gradients that were measured on that same
training graph and the parameter displacement produced by AdamW.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class GradientTransferTraceConfig:
    enabled: bool = False
    update_end_steps: tuple[int, ...] = (1, 2, 5, 10, 20)
    counterfactual_trust_coefficients: tuple[float, ...] = (
        0.0,
        0.1,
        0.2,
        0.5,
        1.0,
    )
    norm_floor: float = 1.0e-12


def resolve_gradient_transfer_trace_config(
    payload: Mapping[str, Any] | None,
) -> GradientTransferTraceConfig:
    values = dict(payload or {})
    allowed = set(GradientTransferTraceConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            f"unknown dgpo.gradient_transfer_trace option(s): {unknown}"
        )
    if "update_end_steps" in values:
        values["update_end_steps"] = tuple(values["update_end_steps"])
    if "counterfactual_trust_coefficients" in values:
        values["counterfactual_trust_coefficients"] = tuple(
            values["counterfactual_trust_coefficients"]
        )
    cfg = GradientTransferTraceConfig(**values)
    if type(cfg.enabled) is not bool:
        raise ValueError("gradient_transfer_trace.enabled must be boolean")
    if not cfg.update_end_steps:
        raise ValueError("gradient_transfer_trace.update_end_steps cannot be empty")
    if any(type(step) is not int or step < 1 for step in cfg.update_end_steps):
        raise ValueError(
            "gradient_transfer_trace.update_end_steps must contain positive integers"
        )
    if len(set(cfg.update_end_steps)) != len(cfg.update_end_steps):
        raise ValueError("gradient_transfer_trace.update_end_steps must be unique")
    if tuple(sorted(cfg.update_end_steps)) != cfg.update_end_steps:
        raise ValueError("gradient_transfer_trace.update_end_steps must be sorted")
    if not cfg.counterfactual_trust_coefficients:
        raise ValueError(
            "gradient_transfer_trace.counterfactual_trust_coefficients cannot be empty"
        )
    coefficients = tuple(float(value) for value in cfg.counterfactual_trust_coefficients)
    if any(not math.isfinite(value) or value < 0.0 for value in coefficients):
        raise ValueError(
            "gradient_transfer_trace counterfactual coefficients must be finite and nonnegative"
        )
    if len(set(coefficients)) != len(coefficients):
        raise ValueError(
            "gradient_transfer_trace counterfactual coefficients must be unique"
        )
    if tuple(sorted(coefficients)) != coefficients:
        raise ValueError(
            "gradient_transfer_trace counterfactual coefficients must be sorted"
        )
    if not math.isfinite(cfg.norm_floor) or cfg.norm_floor <= 0.0:
        raise ValueError("gradient_transfer_trace.norm_floor must be finite and positive")
    return GradientTransferTraceConfig(
        enabled=cfg.enabled,
        update_end_steps=cfg.update_end_steps,
        counterfactual_trust_coefficients=coefficients,
        norm_floor=float(cfg.norm_floor),
    )


def trace_due(cfg: GradientTransferTraceConfig, *, update_end_step: int) -> bool:
    return bool(cfg.enabled and int(update_end_step) in cfg.update_end_steps)


def _coefficient_key(value: float) -> str:
    text = format(float(value), ".8g").replace("-", "m").replace(".", "p")
    return "lambda_" + text


def _norm(vector: Tensor) -> float:
    return math.sqrt(max(0.0, _dot(vector, vector)))


def _dot(left: Tensor, right: Tensor) -> float:
    total = torch.zeros((), device=left.device, dtype=torch.float64)
    for start in range(0, left.numel(), 65536):
        left_chunk = left[start : start + 65536].double()
        right_chunk = right[start : start + 65536].double()
        total.add_(torch.dot(left_chunk, right_chunk))
    return float(total.cpu())


def _cosine(left: Tensor, right: Tensor, *, floor: float) -> float:
    denominator = _norm(left) * _norm(right)
    if denominator <= floor * floor:
        return float("nan")
    return max(-1.0, min(1.0, _dot(left, right) / denominator))


def summarize_gradient_transfer(
    *,
    h4_gradient: Tensor,
    reference_gradient: Tensor,
    actual_unclipped_gradient: Tensor,
    trust_coefficient: float,
    counterfactual_coefficients: Sequence[float],
    update_start_step: int,
    update_end_step: int,
    norm_floor: float,
    adamw_descent: Tensor | None = None,
) -> dict[str, float]:
    """Summarize exact component gradients and the optional AdamW displacement.

    ``adamw_descent`` is ``theta_before - theta_after`` so a positive cosine to
    a loss gradient means the optimizer moved in that loss's descent direction.
    """

    vectors = (h4_gradient, reference_gradient, actual_unclipped_gradient)
    if any(vector.ndim != 1 for vector in vectors):
        raise ValueError("gradient-transfer vectors must be flat")
    if len({int(vector.numel()) for vector in vectors}) != 1:
        raise ValueError("gradient-transfer vectors must have the same length")
    if any(not bool(torch.isfinite(vector).all()) for vector in vectors):
        raise FloatingPointError("gradient-transfer vectors contain non-finite values")

    h4 = h4_gradient.float()
    ref = reference_gradient.float()
    actual = actual_unclipped_gradient.float()
    coefficient = float(trust_coefficient)
    reconstructed = h4 + coefficient * ref
    h4_norm = _norm(h4)
    h4_sq = h4_norm * h4_norm
    prefix = "gradient_transfer/"
    metrics = {
        prefix + "ran": 1.0,
        prefix + "update_start_step": float(update_start_step),
        prefix + "update_end_step": float(update_end_step),
        prefix + "trainable_parameters": float(h4.numel()),
        prefix + "trust_coefficient": coefficient,
        prefix + "h4/norm": h4_norm,
        prefix + "reference_unweighted/norm": _norm(ref),
        prefix + "total_reconstructed/norm": _norm(reconstructed),
        prefix + "actual_unclipped/norm": _norm(actual),
        prefix + "h4_reference/cosine": _cosine(h4, ref, floor=norm_floor),
        prefix + "reconstruction_actual/cosine": _cosine(
            reconstructed, actual, floor=norm_floor
        ),
        prefix + "reconstruction_actual/relative_error": (
            _norm(reconstructed - actual) / max(_norm(actual), norm_floor)
        ),
        prefix + "total_on_h4/projection_ratio": (
            _dot(h4, reconstructed) / h4_sq
            if h4_sq > norm_floor * norm_floor
            else float("nan")
        ),
        prefix + "reference_on_h4/projection_ratio": (
            _dot(h4, ref) / h4_sq
            if h4_sq > norm_floor * norm_floor
            else float("nan")
        ),
    }
    cross = _dot(h4, ref)
    critical = -h4_sq / cross if h4_sq > norm_floor**2 and cross < 0.0 else None
    metrics[prefix + "critical_lambda/defined"] = float(critical is not None)
    if critical is not None:
        metrics[prefix + "critical_lambda/value"] = float(critical)
    for candidate in counterfactual_coefficients:
        candidate_f = float(candidate)
        candidate_total = h4 + candidate_f * ref
        key = prefix + "counterfactual/" + _coefficient_key(candidate_f)
        metrics[key + "/total_on_h4_projection_ratio"] = (
            _dot(h4, candidate_total) / h4_sq
            if h4_sq > norm_floor**2
            else float("nan")
        )
        metrics[key + "/total_norm"] = _norm(candidate_total)

    if adamw_descent is not None:
        if adamw_descent.ndim != 1 or adamw_descent.numel() != h4.numel():
            raise ValueError("AdamW displacement does not match gradient length")
        if not bool(torch.isfinite(adamw_descent).all()):
            raise FloatingPointError("AdamW displacement contains non-finite values")
        adam = adamw_descent.float()
        metrics.update(
            {
                prefix + "adamw_descent/norm": _norm(adam),
                prefix + "adamw_descent/rms": _norm(adam)
                / math.sqrt(max(1, adam.numel())),
                prefix + "adamw_descent_on_h4/cosine": _cosine(
                    adam, h4, floor=norm_floor
                ),
                prefix + "adamw_descent_on_h4/projection_ratio": (
                    _dot(h4, adam) / h4_sq
                    if h4_sq > norm_floor**2
                    else float("nan")
                ),
                prefix + "adamw_descent_on_total/cosine": _cosine(
                    adam, reconstructed, floor=norm_floor
                ),
            }
        )
    return metrics
