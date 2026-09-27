"""Small reversible-state helpers for hard-boundary AdamW proposals."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


def resolve_parameter_update_rms_calibration(
    dgpo_config: Mapping[str, Any],
) -> tuple[float | None, float, float]:
    """Validate the optional exact global AdamW-displacement calibration."""

    raw = dgpo_config.get("parameter_update_rms_calibration")
    if raw is None:
        return None, 1.0, 1.0
    if not isinstance(raw, Mapping):
        raise ValueError("dgpo.parameter_update_rms_calibration must be a mapping")
    unknown = set(raw) - {"enabled", "target_rms", "minimum_scale", "maximum_scale"}
    if unknown:
        raise ValueError(
            "unknown dgpo.parameter_update_rms_calibration keys: "
            f"{sorted(unknown)}"
        )
    enabled = raw.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("parameter_update_rms_calibration.enabled must be boolean")
    if not enabled:
        return None, 1.0, 1.0
    if raw.get("target_rms") is None:
        raise ValueError(
            "enabled parameter_update_rms_calibration requires target_rms"
        )
    target = float(raw["target_rms"])
    minimum = float(raw.get("minimum_scale", 1.0e-3))
    maximum = float(raw.get("maximum_scale", 1.0e3))
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError(
            "parameter_update_rms_calibration.target_rms must be finite and positive"
        )
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum <= 0.0
        or maximum < minimum
    ):
        raise ValueError(
            "parameter_update_rms_calibration scale bounds must be finite, "
            "positive, and ordered"
        )
    return target, minimum, maximum


@torch.no_grad()
def assign_scaled_trainable_update_(
    model: torch.nn.Module,
    theta_old: Mapping[str, Tensor],
    theta_candidate: Mapping[str, Tensor],
    scale: float,
) -> None:
    """Assign ``old + scale * (candidate - old)`` to trainable parameters.

    Unlike hard-trust interpolation, this deliberately permits ``scale > 1``.
    AdamW's moment update is independent of LR and its parameter displacement
    is linear in a common LR multiplier, so this is the exact parameter result
    of rerunning the proposal with every optimizer-group LR scaled equally.
    """

    scale_f = float(scale)
    if not math.isfinite(scale_f) or scale_f <= 0.0:
        raise ValueError("parameter-update RMS scale must be finite and positive")
    expected = set(theta_old)
    if expected != set(theta_candidate):
        raise ValueError("parameter-update RMS snapshots have different keys")
    assigned = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in theta_old or name not in theta_candidate:
            raise KeyError(
                f"missing trainable parameter {name!r} in RMS calibration snapshot"
            )
        old = theta_old[name].to(device=parameter.device, dtype=parameter.dtype)
        candidate = theta_candidate[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        parameter.data.copy_(old + scale_f * (candidate - old))
        assigned += 1
    if assigned != len(expected):
        raise RuntimeError(
            "RMS calibration did not consume every trainable snapshot: "
            f"assigned={assigned}, snapshot={len(expected)}"
        )


def snapshot_adamw_state_for_transaction(
    optimizer: Any,
) -> list[tuple[Tensor, dict[str, Any]]]:
    """Copy AdamW state to CPU before a potentially rejected trust proposal.

    Parameter tensors are restored separately. The scheduler stays outside the
    snapshot because a rejected proposal never advances it. CPU copies avoid a
    second complete set of Adam moments on every GPU during boundary checks.
    """

    inner = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(inner, torch.optim.AdamW):
        raise TypeError(
            "transactional trust rejection currently requires torch.optim.AdamW"
        )
    snapshot: list[tuple[Tensor, dict[str, Any]]] = []
    for parameter, parameter_state in inner.state.items():
        saved: dict[str, Any] = {}
        for key, value in parameter_state.items():
            if isinstance(value, Tensor):
                saved[key] = {
                    "tensor": value.detach().to(device="cpu", copy=True),
                    "device": value.device,
                }
            else:
                saved[key] = copy.deepcopy(value)
        snapshot.append((parameter, saved))
    return snapshot


@torch.no_grad()
def restore_adamw_state_from_transaction_(
    optimizer: Any,
    snapshot: Sequence[tuple[Tensor, Mapping[str, Any]]],
) -> None:
    """Restore the exact pre-proposal AdamW moments and step counters."""

    inner = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(inner, torch.optim.AdamW):
        raise TypeError(
            "transactional trust rejection currently requires torch.optim.AdamW"
        )
    inner.state.clear()
    for parameter, saved in snapshot:
        restored: dict[str, Any] = {}
        for key, value in saved.items():
            if (
                isinstance(value, Mapping)
                and isinstance(value.get("tensor"), Tensor)
                and "device" in value
            ):
                restored[key] = value["tensor"].to(
                    device=value["device"], copy=True
                )
            else:
                restored[key] = copy.deepcopy(value)
        inner.state[parameter] = restored


__all__ = [
    "assign_scaled_trainable_update_",
    "resolve_parameter_update_rms_calibration",
    "restore_adamw_state_from_transaction_",
    "snapshot_adamw_state_for_transaction",
]
