"""Ztautau DGPO posterior calibration and targeted-physics validation metrics.

TARP uses every validation candidate.  Distribution panels deliberately use
candidate zero, an unbiased policy draw, rather than the reward-best member of
the group.  OmniFold fitting and adaptive refits remain independent K=1
population operations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor

from RL.DGPO_neutrino.diagnostics.shared_metrics import (
    BinnedTarpResult,
    GRID,
    Prediction,
    TarpResult,
    evaluate_tarp,
    evaluate_tarp_binned,
    holm_adjusted,
)
from RL.DGPO_neutrino.domains.ztautau import (
    reconstruct_tau_angles_from_deltas,
    tau_back_to_back_metrics,
    tau_calibration_magnitude_metrics,
    wrapped_delta_phi,
)

_log = logging.getLogger(__name__)

TARGET_COMPONENT_NAMES: tuple[str, ...] = (
    "tau_a_delta_theta",
    "tau_a_delta_phi",
    "tau_b_delta_theta",
    "tau_b_delta_phi",
)
DEFAULT_TARP_ARMS: tuple[str, ...] = ("full", "rank_copula")


def _cfg_get(block: Any, key: str, default: Any = None) -> Any:
    if block is None:
        return default
    if isinstance(block, Mapping):
        return block.get(key, default)
    return getattr(block, key, default)


def _to_numpy(value: Tensor) -> np.ndarray:
    return value.detach().to(dtype=torch.float64).cpu().numpy()


def _canonical_target_block(values: Tensor) -> Tensor:
    """Return the first two theta/phi slots with periodic deltas wrapped."""
    if values.shape[-2:] != (2, 2):
        raise ValueError(
            "Ztautau targets must end in (2 slots, 2 theta/phi features), got "
            f"{tuple(values.shape)}"
        )
    result = values.clone()
    result[..., 1] = torch.atan2(torch.sin(result[..., 1]), torch.cos(result[..., 1]))
    return result


def visible_pair_acoplanarity(
    batch: Mapping[str, Any], *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Conditioning coordinate for binned TARP, independent of target/candidate."""
    required = (
        "lead_a_visible_px",
        "lead_a_visible_py",
        "lead_b_visible_px",
        "lead_b_visible_py",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(
            "Ztautau TARP needs visible tau-leg directions; missing "
            + ", ".join(missing)
        )
    phi_a = torch.atan2(
        torch.as_tensor(batch["lead_a_visible_py"], device=device, dtype=dtype),
        torch.as_tensor(batch["lead_a_visible_px"], device=device, dtype=dtype),
    ).reshape(-1)
    phi_b = torch.atan2(
        torch.as_tensor(batch["lead_b_visible_py"], device=device, dtype=dtype),
        torch.as_tensor(batch["lead_b_visible_px"], device=device, dtype=dtype),
    ).reshape(-1)
    return torch.abs(torch.abs(wrapped_delta_phi(phi_a, phi_b)) - math.pi)


def _both_tau_slots_valid(batch: Mapping[str, Any], event_valid: Tensor) -> Tensor:
    valid = event_valid.reshape(-1) > 0
    mask = batch.get("x_invisible_mask")
    if isinstance(mask, Tensor):
        slot_mask = mask
        if slot_mask.ndim == 3 and slot_mask.shape[-1] == 1:
            slot_mask = slot_mask.squeeze(-1)
        if slot_mask.ndim != 2 or slot_mask.shape[1] < 2:
            raise ValueError(
                "x_invisible_mask must be (B,N) or (B,N,1) with two tau slots"
            )
        valid = valid & (slot_mask[:, :2].to(device=valid.device) > 0).all(dim=1)
    return valid


def collect_ztautau_validation_arrays(
    candidates: Tensor,
    reference_candidates: Tensor,
    batch: dict[str, Any],
    event_valid: Tensor,
) -> dict[str, np.ndarray]:
    """Collect one validation batch without selecting a reward-best candidate.

    ``candidates`` is ``(K,B,2,2)`` and all K members are retained for TARP.
    Candidate zero is used for every 1D current-policy distribution.
    """
    if candidates.ndim != 4 or tuple(candidates.shape[2:]) != (2, 2):
        raise ValueError(
            "Ztautau validation candidates must be (K,B,2,2), got "
            f"{tuple(candidates.shape)}"
        )
    if reference_candidates.ndim != 4 or tuple(reference_candidates.shape[2:]) != (2, 2):
        raise ValueError(
            "Ztautau reference candidates must be (Kref,B,2,2), got "
            f"{tuple(reference_candidates.shape)}"
        )
    if candidates.shape[1] != reference_candidates.shape[1]:
        raise ValueError("current and reference validation batches disagree")
    truth_raw = batch.get("x_invisible")
    if not isinstance(truth_raw, Tensor):
        raise KeyError("Ztautau validation needs batch['x_invisible']")
    if truth_raw.ndim != 3 or tuple(truth_raw.shape[1:3]) != (2, 2):
        raise ValueError(f"Ztautau truth must be (B,2,2), got {tuple(truth_raw.shape)}")

    truth = _canonical_target_block(
        truth_raw.to(device=candidates.device, dtype=candidates.dtype)
    )
    current_all = _canonical_target_block(candidates)
    reference_all = _canonical_target_block(reference_candidates)
    valid = _both_tau_slots_valid(batch, event_valid.to(device=candidates.device))
    if not bool(valid.any().item()):
        return {}

    truth_tau_a, truth_tau_b = reconstruct_tau_angles_from_deltas(
        truth.unsqueeze(0), batch
    )
    current_tau_a, current_tau_b = reconstruct_tau_angles_from_deltas(
        current_all[:1], batch
    )
    reference_tau_a, reference_tau_b = reconstruct_tau_angles_from_deltas(
        reference_all[:1], batch
    )

    truth_topology = tau_back_to_back_metrics(truth.unsqueeze(0), batch)
    current_topology = tau_back_to_back_metrics(current_all[:1], batch)
    reference_topology = tau_back_to_back_metrics(reference_all[:1], batch)
    truth_calibration = tau_calibration_magnitude_metrics(truth.unsqueeze(0), batch)
    current_calibration = tau_calibration_magnitude_metrics(current_all[:1], batch)
    reference_calibration = tau_calibration_magnitude_metrics(reference_all[:1], batch)

    truth_b4 = truth.reshape(truth.shape[0], 4)
    current_bk4 = current_all.permute(1, 0, 2, 3).reshape(
        truth.shape[0], current_all.shape[0], 4
    )
    profile = visible_pair_acoplanarity(
        batch, device=candidates.device, dtype=candidates.dtype
    )
    out: dict[str, np.ndarray] = {
        "_tarp_truth": _to_numpy(truth_b4[valid]),
        "_tarp_candidates": _to_numpy(current_bk4[valid]),
        "_tarp_profile": _to_numpy(profile[valid]),
    }

    target_triplets = {
        "target/tau_a_delta_theta": (truth[:, 0, 0], current_all[0, :, 0, 0], reference_all[0, :, 0, 0]),
        "target/tau_a_delta_phi": (truth[:, 0, 1], current_all[0, :, 0, 1], reference_all[0, :, 0, 1]),
        "target/tau_b_delta_theta": (truth[:, 1, 0], current_all[0, :, 1, 0], reference_all[0, :, 1, 0]),
        "target/tau_b_delta_phi": (truth[:, 1, 1], current_all[0, :, 1, 1], reference_all[0, :, 1, 1]),
        "reco/tau_a_theta": (truth_tau_a[0, :, 0], current_tau_a[0, :, 0], reference_tau_a[0, :, 0]),
        "reco/tau_a_phi": (truth_tau_a[0, :, 1], current_tau_a[0, :, 1], reference_tau_a[0, :, 1]),
        "reco/tau_b_theta": (truth_tau_b[0, :, 0], current_tau_b[0, :, 0], reference_tau_b[0, :, 0]),
        "reco/tau_b_phi": (truth_tau_b[0, :, 1], current_tau_b[0, :, 1], reference_tau_b[0, :, 1]),
    }
    for name in ("cos_opening", "delta_phi_to_pi", "back_to_back_loss"):
        target_triplets[f"topology/{name}"] = (
            truth_topology[name][0],
            current_topology[name][0],
            reference_topology[name][0],
        )
    for name in (
        "calibration_deltaR_a",
        "calibration_deltaR_b",
        "calibration_deltaR_sum",
    ):
        target_triplets[f"topology/{name}"] = (
            truth_calibration[name][0],
            current_calibration[name][0],
            reference_calibration[name][0],
        )

    for name, (truth_values, current_values, reference_values) in target_triplets.items():
        out[f"{name}/truth"] = _to_numpy(truth_values[valid])
        out[f"{name}/current"] = _to_numpy(current_values[valid])
        out[f"{name}/ref"] = _to_numpy(reference_values[valid])
    return out


def _finite_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _observable_edges(name: str, arrays: tuple[np.ndarray, ...], bins: int) -> np.ndarray:
    if name.endswith("_phi"):
        return np.linspace(-math.pi, math.pi, bins + 1)
    if name.startswith("reco/") and name.endswith("_theta"):
        return np.linspace(0.0, math.pi, bins + 1)
    if name.endswith("cos_opening"):
        return np.linspace(-1.0, 1.0, bins + 1)
    if name.endswith("delta_phi_to_pi"):
        return np.linspace(0.0, math.pi, bins + 1)
    finite = [_finite_1d(values) for values in arrays]
    finite = [values for values in finite if values.size]
    if not finite:
        return np.linspace(-1.0, 1.0, bins + 1)
    merged = np.concatenate(finite)
    lo, hi = [float(value) for value in np.nanpercentile(merged, [0.5, 99.5])]
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(merged)) if merged.size else 0.0
        span = max(abs(center) * 0.1, 1.0e-3)
        lo, hi = center - span, center + span
    pad = max(0.05 * (hi - lo), 1.0e-6)
    return np.linspace(lo - pad, hi + pad, bins + 1)


def _histogram_jsd(first: np.ndarray, second: np.ndarray, edges: np.ndarray) -> float:
    first = _finite_1d(first)
    second = _finite_1d(second)
    if first.size == 0 or second.size == 0:
        return float("nan")
    p = np.histogram(first, bins=edges)[0].astype(np.float64)
    q = np.histogram(second, bins=edges)[0].astype(np.float64)
    if p.sum() <= 0.0 or q.sum() <= 0.0:
        return float("nan")
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def _kl(left: np.ndarray, right: np.ndarray) -> float:
        keep = left > 0.0
        return float(np.sum(left[keep] * np.log2(left[keep] / right[keep])))

    return float(math.sqrt(max(0.0, 0.5 * (_kl(p, midpoint) + _kl(q, midpoint)))))


def _overlay_image(
    name: str,
    truth: np.ndarray,
    current: np.ndarray,
    reference: np.ndarray,
    edges: np.ndarray,
) -> Any:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for values, label, color, style in (
        (truth, "Truth", "black", "-"),
        (current, "Current policy (candidate 0)", "C0", "-"),
        (reference, "Frozen reference (candidate 0)", "C2", "--"),
    ):
        values = _finite_1d(values)
        counts, _ = np.histogram(values, bins=edges)
        widths = np.diff(edges)
        density = counts / max(float(counts.sum()), 1.0) / widths
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.plot(centers, density, label=label, color=color, linestyle=style, linewidth=2.0)
    axis.set_xlabel(name.rsplit("/", 1)[-1])
    axis.set_ylabel("Density")
    axis.set_title(f"Ztautau validation: {name}")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    image = wandb.Image(figure)
    plt.close(figure)
    return image


def _tarp_scalars(result: BinnedTarpResult, prefix: str = "val_tarp") -> dict[str, float]:
    output = {f"{prefix}/{metric.name}": float(metric.value) for metric in result.metrics}
    family = {
        key: float(value)
        for key, value in dict(result.metrics[0].extra.get("raw_pvalues", {})).items()
    }
    adjusted = holm_adjusted(family) if family else {}
    for bin_index, block in sorted(result.bins.items()):
        output[f"{prefix}/bin{bin_index}/events"] = float(result.counts[bin_index])
        for arm, entry in block.arms.items():
            family_key = f"bin{bin_index}:{arm}"
            output[f"{prefix}/bin{bin_index}/{arm}_pvalue"] = float(entry.pvalue)
            output[f"{prefix}/bin{bin_index}/{arm}_holm_pvalue"] = float(
                adjusted.get(family_key, float("nan"))
            )
            output[f"{prefix}/bin{bin_index}/{arm}_max_gap"] = float(entry.max_abs_gap)
    return output


def _pooled_tarp_scalars(result: TarpResult, prefix: str = "val_tarp/pooled") -> dict[str, float]:
    output = {f"{prefix}/{metric.name}": float(metric.value) for metric in result.metrics}
    for arm, entry in result.arms.items():
        output[f"{prefix}/{arm}_pvalue"] = float(entry.pvalue)
        output[f"{prefix}/{arm}_holm_pvalue"] = float(entry.holm_pvalue)
        output[f"{prefix}/{arm}_max_gap"] = float(entry.max_abs_gap)
    return output


def _tarp_image(result: TarpResult | BinnedTarpResult) -> Any:
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb

    if isinstance(result, BinnedTarpResult):
        bins = sorted(result.bins)
        arms = list(result.bins[bins[0]].arms)
        figure, axes = plt.subplots(
            len(bins), len(arms),
            figsize=(3.5 * len(arms), 3.1 * len(bins)),
            squeeze=False,
        )
        for row, bin_index in enumerate(bins):
            for column, arm in enumerate(arms):
                entry = result.bins[bin_index].arms[arm]
                axis = axes[row][column]
                axis.plot([0, 1], [0, 1], color="gray", linewidth=1.0, linestyle=":")
                axis.plot(GRID, entry.mean_null_curve, color="C2", linewidth=1.4, label="null mean")
                axis.plot(GRID, entry.observed_curve, color="C0", linewidth=2.0, label="observed")
                axis.set_title(f"{arm}; p={entry.pvalue:.3f}", fontsize=9)
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                axis.grid(True, alpha=0.25)
                if column == 0:
                    axis.set_ylabel(
                        f"acoplanarity bin {bin_index}\n"
                        f"[{result.edges[bin_index]:.3g}, {result.edges[bin_index + 1]:.3g}]"
                    )
        figure.suptitle(
            "Ztautau TARP by visible-pair acoplanarity; "
            f"min Holm p={result.metrics[0].value:.3f}"
        )
    else:
        arms = list(result.arms)
        figure, axes = plt.subplots(1, len(arms), figsize=(3.5 * len(arms), 3.4), squeeze=False)
        for column, arm in enumerate(arms):
            entry = result.arms[arm]
            axis = axes[0][column]
            axis.plot([0, 1], [0, 1], color="gray", linewidth=1.0, linestyle=":")
            axis.plot(GRID, entry.mean_null_curve, color="C2", linewidth=1.4, label="null mean")
            axis.plot(GRID, entry.observed_curve, color="C0", linewidth=2.0, label="observed")
            axis.set_title(f"{arm}; p={entry.pvalue:.3f}", fontsize=9)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.25)
        figure.suptitle("Ztautau pooled TARP (orientation only)")
    figure.tight_layout()
    image = wandb.Image(figure)
    plt.close(figure)
    return image


def build_ztautau_validation_metrics(
    arrays: Mapping[str, np.ndarray],
    *,
    val_k: int,
    tarp_config: Any | None,
    metrics_config: Any | None,
    include_images: bool = True,
) -> dict[str, Any]:
    """Build W&B-ready Ztautau metrics from globally gathered validation arrays."""
    output: dict[str, Any] = {}
    physics_enabled = bool(_cfg_get(metrics_config, "enabled", False))
    tarp_enabled = bool(_cfg_get(tarp_config, "enabled", False))
    if not physics_enabled and not tarp_enabled:
        return output
    if physics_enabled:
        candidate_index = int(_cfg_get(metrics_config, "candidate_index", 0))
        if candidate_index != 0:
            raise ValueError(
                "dgpo.ztautau_metrics.candidate_index must be 0: the 1D validation "
                "panels use a fixed unbiased policy draw and never select by reward"
            )
        bins = max(10, int(_cfg_get(metrics_config, "bins", 60)))

        observable_names = sorted(
            key[: -len("/truth")]
            for key in arrays
            if key.endswith("/truth") and not key.startswith("_")
        )
        for name in observable_names:
            truth = np.asarray(arrays.get(f"{name}/truth", []), dtype=np.float64)
            current = np.asarray(arrays.get(f"{name}/current", []), dtype=np.float64)
            reference = np.asarray(arrays.get(f"{name}/ref", []), dtype=np.float64)
            edges = _observable_edges(name, (truth, current, reference), bins)
            output[f"val_ztautau/jsd/current/{name}"] = _histogram_jsd(truth, current, edges)
            output[f"val_ztautau/jsd/ref/{name}"] = _histogram_jsd(truth, reference, edges)
            for label, values in (("current", current), ("ref", reference)):
                n = min(truth.size, values.size)
                if n and name.endswith("_phi"):
                    residual = _finite_1d(
                        np.arctan2(
                            np.sin(values[:n] - truth[:n]),
                            np.cos(values[:n] - truth[:n]),
                        )
                    )
                else:
                    residual = _finite_1d(values[:n] - truth[:n]) if n else np.array([])
                output[f"val_ztautau/residual/{label}/{name}/mean"] = (
                    float(np.mean(residual)) if residual.size else float("nan")
                )
                output[f"val_ztautau/residual/{label}/{name}/abs_mean"] = (
                    float(np.mean(np.abs(residual))) if residual.size else float("nan")
                )
            if include_images:
                output[f"val_ztautau/{name}"] = _overlay_image(
                    name, truth, current, reference, edges
                )

    if not tarp_enabled:
        return output
    output["val_tarp/enabled"] = 1.0
    if int(val_k) < 2:
        output["val_tarp/skipped_insufficient_k"] = 1.0
        _log.warning("[DGPO/tarp] skipped: validation_K=%s; TARP needs K>=2", val_k)
        return output

    truth_flat = np.asarray(arrays.get("_tarp_truth", []), dtype=np.float64).reshape(-1)
    candidate_flat = np.asarray(arrays.get("_tarp_candidates", []), dtype=np.float64).reshape(-1)
    profile = np.asarray(arrays.get("_tarp_profile", []), dtype=np.float64).reshape(-1)
    if truth_flat.size == 0 or truth_flat.size % 4 != 0:
        output["val_tarp/skipped_empty_pool"] = 1.0
        return output
    n_events = truth_flat.size // 4
    expected_candidate_values = n_events * int(val_k) * 4
    if candidate_flat.size != expected_candidate_values or profile.size != n_events:
        raise ValueError(
            "gathered Ztautau TARP arrays disagree: "
            f"truth_events={n_events}, candidate_values={candidate_flat.size}, "
            f"profile_values={profile.size}, validation_K={val_k}"
        )
    truth = truth_flat.reshape(n_events, 4)
    candidates = candidate_flat.reshape(n_events, int(val_k), 4)
    prediction = Prediction(
        truth=truth,
        candidates=candidates,
        component_names=TARGET_COMPONENT_NAMES,
    )
    arms_raw = _cfg_get(tarp_config, "arms", DEFAULT_TARP_ARMS)
    arms = tuple(str(value) for value in arms_raw)
    n_bins = int(_cfg_get(tarp_config, "n_bins", 4))
    n_null = int(_cfg_get(tarp_config, "n_null_assignments", 399))
    alpha = float(_cfg_get(tarp_config, "alpha", 0.05))
    power_floor = float(n_bins * len(arms)) / float(n_null + 1)
    output["val_tarp/geometry/events"] = float(n_events)
    output["val_tarp/geometry/candidates"] = float(val_k)
    output["val_tarp/geometry/holm_power_floor"] = power_floor
    if power_floor >= alpha:
        _log.warning(
            "[DGPO/tarp] underpowered: %s bins x %s arms / (%s+1) gives Holm floor %.4g >= alpha %.4g",
            n_bins,
            len(arms),
            n_null,
            power_floor,
            alpha,
        )
    try:
        binned = evaluate_tarp_binned(
            prediction,
            profile,
            n_bins=n_bins,
            arms=arms,
            n_reference_trials=int(_cfg_get(tarp_config, "n_reference_trials", 8)),
            n_null_assignments=n_null,
            min_events=int(_cfg_get(tarp_config, "min_events", 500)),
            seed=int(_cfg_get(tarp_config, "seed", 20260820)),
            alpha=alpha,
        )
    except ValueError as exc:
        output["val_tarp/skipped_underpopulated"] = 1.0
        _log.warning("[DGPO/tarp] binned TARP skipped: %s", exc)
        binned = None
    if binned is not None:
        output.update(_tarp_scalars(binned))
        if include_images:
            output["val_tarp/coverage"] = _tarp_image(binned)

    if bool(_cfg_get(tarp_config, "pooled_panel", True)):
        pooled = evaluate_tarp(
            prediction,
            arms=arms,
            n_reference_trials=int(_cfg_get(tarp_config, "n_reference_trials", 8)),
            n_null_assignments=n_null,
            seed=int(_cfg_get(tarp_config, "seed", 20260820)),
            alpha=alpha,
        )
        output.update(_pooled_tarp_scalars(pooled))
        if include_images:
            output["val_tarp/pooled_coverage"] = _tarp_image(pooled)
    return output


__all__ = [
    "DEFAULT_TARP_ARMS",
    "TARGET_COMPONENT_NAMES",
    "build_ztautau_validation_metrics",
    "collect_ztautau_validation_arrays",
    "visible_pair_acoplanarity",
]
