from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

DEFAULT_FEATURE_NAMES: tuple[str, ...] = ("theta", "phi")
DEFAULT_TOKEN_FIELDS: dict[str, str] = {
    "event_token": "event_token",
    "object_token": "object_token",
}
TAU_MASS_GEV = 1.777
CM_ENERGY_GEV = 91.2


def wrapped_delta_phi(phi_a: Tensor, phi_b: Tensor) -> Tensor:
    return torch.atan2(torch.sin(phi_a - phi_b), torch.cos(phi_a - phi_b))


def _visible_theta_phi(batch: dict[str, Any], prefix: str, *, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    px = torch.as_tensor(batch[f"{prefix}_px"], device=device, dtype=dtype)
    py = torch.as_tensor(batch[f"{prefix}_py"], device=device, dtype=dtype)
    pz = torch.as_tensor(batch[f"{prefix}_pz"], device=device, dtype=dtype)
    pt = torch.sqrt(px * px + py * py + 1.0e-12)
    theta = torch.atan2(pt, pz)
    phi = torch.atan2(py, px)
    return theta, phi


def reconstruct_tau_angles_from_deltas(
    pred_deltas: Tensor,
    batch: dict[str, Any],
    *,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> tuple[Tensor, Tensor]:
    if tuple(feature_names) != DEFAULT_FEATURE_NAMES:
        raise ValueError(f"Unsupported Ztautau feature_names={feature_names!r}; expected {DEFAULT_FEATURE_NAMES!r}")
    if pred_deltas.dim() != 4 or pred_deltas.shape[2] < 2 or pred_deltas.shape[3] < 2:
        raise ValueError(
            "pred_deltas must have shape (K, B, 2, >=2) for Ztautau theta/phi targets."
        )
    device = pred_deltas.device
    dtype = pred_deltas.dtype
    theta_vis_a, phi_vis_a = _visible_theta_phi(batch, "lead_a_visible", device=device, dtype=dtype)
    theta_vis_b, phi_vis_b = _visible_theta_phi(batch, "lead_b_visible", device=device, dtype=dtype)
    tau_a_theta = theta_vis_a.unsqueeze(0) + pred_deltas[:, :, 0, 0]
    tau_b_theta = theta_vis_b.unsqueeze(0) + pred_deltas[:, :, 1, 0]
    tau_a_phi = phi_vis_a.unsqueeze(0) + pred_deltas[:, :, 0, 1]
    tau_b_phi = phi_vis_b.unsqueeze(0) + pred_deltas[:, :, 1, 1]
    tau_a_phi = torch.atan2(torch.sin(tau_a_phi), torch.cos(tau_a_phi))
    tau_b_phi = torch.atan2(torch.sin(tau_b_phi), torch.cos(tau_b_phi))
    return torch.stack((tau_a_theta, tau_a_phi), dim=-1), torch.stack((tau_b_theta, tau_b_phi), dim=-1)


def tau_back_to_back_metrics(
    pred_deltas: Tensor,
    batch: dict[str, Any],
    *,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> dict[str, Tensor]:
    tau_a, tau_b = reconstruct_tau_angles_from_deltas(pred_deltas, batch, feature_names=feature_names)
    theta_a, phi_a = tau_a[..., 0], tau_a[..., 1]
    theta_b, phi_b = tau_b[..., 0], tau_b[..., 1]

    sin_a = torch.sin(theta_a)
    sin_b = torch.sin(theta_b)
    cos_opening = (
        sin_a * sin_b * torch.cos(phi_a - phi_b)
        + torch.cos(theta_a) * torch.cos(theta_b)
    )
    delta_phi_to_pi = torch.abs(torch.abs(wrapped_delta_phi(phi_a, phi_b)) - math.pi)
    back_to_back_loss = (cos_opening + 1.0).pow(2) + delta_phi_to_pi.pow(2)
    return {
        "cos_opening": cos_opening,
        "delta_phi_to_pi": delta_phi_to_pi,
        "back_to_back_loss": back_to_back_loss,
    }


def tau_calibration_magnitude_metrics(
    pred_deltas: Tensor,
    batch: dict[str, Any],
    *,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> dict[str, Tensor]:
    """Differentiable mirror of the post-calibration direction-change magnitude."""
    tau_a, tau_b = reconstruct_tau_angles_from_deltas(pred_deltas, batch, feature_names=feature_names)
    theta_a, phi_a = tau_a[..., 0], tau_a[..., 1]
    theta_b, phi_b = tau_b[..., 0], tau_b[..., 1]

    energy = pred_deltas.new_tensor(CM_ENERGY_GEV / 2.0)
    tau_mass = pred_deltas.new_tensor(TAU_MASS_GEV)
    momentum = torch.sqrt(torch.clamp(energy * energy - tau_mass * tau_mass, min=1.0e-12))

    sin_theta_a = torch.sin(theta_a)
    sin_theta_b = torch.sin(theta_b)
    px_a = momentum * sin_theta_a * torch.cos(phi_a)
    py_a = momentum * sin_theta_a * torch.sin(phi_a)
    pz_a = momentum * torch.cos(theta_a)
    px_b = momentum * sin_theta_b * torch.cos(phi_b)
    py_b = momentum * sin_theta_b * torch.sin(phi_b)
    pz_b = momentum * torch.cos(theta_b)

    px_shift = px_a + px_b
    py_shift = py_a + py_b
    pz_shift = pz_a + pz_b

    px_a_corr = px_a - 0.5 * px_shift
    py_a_corr = py_a - 0.5 * py_shift
    pz_a_corr = pz_a - 0.5 * pz_shift
    norm_a_corr = torch.sqrt(
        torch.clamp(px_a_corr.pow(2) + py_a_corr.pow(2) + pz_a_corr.pow(2), min=1.0e-12)
    )

    ux_a = px_a_corr / norm_a_corr
    uy_a = py_a_corr / norm_a_corr
    uz_a = pz_a_corr / norm_a_corr

    px_a_final = momentum * ux_a
    py_a_final = momentum * uy_a
    pz_a_final = momentum * uz_a
    px_b_final = -px_a_final
    py_b_final = -py_a_final
    pz_b_final = -pz_a_final

    def _eta_phi_from_xyz(px: Tensor, py: Tensor, pz: Tensor) -> tuple[Tensor, Tensor]:
        pt = torch.sqrt(torch.clamp(px * px + py * py, min=1.0e-12))
        eta = torch.asinh(pz / pt)
        phi = torch.atan2(py, px)
        return eta, phi

    eta_a_before, phi_a_before = _eta_phi_from_xyz(px_a, py_a, pz_a)
    eta_b_before, phi_b_before = _eta_phi_from_xyz(px_b, py_b, pz_b)
    eta_a_after, phi_a_after = _eta_phi_from_xyz(px_a_final, py_a_final, pz_a_final)
    eta_b_after, phi_b_after = _eta_phi_from_xyz(px_b_final, py_b_final, pz_b_final)

    delta_r_a = torch.sqrt(
        torch.clamp(
            (eta_a_before - eta_a_after).pow(2)
            + wrapped_delta_phi(phi_a_before, phi_a_after).pow(2),
            min=0.0,
        )
    )
    delta_r_b = torch.sqrt(
        torch.clamp(
            (eta_b_before - eta_b_after).pow(2)
            + wrapped_delta_phi(phi_b_before, phi_b_after).pow(2),
            min=0.0,
        )
    )
    return {
        "calibration_deltaR_a": delta_r_a,
        "calibration_deltaR_b": delta_r_b,
        "calibration_deltaR_sum": delta_r_a + delta_r_b,
    }


def build_feature_space_scales(
    normalization_dict: dict[str, Any],
    *,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> dict[str, float]:
    if "invisible_std" not in normalization_dict:
        raise ValueError("Ztautau feature-space reward requires normalization.pt['invisible_std'].")
    std_tensor = normalization_dict["invisible_std"]["Source"]
    std_values = std_tensor.detach().cpu().tolist() if hasattr(std_tensor, "detach") else list(std_tensor)
    feature_names = tuple(feature_names)
    if len(std_values) < len(feature_names):
        raise ValueError(
            f"invisible_std has {len(std_values)} entries but feature_names={feature_names!r} needs {len(feature_names)}."
        )
    scales: dict[str, float] = {}
    for slot in ("nu1", "nu2"):
        for index, feature_name in enumerate(feature_names):
            scales[f"{slot}_{feature_name}"] = float(std_values[index])
    return scales
