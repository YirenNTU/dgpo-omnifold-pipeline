"""Composable reward for DGPO neutrino RL: component-normalized truth distance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor

from RL.DGPO_neutrino.domains.ztautau import wrapped_delta_phi


def log_pt_eta_phi_to_cartesian(log_pt: Tensor, eta: Tensor, phi: Tensor) -> Tensor:
    """Map ``(log1p(pT), η, φ)`` to ``(p_x, p_y, p_z)``."""
    pt = torch.expm1(log_pt)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return torch.stack((px, py, pz), dim=-1)


def cartesian_to_log_pt_eta_phi(px: Tensor, py: Tensor, pz: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Inverse of :func:`log_pt_eta_phi_to_cartesian`: ``(p_x, p_y, p_z)`` → ``(log1p(pT), η, φ)``."""
    pt = torch.sqrt(px * px + py * py + 1e-12)
    log_pt = torch.log1p(pt)
    phi = torch.atan2(py, px)
    eta = torch.where(pt > 1e-8, torch.asinh(pz / pt), torch.zeros_like(pz))
    return log_pt, eta, phi


def invisible_kinematics_to_cartesian(x: Tensor) -> Tensor:
    """First three features per slot: ``log1p(pT), η, φ`` → Cartesian ``(p_x, p_y, p_z)``."""
    if x.shape[-1] < 3:
        raise ValueError(f"need at least 3 features (log_pt, eta, phi), got F={x.shape[-1]}")
    return log_pt_eta_phi_to_cartesian(x[..., 0], x[..., 1], x[..., 2])


def get_event_valid_mask(
    batch: dict[str, Any],
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Per-event multiplicative mask for rewards: 1 = train/score, 0 = excluded."""
    valid = torch.ones(B, device=device, dtype=dtype)

    ev = batch.get("event_valid")
    if isinstance(ev, Tensor):
        ev = ev.to(device=device, dtype=dtype).reshape(-1)
        if ev.numel() == B:
            valid = valid * (ev > 0).to(dtype)

    ew = batch.get("event_weight")
    if isinstance(ew, Tensor):
        ew = ew.to(device=device, dtype=dtype).reshape(-1)
        if ew.numel() == B:
            valid = valid * (ew > 1e-12).to(dtype)

    xm = batch.get("x_invisible_mask")
    if isinstance(xm, Tensor) and xm.shape[0] == B:
        m = xm.to(device=device, dtype=dtype)
        has_truth = m.reshape(B, -1).sum(dim=1) > 1e-12
        valid = valid * has_truth.to(dtype)

    return valid


def apply_event_valid_to_rewards(rewards_kb: Tensor, batch: dict[str, Any]) -> Tensor:
    """Multiply ``(K, B)`` rewards by per-event validity."""
    if rewards_kb.dim() != 2:
        raise ValueError(f"expected rewards (K, B), got shape {tuple(rewards_kb.shape)}")
    K, B = rewards_kb.shape
    vm = get_event_valid_mask(batch, B, rewards_kb.device, rewards_kb.dtype).unsqueeze(0)
    return rewards_kb * vm


class BaseReward(ABC):
    """Abstract reward: maps (candidates, batch) to per-(candidate, event) scores."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging and breakdown dict keys."""

    @abstractmethod
    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return rewards of shape ``(K, B)`` for candidates ``(K, B, num_nu, F)``."""


class ComponentNormalizedTruthDistanceReward(BaseReward):
    """Negative sum of per-component normalized squared errors over (nu1, nu2) × (px, py, pz)."""

    COMPONENT_ORDER: tuple[str, ...] = (
        "nu1_px", "nu1_py", "nu1_pz",
        "nu2_px", "nu2_py", "nu2_pz",
    )

    def __init__(
        self,
        scales: dict[str, float] | tuple[float, ...] | list[float],
        *,
        cartesian: bool = False,
        eps: float = 1e-8,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self._feature_names = tuple(feature_names) if feature_names else None
        self._component_order = self._resolve_component_order(self._feature_names)
        if isinstance(scales, dict):
            try:
                scales_list = [float(scales[k]) for k in self._component_order]
            except KeyError as exc:
                raise KeyError(
                    f"scales dict missing component {exc!s}; expected keys {self._component_order}"
                ) from exc
        else:
            if len(scales) != len(self._component_order):
                raise ValueError(
                    f"scales must have {len(self._component_order)} entries, got {len(scales)}"
                )
            scales_list = [float(s) for s in scales]

        self._scales: list[float] = scales_list
        self._cartesian = bool(cartesian)
        self._eps = float(eps)
        self._last_components: dict[str, Tensor] | None = None
        self._last_component_deltas: dict[str, Tensor] | None = None
        self._last_component_truths: dict[str, Tensor] | None = None
        self._last_kinematic_deltas: dict[str, Tensor] | None = None

    @property
    def name(self) -> str:
        return "component_normalized_truth_distance"

    @property
    def scales(self) -> list[float]:
        return list(self._scales)

    def last_component_errors(self) -> dict[str, Tensor] | None:
        return self._last_components

    def last_component_deltas(self) -> dict[str, Tensor] | None:
        return self._last_component_deltas

    def last_component_truths(self) -> dict[str, Tensor] | None:
        return self._last_component_truths

    def last_kinematic_deltas(self) -> dict[str, Tensor] | None:
        return self._last_kinematic_deltas

    @staticmethod
    def _resolve_component_order(feature_names: tuple[str, ...] | None) -> tuple[str, ...]:
        if feature_names is None:
            return ComponentNormalizedTruthDistanceReward.COMPONENT_ORDER
        order: list[str] = []
        for slot in ("nu1", "nu2"):
            for feature_name in feature_names:
                order.append(f"{slot}_{feature_name}")
        return tuple(order)

    def _compute_feature_space_reward(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        m: Tensor,
        m_cand: Tensor,
    ) -> Tensor:
        if "x_invisible" not in batch:
            raise KeyError("Feature-space reward requires batch['x_invisible'].")
        truth = batch["x_invisible"]
        K, B, N_nu, _ = candidates.shape
        feature_names = self._feature_names
        if feature_names is None:
            raise RuntimeError("Feature-space reward path requires feature_names.")
        feature_dim = len(feature_names)
        if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] < feature_dim:
            raise ValueError(
                f"x_invisible shape {tuple(truth.shape)} incompatible with candidates {tuple(candidates.shape)} "
                f"for feature_names={feature_names!r}"
            )
        truth_sel = truth[..., :feature_dim].to(device=candidates.device, dtype=candidates.dtype) * m
        cand_sel = candidates[..., :feature_dim] * m_cand
        diff = cand_sel - truth_sel.unsqueeze(0)
        for feature_index, feature_name in enumerate(feature_names):
            if feature_name == "phi":
                diff[..., feature_index] = wrapped_delta_phi(cand_sel[..., feature_index], truth_sel.unsqueeze(0)[..., feature_index])

        diff_flat = diff[:, :, :2, :].reshape(K, B, -1)
        truth_flat = truth_sel[:, :2, :].reshape(B, -1).unsqueeze(0)
        scales = torch.tensor(self._scales, device=candidates.device, dtype=candidates.dtype)
        denom = scales.pow(2) + self._eps
        err_components = diff_flat.pow(2) / denom
        reward = -err_components.sum(dim=-1)
        reward = apply_event_valid_to_rewards(reward, batch)

        valid_kb = get_event_valid_mask(batch, B, reward.device, reward.dtype).unsqueeze(0)
        masked_errors = (err_components * valid_kb.unsqueeze(-1)).detach()
        self._last_components = {
            cname: masked_errors[..., i].contiguous()
            for i, cname in enumerate(self._component_order)
        }
        masked_deltas = (diff_flat * valid_kb.unsqueeze(-1)).detach()
        self._last_component_deltas = {
            cname: masked_deltas[..., i].contiguous()
            for i, cname in enumerate(self._component_order)
        }
        masked_truth = (truth_flat.expand_as(diff_flat) * valid_kb.unsqueeze(-1)).detach()
        self._last_component_truths = {
            cname: masked_truth[..., i].contiguous()
            for i, cname in enumerate(self._component_order)
        }
        kin_deltas: dict[str, Tensor] = {}
        for feature_index, feature_name in enumerate(feature_names):
            feat_delta = diff[:, :, :2, feature_index]
            truth_feat = truth_sel[:, :2, feature_index].unsqueeze(0)
            kin_deltas[feature_name] = (feat_delta * valid_kb.unsqueeze(-1)).detach().contiguous()
            kin_deltas[f"truth_{feature_name}"] = (truth_feat * valid_kb.unsqueeze(-1)).detach().contiguous()
        self._last_kinematic_deltas = kin_deltas
        return reward

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        K, B, N_nu, F = candidates.shape
        if N_nu < 2:
            raise ValueError(
                f"ComponentNormalizedTruthDistanceReward needs N_nu >= 2 (nu1, nu2), got {N_nu}"
            )

        if mask is None:
            if "x_invisible_mask" in batch:
                mask = batch["x_invisible_mask"]
            else:
                mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

        if mask.dim() == 2:
            m = mask.to(dtype=candidates.dtype, device=candidates.device).unsqueeze(-1)
        elif mask.dim() == 3 and mask.shape[-1] == 1:
            m = mask.to(dtype=candidates.dtype, device=candidates.device)
        else:
            raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")

        m_cand = m.unsqueeze(0)

        if self._feature_names is not None:
            return self._compute_feature_space_reward(candidates, batch, m, m_cand)

        if self._cartesian:
            if "x_invisible_cartesian" not in batch:
                raise KeyError(
                    "ComponentNormalizedTruthDistanceReward(cartesian=True) requires "
                    "batch['x_invisible_cartesian']"
                )
            truth = batch["x_invisible_cartesian"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] < 3:
                raise ValueError(
                    f"x_invisible_cartesian shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)} (need (B, N_nu, >=3))"
                )
            truth_c = (truth[..., :3] * m)
            cand_c = (candidates[..., :3] * m_cand)
            truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
                truth_c[..., 0], truth_c[..., 1], truth_c[..., 2],
            )
            cand_log_pt, cand_eta, cand_phi = cartesian_to_log_pt_eta_phi(
                cand_c[..., 0], cand_c[..., 1], cand_c[..., 2],
            )
        else:
            if "x_invisible" not in batch:
                raise KeyError(
                    "ComponentNormalizedTruthDistanceReward requires batch['x_invisible']"
                )
            truth = batch["x_invisible"]
            if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1] != N_nu or truth.shape[2] != F:
                raise ValueError(
                    f"x_invisible shape {tuple(truth.shape)} incompatible with "
                    f"candidates {tuple(candidates.shape)}"
                )
            if F < 3:
                raise ValueError(f"need F >= 3 (log_pt, eta, phi), got F={F}")
            truth_kin = truth * m
            cand_kin = candidates * m_cand
            truth_c = invisible_kinematics_to_cartesian(truth_kin)
            cand_c = invisible_kinematics_to_cartesian(
                cand_kin.reshape(K * B, N_nu, F)
            ).reshape(K, B, N_nu, 3)
            truth_log_pt = truth_kin[..., 0]
            truth_eta = truth_kin[..., 1]
            truth_phi = truth_kin[..., 2]
            cand_log_pt = cand_kin[..., 0]
            cand_eta = cand_kin[..., 1]
            cand_phi = cand_kin[..., 2]

        cand_six = cand_c[:, :, 0:2, :].reshape(K, B, 6)
        truth_six = truth_c[:, 0:2, :].reshape(B, 6).unsqueeze(0)
        delta_six = cand_six - truth_six

        scales = torch.tensor(
            self._scales, device=candidates.device, dtype=candidates.dtype
        )
        denom = scales.pow(2) + self._eps

        err_components = (cand_six - truth_six).pow(2) / denom
        total_err = err_components.sum(dim=-1)
        reward = -total_err
        reward = apply_event_valid_to_rewards(reward, batch)

        valid_kb = get_event_valid_mask(batch, B, reward.device, reward.dtype).unsqueeze(0)
        masked = (err_components * valid_kb.unsqueeze(-1)).detach()
        self._last_components = {
            cname: masked[..., i].contiguous()
            for i, cname in enumerate(self._component_order)
        }
        delta_masked = (delta_six * valid_kb.unsqueeze(-1)).detach()
        self._last_component_deltas = {
            cname: delta_masked[..., i].contiguous()
            for i, cname in enumerate(self._component_order)
        }
        truth_masked = (
            truth_six.expand_as(delta_six) * valid_kb.unsqueeze(-1)
        ).detach()
        self._last_component_truths = {
            cname: truth_masked[..., i].contiguous()
            for i, cname in enumerate(self._component_order)
        }

        cand_log_pt2 = cand_log_pt[:, :, 0:2]
        cand_eta2 = cand_eta[:, :, 0:2]
        cand_phi2 = cand_phi[:, :, 0:2]
        truth_log_pt2 = truth_log_pt[:, 0:2].unsqueeze(0)
        truth_eta2 = truth_eta[:, 0:2].unsqueeze(0)
        truth_phi2 = truth_phi[:, 0:2].unsqueeze(0)
        delta_phi = torch.atan2(
            torch.sin(cand_phi2 - truth_phi2),
            torch.cos(cand_phi2 - truth_phi2),
        )
        kin_mask = valid_kb.unsqueeze(-1)
        cand_pt2 = torch.expm1(cand_log_pt2.clamp(-10.0, 10.0))
        truth_pt2 = torch.expm1(truth_log_pt2.clamp(-10.0, 10.0))
        delta_pt = cand_pt2 - truth_pt2
        rel_pt = delta_pt / truth_pt2.clamp(min=1e-6)
        self._last_kinematic_deltas = {
            "pt": delta_pt.mul(kin_mask).detach().contiguous(),
            "rel_pt": rel_pt.mul(kin_mask).detach().contiguous(),
            "truth_pt": truth_pt2.mul(kin_mask).detach().contiguous(),
            "truth_log_pt": truth_log_pt2.mul(kin_mask).detach().contiguous(),
            "truth_eta": truth_eta2.mul(kin_mask).detach().contiguous(),
            "truth_phi": truth_phi2.mul(kin_mask).detach().contiguous(),
            "log_pt": ((cand_log_pt2 - truth_log_pt2) * kin_mask).detach().contiguous(),
            "eta": ((cand_eta2 - truth_eta2) * kin_mask).detach().contiguous(),
            "phi": (delta_phi * kin_mask).detach().contiguous(),
        }
        return reward


def compute_truth_l2_distances_kb(
    candidates: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
    mask: Tensor | None = None,
) -> Tensor:
    """Per-candidate masked L2 distance to truth neutrinos ``(K, B)``."""
    K, B, N_nu, F = candidates.shape

    if mask is None:
        if "x_invisible_mask" in batch:
            mask = batch["x_invisible_mask"]
        else:
            mask = torch.ones(B, N_nu, device=candidates.device, dtype=candidates.dtype)

    if mask.dim() == 2:
        m = mask.to(dtype=candidates.dtype, device=candidates.device).unsqueeze(-1)
    elif mask.dim() == 3 and mask.shape[-1] == 1:
        m = mask.to(dtype=candidates.dtype, device=candidates.device)
    else:
        raise ValueError(f"mask must be (B, N_nu) or (B, N_nu, 1), got {tuple(mask.shape)}")

    m_cand = m.unsqueeze(0)

    if cartesian:
        if "x_invisible_cartesian" not in batch:
            raise KeyError("cartesian=True requires batch['x_invisible_cartesian']")
        truth = batch["x_invisible_cartesian"]
        if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1:] != (N_nu, F):
            raise ValueError(
                f"x_invisible_cartesian shape incompatible with candidates {tuple(candidates.shape)}"
            )
        truth_kin = truth * m
        cand_kin = candidates * m_cand
        diff = cand_kin - truth_kin.unsqueeze(0)
        sq = diff.pow(2).sum(dim=(2, 3))
    else:
        if "x_invisible" not in batch:
            raise KeyError("cartesian=False requires batch['x_invisible']")
        truth = batch["x_invisible"]
        if truth.dim() != 3 or truth.shape[0] != B or truth.shape[1:] != (N_nu, F):
            raise ValueError(f"x_invisible shape incompatible with candidates {tuple(candidates.shape)}")
        if F < 3:
            raise ValueError(f"need F >= 3 for pt_eta_phi mode, got F={F}")
        truth_kin = truth * m
        cand_kin = candidates * m_cand
        truth_c = invisible_kinematics_to_cartesian(truth_kin)
        cand_c = invisible_kinematics_to_cartesian(
            cand_kin.reshape(K * B, N_nu, F)
        ).reshape(K, B, N_nu, 3)
        diff = cand_c - truth_c.unsqueeze(0)
        sq = diff.pow(2).sum(dim=(2, 3))

    dist = torch.sqrt(sq.clamp(min=0.0) + 1e-12)
    valid_1d = get_event_valid_mask(batch, B, dist.device, dist.dtype).unsqueeze(0).expand(K, -1)
    dist = torch.where(valid_1d > 0, dist, torch.full_like(dist, float("nan")))
    return dist


class RewardAggregator:
    """Weighted sum of several ``BaseReward`` sources."""

    def __init__(self) -> None:
        self.sources: list[tuple[BaseReward, float]] = []

    def add(self, reward: BaseReward, weight: float) -> None:
        self.sources.append((reward, float(weight)))

    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if not self.sources:
            raise RuntimeError("RewardAggregator has no sources; call add() first.")
        total: Tensor | float = 0.0
        breakdown: dict[str, Tensor] = {}
        for reward, w in self.sources:
            r = reward.compute(candidates, batch, mask=mask)
            breakdown[reward.name] = r
            total = total + w * r
        if not isinstance(total, Tensor):
            raise RuntimeError("Aggregator produced non-tensor total.")
        return total, breakdown


__all__ = [
    "BaseReward",
    "ComponentNormalizedTruthDistanceReward",
    "RewardAggregator",
    "apply_event_valid_to_rewards",
    "compute_truth_l2_distances_kb",
    "get_event_valid_mask",
    "cartesian_to_log_pt_eta_phi",
    "invisible_kinematics_to_cartesian",
    "log_pt_eta_phi_to_cartesian",
]
