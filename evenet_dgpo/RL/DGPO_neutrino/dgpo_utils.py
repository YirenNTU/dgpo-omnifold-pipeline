"""Pure utilities for DGPO neutrino RL: advantages, batch tiling, and DGPO loss."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma


def _dgpo_cfg_get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    """Read YAML/DotDict key from either a dict or DotDict."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


ADVANTAGE_ESTIMATOR_ZSCORE = "zscore"
ADVANTAGE_ESTIMATOR_LOO_UNSCALED = "leave_one_out_unscaled"
VALID_ADVANTAGE_ESTIMATORS = frozenset(
    {ADVANTAGE_ESTIMATOR_ZSCORE, ADVANTAGE_ESTIMATOR_LOO_UNSCALED}
)


def compute_per_event_advantage(
    rewards: Tensor,
    eps: float = 1e-6,
    *,
    estimator: str = ADVANTAGE_ESTIMATOR_ZSCORE,
) -> tuple[Tensor, Tensor]:
    """Per-event advantages over candidates (dim 0 of ``(K, B)``).

    ``leave_one_out_unscaled`` preserves the density-ratio reward scale:
    ``A_i = r_i - mean_{j!=i}(r_j)``. The legacy ``zscore`` path remains the
    default for non-OmniFold configurations.

    Args:
        rewards: Shape ``(K, B)`` — K candidates, B events.
        eps: Added to std for the z-score path only.
        estimator: ``leave_one_out_unscaled`` or ``zscore``.

    Returns:
        ``(advantages, weights)`` each ``(K, B)``; ``weights = |advantages|``.
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be (K, B), got {tuple(rewards.shape)}")
    k = int(rewards.shape[0])
    if estimator == ADVANTAGE_ESTIMATOR_LOO_UNSCALED:
        if k < 2:
            raise ValueError(
                "leave_one_out_unscaled needs K>=2 candidates per event, "
                f"got K={k}"
            )
        other_mean = (rewards.sum(dim=0, keepdim=True) - rewards) / float(k - 1)
        advantages = rewards - other_mean
    elif estimator == ADVANTAGE_ESTIMATOR_ZSCORE:
        mu = rewards.mean(dim=0)
        std = rewards.std(dim=0, unbiased=False) + eps
        advantages = (rewards - mu.unsqueeze(0)) / std.unsqueeze(0)
    else:
        raise ValueError(
            f"unsupported DGPO advantage estimator: {estimator!r}; "
            f"expected one of {sorted(VALID_ADVANTAGE_ESTIMATORS)}"
        )
    weights = advantages.abs()
    return advantages, weights

def repeat_batch_for_candidates(
    batch: dict[str, Any],
    K: int,
    *,
    tensor_keys: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Tile each tensor value K times along batch (dim 0) for flattened K*B forwards.

    For tensor ``v`` of shape ``(B, ...)``, the result has shape ``(K*B, ...)`` with
    layout ``[cand0_evt0..evtB-1, cand1_evt0.., ...]``.

    Non-tensor values are shallow-copied into the output dict unchanged.

    Args:
        batch: Mapping of string keys to tensors or other objects.
        K: Number of candidate repetitions per event.

    Returns:
        New dict with the same keys; tensor values expanded as above.
    """
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, Tensor):
            if tensor_keys is not None and key not in tensor_keys:
                continue
            v = val
            out[key] = (
                v.unsqueeze(0)
                .expand(K, *v.shape)
                .reshape(K * v.shape[0], *v.shape[1:])
                .contiguous()
            )
        else:
            out[key] = val
    return out


def build_dgpo_loss(
    L_cur_2d: Tensor,
    L_ref_2d: Tensor,
    advantages: Tensor,
    beta_dgpo: float,
    K: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Velocity-space DGPO loss with detached per-event gate (pure DGPO main term).

    Detached gate: ``Delta = stopgrad(L_cur - L_ref)``, ``M_e`` from
    ``(beta_dgpo / K) * sum_i A_{i,e} Delta_{i,e}``, ``w_e = stopgrad(sigmoid(M_e))``.
    Main term: ``mean_{e,i}( w_e * A_{i,e} * L_cur )`` — gradients only through ``L_cur_2d``.

    Args:
        L_cur_2d: Per-(candidate, event) current velocity MSE, shape ``(K, B)`` (trainable).
        L_ref_2d: Same for frozen reference, shape ``(K, B)`` (no grad).
        advantages: Shape ``(K, B)``; ``advantages.shape[0]`` must equal ``K``.
        beta_dgpo: Scales the detached group statistic ``M_e``.
        K: Number of candidates per event.

    Returns:
        Scalar ``loss_total`` and diagnostics:
        ``loss_total``, ``loss_main``, ``L_cur_mean``, ``L_ref_mean``,
        ``delta_abs_mean``, ``w_e_mean``, ``w_e_std``, ``w_e_min``, ``w_e_max`` (detached).
    """
    if int(advantages.shape[0]) != int(K):
        raise ValueError(
            f"advantages.shape[0]={advantages.shape[0]} must equal K={K}"
        )

    # Delta, M_e, w_e: no gradient into L_cur / L_ref / gate path
    Delta = L_cur_2d.detach() - L_ref_2d.detach()  # (K, B)
    M_e = (float(beta_dgpo) / float(K)) * (advantages * Delta).sum(dim=0)  # (B,)
    w_e = torch.sigmoid(M_e).detach()  # (B,)
    # Batch statistics for W&B ``parameter/w_e_*`` (per-event gate in [0, 1]).
    w_e_mean = w_e.mean()
    w_e_std = w_e.std(unbiased=False) if w_e.numel() > 1 else torch.zeros((), device=w_e.device, dtype=w_e.dtype)

    loss_main = (w_e.unsqueeze(0) * advantages * L_cur_2d).mean(dim=0).mean()
    loss_total = loss_main

    diag: dict[str, Tensor] = {
        "loss_total": loss_total.detach(),
        "loss_main": loss_main.detach(),
        "L_cur_mean": L_cur_2d.detach().mean(),
        "L_ref_mean": L_ref_2d.detach().mean(),
        "delta_abs_mean": (L_cur_2d - L_ref_2d).detach().abs().mean(),
        "w_e_mean": w_e_mean.detach(),
        "w_e_std": w_e_std.detach(),
        "w_e_min": w_e.min().detach(),
        "w_e_max": w_e.max().detach(),
    }
    return loss_total, diag


def build_reference_trust_loss(
    model_v: Tensor,
    ref_v: Tensor,
    noise_mask: Tensor,
    *,
    L_ref_2d: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Shared-noise velocity-MSE trust proxy against the active round reference.

    The policy and frozen reference are evaluated on the same ``(t, eps)`` draw,
    so gradients flow only through ``model_v`` while the installed OmniFold
    round reference remains the fixed denominator/velocity anchor.
    """
    if model_v.shape != ref_v.shape:
        raise ValueError(
            f"model_v {tuple(model_v.shape)} and ref_v {tuple(ref_v.shape)} must match"
        )
    mask = noise_mask.expand_as(model_v).to(dtype=model_v.dtype)
    denominator = mask.sum().clamp(min=1.0e-8)
    velocity_mse = (
        (model_v - ref_v.detach()).pow(2) * mask
    ).sum() / denominator
    trust_loss = 0.5 * velocity_mse
    diagnostics: dict[str, Tensor] = {
        "reference_trust/loss": trust_loss.detach(),
        "reference_trust/velocity_mse": velocity_mse.detach(),
    }
    if L_ref_2d is not None:
        reference_mean = L_ref_2d.detach().mean().clamp(min=1.0e-12)
        diagnostics["reference_trust/velocity_mse_ratio"] = (
            velocity_mse.detach() / reference_mean
        )
    return trust_loss, diagnostics


def predict_x0_normalized_from_velocity_diffusion(
    x_t: Tensor,
    v_pred: Tensor,
    t_rep: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reconstruct normalized clean neutrinos :math:`x_0` from velocity ``v``.

    Uses the same inversion as ``DDIMSampler.sample`` (velocity branch):
    ``eps_hat = alpha * v + sigma * x_t``, ``x_0 = (x_t - sigma * eps_hat) / alpha``.
    Scheduler ``alpha_t``, ``sigma_t`` match ``policy_evaluation_step`` /
    ``get_logsnr_alpha_sigma(time)``.
    """
    _, alpha, sigma = get_logsnr_alpha_sigma(t_rep, shape=(t_rep.shape[0], 1, 1))
    eps_hat = v_pred * alpha + x_t * sigma
    x0 = (x_t - sigma * eps_hat) / alpha.clamp(min=1e-8)
    return x0, alpha.view(-1), sigma.view(-1)
