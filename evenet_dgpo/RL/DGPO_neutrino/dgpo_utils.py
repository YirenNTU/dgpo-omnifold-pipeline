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


def compute_per_event_advantage(
    rewards: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Per-event z-score advantages over candidates (dim 0).

    The frozen DGPO method uses z-score advantages only:
    ``advantages = (rewards - mean_K) / (std_K + eps)`` per event.

    Args:
        rewards: Shape ``(K, B)`` — K candidates, B events.
        eps: Added to std for numerical stability.

    Returns:
        ``(advantages, weights)`` each ``(K, B)``; ``weights = |advantages|``.
    """
    mu = rewards.mean(dim=0)
    std = rewards.std(dim=0, unbiased=False) + eps
    advantages = (rewards - mu.unsqueeze(0)) / std.unsqueeze(0)
    weights = advantages.abs()
    return advantages, weights

def repeat_batch_for_candidates(batch: dict[str, Any], K: int) -> dict[str, Any]:
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
