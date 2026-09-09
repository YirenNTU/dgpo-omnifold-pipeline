"""Pure utilities for DGPO neutrino RL: advantages, batch tiling, and DGPO loss."""

from __future__ import annotations

import math
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

REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE = "velocity_mse"
REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL = "vp_path_kl"
VALID_REFERENCE_TRUST_OBJECTIVES = frozenset(
    {
        REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
        REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
    }
)


def _scalar_softplus(value: float) -> float:
    """Numerically stable scalar softplus used by the cosine-time sampler."""

    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def sample_cosine_vp_path_kl_timesteps(
    count: int,
    batch_size: int,
    *,
    total_strata: int | None = None,
    stratum_offset: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    logsnr_min: float = -20.0,
    logsnr_max: float = 20.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, float]:
    """Sample stratified cosine-schedule times for a VP path-KL estimate.

    For the cosine schedule used by EveNet, write
    ``alpha(t)=cos(a*t+b)`` and ``sigma(t)=sin(a*t+b)``.  The v-space
    reverse-SDE path-KL integrand has time weight
    ``w(t)=beta(t)*alpha(t)^2/sigma(t)^2 = 2*a*cot(a*t+b)``.  This function
    samples from ``rho(t)=w(t)/Z`` by inverse CDF, so callers only need the
    returned normalization ``Z`` rather than a high-variance per-time weight.

    ``count`` consecutive strata starting at ``stratum_offset`` are returned
    as a ``(count, batch_size)`` tensor.  Splitting one logical draw across
    calls is supported by keeping ``total_strata`` fixed and advancing the
    offset.
    """

    count_i = int(count)
    batch_i = int(batch_size)
    strata_i = count_i if total_strata is None else int(total_strata)
    offset_i = int(stratum_offset)
    if count_i < 1:
        raise ValueError("count must be positive")
    if batch_i < 1:
        raise ValueError("batch_size must be positive")
    if strata_i < 1:
        raise ValueError("total_strata must be positive")
    if offset_i < 0 or offset_i + count_i > strata_i:
        raise ValueError(
            "stratum_offset and count must select strata within total_strata"
        )
    if (
        not isinstance(dtype, torch.dtype)
        or not torch.empty((), dtype=dtype).is_floating_point()
    ):
        raise TypeError("dtype must be a real floating-point torch dtype")

    logsnr_min_f = float(logsnr_min)
    logsnr_max_f = float(logsnr_max)
    if not math.isfinite(logsnr_min_f) or not math.isfinite(logsnr_max_f):
        raise ValueError("logsnr endpoints must be finite")
    if logsnr_min_f >= logsnr_max_f:
        raise ValueError("logsnr_min must be smaller than logsnr_max")

    # log(sigma^2) = -softplus(logSNR).  Its endpoint difference is exactly
    # integral_0^1 beta(t)*alpha(t)^2/sigma(t)^2 dt for this schedule.
    log_sigma_sq_start = -_scalar_softplus(logsnr_max_f)
    log_sigma_sq_end = -_scalar_softplus(logsnr_min_f)
    normalization = log_sigma_sq_end - log_sigma_sq_start
    if not math.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("cosine VP path-KL normalization must be finite and positive")

    # Compute the inverse CDF in float64 to avoid rounding the default
    # high-logSNR endpoint to exactly t=0 before the final dtype conversion.
    jitter = torch.rand(
        (count_i, batch_i),
        device=device,
        dtype=torch.float64,
        generator=generator,
    )
    stratum = torch.arange(
        offset_i,
        offset_i + count_i,
        device=device,
        dtype=torch.float64,
    ).unsqueeze(1)
    quantile = (stratum + jitter) / float(strata_i)
    log_sigma_sq = log_sigma_sq_start + quantile * normalization
    sin_angle = torch.exp(0.5 * log_sigma_sq).clamp(min=0.0, max=1.0)

    angle_start = math.atan(math.exp(-0.5 * logsnr_max_f))
    angle_end = math.atan(math.exp(-0.5 * logsnr_min_f))
    time = (torch.asin(sin_angle) - angle_start) / (angle_end - angle_start)
    return time.clamp_(0.0, 1.0).to(dtype=dtype), float(normalization)


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
    objective: str = REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
    path_kl_normalizer: float | Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Shared-noise trust objective against the active round reference.

    The policy and frozen reference are evaluated on the same ``(t, eps)`` draw,
    so gradients flow only through ``model_v`` while the installed OmniFold
    round reference remains the fixed denominator/velocity anchor.  The legacy
    ``velocity_mse`` objective remains the default.  ``vp_path_kl`` expects
    times drawn by :func:`sample_cosine_vp_path_kl_timesteps` and computes
    ``0.5 * Z * mean_rows(sum_active_dims((model_v-ref_v)^2))``.
    """
    if model_v.shape != ref_v.shape:
        raise ValueError(
            f"model_v {tuple(model_v.shape)} and ref_v {tuple(ref_v.shape)} must match"
        )
    if model_v.ndim < 2:
        raise ValueError("model_v and ref_v must have a row and feature dimension")
    if objective not in VALID_REFERENCE_TRUST_OBJECTIVES:
        raise ValueError(
            f"unsupported reference trust objective: {objective!r}; "
            f"expected one of {sorted(VALID_REFERENCE_TRUST_OBJECTIVES)}"
        )
    try:
        mask = noise_mask.expand_as(model_v).to(
            device=model_v.device,
            dtype=model_v.dtype,
        )
    except RuntimeError as exc:
        raise ValueError(
            f"noise_mask {tuple(noise_mask.shape)} cannot expand to "
            f"model_v {tuple(model_v.shape)}"
        ) from exc
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("noise_mask must contain finite, non-negative values")

    squared_difference = (model_v - ref_v.detach()).pow(2) * mask
    denominator = mask.sum().clamp(min=1.0e-8)
    velocity_mse = squared_difference.sum() / denominator

    vp_path_kl: Tensor | None = None
    if path_kl_normalizer is not None:
        normalizer = torch.as_tensor(
            path_kl_normalizer,
            device=model_v.device,
            dtype=model_v.dtype,
        ).detach()
        if normalizer.numel() != 1:
            raise ValueError("path_kl_normalizer must be a scalar")
        normalizer = normalizer.reshape(())
        if not torch.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("path_kl_normalizer must be finite and positive")
        row_sum = squared_difference.reshape(model_v.shape[0], -1).sum(dim=1)
        vp_path_kl = 0.5 * normalizer * row_sum.mean()
    elif objective == REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL:
        raise ValueError("vp_path_kl objective requires path_kl_normalizer")

    trust_loss = (
        vp_path_kl
        if objective == REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL
        else 0.5 * velocity_mse
    )
    diagnostics: dict[str, Tensor] = {
        "reference_trust/loss": trust_loss.detach(),
        "reference_trust/velocity_mse": velocity_mse.detach(),
    }
    if vp_path_kl is not None:
        diagnostics["reference_trust/vp_path_kl"] = vp_path_kl.detach()
        diagnostics["reference_trust/vp_path_kl_normalizer"] = (
            normalizer.detach()
        )
    if L_ref_2d is not None:
        reference_mean = L_ref_2d.detach().mean().clamp(min=1.0e-12)
        diagnostics["reference_trust/velocity_mse_ratio"] = (
            velocity_mse.detach() / reference_mean
        )
    return trust_loss, diagnostics


def adaptive_trust_update_scale(
    distance: float,
    *,
    delta: float,
    warning_fraction: float,
) -> tuple[float, bool]:
    """Return a linear near-boundary step scale and a fail-closed stop flag."""

    distance_f = float(distance)
    delta_f = float(delta)
    warning_f = float(warning_fraction)
    if not math.isfinite(delta_f) or delta_f <= 0.0:
        raise ValueError("adaptive trust delta must be finite and positive")
    if not 0.0 < warning_f < 1.0:
        raise ValueError("adaptive trust warning_fraction must lie in (0, 1)")
    if not math.isfinite(distance_f):
        return 0.0, True
    if distance_f >= delta_f:
        return 0.0, True
    warning = warning_f * delta_f
    if distance_f <= warning:
        return 1.0, False
    scale = (delta_f - distance_f) / (delta_f - warning)
    return min(max(float(scale), 0.0), 1.0), False


def adaptive_trust_backtracking_scales(
    initial_scale: float,
    *,
    factor: float,
    max_backtracks: int,
) -> tuple[float, ...]:
    """Return strictly decreasing absolute step scales for trust backtracking."""

    scale = float(initial_scale)
    factor_f = float(factor)
    count = int(max_backtracks)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("adaptive trust initial_scale must lie in (0, 1]")
    if not math.isfinite(factor_f) or not 0.0 < factor_f < 1.0:
        raise ValueError("adaptive trust backtrack factor must lie in (0, 1)")
    if count < 1:
        raise ValueError("adaptive trust max_backtracks must be positive")
    return tuple(scale * factor_f**index for index in range(1, count + 1))


def reset_adam_first_moment(optimizer: Any) -> int:
    """Zero Adam/AdamW ``exp_avg`` buffers without changing variance or LR state."""

    state = getattr(optimizer, "state", None)
    if state is None:
        inner = getattr(optimizer, "optimizer", None)
        state = getattr(inner, "state", None)
    if state is None:
        raise TypeError("optimizer does not expose an Adam-style state mapping")
    reset = 0
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping):
            continue
        first_moment = parameter_state.get("exp_avg")
        if isinstance(first_moment, Tensor):
            first_moment.zero_()
            reset += 1
    return reset


@torch.no_grad()
def rebase_trainable_params_for_extragradient_(
    model: torch.nn.Module,
    base_params: Mapping[str, Tensor],
) -> int:
    """Move parameters to the extragradient base without touching gradients.

    The caller first evaluates/backpropagates the vector field at a virtual
    look-ahead model.  Rebasing immediately before ``optimizer.step()`` makes
    that look-ahead gradient update the original incumbent, which is the
    defining predictive-corrective operation of extragradient.  Gradients and
    optimizer state intentionally remain unchanged.
    """

    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(set(trainable) - set(base_params))
    extra = sorted(set(base_params) - set(trainable))
    if missing or extra:
        raise KeyError(
            "extragradient base parameter keys differ from the live model: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    for name, parameter in trainable.items():
        base = base_params[name]
        if tuple(base.shape) != tuple(parameter.shape):
            raise ValueError(
                "extragradient base parameter shape differs for "
                f"{name!r}: {tuple(base.shape)} vs {tuple(parameter.shape)}"
            )
        if not torch.isfinite(base).all():
            raise FloatingPointError(
                f"extragradient base parameter {name!r} is non-finite"
            )
        parameter.copy_(base.to(device=parameter.device, dtype=parameter.dtype))
    return len(trainable)


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
