"""Linear CPO projection repair for DGPO with the latent-SWD constraint.

Enforcement is post-AdamW trust-region projection repair (Achiam 2017 style):
``lambda = v/(b.p)`` puts the constraint gradient ``b`` in the DENOMINATOR, so
the step targets the constraint VALUE and is invariant to coordinate
reparametrization (e.g. inv-CDF phi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import Module

from RL.DGPO_neutrino.dgpo_utils import _dgpo_cfg_get
from RL.DGPO_neutrino.latent_constraint.dgpo_constraint import (
    LatentSWDConfig,
    resolve_latent_swd_config,
)

_MULTI_SAMPLE_T_STRATIFIED_FIXED = "stratified_fixed"

# The only constraint backend producing the CPO scalar ``C``.
CONSTRAINT_LATENT_SWD = "latent_swd"


@dataclass(frozen=True)
class ProjectionConstraintConfig:
    """Resolved ``dgpo.projection_constraint`` block.

    The frozen pre-trained ``latent_swd`` encoder is the only constraint backend,
    enforced via post-AdamW CPO projection repair.
    """

    epsilon: float
    min_b_norm2: float
    max_lambda: float | None
    trust_region_ratio: float | None
    log_diagnostics: bool
    use_adam_preconditioner: bool
    damping: float
    multi_sample_count: int
    latent_swd: LatentSWDConfig
    type: str = CONSTRAINT_LATENT_SWD

    @property
    def active_apply_to(self) -> str:
        """``apply_to`` selector for the latent-SWD constraint."""
        return self.latent_swd.apply_to


def resolve_projection_constraint_config(dg_cfg: Any | None) -> ProjectionConstraintConfig:
    """Parse ``dgpo.projection_constraint`` from the DGPO config namespace."""
    block = _dgpo_cfg_get(dg_cfg, "projection_constraint", None) if dg_cfg is not None else None
    if block is None:
        latent_swd = resolve_latent_swd_config(None)
        return ProjectionConstraintConfig(
            epsilon=float(latent_swd.margin),
            min_b_norm2=1e-12,
            max_lambda=None,
            trust_region_ratio=1.0,
            log_diagnostics=True,
            use_adam_preconditioner=True,
            damping=0.0,
            multi_sample_count=8,
            latent_swd=latent_swd,
        )
    constraint_type = str(_dgpo_cfg_get(block, "type", CONSTRAINT_LATENT_SWD)).strip()
    if constraint_type != CONSTRAINT_LATENT_SWD:
        raise ValueError(
            "dgpo.projection_constraint.type must be 'latent_swd' (the only supported "
            f"constraint backend), got {constraint_type!r}"
        )
    latent_swd = resolve_latent_swd_config(_dgpo_cfg_get(block, "latent_swd", None))
    # epsilon (= CPO activation margin) defaults to the constraint's margin.
    epsilon = float(_dgpo_cfg_get(block, "epsilon", latent_swd.margin))
    min_b = float(_dgpo_cfg_get(block, "min_b_norm2", 1e-12))
    raw_max = _dgpo_cfg_get(block, "max_lambda", None)
    max_lambda = None if raw_max is None else float(raw_max)
    raw_trust = _dgpo_cfg_get(block, "trust_region_ratio", 1.0)
    trust_ratio = None if raw_trust is None else float(raw_trust)
    log_diag = bool(_dgpo_cfg_get(block, "log_diagnostics", True))
    use_adam_pre = bool(_dgpo_cfg_get(block, "use_adam_preconditioner", True))
    damping = float(_dgpo_cfg_get(block, "damping", 0.0))
    ms_block = _dgpo_cfg_get(block, "multi_sample", None)
    multi_sample_count = int(_dgpo_cfg_get(ms_block, "samples", 8))
    if not math.isfinite(epsilon):
        raise ValueError(f"dgpo.projection_constraint.epsilon must be finite, got {epsilon}")
    if not math.isfinite(min_b) or min_b <= 0.0:
        raise ValueError(
            f"dgpo.projection_constraint.min_b_norm2 must be finite and positive, got {min_b}"
        )
    if max_lambda is not None and (not math.isfinite(max_lambda) or max_lambda < 0.0):
        raise ValueError(
            f"dgpo.projection_constraint.max_lambda must be None or finite nonnegative, "
            f"got {max_lambda}"
        )
    if trust_ratio is not None and (not math.isfinite(trust_ratio) or trust_ratio <= 0.0):
        raise ValueError(
            "dgpo.projection_constraint.trust_region_ratio must be None or finite positive, "
            f"got {trust_ratio}"
        )
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError(
            f"dgpo.projection_constraint.damping must be finite and nonnegative, got {damping}"
        )
    if multi_sample_count < 1:
        raise ValueError(
            "dgpo.projection_constraint.multi_sample.samples must be >= 1, "
            f"got {multi_sample_count}"
        )
    return ProjectionConstraintConfig(
        epsilon=epsilon,
        min_b_norm2=min_b,
        max_lambda=max_lambda,
        trust_region_ratio=trust_ratio,
        log_diagnostics=log_diag,
        use_adam_preconditioner=use_adam_pre,
        damping=damping,
        multi_sample_count=multi_sample_count,
        latent_swd=latent_swd,
        type=constraint_type,
    )


def projection_stratified_t_grid(
    ms_count: int,
    batch_size: int,
    *,
    t_min: float,
    t_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Tensor]:
    """Midpoint-stratified fixed ``t`` values in ``[t_min, t_max]`` for projection multi-sample."""
    if ms_count < 1:
        raise ValueError(f"ms_count must be >= 1, got {ms_count}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    t_lo = float(t_min)
    t_hi = float(t_max)
    if t_hi < t_lo:
        raise ValueError(f"t_max must be >= t_min, got t_min={t_lo}, t_max={t_hi}")
    span = t_hi - t_lo
    if span <= 0.0:
        t_flat = torch.full((batch_size,), t_lo, device=device, dtype=dtype)
        return [t_flat for _ in range(ms_count)]
    idx = torch.arange(ms_count, device=device, dtype=dtype)
    t_scalar = t_lo + (idx + 0.5) / float(ms_count) * span
    return [t_scalar[i].expand(batch_size).contiguous() for i in range(ms_count)]


def snapshot_params(model: Module) -> dict[str, Tensor]:
    """Detached clones of every ``requires_grad`` parameter, keyed by name."""
    return {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def trainable_params_all_finite(model: Module) -> bool:
    """True when every trainable parameter is finite."""
    for p in model.parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            return False
    return True


def assign_params_(model: Module, snap: Mapping[str, Tensor]) -> None:
    """Copy snapshot tensors into ``p.data``; does not touch optimizer state."""
    for name, p in model.named_parameters():
        if p.requires_grad and name in snap:
            p.data.copy_(snap[name])


def flatten_param_delta(
    theta_adam: Mapping[str, Tensor],
    theta_old: Mapping[str, Tensor],
) -> Tensor:
    """Flat concatenation of ``theta_adam - theta_old`` in ``theta_old.keys()`` order."""
    chunks: list[Tensor] = []
    for name in theta_old.keys():
        chunks.append((theta_adam[name] - theta_old[name]).reshape(-1))
    return torch.cat(chunks)


def flatten_param_grads(model: Module) -> Tensor:
    """Flat concatenation of parameter gradients (zeros if ``grad is None``)."""
    chunks: list[Tensor] = []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            chunks.append(torch.zeros_like(p.data, memory_format=torch.contiguous_format).reshape(-1))
        else:
            chunks.append(p.grad.detach().reshape(-1))
    return torch.cat(chunks)


def _resolve_torch_optimizer(optimizer: Any) -> Any:
    """Return the underlying ``torch.optim.Optimizer`` (unwraps DGPO schedule bundle)."""
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and hasattr(inner, "state") and hasattr(inner, "param_groups"):
        return inner
    return optimizer


def _adam_param_precond_meta(
    optimizer: Any,
) -> dict[int, tuple[dict[str, Any] | None, float, float]]:
    """Map ``id(param) -> (state, eps, beta2)`` using tensor identity, not value equality."""
    opt = _resolve_torch_optimizer(optimizer)
    out: dict[int, tuple[dict[str, Any] | None, float, float]] = {}
    for group in opt.param_groups:
        eps_adam = float(group.get("eps", 1e-8))
        betas = group.get("betas", (0.9, 0.999))
        beta2 = float(betas[1]) if len(betas) > 1 else 0.999
        for gp in group["params"]:
            out[id(gp)] = (opt.state.get(gp), eps_adam, beta2)
    return out


def flatten_adam_preconditioned_direction(
    model: Module,
    b_flat: Tensor,
    optimizer: Any,
    *,
    use_adam_preconditioner: bool,
) -> tuple[Tensor, dict[str, float]]:
    """Build flat ``p = M^{-1} b`` in the same order as :func:`flatten_param_grads`."""
    if not use_adam_preconditioner:
        return b_flat, {
            "projection/adam_preconditioner_active": 0.0,
            "projection/adam_preconditioner_missing_state_count": 0.0,
        }

    precond_meta = _adam_param_precond_meta(optimizer)
    p_chunks: list[Tensor] = []
    missing_count = 0
    idx = 0
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        b_chunk = b_flat[idx : idx + n].view_as(p.data)
        idx += n
        state, eps_adam, beta2 = precond_meta.get(id(p), (None, 1e-8, 0.999))
        if state is None or "exp_avg_sq" not in state:
            missing_count += 1
            p_chunks.append(b_chunk.reshape(-1))
            continue
        step_raw = state.get("step", 0)
        step = int(step_raw.item()) if isinstance(step_raw, Tensor) else int(step_raw)
        exp_avg_sq = state["exp_avg_sq"]
        if step > 0:
            bias_correction = 1.0 - beta2 ** step
            s_hat = exp_avg_sq / bias_correction
        else:
            s_hat = exp_avg_sq
        denom = torch.sqrt(s_hat) + eps_adam
        p_chunk = b_chunk / denom.to(device=b_chunk.device, dtype=b_chunk.dtype)
        p_chunks.append(p_chunk.reshape(-1))

    p_flat = torch.cat(p_chunks)
    return p_flat, {
        "projection/adam_preconditioner_active": 1.0,
        "projection/adam_preconditioner_missing_state_count": float(missing_count),
    }


def _adamw_metric_denom_for_param(
    state: dict[str, Any] | None,
    eps_adam: float,
    beta2: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Per-element AdamW metric denominator ``sqrt(v_hat) + eps`` matching ``p = b / denom``."""
    if state is None or "exp_avg_sq" not in state:
        return None
    step_raw = state.get("step", 0)
    step = int(step_raw.item()) if isinstance(step_raw, Tensor) else int(step_raw)
    exp_avg_sq = state["exp_avg_sq"]
    if step > 0:
        bias_correction = 1.0 - beta2 ** step
        s_hat = exp_avg_sq / bias_correction
    else:
        s_hat = exp_avg_sq
    return (torch.sqrt(s_hat) + eps_adam).to(device=device, dtype=dtype)


def adamw_metric_norm(
    model: Module,
    delta_flat: Tensor,
    optimizer: Any,
    *,
    use_adam_preconditioner: bool,
) -> float:
    """``||delta||_M`` with diagonal ``M_ii = denom_i^2`` from AdamW ``exp_avg_sq``."""
    sq = adamw_metric_squared_norm(model, delta_flat, optimizer, use_adam_preconditioner=use_adam_preconditioner)
    return math.sqrt(max(0.0, sq))


def adamw_metric_squared_norm(
    model: Module,
    delta_flat: Tensor,
    optimizer: Any,
    *,
    use_adam_preconditioner: bool,
) -> float:
    """``delta^T M delta`` with the AdamW diagonal metric."""
    if not use_adam_preconditioner:
        d = delta_flat.to(dtype=torch.float64)
        return float(torch.dot(d, d).detach().cpu())

    precond_meta = _adam_param_precond_meta(optimizer)
    total = 0.0
    idx = 0
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        chunk = delta_flat[idx : idx + n].view_as(p.data)
        idx += n
        state, eps_adam, beta2 = precond_meta.get(id(p), (None, 1e-8, 0.999))
        denom = _adamw_metric_denom_for_param(state, eps_adam, beta2, chunk.device, chunk.dtype)
        if denom is None:
            total += float((chunk.detach().float() ** 2).sum().cpu())
            continue
        weighted = chunk * denom
        total += float((weighted.detach().float() ** 2).sum().cpu())
    return total


def compute_cpo_adamw_final_update(
    model: Module,
    delta0_flat: Tensor,
    p_flat: Tensor,
    lambda_star: float,
    optimizer: Any,
    cfg: ProjectionConstraintConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Full projection correction, then AdamW-metric cap on the composed final update."""
    lam = max(0.0, float(lambda_star))
    repair_flat = lam * p_flat
    delta_full = delta0_flat - repair_flat
    use_adam = bool(cfg.use_adam_preconditioner)
    delta0_norm_m = adamw_metric_norm(model, delta0_flat, optimizer, use_adam_preconditioner=use_adam)
    delta_full_norm_m = adamw_metric_norm(model, delta_full, optimizer, use_adam_preconditioner=use_adam)

    if cfg.trust_region_ratio is None:
        trust_radius_m = float("inf")
        final_scale = 1.0
    else:
        trust_radius_m = float(cfg.trust_region_ratio) * max(delta0_norm_m, 1e-12)
        if delta_full_norm_m <= trust_radius_m or delta_full_norm_m <= 0.0:
            final_scale = 1.0
        else:
            final_scale = trust_radius_m / delta_full_norm_m

    delta_final = final_scale * delta_full
    delta_final_norm_m = final_scale * delta_full_norm_m
    repair_norm = float(torch.linalg.norm(repair_flat.to(dtype=torch.float64)).detach().cpu())
    if math.isfinite(trust_radius_m) and trust_radius_m > 0.0:
        raw_norm_to_radius = delta_full_norm_m / trust_radius_m
        final_norm_to_radius = delta_final_norm_m / trust_radius_m
        cap_margin_m = trust_radius_m - delta_final_norm_m
    else:
        raw_norm_to_radius = float("nan")
        final_norm_to_radius = float("nan")
        cap_margin_m = float("nan")
    raw_norm_to_delta0 = delta_full_norm_m / delta0_norm_m if delta0_norm_m > 0.0 else float("nan")
    final_norm_to_delta0 = delta_final_norm_m / delta0_norm_m if delta0_norm_m > 0.0 else float("nan")
    diag: dict[str, float] = {
        "projection/cpo_trial/final_update_cap": 1.0,
        "projection/cpo_trial/delta0_norm_adamw": float(delta0_norm_m),
        "projection/cpo_trial/final_update_norm_adamw_raw": float(delta_full_norm_m),
        "projection/cpo_trial/final_update_norm_adamw": float(delta_final_norm_m),
        "projection/cpo_trial/trust_radius_adamw": float(trust_radius_m),
        "projection/cpo_trial/raw_norm_to_radius": float(raw_norm_to_radius),
        "projection/cpo_trial/final_norm_to_radius": float(final_norm_to_radius),
        "projection/cpo_trial/cap_margin_adamw": float(cap_margin_m),
        "projection/cpo_trial/raw_norm_to_delta0_adamw": float(raw_norm_to_delta0),
        "projection/cpo_trial/final_norm_to_delta0_adamw": float(final_norm_to_delta0),
        "projection/cpo_trial/final_update_scale": float(final_scale),
        "projection/cpo_trial/final_update_cap_active": (
            1.0 if final_scale < 1.0 - 1e-12 else 0.0
        ),
        "projection/correction_norm_requested": repair_norm,
    }
    return delta_final, diag


def assign_params_from_theta_old_delta_(
    model: Module,
    theta_old: Mapping[str, Tensor],
    delta_final_flat: Tensor,
) -> float:
    """Write ``theta_old + delta_final`` into trainable parameters; return ``||delta_final||_2``."""
    idx = 0
    norm_sq = 0.0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = int(p.numel())
        chunk = delta_final_flat[idx : idx + n].view_as(p.data).to(device=p.device, dtype=p.dtype)
        idx += n
        old = theta_old[name].to(device=p.device, dtype=p.dtype)
        p.data.copy_(old + chunk)
        norm_sq += float((chunk.detach().float() ** 2).sum().cpu())
    return math.sqrt(norm_sq)


def compute_projection_lambda_from_violation(
    b_flat: Tensor,
    p_flat: Tensor,
    violation: float,
    cfg: ProjectionConstraintConfig,
    *,
    C_raw: float | None = None,
    b_dot_delta0: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Closed-form AdamW-metric projection multiplier from an explicit violation scalar."""
    v = float(violation)
    b_d = b_flat.to(dtype=torch.float64)
    p_d = p_flat.to(dtype=torch.float64)
    b_sq = float(torch.dot(b_d, b_d).detach().cpu())
    b_dot_p = float(torch.dot(b_d, p_d).detach().cpu())
    b_norm = math.sqrt(b_sq)
    p_norm = float(torch.linalg.norm(p_d).detach().cpu())
    b_max_abs = float(b_d.abs().max().detach().cpu())
    p_max_abs = float(p_d.abs().max().detach().cpu())
    denom = b_dot_p + float(cfg.damping)
    lam = 0.0
    applied = 0.0
    if b_sq >= float(cfg.min_b_norm2) and math.isfinite(v):
        if v > 0.0 and math.isfinite(denom) and denom > 0.0:
            lam = v / denom
            applied = 1.0
    if cfg.max_lambda is not None:
        lam = min(lam, float(cfg.max_lambda))
    if not math.isfinite(lam):
        lam = 0.0
        applied = 0.0
    diag: dict[str, float] = {
        "projection/b_norm2": float(b_sq),
        "projection/b_norm": float(b_norm),
        "projection/b_max_abs": float(b_max_abs),
        "projection/p_norm": float(p_norm),
        "projection/p_max_abs": float(p_max_abs),
        "projection/b_dot_p": float(b_dot_p),
        "projection/v": float(v),
        "projection/v_selected": float(v),
        "projection/lambda": float(lam),
        "projection/applied": float(applied),
    }
    if C_raw is not None:
        c_margin = float(C_raw) - float(cfg.epsilon)
        diag["projection/C_raw"] = float(C_raw)
        diag["projection/c_margin"] = float(c_margin)
    if b_dot_delta0 is not None:
        diag["projection/b_dot_delta0"] = float(b_dot_delta0)
    if not cfg.log_diagnostics:
        diag = {k: val for k, val in diag.items() if k in ("projection/lambda", "projection/applied")}
    return lam, diag


def compute_projection_lambda(
    b_flat: Tensor,
    p_flat: Tensor,
    C_raw: float,
    delta0_flat: Tensor,
    cfg: ProjectionConstraintConfig,
) -> tuple[float, dict[str, float]]:
    """Closed-form AdamW-metric projection multiplier."""
    c_margin = float(C_raw) - float(cfg.epsilon)
    b_d = b_flat.to(dtype=torch.float64)
    d0 = delta0_flat.to(dtype=torch.float64)
    b_dot_d0 = float(torch.dot(b_d, d0).detach().cpu())
    v = c_margin + b_dot_d0
    lam, diag = compute_projection_lambda_from_violation(
        b_flat,
        p_flat,
        v,
        cfg,
        C_raw=C_raw,
        b_dot_delta0=b_dot_d0,
    )
    return lam, diag
