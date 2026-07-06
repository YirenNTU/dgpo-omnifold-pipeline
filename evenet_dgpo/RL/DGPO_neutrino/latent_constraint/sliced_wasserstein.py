"""Sliced Wasserstein distance between two latent-space point clouds.

Reusable, GPU-friendly, differentiable utility for the DGPO latent constraint:

    C = sliced_wasserstein_distance(z_pred, z_truth, num_projections=64)

Design notes
------------
- ``z_pred`` keeps its autograd graph (gradient flows back to the DGPO policy
  through the predicted-neutrino features).  ``z_truth`` is used as-is; callers
  that want a fixed target should pass a detached tensor (see ``detach_truth``).
- Random projection directions are sampled on the input device.  For *stable*
  DGPO training the same directions can be reused every step by either passing a
  fixed ``projections`` tensor or a seeded ``generator`` (deterministic mode).
- Unequal batch sizes are supported via quantile matching; when the two clouds
  have the same number of rows this reduces *exactly* to the standard
  sort-and-difference 1-D Wasserstein estimator.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-12


def random_projections(
    num_projections: int,
    dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``num_projections`` unit directions in ``R^dim``.

    Args:
        num_projections: number of random 1-D projection directions ``L``.
        dim: latent dimension ``D``.
        device: device for the returned tensor.
        dtype: dtype for the returned tensor.
        generator: optional ``torch.Generator`` for deterministic sampling. It
            must live on the same device as ``device``.

    Returns:
        ``(L, D)`` tensor of unit-norm rows.
    """
    if num_projections < 1:
        raise ValueError(f"num_projections must be >= 1, got {num_projections}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    v = torch.randn(
        num_projections, dim, device=device, dtype=dtype, generator=generator
    )  # (L, D)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return v


def _sorted_quantiles(sorted_vals: Tensor, num_q: int) -> Tensor:
    """Linear-interpolated quantiles of an already-sorted tensor.

    Args:
        sorted_vals: ``(L, N)`` ascending along the last dim.
        num_q: number of quantile levels ``M``.

    Returns:
        ``(L, M)`` quantile values at levels ``(k + 0.5) / M``. Differentiable
        with respect to ``sorted_vals`` (gather + linear blend of values).
    """
    n = sorted_vals.shape[-1]
    if n == num_q:
        # Levels (k+0.5)/N map onto integer positions k -> identity gather, i.e.
        # the plain sort-and-difference estimator. Keep this exact fast path.
        return sorted_vals
    levels = (
        torch.arange(num_q, device=sorted_vals.device, dtype=sorted_vals.dtype) + 0.5
    ) / num_q  # (M,)
    pos = (levels * n - 0.5).clamp(0.0, float(n - 1))  # (M,)
    lo = pos.floor().long()
    hi = (lo + 1).clamp(max=n - 1)
    frac = (pos - lo.to(sorted_vals.dtype)).unsqueeze(0)  # (1, M)
    v_lo = sorted_vals.index_select(-1, lo)  # (L, M)
    v_hi = sorted_vals.index_select(-1, hi)  # (L, M)
    return v_lo + (v_hi - v_lo) * frac


def sliced_wasserstein_distance(
    z_pred: Tensor,
    z_truth: Tensor,
    num_projections: int = 64,
    *,
    p: int = 1,
    projections: Tensor | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    detach_truth: bool = False,
) -> Tensor:
    """Sliced Wasserstein-``p`` distance between two latent point clouds.

    Args:
        z_pred: ``(B, D)`` predicted-side latents (gradient-carrying in DGPO).
        z_truth: ``(B_t, D)`` truth-side latents. ``B_t`` may differ from ``B``.
        num_projections: number of random 1-D projections ``L`` (ignored when
            ``projections`` is given).
        p: Wasserstein order (``1`` or ``2``). ``p=1`` matches the repo's existing
            1-D Wasserstein convention.
        projections: optional fixed ``(L, D)`` projection directions. Pass this
            (or a seeded ``generator`` / ``seed``) for deterministic DGPO steps.
        generator: optional ``torch.Generator`` (same device as ``z_pred``).
        seed: optional integer seed; builds a deterministic per-call generator on
            ``z_pred``'s device. Ignored if ``generator`` or ``projections`` set.
        detach_truth: when True, detach ``z_truth`` inside the function so no
            gradient flows through the truth branch.

    Returns:
        Scalar tensor; differentiable with respect to ``z_pred``.
    """
    if z_pred.dim() != 2 or z_truth.dim() != 2:
        raise ValueError(
            f"expected 2-D [B, D] tensors, got {tuple(z_pred.shape)} and "
            f"{tuple(z_truth.shape)}"
        )
    if z_pred.shape[1] != z_truth.shape[1]:
        raise ValueError(
            f"latent_dim mismatch: z_pred D={z_pred.shape[1]} vs "
            f"z_truth D={z_truth.shape[1]}"
        )
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p}")

    n_pred = z_pred.shape[0]
    n_truth = z_truth.shape[0]
    if n_pred < 1 or n_truth < 1:
        return z_pred.new_zeros(())

    if detach_truth:
        z_truth = z_truth.detach()

    dim = z_pred.shape[1]
    if projections is None:
        if generator is None and seed is not None:
            generator = torch.Generator(device=z_pred.device)
            generator.manual_seed(int(seed))
        projections = random_projections(
            num_projections,
            dim,
            device=z_pred.device,
            dtype=z_pred.dtype,
            generator=generator,
        )  # (L, D)
    else:
        if projections.dim() != 2 or projections.shape[1] != dim:
            raise ValueError(
                f"projections must be (L, {dim}), got {tuple(projections.shape)}"
            )
        projections = projections.to(device=z_pred.device, dtype=z_pred.dtype)

    # Project onto each direction: (N, D) @ (D, L) -> (N, L) -> (L, N).
    proj_pred = (z_pred @ projections.t()).t()  # (L, B)
    proj_truth = (z_truth @ projections.t()).t()  # (L, B_t)

    # Sort projected values along the sample axis (differentiable gather).
    sort_pred = torch.sort(proj_pred, dim=-1).values  # (L, B)
    sort_truth = torch.sort(proj_truth, dim=-1).values  # (L, B_t)

    # Match sample counts via quantiles (identity when B == B_t).
    num_q = max(n_pred, n_truth)
    q_pred = _sorted_quantiles(sort_pred, num_q)  # (L, M)
    q_truth = _sorted_quantiles(sort_truth, num_q)  # (L, M)

    diff = (q_pred - q_truth).abs()  # (L, M)
    if p == 1:
        per_proj = diff.mean(dim=-1)  # (L,) == W1 per projection
        return per_proj.mean()
    # p == 2: SWD_2 = ( mean_proj W2^2 )^{1/2}, with W2^2 = mean_q diff^2.
    per_proj_pow = (diff ** 2).mean(dim=-1)  # (L,)
    return per_proj_pow.mean().clamp_min(0.0).sqrt()
