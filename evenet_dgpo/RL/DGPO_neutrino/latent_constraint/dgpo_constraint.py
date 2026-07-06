"""Latent-SWD projection constraint for DGPO.

The **only** projection-constraint provider: a **frozen, independently-trained**
latent-space encoder (the object-token bottleneck AE) + sliced Wasserstein
distance. The public entry point is :func:`compute_latent_swd_constraint`:

    C_norm, diag = compute_latent_swd_constraint(
        model_v=..., x_t=..., t_rep=..., noise_mask_rep=...,
        batch_kb=..., core_model=..., cartesian=..., K=..., state=...,
        candidate_weights_kb=..., update_ema=..., world_size=...,
    )

``C_norm`` is a scalar, **differentiable w.r.t. ``model_v``** through the
predicted neutrinos, in ratio-normalized null-excess form (what the post-AdamW
CPO repair consumes):

    C_norm = (SWD(z_pred, z_truth) - SWD_tt) / (SWD_tt + eps)

where ``SWD_tt`` is the truth/truth bootstrap SWD within the batch (the null floor).

Design notes:
- The encoder is **frozen** (pre-trained). There is no on-policy retrain /
  finetune; ``update_ema`` is accepted for signature compatibility but unused
  (the null floor is recomputed per batch).
- Normalization must match the policy: the constraint hands physical
  ``(log_pt, eta, phi)`` neutrinos and the encoder normalizes them the EveNet
  way internally (inv-CDF phi), so build the encoder from the **same
  ``normalization.pt``** the policy uses.
- The encoder additionally conditions on the frozen EveNet ``event_token`` and
  ``object_token`` columns, which must be present in the DGPO batch (train on
  the *_evttok augmented parquet from ``preprocessing/augment_event_token.py
  --object-tokens``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from RL.DGPO_neutrino.latent_constraint.object_token_ae import (
    ObjectTokenBottleneckAutoencoder,
    load_checkpoint,
)
from RL.DGPO_neutrino.latent_constraint.sliced_wasserstein import (
    random_projections,
    sliced_wasserstein_distance,
)

_log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# distributed + batch helpers (shared with the DGPO trainer)
# ----------------------------------------------------------------------
def sync_projection_constraint_C_across_ranks(
    C_scalar: float,
    *,
    device: torch.device,
    world_size: int,
) -> float:
    """All-reduce average the scalar constraint value ``C`` so every rank gets the same lambda.

    Only the scalar is synced here. The constraint gradient ``b = nabla C`` is already
    averaged by DDP during the constraint ``backward()`` — the policy forward runs through
    the DDP wrapper (``model(...)``) without ``no_sync``, so DDP's reducer all-reduces the
    gradients. Averaging ``b`` again here would shrink it by an extra factor of
    ``world_size``. ``C`` is a plain Python float that DDP never touches, so it must be
    averaged explicitly; combined with the DDP-averaged ``b`` and the rank-identical
    ``theta_old``/``theta_adam``, every rank then computes an identical ``lambda`` and
    applies a bit-for-bit identical CPO repair.
    """
    if world_size <= 1:
        return float(C_scalar)
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return float(C_scalar)
    t = torch.tensor([float(C_scalar)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / float(world_size)).cpu())


def _event_mask_kb(
    batch_kb: Mapping[str, Any],
    *,
    K: int,
    KB: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """``(KB,)`` boolean event validity expanded over candidates."""
    from RL.DGPO_neutrino.rewards import get_event_valid_mask

    B = KB // max(int(K), 1)
    if "x" not in batch_kb:
        return torch.ones(KB, device=device, dtype=torch.bool)
    vb = get_event_valid_mask(batch_kb, B, device, dtype) > 0
    return vb.unsqueeze(0).expand(int(K), -1).reshape(KB)


def _candidate_mask_kb(
    candidate_weights_kb: Tensor | None,
    *,
    K: int,
    B: int,
    device: torch.device,
    apply_to: str,
) -> Tensor:
    """``(KB,)`` candidate selector for ``apply_to``."""
    KB = int(K) * int(B)
    if apply_to == "best_candidate":
        if candidate_weights_kb is None:
            raise ValueError("apply_to=best_candidate requires candidate_weights_kb")
        if tuple(candidate_weights_kb.shape) != (int(K), int(B)):
            raise ValueError(
                f"candidate_weights_kb shape {tuple(candidate_weights_kb.shape)} "
                f"!= (K, B)=({K}, {B})"
            )
        return candidate_weights_kb.to(device=device).bool().reshape(KB)
    return torch.ones(KB, device=device, dtype=torch.bool)


def _pred_truth_kin_log_pt_eta_phi(
    nu_phys_kb: Tensor,
    batch_kb: Mapping[str, Any],
    *,
    cartesian: bool,
) -> tuple[Tensor, Tensor]:
    """``pred_kin``, ``truth_kin`` each ``(KB, 2, 3)`` with channels ``[log_pt, eta, phi]``."""
    from RL.DGPO_neutrino.rewards import cartesian_to_log_pt_eta_phi

    KB = nu_phys_kb.shape[0]
    if cartesian:
        pred_log_pt, pred_eta, pred_phi = cartesian_to_log_pt_eta_phi(
            nu_phys_kb[:, :2, 0],
            nu_phys_kb[:, :2, 1],
            nu_phys_kb[:, :2, 2],
        )
        if not isinstance(batch_kb.get("x_invisible_cartesian"), Tensor):
            raise KeyError("latent_swd constraint cartesian=True requires x_invisible_cartesian")
        truth_xyz = batch_kb["x_invisible_cartesian"].to(
            device=nu_phys_kb.device, dtype=nu_phys_kb.dtype
        )[:, :2, :]
        truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
            truth_xyz[..., 0],
            truth_xyz[..., 1],
            truth_xyz[..., 2],
        )
    else:
        pred_log_pt = nu_phys_kb[:, :2, 0]
        pred_eta = nu_phys_kb[:, :2, 1]
        pred_phi = nu_phys_kb[:, :2, 2]
        if not isinstance(batch_kb.get("x_invisible"), Tensor):
            raise KeyError("latent_swd constraint requires batch['x_invisible']")
        truth = batch_kb["x_invisible"].to(device=nu_phys_kb.device, dtype=nu_phys_kb.dtype)[:, :2, :]
        truth_log_pt = truth[..., 0]
        truth_eta = truth[..., 1]
        truth_phi = truth[..., 2]
    pred_kin = torch.stack((pred_log_pt, pred_eta, pred_phi), dim=-1)
    truth_kin = torch.stack((truth_log_pt, truth_eta, truth_phi), dim=-1)
    if tuple(pred_kin.shape) != (KB, 2, 3) or tuple(truth_kin.shape) != (KB, 2, 3):
        raise ValueError(
            f"expected kin shape (KB, 2, 3), got {tuple(pred_kin.shape)} vs {tuple(truth_kin.shape)}"
        )
    return pred_kin, truth_kin


# ----------------------------------------------------------------------
# config + state
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LatentSWDConfig:
    """Resolved ``dgpo.projection_constraint.latent_swd`` block."""

    enabled: bool
    checkpoint_file: str          # frozen latent-constraint checkpoint (.ckpt)
    normalization_file: str       # optional override of the stored normalization.pt
    margin: float                 # CPO activates when C_norm > margin
    eps: float                    # ratio denominator floor
    min_samples: int              # skip the batch below this many valid rows
    apply_to: str                 # "all_candidates" | "best_candidate"
    num_projections: int          # SWD random projection axes (averaged)


def resolve_latent_swd_config(block: Any | None) -> LatentSWDConfig:
    """Parse the ``latent_swd`` projection-constraint block."""
    from RL.DGPO_neutrino.dgpo_utils import _dgpo_cfg_get

    defaults = LatentSWDConfig(
        enabled=True,
        checkpoint_file="",
        normalization_file="",
        margin=1.0,
        eps=1e-6,
        min_samples=8,
        apply_to="all_candidates",
        num_projections=128,
    )
    if block is None:
        return defaults
    margin = float(_dgpo_cfg_get(block, "margin", defaults.margin))
    eps = float(_dgpo_cfg_get(block, "eps", defaults.eps))
    min_samples = int(_dgpo_cfg_get(block, "min_samples", defaults.min_samples))
    num_projections = int(_dgpo_cfg_get(block, "num_projections", defaults.num_projections))
    if not math.isfinite(margin):
        raise ValueError(f"latent_swd.margin must be finite, got {margin}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"latent_swd.eps must be positive, got {eps}")
    if min_samples < 2:
        raise ValueError(f"latent_swd.min_samples must be >= 2, got {min_samples}")
    if num_projections < 1:
        raise ValueError(f"latent_swd.num_projections must be >= 1, got {num_projections}")
    return LatentSWDConfig(
        enabled=bool(_dgpo_cfg_get(block, "enabled", defaults.enabled)),
        checkpoint_file=str(_dgpo_cfg_get(block, "checkpoint_file", defaults.checkpoint_file)).strip(),
        normalization_file=str(
            _dgpo_cfg_get(block, "normalization_file", defaults.normalization_file)
        ).strip(),
        margin=margin,
        eps=eps,
        min_samples=min_samples,
        apply_to=str(_dgpo_cfg_get(block, "apply_to", defaults.apply_to)).strip(),
        num_projections=num_projections,
    )


@dataclass
class LatentSWDState:
    """Runtime state: a frozen latent-constraint encoder + its config.

    Carries no optimizer/EMA: the encoder never trains on-policy.
    """

    model: ObjectTokenBottleneckAutoencoder
    cfg: LatentSWDConfig

    def freeze(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def checkpoint_payload(self) -> dict[str, Any]:
        """Minimal payload for DGPO checkpoints.

        The encoder is frozen and lives in its own ``checkpoint_file``; we only
        record provenance so a resumed DGPO run can rebuild the same state.
        """
        return {
            "constraint_type": "latent_swd",
            "checkpoint_file": self.cfg.checkpoint_file,
            "normalization_file": self.cfg.normalization_file,
        }


def init_latent_swd_state(
    cfg: LatentSWDConfig,
    *,
    device: torch.device,
    resume_payload: Mapping[str, Any] | None = None,
) -> LatentSWDState:
    """Build a frozen latent-constraint state from a standalone checkpoint.

    ``resume_payload`` (DGPO checkpoint) is honored only to recover the encoder
    checkpoint / normalization paths; the weights themselves come from that file.
    """
    ckpt_file = str(cfg.checkpoint_file).strip()
    norm_file = cfg.normalization_file.strip() or None
    if resume_payload is not None:
        ckpt_file = str(resume_payload.get("checkpoint_file", ckpt_file)).strip() or ckpt_file
        norm_resume = str(resume_payload.get("normalization_file", "")).strip()
        if norm_resume:
            norm_file = norm_resume
    if not ckpt_file:
        raise ValueError("latent_swd.checkpoint_file is required to load the frozen encoder")
    ckpt_path = Path(ckpt_file).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"latent_swd checkpoint not found: {ckpt_path} "
            "(frozen latent-constraint encoder .ckpt from train_latent_constraint.py)"
        )
    if norm_file is not None:
        norm_path = Path(norm_file).expanduser()
        if not norm_path.is_file():
            raise FileNotFoundError(
                f"latent_swd.normalization_file not found: {norm_path}"
            )
    model, ckpt_meta = load_checkpoint(
        ckpt_path, device=device, normalization_file=norm_file
    )
    state = LatentSWDState(model=model, cfg=cfg)
    state.freeze()
    _log.info(
        "[latent_swd] loaded encoder: ckpt=%s latent_dim=%s d_model=%s "
        "val_loss=%s lc_epoch=%s params=%.3fM eval=%s requires_grad=False",
        ckpt_path,
        model.latent_dim,
        model.d_model,
        ckpt_meta.get("val_loss"),
        ckpt_meta.get("epoch"),
        sum(p.numel() for p in model.parameters()) / 1e6,
        not model.training,
    )
    return state


def broadcast_latent_swd_state(
    state: LatentSWDState,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Broadcast frozen encoder weights from rank 0 so every rank is bit-identical.

    Each rank can load the checkpoint file independently; this collective
    guarantees identical projection weights across nodes before CPO repair
    (important for multi-node bit-for-bit sync).
    """
    if world_size <= 1:
        return
    import torch.distributed as dist

    if rank == 0:
        cpu_sd = {k: v.detach().cpu() for k, v in state.model.state_dict().items()}
        payload: list[Any] = [cpu_sd]
    else:
        payload = [None]
    dist.broadcast_object_list(payload, src=0)
    if rank != 0:
        state.model.load_state_dict(payload[0], strict=True)
    state.model.to(device)
    state.freeze()


def _validate_dgpo_constraint_resume(
    resume_payload: Mapping[str, Any] | None,
    *,
    expected_type: str,
) -> None:
    """Reject DGPO resume when the saved constraint backend differs from config."""
    if resume_payload is None:
        return
    saved = resume_payload.get("constraint_type")
    if saved is None:
        return
    if str(saved) != expected_type:
        raise ValueError(
            f"DGPO checkpoint constraint_type={saved!r} but config expects "
            f"{expected_type!r}; fix projection_constraint.type or use a matching checkpoint."
        )


# ----------------------------------------------------------------------
# core SWD-ratio constraint
# ----------------------------------------------------------------------
def swd_ratio_constraint(
    z_pred: Tensor,
    z_truth: Tensor,
    *,
    num_projections: int,
    eps: float,
    seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Ratio-normalized null-excess SWD constraint.

    ``C_norm = (SWD(z_pred, z_truth) - SWD_tt) / (SWD_tt + eps)`` where ``SWD_tt``
    is the same-distribution null floor for ``z_truth``. One random projection set
    is sampled per call and **reused** for both distances so the ratio is
    self-consistent; the gradient flows through ``z_pred`` only.

    Null floor (``SWD_tt``): two **size-n bootstrap resamples** drawn (with
    replacement) from *all* ``n`` truth latents, so the null uses the full truth
    cloud and matches the ``n``-vs-``n`` sample size of ``SWD(z_pred, z_truth)``.
    This is markedly lower-variance than the previous disjoint ``n/2``-vs-``n/2``
    half-split (smaller samples + half the data) and removes its sample-size bias,
    giving CPO a steadier ratio denominator.

    ``seed`` (common random numbers): when given, the projection directions **and**
    the bootstrap null indices are drawn from a seeded generator. CPO evaluates this
    constraint several times per step (at ``theta_old`` for ``C`` and ``b``, then at
    ``theta_adam`` for the proxy); passing the *same* seed across those evaluations
    cancels the SWD estimator's sampling variance so the measured ``Delta C``
    reflects the parameter step instead of projection/split noise. ``None`` keeps
    the old per-call random behaviour.
    """
    device = z_pred.device
    dim = z_pred.shape[1]
    gen: torch.Generator | None = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    projections = random_projections(
        num_projections, dim, device=device, dtype=z_pred.dtype, generator=gen
    )

    swd_pred_truth = sliced_wasserstein_distance(
        z_pred, z_truth, projections=projections, detach_truth=True
    )

    n = int(z_truth.shape[0])
    z_truth_d = z_truth.detach()
    if n >= 2:
        # Bootstrap two independent size-n resamples from ALL truth latents.
        idx_a = torch.randint(0, n, (n,), device=device, generator=gen)
        idx_b = torch.randint(0, n, (n,), device=device, generator=gen)
        swd_tt = sliced_wasserstein_distance(
            z_truth_d[idx_a], z_truth_d[idx_b], projections=projections
        )
    else:
        swd_tt = z_pred.new_zeros(())

    swd_tt_d = swd_tt.detach()
    denom = swd_tt_d + float(eps)
    c_norm = (swd_pred_truth - swd_tt_d) / denom

    diag = {
        "latent_constraint/swd_pred_truth": swd_pred_truth.detach().to(torch.float64),
        "latent_constraint/swd_truth_truth": swd_tt_d.to(torch.float64),
        "latent_constraint/swd_ratio": (swd_pred_truth.detach() / denom).to(torch.float64),
        "latent_constraint/C_norm": c_norm.detach().to(torch.float64),
    }
    return c_norm, diag


def _backbone_from_batch(batch_kb: Mapping[str, Any]) -> dict[str, Tensor]:
    """Pull the event context the encoder needs: x, x_mask, conditions, conditions_mask."""
    if "x" not in batch_kb or "x_mask" not in batch_kb:
        raise KeyError("batch_kb must contain x and x_mask for the latent encoder context")
    x = batch_kb["x"]
    x_mask = batch_kb["x_mask"]
    if x_mask.dim() == 3 and x_mask.shape[-1] == 1:
        x_mask = x_mask.squeeze(-1)
    conditions = batch_kb.get("conditions")
    if conditions is None:
        raise KeyError("batch_kb must contain conditions for the latent encoder context")
    conditions_mask = batch_kb.get("conditions_mask")
    if conditions_mask is None:
        conditions_mask = torch.ones(
            conditions.shape[0], 1, device=conditions.device, dtype=conditions.dtype
        )
    out = {
        "x": x,
        "x_mask": x_mask,
        "conditions": conditions,
        "conditions_mask": conditions_mask,
    }
    # Token-bottleneck encoder input: the precomputed frozen event CLS token
    # (present when DGPO trains on the *_evttok augmented parquet). Pass it
    # through so pred and truth batches share the same event token.
    if "event_token" in batch_kb:
        out["event_token"] = batch_kb["event_token"]
    # object_token_bottleneck_ae also conditions on the per-object token set
    # (object_token: (KB, P, D)); forward it too so the encoder can self-attend
    # over objects. x_mask (already forwarded) gives the per-object validity.
    if "object_token" in batch_kb:
        out["object_token"] = batch_kb["object_token"]
    return out


def latent_swd_constraint_from_kin(
    state: LatentSWDState,
    batch_sel: Mapping[str, Any],
    pred_kin: Tensor,
    truth_kin: Tensor,
    *,
    seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Encode pred/truth neutrinos (given the event context) and return the SWD ratio.

    Args:
        state: frozen latent-constraint state.
        batch_sel: dict with the (already row-selected) event context
            (``x``, ``x_mask``, ``conditions``, ``conditions_mask``).
        pred_kin: ``(n, 2, 3)`` physical predicted neutrinos in the ENCODER's
            coordinate (cartesian ``px,py,pz`` or spherical ``log_pt,eta,phi``),
            gradient-carrying -> differentiable constraint.
        truth_kin: ``(n, 2, 3)`` physical truth neutrinos (detached internally).
        seed: optional common-random-numbers seed forwarded to
            :func:`swd_ratio_constraint` (fixes projections + null split).

    Returns:
        ``(C_norm, diag)``.
    """
    backbone = _backbone_from_batch(batch_sel)
    pred_batch = {**backbone, "nu_kin": pred_kin}
    truth_batch = {**backbone, "nu_kin": truth_kin}

    # Frozen encoder: eval mode disables dropout; grad still flows through inputs.
    state.model.eval()
    z_pred = state.model.encode_latent(pred_batch)
    z_truth = state.model.encode_latent(truth_batch, detach_neutrinos=True)
    return swd_ratio_constraint(
        z_pred, z_truth, num_projections=state.cfg.num_projections, eps=state.cfg.eps,
        seed=seed,
    )


# ----------------------------------------------------------------------
# DGPO entry point
# ----------------------------------------------------------------------
def compute_latent_swd_constraint(
    *,
    model_v: Tensor,
    x_t: Tensor,
    t_rep: Tensor,
    noise_mask_rep: Tensor,
    batch_kb: Mapping[str, Any] | dict[str, Any],
    core_model: torch.nn.Module,
    cartesian: bool,
    K: int,
    state: LatentSWDState,
    candidate_weights_kb: Tensor | None = None,
    update_ema: bool = True,
    world_size: int = 1,
    seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Differentiable latent-SWD constraint scalar for the DGPO CPO repair.

    Returns ``(C_norm, diag)`` with ``C_norm`` differentiable w.r.t. ``model_v``
    through the predicted neutrinos. ``update_ema`` / ``world_size`` are accepted
    for signature compatibility (the frozen encoder needs no EMA/grad sync; the
    scalar ``C`` is synced by the caller via
    :func:`sync_projection_constraint_C_across_ranks`).

    ``seed`` enables common random numbers for the SWD estimator: the CPO repair
    passes a per-(step, multi-sample) seed so the constraint at ``theta_old`` and
    its post-AdamW proxy at ``theta_adam`` share the same projections + null split,
    making the measured violation reflect the parameter step rather than SWD noise.
    """
    # Lazy import: keep the heavy DGPO stack out of import time so the core
    # encode+SWD helpers above stay unit-testable on their own.
    from RL.DGPO_neutrino.dgpo_utils import (
        predict_x0_normalized_from_velocity_diffusion,
    )
    from RL.DGPO_neutrino.rewards import log_pt_eta_phi_to_cartesian

    cfg = state.cfg
    if "event_token" not in batch_kb:
        raise KeyError(
            "latent_swd encoder needs 'event_token' in batch_kb: point the DGPO data "
            "pipeline at the *_evttok augmented parquet "
            "(preprocessing/augment_event_token.py)."
        )
    if "object_token" not in batch_kb:
        raise KeyError(
            "latent_swd encoder needs 'object_token' in batch_kb: build the DGPO data "
            "with preprocessing/augment_event_token.py --object-tokens (the *_evttok "
            "mirror must carry object_token:[P,D])."
        )
    zero = torch.zeros((), device=model_v.device, dtype=model_v.dtype)
    diag: dict[str, Tensor] = {
        "latent_constraint/active": torch.tensor(1.0, device=model_v.device, dtype=torch.float64),
    }

    # 1) recover predicted x0 -> physical neutrinos (grad-capable denormalize).
    x0_hat, _, _ = predict_x0_normalized_from_velocity_diffusion(x_t, model_v, t_rep)
    pad = int(getattr(core_model, "invisible_padding", 0))
    invisible_normalizer = getattr(core_model, "invisible_normalizer", None)
    if invisible_normalizer is None:
        raise AttributeError("core_model must expose invisible_normalizer")
    denormalize_grad = getattr(invisible_normalizer, "denormalize_grad", None)
    if denormalize_grad is None:
        raise AttributeError(
            "invisible_normalizer must expose denormalize_grad for the differentiable constraint"
        )
    denorm_mask = noise_mask_rep if noise_mask_rep.dim() == 3 else noise_mask_rep.unsqueeze(-1)
    nu_phys = denormalize_grad(x0_hat, mask=denorm_mask, remove_padding=pad > 0)

    KB = int(nu_phys.shape[0])
    B = KB // max(int(K), 1)
    if KB != int(K) * B:
        raise ValueError(f"expected KB=K*B, got KB={KB}, K={K}, B={B}")

    # 2) pred/truth kin in the ENCODER's coordinate system + validity mask.
    #    A cartesian-trained encoder wants (px, py, pz); the spherical encoder
    #    wants (log_pt, eta, phi). Feed whatever the loaded encoder was trained in.
    if bool(getattr(state.model, "cartesian", False)):
        if cartesian:
            pred_kin = nu_phys[:, :2, :3]                       # policy already cartesian
        else:  # policy spherical -> convert predicted neutrinos to cartesian
            px, py, pz = log_pt_eta_phi_to_cartesian(
                nu_phys[:, :2, 0], nu_phys[:, :2, 1], nu_phys[:, :2, 2]
            )
            pred_kin = torch.stack((px, py, pz), dim=-1)
        truth_xyz = batch_kb.get("x_invisible_cartesian")
        if not isinstance(truth_xyz, Tensor):
            raise KeyError(
                "latent_swd cartesian encoder requires x_invisible_cartesian in batch_kb"
            )
        truth_kin = truth_xyz.to(device=nu_phys.device, dtype=nu_phys.dtype)[:, :2, :3]
    else:
        pred_kin, truth_kin = _pred_truth_kin_log_pt_eta_phi(nu_phys, batch_kb, cartesian=cartesian)
    event_ok = _event_mask_kb(batch_kb, K=int(K), KB=KB, device=nu_phys.device, dtype=nu_phys.dtype)
    cand_ok = _candidate_mask_kb(
        candidate_weights_kb, K=int(K), B=B, device=nu_phys.device, apply_to=cfg.apply_to
    )
    finite = (
        event_ok
        & cand_ok
        & torch.isfinite(pred_kin).all(dim=-1).all(dim=-1)
        & torch.isfinite(truth_kin).all(dim=-1).all(dim=-1)
    )
    idx = torch.nonzero(finite, as_tuple=False).reshape(-1)
    diag["latent_constraint/mask_count"] = torch.tensor(
        float(idx.numel()), device=model_v.device, dtype=torch.float64
    )

    if int(idx.numel()) < int(cfg.min_samples):
        diag["latent_constraint/skipped_small_mask"] = torch.tensor(
            1.0, device=model_v.device, dtype=torch.float64
        )
        # Keep the constraint on the graph (zero) so .backward() never errors.
        return zero + model_v.sum() * 0.0, diag

    batch_sel = {
        k: (v[idx] if isinstance(v, Tensor) and v.shape[0] == KB else v)
        for k, v in batch_kb.items()
    }

    # 3) encode + SWD ratio.
    c_norm, swd_diag = latent_swd_constraint_from_kin(
        state, batch_sel, pred_kin[idx], truth_kin[idx], seed=seed
    )
    diag.update(swd_diag)
    diag["latent_constraint/skipped_small_mask"] = torch.tensor(
        0.0, device=model_v.device, dtype=torch.float64
    )
    return c_norm, diag
