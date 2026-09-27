"""
Standalone DGPO fine-tuning loop for neutrino diffusion (Step 5 of the neutrino RL plan).

Uses the same ``EveNetModel`` backbone and Parquet → Ray → ``iter_torch_batches`` path as
``evenet/train.py``, but replaces Lightning with a plain PyTorch optimizer step and the DGPO
objective from ``dgpo_utils.py``.
"""

from __future__ import annotations

import argparse
import copy
import heapq
import logging
import math
import re
import os
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import jensenshannon
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from evenet.utilities.fourier_integration import optimizer_parameters as module_optimizer_parameters

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ray
import ray.train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from evenet.control.global_config import global_config
from evenet.shared import make_process_fn, prepare_datasets, register_dataset
from evenet.utilities.diffusion_sampler import (
    DDIMSampler,
    get_logsnr_alpha_sigma,
)

from RL.DGPO_neutrino.dgpo_utils import (
    ADVANTAGE_ESTIMATOR_ZSCORE,
    REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
    REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
    VALID_ADVANTAGE_ESTIMATORS,
    VALID_REFERENCE_TRUST_OBJECTIVES,
    _dgpo_cfg_get,
    adaptive_trust_backtracking_scales,
    adaptive_trust_update_scale,
    build_dgpo_loss,
    build_reference_trust_loss,
    compute_per_event_advantage,
    predict_x0_normalized_from_velocity_diffusion,
    rebase_trainable_params_for_extragradient_,
    reset_adam_first_moment,
    repeat_batch_for_candidates,
    sample_cosine_vp_path_kl_timesteps,
)
from RL.DGPO_neutrino.model_utils import (
    apply_component_freezes,
    assert_dgpo_neutrino_policy_deterministic,
    freeze_reference_model,
    generation_uses_ema_shadow,
    load_evenet_model_for_dgpo,
    make_ema,
    make_ema_rollout,
    make_reference_model,
    make_round_reference_model,
    parse_dgpo_resume_from_checkpoint,
    resolve_dgpo_auto_resume_checkpoint,
    save_lightning_compatible_checkpoint,
    select_dgpo_training_state,
    dgpo_snapshot_checkpoint_name,
    update_last_checkpoint_pointer,
    unwrap_for_state_dict,
)
from RL.DGPO_neutrino.endpoint_kl import (
    CHECKPOINT_KEY as ENDPOINT_KL_CHECKPOINT_KEY,
    EndpointKLConfig,
    EndpointKLController,
    validate_endpoint_protocol,
)
from RL.DGPO_neutrino.monitoring import (
    append_reference_prediction_arrays,
    paired_finite_truth_pred,
    scheduled_epoch,
    select_truth_pred_by_class,
    validation_schedule_tier,
)
from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
from RL.DGPO_neutrino.latent_constraint.dgpo_constraint import (
    LatentSWDState,
    broadcast_latent_swd_state,
    sync_projection_constraint_C_across_ranks,
    compute_latent_swd_constraint,
    init_latent_swd_state,
    _validate_dgpo_constraint_resume,
)
from RL.DGPO_neutrino.projection_cpo import (
    ProjectionConstraintConfig,
    assign_params_,
    assign_params_from_theta_old_delta_,
    compute_cpo_adamw_final_update,
    compute_projection_lambda,
    compute_projection_lambda_from_violation,
    flatten_adam_preconditioned_direction,
    flatten_param_delta,
    flatten_param_grads,
    projection_stratified_t_grid,
    resolve_projection_constraint_config,
    snapshot_params,
    trainable_params_all_finite,
)
from RL.DGPO_neutrino.gradient_transfer import (
    resolve_gradient_transfer_trace_config,
    summarize_gradient_transfer,
    trace_due as gradient_transfer_trace_due,
)
from RL.DGPO_neutrino.optimizer_transaction import (
    assign_scaled_trainable_update_,
    resolve_parameter_update_rms_calibration,
    restore_adamw_state_from_transaction_,
    snapshot_adamw_state_for_transaction,
)
from RL.DGPO_neutrino.rewards import (
    CalibrationMagnitudeReward,
    ComponentNormalizedTruthDistanceReward,
    RewardAggregator,
    cartesian_to_log_pt_eta_phi,
    get_event_valid_mask,
    log_pt_eta_phi_to_cartesian,
)
from RL.DGPO_neutrino.domains.ztautau import build_feature_space_scales
from RL.DGPO_neutrino.diagnostics.ztautau_validation import (
    build_ztautau_validation_metrics,
    collect_ztautau_validation_arrays,
)

_log = logging.getLogger(__name__)

# Highest committed transport row. Never use this as a scientific training axis:
# many classifier-fit records can occur at the same completed DGPO step.
_wandb_committed_step: int = -1

# Single source of truth for live classifier phases.  W&B registration and the
# runtime logger must stay in lock-step; otherwise a valid long-running fit can
# fail merely when it emits its first progress row.
_OMNIFOLD_LIVE_PHASE_IDS = {
    "residual_reward": 0,
    "acceptance_audit": 1,
    "topology_acceptance_audit": 2,
    "staleness_audit": 3,
    "baseline_audit": 4,
    "raw_staleness_audit": 5,
    "signed_direction_audit": 6,
    "raw_staleness_monitor": 7,
    "reference_trust_monitor": 8,
    "global_best_candidate": 9,
    "global_best_incumbent": 10,
}

_GRAD_CLIP_NORM = 1.0
# Projection-constraint payload key in DGPO checkpoints. The legacy key (from the
# removed discriminator-Wasserstein era) is still read on resume for older runs.
_DGPO_CONSTRAINT_CKPT_KEY = "dgpo_projection_constraint_state"
_DGPO_CONSTRAINT_CKPT_KEY_LEGACY = "dgpo_discriminator_wasserstein_state"
ProjectionConstraintState = LatentSWDState
_RL_DISABLED_MSG = (
    "RL pipeline disabled. Use evenet/train.py for original foundation training."
)


def _assert_rl_enabled() -> None:
    """Exit cleanly when ``rl.enabled`` is false (foundation training stays in evenet/train.py)."""
    rl = getattr(global_config, "rl", None)
    if rl is None or not bool(getattr(rl, "enabled", False)):
        raise SystemExit(_RL_DISABLED_MSG)



def _dgpo_rollout_ema_decay(global_step: int) -> float:
    """Effective decay for rollout EMA update: ``min(max, ramp * step)`` (Flow GRPO ``ema_ref`` style)."""
    dg = global_config.dgpo
    decay_max = float(dg.get("ema_rollout_decay_max", 0.3))
    decay_ramp = float(dg.get("ema_rollout_decay_ramp", 0.001))
    return min(decay_max, decay_ramp * float(global_step))


class _DGPODDPForward(nn.Module):
    """Routes DDP ``forward`` to ``EveNetModel.predict_diffusion_vector`` (neutrino mode)."""

    def __init__(self, eve_net: nn.Module) -> None:
        super().__init__()
        self.eve_net = eve_net

    def forward(
        self,
        noise_x: Tensor,
        cond_x: dict[str, Any],
        time: Tensor,
        noise_mask: Tensor,
    ) -> Tensor:
        return self.eve_net.predict_diffusion_vector(
            noise_x=noise_x,
            cond_x=cond_x,
            time=time,
            mode="neutrino",
            noise_mask=noise_mask,
        )


def _unwrap_core_evenet(model: nn.Module) -> nn.Module:
    """Unwrap ``DDP(_DGPODDPForward(eve))`` to the underlying ``EveNetModel``."""
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "eve_net") and isinstance(getattr(m, "eve_net"), nn.Module):
        return m.eve_net
    return m


def _set_dgpo_activation_checkpointing(
    model: nn.Module, *, enabled: bool
) -> int:
    """Toggle block-wise checkpointing on PET bodies; return modules changed."""

    count = 0
    for module in _unwrap_core_evenet(model).modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = bool(enabled)
            count += 1
    return count


_DGPO_POLICY_BATCH_TENSOR_KEYS = frozenset(
    {
        "x",
        "x_mask",
        "conditions",
        "conditions_mask",
        "classification",
        "x_invisible",
        "x_invisible_mask",
    }
)


def _dgpo_policy_conditioning_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """View containing only tensors consumed by neutrino policy forwards."""

    required = {"x", "x_mask", "conditions", "conditions_mask", "x_invisible_mask"}
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"DGPO policy batch is missing required tensors: {missing}")
    return {
        key: value
        for key, value in batch.items()
        if not isinstance(value, Tensor) or key in _DGPO_POLICY_BATCH_TENSOR_KEYS
    }


def _slice_event_batch(
    batch: dict[str, Any],
    start: int,
    stop: int,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Slice tensors whose leading dimension is the event dimension."""

    return {
        key: (
            value[start:stop]
            if isinstance(value, Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == int(batch_size)
            else value
        )
        for key, value in batch.items()
    }


def _next_batch_synced(
    iterator: Any,
    *,
    world_size: int,
    device: torch.device,
    require_all_ranks: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    """Pull the next batch from a per-rank Ray DataIterator with cross-rank termination sync.

    Each rank fetches its own batch from its own shard. To keep DDP collectives in lock-step,
    we all-reduce a "has-more" flag with ``MIN``: the loop terminates as soon as **any** rank
    runs out of data. This may drop a few batches from longer shards but prevents NCCL hangs.

    For unwrapped no-grad pool generation, ``require_all_ranks=False`` uses MAX
    so longer shards can finish; empty ranks skip forward but join termination
    collectives. Never use this mode for DDP training forwards.

    Returns ``(batch, can_continue)``; ``batch`` is None if this rank is exhausted.
    """
    try:
        batch = next(iterator)
        local_has = 1
    except StopIteration:
        batch = None
        local_has = 0

    if world_size > 1:
        flag = torch.tensor([local_has], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN if require_all_ranks else dist.ReduceOp.MAX)
        all_have = bool(flag.item() > 0)
    else:
        all_have = local_has > 0
    return batch, all_have


def _all_ranks_scalar_finite(
    value: Tensor,
    *,
    device: torch.device,
    world_size: int,
) -> bool:
    """Return true only when every rank observes a finite scalar."""

    local = torch.tensor(
        int(bool(torch.isfinite(value.detach()).all().item())),
        device=device,
        dtype=torch.int32,
    )
    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed DGPO finite check requires an initialized process group"
            )
        dist.all_reduce(local, op=dist.ReduceOp.MIN)
    return bool(local.item())


@torch.no_grad()
def _all_reduce_accumulated_gradients(
    model: torch.nn.Module,
    *,
    world_size: int,
) -> int:
    """Average accumulated gradients without retaining overlapping DDP graphs.

    Sequential VP-trust backward evaluates both policy graphs through the
    unwrapped EveNet module. One flattened all-reduce per device/dtype group
    reproduces DDP's mean gradient after all microbatches and loss components
    have contributed.
    """

    if world_size <= 1:
        return sum(
            int(parameter.grad is not None)
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "distributed sequential VP-trust backward requires an initialized "
            "process group"
        )
    grouped: dict[
        tuple[torch.device, torch.dtype],
        list[torch.nn.Parameter],
    ] = defaultdict(list)
    for parameter in model.parameters():
        if parameter.requires_grad:
            grouped[(parameter.device, parameter.dtype)].append(parameter)
    reduced = 0
    for (parameter_device, _), parameters in grouped.items():
        # A conditional branch can leave a parameter unused on one rank but
        # active on another.  Select the globally-active set first so every
        # rank launches identical collectives; missing local gradients are the
        # correct zero contribution to the distributed mean.
        present = torch.tensor(
            [int(parameter.grad is not None) for parameter in parameters],
            device=parameter_device,
            dtype=torch.int32,
        )
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        active_parameters = [
            parameter
            for parameter, globally_present in zip(parameters, present.tolist())
            if globally_present
        ]
        if not active_parameters:
            continue
        flat = torch.cat(
            [
                (
                    parameter.grad.reshape(-1)
                    if parameter.grad is not None
                    else torch.zeros_like(parameter).reshape(-1)
                )
                for parameter in active_parameters
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(float(world_size))
        offset = 0
        for parameter in active_parameters:
            count = int(parameter.numel())
            averaged = flat[offset : offset + count].view_as(parameter)
            if parameter.grad is None:
                # Clone so this small gradient does not retain the full flat
                # all-reduce buffer after the helper returns.
                parameter.grad = averaged.clone()
            else:
                parameter.grad.copy_(averaged)
            offset += count
            reduced += 1
    return reduced


def _truth_generation_cartesian() -> bool:
    tg = global_config.options.Training.Components.TruthGeneration
    return bool(getattr(tg, "cartesian", False))


def _histogram_jsd(truth_counts: np.ndarray, pred_counts: np.ndarray) -> float:
    """Jensen-Shannon distance between two count histograms, or ``nan`` when undefined."""
    truth = np.asarray(truth_counts, dtype=np.float64)
    pred = np.asarray(pred_counts, dtype=np.float64)
    truth_sum = float(truth.sum())
    pred_sum = float(pred.sum())
    if not np.isfinite(truth_sum) or not np.isfinite(pred_sum):
        return float("nan")
    if truth_sum <= 0.0 or pred_sum <= 0.0:
        return float("nan")
    return float(jensenshannon(truth / truth_sum, pred / pred_sum))


def _array_histogram_jsd(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
    num_bins: int = 40,
) -> float:
    """Histogram JSD from raw truth/pred arrays using explicit or data-driven bin edges."""
    truth = np.asarray(truth_values, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    truth = truth[np.isfinite(truth)]
    pred = pred[np.isfinite(pred)]
    if truth.size == 0 or pred.size == 0:
        return float("nan")

    edges = None if bin_edges is None else np.asarray(bin_edges, dtype=np.float64).reshape(-1)
    if edges is None or edges.size < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
        merged = np.concatenate((truth, pred), axis=0)
        lo, hi = [float(x) for x in np.nanpercentile(merged, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi):
            return float("nan")
        if hi <= lo:
            center = float(np.nanmean(merged))
            span = max(abs(center) * 0.1, 1.0)
            lo, hi = center - span, center + span
        pad = max(0.05 * (hi - lo), 1e-6)
        edges = np.linspace(lo - pad, hi + pad, max(2, int(num_bins)) + 1)

    truth_counts, _ = np.histogram(truth, bins=edges)
    pred_counts, _ = np.histogram(pred, bins=edges)
    return _histogram_jsd(truth_counts, pred_counts)


def _truth_pred_scalar_metrics(
    truth_values: np.ndarray,
    pred_values: np.ndarray,
) -> dict[str, float]:
    """Scalar summaries for truth-vs-pred arrays used by 2D monitoring panels."""
    truth, pred = paired_finite_truth_pred(
        truth_values,
        pred_values,
        context="truth/pred monitoring",
    )
    if truth.size == 0:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "pearson_r": float("nan"),
            "slope": float("nan"),
            "intercept": float("nan"),
        }
    delta = pred - truth
    mae = float(np.mean(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    bias = float(np.mean(delta))
    if truth.size >= 2:
        truth_mean = float(np.mean(truth))
        pred_mean = float(np.mean(pred))
        truth_centered = truth - truth_mean
        pred_centered = pred - pred_mean
        denom = float(np.sqrt(np.sum(truth_centered * truth_centered) * np.sum(pred_centered * pred_centered)))
        pearson_r = float(np.sum(truth_centered * pred_centered) / denom) if denom > 0.0 else float("nan")
        truth_var = float(np.sum(truth_centered * truth_centered))
        slope = float(np.sum(truth_centered * pred_centered) / truth_var) if truth_var > 0.0 else float("nan")
        intercept = pred_mean - slope * truth_mean if math.isfinite(slope) else float("nan")
    else:
        pearson_r = float("nan")
        slope = float("nan")
        intercept = float("nan")
    return {
        "count": float(truth.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "pearson_r": pearson_r,
        "slope": slope,
        "intercept": intercept,
    }


@torch.no_grad()
def _kin_hist_candidate_indices_per_event(
    rewards_kb: Tensor,
    candidates_kb: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
) -> Tensor:
    """Per-event candidate index ``(B,)`` for ``train_dist/*`` and ``val_neutrino/*`` histograms.

    Uses the scalar ``rewards_kb`` argmax (component-normalized truth-distance reward).
    """
    return rewards_kb.argmax(dim=0)


@torch.no_grad()
def compute_reward_mean_gap(rewards_kb: Tensor, valid_b: Tensor) -> float:
    """Mean over valid events of (mean reward above median − mean reward below median) along ``K``."""
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0 or rewards_kb.shape[0] < 2:
        return float("nan")
    r = rewards_kb[:, vb]
    med = r.median(dim=0).values.unsqueeze(0)
    good_m = r > med
    bad_m = r < med
    good_den = good_m.sum(dim=0).clamp(min=1).to(r.dtype)
    bad_den = bad_m.sum(dim=0).clamp(min=1).to(r.dtype)
    good_mean = (r * good_m.to(r.dtype)).sum(dim=0) / good_den
    bad_mean = (r * bad_m.to(r.dtype)).sum(dim=0) / bad_den
    return float((good_mean - bad_mean).mean().cpu())


@torch.no_grad()
def compute_reward_advantage_pos_neg_gap(
    rewards_kb: Tensor,
    advantages_kb: Tensor,
    valid_b: Tensor,
) -> float:
    """Mean reward where advantage > 0 minus mean reward where advantage < 0 (valid events only)."""
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        return float("nan")
    r = rewards_kb[:, vb]
    a = advantages_kb[:, vb]
    pos = a > 0
    neg = a < 0
    if not pos.any() or not neg.any():
        return float("nan")
    pos_m = r[pos].mean()
    neg_m = r[neg].mean()
    return float((pos_m - neg_m).cpu())


def _grad_norm_pre_clip_and_clip_active(
    model: nn.Module,
    max_norm: float,
) -> tuple[float, float]:
    """Return (total L2 grad norm before clipping, 1.0 if norm exceeded ``max_norm`` else 0.0).

    ``torch.nn.utils.clip_grad_norm_`` returns the norm **before** scaling; clipping applies when
    that norm exceeds ``max_norm``.
    """
    gn = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm))
    )
    active = 1.0 if gn > float(max_norm) + 1e-12 else 0.0
    return gn, active


def _dgpo_nonfinite_fraction(t: Tensor) -> float:
    """Fraction of elements in ``t`` that are not finite."""
    if t.numel() == 0:
        return 0.0
    return float((~torch.isfinite(t)).float().mean().detach().cpu())


def _dgpo_zero_nonfinite(t: Tensor) -> Tensor:
    """Replace non-finite entries with zero."""
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t))


def _dgpo_sanitize_rollout_rewards(
    rewards: Tensor,
    reward_breakdown: dict[str, Tensor],
    *,
    global_step: int,
) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
    """Zero non-finite rewards and emit diagnostic fractions."""
    diag: dict[str, float] = {}
    rew_frac = _dgpo_nonfinite_fraction(rewards)
    if rew_frac > 0.0:
        diag["train/reward_nonfinite_fraction"] = rew_frac
        for name, rb in reward_breakdown.items():
            src_frac = _dgpo_nonfinite_fraction(rb)
            if src_frac > 0.0:
                diag[f"train/reward_nonfinite_fraction/{name}"] = src_frac
                _log.warning(
                    "[DGPO] non-finite reward source %s (frac=%.4g) at global_step=%s.",
                    name,
                    src_frac,
                    global_step,
                )
                reward_breakdown[name] = _dgpo_zero_nonfinite(rb)
        _log.warning(
            "[DGPO] non-finite total rewards (frac=%.4g) at global_step=%s; zeroing.",
            rew_frac,
            global_step,
        )
        rewards = _dgpo_zero_nonfinite(rewards)
    return rewards, reward_breakdown, diag


def _dgpo_assert_train_step_invariants(
    L_ref: Tensor,
    advantages: Tensor,
    rewards: Tensor,
) -> None:
    """Cheap per-step guards for DGPO training."""
    assert not L_ref.requires_grad, "[DGPO CHECK] L_ref must not require grad."
    assert not advantages.requires_grad, "[DGPO CHECK] advantages must not require grad."
    if not torch.isfinite(rewards).all():
        bad = int((~torch.isfinite(rewards)).sum().item())
        raise AssertionError(
            f"[DGPO CHECK] rewards must be finite after sanitization ({bad} bad entries)."
        )
    if not torch.isfinite(L_ref).all():
        bad = int((~torch.isfinite(L_ref)).sum().item())
        raise AssertionError(
            f"[DGPO CHECK] L_ref must be finite ({bad} bad entries); check model weights / DDIM."
        )
    if not torch.isfinite(advantages).all():
        bad = int((~torch.isfinite(advantages)).sum().item())
        raise AssertionError(
            f"[DGPO CHECK] advantages must be finite ({bad} bad entries)."
        )


_REWARD_DIST_OVERLAY_BINS = 40
_REL_PT_DIST_BINS = 50


def _reward_dist_overlaid_figure(
    best: np.ndarray,
    worst: np.ndarray,
    med: np.ndarray,
) -> Any:
    """Three overlapped 1D histograms (density), EveNet validation style, as ``wandb.Image``."""
    import wandb

    stacked = np.concatenate([best, worst, med])
    lo = float(np.min(stacked))
    hi = float(np.max(stacked))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0
    elif hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    else:
        span = hi - lo
        pad = max(1e-6 * span, 1e-9)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REWARD_DIST_OVERLAY_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    labels = ("best (max per event)", "worst (min per event)", "median along K")
    for arr, c, lab in (
        (best, colors[0], labels[0]),
        (worst, colors[1], labels[1]),
        (med, colors[2], labels[2]),
    ):
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.42,
            label=lab,
            color=c,
            histtype="stepfilled",
        )
    ax.set_xlabel("Reward")
    ax.set_ylabel("Density")
    ax.set_title("Per-event reward (best / worst / median among K)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _finite_1d_numpy(arr: np.ndarray) -> np.ndarray:
    """Return finite 1D float values for histogram plotting."""
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def _rel_pt_distribution_figure(
    all_rel_pt: np.ndarray,
    best_rel_pt: np.ndarray,
) -> Any:
    """Overlaid density plot for ``pT_pred / pT_truth - 1`` diagnostics."""
    import wandb

    all_rel_pt = _finite_1d_numpy(all_rel_pt)
    best_rel_pt = _finite_1d_numpy(best_rel_pt)
    stacked = np.concatenate([all_rel_pt, best_rel_pt])
    if stacked.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = [float(x) for x in np.nanpercentile(stacked, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max(0.05 * (hi - lo), 1e-3)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REL_PT_DIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for arr, color, label in (
        (all_rel_pt, "#1f77b4", "all K candidates"),
        (best_rel_pt, "#d62728", "reward-best candidate"),
    ):
        if arr.size == 0:
            continue
        ax.hist(
            arr,
            bins=bins,
            density=True,
            alpha=0.65,
            label=f"{label}: mean={arr.mean():+.3f}, mean abs={np.abs(arr).mean():.3f}",
            color=color,
            histtype="step",
            linewidth=2.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
    ax.set_ylabel("Normalized density")
    ax.set_title("Relative pT residual distribution")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _single_rel_pt_distribution_figure(
    rel_pt: np.ndarray,
    *,
    title: str,
    label: str,
) -> Any:
    """Single-series density plot for reference/rollout relative-pT bias diagnostics."""
    import wandb

    rel_pt = _finite_1d_numpy(rel_pt)
    if rel_pt.size == 0:
        lo, hi = -1.0, 1.0
    else:
        lo, hi = [float(x) for x in np.nanpercentile(rel_pt, [0.5, 99.5])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(rel_pt)) if rel_pt.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max(0.05 * (hi - lo), 1e-3)
        lo -= pad
        hi += pad
    bins = np.linspace(lo, hi, _REL_PT_DIST_BINS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if rel_pt.size > 0:
        ax.hist(
            rel_pt,
            bins=bins,
            density=True,
            alpha=0.75,
            label=f"{label}: mean={rel_pt.mean():+.3f}, mean abs={np.abs(rel_pt).mean():.3f}",
            color="#9467bd",
            histtype="step",
            linewidth=2.0,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{pred} / p_T^{truth} - 1$")
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _pt_delta_vs_truth_pt_figure(
    truth_pt: np.ndarray,
    delta_pt: np.ndarray,
    *,
    title: str,
) -> Any:
    """Profile plot of mean ``pT_pred - pT_truth`` in truth-pT bins."""
    import wandb

    truth_pt = np.asarray(truth_pt, dtype=np.float64).reshape(-1)
    delta_pt = np.asarray(delta_pt, dtype=np.float64).reshape(-1)
    if truth_pt.shape != delta_pt.shape:
        n = min(truth_pt.size, delta_pt.size)
        truth_pt = truth_pt[:n]
        delta_pt = delta_pt[:n]
    keep = np.isfinite(truth_pt) & np.isfinite(delta_pt) & (truth_pt >= 0.0)
    truth_pt = truth_pt[keep]
    delta_pt = delta_pt[keep]

    bin_edges = _diagnostic_bin_edges("pt")
    num_bins = len(bin_edges) - 1
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    means = np.full(num_bins, np.nan, dtype=np.float64)
    errors = np.full(num_bins, np.nan, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    if truth_pt.size > 0:
        bin_idx = np.digitize(truth_pt, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        for i in range(num_bins):
            vals = delta_pt[valid & (bin_idx == i)]
            counts[i] = int(vals.size)
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                errors[i] = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    has_points = np.isfinite(means)
    if np.any(has_points):
        ax.errorbar(
            centers[has_points],
            means[has_points],
            yerr=errors[has_points],
            fmt="o-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label=r"mean $(p_T^{pred} - p_T^{truth})$",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    ax_count.bar(
        centers,
        counts,
        width=float(bin_edges[1] - bin_edges[0]) * 0.85,
        alpha=0.12,
        color="gray",
        label="entries",
    )
    ax_count.set_ylabel("Entries")
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _binned_delta_profile(
    truth_value: np.ndarray,
    delta_value: np.ndarray,
    *,
    bin_edges: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return truth-value bin centers, mean residual, standard error, and counts."""
    truth_value = np.asarray(truth_value, dtype=np.float64).reshape(-1)
    delta_value = np.asarray(delta_value, dtype=np.float64).reshape(-1)
    if truth_value.shape != delta_value.shape:
        n = min(truth_value.size, delta_value.size)
        truth_value = truth_value[:n]
        delta_value = delta_value[:n]
    keep = np.isfinite(truth_value) & np.isfinite(delta_value)
    truth_value = truth_value[keep]
    delta_value = delta_value[keep]

    if bin_edges is None:
        bin_edges = _diagnostic_bin_edges("pt")
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    num_bins = len(bin_edges) - 1
    means = np.full(num_bins, np.nan, dtype=np.float64)
    errors = np.full(num_bins, np.nan, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    if truth_value.size > 0:
        bin_idx = np.digitize(truth_value, bin_edges) - 1
        valid = (bin_idx >= 0) & (bin_idx < num_bins)
        for i in range(num_bins):
            vals = delta_value[valid & (bin_idx == i)]
            counts[i] = int(vals.size)
            if vals.size > 0:
                means[i] = float(np.mean(vals))
                errors[i] = float(np.std(vals) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
    return centers, means, errors, counts


def _pt_delta_selection_profiles_figure(
    truth_pt_all: np.ndarray,
    delta_pt_all: np.ndarray,
    truth_pt_best: np.ndarray,
    delta_pt_best: np.ndarray,
    truth_pt_oracle: np.ndarray | None = None,
    delta_pt_oracle: np.ndarray | None = None,
    *,
    title: str,
) -> Any:
    """Profile plot comparing rollout-all, reward-best, and optional pT-oracle pT delta."""
    import wandb

    centers, mean_all, err_all, _ = _binned_delta_profile(truth_pt_all, delta_pt_all)
    _, mean_best, err_best, counts = _binned_delta_profile(truth_pt_best, delta_pt_best)
    mean_oracle = err_oracle = None
    if truth_pt_oracle is not None and delta_pt_oracle is not None:
        _, mean_oracle, err_oracle, _ = _binned_delta_profile(
            truth_pt_oracle, delta_pt_oracle
        )
    best_gap = mean_best - mean_all
    oracle_gap = mean_oracle - mean_all if mean_oracle is not None else None

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    has_all = np.isfinite(mean_all)
    has_best = np.isfinite(mean_best)
    if np.any(has_all):
        ax.errorbar(
            centers[has_all],
            mean_all[has_all],
            yerr=err_all[has_all],
            fmt="o-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label="all rollout candidates",
        )
    if np.any(has_best):
        ax.errorbar(
            centers[has_best],
            mean_best[has_best],
            yerr=err_best[has_best],
            fmt="s-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            label="reward-best candidates",
        )
    if mean_oracle is not None and err_oracle is not None:
        has_oracle = np.isfinite(mean_oracle)
        if np.any(has_oracle):
            ax.errorbar(
                centers[has_oracle],
                mean_oracle[has_oracle],
                yerr=err_oracle[has_oracle],
                fmt="^-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                label="pT-oracle-best candidates",
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    has_gap = np.isfinite(best_gap)
    if np.any(has_gap):
        ax_gap.plot(
            centers[has_gap],
            best_gap[has_gap],
            "o-",
            linewidth=1.8,
            markersize=4,
            color="#d62728",
            label="reward-best - all",
        )
    if oracle_gap is not None:
        has_oracle_gap = np.isfinite(oracle_gap)
        if np.any(has_oracle_gap):
            ax_gap.plot(
                centers[has_oracle_gap],
                oracle_gap[has_oracle_gap],
                "^-",
                linewidth=1.8,
                markersize=4,
                color="#2ca02c",
                label="pT-oracle - all",
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax_gap.set_ylabel(r"$\Delta p_T$ gap [GeV]")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _profile_bin_edges(profile_name: str, truth_arrays: list[np.ndarray]) -> np.ndarray:
    """Bin edges for residual profile plots keyed by the profiled truth variable."""
    fixed_edges = _diagnostic_bin_edges(profile_name)
    if fixed_edges is not None:
        return fixed_edges

    finite_parts = [
        np.asarray(arr, dtype=np.float64).reshape(-1)
        for arr in truth_arrays
        if isinstance(arr, np.ndarray) and arr.size > 0
    ]
    if not finite_parts:
        lo, hi = -100.0, 100.0
    else:
        values = np.concatenate(finite_parts, axis=0)
        values = values[np.isfinite(values)]
        if values.size == 0:
            lo, hi = -100.0, 100.0
        else:
            lo, hi = [float(x) for x in np.nanpercentile(values, [0.5, 99.5])]
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                center = float(np.nanmean(values)) if values.size > 0 else 0.0
                lo, hi = center - 100.0, center + 100.0
            hi = max(abs(lo), abs(hi), 1.0)
            lo = -hi
    pad = max(0.05 * (hi - lo), 1e-3)
    return np.linspace(lo - pad, hi + pad, _VAL_KIN_NUM_BINS + 1)


def _profile_axis_labels(profile_name: str) -> tuple[str, str, str]:
    """Return x-label, y-label, and display name for a residual profile variable."""
    display = {
        "pt": "pT",
        "eta": "eta",
        "phi": "phi",
        "px": "px",
        "py": "py",
        "pz": "pz",
    }.get(profile_name, profile_name)
    if profile_name in {"pt", "px", "py", "pz"}:
        return f"Truth {display} [GeV]", f"Mean delta {display} [GeV]", display
    if profile_name == "phi":
        return "Truth phi [rad]", "Mean wrapped delta phi [rad]", display
    return f"Truth {display}", f"Mean delta {display}", display


def _invisible_feature_names() -> tuple[str, ...]:
    """Invisible feature names from ``event_info.yaml`` (fallback: reward_config)."""
    event_info = getattr(global_config, "event_info", None)
    raw = getattr(event_info, "invisible_feature_names", None)
    if raw:
        return tuple(str(name) for name in raw)
    reward_names = _reward_feature_names()
    return tuple(reward_names) if reward_names is not None else ()


def _invisible_periodic_feature_indices() -> tuple[int, ...]:
    """Periodic invisible-feature indices from ``event_info.yaml`` uniform/inv-CDF metadata."""
    event_info = getattr(global_config, "event_info", None)
    raw = getattr(event_info, "invisible_inv_cdf_index", None)
    if raw is None:
        return ()
    return tuple(int(index) for index in raw)


def _generation_monitor_feature_names(*, cartesian: bool) -> tuple[str, ...]:
    """Feature names for generation-style monitoring plots."""
    if cartesian:
        return ("px", "py", "pz")
    return _invisible_feature_names()


def _event_signal_class_names() -> tuple[str, ...]:
    """Ordered EVENT/signal names matching integer ``classification`` IDs."""
    event_info = getattr(global_config, "event_info", None)
    class_label = getattr(event_info, "class_label", {}) or {}
    event_labels = _dgpo_cfg_get(class_label, "EVENT", {}) or {}
    raw = _dgpo_cfg_get(event_labels, "signal", []) or []
    raw_values = list(raw)
    if raw_values and not isinstance(raw_values[0], str):
        raw_values = list(raw_values[0])
    return tuple(str(name) for name in raw_values)


def _variance_regularization_config(dg_cfg: Any | None = None) -> Any | None:
    """Optional anti-shrink regularization block under ``dgpo.variance_regularization``."""
    cfg = dg_cfg if dg_cfg is not None else getattr(global_config, "dgpo", None)
    return _dgpo_cfg_get(cfg, "variance_regularization", None)


def _variance_regularization_enabled(dg_cfg: Any | None = None) -> bool:
    """Whether batch-level std matching regularization is enabled."""
    block = _variance_regularization_config(dg_cfg)
    return bool(_dgpo_cfg_get(block, "enabled", False))


def _variance_regularization_weight(dg_cfg: Any | None = None) -> float:
    """Scalar weight for the anti-shrink regularizer."""
    block = _variance_regularization_config(dg_cfg)
    return max(0.0, float(_dgpo_cfg_get(block, "weight", 0.0)))


def _variance_regularization_feature_names(dg_cfg: Any | None = None) -> tuple[str, ...]:
    """Target features for anti-shrink regularization.

    Default: inspect ``event_info.yaml`` via ``event_info.invisible_feature_names`` and keep
    the angular-like entries we actually care about. This makes ``eta/phi`` and ``theta/phi``
    layouts both work without hard-coding one schema.
    """
    block = _variance_regularization_config(dg_cfg)
    raw = _dgpo_cfg_get(block, "features", None)
    if raw:
        return tuple(str(name) for name in raw)
    event_features = _invisible_feature_names()
    preferred = tuple(
        str(name) for name in event_features
        if str(name) in {"eta", "theta", "phi"}
    )
    return preferred if preferred else event_features


def _named_invisible_feature_tensors(
    kin: Tensor,
    *,
    cartesian: bool,
    feature_names: tuple[str, ...],
) -> dict[str, Tensor]:
    """Expose invisible kinematics as named tensors in the current feature space."""
    if int(kin.shape[-1]) <= 0:
        return {}
    if cartesian:
        if int(kin.shape[-1]) < 3:
            return {}
        log_pt, eta, phi = cartesian_to_log_pt_eta_phi(
            kin[..., 0],
            kin[..., 1],
            kin[..., 2],
        )
        return {
            "log_pt": log_pt,
            "pt": torch.expm1(log_pt),
            "eta": eta,
            "phi": phi,
            "px": kin[..., 0],
            "py": kin[..., 1],
            "pz": kin[..., 2],
        }
    out: dict[str, Tensor] = {}
    max_features = min(len(feature_names), int(kin.shape[-1]))
    for index, name in enumerate(feature_names[:max_features]):
        out[str(name)] = kin[..., index]
    return out


def _masked_batch_std(values: Tensor, mask: Tensor, eps: float = 1.0e-8) -> Tensor:
    """Population std over valid batch slots; ``mask`` is 0/1 with the same broadcast shape."""
    weights = mask.to(device=values.device, dtype=values.dtype)
    count = weights.sum().clamp(min=1.0)
    mean = (values * weights).sum() / count
    var = ((values - mean).pow(2) * weights).sum() / count
    return torch.sqrt(var.clamp(min=0.0) + float(eps))


def _variance_matching_penalty(
    pred_phys: Tensor,
    truth_phys: Tensor,
    valid_mask: Tensor,
    *,
    cartesian: bool,
    feature_names: tuple[str, ...],
    selected_features: tuple[str, ...],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Small anti-shrink penalty using relative shrinkage vs truth std."""
    zero = pred_phys.new_zeros(())
    diag: dict[str, Tensor] = {
        "train/regularization/variance/active": pred_phys.new_tensor(0.0, dtype=torch.float64),
        "train/regularization/variance/active_features": pred_phys.new_tensor(0.0, dtype=torch.float64),
        "train/regularization/variance/raw": zero.detach(),
    }
    if not selected_features:
        return zero, diag

    pred_named = _named_invisible_feature_tensors(
        pred_phys,
        cartesian=cartesian,
        feature_names=feature_names,
    )
    truth_named = _named_invisible_feature_tensors(
        truth_phys,
        cartesian=cartesian,
        feature_names=feature_names,
    )
    mask = valid_mask.squeeze(-1) if int(valid_mask.dim()) == int(pred_phys.dim()) else valid_mask
    penalties: list[Tensor] = []
    active_features = 0
    valid_count = float(mask.sum().detach().cpu())

    for feature_name in selected_features:
        prefix = f"train/regularization/variance/{feature_name}"
        pred_feature = pred_named.get(feature_name)
        truth_feature = truth_named.get(feature_name)
        diag[f"{prefix}/active"] = pred_phys.new_tensor(0.0, dtype=torch.float64)
        diag[f"{prefix}/count"] = pred_phys.new_tensor(valid_count, dtype=torch.float64)
        if pred_feature is None or truth_feature is None or valid_count < 2.0:
            diag[f"{prefix}/std_truth"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/std_pred"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/std_delta_ratio"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/std_gap"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            diag[f"{prefix}/penalty"] = pred_phys.new_tensor(float("nan"), dtype=torch.float64)
            continue
        std_truth = _masked_batch_std(truth_feature.detach(), mask)
        std_pred = _masked_batch_std(pred_feature, mask)
        std_scale = std_truth.detach().clamp(min=1.0e-8)
        std_delta_ratio = (std_pred - std_truth) / std_scale
        std_gap = torch.relu(-std_delta_ratio)
        penalty_feature = std_gap.pow(2)
        penalties.append(penalty_feature)
        active_features += 1
        diag[f"{prefix}/active"] = pred_phys.new_tensor(1.0, dtype=torch.float64)
        diag[f"{prefix}/std_truth"] = std_truth.detach().to(dtype=torch.float64)
        diag[f"{prefix}/std_pred"] = std_pred.detach().to(dtype=torch.float64)
        diag[f"{prefix}/std_delta_ratio"] = std_delta_ratio.detach().to(dtype=torch.float64)
        diag[f"{prefix}/std_gap"] = std_gap.detach().to(dtype=torch.float64)
        diag[f"{prefix}/penalty"] = penalty_feature.detach().to(dtype=torch.float64)

    if penalties:
        raw = torch.stack(penalties).mean()
        diag["train/regularization/variance/active"] = pred_phys.new_tensor(1.0, dtype=torch.float64)
        diag["train/regularization/variance/active_features"] = pred_phys.new_tensor(
            float(active_features), dtype=torch.float64
        )
        diag["train/regularization/variance/raw"] = raw.detach().to(dtype=torch.float64)
        return raw, diag
    return zero, diag


def _generation_special_bin_edges(feature_name: str) -> np.ndarray | None:
    """Mirror EveNet ``Generation-Binning`` lookup for ``neutrino-{feature}``."""
    metrics_cfg = getattr(global_config.options, "Metrics", None)
    if metrics_cfg is None:
        return None
    bins_cfg = metrics_cfg.get("Generation-Binning", {})
    raw = bins_cfg.get(f"neutrino-{feature_name}")
    if raw is None or len(raw) != 3:
        return None
    nbins, lo, hi = raw
    try:
        return np.linspace(float(lo), float(hi), int(nbins))
    except (TypeError, ValueError):
        return None


def _available_truth_pred_features(
    arrays: Mapping[str, np.ndarray],
    feature_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Feature names that have non-empty truth/pred arrays for 2D truth-vs-pred plots."""
    available: list[str] = []
    for feature_name in feature_names:
        truth = np.asarray(
            arrays.get(f"{feature_name}_truth", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        pred = np.asarray(
            arrays.get(f"{feature_name}_pred", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if min(truth.size, pred.size) == 0:
            continue
        if not np.isfinite(truth).any() or not np.isfinite(pred).any():
            continue
        available.append(str(feature_name))
    return tuple(available)


def _supports_legacy_invisible_kinematics(*, cartesian: bool, feature_dim: int | None = None) -> bool:
    """Whether legacy ``(log_pt, eta, phi)`` / Cartesian diagnostics are valid."""
    if cartesian:
        return feature_dim is None or int(feature_dim) >= 3
    feature_names = _invisible_feature_names()
    if len(feature_names) < 3:
        return False
    if tuple(feature_names[:3]) != ("log_pt", "eta", "phi"):
        return False
    return feature_dim is None or int(feature_dim) >= 3


def _validation_winrate_enabled(*, compute_winrate: bool, cartesian: bool, feature_dim: int | None = None) -> bool:
    """Validation win-rate is available whenever the extra reference rollout is enabled."""
    del cartesian, feature_dim
    return bool(compute_winrate)


def _validation_profile_feature_names(*, cartesian: bool) -> tuple[str, ...]:
    """Validation residual-profile features derived from ``event_info.yaml``."""
    if _supports_legacy_invisible_kinematics(cartesian=cartesian):
        return ("pt", "eta")
    feature_names = _invisible_feature_names()
    return feature_names if feature_names else ("feature_0",)


def _delta_selection_profiles_figure(
    truth_all: np.ndarray,
    delta_all: np.ndarray,
    truth_best: np.ndarray,
    delta_best: np.ndarray,
    truth_oracle: np.ndarray | None = None,
    delta_oracle: np.ndarray | None = None,
    *,
    profile_name: str,
    title: str,
) -> Any:
    """Profile plot comparing rollout-all, reward-best, and optional variable-oracle residuals."""
    import wandb

    x_label, y_label, display = _profile_axis_labels(profile_name)
    bin_edges = _profile_bin_edges(
        profile_name,
        [truth_all, truth_best] + ([] if truth_oracle is None else [truth_oracle]),
    )
    centers, mean_all, err_all, _ = _binned_delta_profile(
        truth_all, delta_all, bin_edges=bin_edges
    )
    _, mean_best, err_best, counts = _binned_delta_profile(
        truth_best, delta_best, bin_edges=bin_edges
    )
    mean_oracle = err_oracle = None
    if truth_oracle is not None and delta_oracle is not None:
        _, mean_oracle, err_oracle, _ = _binned_delta_profile(
            truth_oracle, delta_oracle, bin_edges=bin_edges
        )
    best_gap = mean_best - mean_all
    oracle_gap = mean_oracle - mean_all if mean_oracle is not None else None

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for means, errs, fmt, label, color in (
        (mean_all, err_all, "o-", "all rollout candidates", "#1f77b4"),
        (mean_best, err_best, "s-", "reward-best candidates", "#d62728"),
    ):
        keep = np.isfinite(means)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                means[keep],
                yerr=errs[keep],
                fmt=fmt,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )
    if mean_oracle is not None and err_oracle is not None:
        keep = np.isfinite(mean_oracle)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                mean_oracle[keep],
                yerr=err_oracle[keep],
                fmt="^-",
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color="#2ca02c",
                label=f"{display}-oracle-best candidates",
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    gap_series = (
        (best_gap, "o-", "reward-best - all", "#d62728"),
        (oracle_gap, "^-", f"{display}-oracle - all", "#2ca02c"),
    )
    for gap, fmt, label, color in gap_series:
        if gap is None:
            continue
        keep = np.isfinite(gap)
        if np.any(keep):
            ax_gap.plot(
                centers[keep],
                gap[keep],
                fmt,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(x_label)
    ax_gap.set_ylabel("Selection gap")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


_DIAG_PROFILE_NAMES = ("pt", "eta", "phi", "px", "py", "pz")


def _diag_profile_raw_key(profile_name: str, suffix: str) -> str:
    """Internal metric key for raw arrays used by accumulated W&B profile images."""
    return f"_diag_{profile_name}_profile_{suffix}"


def _diag_profile_log_key(profile_name: str, *, accumulated: bool = False) -> str:
    """Public W&B image key for a binned residual profile."""
    suffix = "_accumulated" if accumulated else ""
    return (
        f"diagnostics/reward_hacking/profile/"
        f"{profile_name}_delta_vs_truth_{profile_name}{suffix}"
    )


def _diag_profile_title(profile_name: str, *, accumulated_batches: int | None = None) -> str:
    """Human-readable title for a binned residual profile."""
    _x_label, _y_label, display = _profile_axis_labels(profile_name)
    title = f"Reward selection {display} bias vs truth {display}"
    if accumulated_batches is not None:
        title += f" ({accumulated_batches} train batches)"
    return title


def _align_truth_tensor_to_delta(truth: Tensor, delta: Tensor) -> Tensor:
    """Expand cached truth tensors from ``(1, B, S)`` to the candidate shape when needed."""
    if truth.shape == delta.shape:
        return truth
    if truth.dim() == delta.dim() and truth.shape[0] == 1 and truth.shape[1:] == delta.shape[1:]:
        return truth.expand_as(delta)
    return truth


def _finite_profile_numpy(truth: Tensor, delta: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return finite paired truth and residual arrays for plotting."""
    mask = torch.isfinite(truth) & torch.isfinite(delta)
    return (
        truth[mask].detach().float().cpu().numpy(),
        delta[mask].detach().float().cpu().numpy(),
    )


def _pt_delta_prefix_vs_full_figure(
    truth_pt_reward_prefix: np.ndarray,
    delta_pt_reward_prefix: np.ndarray,
    truth_pt_reward_full: np.ndarray,
    delta_pt_reward_full: np.ndarray,
    truth_pt_oracle_prefix: np.ndarray,
    delta_pt_oracle_prefix: np.ndarray,
    truth_pt_oracle_full: np.ndarray,
    delta_pt_oracle_full: np.ndarray,
    *,
    prefix_k: int,
    full_k: int,
    title: str,
) -> Any:
    """Compare first-prefix-K vs full-K selection for reward-best and pT-oracle."""
    import wandb

    centers, reward_prefix, reward_prefix_err, counts = _binned_delta_profile(
        truth_pt_reward_prefix, delta_pt_reward_prefix
    )
    _, reward_full, reward_full_err, _ = _binned_delta_profile(
        truth_pt_reward_full, delta_pt_reward_full
    )
    _, oracle_prefix, oracle_prefix_err, _ = _binned_delta_profile(
        truth_pt_oracle_prefix, delta_pt_oracle_prefix
    )
    _, oracle_full, oracle_full_err, _ = _binned_delta_profile(
        truth_pt_oracle_full, delta_pt_oracle_full
    )

    fig, (ax, ax_gap) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    series = (
        (reward_prefix, reward_prefix_err, "o-", f"reward-best first {prefix_k}", "#ff7f0e"),
        (reward_full, reward_full_err, "s-", f"reward-best full {full_k}", "#d62728"),
        (oracle_prefix, oracle_prefix_err, "^-", f"pT-oracle first {prefix_k}", "#2ca02c"),
        (oracle_full, oracle_full_err, "v-", f"pT-oracle full {full_k}", "#1f77b4"),
    )
    for means, errs, fmt, label, color in series:
        keep = np.isfinite(means)
        if np.any(keep):
            ax.errorbar(
                centers[keep],
                means[keep],
                yerr=errs[keep],
                fmt=fmt,
                linewidth=1.8,
                markersize=4,
                capsize=2,
                color=color,
                label=label,
            )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylabel(r"Mean $\Delta p_T$ [GeV]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    reward_gain = reward_full - reward_prefix
    oracle_gain = oracle_full - oracle_prefix
    for gain, fmt, label, color in (
        (reward_gain, "o-", f"reward full {full_k} - first {prefix_k}", "#d62728"),
        (oracle_gain, "^-", f"oracle full {full_k} - first {prefix_k}", "#2ca02c"),
    ):
        keep = np.isfinite(gain)
        if np.any(keep):
            ax_gap.plot(
                centers[keep],
                gain[keep],
                fmt,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
    ax_gap.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_gap.set_xlabel(r"$p_T^{truth}$ [GeV]")
    ax_gap.set_ylabel(r"Full - prefix [GeV]")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _projection_metric_finite(out: dict[str, Any], key: str) -> float | None:
    """Return a finite float from a metrics dict, or ``None``."""
    val = out.get(key)
    if val is None:
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


def _projection_labeled_bar_figure(
    *,
    title: str,
    series: list[tuple[str, float | None, str]],
    ylabel: str,
    reference_lines: list[tuple[str, float, str, str]] | None = None,
) -> Any | None:
    """Horizontal bar chart for projection estimator comparison (``wandb.Image``)."""
    import wandb

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for label, val, color in series:
        if val is None or not math.isfinite(float(val)):
            continue
        labels.append(label)
        values.append(float(val))
        colors.append(color)
    if not labels:
        return None

    fig_h = max(3.2, 0.55 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    y_pos = list(range(len(labels)))
    ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y_pos, labels=labels)
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.35)
    if reference_lines:
        for ref_label, ref_val, ref_color, ref_style in reference_lines:
            if not math.isfinite(float(ref_val)):
                continue
            ax.axvline(
                float(ref_val),
                color=ref_color,
                linewidth=1.2,
                linestyle=ref_style,
                alpha=0.9,
                label=ref_label,
            )
    if reference_lines:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _projection_violation_compare_figure(out: dict[str, Any]) -> Any | None:
    """Compare violation margins ``v = C - epsilon`` (lambda driver vs baselines)."""
    return _projection_labeled_bar_figure(
        title="Projection violation (v = C - epsilon)",
        ylabel="Violation margin v",
        series=[
            (
                "v_selected (lambda)",
                _projection_metric_finite(out, "projection/v_selected"),
                "#C44E52",
            ),
            (
                "v_linear (ref)",
                _projection_metric_finite(out, "projection/v_linear"),
                "#4C72B0",
            ),
        ],
        reference_lines=[("feasible (v=0)", 0.0, "black", ":")],
    )


def _build_projection_wandb_panel_metrics(
    out: dict[str, Any],
    plot_names: set[str],
) -> dict[str, Any]:
    """Optional W&B media panels for projection estimator comparison."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return {}
    active = out.get("projection/active")
    if active is None or not math.isfinite(float(active)) or float(active) != 1.0:
        return {}
    panels: dict[str, Any] = {}
    if "projection_violation_compare" in plot_names:
        try:
            img = _projection_violation_compare_figure(out)
            if img is not None:
                panels["projection/panel/violation_compare"] = img
        except Exception as exc:
            _log.warning("[DGPO] projection violation panel failed: %s", exc)
    return panels


@torch.no_grad()
def _build_reference_bias_metrics(
    candidates: Tensor,
    batch: dict[str, Any],
    *,
    cartesian: bool,
    log_distribution: bool = False,
    diagnostic_plot_names: set[str] | None = None,
) -> dict[str, Any]:
    """Diagnostics for the rollout/reference policy's raw kinematic bias vs truth."""
    out: dict[str, Any] = {}
    K, B, N_nu, F = candidates.shape
    S = min(2, N_nu)
    if S == 0:
        return out

    if "x_invisible_mask" in batch:
        mask = batch["x_invisible_mask"].to(device=candidates.device, dtype=candidates.dtype)
        if mask.dim() == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.dim() != 2:
            return out
        slot_valid = mask[:, :S] > 0
    else:
        slot_valid = torch.ones(B, S, device=candidates.device, dtype=torch.bool)

    event_valid = get_event_valid_mask(batch, B, candidates.device, candidates.dtype) > 0
    valid_kbs = (slot_valid & event_valid.unsqueeze(-1)).unsqueeze(0).expand(K, B, S)

    if cartesian:
        truth = batch.get("x_invisible_cartesian")
        if not isinstance(truth, Tensor) or truth.dim() != 3 or truth.shape[0] != B:
            return out
        truth_xyz = truth[:, :S, :3].to(device=candidates.device, dtype=candidates.dtype)
        cand_xyz = candidates[:, :, :S, :3]
        truth_log_pt, truth_eta, truth_phi = cartesian_to_log_pt_eta_phi(
            truth_xyz[..., 0],
            truth_xyz[..., 1],
            truth_xyz[..., 2],
        )
        cand_log_pt, cand_eta, cand_phi = cartesian_to_log_pt_eta_phi(
            cand_xyz[..., 0],
            cand_xyz[..., 1],
            cand_xyz[..., 2],
        )
    else:
        truth = batch.get("x_invisible")
        if (
            not isinstance(truth, Tensor)
            or truth.dim() != 3
            or truth.shape[0] != B
            or F < 3
        ):
            return out
        truth_kin = truth[:, :S, :3].to(device=candidates.device, dtype=candidates.dtype)
        cand_kin = candidates[:, :, :S, :3]
        truth_log_pt, truth_eta, truth_phi = truth_kin.unbind(dim=-1)
        cand_log_pt, cand_eta, cand_phi = cand_kin.unbind(dim=-1)

    truth_pt = torch.expm1(truth_log_pt.clamp(-10.0, 10.0)).unsqueeze(0)
    cand_pt = torch.expm1(cand_log_pt.clamp(-10.0, 10.0))
    residuals = {
        "pt": cand_pt - truth_pt,
        "rel_pt": (cand_pt - truth_pt) / truth_pt.clamp(min=1e-6),
        "eta": cand_eta - truth_eta.unsqueeze(0),
        "phi": torch.atan2(
            torch.sin(cand_phi - truth_phi.unsqueeze(0)),
            torch.cos(cand_phi - truth_phi.unsqueeze(0)),
        ),
    }

    finite_residuals: dict[str, Tensor] = {}
    for name, tensor in residuals.items():
        values = tensor[valid_kbs]
        values = values[torch.isfinite(values)]
        finite_residuals[name] = values
        if name == "rel_pt":
            mean_key = f"diagnostics/reference_bias/all/{name}/mean"
            abs_mean_key = f"diagnostics/reference_bias/all/{name}/abs_mean"
        else:
            mean_key = f"diagnostics/reference_bias/all/{name}/delta_mean"
            abs_mean_key = f"diagnostics/reference_bias/all/{name}/delta_abs_mean"
        if values.numel() > 0:
            out[mean_key] = float(values.mean().detach().cpu())
            out[abs_mean_key] = float(values.abs().mean().detach().cpu())
        else:
            out[mean_key] = float("nan")
            out[abs_mean_key] = float("nan")

    plot_names = diagnostic_plot_names or set()
    if log_distribution and "rel_pt_dist" in plot_names:
        rel_pt = finite_residuals.get("rel_pt")
        if rel_pt is not None:
            try:
                import wandb  # noqa: F401

                out["diagnostics/reference_bias/dist/rel_pt"] = (
                    _single_rel_pt_distribution_figure(
                        rel_pt.detach().float().cpu().numpy(),
                        title="Reference / frozen-rollout relative pT bias",
                        label="rollout candidates",
                    )
                )
            except Exception:
                pass
        delta_pt = residuals["pt"][valid_kbs]
        truth_pt_rep = truth_pt.expand(K, B, S)[valid_kbs]
        profile_mask = torch.isfinite(delta_pt) & torch.isfinite(truth_pt_rep)
        delta_pt = delta_pt[profile_mask]
        truth_pt_rep = truth_pt_rep[profile_mask]
        try:
            import wandb  # noqa: F401

            out["diagnostics/reference_bias/profile/pt_delta_vs_truth_pt"] = (
                _pt_delta_vs_truth_pt_figure(
                    truth_pt_rep.detach().float().cpu().numpy(),
                    delta_pt.detach().float().cpu().numpy(),
                    title="Reference / frozen-rollout pT bias vs truth pT",
                )
            )
        except Exception:
            pass
    return out



@torch.no_grad()
def build_reward_distribution_histograms(
    rewards: Tensor,
    valid_b: Tensor,
) -> dict[str, Any]:
    """Panel ``reward/dist``: overlapped 1D histograms for best / worst / median as ``wandb.Image``.

    Logs a **single** media key each time so the W&B Images panel shows one series with a
    **step slider** (same pattern as ``wandb.Image`` validation plots in ``evenet/``).
    """
    try:
        import wandb  # noqa: F401 — require package; figure built in _reward_dist_overlaid_figure
    except ImportError:
        return {}
    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        return {}
    rv = rewards[:, vb]
    best = rv.max(dim=0).values.detach().float().cpu().numpy()
    worst = rv.min(dim=0).values.detach().float().cpu().numpy()
    med = rv.median(dim=0).values.detach().float().cpu().numpy()
    return {"reward/dist/overlap": _reward_dist_overlaid_figure(best, worst, med)}


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a Ray/Lightning-style batch dict to ``device``."""
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _save_trainable_weights(model: torch.nn.Module) -> dict[str, Tensor]:
    """Buffer for EMA rollout swap (only parameters that participate in EMA shadow)."""
    core = _unwrap_core_evenet(model)
    return {n: p.data.clone() for n, p in core.named_parameters() if p.requires_grad}


def _maybe_install_ema_for_generation(
    ema: Any | None,
    model: torch.nn.Module,
    core: Any,
) -> dict[str, Tensor]:
    """Install an EMA shadow only when requested by ``EMA.use_for_generation``."""
    ema_cfg = global_config.options.Training.get("EMA", None) or {}
    if ema is None or not generation_uses_ema_shadow(ema_cfg):
        return {}
    buffer = _save_trainable_weights(model)
    ema.copy_to(core)
    return buffer


def _restore_trainable_weights(model: torch.nn.Module, buf: dict[str, Tensor]) -> None:
    core = _unwrap_core_evenet(model)
    for n, p in core.named_parameters():
        if n in buf:
            p.data.copy_(buf[n])


def _scales_from_normalization(normalization_dict: dict[str, Any] | None) -> dict[str, float]:
    """Build the 6 per-component scales from ``normalization.pt['invisible_cartesian_std']``.

    Reads shape-``(3,)`` std in order ``[px, py, pz]`` (computed sample-wise at
    preprocessing over both ``nu1`` and ``nu2`` slots) and applies the same px/py/pz
    std to both neutrinos. Raises if the key is absent — re-run preprocessing first.
    """
    if normalization_dict is None or "invisible_cartesian_std" not in normalization_dict:
        raise ValueError(
            "component_normalized_truth_distance requires 'invisible_cartesian_std' in "
            "normalization.pt (shape (3,) [px, py, pz]). Re-run preprocessing so the "
            "Cartesian std is saved."
        )
    std_t = normalization_dict["invisible_cartesian_std"]["Source"]
    std = std_t.detach().cpu().tolist() if hasattr(std_t, "detach") else list(std_t)
    if len(std) < 3:
        raise ValueError(
            f"invisible_cartesian_std must have 3 entries [px, py, pz], got {len(std)}"
        )
    return {
        "nu1_px": float(std[0]), "nu1_py": float(std[1]), "nu1_pz": float(std[2]),
        "nu2_px": float(std[0]), "nu2_py": float(std[1]), "nu2_pz": float(std[2]),
    }


def _reward_feature_names() -> tuple[str, ...] | None:
    rc = getattr(global_config, "reward_config", None)
    if rc is None:
        return None
    raw = getattr(rc, "feature_names", None)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def _reward_component_axis_pairs(component_names: tuple[str, ...] | list[str]) -> dict[str, tuple[str, str]]:
    """Pair ``nu1_*`` and ``nu2_*`` components by feature name."""
    names = set(str(name) for name in component_names)
    pairs: dict[str, tuple[str, str]] = {}
    for name in sorted(names):
        if not name.startswith("nu1_"):
            continue
        feature_name = name[4:]
        other = f"nu2_{feature_name}"
        if other in names:
            pairs[feature_name] = (name, other)
    return pairs


def build_reward_aggregator(
    model: torch.nn.Module,
    device: torch.device,
    normalization_dict: dict[str, Any] | None = None,
) -> RewardAggregator:
    """Construct the configured DGPO reward from ``reward_config``."""
    rc = global_config.reward_config
    reward_type = str(getattr(rc, "type", "component_normalized_truth_distance")).strip().lower()
    cn = getattr(rc, "component_normalized", None)
    eps = float(getattr(cn, "eps", 1e-8)) if cn is not None else 1e-8
    weight = float(getattr(rc, "weight", 1.0))
    feature_names = _reward_feature_names()
    agg = RewardAggregator()
    if reward_type in {"omnifold", "omnifold_guided", "ztautau_omnifold"}:
        from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import (
            build_uninstalled_ztautau_omnifold_reward,
            load_ztautau_omnifold_reward,
        )

        block = _dgpo_cfg_get(rc, "omnifold", None)
        bundle_file = _dgpo_cfg_get(block, "bundle_file", None)
        backbone_checkpoint = _dgpo_cfg_get(block, "backbone_checkpoint", None)
        bootstrap_in_dgpo = bool(
            _dgpo_cfg_get(block, "bootstrap_in_dgpo", False)
        )
        candidate_consensus = _dgpo_cfg_get(
            block, "candidate_consensus", None
        )
        if not backbone_checkpoint or (not bootstrap_in_dgpo and not bundle_file):
            raise ValueError(
                "reward_config.type=omnifold requires omnifold.backbone_checkpoint "
                "and either bootstrap_in_dgpo=true or omnifold.bundle_file"
            )
        if normalization_dict is None:
            raise ValueError("OmniFold reward requires the EveNet normalization dictionary")
        if bootstrap_in_dgpo:
            adaptive_block = _dgpo_cfg_get(global_config.dgpo, "adaptive_omnifold", None)
            recal = _dgpo_cfg_get(adaptive_block, "recalibration", None)
            classifier_defaults = {
                "adapter_bottleneck": 16,
                "body_only_checkpoint": False,
                "train_layernorm": False,
                "train_encoder": False,
                "train_grouped_sequential_embedding": False,
                # Keep this in the explicit hand-off to the OmniFold builder.
                # The recalibration block is converted to a plain allow-listed
                # mapping here, so omitting the key silently freezes the
                # projector even when the YAML enables it.
                "train_invisible_projector": False,
                "train_angular_conditioning": False,
                "train_backbone": False,
                "train_last_pet_block": False,
                "asymmetric_attention": False,
                "periodic_pair_features": False,
                "topology_fourier_embedding": False,
                "topology_direct_logit": False,
                "topology_context_residual_scale": 1.0,
                "topology_max_harmonic": 1,
                "topology_include_theta_pair": False,
                "topology_theta_fourier": False,
                "topology_hidden_dim": 64,
                "topology_embedding_dim": 32,
                "topology_fusion_hidden_dim": 64,
                "topology_dropout": 0.15,
                "head_dropout": 0.1,
                "decoder_hidden_dim": 256,
                "decoder_layers": 2,
                "decoder_heads": 8,
            }
            classifier_config = {
                key: _dgpo_cfg_get(recal, key, default)
                for key, default in classifier_defaults.items()
            }
            if _dgpo_cfg_get(recal, "topology_conditioning", False):
                classifier_config["topology_conditioning"] = True
            if _dgpo_cfg_get(recal, "relation_token_count", 0):
                classifier_config["relation_token_count"] = int(_dgpo_cfg_get(recal, "relation_token_count", 0))
            if _dgpo_cfg_get(recal, "topology_pair_token", False):
                classifier_config["topology_pair_token"] = True
            if _dgpo_cfg_get(recal, "visible_pair_rest_frame", False):
                classifier_config["visible_pair_rest_frame"] = True
            reward = build_uninstalled_ztautau_omnifold_reward(
                backbone_checkpoint=backbone_checkpoint,
                training_config=global_config,
                normalization_dict=normalization_dict,
                device=device,
                classifier_config=classifier_config,
                candidate_consensus=candidate_consensus,
            )
            _log.info(
                "[DGPO/reward] OmniFold will fit and install its initial K=1 "
                "residual stack inside DGPO before the first policy update "
                "(classifier=%s).",
                "frozen-backbone PEFT bank",
            )
            agg.add(reward, weight)
            return agg
        expected_iterations = _dgpo_cfg_get(block, "expected_iterations", None)
        reward = load_ztautau_omnifold_reward(
            bundle_file=bundle_file,
            backbone_checkpoint=backbone_checkpoint,
            training_config=global_config,
            normalization_dict=normalization_dict,
            device=device,
            expected_iterations=(
                None if expected_iterations is None else int(expected_iterations)
            ),
            candidate_consensus=candidate_consensus,
        )
        _log.info(
            "[DGPO/reward] using frozen Ztautau OmniFold reward: iterations=%s "
            "fit_K=1 train_K=%s reference=%s bundle=%s weight=%.4g",
            reward.iterations,
            int(_dgpo_cfg_get(global_config.dgpo, "K", 1)),
            reward.policy_reference_sha256[:12],
            reward.artifact_path,
            weight,
        )
        agg.add(reward, weight)
        return agg
    if reward_type in {"calibration_magnitude", "physics_consistency", "ztautau_calibration_magnitude"}:
        _log.info(
            "[DGPO/reward] using calibration_magnitude reward for feature_names=%s.",
            feature_names,
        )
        agg.add(
            CalibrationMagnitudeReward(feature_names=feature_names),
            weight,
        )
        return agg

    if feature_names is not None:
        scales = build_feature_space_scales(
            normalization_dict,
            feature_names=feature_names,
        )
        _log.info(
            "[DGPO/reward] feature-space scales from normalization.pt invisible_std for %s: %s",
            feature_names,
            {k: round(v, 4) for k, v in scales.items()},
        )
    else:
        scales = _scales_from_normalization(normalization_dict)
        _log.info(
            "[DGPO/reward] component_normalized scales from normalization.pt "
            "invisible_cartesian_std [px, py, pz]: %s",
            {k: round(v, 4) for k, v in scales.items()},
        )
    agg.add(
        ComponentNormalizedTruthDistanceReward(
            scales,
            cartesian=_truth_generation_cartesian(),
            eps=eps,
            feature_names=feature_names,
        ),
        weight,
    )
    return agg


def _normalize_candidates_for_policy(
    model: torch.nn.Module,
    c_phys: Tensor,
    inv_mask: Tensor,
) -> Tensor:
    """Map denormalized DDIM output to the normalized space for ``predict_diffusion_vector``.

    Matches training / DDIM: raw invisible is padded to ``sequential_input_dim``, then
    ``invisible_normalizer`` runs on that width. ``predict_diffusion_vector`` (neutrino) then
    applies ``F.pad(..., invisible_padding)`` itself, so ``noise_x`` must be only the first
    ``invisible_input_dim`` channels (same width as ``DDIMSampler`` uses from ``x_invisible``),
    not the full padded-normalized tensor — otherwise features become ``sequential + padding``
    and ``torch.cat`` with jets fails (e.g. 7 vs 11).
    """
    # c_phys: (R, N_nu, F_phys)
    pad = int(getattr(model, "invisible_padding", 0))
    m = inv_mask.unsqueeze(-1).to(dtype=c_phys.dtype)
    x = c_phys
    if pad > 0:
        x = F.pad(x, (0, pad))
    full_norm = model.invisible_normalizer(x=x, mask=m)
    inv_in = int(getattr(model, "invisible_input_dim", full_norm.shape[-1]))
    return full_norm[..., :inv_in]


@dataclass(frozen=True)
class _PolicyEvaluationInputs:
    """Batch tensors that stay constant across a batch's policy-eval draws."""

    batch_size: int
    num_candidates: int
    num_timesteps: int
    candidates_norm: Tensor
    batch_rep: dict[str, Any]
    noise_mask_rep: Tensor


@dataclass(frozen=True)
class _ReferenceTrustProbe:
    """Frozen policy-evaluation rows reused for strict post-step trust checks."""

    x_t: Tensor
    t_rep: Tensor
    noise_mask_rep: Tensor
    ref_v: Tensor
    batch_rep: dict[str, Any]
    reference_loss_sum: Tensor
    reference_loss_count: int
    path_kl_normalizer: float | None = None


def _capture_reference_trust_probe(
    *,
    x_t: Tensor,
    t_rep: Tensor,
    noise_mask_rep: Tensor,
    ref_v: Tensor,
    batch_rep: Mapping[str, Any],
    L_ref_2d: Tensor,
    K: int,
    local_batch_size: int,
    max_events: int,
    path_kl_normalizer: float | None = None,
) -> _ReferenceTrustProbe:
    """Keep a bounded, graph-free subset of one existing shared-noise draw."""

    local_B = int(local_batch_size)
    probe_B = min(local_B, max(1, int(max_events)))
    rows = torch.cat(
        [
            torch.arange(
                candidate * local_B,
                candidate * local_B + probe_B,
                device=x_t.device,
                dtype=torch.long,
            )
            for candidate in range(int(K))
        ]
    )
    expected_rows = int(K) * local_B

    def _select(value: Any) -> Any:
        if (
            isinstance(value, Tensor)
            and value.ndim > 0
            and int(value.shape[0]) == expected_rows
        ):
            return value.index_select(0, rows).detach()
        return value.detach() if isinstance(value, Tensor) else value

    reference_loss = L_ref_2d.reshape(int(K), local_B)[:, :probe_B]
    return _ReferenceTrustProbe(
        x_t=x_t.index_select(0, rows).detach(),
        t_rep=t_rep.index_select(0, rows).detach(),
        noise_mask_rep=noise_mask_rep.index_select(0, rows).detach(),
        ref_v=ref_v.index_select(0, rows).detach(),
        batch_rep={key: _select(value) for key, value in batch_rep.items()},
        reference_loss_sum=reference_loss.detach().to(torch.float64).sum(),
        reference_loss_count=int(reference_loss.numel()),
        path_kl_normalizer=(
            None
            if path_kl_normalizer is None
            else float(path_kl_normalizer)
        ),
    )


def _reference_trust_probe_to_payload(
    probe: _ReferenceTrustProbe,
) -> dict[str, Any]:
    """Serialize a fixed trust probe as CPU tensors for checkpoints/broadcast."""

    def _cpu(value: Any) -> Any:
        return value.detach().cpu() if isinstance(value, Tensor) else value

    return {
        "x_t": _cpu(probe.x_t),
        "t_rep": _cpu(probe.t_rep),
        "noise_mask_rep": _cpu(probe.noise_mask_rep),
        "ref_v": _cpu(probe.ref_v),
        "batch_rep": {key: _cpu(value) for key, value in probe.batch_rep.items()},
        "reference_loss_sum": _cpu(probe.reference_loss_sum),
        "reference_loss_count": int(probe.reference_loss_count),
        "path_kl_normalizer": probe.path_kl_normalizer,
    }


def _reference_trust_probe_from_payload(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
) -> _ReferenceTrustProbe:
    """Restore a checkpointed trust probe on the current worker device."""

    required = {
        "x_t",
        "t_rep",
        "noise_mask_rep",
        "ref_v",
        "batch_rep",
        "reference_loss_sum",
        "reference_loss_count",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"reference trust probe payload missing keys: {missing}")

    def _device(value: Any) -> Any:
        return value.to(device, non_blocking=True) if isinstance(value, Tensor) else value

    batch_payload = payload["batch_rep"]
    if not isinstance(batch_payload, Mapping):
        raise TypeError("reference trust probe batch_rep must be a mapping")
    return _ReferenceTrustProbe(
        x_t=_device(payload["x_t"]),
        t_rep=_device(payload["t_rep"]),
        noise_mask_rep=_device(payload["noise_mask_rep"]),
        ref_v=_device(payload["ref_v"]),
        batch_rep={key: _device(value) for key, value in batch_payload.items()},
        reference_loss_sum=_device(payload["reference_loss_sum"]),
        reference_loss_count=int(payload["reference_loss_count"]),
        path_kl_normalizer=(
            None
            if payload.get("path_kl_normalizer") is None
            else float(payload["path_kl_normalizer"])
        ),
    )


_PER_RANK_TRUST_PROBE_FORMAT = "per_rank_v1"


def _gather_reference_trust_probe(
    probe: _ReferenceTrustProbe,
    *,
    device: torch.device,
    world_size: int,
) -> tuple[_ReferenceTrustProbe, dict[str, Any]]:
    """Keep each rank's distinct probe and checkpoint all rank-local shards.

    The returned live probe remains local to the current rank.  The checkpoint
    payload contains one CPU shard per rank so a resumed job with the same
    world size restores exactly the same global probe without duplicating rank
    0's event conditions across every worker.
    """

    local_payload = _reference_trust_probe_to_payload(probe)
    if world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed fixed trust probe requires an initialized process group"
            )
        actual_world_size = int(dist.get_world_size())
        if actual_world_size != int(world_size):
            raise RuntimeError(
                "fixed trust probe world-size mismatch: "
                f"configured={world_size}, distributed={actual_world_size}"
            )
        rank_payloads: list[Any] = [None] * actual_world_size
        dist.all_gather_object(rank_payloads, local_payload)
    else:
        rank_payloads = [local_payload]

    normalized_payloads: list[dict[str, Any]] = []
    for rank, payload in enumerate(rank_payloads):
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                f"fixed trust probe gather returned no payload for rank {rank}"
            )
        normalized_payloads.append(dict(payload))
    checkpoint_payload: dict[str, Any] = {
        "format": _PER_RANK_TRUST_PROBE_FORMAT,
        "world_size": int(world_size),
        "per_rank": normalized_payloads,
    }
    return probe, checkpoint_payload


def _restore_reference_trust_probe(
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    world_size: int,
) -> _ReferenceTrustProbe:
    """Restore this worker's shard, with compatibility for legacy payloads."""

    if payload.get("format") != _PER_RANK_TRUST_PROBE_FORMAT:
        # Legacy checkpoints stored one rank-0 probe shared by every worker.
        return _reference_trust_probe_from_payload(payload, device=device)

    saved_world_size = int(payload.get("world_size", -1))
    if saved_world_size != int(world_size):
        raise RuntimeError(
            "cannot restore a fixed per-rank trust probe with a different world "
            f"size: checkpoint={saved_world_size}, current={world_size}"
        )
    rank = int(dist.get_rank()) if world_size > 1 and dist.is_initialized() else 0
    rank_payloads = payload.get("per_rank")
    if not isinstance(rank_payloads, list) or len(rank_payloads) != saved_world_size:
        raise ValueError("fixed trust probe checkpoint has invalid per-rank shards")
    local_payload = rank_payloads[rank]
    if not isinstance(local_payload, Mapping):
        raise TypeError(f"fixed trust probe shard for rank {rank} is not a mapping")
    return _reference_trust_probe_from_payload(local_payload, device=device)


@torch.no_grad()
def _measure_reference_trust_probe(
    model: torch.nn.Module,
    probe: _ReferenceTrustProbe,
    *,
    world_size: int,
    distance: str = "velocity_mse_ratio",
) -> tuple[float, float, float]:
    """Return global ``(distance, velocity_mse, reference_loss)`` on fixed rows."""

    distance_kind = str(distance).strip().lower()
    if distance_kind not in {"velocity_mse_ratio", "vp_path_kl"}:
        raise ValueError(
            "reference trust probe distance must be velocity_mse_ratio or "
            "vp_path_kl"
        )
    if distance_kind == "vp_path_kl" and probe.path_kl_normalizer is None:
        raise ValueError("vp_path_kl trust probe is missing its normalizer")

    was_training = bool(model.training)
    model.eval()
    try:
        if isinstance(model, DDP):
            model_v = model(
                probe.x_t,
                probe.batch_rep,
                probe.t_rep,
                probe.noise_mask_rep,
            )
        else:
            model_v = model.predict_diffusion_vector(
                noise_x=probe.x_t,
                cond_x=probe.batch_rep,
                time=probe.t_rep,
                mode="neutrino",
                noise_mask=probe.noise_mask_rep,
            )
    finally:
        model.train(was_training)
    if model_v.shape != probe.ref_v.shape:
        raise RuntimeError(
            "strict trust probe policy/reference shapes differ: "
            f"{tuple(model_v.shape)} vs {tuple(probe.ref_v.shape)}"
        )
    mask = probe.noise_mask_rep.expand_as(model_v).to(dtype=model_v.dtype)
    totals = torch.stack(
        (
            ((model_v - probe.ref_v).pow(2) * mask).sum().to(torch.float64),
            mask.sum().to(torch.float64),
            probe.reference_loss_sum.to(device=model_v.device, dtype=torch.float64),
            torch.tensor(
                float(probe.reference_loss_count),
                device=model_v.device,
                dtype=torch.float64,
            ),
            torch.tensor(
                float(model_v.shape[0]),
                device=model_v.device,
                dtype=torch.float64,
            ),
        )
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    velocity_mse = totals[0] / totals[1].clamp_min(1.0e-12)
    reference_loss = totals[2] / totals[3].clamp_min(1.0)
    distance_value = (
        0.5
        * float(probe.path_kl_normalizer)
        * totals[0]
        / totals[4].clamp_min(1.0)
        if distance_kind == "vp_path_kl"
        else velocity_mse / reference_loss.clamp_min(1.0e-12)
    )
    return (
        float(distance_value.cpu()),
        float(velocity_mse.cpu()),
        float(reference_loss.cpu()),
    )


@torch.no_grad()
def _assign_interpolated_trainable_params_(
    model: torch.nn.Module,
    theta_old: Mapping[str, Tensor],
    theta_candidate: Mapping[str, Tensor],
    fraction: float,
) -> None:
    """Assign ``old + fraction * (candidate - old)`` to trainable parameters."""

    fraction_f = float(fraction)
    if not math.isfinite(fraction_f) or not 0.0 <= fraction_f <= 1.0:
        raise ValueError("trust interpolation fraction must lie in [0, 1]")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in theta_old or name not in theta_candidate:
            raise KeyError(f"missing trainable parameter {name!r} in trust snapshot")
        old = theta_old[name].to(device=parameter.device, dtype=parameter.dtype)
        candidate = theta_candidate[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        parameter.data.copy_(old + fraction_f * (candidate - old))


def _snapshot_signed_trainable_direction(
    model: torch.nn.Module,
    anchor_model: torch.nn.Module,
) -> tuple[torch.nn.Module, dict[str, Tensor], dict[str, Tensor]]:
    """Snapshot live and anchor tensors for one diagnostic policy direction."""

    policy_core = unwrap_for_state_dict(model)
    anchor_core = unwrap_for_state_dict(anchor_model)
    current = snapshot_params(policy_core)
    anchor_parameters = dict(anchor_core.named_parameters())
    anchor: dict[str, Tensor] = {}
    for name, current_value in current.items():
        if name not in anchor_parameters:
            raise KeyError(
                f"round reference is missing trainable policy parameter {name!r}"
            )
        anchor_value = anchor_parameters[name]
        if tuple(anchor_value.shape) != tuple(current_value.shape):
            raise ValueError(
                "round-reference parameter shape differs from live policy for "
                f"{name!r}: {tuple(anchor_value.shape)} vs "
                f"{tuple(current_value.shape)}"
            )
        anchor[name] = anchor_value.detach().to(
            device=current_value.device,
            dtype=current_value.dtype,
        ).clone()
    if not current:
        raise RuntimeError("signed-direction probe found no trainable parameters")
    return policy_core, anchor, current


@torch.no_grad()
def _assign_signed_trainable_direction_(
    model: torch.nn.Module,
    anchor: Mapping[str, Tensor],
    current: Mapping[str, Tensor],
    signed_scale: float,
) -> None:
    """Assign ``anchor + signed_scale * (current - anchor)`` without optimizer state."""

    scale = float(signed_scale)
    if not math.isfinite(scale) or not -1.0 <= scale <= 1.0:
        raise ValueError("signed-direction scale must lie in [-1, 1]")
    assigned = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in anchor or name not in current:
            raise KeyError(f"missing trainable parameter {name!r} in signed snapshot")
        anchor_value = anchor[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        current_value = current[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        parameter.data.copy_(
            anchor_value + scale * (current_value - anchor_value)
        )
        assigned += 1
    if assigned != len(current):
        raise RuntimeError(
            "signed-direction assignment did not consume every trainable snapshot: "
            f"assigned={assigned}, snapshot={len(current)}"
        )


def _trainable_direction_rms(
    anchor: Mapping[str, Tensor],
    current: Mapping[str, Tensor],
) -> float:
    """RMS parameter displacement used only to detect a zero diagnostic direction."""

    square_sum = 0.0
    count = 0
    if set(anchor) != set(current):
        raise ValueError("signed-direction anchor/current parameter keys differ")
    for name, anchor_value in anchor.items():
        current_value = current[name]
        if tuple(anchor_value.shape) != tuple(current_value.shape):
            raise ValueError(
                f"signed-direction parameter shape differs for {name!r}"
            )
        delta = current_value.detach().to(torch.float64) - anchor_value.detach().to(
            torch.float64
        )
        square_sum += float(delta.square().sum().cpu())
        count += int(delta.numel())
    return math.sqrt(square_sum / max(count, 1))


def _prepare_policy_evaluation_inputs(
    model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Tensor,
    *,
    K: int,
    batch_rep: dict[str, Any] | None = None,
) -> _PolicyEvaluationInputs:
    """Normalize candidates and tile conditioning tensors once per rollout batch."""
    B = int(batch["x"].shape[0])
    if int(candidates_phys.shape[0]) != int(K) or int(candidates_phys.shape[1]) != B:
        raise ValueError(
            "candidates_phys must start with (K, B); "
            f"expected ({K}, {B}, ...), got {tuple(candidates_phys.shape)}"
        )

    if batch_rep is None:
        batch_rep = repeat_batch_for_candidates(
            batch,
            K,
            tensor_keys=_DGPO_POLICY_BATCH_TENSOR_KEYS,
        )
    elif (
        "x" not in batch_rep
        or int(batch_rep["x"].shape[0]) != int(K) * B
    ):
        raise ValueError(
            "pre-expanded policy batch must have K*B rows; "
            f"expected {int(K) * B}"
        )
    inv_kb = batch_rep["x_invisible_mask"]
    c_flat = candidates_phys.reshape(K * B, *candidates_phys.shape[2:])
    eve = _unwrap_core_evenet(model)
    candidates_norm = _normalize_candidates_for_policy(eve, c_flat, inv_kb)
    noise_mask_rep = inv_kb.unsqueeze(-1)
    return _PolicyEvaluationInputs(
        batch_size=B,
        num_candidates=int(K),
        num_timesteps=1,
        candidates_norm=candidates_norm,
        batch_rep=batch_rep,
        noise_mask_rep=noise_mask_rep,
    )


def _prepare_parallel_policy_evaluation_inputs(
    base: _PolicyEvaluationInputs,
    num_timesteps: int,
) -> _PolicyEvaluationInputs:
    """Expand fixed K-candidate inputs across parallel policy-eval timesteps."""
    count = int(num_timesteps)
    if count < 1:
        raise ValueError(f"num_timesteps must be positive, got {count}")
    if count == 1:
        return base
    if base.num_timesteps != 1:
        raise ValueError("parallel policy inputs must be expanded from a one-timestep base")
    candidates_norm = (
        base.candidates_norm.unsqueeze(0)
        .expand(count, *base.candidates_norm.shape)
        .reshape(count * base.candidates_norm.shape[0], *base.candidates_norm.shape[1:])
        .contiguous()
    )
    batch_rep = repeat_batch_for_candidates(base.batch_rep, count)
    noise_mask_rep = batch_rep["x_invisible_mask"].unsqueeze(-1)
    return _PolicyEvaluationInputs(
        batch_size=base.batch_size,
        num_candidates=base.num_candidates,
        num_timesteps=count,
        candidates_norm=candidates_norm,
        batch_rep=batch_rep,
        noise_mask_rep=noise_mask_rep,
    )


def per_row_velocity_mse(
    pred_v: Tensor,
    target_v: Tensor,
    noise_mask_bn11: Tensor,
    invisible_padding: int,
) -> Tensor:
    """Masked mean squared error per row (one scalar per event×candidate row)."""
    m = noise_mask_bn11.expand_as(pred_v).to(dtype=pred_v.dtype)
    if invisible_padding > 0:
        m = m.clone()
        m[:, :, -invisible_padding:] = 0.0
    sq = (pred_v - target_v).pow(2) * m
    den = m.sum(dim=(1, 2)).clamp(min=1e-8)
    return sq.sum(dim=(1, 2)) / den


def _diag_scalar_float(diag: dict[str, Tensor], key: str) -> float:
    """Single scalar tensor from diagnostics, or NaN if absent / non-finite."""
    t = diag.get(key)
    if t is None:
        return float("nan")
    v = float(t.detach().float().cpu())
    return v if math.isfinite(v) else float("nan")


def _parameter_panel_from_diag(diag_last: dict[str, Tensor]) -> dict[str, float]:
    """Map loss diagnostics to W&B ``parameter/*`` keys (extend with more tensors here later)."""
    out: dict[str, float] = {}
    mapping = (
        ("w_e_mean", "parameter/w_e/mean"),
        ("w_e_std", "parameter/w_e/std"),
        ("w_e_min", "parameter/w_e/min"),
        ("w_e_max", "parameter/w_e/max"),
        ("kl_weight_mean", "parameter/kl_weight/mean"),
        ("kl_weight_min", "parameter/kl_weight/min"),
        ("kl_weight_max", "parameter/kl_weight/max"),
    )
    for src, dst in mapping:
        t = diag_last.get(src)
        if t is None:
            continue
        v = float(t.detach().cpu())
        if math.isfinite(v):
            out[dst] = v
    return out


def _mean_diag_dict(diags: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Elementwise mean of detached loss diagnostics across training sub-steps (accumulate mode).

    Keys are unioned across sub-steps so optional diagnostic panels survive
    when some substeps omit them—the missing slice is averaged as NaN (then typically filtered by
    the logger).
    """
    if not diags:
        return {}
    all_keys: set[str] = set()
    for d in diags:
        all_keys.update(d.keys())
    out: dict[str, Tensor] = {}
    for k in sorted(all_keys):  # stable ordering helps debugging parity across ranks.
        first: Tensor | None = None
        for d in diags:
            t = d.get(k)
            if t is None:
                continue
            t = t.detach()
            first = t
            break
        if first is None:
            continue
        filled: list[Tensor] = []
        for d in diags:
            t = d.get(k)
            if t is None:
                filled.append(torch.tensor(float("nan"), device=first.device, dtype=first.dtype))
            else:
                filled.append(t.detach())
        out[k] = torch.stack(filled, dim=0).mean(dim=0)
    return out


def _weighted_mean_diag_dict(
    diags: list[dict[str, Tensor]],
    weights: list[float],
) -> dict[str, Tensor]:
    """Event-weighted diagnostic mean for uneven policy-eval microbatches."""

    if len(diags) != len(weights):
        raise ValueError("diagnostic values and weights must have equal length")
    if not diags:
        return {}
    all_keys: set[str] = set()
    for diag in diags:
        all_keys.update(diag.keys())
    out: dict[str, Tensor] = {}
    for key in sorted(all_keys):
        present = [(diag[key].detach(), float(weight)) for diag, weight in zip(diags, weights) if key in diag]
        if not present:
            continue
        denominator = sum(weight for _, weight in present)
        if denominator <= 0.0:
            continue
        out[key] = sum(value * weight for value, weight in present) / denominator
    return out


def _finite_mean_float(values: Tensor) -> float:
    """Mean over finite tensor entries, or NaN if none are finite."""
    flat = values.reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().detach().cpu())


@torch.no_grad()
def _build_reward_source_metrics(
    rewards: Tensor,
    valid_b: Tensor,
    reward_agg: RewardAggregator,
    reward_breakdown: dict[str, Tensor] | None,
) -> dict[str, float]:
    """Generic per-source reward decomposition for spotting competing reward terms."""
    out: dict[str, float] = {}
    if not reward_breakdown:
        return out

    weight_by_name: dict[str, float] = {}
    for src, weight in reward_agg.sources:
        weight_by_name[src.name] = weight_by_name.get(src.name, 0.0) + float(weight)

    vb = valid_b.reshape(-1) > 0
    if vb.sum() == 0:
        for name in reward_breakdown:
            for suffix in (
                "mean",
                "weighted_mean",
                "selected_by_total_mean",
                "selected_by_total_weighted_mean",
                "source_best_of_k",
                "source_last_place",
                "selection_gap",
            ):
                out[f"reward/sources/{name}/{suffix}"] = float("nan")
        return out

    total_v = rewards[:, vb]
    total_best_k = total_v.argmax(dim=0)
    cols = torch.arange(int(total_best_k.numel()), device=rewards.device, dtype=torch.long)

    for name, tensor in reward_breakdown.items():
        if tensor.dim() != 2:
            continue
        rv = tensor[:, vb]
        weight = float(weight_by_name.get(name, 1.0))
        selected = rv[total_best_k, cols]
        mean = _finite_mean_float(rv)
        selected_mean = _finite_mean_float(selected)
        out[f"reward/sources/{name}/mean"] = mean
        out[f"reward/sources/{name}/weighted_mean"] = weight * mean
        out[f"reward/sources/{name}/selected_by_total_mean"] = selected_mean
        out[f"reward/sources/{name}/selected_by_total_weighted_mean"] = weight * selected_mean
        out[f"reward/sources/{name}/source_best_of_k"] = _finite_mean_float(
            rv.max(dim=0).values
        )
        out[f"reward/sources/{name}/source_last_place"] = _finite_mean_float(
            rv.min(dim=0).values
        )
        out[f"reward/sources/{name}/selection_gap"] = selected_mean - mean
    return out


@torch.no_grad()
def _build_reward_extra_metrics(
    rewards: Tensor,
    valid_b: Tensor,
    reward_agg: RewardAggregator,
    reward_breakdown: dict[str, Tensor] | None = None,
    *,
    log_distribution: bool = False,
    collect_profile_accum: bool = False,
    diagnostic_plot_names: set[str] | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """Light extras for W&B: ``reward/raw/{mean,std}`` and per-component contributions.

    Per-component contributions are emitted only when the active reward is
    :class:`ComponentNormalizedTruthDistanceReward`.
    They are reported under the independent ``components/`` panel as **negative** squared errors
    (i.e. per-component reward contributions, ``-err_c``) so the sign convention
    matches ``reward/raw/*`` (``<= 0``, larger = better, increasing = improving).
    By construction ``sum_c components/{c}/mean == reward/raw/mean`` for the
    component-normalized reward.

    Reward-hacking checks keep compact scalar breakdowns for the competing reward
    components: axis reward means, raw residual means, raw absolute residual means,
    and the relative-pT distribution panel.
    """
    out: dict[str, Any] = {}
    vb = valid_b.reshape(-1) > 0
    plot_names = diagnostic_plot_names or set()

    def _log_mean_abs(prefix: str, values: Tensor) -> Tensor:
        finite = values.reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() > 0:
            out[f"{prefix}/delta_mean"] = float(finite.mean().detach().cpu())
            out[f"{prefix}/delta_abs_mean"] = float(finite.abs().mean().detach().cpu())
        else:
            out[f"{prefix}/delta_mean"] = float("nan")
            out[f"{prefix}/delta_abs_mean"] = float("nan")
        return finite

    def _log_mean(prefix: str, values: Tensor) -> None:
        finite = values.reshape(-1)
        finite = finite[torch.isfinite(finite)]
        out[prefix] = float(finite.mean().detach().cpu()) if finite.numel() > 0 else float("nan")

    if vb.sum() > 0:
        r = rewards[:, vb]
        out["reward/raw/mean"] = float(r.mean().detach().cpu())
        out["reward/raw/std"] = float(r.std(unbiased=False).detach().cpu()) if r.numel() > 1 else 0.0
    else:
        out["reward/raw/mean"] = float("nan")
        out["reward/raw/std"] = float("nan")

    for source, _weight in reward_agg.sources:
        interface_metrics = getattr(source, "last_interface_metrics", None)
        if callable(interface_metrics):
            out.update(interface_metrics())
    interface_keys = sorted(
        key
        for key in out
        if key.startswith(("reward_consensus/", "reward_rank_audit/"))
    )
    if interface_keys and dist.is_available() and dist.is_initialized():
        # Every worker evaluates an equally sized fixed panel and live training
        # batch. One packed collective exposes their finite scalar mean to the
        # rank-0 W&B logger without gathering candidate tensors.
        packed = torch.zeros(
            (len(interface_keys), 2),
            device=rewards.device,
            dtype=torch.float64,
        )
        for index, key in enumerate(interface_keys):
            value = float(out[key])
            if math.isfinite(value):
                packed[index, 0] = value
                packed[index, 1] = 1.0
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        for index, key in enumerate(interface_keys):
            count = float(packed[index, 1].item())
            out[key] = (
                float(packed[index, 0].item()) / count
                if count > 0.0
                else float("nan")
            )

    if compact:
        return out

    out.update(_build_reward_source_metrics(rewards, valid_b, reward_agg, reward_breakdown))

    for src, _w in reward_agg.sources:
        if vb.sum() > 0:
            rewards_v = rewards[:, vb]
            best_k = rewards_v.argmax(dim=0)
            bv = int(best_k.numel())
            cols = torch.arange(bv, device=rewards.device, dtype=torch.long)
            topology = src.last_topology_metrics()
            if topology is not None:
                for name, values in topology.items():
                    all_values = values[:, vb]
                    best_values = all_values[best_k, cols]
                    _log_mean(f"diagnostics/ztautau_back_to_back/all/{name}", all_values)
                    _log_mean(f"diagnostics/ztautau_back_to_back/best/{name}", best_values)
        if isinstance(src, ComponentNormalizedTruthDistanceReward):
            comps = src.last_component_errors()
            if comps is None:
                continue
            for cname, ctensor in comps.items():
                if vb.sum() == 0:
                    out[f"components/{cname}/mean"] = float("nan")
                    continue
                cv = ctensor[:, vb]
                # Negate so the sign matches ``reward/raw/*`` (per-component reward, not error).
                out[f"components/{cname}/mean"] = float((-cv).mean().detach().cpu())
            if vb.sum() > 0:
                axis_pairs = _reward_component_axis_pairs(tuple(comps.keys()))
                for axis, (a, b) in axis_pairs.items():
                    axis_reward = -(comps[a] + comps[b])[:, vb]
                    out[f"diagnostics/reward_hacking/all/{axis}/reward_mean"] = float(
                        axis_reward.mean().detach().cpu()
                    )
                    out[f"diagnostics/reward_hacking/best/{axis}/reward_mean"] = float(
                        axis_reward[best_k, cols].mean().detach().cpu()
                    )
                profile_tensors: dict[str, tuple[Tensor, Tensor]] = {}
                deltas = src.last_component_deltas()
                truths = src.last_component_truths()
                if deltas is not None:
                    for axis, (a, b) in axis_pairs.items():
                        all_delta = torch.stack((deltas[a], deltas[b]), dim=-1)[:, vb]
                        best_delta = all_delta[best_k, cols, :]
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/all/{axis}",
                            all_delta,
                        )
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/best/{axis}",
                            best_delta,
                        )
                    if truths is not None:
                        for axis, (a, b) in axis_pairs.items():
                            truth_axis = torch.stack((truths[a], truths[b]), dim=-1)
                            delta_axis = torch.stack((deltas[a], deltas[b]), dim=-1)
                            profile_tensors[axis] = (truth_axis, delta_axis)
                kin_deltas = src.last_kinematic_deltas()
                if kin_deltas is not None:
                    rel_pt = kin_deltas.get("rel_pt")
                    feature_metric_names = tuple(
                        key
                        for key in kin_deltas.keys()
                        if not key.startswith("truth_") and key != "rel_pt"
                    )
                    for name in feature_metric_names:
                        tensor = kin_deltas.get(name)
                        all_delta = tensor[:, vb]
                        best_delta = all_delta[best_k, cols, :]
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/all/{name}",
                            all_delta,
                        )
                        _log_mean_abs(
                            f"diagnostics/reward_hacking/best/{name}",
                            best_delta,
                        )
                        truth_tensor = kin_deltas.get(f"truth_{name}")
                        if truth_tensor is not None:
                            profile_tensors[name] = (truth_tensor, tensor)
                    if rel_pt is not None:
                        all_rel_pt = rel_pt[:, vb].reshape(-1)
                        all_rel_pt = all_rel_pt[torch.isfinite(all_rel_pt)]
                        t_sel = rel_pt[:, vb, :]
                        best_rel_pt = t_sel[best_k, cols, :].reshape(-1)
                        best_rel_pt = best_rel_pt[torch.isfinite(best_rel_pt)]
                        if all_rel_pt.numel() > 0:
                            out["diagnostics/reward_hacking/all/rel_pt/mean"] = float(
                                all_rel_pt.mean().detach().cpu()
                            )
                            out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = float(
                                all_rel_pt.abs().mean().detach().cpu()
                            )
                        else:
                            out["diagnostics/reward_hacking/all/rel_pt/mean"] = float("nan")
                            out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = float("nan")
                        if best_rel_pt.numel() > 0:
                            out["diagnostics/reward_hacking/best/rel_pt/mean"] = float(
                                best_rel_pt.mean().detach().cpu()
                            )
                            out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = float(
                                best_rel_pt.abs().mean().detach().cpu()
                            )
                        else:
                            out["diagnostics/reward_hacking/best/rel_pt/mean"] = float("nan")
                            out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = float("nan")
                    if log_distribution or collect_profile_accum:
                        try:
                            import wandb  # noqa: F401

                            if log_distribution and "rel_pt_dist" in plot_names:
                                out["diagnostics/reward_hacking/dist/rel_pt"] = (
                                    _rel_pt_distribution_figure(
                                        all_rel_pt.detach().float().cpu().numpy(),
                                        best_rel_pt.detach().float().cpu().numpy(),
                                    )
                                )
                            pt_delta = kin_deltas.get("pt")
                            truth_pt = kin_deltas.get("truth_pt")
                            if pt_delta is not None and truth_pt is not None:
                                pt_all = pt_delta[:, vb]
                                truth_all = truth_pt[:, vb, :]
                                if truth_all.shape[0] == 1 and pt_all.shape[0] > 1:
                                    truth_all = truth_all.expand_as(pt_all)
                                best_pt_delta = pt_all[best_k, cols, :]
                                best_truth_pt = truth_all[best_k, cols, :]
                                pt_oracle_k = pt_all.abs().sum(dim=-1).argmin(dim=0)
                                oracle_pt_delta = pt_all[pt_oracle_k, cols, :]
                                oracle_truth_pt = truth_all[pt_oracle_k, cols, :]
                                _log_mean_abs(
                                    "diagnostics/reward_hacking/pt_oracle/pt",
                                    oracle_pt_delta,
                                )
                                prefix_k = min(10, int(pt_all.shape[0]))
                                pt_prefix = pt_all[:prefix_k]
                                truth_prefix = truth_all[:prefix_k]
                                rewards_prefix = rewards_v[:prefix_k]
                                prefix_reward_k = rewards_prefix.argmax(dim=0)
                                prefix_cols = torch.arange(
                                    int(prefix_reward_k.numel()),
                                    device=rewards.device,
                                    dtype=torch.long,
                                )
                                prefix_reward_delta = pt_prefix[
                                    prefix_reward_k, prefix_cols, :
                                ]
                                prefix_reward_truth = truth_prefix[
                                    prefix_reward_k, prefix_cols, :
                                ]
                                prefix_oracle_k = pt_prefix.abs().sum(dim=-1).argmin(dim=0)
                                prefix_oracle_delta = pt_prefix[
                                    prefix_oracle_k, prefix_cols, :
                                ]
                                prefix_oracle_truth = truth_prefix[
                                    prefix_oracle_k, prefix_cols, :
                                ]
                                prefix_reward_mask = (
                                    torch.isfinite(prefix_reward_truth)
                                    & torch.isfinite(prefix_reward_delta)
                                )
                                prefix_oracle_mask = (
                                    torch.isfinite(prefix_oracle_truth)
                                    & torch.isfinite(prefix_oracle_delta)
                                )
                                all_mask = torch.isfinite(truth_all) & torch.isfinite(pt_all)
                                best_mask = torch.isfinite(best_truth_pt) & torch.isfinite(best_pt_delta)
                                oracle_mask = (
                                    torch.isfinite(oracle_truth_pt)
                                    & torch.isfinite(oracle_pt_delta)
                                )
                                if collect_profile_accum:
                                    out[_diag_profile_raw_key("pt", "truth_all")] = (
                                        truth_all[all_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "delta_all")] = (
                                        pt_all[all_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "truth_best")] = (
                                        best_truth_pt[best_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "delta_best")] = (
                                        best_pt_delta[best_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "truth_oracle")] = (
                                        oracle_truth_pt[oracle_mask].detach().float().cpu().numpy()
                                    )
                                    out[_diag_profile_raw_key("pt", "delta_oracle")] = (
                                        oracle_pt_delta[oracle_mask].detach().float().cpu().numpy()
                                    )
                                if (
                                    log_distribution
                                    and "pt_first10_vs_fullK" in plot_names
                                    and int(pt_all.shape[0]) > prefix_k
                                ):
                                    out[
                                        "diagnostics/reward_hacking/profile/pt_delta_first10_vs_fullK"
                                    ] = _pt_delta_prefix_vs_full_figure(
                                        prefix_reward_truth[prefix_reward_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_reward_delta[prefix_reward_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        best_truth_pt[best_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        best_pt_delta[best_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_oracle_truth[prefix_oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_oracle_delta[prefix_oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        oracle_truth_pt[oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        oracle_pt_delta[oracle_mask]
                                        .detach()
                                        .float()
                                        .cpu()
                                        .numpy(),
                                        prefix_k=prefix_k,
                                        full_k=int(pt_all.shape[0]),
                                        title=(
                                            "pT response: first 10 candidates vs full "
                                            f"{int(pt_all.shape[0])}"
                                        ),
                                    )
                                if log_distribution and "pt_profile" in plot_names:
                                    out[
                                        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt"
                                    ] = _pt_delta_selection_profiles_figure(
                                        truth_all.detach().float().cpu().numpy(),
                                        pt_all.detach().float().cpu().numpy(),
                                        best_truth_pt.detach().float().cpu().numpy(),
                                        best_pt_delta.detach().float().cpu().numpy(),
                                        oracle_truth_pt.detach().float().cpu().numpy(),
                                        oracle_pt_delta.detach().float().cpu().numpy(),
                                        title="Reward selection pT bias vs truth pT",
                                    )
                            profile_names = tuple(
                                name for name in profile_tensors.keys() if name != "pt"
                            )
                            for profile_name in profile_names:
                                if not log_distribution or f"{profile_name}_profile" not in plot_names:
                                    continue
                                tensors = profile_tensors.get(profile_name)
                                if tensors is None:
                                    continue
                                truth_tensor, delta_tensor = tensors
                                truth_tensor = _align_truth_tensor_to_delta(
                                    truth_tensor, delta_tensor
                                )
                                profile_delta_all = delta_tensor[:, vb]
                                profile_truth_all = truth_tensor[:, vb]
                                profile_best_delta = profile_delta_all[best_k, cols, :]
                                profile_best_truth = profile_truth_all[best_k, cols, :]
                                profile_oracle_k = profile_delta_all.abs().sum(dim=-1).argmin(dim=0)
                                profile_oracle_delta = profile_delta_all[
                                    profile_oracle_k, cols, :
                                ]
                                profile_oracle_truth = profile_truth_all[
                                    profile_oracle_k, cols, :
                                ]
                                _log_mean_abs(
                                    f"diagnostics/reward_hacking/{profile_name}_oracle/{profile_name}",
                                    profile_oracle_delta,
                                )
                                truth_all_np, delta_all_np = _finite_profile_numpy(
                                    profile_truth_all, profile_delta_all
                                )
                                truth_best_np, delta_best_np = _finite_profile_numpy(
                                    profile_best_truth, profile_best_delta
                                )
                                truth_oracle_np, delta_oracle_np = _finite_profile_numpy(
                                    profile_oracle_truth, profile_oracle_delta
                                )
                                raw_items = {
                                    "truth_all": truth_all_np,
                                    "delta_all": delta_all_np,
                                    "truth_best": truth_best_np,
                                    "delta_best": delta_best_np,
                                    "truth_oracle": truth_oracle_np,
                                    "delta_oracle": delta_oracle_np,
                                }
                                for suffix, arr in raw_items.items():
                                    out[_diag_profile_raw_key(profile_name, suffix)] = arr
                                out[_diag_profile_log_key(profile_name)] = (
                                    _delta_selection_profiles_figure(
                                        truth_all_np,
                                        delta_all_np,
                                        truth_best_np,
                                        delta_best_np,
                                        truth_oracle_np,
                                        delta_oracle_np,
                                        profile_name=profile_name,
                                        title=_diag_profile_title(profile_name),
                                    )
                                )
                        except Exception:
                            pass
            else:
                nan_f = float("nan")
                axis_pairs = _reward_component_axis_pairs(tuple(comps.keys()))
                kin_deltas = src.last_kinematic_deltas() or {}
                feature_metric_names = tuple(
                    key for key in kin_deltas.keys() if not key.startswith("truth_") and key != "rel_pt"
                )
                for scope in ("all", "best"):
                    for axis in axis_pairs:
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/reward_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/delta_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{axis}/delta_abs_mean"] = nan_f
                    for name in feature_metric_names:
                        out[f"diagnostics/reward_hacking/{scope}/{name}/delta_mean"] = nan_f
                        out[f"diagnostics/reward_hacking/{scope}/{name}/delta_abs_mean"] = nan_f
                if kin_deltas.get("rel_pt") is not None:
                    out["diagnostics/reward_hacking/all/rel_pt/mean"] = nan_f
                    out["diagnostics/reward_hacking/all/rel_pt/abs_mean"] = nan_f
                    out["diagnostics/reward_hacking/best/rel_pt/mean"] = nan_f
                    out["diagnostics/reward_hacking/best/rel_pt/abs_mean"] = nan_f
            break

    return out


def _append_projection_constraint_panel_metrics(out: dict[str, float]) -> None:
    """Populate ``projection/constraint/*`` for the dedicated W&B projection panel."""
    pure = out.get("projection/active")
    if pure is None or not math.isfinite(float(pure)) or float(pure) != 1.0:
        return
    prefix = "latent_constraint/"
    for key, val in list(out.items()):
        if isinstance(key, str) and key.startswith(prefix):
            out[f"projection/constraint/{key[len(prefix):]}"] = val


def _append_swd_panel_metrics(out: dict[str, float]) -> None:
    """Populate ``swd/*`` for the dedicated W&B latent-SWD monitoring panel.

    Sources ``latent_constraint/*`` (raw diag) or ``projection/latent_*`` (mapped aliases).
    Only populated when the frozen latent encoder is the active projection backend.
    """
    src_map = {
        "swd/pred_truth": (
            "latent_constraint/swd_pred_truth",
            "projection/latent_swd_pred_truth",
        ),
        "swd/truth_truth": (
            "latent_constraint/swd_truth_truth",
            "projection/latent_swd_truth_truth",
        ),
        "swd/ratio": (
            "latent_constraint/swd_ratio",
            "projection/latent_swd_ratio",
        ),
        "swd/C_norm": (
            "latent_constraint/C_norm",
            "projection/latent_C_norm",
        ),
        "swd/mask_count": (
            "latent_constraint/mask_count",
            "projection/latent_mask_count",
        ),
        "swd/skipped_small_mask": ("latent_constraint/skipped_small_mask",),
    }
    populated = False
    for dst, src_keys in src_map.items():
        for src in src_keys:
            val = out.get(src)
            if val is None:
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                out[dst] = fv
                populated = True
                break
    if populated:
        out["swd/active"] = 1.0
        skipped = out.get("swd/skipped_small_mask")
        if skipped is not None and float(skipped) >= 1.0:
            out["swd/active"] = 0.0


def _append_projection_summary_metrics(out: dict[str, Any]) -> None:
    """Add ``projection/summary/C_projected_minus_old`` for the W&B projection panel."""
    c_projected = out.get("projection/C_projected")
    c_old = out.get("projection/C_old", out.get("projection/C_raw"))
    if c_old is not None and c_projected is not None:
        try:
            out["projection/summary/C_projected_minus_old"] = float(c_projected) - float(c_old)
        except (TypeError, ValueError):
            pass

@torch.no_grad()
def _build_train_metrics(
    diag_last: dict[str, Tensor],
    rewards: Tensor,
    valid_b: Tensor,
    advantages: Tensor | None = None,
) -> dict[str, float]:
    """Training panels ``reward/monitor/*``, ``train/loss/*``, and ``parameter/*`` for Weights & Biases."""
    param = _parameter_panel_from_diag(diag_last)
    vb = valid_b.reshape(-1) > 0
    adv_gap = (
        compute_reward_advantage_pos_neg_gap(rewards, advantages, valid_b)
        if advantages is not None
        else float("nan")
    )
    if vb.sum() == 0:
        nan = float("nan")
        base: dict[str, float] = {
            "train/loss/total": float(diag_last["loss_total"].cpu()),
            "train/loss/velocity": _diag_scalar_float(
                diag_last, "loss_velocity_training"
            ),
            "train/loss/dgpo": float(diag_last["loss_main"].cpu()),
            "train/loss/kl": _diag_scalar_float(diag_last, "train/loss/kl"),
            "train/loss/L_cur": float(diag_last["L_cur_mean"].cpu()),
            "train/loss/L_ref": float(diag_last["L_ref_mean"].cpu()),
            "train/loss/delta": float(diag_last["delta_abs_mean"].cpu()),
            "reward/monitor/best_of_k": nan,
            "reward/monitor/median": nan,
            "reward/monitor/mean_gap": nan,
            "reward/monitor/last_place": nan,
            "reward/monitor/p10": nan,
            "reward/monitor/p30": nan,
            "reward/monitor/p70": nan,
            "reward/monitor/p90": nan,
            "reward/monitor/advantage_pos_neg_gap": adv_gap,
        }
        if not math.isfinite(base["train/loss/velocity"]):
            base["train/loss/velocity"] = float(diag_last["loss_total"].cpu())
        base.update(param)
        for dk, dv in diag_last.items():
            if isinstance(dk, str) and (
                dk.startswith(("projection/", "latent_constraint/", "train/regularization/", "reference_trust/"))
                or dk == "train/loss/variance_regularization"
            ):
                base[dk] = float(dv.detach().float().cpu())
        return base

    r = rewards[:, vb]
    K = r.shape[0]
    best_of_k = float(r.max(dim=0).values.mean().cpu())
    median_e = float(r.median(dim=0).values.mean().cpu())
    last_place = float(r.min(dim=0).values.mean().cpu())
    p10 = float(r.quantile(0.1, dim=0).mean().cpu()) if K >= 2 else last_place
    p30 = float(r.quantile(0.3, dim=0).mean().cpu()) if K >= 2 else last_place
    p70 = float(r.quantile(0.7, dim=0).mean().cpu()) if K >= 2 else last_place
    p90 = float(r.quantile(0.9, dim=0).mean().cpu()) if K >= 2 else last_place
    mean_gap = compute_reward_mean_gap(rewards, valid_b)

    lv = _diag_scalar_float(diag_last, "loss_velocity_training")
    if not math.isfinite(lv):
        lv = float(diag_last["loss_total"].cpu())
    out: dict[str, float] = {
        "train/loss/total": float(diag_last["loss_total"].cpu()),
        "train/loss/velocity": lv,
        "train/loss/dgpo": float(diag_last["loss_main"].cpu()),
        "train/loss/kl": _diag_scalar_float(diag_last, "train/loss/kl"),
        "train/loss/L_cur": float(diag_last["L_cur_mean"].cpu()),
        "train/loss/L_ref": float(diag_last["L_ref_mean"].cpu()),
        "train/loss/delta": float(diag_last["delta_abs_mean"].cpu()),
        "reward/monitor/best_of_k": best_of_k,
        "reward/monitor/median": median_e,
        "reward/monitor/mean_gap": mean_gap,
        "reward/monitor/last_place": last_place,
        "reward/monitor/p10": p10,
        "reward/monitor/p30": p30,
        "reward/monitor/p70": p70,
        "reward/monitor/p90": p90,
        "reward/monitor/advantage_pos_neg_gap": adv_gap,
    }
    out.update(param)
    for dk, dv in diag_last.items():
        if isinstance(dk, str) and (
            dk.startswith(
                (
                    "projection/",
                    "latent_constraint/",
                    "train/regularization/",
                    "train/gradient_sync/",
                    "reference_trust/",
                )
            )
            or dk == "train/loss/variance_regularization"
        ):
            out[dk] = float(dv.detach().float().cpu())
    return out


def policy_evaluation_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Tensor,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    t: Tensor | None = None,
    eps_rep: Tensor | None = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
    num_timesteps: int = 1,
    prepared_inputs: _PolicyEvaluationInputs | None = None,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    dict[str, Any],
    Tensor,
]:
    """Evaluate one or more independent ``(t, eps)`` draws in one model forward.

    With ``num_timesteps=1`` the legacy shapes are preserved. For ``T>1``,
    ``L_cur``/``L_ref`` are ``(T,K,B)``, ``t`` is ``(T,B)``, and flattened
    model tensors use ``T*K*B`` rows ordered by timestep, candidate, event.
    """
    B = int(batch["x"].shape[0])
    T = max(1, int(num_timesteps))
    if prepared_inputs is None:
        base_inputs = _prepare_policy_evaluation_inputs(
            model,
            batch,
            candidates_phys,
            K=K,
        )
        prepared_inputs = _prepare_parallel_policy_evaluation_inputs(base_inputs, T)
    elif (
        prepared_inputs.batch_size != B
        or prepared_inputs.num_candidates != int(K)
        or prepared_inputs.num_timesteps != T
    ):
        raise ValueError(
            "prepared policy-evaluation inputs do not match the requested (T, K, B): "
            f"prepared=({prepared_inputs.num_timesteps}, "
            f"{prepared_inputs.num_candidates}, {prepared_inputs.batch_size}), "
            f"requested=({T}, {K}, {B})"
        )
    c_norm = prepared_inputs.candidates_norm
    batch_rep = prepared_inputs.batch_rep
    noise_mask_rep = prepared_inputs.noise_mask_rep

    if t is None:
        t_pb = (
            torch.rand(T, B, device=device, dtype=torch.float32)
            * (t_max - t_min)
            + t_min
        )
    else:
        t_pb = t.reshape(1, B) if T == 1 and t.ndim == 1 else t
        if tuple(t_pb.shape) != (T, B):
            raise ValueError(
                f"policy_evaluation_step t must have shape {(T, B)}, "
                f"got {tuple(t_pb.shape)}"
            )
    _, alpha, sigma = get_logsnr_alpha_sigma(
        t_pb.reshape(T * B),
        shape=(T * B, 1, 1),
    )
    alpha_rep = (
        alpha.reshape(T, B, 1, 1)
        .unsqueeze(1)
        .expand(T, K, B, 1, 1)
        .reshape(T * K * B, 1, 1)
        .to(dtype)
    )
    sigma_rep = (
        sigma.reshape(T, B, 1, 1)
        .unsqueeze(1)
        .expand(T, K, B, 1, 1)
        .reshape(T * K * B, 1, 1)
        .to(dtype)
    )

    N_nu, F_eff = c_norm.shape[1], c_norm.shape[2]
    expected_eps_shape = (T * K * B, N_nu, F_eff)
    if eps_rep is None:
        if shared_noise:
            eps = torch.randn(T, B, N_nu, F_eff, device=device, dtype=dtype)
            eps_rep = (
                eps.unsqueeze(1)
                .expand(T, K, B, N_nu, F_eff)
                .reshape(*expected_eps_shape)
            )
        else:
            eps_rep = torch.randn(*expected_eps_shape, device=device, dtype=dtype)
    else:
        eps_rep = eps_rep.to(device=device, dtype=dtype)
        if tuple(eps_rep.shape) != expected_eps_shape:
            raise ValueError(
                f"policy_evaluation_step eps_rep must have shape {expected_eps_shape}, "
                f"got {tuple(eps_rep.shape)}"
            )

    x_t = alpha_rep * c_norm + sigma_rep * eps_rep
    target_v = alpha_rep * eps_rep - sigma_rep * c_norm

    t_rep = t_pb.unsqueeze(1).expand(T, K, B).reshape(T * K * B)
    if isinstance(model, DDP):
        model_v = model(x_t, batch_rep, t_rep, noise_mask_rep)
    else:
        model_v = model.predict_diffusion_vector(
            noise_x=x_t,
            cond_x=batch_rep,
            time=t_rep,
            mode="neutrino",
            noise_mask=noise_mask_rep,
        )
    # ``predict_diffusion_vector`` already strips invisible_padding from its output,
    # so ``model_v`` and ``target_v`` share the same width (``invisible_input_dim``).
    L_cur = per_row_velocity_mse(model_v, target_v, noise_mask_rep, invisible_padding=0)

    with torch.no_grad():
        ref_v = ref_model.predict_diffusion_vector(
            noise_x=x_t,
            cond_x=batch_rep,
            time=t_rep,
            mode="neutrino",
            noise_mask=noise_mask_rep,
        )
        L_ref = per_row_velocity_mse(ref_v, target_v, noise_mask_rep, invisible_padding=0)

    L_cur_out = L_cur.reshape(T, K, B)
    L_ref_out = L_ref.reshape(T, K, B)
    t_out = t_pb
    if T == 1:
        L_cur_out = L_cur_out[0]
        L_ref_out = L_ref_out[0]
        t_out = t_out[0]
    return (
        L_cur_out,
        L_ref_out,
        t_out,
        model_v,
        ref_v,
        noise_mask_rep,
        x_t,
        target_v,
        t_rep,
        batch_rep,
        eps_rep,
    )


class _DgpoOptimizerWithSchedule:
    """Bundle ``(AdamW, LambdaLR)`` for DGPO: ``scheduler_step()`` once per batch.

    The train loop performs a single ``optimizer.step()`` per batch (sub-step
    gradients are accumulated).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        *,
        cosine_config: Mapping[str, Any] | None = None,
        warmup_steps: int = 1,
        warmup_groups: list[bool] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cosine_state: dict[str, Any] | None = None
        if cosine_config is not None:
            total = cosine_config.get("total_steps")
            floor = float(cosine_config.get("min_lr_ratio", 0.1))
            if isinstance(total, bool) or not isinstance(total, int) or total <= warmup_steps:
                raise ValueError("cosine total_steps must be an integer greater than warmup_steps")
            if not math.isfinite(floor) or not 0 < floor <= 1:
                raise ValueError("cosine min_lr_ratio must be in (0, 1]")
            groups = list(warmup_groups or [True] * len(optimizer.param_groups))
            if len(groups) != len(optimizer.param_groups):
                raise ValueError("cosine warmup group count mismatch")
            selected_groups = cosine_config.get("groups")
            group_names = [pg.get("group_name") for pg in optimizer.param_groups]
            if selected_groups is not None and (
                not isinstance(selected_groups, (list, tuple)) or not selected_groups
                or any(not isinstance(name, str) for name in selected_groups)
                or len(set(selected_groups)) != len(selected_groups)
                or set(selected_groups) - set(group_names)
            ):
                raise ValueError("cosine groups must name distinct existing optimizer groups")
            self.cosine_state = {
                "kind": "cosine", "total_steps": total, "min_lr_ratio": floor,
                "warmup_steps": int(warmup_steps), "warmup_groups": groups,
                "decay_groups": [selected_groups is None or name in selected_groups for name in group_names],
                "start_step": int(warmup_steps),
                "anchor_factors": [1.0] * len(groups),
            }
            # Plain lambdas keep LambdaLR's state free of bound-object references.
            self.scheduler.lr_lambdas = [
                lambda step, i=i: self._cosine_factor(step, i) for i in range(len(groups))
            ]
            self._apply_cosine_lr()

    def _cosine_factor(self, step: int, group: int) -> float:
        s = self.cosine_state
        assert s is not None
        if step < s["warmup_steps"]:
            return step / s["warmup_steps"] if s["warmup_groups"][group] else 1.0
        if not s["decay_groups"][group]:
            return float(s["anchor_factors"][group])
        progress = min(1.0, max(0.0, (step - s["start_step"]) /
                                (s["total_steps"] - s["start_step"])))
        factor = s["min_lr_ratio"] + (1 - s["min_lr_ratio"]) * .5 * (1 + math.cos(math.pi * progress))
        return float(s["anchor_factors"][group]) * factor

    def _apply_cosine_lr(self) -> None:
        for i, pg in enumerate(self.param_groups):
            pg["lr"] = self.scheduler.base_lrs[i] * self._cosine_factor(self.scheduler.last_epoch, i)
        self.scheduler._last_lr = [pg["lr"] for pg in self.param_groups]

    def _restore_cosine(self, state: Mapping[str, Any]) -> None:
        assert self.cosine_state is not None
        saved = state.get("lr_schedule")
        current = self.cosine_state
        if saved is not None:
            for key in ("kind", "total_steps", "min_lr_ratio", "warmup_steps", "warmup_groups"):
                if saved.get(key) != current[key]:
                    raise ValueError(f"cosine resume protocol mismatch: {key}")
            # Older cosine checkpoints decayed every group.
            saved_decay_groups = saved.get("decay_groups", [True] * len(self.param_groups))
            if saved_decay_groups != current["decay_groups"]:
                raise ValueError("cosine resume protocol mismatch: decay_groups")
            if not current["warmup_steps"] <= saved["start_step"] < current["total_steps"]:
                raise ValueError("invalid saved cosine start_step")
            anchors = saved["anchor_factors"]
            if len(anchors) != len(self.param_groups) or any(
                not math.isfinite(float(x)) or float(x) <= 0 for x in anchors
            ):
                raise ValueError("invalid saved cosine anchor_factors")
            self.cosine_state = dict(saved)
            self.cosine_state["decay_groups"] = list(saved_decay_groups)
        else:
            # One-time migration from linear->constant: anchor at saved LR,
            # preserving Adam moments and scheduler clock with no LR jump.
            step = int(self.scheduler.last_epoch)
            if step >= current["total_steps"]:
                raise ValueError("cosine total_steps must exceed the migration scheduler step")
            current["start_step"] = max(current["warmup_steps"], step)
            if step >= current["warmup_steps"]:
                current["anchor_factors"] = [
                    float(pg["lr"]) / float(base)
                    for pg, base in zip(self.param_groups, self.scheduler.base_lrs, strict=True)
                ]
                if any(not math.isfinite(x) or x <= 0 for x in current["anchor_factors"]):
                    raise ValueError("cosine migration requires finite positive saved learning rates")
            _log.info("[DGPO] Migrated to cosine at scheduler_step=%s, saved_lrs=%s, end_step=%s, floor_ratio=%s",
                      step, [pg["lr"] for pg in self.param_groups], current["total_steps"], current["min_lr_ratio"])
        self._apply_cosine_lr()

    def step(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.step(*args, **kwargs)

    def scheduler_step(self) -> None:
        self.scheduler.step()

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        return self.optimizer.zero_grad(*args, **kwargs)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.optimizer.param_groups

    @property
    def state(self) -> dict[Any, Any]:
        return self.optimizer.state

    def state_dict(self) -> dict[str, Any]:
        result = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.cosine_state is not None:
            result["lr_schedule"] = copy.deepcopy(self.cosine_state)
        return result

    def load_state_dict(self, state: Any, *, use_config_lr_schedule: bool = False) -> None:
        # Resume Adam moments/LR scheduling, but take weight decay from the
        # current YAML. PyTorch otherwise silently restores the old group WD.
        configured_weight_decay = [float(pg["weight_decay"]) for pg in self.param_groups]
        configured_base_lrs = list(self.scheduler.base_lrs)
        if use_config_lr_schedule:
            # Only the explicit startup resume opts in. Rollback/transaction
            # restores still use the checkpoint's exact LR/scheduler state.
            if self.cosine_state is None:
                raise ValueError("resume_use_config requires a configured cosine schedule")
            if not isinstance(state, dict) or "optimizer" not in state or "scheduler" not in state:
                raise ValueError("resume_use_config requires optimizer and scheduler state")
            saved_names = [pg.get("group_name") for pg in state["optimizer"]["param_groups"]]
            current_names = [pg.get("group_name") for pg in self.param_groups]
            if (saved_names != current_names or any(name is None for name in current_names)
                    or len(set(current_names)) != len(current_names)):
                raise ValueError("resume_use_config requires matching named optimizer groups in order")
            if len(state["scheduler"].get("base_lrs", [])) != len(configured_base_lrs):
                raise ValueError("resume_use_config scheduler group count mismatch")
            saved_step = state["scheduler"].get("last_epoch")
            if isinstance(saved_step, bool) or not isinstance(saved_step, int) or saved_step < 0:
                raise ValueError("resume_use_config requires a valid saved scheduler clock")
        if self.cosine_state is not None and not (
            isinstance(state, dict) and "optimizer" in state and "scheduler" in state
        ):
            raise ValueError("cosine resume requires optimizer and scheduler state")
        if isinstance(state, dict) and state.get("lr_schedule") is not None and self.cosine_state is None:
            raise ValueError("checkpoint uses cosine; enable the matching lr_schedule to resume")
        if isinstance(state, dict) and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                self.scheduler.load_state_dict(state["scheduler"])
            for pg, weight_decay in zip(self.param_groups, configured_weight_decay, strict=True):
                pg["weight_decay"] = weight_decay
            if use_config_lr_schedule:
                # New base rates/decay scope at the OLD absolute clock: never
                # reset Adam moments, optimizer steps, warmup, or epoch count.
                saved_lrs = [float(pg["lr"]) for pg in self.param_groups]
                self.scheduler.base_lrs = configured_base_lrs
                for pg, base in zip(self.param_groups, configured_base_lrs, strict=True):
                    pg["initial_lr"] = base
                self._apply_cosine_lr()
                _log.info(
                    "[DGPO] Explicit LR resume override at scheduler_step=%s: saved_lrs=%s "
                    "configured_base_lrs=%s effective_lrs=%s end_step=%s; Adam moments/clock preserved.",
                    self.scheduler.last_epoch, saved_lrs, configured_base_lrs,
                    [pg["lr"] for pg in self.param_groups], self.cosine_state["total_steps"],
                )
            elif self.cosine_state is not None:
                self._restore_cosine(state)
            return
        try:
            self.optimizer.load_state_dict(state)
        except (ValueError, RuntimeError):
            raise
        for pg, weight_decay in zip(self.param_groups, configured_weight_decay, strict=True):
            pg["weight_decay"] = weight_decay
        _log.warning(
            "[DGPO] Loaded legacy optimizer state dict without scheduler keys; "
            "LambdaLR keeps its current step counter (may be out of sync with resume step).",
        )


def _latent_swd_step_seed(base: int | None, sample_idx: int) -> int | None:
    """Per-(step, multi-sample) common-random-numbers seed for the latent SWD.

    Returns ``None`` (legacy per-call randomness) when ``base`` is ``None`` so any
    non-seeded caller is unaffected. The combination is stable within a CPO repair
    step (same ``base``) but advances across global steps so the projection space
    is still covered over training.
    """
    if base is None:
        return None
    return (int(base) * 1000003 + int(sample_idx)) & 0x7FFFFFFF


def _compute_projection_constraint_raw(
    *,
    constraint_state: ProjectionConstraintState,
    model_v: Tensor,
    x_t: Tensor,
    t_rep: Tensor,
    noise_mask_rep: Tensor,
    batch_kb: dict[str, Any],
    core_model: torch.nn.Module,
    cartesian: bool,
    K: int,
    candidate_weights_kb: Tensor | None,
    update_ema: bool = True,
    world_size: int = 1,
    constraint_seed: int | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Batch constraint scalar ``C_B`` from the frozen latent-SWD encoder.

    ``constraint_seed`` enables common random numbers for the stochastic SWD
    estimator (fixed projections + null split across the CPO repair's repeated
    evaluations within one step).
    """
    return compute_latent_swd_constraint(
        model_v=model_v,
        x_t=x_t,
        t_rep=t_rep,
        noise_mask_rep=noise_mask_rep,
        batch_kb=batch_kb,
        core_model=core_model,
        cartesian=cartesian,
        K=K,
        state=constraint_state,
        candidate_weights_kb=candidate_weights_kb,
        update_ema=update_ema,
        world_size=world_size,
        seed=constraint_seed,
    )


def _latent_projection_metrics(off_diag: Mapping[str, Tensor]) -> dict[str, float]:
    """Map latent-SWD constraint diagnostics (``latent_constraint/*``) to ``projection/*`` scalars."""
    key_map = {
        "projection/latent_swd_pred_truth": "latent_constraint/swd_pred_truth",
        "projection/latent_swd_truth_truth": "latent_constraint/swd_truth_truth",
        "projection/latent_swd_ratio": "latent_constraint/swd_ratio",
        "projection/latent_C_norm": "latent_constraint/C_norm",
        "projection/latent_mask_count": "latent_constraint/mask_count",
    }
    out: dict[str, float] = {}
    for dst, src in key_map.items():
        val = off_diag.get(src)
        if isinstance(val, Tensor) and int(val.numel()) == 1:
            fv = float(val.detach().reshape(-1)[0].cpu())
            if math.isfinite(fv):
                out[dst] = fv
    return out


def _projection_constraint_forward(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    t: Tensor | None = None,
    eps_rep: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    """Run policy evaluation and return tensors needed for detached constraint measurement."""
    (
        _L_cur,
        _L_ref,
        t_out,
        model_v,
        _ref_v,
        noise_mask_rep,
        x_t,
        _target_v,
        t_rep,
        batch_rep,
        eps_out,
    ) = policy_evaluation_step(
        model,
        ref_model,
        batch,
        candidates_phys,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        t=t,
        eps_rep=eps_rep,
        t_min=policy_eval_t_min,
        t_max=policy_eval_t_max,
    )
    return model_v, x_t, t_rep, noise_mask_rep, batch_rep


@torch.no_grad()
def _projection_constraint_C_detached(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    candidate_weights_kb: Tensor | None,
    proj_cfg: ProjectionConstraintConfig,
    constraint_state: ProjectionConstraintState,
    t: Tensor,
    eps_rep: Tensor,
    world_size: int = 1,
    constraint_seed: int | None = None,
) -> tuple[float, dict[str, Tensor]]:
    """Evaluate detached ``C_B`` at current model weights with frozen policy-eval randomness."""
    core = _unwrap_core_evenet(model)
    model_v, x_t, t_rep, noise_mask_rep, batch_rep = _projection_constraint_forward(
        model,
        ref_model,
        batch,
        candidates_phys,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        policy_eval_t_min=policy_eval_t_min,
        policy_eval_t_max=policy_eval_t_max,
        t=t,
        eps_rep=eps_rep,
    )
    C_raw_t, off_diag = _compute_projection_constraint_raw(
        constraint_state=constraint_state,
        model_v=model_v,
        x_t=x_t,
        t_rep=t_rep,
        noise_mask_rep=noise_mask_rep,
        batch_kb=batch_rep,
        core_model=core,
        cartesian=_truth_generation_cartesian(),
        K=K,
        candidate_weights_kb=candidate_weights_kb,
        update_ema=True,
        world_size=world_size,
        constraint_seed=constraint_seed,
    )
    return float(C_raw_t.detach().float().cpu()), off_diag


@torch.no_grad()
def _projection_constraint_C_detached_average(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    candidate_weights_kb: Tensor | None,
    proj_cfg: ProjectionConstraintConfig,
    constraint_state: ProjectionConstraintState,
    frozen_eval_inputs: list[tuple[Tensor, Tensor, int | None]],
    world_size: int = 1,
) -> tuple[float, dict[str, Tensor], dict[str, float]]:
    """Average detached ``C_B`` over frozen policy-eval ``(t, eps, seed)`` triples.

    The per-draw ``seed`` is reused from the ``theta_old`` evaluation so each
    post-AdamW proxy uses the **same** SWD projections + null split as its matching
    ``theta_old`` draw (common random numbers); only the parameters differ.
    """
    if not frozen_eval_inputs:
        raise ValueError("frozen_eval_inputs must be non-empty for detached constraint averaging")
    c_vals: list[float] = []
    off_diag_first: dict[str, Tensor] = {}
    for t_frozen, eps_frozen, seed_frozen in frozen_eval_inputs:
        c_i, off_diag = _projection_constraint_C_detached(
            model,
            ref_model,
            batch,
            candidates_phys,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            policy_eval_t_min=policy_eval_t_min,
            policy_eval_t_max=policy_eval_t_max,
            candidate_weights_kb=candidate_weights_kb,
            proj_cfg=proj_cfg,
            constraint_state=constraint_state,
            t=t_frozen,
            eps_rep=eps_frozen,
            world_size=world_size,
            constraint_seed=seed_frozen,
        )
        c_vals.append(float(c_i))
        if not off_diag_first:
            off_diag_first = off_diag
    c_mean = float(sum(c_vals) / len(c_vals))
    proxy_diag: dict[str, float] = {
        "projection/direct_post_adam_proxy/C_mean": c_mean,
    }
    if len(c_vals) > 1:
        proxy_diag["projection/direct_post_adam_proxy/C_std"] = float(
            torch.tensor(c_vals, dtype=torch.float64).std(unbiased=False).cpu()
        )
        proxy_diag["projection/direct_post_adam_proxy/C_min"] = float(min(c_vals))
        proxy_diag["projection/direct_post_adam_proxy/C_max"] = float(max(c_vals))
    else:
        proxy_diag["projection/direct_post_adam_proxy/C_std"] = 0.0
        proxy_diag["projection/direct_post_adam_proxy/C_min"] = c_mean
        proxy_diag["projection/direct_post_adam_proxy/C_max"] = c_mean
    return c_mean, off_diag_first, proxy_diag


def _projection_grad_debug(model: torch.nn.Module) -> str:
    """One-line diagnostic for why the projection constraint lost its grad graph."""
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    inf_mode = getattr(torch, "is_inference_mode_enabled", lambda: "n/a")()
    return (
        f"[grad_debug grad_enabled={torch.is_grad_enabled()} "
        f"inference_mode={inf_mode} "
        f"trainable_params={n_trainable}/{n_total}]"
    )


def _projection_constraint_value_and_grad_at_theta_old(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    *,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    candidate_weights_kb: Tensor | None,
    optimizer: Any,
    proj_cfg: ProjectionConstraintConfig,
    constraint_state: ProjectionConstraintState,
    world_size: int = 1,
    constraint_seed_base: int | None = None,
) -> tuple[Tensor, float, dict[str, Tensor], list[tuple[Tensor, Tensor, int | None]], dict[str, float]]:
    """Evaluate ``C_raw`` and flat ``b = nabla C`` at parameters already set to ``theta_old``.

    When ``(int(proj_cfg.multi_sample_count) > 1)`` is true, averages ``M`` independent
    policy-eval ``(t, eps)`` draws: ``C_bar = mean_i C_i``, ``b = grad C_bar``.
    Gradients are accumulated sequentially via ``(C_i / M).backward()`` to avoid
    retaining ``M`` full autograd graphs simultaneously.

    Returns all frozen ``(t, eps)`` pairs used for ``C_bar`` so post-Adam proxy
    evaluation can reuse the same common randomness.
    """
    core = _unwrap_core_evenet(model)
    ms_count = int(proj_cfg.multi_sample_count) if (int(proj_cfg.multi_sample_count) > 1) else 1
    ms_diag: dict[str, float] = {
        "projection/multi_sample/enabled": 1.0 if (int(proj_cfg.multi_sample_count) > 1) else 0.0,
        "projection/multi_sample/samples": float(ms_count),
        "projection/multi_sample/t_sampling_stratified": 0.0,
    }

    def _one_shot_constraint(
        t_batch: Tensor | None = None,
        constraint_seed: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        # The post-AdamW CPO repair backpropagates through ``C_raw`` w.r.t. the policy
        # parameters, so this forward MUST be differentiable. Force grad tracking here
        # (``inference_mode(False)`` + ``enable_grad``) so an ambient inference_mode /
        # no_grad context inherited from the Ray/Lightning rollout or eval utilities
        # cannot silently yield a non-differentiable constraint and crash the backward.
        # Same guard pattern as ``DDIMSampler.sample_with_log_prob``.
        with torch.inference_mode(False), torch.enable_grad():
            (
                _L_cur,
                _L_ref,
                t_frozen,
                model_v,
                _ref_v,
                noise_mask_rep,
                x_t,
                _target_v,
                t_rep,
                batch_rep,
                eps_frozen,
            ) = policy_evaluation_step(
                model,
                ref_model,
                batch,
                candidates_phys,
                K=K,
                shared_noise=shared_noise,
                device=device,
                dtype=dtype,
                t=t_batch,
                eps_rep=None,
                t_min=policy_eval_t_min,
                t_max=policy_eval_t_max,
            )
            C_raw_t, off_diag = _compute_projection_constraint_raw(
                constraint_state=constraint_state,
                model_v=model_v,
                x_t=x_t,
                t_rep=t_rep,
                noise_mask_rep=noise_mask_rep,
                batch_kb=batch_rep,
                core_model=core,
                cartesian=_truth_generation_cartesian(),
                K=K,
                candidate_weights_kb=candidate_weights_kb,
                update_ema=True,
                world_size=world_size,
                constraint_seed=constraint_seed,
            )
        return C_raw_t, off_diag, t_frozen.detach(), eps_frozen.detach()

    if not (int(proj_cfg.multi_sample_count) > 1) or ms_count <= 1:
        seed_0 = _latent_swd_step_seed(constraint_seed_base, 0)
        C_raw_t, off_diag, t_frozen, eps_frozen = _one_shot_constraint(
            constraint_seed=seed_0
        )
        optimizer.zero_grad(set_to_none=True)
        if C_raw_t.requires_grad:
            C_raw_t.backward()
        else:
            _log.warning(
                "[DGPO] projection constraint C_raw is non-differentiable "
                "(requires_grad=False); skipping CPO repair this step. b=0. %s",
                _projection_grad_debug(model),
            )
        # ``b`` is already DDP-averaged by this backward (DDP forward, no ``no_sync``);
        # only the scalar ``C`` needs explicit cross-rank averaging.
        b_flat = flatten_param_grads(model)
        C_raw = float(C_raw_t.detach().float().cpu())
        C_raw = sync_projection_constraint_C_across_ranks(
            C_raw, device=b_flat.device, world_size=world_size,
        )
        ms_diag["projection/multi_sample/C_mean"] = C_raw
        ms_diag["projection/multi_sample/C_std"] = 0.0
        ms_diag["projection/multi_sample/C_min"] = C_raw
        ms_diag["projection/multi_sample/C_max"] = C_raw
        frozen_eval_inputs = [(t_frozen, eps_frozen, seed_0)]
        return b_flat, C_raw, off_diag, frozen_eval_inputs, ms_diag

    off_diag_first: dict[str, Tensor] = {}
    frozen_eval_inputs: list[tuple[Tensor, Tensor, int | None]] = []
    C_vals: list[float] = []
    inv_m = 1.0 / float(ms_count)
    optimizer.zero_grad(set_to_none=True)
    use_stratified_t = (
        True and ms_count > 1
    )
    if use_stratified_t:
        batch_size = int(batch["x"].shape[0])
        t_grid = projection_stratified_t_grid(
            ms_count,
            batch_size,
            t_min=policy_eval_t_min,
            t_max=policy_eval_t_max,
            device=device,
            dtype=dtype,
        )
        ms_diag["projection/multi_sample/t_sampling_stratified"] = 1.0
        ms_diag["projection/multi_sample/t_grid_first"] = float(t_grid[0][0].detach().cpu())
        ms_diag["projection/multi_sample/t_grid_last"] = float(t_grid[-1][0].detach().cpu())
    else:
        t_grid = [None] * ms_count
    for sample_idx in range(ms_count):
        seed_i = _latent_swd_step_seed(constraint_seed_base, sample_idx)
        C_raw_t, off_diag, t_frozen, eps_frozen = _one_shot_constraint(
            t_grid[sample_idx], constraint_seed=seed_i
        )
        C_vals.append(float(C_raw_t.detach().float().cpu()))
        frozen_eval_inputs.append((t_frozen, eps_frozen, seed_i))
        if not off_diag_first:
            off_diag_first = off_diag
        # Sequential (C_i / M).backward() accumulates grad C_bar without retaining M graphs.
        if C_raw_t.requires_grad:
            (C_raw_t * inv_m).backward()
        elif sample_idx == 0:
            _log.warning(
                "[DGPO] projection constraint C_raw is non-differentiable "
                "(requires_grad=False); skipping CPO repair this step. b=0. %s",
                _projection_grad_debug(model),
            )

    # ``b`` is already DDP-averaged across the per-sample backward passes (DDP forward,
    # no ``no_sync``); only the scalar ``C`` needs explicit cross-rank averaging.
    b_flat = flatten_param_grads(model)
    C_mean = float(sum(C_vals) / len(C_vals))
    C_mean = sync_projection_constraint_C_across_ranks(
        C_mean, device=b_flat.device, world_size=world_size,
    )
    if len(C_vals) > 1:
        C_std = float(torch.tensor(C_vals, dtype=torch.float64).std(unbiased=False).cpu())
    else:
        C_std = 0.0
    ms_diag["projection/multi_sample/C_mean"] = C_mean
    ms_diag["projection/multi_sample/C_std"] = C_std
    ms_diag["projection/multi_sample/C_min"] = float(min(C_vals))
    ms_diag["projection/multi_sample/C_max"] = float(max(C_vals))
    if len(C_vals) >= 1:
        ms_diag["projection/multi_sample/C_first"] = float(C_vals[0])
    return b_flat, C_mean, off_diag_first, frozen_eval_inputs, ms_diag


@dataclass
class _ProjectionRepairEstimate:
    """Resolved linear CPO constraint estimator for one projection repair step."""

    b_flat: Tensor
    C_selected: float
    c_margin: float
    violation_for_lambda: float
    b_dot_d0: float
    C_old: float = float("nan")
    C_adam_pred: float = float("nan")
    C_adam_proxy: float = float("nan")
    v_linear: float = float("nan")
    v_direct: float = float("nan")
    linearization_error: float = float("nan")
    b_proxy_d: Tensor | None = None
    off_diag_old: dict[str, Tensor] = field(default_factory=dict)
    off_diag_adam_pre: dict[str, Tensor] = field(default_factory=dict)
    ms_diag: dict[str, float] = field(default_factory=dict)
    proxy_diag: dict[str, float] = field(default_factory=dict)
    proxy_ok: bool = True
    frozen_eval_inputs: list[tuple[Tensor, Tensor, int | None]] = field(default_factory=list)


def _projection_repair_proxy_estimator(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    theta_old: dict[str, Tensor],
    theta_adam: dict[str, Tensor],
    proj_cfg: ProjectionConstraintConfig,
    candidate_weights_kb: Tensor | None,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    optimizer: Any,
    constraint_state: ProjectionConstraintState,
    world_size: int = 1,
    constraint_seed_base: int | None = None,
) -> _ProjectionRepairEstimate:
    """Linear post-Adam CPO estimator: ``v_linear = (C_old + b^T delta0) - epsilon``."""
    eps_v = float(proj_cfg.epsilon)
    d0 = flatten_param_delta(theta_adam, theta_old).to(dtype=torch.float64)

    assign_params_(model, theta_old)
    optimizer.zero_grad(set_to_none=True)
    b_proxy_flat, C_old, off_diag_old, frozen_eval_inputs, ms_diag = (
        _projection_constraint_value_and_grad_at_theta_old(
            model,
            ref_model,
            batch,
            candidates_phys,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            policy_eval_t_min=policy_eval_t_min,
            policy_eval_t_max=policy_eval_t_max,
            candidate_weights_kb=candidate_weights_kb,
            optimizer=optimizer,
            proj_cfg=proj_cfg,
            constraint_state=constraint_state,
            world_size=world_size,
            constraint_seed_base=constraint_seed_base,
        )
    )
    b_proxy_d = b_proxy_flat.to(dtype=torch.float64)
    b_proxy_dot_d0 = float(torch.dot(b_proxy_d, d0).detach().cpu())
    C_adam_pred = float(C_old) + b_proxy_dot_d0
    v_linear = C_adam_pred - eps_v

    assign_params_(model, theta_adam)
    C_adam_proxy, off_diag_adam_pre, proxy_diag = _projection_constraint_C_detached_average(
        model,
        ref_model,
        batch,
        candidates_phys,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        policy_eval_t_min=policy_eval_t_min,
        policy_eval_t_max=policy_eval_t_max,
        candidate_weights_kb=candidate_weights_kb,
        proj_cfg=proj_cfg,
        constraint_state=constraint_state,
        frozen_eval_inputs=frozen_eval_inputs,
        world_size=world_size,
    )
    v_direct = float(C_adam_proxy) - eps_v
    linearization_error = float(C_adam_proxy) - C_adam_pred
    proxy_diag["projection/direct_post_adam_proxy/active"] = 1.0

    return _ProjectionRepairEstimate(
        b_flat=b_proxy_flat,
        C_selected=float(C_old),
        c_margin=float(C_old) - eps_v,
        violation_for_lambda=v_linear,
        b_dot_d0=b_proxy_dot_d0,
        C_old=float(C_old),
        C_adam_pred=C_adam_pred,
        C_adam_proxy=float(C_adam_proxy),
        v_linear=v_linear,
        v_direct=v_direct,
        linearization_error=linearization_error,
        b_proxy_d=b_proxy_d,
        off_diag_old=off_diag_old,
        off_diag_adam_pre=off_diag_adam_pre,
        ms_diag=ms_diag,
        proxy_diag=proxy_diag,
        proxy_ok=math.isfinite(float(C_adam_proxy)),
        frozen_eval_inputs=frozen_eval_inputs,
    )


def _dgpo_projection_repair_after_adamw(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict[str, Any],
    candidates_phys: Any,
    theta_old: dict[str, Tensor],
    proj_cfg: ProjectionConstraintConfig,
    candidate_weights_kb: Tensor | None,
    K: int,
    shared_noise: bool,
    device: torch.device,
    dtype: torch.dtype,
    policy_eval_t_min: float,
    policy_eval_t_max: float,
    optimizer: Any,
    diag_last: dict[str, Tensor],
    constraint_state: ProjectionConstraintState,
    world_size: int = 1,
    constraint_seed_base: int | None = None,
) -> None:
    """AdamW-metric CPO projection repair after the unconstrained AdamW step."""
    theta_adam = snapshot_params(model)
    delta0_flat = flatten_param_delta(theta_adam, theta_old)
    delta0_norm = float(torch.linalg.norm(delta0_flat).detach().cpu())
    eps_v = float(proj_cfg.epsilon)

    est = _projection_repair_proxy_estimator(
        model=model,
        ref_model=ref_model,
        batch=batch,
        candidates_phys=candidates_phys,
        theta_old=theta_old,
        theta_adam=theta_adam,
        proj_cfg=proj_cfg,
        candidate_weights_kb=candidate_weights_kb,
        K=K,
        shared_noise=shared_noise,
        device=device,
        dtype=dtype,
        policy_eval_t_min=policy_eval_t_min,
        policy_eval_t_max=policy_eval_t_max,
        optimizer=optimizer,
        constraint_state=constraint_state,
        world_size=world_size,
        constraint_seed_base=constraint_seed_base,
    )

    b_flat = est.b_flat
    C_selected = est.C_selected
    c_margin = est.c_margin
    violation_for_lambda = est.violation_for_lambda
    b_dot_d0 = est.b_dot_d0
    C_old = est.C_old
    C_adam_pred = est.C_adam_pred
    C_adam_proxy = est.C_adam_proxy
    v_linear = est.v_linear
    v_direct = est.v_direct
    linearization_error = est.linearization_error
    b_proxy_d = est.b_proxy_d
    off_diag_old = est.off_diag_old
    off_diag_adam_pre = est.off_diag_adam_pre
    ms_diag = est.ms_diag
    proxy_diag = est.proxy_diag

    b_d = b_flat.to(dtype=torch.float64)
    p_flat, precond_diag = flatten_adam_preconditioned_direction(
        model,
        b_flat,
        optimizer,
        use_adam_preconditioner=proj_cfg.use_adam_preconditioner,
    )

    assign_params_(model, theta_old)

    lam, proj_diag = compute_projection_lambda_from_violation(
        b_flat,
        p_flat,
        violation_for_lambda,
        proj_cfg,
        C_raw=C_selected,
        b_dot_delta0=b_dot_d0,
    )
    proj_diag.update(precond_diag)
    proj_diag.update(ms_diag)
    proj_diag.update(proxy_diag)
    # Reward<->constraint alignment: cosine of the constraint gradient b with the AdamW reward
    # step delta0 (b_dot_d0 = b.delta0). Persistently > 0 means the reward step keeps PUSHING the
    # constraint up -> a genuine reward<->constraint tension (the constraint must fight reward to
    # bind). Near 0 means they are ~orthogonal, so the constraint should not block reward and a
    # reward stall points elsewhere (coordinate artifact / margin).
    _b_norm_val = float(proj_diag.get("projection/b_norm", 0.0))
    proj_diag["projection/reward_constraint_align"] = float(b_dot_d0) / (
        _b_norm_val * float(delta0_norm) + 1e-12
    )
    proj_diag["projection/epsilon"] = float(eps_v)
    proj_diag["projection/C_old"] = float(C_old)
    proj_diag["projection/C_adam_pred"] = float(C_adam_pred)
    proj_diag["projection/C_adam_proxy"] = float(C_adam_proxy)
    proj_diag["projection/v_linear"] = float(v_linear)
    proj_diag["projection/v_direct"] = float(v_direct)
    if b_proxy_d is not None:
        proj_diag["projection/b_proxy_norm2"] = float(
            torch.dot(b_proxy_d, b_proxy_d).detach().cpu()
        )
    else:
        proj_diag["projection/b_proxy_norm2"] = float("nan")
    proj_diag["projection/v_before"] = float(v_linear)
    proj_diag["projection/v_selected"] = float(violation_for_lambda)
    proj_diag["projection/linearization_error"] = float(linearization_error)
    proj_diag["projection/C_linear_adam"] = float(C_adam_pred)
    proj_diag["projection/linearization_error_adam"] = float(linearization_error)
    proj_diag["projection/delta0_norm"] = delta0_norm
    repair_flat = max(0.0, float(lam)) * p_flat
    repair_norm = float(torch.linalg.norm(repair_flat.to(dtype=torch.float64)).detach().cpu())
    delta_projected_flat = delta0_flat - repair_flat
    delta_projected_norm = float(
        torch.linalg.norm(delta_projected_flat.to(dtype=torch.float64)).detach().cpu()
    )
    proj_diag["projection/repair_norm"] = repair_norm
    proj_diag["projection/delta_projected_norm"] = delta_projected_norm
    d_proj_norm = repair_norm
    proj_diag["projection/d_proj_norm"] = d_proj_norm
    proj_diag["projection/correction_norm_requested"] = d_proj_norm
    proj_diag["projection/cpo_trial/final_update_cap"] = 1.0
    proj_diag["projection/lambda_effective"] = float(lam)

    theta_projected = theta_adam
    proxy_ok = est.proxy_ok
    skip_projection = (
        not math.isfinite(C_old)
        or not proxy_ok
        or not math.isfinite(violation_for_lambda)
        or not torch.isfinite(b_flat).all()
        or not torch.isfinite(p_flat).all()
        or not torch.isfinite(delta_projected_flat).all()
    )
    if skip_projection:
        _log.warning(
            "[DGPO] projection skipped: non-finite C_raw, C_adam_proxy, grad C, or projected delta; "
            "keeping AdamW weights.",
        )
        assign_params_(model, theta_adam)
        proj_diag["projection/applied"] = 0.0
        proj_diag["projection/lambda"] = 0.0
        proj_diag["projection/reverted_nonfinite"] = 1.0
        proj_diag["projection/correction_norm"] = 0.0
        proj_diag["projection/final_update_norm"] = delta0_norm
        proj_diag["projection/final_update_norm_ratio"] = 1.0
        proj_diag["projection/final_update_scale"] = 1.0
    else:
        delta_final, cpo_diag = compute_cpo_adamw_final_update(
            model,
            delta0_flat,
            p_flat,
            lam,
            optimizer,
            proj_cfg,
        )
        proj_diag.update(cpo_diag)
        proj_diag["projection/trust_radius"] = float(
            cpo_diag.get("projection/cpo_trial/trust_radius_adamw", float("nan"))
        )
        proj_diag["projection/trust_scale"] = float(
            cpo_diag.get("projection/cpo_trial/final_update_scale", 1.0)
        )
        proj_diag["projection/trust_cap_active"] = float(
            cpo_diag.get("projection/cpo_trial/final_update_cap_active", 0.0)
        )
        corr_norm = assign_params_from_theta_old_delta_(model, theta_old, delta_final)
        final_update_norm = corr_norm
        proj_diag["projection/correction_norm"] = d_proj_norm
        proj_diag["projection/final_update_norm"] = final_update_norm
        eps_norm = 1e-12
        if delta0_norm > 0.0:
            proj_diag["projection/final_update_norm_ratio"] = final_update_norm / (
                delta0_norm + eps_norm
            )
            proj_diag["projection/correction_to_delta0_ratio"] = d_proj_norm / delta0_norm
        else:
            proj_diag["projection/final_update_norm_ratio"] = 0.0
            proj_diag["projection/correction_to_delta0_ratio"] = 0.0
        proj_diag["projection/final_update_scale"] = float(
            cpo_diag.get("projection/cpo_trial/final_update_scale", 1.0)
        )
        if not trainable_params_all_finite(model):
            _log.warning(
                "[DGPO] CPO projection produced non-finite weights (lambda=%.4g, "
                "final_update_norm_adamw=%.4g); reverting to AdamW step.",
                float(lam),
                float(cpo_diag.get("projection/cpo_trial/final_update_norm_adamw", float("nan"))),
            )
            assign_params_(model, theta_adam)
            proj_diag["projection/reverted_nonfinite"] = 1.0
            proj_diag["projection/applied"] = 0.0
        else:
            theta_projected = snapshot_params(model)
            proj_diag["projection/reverted_nonfinite"] = 0.0
            proj_diag["projection/applied"] = 1.0 if float(lam) > 0.0 else 0.0
            proj_diag["projection/lambda"] = float(lam)
    optimizer.zero_grad(set_to_none=True)

    delta_star_flat = flatten_param_delta(theta_projected, theta_old)
    d_star = delta_star_flat.to(dtype=torch.float64)
    b_dot_d_star = float(torch.dot(b_d, d_star).detach().cpu())
    v_after = c_margin + b_dot_d_star
    C_lin_projected = float(C_selected) + b_dot_d_star
    final_update_norm = float(torch.linalg.norm(delta_star_flat.to(dtype=torch.float64)).detach().cpu())
    if "projection/final_update_norm" not in proj_diag:
        proj_diag["projection/final_update_norm"] = final_update_norm
    if "projection/final_update_norm_ratio" not in proj_diag:
        eps_norm = 1e-12
        if delta0_norm > 0.0:
            proj_diag["projection/final_update_norm_ratio"] = final_update_norm / (
                delta0_norm + eps_norm
            )
        else:
            proj_diag["projection/final_update_norm_ratio"] = 0.0
    if "projection/final_update_scale" not in proj_diag:
        proj_diag["projection/final_update_scale"] = 1.0

    C_adam = float(C_adam_proxy) if math.isfinite(C_adam_proxy) else float("nan")
    off_diag_adam = off_diag_adam_pre
    proj_diag.update(
        {
            "projection/C_adam": float(C_adam),
            "projection/v_after": float(v_after),
            "projection/b_dot_delta_star": float(b_dot_d_star),
            "projection/actual_violation_adam": (
                max(0.0, float(C_adam) - eps_v) if math.isfinite(C_adam) else float("nan")
            ),
            "projection/C_linear_projected": float(C_lin_projected),
        }
    )
    for _off in (off_diag_old, off_diag_adam):
        if _off:
            proj_diag.update(_latent_projection_metrics(_off))

    for k, v in proj_diag.items():
        diag_last[k] = torch.tensor(float(v), device=device, dtype=torch.float64)
    if off_diag_old:
        diag_last.update(off_diag_old)


def train_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ema_rollout: Any | None,
    ema_save: Any | None,
    batch: dict[str, Any],
    optimizer: Any,
    sampler: DDIMSampler,
    reward_agg: RewardAggregator,
    *,
    beta: float,
    K: int,
    advantage_estimator: str = ADVANTAGE_ESTIMATOR_ZSCORE,
    num_ddim_steps: int,
    rollout_parallel_chains: int = 1,
    global_step: int,
    epoch: int,
    device: torch.device,
    dtype: torch.dtype,
    reference_trust_coefficient: float = 0.0,
    reference_trust_objective: str = REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
    reference_trust_vp_path_kl_diagnostic: bool = False,
    reference_trust_vp_logsnr_min: float = -20.0,
    reference_trust_vp_logsnr_max: float = 20.0,
    reference_trust_max_ratio: float | None = None,
    reference_trust_distance: str = "velocity_mse_ratio",
    reference_trust_warning_fraction: float = 0.8,
    reference_trust_backtrack_factor: float = 0.5,
    reference_trust_max_backtracks: int = 16,
    reference_trust_probe_events_per_rank: int = 64,
    reference_trust_fixed_probe_per_round: bool = False,
    reference_trust_interior_fraction: float = 1.0,
    reference_trust_policy_lr_scale: float = 1.0,
    reference_trust_reset_adam_first_moment_on_zero_step: bool = False,
    reference_trust_transactional_rejection: bool = False,
    reference_trust_probe_cache: dict[str, Any] | None = None,
    log_reward_dist: bool = False,
    log_diagnostic_dist: bool = False,
    collect_train_dist: bool = True,
    diagnostic_plot_names: set[str] | None = None,
    diagnostic_plot_every: int = 1,
    num_train_timesteps: int = 1,
    policy_eval_parallel_timesteps: int = 1,
    policy_eval_event_microbatch_size: int | None = None,
    adv_clip_max: float | None = None,
    grad_clip_norm: float = _GRAD_CLIP_NORM,
    policy_eval_t_min: float = 0.0,
    policy_eval_t_max: float = 1.0,
    constraint_state: ProjectionConstraintState | None = None,
    world_size: int = 1,
    extragradient_optimizer_base_params: Mapping[str, Tensor] | None = None,
    parameter_update_rms_target: float | None = None,
    parameter_update_rms_min_scale: float = 0.0,
    parameter_update_rms_max_scale: float = float("inf"),
) -> dict[str, Any]:
    """Rollout once, accumulate gradients, then apply one controlled AdamW update.

    Frozen method: EMA-rollout candidate generation (when the rollout EMA exists),
    shared noise across the K candidates, configured per-event advantages, pure
    DGPO backward, optional strict post-step trust backtracking, and optional
    post-AdamW latent-SWD CPO projection repair.
    """
    # Frozen method constant: all K candidates share the diffusion timestep t and
    # noise eps during policy evaluation (only the DDIM chain index varies).
    shared_noise = True
    endpoint_controller = getattr(_unwrap_core_evenet(model), "_endpoint_kl_controller", None)
    endpoint_active = endpoint_controller is not None and endpoint_controller.config.enabled
    if endpoint_active and extragradient_optimizer_base_params is not None:
        raise ValueError("endpoint KL currently requires the ordinary optimizer step")
    trust_objective = str(reference_trust_objective).strip().lower()
    if trust_objective not in VALID_REFERENCE_TRUST_OBJECTIVES:
        raise ValueError(
            f"unsupported reference trust objective: {trust_objective!r}; "
            f"expected one of {sorted(VALID_REFERENCE_TRUST_OBJECTIVES)}"
        )
    trust_distance = str(reference_trust_distance).strip().lower()
    if trust_distance not in {"velocity_mse_ratio", "vp_path_kl"}:
        raise ValueError(
            "reference_trust_distance must be velocity_mse_ratio or vp_path_kl"
        )
    if trust_distance == "vp_path_kl" and trust_objective != "vp_path_kl":
        raise ValueError(
            "a vp_path_kl hard boundary requires a vp_path_kl trust objective"
        )
    path_kl_evaluation_enabled = bool(
        (reference_trust_coefficient > 0.0 or reference_trust_max_ratio is not None)
        and (
            trust_objective == REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL
            or reference_trust_vp_path_kl_diagnostic
            or trust_distance == "vp_path_kl"
        )
    )
    sequential_vp_trust_backward = bool(
        _dgpo_cfg_get(
            global_config.dgpo,
            "sequential_vp_trust_backward",
            False,
        )
    )
    if sequential_vp_trust_backward:
        if reference_trust_coefficient <= 0.0:
            raise ValueError(
                "dgpo.sequential_vp_trust_backward requires a positive "
                "reference_trust coefficient"
            )
        if trust_objective != REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL:
            raise ValueError(
                "dgpo.sequential_vp_trust_backward requires objective=vp_path_kl"
            )
        if int(policy_eval_parallel_timesteps) != 1:
            raise ValueError(
                "dgpo.sequential_vp_trust_backward currently requires "
                "policy_eval_parallel_timesteps=1"
            )
    model.train()
    ref_model.eval()
    freeze_reference_model(ref_model)
    proj_cfg = resolve_projection_constraint_config(global_config.dgpo)
    projection_active = bool(proj_cfg.active and constraint_state is not None)
    rms_calibration_active = parameter_update_rms_target is not None
    rms_target = (
        float(parameter_update_rms_target)
        if parameter_update_rms_target is not None
        else float("nan")
    )
    rms_min_scale = float(parameter_update_rms_min_scale)
    rms_max_scale = float(parameter_update_rms_max_scale)
    if rms_calibration_active:
        if not math.isfinite(rms_target) or rms_target <= 0.0:
            raise ValueError(
                "parameter_update_rms_target must be finite and positive"
            )
        if (
            not math.isfinite(rms_min_scale)
            or not math.isfinite(rms_max_scale)
            or rms_min_scale <= 0.0
            or rms_max_scale < rms_min_scale
        ):
            raise ValueError(
                "parameter-update RMS scale bounds must be finite, positive, "
                "and ordered"
            )
        if reference_trust_max_ratio is not None or projection_active:
            raise ValueError(
                "parameter-update RMS calibration requires no hard trust boundary "
                "or post-AdamW projection"
            )
        if extragradient_optimizer_base_params is not None:
            raise ValueError(
                "parameter-update RMS calibration requires the ordinary AdamW path"
            )
    if reference_trust_max_ratio is not None and projection_active:
        raise ValueError(
            "strict adaptive trust backtracking is incompatible with the "
            "post-AdamW CPO projection; disable one of the two treatments"
        )
    if not 0.0 < float(reference_trust_interior_fraction) <= 1.0:
        raise ValueError("reference_trust_interior_fraction must lie in (0, 1]")
    trust_policy_lr_scale = float(reference_trust_policy_lr_scale)
    if not math.isfinite(trust_policy_lr_scale) or not (
        0.0 < trust_policy_lr_scale <= 1.0
    ):
        raise ValueError("reference_trust_policy_lr_scale must lie in (0, 1]")
    variance_reg_active = _variance_regularization_enabled(global_config.dgpo)
    variance_reg_weight = _variance_regularization_weight(global_config.dgpo)
    variance_reg_features = _variance_regularization_feature_names(global_config.dgpo)

    core = _unwrap_core_evenet(model)
    gradient_transfer_cfg = resolve_gradient_transfer_trace_config(
        _dgpo_cfg_get(global_config.dgpo, "gradient_transfer_trace", None)
    )
    gradient_transfer_active = gradient_transfer_trace_due(
        gradient_transfer_cfg,
        update_end_step=int(global_step) + 1,
    )
    gradient_transfer_parameters: tuple[Tensor, ...] = ()
    gradient_transfer_h4_accum: list[Tensor] | None = None
    gradient_transfer_ref_accum: list[Tensor] | None = None
    if gradient_transfer_active:
        if trust_objective != REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE:
            raise ValueError(
                "gradient transfer trace currently requires velocity_mse reference trust"
            )
        if reference_trust_coefficient <= 0.0:
            raise ValueError(
                "gradient transfer trace requires a positive reference trust coefficient"
            )
        if projection_active or reference_trust_max_ratio is not None:
            raise ValueError(
                "gradient transfer trace requires no hard trust boundary or projection"
            )
        if sequential_vp_trust_backward:
            raise ValueError(
                "gradient transfer trace is incompatible with sequential VP backward"
            )
        if float(_dgpo_cfg_get(global_config.dgpo, "beta_kl", 0.0)) != 0.0:
            raise ValueError("gradient transfer trace requires dgpo.beta_kl=0")
        if variance_reg_active and variance_reg_weight != 0.0:
            raise ValueError(
                "gradient transfer trace requires variance regularization to be inactive"
            )
        if extragradient_optimizer_base_params is not None:
            raise ValueError(
                "gradient transfer trace requires the ordinary native AdamW update"
            )
        gradient_transfer_parameters = tuple(
            parameter for parameter in core.parameters() if parameter.requires_grad
        )
        if not gradient_transfer_parameters:
            raise RuntimeError("gradient transfer trace found no trainable policy parameters")
        gradient_transfer_h4_accum = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in gradient_transfer_parameters
        ]
        gradient_transfer_ref_accum = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in gradient_transfer_parameters
        ]
    B = int(batch["x"].shape[0])
    if policy_eval_event_microbatch_size is None:
        policy_event_microbatch_size = B
    else:
        policy_event_microbatch_size = int(policy_eval_event_microbatch_size)
        if policy_event_microbatch_size < 1:
            raise ValueError("policy_eval_event_microbatch_size must be positive or null")
        policy_event_microbatch_size = min(B, policy_event_microbatch_size)
    if policy_event_microbatch_size < B and variance_reg_active:
        raise ValueError(
            "policy-evaluation event microbatching is incompatible with the "
            "batch-coupled variance regularizer; disable variance_regularization "
            "or leave policy_eval_event_microbatch_size null"
        )
    policy_event_ranges = [
        (start, min(B, start + policy_event_microbatch_size))
        for start in range(0, B, policy_event_microbatch_size)
    ]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Fully parallel rollout needs one K-fold conditioning batch. It is released
    # immediately afterward; gradient-bearing policy evaluation builds B-sized
    # event chunks instead.
    policy_batch = _dgpo_policy_conditioning_batch(batch)
    rollout_batch_rep = None
    if int(K) > 1 and int(rollout_parallel_chains) >= int(K):
        rollout_batch_rep = repeat_batch_for_candidates(
            policy_batch,
            K,
            tensor_keys=_DGPO_POLICY_BATCH_TENSOR_KEYS,
        )

    # Keep the rollout on the live policy unless YAML explicitly asks for EMA.
    buf = _maybe_install_ema_for_generation(ema_rollout, model, core)
    try:
        candidates_phys = generate_neutrino_candidates(
            core,
            policy_batch,
            sampler,
            K=K,
            num_ddim_steps=num_ddim_steps,
            device=device,
            parallel_chains=rollout_parallel_chains,
            expanded_batch=rollout_batch_rep,
        )
    finally:
        if buf:
            _restore_trainable_weights(model, buf)
    # Rollout is no-grad. Its K-fold conditioning buffer is not needed by the
    # event-microbatched, gradient-bearing policy evaluation below.
    rollout_batch_rep = None
    del policy_batch

    nonfinite_diag: dict[str, float] = {}
    cand_frac = _dgpo_nonfinite_fraction(candidates_phys)
    if cand_frac > 0.0:
        nonfinite_diag["train/candidate_nonfinite_fraction"] = cand_frac
        _log.warning(
            "[DGPO] non-finite generated candidates (frac=%.4g) at global_step=%s; zeroing.",
            cand_frac,
            global_step,
        )
        candidates_phys = _dgpo_zero_nonfinite(candidates_phys)

    reward_batch = dict(batch)
    reward_batch["_dgpo_reward_context"] = "policy_update"
    rewards, reward_breakdown = reward_agg.compute(candidates_phys, reward_batch)
    rewards, reward_breakdown, rew_nonfinite_diag = _dgpo_sanitize_rollout_rewards(
        rewards,
        reward_breakdown,
        global_step=int(global_step),
    )
    nonfinite_diag.update(rew_nonfinite_diag)
    valid_b = get_event_valid_mask(batch, B, device, dtype)
    advantages, _ = compute_per_event_advantage(
        rewards,
        estimator=advantage_estimator,
    )
    advantages = _dgpo_zero_nonfinite(advantages)

    if adv_clip_max is not None:
        advantages = torch.clamp(advantages, -float(adv_clip_max), float(adv_clip_max))

    beta_dgpo = float(beta)
    if int(global_step) == 0:
        if projection_active:
            _log.info(
                "[DGPO] projection_constraint active (latent_swd, frozen encoder): backward "
                "uses pure DGPO main term only. Post-step AdamW-metric CPO on normalized "
                "C_norm = (SWD(z_pred,z_truth) - swd_tt) / (swd_tt + eps) with epsilon=%.4g. "
                "W&B: swd/* panel + projection/* CPO repair (x-axis global_step).",
                float(proj_cfg.epsilon),
            )
        else:
            _log.info(
                "[DGPO] pure DGPO mode: backward and optimizer step run without projection/CPO repair."
            )
        if variance_reg_active and variance_reg_weight > 0.0:
            _log.info(
                "[DGPO] anti-shrink variance regularization enabled: weight=%.4g, features=%s "
                "(defaults come from event_info.yaml unless overridden).",
                variance_reg_weight,
                ",".join(variance_reg_features) if variance_reg_features else "<none>",
            )

    candidate_weights_kb: Tensor | None = None
    if projection_active and proj_cfg.active_apply_to == "best_candidate":
        best_k = rewards.detach().argmax(dim=0)
        cols = torch.arange(B, device=device, dtype=torch.long)
        candidate_weights_kb = torch.zeros_like(rewards, dtype=torch.bool)
        candidate_weights_kb[best_k, cols] = True

    acc_steps = max(1, int(num_train_timesteps))
    parallel_eval_steps = min(
        acc_steps,
        max(1, int(policy_eval_parallel_timesteps)),
    )
    beta_kl = max(
        0.0,
        float(_dgpo_cfg_get(global_config.dgpo, "beta_kl", 0.0)),
    )
    optimizer_ran = False
    trust_probe: _ReferenceTrustProbe | None = None
    trust_probe_reused = False
    if reference_trust_fixed_probe_per_round:
        if reference_trust_probe_cache is None:
            raise ValueError(
                "fixed per-round trust probe requires reference_trust_probe_cache"
            )
        cached_probe = reference_trust_probe_cache.get("probe")
        if cached_probe is not None:
            if not isinstance(cached_probe, _ReferenceTrustProbe):
                raise TypeError("reference trust probe cache contains invalid data")
            trust_probe = cached_probe
            trust_probe_reused = True
    chunk_sizes = {parallel_eval_steps}
    remainder = acc_steps % parallel_eval_steps
    if remainder:
        chunk_sizes.add(remainder)

    def _dgpo_sequential_vp_substep(
        *,
        eval_batch: dict[str, Any],
        eval_candidates: Tensor,
        eval_advantages: Tensor,
        eval_rewards: Tensor,
        prepared_inputs: _PolicyEvaluationInputs,
        trust_weight_correction: Tensor,
        path_kl_stratum_offset: int,
        path_kl_seed: int,
        backward_weight: float,
    ) -> list[tuple[Tensor, dict[str, Tensor]]]:
        """Backward DGPO and VP trust before constructing the next graph.

        Both losses retain their original scalar weights.  Running their
        backward passes one after the other is algebraically identical to
        backward on their sum, while ensuring their saved activations never
        coexist.  Distributed gradients are averaged once after every event
        microbatch/timestep has accumulated.
        """

        local_B = int(eval_batch["x"].shape[0])
        (
            L_cur,
            L_ref,
            _,
            model_v,
            _ref_v,
            noise_mask_rep,
            x_t,
            _target_v,
            t_rep,
            batch_rep,
            _eps_rep,
        ) = policy_evaluation_step(
            core,
            ref_model,
            eval_batch,
            eval_candidates,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            t=None,
            t_min=policy_eval_t_min,
            t_max=policy_eval_t_max,
            num_timesteps=1,
            prepared_inputs=prepared_inputs,
        )
        _dgpo_assert_train_step_invariants(
            L_ref,
            eval_advantages,
            eval_rewards,
        )
        loss_vel, dlast = build_dgpo_loss(
            L_cur,
            L_ref,
            eval_advantages,
            beta_dgpo,
            K,
        )
        kl_loss = L_cur.mean()
        variance_loss = x_t.new_zeros(())
        variance_diag: dict[str, Tensor] = {
            "train/regularization/variance/active": x_t.new_tensor(
                0.0, dtype=torch.float64
            ),
            "train/regularization/variance/active_features": x_t.new_tensor(
                0.0, dtype=torch.float64
            ),
            "train/regularization/variance/raw": x_t.new_tensor(
                0.0, dtype=torch.float64
            ),
        }
        if variance_reg_active and variance_reg_weight > 0.0 and variance_reg_features:
            pred_x0_norm, _, _ = predict_x0_normalized_from_velocity_diffusion(
                x_t,
                model_v,
                t_rep,
            )
            pred_phys = core.invisible_normalizer.denormalize_grad(
                pred_x0_norm,
                mask=noise_mask_rep,
                remove_padding=True,
            )
            truth_phys = batch_rep["x_invisible"][..., : pred_phys.shape[-1]].to(
                device=pred_phys.device,
                dtype=pred_phys.dtype,
            )
            variance_loss, variance_diag = _variance_matching_penalty(
                pred_phys,
                truth_phys,
                noise_mask_rep,
                cartesian=_truth_generation_cartesian(),
                feature_names=_invisible_feature_names(),
                selected_features=variance_reg_features,
            )
        main_loss = (
            loss_vel
            + beta_kl * kl_loss
            + float(variance_reg_weight) * variance_loss
        )
        main_loss_detached = main_loss.detach()
        dlast["loss_velocity_training"] = loss_vel.detach()
        dlast["train/loss/kl"] = (beta_kl * kl_loss.detach()).to(
            dtype=torch.float64
        )
        dlast["train/loss/variance_regularization"] = (
            float(variance_reg_weight) * variance_loss.detach()
        ).to(dtype=torch.float64)
        dlast["kl_weight_mean"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
        dlast["kl_weight_min"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
        dlast["kl_weight_max"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
        dlast.update(variance_diag)

        if not _all_ranks_scalar_finite(
            main_loss,
            device=device,
            world_size=world_size,
        ):
            dlast["loss_total"] = main_loss_detached
            return [(main_loss_detached, dlast)]

        (main_loss * float(backward_weight)).backward()

        # Drop every reference to the first autograd graph before constructing
        # the independent full-time VP path graph.
        del main_loss, loss_vel, kl_loss, variance_loss
        del L_cur, L_ref, model_v, noise_mask_rep, x_t, t_rep, batch_rep

        path_generator = torch.Generator(device=device)
        path_generator.manual_seed(int(path_kl_seed))
        path_t, path_kl_normalizer = sample_cosine_vp_path_kl_timesteps(
            1,
            local_B,
            total_strata=acc_steps,
            stratum_offset=path_kl_stratum_offset,
            device=device,
            dtype=torch.float32,
            logsnr_min=reference_trust_vp_logsnr_min,
            logsnr_max=reference_trust_vp_logsnr_max,
            generator=path_generator,
        )
        path_nu = int(prepared_inputs.candidates_norm.shape[-2])
        path_features = int(prepared_inputs.candidates_norm.shape[-1])
        path_eps = torch.randn(
            1,
            local_B,
            path_nu,
            path_features,
            device=device,
            dtype=dtype,
            generator=path_generator,
        )
        path_eps_rep = (
            path_eps.unsqueeze(1)
            .expand(1, K, local_B, path_nu, path_features)
            .reshape(K * local_B, path_nu, path_features)
        )
        (
            _path_L_cur,
            path_L_ref,
            _path_t_out,
            path_model_v,
            path_ref_v,
            path_noise_mask,
            _path_x_t,
            _path_target_v,
            _path_t_rep,
            _path_batch_rep,
            _path_eps_rep,
        ) = policy_evaluation_step(
            core,
            ref_model,
            eval_batch,
            eval_candidates,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            t=path_t,
            eps_rep=path_eps_rep,
            t_min=0.0,
            t_max=1.0,
            num_timesteps=1,
            prepared_inputs=prepared_inputs,
        )
        trust_loss, trust_diagnostics = build_reference_trust_loss(
            path_model_v,
            path_ref_v,
            path_noise_mask,
            L_ref_2d=path_L_ref,
            objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
            path_kl_normalizer=path_kl_normalizer,
        )
        trust_component = (
            float(reference_trust_coefficient)
            * trust_weight_correction
            * trust_loss
        )
        if not _all_ranks_scalar_finite(
            trust_component,
            device=device,
            world_size=world_size,
        ):
            raise FloatingPointError(
                "non-finite VP trust component after DGPO backward; refusing "
                "to commit a partial sequential gradient"
            )
        (trust_component * float(backward_weight)).backward()

        for trust_key in (
            "reference_trust/loss",
            "reference_trust/velocity_mse",
        ):
            trust_diagnostics[trust_key] = (
                trust_diagnostics[trust_key] * trust_weight_correction
            )
        dlast.update(trust_diagnostics)
        dlast["reference_trust/vp_path_kl_time_mean"] = path_t.detach().mean()
        dlast["reference_trust/vp_path_kl_time_min"] = path_t.detach().min()
        dlast["reference_trust/vp_path_kl_time_max"] = path_t.detach().max()
        dlast["reference_trust/coefficient"] = torch.tensor(
            float(reference_trust_coefficient),
            device=device,
            dtype=torch.float64,
        )
        dlast["reference_trust/objective_vp_path_kl"] = torch.tensor(
            1.0,
            device=device,
            dtype=torch.float64,
        )
        dlast["reference_trust/vp_path_kl_diagnostic_enabled"] = torch.tensor(
            float(reference_trust_vp_path_kl_diagnostic),
            device=device,
            dtype=torch.float64,
        )
        dlast["reference_trust/sequential_backward"] = torch.tensor(
            1.0,
            device=device,
            dtype=torch.float64,
        )
        dlast["projection/active"] = torch.tensor(
            1.0 if projection_active else 0.0,
            device=device,
            dtype=torch.float64,
        )
        dlast["projection/pure_dgpo_backward"] = torch.tensor(
            1.0,
            device=device,
            dtype=torch.float64,
        )
        total_detached = main_loss_detached + trust_component.detach()
        dlast["loss_total"] = total_detached
        return [(total_detached, dlast)]

    def _dgpo_substeps(
        count: int,
        *,
        eval_batch: dict[str, Any],
        eval_candidates: Tensor,
        eval_advantages: Tensor,
        eval_rewards: Tensor,
        prepared_inputs: _PolicyEvaluationInputs,
        trust_weight_correction: Tensor,
        path_kl_stratum_offset: int,
        path_kl_seed: int,
    ) -> list[tuple[Tensor, dict[str, Tensor], Tensor, Tensor]]:
        nonlocal trust_probe
        local_B = int(eval_batch["x"].shape[0])
        (
            L_cur_all,
            L_ref_all,
            _,
            model_v_all,
            ref_v_all,
            noise_mask_all,
            x_t_all,
            _target_v,
            t_rep_all,
            batch_rep_all,
            _eps_rep,
        ) = policy_evaluation_step(
            model,
            ref_model,
            eval_batch,
            eval_candidates,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            t=None,
            t_min=policy_eval_t_min,
            t_max=policy_eval_t_max,
            num_timesteps=count,
            prepared_inputs=prepared_inputs,
        )
        if count == 1:
            L_cur_all = L_cur_all.unsqueeze(0)
            L_ref_all = L_ref_all.unsqueeze(0)
        rows_per_timestep = int(K) * local_B
        model_v_all = model_v_all.reshape(count, rows_per_timestep, *model_v_all.shape[1:])
        ref_v_all = ref_v_all.reshape(count, rows_per_timestep, *ref_v_all.shape[1:])
        noise_mask_all = noise_mask_all.reshape(
            count,
            rows_per_timestep,
            *noise_mask_all.shape[1:],
        )
        x_t_all = x_t_all.reshape(count, rows_per_timestep, *x_t_all.shape[1:])
        t_rep_all = t_rep_all.reshape(count, rows_per_timestep)

        path_L_ref_all: Tensor | None = None
        path_model_v_all: Tensor | None = None
        path_ref_v_all: Tensor | None = None
        path_noise_mask_all: Tensor | None = None
        path_kl_normalizer: float | None = None
        if path_kl_evaluation_enabled:
            path_generator = torch.Generator(device=device)
            path_generator.manual_seed(int(path_kl_seed))
            path_t, path_kl_normalizer = sample_cosine_vp_path_kl_timesteps(
                count,
                local_B,
                total_strata=acc_steps,
                stratum_offset=path_kl_stratum_offset,
                device=device,
                dtype=torch.float32,
                logsnr_min=reference_trust_vp_logsnr_min,
                logsnr_max=reference_trust_vp_logsnr_max,
                generator=path_generator,
            )
            path_nu = int(prepared_inputs.candidates_norm.shape[-2])
            path_features = int(prepared_inputs.candidates_norm.shape[-1])
            path_eps = torch.randn(
                count,
                local_B,
                path_nu,
                path_features,
                device=device,
                dtype=dtype,
                generator=path_generator,
            )
            path_eps_rep = (
                path_eps.unsqueeze(1)
                .expand(count, K, local_B, path_nu, path_features)
                .reshape(count * K * local_B, path_nu, path_features)
            )
            path_context = (
                nullcontext()
                if trust_objective == REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL
                else torch.no_grad()
            )
            with path_context:
                (
                    _path_L_cur_all,
                    path_L_ref_all,
                    _path_t_out,
                    path_model_v_all,
                    path_ref_v_all,
                    path_noise_mask_all,
                    _path_x_t_all,
                    _path_target_v,
                    _path_t_rep_all,
                    _path_batch_rep_all,
                    _path_eps_rep,
                ) = policy_evaluation_step(
                    # The ordinary DGPO evaluation above is the single DDP
                    # wrapper forward for this accumulated backward.  Evaluate
                    # the second-time trust graph through the wrapped module so
                    # DDP does not prepare its reducer twice before one backward;
                    # the parameter hooks still reduce the combined gradient.
                    core,
                    ref_model,
                    eval_batch,
                    eval_candidates,
                    K=K,
                    shared_noise=shared_noise,
                    device=device,
                    dtype=dtype,
                    t=path_t,
                    eps_rep=path_eps_rep,
                    t_min=0.0,
                    t_max=1.0,
                    num_timesteps=count,
                    prepared_inputs=prepared_inputs,
                )
            if count == 1:
                path_L_ref_all = path_L_ref_all.unsqueeze(0)
            path_model_v_all = path_model_v_all.reshape(
                count,
                rows_per_timestep,
                *path_model_v_all.shape[1:],
            )
            path_ref_v_all = path_ref_v_all.reshape(
                count,
                rows_per_timestep,
                *path_ref_v_all.shape[1:],
            )
            path_noise_mask_all = path_noise_mask_all.reshape(
                count,
                rows_per_timestep,
                *path_noise_mask_all.shape[1:],
            )
        outcomes: list[tuple[Tensor, dict[str, Tensor], Tensor, Tensor]] = []
        for timestep_index in range(count):
            L_cur = L_cur_all[timestep_index]
            L_ref = L_ref_all[timestep_index]
            model_v = model_v_all[timestep_index]
            ref_v = ref_v_all[timestep_index]
            noise_mask_rep = noise_mask_all[timestep_index]
            x_t = x_t_all[timestep_index]
            t_rep = t_rep_all[timestep_index]
            row_start = timestep_index * rows_per_timestep
            row_stop = row_start + rows_per_timestep
            batch_rep = {
                key: (
                    value[row_start:row_stop]
                    if isinstance(value, Tensor)
                    and value.ndim > 0
                    and int(value.shape[0]) == count * rows_per_timestep
                    else value
                )
                for key, value in batch_rep_all.items()
            }

            if (
                reference_trust_max_ratio is not None
                and trust_probe is None
                and trust_distance != "vp_path_kl"
            ):
                capture_x_t = x_t
                capture_t_rep = t_rep
                capture_noise_mask = noise_mask_rep
                capture_ref_v = ref_v
                capture_l_ref = L_ref
                capture_batch_rep = batch_rep
                capture_normalizer = None
                captured_probe = _capture_reference_trust_probe(
                    x_t=capture_x_t,
                    t_rep=capture_t_rep,
                    noise_mask_rep=capture_noise_mask,
                    ref_v=capture_ref_v,
                    batch_rep=capture_batch_rep,
                    L_ref_2d=capture_l_ref,
                    K=K,
                    local_batch_size=local_B,
                    max_events=reference_trust_probe_events_per_rank,
                    path_kl_normalizer=capture_normalizer,
                )
                if reference_trust_fixed_probe_per_round:
                    assert reference_trust_probe_cache is not None
                    trust_probe, probe_payload = _gather_reference_trust_probe(
                        captured_probe,
                        device=device,
                        world_size=world_size,
                    )
                    reference_trust_probe_cache["probe"] = trust_probe
                    reference_trust_probe_cache["payload"] = probe_payload
                    reference_trust_probe_cache["dirty"] = True
                else:
                    trust_probe = captured_probe

            _dgpo_assert_train_step_invariants(
                L_ref,
                eval_advantages,
                eval_rewards,
            )
            loss_vel, dlast = build_dgpo_loss(
                L_cur,
                L_ref,
                eval_advantages,
                beta_dgpo,
                K,
            )
            # ``L_cur`` is this exact per-row velocity MSE, already carrying the
            # required gradient graph. Reuse it instead of recomputing the tensor.
            kl_loss = L_cur.mean()

            variance_loss = x_t.new_zeros(())
            variance_diag: dict[str, Tensor] = {
                "train/regularization/variance/active": x_t.new_tensor(
                    0.0, dtype=torch.float64
                ),
                "train/regularization/variance/active_features": x_t.new_tensor(
                    0.0, dtype=torch.float64
                ),
                "train/regularization/variance/raw": x_t.new_tensor(
                    0.0, dtype=torch.float64
                ),
            }
            if variance_reg_active and variance_reg_weight > 0.0 and variance_reg_features:
                pred_x0_norm, _, _ = predict_x0_normalized_from_velocity_diffusion(
                    x_t,
                    model_v,
                    t_rep,
                )
                pred_phys = core.invisible_normalizer.denormalize_grad(
                    pred_x0_norm,
                    mask=noise_mask_rep,
                    remove_padding=True,
                )
                truth_phys = batch_rep["x_invisible"][..., :pred_phys.shape[-1]].to(
                    device=pred_phys.device,
                    dtype=pred_phys.dtype,
                )
                variance_loss, variance_diag = _variance_matching_penalty(
                    pred_phys,
                    truth_phys,
                    noise_mask_rep,
                    cartesian=_truth_generation_cartesian(),
                    feature_names=_invisible_feature_names(),
                    selected_features=variance_reg_features,
                )

            # Backward uses DGPO plus optional policy anchors and soft regularizers.
            # The reference trust term shares this exact (t, eps) draw with L_cur/L_ref.
            loss_backward = (
                loss_vel
                + beta_kl * kl_loss
                + float(variance_reg_weight) * variance_loss
            )
            # The unweighted component is retained only long enough for the
            # opt-in exact transfer trace.  It never changes ``loss_backward``.
            reference_trace_loss = loss_vel.new_zeros(())
            if reference_trust_coefficient > 0.0:
                if trust_objective == REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL:
                    assert path_model_v_all is not None
                    assert path_ref_v_all is not None
                    assert path_noise_mask_all is not None
                    assert path_L_ref_all is not None
                    assert path_kl_normalizer is not None
                    trust_loss, trust_diagnostics = build_reference_trust_loss(
                        path_model_v_all[timestep_index],
                        path_ref_v_all[timestep_index],
                        path_noise_mask_all[timestep_index],
                        L_ref_2d=path_L_ref_all[timestep_index],
                        objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
                        path_kl_normalizer=path_kl_normalizer,
                    )
                else:
                    trust_loss, trust_diagnostics = build_reference_trust_loss(
                        model_v,
                        ref_v,
                        noise_mask_rep,
                        L_ref_2d=L_ref,
                    )
                reference_trace_loss = trust_weight_correction * trust_loss
                loss_backward = loss_backward + (
                    float(reference_trust_coefficient) * reference_trace_loss
                )
                for trust_key in (
                    "reference_trust/loss",
                    "reference_trust/velocity_mse",
                ):
                    trust_diagnostics[trust_key] = (
                        trust_diagnostics[trust_key] * trust_weight_correction
                    )
                dlast.update(trust_diagnostics)
                if (
                    reference_trust_vp_path_kl_diagnostic
                    and trust_objective
                    != REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL
                ):
                    assert path_model_v_all is not None
                    assert path_ref_v_all is not None
                    assert path_noise_mask_all is not None
                    assert path_kl_normalizer is not None
                    _, path_diagnostics = build_reference_trust_loss(
                        path_model_v_all[timestep_index],
                        path_ref_v_all[timestep_index],
                        path_noise_mask_all[timestep_index],
                        objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
                        path_kl_normalizer=path_kl_normalizer,
                    )
                    dlast["reference_trust/vp_path_kl"] = path_diagnostics[
                        "reference_trust/vp_path_kl"
                    ]
                    dlast["reference_trust/vp_path_kl_normalizer"] = (
                        path_diagnostics[
                            "reference_trust/vp_path_kl_normalizer"
                        ]
                    )
                if path_kl_evaluation_enabled:
                    dlast["reference_trust/vp_path_kl_time_mean"] = (
                        path_t[timestep_index].detach().mean()
                    )
                    dlast["reference_trust/vp_path_kl_time_min"] = (
                        path_t[timestep_index].detach().min()
                    )
                    dlast["reference_trust/vp_path_kl_time_max"] = (
                        path_t[timestep_index].detach().max()
                    )
            dlast["reference_trust/coefficient"] = torch.tensor(
                float(reference_trust_coefficient),
                device=device,
                dtype=torch.float64,
            )
            dlast["reference_trust/objective_vp_path_kl"] = torch.tensor(
                float(
                    trust_objective
                    == REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL
                ),
                device=device,
                dtype=torch.float64,
            )
            dlast["reference_trust/vp_path_kl_diagnostic_enabled"] = (
                torch.tensor(
                    float(reference_trust_vp_path_kl_diagnostic),
                    device=device,
                    dtype=torch.float64,
                )
            )
            dlast["loss_velocity_training"] = loss_vel.detach()
            dlast["train/loss/kl"] = (beta_kl * kl_loss.detach()).to(
                dtype=torch.float64
            )
            dlast["train/loss/variance_regularization"] = (
                float(variance_reg_weight) * variance_loss.detach()
            ).to(dtype=torch.float64)
            dlast["loss_total"] = loss_backward.detach()
            dlast["kl_weight_mean"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
            dlast["kl_weight_min"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
            dlast["kl_weight_max"] = x_t.new_tensor(beta_kl, dtype=torch.float64)
            dlast["projection/active"] = torch.tensor(
                1.0 if projection_active else 0.0,
                device=device,
                dtype=torch.float64,
            )
            dlast["projection/pure_dgpo_backward"] = torch.tensor(
                1.0,
                device=device,
                dtype=torch.float64,
            )
            dlast.update(variance_diag)
            outcomes.append(
                (loss_backward, dlast, loss_vel, reference_trace_loss)
            )
        return outcomes

    # Accumulate the sub-step gradients into ONE AdamW update per batch, then run
    # the post-AdamW CPO projection repair.
    grad_norm_pre_clip_max = 0.0
    grad_clip_active_any = False
    diag_last: dict[str, Tensor] = {}
    skipped_substeps = 0
    diags: list[dict[str, Tensor]] = []
    diag_weights: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    theta_old_snap: dict[str, Tensor] | None = None
    total_backward_units = acc_steps * len(policy_event_ranges)
    full_trust_mask_mass = batch["x_invisible_mask"].detach().float().sum()
    for event_chunk_index, (event_start, event_stop) in enumerate(policy_event_ranges):
        eval_batch = _slice_event_batch(
            batch,
            event_start,
            event_stop,
            batch_size=B,
        )
        eval_candidates = candidates_phys[:, event_start:event_stop]
        eval_advantages = advantages[:, event_start:event_stop]
        eval_rewards = rewards[:, event_start:event_stop]
        local_B = event_stop - event_start
        event_weight = float(local_B) / float(B)
        trust_weight_correction = full_trust_mask_mass.new_ones(())
        if (
            reference_trust_coefficient > 0.0
            and trust_objective == REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE
            and full_trust_mask_mass > 0.0
        ):
            local_trust_mask_mass = (
                eval_batch["x_invisible_mask"].detach().float().sum()
            )
            # Reference trust is normalized by valid invisible elements, while
            # DGPO/KL are normalized by events. Correct its local loss so the
            # outer event weight reconstructs the exact full-batch denominator.
            trust_weight_correction = (
                local_trust_mask_mass / full_trust_mask_mass / event_weight
            )

        # Candidate normalization and K-fold conditioning expansion are constant
        # across this event chunk's sampled (t, eps) draws. Reuse them for all
        # accumulated timesteps, then release them before the next event chunk.
        base_policy_inputs = _prepare_policy_evaluation_inputs(
            model,
            eval_batch,
            eval_candidates,
            K=K,
        )
        parallel_input_cache = {
            count: _prepare_parallel_policy_evaluation_inputs(
                base_policy_inputs,
                count,
            )
            for count in chunk_sizes
        }
        if (
            reference_trust_max_ratio is not None
            and trust_distance == "vp_path_kl"
            and trust_probe is None
        ):
            # The gradient estimator is stratified across all accumulated
            # timesteps, but a single stratum is not a valid fixed estimate of
            # the complete reverse-path KL.  Build the round probe once from
            # the full importance distribution instead.  It is graph-free and
            # is reused for every pre/post-commit check in the reward round.
            probe_generator = torch.Generator(device=device)
            probe_generator.manual_seed(
                2_026_090_599
                + int(global_step) * 1_000_003
                + int(event_start) * 10_007
                + (
                    dist.get_rank()
                    if dist.is_available() and dist.is_initialized()
                    else 0
                )
            )
            probe_t, probe_path_kl_normalizer = (
                sample_cosine_vp_path_kl_timesteps(
                    1,
                    local_B,
                    total_strata=1,
                    stratum_offset=0,
                    device=device,
                    dtype=torch.float32,
                    logsnr_min=reference_trust_vp_logsnr_min,
                    logsnr_max=reference_trust_vp_logsnr_max,
                    generator=probe_generator,
                )
            )
            probe_nu = int(base_policy_inputs.candidates_norm.shape[-2])
            probe_features = int(base_policy_inputs.candidates_norm.shape[-1])
            probe_eps = torch.randn(
                1,
                local_B,
                probe_nu,
                probe_features,
                device=device,
                dtype=dtype,
                generator=probe_generator,
            )
            probe_eps_rep = (
                probe_eps.unsqueeze(1)
                .expand(1, K, local_B, probe_nu, probe_features)
                .reshape(K * local_B, probe_nu, probe_features)
            )
            with torch.no_grad():
                (
                    _probe_L_cur,
                    probe_L_ref,
                    _probe_t_out,
                    _probe_model_v,
                    probe_ref_v,
                    probe_noise_mask,
                    probe_x_t,
                    _probe_target_v,
                    probe_t_rep,
                    probe_batch_rep,
                    _probe_eps_rep,
                ) = policy_evaluation_step(
                    core,
                    ref_model,
                    eval_batch,
                    eval_candidates,
                    K=K,
                    shared_noise=shared_noise,
                    device=device,
                    dtype=dtype,
                    t=probe_t,
                    eps_rep=probe_eps_rep,
                    t_min=0.0,
                    t_max=1.0,
                    num_timesteps=1,
                    prepared_inputs=base_policy_inputs,
                )
            captured_probe = _capture_reference_trust_probe(
                x_t=probe_x_t,
                t_rep=probe_t_rep,
                noise_mask_rep=probe_noise_mask,
                ref_v=probe_ref_v,
                batch_rep=probe_batch_rep,
                L_ref_2d=probe_L_ref,
                K=K,
                local_batch_size=local_B,
                max_events=reference_trust_probe_events_per_rank,
                path_kl_normalizer=probe_path_kl_normalizer,
            )
            if reference_trust_fixed_probe_per_round:
                assert reference_trust_probe_cache is not None
                trust_probe, probe_payload = _gather_reference_trust_probe(
                    captured_probe,
                    device=device,
                    world_size=world_size,
                )
                reference_trust_probe_cache["probe"] = trust_probe
                reference_trust_probe_cache["payload"] = probe_payload
                reference_trust_probe_cache["dirty"] = True
            else:
                trust_probe = captured_probe
        del base_policy_inputs

        completed_substeps = 0
        while completed_substeps < acc_steps:
            chunk_size = min(parallel_eval_steps, acc_steps - completed_substeps)
            if sequential_vp_trust_backward:
                # The sequential path evaluates through ``core`` and performs
                # both backwards here.  DDP reduction happens once after every
                # event chunk and timestep has accumulated.
                chunk_outcomes = _dgpo_sequential_vp_substep(
                    eval_batch=eval_batch,
                    eval_candidates=eval_candidates,
                    eval_advantages=eval_advantages,
                    eval_rewards=eval_rewards,
                    prepared_inputs=parallel_input_cache[chunk_size],
                    trust_weight_correction=trust_weight_correction,
                    path_kl_stratum_offset=completed_substeps,
                    path_kl_seed=(
                        2_026_090_501
                        + int(global_step) * 1_000_003
                        + int(event_start) * 10_007
                        + int(completed_substeps) * 101
                        + (
                            dist.get_rank()
                            if dist.is_available() and dist.is_initialized()
                            else 0
                        )
                    ),
                    backward_weight=event_weight / float(acc_steps),
                )
                for chunk_offset, (loss, dlast) in enumerate(chunk_outcomes):
                    sub = completed_substeps + chunk_offset + 1
                    diags.append(dlast)
                    diag_weights.append(event_weight)
                    if not torch.isfinite(loss):
                        skipped_substeps += 1
                        _log.warning(
                            "[DGPO] non-finite substep loss (%s); skipping backward "
                            "(step=%s events=%s:%s sub=%s/%s).",
                            float(loss.detach().float().cpu()),
                            global_step,
                            event_start,
                            event_stop,
                            sub,
                            acc_steps,
                        )
            else:
                is_last_backward = (
                    event_chunk_index + 1 == len(policy_event_ranges)
                    and completed_substeps + chunk_size == acc_steps
                )
                ctx = (
                    model.no_sync()
                    if isinstance(model, DDP)
                    and (endpoint_active or gradient_transfer_active or not is_last_backward)
                    else nullcontext()
                )
                with ctx:
                    chunk_outcomes = _dgpo_substeps(
                        chunk_size,
                        eval_batch=eval_batch,
                        eval_candidates=eval_candidates,
                        eval_advantages=eval_advantages,
                        eval_rewards=eval_rewards,
                        prepared_inputs=parallel_input_cache[chunk_size],
                        trust_weight_correction=trust_weight_correction,
                        path_kl_stratum_offset=completed_substeps,
                        path_kl_seed=(
                            2_026_090_501
                            + int(global_step) * 1_000_003
                            + int(event_start) * 10_007
                            + int(completed_substeps) * 101
                            + (
                                dist.get_rank()
                                if dist.is_available() and dist.is_initialized()
                                else 0
                            )
                        ),
                    )
                    finite_losses: list[Tensor] = []
                    finite_h4_losses: list[Tensor] = []
                    finite_ref_losses: list[Tensor] = []
                    for chunk_offset, (
                        loss,
                        dlast,
                        h4_component,
                        ref_component,
                    ) in enumerate(chunk_outcomes):
                        sub = completed_substeps + chunk_offset + 1
                        diags.append(dlast)
                        diag_weights.append(event_weight)
                        if not torch.isfinite(loss):
                            skipped_substeps += 1
                            _log.warning(
                                "[DGPO] non-finite substep loss (%s); skipping backward "
                                "(step=%s events=%s:%s sub=%s/%s).",
                                float(loss.detach().float().cpu()),
                                global_step,
                                event_start,
                                event_stop,
                                sub,
                                acc_steps,
                            )
                            continue
                        finite_losses.append(loss)
                        finite_h4_losses.append(h4_component)
                        finite_ref_losses.append(ref_component)
                    if finite_losses:
                        backward_weight = event_weight / float(acc_steps)
                        total_chunk_loss = (
                            torch.stack(finite_losses).sum()
                            * backward_weight
                        )
                        if gradient_transfer_active:
                            assert gradient_transfer_h4_accum is not None
                            assert gradient_transfer_ref_accum is not None
                            h4_chunk_loss = (
                                torch.stack(finite_h4_losses).sum()
                                * backward_weight
                            )
                            ref_chunk_loss = (
                                torch.stack(finite_ref_losses).sum()
                                * backward_weight
                            )
                            h4_grads = torch.autograd.grad(
                                h4_chunk_loss,
                                gradient_transfer_parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            with torch.no_grad():
                                for accumulator, gradient in zip(
                                    gradient_transfer_h4_accum,
                                    h4_grads,
                                    strict=True,
                                ):
                                    if gradient is not None:
                                        accumulator.add_(gradient.detach().float())
                            del h4_grads, h4_chunk_loss
                            ref_grads = torch.autograd.grad(
                                ref_chunk_loss,
                                gradient_transfer_parameters,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            with torch.no_grad():
                                for accumulator, gradient in zip(
                                    gradient_transfer_ref_accum,
                                    ref_grads,
                                    strict=True,
                                ):
                                    if gradient is not None:
                                        accumulator.add_(gradient.detach().float())
                            del ref_grads, ref_chunk_loss
                        total_chunk_loss.backward()
            completed_substeps += chunk_size
        del parallel_input_cache
    endpoint_metrics = {}
    if endpoint_active:
        # All policy backwards stay local until BOTH losses have accumulated.
        # This also permits multiple DDIM forwards without re-arming DDP.
        endpoint_ctx = model.no_sync() if isinstance(model, DDP) else nullcontext()
        with endpoint_ctx:
            endpoint_metrics = endpoint_controller.backward(
                model=core, reference=ref_model, source=reward_agg.omnifold_source,
                batch=batch, sampler=sampler, num_ddim_steps=num_ddim_steps,
                device=device, global_step=global_step, valid=valid_b,
                reduce_gradients=lambda critic: _all_reduce_accumulated_gradients(critic, world_size=world_size),
            )
    manually_reduced_gradient_tensors = 0
    if sequential_vp_trust_backward or gradient_transfer_active or endpoint_active:
        manually_reduced_gradient_tensors = _all_reduce_accumulated_gradients(
            core,
            world_size=world_size,
        )
    gradient_transfer_vectors: dict[str, Tensor] | None = None
    if gradient_transfer_active:
        assert gradient_transfer_h4_accum is not None
        assert gradient_transfer_ref_accum is not None

        def _flatten_and_average_trace(parts: Sequence[Tensor]) -> Tensor:
            vector = torch.cat([part.reshape(-1) for part in parts])
            if world_size > 1:
                if not dist.is_initialized():
                    raise RuntimeError(
                        "distributed gradient transfer trace requires an initialized process group"
                    )
                dist.all_reduce(vector, op=dist.ReduceOp.SUM)
                vector.div_(float(world_size))
            return vector

        gradient_transfer_h4_flat = _flatten_and_average_trace(
            gradient_transfer_h4_accum
        )
        del gradient_transfer_h4_accum
        gradient_transfer_ref_flat = _flatten_and_average_trace(
            gradient_transfer_ref_accum
        )
        del gradient_transfer_ref_accum
        gradient_transfer_vectors = {
            "h4": gradient_transfer_h4_flat,
            "reference": gradient_transfer_ref_flat,
            # The actual accumulated gradient was synchronized above before
            # clipping. ``flatten_param_grads`` returns independent storage.
            "actual_unclipped": flatten_param_grads(core).detach().float(),
        }
    diag_last = _weighted_mean_diag_dict(diags, diag_weights)
    if endpoint_active:
        # Add the endpoint component once, not once per denoising timestep.
        diag_last["loss_total"] = diag_last["loss_total"] + endpoint_metrics["endpoint_kl/weighted_loss"]
    diag_last["reference_trust/sequential_backward"] = torch.tensor(
        float(sequential_vp_trust_backward),
        device=device,
        dtype=torch.float64,
    )
    diag_last["train/gradient_sync/manual_parameter_tensors"] = torch.tensor(
        float(manually_reduced_gradient_tensors),
        device=device,
        dtype=torch.float64,
    )
    extragradient_rebased_params = 0
    if extragradient_optimizer_base_params is not None:
        if reference_trust_max_ratio is not None:
            raise ValueError(
                "extragradient optimizer rebasing owns its post-step trust "
                "audit; disable train_step strict trust for the corrector"
            )
        extragradient_rebased_params = (
            rebase_trainable_params_for_extragradient_(
                core,
                extragradient_optimizer_base_params,
            )
        )
    strict_trust_enabled = reference_trust_max_ratio is not None
    trust_pre_distance = float("nan")
    trust_candidate_distance = float("nan")
    trust_post_distance = float("nan")
    trust_probe_velocity_mse = float("nan")
    trust_probe_reference_loss = float("nan")
    trust_initial_scale = 1.0
    trust_accepted_scale = 1.0
    trust_backtrack_steps = 0
    trust_boundary_hit = False
    preexisting_trust_violation = False
    trust_interior_saturated = False
    trust_nominal_interior_limit = float("nan")
    trust_acceptance_limit = float("nan")
    trust_optimizer_state_advanced = False
    trust_optimizer_state_restored = False
    trust_zero_step_adam_reset_count = 0
    if strict_trust_enabled:
        if trust_probe is None:
            raise RuntimeError(
                "strict adaptive trust requires at least one finite policy-evaluation probe"
            )
        (
            trust_pre_distance,
            trust_probe_velocity_mse,
            trust_probe_reference_loss,
        ) = _measure_reference_trust_probe(
            model,
            trust_probe,
            world_size=world_size,
            distance=trust_distance,
        )
        delta_f = float(reference_trust_max_ratio)
        trust_nominal_interior_limit = (
            float(reference_trust_interior_fraction) * delta_f
        )
        # A restored legacy checkpoint may begin outside the new interior target
        # while still satisfying the hard radius. In that one case, require the
        # candidate not to move farther out; fixed-probe steps then converge to
        # the configured interior instead of making the run unrecoverable.
        trust_acceptance_limit = min(
            delta_f,
            max(trust_nominal_interior_limit, trust_pre_distance),
        )
        preexisting_trust_violation = bool(
            not math.isfinite(trust_pre_distance)
            or trust_pre_distance >= delta_f
        )
        if preexisting_trust_violation:
            trust_initial_scale = 0.0
        else:
            trust_initial_scale, trust_interior_saturated = (
                adaptive_trust_update_scale(
                    trust_pre_distance,
                    delta=trust_acceptance_limit,
                    warning_fraction=float(reference_trust_warning_fraction),
                )
            )
        trust_accepted_scale = trust_initial_scale
        ratio_for_diag = torch.tensor(
            trust_pre_distance,
            device=device,
            dtype=torch.float64,
        )
        diag_last["reference_trust/pre_step_distance"] = ratio_for_diag
        diag_last["reference_trust/probe_velocity_mse"] = ratio_for_diag.new_tensor(
            trust_probe_velocity_mse
        )
        diag_last["reference_trust/probe_reference_loss"] = ratio_for_diag.new_tensor(
            trust_probe_reference_loss
        )
        diag_last["reference_trust/delta"] = ratio_for_diag.new_tensor(
            float(reference_trust_max_ratio)
        )
        diag_last["reference_trust/warning_ratio"] = ratio_for_diag.new_tensor(
            float(reference_trust_warning_fraction)
            * trust_acceptance_limit
        )
        diag_last["reference_trust/boundary_enabled"] = ratio_for_diag.new_tensor(1.0)
        diag_last["reference_trust/interior_fraction"] = ratio_for_diag.new_tensor(
            float(reference_trust_interior_fraction)
        )
        diag_last["reference_trust/nominal_interior_limit"] = (
            ratio_for_diag.new_tensor(trust_nominal_interior_limit)
        )
        diag_last["reference_trust/acceptance_limit"] = ratio_for_diag.new_tensor(
            trust_acceptance_limit
        )
        diag_last["reference_trust/fixed_probe_per_reward_round"] = (
            ratio_for_diag.new_tensor(
                float(reference_trust_fixed_probe_per_round)
            )
        )
        diag_last["reference_trust/probe_reused"] = ratio_for_diag.new_tensor(
            float(trust_probe_reused)
        )

    gn, clip_on = _grad_norm_pre_clip_and_clip_active(
        model, float(grad_clip_norm)
    )
    grad_norm_pre_clip_max = gn
    grad_clip_active_any = clip_on > 0.5
    gradient_transfer_theta_old = (
        snapshot_params(core) if gradient_transfer_active else None
    )
    theta_old_snap: dict[str, Tensor] | None = None
    if preexisting_trust_violation or trust_interior_saturated:
        trust_boundary_hit = True
        trust_accepted_scale = 0.0
        trust_post_distance = trust_pre_distance
        if preexisting_trust_violation:
            _log.info(
                "[DGPO/trust] fixed probe is already outside the hard boundary at "
                "epoch=%s global_step=%s: D_pre=%.6g delta=%.6g; rejecting step.",
                epoch,
                global_step,
                trust_pre_distance,
                float(reference_trust_max_ratio),
            )
        else:
            _log.info(
                "[DGPO/trust] fixed probe exhausted the interior budget at "
                "epoch=%s global_step=%s: D_pre=%.6g target=%.6g delta=%.6g; "
                "holding policy for the adaptive audit.",
                epoch,
                global_step,
                trust_pre_distance,
                trust_acceptance_limit,
                float(reference_trust_max_ratio),
            )
        optimizer.zero_grad(set_to_none=True)
        if reference_trust_reset_adam_first_moment_on_zero_step:
            trust_zero_step_adam_reset_count = reset_adam_first_moment(optimizer)
    elif skipped_substeps == total_backward_units or not math.isfinite(gn):
        if strict_trust_enabled:
            trust_accepted_scale = 0.0
            trust_post_distance = trust_pre_distance
        _log.warning(
            "[DGPO] all substeps non-finite or grad-norm non-finite (%s); "
            "skipping optimizer.step at global_step=%s.",
            gn, global_step,
        )
        optimizer.zero_grad(set_to_none=True)
    else:
        # Strict trust needs theta_old so a rejected candidate can be replaced by
        # a smaller point on the exact same AdamW direction. Adam moments do not
        # depend on LR, and AdamW's parameter displacement is linear in LR, so
        # this interpolation is equivalent to rerunning the step at the accepted
        # smaller LR without copying the full optimizer state.
        theta_old_snap = (
            snapshot_params(model)
            if (
                strict_trust_enabled
                or projection_active
                or rms_calibration_active
                or bool(
                    _dgpo_cfg_get(
                        global_config.dgpo,
                        "log_parameter_update_rms",
                        False,
                    )
                )
            )
            else None
        )
        original_group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        optimizer_transaction = (
            snapshot_adamw_state_for_transaction(optimizer)
            if strict_trust_enabled and reference_trust_transactional_rejection
            else None
        )
        if trust_policy_lr_scale * trust_initial_scale < 1.0:
            for group, original_lr in zip(
                optimizer.param_groups, original_group_lrs, strict=True
            ):
                group["lr"] = (
                    original_lr * trust_policy_lr_scale * trust_initial_scale
                )
        try:
            optimizer.step()
            trust_optimizer_state_advanced = True
        finally:
            for group, original_lr in zip(
                optimizer.param_groups, original_group_lrs, strict=True
            ):
                group["lr"] = original_lr
        optimizer_ran = True
        if rms_calibration_active:
            assert theta_old_snap is not None
            theta_candidate = snapshot_params(model)
            proposed_rms = _trainable_direction_rms(
                theta_old_snap,
                theta_candidate,
            )
            if not math.isfinite(proposed_rms) or proposed_rms <= 0.0:
                raise RuntimeError(
                    "AdamW proposed a zero or non-finite parameter displacement; "
                    "cannot apply RMS calibration"
                )
            rms_scale = rms_target / proposed_rms
            if not rms_min_scale <= rms_scale <= rms_max_scale:
                raise RuntimeError(
                    "required parameter-update RMS scale lies outside the "
                    f"predeclared bounds: target={rms_target:.9g}, "
                    f"proposed={proposed_rms:.9g}, scale={rms_scale:.9g}, "
                    f"bounds=[{rms_min_scale:.9g}, {rms_max_scale:.9g}]"
                )
            assign_scaled_trainable_update_(
                model,
                theta_old_snap,
                theta_candidate,
                rms_scale,
            )
            applied_rms = _trainable_direction_rms(
                theta_old_snap,
                snapshot_params(model),
            )
            relative_error = abs(applied_rms - rms_target) / rms_target
            if not math.isfinite(applied_rms) or relative_error > 5.0e-4:
                raise RuntimeError(
                    "parameter-update RMS calibration missed its target: "
                    f"target={rms_target:.9g}, applied={applied_rms:.9g}, "
                    f"relative_error={relative_error:.6g}"
                )
            metric_device = device
            diag_last.update({
                "parameter_update_rms_calibration/enabled": torch.tensor(
                    1.0, device=metric_device, dtype=torch.float64
                ),
                "parameter_update_rms_calibration/target_rms": torch.tensor(
                    rms_target, device=metric_device, dtype=torch.float64
                ),
                "parameter_update_rms_calibration/proposed_rms": torch.tensor(
                    proposed_rms, device=metric_device, dtype=torch.float64
                ),
                "parameter_update_rms_calibration/applied_rms": torch.tensor(
                    applied_rms, device=metric_device, dtype=torch.float64
                ),
                "parameter_update_rms_calibration/scale": torch.tensor(
                    rms_scale, device=metric_device, dtype=torch.float64
                ),
                "parameter_update_rms_calibration/relative_error": torch.tensor(
                    relative_error, device=metric_device, dtype=torch.float64
                ),
            })
        if strict_trust_enabled:
            assert theta_old_snap is not None
            (
                trust_candidate_distance,
                _,
                _,
            ) = _measure_reference_trust_probe(
                model,
                trust_probe,
                world_size=world_size,
                distance=trust_distance,
            )
            trust_post_distance = trust_candidate_distance
            delta_f = float(reference_trust_max_ratio)
            acceptance_tolerance = max(1.0e-12, 1.0e-6 * delta_f)
            if (
                not math.isfinite(trust_candidate_distance)
                or trust_candidate_distance
                > trust_acceptance_limit + acceptance_tolerance
            ):
                trust_boundary_hit = True
                theta_candidate = snapshot_params(model)
                accepted = False
                for backtrack_index, absolute_scale in enumerate(
                    adaptive_trust_backtracking_scales(
                        trust_initial_scale,
                        factor=reference_trust_backtrack_factor,
                        max_backtracks=reference_trust_max_backtracks,
                    ),
                    start=1,
                ):
                    _assign_interpolated_trainable_params_(
                        model,
                        theta_old_snap,
                        theta_candidate,
                        absolute_scale / trust_initial_scale,
                    )
                    trial_distance, _, _ = _measure_reference_trust_probe(
                        model,
                        trust_probe,
                        world_size=world_size,
                        distance=trust_distance,
                    )
                    if (
                        math.isfinite(trial_distance)
                        and trial_distance
                        <= trust_acceptance_limit + acceptance_tolerance
                    ):
                        trust_accepted_scale = float(absolute_scale)
                        trust_backtrack_steps = int(backtrack_index)
                        trust_post_distance = float(trial_distance)
                        accepted = True
                        break
                if not accepted:
                    assign_params_(model, theta_old_snap)
                    if optimizer_transaction is not None:
                        restore_adamw_state_from_transaction_(
                            optimizer,
                            optimizer_transaction,
                        )
                        trust_optimizer_state_advanced = False
                        trust_optimizer_state_restored = True
                    optimizer.zero_grad(set_to_none=True)
                    if reference_trust_reset_adam_first_moment_on_zero_step:
                        trust_zero_step_adam_reset_count = (
                            reset_adam_first_moment(optimizer)
                        )
                    # Zero displacement is always feasible. Treat failure to
                    # find a positive line-search scale as a safely rejected
                    # policy proposal instead of aborting the distributed job.
                    # The opt-in transactional path restores every AdamW moment
                    # and step counter; EMA and the LR scheduler never advance
                    # because no parameter update was committed.
                    optimizer_ran = False
                    trust_accepted_scale = 0.0
                    trust_backtrack_steps = int(reference_trust_max_backtracks)
                    trust_post_distance = trust_pre_distance
                    _log.warning(
                        "[DGPO/trust] no positive step satisfied the boundary at "
                        "epoch=%s global_step=%s after %s trials; restored "
                        "theta_old and committed alpha=0 (D_pre=%.6g target=%.6g "
                        "delta=%.6g).",
                        epoch,
                        global_step,
                        int(reference_trust_max_backtracks),
                        trust_pre_distance,
                        trust_acceptance_limit,
                        delta_f,
                    )
                else:
                    _log.info(
                        "[DGPO/trust] candidate exceeded boundary at epoch=%s "
                        "global_step=%s: D_candidate=%.6g delta=%.6g target=%.6g; "
                        "accepted scale=%.6g after %s backtrack(s), D_post=%.6g.",
                        epoch,
                        global_step,
                        trust_candidate_distance,
                        delta_f,
                        trust_acceptance_limit,
                        trust_accepted_scale,
                        trust_backtrack_steps,
                        trust_post_distance,
                    )

    if strict_trust_enabled:
        metric_device = device
        metric_dtype = torch.float64
        control_distance = (
            trust_post_distance
            if math.isfinite(trust_post_distance)
            else trust_pre_distance
        )
        trust_metrics = {
            "reference_trust/control_ratio_global": control_distance,
            "reference_trust/candidate_distance": trust_candidate_distance,
            "reference_trust/candidate_distance_over_delta": (
                trust_candidate_distance / float(reference_trust_max_ratio)
                if math.isfinite(trust_candidate_distance)
                else float("nan")
            ),
            "reference_trust/candidate_excess": (
                trust_candidate_distance - float(reference_trust_max_ratio)
                if math.isfinite(trust_candidate_distance)
                else float("nan")
            ),
            "reference_trust/post_step_distance": trust_post_distance,
            "reference_trust/acceptance_limit": trust_acceptance_limit,
            "reference_trust/post_step_distance_over_delta": (
                trust_post_distance / float(reference_trust_max_ratio)
                if math.isfinite(trust_post_distance)
                else float("nan")
            ),
            "reference_trust/update_scale": trust_accepted_scale,
            "reference_trust/accepted_step_scale": trust_accepted_scale,
            "reference_trust/backtrack_steps": float(trust_backtrack_steps),
            "reference_trust/boundary_hit": float(trust_boundary_hit),
            "reference_trust/preexisting_violation": float(
                preexisting_trust_violation
            ),
            "reference_trust/interior_saturated": float(
                trust_interior_saturated
            ),
            "reference_trust/step_accepted": float(optimizer_ran),
            "reference_trust/optimizer_state_advanced": float(
                trust_optimizer_state_advanced
            ),
            "reference_trust/optimizer_state_restored": float(
                trust_optimizer_state_restored
            ),
            "reference_trust/zero_step_adam_first_moments_reset": float(
                trust_zero_step_adam_reset_count
            ),
            "reference_trust/policy_lr/scale": trust_policy_lr_scale,
            "reference_trust/policy_lr/effective_step_scale": (
                trust_policy_lr_scale * trust_accepted_scale
            ),
        }
        for metric_name, metric_value in trust_metrics.items():
            diag_last[metric_name] = torch.tensor(
                metric_value,
                device=metric_device,
                dtype=metric_dtype,
            )
    if optimizer_ran and theta_old_snap is not None and projection_active:
        _dgpo_projection_repair_after_adamw(
            model=model,
            ref_model=ref_model,
            batch=batch,
            candidates_phys=candidates_phys,
            theta_old=theta_old_snap,
            proj_cfg=proj_cfg,
            candidate_weights_kb=candidate_weights_kb,
            K=K,
            shared_noise=shared_noise,
            device=device,
            dtype=dtype,
            policy_eval_t_min=policy_eval_t_min,
            policy_eval_t_max=policy_eval_t_max,
            optimizer=optimizer,
            diag_last=diag_last,
            constraint_state=constraint_state,
            world_size=world_size,
            constraint_seed_base=int(global_step),
        )

    gradient_transfer_metrics: dict[str, float] = {}
    if gradient_transfer_active:
        assert gradient_transfer_vectors is not None
        assert gradient_transfer_theta_old is not None
        adamw_descent = None
        if optimizer_ran:
            adamw_delta = flatten_param_delta(
                snapshot_params(core),
                gradient_transfer_theta_old,
            ).detach().float()
            adamw_descent = -adamw_delta
        gradient_transfer_metrics = summarize_gradient_transfer(
            h4_gradient=gradient_transfer_vectors["h4"],
            reference_gradient=gradient_transfer_vectors["reference"],
            actual_unclipped_gradient=gradient_transfer_vectors[
                "actual_unclipped"
            ],
            trust_coefficient=float(reference_trust_coefficient),
            counterfactual_coefficients=(
                gradient_transfer_cfg.counterfactual_trust_coefficients
            ),
            update_start_step=int(global_step),
            update_end_step=int(global_step) + 1,
            norm_floor=float(gradient_transfer_cfg.norm_floor),
            adamw_descent=adamw_descent,
        )
        gradient_transfer_metrics["gradient_transfer/optimizer_step_ran"] = (
            float(optimizer_ran)
        )
        del gradient_transfer_vectors, gradient_transfer_theta_old, adamw_descent

    if optimizer_ran and ema_rollout is not None:
        ema_rollout.update(core, decay_=_dgpo_rollout_ema_decay(global_step))
    if optimizer_ran and ema_save is not None:
        ema_cfg = global_config.options.Training.get("EMA", None) or {}
        ema_every_n = max(1, int(ema_cfg.get("update_every_n_steps", 1)))
        if global_step % ema_every_n == 0:
            ema_save.update(core)

    out: dict[str, Any] = _build_train_metrics(
        diag_last,
        rewards,
        valid_b,
        advantages=advantages,
    )
    out["train/policy_eval/parallel_timesteps"] = float(parallel_eval_steps)
    out.update(endpoint_metrics)
    out["train/policy_eval/event_microbatch_size"] = float(
        policy_event_microbatch_size
    )
    out["train/policy_eval/event_microbatches"] = float(len(policy_event_ranges))
    if device.type == "cuda":
        gib = float(1024**3)
        out["train/memory/peak_allocated_gib"] = (
            float(torch.cuda.max_memory_allocated(device)) / gib
        )
        out["train/memory/peak_reserved_gib"] = (
            float(torch.cuda.max_memory_reserved(device)) / gib
        )
        out["train/memory/device_total_gib"] = (
            float(torch.cuda.get_device_properties(device).total_memory) / gib
        )

    if log_reward_dist and not _wandb_critical_enabled():
        out.update(build_reward_distribution_histograms(rewards, valid_b))
    plot_names = diagnostic_plot_names or set()
    plot_every = max(1, int(diagnostic_plot_every))
    log_diag_images = bool(log_diagnostic_dist) and not _wandb_critical_enabled() and (
        int(global_step) % plot_every == 0
    )
    collect_profile_accum = log_diag_images and ("pt_profile_accumulated" in plot_names)
    compact_wandb = _wandb_simplified_enabled()
    out.update(
        _build_reward_extra_metrics(
            rewards,
            valid_b,
            reward_agg,
            reward_breakdown,
            log_distribution=log_diag_images,
            collect_profile_accum=collect_profile_accum,
            diagnostic_plot_names=plot_names,
            compact=compact_wandb,
        )
    )
    if not compact_wandb:
        out.update(
            _build_reference_bias_metrics(
                candidates_phys,
                batch,
                cartesian=_truth_generation_cartesian(),
                log_distribution=log_diag_images,
                diagnostic_plot_names=plot_names,
            )
        )
    out["train/grad/global_norm_pre_clip"] = float(grad_norm_pre_clip_max)
    out["train/grad/clip_active"] = 1.0 if grad_clip_active_any else 0.0
    out["train/optimizer_step_ran"] = float(optimizer_ran)
    angular = getattr(getattr(unwrap_for_state_dict(model), "PET", None), "angular_conditioning", None)
    if angular is not None:
        angular_weight = angular.projection.weight
        out["train/angular_conditioning/weight_norm"] = float(angular_weight.detach().float().norm())
        out["train/angular_conditioning/trainable"] = float(angular_weight.requires_grad)
        out["train/angular_conditioning/gradient_present"] = float(angular_weight.grad is not None)
        # Observed after the optimizer transaction; gradient may have been clipped.
        out["train/angular_conditioning/grad_norm_post_clip"] = (
            float(angular_weight.grad.detach().float().norm()) if angular_weight.grad is not None else 0.0
        )
    out.update(gradient_transfer_metrics)
    conditioning = getattr(getattr(unwrap_for_state_dict(model), "TruthGeneration", None), "visible_conditioning", None)
    if conditioning is not None and conditioning.log_diagnostics:
        for name, value in conditioning.diagnostics.items():
            out[f"train/visible_conditioning/{name}"] = float(value)
        for group in ("encoder", "modulations", "token_readout"):
            parameters = [p for name, p in conditioning.named_parameters()
                          if (name.startswith(group + ".") if group != "encoder" else
                              not name.startswith(("modulations.", "token_readout.")))]
            grads = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
            out[f"train/visible_conditioning/{group}_grad_norm_post_clip"] = float(torch.stack(grads).sum().sqrt()) if grads else 0.
    if theta_old_snap is not None and bool(
        _dgpo_cfg_get(global_config.dgpo, "log_parameter_update_rms", False)
    ):
        out["train/parameter_update_rms"] = (
            _trainable_direction_rms(theta_old_snap, snapshot_params(model))
            if optimizer_ran else 0.0
        )
    if extragradient_optimizer_base_params is not None:
        out["reference_trust/extragradient/rebased_optimizer_params"] = float(
            extragradient_rebased_params
        )
    out["projection/active"] = 1.0 if projection_active else 0.0
    out["projection/pure_dgpo_backward"] = 1.0
    _append_projection_summary_metrics(out)
    _append_projection_constraint_panel_metrics(out)
    _append_swd_panel_metrics(out)
    out.update(nonfinite_diag)

    # Training-distribution monitoring duplicates validation's truth/pred arrays.
    # Keep the entire path cold when disabled: no feature resolution, tensor-to-CPU
    # copies, NumPy arrays, histograms, or class-index expansion.
    if collect_train_dist:
        cartesian = _truth_generation_cartesian()
        all_plot_feature_names = _generation_monitor_feature_names(
            cartesian=cartesian
        )
    if collect_train_dist and _supports_legacy_invisible_kinematics(
        cartesian=cartesian,
        feature_dim=int(batch["x_invisible"].shape[-1]),
    ):
        k_sel = _kin_hist_candidate_indices_per_event(
            rewards, candidates_phys, batch, cartesian=cartesian
        )
        ppt, peta, pphi, tpt, teta, tphi = _val_pred_truth_kin_flat(
            candidates_phys, batch, k_sel, cartesian=cartesian, device=device
        )
        k1_sel = torch.zeros(B, device=device, dtype=torch.long)
        k1_pt, k1_eta, k1_phi, k1_tpt, k1_teta, k1_tphi = _val_pred_truth_kin_flat(
            candidates_phys, batch, k1_sel, cartesian=cartesian, device=device
        )
        _td_pt_edges = _diagnostic_bin_edges("pt")
        _td_eta_edges = _diagnostic_bin_edges("eta")
        _td_phi_edges = _diagnostic_bin_edges("phi")
        out["_kin_h_pt_p"] = np.histogram(ppt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_pt_t"] = np.histogram(tpt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_e_p"] = np.histogram(peta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_e_t"] = np.histogram(teta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_p_p"] = np.histogram(pphi, bins=_td_phi_edges)[0].astype(np.float64)
        out["_kin_h_p_t"] = np.histogram(tphi, bins=_td_phi_edges)[0].astype(np.float64)
        out["_kin_h_pt_k1_p"] = np.histogram(k1_pt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_pt_k1_t"] = np.histogram(k1_tpt, bins=_td_pt_edges)[0].astype(np.float64)
        out["_kin_h_e_k1_p"] = np.histogram(k1_eta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_e_k1_t"] = np.histogram(k1_teta, bins=_td_eta_edges)[0].astype(np.float64)
        out["_kin_h_p_k1_p"] = np.histogram(k1_phi, bins=_td_phi_edges)[0].astype(np.float64)
        out["_kin_h_p_k1_t"] = np.histogram(k1_tphi, bins=_td_phi_edges)[0].astype(np.float64)
        if cartesian:
            all_px_p, all_py_p, all_pz_p, all_px_t, all_py_t, all_pz_t = (
                _val_pred_truth_cartesian_flat_all_candidates(
                    candidates_phys,
                    batch,
                    device=device,
                    dtype=dtype,
                )
            )
            out["_kin_all_px_p"] = all_px_p
            out["_kin_all_px_t"] = all_px_t
            out["_kin_all_py_p"] = all_py_p
            out["_kin_all_py_t"] = all_py_t
            out["_kin_all_pz_p"] = all_pz_p
            out["_kin_all_pz_t"] = all_pz_t
    if collect_train_dist and not cartesian:
        feature_arrays = _val_pred_truth_feature_flat_all_candidates(
            candidates_phys,
            batch,
            feature_names=all_plot_feature_names,
            device=device,
        )
        for key, values in feature_arrays.items():
            suffix = "p" if key.endswith("_pred") else "t"
            feature_name = key.rsplit("_", 1)[0]
            out[f"_kin_all_{feature_name}_{suffix}"] = values
        out["_kin_all_class_index"] = _val_class_index_flat_all_candidates(
            candidates_phys,
            batch,
            device=device,
        )

    # A hard trust-boundary stop is not an optimizer update and must not consume
    # a scheduler step. Preserve the historical scheduler behavior for all
    # non-boundary paths, including a non-finite skipped batch.
    if optimizer_ran or not trust_boundary_hit:
        optimizer.scheduler_step()
    return out



def _resolve_dgpo_lr_schedule(
    lr_schedule: Mapping[str, Any] | None, *, steps_per_epoch: int,
) -> dict[str, Any]:
    """Resolve a policy-epoch horizon, never a classifier-fit/data-pass clock."""
    cfg = dict(lr_schedule or {})
    kind = str(cfg.get("type", "constant"))
    if kind not in {"constant", "cosine"}:
        raise ValueError("DGPO lr_schedule.type must be constant or cosine")
    resume_use_config = cfg.get("resume_use_config", False)
    if not isinstance(resume_use_config, bool):
        raise ValueError("lr_schedule.resume_use_config must be boolean")
    if resume_use_config and kind != "cosine":
        raise ValueError("lr_schedule.resume_use_config requires cosine")
    epochs = cfg.get("total_epochs")
    if epochs is not None:
        if kind != "cosine":
            raise ValueError("lr_schedule.total_epochs requires cosine")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("lr_schedule.total_epochs must be a positive integer")
        if cfg.get("total_steps") is not None:
            raise ValueError("Set only lr_schedule.total_epochs or total_steps; clear the other with null")
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
            raise ValueError("Epoch-based cosine requires positive integer steps_per_epoch")
        cfg["total_steps"] = epochs * steps_per_epoch
    return cfg


def build_optimizer(
    model: torch.nn.Module,
    *,
    steps_per_epoch: int,
    warmup_steps: int,
    is_rank0: bool = True,
    lr_schedule: Mapping[str, Any] | None = None,
    conditioning_learning_rates: Mapping[str, float] | None = None,
) -> _DgpoOptimizerWithSchedule:
    """AdamW with grouped LR/WD and linear warmup, optionally followed by cosine decay.

    ``warmup_steps`` counts **batches** (one ``scheduler_step`` per call to :func:`train_step`;
    each ``train_step`` performs one accumulated ``optimizer.step()``).

    Parameters
    ----------
    model:
        ``EveNetModel`` or ``DDP(_DGPODDPForward(EveNetModel))``; parameters are taken from the
        unwrapped core for grouping (same tensor objects as ``model.parameters()``).
    steps_per_epoch:
        Logged on rank 0 for traceability (matches the worker's batch count per epoch).
    warmup_steps:
        Batches for linear LR ramp ``min(1, epoch / warmup_steps)`` on groups with ``warm_up: true``.
    is_rank0:
        When True, log one line per optimizer group on construction.
    conditioning_learning_rates:
        Optional LR overrides for the new PET ``angular_conditioning`` and
        TruthGeneration ``visible_conditioning`` branches. Child parameters
        are removed from their parent groups; WD and warmup stay inherited.
    """
    core = _unwrap_core_evenet(model)
    train_opt = global_config.options.Training
    components = train_opt.Components
    default_lr = float(train_opt.learning_rate)
    default_wd = float(train_opt.weight_decay)
    schedule_cfg = _resolve_dgpo_lr_schedule(lr_schedule, steps_per_epoch=steps_per_epoch)
    schedule_kind = str(schedule_cfg.get("type", "constant"))
    if not math.isfinite(default_wd) or default_wd < 0:
        raise ValueError("DGPO weight_decay must be finite and nonnegative")

    group_meta: dict[str, dict[str, Any]] = {}
    group_modules: dict[str, list[str]] = defaultdict(list)

    for comp_key, cfg in components.items():
        if cfg is None:
            continue
        group = cfg.get("optimizer_group", None)
        if not group:
            continue
        module_attr = getattr(core, comp_key, None)
        if module_attr is None:
            continue
        gname = str(group)
        group_modules[gname].append(str(comp_key))
        if gname not in group_meta:
            lr = float(cfg.get("learning_rate", default_lr))
            wd = float(cfg.get("weight_decay", default_wd))
            if not math.isfinite(wd) or wd < 0:
                raise ValueError(f"DGPO group {gname!r} weight_decay must be finite and nonnegative")
            warm_up = bool(cfg.get("warm_up", True))
            opt_type = str(cfg.get("optimizer_type", "AdamW"))
            group_meta[gname] = {
                "lr": lr,
                "weight_decay": wd,
                "warm_up": warm_up,
                "optimizer_type": opt_type,
            }

    branch_paths = {
        "angular_conditioning": "PET.angular_conditioning",
        "visible_conditioning": "TruthGeneration.visible_conditioning",
    }
    branch_rates = {} if conditioning_learning_rates is None else conditioning_learning_rates
    if not isinstance(branch_rates, Mapping) or set(branch_rates) - set(branch_paths):
        raise ValueError("conditioning_learning_rates supports only angular_conditioning and visible_conditioning")
    # Fixed ordering is independent of config serialization and identical on every rank.
    for name, path in branch_paths.items():
        if name not in branch_rates:
            continue
        rate = branch_rates[name]
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0):
            raise ValueError(f"conditioning_learning_rates.{name} must be finite and positive")
        try:
            branch = core.get_submodule(path)
        except AttributeError as exc:
            raise ValueError(f"conditioning_learning_rates.{name} requires enabled {path}") from exc
        if not any(p.requires_grad for p in branch.parameters()):
            raise ValueError(f"conditioning_learning_rates.{name} requires trainable {path}")
        parent_group = str(components.get(path.split('.')[0], {}).get("optimizer_group", ""))
        if parent_group not in group_meta or name in group_meta:
            raise ValueError(f"Cannot assign independent conditioning optimizer group {name!r}")
        group_meta[name] = {**group_meta[parent_group], "lr": float(rate)}
        group_modules[name] = [path]

    bad = [
        (g, m["optimizer_type"])
        for g, m in group_meta.items()
        if str(m["optimizer_type"]).lower() != "adamw"
    ]
    if bad:
        raise ValueError(
            "DGPO build_optimizer only supports AdamW parameter groups; got: "
            + ", ".join(f"{g}={t!r}" for g, t in bad)
        )

    ws = max(1, int(warmup_steps))
    param_groups: list[dict[str, Any]] = []
    lr_lambdas: list[Any] = []
    nonempty_group_order: list[str] = []
    all_module_paths = [path for paths in group_modules.values() for path in paths]

    for gname, meta in group_meta.items():
        params = [p for p in module_optimizer_parameters(
            core, group_modules[gname], all_module_paths
        ) if p.requires_grad]
        if not params:
            continue
        nonempty_group_order.append(gname)
        param_groups.append(
            {
                "params": params,
                "lr": float(meta["lr"]),
                "weight_decay": float(meta["weight_decay"]),
                "group_name": gname,
            }
        )
        if meta["warm_up"]:
            lr_lambdas.append(lambda epoch, _ws=ws: min(1.0, float(epoch) / float(_ws)))
        else:
            lr_lambdas.append(lambda _epoch: 1.0)

    if not param_groups:
        raise ValueError(
            "[DGPO] build_optimizer: no trainable parameters matched Components with "
            "optimizer_group (check include/freeze settings)."
        )

    # Fallback group for any trainable parameter not covered by an optimizer_group
    # (preserves the original permissive behavior of optimizing everything trainable).
    assigned = {id(p) for pg in param_groups for p in pg["params"]}
    leftover = [
        p for p in core.parameters() if p.requires_grad and id(p) not in assigned
    ]
    if leftover:
        if is_rank0:
            n_leftover = sum(p.numel() for p in leftover)
            _log.warning(
                "[DGPO] %s trainable parameters (%s elements) are not covered by any "
                "Components.<X>.optimizer_group; adding to a fallback group with "
                "lr=%s wd=%s warm_up=true.",
                len(leftover),
                n_leftover,
                default_lr,
                default_wd,
            )
        nonempty_group_order.append("__fallback__")
        param_groups.append(
            {
                "params": leftover,
                "lr": default_lr,
                "weight_decay": default_wd,
                "group_name": "__fallback__",
            }
        )
        lr_lambdas.append(lambda epoch, _ws=ws: min(1.0, float(epoch) / float(_ws)))

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    if is_rank0:
        _log.info(
            "[DGPO] Optimizer: AdamW groups=%s steps/epoch≈%s warmup_batches=%s (linear→%s).",
            len(param_groups),
            int(steps_per_epoch),
            ws,
            schedule_kind,
        )
        for i, gname in enumerate(nonempty_group_order):
            pg = optimizer.param_groups[i]
            npar = sum(p.numel() for p in pg["params"])
            if gname in group_meta:
                mods = ", ".join(group_modules[gname])
                warm = group_meta[gname]["warm_up"]
            else:
                mods = "<fallback>"
                warm = True
            _log.info(
                "[DGPO]   group %r modules=[%s] params=%s lr=%s wd=%s warm_up=%s",
                gname,
                mods,
                npar,
                pg["lr"],
                pg["weight_decay"],
                warm,
            )

    wrapped = _DgpoOptimizerWithSchedule(
        optimizer, scheduler,
        cosine_config=schedule_cfg if schedule_kind == "cosine" else None,
        warmup_steps=ws,
        warmup_groups=[bool(group_meta[g]["warm_up"]) if g in group_meta else True
                       for g in nonempty_group_order],
    )
    if is_rank0 and wrapped.cosine_state is not None:
        _log.info("[DGPO] Cosine LR enabled: policy_epochs=%s steps/logical_epoch=%s end_scheduler_step=%s min_lr_ratio=%s groups=%s; refits/resume preserve its clock.",
                  schedule_cfg.get("total_epochs"), steps_per_epoch,
                  wrapped.cosine_state["total_steps"], wrapped.cosine_state["min_lr_ratio"],
                  [name for name, enabled in zip(nonempty_group_order, wrapped.cosine_state["decay_groups"], strict=True) if enabled])
    return wrapped


def _dgpo_wandb_metric_definition_map() -> dict[str, str]:
    """Explicit definitions for W&B Config → dgpo_metric_definitions (visible in the UI)."""
    return {
        "epoch": "Training epoch index (x-axis for most plots).",
        "train/lr/scheduled_max": "Largest DGPO group LR before this update, before round warmup and trust backtracking. Cosine resumes its saved scheduler clock.",
        "train/lr/scheduled_min": "Smallest DGPO group LR before this update, before round warmup and trust backtracking.",
        "train/lr/scheduled/*": "Named optimizer-group LR before this update; before round warmup and trust backtracking. New conditioning groups exclude their parameters from the pretrained parent groups.",
        "staleness/global_best/improved": "1 when a global-best policy is registered/replaced; candidate replacements require the configured effect-size and paired confirmation gate.",
        "staleness/global_best/failed_rounds": "Consecutive completed plateau windows after refitting the same incumbent best; a confirmed new best resets this counter.",
        "staleness/global_best/stop_requested": "1 when repeated global-best refits fail to improve; save the consistent latest state and stop, preserving the separate best checkpoint. Not convergence.",
        "staleness/global_best/confirmation_delta": "Candidate minus incumbent raw |AUC-.5| in paired equal-start confirmation fits; negative favors candidate. Not a significance statistic.",
        "staleness/global_best/confirmation_valid": "1 only when both confirmation fits are saturated and finite.",
        "staleness/global_best/confirmation_accepted": "1 when paired confirmation favors candidate beyond raw_improvement_min_delta.",
        # --- reward/dist (overlaid figure, every log_reward_dist_every steps) ---
        "reward/dist/overlap": "Matplotlib figure: three overlapped 1D density histograms (best / worst / median per valid event). wandb.Image — use the media step slider to compare across training steps.",
        # --- reward/monitor (scalars, every step) ---
        "reward/monitor/best_of_k": "Mean reward of the argmax (best) candidate per valid event.",
        "reward/monitor/median": "Mean over events of the median reward along K.",
        "reward/monitor/mean_gap": "Mean over events: (mean reward strictly above per-event median) − (mean reward strictly below median).",
        "reward/monitor/last_place": "Mean reward of the worst (min) candidate per valid event.",
        "reward/monitor/p10": "Mean over events of the 10th percentile of rewards along K.",
        "reward/monitor/p30": "Mean over events of the 30th percentile of rewards along K.",
        "reward/monitor/p70": "Mean over events of the 70th percentile of rewards along K.",
        "reward/monitor/p90": "Mean over events of the 90th percentile of rewards along K.",
        "reward/monitor/advantage_pos_neg_gap": "Mean reward where advantage > 0 minus mean where advantage < 0 (valid slots).",
        "reward/sources/*/mean": "Raw mean reward for each additive reward source over all valid rollout candidates.",
        "reward/sources/*/weighted_mean": "Configured reward weight times reward/sources/*/mean; these add up to reward/raw/mean.",
        "reward/sources/*/selected_by_total_mean": "Per-source raw reward after selecting the candidate with highest combined total reward per valid event.",
        "reward/sources/*/selected_by_total_weighted_mean": "Configured reward weight times selected_by_total_mean.",
        "reward/sources/*/source_best_of_k": "For each source alone, mean over events of max_k source_reward[k,event].",
        "reward/sources/*/source_last_place": "For each source alone, mean over events of min_k source_reward[k,event].",
        "reward/sources/*/selection_gap": "selected_by_total_mean - mean. Large shifts show how combined reward selection biases that source.",
        # --- dgpo (train scalars) ---
        "projection/active": "1 when projection_constraint repair runs after AdamW on this step.",
        "projection/pure_dgpo_backward": "1 when the main policy objective stays DGPO-style in backward. Optional soft regularizers may be added, while any latent-SWD CPO repair still happens only post-AdamW.",
        # --- train/loss ---
        "train/loss/total": "Scalar passed to backward(): DGPO main term plus any enabled supervised diffusion anchor and soft regularizers. Post-step latent-SWD CPO repair is after AdamW only.",
        "train/round_warmup/lr_scale": "DGPO-only multiplier relative to scheduled group LRs after each accepted reward install; applied together with trust scaling/backtracking. Not classifier LR.",
        "train/round_warmup/completed_updates": "Accepted DGPO optimizer updates completed before this step in the installed round's warmup. Rejected/nonfinite updates do not advance it; resumes preserve it.",
        "train/loss/dgpo": "DGPO main term (detached gate × advantage × L_cur). Lower is better.",
        "train/loss/L_cur": "Mean velocity MSE for the trainable policy (DDIM target). Lower is better.",
        "train/loss/L_ref": "Mean velocity MSE for the frozen reference policy. Lower is better.",
        "train/loss/delta": "mean(|L_cur - L_ref|): average absolute gap between current and reference velocity MSE. Shows how far the policy has moved from frozen ref_model (not rollout EMA).",
        "train/loss/velocity": "Detached velocity objective slice: pure DGPO main term used for backward.",
        "train/loss/kl": "Weighted supervised diffusion anchor beta_kl * mean_row |v_pred - v_truth|^2 on the same noisy inputs. Keeps the original diffusion preference toward the denoising target while DGPO adds physics steering.",
        "reference_trust/loss": "Configured round-reference trust objective. It is legacy shared-noise velocity MSE for objective=velocity_mse and a full-time cosine-VP reverse-path KL estimate for objective=vp_path_kl.",
        "reference_trust/vp_path_kl": "Separate full-time importance-sampled cosine-VP reverse-path KL estimate: 0.5*Z times the mean active-dimension sum of squared current/reference velocity differences.",
        "reference_trust/objective_vp_path_kl": "1 when VP path-KL, rather than legacy dimension-mean velocity MSE, is used in backward.",
        "reference_trust/round_decay/step": "Installed-reference schedule index since enabling round_decay: delta=max(floor, initial*factor**step). Unchanged by monitors, failed refits, policy-only rollback, and resume.",
        "reference_trust/sequential_backward": "1 when the DGPO and VP path-KL graphs are constructed and backpropagated sequentially to reduce activation-memory peak; their gradients still sum before the one AdamW update.",
        "reference_trust/vp_path_kl_diagnostic_enabled": "1 when a separate no-gradient VP path-KL forward is logged while the legacy velocity-MSE objective remains active.",
        "reference_trust/velocity_mse_ratio": "Round-reference trust velocity MSE divided by the frozen reference policy's own velocity loss.",
        "reference_trust/coefficient": "Configured multiplier for the paired round-reference trust loss.",
        "reference_trust/control_ratio_global": "Accepted post-step fixed-probe functional distance enforced by strict backtracking: velocity-MSE ratio or VP path-KL according to adaptive_boundary.distance.",
        "reference_trust/delta": "Effective hard radius: fixed, AUC-scaled, per-reference round_decay, or feasibility-protected new-raw-best best_decay, according to radius_mode.",
        "reference_trust/best_decay/count": "Number of saturated GLOBAL-best raw-AUC improvements, excluding first baseline registration. Preserved across policy rollback and reference refits.",
        "reference_trust/best_decay/global_best_auc_gap": "Lowest saturated raw abs(AUC-0.5) observed in this experiment; never reset by reward refits or policy-only rollback.",
        "reference_trust/best_decay/target": "Requested radius max(floor, initial*best_decay_factor**count). The effective delta may be larger to keep the current policy feasible.",
        "reference_trust/best_decay/feasibility_limited": "1 when shrinking fully to the target would consume the fixed-probe safety headroom; deferred contraction is applied after recentering.",
        "reference_trust/warning_ratio": "Pre-step distance at which the candidate AdamW update begins with a linearly damped LR.",
        "reference_trust/interior_fraction": "Configured fraction of delta targeted by strict backtracking; values below one preserve room for later updates.",
        "reference_trust/nominal_interior_limit": "interior_fraction times the active hard trust radius.",
        "reference_trust/acceptance_limit": "Effective per-step backtracking limit. Normally the nominal interior limit; legacy resumes already outside that target may hold but not increase their distance.",
        "reference_trust/fixed_probe_per_reward_round": "1 when event conditions, diffusion timesteps, noise, and round-reference outputs are fixed until the reward/reference pair changes.",
        "reference_trust/probe_reused": "1 after the first optimizer attempt in a reward round, when the same fixed probe is reused.",
        "reference_trust/pre_step_distance": "Fixed shared-noise functional distance before the candidate AdamW update.",
        "reference_trust/candidate_distance": "Fixed-probe distance after the initial candidate AdamW update and before any backtracking.",
        "reference_trust/candidate_distance_over_delta": "Candidate fixed-probe distance divided by the active radius; values above one are trust-boundary exceedances.",
        "reference_trust/candidate_excess": "Candidate fixed-probe distance minus the active radius; positive values exceed the trust region.",
        "reference_trust/post_step_distance": "Fixed-probe distance of the accepted policy; strict mode requires it to remain at or below the active acceptance_limit.",
        "reference_trust/post_step_distance_over_delta": "Accepted fixed-probe distance divided by the active radius; strict mode requires this to be at most one.",
        "reference_trust/update_scale": "Accepted absolute multiplier of the scheduled AdamW parameter displacement; alias of accepted_step_scale.",
        "reference_trust/accepted_step_scale": "Final AdamW displacement scale after warning-band damping and strict backtracking.",
        "reference_trust/backtrack_steps": "Number of multiplicative backtracking trials needed to satisfy the current trust radius.",
        "reference_trust/boundary_hit": "1 when the initial candidate exceeded delta or the pre-step probe was already outside; this no longer forces an OmniFold refit.",
        "reference_trust/preexisting_violation": "1 when the fixed pre-step probe was already outside delta, so no optimizer update was attempted.",
        "reference_trust/interior_saturated": "1 when the fixed probe remains inside hard delta but has exhausted the configured interior update budget; the policy is held until a later audit decision.",
        "reference_trust/step_accepted": "1 only when a feasible AdamW update was committed and the scheduler/EMA were advanced.",
        "reference_trust/optimizer_state_advanced": "1 when the committed proposal advanced AdamW moments and step counters. With transactional_rejection=true this returns to 0 after an alpha=0 rejection.",
        "reference_trust/optimizer_state_restored": "1 when an alpha=0 trust rejection restored the exact pre-proposal AdamW moments and step counters.",
        "reference_trust/accepted_updates": "Cumulative committed optimizer updates in the current hard-boundary experiment; this differs from global_step when a proposal is rejected.",
        "reference_trust/rejection_stop_requested": "1 after stop_after_rejection observes its first alpha=0 proposal and schedules the endpoint cold audit.",
        "reference_trust/first_rejected_global_step": "One-indexed proposal clock of the first alpha=0 hard-boundary rejection, or -1 before rejection.",
        "reference_trust/zero_step_adam_first_moments_reset": "Number of Adam first-moment tensors cleared after a trust-boundary alpha=0 rejection; second moments and optimizer step counters remain intact.",
        "reference_trust/policy_lr/scale": "Per-round DGPO LR multiplier sqrt(delta_r/delta_0), clipped by the configured floor; classifier LRs are unaffected.",
        "reference_trust/policy_lr/effective_step_scale": "Product of the per-round policy LR scale and the within-round trust/backtracking scale.",
        "reference_trust/cross_round/cap_active": "1 when a newly installed reward round was prevented from using a radius larger than its predecessor.",
        "reference_trust/probe_velocity_mse": "Fixed-probe masked velocity MSE between policy and paired round reference.",
        "reference_trust/probe_reference_loss": "Round-reference denoising loss used to normalize the fixed-probe velocity MSE.",
        "reference_trust/raw_auc": "Unweighted truth-vs-current-generation AUC from the first cross-fit classifier of the installed OmniFold round.",
        "reference_trust/statistically_closed": "1 when the installed round raw-AUC gap is within the configured closure/statistical-uncertainty band.",
        "reference_trust/adam_first_moments_reset": "Number of AdamW exp_avg buffers zeroed after an accepted reward/reference replacement.",
        "reference_trust/empirical/acceptance_rate": "Fraction of recent optimizer attempts that committed a feasible update on the fixed round probe.",
        "reference_trust/empirical/mean_update_scale": "Mean final AdamW displacement scale across the recent attempt window; rejected attempts contribute zero.",
        "reference_trust/empirical/throughput_starved": "1 when recent acceptance rate or mean update scale is below its configured target.",
        "reference_trust/empirical/radius_action": "Adaptive v2 audit action: -1 contracts the future-round cap, 0 holds, +1 expands the live radius.",
        "reference_trust/empirical/delta_before": "Active trust radius immediately before an adaptive audit decision.",
        "reference_trust/empirical/delta_after": "Active trust radius immediately after an adaptive audit decision.",
        "train/loss/variance_regularization": "Weighted anti-shrink soft penalty added to backward: lambda_var * mean(relu((std_truth - std_pred) / std_truth)^2) over the selected event_info-driven angular features.",
        "train/regularization/variance/active": "1 when batch-level variance anti-shrink regularization found at least one selected feature in the current feature layout.",
        "train/regularization/variance/active_features": "How many selected features contributed to the anti-shrink regularizer on this batch.",
        "train/regularization/variance/raw": "Unweighted anti-shrink penalty before multiplying by dgpo.variance_regularization.weight.",
        "train/regularization/variance/*/std_truth": "Per-feature truth std over valid batch slots for the anti-shrink monitor.",
        "train/regularization/variance/*/std_pred": "Per-feature prediction std over valid batch slots for the anti-shrink monitor.",
        "train/regularization/variance/*/std_delta_ratio": "Signed relative std shift (std_pred - std_truth) / std_truth. Negative means the prediction is narrower than truth; positive means wider.",
        "train/regularization/variance/*/std_gap": "Per-feature positive relative shrinkage max((std_truth - std_pred) / std_truth, 0). Non-zero means the prediction is narrower than truth.",
        "train/regularization/variance/*/penalty": "Per-feature raw anti-shrink penalty relu((std_truth - std_pred) / std_truth)^2.",
        # --- projection (W&B panel: five CPO repair scalars only) ---
        "projection/v_linear": "Linear post-Adam violation estimate: C_adam_pred - epsilon. Drives lambda when positive.",
        "projection/C_adam_pred": "First-order Taylor prediction C_old + b^T delta0 at theta_adam (linear constraint after AdamW step).",
        "projection/lambda": "Closed-form CPO multiplier lambda_star = [v / (b^T p + damping)]_+ applied to the repair direction.",
        "projection/final_update_norm": "L2 norm ||theta_final - theta_old|| after projection (includes final-update cap when active).",
        "projection/summary/C_projected_minus_old": "C_projected - C_old on frozen (t, eps); negative means projection reduced the constraint vs pre-step weights.",
        "projection/multi_sample/C_mean": "Mean normalized constraint C_norm over multi-sample draws at theta_old; per-batch trace for sawtooth / oscillation diagnostics.",
        # --- swd (W&B panel: frozen latent-SWD constraint monitoring) ---
        "swd/active": "1 when latent-SWD constraint diagnostics were logged this step; 0 when the batch was skipped (too few valid rows).",
        "swd/pred_truth": "Sliced Wasserstein distance SWD(z_pred, z_truth) in the frozen encoder latent space.",
        "swd/truth_truth": "Null-floor SWD: random truth/truth split SWD_tt within the batch.",
        "swd/ratio": "swd_pred_truth / (swd_truth_truth + eps); raw ratio before null-excess normalization.",
        "swd/C_norm": "Ratio-normalized constraint (swd_pred_truth - swd_truth_truth) / (swd_truth_truth + eps); drives CPO when > margin.",
        "swd/mask_count": "Number of valid (event, candidate) rows encoded for SWD this step.",
        "swd/skipped_small_mask": "1 when mask_count < latent_swd.min_samples and the constraint was skipped.",
        "projection/reward_constraint_align": "cos(b, delta0) = b.delta0/(|b||delta0|) for the CPO repair. >0 => the reward step keeps pushing the constraint UP (genuine reward<->constraint tension); ~0 => orthogonal, so a reward stall is NOT caused by the constraint.",
        "projection/constraint/swd_pred_truth": "Alias of swd/pred_truth under projection/constraint/* (legacy projection panel).",
        "projection/constraint/swd_truth_truth": "Alias of swd/truth_truth under projection/constraint/*.",
        "projection/constraint/swd_ratio": "Alias of swd/ratio under projection/constraint/*.",
        "projection/constraint/C_norm": "Alias of swd/C_norm under projection/constraint/*.",
        "diagnostics/reward_hacking/all/px/reward_mean": "Mean px reward contribution (ν1+ν2, negative normalized squared error) over all valid rollout candidates.",
        "diagnostics/reward_hacking/all/py/reward_mean": "Mean py reward contribution over all valid rollout candidates.",
        "diagnostics/reward_hacking/all/pz/reward_mean": "Mean pz reward contribution over all valid rollout candidates.",
        "diagnostics/reward_hacking/best/px/reward_mean": "Mean px reward contribution after selecting the combined-reward argmax candidate per valid event.",
        "diagnostics/reward_hacking/best/py/reward_mean": "Mean py reward contribution on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/reward_mean": "Mean pz reward contribution on reward-best candidates.",
        "diagnostics/reward_hacking/all/px/delta_mean": "Signed mean px residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/py/delta_mean": "Signed mean py residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pz/delta_mean": "Signed mean pz residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/px/delta_abs_mean": "Mean absolute px residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/py/delta_abs_mean": "Mean absolute py residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pz/delta_abs_mean": "Mean absolute pz residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/px/delta_mean": "Signed mean px residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/py/delta_mean": "Signed mean py residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/delta_mean": "Signed mean pz residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/px/delta_abs_mean": "Mean absolute px residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/py/delta_abs_mean": "Mean absolute py residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pz/delta_abs_mean": "Mean absolute pz residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/all/pt/delta_mean": "Signed mean pT residual pred−truth in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/pt/delta_abs_mean": "Mean absolute pT residual in GeV over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/pt/delta_mean": "Signed mean pT residual pred−truth in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/best/pt/delta_abs_mean": "Mean absolute pT residual in GeV on reward-best candidates.",
        "diagnostics/reward_hacking/all/eta/delta_mean": "Signed mean η residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/eta/delta_abs_mean": "Mean absolute η residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/eta/delta_mean": "Signed mean η residual on reward-best candidates.",
        "diagnostics/reward_hacking/best/eta/delta_abs_mean": "Mean absolute η residual on reward-best candidates.",
        "diagnostics/reward_hacking/all/phi/delta_mean": "Signed mean wrapped φ residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/all/phi/delta_abs_mean": "Mean absolute wrapped φ residual over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/phi/delta_mean": "Signed mean wrapped φ residual on reward-best candidates.",
        "diagnostics/reward_hacking/best/phi/delta_abs_mean": "Mean absolute wrapped φ residual on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/cos_opening": "Mean cos(opening angle) between the two reconstructed tau directions over all valid rollout candidates. Ideal back-to-back topology is near -1.",
        "diagnostics/ztautau_back_to_back/best/cos_opening": "Same cos(opening angle) metric after selecting the combined-reward argmax candidate per valid event.",
        "diagnostics/ztautau_back_to_back/all/delta_phi_to_pi": "Mean ||Delta phi| - pi| over all valid rollout candidates. Smaller is more back-to-back in azimuth.",
        "diagnostics/ztautau_back_to_back/best/delta_phi_to_pi": "Same azimuthal back-to-back metric on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/back_to_back_loss": "Mean (cos_opening + 1)^2 + (|Delta phi| - pi)^2 over all valid rollout candidates.",
        "diagnostics/ztautau_back_to_back/best/back_to_back_loss": "Same combined back-to-back loss on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/calibration_deltaR_a": "Mean post-calibration direction change DeltaR for tau-a over all valid rollout candidates. Smaller is more physics-consistent.",
        "diagnostics/ztautau_back_to_back/best/calibration_deltaR_a": "Same tau-a post-calibration DeltaR on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/calibration_deltaR_b": "Mean post-calibration direction change DeltaR for tau-b over all valid rollout candidates.",
        "diagnostics/ztautau_back_to_back/best/calibration_deltaR_b": "Same tau-b post-calibration DeltaR on reward-best candidates.",
        "diagnostics/ztautau_back_to_back/all/calibration_deltaR_sum": "Mean calibration magnitude DeltaR_a + DeltaR_b over all valid rollout candidates. This is the physics-consistency reward when reward_config.type=calibration_magnitude.",
        "diagnostics/ztautau_back_to_back/best/calibration_deltaR_sum": "Same calibration magnitude on reward-best candidates.",
        "diagnostics/reward_hacking/all/rel_pt/mean": "Mean pT_pred / pT_truth - 1 over all valid rollout candidates and both ν slots. Negative values indicate pT shrink.",
        "diagnostics/reward_hacking/all/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) over all valid rollout candidates and both ν slots.",
        "diagnostics/reward_hacking/best/rel_pt/mean": "Mean pT_pred / pT_truth - 1 after selecting the combined-reward argmax candidate per valid event. Compare to all/rel_pt/mean to spot reward-driven pT shrink.",
        "diagnostics/reward_hacking/best/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) on reward-best candidates.",
        "diagnostics/reward_hacking/dist/rel_pt": "Matplotlib density overlay of pT_pred / pT_truth - 1 for all rollout candidates vs reward-best candidates (wandb.Image).",
        "diagnostics/reward_hacking/pt_oracle/pt/delta_mean": "Signed mean pT residual pred−truth in GeV after selecting, per event, the candidate with smallest |ΔpT_nu1| + |ΔpT_nu2|. This is a truth oracle for candidate-support diagnosis, not a deployable selector.",
        "diagnostics/reward_hacking/pt_oracle/pt/delta_abs_mean": "Mean absolute pT residual |pred−truth| in GeV for the pT-oracle-best candidate.",
        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt": "Profile plot by truth-pT bin: top panel compares mean delta pT = pT_pred - pT_truth for all rollout candidates, reward-best candidates, and pT-oracle-best candidates; bottom panel shows reward-best minus all and pT-oracle minus all. Gray bars show truth event-slot counts per bin, not K-times all-candidate counts. If pT-oracle fixes high-pT bins, support exists and ranking/reward is the bottleneck; if pT-oracle remains low, generator support is insufficient.",
        "diagnostics/reward_hacking/profile/eta_delta_vs_truth_eta": "Profile plot by truth-eta bin, with the same all/reward-best/eta-oracle comparison used by the pT residual profile.",
        "diagnostics/reward_hacking/profile/phi_delta_vs_truth_phi": "Profile plot by truth-phi bin using wrapped phi residuals, with the same all/reward-best/phi-oracle comparison used by the pT residual profile.",
        "diagnostics/reward_hacking/profile/px_delta_vs_truth_px": "Profile plot by truth-px bin, comparing mean px residual for all rollout, reward-best, and px-oracle candidates.",
        "diagnostics/reward_hacking/profile/py_delta_vs_truth_py": "Profile plot by truth-py bin, comparing mean py residual for all rollout, reward-best, and py-oracle candidates.",
        "diagnostics/reward_hacking/profile/pz_delta_vs_truth_pz": "Profile plot by truth-pz bin, comparing mean pz residual for all rollout, reward-best, and pz-oracle candidates.",
        "diagnostics/reward_hacking/profile/pt_delta_first10_vs_fullK": "Profile plot by truth-pT bin comparing first-10-candidate selection against full-K selection on the same rollout pool. Curves show reward-best first 10, reward-best full K, pT-oracle first 10, and pT-oracle full K; bottom panel shows full-K minus first-10 gains.",
        "diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt_accumulated": "Same as diagnostics/reward_hacking/profile/pt_delta_vs_truth_pt, but concatenates raw diagnostic samples over dgpo.diagnostic_profile_accumulate_steps train batches before plotting. Use this for more stable high-pT tail statistics.",
        "diagnostics/reward_hacking/profile/eta_delta_vs_truth_eta_accumulated": "Accumulated truth-bin eta residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/phi_delta_vs_truth_phi_accumulated": "Accumulated truth-bin phi residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/px_delta_vs_truth_px_accumulated": "Accumulated truth-bin px residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/py_delta_vs_truth_py_accumulated": "Accumulated truth-bin py residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reward_hacking/profile/pz_delta_vs_truth_pz_accumulated": "Accumulated truth-bin pz residual profile over dgpo.diagnostic_profile_accumulate_steps train batches.",
        "diagnostics/reference_bias/all/pt/delta_mean": "Signed mean pT residual pred−truth in GeV over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/pt/delta_abs_mean": "Mean absolute pT residual |pred−truth| in GeV over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/eta/delta_mean": "Signed mean η residual pred−truth over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/eta/delta_abs_mean": "Mean absolute η residual |pred−truth| over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/phi/delta_mean": "Signed mean wrapped φ residual pred−truth over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/phi/delta_abs_mean": "Mean absolute wrapped φ residual |pred−truth| over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/all/rel_pt/mean": "Mean pT_pred / pT_truth - 1 over all rollout candidates and the first two ν slots. Negative values indicate pT shrink. Freeze rollout updates to isolate initial/reference bias.",
        "diagnostics/reference_bias/all/rel_pt/abs_mean": "Mean abs(pT_pred / pT_truth - 1) over all rollout candidates and the first two ν slots.",
        "diagnostics/reference_bias/dist/rel_pt": "Matplotlib density plot of pT_pred / pT_truth - 1 for rollout candidates (wandb.Image). With frozen rollout EMA, this is the initial/reference model's pT bias distribution.",
        "diagnostics/reference_bias/profile/pt_delta_vs_truth_pt": "Profile plot: x-axis truth pT bin [GeV], y-axis mean delta pT = pT_pred - pT_truth [GeV] over all rollout candidates and the first two ν slots. Negative high-pT bins indicate tail shrink / dynamic-range compression.",
        # --- train/grad (one accumulated optimizer step per batch) ---
        "train/grad/global_norm_pre_clip": "Total L2 norm of trainable gradients before clip_grad_norm_ (max over sub-steps in the batch). Compare to dgpo.grad_clip_norm in run config.",
        "train/grad/clip_active": "1.0 if any sub-step had pre-clip norm > dgpo.grad_clip_norm (clipping applied); else 0.0.",
        "train/gradient_sync/manual_parameter_tensors": "Number of globally active trainable parameter tensors averaged across workers after sequential DGPO plus VP-trust gradient accumulation. Zero means the ordinary DDP reducer path was used.",
        "train/candidate_nonfinite_fraction": "Fraction of generated candidate tensor elements that were non-finite before zeroing (DDIM / kinematics blow-up).",
        "train/reward_nonfinite_fraction": "Fraction of (K,B) total rewards that were non-finite before zeroing.",
        "train/reward_nonfinite_fraction/*": "Per reward-source non-finite fraction before zeroing.",
        # --- parameter (scalars, every step; extend with more keys later) ---
        "parameter/w_e/mean": "Mean per-event gate w_e = sigmoid(M_e) in [0,1].",
        "parameter/w_e/std": "Std of w_e across events in the batch (population std).",
        "parameter/w_e/min": "Min w_e in the batch.",
        "parameter/w_e/max": "Max w_e in the batch.",
        # --- val (epoch-end) ---
        "val/reward/mean": "Legacy mean best-of-K reward per valid event. Used for top-k checkpoint selection; NOT all-sample mean.",
        "val/reward/all_sample_mean": "Mean reward over ALL valid candidate × event pairs, without selecting the best candidate.",
        "val/reward/best_of_k_mean": "Explicit alias of legacy val/reward/mean (best-of-K per event).",
        "val/reward/median": "Global median of per-event reward across valid events (with val_K=1: single prediction per event; with val_K>1: best-of-K per event). Epoch x-axis.",
        "val/reward/p10": "10th percentile of per-event reward (val_K=1: single pred; val_K>1: best-of-K per event).",
        "val/reward/p30": "30th percentile.",
        "val/reward/p70": "70th percentile.",
        "val/reward/p90": "90th percentile.",
        "val/winrate": "Fraction of valid events where the current policy's reward-best validation candidate beats the reference-policy sample on combined reward. NaN if validation_compute_winrate=false.",
        "val_diagnostics/profile/pt_delta_vs_truth_pt": "Validation profile plot by truth-pT bin: selected-candidate mean delta pT = pT_pred - pT_truth, with the initial pre-DGPO validation profile overlaid after the baseline pass.",
        "val_diagnostics/profile/eta_delta_vs_truth_eta": "Validation profile plot by truth-eta bin: selected-candidate mean eta residual, with the initial pre-DGPO validation profile overlaid after the baseline pass.",
        "val_diagnostics/profile/pt/delta_mean": "Global validation mean pT residual, pT_pred - pT_truth, over selected candidates and valid neutrino slots.",
        "val_diagnostics/profile/pt/slope": "Linear fit slope of the validation binned mean delta-pT profile versus truth pT.",
        "val_diagnostics/profile/pt/zero_delta_truth": "Truth pT value where the fitted validation mean delta-pT profile crosses zero.",
        "val_diagnostics/profile/eta/delta_mean": "Global validation mean eta residual over selected candidates and valid neutrino slots.",
        "val_diagnostics/profile/eta/slope": "Linear fit slope of the validation binned mean delta-eta profile versus truth eta.",
        "val_diagnostics/profile/eta/zero_delta_truth": "Truth eta value where the fitted validation mean delta-eta profile crosses zero.",
        "val_diagnostics/profile/pt_delta_mean_vs_epoch": "History plot with x-axis epoch and y-axis global validation mean pT residual. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/eta_delta_mean_vs_epoch": "History plot with x-axis epoch and y-axis global validation mean eta residual. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/pt_slope_vs_epoch": "History plot with x-axis epoch and y-axis fitted validation pT-profile slope. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/eta_slope_vs_epoch": "History plot with x-axis epoch and y-axis fitted validation eta-profile slope. The pre-DGPO baseline is logged at epoch -1.",
        "val_diagnostics/profile/pt_zero_delta_truth_vs_epoch": "History plot with x-axis epoch and y-axis fitted truth-pT zero-crossing where mean delta pT is zero.",
        "val_diagnostics/profile/eta_zero_delta_truth_vs_epoch": "History plot with x-axis epoch and y-axis fitted truth-eta zero-crossing where mean delta eta is zero.",
        "val_diagnostics/profile/pt_zero_delta_vs_slope": "History plot with x-axis fitted pT-profile slope and y-axis fitted truth-pT zero-crossing.",
        "val_diagnostics/profile/eta_zero_delta_vs_slope": "History plot with x-axis fitted eta-profile slope and y-axis fitted truth-eta zero-crossing.",
        "val/response/reward_initial_vs_current": "2D validation response matrix with x-axis initial pre-DGPO event reward and y-axis current event reward, logged under one W&B image key each epoch so the Images panel has an epoch slider.",
        "val/response/pt_delta_mean_initial_vs_current": "2D validation response matrix with x-axis initial pre-DGPO event mean delta pT and y-axis current event mean delta pT, logged under one W&B image key each epoch so the Images panel has an epoch slider.",
        "val_neutrino/pt": "1D density overlay: truth vs current-policy vs frozen-reference prediction for pT [GeV] (original scale, expm1 of log1p(pT)) (wandb.Image); x-axis **epoch**. Current-policy histogram uses the same per-event candidate index rule as train_dist/* (combined-reward argmax).",
        "val_neutrino/eta": "Same three-way overlay for η; same candidate selection as val_neutrino/pt.",
        "val_neutrino/phi": "Same three-way overlay for φ [rad]; same candidate selection as val_neutrino/pt.",
        "val_neutrino/all/pt_truth_vs_pred": "Example 2D truth-vs-pred key. Actual validation 2D keys follow event_info.invisible_feature_names as val_neutrino/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "val_neutrino/all/eta_truth_vs_pred": "Example 2D truth-vs-pred key. Actual validation 2D keys follow event_info.invisible_feature_names as val_neutrino/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "val_neutrino/all/phi_truth_vs_pred": "Example 2D truth-vs-pred key. Actual validation 2D keys follow event_info.invisible_feature_names as val_neutrino/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "val_neutrino/all_metrics/*/*": "Scalar summaries for validation truth-vs-pred 2D monitors over all candidates: count, mae, rmse, bias=mean(pred-truth), pearson_r, slope, intercept.",
        "val_neutrino/px": "Same three-way overlay for neutrino p_x [GeV]; truth is denormalized invisible target, pred/ref from DDIM output.",
        "val_neutrino/py": "Same three-way overlay for neutrino p_y [GeV].",
        "val_neutrino/pz": "Same three-way overlay for neutrino p_z [GeV].",
        "val_neutrino/jsd/current/*": "Histogram Jensen-Shannon distance between truth and current-policy validation distributions for the named kinematic. Feature names follow event_info.yaml invisible_feature_names (or px/py/pz in cartesian mode). Lower is better.",
        "val_neutrino/jsd/ref/*": "Histogram Jensen-Shannon distance between truth and frozen-reference validation distributions for the named kinematic. Feature names follow event_info.yaml invisible_feature_names (or px/py/pz in cartesian mode). Lower is better.",
        "val_mass/w_mass": "W-boson mass reconstruction (assigned lepton + neutrino) vs truth-neutrino resonance mass, truth vs current policy vs frozen reference (wandb.Image); x-axis **epoch**.",
        "val_mass/top_mass": "Top mass reconstruction (assigned b + W) vs truth-neutrino resonance mass; same three-way overlay as val_mass/w_mass.",
        "val_mass/jsd/current/*": "Histogram Jensen-Shannon distance between truth and current-policy validation mass distributions. Lower is better.",
        "val_mass/jsd/ref/*": "Histogram Jensen-Shannon distance between truth and frozen-reference validation mass distributions. Lower is better.",
        "val_tarp/tarp_binned_min_holm_pvalue": "Family-wise Holm-adjusted TARP decision p-value over visible-acoplanarity bins and the configured joint arms. TARP uses all validation_K candidates; higher is better and values below alpha indicate rejected calibration.",
        "val_tarp/coverage": "Binned TARP coverage curves for the full four-dimensional tau-direction target and rank-copula arms (wandb.Image).",
        "val_tarp/pooled_coverage": "Pooled TARP coverage shown for orientation only; conditional decisions must use the binned Holm-adjusted value.",
        "staleness/judge_auc_weighted": "Held-out AUC of the single freshly reset classifier trained after applying the installed OmniFold weights. Values near 0.5 indicate weighted closure and this judge drives the staleness controller.",
        "staleness/audit_threshold_reached": "1 when the routine judge's early-stop validation |AUC-0.5| crossed the installed baseline + retrain margin. This permits an early stale decision only when the untouched final split also exceeds that threshold.",
        "staleness/audit_weighted_auc_gap_pvalue_approx": "Two-sided normal-approximation p-value for weighted held-out |AUC-0.5| using weight ESS for effective Gen sample size.",
        "staleness/audit_power_at_retrain_margin": "Approximate power to detect adaptive_omnifold.trigger.retrain_auc_margin with the actual held-out event count and weight ESS.",
        "staleness/audit_minimum_detectable_auc_gap": "Smallest |AUC-0.5| detectable at the configured target power under the null-Mann-Whitney approximation. Lower is better.",
        "staleness/audit_power_sufficient": "1 when audit_power_at_retrain_margin reaches the configured target power; monitoring only, not a retraining gate.",
        "staleness/weighted_auc_gap": "Fresh classifier held-out |weighted AUC-0.5|. A threshold crossing may trigger early after final-split confirmation; a non-crossing healthy decision requires saturation.",
        "staleness/raw_auc": "Validation truth-vs-unweighted-current-policy classifier AUC; may be warm-started. Interpret changes with classifier saturation and accumulated training in mind.",
        "staleness/raw_classifier_warm_started": "1 when the raw monitor fine-tunes its previous saved weights. Optimizer/early stopping reset; trust monitor is always fresh.",
        "staleness/raw_auc_gap": "Absolute raw classifier gap |AUC-0.5|; lower is better and this is the primary stationary-run progress metric.",
        "audit/raw_auc": "Clear alias of the fresh, unweighted truth-vs-current-policy audit AUC in measurement-only raw monitoring.",
        "audit/raw_auc_gap": "Clear alias of abs(audit/raw_auc-0.5); lower is better. This audit cannot refit the reward or roll back the policy.",
        "audit/saturated": "1 when the fresh audit classifier met its saturation criterion.",
        "audit/training_ready": "1 when the fresh audit met the configured training-readiness rule.",
        "audit/raw_auc_gap_change_from_step0": "Current measurement-only raw AUC gap minus the cold step-0 gap; negative means improvement.",
        "audit/raw_auc_gap_slope_per_10_steps": "Least-squares slope of the measurement-only raw AUC-gap trajectory, scaled to ten policy updates; negative means improvement.",
        "gradient_direction/installed_vs_fresh/cosine": "Cosine between the frozen installed-reward DGPO loss gradient and the fresh-audit substitute-reward DGPO loss gradient on one fixed panel. Positive means the two losses ask AdamW to descend in compatible directions.",
        "gradient_direction/installed_vs_fresh/cross_dot_lcb": "Lower confidence bound for the cross-block installed-vs-fresh gradient dot product; values above zero are evidence of positive alignment.",
        "gradient_direction/installed_vs_fresh/cross_dot_ucb": "Upper confidence bound for the cross-block installed-vs-fresh gradient dot product; values below zero are evidence of conflict.",
        "gradient_direction/installed_vs_fresh/alignment": "1 only when both gradient estimates are reliable and the installed-vs-fresh cross-dot interval is strictly positive.",
        "gradient_direction/installed_vs_fresh/conflict": "1 only when both gradient estimates are reliable and the installed-vs-fresh cross-dot interval is strictly negative.",
        "gradient_direction/installed/reliable": "1 when the installed-reward gradient has nonzero self-signal and adequate split-half cosine.",
        "gradient_direction/fresh/reliable": "1 when the fresh-audit gradient has nonzero self-signal and adequate split-half cosine.",
        "gradient_direction/reference_trust_active": "0 in the no-reference diagnostic; confirms that reference trust contributed no applied gradient.",
        "staleness/raw_audit_repeats": "Number of independent fresh H4 judges aggregated at this fixed policy boundary.",
        "staleness/raw_auc_gap_stdev": "Between-judge sample standard deviation of raw |AUC-0.5| on the fixed event panel.",
        "staleness/raw_auc_gap_se": "Between-judge standard error of the mean raw |AUC-0.5|.",
        "staleness/fixed_schedule_diagnostic_raw_audit": "1 when a fresh repeated raw judge was measured at a deterministic reward boundary without controlling the refit decision.",
        "staleness/diagnostic_selection_blind": "1 when the raw audit is logging-only and cannot change policy updates, reward timing, rollback, or endpoint selection.",
        "staleness/raw_auc_gap_slope_per_10_steps": "Least-squares slope of the repeated-audit mean |AUC-0.5| trajectory, scaled to ten DGPO updates; negative is improvement.",
        "staleness/raw_auc_gap_change_from_step0": "Current repeated-audit mean |AUC-0.5| minus the fixed step-0 baseline; negative is improvement.",
        "staleness/fixed_schedule_without_audit": "1 when an age-scheduled reward refit bypasses classifier staleness decisions; the fresh reward's held-out cross-fit validation still runs.",
        "staleness/incumbent_weighted_audit_skipped": "1 when the installed reward's weighted staleness classifier is omitted at a deterministic refit boundary.",
        "staleness/independent_raw_audit_skipped": "1 when the truth-vs-policy raw judge is omitted at a deterministic refit boundary.",
        "staleness/raw_best_auc_gap": "Lowest saturated raw |AUC-0.5| observed in the current reward round.",
        "staleness/raw_no_improvement_streak": "Consecutive saturated raw monitors that did not improve the reward-round best by raw_improvement_min_delta.",
        "staleness/raw_no_improvement_patience": "Effective number of consecutive eligible non-improving raw monitors required before OmniFold refit (with configured rollback). Optional patience_schedule follows persisted DGPO global_step, never the rollback checkpoint step or classifier step.",
        "classifier_trust/balanced_accuracy": "Fresh held-out balanced accuracy for distinguishing the current policy from the installed round reference.",
        "classifier_trust/balanced_accuracy_upper": "Normal-approximation upper confidence bound used by the conservative classifier trust trigger.",
        "classifier_trust/max_balanced_accuracy": "Configured classifier trust ceiling; ideal equal-prior BA 0.525 corresponds to total variation 0.05.",
        "classifier_trust/estimated_total_variation": "Classifier-implied total variation proxy max(0, 2*oriented_balanced_accuracy-1).",
        "classifier_trust/trigger_recalibration": "1 when the current/reference classifier requests a forward reward/reference recenter.",
        "staleness/monitor_mode_raw_only": "1 when the routine epoch audit skips reward scoring, weighted staleness fitting, controller updates, and refits, and trains only the raw truth-vs-policy judge.",
        "staleness/monitor_mode_raw_plateau_refit": "1 when raw-AUC plateau detection and optional classifier trust can trigger OmniFold refitting; raw plateaus may first restore the best fixed-panel raw-AUC checkpoint.",
        "omnifold/incumbent_probe_skipped": "1 when a versioned forced startup refit skips the weighted classifier on the reward that is about to be replaced. The epoch-54 raw baseline plus candidate acceptance and installed-baseline certification still run.",
        "staleness/trigger_threshold": "Fixed installed-round threshold: baseline_auc_gap + retrain_auc_margin.",
        "staleness/previous_audit_auc_gap": "Weighted |AUC-0.5| from the immediately preceding routine audit. Diagnostic only; it is not the trigger anchor.",
        "staleness/next_trigger_threshold": "Current installed-baseline threshold; it remains fixed until a new reward round is installed.",
        "staleness/baseline_auc_gap": "Weighted held-out |AUC-0.5| measured with the same cheap routine-probe population size and fit protocol; it remains the trigger anchor until a better candidate is installed.",
        "staleness/auc_gap_delta_from_baseline": "Current weighted AUC gap minus the fixed installed-round baseline gap.",
        "staleness/required_auc_gap_increase": "Configured retrain_auc_margin required above the installed baseline.",
        "staleness/probe_exceedance_streak": "Number of consecutive eligible post-cooldown audits whose final held-out AUC gap exceeded the installed-round threshold.",
        "staleness/required_consecutive_epochs": "Configured number of consecutive threshold-crossing audits required before AUC-based recalibration.",
        "staleness/retrain_cooldown_epochs": "Minimum DGPO epochs after an accepted reward installation during which audits remain diagnostic and cannot trigger another refit.",
        "staleness/cooldown_active": "1 when the reward-refresh cooldown currently blocks AUC- and age-based recalibration.",
        "staleness/epochs_since_recalibration": "DGPO epochs since the last accepted reward installation; -1 before any accepted recalibration timestamp exists.",
        "staleness/reward_age_epochs": "DGPO epochs elapsed since the currently installed reward round was accepted.",
        "staleness/max_reward_age_epochs": "Configured maximum installed reward age; reaching it proposes a fresh cross-fit candidate at the next routine audit.",
        "staleness/age_refit_due": "1 when the installed reward has reached max_reward_age_epochs.",
        "staleness/age_trigger_recalibration": "1 when reward age, rather than the AUC-gap controller, authorizes a refit attempt.",
        "staleness/decision": "Adaptive controller decision: audit_unsaturated, healthy, threshold_exceeded, recalibrate, or a recalibration outcome.",
        "omnifold/bootstrap_on_start": "1 when the initial K=1 OmniFold population fit was performed inside DGPO before any policy optimizer step.",
        "omnifold/classifier_fits_total": "Number of classifier fits attempted in the residual sequence, including the final closure classifier. Trainable scope follows the active OmniFold classifier config.",
        "omnifold/iterations_fitted": "Number of saturated discriminating classifier snapshots stored in the cumulative reward; excludes the final closure-only classifier.",
        "omnifold/candidate/audit_balanced_accuracy": "Fresh held-out weighted balanced accuracy used only when the optional candidate acceptance audit is enabled.",
        "omnifold/baseline/audit_observed_auc_gap": "Fresh weighted |AUC-0.5| measured on the cheap routine-sized pool immediately before installing a candidate; this becomes the fixed staleness baseline.",
        "omnifold/baseline/probe_events": "Event count used to certify the cheap staleness baseline for the newly installed reward round.",
        "omnifold/acceptance_audit_enabled": "1 when a second fresh classifier gates candidate installation; 0 when saturated cross-fit residual closure installs the stack directly.",
        "omnifold/acceptance_max_balanced_accuracy": "Configured strict upper bound when the optional candidate acceptance audit is enabled.",
        "omnifold/topology_acceptance_repeats": "Number of independently seeded same-architecture EveNet+adapter+Fourier audit fits aggregated before candidate acceptance.",
        "reference_trust/round_acceptance/action": "Round raw-AUC decision: +1 significant improvement, 0 statistical plateau, -1 significant regression, +2 bypass/bootstrap.",
        "reference_trust/round_acceptance/improvement": "Previous installed raw |AUC-0.5| minus candidate raw |AUC-0.5|; positive is better.",
        "reference_trust/round_acceptance/required_improvement": "Uncertainty dead-band width: configured z times the combined previous/candidate raw-AUC standard error.",
        "reference_trust/round_acceptance/rollback_required": "1 when the live policy must be restored to the incumbent round_ref anchor before another optimizer step.",
        "reference_trust/round_acceptance/stop_requested": "1 after the configured failed-direction patience; the rolled-back incumbent checkpoint is saved before clean termination.",
        "reference_trust/trajectory/best_raw_auc_gap": "Lowest fixed-panel unweighted |AUC-0.5| observed among routine-audit policy checkpoints in the current trust-region direction.",
        "reference_trust/trajectory/best_updated": "1 when the current routine-audit policy replaces the best-on-trajectory checkpoint.",
        "reference_trust/trajectory/selected_raw_auc_gap": "Cheap fixed-panel raw-AUC gap of the trajectory checkpoint selected for the independent full refit/acceptance gate.",
        "reference_trust/trajectory/rewound": "1 when exhaustion restored an earlier best-on-trajectory checkpoint instead of evaluating the live endpoint.",
        "reference_trust/round_acceptance/failed_direction_streak": "Number of independently trained directions rejected by the statistical round-AUC gate since the last accepted improvement.",
        "reference_trust/signed_probe/decision_code": "Signed-direction result: +1 only the trained sign significantly improves raw AUC, -1 only the reverse sign improves, +2 both improve, and 0 neither improves.",
        "reference_trust/signed_probe/best_overall_scale": "Scale alpha with the smallest raw |AUC-0.5| among theta_ref +/- alpha*(theta-theta_ref); no probed point is installed.",
        "reference_trust/signed_probe/reward_raw_misaligned": "1 when the probed point with the largest installed-reward mean exceeds the anchor reward while significantly worsening raw |AUC-0.5|.",
        "reference_trust/signed_probe/smaller_positive_step_improves": "1 when a positive scale below one significantly improves raw AUC but the trained +1 endpoint does not, indicating directionally useful overshoot.",
        "reference_trust/signed_probe/all_raw_audits_saturated": "1 when the anchor and every signed candidate classifier reached the configured validation-loss saturation criterion.",
        "reference_trust/signed_probe/candidate/*/raw_auc_gap": "Unweighted truth-vs-policy |AUC-0.5| at one signed parameter scale on the fixed 100k event/noise/classifier-seed panel; lower is better.",
        "reference_trust/signed_probe/recovery_triggered": "1 when an enabled, fully saturated reverse-only result rejects the local direction, restores round_ref, clears optimizer state, and requests a fresh reward fit.",
        "reference_trust/signed_probe/recovery_attempt": "Consecutive independently trained reverse-only directions since the last genuinely improved policy round.",
        "reference_trust/signed_probe/recovery_reward_installed": "1 when the fresh reward trained at the restored incumbent passed all classifier closure, acceptance, and topology gates.",
        "reference_trust/signed_probe/recovery_stop_requested": "1 when repeated reverse-only recoveries reach failed_direction_patience; training stops at the restored incumbent.",
        "reference_trust/extragradient/triggered": "1 when trust exhaustion launches the opt-in predictive-corrective block instead of directly committing the selected trajectory point.",
        "reference_trust/extragradient/lookahead_scale": "Fraction of the selected incumbent-to-trajectory displacement used for the virtual policy that trains the response classifier.",
        "reference_trust/extragradient/lookahead_reward_installed": "1 when the transient look-ahead reward passes residual closure, acceptance, and topology gates; it is not yet a committed policy point.",
        "reference_trust/extragradient/rebased_optimizer_params": "Number of trainable tensors restored to the incumbent after evaluating the corrector gradient at the look-ahead pair and before optimizer.step.",
        "reference_trust/extragradient/corrector_distance": "Functional distance of the corrected policy from the original incumbent on its fixed trust probe.",
        "reference_trust/extragradient/corrector_scale": "Post-gradient interpolation scale accepted by the original incumbent trust region.",
        "reference_trust/extragradient/final_reward_installed": "1 only when the corrected policy passes a new reward fit plus the original-incumbent paired round-AUC gate and becomes the committed pair.",
        "reference_trust/extragradient/final_optimizer_states_cleared": "Number of transient corrector AdamW states discarded after the corrected policy and final reward/reference pair are committed.",
        "reference_trust/extragradient/rejected": "1 when either classifier gate fails, the corrector is zero/non-finite, or trust backtracking fails; the complete incumbent pair is restored.",
        "omnifold/fit/iter*/saturated": "1 when that residual classifier fit reached its configured validation saturation/early-stop condition.",
        "omnifold/fit/iter*/stored_in_reward": "1 when that saturated classifier snapshot was stored as a cumulative log-ratio increment; 0 for the final closure-only classifier.",
        "omnifold/fit/iter*/warm_started_folds": "Number of repeat/fold classifiers initialized from the previous installed round's matching iteration. 0 means fresh initialization; the complete count is crossfit_repeats * crossfit_folds.",
        "omnifold/fit/iter*/validation_balanced_accuracy": "Held-out balanced accuracy for this residual fit. The next iteration begins only after saturation; closure is assessed against the configured chance band.",
        "reward_rank_audit/fixed_panel/*": "Candidate-ordering agreement on the checkpointed K=8 policy-update panel. Current-member metrics measure within-round ensemble agreement; temporal metrics compare adjacent fresh reward rounds on identical candidates.",
        "reward_consensus/live/*": "Live policy-update candidate-consensus diagnostics. The hard gate requires the configured member sign agreement; relative MAD is computed after removing each member's positive per-event scale.",
        "val_ztautau/target/*": "Truth/current/reference 1D density overlays for the four diffusion targets. Current and reference use candidate 0, never reward-best selection.",
        "val_ztautau/reco/*": "Truth/current/reference 1D density overlays for reconstructed tau theta/phi directions, using the shared Ztautau direction reconstruction.",
        "val_ztautau/topology/*": "Truth/current/reference 1D density overlays for tau-pair opening, acoplanarity, back-to-back loss, and post-calibration direction-change magnitudes.",
        "val_ztautau/jsd/current/*": "Jensen-Shannon distance between truth and unbiased candidate-0 current-policy physics distributions. Lower is better.",
        "val_ztautau/jsd/ref/*": "Jensen-Shannon distance between truth and frozen-reference candidate-0 physics distributions. Lower is better.",
        # --- train_dist (epoch end, accumulated over all training batches; own wandb panel) ---
        "train_dist/pt": "1D density overlay: truth vs best-of-K training prediction for pT [GeV] (original scale), accumulated over all training batches in the epoch (wandb.Image). x-axis **epoch**. \"Best\" = combined-reward argmax among K candidates.",
        "train_dist/eta": "Same overlay for η (training); same candidate selection as train_dist/pt.",
        "train_dist/phi": "Same overlay for φ [rad] (training); same candidate selection as train_dist/pt.",
        "train_dist/jsd/current/*": "Histogram Jensen-Shannon distance between truth and reward-best train rollout distributions, accumulated over the epoch. Feature names follow event_info.yaml invisible_feature_names (or px/py/pz in cartesian mode). Lower is better.",
        "train_dist/all/pt_truth_vs_pred": "Example 2D truth-vs-pred key. Actual train epoch-end 2D keys follow event_info.invisible_feature_names as train_dist/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "train_dist/all/eta_truth_vs_pred": "Example 2D truth-vs-pred key. Actual train epoch-end 2D keys follow event_info.invisible_feature_names as train_dist/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "train_dist/all/phi_truth_vs_pred": "Example 2D truth-vs-pred key. Actual train epoch-end 2D keys follow event_info.invisible_feature_names as train_dist/all/{feature}_truth_vs_pred and use Generation-Binning neutrino-{feature} when configured.",
        "train_dist/all_metrics/*/*": "Scalar summaries for train truth-vs-pred 2D monitors over all candidates: count, mae, rmse, bias=mean(pred-truth), pearson_r, slope, intercept.",
        "train_dist/by_class/*/*_truth_vs_pred": "Representative EVENT/signal class-specific train truth-vs-pred matrices over all K candidates. The selected classes and cadence are configured under dgpo.train_dist_representative_classes and train_dist_by_class_every_n_epochs.",
        "train_dist/by_class_metrics/*/*/*": "Count, MAE, RMSE, bias, Pearson r, slope, and intercept for each representative class/feature matrix.",
        "train_dist_k1/pt": "1D density overlay: truth vs candidate-0 training rollout prediction for pT [GeV], accumulated over all training batches in the epoch. This is a K=1 / single-sample proxy on the train rollout pool, separate from reward-best train_dist/*.",
        "train_dist_k1/eta": "Same overlay for η using candidate 0 as the train K=1 proxy.",
        "train_dist_k1/phi": "Same overlay for φ using candidate 0 as the train K=1 proxy.",
    }


def _dgpo_wandb_hyperparameter_definitions() -> dict[str, str]:
    """Explicit definitions for DGPO hyperparameters (visible in W&B Config).

    Explains what each parameter in the ``dgpo:`` and ``reward_config:`` sections does.
    """
    return {
        # --- dgpo: core RL hyperparameters ---
        "dgpo.beta": "beta_dgpo: Temperature parameter that scales the event-level gate logit M_e. Higher beta makes the gate more sensitive to the advantage-weighted velocity gap. M_e = (beta / K) * sum_over_candidates(advantage * Delta). Typical range: 0.1 to 1.0. Current value controls how aggressively events are up/down-weighted in the loss.",
        "dgpo.float32_matmul_precision": "PyTorch float32 matrix-multiplication precision. 'medium' allows faster reduced-internal-precision tensor-core algorithms while keeping float32 inputs/outputs; 'high' preserves the previous DGPO default.",
        "dgpo.advantage_estimator": "Per-event candidate baseline. OmniFold requires 'leave_one_out_unscaled' so the learned log-density-ratio scale is retained; legacy rewards default to 'zscore'.",
        "dgpo.grad_clip_norm": "Global L2 gradient clip for AdamW (torch.nn.utils.clip_grad_norm_). Compare train/grad/global_norm_pre_clip to this value.",
        "dgpo.K": "Number of DDIM candidate samples generated per event **during training** (rollout + DGPO). Each event gets K neutrino reconstructions, and the reward function ranks them. Larger K = more candidates to choose from (better oracle performance) but slower generation. Typical values: 4-16.",
        "dgpo.validation_K": "Candidates per event during **validation** only (independent of training K). Default **1**: one current-policy DDIM sample per event; with validation_compute_winrate, one additional ref-policy DDIM per event for reward-based winrate. Does not advance the training global_step or train-panel x-axis.",
        "dgpo.validation_cheap_K": "Current-policy candidates per event for the cheap scalar-only validation tier. This tier skips reference DDIM, images, per-event gathers, Ztautau panels, and TARP.",
        "dgpo.validation_batch_size": "Per-worker validation event batch, independent of platform.batch_size. Keeping this fixed allows a larger DGPO training batch without changing validation event count, response alignment, or DDIM peak memory.",
        "dgpo.rollout_parallel_chains": "How many DDIM chains to batch together per model call during training rollout. Keeps total K the same, but runs up to this many chains in one larger forward pass. Higher can be faster if GPU headroom exists; too high can OOM.",
        "dgpo.validation_rollout_parallel_chains": "Validation-only version of rollout_parallel_chains. If unset, falls back to the training value.",
        "dgpo.num_ddim_steps": "Number of DDIM denoising steps (T_sample) used for online candidate generation during **training** only. More steps = higher-quality samples but slower. Typical values: 20-100.",
        "dgpo.validation_num_ddim_steps": "Number of DDIM denoising steps used for candidate generation during **validation** only (independent of training num_ddim_steps). If null, falls back to num_ddim_steps. Lets you validate at higher fidelity than the training rollout without slowing training.",
        "dgpo.policy_eval_parallel_timesteps": "Independent policy-evaluation (t, eps) draws batched into one current/reference forward. Keeps dgpo.num_train_timesteps and the one-AdamW-step objective unchanged; higher values reduce launches but increase activation memory roughly proportionally.",
        "dgpo.policy_eval_event_microbatch_size": "Event rows per gradient-bearing current/reference policy-evaluation forward. Gradients are weighted and accumulated into the same one-AdamW-step full-batch objective, preserving K and num_train_timesteps while reducing activation memory. Null uses the full DGPO batch.",
        "dgpo.sequential_vp_trust_backward": "When true with a VP path-KL reference trust objective, backward the DGPO graph before constructing the independent trust graph, then average the fully accumulated gradients once across workers. This preserves the summed objective while avoiding simultaneous activation graphs.",
        "dgpo.activation_checkpointing": "Recompute trainable PET transformer blocks during backward instead of retaining their intermediate activations. Reduces current-policy K*B peak memory while preserving FP32 parameters and loss computation.",
        "dgpo.diagnostic_profile_accumulate_steps": "Number of train batches to concatenate before logging accumulated diagnostics/reward_hacking/profile/*_delta_vs_truth_* images. Larger values stabilize sparse bins but update the W&B images less often.",
        "dgpo.log_every": "Log Python INFO messages (loss, reward, etc.) to the console every N optimizer steps. Does not affect wandb logging frequency (wandb logs every step when enabled). Typical: 1-10.",
        "dgpo.log_reward_dist_every": "Log reward/dist/overlap every N optimizer steps when wandb is enabled. Defaults to dgpo.diagnostic_plots.plot_every when omitted.",
        "dgpo.train_dist_enabled": "Enable duplicate train-rollout truth/pred arrays, cross-rank gathers, and epoch media. False is recommended when full validation already monitors the same distributions.",
        "dgpo.train_dist_representative_classes": "EVENT/signal class names that receive separate train_dist/by_class truth-vs-pred 2D matrices in addition to the pooled train_dist/all panels.",
        "dgpo.train_dist_by_class_every_n_epochs": "Cadence for representative per-class train_dist matrices. Epoch 0 is always logged; later panels follow this logical-epoch cadence to bound W&B/Matplotlib overhead.",
        "dgpo.train_dist_every_n_epochs": "Cadence for collecting, cross-rank gathering, and plotting pooled train_dist arrays. Epoch 0 is included. Outside this cadence, train_step does not materialize the large CPU truth/pred arrays.",
        "dgpo.steps_per_epoch": "Optional positive optimizer-step budget for one logical DGPO epoch. When set, the Ray training iterator remains open across epoch boundaries and is restarted only after the shard is exhausted, so short logical epochs do not repeatedly consume the start of the dataset. When omitted, one epoch is one complete dataset pass.",
        "dgpo.log_parameter_update_rms": "Snapshot trainable parameters around each optimizer step and log their actual RMS displacement. This short-run diagnostic consumes one additional trainable-parameter snapshot.",
        "dgpo.parameter_update_rms_calibration": "Optional target-preserving global AdamW step-size controller. After the native AdamW proposal advances its moments, rescale the complete proposed parameter displacement to target_rms while keeping its direction and relative parameter-group geometry fixed. Incompatible with hard trust, post-AdamW projection, and extragradient.",
        "dgpo.gradient_transfer_trace": "Read-only exact-production trace of the H4 loss gradient, unweighted soft-reference gradient, reconstructed total gradient, and native AdamW displacement on selected applied update endpoints. It does not change the configured objective.",
        "dgpo.fail_on_skipped_optimizer_step": "Abort when a logical DGPO step does not commit an optimizer update. Intended for short fixed-update experiments whose round length must count actual updates.",
        "dgpo.validation_every_n_epochs": "Cadence for the cheap scalar-only validation tier. Epoch -1 baseline remains a full validation; a coincident full validation replaces rather than duplicates the cheap pass.",
        "dgpo.validation_full_every_n_epochs": "Cadence for full validation_K monitoring with reference rollout, response/profile arrays, 2D panels, Ztautau metrics, TARP, and top-K checkpoint selection. A coincident full validation replaces the cheap tier.",
        "dgpo.validation_initial_enabled": "Run the full epoch -1 validation baseline when starting at policy step zero. Disable only for short causal diagnostics whose endpoint is an independent classifier audit.",
        "dgpo.validation_cheap_max_batches": "Per-rank batch cap for the scalar-only cheap validation tier. Separate from validation_max_batches used by full monitoring.",
        "dgpo.validation_max_batches": "If set (e.g. 20): stop validation after this many batches per epoch (faster validation for debugging). If null: run full validation set. Typical: null for real training, 5-20 for smoke tests.",
        "dgpo.validation_compute_winrate": "Full-validation-only switch: generate one extra reference-policy DDIM sample for reward-based val/winrate. Cheap validation never runs the reference policy.",
        "dgpo.checkpoint_every_n_steps": "Periodic last-checkpoint cadence, evaluated only at logical-epoch boundaries. Adaptive audit/refit boundaries and the final epoch are also saved. This replaces duplicate in-step plus every-epoch writes.",
        "dgpo.validation_log_batches": "If true: log INFO messages for each validation batch (start time, DDIM wall time). Useful for monitoring long validation runs. Typical: true.",
        "dgpo.validation_cheap_log_batches": "If true, emit per-batch logs for cheap validation. False keeps the frequent two-batch probe quiet.",
        "dgpo.validation_tqdm_k_chains": "If true: show a tqdm progress bar over the K DDIM chains per validation batch. Typical: true (helps see validation progress).",
        "dgpo.validation_tqdm_ddim": "If true: show a tqdm progress bar for every DDIM step within each chain (very verbose). Typical: false (too much output).",
        "dgpo.adaptive_omnifold.enabled": "Run K=1 fresh EveNet audits during DGPO and refit/install a new ratio stack plus its paired round-reference policy when stale.",
        "dgpo.adaptive_omnifold.monitor_mode": "Use weighted_and_raw for the legacy staleness controller, or raw_only to fit only the truth-vs-unweighted-policy classifier during stationary log-only monitoring.",
        "dgpo.adaptive_omnifold.staleness_every_n_epochs": "Independent cadence for the fresh K=1 audit; it does not change validation_K or dgpo.K.",
        "dgpo.adaptive_omnifold.pool_generation_batch_size": "Per-worker Ray batch used only to generate no-grad K=1 live-policy populations for OmniFold fit and stale probes. Independent of platform.batch_size and classifier fit.batch_size.",
        "dgpo.adaptive_omnifold.audit_fit.train_microbatch_size_per_rank": "Per-GPU gradient-bearing audit classifier rows per class and forward. Multiple microbatches reconstruct one configured global optimizer batch without retaining their activation graphs.",
        "dgpo.adaptive_omnifold.trigger.probe_max_events": "Event cap for the frequent K=1 staleness audit. This may be smaller than recalibration.score_pool_events; it does not reduce the residual-validation population.",
        "dgpo.adaptive_omnifold.trigger.rollback_to_best_on_plateau": "When the raw-AUC plateau patience fires, restore the lowest fixed-panel raw-AUC epoch checkpoint in the current reward round, clear stale optimizer state, refresh EMA, and fit the next OmniFold reward from that restored policy.",
        "dgpo.adaptive_omnifold.classifier_trust.every_n_epochs": "Cadence for the fresh current-vs-round-reference classifier. It must be a multiple of staleness_every_n_epochs, so trust checks can reuse the raw-monitor event pass.",
        "dgpo.adaptive_omnifold.classifier_trust.probe_max_events": "Global event cap for the paired current/reference classifier-trust prefix. Reference DDIM generation stops at this cap even when the raw monitor uses a larger population.",
        "dgpo.adaptive_omnifold.classifier_trust.audit_fit": "Optional fit-setting overrides used only by the classifier-trust monitor, such as a shorter validation_patience_epochs; the raw monitor retains adaptive_omnifold.audit_fit.",
        "dgpo.adaptive_omnifold.classifier_trust.unsafe_early_stop": "Optional one-sided classifier-trust early exit. It stops an otherwise unsaturated trust classifier only after the oriented validation balanced-accuracy lower confidence bound exceeds max_balanced_accuracy for the configured number of consecutive validations; that positive unsafe witness immediately triggers recalibration.",
        "dgpo.adaptive_omnifold.trigger.retrain_auc_margin": "AUC-based retraining threshold. Refit when weighted |AUC-0.5| is strictly greater than installed baseline_auc_gap plus this margin.",
        "dgpo.adaptive_omnifold.trigger.max_reward_age_epochs": "Optional maximum installed reward age. At the next regular audit, reaching this age forces a fresh cross-fit candidate; configured residual closure is still required before installation.",
        "dgpo.adaptive_omnifold.trigger.fixed_schedule_skip_staleness_audit": "When every age boundary deterministically refits the reward, skip the incumbent weighted and independent raw classifiers because they cannot change that decision. Reward cross-fit validation remains mandatory.",
        "dgpo.adaptive_omnifold.trigger.fixed_schedule_log_raw_audit": "With deterministic refits, fit fresh repeated raw truth-vs-policy judges at step 0 and every reward boundary for selection-blind trajectory measurement.",
        "dgpo.adaptive_omnifold.audit_fit.repeats": "Independent fresh raw-audit seeds and identity splits per policy boundary; their mean |AUC-0.5| is the primary trajectory statistic.",
        "dgpo.adaptive_omnifold.audit_fit.training_population": "probe_split retains the legacy internal audit split. omnifold_fold trains a cold diagnostic judge on one fixed repeat-1 OmniFold training fold with newly generated current-policy samples, splitting external validation 50/50 into early-stop and final-test.",
        "dgpo.adaptive_omnifold.audit_fit.training_fold": "One-based OmniFold training fold for the matched raw audit; uses the same identity hash and recalibration seed as reward repeat 1. Does not inherit classifier weights or generated samples.",
        "dgpo.adaptive_omnifold.trigger.required_consecutive_epochs": "Number of consecutive eligible routine audits that must exceed the installed-round threshold before AUC-based refitting.",
        "dgpo.adaptive_omnifold.trigger.patience_schedule": "Optional monotone list of {start_step, required_consecutive_checks}. Step-based raw plateau monitoring uses the latest stage reached by persisted DGPO global_step; no schedule retains fixed patience. Warmup pause and saturation eligibility are unchanged.",
        "dgpo.adaptive_omnifold.trigger.retrain_cooldown_epochs": "Minimum epochs after an accepted reward installation before another routine AUC- or age-triggered refit is allowed; audits continue during cooldown.",
        "dgpo.adaptive_omnifold.fixed_audit_panel": "Reuse the same event-selection, rollout, and classifier seeds at every raw audit so epoch-to-epoch AUC changes are paired rather than panel noise.",
        "dgpo.adaptive_omnifold.recalibration.reset_adam_first_moment_on_install": "After an accepted atomic reward/reference install, clear AdamW exp_avg only; preserve exp_avg_sq, step counters, scheduler, and LR.",
        "dgpo.adaptive_omnifold.recalibration.reset_optimizer_state_on_install": "After each accepted reward/reference install, clear all AdamW parameter state (first/second moments and step counters) and gradients; preserve weights, LR/scheduler and weight decay. Takes precedence over first-moment-only reset.",
        "dgpo.reference_trust.objective": "Round-reference trust objective: legacy dimension-mean velocity_mse or the separately sampled full-time cosine-VP reverse-path estimator vp_path_kl.",
        "dgpo.reference_trust.vp_path_kl.diagnostic": "When objective=velocity_mse, run an additional graph-free full-time VP path-KL measurement without changing backward. Ignored as a backward switch when objective=vp_path_kl.",
        "dgpo.reference_trust.adaptive_boundary.reset_adam_first_moment_on_zero_step": "Clear AdamW first moments after an alpha=0 trust rejection so outward momentum is not carried across rejected policy steps.",
        "dgpo.reference_trust.adaptive_boundary.transactional_rejection": "Snapshot AdamW state before a hard-boundary proposal and restore every moment and step counter when no positive backtracking scale is feasible.",
        "dgpo.reference_trust.adaptive_boundary.stop_after_rejection": "After the first transactional alpha=0 rejection, run the endpoint cold audit and end the experiment at that policy boundary.",
        "dgpo.reference_trust.adaptive_boundary.round_decay_factor": "With radius_mode=round_decay, multiply the hard radius by this factor only after a successful new reward/reference installation, down to delta_floor. The KL coefficient and policy weight decay are independent.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.cross_round_nonexpanding": "Cap each newly installed reward round radius by the previous installed radius.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.scale_policy_lr": "Multiply only the DGPO policy LR by sqrt(delta_r/delta_0); OmniFold classifier LRs are unchanged.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.policy_lr_scale_floor": "Minimum DGPO policy LR multiplier used by trust-radius scaling.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.exhaustion_scale_window_steps": "Trailing optimizer-step window used only to decide trust-boundary exhaustion; the longer empirical-controller history remains unchanged.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.round_acceptance_enabled": "Require a same-protocol candidate round to reduce raw |AUC-0.5| beyond statistical uncertainty; regressions roll back to round_ref.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.round_acceptance_confidence_z": "Normal-approximation z multiplier for the cross-round raw-AUC improvement dead band.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.round_plateau_patience": "Consecutive statistically indistinguishable candidate rounds required before anchor rollback and clean early stop.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.trajectory_search_enabled": "Use one fixed event/noise/classifier-seed panel to rank routine-audit checkpoints, then evaluate the best actual trajectory point instead of only the exhausted endpoint.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.failed_direction_patience": "Number of statistically non-improving best-on-trajectory attempts allowed; each failure rolls back immediately, contracts the radius, clears optimizer state, and starts a new direction.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.signed_direction_probe.enabled": "Once per independently trained direction, audit theta_ref +/- alpha*(theta-theta_ref) on the fixed raw-AUC panel, then restore the live policy after the probe itself.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.signed_direction_probe.scales": "Positive alpha magnitudes used for both signs of the diagnostic direction probe; must be unique in (0,1] and include 1.0.",
        "dgpo.reference_trust.adaptive_boundary.radius_calibration.signed_direction_probe.recover_reverse_only": "When true, a fully saturated reverse-only result rejects the local direction, restores round_ref, clears AdamW state, and refits the reward on a fresh training population while preserving the paired audit panel.",
        "dgpo.adaptive_omnifold.trigger.require_audit_saturation": "If true, an unsaturated audit is diagnostic only: it resets the exceedance streak and cannot trigger retraining.",
        "dgpo.adaptive_omnifold.recalibration.train_parquet_dir": "Optional independent training directory for OmniFold bootstrap/refits. When set, DGPO continues to use platform.data_parquet_dir while the driver registers this complete directory as the dedicated OmniFold shard.",
        "dgpo.adaptive_omnifold.recalibration.pool_events": "Optional cap on the current-policy K=1 fit population. When set, the driver creates one seeded subset and bootstrap/refits reuse the same identities; null drains the complete recalibration.train_parquet_dir when configured, otherwise the ordinary train shards.",
        "dgpo.adaptive_omnifold.recalibration.score_pool_events": "Held-out K=1 population used for cross-fit residual gating. Kept separate from the routine staleness probe.",
        "dgpo.adaptive_omnifold.recalibration.fit.train_microbatch_size_per_rank": "Per-GPU gradient-bearing residual-classifier rows per class and forward. Gradients accumulate across these chunks before one optimizer step and one distributed gradient average.",
        "dgpo.adaptive_omnifold.recalibration.fit.gradient_clip_norm": "Optional global-norm clip applied to the distributed residual-classifier gradient before AdamW. Non-finite local loss/gradients and post-step parameters always fail before they can propagate further.",
        "dgpo.adaptive_omnifold.recalibration.crossfit_repeats": "Independent identity-stable K-fold partitions per residual iteration. Default 1 preserves the existing protocol. Each event receives one unseen OOF logit per repeat; those logits are averaged before tempering. Outer validation and frozen reward average all repeat/fold models. Five held-out trainings use crossfit_folds=5 with repeats=1, not five two-fold repeats.",
        "dgpo.adaptive_omnifold.recalibration.max_reward_rounds": "Optional cap on installed reward rounds, including bootstrap. Monitoring continues after the cap but cannot trigger another classifier fit or reward installation.",
        "dgpo.adaptive_omnifold.recalibration.scheduled_refit_fail_closed": "Abort when an age-scheduled fresh reward fails its classifier gate, instead of continuing a fixed-round experiment with the prior stale reward.",
        "dgpo.adaptive_omnifold.recalibration.warm_start_iterations": "One-based residual iterations to fine-tune from previous-round repeat/fold weights. Uses stable condition-hash partitions, fresh AdamW state and recomputed log weights. Legacy single-repeat checkpoints remain compatible when crossfit_repeats=1. Raw and trust monitors remain fresh.",
        "dgpo.adaptive_omnifold.recalibration.residual_min_auc_gain": "The sole outer-iteration usefulness gate: a residual classifier is stored only when held-out AUC is strictly greater than 0.5 plus this value. Held-out BCE remains diagnostic. Larger values widen the closure band and stop residual iterations earlier.",
        "dgpo.adaptive_omnifold.recalibration.acceptance_audit_enabled": "Enable a second fresh-classifier candidate gate after cross-fit residual closure. Disable it to install directly at residual closure.",
        "dgpo.adaptive_omnifold.recalibration.acceptance_max_balanced_accuracy": "When the optional candidate acceptance audit is enabled, require its fresh held-out weighted balanced accuracy to be strictly below this value.",
        "dgpo.adaptive_omnifold.recalibration.topology_acceptance_repeats": "Independent classifier seeds for the repeated acceptance audit. Each fit uses the exact OmniFold EveNet+adapter+Fourier architecture and trainable scope; canonical topology-named metrics are their means.",
        "dgpo.adaptive_omnifold.recalibration.pool_selection_seed": "Seed used once by the driver to choose the fixed global OmniFold fit identities. It does not control K=1 DDIM noise or classifier optimization.",
        # --- reward_config ---
        "reward_config.type": "Reward backend. 'omnifold' uses the frozen truth-free conditional log-density-ratio bundle; 'component_normalized_truth_distance' uses truth matching; 'calibration_magnitude' uses tau direction-change magnitude.",
        "reward_config.weight": "Global multiplier on the configured reward source before summing into the combined DGPO reward. Typical: 1.0.",
        "reward_config.omnifold.bundle_file": "Path to omnifold_reward.pt produced from the fixed K=1 train/validation populations.",
        "reward_config.omnifold.backbone_checkpoint": "Supervised EveNet checkpoint used to initialize the OmniFold classifier body inside DGPO. In frozen-backbone mode, classifier fits share that body and train independent PEFT banks.",
        "reward_config.omnifold.bootstrap_in_dgpo": "When true, DGPO fits and installs the initial K=1 residual ratio stack before any policy optimizer step; no prefit reward bundle is required.",
        "dgpo.adaptive_omnifold.recalibration.bootstrap_on_start": "Run the fail-closed initial residual sequence inside DGPO. Every classifier fit must saturate before a discriminating snapshot is stored.",
        "reward_config.component_normalized.eps": "Numerical stability added to per-component scale denominators in the truth-distance reward. Unused by calibration_magnitude.",
    }


def _dgpo_wandb_publish_metric_docs() -> None:
    """Expose metric definitions in the W&B UI (Config, Summary, Notes, Artifact).

    W&B does not show definitions next to each chart; Config + Artifact are the supported surfaces.
    """
    import wandb

    run = wandb.run
    if run is None:
        return
    if _wandb_simplified_enabled():
        # The compact profile deliberately keeps W&B Config and Artifacts free
        # of a second, very large copy of documentation already tracked in git.
        try:
            run.summary["dgpo_logging_profile"] = "critical" if _wandb_critical_enabled() else "simplified"
            run.summary["dgpo_clock_schema"] = "completed-policy-steps-v2"
        except Exception as e:
            _log.warning("[DGPO] wandb simplified-profile summary failed: %s", e)
        return
    defs = _dgpo_wandb_metric_definition_map()
    param_defs = _dgpo_wandb_hyperparameter_definitions()
    try:
        wandb.config.update(
            {
                "dgpo_metric_definitions": defs,
                "dgpo_hyperparameter_definitions": param_defs,
                "dgpo_metrics_full_doc_repo_path": (
                    "RL/DGPO_neutrino/diagnostics/metrics_reference.md"
                ),
                "dgpo_dynamic_reward_keys_note": (
                    "Training/controller metrics use completed DGPO global_step; validation uses epoch. "
                    "W&B _step is a separate committed event-row index, not a training counter. "
                    "Groups: reward/dist, reward/monitor, train/loss, train/grad, parameter/*, "
                    "components/*, diagnostics/reward_hacking/* (including all/ vs best/), "
                    "diagnostics/reference_bias/*; projection/* + swd/* (latent-SWD CPO repair, x-axis global_step); "
                    "train_dist/*, train_dist_k1/*; val/reward/*, val/winrate, "
                    "val_diagnostics/*, val_neutrino/*, val_ztautau/*, val_tarp/*."
                ),
            },
            allow_val_change=True,
        )
    except Exception as e:
        _log.warning("[DGPO] wandb.config metric definitions failed: %s", e)
    try:
        run.summary["dgpo_how_to_read_metrics"] = (
            "Config → dgpo_metric_definitions (metrics) + dgpo_hyperparameter_definitions (params). "
            "Artifacts → dgpo-metrics-reference → metrics_reference.md (full doc)."
        )
    except Exception as e:
        _log.warning("[DGPO] wandb.summary metric pointer failed: %s", e)
    try:
        run.notes = (
            "DGPO metric + hyperparameter definitions: open this run's **Config** "
            "(dgpo_metric_definitions, dgpo_hyperparameter_definitions) "
            "or **Artifacts** (dgpo-metrics-reference). "
            "Repo copy: RL/DGPO_neutrino/diagnostics/metrics_reference.md"
        )
    except Exception as e:
        _log.warning("[DGPO] wandb run notes failed: %s", e)
    md_path = Path(__file__).resolve().parent / "diagnostics" / "metrics_reference.md"
    if md_path.is_file():
        try:
            art = wandb.Artifact("dgpo-metrics-reference", type="documentation")
            art.add_file(str(md_path), name="metrics_reference.md")
            run.log_artifact(art)
        except Exception as e:
            _log.warning("[DGPO] wandb artifact for metrics_reference.md failed: %s", e)


def _wandb_is_media_value(v: Any) -> bool:
    """True for ``wandb`` loggable media types (histograms, images, etc.)."""
    mod = getattr(type(v), "__module__", "") or ""
    name = getattr(type(v), "__name__", "")
    return mod.startswith("wandb") and name in (
        "CustomChart",
        "Histogram",
        "Image",
        "Plotly",
        "Video",
        "Html",
    )


def _wandb_simplified_enabled() -> bool:
    """Whether the configured W&B run uses the compact, decision-focused profile."""

    section, _ = _dgpo_wandb_yaml_section()
    return bool(_dgpo_cfg_get(section, "simplified", False)) or _wandb_critical_enabled()


def _wandb_critical_enabled() -> bool:
    section, _ = _dgpo_wandb_yaml_section()
    return str(_dgpo_cfg_get(section, "profile", "")) == "critical"


# An explicit allowlist avoids hundreds of automatic panels from dormant
# ablations, duplicate residual statistics and per-process response matrices.
_WANDB_CLASSIFIER_TRAINING_CHARTS = frozenset(
    f"Classifier training/{role}/{metric}"
    for role in ("Reward", "Fresh audit")
    for metric in (
        "Train loss",
        "Validation loss",
        "Validation AUC",
        "Validation accuracy",
        "Learning rate",
        "Base learning rate",
        "Fixed probe BCE before update",
        "Fixed probe BCE after update",
        "Fixed probe BCE change",
        "Fixed probe separation before update",
        "Fixed probe separation after update",
        "Fixed probe logit change RMS",
        "Largest rank-local gradient norm",
        "Gradient spike trigger",
        *(f"Fixed representation probe: {branch} holdout AUC"
          for branch in ("raw_fourier", "normalized_fourier", "fourier", "decoder", "concat", "fusion", "current_head")),
        *(f"CV representation probe: {branch} holdout AUC"
          for branch in ("raw_fourier", "normalized_fourier", "fourier", "decoder", "concat", "fusion")),
        *(f"{branch} {metric}" for branch in ("fourier", "decoder", "fusion")
          for metric in ("activation_rms", "gradient_rms")),
        *(f"Decoder {branch} gate RMS" for branch in ("self", "cross", "ffn")),
        *(f"{label} {description}"
          for label in ("Adapter", "Decoder", "Backbone", "Fourier context", "Output head")
          for description in ("AdamW update RMS", "relative AdamW update RMS")),
        "Pre-clip gradient norm",
        "Gradient clip scale",
        "Gradient clip fraction",
        "Direct topology head gradient norm",
        "Topology context gradient norm",
        "Context output head gradient norm",
        "Decoder gradient norm",
        "Adapter gradient norm",
        "Input projector gradient norm",
        "Truth logit mean",
        "Generated logit mean",
        "Logit class separation",
        "Logit RMS",
        "Saturated-logit fraction",
        "Direct head parameter RMS",
        "Direct head gradient RMS",
        "Direct head relative gradient RMS",
        "Topology context parameter RMS",
        "Topology context gradient RMS",
        "Topology context relative gradient RMS",
    )
)
_WANDB_CRITICAL_CHARTS = frozenset({
    "val/reward/all_sample_mean", "val/reward/best_of_k_mean",
    "val_cheap/reward/all_sample_mean", "val_cheap/reward/best_of_k_mean",
    "frozen_classifier/events",
    "frozen_classifier/ensemble/auc", "frozen_classifier/ensemble/auc_gap",
    "frozen_classifier/fold01/auc", "frozen_classifier/fold01/auc_gap",
    "frozen_classifier/fold02/auc", "frozen_classifier/fold02/auc_gap",
    "train/loss/total", "train/loss/dgpo", "train/grad/global_norm_pre_clip",
    "train/parameter_update_rms",
    "train/grad/clip_active", "train/memory/peak_allocated_gib",
    "train/round_warmup/lr_scale", "train/round_warmup/completed_updates",
    "train/lr/scheduled_max", "train/lr/scheduled_min",
    "reference_trust/policy_lr/effective_step_scale",
    "reference_trust/vp_path_kl", "reference_trust/velocity_mse",
    "reference_trust/velocity_mse_ratio", "reference_trust/delta",
    "reference_trust/post_step_distance", "reference_trust/accepted_step_scale",
    "reference_trust/step_accepted",
    "reference_trust/best_decay/count", "reference_trust/best_decay/target",
    "reference_trust/best_decay/feasibility_limited",
    "reference_trust/best_decay/global_best_auc_gap",
    "reference_trust/adam_full_state_reset", "reference_trust/adam_states_cleared",
        "staleness/raw_auc", "staleness/raw_balanced_accuracy",
        "staleness/raw_audit_repeats", "staleness/raw_auc_gap_stdev",
        "staleness/raw_auc_gap_se",
        "staleness/fixed_schedule_diagnostic_raw_audit",
        "staleness/diagnostic_selection_blind",
        "staleness/raw_auc_gap_slope_per_10_steps",
        "staleness/raw_auc_gap_change_from_step0",
    "staleness/fixed_schedule_without_audit",
    "staleness/incumbent_weighted_audit_skipped",
    "staleness/independent_raw_audit_skipped",
    "staleness/raw_best_auc_gap", "staleness/raw_no_improvement_streak",
    "staleness/raw_no_improvement_patience",
    "staleness/patience_paused_for_warmup",
    "staleness/global_best/candidate_gap", "staleness/global_best/incumbent_gap",
    "staleness/global_best/improved", "staleness/global_best/failed_rounds",
    "staleness/global_best/stop_requested", "staleness/global_best/confirmation_delta",
    "staleness/global_best/confirmation_valid", "staleness/global_best/confirmation_accepted",
    "staleness/raw_audit_saturated", "staleness/trigger_recalibration",
    "staleness/age_trigger_recalibration",
    "staleness/reward_round_budget_exhausted",
    "staleness/raw_best_rollback_applied", "staleness/reward_round_id",
    "classifier_trust/balanced_accuracy", "classifier_trust/balanced_accuracy_lower",
    "classifier_trust/balanced_accuracy_upper", "classifier_trust/max_balanced_accuracy",
    "classifier_trust/trigger_recalibration", "classifier_trust/saturated",
    "omnifold/accepted", "omnifold/iterations_fitted",
    "omnifold/iteration_one_only", "omnifold/closure_evaluated",
    "omnifold/iteration1_monitor/raw_auc",
    "omnifold/iteration1_monitor/validation_bce",
    "omnifold/iteration1_monitor/train_bce",
    "omnifold/iteration1_monitor/warm_started_folds",
    "omnifold/iteration1_monitor/train_oof/ess_fraction",
    "omnifold/iteration1_monitor/train_oof/top_1pct_mass",
    "omnifold/iteration1_monitor/train_oof/max_mean_one_weight",
    "omnifold/iteration1_monitor/validation_ensemble/ess_fraction",
    "omnifold/iteration1_monitor/validation_ensemble/top_1pct_mass",
    "omnifold/iteration1_monitor/validation_ensemble/max_mean_one_weight",
    "omnifold/candidate/residual_closure_auc",
    "omnifold/fit/iter01/validation_auc",
    "val_ztautau/jsd/current/topology/cos_opening",
    "val_ztautau/jsd/current/topology/delta_phi_to_pi",
    "val_ztautau/jsd/current/target/tau_a_delta_theta",
    "val_ztautau/jsd/current/target/tau_b_delta_theta",
    "val_ztautau/jsd/current/target/tau_a_delta_phi",
    "val_ztautau/jsd/current/target/tau_b_delta_phi",
    "val_ztautau/topology/cos_opening", "val_ztautau/topology/delta_phi_to_pi",
    "val_tarp/tarp_binned_min_holm_pvalue", "val_tarp/coverage",
})
_WANDB_CLOCK_KEYS = frozenset({"global_step", "epoch", "omnifold_live/log_index"})
_WANDB_EPOCH_PREFIXES = (
    "val/", "val_cheap/", "val_diagnostics/", "val_neutrino/", "val_mass/",
    "val_ztautau/", "val_tarp/", "train_dist/", "train_dist_k1/",
)


def _wandb_critical_keep(key: str) -> bool:
    # Keep fit diagnostics searchable but hidden, not connected into a fake
    # training curve across independent folds/refits. Console progress remains.
    return (
        key.startswith(("endpoint_kl/", "train/visible_conditioning/", "train/lr/scheduled/")) or key in _WANDB_CRITICAL_CHARTS or key in _WANDB_CLOCK_KEYS
        or key in _WANDB_CLASSIFIER_TRAINING_CHARTS
        or key.startswith("classifier_fit/")
        or key.startswith("gradient_conflict/")
        or key.startswith("gradient_direction/")
        or key.startswith("audit/")
        or key.startswith("gradient_transfer/")
        or key.startswith("classifier_only/")
        or key.startswith("parameter_update_rms_calibration/")
        or key.startswith("reward_consensus/")
        or key.startswith("reward_rank_audit/")
        or key.startswith("omnifold/iteration1_monitor/")
        or key.startswith("omnifold/weight_guard/")
        or key.startswith("omnifold/adaptive_tempering/")
        or key.startswith("omnifold/minimum_sufficient/")
        or key.startswith("staleness/raw_audit_repeat_")
        or key.startswith("omnifold_live/meta/")
        or (key.startswith("omnifold_live/") and ("/stability/" in key or "/visible_conditioning/" in key))
        or (
            key.startswith("omnifold_live/")
            and (
                key.rsplit("/", 1)[-1] in {
                    "training_loss", "training_balanced_accuracy",
                    "validation_loss", "validation_auc",
                    "validation_balanced_accuracy", "saturated",
                    "learning_rate", "topology_training_stage",
                    "fit_stage", "selected_stage", "stage_a_validation_loss",
                    "stage_b_validation_loss", "accepted",
                    "threshold_reached", "validation_oriented_balanced_accuracy",
                    "validation_balanced_accuracy_lcb",
                    "validation_balanced_accuracy_lcb_standard_error",
                    "validation_balanced_accuracy_lcb_streak",
                    "train_ess_fraction", "validation_ess_fraction",
                    "applied_tempering", "ess_target_reached",
                    "gradient_norm", "gradient_clipped",
                    "gradient_clip_scale", "gradient_clip_fraction",
                }
                or key.rsplit("/", 1)[-1].startswith(
                    (
                        "gradient_norm_",
                        "parameter_rms_",
                        "gradient_rms_",
                        "gradient_to_parameter_rms_ratio_",
                        "parameter_update_rms_",
                        "update_to_parameter_rms_ratio_",
                        "optimizer_group_lr_",
                        "scheduler_",
                        "logit_",
                    )
                )
            )
        )
        or key in {
            "staleness/global_step", "staleness/epoch",
            "staleness/reward_age_epochs", "staleness/max_reward_age_epochs",
            "staleness/max_reward_rounds", "staleness/reward_round_budget_exhausted",
            "staleness/age_refit_due", "staleness/trigger_reason",
            "staleness/raw_best_global_step", "staleness/raw_best_epoch",
            "staleness/raw_best_rollback_global_step", "classifier_trust/epoch",
            "omnifold/reward_round_id", "omnifold/all_fits_saturated",
            "omnifold/all_fits_ready",
            "omnifold/accept_reason", "omnifold/recalibrations_rejected",
            "omnifold/residual_closure_auc_limit", "omnifold/residual_closure_schedule_enabled",
            "omnifold/refit_global_step",
            "staleness/decision", "staleness/raw_best_rollback_applied",
            "staleness/raw_audit_fit_events", "staleness/raw_audit_test_events",
            "staleness/raw_audit_early_stop_events", "staleness/raw_audit_probe_events",
            "staleness/raw_audit_steps_per_epoch", "staleness/raw_audit_validation_interval_steps",
            "staleness/raw_audit_patience_evaluations", "staleness/raw_audit_uses_omnifold_fold",
            "staleness/raw_audit_training_fold",
            "staleness/raw_classifier_warm_started", "staleness/raw_audit_training_ready",
            "staleness/raw_audit_training_min_steps", "staleness/raw_audit_training_steps",
            "staleness/raw_audit_training_epochs",
            "classifier_trust/reference_trust_fit_events", "classifier_trust/reference_trust_test_events",
        }
        or "nonfinite" in key
        or (key.startswith("val_tarp/") and "/skipped_" in key)
    )


_WANDB_SIMPLIFIED_EXACT_KEYS = frozenset(
    {
        "epoch",
        "global_step",
        "omnifold_live/log_index",
        "train/loss/total",
        "train/loss/dgpo",
        "train/loss/L_cur",
        "train/loss/L_ref",
        "train/loss/delta",
        "train/grad/global_norm_pre_clip",
        "train/grad/clip_active",
        "train/parameter_update_rms",
        "train/memory/peak_allocated_gib",
        "train/memory/peak_reserved_gib",
        "reward/raw/mean",
        "reward/raw/std",
        "reward/monitor/best_of_k",
        "reward/monitor/median",
        "reward/monitor/last_place",
        "reward/monitor/mean_gap",
        "parameter/w_e/mean",
        "parameter/w_e/std",
        "reference_trust/loss",
        "reference_trust/vp_path_kl",
        "reference_trust/objective_vp_path_kl",
        "reference_trust/velocity_mse_ratio",
        "val/reward/mean",
        "val/reward/all_sample_mean",
        "val/reward/best_of_k_mean",
        "val/reward/median",
        "val/reward/p10",
        "val/reward/p90",
        "val/winrate",
        "val_cheap/reward/mean",
        "val_cheap/reward/all_sample_mean",
        "val_cheap/reward/best_of_k_mean",
        "val_cheap/reward/median",
        "val_cheap/reward/p10",
        "val_cheap/reward/p90",
        "omnifold/accepted",
        "omnifold/iterations_fitted",
        "omnifold/all_fits_saturated",
        "omnifold/all_fits_ready",
        "omnifold/bootstrap_on_start",
        "omnifold/resume_refit_once_completed",
        "omnifold/resume_refit_once_accepted",
        "omnifold/resume_refit_kept_incumbent",
        "omnifold/topology_acceptance_audit_enabled",
        "omnifold/topology_acceptance_max_auc_gap",
        "omnifold/topology_acceptance_repeats",
        "omnifold/candidate/audit_balanced_accuracy",
        "omnifold/baseline/audit_observed_auc_gap",
        "omnifold/baseline/probe_events",
        "omnifold/baseline/audit_saturated",
        "omnifold/reward_round_id",
        "omnifold/recalibration_count",
        "omnifold/incumbent_probe_skipped",
        "staleness/weighted_auc_gap",
        "staleness/raw_auc",
        "staleness/raw_auc_gap",
        "staleness/raw_balanced_accuracy",
        "staleness/raw_classifier_warm_started",
        "staleness/raw_audit_training_ready",
        "staleness/raw_audit_training_min_steps",
        "staleness/raw_audit_training_steps",
        "staleness/raw_audit_training_epochs",
        "staleness/raw_audit_fit_events",
        "staleness/raw_audit_early_stop_events",
        "staleness/raw_audit_test_events",
        "staleness/raw_audit_probe_events",
        "staleness/raw_audit_steps_per_epoch",
        "staleness/raw_audit_validation_interval_steps",
        "staleness/raw_audit_patience_evaluations",
        "staleness/raw_audit_uses_omnifold_fold",
        "staleness/raw_audit_training_fold",
        "staleness/raw_audit_saturated",
        "staleness/raw_audit_validation_loss",
        "staleness/raw_audit_validation_auc",
        "staleness/raw_auc_null_se_approx",
        "staleness/raw_auc_gap_z_approx",
        "staleness/raw_auc_gap_pvalue_approx",
        "staleness/raw_audit_repeats",
        "staleness/raw_auc_gap_stdev",
        "staleness/raw_auc_gap_se",
        "staleness/monitor_mode_raw_only",
        "staleness/audit_saturated",
        "staleness/audit_threshold_reached",
        "staleness/trigger_threshold",
        "staleness/trigger_recalibration",
        "staleness/reward_round_id",
        "staleness/fixed_schedule_without_audit",
        "staleness/incumbent_weighted_audit_skipped",
        "staleness/independent_raw_audit_skipped",
        "staleness/fixed_schedule_diagnostic_raw_audit",
        "staleness/fixed_schedule_with_diagnostic_audit",
        "staleness/diagnostic_selection_blind",
        "staleness/raw_audit_trajectory_points",
        "staleness/raw_auc_gap_step0",
        "staleness/raw_auc_gap_change_from_step0",
        "staleness/raw_auc_gap_slope_per_10_steps",
        "staleness/max_reward_rounds",
        "staleness/reward_round_budget_exhausted",
    }
)


def _wandb_simplified_keep(key: str, value: Any) -> bool:
    """Keep only non-redundant metrics used to judge training or acceptance."""

    if key.startswith(("endpoint_kl/", "train/visible_conditioning/", "train/lr/scheduled/")):
        return True

    if (
        key.startswith(
            (
                "omnifold/iteration1_monitor/",
                "frozen_classifier/",
                "omnifold/weight_guard/",
                "omnifold/adaptive_tempering/",
                "omnifold/minimum_sufficient/",
            )
        )
        or key in {"omnifold/iteration_one_only", "omnifold/closure_evaluated"}
    ):
        return True
    if key in _WANDB_SIMPLIFIED_EXACT_KEYS:
        return True
    if key.startswith((
        "Classifier training/",
        "classifier_fit/",
        "gradient_conflict/",
        "gradient_transfer/",
        "classifier_only/",
        "parameter_update_rms_calibration/",
    )):
        return True
    if key.startswith(("reward_consensus/", "reward_rank_audit/")):
        return True
    if key.startswith("staleness/raw_audit_repeat_"):
        return True
    if key.startswith("reference_trust/"):
        return True
    if key.startswith("classifier_trust/"):
        return True
    if "nonfinite" in key:
        return True
    if key in _PROJECTION_WANDB_SCALAR_KEYS or key in _SWD_WANDB_SCALAR_KEYS:
        return True
    if key.startswith(_PROJECTION_CONSTRAINT_WANDB_SCALAR_PREFIX):
        return True
    if key.startswith("omnifold_live/meta/"):
        return key.rsplit("/", 1)[-1] in {
            "fit_step",
            "iteration",
            "crossfit_fold",
            "repeat",
            "audit_repeat",
            "phase_id",
            "dgpo_epoch",
            "global_step",
            "signed_scale",
        }
    if key.startswith("omnifold_live/"):
        if "/stability/" in key or "/visible_conditioning/" in key:
            return True
        metric_name = key.rsplit("/", 1)[-1]
        return metric_name in {
            "fit_stage", "selected_stage", "stage_a_validation_loss",
            "stage_b_validation_loss",
            "validation_auc",
            "validation_balanced_accuracy",
            "learning_rate",
            "topology_training_stage",
            "accepted",
            "saturated",
            "threshold_reached",
            "validation_oriented_balanced_accuracy",
            "validation_balanced_accuracy_lcb",
            "validation_balanced_accuracy_lcb_standard_error",
            "validation_balanced_accuracy_lcb_streak",
            "warm_started",
            "warm_started_folds",
            "train_ess_fraction",
            "validation_ess_fraction",
            "applied_tempering",
            "ess_target_reached",
            "gradient_norm", "gradient_clipped",
            "gradient_clip_scale", "gradient_clip_fraction",
        } or metric_name.startswith(
            (
                "gradient_norm_",
                "parameter_rms_",
                "gradient_rms_",
                "gradient_to_parameter_rms_ratio_",
                "parameter_update_rms_",
                "update_to_parameter_rms_ratio_",
                "optimizer_group_lr_",
                "scheduler_",
                "logit_",
            )
        )
    if key.startswith("omnifold/fit/iter"):
        return key.rsplit("/", 1)[-1] in {
            "saturated",
            "minimum_sufficient",
            "threshold_reached_folds",
            "fit_steps_min",
            "fit_steps_mean",
            "fit_steps_max",
            "stored_in_reward",
            "warm_started_folds",
            "validation_auc",
            "validation_balanced_accuracy",
            "applied_tempering",
            "train_ess_fraction",
            "validation_ess_fraction",
            "ess_target_reached",
        }
    if key.startswith("omnifold/candidate/"):
        suffix = key.rsplit("/", 1)[-1]
        if suffix.startswith("topology_audit_repeat_"):
            return True
        return suffix in {
            "weighted_auc_gap",
            "audit_validation_auc",
            "audit_balanced_accuracy",
            "audit_saturated",
            "topology_audit_auc",
            "topology_audit_auc_gap",
            "topology_audit_balanced_accuracy",
            "topology_audit_saturated",
            "topology_audit_same_architecture",
            "topology_audit_fit_events",
            "topology_audit_test_events",
            "topology_audit_repeats",
            "topology_audit_auc_gap_stdev",
            "topology_audit_auc_gap_se",
        }
    if key.startswith("omnifold/baseline/"):
        return key.rsplit("/", 1)[-1] in {
            "audit_observed_auc_gap",
            "probe_events",
            "audit_saturated",
        }
    if key.startswith("val_diagnostics/profile/"):
        return key.rsplit("/", 1)[-1] in {
            "delta_mean",
            "slope",
            "zero_delta_truth",
        }
    if key.startswith("val_neutrino/jsd/current/"):
        return True
    if key.startswith("val_neutrino/all_metrics/"):
        return True
    if key.startswith("val_neutrino/all/") and key.endswith("_truth_vs_pred"):
        return True
    if key.startswith("val_neutrino/by_process/"):
        return True
    if key.startswith("val/response/"):
        return True
    # Ztautau observables and both TARP arms (full + rank-copula) are primary
    # scientific diagnostics, not generic validation clutter. Preserve their
    # scalars and media in the compact profile.
    if key.startswith("val_ztautau/"):
        return True
    if key.startswith("val_tarp/"):
        return True
    return False


def _wandb_apply_simplified_profile(data: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the compact W&B allowlist when ``logger.wandb.simplified`` is true."""

    if _wandb_critical_enabled():
        return {str(key): value for key, value in data.items() if _wandb_critical_keep(str(key))}
    if not _wandb_simplified_enabled():
        return dict(data)
    return {
        str(key): value
        for key, value in data.items()
        if _wandb_simplified_keep(str(key), value)
    }


_PROJECTION_WANDB_SCALAR_KEYS = frozenset({
    "projection/v_linear",
    "projection/C_adam_pred",
    "projection/lambda",
    "projection/final_update_norm",
    "projection/summary/C_projected_minus_old",
    "projection/multi_sample/C_mean",
})

_SWD_WANDB_SCALAR_KEYS = frozenset({
    "swd/active",
    "swd/pred_truth",
    "swd/truth_truth",
    "swd/ratio",
    "swd/C_norm",
    "swd/mask_count",
    "swd/skipped_small_mask",
})

_PROJECTION_CONSTRAINT_WANDB_SCALAR_PREFIX = "projection/constraint/"


def _wandb_train_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Normalize logged keys: known prefixes pass through; bare keys get ``train/``.

    Passes through ``wandb.Histogram`` / ``wandb.Image`` values under ``reward/`` or ``val/``.
    Known prefixes ``train/``, ``val/``, ``reward/``, ``parameter/``, ``components/``,
    ``diagnostics/``, and gradient diagnostic namespaces are logged as-is.
    ``components/`` is the per-component reward breakdown panel; ``diagnostics/``
    is for monitoring-only reward-hacking checks.
    Keys starting with ``_`` are internal (e.g. ``_kin_h_*`` histogram arrays) and are skipped.
    """
    out: dict[str, Any] = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            continue
        if k.endswith("_hist") and isinstance(v, np.ndarray):
            try:
                import wandb

                out[k] = wandb.Histogram(v)
            except Exception:
                continue
        elif k.startswith("projection/"):
            if k in _PROJECTION_WANDB_SCALAR_KEYS or k.startswith(_PROJECTION_CONSTRAINT_WANDB_SCALAR_PREFIX):
                out[k] = v
        elif k.startswith("swd/"):
            if k in _SWD_WANDB_SCALAR_KEYS:
                out[k] = v
        elif k.startswith((
            "train/",
            "val/",
            "reward/",
            "parameter/",
            "components/",
            "diagnostics/",
            "dgpo/",
            "reference_trust/",
            "gradient_conflict/",
            "gradient_transfer/",
            "endpoint_kl/",
        )):
            out[k] = v
        elif _wandb_is_media_value(v):
            out[k] = v
        else:
            out[f"train/{k}"] = v
    return out


def _dgpo_wandb_yaml_section() -> tuple[Any | None, str]:
    """Resolve W&B settings the same way as ``evenet/train.py`` + ``WandbLogger``.

    Prefer ``logger.wandb`` when it defines a project (standard EveNet YAML). Otherwise
    use the top-level ``wandb:`` block (DGPO configs often keep it there alongside
    ``logger:`` for tensorboard-only fields).

    Returns:
        ``(section_dict, source_label)`` where ``source_label`` is ``\"logger.wandb\"``
        or ``\"wandb\"`` for logging.
    """
    gc = global_config._global_config
    logger = gc.get("logger")
    if isinstance(logger, dict):
        nested = logger.get("wandb")
        if isinstance(nested, dict) and nested.get("project") is not None:
            return nested, "logger.wandb"
    top = gc.get("wandb")
    if isinstance(top, dict) and len(top) > 0:
        return top, "wandb"
    return None, ""


_DEFAULT_DIAGNOSTIC_PLOTS = {
    "pt_profile_accumulated",
    "projection_violation_compare",
}
_DEFAULT_DIAGNOSTIC_BIN_RANGES: dict[str, tuple[float, float]] = {
    "pt": (0.0, 300.0),
    "eta": (-4.0, 4.0),
    "phi": (-3.2, 3.2),
    "px": (-300.0, 300.0),
    "py": (-300.0, 300.0),
    "pz": (-800.0, 800.0),
    "wmass": (50.0, 120.0),
    "topmass": (100.0, 250.0),
}


def _diagnostic_plot_block(dg_cfg: Any) -> Any | None:
    """Return ``dgpo.diagnostic_plots`` when present, else ``None``."""
    return _dgpo_cfg_get(dg_cfg, "diagnostic_plots", None)


def _resolve_diagnostic_num_bins(dg_cfg: Any | None = None) -> int:
    """Resolve W&B diagnostic histogram bin count with legacy-key fallback."""
    cfg = dg_cfg if dg_cfg is not None else getattr(global_config, "dgpo", None)
    block = _diagnostic_plot_block(cfg)
    raw = _dgpo_cfg_get(block, "num_bins", None)
    if raw is None:
        raw = _dgpo_cfg_get(cfg, "diagnostic_num_bins", _VAL_KIN_NUM_BINS)
    return max(2, int(raw))


def _resolve_diagnostic_bin_range(name: str, dg_cfg: Any | None = None) -> tuple[float, float] | None:
    """Resolve per-variable diagnostic plot ranges from config or defaults."""
    cfg = dg_cfg if dg_cfg is not None else getattr(global_config, "dgpo", None)
    block = _diagnostic_plot_block(cfg)
    raw_ranges = _dgpo_cfg_get(block, "bin_ranges", None)
    if raw_ranges is None:
        raw_ranges = _dgpo_cfg_get(cfg, "diagnostic_bin_ranges", None)
    raw = None
    if isinstance(raw_ranges, Mapping):
        raw = raw_ranges.get(name, None)
    if raw is not None:
        try:
            lo, hi = [float(x) for x in raw]
        except (TypeError, ValueError):
            lo = hi = float("nan")
        if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
            return lo, hi
    return _DEFAULT_DIAGNOSTIC_BIN_RANGES.get(name)


def _diagnostic_bin_edges(name: str, dg_cfg: Any | None = None) -> np.ndarray | None:
    """Fixed diagnostic histogram edges for known/configured variables."""
    bounds = _resolve_diagnostic_bin_range(name, dg_cfg=dg_cfg)
    if bounds is None:
        return None
    lo, hi = bounds
    return np.linspace(lo, hi, _resolve_diagnostic_num_bins(dg_cfg=dg_cfg) + 1)


def _resolve_diagnostic_plot_settings(dg_cfg: Any) -> tuple[set[str], int]:
    """Resolve W&B media diagnostics from ``dgpo.diagnostic_plots`` (names, plot_every)."""
    block = _diagnostic_plot_block(dg_cfg)
    if block is None:
        raw_names = _dgpo_cfg_get(dg_cfg, "diagnostic_plot_names", None)
        plot_every = max(1, int(_dgpo_cfg_get(dg_cfg, "diagnostic_plot_every", 1)))
        if raw_names is None:
            return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every
        if isinstance(raw_names, str):
            return {raw_names}, plot_every
        try:
            return {str(name) for name in raw_names}, plot_every
        except TypeError:
            return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every
    enabled = _dgpo_cfg_get(block, "enabled", None)
    if enabled is not None and not bool(enabled):
        return set(), 1
    plot_every = max(1, int(_dgpo_cfg_get(block, "plot_every", 1)))
    raw_names = _dgpo_cfg_get(block, "include", None)
    if raw_names is None:
        return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every
    if isinstance(raw_names, str):
        return {raw_names}, plot_every
    try:
        return {str(name) for name in raw_names}, plot_every
    except TypeError:
        return set(_DEFAULT_DIAGNOSTIC_PLOTS), plot_every


def _resolve_diagnostic_plot_names(dg_cfg: Any) -> set[str]:
    """Resolve opt-in W&B media plot names from ``dgpo.diagnostic_plots``."""
    names, _ = _resolve_diagnostic_plot_settings(dg_cfg)
    return names


def _wandb_sanitize_log_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Drop non-finite floats so one bad scalar does not invalidate the whole W&B step.

    Preserves ``wandb`` media objects (histograms, images) unchanged.
    """
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k.startswith("gradient_conflict/") and k.endswith("/panel_sha256") and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v):
            out[k] = v
            continue
        if _wandb_is_media_value(v):
            out[k] = v
            continue
        if isinstance(v, bool):
            out[k] = float(v)
            continue
        if isinstance(v, int) and not isinstance(v, bool):
            out[k] = v
            continue
        if isinstance(v, float):
            if math.isfinite(v):
                out[k] = v
            continue
        try:
            fv = float(v)
            if math.isfinite(fv):
                out[k] = fv
        except (TypeError, ValueError):
            pass
    return out


def _wandb_reset_step_tracker(*, next_row: int = 0) -> None:
    """Reset transport tracking, separately from checkpoint training counters."""
    global _wandb_committed_step
    _wandb_committed_step = int(next_row) - 1


def _wandb_train_step(global_step: int) -> int:
    """Completed DGPO steps, not W&B's internal row counter."""
    return int(global_step)


def _wandb_epoch_end_step(global_step: int) -> int:
    """Same completed-step count as the epoch-end checkpoint and monitor."""
    return int(global_step)


def _wandb_log_with_step(wandb_mod: Any, payload: dict[str, Any], *, step: int) -> None:
    """One committed row per event; ``step`` is the completed DGPO step count.

    Explicit commit prevents a baseline/monitor row from being merged with the
    next training row and having its epoch overwritten. Scientific coordinates
    travel in every row; the W&B transport index advances independently.
    """
    global _wandb_committed_step
    clean = _wandb_sanitize_log_dict(_wandb_apply_simplified_profile(payload))
    if not clean:
        return
    clean["global_step"] = int(step)
    if "epoch" not in clean:
        for key in ("staleness/epoch", "classifier_trust/epoch", "omnifold_live/meta/dgpo_epoch"):
            if key in payload:
                clean["epoch"] = int(payload[key])
                break
    s = _wandb_committed_step + 1
    try:
        wandb_mod.log(clean, step=s, commit=True)
        _wandb_committed_step = s
    except Exception as e:
        _log.warning("[DGPO] wandb.log failed at step=%s: %s", s, e)


def _wandb_log_step(wandb_mod: Any, payload: dict[str, Any], *, step: int) -> None:
    """Log a training/controller event at its completed DGPO step."""
    _wandb_log_with_step(wandb_mod, payload, step=_wandb_train_step(step))


def _wandb_log_auxiliary(
    wandb_mod: Any,
    payload: dict[str, Any],
    *,
    current_global_step: int,
) -> None:
    """A fit row advances transport/log_index, never the DGPO clock."""
    _wandb_log_with_step(wandb_mod, payload, step=current_global_step)


def _wandb_log_validation(
    wandb_mod: Any,
    val_metrics: dict[str, Any],
    *,
    epoch: int,
    wandb_step: int,
) -> None:
    """Log validation at explicit epoch and completed DGPO step coordinates."""
    clean = _wandb_sanitize_log_dict(dict(val_metrics))
    if not clean:
        return
    clean["epoch"] = float(epoch)
    _wandb_log_with_step(wandb_mod, clean, step=wandb_step)


class _ClassifierFitLossTracker:
    """Independent loss curves on local optimizer steps, never DGPO/log steps.

    Fits execute serially. A fold/refit/phase change or restarted fit counter
    starts a distinct series; only the active series is kept in Python memory.
    """

    def __init__(self) -> None:
        self.fit_id = 0
        self.key: tuple[Any, ...] | None = None
        self.last_step = -1
        self.prefix = ""

    def payload(self, wandb_mod: Any, phase: str, row: Mapping[str, Any], *,
                global_step: int, epoch: int) -> dict[str, Any]:
        step = int(float(row.get("step", 0)))
        if step < 1 or "training_loss" not in row:
            return {}
        iteration, fold, repeat = (int(float(row.get(k, default)))
                                   for k, default in (("iteration", 1), ("fold", 0), ("repeat", 1)))
        stage = int(float(row.get("fit_stage", 0)))
        key = (phase, global_step, epoch, iteration, fold, repeat, stage)
        if key != self.key or step <= self.last_step:
            self.fit_id += 1
            self.key = key
            self.prefix = (f"classifier_fit/{phase}/fit{self.fit_id:05d}"
                           f"_g{global_step:06d}_i{iteration:02d}_f{fold:02d}_r{repeat:02d}")
            if stage:
                self.prefix += f"_stage{stage}"
            axis = f"{self.prefix}/step"
            wandb_mod.define_metric(axis, hidden=True)
            for metric in (
                "training_loss",
                "training_balanced_accuracy",
                "validation_loss",
                "validation_balanced_accuracy",
                "validation_auc",
                "learning_rate",
                "topology_training_stage",
            ):
                wandb_mod.define_metric(
                    f"{self.prefix}/{metric}",
                    step_metric=axis,
                    step_sync=False,
                    hidden=False,
                )
        self.last_step = step
        payload = {
            f"{self.prefix}/step": step,
            f"{self.prefix}/training_loss": row["training_loss"],
        }
        if "training_balanced_accuracy" in row:
            payload[f"{self.prefix}/training_balanced_accuracy"] = row[
                "training_balanced_accuracy"
            ]
        for metric in ("learning_rate", "topology_training_stage"):
            if metric in row:
                payload[f"{self.prefix}/{metric}"] = row[metric]
        # The live callback otherwise repeats the last validation result
        # between evaluations; do not place stale values at a new fit step.
        if row.get("validation_evaluated", False):
            for metric in (
                "validation_loss",
                "validation_balanced_accuracy",
                "validation_auc",
            ):
                if metric in row:
                    payload[f"{self.prefix}/{metric}"] = row[metric]
        return payload


class _ClassifierTrainingPlotTracker:
    """Build compact W&B charts from repeat 1 / fold 1 of every iteration.

    The plotted x values are the exact local optimizer steps reported by the
    classifier callback. Other cross-fit and ensemble members remain available
    in ``omnifold_live`` history, but do not create W&B panels. A fixed chart key
    is reused at every refit; the chart title identifies the current DGPO step.
    """

    _PHASE_ROLES = {
        "residual_reward": "Reward",
        "raw_staleness_audit": "Fresh audit",
    }
    _METRICS = (
        ("training_loss", "Train loss", False),
        ("validation_loss", "Validation loss", True),
        ("validation_auc", "Validation AUC", True),
        (
            "validation_balanced_accuracy",
            "Validation accuracy",
            True,
        ),
        ("learning_rate", "Base learning rate", False),
        ("gradient_norm", "Pre-clip gradient norm", False),
        ("gradient_clip_scale", "Gradient clip scale", False),
        ("gradient_clip_fraction", "Gradient clip fraction", False),
        ("stability/probe/bce_before", "Fixed probe BCE before update", False),
        ("stability/probe/bce_after", "Fixed probe BCE after update", False),
        ("stability/probe/bce_delta", "Fixed probe BCE change", False),
        ("stability/probe/separation_before", "Fixed probe separation before update", False),
        ("stability/probe/separation_after", "Fixed probe separation after update", False),
        ("stability/probe/logit_change_rms", "Fixed probe logit change RMS", False),
        ("stability/local_gradient_norm_rankmax", "Largest rank-local gradient norm", False),
        ("stability/gradient_spike", "Gradient spike trigger", False),
        *(
            (f"stability/representation_probe/{branch}/holdout_auc",
             f"Fixed representation probe: {branch} holdout AUC", False)
            for branch in ("raw_fourier", "normalized_fourier", "fourier", "decoder", "concat", "fusion", "current_head")
        ),
        *(
            (f"stability/representation_probe/{branch}/cv/holdout_auc",
             f"CV representation probe: {branch} holdout AUC", False)
            for branch in ("raw_fourier", "normalized_fourier", "fourier", "decoder", "concat", "fusion")
        ),
        *(
            (f"stability/layer/representation/{branch}/{metric}/rankmean",
             f"{branch} {metric}", False)
            for branch in ("fourier", "decoder", "fusion")
            for metric in ("activation_rms", "gradient_rms")
        ),
        *(
            (f"stability/layer/representation/bank.decoder.blocks.0.modulation/gate_{branch}_rms/rankmean",
             f"Decoder {branch} gate RMS", False)
            for branch in ("self", "cross", "ffn")
        ),
        *(
            (f"{metric}_{group}", f"{label} {description}", False)
            for group, label in (
                ("adapter", "Adapter"), ("decoder", "Decoder"),
                ("backbone_other", "Backbone"),
                ("topology_context", "Fourier context"),
                ("context_output_head", "Output head"),
            )
            for metric, description in (
                ("parameter_update_rms", "AdamW update RMS"),
                ("update_to_parameter_rms_ratio", "relative AdamW update RMS"),
            )
        ),
        (
            "gradient_norm_direct_topology_head",
            "Direct topology head gradient norm",
            False,
        ),
        (
            "gradient_norm_topology_context",
            "Topology context gradient norm",
            False,
        ),
        (
            "gradient_norm_context_output_head",
            "Context output head gradient norm",
            False,
        ),
        ("gradient_norm_decoder", "Decoder gradient norm", False),
        ("gradient_norm_adapter", "Adapter gradient norm", False),
        (
            "gradient_norm_input_projector",
            "Input projector gradient norm",
            False,
        ),
        ("logit_mean_positive", "Truth logit mean", False),
        ("logit_mean_negative", "Generated logit mean", False),
        ("logit_class_mean_separation", "Logit class separation", False),
        ("logit_rms", "Logit RMS", False),
        ("logit_abs_gt_5_fraction", "Saturated-logit fraction", False),
        (
            "parameter_rms_direct_topology_head",
            "Direct head parameter RMS",
            False,
        ),
        (
            "gradient_rms_direct_topology_head",
            "Direct head gradient RMS",
            False,
        ),
        (
            "gradient_to_parameter_rms_ratio_direct_topology_head",
            "Direct head relative gradient RMS",
            False,
        ),
        (
            "parameter_rms_topology_context",
            "Topology context parameter RMS",
            False,
        ),
        (
            "gradient_rms_topology_context",
            "Topology context gradient RMS",
            False,
        ),
        (
            "gradient_to_parameter_rms_ratio_topology_context",
            "Topology context relative gradient RMS",
            False,
        ),
    )

    def __init__(self) -> None:
        self._states: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _new_state(*, global_step: int, epoch: int) -> dict[str, Any]:
        return {
            "cycle": (int(global_step), int(epoch)),
            "iterations": {},
        }

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def payload(
        self,
        wandb_mod: Any,
        phase: str,
        row: Mapping[str, Any],
        *,
        global_step: int,
        epoch: int,
    ) -> dict[str, Any]:
        role = self._PHASE_ROLES.get(str(phase))
        step = int(float(row.get("step", 0)))
        if role is None or step < 1 or "training_loss" not in row:
            return {}

        cycle = (int(global_step), int(epoch))
        state = self._states.get(phase)
        if state is None or state["cycle"] != cycle:
            state = self._new_state(global_step=global_step, epoch=epoch)
            self._states[phase] = state

        iteration = int(float(row.get("iteration", 1)))
        repeat = int(float(row.get("repeat", 1)))
        fold = int(float(row.get("fold", 1)))
        if repeat != 1 or fold != 1:
            return {}

        iteration_state = state["iterations"].setdefault(
            iteration,
            {
                "stage": int(float(row.get("fit_stage", 0))),
                "last_raw_step": 0,
                "metrics": {metric: [] for metric, _label, _validation in self._METRICS},
            },
        )
        stage = int(float(row.get("fit_stage", 0)))
        if stage != iteration_state["stage"]:
            iteration_state["stage"] = stage
            iteration_state["last_raw_step"] = 0
        elif step <= int(iteration_state["last_raw_step"]):
            # Ignore duplicate/out-of-order callbacks within the same fit
            # stage; a stage change is handled above with its reported step.
            return {}

        # Keep the exact step published by this classifier fit. W&B transport
        # rows, DGPO steps, other members, and stage counters never alter x.
        plot_step = step
        iteration_state["last_raw_step"] = step
        validation_evaluated = bool(float(row.get("validation_evaluated", 0.0)))
        for metric, _label, validation_only in self._METRICS:
            if validation_only and not validation_evaluated:
                continue
            value = self._finite_float(row.get(metric))
            if value is not None:
                iteration_state["metrics"][metric].append((plot_step, value))

        should_publish = (
            validation_evaluated
            or bool(float(row.get("saturated", 0.0)))
            or bool(float(row.get("threshold_reached", 0.0)))
        )
        if not should_publish:
            return {}

        payload: dict[str, Any] = {}
        for metric, label, _validation_only in self._METRICS:
            xs: list[list[int]] = []
            ys: list[list[float]] = []
            keys: list[str] = []
            for iteration_id in sorted(state["iterations"]):
                points = state["iterations"][iteration_id]["metrics"][metric]
                if not points:
                    continue
                xs.append([point[0] for point in points])
                ys.append([point[1] for point in points])
                keys.append(f"Iteration {iteration_id}")
            if not xs:
                continue
            payload[f"Classifier training/{role}/{label}"] = (
                wandb_mod.plot.line_series(
                    xs,
                    ys,
                    keys=keys,
                    title=f"{role} classifier · {label} · DGPO step {global_step}",
                    xname="Classifier step",
                    split_table=True,
                )
            )
        return payload


def _wandb_define_axes(wandb_mod: Any, *, critical: bool) -> None:
    """Never fall back to the classifier-inflated internal W&B Step axis."""
    wandb_mod.define_metric("*", step_metric="global_step", step_sync=False, hidden=critical)
    for key in _WANDB_CLOCK_KEYS:
        wandb_mod.define_metric(key, hidden=True)
    for prefix in _WANDB_EPOCH_PREFIXES:
        wandb_mod.define_metric(prefix + "*", step_metric="epoch", step_sync=False, hidden=critical)
    # Raw per-member progress stays queryable but does not create hundreds of
    # misleading connected panels.  Compact representative charts below are
    # the visible classifier-training view.
    wandb_mod.define_metric("omnifold_live/*", step_metric="omnifold_live/log_index",
                          step_sync=False, hidden=True)
    wandb_mod.define_metric("train/visible_conditioning/*", step_metric="global_step",
                          step_sync=False, hidden=False)
    wandb_mod.define_metric("train/lr/scheduled/*", step_metric="global_step",
                          step_sync=False, hidden=False)
    for key in sorted(_WANDB_CLASSIFIER_TRAINING_CHARTS):
        wandb_mod.define_metric(key, hidden=False)
    wandb_mod.define_metric("gradient_conflict/*", step_metric="global_step", step_sync=False, hidden=True)
    wandb_mod.define_metric("gradient_direction/*", step_metric="global_step", step_sync=False, hidden=True)
    wandb_mod.define_metric("audit/*", step_metric="global_step", step_sync=False, hidden=True)
    for key in (
        "installed_vs_fresh/cosine",
        "installed_vs_fresh/cross_dot_lcb",
        "installed_vs_fresh/cross_dot_ucb",
        "installed_vs_fresh/conflict",
        "installed_vs_fresh/alignment",
        "installed/reliable",
        "fresh/reliable",
    ):
        wandb_mod.define_metric(
            "gradient_direction/" + key,
            step_metric="global_step",
            step_sync=False,
            hidden=False,
        )
    for key in (
        "raw_auc",
        "raw_auc_gap",
        "saturated",
        "training_ready",
        "raw_auc_gap_change_from_step0",
        "raw_auc_gap_slope_per_10_steps",
    ):
        wandb_mod.define_metric(
            "audit/" + key,
            step_metric="global_step",
            step_sync=False,
            hidden=False,
        )
    wandb_mod.define_metric(
        "gradient_transfer/*",
        step_metric="global_step",
        step_sync=False,
        hidden=True,
    )
    for key in ("omnifold_staleness/cosine", "omnifold_staleness/conflict",
                "omnifold/norm", "staleness/norm", "trust/norm",
                "omnifold_staleness/conclusive", "omnifold_trust/cosine",
                "omnifold/split_cosine", "staleness/split_cosine", "ran",
                "total_on_omnifold/projection_ratio", "total_on_staleness/projection_ratio"):
        wandb_mod.define_metric("gradient_conflict/" + key, step_metric="global_step", step_sync=False, hidden=False)
    for phase in ("pre_refit", "post_install", "post_warmup"):
        for key in ("omnifold_trust/cosine", "trust/norm", "total_on_omnifold/projection_ratio"):
            wandb_mod.define_metric(f"gradient_conflict/{phase}/{key}", step_metric="global_step", step_sync=False, hidden=False)
    wandb_mod.define_metric("frozen_classifier/*", step_metric="global_step", step_sync=False, hidden=False)
    for key in (
        "ran",
        "h4/norm",
        "reference_unweighted/norm",
        "h4_reference/cosine",
        "total_on_h4/projection_ratio",
        "critical_lambda/value",
        "adamw_descent_on_h4/cosine",
        "adamw_descent_on_total/cosine",
        "reconstruction_actual/relative_error",
    ):
        wandb_mod.define_metric(
            "gradient_transfer/" + key,
            step_metric="global_step",
            step_sync=False,
            hidden=False,
        )
    # Exact definitions override the hidden default; no auto-plots for control
    # metadata or unrelated diagnostics. Users can still query hidden fit rows.
    if critical:
        for key in sorted(_WANDB_CRITICAL_CHARTS):
            if key.startswith("omnifold_live/"):
                axis = "omnifold_live/log_index"
            elif key.startswith(_WANDB_EPOCH_PREFIXES):
                axis = "epoch"
            else:
                axis = "global_step"
            wandb_mod.define_metric(key, step_metric=axis, step_sync=False, hidden=False)


def _start_wandb_run(*, disable: bool = False) -> bool:
    """Initialize wandb like Lightning's ``WandbLogger`` in ``evenet/train.py``.

    Uses ``logger.wandb`` when present, else top-level ``wandb``. Applies
    ``Settings(start_method=\"thread\")`` for compatibility with Ray Train workers
    (forked subprocesses + threads do not mix well with the default ``fork`` method).
    """
    if disable:
        return False
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        _log.info("[DGPO] WANDB_DISABLED set; skipping wandb.")
        return False
    try:
        import wandb
    except ImportError:
        _log.warning("[DGPO] wandb not installed; skipping experiment logging.")
        return False
    wb, wb_source = _dgpo_wandb_yaml_section()
    if wb is None:
        _log.info(
            "[DGPO] No wandb settings (add top-level ``wandb:`` or ``logger.wandb:``); "
            "skipping wandb.init."
        )
        return False
    project = wb.get("project")
    if project is None:
        _log.warning("[DGPO] wandb section missing ``project``; skipping wandb.init.")
        return False
    run_name = wb.get("run_name") or wb.get("name")
    tags = wb.get("tags") or []
    if not isinstance(tags, list):
        tags = list(tags)
    run_id = wb.get("id")
    resume = wb.get("resume")
    if bool(wb.get("fresh_run", False)):
        # Training state can resume while logging starts in a new run. An
        # explicit fresh ID also overrides inherited WANDB_RUN_ID settings.
        from uuid import uuid4
        run_id = uuid4().hex[:8]
        resume = "never"
    init_kw: dict[str, Any] = {
        "project": str(project),
        "entity": wb.get("entity"),
        "name": run_name,
        "group": wb.get("group"),
        "tags": tags,
        "id": run_id,
        "config": global_config.to_logger(),
    }
    if resume is not None:
        init_kw["resume"] = resume
    try:
        init_kw["settings"] = wandb.Settings(start_method="thread")
    except Exception:
        pass
    wandb.init(**init_kw)
    _log.info(
        "[DGPO] wandb.init project=%s name=%s (config_key=%s)",
        project,
        run_name,
        wb_source or "unknown",
    )
    try:
        _wandb_define_axes(wandb, critical=_wandb_critical_enabled())
    except Exception as e:
        _log.warning("[DGPO] wandb.define_metric(val/*) failed (val may share step with train): %s", e)
    _wandb_reset_step_tracker(next_row=int(wandb.run.step))
    _dgpo_wandb_publish_metric_docs()
    return True


def _finish_wandb_run(active: bool) -> None:
    if not active:
        return
    try:
        import wandb

        wandb.finish()
    except Exception as e:
        _log.warning("[DGPO] wandb.finish() failed: %s", e)


def _dgpo_constraint_checkpoint_payload(
    constraint_state: ProjectionConstraintState,
) -> dict[str, Any] | None:
    """Serializable projection-constraint state for Lightning-compatible DGPO checkpoints.

    The frozen latent-SWD state serializes only its checkpoint/normalization provenance.
    """
    if constraint_state is None:
        return None
    return constraint_state.checkpoint_payload()


def _dgpo_save_last_ckpt(
    model: torch.nn.Module,
    ema_save: Any | None,
    optimizer: torch.optim.Optimizer,
    ref_model: torch.nn.Module,
    *,
    last_completed_epoch: int,
    dgpo_next_epoch: int,
    global_step: int,
    dgpo_epoch_step: int = 0,
    ema_rollout: Any | None = None,
    round_ref_model: torch.nn.Module | None = None,
    reward_round_id: int = 0,
    dgpo_projection_constraint_state: dict[str, Any] | None = None,
    dgpo_omnifold_reward_metadata: dict[str, Any] | None = None,
    dgpo_adaptive_omnifold_state: dict[str, Any] | None = None,
    dgpo_omnifold_reward_stack: dict[str, Any] | None = None,
) -> None:
    """Write an unpruned snapshot and repoint ``last.ckpt`` to it.

    Refuses to save when any trainable parameter is non-finite, so a single bad
    batch cannot poison the resume state of a long-running DGPO job.
    """
    save_dir = global_config.options.Training.get("model_checkpoint_save_path", None)
    if not save_dir:
        _log.debug("[DGPO] model_checkpoint_save_path unset; skipping checkpoint save.")
        return
    core_for_check = _unwrap_core_evenet(model)
    bad = [n for n, p in core_for_check.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        _log.warning(
            "[DGPO] checkpoint save SKIPPED at epoch=%s step=%s: %s non-finite "
            "trainable params (first: %s). Existing snapshots are preserved.",
            last_completed_epoch, global_step, len(bad), bad[:3],
        )
        return
    save_root = Path(str(save_dir)).expanduser().resolve()
    path = save_root / dgpo_snapshot_checkpoint_name(
        last_completed_epoch=last_completed_epoch,
        dgpo_next_epoch=dgpo_next_epoch,
        global_step=global_step,
    )
    save_lightning_compatible_checkpoint(
        path,
        model,
        ema_save,
        global_config,
        last_completed_epoch=last_completed_epoch,
        dgpo_next_epoch=dgpo_next_epoch,
        global_step=global_step,
        optimizer=optimizer,
        dgpo_epoch_step=dgpo_epoch_step,
        ref_model=ref_model,
        round_ref_model=round_ref_model,
        reward_round_id=reward_round_id,
        ema_rollout=ema_rollout,
        dgpo_projection_constraint_state=dgpo_projection_constraint_state,
        dgpo_omnifold_reward_metadata=dgpo_omnifold_reward_metadata,
        dgpo_adaptive_omnifold_state=dgpo_adaptive_omnifold_state,
        dgpo_omnifold_reward_stack=dgpo_omnifold_reward_stack,
    )
    update_last_checkpoint_pointer(path)


class _DgpoCheckpointTopK:
    """Keep top-k checkpoints for one configured scalar metric."""

    def __init__(
        self,
        save_dir: Path,
        top_k: int,
        *,
        metric_name: str,
        mode: str,
    ) -> None:
        self._save_dir = save_dir
        self._top_k = max(0, int(top_k))
        self._metric_name = str(metric_name)
        self._metric_slug = self._metric_name.replace("/", "_")
        self._mode = str(mode).lower()
        if self._mode not in {"min", "max"}:
            raise ValueError("model_checkpoint_top_k_mode must be 'min' or 'max'")
        # Heap root is always the worst retained item. For min-mode, negate the
        # raw score so the largest accuracy becomes the smallest priority.
        self._worst_heap: list[tuple[float, str, float]] = []

    def maybe_save(
        self,
        *,
        score: float,
        last_completed_epoch: int,
        dgpo_next_epoch: int,
        global_step: int,
        model: torch.nn.Module,
        ema_save: Any | None,
        optimizer: torch.optim.Optimizer,
        ref_model: torch.nn.Module,
        ema_rollout: Any | None = None,
        round_ref_model: torch.nn.Module | None = None,
        reward_round_id: int = 0,
        dgpo_projection_constraint_state: dict[str, Any] | None = None,
        dgpo_omnifold_reward_metadata: dict[str, Any] | None = None,
        dgpo_adaptive_omnifold_state: dict[str, Any] | None = None,
        dgpo_omnifold_reward_stack: dict[str, Any] | None = None,
    ) -> None:
        if self._top_k <= 0:
            return
        raw_score = float(score)
        if not math.isfinite(raw_score):
            _log.warning(
                "[DGPO] %s is non-finite; skipping top-k checkpoint.",
                self._metric_name,
            )
            return

        self._save_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"dgpo-top-{self._metric_slug}={raw_score:.6f}-"
            f"next_ep={dgpo_next_epoch}-step={global_step}.ckpt"
        )
        path = self._save_dir / fname
        save_lightning_compatible_checkpoint(
            path,
            model,
            ema_save,
            global_config,
            last_completed_epoch=last_completed_epoch,
            dgpo_next_epoch=dgpo_next_epoch,
            global_step=global_step,
            optimizer=optimizer,
            ref_model=ref_model,
            round_ref_model=round_ref_model,
            reward_round_id=reward_round_id,
            ema_rollout=ema_rollout,
            dgpo_projection_constraint_state=dgpo_projection_constraint_state,
            dgpo_omnifold_reward_metadata=dgpo_omnifold_reward_metadata,
            dgpo_adaptive_omnifold_state=dgpo_adaptive_omnifold_state,
            dgpo_omnifold_reward_stack=dgpo_omnifold_reward_stack,
        )

        priority = raw_score if self._mode == "max" else -raw_score
        heapq.heappush(self._worst_heap, (priority, str(path), raw_score))

        # Highest priority is the best item for either mode.
        _best_priority, best_path_str, best_score = max(
            self._worst_heap,
            key=lambda x: x[0],
        )
        best_link = self._save_dir / "best.ckpt"
        try:
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(Path(best_path_str).name)
            _log.info(
                "[DGPO] best.ckpt → %s (%s=%.6f, mode=%s)",
                Path(best_path_str).name,
                self._metric_name,
                best_score,
                self._mode,
            )
        except OSError as e:
            _log.warning("[DGPO] Could not update best.ckpt symlink: %s", e)

        while len(self._worst_heap) > self._top_k:
            _worst_priority, worst_path, worst_score = heapq.heappop(
                self._worst_heap
            )
            wp = Path(worst_path)
            if wp.is_file():
                try:
                    wp.unlink()
                    _log.info(
                        "[DGPO] Removed checkpoint outside top-%s: %s (%s=%.6f)",
                        self._top_k,
                        wp.name,
                        self._metric_name,
                        worst_score,
                    )
                except OSError as e:
                    _log.warning("[DGPO] Failed to remove old checkpoint %s: %s", wp, e)


_VAL_KIN_NUM_BINS = 50


@torch.no_grad()
def _val_pred_truth_kin_flat(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``pt`` (GeV, original scale), ``eta``, ``phi`` for selected-candidate pred vs truth (same slots).

    The first invisible feature is ``log1p(pT)``; this function inverts that via ``expm1`` so the
    returned ``ppt`` / ``tpt`` arrays are in GeV (original physics scale), not log space.
    """
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    b_idx = torch.arange(B, device=device)
    pred = candidates[k_sel, b_idx]
    if cartesian:
        t = batch_d["x_invisible_cartesian"]
        plp, pe, pp = cartesian_to_log_pt_eta_phi(pred[..., 0], pred[..., 1], pred[..., 2])
        tlp, te, tp = cartesian_to_log_pt_eta_phi(t[..., 0], t[..., 1], t[..., 2])
    else:
        t = batch_d["x_invisible"]
        plp, pe, pp = pred[..., 0], pred[..., 1], pred[..., 2]
        tlp, te, tp = t[..., 0], t[..., 1], t[..., 2]
    m = (mask > 0).reshape(B, N_nu)
    # Invert log1p to recover pT in GeV (original physics scale).
    ppt = np.expm1(plp[m].detach().float().cpu().numpy())
    pe = pe[m].detach().float().cpu().numpy()
    pp = pp[m].detach().float().cpu().numpy()
    tpt = np.expm1(tlp[m].detach().float().cpu().numpy())
    te = te[m].detach().float().cpu().numpy()
    tp = tp[m].detach().float().cpu().numpy()
    return ppt, pe, pp, tpt, te, tp


@torch.no_grad()
def _val_pred_truth_kin_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    cartesian: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``pt``/``eta``/``phi`` for all candidates vs repeated truth slots."""
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    mask = (mask > 0).reshape(B, N_nu)
    mask_k = mask.unsqueeze(0).expand(K, -1, -1)
    if cartesian:
        truth = batch_d["x_invisible_cartesian"]
        plp, peta, pphi = cartesian_to_log_pt_eta_phi(
            candidates[..., 0], candidates[..., 1], candidates[..., 2]
        )
        tlp, teta, tphi = cartesian_to_log_pt_eta_phi(
            truth[..., 0], truth[..., 1], truth[..., 2]
        )
    else:
        truth = batch_d["x_invisible"]
        plp, peta, pphi = candidates[..., 0], candidates[..., 1], candidates[..., 2]
        tlp, teta, tphi = truth[..., 0], truth[..., 1], truth[..., 2]
    tlp = tlp.unsqueeze(0).expand(K, -1, -1)
    teta = teta.unsqueeze(0).expand(K, -1, -1)
    tphi = tphi.unsqueeze(0).expand(K, -1, -1)
    ppt = np.expm1(plp[mask_k].detach().float().cpu().numpy())
    peta = peta[mask_k].detach().float().cpu().numpy()
    pphi = pphi[mask_k].detach().float().cpu().numpy()
    tpt = np.expm1(tlp[mask_k].detach().float().cpu().numpy())
    teta = teta[mask_k].detach().float().cpu().numpy()
    tphi = tphi[mask_k].detach().float().cpu().numpy()
    return ppt, peta, pphi, tpt, teta, tphi


@torch.no_grad()
def _val_pred_truth_feature_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    feature_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Masked flattened feature arrays for all candidates vs repeated truth slots."""
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    max_features = min(len(feature_names), int(candidates.shape[-1]), int(batch_d["x_invisible"].shape[-1]))
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device, dtype=candidates.dtype)
    else:
        mask = xm.to(device=device, dtype=candidates.dtype)
    mask = (mask > 0).reshape(B, N_nu)
    mask_k = mask.unsqueeze(0).expand(K, -1, -1)
    truth = batch_d["x_invisible"]
    out: dict[str, np.ndarray] = {}
    for i in range(max_features):
        feature_name = str(feature_names[i])
        pred = candidates[..., i][mask_k].detach().float().cpu().numpy()
        target = truth[..., i].unsqueeze(0).expand(K, -1, -1)[mask_k].detach().float().cpu().numpy()
        out[f"{feature_name}_pred"] = pred
        out[f"{feature_name}_truth"] = target
    return out


@torch.no_grad()
def _val_class_index_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    device: torch.device,
) -> np.ndarray:
    """Class ID aligned with the masked ``K x B x N`` feature flattening."""
    class_index = batch_d.get("classification")
    if not isinstance(class_index, Tensor):
        return np.array([], dtype=np.int64)
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    if int(class_index.shape[0]) != B or int(class_index.numel()) != B:
        raise ValueError(
            "classification must contain one integer class ID per event; "
            f"got shape={tuple(class_index.shape)} for batch={B}"
        )
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device)
    else:
        mask = xm.to(device=device)
    mask_k = (mask > 0).reshape(B, N_nu).unsqueeze(0).expand(K, -1, -1)
    class_k = (
        class_index.reshape(B)
        .to(device=device, dtype=torch.long)
        .reshape(1, B, 1)
        .expand(K, B, N_nu)
    )
    return class_k[mask_k].detach().cpu().numpy().astype(np.int64, copy=False)


@torch.no_grad()
def _truth_invisible_kin_phys(
    batch_d: dict[str, Any],
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Truth invisible features in physical units, shape ``(B, N_nu, 3)``.

    The batch stores physical (log-space) targets — ``x_invisible`` is ``[log1p(pT), η, φ]``
    and ``x_invisible_cartesian`` is ``[px, py, pz]`` — exactly the space the DDIM candidates
    are denormalized into. The model normalizes internally during forward, so NO extra
    denormalization is applied here (doing so would double-process and corrupt the values).
    """
    key = "x_invisible_cartesian" if cartesian else "x_invisible"
    raw = batch_d[key].to(device=device, dtype=dtype)
    return raw[..., :3]


@torch.no_grad()
def _kin_to_xyz(kin: Tensor, *, cartesian: bool) -> Tensor:
    """``(B, N, 3)`` kinematics → Cartesian momentum ``(B, N, 3)`` in GeV."""
    if cartesian:
        return kin[..., :3]
    return log_pt_eta_phi_to_cartesian(
        kin[..., 0].clamp(-10.0, 10.0),
        kin[..., 1],
        kin[..., 2],
    )


@torch.no_grad()
def _val_pred_truth_cartesian_flat(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``px``, ``py``, ``pz`` [GeV] for selected-candidate pred vs truth."""
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device) > 0
    else:
        mask = xm.to(device=device) > 0
    mask = mask.reshape(B, N_nu)
    b_idx = torch.arange(B, device=device)
    pred_kin = candidates[k_sel, b_idx][:, :N_nu, :3]
    truth_kin = _truth_invisible_kin_phys(
        batch_d, cartesian=cartesian, device=device, dtype=dtype
    )[:, :N_nu, :]
    pred_xyz = _kin_to_xyz(pred_kin, cartesian=cartesian)
    truth_xyz = _kin_to_xyz(truth_kin, cartesian=cartesian)
    m = mask
    ppx = pred_xyz[..., 0][m].detach().float().cpu().numpy()
    ppy = pred_xyz[..., 1][m].detach().float().cpu().numpy()
    ppz = pred_xyz[..., 2][m].detach().float().cpu().numpy()
    tpx = truth_xyz[..., 0][m].detach().float().cpu().numpy()
    tpy = truth_xyz[..., 1][m].detach().float().cpu().numpy()
    tpz = truth_xyz[..., 2][m].detach().float().cpu().numpy()
    return ppx, ppy, ppz, tpx, tpy, tpz


@torch.no_grad()
def _val_pred_truth_cartesian_flat_all_candidates(
    candidates: Tensor,
    batch_d: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Masked flattened ``px``, ``py``, ``pz`` [GeV] for all candidates vs repeated truth."""
    K = int(candidates.shape[0])
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        mask = xm.squeeze(-1).to(device=device) > 0
    else:
        mask = xm.to(device=device) > 0
    mask = mask.reshape(B, N_nu)
    mask_k = mask.unsqueeze(0).expand(K, -1, -1)
    pred_xyz = candidates[..., :3]
    truth_xyz = _truth_invisible_kin_phys(
        batch_d, cartesian=True, device=device, dtype=dtype
    )[:, :N_nu, :].unsqueeze(0).expand(K, -1, -1, -1)
    ppx = pred_xyz[..., 0][mask_k].detach().float().cpu().numpy()
    ppy = pred_xyz[..., 1][mask_k].detach().float().cpu().numpy()
    ppz = pred_xyz[..., 2][mask_k].detach().float().cpu().numpy()
    tpx = truth_xyz[..., 0][mask_k].detach().float().cpu().numpy()
    tpy = truth_xyz[..., 1][mask_k].detach().float().cpu().numpy()
    tpz = truth_xyz[..., 2][mask_k].detach().float().cpu().numpy()
    return ppx, ppy, ppz, tpx, tpy, tpz


def _pc_row_to_4vec_torch(pc_row: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """First four point-cloud features: ``logE, logPt, η, φ`` → ``(E, px, py, pz)`` [GeV]."""
    row = pc_row[..., :4]
    log_e, log_pt, eta, phi = row.unbind(dim=-1)
    pt = torch.expm1(log_pt.clamp(-10.0, 10.0))
    e = torch.expm1(log_e.clamp(-10.0, 10.0))
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return e, px, py, pz


def _nu_kin_to_4vec_torch(kin: Tensor, *, cartesian: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Neutrino slot kinematics → massless four-vector components [GeV]."""
    if cartesian:
        log_pt, eta, phi = cartesian_to_log_pt_eta_phi(kin[..., 0], kin[..., 1], kin[..., 2])
    else:
        log_pt, eta, phi = kin[..., 0], kin[..., 1], kin[..., 2]
    pt = torch.expm1(log_pt.clamp(-10.0, 10.0))
    e = pt * torch.cosh(eta)
    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    return e, px, py, pz


def _add_4vec_torch(
    a: tuple[Tensor, Tensor, Tensor, Tensor],
    b: tuple[Tensor, Tensor, Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def _mass_from_4vec_torch(
    e: Tensor, px: Tensor, py: Tensor, pz: Tensor
) -> Tensor:
    return torch.sqrt(torch.clamp(e * e - px * px - py * py - pz * pz, min=0.0))


@torch.no_grad()
def _val_mass_reconstruction_masses(
    batch_d: dict[str, Any],
    nu_kin: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """W and top masses [GeV] for valid TT2L events (both tops, flattened).

    Uses ground-truth ``assignments-indices`` to pick b/lepton from the point cloud. The
    point cloud ``x`` and ``nu_kin`` are already physical (log-space) values stored in the
    batch — the model normalizes internally during forward — so NO denormalization is
    applied here. ``nu_kin`` is ``(B, 2, 3)`` physical invisible kinematics (pred or truth).
    """
    assign = batch_d.get("assignments-indices")
    assign_m = batch_d.get("assignments-mask")
    if not isinstance(assign, Tensor) or assign.dim() != 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if not isinstance(assign_m, Tensor):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    B = int(batch_d["x"].shape[0])
    if assign.shape[0] != B or assign.shape[1] < 2 or assign.shape[2] < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    pc = batch_d["x"].to(device=device, dtype=dtype)

    b_idx = torch.arange(B, device=device)
    idx_ok = (assign[..., :2] >= 0).all(dim=-1)
    event_ok = (
        get_event_valid_mask(batch_d, B, device, dtype).reshape(B) > 0
    ) & (assign_m > 0).all(dim=-1) & idx_ok.all(dim=-1)
    if not bool(event_ok.any().item()):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    nu_kin = nu_kin.to(device=device, dtype=dtype)
    nu1 = _nu_kin_to_4vec_torch(nu_kin[:, 0, :], cartesian=cartesian)
    nu2 = _nu_kin_to_4vec_torch(nu_kin[:, 1, :], cartesian=cartesian)

    w_masses: list[Tensor] = []
    top_masses: list[Tensor] = []
    for r in range(2):
        b_pc = _pc_row_to_4vec_torch(pc[b_idx, assign[:, r, 0].long()])
        l_pc = _pc_row_to_4vec_torch(pc[b_idx, assign[:, r, 1].long()])
        nu = nu1 if r == 0 else nu2
        w = _add_4vec_torch(l_pc, nu)
        top = _add_4vec_torch(b_pc, w)
        w_masses.append(_mass_from_4vec_torch(*w))
        top_masses.append(_mass_from_4vec_torch(*top))

    w_all = torch.stack(w_masses, dim=-1)[event_ok].reshape(-1)
    top_all = torch.stack(top_masses, dim=-1)[event_ok].reshape(-1)
    w_np = w_all.detach().float().cpu().numpy()
    top_np = top_all.detach().float().cpu().numpy()
    finite = np.isfinite(w_np) & np.isfinite(top_np)
    return w_np[finite], top_np[finite]


def _val_overlay_kin_figure(
    counts_truth: np.ndarray,
    counts_pred: np.ndarray,
    bin_edges: np.ndarray,
    title: str,
    *,
    pred_label: str = "Pred (val)",
    counts_ref: np.ndarray | None = None,
    ref_label: str = "Ref (frozen)",
    xlabel: str = "Value",
) -> Any:
    """1D density overlay (truth vs current-policy prediction, optionally also reference policy), EveNet-style, as ``wandb.Image``.

    Args:
        counts_truth: Histogram counts for truth.
        counts_pred: Histogram counts for current-policy prediction.
        bin_edges: Bin edges array (length = len(counts_truth) + 1).
        title: Figure title.
        pred_label: Legend label for the current-policy series.
        counts_ref: Optional histogram counts for the frozen reference policy.
        ref_label: Legend label for the reference series.
        xlabel: x-axis label.
    """
    import wandb

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = float(bin_edges[1] - bin_edges[0])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    nt = np.sum(counts_truth) * w + 1e-12
    npred = np.sum(counts_pred) * w + 1e-12
    ax.plot(
        centers,
        counts_truth / nt,
        label="Truth",
        linewidth=2.0,
        marker="o",
        markersize=4,
    )
    ax.plot(
        centers,
        counts_pred / npred,
        label=pred_label,
        linewidth=2.0,
        marker="s",
        markersize=4,
    )
    if counts_ref is not None:
        nref = np.sum(counts_ref) * w + 1e-12
        ax.plot(
            centers,
            counts_ref / nref,
            label=ref_label,
            linewidth=2.0,
            marker="^",
            markersize=4,
            linestyle="--",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def _val_selected_delta_arrays(
    candidates: Tensor,
    batch_d: dict[str, Any],
    k_sel: Tensor,
    *,
    cartesian: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """Selected-candidate validation residual arrays for profile and response plots."""
    B = int(batch_d["x"].shape[0])
    N_nu = int(candidates.shape[2])
    xm = batch_d["x_invisible_mask"]
    if xm.dim() == 3 and xm.shape[-1] == 1:
        slot_mask = xm.squeeze(-1).to(device=device) > 0
    else:
        slot_mask = xm.to(device=device) > 0
    slot_mask = slot_mask.reshape(B, N_nu)
    event_valid = get_event_valid_mask(batch_d, B, device, dtype).reshape(B) > 0
    valid_slots = slot_mask & event_valid.unsqueeze(-1)

    b_idx = torch.arange(B, device=device)
    pred = candidates[k_sel, b_idx]
    if _supports_legacy_invisible_kinematics(
        cartesian=cartesian,
        feature_dim=min(int(pred.shape[-1]), int(batch_d["x_invisible"].shape[-1])),
    ):
        if cartesian:
            truth = batch_d["x_invisible_cartesian"].to(device=device, dtype=dtype)
            plp, pred_eta, _pred_phi = cartesian_to_log_pt_eta_phi(
                pred[..., 0], pred[..., 1], pred[..., 2]
            )
            tlp, truth_eta, _truth_phi = cartesian_to_log_pt_eta_phi(
                truth[..., 0], truth[..., 1], truth[..., 2]
            )
            pred_xyz = pred[:, :2, :3].contiguous()
            truth_xyz = truth[:, :2, :3].contiguous()
        else:
            truth = batch_d["x_invisible"].to(device=device, dtype=dtype)
            plp, pred_eta = pred[..., 0], pred[..., 1]
            pred_phi = pred[..., 2]
            tlp, truth_eta = truth[..., 0], truth[..., 1]
            truth_phi = truth[..., 2]
            pred_xyz = log_pt_eta_phi_to_cartesian(
                plp.clamp(-10.0, 10.0), pred_eta, pred_phi
            )[:, :2, :].contiguous()
            truth_xyz = log_pt_eta_phi_to_cartesian(
                tlp.clamp(-10.0, 10.0), truth_eta, truth_phi
            )[:, :2, :].contiguous()

        pred_pt = torch.expm1(plp.clamp(-10.0, 10.0))
        truth_pt = torch.expm1(tlp.clamp(-10.0, 10.0))
        delta_pt = pred_pt - truth_pt
        delta_eta = pred_eta - truth_eta
        delta_xyz = pred_xyz - truth_xyz

        slot_count = valid_slots.sum(dim=-1)
        valid_events_with_slots = slot_count > 0
        pt_delta_event_mean = (
            (delta_pt * valid_slots.to(delta_pt.dtype)).sum(dim=-1)
            / slot_count.clamp(min=1).to(delta_pt.dtype)
        )

        return {
            "pt_truth": truth_pt[valid_slots].detach().float().cpu().numpy(),
            "pt_delta": delta_pt[valid_slots].detach().float().cpu().numpy(),
            "px_delta": delta_xyz[..., 0][valid_slots].detach().float().cpu().numpy(),
            "py_delta": delta_xyz[..., 1][valid_slots].detach().float().cpu().numpy(),
            "pz_delta": delta_xyz[..., 2][valid_slots].detach().float().cpu().numpy(),
            "eta_truth": truth_eta[valid_slots].detach().float().cpu().numpy(),
            "eta_delta": delta_eta[valid_slots].detach().float().cpu().numpy(),
            "pt_delta_event_mean": pt_delta_event_mean[valid_events_with_slots]
            .detach()
            .float()
            .cpu()
            .numpy(),
        }

    truth = batch_d["x_invisible"].to(device=device, dtype=dtype)
    feature_dim = min(int(pred.shape[-1]), int(truth.shape[-1]))
    feature_names = list(_invisible_feature_names())
    if len(feature_names) < feature_dim:
        feature_names.extend(f"feature_{index}" for index in range(len(feature_names), feature_dim))
    periodic_indices = set(
        index for index in _invisible_periodic_feature_indices() if 0 <= index < feature_dim
    )
    pred_sel = pred[..., :feature_dim]
    truth_sel = truth[..., :feature_dim]
    delta = pred_sel - truth_sel
    for index in periodic_indices:
        delta[..., index] = wrapped_delta_phi(pred_sel[..., index], truth_sel[..., index])

    out: dict[str, np.ndarray] = {}
    for index, feature_name in enumerate(feature_names[:feature_dim]):
        out[f"{feature_name}_truth"] = (
            truth_sel[..., index][valid_slots].detach().float().cpu().numpy()
        )
        out[f"{feature_name}_delta"] = (
            delta[..., index][valid_slots].detach().float().cpu().numpy()
        )
    return out


def _concat_np_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate non-empty numpy chunks into one float64 vector."""
    parts = [np.asarray(x, dtype=np.float64).reshape(-1) for x in chunks if x.size > 0]
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float64)


def _gather_val_array_dict(
    local_arrays: dict[str, np.ndarray],
    *,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    """Gather per-rank validation arrays to rank 0 and concatenate matching keys."""
    if world_size <= 1:
        return {
            k: np.asarray(v, dtype=np.float64).reshape(-1)
            for k, v in local_arrays.items()
        }
    if rank == 0:
        gathered: list[Any] = [None] * world_size
        dist.gather_object(local_arrays, object_gather_list=gathered, dst=0)
        keys = set(local_arrays)
        for part in gathered:
            if isinstance(part, dict):
                keys.update(part)
        merged: dict[str, np.ndarray] = {}
        for key in keys:
            chunks = [
                np.asarray(part.get(key, np.array([], dtype=np.float64)), dtype=np.float64)
                .reshape(-1)
                for part in gathered
                if isinstance(part, dict)
            ]
            merged[key] = _concat_np_chunks(chunks)
        return merged
    dist.gather_object(local_arrays, dst=0)
    return {}


def _gather_val_ndarray_dict(
    local_arrays: dict[str, np.ndarray],
    *,
    rank: int,
    world_size: int,
) -> dict[str, np.ndarray]:
    """Gather validation arrays to rank 0 while preserving non-event dimensions."""
    if world_size <= 1:
        return {key: np.asarray(value) for key, value in local_arrays.items()}
    if rank == 0:
        gathered: list[Any] = [None] * world_size
        dist.gather_object(local_arrays, object_gather_list=gathered, dst=0)
        keys: set[str] = set()
        for part in gathered:
            if isinstance(part, dict):
                keys.update(part)
        merged: dict[str, np.ndarray] = {}
        for key in keys:
            chunks = [
                np.asarray(part[key])
                for part in gathered
                if isinstance(part, dict) and key in part and np.asarray(part[key]).size > 0
            ]
            if chunks:
                merged[key] = np.concatenate(chunks, axis=0)
        return merged
    dist.gather_object(local_arrays, dst=0)
    return {}


def _profile_fit_metrics(
    profile_name: str,
    truth_value: np.ndarray,
    delta_value: np.ndarray,
) -> tuple[float, float]:
    """Fit binned mean residual with ``delta_mean = slope * truth + intercept``."""
    bin_edges = _profile_bin_edges(profile_name, [truth_value])
    centers, means, _errors, counts = _binned_delta_profile(
        truth_value, delta_value, bin_edges=bin_edges
    )
    keep = np.isfinite(centers) & np.isfinite(means) & (counts > 0)
    if int(np.sum(keep)) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(centers[keep], means[keep], deg=1)
    slope_f = float(slope)
    intercept_f = float(intercept)
    zero = (
        float(-intercept_f / slope_f)
        if math.isfinite(slope_f) and abs(slope_f) > 1e-12
        else float("nan")
    )
    return slope_f, zero


def _validation_delta_profile_figure(
    truth_current: np.ndarray,
    delta_current: np.ndarray,
    *,
    profile_name: str,
    title: str,
    truth_initial: np.ndarray | None = None,
    delta_initial: np.ndarray | None = None,
) -> Any:
    """Validation profile plot of mean selected-candidate residual versus truth value."""
    import wandb

    initial_arrays = []
    if truth_initial is not None and delta_initial is not None:
        initial_arrays = [truth_initial]
    x_label, y_label, display = _profile_axis_labels(profile_name)
    bin_edges = _profile_bin_edges(profile_name, [truth_current] + initial_arrays)
    centers, mean_cur, err_cur, counts = _binned_delta_profile(
        truth_current, delta_current, bin_edges=bin_edges
    )
    mean_init = err_init = None
    if truth_initial is not None and delta_initial is not None:
        _centers, mean_init, err_init, _counts = _binned_delta_profile(
            truth_initial, delta_initial, bin_edges=bin_edges
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    if mean_init is not None and err_init is not None:
        keep_init = np.isfinite(mean_init)
        if np.any(keep_init):
            ax.errorbar(
                centers[keep_init],
                mean_init[keep_init],
                yerr=err_init[keep_init],
                fmt="o--",
                linewidth=1.6,
                markersize=4,
                capsize=2,
                color="#7f7f7f",
                label=f"initial validation {display}",
            )
    keep_cur = np.isfinite(mean_cur)
    if np.any(keep_cur):
        ax.errorbar(
            centers[keep_cur],
            mean_cur[keep_cur],
            yerr=err_cur[keep_cur],
            fmt="s-",
            linewidth=1.8,
            markersize=4,
            capsize=2,
            color="#1f77b4",
            label=f"current validation {display}",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax_count = ax.twinx()
    width = float(centers[1] - centers[0]) * 0.85 if centers.size > 1 else 1.0
    ax_count.bar(centers, counts, width=width, alpha=0.12, color="gray", label="entries")
    ax_count.set_ylabel("Entries")

    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_slope_history_figure(
    epochs: list[float],
    slopes: list[float],
    *,
    profile_name: str,
) -> Any:
    """History plot: epoch number versus fitted validation residual-profile slope."""
    import wandb

    _x_label, _y_label, display = _profile_axis_labels(profile_name)
    ep = np.asarray(epochs, dtype=np.float64)
    sl = np.asarray(slopes, dtype=np.float64)
    keep = np.isfinite(ep) & np.isfinite(sl)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if np.any(keep):
        ax.plot(ep[keep], sl[keep], "o-", linewidth=1.8, markersize=4)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Fitted {display} residual slope")
    ax.set_title(f"Validation {display} residual-profile slope over epochs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_epoch_history_figure(
    epochs: list[float],
    values: list[float],
    *,
    ylabel: str,
    title: str,
    zero_line: bool = True,
) -> Any:
    """History plot with epoch on the x-axis and one validation diagnostic on y."""
    import wandb

    ep = np.asarray(epochs, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(ep) & np.isfinite(val)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if np.any(keep):
        ax.plot(ep[keep], val[keep], "o-", linewidth=1.8, markersize=4)
    if zero_line:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _validation_zero_vs_slope_figure(
    slopes: list[float],
    zero_points: list[float],
    epochs: list[float],
    *,
    profile_name: str,
) -> Any:
    """History plot: fitted slope versus truth value where mean residual crosses zero."""
    import wandb

    x_label, _y_label, display = _profile_axis_labels(profile_name)
    sl = np.asarray(slopes, dtype=np.float64)
    zp = np.asarray(zero_points, dtype=np.float64)
    ep = np.asarray(epochs, dtype=np.float64)
    keep = np.isfinite(sl) & np.isfinite(zp)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if np.any(keep):
        sc = ax.scatter(sl[keep], zp[keep], c=ep[keep], cmap="viridis", s=34)
        ax.plot(sl[keep], zp[keep], "-", linewidth=1.1, alpha=0.55)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Epoch")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(f"Fitted {display} residual slope")
    ax.set_ylabel(f"{x_label} where mean residual = 0")
    ax.set_title(f"Validation {display} zero-crossing versus slope")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _response_matrix_figure(
    initial: np.ndarray,
    current: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> Any:
    """2D response matrix comparing event-level initial and current validation values."""
    import wandb

    x = np.asarray(initial, dtype=np.float64).reshape(-1)
    y = np.asarray(current, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)

    stacked = np.concatenate([x, y], axis=0)
    lo, hi = [float(v) for v in np.nanpercentile(stacked, [1.0, 99.0])]
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
        lo, hi = center - 1.0, center + 1.0
    pad = max(0.05 * (hi - lo), 1e-6)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hist = ax.hist2d(x, y, bins=50, range=[[lo, hi], [lo, hi]], cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="Events")
    ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axhline(0.0, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.axvline(0.0, color="white", linestyle=":", linewidth=0.9, alpha=0.65)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _truth_pred_matrix_figure(
    truth: np.ndarray,
    pred: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    bin_edges: np.ndarray | None = None,
) -> Any:
    """2D density matrix for truth-vs-pred comparisons."""
    import wandb

    x, y = paired_finite_truth_pred(
        truth,
        pred,
        context="truth/pred matrix",
    )
    if x.size == 0:
        x = np.array([0.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)

    if bin_edges is not None and len(bin_edges) >= 2:
        edges = np.asarray(bin_edges, dtype=np.float64)
        lo = float(edges[0])
        hi = float(edges[-1])
    else:
        stacked = np.concatenate([x, y], axis=0)
        lo, hi = [float(v) for v in np.nanpercentile(stacked, [1.0, 99.0])]
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.nanmean(stacked)) if stacked.size > 0 else 0.0
            lo, hi = center - 1.0, center + 1.0
        pad = max(0.05 * (hi - lo), 1e-6)
        lo -= pad
        hi += pad
        edges = np.linspace(lo, hi, 51)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    hist = ax.hist2d(x, y, bins=[edges, edges], cmap="viridis")
    fig.colorbar(hist[3], ax=ax, label="Samples")
    ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def _snapshot_policy_state_dict(model: torch.nn.Module) -> dict[str, Tensor]:
    """CPU policy snapshot paired with a newly materialized K=1 denominator."""
    return {
        key: value.detach().cpu().clone()
        for key, value in unwrap_for_state_dict(model).state_dict().items()
    }


def _restore_policy_from_dgpo_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
) -> int:
    """Restore the live policy from one unpruned trajectory checkpoint.

    Only the live ``state_dict`` is restored.  Optimizer and EMA state are
    intentionally handled by the caller because an older trajectory point must
    start a new search direction rather than resume its stale Adam momentum.
    Returns the number of model tensors loaded.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"trajectory checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"trajectory checkpoint has no state_dict: {path}")
    core = unwrap_for_state_dict(model)
    target_state = core.state_dict()
    clean_state = {
        key.removeprefix("model."): value
        for key, value in raw_state.items()
        if key.removeprefix("model.") in target_state
    }
    missing = sorted(set(target_state) - set(clean_state))
    mismatched = sorted(
        key
        for key, value in clean_state.items()
        if tuple(value.shape) != tuple(target_state[key].shape)
    )
    if missing or mismatched:
        raise ValueError(
            "trajectory checkpoint is incompatible with the live policy: "
            f"missing={missing[:5]} mismatched={mismatched[:5]}"
        )
    core.load_state_dict(clean_state, strict=True)
    return len(clean_state)


def _raw_refit_failure_decision(*, rollback_applied: bool) -> str:
    """Global-best bookkeeping alone does not mean the policy was rewound.

    A rejected candidate never installs a new reward/reference. With no actual
    rollback, the existing pair remains valid and the next normal monitor can
    retry. A rewound policy instead requires recovery from a complete checkpoint.
    """
    if rollback_applied:
        raise RuntimeError(
            "OmniFold refit was not accepted after policy rollback; refusing to "
            "train the rolled-back policy with the old reward/reference. "
            "Resume the last complete checkpoint."
        )
    return "forward_recenter_deferred"


def _reset_optimizer_after_reward_install(
    optimizer: torch.optim.Optimizer,
    *,
    cfg: Any,
    accepted: bool,
    adaptive_state: Any = None,
) -> dict[str, float]:
    """Reset only on committed rounds; preserve parameters, groups and LR clock."""
    if not accepted:
        return {}
    diagnostics = {}
    if getattr(cfg, "policy_warmup_steps", 0):
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import start_policy_round_warmup
        if adaptive_state is None:
            raise ValueError("policy round warmup requires checkpointed adaptive state")
        diagnostics.update(start_policy_round_warmup(adaptive_state, cfg=cfg))
    if cfg.reset_optimizer_state_on_install:
        count = len(optimizer.state)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        _log.info(
            "[DGPO/omnifold] new reward round: reset complete AdamW state "
            "for %s parameters (first/second moments and per-parameter steps); "
            "preserved LR, scheduler and weight decay", count,
        )
        return {
            **diagnostics,
            "reference_trust/adam_full_state_reset": 1.0,
            "reference_trust/adam_states_cleared": float(count),
        }
    if cfg.trust_reset_adam_first_moment:
        return {**diagnostics, "reference_trust/adam_first_moments_reset":
                float(reset_adam_first_moment(optimizer))}
    return diagnostics


def _resolve_raw_best_policy_checkpoint(
    adaptive_state: Any,
    save_dir: Path | str | None,
    *,
    global_scope: bool = False,
    source_dirs: Sequence[Path | str] = (),
) -> Path:
    """Resolve the unpruned checkpoint for this reward round's best raw AUC.

    New checkpoints store the exact path.  For v16 checkpoints written before
    best-point rollback was added, fall back to the best epoch reconstructed
    from the saved raw-audit history and locate its snapshot in ``save_dir``.
    """

    if save_dir is None or not str(save_dir).strip():
        raise RuntimeError(
            "raw-AUC best-point rollback requires model_checkpoint_save_path"
        )
    save_root = Path(str(save_dir)).expanduser().resolve()
    recorded = str(getattr(adaptive_state, "raw_best_checkpoint", "") or "")
    if global_scope:
        from RL.DGPO_neutrino.model_utils import _load_checkpoint_metadata, _completed_raw_monitor_records
        epoch, step, next_epoch = (int(adaptive_state.raw_best_epoch),
                                  int(adaptive_state.raw_best_global_step),
                                  int(adaptive_state.raw_best_next_epoch))
        if step < 0 or next_epoch not in (epoch, epoch + 1):
            raise ValueError("global-best checkpoint requires exact epoch/step/next_epoch metadata")
        name = dgpo_snapshot_checkpoint_name(last_completed_epoch=epoch, dgpo_next_epoch=next_epoch, global_step=step)
        candidates = ([Path(recorded).expanduser()] if recorded else []) + [
            Path(root).expanduser() / name for root in (save_root, *source_dirs)
        ]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            payload = _load_checkpoint_metadata(candidate)
            expected = (float(adaptive_state.raw_best_auc_gap), epoch, step, next_epoch)
            records = _completed_raw_monitor_records(payload.get("dgpo_adaptive_omnifold_state", {}))
            if ("state_dict" not in payload or payload.get("epoch") != epoch
                    or payload.get("global_step") != step or payload.get("dgpo_next_epoch") != next_epoch
                    or not any(r[1:] == expected[1:] and abs(r[0] - expected[0]) < 1e-10 for r in records)):
                raise ValueError(f"global-best checkpoint metadata/raw AUC mismatch: {candidate}")
            adaptive_state.raw_best_checkpoint = str(candidate.resolve())
            return candidate.resolve()
        raise FileNotFoundError(f"exact global-best checkpoint unavailable: {name}; searched {candidates}")
    candidates: list[Path] = []
    if recorded:
        recorded_path = Path(recorded).expanduser()
        candidates.append(recorded_path)
        candidates.append(save_root / recorded_path.name)
    best_epoch = int(getattr(adaptive_state, "raw_best_epoch", -1))
    if best_epoch >= 0 and save_root.is_dir():
        candidates.extend(
            sorted(
                save_root.glob(
                    f"dgpo-epoch={best_epoch}-next_ep={best_epoch + 1}-step=*.ckpt"
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        )
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        token = str(resolved)
        if token in seen:
            continue
        seen.add(token)
        if resolved.is_file():
            adaptive_state.raw_best_checkpoint = token
            return resolved
    raise FileNotFoundError(
        "raw-AUC plateau reached but the best-policy checkpoint is unavailable: "
        f"best_epoch={best_epoch} recorded={recorded or '<none>'} "
        f"save_dir={save_root}"
    )


def _rewind_policy_for_raw_best_refit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | str,
    *,
    ema_save: Any | None = None,
    ema_rollout: Any | None = None,
    clear_optimizer: bool = True,
) -> tuple[int, int, bool]:
    """Restore a raw-AUC best policy and discard the stale Adam direction."""

    loaded_tensors = _restore_policy_from_dgpo_checkpoint(
        model,
        checkpoint_path,
    )
    policy_core = unwrap_for_state_dict(model)
    cleared_optimizer_states = len(optimizer.state) if clear_optimizer else 0
    if clear_optimizer:
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
    if ema_save is not None:
        ema_save.update(policy_core, decay_=0.0)
    if ema_rollout is not None:
        ema_rollout.update(policy_core, decay_=0.0)
    return (
        loaded_tensors,
        cleared_optimizer_states,
        bool(ema_save is not None or ema_rollout is not None),
    )


def _confirm_global_raw_candidate(
    *, model: torch.nn.Module, comparison_model: torch.nn.Module,
    best_checkpoint: Path, materialize_pair: Any, fit_judge: Any,
    min_delta: float,
    initial_judge_cache: dict[str, Any] | None = None,
) -> tuple[bool | None, dict[str, float]]:
    """Compare two raw policies with common events/noise and independent equal-start judges.

    The caller's reference model is borrowed only during sample generation;
    its exact state is restored before fitting. No optimizer or monitor cache is
    touched. This is an engineering confirmation, not a significance test.
    """
    reference_snapshot = _snapshot_policy_state_dict(comparison_model)
    try:
        _restore_policy_from_dgpo_checkpoint(comparison_model, best_checkpoint)
        _, candidate_pool, incumbent_pool = materialize_pair(model, comparison_model)
    finally:
        unwrap_for_state_dict(comparison_model).load_state_dict(reference_snapshot, strict=True)
    def fit_one(pool: Any, phase: str) -> Any:
        if initial_judge_cache is None:
            return fit_judge(pool, phase)
        # Each fit may mutate its cache; never pass the first fitted judge to the
        # second, and never install either confirmation judge into the monitor.
        return fit_judge(pool, phase, warm_start_cache=copy.deepcopy(initial_judge_cache))
    candidate = fit_one(candidate_pool, "global_best_candidate")
    incumbent = fit_one(incumbent_pool, "global_best_incumbent")
    gaps = [float(p.get("raw_auc_gap", float("nan"))) for p in (candidate, incumbent)]
    valid = all(math.isfinite(g) and 0 <= g <= .5 for g in gaps) and all(
        float(p.get("raw_audit_saturated", 0.)) >= .5 for p in (candidate, incumbent)
    )
    accepted = bool(gaps[0] < gaps[1] - min_delta) if valid else None
    return accepted, {
        "staleness/global_best/confirmation_valid": float(valid),
        "staleness/global_best/confirmation_accepted": float(accepted is True),
        "staleness/global_best/candidate_gap": gaps[0],
        "staleness/global_best/incumbent_gap": gaps[1],
        "staleness/global_best/confirmation_delta": gaps[0] - gaps[1],
    }


class _FixedEventInputShard:
    """Cache CPU event inputs once; never cache policy-generated candidates.

    This pins event order as well as membership despite Ray re-iteration order
    changes. Classifier minibatch shuffling happens later, after identity splits.
    The cache is process-local; persistent split provenance remains seed/hash
    based. Generation uses one fixed batch size throughout a run.
    """

    def __init__(self, source: Any) -> None:
        self.source = source
        self.batches: list[dict[str, Any]] | None = None
        self.loader_config: dict[str, Any] | None = None

    @staticmethod
    def _copy_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value.detach().cpu().clone() if isinstance(value, Tensor)
                else copy.deepcopy(value) for key, value in batch.items()}

    def iter_torch_batches(self, **kwargs: Any) -> Any:
        if self.batches is None:
            # Finish the read before yielding: a capped monitor must not leave
            # a live Ray iterator whose next pass chooses a different prefix.
            batches = [self._copy_batch(batch)
                       for batch in self.source.iter_torch_batches(**kwargs)]
            self.batches = batches
            self.loader_config = dict(kwargs)
            events = sum(int(batch["x"].shape[0]) for batch in batches)
            size = sum(v.numel() * v.element_size() for batch in batches
                       for v in batch.values() if isinstance(v, Tensor))
            _log.info("[DGPO/omnifold] Fixed CPU event cache: events=%s batches=%s MiB=%.1f; candidates regenerated per policy.",
                      events, len(batches), size / (1024 ** 2))
        elif self.loader_config != kwargs:
            raise ValueError("fixed classifier input cache requires unchanged generation loader settings")
        for batch in self.batches:
            # CPU inference/test paths may mutate their input; preserve cache.
            yield self._copy_batch(batch)


@torch.no_grad()
def _materialize_adaptive_omnifold_pool(
    data_shard: Any,
    loader_config: dict[str, Any],
    *,
    model: torch.nn.Module,
    paired_reference_model: torch.nn.Module | None = None,
    sampler: DDIMSampler,
    device: torch.device,
    world_size: int,
    rank: int,
    quota_events: int | None,
    paired_reference_quota_events: int | None = None,
    num_ddim_steps: int,
    seed: int,
    include_pairwise_context: bool = False,
    include_visible_pair_rest_frame: bool = False,
    collect_policy_noise_mask: bool = False,
    training_crossfit_fold: tuple[int, int, int] | None = None,
) -> Any:
    """Generate and all-gather one or a paired pair of K=1 pools.

    This deliberately samples the live policy, not either EMA. The caller takes
    its round-reference snapshot at the same point, so an accepted ratio's Gen
    denominator and DGPO velocity anchor are the same policy.  When
    ``paired_reference_model`` is provided, both policies consume the same data
    iterator and the same DDIM noise. ``paired_reference_quota_events`` may cap
    that paired prefix below the larger current-policy audit population. Ray
    Data does not promise that two new iterators traverse blocks in the same
    order, so materializing the trust pair in one pass is required to preserve
    exact event identities.
    """
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
        AdaptiveOmniFoldPool,
        gather_pool_across_ranks,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EventPackingSpec,
        _local_identity_fold_labels,
        event_identity_inputs,
        pack_event_inputs,
    )

    if training_crossfit_fold is not None:
        folds, fold, fold_seed = training_crossfit_fold
        if not 1 <= fold <= folds or folds < 2 or quota_events is not None or paired_reference_model is not None:
            raise ValueError("training fold selection needs an uncapped, unpaired crossfit pool")
    core = _unwrap_core_evenet(model)
    paired_core = (
        None
        if paired_reference_model is None
        else _unwrap_core_evenet(paired_reference_model)
    )
    was_training = core.training
    paired_was_training = (
        None if paired_core is None else bool(paired_core.training)
    )
    core.eval()
    if paired_core is not None:
        paired_core.eval()
    packed_chunks: list[Tensor] = []
    truth_chunks: list[Tensor] = []
    candidate_chunks: list[Tensor] = []
    policy_mask_chunks: list[Tensor] = []
    paired_packed_chunks: list[Tensor] = []
    paired_truth_chunks: list[Tensor] = []
    paired_current_candidate_chunks: list[Tensor] = []
    paired_candidate_chunks: list[Tensor] = []
    packing_spec: EventPackingSpec | None = None
    collected = 0
    paired_collected = 0
    per_rank_quota = (
        None
        if quota_events is None
        else max(1, int(math.ceil(int(quota_events) / max(1, int(world_size)))))
    )
    if paired_reference_model is None and paired_reference_quota_events is not None:
        raise ValueError(
            "paired_reference_quota_events requires paired_reference_model"
        )
    if (
        quota_events is not None
        and paired_reference_quota_events is not None
        and int(paired_reference_quota_events) > int(quota_events)
    ):
        raise ValueError(
            "paired reference quota cannot exceed the current-policy quota"
        )
    per_rank_paired_quota = (
        per_rank_quota
        if paired_core is not None and paired_reference_quota_events is None
        else None
        if paired_core is None or paired_reference_quota_events is None
        else max(
            1,
            int(
                math.ceil(
                    int(paired_reference_quota_events)
                    / max(1, int(world_size))
                )
            ),
        )
    )
    data_iter = iter(data_shard.iter_torch_batches(**loader_config))
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed) + int(rank))
            while True:
                batch_cpu, has_more = _next_batch_synced(
                    data_iter,
                    world_size=world_size,
                    device=device,
                    require_all_ranks=quota_events is not None,
                )
                if not has_more:
                    break
                already_full = (
                    batch_cpu is None
                    or (per_rank_quota is not None and collected >= per_rank_quota)
                )
                if not already_full:
                    batch = batch_to_device(batch_cpu, device)
                    batch_size = int(batch["x"].shape[0])
                    valid = get_event_valid_mask(
                        batch,
                        batch_size,
                        device,
                        torch.float32,
                    ) > 0
                    if training_crossfit_fold is not None:
                        # Match repeat-1 reward fit identities BEFORE generation.
                        # No collectives here: per-rank input sizes may differ.
                        packed, packing_spec = pack_event_inputs(
                            batch, packing_spec,
                            include_pairwise_context=include_pairwise_context,
                            include_visible_pair_rest_frame=include_visible_pair_rest_frame,
                        )
                        labels = _local_identity_fold_labels(
                            event_identity_inputs(packed, packing_spec), folds=folds, seed=fold_seed,
                        )
                        keep_fold = valid & (labels != fold - 1)
                        batch = {
                            key: value[keep_fold] if isinstance(value, Tensor)
                            and value.ndim > 0 and len(value) == batch_size else value
                            for key, value in batch.items()
                        }
                        valid = valid[keep_fold]
                    if bool(valid.any().item()):
                        # Snapshot the process-local RNG immediately before
                        # current-policy DDIM. Restoring it for the paired
                        # reference gives common random numbers without
                        # changing the RNG stream seen by the next batch.
                        cpu_rng_state = torch.random.get_rng_state()
                        cuda_rng_state = (
                            torch.cuda.get_rng_state(device)
                            if device.type == "cuda"
                            else None
                        )
                        generated = generate_neutrino_candidates(
                            core,
                            batch,
                            sampler,
                            K=1,
                            num_ddim_steps=int(num_ddim_steps),
                            device=device,
                            parallel_chains=1,
                            tqdm_k_chains=False,
                            use_tqdm_ddim=False,
                        )
                        paired_generated: Tensor | None = None
                        paired_needed = bool(
                            paired_core is not None
                            and (
                                per_rank_paired_quota is None
                                or paired_collected < per_rank_paired_quota
                            )
                        )
                        if paired_needed:
                            assert paired_core is not None
                            torch.random.set_rng_state(cpu_rng_state)
                            if cuda_rng_state is not None:
                                torch.cuda.set_rng_state(cuda_rng_state, device)
                            paired_generated = generate_neutrino_candidates(
                                paired_core,
                                batch,
                                sampler,
                                K=1,
                                num_ddim_steps=int(num_ddim_steps),
                                device=device,
                                parallel_chains=1,
                                tqdm_k_chains=False,
                                use_tqdm_ddim=False,
                            )
                        invisible = batch.get("x_invisible")
                        if not isinstance(invisible, Tensor):
                            raise KeyError("adaptive OmniFold fit needs x_invisible truth")
                        if tuple(generated.shape[2:]) != (2, 2):
                            raise ValueError(
                                "Ztautau adaptive OmniFold requires generated "
                                f"(K,B,2,2), got {tuple(generated.shape)}"
                            )
                        if paired_generated is not None and (
                            tuple(paired_generated.shape) != tuple(generated.shape)
                        ):
                            raise ValueError(
                                "paired current/reference adaptive generations "
                                "must have identical shapes; "
                                f"current={tuple(generated.shape)} "
                                f"reference={tuple(paired_generated.shape)}"
                            )
                        if int(invisible.shape[1]) < 2 or int(invisible.shape[2]) < 2:
                            raise ValueError("adaptive OmniFold truth needs two 2D slots")
                        packed, packing_spec = pack_event_inputs(
                            batch,
                            packing_spec,
                            include_pairwise_context=include_pairwise_context,
                            include_visible_pair_rest_frame=include_visible_pair_rest_frame,
                        )
                        keep = valid.nonzero(as_tuple=True)[0]
                        if per_rank_quota is not None:
                            remaining = max(0, per_rank_quota - collected)
                            keep = keep[:remaining]
                        if int(keep.numel()) > 0:
                            packed_chunks.append(packed[keep].detach().cpu())
                            if collect_policy_noise_mask:
                                policy_mask_chunks.append(batch["x_invisible_mask"][keep].detach().cpu())
                            truth_chunks.append(
                                invisible[keep, :2, :2]
                                .reshape(len(keep), 4)
                                .detach()
                                .float()
                                .cpu()
                            )
                            candidate_chunks.append(
                                generated[:, keep, :2, :2]
                                .permute(1, 0, 2, 3)
                                .reshape(len(keep), 1, 4)
                                .detach()
                                .float()
                                .cpu()
                            )
                            if paired_generated is not None:
                                paired_keep = keep
                                if per_rank_paired_quota is not None:
                                    paired_remaining = max(
                                        0,
                                        per_rank_paired_quota - paired_collected,
                                    )
                                    paired_keep = paired_keep[:paired_remaining]
                                paired_packed_chunks.append(
                                    packed[paired_keep].detach().cpu()
                                )
                                paired_truth_chunks.append(
                                    invisible[paired_keep, :2, :2]
                                    .reshape(len(paired_keep), 4)
                                    .detach()
                                    .float()
                                    .cpu()
                                )
                                paired_current_candidate_chunks.append(
                                    generated[:, paired_keep, :2, :2]
                                    .permute(1, 0, 2, 3)
                                    .reshape(len(paired_keep), 1, 4)
                                    .detach()
                                    .float()
                                    .cpu()
                                )
                                paired_candidate_chunks.append(
                                    paired_generated[:, paired_keep, :2, :2]
                                    .permute(1, 0, 2, 3)
                                    .reshape(len(paired_keep), 1, 4)
                                    .detach()
                                    .float()
                                    .cpu()
                                )
                                paired_collected += int(paired_keep.numel())
                            collected += int(keep.numel())
                if per_rank_quota is not None:
                    done = torch.tensor(
                        [1 if collected >= per_rank_quota else 0],
                        device=device,
                        dtype=torch.int64,
                    )
                    if world_size > 1:
                        dist.all_reduce(done, op=dist.ReduceOp.MIN)
                    if int(done.item()) == 1:
                        break
    finally:
        core.train(was_training)
        if paired_core is not None and paired_was_training is not None:
            paired_core.train(paired_was_training)

    if training_crossfit_fold is not None and packing_spec is not None and not packed_chunks:
        # A small local shard can contain only the other fold. It must still
        # join all-gather; otherwise the remaining ranks could hang.
        packed_chunks.append(torch.empty((0, packing_spec.width)))
        truth_chunks.append(torch.empty((0, 4)))
        candidate_chunks.append(torch.empty((0, 1, 4)))
        if collect_policy_noise_mask:
            policy_mask_chunks.append(torch.empty((0, 2)))
    if not packed_chunks or packing_spec is None:
        raise RuntimeError("adaptive OmniFold pool collected no valid events")
    local = {
        "packed_event": torch.cat(packed_chunks, dim=0),
        "truth": torch.cat(truth_chunks, dim=0),
        "candidates": torch.cat(candidate_chunks, dim=0),
        "packing_spec": packing_spec.to_dict(),
    }
    if collect_policy_noise_mask:
        local["policy_noise_mask"] = torch.cat(policy_mask_chunks, dim=0)
    if paired_core is not None:
        if not paired_candidate_chunks:
            raise RuntimeError(
                "paired adaptive pool collected no reference candidates"
            )
        local["paired_packed_event"] = torch.cat(
            paired_packed_chunks,
            dim=0,
        )
        local["paired_truth"] = torch.cat(paired_truth_chunks, dim=0)
        local["paired_current_candidates"] = torch.cat(
            paired_current_candidate_chunks,
            dim=0,
        )
        local["paired_candidates"] = torch.cat(
            paired_candidate_chunks,
            dim=0,
        )
    gathered = gather_pool_across_ranks(local, world_size=world_size)
    if quota_events is not None:
        stop = min(int(quota_events), int(gathered["truth"].shape[0]))
        for key in (
            "packed_event",
            "truth",
            "candidates",
            "policy_noise_mask",
        ):
            if key not in gathered:
                continue
            gathered[key] = gathered[key][:stop]
    paired_global_quota = (
        paired_reference_quota_events
        if paired_reference_quota_events is not None
        else quota_events
    )
    if paired_core is not None and paired_global_quota is not None:
        paired_stop = min(
            int(paired_global_quota),
            int(gathered["paired_truth"].shape[0]),
        )
        for key in (
            "paired_packed_event",
            "paired_truth",
            "paired_current_candidates",
            "paired_candidates",
        ):
            gathered[key] = gathered[key][:paired_stop]
    pool = AdaptiveOmniFoldPool(
        packed_event=gathered["packed_event"],
        truth=gathered["truth"],
        candidates=gathered["candidates"],
        packing_spec=EventPackingSpec.from_dict(gathered["packing_spec"]),
        policy_noise_mask=gathered.get("policy_noise_mask"),
    )
    if pool.n_events < 30:
        raise RuntimeError(
            f"adaptive OmniFold pool needs at least 30 valid events, got {pool.n_events}"
        )
    if paired_core is None:
        return pool
    paired_current_pool = AdaptiveOmniFoldPool(
        packed_event=gathered["paired_packed_event"],
        truth=gathered["paired_truth"],
        candidates=gathered["paired_current_candidates"],
        packing_spec=EventPackingSpec.from_dict(gathered["packing_spec"]),
    )
    reference_pool = AdaptiveOmniFoldPool(
        # Share the exact one-pass paired identities and common-noise draws.
        packed_event=gathered["paired_packed_event"],
        truth=gathered["paired_truth"],
        candidates=gathered["paired_candidates"],
        packing_spec=EventPackingSpec.from_dict(gathered["packing_spec"]),
    )
    if paired_current_pool.n_events < 30:
        raise RuntimeError(
            "classifier-trust pool needs at least 30 valid events, got "
            f"{paired_current_pool.n_events}"
        )
    return pool, paired_current_pool, reference_pool


def _materialize_raw_audit_training_pool(
    training_shard, loader_cfg, *, cfg, model, sampler, device,
    world_size, rank, num_ddim_steps, panel_seed, evaluation_events,
):
    """Share the exact original fold and generation seeds across both audit paths."""
    if cfg.audit_fit.get("training_population", "probe_split") != "omnifold_fold":
        return None
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _crossfit_repeat_seed

    fold = int(cfg.audit_fit.get("training_fold", 1))
    pool = _materialize_adaptive_omnifold_pool(
        training_shard, loader_cfg, model=model, sampler=sampler, device=device,
        world_size=world_size, rank=rank, quota_events=None,
        num_ddim_steps=num_ddim_steps, seed=int(panel_seed) + 2_000_003,
        include_pairwise_context=cfg.periodic_pair_features_enabled,
        include_visible_pair_rest_frame=cfg.visible_pair_rest_frame_enabled,
        training_crossfit_fold=(cfg.crossfit_folds, fold, _crossfit_repeat_seed(cfg.seed, 1)),
    )
    if rank == 0:
        _log.info(
            "[DGPO/omnifold] raw audit data: train=OmniFold repeat=1 fold=%s "
            "fit_events=%s external_validation_events=%s (50/50 early-stop/final-test); "
            "fresh current-policy K=1 samples, fresh classifier",
            fold, pool.n_events, evaluation_events,
        )
    return pool


@torch.no_grad()
def _validation_reward_sample_stats(rewards: Tensor, valid: Tensor) -> tuple[Tensor, float, int]:
    """Best per event plus all-sample sum/count; padded events never contribute."""
    selected = rewards[:, valid.reshape(-1).bool()]
    return selected.max(dim=0).values, float(selected.detach().double().sum().cpu().item()), selected.numel()


def run_validation_epoch(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    ema_save: Any | None,
    val_loader: Any | None,
    sampler: DDIMSampler,
    reward_agg: RewardAggregator,
    *,
    val_K: int,
    num_ddim_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    cartesian: bool,
    compute_winrate: bool,
    epoch: int | None = None,
    est_total_batches: int | None = None,
    val_log_batches: bool = True,
    val_rollout_parallel_chains: int = 1,
    val_tqdm_k_chains: bool = True,
    val_tqdm_ddim: bool = False,
    max_batches: int | None = None,
    initial_state: dict[str, np.ndarray] | None = None,
    full_diagnostics: bool = True,
    metric_prefix: str = "val",
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    """One pass over the validation dataset; all tensors under ``torch.no_grad()``.

    Validation uses ``val_K`` candidates per event (default **1** in config), independent of training ``K``.
    Typical setup: **one** current-policy DDIM sample per event; if ``compute_winrate``, **one**
    reference-policy DDIM sample per event for reward-based winrate (no extra multi-candidate rollout for val).

    Under Ray Train, every rank receives its own validation shard via
    ``ray.train.get_dataset_shard("validation")``.  Each rank iterates its own shard;
    a cross-rank ``all_reduce(MIN)`` on the "has-more" flag keeps DDP collectives
    synchronized (loop terminates as soon as **any** rank exhausts its shard).
    Accumulators are all-reduced at the end so every rank returns identical aggregates.

    When ``ema_save`` is set, candidate generation uses the save-EMA shadow; checkpoint
    ``state_dict`` still stores live trainable weights for resume.

    When ``val_log_batches`` is True, logs start/end timing per validation batch so long DDIM runs are
    not silent. ``val_tqdm_k_chains`` wraps the ``val_K`` sequential DDIM calls in a tqdm bar;
    ``val_tqdm_ddim`` adds an inner bar for each DDIM chain (verbose).

    If ``max_batches`` is set, stop after that many local batches per rank (partial val).

    ``full_diagnostics=False`` is the cheap monitoring tier: it computes only
    current-policy reward scalars/quantiles and skips the reference rollout,
    profile arrays, truth/pred matrices, images, and Ztautau/TARP diagnostics.
    ``metric_prefix`` keeps those deliberately lower-statistics numbers separate
    from the full ``val/*`` series.

    """
    is_rank0 = rank == 0
    model.eval()
    ref_model.eval()
    freeze_reference_model(ref_model)

    core = _unwrap_core_evenet(model)

    ep_str = f"epoch {epoch}" if epoch is not None else "val"
    if is_rank0:
        est_msg = (
            f"≈{est_total_batches} batches (ceil(val_events/batch_size))"
            if est_total_batches is not None and est_total_batches > 0
            else "unknown batch count (streaming)"
        )
        if max_batches is not None and max_batches > 0:
            est_msg = f"cap {max_batches} batches (partial val); full pass would be {est_msg}"
        if val_log_batches:
            _log.info("[DGPO] val: starting pass (%s, %s, %s GPUs).", ep_str, est_msg, world_size)

    t_epoch = time.perf_counter()
    n_val_batches = 0
    sum_r = 0.0
    cnt_r = 0
    sum_all_r = 0.0
    cnt_all_r = 0
    sum_win = 0.0
    cnt_win = 0

    local_reward_chunks: list[np.ndarray] = []
    local_reward_event_chunks: list[np.ndarray] = []
    local_response_class_chunks: list[np.ndarray] = []
    ztautau_cfg = getattr(global_config, "ztautau_domain", None)
    dg_cfg = getattr(global_config, "dgpo", {})
    ztautau_panel_cfg = _dgpo_cfg_get(dg_cfg, "ztautau_metrics", {})
    tarp_cfg = _dgpo_cfg_get(dg_cfg, "tarp", {})
    compact_wandb = _wandb_simplified_enabled()
    ztautau_metrics_enabled = bool(
        full_diagnostics
        and ztautau_cfg is not None
        and _dgpo_cfg_get(ztautau_cfg, "enabled", False)
        and tuple(_dgpo_cfg_get(ztautau_cfg, "feature_names", ())) == ("theta", "phi")
        and (
            _dgpo_cfg_get(ztautau_panel_cfg, "enabled", False)
            or _dgpo_cfg_get(tarp_cfg, "enabled", False)
        )
    )
    local_ztautau_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    legacy_kinematics = bool(full_diagnostics) and _supports_legacy_invisible_kinematics(
        cartesian=cartesian,
        feature_dim=len(_invisible_feature_names()) or None,
    )
    winrate_enabled = _validation_winrate_enabled(
        compute_winrate=compute_winrate,
        cartesian=cartesian,
        feature_dim=len(_invisible_feature_names()) or None,
    )
    profile_feature_names = (
        tuple(_validation_profile_feature_names(cartesian=cartesian))
        if full_diagnostics
        else ()
    )
    local_pt_delta_event_mean_chunks: list[np.ndarray] = []
    local_profile_chunks: dict[str, list[np.ndarray]] = {
        f"{profile_name}_truth": []
        for profile_name in profile_feature_names
    }
    local_profile_chunks.update({
        f"{profile_name}_delta": []
        for profile_name in profile_feature_names
    })
    all_plot_feature_names = (
        _generation_monitor_feature_names(cartesian=cartesian)
        if full_diagnostics
        else ()
    )
    local_truth_pred_all_chunks: dict[str, list[np.ndarray]] = {
        f"{feature_name}_{suffix}": []
        for feature_name in all_plot_feature_names
        for suffix in ("truth", "pred", "ref")
    }
    local_truth_pred_class_chunks: list[np.ndarray] = []
    # pT in GeV (original physics scale, after expm1 inversion of log1p).
    bin_pt_edges = _diagnostic_bin_edges("pt")
    bin_eta_edges = _diagnostic_bin_edges("eta")
    bin_phi_edges = _diagnostic_bin_edges("phi")
    # px/py track neutrino pT (rarely beyond a few hundred GeV); pz ~ pt*sinh(eta) has a much
    # wider longitudinal spread, so it needs a larger range or the histogram clips to ~empty.
    bin_px_edges = _diagnostic_bin_edges("px")
    bin_py_edges = _diagnostic_bin_edges("py")
    bin_pz_edges = _diagnostic_bin_edges("pz")
    bin_wmass_edges = _diagnostic_bin_edges("wmass")
    bin_topmass_edges = _diagnostic_bin_edges("topmass")
    num_diag_bins = len(bin_pt_edges) - 1
    # _p = current policy, _t = truth, _r = frozen reference policy
    h_pt_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_pt_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_pt_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_e_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_e_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_e_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_p_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_p_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_p_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_x_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_x_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_x_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_y_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_y_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_y_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_z_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_z_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_z_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_wm_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_wm_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_wm_r = np.zeros(num_diag_bins, dtype=np.float64)
    h_tm_p = np.zeros(num_diag_bins, dtype=np.float64)
    h_tm_t = np.zeros(num_diag_bins, dtype=np.float64)
    h_tm_r = np.zeros(num_diag_bins, dtype=np.float64)

    val_iter = iter(val_loader) if val_loader is not None else None
    if val_iter is None:
        if is_rank0 and val_log_batches:
            _log.warning("[DGPO] val: rank=%s has no val shard; returning empty metrics.", rank)
        empty_metrics = {
            f"{metric_prefix}/reward/mean": float("nan"),
            f"{metric_prefix}/reward/all_sample_mean": float("nan"),
            f"{metric_prefix}/reward/best_of_k_mean": float("nan"),
            f"{metric_prefix}/reward/median": float("nan"),
            f"{metric_prefix}/reward/p10": float("nan"),
            f"{metric_prefix}/reward/p30": float("nan"),
            f"{metric_prefix}/reward/p70": float("nan"),
            f"{metric_prefix}/reward/p90": float("nan"),
        }
        if winrate_enabled:
            empty_metrics[f"{metric_prefix}/winrate"] = float("nan")
        return empty_metrics

    batch_round = 0
    while True:
        batch_cpu, has_more = _next_batch_synced(
            val_iter, world_size=world_size, device=device
        )
        if not has_more or batch_cpu is None:
            break

        batch_round += 1
        n_val_batches += 1
        batch_d = batch_to_device(batch_cpu, device)
        B = int(batch_d["x"].shape[0])

        if is_rank0 and val_log_batches:
            batch_suffix = (
                f" {batch_round}/{est_total_batches}"
                if est_total_batches is not None and est_total_batches > 0
                else f" {batch_round}"
            )
            _log.info(
                "[DGPO] val round%s: B=%s×%s GPUs | current policy: val_K=%s DDIM chains (%s steps each)...",
                batch_suffix,
                B,
                world_size,
                val_K,
                num_ddim_steps,
            )

        buf = _maybe_install_ema_for_generation(ema_save, model, core)
        t_gen = time.perf_counter()
        chain_desc = f"val DDIM ({ep_str})"
        try:
            candidates = generate_neutrino_candidates(
                core,
                batch_d,
                sampler,
                K=val_K,
                num_ddim_steps=num_ddim_steps,
                device=device,
                parallel_chains=val_rollout_parallel_chains,
                tqdm_k_chains=val_tqdm_k_chains and is_rank0,
                use_tqdm_ddim=val_tqdm_ddim and is_rank0,
                chain_progress_desc=chain_desc,
            )
        finally:
            if buf:
                _restore_trainable_weights(model, buf)

        t_after_cur = time.perf_counter()
        if is_rank0 and val_log_batches:
            _log.info(
                "[DGPO] val round%s: generation done in %.1fs.",
                batch_suffix,
                t_after_cur - t_gen,
            )

        rewards, _ = reward_agg.compute(candidates, batch_d)

        valid = get_event_valid_mask(batch_d, B, device, dtype).reshape(-1)
        m_sel = valid > 0
        vb = m_sel

        if bool(vb.any().item()):
            r_per_event, all_sum, all_count = _validation_reward_sample_stats(rewards, vb)
            sum_all_r += all_sum
            cnt_all_r += all_count
            sum_r += float(r_per_event.sum().detach().cpu().item())
            cnt_r += int(vb.sum().item())
            r_per_event_np = r_per_event.detach().float().cpu().numpy()
            local_reward_chunks.append(r_per_event_np)
            local_reward_event_chunks.append(r_per_event_np)
            classification = batch_d.get("classification")
            if full_diagnostics and isinstance(classification, Tensor):
                if int(classification.numel()) != B:
                    raise ValueError(
                        "classification must contain one process ID per validation event"
                    )
                local_response_class_chunks.append(
                    classification.reshape(B)[vb]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False)
                )

        # The cheap tier deliberately stops here: no reference DDIM, no large
        # per-event CPU arrays, no cross-rank plot gather, and no TARP.  A caller
        # requesting win-rate still falls through because that scalar needs the
        # reference-policy K=1 rollout.
        if not full_diagnostics and not winrate_enabled:
            if max_batches is not None and max_batches > 0 and batch_round >= max_batches:
                if is_rank0 and val_log_batches:
                    _log.info(
                        "[DGPO] cheap val: stopping at validation_cheap_max_batches=%s.",
                        max_batches,
                    )
                break
            continue
        k_sel = _kin_hist_candidate_indices_per_event(
            rewards, candidates, batch_d, cartesian=cartesian
        )
        selected_delta_arrays = _val_selected_delta_arrays(
            candidates,
            batch_d,
            k_sel,
            cartesian=cartesian,
            device=device,
            dtype=dtype,
        )
        for key in local_profile_chunks:
            local_profile_chunks[key].append(
                selected_delta_arrays.get(key, np.array([], dtype=np.float64))
            )
        if "pt_delta_event_mean" in selected_delta_arrays:
            local_pt_delta_event_mean_chunks.append(
                selected_delta_arrays["pt_delta_event_mean"]
            )
        if not cartesian:
            feature_arrays = _val_pred_truth_feature_flat_all_candidates(
                candidates,
                batch_d,
                feature_names=all_plot_feature_names,
                device=device,
            )
            for key, values in feature_arrays.items():
                local_truth_pred_all_chunks[key].append(values)
            if feature_arrays:
                local_truth_pred_class_chunks.append(
                    _val_class_index_flat_all_candidates(
                        candidates,
                        batch_d,
                        device=device,
                    )
                )

        if legacy_kinematics:
            ppt, peta, pphi, tpt, teta, tphi = _val_pred_truth_kin_flat(
                candidates,
                batch_d,
                k_sel,
                cartesian=cartesian,
                device=device,
            )
            h_pt_p += np.histogram(ppt, bins=bin_pt_edges)[0]
            h_pt_t += np.histogram(tpt, bins=bin_pt_edges)[0]
            h_e_p += np.histogram(peta, bins=bin_eta_edges)[0]
            h_e_t += np.histogram(teta, bins=bin_eta_edges)[0]
            h_p_p += np.histogram(pphi, bins=bin_phi_edges)[0]
            h_p_t += np.histogram(tphi, bins=bin_phi_edges)[0]
            if cartesian:
                all_px_p, all_py_p, all_pz_p, all_px_t, all_py_t, all_pz_t = (
                    _val_pred_truth_cartesian_flat_all_candidates(
                        candidates,
                        batch_d,
                        device=device,
                        dtype=dtype,
                    )
                )
                local_truth_pred_all_chunks["px_truth"].append(all_px_t)
                local_truth_pred_all_chunks["px_pred"].append(all_px_p)
                local_truth_pred_all_chunks["py_truth"].append(all_py_t)
                local_truth_pred_all_chunks["py_pred"].append(all_py_p)
                local_truth_pred_all_chunks["pz_truth"].append(all_pz_t)
                local_truth_pred_all_chunks["pz_pred"].append(all_pz_p)
                local_truth_pred_class_chunks.append(
                    _val_class_index_flat_all_candidates(
                        candidates,
                        batch_d,
                        device=device,
                    )
                )

            ppx, ppy, ppz, tpx, tpy, tpz = _val_pred_truth_cartesian_flat(
                candidates,
                batch_d,
                k_sel,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            h_x_p += np.histogram(ppx, bins=bin_px_edges)[0]
            h_x_t += np.histogram(tpx, bins=bin_px_edges)[0]
            h_y_p += np.histogram(ppy, bins=bin_py_edges)[0]
            h_y_t += np.histogram(tpy, bins=bin_py_edges)[0]
            h_z_p += np.histogram(ppz, bins=bin_pz_edges)[0]
            h_z_t += np.histogram(tpz, bins=bin_pz_edges)[0]

            b_idx = torch.arange(B, device=device)
            pred_nu_kin = candidates[k_sel, b_idx][:, :2, :3]
            truth_nu_kin = _truth_invisible_kin_phys(
                batch_d, cartesian=cartesian, device=device, dtype=dtype
            )[:, :2, :]
            w_p, top_p = _val_mass_reconstruction_masses(
                batch_d,
                pred_nu_kin,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            w_t, top_t = _val_mass_reconstruction_masses(
                batch_d,
                truth_nu_kin,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            if w_p.size:
                h_wm_p += np.histogram(w_p, bins=bin_wmass_edges)[0]
            if top_p.size:
                h_tm_p += np.histogram(top_p, bins=bin_topmass_edges)[0]
            if w_t.size:
                h_wm_t += np.histogram(w_t, bins=bin_wmass_edges)[0]
            if top_t.size:
                h_tm_t += np.histogram(top_t, bins=bin_topmass_edges)[0]

        # Always run one ref-policy DDIM pass (K=1) for val_neutrino overlays; reuse for winrate.
        ref_core = _unwrap_core_evenet(ref_model)
        if is_rank0 and val_log_batches:
            _log.info("[DGPO] val round%s: reference policy DDIM (K=1)...", batch_suffix)
        t_ref = time.perf_counter()
        r_one = generate_neutrino_candidates(
            ref_core,
            batch_d,
            sampler,
            K=1,
            num_ddim_steps=num_ddim_steps,
            device=device,
            parallel_chains=1,
            tqdm_k_chains=False,
            use_tqdm_ddim=val_tqdm_ddim and is_rank0,
            chain_progress_desc=f"val ref DDIM ({ep_str})",
        )
        if is_rank0 and val_log_batches:
            _log.info(
                "[DGPO] val round%s: ref DDIM done in %.1fs.",
                batch_suffix,
                time.perf_counter() - t_ref,
            )
        k_sel_ref = torch.zeros(B, dtype=torch.long, device=device)
        if legacy_kinematics:
            rpt, reta, rphi, _, _, _ = _val_pred_truth_kin_flat(
                r_one, batch_d, k_sel_ref, cartesian=cartesian, device=device
            )
            h_pt_r += np.histogram(rpt, bins=bin_pt_edges)[0]
            h_e_r += np.histogram(reta, bins=bin_eta_edges)[0]
            h_p_r += np.histogram(rphi, bins=bin_phi_edges)[0]

            rpx, rpy, rpz, _, _, _ = _val_pred_truth_cartesian_flat(
                r_one,
                batch_d,
                k_sel_ref,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            h_x_r += np.histogram(rpx, bins=bin_px_edges)[0]
            h_y_r += np.histogram(rpy, bins=bin_py_edges)[0]
            h_z_r += np.histogram(rpz, bins=bin_pz_edges)[0]

        if legacy_kinematics:
            ref_nu_kin = r_one[k_sel_ref, b_idx][:, :2, :3]
            w_r, top_r = _val_mass_reconstruction_masses(
                batch_d,
                ref_nu_kin,
                cartesian=cartesian,
                device=device,
                dtype=dtype,
            )
            if w_r.size:
                h_wm_r += np.histogram(w_r, bins=bin_wmass_edges)[0]
            if top_r.size:
                h_tm_r += np.histogram(top_r, bins=bin_topmass_edges)[0]

        if not cartesian:
            ref_feature_arrays = _val_pred_truth_feature_flat_all_candidates(
                r_one,
                batch_d,
                feature_names=all_plot_feature_names,
                device=device,
            )
            append_reference_prediction_arrays(
                local_truth_pred_all_chunks,
                ref_feature_arrays,
            )
        elif legacy_kinematics:
            all_px_r, all_py_r, all_pz_r, _, _, _ = _val_pred_truth_cartesian_flat_all_candidates(
                r_one,
                batch_d,
                device=device,
                dtype=dtype,
            )
            local_truth_pred_all_chunks["px_ref"].append(all_px_r)
            local_truth_pred_all_chunks["py_ref"].append(all_py_r)
            local_truth_pred_all_chunks["pz_ref"].append(all_pz_r)

        if ztautau_metrics_enabled:
            ztautau_batch_arrays = collect_ztautau_validation_arrays(
                candidates,
                r_one,
                batch_d,
                valid,
            )
            for key, values in ztautau_batch_arrays.items():
                local_ztautau_chunks[key].append(values)

        if winrate_enabled:
            rewards_ref, _ = reward_agg.compute(r_one, batch_d)
            r_cur = rewards.max(dim=0).values
            r_ref = rewards_ref[0]
            wins = (r_cur > r_ref) & m_sel & torch.isfinite(r_cur) & torch.isfinite(r_ref)
            w = wins.float().sum()
            nw = m_sel.sum()
            sum_win += float(w.detach().cpu().item())
            cnt_win += int(nw.detach().cpu().item())

        if max_batches is not None and max_batches > 0 and batch_round >= max_batches:
            if is_rank0 and val_log_batches:
                _log.info(
                    "[DGPO] val: stopping early at validation_max_batches=%s (partial val metrics).",
                    max_batches,
                )
            break

    if is_rank0 and val_log_batches:
        _log.info(
            "[DGPO] val: finished %s batches (%s rounds × %s GPUs) in %.1fs.",
            n_val_batches * world_size,
            n_val_batches,
            world_size,
            time.perf_counter() - t_epoch,
        )

    # All-reduce accumulators so every rank has the global totals.
    if world_size > 1:
        acc = torch.tensor(
            [sum_r, cnt_r, sum_win, cnt_win, sum_all_r, cnt_all_r],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
        a = acc.cpu().tolist()
        sum_r, cnt_r = a[0], int(a[1])
        sum_win, cnt_win = a[2], int(a[3])
        sum_all_r, cnt_all_r = a[4], int(a[5])

    hist_stack = np.stack(
        [
            h_pt_p, h_pt_t, h_pt_r,
            h_e_p, h_e_t, h_e_r,
            h_p_p, h_p_t, h_p_r,
            h_x_p, h_x_t, h_x_r,
            h_y_p, h_y_t, h_y_r,
            h_z_p, h_z_t, h_z_r,
            h_wm_p, h_wm_t, h_wm_r,
            h_tm_p, h_tm_t, h_tm_r,
        ]
    )
    t_hist = torch.from_numpy(hist_stack).to(device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(t_hist, op=dist.ReduceOp.SUM)
    hist_merged = t_hist.cpu().numpy()
    (
        h_pt_p, h_pt_t, h_pt_r,
        h_e_p, h_e_t, h_e_r,
        h_p_p, h_p_t, h_p_r,
        h_x_p, h_x_t, h_x_r,
        h_y_p, h_y_t, h_y_r,
        h_z_p, h_z_t, h_z_r,
        h_wm_p, h_wm_t, h_wm_r,
        h_tm_p, h_tm_t, h_tm_r,
    ) = [hist_merged[i] for i in range(24)]


    p10 = p30 = p50 = p70 = p90 = float("nan")
    if world_size > 1:
        if rank == 0:
            gathered: list[Any] = [None] * world_size
            dist.gather_object(
                local_reward_chunks,
                object_gather_list=gathered,
                dst=0,
            )
            merged_list: list[np.ndarray] = []
            for part in gathered:
                if part:
                    merged_list.extend(part)
            merged_r = (
                np.concatenate(merged_list, axis=0)
                if merged_list
                else np.array([], dtype=np.float64)
            )
            if merged_r.size > 0:
                p10, p30, p50, p70, p90 = [
                    float(x) for x in np.nanpercentile(merged_r, [10, 30, 50, 70, 90])
                ]
        else:
            dist.gather_object(local_reward_chunks, dst=0)
        pct_t = torch.tensor(
            [p10, p30, p50, p70, p90], dtype=torch.float64, device=device
        )
        dist.broadcast(pct_t, src=0)
        p10, p30, p50, p70, p90 = [float(x) for x in pct_t.cpu().tolist()]
    else:
        merged_list = local_reward_chunks
        merged_r = (
            np.concatenate(merged_list, axis=0)
            if merged_list
            else np.array([], dtype=np.float64)
        )
        if merged_r.size > 0:
            p10, p30, p50, p70, p90 = [
                float(x) for x in np.nanpercentile(merged_r, [10, 30, 50, 70, 90])
            ]

    def _mean(num: float, den: int) -> float:
        return float(num / den) if den > 0 else float("nan")

    win_metric = _mean(sum_win, cnt_win) if winrate_enabled else float("nan")

    local_state = {
        "reward": _concat_np_chunks(local_reward_event_chunks),
        "pt_delta_mean": _concat_np_chunks(local_pt_delta_event_mean_chunks),
        "class_index": _concat_np_chunks(local_response_class_chunks).astype(
            np.int64,
            copy=False,
        ),
    }
    for profile_name in profile_feature_names:
        local_state[f"{profile_name}_truth"] = _concat_np_chunks(
            local_profile_chunks[f"{profile_name}_truth"]
        )
        local_state[f"{profile_name}_delta"] = _concat_np_chunks(
            local_profile_chunks[f"{profile_name}_delta"]
        )
    profile_compare_local = {}
    for profile_name in profile_feature_names:
        profile_compare_local[f"{profile_name}_truth"] = local_state[f"{profile_name}_truth"]
        profile_compare_local[f"{profile_name}_delta"] = local_state[f"{profile_name}_delta"]
    if initial_state is not None:
        for profile_name in profile_feature_names:
            for suffix in ("truth", "delta"):
                key = f"{profile_name}_{suffix}"
                profile_compare_local[f"initial_{key}"] = np.asarray(
                    initial_state.get(key, np.array([], dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
    profile_merged = _gather_val_array_dict(
        profile_compare_local, rank=rank, world_size=world_size
    )

    response_initial_state = initial_state
    if response_initial_state is None and epoch == -1:
        # The baseline pass should still create the W&B image series; later epochs
        # then update the same key with initial-vs-current heatmaps.
        response_initial_state = local_state

    response_merged: dict[str, np.ndarray] = {}
    if response_initial_state is not None:
        init_reward = np.asarray(
            response_initial_state.get("reward", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        init_pt_delta = np.asarray(
            response_initial_state.get("pt_delta_mean", np.array([], dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        n_reward = min(init_reward.size, local_state["reward"].size)
        n_pt = min(init_pt_delta.size, local_state["pt_delta_mean"].size)
        response_class_index = np.asarray(
            local_state.get("class_index", np.array([], dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        response_merged = _gather_val_array_dict(
            {
                "reward_initial": init_reward[:n_reward],
                "reward_current": local_state["reward"][:n_reward],
                "reward_class_index": response_class_index[:n_reward],
                "pt_delta_initial": init_pt_delta[:n_pt],
                "pt_delta_current": local_state["pt_delta_mean"][:n_pt],
            },
            rank=rank,
            world_size=world_size,
        )
    truth_pred_all_merged = _gather_val_array_dict(
        {
            **{
                key: _concat_np_chunks(chunks)
                for key, chunks in local_truth_pred_all_chunks.items()
            },
            "class_index": _concat_np_chunks(
                local_truth_pred_class_chunks
            ).astype(np.int64, copy=False),
        },
        rank=rank,
        world_size=world_size,
    )
    local_ztautau_arrays = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in local_ztautau_chunks.items()
        if chunks
    }
    ztautau_arrays_merged = _gather_val_ndarray_dict(
        local_ztautau_arrays,
        rank=rank,
        world_size=world_size,
    )

    out: dict[str, Any] = {
        f"{metric_prefix}/reward/mean": _mean(sum_r, cnt_r),
        f"{metric_prefix}/reward/all_sample_mean": _mean(sum_all_r, cnt_all_r),
        f"{metric_prefix}/reward/best_of_k_mean": _mean(sum_r, cnt_r),
        f"{metric_prefix}/reward/median": p50,
        f"{metric_prefix}/reward/p10": p10,
        f"{metric_prefix}/reward/p30": p30,
        f"{metric_prefix}/reward/p70": p70,
        f"{metric_prefix}/reward/p90": p90,
        f"{metric_prefix}/meta/K": float(val_K),
        f"{metric_prefix}/meta/batches_per_rank": float(n_val_batches),
        f"{metric_prefix}/meta/full_diagnostics": float(full_diagnostics),
        "_val_initial_state": local_state,
    }
    if winrate_enabled:
        out[f"{metric_prefix}/winrate"] = win_metric

    _val_kin_suffix = f"val: {val_K} candidate{'s' if val_K != 1 else ''} vs truth"
    _pred_lbl = "Pred (val)" if val_K == 1 else f"Pred (val, best-of-{val_K})"
    if is_rank0:
        for profile_name in profile_feature_names:
            truth_key = f"{profile_name}_truth"
            delta_key = f"{profile_name}_delta"
            truth_arr = profile_merged.get(truth_key, np.array([], dtype=np.float64))
            delta_arr = profile_merged.get(delta_key, np.array([], dtype=np.float64))
            slope, zero_point = _profile_fit_metrics(
                profile_name, truth_arr, delta_arr
            )
            finite_delta = delta_arr[np.isfinite(delta_arr)]
            delta_mean = float(np.mean(finite_delta)) if finite_delta.size > 0 else float("nan")
            out[f"val_diagnostics/profile/{profile_name}/delta_mean"] = delta_mean
            out[f"val_diagnostics/profile/{profile_name}/slope"] = slope
            out[f"val_diagnostics/profile/{profile_name}/zero_delta_truth"] = zero_point
            if not compact_wandb:
                out[
                    f"val_diagnostics/profile/{profile_name}_delta_vs_truth_{profile_name}"
                ] = _validation_delta_profile_figure(
                    truth_arr,
                    delta_arr,
                    profile_name=profile_name,
                    title=f"Validation {profile_name} residual vs truth {profile_name}",
                    truth_initial=profile_merged.get(f"initial_{truth_key}"),
                    delta_initial=profile_merged.get(f"initial_{delta_key}"),
                )
        if response_initial_state is not None and not _wandb_critical_enabled():
            reward_initial = response_merged.get(
                "reward_initial", np.array([], dtype=np.float64)
            )
            reward_current = response_merged.get(
                "reward_current", np.array([], dtype=np.float64)
            )
            out["val/response/reward_initial_vs_current"] = _response_matrix_figure(
                reward_initial,
                reward_current,
                xlabel="Initial validation reward",
                ylabel="Current validation reward",
                title="Validation 2D correlation: initial reward vs current reward",
            )
            for metric_name, metric_value in _truth_pred_scalar_metrics(
                reward_initial,
                reward_current,
            ).items():
                out[f"val/response/metrics/reward/{metric_name}"] = metric_value

            reward_class_index = np.asarray(
                response_merged.get(
                    "reward_class_index", np.array([], dtype=np.int64)
                ),
                dtype=np.int64,
            ).reshape(-1)
            if reward_class_index.size not in {0, reward_initial.size}:
                raise ValueError(
                    "validation response process IDs lost event alignment: "
                    f"classes={reward_class_index.size}, reward={reward_initial.size}"
                )
            if reward_class_index.size:
                for process_id, process_name in enumerate(
                    _event_signal_class_names()
                ):
                    process_mask = reward_class_index == int(process_id)
                    if not np.any(process_mask):
                        continue
                    process_initial = reward_initial[process_mask]
                    process_current = reward_current[process_mask]
                    process_prefix = f"val/response/by_process/{process_name}"
                    out[f"{process_prefix}/reward_initial_vs_current"] = (
                        _response_matrix_figure(
                            process_initial,
                            process_current,
                            xlabel="Initial validation reward",
                            ylabel="Current validation reward",
                            title=(
                                "Validation reward response: initial vs current "
                                f"({process_name})"
                            ),
                        )
                    )
                    for metric_name, metric_value in _truth_pred_scalar_metrics(
                        process_initial,
                        process_current,
                    ).items():
                        out[
                            f"{process_prefix}/metrics/reward/{metric_name}"
                        ] = metric_value
            if (
                response_merged.get("pt_delta_initial", np.array([], dtype=np.float64)).size > 0
                or response_merged.get("pt_delta_current", np.array([], dtype=np.float64)).size > 0
            ):
                pt_delta_initial = response_merged.get(
                    "pt_delta_initial", np.array([], dtype=np.float64)
                )
                pt_delta_current = response_merged.get(
                    "pt_delta_current", np.array([], dtype=np.float64)
                )
                out["val/response/pt_delta_mean_initial_vs_current"] = (
                    _response_matrix_figure(
                        pt_delta_initial,
                        pt_delta_current,
                        xlabel="Initial event mean delta pT [GeV]",
                        ylabel="Current event mean delta pT [GeV]",
                        title="Validation 2D correlation: initial vs current event mean delta pT",
                    )
                )
                for metric_name, metric_value in _truth_pred_scalar_metrics(
                    pt_delta_initial,
                    pt_delta_current,
                ).items():
                    out[
                        f"val/response/metrics/pt_delta_mean/{metric_name}"
                    ] = metric_value
        available_truth_pred_features = _available_truth_pred_features(
            truth_pred_all_merged,
            all_plot_feature_names,
        )
        if _wandb_critical_enabled():
            available_truth_pred_features = []
        for feature_name in available_truth_pred_features:
            truth_key = f"{feature_name}_truth"
            pred_key = f"{feature_name}_pred"
            feature_truth = truth_pred_all_merged.get(
                truth_key, np.array([], dtype=np.float64)
            )
            feature_pred = truth_pred_all_merged.get(
                pred_key, np.array([], dtype=np.float64)
            )
            out[f"val_neutrino/all/{feature_name}_truth_vs_pred"] = (
                _truth_pred_matrix_figure(
                    feature_truth,
                    feature_pred,
                    xlabel=f"Truth {feature_name}",
                    ylabel=f"Pred {feature_name}",
                    title=(
                        f"Validation 2D truth vs pred {feature_name} "
                        f"({val_K} candidate{'s' if val_K != 1 else ''}, all)"
                    ),
                    bin_edges=_generation_special_bin_edges(feature_name),
                )
            )
            for metric_name, metric_value in _truth_pred_scalar_metrics(
                feature_truth,
                feature_pred,
            ).items():
                out[f"val_neutrino/all_metrics/{feature_name}/{metric_name}"] = metric_value

            class_index = np.asarray(
                truth_pred_all_merged.get(
                    "class_index", np.array([], dtype=np.int64)
                ),
                dtype=np.int64,
            ).reshape(-1)
            if class_index.size not in {0, feature_truth.size}:
                raise ValueError(
                    "validation response process IDs lost candidate alignment: "
                    f"classes={class_index.size}, feature={feature_truth.size}"
                )
            if class_index.size:
                for process_id, process_name in enumerate(
                    _event_signal_class_names()
                ):
                    process_truth, process_pred = select_truth_pred_by_class(
                        feature_truth,
                        feature_pred,
                        class_index,
                        class_id=int(process_id),
                    )
                    if process_truth.size == 0:
                        continue
                    process_prefix = (
                        f"val_neutrino/by_process/{process_name}/{feature_name}"
                    )
                    out[f"{process_prefix}_truth_vs_pred"] = (
                        _truth_pred_matrix_figure(
                            process_truth,
                            process_pred,
                            xlabel=f"Truth {feature_name}",
                            ylabel=f"Pred {feature_name}",
                            title=(
                                f"Validation response {feature_name} "
                                f"({process_name}, all candidates)"
                            ),
                            bin_edges=_generation_special_bin_edges(feature_name),
                        )
                    )
                    for metric_name, metric_value in _truth_pred_scalar_metrics(
                        process_truth,
                        process_pred,
                    ).items():
                        out[
                            f"val_neutrino/by_process/{process_name}/metrics/"
                            f"{feature_name}/{metric_name}"
                        ] = metric_value
            bin_edges = _generation_special_bin_edges(feature_name)
            out[f"val_neutrino/jsd/current/{feature_name}"] = _array_histogram_jsd(
                truth_pred_all_merged.get(truth_key, np.array([], dtype=np.float64)),
                truth_pred_all_merged.get(pred_key, np.array([], dtype=np.float64)),
                bin_edges=bin_edges,
            )
            out[f"val_neutrino/jsd/ref/{feature_name}"] = _array_histogram_jsd(
                truth_pred_all_merged.get(truth_key, np.array([], dtype=np.float64)),
                truth_pred_all_merged.get(f"{feature_name}_ref", np.array([], dtype=np.float64)),
                bin_edges=bin_edges,
            )
        if legacy_kinematics and not compact_wandb:
            out["val_neutrino/pt"] = _val_overlay_kin_figure(
                h_pt_t,
                h_pt_p,
                bin_pt_edges,
                f"Neutrino pT [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_pt_r,
                xlabel="pT [GeV]",
            )
            out["val_neutrino/eta"] = _val_overlay_kin_figure(
                h_e_t,
                h_e_p,
                bin_eta_edges,
                f"Neutrino η ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_e_r,
                xlabel="η",
            )
            out["val_neutrino/phi"] = _val_overlay_kin_figure(
                h_p_t,
                h_p_p,
                bin_phi_edges,
                f"Neutrino φ ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_p_r,
                xlabel="φ [rad]",
            )
            out["val_neutrino/px"] = _val_overlay_kin_figure(
                h_x_t,
                h_x_p,
                bin_px_edges,
                f"Neutrino p_x [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_x_r,
                xlabel="p_x [GeV]",
            )
            out["val_neutrino/py"] = _val_overlay_kin_figure(
                h_y_t,
                h_y_p,
                bin_py_edges,
                f"Neutrino p_y [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_y_r,
                xlabel="p_y [GeV]",
            )
            out["val_neutrino/pz"] = _val_overlay_kin_figure(
                h_z_t,
                h_z_p,
                bin_pz_edges,
                f"Neutrino p_z [GeV] ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_z_r,
                xlabel="p_z [GeV]",
            )
            out["val_neutrino/jsd/current/pt"] = _histogram_jsd(h_pt_t, h_pt_p)
            out["val_neutrino/jsd/current/eta"] = _histogram_jsd(h_e_t, h_e_p)
            out["val_neutrino/jsd/current/phi"] = _histogram_jsd(h_p_t, h_p_p)
            out["val_neutrino/jsd/current/px"] = _histogram_jsd(h_x_t, h_x_p)
            out["val_neutrino/jsd/current/py"] = _histogram_jsd(h_y_t, h_y_p)
            out["val_neutrino/jsd/current/pz"] = _histogram_jsd(h_z_t, h_z_p)
            out["val_neutrino/jsd/ref/pt"] = _histogram_jsd(h_pt_t, h_pt_r)
            out["val_neutrino/jsd/ref/eta"] = _histogram_jsd(h_e_t, h_e_r)
            out["val_neutrino/jsd/ref/phi"] = _histogram_jsd(h_p_t, h_p_r)
            out["val_neutrino/jsd/ref/px"] = _histogram_jsd(h_x_t, h_x_r)
            out["val_neutrino/jsd/ref/py"] = _histogram_jsd(h_y_t, h_y_r)
            out["val_neutrino/jsd/ref/pz"] = _histogram_jsd(h_z_t, h_z_r)
            out["val_mass/w_mass"] = _val_overlay_kin_figure(
                h_wm_t,
                h_wm_p,
                bin_wmass_edges,
                f"W mass reconstruction vs truth resonance ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_wm_r,
                xlabel="W mass [GeV]",
            )
            out["val_mass/top_mass"] = _val_overlay_kin_figure(
                h_tm_t,
                h_tm_p,
                bin_topmass_edges,
                f"Top mass reconstruction vs truth resonance ({_val_kin_suffix})",
                pred_label=_pred_lbl,
                counts_ref=h_tm_r,
                xlabel="Top mass [GeV]",
            )
            out["val_mass/jsd/current/w_mass"] = _histogram_jsd(h_wm_t, h_wm_p)
            out["val_mass/jsd/current/top_mass"] = _histogram_jsd(h_tm_t, h_tm_p)
            out["val_mass/jsd/ref/w_mass"] = _histogram_jsd(h_wm_t, h_wm_r)
            out["val_mass/jsd/ref/top_mass"] = _histogram_jsd(h_tm_t, h_tm_r)
        if ztautau_metrics_enabled:
            out.update(
                build_ztautau_validation_metrics(
                    ztautau_arrays_merged,
                    val_k=val_K,
                    tarp_config=tarp_cfg,
                    metrics_config=ztautau_panel_cfg,
                    include_images=bool(
                        _dgpo_cfg_get(
                            ztautau_panel_cfg,
                            "log_images",
                            True,
                        )
                    ),
                )
            )
    return out


def _prepare_single_pool_datasets(
    *, base_dir: Path, base_val_dir: Path | None, process_fn: Any,
    platform_info: Any, dataset_options: Any,
) -> tuple[Any, Any, int, int]:
    """Read one complete budgeted pool; classifiers split it after generation.

    Policy fitting uses all rows. Physics 'validation' is explicitly in-pool
    monitoring, not an independent test set. Classifier validation is disjoint
    from classifier fitting via a fixed condition hash inside the fit helpers.
    """
    if base_val_dir is not None and base_val_dir.resolve() != base_dir.resolve():
        raise ValueError("single-pool mode forbids an external validation directory")
    if float(_dgpo_cfg_get(dataset_options, "dataset_limit", 1.0)) != 1.0 or float(
        _dgpo_cfg_get(dataset_options, "val_dataset_limit", 1.0)
    ) != 1.0:
        raise ValueError("single-pool mode requires dataset_limit=val_dataset_limit=1.0")
    files = sorted(map(str, base_dir.glob("*.parquet")))
    if not files:
        raise ValueError(f"No parquet files found in the budgeted pool: {base_dir}")
    pool, count = register_dataset(files, process_fn, platform_info, dataset_limit=1.0, file_shuffling=True)
    _log.warning(
        "[DGPO/scaling] SINGLE POOL: %s (%s events); policy/OmniFold/monitors "
        "use only these identities. Classifiers use internal 80/20 splits; "
        "physics validation panels are in-pool diagnostics, NOT external generalization.",
        base_dir, count,
    )
    return pool, pool, int(count), int(count)


def _should_log_pretraining_baseline(start_epoch: int, global_step: int) -> bool:
    """A mid-epoch-0 resume is not the untrained step-zero policy."""
    return int(start_epoch) == 0 and int(global_step) == 0


def _clear_unused_raw_monitor_baseline(state: Any, cfg: Any) -> bool:
    """Honor disabled audits when inheriting a pre-audit bootstrap snapshot.

    Clear only the pending diagnostic; installed reward/reference and counters
    are untouched. Required controller baselines retain their previous path.
    """
    skip = (
        cfg.fixed_schedule_skip_staleness_audit and not cfg.fixed_schedule_log_raw_audit
    ) or (
        cfg.log_only and not cfg.raw_audit_enabled and not cfg.baseline_probe_on_start
    )
    if state.raw_monitor_baseline_pending and skip:
        state.raw_monitor_baseline_pending = False
        return True
    return False


def _resume_logical_epoch_step(checkpoint: dict[str, Any] | None, budget: int | None) -> int:
    """Restore progress inside a logical epoch without replaying its step budget."""
    progress = int((checkpoint or {}).get("dgpo_epoch_step", 0))
    if progress == 0:
        return 0
    if budget is None or not 0 < progress < budget:
        raise ValueError("mid-epoch resume has invalid dgpo_epoch_step or changed epoch budget")
    if checkpoint.get("dgpo_next_epoch") != checkpoint.get("epoch"):
        raise ValueError("mid-epoch resume must continue the recorded current epoch")
    return progress




def _should_apply_pinned_classifier_restart(
    *,
    pinned: bool,
    load_mode: str,
    auto_resume_checkpoint: Path | str | None,
    best_source_dir: Any,
) -> bool:
    """Inherit classifiers on first start; skip when this run already has last.ckpt."""
    if not pinned:
        return False
    if auto_resume_checkpoint is not None:
        return False
    if str(load_mode).strip().lower() != "resume" or best_source_dir:
        raise ValueError(
            "pinned_classifier_restart requires explicit resume source without "
            "auto-resume selection"
        )
    return True


def _prepare_best_point_restart(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Keep the best policy/reward/reference pairing, reset this experiment's clock."""
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldState

    previous = AdaptiveOmniFoldState.from_dict(checkpoint["dgpo_adaptive_omnifold_state"])
    if not previous.calibrated:
        raise ValueError("best-point restart requires an installed calibrated OmniFold round")
    state = AdaptiveOmniFoldState(
        reward_round_id=previous.reward_round_id,
        baseline_auc_gap=previous.baseline_auc_gap,
        previous_audit_auc_gap=previous.previous_audit_auc_gap,
        trigger_threshold=previous.trigger_threshold,
        resume_refit_once_completed=previous.resume_refit_once_completed,
        resume_refit_once_id=previous.resume_refit_once_id,
        raw_monitor_state=previous.raw_monitor_state,
        raw_monitor_baseline_pending=True,
        last_decision="new_experiment_from_best",
    )
    result = dict(checkpoint)
    for key in ("dgpo_optimizer_state_dict", "ema_state_dict",
                "dgpo_ema_rollout_state_dict", "dgpo_projection_constraint_state"):
        result.pop(key, None)
    result.update(epoch=-1, global_step=0, dgpo_next_epoch=0, dgpo_epoch_step=0,
                  dgpo_adaptive_omnifold_state=state.to_dict())
    return result


def _prepare_pinned_classifier_restart(checkpoint: dict[str, Any], dgpo: Mapping[str, Any]) -> dict[str, Any]:
    """Inherit a pinned reward stack and rebuild only a missing raw judge."""
    required = {"state_dict", "dgpo_adaptive_omnifold_state", "dgpo_ref_state_dict",
                "dgpo_round_ref_state_dict", "dgpo_round_ref_sha256",
                "dgpo_omnifold_reward_stack", "dgpo_omnifold_reward_metadata"}
    if checkpoint is None or required.difference(checkpoint):
        raise ValueError("Pinned classifier restart requires a complete policy/reward/reference checkpoint")
    a = dgpo["adaptive_omnifold"]
    r = a["recalibration"]
    stack = checkpoint.get("dgpo_omnifold_reward_stack", {})
    reward_payload = stack.get("reward", {})
    cache = reward_payload.get("warm_start_state", {})
    expected_outer = (
        {
            "schema": "condition-hash-80-20-v1",
            "seed": a["single_pool_split_seed"],
        }
        if bool(a.get("single_pool_train_validation", False))
        else None
    )
    protocol = cache.get("protocol", {})
    repeats = int(r.get("crossfit_repeats", 1))
    if (cache.get("outer_partition") != expected_outer
            or protocol.get("scheme") != "condition_sha256_v1"
            or protocol.get("seed") != r["seed"]
            or protocol.get("folds") != r["crossfit_folds"]
            or int(protocol.get("repeats", 1)) != repeats):
        raise ValueError("Pinned classifier restart requires matching saved OmniFold fold/split provenance")
    expected_member_count = repeats * int(r["crossfit_folds"])
    increments = list(reward_payload.get("increments") or [])
    coefficients = list(reward_payload.get("increment_coefficients") or [])
    iterations = list(reward_payload.get("increment_iterations") or [])
    if (
        len(increments) != expected_member_count
        or iterations != [1] * expected_member_count
        or len(coefficients) != expected_member_count
        or any(
            not math.isclose(
                float(value), 1.0 / expected_member_count,
                rel_tol=1.0e-6, abs_tol=1.0e-6,
            )
            for value in coefficients
        )
        or any(not item.get("state") for item in increments)
    ):
        raise ValueError(
            "Pinned classifier restart requires every serialized iteration-1 "
            "reward ensemble member with equal coefficients"
        )
    base_digests = {item.get("base_digest") for item in increments}
    packing_specs = [item.get("packing_spec") for item in increments]
    if (
        len(base_digests) != 1
        or None in base_digests
        or any(spec != packing_specs[0] for spec in packing_specs)
    ):
        raise ValueError(
            "Pinned classifier restart found inconsistent reward ensemble metadata"
        )
    monitor = checkpoint.get("dgpo_adaptive_omnifold_state", {}).get("raw_monitor_state", {})
    monitor_complete = bool(monitor.get("state") and monitor.get("protocol"))
    if monitor and not monitor_complete:
        raise ValueError(
            "Pinned classifier restart found an incomplete saved raw monitor"
        )
    if (
        not monitor_complete
        and bool(a.get("trigger", {}).get("warm_start_classifier", False))
    ):
        raise ValueError(
            "Pinned classifier restart requires a saved raw monitor when "
            "warm_start_classifier=true"
        )
    saved_states = [item["state"] for item in increments]
    if monitor_complete:
        saved_states.insert(0, monitor["state"])
    for saved in saved_states:
        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
               for value in saved.values()):
            raise ValueError("Pinned classifier restart found invalid saved classifier weights")
    result = _prepare_best_point_restart(checkpoint)
    state = result["dgpo_adaptive_omnifold_state"]
    if monitor_complete:
        state["raw_monitor_state"] = copy.deepcopy(monitor)
        # A fitted judge from this exact policy can be reused as-is.
        state["raw_monitor_baseline_pending"] = False
        state["raw_monitor_state"].pop("recertify_inherited_weights", None)
        state["raw_monitor_state"]["inherit_monitor_as_is"] = True
    else:
        # Older snapshots with warm_start_classifier=false intentionally do
        # not serialize judge weights. Refit only the diagnostic baseline at
        # step 0; the six-member reward stack remains installed and untouched.
        state["raw_monitor_state"] = {}
        state["raw_monitor_baseline_pending"] = True
    return result


def _resolve_distributed_auto_resume_checkpoint(
    checkpoint_save_path: str | Path | None,
    *,
    enabled: bool,
    fallback_checkpoint_path: str | Path | None,
    best_source_checkpoint_dir: str | Path | None,
    world_size: int,
    device: torch.device,
) -> Path | None:
    """Select once on rank zero, then require the same snapshot on every rank."""
    kwargs = dict(
        enabled=enabled,
        fallback_checkpoint_path=fallback_checkpoint_path,
        best_source_checkpoint_dir=best_source_checkpoint_dir,
    )
    if world_size <= 1:
        return resolve_dgpo_auto_resume_checkpoint(checkpoint_save_path, **kwargs)
    if not dist.is_initialized():
        raise RuntimeError("distributed DGPO auto-resume requires an initialized process group")
    message = [None, None]  # canonical path, resolution error
    if dist.get_rank() == 0:
        try:
            selected = resolve_dgpo_auto_resume_checkpoint(checkpoint_save_path, **kwargs)
            message[0] = str(selected) if selected is not None else None
        except Exception as exc:
            # Broadcast failures too; peers must not hang waiting for rank zero.
            message[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(message, src=0, device=device)
    if message[1] is not None:
        raise RuntimeError(f"DGPO auto-resume selection failed: {message[1]}")
    if message[0] is None:
        return None
    selected = Path(message[0])
    visible = torch.tensor(int(selected.is_file()), device=device, dtype=torch.int32)
    dist.all_reduce(visible, op=dist.ReduceOp.MIN)
    if not int(visible.item()):
        raise RuntimeError(
            f"DGPO resume checkpoint is not visible on every worker: {selected}"
        )
    return selected


def dgpo_train_loop(cfg: dict[str, Any]) -> None:
    """Per-worker DGPO training loop launched by ``ray.train.torch.TorchTrainer``.

    Each Ray Train worker runs this function in its own process.  Ray Train
    initialises the torch distributed process group and gives each worker its
    own per-rank Ray Data shard via ``ray.train.get_dataset_shard``.  This
    function:

    1. Resolves rank / world-size / device from the Ray Train context.
    2. Pulls its own train (and validation) shard.
    3. Builds the EveNet backbone, EMA shadows, reference policy, and DGPO optimizer.
    4. Iterates the DGPO algorithm in lock-step across ranks (``_next_batch_synced``
       all-reduces the ``has-more`` flag so collectives stay aligned).
    5. Runs validation, checkpoint top-K save, and ``last`` checkpoint on rank 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = ray.train.get_context()
    rank = int(ctx.get_world_rank())
    world_size = int(ctx.get_world_size())
    local_rank = int(ctx.get_local_rank())
    is_rank0 = rank == 0
    device = ray.train.torch.get_device()

    # Earliest possible visibility check: emitted by every worker before any I/O or model build,
    # so the user can see the actual world_size immediately and confirm multi-node DDP is live.
    _log.info(
        "[DGPO][boot] rank=%s/%s local_rank=%s host=%s device=%s",
        rank, world_size, local_rank, os.uname().nodename, device,
    )

    config_path = Path(str(cfg["config_path"])).resolve()
    max_steps: int | None = cfg.get("max_steps")
    wandb_flag = bool(cfg.get("wandb", True))
    total_events = int(cfg["total_events"])
    omnifold_train_events = int(cfg.get("omnifold_train_events", total_events))
    val_events_in = cfg.get("val_events", 0)
    val_events: int | None = int(val_events_in) if val_events_in else None
    config_yaml_text = cfg.get("config_yaml", None)

    if not config_path.is_file():
        if not isinstance(config_yaml_text, str) or not config_yaml_text.strip():
            raise FileNotFoundError(
                f"DGPO worker cannot access config path {config_path} and no config_yaml payload was provided."
            )
        worker_runtime_dir = Path(
            tempfile.mkdtemp(prefix="dgpo_worker_runtime_", dir=os.environ.get("TMPDIR", None))
        )
        config_path = worker_runtime_dir / config_path.name
        config_path.write_text(config_yaml_text)
        _log.info(
            "[DGPO][boot] rank=%s materialized worker-local runtime config at %s",
            rank,
            config_path,
        )

    global_config.load_yaml(config_path)
    _assert_rl_enabled()
    endpoint_config = EndpointKLConfig.parse(_dgpo_cfg_get(global_config.dgpo, "endpoint_kl", None))
    validate_endpoint_protocol(endpoint_config, global_config.dgpo,
        generation_uses_ema=generation_uses_ema_shadow(global_config.options.Training.get("EMA", None)))
    platform_info = global_config.platform

    matmul_precision = str(
        _dgpo_cfg_get(global_config.dgpo, "float32_matmul_precision", "high")
    ).lower()
    if matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "dgpo.float32_matmul_precision must be one of "
            f"highest/high/medium, got {matmul_precision!r}"
        )
    torch.set_float32_matmul_precision(matmul_precision)
    if is_rank0:
        _log.info("[DGPO] float32 matmul precision=%s", matmul_precision)

    wandb_active = _start_wandb_run(disable=not wandb_flag) if is_rank0 else False

    # Per-rank Ray Data shards.  Ray Train assigns each worker a disjoint subset.
    train_shard = ray.train.get_dataset_shard("train")
    fixed_omnifold_pool = bool(cfg.get("fixed_omnifold_pool", False))
    omnifold_train_shard = (
        ray.train.get_dataset_shard("omnifold_train")
        if fixed_omnifold_pool
        else train_shard
    )
    if omnifold_train_shard is None:
        raise RuntimeError("fixed OmniFold pool dataset shard is unavailable")
    val_shard = ray.train.get_dataset_shard("validation") if val_events else None

    batch_size = int(platform_info.batch_size)
    prefetch = int(getattr(platform_info, "prefetch_batches", 1))
    train_loader_cfg = {
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
        # Mirror EveNet pretraining: enable in-shard random shuffling.
        "local_shuffle_buffer_size": batch_size * prefetch,
    }
    # Keep validation order deterministic so pre-DGPO and later response matrices
    # compare the same validation rows in the same per-rank order.
    validation_batch_size = int(
        _dgpo_cfg_get(global_config.dgpo, "validation_batch_size", batch_size)
    )
    if validation_batch_size < 1:
        raise ValueError("dgpo.validation_batch_size must be positive")
    val_loader_cfg = {
        "batch_size": validation_batch_size,
        "prefetch_batches": prefetch,
    }

    configured_checkpoint_load_mode = str(
        _dgpo_cfg_get(global_config.dgpo, "checkpoint_load_mode", "resume")
    ).strip().lower()
    # Validate the fallback mode even when a recovery snapshot is present, so
    # an invalid config cannot silently become valid only on resume machines.
    select_dgpo_training_state(None, load_mode=configured_checkpoint_load_mode)
    auto_resume_enabled = bool(
        _dgpo_cfg_get(global_config.dgpo, "auto_resume_from_last", False)
    )
    auto_resume_checkpoint = _resolve_distributed_auto_resume_checkpoint(
        global_config.options.Training.get("model_checkpoint_save_path", None),
        enabled=auto_resume_enabled,
        fallback_checkpoint_path=_dgpo_cfg_get(
            global_config.dgpo, "auto_resume_fallback_checkpoint_path", None
        ),
        best_source_checkpoint_dir=_dgpo_cfg_get(
            global_config.dgpo, "auto_resume_best_source_checkpoint_dir", None
        ),
        world_size=world_size,
        device=device,
    )
    if auto_resume_checkpoint is not None:
        startup_checkpoint_path: Path | None = auto_resume_checkpoint
        checkpoint_load_mode = "resume"
        if is_rank0:
            _log.info(
                "[DGPO] Auto-resume found %s; restoring the complete DGPO + "
                "OmniFold state instead of fitting the initial classifiers again.",
                auto_resume_checkpoint,
            )
    else:
        startup_checkpoint_path = None
        checkpoint_load_mode = configured_checkpoint_load_mode
        if is_rank0 and auto_resume_enabled:
            _log.info(
                "[DGPO] No last.ckpt in %s; starting from the configured %s "
                "checkpoint and saving the installed OmniFold round before training.",
                global_config.options.Training.get(
                    "model_checkpoint_save_path", None
                ),
                checkpoint_load_mode,
            )

    bundle = load_evenet_model_for_dgpo(
        None,
        device,
        checkpoint_path=startup_checkpoint_path,
        config=global_config,
    )
    eve_net = bundle.model

    loaded_ckpt_dict = None
    if bundle.checkpoint_path is not None:
        loaded_ckpt_dict = torch.load(
            str(bundle.checkpoint_path), map_location=device, weights_only=False
        )
    if auto_resume_checkpoint is not None:
        required_recovery_keys = {
            "dgpo_checkpoint_version",
            "dgpo_next_epoch",
            "dgpo_optimizer_state_dict",
            "dgpo_round_ref_state_dict",
            "dgpo_adaptive_omnifold_state",
            "dgpo_omnifold_reward_stack",
        }
        missing_recovery_keys = sorted(
            required_recovery_keys.difference(loaded_ckpt_dict or {})
        )
        if missing_recovery_keys:
            raise RuntimeError(
                "automatic DGPO resume checkpoint is incomplete and cannot "
                "safely skip OmniFold training; missing keys: "
                + ", ".join(missing_recovery_keys)
            )
    ckpt_dict = select_dgpo_training_state(
        loaded_ckpt_dict,
        load_mode=checkpoint_load_mode,
    )
    best_source_dir = _dgpo_cfg_get(global_config.dgpo, "auto_resume_best_source_checkpoint_dir", None)
    best_point_restart = bool(
        _dgpo_cfg_get(global_config.dgpo, "best_source_start_new_experiment", False)
        and best_source_dir and auto_resume_checkpoint is not None
        and auto_resume_checkpoint.parent == Path(str(best_source_dir)).expanduser().resolve()
    )
    pinned_restart = _should_apply_pinned_classifier_restart(
        pinned=bool(_dgpo_cfg_get(global_config.dgpo, "pinned_classifier_restart", False)),
        load_mode=checkpoint_load_mode,
        auto_resume_checkpoint=auto_resume_checkpoint,
        best_source_dir=best_source_dir,
    )
    if pinned_restart:
        ckpt_dict = _prepare_pinned_classifier_restart(ckpt_dict, global_config.dgpo)
        rebuild_raw_monitor = bool(
            ckpt_dict["dgpo_adaptive_omnifold_state"].get(
                "raw_monitor_baseline_pending", False
            )
        )
        _log.info(
            "[DGPO] PINNED CLASSIFIER RESTART: inherited installed OmniFold; "
            "reset epoch/step/optimizer and historical best; raw monitor=%s.",
            (
                "fresh step-0 diagnostic baseline before the first policy update"
                if rebuild_raw_monitor
                else "inherited as-is"
            ),
        )
    if best_point_restart:
        ckpt_dict = _prepare_best_point_restart(ckpt_dict)
        if is_rank0:
            _log.info(
                "[DGPO] New experiment from best point: epoch/step=0, fresh AdamW/scheduler/EMA, "
                "trust schedule age=0; preserve installed OmniFold + matching reference."
            )
    architecture_bootstrap = bool(_dgpo_cfg_get(global_config.dgpo, "step_zero_architecture_bootstrap", False))
    if architecture_bootstrap:
        if checkpoint_load_mode != "resume" or pinned_restart or best_point_restart or auto_resume_checkpoint is not None:
            raise ValueError("step_zero_architecture_bootstrap requires explicit step-zero resume without auto-resume/restart")
        from RL.DGPO_neutrino.model_utils import prepare_step_zero_architecture_bootstrap
        ckpt_dict = prepare_step_zero_architecture_bootstrap(ckpt_dict)
        _log.info("[DGPO] Architecture A/B bootstrap: empty Adam groups rebuilt; saved policy, reward, references and step 0 retained.")
    if is_rank0:
        if checkpoint_load_mode == "weights_only" and loaded_ckpt_dict is not None:
            _log.info(
                "[DGPO] Fresh policy warm start from %s: ignoring checkpoint "
                "optimizer/EMA/reference/epoch/OmniFold state.",
                bundle.checkpoint_path,
            )
        else:
            _log.info("[DGPO] checkpoint_load_mode=%s", checkpoint_load_mode)

    eve_net.train()
    apply_component_freezes(eve_net, global_config)
    if bool(
        _dgpo_cfg_get(global_config.dgpo, "require_deterministic_policy", False)
    ):
        assert_dgpo_neutrino_policy_deterministic(eve_net)
    activation_checkpointing = bool(
        _dgpo_cfg_get(global_config.dgpo, "activation_checkpointing", False)
    )
    checkpointed_pet_bodies = _set_dgpo_activation_checkpointing(
        eve_net,
        enabled=activation_checkpointing,
    )
    if activation_checkpointing and checkpointed_pet_bodies < 1:
        raise RuntimeError(
            "dgpo.activation_checkpointing=true but no checkpointable PET body was found"
        )
    if is_rank0:
        _log.info(
            "[DGPO] PET activation checkpointing=%s (%s body module(s)).",
            activation_checkpointing,
            checkpointed_pet_bodies,
        )
    ref_model = make_reference_model(
        eve_net, global_config, bundle.normalization_dict, device, checkpoint=ckpt_dict
    )
    ema_save = make_ema(eve_net, global_config, checkpoint=ckpt_dict, device=device)
    ema_rollout = make_ema_rollout(
        eve_net, global_config, checkpoint=ckpt_dict, device=device
    )

    # Wrap the diffusion-vector forward in a thin nn.Module, then let Ray Train's
    # ``prepare_model`` install DDP with the right device + process-group config.
    fw = _DGPODDPForward(eve_net)
    if world_size > 1:
        model = ray.train.torch.prepare_model(
            fw,
            parallel_strategy_kwargs={"find_unused_parameters": True},
        )
    else:
        model = fw

    dtype = next(eve_net.parameters()).dtype
    # DDIM is the only rollout sampler.
    sampler = DDIMSampler(device=device)
    dg = global_config.dgpo
    reward_agg = build_reward_aggregator(
        eve_net, device, normalization_dict=bundle.normalization_dict
    )
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
        AdaptiveOmniFoldState,
        adaptive_audit_protocol_signature,
        adaptive_trust_policy_lr_scale,
        policy_round_warmup_metrics,
        migrate_unstarted_policy_warmup_after_resume,
        start_inherited_round_policy_warmup,
        advance_policy_round_warmup,
        clamp_fixed_trust_radius_after_resume,
        record_reference_trust_attempt,
        resolve_adaptive_config,
        should_probe_training_boundary,
        trust_region_exhausted,
        update_empirical_trust_radius_from_audit,
        validate_adaptive_pairing,
        initialize_global_raw_best,
    )
    from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import (
        REWARD_STACK_CHECKPOINT_KEY,
        validate_omnifold_reward_startup,
    )

    adaptive_cfg = resolve_adaptive_config(
        dg, classifier_only=bool(_dgpo_cfg_get(
            getattr(global_config, "experiment", {}), "classifier_only", False,
        )),
    )
    omnifold_pool_batch_size = int(
        adaptive_cfg.pool_generation_batch_size
        if adaptive_cfg.pool_generation_batch_size is not None
        else batch_size
    )
    omnifold_train_loader_cfg = {
        "batch_size": omnifold_pool_batch_size,
        "prefetch_batches": prefetch,
        "local_shuffle_buffer_size": omnifold_pool_batch_size * prefetch,
    }
    omnifold_val_loader_cfg = {
        "batch_size": omnifold_pool_batch_size,
        "prefetch_batches": prefetch,
    }
    omnifold_val_shard = val_shard
    if adaptive_cfg.enabled and adaptive_cfg.cache_event_inputs:
        omnifold_train_loader_cfg["local_shuffle_seed"] = adaptive_cfg.pool_selection_seed
        omnifold_train_shard = _FixedEventInputShard(omnifold_train_shard)
        if val_shard is not None:
            omnifold_val_shard = _FixedEventInputShard(val_shard)
    if is_rank0 and adaptive_cfg.enabled:
        _log.info(
            "[DGPO/omnifold] K=1 pool generation batch/worker=%s "
            "(DGPO policy batch/worker=%s; classifier global batch=%s).",
            omnifold_pool_batch_size,
            batch_size,
            int(adaptive_cfg.fit.get("batch_size", 8192)),
        )
        if adaptive_cfg.trust_boundary_enabled:
            _log.info(
                "[DGPO/trust] adaptive boundary enabled: delta_max=%.6g "
                "delta_floor=%.6g warning=%.3g power=%.3g AUC_z=%.3g; "
                "enforcement=%s backtrack_factor=%.3g max_backtracks=%s "
                "probe_events/rank=%s fixed_probe/round=%s interior=%.3g.",
                adaptive_cfg.trust_delta_max,
                adaptive_cfg.trust_delta_floor,
                adaptive_cfg.trust_warning_fraction,
                adaptive_cfg.trust_adaptive_power,
                adaptive_cfg.trust_auc_confidence_z,
                adaptive_cfg.trust_enforcement,
                adaptive_cfg.trust_backtrack_factor,
                adaptive_cfg.trust_max_backtracks,
                adaptive_cfg.trust_probe_events_per_rank,
                adaptive_cfg.trust_fixed_probe_per_reward_round,
                adaptive_cfg.trust_interior_fraction,
            )
            if adaptive_cfg.trust_empirical_radius_enabled:
                _log.info(
                    "[DGPO/trust] empirical audit-envelope calibration enabled: "
                    "bidirectional=%s safety_factor=%.3g expand=%.3g shrink=%.3g "
                    "target_accept=%.3g target_scale=%.3g safe_audits=%s "
                    "attempt_window=%s exhaustion_window=%s distance_window=%s "
                    "min_samples=%s "
                    "confidence_z=%.3g cross_round_nonexpanding=%s "
                    "policy_lr_scaling=%s policy_lr_floor=%.3g; "
                    "round_AUC_guard=%s round_z=%.3g plateau_patience=%s; "
                    "trajectory_search=%s failed_direction_patience=%s; "
                    "signed_direction_probe=%s signed_scales=%s "
                    "recover_reverse_only=%s; "
                    "lookahead_extragradient=%s lookahead_scale=%.3g; "
                    "dgpo.beta is locked to 1.0.",
                    adaptive_cfg.trust_empirical_bidirectional,
                    adaptive_cfg.trust_empirical_safety_factor,
                    adaptive_cfg.trust_empirical_expand_factor,
                    adaptive_cfg.trust_empirical_shrink_factor,
                    adaptive_cfg.trust_empirical_target_acceptance_rate,
                    adaptive_cfg.trust_empirical_target_update_scale,
                    adaptive_cfg.trust_empirical_safe_audits_required,
                    adaptive_cfg.trust_empirical_attempt_window_steps,
                    adaptive_cfg.trust_exhaustion_scale_window_steps,
                    adaptive_cfg.trust_empirical_distance_window_steps,
                    adaptive_cfg.trust_empirical_min_distance_samples,
                    adaptive_cfg.trust_empirical_confidence_z,
                    adaptive_cfg.trust_cross_round_nonexpanding,
                    adaptive_cfg.trust_policy_lr_scaling_enabled,
                    adaptive_cfg.trust_policy_lr_scale_floor,
                    adaptive_cfg.trust_round_acceptance_enabled,
                    adaptive_cfg.trust_round_acceptance_confidence_z,
                    adaptive_cfg.trust_round_plateau_patience,
                    adaptive_cfg.trust_trajectory_search_enabled,
                    adaptive_cfg.trust_failed_direction_patience,
                    adaptive_cfg.trust_signed_direction_probe_enabled,
                    adaptive_cfg.trust_signed_direction_probe_scales,
                    adaptive_cfg.trust_signed_direction_recovery_enabled,
                    adaptive_cfg.trust_extragradient_enabled,
                    adaptive_cfg.trust_extragradient_lookahead_scale,
                )
    omnifold_source = reward_agg.omnifold_source
    if adaptive_cfg.enabled and omnifold_source is None:
        raise ValueError(
            "dgpo.adaptive_omnifold.enabled requires reward_config.type=omnifold"
        )
    saved_stack = (
        None if ckpt_dict is None else ckpt_dict.get(REWARD_STACK_CHECKPOINT_KEY)
    )
    adaptive_state = AdaptiveOmniFoldState.from_dict(
        None if ckpt_dict is None else ckpt_dict.get("dgpo_adaptive_omnifold_state")
    )
    if saved_stack is not None and adaptive_cfg.enabled:
        old_warmup_protocol = adaptive_state.policy_warmup_protocol
        if migrate_unstarted_policy_warmup_after_resume(adaptive_state, cfg=adaptive_cfg) and is_rank0:
            _log.info("[DGPO/resume] Unstarted round warmup changed: %s -> %s; saved clocks and reward stack preserved.",
                      old_warmup_protocol, adaptive_state.policy_warmup_protocol)
        inherited_warmup = start_inherited_round_policy_warmup(
            adaptive_state,
            cfg=adaptive_cfg,
            global_step=int((ckpt_dict or {}).get("global_step", 0) or 0),
            restart=bool(pinned_restart or best_point_restart),
        )
        if inherited_warmup and is_rank0:
            _log.info(
                "[DGPO] inherited installed round %s without a new install; starting policy warmup %s",
                adaptive_state.reward_round_id, adaptive_state.policy_warmup_protocol,
            )
    pending_resume_trust_diagnostics = clamp_fixed_trust_radius_after_resume(
        adaptive_state,
        cfg=adaptive_cfg,
        initialize_round_decay=saved_stack is not None,
    )
    if (
        is_rank0
        and float(
            pending_resume_trust_diagnostics.get(
                "reference_trust/resume_radius_clamp_applied", 0.0
            )
        )
        >= 0.5
    ):
        _log.info(
            "[DGPO/trust] resume synchronized trust delta %.6g -> %.6g ",
            float(
                pending_resume_trust_diagnostics[
                    "reference_trust/resume_radius_before"
                ]
            ),
            float(
                pending_resume_trust_diagnostics[
                    "reference_trust/resume_radius_after"
                ]
            ),
        )
    resume_refit_version_completed_at_load = bool(
        adaptive_state.resume_refit_once_completed
        and (
            not adaptive_cfg.refit_once_id
            or adaptive_state.resume_refit_once_id == adaptive_cfg.refit_once_id
        )
    )
    allow_reward_bundle_migration = bool(
        saved_stack is not None
        and adaptive_cfg.enabled
        and adaptive_cfg.refit_once_on_resume
        and bool(adaptive_cfg.refit_once_id)
        and not resume_refit_version_completed_at_load
    )
    if saved_stack is not None:
        if omnifold_source is None:
            raise ValueError(
                "DGPO checkpoint contains an OmniFold stack but the reward is disabled"
            )
        omnifold_source.load_stack_payload(
            saved_stack,
            allow_source_bundle_migration=allow_reward_bundle_migration,
        )
    elif (
        ckpt_dict is not None
        and int(ckpt_dict.get("dgpo_reward_round_id", 0)) > 0
    ):
        raise ValueError(
            "adaptive DGPO checkpoint has a later reward round but no saved ratio stack"
        )

    reward_checkpoint_metadata = reward_agg.checkpoint_metadata()
    if reward_checkpoint_metadata is not None:
        validate_omnifold_reward_startup(
            checkpoint=ckpt_dict,
            current_metadata=reward_checkpoint_metadata,
            policy_checkpoint=bundle.checkpoint_path,
            allow_source_bundle_migration=allow_reward_bundle_migration,
        )

    if adaptive_cfg.enabled:
        current_audit_signature = adaptive_audit_protocol_signature(adaptive_cfg)
        restored_audit_signature = str(adaptive_state.audit_protocol_signature or "")
        if (adaptive_cfg.raw_best_scope == "global" and restored_audit_signature
                and restored_audit_signature != current_audit_signature):
            raise ValueError("global-best rollback requires the checkpoint's matching raw audit protocol")
        if not restored_audit_signature:
            adaptive_state.audit_protocol_signature = current_audit_signature
        elif restored_audit_signature != current_audit_signature:
            if is_rank0:
                _log.warning(
                    "[DGPO/omnifold] restored audit protocol %s differs from %s; "
                    "preserving the installed-round baseline and its original "
                    "protocol provenance until a new reward round is accepted.",
                    restored_audit_signature[:12] or "<legacy>",
                    current_audit_signature[:12],
                )
    if omnifold_source is not None and not adaptive_state.probe_history:
        adaptive_state.reward_round_id = int(omnifold_source.reward_round_id)
    if (
        ckpt_dict is not None
        and "dgpo_reward_round_id" in ckpt_dict
        and omnifold_source is not None
        and int(ckpt_dict["dgpo_reward_round_id"])
        != int(omnifold_source.reward_round_id)
    ):
        raise ValueError("checkpoint reward round and restored OmniFold stack disagree")
    round_ref_model = (
        make_round_reference_model(
            ref_model,
            global_config,
            bundle.normalization_dict,
            device,
            checkpoint=ckpt_dict,
        )
        if adaptive_cfg.enabled
        else ref_model
    )
    reference_trust_probe_cache: dict[str, Any] = {}
    if endpoint_config.enabled:
        endpoint_saved = None if ckpt_dict is None else ckpt_dict.get(ENDPOINT_KL_CHECKPOINT_KEY)
        _unwrap_core_evenet(model)._endpoint_kl_controller = EndpointKLController(endpoint_config, endpoint_saved)
        _log.info("[DGPO/endpoint-KL] enabled coefficient=%g; online current/reference H4 critic; full DDIM pathwise gradient; raw endpoints", endpoint_config.coefficient)
    restore_fixed_probe = bool(
        adaptive_cfg.trust_boundary_enabled
        and adaptive_cfg.trust_fixed_probe_per_reward_round
        and adaptive_state.trust_probe_payload is not None
        and int(adaptive_state.trust_probe_round_id)
        == int(adaptive_state.reward_round_id)
    )
    if restore_fixed_probe and (
        adaptive_state.trust_probe_payload.get("format")
        != _PER_RANK_TRUST_PROBE_FORMAT
    ):
        # A legacy payload contains only rank 0's rows.  Preserve the complete
        # training/reward state but rebuild the probe so every rank contributes
        # distinct conditions under the corrected protocol.
        adaptive_state.trust_probe_payload = None
        adaptive_state.trust_probe_round_id = -1
        restore_fixed_probe = False
        if is_rank0:
            _log.info(
                "[DGPO/trust] discarded legacy rank-0-only fixed probe; "
                "a distinct per-rank probe will be captured on the next step."
            )
    if restore_fixed_probe:
        assert adaptive_state.trust_probe_payload is not None
        reference_trust_probe_cache.update(
            {
                "probe": _restore_reference_trust_probe(
                    adaptive_state.trust_probe_payload,
                    device=device,
                    world_size=world_size,
                ),
                "payload": adaptive_state.trust_probe_payload,
                "round_id": int(adaptive_state.reward_round_id),
                "dirty": False,
            }
        )
        if is_rank0:
            _log.info(
                "[DGPO/trust] restored fixed probe for reward round=%s from checkpoint.",
                adaptive_state.reward_round_id,
            )
    if (
        omnifold_source is not None
        and int(omnifold_source.reward_round_id) > 0
        and not adaptive_cfg.enabled
    ):
        raise ValueError(
            "checkpoint contains a dynamic OmniFold round; adaptive_omnifold "
            "must remain enabled on resume"
        )
    if adaptive_cfg.enabled:
        if val_shard is None:
            raise ValueError(
                "adaptive OmniFold needs platform.data_parquet_val_dir for held-out audits"
            )
        validate_adaptive_pairing(
            reward_source=omnifold_source,
            state=adaptive_state,
            round_ref_model=round_ref_model,
            checkpoint=ckpt_dict,
            where="adaptive_startup",
        )
    effective_batch = batch_size * world_size
    validation_effective_batch = validation_batch_size * world_size
    data_steps_per_pass = max(1, math.ceil(total_events / effective_batch))
    configured_steps_per_epoch = dg.get("steps_per_epoch", None)
    if configured_steps_per_epoch is None:
        steps_per_epoch = data_steps_per_pass
    else:
        steps_per_epoch = int(configured_steps_per_epoch)
        if steps_per_epoch < 1:
            raise ValueError(
                "dgpo.steps_per_epoch must be a positive optimizer-step count "
                f"or null, got {configured_steps_per_epoch!r}"
            )
    train_opt_lr = global_config.options.Training
    if adaptive_cfg.staleness_every_n_steps is not None and (
        configured_steps_per_epoch is None
        or steps_per_epoch % adaptive_cfg.staleness_every_n_steps != 0
    ):
        raise ValueError(
            "step-based staleness requires a fixed steps_per_epoch divisible by "
            "staleness_every_n_steps, so epoch-end trust monitoring stays aligned"
        )
    warm_up_factor = float(train_opt_lr.get("learning_rate_warm_up_factor", 1.0))
    warmup_steps = max(1, math.ceil(warm_up_factor * steps_per_epoch))
    optimizer = build_optimizer(
        model,
        steps_per_epoch=steps_per_epoch,
        warmup_steps=warmup_steps,
        is_rank0=is_rank0,
        lr_schedule=dg.get("lr_schedule"),
        conditioning_learning_rates=dg.get("conditioning_learning_rates"),
    )

    start_epoch, global_step = parse_dgpo_resume_from_checkpoint(ckpt_dict)
    if checkpoint_load_mode == "resume" and not (best_point_restart or pinned_restart or architecture_bootstrap) and optimizer.cosine_state is not None and (
        ckpt_dict is None or "dgpo_optimizer_state_dict" not in ckpt_dict
    ):
        raise ValueError("cosine full resume requires checkpointed DGPO optimizer/scheduler state")
    if is_rank0 and wandb_active:
        import wandb
        try:
            wandb.run.summary.update({
                "resume/start_epoch": int(start_epoch),
                "resume/completed_dgpo_steps": int(global_step),
                "resume/epoch_step": int((ckpt_dict or {}).get("dgpo_epoch_step", 0)),
                "resume/checkpoint_load_mode": checkpoint_load_mode,
                "resume/source_checkpoint": str(bundle.checkpoint_path or ""),
                "resume/pinned_classifier_restart": int(pinned_restart),
                "resume/step_zero_architecture_bootstrap": int(architecture_bootstrap),
            })
        except Exception as exc:
            _log.warning("[DGPO] W&B resume metadata could not be published: %s", exc)
    if ckpt_dict is not None and "dgpo_optimizer_state_dict" in ckpt_dict:
        try:
            use_config_lr_schedule = (dg.get("lr_schedule") or {}).get("resume_use_config", False)
            optimizer.load_state_dict(
                ckpt_dict["dgpo_optimizer_state_dict"],
                use_config_lr_schedule=use_config_lr_schedule,
            )
            if is_rank0:
                _log.info(
                    "[DGPO] Restored optimizer state from checkpoint; effective "
                    "AdamW weight_decay per group (current config): %s",
                    [pg["weight_decay"] for pg in optimizer.param_groups],
                )
            if is_rank0 and wandb_active:
                import wandb
                try:
                    wandb.run.summary.update({
                        "resume/lr_schedule_use_config": int(use_config_lr_schedule),
                        "resume/lr_scheduler_step": int(optimizer.scheduler.last_epoch),
                        "resume/lr_base_by_group": {
                            pg["group_name"]: float(base) for pg, base in
                            zip(optimizer.param_groups, optimizer.scheduler.base_lrs, strict=True)
                        },
                        "resume/lr_effective_by_group": {
                            pg["group_name"]: float(pg["lr"]) for pg in optimizer.param_groups
                        },
                    })
                except Exception as exc:
                    _log.warning("[DGPO] W&B resume LR metadata could not be published: %s", exc)
        except (ValueError, RuntimeError) as ex:
            if optimizer.cosine_state is not None or ckpt_dict["dgpo_optimizer_state_dict"].get("lr_schedule") is not None:
                raise RuntimeError("Cannot safely resume DGPO cosine optimizer/scheduler state") from ex
            if is_rank0:
                _log.warning(
                    "[DGPO] Could not load optimizer state (continuing fresh optimizer): %s", ex
                )

    _vm_raw = dg.get("validation_max_batches", None)
    val_max_batches: int | None = None
    if _vm_raw is not None:
        val_max_batches = int(_vm_raw)
        if val_max_batches <= 0:
            if is_rank0:
                _log.warning(
                    "[DGPO] validation_max_batches=%s is not positive; running full validation.",
                    _vm_raw,
                )
            val_max_batches = None
    _cheap_vm_raw = dg.get("validation_cheap_max_batches", 2)
    val_cheap_max_batches: int | None = None
    if _cheap_vm_raw is not None:
        val_cheap_max_batches = int(_cheap_vm_raw)
        if val_cheap_max_batches <= 0:
            raise ValueError(
                "dgpo.validation_cheap_max_batches must be positive or null, "
                f"got {_cheap_vm_raw!r}"
            )
    K = int(_dgpo_cfg_get(dg, "K", 1))
    uses_omnifold_reward = any(
        source.name == "omnifold" for source, _weight in reward_agg.sources
    )
    advantage_raw = dg.get("advantage_estimator", None)
    if advantage_raw is None:
        advantage_estimator = (
            "leave_one_out_unscaled"
            if uses_omnifold_reward
            else ADVANTAGE_ESTIMATOR_ZSCORE
        )
    else:
        advantage_estimator = str(advantage_raw)
    if advantage_estimator not in VALID_ADVANTAGE_ESTIMATORS:
        raise ValueError(
            f"unsupported dgpo.advantage_estimator={advantage_estimator!r}; "
            f"expected one of {sorted(VALID_ADVANTAGE_ESTIMATORS)}"
        )
    if uses_omnifold_reward and advantage_estimator != "leave_one_out_unscaled":
        raise ValueError(
            "OmniFold-guided DGPO requires advantage_estimator="
            "leave_one_out_unscaled so the density-ratio scale is preserved"
        )
    if advantage_estimator == "leave_one_out_unscaled" and K < 2:
        raise ValueError("leave_one_out_unscaled requires dgpo.K >= 2")
    val_K = max(1, int(dg.get("validation_K", 1)))
    val_cheap_K = max(1, int(dg.get("validation_cheap_K", min(val_K, K))))
    rollout_parallel_chains = max(1, int(dg.get("rollout_parallel_chains", 1)))
    val_rollout_parallel_chains = max(
        1,
        int(dg.get("validation_rollout_parallel_chains", rollout_parallel_chains)),
    )
    validation_every_n_epochs = max(1, int(dg.get("validation_every_n_epochs", 1)))
    validation_full_every_n_epochs = max(
        1,
        int(dg.get("validation_full_every_n_epochs", validation_every_n_epochs)),
    )
    beta = float(_dgpo_cfg_get(dg, "beta", 1.0))
    # Training and validation use independent DDIM rollout-step budgets: training uses
    # num_ddim, validation uses num_ddim_val. The validation-specific key falls back
    # to the training value when unset (null) for backward-compatible behavior.
    num_ddim = int(_dgpo_cfg_get(dg, "num_ddim_steps", 1))
    _val_steps_raw = dg.get("validation_num_ddim_steps", None)
    num_ddim_val = int(_val_steps_raw) if _val_steps_raw is not None else num_ddim
    if is_rank0:
        _log.info(
            "[DGPO] DDIM rollout steps: training=%s, validation=%s. Parallel chains: training=%s, validation=%s.",
            num_ddim,
            num_ddim_val,
            rollout_parallel_chains,
            val_rollout_parallel_chains,
        )
        _log.info(
            "[DGPO] advantage_estimator=%s%s",
            advantage_estimator,
            " (required by OmniFold density-ratio reward)"
            if uses_omnifold_reward
            else "",
        )
        _log.info(
            "[DGPO] validation tiers: cheap every=%s epoch(s), K=%s, max_batches=%s; "
            "full every=%s epoch(s), K=%s, max_batches=%s.",
            validation_every_n_epochs,
            val_cheap_K,
            val_cheap_max_batches,
            validation_full_every_n_epochs,
            val_K,
            val_max_batches,
        )
    log_every = max(1, int(dg.get("log_every", 1)))
    diagnostic_plot_names, diagnostic_plot_every = _resolve_diagnostic_plot_settings(dg)
    log_reward_dist_every = max(
        1, int(dg.get("log_reward_dist_every", diagnostic_plot_every))
    )
    train_dist_enabled = bool(dg.get("train_dist_enabled", True))
    requested_train_dist_classes: tuple[str, ...] = ()
    train_dist_by_class_every = 1
    train_dist_every = 1
    representative_class_indices: dict[str, int] = {}
    missing_representative_classes: tuple[str, ...] = ()
    if train_dist_enabled:
        requested_train_dist_classes = tuple(
            str(name)
            for name in (dg.get("train_dist_representative_classes", []) or [])
        )
        train_dist_by_class_every = max(
            1, int(dg.get("train_dist_by_class_every_n_epochs", 5))
        )
        train_dist_every = max(
            1, int(dg.get("train_dist_every_n_epochs", train_dist_by_class_every))
        )
        all_signal_classes = _event_signal_class_names()
        signal_class_to_index = {
            class_name: index for index, class_name in enumerate(all_signal_classes)
        }
        representative_class_indices = {
            class_name: signal_class_to_index[class_name]
            for class_name in requested_train_dist_classes
            if class_name in signal_class_to_index
        }
        missing_representative_classes = tuple(
            name
            for name in requested_train_dist_classes
            if name not in signal_class_to_index
        )
    if is_rank0 and train_dist_enabled and requested_train_dist_classes:
        _log.info(
            "[DGPO] pooled train_dist every=%s epoch(s); representative classes=%s every=%s epoch(s)",
            train_dist_every,
            tuple(representative_class_indices),
            train_dist_by_class_every,
        )
        if missing_representative_classes:
            _log.warning(
                "[DGPO] representative train_dist classes absent from event_info: %s",
                missing_representative_classes,
            )
    diagnostic_profile_accumulate_steps = max(
        1, int(dg.get("diagnostic_profile_accumulate_steps", 1))
    )
    num_train_timesteps = max(1, int(dg.get("num_train_timesteps", 1)))
    policy_eval_parallel_timesteps = max(
        1,
        int(dg.get("policy_eval_parallel_timesteps", 1)),
    )
    policy_eval_parallel_timesteps = min(
        policy_eval_parallel_timesteps,
        num_train_timesteps,
    )
    raw_policy_event_microbatch_size = dg.get(
        "policy_eval_event_microbatch_size",
        None,
    )
    policy_eval_event_microbatch_size: int | None = None
    if raw_policy_event_microbatch_size is not None:
        policy_eval_event_microbatch_size = int(raw_policy_event_microbatch_size)
        if policy_eval_event_microbatch_size < 1:
            raise ValueError(
                "dgpo.policy_eval_event_microbatch_size must be positive or null"
            )
    if is_rank0:
        _log.info(
            "[DGPO] policy evaluation: %s noise draws, %s parallel timestep(s) per "
            "forward, event microbatch=%s.",
            num_train_timesteps,
            policy_eval_parallel_timesteps,
            policy_eval_event_microbatch_size or "full batch",
        )
    _adv_raw = dg.get("adv_clip_max", None)
    adv_clip_max_cfg: float | None = float(_adv_raw) if _adv_raw is not None else None
    grad_clip_norm_cfg = float(dg.get("grad_clip_norm", _GRAD_CLIP_NORM))
    policy_eval_t_min_cfg = float(dg.get("policy_eval_t_min", 0.0))
    policy_eval_t_max_cfg = float(dg.get("policy_eval_t_max", 1.0))
    trust_cfg = dg.get("reference_trust", None) or {}
    reference_trust_coefficient = (
        float(_dgpo_cfg_get(trust_cfg, "coefficient", 0.0))
        if bool(_dgpo_cfg_get(trust_cfg, "enabled", False))
        else 0.0
    )
    reference_trust_objective = str(
        _dgpo_cfg_get(
            trust_cfg,
            "objective",
            REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
        )
    ).strip().lower()
    if reference_trust_objective not in VALID_REFERENCE_TRUST_OBJECTIVES:
        raise ValueError(
            "dgpo.reference_trust.objective must be one of "
            f"{sorted(VALID_REFERENCE_TRUST_OBJECTIVES)}, got "
            f"{reference_trust_objective!r}"
        )
    vp_path_kl_cfg = _dgpo_cfg_get(trust_cfg, "vp_path_kl", None) or {}
    from RL.DGPO_neutrino.gradient_conflict import resolve_gradient_conflict_config
    gradient_conflict_cfg = resolve_gradient_conflict_config(dg.get("gradient_conflict"))
    if gradient_conflict_cfg.enabled:
        if not (adaptive_cfg.enabled and adaptive_cfg.monitor_mode in {"raw_plateau_refit", "raw_only"}
                and adaptive_cfg.fixed_audit_panel and adaptive_cfg.cache_event_inputs):
            raise ValueError(
                "gradient conflict monitor requires fixed-input, identity-split "
                "raw_plateau_refit or raw_only monitoring"
            )
        if gradient_conflict_cfg.monitor_refit_lifecycle and not adaptive_cfg.raw_monitor_warm_start:
            raise ValueError(
                "gradient conflict refit-lifecycle probes require a persistent warm-start raw monitor"
            )
        cadence = adaptive_cfg.staleness_every_n_steps
        if cadence is None:
            cadence = int(steps_per_epoch) * int(
                adaptive_cfg.staleness_every_n_epochs
            )
        if gradient_conflict_cfg.every_n_steps % cadence:
            raise ValueError(
                "gradient conflict cadence must be a multiple of the raw-audit cadence"
            )
        if reference_trust_objective != REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE:
            raise ValueError("gradient conflict monitor currently supports velocity_mse trust only")
        if not bool(dg.get("require_deterministic_policy", False)) or generation_uses_ema_shadow(global_config.options.Training.get("EMA", None) or {}):
            raise ValueError("gradient conflict monitor requires deterministic live-policy generation (no EMA)")
    reference_trust_vp_path_kl_diagnostic = bool(
        _dgpo_cfg_get(vp_path_kl_cfg, "diagnostic", False)
    )
    reference_trust_vp_logsnr_min = float(
        _dgpo_cfg_get(vp_path_kl_cfg, "logsnr_min", -20.0)
    )
    reference_trust_vp_logsnr_max = float(
        _dgpo_cfg_get(vp_path_kl_cfg, "logsnr_max", 20.0)
    )
    if (
        not math.isfinite(reference_trust_vp_logsnr_min)
        or not math.isfinite(reference_trust_vp_logsnr_max)
        or reference_trust_vp_logsnr_min
        >= reference_trust_vp_logsnr_max
    ):
        raise ValueError(
            "dgpo.reference_trust.vp_path_kl requires finite "
            "logsnr_min < logsnr_max"
        )
    if not (
        math.isclose(reference_trust_vp_logsnr_min, -20.0)
        and math.isclose(reference_trust_vp_logsnr_max, 20.0)
    ):
        raise ValueError(
            "dgpo.reference_trust.vp_path_kl log-SNR endpoints must match "
            "EveNet's fixed cosine schedule (-20, 20)"
        )
    if reference_trust_coefficient < 0.0 or not math.isfinite(
        reference_trust_coefficient
    ):
        raise ValueError(
            "dgpo.reference_trust.coefficient must be finite and nonnegative, got "
            f"{reference_trust_coefficient}"
        )
    if is_rank0:
        _log.info(
            "[DGPO] round-reference trust coefficient=%.4g (%s), "
            "objective=%s, separate VP path-KL diagnostic=%s",
            reference_trust_coefficient,
            "active" if reference_trust_coefficient > 0.0 else "disabled",
            reference_trust_objective,
            reference_trust_vp_path_kl_diagnostic,
        )
    # Frozen DGPO method: configured per-event advantages, shared noise, and
    # accumulated sub-step gradients into one AdamW update. Candidate rollout
    # follows EMA.use_for_generation (live policy in the memory ablation).

    proj_cfg_startup = resolve_projection_constraint_config(dg)
    gradient_transfer_cfg_startup = resolve_gradient_transfer_trace_config(
        dg.get("gradient_transfer_trace")
    )
    (
        parameter_update_rms_target_cfg,
        parameter_update_rms_min_scale_cfg,
        parameter_update_rms_max_scale_cfg,
    ) = resolve_parameter_update_rms_calibration(dg)
    if parameter_update_rms_target_cfg is not None:
        if proj_cfg_startup.active or adaptive_cfg.trust_boundary_enabled:
            raise ValueError(
                "parameter-update RMS calibration requires no projection or "
                "hard trust boundary"
            )
        if adaptive_cfg.trust_extragradient_enabled:
            raise ValueError(
                "parameter-update RMS calibration requires ordinary AdamW, "
                "not extragradient"
            )
        if is_rank0:
            _log.info(
                "[DGPO/RMS calibration] target=%.9g scale_bounds=[%.6g, %.6g]",
                parameter_update_rms_target_cfg,
                parameter_update_rms_min_scale_cfg,
                parameter_update_rms_max_scale_cfg,
            )
    if gradient_transfer_cfg_startup.enabled:
        if reference_trust_objective != REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE:
            raise ValueError(
                "gradient transfer trace requires reference_trust.objective=velocity_mse"
            )
        if reference_trust_coefficient <= 0.0:
            raise ValueError(
                "gradient transfer trace requires a positive reference trust coefficient"
            )
        if float(_dgpo_cfg_get(dg, "beta_kl", 0.0)) != 0.0:
            raise ValueError("gradient transfer trace requires dgpo.beta_kl=0")
        variance_cfg = dg.get("variance_regularization") or {}
        if bool(variance_cfg.get("enabled", False)) and float(
            variance_cfg.get("weight", 0.0)
        ) != 0.0:
            raise ValueError(
                "gradient transfer trace requires inactive variance regularization"
            )
        if proj_cfg_startup.active or adaptive_cfg.trust_boundary_enabled:
            raise ValueError(
                "gradient transfer trace requires no projection or hard trust boundary"
            )
        if adaptive_cfg.trust_extragradient_enabled:
            raise ValueError(
                "gradient transfer trace requires the ordinary native AdamW path"
            )
        if bool(dg.get("sequential_vp_trust_backward", False)):
            raise ValueError(
                "gradient transfer trace is incompatible with sequential VP backward"
            )
        if is_rank0:
            _log.info(
                "[DGPO/gradient_transfer] exact production trace at update endpoints=%s; "
                "counterfactual trust coefficients=%s (read-only).",
                gradient_transfer_cfg_startup.update_end_steps,
                gradient_transfer_cfg_startup.counterfactual_trust_coefficients,
            )
    constraint_ckpt_blob = (
        (
            ckpt_dict.get(_DGPO_CONSTRAINT_CKPT_KEY)
            or ckpt_dict.get(_DGPO_CONSTRAINT_CKPT_KEY_LEGACY)
        )
        if ckpt_dict is not None
        else None
    )
    constraint_state: ProjectionConstraintState | None = None
    if proj_cfg_startup.active:
        _validate_dgpo_constraint_resume(constraint_ckpt_blob, expected_type="latent_swd")
        latent_cfg = proj_cfg_startup.latent_swd
        if latent_cfg is None or not latent_cfg.checkpoint_file:
            raise ValueError(
                "dgpo.projection_constraint.latent_swd.checkpoint_file is required "
                "(frozen encoder checkpoint)."
            )
        ckpt_file = Path(latent_cfg.checkpoint_file).expanduser()
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"latent_swd.checkpoint_file not found: {ckpt_file} "
                "(frozen latent-constraint encoder)"
            )
        policy_norm = Path(
            str(global_config.options.Dataset.normalization_file)
        ).expanduser()
        latent_norm_raw = latent_cfg.normalization_file.strip()
        if latent_norm_raw:
            latent_norm = Path(latent_norm_raw).expanduser()
            if policy_norm.resolve() != latent_norm.resolve() and is_rank0:
                _log.warning(
                    "[DGPO] latent_swd.normalization_file (%s) differs from policy "
                    "Dataset.normalization_file (%s); latent/policy neutrino spaces may "
                    "diverge.",
                    latent_norm,
                    policy_norm,
                )
        constraint_state = init_latent_swd_state(
            latent_cfg,
            device=device,
            resume_payload=constraint_ckpt_blob,
        )
        broadcast_latent_swd_state(
            constraint_state,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        if is_rank0:
            enc = constraint_state.model
            policy_norm_res = policy_norm.resolve()
            latent_norm_cfg = latent_cfg.normalization_file.strip()
            _log.info(
                "[DGPO] DGPO + CPO + latent-SWD (frozen): checkpoint=%s margin=%.4g "
                "num_projections=%s apply_to=%s world_size=%s latent_dim=%s d_model=%s "
                "encoder_params=%.3fM policy_norm=%s latent_swd_norm=%s. "
                "Encoder broadcast from rank 0; no on-policy retrain/finetune.",
                str(ckpt_file),
                float(latent_cfg.margin),
                int(latent_cfg.num_projections),
                latent_cfg.apply_to,
                world_size,
                enc.latent_dim,
                enc.d_model,
                sum(p.numel() for p in enc.parameters()) / 1e6,
                policy_norm_res,
                latent_norm_cfg or "(from checkpoint payload)",
            )
            if constraint_ckpt_blob is not None:
                _log.info(
                    "[DGPO] latent-SWD resume metadata from DGPO checkpoint: %s",
                    {
                        k: constraint_ckpt_blob.get(k)
                        for k in ("constraint_type", "checkpoint_file", "normalization_file")
                    },
                )
    elif is_rank0:
        _log.info("[DGPO] projection_constraint.type=none -> pure DGPO (no CPO / latent-SWD repair).")
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    save_dir_raw = global_config.options.Training.get("model_checkpoint_save_path", None)
    if (adaptive_cfg.raw_rollback_to_best_on_plateau or adaptive_cfg.raw_best_scope == "global") and not save_dir_raw:
        raise ValueError(
            "raw-AUC best-point rollback requires "
            "options.Training.model_checkpoint_save_path"
        )
    global_best_source_dirs = [Path(bundle.checkpoint_path).parent] if bundle.checkpoint_path else []
    global_best_source_dirs.extend(dg.get("global_best_checkpoint_search_dirs", []) or [])
    if adaptive_state.raw_global_initialized and adaptive_cfg.raw_best_scope != "global":
        raise ValueError("checkpoint uses global-best rollback; preserve best_scope=global on full resume")
    if adaptive_cfg.raw_best_scope == "global":
        initialize_global_raw_best(adaptive_state)
        if math.isfinite(adaptive_state.raw_best_auc_gap):
            best_path = _resolve_raw_best_policy_checkpoint(
                adaptive_state, save_dir_raw, global_scope=True, source_dirs=global_best_source_dirs,
            )
            if is_rank0:
                _log.info("[DGPO/global-best] restored gap=%.8g epoch=%s step=%s checkpoint=%s",
                          adaptive_state.raw_best_auc_gap, adaptive_state.raw_best_epoch,
                          adaptive_state.raw_best_global_step, best_path)
        if adaptive_state.raw_global_stop_requested:
            if (adaptive_cfg.raw_global_max_failed_rounds > 0
                    and adaptive_state.raw_global_failed_rounds >= adaptive_cfg.raw_global_max_failed_rounds):
                raise ValueError("global-best stagnation stop is checkpointed; review results and explicitly raise global_max_failed_rounds to continue")
            adaptive_state.raw_global_stop_requested = False
    top_k_ckpt = int(global_config.options.Training.get("model_checkpoint_save_top_k", 5))
    top_k_metric = str(
        global_config.options.Training.get(
            "model_checkpoint_top_k_metric",
            "val/reward/mean",
        )
    )
    top_k_mode = str(
        global_config.options.Training.get("model_checkpoint_top_k_mode", "max")
    ).lower()
    supported_top_k_metrics = {
        "val/reward/mean",
        "staleness/audit_balanced_accuracy",
    }
    if top_k_metric not in supported_top_k_metrics:
        raise ValueError(
            "unsupported model_checkpoint_top_k_metric="
            f"{top_k_metric!r}; expected one of {sorted(supported_top_k_metrics)}"
        )
    ckpt_topk: _DgpoCheckpointTopK | None = None
    if save_dir_raw and is_rank0:
        ckpt_topk = _DgpoCheckpointTopK(
            Path(str(save_dir_raw)).expanduser().resolve(),
            top_k_ckpt,
            metric_name=top_k_metric,
            mode=top_k_mode,
        )
        _log.info(
            "[DGPO] top-k checkpoint metric=%s mode=%s k=%s",
            top_k_metric,
            top_k_mode,
            top_k_ckpt,
        )

    # Periodic ``last`` saves are evaluated at logical-epoch boundaries only, so
    # checkpoint serialization never interrupts the middle of a 10-step control
    # interval. Adaptive audit/refit boundaries and the final epoch save even
    # when this cadence is zero or does not divide the current global step.
    ckpt_every_n_steps = max(
        0, int(global_config.dgpo.get("checkpoint_every_n_steps", 0))
    )
    ckpt_every_n_epochs = max(
        0, int(global_config.dgpo.get("checkpoint_every_n_epochs", 0))
    )
    if is_rank0:
        _log.info(
            "[DGPO] last-checkpoint cadence=%s optimizer step(s) / %s logical "
            "epoch(s), checked at logical-epoch/audit/final boundaries.",
            ckpt_every_n_steps,
            ckpt_every_n_epochs,
        )

    epochs = (
        None if bool(dg.get("unbounded_training", False))
        else int(global_config.options.Training.epochs)
    )

    if start_epoch > 0 or global_step > 0:
        if is_rank0:
            _log.info(
                "[DGPO] Resuming: start_epoch=%s global_step=%s (total epochs in config=%s).",
                start_epoch,
                global_step,
                epochs,
            )
    if epochs is not None and start_epoch >= epochs:
        if is_rank0:
            _log.info(
                "[DGPO] start_epoch=%s >= epochs=%s; nothing to train. Check config or checkpoint.",
                start_epoch,
                epochs,
            )
        _finish_wandb_run(wandb_active)
        return

    if is_rank0:
        _log.info(
            "[DGPO] rank=%s/%s device=%s train_events≈%s "
            "omnifold_train_events≈%s val_events≈%s batch=%s train_K=%s val_K=%s "
            "val_every_n_epochs=%s ddim=%s train_timesteps=%s steps/logical_epoch=%s "
            "steps/data_pass≈%s epochs=%s "
            "(advantage=%s, adaptive_omnifold=%s, staleness_every=%s)",
            rank,
            world_size,
            device,
            total_events,
            omnifold_train_events,
            val_events if val_events is not None else 0,
            batch_size,
            K,
            val_K,
            validation_every_n_epochs,
            num_ddim,
            num_train_timesteps,
            steps_per_epoch,
            data_steps_per_pass,
            epochs,
            advantage_estimator,
            adaptive_cfg.enabled,
            adaptive_cfg.staleness_every_n_epochs,
        )

    wandb_mod = None
    if wandb_active:
        import wandb as wandb_mod

    omnifold_live_log_index = 0
    omnifold_phase_ids = _OMNIFOLD_LIVE_PHASE_IDS
    classifier_loss_tracker = _ClassifierFitLossTracker()
    classifier_plot_tracker = _ClassifierTrainingPlotTracker()
    last_gradient_conflict_snapshot = None

    def _log_omnifold_fit_progress(
        phase: str,
        row: Mapping[str, Any],
        *,
        epoch_value: int,
    ) -> None:
        """Publish rank-0 classifier progress without changing fit control."""
        nonlocal omnifold_live_log_index
        if not is_rank0:
            return
        phase_name = str(phase)
        if phase_name not in omnifold_phase_ids:
            raise ValueError(f"unknown OmniFold progress phase: {phase_name}")
        omnifold_live_log_index += 1
        fit_step = int(float(row.get("step", 0.0)))
        iteration = int(float(row.get("iteration", 1.0)))
        crossfit_fold = int(float(row.get("fold", 0.0)))
        repeat_index = int(float(row.get("repeat", 1.0)))
        signed_scale = float(row.get("signed_scale", float("nan")))
        accepted_value = row.get("accepted")
        prefix = f"omnifold_live/{phase_name}"
        payload: dict[str, Any] = {
            "omnifold_live/log_index": int(omnifold_live_log_index),
            "omnifold_live/meta/phase_id": int(omnifold_phase_ids[phase_name]),
            "omnifold_live/meta/fit_step": fit_step,
            "omnifold_live/meta/iteration": iteration,
            "omnifold_live/meta/crossfit_fold": crossfit_fold,
            "omnifold_live/meta/repeat": repeat_index,
            "omnifold_live/meta/dgpo_epoch": int(epoch_value),
            "omnifold_live/meta/global_step": int(global_step),
        }
        if math.isfinite(signed_scale):
            payload["omnifold_live/meta/signed_scale"] = signed_scale
        for metric_name in (
            "training_loss",
            "training_balanced_accuracy",
            "gradient_norm",
            "gradient_clipped",
            "validation_loss",
            "validation_balanced_accuracy",
            "best_validation_loss",
            "validation_auc",
            "null_validation_loss",
            "validation_loss_gain",
            "accepted",
            "saturated",
            "threshold_reached",
            "validation_oriented_balanced_accuracy",
            "validation_balanced_accuracy_lcb",
            "validation_balanced_accuracy_lcb_standard_error",
            "validation_balanced_accuracy_lcb_streak",
            "warm_started",
            "warm_started_folds",
            "learning_rate",
            "topology_training_stage",
            "fit_stage",
            "selected_stage",
            "stage_a_validation_loss",
            "stage_b_validation_loss",
        ):
            if metric_name in row:
                payload[f"{prefix}/{metric_name}"] = row[metric_name]
        for metric_name, metric_value in row.items():
            if metric_name.startswith(
                (
                    "gradient_norm_",
                    "parameter_rms_",
                    "gradient_rms_",
                    "gradient_to_parameter_rms_ratio_",
                    "parameter_update_rms_",
                    "update_to_parameter_rms_ratio_",
                    "optimizer_group_lr_",
                    "scheduler_",
                    "stability/",
                    "visible_conditioning/",
                    "ratio_health/",
                    "logit_",
                )
            ) or metric_name in {
                "gradient_clip_scale",
                "gradient_clip_fraction",
            }:
                payload[f"{prefix}/{metric_name}"] = metric_value
        _log.info(
            "[DGPO/omnifold/live] phase=%s repeat=%s epoch=%s iteration=%s fold=%s "
            "signed_scale=%s fit_stage=%s topology_stage=%s base_lr=%.6g actual_lrs=[%s] "
            "step=%s train_loss=%.6g grad_norm=%.6g grad_clipped=%s "
            "val_loss=%.6g val_bal_acc=%.6g "
            "val_auc=%s accepted=%s saturated=%s",
            phase_name,
            repeat_index,
            epoch_value,
            iteration,
            crossfit_fold,
            "n/a" if not math.isfinite(signed_scale) else f"{signed_scale:+.6g}",
            int(float(row.get("fit_stage", 0))),
            int(float(row.get("topology_training_stage", 0))),
            float(row.get("learning_rate", float("nan"))),
            ", ".join(
                f"{key.removeprefix('optimizer_group_lr_')}={float(value):.6g}"
                for key, value in row.items()
                if key.startswith("optimizer_group_lr_")
                and not key.removeprefix("optimizer_group_lr_").isdigit()
            ) or "unavailable",
            fit_step,
            float(row.get("training_loss", float("nan"))),
            float(row.get("gradient_norm", float("nan"))),
            bool(float(row.get("gradient_clipped", 0.0))),
            float(row.get("validation_loss", float("nan"))),
            float(row.get("validation_balanced_accuracy", float("nan"))),
            "n/a" if "validation_auc" not in row else str(row["validation_auc"]),
            "n/a"
            if accepted_value is None
            else bool(float(accepted_value)),
            bool(float(row.get("saturated", 0.0))),
        )
        if wandb_mod is not None:
            wandb_settings, _ = _dgpo_wandb_yaml_section()
            if _dgpo_cfg_get(wandb_settings, "classifier_loss_curves", False):
                payload.update(classifier_plot_tracker.payload(
                    wandb_mod, phase_name, row, global_step=int(global_step), epoch=int(epoch_value),
                ))
            if _dgpo_cfg_get(wandb_settings, "classifier_loss_curves_raw", False):
                payload.update(classifier_loss_tracker.payload(
                    wandb_mod, phase_name, row, global_step=int(global_step), epoch=int(epoch_value),
                ))
            _wandb_log_auxiliary(
                wandb_mod,
                payload,
                current_global_step=int(global_step),
            )

    val_baseline_state: dict[str, np.ndarray] | None = None
    val_profile_history: dict[str, dict[str, list[float]]] = {
        name: {"epoch": [], "delta_mean": [], "slope": [], "zero": []}
        for name in _validation_profile_feature_names(cartesian=_truth_generation_cartesian())
    }

    def _append_validation_history_plots(
        val_metrics: dict[str, Any],
        *,
        epoch_value: int,
    ) -> None:
        if not is_rank0:
            return
        for profile_name, hist in val_profile_history.items():
            delta_mean = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/delta_mean",
                    float("nan"),
                )
            )
            slope = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/slope",
                    float("nan"),
                )
            )
            zero = float(
                val_metrics.get(
                    f"val_diagnostics/profile/{profile_name}/zero_delta_truth",
                    float("nan"),
                )
            )
            if not (math.isfinite(slope) and math.isfinite(zero)):
                continue
            hist["epoch"].append(float(epoch_value))
            hist["delta_mean"].append(delta_mean)
            hist["slope"].append(slope)
            hist["zero"].append(zero)
            if wandb_mod is None or _wandb_simplified_enabled():
                continue
            _x_label, y_label, display = _profile_axis_labels(profile_name)
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_delta_mean_vs_epoch"
            ] = _validation_epoch_history_figure(
                hist["epoch"],
                hist["delta_mean"],
                ylabel=y_label.replace("Mean ", "Global mean "),
                title=f"Validation {display} residual mean over epochs",
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_slope_vs_epoch"
            ] = _validation_slope_history_figure(
                hist["epoch"],
                hist["slope"],
                profile_name=profile_name,
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_zero_delta_truth_vs_epoch"
            ] = _validation_epoch_history_figure(
                hist["epoch"],
                hist["zero"],
                ylabel=f"{_x_label} where mean residual = 0",
                title=f"Validation {display} zero-crossing over epochs",
                zero_line=False,
            )
            val_metrics[
                f"val_diagnostics/profile/{profile_name}_zero_delta_vs_slope"
            ] = _validation_zero_vs_slope_figure(
                hist["slope"],
                hist["zero"],
                hist["epoch"],
                profile_name=profile_name,
            )

    profile_accum_suffixes = (
        "truth_all",
        "delta_all",
        "truth_best",
        "delta_best",
        "truth_oracle",
        "delta_oracle",
    )
    profile_accum: dict[str, dict[str, list[np.ndarray]]] = {
        name: {
            _diag_profile_raw_key(name, suffix): []
            for suffix in profile_accum_suffixes
        }
        for name in _DIAG_PROFILE_NAMES
    }
    profile_accum_batches: dict[str, int] = {name: 0 for name in _DIAG_PROFILE_NAMES}

    def _append_profile_accum(metrics: dict[str, Any]) -> None:
        if wandb_mod is None:
            return
        for profile_name, key_lists in profile_accum.items():
            have_all = all(
                isinstance(metrics.get(k), np.ndarray) and metrics[k].size > 0
                for k in key_lists
            )
            if not have_all:
                continue
            for k in key_lists:
                key_lists[k].append(metrics[k])
            profile_accum_batches[profile_name] += 1

    def _flush_profile_accum(*, step: int, force: bool = False) -> None:
        nonlocal profile_accum, profile_accum_batches
        if wandb_mod is None:
            return
        if not force and int(step) % int(diagnostic_plot_every) != 0:
            return
        payload: dict[str, Any] = {}
        flushed_names: list[str] = []
        for profile_name, key_lists in profile_accum.items():
            batches = profile_accum_batches[profile_name]
            if batches <= 0:
                continue
            if not force and batches < diagnostic_profile_accumulate_steps:
                continue
            if f"{profile_name}_profile_accumulated" not in diagnostic_plot_names:
                continue
            merged = {
                k: np.concatenate(v, axis=0) if v else np.array([], dtype=np.float64)
                for k, v in key_lists.items()
            }
            payload[_diag_profile_log_key(profile_name, accumulated=True)] = (
                _delta_selection_profiles_figure(
                    merged[_diag_profile_raw_key(profile_name, "truth_all")],
                    merged[_diag_profile_raw_key(profile_name, "delta_all")],
                    merged[_diag_profile_raw_key(profile_name, "truth_best")],
                    merged[_diag_profile_raw_key(profile_name, "delta_best")],
                    merged[_diag_profile_raw_key(profile_name, "truth_oracle")],
                    merged[_diag_profile_raw_key(profile_name, "delta_oracle")],
                    profile_name=profile_name,
                    title=_diag_profile_title(
                        profile_name,
                        accumulated_batches=batches,
                    )
                )
            )
            flushed_names.append(profile_name)
        if not payload:
            return
        try:
            _wandb_log_step(wandb_mod, payload, step=step)
        finally:
            for profile_name in flushed_names:
                profile_accum[profile_name] = {
                    _diag_profile_raw_key(profile_name, suffix): []
                    for suffix in profile_accum_suffixes
                }
                profile_accum_batches[profile_name] = 0

    def _barrier() -> None:
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

    def _adaptive_state_payload() -> dict[str, Any] | None:
        return adaptive_state.to_dict() if adaptive_cfg.enabled else None

    def _adaptive_stack_payload() -> dict[str, Any] | None:
        if not adaptive_cfg.enabled or omnifold_source is None:
            return None
        return omnifold_source.stack_payload()

    def _assert_current_reward_reference_pairing(where: str) -> None:
        if not adaptive_cfg.enabled or omnifold_source is None:
            return
        validate_adaptive_pairing(
            reward_source=omnifold_source,
            state=adaptive_state,
            round_ref_model=round_ref_model,
            checkpoint=None,
            where=where,
        )

    def _restore_live_policy_to_round_reference() -> tuple[int, bool]:
        """Restore the incumbent policy and discard optimizer direction state."""

        policy_core = unwrap_for_state_dict(model)
        policy_core.load_state_dict(round_ref_model.state_dict(), strict=True)
        cleared_optimizer_states = len(optimizer.state)
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
        if ema_save is not None:
            ema_save.update(policy_core, decay_=0.0)
        if ema_rollout is not None:
            ema_rollout.update(policy_core, decay_=0.0)
        return cleared_optimizer_states, bool(
            ema_save is not None or ema_rollout is not None
        )

    def _run_signed_direction_probe(
        *,
        epoch: int,
        probe_panel_seed: int,
        current_probe: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Audit both signs of the accumulated DGPO direction, then restore it."""

        enabled = bool(adaptive_cfg.trust_signed_direction_probe_enabled)
        direction_index = int(adaptive_state.trust_failed_direction_streak)
        diagnostics: dict[str, Any] = {
            "reference_trust/signed_probe/enabled": float(enabled),
            "reference_trust/signed_probe/completed": 0.0,
            "reference_trust/signed_probe/reward_round_id": float(
                adaptive_state.reward_round_id
            ),
            "reference_trust/signed_probe/direction_index": float(
                direction_index
            ),
            "reference_trust/signed_probe/global_step": float(global_step),
        }
        if not enabled:
            return diagnostics
        if (
            int(adaptive_state.trust_signed_probe_round_id)
            == int(adaptive_state.reward_round_id)
            and int(adaptive_state.trust_signed_probe_direction_index)
            == direction_index
        ):
            diagnostics[
                "reference_trust/signed_probe/skipped_already_completed"
            ] = 1.0
            return diagnostics

        required_current_keys = (
            "raw_auc",
            "raw_auc_gap",
            "raw_auc_null_se_approx",
            "reward_mean",
        )
        missing_current = [
            key for key in required_current_keys if key not in current_probe
        ]
        if missing_current:
            raise KeyError(
                "signed-direction probe requires current raw audit metrics: "
                + ", ".join(missing_current)
            )

        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
            evaluate_signed_direction_probe,
            fit_raw_policy_audit,
            score_reward_on_pool,
        )

        policy_core, anchor_snapshot, current_snapshot = (
            _snapshot_signed_trainable_direction(model, round_ref_model)
        )
        direction_rms = _trainable_direction_rms(
            anchor_snapshot,
            current_snapshot,
        )
        diagnostics["reference_trust/signed_probe/direction_rms"] = direction_rms
        if not math.isfinite(direction_rms):
            raise FloatingPointError(
                "signed-direction probe found non-finite policy displacement"
            )
        if direction_rms == 0.0:
            diagnostics["reference_trust/signed_probe/skipped_zero_direction"] = 1.0
            return diagnostics

        signed_candidates: list[dict[str, float]] = [
            {
                "signed_scale": 1.0,
                "raw_auc": float(current_probe["raw_auc"]),
                "raw_auc_gap": float(current_probe["raw_auc_gap"]),
                "raw_auc_se": float(current_probe["raw_auc_null_se_approx"]),
                "reward_mean": float(current_probe["reward_mean"]),
            }
        ]
        current_raw_saturated = float(
            current_probe.get("raw_audit_saturated", 0.0)
        )
        candidate_saturation: list[float] = [current_raw_saturated]
        anchor_audit: dict[str, float] | None = None
        anchor_reward_mean = float("nan")
        positive_scales = sorted(
            float(scale)
            for scale in adaptive_cfg.trust_signed_direction_probe_scales
        )
        evaluation_scales = [0.0]
        evaluation_scales.extend(
            scale
            for scale in positive_scales
            if not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        )
        evaluation_scales.extend(-scale for scale in positive_scales)
        if is_rank0:
            _log.info(
                "[DGPO/trust/signed] round=%s direction=%s step=%s "
                "parameter_rms=%.6g scales=%s",
                adaptive_state.reward_round_id,
                direction_index,
                global_step,
                direction_rms,
                [*evaluation_scales, 1.0],
            )
        try:
            for signed_scale in evaluation_scales:
                _assign_signed_trainable_direction_(
                    policy_core,
                    anchor_snapshot,
                    current_snapshot,
                    signed_scale,
                )
                candidate_pool = _materialize_adaptive_omnifold_pool(
                    omnifold_val_shard,
                    omnifold_val_loader_cfg,
                    model=model,
                    sampler=sampler,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    quota_events=adaptive_cfg.probe_max_events,
                    num_ddim_steps=num_ddim_val,
                    seed=int(probe_panel_seed),
                    include_pairwise_context=(
                        adaptive_cfg.periodic_pair_features_enabled
                    ),
                    include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                )
                log_reward = score_reward_on_pool(
                    omnifold_source.frozen_reward,
                    candidate_pool,
                    row_budget=adaptive_cfg.score_row_budget,
                )
                reward_mean = float(log_reward.mean().detach().cpu())
                raw_audit = fit_raw_policy_audit(
                    pool=candidate_pool,
                    model_builder=omnifold_source.model_builder,
                    cfg=adaptive_cfg,
                    device=device,
                    seed=int(adaptive_cfg.probe_seed),
                    progress_callback=lambda row, scale=signed_scale: (
                        _log_omnifold_fit_progress(
                            "signed_direction_audit",
                            {"signed_scale": float(scale), **dict(row)},
                            epoch_value=int(epoch),
                        )
                    ),
                )
                if signed_scale == 0.0:
                    anchor_audit = raw_audit
                    anchor_reward_mean = reward_mean
                else:
                    signed_candidates.append(
                        {
                            "signed_scale": float(signed_scale),
                            "raw_auc": float(raw_audit["raw_auc"]),
                            "raw_auc_gap": float(raw_audit["raw_auc_gap"]),
                            "raw_auc_se": float(
                                raw_audit["raw_auc_null_se_approx"]
                            ),
                            "reward_mean": reward_mean,
                        }
                    )
                    candidate_saturation.append(
                        float(raw_audit["raw_audit_saturated"])
                    )
                del log_reward, candidate_pool
        finally:
            # This probe must not change the optimizer, EMA, scheduler, or live
            # policy. Only trainable tensors were perturbed above.
            assign_params_(policy_core, current_snapshot)

        if anchor_audit is None or not math.isfinite(anchor_reward_mean):
            raise RuntimeError("signed-direction probe did not evaluate its anchor")
        signed_metrics = evaluate_signed_direction_probe(
            signed_candidates,
            anchor_raw_auc_gap=float(anchor_audit["raw_auc_gap"]),
            anchor_raw_auc_se=float(anchor_audit["raw_auc_null_se_approx"]),
            anchor_reward_mean=anchor_reward_mean,
            confidence_z=float(
                adaptive_cfg.trust_round_acceptance_confidence_z
            ),
        )
        signed_metrics.update(
            {
                "reference_trust/signed_probe/anchor_raw_auc": float(
                    anchor_audit["raw_auc"]
                ),
                "reference_trust/signed_probe/anchor_raw_balanced_accuracy": float(
                    anchor_audit["raw_balanced_accuracy"]
                ),
                "reference_trust/signed_probe/anchor_raw_audit_saturated": float(
                    anchor_audit["raw_audit_saturated"]
                ),
                "reference_trust/signed_probe/all_raw_audits_saturated": float(
                    float(anchor_audit["raw_audit_saturated"]) >= 0.5
                    and all(value >= 0.5 for value in candidate_saturation)
                ),
                "reference_trust/signed_probe/direction_rms": direction_rms,
                "reference_trust/signed_probe/reward_round_id": float(
                    adaptive_state.reward_round_id
                ),
                "reference_trust/signed_probe/direction_index": float(
                    direction_index
                ),
                "reference_trust/signed_probe/global_step": float(global_step),
            }
        )
        adaptive_state.trust_signed_probe_round_id = int(
            adaptive_state.reward_round_id
        )
        adaptive_state.trust_signed_probe_direction_index = direction_index
        adaptive_state.trust_signed_probe_global_step = int(global_step)
        if is_rank0:
            _log.info(
                "[DGPO/trust/signed] decision=%+.0f best_scale=%+.6g "
                "anchor_gap=%.6g best_gap=%.6g reward_misaligned=%s",
                signed_metrics["reference_trust/signed_probe/decision_code"],
                signed_metrics[
                    "reference_trust/signed_probe/best_overall_scale"
                ],
                signed_metrics[
                    "reference_trust/signed_probe/anchor_raw_auc_gap"
                ],
                signed_metrics[
                    "reference_trust/signed_probe/best_overall_raw_auc_gap"
                ],
                bool(
                    signed_metrics[
                        "reference_trust/signed_probe/reward_raw_misaligned"
                    ]
                ),
            )
        diagnostics.update(signed_metrics)
        return diagnostics

    def _run_adaptive_cycle(
        *,
        epoch: int,
        baseline_only: bool = False,
        raw_baseline_only: bool = False,
        checkpoint_next_epoch: int | None = None,
        allow_classifier_trust: bool = True,
        force_refit: bool = False,
        force_reason: str | None = None,
        extragradient_batch: dict[str, Any] | None = None,
        diagnostic_raw_only: bool = False,
        force_gradient_conflict: bool = False,
    ) -> dict[str, Any]:
        nonlocal global_step, reward_checkpoint_metadata
        nonlocal pending_resume_trust_diagnostics
        nonlocal last_gradient_conflict_snapshot
        if not adaptive_cfg.enabled or omnifold_source is None:
            return {}
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
            baseline_probe_auc_gap,
            build_reference_trust_pool,
            evaluate_signed_direction_recovery,
            fit_reference_trust_audit,
            fit_raw_policy_audit,
            probe_installed_reward,
            record_extragradient_rejection,
            reward_refit_due_to_age,
            reward_round_budget_exhausted,
            run_adaptive_refit,
            scheduled_raw_refit_due,
            should_probe_epoch,
            should_skip_incumbent_probe,
            should_run_raw_only_monitor,
            update_classifier_trust_controller,
            update_controller,
            update_raw_plateau_controller,
            shrink_trust_radius_on_raw_best,
            raw_best_trust_shrink_due,
            global_raw_candidate_due,
            update_trust_trajectory_candidate,
        )

        if is_rank0:
            _log.info(
                "[DGPO/omnifold] K=1 adapter monitor at epoch=%s "
                "(raw_cap=%s trust_cap=%s)",
                epoch,
                adaptive_cfg.probe_max_events,
                adaptive_cfg.classifier_trust_probe_max_events,
            )
        # A fixed event/noise/classifier-seed panel makes successive raw-AUC
        # measurements comparable.  Legacy controllers retain epoch-varying
        # probes unless trajectory search is explicitly enabled.
        fixed_audit_panel = bool(
            adaptive_cfg.fixed_audit_panel
            or adaptive_cfg.trust_trajectory_search_enabled
        )
        probe_panel_seed = (
            int(adaptive_cfg.probe_seed)
            if fixed_audit_panel
            else int(adaptive_cfg.probe_seed) + 10000 * max(int(epoch), 0)
        )
        fixed_schedule_age_trigger, fixed_schedule_age_diagnostics = (
            scheduled_raw_refit_due(
                adaptive_state,
                cfg=adaptive_cfg,
                epoch=int(epoch),
                baseline_only=bool(raw_baseline_only or baseline_only),
            )
        )
        direct_fixed_schedule_refit = bool(
            adaptive_cfg.fixed_schedule_skip_staleness_audit
            and not baseline_only
            and not raw_baseline_only
            and (force_refit or fixed_schedule_age_trigger)
        )
        raw_plateau_cycle = bool(
            adaptive_cfg.monitor_mode == "raw_plateau_refit"
            and adaptive_state.calibrated
            and (raw_baseline_only or not baseline_only)
            and not force_refit
            and not direct_fixed_schedule_refit
        )
        classifier_trust_due = bool(
            raw_plateau_cycle
            and not raw_baseline_only
            and allow_classifier_trust
            and adaptive_cfg.classifier_trust_enabled
            and should_probe_epoch(
                int(epoch),
                adaptive_cfg.classifier_trust_every_n_epochs,
            )
        )
        paired_classifier_trust_pool = bool(
            classifier_trust_due
        )
        if paired_classifier_trust_pool and is_rank0:
            _log.info(
                "[DGPO/omnifold] materializing current/reference classifier-trust "
                "pool from one event pass with common DDIM noise."
            )
        materialized_score_pool = _materialize_adaptive_omnifold_pool(
            omnifold_val_shard,
            omnifold_val_loader_cfg,
            model=model,
            paired_reference_model=(
                round_ref_model if paired_classifier_trust_pool else None
            ),
            sampler=sampler,
            device=device,
            world_size=world_size,
            rank=rank,
            quota_events=adaptive_cfg.probe_max_events,
            paired_reference_quota_events=(
                adaptive_cfg.classifier_trust_probe_max_events
                if paired_classifier_trust_pool
                else None
            ),
            num_ddim_steps=num_ddim_val,
            seed=probe_panel_seed,
            include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
            include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
            collect_policy_noise_mask=gradient_conflict_cfg.enabled,
        )
        reference_pool = None
        classifier_trust_current_pool = None
        if paired_classifier_trust_pool:
            if not isinstance(materialized_score_pool, tuple):
                raise RuntimeError(
                    "classifier trust requested a paired adaptive pool"
                )
            (
                score_pool,
                classifier_trust_current_pool,
                reference_pool,
            ) = materialized_score_pool
        else:
            score_pool = materialized_score_pool

        def _selection_blind_fixed_schedule_audit() -> dict[str, Any]:
            """Measure the live policy without changing any refit decision."""

            training_pool = _materialize_raw_audit_training_pool(
                omnifold_train_shard, omnifold_train_loader_cfg, cfg=adaptive_cfg,
                model=model, sampler=sampler, device=device, world_size=world_size,
                rank=rank, num_ddim_steps=num_ddim_val, panel_seed=probe_panel_seed,
                evaluation_events=score_pool.n_events,
            )
            raw_probe = fit_raw_policy_audit(
                pool=score_pool,
                model_builder=omnifold_source.model_builder,
                cfg=adaptive_cfg,
                device=device,
                seed=int(adaptive_cfg.probe_seed),
                warm_start_cache=None,
                **({"training_pool": training_pool} if training_pool is not None else {}),
                progress_callback=lambda row: _log_omnifold_fit_progress(
                    "raw_staleness_audit",
                    row,
                    epoch_value=int(epoch),
                ),
            )
            history_row = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "reward_round_id": float(adaptive_state.reward_round_id),
                "decision_recalibrate": 0.0,
                "fixed_schedule_diagnostic_only": 1.0,
                "measurement_only_raw_audit": 1.0,
                **{key: float(value) for key, value in raw_probe.items()},
            }
            adaptive_state.probe_history.append(history_row)
            if len(adaptive_state.probe_history) > 256:
                del adaptive_state.probe_history[:-256]
            trajectory_rows = [
                row
                for row in adaptive_state.probe_history
                if float(
                    row.get("fixed_schedule_diagnostic_only", 0.0)
                )
                >= 0.5
                and math.isfinite(float(row.get("raw_auc_gap", float("nan"))))
                # An old small-pool audit is not a baseline for the larger
                # matched-fold protocol after a resume/config update.
                and float(row.get("raw_audit_uses_omnifold_fold", 0.0))
                == float(raw_probe.get("raw_audit_uses_omnifold_fold", 0.0))
                and float(row.get("raw_audit_training_fold", 0.0))
                == float(raw_probe.get("raw_audit_training_fold", 0.0))
            ]
            trajectory_rows.sort(key=lambda row: float(row["global_step"]))
            trajectory_steps = [float(row["global_step"]) for row in trajectory_rows]
            trajectory_gaps = [float(row["raw_auc_gap"]) for row in trajectory_rows]
            gap_slope_per_10 = float("nan")
            if len(trajectory_rows) >= 2:
                step_mean = sum(trajectory_steps) / len(trajectory_steps)
                gap_mean = sum(trajectory_gaps) / len(trajectory_gaps)
                step_variance = sum(
                    (value - step_mean) ** 2 for value in trajectory_steps
                )
                if step_variance > 0.0:
                    gap_slope_per_10 = 10.0 * sum(
                        (step - step_mean) * (gap - gap_mean)
                        for step, gap in zip(trajectory_steps, trajectory_gaps)
                    ) / step_variance
            return {
                **{
                    f"staleness/{key}": value
                    for key, value in history_row.items()
                },
                "staleness/decision": "fixed_schedule_diagnostic_only",
                "staleness/trigger_recalibration": 0.0,
                "staleness/fixed_schedule_diagnostic_raw_audit": 1.0,
                "staleness/diagnostic_selection_blind": 1.0,
                "staleness/raw_audit_trajectory_points": float(
                    len(trajectory_rows)
                ),
                "staleness/raw_auc_gap_step0": float(trajectory_gaps[0]),
                "staleness/raw_auc_gap_change_from_step0": float(
                    trajectory_gaps[-1] - trajectory_gaps[0]
                ),
                "staleness/raw_auc_gap_slope_per_10_steps": float(
                    gap_slope_per_10
                ),
                "audit/raw_auc": float(raw_probe["raw_auc"]),
                "audit/raw_auc_gap": float(raw_probe["raw_auc_gap"]),
                "audit/saturated": float(raw_probe["raw_audit_saturated"]),
                "audit/training_ready": float(
                    raw_probe["raw_audit_training_ready"]
                ),
                "audit/epoch": float(epoch),
                "audit/policy_step": float(global_step),
                "audit/trajectory_points": float(len(trajectory_rows)),
                "audit/raw_auc_gap_change_from_step0": float(
                    trajectory_gaps[-1] - trajectory_gaps[0]
                ),
                "audit/raw_auc_gap_slope_per_10_steps": float(
                    gap_slope_per_10
                ),
            }

        if diagnostic_raw_only or (
            adaptive_cfg.audit_fit.get("training_population") == "omnifold_fold"
            and not direct_fixed_schedule_refit
        ):
            if not (
                adaptive_cfg.fixed_schedule_log_raw_audit
                or adaptive_cfg.monitor_mode == "raw_only"
            ):
                raise ValueError(
                    "diagnostic_raw_only requires "
                    "trigger.fixed_schedule_log_raw_audit=true or "
                    "monitor_mode=raw_only"
                )
            _assert_current_reward_reference_pairing(
                "adaptive_cycle_fixed_schedule_diagnostic"
            )
            return _selection_blind_fixed_schedule_audit()
        if direct_fixed_schedule_refit:
            if gradient_conflict_cfg.enabled:
                raise RuntimeError(
                    "fixed-schedule audit skipping is incompatible with the "
                    "gradient-conflict monitor"
                )
            round_budget_exhausted = reward_round_budget_exhausted(
                adaptive_state,
                max_reward_rounds=adaptive_cfg.max_reward_rounds,
            )
            diagnostics: dict[str, Any] = {
                **fixed_schedule_age_diagnostics,
                "staleness/decision": "fixed_schedule_refit",
                "staleness/trigger_recalibration": float(
                    not round_budget_exhausted
                ),
                "staleness/trigger_reason": (
                    str(force_reason or "forced_protocol_refit")
                    if force_refit
                    else "max_reward_age"
                ),
                "staleness/reward_round_id": float(
                    adaptive_state.reward_round_id
                ),
                "staleness/max_reward_rounds": float(
                    adaptive_cfg.max_reward_rounds
                    if adaptive_cfg.max_reward_rounds is not None
                    else 0
                ),
                "staleness/reward_round_budget_exhausted": float(
                    round_budget_exhausted
                ),
                "staleness/incumbent_weighted_audit_skipped": 1.0,
                "staleness/independent_raw_audit_skipped": 1.0,
                "staleness/fixed_schedule_without_audit": 1.0,
            }
            if adaptive_cfg.fixed_schedule_log_raw_audit:
                diagnostics.update(_selection_blind_fixed_schedule_audit())
                diagnostics.update(
                    {
                        "staleness/decision": "fixed_schedule_refit",
                        "staleness/trigger_recalibration": float(
                            not round_budget_exhausted
                        ),
                        "staleness/independent_raw_audit_skipped": 0.0,
                        "staleness/fixed_schedule_without_audit": 0.0,
                        "staleness/fixed_schedule_with_diagnostic_audit": 1.0,
                    }
                )
            if round_budget_exhausted:
                adaptive_state.last_decision = "reward_round_budget_complete"
                diagnostics["staleness/decision"] = adaptive_state.last_decision
                diagnostics["staleness/trigger_recalibration"] = 0.0
                diagnostics["staleness/trigger_reason"] = "reward_round_budget"
                _assert_current_reward_reference_pairing(
                    "adaptive_cycle_fixed_schedule_budget_complete"
                )
                return diagnostics

            if is_rank0:
                _log.info(
                    "[DGPO/omnifold] fixed-schedule refit at epoch=%s step=%s; "
                    "%s",
                    epoch,
                    global_step,
                    (
                        "logging selection-blind repeated raw audit and "
                        "skipping the incumbent weighted audit"
                        if adaptive_cfg.fixed_schedule_log_raw_audit
                        else "skipping incumbent weighted and independent raw audits"
                    ),
                )
            policy_snapshot = _snapshot_policy_state_dict(model)
            fit_pool_seed = int(adaptive_cfg.seed) + int(
                adaptive_state.recalibration_count
            )
            fit_pool = _materialize_adaptive_omnifold_pool(
                omnifold_train_shard,
                omnifold_train_loader_cfg,
                model=model,
                sampler=sampler,
                device=device,
                world_size=world_size,
                rank=rank,
                quota_events=(
                    None if fixed_omnifold_pool else adaptive_cfg.pool_events
                ),
                num_ddim_steps=num_ddim,
                seed=fit_pool_seed,
                include_pairwise_context=(
                    adaptive_cfg.periodic_pair_features_enabled
                ),
                include_visible_pair_rest_frame=(
                    adaptive_cfg.visible_pair_rest_frame_enabled
                ),
            )
            if adaptive_cfg.single_pool_train_validation:
                score_pool = fit_pool
            if adaptive_cfg.refit_score_events == adaptive_cfg.probe_max_events:
                refit_score_pool = score_pool
            else:
                refit_score_pool = _materialize_adaptive_omnifold_pool(
                    omnifold_val_shard,
                    omnifold_val_loader_cfg,
                    model=model,
                    sampler=sampler,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    quota_events=adaptive_cfg.refit_score_events,
                    num_ddim_steps=num_ddim_val,
                    seed=int(probe_panel_seed) + 1_000_003,
                    include_pairwise_context=(
                        adaptive_cfg.periodic_pair_features_enabled
                    ),
                    include_visible_pair_rest_frame=(
                        adaptive_cfg.visible_pair_rest_frame_enabled
                    ),
                )
            refit_diagnostics = run_adaptive_refit(
                global_step=int(global_step),
                state=adaptive_state,
                cfg=adaptive_cfg,
                reward_source=omnifold_source,
                round_ref_model=round_ref_model,
                policy_snapshot_state_dict=policy_snapshot,
                fit_pool=fit_pool,
                score_pool=refit_score_pool,
                baseline_pool=None,
                epoch=int(epoch),
                device=device,
                world_size=world_size,
                enforce_round_acceptance=False,
                progress_callback=lambda phase, row: _log_omnifold_fit_progress(
                    phase,
                    row,
                    epoch_value=int(epoch),
                ),
            )
            diagnostics.update(refit_diagnostics)
            refit_accepted = bool(
                float(refit_diagnostics.get("omnifold/accepted", 0.0)) >= 0.5
            )
            if not refit_accepted and (
                adaptive_cfg.scheduled_refit_fail_closed
                or (force_refit and adaptive_cfg.refit_once_fail_closed)
            ):
                raise RuntimeError(
                    "fixed-schedule fresh reward refit failed; refusing to "
                    "continue with a stale reward: "
                    + str(
                        refit_diagnostics.get(
                            "omnifold/accept_reason",
                            "unspecified classifier gate",
                        )
                    )
                )
            if refit_accepted:
                reference_trust_probe_cache.clear()
                adaptive_state.trust_probe_payload = None
                adaptive_state.trust_probe_round_id = -1
                diagnostics.update(
                    _reset_optimizer_after_reward_install(
                        optimizer,
                        cfg=adaptive_cfg,
                        accepted=True,
                        adaptive_state=adaptive_state,
                    )
                )
                adaptive_state.audit_protocol_signature = current_audit_signature
                adaptive_state.last_decision = "fixed_schedule_refit_installed"
                diagnostics["staleness/decision"] = adaptive_state.last_decision
            else:
                adaptive_state.last_decision = "fixed_schedule_refit_rejected"
                diagnostics["staleness/decision"] = adaptive_state.last_decision
            reward_checkpoint_metadata = reward_agg.checkpoint_metadata()
            if pending_resume_trust_diagnostics:
                diagnostics.update(pending_resume_trust_diagnostics)
                pending_resume_trust_diagnostics = {}
            _assert_current_reward_reference_pairing(
                "adaptive_cycle_fixed_schedule_refit"
            )
            return diagnostics
        if raw_plateau_cycle:
            # Capture before the routine monitor adapts to the current candidate.
            confirmation_start_cache = (
                copy.deepcopy(adaptive_state.raw_monitor_state)
                if adaptive_cfg.raw_global_confirmation_warm_start else None
            )
            # A periodic gradient probe needs the fitted judge weights.  When
            # the scientific audit is configured cold, capture those weights
            # in a transient cache and discard it after this boundary; never
            # feed them into the next audit.
            current_raw_monitor_cache = (
                adaptive_state.raw_monitor_state
                if adaptive_cfg.raw_monitor_warm_start
                else ({} if gradient_conflict_cfg.enabled else None)
            )
            raw_probe = fit_raw_policy_audit(
                pool=score_pool,
                model_builder=omnifold_source.model_builder,
                cfg=adaptive_cfg,
                device=device,
                seed=int(adaptive_cfg.probe_seed),
                warm_start_cache=current_raw_monitor_cache,
                progress_callback=lambda row: _log_omnifold_fit_progress(
                    "raw_staleness_monitor",
                    row,
                    epoch_value=int(epoch),
                ),
            )
            # Keep the same event panel for pre/post install. Pool candidates
            # are not reused: the probe generates from the live policy with CRN.
            gradient_panel = score_pool if gradient_conflict_cfg.monitor_refit_lifecycle else None
            gradient_cache = {}

            def _measure_gradient_phase(phase, panel):
                nonlocal last_gradient_conflict_snapshot
                from RL.DGPO_neutrino.gradient_conflict import probe_gradients, phase_metrics
                from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import peft_bank_factory
                round_id = int(adaptive_state.reward_round_id)
                reused_judge = bool(phase == "post_warmup"
                                    and adaptive_state.gradient_post_install_probe_round_id == round_id
                                    and adaptive_state.gradient_lifecycle_monitor_state)
                judge_cache = (
                    adaptive_state.gradient_lifecycle_monitor_state
                    if reused_judge
                    else current_raw_monitor_cache
                )
                if judge_cache is None:
                    judge_cache = {}
                judge_step = adaptive_state.gradient_lifecycle_monitor_step if reused_judge else int(global_step)
                judge_auc = adaptive_state.gradient_lifecycle_raw_auc if reused_judge else raw_probe["raw_auc"]
                snapshot = (int(global_step), round_id, judge_step, reused_judge)
                overrides = {key: adaptive_cfg.audit_fit[key] for key in (
                    "head_dropout", "topology_dropout", "decoder_hidden_dim", "decoder_layers",
                    "decoder_heads", "periodic_pair_features", "topology_fourier_embedding",
                    "topology_direct_logit", "topology_context_residual_scale",
                    "topology_conditioning", "topology_pair_token", "relation_token_count", "visible_pair_rest_frame",
                    "topology_max_harmonic", "topology_include_theta_pair", "topology_theta_fourier",
                    "topology_hidden_dim", "topology_embedding_dim",
                    "topology_fusion_hidden_dim",
                    "train_layernorm", "train_encoder",
                    "train_grouped_sequential_embedding", "train_invisible_projector",
                    "train_angular_conditioning",
                    "train_backbone", "train_last_pet_block", "asymmetric_attention",
                ) if adaptive_cfg.audit_fit.get(key) is not None}
                if is_rank0:
                    _log.info("[DGPO/gradient_conflict] phase=%s step=%s round=%s blocks=%s global_events/block=%s K=%s timesteps=%s (diagnostic only)",
                              phase, global_step, round_id, gradient_conflict_cfg.blocks,
                              gradient_conflict_cfg.events_per_block, K, num_train_timesteps)
                gradient_metrics = gradient_cache.get(snapshot)
                if gradient_metrics is None:
                    gradient_metrics = probe_gradients(
                        cfg=gradient_conflict_cfg, model=model, core=_unwrap_core_evenet(model),
                        reference=round_ref_model, pool=panel,
                        monitor_factory=peft_bank_factory(omnifold_source.model_builder, panel.packing_spec,
                                                          classifier_overrides=overrides),
                        monitor_cache=judge_cache,
                        reward_compute=lambda candidates, batch: reward_agg.compute(candidates, batch)[0],
                        generate=generate_neutrino_candidates, evaluate=policy_evaluation_step,
                        sampler=sampler, device=device, dtype=dtype, rank=rank, world_size=world_size,
                        K=K, beta=beta, num_ddim_steps=num_ddim, rollout_parallel_chains=rollout_parallel_chains,
                        num_train_timesteps=num_train_timesteps, t_min=policy_eval_t_min_cfg, t_max=policy_eval_t_max_cfg,
                        advantage_estimator=advantage_estimator, adv_clip_max=adv_clip_max_cfg,
                        trust_coefficient=reference_trust_coefficient, global_step=global_step,
                        reward_round_id=round_id, raw_auc=judge_auc,
                        raw_ready=bool(
                            raw_probe["raw_audit_training_ready"]
                            if adaptive_cfg.audit_fit.get("training_readiness")
                            is not None
                            else raw_probe["raw_audit_saturated"]
                        ),
                    )
                    gradient_cache[snapshot] = gradient_metrics
                gradient_metrics = dict(gradient_metrics)
                gradient_metrics["gradient_conflict/judge_fit_global_step"] = judge_step
                gradient_metrics["gradient_conflict/judge_reused_from_install"] = float(reused_judge)
                gradient_metrics["gradient_conflict/raw_auc_is_current"] = float(
                    not reused_judge and not (phase == "post_install" and adaptive_cfg.raw_rollback_to_best_on_plateau)
                )
                gradient_metrics["gradient_conflict/warmup_completed_updates"] = adaptive_state.policy_warmup_completed_updates
                if phase == "periodic":
                    last_gradient_conflict_snapshot = (int(global_step), round_id)
                ran = gradient_metrics.get("gradient_conflict/ran", 0) >= .5
                if ran and phase == "post_install":
                    adaptive_state.gradient_post_install_probe_round_id = round_id
                    adaptive_state.gradient_lifecycle_monitor_state = copy.deepcopy(judge_cache)
                    adaptive_state.gradient_lifecycle_monitor_step = judge_step
                    adaptive_state.gradient_lifecycle_raw_auc = judge_auc
                if ran and phase == "post_warmup":
                    adaptive_state.gradient_warmup_probe_round_id = round_id
                    adaptive_state.gradient_lifecycle_monitor_state = {}
                logged_metrics = phase_metrics(gradient_metrics, phase)
                logged_metrics["epoch"] = int(epoch)
                if is_rank0:
                    _log.info("[DGPO/gradient_conflict] %s", logged_metrics)
                    if wandb_mod is not None:
                        _wandb_log_auxiliary(wandb_mod, logged_metrics, current_global_step=int(global_step))

            gradient_snapshot = (int(global_step), int(adaptive_state.reward_round_id))
            if (gradient_conflict_cfg.enabled and global_step > 0
                    and (
                        force_gradient_conflict
                        or global_step % gradient_conflict_cfg.every_n_steps == 0
                    )
                    and gradient_snapshot != last_gradient_conflict_snapshot):
                _measure_gradient_phase("periodic", score_pool)
            from RL.DGPO_neutrino.gradient_conflict import lifecycle_probe_due
            lifecycle_phase = lifecycle_probe_due(
                gradient_conflict_cfg, adaptive_state, warmup_steps=adaptive_cfg.policy_warmup_steps,
            )
            if lifecycle_phase is not None:
                _measure_gradient_phase(lifecycle_phase, score_pool)
            raw_checkpoint_path = ""
            if adaptive_cfg.raw_rollback_to_best_on_plateau or adaptive_cfg.raw_best_scope == "global":
                assert save_dir_raw
                raw_checkpoint_path = str(
                    Path(str(save_dir_raw)).expanduser().resolve()
                    / dgpo_snapshot_checkpoint_name(
                        last_completed_epoch=int(epoch),
                        dgpo_next_epoch=(int(epoch) + 1 if checkpoint_next_epoch is None else checkpoint_next_epoch),
                        global_step=int(global_step),
                    )
                )
            confirmation = None
            confirmation_metrics = {}
            if global_raw_candidate_due(adaptive_state, raw_probe, cfg=adaptive_cfg):
                best_path = _resolve_raw_best_policy_checkpoint(
                    adaptive_state, save_dir_raw, global_scope=True, source_dirs=global_best_source_dirs,
                )
                confirmation, confirmation_metrics = _confirm_global_raw_candidate(
                    model=model, comparison_model=round_ref_model, best_checkpoint=best_path,
                    min_delta=adaptive_cfg.raw_improvement_min_delta,
                    initial_judge_cache=confirmation_start_cache,
                    materialize_pair=lambda candidate, incumbent: _materialize_adaptive_omnifold_pool(
                        omnifold_val_shard, omnifold_val_loader_cfg, model=candidate,
                        paired_reference_model=incumbent, sampler=sampler, device=device,
                        world_size=world_size, rank=rank, quota_events=adaptive_cfg.probe_max_events,
                        num_ddim_steps=num_ddim_val, seed=probe_panel_seed,
                        include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
                        include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                    ),
                    fit_judge=lambda pool, phase, warm_start_cache=None: fit_raw_policy_audit(
                        pool=pool, model_builder=omnifold_source.model_builder,
                        cfg=adaptive_cfg, device=device, seed=int(adaptive_cfg.probe_seed),
                        warm_start_cache=warm_start_cache,
                        fit_overrides=adaptive_cfg.raw_global_confirmation_fit,
                        progress_callback=lambda row: _log_omnifold_fit_progress(phase, row, epoch_value=int(epoch)),
                    ),
                )
            raw_trigger, diagnostics = update_raw_plateau_controller(
                adaptive_state,
                raw_probe,
                cfg=adaptive_cfg,
                epoch=int(epoch),
                global_step=int(global_step),
                checkpoint_path=raw_checkpoint_path,
                checkpoint_next_epoch=checkpoint_next_epoch,
                global_confirmation=confirmation,
            )
            diagnostics.update(confirmation_metrics)
            if adaptive_state.raw_global_stop_requested:
                # Keep the latest policy/reward/reference consistent; the best
                # policy remains in its separate immutable checkpoint. Caller
                # saves this terminal state before leaving the training loop.
                _assert_current_reward_reference_pairing("global_best_stagnation_stop")
                return diagnostics
            if adaptive_cfg.trust_boundary_enabled and adaptive_cfg.trust_radius_mode == "best_decay":
                probe_distance = float("nan")
                # Only a global record needs a new fixed-probe evaluation.
                if raw_best_trust_shrink_due(
                    adaptive_state, cfg=adaptive_cfg, diagnostics=diagnostics,
                    global_step=int(global_step),
                ):
                    fixed_probe = reference_trust_probe_cache.get("probe")
                    if fixed_probe is None:
                        raise RuntimeError("new-best trust shrink requires the active fixed reference probe")
                    probe_distance, _, _ = _measure_reference_trust_probe(
                        model, fixed_probe, world_size=world_size,
                        distance=adaptive_cfg.trust_distance,
                    )
                decay_metrics = shrink_trust_radius_on_raw_best(
                    adaptive_state, cfg=adaptive_cfg, diagnostics=diagnostics,
                    current_distance=probe_distance, global_step=int(global_step),
                )
                diagnostics.update(decay_metrics)
                if is_rank0 and decay_metrics.get("reference_trust/best_decay/new_best", 0.0):
                    _log.info(
                        "[DGPO/trust] new global raw best step=%s: target=%.6g effective=%.6g "
                        "probe_distance=%.6g feasibility_limited=%s",
                        global_step, decay_metrics["reference_trust/best_decay/target"],
                        adaptive_state.trust_current_delta, probe_distance,
                        bool(decay_metrics["reference_trust/best_decay/feasibility_limited"]),
                    )
            classifier_trust_trigger = False
            if classifier_trust_due:
                if (
                    classifier_trust_current_pool is None
                    or reference_pool is None
                ):
                    raise RuntimeError(
                        "classifier trust paired reference pool is unavailable"
                    )
                classifier_trust_pool = build_reference_trust_pool(
                    classifier_trust_current_pool,
                    reference_pool,
                )
                classifier_trust_probe = fit_reference_trust_audit(
                    pool=classifier_trust_pool,
                    model_builder=omnifold_source.model_builder,
                    cfg=adaptive_cfg,
                    device=device,
                    seed=int(adaptive_cfg.probe_seed),
                    progress_callback=lambda row: _log_omnifold_fit_progress(
                        "reference_trust_monitor",
                        row,
                        epoch_value=int(epoch),
                    ),
                )
                classifier_trust_trigger, classifier_trust_diagnostics = (
                    update_classifier_trust_controller(
                        adaptive_state,
                        classifier_trust_probe,
                        cfg=adaptive_cfg,
                        epoch=int(epoch),
                    )
                )
                diagnostics.update(
                    {
                        **{
                            f"classifier_trust/{key}": float(value)
                            for key, value in classifier_trust_probe.items()
                        },
                        **classifier_trust_diagnostics,
                    }
                )
                del (
                    classifier_trust_pool,
                    classifier_trust_current_pool,
                    reference_pool,
                )
            elif adaptive_cfg.classifier_trust_enabled:
                diagnostics.update(
                    {
                        "classifier_trust/skipped_cadence": 1.0,
                        "classifier_trust/every_n_epochs": float(
                            adaptive_cfg.classifier_trust_every_n_epochs
                        ),
                    }
                )
            age_trigger, age_diagnostics = scheduled_raw_refit_due(
                adaptive_state, cfg=adaptive_cfg, epoch=int(epoch),
                baseline_only=raw_baseline_only,
            )
            diagnostics.update(age_diagnostics)
            trigger_refit = bool(raw_trigger or classifier_trust_trigger or age_trigger)
            if age_trigger:
                adaptive_state.last_decision = "max_reward_age_recalibrate"
                diagnostics.update({
                    "staleness/decision": adaptive_state.last_decision,
                    "staleness/trigger_recalibration": 1.0,
                    "staleness/trigger_reason": "max_reward_age",
                })
                if is_rank0:
                    _log.info(
                        "[DGPO/omnifold] scheduled refit at epoch=%s step=%s "
                        "reward_age=%s; rollback_to_best=%s",
                        epoch, global_step, age_diagnostics["staleness/reward_age_epochs"],
                        adaptive_cfg.raw_rollback_to_best_on_plateau,
                    )
            if classifier_trust_trigger:
                diagnostics.update(
                    {
                        "staleness/decision": "classifier_trust_recalibrate",
                        "staleness/trigger_recalibration": 1.0,
                        "staleness/trigger_reason": "classifier_trust",
                    }
                )
                adaptive_state.last_decision = "classifier_trust_recalibrate"
            round_budget_exhausted = reward_round_budget_exhausted(
                adaptive_state,
                max_reward_rounds=adaptive_cfg.max_reward_rounds,
            )
            diagnostics.update(
                {
                    "staleness/max_reward_rounds": float(
                        adaptive_cfg.max_reward_rounds
                        if adaptive_cfg.max_reward_rounds is not None
                        else 0
                    ),
                    "staleness/reward_round_budget_exhausted": float(
                        round_budget_exhausted
                    ),
                }
            )
            if round_budget_exhausted and trigger_refit:
                trigger_refit = False
                adaptive_state.last_decision = "reward_round_budget_complete"
                diagnostics.update(
                    {
                        "staleness/decision": adaptive_state.last_decision,
                        "staleness/trigger_recalibration": 0.0,
                        "staleness/trigger_reason": "reward_round_budget",
                    }
                )
            if not trigger_refit:
                if pending_resume_trust_diagnostics:
                    diagnostics.update(pending_resume_trust_diagnostics)
                    pending_resume_trust_diagnostics = {}
                _assert_current_reward_reference_pairing(
                    "adaptive_cycle_raw_plateau_monitor"
                )
                if is_rank0:
                    _log.info(
                        "[DGPO/omnifold] epoch=%s raw_gap=%.6f best=%.6f "
                        "plateau=%s/%s trust_trigger=%s round=%s",
                        epoch,
                        float(raw_probe["raw_auc_gap"]),
                        float(adaptive_state.raw_best_auc_gap),
                        adaptive_state.raw_no_improvement_streak,
                        int(diagnostics["staleness/raw_no_improvement_patience"]),
                        classifier_trust_trigger,
                        adaptive_state.reward_round_id,
                    )
                return diagnostics

            if gradient_conflict_cfg.enabled and gradient_conflict_cfg.monitor_refit_lifecycle:
                _measure_gradient_phase("pre_refit", gradient_panel)

            if raw_trigger and adaptive_cfg.raw_rollback_to_best_on_plateau:
                selected_checkpoint = _resolve_raw_best_policy_checkpoint(
                    adaptive_state,
                    save_dir_raw,
                    global_scope=adaptive_cfg.raw_best_scope == "global",
                    source_dirs=global_best_source_dirs,
                )
                (
                    loaded_tensors,
                    cleared_optimizer_states,
                    ema_restored,
                ) = _rewind_policy_for_raw_best_refit(
                    model,
                    optimizer,
                    selected_checkpoint,
                    ema_save=ema_save,
                    ema_rollout=ema_rollout,
                    clear_optimizer=adaptive_cfg.raw_best_scope != "global",
                )
                adaptive_state.raw_plateau_rollbacks += 1
                # The routine score pool belongs to the plateau endpoint. It
                # cannot certify an OmniFold ratio whose denominator is the
                # restored best policy, so release it before regenerating the
                # configured residual-closure panel below.
                del score_pool
                score_pool = None
                diagnostics.update(
                    {
                        "staleness/raw_best_rollback_applied": 1.0,
                        "staleness/raw_best_rollback_epoch": float(
                            adaptive_state.raw_best_epoch
                        ),
                        "staleness/raw_best_rollback_global_step": float(
                            adaptive_state.raw_best_global_step
                        ),
                        "staleness/raw_best_rollback_auc_gap": float(
                            adaptive_state.raw_best_auc_gap
                        ),
                        "staleness/raw_best_rollback_loaded_tensors": float(
                            loaded_tensors
                        ),
                        "staleness/raw_best_rollback_optimizer_states_cleared": float(
                            cleared_optimizer_states
                        ),
                        "staleness/raw_best_rollback_ema_restored": float(
                            ema_restored
                        ),
                        "staleness/raw_plateau_rollbacks": float(
                            adaptive_state.raw_plateau_rollbacks
                        ),
                    }
                )
                if is_rank0:
                    _log.info(
                        "[DGPO/omnifold] raw plateau rewound to best epoch=%s "
                        "step=%s raw_gap=%.6g checkpoint=%s; cleared %s "
                        "optimizer states before fresh reward fitting",
                        adaptive_state.raw_best_epoch,
                        adaptive_state.raw_best_global_step,
                        adaptive_state.raw_best_auc_gap,
                        selected_checkpoint,
                        cleared_optimizer_states,
                    )
            else:
                diagnostics["staleness/raw_best_rollback_applied"] = 0.0

            # The restored best policy (or current policy when rollback is
            # disabled) is the new forward anchor. Fit a fresh q/policy ratio,
            # require residual closure, then atomically install the paired
            # reward/reference state.
            policy_snapshot = _snapshot_policy_state_dict(model)
            fit_pool_seed = int(adaptive_cfg.seed) + int(
                adaptive_state.recalibration_count
            )
            fit_pool = _materialize_adaptive_omnifold_pool(
                omnifold_train_shard,
                omnifold_train_loader_cfg,
                model=model,
                sampler=sampler,
                device=device,
                world_size=world_size,
                rank=rank,
                quota_events=(
                    None if fixed_omnifold_pool else adaptive_cfg.pool_events
                ),
                num_ddim_steps=num_ddim,
                seed=fit_pool_seed,
                include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
                include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
            )
            if adaptive_cfg.single_pool_train_validation:
                score_pool = fit_pool  # run_adaptive_refit performs the disjoint 80/20 split
            if adaptive_cfg.refit_score_events == adaptive_cfg.probe_max_events:
                if score_pool is None:
                    score_pool = _materialize_adaptive_omnifold_pool(
                        omnifold_val_shard,
                        omnifold_val_loader_cfg,
                        model=model,
                        sampler=sampler,
                        device=device,
                        world_size=world_size,
                        rank=rank,
                        quota_events=adaptive_cfg.probe_max_events,
                        num_ddim_steps=num_ddim_val,
                        seed=int(probe_panel_seed),
                        include_pairwise_context=(
                            adaptive_cfg.periodic_pair_features_enabled
                        ),
                        include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                    )
                refit_score_pool = score_pool
            else:
                refit_score_pool = _materialize_adaptive_omnifold_pool(
                    omnifold_val_shard,
                    omnifold_val_loader_cfg,
                    model=model,
                    sampler=sampler,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    quota_events=adaptive_cfg.refit_score_events,
                    num_ddim_steps=num_ddim_val,
                    seed=int(probe_panel_seed) + 1_000_003,
                    include_pairwise_context=(
                        adaptive_cfg.periodic_pair_features_enabled
                    ),
                    include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                )
            refit_diagnostics = run_adaptive_refit(
                global_step=int(global_step),
                state=adaptive_state,
                cfg=adaptive_cfg,
                reward_source=omnifold_source,
                round_ref_model=round_ref_model,
                policy_snapshot_state_dict=policy_snapshot,
                fit_pool=fit_pool,
                score_pool=refit_score_pool,
                # This controller intentionally has no acceptance/baseline
                # audit. The scheduled held-out residual AUC is the closure gate.
                baseline_pool=None,
                epoch=int(epoch),
                device=device,
                world_size=world_size,
                enforce_round_acceptance=False,
                progress_callback=lambda phase, row: _log_omnifold_fit_progress(
                    phase,
                    row,
                    epoch_value=int(epoch),
                ),
            )
            diagnostics.update(refit_diagnostics)
            refit_accepted = bool(
                float(refit_diagnostics.get("omnifold/accepted", 0.0)) >= 0.5
            )
            if not refit_accepted and adaptive_cfg.scheduled_refit_fail_closed:
                raise RuntimeError(
                    "scheduled fresh reward refit failed; refusing to continue "
                    "the fixed-round experiment with a stale reward: "
                    + str(
                        refit_diagnostics.get(
                            "omnifold/accept_reason", "unspecified classifier gate"
                        )
                    )
                )
            if refit_accepted:
                if adaptive_cfg.raw_best_scope == "global":
                    adaptive_state.raw_global_refit_pending = True
                reference_trust_probe_cache.clear()
                adaptive_state.trust_probe_payload = None
                adaptive_state.trust_probe_round_id = -1
                diagnostics.update(_reset_optimizer_after_reward_install(
                    optimizer, cfg=adaptive_cfg, accepted=True, adaptive_state=adaptive_state,
                ))
                adaptive_state.audit_protocol_signature = current_audit_signature
                diagnostics["staleness/decision"] = "forward_recenter_installed"
                if gradient_conflict_cfg.enabled and gradient_conflict_cfg.monitor_refit_lifecycle:
                    _measure_gradient_phase("post_install", gradient_panel)
            else:
                diagnostics["staleness/decision"] = _raw_refit_failure_decision(
                    rollback_applied=bool(diagnostics.get("staleness/raw_best_rollback_applied", 0.0)),
                )
                if is_rank0:
                    _log.warning(
                        "[DGPO/omnifold] refit rejected without rollback; preserving "
                        "current policy, installed reward/reference and optimizer. "
                        "Retry remains governed by the existing raw monitor: %s",
                        refit_diagnostics.get("omnifold/accept_reason", "unspecified"),
                    )
            reward_checkpoint_metadata = reward_agg.checkpoint_metadata()
            if pending_resume_trust_diagnostics:
                diagnostics.update(pending_resume_trust_diagnostics)
                pending_resume_trust_diagnostics = {}
            _assert_current_reward_reference_pairing(
                "adaptive_cycle_raw_plateau_refit"
            )
            return diagnostics
        routine_raw_only = should_run_raw_only_monitor(
            adaptive_state,
            cfg=adaptive_cfg,
            baseline_only=baseline_only,
            force_refit=force_refit,
        )
        if routine_raw_only:
            frozen_metrics = {}
            if adaptive_cfg.iteration_one_only:
                from RL.DGPO_neutrino.omnifold_ztautau.adaptive import frozen_classifier_metrics
                frozen_metrics = frozen_classifier_metrics(
                    omnifold_source.frozen_reward, score_pool,
                    row_budget=adaptive_cfg.score_row_budget,
                )
            # A cold audit normally discards its fitted judge. Keep it only for
            # this boundary when the read-only gradient-direction probe needs
            # the judge weights; it is never reused by the next audit.
            current_raw_monitor_cache = (
                adaptive_state.raw_monitor_state
                if adaptive_cfg.raw_monitor_warm_start
                else ({} if gradient_conflict_cfg.enabled else None)
            )
            raw_probe = fit_raw_policy_audit(
                pool=score_pool,
                model_builder=omnifold_source.model_builder,
                cfg=adaptive_cfg,
                device=device,
                seed=(
                    adaptive_cfg.probe_seed
                    if fixed_audit_panel
                    else adaptive_cfg.probe_seed + max(int(epoch), 0)
                ),
                warm_start_cache=current_raw_monitor_cache,
                progress_callback=lambda row: _log_omnifold_fit_progress(
                    "raw_staleness_audit",
                    row,
                    epoch_value=int(epoch),
                ),
            )
            history_row = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "reward_round_id": float(adaptive_state.reward_round_id),
                "decision_recalibrate": 0.0,
                "measurement_only_raw_audit": 1.0,
                **frozen_metrics,
                **{key: float(value) for key, value in raw_probe.items()},
            }
            adaptive_state.probe_history.append(history_row)
            if len(adaptive_state.probe_history) > 256:
                del adaptive_state.probe_history[:-256]
            trajectory_rows = [
                row
                for row in adaptive_state.probe_history
                if float(row.get("measurement_only_raw_audit", 0.0)) >= 0.5
                and math.isfinite(float(row.get("raw_auc_gap", float("nan"))))
            ]
            trajectory_rows.sort(key=lambda row: float(row["global_step"]))
            trajectory_steps = [float(row["global_step"]) for row in trajectory_rows]
            trajectory_gaps = [float(row["raw_auc_gap"]) for row in trajectory_rows]
            gap_slope_per_10 = float("nan")
            if len(trajectory_rows) >= 2:
                step_mean = sum(trajectory_steps) / len(trajectory_steps)
                gap_mean = sum(trajectory_gaps) / len(trajectory_gaps)
                step_variance = sum(
                    (value - step_mean) ** 2 for value in trajectory_steps
                )
                if step_variance > 0.0:
                    gap_slope_per_10 = 10.0 * sum(
                        (step - step_mean) * (gap - gap_mean)
                        for step, gap in zip(trajectory_steps, trajectory_gaps)
                    ) / step_variance
            diagnostics: dict[str, Any] = {
                **frozen_metrics,
                **{
                    f"staleness/{key}": float(value)
                    for key, value in raw_probe.items()
                },
                "staleness/epoch": float(epoch),
                "staleness/reward_round_id": float(
                    adaptive_state.reward_round_id
                ),
                "staleness/decision": "raw_monitor_only",
                "staleness/trigger_recalibration": 0.0,
                "staleness/monitor_mode_raw_only": 1.0,
                "staleness/log_only": 1.0,
                "audit/raw_auc": float(raw_probe["raw_auc"]),
                "audit/raw_auc_gap": float(raw_probe["raw_auc_gap"]),
                "audit/saturated": float(raw_probe["raw_audit_saturated"]),
                "audit/training_ready": float(
                    raw_probe["raw_audit_training_ready"]
                ),
                "audit/epoch": float(epoch),
                "audit/policy_step": float(global_step),
                "audit/trajectory_points": float(len(trajectory_rows)),
                "audit/raw_auc_gap_change_from_step0": float(
                    trajectory_gaps[-1] - trajectory_gaps[0]
                ),
                "audit/raw_auc_gap_slope_per_10_steps": float(
                    gap_slope_per_10
                ),
            }
            gradient_snapshot = (
                int(global_step),
                int(adaptive_state.reward_round_id),
            )
            if (
                gradient_conflict_cfg.enabled
                and global_step > 0
                and global_step % gradient_conflict_cfg.every_n_steps == 0
                and gradient_snapshot != last_gradient_conflict_snapshot
            ):
                from RL.DGPO_neutrino.gradient_conflict import (
                    phase_metrics,
                    probe_gradients,
                )
                from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
                    peft_bank_factory,
                )

                overrides = {
                    key: adaptive_cfg.audit_fit[key]
                    for key in (
                        "head_dropout", "topology_dropout", "decoder_hidden_dim",
                        "decoder_layers", "decoder_heads", "periodic_pair_features",
                        "topology_fourier_embedding", "topology_direct_logit",
                        "topology_context_residual_scale", "topology_conditioning", "topology_pair_token", "relation_token_count",
                        "visible_pair_rest_frame", "topology_max_harmonic",
                        "topology_include_theta_pair", "topology_theta_fourier", "topology_hidden_dim",
                        "topology_embedding_dim", "topology_fusion_hidden_dim",
                        "train_layernorm", "train_encoder",
                        "train_grouped_sequential_embedding",
                        "train_invisible_projector", "train_backbone",
                        "train_angular_conditioning",
                        "train_last_pet_block",
                        "asymmetric_attention",
                    )
                    if adaptive_cfg.audit_fit.get(key) is not None
                }
                gradient_metrics = probe_gradients(
                    cfg=gradient_conflict_cfg,
                    model=model,
                    core=_unwrap_core_evenet(model),
                    reference=round_ref_model,
                    pool=score_pool,
                    monitor_factory=peft_bank_factory(
                        omnifold_source.model_builder,
                        score_pool.packing_spec,
                        classifier_overrides=overrides,
                    ),
                    monitor_cache=current_raw_monitor_cache or {},
                    reward_compute=lambda candidates, batch: reward_agg.compute(
                        candidates, batch
                    )[0],
                    generate=generate_neutrino_candidates,
                    evaluate=policy_evaluation_step,
                    sampler=sampler,
                    device=device,
                    dtype=dtype,
                    rank=rank,
                    world_size=world_size,
                    K=K,
                    beta=beta,
                    num_ddim_steps=num_ddim,
                    rollout_parallel_chains=rollout_parallel_chains,
                    num_train_timesteps=num_train_timesteps,
                    t_min=policy_eval_t_min_cfg,
                    t_max=policy_eval_t_max_cfg,
                    advantage_estimator=advantage_estimator,
                    adv_clip_max=adv_clip_max_cfg,
                    trust_coefficient=reference_trust_coefficient,
                    global_step=global_step,
                    reward_round_id=int(adaptive_state.reward_round_id),
                    raw_auc=raw_probe["raw_auc"],
                    raw_ready=bool(
                        raw_probe["raw_audit_training_ready"]
                        if adaptive_cfg.audit_fit.get("training_readiness")
                        is not None
                        else raw_probe["raw_audit_saturated"]
                    ),
                )
                gradient_metrics = phase_metrics(gradient_metrics, "periodic")
                gradient_metrics.update(
                    {
                        "gradient_conflict/judge_fit_global_step": int(global_step),
                        "gradient_conflict/judge_reused_from_install": 0.0,
                        "gradient_conflict/raw_auc_is_current": 1.0,
                        "gradient_conflict/warmup_completed_updates": (
                            adaptive_state.policy_warmup_completed_updates
                        ),
                    }
                )
                last_gradient_conflict_snapshot = gradient_snapshot
                diagnostics.update(gradient_metrics)
                direction_aliases = {
                    "installed/norm": "omnifold/norm",
                    "installed/split_cosine": "omnifold/split_cosine",
                    "installed/reliable": "omnifold/reliable",
                    "fresh/norm": "staleness/norm",
                    "fresh/split_cosine": "staleness/split_cosine",
                    "fresh/reliable": "staleness/reliable",
                    "installed_vs_fresh/cosine": "omnifold_staleness/cosine",
                    "installed_vs_fresh/cross_dot": "omnifold_staleness/cross_dot",
                    "installed_vs_fresh/cross_dot_lcb": "omnifold_staleness/cross_dot_lcb",
                    "installed_vs_fresh/cross_dot_ucb": "omnifold_staleness/cross_dot_ucb",
                    "installed_vs_fresh/conclusive": "omnifold_staleness/conclusive",
                    "installed_vs_fresh/conflict": "omnifold_staleness/conflict",
                    "installed_vs_fresh/alignment": "omnifold_staleness/alignment",
                }
                for alias, source in direction_aliases.items():
                    source_key = "gradient_conflict/" + source
                    if source_key in gradient_metrics:
                        diagnostics["gradient_direction/" + alias] = (
                            gradient_metrics[source_key]
                        )
                diagnostics["gradient_direction/reference_trust_active"] = 0.0
            if pending_resume_trust_diagnostics:
                diagnostics.update(pending_resume_trust_diagnostics)
                pending_resume_trust_diagnostics = {}
            _assert_current_reward_reference_pairing("adaptive_cycle_raw_only")
            if is_rank0:
                _log.info(
                    "[DGPO/omnifold] epoch=%s raw-only monitor AUC=%.6f "
                    "gap=%.6f round=%s",
                    epoch,
                    float(raw_probe["raw_auc"]),
                    float(raw_probe["raw_auc_gap"]),
                    adaptive_state.reward_round_id,
                )
            return diagnostics
        skip_incumbent_probe = should_skip_incumbent_probe(
            cfg=adaptive_cfg,
            force_refit=force_refit,
        )
        probe = (
            fit_raw_policy_audit(
                pool=score_pool,
                model_builder=omnifold_source.model_builder,
                cfg=adaptive_cfg,
                device=device,
                seed=(
                    adaptive_cfg.probe_seed
                    if fixed_audit_panel
                    else adaptive_cfg.probe_seed + max(int(epoch), 0)
                ),
                warm_start_cache=(
                    adaptive_state.raw_monitor_state
                    if adaptive_cfg.raw_monitor_warm_start
                    else None
                ),
                progress_callback=lambda row: _log_omnifold_fit_progress(
                    "raw_staleness_audit",
                    row,
                    epoch_value=int(epoch),
                ),
            )
            if skip_incumbent_probe
            else probe_installed_reward(
                omnifold_source,
                score_pool,
                cfg=adaptive_cfg,
                device=device,
                seed=(
                    adaptive_cfg.probe_seed
                    if fixed_audit_panel
                    else adaptive_cfg.probe_seed + max(int(epoch), 0)
                ),
                early_stop_auc_gap=(
                    float(adaptive_state.trigger_threshold)
                    if adaptive_state.calibrated and not baseline_only
                    else None
                ),
                raw_warm_start_cache=(
                    adaptive_state.raw_monitor_state
                    if adaptive_cfg.raw_monitor_warm_start
                    else None
                ),
                progress_callback=lambda phase, row: _log_omnifold_fit_progress(
                    phase,
                    row,
                    epoch_value=int(epoch),
                ),
            )
        )
        signed_probe_diagnostics: dict[str, Any] = {}
        if (
            adaptive_cfg.trust_signed_direction_probe_enabled
            and not baseline_only
            and not force_refit
        ):
            signed_probe_diagnostics = _run_signed_direction_probe(
                epoch=int(epoch),
                probe_panel_seed=int(probe_panel_seed),
                current_probe=probe,
            )
        signed_recovery_diagnostics = (
            evaluate_signed_direction_recovery(
                adaptive_state,
                cfg=adaptive_cfg,
                signed_probe=signed_probe_diagnostics,
            )
            if adaptive_cfg.trust_signed_direction_recovery_enabled
            else {}
        )
        signed_recovery_triggered = bool(
            float(
                signed_recovery_diagnostics.get(
                    "reference_trust/signed_probe/recovery_triggered",
                    0.0,
                )
            )
            >= 0.5
        )
        signed_recovery_stop_requested = bool(
            float(
                signed_recovery_diagnostics.get(
                    "reference_trust/signed_probe/recovery_stop_requested",
                    0.0,
                )
            )
            >= 0.5
        )
        trajectory_diagnostics: dict[str, float] = {}
        if (
            adaptive_cfg.trust_trajectory_search_enabled
            and not baseline_only
            and not force_refit
            and not signed_recovery_triggered
        ):
            expected_checkpoint = ""
            if save_dir_raw:
                expected_checkpoint = str(
                    Path(str(save_dir_raw)).expanduser().resolve()
                    / dgpo_snapshot_checkpoint_name(
                        last_completed_epoch=int(epoch),
                        dgpo_next_epoch=int(epoch) + 1,
                        global_step=int(global_step),
                    )
                )
            trajectory_diagnostics = update_trust_trajectory_candidate(
                adaptive_state,
                cfg=adaptive_cfg,
                raw_auc_gap=float(probe.get("raw_auc_gap", float("nan"))),
                raw_auc_se=float(
                    probe.get("raw_auc_null_se_approx", float("nan"))
                ),
                epoch=int(epoch),
                global_step=int(global_step),
                checkpoint_path=expected_checkpoint,
                audit_saturated=bool(
                    float(probe.get("raw_audit_saturated", 0.0)) >= 0.5
                ),
            )
        empirical_radius_diagnostics: dict[str, float] = {}
        if (
            not baseline_only
            and not force_refit
            and adaptive_state.calibrated
            and not signed_recovery_triggered
            and adaptive_cfg.trust_empirical_radius_enabled
        ):
            empirical_radius_diagnostics = (
                update_empirical_trust_radius_from_audit(
                    adaptive_state,
                    probe,
                    cfg=adaptive_cfg,
                )
            )
        trigger_refit = bool(force_refit)
        if force_refit and force_reason:
            diagnostics_force_reason = str(force_reason)
        else:
            diagnostics_force_reason = ""
        if force_refit and skip_incumbent_probe:
            diagnostics = {
                **{
                    f"staleness/{key}": float(value)
                    for key, value in probe.items()
                },
                "staleness/decision": "forced_refit",
                "staleness/trigger_recalibration": 1.0,
                "staleness/reward_round_id": float(
                    adaptive_state.reward_round_id
                ),
                "omnifold/incumbent_probe_skipped": 1.0,
            }
        elif baseline_only or not adaptive_state.calibrated:
            if (
                adaptive_cfg.require_audit_saturation
                and float(probe.get("audit_saturated", 0.0)) < 0.5
            ):
                raise RuntimeError(
                    "initial adaptive OmniFold audit did not saturate; refusing "
                    "to freeze an uncertified staleness threshold"
                )
            adaptive_state.install(
                baseline_auc_gap=baseline_probe_auc_gap(
                    probe,
                    cfg=adaptive_cfg,
                ),
                cfg=adaptive_cfg,
                epoch=int(epoch),
                round_id=int(omnifold_source.reward_round_id),
            )
            adaptive_state.audit_protocol_signature = current_audit_signature
            adaptive_state.last_decision = "installed_baseline"
            adaptive_state.probe_history.append(
                {
                    "epoch": float(epoch),
                    "reward_round_id": float(adaptive_state.reward_round_id),
                    "decision_recalibrate": 0.0,
                    **{key: float(value) for key, value in probe.items()},
                }
            )
            diagnostics: dict[str, Any] = {
                **{f"staleness/{key}": float(value) for key, value in probe.items()},
                "staleness/decision": "installed_baseline",
                "staleness/trigger_recalibration": 0.0,
                "staleness/trigger_threshold": float(
                    adaptive_state.trigger_threshold
                ),
                "staleness/baseline_auc_gap": float(
                    adaptive_state.baseline_auc_gap
                ),
                "staleness/previous_audit_auc_gap": float(
                    adaptive_state.previous_audit_auc_gap
                ),
                "staleness/next_trigger_threshold": float(
                    adaptive_state.trigger_threshold
                ),
                "staleness/reward_round_id": float(
                    adaptive_state.reward_round_id
                ),
                **trajectory_diagnostics,
            }
        else:
            controller_trigger, diagnostics = update_controller(
                adaptive_state,
                probe,
                cfg=adaptive_cfg,
                epoch=int(epoch),
            )
            diagnostics.update(empirical_radius_diagnostics)
            diagnostics.update(trajectory_diagnostics)
            trigger_refit = bool(trigger_refit or controller_trigger)
            if adaptive_cfg.trust_boundary_enabled:
                trust_exhausted, trust_exhaustion_diagnostics = (
                    trust_region_exhausted(
                        adaptive_state,
                        cfg=adaptive_cfg,
                    )
                )
            else:
                trust_exhausted, trust_exhaustion_diagnostics = False, {}
            diagnostics.update(trust_exhaustion_diagnostics)
            age_refit_due, reward_age = reward_refit_due_to_age(
                adaptive_state,
                epoch=int(epoch),
                max_reward_age_epochs=adaptive_cfg.max_reward_age_epochs,
            )
            cooldown_active = bool(
                float(diagnostics.get("staleness/cooldown_active", 0.0)) >= 0.5
            )
            age_trigger = bool(
                age_refit_due
                and not adaptive_cfg.log_only
                and not cooldown_active
            )
            exhaustion_trigger = bool(
                trust_exhausted
                and not adaptive_cfg.log_only
                and not cooldown_active
            )
            diagnostics.update(
                {
                    "staleness/reward_age_epochs": float(reward_age),
                    "staleness/max_reward_age_epochs": float(
                        adaptive_cfg.max_reward_age_epochs
                        if adaptive_cfg.max_reward_age_epochs is not None
                        else float("nan")
                    ),
                    "staleness/age_refit_due": float(age_refit_due),
                    "staleness/age_trigger_recalibration": float(age_trigger),
                    "staleness/trust_exhausted": float(trust_exhausted),
                    "staleness/trust_exhaustion_trigger_recalibration": float(
                        exhaustion_trigger
                    ),
                }
            )
            if exhaustion_trigger:
                trigger_refit = True
                adaptive_state.last_decision = "trust_exhausted_recalibrate"
                diagnostics["staleness/decision"] = adaptive_state.last_decision
                diagnostics["staleness/trigger_recalibration"] = 1.0
                # Exhaustion takes priority when the weighted-AUC controller
                # crosses on the same audit.  It is the condition that enables
                # best-on-trajectory selection instead of accepting the endpoint.
                diagnostics["staleness/trigger_reason"] = "trust_exhausted"
            elif age_trigger and not trigger_refit:
                trigger_refit = True
                adaptive_state.last_decision = "max_reward_age_recalibrate"
                diagnostics["staleness/decision"] = (
                    adaptive_state.last_decision
                )
                diagnostics["staleness/trigger_recalibration"] = 1.0
                diagnostics["staleness/trigger_reason"] = "max_reward_age"
            elif controller_trigger:
                diagnostics["staleness/trigger_reason"] = "auc_gap"
            elif trust_exhausted and cooldown_active:
                diagnostics["staleness/trigger_reason"] = (
                    "trust_exhausted_cooldown"
                )
            elif trust_exhausted and adaptive_cfg.log_only:
                diagnostics["staleness/trigger_reason"] = (
                    "trust_exhausted_log_only"
                )
            elif age_refit_due and cooldown_active:
                diagnostics["staleness/trigger_reason"] = (
                    "max_reward_age_cooldown"
                )
            elif age_refit_due and adaptive_cfg.log_only:
                diagnostics["staleness/trigger_reason"] = "max_reward_age_log_only"
        if adaptive_cfg.trust_extragradient_enabled:
            diagnostics["reference_trust/extragradient/enabled"] = 1.0
        diagnostics.update(signed_probe_diagnostics)
        diagnostics.update(signed_recovery_diagnostics)
        if signed_recovery_triggered:
            # A statistically saturated reverse-only direction is evidence
            # against this *local* surrogate update, not evidence that the
            # global DGPO sign should be flipped. Return to the reward's exact
            # denominator, discard momentum, and train a fresh classifier on a
            # newly generated fit population. The fixed audit panel remains
            # unchanged, preserving the paired raw-AUC comparison.
            adaptive_state.trust_failed_direction_streak = int(
                signed_recovery_diagnostics[
                    "reference_trust/signed_probe/recovery_attempt"
                ]
            )
            adaptive_state.trust_round_rollbacks += 1
            adaptive_state.trust_round_stop_requested = bool(
                signed_recovery_stop_requested
            )
            adaptive_state.probe_exceedance_streak = 0
            adaptive_state.trust_distance_window.clear()
            adaptive_state.trust_step_acceptance_window.clear()
            adaptive_state.trust_step_scale_window.clear()
            adaptive_state.reset_trust_trajectory(
                reset_failed_directions=False
            )
            cleared_optimizer_states, ema_restored = (
                _restore_live_policy_to_round_reference()
            )
            recovery_metrics = {
                "reference_trust/signed_probe/recovery_policy_restored": 1.0,
                "reference_trust/signed_probe/recovery_optimizer_states_cleared": float(
                    cleared_optimizer_states
                ),
                "reference_trust/signed_probe/recovery_ema_restored": float(
                    ema_restored
                ),
                "reference_trust/round_acceptance/rollback_required": 1.0,
                "reference_trust/round_acceptance/rollbacks": float(
                    adaptive_state.trust_round_rollbacks
                ),
                "reference_trust/round_acceptance/failed_direction_streak": float(
                    adaptive_state.trust_failed_direction_streak
                ),
                "reference_trust/round_acceptance/stop_requested": float(
                    signed_recovery_stop_requested
                ),
            }
            diagnostics.update(recovery_metrics)
            if signed_recovery_stop_requested:
                trigger_refit = False
                adaptive_state.last_decision = (
                    "signed_reverse_only_patience_stop"
                )
                diagnostics.update(
                    {
                        "staleness/decision": adaptive_state.last_decision,
                        "staleness/trigger_recalibration": 0.0,
                        "staleness/trigger_reason": (
                            "signed_reverse_only_patience"
                        ),
                    }
                )
            else:
                trigger_refit = True
                adaptive_state.last_decision = (
                    "signed_reverse_only_fresh_reward"
                )
                diagnostics.update(
                    {
                        "staleness/decision": adaptive_state.last_decision,
                        "staleness/trigger_recalibration": 1.0,
                        "staleness/trigger_reason": "signed_reverse_only",
                    }
                )
                # The old score pool was generated at the rejected endpoint.
                # Recreate the same fixed identities/noise at the restored
                # incumbent before fitting or auditing the replacement reward.
                del score_pool
                score_pool = _materialize_adaptive_omnifold_pool(
                    omnifold_val_shard,
                    omnifold_val_loader_cfg,
                    model=model,
                    sampler=sampler,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    quota_events=adaptive_cfg.probe_max_events,
                    num_ddim_steps=num_ddim_val,
                    seed=int(probe_panel_seed),
                    include_pairwise_context=(
                        adaptive_cfg.periodic_pair_features_enabled
                    ),
                    include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                )
            if is_rank0:
                _log.warning(
                    "[DGPO/trust/signed] reverse-only direction rejected; "
                    "restored round_ref=%s, cleared=%s optimizer states, "
                    "recovery=%s/%s stop=%s",
                    adaptive_state.reward_round_id,
                    cleared_optimizer_states,
                    adaptive_state.trust_failed_direction_streak,
                    adaptive_cfg.trust_failed_direction_patience,
                    signed_recovery_stop_requested,
                )
        round_budget_exhausted = reward_round_budget_exhausted(
            adaptive_state,
            max_reward_rounds=adaptive_cfg.max_reward_rounds,
        )
        diagnostics.update(
            {
                "staleness/max_reward_rounds": float(
                    adaptive_cfg.max_reward_rounds
                    if adaptive_cfg.max_reward_rounds is not None
                    else 0
                ),
                "staleness/reward_round_budget_exhausted": float(
                    round_budget_exhausted
                ),
            }
        )
        if round_budget_exhausted and trigger_refit:
            trigger_refit = False
            adaptive_state.last_decision = "reward_round_budget_complete"
            diagnostics.update(
                {
                    "staleness/decision": adaptive_state.last_decision,
                    "staleness/trigger_recalibration": 0.0,
                    "staleness/trigger_reason": "reward_round_budget",
                }
            )
        if trigger_refit:
            if diagnostics_force_reason:
                diagnostics["staleness/trigger_reason"] = diagnostics_force_reason
                diagnostics["staleness/trigger_recalibration"] = 1.0
            extragradient_requested = bool(
                adaptive_cfg.trust_extragradient_enabled
                and not baseline_only
                and not force_refit
                and not signed_recovery_triggered
                and diagnostics.get("staleness/trigger_reason")
                == "trust_exhausted"
            )
            if extragradient_requested and extragradient_batch is None:
                raise RuntimeError(
                    "look-ahead extragradient needs the most recent DGPO batch "
                    "for its synchronous corrector step"
                )
            extragradient_anchor_policy: dict[str, Tensor] | None = None
            extragradient_anchor_params: dict[str, Tensor] | None = None
            extragradient_anchor_reward: dict[str, Any] | None = None
            extragradient_anchor_state: dict[str, Any] | None = None
            extragradient_anchor_probe: _ReferenceTrustProbe | None = None
            extragradient_anchor_probe_cache: dict[str, Any] | None = None
            extragradient_anchor_delta = float("nan")
            if extragradient_requested:
                cached_anchor_probe = reference_trust_probe_cache.get("probe")
                if not isinstance(cached_anchor_probe, _ReferenceTrustProbe):
                    raise RuntimeError(
                        "look-ahead extragradient requires the incumbent's "
                        "fixed functional trust probe"
                    )
                extragradient_anchor_policy = _snapshot_policy_state_dict(
                    round_ref_model
                )
                extragradient_anchor_reward = omnifold_source.stack_payload()
                extragradient_anchor_state = adaptive_state.to_dict()
                extragradient_anchor_probe = cached_anchor_probe
                extragradient_anchor_probe_cache = dict(
                    reference_trust_probe_cache
                )
                extragradient_anchor_delta = float(
                    adaptive_state.trust_current_delta
                )
            trajectory_rewound = False
            selected_checkpoint = ""
            if (
                adaptive_cfg.trust_trajectory_search_enabled
                and diagnostics.get("staleness/trigger_reason")
                == "trust_exhausted"
                and adaptive_state.trust_trajectory_best_checkpoint
            ):
                selected_checkpoint = str(
                    adaptive_state.trust_trajectory_best_checkpoint
                )
                best_is_current = bool(
                    int(adaptive_state.trust_trajectory_best_epoch) == int(epoch)
                    and int(adaptive_state.trust_trajectory_best_global_step)
                    == int(global_step)
                )
                if not best_is_current:
                    loaded_tensors = _restore_policy_from_dgpo_checkpoint(
                        model,
                        selected_checkpoint,
                    )
                    trajectory_rewound = True
                    policy_core = unwrap_for_state_dict(model)
                    cleared_optimizer_states = len(optimizer.state)
                    optimizer.state.clear()
                    optimizer.zero_grad(set_to_none=True)
                    if ema_save is not None:
                        ema_save.update(policy_core, decay_=0.0)
                    if ema_rollout is not None:
                        ema_rollout.update(policy_core, decay_=0.0)
                    # The cheap endpoint probe was generated before rewinding.
                    # Rebuild the candidate population on the identical fixed
                    # panel so every later classifier sees the selected policy.
                    score_pool = _materialize_adaptive_omnifold_pool(
                        omnifold_val_shard,
                        omnifold_val_loader_cfg,
                        model=model,
                        sampler=sampler,
                        device=device,
                        world_size=world_size,
                        rank=rank,
                        quota_events=adaptive_cfg.probe_max_events,
                        num_ddim_steps=num_ddim_val,
                        seed=probe_panel_seed,
                        include_pairwise_context=(
                            adaptive_cfg.periodic_pair_features_enabled
                        ),
                        include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                    )
                    diagnostics.update(
                        {
                            "reference_trust/trajectory/rewound": 1.0,
                            "reference_trust/trajectory/loaded_tensors": float(
                                loaded_tensors
                            ),
                            "reference_trust/trajectory/optimizer_states_cleared": float(
                                cleared_optimizer_states
                            ),
                            "reference_trust/trajectory/ema_restored": float(
                                ema_save is not None or ema_rollout is not None
                            ),
                        }
                    )
                else:
                    diagnostics["reference_trust/trajectory/rewound"] = 0.0
                diagnostics.update(
                    {
                        "reference_trust/trajectory/selected": 1.0,
                        "reference_trust/trajectory/selected_epoch": float(
                            adaptive_state.trust_trajectory_best_epoch
                        ),
                        "reference_trust/trajectory/selected_global_step": float(
                            adaptive_state.trust_trajectory_best_global_step
                        ),
                        "reference_trust/trajectory/selected_raw_auc_gap": float(
                            adaptive_state.trust_trajectory_best_raw_auc_gap
                        ),
                    }
                )
                if is_rank0:
                    _log.info(
                        "[DGPO/trust] trajectory search selected epoch=%s "
                        "step=%s raw_gap=%.6g rewound=%s checkpoint=%s",
                        adaptive_state.trust_trajectory_best_epoch,
                        adaptive_state.trust_trajectory_best_global_step,
                        adaptive_state.trust_trajectory_best_raw_auc_gap,
                        trajectory_rewound,
                        selected_checkpoint,
                    )
            if extragradient_requested:
                (
                    policy_core,
                    extragradient_anchor_params,
                    selected_trainable_params,
                ) = _snapshot_signed_trainable_direction(
                    model,
                    round_ref_model,
                )
                selected_direction_rms = _trainable_direction_rms(
                    extragradient_anchor_params,
                    selected_trainable_params,
                )
                lookahead_scale = float(
                    adaptive_cfg.trust_extragradient_lookahead_scale
                )
                _assign_signed_trainable_direction_(
                    policy_core,
                    extragradient_anchor_params,
                    selected_trainable_params,
                    lookahead_scale,
                )
                lookahead_direction_rms = _trainable_direction_rms(
                    extragradient_anchor_params,
                    snapshot_params(policy_core),
                )
                # The endpoint probe was generated before interpolation.  The
                # transient classifier must see the actual virtual look-ahead
                # distribution on the identical event/noise panel.
                del score_pool
                score_pool = _materialize_adaptive_omnifold_pool(
                    omnifold_val_shard,
                    omnifold_val_loader_cfg,
                    model=model,
                    sampler=sampler,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    quota_events=adaptive_cfg.probe_max_events,
                    num_ddim_steps=num_ddim_val,
                    seed=int(probe_panel_seed),
                    include_pairwise_context=(
                        adaptive_cfg.periodic_pair_features_enabled
                    ),
                    include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                )
                diagnostics.update(
                    {
                        "reference_trust/extragradient/enabled": 1.0,
                        "reference_trust/extragradient/triggered": 1.0,
                        "reference_trust/extragradient/lookahead_scale": (
                            lookahead_scale
                        ),
                        "reference_trust/extragradient/selected_direction_rms": (
                            selected_direction_rms
                        ),
                        "reference_trust/extragradient/lookahead_direction_rms": (
                            lookahead_direction_rms
                        ),
                    }
                )
                if is_rank0:
                    _log.info(
                        "[DGPO/trust/extragradient] virtual look-ahead "
                        "scale=%.6g selected_rms=%.6g lookahead_rms=%.6g",
                        lookahead_scale,
                        selected_direction_rms,
                        lookahead_direction_rms,
                    )
            # No optimizer step occurs between this snapshot and any refit
            # population. They therefore share this exact policy denominator.
            policy_snapshot = _snapshot_policy_state_dict(model)
            if signed_recovery_triggered:
                # Change only the reward-fit population. The fixed validation
                # panel and classifier seeds stay paired across comparisons.
                fit_pool_seed = (
                    int(adaptive_cfg.seed)
                    + 1_000_003
                    * (int(adaptive_state.recalibration_count) + 1)
                    + 10_007
                    * int(adaptive_state.trust_failed_direction_streak)
                )
            elif adaptive_cfg.trust_trajectory_search_enabled:
                fit_pool_seed = int(adaptive_cfg.seed)
            else:
                fit_pool_seed = int(adaptive_cfg.seed) + int(
                    adaptive_state.recalibration_count
                )
            if signed_recovery_triggered:
                diagnostics[
                    "reference_trust/signed_probe/recovery_fit_pool_seed"
                ] = float(fit_pool_seed)
            if is_rank0:
                if force_refit:
                    refit_label = "one-shot resume adapter refit"
                elif signed_recovery_triggered:
                    refit_label = "signed reverse-only fresh-reward recovery"
                elif (
                    diagnostics.get("staleness/trigger_reason")
                    == "max_reward_age"
                ):
                    refit_label = "reward maximum age reached"
                else:
                    refit_label = "adapter staleness triggered"
                _log.info(
                    "[DGPO/omnifold] %s; fitting current-policy K=1 pool "
                    "(cap=%s)",
                    refit_label,
                    adaptive_cfg.pool_events,
                )
            fit_pool = _materialize_adaptive_omnifold_pool(
                omnifold_train_shard,
                omnifold_train_loader_cfg,
                model=model,
                sampler=sampler,
                device=device,
                world_size=world_size,
                rank=rank,
                quota_events=(
                    None if fixed_omnifold_pool else adaptive_cfg.pool_events
                ),
                num_ddim_steps=num_ddim,
                seed=fit_pool_seed,
                include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
                include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
            )
            refit_score_pool = score_pool
            if adaptive_cfg.refit_score_events != adaptive_cfg.probe_max_events:
                if is_rank0:
                    _log.info(
                        "[DGPO/omnifold] materializing full held-out residual-"
                        "validation pool (cap=%s; routine probe cap=%s)",
                        adaptive_cfg.refit_score_events,
                        adaptive_cfg.probe_max_events,
                    )
                refit_score_pool = _materialize_adaptive_omnifold_pool(
                    omnifold_val_shard,
                    omnifold_val_loader_cfg,
                    model=model,
                    sampler=sampler,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    quota_events=adaptive_cfg.refit_score_events,
                    num_ddim_steps=num_ddim_val,
                    seed=(probe_panel_seed + 1_000_003),
                    include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
                    include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                )
            refit_diagnostics = run_adaptive_refit(
                global_step=int(global_step),
                state=adaptive_state,
                cfg=adaptive_cfg,
                reward_source=omnifold_source,
                round_ref_model=round_ref_model,
                policy_snapshot_state_dict=policy_snapshot,
                fit_pool=fit_pool,
                score_pool=refit_score_pool,
                # Establish the new reward's controller baseline on the same
                # cheap protocol used by subsequent staleness probes.
                baseline_pool=score_pool,
                epoch=int(epoch),
                device=device,
                world_size=world_size,
                # Resume protocol changes and signed recovery both establish a
                # reward estimator at a fixed policy. They bypass only the
                # cross-round policy-improvement comparison; classifier
                # saturation, residual closure, acceptance, and topology gates
                # remain mandatory.
                enforce_round_acceptance=not (
                    force_refit
                    or signed_recovery_triggered
                    or extragradient_requested
                ),
                progress_callback=lambda phase, row: _log_omnifold_fit_progress(
                    phase,
                    row,
                    epoch_value=int(epoch),
                ),
            )
            diagnostics.update(refit_diagnostics)
            refit_accepted = (
                float(refit_diagnostics.get("omnifold/accepted", 0.0)) >= 0.5
            )
            if (
                not refit_accepted
                and adaptive_cfg.scheduled_refit_fail_closed
                and not force_refit
            ):
                raise RuntimeError(
                    "scheduled fresh reward refit failed; refusing to continue "
                    "the fixed-round experiment with a stale reward: "
                    + str(
                        refit_diagnostics.get(
                            "omnifold/accept_reason", "unspecified classifier gate"
                        )
                    )
                )
            if extragradient_requested:
                diagnostics.update(
                    {
                        "reference_trust/extragradient/lookahead_reward_installed": float(
                            refit_accepted
                        ),
                        "reference_trust/extragradient/lookahead_reward_round_id": float(
                            refit_diagnostics.get(
                                "omnifold/reward_round_id",
                                adaptive_state.reward_round_id,
                            )
                        ),
                        "reference_trust/extragradient/lookahead_raw_auc_gap": float(
                            refit_diagnostics.get(
                                "reference_trust/round_acceptance/current_auc_gap",
                                float("nan"),
                            )
                        ),
                        "reference_trust/extragradient/lookahead_acceptance_auc": float(
                            refit_diagnostics.get(
                                "omnifold/candidate/audit_validation_auc",
                                float("nan"),
                            )
                        ),
                        "reference_trust/extragradient/lookahead_topology_auc": float(
                            refit_diagnostics.get(
                                "omnifold/candidate/topology_audit_auc",
                                float("nan"),
                            )
                        ),
                    }
                )
            if extragradient_requested and not refit_accepted:
                # The first refit is only a transient look-ahead estimator.
                # A failed classifier gate must not leave its virtual policy
                # (or any refit bookkeeping) installed as live state.
                assert extragradient_anchor_policy is not None
                assert extragradient_anchor_reward is not None
                assert extragradient_anchor_state is not None
                assert extragradient_anchor_probe_cache is not None
                policy_core = unwrap_for_state_dict(model)
                policy_core.load_state_dict(
                    extragradient_anchor_policy,
                    strict=True,
                )
                round_ref_model.load_state_dict(
                    extragradient_anchor_policy,
                    strict=True,
                )
                freeze_reference_model(round_ref_model)
                omnifold_source.load_stack_payload(
                    extragradient_anchor_reward
                )
                restored_state = type(adaptive_state).from_dict(
                    extragradient_anchor_state
                )
                adaptive_state.__dict__.clear()
                adaptive_state.__dict__.update(restored_state.__dict__)
                optimizer.state.clear()
                optimizer.zero_grad(set_to_none=True)
                if ema_save is not None:
                    ema_save.update(policy_core, decay_=0.0)
                if ema_rollout is not None:
                    ema_rollout.update(policy_core, decay_=0.0)
                reference_trust_probe_cache.clear()
                reference_trust_probe_cache.update(
                    extragradient_anchor_probe_cache
                )
                diagnostics.update(
                    record_extragradient_rejection(
                        adaptive_state,
                        cfg=adaptive_cfg,
                        reason="lookahead_classifier_gate_failed",
                    )
                )
                diagnostics.update(
                    {
                        "omnifold/accepted": 0.0,
                        "omnifold/reward_round_id": float(
                            adaptive_state.reward_round_id
                        ),
                        "reference_trust/extragradient/final_reward_installed": 0.0,
                        "staleness/decision": adaptive_state.last_decision,
                    }
                )
                reward_checkpoint_metadata = (
                    omnifold_source.checkpoint_metadata()
                )
                _assert_current_reward_reference_pairing(
                    "extragradient_lookahead_rejection"
                )
                if is_rank0:
                    _log.warning(
                        "[DGPO/trust/extragradient] rejected look-ahead "
                        "classifier; restored incumbent round=%s",
                        adaptive_state.reward_round_id,
                    )
                return diagnostics
            if extragradient_requested and refit_accepted:
                assert extragradient_batch is not None
                assert extragradient_anchor_policy is not None
                assert extragradient_anchor_params is not None
                assert extragradient_anchor_reward is not None
                assert extragradient_anchor_state is not None
                assert extragradient_anchor_probe is not None
                assert extragradient_anchor_probe_cache is not None

                lookahead_round_id = int(adaptive_state.reward_round_id)
                policy_core = unwrap_for_state_dict(model)
                scheduler_state = copy.deepcopy(optimizer.scheduler.state_dict())
                cleared_optimizer_states = len(optimizer.state)
                optimizer.state.clear()
                optimizer.zero_grad(set_to_none=True)
                if ema_save is not None:
                    ema_save.update(policy_core, decay_=0.0)
                if ema_rollout is not None:
                    ema_rollout.update(policy_core, decay_=0.0)

                corrector_lr_diagnostics = adaptive_trust_policy_lr_scale(
                    type(adaptive_state).from_dict(
                        extragradient_anchor_state
                    ),
                    cfg=adaptive_cfg,
                )
                corrector_global_step = int(global_step)
                corrector_metrics = train_step(
                    model,
                    round_ref_model,
                    ema_rollout,
                    ema_save,
                    extragradient_batch,
                    optimizer,
                    sampler,
                    reward_agg,
                    beta=beta,
                    K=K,
                    advantage_estimator=advantage_estimator,
                    num_ddim_steps=num_ddim,
                    rollout_parallel_chains=rollout_parallel_chains,
                    global_step=corrector_global_step,
                    epoch=epoch,
                    device=device,
                    dtype=dtype,
                    reference_trust_coefficient=reference_trust_coefficient,
                    reference_trust_objective=reference_trust_objective,
                    reference_trust_vp_path_kl_diagnostic=(
                        reference_trust_vp_path_kl_diagnostic
                    ),
                    reference_trust_vp_logsnr_min=(
                        reference_trust_vp_logsnr_min
                    ),
                    reference_trust_vp_logsnr_max=(
                        reference_trust_vp_logsnr_max
                    ),
                    # The corrector evaluates F at the look-ahead pair and is
                    # applied from the original incumbent. Its trust audit is
                    # performed below on the incumbent's fixed probe.
                    reference_trust_max_ratio=None,
                    reference_trust_distance=adaptive_cfg.trust_distance,
                    reference_trust_warning_fraction=(
                        adaptive_cfg.trust_warning_fraction
                    ),
                    reference_trust_backtrack_factor=(
                        adaptive_cfg.trust_backtrack_factor
                    ),
                    reference_trust_max_backtracks=(
                        adaptive_cfg.trust_max_backtracks
                    ),
                    reference_trust_probe_events_per_rank=(
                        adaptive_cfg.trust_probe_events_per_rank
                    ),
                    reference_trust_fixed_probe_per_round=False,
                    reference_trust_interior_fraction=(
                        adaptive_cfg.trust_interior_fraction
                    ),
                    reference_trust_policy_lr_scale=float(
                        corrector_lr_diagnostics[
                            "reference_trust/policy_lr/scale"
                        ]
                    ),
                    reference_trust_reset_adam_first_moment_on_zero_step=False,
                    reference_trust_probe_cache=None,
                    log_reward_dist=False,
                    log_diagnostic_dist=False,
                    collect_train_dist=False,
                    diagnostic_plot_names=set(),
                    diagnostic_plot_every=1,
                    num_train_timesteps=num_train_timesteps,
                    policy_eval_parallel_timesteps=(
                        policy_eval_parallel_timesteps
                    ),
                    policy_eval_event_microbatch_size=(
                        policy_eval_event_microbatch_size
                    ),
                    adv_clip_max=adv_clip_max_cfg,
                    grad_clip_norm=grad_clip_norm_cfg,
                    policy_eval_t_min=policy_eval_t_min_cfg,
                    policy_eval_t_max=policy_eval_t_max_cfg,
                    constraint_state=constraint_state,
                    world_size=world_size,
                    extragradient_optimizer_base_params=(
                        extragradient_anchor_params
                    ),
                )
                global_step += 1
                corrector_step_ran = bool(
                    float(corrector_metrics.get("train/optimizer_step_ran", 0.0))
                    >= 0.5
                )
                corrected_params = snapshot_params(policy_core)
                corrected_direction_rms = _trainable_direction_rms(
                    extragradient_anchor_params,
                    corrected_params,
                )
                corrector_distance = float("nan")
                corrector_scale = 0.0
                corrector_backtracks = 0
                correction_failure_reason = ""
                if corrector_step_ran and corrected_direction_rms > 0.0:
                    corrector_distance, _, _ = _measure_reference_trust_probe(
                        model,
                        extragradient_anchor_probe,
                        world_size=world_size,
                        distance=adaptive_cfg.trust_distance,
                    )
                    correction_limit = (
                        float(adaptive_cfg.trust_interior_fraction)
                        * extragradient_anchor_delta
                    )
                    correction_tolerance = max(
                        1.0e-12,
                        1.0e-6 * extragradient_anchor_delta,
                    )
                    if (
                        math.isfinite(corrector_distance)
                        and corrector_distance
                        <= correction_limit + correction_tolerance
                    ):
                        corrector_scale = 1.0
                    else:
                        corrector_candidate = corrected_params
                        for backtrack_index, scale in enumerate(
                            adaptive_trust_backtracking_scales(
                                1.0,
                                factor=adaptive_cfg.trust_backtrack_factor,
                                max_backtracks=adaptive_cfg.trust_max_backtracks,
                            ),
                            start=1,
                        ):
                            _assign_interpolated_trainable_params_(
                                policy_core,
                                extragradient_anchor_params,
                                corrector_candidate,
                                scale,
                            )
                            trial_distance, _, _ = (
                                _measure_reference_trust_probe(
                                    model,
                                    extragradient_anchor_probe,
                                    world_size=world_size,
                                    distance=adaptive_cfg.trust_distance,
                                )
                            )
                            if (
                                math.isfinite(trial_distance)
                                and trial_distance
                                <= correction_limit + correction_tolerance
                            ):
                                corrector_scale = float(scale)
                                corrector_backtracks = int(backtrack_index)
                                corrector_distance = float(trial_distance)
                                break
                        if corrector_scale == 0.0:
                            correction_failure_reason = (
                                "trust_backtracking_failed"
                            )
                else:
                    correction_failure_reason = "nonfinite_or_zero_corrector"

                diagnostics.update(
                    {
                        "reference_trust/extragradient/corrector_global_step": float(
                            corrector_global_step
                        ),
                        "reference_trust/extragradient/corrector_step_ran": float(
                            corrector_step_ran
                        ),
                        "reference_trust/extragradient/rebased_optimizer_params": float(
                            corrector_metrics.get(
                                "reference_trust/extragradient/rebased_optimizer_params",
                                0.0,
                            )
                        ),
                        "reference_trust/extragradient/corrector_grad_norm_pre_clip": float(
                            corrector_metrics.get(
                                "train/grad/global_norm_pre_clip",
                                float("nan"),
                            )
                        ),
                        "reference_trust/extragradient/corrector_lr_scale": float(
                            corrector_lr_diagnostics[
                                "reference_trust/policy_lr/scale"
                            ]
                        ),
                        "reference_trust/extragradient/corrector_direction_rms": (
                            corrected_direction_rms
                        ),
                        "reference_trust/extragradient/corrector_distance": (
                            corrector_distance
                        ),
                        "reference_trust/extragradient/corrector_scale": (
                            corrector_scale
                        ),
                        "reference_trust/extragradient/corrector_backtracks": float(
                            corrector_backtracks
                        ),
                        "reference_trust/extragradient/optimizer_states_cleared": float(
                            cleared_optimizer_states
                        ),
                        "reference_trust/extragradient/lookahead_round_id": float(
                            lookahead_round_id
                        ),
                        "reference_trust/extragradient/corrector_reward_mean": float(
                            corrector_metrics.get(
                                "reward/monitor/mean",
                                float("nan"),
                            )
                        ),
                    }
                )

                final_refit_diagnostics: dict[str, Any] = {}
                if not correction_failure_reason:
                    if ema_save is not None:
                        ema_save.update(policy_core, decay_=0.0)
                    if ema_rollout is not None:
                        ema_rollout.update(policy_core, decay_=0.0)
                    # The look-ahead reward/reference/state existed only long
                    # enough to evaluate the corrector field. Reinstall the
                    # original certified pair before fitting the corrected
                    # point so a successful block consumes exactly one public
                    # reward round and compares against the original paired
                    # raw-AUC baseline.
                    round_ref_model.load_state_dict(
                        extragradient_anchor_policy,
                        strict=True,
                    )
                    freeze_reference_model(round_ref_model)
                    omnifold_source.load_stack_payload(
                        extragradient_anchor_reward
                    )
                    restored_state = type(adaptive_state).from_dict(
                        extragradient_anchor_state
                    )
                    adaptive_state.__dict__.clear()
                    adaptive_state.__dict__.update(restored_state.__dict__)
                    reference_trust_probe_cache.clear()
                    reference_trust_probe_cache.update(
                        extragradient_anchor_probe_cache
                    )
                    _assert_current_reward_reference_pairing(
                        "extragradient_final_refit_anchor"
                    )
                    corrected_policy_snapshot = _snapshot_policy_state_dict(model)
                    corrected_pool_seed = (
                        int(adaptive_cfg.seed)
                        + 30_000_091
                        + 1009 * int(lookahead_round_id)
                    )
                    corrected_baseline_pool = _materialize_adaptive_omnifold_pool(
                        omnifold_val_shard,
                        omnifold_val_loader_cfg,
                        model=model,
                        sampler=sampler,
                        device=device,
                        world_size=world_size,
                        rank=rank,
                        quota_events=adaptive_cfg.probe_max_events,
                        num_ddim_steps=num_ddim_val,
                        seed=int(probe_panel_seed),
                        include_pairwise_context=(
                            adaptive_cfg.periodic_pair_features_enabled
                        ),
                        include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                    )
                    corrected_fit_pool = _materialize_adaptive_omnifold_pool(
                        omnifold_train_shard,
                        omnifold_train_loader_cfg,
                        model=model,
                        sampler=sampler,
                        device=device,
                        world_size=world_size,
                        rank=rank,
                        quota_events=(
                            None
                            if fixed_omnifold_pool
                            else adaptive_cfg.pool_events
                        ),
                        num_ddim_steps=num_ddim,
                        seed=corrected_pool_seed,
                        include_pairwise_context=(
                            adaptive_cfg.periodic_pair_features_enabled
                        ),
                        include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                    )
                    corrected_score_pool = corrected_baseline_pool
                    if (
                        adaptive_cfg.refit_score_events
                        != adaptive_cfg.probe_max_events
                    ):
                        corrected_score_pool = (
                            _materialize_adaptive_omnifold_pool(
                                omnifold_val_shard,
                                omnifold_val_loader_cfg,
                                model=model,
                                sampler=sampler,
                                device=device,
                                world_size=world_size,
                                rank=rank,
                                quota_events=adaptive_cfg.refit_score_events,
                                num_ddim_steps=num_ddim_val,
                                seed=(probe_panel_seed + 1_000_003),
                                include_pairwise_context=(
                                    adaptive_cfg.periodic_pair_features_enabled
                                ),
                                include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
                            )
                        )
                    final_refit_diagnostics = run_adaptive_refit(
                        global_step=int(global_step),
                        state=adaptive_state,
                        cfg=adaptive_cfg,
                        reward_source=omnifold_source,
                        round_ref_model=round_ref_model,
                        policy_snapshot_state_dict=corrected_policy_snapshot,
                        fit_pool=corrected_fit_pool,
                        score_pool=corrected_score_pool,
                        baseline_pool=corrected_baseline_pool,
                        epoch=int(epoch),
                        device=device,
                        world_size=world_size,
                        enforce_round_acceptance=True,
                        progress_callback=lambda phase, row: (
                            _log_omnifold_fit_progress(
                                phase,
                                row,
                                epoch_value=int(epoch),
                            )
                        ),
                    )
                    if not bool(
                        float(
                            final_refit_diagnostics.get(
                                "omnifold/accepted",
                                0.0,
                            )
                        )
                        >= 0.5
                    ):
                        correction_failure_reason = (
                            "final_classifier_gate_failed"
                        )

                if correction_failure_reason:
                    policy_core.load_state_dict(
                        extragradient_anchor_policy,
                        strict=True,
                    )
                    round_ref_model.load_state_dict(
                        extragradient_anchor_policy,
                        strict=True,
                    )
                    freeze_reference_model(round_ref_model)
                    omnifold_source.load_stack_payload(
                        extragradient_anchor_reward
                    )
                    restored_state = type(adaptive_state).from_dict(
                        extragradient_anchor_state
                    )
                    adaptive_state.__dict__.clear()
                    adaptive_state.__dict__.update(restored_state.__dict__)
                    optimizer.state.clear()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer.scheduler.load_state_dict(scheduler_state)
                    if ema_save is not None:
                        ema_save.update(policy_core, decay_=0.0)
                    if ema_rollout is not None:
                        ema_rollout.update(policy_core, decay_=0.0)
                    reference_trust_probe_cache.clear()
                    reference_trust_probe_cache.update(
                        extragradient_anchor_probe_cache
                    )
                    diagnostics.update(final_refit_diagnostics)
                    diagnostics.update(
                        record_extragradient_rejection(
                            adaptive_state,
                            cfg=adaptive_cfg,
                            reason=correction_failure_reason,
                        )
                    )
                    diagnostics.update(
                        {
                            "omnifold/accepted": 0.0,
                            "omnifold/reward_round_id": float(
                                adaptive_state.reward_round_id
                            ),
                            "reference_trust/extragradient/final_reward_installed": 0.0,
                            "staleness/decision": adaptive_state.last_decision,
                        }
                    )
                    reward_checkpoint_metadata = (
                        omnifold_source.checkpoint_metadata()
                    )
                    _assert_current_reward_reference_pairing(
                        "extragradient_rejection"
                    )
                    if is_rank0:
                        _log.warning(
                            "[DGPO/trust/extragradient] rejected corrector "
                            "reason=%s; restored incumbent round=%s",
                            correction_failure_reason,
                            adaptive_state.reward_round_id,
                        )
                    return diagnostics

                refit_diagnostics = final_refit_diagnostics
                # The one-step Adam state was estimated with the transient
                # look-ahead reward. The corrected parameters are committed,
                # but that estimator state must not leak into the new final
                # reward round.
                final_corrector_optimizer_states = len(optimizer.state)
                optimizer.state.clear()
                optimizer.zero_grad(set_to_none=True)
                diagnostics.update(final_refit_diagnostics)
                diagnostics.update(
                    {
                        "reference_trust/extragradient/final_reward_installed": 1.0,
                        "reference_trust/extragradient/final_reward_round_id": float(
                            adaptive_state.reward_round_id
                        ),
                        "reference_trust/extragradient/final_raw_auc_gap": float(
                            final_refit_diagnostics.get(
                                "reference_trust/round_acceptance/current_auc_gap",
                                float("nan"),
                            )
                        ),
                        "reference_trust/extragradient/final_optimizer_states_cleared": float(
                            final_corrector_optimizer_states
                        ),
                    }
                )
                refit_accepted = True
                policy_snapshot = _snapshot_policy_state_dict(model)
                if is_rank0:
                    _log.info(
                        "[DGPO/trust/extragradient] accepted corrected point "
                        "round=%s scale=%.6g D=%.6g",
                        adaptive_state.reward_round_id,
                        corrector_scale,
                        corrector_distance,
                    )
            if signed_recovery_triggered:
                refit_diagnostics[
                    "reference_trust/signed_probe/recovery_reward_installed"
                ] = float(refit_accepted)
                diagnostics[
                    "reference_trust/signed_probe/recovery_reward_installed"
                ] = float(refit_accepted)
                if refit_accepted:
                    # Installing a new reward round resets ordinary trajectory
                    # state. Keep the recovery streak across estimator refreshes
                    # so repeated reverse-only rounds stop at the configured
                    # failed-direction patience instead of looping forever.
                    adaptive_state.trust_failed_direction_streak = int(
                        signed_recovery_diagnostics[
                            "reference_trust/signed_probe/recovery_attempt"
                        ]
                    )
                    adaptive_state.last_decision = (
                        "signed_reverse_only_fresh_reward_installed"
                    )
                else:
                    adaptive_state.last_decision = (
                        "signed_reverse_only_fresh_reward_rejected"
                    )
                    diagnostics["staleness/decision"] = (
                        adaptive_state.last_decision
                    )
            rollback_required = bool(
                float(
                    refit_diagnostics.get(
                        "reference_trust/round_acceptance/rollback_required",
                        0.0,
                    )
                )
                >= 0.5
            )
            if (
                selected_checkpoint
                and not refit_accepted
                and not rollback_required
            ):
                # Residual-closure or acceptance/topology failure is also a
                # failed trajectory proposal.  Do not continue training from
                # an uncertified rewound checkpoint under the old reward.
                delta_before = float(adaptive_state.trust_current_delta)
                adaptive_state.trust_failed_direction_streak += 1
                adaptive_state.trust_round_rollbacks += 1
                if math.isfinite(delta_before) and delta_before > 0.0:
                    adaptive_state.trust_current_delta = max(
                        float(adaptive_cfg.trust_delta_floor),
                        delta_before
                        * float(adaptive_cfg.trust_empirical_shrink_factor),
                    )
                adaptive_state.trust_distance_window.clear()
                adaptive_state.trust_step_acceptance_window.clear()
                adaptive_state.trust_step_scale_window.clear()
                adaptive_state.reset_trust_trajectory(
                    reset_failed_directions=False
                )
                stop_after_rejection = bool(
                    adaptive_state.trust_failed_direction_streak
                    >= int(adaptive_cfg.trust_failed_direction_patience)
                )
                adaptive_state.trust_round_stop_requested = (
                    stop_after_rejection
                )
                adaptive_state.last_decision = (
                    "trajectory_candidate_rejected_stop"
                    if stop_after_rejection
                    else "trajectory_candidate_rejected_rollback"
                )
                diagnostics["staleness/decision"] = (
                    adaptive_state.last_decision
                )
                rejection_metrics = {
                    "reference_trust/round_acceptance/rollback_required": 1.0,
                    "reference_trust/round_acceptance/stop_requested": float(
                        stop_after_rejection
                    ),
                    "reference_trust/round_acceptance/failed_direction_streak": float(
                        adaptive_state.trust_failed_direction_streak
                    ),
                    "reference_trust/round_acceptance/failed_direction_patience": float(
                        adaptive_cfg.trust_failed_direction_patience
                    ),
                    "reference_trust/round_acceptance/delta_before": delta_before,
                    "reference_trust/round_acceptance/delta_after": float(
                        adaptive_state.trust_current_delta
                    ),
                    "reference_trust/trajectory/candidate_gate_failed": 1.0,
                }
                refit_diagnostics.update(rejection_metrics)
                diagnostics.update(rejection_metrics)
                rollback_required = True
            if rollback_required:
                # The installed reward remains paired with round_ref. Restore
                # the live policy to that incumbent anchor and discard AdamW's
                # stale direction before any further optimizer step.
                cleared_optimizer_states, ema_restored = (
                    _restore_live_policy_to_round_reference()
                )
                rollback_metrics = {
                    "reference_trust/round_acceptance/policy_restored": 1.0,
                    "reference_trust/round_acceptance/optimizer_states_cleared": float(
                        cleared_optimizer_states
                    ),
                    "reference_trust/round_acceptance/ema_restored": float(
                        ema_restored
                    ),
                }
                refit_diagnostics.update(rollback_metrics)
                diagnostics.update(rollback_metrics)
                if is_rank0:
                    _log.warning(
                        "[DGPO/trust] round-AUC guard restored policy to "
                        "round_ref=%s and cleared %s optimizer states",
                        adaptive_state.reward_round_id,
                        cleared_optimizer_states,
                    )
            if refit_accepted:
                if adaptive_cfg.trust_fixed_probe_per_reward_round:
                    # The probe contains outputs from the previous round_ref.
                    # Rebuild it on the first optimizer step of the new pair.
                    reference_trust_probe_cache.clear()
                    adaptive_state.trust_probe_payload = None
                    adaptive_state.trust_probe_round_id = -1
                reset_metrics = _reset_optimizer_after_reward_install(
                    optimizer, cfg=adaptive_cfg, accepted=True, adaptive_state=adaptive_state,
                )
                refit_diagnostics.update(reset_metrics)
                diagnostics.update(reset_metrics)
                # A newly installed reward round gets the protocol provenance
                # of the audit that certified its new fixed baseline.
                adaptive_state.audit_protocol_signature = current_audit_signature
                diagnostics.update(
                    {
                        "staleness/decision": adaptive_state.last_decision,
                        "staleness/baseline_auc_gap": float(
                            adaptive_state.baseline_auc_gap
                        ),
                        "staleness/trigger_threshold": float(
                            adaptive_state.trigger_threshold
                        ),
                        "staleness/next_trigger_threshold": float(
                            adaptive_state.trigger_threshold
                        ),
                    }
                )
            if (
                force_refit
                and not refit_accepted
            ):
                # A rejected candidate cannot rewrite the incumbent round's
                # certified baseline. Keep both its threshold and protocol
                # provenance unchanged.
                adaptive_state.last_decision = (
                    "resume_refit_rejected_incumbent_baseline_preserved"
                )
                diagnostics.update(
                    {
                        "omnifold/resume_refit_kept_incumbent": 1.0,
                        "omnifold/resume_refit_preserved_baseline": 1.0,
                        "staleness/decision": adaptive_state.last_decision,
                        "staleness/baseline_auc_gap": float(
                            adaptive_state.baseline_auc_gap
                        ),
                        "staleness/trigger_threshold": float(
                            adaptive_state.trigger_threshold
                        ),
                    }
                )
            reward_checkpoint_metadata = reward_agg.checkpoint_metadata()
        if pending_resume_trust_diagnostics:
            diagnostics.update(pending_resume_trust_diagnostics)
            pending_resume_trust_diagnostics = {}
        _assert_current_reward_reference_pairing("adaptive_cycle")
        if is_rank0:
            _log.info(
                "[DGPO/omnifold] epoch=%s decision=%s gap=%.5g threshold=%.5g round=%s",
                epoch,
                diagnostics.get("staleness/decision", adaptive_state.last_decision),
                float(
                    probe.get(
                        "weighted_auc_gap",
                        probe.get("raw_auc_gap", float("nan")),
                    )
                ),
                float(
                    diagnostics.get(
                        "staleness/trigger_threshold",
                        adaptive_state.trigger_threshold,
                    )
                ),
                adaptive_state.reward_round_id,
            )
        return diagnostics

    experiment_cfg = getattr(global_config, "experiment", {}) or {}
    classifier_only = bool(
        experiment_cfg.get("classifier_only", False)
        if isinstance(experiment_cfg, Mapping)
        else getattr(experiment_cfg, "classifier_only", False)
    )
    if classifier_only:
        # This is a terminal measurement path, not a shortened DGPO run.  It
        # intentionally exits before reward bootstrap, baseline probes, policy
        # validation, checkpointing, or the policy-training loop.
        if not adaptive_cfg.enabled or omnifold_source is None:
            raise ValueError(
                "experiment.classifier_only requires adaptive OmniFold only "
                "as the fixed-pool audit model builder"
            )
        if omnifold_val_shard is None:
            raise RuntimeError(
                "classifier-only audit needs a held-out validation shard"
            )
        if checkpoint_load_mode != "weights_only":
            raise ValueError(
                "classifier-only audit requires dgpo.checkpoint_load_mode=weights_only"
            )
        if start_epoch != 0 or global_step != 0:
            raise ValueError(
                "classifier-only audit must start from policy step 0"
            )
        if adaptive_cfg.bootstrap_on_start:
            raise ValueError(
                "classifier-only audit forbids adaptive_omnifold.recalibration."
                "bootstrap_on_start"
            )
        if adaptive_cfg.baseline_probe_on_start:
            raise ValueError(
                "classifier-only audit forbids adaptive_omnifold."
                "baseline_probe_on_start"
            )
        if int(adaptive_cfg.audit_fit.get("repeats", 1)) != 1:
            raise ValueError("classifier-only probe requires exactly one cold fit")
        classifier_lr_replay = _dgpo_cfg_get(
            getattr(global_config, "experiment", {}), "classifier_design_arm", ""
        ) in {"adapter_decoder_lr_stability", "warmup_cosine_scheduler"}
        if classifier_lr_replay:
            plateau_protocol = _dgpo_cfg_get(
                getattr(global_config, "experiment", {}), "protocol", ""
            ) in {"h4-frozen-readout-quick300-v1", "h4-output-scale300-v1"}
            if plateau_protocol:
                if (
                    adaptive_cfg.audit_fit.get("steps") != 300
                    or adaptive_cfg.audit_fit.get("min_steps") != 300
                    or adaptive_cfg.audit_fit.get("require_saturation") is not False
                    or adaptive_cfg.audit_fit.get("checkpoint_selection_metric", "loss") != "loss"
                    or adaptive_cfg.audit_fit.get("lr_scheduler", "constant") != "constant"
                ):
                    raise ValueError("Plateau screen requires exactly 300 updates, constant LR, BCE selection and no saturation requirement")
            elif _dgpo_cfg_get(
                getattr(global_config, "experiment", {}), "protocol", ""
            ) == "old-classifier-pretrain-v1":
                fit = adaptive_cfg.audit_fit
                if (
                    fit.get("steps") != 3000 or fit.get("min_steps") != 1000
                    or fit.get("validation_patience_epochs") != 10
                    or fit.get("validation_min_delta") != 1e-3
                    or any(fit.get(k) for k in (
                        "periodic_pair_features", "topology_pair_token",
                        "topology_fourier_embedding", "fourier_output_standardization",
                        "topology_conditioning", "topology_direct_logit", "train_last_pet_block"))
                    or fit.get("train_grouped_sequential_embedding") is not True
                    or fit.get("train_invisible_projector") is not True
                    or fit.get("checkpoint_selection_metric") != "loss"
                    or fit.get("lr_scheduler") != "constant"
                ):
                    raise ValueError("Old-pretrain fit requires the historical no-pair classifier and 1000/3000 BCE budget")
            elif _dgpo_cfg_get(
                getattr(global_config, "experiment", {}), "protocol", ""
            ) == "h4-candidate-only-v1":
                fit = adaptive_cfg.audit_fit
                if (fit.get("relation_token_count") != 0 or fit.get("decoder_layers") != 1
                    or fit.get("decoder_hidden_dim") != 128 or fit.get("decoder_heads") != 4
                    or fit.get("steps") != 3000 or fit.get("min_steps") != 1000
                    or fit.get("validation_patience_epochs") != 10
                    or fit.get("validation_min_delta") != 1e-3
                    or any(fit.get(k) for k in ("periodic_pair_features", "topology_pair_token",
                        "topology_fourier_embedding", "fourier_output_standardization",
                        "topology_conditioning", "topology_direct_logit", "visible_pair_rest_frame",
                        "conditional_residual_rank"))
                    or fit.get("require_saturation") is not False
                    or fit.get("checkpoint_selection_metric") != "loss"
                    or fit.get("lr_scheduler") != "constant"):
                    raise ValueError("Candidate-only fit requires two direct candidate tokens, one block and no engineered physics branches")
            elif _dgpo_cfg_get(
                getattr(global_config, "experiment", {}), "protocol", ""
            ) == "h4-relation-tokens-v1":
                fit = adaptive_cfg.audit_fit
                if (fit.get("relation_token_count") != 4 or fit.get("decoder_layers") != 2
                    or fit.get("steps") != 3000 or fit.get("min_steps") != 1000
                    or fit.get("validation_patience_epochs") != 10
                    or fit.get("validation_min_delta") != 1e-3
                    or any(fit.get(k) for k in ("periodic_pair_features", "topology_pair_token",
                        "topology_fourier_embedding", "fourier_output_standardization",
                        "topology_conditioning", "topology_direct_logit", "visible_pair_rest_frame",
                        "conditional_residual_rank"))
                    or fit.get("checkpoint_selection_metric") != "loss"
                    or fit.get("lr_scheduler") != "constant"):
                    raise ValueError("Relation-token fit requires four learned latents, two rounds and no engineered physics branches")
            elif _dgpo_cfg_get(
                getattr(global_config, "experiment", {}), "protocol", ""
            ) == "h4-pair-token-v1":
                fit = adaptive_cfg.audit_fit
                if (
                    fit.get("steps") != 3000 or fit.get("min_steps") != 1000
                    or fit.get("validation_patience_epochs") != 10
                    or fit.get("validation_min_delta") != 1e-3
                    or fit.get("topology_pair_token") is not True
                    or fit.get("periodic_pair_features") is not True
                    or fit.get("topology_fourier_embedding") is not False
                    or fit.get("fourier_output_standardization") is not False
                    or fit.get("topology_conditioning") is not False
                    or fit.get("topology_direct_logit") is not False
                    or fit.get("require_saturation") is not False
                    or fit.get("checkpoint_selection_metric") != "loss"
                    or fit.get("lr_scheduler") != "constant"
                ):
                    raise ValueError("Pair-token fit requires the matched 1000/3000 BCE protocol without Fourier fusion or standardization")
            elif _dgpo_cfg_get(
                getattr(global_config, "experiment", {}), "protocol", ""
            ) == "h4-output-scale-long-v1":
                if (
                    adaptive_cfg.audit_fit.get("steps") != 3000
                    or adaptive_cfg.audit_fit.get("min_steps") != 1000
                    or adaptive_cfg.audit_fit.get("validation_patience_epochs") != 10.0
                    or adaptive_cfg.audit_fit.get("fourier_output_standardization") is not True
                    or adaptive_cfg.audit_fit.get("require_saturation") is not False
                    or adaptive_cfg.audit_fit.get("checkpoint_selection_metric", "loss") != "loss"
                    or adaptive_cfg.audit_fit.get("lr_scheduler", "constant") != "constant"
                ):
                    raise ValueError("Long scaling fit requires 1000 minimum / 3000 maximum updates, ten-epoch patience, fixed standardization, constant LR and BCE selection")
            elif (
                adaptive_cfg.audit_fit.get("steps") is not None
                or adaptive_cfg.audit_fit.get("min_steps") != 1000
                or adaptive_cfg.audit_fit.get("validation_patience_epochs") != 10.0
            ):
                raise ValueError("LR replay requires no maximum, minimum 1000 updates and ten-epoch patience")
        token_conditioning_audit = _dgpo_cfg_get(
            experiment_cfg, "protocol", ""
        ) in {"h4-token-conditioning-fresh-audit-v1", "h4-matched-fold-fresh-audit-v1"}
        if token_conditioning_audit:
            fit = adaptive_cfg.audit_fit
            if (fit.get("steps") is not None or fit.get("min_steps") != 0
                    or fit.get("validation_patience_epochs") != 25
                    or not fit.get("restore_best") or not fit.get("disjoint_final_audit")
                    or fit.get("checkpoint_selection_metric") != "loss"):
                raise ValueError("token-conditioning audit requires the matched 25-epoch patience, best-BCE/disjoint-test protocol")
            if (_dgpo_cfg_get(experiment_cfg, "protocol", "") == "h4-matched-fold-fresh-audit-v1"
                    and (fit.get("training_population") != "omnifold_fold"
                         or fit.get("training_fold") != 1)):
                raise ValueError("matched-fold audit requires the original OmniFold training fold 1")
        if not classifier_lr_replay and not token_conditioning_audit and (
            adaptive_cfg.audit_fit.get("steps") != 3000
            or adaptive_cfg.audit_fit.get("min_steps") != 3000
        ):
            raise ValueError(
                "classifier-only architecture probe requires one exact "
                "3000-step cold audit"
            )

        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
            fit_raw_policy_audit,
        )

        if is_rank0:
            _log.info(
                "[classifier-only] frozen source policy step=%s (audit clock=0): "
                "materializing K=1 samples with training_population=%s, fitting "
                "one cold classifier, then exiting with zero reward fits/policy updates.",
                _dgpo_cfg_get(experiment_cfg, "source_policy_step", None),
                adaptive_cfg.audit_fit.get("training_population", "probe_split"),
            )
        classifier_pool = _materialize_adaptive_omnifold_pool(
            omnifold_val_shard,
            omnifold_val_loader_cfg,
            model=model,
            sampler=sampler,
            device=device,
            world_size=world_size,
            rank=rank,
            quota_events=adaptive_cfg.probe_max_events,
            num_ddim_steps=num_ddim_val,
            seed=int(adaptive_cfg.probe_seed),
            # Keep geometry for held-out ratio diagnostics even when the old
            # classifier does not consume engineered pair features.
            include_pairwise_context=(adaptive_cfg.periodic_pair_features_enabled
                                      or bool(adaptive_cfg.audit_fit.get("ratio_audit_export_dir"))),
            include_visible_pair_rest_frame=(
                adaptive_cfg.visible_pair_rest_frame_enabled
            ),
        )
        classifier_training_pool = _materialize_raw_audit_training_pool(
            omnifold_train_shard, omnifold_train_loader_cfg, cfg=adaptive_cfg,
            model=model, sampler=sampler, device=device, world_size=world_size,
            rank=rank, num_ddim_steps=num_ddim_val, panel_seed=adaptive_cfg.probe_seed,
            evaluation_events=classifier_pool.n_events,
        )
        classifier_result = fit_raw_policy_audit(
            pool=classifier_pool,
            model_builder=omnifold_source.model_builder,
            cfg=adaptive_cfg,
            device=device,
            seed=int(adaptive_cfg.probe_seed),
            # Original step-50 audit retained an empty cache for gradient
            # diagnostics, selecting the identity-based split. Replay it cold.
            warm_start_cache={} if classifier_lr_replay else None,
            **({"training_pool": classifier_training_pool} if classifier_training_pool is not None else {}),
            progress_callback=lambda row: _log_omnifold_fit_progress(
                "raw_staleness_audit",
                row,
                epoch_value=-1,
            ),
        )
        final_metrics: dict[str, Any] = {
            f"staleness/{key}": value
            for key, value in classifier_result.items()
        }
        final_metrics.update(
            {
                "staleness/epoch": -1.0,
                "staleness/global_step": 0.0,
                "classifier_only/enabled": 1.0,
                "classifier_only/policy_updates": 0.0,
                "classifier_only/reward_fits": 0.0,
                "classifier_only/classifier_fits": 1.0,
                "classifier_only/completed": 1.0,
            }
        )
        source_step = _dgpo_cfg_get(experiment_cfg, "source_policy_step", None)
        if source_step is not None:
            final_metrics["classifier_only/source_policy_step"] = int(source_step)
        if is_rank0:
            _log.info(
                "[classifier-only] complete: AUC=%.6g balanced_accuracy=%.6g "
                "classifier_steps=%s; reward_fits=0 policy_updates=0.",
                float(classifier_result["raw_auc"]),
                float(classifier_result["raw_balanced_accuracy"]),
                int(classifier_result["raw_audit_training_steps"]),
            )
            if wandb_mod is not None:
                _wandb_log_step(wandb_mod, final_metrics, step=0)
                try:
                    wandb_mod.run.summary.update(
                        {
                            "classifier_only/enabled": 1,
                            "classifier_only/policy_updates": 0,
                            "classifier_only/reward_fits": 0,
                            "classifier_only/classifier_fits": 1,
                            "classifier_only/completed": 1,
                            "classifier_only/final_auc": float(
                                classifier_result["raw_auc"]
                            ),
                            "classifier_only/final_balanced_accuracy": float(
                                classifier_result["raw_balanced_accuracy"]
                            ),
                            "classifier_only/classifier_steps": int(
                                classifier_result["raw_audit_training_steps"]
                            ),
                        }
                    )
                except Exception as exc:
                    _log.warning(
                        "[classifier-only] W&B summary could not be published: %s",
                        exc,
                    )
        del classifier_pool, classifier_training_pool
        _barrier()
        _finish_wandb_run(wandb_active)
        return

    need_initial_omnifold_bootstrap = bool(
        adaptive_cfg.enabled
        and adaptive_cfg.bootstrap_on_start
        and start_epoch == 0
        and omnifold_source is not None
        and not bool(getattr(omnifold_source, "is_installed", True))
    )
    if need_initial_omnifold_bootstrap:
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import run_adaptive_refit, bootstrap_baseline_pool

        if val_shard is None:
            raise RuntimeError(
                "in-DGPO OmniFold bootstrap needs a held-out validation shard"
            )
        if is_rank0:
            _log.info(
                "[DGPO/omnifold] bootstrap before policy training: materializing "
                "K=1 train/held-out populations; every residual classifier must "
                "saturate before its weights are snapshotted"
            )
        policy_snapshot = _snapshot_policy_state_dict(model)
        score_pool = None if adaptive_cfg.single_pool_train_validation else _materialize_adaptive_omnifold_pool(
            omnifold_val_shard,
            omnifold_val_loader_cfg,
            model=model,
            sampler=sampler,
            device=device,
            world_size=world_size,
            rank=rank,
            quota_events=adaptive_cfg.refit_score_events,
            num_ddim_steps=num_ddim_val,
            seed=adaptive_cfg.probe_seed,
            include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
            include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
        )
        fit_pool = _materialize_adaptive_omnifold_pool(
            omnifold_train_shard,
            omnifold_train_loader_cfg,
            model=model,
            sampler=sampler,
            device=device,
            world_size=world_size,
            rank=rank,
            quota_events=(None if fixed_omnifold_pool else adaptive_cfg.pool_events),
            num_ddim_steps=num_ddim,
            seed=adaptive_cfg.seed,
            include_pairwise_context=adaptive_cfg.periodic_pair_features_enabled,
            include_visible_pair_rest_frame=adaptive_cfg.visible_pair_rest_frame_enabled,
        )
        if adaptive_cfg.single_pool_train_validation:
            score_pool = fit_pool
        bootstrap_metrics = run_adaptive_refit(
            global_step=int(global_step),
            state=adaptive_state,
            cfg=adaptive_cfg,
            reward_source=omnifold_source,
            round_ref_model=round_ref_model,
            policy_snapshot_state_dict=policy_snapshot,
            fit_pool=fit_pool,
            score_pool=score_pool,
            # Residual closure uses the full score pool; establish the trigger
            # baseline with the same cheap population used by routine audits.
            baseline_pool=bootstrap_baseline_pool(adaptive_cfg, score_pool),
            epoch=-1,
            device=device,
            world_size=world_size,
            enforce_round_acceptance=False,
            progress_callback=lambda phase, row: _log_omnifold_fit_progress(
                phase,
                row,
                epoch_value=-1,
            ),
        )
        bootstrap_metrics["omnifold/bootstrap_on_start"] = 1.0
        reward_checkpoint_metadata = reward_agg.checkpoint_metadata()
        accepted = float(bootstrap_metrics.get("omnifold/accepted", 0.0)) >= 0.5
        if not accepted or not bool(getattr(omnifold_source, "is_installed", False)):
            reason = bootstrap_metrics.get(
                "omnifold/accept_reason", "initial ratio stack was not installed"
            )
            if adaptive_cfg.bootstrap_fail_closed:
                raise RuntimeError(
                    "initial in-DGPO OmniFold bootstrap failed closed: " + str(reason)
                )
            raise RuntimeError(
                "DGPO cannot start without an installed OmniFold reward: " + str(reason)
            )
        bootstrap_metrics.update(_reset_optimizer_after_reward_install(
            optimizer, cfg=adaptive_cfg, accepted=True, adaptive_state=adaptive_state,
        ))
        if adaptive_cfg.iteration_one_only and adaptive_cfg.log_only:
            from RL.DGPO_neutrino.omnifold_ztautau.adaptive import frozen_classifier_metrics
            bootstrap_metrics.update(frozen_classifier_metrics(
                omnifold_source.frozen_reward,
                score_pool.prefix(adaptive_cfg.probe_max_events),
                row_budget=adaptive_cfg.score_row_budget,
            ))
        # Save the installed reward before the first raw-monitor fit. If that
        # long fit is interrupted, resume it without repeating OmniFold.
        adaptive_state.mark_bootstrap_complete(cfg=adaptive_cfg)
        _assert_current_reward_reference_pairing("initial_omnifold_bootstrap")
        if is_rank0:
            _log.info(
                "[DGPO/omnifold] bootstrap installed round=%s stored_iterations=%s "
                "classifier_fits=%s",
                adaptive_state.reward_round_id,
                bootstrap_metrics.get("omnifold/iterations_fitted"),
                bootstrap_metrics.get("omnifold/classifier_fits_total"),
            )
            if wandb_mod is not None:
                _wandb_log_step(
                    wandb_mod,
                    {"epoch": -1, **bootstrap_metrics},
                    step=int(global_step),
                )
            # Match the EveNet-private recovery boundary: persist the installed
            # reward/controller/reference triplet before the potentially long
            # epoch=-1 validation pass.  A timeout after a successful full-pool
            # fit can then resume at DGPO epoch 0 without repeating bootstrap.
            _dgpo_save_last_ckpt(
                model,
                ema_save,
                optimizer,
                ref_model,
                last_completed_epoch=-1,
                dgpo_next_epoch=0,
                global_step=int(global_step),
                ema_rollout=ema_rollout,
                round_ref_model=round_ref_model,
                reward_round_id=int(adaptive_state.reward_round_id),
                dgpo_projection_constraint_state=(
                    _dgpo_constraint_checkpoint_payload(constraint_state)
                ),
                dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
            )
        _barrier()

        # Bootstrap used the full train/validation pools. Release them before
        # allocating the configured raw-monitor panel.
        del fit_pool, score_pool, policy_snapshot

    fixed_schedule_step_zero_audit_done = any(
        int(row.get("global_step", -1)) == 0
        and float(row.get("fixed_schedule_diagnostic_only", 0.0)) >= 0.5
        and (
            adaptive_cfg.audit_fit.get("training_population") != "omnifold_fold"
            or (
                float(row.get("raw_audit_uses_omnifold_fold", 0.0)) == 1.0
                and int(row.get("raw_audit_training_fold", 0))
                == int(adaptive_cfg.audit_fit.get("training_fold", 1))
            )
        )
        for row in adaptive_state.probe_history
    )
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import step_zero_raw_audit_enabled
    if (
        adaptive_cfg.enabled
        and step_zero_raw_audit_enabled(adaptive_cfg)
        and start_epoch == 0
        and global_step == 0
        and bool(getattr(omnifold_source, "is_installed", False))
        and not fixed_schedule_step_zero_audit_done
    ):
        if is_rank0:
            _log.info(
                "[DGPO/omnifold] fitting the selection-blind step-0 raw "
                "audit before the first policy update"
            )
        step_zero_audit_metrics = _run_adaptive_cycle(
            epoch=-1,
            diagnostic_raw_only=True,
        )
        if float(
            step_zero_audit_metrics.get(
                "staleness/raw_audit_saturated", 0.0
            )
        ) < 0.5:
            raise RuntimeError(
                "selection-blind step-0 raw audit did not saturate"
            )
        # This fully trained audit replaces the controller-oriented bootstrap
        # baseline for fixed-schedule runs. The reward schedule has no AUC
        # threshold, so a second state-mutating raw-baseline fit is unnecessary.
        adaptive_state.raw_monitor_baseline_pending = False
        if is_rank0:
            if wandb_mod is not None:
                _wandb_log_step(
                    wandb_mod,
                    {"epoch": -1, **step_zero_audit_metrics},
                    step=int(global_step),
                )
            # Persist the completed audit marker. A preemption after this point
            # resumes directly into epoch 0 instead of fitting the baseline a
            # second time.
            _dgpo_save_last_ckpt(
                model,
                ema_save,
                optimizer,
                ref_model,
                last_completed_epoch=-1,
                dgpo_next_epoch=0,
                global_step=int(global_step),
                ema_rollout=ema_rollout,
                round_ref_model=round_ref_model,
                reward_round_id=int(adaptive_state.reward_round_id),
                dgpo_projection_constraint_state=(
                    _dgpo_constraint_checkpoint_payload(constraint_state)
                ),
                dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
            )
        _barrier()

    resume_refit_version_completed = bool(
        adaptive_state.resume_refit_once_completed
        and (
            not adaptive_cfg.refit_once_id
            or adaptive_state.resume_refit_once_id == adaptive_cfg.refit_once_id
        )
    )
    need_resume_adapter_refit = bool(
        adaptive_cfg.enabled
        and adaptive_cfg.refit_once_on_resume
        and start_epoch > 0
        and omnifold_source is not None
        and bool(getattr(omnifold_source, "is_installed", False))
        and not resume_refit_version_completed
    )
    if need_resume_adapter_refit:
        startup_refit_epoch = int(start_epoch) - 1
        if is_rank0:
            _log.info(
                "[DGPO/omnifold] running the checkpointed one-shot adapter "
                "refit version=%s before resumed DGPO epoch %s",
                adaptive_cfg.refit_once_id or "<unversioned>",
                start_epoch,
            )
        startup_refit_metrics = _run_adaptive_cycle(
            epoch=startup_refit_epoch,
            force_refit=True,
        )
        startup_refit_accepted = float(
            startup_refit_metrics.get("omnifold/accepted", 0.0)
        )
        if adaptive_cfg.refit_once_fail_closed and startup_refit_accepted < 0.5:
            startup_refit_metrics.update(
                {
                    "omnifold/resume_refit_once_completed": 0.0,
                    "omnifold/resume_refit_once_accepted": 0.0,
                }
            )
            if is_rank0 and wandb_mod is not None:
                _wandb_log_step(
                    wandb_mod,
                    {"epoch": startup_refit_epoch, **startup_refit_metrics},
                    step=int(global_step),
                )
            _barrier()
            raise RuntimeError(
                "one-shot resume OmniFold refit failed its acceptance gate; "
                "refusing to continue an ablation with the incumbent reward"
            )
        # Set this only after the complete distributed audit/refit returns. A
        # timeout during fitting leaves the old checkpoint marker false, while
        # the successful recovery checkpoint below makes every later resume a
        # pure continuation with no repeated startup refit.
        adaptive_state.resume_refit_once_completed = True
        adaptive_state.resume_refit_once_id = adaptive_cfg.refit_once_id
        startup_refit_metrics.update(
            {
                "omnifold/resume_refit_once_completed": 1.0,
                "omnifold/resume_refit_once_accepted": startup_refit_accepted,
            }
        )
        _assert_current_reward_reference_pairing("resume_adapter_refit_once")
        if is_rank0:
            if wandb_mod is not None:
                _wandb_log_step(
                    wandb_mod,
                    {"epoch": startup_refit_epoch, **startup_refit_metrics},
                    step=int(global_step),
                )
            _dgpo_save_last_ckpt(
                model,
                ema_save,
                optimizer,
                ref_model,
                last_completed_epoch=startup_refit_epoch,
                dgpo_next_epoch=int(start_epoch),
                global_step=int(global_step),
                ema_rollout=ema_rollout,
                round_ref_model=round_ref_model,
                reward_round_id=int(adaptive_state.reward_round_id),
                dgpo_projection_constraint_state=(
                    _dgpo_constraint_checkpoint_payload(constraint_state)
                ),
                dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
            )
            _log.info(
                "[DGPO/omnifold] one-shot adapter refit version=%s saved; "
                "later resumes will inherit round=%s without repeating it",
                adaptive_cfg.refit_once_id or "<unversioned>",
                adaptive_state.reward_round_id,
            )
        _barrier()

    if _clear_unused_raw_monitor_baseline(adaptive_state, adaptive_cfg):
        if is_rank0:
            _log.info(
                "[DGPO/omnifold] skipping unused startup raw-monitor baseline "
                "disabled by monitoring config; inherited reward/reference unchanged"
            )

    if adaptive_state.raw_monitor_baseline_pending:
        if adaptive_cfg.monitor_mode != "raw_plateau_refit":
            raise ValueError("new best-point experiment requires raw_plateau_refit monitoring")
        baseline_metrics = _run_adaptive_cycle(epoch=-1, raw_baseline_only=True)
        if float(baseline_metrics.get("staleness/raw_audit_saturated", 0.0)) < 0.5:
            raise RuntimeError("best-point raw monitor baseline did not saturate; refusing policy updates")
        adaptive_state.raw_monitor_baseline_pending = False
        if is_rank0:
            if wandb_mod is not None:
                _wandb_log_step(wandb_mod, {"epoch": -1, **baseline_metrics}, step=int(global_step))
            _dgpo_save_last_ckpt(
                model, ema_save, optimizer, ref_model,
                last_completed_epoch=-1, dgpo_next_epoch=0, global_step=int(global_step),
                ema_rollout=ema_rollout, round_ref_model=round_ref_model,
                reward_round_id=int(adaptive_state.reward_round_id),
                dgpo_projection_constraint_state=_dgpo_constraint_checkpoint_payload(constraint_state),
                dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
            )
        _barrier()

    if adaptive_cfg.trust_boundary_enabled:
        if (
            not math.isfinite(adaptive_state.trust_current_delta)
            or adaptive_state.trust_current_delta <= 0.0
        ):
            raise RuntimeError(
                "adaptive trust boundary has no calibrated round radius; "
                "enable a successful bootstrap/refit_once_on_resume for this ablation"
            )
        if is_rank0:
            _log.info(
                "[DGPO/trust] round=%s raw_auc=%.6g delta=%.6g closed=%s.",
                adaptive_state.reward_round_id,
                adaptive_state.trust_current_raw_auc,
                adaptive_state.trust_current_delta,
                adaptive_state.trust_statistically_closed,
            )

    # Following the EveNet ``train.py`` pattern: no rank-0-only synchronous setup
    # before the training loop.  All ranks proceed straight into ``fit``-style
    # iteration and hit the data pipeline simultaneously, avoiding NCCL barriers
    # that would otherwise busy-wait the GPU while rank 0 does cold-start work.
    ve_initial = int(val_events) if val_events is not None else 0
    if (
        bool(dg.get("validation_initial_enabled", True))
        and _should_log_pretraining_baseline(start_epoch, global_step)
        and ve_initial > 0
        and val_shard is not None
    ):
        if is_rank0:
            _log.info(
                "[DGPO] val: running pre-DGPO baseline validation (epoch=-1) for response diagnostics."
            )
        val_loader = val_shard.iter_torch_batches(**val_loader_cfg)
        est_val_batches = (
            max(1, math.ceil(ve_initial / validation_effective_batch))
            if ve_initial > 0
            else None
        )
        initial_val_metrics = run_validation_epoch(
            model,
            ref_model,
            ema_save,
            val_loader,
            sampler,
            reward_agg,
            val_K=val_K,
            num_ddim_steps=num_ddim_val,
            device=device,
            dtype=dtype,
            cartesian=_truth_generation_cartesian(),
            compute_winrate=bool(dg.get("validation_compute_winrate", False)),
            epoch=-1,
            est_total_batches=est_val_batches,
            val_log_batches=bool(dg.get("validation_log_batches", True)),
            val_rollout_parallel_chains=val_rollout_parallel_chains,
            val_tqdm_k_chains=bool(dg.get("validation_tqdm_k_chains", True)),
            val_tqdm_ddim=bool(dg.get("validation_tqdm_ddim", False)),
            max_batches=val_max_batches,
            initial_state=None,
            rank=rank,
            world_size=world_size,
        )
        maybe_initial_state = initial_val_metrics.get("_val_initial_state")
        if isinstance(maybe_initial_state, dict):
            val_baseline_state = maybe_initial_state

        if is_rank0:
            _append_validation_history_plots(initial_val_metrics, epoch_value=-1)
            profile_summary = " ".join(
                (
                    f"{profile_name}_slope="
                    f"{initial_val_metrics.get(f'val_diagnostics/profile/{profile_name}/slope', float('nan')):.6g} "
                    f"{profile_name}_zero="
                    f"{initial_val_metrics.get(f'val_diagnostics/profile/{profile_name}/zero_delta_truth', float('nan')):.6g}"
                )
                for profile_name in val_profile_history.keys()
            )
            _log.info(
                "[DGPO] initial val r_mean=%.6f %s",
                initial_val_metrics["val/reward/mean"],
                profile_summary.strip(),
            )
            if wandb_mod is not None:
                _wandb_log_validation(
                    wandb_mod,
                    initial_val_metrics,
                    epoch=-1,
                    wandb_step=_wandb_train_step(global_step),
                )
        _barrier()
    elif (start_epoch > 0 or global_step > 0) and ve_initial > 0 and is_rank0:
        _log.warning(
            "[DGPO] Response matrices need the pre-DGPO validation baseline; "
            "this run is resuming at start_epoch=%s, so val/response/* will be skipped.",
            start_epoch,
        )

    if (
        adaptive_cfg.enabled
        and adaptive_cfg.baseline_probe_on_start
        and start_epoch == 0
        and not adaptive_state.calibrated
    ):
        baseline_adaptive_metrics = _run_adaptive_cycle(
            epoch=-1,
            baseline_only=True,
        )
        if is_rank0 and wandb_mod is not None:
            _wandb_log_step(
                wandb_mod,
                baseline_adaptive_metrics,
                step=int(global_step),
            )
        _barrier()

    def constraint_ckpt_payload_for_save() -> dict[str, Any] | None:
        return _dgpo_constraint_checkpoint_payload(constraint_state)

    adaptive_early_stop = False
    trust_rejection_endpoint_requested = bool(
        adaptive_state.trust_rejection_stop_requested
    )
    try:
        legacy_train_kinematics = _supports_legacy_invisible_kinematics(
            cartesian=_truth_generation_cartesian(),
            feature_dim=len(_invisible_feature_names()) or None,
        )
        # With a configured step budget, an epoch is a control/logging interval,
        # not a data boundary. Keep the streaming Ray iterator alive across those
        # logical epochs so a 10-step epoch does not repeatedly consume the start
        # of the shard. The iterator is renewed only after a complete data pass.
        logical_epoch_step_budget = (
            None if configured_steps_per_epoch is None else steps_per_epoch
        )
        train_it: Any | None = None
        completed_data_passes = 0

        from itertools import count
        epoch_iterator = count(start_epoch) if epochs is None else range(start_epoch, epochs)
        for epoch in epoch_iterator:
            if train_it is None:
                # ``local_shuffle_buffer_size`` provides per-shard shuffling for
                # every complete data pass.
                train_it = iter(train_shard.iter_torch_batches(**train_loader_cfg))
            collect_train_dist_epoch = bool(
                wandb_flag
                and train_dist_enabled
                and scheduled_epoch(
                    epoch,
                    train_dist_every,
                    include_epoch_zero=True,
                )
            )

            # These duplicate validation arrays and can be large. Do not even
            # allocate the epoch accumulators unless this epoch will upload them.
            td_all_feature_names: tuple[str, ...] = ()
            td_all_chunks: dict[str, list[np.ndarray]] = {}
            td_all_class_chunks: list[np.ndarray] = []
            if collect_train_dist_epoch:
                num_diag_bins = _resolve_diagnostic_num_bins()
                td_pt_p = np.zeros(num_diag_bins, dtype=np.float64)
                td_pt_t = np.zeros(num_diag_bins, dtype=np.float64)
                td_e_p = np.zeros(num_diag_bins, dtype=np.float64)
                td_e_t = np.zeros(num_diag_bins, dtype=np.float64)
                td_p_p = np.zeros(num_diag_bins, dtype=np.float64)
                td_p_t = np.zeros(num_diag_bins, dtype=np.float64)
                td_k1_pt_p = np.zeros(num_diag_bins, dtype=np.float64)
                td_k1_pt_t = np.zeros(num_diag_bins, dtype=np.float64)
                td_k1_e_p = np.zeros(num_diag_bins, dtype=np.float64)
                td_k1_e_t = np.zeros(num_diag_bins, dtype=np.float64)
                td_k1_p_p = np.zeros(num_diag_bins, dtype=np.float64)
                td_k1_p_t = np.zeros(num_diag_bins, dtype=np.float64)
                td_all_feature_names = _generation_monitor_feature_names(
                    cartesian=_truth_generation_cartesian()
                )
                td_all_chunks = {
                    f"{feature_name}_{suffix}": []
                    for feature_name in td_all_feature_names
                    for suffix in ("truth", "pred")
                }
            steps_this_epoch = (
                _resume_logical_epoch_step(ckpt_dict, logical_epoch_step_budget)
                if epoch == start_epoch else 0
            )
            while True:
                if (
                    logical_epoch_step_budget is not None
                    and steps_this_epoch >= logical_epoch_step_budget
                ):
                    break
                if max_steps is not None and global_step >= max_steps:
                    last_done = epoch
                    if is_rank0:
                        _dgpo_save_last_ckpt(
                            model,
                            ema_save,
                            optimizer,
                            ref_model,
                            last_completed_epoch=last_done,
                            dgpo_next_epoch=epoch,
                            global_step=global_step,
                            dgpo_epoch_step=steps_this_epoch,
                            ema_rollout=ema_rollout,
                            round_ref_model=(
                                round_ref_model if adaptive_cfg.enabled else None
                            ),
                            reward_round_id=int(adaptive_state.reward_round_id),
                            dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                            dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                            dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                            dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
                        )
                        _log.info("[DGPO] max_steps=%s reached; stopping.", max_steps)
                    _barrier()
                    return

                batch_cpu, has_more = _next_batch_synced(
                    train_it, world_size=world_size, device=device
                )
                if not has_more or batch_cpu is None:
                    train_it = None
                    completed_data_passes += 1
                    if is_rank0:
                        _log.info(
                            "[DGPO] completed data pass=%s at logical_epoch=%s "
                            "global_step=%s.",
                            completed_data_passes,
                            epoch,
                            global_step,
                        )
                    if logical_epoch_step_budget is None:
                        break
                    train_it = iter(
                        train_shard.iter_torch_batches(**train_loader_cfg)
                    )
                    batch_cpu, has_more = _next_batch_synced(
                        train_it, world_size=world_size, device=device
                    )
                    if not has_more or batch_cpu is None:
                        raise RuntimeError(
                            "DGPO training dataset produced no batches after "
                            "restarting its iterator"
                        )

                batch_d = batch_to_device(batch_cpu, device)
                reward_dist_step = wandb_active and (
                    not _wandb_simplified_enabled()
                    and global_step % log_reward_dist_every == 0
                )
                diagnostic_dist_step = wandb_active
                if (
                    adaptive_cfg.trust_fixed_probe_per_reward_round
                    and reference_trust_probe_cache.get("round_id")
                    not in (None, int(adaptive_state.reward_round_id))
                ):
                    reference_trust_probe_cache.clear()
                trust_lr_diagnostics = (
                    adaptive_trust_policy_lr_scale(
                        adaptive_state,
                        cfg=adaptive_cfg,
                    )
                    if adaptive_cfg.trust_boundary_enabled
                    else {}
                )
                trust_policy_lr_scale = float(
                    trust_lr_diagnostics.get(
                        "reference_trust/policy_lr/scale",
                        1.0,
                    )
                )
                round_warmup_metrics = policy_round_warmup_metrics(adaptive_state, cfg=adaptive_cfg)
                round_warmup_scale = round_warmup_metrics.get("train/round_warmup/lr_scale", 1.0)
                scheduled_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
                metrics = train_step(
                    model,
                    round_ref_model if adaptive_cfg.enabled else ref_model,
                    ema_rollout,
                    ema_save,
                    batch_d,
                    optimizer,
                    sampler,
                    reward_agg,
                    beta=beta,
                    K=K,
                    advantage_estimator=advantage_estimator,
                    num_ddim_steps=num_ddim,
                    rollout_parallel_chains=rollout_parallel_chains,
                    global_step=global_step,
                    epoch=epoch,
                    device=device,
                    dtype=dtype,
                    reference_trust_coefficient=reference_trust_coefficient,
                    reference_trust_objective=reference_trust_objective,
                    reference_trust_vp_path_kl_diagnostic=(
                        reference_trust_vp_path_kl_diagnostic
                    ),
                    reference_trust_vp_logsnr_min=(
                        reference_trust_vp_logsnr_min
                    ),
                    reference_trust_vp_logsnr_max=(
                        reference_trust_vp_logsnr_max
                    ),
                    reference_trust_max_ratio=(
                        float(adaptive_state.trust_current_delta)
                        if adaptive_cfg.trust_boundary_enabled
                        else None
                    ),
                    reference_trust_distance=adaptive_cfg.trust_distance,
                    reference_trust_warning_fraction=(
                        adaptive_cfg.trust_warning_fraction
                    ),
                    reference_trust_backtrack_factor=(
                        adaptive_cfg.trust_backtrack_factor
                    ),
                    reference_trust_max_backtracks=(
                        adaptive_cfg.trust_max_backtracks
                    ),
                    reference_trust_probe_events_per_rank=(
                        adaptive_cfg.trust_probe_events_per_rank
                    ),
                    reference_trust_fixed_probe_per_round=(
                        adaptive_cfg.trust_fixed_probe_per_reward_round
                    ),
                    reference_trust_interior_fraction=(
                        adaptive_cfg.trust_interior_fraction
                    ),
                    reference_trust_policy_lr_scale=trust_policy_lr_scale * round_warmup_scale,
                    reference_trust_reset_adam_first_moment_on_zero_step=(
                        adaptive_cfg.trust_reset_adam_first_moment_on_zero_step
                    ),
                    reference_trust_transactional_rejection=(
                        adaptive_cfg.trust_transactional_rejection
                    ),
                    reference_trust_probe_cache=(
                        reference_trust_probe_cache
                        if adaptive_cfg.trust_fixed_probe_per_reward_round
                        else None
                    ),
                    log_reward_dist=reward_dist_step,
                    log_diagnostic_dist=diagnostic_dist_step,
                    collect_train_dist=collect_train_dist_epoch,
                    diagnostic_plot_names=diagnostic_plot_names,
                    diagnostic_plot_every=diagnostic_plot_every,
                    num_train_timesteps=num_train_timesteps,
                    policy_eval_parallel_timesteps=policy_eval_parallel_timesteps,
                    policy_eval_event_microbatch_size=(
                        policy_eval_event_microbatch_size
                    ),
                    adv_clip_max=adv_clip_max_cfg,
                    grad_clip_norm=grad_clip_norm_cfg,
                    policy_eval_t_min=policy_eval_t_min_cfg,
                    policy_eval_t_max=policy_eval_t_max_cfg,
                    constraint_state=constraint_state,
                    world_size=world_size,
                    parameter_update_rms_target=(
                        parameter_update_rms_target_cfg
                    ),
                    parameter_update_rms_min_scale=(
                        parameter_update_rms_min_scale_cfg
                    ),
                    parameter_update_rms_max_scale=(
                        parameter_update_rms_max_scale_cfg
                    ),
                )
                metrics.update(trust_lr_diagnostics)
                metrics.update(round_warmup_metrics)
                metrics["train/lr/scheduled_max"] = max(scheduled_lrs)
                metrics["train/lr/scheduled_min"] = min(scheduled_lrs)
                for index, (group, rate) in enumerate(zip(optimizer.param_groups, scheduled_lrs, strict=True)):
                    metrics[f"train/lr/scheduled/{group.get('group_name', index)}"] = rate
                advance_policy_round_warmup(
                    adaptive_state, cfg=adaptive_cfg,
                    accepted=bool(metrics.get("train/optimizer_step_ran", 0.0) >= .5),
                )
                if (
                    adaptive_cfg.trust_fixed_probe_per_reward_round
                    and reference_trust_probe_cache.get("dirty", False)
                ):
                    adaptive_state.trust_probe_payload = dict(
                        reference_trust_probe_cache["payload"]
                    )
                    adaptive_state.trust_probe_round_id = int(
                        adaptive_state.reward_round_id
                    )
                    reference_trust_probe_cache["round_id"] = int(
                        adaptive_state.reward_round_id
                    )
                    reference_trust_probe_cache["dirty"] = False
                if adaptive_cfg.trust_empirical_radius_enabled:
                    metrics.update(
                        record_reference_trust_attempt(
                            adaptive_state,
                            cfg=adaptive_cfg,
                            accepted=(
                                float(
                                    metrics.get(
                                        "reference_trust/step_accepted",
                                        0.0,
                                    )
                                )
                                >= 0.5
                            ),
                            update_scale=float(
                                metrics.get(
                                    "reference_trust/accepted_step_scale",
                                    0.0,
                                )
                            ),
                            distance=float(
                                metrics.get(
                                    "reference_trust/post_step_distance",
                                    float("nan"),
                                )
                            ),
                        )
                    )
                trust_boundary_this_step = bool(
                    adaptive_cfg.trust_boundary_enabled
                    and float(
                        metrics.get("reference_trust/boundary_hit", 0.0)
                    )
                    >= 0.5
                )
                if trust_boundary_this_step:
                    adaptive_state.trust_boundary_count += 1
                    if is_rank0:
                        _log.info(
                            "[DGPO/trust] strict boundary handled in-step without "
                            "forcing a reward refit (boundary_count=%s, scale=%.6g, "
                            "D_post=%.6g).",
                            adaptive_state.trust_boundary_count,
                            float(
                                metrics.get(
                                    "reference_trust/accepted_step_scale",
                                    0.0,
                                )
                            ),
                            float(
                                metrics.get(
                                    "reference_trust/post_step_distance",
                                    float("nan"),
                                )
                            ),
                        )
                trust_step_accepted = bool(
                    float(
                        metrics.get("reference_trust/step_accepted", 0.0)
                    )
                    >= 0.5
                )
                if adaptive_cfg.trust_boundary_enabled and trust_step_accepted:
                    adaptive_state.trust_accepted_updates += 1
                trust_rejection_this_step = bool(
                    adaptive_cfg.trust_boundary_enabled
                    and adaptive_cfg.trust_stop_after_rejection
                    and trust_boundary_this_step
                    and not trust_step_accepted
                )
                if trust_rejection_this_step:
                    adaptive_state.trust_rejection_stop_requested = True
                    if adaptive_state.trust_first_rejected_global_step < 0:
                        adaptive_state.trust_first_rejected_global_step = int(
                            global_step
                        ) + 1
                    trust_rejection_endpoint_requested = True
                if adaptive_cfg.trust_boundary_enabled:
                    metrics.update(
                        {
                            "reference_trust/accepted_updates": float(
                                adaptive_state.trust_accepted_updates
                            ),
                            "reference_trust/rejection_stop_requested": float(
                                adaptive_state.trust_rejection_stop_requested
                            ),
                            "reference_trust/first_rejected_global_step": float(
                                adaptive_state.trust_first_rejected_global_step
                            ),
                        }
                    )
                if adaptive_cfg.trust_boundary_enabled:
                    metrics["reference_trust/boundary_count"] = float(
                        adaptive_state.trust_boundary_count
                    )
                    if adaptive_cfg.trust_radius_mode == "round_decay":
                        metrics["reference_trust/round_decay/step"] = float(
                            adaptive_state.trust_radius_decay_step
                        )
                        metrics["reference_trust/radius_mode_round_decay"] = 1.0
                if wandb_mod is not None:
                    payload = _wandb_train_payload(metrics)
                    payload["epoch"] = float(epoch)
                    # This update has finished; use the same count that will
                    # be saved in the checkpoint and used by the next monitor.
                    payload["global_step"] = int(global_step) + 1
                    _wandb_log_step(wandb_mod, payload, step=global_step + 1)
                    _append_profile_accum(metrics)
                    _flush_profile_accum(step=global_step + 1)
                if (
                    bool(
                        _dgpo_cfg_get(
                            global_config.dgpo,
                            "fail_on_skipped_optimizer_step",
                            False,
                        )
                    )
                    and float(metrics.get("train/optimizer_step_ran", 0.0)) < 0.5
                ):
                    raise RuntimeError(
                        "fixed-update experiment encountered a skipped/nonfinite "
                        f"optimizer step at global_step={global_step}; refusing "
                        "to count it toward the round budget"
                    )

                if collect_train_dist_epoch and legacy_train_kinematics:
                    td_pt_p += metrics["_kin_h_pt_p"]
                    td_pt_t += metrics["_kin_h_pt_t"]
                    td_e_p += metrics["_kin_h_e_p"]
                    td_e_t += metrics["_kin_h_e_t"]
                    td_p_p += metrics["_kin_h_p_p"]
                    td_p_t += metrics["_kin_h_p_t"]
                    td_k1_pt_p += metrics["_kin_h_pt_k1_p"]
                    td_k1_pt_t += metrics["_kin_h_pt_k1_t"]
                    td_k1_e_p += metrics["_kin_h_e_k1_p"]
                    td_k1_e_t += metrics["_kin_h_e_k1_t"]
                    td_k1_p_p += metrics["_kin_h_p_k1_p"]
                    td_k1_p_t += metrics["_kin_h_p_k1_t"]
                if collect_train_dist_epoch:
                    for feature_name in td_all_feature_names:
                        truth_key = f"_kin_all_{feature_name}_t"
                        pred_key = f"_kin_all_{feature_name}_p"
                        if truth_key not in metrics or pred_key not in metrics:
                            continue
                        td_all_chunks[f"{feature_name}_truth"].append(metrics[truth_key])
                        td_all_chunks[f"{feature_name}_pred"].append(metrics[pred_key])
                    if "_kin_all_class_index" in metrics:
                        td_all_class_chunks.append(metrics["_kin_all_class_index"])

                if is_rank0 and global_step % log_every == 0:
                    _log.info(
                        "epoch=%s step=%s L_total=%.6f L_dgpo=%.6f "
                        "L_cur=%.4f L_ref=%.4f delta=%.4f "
                        "r_best=%.4f r_med=%.4f gap=%.4f",
                        epoch,
                        global_step + 1,
                        metrics["train/loss/total"],
                        metrics["train/loss/dgpo"],
                        metrics["train/loss/L_cur"],
                        metrics["train/loss/L_ref"],
                        metrics["train/loss/delta"],
                        metrics["reward/monitor/best_of_k"],
                        metrics["reward/monitor/median"],
                        metrics["reward/monitor/mean_gap"],
                    )
                global_step += 1
                steps_this_epoch += 1

                if trust_rejection_this_step:
                    if is_rank0:
                        _log.info(
                            "[DGPO/trust] transactional proposal rejection at "
                            "global_step=%s; running the cold endpoint audit and "
                            "ending this boundary experiment.",
                            global_step,
                        )
                    break

                # Check the raw policy midway through the logical epoch without
                # changing epoch length, LR schedule or epoch-end validation.
                # The last step is handled below, with the independent trust fit.
                if (
                    adaptive_cfg.enabled
                    and logical_epoch_step_budget is not None
                    and steps_this_epoch < logical_epoch_step_budget
                    and should_probe_training_boundary(
                        adaptive_state, cfg=adaptive_cfg, epoch=epoch,
                        global_step=global_step, epoch_end=False,
                    )
                ):
                    step_monitor_metrics = _run_adaptive_cycle(
                        epoch=epoch, checkpoint_next_epoch=epoch,
                        allow_classifier_trust=False,
                    )
                    if is_rank0:
                        if wandb_mod is not None:
                            _wandb_log_with_step(
                                wandb_mod, step_monitor_metrics,
                                step=_wandb_epoch_end_step(global_step),
                            )
                        _dgpo_save_last_ckpt(
                            model, ema_save, optimizer, ref_model,
                            last_completed_epoch=epoch, dgpo_next_epoch=epoch,
                            global_step=global_step, dgpo_epoch_step=steps_this_epoch,
                            ema_rollout=ema_rollout, round_ref_model=round_ref_model,
                            reward_round_id=int(adaptive_state.reward_round_id),
                            dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                            dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                            dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                            dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
                        )
                    _barrier()

                    if adaptive_state.raw_global_stop_requested:
                        adaptive_early_stop = True
                        if is_rank0:
                            _log.info("[DGPO/global-best] saved terminal mid-epoch state at step=%s; stopping after %s failed refit rounds. Best checkpoint remains %s",
                                      global_step, adaptive_state.raw_global_failed_rounds, adaptive_state.raw_best_checkpoint)
                        break

            # --- Epoch-end: build training-distribution figures from accumulated histograms ---
            if adaptive_state.raw_global_stop_requested:
                break  # Already saved with correct within-epoch progress above.
            if wandb_mod is not None:
                _flush_profile_accum(step=global_step, force=True)

            if collect_train_dist_epoch and legacy_train_kinematics and world_size > 1:
                td_stack = np.stack([
                    td_pt_p, td_pt_t, td_e_p, td_e_t, td_p_p, td_p_t,
                    td_k1_pt_p, td_k1_pt_t, td_k1_e_p, td_k1_e_t, td_k1_p_p, td_k1_p_t,
                ])
                td_hist_t = torch.from_numpy(td_stack).to(device=device, dtype=torch.float64)
                dist.all_reduce(td_hist_t, op=dist.ReduceOp.SUM)
                td_merged = td_hist_t.cpu().numpy()
                (
                    td_pt_p, td_pt_t, td_e_p, td_e_t, td_p_p, td_p_t,
                    td_k1_pt_p, td_k1_pt_t, td_k1_e_p, td_k1_e_t, td_k1_p_p, td_k1_p_t,
                ) = [td_merged[i] for i in range(12)]

            td_all_merged: dict[str, np.ndarray] = {}
            if collect_train_dist_epoch:
                td_all_merged = _gather_val_array_dict(
                    {
                        **{
                            key: _concat_np_chunks(chunks)
                            for key, chunks in td_all_chunks.items()
                        },
                        "class_index": _concat_np_chunks(td_all_class_chunks),
                    },
                    rank=rank,
                    world_size=world_size,
                )

            if is_rank0 and wandb_mod is not None and collect_train_dist_epoch:
                _td_bin_pt = _diagnostic_bin_edges("pt")
                _td_bin_eta = _diagnostic_bin_edges("eta")
                _td_bin_phi = _diagnostic_bin_edges("phi")
                try:
                    td_log: dict[str, Any] = {"epoch": float(epoch)}
                    if legacy_train_kinematics:
                        _td_suffix = "train: reward best-of-K vs truth (all batches)"
                        td_log.update({
                            "train_dist/pt": _val_overlay_kin_figure(
                                td_pt_t, td_pt_p, _td_bin_pt,
                                f"Neutrino pT [GeV] ({_td_suffix})",
                                pred_label="Pred (train)", xlabel="pT [GeV]",
                            ),
                            "train_dist/eta": _val_overlay_kin_figure(
                                td_e_t, td_e_p, _td_bin_eta,
                                f"Neutrino η ({_td_suffix})",
                                pred_label="Pred (train)", xlabel="η",
                            ),
                            "train_dist/phi": _val_overlay_kin_figure(
                                td_p_t, td_p_p, _td_bin_phi,
                                f"Neutrino φ ({_td_suffix})",
                                pred_label="Pred (train)", xlabel="φ [rad]",
                            ),
                            "train_dist_k1/pt": _val_overlay_kin_figure(
                                td_k1_pt_t, td_k1_pt_p, _td_bin_pt,
                                "Neutrino pT [GeV] (train: candidate 0 / K=1 proxy vs truth, all batches)",
                                pred_label="Pred (train K=1 proxy)", xlabel="pT [GeV]",
                            ),
                            "train_dist_k1/eta": _val_overlay_kin_figure(
                                td_k1_e_t, td_k1_e_p, _td_bin_eta,
                                "Neutrino η (train: candidate 0 / K=1 proxy vs truth, all batches)",
                                pred_label="Pred (train K=1 proxy)", xlabel="η",
                            ),
                            "train_dist_k1/phi": _val_overlay_kin_figure(
                                td_k1_p_t, td_k1_p_p, _td_bin_phi,
                                "Neutrino φ (train: candidate 0 / K=1 proxy vs truth, all batches)",
                                pred_label="Pred (train K=1 proxy)", xlabel="φ [rad]",
                            ),
                        })
                    available_td_jsd_features = _available_truth_pred_features(
                        td_all_merged,
                        td_all_feature_names,
                    )
                    for feature_name in available_td_jsd_features:
                        bin_edges = _generation_special_bin_edges(feature_name)
                        td_log[f"train_dist/jsd/current/{feature_name}"] = _array_histogram_jsd(
                            td_all_merged.get(
                                f"{feature_name}_truth",
                                np.array([], dtype=np.float64),
                            ),
                            td_all_merged.get(
                                f"{feature_name}_pred",
                                np.array([], dtype=np.float64),
                            ),
                            bin_edges=bin_edges,
                        )
                    for feature_name in _available_truth_pred_features(
                        td_all_merged,
                        td_all_feature_names,
                    ):
                        td_log[f"train_dist/all/{feature_name}_truth_vs_pred"] = _truth_pred_matrix_figure(
                            td_all_merged.get(f"{feature_name}_truth", np.array([], dtype=np.float64)),
                            td_all_merged.get(f"{feature_name}_pred", np.array([], dtype=np.float64)),
                            xlabel=f"Truth {feature_name}",
                            ylabel=f"Pred {feature_name}",
                            title=f"Train 2D truth vs pred {feature_name} (all candidates, all batches)",
                            bin_edges=_generation_special_bin_edges(feature_name),
                        )
                        for metric_name, metric_value in _truth_pred_scalar_metrics(
                            td_all_merged.get(f"{feature_name}_truth", np.array([], dtype=np.float64)),
                            td_all_merged.get(f"{feature_name}_pred", np.array([], dtype=np.float64)),
                        ).items():
                            td_log[f"train_dist/all_metrics/{feature_name}/{metric_name}"] = metric_value
                    log_representative_classes = bool(
                        representative_class_indices
                        and (
                            int(epoch) == 0
                            or (int(epoch) + 1) % train_dist_by_class_every == 0
                        )
                    )
                    if log_representative_classes:
                        class_index = np.asarray(
                            td_all_merged.get(
                                "class_index",
                                np.array([], dtype=np.int64),
                            ),
                            dtype=np.int64,
                        ).reshape(-1)
                        for class_name, class_id in representative_class_indices.items():
                            class_mask = class_index == int(class_id)
                            if not np.any(class_mask):
                                continue
                            for feature_name in _available_truth_pred_features(
                                td_all_merged,
                                td_all_feature_names,
                            ):
                                truth_class, pred_class = select_truth_pred_by_class(
                                    td_all_merged[f"{feature_name}_truth"],
                                    td_all_merged[f"{feature_name}_pred"],
                                    class_index,
                                    class_id=int(class_id),
                                )
                                td_log[
                                    f"train_dist/by_class/{class_name}/"
                                    f"{feature_name}_truth_vs_pred"
                                ] = _truth_pred_matrix_figure(
                                    truth_class,
                                    pred_class,
                                    xlabel=f"Truth {feature_name}",
                                    ylabel=f"Pred {feature_name}",
                                    title=(
                                        f"Train 2D truth vs pred {feature_name} "
                                        f"({class_name}, all K candidates)"
                                    ),
                                    bin_edges=_generation_special_bin_edges(
                                        feature_name
                                    ),
                                )
                                for metric_name, metric_value in _truth_pred_scalar_metrics(
                                    truth_class,
                                    pred_class,
                                ).items():
                                    td_log[
                                        f"train_dist/by_class_metrics/{class_name}/"
                                        f"{feature_name}/{metric_name}"
                                    ] = metric_value
                    if len(td_log) > 1:
                        _wandb_log_with_step(
                            wandb_mod,
                            td_log,
                            step=_wandb_epoch_end_step(global_step),
                        )
                except Exception as _e:
                    _log.warning("[DGPO] train_dist figures failed at epoch=%s: %s", epoch, _e)

            ve = int(val_events) if val_events is not None else 0
            validation_tier = validation_schedule_tier(
                epoch,
                cheap_every_n_epochs=validation_every_n_epochs,
                full_every_n_epochs=validation_full_every_n_epochs,
            )
            run_epoch_val = (
                ve > 0
                and val_shard is not None
                and validation_tier is not None
            )
            if run_epoch_val:
                full_validation = validation_tier == "full"
                active_val_k = val_K if full_validation else val_cheap_K
                active_max_batches = (
                    val_max_batches if full_validation else val_cheap_max_batches
                )
                active_prefix = "val" if full_validation else "val_cheap"
                active_compute_winrate = bool(
                    full_validation
                    and dg.get("validation_compute_winrate", False)
                )
                if is_rank0:
                    _log.info(
                        "[DGPO] %s validation: requesting iterator (K=%s, max_batches=%s).",
                        validation_tier,
                        active_val_k,
                        active_max_batches,
                    )
                val_loader = val_shard.iter_torch_batches(**val_loader_cfg)
                est_val_batches = (
                    max(1, math.ceil(ve / validation_effective_batch))
                    if ve > 0
                    else None
                )
                val_metrics = run_validation_epoch(
                    model,
                    ref_model,
                    ema_save,
                    val_loader,
                    sampler,
                    reward_agg,
                    val_K=active_val_k,
                    num_ddim_steps=num_ddim_val,
                    device=device,
                    dtype=dtype,
                    cartesian=_truth_generation_cartesian(),
                    compute_winrate=active_compute_winrate,
                    epoch=epoch,
                    est_total_batches=est_val_batches,
                    val_log_batches=bool(
                        dg.get(
                            "validation_log_batches"
                            if full_validation
                            else "validation_cheap_log_batches",
                            full_validation,
                        )
                    ),
                    val_rollout_parallel_chains=min(
                        val_rollout_parallel_chains,
                        active_val_k,
                    ),
                    val_tqdm_k_chains=bool(
                        full_validation
                        and dg.get("validation_tqdm_k_chains", True)
                    ),
                    val_tqdm_ddim=bool(
                        full_validation
                        and dg.get("validation_tqdm_ddim", False)
                    ),
                    max_batches=active_max_batches,
                    initial_state=(val_baseline_state if full_validation else None),
                    full_diagnostics=full_validation,
                    metric_prefix=active_prefix,
                    rank=rank,
                    world_size=world_size,
                )
                if is_rank0:
                    if full_validation:
                        _append_validation_history_plots(val_metrics, epoch_value=epoch)
                    _log.info(
                        "[DGPO] %s val epoch=%s r_mean=%.6f r_med=%.6f p10=%.6f p90=%.6f",
                        validation_tier,
                        epoch,
                        val_metrics[f"{active_prefix}/reward/mean"],
                        val_metrics[f"{active_prefix}/reward/median"],
                        val_metrics[f"{active_prefix}/reward/p10"],
                        val_metrics[f"{active_prefix}/reward/p90"],
                    )
                    if wandb_mod is not None:
                        _wandb_log_validation(
                            wandb_mod,
                            val_metrics,
                            epoch=epoch,
                            wandb_step=_wandb_epoch_end_step(global_step),
                        )
                    if (
                        full_validation
                        and ckpt_topk is not None
                        and top_k_metric == "val/reward/mean"
                    ):
                        ckpt_topk.maybe_save(
                            score=val_metrics["val/reward/mean"],
                            last_completed_epoch=epoch,
                            dgpo_next_epoch=epoch + 1,
                            global_step=global_step,
                            model=model,
                            ema_save=ema_save,
                            optimizer=optimizer,
                            ref_model=ref_model,
                            ema_rollout=ema_rollout,
                            round_ref_model=(
                                round_ref_model if adaptive_cfg.enabled else None
                            ),
                            reward_round_id=int(adaptive_state.reward_round_id),
                            dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                            dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                            dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                            dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
                        )

            _barrier()

            adaptive_cycle_ran = False
            adaptive_stop_requested = False
            if adaptive_cfg.enabled:
                if trust_rejection_endpoint_requested or should_probe_training_boundary(
                    adaptive_state, cfg=adaptive_cfg, epoch=epoch,
                    global_step=global_step, epoch_end=True,
                ):
                    adaptive_cycle_ran = True
                    adaptive_metrics = _run_adaptive_cycle(
                        epoch=epoch,
                        extragradient_batch=(
                            batch_d
                            if adaptive_cfg.trust_extragradient_enabled
                            else None
                        ),
                        force_gradient_conflict=trust_rejection_endpoint_requested,
                    )
                    adaptive_stop_requested = bool(
                        float(
                            adaptive_metrics.get(
                                "reference_trust/round_acceptance/stop_requested",
                                0.0,
                            )
                        )
                        >= 0.5
                    ) or adaptive_state.raw_global_stop_requested
                    adaptive_stop_requested = bool(
                        adaptive_stop_requested
                        or trust_rejection_endpoint_requested
                    )
                    if trust_rejection_endpoint_requested:
                        adaptive_metrics.update(
                            {
                                "reference_trust/endpoint/at_rejection": 1.0,
                                "reference_trust/endpoint/proposal_step": float(
                                    global_step
                                ),
                                "reference_trust/endpoint/accepted_updates": float(
                                    adaptive_state.trust_accepted_updates
                                ),
                            }
                        )
                    if adaptive_cfg.trust_boundary_enabled:
                        adaptive_metrics[
                            "reference_trust/boundary_count"
                        ] = float(adaptive_state.trust_boundary_count)
                    if is_rank0:
                        if wandb_mod is not None:
                            _wandb_log_with_step(
                                wandb_mod,
                                adaptive_metrics,
                                step=_wandb_epoch_end_step(global_step),
                            )
                        if (
                            ckpt_topk is not None
                            and top_k_metric
                            == "staleness/audit_balanced_accuracy"
                        ):
                            audit_saturated = (
                                float(
                                    adaptive_metrics.get(
                                        "staleness/audit_saturated",
                                        0.0,
                                    )
                                )
                                >= 0.5
                            )
                            if audit_saturated:
                                ckpt_topk.maybe_save(
                                    score=float(
                                        adaptive_metrics.get(
                                            "staleness/audit_balanced_accuracy",
                                            float("nan"),
                                        )
                                    ),
                                    last_completed_epoch=epoch,
                                    dgpo_next_epoch=epoch + 1,
                                    global_step=global_step,
                                    model=model,
                                    ema_save=ema_save,
                                    optimizer=optimizer,
                                    ref_model=ref_model,
                                    ema_rollout=ema_rollout,
                                    round_ref_model=round_ref_model,
                                    reward_round_id=int(
                                        adaptive_state.reward_round_id
                                    ),
                                    dgpo_projection_constraint_state=(
                                        constraint_ckpt_payload_for_save()
                                    ),
                                    dgpo_omnifold_reward_metadata=(
                                        reward_checkpoint_metadata
                                    ),
                                    dgpo_adaptive_omnifold_state=(
                                        _adaptive_state_payload()
                                    ),
                                    dgpo_omnifold_reward_stack=(
                                        _adaptive_stack_payload()
                                    ),
                                )
                            else:
                                _log.info(
                                    "[DGPO] audit not saturated at epoch=%s; "
                                    "skipping audit-accuracy top-k checkpoint.",
                                    epoch,
                                )
                    _barrier()

            checkpoint_due = bool(
                (ckpt_every_n_steps > 0 and global_step % ckpt_every_n_steps == 0)
                or (
                    ckpt_every_n_epochs > 0
                    and (epoch + 1) % ckpt_every_n_epochs == 0
                )
                or adaptive_cycle_ran
                or (epochs is not None and epoch + 1 >= epochs)
            )
            if checkpoint_due:
                if is_rank0:
                    _dgpo_save_last_ckpt(
                        model,
                        ema_save,
                        optimizer,
                        ref_model,
                        last_completed_epoch=epoch,
                        dgpo_next_epoch=epoch + 1,
                        global_step=global_step,
                        ema_rollout=ema_rollout,
                        round_ref_model=(
                            round_ref_model if adaptive_cfg.enabled else None
                        ),
                        reward_round_id=int(adaptive_state.reward_round_id),
                        dgpo_projection_constraint_state=constraint_ckpt_payload_for_save(),
                        dgpo_omnifold_reward_metadata=reward_checkpoint_metadata,
                        dgpo_adaptive_omnifold_state=_adaptive_state_payload(),
                        dgpo_omnifold_reward_stack=_adaptive_stack_payload(),
                    )
                _barrier()

            if adaptive_stop_requested:
                adaptive_early_stop = True
                if is_rank0:
                    if trust_rejection_endpoint_requested:
                        _log.info(
                            "[DGPO/trust] boundary endpoint saved after proposal "
                            "step=%s (%s accepted update(s)).",
                            global_step,
                            adaptive_state.trust_accepted_updates,
                        )
                    elif adaptive_state.raw_global_stop_requested:
                        _log.info("[DGPO/global-best] stagnation: saved terminal state after %s failed refit rounds; best checkpoint=%s",
                                  adaptive_state.raw_global_failed_rounds, adaptive_state.raw_best_checkpoint)
                    elif bool(
                        float(
                            adaptive_metrics.get(
                                "reference_trust/signed_probe/recovery_stop_requested",
                                0.0,
                            )
                        )
                        >= 0.5
                    ):
                        _log.info(
                            "[DGPO/trust/signed] repeated reverse-only "
                            "directions reached patience; saved the restored "
                            "incumbent and stopped at epoch=%s global_step=%s.",
                            epoch,
                            global_step,
                        )
                    else:
                        _log.info(
                            "[DGPO/trust] statistically confirmed round-AUC "
                            "plateau; saved the rolled-back anchor checkpoint "
                            "and stopped at epoch=%s global_step=%s.",
                            epoch,
                            global_step,
                        )
                break

        if is_rank0:
            if adaptive_state.raw_global_stop_requested:
                _log.info("[DGPO] stopped for global-best stagnation, not convergence; epoch=%s step=%s", epoch, global_step)
            elif adaptive_early_stop:
                _log.info(
                    "[DGPO] converged early after epoch=%s (%s optimizer steps).",
                    epoch,
                    global_step,
                )
            else:
                _log.info(
                    "[DGPO] finished %s epochs (%s optimizer steps).",
                    epochs,
                    global_step,
                )
    finally:
        _finish_wandb_run(wandb_active)


def main() -> None:
    """CLI entry point: build a Ray ``TorchTrainer`` and ``fit()`` across the cluster."""
    p = argparse.ArgumentParser(description="DGPO neutrino RL (Ray Train + DGPO loop)")
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "config.yaml",
        help="YAML config (same merge rules as EveNet training)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (smoke test)",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides config)",
    )
    p.add_argument(
        "--ray-dir",
        type=str,
        default="~/ray_results",
        help="Ray Train RunConfig.storage_path",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    global_config.load_yaml(config_path)
    _assert_rl_enabled()
    global_config.display()
    platform_info = global_config.platform
    # Leave Ray's CUDA masking opt-out absent by default. Ray treats a present,
    # non-empty value as enabled, so exporting the string ``0`` is not a safe
    # false value. Explicit truthy values remain available for the legacy
    # Shifter/NCCL workaround.
    ray_cuda_opt_out_key = "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
    ray_cuda_opt_out = os.environ.get(ray_cuda_opt_out_key)
    if ray_cuda_opt_out is not None and ray_cuda_opt_out.strip().lower() in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        os.environ.pop(ray_cuda_opt_out_key, None)
        ray_cuda_opt_out = None

    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{_REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
            "TORCH_NCCL_TIMEOUT": "180",
        },
    }
    if ray_cuda_opt_out is not None:
        runtime_env["env_vars"][ray_cuda_opt_out_key] = ray_cuda_opt_out
    if "WANDB_API_KEY" in os.environ:
        runtime_env["env_vars"]["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    # Ray workers do not automatically inherit driver-side diagnostic flags.
    from RL.DGPO_neutrino.omnifold_ztautau.attention_diagnostic import ENV_KEYS
    for key in ENV_KEYS:
        if key in os.environ:
            runtime_env["env_vars"][key] = os.environ[key]

    # ``address="auto"`` forces a connection to the Ray cluster already started by
    # ``NERSC/start-head.sh`` / ``start-worker.sh`` instead of silently spinning up a
    # fresh single-node cluster on the head.  ``RAY_ADDRESS`` (set by the sbatch helper)
    # takes precedence when present.  Outside Slurm we fall back to a local cluster.
    ray_addr_env = os.environ.get("RAY_ADDRESS")
    try:
        ray.init(
            address=ray_addr_env or "auto",
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    except (ConnectionError, ValueError) as ex:
        _log.warning(
            "[DGPO][launch] No existing Ray cluster (%s); falling back to local init.",
            ex,
        )
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    # Wait for the expected number of Ray workers to join.  Worker srun in the sbatch
    # script typically needs 30-90s on NERSC; without this wait, ``trainer.fit()`` may
    # see only the head node and silently run on 1 node.
    expected_workers = int(platform_info.number_of_workers)
    expected_gpus_per_worker = float(dict(platform_info.resources_per_worker).get("GPU", 1))
    expected_gpus = float(expected_workers) * expected_gpus_per_worker
    wait_timeout_s = float(os.environ.get("DGPO_RAY_WAIT_S", "300"))
    poll_every = 5.0
    waited = 0.0
    while waited < wait_timeout_s:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        cur_nodes = len(ray.nodes())
        if cur_gpus >= expected_gpus:
            _log.info(
                "[DGPO][launch] Ray cluster ready: nodes=%s GPUs=%s (expected %s).",
                cur_nodes, cur_gpus, expected_gpus,
            )
            break
        _log.info(
            "[DGPO][launch] waiting for Ray workers... nodes=%s GPUs=%s/%s (%.0fs/%.0fs)",
            cur_nodes, cur_gpus, expected_gpus, waited, wait_timeout_s,
        )
        time.sleep(poll_every)
        waited += poll_every
    else:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        _log.warning(
            "[DGPO][launch] Timed out after %.0fs waiting for cluster: GPUs=%s (expected %s). "
            "Continuing — Ray Train may run with fewer workers or hang.",
            wait_timeout_s, cur_gpus, expected_gpus,
        )

    base_dir = Path(platform_info.data_parquet_dir)
    base_val_dir = (
        Path(platform_info.data_parquet_val_dir)
        if "data_parquet_val_dir" in platform_info
        else None
    )
    process_fn = make_process_fn(base_dir)
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
    launch_adaptive_cfg = resolve_adaptive_config(
        global_config.dgpo, classifier_only=bool(_dgpo_cfg_get(
            getattr(global_config, "experiment", {}), "classifier_only", False,
        )),
    )
    if launch_adaptive_cfg.single_pool_train_validation:
        train_ds, val_ds, total_events, val_events = _prepare_single_pool_datasets(
            base_dir=base_dir, base_val_dir=base_val_dir, process_fn=process_fn,
            platform_info=platform_info, dataset_options=global_config.options.Dataset,
        )
    else:
        train_ds, val_ds, total_events, val_events = prepare_datasets(
            base_dir=base_dir,
            process_event_batch_partial=process_fn,
            platform_info=platform_info,
            load_all_in_ram=False,
            base_val_dir=base_val_dir,
            predict=False,
        )

    datasets: dict[str, Any] = {"train": train_ds}
    separate_omnifold_source = bool(
        launch_adaptive_cfg.enabled
        and launch_adaptive_cfg.pool_data_parquet_dir is not None
    )
    fixed_omnifold_pool = bool(
        launch_adaptive_cfg.enabled
        and (
            separate_omnifold_source
            or launch_adaptive_cfg.pool_events is not None
        )
    )
    omnifold_train_events = int(total_events)
    if separate_omnifold_source:
        omnifold_base_dir = Path(
            str(launch_adaptive_cfg.pool_data_parquet_dir)
        )
        omnifold_parquet_files = sorted(
            map(str, omnifold_base_dir.glob("*.parquet"))
        )
        if not omnifold_parquet_files:
            raise ValueError(
                "No parquet files found in the dedicated OmniFold training "
                f"directory: {omnifold_base_dir}"
            )
        omnifold_process_fn = make_process_fn(omnifold_base_dir)
        omnifold_train_ds, omnifold_train_events = register_dataset(
            omnifold_parquet_files,
            omnifold_process_fn,
            platform_info,
            dataset_limit=1.0,
            file_shuffling=True,
        )
        if launch_adaptive_cfg.pool_events is not None:
            requested_pool_events = int(launch_adaptive_cfg.pool_events)
            omnifold_train_ds = omnifold_train_ds.random_shuffle(
                seed=int(launch_adaptive_cfg.pool_selection_seed)
            ).limit(requested_pool_events)
            omnifold_train_events = min(
                requested_pool_events, int(omnifold_train_events)
            )
        datasets["omnifold_train"] = omnifold_train_ds
        _log.info(
            "[DGPO/launch] dedicated OmniFold fit population: path=%s "
            "events=%s cap=%s seed=%s",
            omnifold_base_dir,
            omnifold_train_events,
            launch_adaptive_cfg.pool_events,
            launch_adaptive_cfg.pool_selection_seed,
        )
    elif fixed_omnifold_pool:
        # Choose the identities globally before Ray Train shards the dataset.
        # This immutable, seeded Dataset is reused by the bootstrap and every
        # adaptive refit; only the live-policy K=1 candidates are regenerated.
        datasets["omnifold_train"] = train_ds.random_shuffle(
            seed=int(launch_adaptive_cfg.pool_selection_seed)
        ).limit(int(launch_adaptive_cfg.pool_events))
        _log.info(
            "[DGPO/launch] fixed OmniFold fit population: events=%s seed=%s",
            min(int(launch_adaptive_cfg.pool_events), int(total_events)),
            launch_adaptive_cfg.pool_selection_seed,
        )
    if val_ds is not None and val_events:
        datasets["validation"] = val_ds

    scaling_config = ScalingConfig(
        num_workers=int(platform_info.number_of_workers),
        resources_per_worker=dict(platform_info.resources_per_worker),
        use_gpu=bool(platform_info.get("use_gpu", True)),
    )
    run_config = RunConfig(
        name="DGPO-Training",
        storage_path=args.ray_dir,
    )

    # Driver-side launch banner: visible on the head node before any worker spawns,
    # so a wrong cluster size is caught before the data pipeline starts.
    try:
        cluster_resources = ray.cluster_resources()
    except Exception:
        cluster_resources = {}
    _log.info(
        "[DGPO][launch] num_workers=%s resources_per_worker=%s use_gpu=%s "
        "cluster_GPUs=%s cluster_CPUs=%s nodes=%s",
        scaling_config.num_workers,
        scaling_config.resources_per_worker,
        scaling_config.use_gpu,
        cluster_resources.get("GPU"),
        cluster_resources.get("CPU"),
        len(ray.nodes()) if ray.is_initialized() else "?",
    )
    trainer_config = {
        "config_path": str(config_path),
        "config_yaml": config_path.read_text(),
        "max_steps": args.max_steps,
        "wandb": not args.no_wandb,
        "total_events": int(total_events),
        "val_events": int(val_events) if val_events else 0,
        "omnifold_train_events": int(omnifold_train_events),
        "fixed_omnifold_pool": fixed_omnifold_pool,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=dgpo_train_loop,
        train_loop_config=trainer_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets=datasets,
    )
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
