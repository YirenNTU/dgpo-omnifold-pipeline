"""Independent training for the DGPO latent constraint autoencoder.

Trains the single supported model — the **object-token bottleneck AE**
(``ObjectTokenBottleneckAutoencoder``): inputs are the frozen EveNet event CLS
token + all per-object ObjectEncoder tokens + the two-neutrino kinematics; the
target is the original pretrain-model event token and the neutrino kinematics.

Mirrors the EveNet / DGPO distributed style: Ray Train ``TorchTrainer`` spawns
one worker per rank, each pulls its own Ray Data shard, DDP is installed via
``ray.train.torch.prepare_model``, and only rank 0 writes checkpoints.  Works
single-GPU and CPU (``num_workers: 1``, ``use_gpu: false``) for debugging.

Run from the repo root:

    python RL/DGPO_neutrino/latent_constraint/train_latent_constraint.py \
        RL/DGPO_neutrino/latent_constraint/config.yaml

The trained checkpoint is loadable inside DGPO via
``RL.DGPO_neutrino.latent_constraint.object_token_ae.load_checkpoint``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ray  # noqa: E402
import ray.train  # noqa: E402
import ray.train.torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from ray.train import RunConfig, ScalingConfig  # noqa: E402
from ray.train.torch import TorchTrainer  # noqa: E402
from transformers import get_cosine_schedule_with_warmup  # noqa: E402

from evenet.control.global_config import global_config  # noqa: E402
from evenet.shared import make_process_fn, prepare_datasets  # noqa: E402
from RL.DGPO_neutrino.latent_constraint.object_token_ae import (  # noqa: E402
    ObjectTokenBottleneckAutoencoder,
    parse_lc_resume_from_checkpoint,
    save_checkpoint,
)
from RL.DGPO_neutrino.latent_constraint.sliced_wasserstein import (  # noqa: E402
    sliced_wasserstein_distance,
)
from RL.DGPO_neutrino.latent_constraint.plots import ReconPlotState  # noqa: E402
from RL.DGPO_neutrino.latent_constraint.mass_diagnostics import MassDiagState  # noqa: E402
from RL.DGPO_neutrino.latent_constraint.normalizers import load_normalizers_from_pt  # noqa: E402

_log = logging.getLogger("latent_constraint.train")

_DEFAULT_SPHERICAL_FEATURES: tuple[str, ...] = ("log_pt", "eta", "phi")
_DEFAULT_CARTESIAN_FEATURES: tuple[str, ...] = ("px", "py", "pz")


class _LCForward(torch.nn.Module):
    """Thin DDP wrapper: its ``forward`` runs the reconstruction loss.

    Training MUST go through the DDP-wrapped module's ``forward`` so DDP's
    reducer is prepared each step (``prepare_for_backward``). Calling the
    unwrapped module directly and then ``loss.backward()`` leaves the reducer
    unprepared and deadlocks NCCL across ranks. Mirrors DGPO's ``_DGPODDPForward``.
    """

    def __init__(self, model: ObjectTokenBottleneckAutoencoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.model.reconstruction_loss(batch)


def _compute_token_stats(
    train_shard,
    loader_cfg: dict[str, Any],
    model: ObjectTokenBottleneckAutoencoder,
    device: torch.device,
    world_size: int,
    n_batches: int,
    rank: int,
) -> None:
    """One-time per-dim mean/std of ``event_token`` and ``object_token`` (first ``n_batches``).

    Rank-symmetric: every rank scans its own shard for the SAME number of
    batches, then sums (sum, sumsq, count) across ranks so all ranks install
    bit-identical buffers before DDP wrapping. Per-object stats are accumulated
    over VALID objects only (via ``x_mask``). Stats persist in the checkpoint
    (state_dict buffers), so resumed runs and the frozen DGPO encoder reuse them.
    """
    dim = int(model.token_dim)
    s = torch.zeros(dim, device=device, dtype=torch.float64)
    ss = torch.zeros(dim, device=device, dtype=torch.float64)
    cnt = torch.zeros(1, device=device, dtype=torch.float64)
    os_ = torch.zeros(dim, device=device, dtype=torch.float64)
    oss = torch.zeros(dim, device=device, dtype=torch.float64)
    ocnt = torch.zeros(1, device=device, dtype=torch.float64)
    it = iter(train_shard.iter_torch_batches(**loader_cfg))
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        tok = batch.get("event_token")
        if tok is None:
            raise KeyError(
                "[lc] the latent-constraint AE needs 'event_token' in the batch: point "
                "platform.data_parquet_dir at the *_evttok augmented parquet "
                "(preprocessing/augment_event_token.py)."
            )
        tok = tok.to(device=device, dtype=torch.float64)
        s += tok.sum(dim=0)
        ss += (tok ** 2).sum(dim=0)
        cnt += float(tok.shape[0])
        obj = batch.get("object_token")
        if obj is None:
            raise KeyError(
                "[lc] the latent-constraint AE needs 'object_token' in the batch: "
                "point platform.data_parquet_dir at the *_evttok augmented parquet "
                "built with preprocessing/augment_event_token.py --object-tokens."
            )
        obj = obj.to(device=device, dtype=torch.float64)          # (B, P, D) or packed (B, P*D)
        if obj.dim() == 2:                                         # packed -> (B, P, D)
            obj = obj.reshape(obj.shape[0], -1, dim)
        xm = batch.get("x_mask")
        m = (xm.squeeze(-1) if xm.dim() == 3 else xm).to(device=device)  # (B, P)
        m = (m > 0).to(torch.float64).unsqueeze(-1)               # (B, P, 1)
        os_ += (obj * m).sum(dim=(0, 1))
        oss += ((obj ** 2) * m).sum(dim=(0, 1))
        ocnt += float(m.sum().item())
    if world_size > 1:
        import torch.distributed as dist
        for t in (s, ss, cnt, os_, oss, ocnt):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    n = float(cnt.item())
    if n < 2:
        raise RuntimeError("[lc] token-stats pass saw <2 events; check the train shard")
    mean = s / n
    var = (ss / n - mean ** 2).clamp_min(0.0)
    model.set_token_stats(mean.float(), var.sqrt().float())
    _log.info(
        "[lc][rank=%s] event_token stats from %d events: mean_rms=%.4f std_mean=%.4f",
        rank, int(n),
        float(mean.float().pow(2).mean().sqrt()),
        float(var.sqrt().float().mean()),
    )
    no = float(ocnt.item())
    if no < 2:
        raise RuntimeError("[lc] object-token stats pass saw <2 valid objects")
    omean = os_ / no
    ovar = (oss / no - omean ** 2).clamp_min(0.0)
    model.set_object_token_stats(omean.float(), ovar.sqrt().float())
    _log.info(
        "[lc][rank=%s] object_token stats from %d valid objects: "
        "mean_rms=%.4f std_mean=%.4f",
        rank, int(no),
        float(omean.float().pow(2).mean().sqrt()),
        float(ovar.sqrt().float().mean()),
    )


def _dbg(enabled: bool, rank: int, world_size: int, msg: str) -> None:
    """Per-rank debug line (flushed via logging) to trace where a hang occurs."""
    if enabled:
        _log.info("[lc][dbg][rank=%s/%s] %s", rank, world_size, msg)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Move a Ray ``iter_torch_batches`` dict onto ``device`` (tensors only)."""
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
    return out


def _validate_truth_batch(
    batch: dict[str, torch.Tensor],
    *,
    rank: int,
    split: str,
    nu_kin_dim: int,
    invisible_key: str = "x_invisible",
    cartesian: bool = False,
) -> None:
    """Fail early if the AD is not training on truth neutrino kinematics."""
    feature_label = "(px,py,pz)" if cartesian else "(log_pt,eta,phi)"
    if invisible_key not in batch:
        raise KeyError(
            f"[lc][truth-check][rank={rank}] {split} batch missing {invisible_key}; "
            "latent-constraint training requires TruthGeneration.include: true"
            + (
                " and a parquet that contains x_invisible_cartesian for cartesian training"
                if cartesian
                else ""
            )
        )
    if "x_invisible_mask" not in batch:
        raise KeyError(
            f"[lc][truth-check][rank={rank}] {split} batch missing x_invisible_mask; "
            "cannot safely mask the two-neutrino regression loss"
        )

    x_inv = batch[invisible_key]
    x_mask = batch["x_invisible_mask"]
    if x_inv.dim() != 3 or x_inv.shape[1] != 2:
        raise ValueError(
            f"[lc][truth-check][rank={rank}] {split} {invisible_key} must be (B, 2, F), "
            f"got {tuple(x_inv.shape)}"
        )
    if int(x_inv.shape[-1]) != int(nu_kin_dim):
        raise ValueError(
            f"[lc][truth-check][rank={rank}] {split} expected {invisible_key} last dim "
            f"{nu_kin_dim} = {feature_label}, got {x_inv.shape[-1]}. "
            "Check TruthGeneration.cartesian and normalization match."
        )
    if x_mask.shape[:2] != x_inv.shape[:2]:
        raise ValueError(
            f"[lc][truth-check][rank={rank}] {split} x_invisible_mask shape "
            f"{tuple(x_mask.shape)} does not match {invisible_key} slots {tuple(x_inv.shape[:2])}"
        )
    if not torch.isfinite(x_inv).all():
        raise ValueError(f"[lc][truth-check][rank={rank}] {split} {invisible_key} has NaN/Inf")

    valid_slots = float(x_mask.float().sum().detach().cpu())
    mean = x_inv.detach().float().mean(dim=(0, 1)).cpu().tolist()
    std = x_inv.detach().float().std(dim=(0, 1), unbiased=False).cpu().tolist()
    _log.info(
        "[lc][truth-check][rank=%s] %s %s truth shape=%s mask_valid_slots=%.0f "
        "features=" + feature_label + " mean=%s std=%s",
        rank,
        split,
        invisible_key,
        tuple(x_inv.shape),
        valid_slots,
        [round(float(v), 4) for v in mean],
        [round(float(v), 4) for v in std],
    )


def _resolve_lc_cfg() -> dict[str, Any]:
    """Read the ``latent_constraint`` block from the merged config (with defaults)."""
    lc = global_config.get("latent_constraint", {}) or {}
    model_cfg = lc.get("model", {}) or {}
    train_cfg = lc.get("train", {}) or {}
    es_cfg = train_cfg.get("early_stopping", {}) or {}
    monitor = str(es_cfg.get("monitor", "val_loss")).strip()
    if monitor not in ("val_loss", "swd_separation"):
        raise ValueError(
            "latent_constraint.train.early_stopping.monitor must be "
            f"'val_loss' or 'swd_separation', got {monitor!r}"
        )
    mtype = str(model_cfg.get("type", "")).strip()
    if mtype not in ("", "object_token_bottleneck_ae"):
        raise ValueError(
            "latent_constraint.model.type must be object_token_bottleneck_ae "
            f"(the only supported model), got {mtype!r}"
        )
    feature_names_raw = model_cfg.get("feature_names", None)
    feature_names = tuple(str(name) for name in feature_names_raw) if feature_names_raw else ()
    return {
        "nu_kin_dim": int(model_cfg.get("nu_kin_dim", 0)),
        "feature_names": feature_names,
        "d_model": int(model_cfg.get("d_model", 64)),
        "latent_dim": int(model_cfg.get("latent_dim", 32)),
        "num_layers": int(model_cfg.get("num_layers", 3)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "ffn_mult": int(model_cfg.get("ffn_mult", 2)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "phi_index": model_cfg.get("phi_index", None),
        # event_token / object_token columns from augment_event_token.py
        "token_dim": int(model_cfg.get("token_dim", 256)),
        "token_stats_batches": max(1, int(model_cfg.get("token_stats_batches", 16))),
        "lr": float(train_cfg.get("lr", 3.0e-4)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.05)),
        "grad_clip_norm": float(train_cfg.get("grad_clip_norm", 1.0)),
        "log_every": max(1, int(train_cfg.get("log_every", 1))),
        "early_stopping_enabled": bool(es_cfg.get("enabled", False)),
        "early_stopping_patience": max(1, int(es_cfg.get("patience", 15))),
        "early_stopping_min_delta": float(es_cfg.get("min_delta", 0.0)),
        "early_stopping_monitor": monitor,
        "feature_shift_diagnostics_pct": [
            float(s) for s in (train_cfg.get("feature_shift_diagnostics_pct", [5.0, 10.0]) or [])
        ],
        # Monitoring plots (truth-vs-recon histograms) logged to W&B as images,
        # mirroring the main model's generation plots. Gated to every N epochs
        # (0 disables); only the first ``plot_max_batches`` val batches feed them.
        "plot_every_n_epochs": int(train_cfg.get("plot_every_n_epochs", 5)),
        "plot_max_batches": max(1, int(train_cfg.get("plot_max_batches", 16))),
    }


def _infer_feature_layout(
    *,
    lc: dict[str, Any],
    event_info: Any,
    normalization_file: str | Path,
    cartesian: bool,
) -> tuple[tuple[str, ...], tuple[bool, ...], tuple[bool, ...], int, int | None]:
    """Resolve invisible feature metadata from event_info first, then validate against normalization."""
    _, _, inv_norm = load_normalizers_from_pt(normalization_file, cartesian=cartesian)
    n_inv = int(inv_norm.mean.numel())

    feature_names_cfg = tuple(lc.get("feature_names", ()) or ())
    if cartesian:
        feature_names = feature_names_cfg or _DEFAULT_CARTESIAN_FEATURES[:n_inv]
        log_scaled = tuple(False for _ in range(len(feature_names)))
        uniform = tuple(False for _ in range(len(feature_names)))
    else:
        event_feature_names = tuple(str(name) for name in event_info.invisible_feature_names)
        if feature_names_cfg and feature_names_cfg != event_feature_names:
            raise ValueError(
                "latent_constraint.model.feature_names does not match event_info invisible feature names: "
                f"cfg={feature_names_cfg} event_info={event_feature_names}"
            )
        feature_names = event_feature_names
        log_scaled = tuple(bool(feature.log_scale) for feature in event_info.invisible_input_features)
        uniform = tuple(bool(feature.uniform) for feature in event_info.invisible_input_features)

    if len(feature_names) != n_inv:
        raise ValueError(
            f"invisible feature count {len(feature_names)} != normalization invisible dim {n_inv}"
        )

    nu_kin_dim_cfg = int(lc["nu_kin_dim"])
    if nu_kin_dim_cfg != n_inv:
        _log.warning(
            "[lc] overriding latent_constraint.model.nu_kin_dim=%s with normalization invisible dim=%s",
            nu_kin_dim_cfg,
            n_inv,
        )
    nu_kin_dim = n_inv

    phi_index_cfg = lc.get("phi_index", None)
    if cartesian:
        phi_index = None
    elif phi_index_cfg is not None:
        phi_index = int(phi_index_cfg)
    elif getattr(event_info, "invisible_inv_cdf_index", None):
        indices = tuple(int(index) for index in event_info.invisible_inv_cdf_index)
        phi_index = indices[0] if len(indices) == 1 else None
    elif "phi" in feature_names:
        phi_index = feature_names.index("phi")
    else:
        phi_index = None

    return feature_names, log_scaled, uniform, nu_kin_dim, phi_index


def _early_stopping_mode(monitor: str) -> str:
    """``min`` for losses, ``max`` for separation metrics."""
    return "max" if monitor == "swd_separation" else "min"


def _early_stopping_improved(
    current: float,
    best: float,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    """True when ``current`` beats ``best`` by at least ``min_delta``."""
    if not math.isfinite(current):
        return False
    if not math.isfinite(best):
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


def _early_stopping_metric_value(
    monitor: str,
    *,
    val_loss: float,
    sep: dict[str, float],
) -> float:
    if monitor == "swd_separation":
        return float(sep.get("swd_separation", float("nan")))
    return float(val_loss)


def _route_extra_metric_key(k: str) -> str:
    """W&B section routing for aggregated ``extra_reduced`` metrics.

    - ``res_mse/*``   -> ``residual/*``  (neutrino reconstruction residuals get their
      OWN clean W&B section, separate from the busy ``val/*``)
    - everything else -> ``val/*``
    """
    if k.startswith("res_mse/"):
        return f"residual/{k}"
    return f"val/{k}"


def _resolve_lc_checkpoint_path(explicit: str | Path | None = None) -> Path | None:
    """Explicit CLI/config path, else ``options.Training.model_checkpoint_load_path`` (EveNet/DGPO style)."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None
    raw = getattr(global_config.options.Training, "model_checkpoint_load_path", None)
    if not raw:
        return None
    p = Path(str(raw)).expanduser().resolve()
    return p if p.is_file() else None


def _maybe_init_wandb(enabled: bool, is_rank0: bool, run_config: dict):
    """Init a W&B run on rank 0 only. Returns the run handle or None.

    Mirrors DGPO's ``_start_wandb_run``: uses ``Settings(start_method="thread")``
    because Ray Train workers are forked subprocesses and the default ``fork``
    start method deadlocks ``wandb.init`` inside them (a classic "stuck at epoch
    0" symptom). Honors ``WANDB_DISABLED`` too.
    """
    if not (enabled and is_rank0):
        return None
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        _log.info("[lc] WANDB_DISABLED set; skipping wandb.")
        return None
    try:
        import wandb
    except ImportError:
        _log.warning("[lc] wandb not installed; skipping W&B logging.")
        return None
    wcfg = global_config.get("wandb", {}) or {}
    init_kw: dict[str, Any] = {
        "project": wcfg.get("project", "latent-constraint"),
        "entity": wcfg.get("entity"),
        "name": wcfg.get("run_name") or wcfg.get("name"),
        "id": wcfg.get("id"),
        "resume": wcfg.get("resume", "allow"),
        "tags": wcfg.get("tags"),
        "config": run_config,
    }
    try:
        init_kw["settings"] = wandb.Settings(start_method="thread")
    except Exception:
        pass
    try:
        run = wandb.init(**init_kw)
        _log.info("[lc] wandb.init project=%s name=%s", init_kw["project"], init_kw["name"])
        return run
    except Exception as ex:
        _log.warning("[lc] wandb.init failed (continuing without W&B): %s", ex)
        return None


@torch.no_grad()
def _latent_separation_diagnostics(
    z_truth: torch.Tensor,
    z_shuffled: torch.Tensor,
) -> dict[str, float]:
    """The key constraint-usefulness check, in latent space.

    - ``swd_null``: SWD between two size-n bootstrap resamples (with replacement) of the
      truth latents -- the same-distribution noise floor at the SAME sample size as
      ``swd_shuffled``. (Earlier this used disjoint n/2 halves, which -- because the 1-D
      SWD noise floor scales ~1/sqrt(m) -- inflated the null by ~sqrt(2) and pinned
      ``swd_separation`` at ~1/sqrt(2)~=0.71 even with zero real signal. The n-vs-n
      bootstrap matches the deploy-time ``SWD_tt`` in swd_ratio_constraint.)
    - ``swd_shuffled``: SWD between truth latents and latents of the same events
      paired with *other* events' neutrinos (a wrong-pairing distribution). Should
      grow as the latent learns to react to neutrino<->event mismatch.
    - ``swd_separation``: ``swd_shuffled / (swd_null + eps)`` -- now ~1.0 with no signal,
      >1 means the latent genuinely separates good from bad pairings.

    Fixed ``seed`` keeps projections identical across epochs so the curve is
    comparable epoch-to-epoch.
    """
    n = z_truth.shape[0]
    if n < 4:
        return {}
    gen = torch.Generator(device=z_truth.device).manual_seed(0)
    idx_a = torch.randint(0, n, (n,), device=z_truth.device, generator=gen)
    idx_b = torch.randint(0, n, (n,), device=z_truth.device, generator=gen)
    swd_null = float(sliced_wasserstein_distance(z_truth[idx_a], z_truth[idx_b], num_projections=128, seed=0))
    swd_shuffled = float(sliced_wasserstein_distance(z_truth, z_shuffled, num_projections=128, seed=0))
    return {
        "swd_null": swd_null,
        "swd_shuffled": swd_shuffled,
        "swd_separation": swd_shuffled / (swd_null + 1e-6),
    }


def _feature_shift_tag(feature_name: str, shift_pct: float) -> str:
    safe_name = feature_name.replace("/", "_")
    return f"{safe_name}_{shift_pct:g}pct"


def _wrap_periodic_feature(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _shift_feature_percent(
    x_invisible: torch.Tensor,
    *,
    feature_index: int,
    shift_pct: float,
    log_scaled: bool,
    periodic: bool,
) -> torch.Tensor:
    """Apply a multiplicative raw-feature shift to one invisible feature column.

    For ``log_*`` features, the shift is applied in raw space and then mapped
    back with ``log1p`` so the perturbation respects the feature definition.
    """
    out = x_invisible.clone()
    scale = 1.0 + float(shift_pct) / 100.0
    col = out[..., feature_index]
    if log_scaled:
        raw = torch.expm1(col)
        shifted = raw * scale
        col_shifted = torch.log1p(shifted.clamp_min(0.0))
    else:
        col_shifted = col * scale
    if periodic:
        col_shifted = _wrap_periodic_feature(col_shifted)
    out[..., feature_index] = col_shifted
    return out


@torch.no_grad()
def _feature_shift_diagnostics(
    z_truth: torch.Tensor,
    z_shifted: dict[str, torch.Tensor],
) -> dict[str, float]:
    """SWD between truth latents and feature-shifted truth latents."""
    out: dict[str, float] = {}
    for tag, z_shift in z_shifted.items():
        if z_shift.shape[0] != z_truth.shape[0] or z_truth.shape[0] < 1:
            continue
        out[f"swd_feature_shift_{tag}"] = float(
            sliced_wasserstein_distance(z_truth, z_shift, num_projections=128, seed=0)
        )
    return out


def _next_batch_synced(iterator, *, world_size: int, device: torch.device):
    """Pull the next batch with cross-rank termination sync (mirrors DGPO).

    Each rank reads its own Ray Data shard. To keep the per-step DDP gradient
    all-reduce in lock-step, we all-reduce a "has-more" flag with ``MIN``: the
    loop stops as soon as *any* rank's shard is exhausted. This may drop a few
    batches from longer shards but prevents NCCL hangs on uneven shards.

    Returns ``(batch_or_None, all_have)``.
    """
    try:
        batch = next(iterator)
        local_has = 1
    except StopIteration:
        batch, local_has = None, 0
    if world_size > 1:
        flag = torch.tensor([local_has], device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return batch, bool(flag.item() > 0)
    return batch, local_has > 0


def _all_reduce_mean(value: float, weight: float, device: torch.device, world_size: int) -> float:
    """Cross-rank weighted-mean reduction (single collective; identity on 1 rank)."""
    if world_size <= 1:
        return value if weight > 0 else 0.0
    t = torch.tensor([value * weight, weight], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_w = float(t[1].cpu())
    return float((t[0] / t[1]).cpu()) if total_w > 0 else 0.0


def _train_loop(cfg: dict[str, Any]) -> None:
    """Per-worker training loop launched by Ray Train."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ctx = ray.train.get_context()
    rank = int(ctx.get_world_rank())
    world_size = int(ctx.get_world_size())
    local_rank = int(ctx.get_local_rank())
    is_rank0 = rank == 0
    device = ray.train.torch.get_device()
    # Verbose per-rank tracing of dataset/collective/checkpoint milestones.
    # On by default while debugging the hang; set LC_DEBUG=0 to quiet it.
    debug = bool(int(os.environ.get("LC_DEBUG", "1")))
    _log.info(
        "[lc][boot] rank=%s/%s local_rank=%s node=%s device=%s debug=%s",
        rank, world_size, local_rank, os.uname().nodename, device, debug,
    )

    config_path = Path(str(cfg["config_path"])).resolve()
    global_config.load_yaml(config_path)
    lc = _resolve_lc_cfg()
    platform_info = global_config.platform
    normalization_file = global_config.options.Dataset.normalization_file
    truth_generation = global_config.options.Training.Components.TruthGeneration
    event_info = global_config.event_info
    if not bool(getattr(truth_generation, "include", False)):
        raise ValueError(
            "[lc] latent-constraint training needs truth neutrinos in the batch: "
            "set options.Training.Components.TruthGeneration.include: true"
        )
    cartesian = bool(getattr(truth_generation, "cartesian", False))
    invisible_key = "x_invisible_cartesian" if cartesian else "x_invisible"
    feature_names, feature_log_scaled, feature_uniform, resolved_nu_kin_dim, resolved_phi_index = _infer_feature_layout(
        lc=lc,
        event_info=event_info,
        normalization_file=normalization_file,
        cartesian=cartesian,
    )
    lc["feature_names"] = feature_names
    lc["nu_kin_dim"] = resolved_nu_kin_dim
    lc["phi_index"] = resolved_phi_index
    plot_feature_names = feature_names
    if is_rank0:
        if cartesian:
            _log.info(
                "[lc][truth-check] using TruthGeneration.include=true cartesian=true; "
                "%s is truth (px, py, pz), not model prediction.",
                invisible_key,
            )
        else:
            _log.info(
                "[lc][truth-check] using TruthGeneration.include=true cartesian=false; "
                "%s is truth with feature_names=%s, not model prediction.",
                invisible_key,
                feature_names,
            )
        _log.info(
            "[lc] latent feature layout: feature_names=%s log_scaled=%s uniform=%s nu_kin_dim=%s phi_index=%s cartesian=%s",
            feature_names,
            feature_log_scaled,
            feature_uniform,
            lc["nu_kin_dim"],
            lc["phi_index"],
            cartesian,
        )
    epochs = int(global_config.options.Training.epochs)
    total_events = int(cfg["total_events"])
    val_events = int(cfg.get("val_events", 0) or 0)
    max_steps: int | None = cfg.get("max_steps")
    wandb_flag = bool(cfg.get("wandb", True))
    # Cap how many val batches feed the (small) latent-separation diagnostic.
    swd_diag_max_batches = 16
    save_dir = Path(global_config.options.Training.model_checkpoint_save_path)

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("validation") if val_events else None
    batch_size = int(platform_info.batch_size)
    prefetch = int(getattr(platform_info, "prefetch_batches", 1))
    train_loader_cfg = {
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
        "local_shuffle_buffer_size": batch_size * prefetch,
    }
    val_loader_cfg = {"batch_size": batch_size, "prefetch_batches": prefetch}

    model = ObjectTokenBottleneckAutoencoder(
        normalization_file=normalization_file,
        token_dim=lc["token_dim"],
        nu_kin_dim=lc["nu_kin_dim"],
        d_model=lc["d_model"] if lc["d_model"] != 64 else None,  # default: token_dim
        latent_dim=lc["latent_dim"],
        num_layers=lc["num_layers"],
        num_heads=lc["num_heads"],
        ffn_mult=lc["ffn_mult"],
        dropout=lc["dropout"],
        cartesian=cartesian,
        phi_index=lc["phi_index"],
        device=device,
    )
    if is_rank0:
        _log.info(
            "[lc] ObjectTokenBottleneckAutoencoder: [z-cls, event_token, "
            "object_tokens(P), nu, antinu] self-attn -> bottleneck z(%s) -> "
            "decode [nu ; event_token]; loss = mse_nu + mse_token (no weight)",
            lc["latent_dim"],
        )

    ckpt_dict: dict[str, Any] | None = None
    ckpt_path = _resolve_lc_checkpoint_path(cfg.get("resume_checkpoint"))
    if ckpt_path is not None:
        ckpt_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt_dict["model_state_dict"])
        if is_rank0:
            _log.info("[lc] Loaded model weights from %s", ckpt_path)

    # Install event_token + object_token standardization once (fresh runs);
    # resumed runs restore the buffers from the checkpoint's state_dict.
    if ckpt_dict is None:
        _compute_token_stats(
            train_shard, val_loader_cfg, model, device, world_size,
            lc["token_stats_batches"], rank,
        )

    if is_rank0:
        n_params = sum(p.numel() for p in model.parameters())
        _log.info("[lc] model params=%.3fM  latent_dim=%s", n_params / 1e6, lc["latent_dim"])

    # Train through the DDP wrapper's forward (see _LCForward). All params are
    # used in reconstruction_loss, so find_unused_parameters=False is correct.
    fw = _LCForward(model)
    if world_size > 1:
        ddp_model = ray.train.torch.prepare_model(
            fw, parallel_strategy_kwargs={"find_unused_parameters": False}
        )
    else:
        ddp_model = fw
    core = (ddp_model.module if hasattr(ddp_model, "module") else ddp_model).model

    effective_batch = batch_size * world_size
    steps_per_epoch = max(1, math.ceil(total_events / effective_batch))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(lc["warmup_ratio"] * total_steps))
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(), lr=lc["lr"], weight_decay=lc["weight_decay"]
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch, global_step = parse_lc_resume_from_checkpoint(ckpt_dict)
    # Validation step counter: independent of global_step, continuous across
    # epochs and resume so train and val have separate monotonic W&B x-axes.
    val_step = int(ckpt_dict.get("val_step", 0)) if ckpt_dict is not None else 0
    if ckpt_dict is not None and "optimizer_state_dict" in ckpt_dict:
        try:
            optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            if is_rank0:
                _log.info("[lc] Restored optimizer state from checkpoint.")
        except (ValueError, RuntimeError) as ex:
            if is_rank0:
                _log.warning("[lc] Could not load optimizer state (fresh optimizer): %s", ex)
    if ckpt_dict is not None and "scheduler_state_dict" in ckpt_dict:
        try:
            scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
            if is_rank0:
                _log.info("[lc] Restored scheduler state from checkpoint.")
        except (ValueError, RuntimeError) as ex:
            if is_rank0:
                _log.warning("[lc] Could not load scheduler state (fresh schedule): %s", ex)

    log_every = lc["log_every"]
    wandb_run = _maybe_init_wandb(wandb_flag, is_rank0, {**lc, "epochs": epochs, "world_size": world_size})
    if wandb_run is not None:
        # Separate, continuous x-axes: train/* against train/step, val/* against
        # val/step. Each is logged in its own wandb.log call (no global step=).
        wandb_run.define_metric("train/step")
        wandb_run.define_metric("train/*", step_metric="train/step")
        wandb_run.define_metric("val/step")
        wandb_run.define_metric("val/*", step_metric="val/step")
        # Neutrino reconstruction residuals get their own clean W&B section, on the
        # val/step axis (they are computed + aggregated during validation).
        wandb_run.define_metric("residual/*", step_metric="val/step")

    if start_epoch > 0 or global_step > 0:
        _log.info(
            "[lc][rank=%s/%s] Resuming: start_epoch=%s global_step=%s (epochs in config=%s).",
            rank, world_size, start_epoch, global_step, epochs,
        )
    if start_epoch >= epochs:
        if is_rank0:
            _log.info(
                "[lc] start_epoch=%s >= epochs=%s; nothing to train.",
                start_epoch, epochs,
            )
        if wandb_run is not None:
            wandb_run.finish()
        return

    _log.info(
        "[lc][rank=%s/%s] device=%s batch=%s effective_batch=%s steps/epoch≈%s",
        rank, world_size, device, batch_size, effective_batch, steps_per_epoch,
    )

    best_val = float("inf")
    if ckpt_dict is not None and ckpt_dict.get("val_loss") is not None:
        best_val = float(ckpt_dict["val_loss"])

    es_enabled = bool(lc["early_stopping_enabled"]) and val_shard is not None
    es_monitor = lc["early_stopping_monitor"]
    es_mode = _early_stopping_mode(es_monitor)
    es_patience = int(lc["early_stopping_patience"])
    es_min_delta = float(lc["early_stopping_min_delta"])
    es_best = float("inf") if es_mode == "min" else float("-inf")
    es_wait = 0
    if ckpt_dict is not None:
        raw_es_best = ckpt_dict.get("early_stopping_best")
        if raw_es_best is not None and math.isfinite(float(raw_es_best)):
            es_best = float(raw_es_best)
        es_wait = max(0, int(ckpt_dict.get("early_stopping_wait", 0)))
    if es_enabled and is_rank0:
        _log.info(
            "[lc] early_stopping enabled: monitor=%s mode=%s patience=%s min_delta=%g",
            es_monitor, es_mode, es_patience, es_min_delta,
        )

    feature_shift_pcts: list[float] = list(lc["feature_shift_diagnostics_pct"])
    if feature_shift_pcts and is_rank0:
        _log.info(
            "[lc] feature-shift validation probes: feature_names=%s shifts=%s%%",
            feature_names,
            feature_shift_pcts,
        )

    stop_training = False
    for epoch in range(start_epoch, epochs):
        ddp_model.train()
        _dbg(debug, rank, world_size, f"epoch={epoch} creating train iterator")
        train_iter = iter(train_shard.iter_torch_batches(**train_loader_cfg))
        running, seen = 0.0, 0
        step_in_epoch = 0
        # Synced iteration keeps per-step DDP gradient all-reduces in lock-step
        # across ranks even when Ray Data shards have unequal length.
        while True:
            _dbg(debug and step_in_epoch < 3, rank, world_size,
                 f"epoch={epoch} fetching batch {step_in_epoch} (synced)")
            batch_cpu, all_have = _next_batch_synced(train_iter, world_size=world_size, device=device)
            if not all_have:
                _dbg(debug, rank, world_size, f"epoch={epoch} shard drained at step_in_epoch={step_in_epoch}")
                break
            batch = _batch_to_device(batch_cpu, device)
            if epoch == start_epoch and step_in_epoch == 0:
                _validate_truth_batch(batch, rank=rank, split="train", nu_kin_dim=lc["nu_kin_dim"],
                                      invisible_key=invisible_key, cartesian=cartesian)
            _dbg(debug and step_in_epoch < 3, rank, world_size,
                 f"epoch={epoch} forward+backward step {step_in_epoch} bs={int(batch['x'].shape[0])}")
            optimizer.zero_grad(set_to_none=True)
            # IMPORTANT: forward through the DDP wrapper so the reducer is prepared.
            loss, _ = ddp_model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), lc["grad_clip_norm"])
            optimizer.step()
            scheduler.step()
            _dbg(debug and step_in_epoch < 3, rank, world_size,
                 f"epoch={epoch} optimizer.step done step {step_in_epoch}")
            global_step += 1
            step_in_epoch += 1
            bs = int(batch["x"].shape[0])
            loss_f = float(loss.detach().cpu())
            running += loss_f * bs
            seen += bs
            if global_step % log_every == 0:
                lr_now = float(scheduler.get_last_lr()[0])
                _log.info(
                    "[lc][rank=%s/%s] epoch=%s global_step=%s batch=%s loss=%.6f lr=%.3e",
                    rank, world_size, epoch, global_step, bs, loss_f, lr_now,
                )
                if wandb_run is not None and is_rank0:
                    wandb_run.log(
                        {
                            "train/step": global_step,
                            "train/loss_step": loss_f,
                            "train/lr": lr_now,
                        }
                    )
            if max_steps is not None and global_step >= max_steps:
                break

        _dbg(debug, rank, world_size, f"epoch={epoch} train done steps={step_in_epoch}; all-reduce train_loss")
        train_loss = _all_reduce_mean(running / max(seen, 1), float(seen), device, world_size)
        _dbg(debug, rank, world_size, f"epoch={epoch} train_loss reduced={train_loss:.5f}")

        val_loss = float("nan")
        latent_rms = float("nan")
        sep: dict[str, float] = {}
        feature_shift_metrics: dict[str, float] = {}
        mass_scalars: dict[str, float] = {}
        # Extra per-model scalar metrics (rank-identical key list so all-reduce stays
        # symmetric): recon_nu_mse, recon_token_mse, res_mse/* residuals.
        extra_keys = list(core.metric_keys)
        extra_reduced = {k: float("nan") for k in extra_keys}
        # Deterministic, rank-identical feature-shift keys (from config) so the
        # all-reduce loop below issues the SAME collectives on every rank.
        feature_shift_keys = [
            f"swd_feature_shift_{_feature_shift_tag(feature_name, shift_pct)}"
            for feature_name in feature_names
            for shift_pct in feature_shift_pcts
        ]
        if val_shard is not None:
            _dbg(debug, rank, world_size, f"epoch={epoch} starting validation")
            ddp_model.eval()
            v_running, v_seen = 0.0, 0
            v_extra = {k: 0.0 for k in extra_keys}
            z_truth_chunks: list[torch.Tensor] = []
            z_shuf_chunks: list[torch.Tensor] = []
            z_shift_chunks: dict[str, list[torch.Tensor]] = {
                _feature_shift_tag(feature_name, shift_pct): []
                for feature_name in feature_names
                for shift_pct in feature_shift_pcts
            }
            n_diag_batches = 0
            v_batch_idx = 0
            # Monitoring plots: truth-vs-recon histograms (every plot_every_n_epochs).
            # Gated purely on epoch (rank-identical) so the all-reduce below is symmetric.
            plot_every = lc["plot_every_n_epochs"]
            plot_on = plot_every > 0 and (epoch % plot_every) == (plot_every - 1)
            plot_state = ReconPlotState(
                plot_feature_names,
                cartesian=cartesian,
                include_pt_overlay=(cartesian or feature_names == _DEFAULT_SPHERICAL_FEATURES),
            ) if plot_on else None
            n_plot_batches = 0
            # Physics diagnostics (every epoch): neutrino |p| + W/top mass, truth vs
            # AE-recon vs SHUFFLED pairing. Scalars (val_mass/*_jsd_*) log every epoch;
            # figures only on plot epochs.
            supports_mass_diag = cartesian or feature_names == _DEFAULT_SPHERICAL_FEATURES
            mass_state = MassDiagState(cartesian=cartesian) if supports_mass_diag else None
            if not supports_mass_diag and is_rank0 and plot_on:
                _log.warning(
                    "[lc] skipping mass diagnostics for feature_names=%s; they require cartesian or %s.",
                    feature_names,
                    _DEFAULT_SPHERICAL_FEATURES,
                )
            with torch.no_grad():
                for batch_cpu in val_shard.iter_torch_batches(**val_loader_cfg):
                    batch = _batch_to_device(batch_cpu, device)
                    if epoch == start_epoch and v_batch_idx == 0:
                        _validate_truth_batch(batch, rank=rank, split="validation", nu_kin_dim=lc["nu_kin_dim"],
                                              invisible_key=invisible_key, cartesian=cartesian)
                    vloss, metrics = core.reconstruction_loss(batch)
                    bs = int(batch["x"].shape[0])
                    vloss_f = float(vloss.cpu())
                    v_running += vloss_f * bs
                    v_seen += bs
                    for k in extra_keys:
                        if k in metrics:
                            v_extra[k] += float(metrics[k].cpu()) * bs
                    # Per-batch validation curve on its own continuous axis (rank-0 local).
                    val_step += 1
                    if wandb_run is not None and is_rank0:
                        wandb_run.log({"val/step": val_step, "val/loss_step": vloss_f})
                    # Latent-separation + feature-shift diagnostics on the first few val batches.
                    if n_diag_batches < swd_diag_max_batches and bs >= 4:
                        z_t = core.encode_latent(batch, detach_neutrinos=True)
                        shuf = {k: v for k, v in batch.items()}
                        perm = torch.randperm(bs, device=device)
                        shuf[invisible_key] = batch[invisible_key][perm]  # wrong nu<->event pairing
                        z_s = core.encode_latent(shuf, detach_neutrinos=True)
                        z_truth_chunks.append(z_t)
                        z_shuf_chunks.append(z_s)
                        for feature_index, feature_name in enumerate(feature_names):
                            for shift_pct in feature_shift_pcts:
                                shifted = {k: v for k, v in batch.items()}
                                shifted[invisible_key] = _shift_feature_percent(
                                    batch[invisible_key],
                                    feature_index=feature_index,
                                    shift_pct=shift_pct,
                                    log_scaled=feature_log_scaled[feature_index],
                                    periodic=feature_uniform[feature_index],
                                )
                                z_shift_chunks[_feature_shift_tag(feature_name, shift_pct)].append(
                                    core.encode_latent(shifted, detach_neutrinos=True)
                                )
                        # Physics diagnostics on the same batches (reuses the shuffle
                        # perm above so "shuffled" = same wrong nu<->event pairing the
                        # latent SWD is supposed to flag).
                        if mass_state is not None:
                            truth_kin_d = core.neutrino_kin_from_batch(batch)
                            _, nu_reco_norm_d = core(batch)
                            recon_kin_d = core.denormalize_neutrinos(nu_reco_norm_d)
                            mass_state.update(batch, truth_kin_d, recon_kin_d, truth_kin_d[perm])
                        n_diag_batches += 1
                    # Truth-vs-reconstruction histograms for the monitoring plots.
                    if plot_state is not None and n_plot_batches < lc["plot_max_batches"]:
                        z_pl, nu_reco_norm = core(batch)  # (B, latent), (B, 2, F) normalized
                        recon_phys = core.denormalize_neutrinos(nu_reco_norm)
                        truth_phys = core.neutrino_kin_from_batch(batch)
                        plot_state.update(truth_phys, recon_phys, batch["x_invisible_mask"], z_pl)
                        n_plot_batches += 1
                    v_batch_idx += 1
            _dbg(debug, rank, world_size,
                 f"epoch={epoch} val loop done v_batches={n_diag_batches}; all-reduce val_loss")
            val_loss = _all_reduce_mean(v_running / max(v_seen, 1), float(v_seen), device, world_size)
            extra_reduced = {
                k: _all_reduce_mean(v_extra[k] / max(v_seen, 1), float(v_seen), device, world_size)
                for k in extra_keys
            }

            # Per-rank diagnostic latents (a rank may have none). Reductions below
            # are issued unconditionally with weight 0 on no-data ranks so every
            # rank performs the SAME collectives and DDP stays in lock-step.
            if z_truth_chunks:
                z_truth = torch.cat(z_truth_chunks, dim=0)
                z_shuf = torch.cat(z_shuf_chunks, dim=0)
                local_rms = float(z_truth.pow(2).mean().sqrt().cpu())
                local_sep = _latent_separation_diagnostics(z_truth, z_shuf)
                z_shift_cat = {
                    tag: torch.cat(chunks, dim=0)
                    for tag, chunks in z_shift_chunks.items() if chunks
                }
                local_feature_shift = _feature_shift_diagnostics(z_truth, z_shift_cat)
                w = float(z_truth.shape[0])
            else:
                local_rms, local_sep, local_feature_shift, w = 0.0, {}, {}, 0.0

            _dbg(debug, rank, world_size, f"epoch={epoch} reducing latent_rms + sep diagnostics")
            latent_rms = _all_reduce_mean(local_rms, w, device, world_size)
            w_sep = w if local_sep else 0.0
            sep = {
                k: _all_reduce_mean(local_sep.get(k, 0.0), w_sep, device, world_size)
                for k in ("swd_null", "swd_shuffled", "swd_separation")
            }
            # Iterate the rank-identical key list (not local_feature_shift.keys()) so the
            # collective count matches across ranks even when a rank has no data.
            w_pt = w if local_feature_shift else 0.0
            feature_shift_metrics = {
                k: _all_reduce_mean(local_feature_shift.get(k, 0.0), w_pt, device, world_size)
                for k in feature_shift_keys
            }
            _dbg(debug, rank, world_size, f"epoch={epoch} validation collectives done")

            # Physics diagnostics: sum histogram counts across ranks (symmetric --
            # every rank constructed mass_state from the same epoch-independent
            # condition), then compute JSD scalars from the reduced counts.
            if mass_state is not None:
                _dbg(debug, rank, world_size, f"epoch={epoch} reducing mass diagnostics")
                mass_state.all_reduce(device, world_size)
                mass_scalars = mass_state.jsd_scalars()
                if is_rank0 and wandb_run is not None and plot_on:
                    import wandb  # local import: only rank 0 needs it
                    mass_figs = mass_state.build_figures()
                    payload = {"val/step": val_step}
                    for tag, fig in mass_figs.items():
                        payload[f"val_mass/{tag}"] = wandb.Image(fig)
                    wandb_run.log(payload)
                    import matplotlib.pyplot as _plt
                    for fig in mass_figs.values():
                        _plt.close(fig)
                    _log.info("[lc] logged %d val_mass plots to W&B (epoch=%s)", len(mass_figs), epoch)

            # Monitoring plots: sum histogram counts across ranks (rank-symmetric,
            # every rank issues the same collectives because plot_on is epoch-gated),
            # then render + log W&B images on rank 0 -- same category as the main
            # model's neutrino generation plots ("generation-invisible/*").
            if plot_state is not None:
                _dbg(debug, rank, world_size, f"epoch={epoch} reducing monitoring-plot histograms")
                plot_state.all_reduce(device, world_size)
                if is_rank0 and wandb_run is not None:
                    import wandb  # local import: only rank 0 needs it
                    figs, jsd_results = plot_state.build_figures()
                    payload = {"val/step": val_step}
                    for tag, fig in figs.items():
                        clean = tag.replace("neutrino-", "")
                        payload[f"generation-invisible/{clean}"] = wandb.Image(fig)
                    for key, score in jsd_results.items():
                        clean = key.replace("neutrino-", "")
                        payload[f"generation-invisible/jsd/{clean}"] = float(score)
                    wandb_run.log(payload)
                    import matplotlib.pyplot as _plt
                    for fig in figs.values():
                        _plt.close(fig)
                    _log.info("[lc] logged %d monitoring plots to W&B (epoch=%s)", len(figs), epoch)

        lr_now = float(scheduler.get_last_lr()[0])
        # Build metrics on every rank (all values are already all-reduced) and
        # report from every rank: Ray Train expects symmetric report() calls.
        report_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "val_step": val_step,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/latent_rms": latent_rms,
            "train/lr": lr_now,
            **{_route_extra_metric_key(k): v for k, v in extra_reduced.items()},
            **{f"val/{k}": v for k, v in sep.items()},
            **{f"val/{k}": v for k, v in feature_shift_metrics.items()},
            **{f"val_mass/{k}": v for k, v in mass_scalars.items()},
        }
        _dbg(debug, rank, world_size, f"epoch={epoch} ray.train.report")
        ray.train.report(report_metrics)

        if is_rank0:
            _log.info(
                "[lc] epoch=%d step=%d val_step=%d train_loss=%.5f val_loss=%.5f sep=%.2f%s",
                epoch, global_step, val_step, train_loss, val_loss,
                sep.get("swd_separation", float("nan")),
                "".join(
                    f" {k.replace('swd_feature_shift_', 'shift:')}={v:.4f}"
                    for k, v in feature_shift_metrics.items()
                ),
            )
            if wandb_run is not None:
                # Two separate calls -> train and val land on their own continuous axes.
                wandb_run.log(
                    {
                        "train/step": global_step,
                        "train/epoch": epoch,
                        "train/loss": train_loss,
                        "train/lr": lr_now,
                    }
                )
                wandb_run.log(
                    {
                        "val/step": val_step,
                        "val/epoch": epoch,
                        "val/loss": val_loss,
                        "val/latent_rms": latent_rms,
                        **{_route_extra_metric_key(k): v for k, v in extra_reduced.items()},
                        **{f"val/{k}": v for k, v in sep.items()},
                        **{f"val/{k}": v for k, v in feature_shift_metrics.items()},
                        **{f"val_mass/{k}": v for k, v in mass_scalars.items()},
                    }
                )
        if es_enabled:
            es_current = _early_stopping_metric_value(
                es_monitor, val_loss=val_loss, sep=sep,
            )
            if _early_stopping_improved(
                es_current, es_best, mode=es_mode, min_delta=es_min_delta,
            ):
                es_best = es_current
                es_wait = 0
            else:
                es_wait += 1
            if wandb_run is not None and is_rank0:
                wandb_run.log(
                    {
                        "val/step": val_step,
                        "val/early_stopping_wait": float(es_wait),
                        "val/early_stopping_best": float(es_best),
                    }
                )
            if es_wait >= es_patience:
                if is_rank0:
                    _log.info(
                        "[lc] early_stopping triggered at epoch=%s: monitor=%s best=%.6f "
                        "wait=%s/%s",
                        epoch, es_monitor, es_best, es_wait, es_patience,
                    )
                stop_training = True

        if is_rank0:
            lc_next_epoch = epoch + 1
            ckpt_extra = {
                "val_step": val_step,
                "early_stopping_best": es_best,
                "early_stopping_wait": es_wait,
                "early_stopping_monitor": es_monitor,
            }
            save_checkpoint(
                save_dir / "last.ckpt", core,
                normalization_file=normalization_file,
                epoch=epoch,
                val_loss=val_loss,
                global_step=global_step,
                lc_next_epoch=lc_next_epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                extra=ckpt_extra,
            )
            monitored = val_loss if math.isfinite(val_loss) else train_loss
            if monitored < best_val:
                best_val = monitored
                save_checkpoint(
                    save_dir / "best.ckpt", core,
                    normalization_file=normalization_file,
                    epoch=epoch,
                    val_loss=monitored,
                    global_step=global_step,
                    lc_next_epoch=lc_next_epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra=ckpt_extra,
                )

        if world_size > 1:
            _dbg(debug, rank, world_size, f"epoch={epoch} entering end-of-epoch barrier")
            dist.barrier()
            _dbg(debug, rank, world_size, f"epoch={epoch} passed barrier")
        if max_steps is not None and global_step >= max_steps:
            _dbg(debug, rank, world_size, f"max_steps={max_steps} reached; stopping")
            break
        if stop_training:
            _dbg(debug, rank, world_size, "early_stopping; stopping")
            break

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    p = argparse.ArgumentParser(description="Train the DGPO latent constraint autoencoder")
    p.add_argument(
        "config", type=Path, nargs="?",
        default=Path(__file__).resolve().parent / "config.yaml",
        help="YAML config (EveNet merge rules + latent_constraint block)",
    )
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N optimizer steps (smoke test)")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint (overrides config model_checkpoint_load_path)",
    )
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    p.add_argument("--ray-dir", type=str, default="~/ray_results", help="Ray RunConfig.storage_path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_path = args.config.resolve()
    global_config.load_yaml(config_path)
    global_config.display()
    platform_info = global_config.platform

    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "0")
    runtime_env = {
        "env_vars": {
            "PYTHONPATH": f"{_REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}",
            "TORCH_NCCL_TIMEOUT": "180",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": os.environ[
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
            ],
            # Silence Ray Data's in-place progress bars so the real training logs
            # (epoch/step/loss) are not buried and the run does not *look* stuck.
            "RAY_DATA_DISABLE_PROGRESS_BARS": os.environ.get(
                "RAY_DATA_DISABLE_PROGRESS_BARS", "1"
            ),
        }
    }
    if "WANDB_API_KEY" in os.environ:
        runtime_env["env_vars"]["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    ray_addr_env = os.environ.get("RAY_ADDRESS")
    try:
        ray.init(
            address=ray_addr_env or "auto",
            runtime_env=runtime_env,
            ignore_reinit_error=True,
        )
    except (ConnectionError, ValueError) as ex:
        _log.warning("[lc][launch] no existing Ray cluster (%s); local init.", ex)
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    expected_workers = int(platform_info.number_of_workers)
    expected_gpus_per_worker = float(dict(platform_info.resources_per_worker).get("GPU", 1))
    expected_gpus = float(expected_workers) * expected_gpus_per_worker
    wait_timeout_s = float(os.environ.get("LC_RAY_WAIT_S", os.environ.get("DGPO_RAY_WAIT_S", "300")))
    poll_every = 5.0
    waited = 0.0
    while waited < wait_timeout_s:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        cur_nodes = len(ray.nodes())
        if cur_gpus >= expected_gpus:
            _log.info(
                "[lc][launch] Ray cluster ready: nodes=%s GPUs=%s (expected %s).",
                cur_nodes, cur_gpus, expected_gpus,
            )
            break
        _log.info(
            "[lc][launch] waiting for Ray workers... nodes=%s GPUs=%s/%s (%.0fs/%.0fs)",
            cur_nodes, cur_gpus, expected_gpus, waited, wait_timeout_s,
        )
        time.sleep(poll_every)
        waited += poll_every
    else:
        cur_gpus = float(ray.cluster_resources().get("GPU", 0))
        _log.warning(
            "[lc][launch] timed out after %.0fs: GPUs=%s (expected %s). "
            "Continuing — Ray Train may run with fewer workers or hang.",
            wait_timeout_s, cur_gpus, expected_gpus,
        )

    base_dir = Path(platform_info.data_parquet_dir)
    base_val_dir = (
        Path(platform_info.data_parquet_val_dir) if "data_parquet_val_dir" in platform_info else None
    )
    process_fn = make_process_fn(base_dir)
    train_ds, val_ds, total_events, val_events = prepare_datasets(
        base_dir=base_dir,
        process_event_batch_partial=process_fn,
        platform_info=platform_info,
        load_all_in_ram=False,
        base_val_dir=base_val_dir,
        predict=False,
    )
    datasets: dict[str, Any] = {"train": train_ds}
    if val_ds is not None and val_events:
        datasets["validation"] = val_ds

    scaling_config = ScalingConfig(
        num_workers=int(platform_info.number_of_workers),
        resources_per_worker=dict(platform_info.resources_per_worker),
        use_gpu=bool(platform_info.get("use_gpu", True)),
    )
    try:
        cluster_resources = ray.cluster_resources()
    except Exception:
        cluster_resources = {}
    _log.info(
        "[lc][launch] num_workers=%s resources_per_worker=%s use_gpu=%s "
        "cluster_GPUs=%s cluster_CPUs=%s nodes=%s",
        scaling_config.num_workers,
        scaling_config.resources_per_worker,
        scaling_config.use_gpu,
        cluster_resources.get("GPU"),
        cluster_resources.get("CPU"),
        len(ray.nodes()) if ray.is_initialized() else "?",
    )
    run_config = RunConfig(name="LatentConstraint-Training", storage_path=args.ray_dir)
    trainer_config = {
        "config_path": str(config_path),
        "max_steps": args.max_steps,
        "resume_checkpoint": str(args.resume) if args.resume is not None else None,
        "wandb": not args.no_wandb,
        "total_events": int(total_events),
        "val_events": int(val_events) if val_events else 0,
    }
    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop,
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
