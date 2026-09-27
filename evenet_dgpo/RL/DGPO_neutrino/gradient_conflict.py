"""Read-only, block-replicated DGPO gradient diagnostics (not gradient surgery).

All vectors live in policy parameter space. Staleness logits are a detached,
truth-positive substitute reward, NOT an additional policy training objective.
Intervals are approximate block-jackknife intervals conditional on the fitted
classifiers and validation panel, not sequential tests or AUC guarantees.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from scipy.stats import t as student_t

from .dgpo_utils import build_dgpo_loss, build_reference_trust_loss, compute_per_event_advantage


@dataclass(frozen=True)
class GradientConflictConfig:
    enabled: bool = False
    monitor_refit_lifecycle: bool = False
    every_n_steps: int = 10
    blocks: int = 8
    events_per_block: int = 512  # GLOBAL, split over ranks, not per GPU
    event_microbatch_size: int = 32
    seed: int = 20260910
    confidence: float = .95
    min_split_cosine: float = .1
    norm_floor: float = 1e-12


def resolve_gradient_conflict_config(payload: Mapping[str, Any] | None) -> GradientConflictConfig:
    values = dict(payload or {})
    if set(values) - set(GradientConflictConfig.__dataclass_fields__):
        raise ValueError("unknown dgpo.gradient_conflict option")
    cfg = GradientConflictConfig(**values)
    for name in ("enabled", "monitor_refit_lifecycle"):
        if type(getattr(cfg, name)) is not bool:
            raise ValueError(f"gradient_conflict.{name} must be boolean")
    for name in ("every_n_steps", "blocks", "events_per_block", "event_microbatch_size", "seed"):
        value = getattr(cfg, name)
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"gradient_conflict.{name} must be a valid integer")
    if cfg.blocks < 8 or cfg.blocks % 2:
        raise ValueError("gradient_conflict.blocks must be even and at least 8")
    if not .5 < cfg.confidence < 1 or not 0 <= cfg.min_split_cosine < 1:
        raise ValueError("invalid gradient conflict confidence/repeatability threshold")
    if not math.isfinite(cfg.norm_floor) or cfg.norm_floor <= 0:
        raise ValueError("gradient_conflict.norm_floor must be finite and positive")
    return cfg


def lifecycle_probe_due(cfg, state, *, warmup_steps: int) -> str | None:
    """Called at raw-monitor checkpoints; markers persist in adaptive state."""
    if not cfg.enabled or not cfg.monitor_refit_lifecycle:
        return None
    if state.policy_warmup_round_id != state.reward_round_id:
        return None
    if (state.policy_warmup_completed_updates == 0
            and state.gradient_post_install_probe_round_id != state.reward_round_id):
        return "post_install"
    if (state.policy_warmup_completed_updates >= warmup_steps
            and state.gradient_warmup_probe_round_id != state.reward_round_id):
        return "post_warmup"
    return None


def phase_metrics(metrics, phase: str):
    """Separate series prevent before/after points at the same step overwriting."""
    if phase not in {"periodic", "pre_refit", "post_install", "post_warmup"}:
        raise ValueError("unknown gradient probe phase")
    prefix = "gradient_conflict/"
    target = prefix if phase == "periodic" else prefix + phase + "/"
    return {target + key[len(prefix):]: value for key, value in metrics.items()
            if key.startswith(prefix)}


def _cross_interval(matrix: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """Order-2 U-statistic, excluding shared-block covariance on the diagonal."""
    n = len(matrix)
    matrix = (matrix + matrix.T) * .5
    off = matrix.copy()
    np.fill_diagonal(off, 0.)
    total = off.sum()
    estimate = float(total / (n * (n - 1)))
    leave_one_out = (total - 2 * off.sum(axis=1)) / ((n - 1) * (n - 2))
    se = float(np.sqrt((n - 1) / n * np.sum((leave_one_out - leave_one_out.mean()) ** 2)))
    radius = float(student_t.ppf((1 + confidence) / 2, n - 1)) * se
    return estimate, estimate - radius, estimate + radius


def summarize_gradient_gram(gram: np.ndarray, cfg: GradientConflictConfig) -> dict[str, float]:
    """Gram layout [block, objective, block, objective], O/S/T respectively."""
    n = cfg.blocks
    if gram.shape != (n, 3, n, 3) or not np.isfinite(gram).all():
        raise ValueError("nonfinite or incorrectly shaped gradient Gram matrix")
    norms, reliable = [], []
    out: dict[str, float] = {}
    half = n // 2
    for j, name in enumerate(("omnifold", "staleness", "trust")):
        mat = gram[:, j, :, j]
        norm = math.sqrt(max(0., float(mat.mean())))
        norms.append(norm)
        aa, bb, ab = mat[:half, :half].mean(), mat[half:, half:].mean(), mat[:half, half:].mean()
        denominator = math.sqrt(max(0., float(aa * bb)))
        split = float(np.clip(ab / denominator, -1, 1)) if denominator > cfg.norm_floor ** 2 else float("nan")
        signal, lo, hi = _cross_interval(mat, cfg.confidence)
        ready = norm > cfg.norm_floor and lo > 0 and split >= cfg.min_split_cosine
        reliable.append(ready)
        out.update({f"{name}/norm": norm, f"{name}/split_cosine": split,
                    f"{name}/cross_self_dot": signal, f"{name}/cross_self_dot_lcb": lo,
                    f"{name}/cross_self_dot_ucb": hi, f"{name}/reliable": float(ready)})
    for a, b, name in ((0, 1, "omnifold_staleness"), (0, 2, "omnifold_trust"), (1, 2, "staleness_trust")):
        mat = gram[:, a, :, b]
        denominator = norms[a] * norms[b]
        cosine = float(np.clip(mat.mean() / denominator, -1, 1)) if denominator > cfg.norm_floor ** 2 else float("nan")
        cross, lo, hi = _cross_interval(mat, cfg.confidence)
        conclusive = reliable[a] and reliable[b] and (hi < 0 or lo > 0)
        out.update({f"{name}/cosine": cosine, f"{name}/cross_dot": cross,
                    f"{name}/cross_dot_lcb": lo, f"{name}/cross_dot_ucb": hi,
                    f"{name}/conclusive": float(conclusive),
                    f"{name}/conflict": float(conclusive and hi < 0),
                    f"{name}/alignment": float(conclusive and lo > 0)})
    # O+T is the raw loss gradient, NOT the AdamW-preconditioned update.
    # Cross-block intervals avoid declaring cancellation from shared-block noise.
    for j, name in enumerate(("omnifold", "staleness")):
        matrix = gram[:, j, :, 0] + gram[:, j, :, 2]
        estimate, lo, hi = _cross_interval(matrix, cfg.confidence)
        denominator = norms[j] ** 2
        out.update({
            f"total_on_{name}/projection_ratio": (
                float(matrix.mean()) / denominator
                if denominator > cfg.norm_floor ** 2 else float("nan")),
            f"total_on_{name}/cross_dot": estimate,
            f"total_on_{name}/cross_dot_lcb": lo,
            f"total_on_{name}/cross_dot_ucb": hi,
            f"total_on_{name}/opposed": float(reliable[j] and hi < 0),
        })
    return {"gradient_conflict/" + key: value for key, value in out.items()}


def gradient_gram(vectors: list[torch.Tensor]) -> np.ndarray:
    """Exact dots, CPU storage, bounded float64 working set (no random sketch)."""
    size = len(vectors)
    gram = torch.zeros(size, size, dtype=torch.float64)
    for start in range(0, vectors[0].numel(), 65536):
        chunk = torch.stack([v[start:start + 65536] for v in vectors]).double()
        gram.addmm_(chunk, chunk.T)
    return gram.numpy().reshape(size // 3, 3, size // 3, 3)


@contextmanager
def isolated_probe_state(model: torch.nn.Module, reference: torch.nn.Module, device: torch.device):
    """Preserve stochastic state, modes and buffers, never touch .grad or AdamW."""
    modules = list(dict.fromkeys([*model.modules(), *reference.modules()]))
    modes = [(module, module.training) for module in modules]
    buffers = [(buffer, buffer.detach().clone()) for root in (model, reference) for buffer in root.buffers()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            reference.eval()
            with torch.enable_grad():
                yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        for module, training in modes:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def _flat_gradient(loss, parameters, *, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).float().reshape(-1)
                      for p, g in zip(parameters, gradients)])


def probe_gradients(*, cfg, model, core, reference, pool, monitor_factory, monitor_cache,
                    reward_compute, generate, evaluate, sampler, device, dtype, rank, world_size,
                    K, beta, num_ddim_steps, rollout_parallel_chains, num_train_timesteps,
                    t_min, t_max, advantage_estimator, adv_clip_max, trust_coefficient,
                    global_step, reward_round_id, raw_auc, raw_ready) -> dict[str, Any]:
    """All ranks enter together after raw fitting, BEFORE rollback/refit.

    Generation and reward scores are detached. All three objectives share each
    graph and (t, eps). Each block is a global event-weighted mean. Parameters
    stay fixed across blocks; only gradients (not AdamW-preconditioned updates)
    are compared. Inactive/unsupported situations are explicitly skipped.
    """
    prefix = "gradient_conflict/"
    meta = {prefix + "global_step": float(global_step), prefix + "reward_round_id": float(reward_round_id),
            prefix + "raw_auc": float(raw_auc), prefix + "raw_training_ready": float(raw_ready),
            prefix + "ran": 0., prefix + "confidence": cfg.confidence}
    if not raw_ready or not monitor_cache.get("state"):
        return {**meta, prefix + "skipped_unready": 1.}
    if pool.policy_noise_mask is None:
        return {**meta, prefix + "skipped_missing_masks": 1.}
    from .omnifold_ztautau.evenet_ratio import unpack_event_inputs, _identity_crossfit_splits
    protocol = monitor_cache.get("protocol", {})
    # All supported protocols assign identities with the same deterministic
    # five-fold hash split.  The newer names record whether fold 1 is reserved
    # for early stopping; fold 0 remains the probe/audit holdout in each case.
    identity_stable_schemas = {
        "raw-monitor-condition-split-v1",       # legacy 80/20 name
        "raw-monitor-condition-80-20-v1",
        "raw-monitor-condition-60-20-20-v1",
    }
    if protocol.get("schema") not in identity_stable_schemas or protocol.get("folds") != 5:
        raise ValueError("gradient probe requires the certified identity-stable raw monitor split")
    _, validation = _identity_crossfit_splits(pool.identity_inputs, folds=5, seed=int(protocol["seed"]))[0]
    total = cfg.blocks * cfg.events_per_block
    # Duplicate identities must not masquerade as independent statistical blocks.
    if len(validation) < total or cfg.events_per_block < world_size:
        return {**meta, prefix + "skipped_insufficient_events": 1., prefix + "available_events": float(len(validation))}
    order = torch.randperm(len(validation), generator=torch.Generator().manual_seed(cfg.seed))
    identities = pool.identity_inputs.detach().cpu()
    seen, selected_panel = set(), []
    for index in validation.cpu()[order].tolist():
        digest = hashlib.sha256(identities[index].contiguous().numpy().tobytes()).digest()
        if digest not in seen:
            seen.add(digest)
            selected_panel.append(index)
        if len(selected_panel) == total:
            break
    if len(selected_panel) < total:
        return {**meta, prefix + "skipped_insufficient_events": 1., prefix + "available_events": float(len(selected_panel))}
    panel = torch.tensor(selected_panel, dtype=torch.long)
    panel_bytes = pool.identity_inputs[panel.to(pool.identity_inputs.device)].detach().cpu().contiguous().numpy().tobytes()
    meta[prefix + "panel_sha256"] = hashlib.sha256(panel_bytes).hexdigest()
    parameters = tuple(p for p in core.parameters() if p.requires_grad)
    if not parameters:
        return {**meta, prefix + "skipped_no_parameters": 1.}
    started = time.monotonic()
    vectors: list[torch.Tensor] = []
    sync_context = model.no_sync() if hasattr(model, "no_sync") else nullcontext()
    with isolated_probe_state(core, reference, device), sync_context:
        torch.manual_seed(cfg.seed)
        judge = monitor_factory().to(device)
        # Load before freezing: EveNet state_dict includes trainable body/adapters.
        judge.load_state_dict(monitor_cache["state"], strict=True)
        judge.eval()
        for p in judge.parameters():
            p.requires_grad_(False)
        for block in range(cfg.blocks):
            torch.manual_seed(cfg.seed + block * 100003 + rank)
            block_panel = panel[block * cfg.events_per_block:(block + 1) * cfg.events_per_block]
            selected = block_panel[rank::world_size]
            mask_mass = float(pool.policy_noise_mask[block_panel.to(pool.policy_noise_mask.device)].float().sum())
            accum = [torch.zeros(sum(p.numel() for p in parameters), device=device, dtype=torch.float32) for _ in range(3)]
            for start in range(0, len(selected), cfg.event_microbatch_size):
                ix = selected[start:start + cfg.event_microbatch_size]
                packed = pool.packed_event[ix.to(pool.packed_event.device)].to(device)
                batch = unpack_event_inputs(packed, pool.packing_spec)
                batch["x_invisible"] = pool.truth[ix.to(pool.truth.device)].to(device).reshape(-1, 2, 2)
                batch["x_invisible_mask"] = pool.policy_noise_mask[ix.to(pool.policy_noise_mask.device)].to(device)
                with torch.no_grad():
                    candidates = generate(core, batch, sampler, K=K, num_ddim_steps=num_ddim_steps,
                                          device=device, parallel_chains=rollout_parallel_chains)
                    reward = reward_compute(candidates, batch)
                    stale = judge(packed[:, None].expand(-1, K, -1).reshape(-1, packed.shape[-1]),
                                  candidates.permute(1, 0, 2, 3).reshape(-1, 4)).reshape(len(ix), K).T
                    advantages = [compute_per_event_advantage(r.detach(), estimator=advantage_estimator)[0]
                                  for r in (reward, stale)]
                    if adv_clip_max is not None:
                        advantages = [a.clamp(-adv_clip_max, adv_clip_max) for a in advantages]
                for _ in range(num_train_timesteps):
                    values = evaluate(core, reference, batch, candidates, K=K, shared_noise=True,
                                      device=device, dtype=dtype, t_min=t_min, t_max=t_max)
                    current, ref, _, velocity, ref_velocity, noise_mask = values[:6]
                    losses = [build_dgpo_loss(current, ref, a, beta_dgpo=beta, K=K)[0] for a in advantages]
                    trust = build_reference_trust_loss(velocity, ref_velocity, noise_mask, L_ref_2d=ref)[0]
                    losses.append(trust * trust_coefficient)
                    weight = len(ix) / (cfg.events_per_block * num_train_timesteps)
                    for j, loss in enumerate(losses):
                        objective_weight = weight if j < 2 else (
                            float(batch["x_invisible_mask"].float().sum()) / max(mask_mass, 1e-12) / num_train_timesteps)
                        accum[j].add_(_flat_gradient(loss, parameters, retain_graph=j < 2), alpha=objective_weight)
                    del values, losses, current, ref, velocity, ref_velocity, trust
            # Explicit reduction, not DDP reducer hooks; unequal rank counts are
            # already weighted by GLOBAL events, so SUM (never mean of means).
            for vector in accum:
                if world_size > 1:
                    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
            if not all(bool(torch.isfinite(v).all()) for v in accum):
                return {**meta, prefix + "skipped_nonfinite": 1.}
            if rank == 0:
                vectors.extend(v.cpu() for v in accum)
            del accum
        del judge
    metrics = summarize_gradient_gram(gradient_gram(vectors), cfg) if rank == 0 else {}
    return {**meta, **metrics, prefix + "ran": 1., prefix + "blocks": float(cfg.blocks),
            prefix + "events": float(total), prefix + "K": float(K),
            prefix + "timesteps": float(num_train_timesteps), prefix + "seconds": time.monotonic() - started,
            prefix + "trainable_parameters": float(sum(p.numel() for p in parameters)),
            prefix + "trust_coefficient": float(trust_coefficient)}
