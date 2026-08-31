"""Paper-faithful population reweighting with a truth-free reward interface.

This module implements the two density-ratio updates in OmniFold for the toy
identity-response setting.  Reference data are used only as one population in
the Step-1 classifier.  Every classifier and reward evaluation receives only a
condition and a sample; there is no paired-reference input or event-wise error
feature.

The population weights are normalized by one scalar over the complete pool.
They are never normalized independently inside an event, and classifier logits
are not clipped during training or inference.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OmniFoldPassOneESSError(RuntimeError):
    """Pass-one push weights collapsed below the ESS floor."""

    def __init__(self, ess_value: float, ess_floor: float) -> None:
        self.ess_value = float(ess_value)
        self.ess_floor = float(ess_floor)
        super().__init__(
            f"OmniFold pass-one ESS {self.ess_value:.6f} is below floor {self.ess_floor:.6f}"
        )


def distributed_context() -> tuple[int, int]:
    """``(rank, world_size)``; ``(0, 1)`` when no process group is active."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def shard_global_batch(indices: Tensor, rank: int, world_size: int) -> Tensor:
    """Give each rank a disjoint contiguous slice of one global shuffled batch."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed shard arguments")
    if world_size == 1:
        return indices
    n = int(indices.numel())
    if n < world_size:
        raise ValueError(
            f"global batch {n} is smaller than world_size {world_size}"
        )
    start = (n * rank) // world_size
    stop = (n * (rank + 1)) // world_size
    return indices[start:stop]


def _needs_nccl_cpu_staging(tensor: Tensor) -> bool:
    """NCCL has no CPU backend; CPU tensors must stage through the local CUDA device."""
    if tensor.device.type != "cpu":
        return False
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return False
    return str(torch.distributed.get_backend()) == "nccl"


def _nccl_staging_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "NCCL process group cannot operate on CPU tensors, and no CUDA device "
            "is visible to stage through. Move the OmniFold fit to CUDA or use gloo."
        )
    return torch.device("cuda", torch.cuda.current_device())


def _broadcast_tensor(tensor: Tensor) -> None:
    """In-place broadcast of a live parameter/buffer, NCCL-safe for CPU tensors."""
    if _needs_nccl_cpu_staging(tensor):
        buf = tensor.detach().to(_nccl_staging_device())
        torch.distributed.broadcast(buf, src=0)
        tensor.copy_(buf.to(device=tensor.device))
        return
    torch.distributed.broadcast(tensor.detach(), src=0)


def _all_reduce_sum_tensor(tensor: Tensor) -> None:
    """In-place sum all-reduce, NCCL-safe for CPU tensors."""
    _, world = distributed_context()
    if world <= 1:
        return
    if _needs_nccl_cpu_staging(tensor):
        buf = tensor.detach().to(_nccl_staging_device())
        torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
        tensor.copy_(buf.to(device=tensor.device))
        return
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)


def _all_reduce_mean_tensor(tensor: Tensor, world: int) -> None:
    """In-place mean all-reduce, NCCL-safe for CPU tensors."""
    _all_reduce_sum_tensor(tensor)
    if world > 1:
        tensor.div_(world)


def _flat_collective(tensors: list[Tensor], *, op: str, world: int) -> None:
    """One flat collective per dtype instead of one per tensor.

    A 21M-parameter EveNet ratio classifier carries several hundred parameter
    tensors. Across 4 nodes, per-tensor NCCL calls are launch-latency bound
    rather than bandwidth bound, so a single flattened buffer is much faster.
    Callers must pass the tensors in a rank-independent order.
    """
    if not tensors:
        return
    buckets: dict[tuple[torch.dtype, torch.device], list[Tensor]] = {}
    for tensor in tensors:
        buckets.setdefault((tensor.dtype, tensor.device), []).append(tensor)
    for (_dtype, device), bucket in buckets.items():
        stage = _needs_nccl_cpu_staging(bucket[0])
        flat = torch.cat([tensor.detach().reshape(-1) for tensor in bucket])
        if stage:
            flat = flat.to(_nccl_staging_device())
        if op == "mean":
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            if world > 1:
                flat.div_(world)
        elif op == "broadcast":
            torch.distributed.broadcast(flat, src=0)
        else:
            raise ValueError(f"unsupported flat collective op: {op!r}")
        if stage:
            flat = flat.to(device=device)
        offset = 0
        for tensor in bucket:
            count = tensor.numel()
            tensor.detach().copy_(flat[offset : offset + count].view_as(tensor))
            offset += count


def _broadcast_module(model: nn.Module) -> None:
    _, world = distributed_context()
    if world <= 1:
        return
    # Iterate live parameters/buffers, not ``state_dict()`` copies: a broadcast
    # into a detached clone would leave the module unchanged. Sorting by name
    # keeps the flattened layout identical on every rank.
    named = sorted(model.named_parameters(), key=lambda item: item[0]) + sorted(
        model.named_buffers(), key=lambda item: item[0]
    )
    _flat_collective(
        [
            tensor
            for _name, tensor in named
            if tensor.is_floating_point()
            or tensor.dtype in (torch.int32, torch.int64, torch.uint8, torch.bool)
        ],
        op="broadcast",
        world=world,
    )


def _average_gradients(model: nn.Module) -> None:
    _, world = distributed_context()
    if world <= 1:
        return
    _flat_collective(
        [
            parameter.grad
            for _name, parameter in sorted(
                model.named_parameters(), key=lambda item: item[0]
            )
            if parameter.grad is not None
        ],
        op="mean",
        world=world,
    )


def _broadcast_flag(value: bool, device: torch.device) -> bool:
    _, world = distributed_context()
    if world <= 1:
        return bool(value)
    tensor = torch.tensor([int(value)], device=device, dtype=torch.int32)
    torch.distributed.broadcast(tensor, src=0)
    return bool(tensor.item())


def _reduce_mean(tensor: Tensor) -> Tensor:
    """Average a scalar or tensor across ranks; identity when world size is 1."""
    _, world = distributed_context()
    if world <= 1:
        return tensor
    reduced = tensor.detach().clone()
    torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
    return reduced / world


class ConditionFiLMResidualBlock(nn.Module):
    """Residual block whose hidden state is FiLM-modulated by the condition."""

    def __init__(self, hidden_dim: int, expansion: int = 2) -> None:
        super().__init__()
        if hidden_dim < 1 or expansion < 1:
            raise ValueError("hidden_dim and expansion must be positive")
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.condition_projection = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, expansion * hidden_dim),
            nn.SiLU(),
            nn.Linear(expansion * hidden_dim, hidden_dim),
        )

    def forward(self, hidden: Tensor, condition_embedding: Tensor) -> Tensor:
        scale, shift = self.condition_projection(condition_embedding).chunk(2, dim=-1)
        modulated = self.layer_norm(hidden) * (1.0 + scale) + shift
        return hidden + self.mlp(modulated)


SUPPORTED_RATIO_ARCHITECTURES = ("film", "concat_mlp")


class ConditionalRatioMLP(nn.Module):
    """Estimate one conditional log density ratio from ``(condition, sample)``.

    Both architectures keep raw ``(condition, sample)`` inputs.  ``film`` encodes
    the condition once and FiLM-modulates a sample trunk; this is the locked
    toy-DGPO classifier.  ``concat_mlp`` concatenates then applies a SiLU MLP
    and is comparison-only.
    """

    def __init__(
        self,
        condition_dim: int,
        sample_dim: int,
        hidden_dim: int = 64,
        hidden_layers: int = 3,
        feature_mode: str = "raw",
        architecture: str = "film",
    ) -> None:
        super().__init__()
        if condition_dim < 1 or sample_dim < 1:
            raise ValueError("condition_dim and sample_dim must be positive")
        if hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("hidden_dim and hidden_layers must be positive")
        self.condition_dim = int(condition_dim)
        self.sample_dim = int(sample_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.feature_mode = str(feature_mode)
        self.architecture = str(architecture)
        if self.feature_mode != "raw":
            raise ValueError(
                "OmniFold classifier accepts only raw (condition, sample) features"
            )
        if self.architecture not in SUPPORTED_RATIO_ARCHITECTURES:
            raise ValueError(
                f"unsupported OmniFold classifier architecture: {self.architecture}"
            )
        self.feature_width = self.condition_dim + self.sample_dim
        if self.architecture == "concat_mlp":
            layers: list[nn.Module] = []
            width = self.feature_width
            for _ in range(self.hidden_layers):
                layers.extend((nn.Linear(width, self.hidden_dim), nn.SiLU()))
                width = self.hidden_dim
            layers.append(nn.Linear(width, 1))
            self.network = nn.Sequential(*layers)
            return
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.sample_input = nn.Linear(self.sample_dim, self.hidden_dim)
        self.blocks = nn.ModuleList(
            ConditionFiLMResidualBlock(self.hidden_dim)
            for _ in range(self.hidden_layers)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)

    def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
        flat_condition, flat_sample, output_shape = _flatten_condition_sample(
            condition, sample
        )
        if self.architecture == "concat_mlp":
            features = torch.cat((flat_condition, flat_sample), dim=-1)
            return self.network(features).squeeze(-1).reshape(output_shape)
        hidden = self.sample_input(flat_sample)
        condition_embedding = self.condition_encoder(flat_condition)
        for block in self.blocks:
            hidden = block(hidden, condition_embedding)
        return self.output(F.silu(self.output_norm(hidden))).squeeze(-1).reshape(
            output_shape
        )


@dataclass(frozen=True)
class RatioFitConfig:
    steps: int | None = 400
    batch_size: int = 128
    train_microbatch_size_per_rank: int | None = None
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    sampling: str = "independent_with_replacement"
    min_steps: int = 0
    validation_interval_steps: int = 0
    validation_patience_evaluations: int = 0
    validation_min_delta: float = 0.0
    validation_batch_size: int = 8192
    restore_best: bool = True
    require_saturation: bool = False
    progress_interval_steps: int = 0
    checkpoint_interval_steps: int = 0
    train_candidates_per_event: int | None = None

    def validate(self) -> None:
        if (self.steps is not None and self.steps < 1) or self.batch_size < 1:
            raise ValueError("steps must be positive or null; batch_size must be positive")
        if (
            self.train_microbatch_size_per_rank is not None
            and int(self.train_microbatch_size_per_rank) < 1
        ):
            raise ValueError("train_microbatch_size_per_rank must be positive or null")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are outside their valid range")
        if self.sampling not in {
            "independent_with_replacement", "independent_epoch_shuffle",
        }:
            raise ValueError("unsupported density-ratio sampling mode")
        if self.min_steps < 0 or (
            self.steps is not None and self.min_steps > self.steps
        ):
            raise ValueError("min_steps must be nonnegative and no greater than steps")
        if self.steps is None and self.sampling != "independent_epoch_shuffle":
            raise ValueError("unbounded fitting requires independent_epoch_shuffle")
        if min(
            self.validation_interval_steps,
            self.validation_patience_evaluations,
            self.validation_batch_size,
            self.progress_interval_steps,
            self.checkpoint_interval_steps,
        ) < 0:
            raise ValueError("validation controls must be nonnegative")
        if self.validation_min_delta < 0.0:
            raise ValueError("validation_min_delta must be nonnegative")
        if self.require_saturation and (
            self.sampling != "independent_epoch_shuffle"
            or self.min_steps < 1
            or self.validation_interval_steps < 1
            or self.validation_patience_evaluations < 1
            or self.validation_batch_size < 1
            or not self.restore_best
        ):
            raise ValueError(
                "required saturation needs epoch-shuffle sampling, positive "
                "validation controls, min_steps, and restore_best"
            )
        if self.train_candidates_per_event is not None:
            if int(self.train_candidates_per_event) < 1:
                raise ValueError("train_candidates_per_event must be >= 1")
            if self.sampling != "independent_epoch_shuffle":
                raise ValueError(
                    "train_candidates_per_event requires independent_epoch_shuffle"
                )


@dataclass(frozen=True)
class RatioFitDiagnostics:
    loss: float
    balanced_accuracy: float
    steps: int | None
    steps_completed: int | None = None
    best_step: int | None = None
    initial_validation_loss: float | None = None
    initial_validation_balanced_accuracy: float | None = None
    initial_validation_auc: float | None = None
    validation_loss: float | None = None
    validation_balanced_accuracy: float | None = None
    validation_auc: float | None = None
    saturated: bool = False
    threshold_reached: bool = False
    hit_step_cap: bool = False
    positive_seen_fraction: float = 0.0
    negative_seen_fraction: float = 0.0
    validation_history: tuple[dict[str, float], ...] = ()


@dataclass(frozen=True)
class OmniFoldIterationDiagnostics:
    iteration: int
    step1: RatioFitDiagnostics
    step2: RatioFitDiagnostics
    pull_min: float
    pull_max: float
    push_min: float
    push_max: float
    push_ess_fraction: float


@dataclass(frozen=True)
class OmniFoldResult:
    """Complete push/pull history and the two warm-started classifiers."""

    step1_model: ConditionalRatioMLP
    step2_model: ConditionalRatioMLP
    step2_snapshots: tuple[ConditionalRatioMLP, ...]
    pull_weights: tuple[Tensor, ...]
    push_weights: tuple[Tensor, ...]
    diagnostics: tuple[OmniFoldIterationDiagnostics, ...]
    log_weight_scale: float = 1.0
    discarded_last_iteration: bool = False
    stopped_early: bool = False

    @property
    def final_push_weights(self) -> Tensor:
        return self.push_weights[-1]


def _flatten_condition_sample(
    condition: Tensor, sample: Tensor
) -> tuple[Tensor, Tensor, torch.Size]:
    if condition.ndim < 2 or sample.ndim < 2:
        raise ValueError("condition and sample must each have a feature dimension")
    if condition.shape[-1] < 1 or sample.shape[-1] < 1:
        raise ValueError("feature dimensions must be non-empty")

    if condition.shape[:-1] == sample.shape[:-1]:
        expanded_condition = condition
    elif (
        sample.ndim == condition.ndim + 1
        and condition.shape[:-1] == sample.shape[:-2]
    ):
        expanded_condition = condition.unsqueeze(-2).expand(
            *sample.shape[:-1], condition.shape[-1]
        )
    else:
        raise ValueError(
            "condition/sample leading shapes must match, with at most one "
            "candidate axis"
        )
    output_shape = sample.shape[:-1]
    return (
        expanded_condition.reshape(-1, condition.shape[-1]),
        sample.reshape(-1, sample.shape[-1]),
        output_shape,
    )


def _flatten_population(
    condition: Tensor, sample: Tensor, weight: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    flat_condition, flat_sample, output_shape = _flatten_condition_sample(
        condition, sample
    )
    if weight.shape != output_shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match population shape "
            f"{tuple(output_shape)}"
        )
    return flat_condition, flat_sample, weight.reshape(-1)


def subsample_event_candidates(
    condition: Tensor,
    sample: Tensor,
    weight: Tensor,
    *,
    k: int,
    seed: int,
    epoch: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw ``k`` of ``K`` candidates per event and flatten (shared seed)."""
    if sample.ndim != 3:
        raise ValueError(
            f"candidate subsample needs (N, K, F) samples, got {tuple(sample.shape)}"
        )
    n_events, n_pool = int(sample.shape[0]), int(sample.shape[1])
    k = int(k)
    if not 1 <= k <= n_pool:
        raise ValueError(f"k={k} is outside 1..{n_pool}")
    if weight.shape[0] != n_events:
        raise ValueError("weight batch does not match candidate events")
    if weight.ndim == 1:
        weight = weight[:, None].expand(-1, n_pool)
    if tuple(weight.shape[:2]) != (n_events, n_pool):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match "
            f"({n_events}, {n_pool})"
        )
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 1_000_003 * int(epoch)
    )
    scores = torch.rand(n_events, n_pool, generator=generator)
    index = scores.topk(k, dim=1, largest=True).indices
    gather = index.unsqueeze(-1).expand(-1, -1, int(sample.shape[-1]))
    sample_k = torch.gather(sample.detach().cpu(), 1, gather)
    weight_k = torch.gather(weight.detach().cpu(), 1, index)
    condition_cpu = condition.detach().cpu()
    return _flatten_population(condition_cpu, sample_k, weight_k)


def global_mean_one_from_log_weights(log_weight: Tensor) -> Tensor:
    """Normalize a whole population with one stable scalar multiplier."""

    if log_weight.numel() == 0:
        raise ValueError("cannot normalize an empty population")
    if not torch.isfinite(log_weight).all():
        raise ValueError("log weights must be finite")
    log_scale = torch.logsumexp(log_weight.reshape(-1), dim=0)
    log_scale = log_scale - log_weight.new_tensor(float(log_weight.numel())).log()
    weight = torch.exp(log_weight - log_scale)
    # Logits stay unclipped; lift float underflow to a positive floor and
    # renormalize so Step-2 never sees exact zeros from a peaked classifier.
    floor = torch.finfo(weight.dtype).tiny
    if torch.any(weight <= 0.0):
        weight = torch.clamp(weight, min=weight.new_tensor(floor))
        weight = weight / weight.mean().clamp_min(floor)
    return weight


def global_mean_one(weight: Tensor) -> Tensor:
    """Normalize positive weights globally, never separately by condition/event."""

    if weight.numel() == 0:
        raise ValueError("cannot normalize an empty population")
    if not torch.isfinite(weight).all() or torch.any(weight < 0.0) or not torch.any(weight > 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if torch.any(weight == 0.0):
        weight = torch.clamp(weight, min=torch.finfo(weight.dtype).tiny)
    return global_mean_one_from_log_weights(weight.log())


def step1_pull_weights(previous_push: Tensor, data_over_sim_log_odds: Tensor) -> Tensor:
    """Apply ``pull = previous_push * (data / current_sim)`` globally."""

    _validate_update_inputs(previous_push, data_over_sim_log_odds)
    return global_mean_one_from_log_weights(
        previous_push.log() + data_over_sim_log_odds
    )


def step2_push_weights(
    previous_push: Tensor, pulled_over_previous_log_odds: Tensor
) -> Tensor:
    """Apply ``push = previous_push * (pulled_gen / previous_gen)`` globally."""

    _validate_update_inputs(previous_push, pulled_over_previous_log_odds)
    return global_mean_one_from_log_weights(
        previous_push.log() + pulled_over_previous_log_odds
    )


def _validate_update_inputs(previous_push: Tensor, log_odds: Tensor) -> None:
    if previous_push.shape != log_odds.shape:
        raise ValueError("previous weights and log odds must have identical shapes")
    if not torch.isfinite(previous_push).all() or torch.any(previous_push <= 0.0):
        raise ValueError("previous weights must be finite and strictly positive")
    if not torch.isfinite(log_odds).all():
        raise ValueError("log odds must be finite")


def draw_independent_class_indices(
    positive_size: int,
    negative_size: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Draw separate population batches; indices are never event-paired."""

    if min(positive_size, negative_size, batch_size, steps) < 1:
        raise ValueError("population sizes, batch size, and steps must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positive_index = torch.randint(
        positive_size, (steps, batch_size), generator=generator
    )
    negative_index = torch.randint(
        negative_size, (steps, batch_size), generator=generator
    )
    return positive_index, negative_index


def draw_epoch_shuffle_class_indices(
    positive_size: int,
    negative_size: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Cycle independent shuffled populations without replacement per epoch."""

    if min(positive_size, negative_size, batch_size, steps) < 1:
        raise ValueError("population sizes, batch size, and steps must be positive")

    def draw(size: int, draw_seed: int) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(draw_seed)
        required = steps * batch_size
        parts: list[Tensor] = []
        drawn = 0
        while drawn < required:
            permutation = torch.randperm(size, generator=generator)
            parts.append(permutation)
            drawn += size
        return torch.cat(parts)[:required].reshape(steps, batch_size)

    return draw(positive_size, seed), draw(negative_size, seed + 1)


class _EpochShuffleBatcher:
    """Yield one deterministic permutation per epoch, including its short last batch."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._epoch = -1
        self._permutation: Tensor | None = None

    def _permutation_for(self, epoch: int) -> Tensor:
        if self._permutation is None or self._epoch != epoch:
            generator = torch.Generator(device="cpu").manual_seed(
                self.seed + 1_000_003 * int(epoch)
            )
            self._permutation = torch.randperm(self.size, generator=generator)
            self._epoch = int(epoch)
        return self._permutation

    def draw(self, step: int) -> Tensor:
        steps_per_epoch = max(1, math.ceil(self.size / self.batch_size))
        epoch = int(step) // steps_per_epoch
        batch_in_epoch = int(step) % steps_per_epoch
        start = batch_in_epoch * self.batch_size
        stop = min(start + self.batch_size, self.size)
        return self._permutation_for(epoch)[start:stop]


@torch.no_grad()
def evaluate_density_ratio(
    model: ConditionalRatioMLP,
    positive_condition: Tensor,
    positive_sample: Tensor,
    positive_weight: Tensor,
    negative_condition: Tensor,
    negative_sample: Tensor,
    negative_weight: Tensor,
    *,
    batch_size: int,
) -> tuple[float, float]:
    """Evaluate balanced weighted BCE and accuracy on a fixed population.

    When a process group is live, each rank scores a disjoint slice and the
    weighted sums are all-reduced, so validation uses every GPU instead of
    repeating the full scan 16 times.
    """

    if batch_size < 1:
        raise ValueError("validation batch_size must be positive")
    pos_c, pos_z, pos_w = _flatten_population(
        positive_condition, positive_sample, positive_weight
    )
    neg_c, neg_z, neg_w = _flatten_population(
        negative_condition, negative_sample, negative_weight
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rank, world = distributed_context()
    pos_c = pos_c.to(device=device, dtype=dtype)
    pos_z = pos_z.to(device=device, dtype=dtype)
    neg_c = neg_c.to(device=device, dtype=dtype)
    neg_z = neg_z.to(device=device, dtype=dtype)
    pos_w = global_mean_one(pos_w.to(device=device, dtype=dtype))
    neg_w = global_mean_one(neg_w.to(device=device, dtype=dtype))

    def class_metrics(
        condition: Tensor,
        sample: Tensor,
        weight: Tensor,
        *,
        positive: bool,
    ) -> tuple[Tensor, Tensor]:
        n = int(len(sample))
        if n < 1:
            raise ValueError("validation class must be non-empty")
        shard_start = (n * rank) // world
        shard_stop = (n * (rank + 1)) // world
        loss_sum = torch.zeros((), device=device, dtype=dtype)
        accuracy_sum = torch.zeros((), device=device, dtype=dtype)
        for start in range(shard_start, shard_stop, batch_size):
            stop = min(start + batch_size, shard_stop)
            logits = model(condition[start:stop], sample[start:stop])
            weights = weight[start:stop]
            losses = F.softplus(-logits) if positive else F.softplus(logits)
            # A zero logit is an exact 50/50 prediction.  Give ties half credit
            # so the zero-output residual baseline reports chance accuracy rather
            # than zero accuracy for both classes.
            correct = (
                (logits > 0.0).to(dtype)
                + 0.5 * (logits == 0.0).to(dtype)
                if positive
                else (logits < 0.0).to(dtype)
                + 0.5 * (logits == 0.0).to(dtype)
            )
            loss_sum += (weights * losses).sum()
            accuracy_sum += (weights * correct).sum()
        _all_reduce_sum_tensor(loss_sum)
        _all_reduce_sum_tensor(accuracy_sum)
        return loss_sum / n, accuracy_sum / n

    was_training = model.training
    model.eval()
    pos_loss, pos_accuracy = class_metrics(pos_c, pos_z, pos_w, positive=True)
    neg_loss, neg_accuracy = class_metrics(neg_c, neg_z, neg_w, positive=False)
    model.train(was_training)
    return (
        float((0.5 * (pos_loss + neg_loss)).cpu()),
        float((0.5 * (pos_accuracy + neg_accuracy)).cpu()),
    )


def _clone_to_cpu(value: Any) -> Any:
    """Detach recovery payloads from accelerator storage and live objects."""

    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _as_module_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept a raw ``state_dict`` or the serialized classifier wrapper."""
    nested = payload.get("state_dict")
    if "condition_dim" in payload and isinstance(nested, dict):
        if any(
            key.startswith(prefix)
            for key in nested
            for prefix in ("sample_input.", "condition_encoder.", "network.")
        ):
            return nested
    return payload


def fit_density_ratio(
    model: ConditionalRatioMLP,
    positive_condition: Tensor,
    positive_sample: Tensor,
    positive_weight: Tensor,
    negative_condition: Tensor,
    negative_sample: Tensor,
    negative_weight: Tensor,
    config: RatioFitConfig,
    seed: int,
    validation: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None,
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    validation_evaluator: (
        Callable[[nn.Module], tuple[float, float, float]] | None
    ) = None,
    stop_when_validation_auc_gap_exceeds: float | None = None,
) -> RatioFitDiagnostics:
    """Fit balanced weighted classification on two independently sampled pools."""

    config.validate()
    if validation is not None and (
        config.validation_interval_steps < 1
        or config.validation_patience_evaluations < 1
        or config.validation_batch_size < 1
    ):
        raise ValueError("validation populations require positive validation controls")
    if stop_when_validation_auc_gap_exceeds is not None:
        threshold = float(stop_when_validation_auc_gap_exceeds)
        if not 0.0 <= threshold < 0.5:
            raise ValueError("validation AUC-gap stop threshold must lie in [0, 0.5)")
        if validation is None or validation_evaluator is None:
            raise ValueError(
                "validation AUC-gap stopping requires validation and an AUC evaluator"
            )
    pos_c, pos_z, pos_w = _flatten_population(
        positive_condition, positive_sample, positive_weight
    )
    resample_neg = config.train_candidates_per_event is not None
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rank, world = distributed_context()
    _broadcast_module(model)
    pos_c, pos_z = pos_c.to(device=device, dtype=dtype), pos_z.to(
        device=device, dtype=dtype
    )
    pos_w = global_mean_one(pos_w.to(device=device, dtype=dtype))
    negative_epoch_steps = 0
    current_neg_epoch = -1
    if resample_neg:
        if negative_sample.ndim != 3:
            raise ValueError(
                "train_candidates_per_event needs (N, K, F) negative samples"
            )
        k_train = int(config.train_candidates_per_event)
        n_neg_flat = int(negative_sample.shape[0]) * k_train
        negative_epoch_steps = max(1, math.ceil(n_neg_flat / int(config.batch_size)))
        neg_c = neg_z = neg_w = None

        def _materialize_negatives(epoch: int) -> None:
            nonlocal neg_c, neg_z, neg_w, negative_batcher, current_neg_epoch
            if int(epoch) == current_neg_epoch and neg_c is not None:
                return
            flat_c, flat_z, flat_w = subsample_event_candidates(
                negative_condition,
                negative_sample,
                negative_weight,
                k=k_train,
                seed=seed,
                epoch=int(epoch),
            )
            neg_c = flat_c.to(device=device, dtype=dtype)
            neg_z = flat_z.to(device=device, dtype=dtype)
            neg_w = global_mean_one(flat_w.to(device=device, dtype=dtype))
            if config.sampling == "independent_epoch_shuffle":
                negative_batcher = _EpochShuffleBatcher(
                    len(neg_z), config.batch_size, seed + 1 + 10_007 * int(epoch)
                )
            current_neg_epoch = int(epoch)

        negative_batcher = None
        _materialize_negatives(0)
    else:
        neg_c, neg_z, neg_w = _flatten_population(
            negative_condition, negative_sample, negative_weight
        )
        neg_c, neg_z = neg_c.to(device=device, dtype=dtype), neg_z.to(
            device=device, dtype=dtype
        )
        neg_w = global_mean_one(neg_w.to(device=device, dtype=dtype))

    if config.require_saturation and validation is None:
        raise ValueError("required saturation needs an independent validation population")
    if config.sampling == "independent_epoch_shuffle":
        positive_index = negative_index = None
    else:
        assert config.steps is not None
        positive_index, negative_index = draw_independent_class_indices(
            len(pos_z), len(neg_z), config.batch_size, config.steps, seed
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.train()
    final_loss = torch.zeros((), device=device, dtype=dtype)
    final_accuracy = torch.zeros((), device=device, dtype=dtype)
    steps_completed = 0
    best_step: int | None = None
    best_validation_loss = float("inf")
    best_validation_accuracy: float | None = None
    best_validation_auc: float | None = None
    best_training_loss: float | None = None
    best_training_accuracy: float | None = None
    best_state: dict[str, Tensor] | None = None
    patience_validation_loss = float("inf")
    evaluations_without_improvement = 0
    validation_history: list[dict[str, float]] = []
    last_validation_loss = float("nan")
    last_validation_accuracy = float("nan")
    last_validation_auc = float("nan")
    saturated = False
    threshold_reached = False
    threshold_state: dict[str, Tensor] | None = None
    threshold_training_loss: float | None = None
    threshold_training_accuracy: float | None = None
    initial_validation_loss: float | None = None
    initial_validation_accuracy: float | None = None
    initial_validation_auc: float | None = None

    def run_validation() -> tuple[float, float, float]:
        if validation_evaluator is not None:
            loss_value, accuracy_value, auc_value = validation_evaluator(model)
            return float(loss_value), float(accuracy_value), float(auc_value)
        assert validation is not None
        loss_value, accuracy_value = evaluate_density_ratio(
            model, *validation, batch_size=config.validation_batch_size
        )
        return float(loss_value), float(accuracy_value), float("nan")

    if validation is not None and resume_state is None:
        # Preserve the pre-training no-op model.  Fresh EveNet residual heads are
        # zero-output initialized, so this is the exact balanced null classifier
        # with BCE log(2).  It must remain eligible even while ``min_steps`` keeps
        # early stopping from firing during the warm-up epoch.
        (
            initial_validation_loss,
            initial_validation_accuracy,
            initial_validation_auc,
        ) = run_validation()
        best_step = 0
        best_validation_loss = float(initial_validation_loss)
        best_validation_accuracy = float(initial_validation_accuracy)
        best_validation_auc = float(initial_validation_auc)
        best_training_loss = float(initial_validation_loss)
        best_training_accuracy = float(initial_validation_accuracy)
        best_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        patience_validation_loss = float(initial_validation_loss)
        last_validation_loss = float(initial_validation_loss)
        last_validation_accuracy = float(initial_validation_accuracy)
        last_validation_auc = float(initial_validation_auc)
        initial_row = {
            "step": 0.0,
            "loss": float(initial_validation_loss),
            "balanced_accuracy": float(initial_validation_accuracy),
        }
        if math.isfinite(initial_validation_auc):
            initial_row["auc"] = float(initial_validation_auc)
            initial_row["auc_gap"] = abs(float(initial_validation_auc) - 0.5)
        validation_history.append(initial_row)
    if resume_state is not None:
        if int(resume_state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported density-ratio recovery checkpoint schema")
        if int(resume_state.get("seed", -1)) != seed:
            raise ValueError("density-ratio recovery checkpoint seed mismatch")
        if resume_state.get("fit_config") != asdict(config):
            raise ValueError("density-ratio recovery checkpoint configuration mismatch")
        model.load_state_dict(resume_state["model_state"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        steps_completed = int(resume_state["steps_completed"])
        if steps_completed < 0 or (
            config.steps is not None and steps_completed > config.steps
        ):
            raise ValueError("invalid completed-step cursor in recovery checkpoint")
        best_step = resume_state["best_step"]
        best_validation_loss = float(resume_state["best_validation_loss"])
        best_validation_accuracy = resume_state["best_validation_accuracy"]
        best_validation_auc = resume_state.get("best_validation_auc")
        best_training_loss = resume_state["best_training_loss"]
        best_training_accuracy = resume_state["best_training_accuracy"]
        best_state = resume_state["best_state"]
        patience_validation_loss = float(
            resume_state.get("patience_validation_loss", best_validation_loss)
        )
        evaluations_without_improvement = int(
            resume_state["evaluations_without_improvement"]
        )
        validation_history = [dict(item) for item in resume_state["validation_history"]]
        if validation_history:
            initial_validation_loss = float(validation_history[0]["loss"])
            initial_validation_accuracy = float(
                validation_history[0]["balanced_accuracy"]
            )
            if "auc" in validation_history[0]:
                initial_validation_auc = float(validation_history[0]["auc"])
            last_validation_loss = float(validation_history[-1]["loss"])
            last_validation_accuracy = float(
                validation_history[-1]["balanced_accuracy"]
            )
            if "auc" in validation_history[-1]:
                last_validation_auc = float(validation_history[-1]["auc"])
        saturated = bool(resume_state["saturated"])
        threshold_reached = bool(resume_state.get("threshold_reached", False))
        threshold_state = resume_state.get("threshold_state")
        threshold_training_loss = resume_state.get("threshold_training_loss")
        threshold_training_accuracy = resume_state.get(
            "threshold_training_accuracy"
        )
        final_loss = torch.tensor(
            float(resume_state["final_loss"]), device=device, dtype=dtype
        )
        final_accuracy = torch.tensor(
            float(resume_state["final_accuracy"]), device=device, dtype=dtype
        )

    if not resample_neg:
        negative_batcher = None
    positive_batcher = None
    if config.sampling == "independent_epoch_shuffle":
        positive_batcher = _EpochShuffleBatcher(
            len(pos_z), config.batch_size, seed
        )
        if not resample_neg:
            negative_batcher = _EpochShuffleBatcher(
                len(neg_z), config.batch_size, seed + 1
            )
        elif steps_completed > 0:
            _materialize_negatives(steps_completed // negative_epoch_steps)

    def recovery_state() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "seed": seed,
            "fit_config": asdict(config),
            "model_state": _clone_to_cpu(model.state_dict()),
            "optimizer_state": _clone_to_cpu(optimizer.state_dict()),
            "steps_completed": steps_completed,
            "best_step": best_step,
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            "best_validation_auc": best_validation_auc,
            "best_training_loss": best_training_loss,
            "best_training_accuracy": best_training_accuracy,
            "best_state": _clone_to_cpu(best_state),
            "patience_validation_loss": patience_validation_loss,
            "evaluations_without_improvement": evaluations_without_improvement,
            "validation_history": copy.deepcopy(validation_history),
            "saturated": saturated,
            "threshold_reached": threshold_reached,
            "threshold_state": _clone_to_cpu(threshold_state),
            "threshold_training_loss": threshold_training_loss,
            "threshold_training_accuracy": threshold_training_accuracy,
            "final_loss": float(final_loss.cpu()),
            "final_accuracy": float(final_accuracy.cpu()),
        }

    step = steps_completed
    while (
        not saturated
        and not threshold_reached
        and (config.steps is None or step < config.steps)
    ):
        if config.sampling == "independent_epoch_shuffle":
            assert positive_batcher is not None and negative_batcher is not None
            pos_i = positive_batcher.draw(step).to(device)
            if resample_neg:
                _materialize_negatives(step // negative_epoch_steps)
                neg_i = negative_batcher.draw(step % negative_epoch_steps).to(device)
            else:
                neg_i = negative_batcher.draw(step).to(device)
        else:
            assert positive_index is not None and negative_index is not None
            pos_i = positive_index[step].to(device)
            neg_i = negative_index[step].to(device)
        pos_i = shard_global_batch(pos_i, rank, world)
        neg_i = shard_global_batch(neg_i, rank, world)
        if int(pos_i.numel()) != int(neg_i.numel()):
            raise RuntimeError(
                "balanced ratio training needs equal local truth/gen batch sizes"
            )
        local_rows = int(pos_i.numel())
        microbatch_rows = min(
            local_rows,
            int(config.train_microbatch_size_per_rank or local_rows),
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=device, dtype=dtype)
        pos_accuracy = torch.zeros((), device=device, dtype=dtype)
        neg_accuracy = torch.zeros((), device=device, dtype=dtype)
        for micro_start in range(0, local_rows, microbatch_rows):
            micro_stop = min(micro_start + microbatch_rows, local_rows)
            micro_fraction = float(micro_stop - micro_start) / float(local_rows)
            pos_micro_i = pos_i[micro_start:micro_stop]
            neg_micro_i = neg_i[micro_start:micro_stop]
            pos_logit = model(pos_c[pos_micro_i], pos_z[pos_micro_i])
            neg_logit = model(neg_c[neg_micro_i], neg_z[neg_micro_i])
            micro_loss = 0.5 * (
                (pos_w[pos_micro_i] * F.softplus(-pos_logit)).mean()
                + (neg_w[neg_micro_i] * F.softplus(neg_logit)).mean()
            )
            (micro_loss * micro_fraction).backward()
            with torch.no_grad():
                loss += micro_loss.detach() * micro_fraction
                pos_credit = (pos_logit > 0.0).to(dtype) + 0.5 * (
                    pos_logit == 0.0
                ).to(dtype)
                neg_credit = (neg_logit < 0.0).to(dtype) + 0.5 * (
                    neg_logit == 0.0
                ).to(dtype)
                pos_accuracy += (
                    pos_w[pos_micro_i] * pos_credit
                ).mean() * micro_fraction
                neg_accuracy += (
                    neg_w[neg_micro_i] * neg_credit
                ).mean() * micro_fraction
        _average_gradients(model)
        optimizer.step()
        with torch.no_grad():
            final_loss = _reduce_mean(loss.detach())
            final_accuracy = _reduce_mean(0.5 * (pos_accuracy + neg_accuracy))
        steps_completed = step + 1
        hit_configured_last_step = bool(
            config.steps is not None and steps_completed == config.steps
        )
        should_validate = validation is not None and (
            steps_completed % config.validation_interval_steps == 0
            or hit_configured_last_step
        )
        if should_validate:
            validation_loss, validation_accuracy, validation_auc = run_validation()
            last_validation_loss = float(validation_loss)
            last_validation_accuracy = float(validation_accuracy)
            last_validation_auc = float(validation_auc)
            validation_row = {
                "step": float(steps_completed),
                "loss": validation_loss,
                "balanced_accuracy": validation_accuracy,
            }
            if math.isfinite(validation_auc):
                validation_row["auc"] = validation_auc
                validation_row["auc_gap"] = abs(validation_auc - 0.5)
            validation_history.append(validation_row)
            # Always retain the numerically best validation checkpoint.  The
            # min-delta tolerance belongs only to early-stop patience; coupling
            # it to checkpoint selection can restore the initial null classifier
            # after a real but weak improvement and force the held-out AUC to 0.5.
            checkpoint_improved = validation_loss < best_validation_loss
            patience_improved = (
                validation_loss
                < patience_validation_loss - config.validation_min_delta
            )
            if checkpoint_improved:
                best_validation_loss = validation_loss
                best_validation_accuracy = validation_accuracy
                best_validation_auc = (
                    validation_auc if math.isfinite(validation_auc) else None
                )
                best_training_loss = float(final_loss.cpu())
                best_training_accuracy = float(final_accuracy.cpu())
                best_step = steps_completed
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            if patience_improved:
                patience_validation_loss = validation_loss
            checkpoint_eligible = steps_completed >= config.min_steps
            if checkpoint_eligible:
                if patience_improved:
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1
                if (
                    evaluations_without_improvement
                    >= config.validation_patience_evaluations
                ):
                    saturated = True
            saturated = _broadcast_flag(saturated, device)
            if (
                stop_when_validation_auc_gap_exceeds is not None
                and math.isfinite(validation_auc)
                and abs(validation_auc - 0.5)
                > float(stop_when_validation_auc_gap_exceeds)
            ):
                threshold_reached = True
                threshold_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
                threshold_training_loss = float(final_loss.cpu())
                threshold_training_accuracy = float(final_accuracy.cpu())
            threshold_reached = _broadcast_flag(threshold_reached, device)
        should_report_progress = progress_callback is not None and (
            (
                config.progress_interval_steps > 0
                and steps_completed % config.progress_interval_steps == 0
            )
            or (
                config.progress_interval_steps <= 0
                and should_validate
            )
            or saturated
            or threshold_reached
            or hit_configured_last_step
        )
        if should_report_progress:
            progress_row = {
                "step": float(steps_completed),
                "training_loss": float(final_loss.cpu()),
                "training_balanced_accuracy": float(final_accuracy.cpu()),
                "saturated": float(saturated),
            }
            if math.isfinite(last_validation_loss):
                progress_row["validation_loss"] = last_validation_loss
            if math.isfinite(last_validation_accuracy):
                progress_row["validation_balanced_accuracy"] = (
                    last_validation_accuracy
                )
            if math.isfinite(last_validation_auc):
                progress_row["validation_auc"] = last_validation_auc
                progress_row["validation_auc_gap"] = abs(
                    last_validation_auc - 0.5
                )
            progress_row["threshold_reached"] = float(threshold_reached)
            if math.isfinite(best_validation_loss):
                progress_row["best_validation_loss"] = best_validation_loss
            progress_callback(progress_row)
        should_checkpoint = checkpoint_callback is not None and (
            should_validate
            or hit_configured_last_step
            or (
                config.checkpoint_interval_steps > 0
                and steps_completed % config.checkpoint_interval_steps == 0
            )
        )
        if should_checkpoint:
            checkpoint_callback(recovery_state())
        if saturated or threshold_reached:
            break
        step += 1
    if validation is not None and best_state is None:
        raise RuntimeError("density-ratio validation produced no finite checkpoint")
    if threshold_reached and threshold_state is not None:
        model.load_state_dict(threshold_state, strict=True)
        final_loss = torch.tensor(
            threshold_training_loss, device=device, dtype=dtype
        )
        final_accuracy = torch.tensor(
            threshold_training_accuracy, device=device, dtype=dtype
        )
    elif best_state is not None and config.restore_best:
        model.load_state_dict(best_state, strict=True)
        final_loss = torch.tensor(best_training_loss, device=device, dtype=dtype)
        final_accuracy = torch.tensor(
            best_training_accuracy, device=device, dtype=dtype
        )
    hit_step_cap = bool(
        validation is not None
        and config.steps is not None
        and not saturated
        and not threshold_reached
    )
    positive_seen_fraction = min(
        1.0, steps_completed * config.batch_size / float(len(pos_z))
    )
    negative_seen_fraction = min(
        1.0, steps_completed * config.batch_size / float(len(neg_z))
    )
    model.eval()
    return RatioFitDiagnostics(
        loss=float(final_loss.cpu()),
        balanced_accuracy=float(final_accuracy.cpu()),
        steps=config.steps,
        steps_completed=steps_completed,
        best_step=best_step,
        initial_validation_loss=initial_validation_loss,
        initial_validation_balanced_accuracy=initial_validation_accuracy,
        initial_validation_auc=initial_validation_auc,
        validation_loss=(
            None if validation is None else float(best_validation_loss)
        ),
        validation_balanced_accuracy=best_validation_accuracy,
        validation_auc=(
            last_validation_auc if threshold_reached else best_validation_auc
        ),
        saturated=saturated,
        threshold_reached=threshold_reached,
        hit_step_cap=hit_step_cap,
        positive_seen_fraction=positive_seen_fraction,
        negative_seen_fraction=negative_seen_fraction,
        validation_history=tuple(validation_history),
    )


def fit_step1(
    model: ConditionalRatioMLP,
    data_condition: Tensor,
    data_sample: Tensor,
    gen_condition: Tensor,
    gen_sample: Tensor,
    previous_push: Tensor,
    config: RatioFitConfig,
    seed: int,
    validation: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None,
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> tuple[Tensor, RatioFitDiagnostics]:
    """Fit Data/current-Sim odds and transport them back to the Gen pool."""

    data_weight = torch.ones(
        data_sample.shape[:-1], device=data_sample.device, dtype=data_sample.dtype
    )
    diagnostics = fit_density_ratio(
        model,
        data_condition,
        data_sample,
        data_weight,
        gen_condition,
        gen_sample,
        previous_push,
        config,
        seed,
        validation,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
        progress_callback=progress_callback,
    )
    log_odds = score_reward(model, gen_condition, gen_sample)
    return step1_pull_weights(previous_push.to(log_odds), log_odds), diagnostics


def fit_step2(
    model: ConditionalRatioMLP,
    gen_condition: Tensor,
    gen_sample: Tensor,
    pulled_weight: Tensor,
    previous_push: Tensor,
    config: RatioFitConfig,
    seed: int,
    validation: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None = None,
    *,
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
    tempering: float = 1.0,
) -> tuple[Tensor, RatioFitDiagnostics]:
    """Fit pulled-Gen/previous-Gen odds and update the persistent Gen weights."""

    if not math.isfinite(float(tempering)) or float(tempering) <= 0.0:
        raise ValueError("tempering must be a positive finite scale")
    diagnostics = fit_density_ratio(
        model,
        gen_condition,
        gen_sample,
        pulled_weight,
        gen_condition,
        gen_sample,
        previous_push,
        config,
        seed,
        validation,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
        progress_callback=progress_callback,
    )
    log_odds = score_reward(model, gen_condition, gen_sample)
    return (
        step2_push_weights(previous_push.to(log_odds), float(tempering) * log_odds),
        diagnostics,
    )


def _ratio_diagnostics_from_dict(payload: dict[str, Any]) -> RatioFitDiagnostics:
    values = dict(payload)
    values["validation_history"] = tuple(values.get("validation_history", ()))
    return RatioFitDiagnostics(**values)


def _iteration_diagnostics_from_dict(
    payload: dict[str, Any],
) -> OmniFoldIterationDiagnostics:
    values = dict(payload)
    values["step1"] = _ratio_diagnostics_from_dict(values["step1"])
    values["step2"] = _ratio_diagnostics_from_dict(values["step2"])
    return OmniFoldIterationDiagnostics(**values)


def run_full_omnifold_identity(
    data_condition: Tensor,
    data_sample: Tensor,
    gen_condition: Tensor,
    gen_sample: Tensor,
    iterations: int,
    fit_config: RatioFitConfig,
    hidden_dim: int = 64,
    hidden_layers: int = 3,
    seed: int = 0,
    initial_push: Tensor | None = None,
    validation_data_condition: Tensor | None = None,
    validation_data_sample: Tensor | None = None,
    validation_gen_condition: Tensor | None = None,
    validation_gen_sample: Tensor | None = None,
    *,
    feature_mode: str = "raw",
    architecture: str = "film",
    resume_state: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    tempering: float = 1.0,
    ess_floor: float = 0.0,
    min_iterations: int = 1,
    require_final_closure: bool = False,
    closure_callback: Callable[[dict[str, Any]], bool] | None = None,
    initial_step1_state: dict[str, Any] | None = None,
    initial_step2_state: dict[str, Any] | None = None,
) -> OmniFoldResult:
    """Run the complete paper recurrence for an identity Gen-to-Sim response.

    A fixed Gen sample is also its matched Sim image.  Step 1 reweights the Sim
    population to Data and pulls those instance weights back through that fixed
    identity pairing.  Step 2 turns the pulled instance weights into a valid Gen
    weighting function relative to the previous Gen weights.  The resulting
    push weights are then used by Step 1 of the next outer iteration.

    The Step-1 and Step-2 networks are distinct.  Each is constructed once and
    warm-started across outer iterations, as in the paper implementation.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not math.isfinite(float(tempering)) or float(tempering) <= 0.0:
        raise ValueError("tempering must be a positive finite scale")
    if not math.isfinite(float(ess_floor)) or float(ess_floor) < 0.0:
        raise ValueError("ess_floor must be a nonnegative finite value")
    if not 1 <= int(min_iterations) <= int(iterations):
        raise ValueError("min_iterations must lie between 1 and iterations")
    if data_condition.shape[-1] != gen_condition.shape[-1]:
        raise ValueError("Data and Gen condition dimensions must match")
    if data_sample.shape[-1] != gen_sample.shape[-1]:
        raise ValueError("Data and Gen sample dimensions must match")
    expected_weight_shape = gen_sample.shape[:-1]
    if initial_push is None:
        previous_push = torch.ones(
            expected_weight_shape, device=gen_sample.device, dtype=gen_sample.dtype
        )
    else:
        if initial_push.shape != expected_weight_shape:
            raise ValueError("initial_push shape must match the Gen population")
        previous_push = initial_push.to(gen_sample)
    previous_push = global_mean_one(previous_push)
    validation_values = (
        validation_data_condition,
        validation_data_sample,
        validation_gen_condition,
        validation_gen_sample,
    )
    validation_enabled = any(value is not None for value in validation_values)
    if validation_enabled and not all(value is not None for value in validation_values):
        raise ValueError("OmniFold validation requires all four validation populations")
    validation_previous_push = None
    if validation_enabled:
        assert validation_gen_sample is not None
        validation_previous_push = torch.ones(
            validation_gen_sample.shape[:-1],
            device=validation_gen_sample.device,
            dtype=validation_gen_sample.dtype,
        )

    step1_model = make_ratio_mlp(
        condition_dim=gen_condition.shape[-1],
        sample_dim=gen_sample.shape[-1],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        feature_mode=feature_mode,
        architecture=architecture,
        seed=seed,
        device=gen_sample.device,
    )
    step2_model = make_ratio_mlp(
        condition_dim=gen_condition.shape[-1],
        sample_dim=gen_sample.shape[-1],
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        feature_mode=feature_mode,
        architecture=architecture,
        seed=seed + 1,
        device=gen_sample.device,
    )
    if initial_step1_state is not None:
        step1_model.load_state_dict(_as_module_state_dict(initial_step1_state), strict=True)
        step1_model.train()
        for parameter in step1_model.parameters():
            parameter.requires_grad_(True)
    if initial_step2_state is not None:
        step2_model.load_state_dict(_as_module_state_dict(initial_step2_state), strict=True)
        step2_model.train()
        for parameter in step2_model.parameters():
            parameter.requires_grad_(True)
    _broadcast_module(step1_model)
    _broadcast_module(step2_model)

    pull_history: list[Tensor] = []
    push_history: list[Tensor] = []
    diagnostics: list[OmniFoldIterationDiagnostics] = []
    step2_snapshots: list[ConditionalRatioMLP] = []
    resume_phase = "step1"
    resume_iteration = 1
    active_fit_state: dict[str, Any] | None = None
    resumed_pulled: Tensor | None = None
    resumed_step1_diagnostics: RatioFitDiagnostics | None = None
    discarded_last_iteration = False
    stopped_early = False
    last_closure_increments = -1
    accepted_step1_state = _clone_to_cpu(step1_model.state_dict())
    accepted_step2_state = _clone_to_cpu(step2_model.state_dict())
    accepted_validation_previous_push = _clone_to_cpu(validation_previous_push)
    if resume_state is not None:
        if int(resume_state.get("schema_version", -1)) != 1:
            raise ValueError("unsupported OmniFold recurrence recovery schema")
        expected_contract = {
            "iterations": iterations,
            "fit_config": asdict(fit_config),
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "architecture": architecture,
            "feature_mode": feature_mode,
            "seed": seed,
            "tempering": float(tempering),
            "ess_floor": float(ess_floor),
            "min_iterations": int(min_iterations),
            "require_final_closure": bool(require_final_closure),
            "closure_enabled": closure_callback is not None,
        }
        if resume_state.get("contract") != expected_contract:
            raise ValueError("OmniFold recurrence recovery contract mismatch")
        step1_model.load_state_dict(resume_state["step1_model_state"], strict=True)
        step2_model.load_state_dict(resume_state["step2_model_state"], strict=True)
        previous_push = resume_state["previous_push"].to(gen_sample)
        if validation_enabled:
            assert validation_gen_sample is not None
            validation_previous_push = resume_state[
                "validation_previous_push"
            ].to(validation_gen_sample)
        pull_history = [item.to(gen_sample) for item in resume_state["pull_history"]]
        push_history = [item.to(gen_sample) for item in resume_state["push_history"]]
        diagnostics = [
            _iteration_diagnostics_from_dict(item)
            for item in resume_state["diagnostics"]
        ]
        for model_state in resume_state["step2_snapshot_states"]:
            snapshot = make_ratio_mlp(
                condition_dim=gen_condition.shape[-1],
                sample_dim=gen_sample.shape[-1],
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                feature_mode=feature_mode,
                architecture=architecture,
                seed=seed + 1,
                device=gen_sample.device,
            )
            snapshot.load_state_dict(model_state, strict=True)
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
            step2_snapshots.append(snapshot)
        resume_phase = str(resume_state["phase"])
        resume_iteration = int(resume_state["iteration"])
        if resume_phase == "iteration_complete":
            resume_iteration += 1
            resume_phase = "step1"
        elif resume_phase == "step2":
            resumed_pulled = resume_state["current_pulled"].to(gen_sample)
            resumed_step1_diagnostics = _ratio_diagnostics_from_dict(
                resume_state["current_step1_diagnostics"]
            )
            active_fit_state = resume_state.get("active_fit_state")
        elif resume_phase == "step1":
            active_fit_state = resume_state.get("active_fit_state")
        else:
            raise ValueError("invalid OmniFold recurrence recovery phase")
        if not 1 <= resume_iteration <= iterations + 1:
            raise ValueError("invalid OmniFold recurrence recovery iteration")
        accepted_step1_state = _clone_to_cpu(step1_model.state_dict())
        accepted_step2_state = _clone_to_cpu(step2_model.state_dict())
        accepted_validation_previous_push = _clone_to_cpu(validation_previous_push)

    contract = {
        "iterations": iterations,
        "fit_config": asdict(fit_config),
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "architecture": architecture,
        "feature_mode": feature_mode,
        "seed": seed,
        "tempering": float(tempering),
        "ess_floor": float(ess_floor),
        "min_iterations": int(min_iterations),
        "require_final_closure": bool(require_final_closure),
        "closure_enabled": closure_callback is not None,
    }

    def recurrence_state(
        phase: str,
        iteration: int,
        *,
        active: dict[str, Any] | None = None,
        current_pulled: Tensor | None = None,
        current_step1: RatioFitDiagnostics | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "contract": contract,
            "phase": phase,
            "iteration": iteration,
            "step1_model_state": _clone_to_cpu(step1_model.state_dict()),
            "step2_model_state": _clone_to_cpu(step2_model.state_dict()),
            "previous_push": _clone_to_cpu(previous_push),
            "validation_previous_push": _clone_to_cpu(validation_previous_push),
            "pull_history": _clone_to_cpu(pull_history),
            "push_history": _clone_to_cpu(push_history),
            "diagnostics": [asdict(item) for item in diagnostics],
            "step2_snapshot_states": [
                _clone_to_cpu(model.state_dict()) for model in step2_snapshots
            ],
            "active_fit_state": _clone_to_cpu(active),
            "current_pulled": _clone_to_cpu(current_pulled),
            "current_step1_diagnostics": (
                None if current_step1 is None else asdict(current_step1)
            ),
        }

    for iteration in range(resume_iteration, iterations + 1):
        step1_validation = None
        if validation_enabled:
            assert validation_data_condition is not None
            assert validation_data_sample is not None
            assert validation_gen_condition is not None
            assert validation_gen_sample is not None
            assert validation_previous_push is not None
            step1_validation = (
                validation_data_condition,
                validation_data_sample,
                torch.ones_like(validation_data_sample[..., 0]),
                validation_gen_condition,
                validation_gen_sample,
                validation_previous_push,
            )
        if iteration == resume_iteration and resume_phase == "step2":
            assert resumed_pulled is not None
            assert resumed_step1_diagnostics is not None
            pulled = resumed_pulled
            step1_diagnostics = resumed_step1_diagnostics
        else:
            def save_step1(fit_state: dict[str, Any]) -> None:
                if checkpoint_callback is not None:
                    checkpoint_callback(recurrence_state(
                        "step1", iteration, active=fit_state
                    ))

            def report_step1(metrics: dict[str, float]) -> None:
                if progress_callback is not None:
                    progress_callback({
                        "iteration": iteration, "phase": "step1", **metrics
                    })

            pulled, step1_diagnostics = fit_step1(
                step1_model,
                data_condition,
                data_sample,
                gen_condition,
                gen_sample,
                previous_push,
                fit_config,
                seed + 1_000 * iteration + 1,
                step1_validation,
                resume_state=(
                    active_fit_state
                    if iteration == resume_iteration and resume_phase == "step1"
                    else None
                ),
                checkpoint_callback=save_step1,
                progress_callback=report_step1,
            )
        if fit_config.require_saturation and not step1_diagnostics.saturated:
            raise RuntimeError(
                f"OmniFold iteration {iteration} Step 1 did not saturate before the step cap"
            )
        validation_pulled = None
        if validation_enabled:
            assert validation_gen_condition is not None
            assert validation_gen_sample is not None
            assert validation_previous_push is not None
            validation_step1_log_odds = score_reward(
                step1_model, validation_gen_condition, validation_gen_sample
            )
            validation_pulled = step1_pull_weights(
                validation_previous_push.to(validation_step1_log_odds),
                validation_step1_log_odds,
            )
        step2_validation = None
        if validation_enabled:
            assert validation_gen_condition is not None
            assert validation_gen_sample is not None
            assert validation_pulled is not None
            assert validation_previous_push is not None
            step2_validation = (
                validation_gen_condition,
                validation_gen_sample,
                validation_pulled,
                validation_gen_condition,
                validation_gen_sample,
                validation_previous_push,
            )
        if checkpoint_callback is not None:
            checkpoint_callback(recurrence_state(
                "step2",
                iteration,
                active=(
                    active_fit_state
                    if iteration == resume_iteration and resume_phase == "step2"
                    else None
                ),
                current_pulled=pulled,
                current_step1=step1_diagnostics,
            ))

        def save_step2(fit_state: dict[str, Any]) -> None:
            if checkpoint_callback is not None:
                checkpoint_callback(recurrence_state(
                    "step2",
                    iteration,
                    active=fit_state,
                    current_pulled=pulled,
                    current_step1=step1_diagnostics,
                ))

        def report_step2(metrics: dict[str, float]) -> None:
            if progress_callback is not None:
                progress_callback({
                    "iteration": iteration, "phase": "step2", **metrics
                })

        pushed, step2_diagnostics = fit_step2(
            step2_model,
            gen_condition,
            gen_sample,
            pulled,
            previous_push,
            fit_config,
            seed + 1_000 * iteration + 2,
            step2_validation,
            resume_state=(
                active_fit_state
                if iteration == resume_iteration and resume_phase == "step2"
                else None
            ),
            checkpoint_callback=save_step2,
            progress_callback=report_step2,
            tempering=float(tempering),
        )
        if fit_config.require_saturation and not step2_diagnostics.saturated:
            raise RuntimeError(
                f"OmniFold iteration {iteration} Step 2 did not saturate before the step cap"
            )
        if validation_enabled:
            assert validation_gen_condition is not None
            assert validation_gen_sample is not None
            assert validation_previous_push is not None
            validation_step2_log_odds = score_reward(
                step2_model, validation_gen_condition, validation_gen_sample
            )
            validation_previous_push = step2_push_weights(
                validation_previous_push.to(validation_step2_log_odds),
                float(tempering) * validation_step2_log_odds,
            ).detach()
        pulled_snapshot = pulled.detach().clone()
        pushed_snapshot = pushed.detach().clone()
        frozen_increment = copy.deepcopy(step2_model).eval()
        for parameter in frozen_increment.parameters():
            parameter.requires_grad_(False)
        flat_push = pushed_snapshot.reshape(-1)
        ess_fraction = flat_push.sum().square() / (
            flat_push.numel() * flat_push.square().sum()
        )
        ess_value = float(ess_fraction.cpu())
        if float(ess_floor) > 0.0 and ess_value < float(ess_floor):
            if iteration == 1 or not step2_snapshots:
                raise OmniFoldPassOneESSError(ess_value, float(ess_floor))
            discarded_last_iteration = True
            step1_model.load_state_dict(accepted_step1_state, strict=True)
            step2_model.load_state_dict(accepted_step2_state, strict=True)
            if validation_enabled:
                validation_previous_push = accepted_validation_previous_push
            previous_push = push_history[-1]
            active_fit_state = None
            resume_phase = "step1"
            if checkpoint_callback is not None:
                checkpoint_callback(recurrence_state("iteration_complete", iteration - 1))
            break
        step2_snapshots.append(frozen_increment)
        pull_history.append(pulled_snapshot)
        push_history.append(pushed_snapshot)
        diagnostics.append(
            OmniFoldIterationDiagnostics(
                iteration=iteration,
                step1=step1_diagnostics,
                step2=step2_diagnostics,
                pull_min=float(pulled_snapshot.min().cpu()),
                pull_max=float(pulled_snapshot.max().cpu()),
                push_min=float(pushed_snapshot.min().cpu()),
                push_max=float(pushed_snapshot.max().cpu()),
                push_ess_fraction=ess_value,
            )
        )
        previous_push = pushed_snapshot
        accepted_step1_state = _clone_to_cpu(step1_model.state_dict())
        accepted_step2_state = _clone_to_cpu(step2_model.state_dict())
        if validation_enabled:
            accepted_validation_previous_push = _clone_to_cpu(validation_previous_push)
        active_fit_state = None
        resume_phase = "step1"
        if checkpoint_callback is not None:
            checkpoint_callback(recurrence_state("iteration_complete", iteration))
        if closure_callback is not None and iteration >= int(min_iterations):
            closed = bool(
                closure_callback(
                    {
                        "iteration": iteration,
                        "step2_snapshots": list(step2_snapshots),
                        "log_weight_scale": float(tempering),
                        "push_weights": pushed_snapshot,
                        "step1_model": step1_model,
                        "step2_model": step2_model,
                        "diagnostics": tuple(diagnostics),
                    }
                )
            )
            closed = _broadcast_flag(closed, gen_sample.device)
            last_closure_increments = len(step2_snapshots)
            if closed:
                stopped_early = True
                break

    if not step2_snapshots:
        raise RuntimeError("OmniFold fit produced no accepted Step-2 increments")
    if (
        require_final_closure
        and closure_callback is not None
        and not stopped_early
        and not discarded_last_iteration
        and last_closure_increments == len(step2_snapshots)
    ):
        # The in-loop check already scored this exact state and it was not closed.
        raise RuntimeError(
            "OmniFold closure still failing after the maximum outer passes"
        )
    if (
        require_final_closure
        and closure_callback is not None
        and not stopped_early
        and not discarded_last_iteration
    ):
        closed = bool(
            closure_callback(
                {
                    "iteration": len(step2_snapshots),
                    "step2_snapshots": list(step2_snapshots),
                    "log_weight_scale": float(tempering),
                    "push_weights": push_history[-1],
                    "step1_model": step1_model,
                    "step2_model": step2_model,
                    "diagnostics": tuple(diagnostics),
                }
            )
        )
        closed = _broadcast_flag(closed, gen_sample.device)
        if not closed:
            raise RuntimeError(
                "OmniFold closure still failing after the maximum outer passes"
            )

    return OmniFoldResult(
        step1_model=step1_model,
        step2_model=step2_model,
        step2_snapshots=tuple(step2_snapshots),
        pull_weights=tuple(pull_history),
        push_weights=tuple(push_history),
        diagnostics=tuple(diagnostics),
        log_weight_scale=float(tempering),
        discarded_last_iteration=discarded_last_iteration,
        stopped_early=stopped_early,
    )


@torch.no_grad()
def score_cumulative_reward(
    result: OmniFoldResult, condition: Tensor, sample: Tensor
) -> Tensor:
    """Evaluate ``log nu_n`` by summing every frozen Step-2 increment."""

    if not result.step2_snapshots:
        raise ValueError("OmniFold result contains no Step-2 increments")
    scores = [model(condition, sample) for model in result.step2_snapshots]
    return float(result.log_weight_scale) * torch.stack(scores, dim=0).sum(dim=0)


@torch.no_grad()
def score_reward(
    model: ConditionalRatioMLP, condition: Tensor, sample: Tensor
) -> Tensor:
    """Return a population log-density-ratio reward from only ``(c, z)``."""

    was_training = model.training
    model.eval()
    score = model(condition, sample)
    model.train(was_training)
    return score


def make_ratio_mlp(
    condition_dim: int,
    sample_dim: int,
    hidden_dim: int = 64,
    hidden_layers: int = 3,
    seed: int = 0,
    device: torch.device | str = "cpu",
    feature_mode: str = "raw",
    architecture: str = "film",
) -> ConditionalRatioMLP:
    """Construct a deterministically initialized population-ratio classifier."""

    torch.manual_seed(seed)
    return ConditionalRatioMLP(
        condition_dim=condition_dim,
        sample_dim=sample_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        feature_mode=feature_mode,
        architecture=architecture,
    ).to(device)


__all__: Sequence[str] = (
    "ConditionFiLMResidualBlock",
    "ConditionalRatioMLP",
    "SUPPORTED_RATIO_ARCHITECTURES",
    "OmniFoldIterationDiagnostics",
    "OmniFoldPassOneESSError",
    "OmniFoldResult",
    "RatioFitConfig",
    "RatioFitDiagnostics",
    "draw_independent_class_indices",
    "fit_density_ratio",
    "fit_step1",
    "fit_step2",
    "global_mean_one",
    "global_mean_one_from_log_weights",
    "make_ratio_mlp",
    "run_full_omnifold_identity",
    "score_cumulative_reward",
    "score_reward",
    "step1_pull_weights",
    "step2_push_weights",
)
