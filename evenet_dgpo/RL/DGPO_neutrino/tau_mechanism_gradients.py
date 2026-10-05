"""Read-only decomposition of the exact native conditional-tau DGPO graph.

Cells retain the full-call normalization and the original accumulation weights.
Their sum must reconstruct the unclipped native gradient before interpretation.
The caller supplies a gate from the *full* native loss call; no loss/gate is
recomputed within a condition or noise band. Labels describe observed inputs,
never truth or event identity. DDP ``no_sync`` is the caller's responsibility.

Norms and cosines describe sampled surrogate gradients in parameter space, not
the population reward gradient or the effect of an AdamW update on physics.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor


def _bin_observed(values: Tensor, edges: Sequence[float], *, name: str) -> Tensor:
    if values.ndim != 1 or values.numel() < 1 or not values.is_floating_point():
        raise ValueError(f"{name} must be a nonempty floating (B,) vector")
    edges = tuple(float(edge) for edge in edges)
    if len(edges) < 2 or any(not math.isfinite(edge) for edge in edges) or any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError(f"{name} bin edges must be finite and strictly increasing")
    if not bool(torch.isfinite(values).all()) or bool(((values < edges[0]) | (values > edges[-1])).any()):
        raise ValueError(f"{name} values lie outside the declared diagnostic range")
    # Interior edges belong to the upper bin; both exterior endpoints are kept.
    return torch.bucketize(values.detach().contiguous(), torch.tensor(edges[1:-1], device=values.device, dtype=values.dtype), right=True)


def observed_visible_opening_labels(
    batch: Mapping[str, Tensor], *, edges: Sequence[float] = (-1.0, -0.5, 0.5, 1.0),
) -> Tensor:
    """Label observed visible-pair lab opening cosine, without truth access.

This is a conditioning stratum, not a tau-rest-frame spin observable. Physical
visible Cartesian components are used directly; event IDs are never inspected.
"""
    momenta = []
    for leg in ("a", "b"):
        keys = [f"lead_{leg}_visible_{axis}" for axis in ("px", "py", "pz")]
        if any(key not in batch for key in keys):
            raise ValueError("observed strata require physical visible Cartesian momenta")
        components = [batch[key] for key in keys]
        if any(not isinstance(value, Tensor) or value.ndim != 1 or not value.is_floating_point() for value in components):
            raise ValueError("visible momentum components must be floating (B,) tensors")
        if len({tuple(value.shape) for value in components}) != 1:
            raise ValueError("visible momentum components have different event shapes")
        momentum = torch.stack(components, dim=-1).detach().double()
        if not bool(torch.isfinite(momentum).all()) or bool((momentum.norm(dim=-1) <= 0).any()):
            raise ValueError("visible momentum must be finite and nonzero")
        momenta.append(momentum / momentum.norm(dim=-1, keepdim=True))
    if momenta[0].shape != momenta[1].shape:
        raise ValueError("visible legs have different event shapes")
    opening_cos = (momenta[0] * momenta[1]).sum(-1).clamp(-1.0, 1.0)
    return _bin_observed(opening_cos, edges, name="observed visible opening cosine")


def normalized_time_labels(
    event_times: Tensor, *, edges: Sequence[float] = (0.0, 0.7 / 3.0, 2.0 * 0.7 / 3.0, 0.7),
) -> Tensor:
    """Bin native event times in the predeclared normalized diffusion range."""
    return _bin_observed(event_times, edges, name="normalized diffusion time")


def native_reward_event_terms(
    L_cur_2d: Tensor, advantages: Tensor, detached_gate: Tensor,
) -> Tensor:
    """Return B terms whose sum equals the native full-call DGPO main loss."""
    if L_cur_2d.ndim != 2 or advantages.shape != L_cur_2d.shape:
        raise ValueError("L_cur and advantages must have matching (K, B) shapes")
    if min(L_cur_2d.shape) < 1:
        raise ValueError("native reward loss requires nonempty K and B")
    if detached_gate.shape != L_cur_2d.shape[1:]:
        raise ValueError("the full native detached gate must have shape (B,)")
    if detached_gate.requires_grad or advantages.requires_grad:
        raise ValueError("native gate and advantages must be detached")
    if not bool(torch.isfinite(detached_gate).all()) or bool(
        ((detached_gate < 0) | (detached_gate > 1)).any()
    ):
        raise ValueError("native gate must contain finite values in [0, 1]")
    return (L_cur_2d * advantages * detached_gate.unsqueeze(0)).mean(0) / L_cur_2d.shape[1]


def velocity_reference_event_terms(
    model_v: Tensor, ref_v: Tensor, noise_mask: Tensor, *, K: int, B: int,
    weight_correction: float | Tensor = 1.0,
) -> Tensor:
    """Split the unweighted native 0.5 velocity-MSE trust loss by event.

The denominator is the full-call valid-element mass. ``weight_correction``
is the trainer's existing correction for event microbatching. The trust
coefficient is intentionally absent and is applied only on reconstruction.
Rows must be candidate-major, matching native ``(K, B, ...)`` expansion.
"""
    if type(K) is not int or type(B) is not int or min(K, B) < 1:
        raise ValueError("K and B must be positive integers")
    if model_v.shape != ref_v.shape or model_v.ndim < 2 or model_v.shape[0] != K * B:
        raise ValueError("velocity rows must have matching candidate-major K*B shapes")
    try:
        mask = noise_mask.expand_as(model_v).to(model_v)
    except RuntimeError as exc:
        raise ValueError("noise mask cannot expand to native velocity shape") from exc
    if mask.requires_grad or not bool(torch.isfinite(mask).all()) or bool((mask < 0).any()):
        raise ValueError("noise mask must be detached, finite and nonnegative")
    correction = torch.as_tensor(weight_correction, device=model_v.device, dtype=model_v.dtype)
    if correction.numel() != 1 or correction.requires_grad or not bool(torch.isfinite(correction).all()) or bool((correction < 0).any()):
        raise ValueError("weight correction must be a detached finite nonnegative scalar")
    numerator = ((model_v - ref_v.detach()).square() * mask).reshape(K, B, -1).sum((0, 2))
    return 0.5 * numerator / mask.sum().clamp_min(1e-8) * correction.reshape(())


def _dot(left: Tensor, right: Tensor) -> float:
    # Bounded float64 working set for large full-parameter vectors.
    result = torch.zeros((), dtype=torch.float64, device=left.device)
    for start in range(0, left.numel(), 65536):
        result.add_(torch.dot(left[start:start + 65536].double(), right[start:start + 65536].double()))
    return float(result.cpu())


def _norm(vector: Tensor) -> float:
    return math.sqrt(max(0.0, _dot(vector, vector)))


def _pair(left: Tensor, right: Tensor, floor: float) -> dict[str, float | None]:
    cross = _dot(left, right)
    denominator = _norm(left) * _norm(right)
    return {"cross_dot": cross, "cosine": max(-1.0, min(1.0, cross / denominator)) if denominator > floor ** 2 else None}


def _family(vectors: Mapping[str, Tensor], floor: float) -> dict:
    names = list(vectors)
    total = torch.zeros_like(next(iter(vectors.values())))
    norms = {name: _norm(vector) for name, vector in vectors.items()}
    for vector in vectors.values():
        total.add_(vector)
    denominator = sum(norms.values())
    pairs = {f"{left}__{right}": _pair(vectors[left], vectors[right], floor) for left, right in combinations(names, 2)}
    defined = [row["cosine"] for row in pairs.values() if row["cosine"] is not None]
    return {
        "norms": norms,
        "sum_norm": _norm(total),
        "coherence": min(1.0, _norm(total) / denominator) if denominator > floor else None,
        "pairs": pairs,
        "negative_cosine_fraction": sum(value < 0 for value in defined) / len(defined) if defined else None,
    }


class NativeGradientAccumulator:
    """Accumulate already weighted reward/reference gradients for fixed cells.

``capture_call`` uses autograd.grad and never reads or writes parameter .grad.
The native backward runs afterward on the same graph. Supply a common label
vocabulary on every rank so detached collectives have identical order, even
when a rank has no examples in a cell. CPU storage limits GPU memory overhead.
"""

    def __init__(
        self, *, condition_names: Sequence[str], noise_band_names: Sequence[str],
        parameters: Sequence[Tensor] = (), vector_size: int | None = None,
        storage_device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
    ):
        self.parameters = tuple(parameters)
        if any(not parameter.requires_grad for parameter in self.parameters):
            raise ValueError("capture parameters must all require gradients")
        self.condition_names = tuple(condition_names)
        self.noise_band_names = tuple(noise_band_names)
        for names in (self.condition_names, self.noise_band_names):
            if not names or any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
                raise ValueError("cell names must be nonempty unique strings")
            if any("__" in name or "/" in name for name in names):
                raise ValueError("cell names cannot contain '/' or '__'")
        parameter_size = sum(parameter.numel() for parameter in self.parameters)
        self.vector_size = parameter_size if vector_size is None else vector_size
        if type(self.vector_size) is not int or self.vector_size < 1:
            raise ValueError("a positive vector_size or trainable parameters are required")
        if self.parameters and self.vector_size != parameter_size:
            raise ValueError("vector_size differs from capture parameters")
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("gradient storage dtype must be float32 or float64")
        self.device, self.dtype = torch.device(storage_device), dtype
        self.cells = {
            (condition, noise): {
                "reward": torch.zeros(self.vector_size, device=self.device, dtype=dtype),
                "reference": torch.zeros(self.vector_size, device=self.device, dtype=dtype),
            }
            for condition in self.condition_names for noise in self.noise_band_names
        }
        self.event_draw_counts = {key: 0.0 for key in self.cells}
        self.weighted_event_draw_mass = {key: 0.0 for key in self.cells}
        self.capture_calls = 0
        self.reduced = False
        self.world_size = 1

    def _check_vector(self, vector: Tensor) -> Tensor:
        if vector.ndim != 1 or vector.numel() != self.vector_size:
            raise ValueError("contribution must be a flat vector of the configured size")
        if not bool(torch.isfinite(vector).all()):
            raise FloatingPointError("gradient contribution contains non-finite values")
        return vector.detach().to(device=self.device, dtype=self.dtype)

    def add_flat(
        self, condition: str, noise_band: str, *, reward: Tensor, reference: Tensor,
        weight: float = 1.0, event_draw_count: int = 0,
    ) -> None:
        """Add a precomputed cell contribution without renormalizing its mass."""
        if self.reduced:
            raise RuntimeError("cannot add contributions after distributed reduction")
        key = (condition, noise_band)
        if key not in self.cells:
            raise ValueError("unknown condition/noise cell")
        if not math.isfinite(weight) or weight < 0 or type(event_draw_count) is not int or event_draw_count < 0:
            raise ValueError("weight and event count must be finite and nonnegative")
        reward_flat, reference_flat = self._check_vector(reward), self._check_vector(reference)
        with torch.no_grad():
            self.cells[key]["reward"].add_(reward_flat, alpha=float(weight))
            self.cells[key]["reference"].add_(reference_flat, alpha=float(weight))
        self.event_draw_counts[key] += event_draw_count
        self.weighted_event_draw_mass[key] += event_draw_count * float(weight)

    def _gradient(self, scalar: Tensor) -> Tensor:
        if not scalar.requires_grad:
            return torch.zeros(self.vector_size, device=self.device, dtype=self.dtype)
        gradients = torch.autograd.grad(scalar, self.parameters, retain_graph=True, allow_unused=True)
        return torch.cat([
            (torch.zeros_like(parameter) if gradient is None else gradient.detach()).reshape(-1).to(device=self.device, dtype=self.dtype)
            for parameter, gradient in zip(self.parameters, gradients, strict=True)
        ])

    def capture_call(
        self, *, reward_terms_b: Tensor, reference_terms_b: Tensor,
        condition_labels: Tensor, noise_labels: Tensor, weight: float = 1.0,
    ) -> None:
        """Measure cells of one native full-call graph before native backward.

Each input terms vector sums to the corresponding native scalar loss. Label
integers index the declared vocabularies. Missing cells contribute zero, never
a new per-cell mean. ``weight`` is event_weight / gradient_accumulation_steps.
"""
        if not self.parameters:
            raise RuntimeError("capture_call requires trainable parameters")
        if self.reduced:
            raise RuntimeError("cannot capture after distributed reduction")
        if reward_terms_b.ndim != 1 or reward_terms_b.numel() < 1 or reference_terms_b.shape != reward_terms_b.shape:
            raise ValueError("native event terms must have matching nonempty (B,) shapes")
        if not bool(torch.isfinite(reward_terms_b).all()) or not bool(torch.isfinite(reference_terms_b).all()):
            raise FloatingPointError("native event terms contain non-finite values")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("accumulation weight must be finite and nonnegative")
        labels = []
        for values, names in ((condition_labels, self.condition_names), (noise_labels, self.noise_band_names)):
            if values.shape != reward_terms_b.shape or values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
                raise ValueError("observed labels must be integer (B,) tensors")
            if bool(((values < 0) | (values >= len(names))).any()):
                raise ValueError("observed label lies outside the declared vocabulary")
            labels.append(values.to(device=reward_terms_b.device))
        for condition_index, condition in enumerate(self.condition_names):
            for noise_index, noise in enumerate(self.noise_band_names):
                mask = (labels[0] == condition_index) & (labels[1] == noise_index)
                count = int(mask.sum())
                if count:
                    reward = self._gradient(reward_terms_b[mask].sum())
                    reference = self._gradient(reference_terms_b[mask].sum())
                    self.add_flat(condition, noise, reward=reward, reference=reference, weight=weight, event_draw_count=count)
        self.capture_calls += 1

    def reduce_(
        self, *, world_size: int, collective_device: str | torch.device | None = None,
        reducer: Callable[[Tensor], Tensor | None] | None = None,
    ) -> None:
        """Stream detached SUM reductions then divide gradients by world size.

For NCCL with CPU storage, pass the native CUDA device. An injectable SUM
reducer supports tests and other distributed runners; a returned vector is
accepted, or the reducer may modify its detached argument in place. Counts are
global sums. Every rank must invoke this method once after all native calls.
"""
        if self.reduced:
            raise RuntimeError("cell gradients have already been reduced")
        if type(world_size) is not int or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if world_size > 1 and reducer is None:
            if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() != world_size:
                raise RuntimeError("distributed diagnostics require the matching initialized process group")
            reducer = lambda value: dist.all_reduce(value, op=dist.ReduceOp.SUM)
        collective_device = self.device if collective_device is None else torch.device(collective_device)
        if world_size > 1:
            assert reducer is not None
            with torch.no_grad():
                for components in self.cells.values():
                    for vector in components.values():
                        detached = vector.detach().to(collective_device).clone()
                        result = reducer(detached)
                        if result is not None:
                            detached = result.detach()
                        vector.copy_(self._check_vector(detached) / float(world_size))
                keys = list(self.cells)
                metadata = torch.tensor(
                    [self.capture_calls, *[self.event_draw_counts[key] for key in keys], *[self.weighted_event_draw_mass[key] for key in keys]],
                    device=collective_device, dtype=torch.float64,
                )
                result = reducer(metadata)
                metadata = metadata if result is None else result.detach()
                if metadata.shape != (1 + 2 * len(keys),) or not bool(torch.isfinite(metadata).all()):
                    raise ValueError("reduced diagnostic metadata is invalid")
                values = metadata.cpu().tolist()
                self.capture_calls = int(values[0])
                for index, key in enumerate(keys):
                    self.event_draw_counts[key] = values[1 + index]
                    self.weighted_event_draw_mass[key] = values[1 + len(keys) + index]
        self.world_size, self.reduced = world_size, True

    def total(self, component: str) -> Tensor:
        if component not in {"reward", "reference"}:
            raise ValueError("component must be reward or unweighted reference")
        result = torch.zeros(self.vector_size, device=self.device, dtype=self.dtype)
        for vectors in self.cells.values():
            result.add_(vectors[component])
        return result

    def summarize(
        self, *, actual_total_gradient: Tensor, trust_coefficient: float,
        parameter_blocks: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        reconstruction_tolerance: float = 5e-5, norm_floor: float = 1e-12,
        adamw_descent: Tensor | None = None,
    ) -> dict:
        """Gate interpretation on exact reconstruction of the native gradient."""
        if not math.isfinite(trust_coefficient) or trust_coefficient < 0:
            raise ValueError("trust coefficient must be finite and nonnegative")
        if not math.isfinite(reconstruction_tolerance) or reconstruction_tolerance <= 0 or not math.isfinite(norm_floor) or norm_floor <= 0:
            raise ValueError("reconstruction tolerance and norm floor must be positive")
        actual = self._check_vector(actual_total_gradient)
        reward, reference = self.total("reward"), self.total("reference")
        reconstructed = reward + float(trust_coefficient) * reference
        error = _norm(reconstructed - actual) / max(_norm(actual), norm_floor)
        reward_sq = _dot(reward, reward)
        report = {
            "reconstruction": {"relative_error": error, "tolerance": reconstruction_tolerance, "passed": error <= reconstruction_tolerance,
                               **_pair(reconstructed, actual, norm_floor)},
            "world_size": self.world_size, "distributed_reduced": self.reduced,
            "capture_calls": self.capture_calls,
            "trainable_parameters": self.vector_size,
            "trust_coefficient": float(trust_coefficient),
            "reward_reference": {**_pair(reward, reference, norm_floor),
                                 "reward_norm": _norm(reward), "reference_unweighted_norm": _norm(reference),
                                 "remaining_reward_projection": _dot(reward, reconstructed) / reward_sq if reward_sq > norm_floor ** 2 else None},
            "interpretation": "Sampled native surrogate contributions with original weights; parameter-space geometry does not establish reward or physics improvement.",
        }
        if error > reconstruction_tolerance:
            # A failed reconstruction is useful evidence about instrumentation,
            # but cancellation summaries would not identify the native update.
            return report
        components = {}
        for component in ("reward", "reference"):
            by_cell = {f"{condition}__{noise}": vectors[component] for (condition, noise), vectors in self.cells.items()}
            by_condition = {condition: sum((self.cells[condition, noise][component] for noise in self.noise_band_names), torch.zeros_like(reward)) for condition in self.condition_names}
            by_noise = {noise: sum((self.cells[condition, noise][component] for condition in self.condition_names), torch.zeros_like(reward)) for noise in self.noise_band_names}
            components[component] = {
                "cells": _family(by_cell, norm_floor),
                "conditions": _family(by_condition, norm_floor),
                "noise_bands": _family(by_noise, norm_floor),
                "noise_within_condition": {condition: _family({noise: self.cells[condition, noise][component] for noise in self.noise_band_names}, norm_floor) for condition in self.condition_names},
                "conditions_within_noise": {noise: _family({condition: self.cells[condition, noise][component] for condition in self.condition_names}, norm_floor) for noise in self.noise_band_names},
            }
            blocks = {}
            for name, ranges in (parameter_blocks or {}).items():
                ranges = list(ranges)
                if not ranges or any(type(start) is not int or type(stop) is not int or not 0 <= start < stop <= self.vector_size for start, stop in ranges):
                    raise ValueError("parameter block ranges must lie within the flat vector")
                ordered = sorted(ranges)
                if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
                    raise ValueError("parameter ranges within a block cannot overlap")
                sliced = lambda vectors: {key: torch.cat([value[start:stop] for start, stop in ranges]) for key, value in vectors.items()}
                blocks[name] = {"cells": _family(sliced(by_cell), norm_floor), "conditions": _family(sliced(by_condition), norm_floor), "noise_bands": _family(sliced(by_noise), norm_floor)}
            components[component]["parameter_blocks"] = blocks
        report["components"] = components
        report["event_draw_counts"] = {f"{condition}__{noise}": count for (condition, noise), count in self.event_draw_counts.items()}
        report["weighted_event_draw_mass"] = {f"{condition}__{noise}": mass for (condition, noise), mass in self.weighted_event_draw_mass.items()}
        if adamw_descent is not None:
            adam = self._check_vector(adamw_descent)
            report["adamw_descent"] = {"norm": _norm(adam), "reward": _pair(adam, reward, norm_floor), "total": _pair(adam, reconstructed, norm_floor)}
        return report


def summarize_replicas(
    replicas: Sequence[NativeGradientAccumulator], *, norm_floor: float = 1e-12,
) -> dict:
    """Report point stability across independent (t, eps) sampling replicas.

The off-diagonal cross-dot omits the squared norm's same-replica noise term.
It is an estimator conditional on the supplied fixed candidates/classifier,
not a population-gradient norm, confidence interval, or causal finding.
"""
    if len(replicas) < 2 or not math.isfinite(norm_floor) or norm_floor <= 0:
        raise ValueError("replica stability requires at least two replicas and positive norm_floor")
    first = replicas[0]
    if any((replica.vector_size, replica.condition_names, replica.noise_band_names) != (first.vector_size, first.condition_names, first.noise_band_names) for replica in replicas):
        raise ValueError("replicas must share parameter layout and declared cells")
    def stability(vectors):
        family = _family({str(index): vector.to(first.device) for index, vector in enumerate(vectors)}, norm_floor)
        pairs = list(family["pairs"].values())
        defined = [row["cosine"] for row in pairs if row["cosine"] is not None]
        half = len(vectors) // 2
        left = sum((value.to(first.device) for value in vectors[:half]), torch.zeros_like(vectors[0], device=first.device)) / half
        right = sum((value.to(first.device) for value in vectors[half:]), torch.zeros_like(vectors[0], device=first.device)) / (len(vectors) - half)
        return {"replica_norms": family["norms"], "pairs": family["pairs"],
                "mean_pairwise_cosine": sum(defined) / len(defined) if defined else None,
                "min_pairwise_cosine": min(defined) if defined else None,
                "cross_replica_self_dot": sum(row["cross_dot"] for row in pairs) / len(pairs),
                "half_mean_cosine": _pair(left, right, norm_floor)["cosine"]}
    return {
        "replicas": len(replicas),
        "interpretation": "Sampling stability conditional on fixed candidates and classifiers. No population-gradient or update-effect claim; no confidence interval.",
        "components": {
            component: {"total": stability([replica.total(component) for replica in replicas]),
                        "cells": {f"{condition}__{noise}": stability([replica.cells[condition, noise][component] for replica in replicas]) for condition, noise in first.cells}}
            for component in ("reward", "reference")
        },
    }
