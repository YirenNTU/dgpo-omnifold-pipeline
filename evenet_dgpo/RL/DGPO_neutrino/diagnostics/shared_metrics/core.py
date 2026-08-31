"""Shared evaluation primitives, deliberately independent of any toy.

Everything in this package consumes one structure, ``Prediction``, built from
plain arrays:

    truth       (N, D)      one realisation per event, which is all Monte Carlo
                            gives in the real problem
    candidates  (N, K, D)   the model's conditional sample pool
    condition   (N, C)      optional, only needed by conditional metrics
    truth_pool  (N, K, D)   optional, extra truth draws per event

No metric may import a toy module, read a toy config, or assume a dimension.
That is what makes the same code usable on ring2d, on the four-mode mixture,
and on real TT2L predictions.

``truth_pool`` deserves emphasis: it is what makes a *floor* computable. A
metric value alone is not interpretable, because a finite candidate pool and a
finite event count are always slightly wrong even for a perfect model. Toys can
supply extra truth draws and therefore get exact floors; real data cannot, and
those metrics honestly report ``floor=None`` rather than a fabricated zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

Direction = Literal["lower_is_better", "higher_is_better"]
FloorSource = Literal[
    "truth_pool", "split_half", "exchangeable_null", "analytic", "unavailable"
]
Reliability = Literal["primary", "supporting", "diagnostic_only"]
"""How much weight a metric may carry in a conclusion.

Assigned from measured separation against controlled failures, not from
intuition, and fixed in the code so it cannot drift run to run. ``primary``
metrics showed tens of sigma against the failure they target and near zero
against the others, so they form a basis. ``diagnostic_only`` metrics were
measured to be too weak or too saturated to decide anything: they are worth
plotting and worth reading, but a conclusion may not rest on them.
"""


@dataclass(frozen=True)
class Metric:
    """One measurement together with what makes it interpretable.

    ``floor`` is what a perfect model scores under the identical estimator and
    sample budget; ``null_reference`` is what a model that learned nothing
    scores. Either may be ``None`` when the available data cannot produce it,
    in which case ``floor_source`` records why.
    """

    name: str
    value: float
    direction: Direction = "lower_is_better"
    floor: float | None = None
    null_reference: float | None = None
    floor_source: FloorSource = "unavailable"
    reliability: Reliability = "supporting"
    threshold: float | None = None
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def headroom(self) -> float | None:
        """Distance from the floor in units of the floor-to-null range.

        0.0 means the model is at the achievable floor, 1.0 means it is no
        better than having learned nothing. Undefined without both references,
        and undefined when the two coincide, which itself means the metric has
        no resolving power on this problem.
        """
        if self.floor is None or self.null_reference is None:
            return None
        span = self.null_reference - self.floor
        if abs(span) < 1e-12:
            return None
        return float((self.value - self.floor) / span)

    @property
    def resolving_power(self) -> float | None:
        """Fraction of the null-to-floor gap that estimator noise leaves usable.

        Near 1 the metric can separate a good model from a useless one; near 0
        the floor has risen to meet the null and the metric cannot distinguish
        anything at this sample size, whatever value it reports. Checking this
        is what catches an underpowered evaluation before it is read as a result.
        """
        if self.floor is None or self.null_reference is None:
            return None
        scale = max(abs(self.null_reference), abs(self.floor))
        if scale < 1e-12:
            return None
        gap = (
            self.null_reference - self.floor
            if self.direction == "lower_is_better"
            else self.floor - self.null_reference
        )
        return float(gap / scale)

    @property
    def passed(self) -> bool | None:
        if self.threshold is None:
            return None
        if self.direction == "lower_is_better":
            return bool(self.value <= self.threshold)
        return bool(self.value >= self.threshold)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "value": float(self.value),
            "direction": self.direction,
            "floor": None if self.floor is None else float(self.floor),
            "null_reference": (
                None if self.null_reference is None else float(self.null_reference)
            ),
            "floor_source": self.floor_source,
            "reliability": self.reliability,
            "threshold": None if self.threshold is None else float(self.threshold),
            "headroom": self.headroom,
            "resolving_power": self.resolving_power,
            "pass": self.passed,
        }
        if self.note:
            payload["note"] = self.note
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass(frozen=True)
class Prediction:
    """Model output for one evaluation set, in a toy-agnostic form."""

    truth: np.ndarray
    candidates: np.ndarray
    condition: np.ndarray | None = None
    weights: np.ndarray | None = None
    truth_pool: np.ndarray | None = None
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        truth = np.asarray(self.truth, dtype=np.float64)
        candidates = np.asarray(self.candidates, dtype=np.float64)
        if truth.ndim != 2:
            raise ValueError("truth must have shape (n_events, n_components)")
        if candidates.ndim != 3:
            raise ValueError(
                "candidates must have shape (n_events, n_candidates, n_components)"
            )
        if candidates.shape[0] != truth.shape[0]:
            raise ValueError("truth and candidates disagree on the event count")
        if candidates.shape[2] != truth.shape[1]:
            raise ValueError("truth and candidates disagree on the component count")
        object.__setattr__(self, "truth", truth)
        object.__setattr__(self, "candidates", candidates)

        if self.condition is not None:
            condition = np.asarray(self.condition, dtype=np.float64)
            if condition.ndim == 1:
                condition = condition[:, None]
            if condition.ndim != 2 or condition.shape[0] != truth.shape[0]:
                raise ValueError("condition must have shape (n_events, n_condition)")
            object.__setattr__(self, "condition", condition)

        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape != candidates.shape[:2]:
                raise ValueError("weights must have shape (n_events, n_candidates)")
            if np.any(weights < 0.0):
                raise ValueError("weights must be nonnegative")
            totals = weights.sum(axis=1)
            if np.any(totals <= 0.0):
                raise ValueError("every event needs positive total weight")
            object.__setattr__(self, "weights", weights / totals[:, None])

        if self.truth_pool is not None:
            truth_pool = np.asarray(self.truth_pool, dtype=np.float64)
            if truth_pool.shape != candidates.shape:
                raise ValueError(
                    "truth_pool must match candidates in shape so that floors are "
                    "computed under the identical sample budget"
                )
            object.__setattr__(self, "truth_pool", truth_pool)

        if self.component_names:
            if len(self.component_names) != truth.shape[1]:
                raise ValueError("component_names must cover every component")
        else:
            object.__setattr__(
                self,
                "component_names",
                tuple(f"y{index}" for index in range(truth.shape[1])),
            )

    @property
    def n_events(self) -> int:
        return int(self.truth.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def n_components(self) -> int:
        return int(self.truth.shape[1])

    @property
    def has_floor(self) -> bool:
        return self.truth_pool is not None

    def normalized_weights(self) -> np.ndarray:
        if self.weights is not None:
            return self.weights
        return np.full(
            (self.n_events, self.n_candidates), 1.0 / self.n_candidates, dtype=np.float64
        )

    def oracle(self) -> "Prediction":
        """The same evaluation set with the truth pool substituted in.

        Running any metric on this returns that metric's floor: it is a model
        that is exactly right by construction, measured with the identical
        estimator, event count, and candidate budget.
        """
        if self.truth_pool is None:
            raise ValueError("no truth_pool available to build the oracle arm")
        return Prediction(
            truth=self.truth,
            candidates=self.truth_pool,
            condition=self.condition,
            truth_pool=None,
            component_names=self.component_names,
        )

    def subset(self, index: np.ndarray) -> "Prediction":
        index = np.asarray(index)
        return Prediction(
            truth=self.truth[index],
            candidates=self.candidates[index],
            condition=None if self.condition is None else self.condition[index],
            weights=None if self.weights is None else self.weights[index],
            truth_pool=None if self.truth_pool is None else self.truth_pool[index],
            component_names=self.component_names,
        )


def standardize(
    values: np.ndarray, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - location) / scale


def truth_standardizer(truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-component location and scale fitted on truth only.

    Fitting on the model's own pool would absorb part of the very discrepancy
    the metrics are meant to detect, and would make two arms incomparable. Truth
    is the one reference every arm shares.
    """
    truth = np.asarray(truth, dtype=np.float64)
    scale = truth.std(axis=0)
    if np.any(scale <= 0.0):
        raise ValueError("truth has a degenerate component; cannot standardize")
    return truth.mean(axis=0), scale


def weighted_pearson(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted Pearson correlation of two flat arrays.

    Note for callers: this is invariant to rescaling either input, so it cannot
    see variance collapse. Never report a correlation metric without a scale
    metric alongside it.
    """
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise ValueError("inputs must have the same length")
    if len(first) < 2:
        return float("nan")
    if weights is None:
        weights = np.ones_like(first)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = weights.sum()
    if total <= 0.0:
        return float("nan")
    mean_first = np.sum(weights * first) / total
    mean_second = np.sum(weights * second) / total
    delta_first = first - mean_first
    delta_second = second - mean_second
    covariance = np.sum(weights * delta_first * delta_second) / total
    variance_first = np.sum(weights * delta_first**2) / total
    variance_second = np.sum(weights * delta_second**2) / total
    denominator = np.sqrt(variance_first * variance_second)
    if denominator <= 0.0:
        return float("nan")
    return float(covariance / denominator)


def bin_edges_from_quantiles(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-occupancy edges, so no bin is dominated by estimator noise."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    unique = np.unique(edges)
    if len(unique) < 2:
        raise ValueError("cannot bin a constant profile variable")
    return unique


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def metrics_to_dict(metrics: Sequence[Metric]) -> dict[str, Any]:
    return {metric.name: metric.to_dict() for metric in metrics}


__all__ = [
    "Direction",
    "FloorSource",
    "Metric",
    "Prediction",
    "assign_bins",
    "bin_edges_from_quantiles",
    "metrics_to_dict",
    "standardize",
    "truth_standardizer",
    "weighted_pearson",
]
