"""TARP coverage, decomposed into full, per-component, and copula arms.

The construction is the pooled-member randomisation used elsewhere in this
repository: truth plus K candidates form one exchangeability pool of M = K + 1
members, and the observed assignment (always truth) is compared against null
assignments drawn uniformly from the same pool, sharing references, strict
ranks, and jitter. Under a correctly calibrated model the observed coverage
curve is exchangeable with the nulls.

The decomposition is the point, and it separates cleanly. Measured on the ring
toy, restricted to the correlated annulus, N=3617, K=32:

    model                      full   marginal_y1  marginal_y2  rank_copula
    correct                    0.450     0.485        0.555        0.235
    correlation destroyed      0.005     0.485        0.615        0.040
    marginals scaled x0.85     0.005     0.005        0.005        0.005

Read the arms jointly, never one alone. Marginal arms passing while the copula
arm rejects isolates a dependence failure, which is row two. A copula rejection
on its own does not: the rank transform removes a monotone map applied to the
whole pool, but the candidates are transformed while truth is not, so a marginal
mismatch shifts truth's position in the pool and reaches the ranks. That is row
three, where the correlation is exact and every arm still rejects.

The single most important thing to know before using this: on a conditional
problem, run ``evaluate_tarp_binned``, not ``evaluate_tarp``. Pooling every
event into one test dilutes a conditional failure until it vanishes. Measured
on the ring toy at N=12000, K=32, with the entire conditional correlation
destroyed, the pooled copula arm returns p=0.79 while the same test inside the
annulus that carries the correlation returns p=0.005. Roughly 70% of events sit
in regions with no true correlation, and they contribute noise to the coverage
curve while carrying no signal. Reading the pooled p-value alone would certify
a model that learned nothing.

Two further limits travel with the numbers. TARP is necessary, not sufficient:
passing does not certify the conditional density. And p-values are comparable
across experiments only when N, K, the reference trials, and the null count are
all identical, so they are recorded in the result rather than left implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .core import (
    Metric,
    Prediction,
    assign_bins,
    bin_edges_from_quantiles,
    standardize,
    truth_standardizer,
)

GRID = np.linspace(0.0, 1.0, 201)


def pool_members(candidates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return ``(N, K+1, D)`` with truth fixed at member zero."""
    candidates = np.asarray(candidates, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if candidates.ndim != 3 or truth.ndim != 2:
        raise ValueError("candidates must be (N,K,D) and truth must be (N,D)")
    if candidates.shape[0] != truth.shape[0] or candidates.shape[2] != truth.shape[1]:
        raise ValueError("candidate and truth shapes disagree")
    return np.concatenate([truth[:, None, :], candidates], axis=1)


def rank_copula_members(members: np.ndarray) -> np.ndarray:
    """Replace each coordinate by its within-pool rank.

    This removes a monotone transform applied to the *whole* pool, but not one
    applied to the candidates alone. Truth is a member of the pool and is not
    transformed with them, so when the model's marginals disagree with truth the
    truth's position inside the pool shifts and the mismatch reaches the ranks.
    Measured: rescaling the candidates to 85% of the true scale, leaving the
    correlation exact, rejects the copula arm at p=0.005.

    So this arm is *not* marginal-free and must be read together with the
    marginal arms. The diagnostic combination is marginal arms passing while the
    copula arm rejects, which does isolate a dependence failure; the copula arm
    rejecting on its own does not.
    """
    members = np.asarray(members, dtype=np.float64)
    order = np.argsort(members, axis=1, kind="mergesort")
    ranks = np.empty_like(order)
    values = np.broadcast_to(np.arange(members.shape[1])[None, :, None], order.shape)
    np.put_along_axis(ranks, order, values, axis=1)
    return (ranks.astype(np.float64) + 0.5) / float(members.shape[1])


def component_members(members: np.ndarray, arm: str, names: Sequence[str]) -> np.ndarray:
    if arm == "full":
        return members
    if arm == "rank_copula":
        return rank_copula_members(members)
    if arm.startswith("marginal_"):
        name = arm[len("marginal_") :]
        if name not in names:
            raise ValueError(f"unknown component for arm {arm!r}")
        index = list(names).index(name)
        return members[..., index : index + 1]
    raise ValueError(f"unknown TARP arm: {arm!r}")


def default_arms(names: Sequence[str]) -> tuple[str, ...]:
    return ("full", "rank_copula", *(f"marginal_{name}" for name in names))


def draw_references(
    arm: str, n_events: int, dimension: int, rng: np.random.Generator
) -> np.ndarray:
    """Reference points matched to the support of the arm's member space.

    This is not cosmetic. TARP ranks members by distance to a reference, so the
    references must live where the members live. The copula transform maps
    members into the unit cube, and standard-normal references then sit almost
    entirely outside it, which collapses the distance ordering towards "distance
    from the corner nearest the origin" and throws away most of the dependence
    signal the arm exists to measure. Uniform references restore it.
    """
    if arm == "rank_copula":
        return rng.random((n_events, dimension))
    return rng.standard_normal((n_events, dimension))


def member_distances_to_references(
    members: np.ndarray, references: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(members - references[:, None, :], axis=-1)


def ranks_from_distances(distances: np.ndarray, member_index: np.ndarray) -> np.ndarray:
    """Strict integer rank of each selected member among the complete pool."""
    distances = np.asarray(distances, dtype=np.float64)
    member_index = np.asarray(member_index, dtype=np.int64)
    n_events = distances.shape[0]
    if member_index.ndim == 1:
        selected = distances[np.arange(n_events), member_index]
        return np.sum(distances < selected[:, None], axis=1)
    if member_index.ndim == 2:
        selected = distances[np.arange(n_events)[None, :], member_index]
        return np.sum(distances[None, :, :] < selected[:, :, None], axis=2)
    raise ValueError("member_index must have shape (N,) or (B,N)")


def randomized_u_from_ranks(
    ranks: np.ndarray, n_members: int, jitter: np.ndarray
) -> np.ndarray:
    """Finite-member map ``u = (rank + U) / M``."""
    values = (np.asarray(ranks, dtype=np.float64) + jitter) / float(n_members)
    if np.any((values < 0.0) | (values >= 1.0)):
        raise ValueError("randomized ranks fall outside [0,1)")
    return values


def build_assignments(
    n_events: int, n_members: int, n_null: int, rng: np.random.Generator
) -> np.ndarray:
    """Observed truth row (index 0) plus pooled-member null assignments."""
    assignments = np.empty((n_null + 1, n_events), dtype=np.int64)
    assignments[0] = 0
    assignments[1:] = rng.integers(0, n_members, size=(n_null, n_events), dtype=np.int64)
    return assignments


def coverage_curves(values: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    bins = np.searchsorted(grid, values, side="left")
    if np.any((bins < 0) | (bins >= len(grid))):
        raise ValueError("randomized ranks fall outside the coverage grid")
    curves = np.empty((values.shape[0], len(grid)), dtype=np.float64)
    for row in range(values.shape[0]):
        counts = np.bincount(bins[row], minlength=len(grid))
        curves[row] = np.cumsum(counts) / float(values.shape[1])
    return curves


def leave_one_out_curve_statistic(curves: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[0] < 2:
        raise ValueError("curves must be (assignments, grid) with at least two rows")
    total = np.sum(curves, axis=0)
    reference = (total[None, :] - curves) / float(curves.shape[0] - 1)
    return np.mean((curves - reference) ** 2, axis=1)


def monte_carlo_pvalue(statistics: np.ndarray) -> float:
    statistics = np.asarray(statistics, dtype=np.float64)
    return float(1 + np.sum(statistics[1:] >= statistics[0])) / float(len(statistics))


def holm_adjusted(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment.

    Gating on "every arm passes" without this inflates the false rejection rate
    with the number of arms, which is easy to do here because the decomposition
    deliberately produces several correlated tests.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for position, (name, value) in enumerate(ordered):
        running = max(running, (total - position) * value)
        adjusted[name] = float(min(1.0, running))
    return adjusted


@dataclass(frozen=True)
class TarpArm:
    name: str
    pvalue: float
    holm_pvalue: float
    observed_statistic: float
    null_statistic_mean: float
    max_abs_gap: float
    observed_curve: np.ndarray
    mean_null_curve: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "pvalue": float(self.pvalue),
            "holm_pvalue": float(self.holm_pvalue),
            "observed_statistic": float(self.observed_statistic),
            "null_statistic_mean": float(self.null_statistic_mean),
            "max_abs_obs_minus_null_gap": float(self.max_abs_gap),
            "observed_curve": self.observed_curve.tolist(),
            "mean_null_curve": self.mean_null_curve.tolist(),
        }


@dataclass(frozen=True)
class TarpResult:
    geometry: dict[str, Any]
    arms: dict[str, TarpArm]
    metrics: list[Metric] = field(default_factory=list)
    caveat: str = (
        "TARP is necessary, not sufficient: a passing p-value does not certify "
        "the conditional density. p-values compare across runs only at identical "
        "geometry."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry": self.geometry,
            "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
            "metrics": {metric.name: metric.to_dict() for metric in self.metrics},
            "caveat": self.caveat,
        }


def evaluate_tarp(
    prediction: Prediction,
    *,
    arms: Sequence[str] | None = None,
    n_reference_trials: int = 8,
    n_null_assignments: int = 199,
    seed: int = 0,
    alpha: float = 0.05,
) -> TarpResult:
    """Pooled-member TARP over the requested arms, with Holm-adjusted decisions."""
    if n_null_assignments < 19:
        raise ValueError("at least 19 null assignments are needed for alpha=0.05")
    names = prediction.component_names
    arms = tuple(arms) if arms is not None else default_arms(names)

    location, scale = truth_standardizer(prediction.truth)
    members_full = pool_members(
        standardize(prediction.candidates, location, scale),
        standardize(prediction.truth, location, scale),
    )
    n_events, n_members, _ = members_full.shape
    arm_members = {arm: component_members(members_full, arm, names) for arm in arms}

    assignments = build_assignments(
        n_events, n_members, n_null_assignments, np.random.default_rng(seed + 7_919)
    )
    sums = {
        arm: np.zeros((n_null_assignments + 1, len(GRID)), dtype=np.float64)
        for arm in arms
    }
    for trial in range(n_reference_trials):
        rng = np.random.default_rng(seed + 1_009 * trial)
        ranks: dict[str, np.ndarray] = {}
        for arm, block in arm_members.items():
            references = draw_references(arm, n_events, block.shape[-1], rng)
            distances = member_distances_to_references(block, references)
            if np.any(np.diff(np.sort(distances, axis=1), axis=1) == 0.0):
                raise ValueError(
                    f"exact distance tie in arm {arm!r}; the pool is not continuous"
                )
            ranks[arm] = ranks_from_distances(distances, assignments)
        jitter = rng.random(next(iter(ranks.values())).shape)
        for arm, arm_ranks in ranks.items():
            sums[arm] += coverage_curves(
                randomized_u_from_ranks(arm_ranks, n_members, jitter)
            )

    raw: dict[str, float] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        curves = sums[arm] / float(n_reference_trials)
        statistics = leave_one_out_curve_statistic(curves)
        observed = curves[0]
        mean_null = np.mean(curves[1:], axis=0)
        raw[arm] = monte_carlo_pvalue(statistics)
        summaries[arm] = {
            "observed_statistic": float(statistics[0]),
            "null_statistic_mean": float(np.mean(statistics[1:])),
            "max_abs_gap": float(np.max(np.abs(observed - mean_null))),
            "observed_curve": observed,
            "mean_null_curve": mean_null,
        }

    adjusted = holm_adjusted(raw)
    built = {
        arm: TarpArm(name=arm, pvalue=raw[arm], holm_pvalue=adjusted[arm], **summaries[arm])
        for arm in arms
    }

    metrics = [
        Metric(
            name=f"tarp_{arm}_pvalue",
            value=built[arm].pvalue,
            direction="higher_is_better",
            floor=None,
            null_reference=None,
            floor_source="exchangeable_null",
            threshold=alpha,
            reliability="supporting",
            note=(
                "dependence only, marginals ranked away: the arm that carries the "
                "signal when marginals are correct by construction. Only useful "
                "when the events share a conditioning region, so prefer "
                "evaluate_tarp_binned."
                if arm == "rank_copula"
                else ""
            ),
            extra={"holm_pvalue": built[arm].holm_pvalue},
        )
        for arm in arms
    ]
    metrics.append(
        Metric(
            name="tarp_min_holm_pvalue",
            value=float(min(adjusted.values())),
            direction="higher_is_better",
            floor=None,
            null_reference=None,
            floor_source="exchangeable_null",
            reliability="supporting",
            threshold=alpha,
            note=(
                "family-wise across arms; use this, not the raw minimum. Still "
                "pooled over all events, so on a conditional problem it can pass a "
                "model whose conditional structure is entirely absent."
            ),
        )
    )

    geometry = {
        "n_events": int(n_events),
        "k_candidates": int(prediction.n_candidates),
        "n_members": int(n_members),
        "n_reference_trials": int(n_reference_trials),
        "n_null_assignments": int(n_null_assignments),
        "grid_size": int(len(GRID)),
        "seed": int(seed),
        "arms": list(arms),
    }
    return TarpResult(geometry=geometry, arms=built, metrics=metrics)


@dataclass(frozen=True)
class BinnedTarpResult:
    """TARP run separately inside bins of a conditioning variable."""

    bins: dict[int, TarpResult]
    edges: np.ndarray
    counts: np.ndarray
    metrics: list[Metric] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": self.edges.tolist(),
            "counts": self.counts.tolist(),
            "bins": {str(index): result.to_dict() for index, result in self.bins.items()},
            "metrics": {metric.name: metric.to_dict() for metric in self.metrics},
        }


def evaluate_tarp_binned(
    prediction: Prediction,
    profile_values: np.ndarray,
    *,
    n_bins: int = 4,
    arms: Sequence[str] | None = None,
    n_reference_trials: int = 8,
    n_null_assignments: int = 199,
    min_events: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> BinnedTarpResult:
    """TARP inside bins of the conditioning variable. Prefer this on conditional problems.

    Pooling every event into one TARP dilutes a conditional failure into
    invisibility. Measured on the ring toy at N=12000, K=32, with the entire
    conditional correlation destroyed: pooled over all events the copula arm
    returns p=0.79 and sees nothing, while the same test restricted to the
    annulus that carries the correlation returns p=0.005, the smallest value
    199 null assignments can produce. The failure was never hidden from the
    statistic, only averaged away by the 70% of events that carry no signal.

    Bins multiply the size of the test family, so Holm is applied across every
    bin and arm jointly and ``tarp_binned_min_holm_pvalue`` is the only decision
    number in the result.
    """
    profile_values = np.asarray(profile_values, dtype=np.float64).reshape(-1)
    if len(profile_values) != prediction.n_events:
        raise ValueError("profile_values must supply one scalar per event")
    edges = bin_edges_from_quantiles(profile_values, n_bins)
    assigned = assign_bins(profile_values, edges)
    counts = np.array(
        [int((assigned == index).sum()) for index in range(len(edges) - 1)]
    )
    if not np.any(counts >= min_events):
        raise ValueError(
            f"no bin reached min_events={min_events}; TARP needs a substantial "
            "event count per bin to have any power"
        )

    results: dict[int, TarpResult] = {}
    family: dict[str, float] = {}
    for index, count in enumerate(counts):
        if count < min_events:
            continue
        block = prediction.subset(np.where(assigned == index)[0])
        result = evaluate_tarp(
            block,
            arms=arms,
            n_reference_trials=n_reference_trials,
            n_null_assignments=n_null_assignments,
            seed=seed + 101 * index,
            alpha=alpha,
        )
        results[index] = result
        for name, entry in result.arms.items():
            family[f"bin{index}:{name}"] = entry.pvalue

    adjusted = holm_adjusted(family)
    worst = min(adjusted, key=lambda key: adjusted[key])
    metrics = [
        Metric(
            name="tarp_binned_min_holm_pvalue",
            value=float(adjusted[worst]),
            direction="higher_is_better",
            floor=None,
            null_reference=None,
            floor_source="exchangeable_null",
            reliability="supporting",
            threshold=alpha,
            note=(
                "family-wise across every bin and arm. The only TARP number that "
                "should drive a decision on a conditional problem; the unconditional "
                "version is diluted and can pass a model with no conditional "
                "structure at all."
            ),
            extra={"worst": worst, "raw_pvalues": family},
        )
    ]
    return BinnedTarpResult(
        bins=results, edges=edges, counts=counts, metrics=metrics
    )


__all__ = [
    "GRID",
    "BinnedTarpResult",
    "TarpArm",
    "TarpResult",
    "build_assignments",
    "component_members",
    "coverage_curves",
    "default_arms",
    "draw_references",
    "evaluate_tarp",
    "evaluate_tarp_binned",
    "holm_adjusted",
    "leave_one_out_curve_statistic",
    "member_distances_to_references",
    "monte_carlo_pvalue",
    "pool_members",
    "randomized_u_from_ranks",
    "rank_copula_members",
    "ranks_from_distances",
]
