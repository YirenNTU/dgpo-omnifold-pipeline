"""Reward-interface diagnostics for fixed-policy DGPO experiments.

The functions in this module are deliberately independent of classifier and
policy architecture.  They turn one fixed ``(K, B)`` logit panel into matched
DGPO advantages and summarize whether classifier members agree on candidate
ordering.  No function mutates a reward model or a diffusion policy.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor

from .dgpo_utils import compute_per_event_advantage


REWARD_ARMS = ("raw_loo", "calibrated_loo", "event_zscore", "centered_rank")


def balanced_logistic_loss(
    truth_logits: Tensor, generated_logits: Tensor, *, temperature: float = 1.0
) -> float:
    """Balanced BCE for truth-positive and generated-negative logits."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    truth = truth_logits.detach().double().reshape(-1)
    generated = generated_logits.detach().double().reshape(-1)
    if min(truth.numel(), generated.numel()) < 1:
        raise ValueError("temperature calibration populations must be non-empty")
    if not bool(torch.isfinite(truth).all() and torch.isfinite(generated).all()):
        raise ValueError("temperature calibration logits must be finite")
    scale = 1.0 / float(temperature)
    loss = 0.5 * (
        torch.nn.functional.softplus(-truth * scale).mean()
        + torch.nn.functional.softplus(generated * scale).mean()
    )
    return float(loss)


def fit_scalar_temperature(
    truth_logits: Tensor,
    generated_logits: Tensor,
    *,
    minimum: float = 0.05,
    maximum: float = 20.0,
    iterations: int = 96,
) -> dict[str, float]:
    """Fit one positive temperature on a held-out calibration population.

    Optimization is a deterministic golden-section search in log-temperature.
    A bounded one-dimensional search is easier to audit than an optimizer whose
    stopping state would become another source of experimental variation.
    """

    if not (0.0 < float(minimum) < float(maximum)):
        raise ValueError("temperature bounds must satisfy 0 < minimum < maximum")
    if type(iterations) is not int or iterations < 16:
        raise ValueError("temperature search needs at least 16 iterations")
    # Validate once before entering the search.
    uncalibrated = balanced_logistic_loss(truth_logits, generated_logits)
    lo, hi = math.log(float(minimum)), math.log(float(maximum))
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = hi - ratio * (hi - lo)
    right = lo + ratio * (hi - lo)

    def objective(log_temperature: float) -> float:
        return balanced_logistic_loss(
            truth_logits, generated_logits, temperature=math.exp(log_temperature)
        )

    f_left, f_right = objective(left), objective(right)
    for _ in range(iterations):
        if f_left <= f_right:
            hi, right, f_right = right, left, f_left
            left = hi - ratio * (hi - lo)
            f_left = objective(left)
        else:
            lo, left, f_left = left, right, f_right
            right = lo + ratio * (hi - lo)
            f_right = objective(right)
    candidates = [
        (float(minimum), objective(math.log(float(minimum)))),
        (math.exp(0.5 * (lo + hi)), objective(0.5 * (lo + hi))),
        (float(maximum), objective(math.log(float(maximum)))),
    ]
    temperature, calibrated = min(candidates, key=lambda item: item[1])
    return {
        "temperature": float(temperature),
        "bce_before": float(uncalibrated),
        "bce_after": float(calibrated),
        "bce_improvement": float(uncalibrated - calibrated),
        "hit_lower_bound": float(math.isclose(temperature, minimum)),
        "hit_upper_bound": float(math.isclose(temperature, maximum)),
    }


def centered_candidate_rank(values: Tensor) -> Tensor:
    """Tie-safe candidate ranks in [-1, 1], centered within each event."""

    if values.ndim != 2 or int(values.shape[0]) < 2:
        raise ValueError("candidate ranks require a (K,B) tensor with K >= 2")
    # Pairwise wins minus losses gives average ranks without an arbitrary tie
    # order.  The candidate axes are the first and second dimensions below.
    left = values.unsqueeze(1)
    right = values.unsqueeze(0)
    signed = (left > right).to(values.dtype) - (left < right).to(values.dtype)
    return signed.sum(dim=1) / float(values.shape[0] - 1)


def reward_advantage_arms(
    logits: Tensor,
    *,
    temperature: float,
    raw_tempering: float = 0.75,
    zscore_epsilon: float = 1.0e-8,
) -> dict[str, Tensor]:
    """Return four matched advantage tensors from the same fixed logits."""

    if logits.ndim != 2 or int(logits.shape[0]) < 2:
        raise ValueError("reward logits must have shape (K,B) with K >= 2")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("reward logits contain NaN/Inf")
    if not math.isfinite(float(raw_tempering)) or float(raw_tempering) <= 0.0:
        raise ValueError("raw_tempering must be finite and positive")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(float(zscore_epsilon)) or float(zscore_epsilon) <= 0.0:
        raise ValueError("zscore_epsilon must be finite and positive")
    raw = logits * float(raw_tempering)
    calibrated = logits / float(temperature)
    return {
        "raw_loo": compute_per_event_advantage(
            raw, estimator="leave_one_out_unscaled"
        )[0],
        "calibrated_loo": compute_per_event_advantage(
            calibrated, estimator="leave_one_out_unscaled"
        )[0],
        "event_zscore": compute_per_event_advantage(
            raw, estimator="zscore", eps=float(zscore_epsilon)
        )[0],
        "centered_rank": centered_candidate_rank(raw),
    }


def _quantiles(values: Tensor) -> dict[str, float]:
    flat = values.detach().double().reshape(-1).cpu()
    return {
        "p01": float(torch.quantile(flat, 0.01)),
        "p50": float(torch.quantile(flat, 0.50)),
        "p99": float(torch.quantile(flat, 0.99)),
    }


def advantage_metrics(advantages: Tensor) -> dict[str, float]:
    """Tail, scale, and event-concentration metrics for one reward arm."""

    if advantages.ndim != 2 or int(advantages.shape[0]) < 2:
        raise ValueError("advantages must have shape (K,B) with K >= 2")
    values = advantages.detach().double()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("advantages contain NaN/Inf")
    event_energy = values.square().mean(dim=0)
    total = event_energy.sum().clamp_min(torch.finfo(event_energy.dtype).tiny)
    count_1pct = max(1, math.ceil(event_energy.numel() * 0.01))
    count_10pct = max(1, math.ceil(event_energy.numel() * 0.10))
    metrics = {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "abs_mean": float(values.abs().mean()),
        "abs_max": float(values.abs().max()),
        "positive_fraction": float((values > 0).double().mean()),
        "negative_fraction": float((values < 0).double().mean()),
        "zero_fraction": float((values == 0).double().mean()),
        "event_rms_median": float(event_energy.sqrt().median()),
        "event_rms_p99": float(torch.quantile(event_energy.sqrt(), 0.99)),
        "top_1pct_event_energy_fraction": float(
            event_energy.topk(count_1pct).values.sum() / total
        ),
        "top_10pct_event_energy_fraction": float(
            event_energy.topk(count_10pct).values.sum() / total
        ),
    }
    metrics.update({f"advantage_{key}": value for key, value in _quantiles(values).items()})
    return metrics


def _cosine(left: Tensor, right: Tensor, floor: float = 1.0e-12) -> float:
    left = left.detach().double().reshape(-1)
    right = right.detach().double().reshape(-1)
    denominator = float(left.norm() * right.norm())
    if denominator <= floor:
        return float("nan")
    return float(torch.clamp(torch.dot(left, right) / denominator, -1.0, 1.0))


def member_ordering_metrics(member_logits: Tensor) -> dict[str, float]:
    """Agreement of M independently fitted members on the same K candidates."""

    if member_logits.ndim != 3:
        raise ValueError("member logits must have shape (M,K,B)")
    members, candidates, _ = member_logits.shape
    if members < 2 or candidates < 2:
        raise ValueError("member ordering metrics need M >= 2 and K >= 2")
    ranks = torch.stack(
        [centered_candidate_rank(member_logits[index]) for index in range(members)]
    )
    advantages = torch.stack(
        [
            compute_per_event_advantage(
                member_logits[index], estimator="leave_one_out_unscaled"
            )[0]
            for index in range(members)
        ]
    )
    rank_cosines, advantage_cosines = [], []
    for left in range(members):
        for right in range(left + 1, members):
            rank_cosines.append(_cosine(ranks[left], ranks[right]))
            advantage_cosines.append(_cosine(advantages[left], advantages[right]))
    positive = advantages > 0
    negative = advantages < 0
    disagreement = positive.any(dim=0) & negative.any(dim=0)
    ensemble = member_logits.mean(dim=0)
    winners = member_logits.argmax(dim=1)
    ensemble_winner = ensemble.argmax(dim=0)
    winner_agreement = (winners == ensemble_winner.unsqueeze(0)).double().mean()
    finite_rank_cosines = [value for value in rank_cosines if math.isfinite(value)]
    finite_advantage_cosines = [
        value for value in advantage_cosines if math.isfinite(value)
    ]
    return {
        "pairwise_rank_cosine_mean": (
            sum(finite_rank_cosines) / len(finite_rank_cosines)
            if finite_rank_cosines
            else float("nan")
        ),
        "pairwise_rank_cosine_min": (
            min(finite_rank_cosines) if finite_rank_cosines else float("nan")
        ),
        "pairwise_advantage_cosine_mean": (
            sum(finite_advantage_cosines) / len(finite_advantage_cosines)
            if finite_advantage_cosines
            else float("nan")
        ),
        "candidate_sign_disagreement_fraction": float(disagreement.double().mean()),
        "winner_agreement_with_ensemble": float(winner_agreement),
    }


def paired_ordering_metrics(
    current_logits: Tensor,
    previous_logits: Tensor,
) -> dict[str, float]:
    """Ordering agreement of two reward ensembles on one fixed ``(K,B)`` panel.

    Inputs are already ensemble-aggregated scores.  Additive event offsets and
    positive scalar calibration therefore cannot change the reported rank or
    leave-one-out direction agreement.
    """

    if current_logits.ndim != 2 or previous_logits.ndim != 2:
        raise ValueError("paired reward logits must both have shape (K,B)")
    if current_logits.shape != previous_logits.shape or int(current_logits.shape[0]) < 2:
        raise ValueError("paired reward panels must have the same K>=2 shape")
    if not bool(
        torch.isfinite(current_logits).all()
        and torch.isfinite(previous_logits).all()
    ):
        raise ValueError("paired reward panels contain NaN/Inf")

    current_rank = centered_candidate_rank(current_logits)
    previous_rank = centered_candidate_rank(previous_logits)
    current_advantage = compute_per_event_advantage(
        current_logits, estimator="leave_one_out_unscaled"
    )[0]
    previous_advantage = compute_per_event_advantage(
        previous_logits, estimator="leave_one_out_unscaled"
    )[0]
    current_winner = current_logits.argmax(dim=0)
    previous_winner = previous_logits.argmax(dim=0)
    current_sign = torch.sign(current_advantage)
    previous_sign = torch.sign(previous_advantage)
    comparable = (current_sign != 0) | (previous_sign != 0)
    if bool(comparable.any()):
        sign_agreement = (
            (current_sign[comparable] == previous_sign[comparable])
            .double()
            .mean()
        )
    else:
        sign_agreement = current_logits.new_tensor(1.0, dtype=torch.float64)
    return {
        "rank_cosine": _cosine(current_rank, previous_rank),
        "advantage_cosine": _cosine(current_advantage, previous_advantage),
        "winner_agreement": float(
            (current_winner == previous_winner).double().mean()
        ),
        "candidate_sign_agreement_fraction": float(sign_agreement),
    }


def consensus_gated_advantage(
    current_logits: Tensor,
    member_logits: Tensor,
    *,
    minimum_sign_agreement: float = 0.75,
    uncertainty_scale: float = 1.0,
    epsilon: float = 1.0e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Downweight a raw-LOO direction when reward models disagree.

    ``current_logits`` is the ordinary current-round ensemble score ``(K,B)``.
    ``member_logits`` may additionally contain recent-round members and has
    shape ``(M,K,B)``.  The base direction is unchanged; member predictions
    only define a confidence gate.  The returned tensor is centered per event,
    so it can be encoded as scores and passed through DGPO's existing raw LOO
    estimator without introducing an event offset.
    """

    if current_logits.ndim != 2 or int(current_logits.shape[0]) < 2:
        raise ValueError("current reward logits must have shape (K,B), K>=2")
    if member_logits.ndim != 3 or member_logits.shape[1:] != current_logits.shape:
        raise ValueError("member reward logits must have shape (M,K,B)")
    if int(member_logits.shape[0]) < 2:
        raise ValueError("consensus gating needs at least two reward models")
    if not 0.5 <= float(minimum_sign_agreement) <= 1.0:
        raise ValueError("minimum_sign_agreement must lie in [0.5,1]")
    if not math.isfinite(float(uncertainty_scale)) or float(uncertainty_scale) < 0.0:
        raise ValueError("uncertainty_scale must be finite and nonnegative")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("consensus epsilon must be finite and positive")
    if not bool(
        torch.isfinite(current_logits).all() and torch.isfinite(member_logits).all()
    ):
        raise ValueError("consensus reward logits contain NaN/Inf")

    base = compute_per_event_advantage(
        current_logits, estimator="leave_one_out_unscaled"
    )[0]
    member_advantages = torch.stack(
        [
            compute_per_event_advantage(
                member_logits[index], estimator="leave_one_out_unscaled"
            )[0]
            for index in range(int(member_logits.shape[0]))
        ]
    )
    base_sign = torch.sign(base).unsqueeze(0)
    member_sign = torch.sign(member_advantages)
    # Exact zero advantages are treated as abstentions rather than votes
    # against the base direction.
    nonzero = member_sign != 0
    agreeing = (member_sign == base_sign) & nonzero
    votes = nonzero.sum(dim=0)
    agreement = agreeing.sum(dim=0).to(base.dtype) / votes.clamp_min(1).to(base.dtype)
    agreement = torch.where(votes > 0, agreement, torch.ones_like(agreement))

    # Remove each member's positive per-event logit scale before estimating
    # uncertainty. Adaptive ESS tempering and classifier calibration may change
    # score magnitude across rounds without changing candidate ordering.
    member_event_rms = member_advantages.square().mean(dim=1, keepdim=True).sqrt()
    member_directions = member_advantages / member_event_rms.clamp_min(
        float(epsilon)
    )
    median = member_directions.median(dim=0).values
    mad = (member_directions - median.unsqueeze(0)).abs().median(dim=0).values
    robust_scale = member_directions.abs().median(dim=0).values.clamp_min(
        float(epsilon)
    )
    relative_mad = mad / robust_scale
    uncertainty_weight = 1.0 / (
        1.0 + float(uncertainty_scale) * relative_mad
    )
    # A candidate whose current ensemble advantage is exactly zero has no
    # actionable direction. Exclude it from agreement-rate denominators and
    # force its gate to zero.
    actionable = base != 0
    pass_gate = (agreement >= float(minimum_sign_agreement)) & actionable
    gate = uncertainty_weight * pass_gate.to(uncertainty_weight.dtype)
    gated = base * gate
    gated = gated - gated.mean(dim=0, keepdim=True)

    base_rms = base.square().mean().sqrt()
    gated_rms = gated.square().mean().sqrt()
    actionable_agreement = agreement[actionable]
    actionable_pass = pass_gate[actionable]
    actionable_gate = gate[actionable]
    return gated, {
        "models": float(member_logits.shape[0]),
        "actionable_fraction": float(actionable.double().mean()),
        "sign_agreement_mean": float(
            actionable_agreement.mean() if actionable_agreement.numel() else 1.0
        ),
        "sign_agreement_p10": float(
            torch.quantile(actionable_agreement.double(), 0.10)
            if actionable_agreement.numel()
            else 1.0
        ),
        "hard_gate_pass_fraction": float(
            actionable_pass.double().mean() if actionable_pass.numel() else 0.0
        ),
        "uncertainty_weight_mean": float(uncertainty_weight.mean()),
        "relative_mad_median": float(relative_mad.median()),
        "zeroed_fraction": float(
            (actionable_gate == 0).double().mean()
            if actionable_gate.numel()
            else 1.0
        ),
        "base_advantage_rms": float(base_rms),
        "gated_advantage_rms": float(gated_rms),
        "rms_ratio": float(gated_rms / base_rms.clamp_min(float(epsilon))),
    }


def vector_cosine(left: Tensor, right: Tensor) -> float:
    """Public cosine helper used for policy-gradient comparisons."""

    return _cosine(left, right)


def vector_concentration(norms: Tensor) -> dict[str, float]:
    """Concentration of per-event gradient squared norms."""

    values = norms.detach().double().reshape(-1)
    if not values.numel() or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("gradient norms must be a non-empty finite nonnegative tensor")
    energy = values.square()
    total = energy.sum().clamp_min(torch.finfo(energy.dtype).tiny)
    one = max(1, math.ceil(len(energy) * 0.01))
    ten = max(1, math.ceil(len(energy) * 0.10))
    return {
        "events": float(len(values)),
        "norm_mean": float(values.mean()),
        "norm_median": float(values.median()),
        "norm_p99": float(torch.quantile(values, 0.99)),
        "top_1pct_gradient_energy_fraction": float(energy.topk(one).values.sum() / total),
        "top_10pct_gradient_energy_fraction": float(energy.topk(ten).values.sum() / total),
    }


def reward_interface_diagnosis(
    signed_results: Mapping[str, Mapping[str, float]],
    *,
    judge_metric: str = "judge_auc_gap",
    response_metric: str = "response_mean_abs_bin_offset",
    tolerance: float = 1.0e-6,
) -> dict[str, str]:
    """Apply the predeclared decision table to completed signed probes.

    Each arm mapping must contain ``minus_*``, ``zero_*`` and ``plus_*`` values.
    Lower judge AUC gap and lower response offset are treated as improvements.
    This function records the observed branch; it does not claim significance.
    """

    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("diagnosis tolerance must be finite and nonnegative")

    def improves(arm: str, metric: str) -> bool:
        row = signed_results[arm]
        plus = float(row[f"plus_{metric}"])
        zero = float(row[f"zero_{metric}"])
        minus = float(row[f"minus_{metric}"])
        return plus < zero - float(tolerance) and plus < minus - float(tolerance)

    usable = {
        arm: improves(arm, judge_metric) and improves(arm, response_metric)
        for arm in REWARD_ARMS
        if arm in signed_results
    }
    if usable.get("calibrated_loo") and not usable.get("raw_loo"):
        # At the exact policy anchor the two gradients differ only by a positive
        # scalar, so a normalized signed-probe difference is numerical or comes
        # from later nonlinear relinearization.  Do not over-interpret it here.
        finding = "scalar_scale_requires_finite_step_relinearization"
    elif (usable.get("event_zscore") or usable.get("centered_rank")) and not usable.get("raw_loo"):
        finding = "within_event_scale_or_tail_heterogeneity"
    elif any(usable.values()):
        finding = "at_least_one_reward_interface_has_a_usable_local_direction"
    else:
        finding = "no_reward_interface_has_a_usable_local_direction"
    return {"finding": finding, "next_step": {
        "scalar_scale_requires_finite_step_relinearization": "inspect the recorded finite-step raw/calibrated gradient relinearization",
        "within_event_scale_or_tail_heterogeneity": "test the winning invariant mapping in a 50-step frozen-reward pilot",
        "at_least_one_reward_interface_has_a_usable_local_direction": "check seed/fold repeatability before a 50-step pilot",
        "no_reward_interface_has_a_usable_local_direction": "run feature/condition ablations before policy training",
    }[finding]}


__all__ = [
    "REWARD_ARMS",
    "advantage_metrics",
    "balanced_logistic_loss",
    "centered_candidate_rank",
    "consensus_gated_advantage",
    "fit_scalar_temperature",
    "member_ordering_metrics",
    "paired_ordering_metrics",
    "reward_advantage_arms",
    "reward_interface_diagnosis",
    "vector_concentration",
    "vector_cosine",
]
