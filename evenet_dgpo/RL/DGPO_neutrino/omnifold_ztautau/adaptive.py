"""Adaptive K=1 probe and training-time refit for Ztautau OmniFold DGPO.

The classifier is always built through :class:`EvenetAdapterModelBuilder` from
this repository.  EveNet-private supplies the controller semantics, not a model
implementation: a fresh event-held-out audit detects stale weights, an accepted
refit installs a new residual ratio stack, and its denominator policy becomes
the DGPO round reference in the same operation.
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EventPackingSpec,
    FrozenResidualRatioReward,
    _score_population,
    fit_independent_evenet_audit,
    fit_residual_ratio_stack,
    peft_bank_factory,
)
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import (
    global_mean_one_from_log_weights,
)
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


_log = logging.getLogger(__name__)

OmniFoldProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


@dataclass(frozen=True)
class AdaptiveOmniFoldConfig:
    enabled: bool
    log_only: bool
    baseline_probe_on_start: bool
    bootstrap_on_start: bool
    bootstrap_fail_closed: bool
    refit_once_on_resume: bool
    refit_once_id: str
    staleness_every_n_epochs: int
    pool_generation_batch_size: int | None
    retrain_auc_margin: float
    required_consecutive_epochs: int
    require_audit_saturation: bool
    power_alpha: float
    power_target: float
    probe_seed: int
    probe_max_events: int | None
    refit_score_events: int | None
    pool_events: int | None
    pool_selection_seed: int
    candidates_per_event: int
    min_iterations: int
    max_iterations: int
    acceptance_max_balanced_accuracy: float
    tempering: float
    crossfit_folds: int
    residual_min_auc_gain: float
    seed: int
    score_row_budget: int
    audit_fit: dict[str, Any]
    fit: dict[str, Any]


def resolve_adaptive_config(dgpo_config: Any) -> AdaptiveOmniFoldConfig:
    block = _cfg_get(dgpo_config, "adaptive_omnifold", None)
    trigger = _cfg_get(block, "trigger", None)
    recal = _cfg_get(block, "recalibration", None)
    config = AdaptiveOmniFoldConfig(
        enabled=bool(_cfg_get(block, "enabled", False)),
        log_only=bool(_cfg_get(block, "log_only", False)),
        baseline_probe_on_start=bool(
            _cfg_get(block, "baseline_probe_on_start", True)
        ),
        bootstrap_on_start=bool(_cfg_get(recal, "bootstrap_on_start", False)),
        bootstrap_fail_closed=bool(
            _cfg_get(recal, "bootstrap_fail_closed", True)
        ),
        refit_once_on_resume=bool(
            _cfg_get(recal, "refit_once_on_resume", False)
        ),
        refit_once_id=str(_cfg_get(recal, "refit_once_id", "")).strip(),
        staleness_every_n_epochs=max(
            1, int(_cfg_get(block, "staleness_every_n_epochs", 1))
        ),
        pool_generation_batch_size=_optional_int(
            _cfg_get(block, "pool_generation_batch_size", None)
        ),
        retrain_auc_margin=float(
            _cfg_get(trigger, "retrain_auc_margin", 0.01)
        ),
        required_consecutive_epochs=max(
            1, int(_cfg_get(trigger, "required_consecutive_epochs", 1))
        ),
        require_audit_saturation=bool(
            _cfg_get(trigger, "require_audit_saturation", True)
        ),
        power_alpha=float(_cfg_get(trigger, "power_alpha", 0.05)),
        power_target=float(_cfg_get(trigger, "power_target", 0.80)),
        probe_seed=int(_cfg_get(trigger, "probe_seed", 20260818)),
        probe_max_events=_optional_int(
            _cfg_get(trigger, "probe_max_events", None)
        ),
        refit_score_events=_optional_int(
            _cfg_get(
                recal,
                "score_pool_events",
                _cfg_get(trigger, "probe_max_events", None),
            )
        ),
        pool_events=_optional_int(_cfg_get(recal, "pool_events", None)),
        pool_selection_seed=int(_cfg_get(recal, "pool_selection_seed", 42)),
        candidates_per_event=int(_cfg_get(recal, "candidates_per_event", 1)),
        min_iterations=int(_cfg_get(recal, "min_iterations", 2)),
        max_iterations=int(_cfg_get(recal, "max_iterations", 12)),
        acceptance_max_balanced_accuracy=float(
            _cfg_get(recal, "acceptance_max_balanced_accuracy", 0.51)
        ),
        tempering=float(_cfg_get(recal, "tempering", 1.0)),
        crossfit_folds=int(_cfg_get(recal, "crossfit_folds", 2)),
        residual_min_auc_gain=float(
            _cfg_get(recal, "residual_min_auc_gain", 1.0e-3)
        ),
        seed=int(_cfg_get(recal, "seed", 20260819)),
        score_row_budget=max(1, int(_cfg_get(recal, "score_row_budget", 512))),
        audit_fit=dict(
            _cfg_get(block, "audit_fit", _cfg_get(recal, "fit", {})) or {}
        ),
        fit=dict(_cfg_get(recal, "fit", {}) or {}),
    )
    if config.candidates_per_event != 1:
        raise ValueError(
            "adaptive OmniFold probe/refit requires candidates_per_event=1; "
            "DGPO's online group size remains dgpo.K"
        )
    if (
        config.pool_generation_batch_size is not None
        and config.pool_generation_batch_size < 1
    ):
        raise ValueError("pool_generation_batch_size must be positive")
    if config.min_iterations < 1 or config.max_iterations < config.min_iterations:
        raise ValueError("adaptive OmniFold iterations require 1 <= min <= max")
    if config.crossfit_folds < 2:
        raise ValueError("adaptive OmniFold crossfit_folds must be at least two")
    if not 0.0 <= config.residual_min_auc_gain < 0.5:
        raise ValueError("residual_min_auc_gain must lie in [0, 0.5)")
    if not 0.5 < config.acceptance_max_balanced_accuracy < 1.0:
        raise ValueError("acceptance_max_balanced_accuracy must lie in (0.5, 1)")
    if not 0.0 < config.retrain_auc_margin < 0.5:
        raise ValueError("retrain_auc_margin must lie in (0, 0.5)")
    if not 0.0 < config.power_alpha < 1.0:
        raise ValueError("power_alpha must lie in (0, 1)")
    if not 0.0 < config.power_target < 1.0:
        raise ValueError("power_target must lie in (0, 1)")
    if config.probe_max_events is not None and config.probe_max_events < 30:
        raise ValueError("probe_max_events must be at least 30")
    if config.refit_score_events is not None and config.refit_score_events < 30:
        raise ValueError("recalibration.score_pool_events must be at least 30")
    if config.pool_events is not None and config.pool_events < 30:
        raise ValueError("recalibration.pool_events must be at least 30")
    return config


def resolve_trigger_threshold(
    baseline_auc_gap: float,
    *,
    retrain_auc_margin: float,
) -> float:
    baseline = float(baseline_auc_gap)
    if not math.isfinite(baseline) or baseline < 0.0:
        raise ValueError("fresh-audit baseline gap must be finite and nonnegative")
    margin = float(retrain_auc_margin)
    if not math.isfinite(margin) or not 0.0 < margin < 0.5:
        raise ValueError("retrain_auc_margin must lie in (0, 0.5)")
    threshold = baseline + margin
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("adaptive OmniFold trigger threshold is invalid")
    return threshold


@dataclass
class AdaptiveOmniFoldState:
    reward_round_id: int = 0
    baseline_auc_gap: float = float("nan")
    # Retained as a diagnostic showing the immediately preceding routine audit.
    # Triggering is anchored to ``baseline_auc_gap`` from the installed reward
    # round, so gradual drift cannot disappear into a rolling one-step baseline.
    previous_audit_auc_gap: float = float("nan")
    audit_protocol_signature: str = ""
    trigger_threshold: float = float("nan")
    probe_exceedance_streak: int = 0
    recalibration_count: int = 0
    recalibrations_rejected: int = 0
    resume_refit_once_completed: bool = False
    resume_refit_once_id: str = ""
    installed_at_epoch: int = -1
    last_recalibration_epoch: int | None = None
    last_decision: str = "uninitialized"
    probe_history: list[dict[str, float]] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return (
            math.isfinite(self.baseline_auc_gap)
            and self.baseline_auc_gap >= 0.0
            and math.isfinite(self.trigger_threshold)
            and self.trigger_threshold > 0.0
        )

    def install(
        self,
        *,
        baseline_auc_gap: float,
        cfg: AdaptiveOmniFoldConfig,
        epoch: int,
        round_id: int | None = None,
    ) -> None:
        self.baseline_auc_gap = float(baseline_auc_gap)
        self.previous_audit_auc_gap = float(baseline_auc_gap)
        self.trigger_threshold = resolve_trigger_threshold(
            baseline_auc_gap,
            retrain_auc_margin=cfg.retrain_auc_margin,
        )
        self.probe_exceedance_streak = 0
        self.installed_at_epoch = int(epoch)
        if round_id is not None:
            self.reward_round_id = int(round_id)

    def invalidate_audit_baseline(self, *, reason: str) -> None:
        """Keep the installed reward round but require a fresh routine baseline."""

        self.baseline_auc_gap = float("nan")
        self.previous_audit_auc_gap = float("nan")
        self.trigger_threshold = float("nan")
        self.probe_exceedance_streak = 0
        self.last_decision = str(reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "AdaptiveOmniFoldState":
        if not payload:
            return cls()
        values = dict(payload)
        values["probe_history"] = [
            dict(item) for item in values.get("probe_history", []) or []
        ]
        state = cls(
            **{
                key: values[key]
                for key in cls.__dataclass_fields__
                if key in values
            }
        )
        # Backward-compatible diagnostic migration for checkpoints written
        # before ``previous_audit_auc_gap`` existed. Triggering no longer uses
        # this value; restored installed-round baselines remain fixed until a
        # replacement reward round is accepted.
        if not math.isfinite(state.previous_audit_auc_gap):
            for item in reversed(state.probe_history):
                candidate = item.get("weighted_auc_gap")
                if candidate is not None and math.isfinite(float(candidate)):
                    state.previous_audit_auc_gap = float(candidate)
                    break
        return state


def adaptive_audit_protocol_signature(cfg: AdaptiveOmniFoldConfig) -> str:
    """Fingerprint the audit protocol that certified an installed baseline."""

    payload = {
        "schema": "single-weighted-audit-asymmetric-stop-v3",
        "probe_max_events": cfg.probe_max_events,
        "require_audit_saturation": cfg.require_audit_saturation,
        "score_row_budget": cfg.score_row_budget,
        "audit_fit": cfg.audit_fit,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def should_probe_epoch(epoch: int, every_n_epochs: int) -> bool:
    if int(epoch) == -1:
        return True
    return (int(epoch) + 1) % max(1, int(every_n_epochs)) == 0


def validate_adaptive_pairing(
    *,
    reward_source: Any,
    state: AdaptiveOmniFoldState,
    round_ref_model: torch.nn.Module,
    checkpoint: Mapping[str, Any] | None = None,
    where: str,
) -> None:
    """Fail closed if controller, ratio denominator, and round anchor diverge."""
    from RL.DGPO_neutrino.model_utils import state_dict_sha256

    reward_round = int(reward_source.reward_round_id)
    if int(state.reward_round_id) != reward_round:
        raise ValueError(f"{where}: controller and reward round disagree")
    observed = state_dict_sha256(round_ref_model)
    if reward_round > 0:
        if str(reward_source.reference_kind) != "state_dict_sha256":
            raise ValueError(f"{where}: dynamic reward lacks a state-dict reference")
        if observed != str(reward_source.policy_reference_sha256):
            raise ValueError(f"{where}: reward and round-reference policy disagree")
    if checkpoint is not None:
        saved_round = checkpoint.get("dgpo_reward_round_id")
        if saved_round is not None and int(saved_round) != reward_round:
            raise ValueError(f"{where}: checkpoint and reward round disagree")
        saved_digest = checkpoint.get("dgpo_round_ref_sha256")
        if saved_digest is not None and str(saved_digest) != observed:
            raise ValueError(f"{where}: checkpointed round-reference digest is invalid")


def update_controller(
    state: AdaptiveOmniFoldState,
    probe: Mapping[str, float],
    *,
    cfg: AdaptiveOmniFoldConfig,
    epoch: int,
) -> tuple[bool, dict[str, Any]]:
    if not state.calibrated:
        raise RuntimeError("adaptive OmniFold controller has no installed baseline")
    gap = float(probe["weighted_auc_gap"])
    previous_gap = float(state.previous_audit_auc_gap)
    audit_saturated = float(probe.get("audit_saturated", 0.0)) >= 0.5
    audit_threshold_reached = (
        float(probe.get("audit_threshold_reached", 0.0)) >= 0.5
    )
    decision_threshold = resolve_trigger_threshold(
        state.baseline_auc_gap,
        retrain_auc_margin=cfg.retrain_auc_margin,
    )
    state.trigger_threshold = float(decision_threshold)
    final_gap_exceeded = bool(
        not math.isfinite(gap) or gap > decision_threshold
    )
    # A routine staleness audit may stop as soon as its early-stop AUC gap
    # crosses the installed-round threshold, but the untouched final split must
    # independently confirm that crossing. A non-crossing result still needs
    # the configured saturation window before it can certify "healthy".
    decision_eligible = bool(
        audit_saturated
        or (audit_threshold_reached and final_gap_exceeded)
        or not cfg.require_audit_saturation
    )
    exceeded = bool(
        decision_eligible and final_gap_exceeded
    )
    state.probe_exceedance_streak = (
        state.probe_exceedance_streak + 1
        if exceeded
        else 0
    )
    fired = state.probe_exceedance_streak >= cfg.required_consecutive_epochs
    recalibrate = bool(fired and not cfg.log_only)
    decision = (
        "audit_unsaturated"
        if not decision_eligible
        else "stale_log_only"
        if fired and cfg.log_only
        else "recalibrate"
        if recalibrate
        else "threshold_exceeded"
        if exceeded
        else "healthy"
    )
    state.last_decision = decision
    row = {
        "epoch": float(epoch),
        "reward_round_id": float(state.reward_round_id),
        "decision_recalibrate": float(recalibrate),
        **{key: float(value) for key, value in probe.items()},
    }
    state.probe_history.append(row)
    if len(state.probe_history) > 256:
        del state.probe_history[:-256]
    diagnostics: dict[str, Any] = {
        **{f"staleness/{key}": value for key, value in row.items()},
        "staleness/decision": decision,
        "staleness/trigger_recalibration": float(recalibrate),
        "staleness/trigger_threshold": float(decision_threshold),
        "staleness/previous_audit_auc_gap": float(previous_gap),
        "staleness/baseline_auc_gap": float(state.baseline_auc_gap),
        "staleness/auc_gap_delta_from_baseline": float(
            gap - state.baseline_auc_gap
        ),
        "staleness/required_auc_gap_increase": float(
            cfg.retrain_auc_margin
        ),
        "staleness/probe_exceedance_streak": float(
            state.probe_exceedance_streak
        ),
        "staleness/required_consecutive_epochs": float(
            cfg.required_consecutive_epochs
        ),
        "staleness/log_only": float(cfg.log_only),
    }
    # Keep the previous routine result for diagnostics only.  The trigger anchor
    # remains the installed reward's certified baseline, including after a
    # rejected refit, so slow cumulative drift and retry eligibility are kept.
    if decision_eligible and math.isfinite(gap) and gap >= 0.0:
        state.previous_audit_auc_gap = float(gap)
    diagnostics["staleness/next_trigger_threshold"] = float(
        state.trigger_threshold
    )
    return recalibrate, diagnostics


@dataclass(frozen=True)
class AdaptiveOmniFoldPool:
    packed_event: Tensor
    truth: Tensor
    candidates: Tensor
    packing_spec: EventPackingSpec

    def __post_init__(self) -> None:
        if self.packed_event.ndim != 2:
            raise ValueError("adaptive pool packed_event must be (N,C)")
        if self.truth.ndim != 2 or int(self.truth.shape[-1]) != 4:
            raise ValueError("adaptive pool truth must be (N,4)")
        if self.candidates.ndim != 3 or int(self.candidates.shape[-1]) != 4:
            raise ValueError("adaptive pool candidates must be (N,K,4)")
        n_events = int(self.packed_event.shape[0])
        if int(self.truth.shape[0]) != n_events or int(self.candidates.shape[0]) != n_events:
            raise ValueError("adaptive pool event axes do not match")
        if int(self.candidates.shape[1]) != 1:
            raise ValueError("adaptive OmniFold pools must contain exactly K=1")

    @property
    def n_events(self) -> int:
        return int(self.packed_event.shape[0])

    def to(self, device: torch.device) -> "AdaptiveOmniFoldPool":
        return AdaptiveOmniFoldPool(
            packed_event=self.packed_event.to(device=device, dtype=torch.float32),
            truth=self.truth.to(device=device, dtype=torch.float32),
            candidates=self.candidates.to(device=device, dtype=torch.float32),
            packing_spec=self.packing_spec,
        )


def gather_pool_across_ranks(
    local: Mapping[str, Any], *, world_size: int
) -> dict[str, Any]:
    """All-gather local pool shards so distributed ratio fitting sees one pool."""
    if int(world_size) <= 1:
        return {
            key: value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in local.items()
        }
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("adaptive OmniFold pool gather needs torch.distributed")
    gathered: dict[str, Any] = {}
    for key in sorted(local):
        value = local[key]
        item = value.detach().cpu() if isinstance(value, Tensor) else value
        bucket: list[Any] = [None] * int(world_size)
        dist.all_gather_object(bucket, item)
        if isinstance(item, Tensor):
            gathered[key] = torch.cat(bucket, dim=0)
        else:
            if any(other != bucket[0] for other in bucket[1:]):
                raise RuntimeError(f"adaptive pool metadata {key!r} differs across ranks")
            gathered[key] = bucket[0]
    return gathered


@torch.no_grad()
def score_reward_on_pool(
    stack: FrozenResidualRatioReward,
    pool: AdaptiveOmniFoldPool,
    *,
    row_budget: int,
) -> Tensor:
    try:
        device = next(stack.parameters()).device
    except StopIteration:
        device = pool.packed_event.device
    aligned = pool if pool.packed_event.device == device else pool.to(device)
    score = _score_population(
        stack,
        aligned.packed_event,
        aligned.candidates,
        int(row_budget),
    )
    if tuple(score.shape) != tuple(aligned.candidates.shape[:2]):
        raise RuntimeError(
            f"adaptive reward returned {tuple(score.shape)}, expected "
            f"{tuple(aligned.candidates.shape[:2])}"
        )
    if not bool(torch.isfinite(score).all().item()):
        raise FloatingPointError("adaptive OmniFold reward produced NaN/Inf")
    return score


def _weight_diagnostics(log_weight: Tensor) -> dict[str, float]:
    weights = global_mean_one_from_log_weights(log_weight).reshape(-1)
    n_rows = max(int(weights.numel()), 1)
    ess = weights.sum().square() / weights.square().sum().clamp_min(1.0e-12)
    return {
        "ess_fraction": float((ess / n_rows).detach().cpu()),
        "top1_weight_share": float(
            (weights.max() / weights.sum().clamp_min(1.0e-12)).detach().cpu()
        ),
        "reward_mean": float(log_weight.mean().detach().cpu()),
        "reward_std": float(log_weight.std(unbiased=False).detach().cpu()),
        "probe_events": float(log_weight.shape[0]),
        "probe_rows": float(log_weight.numel()),
    }


_STANDARD_NORMAL = NormalDist()


def _two_sided_normal_power(effect: float, standard_error: float, alpha: float) -> float:
    """Approximate two-sided power for an AUC departure from chance."""

    if not (
        math.isfinite(effect)
        and math.isfinite(standard_error)
        and standard_error > 0.0
        and 0.0 < alpha < 1.0
    ):
        return float("nan")
    critical = _STANDARD_NORMAL.inv_cdf(1.0 - 0.5 * alpha)
    shifted = abs(float(effect)) / float(standard_error)
    return float(
        1.0
        - _STANDARD_NORMAL.cdf(critical - shifted)
        + _STANDARD_NORMAL.cdf(-critical - shifted)
    )


def _auc_power_diagnostics(
    *,
    observed_weighted_gap: float,
    audit_events: int,
    ess_fraction: float,
    retrain_auc_margin: float,
    alpha: float,
    target_power: float,
) -> dict[str, float]:
    """Null-Mann-Whitney AUC uncertainty using weight ESS as effective Gen N.

    This is an explicit design/power approximation, not a replacement for the
    event-held-out audit. It shows whether a near-chance result is precise enough
    to resolve the configured retraining margin.
    """

    n_truth = max(float(audit_events), 1.0)
    n_gen_weighted = max(n_truth * max(min(float(ess_fraction), 1.0), 0.0), 1.0)

    def _null_se(n_positive: float, n_negative: float) -> float:
        return math.sqrt(
            (n_positive + n_negative + 1.0)
            / (12.0 * n_positive * n_negative)
        )

    weighted_se = _null_se(n_truth, n_gen_weighted)

    def _pvalue(gap: float, se: float) -> float:
        z_value = abs(float(gap)) / se
        return float(math.erfc(z_value / math.sqrt(2.0)))

    lower, upper = 0.0, 0.5
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _two_sided_normal_power(midpoint, weighted_se, alpha) >= target_power:
            upper = midpoint
        else:
            lower = midpoint
    achieved_power = _two_sided_normal_power(
        retrain_auc_margin, weighted_se, alpha
    )
    return {
        "audit_auc_null_se_approx": float(weighted_se),
        "audit_weighted_auc_gap_z_approx": float(
            abs(float(observed_weighted_gap)) / weighted_se
        ),
        "audit_weighted_auc_gap_pvalue_approx": _pvalue(
            observed_weighted_gap, weighted_se
        ),
        "audit_power_alpha": float(alpha),
        "audit_power_target": float(target_power),
        "audit_power_at_retrain_margin": float(achieved_power),
        "audit_minimum_detectable_auc_gap": float(upper),
        "audit_power_sufficient": float(achieved_power >= target_power),
        "audit_effective_truth_events": float(n_truth),
        "audit_effective_gen_events": float(n_gen_weighted),
    }


def fit_fresh_audit(
    *,
    pool: AdaptiveOmniFoldPool,
    log_weight: Tensor,
    model_builder: Any,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int,
    early_stop_auc_gap: float | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, float]:
    if pool.n_events < 30:
        raise ValueError("fresh adaptive OmniFold audit needs at least 30 events")
    aligned = pool.to(device)
    aligned_log_weight = log_weight.to(device=device, dtype=torch.float32)
    weights = global_mean_one_from_log_weights(aligned_log_weight)
    resolved_fit_block = {**cfg.audit_fit, "require_saturation": False}
    n_fit = max(1, int(round(0.60 * pool.n_events)))
    n_valid = max(1, int(round(0.20 * pool.n_events)))
    fit_config = build_fit_config(
        resolved_fit_block,
        n_train=n_fit,
        n_validation=n_valid,
    )
    factory = peft_bank_factory(
        model_builder,
        pool.packing_spec,
        "audit",
        reset=True,
    )
    weighted_bank_name = "audit"
    try:
        weighted_result = fit_independent_evenet_audit(
            model_factory=factory,
            data_condition=aligned.packed_event,
            data_sample=aligned.truth,
            gen_condition=aligned.packed_event,
            gen_sample=aligned.candidates,
            gen_weight=weights,
            fit_config=fit_config,
            seed=int(seed),
            early_stop_auc_gap=early_stop_auc_gap,
            progress_callback=progress_callback,
        )
    finally:
        discard = getattr(model_builder, "discard_bank", None)
        if callable(discard):
            discard(weighted_bank_name)

    fit_diag = weighted_result.fit_diagnostics
    saturated = bool(getattr(fit_diag, "saturated", False))
    threshold_reached = bool(getattr(fit_diag, "threshold_reached", False))
    observed_gap = float(weighted_result.auc_gap)

    def _finite_or_nan(diagnostics: Any, name: str) -> float:
        value = getattr(diagnostics, name, None)
        return float("nan") if value is None else float(value)

    metrics = {
        "weighted_auc_gap": observed_gap,
        "audit_observed_auc_gap": observed_gap,
        "judge_auc_weighted": float(weighted_result.auc),
        "audit_balanced_accuracy": float(weighted_result.balanced_accuracy),
        "audit_saturated": float(saturated),
        "audit_threshold_reached": float(threshold_reached),
        "audit_fresh_pretrained_initialization": 1.0,
        "audit_reused_previous_classifier": 0.0,
        "audit_saturation_required_for_trigger": float(
            cfg.require_audit_saturation
        ),
        "audit_validation_loss": _finite_or_nan(fit_diag, "validation_loss"),
        "audit_validation_balanced_accuracy": _finite_or_nan(
            fit_diag,
            "validation_balanced_accuracy"
        ),
        "audit_validation_auc": _finite_or_nan(
            fit_diag, "validation_auc"
        ),
        "audit_fit_events": float(weighted_result.fit_events),
        "audit_test_events": float(weighted_result.audit_events),
        "audit_probe_events": float(pool.n_events),
    }
    weight_diag = _weight_diagnostics(log_weight)
    metrics.update(
        _auc_power_diagnostics(
            observed_weighted_gap=observed_gap,
            audit_events=int(weighted_result.audit_events),
            ess_fraction=float(weight_diag["ess_fraction"]),
            retrain_auc_margin=float(cfg.retrain_auc_margin),
            alpha=float(cfg.power_alpha),
            target_power=float(cfg.power_target),
        )
    )
    return metrics


def probe_installed_reward(
    reward_source: Any,
    pool: AdaptiveOmniFoldPool,
    *,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int | None = None,
    early_stop_auc_gap: float | None = None,
    progress_callback: OmniFoldProgressCallback | None = None,
) -> dict[str, float]:
    log_weight = score_reward_on_pool(
        reward_source.frozen_reward,
        pool,
        row_budget=cfg.score_row_budget,
    )
    probe = _weight_diagnostics(log_weight)
    probe.update(
        fit_fresh_audit(
            pool=pool,
            log_weight=log_weight,
            model_builder=reward_source.model_builder,
            cfg=cfg,
            device=device,
            seed=cfg.probe_seed if seed is None else int(seed),
            early_stop_auc_gap=early_stop_auc_gap,
            progress_callback=(
                None
                if progress_callback is None
                else lambda row: progress_callback("staleness_audit", row)
            ),
        )
    )
    return probe


def _broadcast_bool(value: bool, *, world_size: int, device: torch.device) -> bool:
    if int(world_size) <= 1:
        return bool(value)
    payload = torch.tensor([int(bool(value))], device=device, dtype=torch.int64)
    dist.broadcast(payload, src=0)
    return bool(int(payload.item()))


def run_adaptive_refit(
    *,
    state: AdaptiveOmniFoldState,
    cfg: AdaptiveOmniFoldConfig,
    reward_source: Any,
    round_ref_model: torch.nn.Module,
    policy_snapshot_state_dict: Mapping[str, Tensor],
    fit_pool: AdaptiveOmniFoldPool,
    score_pool: AdaptiveOmniFoldPool,
    epoch: int,
    device: torch.device,
    world_size: int,
    progress_callback: OmniFoldProgressCallback | None = None,
) -> dict[str, Any]:
    """Fit, independently audit, and atomically install reward/reference pair."""
    if fit_pool.packing_spec != score_pool.packing_spec:
        raise RuntimeError("adaptive fit and held-out pools use different EveNet shapes")
    score_on_device = score_pool.to(device)
    fit_config = build_fit_config(
        cfg.fit,
        n_train=fit_pool.n_events,
        n_validation=score_pool.n_events,
    )
    reward_bank_name = "adaptive_reward"
    factory = peft_bank_factory(
        reward_source.model_builder,
        fit_pool.packing_spec,
        reward_bank_name,
        reset=True,
    )
    try:
        result = fit_residual_ratio_stack(
            model_factory=factory,
            data_condition=fit_pool.packed_event,
            data_sample=fit_pool.truth,
            gen_condition=fit_pool.packed_event,
            gen_sample=fit_pool.candidates,
            iterations=cfg.max_iterations,
            min_iterations=cfg.min_iterations,
            fit_config=fit_config,
            tempering=cfg.tempering,
            crossfit_folds=cfg.crossfit_folds,
            residual_min_auc_gain=cfg.residual_min_auc_gain,
            seed=cfg.seed + int(state.recalibration_count),
            validation_data_condition=score_on_device.packed_event,
            validation_data_sample=score_on_device.truth,
            validation_gen_condition=score_on_device.packed_event,
            validation_gen_sample=score_on_device.candidates,
            device=device,
            progress_callback=(
                None
                if progress_callback is None
                else lambda row: progress_callback("residual_reward", row)
            ),
        )
    except RuntimeError as exc:
        message = str(exc)
        if not any(
            marker in message
            for marker in (
                "did not saturate",
                "did not enter closure band",
                "failed the null/AUC gate",
                "stopped before min_iterations",
                "did not produce a held-out no-op",
                "unsaturated",
            )
        ):
            raise
        state.recalibrations_rejected += 1
        state.probe_exceedance_streak = 0
        state.last_decision = "recalibration_failed"
        return {
            "omnifold/accepted": 0.0,
            "omnifold/accept_reason": message,
            "omnifold/reward_round_id": float(state.reward_round_id),
            "omnifold/recalibrations_rejected": float(
                state.recalibrations_rejected
            ),
        }

    new_stack = FrozenResidualRatioReward.from_fit_result(
        result,
        tempering=cfg.tempering,
    ).to(device).eval()
    new_stack.assert_frozen()
    acceptance_seed = cfg.probe_seed + 1000 + int(state.recalibration_count)
    initial_bootstrap = not bool(getattr(reward_source, "is_installed", True))
    candidate_log_weight = score_reward_on_pool(
        new_stack,
        score_pool,
        row_budget=cfg.score_row_budget,
    )
    candidate_probe = _weight_diagnostics(candidate_log_weight)
    candidate_probe.update(
        fit_fresh_audit(
            pool=score_pool,
            log_weight=candidate_log_weight,
            model_builder=reward_source.model_builder,
            cfg=cfg,
            device=device,
            seed=acceptance_seed,
            progress_callback=(
                None
                if progress_callback is None
                else lambda row: progress_callback("acceptance_audit", row)
            ),
        )
    )
    all_saturated = bool(result.diagnostics) and all(
        bool(getattr(item, "saturated", False)) for item in result.diagnostics
    )
    audit_gap = float(candidate_probe["audit_observed_auc_gap"])
    audit_accuracy = float(candidate_probe["audit_balanced_accuracy"])
    accuracy_limit = float(cfg.acceptance_max_balanced_accuracy)
    accepted_local = bool(
        all_saturated
        and candidate_probe.get("audit_saturated", 0.0) >= 0.5
        and math.isfinite(audit_accuracy)
        and audit_accuracy < accuracy_limit
    )
    accepted = _broadcast_bool(
        accepted_local,
        world_size=world_size,
        device=device,
    )
    diagnostics: dict[str, Any] = {
        "omnifold/accepted": float(accepted),
        "omnifold/iterations_fitted": float(result.iterations),
        "omnifold/classifier_fits_total": float(
            sum(
                len(getattr(item, "fold_diagnostics", (item,)))
                for item in result.diagnostics
            )
        ),
        "omnifold/all_fits_saturated": float(all_saturated),
        "omnifold/acceptance_max_balanced_accuracy": accuracy_limit,
        "omnifold/initial_bootstrap": float(initial_bootstrap),
        **{
            f"omnifold/candidate/{key}": float(value)
            for key, value in candidate_probe.items()
        },
    }
    for index, fit_diag in enumerate(result.diagnostics, start=1):
        prefix = f"omnifold/fit/iter{index:02d}"
        diagnostics[f"{prefix}/saturated"] = float(
            bool(getattr(fit_diag, "saturated", False))
        )
        diagnostics[f"{prefix}/stored_in_reward"] = float(
            index <= int(result.iterations)
        )
        for source_name, metric_name in (
            ("validation_loss", "validation_loss"),
            ("validation_balanced_accuracy", "validation_balanced_accuracy"),
            ("validation_auc", "validation_auc"),
            ("null_validation_loss", "null_validation_loss"),
            ("validation_loss_gain", "validation_loss_gain"),
            ("final_loss", "training_loss"),
            ("final_accuracy", "training_balanced_accuracy"),
        ):
            value = getattr(fit_diag, source_name, None)
            if value is not None and math.isfinite(float(value)):
                diagnostics[f"{prefix}/{metric_name}"] = float(value)
    if not accepted:
        state.recalibrations_rejected += 1
        state.probe_exceedance_streak = 0
        state.last_decision = "recalibration_rejected"
        accept_reason = (
            "candidate did not reach saturated balanced-accuracy closure: "
            f"accuracy={audit_accuracy:.5g}, required<{accuracy_limit:.5g}, "
            f"fits_saturated={all_saturated}, "
            f"audit_saturated={bool(candidate_probe.get('audit_saturated', 0.0))}"
        )
        diagnostics.update(
            {
                "omnifold/accept_reason": accept_reason,
                "omnifold/recalibrations_rejected": float(
                    state.recalibrations_rejected
                ),
                "omnifold/reward_round_id": float(state.reward_round_id),
            }
        )
        discard = getattr(reward_source.model_builder, "discard_bank", None)
        if callable(discard):
            discard(reward_bank_name)
        return diagnostics

    from RL.DGPO_neutrino.model_utils import (
        freeze_reference_model,
        state_dict_sha256,
    )

    new_round = int(state.reward_round_id) + 1
    previous_round_reference = {
        key: value.detach().cpu().clone()
        for key, value in round_ref_model.state_dict().items()
    }
    try:
        round_ref_model.load_state_dict(dict(policy_snapshot_state_dict), strict=True)
        freeze_reference_model(round_ref_model)
        reference_digest = state_dict_sha256(round_ref_model)
        reward_source.replace_stack(
            new_stack,
            round_id=new_round,
            reference_sha256=reference_digest,
            reference_kind="state_dict_sha256",
        )
    except BaseException:
        round_ref_model.load_state_dict(previous_round_reference, strict=True)
        freeze_reference_model(round_ref_model)
        raise
    state.install(
        baseline_auc_gap=audit_gap,
        cfg=cfg,
        epoch=epoch,
        round_id=new_round,
    )
    state.recalibration_count += 1
    state.last_recalibration_epoch = int(epoch)
    state.last_decision = "recalibration_installed"
    diagnostics.update(
        {
            "omnifold/accept_reason": (
                "candidate reached saturated balanced-accuracy closure"
            ),
            "omnifold/reward_round_id": float(new_round),
            "omnifold/reference_sha256": reference_digest,
            "omnifold/trigger_threshold": float(state.trigger_threshold),
            "omnifold/recalibration_count": float(state.recalibration_count),
        }
    )
    _log.info(
        "[DGPO/omnifold] installed adaptive round %s at epoch %s (anchor=%s)",
        new_round,
        epoch,
        reference_digest[:12],
    )
    return diagnostics


__all__ = [
    "AdaptiveOmniFoldConfig",
    "AdaptiveOmniFoldPool",
    "AdaptiveOmniFoldState",
    "adaptive_audit_protocol_signature",
    "gather_pool_across_ranks",
    "probe_installed_reward",
    "resolve_adaptive_config",
    "run_adaptive_refit",
    "should_probe_epoch",
    "update_controller",
    "validate_adaptive_pairing",
]
