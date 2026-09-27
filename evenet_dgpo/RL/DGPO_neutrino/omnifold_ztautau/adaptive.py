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
    event_identity_inputs,
    FrozenResidualRatioReward,
    _score_population,
    _crossfit_repeat_seed,
    _identity_crossfit_splits,
    fit_independent_evenet_audit,
    fit_residual_ratio_stack,
    pack_event_inputs,
    peft_bank_factory,
    unpack_event_inputs,
    validate_monitor_training_readiness,
)
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import (
    global_mean_one_from_log_weights,
)
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
from RL.DGPO_neutrino.omnifold_ztautau.rest_frame import REST_FRAME_KEY


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return None if not normalized or normalized.lower() == "null" else normalized


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of numbers")
    return tuple(float(item) for item in value)


def _raw_patience_schedule(value: Any) -> tuple[tuple[int, int], ...]:
    """Validate an optional monotone (start_step, required_checks) schedule."""
    if value is None:
        return ()
    name = "trigger.patience_schedule"
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of start_step/required_consecutive_checks entries")
    stages = []
    for entry in value:
        start = _cfg_get(entry, "start_step", None)
        checks = _cfg_get(entry, "required_consecutive_checks", None)
        if type(start) is not int or start < 0 or type(checks) is not int or checks < 1:
            raise ValueError(f"{name} requires nonnegative integer start_step and positive integer checks")
        if (not stages and start != 0) or (stages and start <= stages[-1][0]):
            raise ValueError(f"{name} must start at step 0 with strictly increasing start steps")
        if stages and checks < stages[-1][1]:
            raise ValueError(f"{name} patience must be nondecreasing")
        stages.append((start, checks))
    return tuple(stages)


def _residual_closure_schedule(value: Any) -> tuple[tuple[int, float], ...]:
    """Optional step-based tightening of the residual (not raw) AUC gate."""
    if value is None:
        return ()
    name = "recalibration.residual_closure_schedule"
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    stages = []
    for entry in value:
        start = _cfg_get(entry, "start_step", None)
        auc = _cfg_get(entry, "max_auc", None)
        if (type(start) is not int or start < 0 or isinstance(auc, bool)
                or not isinstance(auc, (int, float)) or not math.isfinite(auc)
                or not .5 < auc < 1.):
            raise ValueError(f"{name} requires nonnegative integer start_step and 0.5 < max_auc < 1")
        if (not stages and start != 0) or (stages and start <= stages[-1][0]):
            raise ValueError(f"{name} must start at 0 with strictly increasing steps")
        if stages and auc > stages[-1][1]:
            raise ValueError(f"{name} max_auc must be nonincreasing")
        stages.append((start, float(auc)))
    return tuple(stages)


def _classifier_architecture_overrides(
    value: Any,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("recalibration.reward_classifier must be a mapping")
    overrides = dict(value)
    dropout_keys = {"head_dropout", "topology_dropout"}
    scale_keys = {"topology_context_residual_scale"}
    dimension_keys = {"decoder_hidden_dim", "decoder_layers", "decoder_heads", "relation_token_count"}
    flag_keys = {
        "periodic_pair_features",
        "topology_fourier_embedding",
        "topology_direct_logit",
        "topology_conditioning",
        "topology_pair_token",
        "visible_pair_rest_frame",
    }
    unknown = set(overrides) - dropout_keys - scale_keys - dimension_keys - flag_keys
    if unknown:
        raise ValueError(
            "recalibration.reward_classifier has unsupported keys: "
            f"{sorted(unknown)}"
        )
    if any(
        isinstance(overrides[key], bool)
        or not isinstance(overrides[key], (int, float))
        or not math.isfinite(float(overrides[key]))
        or not 0.0 <= float(overrides[key]) < 1.0
        for key in dropout_keys & overrides.keys()
    ):
        raise ValueError("reward classifier dropout must be finite and in [0, 1)")
    if any(
        type(overrides[key]) is not int or int(overrides[key]) < 1
        for key in dimension_keys & overrides.keys()
    ):
        raise ValueError("reward classifier dimensions must be positive integers")
    if any(type(overrides[key]) is not bool for key in flag_keys & overrides.keys()):
        raise ValueError("reward classifier feature flags must be boolean")
    if any(
        isinstance(overrides[key], bool)
        or not isinstance(overrides[key], (int, float))
        or not math.isfinite(float(overrides[key]))
        or not 0.0 <= float(overrides[key]) <= 1.0
        for key in scale_keys & overrides.keys()
    ):
        raise ValueError(
            "reward classifier residual scales must be finite and in [0, 1]"
        )
    return overrides


@dataclass(frozen=True)
class AdaptiveOmniFoldConfig:
    enabled: bool
    log_only: bool
    monitor_mode: str
    baseline_probe_on_start: bool
    bootstrap_on_start: bool
    bootstrap_fail_closed: bool
    refit_once_on_resume: bool
    refit_once_fail_closed: bool
    refit_once_id: str
    staleness_every_n_epochs: int
    fixed_audit_panel: bool
    pool_generation_batch_size: int | None
    retrain_auc_margin: float
    raw_audit_enabled: bool
    raw_improvement_min_delta: float
    raw_rollback_to_best_on_plateau: bool
    max_reward_age_epochs: int | None
    fixed_schedule_skip_staleness_audit: bool
    fixed_schedule_log_raw_audit: bool
    required_consecutive_epochs: int
    retrain_cooldown_epochs: int
    classifier_trust_enabled: bool
    classifier_trust_every_n_epochs: int
    classifier_trust_probe_max_events: int | None
    classifier_trust_max_balanced_accuracy: float
    classifier_trust_confidence_z: float
    classifier_trust_required_consecutive_epochs: int
    classifier_trust_require_saturation: bool
    classifier_trust_audit_fit: dict[str, Any]
    classifier_trust_unsafe_early_stop_enabled: bool
    classifier_trust_unsafe_early_stop_confidence_z: float
    classifier_trust_unsafe_early_stop_required_consecutive_validations: int
    trust_boundary_enabled: bool
    trust_radius_mode: str
    trust_distance: str
    trust_delta_max: float
    trust_delta_floor: float
    trust_initial_raw_auc_gap: float | None
    trust_warning_fraction: float
    trust_adaptive_power: float
    trust_auc_confidence_z: float
    trust_auc_stop_gap: float
    trust_reset_adam_first_moment: bool
    trust_reset_adam_first_moment_on_zero_step: bool
    trust_transactional_rejection: bool
    trust_stop_after_rejection: bool
    trust_enforcement: str
    trust_backtrack_factor: float
    trust_max_backtracks: int
    trust_probe_events_per_rank: int
    trust_fixed_probe_per_reward_round: bool
    trust_interior_fraction: float
    trust_empirical_radius_enabled: bool
    trust_empirical_bidirectional: bool
    trust_empirical_allow_expansion: bool
    trust_cross_round_nonexpanding: bool
    trust_policy_lr_scaling_enabled: bool
    trust_policy_lr_scale_floor: float
    trust_refit_on_exhaustion: bool
    trust_exhaustion_distance_fraction: float
    trust_exhaustion_mean_update_scale: float
    trust_exhaustion_scale_window_steps: int
    trust_round_acceptance_enabled: bool
    trust_round_acceptance_confidence_z: float
    trust_round_plateau_patience: int
    trust_trajectory_search_enabled: bool
    trust_failed_direction_patience: int
    trust_signed_direction_probe_enabled: bool
    trust_signed_direction_probe_scales: tuple[float, ...]
    trust_signed_direction_recovery_enabled: bool
    trust_extragradient_enabled: bool
    trust_extragradient_lookahead_scale: float
    trust_empirical_safety_factor: float
    trust_empirical_expand_factor: float
    trust_empirical_shrink_factor: float
    trust_empirical_target_acceptance_rate: float
    trust_empirical_target_update_scale: float
    trust_empirical_safe_audits_required: int
    trust_empirical_attempt_window_steps: int
    trust_empirical_distance_window_steps: int
    trust_empirical_min_distance_samples: int
    trust_empirical_confidence_z: float
    require_audit_saturation: bool
    power_alpha: float
    power_target: float
    probe_seed: int
    probe_max_events: int | None
    refit_score_events: int | None
    pool_data_parquet_dir: str | None
    pool_events: int | None
    pool_selection_seed: int
    candidates_per_event: int
    min_iterations: int
    max_iterations: int
    acceptance_audit_enabled: bool
    acceptance_max_balanced_accuracy: float
    periodic_pair_features_enabled: bool
    visible_pair_rest_frame_enabled: bool
    topology_fourier_embedding_enabled: bool
    topology_max_harmonic: int
    topology_include_theta_pair: bool
    topology_theta_fourier: bool
    topology_acceptance_audit_enabled: bool
    topology_acceptance_max_auc_gap: float
    topology_acceptance_repeats: int
    tempering: float
    adaptive_tempering_enabled: bool
    target_ess_fraction: float
    minimum_tempering: float
    tempering_grid_steps: int
    inherit_previous_tempering: bool
    ess_aware_checkpoint_selection: bool
    ess_aware_max_checkpoints: int
    ess_aware_first_residual_only: bool
    log_ratio_clip: float | None
    minimum_ess_fraction: float
    crossfit_folds: int
    crossfit_repeats: int
    residual_min_auc_gain: float
    seed: int
    score_row_budget: int
    audit_fit: dict[str, Any]
    reward_classifier: dict[str, Any]
    fit: dict[str, Any]
    residual_closure_schedule: tuple[tuple[int, float], ...] = ()
    warm_start_iterations: tuple[int, ...] = ()
    warm_start_from_iteration_one: bool = False
    crossfit_partition: str = "auto"
    trust_round_decay_factor: float = 0.9
    trust_best_decay_factor: float = 0.9
    raw_monitor_warm_start: bool = False
    staleness_every_n_steps: int | None = None
    single_pool_train_validation: bool = False
    single_pool_split_seed: int = 42
    cache_event_inputs: bool = False
    reset_optimizer_state_on_install: bool = False
    policy_warmup_steps: int = 0
    policy_warmup_start_factor: float = 0.1
    raw_best_scope: str = "round"
    raw_global_confirm_candidates: bool = False
    raw_global_max_failed_rounds: int = 2
    raw_global_confirmation_fit: dict[str, Any] = field(default_factory=dict)
    raw_global_confirmation_warm_start: bool = False
    raw_pause_patience_during_warmup: bool = False
    raw_patience_schedule: tuple[tuple[int, int], ...] = ()
    iteration_one_only: bool = False
    fixed_iteration_budget: bool = False
    max_reward_rounds: int | None = None
    scheduled_refit_fail_closed: bool = False


def resolve_adaptive_config(
    dgpo_config: Any, *, classifier_only: bool = False,
) -> AdaptiveOmniFoldConfig:
    block = _cfg_get(dgpo_config, "adaptive_omnifold", None)
    trigger = _cfg_get(block, "trigger", None)
    classifier_trust = _cfg_get(block, "classifier_trust", None)
    classifier_trust_unsafe_early_stop = _cfg_get(
        classifier_trust,
        "unsafe_early_stop",
        None,
    )
    recal = _cfg_get(block, "recalibration", None)
    adaptive_tempering = _cfg_get(recal, "adaptive_tempering", None)
    ess_aware_checkpoint_selection = _cfg_get(
        recal, "ess_aware_checkpoint_selection", None
    )
    if isinstance(ess_aware_checkpoint_selection, Mapping):
        ess_aware_enabled = bool(
            _cfg_get(ess_aware_checkpoint_selection, "enabled", False)
        )
        ess_aware_max_checkpoints = int(
            _cfg_get(ess_aware_checkpoint_selection, "max_checkpoints", 16)
        )
        ess_aware_first_residual_only = bool(
            _cfg_get(
                ess_aware_checkpoint_selection, "first_residual_only", False
            )
        )
    else:
        ess_aware_enabled = bool(ess_aware_checkpoint_selection or False)
        ess_aware_max_checkpoints = 16
        ess_aware_first_residual_only = False
    if type(_cfg_get(recal, "iteration_one_only", False)) is not bool:
        raise ValueError("iteration_one_only must be a boolean")
    if type(_cfg_get(adaptive_tempering, "enabled", False)) is not bool:
        raise ValueError("adaptive_tempering.enabled must be a boolean")
    if type(_cfg_get(adaptive_tempering, "inherit_previous", False)) is not bool:
        raise ValueError(
            "adaptive_tempering.inherit_previous must be a boolean"
        )
    if type(ess_aware_enabled) is not bool:
        raise ValueError(
            "ess_aware_checkpoint_selection.enabled must be a boolean"
        )
    if type(ess_aware_first_residual_only) is not bool:
        raise ValueError(
            "ess_aware_checkpoint_selection.first_residual_only must be a boolean"
        )
    if type(_cfg_get(block, "cache_event_inputs", False)) is not bool:
        raise ValueError("cache_event_inputs must be a boolean")
    if type(_cfg_get(recal, "visible_pair_rest_frame", False)) is not bool:
        raise ValueError("visible_pair_rest_frame must be a boolean")
    reference_trust = _cfg_get(dgpo_config, "reference_trust", None)
    trust_boundary = _cfg_get(reference_trust, "adaptive_boundary", None)
    trust_radius_calibration = _cfg_get(
        trust_boundary, "radius_calibration", None
    )
    signed_direction_probe = _cfg_get(
        trust_radius_calibration, "signed_direction_probe", None
    )
    lookahead_extragradient = _cfg_get(
        trust_radius_calibration, "lookahead_extragradient", None
    )
    # Resetting optimizer momentum when a new reward/reference pair is installed
    # is an objective-change concern, not a hard-boundary concern.  Keep the
    # nested legacy key working for old hard-trust configs, while allowing the
    # recalibration block to opt in when no boundary is active.
    reset_first_moment_on_install = _cfg_get(
        recal,
        "reset_adam_first_moment_on_install",
        _cfg_get(
            reference_trust,
            "reset_adam_first_moment_on_install",
            None,
        ),
    )
    if reset_first_moment_on_install is None:
        reset_first_moment_on_install = (
            _cfg_get(
                trust_boundary,
                "reset_adam_first_moment_on_install",
                True,
            )
            if bool(_cfg_get(trust_boundary, "enabled", False))
            else False
        )
    config = AdaptiveOmniFoldConfig(
        enabled=bool(_cfg_get(block, "enabled", False)),
        log_only=bool(_cfg_get(block, "log_only", False)),
        monitor_mode=str(
            _cfg_get(block, "monitor_mode", "weighted_and_raw")
        ).strip().lower(),
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
        refit_once_fail_closed=bool(
            _cfg_get(recal, "refit_once_fail_closed", False)
        ),
        refit_once_id=str(_cfg_get(recal, "refit_once_id", "")).strip(),
        staleness_every_n_epochs=max(
            1, int(_cfg_get(block, "staleness_every_n_epochs", 1))
        ),
        staleness_every_n_steps=_optional_int(_cfg_get(block, "staleness_every_n_steps", None)),
        single_pool_train_validation=bool(_cfg_get(block, "single_pool_train_validation", False)),
        single_pool_split_seed=int(_cfg_get(block, "single_pool_split_seed", 42)),
        cache_event_inputs=_cfg_get(block, "cache_event_inputs", False),
        fixed_audit_panel=bool(
            _cfg_get(block, "fixed_audit_panel", False)
        ),
        pool_generation_batch_size=_optional_int(
            _cfg_get(block, "pool_generation_batch_size", None)
        ),
        retrain_auc_margin=float(
            _cfg_get(trigger, "retrain_auc_margin", 0.01)
        ),
        raw_audit_enabled=bool(
            _cfg_get(trigger, "raw_audit_enabled", False)
        ),
        raw_monitor_warm_start=bool(_cfg_get(trigger, "warm_start_classifier", False)),
        raw_improvement_min_delta=float(
            _cfg_get(trigger, "raw_improvement_min_delta", 0.0)
        ),
        raw_rollback_to_best_on_plateau=bool(
            _cfg_get(trigger, "rollback_to_best_on_plateau", False)
        ),
        raw_best_scope=str(_cfg_get(trigger, "best_scope", "round")),
        raw_global_confirm_candidates=bool(_cfg_get(trigger, "global_confirm_candidates", False)),
        raw_global_max_failed_rounds=int(_cfg_get(trigger, "global_max_failed_rounds", 2)),
        raw_global_confirmation_fit=dict(_cfg_get(trigger, "global_confirmation_fit", {}) or {}),
        raw_global_confirmation_warm_start=bool(_cfg_get(trigger, "global_confirmation_warm_start", False)),
        raw_pause_patience_during_warmup=bool(_cfg_get(trigger, "pause_patience_during_warmup", False)),
        raw_patience_schedule=_raw_patience_schedule(_cfg_get(trigger, "patience_schedule", None)),
        max_reward_age_epochs=_optional_int(
            _cfg_get(trigger, "max_reward_age_epochs", None)
        ),
        fixed_schedule_skip_staleness_audit=bool(
            _cfg_get(trigger, "fixed_schedule_skip_staleness_audit", False)
        ),
        fixed_schedule_log_raw_audit=bool(
            _cfg_get(trigger, "fixed_schedule_log_raw_audit", False)
        ),
        required_consecutive_epochs=max(
            1, int(_cfg_get(trigger, "required_consecutive_checks",
                            _cfg_get(trigger, "required_consecutive_epochs", 1)))
        ),
        retrain_cooldown_epochs=max(
            0, int(_cfg_get(trigger, "retrain_cooldown_epochs", 0))
        ),
        classifier_trust_enabled=bool(
            _cfg_get(classifier_trust, "enabled", False)
        ),
        classifier_trust_every_n_epochs=max(
            1,
            int(
                _cfg_get(
                    classifier_trust,
                    "every_n_epochs",
                    _cfg_get(block, "staleness_every_n_epochs", 1),
                )
            ),
        ),
        classifier_trust_probe_max_events=_optional_int(
            _cfg_get(
                classifier_trust,
                "probe_max_events",
                _cfg_get(trigger, "probe_max_events", None),
            )
        ),
        classifier_trust_max_balanced_accuracy=float(
            _cfg_get(classifier_trust, "max_balanced_accuracy", 0.525)
        ),
        classifier_trust_confidence_z=float(
            _cfg_get(classifier_trust, "confidence_z", 1.96)
        ),
        classifier_trust_required_consecutive_epochs=max(
            1,
            int(
                _cfg_get(
                    classifier_trust,
                    "required_consecutive_epochs",
                    1,
                )
            ),
        ),
        classifier_trust_require_saturation=bool(
            _cfg_get(classifier_trust, "require_saturation", True)
        ),
        classifier_trust_audit_fit=dict(
            _cfg_get(classifier_trust, "audit_fit", {}) or {}
        ),
        classifier_trust_unsafe_early_stop_enabled=bool(
            _cfg_get(
                classifier_trust_unsafe_early_stop,
                "enabled",
                False,
            )
        ),
        classifier_trust_unsafe_early_stop_confidence_z=float(
            _cfg_get(
                classifier_trust_unsafe_early_stop,
                "confidence_z",
                _cfg_get(classifier_trust, "confidence_z", 1.96),
            )
        ),
        classifier_trust_unsafe_early_stop_required_consecutive_validations=max(
            1,
            int(
                _cfg_get(
                    classifier_trust_unsafe_early_stop,
                    "required_consecutive_validations",
                    1,
                )
            ),
        ),
        trust_boundary_enabled=bool(
            _cfg_get(trust_boundary, "enabled", False)
        ),
        trust_radius_mode=str(
            _cfg_get(trust_boundary, "radius_mode", "auc_scaled")
        ).strip().lower(),
        trust_distance=str(
            _cfg_get(trust_boundary, "distance", "velocity_mse_ratio")
        ).strip().lower(),
        trust_delta_max=float(
            _cfg_get(trust_boundary, "delta_max", 1.0e-3)
        ),
        trust_delta_floor=float(
            _cfg_get(trust_boundary, "delta_floor", 5.0e-6)
        ),
        trust_initial_raw_auc_gap=(
            None
            if _cfg_get(trust_boundary, "initial_raw_auc_gap", None)
            in (None, "", "null")
            else float(_cfg_get(trust_boundary, "initial_raw_auc_gap"))
        ),
        trust_warning_fraction=float(
            _cfg_get(trust_boundary, "warning_fraction", 0.8)
        ),
        trust_adaptive_power=float(
            _cfg_get(trust_boundary, "adaptive_power", 2.0)
        ),
        trust_auc_confidence_z=float(
            _cfg_get(trust_boundary, "auc_confidence_z", 1.96)
        ),
        trust_auc_stop_gap=float(
            _cfg_get(trust_boundary, "auc_stop_gap", 0.01)
        ),
        trust_reset_adam_first_moment=bool(
            reset_first_moment_on_install
        ),
        trust_reset_adam_first_moment_on_zero_step=bool(
            _cfg_get(
                trust_boundary,
                "reset_adam_first_moment_on_zero_step",
                False,
            )
        ),
        trust_transactional_rejection=bool(
            _cfg_get(trust_boundary, "transactional_rejection", False)
        ),
        trust_stop_after_rejection=bool(
            _cfg_get(trust_boundary, "stop_after_rejection", False)
        ),
        trust_enforcement=str(
            _cfg_get(trust_boundary, "enforcement", "post_step_backtracking")
        ).strip(),
        trust_backtrack_factor=float(
            _cfg_get(trust_boundary, "backtrack_factor", 0.5)
        ),
        trust_max_backtracks=int(
            _cfg_get(trust_boundary, "max_backtracks", 16)
        ),
        trust_probe_events_per_rank=int(
            _cfg_get(trust_boundary, "probe_events_per_rank", 64)
        ),
        trust_fixed_probe_per_reward_round=bool(
            _cfg_get(trust_boundary, "fixed_probe_per_reward_round", False)
        ),
        trust_interior_fraction=float(
            _cfg_get(trust_boundary, "interior_fraction", 1.0)
        ),
        trust_empirical_radius_enabled=bool(
            _cfg_get(trust_radius_calibration, "enabled", False)
        ),
        trust_empirical_bidirectional=bool(
            _cfg_get(trust_radius_calibration, "bidirectional", False)
        ),
        trust_empirical_allow_expansion=bool(
            _cfg_get(trust_radius_calibration, "allow_expansion", True)
        ),
        trust_cross_round_nonexpanding=bool(
            _cfg_get(
                trust_radius_calibration,
                "cross_round_nonexpanding",
                False,
            )
        ),
        trust_policy_lr_scaling_enabled=bool(
            _cfg_get(trust_radius_calibration, "scale_policy_lr", False)
        ),
        trust_policy_lr_scale_floor=float(
            _cfg_get(trust_radius_calibration, "policy_lr_scale_floor", 0.1)
        ),
        trust_refit_on_exhaustion=bool(
            _cfg_get(trust_radius_calibration, "refit_on_exhaustion", False)
        ),
        trust_exhaustion_distance_fraction=float(
            _cfg_get(
                trust_radius_calibration,
                "exhaustion_distance_fraction",
                0.98,
            )
        ),
        trust_exhaustion_mean_update_scale=float(
            _cfg_get(
                trust_radius_calibration,
                "exhaustion_mean_update_scale",
                0.01,
            )
        ),
        trust_exhaustion_scale_window_steps=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "exhaustion_scale_window_steps",
                    _cfg_get(
                        trust_radius_calibration,
                        "attempt_window_steps",
                        50,
                    ),
                )
            ),
        ),
        trust_round_acceptance_enabled=bool(
            _cfg_get(
                trust_radius_calibration,
                "round_acceptance_enabled",
                False,
            )
        ),
        trust_round_acceptance_confidence_z=float(
            _cfg_get(
                trust_radius_calibration,
                "round_acceptance_confidence_z",
                1.96,
            )
        ),
        trust_round_plateau_patience=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "round_plateau_patience",
                    2,
                )
            ),
        ),
        trust_trajectory_search_enabled=bool(
            _cfg_get(
                trust_radius_calibration,
                "trajectory_search_enabled",
                False,
            )
        ),
        trust_failed_direction_patience=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "failed_direction_patience",
                    3,
                )
            ),
        ),
        trust_signed_direction_probe_enabled=bool(
            _cfg_get(signed_direction_probe, "enabled", False)
        ),
        trust_signed_direction_probe_scales=_float_tuple(
            _cfg_get(signed_direction_probe, "scales", (0.25, 0.5, 1.0)),
            name="trust radius_calibration.signed_direction_probe.scales",
        ),
        trust_signed_direction_recovery_enabled=bool(
            _cfg_get(signed_direction_probe, "recover_reverse_only", False)
        ),
        trust_extragradient_enabled=bool(
            _cfg_get(lookahead_extragradient, "enabled", False)
        ),
        trust_extragradient_lookahead_scale=float(
            _cfg_get(lookahead_extragradient, "scale", 0.5)
        ),
        trust_empirical_safety_factor=float(
            _cfg_get(trust_radius_calibration, "safety_factor", 0.5)
        ),
        trust_empirical_expand_factor=float(
            _cfg_get(trust_radius_calibration, "expand_factor", 1.25)
        ),
        trust_empirical_shrink_factor=float(
            _cfg_get(trust_radius_calibration, "shrink_factor", 0.5)
        ),
        trust_empirical_target_acceptance_rate=float(
            _cfg_get(
                trust_radius_calibration,
                "target_acceptance_rate",
                0.5,
            )
        ),
        trust_empirical_target_update_scale=float(
            _cfg_get(
                trust_radius_calibration,
                "target_update_scale",
                0.05,
            )
        ),
        trust_empirical_safe_audits_required=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "safe_audits_required",
                    2,
                )
            ),
        ),
        trust_empirical_attempt_window_steps=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "attempt_window_steps",
                    50,
                )
            ),
        ),
        trust_empirical_distance_window_steps=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "distance_window_steps",
                    10,
                )
            ),
        ),
        trust_empirical_min_distance_samples=max(
            1,
            int(
                _cfg_get(
                    trust_radius_calibration,
                    "min_distance_samples",
                    5,
                )
            ),
        ),
        trust_empirical_confidence_z=float(
            _cfg_get(trust_radius_calibration, "confidence_z", 1.96)
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
        pool_data_parquet_dir=_optional_str(
            _cfg_get(recal, "train_parquet_dir", None)
        ),
        pool_events=_optional_int(_cfg_get(recal, "pool_events", None)),
        pool_selection_seed=int(_cfg_get(recal, "pool_selection_seed", 42)),
        candidates_per_event=int(_cfg_get(recal, "candidates_per_event", 1)),
        min_iterations=int(_cfg_get(recal, "min_iterations", 2)),
        max_iterations=int(_cfg_get(recal, "max_iterations", 12)),
        max_reward_rounds=_optional_int(
            _cfg_get(recal, "max_reward_rounds", None)
        ),
        scheduled_refit_fail_closed=bool(
            _cfg_get(recal, "scheduled_refit_fail_closed", False)
        ),
        iteration_one_only=_cfg_get(recal, "iteration_one_only", False),
        fixed_iteration_budget=_cfg_get(recal, "fixed_iteration_budget", False),
        acceptance_audit_enabled=bool(
            _cfg_get(recal, "acceptance_audit_enabled", True)
        ),
        acceptance_max_balanced_accuracy=float(
            _cfg_get(recal, "acceptance_max_balanced_accuracy", 0.51)
        ),
        periodic_pair_features_enabled=bool(
            _cfg_get(recal, "periodic_pair_features", False)
        ),
        visible_pair_rest_frame_enabled=bool(
            _cfg_get(recal, "visible_pair_rest_frame", False)
        ),
        topology_fourier_embedding_enabled=bool(
            _cfg_get(recal, "topology_fourier_embedding", False)
        ),
        topology_max_harmonic=int(
            _cfg_get(recal, "topology_max_harmonic", 1)
        ),
        topology_include_theta_pair=bool(
            _cfg_get(recal, "topology_include_theta_pair", False)
        ),
        topology_theta_fourier=bool(
            _cfg_get(recal, "topology_theta_fourier", False)
        ),
        topology_acceptance_audit_enabled=bool(
            _cfg_get(recal, "topology_acceptance_audit_enabled", False)
        ),
        topology_acceptance_max_auc_gap=float(
            _cfg_get(recal, "topology_acceptance_max_auc_gap", 0.005)
        ),
        topology_acceptance_repeats=max(
            1,
            int(_cfg_get(recal, "topology_acceptance_repeats", 1)),
        ),
        tempering=float(_cfg_get(recal, "tempering", 1.0)),
        adaptive_tempering_enabled=bool(
            _cfg_get(adaptive_tempering, "enabled", False)
        ),
        target_ess_fraction=float(
            _cfg_get(adaptive_tempering, "target_ess_fraction", 0.2)
        ),
        minimum_tempering=float(
            _cfg_get(adaptive_tempering, "minimum", 0.1)
        ),
        tempering_grid_steps=int(
            _cfg_get(adaptive_tempering, "grid_steps", 14)
        ),
        inherit_previous_tempering=bool(
            _cfg_get(adaptive_tempering, "inherit_previous", False)
        ),
        ess_aware_checkpoint_selection=ess_aware_enabled,
        ess_aware_max_checkpoints=ess_aware_max_checkpoints,
        ess_aware_first_residual_only=ess_aware_first_residual_only,
        log_ratio_clip=_optional_positive_float(
            _cfg_get(recal, "log_ratio_clip", None),
            name="recalibration.log_ratio_clip",
        ),
        minimum_ess_fraction=float(
            _cfg_get(recal, "minimum_ess_fraction", 0.0)
        ),
        crossfit_folds=int(_cfg_get(recal, "crossfit_folds", 2)),
        crossfit_repeats=int(_cfg_get(recal, "crossfit_repeats", 1)),
        residual_min_auc_gain=float(
            _cfg_get(recal, "residual_min_auc_gain", 1.0e-3)
        ),
        residual_closure_schedule=_residual_closure_schedule(
            _cfg_get(recal, "residual_closure_schedule", None)
        ),
        seed=int(_cfg_get(recal, "seed", 20260819)),
        score_row_budget=max(1, int(_cfg_get(recal, "score_row_budget", 512))),
        audit_fit=dict(
            _cfg_get(block, "audit_fit", _cfg_get(recal, "fit", {})) or {}
        ),
        reward_classifier=_classifier_architecture_overrides(
            _cfg_get(recal, "reward_classifier", None)
        ),
        fit=dict(_cfg_get(recal, "fit", {}) or {}),
        warm_start_iterations=tuple(
            _cfg_get(recal, "warm_start_iterations", ()) or ()
        ),
        warm_start_from_iteration_one=_cfg_get(recal, "warm_start_from_iteration_one", False),
        crossfit_partition=str(_cfg_get(recal, "crossfit_partition", "auto")),
        reset_optimizer_state_on_install=bool(
            _cfg_get(recal, "reset_optimizer_state_on_install", False)
        ),
        policy_warmup_steps=_cfg_get(recal, "policy_warmup_steps", 0),
        policy_warmup_start_factor=float(_cfg_get(recal, "policy_warmup_start_factor", 0.1)),
        trust_round_decay_factor=float(
            _cfg_get(trust_boundary, "round_decay_factor", 0.9)
        ),
        trust_best_decay_factor=float(
            _cfg_get(trust_boundary, "best_decay_factor", 0.9)
        ),
    )
    if (isinstance(config.policy_warmup_steps, bool)
            or not isinstance(config.policy_warmup_steps, int) or config.policy_warmup_steps < 0):
        raise ValueError("policy_warmup_steps must be a nonnegative integer")
    minimum_sufficient_target = config.fit.get(
        "minimum_sufficient_balanced_accuracy"
    )
    if minimum_sufficient_target is not None:
        if (
            isinstance(minimum_sufficient_target, bool)
            or not isinstance(minimum_sufficient_target, (int, float))
            or not math.isfinite(float(minimum_sufficient_target))
            or not 0.5 < float(minimum_sufficient_target) < 1.0
        ):
            raise ValueError(
                "fit.minimum_sufficient_balanced_accuracy must lie in (0.5, 1)"
            )
        if bool(config.fit.get("require_saturation", True)):
            raise ValueError(
                "fit.minimum_sufficient_balanced_accuracy requires "
                "fit.require_saturation=false"
            )
        confidence_z = config.fit.get(
            "minimum_sufficient_confidence_z", 0.0
        )
        if (
            isinstance(confidence_z, bool)
            or not isinstance(confidence_z, (int, float))
            or not math.isfinite(float(confidence_z))
            or float(confidence_z) < 0.0
        ):
            raise ValueError(
                "fit.minimum_sufficient_confidence_z must be finite and nonnegative"
            )
        required_consecutive = config.fit.get(
            "minimum_sufficient_required_consecutive", 1
        )
        if type(required_consecutive) is not int or required_consecutive < 1:
            raise ValueError(
                "fit.minimum_sufficient_required_consecutive must be a positive integer"
            )
    if not 0.0 < config.policy_warmup_start_factor <= 1.0:
        raise ValueError("policy_warmup_start_factor must be finite and in (0, 1]")
    if (
        config.log_ratio_clip is not None
        and (
            not math.isfinite(config.log_ratio_clip)
            or config.log_ratio_clip <= 0.0
        )
    ):
        raise ValueError("recalibration.log_ratio_clip must be finite and positive")
    if config.adaptive_tempering_enabled:
        if (
            not math.isfinite(config.minimum_tempering)
            or not 0.0 < config.minimum_tempering <= config.tempering
        ):
            raise ValueError(
                "adaptive_tempering.minimum must lie in (0, tempering]"
            )
        if (
            not math.isfinite(config.target_ess_fraction)
            or not config.minimum_ess_fraction
            <= config.target_ess_fraction
            <= 1.0
        ):
            raise ValueError(
                "adaptive_tempering.target_ess_fraction must lie in "
                "[minimum_ess_fraction, 1]"
            )
        if config.tempering_grid_steps < 2:
            raise ValueError(
                "adaptive_tempering.grid_steps must be at least two"
            )
    elif config.inherit_previous_tempering:
        raise ValueError(
            "adaptive_tempering.inherit_previous requires adaptive_tempering.enabled"
        )
    if config.ess_aware_max_checkpoints < 1:
        raise ValueError(
            "ess_aware_checkpoint_selection.max_checkpoints must be positive"
        )
    if (
        config.ess_aware_first_residual_only
        and not config.ess_aware_checkpoint_selection
    ):
        raise ValueError(
            "ess_aware_checkpoint_selection.first_residual_only requires "
            "ess_aware_checkpoint_selection.enabled"
        )
    if (
        not math.isfinite(config.minimum_ess_fraction)
        or not 0.0 <= config.minimum_ess_fraction <= 1.0
    ):
        raise ValueError(
            "recalibration.minimum_ess_fraction must lie in [0, 1]"
        )
    if config.policy_warmup_steps and config.trust_extragradient_enabled:
        raise ValueError("policy round warmup is not supported with extragradient")
    if config.candidates_per_event != 1:
        raise ValueError(
            "adaptive OmniFold probe/refit requires candidates_per_event=1; "
            "DGPO's online group size remains dgpo.K"
        )
    if config.refit_once_fail_closed and not config.refit_once_on_resume:
        raise ValueError(
            "refit_once_fail_closed requires refit_once_on_resume"
        )
    if config.refit_once_fail_closed and not config.refit_once_id:
        raise ValueError(
            "refit_once_fail_closed requires a versioned refit_once_id"
        )
    if config.monitor_mode not in {
        "weighted_and_raw",
        "raw_only",
        "raw_plateau_refit",
    }:
        raise ValueError(
            "adaptive_omnifold.monitor_mode must be 'weighted_and_raw', "
            "'raw_only', or 'raw_plateau_refit'"
        )
    if (
        config.monitor_mode in {"raw_only", "raw_plateau_refit"}
        and not config.raw_audit_enabled
    ):
        raise ValueError(
            "raw-only adaptive OmniFold monitoring requires "
            "trigger.raw_audit_enabled=true"
        )
    if config.monitor_mode == "raw_only" and not config.log_only:
        raise ValueError(
            "adaptive_omnifold.monitor_mode=raw_only requires log_only=true; "
            "the weighted staleness controller is intentionally unavailable"
        )
    if config.monitor_mode == "raw_plateau_refit" and config.log_only:
        raise ValueError(
            "adaptive_omnifold.monitor_mode=raw_plateau_refit requires "
            "log_only=false"
        )
    if config.raw_monitor_warm_start and not (
        config.fixed_audit_panel
        and config.raw_audit_enabled
        and config.require_audit_saturation
    ):
        raise ValueError(
            "warm-start raw monitor requires a fixed audit panel, "
            "raw_audit_enabled, and saturation"
        )
    monitor_readiness = validate_monitor_training_readiness(config.audit_fit.get("training_readiness"))
    if monitor_readiness is not None and not (
        config.raw_monitor_warm_start
        and config.fixed_audit_panel
        and config.require_audit_saturation
    ):
        raise ValueError(
            "monitor training_readiness requires a fixed-panel, warm-start "
            "raw monitor with saturation"
        )
    if config.staleness_every_n_steps is not None:
        if config.staleness_every_n_steps < 1:
            raise ValueError("staleness_every_n_steps must be positive or null")
        if config.monitor_mode != "raw_plateau_refit" or config.trust_extragradient_enabled:
            raise ValueError("step-based staleness requires raw_plateau_refit without extragradient")
    if config.raw_patience_schedule and (
        config.monitor_mode != "raw_plateau_refit" or config.staleness_every_n_steps is None
    ):
        raise ValueError("trigger.patience_schedule requires step-based raw_plateau_refit monitoring")
    if config.single_pool_train_validation:
        if config.monitor_mode != "raw_plateau_refit" or config.acceptance_audit_enabled or config.topology_acceptance_audit_enabled:
            raise ValueError("single-pool mode requires raw_plateau_refit without acceptance/topology audits")
        if any(value is not None for value in (
            config.pool_data_parquet_dir, config.pool_events,
            config.classifier_trust_probe_max_events, config.refit_score_events,
        )):
            raise ValueError("single-pool mode uses every budgeted event; separate source and pool caps must be null")
        # Raw staleness may use a bounded prefix of the SAME training source.
        # This does not cap OmniFold fitting/closure or change identity-hash splits.
        # probe_max_events is separately validated below (at least 30 events).
    if (
        not math.isfinite(config.raw_improvement_min_delta)
        or config.raw_improvement_min_delta < 0.0
        or config.raw_improvement_min_delta >= 0.5
    ):
        raise ValueError("raw_improvement_min_delta must lie in [0, 0.5)")
    if config.raw_rollback_to_best_on_plateau and not (
        config.monitor_mode == "raw_plateau_refit"
        and config.raw_audit_enabled
        and config.fixed_audit_panel
        and config.require_audit_saturation
    ):
        raise ValueError(
            "trigger.rollback_to_best_on_plateau requires "
            "monitor_mode=raw_plateau_refit, raw_audit_enabled=true, "
            "fixed_audit_panel=true, and require_audit_saturation=true"
        )
    if config.raw_best_scope not in {"round", "global"}:
        raise ValueError("trigger.best_scope must be round or global")
    if config.raw_best_scope == "global" and (
        config.monitor_mode != "raw_plateau_refit"
        or not config.raw_audit_enabled or not config.fixed_audit_panel
        or not config.require_audit_saturation
        or config.classifier_trust_enabled or config.max_reward_age_epochs is not None
        or config.staleness_every_n_steps is None or config.raw_global_max_failed_rounds < 0
    ):
        raise ValueError("global raw best requires saturated fixed-panel step monitoring, nonnegative failed-round limit (0 disables stop), and no competing age/classifier triggers")
    if config.raw_global_confirm_candidates and config.raw_best_scope != "global":
        raise ValueError("global candidate confirmation requires best_scope=global")
    allowed_confirmation_overrides = {"min_steps", "min_epochs", "enforce_min_epochs"}
    if set(config.raw_global_confirmation_fit) - allowed_confirmation_overrides:
        raise ValueError("global_confirmation_fit may only override minimum training duration")
    for key in ("min_steps", "min_epochs"):
        value = config.raw_global_confirmation_fit.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"global_confirmation_fit.{key} must be finite and nonnegative")
        if key == "min_steps" and int(value) != value:
            raise ValueError("global_confirmation_fit.min_steps must be an integer")
    if not 0.5 < config.classifier_trust_max_balanced_accuracy < 1.0:
        raise ValueError(
            "classifier_trust.max_balanced_accuracy must lie in (0.5, 1)"
        )
    if (
        not math.isfinite(config.classifier_trust_confidence_z)
        or config.classifier_trust_confidence_z < 0.0
    ):
        raise ValueError("classifier_trust.confidence_z must be nonnegative")
    if (
        not math.isfinite(
            config.classifier_trust_unsafe_early_stop_confidence_z
        )
        or config.classifier_trust_unsafe_early_stop_confidence_z < 0.0
    ):
        raise ValueError(
            "classifier_trust.unsafe_early_stop.confidence_z must be "
            "nonnegative"
        )
    if (
        config.classifier_trust_unsafe_early_stop_enabled
        and not config.classifier_trust_enabled
    ):
        raise ValueError(
            "classifier_trust.unsafe_early_stop requires classifier_trust.enabled"
        )
    if config.classifier_trust_enabled and config.monitor_mode != "raw_plateau_refit":
        raise ValueError(
            "classifier_trust requires monitor_mode=raw_plateau_refit"
        )
    if (
        config.classifier_trust_enabled
        and config.classifier_trust_every_n_epochs
        % config.staleness_every_n_epochs
        != 0
    ):
        raise ValueError(
            "classifier_trust.every_n_epochs must be a multiple of "
            "adaptive_omnifold.staleness_every_n_epochs"
        )
    if (
        config.pool_generation_batch_size is not None
        and config.pool_generation_batch_size < 1
    ):
        raise ValueError("pool_generation_batch_size must be positive")
    if type(config.fixed_iteration_budget) is not bool:
        raise ValueError("fixed_iteration_budget must be a boolean")
    if config.fixed_iteration_budget:
        if config.iteration_one_only or config.min_iterations != config.max_iterations:
            raise ValueError("fixed_iteration_budget requires equal min/max iterations and iteration_one_only=false")
        if config.acceptance_audit_enabled or config.residual_closure_schedule:
            raise ValueError("fixed_iteration_budget does not use closure acceptance audits/schedules")
    if config.iteration_one_only:
        if config.monitor_mode != "raw_plateau_refit" and not (
            config.monitor_mode == "raw_only" and config.log_only
        ):
            raise ValueError("iteration_one_only requires raw_plateau_refit or log-only raw_only monitoring")
        if config.min_iterations != 1 or config.max_iterations != 1:
            raise ValueError("iteration_one_only requires min_iterations=max_iterations=1")
        if config.acceptance_audit_enabled or config.topology_acceptance_audit_enabled:
            raise ValueError("iteration_one_only does not use additional closure/acceptance classifiers")
        if config.residual_closure_schedule:
            raise ValueError("iteration_one_only must not claim a residual_closure_schedule")
    if config.min_iterations < 1 or config.max_iterations < config.min_iterations:
        raise ValueError("adaptive OmniFold iterations require 1 <= min <= max")
    if config.max_reward_rounds is not None and config.max_reward_rounds < 1:
        raise ValueError("recalibration.max_reward_rounds must be positive or null")
    if config.scheduled_refit_fail_closed and config.max_reward_age_epochs is None:
        raise ValueError(
            "scheduled_refit_fail_closed requires trigger.max_reward_age_epochs"
        )
    if config.fixed_schedule_skip_staleness_audit and (
        config.monitor_mode != "raw_plateau_refit"
        or config.log_only
        or config.max_reward_age_epochs is None
        or config.staleness_every_n_steps is not None
        or config.classifier_trust_enabled
        or config.raw_rollback_to_best_on_plateau
        or config.trust_boundary_enabled
        or config.trust_trajectory_search_enabled
        or config.trust_signed_direction_probe_enabled
        or config.trust_signed_direction_recovery_enabled
        or config.trust_extragradient_enabled
    ):
        raise ValueError(
            "trigger.fixed_schedule_skip_staleness_audit requires an epoch-based "
            "raw_plateau_refit schedule with max_reward_age_epochs, log_only=false, "
            "and no classifier/trust/rollback controller"
        )
    if config.fixed_schedule_log_raw_audit and (
        not config.fixed_schedule_skip_staleness_audit
        or not config.raw_audit_enabled
        or not config.fixed_audit_panel
        or config.raw_monitor_warm_start
    ):
        raise ValueError(
            "trigger.fixed_schedule_log_raw_audit requires the fixed schedule, "
            "raw_audit_enabled=true, fixed_audit_panel=true, and a fresh "
            "classifier at every boundary"
        )
    audit_repeats = config.audit_fit.get("repeats", 1)
    audit_population = config.audit_fit.get("training_population", "probe_split")
    if audit_population not in ("probe_split", "omnifold_fold"):
        raise ValueError("audit_fit.training_population must be probe_split or omnifold_fold")
    if audit_population == "omnifold_fold":
        if ((not config.fixed_schedule_log_raw_audit and not classifier_only)
                or config.raw_monitor_warm_start or not config.fixed_audit_panel
                or config.single_pool_train_validation
                or config.crossfit_partition != "identity" or config.pool_events is not None
                or not config.cache_event_inputs
                or not config.audit_fit.get("disjoint_final_audit", False)
                or config.audit_fit.get("training_readiness") is not None):
            raise ValueError(
                "omnifold_fold audit requires cold fixed-schedule or classifier-only diagnostics, cached full "
                "identity-crossfit training data, separate validation and disjoint final audit"
            )
        audit_fold = config.audit_fit.get("training_fold", 1)
        if type(audit_fold) is not int or not 1 <= audit_fold <= config.crossfit_folds:
            raise ValueError("audit_fit.training_fold must identify an OmniFold fold (one-based)")
    audit_repeat_seed_stride = config.audit_fit.get(
        "repeat_seed_stride", 104_729
    )
    if type(audit_repeats) is not int or audit_repeats < 1:
        raise ValueError("audit_fit.repeats must be a positive integer")
    if (
        type(audit_repeat_seed_stride) is not int
        or audit_repeat_seed_stride < 1
    ):
        raise ValueError(
            "audit_fit.repeat_seed_stride must be a positive integer"
        )
    if audit_repeats > 1 and config.raw_monitor_warm_start:
        raise ValueError(
            "repeated raw audits require fresh classifier initialization"
        )
    if config.crossfit_folds < 2:
        raise ValueError("adaptive OmniFold crossfit_folds must be at least two")
    if config.crossfit_repeats < 1:
        raise ValueError(
            "adaptive OmniFold crossfit_repeats must be at least one"
        )
    if config.crossfit_partition not in ("auto", "identity"):
        raise ValueError("crossfit_partition must be 'auto' or 'identity'")
    fold_minimum = config.fit.get("min_steps_per_fold", 0)
    if type(fold_minimum) is not int or fold_minimum < 0:
        raise ValueError("min_steps_per_fold must be a nonnegative integer")
    warm_epochs = config.fit.get("warm_start_min_epochs_per_fold")
    if warm_epochs is not None and (
        isinstance(warm_epochs, bool) or not isinstance(warm_epochs, (int, float))
        or not math.isfinite(warm_epochs) or warm_epochs <= 0
    ):
        raise ValueError("warm_start_min_epochs_per_fold must be a finite positive number")
    if warm_epochs is not None and config.fit.get("sampling", "independent_epoch_shuffle") != "independent_epoch_shuffle":
        raise ValueError("warm_start_min_epochs_per_fold requires independent_epoch_shuffle")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        or not 1 <= value <= config.max_iterations
        for value in config.warm_start_iterations
    ) or len(set(config.warm_start_iterations)) != len(config.warm_start_iterations):
        raise ValueError("warm_start_iterations must contain unique iteration ids in [1, max_iterations]")
    if type(config.warm_start_from_iteration_one) is not bool:
        raise ValueError("warm_start_from_iteration_one must be a boolean")
    if config.warm_start_from_iteration_one and config.warm_start_iterations != (1,):
        raise ValueError("warm_start_from_iteration_one requires warm_start_iterations: [1]")
    later_lr = config.fit.get("later_iteration_learning_rate")
    later_mode = config.fit.get("later_iteration_train_mode", "full")
    if later_mode not in ("full", "last_decoder_and_output", "output_then_last_decoder"):
        raise ValueError("unsupported later_iteration_train_mode")
    if later_mode != "full" and not config.warm_start_from_iteration_one:
        raise ValueError("later_iteration_train_mode requires warm_start_from_iteration_one")
    if later_lr is not None:
        if (isinstance(later_lr, bool) or not isinstance(later_lr, (int, float))
            or not math.isfinite(later_lr) or later_lr <= 0):
            raise ValueError("later_iteration_learning_rate must be finite and positive")
        if not config.warm_start_from_iteration_one:
            raise ValueError("later_iteration_learning_rate requires warm_start_from_iteration_one")
    decoder_lr = config.fit.get("later_iteration_decoder_learning_rate")
    if later_mode == "output_then_last_decoder":
        output_lr = later_lr or config.fit.get("learning_rate", 2e-3)
        if (isinstance(decoder_lr, bool) or not isinstance(decoder_lr, (int, float))
                or not math.isfinite(decoder_lr) or not 0 < decoder_lr <= output_lr):
            raise ValueError("staged fitting requires positive decoder LR <= output LR")
        if not config.fit.get("restore_best", True):
            raise ValueError("staged fitting requires restore_best")
    elif decoder_lr is not None:
        raise ValueError("later_iteration_decoder_learning_rate requires output_then_last_decoder")
    if not 0.0 <= config.residual_min_auc_gain < 0.5:
        raise ValueError("residual_min_auc_gain must lie in [0, 0.5)")
    if not 0.5 < config.acceptance_max_balanced_accuracy < 1.0:
        raise ValueError("acceptance_max_balanced_accuracy must lie in (0.5, 1)")
    if not 0.0 < config.topology_acceptance_max_auc_gap < 0.5:
        raise ValueError("topology_acceptance_max_auc_gap must lie in (0, 0.5)")
    if config.topology_acceptance_repeats < 1:
        raise ValueError("topology_acceptance_repeats must be positive")
    if (
        config.topology_acceptance_audit_enabled
        and not config.periodic_pair_features_enabled
    ):
        raise ValueError(
            "topology acceptance audit requires recalibration.periodic_pair_features"
        )
    if config.topology_max_harmonic < 1:
        raise ValueError("topology_max_harmonic must be at least one")
    if (
        config.topology_fourier_embedding_enabled
        and not config.periodic_pair_features_enabled
    ):
        raise ValueError(
            "topology_fourier_embedding requires periodic_pair_features"
        )
    if (
        config.topology_acceptance_audit_enabled
        and not config.acceptance_audit_enabled
    ):
        raise ValueError(
            "topology acceptance audit requires acceptance_audit_enabled"
        )
    if not 0.0 < config.retrain_auc_margin < 0.5:
        raise ValueError("retrain_auc_margin must lie in (0, 0.5)")
    if (
        config.max_reward_age_epochs is not None
        and config.max_reward_age_epochs < 1
    ):
        raise ValueError("max_reward_age_epochs must be positive or null")
    if config.trust_boundary_enabled:
        if not config.enabled:
            raise ValueError(
                "reference_trust.adaptive_boundary requires adaptive_omnifold.enabled"
            )
        if not bool(_cfg_get(reference_trust, "enabled", False)):
            raise ValueError(
                "reference_trust.adaptive_boundary requires reference_trust.enabled"
            )
        if config.trust_radius_mode not in {"auc_scaled", "fixed", "round_decay", "best_decay"}:
            raise ValueError(
                "adaptive trust radius_mode must be 'auc_scaled', 'fixed', 'round_decay', or 'best_decay'"
            )
        if config.trust_distance not in {"velocity_mse_ratio", "vp_path_kl"}:
            raise ValueError(
                "adaptive trust distance must be 'velocity_mse_ratio' or "
                "'vp_path_kl'"
            )
        if (
            config.trust_distance == "vp_path_kl"
            and str(_cfg_get(reference_trust, "objective", "velocity_mse"))
            .strip()
            .lower()
            != "vp_path_kl"
        ):
            raise ValueError(
                "a vp_path_kl hard boundary requires reference_trust.objective="
                "vp_path_kl"
            )
        if config.trust_radius_mode == "fixed" and (
            config.trust_empirical_radius_enabled
            or config.trust_cross_round_nonexpanding
            or config.trust_policy_lr_scaling_enabled
        ):
            raise ValueError(
                "fixed trust radius_mode is incompatible with empirical radius "
                "calibration, cross-round caps, and radius-scaled policy LR"
            )
        if config.trust_radius_mode in {"round_decay", "best_decay"}:
            factor = (config.trust_round_decay_factor if config.trust_radius_mode == "round_decay"
                      else config.trust_best_decay_factor)
            if not 0.0 < factor < 1.0:
                raise ValueError(f"{config.trust_radius_mode}_factor must be finite and in (0, 1)")
            if config.trust_radius_mode == "best_decay" and (
                config.monitor_mode != "raw_plateau_refit"
                or not bool(_cfg_get(trust_boundary, "fixed_probe_per_reward_round", True))
            ):
                raise ValueError("best_decay requires raw_plateau_refit and a fixed per-round probe")
            if (
                config.trust_empirical_radius_enabled
                or config.trust_cross_round_nonexpanding
                or config.trust_policy_lr_scaling_enabled
                or config.trust_round_acceptance_enabled
                or config.trust_trajectory_search_enabled
                or config.trust_signed_direction_recovery_enabled
                or config.trust_extragradient_enabled
            ):
                raise ValueError(
                    f"{config.trust_radius_mode} cannot be combined with empirical/trajectory radius "
                    "controllers, round acceptance, or radius-scaled policy LR"
                )
        if not math.isfinite(config.trust_delta_max) or config.trust_delta_max <= 0.0:
            raise ValueError("adaptive trust delta_max must be finite and positive")
        if (
            not math.isfinite(config.trust_delta_floor)
            or config.trust_delta_floor <= 0.0
            or config.trust_delta_floor > config.trust_delta_max
        ):
            raise ValueError(
                "adaptive trust delta_floor must be in (0, delta_max]"
            )
        if (
            config.trust_initial_raw_auc_gap is not None
            and (
                not math.isfinite(config.trust_initial_raw_auc_gap)
                or not 0.0 < config.trust_initial_raw_auc_gap <= 0.5
            )
        ):
            raise ValueError(
                "adaptive trust initial_raw_auc_gap must be null or in (0, 0.5]"
            )
        if not 0.0 < config.trust_warning_fraction < 1.0:
            raise ValueError("adaptive trust warning_fraction must lie in (0, 1)")
        if (
            not math.isfinite(config.trust_adaptive_power)
            or config.trust_adaptive_power <= 0.0
        ):
            raise ValueError("adaptive trust adaptive_power must be positive")
        if (
            not math.isfinite(config.trust_auc_confidence_z)
            or config.trust_auc_confidence_z < 0.0
        ):
            raise ValueError("adaptive trust auc_confidence_z must be nonnegative")
        if not 0.0 <= config.trust_auc_stop_gap < 0.5:
            raise ValueError("adaptive trust auc_stop_gap must lie in [0, 0.5)")
        if config.trust_enforcement != "post_step_backtracking":
            raise ValueError(
                "adaptive trust enforcement must be 'post_step_backtracking'"
            )
        if (
            config.trust_stop_after_rejection
            and not config.trust_transactional_rejection
        ):
            raise ValueError(
                "stop_after_rejection requires transactional_rejection=true"
            )
        if (
            config.trust_transactional_rejection
            and config.trust_reset_adam_first_moment_on_zero_step
        ):
            raise ValueError(
                "transactional rejection restores the complete AdamW state; "
                "disable reset_adam_first_moment_on_zero_step"
            )
        if not 0.0 < config.trust_backtrack_factor < 1.0:
            raise ValueError("adaptive trust backtrack_factor must lie in (0, 1)")
        if config.trust_max_backtracks < 1:
            raise ValueError("adaptive trust max_backtracks must be positive")
        if config.trust_probe_events_per_rank < 1:
            raise ValueError(
                "adaptive trust probe_events_per_rank must be positive"
            )
        if not 0.0 < config.trust_interior_fraction <= 1.0:
            raise ValueError(
                "adaptive trust interior_fraction must lie in (0, 1]"
            )
        if not 0.0 < config.trust_policy_lr_scale_floor <= 1.0:
            raise ValueError(
                "trust radius_calibration.policy_lr_scale_floor must lie in (0, 1]"
            )
        if config.trust_empirical_radius_enabled:
            beta = float(_cfg_get(dgpo_config, "beta", 1.0))
            if not math.isfinite(beta) or not math.isclose(
                beta, 1.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    "empirical trust-radius calibration requires dgpo.beta=1.0; "
                    "change the optimizer learning rate or accepted step scale, "
                    "not the OmniFold/DGPO beta"
                )
            if not 0.0 < config.trust_empirical_safety_factor <= 1.0:
                raise ValueError(
                    "trust radius_calibration.safety_factor must lie in (0, 1]"
                )
            if (
                not math.isfinite(config.trust_empirical_expand_factor)
                or config.trust_empirical_expand_factor <= 1.0
            ):
                raise ValueError(
                    "trust radius_calibration.expand_factor must be greater than 1"
                )
            if not 0.0 < config.trust_empirical_shrink_factor < 1.0:
                raise ValueError(
                    "trust radius_calibration.shrink_factor must lie in (0, 1)"
                )
            if not 0.0 <= config.trust_empirical_target_acceptance_rate <= 1.0:
                raise ValueError(
                    "trust radius_calibration.target_acceptance_rate must lie in [0, 1]"
                )
            if not 0.0 <= config.trust_empirical_target_update_scale <= 1.0:
                raise ValueError(
                    "trust radius_calibration.target_update_scale must lie in [0, 1]"
                )
            if not 0.0 < config.trust_exhaustion_distance_fraction <= 1.0:
                raise ValueError(
                    "trust radius_calibration.exhaustion_distance_fraction "
                    "must lie in (0, 1]"
                )
            if not 0.0 <= config.trust_exhaustion_mean_update_scale <= 1.0:
                raise ValueError(
                    "trust radius_calibration.exhaustion_mean_update_scale "
                    "must lie in [0, 1]"
                )
            if (
                config.trust_exhaustion_scale_window_steps
                > config.trust_empirical_attempt_window_steps
            ):
                raise ValueError(
                    "trust radius_calibration.exhaustion_scale_window_steps "
                    "cannot exceed attempt_window_steps"
                )
            if (
                not math.isfinite(config.trust_round_acceptance_confidence_z)
                or config.trust_round_acceptance_confidence_z < 0.0
            ):
                raise ValueError(
                    "trust radius_calibration.round_acceptance_confidence_z "
                    "must be nonnegative"
                )
            if config.trust_round_plateau_patience < 1:
                raise ValueError(
                    "trust radius_calibration.round_plateau_patience must be positive"
                )
            if config.trust_failed_direction_patience < 1:
                raise ValueError(
                    "trust radius_calibration.failed_direction_patience must be positive"
                )
            signed_scales = config.trust_signed_direction_probe_scales
            if not signed_scales:
                raise ValueError(
                    "trust signed-direction probe scales cannot be empty"
                )
            if any(
                not math.isfinite(scale) or not 0.0 < scale <= 1.0
                for scale in signed_scales
            ):
                raise ValueError(
                    "trust signed-direction probe scales must lie in (0, 1]"
                )
            if len(set(signed_scales)) != len(signed_scales):
                raise ValueError(
                    "trust signed-direction probe scales must be unique"
                )
            if not any(
                math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
                for scale in signed_scales
            ):
                raise ValueError(
                    "trust signed-direction probe scales must include 1.0"
                )
            if (
                config.trust_trajectory_search_enabled
                and not config.raw_audit_enabled
            ):
                raise ValueError(
                    "trust trajectory search requires trigger.raw_audit_enabled"
                )
            if (
                config.trust_signed_direction_probe_enabled
                and not config.trust_trajectory_search_enabled
            ):
                raise ValueError(
                    "trust signed-direction probe requires trajectory_search_enabled"
                )
            if (
                config.trust_empirical_min_distance_samples
                > config.trust_empirical_distance_window_steps
            ):
                raise ValueError(
                    "trust radius_calibration.min_distance_samples cannot exceed "
                    "distance_window_steps"
                )
            if (
                not math.isfinite(config.trust_empirical_confidence_z)
                or config.trust_empirical_confidence_z < 0.0
            ):
                raise ValueError(
                    "trust radius_calibration.confidence_z must be nonnegative"
                )
        if (
            config.trust_round_acceptance_enabled
            and not config.trust_empirical_radius_enabled
        ):
            raise ValueError(
                "round_acceptance_enabled requires radius_calibration.enabled"
            )
    if config.trust_signed_direction_probe_enabled and not (
        config.trust_boundary_enabled
        and config.trust_empirical_radius_enabled
        and config.raw_audit_enabled
    ):
        raise ValueError(
            "trust signed-direction probe requires adaptive trust, empirical "
            "radius calibration, and trigger.raw_audit_enabled"
        )
    if (
        config.trust_signed_direction_recovery_enabled
        and not config.trust_signed_direction_probe_enabled
    ):
        raise ValueError(
            "trust signed-direction reverse recovery requires the "
            "signed-direction probe"
        )
    if config.trust_signed_direction_recovery_enabled and config.log_only:
        raise ValueError(
            "trust signed-direction reverse recovery cannot run in log_only mode"
        )
    if config.trust_extragradient_enabled:
        if not (
            config.trust_boundary_enabled
            and config.trust_empirical_radius_enabled
            and config.trust_refit_on_exhaustion
            and config.trust_round_acceptance_enabled
            and config.trust_trajectory_search_enabled
            and config.trust_fixed_probe_per_reward_round
            and config.raw_audit_enabled
        ):
            raise ValueError(
                "look-ahead extragradient requires adaptive trust, empirical "
                "radius calibration, refit_on_exhaustion, round acceptance, "
                "trajectory search, a fixed per-round probe, and raw audit"
            )
        if config.log_only:
            raise ValueError(
                "look-ahead extragradient cannot run in log_only mode"
            )
        reference_trust_coefficient = float(
            _cfg_get(reference_trust, "coefficient", 0.0)
        )
        if not math.isfinite(reference_trust_coefficient) or not math.isclose(
            reference_trust_coefficient,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "look-ahead extragradient ablation requires "
                "dgpo.reference_trust.coefficient=1.0"
            )
        if (
            not math.isfinite(config.trust_extragradient_lookahead_scale)
            or not 0.0 < config.trust_extragradient_lookahead_scale <= 1.0
        ):
            raise ValueError(
                "look-ahead extragradient scale must lie in (0, 1]"
            )
    if not 0.0 < config.power_alpha < 1.0:
        raise ValueError("power_alpha must lie in (0, 1)")
    if not 0.0 < config.power_target < 1.0:
        raise ValueError("power_target must lie in (0, 1)")
    if config.probe_max_events is not None and config.probe_max_events < 30:
        raise ValueError("probe_max_events must be at least 30")
    if (
        config.classifier_trust_probe_max_events is not None
        and config.classifier_trust_probe_max_events < 30
    ):
        raise ValueError("classifier_trust.probe_max_events must be at least 30")
    if (
        config.probe_max_events is not None
        and config.classifier_trust_probe_max_events is not None
        and config.classifier_trust_probe_max_events > config.probe_max_events
    ):
        raise ValueError(
            "classifier_trust.probe_max_events cannot exceed "
            "trigger.probe_max_events"
        )
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


def baseline_probe_auc_gap(
    probe: Mapping[str, Any],
    *,
    cfg: AdaptiveOmniFoldConfig,
) -> float:
    """Select the baseline measured by the active staleness controller."""

    key = (
        "raw_auc_gap"
        if cfg.monitor_mode == "raw_plateau_refit"
        else "weighted_auc_gap"
    )
    if key not in probe:
        raise KeyError(
            f"adaptive baseline probe is missing controller metric {key!r}"
        )
    value = float(probe[key])
    if not math.isfinite(value) or not 0.0 <= value <= 0.5:
        raise ValueError(
            f"adaptive baseline probe {key} must be finite and in [0, 0.5]"
        )
    return value


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
    raw_best_auc_gap: float = float("nan")
    raw_best_epoch: int = -1
    raw_best_global_step: int = -1
    raw_best_checkpoint: str = ""
    raw_best_next_epoch: int = -1
    raw_global_initialized: bool = False
    raw_global_failed_rounds: int = 0
    raw_global_refit_pending: bool = False
    raw_global_stop_requested: bool = False
    raw_previous_auc_gap: float = float("nan")
    raw_no_improvement_streak: int = 0
    raw_plateau_rollbacks: int = 0
    classifier_trust_exceedance_streak: int = 0
    recalibration_count: int = 0
    recalibrations_rejected: int = 0
    resume_refit_once_completed: bool = False
    resume_refit_once_id: str = ""
    installed_at_epoch: int = -1
    last_recalibration_epoch: int | None = None
    # Unlike last_recalibration_epoch, this advances for rejected fits too.
    # It prevents an over-age incumbent from launching the same expensive
    # candidate again at every audit while preserving the incumbent's true age.
    last_recalibration_attempt_epoch: int | None = None
    last_decision: str = "uninitialized"
    trust_initial_effective_raw_auc_gap: float = float("nan")
    trust_current_raw_auc: float = float("nan")
    trust_current_raw_auc_gap: float = float("nan")
    trust_current_raw_auc_se: float = float("nan")
    trust_current_effective_raw_auc_gap: float = float("nan")
    trust_current_delta: float = float("nan")
    # Schedule age counts installed references since opting in, not epochs or
    # historical reward_round_id. Policy-only rollback must not rewind it.
    trust_radius_decay_step: int = -1
    trust_radius_decay_round_id: int = -1
    trust_radius_decay_protocol: dict[str, float] | None = None
    # Global-best schedule persists across reference installs and policy-only rollback.
    trust_best_decay_count: int = 0
    trust_best_decay_last_step: int = -1
    trust_best_decay_round_id: int = -1
    trust_best_decay_protocol: dict[str, float] | None = None
    trust_best_global_auc_gap: float = float("nan")
    trust_best_observation_step: int = -1
    policy_warmup_round_id: int = -1
    policy_warmup_completed_updates: int = 0
    gradient_post_install_probe_round_id: int = -1
    gradient_warmup_probe_round_id: int = -1
    gradient_lifecycle_monitor_state: dict[str, Any] = field(default_factory=dict)
    gradient_lifecycle_monitor_step: int = -1
    gradient_lifecycle_raw_auc: float = float("nan")
    raw_patience_warmup_updates_seen: int = 0
    policy_warmup_protocol: dict[str, Any] | None = None
    # Fixed denominator for eta_r / eta_0 = sqrt(delta_r / delta_0).
    # Newer reward rounds must not silently rebase the DGPO LR schedule.
    trust_policy_lr_reference_delta: float = float("nan")
    trust_boundary_count: int = 0
    trust_accepted_updates: int = 0
    trust_rejection_stop_requested: bool = False
    trust_first_rejected_global_step: int = -1
    trust_statistically_closed: bool = False
    # Optional empirical cap learned from actual policy drift at a statistically
    # stale fresh audit. It is carried into later reward rounds; the current
    # round is not retroactively made infeasible after observing the boundary.
    trust_empirical_delta_cap: float = float("nan")
    trust_empirical_safe_distance: float = float("nan")
    trust_empirical_unsafe_distance: float = float("nan")
    trust_empirical_observations: int = 0
    trust_empirical_updates: int = 0
    trust_empirical_expansions: int = 0
    trust_empirical_contractions: int = 0
    trust_empirical_safe_streak: int = 0
    trust_round_plateau_streak: int = 0
    trust_round_regressions: int = 0
    trust_round_rollbacks: int = 0
    trust_round_stop_requested: bool = False
    # Best raw-AUC policy observed along the current round's *actual* training
    # trajectory.  Only the checkpoint path and scalar audit result are kept;
    # the regular unpruned epoch checkpoint already owns the model tensors.
    trust_trajectory_round_id: int = -1
    trust_trajectory_best_raw_auc_gap: float = float("nan")
    trust_trajectory_best_raw_auc_se: float = float("nan")
    trust_trajectory_best_epoch: int = -1
    trust_trajectory_best_global_step: int = -1
    trust_trajectory_best_checkpoint: str = ""
    trust_trajectory_candidate_count: int = 0
    trust_failed_direction_streak: int = 0
    # A signed probe is diagnostic-only and runs once for each independently
    # trained direction, identified by (reward round, failed-direction count).
    trust_signed_probe_round_id: int = -1
    trust_signed_probe_direction_index: int = -1
    trust_signed_probe_global_step: int = -1
    trust_distance_window: list[float] = field(default_factory=list)
    trust_step_acceptance_window: list[float] = field(default_factory=list)
    trust_step_scale_window: list[float] = field(default_factory=list)
    # CPU tensor payload containing one distinct fixed-probe shard per rank.
    # Keeping every shard in the adaptive checkpoint state makes resumed jobs
    # evaluate the same global event/timestep/noise rows as the uninterrupted
    # reward round when resumed with the same world size.
    trust_probe_payload: dict[str, Any] | None = None
    trust_probe_round_id: int = -1
    probe_history: list[dict[str, float]] = field(default_factory=list)
    raw_monitor_state: dict[str, Any] = field(default_factory=dict)
    raw_monitor_baseline_pending: bool = False
    last_staleness_step: int = -1

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
        previous_round_id = int(self.reward_round_id)
        self.baseline_auc_gap = float(baseline_auc_gap)
        self.previous_audit_auc_gap = float(baseline_auc_gap)
        self.trigger_threshold = resolve_trigger_threshold(
            baseline_auc_gap,
            retrain_auc_margin=cfg.retrain_auc_margin,
        )
        self.probe_exceedance_streak = 0
        if cfg.raw_best_scope != "global":
            self.raw_best_auc_gap = float("nan")
            self.raw_best_epoch = -1
            self.raw_best_global_step = -1
            self.raw_best_checkpoint = ""
            self.raw_best_next_epoch = -1
        self.raw_previous_auc_gap = float("nan")
        self.raw_no_improvement_streak = 0
        self.classifier_trust_exceedance_streak = 0
        self.installed_at_epoch = int(epoch)
        if round_id is not None:
            self.reward_round_id = int(round_id)
            if int(round_id) != previous_round_id:
                self.trust_distance_window.clear()
                self.trust_step_acceptance_window.clear()
                self.trust_step_scale_window.clear()
                self.trust_empirical_safe_streak = 0
                self.trust_round_plateau_streak = 0
                self.trust_round_stop_requested = False
                self.reset_trust_trajectory(reset_failed_directions=True)
                self.trust_signed_probe_round_id = -1
                self.trust_signed_probe_direction_index = -1
                self.trust_signed_probe_global_step = -1
                self.trust_probe_payload = None
                self.trust_probe_round_id = -1

    def mark_bootstrap_complete(self, *, cfg: AdaptiveOmniFoldConfig) -> None:
        """Persist reward installation before fitting the initial raw baseline."""
        if not self.calibrated:
            raise ValueError("cannot complete bootstrap without an installed calibrated round")
        self.resume_refit_once_completed = True
        self.resume_refit_once_id = cfg.refit_once_id
        self.raw_monitor_baseline_pending = cfg.monitor_mode == "raw_plateau_refit"

    def reset_trust_trajectory(
        self,
        *,
        reset_failed_directions: bool = False,
    ) -> None:
        """Forget candidates from one attempted direction, keeping the incumbent."""

        self.trust_trajectory_round_id = int(self.reward_round_id)
        self.trust_trajectory_best_raw_auc_gap = float("nan")
        self.trust_trajectory_best_raw_auc_se = float("nan")
        self.trust_trajectory_best_epoch = -1
        self.trust_trajectory_best_global_step = -1
        self.trust_trajectory_best_checkpoint = ""
        self.trust_trajectory_candidate_count = 0
        if reset_failed_directions:
            self.trust_failed_direction_streak = 0

    def invalidate_audit_baseline(self, *, reason: str) -> None:
        """Keep the installed reward round but require a fresh routine baseline."""

        self.baseline_auc_gap = float("nan")
        self.previous_audit_auc_gap = float("nan")
        self.trigger_threshold = float("nan")
        self.probe_exceedance_streak = 0
        self.raw_best_auc_gap = float("nan")
        self.raw_best_epoch = -1
        self.raw_best_global_step = -1
        self.raw_best_checkpoint = ""
        self.raw_best_next_epoch = -1
        self.raw_global_initialized = False
        self.raw_global_failed_rounds = 0
        self.raw_global_refit_pending = False
        self.raw_global_stop_requested = False
        self.raw_previous_auc_gap = float("nan")
        self.raw_no_improvement_streak = 0
        self.classifier_trust_exceedance_streak = 0
        self.last_decision = str(reason)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "AdaptiveOmniFoldState":
        if not payload:
            return cls()
        values = dict(payload)
        # Legacy resumes already completed their last logged warmup interval.
        # New checkpoints preserve the interval marker, including mid-warmup saves.
        values.setdefault("raw_patience_warmup_updates_seen", values.get("policy_warmup_completed_updates", 0))
        values["probe_history"] = [
            dict(item) for item in values.get("probe_history", []) or []
        ]
        values["trust_distance_window"] = [
            float(item)
            for item in values.get("trust_distance_window", []) or []
            if math.isfinite(float(item)) and float(item) >= 0.0
        ]
        values["trust_step_acceptance_window"] = [
            min(max(float(item), 0.0), 1.0)
            for item in values.get("trust_step_acceptance_window", []) or []
            if math.isfinite(float(item))
        ]
        values["trust_step_scale_window"] = [
            min(max(float(item), 0.0), 1.0)
            for item in values.get("trust_step_scale_window", []) or []
            if math.isfinite(float(item))
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
        # Checkpoints written before best-point rollback support already carry
        # the fixed-panel raw audit history. Recover the best epoch so the
        # trainer can resolve its unpruned epoch snapshot after a v16 restart.
        # A certified global bootstrap best legitimately has epoch=-1,
        # step=0, next_epoch=0. Never reinterpret it as missing legacy metadata
        # and overwrite only epoch/step with a later round's closest AUC.
        if (not state.raw_global_initialized
                and math.isfinite(state.raw_best_auc_gap) and state.raw_best_epoch < 0):
            best_row: Mapping[str, Any] | None = None
            best_distance = float("inf")
            for item in state.probe_history:
                if int(item.get("reward_round_id", state.reward_round_id)) != int(
                    state.reward_round_id
                ):
                    continue
                candidate = item.get("raw_auc_gap")
                if candidate is None or not math.isfinite(float(candidate)):
                    continue
                distance = abs(float(candidate) - float(state.raw_best_auc_gap))
                if distance < best_distance:
                    best_distance = distance
                    best_row = item
            if best_row is not None:
                state.raw_best_epoch = int(best_row.get("epoch", -1))
                state.raw_best_global_step = int(
                    best_row.get("global_step", -1)
                )
        return state


def auc_null_standard_error(n_positive: int, n_negative: int) -> float:
    """Mann--Whitney null standard error for an independent AUC estimate."""

    n_pos = max(float(n_positive), 1.0)
    n_neg = max(float(n_negative), 1.0)
    return math.sqrt((n_pos + n_neg + 1.0) / (12.0 * n_pos * n_neg))


def evaluate_round_auc_change(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    raw_auc: float,
    raw_auc_se: float,
    enforce: bool = True,
) -> dict[str, Any]:
    """Classify cross-round raw-AUC progress with an uncertainty dead band.

    The lower-is-better quantity is ``abs(raw_auc - 0.5)``.  A new round is
    accepted by this guard only when its reduction exceeds the configured
    normal-approximation threshold. Changes inside the threshold are plateaus;
    significant increases are regressions. Bootstrap and protocol-changing
    resume refits explicitly bypass this comparison.
    """

    current_auc = float(raw_auc)
    current_se = float(raw_auc_se)
    if not math.isfinite(current_auc) or not 0.0 <= current_auc <= 1.0:
        raise ValueError("round acceptance raw AUC must be finite and in [0, 1]")
    if not math.isfinite(current_se) or current_se < 0.0:
        raise ValueError("round acceptance raw AUC SE must be nonnegative")
    previous_gap = float(state.trust_current_raw_auc_gap)
    previous_se = float(state.trust_current_raw_auc_se)
    current_gap = abs(current_auc - 0.5)
    eligible = bool(
        cfg.trust_round_acceptance_enabled
        and enforce
        and math.isfinite(previous_gap)
        and previous_gap >= 0.0
        and math.isfinite(previous_se)
        and previous_se >= 0.0
    )
    combined_se = (
        math.hypot(previous_se, current_se) if eligible else float("nan")
    )
    required_improvement = (
        float(cfg.trust_round_acceptance_confidence_z) * combined_se
        if eligible
        else float("nan")
    )
    improvement = (
        previous_gap - current_gap if eligible else float("nan")
    )
    if not eligible:
        decision = "bypass"
        action = 2
    elif improvement > required_improvement:
        decision = "improved"
        action = 1
    elif improvement < -required_improvement:
        decision = "regressed"
        action = -1
    else:
        decision = "plateau"
        action = 0
    return {
        "enabled": bool(cfg.trust_round_acceptance_enabled),
        "enforced": bool(enforce),
        "eligible": eligible,
        "decision": decision,
        "action": action,
        "previous_auc_gap": previous_gap,
        "current_auc_gap": current_gap,
        "previous_auc_se": previous_se,
        "current_auc_se": current_se,
        "improvement": improvement,
        "combined_se": combined_se,
        "required_improvement": required_improvement,
    }


def _signed_probe_scale_token(scale: float) -> str:
    sign = "plus" if float(scale) > 0.0 else "minus"
    magnitude = f"{abs(float(scale)):.6g}".replace(".", "p")
    return f"{sign}_{magnitude}"


def evaluate_signed_direction_probe(
    candidates: list[Mapping[str, float]],
    *,
    anchor_raw_auc_gap: float,
    anchor_raw_auc_se: float,
    anchor_reward_mean: float,
    confidence_z: float,
) -> dict[str, float]:
    """Summarize raw-AUC alignment along both signs of one trained direction.

    Each candidate must provide ``signed_scale``, ``raw_auc``,
    ``raw_auc_gap``, ``raw_auc_se``, and ``reward_mean``.  The diagnostic is
    deliberately read-only: it reports whether either sign beats the installed
    anchor beyond the same normal-approximation dead band used by the round
    guard, but it never changes the live policy or acceptance controller.

    ``decision_code`` is +1 when only the trained direction improves, -1 when
    only its reverse improves, +2 when both improve, and 0 when neither does.
    """

    anchor_gap = float(anchor_raw_auc_gap)
    anchor_se = float(anchor_raw_auc_se)
    anchor_reward = float(anchor_reward_mean)
    z_value = float(confidence_z)
    if not math.isfinite(anchor_gap) or not 0.0 <= anchor_gap <= 0.5:
        raise ValueError("signed-direction anchor raw AUC gap must lie in [0, 0.5]")
    if not math.isfinite(anchor_se) or anchor_se < 0.0:
        raise ValueError("signed-direction anchor raw AUC SE must be nonnegative")
    if not math.isfinite(anchor_reward):
        raise ValueError("signed-direction anchor reward mean must be finite")
    if not math.isfinite(z_value) or z_value < 0.0:
        raise ValueError("signed-direction confidence_z must be nonnegative")
    if not candidates:
        raise ValueError("signed-direction probe requires at least one candidate")

    rows: list[dict[str, float]] = []
    observed_scales: set[float] = set()
    for candidate in candidates:
        scale = float(candidate["signed_scale"])
        raw_auc = float(candidate["raw_auc"])
        raw_gap = float(candidate["raw_auc_gap"])
        raw_se = float(candidate["raw_auc_se"])
        reward_mean = float(candidate["reward_mean"])
        if (
            not math.isfinite(scale)
            or scale == 0.0
            or abs(scale) > 1.0
        ):
            raise ValueError("signed-direction candidate scale must lie in [-1, 1] excluding 0")
        if scale in observed_scales:
            raise ValueError("signed-direction candidate scales must be unique")
        observed_scales.add(scale)
        if not math.isfinite(raw_auc) or not 0.0 <= raw_auc <= 1.0:
            raise ValueError("signed-direction candidate raw AUC must lie in [0, 1]")
        if not math.isfinite(raw_gap) or not 0.0 <= raw_gap <= 0.5:
            raise ValueError("signed-direction candidate raw AUC gap must lie in [0, 0.5]")
        if not math.isfinite(raw_se) or raw_se < 0.0:
            raise ValueError("signed-direction candidate raw AUC SE must be nonnegative")
        if not math.isfinite(reward_mean):
            raise ValueError("signed-direction candidate reward mean must be finite")
        required = z_value * math.hypot(anchor_se, raw_se)
        improvement = anchor_gap - raw_gap
        rows.append(
            {
                "signed_scale": scale,
                "raw_auc": raw_auc,
                "raw_auc_gap": raw_gap,
                "raw_auc_se": raw_se,
                "reward_mean": reward_mean,
                "improvement": improvement,
                "required_improvement": required,
                "significant_improvement": float(improvement > required),
                "significant_regression": float(improvement < -required),
            }
        )

    positive = [row for row in rows if row["signed_scale"] > 0.0]
    negative = [row for row in rows if row["signed_scale"] < 0.0]
    if not positive or not negative:
        raise ValueError("signed-direction probe requires both positive and negative candidates")
    positive_endpoint = next(
        (
            row
            for row in positive
            if math.isclose(
                row["signed_scale"], 1.0, rel_tol=0.0, abs_tol=1.0e-12
            )
        ),
        None,
    )
    if positive_endpoint is None:
        raise ValueError("signed-direction probe requires the trained +1 endpoint")
    best_positive = min(positive, key=lambda row: row["raw_auc_gap"])
    best_negative = min(negative, key=lambda row: row["raw_auc_gap"])
    best_overall = min(rows, key=lambda row: row["raw_auc_gap"])
    reward_best = max(rows, key=lambda row: row["reward_mean"])
    positive_improved = bool(best_positive["significant_improvement"] >= 0.5)
    negative_improved = bool(best_negative["significant_improvement"] >= 0.5)
    decision_code = (
        2.0
        if positive_improved and negative_improved
        else 1.0
        if positive_improved
        else -1.0
        if negative_improved
        else 0.0
    )
    best_sign = (
        math.copysign(1.0, best_overall["signed_scale"])
        if best_overall["raw_auc_gap"] < anchor_gap
        else 0.0
    )
    metrics: dict[str, float] = {
        "reference_trust/signed_probe/completed": 1.0,
        "reference_trust/signed_probe/anchor_raw_auc_gap": anchor_gap,
        "reference_trust/signed_probe/anchor_raw_auc_se": anchor_se,
        "reference_trust/signed_probe/anchor_reward_mean": anchor_reward,
        "reference_trust/signed_probe/confidence_z": z_value,
        "reference_trust/signed_probe/decision_code": decision_code,
        "reference_trust/signed_probe/best_sign": best_sign,
        "reference_trust/signed_probe/positive_significant_improvement": float(
            positive_improved
        ),
        "reference_trust/signed_probe/negative_significant_improvement": float(
            negative_improved
        ),
        "reference_trust/signed_probe/reverse_only_improves": float(
            decision_code == -1.0
        ),
        "reference_trust/signed_probe/neither_sign_improves": float(
            decision_code == 0.0
        ),
        "reference_trust/signed_probe/smaller_positive_step_improves": float(
            positive_improved
            and best_positive["signed_scale"] < 1.0
            and positive_endpoint["significant_improvement"] < 0.5
        ),
        "reference_trust/signed_probe/trained_endpoint_improvement": float(
            positive_endpoint["improvement"]
        ),
        "reference_trust/signed_probe/trained_endpoint_reward_change": float(
            positive_endpoint["reward_mean"] - anchor_reward
        ),
        "reference_trust/signed_probe/best_positive_scale": float(
            best_positive["signed_scale"]
        ),
        "reference_trust/signed_probe/best_positive_raw_auc_gap": float(
            best_positive["raw_auc_gap"]
        ),
        "reference_trust/signed_probe/best_negative_scale": float(
            best_negative["signed_scale"]
        ),
        "reference_trust/signed_probe/best_negative_raw_auc_gap": float(
            best_negative["raw_auc_gap"]
        ),
        "reference_trust/signed_probe/best_overall_scale": float(
            best_overall["signed_scale"]
        ),
        "reference_trust/signed_probe/best_overall_raw_auc_gap": float(
            best_overall["raw_auc_gap"]
        ),
        "reference_trust/signed_probe/reward_best_scale": float(
            reward_best["signed_scale"]
        ),
        "reference_trust/signed_probe/reward_best_reward_change": float(
            reward_best["reward_mean"] - anchor_reward
        ),
        "reference_trust/signed_probe/reward_best_raw_regression": float(
            reward_best["significant_regression"]
        ),
        "reference_trust/signed_probe/reward_raw_misaligned": float(
            reward_best["reward_mean"] > anchor_reward
            and reward_best["significant_regression"] >= 0.5
        ),
    }
    for row in rows:
        prefix = (
            "reference_trust/signed_probe/candidate/"
            + _signed_probe_scale_token(row["signed_scale"])
        )
        for key, value in row.items():
            if key == "signed_scale":
                continue
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics


def evaluate_signed_direction_recovery(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    signed_probe: Mapping[str, Any],
) -> dict[str, float]:
    """Decide whether a saturated reverse-only probe needs a fresh reward.

    This helper is deliberately side-effect free.  The trainer owns the
    distributed policy rollback, optimizer reset, and reward refit; keeping the
    decision here makes the statistical preconditions independently testable.
    A reverse-only result is actionable only after every raw classifier audit
    has saturated.  Repeated recoveries use the existing failed-direction
    patience so an unproductive refresh loop terminates at the incumbent.
    """

    enabled = bool(cfg.trust_signed_direction_recovery_enabled)
    completed = bool(
        float(
            signed_probe.get(
                "reference_trust/signed_probe/completed",
                0.0,
            )
        )
        >= 0.5
    )
    saturated = bool(
        float(
            signed_probe.get(
                "reference_trust/signed_probe/all_raw_audits_saturated",
                0.0,
            )
        )
        >= 0.5
    )
    decision_code = float(
        signed_probe.get(
            "reference_trust/signed_probe/decision_code",
            float("nan"),
        )
    )
    eligible = bool(enabled and completed and saturated)
    triggered = bool(eligible and decision_code == -1.0)
    previous_streak = int(state.trust_failed_direction_streak)
    recovery_attempt = previous_streak + int(triggered)
    patience = int(cfg.trust_failed_direction_patience)
    stop_requested = bool(triggered and recovery_attempt >= patience)
    return {
        "reference_trust/signed_probe/recovery_enabled": float(enabled),
        "reference_trust/signed_probe/recovery_eligible": float(eligible),
        "reference_trust/signed_probe/recovery_triggered": float(triggered),
        "reference_trust/signed_probe/recovery_attempt": float(
            recovery_attempt
        ),
        "reference_trust/signed_probe/recovery_patience": float(patience),
        "reference_trust/signed_probe/recovery_stop_requested": float(
            stop_requested
        ),
    }


def record_extragradient_rejection(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    reason: str,
) -> dict[str, float]:
    """Reject one predictive-corrective direction at the saved incumbent.

    The trainer restores the policy/reward/reference snapshot before calling
    this helper.  Keeping the counter/radius transition here makes a failed
    transient look-ahead indistinguishable from any other failed trajectory:
    one failure consumes one patience slot and contracts the next radius once.
    """

    delta_before = float(state.trust_current_delta)
    if str(reason) in {
        "lookahead_classifier_gate_failed",
        "final_classifier_gate_failed",
    }:
        state.recalibrations_rejected += 1
    state.trust_failed_direction_streak += 1
    state.trust_round_rollbacks += 1
    if math.isfinite(delta_before) and delta_before > 0.0:
        state.trust_current_delta = max(
            float(cfg.trust_delta_floor),
            delta_before * float(cfg.trust_empirical_shrink_factor),
        )
    state.trust_distance_window.clear()
    state.trust_step_acceptance_window.clear()
    state.trust_step_scale_window.clear()
    state.reset_trust_trajectory(reset_failed_directions=False)
    stop_requested = bool(
        state.trust_failed_direction_streak
        >= int(cfg.trust_failed_direction_patience)
    )
    state.trust_round_stop_requested = stop_requested
    state.last_decision = (
        "extragradient_corrector_rejected_stop"
        if stop_requested
        else "extragradient_corrector_rejected_rollback"
    )
    return {
        "reference_trust/extragradient/rejected": 1.0,
        "reference_trust/extragradient/stop_requested": float(stop_requested),
        "reference_trust/extragradient/failed_direction_streak": float(
            state.trust_failed_direction_streak
        ),
        "reference_trust/extragradient/rejection_reason_code": float(
            {
                "nonfinite_or_zero_corrector": 1,
                "trust_backtracking_failed": 2,
                "final_classifier_gate_failed": 3,
                "lookahead_classifier_gate_failed": 4,
            }.get(str(reason), 0)
        ),
        "reference_trust/extragradient/delta_before": delta_before,
        "reference_trust/extragradient/delta_after": float(
            state.trust_current_delta
        ),
        "reference_trust/round_acceptance/rollback_required": 1.0,
        "reference_trust/round_acceptance/rollbacks": float(
            state.trust_round_rollbacks
        ),
        "reference_trust/round_acceptance/failed_direction_streak": float(
            state.trust_failed_direction_streak
        ),
        "omnifold/recalibrations_rejected": float(
            state.recalibrations_rejected
        ),
        "reference_trust/round_acceptance/stop_requested": float(
            stop_requested
        ),
        "reference_trust/round_acceptance/delta_before": delta_before,
        "reference_trust/round_acceptance/delta_after": float(
            state.trust_current_delta
        ),
    }


def update_trust_trajectory_candidate(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    raw_auc_gap: float,
    raw_auc_se: float,
    epoch: int,
    global_step: int,
    checkpoint_path: str,
    audit_saturated: bool,
) -> dict[str, float]:
    """Record the lowest raw-AUC checkpoint on the current training direction.

    Routine audits use a fixed event/noise/classifier-seed panel when trajectory
    search is enabled.  This cheap ranking is deliberately permissive: the
    selected checkpoint must still pass the independent, large round-acceptance
    audit before it can become the next reward/reference pair.
    """

    enabled = bool(cfg.trust_trajectory_search_enabled)
    gap = float(raw_auc_gap)
    se = float(raw_auc_se)
    eligible = bool(
        enabled
        and audit_saturated
        and math.isfinite(gap)
        and 0.0 <= gap <= 0.5
        and math.isfinite(se)
        and se >= 0.0
        and bool(str(checkpoint_path))
    )
    if int(state.trust_trajectory_round_id) != int(state.reward_round_id):
        state.reset_trust_trajectory(reset_failed_directions=False)
    updated = bool(
        eligible
        and (
            not math.isfinite(state.trust_trajectory_best_raw_auc_gap)
            or gap < float(state.trust_trajectory_best_raw_auc_gap)
        )
    )
    if eligible:
        state.trust_trajectory_candidate_count += 1
    if updated:
        state.trust_trajectory_best_raw_auc_gap = gap
        state.trust_trajectory_best_raw_auc_se = se
        state.trust_trajectory_best_epoch = int(epoch)
        state.trust_trajectory_best_global_step = int(global_step)
        state.trust_trajectory_best_checkpoint = str(checkpoint_path)
    return {
        "reference_trust/trajectory/enabled": float(enabled),
        "reference_trust/trajectory/eligible": float(eligible),
        "reference_trust/trajectory/best_updated": float(updated),
        "reference_trust/trajectory/current_raw_auc_gap": gap,
        "reference_trust/trajectory/current_raw_auc_se": se,
        "reference_trust/trajectory/best_raw_auc_gap": float(
            state.trust_trajectory_best_raw_auc_gap
        ),
        "reference_trust/trajectory/best_raw_auc_se": float(
            state.trust_trajectory_best_raw_auc_se
        ),
        "reference_trust/trajectory/best_epoch": float(
            state.trust_trajectory_best_epoch
        ),
        "reference_trust/trajectory/best_global_step": float(
            state.trust_trajectory_best_global_step
        ),
        "reference_trust/trajectory/candidate_count": float(
            state.trust_trajectory_candidate_count
        ),
        "reference_trust/trajectory/failed_direction_streak": float(
            state.trust_failed_direction_streak
        ),
    }


def migrate_unstarted_policy_warmup_after_resume(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
) -> bool:
    """Allow a new warmup length only before any updates in the saved round."""
    old = state.policy_warmup_protocol
    if not cfg.policy_warmup_steps or old is None:
        return False
    target = {"steps": cfg.policy_warmup_steps, "start_factor": cfg.policy_warmup_start_factor}
    if old == target:
        return False
    if (not isinstance(old, dict)
            or set(old) != {"steps", "start_factor"}
            or type(old["steps"]) is not int or old["steps"] <= 0
            or old["start_factor"] != cfg.policy_warmup_start_factor
            or state.policy_warmup_round_id != state.reward_round_id
            or type(state.policy_warmup_completed_updates) is not int
            or state.policy_warmup_completed_updates != 0
            or state.raw_patience_warmup_updates_seen != 0):
        raise ValueError("warmup length can change on full resume only before any updates in the installed round")
    state.policy_warmup_protocol = target
    return True


def start_inherited_round_policy_warmup(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
    global_step: int | None = None, restart: bool = False,
) -> dict[str, float]:
    """Start warmup after inheriting an installed round without a new install.

    Best-point and pinned-classifier restarts keep ``reward_round_id`` but
    rebuild adaptive state without warmup clocks. The post-baseline
    last.ckpt may already have overwritten ``last_decision``, so a step-0
    checkpoint whose warmup never started is also eligible. Regular resume
    of a mid-round checkpoint that never started warmup still fail-closes.
    """
    if cfg.policy_warmup_steps == 0 or state.policy_warmup_protocol is not None:
        return {}
    if state.reward_round_id < 0:
        return {}
    unstarted = (
        state.policy_warmup_round_id == -1
        and type(state.policy_warmup_completed_updates) is int
        and state.policy_warmup_completed_updates == 0
        and state.raw_patience_warmup_updates_seen == 0
    )
    if not unstarted:
        return {}
    if not (
        restart
        or state.last_decision == "new_experiment_from_best"
        or global_step == 0
    ):
        return {}
    return start_policy_round_warmup(state, cfg=cfg)


def start_policy_round_warmup(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
) -> dict[str, float]:
    """Call only after accepted reward installation; idempotent within a round."""
    if cfg.policy_warmup_steps == 0:
        return {}
    protocol = {"steps": cfg.policy_warmup_steps, "start_factor": cfg.policy_warmup_start_factor}
    if state.policy_warmup_round_id > state.reward_round_id:
        raise ValueError("policy warmup reference round cannot rewind")
    if state.policy_warmup_round_id != state.reward_round_id:
        state.policy_warmup_round_id = state.reward_round_id
        state.policy_warmup_completed_updates = 0
        state.raw_patience_warmup_updates_seen = 0
        state.policy_warmup_protocol = protocol
    elif state.policy_warmup_protocol != protocol:
        raise ValueError("policy warmup protocol changed within an installed round")
    return policy_round_warmup_metrics(state, cfg=cfg)


def policy_round_warmup_metrics(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
) -> dict[str, float]:
    """LR multiplier for the NEXT accepted update, relative to scheduled group LRs."""
    if cfg.policy_warmup_steps == 0:
        return {}
    protocol = {"steps": cfg.policy_warmup_steps, "start_factor": cfg.policy_warmup_start_factor}
    if state.policy_warmup_protocol != protocol or state.policy_warmup_round_id != state.reward_round_id:
        raise ValueError("policy warmup state does not match installed round/config; refit or weights-only restart required")
    completed = state.policy_warmup_completed_updates
    if (isinstance(completed, bool) or not isinstance(completed, int)
            or not 0 <= completed <= cfg.policy_warmup_steps):
        raise ValueError("invalid checkpointed policy warmup update count")
    progress = min(1., completed / max(1, cfg.policy_warmup_steps - 1))
    scale = (1. if cfg.policy_warmup_steps == 1 else
             cfg.policy_warmup_start_factor + (1. - cfg.policy_warmup_start_factor) * progress)
    return {"train/round_warmup/lr_scale": scale,
            "train/round_warmup/completed_updates": float(completed),
            "train/round_warmup/round_id": float(state.policy_warmup_round_id)}


def advance_policy_round_warmup(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig, accepted: bool,
) -> None:
    if cfg.policy_warmup_steps == 0:
        return
    policy_round_warmup_metrics(state, cfg=cfg)  # Validate before changing serialized state.
    if accepted:
        state.policy_warmup_completed_updates = min(
            cfg.policy_warmup_steps, state.policy_warmup_completed_updates + 1,
        )


def _preview_round_decay_radius(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    round_id: int,
) -> tuple[float, int, dict[str, float]]:
    """Preview an idempotent, checkpointed schedule without moving its clock."""

    protocol = {
        "initial": float(cfg.trust_delta_max),
        "floor": float(cfg.trust_delta_floor),
        "factor": float(cfg.trust_round_decay_factor),
    }
    step = state.trust_radius_decay_step
    saved_round = state.trust_radius_decay_round_id
    if (
        isinstance(step, bool) or not isinstance(step, int) or step < -1
        or isinstance(saved_round, bool) or not isinstance(saved_round, int)
        or saved_round < -1 or round_id < 0
    ):
        raise ValueError("invalid checkpointed round-decay counters")
    if step == -1:
        if saved_round != -1 or state.trust_radius_decay_protocol is not None:
            raise ValueError("incomplete checkpointed round-decay schedule")
        next_step = 0
    else:
        if state.trust_radius_decay_protocol != protocol:
            raise ValueError(
                "round-decay schedule differs from checkpoint; keep its initial, "
                "floor, and factor unchanged on resume"
            )
        if saved_round < 0 or round_id < saved_round:
            raise ValueError("round-decay schedule cannot rewind its reference round")
        expected = max(protocol["floor"], protocol["initial"] * protocol["factor"] ** step)
        if not math.isclose(float(state.trust_current_delta), expected, rel_tol=1e-12):
            raise ValueError("checkpointed radius does not match its round-decay schedule")
        next_step = step + int(round_id > saved_round)
    delta = max(protocol["floor"], protocol["initial"] * protocol["factor"] ** next_step)
    return delta, next_step, protocol


def _preview_best_decay_radius(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig, round_id: int,
) -> tuple[float, float, dict[str, float]]:
    protocol = {"initial": cfg.trust_delta_max, "floor": cfg.trust_delta_floor,
                "factor": cfg.trust_best_decay_factor}
    count = state.trust_best_decay_count
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("invalid checkpointed best-decay counter")
    observed_step = state.trust_best_observation_step
    if (isinstance(observed_step, bool) or not isinstance(observed_step, int)
            or observed_step < -1 or state.trust_best_decay_last_step > observed_step):
        raise ValueError("invalid checkpointed global-best observation step")
    global_gap = float(state.trust_best_global_auc_gap)
    if (count > 0 or observed_step >= 0) and not (
        math.isfinite(global_gap) and 0.0 <= global_gap <= .5
    ):
        raise ValueError("missing or invalid checkpointed global raw-AUC best")
    if state.trust_best_decay_protocol is None:
        if (count != 0 or state.trust_best_decay_last_step != -1
                or state.trust_best_decay_round_id != -1 or observed_step != -1):
            raise ValueError("incomplete checkpointed best-decay schedule")
    elif state.trust_best_decay_protocol != protocol:
        raise ValueError("best-decay schedule differs from checkpoint; keep initial/floor/factor unchanged")
    if round_id < 0 or round_id < state.trust_best_decay_round_id:
        raise ValueError("best-decay reference round cannot rewind")
    target = max(protocol["floor"], protocol["initial"] * protocol["factor"] ** count)
    if state.trust_best_decay_protocol is not None:
        current = float(state.trust_current_delta)
        if not math.isfinite(current) or not target <= current <= protocol["initial"]:
            raise ValueError("checkpointed best-decay radius is outside schedule bounds")
        # Same-reference previews must preserve a feasibility-limited radius.
        if round_id == state.trust_best_decay_round_id:
            return current, target, protocol
    # A newly installed reference is centered on the live policy (distance zero).
    return target, target, protocol


def raw_best_trust_shrink_due(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
    diagnostics: Mapping[str, Any], global_step: int,
) -> bool:
    """Pure predicate for a saturated GLOBAL improvement requiring a trust probe."""
    gap = float(diagnostics.get("staleness/raw_auc_gap", float("nan")))
    return bool(
        cfg.trust_boundary_enabled and cfg.trust_radius_mode == "best_decay"
        and (cfg.raw_best_scope != "global" or float(diagnostics.get("staleness/global_best/improved", 0)) >= .5)
        and float(diagnostics.get("staleness/raw_audit_saturated", 0)) >= .5
        and math.isfinite(gap) and 0.0 <= gap <= .5
        and math.isfinite(state.trust_best_global_auc_gap)
        and gap < state.trust_best_global_auc_gap - cfg.raw_improvement_min_delta
        and int(global_step) > state.trust_best_observation_step
    )


def shrink_trust_radius_on_raw_best(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
    diagnostics: Mapping[str, Any], current_distance: float, global_step: int,
) -> dict[str, float]:
    """Contract on global raw-best improvements, never on first baseline registration.

    Retain headroom above the SAME fixed probe used for enforcement. If the
    requested radius would exclude the live policy, defer that portion until a
    later improvement or reference recenter. No policy/reference weights move.
    """
    if not cfg.trust_boundary_enabled or cfg.trust_radius_mode != "best_decay":
        return {}
    before, target, _ = _preview_best_decay_radius(state, cfg=cfg, round_id=state.reward_round_id)
    if state.trust_best_decay_protocol is None:
        raise ValueError("best-decay must be initialized by a reference install")
    gap = float(diagnostics.get("staleness/raw_auc_gap", float("nan")))
    observed = (
        float(diagnostics.get("staleness/raw_audit_saturated", 0)) >= .5
        and math.isfinite(gap) and 0.0 <= gap <= .5
        and int(global_step) > state.trust_best_observation_step
    )
    eligible = raw_best_trust_shrink_due(state, cfg=cfg, diagnostics=diagnostics, global_step=global_step)
    if eligible:
        if not math.isfinite(current_distance) or current_distance < 0:
            raise ValueError("best-decay requires a finite nonnegative fixed-probe distance")
        if current_distance >= before:
            raise ValueError("cannot shrink best-decay radius around an already infeasible policy")
        state.trust_best_decay_count += 1
        state.trust_best_decay_last_step = int(global_step)
        target = max(cfg.trust_delta_floor,
                     cfg.trust_delta_max * cfg.trust_best_decay_factor ** state.trust_best_decay_count)
        safe_radius = current_distance / cfg.trust_warning_fraction
        state.trust_current_delta = min(before, max(target, safe_radius))
    if observed:
        state.trust_best_observation_step = int(global_step)
        if eligible or not math.isfinite(state.trust_best_global_auc_gap):
            state.trust_best_global_auc_gap = gap
    return {
        "reference_trust/delta": float(state.trust_current_delta),
        "reference_trust/best_decay/count": float(state.trust_best_decay_count),
        "reference_trust/best_decay/target": target,
        "reference_trust/best_decay/new_best": float(eligible),
        "reference_trust/best_decay/shrink_applied": float(state.trust_current_delta < before),
        "reference_trust/best_decay/feasibility_limited": float(state.trust_current_delta > target),
        "reference_trust/best_decay/global_best_auc_gap": float(state.trust_best_global_auc_gap),
    }


def install_adaptive_trust_round(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    raw_auc: float,
    raw_auc_se: float,
    commit: bool = True,
    round_id: int | None = None,
) -> dict[str, float]:
    """Install a fixed, AUC-scaled, round-decaying, or global-best-decaying radius.

    ``velocity_mse_ratio`` is a squared functional distance.  The statistically
    resolved raw-classifier AUC gap is therefore raised to ``adaptive_power=2``
    by default, making the permitted RMS velocity move shrink linearly with the
    remaining classifier discrepancy.
    """

    if not cfg.trust_boundary_enabled:
        return {}
    target_round = int(state.reward_round_id if round_id is None else round_id)
    decay_preview = (
        _preview_round_decay_radius(state, cfg=cfg, round_id=target_round)
        if cfg.trust_radius_mode == "round_decay"
        else None
    )
    best_preview = (
        _preview_best_decay_radius(state, cfg=cfg, round_id=target_round)
        if cfg.trust_radius_mode == "best_decay" else None
    )
    auc = float(raw_auc)
    auc_se = float(raw_auc_se)
    if not math.isfinite(auc) or not 0.0 <= auc <= 1.0:
        raise ValueError("adaptive trust raw AUC must be finite and in [0, 1]")
    if not math.isfinite(auc_se) or auc_se < 0.0:
        raise ValueError("adaptive trust raw AUC SE must be finite and nonnegative")
    observed_gap = abs(auc - 0.5)
    effective_gap = max(
        observed_gap - float(cfg.trust_auc_confidence_z) * auc_se,
        0.0,
    )
    initial_gap = float(state.trust_initial_effective_raw_auc_gap)
    if not math.isfinite(initial_gap) or initial_gap <= 0.0:
        # Keep a nonzero scale even if the very first classifier is already at
        # statistical closure.  ``trust_statistically_closed`` then tells the
        # outer loop not to spend this numerical floor as a real update budget.
        initial_gap = max(
            float(cfg.trust_initial_raw_auc_gap)
            if cfg.trust_initial_raw_auc_gap is not None
            else effective_gap,
            1.0e-12,
        )
        if commit:
            state.trust_initial_effective_raw_auc_gap = initial_gap
    relative_gap = min(max(effective_gap / initial_gap, 0.0), 1.0)
    auc_scaled_delta = max(
        float(cfg.trust_delta_floor),
        float(cfg.trust_delta_max)
        * relative_gap ** float(cfg.trust_adaptive_power),
    )
    configured_delta = (
        float(cfg.trust_delta_max)
        if cfg.trust_radius_mode == "fixed"
        else auc_scaled_delta
    )
    if decay_preview is not None:
        configured_delta = decay_preview[0]
    if best_preview is not None:
        configured_delta = best_preview[0]
    empirical_cap = float(state.trust_empirical_delta_cap)
    empirical_cap_active = bool(
        cfg.trust_empirical_radius_enabled
        and math.isfinite(empirical_cap)
        and empirical_cap > 0.0
    )
    delta_before_cross_round_cap = (
        max(
            float(cfg.trust_delta_floor),
            min(configured_delta, empirical_cap),
        )
        if empirical_cap_active
        else configured_delta
    )
    previous_round_delta = float(state.trust_current_delta)
    cross_round_cap_active = bool(
        cfg.trust_cross_round_nonexpanding
        and math.isfinite(previous_round_delta)
        and previous_round_delta > 0.0
    )
    delta = (
        max(
            float(cfg.trust_delta_floor),
            min(delta_before_cross_round_cap, previous_round_delta),
        )
        if cross_round_cap_active
        else delta_before_cross_round_cap
    )
    statistically_closed = bool(
        observed_gap
        <= max(
            float(cfg.trust_auc_stop_gap),
            float(cfg.trust_auc_confidence_z) * auc_se,
        )
    )
    if commit:
        state.trust_current_raw_auc = auc
        state.trust_current_raw_auc_gap = observed_gap
        state.trust_current_raw_auc_se = auc_se
        state.trust_current_effective_raw_auc_gap = effective_gap
        state.trust_current_delta = delta
        if decay_preview is not None:
            state.trust_radius_decay_step = decay_preview[1]
            state.trust_radius_decay_round_id = target_round
            state.trust_radius_decay_protocol = decay_preview[2]
        if best_preview is not None:
            state.trust_best_decay_round_id = target_round
            state.trust_best_decay_protocol = best_preview[2]
        if (
            cfg.trust_policy_lr_scaling_enabled
            and (
                not math.isfinite(state.trust_policy_lr_reference_delta)
                or state.trust_policy_lr_reference_delta <= 0.0
            )
        ):
            state.trust_policy_lr_reference_delta = delta
        state.trust_statistically_closed = statistically_closed
    lr_reference_delta = float(state.trust_policy_lr_reference_delta)
    if (
        cfg.trust_policy_lr_scaling_enabled
        and (not math.isfinite(lr_reference_delta) or lr_reference_delta <= 0.0)
    ):
        # A side-effect-free preview of the first round should report the same
        # scale that its committed installation will establish.
        lr_reference_delta = delta
    policy_lr_scale = (
        min(
            1.0,
            max(
                float(cfg.trust_policy_lr_scale_floor),
                math.sqrt(max(delta, 0.0) / lr_reference_delta),
            ),
        )
        if cfg.trust_policy_lr_scaling_enabled
        and math.isfinite(lr_reference_delta)
        and lr_reference_delta > 0.0
        else 1.0
    )
    return {
        "reference_trust/raw_auc": auc,
        "reference_trust/raw_auc_gap": observed_gap,
        "reference_trust/raw_auc_se": auc_se,
        "reference_trust/effective_raw_auc_gap": effective_gap,
        "reference_trust/initial_effective_raw_auc_gap": initial_gap,
        "reference_trust/relative_raw_auc_gap": relative_gap,
        "reference_trust/auc_scaled_delta": auc_scaled_delta,
        "reference_trust/radius_mode_fixed": float(
            cfg.trust_radius_mode == "fixed"
        ),
        **({
            "reference_trust/radius_mode_round_decay": 1.0,
            "reference_trust/round_decay/step": float(decay_preview[1]),
            "reference_trust/round_decay/factor": cfg.trust_round_decay_factor,
            "reference_trust/round_decay/floor": cfg.trust_delta_floor,
        } if decay_preview is not None else {}),
        "reference_trust/cross_round/delta_before_cap": (
            delta_before_cross_round_cap
        ),
        "reference_trust/cross_round/previous_delta": previous_round_delta,
        "reference_trust/cross_round/cap_active": float(
            cross_round_cap_active
        ),
        "reference_trust/empirical_delta_cap": (
            empirical_cap if empirical_cap_active else float("nan")
        ),
        "reference_trust/empirical_delta_cap_active": float(
            empirical_cap_active
        ),
        "reference_trust/delta": delta,
        **({
            "reference_trust/best_decay/count": float(state.trust_best_decay_count),
            "reference_trust/best_decay/target": best_preview[1],
        } if best_preview is not None else {}),
        "reference_trust/policy_lr/reference_delta": lr_reference_delta,
        "reference_trust/policy_lr/scale": policy_lr_scale,
        "reference_trust/statistically_closed": float(statistically_closed),
    }


def adaptive_trust_policy_lr_scale(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
) -> dict[str, float]:
    """Return the current DGPO LR multiplier implied by the trust radius.

    The trust distance is a squared functional displacement, so the optimizer
    scale follows ``sqrt(delta_r / delta_0)``. The first active round fixes
    ``delta_0`` in checkpoint state; later rounds may shrink but never rebase it.
    """

    delta = float(state.trust_current_delta)
    enabled = bool(
        cfg.trust_boundary_enabled and cfg.trust_policy_lr_scaling_enabled
    )
    if not (enabled and math.isfinite(delta) and delta > 0.0):
        return {
            "reference_trust/policy_lr/enabled": float(enabled),
            "reference_trust/policy_lr/reference_delta": float(
                state.trust_policy_lr_reference_delta
            ),
            "reference_trust/policy_lr/scale": 1.0,
        }
    reference_delta = float(state.trust_policy_lr_reference_delta)
    if not math.isfinite(reference_delta) or reference_delta <= 0.0:
        reference_delta = delta
        state.trust_policy_lr_reference_delta = reference_delta
    scale = min(
        1.0,
        max(
            float(cfg.trust_policy_lr_scale_floor),
            math.sqrt(delta / reference_delta),
        ),
    )
    return {
        "reference_trust/policy_lr/enabled": 1.0,
        "reference_trust/policy_lr/reference_delta": reference_delta,
        "reference_trust/policy_lr/scale": scale,
    }


def clamp_fixed_trust_radius_after_resume(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    initialize_round_decay: bool = True,
) -> dict[str, float]:
    """Synchronize the restored live radius with the active resume protocol.

    A genuinely fixed-radius run must use the value in the current config, not
    a stale value serialized by the checkpoint it branches from. For adaptive
    non-expanding protocols, recompute the installed reward round's AUC-scaled
    radius and only ever shrink the restored live radius to it. Any stricter
    empirical cap remains active. Round-decay resumes preserve the stored age;
    legacy installed references opt in at age zero, while cold bootstrap leaves
    initialization to its first successful install. Global-best resumes retain
    their feasibility-protected radius and global record without another decay;
    a best-point / pinned restart (installed reference, reset schedule) opts in
    at age zero with the full initial radius. This is idempotent.
    """

    if not cfg.trust_boundary_enabled:
        return {}
    if cfg.trust_radius_mode == "best_decay":
        if state.trust_best_decay_protocol is None:
            if not initialize_round_decay:
                return {}
            # A best-point / pinned-classifier restart keeps the installed
            # reference but resets its trust schedule (no live radius, no
            # global record).  Opt that inherited reference in at age zero,
            # exactly as round_decay does.  A checkpoint that still carries a
            # live radius without a best-decay schedule is a mid-run protocol
            # switch; expanding it silently is not allowed.
            if not state.calibrated or math.isfinite(state.trust_current_delta):
                raise ValueError("best_decay requires a fresh reference install; use weights_only to branch")
            delta, target, protocol = _preview_best_decay_radius(
                state, cfg=cfg, round_id=int(state.reward_round_id),
            )
            state.trust_current_delta = delta
            state.trust_best_decay_round_id = int(state.reward_round_id)
            state.trust_best_decay_protocol = protocol
            return {"reference_trust/delta": delta,
                    "reference_trust/best_decay/count": float(state.trust_best_decay_count),
                    "reference_trust/best_decay/target": target,
                    "reference_trust/best_decay/inherited_reference_age_zero": 1.0}
        if state.trust_best_decay_round_id != state.reward_round_id:
            raise ValueError("checkpointed best-decay reference id does not match reward")
        delta, target, _ = _preview_best_decay_radius(state, cfg=cfg, round_id=state.reward_round_id)
        return {"reference_trust/delta": delta,
                "reference_trust/best_decay/count": float(state.trust_best_decay_count),
                "reference_trust/best_decay/target": target}
    if cfg.trust_radius_mode == "round_decay":
        # Cold startup has no installed reference yet: its first successful
        # bootstrap must receive step 0, not a premature decay to step 1.
        if state.trust_radius_decay_step == -1 and not initialize_round_decay:
            return {}
        if (
            state.trust_radius_decay_step >= 0
            and state.trust_radius_decay_round_id != state.reward_round_id
        ):
            raise ValueError("checkpointed round-decay reference id does not match reward")
        previous_delta = float(state.trust_current_delta)
        delta, step, protocol = _preview_round_decay_radius(
            state, cfg=cfg, round_id=int(state.reward_round_id),
        )
        if state.trust_radius_decay_step == -1 and previous_delta > delta:
            raise ValueError(
                "cannot start round_decay by shrinking an installed reference's "
                "radius; recenter first or use an initial radius at least as large"
            )
        state.trust_current_delta = delta
        state.trust_radius_decay_step = step
        state.trust_radius_decay_round_id = int(state.reward_round_id)
        state.trust_radius_decay_protocol = protocol
        return {
            "reference_trust/resume_radius_clamp_enabled": 1.0,
            "reference_trust/resume_radius_clamp_applied": float(
                not math.isclose(previous_delta, delta, rel_tol=1e-12)
            ),
            "reference_trust/resume_radius_before": previous_delta,
            "reference_trust/resume_radius_after": delta,
            "reference_trust/radius_mode_round_decay": 1.0,
            "reference_trust/round_decay/step": float(step),
        }
    if cfg.trust_radius_mode == "fixed":
        previous_delta = float(state.trust_current_delta)
        configured_delta = float(cfg.trust_delta_max)
        applied = bool(
            not math.isfinite(previous_delta)
            or not math.isclose(
                previous_delta,
                configured_delta,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        )
        state.trust_current_delta = configured_delta
        return {
            "reference_trust/resume_radius_clamp_enabled": 1.0,
            "reference_trust/resume_radius_clamp_applied": float(applied),
            "reference_trust/resume_radius_before": previous_delta,
            "reference_trust/resume_radius_after": configured_delta,
        }
    if not (
        cfg.trust_empirical_radius_enabled
        and not cfg.trust_empirical_allow_expansion
    ):
        return {}
    if not (
        math.isfinite(state.trust_current_raw_auc)
        and 0.0 <= state.trust_current_raw_auc <= 1.0
        and math.isfinite(state.trust_current_raw_auc_se)
        and state.trust_current_raw_auc_se >= 0.0
    ):
        return {
            "reference_trust/resume_radius_clamp_enabled": 1.0,
            "reference_trust/resume_radius_clamp_applied": 0.0,
        }
    previous_delta = float(state.trust_current_delta)
    preview = install_adaptive_trust_round(
        state,
        cfg=cfg,
        raw_auc=float(state.trust_current_raw_auc),
        raw_auc_se=float(state.trust_current_raw_auc_se),
        commit=False,
    )
    auc_scaled_delta = float(preview["reference_trust/delta"])
    next_delta = (
        min(previous_delta, auc_scaled_delta)
        if math.isfinite(previous_delta) and previous_delta > 0.0
        else auc_scaled_delta
    )
    applied = bool(
        not math.isfinite(previous_delta)
        or not math.isclose(next_delta, previous_delta, rel_tol=0.0, abs_tol=1.0e-15)
    )
    state.trust_current_delta = next_delta
    if applied:
        state.trust_empirical_safe_streak = 0
    return {
        "reference_trust/resume_radius_clamp_enabled": 1.0,
        "reference_trust/resume_radius_clamp_applied": float(applied),
        "reference_trust/resume_radius_before": previous_delta,
        "reference_trust/resume_radius_auc_scaled": auc_scaled_delta,
        "reference_trust/resume_radius_after": next_delta,
    }


def record_reference_trust_distance(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    distance: float,
) -> dict[str, float]:
    """Record real accepted-policy drift for empirical radius calibration.

    The samples come from the same fixed-shared-noise probe used by strict
    post-step backtracking. Only a short trailing window is checkpointed.
    """

    if not (
        cfg.trust_boundary_enabled and cfg.trust_empirical_radius_enabled
    ):
        return {}
    value = float(distance)
    if not math.isfinite(value) or value < 0.0:
        return {
            "reference_trust/empirical/distance_sample_valid": 0.0,
            "reference_trust/empirical/distance_window_count": float(
                len(state.trust_distance_window)
            ),
        }
    state.trust_distance_window.append(value)
    keep = int(cfg.trust_empirical_distance_window_steps)
    if len(state.trust_distance_window) > keep:
        del state.trust_distance_window[:-keep]
    return {
        "reference_trust/empirical/distance_sample_valid": 1.0,
        "reference_trust/empirical/distance_latest": value,
        "reference_trust/empirical/distance_window_count": float(
            len(state.trust_distance_window)
        ),
    }


def record_reference_trust_attempt(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    accepted: bool,
    update_scale: float,
    distance: float,
) -> dict[str, float]:
    """Record optimizer throughput and accepted drift for radius control."""

    if not (
        cfg.trust_boundary_enabled and cfg.trust_empirical_radius_enabled
    ):
        return {}
    accepted_value = float(bool(accepted))
    scale = float(update_scale)
    if not math.isfinite(scale):
        scale = 0.0
    scale = min(max(scale, 0.0), 1.0)
    state.trust_step_acceptance_window.append(accepted_value)
    state.trust_step_scale_window.append(scale)
    keep = int(cfg.trust_empirical_attempt_window_steps)
    if len(state.trust_step_acceptance_window) > keep:
        del state.trust_step_acceptance_window[:-keep]
    if len(state.trust_step_scale_window) > keep:
        del state.trust_step_scale_window[:-keep]
    diagnostics = {
        "reference_trust/empirical/attempt_window_count": float(
            len(state.trust_step_acceptance_window)
        ),
        "reference_trust/empirical/acceptance_rate": float(
            sum(state.trust_step_acceptance_window)
            / max(len(state.trust_step_acceptance_window), 1)
        ),
        "reference_trust/empirical/mean_update_scale": float(
            sum(state.trust_step_scale_window)
            / max(len(state.trust_step_scale_window), 1)
        ),
    }
    if accepted:
        diagnostics.update(
            record_reference_trust_distance(
                state,
                cfg=cfg,
                distance=distance,
            )
        )
    return diagnostics


def update_empirical_trust_radius_from_audit(
    state: AdaptiveOmniFoldState,
    probe: Mapping[str, float],
    *,
    cfg: AdaptiveOmniFoldConfig,
) -> dict[str, float]:
    """Adapt the trust radius from fresh audits and real optimizer throughput.

    Legacy mode is one-sided: a statistically unsafe audit supplies a cap for
    later reward rounds. Opt-in bidirectional mode retains that conservative
    contraction and can also expand the current radius after repeated point-safe
    audits when recent optimizer acceptance or effective update scale is starved.
    """

    if not (
        cfg.trust_boundary_enabled and cfg.trust_empirical_radius_enabled
    ):
        return {}
    samples = [
        float(value)
        for value in state.trust_distance_window
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]
    count = len(samples)
    diagnostics = {
        "reference_trust/empirical/enabled": 1.0,
        "reference_trust/empirical/distance_window_count": float(count),
        "reference_trust/empirical/min_distance_samples": float(
            cfg.trust_empirical_min_distance_samples
        ),
        "reference_trust/empirical/safety_factor": float(
            cfg.trust_empirical_safety_factor
        ),
    }
    if count < int(cfg.trust_empirical_min_distance_samples):
        diagnostics["reference_trust/empirical/status"] = 0.0
        diagnostics["reference_trust/empirical/updated"] = 0.0
        return diagnostics

    ordered = sorted(samples)
    midpoint = count // 2
    median = (
        ordered[midpoint]
        if count % 2
        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    )
    deviations = sorted(abs(value - median) for value in samples)
    mad = (
        deviations[midpoint]
        if count % 2
        else 0.5 * (deviations[midpoint - 1] + deviations[midpoint])
    )
    distance_se = 1.4826 * mad / math.sqrt(float(count))
    z_value = float(cfg.trust_empirical_confidence_z)
    distance_lcb = max(0.0, median - z_value * distance_se)
    distance_ucb = median + z_value * distance_se

    auc_gap = float(probe.get("weighted_auc_gap", float("nan")))
    auc_se = float(probe.get("audit_auc_null_se_approx", float("nan")))
    threshold = float(state.trigger_threshold)
    if not (
        math.isfinite(auc_gap)
        and auc_gap >= 0.0
        and math.isfinite(auc_se)
        and auc_se >= 0.0
        and math.isfinite(threshold)
        and threshold > 0.0
    ):
        diagnostics.update(
            {
                "reference_trust/empirical/status": 0.0,
                "reference_trust/empirical/updated": 0.0,
                "reference_trust/empirical/distance_median": median,
                "reference_trust/empirical/distance_se": distance_se,
            }
        )
        return diagnostics

    auc_lcb = max(0.0, auc_gap - z_value * auc_se)
    auc_ucb = auc_gap + z_value * auc_se
    audit_saturated = bool(float(probe.get("audit_saturated", 1.0)) >= 0.5)
    unsafe = bool(audit_saturated and auc_lcb > threshold)
    statistically_safe = bool(audit_saturated and auc_ucb <= threshold)
    point_safe = bool(audit_saturated and auc_gap < threshold)
    safe = bool(
        point_safe
        if cfg.trust_empirical_bidirectional
        else statistically_safe
    )
    state.trust_empirical_observations += 1
    updated = False
    proposed_cap = float("nan")
    radius_action = 0.0
    previous_delta = float(state.trust_current_delta)
    next_delta = previous_delta
    attempt_count = len(state.trust_step_acceptance_window)
    acceptance_rate = (
        sum(state.trust_step_acceptance_window) / float(attempt_count)
        if attempt_count > 0
        else float("nan")
    )
    mean_update_scale = (
        sum(state.trust_step_scale_window)
        / float(len(state.trust_step_scale_window))
        if state.trust_step_scale_window
        else float("nan")
    )
    throughput_starved = bool(
        attempt_count >= min(5, int(cfg.trust_empirical_attempt_window_steps))
        and (
            acceptance_rate < cfg.trust_empirical_target_acceptance_rate
            or mean_update_scale < cfg.trust_empirical_target_update_scale
        )
    )
    if unsafe and distance_lcb > 0.0:
        proposed_cap = max(
            float(cfg.trust_delta_floor),
            min(
                float(cfg.trust_delta_max),
                float(cfg.trust_empirical_safety_factor) * distance_lcb,
                (
                    previous_delta * float(cfg.trust_empirical_shrink_factor)
                    if cfg.trust_empirical_bidirectional
                    and math.isfinite(previous_delta)
                    and previous_delta > 0.0
                    else float(cfg.trust_delta_max)
                ),
            ),
        )
        previous_cap = float(state.trust_empirical_delta_cap)
        if not math.isfinite(previous_cap) or proposed_cap < previous_cap:
            state.trust_empirical_delta_cap = proposed_cap
            state.trust_empirical_updates += 1
            if cfg.trust_empirical_bidirectional:
                state.trust_empirical_contractions += 1
                radius_action = -1.0
            updated = True
        unsafe_distance = float(state.trust_empirical_unsafe_distance)
        if not math.isfinite(unsafe_distance) or distance_lcb < unsafe_distance:
            state.trust_empirical_unsafe_distance = distance_lcb
        state.trust_empirical_safe_streak = 0
    elif safe:
        safe_distance = float(state.trust_empirical_safe_distance)
        if not math.isfinite(safe_distance) or distance_ucb > safe_distance:
            state.trust_empirical_safe_distance = distance_ucb
        if (
            cfg.trust_empirical_bidirectional
            and cfg.trust_empirical_allow_expansion
        ):
            state.trust_empirical_safe_streak += 1
            if (
                state.trust_empirical_safe_streak
                >= int(cfg.trust_empirical_safe_audits_required)
                and throughput_starved
                and math.isfinite(previous_delta)
                and previous_delta > 0.0
            ):
                expansion_ceiling = float(cfg.trust_delta_max)
                learned_cap = float(state.trust_empirical_delta_cap)
                if math.isfinite(learned_cap) and learned_cap > 0.0:
                    expansion_ceiling = min(expansion_ceiling, learned_cap)
                next_delta = min(
                    expansion_ceiling,
                    max(
                        float(cfg.trust_delta_floor),
                        previous_delta * float(cfg.trust_empirical_expand_factor),
                    ),
                )
                if next_delta > previous_delta:
                    state.trust_current_delta = next_delta
                    state.trust_empirical_updates += 1
                    state.trust_empirical_expansions += 1
                    updated = True
                    radius_action = 1.0
                state.trust_empirical_safe_streak = 0
        else:
            state.trust_empirical_safe_streak = 0
    else:
        state.trust_empirical_safe_streak = 0

    diagnostics.update(
        {
            # status: -1=safe, 0=ambiguous/unavailable, +1=unsafe.
            "reference_trust/empirical/status": float(1 if unsafe else -1 if safe else 0),
            "reference_trust/empirical/updated": float(updated),
            "reference_trust/empirical/bidirectional": float(
                cfg.trust_empirical_bidirectional
            ),
            "reference_trust/empirical/allow_expansion": float(
                cfg.trust_empirical_allow_expansion
            ),
            # -1=contract future-round cap, 0=hold, +1=expand current radius.
            "reference_trust/empirical/radius_action": radius_action,
            "reference_trust/empirical/point_safe": float(point_safe),
            "reference_trust/empirical/audit_saturated": float(
                audit_saturated
            ),
            "reference_trust/empirical/statistically_safe": float(
                statistically_safe
            ),
            "reference_trust/empirical/safe_streak": float(
                state.trust_empirical_safe_streak
            ),
            "reference_trust/empirical/throughput_starved": float(
                throughput_starved
            ),
            "reference_trust/empirical/attempt_window_count": float(
                attempt_count
            ),
            "reference_trust/empirical/acceptance_rate": acceptance_rate,
            "reference_trust/empirical/mean_update_scale": mean_update_scale,
            "reference_trust/empirical/delta_before": previous_delta,
            "reference_trust/empirical/delta_after": float(
                state.trust_current_delta
            ),
            "reference_trust/empirical/distance_median": median,
            "reference_trust/empirical/distance_se": distance_se,
            "reference_trust/empirical/distance_lcb": distance_lcb,
            "reference_trust/empirical/distance_ucb": distance_ucb,
            "reference_trust/empirical/audit_auc_gap": auc_gap,
            "reference_trust/empirical/audit_auc_se": auc_se,
            "reference_trust/empirical/audit_auc_gap_lcb": auc_lcb,
            "reference_trust/empirical/audit_auc_gap_ucb": auc_ucb,
            "reference_trust/empirical/audit_threshold": threshold,
            "reference_trust/empirical/proposed_delta_cap": proposed_cap,
            "reference_trust/empirical/delta_cap": float(
                state.trust_empirical_delta_cap
            ),
            "reference_trust/empirical/safe_distance": float(
                state.trust_empirical_safe_distance
            ),
            "reference_trust/empirical/unsafe_distance": float(
                state.trust_empirical_unsafe_distance
            ),
            "reference_trust/empirical/observations": float(
                state.trust_empirical_observations
            ),
            "reference_trust/empirical/updates": float(
                state.trust_empirical_updates
            ),
            "reference_trust/empirical/expansions": float(
                state.trust_empirical_expansions
            ),
            "reference_trust/empirical/contractions": float(
                state.trust_empirical_contractions
            ),
        }
    )
    return diagnostics


def trust_region_exhausted(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
) -> tuple[bool, dict[str, float]]:
    """Detect a policy that has consumed its useful fixed-round update budget.

    Distance alone is not sufficient: a healthy update may transiently sit near
    the interior boundary. Exhaustion requires both a near-boundary accepted
    policy and a trailing AdamW scale window showing that backtracking is now
    suppressing nearly the whole proposed update.
    """

    enabled = bool(
        cfg.trust_boundary_enabled
        and cfg.trust_empirical_radius_enabled
        and cfg.trust_refit_on_exhaustion
    )
    distances = [
        float(value)
        for value in state.trust_distance_window
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]
    all_scales = [
        float(value)
        for value in state.trust_step_scale_window
        if math.isfinite(float(value))
    ]
    scale_window_steps = int(cfg.trust_exhaustion_scale_window_steps)
    scales = all_scales[-scale_window_steps:]
    delta = float(state.trust_current_delta)
    interior_limit = (
        float(cfg.trust_interior_fraction) * delta
        if math.isfinite(delta) and delta > 0.0
        else float("nan")
    )
    latest_distance = distances[-1] if distances else float("nan")
    distance_fraction = (
        latest_distance / interior_limit
        if math.isfinite(latest_distance)
        and math.isfinite(interior_limit)
        and interior_limit > 0.0
        else float("nan")
    )
    mean_update_scale = (
        sum(scales) / float(len(scales)) if scales else float("nan")
    )
    min_attempts = min(5, scale_window_steps)
    enough_attempts = len(scales) >= min_attempts
    exhausted = bool(
        enabled
        and enough_attempts
        and math.isfinite(distance_fraction)
        and distance_fraction >= cfg.trust_exhaustion_distance_fraction
        and math.isfinite(mean_update_scale)
        and mean_update_scale < cfg.trust_exhaustion_mean_update_scale
    )
    diagnostics = {
        "reference_trust/exhaustion/enabled": float(enabled),
        "reference_trust/exhaustion/triggered": float(exhausted),
        "reference_trust/exhaustion/latest_distance": latest_distance,
        "reference_trust/exhaustion/interior_limit": interior_limit,
        "reference_trust/exhaustion/distance_fraction": distance_fraction,
        "reference_trust/exhaustion/distance_fraction_threshold": float(
            cfg.trust_exhaustion_distance_fraction
        ),
        "reference_trust/exhaustion/mean_update_scale": mean_update_scale,
        "reference_trust/exhaustion/mean_update_scale_threshold": float(
            cfg.trust_exhaustion_mean_update_scale
        ),
        "reference_trust/exhaustion/attempt_count": float(len(scales)),
        "reference_trust/exhaustion/available_attempt_count": float(
            len(all_scales)
        ),
        "reference_trust/exhaustion/scale_window_steps": float(
            scale_window_steps
        ),
        "reference_trust/exhaustion/min_attempts": float(min_attempts),
    }
    return exhausted, diagnostics


def adaptive_audit_protocol_signature(cfg: AdaptiveOmniFoldConfig) -> str:
    """Fingerprint the audit protocol that certified an installed baseline."""

    payload = {
        "schema": "cheap-weighted-trigger-baseline-v6",
        "probe_max_events": cfg.probe_max_events,
        "require_audit_saturation": cfg.require_audit_saturation,
        "score_row_budget": cfg.score_row_budget,
        "fixed_common_random_panel": bool(
            cfg.fixed_audit_panel or cfg.trust_trajectory_search_enabled
        ),
        "audit_fit": cfg.audit_fit,
        "raw_monitor_warm_start": cfg.raw_monitor_warm_start,
        "single_pool_train_validation": cfg.single_pool_train_validation,
        "single_pool_split_seed": cfg.single_pool_split_seed,
    }
    if cfg.cache_event_inputs:
        payload["fixed_input_cache"] = "cpu-shard-v1"
    if cfg.audit_fit.get("training_population") == "omnifold_fold":
        payload["training_crossfit"] = {"seed": cfg.seed, "folds": cfg.crossfit_folds, "repeat": 1}
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


def should_probe_training_boundary(
    state: AdaptiveOmniFoldState, *, cfg: AdaptiveOmniFoldConfig,
    epoch: int, global_step: int, epoch_end: bool,
) -> bool:
    """Step cadence takes precedence; never repeat a completed monitor on resume."""
    if cfg.staleness_every_n_steps is None:
        return bool(epoch_end and should_probe_epoch(epoch, cfg.staleness_every_n_epochs))
    return bool(
        global_step > 0
        and global_step % cfg.staleness_every_n_steps == 0
        and global_step > state.last_staleness_step
    )


def should_run_raw_only_monitor(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    baseline_only: bool = False,
    force_refit: bool = False,
) -> bool:
    """Select stationary raw-AUC monitoring without bypassing certification."""

    return bool(
        cfg.monitor_mode == "raw_only"
        and state.calibrated
        and not baseline_only
        and not force_refit
    )


def should_skip_incumbent_probe(
    *,
    cfg: AdaptiveOmniFoldConfig,
    force_refit: bool = False,
) -> bool:
    """Skip the weighted audit of a reward a stationary refit will replace."""

    return bool(force_refit and cfg.monitor_mode == "raw_only")


def reward_refit_due_to_age(
    state: AdaptiveOmniFoldState,
    *,
    epoch: int,
    max_reward_age_epochs: int | None,
) -> tuple[bool, int]:
    """Return whether the installed reward reached its configured maximum age.

    This is evaluated only at the normal audit cadence. The candidate still
    has to reach the configured cross-fit residual closure before installation.
    """

    # A baseline-only protocol refresh also calls ``state.install`` but does
    # not replace the reward. Prefer the accepted-refit timestamp so such a
    # refresh cannot make an old reward appear young again.
    age_anchor = (
        state.last_recalibration_epoch
        if state.last_recalibration_epoch is not None
        else state.installed_at_epoch
    )
    reward_age = max(0, int(epoch) - int(age_anchor))
    if max_reward_age_epochs is None:
        return False, reward_age
    limit = int(max_reward_age_epochs)
    if limit < 1:
        raise ValueError("max_reward_age_epochs must be positive or null")
    return reward_age >= limit, reward_age


def reward_round_budget_exhausted(
    state: "AdaptiveOmniFoldState", *, max_reward_rounds: int | None
) -> bool:
    """Return whether bootstrap/refits already installed the allowed rounds."""

    if max_reward_rounds is None:
        return False
    limit = int(max_reward_rounds)
    if limit < 1:
        raise ValueError("max_reward_rounds must be positive or null")
    return int(state.reward_round_id) >= limit


def scheduled_raw_refit_due(
    state: AdaptiveOmniFoldState,
    *,
    cfg: AdaptiveOmniFoldConfig,
    epoch: int,
    baseline_only: bool = False,
) -> tuple[bool, dict[str, float]]:
    """Apply the reward-age limit independently of raw-AUC improvement.

    Called at the configured monitor cadence. With epoch-only monitoring and
    max_reward_age_epochs=1, bootstrap at -1 is followed by a refit at the end
    of epoch 0, then each subsequent epoch. Baseline refreshes never refit.
    """
    due, age = reward_refit_due_to_age(
        state, epoch=epoch, max_reward_age_epochs=cfg.max_reward_age_epochs,
    )
    cooldown_anchor = (
        state.last_recalibration_attempt_epoch
        if state.last_recalibration_attempt_epoch is not None
        else state.last_recalibration_epoch
    )
    cooldown = bool(
        cooldown_anchor is not None
        and int(epoch) - int(cooldown_anchor) < cfg.retrain_cooldown_epochs
    )
    trigger = bool(due and epoch >= 0 and not baseline_only and not cfg.log_only and not cooldown)
    return trigger, {
        "staleness/reward_age_epochs": float(age),
        "staleness/max_reward_age_epochs": float(cfg.max_reward_age_epochs or 0),
        "staleness/age_refit_due": float(due),
        "staleness/age_trigger_recalibration": float(trigger),
        "staleness/age_refit_cooldown_active": float(cooldown),
        "staleness/last_refit_attempt_epoch": float(
            -1 if state.last_recalibration_attempt_epoch is None
            else state.last_recalibration_attempt_epoch
        ),
    }


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
    cooldown_anchor = (
        state.last_recalibration_attempt_epoch
        if state.last_recalibration_attempt_epoch is not None
        else state.last_recalibration_epoch
    )
    epochs_since_recalibration = (
        None
        if cooldown_anchor is None
        else max(0, int(epoch) - int(cooldown_anchor))
    )
    cooldown_active = bool(
        cfg.retrain_cooldown_epochs > 0
        and epochs_since_recalibration is not None
        and epochs_since_recalibration < cfg.retrain_cooldown_epochs
    )
    # Keep probing during cooldown for observability, but do not carry stale
    # evidence across the protected interval.  A post-cooldown refit therefore
    # needs the configured number of fresh consecutive confirmations.
    state.probe_exceedance_streak = (
        0
        if cooldown_active
        else state.probe_exceedance_streak + 1
        if exceeded
        else 0
    )
    fired = bool(
        not cooldown_active
        and state.probe_exceedance_streak >= cfg.required_consecutive_epochs
    )
    recalibrate = bool(fired and not cfg.log_only)
    decision = (
        "cooldown"
        if cooldown_active
        else "audit_unsaturated"
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
        "staleness/retrain_cooldown_epochs": float(
            cfg.retrain_cooldown_epochs
        ),
        "staleness/cooldown_active": float(cooldown_active),
        "staleness/epochs_since_recalibration": float(
            epochs_since_recalibration
            if epochs_since_recalibration is not None
            else -1
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


def initialize_global_raw_best(state: AdaptiveOmniFoldState) -> None:
    """Migrate legacy round-best state using all recorded, saturated raw checks.

    No tensor/checkpoint IO here: the trainer resolves and verifies the exact
    selected snapshot before allowing another update. Never select a runner-up.
    """
    if state.raw_global_initialized:
        return
    from RL.DGPO_neutrino.model_utils import _completed_raw_monitor_records
    records = _completed_raw_monitor_records({"probe_history": state.probe_history})
    if records:
        gap, epoch, step, next_epoch = min(records)
        if math.isfinite(state.trust_best_global_auc_gap) and gap > state.trust_best_global_auc_gap + 1e-10:
            raise ValueError("global-best raw history is truncated or missing the inherited global record")
        recorded = state.raw_best_checkpoint if (state.raw_best_epoch, state.raw_best_global_step) == (epoch, step) else ""
        state.raw_best_auc_gap, state.raw_best_epoch = gap, epoch
        state.raw_best_global_step, state.raw_best_next_epoch = step, next_epoch
        state.raw_best_checkpoint = recorded
    elif math.isfinite(state.raw_best_auc_gap) or math.isfinite(state.trust_best_global_auc_gap):
        raise ValueError("cannot migrate global best without saturated raw history")
    state.raw_global_initialized = True
    # Old streaks used a different, round-local target; start a new five-check window.
    state.raw_no_improvement_streak = 0
    state.raw_global_failed_rounds = 0
    state.raw_global_refit_pending = False
    state.raw_global_stop_requested = False


def global_raw_candidate_due(state: AdaptiveOmniFoldState, probe: Mapping[str, float],
                             *, cfg: AdaptiveOmniFoldConfig) -> bool:
    gap = float(probe.get("raw_auc_gap", float("nan")))
    return bool(cfg.raw_best_scope == "global" and cfg.raw_global_confirm_candidates
                and math.isfinite(state.raw_best_auc_gap)
                and float(probe.get("raw_audit_saturated", 0.)) >= .5
                and (cfg.audit_fit.get("training_readiness") is None
                     or float(probe.get("raw_audit_training_ready", 0.)) >= .5)
                and math.isfinite(gap) and 0 <= gap <= .5
                and gap < state.raw_best_auc_gap - cfg.raw_improvement_min_delta)


def raw_staleness_patience(cfg: AdaptiveOmniFoldConfig, *, global_step: int) -> int:
    """Patience follows the persisted training clock, not the rollback point/round.

    No extra scheduler state or classifier cache invalidation is necessary.
    Without a schedule, preserve the legacy fixed-patience behavior exactly.
    """
    if not cfg.raw_patience_schedule:
        return int(cfg.required_consecutive_epochs)
    if type(global_step) is not int or global_step < 0:
        raise ValueError("scheduled raw patience requires a nonnegative integer global_step")
    patience = cfg.raw_patience_schedule[0][1]
    for start, checks in cfg.raw_patience_schedule:
        if global_step < start:
            break
        patience = checks
    return patience


def update_raw_plateau_controller(
    state: AdaptiveOmniFoldState,
    probe: Mapping[str, float],
    *,
    cfg: AdaptiveOmniFoldConfig,
    epoch: int,
    global_step: int = -1,
    checkpoint_path: str = "",
    checkpoint_next_epoch: int | None = None,
    global_confirmation: bool | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Trigger a refit after saturated raw AUC stops improving.

    The lower-is-better statistic is ``abs(AUC - 0.5)`` from a fresh
    truth-vs-current-policy classifier.  Improvement is measured against the
    best observation in the configured scope (round by default, or global).
    """

    if cfg.monitor_mode != "raw_plateau_refit":
        raise ValueError(
            "raw plateau controller requires monitor_mode=raw_plateau_refit"
        )
    patience = raw_staleness_patience(cfg, global_step=global_step)
    if cfg.raw_best_scope == "global" and not state.raw_global_initialized:
        initialize_global_raw_best(state)
    gap = float(probe.get("raw_auc_gap", float("nan")))
    saturated = float(probe.get("raw_audit_saturated", 0.0)) >= 0.5
    training_ready = bool(cfg.audit_fit.get("training_readiness") is None
                          or float(probe.get("raw_audit_training_ready", 0.)) >= .5)
    eligible = bool((saturated or not cfg.require_audit_saturation) and training_ready)
    previous = float(state.raw_previous_auc_gap)
    best_before = float(state.raw_best_auc_gap)
    finite = bool(math.isfinite(gap) and 0.0 <= gap <= 0.5)
    warmup_interval = bool(
        cfg.raw_pause_patience_during_warmup and cfg.policy_warmup_steps > 0
        and state.policy_warmup_round_id == state.reward_round_id
        and (state.policy_warmup_completed_updates < cfg.policy_warmup_steps
             or state.raw_patience_warmup_updates_seen < cfg.policy_warmup_steps)
    )
    if global_raw_candidate_due(state, probe, cfg=cfg) and global_confirmation is None:
        eligible = False  # Inconclusive confirmation is not a valid failed check.
    improved = bool(
        eligible
        and finite
        and (
            not math.isfinite(best_before)
            or gap < best_before - float(cfg.raw_improvement_min_delta)
        )
    )
    if (improved and cfg.raw_best_scope == "global" and cfg.raw_global_confirm_candidates
            and math.isfinite(best_before)):
        improved = global_confirmation is True
    if cfg.raw_best_scope == "global" and global_step <= state.last_staleness_step:
        raise ValueError("global raw controller requires a strictly newer monitor step")
    if not eligible:
        decision = "raw_monitor_unsaturated" if training_ready else "raw_monitor_training_unready"
        if cfg.staleness_every_n_steps is not None:
            state.raw_no_improvement_streak = 0
    elif not finite and cfg.staleness_every_n_steps is not None:
        decision = "raw_monitor_nonfinite"
        state.raw_no_improvement_streak = 0
    elif improved:
        state.raw_best_auc_gap = gap
        state.raw_best_epoch = int(epoch)
        state.raw_best_global_step = int(global_step)
        state.raw_best_checkpoint = str(checkpoint_path)
        state.raw_best_next_epoch = int(epoch + 1 if checkpoint_next_epoch is None else checkpoint_next_epoch)
        state.raw_no_improvement_streak = 0
        if cfg.raw_best_scope == "global":
            state.raw_global_failed_rounds = 0
            state.raw_global_refit_pending = False
        decision = "raw_improved"
    elif warmup_interval:
        state.raw_no_improvement_streak = 0
        decision = "raw_patience_warmup"
    else:
        state.raw_no_improvement_streak += 1
        decision = "raw_no_improvement"
    if finite and training_ready:
        state.raw_previous_auc_gap = gap
    fired = bool(
        eligible
        and (finite or cfg.staleness_every_n_steps is None)
        and not improved
        and not warmup_interval
        and state.raw_no_improvement_streak >= patience
    )
    if fired:
        decision = "raw_plateau_recalibrate"
        if cfg.raw_best_scope == "global" and state.raw_global_refit_pending:
            state.raw_global_failed_rounds += 1
            state.raw_global_refit_pending = False
            state.raw_global_stop_requested = (
                cfg.raw_global_max_failed_rounds > 0
                and state.raw_global_failed_rounds >= cfg.raw_global_max_failed_rounds
            )
            if state.raw_global_stop_requested:
                fired = False
                decision = "raw_global_stagnation_stop"
    state.last_decision = decision
    state.raw_patience_warmup_updates_seen = state.policy_warmup_completed_updates
    state.last_staleness_step = int(global_step)
    row = {
        "epoch": float(epoch),
        "global_step": float(global_step),
        "checkpoint_next_epoch": float(epoch + 1 if checkpoint_next_epoch is None else checkpoint_next_epoch),
        "reward_round_id": float(state.reward_round_id),
        "decision_recalibrate": float(fired),
        **{key: float(value) for key, value in probe.items()},
        "raw_no_improvement_patience": float(patience),
    }
    if not training_ready:
        # History readers used during resume also require saturation. Do not let
        # an unqualified numerical plateau re-enter through best-history recovery.
        row["raw_audit_saturated"] = 0.0
    state.probe_history.append(row)
    history_limit = 4096 if cfg.staleness_every_n_steps is not None else 256
    if len(state.probe_history) > history_limit:
        del state.probe_history[:-history_limit]
    return fired, {
        **{f"staleness/{key}": value for key, value in row.items()},
        "staleness/decision": decision,
        "staleness/trigger_recalibration": float(fired),
        "staleness/trigger_reason": "raw_plateau" if fired else "none",
        "staleness/global_best/enabled": float(cfg.raw_best_scope == "global"),
        "staleness/global_best/improved": float(improved and cfg.raw_best_scope == "global"),
        "staleness/global_best/failed_rounds": float(state.raw_global_failed_rounds),
        "staleness/patience_paused_for_warmup": float(warmup_interval),
        "staleness/global_best/stop_requested": float(state.raw_global_stop_requested),
        "staleness/raw_previous_auc_gap": previous,
        "staleness/raw_best_auc_gap_before": best_before,
        "staleness/raw_best_auc_gap": float(state.raw_best_auc_gap),
        "staleness/raw_best_epoch": float(state.raw_best_epoch),
        "staleness/raw_best_global_step": float(
            state.raw_best_global_step
        ),
        "staleness/raw_best_checkpoint_recorded": float(
            bool(state.raw_best_checkpoint)
        ),
        "staleness/rollback_to_best_on_plateau": float(
            cfg.raw_rollback_to_best_on_plateau
        ),
        "staleness/raw_plateau_rollbacks": float(
            state.raw_plateau_rollbacks
        ),
        "staleness/raw_improved": float(improved),
        "staleness/raw_improvement_min_delta": float(
            cfg.raw_improvement_min_delta
        ),
        "staleness/raw_no_improvement_streak": float(
            state.raw_no_improvement_streak
        ),
        "staleness/raw_no_improvement_patience": float(
            patience
        ),
        "staleness/every_n_steps": float(cfg.staleness_every_n_steps or 0),
        "staleness/raw_decision_eligible": float(eligible),
        "staleness/monitor_mode_raw_plateau_refit": 1.0,
        "staleness/log_only": 0.0,
    }


def update_classifier_trust_controller(
    state: AdaptiveOmniFoldState,
    probe: Mapping[str, float],
    *,
    cfg: AdaptiveOmniFoldConfig,
    epoch: int,
) -> tuple[bool, dict[str, Any]]:
    """Request forward recentering when current/reference separation is large.

    With equal class priors, the Bayes balanced accuracy is
    ``0.5 * (1 + total_variation)``.  The fitted classifier is only a finite
    lower-bound proxy for that discriminator; the normal upper confidence bound
    below handles validation-sample uncertainty, while saturation and a strong
    shared architecture address classifier optimization error.
    """

    if not cfg.classifier_trust_enabled:
        state.classifier_trust_exceedance_streak = 0
        return False, {}
    balanced_accuracy = float(
        probe.get("reference_trust_balanced_accuracy", float("nan"))
    )
    audit_events = int(probe.get("reference_trust_test_events", 0.0))
    saturated = float(
        probe.get("reference_trust_saturated", 0.0)
    ) >= 0.5
    unsafe_early_stop = float(
        probe.get(
            "reference_trust_unsafe_balanced_accuracy_lcb_reached",
            0.0,
        )
    ) >= 0.5
    oriented = (
        max(balanced_accuracy, 1.0 - balanced_accuracy)
        if math.isfinite(balanced_accuracy)
        else float("nan")
    )
    # Worst-case binomial standard error for balanced accuracy with the same
    # held-out event count in each class.
    standard_error = (
        math.sqrt(0.125 / float(audit_events))
        if audit_events > 0
        else float("inf")
    )
    upper = min(
        1.0,
        oriented + float(cfg.classifier_trust_confidence_z) * standard_error,
    )
    lower = max(
        0.5,
        oriented - float(cfg.classifier_trust_confidence_z) * standard_error,
    )
    eligible = bool(
        math.isfinite(oriented)
        and (
            saturated
            or unsafe_early_stop
            or not cfg.classifier_trust_require_saturation
        )
    )
    exceeded = bool(
        unsafe_early_stop
        or (
            eligible
            and upper > float(cfg.classifier_trust_max_balanced_accuracy)
        )
    )
    state.classifier_trust_exceedance_streak = (
        state.classifier_trust_exceedance_streak + 1 if exceeded else 0
    )
    fired = bool(
        state.classifier_trust_exceedance_streak
        >= cfg.classifier_trust_required_consecutive_epochs
    )
    return fired, {
        "classifier_trust/epoch": float(epoch),
        "classifier_trust/reward_round_id": float(state.reward_round_id),
        "classifier_trust/balanced_accuracy": balanced_accuracy,
        "classifier_trust/oriented_balanced_accuracy": oriented,
        "classifier_trust/balanced_accuracy_se_approx": standard_error,
        "classifier_trust/balanced_accuracy_upper": upper,
        "classifier_trust/balanced_accuracy_lower": lower,
        "classifier_trust/max_balanced_accuracy": float(
            cfg.classifier_trust_max_balanced_accuracy
        ),
        "classifier_trust/estimated_total_variation": (
            max(0.0, 2.0 * oriented - 1.0)
            if math.isfinite(oriented)
            else float("nan")
        ),
        "classifier_trust/saturated": float(saturated),
        "classifier_trust/unsafe_early_stop": float(unsafe_early_stop),
        "classifier_trust/decision_eligible": float(eligible),
        "classifier_trust/exceeded": float(exceeded),
        "classifier_trust/exceedance_streak": float(
            state.classifier_trust_exceedance_streak
        ),
        "classifier_trust/required_consecutive_epochs": float(
            cfg.classifier_trust_required_consecutive_epochs
        ),
        "classifier_trust/trigger_recalibration": float(fired),
    }


@dataclass(frozen=True)
class AdaptiveOmniFoldPool:
    packed_event: Tensor
    truth: Tensor
    candidates: Tensor
    packing_spec: EventPackingSpec
    # Optional policy-only sidecar. Never part of classifier inputs or identity hashing.
    policy_noise_mask: Tensor | None = None

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
        if self.policy_noise_mask is not None and tuple(self.policy_noise_mask.shape) != (n_events, 2):
            raise ValueError("policy noise mask must be (N,2)")

    @property
    def n_events(self) -> int:
        return int(self.packed_event.shape[0])

    @property
    def identity_inputs(self) -> Tensor:
        return event_identity_inputs(self.packed_event, self.packing_spec)

    @property
    def identity_override(self) -> Tensor | None:
        return self.identity_inputs if REST_FRAME_KEY in self.packing_spec.shapes else None

    def to(self, device: torch.device) -> "AdaptiveOmniFoldPool":
        return AdaptiveOmniFoldPool(
            packed_event=self.packed_event.to(device=device, dtype=torch.float32),
            truth=self.truth.to(device=device, dtype=torch.float32),
            candidates=self.candidates.to(device=device, dtype=torch.float32),
            packing_spec=self.packing_spec,
            policy_noise_mask=None if self.policy_noise_mask is None else self.policy_noise_mask.to(device),
        )

    def prefix(self, max_events: int | None) -> "AdaptiveOmniFoldPool":
        """Return a deterministic prefix without copying the underlying tensors."""

        if max_events is None or int(max_events) >= self.n_events:
            return self
        count = max(1, int(max_events))
        return AdaptiveOmniFoldPool(
            packed_event=self.packed_event[:count],
            truth=self.truth[:count],
            candidates=self.candidates[:count],
            packing_spec=self.packing_spec,
            policy_noise_mask=None if self.policy_noise_mask is None else self.policy_noise_mask[:count],
        )

    def select(self, indices: Tensor) -> "AdaptiveOmniFoldPool":
        return AdaptiveOmniFoldPool(
            packed_event=self.packed_event.index_select(0, indices.to(self.packed_event.device)),
            truth=self.truth.index_select(0, indices.to(self.truth.device)),
            candidates=self.candidates.index_select(0, indices.to(self.candidates.device)),
            packing_spec=self.packing_spec,
            policy_noise_mask=None if self.policy_noise_mask is None else self.policy_noise_mask.index_select(0, indices.to(self.policy_noise_mask.device)),
        )

    def repack(self, target_spec: EventPackingSpec) -> "AdaptiveOmniFoldPool":
        """Project an enriched pool onto a reward checkpoint's packing contract."""

        if self.packing_spec == target_spec:
            return self
        source = unpack_event_inputs(self.packed_event, self.packing_spec)
        packed, observed = pack_event_inputs(source, target_spec)
        if observed != target_spec:
            raise RuntimeError("adaptive pool repack did not preserve target spec")
        return AdaptiveOmniFoldPool(
            packed_event=packed,
            truth=self.truth,
            candidates=self.candidates,
            packing_spec=target_spec,
            policy_noise_mask=self.policy_noise_mask,
        )


def split_single_classifier_pool(
    pool: AdaptiveOmniFoldPool, *, seed: int,
) -> tuple[AdaptiveOmniFoldPool, AdaptiveOmniFoldPool]:
    """Use every budgeted identity once across stable 80/20 classifier splits."""
    fit_idx, val_idx = _identity_crossfit_splits(pool.identity_inputs, folds=5, seed=seed)[0]
    return pool.select(fit_idx), pool.select(val_idx)


def build_reference_trust_pool(
    current: AdaptiveOmniFoldPool,
    reference: AdaptiveOmniFoldPool,
) -> AdaptiveOmniFoldPool:
    """Pair K=1 current/reference generations for a fresh trust classifier."""

    if current.packing_spec != reference.packing_spec:
        raise ValueError("current/reference trust pools use different packing specs")
    if current.n_events != reference.n_events:
        raise ValueError("current/reference trust pools use different event counts")
    if not torch.equal(current.packed_event.cpu(), reference.packed_event.cpu()):
        raise ValueError("current/reference trust pools use different event identities")
    return AdaptiveOmniFoldPool(
        packed_event=current.packed_event,
        truth=reference.candidates[:, 0, :],
        candidates=current.candidates,
        packing_spec=current.packing_spec,
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
    compatible = pool.repack(stack.packing_spec)
    aligned = compatible if compatible.packed_event.device == device else compatible.to(device)
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


@torch.no_grad()
def evaluate_frozen_checkpoint_auc(
    stack: FrozenResidualRatioReward,
    pool: AdaptiveOmniFoldPool,
    *,
    checkpoint_index: int | None = 0,
    row_budget: int,
) -> float:
    """Evaluate one unchanged reward member on truth versus current generation."""

    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        _weighted_binary_score_metrics,
    )

    index = None if checkpoint_index is None else int(checkpoint_index)
    if index is not None and (index < 0 or index >= stack.num_checkpoints):
        raise IndexError("frozen reward checkpoint index is out of range")
    try:
        device = next(stack.parameters()).device
    except StopIteration:
        device = pool.packed_event.device
    compatible = pool.repack(stack.packing_spec)
    aligned = compatible if compatible.packed_event.device == device else compatible.to(device)
    def score(candidate: Tensor) -> Tensor:
        if index is None:
            return _score_population(stack, aligned.packed_event, candidate, int(row_budget))
        return stack.checkpoint_logits(
            index, aligned.packed_event, candidate, batch_size=int(row_budget)
        )

    truth_score = score(aligned.truth.unsqueeze(1))
    generated_score = score(aligned.candidates)
    if generated_score.shape[1] != 1:
        raise ValueError("fixed classifier AUC requires a K=1 evaluation pool")
    _loss, _balanced_accuracy, auc = _weighted_binary_score_metrics(
        truth_score,
        generated_score,
        torch.ones_like(generated_score),
    )
    return float(auc)


def bootstrap_baseline_pool(
    cfg: AdaptiveOmniFoldConfig, pool: AdaptiveOmniFoldPool,
) -> AdaptiveOmniFoldPool | None:
    """The bootstrap must honor the explicit request to skip baseline fits."""
    if not cfg.baseline_probe_on_start or cfg.monitor_mode == "raw_plateau_refit":
        return None
    return pool.prefix(cfg.probe_max_events)


def step_zero_raw_audit_enabled(cfg: AdaptiveOmniFoldConfig) -> bool:
    return bool(cfg.baseline_probe_on_start and (
        cfg.fixed_schedule_log_raw_audit or cfg.monitor_mode == "raw_only"
    ))


def frozen_classifier_metrics(
    stack: FrozenResidualRatioReward, pool: AdaptiveOmniFoldPool, *, row_budget: int,
) -> dict[str, float]:
    """Read-only, oriented AUC on one common external K=1 event panel."""
    metrics = {"frozen_classifier/events": float(pool.n_events)}
    for label, index in [("ensemble", None)] + [
        (f"fold{i + 1:02d}", i) for i in range(stack.num_checkpoints)
    ]:
        auc = evaluate_frozen_checkpoint_auc(
            stack, pool, checkpoint_index=index, row_budget=row_budget,
        )
        metrics[f"frozen_classifier/{label}/auc"] = auc
        metrics[f"frozen_classifier/{label}/auc_gap"] = abs(auc - 0.5)
    return metrics


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


def _iteration_one_weight_metrics(log_weight: Tensor) -> dict[str, float]:
    """Read-only statistics of the actual tempered, optionally clipped weights.

    Inputs are replicated full OOF/validation vectors, not per-rank shards.
    Float64 softmax avoids overflowing unnormalised exp(log_weight).
    """
    flat = log_weight.detach().reshape(-1).double()
    if not flat.numel() or not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("iteration-one monitor received empty/nonfinite log weights")
    shares = torch.softmax(flat, dim=0)
    n = flat.numel()
    ess = 1.0 / shares.square().sum()
    return {
        "ess": float(ess.cpu()),
        "ess_fraction": float((ess / n).cpu()),
        "max_mean_one_weight": float((shares.max() * n).cpu()),
        "top_1pct_mass": float(shares.topk(max(1, math.ceil(n * .01))).values.sum().cpu()),
        "log_weight_std": float(flat.std(unbiased=False).cpu()),
        "rows": float(n),
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

    weighted_se = auc_null_standard_error(
        int(round(n_truth)), int(round(n_gen_weighted))
    )

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
    reuse_validation_for_final: bool = False,
    early_stop_auc_gap: float | None = None,
    early_stop_balanced_accuracy_lcb: float | None = None,
    early_stop_balanced_accuracy_confidence_z: float = 1.96,
    early_stop_balanced_accuracy_required_consecutive: int = 1,
    fit_overrides: Mapping[str, Any] | None = None,
    warm_start_cache: dict[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    training_readiness: Mapping[str, Any] | None = None,
    explicit_split_indices: tuple[Tensor, Tensor, Tensor] | None = None,
) -> dict[str, float]:
    if pool.n_events < 30:
        raise ValueError("fresh adaptive OmniFold audit needs at least 30 events")
    aligned = pool.to(device)
    aligned_log_weight = log_weight.to(device=device, dtype=torch.float32)
    weights = global_mean_one_from_log_weights(aligned_log_weight)
    resolved_fit_block = {
        **cfg.audit_fit,
        **dict(fit_overrides or {}),
        "require_saturation": False,
    }
    n_valid = max(1, int(round(0.20 * pool.n_events)))
    n_fit = (
        pool.n_events - n_valid
        if reuse_validation_for_final
        else max(1, int(round(0.60 * pool.n_events)))
    )
    if cfg.single_pool_train_validation:
        fit_index, validation_index = _identity_crossfit_splits(
            pool.identity_inputs, folds=5, seed=cfg.single_pool_split_seed,
        )[0]
        n_fit, n_valid = len(fit_index), len(validation_index)
    if explicit_split_indices is not None:
        n_fit, n_valid = (len(index) for index in explicit_split_indices[:2])
    fit_config = build_fit_config(
        resolved_fit_block,
        n_train=n_fit,
        n_validation=n_valid,
        max_batch_population=n_fit if cfg.single_pool_train_validation else None,
    )
    _log.info(
        "[DGPO/omnifold] audit fit budget: n_fit=%s n_valid=%s batch=%s "
        "min_steps=%s patience_evaluations=%s interval_steps=%s "
        "require_saturation=%s",
        n_fit,
        n_valid,
        fit_config.batch_size,
        fit_config.min_steps,
        fit_config.validation_patience_evaluations,
        fit_config.validation_interval_steps,
        fit_config.require_saturation,
    )
    factory = peft_bank_factory(
        model_builder,
        pool.packing_spec,
        "audit",
        reset=True,
        classifier_overrides={
            key: resolved_fit_block[key]
            for key in (
                "head_dropout",
                "topology_dropout",
                "decoder_hidden_dim",
                "decoder_layers",
                "decoder_heads",
                "periodic_pair_features",
                "topology_fourier_embedding",
                "topology_direct_logit",
                "topology_context_residual_scale",
                "topology_conditioning",
                "topology_pair_token",
                "relation_token_count",
                "visible_pair_rest_frame",
                "topology_max_harmonic",
                "topology_include_theta_pair",
                "topology_theta_fourier",
                "topology_hidden_dim",
                "topology_embedding_dim",
                "topology_fusion_hidden_dim",
                "train_layernorm",
                "train_encoder",
                "train_grouped_sequential_embedding",
                "train_invisible_projector",
                "train_angular_conditioning",
                "train_backbone",
                "train_last_pet_block",
                "asymmetric_attention",
            )
            if resolved_fit_block.get(key) is not None
        },
    )
    classifier_kind = resolved_fit_block.get("classifier_kind", "evenet")
    if classifier_kind != "evenet":
        raise ValueError(f"unknown audit classifier_kind: {classifier_kind}")
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
            reuse_early_stop_for_audit=bool(reuse_validation_for_final),
            early_stop_auc_gap=early_stop_auc_gap,
            early_stop_balanced_accuracy_lcb=(
                early_stop_balanced_accuracy_lcb
            ),
            early_stop_balanced_accuracy_confidence_z=(
                early_stop_balanced_accuracy_confidence_z
            ),
            early_stop_balanced_accuracy_required_consecutive=(
                early_stop_balanced_accuracy_required_consecutive
            ),
            progress_callback=progress_callback,
            warm_start_cache=warm_start_cache,
            identity_split_seed=(
                cfg.single_pool_split_seed
                if cfg.single_pool_train_validation
                else (
                    int(seed)
                    if warm_start_cache is not None
                    or training_readiness is not None
                    else None
                )
            ),
            **({"identity_condition": aligned.identity_override} if aligned.identity_override is not None else {}),
            **({"training_readiness": training_readiness} if training_readiness is not None else {}),
            **({"explicit_split_indices": explicit_split_indices} if explicit_split_indices is not None else {}),
        )
    finally:
        discard = getattr(model_builder, "discard_bank", None)
        if callable(discard):
            discard(weighted_bank_name)

    fit_diag = weighted_result.fit_diagnostics
    saturated = bool(getattr(fit_diag, "saturated", False))
    threshold_reached = bool(getattr(fit_diag, "threshold_reached", False))
    unsafe_lcb_reached = bool(
        getattr(fit_diag, "unsafe_balanced_accuracy_lcb_reached", False)
    )
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
        "audit_unsafe_balanced_accuracy_lcb_reached": float(
            unsafe_lcb_reached
        ),
        "audit_fresh_pretrained_initialization": float(not getattr(weighted_result, "warm_started", False)),
        "audit_fresh_initialization": float(not getattr(weighted_result, "warm_started", False)),
        "audit_reused_previous_classifier": float(getattr(weighted_result, "warm_started", False)),
        "audit_training_ready": float(getattr(weighted_result, "training_ready", False)),
        "audit_training_min_steps": float(getattr(weighted_result, "training_min_steps", 0)),
        "audit_training_steps": float(getattr(fit_diag, "steps_completed", 0) or 0),
        "audit_training_epochs": (
            float(getattr(fit_diag, "steps_completed", 0) or 0)
            / max(1, int(getattr(weighted_result, "training_steps_per_epoch", 0)))
            if getattr(weighted_result, "training_steps_per_epoch", 0) else float("nan")
        ),
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
        "audit_validation_oriented_balanced_accuracy": _finite_or_nan(
            fit_diag,
            "validation_oriented_balanced_accuracy",
        ),
        "audit_validation_balanced_accuracy_lcb": _finite_or_nan(
            fit_diag,
            "validation_balanced_accuracy_lcb",
        ),
        "audit_validation_balanced_accuracy_lcb_standard_error": (
            _finite_or_nan(
                fit_diag,
                "validation_balanced_accuracy_lcb_standard_error",
            )
        ),
        "audit_validation_balanced_accuracy_lcb_streak": float(
            getattr(
                fit_diag,
                "validation_balanced_accuracy_lcb_streak",
                0,
            )
        ),
        "audit_fit_events": float(weighted_result.fit_events),
        "audit_early_stop_events": float(getattr(weighted_result, "early_stop_events", n_valid)),
        "audit_test_events": float(weighted_result.audit_events),
        "audit_steps_per_epoch": float(getattr(weighted_result, "training_steps_per_epoch", 0)),
        "audit_validation_interval_steps": float(fit_config.validation_interval_steps),
        "audit_patience_evaluations": float(fit_config.validation_patience_evaluations),
        "audit_probe_events": float(pool.n_events),
        "audit_reused_validation_for_final": float(
            reuse_validation_for_final
        ),
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


def fit_fresh_topology_audit(
    *,
    pool: AdaptiveOmniFoldPool,
    log_weight: Tensor,
    model_builder: Any,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Run an independent audit with the exact OmniFold classifier builder.

    This intentionally changes only initialization and event split. Architecture,
    trainable scope, periodic pair features, Fourier embedding, and dropout are
    inherited from the same ``EvenetAdapterModelBuilder`` as residual OmniFold.
    """

    result = fit_fresh_audit(
        pool=pool,
        log_weight=log_weight,
        model_builder=model_builder,
        cfg=cfg,
        device=device,
        seed=seed,
        progress_callback=progress_callback,
    )
    return {
        "topology_audit_auc": float(result["judge_auc_weighted"]),
        "topology_audit_auc_gap": float(result["weighted_auc_gap"]),
        "topology_audit_balanced_accuracy": float(
            result["audit_balanced_accuracy"]
        ),
        "topology_audit_saturated": float(result["audit_saturated"]),
        "topology_audit_fit_events": float(result["audit_fit_events"]),
        "topology_audit_test_events": float(result["audit_test_events"]),
        "topology_audit_same_architecture": 1.0,
    }


def fit_repeated_topology_audit(
    *,
    pool: AdaptiveOmniFoldPool,
    log_weight: Tensor,
    model_builder: Any,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int,
    initial_audit: Mapping[str, float] | None = None,
    progress_callback: OmniFoldProgressCallback | None = None,
) -> dict[str, float]:
    """Aggregate same-architecture audits across independent fit seeds.

    The already-independent primary acceptance audit can be supplied as repeat
    one, avoiding a redundant fourth full EveNet fit when three confirmations
    are requested.
    """

    repeats = int(cfg.topology_acceptance_repeats)
    results: list[dict[str, float]] = []
    if initial_audit is not None:
        results.append(
            {
                "topology_audit_auc": float(
                    initial_audit["judge_auc_weighted"]
                ),
                "topology_audit_auc_gap": float(
                    initial_audit["weighted_auc_gap"]
                ),
                "topology_audit_balanced_accuracy": float(
                    initial_audit["audit_balanced_accuracy"]
                ),
                "topology_audit_saturated": float(
                    initial_audit["audit_saturated"]
                ),
                "topology_audit_fit_events": float(
                    initial_audit["audit_fit_events"]
                ),
                "topology_audit_test_events": float(
                    initial_audit["audit_test_events"]
                ),
                "topology_audit_same_architecture": 1.0,
            }
        )
    for index in range(len(results), repeats):
        repeat_number = index + 1
        seed_index = index if initial_audit is None else index - 1
        results.append(
            fit_fresh_topology_audit(
                pool=pool,
                log_weight=log_weight,
                model_builder=model_builder,
                cfg=cfg,
                device=device,
                seed=int(seed) + seed_index * 104_729,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda row, repeat_number=repeat_number: progress_callback(
                        "topology_acceptance_audit",
                        {"repeat": float(repeat_number), **row},
                    )
                ),
            )
        )
    mean_keys = (
        "topology_audit_auc",
        "topology_audit_auc_gap",
        "topology_audit_balanced_accuracy",
        "topology_audit_fit_events",
        "topology_audit_test_events",
    )
    aggregate = {
        key: sum(item[key] for item in results) / float(repeats)
        for key in mean_keys
    }
    gaps = [item["topology_audit_auc_gap"] for item in results]
    gap_mean = aggregate["topology_audit_auc_gap"]
    gap_stdev = (
        math.sqrt(
            sum((gap - gap_mean) ** 2 for gap in gaps) / float(repeats - 1)
        )
        if repeats > 1
        else 0.0
    )
    aggregate.update(
        {
            "topology_audit_saturated": float(
                all(item["topology_audit_saturated"] >= 0.5 for item in results)
            ),
            "topology_audit_same_architecture": float(
                all(
                    item.get("topology_audit_same_architecture", 0.0) >= 0.5
                    for item in results
                )
            ),
            "topology_audit_repeats": float(repeats),
            "topology_audit_auc_gap_stdev": gap_stdev,
            "topology_audit_auc_gap_se": gap_stdev / math.sqrt(float(repeats)),
        }
    )
    for index, item in enumerate(results, start=1):
        for key, value in item.items():
            suffix = key.removeprefix("topology_audit_")
            aggregate[f"topology_audit_repeat_{index:02d}_{suffix}"] = value
    return aggregate


def _external_raw_audit_population(
    training_pool: AdaptiveOmniFoldPool, evaluation_pool: AdaptiveOmniFoldPool, *, seed: int,
) -> tuple[AdaptiveOmniFoldPool, tuple[Tensor, Tensor, Tensor]]:
    """Keep the entire selected training fold; split external evaluation 50/50.

    Hash only visible identities, so Ray reorderings and newly generated samples
    cannot move an event between early-stop and final-test populations.
    """
    if training_pool.packing_spec != evaluation_pool.packing_spec:
        raise ValueError("raw audit training and evaluation packing must match")
    early, final = _identity_crossfit_splits(evaluation_pool.identity_inputs, folds=2, seed=seed)[0]
    n_fit = training_pool.n_events
    combined = AdaptiveOmniFoldPool(
        packed_event=torch.cat((training_pool.packed_event, evaluation_pool.packed_event)),
        truth=torch.cat((training_pool.truth, evaluation_pool.truth)),
        candidates=torch.cat((training_pool.candidates, evaluation_pool.candidates)),
        packing_spec=training_pool.packing_spec,
    )
    return combined, (torch.arange(n_fit, device=early.device), early + n_fit, final + n_fit)


def fit_raw_policy_audit(
    *,
    pool: AdaptiveOmniFoldPool,
    model_builder: Any,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int,
    early_stop_balanced_accuracy_lcb: float | None = None,
    early_stop_balanced_accuracy_confidence_z: float = 1.96,
    early_stop_balanced_accuracy_required_consecutive: int = 1,
    fit_overrides: Mapping[str, Any] | None = None,
    warm_start_cache: dict[str, Any] | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    use_training_readiness: bool = True,
    repeats: int | None = None,
    training_pool: AdaptiveOmniFoldPool | None = None,
) -> dict[str, float]:
    """Fit fresh unweighted truth-vs-policy judges on one fixed event pool.

    ``audit_fit.repeats`` is opt-in. Each repeat gets an independent model
    initialization and identity split, while every policy boundary reuses the
    same repeat seeds. This makes the mean gap a paired trajectory statistic
    and exposes between-judge uncertainty without feeding it back into DGPO.
    """

    repeat_count = int(
        cfg.audit_fit.get("repeats", 1) if repeats is None else repeats
    )
    repeat_seed_stride = int(cfg.audit_fit.get("repeat_seed_stride", 104_729))
    if repeat_count < 1:
        raise ValueError("raw audit repeats must be positive")
    if repeat_count > 1 and warm_start_cache is not None:
        raise ValueError("repeated raw audits cannot share a warm-start cache")
    matched_fold = cfg.audit_fit.get("training_population", "probe_split") == "omnifold_fold"
    if matched_fold != (training_pool is not None):
        raise ValueError("omnifold_fold raw audit requires its freshly generated training fold")
    if matched_fold and warm_start_cache is not None:
        raise ValueError("omnifold_fold raw audits are cold, not warm-started")

    def _as_raw_metrics(raw_audit: Mapping[str, float]) -> dict[str, float]:
        return {
            "raw_auc": float(raw_audit["judge_auc_weighted"]),
            "raw_classifier_warm_started": float(raw_audit.get("audit_reused_previous_classifier", 0.0)),
            "raw_audit_training_ready": float(raw_audit.get("audit_training_ready", 0.0)),
            "raw_audit_training_min_steps": float(raw_audit.get("audit_training_min_steps", 0.0)),
            "raw_audit_training_steps": float(raw_audit.get("audit_training_steps", 0.0)),
            "raw_audit_training_epochs": float(raw_audit.get("audit_training_epochs", float("nan"))),
            "raw_auc_gap": float(raw_audit["weighted_auc_gap"]),
            "raw_balanced_accuracy": float(raw_audit["audit_balanced_accuracy"]),
            "raw_audit_saturated": float(raw_audit["audit_saturated"]),
            "raw_audit_unsafe_balanced_accuracy_lcb_reached": float(
                raw_audit.get("audit_unsafe_balanced_accuracy_lcb_reached", 0.0)
            ),
            "raw_audit_validation_loss": float(raw_audit["audit_validation_loss"]),
            "raw_audit_validation_auc": float(raw_audit["audit_validation_auc"]),
            "raw_audit_validation_oriented_balanced_accuracy": float(
                raw_audit.get("audit_validation_oriented_balanced_accuracy", float("nan"))
            ),
            "raw_audit_validation_balanced_accuracy_lcb": float(
                raw_audit.get("audit_validation_balanced_accuracy_lcb", float("nan"))
            ),
            "raw_audit_validation_balanced_accuracy_lcb_standard_error": float(
                raw_audit.get("audit_validation_balanced_accuracy_lcb_standard_error", float("nan"))
            ),
            "raw_audit_validation_balanced_accuracy_lcb_streak": float(
                raw_audit.get("audit_validation_balanced_accuracy_lcb_streak", 0.0)
            ),
            "raw_audit_fit_events": float(raw_audit["audit_fit_events"]),
            "raw_audit_early_stop_events": float(raw_audit.get("audit_early_stop_events", float("nan"))),
            "raw_audit_test_events": float(raw_audit["audit_test_events"]),
            "raw_audit_probe_events": float(raw_audit["audit_probe_events"]),
            "raw_audit_steps_per_epoch": float(raw_audit.get("audit_steps_per_epoch", float("nan"))),
            "raw_audit_validation_interval_steps": float(raw_audit.get("audit_validation_interval_steps", float("nan"))),
            "raw_audit_patience_evaluations": float(raw_audit.get("audit_patience_evaluations", float("nan"))),
            "raw_audit_uses_omnifold_fold": float(matched_fold),
            "raw_audit_training_fold": float(cfg.audit_fit.get("training_fold", 1)) if matched_fold else 0.0,
            "raw_auc_null_se_approx": float(
                raw_audit.get("audit_auc_null_se_approx", float("nan"))
            ),
            "raw_auc_gap_z_approx": float(
                raw_audit.get("audit_weighted_auc_gap_z_approx", float("nan"))
            ),
            "raw_auc_gap_pvalue_approx": float(
                raw_audit.get("audit_weighted_auc_gap_pvalue_approx", float("nan"))
            ),
        }

    results: list[dict[str, float]] = []
    for repeat_index in range(repeat_count):
        repeat_number = repeat_index + 1
        audit_seed = int(seed) + 7919 + repeat_index * repeat_seed_stride
        fit_pool, explicit_splits = pool, None
        if training_pool is not None:
            fit_pool, explicit_splits = _external_raw_audit_population(training_pool, pool, seed=audit_seed)
        zero_log_weight = torch.zeros(
            tuple(fit_pool.candidates.shape[:2]), device=fit_pool.candidates.device, dtype=torch.float32,
        )
        raw_audit = fit_fresh_audit(
            pool=fit_pool,
            log_weight=zero_log_weight,
            model_builder=model_builder,
            cfg=cfg,
            device=device,
            seed=audit_seed,
            reuse_validation_for_final=not bool(
                cfg.audit_fit.get("disjoint_final_audit", False)
            ),
            # Raw AUC measures policy-vs-truth discrepancy. Train to the
            # validation-loss plateau; threshold stopping is reserved for the
            # deliberately capacity-limited reward classifiers.
            early_stop_auc_gap=None,
            early_stop_balanced_accuracy_lcb=early_stop_balanced_accuracy_lcb,
            early_stop_balanced_accuracy_confidence_z=(
                early_stop_balanced_accuracy_confidence_z
            ),
            early_stop_balanced_accuracy_required_consecutive=(
                early_stop_balanced_accuracy_required_consecutive
            ),
            fit_overrides=fit_overrides,
            warm_start_cache=warm_start_cache,
            **({"explicit_split_indices": explicit_splits} if explicit_splits is not None else {}),
            progress_callback=(
                None
                if progress_callback is None
                else lambda row, repeat_number=repeat_number: progress_callback(
                    {"repeat": float(repeat_number), **row}
                )
            ),
            **(
                {"training_readiness": cfg.audit_fit["training_readiness"]}
                if use_training_readiness
                and cfg.audit_fit.get("training_readiness") is not None
                else {}
            ),
        )
        if training_pool is not None:
            raw_audit = {**raw_audit, "audit_probe_events": float(pool.n_events)}
        results.append(_as_raw_metrics(raw_audit))

    if repeat_count == 1:
        aggregate = {
            **results[0],
            "raw_audit_repeats": 1.0,
            "raw_auc_gap_stdev": 0.0,
            "raw_auc_gap_se": 0.0,
        }
    else:
        aggregate = {
            key: sum(item[key] for item in results) / float(repeat_count)
            for key in results[0]
        }
        aggregate["raw_classifier_warm_started"] = float(
            any(item["raw_classifier_warm_started"] >= 0.5 for item in results)
        )
        aggregate["raw_audit_training_ready"] = float(
            all(item["raw_audit_training_ready"] >= 0.5 for item in results)
        )
        aggregate["raw_audit_saturated"] = float(
            all(item["raw_audit_saturated"] >= 0.5 for item in results)
        )
        aggregate["raw_audit_unsafe_balanced_accuracy_lcb_reached"] = float(
            any(
                item["raw_audit_unsafe_balanced_accuracy_lcb_reached"] >= 0.5
                for item in results
            )
        )
        gaps = [item["raw_auc_gap"] for item in results]
        gap_mean = aggregate["raw_auc_gap"]
        gap_stdev = math.sqrt(
            sum((gap - gap_mean) ** 2 for gap in gaps)
            / float(repeat_count - 1)
        )
        aggregate.update(
            {
                "raw_audit_repeats": float(repeat_count),
                "raw_auc_gap_stdev": float(gap_stdev),
                "raw_auc_gap_se": float(
                    gap_stdev / math.sqrt(float(repeat_count))
                ),
            }
        )
        for index, item in enumerate(results, start=1):
            for key, value in item.items():
                aggregate[f"raw_audit_repeat_{index:02d}/{key.removeprefix('raw_')}"] = value

    if (
        bool(cfg.audit_fit.get("fail_if_unsaturated", False))
        and aggregate["raw_audit_saturated"] < 0.5
    ):
        raise RuntimeError(
            "one or more repeated raw policy audits did not reach validation "
            "saturation"
        )
    return aggregate


def fit_reference_trust_audit(
    *,
    pool: AdaptiveOmniFoldPool,
    model_builder: Any,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Fit a fresh reference-vs-current classifier on a paired K=1 pool."""

    raw = fit_raw_policy_audit(
        pool=pool,
        model_builder=model_builder,
        cfg=cfg,
        device=device,
        seed=int(seed) + 104_729,
        early_stop_balanced_accuracy_lcb=(
            float(cfg.classifier_trust_max_balanced_accuracy)
            if cfg.classifier_trust_unsafe_early_stop_enabled
            else None
        ),
        early_stop_balanced_accuracy_confidence_z=(
            cfg.classifier_trust_unsafe_early_stop_confidence_z
        ),
        early_stop_balanced_accuracy_required_consecutive=(
            cfg.classifier_trust_unsafe_early_stop_required_consecutive_validations
        ),
        fit_overrides=cfg.classifier_trust_audit_fit,
        use_training_readiness=False,
        repeats=1,
        progress_callback=progress_callback,
    )
    return {
        "reference_trust_auc": float(raw["raw_auc"]),
        "reference_trust_auc_gap": float(raw["raw_auc_gap"]),
        "reference_trust_balanced_accuracy": float(
            raw["raw_balanced_accuracy"]
        ),
        "reference_trust_saturated": float(raw["raw_audit_saturated"]),
        "reference_trust_unsafe_balanced_accuracy_lcb_reached": float(
            raw["raw_audit_unsafe_balanced_accuracy_lcb_reached"]
        ),
        "reference_trust_validation_balanced_accuracy_lcb": float(
            raw["raw_audit_validation_balanced_accuracy_lcb"]
        ),
        "reference_trust_validation_balanced_accuracy_lcb_streak": float(
            raw["raw_audit_validation_balanced_accuracy_lcb_streak"]
        ),
        "reference_trust_validation_loss": float(
            raw["raw_audit_validation_loss"]
        ),
        "reference_trust_validation_auc": float(
            raw["raw_audit_validation_auc"]
        ),
        "reference_trust_fit_events": float(raw["raw_audit_fit_events"]),
        "reference_trust_test_events": float(raw["raw_audit_test_events"]),
        "reference_trust_probe_events": float(raw["raw_audit_probe_events"]),
    }


def probe_installed_reward(
    reward_source: Any,
    pool: AdaptiveOmniFoldPool,
    *,
    cfg: AdaptiveOmniFoldConfig,
    device: torch.device,
    seed: int | None = None,
    early_stop_auc_gap: float | None = None,
    raw_warm_start_cache: dict[str, Any] | None = None,
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
            reuse_validation_for_final=True,
            early_stop_auc_gap=early_stop_auc_gap,
            progress_callback=(
                None
                if progress_callback is None
                else lambda row: progress_callback("staleness_audit", row)
            ),
        )
    )
    if cfg.raw_audit_enabled:
        probe.update(
            fit_raw_policy_audit(
            pool=pool,
            model_builder=reward_source.model_builder,
            cfg=cfg,
            device=device,
            seed=cfg.probe_seed if seed is None else int(seed),
            warm_start_cache=raw_warm_start_cache,
            progress_callback=(
                None
                if progress_callback is None
                else lambda row: progress_callback("raw_staleness_audit", row)
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


def _broadcast_int(value: int, *, world_size: int, device: torch.device) -> int:
    if int(world_size) <= 1:
        return int(value)
    payload = torch.tensor([int(value)], device=device, dtype=torch.int64)
    dist.broadcast(payload, src=0)
    return int(payload.item())


def residual_closure_auc_limit(
    cfg: AdaptiveOmniFoldConfig, *, global_step: int | None = None,
) -> float:
    """Resolve once per refit from the persisted policy clock, not fit updates."""
    if not cfg.residual_closure_schedule:
        return .5 + float(cfg.residual_min_auc_gain)
    if type(global_step) is not int or global_step < 0:
        raise ValueError("residual closure schedule requires a nonnegative DGPO global_step")
    limit = cfg.residual_closure_schedule[0][1]
    for start, auc in cfg.residual_closure_schedule:
        if global_step < start:
            break
        limit = auc
    return float(limit)


def run_adaptive_refit(
    *,
    state: AdaptiveOmniFoldState,
    cfg: AdaptiveOmniFoldConfig,
    reward_source: Any,
    round_ref_model: torch.nn.Module,
    policy_snapshot_state_dict: Mapping[str, Tensor],
    fit_pool: AdaptiveOmniFoldPool,
    score_pool: AdaptiveOmniFoldPool,
    baseline_pool: AdaptiveOmniFoldPool | None = None,
    epoch: int,
    device: torch.device,
    world_size: int,
    enforce_round_acceptance: bool = True,
    progress_callback: OmniFoldProgressCallback | None = None,
    global_step: int | None = None,
) -> dict[str, Any]:
    """Fit, independently audit, and atomically install reward/reference pair."""
    state.last_recalibration_attempt_epoch = int(epoch)
    closure_limit = residual_closure_auc_limit(cfg, global_step=global_step)
    closure_metrics = {
        "omnifold/residual_closure_auc_limit": closure_limit,
        "omnifold/residual_closure_schedule_enabled": float(bool(cfg.residual_closure_schedule)),
        "omnifold/refit_global_step": float(global_step if global_step is not None else -1),
    }
    if cfg.iteration_one_only:
        closure_metrics = {
            "omnifold/iteration_one_only": 1.0,
            "omnifold/closure_evaluated": 0.0,
            "omnifold/refit_global_step": float(global_step if global_step is not None else -1),
            "omnifold/iteration1_monitor/min_signal_auc": closure_limit,
        }
    if cfg.fixed_iteration_budget:
        closure_metrics = {
            "omnifold/fixed_iteration_budget": float(cfg.max_iterations),
            "omnifold/closure_evaluated": 0.0,
            "omnifold/refit_global_step": float(global_step if global_step is not None else -1),
        }
        _log.info("[DGPO/omnifold] fixed budget: %s iterations; install useful increments without requiring closure", cfg.max_iterations)
    elif cfg.iteration_one_only:
        _log.info("[DGPO/omnifold] refit global_step=%s iteration-one-only: signal AUC>%.6g; closure NOT evaluated",
                  global_step, closure_limit)
    else:
        _log.info("[DGPO/omnifold] refit global_step=%s residual closure AUC<=%.6g (fixed for this refit)",
                  global_step, closure_limit)
    if cfg.single_pool_train_validation:
        # Both sides come from this policy's one generated, budgeted pool.
        # Warm-started reward classifiers never fit the fixed validation 20%.
        fit_pool, score_pool = split_single_classifier_pool(
            fit_pool, seed=cfg.single_pool_split_seed,
        )
        _log.info("[DGPO/omnifold] single-pool internal split: fit=%s validation=%s seed=%s",
                  fit_pool.n_events, score_pool.n_events, cfg.single_pool_split_seed)
    if fit_pool.packing_spec != score_pool.packing_spec:
        raise RuntimeError("adaptive fit and held-out pools use different EveNet shapes")
    if (
        baseline_pool is not None
        and baseline_pool.packing_spec != score_pool.packing_spec
    ):
        raise RuntimeError(
            "adaptive acceptance and trigger-baseline pools use different EveNet shapes"
        )
    score_on_device = score_pool.to(device)
    smallest_fold = None
    if cfg.single_pool_train_validation:
        if cfg.warm_start_iterations or cfg.crossfit_partition == "identity":
            smallest_fold = min(
                len(fit_index)
                for repeat in range(1, cfg.crossfit_repeats + 1)
                for fit_index, _ in _identity_crossfit_splits(
                    fit_pool.identity_inputs,
                    folds=cfg.crossfit_folds,
                    seed=_crossfit_repeat_seed(cfg.seed, repeat),
                )
            )
        else:
            smallest_fold = fit_pool.n_events - math.ceil(fit_pool.n_events / cfg.crossfit_folds)
    fit_config = build_fit_config(
        cfg.fit,
        n_train=fit_pool.n_events,
        n_validation=score_pool.n_events,
        max_batch_population=smallest_fold,
    )
    reward_bank_name = "adaptive_reward"
    factory = peft_bank_factory(
        reward_source.model_builder,
        fit_pool.packing_spec,
        reward_bank_name,
        reset=True,
        classifier_overrides=cfg.reward_classifier,
    )
    previous_warm_state = None
    outer_partition = (
        {"schema": "condition-hash-80-20-v1", "seed": cfg.single_pool_split_seed}
        if cfg.single_pool_train_validation else None
    )
    if cfg.warm_start_iterations and bool(getattr(reward_source, "is_installed", False)):
        previous_reward = reward_source.frozen_reward
        if previous_reward.packing_spec == fit_pool.packing_spec:
            previous_warm_state = getattr(previous_reward, "warm_start_state", None)
            if previous_warm_state is not None and previous_warm_state.get("outer_partition") != outer_partition:
                # Legacy weights might have fitted the new validation identities.
                previous_warm_state = None
    try:
        result = fit_residual_ratio_stack(
            model_factory=factory,
            data_condition=fit_pool.packed_event,
            data_sample=fit_pool.truth,
            gen_condition=fit_pool.packed_event,
            gen_sample=fit_pool.candidates,
            iterations=cfg.max_iterations,
            **({"iteration_one_only": True} if cfg.iteration_one_only else {}),
            **({"fixed_iteration_budget": True} if cfg.fixed_iteration_budget else {}),
            min_iterations=cfg.min_iterations,
            fit_config=fit_config,
            tempering=cfg.tempering,
            crossfit_folds=cfg.crossfit_folds,
            crossfit_repeats=cfg.crossfit_repeats,
            residual_min_auc_gain=(closure_limit - .5 if cfg.residual_closure_schedule
                                   else cfg.residual_min_auc_gain),
            seed=(
                cfg.seed
                if cfg.trust_trajectory_search_enabled
                else cfg.seed + int(state.recalibration_count)
            ),
            validation_data_condition=score_on_device.packed_event,
            validation_data_sample=score_on_device.truth,
            validation_gen_condition=score_on_device.packed_event,
            validation_gen_sample=score_on_device.candidates,
            device=device,
            warm_start_iterations=cfg.warm_start_iterations,
            warm_start_state=previous_warm_state,
            **({"warm_start_from_iteration_one": True} if cfg.warm_start_from_iteration_one else {}),
            **({"later_iteration_learning_rate": cfg.fit["later_iteration_learning_rate"]}
               if cfg.fit.get("later_iteration_learning_rate") is not None else {}),
            **({"later_iteration_train_mode": cfg.fit["later_iteration_train_mode"]}
               if "later_iteration_train_mode" in cfg.fit else {}),
            **({"later_iteration_decoder_learning_rate": cfg.fit["later_iteration_decoder_learning_rate"]}
               if "later_iteration_decoder_learning_rate" in cfg.fit else {}),
            crossfit_seed=cfg.seed,
            crossfit_partition=cfg.crossfit_partition,
            min_steps_per_fold=cfg.fit.get("min_steps_per_fold", 0),
            **({"max_steps_per_fold": cfg.fit["max_steps_per_fold"]}
               if cfg.fit.get("max_steps_per_fold") is not None else {}),
            # Opt-in exact fold-epoch controls; legacy overlays keep their
            # original scaled update budgets. Explicit step/evaluation controls
            # still take precedence over epoch-based validation settings.
            **({
                "warm_start_min_epochs_per_fold": cfg.fit["warm_start_min_epochs_per_fold"],
                "validation_interval_epochs": (
                    cfg.fit.get("validation_interval_epochs", 0.2)
                    if cfg.fit.get("validation_interval_steps") is None else None
                ),
                "validation_patience_epochs": (
                    cfg.fit.get("validation_patience_epochs", 5.0)
                    if "validation_patience_evaluations" not in cfg.fit else None
                ),
            } if cfg.fit.get("warm_start_min_epochs_per_fold") is not None else {}),
            **({"identity_condition": fit_pool.identity_override} if fit_pool.identity_override is not None else {}),
            progress_callback=(
                None
                if progress_callback is None
                else lambda row: progress_callback("residual_reward", row)
            ),
            log_ratio_clip=cfg.log_ratio_clip,
            minimum_ess_fraction=cfg.minimum_ess_fraction,
            adaptive_tempering=cfg.adaptive_tempering_enabled,
            target_ess_fraction=cfg.target_ess_fraction,
            minimum_tempering=cfg.minimum_tempering,
            tempering_grid_steps=cfg.tempering_grid_steps,
            inherit_previous_tempering=cfg.inherit_previous_tempering,
            ess_aware_checkpoint_selection=cfg.ess_aware_checkpoint_selection,
            ess_aware_max_checkpoints=cfg.ess_aware_max_checkpoints,
            ess_aware_first_residual_only=cfg.ess_aware_first_residual_only,
            minimum_sufficient_balanced_accuracy=cfg.fit.get(
                "minimum_sufficient_balanced_accuracy"
            ),
            minimum_sufficient_confidence_z=cfg.fit.get(
                "minimum_sufficient_confidence_z", 0.0
            ),
            minimum_sufficient_required_consecutive=cfg.fit.get(
                "minimum_sufficient_required_consecutive", 1
            ),
        )
        if outer_partition is not None and getattr(result, "warm_start_state", None) is not None:
            result.warm_start_state["outer_partition"] = outer_partition
    except RuntimeError as exc:
        message = str(exc)
        if not any(
            marker in message
            for marker in (
                "did not saturate",
                "did not reach minimum-sufficient balanced accuracy",
                "did not enter closure band",
                "failed the null/AUC gate",
                "first residual classifier failed the AUC gate",
                "first residual classifier failed the ESS gate",
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
            **closure_metrics,
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
        log_ratio_clip=cfg.log_ratio_clip,
    ).to(device).eval()
    new_stack.assert_frozen()
    acceptance_seed = cfg.probe_seed + 1000 + int(state.recalibration_count)
    initial_bootstrap = not bool(getattr(reward_source, "is_installed", True))
    candidate_probe: dict[str, float] = {}
    all_saturated = bool(result.diagnostics) and all(
        bool(getattr(item, "saturated", False)) for item in result.diagnostics
    )
    minimum_sufficient_target = cfg.fit.get(
        "minimum_sufficient_balanced_accuracy"
    )
    minimum_sufficient_enabled = minimum_sufficient_target is not None
    all_minimum_sufficient = bool(result.diagnostics) and all(
        bool(getattr(item, "minimum_sufficient", False))
        for item in result.diagnostics
    )
    all_fits_ready = (
        all_minimum_sufficient if minimum_sufficient_enabled else all_saturated
    )
    accuracy_limit = float(cfg.acceptance_max_balanced_accuracy)
    closure_auc = float(
        getattr(result.diagnostics[-1], "validation_auc", float("nan"))
    )
    raw_auc = float(
        getattr(result.diagnostics[0], "validation_auc", float("nan"))
    )
    train_log_weight = getattr(result, "train_log_weight", None)
    validation_log_weight = getattr(result, "validation_log_weight", None)
    if train_log_weight is None or validation_log_weight is None:
        if cfg.minimum_ess_fraction > 0.0:
            raise RuntimeError(
                "ESS installation guard requires train and validation log weights"
            )
        # Compatibility for synthetic/legacy fit-result adapters when the
        # guard is disabled. Production ResidualRatioResult always has both.
        train_weight_diagnostics: dict[str, float] = {}
        validation_weight_diagnostics: dict[str, float] = {}
        observed_ess_fraction = 1.0
    else:
        train_weight_diagnostics = _weight_diagnostics(train_log_weight)
        validation_weight_diagnostics = _weight_diagnostics(
            validation_log_weight
        )
        observed_ess_fraction = min(
            float(train_weight_diagnostics["ess_fraction"]),
            float(validation_weight_diagnostics["ess_fraction"]),
        )
    ess_guard_passed = bool(
        math.isfinite(observed_ess_fraction)
        and observed_ess_fraction >= float(cfg.minimum_ess_fraction)
    )
    raw_auc_se = auc_null_standard_error(
        score_pool.n_events,
        score_pool.n_events,
    )
    round_acceptance = evaluate_round_auc_change(
        state,
        cfg=cfg,
        raw_auc=raw_auc,
        raw_auc_se=raw_auc_se,
        enforce=bool(enforce_round_acceptance and all_fits_ready),
    )
    round_action = _broadcast_int(
        int(round_acceptance["action"]),
        world_size=world_size,
        device=device,
    )
    round_decision = {
        -1: "regressed",
        0: "plateau",
        1: "improved",
        2: "bypass",
    }[round_action]
    round_acceptance["action"] = round_action
    round_acceptance["decision"] = round_decision
    round_allowed = round_action in (1, 2)
    candidate_probe.update(
        {
            "residual_closure_auc": closure_auc,
            "residual_closure_auc_gap": abs(closure_auc - 0.5),
        }
    )
    if cfg.iteration_one_only:
        # The only fitted classifier measured RAW separation. Do not publish
        # its AUC as post-weighting closure.
        candidate_probe.clear()
        candidate_probe["iteration1_raw_auc"] = raw_auc
    if cfg.acceptance_audit_enabled:
        candidate_log_weight = score_reward_on_pool(
            new_stack,
            score_pool,
            row_budget=cfg.score_row_budget,
        )
        candidate_probe.update(_weight_diagnostics(candidate_log_weight))
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
        audit_accuracy = float(candidate_probe["audit_balanced_accuracy"])
        topology_accepted = True
        if cfg.topology_acceptance_audit_enabled:
            candidate_probe.update(
                fit_repeated_topology_audit(
                    pool=score_pool,
                    log_weight=candidate_log_weight,
                    model_builder=reward_source.model_builder,
                    cfg=cfg,
                    device=device,
                    seed=acceptance_seed + 500_009,
                    initial_audit=candidate_probe,
                    progress_callback=progress_callback,
                )
            )
            topology_gap = float(
                candidate_probe["topology_audit_auc_gap"]
            )
            topology_accepted = bool(
                candidate_probe.get("topology_audit_saturated", 0.0) >= 0.5
                and math.isfinite(topology_gap)
                and topology_gap < cfg.topology_acceptance_max_auc_gap
            )
        del candidate_log_weight
        accepted_local = bool(
            all_fits_ready
            and candidate_probe.get("audit_saturated", 0.0) >= 0.5
            and math.isfinite(audit_accuracy)
            and audit_accuracy < accuracy_limit
            and topology_accepted
            and ess_guard_passed
            and round_allowed
        )
    else:
        audit_accuracy = float("nan")
        accepted_local = bool(
            all_fits_ready
            and math.isfinite(closure_auc)
            and (
                (result.iterations == 1 and len(result.diagnostics) == 1
                 and bool(getattr(result.diagnostics[0], "accepted", False))
                 and closure_auc > closure_limit)
                if cfg.iteration_one_only else
                (result.iterations == cfg.max_iterations
                 and len(result.diagnostics) == cfg.max_iterations
                 and all(bool(d.accepted) for d in result.diagnostics))
                if cfg.fixed_iteration_budget else closure_auc <= closure_limit
            )
            and ess_guard_passed
            and round_allowed
        )
    accepted = _broadcast_bool(
        accepted_local,
        world_size=world_size,
        device=device,
    )
    # Residual closure is evaluated on the large independent score pool. A
    # separate cheap pool establishes the installed round's staleness baseline
    # with exactly the same sample size/protocol used by routine probes.
    baseline_probe = candidate_probe
    if accepted and baseline_pool is not None:
        baseline_log_weight = score_reward_on_pool(
            new_stack,
            baseline_pool,
            row_budget=cfg.score_row_budget,
        )
        baseline_probe = _weight_diagnostics(baseline_log_weight)
        baseline_probe.update(
            fit_fresh_audit(
                pool=baseline_pool,
                log_weight=baseline_log_weight,
                model_builder=reward_source.model_builder,
                cfg=cfg,
                device=device,
                seed=acceptance_seed + 1_000_003,
                reuse_validation_for_final=True,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda row: progress_callback("baseline_audit", row)
                ),
            )
        )
        baseline_gap_candidate = float(
            baseline_probe["audit_observed_auc_gap"]
        )
        baseline_ready_local = bool(
            math.isfinite(baseline_gap_candidate)
            and baseline_gap_candidate >= 0.0
            and (
                not cfg.require_audit_saturation
                or baseline_probe.get("audit_saturated", 0.0) >= 0.5
            )
        )
        # With the extra candidate-acceptance classifier disabled, this audit
        # establishes the staleness-controller baseline only; it must not veto
        # a stack that already reached the configured cross-fit closure.
        if cfg.acceptance_audit_enabled:
            accepted = _broadcast_bool(
                baseline_ready_local,
                world_size=world_size,
                device=device,
            )
    baseline_gap = float(
        baseline_probe.get(
            "audit_observed_auc_gap",
            abs(raw_auc - .5) if cfg.iteration_one_only else candidate_probe["residual_closure_auc_gap"],
        )
    )
    round_delta_before = float(state.trust_current_delta)
    round_rollback_required = False
    round_stop_requested = False
    if (
        cfg.trust_trajectory_search_enabled
        and round_decision in ("regressed", "plateau")
    ):
        # The trajectory minimum was already selected before this expensive
        # gate.  A non-improving result therefore rejects one complete search
        # direction: return to the incumbent immediately, shrink the radius,
        # and let fresh rollout/minibatch randomness propose another direction.
        if round_decision == "regressed":
            state.trust_round_regressions += 1
            state.trust_round_plateau_streak = 0
        else:
            state.trust_round_plateau_streak += 1
        state.trust_failed_direction_streak += 1
        round_rollback_required = True
        round_stop_requested = bool(
            state.trust_failed_direction_streak
            >= int(cfg.trust_failed_direction_patience)
        )
        state.trust_round_stop_requested = round_stop_requested
    elif round_decision == "regressed":
        state.trust_round_regressions += 1
        state.trust_round_plateau_streak = 0
        state.trust_round_stop_requested = False
        round_rollback_required = True
    elif round_decision == "plateau":
        state.trust_round_plateau_streak += 1
        round_stop_requested = bool(
            state.trust_round_plateau_streak
            >= int(cfg.trust_round_plateau_patience)
        )
        state.trust_round_stop_requested = round_stop_requested
        round_rollback_required = round_stop_requested
    elif round_decision == "improved":
        state.trust_round_plateau_streak = 0
        state.trust_failed_direction_streak = 0
        state.trust_round_stop_requested = False
    if round_rollback_required:
        state.trust_round_rollbacks += 1
        if math.isfinite(round_delta_before) and round_delta_before > 0.0:
            state.trust_current_delta = max(
                float(cfg.trust_delta_floor),
                round_delta_before * float(cfg.trust_empirical_shrink_factor),
            )
        state.trust_distance_window.clear()
        state.trust_step_acceptance_window.clear()
        state.trust_step_scale_window.clear()
        state.reset_trust_trajectory(reset_failed_directions=False)
    diagnostics: dict[str, Any] = {
        "omnifold/accepted": float(accepted),
        "omnifold/iterations_fitted": float(result.iterations),
        "omnifold/classifier_fits_total": float(
            sum(
                len(getattr(item, "fold_diagnostics", (item,)))
                for item in result.diagnostics
            )
        ),
        "omnifold/crossfit_folds": float(cfg.crossfit_folds),
        "omnifold/crossfit_repeats": float(cfg.crossfit_repeats),
        "omnifold/all_fits_saturated": float(all_saturated),
        "omnifold/all_fits_ready": float(all_fits_ready),
        "omnifold/minimum_sufficient/enabled": float(
            minimum_sufficient_enabled
        ),
        "omnifold/minimum_sufficient/target_balanced_accuracy": (
            float("nan")
            if minimum_sufficient_target is None
            else float(minimum_sufficient_target)
        ),
        "omnifold/minimum_sufficient/confidence_z": float(
            cfg.fit.get("minimum_sufficient_confidence_z", 0.0)
        ),
        "omnifold/minimum_sufficient/required_consecutive": float(
            cfg.fit.get("minimum_sufficient_required_consecutive", 1)
        ),
        "omnifold/acceptance_audit_enabled": float(
            cfg.acceptance_audit_enabled
        ),
        **closure_metrics,
        "omnifold/acceptance_max_balanced_accuracy": accuracy_limit,
        "omnifold/topology_acceptance_audit_enabled": float(
            cfg.topology_acceptance_audit_enabled
        ),
        "omnifold/topology_acceptance_max_auc_gap": float(
            cfg.topology_acceptance_max_auc_gap
        ),
        "omnifold/topology_acceptance_repeats": float(
            cfg.topology_acceptance_repeats
        ),
        "omnifold/weight_guard/minimum_ess_fraction": float(
            cfg.minimum_ess_fraction
        ),
        "omnifold/weight_guard/observed_ess_fraction": float(
            observed_ess_fraction
        ),
        "omnifold/weight_guard/passed": float(ess_guard_passed),
        "omnifold/weight_guard/log_ratio_clip": (
            float("nan")
            if cfg.log_ratio_clip is None
            else float(cfg.log_ratio_clip)
        ),
        "omnifold/adaptive_tempering/enabled": float(
            cfg.adaptive_tempering_enabled
        ),
        "omnifold/adaptive_tempering/maximum": float(cfg.tempering),
        "omnifold/adaptive_tempering/minimum": float(
            cfg.minimum_tempering
        ),
        "omnifold/adaptive_tempering/target_ess_fraction": float(
            cfg.target_ess_fraction
        ),
        "omnifold/adaptive_tempering/grid_steps": float(
            cfg.tempering_grid_steps
        ),
        "omnifold/adaptive_tempering/inherit_previous": float(
            cfg.inherit_previous_tempering
        ),
        "omnifold/ess_aware_checkpoint_selection/enabled": float(
            cfg.ess_aware_checkpoint_selection
        ),
        "omnifold/ess_aware_checkpoint_selection/max_checkpoints": float(
            cfg.ess_aware_max_checkpoints
        ),
        "omnifold/ess_aware_checkpoint_selection/first_residual_only": float(
            cfg.ess_aware_first_residual_only
        ),
        **{
            f"omnifold/weight_guard/train_{key}": float(value)
            for key, value in train_weight_diagnostics.items()
        },
        **{
            f"omnifold/weight_guard/validation_{key}": float(value)
            for key, value in validation_weight_diagnostics.items()
        },
        "omnifold/initial_bootstrap": float(initial_bootstrap),
        "reference_trust/round_acceptance/enabled": float(
            round_acceptance["enabled"]
        ),
        "reference_trust/round_acceptance/enforced": float(
            round_acceptance["enforced"]
        ),
        "reference_trust/round_acceptance/eligible": float(
            round_acceptance["eligible"]
        ),
        "reference_trust/round_acceptance/action": float(round_action),
        "reference_trust/round_acceptance/decision": round_decision,
        "reference_trust/round_acceptance/previous_auc_gap": float(
            round_acceptance["previous_auc_gap"]
        ),
        "reference_trust/round_acceptance/current_auc_gap": float(
            round_acceptance["current_auc_gap"]
        ),
        "reference_trust/round_acceptance/previous_auc_se": float(
            round_acceptance["previous_auc_se"]
        ),
        "reference_trust/round_acceptance/current_auc_se": float(
            round_acceptance["current_auc_se"]
        ),
        "reference_trust/round_acceptance/improvement": float(
            round_acceptance["improvement"]
        ),
        "reference_trust/round_acceptance/combined_se": float(
            round_acceptance["combined_se"]
        ),
        "reference_trust/round_acceptance/required_improvement": float(
            round_acceptance["required_improvement"]
        ),
        "reference_trust/round_acceptance/plateau_streak": float(
            state.trust_round_plateau_streak
        ),
        "reference_trust/round_acceptance/plateau_patience": float(
            cfg.trust_round_plateau_patience
        ),
        "reference_trust/round_acceptance/trajectory_search_enabled": float(
            cfg.trust_trajectory_search_enabled
        ),
        "reference_trust/round_acceptance/failed_direction_streak": float(
            state.trust_failed_direction_streak
        ),
        "reference_trust/round_acceptance/failed_direction_patience": float(
            cfg.trust_failed_direction_patience
        ),
        "reference_trust/round_acceptance/regressions": float(
            state.trust_round_regressions
        ),
        "reference_trust/round_acceptance/rollbacks": float(
            state.trust_round_rollbacks
        ),
        "reference_trust/round_acceptance/rollback_required": float(
            round_rollback_required
        ),
        "reference_trust/round_acceptance/stop_requested": float(
            round_stop_requested
        ),
        "reference_trust/round_acceptance/delta_before": round_delta_before,
        "reference_trust/round_acceptance/delta_after": float(
            state.trust_current_delta
        ),
        **{
            f"omnifold/candidate/{key}": float(value)
            for key, value in candidate_probe.items()
        },
        **{
            f"omnifold/baseline/{key}": float(value)
            for key, value in baseline_probe.items()
        },
    }
    if not cfg.trust_round_acceptance_enabled:
        # Keep the legacy round-search implementation available to historical
        # configs without polluting a stationary fixed-pair run with disabled
        # guard fields and NaNs.
        diagnostics = {
            key: value
            for key, value in diagnostics.items()
            if not key.startswith("reference_trust/round_acceptance/")
        }
    for index, fit_diag in enumerate(result.diagnostics, start=1):
        prefix = f"omnifold/fit/iter{index:02d}"
        diagnostics[f"{prefix}/saturated"] = float(
            bool(getattr(fit_diag, "saturated", False))
        )
        diagnostics[f"{prefix}/minimum_sufficient"] = float(
            bool(getattr(fit_diag, "minimum_sufficient", False))
        )
        fold_steps = [
            int(getattr(item, "steps_completed"))
            for item in getattr(fit_diag, "fold_diagnostics", ())
            if getattr(item, "steps_completed", None) is not None
        ]
        threshold_reached_folds = sum(
            bool(getattr(item, "threshold_reached", False))
            for item in getattr(fit_diag, "fold_diagnostics", ())
        )
        diagnostics[f"{prefix}/threshold_reached_folds"] = float(
            threshold_reached_folds
        )
        if fold_steps:
            diagnostics[f"{prefix}/fit_steps_min"] = float(min(fold_steps))
            diagnostics[f"{prefix}/fit_steps_mean"] = float(
                sum(fold_steps) / len(fold_steps)
            )
            diagnostics[f"{prefix}/fit_steps_max"] = float(max(fold_steps))
        diagnostics[f"{prefix}/stored_in_reward"] = float(
            index <= int(result.iterations)
        )
        diagnostics[f"{prefix}/warm_started_folds"] = float(
            len(getattr(fit_diag, "warm_started_folds", ()))
        )
        diagnostics[f"{prefix}/applied_tempering"] = float(
            getattr(fit_diag, "applied_tempering", cfg.tempering)
        )
        diagnostics[f"{prefix}/train_ess_fraction"] = float(
            getattr(fit_diag, "train_ess_fraction", float("nan"))
        )
        diagnostics[f"{prefix}/validation_ess_fraction"] = float(
            getattr(fit_diag, "validation_ess_fraction", float("nan"))
        )
        diagnostics[f"{prefix}/ess_target_reached"] = float(
            bool(getattr(fit_diag, "ess_target_reached", False))
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
        for fold_number, fold_diag in enumerate(
            getattr(fit_diag, "fold_diagnostics", ()), start=1
        ):
            for source_name, metric_name in (
                ("validation_auc", "validation_auc"),
                ("validation_loss", "validation_loss"),
                ("validation_balanced_accuracy", "validation_balanced_accuracy"),
            ):
                value = getattr(fold_diag, source_name, None)
                if value is not None and math.isfinite(float(value)):
                    diagnostics[
                        f"{prefix}/fold{fold_number:02d}/{metric_name}"
                    ] = float(value)
    if cfg.fixed_iteration_budget:
        # The last residual was measured BEFORE the final increment, not after it.
        for key in list(diagnostics):
            if "residual_closure_auc" in key:
                diagnostics.pop(key)
        diagnostics["omnifold/last_residual_auc_before_update"] = closure_auc
    if cfg.iteration_one_only:
        monitor_prefix = "omnifold/iteration1_monitor"
        for name, logw in (("train_oof", result.train_log_weight),
                           ("validation_ensemble", result.validation_log_weight)):
            if logw is not None:
                diagnostics.update({f"{monitor_prefix}/{name}/{key}": value
                                    for key, value in _iteration_one_weight_metrics(logw).items()})
        first = result.diagnostics[0]
        diagnostics.update({
            f"{monitor_prefix}/raw_auc": raw_auc,
            f"{monitor_prefix}/validation_bce": float(first.validation_loss),
            f"{monitor_prefix}/train_bce": float(first.final_loss),
            f"{monitor_prefix}/warm_started_folds": float(len(first.warm_started_folds)),
            f"{monitor_prefix}/tempering": float(cfg.tempering),
        })
    if not accepted:
        state.recalibrations_rejected += 1
        state.probe_exceedance_streak = 0
        state.last_decision = (
            "round_auc_regressed_rollback"
            if round_decision == "regressed"
            else (
                "round_auc_plateau_stop"
                if round_stop_requested
                else (
                    "round_auc_plateau"
                    if round_decision == "plateau"
                    else "recalibration_rejected"
                )
            )
        )
        if round_decision == "regressed":
            accept_reason = (
                "round raw-AUC gap significantly regressed: "
                f"improvement={round_acceptance['improvement']:.5g}, "
                f"required>{round_acceptance['required_improvement']:.5g}; "
                "restore the incumbent round anchor"
            )
        elif round_decision == "plateau":
            accept_reason = (
                "round raw-AUC change is inside the uncertainty band: "
                f"improvement={round_acceptance['improvement']:.5g}, "
                f"required>{round_acceptance['required_improvement']:.5g}, "
                + (
                    f"failed_direction={state.trust_failed_direction_streak}/"
                    f"{cfg.trust_failed_direction_patience}"
                    if cfg.trust_trajectory_search_enabled
                    else (
                        f"plateau={state.trust_round_plateau_streak}/"
                        f"{cfg.trust_round_plateau_patience}"
                    )
                )
            )
        elif not ess_guard_passed:
            accept_reason = (
                "candidate density-ratio weights failed ESS guard: "
                f"ESS/N={observed_ess_fraction:.5g}, "
                f"required>={cfg.minimum_ess_fraction:.5g}"
            )
        elif cfg.acceptance_audit_enabled:
            accept_reason = (
                "candidate did not reach saturated acceptance/baseline closure: "
                f"accuracy={audit_accuracy:.5g}, required<{accuracy_limit:.5g}, "
                f"fits_saturated={all_saturated}, "
                f"audit_saturated={bool(candidate_probe.get('audit_saturated', 0.0))}, "
                f"baseline_saturated={bool(baseline_probe.get('audit_saturated', 0.0))}, "
                f"topology_gap={candidate_probe.get('topology_audit_auc_gap', float('nan')):.5g}, "
                f"topology_saturated={bool(candidate_probe.get('topology_audit_saturated', 0.0))}"
            )
        elif cfg.fixed_iteration_budget:
            accept_reason = "fixed-budget candidate failed fit/signal gate; closure not required"
        elif cfg.iteration_one_only:
            accept_reason = (
                "iteration-one candidate failed minimum-sufficient/signal gate; "
                "closure not evaluated"
                if minimum_sufficient_enabled
                else "iteration-one candidate failed saturation/signal gate; "
                "closure not evaluated"
            )
        else:
            accept_reason = (
                "candidate did not reach saturated cross-fit residual closure: "
                f"auc={closure_auc:.5g}, required<={closure_limit:.5g}, "
                f"fits_saturated={all_saturated}"
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

    # Validate and compute the new radius before mutating either half of the
    # reward/reference pair. The committed call below is then side-effect-safe.
    new_round = int(state.reward_round_id) + 1
    install_adaptive_trust_round(
        state,
        cfg=cfg,
        raw_auc=raw_auc,
        raw_auc_se=raw_auc_se,
        commit=False,
        round_id=new_round,
    )
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
        baseline_auc_gap=baseline_gap,
        cfg=cfg,
        epoch=epoch,
        round_id=new_round,
    )
    trust_diagnostics = install_adaptive_trust_round(
        state,
        cfg=cfg,
        raw_auc=raw_auc,
        raw_auc_se=raw_auc_se,
    )
    state.recalibration_count += 1
    state.last_recalibration_epoch = int(epoch)
    state.last_decision = "recalibration_installed"
    diagnostics.update(
        {
            "omnifold/accept_reason": (
                "fixed iteration budget completed; useful cross-fit increments installed; closure not evaluated"
                if cfg.fixed_iteration_budget else
                "iteration-one-only ablation: minimum-sufficient cross-fit "
                "signal installed; closure not evaluated"
                if cfg.iteration_one_only and minimum_sufficient_enabled else
                "iteration-one-only ablation: saturated cross-fit signal installed; closure not evaluated"
                if cfg.iteration_one_only else
                "candidate reached saturated cross-fit residual closure; "
                "fresh acceptance audit disabled"
                if not cfg.acceptance_audit_enabled
                else (
                    "candidate reached saturated balanced-accuracy closure"
                    if baseline_pool is None
                    else "candidate passed acceptance and trigger-baseline audits"
                )
            ),
            "omnifold/reward_round_id": float(new_round),
            "omnifold/reference_sha256": reference_digest,
            "omnifold/trigger_threshold": float(state.trigger_threshold),
            "omnifold/recalibration_count": float(state.recalibration_count),
            **trust_diagnostics,
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
    "auc_null_standard_error",
    "adaptive_audit_protocol_signature",
    "adaptive_trust_policy_lr_scale",
    "clamp_fixed_trust_radius_after_resume",
    "evaluate_round_auc_change",
    "evaluate_signed_direction_probe",
    "evaluate_signed_direction_recovery",
    "record_extragradient_rejection",
    "reward_round_budget_exhausted",
    "build_reference_trust_pool",
    "fit_reference_trust_audit",
    "fit_raw_policy_audit",
    "fit_repeated_topology_audit",
    "gather_pool_across_ranks",
    "probe_installed_reward",
    "record_reference_trust_attempt",
    "record_reference_trust_distance",
    "resolve_adaptive_config",
    "reward_refit_due_to_age",
    "run_adaptive_refit",
    "install_adaptive_trust_round",
    "start_inherited_round_policy_warmup",
    "start_policy_round_warmup",
    "policy_round_warmup_metrics",
    "advance_policy_round_warmup",
    "raw_best_trust_shrink_due",
    "shrink_trust_radius_on_raw_best",
    "should_probe_epoch",
    "should_skip_incumbent_probe",
    "should_run_raw_only_monitor",
    "trust_region_exhausted",
    "update_trust_trajectory_candidate",
    "update_classifier_trust_controller",
    "update_controller",
    "update_raw_plateau_controller",
    "update_empirical_trust_radius_from_audit",
    "validate_adaptive_pairing",
]
