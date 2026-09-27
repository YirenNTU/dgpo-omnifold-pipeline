from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))

import train_dgpo_h4_gradient_transfer_trace as trace
import train_dgpo_h4_seed_replication as replication
from train_neutrino_backend import read_overlay_yaml


def _resolved_config() -> dict:
    config = read_overlay_yaml(trace.DEFAULT)
    trace.apply_epsilon_result(
        config,
        {"epsilon_rms": 3.0e-7, "temperature": 2.0, "tempering": 0.5},
    )
    return config


def _resolved_continuation_config() -> dict:
    path = trace.ROOT / (
        "config/"
        "dgpo_omnifold_ztautau_10pct_h4_gradient_transfer_trace_resume20_to50.yaml"
    )
    config = read_overlay_yaml(path)
    trace.apply_epsilon_result(
        config,
        {"epsilon_rms": 3.0e-7, "temperature": 2.0, "tempering": 0.5},
    )
    return config


def _resolved_path(path: Path) -> dict:
    config = read_overlay_yaml(path)
    trace.apply_epsilon_result(
        config,
        {"epsilon_rms": 3.0e-7, "temperature": 2.0, "tempering": 0.5},
    )
    return config


def _training_contract_without_seed_or_provenance(config: dict) -> dict:
    payload = copy.deepcopy(config)
    for key in ("experiment", "nersc", "logger"):
        payload.pop(key, None)
    training = payload["options"]["Training"]
    training.pop("model_checkpoint_load_path", None)
    training.pop("model_checkpoint_save_path", None)
    dgpo = payload["dgpo"]
    dgpo["gradient_conflict"].pop("seed", None)
    dgpo["tarp"].pop("seed", None)
    adaptive = dgpo["adaptive_omnifold"]
    adaptive["trigger"].pop("probe_seed", None)
    adaptive["recalibration"].pop("pool_selection_seed", None)
    adaptive["recalibration"].pop("seed", None)
    return payload


def test_gradient_transfer_overlay_resolves_to_pinned_contract() -> None:
    config = _resolved_config()
    trace.assert_contract(config)
    assert config["logger"]["wandb"]["id"] == "h4xfer01"
    assert config["options"]["Training"]["total_epochs"] == 2
    assert config["dgpo"]["gradient_transfer_trace"]["update_end_steps"] == [
        1,
        2,
        5,
        10,
        20,
    ]
    assert config["dgpo"]["adaptive_omnifold"]["audit_fit"]["min_steps"] >= 1000
    assert config["dgpo"]["adaptive_omnifold"]["audit_fit"][
        "training_readiness"
    ] is None
    assert not config["dgpo"]["adaptive_omnifold"]["trigger"][
        "warm_start_classifier"
    ]


def test_step20_to50_continuation_resolves_to_pinned_contract() -> None:
    baseline = _resolved_config()
    continuation = _resolved_continuation_config()
    trace.assert_contract(continuation)
    assert continuation["logger"]["wandb"]["id"] == "h4xfer02"
    assert continuation["experiment"]["continuation_of"].endswith("/h4xfer01")
    assert trace._configured_seed_bundle(baseline) == trace.FIXED_SEED_BUNDLE
    assert trace._configured_seed_bundle(continuation) == trace.FIXED_SEED_BUNDLE
    assert continuation["dgpo"]["checkpoint_load_mode"] == "resume"
    assert not continuation["dgpo"]["adaptive_omnifold"]["recalibration"][
        "bootstrap_on_start"
    ]
    assert continuation["dgpo"]["gradient_transfer_trace"]["update_end_steps"] == [
        30,
        40,
        50,
    ]
    for keys in (
        ("options", "Training", "learning_rate"),
        ("options", "Training", "learning_rate_body"),
        ("dgpo", "K"),
        ("dgpo", "beta"),
        ("dgpo", "advantage_estimator"),
        ("dgpo", "reference_trust", "coefficient"),
        ("dgpo", "reference_trust", "objective"),
        ("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"),
        ("dgpo", "adaptive_omnifold", "recalibration", "crossfit_folds"),
        ("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"),
        ("dgpo", "adaptive_omnifold", "audit_fit", "min_steps"),
    ):
        left = baseline
        right = continuation
        for key in keys:
            left = left[key]
            right = right[key]
        assert right == left, ".".join(keys)


def test_seed2_replication_changes_only_seed_and_provenance() -> None:
    base_control = _resolved_config()
    continuation_control = _resolved_continuation_config()
    base_replication = _resolved_path(replication.PHASE20)
    continuation_replication = _resolved_path(replication.PHASE50)

    trace.assert_contract(base_replication)
    trace.assert_contract(continuation_replication)
    assert trace._configured_seed_bundle(base_replication) == (
        trace.INDEPENDENT_SEED_BUNDLE
    )
    assert trace._configured_seed_bundle(continuation_replication) == (
        trace.INDEPENDENT_SEED_BUNDLE
    )
    assert trace.INDEPENDENT_SEED_BUNDLE != trace.FIXED_SEED_BUNDLE
    assert base_replication["logger"]["wandb"]["id"] == "h4rep01"
    assert continuation_replication["logger"]["wandb"]["id"] == "h4rep01"
    assert continuation_replication["experiment"]["continuation_of"].endswith(
        "/h4rep01"
    )
    assert _training_contract_without_seed_or_provenance(
        base_replication
    ) == _training_contract_without_seed_or_provenance(base_control)
    assert _training_contract_without_seed_or_provenance(
        continuation_replication
    ) == _training_contract_without_seed_or_provenance(continuation_control)


def test_seed2_driver_runs_the_full_state_boundary_in_order(monkeypatch) -> None:
    calls = []
    published = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(replication.subprocess, "run", fake_run)
    monkeypatch.setattr(
        replication,
        "publish_endpoint",
        lambda: published.append(True) or {
            "late_window_mean_gap": 0.35,
            "step0_gap": 0.36,
            "delta_vs_step0": -0.01,
            "passed": True,
        },
    )
    assert replication.main([]) == 0
    assert str(replication.PHASE20) in calls[0][0]
    assert str(replication.PHASE50) in calls[1][0]
    assert all(call[1]["check"] for call in calls)
    assert published == [True]


def test_seed2_endpoint_uses_valid_cold_late_window() -> None:
    gaps = {0: 0.36, 10: 0.39, 20: 0.32, 30: 0.38, 40: 0.34, 50: 0.29}
    checkpoint = {
        "global_step": 50,
        "dgpo_adaptive_omnifold_state": {
            "probe_history": [
                {
                    "global_step": step,
                    "raw_auc": 0.5 + gap,
                    "raw_auc_gap": gap,
                    "raw_audit_training_steps": 1000 + step,
                    "raw_audit_saturated": 1.0,
                }
                for step, gap in gaps.items()
            ]
        },
    }
    report = replication.summarize_endpoint(checkpoint)
    assert report["late_window_mean_gap"] == pytest.approx((0.38 + 0.34 + 0.29) / 3)
    assert report["delta_vs_step0"] < 0
    assert report["passed"] is True


def test_seed2_endpoint_rejects_undertrained_audit() -> None:
    checkpoint = {
        "global_step": 50,
        "dgpo_adaptive_omnifold_state": {
            "probe_history": [
                {
                    "global_step": step,
                    "raw_auc": 0.8,
                    "raw_auc_gap": 0.3,
                    "raw_audit_training_steps": 999 if step == 40 else 1000,
                    "raw_audit_saturated": 1.0,
                }
                for step in replication.AUDIT_STEPS
            ]
        },
    }
    with pytest.raises(ValueError, match="step-40 cold H4 audit is invalid"):
        replication.summarize_endpoint(checkpoint)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dgpo", "reference_trust", "coefficient"), 0.2),
        (("dgpo", "gradient_transfer_trace", "update_end_steps"), [1, 10, 20]),
        (("dgpo", "adaptive_omnifold", "audit_fit", "min_steps"), 999),
        (("dgpo", "adaptive_omnifold", "trigger", "warm_start_classifier"), True),
        (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
        (("options", "Training", "total_epochs"), 3),
        (("logger", "wandb", "id"), "wrong"),
        (("dgpo", "adaptive_omnifold", "recalibration", "seed"), 7),
    ],
)
def test_contract_rejects_training_or_measurement_drift(path, value) -> None:
    config = copy.deepcopy(_resolved_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        trace.assert_contract(config)


def test_continuation_rejects_source_metadata_drift() -> None:
    config = copy.deepcopy(_resolved_continuation_config())
    config["experiment"]["source_reward_round_id"] = 2
    with pytest.raises(ValueError, match="source or update horizon"):
        trace.assert_contract(config)
