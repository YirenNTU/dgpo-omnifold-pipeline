from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evenet_dgpo"))

import train_dgpo_h4_function_trust as trust  # noqa: E402
from train_neutrino_backend import read_overlay_yaml  # noqa: E402
from RL.DGPO_neutrino.optimizer_transaction import (  # noqa: E402
    restore_adamw_state_from_transaction_,
    snapshot_adamw_state_for_transaction,
)


def _resolved_config() -> dict:
    config = read_overlay_yaml(trust.DEFAULT)
    trust.apply_epsilon_result(
        config,
        {"epsilon_rms": 3.0e-7, "temperature": 2.0, "tempering": 0.5},
    )
    return config


def _audit(step: int, gap: float, *, updates: int = 1200) -> dict:
    return {
        "global_step": step,
        "raw_auc": 0.5 + gap,
        "raw_auc_gap": gap,
        "raw_audit_training_steps": updates,
        "raw_audit_saturated": 1.0,
    }


def test_checked_in_function_trust_contract() -> None:
    config = _resolved_config()
    trust.assert_contract(config)
    boundary = config["dgpo"]["reference_trust"]["adaptive_boundary"]
    assert config["logger"]["wandb"]["id"] == "h4trust01"
    assert config["dgpo"]["reference_trust"]["coefficient"] == 0.0
    assert boundary["delta_max"] == pytest.approx(1.0e-4)
    assert boundary["transactional_rejection"]
    assert boundary["stop_after_rejection"]
    assert boundary["max_backtracks"] == 6


def test_rejection_endpoint_uses_boundary_audit_and_proposal_clock() -> None:
    checkpoint = {
        "global_step": 4,
        "dgpo_adaptive_omnifold_state": {
            "trust_rejection_stop_requested": True,
            "trust_first_rejected_global_step": 4,
            "trust_accepted_updates": 3,
            "probe_history": [
                _audit(0, trust.FIXED_BASELINE_GAP),
                _audit(4, 0.34),
            ],
        },
    }
    report = trust.summarize_endpoint(checkpoint)
    assert report["proposal_step"] == 4
    assert report["accepted_updates"] == 3
    assert report["stopped_at_rejection"]
    assert report["delta_vs_fixed_baseline"] == pytest.approx(
        0.34 - trust.FIXED_BASELINE_GAP
    )
    assert report["passed"]


def test_ten_accepted_proposals_are_a_valid_nonrejection_endpoint() -> None:
    checkpoint = {
        "global_step": 10,
        "dgpo_adaptive_omnifold_state": {
            "trust_rejection_stop_requested": False,
            "trust_first_rejected_global_step": -1,
            "trust_accepted_updates": 10,
            "probe_history": [
                _audit(0, trust.FIXED_BASELINE_GAP),
                _audit(10, 0.36),
            ],
        },
    }
    report = trust.summarize_endpoint(checkpoint)
    assert report["proposal_step"] == 10
    assert report["accepted_updates"] == 10
    assert not report["stopped_at_rejection"]


def test_endpoint_rejects_undertrained_or_mismatched_audit() -> None:
    checkpoint = {
        "global_step": 2,
        "dgpo_adaptive_omnifold_state": {
            "trust_rejection_stop_requested": True,
            "trust_first_rejected_global_step": 2,
            "trust_accepted_updates": 1,
            "probe_history": [
                _audit(0, trust.FIXED_BASELINE_GAP),
                _audit(2, 0.30, updates=999),
            ],
        },
    }
    with pytest.raises(ValueError, match="endpoint H4 audit is missing"):
        trust.summarize_endpoint(checkpoint)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("dgpo", "reference_trust", "coefficient"), 1.0),
        (
            (
                "dgpo",
                "reference_trust",
                "adaptive_boundary",
                "transactional_rejection",
            ),
            False,
        ),
        (
            ("dgpo", "reference_trust", "adaptive_boundary", "delta_max"),
            2.0e-4,
        ),
        (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
        (("dgpo", "adaptive_omnifold", "audit_fit", "min_steps"), 999),
        (("logger", "wandb", "id"), "wrong"),
    ],
)
def test_contract_rejects_causal_or_measurement_drift(path, value) -> None:
    config = copy.deepcopy(_resolved_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        trust.assert_contract(config)


def test_transactional_rejection_restores_complete_adamw_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.01)
    parameter.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    before = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[parameter].items()
    }
    snapshot = snapshot_adamw_state_for_transaction(optimizer)
    parameter.grad = torch.tensor([-1.5, 0.75])
    optimizer.step()
    restore_adamw_state_from_transaction_(optimizer, snapshot)

    restored = optimizer.state[parameter]
    assert restored.keys() == before.keys()
    for key, expected in before.items():
        if isinstance(expected, torch.Tensor):
            assert torch.equal(restored[key], expected)
        else:
            assert restored[key] == expected


def test_transaction_restores_empty_pre_step_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    snapshot = snapshot_adamw_state_for_transaction(optimizer)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    assert optimizer.state
    restore_adamw_state_from_transaction_(optimizer, snapshot)
    assert not optimizer.state
