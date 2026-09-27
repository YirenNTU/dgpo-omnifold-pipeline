from __future__ import annotations

import math

import pytest
import torch

from .gradient_transfer import (
    resolve_gradient_transfer_trace_config,
    summarize_gradient_transfer,
    trace_due,
)


def test_trace_config_resolves_schedule_and_rejects_unknown_options() -> None:
    cfg = resolve_gradient_transfer_trace_config(
        {
            "enabled": True,
            "update_end_steps": [1, 2, 10],
            "counterfactual_trust_coefficients": [0.0, 0.2, 1.0],
        }
    )
    assert cfg.update_end_steps == (1, 2, 10)
    assert cfg.counterfactual_trust_coefficients == (0.0, 0.2, 1.0)
    assert trace_due(cfg, update_end_step=2)
    assert not trace_due(cfg, update_end_step=3)
    with pytest.raises(ValueError, match="unknown"):
        resolve_gradient_transfer_trace_config({"enabled": True, "typo": 1})


def test_summary_detects_reference_cancellation_and_adamw_direction() -> None:
    h4 = torch.tensor([1.0, 0.0])
    reference = torch.tensor([-2.0, 1.0])
    total = h4 + reference
    metrics = summarize_gradient_transfer(
        h4_gradient=h4,
        reference_gradient=reference,
        actual_unclipped_gradient=total,
        trust_coefficient=1.0,
        counterfactual_coefficients=(0.0, 0.1, 0.2, 0.5, 1.0),
        update_start_step=9,
        update_end_step=10,
        norm_floor=1.0e-12,
        adamw_descent=torch.tensor([1.0, 0.0]),
    )

    assert metrics["gradient_transfer/total_on_h4/projection_ratio"] == -1.0
    assert metrics["gradient_transfer/critical_lambda/defined"] == 1.0
    assert metrics["gradient_transfer/critical_lambda/value"] == 0.5
    assert metrics[
        "gradient_transfer/counterfactual/lambda_0/total_on_h4_projection_ratio"
    ] == 1.0
    assert metrics[
        "gradient_transfer/counterfactual/lambda_0p5/total_on_h4_projection_ratio"
    ] == 0.0
    assert metrics[
        "gradient_transfer/counterfactual/lambda_1/total_on_h4_projection_ratio"
    ] == -1.0
    assert metrics["gradient_transfer/reconstruction_actual/relative_error"] == 0.0
    assert math.isclose(
        metrics["gradient_transfer/adamw_descent_on_h4/cosine"], 1.0
    )
