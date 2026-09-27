#!/usr/bin/env python3
"""Compare the matched old/rank-2/rank-4 classifier actionability reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


EXPECTED = {"old": 0, "rank2": 2, "rank4": 4}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"report must be a JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def summarize(report: Mapping[str, Any], *, gate_radius: float) -> dict[str, Any]:
    arm = dict(report.get("series_arm") or {})
    name = str(arm.get("name", ""))
    if name not in EXPECTED:
        raise ValueError(f"unknown nested-residual arm: {name!r}")
    rank = int(arm.get("conditional_residual_rank", -1))
    if rank != EXPECTED[name]:
        raise ValueError(f"arm/rank mismatch for {name}: {rank}")

    classification = report["classification"]["fully_trained"]
    audit = classification["final_audit_uncalibrated"]
    radius_key = f"{float(gate_radius):.12g}"
    action = report["decision"]["stages"]["fully_trained"][radius_key]
    plus = action["plus_minus_zero"]
    delta = plus["plus_minus_zero"]

    phase = "old_base" if rank == 0 else f"rank{rank}_residual"
    stage_rows = [
        member["stage_diagnostics"][phase]
        for member in report["classifier_members"]
    ]
    training_ba = fmean(_finite(row["balanced_accuracy"], "training BA") for row in stage_rows)
    validation_ba = fmean(
        _finite(row["validation_balanced_accuracy"], "validation BA")
        for row in stage_rows
    )
    return {
        "arm": name,
        "conditional_residual_rank": rank,
        "heldout_auc": _finite(audit["auc"], "held-out AUC"),
        "heldout_auc_gap": abs(_finite(audit["auc"], "held-out AUC") - 0.5),
        "heldout_balanced_accuracy": _finite(
            audit["balanced_accuracy"], "held-out BA"
        ),
        "mean_training_balanced_accuracy": training_ba,
        "mean_validation_balanced_accuracy": validation_ba,
        "mean_train_validation_ba_gap": training_ba - validation_ba,
        "gate_rms": float(gate_radius),
        "mean_plus_gap_delta": _finite(delta["mean"], "plus delta"),
        "plus_gap_delta_ci90_low": _finite(delta["ci90_low"], "plus CI low"),
        "plus_gap_delta_ci90_high": _finite(delta["ci90_high"], "plus CI high"),
        "plus_beats_zero_fraction": _finite(
            plus["plus_beats_zero_fraction"], "plus-beats-zero fraction"
        ),
        "plus_beats_minus_fraction": _finite(
            plus["plus_beats_minus_fraction"], "plus-beats-minus fraction"
        ),
        "actionability_reliable": bool(action["reliable"]),
        "candidate_advantage_cosine_h4": _finite(
            report["signal_alignment_with_independent_judge"]["fully_trained"][
                "advantage_cosine"
            ],
            "candidate advantage cosine",
        ),
        "policy_gradient_cosine_h4": _finite(
            report["gradient_alignment_with_independent_judge"]["fully_trained"],
            "policy gradient cosine",
        ),
    }


def compare(reports: Mapping[str, Mapping[str, Any]], *, gate_radius: float) -> dict[str, Any]:
    if set(reports) != set(EXPECTED):
        raise ValueError(f"reports must contain exactly {sorted(EXPECTED)}")
    source = {str(report.get("source_wandb_run")) for report in reports.values()}
    steps = {int(report.get("policy_global_step", -1)) for report in reports.values()}
    judges = {
        json.dumps(report.get("judge_classifier_overrides"), sort_keys=True)
        for report in reports.values()
    }
    if len(source) != 1 or len(steps) != 1 or len(judges) != 1:
        raise ValueError("series reports do not share source, policy step, and H4 judge")

    rows = {
        name: summarize(report, gate_radius=gate_radius)
        for name, report in reports.items()
    }
    old = rows["old"]
    eligible = []
    for name in ("rank2", "rank4"):
        row = rows[name]
        row["improves_heldout_discrimination_over_old"] = bool(
            row["heldout_auc_gap"] > old["heldout_auc_gap"]
        )
        row["improves_actionability_over_old"] = bool(
            row["mean_plus_gap_delta"] < old["mean_plus_gap_delta"]
        )
        row["positive_h4_alignment"] = bool(
            row["candidate_advantage_cosine_h4"] > 0.0
            and row["policy_gradient_cosine_h4"] > 0.0
        )
        row["eligible_for_closed_loop_pilot"] = bool(
            row["improves_heldout_discrimination_over_old"]
            and row["improves_actionability_over_old"]
            and row["actionability_reliable"]
            and row["positive_h4_alignment"]
        )
        if row["eligible_for_closed_loop_pilot"]:
            eligible.append(name)

    selected = eligible[0] if eligible else None
    return {
        "schema": "nested-residual-critic-series-comparison-v1",
        "source_wandb_run": next(iter(source)),
        "policy_global_step": next(iter(steps)),
        "gate_rms": float(gate_radius),
        "arms": rows,
        "selected_arm": selected,
        "finding": (
            "nested_residual_supported" if selected is not None
            else "no_nested_residual_arm_passed"
        ),
        "selection_rule": (
            "Choose the lowest rank that improves held-out AUC gap and H4-judged "
            "plus-step delta over old, passes the predeclared multi-seed "
            "actionability gate, and has positive candidate/gradient H4 alignment."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--rank2", type=Path, required=True)
    parser.add_argument("--rank4", type=Path, required=True)
    parser.add_argument("--gate-rms", type=float, default=3.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        {
            "old": _read(args.old),
            "rank2": _read(args.rank2),
            "rank4": _read(args.rank4),
        },
        gate_radius=args.gate_rms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
