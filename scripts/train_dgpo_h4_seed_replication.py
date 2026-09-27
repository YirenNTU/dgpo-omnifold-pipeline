#!/usr/bin/env python3
"""Run the two-phase independent-seed H4 classifier-closure replication."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
TRACE_LAUNCHER = ROOT / "scripts/train_dgpo_h4_gradient_transfer_trace.py"
PHASE20 = ROOT / (
    "config/"
    "dgpo_omnifold_ztautau_10pct_h4_gradient_transfer_"
    "seed2_replication_20step.yaml"
)
PHASE50 = ROOT / (
    "config/"
    "dgpo_omnifold_ztautau_10pct_h4_gradient_transfer_"
    "seed2_replication_resume20_to50.yaml"
)
FINAL_ROOT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "c4a91e07_h4_seed2_replication_resume20_to50_v1"
)
FINAL_CHECKPOINT = FINAL_ROOT / "checkpoints/last.ckpt"
FINAL_REPORT = FINAL_ROOT / "replication_endpoint.json"
AUDIT_STEPS = (0, 10, 20, 30, 40, 50)


def _run_phase(config: Path, *, epsilon_report: Path | None) -> None:
    command = [sys.executable, str(TRACE_LAUNCHER), "--config", str(config)]
    if epsilon_report is not None:
        command.extend(("--epsilon-report", str(epsilon_report)))
    subprocess.run(command, cwd=ROOT, check=True)


def summarize_endpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the cold H4 trajectory and compute its predeclared endpoint."""

    if int(checkpoint.get("global_step", -1)) != 50:
        raise ValueError("h4rep01 endpoint checkpoint must be global_step=50")
    history = list(
        (checkpoint.get("dgpo_adaptive_omnifold_state") or {}).get(
            "probe_history", []
        )
        or []
    )
    by_step: dict[int, dict[str, Any]] = {}
    for row in history:
        if not isinstance(row, Mapping):
            continue
        try:
            step = int(row.get("global_step", -1))
            gap = float(row.get("raw_auc_gap", float("nan")))
        except (TypeError, ValueError):
            continue
        if step in AUDIT_STEPS and math.isfinite(gap):
            by_step[step] = dict(row)
    missing = [step for step in AUDIT_STEPS if step not in by_step]
    if missing:
        raise ValueError(f"h4rep01 checkpoint lacks cold H4 audits at steps {missing}")

    audits: dict[str, dict[str, Any]] = {}
    for step in AUDIT_STEPS:
        row = by_step[step]
        updates = int(row.get("raw_audit_training_steps", 0))
        saturated = float(row.get("raw_audit_saturated", 0.0)) >= 0.5
        auc = float(row.get("raw_auc", float("nan")))
        gap = float(row["raw_auc_gap"])
        if not math.isfinite(auc) or not math.isfinite(gap):
            raise ValueError(f"step-{step} cold H4 audit is nonfinite")
        if updates < 1000 or not saturated:
            raise ValueError(
                f"step-{step} cold H4 audit is invalid: "
                f"updates={updates}, saturated={saturated}"
            )
        audits[str(step)] = {
            "raw_auc": auc,
            "raw_auc_gap": gap,
            "raw_audit_training_steps": updates,
            "raw_audit_saturated": True,
        }

    late_mean = (
        sum(audits[str(step)]["raw_auc_gap"] for step in (30, 40, 50)) / 3.0
    )
    step0_gap = float(audits["0"]["raw_auc_gap"])
    delta = late_mean - step0_gap
    return {
        "schema": "c4a91e07-h4-seed2-replication-endpoint-v1",
        "wandb_run_id": "h4rep01",
        "source_policy_step": 1110,
        "policy_endpoint_step": 50,
        "primary_endpoint": "mean_gap30_gap40_gap50_minus_gap0",
        "cold_h4_audits": audits,
        "late_window_mean_gap": late_mean,
        "step0_gap": step0_gap,
        "delta_vs_step0": delta,
        "passed": delta < 0.0,
        "classifier_audit_min_updates": 1000,
        "objective_changed": False,
    }


def _load_final_checkpoint() -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(
            FINAL_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        payload = torch.load(
            FINAL_CHECKPOINT, map_location="cpu", weights_only=False
        )
    if not isinstance(payload, dict):
        raise ValueError("h4rep01 endpoint checkpoint is not a mapping")
    return payload


def publish_endpoint() -> dict[str, Any]:
    """Persist and publish the completed replication decision to W&B."""

    report = summarize_endpoint(_load_final_checkpoint())
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if FINAL_REPORT.exists():
        if FINAL_REPORT.read_text() != encoded:
            raise ValueError(
                "existing h4rep01 endpoint report disagrees with checkpoint"
            )
    else:
        FINAL_REPORT.write_text(encoded)

    import wandb

    run = wandb.init(
        entity="ytchou97-university-of-washington",
        project="nu2flow-RL",
        id="h4rep01",
        resume="allow",
        name="c4a91e07_h4_seed2_replication_50step_v1",
        job_type="replication-decision",
    )
    metrics: dict[str, float] = {
        "replication/primary/late_window_mean_gap": float(
            report["late_window_mean_gap"]
        ),
        "replication/primary/step0_gap": float(report["step0_gap"]),
        "replication/primary/delta_vs_step0": float(report["delta_vs_step0"]),
        "replication/primary/passed": float(report["passed"]),
    }
    for raw_step, row in report["cold_h4_audits"].items():
        prefix = f"replication/cold_h4/step_{raw_step}"
        metrics[f"{prefix}/raw_auc"] = float(row["raw_auc"])
        metrics[f"{prefix}/raw_auc_gap"] = float(row["raw_auc_gap"])
        metrics[f"{prefix}/training_steps"] = float(
            row["raw_audit_training_steps"]
        )
        metrics[f"{prefix}/saturated"] = 1.0
    run.log(metrics)
    run.summary.update(
        {
            "replication/status": "passed" if report["passed"] else "failed",
            "replication/primary_endpoint": report["primary_endpoint"],
            "replication/primary/delta_vs_step0": report["delta_vs_step0"],
            "replication/primary/late_window_mean_gap": report[
                "late_window_mean_gap"
            ],
            "replication/primary/step0_gap": report["step0_gap"],
            "replication/all_required_audits_valid": True,
        }
    )
    artifact = wandb.Artifact(
        "c4a91e07-h4-seed2-replication-endpoint", type="diagnostic"
    )
    artifact.add_file(str(FINAL_REPORT))
    run.log_artifact(artifact)
    run.finish()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "20", "50"),
        default="all",
        help=(
            "Run both phases, only the weights-only 0->20 phase, or only the "
            "full-state 20->50 phase."
        ),
    )
    parser.add_argument("--epsilon-report", type=Path)
    args = parser.parse_args(argv)
    epsilon_report = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else None
    )

    if args.phase in {"all", "20"}:
        print(
            "[h4rep01] phase 1/2: independent seed2 weights-only step 0->20",
            flush=True,
        )
        _run_phase(PHASE20, epsilon_report=epsilon_report)
    if args.phase in {"all", "50"}:
        print(
            "[h4rep01] phase 2/2: preserve full step-20 state and continue 20->50",
            flush=True,
        )
        _run_phase(PHASE50, epsilon_report=epsilon_report)
        report = publish_endpoint()
        print(
            "[h4rep01] predeclared endpoint: "
            f"late_mean={report['late_window_mean_gap']:.6f}, "
            f"gap0={report['step0_gap']:.6f}, "
            f"delta={report['delta_vs_step0']:+.6f}, "
            f"passed={report['passed']}",
            flush=True,
        )
    print(
        "[h4rep01] requested replication phase(s) completed; results are in live W&B h4rep01.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
