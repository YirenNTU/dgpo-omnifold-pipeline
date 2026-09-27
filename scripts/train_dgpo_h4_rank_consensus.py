#!/usr/bin/env python3
"""Run the matched fixed-rank-audit and consensus-gated BA70 experiments."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_ba70_closed_loop import (  # noqa: E402
    AUDIT_STEPS,
    SOURCE_CHECKPOINT,
    assert_contract as assert_ba70_contract,
    verify_paths,
)
from train_dgpo_h4_fresh_ensemble_pilot import (  # noqa: E402
    EXPECTED_SOURCE_RUN,
    EXPECTED_SOURCE_STEP,
    _load_checkpoint,
    apply_epsilon_result,
    assert_wandb_runtime_ready,
    read_epsilon_result,
)
from train_dgpo_old_method_10pct import assert_live_ray_16gpu  # noqa: E402
from train_neutrino_backend import read_overlay_yaml  # noqa: E402


BASE = ROOT / "config/train_diffusion_nersc.yaml"
AUDIT_CONFIG = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba70_rank_audit_50step.yaml"
)
GATED_CONFIG = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba70_consensus_gate_50step.yaml"
)
DEFAULT = AUDIT_CONFIG
VARIANTS = {
    "h4-ba70-fixed-rank-audit-50step-v1": {
        "config": AUDIT_CONFIG,
        "wandb_id": "h4ranka1",
        "apply_to_reward": False,
    },
    "h4-ba70-consensus-gate-50step-v1": {
        "config": GATED_CONFIG,
        "wandb_id": "h4consg1",
        "apply_to_reward": True,
    },
}


def _as_baseline_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the proven BA70 contract after restoring its audit-only metadata."""

    baseline = copy.deepcopy(dict(config))
    baseline["experiment"].update(
        {
            "protocol": "h4-ba70-m4-ess15-refit10-keepadam-50step-audit1-v1",
            "rounds": 5,
            "policy_updates_per_round": 10,
            "reward_lifetime_updates": 10,
            "reward_refresh_steps": [0, 10, 20, 30, 40],
            "audit_steps": AUDIT_STEPS,
            "endpoint_global_step": 50,
            "endpoint_selection": "fixed_step_50",
            "reward_members": 4,
            "audit_repeats": 1,
            "primary_endpoint": "staleness/raw_auc_gap",
            "primary_direction": "lower",
        }
    )
    baseline["logger"]["wandb"]["id"] = "h4b70c50"
    baseline["dgpo"]["adaptive_omnifold"]["trigger"][
        "fixed_schedule_log_raw_audit"
    ] = True
    baseline["reward_config"]["omnifold"].pop("candidate_consensus", None)
    return baseline


def _trajectory_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove declared treatment/provenance fields before matched-arm comparison."""

    payload = copy.deepcopy(dict(config))
    payload.pop("experiment", None)
    payload.pop("logger", None)
    nersc = payload.get("nersc", {})
    nersc.pop("reproducibility", None)
    nersc.pop("ray", None)
    nersc.pop("execution", None)
    payload["options"]["Training"].pop("model_checkpoint_save_path", None)
    consensus = payload["reward_config"]["omnifold"]["candidate_consensus"]
    consensus.pop("apply_to_reward", None)
    return payload


def assert_matched_configs() -> None:
    audit = read_overlay_yaml(AUDIT_CONFIG)
    gated = read_overlay_yaml(GATED_CONFIG)
    if _trajectory_payload(audit) != _trajectory_payload(gated):
        raise ValueError(
            "audit and treatment configs differ outside apply_to_reward/provenance"
        )


def assert_contract(config: Mapping[str, Any]) -> None:
    protocol = str(config["experiment"].get("protocol", ""))
    if protocol not in VARIANTS:
        raise ValueError(f"unsupported rank-consensus protocol: {protocol}")
    variant = VARIANTS[protocol]
    assert_ba70_contract(_as_baseline_contract(config))
    assert_matched_configs()

    if int(config["dgpo"].get("K", 0)) != 8:
        raise ValueError("rank-consensus protocol requires exactly K=8 candidates")

    trigger = config["dgpo"]["adaptive_omnifold"]["trigger"]
    if trigger.get("fixed_schedule_log_raw_audit") is not False:
        raise ValueError(
            "rank-consensus experiments skip the redundant raw classifier audit"
        )
    consensus = config["reward_config"]["omnifold"].get(
        "candidate_consensus"
    )
    expected = {
        "enabled": True,
        "apply_to_reward": bool(variant["apply_to_reward"]),
        "temporal_history_rounds": 1,
        "minimum_sign_agreement": 0.75,
        "uncertainty_scale": 1.0,
        "epsilon": 1.0e-8,
        "fixed_panel_events_per_rank": 128,
        "fixed_panel_candidates": 8,
    }
    if consensus != expected:
        raise ValueError(
            f"candidate-consensus protocol mismatch: {consensus!r} != {expected!r}"
        )
    if config["logger"]["wandb"].get("id") != variant["wandb_id"]:
        raise ValueError("rank-consensus W&B id is not pinned")
    output = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    if protocol not in str(output.parent):
        # Directory names are intentionally descriptive instead of protocol
        # strings; require the distinguishing audit/treatment token below.
        token = "consensus_gate" if variant["apply_to_reward"] else "fixed_rank_audit"
        if token not in str(output.parent):
            raise ValueError("rank-consensus output directory is not isolated")


def _write_exact(path: Path, payload: Any, *, checkpoints_exist: bool) -> None:
    serialized = (
        yaml.safe_dump(payload, sort_keys=False)
        if path.suffix == ".yaml"
        else json.dumps(payload, indent=2) + "\n"
    )
    if path.exists() and path.read_text() != serialized and checkpoints_exist:
        raise ValueError(f"{path.name} changed after a checkpoint was written")
    path.write_text(serialized)


def _rank_audit_report(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    stack = checkpoint.get("dgpo_omnifold_reward_stack") or {}
    rows = [dict(row) for row in stack.get("rank_audit_history", [])]
    observed = [int(row.get("reward_round_id", -1)) for row in rows]
    if observed != [1, 2, 3, 4, 5]:
        raise RuntimeError(
            f"expected fixed-panel rank audits for rounds 1..5, got {observed}"
        )
    if any(int(row.get("panel_events_per_rank", 0)) != 128 for row in rows):
        raise RuntimeError("fixed rank-audit panel size changed across rounds")
    if any(int(row.get("candidates_per_event", 0)) != 8 for row in rows):
        raise RuntimeError("fixed rank-audit candidate count changed across rounds")
    if float(rows[0].get("temporal_available", -1.0)) != 0.0 or any(
        float(row.get("temporal_available", 0.0)) != 1.0 for row in rows[1:]
    ):
        raise RuntimeError("adjacent-round rank audit is incomplete")
    return {
        "schema": "fixed-candidate-adjacent-reward-rank-audit-v1",
        "policy_source": EXPECTED_SOURCE_RUN,
        "policy_source_step": EXPECTED_SOURCE_STEP,
        "fixed_panel_events_per_rank": 128,
        "wandb_fixed_panel_events_total": 2048,
        "candidates_per_event": 8,
        "wandb_rank_reduction": "finite_scalar_mean_across_16_ranks",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    supported = {Path(row["config"]).resolve() for row in VARIANTS.values()}
    if config_path not in supported:
        parser.error(
            "--config must be one of: "
            + ", ".join(str(path) for path in sorted(supported))
        )
    config = read_overlay_yaml(config_path)
    protocol = str(config["experiment"]["protocol"])
    variant = VARIANTS[protocol]
    report_path = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else Path(config["experiment"]["epsilon_sweep_report"])
        .expanduser()
        .resolve()
    )
    epsilon = read_epsilon_result(report_path, config["experiment"])
    apply_epsilon_result(config, epsilon)
    assert_contract(config)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)

    treatment = bool(variant["apply_to_reward"])
    print(
        "Rank-consensus preflight passed: " + mode + "\n"
        f"source={SOURCE_CHECKPOINT} step={EXPECTED_SOURCE_STEP}; "
        f"epsilon/LR={epsilon['epsilon_rms']:.6g}\n"
        "Protocol: five fresh BA70 H4 2x2 rewards, 10 committed DGPO updates "
        "per reward, K=8, one checkpointed 128-event/rank fixed candidate "
        f"panel, one-round temporal history, apply_to_reward={treatment}. "
        f"Live W&B id={variant['wandb_id']}.",
        flush=True,
    )
    if args.check_only:
        return 0

    assert_wandb_runtime_ready()
    assert_live_ray_16gpu()
    checkpoint_dir = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    output_root = checkpoint_dir.parent
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_exist = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt"))
    resolved = output_root / "resolved_overlay.yaml"
    _write_exact(resolved, config, checkpoints_exist=checkpoints_exist)
    manifest = {
        "schema": "h4-ba70-rank-consensus-v1",
        "protocol": protocol,
        "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "epsilon_report": str(report_path),
        **epsilon,
        "reward_members_per_round": 4,
        "reward_rounds": 5,
        "policy_updates_per_round": 10,
        "fixed_panel_events_per_rank": 128,
        "wandb_fixed_panel_events_total": 2048,
        "candidates_per_event": 8,
        "wandb_rank_reduction": "finite_scalar_mean_across_16_ranks",
        "temporal_history_rounds": 1,
        "minimum_sign_agreement": 0.75,
        "uncertainty_scale": 1.0,
        "consensus_applied_to_reward": treatment,
        "raw_classifier_audit_run": False,
        "endpoint_global_step": 50,
        "wandb_run": (
            f"{config['logger']['wandb']['entity']}/"
            f"{config['logger']['wandb']['project']}/"
            f"{variant['wandb_id']}"
        ),
    }
    _write_exact(
        output_root / "experiment_manifest.json",
        manifest,
        checkpoints_exist=checkpoints_exist,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_neutrino_backend.py"),
            "--backend",
            "dgpo-evenet",
            "--base-config",
            str(BASE),
            "--overlay-config",
            str(resolved),
            "--",
            "--ray-dir",
            str(config["nersc"]["ray"]["results_dir"]),
        ],
        cwd=ROOT,
        check=True,
    )
    final = _load_checkpoint(checkpoint_dir / "last.ckpt")
    if int(final.get("global_step", -1)) != 50 or int(
        final.get("dgpo_reward_round_id", -1)
    ) != 5:
        raise RuntimeError("rank-consensus run did not finish step 50 / round 5")
    report = _rank_audit_report(final)
    report["consensus_applied_to_reward"] = treatment
    _write_exact(
        output_root / "rank_audit.json",
        report,
        checkpoints_exist=False,
    )
    print(
        f"Rank-consensus run complete: step=50 rounds=5 treatment={treatment}; "
        f"report={output_root / 'rank_audit.json'}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
