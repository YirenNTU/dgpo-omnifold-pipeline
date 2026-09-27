#!/usr/bin/env python3
"""Run the seed-2 H4-only functional trust-region boundary experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_fresh_ensemble_pilot import (  # noqa: E402
    RECOVERY_KEYS,
    apply_epsilon_result,
    read_epsilon_result,
)
from train_dgpo_h4_gradient_transfer_trace import (  # noqa: E402
    INDEPENDENT_SEED_BUNDLE,
    _configured_seed_bundle,
)
from train_dgpo_old_method_10pct import assert_live_ray_16gpu  # noqa: E402
from train_neutrino_backend import read_overlay_yaml  # noqa: E402


DEFAULT = ROOT / (
    "config/"
    "dgpo_omnifold_ztautau_10pct_h4_seed2_function_trust_10proposal.yaml"
)
BASE = ROOT / "config/train_diffusion_nersc.yaml"
PROTOCOL = "h4-seed2-function-trust-10proposal-v1"
WANDB_ID = "h4trust01"
SOURCE = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "c4a91e07_h4_seed2_replication_20step_v1/checkpoints/"
    "dgpo-epoch=-1-next_ep=0-step=0.ckpt"
)
OUTPUT_ROOT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "c4a91e07_h4_seed2_function_trust_10proposal_v1"
)
FINAL_CHECKPOINT = OUTPUT_ROOT / "checkpoints/last.ckpt"
FINAL_REPORT = OUTPUT_ROOT / "function_trust_endpoint.json"
FIXED_BASELINE_GAP = 0.3711905
CONTROL_STEP10_GAP = 0.4083256
MAX_PROPOSALS = 10


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _valid_cold_audits(checkpoint: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    history = list(
        (checkpoint.get("dgpo_adaptive_omnifold_state") or {}).get(
            "probe_history", []
        )
        or []
    )
    result: dict[int, dict[str, Any]] = {}
    for item in history:
        if not isinstance(item, Mapping):
            continue
        try:
            step = int(item.get("global_step", -1))
            auc = float(item.get("raw_auc", float("nan")))
            gap = float(item.get("raw_auc_gap", float("nan")))
            updates = int(item.get("raw_audit_training_steps", 0))
            saturated = float(item.get("raw_audit_saturated", 0.0)) >= 0.5
        except (TypeError, ValueError):
            continue
        if (
            step >= 0
            and math.isfinite(auc)
            and math.isfinite(gap)
            and updates >= 1000
            and saturated
        ):
            result[step] = {
                "raw_auc": auc,
                "raw_auc_gap": gap,
                "raw_audit_training_steps": updates,
                "raw_audit_saturated": True,
            }
    return result


def summarize_endpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the boundary audit and apply the predeclared decision rule."""

    proposal_step = int(checkpoint.get("global_step", -1))
    if not 1 <= proposal_step <= MAX_PROPOSALS:
        raise ValueError(
            f"h4trust01 endpoint proposal step must lie in [1, 10], got {proposal_step}"
        )
    state = checkpoint.get("dgpo_adaptive_omnifold_state") or {}
    stopped_at_rejection = bool(state.get("trust_rejection_stop_requested", False))
    first_rejected = int(state.get("trust_first_rejected_global_step", -1))
    accepted_updates = int(state.get("trust_accepted_updates", 0))
    if stopped_at_rejection:
        if first_rejected != proposal_step:
            raise ValueError(
                "transactional rejection step does not match the endpoint proposal clock"
            )
        if accepted_updates >= proposal_step:
            raise ValueError("rejection endpoint must contain fewer accepted updates than proposals")
    elif proposal_step != MAX_PROPOSALS:
        raise ValueError("a non-rejection endpoint must complete all ten proposals")
    elif accepted_updates != MAX_PROPOSALS:
        raise ValueError("ten-proposal endpoint must contain ten accepted updates")

    audits = _valid_cold_audits(checkpoint)
    if 0 not in audits:
        raise ValueError("inherited h4rep01 step-0 cold H4 audit is missing or invalid")
    if proposal_step not in audits:
        raise ValueError(
            f"cold saturated endpoint H4 audit is missing at proposal step {proposal_step}"
        )
    baseline = audits[0]
    endpoint = audits[proposal_step]
    if not math.isclose(
        float(baseline["raw_auc_gap"]),
        FIXED_BASELINE_GAP,
        rel_tol=0.0,
        abs_tol=5.0e-6,
    ):
        raise ValueError(
            "inherited seed-2 baseline gap changed: "
            f"expected {FIXED_BASELINE_GAP}, got {baseline['raw_auc_gap']}"
        )
    delta = float(endpoint["raw_auc_gap"]) - FIXED_BASELINE_GAP
    return {
        "schema": "c4a91e07-h4-seed2-function-trust-endpoint-v1",
        "wandb_run_id": WANDB_ID,
        "source_wandb_run": "h4rep01",
        "source_policy_step": 1110,
        "source_replication_global_step": 0,
        "proposal_step": proposal_step,
        "accepted_updates": accepted_updates,
        "stopped_at_rejection": stopped_at_rejection,
        "first_rejected_proposal_step": first_rejected,
        "fixed_velocity_mse_ratio_radius": 1.0e-4,
        "baseline_cold_h4": baseline,
        "endpoint_cold_h4": endpoint,
        "fixed_baseline_gap": FIXED_BASELINE_GAP,
        "matched_control_step10_gap": CONTROL_STEP10_GAP,
        "delta_vs_fixed_baseline": delta,
        "passed": delta < 0.0,
        "classifier_audit_min_updates": 1000,
        "h4_reward_formula_changed": False,
        "dgpo_candidate_objective_changed": False,
        "reference_formulation_changed": True,
    }


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    training = config["options"]["Training"]
    dgpo = config["dgpo"]
    trust = dgpo["reference_trust"]
    boundary = trust["adaptive_boundary"]
    adaptive = dgpo["adaptive_omnifold"]
    trigger = adaptive["trigger"]
    audit = adaptive["audit_fit"]
    recal = adaptive["recalibration"]

    if experiment.get("protocol") != PROTOCOL:
        raise ValueError(f"expected experiment.protocol={PROTOCOL}")
    if (
        experiment.get("single_experimental_change")
        != "additive_reference_penalty_to_function_space_constraint"
        or experiment.get("h4_reward_formula_changed") is not False
        or experiment.get("dgpo_candidate_objective_changed") is not False
        or experiment.get("reference_formulation_changed") is not True
    ):
        raise ValueError("the one-question objective declaration changed")
    if (
        int(experiment.get("source_replication_global_step", -1)) != 0
        or int(experiment.get("source_reward_round_id", -1)) != 1
        or int(experiment.get("max_policy_proposals", -1)) != MAX_PROPOSALS
        or not math.isclose(
            float(experiment.get("fixed_baseline_gap", float("nan"))),
            FIXED_BASELINE_GAP,
        )
    ):
        raise ValueError("source boundary or predeclared endpoint changed")
    if _configured_seed_bundle(config) != INDEPENDENT_SEED_BUNDLE:
        raise ValueError("h4trust01 must preserve the h4rep01 seed-2 bundle")
    if (
        int(training.get("total_epochs", -1)) != 1
        or int(training.get("epochs", -1)) != 1
        or int(dgpo.get("steps_per_epoch", -1)) != MAX_PROPOSALS
    ):
        raise ValueError("h4trust01 requires at most ten proposals in one epoch")
    if Path(str(training.get("model_checkpoint_load_path"))) != SOURCE:
        raise ValueError("h4trust01 must branch from the exact h4rep01 step-0 snapshot")
    if (
        dgpo.get("checkpoint_load_mode") != "resume"
        or dgpo.get("auto_resume_from_last") is not True
        or Path(str(dgpo.get("auto_resume_fallback_checkpoint_path"))) != SOURCE
    ):
        raise ValueError("the installed H4/reference/AdamW state must be full-state resumed")
    if (
        dgpo.get("advantage_estimator") != "leave_one_out_unscaled"
        or int(dgpo.get("K", -1)) != 8
        or float(dgpo.get("beta", float("nan"))) != 1.0
        or float(dgpo.get("beta_kl", float("nan"))) != 0.0
    ):
        raise ValueError("the H4 DGPO candidate objective changed")
    if not (
        trust.get("enabled") is True
        and trust.get("objective") == "velocity_mse"
        and float(trust.get("coefficient", float("nan"))) == 0.0
    ):
        raise ValueError("h4trust01 requires H4-only backward with no additive reference")
    required_boundary = {
        "enabled": True,
        "radius_mode": "fixed",
        "distance": "velocity_mse_ratio",
        "delta_max": 1.0e-4,
        "delta_floor": 1.0e-4,
        "transactional_rejection": True,
        "stop_after_rejection": True,
        "enforcement": "post_step_backtracking",
        "backtrack_factor": 0.5,
        "max_backtracks": 6,
        "probe_events_per_rank": 256,
        "fixed_probe_per_reward_round": True,
        "interior_fraction": 1.0,
    }
    mismatched = {
        key: (boundary.get(key), expected)
        for key, expected in required_boundary.items()
        if boundary.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"functional trust contract changed: {mismatched}")
    if boundary.get("reset_adam_first_moment_on_zero_step"):
        raise ValueError("transactional rejection must restore, not reset, AdamW state")
    calibration = boundary.get("radius_calibration") or {}
    if any(
        bool(calibration.get(key, False))
        for key in (
            "enabled",
            "allow_expansion",
            "cross_round_nonexpanding",
            "scale_policy_lr",
            "refit_on_exhaustion",
            "round_acceptance_enabled",
            "trajectory_search_enabled",
        )
    ):
        raise ValueError("h4trust01 uses one fixed radius with no outer controller")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("no second projection constraint is allowed")
    if (dgpo.get("gradient_transfer_trace") or {}).get("enabled"):
        raise ValueError("the incompatible additive-reference gradient trace must be off")
    if dgpo.get("fail_on_skipped_optimizer_step") is not False:
        raise ValueError("an intentional alpha=0 rejection must be an allowed endpoint")
    if (
        dgpo.get("train_dist_enabled") is not False
        or dgpo.get("validation_initial_enabled") is not False
        or int(dgpo.get("validation_every_n_epochs", -1)) != 2
        or int(dgpo.get("validation_full_every_n_epochs", -1)) != 2
    ):
        raise ValueError(
            "h4trust01 keeps redundant train/validation panels off; the cold H4 "
            "audit is its endpoint"
        )
    if not (
        adaptive.get("enabled") is True
        and adaptive.get("monitor_mode") == "raw_plateau_refit"
        and adaptive.get("fixed_audit_panel") is True
        and adaptive.get("cache_event_inputs") is True
        and int(adaptive.get("staleness_every_n_epochs", -1)) == 1
        and adaptive.get("staleness_every_n_steps") is None
    ):
        raise ValueError("the fixed cold-audit panel or endpoint cadence changed")
    if (
        trigger.get("warm_start_classifier")
        or trigger.get("rollback_to_best_on_plateau")
        or not trigger.get("require_audit_saturation")
        or int(audit.get("min_steps", 0)) < 1000
        or not audit.get("fail_if_unsaturated")
        or audit.get("training_readiness") is not None
    ):
        raise ValueError("endpoint H4 must be cold, saturated, and >=1000 updates")
    if (
        recal.get("bootstrap_on_start")
        or recal.get("refit_once_on_resume")
        or int(recal.get("max_reward_rounds", -1)) != 1
        or recal.get("reset_optimizer_state_on_install")
        or recal.get("reset_adam_first_moment_on_install")
    ):
        raise ValueError("the inherited four-member reward must remain fixed")
    h4 = {
        "periodic_pair_features": True,
        "topology_fourier_embedding": True,
        "topology_conditioning": False,
        "visible_pair_rest_frame": False,
        "topology_max_harmonic": 4,
        "topology_include_theta_pair": False,
        "topology_direct_logit": False,
    }
    for label, block in (("reward", recal), ("cold audit", audit)):
        mismatch = {
            key: (block.get(key), expected)
            for key, expected in h4.items()
            if block.get(key) != expected
        }
        if mismatch:
            raise ValueError(f"{label} is not the pinned H4 classifier: {mismatch}")
    wandb = config["logger"]["wandb"]
    if (
        wandb.get("id") != WANDB_ID
        or wandb.get("fresh_run")
        or wandb.get("resume") != "allow"
        or experiment.get("wandb_required") is not True
    ):
        raise ValueError("h4trust01 requires preempt-safe live W&B logging")
    platform = config["platform"]
    if (
        int(platform.get("number_of_workers", 0)) != 16
        or int(platform.get("resources_per_worker", {}).get("GPU", 0)) != 1
    ):
        raise ValueError("h4trust01 requires 16 one-GPU workers")


def assert_wandb_runtime_ready() -> None:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("WANDB_DISABLED conflicts with h4trust01")
    if os.environ.get("WANDB_MODE", "").lower() in {
        "offline",
        "disabled",
        "dryrun",
    }:
        raise RuntimeError("WANDB_MODE must allow online logging")
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is not installed in the launcher environment")


def _checkpoint_is_complete(checkpoint: Mapping[str, Any]) -> bool:
    try:
        summarize_endpoint(checkpoint)
    except ValueError:
        return False
    return True


def verify_paths(config: Mapping[str, Any]) -> str:
    training = config["options"]["Training"]
    for value in (
        SOURCE,
        Path(config["reward_config"]["omnifold"]["backbone_checkpoint"]),
        Path(config["platform"]["data_parquet_dir"]),
        Path(config["platform"]["data_parquet_val_dir"]),
    ):
        if not value.exists():
            raise FileNotFoundError(value)
    source = _load_checkpoint(SOURCE)
    missing = sorted(RECOVERY_KEYS - set(source))
    if missing:
        raise ValueError(f"h4rep01 step-0 source is incomplete: missing {missing}")
    if (
        int(source.get("global_step", -1)) != 0
        or int(source.get("dgpo_reward_round_id", -1)) != 1
    ):
        raise ValueError("source must be h4rep01 reward round 1 at global step 0")
    source_audits = _valid_cold_audits(source)
    if 0 not in source_audits:
        raise ValueError("source lacks the completed saturated seed-2 step-0 audit")
    if not math.isclose(
        source_audits[0]["raw_auc_gap"],
        FIXED_BASELINE_GAP,
        rel_tol=0.0,
        abs_tol=5.0e-6,
    ):
        raise ValueError("source step-0 audit does not match h4rep01")

    output = Path(training["model_checkpoint_save_path"])
    if output.resolve() == SOURCE.parent.resolve():
        raise ValueError("treatment output cannot overwrite h4rep01")
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError(f"incomplete h4trust01 recovery checkpoint: {missing}")
        step = int(resumed.get("global_step", -1))
        if not 0 <= step <= MAX_PROPOSALS:
            raise ValueError(f"resume proposal step is outside [0, 10]: {step}")
        if _checkpoint_is_complete(resumed):
            return f"complete endpoint at proposal step={step}"
        return f"resume incomplete branch at proposal step={step}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("output contains checkpoints but no recoverable last.ckpt")
    return "fresh full-state branch from h4rep01 seed-2 step 0"


def _write_reproducibility_files(
    config: Mapping[str, Any],
    *,
    epsilon_report: Path,
    epsilon_result: Mapping[str, Any],
) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt"))
    resolved = OUTPUT_ROOT / "resolved_overlay.yaml"
    resolved_payload = dict(config)
    if resolved.exists():
        if yaml.safe_load(resolved.read_text()) != resolved_payload:
            if has_checkpoint:
                raise ValueError("resolved h4trust01 overlay changed after checkpointing")
            resolved.write_text(yaml.safe_dump(resolved_payload, sort_keys=False))
    else:
        resolved.write_text(yaml.safe_dump(resolved_payload, sort_keys=False))

    manifest_payload = {
        "schema": PROTOCOL,
        "wandb_run_id": WANDB_ID,
        "source_checkpoint": str(SOURCE),
        "source_wandb_run": "h4rep01",
        "source_replication_global_step": 0,
        "seed_bundle": _configured_seed_bundle(config),
        "max_policy_proposals": MAX_PROPOSALS,
        "fixed_baseline_gap": FIXED_BASELINE_GAP,
        "matched_control_step10_gap": CONTROL_STEP10_GAP,
        "velocity_mse_ratio_radius": 1.0e-4,
        "backtracking_scales": [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        "transactional_rejection": True,
        "stop_after_rejection": True,
        "cold_h4_audit_min_updates": 1000,
        "h4_reward_formula_changed": False,
        "dgpo_candidate_objective_changed": False,
        "reference_formulation_changed": True,
        "epsilon_report": str(epsilon_report),
        **dict(epsilon_result),
    }
    manifest = OUTPUT_ROOT / "experiment_manifest.json"
    encoded = json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n"
    if manifest.exists() and manifest.read_text() != encoded and has_checkpoint:
        raise ValueError("h4trust01 manifest changed after checkpointing")
    manifest.write_text(encoded)
    return resolved


def publish_endpoint() -> dict[str, Any]:
    report = summarize_endpoint(_load_checkpoint(FINAL_CHECKPOINT))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if FINAL_REPORT.exists() and FINAL_REPORT.read_text() != encoded:
        raise ValueError("existing h4trust01 report disagrees with final checkpoint")
    FINAL_REPORT.write_text(encoded)

    import wandb

    run = wandb.init(
        entity="ytchou97-university-of-washington",
        project="nu2flow-RL",
        id=WANDB_ID,
        resume="allow",
        name="c4a91e07_h4_seed2_function_trust_10proposal_v1",
        job_type="function-trust-decision",
    )
    run.log(
        {
            "function_trust/primary/fixed_baseline_gap": FIXED_BASELINE_GAP,
            "function_trust/primary/endpoint_gap": report["endpoint_cold_h4"][
                "raw_auc_gap"
            ],
            "function_trust/primary/delta_vs_fixed_baseline": report[
                "delta_vs_fixed_baseline"
            ],
            "function_trust/primary/passed": float(report["passed"]),
            "function_trust/proposal_step": float(report["proposal_step"]),
            "function_trust/accepted_updates": float(report["accepted_updates"]),
            "function_trust/stopped_at_rejection": float(
                report["stopped_at_rejection"]
            ),
            "function_trust/endpoint_audit_training_steps": float(
                report["endpoint_cold_h4"]["raw_audit_training_steps"]
            ),
            "function_trust/matched_control_step10_gap": CONTROL_STEP10_GAP,
        }
    )
    run.summary.update(
        {
            "function_trust/status": "passed" if report["passed"] else "failed",
            "function_trust/primary_endpoint": (
                "endpoint_cold_h4_gap_minus_fixed_seed2_baseline"
            ),
            "function_trust/delta_vs_fixed_baseline": report[
                "delta_vs_fixed_baseline"
            ],
            "function_trust/proposal_step": report["proposal_step"],
            "function_trust/accepted_updates": report["accepted_updates"],
            "function_trust/stopped_at_rejection": report[
                "stopped_at_rejection"
            ],
            "function_trust/endpoint_audit_valid": True,
        }
    )
    artifact = wandb.Artifact(
        "c4a91e07-h4-seed2-function-trust-endpoint",
        type="diagnostic",
    )
    artifact.add_file(str(FINAL_REPORT))
    run.log_artifact(artifact)
    run.finish()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config = read_overlay_yaml(args.config.expanduser().resolve())
    report_path = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else Path(config["experiment"]["epsilon_sweep_report"]).expanduser().resolve()
    )
    epsilon_result = read_epsilon_result(report_path, config["experiment"])
    apply_epsilon_result(config, epsilon_result)
    assert_contract(config)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)

    print(
        "H4 functional-trust preflight passed: "
        f"{mode}; LR={epsilon_result['epsilon_rms']:.6g}, "
        f"reward tempering={epsilon_result['tempering']:.6g}.\n"
        "Contract: exact h4rep01 seed-2 step-0 H4/reference/AdamW state, "
        "H4-only calibrated-LOO objective, ESS15, K=8, fixed "
        "velocity_mse_ratio radius=1e-4, transactional AdamW backtracking "
        "through 1/64, at most 10 proposals, stop and cold >=1000-update "
        "H4 audit at first rejection, no refit, live W&B h4trust01.",
        flush=True,
    )
    if args.check_only:
        return 0

    assert_wandb_runtime_ready()
    if FINAL_CHECKPOINT.is_file() and _checkpoint_is_complete(
        _load_checkpoint(FINAL_CHECKPOINT)
    ):
        report = publish_endpoint()
        print(
            "h4trust01 already complete: "
            f"proposal={report['proposal_step']} accepted={report['accepted_updates']} "
            f"delta={report['delta_vs_fixed_baseline']:+.6f} "
            f"passed={report['passed']}",
            flush=True,
        )
        return 0

    assert_live_ray_16gpu()
    resolved = _write_reproducibility_files(
        config,
        epsilon_report=report_path,
        epsilon_result=epsilon_result,
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
    report = publish_endpoint()
    print(
        "H4 functional-trust experiment complete: "
        f"proposal={report['proposal_step']} accepted={report['accepted_updates']} "
        f"boundary_rejection={report['stopped_at_rejection']} "
        f"gap={report['endpoint_cold_h4']['raw_auc_gap']:.6f} "
        f"delta={report['delta_vs_fixed_baseline']:+.6f} "
        f"passed={report['passed']}; live result is on W&B h4trust01.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
