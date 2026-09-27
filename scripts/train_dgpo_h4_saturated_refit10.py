#!/usr/bin/env python3
"""Run the matched saturated-H4 paired-refit reference-dynamics experiment."""

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

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evenet_dgpo"))

from train_dgpo_h4_fresh_ensemble_pilot import (  # noqa: E402
    EXPECTED_SOURCE_RUN,
    EXPECTED_SOURCE_STEP,
    RECOVERY_KEYS,
    _load_checkpoint,
    apply_epsilon_result,
    read_epsilon_result,
)
from train_dgpo_old_method_10pct import assert_live_ray_16gpu  # noqa: E402
from train_neutrino_backend import read_overlay_yaml  # noqa: E402
from RL.DGPO_neutrino.gradient_transfer import (  # noqa: E402
    resolve_gradient_transfer_trace_config,
)


BASE = ROOT / "config/train_diffusion_nersc.yaml"
DEFAULT = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_"
    "h4_saturated_refit10_keepadam_50step.yaml"
)
PROTOCOL = "h4-saturated-refit10-keepadam-50step-v1"
WANDB_ID = "h4ref10"
SOURCE_CHECKPOINT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_"
    "nohardtrust_seed42/checkpoints/last.ckpt"
)
OUTPUT_ROOT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "c4a91e07_h4_saturated_refit10_keepadam_50step_v1"
)
AUDIT_STEPS = (0, 10, 20, 30, 40, 50)
REFRESH_STEPS = (0, 10, 20, 30, 40)
TRACE_STEPS = (1, 2, 5, 10, 20, 30, 40, 50)
SEED_BUNDLE = {
    "reward_fit": 20261020,
    "reward_pool_order": 314159,
    "cold_audit": 20261021,
    "gradient_conflict_panel": 20261015,
    "tarp": 20261022,
}


def _seed_bundle(config: Mapping[str, Any]) -> dict[str, int]:
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    return {
        "reward_fit": int(adaptive["recalibration"]["seed"]),
        "reward_pool_order": int(
            adaptive["recalibration"]["pool_selection_seed"]
        ),
        "cold_audit": int(adaptive["trigger"]["probe_seed"]),
        "gradient_conflict_panel": int(dgpo["gradient_conflict"]["seed"]),
        "tarp": int(dgpo["tarp"]["seed"]),
    }


def assert_contract(config: Mapping[str, Any]) -> None:
    """Fail closed if the paired arm differs from h4rep01 beyond cadence."""

    experiment = config["experiment"]
    training = config["options"]["Training"]
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    trigger = adaptive["trigger"]
    audit = adaptive["audit_fit"]
    recal = adaptive["recalibration"]
    fit = recal["fit"]
    trust = dgpo["reference_trust"]
    trace = resolve_gradient_transfer_trace_config(
        dgpo.get("gradient_transfer_trace")
    )

    expected_experiment = {
        "protocol": PROTOCOL,
        "single_experimental_change": (
            "paired_h4_reward_reference_refresh_every_10_updates"
        ),
        "control_wandb_run": (
            "ytchou97-university-of-washington/nu2flow-RL/h4rep01"
        ),
        "rounds": 5,
        "policy_updates_per_round": 10,
        "reward_lifetime_updates": 10,
        "reward_refresh_steps": list(REFRESH_STEPS),
        "audit_steps": list(AUDIT_STEPS),
        "endpoint_global_step": 50,
        "endpoint_selection": "fixed_step_50",
        "reward_members": 4,
        "audit_repeats": 1,
        "applied_objective_changed": False,
        "h4_reward_formula_changed": False,
        "reference_formula_changed": False,
        "reference_anchor_dynamics_changed": True,
        "gradient_trace_update_end_steps": list(TRACE_STEPS),
        "cold_h4_audit_policy_steps": list(AUDIT_STEPS),
        "cold_h4_audit_min_optimizer_updates": 1000,
        "primary_endpoint": "late_window_mean_staleness_raw_auc_gap",
    }
    mismatch = {
        key: (experiment.get(key), expected)
        for key, expected in expected_experiment.items()
        if experiment.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"reference-dynamics metadata mismatch: {mismatch}")
    if (
        experiment.get("source_wandb_run") != EXPECTED_SOURCE_RUN
        or int(experiment.get("source_policy_step", -1))
        != EXPECTED_SOURCE_STEP
        or _seed_bundle(config) != SEED_BUNDLE
    ):
        raise ValueError("reference arm must match h4rep01 source and seed2 bundle")

    if (
        int(training.get("epochs", -1)),
        int(training.get("total_epochs", -1)),
        int(dgpo.get("steps_per_epoch", -1)),
    ) != (5, 5, 10):
        raise ValueError("reference arm requires five ten-update policy rounds")
    if (
        dgpo.get("checkpoint_load_mode") != "weights_only"
        or dgpo.get("auto_resume_from_last") is not True
        or Path(training["model_checkpoint_load_path"]) != SOURCE_CHECKPOINT
    ):
        raise ValueError("reference arm must weights-only start from c4a91e07")
    if not math.isclose(
        float(training["learning_rate"]),
        float(experiment["resolved_epsilon_rms"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("policy LR must equal the selected epsilon RMS")
    if (
        dgpo.get("advantage_estimator") != "leave_one_out_unscaled"
        or int(dgpo.get("K", -1)) != 8
        or float(dgpo.get("beta", float("nan"))) != 1.0
        or float(dgpo.get("beta_kl", float("nan"))) != 0.0
    ):
        raise ValueError("H4 DGPO mathematical objective changed")
    if not (
        trust.get("enabled") is True
        and trust.get("objective") == "velocity_mse"
        and math.isclose(float(trust.get("coefficient", float("nan"))), 1.0)
        and not trust.get("adaptive_boundary", {}).get("enabled")
    ):
        raise ValueError("coefficient-1 soft velocity-MSE reference changed")
    if (
        dgpo.get("projection_constraint", {}).get("type") != "none"
        or dgpo.get("sequential_vp_trust_backward")
        or not dgpo.get("fail_on_skipped_optimizer_step")
        or not dgpo.get("log_parameter_update_rms")
    ):
        raise ValueError("ordinary committed native AdamW update path changed")
    variance = dgpo.get("variance_regularization") or {}
    if variance.get("enabled") or float(variance.get("weight", 0.0)) != 0.0:
        raise ValueError("variance regularization must remain disabled")

    if (
        adaptive.get("enabled") is not True
        or adaptive.get("monitor_mode") != "raw_plateau_refit"
        or int(adaptive.get("staleness_every_n_epochs", -1)) != 1
        or adaptive.get("staleness_every_n_steps") is not None
        or adaptive.get("fixed_audit_panel") is not True
        or adaptive.get("cache_event_inputs") is not True
        or adaptive.get("single_pool_train_validation") is not False
        or trigger.get("fixed_schedule_skip_staleness_audit") is not True
        or trigger.get("fixed_schedule_log_raw_audit") is not True
        or int(trigger.get("max_reward_age_epochs", -1)) != 1
        or trigger.get("warm_start_classifier")
        or trigger.get("rollback_to_best_on_plateau")
        or trigger.get("require_audit_saturation") is not True
    ):
        raise ValueError("reward/reference refresh must be fixed and selection-blind")
    if (
        int(audit.get("min_steps", 0)) != 1000
        or audit.get("fail_if_unsaturated") is not True
        or audit.get("disjoint_final_audit") is not True
        or audit.get("checkpoint_selection_metric") != "balanced_accuracy"
        or int(audit.get("repeats", 0)) != 1
        or int(audit.get("repeat_seed_stride", 0)) != 104_729
    ):
        raise ValueError("cold H4 audit contract changed")

    if (
        recal.get("bootstrap_on_start") is not True
        or int(recal.get("max_reward_rounds", -1)) != 5
        or (recal.get("min_iterations"), recal.get("max_iterations")) != (1, 1)
        or recal.get("iteration_one_only") is not True
        or (recal.get("crossfit_repeats"), recal.get("crossfit_folds"))
        != (2, 2)
        or recal.get("warm_start_iterations")
        or recal.get("warm_start_from_iteration_one")
        or recal.get("reset_optimizer_state_on_install") is not False
        or recal.get("reset_adam_first_moment_on_install") is not False
        or recal.get("scheduled_refit_fail_closed") is not True
        or recal.get("acceptance_audit_enabled")
        or recal.get("topology_acceptance_audit_enabled")
    ):
        raise ValueError("paired refresh must use saturated H4 and continuous AdamW")
    if (
        int(fit.get("min_steps_per_fold", 0)) != 1000
        or fit.get("require_saturation") is not True
        or fit.get("minimum_sufficient_balanced_accuracy") is not None
        or fit.get("checkpoint_selection_metric") != "balanced_accuracy"
    ):
        raise ValueError("every reward member must train to saturation after 1000 steps")

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
            raise ValueError(f"{label} is not pinned H4: {mismatch}")
    if (recal.get("adaptive_tempering") or {}) != {
        "enabled": True,
        "target_ess_fraction": 0.15,
        "minimum": 0.10,
        "grid_steps": 14,
        "inherit_previous": False,
    } or float(recal.get("minimum_ess_fraction", -1.0)) != 0.0:
        raise ValueError("soft adaptive ESS15 tempering changed")

    gradient_conflict = dgpo.get("gradient_conflict") or {}
    if not (
        gradient_conflict.get("enabled") is True
        and gradient_conflict.get("monitor_refit_lifecycle") is True
        and int(gradient_conflict.get("every_n_steps", -1)) == 10
    ):
        raise ValueError("matched pre-refit/post-install probes are required")
    if (
        not trace.enabled
        or trace.update_end_steps != TRACE_STEPS
        or trace.counterfactual_trust_coefficients
        != (0.0, 0.1, 0.2, 0.5, 1.0)
    ):
        raise ValueError("exact gradient-transfer schedule changed")

    wb = config["logger"]["wandb"]
    if (
        wb.get("id") != WANDB_ID
        or wb.get("fresh_run")
        or wb.get("resume") != "allow"
        or wb.get("profile") != "standard"
        or not wb.get("classifier_loss_curves")
        or not wb.get("simplified")
        or experiment.get("wandb_required") is not True
    ):
        raise ValueError("live resumable W&B h4ref10 is required")
    platform = config["platform"]
    if (
        int(platform.get("number_of_workers", 0)) != 16
        or int(platform["resources_per_worker"].get("GPU", 0)) != 1
    ):
        raise ValueError("reference arm requires 16 one-GPU Ray workers")


def assert_wandb_runtime_ready() -> None:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("WANDB_DISABLED conflicts with this live experiment")
    if os.environ.get("WANDB_MODE", "").lower() in {
        "offline",
        "disabled",
        "dryrun",
    }:
        raise RuntimeError("WANDB_MODE must allow online logging")
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is not installed in the launcher environment")


def verify_paths(config: Mapping[str, Any]) -> str:
    training = config["options"]["Training"]
    for value in (
        training["model_checkpoint_load_path"],
        config["reward_config"]["omnifold"]["backbone_checkpoint"],
        config["platform"]["data_parquet_dir"],
        config["platform"]["data_parquet_val_dir"],
    ):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    source = _load_checkpoint(SOURCE_CHECKPOINT)
    if int(source.get("global_step", -1)) != EXPECTED_SOURCE_STEP:
        raise ValueError("source checkpoint is not c4a91e07 step 1110")

    output = Path(training["model_checkpoint_save_path"])
    if output.parent.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("reference arm output root changed")
    if output.resolve() == SOURCE_CHECKPOINT.parent.resolve():
        raise ValueError("reference arm cannot overwrite its source")
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError("incomplete last.ckpt: " + ", ".join(missing))
        step = int(resumed.get("global_step", -1))
        reward_round = int(resumed.get("dgpo_reward_round_id", -1))
        if not 0 <= step <= 50 or not 1 <= reward_round <= 5:
            raise ValueError(
                f"invalid recovery state step={step} round={reward_round}"
            )
        return f"resume step={step} reward_round={reward_round}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("output contains checkpoints but no complete last.ckpt")
    return "fresh weights-only start from c4a91e07 step 1110"


def _write_exact(path: Path, payload: Any, *, checkpoints_exist: bool) -> None:
    encoded = (
        yaml.safe_dump(payload, sort_keys=False)
        if path.suffix == ".yaml"
        else json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    if path.exists() and path.read_text() != encoded and checkpoints_exist:
        raise ValueError(f"{path.name} changed after a checkpoint was written")
    path.write_text(encoded)


def summarize_endpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed cold-audit panel and compute the declared endpoint."""

    if int(checkpoint.get("global_step", -1)) != 50:
        raise ValueError("reference-dynamics endpoint must be global_step=50")
    if int(checkpoint.get("dgpo_reward_round_id", -1)) != 5:
        raise ValueError("reference-dynamics endpoint must retain reward round 5")
    history = list(
        (checkpoint.get("dgpo_adaptive_omnifold_state") or {}).get(
            "probe_history", []
        )
        or []
    )
    by_step: dict[int, dict[str, Any]] = {}
    for raw in history:
        if not isinstance(raw, Mapping):
            continue
        if float(raw.get("fixed_schedule_diagnostic_only", 0.0)) < 0.5:
            continue
        step = int(raw.get("global_step", -1))
        if step in AUDIT_STEPS:
            by_step[step] = dict(raw)
    missing = [step for step in AUDIT_STEPS if step not in by_step]
    if missing:
        raise ValueError(f"missing cold H4 audits at steps {missing}")

    audits: dict[str, dict[str, Any]] = {}
    for step in AUDIT_STEPS:
        raw = by_step[step]
        auc = float(raw.get("raw_auc", float("nan")))
        gap = float(raw.get("raw_auc_gap", float("nan")))
        updates = int(raw.get("raw_audit_training_steps", 0))
        saturated = float(raw.get("raw_audit_saturated", 0.0)) >= 0.5
        if not math.isfinite(auc) or not math.isfinite(gap):
            raise ValueError(f"step-{step} cold H4 audit is nonfinite")
        if updates < 1000 or not saturated:
            raise ValueError(
                f"step-{step} cold audit invalid: updates={updates}, "
                f"saturated={saturated}"
            )
        audits[str(step)] = {
            "raw_auc": auc,
            "raw_auc_gap": gap,
            "raw_audit_training_steps": updates,
            "raw_audit_saturated": True,
        }

    gaps = [audits[str(step)]["raw_auc_gap"] for step in AUDIT_STEPS]
    late_mean = sum(audits[str(step)]["raw_auc_gap"] for step in (30, 40, 50)) / 3
    step0_gap = gaps[0]
    delta = late_mean - step0_gap
    total_variation = sum(abs(right - left) for left, right in zip(gaps, gaps[1:]))
    worst_excess = max(gaps) - step0_gap
    return {
        "schema": "c4a91e07-h4-saturated-refit10-reference-dynamics-v1",
        "wandb_run_id": WANDB_ID,
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "policy_endpoint_step": 50,
        "reward_rounds": 5,
        "refresh_steps": list(REFRESH_STEPS),
        "cold_h4_audits": audits,
        "primary_endpoint": "mean_gap30_gap40_gap50_minus_gap0",
        "late_window_mean_gap": late_mean,
        "step0_gap": step0_gap,
        "delta_vs_step0": delta,
        "passed": delta < 0.0,
        "total_variation": total_variation,
        "worst_excess_vs_step0": worst_excess,
        "classifier_audit_min_updates": 1000,
        "h4_reward_formula_changed": False,
        "reference_formula_changed": False,
        "reference_anchor_dynamics_changed": True,
        "optimizer_state_preserved_across_refits": True,
    }


def publish_endpoint(report_path: Path, report: Mapping[str, Any]) -> None:
    import wandb

    run = wandb.init(
        entity="ytchou97-university-of-washington",
        project="nu2flow-RL",
        id=WANDB_ID,
        resume="allow",
        name="c4a91e07_h4_saturated_refit10_keepadam_50step_v1",
        job_type="reference-dynamics-decision",
    )
    metrics = {
        "reference_dynamics/primary/late_window_mean_gap": float(
            report["late_window_mean_gap"]
        ),
        "reference_dynamics/primary/step0_gap": float(report["step0_gap"]),
        "reference_dynamics/primary/delta_vs_step0": float(
            report["delta_vs_step0"]
        ),
        "reference_dynamics/primary/passed": float(report["passed"]),
        "reference_dynamics/stability/total_variation": float(
            report["total_variation"]
        ),
        "reference_dynamics/stability/worst_excess_vs_step0": float(
            report["worst_excess_vs_step0"]
        ),
    }
    for raw_step, row in report["cold_h4_audits"].items():
        prefix = f"reference_dynamics/cold_h4/step_{raw_step}"
        metrics[f"{prefix}/raw_auc"] = float(row["raw_auc"])
        metrics[f"{prefix}/raw_auc_gap"] = float(row["raw_auc_gap"])
        metrics[f"{prefix}/training_steps"] = float(
            row["raw_audit_training_steps"]
        )
        metrics[f"{prefix}/saturated"] = 1.0
    run.log(metrics)
    run.summary.update(
        {
            "reference_dynamics/status": (
                "passed" if report["passed"] else "failed"
            ),
            "reference_dynamics/all_required_audits_valid": True,
            "reference_dynamics/primary/delta_vs_step0": report[
                "delta_vs_step0"
            ],
            "reference_dynamics/stability/total_variation": report[
                "total_variation"
            ],
        }
    )
    artifact = wandb.Artifact(
        "c4a91e07-h4-saturated-refit10-reference-dynamics",
        type="diagnostic",
    )
    artifact.add_file(str(report_path))
    run.log_artifact(artifact)
    run.finish()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if config_path != DEFAULT.resolve():
        parser.error(f"--config must be {DEFAULT}")
    config = read_overlay_yaml(config_path)
    experiment = config["experiment"]
    report_path = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else Path(experiment["epsilon_sweep_report"]).expanduser().resolve()
    )
    epsilon_result = read_epsilon_result(report_path, experiment)
    apply_epsilon_result(config, epsilon_result)
    assert_contract(config)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)

    print(
        "Reference-dynamics preflight passed: " + mode + "\n"
        f"source={EXPECTED_SOURCE_RUN}/last.ckpt step={EXPECTED_SOURCE_STEP}; "
        f"epsilon/LR={epsilon_result['epsilon_rms']:.6g}\n"
        "Protocol: five fresh saturated 2x2 H4 rewards, ten committed DGPO "
        "updates per paired reward/reference, continuous native AdamW, soft "
        "ESS15, cold >=1000-update audits at steps 0/10/20/30/40/50, and "
        f"live W&B id={WANDB_ID}.",
        flush=True,
    )
    if args.check_only:
        return 0

    assert_wandb_runtime_ready()
    assert_live_ray_16gpu()
    checkpoint_dir = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    output_root = checkpoint_dir.parent
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_exist = checkpoint_dir.exists() and any(
        checkpoint_dir.glob("*.ckpt")
    )
    resolved_path = output_root / "resolved_overlay.yaml"
    _write_exact(resolved_path, config, checkpoints_exist=checkpoints_exist)
    manifest = {
        "schema": "h4-saturated-refit10-reference-dynamics-manifest-v1",
        "protocol": PROTOCOL,
        "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "control_wandb_run": experiment["control_wandb_run"],
        "epsilon_report": str(report_path),
        **epsilon_result,
        "seed_bundle": SEED_BUNDLE,
        "reward_members_per_round": 4,
        "reward_rounds": 5,
        "reward_member_minimum_steps": 1000,
        "reward_requires_saturation": True,
        "reward_refresh_steps": list(REFRESH_STEPS),
        "policy_updates_per_round": 10,
        "reset_policy_optimizer_on_reward_install": False,
        "target_ess_fraction": 0.15,
        "audit_minimum_steps": 1000,
        "audit_steps": list(AUDIT_STEPS),
        "audit_controls_training": False,
        "endpoint_global_step": 50,
        "wandb_run": (
            f"{config['logger']['wandb']['entity']}/"
            f"{config['logger']['wandb']['project']}/{WANDB_ID}"
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
            str(resolved_path),
            "--",
            "--ray-dir",
            str(config["nersc"]["ray"]["results_dir"]),
        ],
        cwd=ROOT,
        check=True,
    )
    final = _load_checkpoint(checkpoint_dir / "last.ckpt")
    report = summarize_endpoint(final)
    endpoint_path = output_root / "reference_dynamics_endpoint.json"
    _write_exact(endpoint_path, report, checkpoints_exist=False)
    publish_endpoint(endpoint_path, report)
    print(
        "Reference dynamics complete: "
        f"late_mean={report['late_window_mean_gap']:.6g} "
        f"gap0={report['step0_gap']:.6g} "
        f"delta={report['delta_vs_step0']:+.6g} "
        f"total_variation={report['total_variation']:.6g} "
        f"passed={report['passed']}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
