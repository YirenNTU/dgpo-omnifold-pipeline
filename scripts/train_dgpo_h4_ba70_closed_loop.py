#!/usr/bin/env python3
"""Run the five-round BA70 H4 closed-loop distribution-matching test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_fresh_ensemble_pilot import (
    EXPECTED_SOURCE_RUN,
    EXPECTED_SOURCE_STEP,
    RECOVERY_KEYS,
    _load_checkpoint,
    apply_epsilon_result,
    assert_wandb_runtime_ready,
    read_epsilon_result,
)
from train_dgpo_old_method_10pct import assert_live_ray_16gpu
from train_neutrino_backend import read_overlay_yaml


BASE = ROOT / "config/train_diffusion_nersc.yaml"
OLD_CONFIG = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba70_m4_ess15_"
    "refit10_keepadam_50step_audit1.yaml"
)
DEFAULT = OLD_CONFIG
SOURCE_CHECKPOINT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_"
    "nohardtrust_seed42/checkpoints/last.ckpt"
)
VARIANTS = {
    "h4-ba70-m4-ess15-refit10-keepadam-50step-audit1-v1": {
        "config": OLD_CONFIG,
        "wandb_id": "h4b70c50",
        "reward_minimum_steps": 1000,
        "audit_minimum_steps": 1000,
    },
}
AUDIT_STEPS = [0, 10, 20, 30, 40, 50]


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    protocol = experiment.get("protocol")
    if protocol not in VARIANTS:
        raise ValueError(f"unsupported closed-loop protocol: {protocol}")
    variant = VARIANTS[protocol]
    reward_minimum_steps = int(variant["reward_minimum_steps"])
    audit_minimum_steps = int(variant["audit_minimum_steps"])
    training = config["options"]["Training"]
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    trigger = adaptive["trigger"]
    audit = adaptive["audit_fit"]
    recal = adaptive["recalibration"]
    fit = recal["fit"]

    expected_experiment = {
        "protocol": protocol,
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
    mismatch = {
        key: (experiment.get(key), value)
        for key, value in expected_experiment.items()
        if experiment.get(key) != value
    }
    if mismatch:
        raise ValueError(f"closed-loop experiment metadata mismatch: {mismatch}")

    reproducibility = config["nersc"]["reproducibility"]
    if (
        reproducibility.get("source_wandb_run") != EXPECTED_SOURCE_RUN
        or reproducibility.get("source_dgpo_global_step")
        != EXPECTED_SOURCE_STEP
        or Path(training["model_checkpoint_load_path"]) != SOURCE_CHECKPOINT
    ):
        raise ValueError(
            "closed loop must use c4a91e07 last.ckpt at source step 1110"
        )
    if (
        int(training.get("epochs", -1)),
        int(training.get("total_epochs", -1)),
        int(dgpo.get("steps_per_epoch", -1)),
    ) != (5, 5, 10):
        raise ValueError("closed loop requires five ten-update policy rounds")
    if (
        dgpo.get("checkpoint_load_mode") != "weights_only"
        or not dgpo.get("auto_resume_from_last")
    ):
        raise ValueError(
            "source policy must start weights-only and output must auto-resume"
        )
    if dgpo.get("advantage_estimator") != "leave_one_out_unscaled":
        raise ValueError("closed loop requires raw unscaled leave-one-out reward")
    if not math.isclose(
        float(training["learning_rate"]),
        float(experiment["resolved_epsilon_rms"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("policy LR must equal the selected epsilon RMS")
    if dgpo["reference_trust"]["adaptive_boundary"].get("enabled"):
        raise ValueError("hard trust must remain disabled")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("projection must remain disabled")
    if not dgpo.get("fail_on_skipped_optimizer_step"):
        raise ValueError("all fifty requested updates must be committed updates")

    if (
        adaptive.get("monitor_mode") != "raw_plateau_refit"
        or adaptive.get("staleness_every_n_epochs") != 1
        or adaptive.get("staleness_every_n_steps") is not None
        or trigger.get("max_reward_age_epochs") != 1
        or trigger.get("fixed_schedule_skip_staleness_audit") is not True
        or trigger.get("fixed_schedule_log_raw_audit") is not True
        or trigger.get("warm_start_classifier")
        or trigger.get("rollback_to_best_on_plateau")
    ):
        raise ValueError(
            "reward refits must use a selection-blind fixed ten-update schedule"
        )
    if (
        trigger.get("raw_audit_enabled") is not True
        or trigger.get("require_audit_saturation") is not True
        or adaptive.get("fixed_audit_panel") is not True
        or audit.get("disjoint_final_audit") is not True
        or int(audit.get("min_steps", 0)) != audit_minimum_steps
        or audit.get("checkpoint_selection_metric") != "balanced_accuracy"
        or audit.get("repeats") != 1
        or audit.get("repeat_seed_stride") != 104_729
        or audit.get("fail_if_unsaturated") is not True
    ):
        raise ValueError(
            "primary audit must be a fresh saturated H4 fit on a fixed panel"
        )

    if (
        not recal.get("bootstrap_on_start")
        or recal.get("max_reward_rounds") != 5
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
        raise ValueError(
            "reward must be fresh 2x2 iteration-one H4 with continuous policy AdamW"
        )
    if (
        fit.get("min_steps_per_fold") != reward_minimum_steps
        or fit.get("require_saturation") is not False
        or fit.get("minimum_sufficient_balanced_accuracy") != 0.70
        or fit.get("minimum_sufficient_confidence_z") != 0.0
        or fit.get("minimum_sufficient_required_consecutive") != 3
        or fit.get("checkpoint_selection_metric") != "balanced_accuracy"
    ):
        raise ValueError("reward members require the exact BA70 stopping rule")

    h4 = {
        "periodic_pair_features": True,
        "topology_fourier_embedding": True,
        "topology_conditioning": False,
        "visible_pair_rest_frame": False,
        "topology_max_harmonic": 4,
        "topology_include_theta_pair": False,
        "topology_direct_logit": False,
    }
    for label, block in (("reward", recal), ("audit", audit)):
        mismatch = {
            key: (block.get(key), value)
            for key, value in h4.items()
            if block.get(key) != value
        }
        if mismatch:
            raise ValueError(f"{label} classifier is not pinned H4: {mismatch}")

    if (recal.get("adaptive_tempering") or {}) != {
        "enabled": True,
        "target_ess_fraction": 0.15,
        "minimum": 0.10,
        "grid_steps": 14,
        "inherit_previous": False,
    } or float(recal.get("minimum_ess_fraction", -1.0)) != 0.0:
        raise ValueError("reward requires soft adaptive ESS15 tempering")

    wb = config["logger"]["wandb"]
    if (
        wb.get("id") != variant["wandb_id"]
        or wb.get("profile") != "standard"
        or not wb.get("classifier_loss_curves")
        or not wb.get("simplified")
        or wb.get("fresh_run")
        or wb.get("resume") != "allow"
        or not experiment.get("wandb_required")
    ):
        raise ValueError("live resumable W&B logging is required")
    platform = config["platform"]
    if (
        int(platform.get("number_of_workers", 0)) != 16
        or int(platform["resources_per_worker"].get("GPU", 0)) != 1
    ):
        raise ValueError("closed loop requires 16 one-GPU Ray workers")


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
    if output.resolve() == SOURCE_CHECKPOINT.parent.resolve():
        raise ValueError("closed-loop output cannot overwrite c4a91e07")
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError(
                "incomplete closed-loop last.ckpt: " + ", ".join(missing)
            )
        step = int(resumed.get("global_step", -1))
        reward_round = int(resumed.get("dgpo_reward_round_id", -1))
        if not 0 <= step <= 50 or not 1 <= reward_round <= 5:
            raise ValueError(
                f"invalid recovery state step={step} round={reward_round}"
            )
        return f"resume step={step} reward_round={reward_round}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError(
            "closed-loop output has checkpoints but no complete last.ckpt"
        )
    return "fresh weights-only start from c4a91e07 step 1110"


def _write_exact(path: Path, payload: Any, *, checkpoints_exist: bool) -> None:
    serialized = (
        yaml.safe_dump(payload, sort_keys=False)
        if path.suffix == ".yaml"
        else json.dumps(payload, indent=2) + "\n"
    )
    if path.exists() and path.read_text() != serialized and checkpoints_exist:
        raise ValueError(
            f"{path.name} changed after a closed-loop checkpoint was written"
        )
    path.write_text(serialized)


def _audit_trajectory(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    adaptive_state = checkpoint.get("dgpo_adaptive_omnifold_state") or {}
    rows = [
        dict(row)
        for row in adaptive_state.get("probe_history", [])
        if float(row.get("fixed_schedule_diagnostic_only", 0.0)) >= 0.5
    ]
    rows.sort(key=lambda row: int(row.get("global_step", -1)))
    observed_steps = [int(row.get("global_step", -1)) for row in rows]
    if observed_steps != AUDIT_STEPS:
        raise RuntimeError(
            f"expected audit steps {AUDIT_STEPS}, got {observed_steps}"
        )
    if any(
        float(row.get("raw_audit_saturated", 0.0)) < 0.5
        or int(row.get("raw_audit_repeats", 0)) != 1
        for row in rows
    ):
        raise RuntimeError("one or more endpoint audits are incomplete")
    gaps = [float(row["raw_auc_gap"]) for row in rows]
    step_mean = sum(AUDIT_STEPS) / len(AUDIT_STEPS)
    gap_mean = sum(gaps) / len(gaps)
    slope_per_10 = 10.0 * sum(
        (step - step_mean) * (gap - gap_mean)
        for step, gap in zip(AUDIT_STEPS, gaps)
    ) / sum((step - step_mean) ** 2 for step in AUDIT_STEPS)
    return {
        "schema": "selection-blind-raw-audit-trajectory-v1",
        "audit_steps": AUDIT_STEPS,
        "raw_auc_gaps": gaps,
        "step0_gap": gaps[0],
        "step50_gap": gaps[-1],
        "step50_minus_step0": gaps[-1] - gaps[0],
        "slope_per_10_steps": slope_per_10,
        "endpoint_improved": gaps[-1] < gaps[0],
        "negative_overall_slope": slope_per_10 < 0.0,
        "primary_success": gaps[-1] < gaps[0] and slope_per_10 < 0.0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    supported_paths = {Path(item["config"]).resolve() for item in VARIANTS.values()}
    if config_path not in supported_paths:
        choices = ", ".join(str(path) for path in sorted(supported_paths))
        parser.error(f"--config must be one of: {choices}")
    config = read_overlay_yaml(config_path)
    experiment = config["experiment"]
    variant = VARIANTS[experiment["protocol"]]
    protocol = str(experiment["protocol"])
    wandb_id = str(variant["wandb_id"])
    reward_minimum_steps = int(variant["reward_minimum_steps"])
    audit_minimum_steps = int(variant["audit_minimum_steps"])
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
        "Closed-loop preflight passed: " + mode + "\n"
        f"source={EXPECTED_SOURCE_RUN}/last.ckpt step={EXPECTED_SOURCE_STEP}; "
        f"epsilon/LR={epsilon_result['epsilon_rms']:.6g}\n"
        f"Protocol: five fresh BA70 H4 2x2 rewards (minimum "
        f"{reward_minimum_steps} steps/member), ten committed DGPO updates "
        "per reward, continuous policy AdamW, soft ESS15, and one fresh "
        f"saturated selection-blind audit (minimum {audit_minimum_steps} steps) "
        "at steps 0,10,20,30,40,50. "
        f"Live W&B id={wandb_id}.",
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
        "schema": "h4-ba70-closed-loop-v1",
        "protocol": protocol,
        "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "epsilon_report": str(report_path),
        **epsilon_result,
        "reward_members_per_round": 4,
        "reward_rounds": 5,
        "reward_minimum_steps": reward_minimum_steps,
        "reward_stop_balanced_accuracy": 0.70,
        "policy_updates_per_round": 10,
        "reset_policy_optimizer_on_reward_install": False,
        "target_ess_fraction": 0.15,
        "audit_repeats": 1,
        "audit_minimum_steps": audit_minimum_steps,
        "audit_steps": AUDIT_STEPS,
        "audit_controls_training": False,
        "endpoint_global_step": 50,
        "wandb_run": (
            f"{config['logger']['wandb']['entity']}/"
            f"{config['logger']['wandb']['project']}/{wandb_id}"
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
    if (
        int(final.get("global_step", -1)) != 50
        or int(final.get("dgpo_reward_round_id", -1)) != 5
    ):
        raise RuntimeError(
            "closed loop returned without step=50 and five installed rewards"
        )
    trajectory = _audit_trajectory(final)
    _write_exact(
        output_root / "audit_trajectory.json",
        trajectory,
        checkpoints_exist=False,
    )
    print(
        "Closed loop complete: "
        f"step0_gap={trajectory['step0_gap']:.6g} "
        f"step50_gap={trajectory['step50_gap']:.6g} "
        f"slope/10={trajectory['slope_per_10_steps']:.6g} "
        f"primary_success={trajectory['primary_success']}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
