#!/usr/bin/env python3
"""Launch BA70 H4 reward training against the completed h4e15r01 control."""

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
HISTORICAL_CONTROL = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble_ess15_5round.yaml"
)
BA70 = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba70_m4_ess15_10step.yaml"
)
SOURCE_CHECKPOINT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/"
    "checkpoints/last.ckpt"
)
BA70_PROTOCOL = "h4-ba70-m4-ess15-10step-v1"
CONTROL_WANDB_RUN = (
    "ytchou97-university-of-washington/nu2flow-RL/h4e15r01"
)


def _training_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can change model training or evaluation."""

    signature = {
        key: copy.deepcopy(config[key])
        for key in ("platform", "network", "reward_config", "dgpo")
    }
    training = copy.deepcopy(config["options"]["Training"])
    for key in ("model_checkpoint_save_path", "epochs", "total_epochs"):
        training.pop(key, None)
    signature["Training"] = training
    recalibration = signature["dgpo"]["adaptive_omnifold"]["recalibration"]
    recalibration.pop("max_reward_rounds", None)
    fit = recalibration["fit"]
    for key in (
        "require_saturation",
        "minimum_sufficient_balanced_accuracy",
        "minimum_sufficient_confidence_z",
        "minimum_sufficient_required_consecutive",
    ):
        fit.pop(key, None)
    return signature


def assert_paired_configs(
    candidate: Mapping[str, Any], epsilon_result: Mapping[str, float]
) -> None:
    control = read_overlay_yaml(HISTORICAL_CONTROL)
    apply_epsilon_result(control, epsilon_result)
    if _training_signature(candidate) != _training_signature(control):
        raise ValueError(
            "ablation arms differ outside the classifier stopping/readiness rule"
        )


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    protocol = experiment.get("protocol")
    if protocol != BA70_PROTOCOL:
        raise ValueError("unsupported minimum-sufficient ablation protocol")
    if (
        experiment.get("rounds") != 1
        or experiment.get("policy_updates_per_round") != 10
        or experiment.get("endpoint_global_step") != 10
        or experiment.get("endpoint_selection") != "fixed_step_10"
    ):
        raise ValueError("ablation requires one reward and a fixed ten-step endpoint")
    reproducibility = config["nersc"]["reproducibility"]
    if (
        reproducibility.get("source_wandb_run") != EXPECTED_SOURCE_RUN
        or reproducibility.get("source_dgpo_global_step") != EXPECTED_SOURCE_STEP
    ):
        raise ValueError("ablation source must be c4a91e07 last.ckpt at step 1110")

    training = config["options"]["Training"]
    if Path(training["model_checkpoint_load_path"]) != SOURCE_CHECKPOINT:
        raise ValueError("ablation must load the old-classifier DGPO last.ckpt")
    if (training.get("epochs"), training.get("total_epochs")) != (1, 1):
        raise ValueError("ablation must run one ten-update epoch")
    if Path(training["model_checkpoint_save_path"]).parent == SOURCE_CHECKPOINT.parent:
        raise ValueError("ablation output cannot overwrite the source run")

    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    recal = adaptive["recalibration"]
    fit = recal["fit"]
    if dgpo.get("checkpoint_load_mode") != "weights_only":
        raise ValueError("c4a91e07 must be loaded weights-only with a fresh policy optimizer")
    if dgpo.get("steps_per_epoch") != 10:
        raise ValueError("ablation requires ten DGPO optimizer updates")
    if dgpo["reference_trust"]["adaptive_boundary"].get("enabled"):
        raise ValueError("hard trust must remain disabled")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("ablation cannot add a projection constraint")
    if (
        adaptive.get("monitor_mode") != "raw_plateau_refit"
        or adaptive["trigger"].get("rollback_to_best_on_plateau")
    ):
        raise ValueError("ablation requires raw monitoring without policy rollback")
    if (
        not recal.get("bootstrap_on_start")
        or recal.get("max_reward_rounds") != 1
        or (recal.get("min_iterations"), recal.get("max_iterations")) != (1, 1)
        or recal.get("iteration_one_only") is not True
    ):
        raise ValueError("ablation must install exactly one residual reward")
    if (recal.get("crossfit_repeats"), recal.get("crossfit_folds")) != (2, 2):
        raise ValueError("ablation requires a four-member 2-repeat x 2-fold ensemble")
    if recal.get("warm_start_iterations") or recal.get("warm_start_from_iteration_one"):
        raise ValueError("all reward classifiers must start fresh")
    if recal.get("reset_optimizer_state_on_install") is not True:
        raise ValueError("paired runs require a fresh policy optimizer at reward install")
    if fit.get("min_steps_per_fold") != 1000:
        raise ValueError("every reward member requires at least 1000 optimizer steps")
    if fit.get("checkpoint_selection_metric") != "balanced_accuracy":
        raise ValueError("reward checkpoints must use held-out balanced accuracy")

    h4 = {
        "periodic_pair_features": True,
        "topology_fourier_embedding": True,
        "topology_conditioning": False,
        "visible_pair_rest_frame": False,
        "topology_max_harmonic": 4,
        "topology_include_theta_pair": False,
        "topology_direct_logit": False,
    }
    mismatch = {
        key: (recal.get(key), value)
        for key, value in h4.items()
        if recal.get(key) != value
    }
    if mismatch:
        raise ValueError(f"reward is not the pinned nonlinear H4 model: {mismatch}")
    tempering = recal.get("adaptive_tempering") or {}
    if tempering != {
        "enabled": True,
        "target_ess_fraction": 0.15,
        "minimum": 0.10,
        "grid_steps": 14,
        "inherit_previous": False,
    }:
        raise ValueError("both arms require the same soft ESS15 tempering search")
    if recal.get("minimum_ess_fraction") != 0.0:
        raise ValueError("ESS15 is a soft tempering target, not a rejection gate")

    if experiment.get("control_wandb_run") != CONTROL_WANDB_RUN:
        raise ValueError("BA70 arm does not point to h4e15r01")
    if (
        fit.get("require_saturation") is not False
        or fit.get("minimum_sufficient_balanced_accuracy") != 0.70
        or fit.get("minimum_sufficient_confidence_z") != 0.0
        or fit.get("minimum_sufficient_required_consecutive") != 3
    ):
        raise ValueError("BA70 arm has the wrong minimum-sufficient stopping rule")
    if config["logger"]["wandb"].get("id") != "h4ba7010m4":
        raise ValueError("BA70 arm has the wrong W&B id")

    wb = config["logger"]["wandb"]
    if (
        wb.get("profile") != "standard"
        or not wb.get("classifier_loss_curves")
        or not wb.get("simplified")
        or wb.get("fresh_run")
        or wb.get("resume") != "allow"
        or not wb.get("id")
        or not experiment.get("wandb_required")
    ):
        raise ValueError("live, resumable W&B logging with classifier curves is required")
    platform = config["platform"]
    if (
        platform.get("number_of_workers") != 16
        or platform["resources_per_worker"].get("GPU") != 1
    ):
        raise ValueError("ablation requires 16 Ray workers with one GPU each")


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
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError(f"incomplete ablation last.ckpt: missing {missing}")
        step = int(resumed.get("global_step", -1))
        if not 0 <= step <= 10:
            raise ValueError(f"ablation last.ckpt step is outside [0, 10]: {step}")
        return f"resume step={step} round={resumed.get('dgpo_reward_round_id')}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("ablation output has checkpoints but no complete last.ckpt")
    return "fresh weights-only start from c4a91e07 step 1110"


def _write_exact(path: Path, payload: Any, *, checkpoints_exist: bool) -> None:
    serialized = (
        yaml.safe_dump(payload, sort_keys=False)
        if path.suffix == ".yaml"
        else json.dumps(payload, indent=2) + "\n"
    )
    if path.exists() and path.read_text() != serialized and checkpoints_exist:
        raise ValueError(f"{path.name} changed after an ablation checkpoint was written")
    path.write_text(serialized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    if config_path != BA70:
        parser.error(f"--config must be {BA70}")
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
    assert_paired_configs(config, epsilon_result)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)
    fit = config["dgpo"]["adaptive_omnifold"]["recalibration"]["fit"]
    stopping = (
        "validation saturation"
        if fit.get("minimum_sufficient_balanced_accuracy") is None
        else "BA>0.70 for 3 consecutive validations after >=1000 steps"
    )
    print(
        f"Ablation preflight passed: {mode}\n"
        f"source={EXPECTED_SOURCE_RUN}/last.ckpt step={EXPECTED_SOURCE_STEP}; "
        f"epsilon/LR={epsilon_result['epsilon_rms']:.6g}; stopping={stopping}\n"
        "Historical control=h4e15r01 step 0->10. Contract: fresh H4 2x2 reward "
        "ensemble, soft ESS15 tempering, fresh policy optimizer, exactly 10 "
        "DGPO updates, no hard trust/rollback, live W&B.",
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
    resolved_path = output_root / "resolved_overlay.yaml"
    _write_exact(resolved_path, config, checkpoints_exist=checkpoints_exist)
    manifest = {
        "schema": "h4-minimum-sufficient-reward-ablation-v1",
        "protocol": experiment["protocol"],
        "historical_control_wandb_run": CONTROL_WANDB_RUN,
        "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "epsilon_report": str(report_path),
        **epsilon_result,
        "classifier_members": 4,
        "minimum_steps_per_member": 1000,
        "minimum_sufficient_balanced_accuracy": fit.get(
            "minimum_sufficient_balanced_accuracy"
        ),
        "minimum_sufficient_confidence_z": fit.get(
            "minimum_sufficient_confidence_z"
        ),
        "minimum_sufficient_required_consecutive": fit.get(
            "minimum_sufficient_required_consecutive"
        ),
        "target_ess_fraction": 0.15,
        "policy_updates": 10,
        "wandb_run": (
            f"{config['logger']['wandb']['entity']}/"
            f"{config['logger']['wandb']['project']}/"
            f"{config['logger']['wandb']['id']}"
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
    if int(final.get("global_step", -1)) != 10:
        raise RuntimeError("ablation returned without reaching global step 10")
    if int(final.get("dgpo_reward_round_id", -1)) != 1:
        raise RuntimeError("ablation did not finish with exactly one reward round")
    print("Ablation arm complete: step=10, installed reward rounds=1.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
