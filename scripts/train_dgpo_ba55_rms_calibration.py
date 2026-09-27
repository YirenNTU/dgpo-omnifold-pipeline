#!/usr/bin/env python3
"""Run one-step native versus RMS-calibrated AdamW BA55 arms."""

from __future__ import annotations

import argparse
import copy
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
    assert_wandb_runtime_ready,
    read_epsilon_result,
)
from train_dgpo_old_method_10pct import assert_live_ray_16gpu
from train_neutrino_backend import read_overlay_yaml


BASE = ROOT / "config/train_diffusion_nersc.yaml"
NATIVE = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba55_m2_"
    "adamw_native_1step.yaml"
)
CALIBRATED = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba55_m2_"
    "adamw_rms3e5_1step.yaml"
)
LR1E5 = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_ba55_m2_"
    "adamw_lr1e5_1step.yaml"
)
SOURCE_CHECKPOINT = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/"
    "checkpoints/last.ckpt"
)
PROTOCOL = "h4-ba55-m2-adamw-rms-one-step-v1"
LR_PROTOCOL = "h4-ba55-m2-adamw-lr-one-step-v1"
TARGET_RMS = 3.0e-5
LR1E5_VALUE = 1.0e-5


def apply_policy_lr_only(
    config: dict[str, Any], result: Mapping[str, Any]
) -> None:
    """Resolve the arm's AdamW LR without changing BA55 tempering."""

    epsilon = float(
        config["experiment"].get("native_adamw_lr_override")
        or result["epsilon_rms"]
    )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon result must contain a positive finite radius")
    training = config["options"]["Training"]
    training["learning_rate"] = epsilon
    training["learning_rate_body"] = 0.1 * epsilon
    for name in (
        "InvisibleInputProjector",
        "GroupedSequentialEmbedding",
        "GlobalEmbedding",
        "TruthGeneration",
    ):
        training["Components"][name]["learning_rate"] = epsilon
    training["Components"]["PET"]["learning_rate"] = 0.1 * epsilon
    experiment = config["experiment"]
    experiment["resolved_native_adamw_lr"] = epsilon
    experiment["resolved_reward_tempering"] = 0.75


def _paired_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return every field allowed to affect the paired scientific result."""

    payload = copy.deepcopy(dict(config))
    payload.pop("experiment", None)
    payload.pop("nersc", None)
    payload.pop("logger", None)
    training = payload["options"]["Training"]
    training.pop("model_checkpoint_save_path", None)
    payload["dgpo"].pop("parameter_update_rms_calibration", None)
    return payload


def _lr_agnostic_paired_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the paired signature after masking only policy LR fields."""

    payload = _paired_signature(config)
    training = payload["options"]["Training"]
    training["learning_rate"] = "<policy-lr>"
    training["learning_rate_body"] = "<body-lr>"
    for name in (
        "InvisibleInputProjector",
        "GroupedSequentialEmbedding",
        "GlobalEmbedding",
        "TruthGeneration",
    ):
        training["Components"][name]["learning_rate"] = "<policy-lr>"
    training["Components"]["PET"]["learning_rate"] = "<body-lr>"
    return payload


def assert_paired_configs(
    candidate: Mapping[str, Any], epsilon_result: Mapping[str, Any]
) -> None:
    native = read_overlay_yaml(NATIVE)
    calibrated = read_overlay_yaml(CALIBRATED)
    apply_policy_lr_only(native, epsilon_result)
    apply_policy_lr_only(calibrated, epsilon_result)
    if _paired_signature(native) != _paired_signature(calibrated):
        raise ValueError(
            "native and calibrated arms differ outside RMS calibration and "
            "isolated metadata/output paths"
        )
    arm = str(candidate["experiment"].get("arm"))
    if arm == "native_adamw_lr1e5":
        lr1e5 = read_overlay_yaml(LR1E5)
        apply_policy_lr_only(lr1e5, epsilon_result)
        if _lr_agnostic_paired_signature(lr1e5) != _lr_agnostic_paired_signature(native):
            raise ValueError("1e-5 arm differs from native control outside policy LR")
        if _paired_signature(candidate) != _paired_signature(lr1e5):
            raise ValueError("selected config is not the predeclared 1e-5 LR arm")
    elif _paired_signature(candidate) != _paired_signature(native):
        raise ValueError("selected config is not one of the predeclared RMS arms")


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    arm = str(experiment.get("arm"))
    expected_protocol = (
        LR_PROTOCOL if arm == "native_adamw_lr1e5" else PROTOCOL
    )
    if experiment.get("protocol") != expected_protocol:
        raise ValueError("unsupported BA55 RMS-calibration protocol")
    if arm not in {
        "native_adamw",
        "rms_calibrated_adamw",
        "native_adamw_lr1e5",
    }:
        raise ValueError("unknown BA55 RMS-calibration arm")
    if (
        experiment.get("rounds") != 1
        or experiment.get("policy_updates_per_round") != 1
        or experiment.get("endpoint_global_step") != 1
        or experiment.get("endpoint_selection") != "fixed_step_1_cold_h4"
    ):
        raise ValueError("experiment must use one reward and one policy update")
    reproducibility = config["nersc"]["reproducibility"]
    if (
        reproducibility.get("source_wandb_run") != EXPECTED_SOURCE_RUN
        or reproducibility.get("source_dgpo_global_step") != EXPECTED_SOURCE_STEP
    ):
        raise ValueError("source must remain c4a91e07 step 1110")

    training = config["options"]["Training"]
    if Path(training["model_checkpoint_load_path"]) != SOURCE_CHECKPOINT:
        raise ValueError("experiment must load c4a91e07 last.ckpt")
    if (training.get("epochs"), training.get("total_epochs")) != (1, 1):
        raise ValueError("experiment must run one logical epoch")
    if Path(training["model_checkpoint_save_path"]).parent == SOURCE_CHECKPOINT.parent:
        raise ValueError("experiment output cannot overwrite the source run")

    dgpo = config["dgpo"]
    if dgpo.get("checkpoint_load_mode") != "weights_only":
        raise ValueError("policy source must be loaded weights-only")
    if dgpo.get("steps_per_epoch") != 1:
        raise ValueError("one logical epoch must contain one optimizer update")
    if not dgpo.get("log_parameter_update_rms"):
        raise ValueError("actual parameter displacement must be logged")
    if not dgpo.get("fail_on_skipped_optimizer_step"):
        raise ValueError("the fixed endpoint cannot accept a skipped update")
    if dgpo["reference_trust"]["adaptive_boundary"].get("enabled"):
        raise ValueError("hard trust must remain disabled")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("post-AdamW projection must remain disabled")

    adaptive = dgpo["adaptive_omnifold"]
    recal = adaptive["recalibration"]
    fit = recal["fit"]
    audit = adaptive["audit_fit"]
    if (
        not adaptive.get("baseline_probe_on_start")
        or adaptive.get("staleness_every_n_epochs") != 1
        or not adaptive["trigger"].get("raw_audit_enabled")
        or adaptive["trigger"].get("warm_start_classifier")
        or not adaptive["trigger"].get("require_audit_saturation")
    ):
        raise ValueError("both step-0 and step-1 audits must be cold and saturated")
    if (
        int(audit.get("min_steps", 0)) < 1000
        or audit.get("require_saturation") is not True
        or audit.get("checkpoint_selection_metric") != "balanced_accuracy"
    ):
        raise ValueError("cold endpoint H4 audit must use >=1000 updates and saturate")
    if (
        not recal.get("bootstrap_on_start")
        or (recal.get("min_iterations"), recal.get("max_iterations")) != (1, 1)
        or recal.get("iteration_one_only") is not True
        or recal.get("max_reward_rounds") != 1
        or (recal.get("crossfit_repeats"), recal.get("crossfit_folds")) != (1, 2)
        or recal.get("seed") != 20260913
        or recal.get("reset_optimizer_state_on_install") is not True
        or not math.isclose(float(recal.get("tempering")), 0.75)
        or bool((recal.get("adaptive_tempering") or {}).get("enabled"))
        or float(recal.get("minimum_ess_fraction")) != 0.0
    ):
        raise ValueError("reward must be the fixed BA55 two-fold/one-repeat protocol")
    if (
        fit.get("require_saturation") is not False
        or fit.get("minimum_sufficient_balanced_accuracy") != 0.55
        or fit.get("minimum_sufficient_confidence_z") != 1.96
        or fit.get("minimum_sufficient_required_consecutive") != 1
        or fit.get("checkpoint_selection_metric") != "balanced_accuracy"
    ):
        raise ValueError("reward classifiers must stop at the BA55 LCB checkpoint")

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

    calibration = dgpo.get("parameter_update_rms_calibration") or {}
    target = experiment.get("target_parameter_update_rms")
    if arm == "native_adamw":
        if calibration.get("enabled") or target is not None:
            raise ValueError("native control must not calibrate its AdamW proposal")
        expected_id = "ba55nat1"
        expected_lr = float(experiment["resolved_native_adamw_lr"])
    elif arm == "rms_calibrated_adamw":
        if (
            calibration.get("enabled") is not True
            or not math.isclose(float(calibration.get("target_rms")), TARGET_RMS)
            or not math.isclose(float(target), TARGET_RMS)
            or float(calibration.get("minimum_scale")) != 1.0
            or float(calibration.get("maximum_scale")) != 100.0
        ):
            raise ValueError("treatment must target exactly 3e-5 RMS")
        expected_id = "ba55rms1"
        expected_lr = float(experiment["resolved_native_adamw_lr"])
    else:
        if calibration.get("enabled") or target is not None:
            raise ValueError("1e-5 LR arm must use an uncalibrated AdamW proposal")
        if (
            not math.isclose(
                float(experiment.get("native_adamw_lr_override")), LR1E5_VALUE
            )
            or experiment.get("single_experimental_change")
            != "native_adamw_learning_rate"
            or experiment.get("paired_run_id") != "ba55nat1"
        ):
            raise ValueError("1e-5 LR arm metadata is not predeclared")
        expected_id = "ba55lr10"
        expected_lr = LR1E5_VALUE

    if not math.isclose(float(training["learning_rate"]), expected_lr):
        raise ValueError("resolved policy learning rate does not match the arm")
    if not math.isclose(float(training["learning_rate_body"]), 0.1 * expected_lr):
        raise ValueError("resolved body learning rate does not match the arm")
    for name in (
        "InvisibleInputProjector",
        "GroupedSequentialEmbedding",
        "GlobalEmbedding",
        "TruthGeneration",
    ):
        if not math.isclose(
            float(training["Components"][name]["learning_rate"]), expected_lr
        ):
            raise ValueError(f"{name} learning rate does not match the arm")
    if not math.isclose(
        float(training["Components"]["PET"]["learning_rate"]),
        0.1 * expected_lr,
    ):
        raise ValueError("PET learning rate does not match the arm")

    wb = config["logger"]["wandb"]
    if (
        wb.get("id") != expected_id
        or wb.get("profile") != "standard"
        or not wb.get("classifier_loss_curves")
        or not wb.get("simplified")
        or wb.get("fresh_run")
        or wb.get("resume") != "allow"
        or " | " not in str(wb.get("run_name", ""))
    ):
        raise ValueError("live W&B identity/profile is not the predeclared readable run")
    platform = config["platform"]
    if (
        platform.get("number_of_workers") != 16
        or platform["resources_per_worker"].get("GPU") != 1
    ):
        raise ValueError("experiment requires 16 Ray workers with one GPU each")


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
            raise ValueError(f"incomplete one-step checkpoint: missing {missing}")
        step = int(resumed.get("global_step", -1))
        if not 0 <= step <= 1:
            raise ValueError(f"one-step checkpoint is outside [0, 1]: {step}")
        return f"resume step={step} round={resumed.get('dgpo_reward_round_id')}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("output contains checkpoints but no complete last.ckpt")
    return "fresh weights-only start from c4a91e07 step 1110"


def _write_exact(path: Path, payload: Any, *, checkpoints_exist: bool) -> None:
    serialized = (
        yaml.safe_dump(payload, sort_keys=False)
        if path.suffix == ".yaml"
        else json.dumps(payload, indent=2) + "\n"
    )
    if path.exists() and path.read_text() != serialized and checkpoints_exist:
        raise ValueError(f"{path.name} changed after a checkpoint was written")
    path.write_text(serialized)


def _assert_endpoint_audit(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    history = list(
        (checkpoint.get("dgpo_adaptive_omnifold_state") or {}).get(
            "probe_history", []
        )
        or []
    )
    rows = [
        row for row in history
        if isinstance(row, Mapping)
        and int(row.get("global_step", -1)) == 1
        and math.isfinite(float(row.get("raw_auc_gap", float("nan"))))
    ]
    if not rows:
        raise RuntimeError("step-1 checkpoint has no valid cold H4 audit")
    row = rows[-1]
    if int(row.get("raw_audit_training_steps", 0)) < 1000:
        raise RuntimeError("step-1 H4 audit did not complete 1000 updates")
    if float(row.get("raw_audit_saturated", 0.0)) < 0.5:
        raise RuntimeError("step-1 H4 audit did not reach saturation")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    if config_path not in {NATIVE, CALIBRATED, LR1E5}:
        parser.error(f"--config must be {NATIVE}, {CALIBRATED}, or {LR1E5}")
    config = read_overlay_yaml(config_path)
    experiment = config["experiment"]
    report_path = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else Path(experiment["epsilon_sweep_report"]).expanduser().resolve()
    )
    epsilon_result = read_epsilon_result(report_path, experiment)
    apply_policy_lr_only(config, epsilon_result)
    assert_contract(config)
    assert_paired_configs(config, epsilon_result)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)
    calibration = config["dgpo"]["parameter_update_rms_calibration"]
    print(
        f"BA55 AdamW RMS preflight passed: arm={experiment['arm']}; {mode}\n"
        f"source={EXPECTED_SOURCE_RUN}/last.ckpt step={EXPECTED_SOURCE_STEP}; "
        f"native_lr={epsilon_result['epsilon_rms']:.6g}; "
        f"calibration={calibration}\n"
        "Contract: one-repeat/two-fold BA55-LCB reward, raw LOO, fixed 0.75 "
        "tempering, K=8, one native AdamW proposal, cold saturated H4 audits "
        "at step 0 and step 1, live W&B.",
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
        "schema": experiment["protocol"],
        "arm": experiment["arm"],
        "paired_run_id": experiment["paired_run_id"],
        "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "epsilon_report": str(report_path),
        "native_adamw_lr": float(epsilon_result["epsilon_rms"]),
        "reward_members": 2,
        "reward_stopping": "BA55 LCB z=1.96 first checkpoint",
        "reward_tempering": 0.75,
        "policy_updates": 1,
        "parameter_update_rms_calibration": calibration,
        "primary_endpoint": "step1 cold saturated H4 raw_auc_gap",
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
    if int(final.get("global_step", -1)) != 1:
        raise RuntimeError("experiment returned without reaching global step 1")
    if int(final.get("dgpo_reward_round_id", -1)) != 1:
        raise RuntimeError("experiment did not finish with exactly one reward round")
    endpoint = _assert_endpoint_audit(final)
    print(
        "BA55 AdamW arm complete: "
        f"arm={experiment['arm']} step=1 "
        f"cold_H4_gap={float(endpoint['raw_auc_gap']):.8g} "
        f"audit_steps={int(endpoint['raw_audit_training_steps'])}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
