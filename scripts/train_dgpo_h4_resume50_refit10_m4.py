#!/usr/bin/env python3
"""Run an open-ended h4e20m6 continuation with fresh H4 rewards every 10 steps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_fresh_ensemble_pilot import (
    apply_epsilon_result,
    assert_live_ray_16gpu,
    assert_wandb_runtime_ready,
    read_epsilon_result,
)
from train_neutrino_backend import read_overlay_yaml


DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble4_ess15_refit10_keepadam_resume50_unbounded.yaml"
BASE = ROOT / "config/train_diffusion_nersc.yaml"
PROTOCOL = "h4-fresh-ensemble4-ess15-refit10-keepadam-resume50-unbounded-v2"
SOURCE_RUN = "ytchou97-university-of-washington/nu2flow-RL/h4e20m6"
SOURCE = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "c4a91e07_h4_ensemble6_ess15_refit20_keepadam_50step_v1/"
    "checkpoints/last.ckpt"
)
SOURCE_STEP = 50
SOURCE_EPOCH = 4
SOURCE_NEXT_EPOCH = 5
SOURCE_ROUND = 3
UNBOUNDED_EPOCH_SENTINEL = 2_147_483_647

RECOVERY_KEYS = {
    "state_dict",
    "dgpo_checkpoint_version",
    "dgpo_next_epoch",
    "dgpo_optimizer_state_dict",
    "dgpo_ref_state_dict",
    "dgpo_round_ref_state_dict",
    "dgpo_round_ref_sha256",
    "dgpo_omnifold_reward_stack",
    "dgpo_omnifold_reward_metadata",
    "dgpo_adaptive_omnifold_state",
}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _assert_complete_adam(payload: Mapping[str, Any]) -> None:
    saved = payload.get("dgpo_optimizer_state_dict")
    if not isinstance(saved, Mapping):
        raise ValueError("resume checkpoint has no DGPO optimizer state")
    if not {"optimizer", "scheduler"}.issubset(saved):
        raise ValueError("resume checkpoint lacks AdamW or scheduler state")
    optimizer = saved["optimizer"]
    if not isinstance(optimizer, Mapping) or not optimizer.get("state"):
        raise ValueError("resume checkpoint has no populated AdamW moments")
    states = list(optimizer["state"].values())
    if not states or not all(
        isinstance(item, Mapping) and "exp_avg" in item and "exp_avg_sq" in item
        for item in states
    ):
        raise ValueError("resume checkpoint does not contain complete AdamW moments")


def _assert_six_member_source(payload: Mapping[str, Any]) -> None:
    reward = dict(payload["dgpo_omnifold_reward_stack"].get("reward") or {})
    increments = list(reward.get("increments") or [])
    coefficients = list(reward.get("increment_coefficients") or [])
    iterations = list(reward.get("increment_iterations") or [])
    protocol = dict((reward.get("warm_start_state") or {}).get("protocol") or {})
    if (
        len(increments) != 6
        or len(coefficients) != 6
        or iterations != [1] * 6
        or any(not item.get("state") for item in increments)
        or any(
            not math.isclose(float(value), 1.0 / 6.0, rel_tol=1.0e-6, abs_tol=1.0e-6)
            for value in coefficients
        )
        or protocol.get("scheme") != "condition_sha256_v1"
        or int(protocol.get("folds", -1)) != 2
        or int(protocol.get("repeats", -1)) != 3
    ):
        raise ValueError("source checkpoint is not h4e20m6's complete six-member reward")


def assert_source_checkpoint(payload: Mapping[str, Any]) -> None:
    missing = sorted(RECOVERY_KEYS - set(payload))
    if missing:
        raise ValueError("incomplete h4e20m6 checkpoint: " + ", ".join(missing))
    observed = (
        int(payload.get("global_step", -1)),
        int(payload.get("epoch", -1)),
        int(payload.get("dgpo_next_epoch", -1)),
        int(payload.get("dgpo_epoch_step", -1)),
        int(payload.get("dgpo_reward_round_id", -1)),
    )
    expected = (
        SOURCE_STEP,
        SOURCE_EPOCH,
        SOURCE_NEXT_EPOCH,
        0,
        SOURCE_ROUND,
    )
    if observed != expected:
        raise ValueError(f"expected completed h4e20m6 state {expected}, got {observed}")
    state = dict(payload.get("dgpo_adaptive_omnifold_state") or {})
    if int(state.get("reward_round_id", -1)) != SOURCE_ROUND:
        raise ValueError("source adaptive state and checkpoint reward round disagree")
    _assert_complete_adam(payload)
    _assert_six_member_source(payload)


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    training = config["options"]["Training"]
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    trigger = adaptive["trigger"]
    audit = adaptive["audit_fit"]
    recal = adaptive["recalibration"]

    if experiment.get("protocol") != PROTOCOL:
        raise ValueError("wrong continuation protocol")
    expected_experiment = {
        "control_wandb_run": SOURCE_RUN,
        "source_wandb_run": SOURCE_RUN,
        "source_global_step": SOURCE_STEP,
        "source_reward_round_id": SOURCE_ROUND,
        "continuation_policy_updates": None,
        "endpoint_global_step": None,
        "rounds": None,
        "policy_updates_per_round": 10,
        "reward_lifetime_updates": 10,
        "reward_refresh_start_step": 50,
        "reward_refresh_every_updates": 10,
        "reward_refresh_steps": None,
        "endpoint_selection": "manual_stop",
        "ensemble_seeds": 2,
        "crossfit_folds": 2,
    }
    mismatch = {
        key: (experiment.get(key), value)
        for key, value in expected_experiment.items()
        if experiment.get(key) != value
    }
    if mismatch:
        raise ValueError(f"continuation experiment metadata mismatch: {mismatch}")
    if (
        int(training.get("epochs", -1)),
        int(training.get("total_epochs", -1)),
        int(dgpo.get("steps_per_epoch", -1)),
    ) != (UNBOUNDED_EPOCH_SENTINEL, UNBOUNDED_EPOCH_SENTINEL, 10):
        raise ValueError("formal continuation must use the open-ended epoch sentinel")
    if Path(training["model_checkpoint_load_path"]) != SOURCE:
        raise ValueError("continuation must load h4e20m6/checkpoints/last.ckpt")
    if Path(training["model_checkpoint_save_path"]).resolve() == SOURCE.parent.resolve():
        raise ValueError("continuation output must not overwrite h4e20m6")
    if (
        dgpo.get("checkpoint_load_mode") != "resume"
        or not dgpo.get("auto_resume_from_last")
        or dgpo.get("pinned_classifier_restart")
    ):
        raise ValueError("full-state continuation requires resume without clock restart")
    if dgpo.get("lr_schedule", {}).get("type") != "constant":
        raise ValueError("continuation must preserve the parent constant LR schedule")
    if dgpo["reference_trust"]["adaptive_boundary"].get("enabled"):
        raise ValueError("hard trust must remain disabled")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("continuation cannot add a projection constraint")
    if (
        adaptive.get("staleness_every_n_epochs") != 1
        or adaptive.get("staleness_every_n_steps") is not None
        or trigger.get("max_reward_age_epochs") != 1
    ):
        raise ValueError("reward boundaries must occur every ten updates")
    if trigger.get("fixed_schedule_skip_staleness_audit") is not True:
        raise ValueError(
            "fixed ten-step refits must skip classifier-based staleness audits"
        )
    if int(audit.get("min_steps", 0)) < 1000:
        raise ValueError("H4 raw judges require at least 1000 optimizer updates")
    h4 = {
        "periodic_pair_features": True,
        "topology_fourier_embedding": True,
        "topology_conditioning": False,
        "visible_pair_rest_frame": False,
        "topology_max_harmonic": 4,
        "topology_include_theta_pair": False,
        "topology_direct_logit": False,
    }
    for label, block in (("reward", recal), ("judge", audit)):
        mismatch = {
            key: (block.get(key), value)
            for key, value in h4.items()
            if block.get(key) != value
        }
        if mismatch:
            raise ValueError(f"{label} is not the pinned H4 classifier: {mismatch}")
    if (
        recal.get("bootstrap_on_start")
        or not recal.get("refit_once_on_resume")
        or not recal.get("refit_once_fail_closed")
        or recal.get("refit_once_id")
        != "h4_m6_to_m4_refit10_keepadam_step50_no_staleness_v2"
    ):
        raise ValueError("six-to-four migration needs one versioned fail-closed step-50 refit")
    if recal.get("warm_start_iterations") or recal.get("warm_start_from_iteration_one"):
        raise ValueError("every reward classifier must start from fresh weights and AdamW")
    if (
        int(recal.get("crossfit_repeats", -1)),
        int(recal.get("crossfit_folds", -1)),
    ) != (2, 2):
        raise ValueError("continuation requires a four-member reward ensemble")
    if (
        recal.get("reset_optimizer_state_on_install") is not False
        or dgpo["reference_trust"]["adaptive_boundary"].get(
            "reset_adam_first_moment_on_install", False
        )
        or dgpo["reference_trust"]["adaptive_boundary"].get(
            "reset_adam_first_moment_on_zero_step", False
        )
    ):
        raise ValueError("policy AdamW state must remain continuous across refits")
    if (
        int(recal.get("min_iterations", -1)),
        int(recal.get("max_iterations", -1)),
        recal.get("iteration_one_only"),
        recal.get("max_reward_rounds"),
    ) != (1, 1, True, None):
        raise ValueError("formal continuation must keep fitting reward rounds without a cap")
    if int(recal["fit"].get("min_steps_per_fold", 0)) < 1000:
        raise ValueError("every fresh H4 reward member requires at least 1000 updates")
    if recal.get("scheduled_refit_fail_closed") is not True:
        raise ValueError("a failed scheduled reward fit must stop the continuation")
    if recal.get("acceptance_audit_enabled") or recal.get("topology_acceptance_audit_enabled"):
        raise ValueError("formal run cannot use metric gates to change its trajectory")
    tempering = recal.get("adaptive_tempering") or {}
    expected_tempering = {
        "enabled": True,
        "target_ess_fraction": 0.15,
        "minimum": 0.10,
        "grid_steps": 14,
        "inherit_previous": False,
    }
    if tempering != expected_tempering or float(recal.get("minimum_ess_fraction", 0.0)) != 0.0:
        raise ValueError("continuation must preserve the soft ESS15 protocol")
    wb = config["logger"]["wandb"]
    if (
        wb.get("id") != "h4e10m4fs"
        or wb.get("fresh_run") is not False
        or wb.get("resume") != "allow"
        or wb.get("profile") != "standard"
    ):
        raise ValueError("continuation requires its fixed live W&B run")
    platform = config["platform"]
    if (
        int(platform.get("number_of_workers", 0)) != 16
        or int(platform["resources_per_worker"].get("GPU", 0)) != 1
    ):
        raise ValueError("continuation requires 16 one-GPU Ray workers")


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

    output = Path(training["model_checkpoint_save_path"])
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError("incomplete continuation last.ckpt: " + ", ".join(missing))
        step = int(resumed.get("global_step", -1))
        next_epoch = int(resumed.get("dgpo_next_epoch", -1))
        reward_round = int(resumed.get("dgpo_reward_round_id", -1))
        if step < SOURCE_STEP:
            raise ValueError(f"continuation step is below its step-50 source: {step}")
        if not SOURCE_NEXT_EPOCH <= next_epoch <= UNBOUNDED_EPOCH_SENTINEL:
            raise ValueError(f"continuation next epoch is invalid: {next_epoch}")
        if reward_round < SOURCE_ROUND + 1:
            raise ValueError(f"continuation reward round is below 4: {reward_round}")
        _assert_complete_adam(resumed)
        return f"resume continuation step={step} next_epoch={next_epoch} round={reward_round}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("continuation output has checkpoints but no complete last.ckpt")

    source = _load_checkpoint(SOURCE)
    assert_source_checkpoint(source)
    return "full-state h4e20m6 step-50 source; AdamW moments verified"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config = read_overlay_yaml(args.config.resolve())
    experiment = config["experiment"]
    report_path = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else Path(experiment["epsilon_sweep_report"]).expanduser().resolve()
    )
    result = read_epsilon_result(report_path, experiment)
    apply_epsilon_result(config, result)
    assert_contract(config)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)

    print(
        "Continuation preflight passed: " + mode + "\n"
        f"Source: h4e20m6 step={SOURCE_STEP} round={SOURCE_ROUND}; "
        "full policy AdamW state preserved\n"
        "Protocol: fresh 2x2 H4 reward at step 50 and every 10 updates; "
        "soft ESS>=15%; no online staleness audit; no fixed endpoint; "
        "live W&B=h4e10m4fs",
        flush=True,
    )
    if args.check_only:
        return 0

    assert_wandb_runtime_ready()
    assert_live_ray_16gpu()
    output_root = Path(config["options"]["Training"]["model_checkpoint_save_path"]).parent
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt"))
    resolved_path = output_root / "resolved_overlay.yaml"
    if resolved_path.exists():
        existing = yaml.safe_load(resolved_path.read_text())
        if existing != config:
            if has_checkpoint:
                raise ValueError("resolved continuation overlay changed after a checkpoint was written")
            resolved_path.write_text(yaml.safe_dump(config, sort_keys=False))
    else:
        with resolved_path.open("x") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    manifest_path = output_root / "experiment_manifest.json"
    manifest = {
        "protocol": PROTOCOL,
        "source_wandb_run": SOURCE_RUN,
        "source_checkpoint": str(SOURCE),
        "source_global_step": SOURCE_STEP,
        "source_reward_round_id": SOURCE_ROUND,
        "endpoint_global_step": None,
        "endpoint_reward_round_id": None,
        "continuation_policy_updates": None,
        "reward_refresh_start_step": 50,
        "reward_refresh_every_updates": 10,
        "classifier_members_per_round": 4,
        "reset_policy_optimizer_on_reward_install": False,
        "preserve_source_policy_optimizer": True,
        "fresh_classifier_optimizer_every_round": True,
        "online_staleness_audit": False,
        "independent_raw_audit": "offline_from_saved_checkpoints",
        "target_ess_fraction": 0.15,
        "epsilon_report": str(report_path),
        **result,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            if has_checkpoint:
                raise ValueError("continuation manifest changed after a checkpoint was written")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")

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
    if int(final.get("global_step", -1)) < SOURCE_STEP:
        raise RuntimeError("formal continuation returned an invalid checkpoint")
    _assert_complete_adam(final)
    print(
        "Continuation process exited with a valid resumable checkpoint at "
        f"step={final.get('global_step')} round={final.get('dgpo_reward_round_id')}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
