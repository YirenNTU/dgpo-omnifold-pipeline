#!/usr/bin/env python3
"""Launch the 20-update exact H4-to-DGPO gradient-transfer diagnostic."""

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
sys.path.insert(0, str(ROOT / "evenet_dgpo"))

from train_dgpo_h4_fresh_ensemble_pilot import (  # noqa: E402
    EXPECTED_SOURCE_RUN,
    EXPECTED_SOURCE_STEP,
    RECOVERY_KEYS,
    apply_epsilon_result,
    read_epsilon_result,
)
from train_dgpo_old_method_10pct import assert_live_ray_16gpu  # noqa: E402
from train_neutrino_backend import read_overlay_yaml  # noqa: E402
from RL.DGPO_neutrino.gradient_transfer import (  # noqa: E402
    resolve_gradient_transfer_trace_config,
)


DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_gradient_transfer_trace_20step.yaml"
BASE = ROOT / "config/train_diffusion_nersc.yaml"
BASE_PROTOCOL = "h4-gradient-transfer-trace-20step-v1"
CONTINUATION_PROTOCOL = "h4-gradient-transfer-trace-resume20-to50-v1"
REPLICATION_BASE_PROTOCOL = "h4-gradient-transfer-seed2-replication-20step-v1"
REPLICATION_CONTINUATION_PROTOCOL = (
    "h4-gradient-transfer-seed2-replication-resume20-to50-v1"
)
COUNTERFACTUAL_COEFFICIENTS = (0.0, 0.1, 0.2, 0.5, 1.0)
FIXED_SEED_BUNDLE = {
    "reward_fit": 20260920,
    "reward_pool_order": 42,
    "cold_audit": 20260921,
    "gradient_conflict_panel": 20260915,
    "tarp": 20260820,
}
INDEPENDENT_SEED_BUNDLE = {
    "reward_fit": 20261020,
    "reward_pool_order": 314159,
    "cold_audit": 20261021,
    "gradient_conflict_panel": 20261015,
    "tarp": 20261022,
}
RUN_CONTRACTS = {
    BASE_PROTOCOL: {
        "wandb_id": "h4xfer01",
        "single_experimental_change": (
            "read_only_exact_gradient_transfer_instrumentation"
        ),
        "start_global_step": 0,
        "source_checkpoint_step": EXPECTED_SOURCE_STEP,
        "endpoint_global_step": 20,
        "total_epochs": 2,
        "trace_steps": (1, 2, 5, 10, 20),
        "cold_audit_steps": (0, 10, 20),
        "checkpoint_load_mode": "weights_only",
        "bootstrap_on_start": True,
        "is_continuation": False,
        "continuation_of": None,
        "seed_bundle": FIXED_SEED_BUNDLE,
        "source_checkpoint_suffix": (
            "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_"
            "nohardtrust_seed42/checkpoints/last.ckpt"
        ),
    },
    CONTINUATION_PROTOCOL: {
        "wandb_id": "h4xfer02",
        "single_experimental_change": (
            "extend_same_fixed_reward_trajectory_step20_to50"
        ),
        "start_global_step": 20,
        "source_checkpoint_step": 20,
        "endpoint_global_step": 50,
        "total_epochs": 5,
        "trace_steps": (30, 40, 50),
        "cold_audit_steps": (30, 40, 50),
        "checkpoint_load_mode": "resume",
        "bootstrap_on_start": False,
        "is_continuation": True,
        "continuation_of": (
            "ytchou97-university-of-washington/nu2flow-RL/h4xfer01"
        ),
        "seed_bundle": FIXED_SEED_BUNDLE,
        "source_checkpoint_suffix": (
            "c4a91e07_h4_gradient_transfer_trace_20step_v1/checkpoints/last.ckpt"
        ),
    },
    REPLICATION_BASE_PROTOCOL: {
        "wandb_id": "h4rep01",
        "single_experimental_change": "independent_seed_bundle_replication",
        "start_global_step": 0,
        "source_checkpoint_step": EXPECTED_SOURCE_STEP,
        "endpoint_global_step": 20,
        "total_epochs": 2,
        "trace_steps": (1, 2, 5, 10, 20),
        "cold_audit_steps": (0, 10, 20),
        "checkpoint_load_mode": "weights_only",
        "bootstrap_on_start": True,
        "is_continuation": False,
        "continuation_of": None,
        "seed_bundle": INDEPENDENT_SEED_BUNDLE,
        "source_checkpoint_suffix": (
            "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_"
            "nohardtrust_seed42/checkpoints/last.ckpt"
        ),
    },
    REPLICATION_CONTINUATION_PROTOCOL: {
        "wandb_id": "h4rep01",
        "single_experimental_change": (
            "continue_independent_seed_replication_step20_to50"
        ),
        "start_global_step": 20,
        "source_checkpoint_step": 20,
        "endpoint_global_step": 50,
        "total_epochs": 5,
        "trace_steps": (30, 40, 50),
        "cold_audit_steps": (30, 40, 50),
        "checkpoint_load_mode": "resume",
        "bootstrap_on_start": False,
        "is_continuation": True,
        "continuation_of": (
            "ytchou97-university-of-washington/nu2flow-RL/h4rep01"
        ),
        "seed_bundle": INDEPENDENT_SEED_BUNDLE,
        "source_checkpoint_suffix": (
            "c4a91e07_h4_seed2_replication_20step_v1/checkpoints/last.ckpt"
        ),
    },
}


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


def _configured_seed_bundle(config: Mapping[str, Any]) -> dict[str, int]:
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


def _run_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = config["experiment"].get("protocol")
    try:
        return RUN_CONTRACTS[str(protocol)]
    except KeyError as exc:
        raise ValueError(f"unsupported gradient-transfer protocol: {protocol}") from exc


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    training = config["options"]["Training"]
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    trigger = adaptive["trigger"]
    audit = adaptive["audit_fit"]
    recal = adaptive["recalibration"]
    trust = dgpo["reference_trust"]
    trace = resolve_gradient_transfer_trace_config(
        dgpo.get("gradient_transfer_trace")
    )

    run_contract = _run_contract(config)
    if experiment.get("single_experimental_change") != run_contract[
        "single_experimental_change"
    ]:
        raise ValueError("gradient-transfer experiment metadata is stale")
    configured_seeds = _configured_seed_bundle(config)
    expected_seeds = dict(run_contract["seed_bundle"])
    if configured_seeds != expected_seeds:
        raise ValueError(
            "gradient-transfer fixed seed bundle changed: "
            f"expected {expected_seeds}, got {configured_seeds}"
        )
    is_continuation = bool(run_contract["is_continuation"])
    if is_continuation:
        if experiment.get("continuation_of") != run_contract["continuation_of"]:
            raise ValueError(
                "continuation must identify its pinned step-20 W&B source run"
            )
        if (
            int(experiment.get("source_global_step", -1)) != 20
            or int(experiment.get("source_reward_round_id", -1)) != 1
            or int(experiment.get("continuation_policy_updates", -1)) != 30
            or int(experiment.get("rounds", -1)) != 1
            or int(experiment.get("policy_updates_per_round", -1)) != 50
            or int(experiment.get("reward_lifetime_updates", -1)) != 50
        ):
            raise ValueError("continuation source or update horizon changed")
    if int(experiment.get("source_policy_step", -1)) != EXPECTED_SOURCE_STEP:
        raise ValueError("gradient-transfer trace must start at c4a91e07 step 1110")
    if experiment.get("source_wandb_run") != EXPECTED_SOURCE_RUN:
        raise ValueError("gradient-transfer trace has the wrong source W&B run")
    if experiment.get("applied_objective_changed") is not False:
        raise ValueError("this diagnostic must declare the applied objective unchanged")
    expected_epochs = int(run_contract["total_epochs"])
    if (
        int(training.get("total_epochs", -1)),
        int(training.get("epochs", -1)),
        int(dgpo.get("steps_per_epoch", -1)),
    ) != (expected_epochs, expected_epochs, 10):
        raise ValueError(
            f"trace requires exactly {expected_epochs} ten-update total epochs"
        )
    endpoint = int(run_contract["endpoint_global_step"])
    if int(experiment.get("endpoint_global_step", -1)) != endpoint:
        raise ValueError(f"trace endpoint must remain fixed at step {endpoint}")
    if experiment.get("endpoint_selection") != f"fixed_step_{endpoint}":
        raise ValueError(f"trace endpoint selection must be fixed_step_{endpoint}")
    if tuple(experiment.get("cold_h4_audit_policy_steps", ())) != tuple(
        run_contract["cold_audit_steps"]
    ):
        raise ValueError("cold H4 audit steps changed")
    if not is_continuation and (
        int(experiment.get("rounds", -1)),
        int(experiment.get("policy_updates_per_round", -1)),
    ) != (1, 20):
        raise ValueError("base trace requires one fixed reward for twenty updates")

    if not trace.enabled or trace.update_end_steps != tuple(run_contract["trace_steps"]):
        raise ValueError("exact gradient trace endpoints changed")
    if trace.counterfactual_trust_coefficients != COUNTERFACTUAL_COEFFICIENTS:
        raise ValueError("read-only counterfactual coefficient grid changed")
    if (
        dgpo.get("advantage_estimator") != "leave_one_out_unscaled"
        or int(dgpo.get("K", -1)) != 8
        or float(dgpo.get("beta", float("nan"))) != 1.0
        or float(dgpo.get("beta_kl", float("nan"))) != 0.0
    ):
        raise ValueError("production H4 DGPO objective changed")
    if not (
        trust.get("enabled") is True
        and trust.get("objective") == "velocity_mse"
        and math.isclose(float(trust.get("coefficient", float("nan"))), 1.0)
    ):
        raise ValueError("trace requires the production coefficient-1 velocity-MSE reference")
    if trust.get("adaptive_boundary", {}).get("enabled"):
        raise ValueError("hard trust boundary must remain disabled")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("projection must remain disabled")
    variance = dgpo.get("variance_regularization") or {}
    if variance.get("enabled") or float(variance.get("weight", 0.0)) != 0.0:
        raise ValueError("variance regularization must remain disabled")
    if dgpo.get("sequential_vp_trust_backward"):
        raise ValueError("trace requires the ordinary production backward")
    if not dgpo.get("log_parameter_update_rms") or not dgpo.get(
        "fail_on_skipped_optimizer_step"
    ):
        raise ValueError("trace must measure and require every native AdamW update")

    gradient_conflict = dgpo.get("gradient_conflict") or {}
    if not (
        gradient_conflict.get("enabled") is True
        and gradient_conflict.get("monitor_refit_lifecycle") is False
        and int(gradient_conflict.get("every_n_steps", -1)) == 10
    ):
        raise ValueError("fresh-H4 alignment probes must run every ten steps")
    if not (
        adaptive.get("enabled") is True
        and adaptive.get("monitor_mode") == "raw_plateau_refit"
        and adaptive.get("baseline_probe_on_start") is True
        and adaptive.get("fixed_audit_panel") is True
        and adaptive.get("cache_event_inputs") is True
        and adaptive.get("single_pool_train_validation") is False
        and int(adaptive.get("staleness_every_n_epochs", -1)) == 1
        and adaptive.get("staleness_every_n_steps") is None
    ):
        raise ValueError("cold H4 audit schedule, reward pools, or fixed identity panel changed")
    if trigger.get("warm_start_classifier") or trigger.get(
        "rollback_to_best_on_plateau"
    ):
        raise ValueError("H4 trajectory audits must be cold and cannot roll back policy")
    if not trigger.get("require_audit_saturation"):
        raise ValueError("H4 trajectory requires saturated audits")
    if int(audit.get("min_steps", 0)) < 1000 or not audit.get(
        "fail_if_unsaturated"
    ):
        raise ValueError("each cold H4 audit must run at least 1000 updates and saturate")
    if audit.get("training_readiness") is not None:
        raise ValueError(
            "cold H4 trace uses min_steps plus saturation, not warm-monitor training_readiness"
        )
    if recal.get("bootstrap_on_start") is not run_contract["bootstrap_on_start"]:
        raise ValueError("trace bootstrap behavior changed")
    if (
        int(recal.get("max_reward_rounds", -1)) != 1
        or recal.get("warm_start_iterations")
        or recal.get("warm_start_from_iteration_one")
        or recal.get("refit_once_on_resume")
        or (
            int(recal.get("crossfit_repeats", -1)),
            int(recal.get("crossfit_folds", -1)),
        )
        != (2, 2)
    ):
        raise ValueError("trace requires one four-member H4 reward kept at round 1")
    if (
        str(dgpo.get("checkpoint_load_mode"))
        != run_contract["checkpoint_load_mode"]
        or dgpo.get("auto_resume_from_last") is not True
    ):
        raise ValueError("checkpoint continuation mode changed")
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
        wandb.get("id") != run_contract["wandb_id"]
        or wandb.get("fresh_run")
        or wandb.get("resume") != "allow"
        or experiment.get("wandb_required") is not True
    ):
        raise ValueError(
            "fixed preempt-safe online W&B run "
            f"{run_contract['wandb_id']} is required"
        )
    platform = config["platform"]
    if (
        int(platform.get("number_of_workers", 0)) != 16
        or int(platform.get("resources_per_worker", {}).get("GPU", 0)) != 1
    ):
        raise ValueError("gradient-transfer trace requires 16 one-GPU workers")


def assert_wandb_runtime_ready() -> None:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("WANDB_DISABLED conflicts with this live-logging experiment")
    if os.environ.get("WANDB_MODE", "").lower() in {
        "offline",
        "disabled",
        "dryrun",
    }:
        raise RuntimeError("WANDB_MODE must allow online logging")
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is not installed in the launcher environment")


def verify_paths(config: Mapping[str, Any]) -> str:
    run_contract = _run_contract(config)
    training = config["options"]["Training"]
    for value in (
        training["model_checkpoint_load_path"],
        config["reward_config"]["omnifold"]["backbone_checkpoint"],
        config["platform"]["data_parquet_dir"],
        config["platform"]["data_parquet_val_dir"],
    ):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    source = Path(training["model_checkpoint_load_path"])
    if not str(source).endswith(str(run_contract["source_checkpoint_suffix"])):
        raise ValueError("gradient-transfer source checkpoint path changed")
    source_payload = _load_checkpoint(source)
    source_step = int(run_contract["source_checkpoint_step"])
    if int(source_payload.get("global_step", -1)) != source_step:
        raise ValueError(f"source checkpoint is not the required step {source_step}")
    if run_contract["checkpoint_load_mode"] == "resume":
        missing = sorted(RECOVERY_KEYS - set(source_payload))
        if missing:
            raise ValueError(f"continuation source is incomplete: missing {missing}")
        if int(source_payload.get("dgpo_reward_round_id", -1)) != 1:
            raise ValueError("continuation source no longer has fixed reward round 1")

    output = Path(training["model_checkpoint_save_path"])
    if output.resolve() == source.resolve().parent:
        raise ValueError("diagnostic output cannot overwrite the source checkpoint")
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError(f"incomplete diagnostic last.ckpt: missing {missing}")
        step = int(resumed.get("global_step", -1))
        start = int(run_contract["start_global_step"])
        endpoint = int(run_contract["endpoint_global_step"])
        if not start <= step <= endpoint:
            raise ValueError(f"resume step outside [{start}, {endpoint}]: {step}")
        if int(resumed.get("dgpo_reward_round_id", -1)) != 1:
            raise ValueError("continued output changed the fixed reward round")
        return f"resume step={step} round={resumed.get('dgpo_reward_round_id')}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("output contains checkpoints but no complete last.ckpt")
    if run_contract["checkpoint_load_mode"] == "resume":
        return "full-state continuation from h4xfer01 step 20"
    return "fresh weights-only start from c4a91e07 step 1110"


def _write_reproducibility_files(
    config: Mapping[str, Any],
    *,
    report_path: Path,
    epsilon_result: Mapping[str, Any],
) -> Path:
    output_root = Path(
        config["options"]["Training"]["model_checkpoint_save_path"]
    ).parent
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt"))
    resolved_path = output_root / "resolved_overlay.yaml"
    if resolved_path.exists():
        existing = yaml.safe_load(resolved_path.read_text())
        if existing != config:
            if has_checkpoint:
                raise ValueError("resolved overlay changed after a checkpoint was written")
            resolved_path.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    else:
        with resolved_path.open("x") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)

    run_contract = _run_contract(config)
    experiment = config["experiment"]
    manifest_payload = {
        "schema": str(experiment["protocol"]),
        "wandb_run_id": str(run_contract["wandb_id"]),
        "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_policy_step": EXPECTED_SOURCE_STEP,
        "continuation_of": experiment.get("continuation_of"),
        "single_experimental_change": experiment["single_experimental_change"],
        "seed_bundle": _configured_seed_bundle(config),
        "source_global_step": int(run_contract["start_global_step"]),
        "endpoint_global_step": int(run_contract["endpoint_global_step"]),
        "applied_objective_changed": False,
        "applied_reference_trust_coefficient": 1.0,
        "gradient_trace_update_end_steps": list(run_contract["trace_steps"]),
        "cold_h4_audit_policy_steps": list(run_contract["cold_audit_steps"]),
        "cold_h4_audit_min_optimizer_updates": 1000,
        "classifier_members": 4,
        "epsilon_report": str(report_path),
        **dict(epsilon_result),
    }
    manifest = output_root / "experiment_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != manifest_payload:
            if has_checkpoint:
                raise ValueError("manifest changed after a checkpoint was written")
            manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    else:
        with manifest.open("x") as handle:
            json.dump(manifest_payload, handle, indent=2)
            handle.write("\n")
    return resolved_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)

    config = read_overlay_yaml(args.config.expanduser().resolve())
    experiment = config["experiment"]
    run_contract = _run_contract(config)
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
        "Gradient-transfer preflight passed: "
        f"{mode}; LR={epsilon_result['epsilon_rms']:.6g}, "
        f"reward tempering={epsilon_result['tempering']:.6g}.\n"
        "Contract: 16 GPUs, one four-member fixed H4 reward, "
        f"unchanged-objective AdamW trajectory through step "
        f"{run_contract['endpoint_global_step']}, exact traces at "
        f"{list(run_contract['trace_steps'])}, cold saturated >=1000-update "
        f"H4 audits at {list(run_contract['cold_audit_steps'])}, "
        "no hard boundary/projection/refit, live W&B "
        f"{run_contract['wandb_id']}. Seed bundle="
        f"{_configured_seed_bundle(config)}.",
        flush=True,
    )
    if args.check_only:
        return 0

    assert_wandb_runtime_ready()
    assert_live_ray_16gpu()
    resolved_path = _write_reproducibility_files(
        config,
        report_path=report_path,
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
            str(resolved_path),
            "--",
            "--ray-dir",
            str(config["nersc"]["ray"]["results_dir"]),
        ],
        cwd=ROOT,
        check=True,
    )
    final = _load_checkpoint(
        Path(config["options"]["Training"]["model_checkpoint_save_path"])
        / "last.ckpt"
    )
    endpoint = int(run_contract["endpoint_global_step"])
    if int(final.get("global_step", -1)) != endpoint:
        raise RuntimeError(
            f"diagnostic returned before the fixed step-{endpoint} endpoint"
        )
    if int(final.get("dgpo_reward_round_id", -1)) != 1:
        raise RuntimeError("diagnostic changed the fixed H4 reward during the trajectory")
    print(
        f"Gradient-transfer diagnostic complete: endpoint step={endpoint}, "
        "reward round=1, "
        f"results logged to W&B run {run_contract['wandb_id']}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
