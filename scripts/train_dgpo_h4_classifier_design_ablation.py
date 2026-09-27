#!/usr/bin/env python3
"""Validate and launch a fixed-policy 10% H4 classifier experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_neutrino_backend import read_overlay_yaml  # noqa: E402


BASE = ROOT / "config/train_diffusion_nersc.yaml"
CONFIG = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_classifier_direct_logit.yaml"
FOURIER_CONFIG = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_classifier_fourier.yaml"


def assert_contract(
    config: Mapping[str, Any],
) -> None:
    experiment = config["experiment"]
    reproducibility = config["nersc"]["reproducibility"]
    training = config["options"]["Training"]
    nersc = config["nersc"]
    execution = nersc["execution"]
    execution_command = str(execution.get("command", ""))
    platform = config["platform"]
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    audit = adaptive["audit_fit"]
    recal = adaptive["recalibration"]
    wandb = config["logger"]["wandb"]
    arm = experiment.get("classifier_design_arm")
    arms = {
        "direct_logit_single_probe": ("h4clf02", "h4_classifier_direct_logit_only_10pct"),
        "fourier_late_fusion": ("h4clff01", "h4_classifier_fourier_only_10pct"),
        "fourier_unfreeze_regularized": ("h4clfur1", "h4_classifier_unfreeze_reg_10pct"),
        "fourier_lastblock_regularized": ("h4clflb1", "h4_classifier_lastblock_reg_10pct"),
    }
    if arm not in arms:
        raise ValueError(f"unknown classifier design arm: {arm}")
    fourier = arm != "direct_logit_single_probe"
    expected_id, output_name = arms[arm]
    if arm in {"fourier_unfreeze_regularized", "fourier_lastblock_regularized"}:
        for key, value in {
            "train_backbone": arm == "fourier_unfreeze_regularized",
            "learning_rate": 2e-4,
            "backbone_learning_rate": 1e-5,
            "head_dropout": 0.25,
            "topology_dropout": 0.25,
            "weight_decay": 1e-3,
            "restore_best": True,
            "disjoint_final_audit": True,
        }.items():
            if audit.get(key) != value:
                raise ValueError(f"regularized unfreeze requires audit_fit.{key}={value}")
    if arm == "fourier_lastblock_regularized":
        for key, value in {
            "train_last_pet_block": True,
            "train_grouped_sequential_embedding": False,
            "train_invisible_projector": False,
            "train_encoder": False,
            "train_layernorm": False,
            "checkpoint_selection_metric": "loss",
        }.items():
            if audit.get(key) != value:
                raise ValueError(f"last-block probe requires audit_fit.{key}={value}")
    if (
        float(experiment.get("dataset_fraction", -1.0)) != 0.10
        or float(reproducibility.get("diffusion_pretrain_fraction", -1.0)) != 0.10
        or float(reproducibility.get("dgpo_training_fraction", -1.0)) != 0.10
    ):
        raise ValueError("direct probe must use the matched 10% dataset")
    data_path = str(config["platform"]["data_parquet_dir"])
    backbone_path = str(config["reward_config"]["omnifold"]["backbone_checkpoint"])
    if "10pct" not in data_path or "10pct" not in backbone_path:
        raise ValueError("direct probe does not use the 10% data and classifier backbone")
    worker_resources = dict(platform.get("resources_per_worker", {}))
    configured_gpus = int(platform.get("number_of_workers", 0)) * int(
        worker_resources.get("GPU", 0)
    )
    allocated_gpus = int(nersc.get("nodes", 0)) * int(
        nersc.get("gpus_per_node", 0)
    )
    if (
        experiment.get("distributed_world_size") != 16
        or configured_gpus != 16
        or allocated_gpus != 16
        or platform.get("number_of_workers") != 16
        or worker_resources.get("GPU") != 1
        or platform.get("use_gpu") is not True
        or execution.get("mode") != "ray_train_ddp"
        or execution.get("workers") != 16
        or execution.get("gpus_per_worker") != 1
        or not execution_command.startswith(
            "shifter python3 scripts/train_dgpo_h4_classifier_design_ablation.py"
        )
        or f"{output_name}/ray_results"
        not in execution_command
    ):
        raise ValueError(
            "direct probe must launch through Shifter as 16 one-GPU Ray/DDP workers"
        )
    policy_checkpoint = str(training.get("model_checkpoint_load_path"))
    if (
        reproducibility.get("source_wandb_run")
        != "ytchou97-university-of-washington/nu2flow-RL/c4a91e07"
        or reproducibility.get("source_dgpo_global_step") != 1110
        or "dgpo_omnifold_10pct_old_method_hard" not in policy_checkpoint
        or not policy_checkpoint.endswith((
            "/checkpoints/last.ckpt",
            "/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt",
        ))
        or training.get("pretrain_model_load_path") is not None
        or dgpo.get("checkpoint_load_mode") != "weights_only"
    ):
        raise ValueError("direct probe must start weights-only from DGPO-finetuned c4a91e07 step 1110")
    if (
        experiment.get("classifier_only") is not True
        or experiment.get("rounds") != 0
        or experiment.get("policy_updates_per_round") != 0
        or experiment.get("reward_lifetime_updates") != 0
        or experiment.get("reward_fit_count") != 0
        or experiment.get("classifier_fit_count") != 1
        or experiment.get("cold_h4_audit_policy_steps") != [0]
        or experiment.get("gradient_direction_policy_steps") != []
    ):
        raise ValueError(
            "probe must run one classifier fit, zero reward fits, and zero policy updates"
        )
    if (
        audit.get("steps") != 3000
        or audit.get("min_steps") != 3000
        or audit.get("repeats") != 1
        or audit.get("validation_interval_steps") != 40
        or audit.get("progress_every_n_steps") != 10
        or audit.get("require_saturation") is not False
        or audit.get("fail_if_unsaturated") is not False
    ):
        raise ValueError("direct cold audit is not an exact 3000-step fit")
    if (
        wandb.get("id") != expected_id
        or "classifier only" not in str(wandb.get("run_name", "")).lower()
        or wandb.get("group") != "H4 classifier design"
        or wandb.get("profile") != "standard"
        or wandb.get("classifier_loss_curves") is not True
        or wandb.get("simplified") is not True
        or "ClassifierBottleneckDiagnostics" not in wandb.get("tags", [])
        or "ClassifierOnly" not in wandb.get("tags", [])
        or "NoPolicyUpdate" not in wandb.get("tags", [])
    ):
        raise ValueError("direct probe W&B classifier diagnostics are not enabled")
    if adaptive["trigger"].get("require_audit_saturation") is not False:
        raise ValueError("direct probe still gates fixed-budget audits on patience")
    if recal.get("bootstrap_on_start") is not False:
        raise ValueError("classifier-only probe must disable reward bootstrap")
    if adaptive.get("baseline_probe_on_start") is not False:
        raise ValueError("classifier-only probe must disable the extra baseline audit")
    if dgpo.get("auto_resume_from_last") is not False:
        raise ValueError("classifier-only probe must not resume a DGPO run")
    if dgpo.get("gradient_conflict", {}).get("enabled") is not False:
        raise ValueError("classifier-only probe must not run DGPO gradient diagnostics")
    if dgpo["reference_trust"].get("enabled") is not False:
        raise ValueError("direct probe unexpectedly re-enabled reference trust")
    if recal.get("topology_direct_logit", False):
        raise ValueError("direct probe changed the installed reward")
    if audit.get("topology_direct_logit") is not (not fourier):
        raise ValueError("classifier shortcut setting disagrees with the selected arm")
    if fourier:
        required = {
            "periodic_pair_features": True,
            "topology_fourier_embedding": True,
            "topology_max_harmonic": 4,
            "topology_conditioning": False,
            "topology_include_theta_pair": False,
            "visible_pair_rest_frame": False,
            "topology_warmup_steps": 0,
            "topology_body_unfreeze_step": 0,
        }
        for key, value in required.items():
            if audit.get(key) != value:
                raise ValueError(f"Fourier-only extension requires audit_fit.{key}={value}")
    global_batch = int(audit.get("batch_size", 0))
    microbatch = int(audit.get("train_microbatch_size_per_rank", 0))
    validation_batch = int(audit.get("validation_batch_size", 0))
    if (
        global_batch <= 0
        or global_batch % 16
        or microbatch <= 0
        or validation_batch <= 0
        or validation_batch % 16
    ):
        raise ValueError("audit classifier batches must shard exactly across 16 GPUs")


def validated_config(path: Path = FOURIER_CONFIG) -> dict[str, Any]:
    config = read_overlay_yaml(path)
    assert_contract(config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--ray-dir", type=Path)
    parser.add_argument("--config", type=Path, default=FOURIER_CONFIG)
    args = parser.parse_args()
    config = validated_config(args.config)
    print(
        "[classifier-design] contract OK: DGPO-finetuned c4a91e07 step 1110, "
        "10% data, NERSC Shifter, 16 one-GPU DDP workers, weights-only start, and one "
        f"3000-step {config['experiment']['classifier_design_arm']} classifier fit; "
        "reward fits=0, policy updates=0.",
        flush=True,
    )
    if args.validate_only:
        return 0
    ray_dir = args.ray_dir
    if ray_dir is None:
        ray_dir = Path(config["nersc"]["ray"]["results_dir"])
    command = [
        sys.executable,
        str(ROOT / "scripts/train_neutrino_backend.py"),
        "--backend",
        "dgpo-evenet",
        "--base-config",
        str(BASE),
        "--overlay-config",
        str(args.config.resolve()),
        "--",
        "--ray-dir",
        str(ray_dir),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
