#!/usr/bin/env python3
"""Cold classifier tests on fixed step-50 or old-DGPO policy checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_neutrino_backend import read_overlay_yaml

CONFIG = ROOT / "config/dgpo_h4_classifier_lr_stability.yaml"
DIAGNOSTIC_CONFIG = ROOT / "config/dgpo_h4_classifier_stability_diagnostics.yaml"
SCHEDULER_CONFIG = ROOT / "config/dgpo_h4_classifier_cosine.yaml"
OLD_CHECKPOINT_CONFIG = ROOT / "config/dgpo_h4_classifier_cosine_old_checkpoint.yaml"
PARENT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_lastblock_no_ref_100step.yaml"


def validated_config(path=CONFIG):
    path = Path(path)
    config = read_overlay_yaml(path)
    parent = read_overlay_yaml(PARENT)
    adaptive = config["dgpo"]["adaptive_omnifold"]
    audit = adaptive["audit_fit"]
    expected = dict(parent["dgpo"]["adaptive_omnifold"]["audit_fit"])
    expected.update(adapter_learning_rate=5e-5, decoder_learning_rate=5e-5,
                    decoder_learning_rate_scope="all", log_parameter_updates=True)
    if path in (DIAGNOSTIC_CONFIG, SCHEDULER_CONFIG, OLD_CHECKPOINT_CONFIG):
        expected.update(diagnostic_enabled=True, diagnostic_interval_steps=10, diagnostic_probe_rows=16,
                        diagnostic_snapshot_dir="/pscratch/sd/y/yiren/Ztautau/h4_classifier_stability_diagnostics/diagnostic_snapshots",
                        diagnostic_max_snapshots=2, diagnostic_gradient_threshold=1000.0,
                        diagnostic_spike_factor=20.0, diagnostic_probe_bce_jump=0.02)
        if path in (SCHEDULER_CONFIG, OLD_CHECKPOINT_CONFIG):
            schedule = dict(lr_scheduler="cosine", lr_warmup_epochs=1.0,
                            lr_cosine_epochs=250.0, lr_min_ratio=.1)
            expected.update(**schedule, diagnostic_snapshot_dir="/pscratch/sd/y/yiren/Ztautau/h4_classifier_cosine/diagnostic_snapshots")
            if path == OLD_CHECKPOINT_CONFIG:
                expected['diagnostic_snapshot_dir'] = '/pscratch/sd/y/yiren/Ztautau/h4_classifier_cosine_old_checkpoint/diagnostic_snapshots'
            if any(adaptive["recalibration"]["fit"].get(k) != v for k, v in schedule.items()):
                raise ValueError("OmniFold and audit must share scheduler settings")
        expected_id = "h4clfcos1" if path == SCHEDULER_CONFIG else "h4clfd1"
        if path == OLD_CHECKPOINT_CONFIG:
            expected_id = "h4clfold1"
        if config["logger"]["wandb"]["id"] != expected_id:
            raise ValueError("Diagnostic replay must use its own run ID")
    if audit != expected:
        raise ValueError("Unexpected classifier change outside LR, scheduler and diagnostics")
    if (audit["steps"] is not None or audit["min_steps"] != 1000
            or audit["validation_patience_epochs"] != 10):
        raise ValueError("Source audit stopping budget changed")
    experiment = config["experiment"]
    if not experiment["classifier_only"] or experiment["classifier_fit_count"] != 1:
        raise ValueError("Requires exactly one classifier-only fit")
    if any(experiment[k] != 0 for k in ("rounds", "policy_updates_per_round", "reward_fit_count")):
        raise ValueError("Policy updates and reward fitting must remain disabled")
    if adaptive["baseline_probe_on_start"] or adaptive["recalibration"]["bootstrap_on_start"]:
        raise ValueError("No bootstrap or extra baseline audit allowed")
    if config["dgpo"]["checkpoint_load_mode"] != "weights_only":
        raise ValueError("Requires weights-only policy loading")
    source = config["options"]["Training"]["model_checkpoint_load_path"]
    old_checkpoint = path == OLD_CHECKPOINT_CONFIG
    expected_source = (parent if old_checkpoint else read_overlay_yaml(CONFIG))["options"]["Training"]["model_checkpoint_load_path"]
    if source != config["nersc"]["reproducibility"]["source_checkpoint"] or source != expected_source:
        raise ValueError("Requires the pinned old-DGPO checkpoint" if old_checkpoint else "Requires the pinned step-50 policy checkpoint")
    expected_step = 1110 if old_checkpoint else 50
    expected_run = 'ytchou97-university-of-washington/nu2flow-RL/' + ('c4a91e07' if old_checkpoint else 'h4lbnr01')
    if (experiment['source_policy_step'] != expected_step
            or config['nersc']['reproducibility']['source_dgpo_global_step'] != expected_step
            or experiment['source_wandb_run'] != expected_run
            or config['nersc']['reproducibility']['source_wandb_run'] != expected_run):
        raise ValueError('Source checkpoint provenance mismatch')
    platform = config["platform"]
    if platform != parent["platform"]:
        raise ValueError("Keep source data and distributed loader geometry unchanged")
    if platform["number_of_workers"] != 16 or platform["resources_per_worker"]["GPU"] != 1:
        raise ValueError("Requires 16 one-GPU workers")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Enable measurement-only probes and bounded snapshots in a separate W&B run")
    parser.add_argument("--scheduler", action="store_true", help="Enable EveNet-style warmup/cosine with diagnostics in a separate run")
    parser.add_argument("--old-checkpoint", action="store_true", help="With --scheduler, test the original old-DGPO step-1110 checkpoint")
    parser.add_argument("--ray-dir", type=Path)
    args = parser.parse_args()
    if args.old_checkpoint and not args.scheduler:
        parser.error('--old-checkpoint requires --scheduler')
    path = SCHEDULER_CONFIG if args.scheduler else (DIAGNOSTIC_CONFIG if args.diagnostics else CONFIG)
    if args.old_checkpoint:
        path = OLD_CHECKPOINT_CONFIG
    config = validated_config(path)
    print(f"Contract OK: fixed source policy step {config['experiment']['source_policy_step']}; one cold classifier; 16 GPUs; no policy updates.")
    if args.validate_only:
        return 0
    checkpoint = Path(config["options"]["Training"]["model_checkpoint_load_path"])
    if not checkpoint.is_file():
        parser.error(f"Source checkpoint not found: {checkpoint}")
    return subprocess.run([
        sys.executable, str(ROOT / "scripts/train_neutrino_backend.py"),
        "--backend", "dgpo-evenet", "--base-config", str(ROOT / "config/train_diffusion_nersc.yaml"),
        "--overlay-config", str(path), "--", "--ray-dir",
        str(args.ray_dir or config["nersc"]["ray"]["results_dir"]),
    ], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
