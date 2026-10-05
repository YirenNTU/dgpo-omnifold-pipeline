#!/usr/bin/env python3
"""Pin the latest DGPO state and launch the native 0/1/5/20-update probe.

Run this inside the user's existing 16-GPU Shifter/Ray allocation. No allocation
or job submission is performed. --dry-run prepares files without starting Ray.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

from train_neutrino_backend import (
    REPO_ROOT, EVENET_DGPO_ROOT, absolutize_default_paths, command_for_backend,
    deep_update, read_overlay_yaml, read_yaml,
)

DEFAULT_CONFIG = REPO_ROOT / "config/dgpo_checkpoint_transfer.yaml"
DEFAULT_OUTPUT = Path("/pscratch/sd/y/yiren/Ztautau/dgpo_checkpoint_transfer")


def validate_probe_runtime(cfg: dict) -> None:
    """Use the actual trainer parsers before connecting to Ray (also in dry-run).

    Overlay value assertions alone miss dependencies inherited from parents.
    Do not relax the production validator just because audits are bypassed.
    """
    if str(EVENET_DGPO_ROOT) not in sys.path:
        sys.path.insert(0, str(EVENET_DGPO_ROOT))
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
    from RL.DGPO_neutrino.gradient_transfer import resolve_gradient_transfer_trace_config

    adaptive = resolve_adaptive_config(cfg["dgpo"], classifier_only=False)
    trace = resolve_gradient_transfer_trace_config(cfg["dgpo"].get("gradient_transfer_trace"))
    if not adaptive.enabled or adaptive.bootstrap_on_start or adaptive.refit_once_on_resume:
        raise ValueError("Checkpoint-transfer must restore the paired reference without bootstrap/refit")
    if not trace.enabled or not cfg["dgpo"].get("checkpoint_transfer", {}).get("enabled"):
        raise ValueError("Checkpoint-transfer requires paired measurements and native gradient tracing")


def source_metadata(checkpoint: dict) -> dict:
    required = ("state_dict", "dgpo_optimizer_state_dict", "dgpo_ref_state_dict",
                "dgpo_round_ref_state_dict", "dgpo_omnifold_reward_stack",
                "dgpo_adaptive_omnifold_state")
    missing = [key for key in required if not checkpoint.get(key)]
    if missing:
        raise ValueError(f"Not a full-state DGPO checkpoint: missing {missing}")
    for key in ("global_step", "epoch", "dgpo_next_epoch", "dgpo_reward_round_id"):
        if type(checkpoint.get(key)) is not int or checkpoint[key] < 0:
            raise ValueError(f"Missing/invalid saved counter: {key}")
    opt = checkpoint["dgpo_optimizer_state_dict"]
    if not opt.get("optimizer", {}).get("state") or not opt.get("scheduler") or not opt.get("lr_schedule"):
        raise ValueError("Expected populated AdamW moments and cosine scheduler")
    adaptive = checkpoint["dgpo_adaptive_omnifold_state"]
    gap, threshold = float(adaptive.get("baseline_auc_gap", float("nan"))), float(adaptive.get("trigger_threshold", float("nan")))
    if not (math.isfinite(gap) and gap >= 0 and math.isfinite(threshold) and threshold > 0):
        raise ValueError("Installed reward is not calibrated; this experiment never bootstraps")
    for key in ("raw_monitor_baseline_pending", "raw_global_stop_requested", "trust_rejection_stop_requested"):
        if adaptive.get(key):
            raise ValueError(f"Saved state requires controller action ({key}); do not silently reset it")
    return {**{key: checkpoint[key] for key in ("global_step", "epoch", "dgpo_next_epoch", "dgpo_reward_round_id")},
            "dgpo_epoch_step": checkpoint.get("dgpo_epoch_step", 0),
            "optimizer_groups": [{k: group.get(k) for k in ("group_name", "lr", "weight_decay", "betas", "eps")}
                                 for group in opt["optimizer"]["param_groups"]],
            "scheduler_last_epoch": opt["scheduler"]["last_epoch"],
            "lr_schedule": opt["lr_schedule"],
            "rng_data_iterator_restored": False}


def build_probe_config(base: Path, overlay: Path, *, pinned: Path, output: Path,
                       metadata: dict, events_per_rank: int | None = None) -> dict:
    cfg = deep_update(read_yaml(base), read_overlay_yaml(overlay))
    cfg = absolutize_default_paths(cfg, base.parent)
    cfg.setdefault("compat", {}).update(backend="dgpo-evenet", repo_root=str(REPO_ROOT))
    cfg.setdefault("rl", {})["enabled"] = True
    platform, dg = cfg["platform"], cfg["dgpo"]
    if platform["number_of_workers"] != 16 or platform["resources_per_worker"]["GPU"] != 1:
        raise ValueError("Protocol requires sixteen workers with one GPU each")
    if Path(platform["data_parquet_dir"]).resolve() == Path(platform["data_parquet_val_dir"]).resolve():
        raise ValueError("Policy update and validation datasets must differ")
    if dg["reference_trust"]["objective"] != "velocity_mse" or dg["reference_trust"]["coefficient"] != 1:
        raise ValueError("Protocol preserves coefficient=1 velocity-MSE")
    training = cfg["options"]["Training"]
    if training["EMA"]["enable"]:
        raise ValueError("Raw policy required, never EMA")
    training.update(model_checkpoint_load_path=str(pinned), pretrain_model_load_path=None,
                    model_checkpoint_save_path=str(output / "checkpoints"))
    # A relative step cap is independent of the absolute training/scheduler clock.
    # Do not shorten/reset cosine or change production steps_per_epoch.
    needed_epoch = metadata["dgpo_next_epoch"] + 4
    if needed_epoch >= int(training["epochs"]):
        training["epochs"] = needed_epoch
    dg["unbounded_training"] = False
    dg["auto_resume_from_last"] = False
    probe = dg["checkpoint_transfer"]
    if events_per_rank is not None:
        probe["events_per_rank"] = events_per_rank
    if int(probe["events_per_rank"]) < 2:
        raise ValueError("events-per-rank must be >=2")
    probe.update(source_step=metadata["global_step"], source_reward_round=metadata["dgpo_reward_round_id"],
                 world_size=16, output_directory=str(output / "measurements"))
    dg["gradient_transfer_trace"]["update_end_steps"] = [metadata["global_step"] + x for x in (1, 5, 20)]
    cfg["logger"]["local"].update(name="dgpo-checkpoint-transfer", save_dir=str(output / "logs"))
    cfg["nersc"]["ray"]["results_dir"] = str(output / "ray_results")
    cfg["nersc"]["reproducibility"] = {
        "source_checkpoint": str(pinned), "note": "Native saved-state continuation with fixed reward/reference; raw weights; paired all-sample evaluation. No audits/refits. RNG/data iterator not saved in original checkpoint."}
    cfg["nersc"]["execution"] = {"command": "python3 -u scripts/diagnose_dgpo_checkpoint_transfer.py"}
    cfg["experiment"].update(source_policy_step=metadata["global_step"],
        retained_classifier_checkpoint=str(pinned), source_metadata=metadata,
        endpoint_selection="predeclared_relative_step20", cold_h4_audit_min_optimizer_updates=None,
        startup_mode="full_state_fixed_reward_continuation", residual_iterations_max=None)
    validate_probe_runtime(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the current production last.ckpt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--events-per-rank", type=int, help="Default 2048: 32768 held-out events on 16 GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Validate/pin checkpoint and write config only")
    args = parser.parse_args()
    import torch

    overlay = read_overlay_yaml(args.config)
    validate_probe_runtime(overlay)
    source = args.checkpoint or Path(overlay["options"]["Training"]["model_checkpoint_load_path"])
    source = source.expanduser().resolve(strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("probe-%Y%m%dT%H%M%S-"),
                                  dir=args.output_root.resolve()))
    pinned = output / "source.ckpt"
    # DGPO writes atomically via replace. A hard link pins its current inode
    # even if last.ckpt changes; cross-filesystem copies use the same source.
    try:
        os.link(source, pinned)
        pin_method = "hard_link"
    except OSError:
        shutil.copy2(source, pinned)
        pin_method = "copy"
    checkpoint = torch.load(pinned, map_location="cpu", weights_only=False, mmap=True)
    metadata = source_metadata(checkpoint)
    del checkpoint
    metadata.update(source_checkpoint=str(source), pinned_checkpoint=str(pinned), pin_method=pin_method)
    base = REPO_ROOT / "config/train_diffusion_nersc.yaml"
    cfg = build_probe_config(base, args.config, pinned=pinned, output=output,
                             metadata=metadata, events_per_rank=args.events_per_rank)
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = command_for_backend("dgpo-evenet", runtime) + ["--max-steps", str(metadata["global_step"] + 20),
                                                           "--ray-dir", str(output / "ray_results")]
    print(json.dumps({"source": metadata, "output": str(output),
        "heldout_events": 16 * cfg["dgpo"]["checkpoint_transfer"]["events_per_rank"],
        "relative_updates": [0, 1, 5, 20], "raw_policy": True, "refits": False, "fresh_audits": False,
        "command": command}, indent=2), flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(EVENET_DGPO_ROOT), env.get("PYTHONPATH", "")])
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
