#!/usr/bin/env python3
"""Launch the endpoint KL experiment with an explicit on/off switch."""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
import yaml

from train_neutrino_backend import read_overlay_yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
from RL.DGPO_neutrino.endpoint_kl import EndpointKLConfig, validate_endpoint_protocol


def make_config(enabled=True, coefficient=1.0, run_id=None, output=None,
                resume=False, total_steps=None):
    config = read_overlay_yaml(ROOT / "config/dgpo_endpoint_kl_10pct.yaml")
    config["dgpo"]["endpoint_kl"].update(enabled=enabled, coefficient=coefficient)
    suffix = "on" if enabled else "off"
    output = str(output or f"/pscratch/sd/y/yiren/Ztautau/h4_endpoint_kl_{suffix}")
    config["options"]["Training"]["model_checkpoint_save_path"] = output + "/checkpoints"
    local = config["logger"]["local"]
    local.update(name=f"h4_endpoint_kl_{suffix}", version=f"h4_endpoint_kl_{suffix}_v1", save_dir=output + "/logs")
    wandb = config["logger"]["wandb"]
    wandb.update(id=run_id or f"h4epkl1{suffix}",
                 run_name=f"Does endpoint KL improve reward transfer? | KL {suffix} | raw step 1110")
    config["nersc"]["ray"]["results_dir"] = output + "/ray_results"
    config["experiment"]["retained_classifier_checkpoint"] = output + "/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt"
    config["experiment"]["intervention"] = "online_endpoint_kl" if enabled else "endpoint_kl_disabled"
    config["nersc"]["reproducibility"]["note"] = (
        "Raw step1110, 10pct training data, cleaned validation, frozen one-round H4; "
        f"endpoint KL {suffix}. Audit and policy validation every 10 DGPO epochs (100 updates). "
        "Online current/reference critic and full DDIM pathwise penalty only when enabled."
    )
    if total_steps is not None:
        epoch_steps = config["dgpo"]["steps_per_epoch"]
        if type(total_steps) is not int or total_steps <= 0 or total_steps % epoch_steps:
            raise ValueError(f"total_steps must be a positive multiple of {epoch_steps}")
        config["options"]["Training"].update(
            epochs=total_steps // epoch_steps, total_epochs=total_steps // epoch_steps)
        config["experiment"].update(
            policy_updates_per_round=total_steps, reward_lifetime_updates=total_steps,
            stop_rule=f"Stop at total policy step {total_steps}; no reward refit or rollback.",
            cold_h4_audit_policy_steps=list(range(100, total_steps + 1, 100)),
            gradient_direction_policy_steps=list(range(100, total_steps + 1, 100)),
            endpoint_selection=f"fixed_step_{total_steps}",
            primary_decision_rule="Track fixed-classifier AUC and separate fresh audits every 100 updates through the declared endpoint; no test-driven checkpoint selection.")
    if resume:
        if total_steps is None:
            config["dgpo"]["unbounded_training"] = True
            config["experiment"].update(
                policy_updates_per_round=None, reward_lifetime_updates=None,
                stop_rule="No step or epoch limit; stop manually or at job walltime.",
                cold_h4_audit_policy_steps=None, gradient_direction_policy_steps=None,
                endpoint_selection="unbounded_continuation",
                primary_decision_rule="Track fixed-classifier AUC and separate fresh audits every 100 updates; no fixed terminal endpoint.")
        config["dgpo"].update(checkpoint_load_mode="resume", auto_resume_from_last=True)
        config["options"]["Training"]["model_checkpoint_load_path"] = output + "/checkpoints/last.ckpt"
        config["dgpo"]["adaptive_omnifold"]["recalibration"].update(
            bootstrap_on_start=False, refit_once_on_resume=False)
        wandb.update(resume="must", fresh_run=False)
        config["nersc"]["reproducibility"]["note"] += " Full-state continuation from this run's last.ckpt."
    parsed = EndpointKLConfig.parse(config["dgpo"]["endpoint_kl"])
    validate_endpoint_protocol(parsed, config["dgpo"])
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-kl", choices=("on", "off"), default="on")
    p.add_argument("--coefficient", type=float, default=1.0)
    p.add_argument("--run-id")
    p.add_argument("--output", type=Path)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Restore this output directory's last.ckpt, including optimizer/reference/critics")
    p.add_argument("--total-steps", type=int, help="Optional absolute endpoint; --resume without this runs without a step/epoch limit")
    args = p.parse_args()
    config = make_config(args.endpoint_kl == "on", args.coefficient, args.run_id, args.output,
                         resume=args.resume, total_steps=args.total_steps)
    print(yaml.safe_dump({"endpoint_kl": config["dgpo"]["endpoint_kl"],
                         "omnifold_classifier_fit": config["dgpo"]["adaptive_omnifold"]["recalibration"]["fit"],
                         "audit_classifier_fit": config["dgpo"]["adaptive_omnifold"]["audit_fit"],
                         "wandb": config["logger"]["wandb"],
                         "checkpoints": config["options"]["Training"]["model_checkpoint_save_path"]}, sort_keys=False))
    if args.validate_only:
        return 0
    if args.resume:
        import torch
        path = Path(config["options"]["Training"]["model_checkpoint_load_path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        required = {"global_step", "dgpo_next_epoch", "dgpo_optimizer_state_dict",
                    "dgpo_round_ref_state_dict", "dgpo_adaptive_omnifold_state",
                    "dgpo_omnifold_reward_stack"}
        if args.endpoint_kl == "on":
            required.add("dgpo_endpoint_kl_state")
        missing = required - checkpoint.keys()
        if missing:
            p.error(f"Incomplete full-resume checkpoint: {sorted(missing)}")
        if args.total_steps is not None and int(checkpoint["global_step"]) >= args.total_steps:
            p.error("--total-steps must exceed the checkpoint's global_step")
        print(f"Full resume: {path}; step={checkpoint['global_step']}; target={args.total_steps}", flush=True)
        del checkpoint
    with tempfile.TemporaryDirectory(prefix="dgpo_endpoint_kl_") as tmp:
        overlay = Path(tmp) / "overlay.yaml"
        overlay.write_text(yaml.safe_dump(config, sort_keys=False))
        return subprocess.run([sys.executable, str(ROOT / "scripts/train_neutrino_backend.py"),
            "--backend", "dgpo-evenet", "--base-config", str(ROOT / "config/train_diffusion_nersc.yaml"),
            "--overlay-config", str(overlay), "--", "--ray-dir", config["nersc"]["ray"]["results_dir"]],
            cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
