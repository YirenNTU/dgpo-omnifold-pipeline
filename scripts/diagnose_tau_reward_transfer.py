#!/usr/bin/env python3
"""User-launched, isolated step-1920 tau reward transfer diagnostic.

The full saved actor, optimizer and schedule are restored. One new best-val
tau head is fitted at startup and the velocity reference is recentered once;
both then remain fixed for 50 native updates. --prepare-only validates and pins
the input files without launching the trainer or connecting to Ray.
"""
from __future__ import annotations

import argparse
import copy
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


REPO = Path(__file__).resolve().parents[1]
EVENET = REPO / "evenet_dgpo"
SOURCE_STEP = 1920
RELATIVE_STEPS = [0, 1, 5, 10, 20, 35, 50]
TRAIN_DIRECTORY = "/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train"
VALIDATION_DIRECTORY = "/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val"
POLICY_CONTRACT = "saved_tau_denominator_label0_v1"
WAND_NAME = "Can refit reward be retained? | tau attention | frozen 50 updates | step 1920"


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping: {name}")
    return value


def read_mapping(path):
    return _mapping(yaml.safe_load(Path(path).read_text()), str(path))


def validate_settings(settings):
    if settings.get("source_step") != SOURCE_STEP:
        raise ValueError("This diagnostic requires the pinned step1920 source")
    if settings.get("workers") != 16:
        raise ValueError("This real-case diagnostic requires sixteen GPUs")
    if settings.get("relative_steps") != RELATIVE_STEPS:
        raise ValueError("Preserve the predeclared 0/1/5/10/20/35/50 endpoints")
    if settings.get("audit_relative_steps") != [0, 50]:
        raise ValueError("Fresh diagnostic audits belong at baseline and +50 only")
    if type(settings.get("bootstrap_replicates")) is not int or settings["bootstrap_replicates"] < 2:
        raise ValueError("bootstrap_replicates must be an integer >= 2")
    if type(settings.get("bootstrap_seed")) is not int or settings["bootstrap_seed"] < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    tolerance = settings.get("reconstruction_tolerance")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("reconstruction_tolerance must be finite and positive")
    if settings.get("wandb_name") != WAND_NAME:
        raise ValueError("Use the predeclared readable diagnostic W&B name")
    for key in ("checkpoint", "source_runtime", "output_root"):
        if not isinstance(settings.get(key), str) or not Path(settings[key]).is_absolute():
            raise ValueError(f"{key} must be an explicit absolute path")


def source_metadata(checkpoint):
    """Inspect the actual source, not filenames or an inherited config clock."""
    checkpoint = _mapping(checkpoint, "checkpoint")
    for key in ("state_dict", "dgpo_optimizer_state_dict", "dgpo_ref_state_dict", "dgpo_omnifold_reward_stack"):
        if not _mapping(checkpoint.get(key), key):
            raise ValueError("Missing populated full-resume state: " + key)
    expected = {"global_step": SOURCE_STEP, "epoch": 191, "dgpo_next_epoch": 192, "dgpo_epoch_step": 0}
    for key, value in expected.items():
        if type(checkpoint.get(key)) is not int or checkpoint[key] != value:
            raise ValueError(f"Expected {key}={value} in the actual source checkpoint")
    stack = checkpoint["dgpo_omnifold_reward_stack"]
    if stack.get("kind") != "conditional_tau_bound30" or stack.get("schema_version") != 1:
        raise ValueError("Expected the conditional-tau bound30 reward, not H4")
    if stack.get("policy_conditioning_contract") != POLICY_CONTRACT:
        raise ValueError("The saved tau label0 denominator contract must be preserved")
    head = _mapping(stack.get("head"), "saved tau head")
    expected_head = {"head_kind": "film", "head_depth": 3, "condition_width": 256,
                     "condition_hidden": 256, "relative_dim": 6, "ratio_bound": 30,
                     "cross_attention": True}
    for key, value in expected_head.items():
        if head.get(key) != value or (key == "cross_attention" and head.get(key) is not True):
            raise ValueError(f"Expected the saved attention tau head {key}={value}")
    if not _mapping(head.get("state_dict"), "saved head state"):
        raise ValueError("Saved classifier weights are empty")
    if head.get("attention_heads", 4) != 4:
        raise ValueError("Preserve the saved four-head classifier attention")
    if head.get("explicit_input") or head.get("ratio_objective", "bce") != "bce":
        raise ValueError("Preserve the paired-BCE tau head and its inputs")
    for key in ("condition_mean", "condition_scale", "normalization_file", "source_checkpoint"):
        if key not in stack:
            raise ValueError("Missing pinned classifier preprocessing: " + key)
    for key in ("round_id", "denominator_step", "last_refit_epoch"):
        if type(stack.get(key)) is not int or stack[key] < 0:
            raise ValueError("Missing/invalid saved reward clock: " + key)
    if stack["denominator_step"] > SOURCE_STEP or stack["last_refit_epoch"] >= expected["dgpo_next_epoch"]:
        raise ValueError("Saved reward clocks are ahead of the actor")
    token_prefix = "TruthGeneration.visible_conditioning.token_readout."
    for key in ("state_dict", "dgpo_ref_state_dict"):
        if any(name.removeprefix("model.").startswith(token_prefix) for name in checkpoint[key]):
            raise ValueError("Expected the original diffusion without extra token-readout attention")
    opt = checkpoint["dgpo_optimizer_state_dict"]
    adam = _mapping(opt.get("optimizer"), "saved AdamW state")
    states = _mapping(adam.get("state"), "saved AdamW moments")
    if not states or not any(isinstance(value, dict) and "exp_avg" in value and "exp_avg_sq" in value
                             for value in states.values()):
        raise ValueError("Expected populated AdamW first and second moments")
    scheduler = _mapping(opt.get("scheduler"), "saved scheduler")
    schedule = _mapping(opt.get("lr_schedule"), "saved LR schedule")
    if scheduler.get("last_epoch") != SOURCE_STEP or schedule.get("kind") != "cosine":
        raise ValueError("Expected the inherited step1920 cosine scheduler")
    groups = adam.get("param_groups")
    base_lrs = scheduler.get("base_lrs")
    if not isinstance(groups, list) or not groups or not isinstance(base_lrs, list) or len(groups) != len(base_lrs):
        raise ValueError("Saved optimizer/scheduler groups do not match")
    group_names = [group.get("group_name") for group in groups]
    if any(not isinstance(name, str) for name in group_names) or len(set(group_names)) != len(group_names):
        raise ValueError("Preserve named optimizer groups")
    for group, base in zip(groups, base_lrs, strict=True):
        if not group.get("params") or any(not math.isfinite(float(value)) or float(value) <= 0
                                          for value in (group.get("lr", float("nan")), base)):
            raise ValueError("Saved optimizer group has empty parameters or invalid LR")
    return {**expected, "inherited_reward_round": stack["round_id"],
            "inherited_denominator_step": stack["denominator_step"],
            "inherited_last_refit_epoch": stack["last_refit_epoch"],
            "head_cross_attention": True, "head_depth": 3, "ratio_bound": 30,
            "policy_conditioning_contract": POLICY_CONTRACT,
            "normalization_file": str(stack["normalization_file"]),
            "classifier_trunk_source_checkpoint": str(stack["source_checkpoint"]),
            "optimizer_groups": [{**{key: group.get(key) for key in ("group_name", "lr", "weight_decay", "betas", "eps")},
                                  "base_lr": float(base)} for group, base in zip(groups, base_lrs, strict=True)],
            "scheduler_last_epoch": scheduler["last_epoch"], "lr_schedule": copy.deepcopy(schedule),
            "rng_data_iterator_restored": False,
            "scope": "Counterfactual continuation; the original checkpoint does not save RNG/data iterator state"}


def validate_source_runtime(cfg):
    """Reject a different experiment instead of silently patching its objective."""
    if "extends_overlay" in cfg:
        raise ValueError("source_runtime must be the complete resolved production runtime")
    if cfg.get("reward_config", {}).get("type") != "conditional_tau":
        raise ValueError("Expected the conditional-tau backend")
    platform, dg, training = cfg["platform"], cfg["dgpo"], cfg["options"]["Training"]
    if platform.get("number_of_workers") != 16 or platform.get("resources_per_worker", {}).get("GPU") != 1 or not platform.get("use_gpu"):
        raise ValueError("Preserve sixteen workers with one GPU each")
    for key, expected in (("data_parquet_dir", TRAIN_DIRECTORY), ("data_parquet_val_dir", VALIDATION_DIRECTORY)):
        if str(Path(platform[key])) != expected:
            raise ValueError("Use the exact pinned filtered population: " + key)
    trust = dg.get("reference_trust", {})
    if trust.get("enabled") is not True or trust.get("objective") != "velocity_mse" or trust.get("coefficient") != 1:
        raise ValueError("Preserve enabled coefficient-1 velocity-MSE reference")
    if dg.get("endpoint_kl", {}).get("enabled", False) or dg.get("beta_kl", 0) != 0:
        raise ValueError("This diagnostic must not add an endpoint or original KL term")
    if dg.get("adaptive_omnifold", {}).get("enabled", False):
        raise ValueError("Adaptive H4 must remain disabled")
    if dg.get("K") != 8 or dg.get("num_ddim_steps") != 20 or dg.get("num_train_timesteps") != 8:
        raise ValueError("Preserve native K8/DDIM20/eight accumulated timesteps")
    if dg.get("policy_eval_t_min") != 0 or dg.get("policy_eval_t_max") != 0.7:
        raise ValueError("Preserve the native diffusion time distribution")
    if dg.get("advantage_estimator") != "leave_one_out_unscaled" or dg.get("beta") != 1:
        raise ValueError("Preserve the native unscaled LOO DGPO objective")
    if dg.get("steps_per_epoch") != 10:
        raise ValueError("Preserve the ten-update logical epoch clock")
    if training.get("epochs") != 1500 or training.get("total_epochs") != 1500:
        raise ValueError("Do not reset the original 1500-epoch training horizon")
    schedule = dg.get("lr_schedule", {})
    if schedule.get("type") != "cosine" or schedule.get("total_epochs") != 1500 or schedule.get("resume_use_config", False):
        raise ValueError("Preserve the inherited 1500-epoch cosine schedule")
    if any(training.get("EMA", {}).get(key, False) for key in
           ("enable", "replace_model_after_load", "replace_model_at_end", "use_ema_during_training_eval", "use_for_generation")):
        raise ValueError("Use raw actor weights, never EMA")
    visible = cfg.get("network", {}).get("VisibleConditioning", {})
    if visible.get("diffusion_token_readout", {}).get("enabled", False) or dg.get("add_diffusion_attention", False):
        raise ValueError("Do not change the original diffusion architecture")
    if visible.get("diffusion_fourier_enabled", True) or cfg.get("network", {}).get("Body", {}).get("PET", {}).get("visible_angular_fourier", {}).get("fourier_enabled", True):
        raise ValueError("This source has the no-policy-Fourier architecture")
    tau = dg["tau_ratio"]
    if tau.get("policy_conditioning_contract") != POLICY_CONTRACT or tau.get("workers") != 16:
        raise ValueError("Preserve the sixteen-GPU saved tau label0 contract")
    if tau.get("fit", {}).get("batch_size") != 1024:
        raise ValueError("Preserve 1024 paired classifier events per GPU")
    if tau.get("fit", {}).get("epochs") != 250 or tau.get("fit", {}).get("min_steps") != 1000:
        raise ValueError("Preserve adequate best-validation classifier fitting")
    if not cfg.get("logger", {}).get("wandb", {}).get("project") and not cfg.get("wandb", {}).get("project"):
        raise ValueError("Live W&B diagnostics require the inherited project settings")


def configure(cfg, settings, metadata, *, pinned, output):
    validate_settings(settings)
    validate_source_runtime(cfg)
    cfg = copy.deepcopy(cfg)
    dg, tau = cfg["dgpo"], cfg["dgpo"]["tau_ratio"]
    dg.update(checkpoint_load_mode="resume", auto_resume_from_last=False,
              auto_resume_fallback_checkpoint_path=None, auto_resume_best_source_checkpoint_dir=None,
              unbounded_training=False, checkpoint_transfer={"enabled": False})
    # These unrelated old diagnostics are not the tau transfer protocol.
    for name in ("gradient_conflict", "tarp"):
        if name in dg:
            dg[name]["enabled"] = False
    tau.update(classifier_only=False, classifier_attention_ablation=False,
               production_cross_attention=True, reward_cij_probe=False,
               output=str(output / "tau_diagnostics"), output_root=str(output))
    tau["transfer_probe"] = {
        "enabled": True, "source_step": SOURCE_STEP,
        "gradient_trace_enabled": True,
        "expected_source_reward_round": metadata["inherited_reward_round"],
        "relative_steps": list(settings["relative_steps"]),
        "bootstrap_replicates": settings["bootstrap_replicates"],
        "bootstrap_seed": settings["bootstrap_seed"],
        "reconstruction_tolerance": settings["reconstruction_tolerance"],
        "audit_relative_steps": list(settings["audit_relative_steps"]),
    }
    dg["gradient_transfer_trace"] = {
        "enabled": True,
        "update_end_steps": [SOURCE_STEP + step for step in RELATIVE_STEPS if step],
        "counterfactual_trust_coefficients": [1.0],
    }
    cfg["options"]["Training"].update(model_checkpoint_load_path=str(pinned), pretrain_model_load_path=None,
                                        model_checkpoint_save_path=str(output / "checkpoints"))
    cfg.setdefault("compat", {}).update(backend="dgpo-evenet", repo_root=str(REPO))
    cfg.setdefault("rl", {})["enabled"] = True
    cfg["logger"]["local"]["save_dir"] = str(output / "logs")
    wb = cfg["logger"]["wandb"] if cfg["logger"].get("wandb", {}).get("project") else cfg["wandb"]
    wb.update(id=None, resume="never", fresh_run=True, run_name=settings["wandb_name"],
              name=settings["wandb_name"], group=settings.get("wandb_group", "Tau reward transfer"))
    inherited_tags = [tag for tag in wb.get("tags", []) if not str(tag).lower().startswith("refit")]
    wb["tags"] = list(dict.fromkeys([*inherited_tags, "TauRewardTransfer", "Step1920", "FrozenReward50", "RawWeights", "SixteenGPU"]))
    # Prevent an unused name/ID alias from suggesting a different run.
    if "wandb" in cfg and cfg["wandb"] is not wb:
        cfg["wandb"].update(id=None, resume="never", fresh_run=True,
                            run_name=settings["wandb_name"], name=settings["wandb_name"], group=wb["group"])
    cfg["nersc"]["ray"]["results_dir"] = str(output / "ray_results")
    cfg["nersc"]["submit"] = False
    cfg["nersc"]["execution"]["command"] = "shifter python3 -u scripts/diagnose_tau_reward_transfer.py config/tau_reward_transfer_1920.yaml"
    cfg["nersc"]["reproducibility"] = {"source_checkpoint": str(pinned),
        "source_policy_step": SOURCE_STEP,
        "note": "Full-state step1920 continuation; one fresh head/reference install then 50 frozen-reward native updates; not a bitwise replay of the original RNG/data iterator."}
    experiment = cfg.setdefault("experiment", {})
    experiment.pop("actor_updates", None)
    experiment.update(classifier_only=False, tau_classifier_only=False,
        protocol="tau-frozen-reward-transfer", source_policy_step=SOURCE_STEP,
        source_metadata=copy.deepcopy(metadata),
        source_checkpoint_provenance="6d8bb6c2 resume/source_checkpoint; originating attention trajectory 8e45a217",
        first_head="one fresh best-validation attention refit at step1920; recenter reference once",
        purpose="Measure reward uptake and retention, with classifier and reference fixed over 50 native updates",
        primary="paired held-out all-K fixed-head mean reward change at +50 with event-bootstrap 95% interval",
        secondary="raw Cij and per-component guardrails; fresh baseline/+50 audit BCE/AUC; actual-gradient/AdamW displacement diagnostics",
        causal_scope="Head replacement and reference recentering are joint; this experiment cannot attribute between them",
        continuation_scope="Counterfactual continuation, not bitwise replay; original RNG/data iterator were not checkpointed")
    return cfg


def command_for_runtime(runtime, output):
    return [sys.executable, "-u", str(EVENET / "RL/DGPO_neutrino/dgpo_trainer.py"), str(runtime),
            "--max-steps", str(SOURCE_STEP + RELATIVE_STEPS[-1]), "--ray-dir", str(output / "ray_results")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Validate and pin files without starting the trainer/Ray")
    args = parser.parse_args()
    settings = read_mapping(args.config)
    validate_settings(settings)
    source = Path(settings["checkpoint"]).expanduser().resolve(strict=True)
    source_runtime = Path(settings["source_runtime"]).expanduser().resolve(strict=True)
    cfg = read_mapping(source_runtime)
    validate_source_runtime(cfg)
    sys.path[:0] = [str(REPO), str(EVENET)]
    from evenet.dataset.filtered_data import validate_filtered_dataset
    manifests = {}
    for key, rows in (("data_parquet_dir", 416701), ("data_parquet_val_dir", 119002)):
        manifests[key] = validate_filtered_dataset(cfg["platform"][key])
        if manifests[key]["rows"] != rows:
            raise ValueError("Filtered dataset population changed: " + key)
    root = Path(settings["output_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("probe-%Y%m%dT%H%M%S-"), dir=root.resolve()))
    pinned = output / "source.ckpt"
    try:
        os.link(source, pinned)
        pin_method = "hard_link"
    except OSError:
        shutil.copy2(source, pinned)
        pin_method = "copy"
    shutil.copy2(source_runtime, output / "source_runtime.yaml")
    import torch
    checkpoint = torch.load(pinned, map_location="cpu", weights_only=False, mmap=True)
    metadata = source_metadata(checkpoint)
    del checkpoint
    metadata.update(source_checkpoint=str(source), pinned_checkpoint=str(pinned),
                    source_runtime=str(source_runtime), pinned_runtime=str(output / "source_runtime.yaml"),
                    pin_method=pin_method, filtered_manifests=manifests)
    configured = configure(cfg, settings, metadata, pinned=pinned, output=output)
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump(configured, sort_keys=False))
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = command_for_runtime(runtime, output)
    print(json.dumps({"output": str(output), "source": metadata, "runtime": str(runtime),
                      "workers": 16, "heldout_conditions": 119002, "candidates_per_condition": 8,
                      "relative_updates": RELATIVE_STEPS, "absolute_stop_step": 1970,
                      "startup": "fresh best-val head plus reference recentering once",
                      "periodic_refits": False, "fresh_diagnostic_audits": [0, 50],
                      "command": command, "prepare_only": args.prepare_only}, indent=2), flush=True)
    if args.prepare_only:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "scripts"), str(EVENET), env.get("PYTHONPATH", "")])
    subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
