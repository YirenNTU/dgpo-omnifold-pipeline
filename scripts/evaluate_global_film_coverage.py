#!/usr/bin/env python3
"""Evaluate a raw global-FiLM checkpoint against the saved low-noise coverage panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
REMOTE = Path("/pscratch/sd/y/yiren/Ztautau")
DEFAULT_CHECKPOINT = REMOTE / "diffusion_global_film_long_resume/checkpoints/last.ckpt"
DEFAULT_BASELINE = REMOTE / "diffusion_low_noise_lr_10pct_seed42/checkpoints/joint_coverage/epoch-0100.json"
PROTOCOL = dict(events=1024, K=32, seed=42017, batch_size=16,
                ddim_steps=20, x0_mode="legacy", bootstrap=500, every_n_epochs=25)
RUN_NAME = "Does conditioning recover joint coverage? | global FiLM | matched old finetuning panel"
GROUP = "Supervised conditioning coverage"
SCHEMA = "global-film-matched-coverage-v1"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def runtime_configuration():
    from train_neutrino_backend import read_yaml, read_overlay_yaml, deep_update, absolutize_default_paths
    cfg = deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                      read_overlay_yaml(ROOT / "config/train_diffusion_global_film_long.yaml"))
    cfg = absolutize_default_paths(cfg, ROOT / "config")
    training = cfg["options"]["Training"]
    # Construct only the model. Raw weights are loaded strictly by this evaluator.
    training.update(model_checkpoint_load_path=None, pretrain_model_load_path=None,
                    strict_conditioning_ablation_source=False, strict_no_fourier_source=False)
    training["EMA"] = dict(enable=False, use_for_generation=False,
                           replace_model_after_load=False, use_ema_during_training_eval=False)
    cfg.setdefault("rl", {})["enabled"] = False
    cfg.setdefault("compat", {}).update(backend="pure-evenet", repo_root=str(ROOT))
    cfg["logger"]["wandb"].update(id=None, project="nu2flow-RL", run_name=RUN_NAME, group=GROUP)
    cfg["experiment"] = dict(question="Does the global-FiLM checkpoint improve old-panel joint coverage?",
                             policy_updates=0, classifier_fits=0, weights="raw_state_dict_only")
    cfg["nersc"]["execution"] = dict(mode="evaluation_only_existing_ray_cluster",
                                    command="python3 -u scripts/evaluate_global_film_coverage.py")
    return cfg


def inspect_checkpoint(path):
    path = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("dgpo_checkpoint_version", 0) or any(k.startswith("dgpo_") for k in payload):
        raise ValueError("Use a supervised diffusion checkpoint, before subsequent DGPO updates")
    epoch, step = payload.get("epoch"), payload.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(step) is not int or step < 0:
        raise ValueError("Checkpoint must record its supervised epoch and global_step")
    state = payload.get("state_dict", {})
    if not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint must contain a tensor-only raw state_dict")
    keys = [k.removeprefix("model.") for k in state]
    if not any("TruthGeneration.visible_conditioning.modulations." in k for k in keys):
        raise ValueError("Checkpoint has no global visible-FiLM modulation weights")
    if not any("PET.angular_conditioning." in k for k in keys):
        raise ValueError("Checkpoint has no PET angular Fourier weights")
    if any("TruthGeneration.visible_conditioning.token_readout." in k for k in keys):
        raise ValueError("This comparison requires global FiLM without an additional token readout")
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
        raise ValueError("The historical comparison requires FP32 raw weights")
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite checkpoint tensor")
    metadata = dict(path=str(path), epoch=epoch, global_step=step, weights="raw_state_dict_only")
    # Drop optimizer, EMA and callback state. The driver writes this snapshot once;
    # all ranks subsequently read it, even if the source last.ckpt later changes.
    return metadata, dict(state_dict=state, epoch=epoch, global_step=step)


def validate_panel(panel, events):
    if panel["truth"].shape != (events, 4):
        raise ValueError("Historical panel target shape differs")
    for key in ("condition", "test_rows", "pool_rows"):
        if len(panel[key]) != events:
            raise ValueError(f"Historical panel is unaligned: {key}")
    for key in ("test_rows", "pool_rows"):
        ids = panel[key]
        if ids.ndim != 1 or ids.dtype != torch.int64 or len(ids.unique()) != events:
            raise ValueError(f"Duplicate/malformed panel identities: {key}")
    if not torch.isfinite(panel["truth"]).all() or not torch.isfinite(panel["condition"]).all():
        raise ValueError("Nonfinite historical panel")


def load_baseline(report_path):
    from diagnose_h4_ddim_coverage import analyze_arm
    from evenet.utilities.joint_coverage import numeric_metrics
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_text())
    cfg = report["config"]
    for key, value in PROTOCOL.items():
        if cfg.get(key) != value:
            raise ValueError(f"Historical protocol differs: {key}")
    if report["completed_epochs"] != 100:
        raise ValueError("Predeclared reference is the old finetuning completed-epoch-100 evaluation")
    panel = torch.load(cfg["panel_path"], map_location="cpu", weights_only=True)
    validate_panel(panel, cfg["events"])
    saved = torch.load(report_path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    generated = saved["generated"]
    if generated.shape != (cfg["events"], cfg["K"], 4) or generated.dtype != torch.float32:
        raise ValueError("Historical candidates must be aligned FP32 N x K x 4")
    if not torch.isfinite(generated).all():
        raise ValueError("Nonfinite historical candidates")
    result, _, angles = analyze_arm(panel, generated)
    if saved["angles"].shape != angles.shape or not np.allclose(
            saved["angles"].numpy(), angles, rtol=0, atol=1e-12):
        raise ValueError("Historical angles do not reconstruct from this panel and candidates")
    metrics, recorded = numeric_metrics(result), numeric_metrics(report["result"])
    if metrics.keys() != recorded.keys() or any(
            not np.isclose(v, recorded[k], rtol=1e-10, atol=1e-12) for k, v in metrics.items()):
        raise ValueError("Historical report no longer reproduces from its saved candidates/panel")
    return panel, saved, report, metrics


def decision(baseline_result, current_report):
    result = current_report["result"]
    row = next(r for r in current_report["paired"]
               if r["region"] == "joint" and r["threshold_radians"] == 1e-4)
    gap_improved = row["absolute_truth_gap_change_ci95"][1] < 0
    radius_change = (result["topology"]["joint_radius"]["w1_radians"]
                     - baseline_result["topology"]["joint_radius"]["w1_radians"])
    marginal_changes = {k: v["w1_radians"] - baseline_result["target_marginals"][k]["w1_radians"]
                        for k, v in result["target_marginals"].items()}
    invalid_increased = any(v > baseline_result["invalid_direction_inputs"].get(k, 0)
                            for k, v in result["invalid_direction_inputs"].items())
    return dict(primary="absolute truth-gap change at joint threshold 1e-4 rad",
                primary_gap_ci95_below_zero=gap_improved,
                joint_radius_w1_change=radius_change,
                target_marginal_w1_changes=marginal_changes,
                invalid_direction_counts_increased=invalid_increased,
                coverage_and_radius_improved=gap_improved and radius_change < 0,
                interpretation="Coverage/radius improvement supports this checkpoint on the reused panel; "
                "inspect marginal and invalid-direction tradeoffs. Otherwise the result is mixed or unresolved. "
                "Neither outcome isolates FiLM from depth or establishes fresh-H4 closure.")


def worker(cfg):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.joint_coverage import JointCoverageValidation, numeric_metrics
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from diagnose_h4_spike_coverage import load_raw_state
    from lightning.pytorch.loggers import WandbLogger

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    device = ray.train.torch.get_device()
    output = Path(cfg["output"])
    global_config.load_yaml(str(output / "runtime.yaml"))
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device)
    source = torch.load(output / "raw_policy.pt", map_location="cpu", weights_only=True)
    # Shared strict raw-state loader: exact keys, shapes, dtypes, and loaded values;
    # it never installs EMA or silently initializes missing conditioning weights.
    load_raw_state(policy, source, "pretrained10pct")
    del source
    policy.eval().requires_grad_(False)
    if int(getattr(policy, "invisible_input_dim", 2)) != 2:
        raise ValueError("Expected two tau-correction coordinates per slot")
    old = torch.load(output / "baseline_candidates.pt", map_location="cpu", weights_only=True)
    old_report = json.loads((output / "baseline_report.json").read_text())
    protocol = {**old_report["config"], "panel_path": str(output / "panel.pt")}
    callback = JointCoverageValidation(protocol, output)
    callback.baseline = old["angles"].clone()
    callback.baseline_metrics = numeric_metrics(old_report["result"])
    loggers, run = [], None
    if rank == 0 and cfg["wandb"]:
        import wandb
        run = wandb.init(project="nu2flow-RL", entity="ytchou97-university-of-washington",
                         id=cfg["run_id"], resume="never", name=RUN_NAME, group=GROUP,
                         tags=["Coverage", "GlobalFiLM", "RawWeights", "NoPolicyUpdate", "PairedNoise"], config=cfg)
        loggers = [WandbLogger(experiment=run)]
    # Only use the existing evaluation routine, not a Lightning fit loop.
    trainer = SimpleNamespace(global_rank=rank, world_size=world, global_step=0, loggers=loggers)
    module = SimpleNamespace(model=policy, device=device)
    completed = cfg["checkpoint"]["epoch"] + 1
    success = False
    try:
        callback.evaluate(trainer, module, completed)
        if rank == 0:
            current = json.loads((output / f"joint_coverage/epoch-{completed:04d}.json").read_text())
            summary = decision(old_report["result"], current)
            report = dict(schema=SCHEMA, complete=True, config=cfg,
                          baseline=old_report, current=current, decision=summary)
            write_json(output / "report.json", report)
            if run:
                run.summary.update({"evaluation/complete": True, "policy_updates": 0, "classifier_fits": 0,
                                    "decision/primary_gap_ci95_below_zero": summary["primary_gap_ci95_below_zero"],
                                    "decision/joint_radius_w1_change": summary["joint_radius_w1_change"],
                                    "checkpoint/epoch": cfg["checkpoint"]["epoch"],
                                    "checkpoint/global_step": cfg["checkpoint"]["global_step"]})
            (output / "COMPLETE").write_text(SCHEMA + "\n")
            print(json.dumps(summary, indent=2), flush=True)
        torch.distributed.barrier()
        success = True
    finally:
        if run:
            run.finish(exit_code=0 if success else 1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    p.add_argument("--output", type=Path, default=REMOTE / "global_film_matched_coverage")
    p.add_argument("--run-id", default="globcov1")
    p.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS") or "auto")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print resolved setup; no remote files, W&B, Ray or sampling")
    p.add_argument("--check-only", action="store_true", help="Validate saved source files on their host; no GPU/W&B/Ray")
    args = p.parse_args(argv)
    cfg = dict(schema=SCHEMA, checkpoint_path=str(args.checkpoint), baseline_report=str(args.baseline_report),
               output=str(args.output.resolve()), protocol=PROTOCOL, workers=16,
               run_id=args.run_id, run_name=RUN_NAME, group=GROUP, wandb=not args.no_wandb,
               policy_updates=0, classifier_fits=0, parallel_chains=1,
               initial_noise="existing callback event_noise: CPU generator seed 42017 + panel position",
               primary_endpoint="paired absolute truth-gap change; joint radius <1e-4 rad; whole-event bootstrap",
               exploratory=True, checkpoint_files_verified=False)
    runtime = runtime_configuration()
    if args.dry_run:
        print(yaml.safe_dump({"evaluation": cfg, "runtime": runtime}, sort_keys=False))
        return
    if int(os.environ.get("SLURM_PROCID", "0")) != 0:
        p.error("Launch once from the existing allocation, not once per GPU")
    output = Path(cfg["output"])
    if output.exists():
        p.error("Choose a fresh output directory; existing results are never overwritten")
    panel, old, old_report, _ = load_baseline(args.baseline_report)
    metadata, raw = inspect_checkpoint(args.checkpoint)
    cfg.update(checkpoint=metadata, checkpoint_files_verified=True)
    if args.check_only:
        print(json.dumps(cfg, indent=2))
        return
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    ray.init(address=args.ray_address, runtime_env={"env_vars": {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")])}})
    if ray.cluster_resources().get("GPU", 0) < 16:
        raise RuntimeError("Requires the user's existing 16-GPU Ray allocation; no local fallback")
    output.mkdir(parents=True, exist_ok=False)
    torch.save(raw, output / "raw_policy.pt")
    del raw
    torch.save(panel, output / "panel.pt")
    torch.save(old, output / "baseline_candidates.pt")
    write_json(output / "baseline_report.json", old_report)
    write_json(output / "manifest.json", cfg)
    (output / "runtime.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False))
    TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg,
                 scaling_config=ScalingConfig(num_workers=16, use_gpu=True),
                 run_config=RunConfig(name="global-film-coverage", storage_path=str(output / "ray_results"),
                                      failure_config=FailureConfig(max_failures=0))).fit()
    print("REPORT:", output / "report.json")


if __name__ == "__main__":
    main()
