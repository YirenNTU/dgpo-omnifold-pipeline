#!/usr/bin/env python3
"""User-launched 16-GPU calibration, then supervised diffusion in frozen coordinates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "evenet_dgpo" / "evenet"))
from train_neutrino_backend import build_runtime_config, command_for_backend


def validate_config(config):
    cfg = config.to_dict() if hasattr(config, "to_dict") else config
    platform, training = cfg["platform"], cfg["options"]["Training"]
    preconditioning = cfg["network"]["ConditionalPreconditioning"]
    if (int(platform["number_of_workers"]) != 16 or not platform.get("use_gpu", True)
            or platform["resources_per_worker"]["GPU"] != 1):
        raise ValueError("This real-case experiment requires 16 workers with one GPU each")
    if not preconditioning.get("enabled") or not preconditioning.get("calibration_path"):
        raise ValueError("Enable conditional preconditioning and set calibration_path")
    if training.get("model_checkpoint_load_path"):
        raise ValueError("Use train_neutrino_backend.py for a full diffusion resume, not this fresh-run launcher")
    if not training.get("strict_relation_source") or not training.get("pretrain_model_load_path"):
        raise ValueError("A strict pinned raw source is required")
    if cfg.get("rl", {}).get("enabled") or training.get("apply_event_weight"):
        raise ValueError("Calibration is an unweighted supervised experiment")
    fit = cfg["preconditioning_fit"]
    if int(fit["epochs"]) < 1 or float(fit["learning_rate"]) <= 0:
        raise ValueError("Invalid calibration training budget")
    return Path(preconditioning["calibration_path"])


def atomic_save(payload, path):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_calibration_data(artifact, platform):
    fitted_platform = artifact["fit_protocol"]["platform"]
    for key in ("data_parquet_dir", "data_parquet_val_dir"):
        if Path(fitted_platform[key]).resolve() != Path(platform[key]).resolve():
            raise ValueError("Calibration used different data; fit fresh coordinates on the filtered inputs")


def fit_worker(settings):
    import torch
    import torch.distributed as dist
    import ray.train
    import ray.train.torch
    import wandb
    from evenet.control.global_config import global_config
    from evenet.network.evenet_model import build_evenet_model_from_training_config
    from evenet.network.body.relation_conditioning import load_relation_weights
    global_config.load_yaml(settings["runtime"], current_dir=str(ROOT))
    fit = global_config.preconditioning_fit
    rank = ray.train.get_context().get_world_rank()
    world = ray.train.get_context().get_world_size()
    if world != 16:
        raise ValueError("Calibration must use 16 GPUs")
    device = ray.train.torch.get_device()
    torch.manual_seed(int(fit.seed))
    torch.set_float32_matmul_precision("highest")
    normalization = torch.load(global_config.options.Dataset.normalization_file, map_location=device, weights_only=False)
    model = build_evenet_model_from_training_config(global_config, normalization, device).to(device)
    source = torch.load(global_config.options.Training.pretrain_model_load_path, map_location=device, weights_only=False)
    load_relation_weights(model, source)
    preconditioner = model.conditional_preconditioning
    if preconditioner is None:
        raise ValueError("Missing preconditioning module")
    preconditioner.requires_grad_(True)
    del model, source, normalization
    wrapped = ray.train.torch.prepare_model(preconditioner)
    optimizer = torch.optim.AdamW(preconditioner.parameters(), lr=float(fit.learning_rate), weight_decay=float(fit.weight_decay))
    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("validation")
    batch_size = int(global_config.platform.batch_size)
    # Ray Data splits each dataset equally across workers. Fixed full batches
    # keep every rank on the same number of DDP collectives.
    steps = int(settings["train_events"]) // (world * batch_size)
    if steps < 1:
        raise ValueError("Training split is too small for one 16-GPU full batch")
    output = Path(settings["output"])
    best_path, last_path = output / "best-fit.pt", output / "last-fit.pt"
    start, best = 0, float("inf")
    if settings["resume_fit"]:
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["source_sha256"] != settings["source_sha256"] or saved["spec"] != preconditioner.spec:
            raise ValueError("Cannot resume calibration with a changed source or architecture")
        if saved["fit_protocol"] != settings["fit_protocol"]:
            raise ValueError("Cannot resume calibration with a changed training/data protocol")
        preconditioner.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["completed_epochs"], saved["best_val_nll"]
    run = None
    if rank == 0:
        logger = dict(fit.wandb)
        run = wandb.init(**logger, config={"source_sha256": settings["source_sha256"],
                         "preconditioning": preconditioner.spec, "fit": settings["fit_protocol"],
                         "train_events": settings["train_events"], "val_events": settings["val_events"]},
                         resume="allow" if settings["resume_fit"] else "never")
    selected_keys = ("x", "x_mask", "conditions", "conditions_mask", "x_invisible", "x_invisible_mask")
    def move(batch):
        return {k: batch[k].to(device=device, dtype=torch.float32 if batch[k].is_floating_point() else batch[k].dtype)
                for k in selected_keys}
    for epoch in range(start, int(fit.epochs)):
        wrapped.train()
        train_sum = torch.zeros(2, device=device, dtype=torch.float64)
        seen_steps = 0
        # Exhaust the streaming iterator, including its dropped tail, so Ray
        # closes this epoch before any worker requests the next one.
        for raw in train_ds.iter_torch_batches(batch_size=batch_size, drop_last=True, prefetch_batches=1):
            if seen_steps >= steps:
                raise RuntimeError("Ray train shard exceeds the fixed DDP step budget")
            batch = move(raw)
            nll, _, _ = preconditioner.gaussian_loss(batch, coordinates=wrapped(batch))
            optimizer.zero_grad(set_to_none=True)
            nll.mean().backward()
            torch.nn.utils.clip_grad_norm_(preconditioner.parameters(), float(fit.gradient_clip), error_if_nonfinite=True)
            optimizer.step()
            train_sum += torch.stack((nll.detach().double().sum(), nll.new_tensor(len(nll), dtype=torch.float64)))
            seen_steps += 1
        if seen_steps != steps:
            raise RuntimeError("Ray train shard shorter than the fixed DDP step budget")
        wrapped.eval()
        # Validation only reports/selects NLL: never backpropagates or updates coordinates.
        stats = torch.zeros(5, device=device, dtype=torch.float64)
        with torch.no_grad():
            for raw in val_ds.iter_torch_batches(batch_size=batch_size, prefetch_batches=1):
                nll, residual, factor = preconditioner.gaussian_loss(move(raw))
                eigenvalues = torch.linalg.eigvalsh(factor.double() @ factor.double().mT)
                stats += torch.stack((nll.double().sum(), nll.new_tensor(len(nll), dtype=torch.float64),
                                      residual.double().square().sum(), eigenvalues[:, 0].sqrt().sum(),
                                      eigenvalues[:, -1].sqrt().sum()))
        dist.all_reduce(train_sum)
        dist.all_reduce(stats)
        if stats[1] == 0:
            raise ValueError("Empty validation split")
        val_nll = float(stats[0] / stats[1])
        preconditioner.fitted.fill_(True)
        improved = val_nll < best
        best = min(best, val_nll)
        if rank == 0:
            record = {"epoch": epoch + 1, "train/gaussian_nll": float(train_sum[0] / train_sum[1]),
                      "val/gaussian_nll": val_nll, "val/whitened_rms": float((stats[2] / (4 * stats[1])).sqrt()),
                      "val/mean_min_scale": float(stats[3] / stats[1]), "val/mean_max_scale": float(stats[4] / stats[1])}
            run.log(record, step=epoch + 1)
            print(json.dumps(record), flush=True)
            state = {k: v.detach().cpu() for k, v in preconditioner.state_dict().items()}
            payload = dict(schema="conditional-preconditioning-v1", spec=preconditioner.spec,
                           state_dict=state, source_sha256=settings["source_sha256"],
                           completed_epochs=epoch + 1, fit_complete=False,
                           fit_protocol=settings["fit_protocol"], best_val_nll=best)
            if improved:
                atomic_save(payload, best_path)
            atomic_save({**payload, "optimizer": optimizer.state_dict()}, last_path)
        dist.barrier()
        ray.train.report({"epoch": epoch + 1, "val_nll": val_nll})
    if rank == 0:
        selected = torch.load(best_path, map_location="cpu", weights_only=True)
        selected.update(fit_complete=True, fit_completed_epochs=int(fit.epochs))
        atomic_save(selected, settings["calibration_path"])
        (output / "COMPLETE").write_text(json.dumps({"source_sha256": settings["source_sha256"],
            "selected_epoch": selected["completed_epochs"], "fit_completed_epochs": int(fit.epochs)}, indent=2))
        run.finish()
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=ROOT / "config/train_diffusion_nersc.yaml")
    parser.add_argument("--overlay-config", type=Path, default=ROOT / "config/train_diffusion_conditional_preconditioning.yaml")
    parser.add_argument("--stage", choices=("check", "fit", "train", "all"), default="all")
    parser.add_argument("--resume-fit", action="store_true")
    args = parser.parse_args()
    runtime = build_runtime_config(base_config=args.base_config.resolve(), overlay_config=args.overlay_config.resolve(), backend="pure-evenet")
    from evenet.control.global_config import global_config
    from evenet.network.body.conditional_preconditioning import file_sha256
    global_config.load_yaml(str(runtime), current_dir=str(ROOT))
    calibration = validate_config(global_config._global_config)
    output = calibration.parent
    source = Path(global_config.options.Training.pretrain_model_load_path)
    required = [source, Path(global_config.options.Dataset.normalization_file),
                Path(global_config.options.Training.JointCoverage.panel_path)]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    from evenet.dataset.filtered_data import validate_filtered_dataset
    filtered_inputs = [validate_filtered_dataset(global_config.platform[key])
                       for key in ("data_parquet_dir", "data_parquet_val_dir")]
    if args.stage == "check":
        print(json.dumps({"workers": 16, "runtime": str(runtime), "calibration": str(calibration),
                          "filtered_inputs": filtered_inputs,
                          "fit_epochs": global_config.preconditioning_fit.epochs,
                          "diffusion_epochs": global_config.options.Training.epochs}, indent=2))
        return
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Set WANDB_API_KEY before launch, as required by the existing EveNet trainer")
    if args.stage in ("fit", "all"):
        if calibration.exists() or (output / "COMPLETE").exists():
            raise ValueError("Calibration already completed; use --stage train to reuse it")
        if output.exists() and any(output.iterdir()) and not args.resume_fit:
            raise ValueError("Interrupted calibration exists; use --resume-fit to restore last-fit.pt")
        if args.resume_fit and not (output / "last-fit.pt").is_file():
            raise FileNotFoundError("No last-fit.pt to resume")
        output.mkdir(parents=True, exist_ok=True)
        import ray
        from ray.train import RunConfig, ScalingConfig
        from ray.train.torch import TorchTrainer
        from shared import prepare_datasets, make_process_fn
        env = {"PYTHONPATH": os.pathsep.join((str(ROOT / "evenet_dgpo"), str(ROOT / "evenet_dgpo/evenet"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")))}
        if os.environ.get("WANDB_API_KEY"):
            env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
        ray.init(runtime_env={"env_vars": env})
        ray.data.DataContext.get_current().execution_options.preserve_order = True
        platform = global_config.platform
        base = Path(platform.data_parquet_dir)
        val = Path(platform.data_parquet_val_dir)
        if base.resolve() == val.resolve():
            raise ValueError("Calibration requires separate train and validation directories")
        train, valid, ntrain, nvalid = prepare_datasets(base, make_process_fn(base), platform, base_val_dir=val)
        protocol = json.loads(json.dumps({"fit": dict(global_config.preconditioning_fit),
            "dataset": dict(global_config.options.Dataset), "platform": dict(platform)}, default=str))
        settings = dict(runtime=str(runtime), output=str(output), calibration_path=str(calibration),
                        source_sha256=file_sha256(source), train_events=ntrain, val_events=nvalid,
                        fit_protocol=protocol, resume_fit=args.resume_fit)
        trainer = TorchTrainer(train_loop_per_worker=fit_worker, train_loop_config=settings,
            scaling_config=ScalingConfig(num_workers=16, use_gpu=True, resources_per_worker=dict(platform.resources_per_worker)),
            run_config=RunConfig(name="Conditional-coordinate-fit", storage_path=str(output / "ray_results")),
            datasets={"train": train, "validation": valid})
        trainer.fit()
        ray.shutdown()
    if args.stage in ("train", "all"):
        if not calibration.is_file() or not (output / "COMPLETE").is_file():
            raise FileNotFoundError("Complete calibration first with --stage fit")
        import torch
        fitted = torch.load(calibration, map_location="cpu", weights_only=True)
        validate_calibration_data(fitted, global_config.platform)
        del fitted
        checkpoint_dir = Path(global_config.options.Training.model_checkpoint_save_path)
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise ValueError("Diffusion outputs exist; use a full checkpoint resume or a new experiment output")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "evenet_dgpo"), env.get("PYTHONPATH", "")))
        subprocess.run([*command_for_backend("pure-evenet", runtime), "--ray_dir", str(global_config.nersc.ray.results_dir)],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
