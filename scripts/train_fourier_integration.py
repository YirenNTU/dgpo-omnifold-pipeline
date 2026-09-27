#!/usr/bin/env python3
"""Prepare/check/run one user-launched supervised integration arm; never submit jobs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from train_neutrino_backend import (
    REPO_ROOT, absolutize_default_paths, deep_update, read_overlay_yaml, read_yaml,
)

READOUT_ARMS = ("readout_k4", "readout_multiscale")
ARMS = ("none", "input", "output", "input_slow", "input_fast", "input_mlp", *READOUT_ARMS)
PRETRAIN_ARMS = ("none", "input", "input_slow", "input_fast")
BASE = REPO_ROOT / "config/train_diffusion_nersc.yaml"


def make_config(arm, checkpoint, output_root, seed=42):
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    cfg = deep_update(read_yaml(BASE), read_overlay_yaml(
        REPO_ROOT / f"config/train_diffusion_fourier_integration_{arm}.yaml"))
    cfg = absolutize_default_paths(cfg, BASE.parent)
    run_dir = Path(output_root).expanduser().resolve() / f"seed{seed}" / arm
    training = cfg["options"]["Training"]
    checkpoint = checkpoint if checkpoint is not None else training.get("pretrain_model_load_path")
    training.update(seed=seed, paired_diffusion_seed=420024 + seed - 42,
                    pretrain_model_load_path=str(Path(checkpoint).expanduser().resolve()) if checkpoint else None,
                    model_checkpoint_load_path=None,
                    model_checkpoint_save_path=str(run_dir / "checkpoints"))
    cfg["logger"]["wandb"]["id"] = None  # fresh run ID, no accidental history append
    cfg["logger"]["wandb"]["tags"] += [f"seed{seed}", arm]
    cfg["logger"]["local"] = dict(save_dir=str(run_dir / "logs"), name=arm, version="v1")
    cfg["rl"] = {"enabled": False}
    cfg["compat"] = {"backend": "pure-evenet", "repo_root": str(REPO_ROOT)}
    cfg["nersc"]["submit"] = False
    cfg["nersc"]["ray"]["results_dir"] = str(run_dir / "ray_results")
    cfg["nersc"]["execution"]["command"] = f"shifter python3 scripts/train_fourier_integration.py --arm {arm}"
    cfg["experiment"]["arm"] = arm
    cfg["experiment"]["source_checkpoint"] = training["pretrain_model_load_path"]
    return cfg


def validate_config(cfg):
    t = cfg["options"]["Training"]
    p = cfg["platform"]
    checks = {
        "pure supervised diffusion": not cfg["rl"]["enabled"],
        "weights-only loading": t["model_checkpoint_load_path"] is None,
        "strict baseline load": t["strict_no_fourier_source"],
        "raw weights": not t["EMA"]["enable"] and not t["EMA"]["replace_model_after_load"],
        "all backbone weights trainable": t["freeze_modules"] == [],
        "fixed budget": t["epochs"] == t["total_epochs"] == 50 and t["EarlyStopping"]["patience"] > 50,
        "paired batches": p["ordered_data"] and t["paired_diffusion_seed"] is not None,
        "16 GPUs, same batch": p["number_of_workers"] == 16 and p["batch_size"] == 2048,
        "explicit effective LR": not t["scale_lr_with_world_size"] and t["learning_rate_factor"] == 1,
        "unchanged loss": t["Components"]["TruthGeneration"]["low_noise_weight"] == 1,
        "readable W&B name": len(cfg["logger"]["wandb"]["run_name"]) < 96,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid integration experiment: {failed}")


def preflight(checkpoint, output_root, seed, arms=PRETRAIN_ARMS):
    """CPU-only real-checkpoint/real-batch equivalence check, no Ray or training."""
    import torch
    import pyarrow.parquet as pq
    sys.path.insert(0, str(REPO_ROOT / "evenet_dgpo"))
    from evenet.control.global_config import Config
    from evenet.dataset.preprocess import unflatten_dict
    from evenet.network.evenet_model import build_evenet_model_from_training_config
    from evenet.utilities.fourier_integration import load_no_fourier_weights, paired_diffusion_rng

    # Only load user-selected trusted local training checkpoints.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    report = {"source_checkpoint": str(checkpoint), "source_epoch": ckpt.get("epoch"),
              "source_global_step": ckpt.get("global_step"), "seed": seed, "arms": {}}
    batch = None
    references = {}
    for arm in ("none", *(a for a in arms if a != "none")):
        cfg = make_config(arm, checkpoint, output_root, seed)
        validate_config(cfg)
        with tempfile.TemporaryDirectory(prefix="fourier-preflight-") as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            config = Config()
            config.load_yaml(path)
        if batch is None:
            val_dir = Path(cfg["platform"]["data_parquet_val_dir"])
            files = sorted(val_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No validation parquet: {val_dir}")
            for file in files:
                record = next(pq.ParquetFile(file).iter_batches(batch_size=2), None)
                if record is not None and record.num_rows:
                    break
            else:
                raise ValueError("Validation files contain no events")
            flat = {name: record.column(i).to_numpy(zero_copy_only=False)
                    for i, name in enumerate(record.schema.names)}
            # Production also reconstructs validation columns using the train schema.
            metadata = json.loads((Path(cfg["platform"]["data_parquet_dir"]) / "shape_metadata.json").read_text())
            arrays = unflatten_dict(flat, metadata,
                drop_column_prefix=["EXTRA/", "regression-", "assignments-", "segmentation-"])
            batch = {k: torch.as_tensor(v.copy()) for k, v in arrays.items()}
            report["validation_file"] = str(file)
            report["validation_events"] = int(batch["x"].shape[0])
        normalization = torch.load(config.options.Dataset.normalization_file,
                                   map_location="cpu", weights_only=False)
        torch.manual_seed(seed)
        model = build_evenet_model_from_training_config(config, normalization, torch.device("cpu"))
        load_no_fourier_weights(model, ckpt)
        arm_report = {"parameters": sum(p.numel() for p in model.parameters()),
                      "all_parameters_trainable": all(p.requires_grad for p in model.parameters())}
        if not arm_report["all_parameters_trainable"]:
            raise ValueError(f"Unexpected frozen parameters in {arm}")
        for training in (False, True):
            model.train(training)
            with torch.no_grad(), paired_diffusion_rng(
                seed, training=training, epoch=0, batch_idx=0, rank=0, device="cpu"
            ):
                result = model.shared_step(batch, batch["x"].shape[0], {},
                    schedules=[("neutrino_generation", True)])['generations']['neutrino']
            mode = "train" if training else "eval"
            for key in ("vector", "truth", "time"):
                tensor = result[key]
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite initial {mode}/{key} in {arm}")
                if arm == "none":
                    references[(mode, key)] = tensor.clone()
                else:
                    torch.testing.assert_close(tensor, references[(mode, key)], rtol=0, atol=0)
            mask = result['mask'].float()
            mse = (((result['vector'] - result['truth']).square() * mask).sum()
                   / (mask.sum() * result['vector'].shape[-1]))
            arm_report[f"initial_{mode}_velocity_mse"] = float(mse)
        if arm == "none":
            report["source_reference"] = {**arm_report, "optimizer_updates": 0}
        else:
            report["arms"][arm] = arm_report
        del model
    report["initial_predictions_equal"] = True
    report["limitation"] = "Two-event CPU check; distributed GPU/data-order equivalence still needs NERSC verification"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional override of the shared YAML source checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path,
                        default=Path('/pscratch/sd/y/yiren/Ztautau/fourier_integration'))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Validate config without data, Torch, or training")
    modes.add_argument("--preflight", action="store_true", help="Check the matched experiment arms on real data, without training")
    args = parser.parse_args()
    cfg = make_config(args.arm, args.checkpoint, args.output_root, args.seed)
    validate_config(cfg)
    if args.check_only:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    source = cfg['options']['Training']['pretrain_model_load_path']
    if not source or not Path(source).is_file():
        parser.error(f"Missing source checkpoint: {source}. Set the YAML path or override --checkpoint.")
    args.checkpoint = Path(source).resolve()
    if args.preflight:
        arms = (("input", *READOUT_ARMS) if args.arm in READOUT_ARMS else
                ("input", args.arm) if args.arm in ("output", "input_mlp") else PRETRAIN_ARMS)
        report = preflight(args.checkpoint, args.output_root, args.seed, arms=arms)
        print(json.dumps(report, indent=2))
        return
    run_dir = Path(cfg['options']['Training']['model_checkpoint_save_path']).parent
    # Refuse to mix a new run with an earlier arm's outputs. Choose a new root to rerun.
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'runtime.yaml'
    with path.open('x') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(REPO_ROOT / 'evenet_dgpo'), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    env.pop('WANDB_RUN_ID', None)
    env.pop('WANDB_RESUME', None)
    command = [sys.executable, str(REPO_ROOT / 'evenet_dgpo/evenet/train.py'), str(path),
               '--ray_dir', str(run_dir / 'ray_results')]
    subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)


if __name__ == '__main__':
    main()
