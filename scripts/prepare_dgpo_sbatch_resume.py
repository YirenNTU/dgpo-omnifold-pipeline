#!/usr/bin/env python3
"""Snapshot a full DGPO resume config into a new, job-specific output directory."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from train_neutrino_backend import (
    REPO_ROOT, absolutize_default_paths, deep_update, read_yaml,
)


def check_10pct_protocol(cfg: dict) -> None:
    """Fail closed if the template was accidentally switched to another experiment."""
    provenance = cfg["nersc"]["reproducibility"]
    for key in ("diffusion_pretrain_fraction", "dgpo_training_fraction"):
        if provenance.get(key) != 0.10:
            raise ValueError(f"10pct resume requires {key}=0.10")
    pool = "/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train"
    for key in ("data_parquet_dir", "data_parquet_val_dir"):
        if cfg["platform"][key] != pool:
            raise ValueError(f"10pct resume requires the existing STIC-filtered 10% pool: {key}")
    if "diffusion_pretrain_10pct_seed42/checkpoints/" not in cfg["reward_config"]["omnifold"]["backbone_checkpoint"]:
        raise ValueError("Expected the matched 10pct classifier backbone")
    adaptive = cfg["dgpo"]["adaptive_omnifold"]
    if adaptive["trigger"]["probe_max_events"] != 250000:
        raise ValueError("Preserve the 10pct staleness pool: 250k total event identities")
    if cfg["platform"]["number_of_workers"] != 16:
        raise ValueError("This sbatch requires the 16-worker configuration")
    for key in ("dataset_limit", "val_dataset_limit"):
        if cfg["options"]["Dataset"][key] != 1.0:
            raise ValueError("Use the full already-subsetted 10pct pool, not another 10% cut")


def prepare_resume(base: Path, overlay: Path, checkpoint: Path, output: Path) -> Path:
    # Pin last.ckpt's symlink target once; a running source job may advance it later.
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"Missing/empty resume checkpoint: {checkpoint}")
    output = output.resolve()
    if output == checkpoint.parent.parent or output in checkpoint.parents:
        raise ValueError("Resume output must be separate from the source checkpoint.")
    base = base.resolve()
    cfg = absolutize_default_paths(deep_update(read_yaml(base), read_yaml(overlay)), base.parent)
    check_10pct_protocol(cfg)
    training = cfg["options"]["Training"]
    dgpo = cfg["dgpo"]
    search_dirs = list(dgpo.get("global_best_checkpoint_search_dirs", []))
    # Preserve older rollback sources as well as the source of this continuation.
    for source in (training.get("model_checkpoint_load_path"), str(checkpoint)):
        if source:
            directory = str(Path(source).parent)
            if directory not in search_dirs:
                search_dirs.append(directory)
    training.update(
        model_checkpoint_load_path=str(checkpoint), pretrain_model_load_path=None,
        model_checkpoint_save_path=str(output / "checkpoints"),
    )
    training.setdefault("EMA", {}).update(replace_model_after_load=False, use_for_generation=False)
    dgpo.update(
        checkpoint_load_mode="resume", auto_resume_from_last=False,
        auto_resume_best_source_checkpoint_dir=None, best_source_start_new_experiment=False,
        auto_resume_fallback_checkpoint_path=None, global_best_checkpoint_search_dirs=search_dirs,
    )
    dgpo["adaptive_omnifold"]["recalibration"].update(
        bootstrap_on_start=False, refit_once_on_resume=False,
    )
    cfg["logger"]["wandb"].update(
        resume="never", fresh_run=True, id=None, run_name=output.name,
    )
    cfg["logger"]["local"].update(save_dir=str(output / "logs"), name=output.name, version=output.name)
    cfg.setdefault("nersc", {}).setdefault("ray", {})["results_dir"] = str(output / "ray_results")
    cfg["nersc"].setdefault("execution", {})["command"] = (
        f"shifter python3 evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py {output / 'runtime.yaml'} "
        f"--ray-dir {output / 'ray_results'}"
    )
    cfg["nersc"].setdefault("reproducibility", {})["note"] = (
        f"Full DGPO continuation from {checkpoint}; preserve optimizer, clocks, OmniFold, "
        "monitor and global-best state; fresh W&B run, isolated sbatch outputs."
    )
    cfg.setdefault("compat", {}).update(backend="dgpo-evenet", repo_root=str(REPO_ROOT))
    cfg.setdefault("rl", {})["enabled"] = True
    # Refuse reuse, including an existing job directory; never truncate prior results.
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime.yaml"
    with runtime.open("x") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print(f"Resume checkpoint (pinned): {checkpoint}", flush=True)
    print(f"Runtime config: {runtime}", flush=True)
    print(f"10pct training/validation pool: {cfg['platform']['data_parquet_dir']}", flush=True)
    print("Full resume: saved epoch/step, optimizer/scheduler, OmniFold and monitor state.", flush=True)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "config/train_diffusion_nersc.yaml")
    parser.add_argument("--overlay-config", type=Path, default=REPO_ROOT / "config/dgpo_omnifold_ztautau_10pct_resume_v26.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_resume(args.base_config, args.overlay_config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
