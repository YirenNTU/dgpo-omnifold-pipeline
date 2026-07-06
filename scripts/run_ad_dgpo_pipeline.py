#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for the AD/DGPO pipeline scripts. "
        "Install the EveNet / DGPO Python requirements first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
LATENT_TEMPLATE = REPO_ROOT / "evenet_dgpo" / "RL" / "DGPO_neutrino" / "latent_constraint" / "config.yaml"
DGPO_TEMPLATE = REPO_ROOT / "config" / "dgpo_post_training_overlay.yaml"
AD_TEMPLATE = REPO_ROOT / "config" / "ad_stage_overlay.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping YAML at {path}, got {type(data)!r}")
    return data


def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def absolutize_default_paths(payload: Any, base_dir: Path) -> Any:
    if isinstance(payload, dict):
        output: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "default" and isinstance(value, str):
                candidate = Path(value).expanduser()
                output[key] = str(candidate if candidate.is_absolute() else (base_dir / candidate).resolve())
            else:
                output[key] = absolutize_default_paths(value, base_dir)
        return output
    if isinstance(payload, list):
        return [absolutize_default_paths(item, base_dir) for item in payload]
    return payload


def write_runtime_yaml(prefix: str, payload: dict[str, Any]) -> Path:
    runtime_dir = Path(tempfile.mkdtemp(prefix=prefix))
    runtime_path = runtime_dir / "runtime.yaml"
    with runtime_path.open("w") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return runtime_path


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def resolve_checkpoint(path_or_dir: Path) -> Path:
    candidate = path_or_dir.expanduser().resolve()
    if candidate.is_file():
        return candidate
    ckpts = sorted(candidate.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {candidate}")
    last_link = candidate / "last.ckpt"
    if last_link.is_file():
        return last_link.resolve()
    return ckpts[-1]


def ensure_split_root(config_dict: dict[str, Any]) -> Path:
    train_input_dir = Path(config_dict["platform"]["data_parquet_dir"]).resolve()
    val_input_dir = Path(config_dict["platform"]["data_parquet_val_dir"]).resolve()
    if train_input_dir.name != "train" or val_input_dir.name != "val" or train_input_dir.parent != val_input_dir.parent:
        raise ValueError(
            "This pipeline expects platform.data_parquet_dir=<root>/train and "
            "platform.data_parquet_val_dir=<root>/val so both splits can be processed together."
        )
    return train_input_dir.parent


def merged_config(base_config: Path, overlay_config: Path | None) -> dict[str, Any]:
    merged = read_yaml(base_config)
    if overlay_config is not None:
        merged = deep_update(merged, read_yaml(overlay_config))
    return absolutize_default_paths(merged, base_config.parent)


def build_latent_runtime_config(
    *,
    ad_config: dict[str, Any],
    augmented_root: Path,
    stage_root: Path,
) -> Path:
    template = read_yaml(LATENT_TEMPLATE)
    payload = deep_update(template, {
        "platform": {
            "data_parquet_dir": str((augmented_root / "train").resolve()),
            "data_parquet_val_dir": str((augmented_root / "val").resolve()),
            "number_of_workers": ad_config["platform"]["number_of_workers"],
            "resources_per_worker": ad_config["platform"]["resources_per_worker"],
            "batch_size": ad_config["platform"]["batch_size"],
            "prefetch_batches": ad_config["platform"]["prefetch_batches"],
            "use_gpu": ad_config["platform"].get("use_gpu", True),
        },
        "options": {
            "default": ad_config["options"]["default"],
            "Training": {
                "total_epochs": ad_config["options"]["Training"]["total_epochs"],
                "epochs": ad_config["options"]["Training"]["epochs"],
                "model_checkpoint_save_path": str((stage_root / "latent_constraint" / "checkpoints").resolve()),
                "Components": {
                    "Classification": {"include": False},
                    "Regression": {"include": False},
                    "Assignment": {"include": True},
                    "GlobalGeneration": {"include": False},
                    "ReconGeneration": {"include": False},
                    "TruthGeneration": {
                        "include": True,
                        "cartesian": ad_config["options"]["Training"]["Components"]["TruthGeneration"].get("cartesian", False),
                    },
                },
            },
            "Dataset": {
                "dataset_limit": ad_config["options"]["Dataset"]["dataset_limit"],
                "normalization_file": ad_config["options"]["Dataset"]["normalization_file"],
                "val_split": ad_config["options"]["Dataset"]["val_split"],
            },
        },
        "network": ad_config["network"],
        "event_info": ad_config["event_info"],
        "resonance": ad_config["resonance"],
        "logger": {
            **template.get("logger", {}),
            "save_dir": str((stage_root / "latent_constraint" / "logs").resolve()),
            "name": "latent_constraint_ztautau",
        },
        "wandb": {
            **template.get("wandb", {}),
            "run_name": "latent-constraint-ztautau",
        },
    })
    return write_runtime_yaml("ad_latent_", payload)


def build_ad_runtime_config(
    *,
    base_config: Path,
    overlay_config: Path | None,
) -> Path:
    payload = merged_config(base_config, overlay_config)
    payload.setdefault("compat", {})
    payload["compat"]["backend"] = "pure-evenet"
    payload["compat"]["repo_root"] = str(REPO_ROOT)
    payload.setdefault("rl", {})
    payload["rl"]["enabled"] = False
    return write_runtime_yaml("ad_runtime_", payload)


def build_dgpo_runtime_overlay(
    *,
    augmented_root: Path,
    latent_checkpoint: Path,
    diffusion_config: dict[str, Any],
    stage_root: Path,
    dgpo_init_checkpoint: Path | None,
) -> Path:
    template = read_yaml(DGPO_TEMPLATE)
    payload = deep_update(template, {
        "platform": {
            "data_parquet_dir": str((augmented_root / "train").resolve()),
            "data_parquet_val_dir": str((augmented_root / "val").resolve()),
        },
        "options": {
            "Training": {
                "model_checkpoint_save_path": str((stage_root / "diffusion" / "checkpoints").resolve()),
            },
            "Dataset": {
                "normalization_file": diffusion_config["options"]["Dataset"]["normalization_file"],
            },
        },
        "logger": {
            "wandb": {
                "run_name": "EveNet-DGPO-Diffusion",
            },
            "local": {
                "name": "EveNet-DGPO-Diffusion",
            },
        },
        "dgpo": {
            "projection_constraint": {
                "latent_swd": {
                    "checkpoint_file": str(latent_checkpoint.resolve()),
                    "normalization_file": diffusion_config["options"]["Dataset"]["normalization_file"],
                }
            }
        },
        "ztautau_domain": {
            "token_sources": {
                "event_token": "event_token",
                "object_token": "object_token",
            }
        }
    })
    if dgpo_init_checkpoint is not None:
        payload["options"]["Training"]["model_checkpoint_load_path"] = str(dgpo_init_checkpoint.resolve())
        payload["options"]["Training"]["pretrain_model_load_path"] = None
    return write_runtime_yaml("dgpo_overlay_", payload)


def resolve_ad_checkpoint(
    *,
    ad_config: dict[str, Any],
    ad_checkpoint: Path | None,
) -> Path:
    if ad_checkpoint is not None:
        return resolve_checkpoint(ad_checkpoint)
    seed_path = (
        ad_config["options"]["Training"].get("model_checkpoint_load_path")
        or ad_config["options"]["Training"].get("pretrain_model_load_path")
    )
    if not seed_path:
        raise ValueError(
            "AD stage needs a frozen backbone checkpoint. Pass --ad-checkpoint or set "
            "options.Training.model_checkpoint_load_path / pretrain_model_load_path in the AD config."
        )
    return resolve_checkpoint(Path(seed_path))


def run_ad_stage(
    *,
    ad_base_config: Path,
    ad_overlay_config: Path | None,
    stage_root: Path,
    augmented_dirname: str,
    ad_checkpoint: Path | None,
    latent_checkpoint: Path | None,
    token_batch_size: int,
    token_workers: int | None,
    token_devices: str | None,
) -> tuple[Path, Path]:
    ad_config_dict = merged_config(ad_base_config, ad_overlay_config)
    ad_runtime_config = build_ad_runtime_config(
        base_config=ad_base_config,
        overlay_config=ad_overlay_config,
    )
    resolved_ad_checkpoint = resolve_ad_checkpoint(
        ad_config=ad_config_dict,
        ad_checkpoint=ad_checkpoint,
    )
    input_root = ensure_split_root(ad_config_dict)
    augmented_root = stage_root / augmented_dirname

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "augment_event_tokens.py"),
        "--train-config",
        str(ad_runtime_config),
        "--checkpoint",
        str(resolved_ad_checkpoint),
        "--input-dir",
        str(input_root),
        "--output-dir",
        str(augmented_root),
        "--splits",
        "train",
        "val",
        "--batch-size",
        str(token_batch_size),
        "--use-gpu" if ad_config_dict["platform"].get("use_gpu", True) else "--no-use-gpu",
        "--devices",
        token_devices if token_devices is not None else "auto",
    ]
    if token_workers is not None:
        command.extend([
            "--num-workers",
            str(token_workers),
        ])
    run_command(command)

    if latent_checkpoint is None:
        latent_runtime = build_latent_runtime_config(
            ad_config=ad_config_dict,
            augmented_root=augmented_root,
            stage_root=stage_root,
        )
        run_command([
            sys.executable,
            str(REPO_ROOT / "evenet_dgpo" / "RL" / "DGPO_neutrino" / "latent_constraint" / "train_latent_constraint.py"),
            str(latent_runtime),
        ])
        resolved_latent_checkpoint = resolve_checkpoint(stage_root / "latent_constraint" / "checkpoints")
    else:
        resolved_latent_checkpoint = resolve_checkpoint(latent_checkpoint)

    return augmented_root, resolved_latent_checkpoint


def run_diffusion_stage(
    *,
    diffusion_base_config: Path,
    diffusion_overlay_config: Path | None,
    stage_root: Path,
    augmented_dirname: str,
    latent_checkpoint: Path | None,
    dgpo_init_checkpoint: Path | None,
) -> None:
    augmented_root = stage_root / augmented_dirname
    if not (augmented_root / "train").is_dir() or not (augmented_root / "val").is_dir():
        raise FileNotFoundError(
            f"Augmented AD parquet not found under {augmented_root}. Run the AD stage first or pass the matching --stage-root."
        )
    resolved_latent_checkpoint = (
        resolve_checkpoint(latent_checkpoint)
        if latent_checkpoint is not None
        else resolve_checkpoint(stage_root / "latent_constraint" / "checkpoints")
    )
    diffusion_config_dict = merged_config(diffusion_base_config, diffusion_overlay_config)
    dgpo_overlay = build_dgpo_runtime_overlay(
        augmented_root=augmented_root,
        latent_checkpoint=resolved_latent_checkpoint,
        diffusion_config=diffusion_config_dict,
        stage_root=stage_root,
        dgpo_init_checkpoint=dgpo_init_checkpoint,
    )
    run_command([
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_neutrino_backend.py"),
        "--backend",
        "dgpo-evenet",
        "--base-config",
        str(diffusion_base_config),
        "--overlay-config",
        str(dgpo_overlay),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wrapped two-stage AD + diffusion pipeline.")
    parser.add_argument("--stage", choices=("ad", "diffusion", "all"), default="all")
    parser.add_argument("--ad-base-config", type=Path, default=REPO_ROOT / "config" / "train_pretrain_cls.yaml")
    parser.add_argument("--ad-overlay-config", type=Path, default=AD_TEMPLATE)
    parser.add_argument("--diffusion-base-config", type=Path, default=REPO_ROOT / "config" / "train_pretrain.yaml")
    parser.add_argument("--diffusion-overlay-config", type=Path, default=DGPO_TEMPLATE)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--augmented-dirname", default="ad_augmented")
    parser.add_argument("--train-ad-backbone", action="store_true", help="Run pure-evenet classification training before frozen token export.")
    parser.add_argument("--ad-checkpoint", type=Path, default=None, help="Existing frozen backbone checkpoint for token export.")
    parser.add_argument("--latent-checkpoint", type=Path, default=None, help="Existing latent constraint checkpoint.")
    parser.add_argument(
        "--dgpo-init-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to initialize DGPO stage 2. This is independent from the AD frozen backbone checkpoint.",
    )
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--token-workers", type=int, default=None)
    parser.add_argument("--token-devices", default=None, help="Comma-separated GPU device list for AD token export, or 'auto'.")
    args = parser.parse_args()

    stage_root = args.stage_root.expanduser().resolve()
    stage_root.mkdir(parents=True, exist_ok=True)
    ad_base_config = args.ad_base_config.resolve()
    ad_overlay_config = args.ad_overlay_config.resolve() if args.ad_overlay_config else None
    diffusion_base_config = args.diffusion_base_config.resolve()
    diffusion_overlay_config = args.diffusion_overlay_config.resolve() if args.diffusion_overlay_config else None
    ad_config_dict = merged_config(ad_base_config, ad_overlay_config)
    ad_checkpoint = args.ad_checkpoint.expanduser().resolve() if args.ad_checkpoint is not None else None
    latent_checkpoint = args.latent_checkpoint.expanduser().resolve() if args.latent_checkpoint is not None else None
    dgpo_init_checkpoint = (
        args.dgpo_init_checkpoint.expanduser().resolve()
        if args.dgpo_init_checkpoint is not None
        else None
    )

    if args.stage in ("ad", "all"):
        if args.train_ad_backbone:
            run_command([
                sys.executable,
                str(REPO_ROOT / "scripts" / "train_neutrino_backend.py"),
                "--backend",
                "pure-evenet",
                "--base-config",
                str(ad_base_config),
                "--overlay-config",
                str(ad_overlay_config),
            ])
            ad_checkpoint = resolve_checkpoint(Path(ad_config_dict["options"]["Training"]["model_checkpoint_save_path"]))
        augmented_root, latent_checkpoint = run_ad_stage(
            ad_base_config=ad_base_config,
            ad_overlay_config=ad_overlay_config,
            stage_root=stage_root,
            augmented_dirname=args.augmented_dirname,
            ad_checkpoint=ad_checkpoint,
            latent_checkpoint=latent_checkpoint,
            token_batch_size=args.token_batch_size,
            token_workers=args.token_workers,
            token_devices=args.token_devices,
        )
    else:
        augmented_root = stage_root / args.augmented_dirname

    if args.stage in ("diffusion", "all"):
        run_diffusion_stage(
            diffusion_base_config=diffusion_base_config,
            diffusion_overlay_config=diffusion_overlay_config,
            stage_root=stage_root,
            augmented_dirname=args.augmented_dirname,
            latent_checkpoint=latent_checkpoint,
            dgpo_init_checkpoint=dgpo_init_checkpoint,
        )


if __name__ == "__main__":
    main()
