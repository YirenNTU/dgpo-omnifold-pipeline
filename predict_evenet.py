from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
import vector
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_PIPELINE_ROOT = REPO_ROOT / "ml_pipeline"
EVENET_ROOT = ML_PIPELINE_ROOT / "EveNet-Full"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EVENET_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENET_ROOT))
if str(ML_PIPELINE_ROOT / "util") not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT / "util"))

from common import (
    read_yaml
)

from generate_event_info_yaml import(
    FeatureConfig, parse_evenet_config, parse_feature_config
)
from build_evenet_input_from_parquet import (
    parse_samples
)

from evenet.control.global_config import global_config
from evenet.dataset.preprocess import unflatten_dict
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.diffusion_sampler import DDIMSampler
from evenet.utilities.tool import safe_load_state

vector.register_awkward()

@dataclass(frozen=True)
class InferenceTask:
    sample_name: str
    input_path: str
    output_path: str
    event_start: int | None = None
    event_stop: int | None = None
    final_output_path: str | None = None

def infer_chunks_per_file(args: argparse.Namespace, num_workers: int) -> int:
    """
    inputs:
      args: argparse.Namespace, CLI options.
      num_workers: int, local GPU/CPU worker count.
    outputs:
      chunks_per_file: int, number of parquet pieces per input file.
    goal:
      Keep single-node behavior unchanged while allowing multi-node jobs to
      create enough chunks for all participating GPUs.
    """
    if args.chunks_per_file is not None:
        if args.chunks_per_file < 1:
            raise ValueError(f"--chunks-per-file must be >= 1, got {args.chunks_per_file}.")
        return int(args.chunks_per_file)
    if args.task_num_shards > 1:
        return max(1, int(args.num_gpus) * int(args.task_num_shards))
    return max(1, int(num_workers))

def resolve_converted_parquets(converted_parquets) -> list[str]:
    resolved = []
    for item in converted_parquets:
        path_text = os.path.expanduser(str(item))
        matches = sorted(glob.glob(path_text))
        candidates = matches if matches else [path_text]
        for candidate in candidates:
            path = Path(candidate)
            if path.is_dir():
                resolved.extend(str(parquet_path.resolve()) for parquet_path in sorted(path.glob("*.parquet")))
            else:
                resolved.append(str(path.resolve()))
    return resolved

def parquet_num_rows(parquet_path: Path) -> int:
    return int(pq.ParquetFile(parquet_path).metadata.num_rows)

def split_event_ranges(num_events: int, num_chunks: int) -> list[tuple[int, int]]:
    if num_chunks <= 1 or num_events <= 0:
        return [(0, num_events)]
    edges = np.linspace(0, num_events, num_chunks + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(num_chunks) if int(edges[i + 1]) > int(edges[i])]

def merge_converted_chunk_outputs(tasks: list[InferenceTask], delete_parts: bool = False) -> None:
    grouped_tasks: dict[str, list[InferenceTask]] = {}
    for task in tasks:
        final_output_path = task.final_output_path or task.output_path
        grouped_tasks.setdefault(final_output_path, []).append(task)

    for final_output_path, group in grouped_tasks.items():
        if len(group) == 1 and group[0].output_path == final_output_path:
            continue

        chunk_paths = [Path(task.output_path) for task in sorted(group, key=lambda item: item.event_start or 0)]
        final_path = Path(final_output_path)
        missing_chunk_paths = [chunk_path for chunk_path in chunk_paths if not chunk_path.exists()]
        if missing_chunk_paths:
            if final_path.exists():
                print(
                    f"[converted-merge] skipped {final_path}; final output already exists "
                    f"and {len(missing_chunk_paths)}/{len(chunk_paths)} chunk(s) are absent",
                    flush=True,
                )
                continue
            missing_preview = ", ".join(str(path) for path in missing_chunk_paths[:5])
            raise FileNotFoundError(
                f"Cannot merge {final_path}; missing {len(missing_chunk_paths)}/{len(chunk_paths)} chunk file(s): "
                f"{missing_preview}"
            )
        arrays = [ak.from_parquet(chunk_path) for chunk_path in chunk_paths]
        merged = ak.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        order = np.argsort(ak.to_numpy(merged["event_index"], allow_missing=False))
        merged = merged[order]

        final_path.parent.mkdir(parents=True, exist_ok=True)
        ak.to_parquet(merged, final_path)
        print(f"[converted-merge] wrote {final_path} from {len(chunk_paths)} chunk(s)", flush=True)

        if delete_parts:
            for chunk_path in chunk_paths:
                if chunk_path != final_path and chunk_path.exists():
                    chunk_path.unlink()


def build_converted_tasks(
    converted_parquet: list[str],
    output_dir: Path,
    num_chunks_per_file: int,
    max_events_per_chunk: int | None = None,
) -> list[InferenceTask]:
    tasks: list[InferenceTask] = []
    for input_path in converted_parquet:
        input_file = Path(input_path).expanduser().resolve()
        final_output_path = output_dir / f"{input_file.stem}__evenet_pred.parquet"
        num_events = parquet_num_rows(input_file)
        file_chunks = int(num_chunks_per_file)
        if max_events_per_chunk is not None and max_events_per_chunk > 0:
            file_chunks = min(file_chunks, max(1, math.ceil(num_events / max_events_per_chunk)))
        for shard_index, (start, stop) in enumerate(split_event_ranges(num_events, file_chunks)):
            chunk_output_path = output_dir / f"{input_file.stem}__evenet_pred.part{shard_index:03d}.parquet"
            tasks.append(
                InferenceTask(
                    sample_name=input_file.stem,
                    input_path=str(input_file),
                    output_path=str(chunk_output_path),
                    event_start=start,
                    event_stop=stop,
                    final_output_path=str(final_output_path),
                )
            )
    return tasks


def prepare_runtime_train_config(
    train_config_path: Path,
    analysis_config_path: Path,
    evenet_config_path: Path,
    checkpoint_override: Path | None,
) -> Path:
    train_cfg = read_yaml(train_config_path)
    analysis_cfg = read_yaml(analysis_config_path)
    feature_config = parse_feature_config(analysis_cfg)
    samples = parse_samples(analysis_cfg)
    evenet_schema_cfg = read_yaml(evenet_config_path)
    evenet_config = parse_evenet_config(
        evenet_schema_cfg, analysis_cfg,
        feature_config,
    )

    event_info_cfg = train_cfg.get("event_info", {})
    existing_event_info = event_info_cfg.get("default")
    if existing_event_info is not None:
        existing_event_info = str((train_config_path.parent / existing_event_info).resolve()) \
            if not Path(existing_event_info).is_absolute() else str(Path(existing_event_info).resolve())

    train_cfg["event_info"] = {"default": existing_event_info}
    if checkpoint_override is not None:
        train_cfg.setdefault("options", {}).setdefault("Training", {})["model_checkpoint_load_path"] = str(
            checkpoint_override.resolve()
        )

    print(train_cfg)
    runtime_dir = Path(tempfile.mkdtemp(prefix="evenet_predict_cfg_"))
    runtime_train_cfg = runtime_dir / "train_runtime.yaml"
    with runtime_train_cfg.open("w") as handle:
        yaml.safe_dump(train_cfg, handle, sort_keys=False)
    return runtime_train_cfg

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone EveNet predictor for converted parquet inputs."
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=REPO_ROOT / "ml_pipeline" / "config" / "analysis.yaml",
        help="analysis.yaml used for feature layout, class ordering, and MC normalization.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=REPO_ROOT / "ml_pipeline" / "config" / "train_pretrain.yaml",
        help="train.yaml used to construct the EveNet model and normalization setup.",
    )
    parser.add_argument(
        "--evenet-config",
        type=Path,
        default=REPO_ROOT / "ml_pipeline" / "config" / "evenet_schema.yaml",
        help="EveNet schema config used to generate a temporary event_info if needed.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Legacy checkpoint override used for both classification and diffusion. "
            "If omitted, use options.Training.model_checkpoint_load_path from train config. "
            "Not required when both --classification-checkpoint and --diffusion-checkpoint are set."
        ),
    )
    parser.add_argument(
        "--classification-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional classification checkpoint. When set, classification is evaluated with this "
            "checkpoint while neutrino diffusion uses --diffusion-checkpoint or --checkpoint."
        ),
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional diffusion/generation checkpoint. When set, neutrino sampling uses this "
            "checkpoint while classification uses --classification-checkpoint or --checkpoint."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where augmented parquet outputs are written. If omitted, uses "
            "EveNetPrediction.predict_output_dir or EveNetPrediction.output_dir from analysis.yaml."
        ),
    )
    parser.add_argument(
        "--converted-parquet",
        nargs="+",
        default=None,
        help=(
            "EveNet-converted parquet files to run prediction on. If omitted, uses "
            "EveNetPrediction.converted_parquets from analysis.yaml."
        ),
    )
    parser.add_argument(
        "--shape-metadata",
        type=Path,
        default=None,
        help="Optional shape_metadata.json. Defaults to <parquet_dir>/shape_metadata.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPU workers. Falls back to CPU single-process if CUDA is unavailable.",
    )
    parser.add_argument(
        "--task-num-shards",
        type=int,
        default=1,
        help=(
            "Split prediction tasks across independent jobs, for example one job per NERSC node. "
            "Each job writes only its assigned part files."
        ),
    )
    parser.add_argument(
        "--task-shard-index",
        type=int,
        default=0,
        help="Shard index for this independent prediction job. Must be in [0, --task-num-shards).",
    )
    parser.add_argument(
        "--chunks-per-file",
        type=int,
        default=None,
        help=(
            "Number of event chunks to create for each converted parquet. Defaults to local GPU workers "
            "for a single job, or --num-gpus * --task-num-shards for sharded multi-node jobs."
        ),
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Keep .part*.parquet outputs and do not merge them into final prediction parquets.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing .part*.parquet outputs and exit without loading checkpoints or running inference.",
    )
    parser.add_argument(
        "--delete-merged-parts",
        action="store_true",
        help="After a successful merge, delete .part*.parquet chunk files. Defaults to keeping parts.",
    )
    parser.add_argument(
        "--converted-split-fraction",
        type=float,
        default=None,
        help=(
            "Optional fraction of the original MC sample represented by each converted parquet "
            "(for example 0.5 for evenet_test.parquet). When set, MC weights are rescaled by 1/fraction."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="TruthGeneration diffusion sampling steps. Default: 200.",
    )
    parser.add_argument(
        "--disable-ema",
        action="store_true",
        help="If set, load raw state_dict for diffusion instead of ema_state_dict. Classification always uses raw state_dict.",
    )
    parser.add_argument(
        "--use-truth-classification",
        action="store_true",
        help="If set, use truth classification, skip classification prediction"
    )
    args = parser.parse_args()

    analysis_config = read_yaml(args.analysis_config)
    output_dir = args.output_dir
    converted_parquets = resolve_converted_parquets(args.converted_parquet)
    converted_split_fraction = args.converted_split_fraction
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and args.num_gpus > 0
    num_workers = min(args.num_gpus, torch.cuda.device_count()) if use_cuda else 1
    chunks_per_file = infer_chunks_per_file(args, num_workers)
    all_tasks = build_converted_tasks(
        converted_parquets,
        output_dir,
        num_chunks_per_file=chunks_per_file,
        max_events_per_chunk=args.batch_size if args.chunks_per_file is None else None,
    )

    if args.merge_only:
        if args.task_num_shards > 1 and args.task_shard_index != 0:
            print(
                f"[converted-merge] skipped on shard {args.task_shard_index}; "
                "run merge once from shard 0 to avoid duplicate merge work",
                flush=True,
            )
            return
        merge_converted_chunk_outputs(all_tasks, delete_parts=args.delete_merged_parts)
        return

    runtime_train_config = prepare_runtime_train_config(
        train_config_path=args.train_config.resolve(),
        analysis_config_path=args.analysis_config.resolve(),
        evenet_config_path=args.evenet_config.resolve(),
        checkpoint_override=args.checkpoint.resolve() if args.checkpoint is not None else None,
    )


if __name__ == "__main__":
    main()

