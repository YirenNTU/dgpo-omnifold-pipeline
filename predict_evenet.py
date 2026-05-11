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

PROGRESS_PRINT_EVERY = 10


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

    runtime_dir = Path(tempfile.mkdtemp(prefix="evenet_predict_cfg_"))
    runtime_train_cfg = runtime_dir / "train_runtime.yaml"
    with runtime_train_cfg.open("w") as handle:
        yaml.safe_dump(train_cfg, handle, sort_keys=False)
    return runtime_train_cfg


def select_task_shard(tasks: list[InferenceTask], task_num_shards: int, task_shard_index: int) -> list[InferenceTask]:
    """
    inputs:
      tasks: list[InferenceTask], all event chunks to run.
      task_num_shards: int, number of independent jobs.
      task_shard_index: int, current independent job index.
    outputs:
      shard_tasks: list[InferenceTask], tasks assigned to this job.
    goal:
      Let several scheduler jobs process one prediction campaign without
      duplicating output files.
    """
    if task_num_shards < 1:
        raise ValueError(f"--task-num-shards must be >= 1, got {task_num_shards}.")
    if not (0 <= task_shard_index < task_num_shards):
        raise ValueError(
            f"--task-shard-index must be in [0, {task_num_shards}), got {task_shard_index}."
        )
    if task_num_shards == 1:
        return tasks
    return tasks[task_shard_index::task_num_shards]

def load_flat_parquet_dict(
    parquet_path: Path,
    event_start: int | None = None,
    event_stop: int | None = None,
) -> dict[str, np.ndarray]:
    table = pq.read_table(parquet_path)
    if event_start is not None or event_stop is not None:
        start = 0 if event_start is None else int(event_start)
        stop = table.num_rows if event_stop is None else int(event_stop)
        table = table.slice(start, max(0, stop - start))
    pydict = table.to_pydict()
    return {key: np.asarray(value) for key, value in pydict.items()}

def ensure_converted_batch_fields(
    batch_np: dict[str, np.ndarray],
    invisible_dim: int,
    default_num_tokens: int = 2,
) -> dict[str, np.ndarray]:
    output = dict(batch_np)
    reference_key = next((key for key in ["x", "conditions", "num_vectors"] if key in output), None)
    if reference_key is None:
        raise ValueError("Converted parquet is missing reference tensors such as x/conditions/num_vectors.")
    num_events = int(output[reference_key].shape[0])

    if "x_invisible" not in output:
        output["x_invisible"] = np.zeros((num_events, default_num_tokens, invisible_dim), dtype=np.float32)
    if "x_invisible_mask" not in output:
        output["x_invisible_mask"] = np.ones((num_events, output["x_invisible"].shape[1]), dtype=bool)

    if "classification" not in output:
        output["classification"] = np.full(num_events, -1, dtype=np.int64)
    if "event_weight" not in output:
        raise ValueError(
            "Converted parquet must contain event_weight. "
            "Prediction weighting no longer falls back to central_weight or class weights."
        )

    return output


def fill_invisible_feature_outputs(
    output_columns: dict[str, np.ndarray],
    prefix: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    feature_names: list[str],
    target_indices: np.ndarray,
) -> None:
    name_order = ["a", "b"]
    for slot in range(values.shape[1]):
        name = name_order[slot]
        output_columns[f"{prefix}_{name}_valid"][target_indices] = valid_mask[:, slot]
        for feature_index, feature_name in enumerate(feature_names):
            slot_values = values[:, slot, feature_index].astype(np.float32)
            masked_values = np.where(valid_mask[:, slot], slot_values, 0).astype(np.float32)
            output_columns[f"{prefix}_{name}_{feature_name}"][target_indices] = masked_values
            if feature_name.startswith("log_"):
                linear_name = feature_name[4:]
                linear_values = np.where(valid_mask[:, slot], np.expm1(slot_values), 0).astype(np.float32)
                output_columns[f"{prefix}_{name}_{linear_name}"][target_indices] = linear_values

def to_torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        output[key] = tensor.to(device=device)
    return output

def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_output_columns_to_events(
    input_path: Path,
    output_path: Path,
    output_columns: dict[str, np.ndarray],
    event_start: int | None = None,
    event_stop: int | None = None,
) -> None:
    events = ak.from_parquet(input_path)

    if event_start is not None or event_stop is not None:
        start = 0 if event_start is None else int(event_start)
        stop = len(events) if event_stop is None else int(event_stop)
        events = events[start:stop]

    for name, values in output_columns.items():
        events = ak.with_field(events, values, name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(events, output_path)


def extract_invisible_prediction(
    model_outputs: Any,
    invisible_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred: shape (B, N_invisible, N_features)
      mask: shape (B, N_invisible)

    This accepts either:
      - direct tensor / array
      - {"x_invisible": tensor}
      - {"pred_invisible": tensor}
      - {"neutrinos": {"predict": {"pt": ..., "eta": ..., "phi": ...}}}
    """
    if isinstance(model_outputs, torch.Tensor):
        pred = tensor_to_numpy(model_outputs)
        mask = np.ones(pred.shape[:2], dtype=bool)
        return pred.astype(np.float32), mask

    if not isinstance(model_outputs, dict):
        raise TypeError(f"Cannot extract invisible prediction from {type(model_outputs)!r}")

    for key in ("x_invisible", "pred_invisible", "predict_invisible", "sample"):
        if key in model_outputs:
            pred = tensor_to_numpy(model_outputs[key])
            mask = tensor_to_numpy(
                model_outputs.get("x_invisible_mask", np.ones(pred.shape[:2], dtype=bool))
            ).astype(bool)
            return pred.astype(np.float32), mask



def predict_converted_events(
    batch_np: dict[str, np.ndarray],
    model_bundle: dict[str, Any],
    batch_size: int,
    num_steps: int | None,
    converted_split_fraction: float | None,
    skip_classification: bool,
    device: torch.device,
    ):

    invisible_feature_names = model_bundle["invisible_feature_names"]
    diffusion_model = model_bundle["diffusion_model"]
    classification_model = model_bundle["classification_model"]
    sampler = model_bundle["sampler"]

    batch_np = ensure_converted_batch_fields(batch_np, invisible_dim=len(invisible_feature_names))
    num_events = int(batch_np["x"].shape[0])
    num_invisibles = int(batch_np["x_invisible"].shape[1])

    output_columns: dict[str, np.ndarray] = {
        "evenet_class_index": np.full(num_events, -1, dtype=np.int64),
        "evenet_class_prob": np.full(num_events, -1.0, dtype=np.float32),
    }

    valid_signal_prediction = np.zeros(num_events, dtype=bool)
    pred_invisible = np.zeros(
        (num_events, num_invisibles, len(invisible_feature_names)),
        dtype=np.float32
    )
    pred_invisible_mask = np.zeros(
        (num_events, num_invisibles),
        dtype=bool
    )
    total_batches = max(1, math.ceil(num_events / batch_size))
    for start in range(0, num_events, batch_size):
        stop = min(start + batch_size, num_events)
        batch_id = start // batch_size + 1
        if batch_id == 1 or batch_id == total_batches or batch_id % PROGRESS_PRINT_EVERY == 0:
            print(
                f"[converted-predict] classification batch {batch_id}/{total_batches} "
                f"events={start}:{stop}",
                flush=True,
            )
        batch_slice = {
            key: value[start:stop]
            for key, value in batch_np.items()
            if isinstance(value, np.ndarray)
        }
        batch_torch = to_torch_batch(batch_slice, device=device)
        with torch.no_grad():
            if skip_classification:
                batch_class_index = batch_np["classification"]
                batch_class_prob = np.ones(stop - start, dtype=np.float32)
            else:
                cls_outputs = classification_model.shared_step(
                    batch=batch_torch,
                    batch_size = stop - start,
                    train_parameters=None,
                    schedules=[
                        ("generation", False),
                        ("neutrion_generation", False),
                        ("deterministic", True)
                    ],
                )
                classification_outputs = cls_outputs.get("classification")
                if not classification_outputs:
                    raise RuntimeError(
                        "Classification model produced no classification outputs. "
                        "Use a train config/checkpoint with options.Training.Components.Classification.include=true, "
                        "or pass --classification-checkpoint for split classification/diffusion prediction."
                    )
                logits = next(iter(classification_outputs.values()))
                probs = torch.softmax(logits, dim=-1)
                pred_index = torch.argmax(probs, dim=-1)
                pred_prob = torch.gather(probs, -1, pred_index.unsqueeze(-1)).squeeze(-1)

                batch_class_index = pred_index.detach().cpu().numpy().astype(np.int64)
                batch_class_prob = pred_prob.detach().cpu().numpy().astype(np.float32)

            output_columns["evenet_class_index"][start:stop] = batch_class_index
            output_columns["evenet_class_prob"][start:stop] = batch_class_prob

            # Make generation use the class you chose.
            batch_slice["classification"] = batch_class_index
            batch_torch = to_torch_batch(batch_slice, device=device)

            # --------------------------------------------------
            # 2. Invisible / neutrino generation
            # --------------------------------------------------
            gen_outputs = sampler.sample(
                model=diffusion_model,
                batch=batch_torch,
                batch_size=batch_size,
                num_steps=num_steps,
            )
            batch_pred_invisible, batch_pred_mask = extract_invisible_prediction(
                gen_outputs,
                invisible_feature_names=invisible_feature_names,
            )
            pred_invisible[start:stop] = batch_pred_invisible.astype(np.float32)
            pred_invisible_mask[start:stop] = batch_pred_mask.astype(bool)

    fill_invisible_feature_outputs(
        output_columns,
        prefix="evenet_invisible",
        values=pred_invisible,
        valid_mask=pred_invisible_mask,
        feature_names=invisible_feature_names,
        target_indices=np.arange(num_events, dtype=np.int64),
    )

    return output_columns

def load_converted_batch(
    parquet_path: Path,
    shape_metadata_path: Path,
    event_start: int | None = None,
    event_stop: int | None = None,
) -> dict[str, np.ndarray]:
    with shape_metadata_path.open() as handle:
        shape_metadata = json.load(handle)

    flat_batch = load_flat_parquet_dict(parquet_path, event_start=event_start, event_stop=event_stop)
    return unflatten_dict(flat_batch, shape_metadata, drop_column_prefix=None)


def augment_converted_parquet_task(
    task: InferenceTask,
    model_bundle: dict[str, Any],
    batch_size: int,
    num_steps: int | None,
    converted_split_fraction: float | None,
    shape_metadata_path: Path | None,
    skip_classification: bool,
    device: torch.device,
) -> None:
    input_path = Path(task.input_path).resolve()
    output_path = Path(task.output_path).resolve()

    if shape_metadata_path is None:
        metadata_path = input_path.parent / "shape_metadata.json"
    else:
        metadata_path = Path(shape_metadata_path).resolve()

    print(f"[converted-task] loading {input_path}", flush=True)

    batch_np = load_converted_batch(
        parquet_path=input_path,
        shape_metadata_path=metadata_path,
        event_start=task.event_start,
        event_stop=task.event_stop,
    )

    num_events = int(batch_np["x"].shape[0])
    print(
        f"[converted-task] loaded {input_path.name} "
        f"events={num_events} "
        f"range=[{task.event_start if task.event_start is not None else 0}, "
        f"{task.event_stop if task.event_stop is not None else num_events})",
        flush=True,
    )

    output_columns = predict_converted_events(
        batch_np=batch_np,
        model_bundle=model_bundle,
        batch_size=batch_size,
        num_steps=num_steps,
        converted_split_fraction=converted_split_fraction,
        skip_classification=skip_classification,
        device=device,
    )

    add_output_columns_to_events(
        input_path=input_path,
        output_path=output_path,
        output_columns=output_columns,
        event_start=task.event_start,
        event_stop=task.event_stop,
    )

    print(f"[converted-task] wrote {output_path}", flush=True)

def load_checkpoint_into_model(
    model: EveNetModel,
    checkpoint_path: Path,
    *,
    use_ema: bool,
    device: torch.device,
) -> EveNetModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if use_ema and isinstance(checkpoint, dict) and "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    safe_load_state(model, state_dict)
    model.to(device)
    model.eval()
    return model


def load_model_bundle(
    runtime_train_config: Path,
    classification_checkpoint: Path,
    diffusion_checkpoint: Path,
    *,
    diffusion_use_ema: bool,
    device: torch.device,
) -> dict[str, Any]:
    # Keep this consistent with the way EveNetModel is normally initialized in your repo.
    global_config.load_yaml(runtime_train_config, current_dir=REPO_ROOT)
    normalization_dict = torch.load(global_config.options.Dataset.normalization_file, map_location=device)
    classification_model = EveNetModel(
        config=global_config,
        device=device,
        classification=True,
        normalization_dict=normalization_dict,
    )
    diffusion_model = EveNetModel(
        config=global_config,
        device=device,
        classification=False,
        neutrino_generation=True,
        normalization_dict=normalization_dict,
    )

    classification_model = load_checkpoint_into_model(
        classification_model,
        classification_checkpoint,
        use_ema=False,
        device=device,
    )

    diffusion_model = load_checkpoint_into_model(
        diffusion_model,
        diffusion_checkpoint,
        use_ema=diffusion_use_ema,
        device=device,
    )

    sampler = DDIMSampler(diffusion_model)

    return {
        "classification_model": classification_model,
        "diffusion_model": diffusion_model,
        "sampler": sampler,
    }



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
    invisible_features = tuple(str(key) for key in analysis_config["Normalization"].get("Invisible"))
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

    runtime_train_cfg_data = read_yaml(runtime_train_config)
    runtime_train_cfg_data["_analysis_config_path"] = str(args.analysis_config.resolve())
    runtime_train_cfg_data.setdefault("options", {}).setdefault("prediction", {})["disable_ema"] = bool(args.disable_ema)
    with runtime_train_config.open("w") as handle:
        yaml.safe_dump(runtime_train_cfg_data, handle, sort_keys=False)
    print(runtime_train_cfg_data)

    diffusion_use_ema = not args.disable_ema
    if args.classification_checkpoint is not None:
        classification_check_point = args.classification_checkpoint
    else:
        classification_check_point = args.checkpoint.resolve()

    if args.diffusion_checkpoint is not None:
        diffusion_check_point = args.diffusion_checkpoint
    else:
        diffusion_check_point = args.checkpoint.resolve()

    tasks = select_task_shard(all_tasks, args.task_num_shards, args.task_shard_index)
    if not tasks:
        print("[converted-predict] no tasks assigned to this shard", flush=True)
        return

    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"[converted-predict] using device={device}", flush=True)

    model_bundle = load_model_bundle(
        runtime_train_config=runtime_train_cfg,
        classification_checkpoint=classification_check_point,
        diffusion_checkpoint=diffusion_check_point,
        diffusion_use_ema=diffusion_use_ema,
        device=device,
    )
    model_bundle["invisible_feature_names"] = invisible_features

    for task in tasks:
        augment_converted_parquet_task(
            task=task,
            model_bundle=model_bundle,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            converted_split_fraction=converted_split_fraction,
            shape_metadata_path=args.shape_metadata,
            skip_classification=args.use_truth_classification,
            device=device,
        )

    if not args.skip_merge:
        if args.task_num_shards > 1 and args.task_shard_index != 0:
            print(
                f"[converted-merge] skipped on shard {args.task_shard_index}; "
                "run merge once from shard 0 after all shards finish",
                flush=True,
            )
            return

        merge_converted_chunk_outputs(
            all_tasks,
            delete_parts=args.delete_merged_parts,
        )
if __name__ == "__main__":
    main()

