#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from build_evenet_input_from_parquet import (
    build_event_info_yaml,
    expand_input_files,
    merge_evenet_config,
    parse_config,
    read_yaml,
)
from evenet.control.global_config import global_config
from evenet.dataset.preprocess import unflatten_dict
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.diffusion_sampler import DDIMSampler
from evenet.utilities.tool import safe_load_state
from ml_pipeline_config import FeatureConfig, parse_evenet_config
from parquet_plot_common import infer_luminosity


vector.register_awkward()


DEFAULT_FLOAT = -99.0
DEFAULT_CLASS_NAME = "unselected"
PROGRESS_PRINT_EVERY = 10


@dataclass(frozen=True)
class InferenceTask:
    sample_name: str
    input_path: str
    output_path: str
    event_start: int | None = None
    event_stop: int | None = None
    final_output_path: str | None = None


def parse_args() -> argparse.Namespace:
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
        default=REPO_ROOT / "ml_pipeline" / "config" / "train.yaml",
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
        help="Optional checkpoint override. If omitted, use options.Training.model_checkpoint_load_path from train config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where augmented parquet outputs are written.",
    )
    parser.add_argument(
        "--converted-parquet",
        nargs="+",
        required=True,
        help="EveNet-converted parquet files to run prediction on.",
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
        default=None,
        help="Optional override for TruthGeneration diffusion steps.",
    )
    parser.add_argument(
        "--disable-ema",
        action="store_true",
        help="If set, load checkpoint state_dict instead of ema_state_dict.",
    )
    parser.add_argument(
        "--unweighted-output",
        action="store_true",
        help="If set, write evenet_weight=1 for all events instead of physics-normalized MC weights.",
    )
    return parser.parse_args()


def resolve_default_paths(config_path: Path, config_data: dict[str, Any]) -> dict[str, Any]:
    resolved = yaml.safe_load(yaml.safe_dump(config_data)) or {}
    for _, content in resolved.items():
        if isinstance(content, dict) and "default" in content:
            content["default"] = str((config_path.parent / content["default"]).resolve())
    return resolved


def build_training_class_labels(
    samples: dict,
    subcategories: dict,
    evenet_config,
) -> list[str]:
    class_labels: list[str] = []
    for sample_key, sample in samples.items():
        if sample.is_data:
            continue
        splits = subcategories.get(sample_key) or subcategories.get(sample.name)
        if splits:
            class_labels.extend(split.name for split in splits)
            remainder_name = f"{sample.name}_others"
            if remainder_name in evenet_config.process_topologies:
                class_labels.append(remainder_name)
        else:
            class_labels.append(sample.name)
    return class_labels


def build_class_to_sample_map(samples: dict, subcategories: dict, evenet_config) -> dict[str, Any]:
    class_to_sample: dict[str, Any] = {}
    for sample_key, sample in samples.items():
        if sample.is_data:
            continue
        splits = subcategories.get(sample_key) or subcategories.get(sample.name)
        if splits:
            for split in splits:
                class_to_sample[split.name] = sample
            remainder_name = f"{sample.name}_others"
            if remainder_name in evenet_config.process_topologies:
                class_to_sample[remainder_name] = sample
        else:
            class_to_sample[sample.name] = sample
    return class_to_sample


def sample_initial_total_num_events(sample) -> int:
    parquet_files = expand_input_files(sample.input_files)
    values: list[int] = []
    for path in parquet_files:
        events = ak.from_parquet(path, columns=["initial_total_num_events"])
        if len(events) == 0 or "initial_total_num_events" not in events.fields:
            continue
        values.append(int(ak.to_numpy(events["initial_total_num_events"][:1], allow_missing=False)[0]))
    if values:
        return max(values)
    return 0


def build_converted_class_weights(
    analysis_config_path: Path,
    evenet_config,
    class_names: list[str],
) -> dict[str, float]:
    samples, subcategories, _ = parse_config(analysis_config_path)
    luminosity = infer_luminosity(samples, None)
    class_to_sample = build_class_to_sample_map(samples, subcategories, evenet_config)
    output: dict[str, float] = {}
    for class_name in class_names:
        sample = class_to_sample.get(class_name)
        if sample is None or sample.is_data or luminosity is None:
            output[class_name] = 1.0
            continue
        initial_total_num_events = sample_initial_total_num_events(sample)
        if initial_total_num_events <= 0:
            output[class_name] = 1.0
            continue
        output[class_name] = float(sample.norm_factor) / float(initial_total_num_events) * float(luminosity)
    return output


def prepare_runtime_train_config(
    train_config_path: Path,
    analysis_config_path: Path,
    evenet_config_path: Path,
    checkpoint_override: Path | None,
) -> Path:
    train_cfg = resolve_default_paths(train_config_path, read_yaml(train_config_path))
    analysis_cfg = read_yaml(analysis_config_path)
    samples, subcategories, feature_config = parse_config(analysis_config_path)
    evenet_schema_cfg = read_yaml(evenet_config_path)
    evenet_config = parse_evenet_config(
        merge_evenet_config(evenet_schema_cfg, analysis_cfg),
        feature_config,
    )

    event_info_cfg = train_cfg.get("event_info", {})
    existing_event_info = event_info_cfg.get("default")
    if existing_event_info is not None:
        existing_event_info = str((train_config_path.parent / existing_event_info).resolve()) \
            if not Path(existing_event_info).is_absolute() else str(Path(existing_event_info).resolve())

    if existing_event_info is None or not Path(existing_event_info).exists():
        metadata = {
            "point_cloud_features": list(feature_config.raw_sequential_fields),
            "global_features": list(feature_config.global_fields),
            "invisible_features": list(evenet_config.invisible_features),
            "class_labels": build_training_class_labels(samples, subcategories, evenet_config),
        }
        event_info_payload = build_event_info_yaml(metadata, feature_config, evenet_config)
        runtime_dir = Path(tempfile.mkdtemp(prefix="evenet_predict_"))
        runtime_event_info = runtime_dir / "event_info.yaml"
        with runtime_event_info.open("w") as handle:
            yaml.safe_dump(event_info_payload, handle, sort_keys=False)
        train_cfg["event_info"] = {"default": str(runtime_event_info)}
    else:
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


def resolve_checkpoint_path(runtime_train_config: Path) -> Path:
    train_cfg = read_yaml(runtime_train_config)
    training_cfg = train_cfg.get("options", {}).get("Training", {})
    checkpoint_value = training_cfg.get("model_checkpoint_load_path")
    if checkpoint_value is None:
        raise ValueError("No checkpoint configured. Pass --checkpoint or set options.Training.model_checkpoint_load_path.")

    checkpoint_path = Path(checkpoint_value).expanduser()
    if checkpoint_path.is_dir():
        candidates = sorted(checkpoint_path.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError(f"No .ckpt files found in checkpoint directory: {checkpoint_path}")
        return candidates[-1].resolve()
    return checkpoint_path.resolve()


def build_converted_tasks(converted_parquet: list[str], output_dir: Path, num_chunks_per_file: int) -> list[InferenceTask]:
    tasks: list[InferenceTask] = []
    for input_path in converted_parquet:
        input_file = Path(input_path).expanduser().resolve()
        final_output_path = output_dir / f"{input_file.stem}__evenet_pred.parquet"
        num_events = parquet_num_rows(input_file)
        for shard_index, (start, stop) in enumerate(split_event_ranges(num_events, num_chunks_per_file)):
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


def parquet_num_rows(parquet_path: Path) -> int:
    return int(pq.ParquetFile(parquet_path).metadata.num_rows)


def split_event_ranges(num_events: int, num_chunks: int) -> list[tuple[int, int]]:
    if num_chunks <= 1 or num_events <= 0:
        return [(0, num_events)]
    edges = np.linspace(0, num_events, num_chunks + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(num_chunks) if int(edges[i + 1]) > int(edges[i])]


def to_torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        output[key] = tensor.to(device=device)
    return output


def load_model_bundle(runtime_train_config: Path, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    global_config.load_yaml(runtime_train_config, current_dir=REPO_ROOT)
    normalization_dict = torch.load(global_config.options.Dataset.normalization_file, map_location=device)

    components = global_config.options.Training.Components
    model = EveNetModel(
        config=global_config,
        device=device,
        classification=components.Classification.include,
        regression=components.Regression.include,
        global_generation=components.GlobalGeneration.include,
        point_cloud_generation=components.ReconGeneration.include,
        neutrino_generation=components.TruthGeneration.include,
        assignment=components.Assignment.include,
        segmentation=components.Segmentation.include,
        normalization_dict=normalization_dict,
    )
    model = model.to(device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ema_cfg = global_config.options.Training.EMA
    prediction_cfg = global_config.options.get("prediction", {})
    use_ema = (
        bool(ema_cfg.get("enable", False))
        and bool(ema_cfg.get("replace_model_after_load", False))
        and not bool(prediction_cfg.get("disable_ema", False))
    )
    state_dict = checkpoint.get("ema_state_dict") if use_ema and "ema_state_dict" in checkpoint else checkpoint["state_dict"]
    safe_load_state(model, state_dict, verbose=False)
    model.eval()

    return {
        "model": model,
        "sampler": DDIMSampler(device=device),
        "class_names": list(global_config.event_info.class_label["EVENT"]["signal"][0]),
        "invisible_feature_names": list(global_config.event_info.invisible_feature_names),
        "feature_config": infer_feature_config(runtime_train_config),
        "num_steps": int(global_config.options.Training.Components.TruthGeneration.diffusion_steps),
    }


def infer_feature_config(runtime_train_config: Path) -> FeatureConfig:
    train_cfg = read_yaml(runtime_train_config)
    analysis_cfg = None
    if "_analysis_config_path" in train_cfg:
        analysis_cfg = Path(train_cfg["_analysis_config_path"])
    if analysis_cfg is None:
        raise ValueError("Runtime train config is missing _analysis_config_path.")
    _, _, feature_config = parse_config(analysis_cfg)
    return feature_config


def runtime_class_names(runtime_train_config: Path) -> list[str]:
    global_config.load_yaml(runtime_train_config, current_dir=REPO_ROOT)
    return list(global_config.event_info.class_label["EVENT"]["signal"][0])


def validate_feature_alignment(feature_config: FeatureConfig, class_names: list[str]) -> None:
    raw_seq = list(global_config.event_info.raw_sequential_feature_names)
    if tuple(raw_seq) != tuple(feature_config.raw_sequential_fields):
        raise ValueError(
            "Raw sequential features from analysis.yaml do not match the loaded event_info.\n"
            f"analysis={feature_config.raw_sequential_fields}\n"
            f"event_info={tuple(raw_seq)}"
        )
    global_features = [
        feature.name
        for feature in global_config.event_info.input_features["Conditions"]
    ]
    if tuple(global_features) != tuple(feature_config.global_fields):
        raise ValueError(
            "Global condition features from analysis.yaml do not match the loaded event_info.\n"
            f"analysis={feature_config.global_fields}\n"
            f"event_info={tuple(global_features)}"
        )
    if len(class_names) == 0:
        raise ValueError("Loaded classification head has no class labels.")


def slice_torch_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in batch.items()}


def invisible_feature_defaults(
    num_events: int,
    num_slots: int,
    feature_names: list[str],
    prefix: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for slot in range(num_slots):
        output[f"{prefix}_slot{slot}_valid"] = np.zeros(num_events, dtype=bool)
        for feature_name in feature_names:
            output[f"{prefix}_slot{slot}_{feature_name}"] = np.full(num_events, DEFAULT_FLOAT, dtype=np.float32)
            if feature_name.startswith("log_"):
                output[f"{prefix}_slot{slot}_{feature_name[4:]}"] = np.full(num_events, DEFAULT_FLOAT, dtype=np.float32)
    return output


def fill_invisible_feature_outputs(
    output_columns: dict[str, np.ndarray],
    prefix: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    feature_names: list[str],
    target_indices: np.ndarray,
) -> None:
    for slot in range(values.shape[1]):
        output_columns[f"{prefix}_slot{slot}_valid"][target_indices] = valid_mask[:, slot]
        for feature_index, feature_name in enumerate(feature_names):
            slot_values = values[:, slot, feature_index].astype(np.float32)
            masked_values = np.where(valid_mask[:, slot], slot_values, DEFAULT_FLOAT).astype(np.float32)
            output_columns[f"{prefix}_slot{slot}_{feature_name}"][target_indices] = masked_values
            if feature_name.startswith("log_"):
                linear_name = feature_name[4:]
                linear_values = np.where(valid_mask[:, slot], np.expm1(slot_values), DEFAULT_FLOAT).astype(np.float32)
                output_columns[f"{prefix}_slot{slot}_{linear_name}"][target_indices] = linear_values


def visible_feature_defaults(
    num_events: int,
    num_slots: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for slot in range(num_slots):
        output[f"{prefix}_slot{slot}_valid"] = np.zeros(num_events, dtype=bool)
        for feature_name in ["energy", "pt", "eta", "phi"]:
            output[f"{prefix}_slot{slot}_{feature_name}"] = np.full(num_events, DEFAULT_FLOAT, dtype=np.float32)
    return output


def fill_visible_feature_outputs(
    output_columns: dict[str, np.ndarray],
    prefix: str,
    values: np.ndarray,
    valid_mask: np.ndarray,
    target_indices: np.ndarray,
) -> None:
    for slot in range(values.shape[1]):
        output_columns[f"{prefix}_slot{slot}_valid"][target_indices] = valid_mask[:, slot]
        output_columns[f"{prefix}_slot{slot}_energy"][target_indices] = np.where(
            valid_mask[:, slot], values[:, slot, 0], DEFAULT_FLOAT
        ).astype(np.float32)
        output_columns[f"{prefix}_slot{slot}_pt"][target_indices] = np.where(
            valid_mask[:, slot], values[:, slot, 1], DEFAULT_FLOAT
        ).astype(np.float32)
        output_columns[f"{prefix}_slot{slot}_eta"][target_indices] = np.where(
            valid_mask[:, slot], values[:, slot, 2], DEFAULT_FLOAT
        ).astype(np.float32)
        output_columns[f"{prefix}_slot{slot}_phi"][target_indices] = np.where(
            valid_mask[:, slot], values[:, slot, 3], DEFAULT_FLOAT
        ).astype(np.float32)


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
        output["event_weight"] = np.ones(num_events, dtype=np.float32)

    return output


def resolve_shape_metadata_path(input_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    candidate = input_path.parent / "shape_metadata.json"
    if not candidate.exists():
        raise FileNotFoundError(
            f"shape_metadata.json not found next to converted parquet {input_path}. "
            "Pass --shape-metadata explicitly."
        )
    return candidate.resolve()


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


def predict_converted_events(
    batch_np: dict[str, np.ndarray],
    model_bundle: dict[str, Any],
    signal_class_names: set[str],
    batch_size: int,
    num_steps: int | None,
    class_weight_map: dict[str, float] | None,
    converted_split_fraction: float | None,
    use_weighted_output: bool,
    device: torch.device,
) -> dict[str, Any]:
    class_names = model_bundle["class_names"]
    invisible_feature_names = model_bundle["invisible_feature_names"]
    model: EveNetModel = model_bundle["model"]
    sampler: DDIMSampler = model_bundle["sampler"]
    diffusion_steps = int(num_steps if num_steps is not None else model_bundle["num_steps"])

    batch_np = ensure_converted_batch_fields(batch_np, invisible_dim=len(invisible_feature_names))
    num_events = int(batch_np["x"].shape[0])
    num_slots = int(batch_np["x_invisible"].shape[1])

    pred_class_index = np.full(num_events, -1, dtype=np.int64)
    pred_class_prob = np.full(num_events, -1.0, dtype=np.float32)
    pred_class_name = np.full(num_events, DEFAULT_CLASS_NAME, dtype=object)
    valid_signal_prediction = np.zeros(num_events, dtype=bool)
    pred_invisible = invisible_feature_defaults(num_events, num_slots, invisible_feature_names, prefix="pred_invisible")
    target_invisible = invisible_feature_defaults(num_events, num_slots, invisible_feature_names, prefix="target_invisible")
    total_batches = max(1, math.ceil(num_events / batch_size))

    target_mask = batch_np["x_invisible_mask"].astype(bool)
    fill_invisible_feature_outputs(
        target_invisible,
        prefix="target_invisible",
        values=batch_np["x_invisible"].astype(np.float32),
        valid_mask=target_mask,
        feature_names=invisible_feature_names,
        target_indices=np.arange(num_events, dtype=np.int64),
    )

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
            cls_outputs = model.shared_step(
                batch=batch_torch,
                batch_size=stop - start,
                train_parameters=None,
                schedules=[
                    ("generation", False),
                    ("neutrino_generation", False),
                    ("deterministic", True),
                ],
            )
            logits = next(iter(cls_outputs["classification"].values()))
            probs = torch.softmax(logits, dim=-1)
            batch_pred_index = torch.argmax(probs, dim=-1)
            batch_pred_prob = torch.gather(probs, -1, batch_pred_index.unsqueeze(-1)).squeeze(-1)

        batch_pred_index_np = batch_pred_index.detach().cpu().numpy().astype(np.int64)
        batch_pred_prob_np = batch_pred_prob.detach().cpu().numpy().astype(np.float32)
        batch_pred_name = np.array(
            [class_names[index] if 0 <= index < len(class_names) else DEFAULT_CLASS_NAME for index in batch_pred_index_np],
            dtype=object,
        )

        pred_class_index[start:stop] = batch_pred_index_np
        pred_class_prob[start:stop] = batch_pred_prob_np
        pred_class_name[start:stop] = batch_pred_name

        batch_signal_mask_np = np.array([name in signal_class_names for name in batch_pred_name], dtype=bool)
        if not np.any(batch_signal_mask_np):
            continue

        print(
            f"[converted-predict] neutrino sampling batch {batch_id}/{total_batches} "
            f"signal_events={int(batch_signal_mask_np.sum())}",
            flush=True,
        )
        signal_indices = torch.from_numpy(np.nonzero(batch_signal_mask_np)[0].astype(np.int64)).to(device=device)
        signal_batch = slice_torch_batch(batch_torch, signal_indices)
        signal_batch["classification"] = batch_pred_index.index_select(0, signal_indices).to(torch.int64)

        generated = sampler.sample(
            data_shape=signal_batch["x_invisible"].shape,
            pred_fn=lambda noise_x, time: model.predict_diffusion_vector(
                noise_x=noise_x,
                cond_x=signal_batch,
                time=time,
                mode="neutrino",
                noise_mask=signal_batch["x_invisible_mask"].unsqueeze(-1),
            ),
            normalize_fn=model.invisible_normalizer,
            eta=1.0,
            num_steps=diffusion_steps,
            use_tqdm=False,
            process_name="NeutrinoPredict",
            remove_padding=(getattr(model, "invisible_padding", 0) > 0),
            noise_mask=signal_batch["x_invisible_mask"].unsqueeze(-1),
        )
        generated_np = generated.detach().cpu().numpy().astype(np.float32)
        signal_target_mask = batch_slice["x_invisible_mask"][batch_signal_mask_np].astype(bool)
        target_positions = np.nonzero(batch_signal_mask_np)[0]
        valid_signal_prediction[start + target_positions] = np.all(signal_target_mask, axis=1)
        fill_invisible_feature_outputs(
            pred_invisible,
            prefix="pred_invisible",
            values=generated_np,
            valid_mask=signal_target_mask,
            feature_names=invisible_feature_names,
            target_indices=start + target_positions,
        )

    target_class_index = batch_np["classification"].astype(np.int64)
    target_class_name = np.full(num_events, DEFAULT_CLASS_NAME, dtype=object)
    valid_target_class = (target_class_index >= 0) & (target_class_index < len(class_names))
    target_class_name[valid_target_class] = np.asarray(class_names, dtype=object)[target_class_index[valid_target_class]]
    physics_weight = np.ones(num_events, dtype=np.float32)
    if use_weighted_output and class_weight_map is not None:
        for class_name, class_weight in class_weight_map.items():
            physics_weight[target_class_name == class_name] = np.float32(class_weight)
    if use_weighted_output and converted_split_fraction is not None:
        if not (0.0 < converted_split_fraction <= 1.0):
            raise ValueError(
                f"converted_split_fraction must be in (0, 1], got {converted_split_fraction}."
            )
        mc_like_mask = valid_target_class
        physics_weight[mc_like_mask] = physics_weight[mc_like_mask] / np.float32(converted_split_fraction)

    return {
        "pred_class_index": pred_class_index,
        "pred_class_prob": pred_class_prob,
        "pred_class_name": pred_class_name,
        "valid_signal_prediction": valid_signal_prediction,
        "target_class_index": target_class_index,
        "target_class_name": target_class_name,
        "event_weight": batch_np["event_weight"].astype(np.float32),
        "physics_weight": physics_weight,
        **pred_invisible,
        **target_invisible,
    }


def augment_converted_parquet_task(
    task: InferenceTask,
    model_bundle: dict[str, Any],
    batch_size: int,
    num_steps: int | None,
    signal_class_names: set[str],
    class_weight_map: dict[str, float] | None,
    converted_split_fraction: float | None,
    use_weighted_output: bool,
    shape_metadata_path: Path | None,
    device: torch.device,
) -> None:
    input_path = Path(task.input_path).resolve()
    print(f"[converted-task] loading {input_path}", flush=True)
    metadata_path = resolve_shape_metadata_path(input_path, shape_metadata_path)
    batch_np = load_converted_batch(
        input_path,
        metadata_path,
        event_start=task.event_start,
        event_stop=task.event_stop,
    )
    print(
        f"[converted-task] loaded {input_path.name} "
        f"events={int(batch_np['x'].shape[0])} "
        f"range=[{task.event_start if task.event_start is not None else 0},"
        f"{task.event_stop if task.event_stop is not None else int(batch_np['x'].shape[0])}) "
        f"shape_metadata={metadata_path}",
        flush=True,
    )
    outputs = predict_converted_events(
        batch_np=batch_np,
        model_bundle=model_bundle,
        signal_class_names=signal_class_names,
        batch_size=batch_size,
        num_steps=num_steps,
        class_weight_map=class_weight_map,
        converted_split_fraction=converted_split_fraction,
        use_weighted_output=use_weighted_output,
        device=device,
    )

    num_events = int(batch_np["x"].shape[0])
    event_start = 0 if task.event_start is None else int(task.event_start)
    output_columns: dict[str, Any] = {
        "event_index": np.arange(event_start, event_start + num_events, dtype=np.int64),
        "evenet_pred_class_index": outputs["pred_class_index"],
        "evenet_pred_class_prob": outputs["pred_class_prob"],
        "evenet_pred_class_name": ak.Array(outputs["pred_class_name"].tolist()),
        "flags_valid": outputs["valid_signal_prediction"],
        "evenet_truth_class_index": outputs["target_class_index"],
        "evenet_truth_class_name": ak.Array(outputs["target_class_name"].tolist()),
        "event_weight": outputs["event_weight"],
        "evenet_weight": outputs["physics_weight"],
    }

    if "tau_vis_prong" in batch_np:
        tau_vis_prong = batch_np["tau_vis_prong"].astype(np.float32)
        tau_vis_prong_mask = batch_np.get(
            "tau_vis_prong_mask",
            np.ones(tau_vis_prong.shape[:2], dtype=bool),
        ).astype(bool)
        tau_vis_output = visible_feature_defaults(num_events, tau_vis_prong.shape[1], prefix="tau_vis_prong")
        fill_visible_feature_outputs(
            tau_vis_output,
            prefix="tau_vis_prong",
            values=tau_vis_prong,
            valid_mask=tau_vis_prong_mask,
            target_indices=np.arange(num_events, dtype=np.int64),
        )
        output_columns.update(tau_vis_output)

    for key, value in outputs.items():
        if key in {
            "pred_class_index",
            "pred_class_prob",
            "pred_class_name",
            "valid_signal_prediction",
            "target_class_index",
            "target_class_name",
            "event_weight",
            "physics_weight",
        }:
            continue
        output_columns[key] = value

    augmented_events = ak.Array(output_columns)
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(augmented_events, output_path)
    print(
        f"[converted:{task.sample_name}] wrote {output_path} "
        f"(events={num_events}, range=[{event_start},{event_start + num_events}), "
        f"valid={int(outputs['valid_signal_prediction'].sum())})",
        flush=True,
    )


def signal_class_names_from_analysis(analysis_config_path: Path) -> set[str]:
    samples, subcategories, _ = parse_config(analysis_config_path)
    signal_names: set[str] = set()
    for sample_key, sample in samples.items():
        if not sample.is_signal:
            continue
        splits = subcategories.get(sample_key) or subcategories.get(sample.name)
        if splits:
            signal_names.update(split.name for split in splits)
        else:
            signal_names.add(sample.name)
    return signal_names


def merge_converted_chunk_outputs(tasks: list[InferenceTask]) -> None:
    grouped_tasks: dict[str, list[InferenceTask]] = {}
    for task in tasks:
        final_output_path = task.final_output_path or task.output_path
        grouped_tasks.setdefault(final_output_path, []).append(task)

    for final_output_path, group in grouped_tasks.items():
        if len(group) == 1 and group[0].output_path == final_output_path:
            continue

        chunk_paths = [Path(task.output_path) for task in sorted(group, key=lambda item: item.event_start or 0)]
        arrays = [ak.from_parquet(chunk_path) for chunk_path in chunk_paths]
        merged = ak.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        order = np.argsort(ak.to_numpy(merged["event_index"], allow_missing=False))
        merged = merged[order]

        final_path = Path(final_output_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        ak.to_parquet(merged, final_path)
        print(f"[converted-merge] wrote {final_path} from {len(chunk_paths)} chunk(s)", flush=True)

        for chunk_path in chunk_paths:
            if chunk_path != final_path and chunk_path.exists():
                chunk_path.unlink()


def worker_main(
    rank: int,
    world_size: int,
    tasks: list[InferenceTask],
    runtime_train_config: str,
    checkpoint_path: str,
    batch_size: int,
    num_steps: int | None,
    signal_class_names: set[str],
    use_weighted_output: bool,
    shape_metadata_path: str | None = None,
    class_weight_map: dict[str, float] | None = None,
    converted_split_fraction: float | None = None,
) -> None:
    use_cuda = torch.cuda.is_available() and world_size > 0
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(rank)

    model_bundle = load_model_bundle(Path(runtime_train_config), Path(checkpoint_path), device=device)
    validate_feature_alignment(model_bundle["feature_config"], model_bundle["class_names"])

    worker_tasks = tasks[rank::world_size] if world_size > 0 else tasks
    for task in worker_tasks:
        augment_converted_parquet_task(
            task=task,
            model_bundle=model_bundle,
            batch_size=batch_size,
            num_steps=num_steps,
            signal_class_names=signal_class_names,
            class_weight_map=class_weight_map,
            converted_split_fraction=converted_split_fraction,
            use_weighted_output=use_weighted_output,
            shape_metadata_path=Path(shape_metadata_path) if shape_metadata_path is not None else None,
            device=device,
        )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    checkpoint_path = resolve_checkpoint_path(runtime_train_config)
    use_cuda = torch.cuda.is_available() and args.num_gpus > 0
    num_workers = min(args.num_gpus, torch.cuda.device_count()) if use_cuda else 1
    analysis_config_data = read_yaml(args.analysis_config.resolve())
    _, _, analysis_feature_config = parse_config(args.analysis_config.resolve())
    merged_evenet_config = parse_evenet_config(
        merge_evenet_config(read_yaml(args.evenet_config.resolve()), analysis_config_data),
        analysis_feature_config,
    )
    loaded_class_names = runtime_class_names(Path(runtime_train_config))
    tasks = build_converted_tasks(args.converted_parquet, output_dir, num_chunks_per_file=max(1, num_workers))
    if not tasks:
        raise ValueError("No converted parquet tasks were found.")

    signal_class_names = signal_class_names_from_analysis(args.analysis_config.resolve())
    class_weight_map = build_converted_class_weights(
        analysis_config_path=args.analysis_config.resolve(),
        evenet_config=merged_evenet_config,
        class_names=loaded_class_names,
    )

    meta_path = output_dir / "prediction_metadata.yaml"
    with meta_path.open("w") as handle:
        yaml.safe_dump(
            {
                "mode": "converted",
                "analysis_config": str(args.analysis_config.resolve()),
                "train_config": str(args.train_config.resolve()),
                "runtime_train_config": str(runtime_train_config.resolve()),
                "checkpoint": str(checkpoint_path),
                "num_tasks": len(tasks),
                "signal_classes": sorted(signal_class_names),
                "converted_parquet": [str(Path(path).resolve()) for path in args.converted_parquet],
                "shape_metadata": str(args.shape_metadata.resolve()) if args.shape_metadata is not None else None,
                "class_weight_map": class_weight_map,
                "converted_split_fraction": args.converted_split_fraction,
                "disable_ema": bool(args.disable_ema),
                "use_weighted_output": not bool(args.unweighted_output),
            },
            handle,
            sort_keys=False,
        )

    if num_workers <= 1:
        worker_main(
            rank=0,
            world_size=1 if use_cuda else 0,
            tasks=tasks,
            runtime_train_config=str(runtime_train_config),
            checkpoint_path=str(checkpoint_path),
            batch_size=args.batch_size  ,
            num_steps=args.num_steps,
            signal_class_names=signal_class_names,
            use_weighted_output=not args.unweighted_output,
            shape_metadata_path=str(args.shape_metadata.resolve()) if args.shape_metadata is not None else None,
            class_weight_map=class_weight_map,
            converted_split_fraction=args.converted_split_fraction,
        )
        merge_converted_chunk_outputs(tasks)
        return

    mp.spawn(
        worker_main,
        args=(
            num_workers,
            tasks,
            str(runtime_train_config),
            str(checkpoint_path),
            args.batch_size,
            args.num_steps,
            signal_class_names,
            not args.unweighted_output,
            str(args.shape_metadata.resolve()) if args.shape_metadata is not None else None,
            class_weight_map,
            args.converted_split_fraction,
        ),
        nprocs=num_workers,
        join=True,
    )
    merge_converted_chunk_outputs(tasks)


if __name__ == "__main__":
    main()
