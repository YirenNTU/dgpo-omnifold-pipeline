#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
import torch
import torch.multiprocessing as mp
import vector
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ML_PIPELINE_ROOT = REPO_ROOT / "ml_pipeline"
EVENET_ROOT = ML_PIPELINE_ROOT / "EveNet-Full"

if str(EVENET_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENET_ROOT))
if str(ML_PIPELINE_ROOT / "util") not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_ROOT / "util"))
if str(REPO_ROOT / "processor") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "processor"))

from build_evenet_input_from_parquet import (  # noqa: E402
    build_event_info_yaml,
    build_global_conditions,
    build_point_cloud,
    compute_event_totals,
    expand_input_files,
    merge_evenet_config,
    parse_config,
    read_yaml,
)
from evenet.control.global_config import global_config  # noqa: E402
from evenet.network.evenet_model import EveNetModel  # noqa: E402
from evenet.utilities.diffusion_sampler import DDIMSampler  # noqa: E402
from evenet.utilities.tool import safe_load_state  # noqa: E402
from evenet.dataset.preprocess import unflatten_dict  # noqa: E402
from evenet_parquet_common import (  # noqa: E402
    _build_visible_tau_layout,
    build_momentum4d,
    build_tau_targets,
    build_visible_tau_assumptions,
    part_energy_mask,
    reorder_tau_pair,
    stack_tau_pair,
    stack_tau_pair_mask,
)
from ml_pipeline_config import FeatureConfig, parse_evenet_config  # noqa: E402
from parquet_plot_common import infer_luminosity, sample_scale  # noqa: E402
from DataLoader import filter_event  # noqa: E402


vector.register_awkward()


DEFAULT_FLOAT = -99.0
DEFAULT_INT = -99
DEFAULT_CLASS_NAME = "unselected"


@dataclass(frozen=True)
class InferenceTask:
    sample_key: str
    sample_name: str
    is_data: bool
    is_signal: bool
    norm_factor: float
    lumi: float | None
    input_path: str
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone EveNet raw-parquet predictor with tautau selection, classification, and neutrino inference."
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=REPO_ROOT / "ml_pipeline" / "config" / "analysis.yaml",
        help="analysis.yaml used for raw parquet sample definitions and feature layout.",
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
        "--batch-size",
        type=int,
        default=2048,
        help="Inference batch size in selected tautau events.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPU workers. Falls back to CPU single-process if CUDA is unavailable.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional subset of sample keys from analysis.yaml.",
    )
    parser.add_argument(
        "--converted-parquet",
        nargs="*",
        default=None,
        help="Optional EveNet-converted parquet files (for example data.parquet evenet_test.parquet).",
    )
    parser.add_argument(
        "--shape-metadata",
        type=Path,
        default=None,
        help="Optional shape_metadata.json for converted parquet mode. Defaults to <parquet_dir>/shape_metadata.json.",
    )
    parser.add_argument(
        "--converted-split-fraction",
        type=float,
        default=None,
        help=(
            "Optional fraction of the original MC sample represented by each converted parquet "
            "(for example 0.5 for evenet_test.parquet). When set, converted-mode MC weights are "
            "rescaled by 1/fraction for data/MC comparison."
        ),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Optional override for TruthGeneration diffusion steps.",
    )
    return parser.parse_args()


def resolve_default_paths(config_path: Path, config_data: dict[str, Any]) -> dict[str, Any]:
    resolved = yaml.safe_load(yaml.safe_dump(config_data)) or {}
    for section, content in resolved.items():
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
        runtime_dir = Path(tempfile.mkdtemp(prefix="evenet_raw_predict_"))
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

    runtime_dir = Path(tempfile.mkdtemp(prefix="evenet_raw_predict_cfg_"))
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


def build_tasks(
    analysis_config_path: Path,
    output_dir: Path,
    sample_filter: set[str] | None = None,
) -> tuple[list[InferenceTask], dict, dict]:
    samples, subcategories, _ = parse_config(analysis_config_path)
    luminosity = infer_luminosity(samples, None)
    tasks: list[InferenceTask] = []

    for sample_key, sample in samples.items():
        if sample_filter and sample_key not in sample_filter and sample.name not in sample_filter:
            continue
        sample_output_dir = output_dir / sample.name
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        for input_path in expand_input_files(sample.input_files):
            input_file = Path(input_path)
            output_path = sample_output_dir / f"{input_file.stem}__evenet.parquet"
            tasks.append(
                InferenceTask(
                    sample_key=sample_key,
                    sample_name=sample.name,
                    is_data=sample.is_data,
                    is_signal=sample.is_signal,
                    norm_factor=sample.norm_factor,
                    lumi=sample.lumi,
                    input_path=str(input_file),
                    output_path=str(output_path),
                )
            )
    return tasks, samples, subcategories


def build_converted_tasks(converted_parquet: list[str], output_dir: Path) -> list[InferenceTask]:
    tasks: list[InferenceTask] = []
    for input_path in converted_parquet:
        input_file = Path(input_path).expanduser().resolve()
        output_path = output_dir / f"{input_file.stem}__evenet_pred.parquet"
        tasks.append(
            InferenceTask(
                sample_key=input_file.stem,
                sample_name=input_file.stem,
                is_data=False,
                is_signal=False,
                norm_factor=1.0,
                lumi=None,
                input_path=str(input_file),
                output_path=str(output_path),
            )
        )
    return tasks


def to_torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(value)
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        output[key] = tensor.to(device=device)
    return output


def event_index_array(num_events: int) -> ak.Array:
    return ak.Array(np.arange(num_events, dtype=np.int64))


def run_tautau_selection(events: ak.Array) -> tuple[ak.Array, dict[str, np.ndarray]]:
    num_events = len(events)
    events_for_filter = events
    if "evtNumber" not in events_for_filter.fields and "Event_evtNumber" in events_for_filter.fields:
        events_for_filter = ak.with_field(events_for_filter, events_for_filter["Event_evtNumber"], "evtNumber")
    events_for_filter = ak.with_field(events_for_filter, event_index_array(num_events), "__event_index")

    filtered_dict, _ = filter_event(
        events_for_filter,
        filter_log_dict={},
        input_channel="raw",
        skip_channels=["pion", "pilep"],
    )
    selected = filtered_dict.get("tautau", ak.Array([]))
    selected_indices = (
        ak.to_numpy(selected["__event_index"], allow_missing=False).astype(np.int64)
        if len(selected) > 0 and "__event_index" in selected.fields
        else np.array([], dtype=np.int64)
    )

    tautau_selected = np.zeros(num_events, dtype=bool)
    tautau_selected[selected_indices] = True

    scalar_defaults: dict[str, tuple[np.ndarray, str]] = {
        "nprong": (np.full(num_events, DEFAULT_INT, dtype=np.int32), "nprong"),
        "charged_E": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "charged_E"),
        "missing_px": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "missing_px"),
        "missing_py": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "missing_py"),
        "missing_pt": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "missing_pt"),
        "isolation_angle": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "isolation_angle"),
        "E_rad": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "E_rad"),
        "P_rad": (np.full(num_events, DEFAULT_FLOAT, dtype=np.float32), "P_rad"),
    }
    for _, (full_values, field_name) in scalar_defaults.items():
        if len(selected) > 0 and field_name in selected.fields:
            full_values[selected_indices] = ak.to_numpy(selected[field_name], allow_missing=False)

    selection_summary = {
        "tautau_selected": tautau_selected,
        **{key: values for key, (values, _) in scalar_defaults.items()},
    }
    return selected, selection_summary


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
    use_ema = bool(ema_cfg.get("enable", False)) and bool(ema_cfg.get("replace_model_after_load", False))
    state_dict = checkpoint.get("ema_state_dict") if use_ema and "ema_state_dict" in checkpoint else checkpoint["state_dict"]
    safe_load_state(model, state_dict, verbose=False)
    model.eval()

    class_names = list(global_config.event_info.class_label["EVENT"]["signal"][0])
    invisible_feature_names = list(global_config.event_info.invisible_feature_names)
    return {
        "model": model,
        "sampler": DDIMSampler(device=device),
        "class_names": class_names,
        "invisible_feature_names": invisible_feature_names,
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


def hemisphere_prong_charge_sums(events: ak.Array) -> tuple[ak.Array, ak.Array]:
    charge = events["Part_charge"]
    hemisphere = events["Part_hemisphere"]
    energy_valid = part_energy_mask(events)
    prong_a = (hemisphere == 1) & (charge != 0) & energy_valid
    prong_b = (hemisphere == -1) & (charge != 0) & energy_valid
    charge_a = ak.values_astype(ak.sum(charge[prong_a], axis=1), np.float32)
    charge_b = ak.values_astype(ak.sum(charge[prong_b], axis=1), np.float32)
    return charge_a, charge_b


def canonical_pair_to_hemisphere(pair_values, pair_mask, events: ak.Array):
    _, _, swap_mask = _build_visible_tau_layout(events)
    pair_tau_sign = reorder_tau_pair(pair_values, swap_mask)
    pair_tau_sign_mask = reorder_tau_pair(pair_mask, swap_mask)
    charge_a, charge_b = hemisphere_prong_charge_sums(events)

    a_is_tau_minus = (charge_a < 0) & (charge_b > 0)
    hemi_a = ak.where(a_is_tau_minus, pair_tau_sign[:, 0], pair_tau_sign[:, 1])
    hemi_b = ak.where(a_is_tau_minus, pair_tau_sign[:, 1], pair_tau_sign[:, 0])
    hemi_a_mask = ak.where(a_is_tau_minus, pair_tau_sign_mask[:, 0], pair_tau_sign_mask[:, 1])
    hemi_b_mask = ak.where(a_is_tau_minus, pair_tau_sign_mask[:, 1], pair_tau_sign_mask[:, 0])
    return (
        stack_tau_pair(hemi_a, hemi_b),
        stack_tau_pair_mask(hemi_a_mask, hemi_b_mask),
    )


def build_inference_batch(events: ak.Array, feature_config: FeatureConfig, invisible_dim: int) -> tuple[dict[str, np.ndarray], Any, Any]:
    max_particles = max(1, int(ak.max(ak.num(events["Part_pdgId"], axis=1))))
    x, x_mask, num_sequential_vectors, _ = build_point_cloud(events, max_particles, feature_config)
    conditions, conditions_mask, _ = build_global_conditions(events, feature_config)
    num_vectors = compute_event_totals(num_sequential_vectors, conditions_mask)
    tau_vis_prong_p4, tau_vis_prong_mask, _, _ = build_visible_tau_assumptions(events)

    batch = {
        "x": ak.to_numpy(x, allow_missing=False).astype(np.float32),
        "x_mask": ak.to_numpy(x_mask, allow_missing=False).astype(bool),
        "conditions": ak.to_numpy(conditions, allow_missing=False).astype(np.float32),
        "conditions_mask": ak.to_numpy(conditions_mask, allow_missing=False).astype(bool),
        "num_vectors": ak.to_numpy(num_vectors, allow_missing=False).astype(np.float32),
        "num_sequential_vectors": ak.to_numpy(num_sequential_vectors, allow_missing=False).astype(np.float32),
        "x_invisible": np.zeros((len(events), 2, invisible_dim), dtype=np.float32),
        "x_invisible_mask": ak.to_numpy(tau_vis_prong_mask, allow_missing=False).astype(bool),
    }
    return batch, tau_vis_prong_p4, tau_vis_prong_mask


def slice_torch_batch(batch: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value.index_select(0, indices) for key, value in batch.items()}


def undo_log_feature(name: str, values: np.ndarray) -> tuple[str, np.ndarray]:
    if name.startswith("log_"):
        return name[4:], np.expm1(values)
    return name, values


def predicted_features_to_p4(predicted: np.ndarray, feature_names: list[str]):
    values = {}
    for index, feature_name in enumerate(feature_names):
        resolved_name, resolved_values = undo_log_feature(feature_name, predicted[..., index])
        values[resolved_name] = resolved_values.astype(np.float32)

    if {"energy", "pt", "eta", "phi"}.issubset(values):
        pt = values["pt"]
        eta = values["eta"]
        phi = values["phi"]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = values["energy"]
        return build_momentum4d(px, py, pz, energy)

    if {"E", "px", "py", "pz"}.issubset(values):
        return build_momentum4d(values["px"], values["py"], values["pz"], values["E"])

    raise ValueError(
        "Unsupported invisible feature layout for p4 conversion. "
        f"Received {feature_names}."
    )


def component_defaults(num_events: int) -> dict[str, np.ndarray]:
    return {
        "lead_a_missing_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_missing_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_missing_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_missing_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_missing_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_missing_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_missing_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_missing_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_visible_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_visible_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_visible_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_a_visible_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_visible_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_visible_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_visible_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "lead_b_visible_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
    }


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


def load_converted_batch(parquet_path: Path, shape_metadata_path: Path) -> dict[str, np.ndarray]:
    flat_events = ak.from_parquet(parquet_path)
    with shape_metadata_path.open() as handle:
        shape_metadata = json.load(handle)

    flat_batch = {
        field: ak.to_numpy(flat_events[field], allow_missing=False)
        for field in flat_events.fields
    }
    return unflatten_dict(flat_batch, shape_metadata, drop_column_prefix=None)


def truth_component_defaults(num_events: int) -> dict[str, np.ndarray]:
    return {
        "truth_lead_a_missing_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_missing_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_missing_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_missing_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_missing_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_missing_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_missing_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_missing_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_visible_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_visible_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_visible_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_a_visible_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_visible_px": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_visible_py": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_visible_pz": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_lead_b_visible_E": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
    }


def build_truth_classification(
    events: ak.Array,
    task: InferenceTask,
    class_names: list[str],
    subcategories: dict,
    signal_class_names: set[str],
) -> dict[str, np.ndarray | ak.Array]:
    num_events = len(events)
    truth_class_index = np.full(num_events, -1, dtype=np.int64)
    truth_class_name = np.full(num_events, DEFAULT_CLASS_NAME, dtype=object)
    truth_is_signal = np.zeros(num_events, dtype=bool)
    truth_event_category = np.full(num_events, DEFAULT_INT, dtype=np.int32)

    if task.is_data:
        return {
            "evenet_truth_class_index": truth_class_index,
            "evenet_truth_class_name": ak.Array(truth_class_name.tolist()),
            "evenet_truth_is_signal": truth_is_signal,
            "evenet_truth_event_category": truth_event_category,
        }

    if "event_category" in events.fields:
        truth_event_category = ak.to_numpy(events["event_category"], allow_missing=False).astype(np.int32)

    if task.is_signal and "event_category" in events.fields:
        splits = subcategories.get(task.sample_key) or subcategories.get(task.sample_name) or []
        fallback_name = f"{task.sample_name}_others" if f"{task.sample_name}_others" in class_names else task.sample_name
        truth_class_name[:] = fallback_name
        category_values = ak.to_numpy(events["event_category"], allow_missing=False)
        for split in splits:
            mask = np.isin(category_values, np.asarray(split.categories, dtype=np.int64))
            truth_class_name[mask] = split.name
    else:
        truth_class_name[:] = task.sample_name

    class_to_index = {name: index for index, name in enumerate(class_names)}
    for name, index in class_to_index.items():
        truth_class_index[truth_class_name == name] = index
    truth_is_signal = np.isin(truth_class_name, np.asarray(sorted(signal_class_names), dtype=object))

    return {
        "evenet_truth_class_index": truth_class_index,
        "evenet_truth_class_name": ak.Array(truth_class_name.tolist()),
        "evenet_truth_is_signal": truth_is_signal,
        "evenet_truth_event_category": truth_event_category,
    }


def build_truth_selected_outputs(selected_events: ak.Array) -> dict[str, Any]:
    num_selected = len(selected_events)
    truth_outputs: dict[str, Any] = {
        "evenet_truth_valid": np.zeros(num_selected, dtype=bool),
        **truth_component_defaults(num_selected),
    }
    if num_selected == 0:
        return truth_outputs

    tau_vis_prong_p4, tau_vis_prong_mask, tau_vis_rho_p4, tau_vis_rho_mask = build_visible_tau_assumptions(selected_events)
    x_invisible_p4, x_invisible_mask, _, _, tau_vis_target_p4, tau_vis_target_mask = build_tau_targets(
        selected_events,
        tau_vis_prong_p4,
        tau_vis_prong_mask,
        tau_vis_rho_p4,
        tau_vis_rho_mask,
    )

    truth_visible_hemi_p4, truth_visible_hemi_mask = canonical_pair_to_hemisphere(
        tau_vis_target_p4,
        tau_vis_target_mask,
        selected_events,
    )
    truth_missing_hemi_p4, truth_missing_hemi_mask = canonical_pair_to_hemisphere(
        x_invisible_p4,
        x_invisible_mask,
        selected_events,
    )

    valid_truth = (
        ak.to_numpy(ak.all(truth_visible_hemi_mask, axis=1), allow_missing=False)
        & ak.to_numpy(ak.all(truth_missing_hemi_mask, axis=1), allow_missing=False)
    )
    truth_outputs["evenet_truth_valid"] = valid_truth

    for prefix, p4_values, mask_values in [
        ("truth_lead_a_visible", truth_visible_hemi_p4[:, 0], truth_visible_hemi_mask[:, 0]),
        ("truth_lead_b_visible", truth_visible_hemi_p4[:, 1], truth_visible_hemi_mask[:, 1]),
        ("truth_lead_a_missing", truth_missing_hemi_p4[:, 0], truth_missing_hemi_mask[:, 0]),
        ("truth_lead_b_missing", truth_missing_hemi_p4[:, 1], truth_missing_hemi_mask[:, 1]),
    ]:
        local_mask = ak.to_numpy(mask_values, allow_missing=False)
        truth_outputs[f"{prefix}_px"][local_mask] = ak.to_numpy(p4_values.px[local_mask], allow_missing=False).astype(np.float32)
        truth_outputs[f"{prefix}_py"][local_mask] = ak.to_numpy(p4_values.py[local_mask], allow_missing=False).astype(np.float32)
        truth_outputs[f"{prefix}_pz"][local_mask] = ak.to_numpy(p4_values.pz[local_mask], allow_missing=False).astype(np.float32)
        truth_outputs[f"{prefix}_E"][local_mask] = ak.to_numpy(p4_values.E[local_mask], allow_missing=False).astype(np.float32)

    return truth_outputs


def predict_selected_events(
    selected_events: ak.Array,
    model_bundle: dict[str, Any],
    signal_class_names: set[str],
    batch_size: int,
    num_steps: int | None,
    device: torch.device,
) -> dict[str, Any]:
    num_selected = len(selected_events)
    class_names = model_bundle["class_names"]
    invisible_feature_names = model_bundle["invisible_feature_names"]
    model: EveNetModel = model_bundle["model"]
    sampler: DDIMSampler = model_bundle["sampler"]
    feature_config: FeatureConfig = model_bundle["feature_config"]
    diffusion_steps = int(num_steps if num_steps is not None else model_bundle["num_steps"])

    pred_class_index = np.full(num_selected, -1, dtype=np.int64)
    pred_class_prob = np.full(num_selected, -1.0, dtype=np.float32)
    pred_class_name = np.full(num_selected, DEFAULT_CLASS_NAME, dtype=object)
    valid_signal_prediction = np.zeros(num_selected, dtype=bool)
    outputs = component_defaults(num_selected)

    for start in range(0, num_selected, batch_size):
        stop = min(start + batch_size, num_selected)
        event_batch = selected_events[start:stop]
        batch_np, tau_vis_prong_p4, tau_vis_prong_mask = build_inference_batch(
            event_batch,
            feature_config,
            invisible_dim=len(invisible_feature_names),
        )
        batch_torch = to_torch_batch(batch_np, device=device)

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
        batch_pred_name = np.array([class_names[index] for index in batch_pred_index_np], dtype=object)

        pred_class_index[start:stop] = batch_pred_index_np
        pred_class_prob[start:stop] = batch_pred_prob_np
        pred_class_name[start:stop] = batch_pred_name

        batch_signal_mask_np = np.array([name in signal_class_names for name in batch_pred_name], dtype=bool)
        if not np.any(batch_signal_mask_np):
            continue

        tau_vis_hemi_p4, tau_vis_hemi_mask = canonical_pair_to_hemisphere(
            tau_vis_prong_p4,
            tau_vis_prong_mask,
            event_batch,
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
        generated_np = generated.detach().cpu().numpy()
        missing_p4_canonical = predicted_features_to_p4(generated_np, invisible_feature_names)

        signal_events = event_batch[batch_signal_mask_np]
        tau_vis_hemi_signal = tau_vis_hemi_p4[batch_signal_mask_np]
        tau_vis_hemi_mask_signal = tau_vis_hemi_mask[batch_signal_mask_np]
        missing_hemi_p4, missing_hemi_mask = canonical_pair_to_hemisphere(
            missing_p4_canonical,
            ak.Array(np.ones((len(signal_events), 2), dtype=bool)),
            signal_events,
        )

        valid_batch_signal = (
            ak.to_numpy(ak.all(tau_vis_hemi_mask_signal, axis=1), allow_missing=False)
            & ak.to_numpy(ak.all(missing_hemi_mask, axis=1), allow_missing=False)
        )

        target_positions = np.nonzero(batch_signal_mask_np)[0]
        valid_positions = target_positions[valid_batch_signal]
        valid_signal_prediction[start + valid_positions] = True

        for prefix, p4_values, mask_values in [
            ("lead_a_visible", tau_vis_hemi_signal[:, 0], tau_vis_hemi_mask_signal[:, 0]),
            ("lead_b_visible", tau_vis_hemi_signal[:, 1], tau_vis_hemi_mask_signal[:, 1]),
            ("lead_a_missing", missing_hemi_p4[:, 0], missing_hemi_mask[:, 0]),
            ("lead_b_missing", missing_hemi_p4[:, 1], missing_hemi_mask[:, 1]),
        ]:
            local_mask = ak.to_numpy(mask_values, allow_missing=False) & valid_batch_signal
            local_positions = target_positions[local_mask]
            outputs[f"{prefix}_px"][start + local_positions] = ak.to_numpy(
                p4_values.px[local_mask], allow_missing=False
            ).astype(np.float32)
            outputs[f"{prefix}_py"][start + local_positions] = ak.to_numpy(
                p4_values.py[local_mask], allow_missing=False
            ).astype(np.float32)
            outputs[f"{prefix}_pz"][start + local_positions] = ak.to_numpy(
                p4_values.pz[local_mask], allow_missing=False
            ).astype(np.float32)
            outputs[f"{prefix}_E"][start + local_positions] = ak.to_numpy(
                p4_values.E[local_mask], allow_missing=False
            ).astype(np.float32)

    return {
        "pred_class_index": pred_class_index,
        "pred_class_prob": pred_class_prob,
        "pred_class_name": pred_class_name,
        "valid_signal_prediction": valid_signal_prediction,
        **outputs,
    }


def predict_converted_events(
    batch_np: dict[str, np.ndarray],
    model_bundle: dict[str, Any],
    signal_class_names: set[str],
    batch_size: int,
    num_steps: int | None,
    class_weight_map: dict[str, float] | None,
    converted_split_fraction: float | None,
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
    if class_weight_map is not None:
        for class_name, class_weight in class_weight_map.items():
            physics_weight[target_class_name == class_name] = np.float32(class_weight)
    if converted_split_fraction is not None:
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


def add_output_fields(events: ak.Array, output_columns: dict[str, Any]) -> ak.Array:
    result = events
    for field_name, values in output_columns.items():
        result = ak.with_field(result, values, field_name)
    return result


def augment_events_for_task(
    task: InferenceTask,
    model_bundle: dict[str, Any],
    batch_size: int,
    num_steps: int | None,
    signal_class_names: set[str],
    subcategories: dict,
    luminosity: float | None,
    device: torch.device,
) -> None:
    events = ak.from_parquet(task.input_path)
    selected_events, selection_summary = run_tautau_selection(events)
    num_events = len(events)
    selected_indices = (
        ak.to_numpy(selected_events["__event_index"], allow_missing=False).astype(np.int64)
        if len(selected_events) > 0 and "__event_index" in selected_events.fields
        else np.array([], dtype=np.int64)
    )

    mc_weight = sample_scale(task, events, luminosity)
    pred_class_index = np.full(num_events, -1, dtype=np.int64)
    pred_class_prob = np.full(num_events, -1.0, dtype=np.float32)
    pred_class_name = np.full(num_events, DEFAULT_CLASS_NAME, dtype=object)
    flags_valid = np.zeros(num_events, dtype=bool)
    predicted_signal = np.zeros(num_events, dtype=bool)
    mmc_likelihood = np.full(num_events, DEFAULT_FLOAT, dtype=np.float32)
    p4_outputs = component_defaults(num_events)
    truth_outputs = {
        "evenet_truth_valid": np.zeros(num_events, dtype=bool),
        **truth_component_defaults(num_events),
    }
    truth_classification = build_truth_classification(
        events=events,
        task=task,
        class_names=model_bundle["class_names"],
        subcategories=subcategories,
        signal_class_names=signal_class_names,
    )

    if len(selected_events) > 0:
        selected_prediction = predict_selected_events(
            selected_events=selected_events,
            model_bundle=model_bundle,
            signal_class_names=signal_class_names,
            batch_size=batch_size,
            num_steps=num_steps,
            device=device,
        )
        pred_class_index[selected_indices] = selected_prediction["pred_class_index"]
        pred_class_prob[selected_indices] = selected_prediction["pred_class_prob"]
        pred_class_name[selected_indices] = selected_prediction["pred_class_name"]
        flags_valid[selected_indices] = selected_prediction["valid_signal_prediction"]
        predicted_signal[selected_indices] = np.array(
            [name in signal_class_names for name in selected_prediction["pred_class_name"]],
            dtype=bool,
        )
        mmc_likelihood[selected_indices[flags_valid[selected_indices]]] = 0.0
        for key in p4_outputs:
            p4_outputs[key][selected_indices] = selected_prediction[key]
        if not task.is_data:
            selected_truth_outputs = build_truth_selected_outputs(selected_events)
            truth_outputs["evenet_truth_valid"][selected_indices] = selected_truth_outputs["evenet_truth_valid"]
            for key in truth_component_defaults(num_events):
                truth_outputs[key][selected_indices] = selected_truth_outputs[key]

    output_columns: dict[str, Any] = {
        "evenet_tautau_selected": selection_summary["tautau_selected"],
        "evenet_pred_class_index": pred_class_index,
        "evenet_pred_class_prob": pred_class_prob,
        "evenet_pred_class_name": ak.Array(pred_class_name.tolist()),
        "evenet_pred_is_signal": predicted_signal,
        "flags_valid": flags_valid,
        "mmc_likelihood": mmc_likelihood,
        "evenet_weight": np.full(num_events, mc_weight, dtype=np.float32),
        "nprong": selection_summary["nprong"],
        "charged_E": selection_summary["charged_E"],
        "missing_px": selection_summary["missing_px"],
        "missing_py": selection_summary["missing_py"],
        "missing_pt": selection_summary["missing_pt"],
        "isolation_angle": selection_summary["isolation_angle"],
        "E_rad": selection_summary["E_rad"],
        "P_rad": selection_summary["P_rad"],
        **p4_outputs,
        **truth_outputs,
        **truth_classification,
    }

    output_columns["lead_a_missing_p4"] = build_momentum4d(
        output_columns["lead_a_missing_px"],
        output_columns["lead_a_missing_py"],
        output_columns["lead_a_missing_pz"],
        output_columns["lead_a_missing_E"],
    )
    output_columns["lead_b_missing_p4"] = build_momentum4d(
        output_columns["lead_b_missing_px"],
        output_columns["lead_b_missing_py"],
        output_columns["lead_b_missing_pz"],
        output_columns["lead_b_missing_E"],
    )
    output_columns["lead_a_visible_p4"] = build_momentum4d(
        output_columns["lead_a_visible_px"],
        output_columns["lead_a_visible_py"],
        output_columns["lead_a_visible_pz"],
        output_columns["lead_a_visible_E"],
    )
    output_columns["lead_b_visible_p4"] = build_momentum4d(
        output_columns["lead_b_visible_px"],
        output_columns["lead_b_visible_py"],
        output_columns["lead_b_visible_pz"],
        output_columns["lead_b_visible_E"],
    )
    output_columns["truth_lead_a_missing_p4"] = build_momentum4d(
        output_columns["truth_lead_a_missing_px"],
        output_columns["truth_lead_a_missing_py"],
        output_columns["truth_lead_a_missing_pz"],
        output_columns["truth_lead_a_missing_E"],
    )
    output_columns["truth_lead_b_missing_p4"] = build_momentum4d(
        output_columns["truth_lead_b_missing_px"],
        output_columns["truth_lead_b_missing_py"],
        output_columns["truth_lead_b_missing_pz"],
        output_columns["truth_lead_b_missing_E"],
    )
    output_columns["truth_lead_a_visible_p4"] = build_momentum4d(
        output_columns["truth_lead_a_visible_px"],
        output_columns["truth_lead_a_visible_py"],
        output_columns["truth_lead_a_visible_pz"],
        output_columns["truth_lead_a_visible_E"],
    )
    output_columns["truth_lead_b_visible_p4"] = build_momentum4d(
        output_columns["truth_lead_b_visible_px"],
        output_columns["truth_lead_b_visible_py"],
        output_columns["truth_lead_b_visible_pz"],
        output_columns["truth_lead_b_visible_E"],
    )

    augmented_events = add_output_fields(events, output_columns)
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(augmented_events, output_path)
    print(
        f"[{task.sample_name}] wrote {output_path} "
        f"(selected={int(selection_summary['tautau_selected'].sum())}/{num_events}, valid={int(flags_valid.sum())})"
    )


def augment_converted_parquet_task(
    task: InferenceTask,
    model_bundle: dict[str, Any],
    batch_size: int,
    num_steps: int | None,
    signal_class_names: set[str],
    class_weight_map: dict[str, float] | None,
    converted_split_fraction: float | None,
    shape_metadata_path: Path | None,
    device: torch.device,
) -> None:
    input_path = Path(task.input_path).resolve()
    metadata_path = resolve_shape_metadata_path(input_path, shape_metadata_path)
    batch_np = load_converted_batch(input_path, metadata_path)
    outputs = predict_converted_events(
        batch_np=batch_np,
        model_bundle=model_bundle,
        signal_class_names=signal_class_names,
        batch_size=batch_size,
        num_steps=num_steps,
        class_weight_map=class_weight_map,
        converted_split_fraction=converted_split_fraction,
        device=device,
    )

    num_events = int(batch_np["x"].shape[0])
    output_columns: dict[str, Any] = {
        "event_index": np.arange(num_events, dtype=np.int64),
        "evenet_pred_class_index": outputs["pred_class_index"],
        "evenet_pred_class_prob": outputs["pred_class_prob"],
        "evenet_pred_class_name": ak.Array(outputs["pred_class_name"].tolist()),
        "flags_valid": outputs["valid_signal_prediction"],
        "evenet_truth_class_index": outputs["target_class_index"],
        "evenet_truth_class_name": ak.Array(outputs["target_class_name"].tolist()),
        "event_weight": outputs["event_weight"],
        "evenet_weight": outputs["physics_weight"],
    }

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
        f"(events={num_events}, valid={int(outputs['valid_signal_prediction'].sum())})"
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


def worker_main(
    rank: int,
    world_size: int,
    tasks: list[InferenceTask],
    runtime_train_config: str,
    checkpoint_path: str,
    batch_size: int,
    num_steps: int | None,
    signal_class_names: set[str],
    subcategories: dict,
    luminosity: float | None,
    converted_mode: bool = False,
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
        if converted_mode:
            augment_converted_parquet_task(
                task=task,
                model_bundle=model_bundle,
                batch_size=batch_size,
                num_steps=num_steps,
                signal_class_names=signal_class_names,
                class_weight_map=class_weight_map,
                converted_split_fraction=converted_split_fraction,
                shape_metadata_path=Path(shape_metadata_path) if shape_metadata_path is not None else None,
                device=device,
            )
        else:
            augment_events_for_task(
                task=task,
                model_bundle=model_bundle,
                batch_size=batch_size,
                num_steps=num_steps,
                signal_class_names=signal_class_names,
                subcategories=subcategories,
                luminosity=luminosity,
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
    with runtime_train_config.open("w") as handle:
        yaml.safe_dump(runtime_train_cfg_data, handle, sort_keys=False)

    checkpoint_path = resolve_checkpoint_path(runtime_train_config)
    sample_filter = set(args.samples) if args.samples else None
    converted_mode = bool(args.converted_parquet)
    analysis_config_data = read_yaml(args.analysis_config.resolve())
    _, _, analysis_feature_config = parse_config(args.analysis_config.resolve())
    merged_evenet_config = parse_evenet_config(
        merge_evenet_config(read_yaml(args.evenet_config.resolve()), analysis_config_data),
        analysis_feature_config,
    )
    loaded_class_names = runtime_class_names(Path(runtime_train_config))
    if converted_mode:
        tasks = build_converted_tasks(args.converted_parquet, output_dir)
        samples, subcategories = {}, {}
    else:
        tasks, samples, subcategories = build_tasks(args.analysis_config.resolve(), output_dir, sample_filter=sample_filter)
    if not tasks:
        raise ValueError("No input parquet tasks were found for the requested samples.")

    signal_class_names = signal_class_names_from_analysis(args.analysis_config.resolve())
    luminosity = infer_luminosity(samples, None) if samples else None
    class_weight_map = (
        build_converted_class_weights(
            analysis_config_path=args.analysis_config.resolve(),
            evenet_config=merged_evenet_config,
            class_names=loaded_class_names,
        )
        if converted_mode
        else None
    )

    meta_path = output_dir / "prediction_metadata.yaml"
    with meta_path.open("w") as handle:
        yaml.safe_dump(
            {
                "mode": "converted" if converted_mode else "raw",
                "analysis_config": str(args.analysis_config.resolve()),
                "train_config": str(args.train_config.resolve()),
                "runtime_train_config": str(runtime_train_config.resolve()),
                "checkpoint": str(checkpoint_path),
                "num_tasks": len(tasks),
                "signal_classes": sorted(signal_class_names),
                "converted_parquet": [str(Path(path).resolve()) for path in (args.converted_parquet or [])],
                "shape_metadata": str(args.shape_metadata.resolve()) if args.shape_metadata is not None else None,
                "class_weight_map": class_weight_map,
                "converted_split_fraction": args.converted_split_fraction,
            },
            handle,
            sort_keys=False,
        )

    use_cuda = torch.cuda.is_available() and args.num_gpus > 0
    num_workers = min(args.num_gpus, torch.cuda.device_count()) if use_cuda else 1
    if num_workers <= 1:
        worker_main(
            rank=0,
            world_size=1 if use_cuda else 0,
            tasks=tasks,
            runtime_train_config=str(runtime_train_config),
            checkpoint_path=str(checkpoint_path),
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            signal_class_names=signal_class_names,
            subcategories=subcategories,
            luminosity=luminosity,
            converted_mode=converted_mode,
            shape_metadata_path=str(args.shape_metadata.resolve()) if args.shape_metadata is not None else None,
            class_weight_map=class_weight_map,
            converted_split_fraction=args.converted_split_fraction,
        )
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
                subcategories,
                luminosity,
                converted_mode,
                str(args.shape_metadata.resolve()) if args.shape_metadata is not None else None,
                class_weight_map,
                args.converted_split_fraction,
            ),
            nprocs=num_workers,
            join=True,
        )


if __name__ == "__main__":
    main()
