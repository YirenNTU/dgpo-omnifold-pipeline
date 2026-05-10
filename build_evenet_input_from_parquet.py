#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import vector
import yaml

from common import (
    read_yaml, process_latex_label, build_classification_lookup,
    event_preselection_mask, build_input_particle_mask,
    rebuild_vector, to_numpy,
    classification_targets_for_sample, build_momentum4d,
    pad_and_flatten_part_feature, make_json_serializable
)
from generate_event_info_yaml import parse_feature_config

vector.register_awkward()
PART_MOMENTUM_SOURCE_FIELDS = (
    "Part_fourMomentum_fCoordinates_fX",
    "Part_fourMomentum_fCoordinates_fY",
    "Part_fourMomentum_fCoordinates_fZ",
    "Part_fourMomentum_fCoordinates_fT",
)

@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    is_data: bool
    is_signal: bool
    files: tuple[str, ...]
    file_source: str
    norm_factor: float | None
    lumi: float | None
    total_initial_num_events: float | None
    plot_label: str
    predict_neutrino: dict[str, Any]



def parse_samples(config: dict[str, Any], selected_keys: list[str] | None = None) -> list[Sample]:
    selected = set(selected_keys or [])
    samples: list[Sample] = []
    neutrino_cfg = config["NeutrinoPrediction"]
    for key, sample_cfg in config["Samples"].items():
        if selected and key not in selected:
            continue
        input_files = sample_cfg["input_files"]

        samples.append(
            Sample(
                key=key,
                name=str(sample_cfg.get("name", key)),
                is_data=bool(sample_cfg.get("is_data", False)),
                is_signal=bool(sample_cfg.get("is_signal", False)),
                files=tuple(str(item) for item in input_files),
                file_source=sample_cfg["input_files"],
                norm_factor=float(sample_cfg.get("norm_factor", 1)),
                lumi=float(sample_cfg.get("lumi", 1)),
                total_initial_num_events=None,
                plot_label=process_latex_label(str(sample_cfg.get("name", key))),
                predict_neutrino=neutrino_cfg[key] if key in neutrino_cfg else [],
            )
        )
    return samples


def nominal_event_weight(sample: Sample, luminosity: float | None, events: ak.Array) -> np.ndarray:
    if sample.is_data:
        return np.ones(len(events), dtype=np.float32)
    if luminosity is not None and sample.norm_factor is not None and sample.total_initial_num_events is not None:
        scale = np.float32(luminosity * sample.norm_factor / sample.total_initial_num_events)
        return np.full(len(events), scale, dtype=np.float32)
    if "central_weight" in events.fields:
        return to_numpy(events["central_weight"], np.float32)
    raise ValueError(
        f"Sample '{sample.key}' needs (luminosity, norm_factor, total_initial_num_events) or central_weight."
    )

def read_file_initial_total_num_events(path: str) -> float | None:
    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=1, columns=["initial_total_num_events"]):
        values = ak.to_numpy(ak.from_arrow(record_batch)["initial_total_num_events"], allow_missing=False)
        if len(values) == 0:
            continue
        return float(values[0])
    return None

def attach_sample_total_intial_events(samples: list[Sample]) -> list[Sample]:
    resolved_samples: list[Sample] = []
    for sample in samples:
        if sample.is_data:
            resolved_samples.append(replace(sample))
            continue
        total = 0.0
        for path in sample.files:
            file_total = read_file_initial_total_num_events(path)
            if file_total is None:
                raise ValueError(
                    f"Sample '{sample.key}' file '{path}' is missing initial_total_num_events."
                )
            total += float(file_total)
        resolved_samples.append(replace(sample, total_initial_num_events=total))
    return resolved_samples

def infer_luminosity(samples: list[Sample]) -> float | None:
    data_lumis = [sample.lumi for sample in samples if sample.is_data and sample.lumi is not None]
    return sum(data_lumis) if data_lumis else None


def required_columns(schema_names: set[str], feature_config, sample: Sample) -> list[str]:
    columns = {
        "lead_a_visible_p4",
        "lead_b_visible_p4",
        "nprong",
        "initial_total_num_events",
        "weight",
        "central_weight",
        "Part_pdgId",
    }
    if "evtNumber" in schema_names:
        columns.add("evtNumber")
    if "Event_evtNumber" in schema_names:
        columns.add("Event_evtNumber")
    if sample.is_signal:
        columns.update(
            {
                "truth_tau_a_p4",
                "truth_tau_b_p4",
                "event_category",
                "truth_QI_region",
                "analyzing_power",
                "analyzing_power_a",
                "analyzing_power_b",
                "mmc_likelihood",
                "theta_cm",
                "flags_valid"
            }
        )
        columns.update(name for name in schema_names if name.startswith("truth_"))
        columns.update(name for name in schema_names if name.startswith("cos_"))

    for feature_name in feature_config.all_sequential_fields:
        if feature_name in {"Part_energy", "Part_pt", "Part_eta", "Part_phi"}:
            columns.update(PART_MOMENTUM_SOURCE_FIELDS)
        else:
            columns.add(feature_name)
    for field_name in feature_config.global_fields:
        if field_name in schema_names:
            columns.add(field_name)
        elif field_name.startswith("missing_") or field_name in {"missing_pt", "missing_E", "missing_energy"}:
            columns.add("missing_p4")
    columns.update(name for name in schema_names if name.endswith("_cut"))
    columns.update(name for name in schema_names if name.endswith("_p4"))
    return sorted(name for name in columns if name in schema_names)

def build_point_cloud(
    events: ak.Array,
    max_particles: int,
    feature_config,
    remove_neutral_non_photon: bool,
) -> tuple[ak.Array, ak.Array, np.ndarray, ak.Array, list[str]]:
    input_part_mask = build_input_particle_mask(events, remove_neutral_non_photon)
    part_p4 = build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"][input_part_mask],
        events["Part_fourMomentum_fCoordinates_fY"][input_part_mask],
        events["Part_fourMomentum_fCoordinates_fZ"][input_part_mask],
        events["Part_fourMomentum_fCoordinates_fT"][input_part_mask],
    )
    available_momentum_features = {
        "Part_energy": part_p4.E,
        "Part_pt": part_p4.pt,
        "Part_eta": part_p4.eta,
        "Part_phi": part_p4.phi,
    }
    features = []
    feature_names: list[str] = []
    for feature_name in feature_config.raw_sequential_fields:
        if feature_name in available_momentum_features:
            values = available_momentum_features[feature_name]
        else:
            values = events[feature_name][input_part_mask]
        features.append(pad_and_flatten_part_feature(values, max_particles))
        feature_names.append(feature_name)
    x = ak.concatenate(features, axis=2)
    num_particles = ak.to_numpy(ak.num(events["Part_pdgId"][input_part_mask], axis=1), allow_missing=False).astype(np.float32)
    x_mask = ak.Array(np.arange(max_particles)[None, :] < num_particles[:, None])
    return x, x_mask, num_particles, input_part_mask, feature_names

def _p4_component(p4: ak.Array, component: str) -> ak.Array:
    fields = set(getattr(p4, "fields", []))
    if component == "px":
        return p4["px"] if "px" in fields else p4["x"]
    if component == "py":
        return p4["py"] if "py" in fields else p4["y"]
    if component == "pz":
        return p4["pz"] if "pz" in fields else p4["z"]
    if component in {"E", "energy"}:
        return p4["E"] if "E" in fields else p4["t"]
    raise ValueError(f"Unsupported p4 component '{component}'.")


def resolve_global_feature(events: ak.Array, field_name: str) -> ak.Array:
    if field_name in events.fields:
        return events[field_name]
    field_four_momentum = "_".join(field_name.split("_")[:-1]) + "_p4"
    if field_four_momentum in events.fields:
        four_momentum = events[field_four_momentum]
        if "px" in field_name:
            return _p4_component(four_momentum, "px")
        if "py" in field_name:
            return _p4_component(four_momentum, "py")
        if "pz" in field_name:
            return _p4_component(four_momentum, "pz")
        if "pt" in field_name:
            px = _p4_component(four_momentum, "px")
            py = _p4_component(four_momentum, "py")
            return np.sqrt(px*px + py*py)
        if "E" in field_name or "energy" in field_name:
            return _p4_component(four_momentum, "E")

    # if "missing_p4" in events.fields:
    #     missing_p4 = events["missing_p4"]
    #     if field_name == "missing_px":
    #         return _p4_component(missing_p4, "px")
    #     if field_name == "missing_py":
    #         return _p4_component(missing_p4, "py")
    #     if field_name == "missing_pz":
    #         return _p4_component(missing_p4, "pz")
    #     if field_name in {"missing_E", "missing_energy"}:
    #         return _p4_component(missing_p4, "E")
    #     if field_name == "missing_pt":
    #         px = _p4_component(missing_p4, "px")
    #         py = _p4_component(missing_p4, "py")
    #         return np.sqrt(px * px + py * py)
    preview = ", ".join(events.fields[:25])
    suffix = " ..." if len(events.fields) > 25 else ""
    raise KeyError(f"Global feature '{field_name}' is missing. Available fields include: {preview}{suffix}")

def flatten_global_feature(values: ak.Array) -> ak.Array:
    filled = ak.fill_none(values, 0)
    return ak.values_astype(filled, np.float32)[..., np.newaxis]

def build_global_conditions(events: ak.Array, feature_config) -> tuple[ak.Array, ak.Array, list[str]]:
    features = []
    feature_names: list[str] = []
    for field_name in feature_config.global_fields:
        features.append(flatten_global_feature(resolve_global_feature(events, field_name)))
        feature_names.append(field_name)
    conditions = ak.concatenate(features, axis=1)
    conditions_mask = ak.Array(np.ones((len(events), 1), dtype=bool))
    return conditions, conditions_mask, feature_names

def source_event_key_array(events: ak.Array, fallback_index: np.ndarray) -> np.ndarray:
    for key in ("evtNumber", "Event_evtNumber"):
        if key in events.fields:
            return to_numpy(events[key], np.int64)
    return fallback_index.astype(np.int64, copy=False)

def features_from_p4_local(p4: ak.Array, feature_names: tuple[str, ...]) -> np.ndarray:
    components: list[np.ndarray] = []
    for feature_name in feature_names:
        if feature_name in {"energy", "E"}:
            values = ak.to_numpy(p4.E, allow_missing=False)
        elif feature_name == "px":
            values = ak.to_numpy(p4.px, allow_missing=False)
        elif feature_name == "py":
            values = ak.to_numpy(p4.py, allow_missing=False)
        elif feature_name == "pz":
            values = ak.to_numpy(p4.pz, allow_missing=False)
        elif feature_name == "pt":
            values = ak.to_numpy(p4.pt, allow_missing=False)
        elif feature_name == "eta":
            values = ak.to_numpy(p4.eta, allow_missing=False)
        elif feature_name == "phi":
            values = ak.to_numpy(p4.phi, allow_missing=False)
        elif feature_name == "mass":
            values = ak.to_numpy(p4.mass, allow_missing=False)
        else:
            raise ValueError(f"Unsupported four-vector feature '{feature_name}'.")
        components.append(values.astype(np.float32))
    return np.stack(components, axis=-1).astype(np.float32)

def build_output_events(
    selected_events: ak.Array,
    sample: Sample,
    luminosity: float | None,
    classification_lookup,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
    remove_neutral_non_photon: bool,
    source_sample_index: int,
    source_file_index: int,
    source_row_index: np.ndarray,
    ) -> ak.Array:

    event_weight = nominal_event_weight(sample, luminosity, selected_events)
    event_categories = (
        to_numpy(selected_events["event_category"], np.int64)
        if "event_category" in selected_events.fields
        else None
    )
    classification_indices, classification_names = classification_targets_for_sample(
        sample_key=sample.key,
        sample_name=sample.name,
        is_data=sample.is_data,
        num_rows=len(selected_events),
        event_categories=event_categories,
        lookup=classification_lookup,
    )

    predict_neutrino = ak.from_numpy(np.array([str(name) in sample.predict_neutrino for name in classification_names]))


    if "central_weight" in selected_events.fields:
        central_weight = to_numpy(selected_events["central_weight"], np.float32)
    else:
        central_weight = event_weight.copy()

    x, x_mask, num_sequential_vectors, input_part_mask, point_cloud_feature_names = build_point_cloud(
        selected_events,
        max_particles,
        feature_config,
        remove_neutral_non_photon=remove_neutral_non_photon,
    )
    conditions, conditions_mask, codition_names = build_global_conditions(selected_events, feature_config)
    num_vectors = num_sequential_vectors + ak.to_numpy(ak.values_astype(conditions_mask[:, 0], np.float32), allow_missing=False)

    visible_a = rebuild_vector(selected_events["lead_a_visible_p4"])
    visible_b = rebuild_vector(selected_events["lead_b_visible_p4"])

    if sample.is_signal:
        truth_tau_a = rebuild_vector(selected_events["truth_tau_a_p4"])
        truth_tau_b = rebuild_vector(selected_events["truth_tau_b_p4"])
        truth_visible_a = rebuild_vector(selected_events["truth_visible_a_p4"])
        truth_visible_b = rebuild_vector(selected_events["truth_visible_b_p4"])

    else:
        truth_visible_a = visible_a * 0
        truth_visible_b = visible_b * 0
        truth_tau_a = visible_a * 0 # meaningless stuff
        truth_tau_b = visible_b * 0 # meaningless stuff

    invisible_a = truth_tau_a - visible_a
    invisible_b = truth_tau_b - visible_b

    invisible_a_for_delta = features_from_p4_local(invisible_a, invisible_features)
    invisible_b_for_delta = features_from_p4_local(invisible_b, invisible_features)
    visible_a_for_delta =  features_from_p4_local(visible_a, invisible_features)
    visible_b_for_delta =  features_from_p4_local(visible_b, invisible_features)

    delta_invisible_a = invisible_a_for_delta - visible_a_for_delta
    delta_invisible_b = invisible_b_for_delta - visible_b_for_delta

    delta_invisible_input = ak.concatenate([delta_invisible_a[:, np.newaxis], delta_invisible_b[:, np.newaxis]], axis=1)
    delta_invisible_mask = ak.ones_like(invisible_a.px) * predict_neutrino[:, np.newaxis]


    num_invisible_raw = np.full(len(selected_events), 2, dtype=np.int64)
    num_invisible_valid = np.sum(ak.to_numpy(delta_invisible_mask, allow_missing=False).astype(np.int64), axis=1).astype(np.int64)

    fields: dict[str, Any] = {
        "sample_key": ak.Array([sample.key] * len(selected_events)),
        "sample_name": ak.Array([sample.name] * len(selected_events)),
        "sample_is_data": np.full(len(selected_events), sample.is_data, dtype=bool),
        "sample_is_signal": np.full(len(selected_events), sample.is_signal, dtype=bool),
        "source_sample_index": np.full(len(selected_events), source_sample_index, dtype=np.int64),
        "source_file_index": np.full(len(selected_events), source_file_index, dtype=np.int32),
        "source_event_index": source_row_index.astype(np.int64),
        "source_event_key": source_event_key_array(selected_events, source_row_index),
        "classification": classification_indices.astype(np.int64),
        "classification_target_index": classification_indices,
        "classification_target_name": ak.Array(classification_names.tolist()),
        "event_weight": event_weight.astype(np.float32),
        "central_weight": central_weight.astype(np.float32),
        "x": to_numpy(x, np.float32),
        "x_mask": to_numpy(x_mask, bool),
        "conditions": to_numpy(conditions, np.float32),
        "conditions_mask": to_numpy(conditions_mask, bool),
        "num_vectors": num_vectors.astype(np.float32),
        "num_sequential_vectors": num_sequential_vectors.astype(np.float32),
        "x_invisible": to_numpy(delta_invisible_input, np.float32),
        "x_invisible_mask": to_numpy(delta_invisible_mask, np.float32),
        "num_invisible_raw": num_invisible_raw.astype(np.int64),
        "num_invisible_valid": num_invisible_valid.astype(np.int64),
        "lead_a_visible_p4": visible_a,
        "lead_b_visible_p4": visible_b,
        "target_a_invisible_p4": invisible_a,
        "target_b_invisible_p4": invisible_b,
        "truth_tau_a_p4": truth_tau_a,
        "truth_tau_b_p4": truth_tau_b,
        "truth_a_invisible_p4": truth_visible_a,
        "truth_b_invisible_p4": truth_visible_b,
        "lead_a_visible_px": visible_a.px,
        "lead_a_visible_py": visible_a.py,
        "lead_a_visible_pz": visible_a.pz,
        "lead_a_visible_E": visible_a.E,
        "lead_b_visible_px": visible_b.px,
        "lead_b_visible_py": visible_b.py,
        "lead_b_visible_pz": visible_b.pz,
        "lead_b_visible_E": visible_b.E,
        "truth_a_visible_px": truth_visible_a.px,
        "truth_b_visible_px": truth_visible_b.px,
        "truth_a_visible_py": truth_visible_a.py,
        "truth_b_visible_py": truth_visible_b.py,
        "truth_a_visible_pz": truth_visible_a.pz,
        "truth_b_visible_pz": truth_visible_b.pz,
        "truth_a_visible_E": truth_visible_a.E,
        "truth_b_visible_E": truth_visible_b.E,
        "truth_tau_a_px": truth_tau_a.px,
        "truth_tau_b_px": truth_tau_b.px,
        "truth_tau_a_py": truth_tau_a.py,
        "truth_tau_b_py": truth_tau_b.py,
        "truth_tau_a_pz": truth_tau_a.pz,
        "truth_tau_b_pz": truth_tau_b.pz,
        "truth_tau_a_E": truth_tau_a.E,
        "truth_tau_b_E": truth_tau_b.E,
        "target_a_invisible_px": invisible_a.px,
        "target_a_invisible_py": invisible_a.py,
        "target_a_invisible_pz": invisible_a.pz,
        "target_a_invisible_E": invisible_a.E,
        "target_b_invisible_px": invisible_b.px,
        "target_b_invisible_py": invisible_b.py,
        "target_b_invisible_pz": invisible_b.pz,
        "target_b_invisible_E": invisible_b.E,
    }

    if sample.total_initial_num_events is not None:
        fields["initial_num_events"] = np.full(
            len(selected_events),
            sample.total_initial_num_events,
            dtype=np.float64
        )

    if "nprong" in selected_events.fields:
        fields["nprong"] = to_numpy(selected_events["nprong"], np.int32)

    passthrough = {
        "event_category",
        "truth_QI_region",
        "analyzing_power",
        "analyzing_power_a",
        "analyzing_power_b",
        "weight",
        "central_weight",
        "truth_theta_cm",
        "truth_mtautau"
    }
    passthrough.update(f"truth_cos_theta_A_{r}" for r in ["k", "n", "r"])
    passthrough.update(f"truth_cos_theta_B_{r}" for r in ["k", "n", "r"])
    passthrough.update(f"truth_cos_theta_A_{r}_times_cos_theta_B_{l}" for r in ["k", "n", "r"] for l in ["k", "n", "r"])
    passthrough.update(name for name in selected_events.fields if name.endswith("_cut"))


    baseline_passthrough = {
        "mmc_likelihood",
        "theta_cm",
        "flags_valid"
    }
    baseline_passthrough.update(f"cos_theta_A_{r}" for r in ["k", "n", "r"])
    baseline_passthrough.update(f"cos_theta_B_{r}" for r in ["k", "n", "r"])
    baseline_passthrough.update(f"cos_theta_A_{r}_times_cos_theta_B_{l}" for r in ["k", "n", "r"] for l in ["k", "n", "r"])

    for field in sorted(passthrough):
        if field in selected_events.fields and field not in fields:
            fields[field] = selected_events[field]

    for field in sorted(baseline_passthrough):
        if field in selected_events.fields and f"baseline_{field}" not in fields:
            fields[f"baseline_{field}"] = selected_events[field]
    for field_name, values in list(fields.items()):
        if isinstance(values, np.ndarray):
            fields[field_name] = np.ascontiguousarray(values)

    return ak.Array(fields)



def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    samples: list[Sample],
    luminosity: float | None,
    classification_lookup,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
    remove_neutral_non_photon: bool,
    worker_results: list[dict[str, Any]],
) -> None:
    sample_manifest: dict[str, Any] = {}
    for sample in samples:
        sample_manifest[sample.key] = {
            "name": sample.name,
            "is_data": sample.is_data,
            "is_signal": sample.is_signal,
            "file_source": sample.file_source,
            "plot_label": sample.plot_label,
            "files": sample.files,
            "total_initial_num_events": sample.total_initial_num_events,
        }
    shards_by_sample: dict[str, list[dict[str, Any]]] = {sample.key: [] for sample in samples}
    for result in worker_results:
        shards_by_sample[result["sample_key"]].extend(result["shards"])
    payload = {
        "format": "ml_pipeline_lite_evenet_input_v2",
        "analysis_config": str(args.analysis_config),
        "batch_size": args.batch_size,
        "rows_per_shard": args.rows_per_shard,
        "num_workers": args.num_workers,
        "luminosity": luminosity,
        "max_particles": max_particles,
        "remove_neutral_non_photon": remove_neutral_non_photon,
        "class_labels": list(classification_lookup.class_labels),
        "sequential_features": list(feature_config.all_sequential_fields),
        "global_condition_features": list(feature_config.global_fields),
        "invisible_features": list(invisible_features),
        "samples": sample_manifest,
        "shards": shards_by_sample,
    }
    manifest_path = output_dir / "manifest.json"
    payload = make_json_serializable(payload)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[ml_pipeline_lite] wrote {manifest_path}")



def write_shard(events: ak.Array, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(events, path, compression="snappy")


def worker_process_file(
    sample_payload: dict[str, Any],
    luminosity: float | None,
    classification_lookup,
    feature_config,
    invisible_features: tuple[str, ...],
    max_particles: int,
    remove_neutral_non_photon: bool,
    source_sample_index: int,
    file_path: str,
    source_file_index: int,
    output_dir: str,
    batch_size: int,
    rows_per_shard: int,
):
    sample = Sample(**sample_payload)
    output_root = Path(output_dir)
    parquet = pq.ParquetFile(file_path)
    schema_names = {field.name for field in parquet.schema_arrow}
    columns = required_columns(schema_names, feature_config, sample)

    shard_buffers: list[ak.Array] = []
    selected_in_buffer = 0
    written_shards: list[dict[str, Any]] = []
    row_offset = 0
    shard_index = 0

    def flush_buffer() -> None:
        nonlocal shard_buffers, selected_in_buffer, shard_index
        if not shard_buffers:
            return
        shard_events = shard_buffers[0] if len(shard_buffers) == 1 else  ak.concatenate(shard_buffers, axis=0)
        shard_path = (
            output_root
            / "shards"
            / sample.key
            / f"{sample.key}__file{source_file_index:03d}__shard{shard_index:05d}.parquet"
        )
        write_shard(shard_events, shard_path)
        written_shards.append(
            {
                "path": str(shard_path.relative_to(output_root)),
                "rows": int(len(shard_events)),
                "source_file": file_path,
                "source_file_index": source_file_index,
            }
        )
        shard_index += 1
        shard_buffers = []
        selected_in_buffer = 0

    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        events = ak.from_arrow(record_batch)
        vector_fields = ["lead_a_visible_p4", "lead_b_visible_p4"]
        if sample.is_signal:
            vector_fields.extend(["truth_tau_a_p4", "truth_tau_b_p4"])
            if "truth_visible_a_p4" in events.fields:
                vector_fields.append("truth_visible_a_p4")
            if "truth_visible_b_p4" in events.fields:
                vector_fields.append("truth_visible_b_p4")
        if "missing_p4" in events.fields:
            vector_fields.append("missing_p4")
        for field in vector_fields:
            events[field] = rebuild_vector(events[field])

        mask = event_preselection_mask(events)
        if not np.any(mask):
            row_offset += len(events)
            continue
        selected = events[mask]
        selected_indices = np.flatnonzero(mask).astype(np.int64) + row_offset
        output_events = build_output_events(
            selected_events=selected,
            sample=sample,
            luminosity=luminosity,
            classification_lookup=classification_lookup,
            feature_config=feature_config,
            invisible_features=invisible_features,
            max_particles=max_particles,
            remove_neutral_non_photon=remove_neutral_non_photon,
            source_sample_index=source_sample_index,
            source_file_index=source_file_index,
            source_row_index=selected_indices
        )
        shard_buffers.append(output_events)
        selected_in_buffer += len(output_events)
        if selected_in_buffer >= rows_per_shard:
            flush_buffer()
        row_offset += len(events)
    flush_buffer()
    return {
        "sample_key": sample.key,
        "source_file": file_path,
        "source_file_index": source_file_index,
        "shards": written_shards,
    }





def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evenet input from parquet"
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("ml_pipeline/config/analysis.yaml"),
        help="Analysis YAML with Samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for lite parquet shards and monitoring.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Optional subset of sample keys from analysis.yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Rows per parquet record-batch read.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=100000,
        help="Selected rows per output shard.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers. Each worker processes one input parquet file.",
    )
    parser.add_argument(
        "--remove_neutral_non_photon",
        action="store_true",
        default=False,
        help="Remove non-photon candidates from input point cloud",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.analysis_config)
    feature_config = parse_feature_config(config)
    invisible_features = tuple(str(key) for key in config["Normalization"].get("Invisible"))
    remove_neutral_non_photon = args.remove_neutral_non_photon
    samples = attach_sample_total_intial_events(
        parse_samples(config, args.samples)
    )

    selected_keys = {sample.key for sample in samples}
    luminosity = infer_luminosity(samples)
    classification_lookup = build_classification_lookup(config, selected_keys)
    sample_index_lookup = {sample.key: index for index, sample in enumerate(samples)}

    # configure multi-cpu jobs
    jobs: list[tuple[Sample, str, int]] = []
    for sample in samples:
        files = sample.files
        for file_index, file_path in enumerate(files):
            jobs.append((sample, file_path, file_index))

    max_particles = 0
    max_scan_columns = ["nprong", "Part_fourMomentum_fCoordinates_fT", "Part_pdgId", "Part_charge"]
    for sample in samples:
        if sample.is_data:
            continue
        for file_path in sample.files:
            parquet = pq.ParquetFile(file_path)
            schema_names = {field.name for field in parquet.schema_arrow}
            scan_columns = [column for column in max_scan_columns if column in schema_names]
            row_offset = 0
            for record_batch in parquet.iter_batches(batch_size=args.batch_size, columns=scan_columns):
                events = ak.from_arrow(record_batch)
                mask = event_preselection_mask(events)
                if np.any(mask):
                    selected = events[mask]
                    input_part_mask = build_input_particle_mask(selected, remove_neutral_non_photon)
                    num_particles = ak.sum(input_part_mask, axis=1)

                    if max(num_particles) > max_particles:
                        max_particles = max(num_particles)
                row_offset += len(events)
    print(f"[ml_pipeline_lite] output_dir={output_dir}")
    print(f"[ml_pipeline_lite] samples={[sample.key for sample in samples]}")
    print(f"[ml_pipeline_lite] class_labels={list(classification_lookup.class_labels)}")
    print(f"[ml_pipeline_lite] jobs={len(jobs)} workers={args.num_workers}")
    print(f"[ml_pipeline_lite] invisible_features={list(invisible_features)}")
    print(f"[ml_pipeline_lite] remove_neutral_non_photon={remove_neutral_non_photon}")
    print(f"[ml_pipeline_lite] max_particles={max_particles}")

    worker_results: list[dict[str, Any]] = []
    max_workers = max(1, min(args.num_workers, len(jobs)))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                worker_process_file,
                sample_payload={
                    "key": sample.key,
                    "name": sample.name,
                    "is_data": sample.is_data,
                    "is_signal": sample.is_signal,
                    "files": sample.files,
                    "file_source": sample.file_source,
                    "norm_factor": sample.norm_factor,
                    "lumi": sample.lumi,
                    "total_initial_num_events": sample.total_initial_num_events,
                    "plot_label": sample.plot_label,
                    "predict_neutrino": sample.predict_neutrino,
                },
                luminosity=luminosity,
                classification_lookup=classification_lookup,
                feature_config=feature_config,
                invisible_features=invisible_features,
                max_particles=max_particles,
                remove_neutral_non_photon=remove_neutral_non_photon,
                source_sample_index=sample_index_lookup[sample.key],
                file_path=file_path,
                source_file_index=file_index,
                output_dir=str(output_dir),
                batch_size=args.batch_size,
                rows_per_shard=args.rows_per_shard,
            ): (sample.key, file_path)
            for sample, file_path, file_index in jobs
        }

        for future in as_completed(future_map):
            sample_key, file_path = future_map[future]
            result = future.result()
            worker_results.append(result)
            row_written = sum(int(item["rows"]) for item in result["shards"])
            print(
                f"[ml_pipeline_lite] finished sample={sample_key} file={file_path} "
                f"shards={len(result['shards'])} rows={row_written}"
            )
    write_manifest(
        output_dir,
        args,
        samples,
        luminosity,
        classification_lookup,
        feature_config,
        invisible_features,
        max_particles,
        remove_neutral_non_photon,
        worker_results,
    )



if __name__ == "__main__":
    main()