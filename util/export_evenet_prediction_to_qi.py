#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables, get_observable_names
from utils.common_functions import rebuild_p4
from build_evenet_input_from_parquet import apply_preselection, expand_samples, parse_config, read_yaml


vector.register_awkward()

DEFAULT_FLOAT = -99.0
DEFAULT_REGIONS = ("baseline", "hadhad", "ee", "mumu", "emu")
EXPORT_PASSTHROUGH_EXACT_FIELDS = {
    "event_category",
    "truth_QI_region",
    "analyzing_power_a",
    "analyzing_power_b",
    "analyzing_power",
    "initial_total_num_events",
    "weight",
    "central_weight",
    "event_weight",
    "evenet_weight",
    "source_sample_index",
    "source_slot_for_a",
    "source_slot_for_b",
    "evenet_pred_class_index",
    "evenet_pred_class_prob",
    "evenet_pred_class_name",
    "evenet_truth_class_index",
    "evenet_truth_class_name",
}
EXPORT_PASSTHROUGH_PREFIXES = ("truth_", "pred_invisible_", "target_invisible_", "tau_vis_prong_", "tau_vis_target_", "evenet_")
EXPORT_PASSTHROUGH_SUFFIXES = ("_cut",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export EveNet prediction parquet into the central QI/unfolding parquet schema "
            "without modifying the central framework."
        )
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=None,
        help=(
            "analysis.yaml controlling Samples.*.input_files/raw_files and optional EveNetPrediction paths. "
            "When provided with prediction parquets, this is the preferred config-driven mode."
        ),
    )
    parser.add_argument(
        "--mc-pred-parquet",
        type=Path,
        default=None,
        help="MC/test EveNet prediction parquet. Overrides EveNetPrediction.mc_parquet in analysis.yaml.",
    )
    parser.add_argument(
        "--data-pred-parquet",
        type=Path,
        default=None,
        help="Data EveNet prediction parquet. Overrides EveNetPrediction.data_parquet in analysis.yaml.",
    )
    parser.add_argument(
        "--central-parquet",
        type=Path,
        default=None,
        help=(
            "Full central/DataLoader raw parquet that defines region cuts, weights, truth QI fields, "
            "and event_category. Usually filtered___raw.parquet. Events without EveNet prediction "
            "are kept with invalid/default reconstruction."
        ),
    )
    parser.add_argument(
        "--prediction-central-parquet",
        type=Path,
        default=None,
        help=(
            "Optional central/DataLoader parquet corresponding row-by-row to the EveNet prediction parquet "
            "(for example the nprong==2/preselected parquet used to build EveNet inputs). When provided, "
            "EveNet predictions are merged back into --central-parquet by event key or row matching."
        ),
    )
    parser.add_argument(
        "--evenet-pred-parquet",
        type=Path,
        default=None,
        help="Prediction parquet produced by util/predict_evenet_from_raw_parquet.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory that will contain <sample-name>/filtered___*.parquet.",
    )
    parser.add_argument(
        "--sample-name",
        default="Ztautau",
        help="Central sample directory name to write under output-dir.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=list(DEFAULT_REGIONS),
        help="Region parquet files to write. A region is used only if <region>_cut exists, except raw.",
    )
    parser.add_argument(
        "--allow-prefix-match",
        action="store_true",
        help=(
            "Allow prefix matching when a row-aligned prediction-central parquet is omitted. "
            "This keeps the full central raw output and applies predictions to the first N rows only."
        ),
    )
    parser.add_argument(
        "--prediction-split-fraction",
        type=float,
        default=None,
        help=(
            "Legacy fallback fraction represented by the EveNet-predicted subset, e.g. 0.5 for a test split. "
            "If the prediction parquet has evenet_weight, that already includes the split correction and is used "
            "directly for predicted rows. This option is only applied when evenet_weight is absent."
        ),
    )
    parser.add_argument(
        "--qi-method-label",
        default=None,
        help=(
            "Output method subdirectory for config-driven export, e.g. evenet_pretrain or evenet_scratch. "
            "Overrides EveNetPrediction.qi_method_label."
        ),
    )
    parser.add_argument(
        "--write-baseline-copy",
        action="store_true",
        help=(
            "Also write a baseline copy under <output-dir>/baseline/<sample-name>. "
            "This preserves central MMC/algebraic lead_*_missing_p4 fields for comparison."
        ),
    )
    parser.add_argument(
        "--baseline-output-dir",
        type=Path,
        default=None,
        help="Optional explicit baseline output directory. Defaults to <output-dir>/baseline.",
    )
    parser.add_argument(
        "--method-output-subdir",
        default=None,
        help=(
            "Optional subdirectory under output-dir for the EveNet export, e.g. 'evenet'. "
            "If omitted, writes directly under output-dir."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Number of CPU workers for config-driven export. "
            "If omitted, use EveNetPrediction.num_workers from analysis.yaml or 1."
        ),
    )
    parser.add_argument(
        "--worker-backend",
        choices=["thread", "process"],
        default=None,
        help=(
            "Parallel backend for config-driven export. Defaults to 'thread' to avoid copying large "
            "awkward arrays into subprocesses. Use 'process' only if memory is sufficient."
        ),
    )
    return parser.parse_args()


def load_events(path: Path) -> ak.Array:
    events = ak.from_parquet(path)
    for field in events.fields:
        if field.endswith("_p4"):
            events[field] = rebuild_vector(events[field])
    return events


def expand_paths(patterns: list[str] | tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(str(pattern)))
        paths.extend(matched if matched else [str(pattern)])
    return paths


def load_concat_events(paths: list[str]) -> ak.Array:
    if not paths:
        raise ValueError("No parquet paths were provided.")
    arrays = [load_events(Path(path)) for path in paths]
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(values.fields)
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip(
            {
                "px": values["px"],
                "py": values["py"],
                "pz": values["pz"],
                "E": values["E"],
            }
        )
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


def build_momentum4d(px, py, pz, energy):
    return ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        },
        with_name="Momentum4D",
    )


def materialize_p4_for_parquet(values: ak.Array) -> ak.Array:
    vector_values = rebuild_vector(values)
    return ak.zip(
        {
            "x": vector_values.px,
            "y": vector_values.py,
            "z": vector_values.pz,
            "t": vector_values.E,
        },
        with_name="Momentum4D",
    )


def prepare_events_for_parquet(events: ak.Array) -> ak.Array:
    output = events
    for field in output.fields:
        if field.endswith("_p4"):
            output[field] = materialize_p4_for_parquet(output[field])
    return output


def canonicalize_p4(p4: ak.Array) -> ak.Array:
    return build_momentum4d(
        ak.to_numpy(p4.px, allow_missing=False).astype(np.float64),
        ak.to_numpy(p4.py, allow_missing=False).astype(np.float64),
        ak.to_numpy(p4.pz, allow_missing=False).astype(np.float64),
        ak.to_numpy(p4.E, allow_missing=False).astype(np.float64),
    )


def p4_from_components(events: ak.Array, prefix: str, slot: int) -> tuple[ak.Array, np.ndarray]:
    valid_field = f"{prefix}_slot{slot}_valid"
    valid = (
        ak.to_numpy(events[valid_field], allow_missing=False).astype(bool)
        if valid_field in events.fields
        else np.ones(len(events), dtype=bool)
    )

    component = lambda name: ak.to_numpy(events[f"{prefix}_slot{slot}_{name}"], allow_missing=False).astype(np.float64)

    if all(f"{prefix}_slot{slot}_{name}" in events.fields for name in ("E", "px", "py", "pz")):
        p4 = vector.zip(
            {
                "px": component("px"),
                "py": component("py"),
                "pz": component("pz"),
                "E": component("E"),
            }
        )
        return canonicalize_p4(p4), valid

    # convert
    if all(f"{prefix}_slot{slot}_{name}" in events.fields for name in ("energy", "pt", "eta", "phi")):
        pt = component("pt")
        eta = component("eta")
        phi = component("phi")
        energy = component("energy")
        p4 = build_momentum4d(
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            energy,
        )
        return canonicalize_p4(p4), valid

    if all(f"{prefix}_slot{slot}_{name}" in events.fields for name in ("log_energy", "log_pt", "eta", "phi")):
        pt = np.expm1(component("log_pt"))
        eta = component("eta")
        phi = component("phi")
        energy = np.expm1(component("log_energy"))
        p4 = build_momentum4d(
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            energy,
        )
        return canonicalize_p4(p4), valid

    available = [field for field in events.fields if field.startswith(f"{prefix}_slot{slot}_")]
    raise KeyError(
        f"Cannot build p4 for {prefix}_slot{slot}. Need E/px/py/pz, energy/pt/eta/phi, "
        f"or log_energy/log_pt/eta/phi. Available: {available}"
    )


def choose_by_slot(slot0, slot1, choose_slot1: np.ndarray):
    return ak.where(ak.Array(choose_slot1), slot1, slot0)


def choose_component_by_slot(events: ak.Array, prefix: str, slot_indices: np.ndarray, component: str) -> np.ndarray | None:
    slot0_name = f"{prefix}_slot0_{component}"
    slot1_name = f"{prefix}_slot1_{component}"
    fields = set(events.fields)
    if slot0_name not in fields or slot1_name not in fields:
        return None
    slot0 = ak.to_numpy(events[slot0_name], allow_missing=False).astype(np.float64)
    slot1 = ak.to_numpy(events[slot1_name], allow_missing=False).astype(np.float64)
    return np.where(slot_indices == 0, slot0, slot1)


def choose_valid_by_slot(events: ak.Array, prefix: str, slot_indices: np.ndarray) -> np.ndarray:
    valid_field0 = f"{prefix}_slot0_valid"
    valid_field1 = f"{prefix}_slot1_valid"
    fields = set(events.fields)
    valid0 = (
        ak.to_numpy(events[valid_field0], allow_missing=False).astype(bool)
        if valid_field0 in fields
        else np.ones(len(events), dtype=bool)
    )
    valid1 = (
        ak.to_numpy(events[valid_field1], allow_missing=False).astype(bool)
        if valid_field1 in fields
        else np.ones(len(events), dtype=bool)
    )
    return np.where(slot_indices == 0, valid0, valid1)


def build_remapped_p4(events: ak.Array, prefix: str, slot_indices: np.ndarray) -> tuple[ak.Array, np.ndarray]:
    px = choose_component_by_slot(events, prefix, slot_indices, "px")
    py = choose_component_by_slot(events, prefix, slot_indices, "py")
    pz = choose_component_by_slot(events, prefix, slot_indices, "pz")
    energy = choose_component_by_slot(events, prefix, slot_indices, "E")
    if px is not None and py is not None and pz is not None and energy is not None:
        return build_momentum4d(px, py, pz, energy), choose_valid_by_slot(events, prefix, slot_indices)

    energy = choose_component_by_slot(events, prefix, slot_indices, "energy")
    pt = choose_component_by_slot(events, prefix, slot_indices, "pt")
    eta = choose_component_by_slot(events, prefix, slot_indices, "eta")
    phi = choose_component_by_slot(events, prefix, slot_indices, "phi")
    if energy is not None and pt is not None and eta is not None and phi is not None:
        return build_momentum4d(
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            energy,
        ), choose_valid_by_slot(events, prefix, slot_indices)

    log_energy = choose_component_by_slot(events, prefix, slot_indices, "log_energy")
    log_pt = choose_component_by_slot(events, prefix, slot_indices, "log_pt")
    eta = choose_component_by_slot(events, prefix, slot_indices, "eta")
    phi = choose_component_by_slot(events, prefix, slot_indices, "phi")
    if log_energy is not None and log_pt is not None and eta is not None and phi is not None:
        pt = np.expm1(log_pt)
        energy = np.expm1(log_energy)
        return build_momentum4d(
            pt * np.cos(phi),
            pt * np.sin(phi),
            pt * np.sinh(eta),
            energy,
        ), choose_valid_by_slot(events, prefix, slot_indices)

    available = [field for field in events.fields if field.startswith(f"{prefix}_slot")]
    raise KeyError(
        f"Cannot build remapped p4 for prefix={prefix}. Need slot-wise E/px/py/pz, energy/pt/eta/phi, "
        f"or log_energy/log_pt/eta/phi. Available: {available}"
    )


def finite_p4_mask(p4: ak.Array) -> np.ndarray:
    return (
        np.isfinite(ak.to_numpy(p4.px, allow_missing=False))
        & np.isfinite(ak.to_numpy(p4.py, allow_missing=False))
        & np.isfinite(ak.to_numpy(p4.pz, allow_missing=False))
        & np.isfinite(ak.to_numpy(p4.E, allow_missing=False))
    )


def zero_p4(num_events: int) -> ak.Array:
    nan_values = np.full(num_events, np.nan, dtype=np.float64)
    return build_momentum4d(nan_values, nan_values, nan_values, nan_values)


def default_float_array(num_events: int, dtype=np.float32) -> np.ndarray:
    return np.full(num_events, DEFAULT_FLOAT, dtype=dtype)


def default_object_array(num_events: int, value: str = "unpredicted") -> ak.Array:
    return ak.Array([value] * num_events)


def default_field_like(field_values: ak.Array, num_events: int):
    if hasattr(field_values, "fields") and (
        {"px", "py", "pz", "E"}.issubset(set(field_values.fields))
        or {"x", "y", "z", "t"}.issubset(set(field_values.fields))
    ):
        return zero_p4(num_events)
    array = ak.to_numpy(field_values, allow_missing=False)
    if array.dtype == np.bool_:
        return np.zeros(num_events, dtype=bool)
    if np.issubdtype(array.dtype, np.integer):
        return np.full(num_events, -1, dtype=array.dtype)
    if np.issubdtype(array.dtype, np.floating):
        return default_float_array(num_events, dtype=array.dtype)
    return default_object_array(num_events)


def should_passthrough_export_field(field: str) -> bool:
    return (
        field in EXPORT_PASSTHROUGH_EXACT_FIELDS
        or field.startswith(EXPORT_PASSTHROUGH_PREFIXES)
        or field.endswith(EXPORT_PASSTHROUGH_SUFFIXES)
    )


def compact_export_fields(events: ak.Array) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in events.fields:
        if not should_passthrough_export_field(field):
            continue
        values = events[field]
        if len(getattr(values, "fields", [])) > 0:
            continue
        try:
            array = ak.to_numpy(values, allow_missing=False)
        except Exception:
            continue
        if array.ndim != 1:
            continue
        if array.dtype.kind in {"b", "i", "u", "f", "U", "S", "O"}:
            output[field] = values
    return output


def align_fields_for_concat(arrays: list[ak.Array]) -> list[ak.Array]:
    if not arrays:
        return arrays
    template_by_field: dict[str, Any] = {}
    for events in arrays:
        for field in events.fields:
            template_by_field.setdefault(field, events[field])

    aligned: list[ak.Array] = []
    for events in arrays:
        output = events
        for field, template in template_by_field.items():
            if field not in output.fields:
                output[field] = default_field_like(template, len(output))
        aligned.append(output)
    return aligned


def safe_delta_r(first: ak.Array, second: ak.Array) -> np.ndarray:
    values = ak.to_numpy(first.deltaR(second), allow_missing=False).astype(np.float64)
    return np.where(np.isfinite(values), values, 1.0e9)


def prediction_selection_mask(full_events: ak.Array) -> np.ndarray:
    fields = set(full_events.fields)
    mask = np.ones(len(full_events), dtype=bool)
    if "baseline_cut" in fields:
        mask &= ak.to_numpy(full_events["baseline_cut"], allow_missing=False).astype(bool)
    if "nprong" in fields:
        mask &= ak.to_numpy(full_events["nprong"], allow_missing=False).astype(np.int64) == 2
    return mask


def build_predicted_reconstruction(
    central_pred_events: ak.Array,
    pred_events: ak.Array,
) -> tuple[dict[str, Any], dict[str, Any]]:
    num_predicted_rows = len(pred_events)

    if "source_slot_for_a" not in pred_events.fields or "source_slot_for_b" not in pred_events.fields:
        raise ValueError(
            "Prediction parquet is missing source_slot_for_a/source_slot_for_b. "
            "Rebuild the EveNet input and rerun prediction so export can restore the central a/b definition "
            "without reading visible tau p4 from raw parquet."
        )
    slot_for_a = ak.to_numpy(pred_events["source_slot_for_a"], allow_missing=False).astype(np.int64)
    slot_for_b = ak.to_numpy(pred_events["source_slot_for_b"], allow_missing=False).astype(np.int64)
    if np.any((slot_for_a < 0) | (slot_for_a > 1) | (slot_for_b < 0) | (slot_for_b > 1)):
        raise ValueError("source_slot_for_a/source_slot_for_b must be 0 or 1 for every prediction row.")

    lead_a_visible, vis_valid_a = build_remapped_p4(pred_events, "tau_vis_prong", slot_for_a)
    lead_b_visible, vis_valid_b = build_remapped_p4(pred_events, "tau_vis_prong", slot_for_b)
    lead_a_missing, pred_valid_a = build_remapped_p4(pred_events, "pred_invisible", slot_for_a)
    lead_b_missing, pred_valid_b = build_remapped_p4(pred_events, "pred_invisible", slot_for_b)
    reco_tau_a = lead_a_visible + lead_a_missing
    reco_tau_b = lead_b_visible + lead_b_missing

    del central_pred_events
    delta_r_a = np.full(num_predicted_rows, np.nan, dtype=np.float32)
    delta_r_b = np.full(num_predicted_rows, np.nan, dtype=np.float32)

    flags_valid = (
        vis_valid_a
        & vis_valid_b
        & pred_valid_a
        & pred_valid_b
        & finite_p4_mask(lead_a_visible)
        & finite_p4_mask(lead_b_visible)
        & finite_p4_mask(reco_tau_a)
        & finite_p4_mask(reco_tau_b)
    )
    observables = build_observables(
        tau_a_p4=reco_tau_a,
        tau_b_p4=reco_tau_b,
        vis_a_p4=lead_a_visible,
        vis_b_p4=lead_b_visible,
    )

    values: dict[str, Any] = {
        "lead_a_visible_p4": lead_a_visible,
        "lead_b_visible_p4": lead_b_visible,
        "lead_a_missing_p4": lead_a_missing,
        "lead_b_missing_p4": lead_b_missing,
        "reco_tau_a_p4": reco_tau_a,
        "reco_tau_b_p4": reco_tau_b,
        "flags_valid": flags_valid,
        "mmc_likelihood": np.zeros(num_predicted_rows, dtype=np.float32),
        "neutrino_method": ak.Array(["EveNet"] * num_predicted_rows),
        "evenet_has_prediction": np.ones(num_predicted_rows, dtype=bool),
        "evenet_slot_for_a": slot_for_a.astype(np.int8),
        "evenet_slot_for_b": slot_for_b.astype(np.int8),
        "evenet_leg_match_deltaR_a": delta_r_a,
        "evenet_leg_match_deltaR_b": delta_r_b,
    }
    for obs_name, obs_values in observables.items():
        values[obs_name] = ak.where(flags_valid, obs_values, np.nan)
    for field in pred_events.fields:
        if (
            field.startswith("evenet_")
            or field.startswith("pred_invisible_")
            or field.startswith("target_invisible_")
            or field.startswith("tau_vis_prong_")
            or field.startswith("tau_vis_target_")
        ):
            values[field] = pred_events[field]

    metrics = {
        "num_predicted_events": int(num_predicted_rows),
        "valid_predicted_events": int(np.sum(flags_valid)),
        "valid_predicted_fraction": float(np.mean(flags_valid)) if len(flags_valid) else 0.0,
        "slot_alignment": "prediction_metadata_source_slot_for_a",
        "slot_swap_fraction": float(np.mean(slot_for_a == 0)) if len(slot_for_a) else 0.0,
        "median_deltaR_a": finite_median(delta_r_a),
        "median_deltaR_b": finite_median(delta_r_b),
        "p95_deltaR_a": finite_percentile(delta_r_a, 95.0),
        "p95_deltaR_b": finite_percentile(delta_r_b, 95.0),
    }
    return values, metrics


def default_evenet_columns(full_events: ak.Array, pred_values: dict[str, Any]) -> dict[str, Any]:
    num_events = len(full_events)
    defaults: dict[str, Any] = {
        "lead_a_visible_p4": zero_p4(num_events),
        "lead_b_visible_p4": zero_p4(num_events),
        "lead_a_missing_p4": zero_p4(num_events),
        "lead_b_missing_p4": zero_p4(num_events),
        "reco_tau_a_p4": zero_p4(num_events),
        "reco_tau_b_p4": zero_p4(num_events),
        "flags_valid": np.zeros(num_events, dtype=bool),
        "mmc_likelihood": np.zeros(num_events, dtype=np.float32),
        "neutrino_method": default_object_array(num_events, "EveNet_default"),
        "evenet_has_prediction": np.zeros(num_events, dtype=bool),
        "evenet_slot_for_a": np.full(num_events, -1, dtype=np.int8),
        "evenet_slot_for_b": np.full(num_events, -1, dtype=np.int8),
        "evenet_leg_match_deltaR_a": np.full(num_events, np.nan, dtype=np.float32),
        "evenet_leg_match_deltaR_b": np.full(num_events, np.nan, dtype=np.float32),
    }
    for obs_name in get_observable_names_safe(pred_values):
        defaults[obs_name] = np.full(num_events, np.nan, dtype=np.float32)
    for field, values in pred_values.items():
        if field in defaults:
            continue
        defaults[field] = default_field_like(values, num_events)
    return defaults


def get_observable_names_safe(pred_values: dict[str, Any]) -> list[str]:
    return [field for field in pred_values if field == "theta_cm" or field.startswith("cos_theta_")]


def assign_at_indices(base_values: Any, indices: np.ndarray, pred_values: Any):
    if (
        isinstance(base_values, ak.Array)
        and hasattr(base_values, "fields")
        and (
            {"px", "py", "pz", "E"}.issubset(set(base_values.fields))
            or {"x", "y", "z", "t"}.issubset(set(base_values.fields))
        )
    ):
        pred_p4 = rebuild_vector(pred_values)
        px = ak.to_numpy(base_values.px, allow_missing=False)
        py = ak.to_numpy(base_values.py, allow_missing=False)
        pz = ak.to_numpy(base_values.pz, allow_missing=False)
        energy = ak.to_numpy(base_values.E, allow_missing=False)
        px[indices] = ak.to_numpy(pred_p4.px, allow_missing=False)
        py[indices] = ak.to_numpy(pred_p4.py, allow_missing=False)
        pz[indices] = ak.to_numpy(pred_p4.pz, allow_missing=False)
        energy[indices] = ak.to_numpy(pred_p4.E, allow_missing=False)
        return build_momentum4d(px, py, pz, energy)

    if isinstance(base_values, ak.Array):
        try:
            array = ak.to_numpy(base_values, allow_missing=False)
            pred_array = ak.to_numpy(pred_values, allow_missing=False)
            if array.dtype.kind in {"U", "S", "O"} or pred_array.dtype.kind in {"U", "S", "O"}:
                # Preserve full string labels such as "Ztautau_rhorho" when merging prediction
                # fields back into the full raw event table. Fixed-width Unicode numpy arrays would
                # silently truncate longer labels and break the exported region cuts.
                object_array = np.asarray(array, dtype=object)
                object_pred_array = np.asarray(pred_array, dtype=object)
                object_array[indices] = object_pred_array
                return ak.Array(object_array.tolist())
            array[indices] = pred_array
            return ak.Array(array) if array.dtype.kind in {"U", "O"} else array
        except Exception:
            python_values = ak.to_list(base_values)
            pred_python_values = ak.to_list(pred_values)
            for out_index, value in zip(indices, pred_python_values):
                python_values[int(out_index)] = value
            return ak.Array(python_values)

    array = np.asarray(base_values).copy()
    pred_array = ak.to_numpy(pred_values, allow_missing=False) if isinstance(pred_values, ak.Array) else np.asarray(pred_values)
    array[indices] = pred_array
    return array


def with_evenet_reconstruction(
    full_events: ak.Array,
    central_pred_events: ak.Array,
    pred_events: ak.Array,
    full_indices: np.ndarray,
    prediction_split_fraction: float | None,
    selected_full_indices: np.ndarray | None = None,
    zero_unpredicted_selected_mc: bool = False,
) -> tuple[ak.Array, dict[str, Any]]:
    pred_values, metrics = build_predicted_reconstruction(central_pred_events, pred_events)
    output = full_events
    columns = default_evenet_columns(full_events, pred_values)
    for field, base_values in columns.items():
        output[field] = assign_at_indices(base_values, full_indices, pred_values[field])

    if "weight" in output.fields:
        original_weight = ak.to_numpy(output["weight"], allow_missing=False).astype(np.float64)
        output["central_weight"] = original_weight.astype(np.float32)
        predicted_weight = None
        if "evenet_weight" in pred_events.fields:
            predicted_weight = ak.to_numpy(pred_events["evenet_weight"], allow_missing=False).astype(np.float64)
            if len(predicted_weight) != len(full_indices):
                raise ValueError(
                    "Prediction parquet evenet_weight length does not match the mapped prediction rows: "
                    f"evenet_weight={len(predicted_weight)}, mapped_rows={len(full_indices)}."
                )
        weight_scale = np.ones(len(output), dtype=np.float32)
        export_weight = original_weight.copy()
        if predicted_weight is not None:
            export_weight[full_indices] = predicted_weight
            with np.errstate(divide="ignore", invalid="ignore"):
                predicted_scale = np.divide(
                    predicted_weight,
                    original_weight[full_indices],
                    out=np.ones_like(predicted_weight, dtype=np.float64),
                    where=original_weight[full_indices] != 0,
                )
            weight_scale[full_indices] = predicted_scale.astype(np.float32)
            metrics["prediction_weight_source"] = "prediction_parquet_evenet_weight"
            metrics["prediction_split_fraction_applied_in_export"] = None
        elif prediction_split_fraction is not None:
            if not (0.0 < prediction_split_fraction <= 1.0):
                raise ValueError(
                    f"--prediction-split-fraction must be in (0, 1], got {prediction_split_fraction}."
                )
            weight_scale[full_indices] = np.float32(1.0 / prediction_split_fraction)
            export_weight[full_indices] = original_weight[full_indices] * weight_scale[full_indices].astype(np.float64)
            metrics["prediction_weight_source"] = "central_weight_times_legacy_split_fraction"
            metrics["prediction_split_fraction_applied_in_export"] = prediction_split_fraction
        else:
            metrics["prediction_weight_source"] = "central_weight"
            metrics["prediction_split_fraction_applied_in_export"] = None

        if zero_unpredicted_selected_mc and selected_full_indices is not None and len(selected_full_indices) > 0:
            selected_mask = np.zeros(len(output), dtype=bool)
            selected_mask[selected_full_indices] = True
            predicted_mask = np.zeros(len(output), dtype=bool)
            predicted_mask[full_indices] = True
            selected_unpredicted_mask = selected_mask & ~predicted_mask
            if np.any(selected_unpredicted_mask):
                export_weight[selected_unpredicted_mask] = 0.0
                weight_scale[selected_unpredicted_mask] = 0.0
            metrics["selected_source_rows"] = int(np.sum(selected_mask))
            metrics["selected_source_rows_with_prediction"] = int(np.sum(predicted_mask & selected_mask))
            metrics["selected_source_rows_zero_weighted"] = int(np.sum(selected_unpredicted_mask))
        else:
            metrics["selected_source_rows"] = 0
            metrics["selected_source_rows_with_prediction"] = 0
            metrics["selected_source_rows_zero_weighted"] = 0
        output["evenet_qi_weight_scale"] = weight_scale
        output["weight"] = export_weight.astype(np.float32)

    metrics["num_raw_events"] = int(len(full_events))
    metrics["num_events_with_prediction"] = int(len(full_indices))
    metrics["prediction_coverage_fraction"] = float(len(full_indices) / len(full_events)) if len(full_events) else 0.0
    metrics["prediction_split_fraction"] = prediction_split_fraction
    metrics["legacy_prediction_weight_scale_request"] = float(1.0 / prediction_split_fraction) if prediction_split_fraction else 1.0
    metrics["valid_events"] = int(ak.sum(output["flags_valid"] > 0))
    metrics["valid_fraction"] = float(ak.sum(output["flags_valid"] > 0) / len(output)) if len(output) else 0.0
    if "central_weight" in output.fields and "weight" in output.fields:
        predicted_mask = ak.to_numpy(output["evenet_has_prediction"], allow_missing=False).astype(bool)
        effective_scale = ak.to_numpy(output["evenet_qi_weight_scale"], allow_missing=False).astype(np.float64)
        metrics["central_weight_sum_all"] = float(ak.sum(output["central_weight"]))
        metrics["export_weight_sum_all"] = float(ak.sum(output["weight"]))
        metrics["central_weight_sum_predicted"] = float(np.sum(ak.to_numpy(output["central_weight"], allow_missing=False)[predicted_mask]))
        metrics["export_weight_sum_predicted"] = float(np.sum(ak.to_numpy(output["weight"], allow_missing=False)[predicted_mask]))
        metrics["central_weight_sum_default"] = float(np.sum(ak.to_numpy(output["central_weight"], allow_missing=False)[~predicted_mask]))
        metrics["export_weight_sum_default"] = float(np.sum(ak.to_numpy(output["weight"], allow_missing=False)[~predicted_mask]))
        metrics["mean_prediction_weight_scale"] = float(np.mean(effective_scale[predicted_mask])) if np.any(predicted_mask) else 1.0
    return output, metrics


def finite_median(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


def mark_baseline_method(events: ak.Array) -> ak.Array:
    output = events
    output["neutrino_method"] = ak.Array(["central_baseline"] * len(output))
    return output


def add_evenet_region_cut_fields(events: ak.Array, regions: list[str]) -> ak.Array:
    output = events
    if "evenet_pred_class_name" not in output.fields:
        return output
    pred_class = np.asarray(ak.to_list(output["evenet_pred_class_name"]), dtype=object)
    has_prediction = (
        ak.to_numpy(output["evenet_has_prediction"], allow_missing=False).astype(bool)
        if "evenet_has_prediction" in output.fields
        else pred_class != ""
    )
    for region in regions:
        if not region.startswith("Ztautau_"):
            continue
        output[f"{region}_cut"] = (pred_class == region) & has_prediction
    return output


def write_qi_tree(events: ak.Array, output_root: Path, sample_name: str, regions: list[str]) -> dict[str, int]:
    sample_dir = output_root / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    events = add_evenet_region_cut_fields(events, regions)
    parquet_events = prepare_events_for_parquet(events)

    counts: dict[str, int] = {"raw": int(len(parquet_events))}
    ak.to_parquet(parquet_events, sample_dir / "filtered___raw.parquet", compression="snappy")

    for region in regions:
        if region == "raw":
            continue
        cut_field = f"{region}_cut"
        if cut_field not in parquet_events.fields:
            continue
        region_events = parquet_events[parquet_events[cut_field] == 1]
        counts[region] = int(len(region_events))
        ak.to_parquet(region_events, sample_dir / f"filtered___{region}.parquet", compression="snappy")
    return counts


def robust_event_keys(events: ak.Array) -> np.ndarray:
    field_sets = [
        ("run", "luminosityBlock", "evtNumber"),
        ("run", "lumi", "evtNumber"),
        ("Event_runNumber", "Event_lumiBlock", "Event_evtNumber"),
        ("Event_run", "Event_lumi", "Event_evtNumber"),
        ("evtNumber",),
        ("Event_evtNumber",),
    ]
    fields = set(events.fields)
    for field_set in field_sets:
        if set(field_set).issubset(fields):
            arrays = [ak.to_numpy(events[field], allow_missing=False) for field in field_set]
            if len(arrays) == 1:
                return arrays[0].astype(object)
            parts = [array.astype(str) for array in arrays]
            keys = parts[0]
            for part in parts[1:]:
                keys = np.char.add(np.char.add(keys, ":"), part)
            return keys.astype(object)
    return np.arange(len(events), dtype=np.int64).astype(object)


def map_by_event_key_occurrence(
    full_keys: np.ndarray,
    subset_keys: np.ndarray,
    context: str,
) -> np.ndarray:
    positions_by_key: dict[Any, list[int]] = {}
    for index, value in enumerate(full_keys.tolist()):
        positions_by_key.setdefault(value, []).append(index)

    occurrence_by_key: dict[Any, int] = {}
    indices: list[int] = []
    missing: list[Any] = []
    exhausted: list[Any] = []
    for value in subset_keys.tolist():
        occurrence = occurrence_by_key.get(value, 0)
        positions = positions_by_key.get(value)
        if positions is None:
            missing.append(value)
            indices.append(-1)
        elif occurrence >= len(positions):
            exhausted.append(value)
            indices.append(-1)
        else:
            indices.append(positions[occurrence])
        occurrence_by_key[value] = occurrence + 1

    if missing or exhausted:
        raise ValueError(
            f"Cannot map {context} back to full raw events. "
            f"missing_keys={missing[:5]}, exhausted_duplicate_keys={exhausted[:5]}."
        )
    return np.asarray(indices, dtype=np.int64)


def map_subset_rows_to_full(full_events: ak.Array, subset_events: ak.Array, context: str) -> np.ndarray:
    if len(subset_events) == len(full_events):
        return np.arange(len(full_events), dtype=np.int64)
    return map_by_event_key_occurrence(
        full_keys=robust_event_keys(full_events),
        subset_keys=robust_event_keys(subset_events),
        context=context,
    )


def map_prediction_rows_to_full(
    full_events: ak.Array,
    pred_source_events: ak.Array,
    allow_prefix_match: bool,
) -> np.ndarray:
    if len(pred_source_events) == len(full_events):
        return np.arange(len(full_events), dtype=np.int64)

    try:
        return map_subset_rows_to_full(full_events, pred_source_events, context="prediction rows")
    except ValueError:
        if not allow_prefix_match:
            raise

    if allow_prefix_match and len(full_events) > len(pred_source_events):
        return np.arange(len(pred_source_events), dtype=np.int64)

    raise ValueError(
        "Cannot map EveNet predictions back to full raw central events. Provide "
        "--prediction-central-parquet with evtNumber/Event_evtNumber, or pass --allow-prefix-match "
        "only if the prediction rows are exactly the first N rows of full raw. "
        f"full={len(full_events)}, prediction_source={len(pred_source_events)}."
    )


def prepare_merge_inputs(
    full_events: ak.Array,
    pred_events: ak.Array,
    prediction_central_events: ak.Array | None,
    allow_prefix_match: bool,
) -> tuple[ak.Array, np.ndarray]:
    if prediction_central_events is None:
        if len(pred_events) == len(full_events):
            return full_events, np.arange(len(full_events), dtype=np.int64)
        if allow_prefix_match and len(full_events) > len(pred_events):
            return full_events[: len(pred_events)], np.arange(len(pred_events), dtype=np.int64)
        raise ValueError(
            "EveNet prediction parquet does not cover the full raw central parquet. "
            "Pass --prediction-central-parquet for the selected/preprocessed rows so predictions "
            "can be merged back into full raw with default invalid values for the rest. "
            f"full={len(full_events)}, evenet={len(pred_events)}."
        )

    if len(prediction_central_events) != len(pred_events):
        raise ValueError(
            "--prediction-central-parquet must be row-aligned to --evenet-pred-parquet. "
            f"prediction_central={len(prediction_central_events)}, evenet={len(pred_events)}."
        )
    full_indices = map_prediction_rows_to_full(full_events, prediction_central_events, allow_prefix_match)
    return prediction_central_events, full_indices


def sample_raw_files(analysis_cfg: dict, sample_key: str, sample_name: str) -> list[str]:
    samples_cfg = analysis_cfg.get("Samples", {})
    sample_cfg = samples_cfg.get(sample_key)
    if sample_cfg is None:
        for cfg in samples_cfg.values():
            if cfg.get("name") == sample_name:
                sample_cfg = cfg
                break
    if sample_cfg is None:
        raise KeyError(f"Cannot find sample config for key/name '{sample_key}'/'{sample_name}'.")

    raw_files = sample_cfg.get("raw_files") or sample_cfg.get("raw_input_files")
    if raw_files is None:
        raise ValueError(
            f"Sample '{sample_name}' is missing raw_files in analysis.yaml. "
            "QI export needs full raw parquet so non-predicted events can remain as default invalid."
        )
    return expand_paths(tuple(raw_files))


def build_source_mapping(expanded_entries) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for source_index, entry in enumerate(expanded_entries):
        if isinstance(entry, tuple):
            expanded_sample, _source_events, parent_sample = entry
            entry = {
                "expanded_name": expanded_sample.name,
                "parent_key": expanded_sample.key,
                "parent_name": parent_sample.name,
                "is_data": bool(parent_sample.is_data),
            }
        mapping[source_index] = {
            "expanded_name": entry["expanded_name"],
            "parent_key": entry["parent_key"],
            "parent_name": entry["parent_name"],
            "is_data": bool(entry["is_data"]),
        }
    return mapping


def selected_events_by_source_index(analysis_config_path: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    analysis_cfg = read_yaml(analysis_config_path)
    samples, subcategories, _ = parse_config(analysis_config_path)
    selected_events = {
        sample_key: apply_preselection(load_concat_events(expand_paths(sample.input_files)))
        for sample_key, sample in samples.items()
    }
    expanded_samples = expand_samples(samples, selected_events, subcategories)
    mc_entries = []
    data_entries = []
    for expanded_sample, events in expanded_samples:
        parent_sample = samples[expanded_sample.key]
        entry = (expanded_sample, events, parent_sample)
        if parent_sample.is_data:
            data_entries.append(entry)
        else:
            mc_entries.append(entry)
    return {"mc": build_source_mapping(mc_entries), "data": build_source_mapping(data_entries)}, analysis_cfg


def require_source_columns(pred_events: ak.Array, pred_path: Path) -> None:
    missing = [
        field
        for field in ("source_sample_index",)
        if field not in pred_events.fields
    ]
    if missing:
        raise ValueError(
            f"{pred_path} is missing source columns {missing}. Rebuild the EveNet input and rerun "
            "prediction with the updated ml_pipeline scripts so rows can be grouped by sample safely."
        )


def config_prediction_paths(args: argparse.Namespace, analysis_cfg: dict) -> tuple[Path | None, Path | None]:
    pred_cfg = analysis_cfg.get("EveNetPrediction", {})
    mc_path = args.mc_pred_parquet or pred_cfg.get("mc_parquet") or pred_cfg.get("test_parquet")
    data_path = args.data_pred_parquet or pred_cfg.get("data_parquet")
    return (Path(mc_path) if mc_path else None, Path(data_path) if data_path else None)


def config_output_dir(args: argparse.Namespace, analysis_cfg: dict) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    pred_cfg = analysis_cfg.get("EveNetPrediction", {})
    output_dir = pred_cfg.get("qi_output_dir") or pred_cfg.get("output_dir")
    if output_dir is None:
        raise ValueError("Pass --output-dir or set EveNetPrediction.qi_output_dir in analysis.yaml.")
    return Path(output_dir)


def config_split_fraction(args: argparse.Namespace, analysis_cfg: dict) -> float | None:
    if args.prediction_split_fraction is not None:
        return args.prediction_split_fraction
    pred_cfg = analysis_cfg.get("EveNetPrediction", {})
    value = pred_cfg.get("mc_split_fraction") or pred_cfg.get("prediction_split_fraction")
    return float(value) if value is not None else None


def config_num_workers(args: argparse.Namespace, analysis_cfg: dict) -> int:
    if args.num_workers is not None:
        return max(1, int(args.num_workers))
    pred_cfg = analysis_cfg.get("EveNetPrediction", {})
    value = pred_cfg.get("num_workers") or pred_cfg.get("export_num_workers")
    return max(1, int(value)) if value is not None else 1


def config_worker_backend(args: argparse.Namespace, analysis_cfg: dict) -> str:
    if args.worker_backend is not None:
        return args.worker_backend
    pred_cfg = analysis_cfg.get("EveNetPrediction", {})
    return str(pred_cfg.get("worker_backend") or pred_cfg.get("export_worker_backend") or "thread")


def print_progress(label: str, done: int, total: int, detail: str = "") -> None:
    total = max(total, 1)
    width = 28
    filled = min(width, int(round(width * done / total)))
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {detail}" if detail else ""
    print(f"[export-evenet-to-qi] {label} [{bar}] {done}/{total}{suffix}", flush=True)


def merge_prediction_group(
    full_events: ak.Array,
    source_events: ak.Array,
    pred_events: ak.Array,
    prediction_split_fraction: float | None,
) -> tuple[ak.Array, dict[str, Any]]:
    full_indices = map_prediction_rows_to_full(full_events, source_events, allow_prefix_match=False)
    return with_evenet_reconstruction(
        full_events=full_events,
        central_pred_events=source_events,
        pred_events=pred_events,
        full_indices=full_indices,
        prediction_split_fraction=prediction_split_fraction,
    )


def build_compact_base_events(events: ak.Array, fallback_weight: np.ndarray | None = None) -> ak.Array:
    columns = compact_export_fields(events)
    num_events = len(events)
    if "weight" not in columns:
        if fallback_weight is not None:
            columns["weight"] = np.asarray(fallback_weight, dtype=np.float32)
        else:
            columns["weight"] = np.ones(num_events, dtype=np.float32)
    if "central_weight" not in columns and "weight" in columns:
        columns["central_weight"] = ak.to_numpy(columns["weight"], allow_missing=False).astype(np.float32)
    return ak.Array(columns)


def require_concat_prediction_columns(pred_events: ak.Array, is_data: bool) -> None:
    required = {"initial_total_num_events", "source_slot_for_a", "source_slot_for_b", "tau_vis_prong_slot0_valid"}
    if not is_data:
        required.update({"event_category", "truth_QI_region", "analyzing_power_a", "analyzing_power_b"})
        required.update({f"truth_{name}" for name in get_observable_names()})
    missing = sorted(field for field in required if field not in pred_events.fields)
    if missing:
        raise ValueError(
            "Prediction parquet is missing fields required for concat-based unfolding export: "
            f"{missing[:10]}. Rebuild the EveNet input, preprocess parquet, and prediction parquet "
            "with the updated ml_pipeline scripts so the selected-source rows carry the needed central truth/cut metadata."
        )


def build_concat_prediction_rows(
    pred_events: ak.Array,
    prediction_split_fraction: float | None,
    is_data: bool,
) -> tuple[ak.Array, dict[str, Any]]:
    if len(pred_events) == 0:
        raise ValueError("Prediction parquet group is empty; cannot build concat rows.")
    require_concat_prediction_columns(pred_events, is_data=is_data)

    fallback_weight = (
        ak.to_numpy(pred_events["central_weight"], allow_missing=False).astype(np.float32)
        if "central_weight" in pred_events.fields
        else ak.to_numpy(pred_events["event_weight"], allow_missing=False).astype(np.float32)
        if "event_weight" in pred_events.fields
        else np.ones(len(pred_events), dtype=np.float32)
    )
    base_events = build_compact_base_events(pred_events, fallback_weight=fallback_weight)
    full_indices = np.arange(len(base_events), dtype=np.int64)
    return with_evenet_reconstruction(
        full_events=base_events,
        central_pred_events=base_events,
        pred_events=pred_events,
        full_indices=full_indices,
        prediction_split_fraction=prediction_split_fraction,
        selected_full_indices=full_indices,
        zero_unpredicted_selected_mc=False,
    )


def build_concat_raw_complement_rows(
    raw_events: ak.Array,
    pred_template_events: ak.Array,
) -> tuple[ak.Array, dict[str, Any]]:
    compact_raw = build_compact_base_events(raw_events)
    return with_evenet_reconstruction(
        full_events=compact_raw,
        central_pred_events=pred_template_events[:0],
        pred_events=pred_template_events[:0],
        full_indices=np.array([], dtype=np.int64),
        prediction_split_fraction=None,
        selected_full_indices=None,
        zero_unpredicted_selected_mc=False,
    )


def export_config_group(
    parent_name: str,
    group: dict[str, Any],
    analysis_cfg: dict,
    output_root: Path,
    regions: list[str],
    prediction_split_fraction: float | None,
    output_label: str,
) -> tuple[str, dict[str, Any]]:
    print(f"[export-evenet-to-qi] worker {os.getpid()} start {parent_name}", flush=True)
    raw_files = sample_raw_files(analysis_cfg, group["parent_key"], parent_name)
    full_events = load_concat_events(raw_files)
    selected_mask = prediction_selection_mask(full_events)
    outside_selected_events = full_events[~selected_mask]
    print(
        f"[export-evenet-to-qi] worker {os.getpid()} loaded {parent_name} raw_events={len(full_events)} "
        f"selected_for_prediction={int(np.sum(selected_mask))} outside_selected={len(outside_selected_events)}",
        flush=True,
    )
    split_fraction = None if group["is_data"] else prediction_split_fraction
    pred_group = ak.concatenate(group["pred_parts"], axis=0) if len(group["pred_parts"]) > 1 else group["pred_parts"][0]
    predicted_events, predicted_metrics = build_concat_prediction_rows(
        pred_events=pred_group,
        prediction_split_fraction=split_fraction,
        is_data=group["is_data"],
    )
    outside_events, outside_metrics = build_concat_raw_complement_rows(
        raw_events=outside_selected_events,
        pred_template_events=pred_group,
    )
    outside_events, predicted_events = align_fields_for_concat([outside_events, predicted_events])
    evenet_events = ak.concatenate([outside_events, predicted_events], axis=0)
    metrics = {
        "num_raw_events": int(len(full_events)),
        "num_raw_outside_selected": int(len(outside_events)),
        "num_predicted_events": int(len(predicted_events)),
        "prediction_split_fraction": split_fraction,
        "prediction_parquet_rows": int(len(pred_group)),
        "concat_mode": "raw_outside_selected_plus_prediction_rows",
        "prediction_metrics": predicted_metrics,
        "outside_selected_metrics": outside_metrics,
    }
    print(
        f"[export-evenet-to-qi] worker {os.getpid()} concat {parent_name} raw_outside={len(outside_events)} "
        f"predicted={len(predicted_events)} total={len(evenet_events)}",
        flush=True,
    )
    sample_output_root = output_root / output_label
    counts = write_qi_tree(evenet_events, sample_output_root, parent_name, regions)
    print(f"[export-evenet-to-qi] worker {os.getpid()} wrote {parent_name}", flush=True)
    metrics["region_counts"] = counts
    metrics["parent_sample"] = parent_name
    metrics["expanded_samples"] = group["expanded_samples"]
    metrics["raw_files"] = raw_files
    metrics["is_data"] = group["is_data"]
    metrics["worker_pid"] = os.getpid()
    return parent_name, metrics


def export_config_group_worker(payload: tuple[str, dict[str, Any], dict, Path, list[str], float | None, str]) -> tuple[str, dict[str, Any]]:
    return export_config_group(*payload)


def export_config_prediction(
    pred_path: Path,
    source_mapping: dict[int, dict[str, Any]],
    analysis_cfg: dict,
    output_root: Path,
    regions: list[str],
    prediction_split_fraction: float | None,
    output_label: str,
    summary_label: str,
    num_workers: int,
    worker_backend: str,
) -> dict[str, Any]:
    pred_events = load_events(pred_path)
    require_source_columns(pred_events, pred_path)

    source_indices = ak.to_numpy(pred_events["source_sample_index"], allow_missing=False).astype(np.int64)
    summary: dict[str, Any] = {
        "prediction_parquet": str(pred_path),
        "output_label": output_label,
        "summary_label": summary_label,
        "samples": {},
    }
    grouped: dict[str, dict[str, Any]] = {}

    for source_index in sorted(np.unique(source_indices).tolist()):
        if int(source_index) not in source_mapping:
            raise ValueError(f"{pred_path} references source_sample_index={source_index}, absent from analysis.yaml expansion.")
        source_info = source_mapping[int(source_index)]
        parent_name = source_info["parent_name"]
        parent_key = source_info["parent_key"]
        row_mask = source_indices == int(source_index)
        pred_subset = pred_events[row_mask]

        group = grouped.setdefault(
            parent_name,
            {
                "parent_key": parent_key,
                "parent_name": parent_name,
                "is_data": source_info["is_data"],
                "pred_parts": [],
                "source_infos": [],
                "expanded_samples": [],
            },
        )
        group["pred_parts"].append(pred_subset)
        group["source_infos"].append(source_info)
        group["expanded_samples"].append(source_info["expanded_name"])

    group_items = list(grouped.items())
    total_groups = len(group_items)
    worker_count = min(max(1, int(num_workers)), max(total_groups, 1))
    backend = "process" if worker_backend == "process" else "thread"
    print_progress(f"{summary_label} export", 0, total_groups, f"workers={worker_count} backend={backend}")
    if worker_count == 1 or total_groups <= 1:
        for done, (parent_name, group) in enumerate(group_items, start=1):
            print(f"[export-evenet-to-qi] start {summary_label}:{parent_name}", flush=True)
            _, metrics = export_config_group(
                parent_name=parent_name,
                group=group,
                analysis_cfg=analysis_cfg,
                output_root=output_root,
                regions=regions,
                prediction_split_fraction=prediction_split_fraction,
                output_label=output_label,
            )
            summary["samples"][parent_name] = metrics
            print_progress(f"{summary_label} export", done, total_groups, parent_name)
    else:
        payloads = [
            (parent_name, group, analysis_cfg, output_root, regions, prediction_split_fraction, output_label)
            for parent_name, group in group_items
        ]
        executor_class = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
        with executor_class(max_workers=worker_count) as executor:
            future_to_parent = {
                executor.submit(export_config_group_worker, payload): payload[0]
                for payload in payloads
            }
            done = 0
            for future in as_completed(future_to_parent):
                parent_name = future_to_parent[future]
                result_parent_name, metrics = future.result()
                summary["samples"][result_parent_name] = metrics
                done += 1
                print_progress(f"{summary_label} export", done, total_groups, parent_name)

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f"{output_label}__{summary_label}_analysis_config_export_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run_config_mode(args: argparse.Namespace) -> None:
    if args.analysis_config is None:
        raise ValueError("--analysis-config is required for config-driven mode.")
    source_mappings, analysis_cfg = selected_events_by_source_index(args.analysis_config)
    output_root = config_output_dir(args, analysis_cfg)
    pred_cfg = analysis_cfg.get("EveNetPrediction", {})
    regions = list(pred_cfg.get("regions", args.regions))
    output_label = str(args.qi_method_label or pred_cfg.get("qi_method_label", "evenet"))
    mc_pred, data_pred = config_prediction_paths(args, analysis_cfg)
    split_fraction = config_split_fraction(args, analysis_cfg)
    num_workers = config_num_workers(args, analysis_cfg)
    worker_backend = config_worker_backend(args, analysis_cfg)

    summaries: dict[str, Any] = {}
    if mc_pred is not None:
        summaries["mc"] = export_config_prediction(
            pred_path=mc_pred,
            source_mapping=source_mappings["mc"],
            analysis_cfg=analysis_cfg,
            output_root=output_root,
            regions=regions,
            prediction_split_fraction=split_fraction,
            output_label=output_label,
            summary_label="mc",
            num_workers=num_workers,
            worker_backend=worker_backend,
        )
    if data_pred is not None:
        summaries["data"] = export_config_prediction(
            pred_path=data_pred,
            source_mapping=source_mappings["data"],
            analysis_cfg=analysis_cfg,
            output_root=output_root,
            regions=regions,
            prediction_split_fraction=None,
            output_label=output_label,
            summary_label="data",
            num_workers=num_workers,
            worker_backend=worker_backend,
        )
    if not summaries:
        raise ValueError(
            "No prediction parquet configured. Pass --mc-pred-parquet/--data-pred-parquet "
            "or set EveNetPrediction.mc_parquet/data_parquet in analysis.yaml."
        )
    with (output_root / "analysis_config_export_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    print(f"[export-evenet-to-qi] wrote config-driven exports under {output_root}", flush=True)


def main() -> None:
    args = parse_args()

    if args.analysis_config is not None and (
        args.mc_pred_parquet is not None
        or args.data_pred_parquet is not None
        or read_yaml(args.analysis_config).get("EveNetPrediction") is not None
    ):
        run_config_mode(args)
        return

    required_low_level = {
        "--central-parquet": args.central_parquet,
        "--evenet-pred-parquet": args.evenet_pred_parquet,
        "--output-dir": args.output_dir,
    }
    missing_low_level = [name for name, value in required_low_level.items() if value is None]
    if missing_low_level:
        raise ValueError(
            "Low-level export mode requires "
            + ", ".join(missing_low_level)
            + ". Prefer config mode with --analysis-config plus --mc-pred-parquet/--data-pred-parquet."
        )

    central_events = load_events(args.central_parquet)
    baseline_events = load_events(args.central_parquet) if args.write_baseline_copy else None
    prediction_central_events = load_events(args.prediction_central_parquet) if args.prediction_central_parquet else None
    pred_events = load_events(args.evenet_pred_parquet)
    central_pred_events, full_indices = prepare_merge_inputs(
        central_events,
        pred_events,
        prediction_central_events,
        allow_prefix_match=bool(args.allow_prefix_match),
    )

    evenet_events, metrics = with_evenet_reconstruction(
        central_events,
        central_pred_events,
        pred_events,
        full_indices,
        prediction_split_fraction=args.prediction_split_fraction,
    )
    method_output_root = args.output_dir / args.method_output_subdir if args.method_output_subdir else args.output_dir
    counts = write_qi_tree(evenet_events, method_output_root, args.sample_name, list(args.regions))
    metrics["region_counts"] = counts
    metrics["central_parquet"] = str(args.central_parquet)
    metrics["prediction_central_parquet"] = str(args.prediction_central_parquet) if args.prediction_central_parquet else None
    metrics["evenet_pred_parquet"] = str(args.evenet_pred_parquet)
    metrics["sample_name"] = args.sample_name

    method_output_root.mkdir(parents=True, exist_ok=True)
    with (method_output_root / f"{args.sample_name}__evenet_qi_export_summary.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    if args.write_baseline_copy:
        if baseline_events is None:
            raise RuntimeError("Internal error: baseline_events was not loaded.")
        baseline_root = args.baseline_output_dir or (args.output_dir / "baseline")
        baseline_counts = write_qi_tree(mark_baseline_method(baseline_events), baseline_root, args.sample_name, list(args.regions))
        with (baseline_root / f"{args.sample_name}__baseline_qi_export_summary.json").open("w") as handle:
            json.dump(
                {
                    "num_events": int(len(baseline_events)),
                    "region_counts": baseline_counts,
                    "central_parquet": str(args.central_parquet),
                    "sample_name": args.sample_name,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    print(
        "[export-evenet-to-qi] wrote "
        f"{method_output_root / args.sample_name} with valid_fraction={metrics['valid_fraction']:.4f} "
        f"slot_swap_fraction={metrics['slot_swap_fraction']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
