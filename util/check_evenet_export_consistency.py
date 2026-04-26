#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import vector


DEFAULT_FLOAT = -99.0
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

vector.register_awkward()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether exported EveNet raw parquet is consistent with the standalone prediction parquet."
    )
    parser.add_argument("--prediction-parquet", type=Path, required=True)
    parser.add_argument("--export-raw-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-scatter", type=int, default=30000)
    return parser.parse_args()


def to_numpy(values: ak.Array, dtype=None) -> np.ndarray:
    output = ak.to_numpy(values, allow_missing=False)
    if dtype is not None:
        output = output.astype(dtype)
    return output


def export_vector_component(values: ak.Array, component: str) -> np.ndarray:
    fields = set(getattr(values, "fields", []))
    px = None
    py = None
    pz = None
    energy = None
    if "px" in fields:
        px = to_numpy(values["px"], np.float64)
    elif "x" in fields:
        px = to_numpy(values["x"], np.float64)
    if "py" in fields:
        py = to_numpy(values["py"], np.float64)
    elif "y" in fields:
        py = to_numpy(values["y"], np.float64)
    if "pz" in fields:
        pz = to_numpy(values["pz"], np.float64)
    elif "z" in fields:
        pz = to_numpy(values["z"], np.float64)
    if "E" in fields:
        energy = to_numpy(values["E"], np.float64)
    elif "t" in fields:
        energy = to_numpy(values["t"], np.float64)

    if component == "E":
        if energy is not None:
            return energy
    if component == "px":
        if px is not None:
            return px
    if component == "py":
        if py is not None:
            return py
    if component == "pz":
        if pz is not None:
            return pz
    if component == "energy":
        if energy is not None:
            return energy
    if component == "pt" and px is not None and py is not None:
        return np.sqrt(px ** 2 + py ** 2)
    if component == "eta" and px is not None and py is not None and pz is not None:
        pt = np.sqrt(px ** 2 + py ** 2)
        return np.arcsinh(np.divide(pz, np.maximum(pt, 1e-8)))
    if component == "phi" and px is not None and py is not None:
        return np.arctan2(py, px)
    raise KeyError(f"Unable to extract component '{component}' from vector fields={sorted(fields)}")


def prediction_slot_p4(pred_events: ak.Array, prefix: str, slot: int) -> tuple[ak.Array | None, np.ndarray]:
    valid_field = f"{prefix}_slot{slot}_valid"
    valid = (
        to_numpy(pred_events[valid_field], bool)
        if valid_field in pred_events.fields
        else np.ones(len(pred_events), dtype=bool)
    )
    fields = set(pred_events.fields)

    def component(name: str) -> np.ndarray:
        return to_numpy(pred_events[f"{prefix}_slot{slot}_{name}"], np.float64)

    if all(f"{prefix}_slot{slot}_{name}" in fields for name in ("E", "px", "py", "pz")):
        return vector.zip({"px": component("px"), "py": component("py"), "pz": component("pz"), "E": component("E")}), valid

    if all(f"{prefix}_slot{slot}_{name}" in fields for name in ("energy", "pt", "eta", "phi")):
        return vector.zip(
            {
                "pt": component("pt"),
                "eta": component("eta"),
                "phi": component("phi"),
                "E": component("energy"),
            }
        ), valid

    if all(f"{prefix}_slot{slot}_{name}" in fields for name in ("log_energy", "log_pt", "eta", "phi")):
        return vector.zip(
            {
                "pt": np.expm1(component("log_pt")),
                "eta": component("eta"),
                "phi": component("phi"),
                "E": np.expm1(component("log_energy")),
            }
        ), valid

    return None, valid


def p4_component(p4: ak.Array, component: str) -> np.ndarray:
    if component in {"E", "energy"}:
        return to_numpy(p4.E, np.float64)
    if component == "px":
        return to_numpy(p4.px, np.float64)
    if component == "py":
        return to_numpy(p4.py, np.float64)
    if component == "pz":
        return to_numpy(p4.pz, np.float64)
    if component == "pt":
        return to_numpy(p4.pt, np.float64)
    if component == "eta":
        return to_numpy(p4.eta, np.float64)
    if component == "phi":
        return to_numpy(p4.phi, np.float64)
    raise KeyError(f"Unsupported p4 component '{component}'")


def composite_key(events: ak.Array) -> np.ndarray:
    fields = set(events.fields)
    key_parts: list[np.ndarray] = []
    for field in ("source_sample_index", "source_event_key", "source_event_index"):
        if field in fields:
            key_parts.append(to_numpy(events[field]))
    if not key_parts and "event_index" in fields:
        key_parts.append(to_numpy(events["event_index"]))
    if not key_parts:
        return np.arange(len(events), dtype=np.int64).astype(object)

    output = key_parts[0].astype(str)
    for part in key_parts[1:]:
        output = np.char.add(np.char.add(output, ":"), part.astype(str))
    return output.astype(object)


def infer_sample_and_region_from_export_path(export_path: Path) -> tuple[str | None, str | None]:
    sample_name = export_path.parent.name if export_path.parent is not None else None
    region_name = None
    stem = export_path.stem
    if stem.startswith("filtered___"):
        region_name = stem.removeprefix("filtered___")
    return sample_name, region_name


def prediction_sample_mask(pred_events: ak.Array, sample_name: str | None) -> np.ndarray:
    if sample_name is None or "evenet_truth_class_name" not in pred_events.fields:
        return np.ones(len(pred_events), dtype=bool)
    truth_name = np.asarray(ak.to_list(pred_events["evenet_truth_class_name"]), dtype=object)
    if sample_name == "Ztautau":
        return np.char.startswith(truth_name.astype(str), "Ztautau_")
    return truth_name.astype(str) == sample_name


def prediction_region_mask(pred_events: ak.Array, region_name: str | None) -> np.ndarray:
    if region_name is None:
        return np.ones(len(pred_events), dtype=bool)
    if f"{region_name}_cut" in pred_events.fields:
        return to_numpy(pred_events[f"{region_name}_cut"], bool)
    if "evenet_pred_class_name" in pred_events.fields and region_name.startswith("Ztautau_"):
        pred_name = np.asarray(ak.to_list(pred_events["evenet_pred_class_name"]), dtype=object).astype(str)
        valid = to_numpy(pred_events["flags_valid"], bool) if "flags_valid" in pred_events.fields else np.ones(len(pred_events), dtype=bool)
        return (pred_name == region_name) & valid
    return np.ones(len(pred_events), dtype=bool)


def prediction_subset_for_export_region(pred_events: ak.Array, export_path: Path | None) -> ak.Array:
    if export_path is None:
        return pred_events
    sample_name, region_name = infer_sample_and_region_from_export_path(export_path)
    sample_mask = prediction_sample_mask(pred_events, sample_name)
    region_mask = prediction_region_mask(pred_events, region_name)
    return pred_events[sample_mask & region_mask]


def align_events(
    pred_events: ak.Array,
    export_events: ak.Array,
    export_path: Path | None = None,
) -> tuple[ak.Array, ak.Array, dict[str, Any]]:
    export_mask = (
        to_numpy(export_events["evenet_has_prediction"], bool)
        if "evenet_has_prediction" in export_events.fields
        else np.ones(len(export_events), dtype=bool)
    )
    export_pred = export_events[export_mask]

    pred_keys = composite_key(pred_events)
    export_keys = composite_key(export_pred)
    export_lookup = {key: index for index, key in enumerate(export_keys.tolist())}
    matched_export_indices = np.array([export_lookup.get(key, -1) for key in pred_keys.tolist()], dtype=np.int64)
    matched_mask = matched_export_indices >= 0

    missing_pred = int(np.sum(~matched_mask))
    extra_export = int(len(export_pred) - np.sum(matched_mask))

    aligned_pred = pred_events[matched_mask]
    aligned_export = export_pred[matched_export_indices[matched_mask]]
    summary = {
        "mode": "key_match",
        "prediction_rows": int(len(pred_events)),
        "export_predicted_rows": int(len(export_pred)),
        "matched_rows": int(len(aligned_pred)),
        "missing_prediction_rows_in_export": missing_pred,
        "extra_export_rows": extra_export,
    }
    if len(aligned_pred) > 0 or export_path is None:
        return aligned_pred, aligned_export, summary

    sample_name, region_name = infer_sample_and_region_from_export_path(export_path)
    sample_mask = prediction_sample_mask(pred_events, sample_name)
    region_mask = prediction_region_mask(pred_events, region_name)
    pred_subset = pred_events[sample_mask & region_mask]
    if "source_sample_index" in pred_subset.fields and "source_sample_index" in export_pred.fields:
        pred_source = to_numpy(pred_subset["source_sample_index"], np.int64)
        export_source = to_numpy(export_pred["source_sample_index"], np.int64)
        shared = sorted(set(pred_source.tolist()) & set(export_source.tolist()))
        pred_parts = []
        export_parts = []
        per_source: dict[str, Any] = {}
        for source_index in shared:
            pred_part = pred_subset[pred_source == source_index]
            export_part = export_pred[export_source == source_index]
            row_count = min(len(pred_part), len(export_part))
            if row_count <= 0:
                continue
            pred_parts.append(pred_part[:row_count])
            export_parts.append(export_part[:row_count])
            per_source[str(int(source_index))] = {
                "prediction_rows": int(len(pred_part)),
                "export_rows": int(len(export_part)),
                "matched_rows": int(row_count),
            }

        if pred_parts and export_parts:
            aligned_pred = pred_parts[0] if len(pred_parts) == 1 else ak.concatenate(pred_parts, axis=0)
            aligned_export = export_parts[0] if len(export_parts) == 1 else ak.concatenate(export_parts, axis=0)
            summary = {
                "mode": "source_sample_index_region_order_fallback",
                "prediction_rows": int(len(pred_events)),
                "export_predicted_rows": int(len(export_pred)),
                "sample_name": sample_name,
                "region_name": region_name,
                "prediction_subset_rows": int(len(pred_subset)),
                "matched_rows": int(len(aligned_pred)),
                "missing_prediction_rows_in_export": int(len(pred_subset) - len(aligned_pred)),
                "extra_export_rows": int(len(export_pred) - len(aligned_export)),
                "per_source_sample_index": per_source,
            }
            return aligned_pred, aligned_export, summary

    row_count = min(len(pred_subset), len(export_pred))
    aligned_pred = pred_subset[:row_count]
    aligned_export = export_pred[:row_count]
    summary = {
        "mode": "region_order_fallback",
        "prediction_rows": int(len(pred_events)),
        "export_predicted_rows": int(len(export_pred)),
        "sample_name": sample_name,
        "region_name": region_name,
        "prediction_subset_rows": int(len(pred_subset)),
        "matched_rows": int(row_count),
        "missing_prediction_rows_in_export": int(max(len(pred_subset) - row_count, 0)),
        "extra_export_rows": int(max(len(export_pred) - row_count, 0)),
    }
    return aligned_pred, aligned_export, summary


def weighted_yields(events: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
    weights = (
        to_numpy(events["evenet_weight"], np.float64)
        if "evenet_weight" in events.fields
        else np.ones(len(events), dtype=np.float64)
    )
    unique_names = np.array(sorted(set(names.tolist())), dtype=object)
    yields = np.array([float(np.sum(weights[names == name])) for name in unique_names], dtype=np.float64)
    return unique_names, yields


def plot_class_yields(pred_events: ak.Array, export_events: ak.Array, output_path: Path) -> dict[str, Any]:
    pred_names, pred_yields = weighted_yields(pred_events)
    export_names, export_yields = weighted_yields(export_events)
    all_names = np.array(sorted(set(pred_names.tolist()) | set(export_names.tolist())), dtype=object)
    pred_map = {name: value for name, value in zip(pred_names.tolist(), pred_yields.tolist())}
    export_map = {name: value for name, value in zip(export_names.tolist(), export_yields.tolist())}
    pred_values = np.array([pred_map.get(name, 0.0) for name in all_names], dtype=np.float64)
    export_values = np.array([export_map.get(name, 0.0) for name in all_names], dtype=np.float64)

    x = np.arange(len(all_names))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(all_names)), 6), dpi=220)
    ax.bar(x - width / 2, pred_values, width=width, color=OKABE_ITO[0], label="Prediction parquet")
    ax.bar(x + width / 2, export_values, width=width, color=OKABE_ITO[1], label="Export raw predicted rows")
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha="right")
    ax.set_ylabel("Weighted yield")
    ax.set_title("Predicted-class yield consistency")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        str(name): {
            "prediction_yield": float(pred_map.get(name, 0.0)),
            "export_yield": float(export_map.get(name, 0.0)),
            "difference": float(export_map.get(name, 0.0) - pred_map.get(name, 0.0)),
        }
        for name in all_names.tolist()
    }


def choose_sample_indices(num_points: int, max_points: int) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    return np.linspace(0, num_points - 1, max_points, dtype=np.int64)


def prediction_field_component(pred_events: ak.Array, prefix: str, slot: int, component: str) -> np.ndarray | None:
    field_name = f"{prefix}_slot{slot}_{component}"
    if field_name not in pred_events.fields:
        return None
    return to_numpy(pred_events[field_name], np.float64)


def scatter_consistency_plot(
    pred_events: ak.Array,
    export_events: ak.Array,
    field_pairs: list[tuple[str, str, str]],
    output_path: Path,
    max_scatter: int,
    title: str,
) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=220)
    metrics: dict[str, Any] = {}

    for axis, (pred_field, export_field, label) in zip(axes.flat, field_pairs):
        if pred_field not in pred_events.fields or export_field not in export_events.fields:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "missing"}
            continue

        pred_values = to_numpy(pred_events[pred_field], np.float64)
        export_values = to_numpy(export_events[export_field], np.float64)
        valid = np.isfinite(pred_values) & np.isfinite(export_values)
        valid &= ~np.isclose(pred_values, DEFAULT_FLOAT)
        valid &= ~np.isclose(export_values, DEFAULT_FLOAT)
        if not np.any(valid):
            axis.text(0.5, 0.5, "no valid entries", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "no_valid_entries"}
            continue

        pred_values = pred_values[valid]
        export_values = export_values[valid]
        sampled = choose_sample_indices(len(pred_values), max_scatter)
        axis.scatter(pred_values[sampled], export_values[sampled], s=4, alpha=0.25, color=OKABE_ITO[0], linewidths=0)

        low = float(min(np.min(pred_values), np.min(export_values)))
        high = float(max(np.max(pred_values), np.max(export_values)))
        if np.isfinite(low) and np.isfinite(high):
            axis.plot([low, high], [low, high], linestyle="--", color=OKABE_ITO[3], linewidth=1.2)

        axis.set_title(label)
        axis.set_xlabel("Prediction parquet")
        axis.set_ylabel("Export raw")
        axis.grid(alpha=0.2)
        metrics[label] = {
            "status": "ok",
            "num_valid_entries": int(len(pred_values)),
            "max_abs_diff": float(np.max(np.abs(export_values - pred_values))),
            "mean_abs_diff": float(np.mean(np.abs(export_values - pred_values))),
        }

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def scatter_consistency_arrays(
    array_pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]],
    output_path: Path,
    max_scatter: int,
    title: str,
    x_label: str,
    y_label: str,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=220)
    metrics: dict[str, Any] = {}

    for axis, (x_values, y_values, label) in zip(axes.flat, array_pairs):
        if x_values is None or y_values is None:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "missing"}
            continue

        valid = np.isfinite(x_values) & np.isfinite(y_values)
        valid &= ~np.isclose(x_values, DEFAULT_FLOAT)
        valid &= ~np.isclose(y_values, DEFAULT_FLOAT)
        if valid_mask is not None:
            valid &= valid_mask
        if not np.any(valid):
            axis.text(0.5, 0.5, "no valid entries", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "no_valid_entries"}
            continue

        x_valid = x_values[valid]
        y_valid = y_values[valid]
        sampled = choose_sample_indices(len(x_valid), max_scatter)
        axis.scatter(x_valid[sampled], y_valid[sampled], s=4, alpha=0.25, color=OKABE_ITO[0], linewidths=0)

        low = float(min(np.min(x_valid), np.min(y_valid)))
        high = float(max(np.max(x_valid), np.max(y_valid)))
        if np.isfinite(low) and np.isfinite(high):
            axis.plot([low, high], [low, high], linestyle="--", color=OKABE_ITO[3], linewidth=1.2)

        axis.set_title(label)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
        metrics[label] = {
            "status": "ok",
            "num_valid_entries": int(len(x_valid)),
            "max_abs_diff": float(np.max(np.abs(y_valid - x_valid))),
            "mean_abs_diff": float(np.mean(np.abs(y_valid - x_valid))),
        }

    for axis in axes.flat[len(array_pairs):]:
        axis.text(0.5, 0.5, "not requested", ha="center", va="center", fontsize=10, transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def build_expected_ab_fields(pred_events: ak.Array) -> dict[str, np.ndarray]:
    if "source_slot_for_a" not in pred_events.fields or "source_slot_for_b" not in pred_events.fields:
        return {}
    slot_for_a = to_numpy(pred_events["source_slot_for_a"], np.int64)
    slot_for_b = to_numpy(pred_events["source_slot_for_b"], np.int64)
    fields = set(pred_events.fields)

    def choose(prefix: str, slot_indices: np.ndarray, component: str) -> np.ndarray | None:
        slot0_name = f"{prefix}_slot0_{component}"
        slot1_name = f"{prefix}_slot1_{component}"
        if slot0_name not in fields or slot1_name not in fields:
            return None
        slot0 = to_numpy(pred_events[slot0_name], np.float64)
        slot1 = to_numpy(pred_events[slot1_name], np.float64)
        return np.where(slot_indices == 0, slot0, slot1)

    candidates = {
        "expected_lead_a_visible_energy": ("tau_vis_prong", slot_for_a, "energy"),
        "expected_lead_a_visible_pt": ("tau_vis_prong", slot_for_a, "pt"),
        "expected_lead_a_visible_eta": ("tau_vis_prong", slot_for_a, "eta"),
        "expected_lead_a_visible_phi": ("tau_vis_prong", slot_for_a, "phi"),
        "expected_lead_b_visible_energy": ("tau_vis_prong", slot_for_b, "energy"),
        "expected_lead_b_visible_pt": ("tau_vis_prong", slot_for_b, "pt"),
        "expected_lead_b_visible_eta": ("tau_vis_prong", slot_for_b, "eta"),
        "expected_lead_b_visible_phi": ("tau_vis_prong", slot_for_b, "phi"),
        "expected_lead_a_missing_E": ("pred_invisible", slot_for_a, "E"),
        "expected_lead_a_missing_energy": ("pred_invisible", slot_for_a, "energy"),
        "expected_lead_a_missing_pt": ("pred_invisible", slot_for_a, "pt"),
        "expected_lead_a_missing_eta": ("pred_invisible", slot_for_a, "eta"),
        "expected_lead_a_missing_phi": ("pred_invisible", slot_for_a, "phi"),
        "expected_lead_a_missing_px": ("pred_invisible", slot_for_a, "px"),
        "expected_lead_a_missing_py": ("pred_invisible", slot_for_a, "py"),
        "expected_lead_a_missing_pz": ("pred_invisible", slot_for_a, "pz"),
        "expected_lead_b_missing_E": ("pred_invisible", slot_for_b, "E"),
        "expected_lead_b_missing_energy": ("pred_invisible", slot_for_b, "energy"),
        "expected_lead_b_missing_pt": ("pred_invisible", slot_for_b, "pt"),
        "expected_lead_b_missing_eta": ("pred_invisible", slot_for_b, "eta"),
        "expected_lead_b_missing_phi": ("pred_invisible", slot_for_b, "phi"),
        "expected_lead_b_missing_px": ("pred_invisible", slot_for_b, "px"),
        "expected_lead_b_missing_py": ("pred_invisible", slot_for_b, "py"),
        "expected_lead_b_missing_pz": ("pred_invisible", slot_for_b, "pz"),
    }
    output: dict[str, np.ndarray] = {}
    for output_name, (prefix, slot_indices, component) in candidates.items():
        values = choose(prefix, slot_indices, component)
        if values is not None:
            output[output_name] = values
    return output


def build_ab_truth_pred_fields(pred_events: ak.Array) -> dict[str, np.ndarray]:
    if "source_slot_for_a" not in pred_events.fields or "source_slot_for_b" not in pred_events.fields:
        return {}
    slot_for_a = to_numpy(pred_events["source_slot_for_a"], np.int64)
    slot_for_b = to_numpy(pred_events["source_slot_for_b"], np.int64)

    def choose(prefix: str, slot_indices: np.ndarray, component: str) -> np.ndarray | None:
        slot0 = prediction_field_component(pred_events, prefix, 0, component)
        slot1 = prediction_field_component(pred_events, prefix, 1, component)
        if slot0 is None or slot1 is None:
            return None
        return np.where(slot_indices == 0, slot0, slot1)

    output: dict[str, np.ndarray] = {}
    for leg_name, slot_indices in (("a", slot_for_a), ("b", slot_for_b)):
        for component in ("energy", "pt", "eta", "phi"):
            pred_values = choose("pred_invisible", slot_indices, component)
            truth_values = choose("target_invisible", slot_indices, component)
            if pred_values is not None:
                output[f"pred_lead_{leg_name}_missing_{component}"] = pred_values
            if truth_values is not None:
                output[f"truth_lead_{leg_name}_missing_{component}"] = truth_values
    return output


def plot_ab_truth_vs_pred(
    pred_events: ak.Array,
    output_path: Path,
    max_scatter: int,
) -> dict[str, Any]:
    remapped = build_ab_truth_pred_fields(pred_events)
    panel_specs = [
        ("a", "energy"),
        ("a", "pt"),
        ("a", "eta"),
        ("a", "phi"),
        ("b", "energy"),
        ("b", "pt"),
        ("b", "eta"),
        ("b", "phi"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=220)
    metrics: dict[str, Any] = {}
    weights = to_numpy(pred_events["event_weight"], np.float64) if "event_weight" in pred_events.fields else np.ones(len(pred_events), dtype=np.float64)
    flags_valid = to_numpy(pred_events["flags_valid"], bool) if "flags_valid" in pred_events.fields else np.ones(len(pred_events), dtype=bool)

    for axis, (leg_name, component) in zip(axes.flat, panel_specs):
        truth_key = f"truth_lead_{leg_name}_missing_{component}"
        pred_key = f"pred_lead_{leg_name}_missing_{component}"
        label = f"lead_{leg_name} missing {component}"
        if truth_key not in remapped or pred_key not in remapped:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "missing"}
            continue
        truth_values = remapped[truth_key]
        pred_values = remapped[pred_key]
        valid = np.isfinite(truth_values) & np.isfinite(pred_values) & np.isfinite(weights) & (weights > 0) & flags_valid
        valid &= ~np.isclose(truth_values, DEFAULT_FLOAT)
        valid &= ~np.isclose(pred_values, DEFAULT_FLOAT)
        if not np.any(valid):
            axis.text(0.5, 0.5, "no valid entries", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "no_valid_entries"}
            continue
        truth_valid = truth_values[valid]
        pred_valid = pred_values[valid]
        sampled = choose_sample_indices(len(truth_valid), max_scatter)
        axis.scatter(truth_valid[sampled], pred_valid[sampled], s=4, alpha=0.25, color=OKABE_ITO[0], linewidths=0)
        low = float(min(np.min(truth_valid), np.min(pred_valid)))
        high = float(max(np.max(truth_valid), np.max(pred_valid)))
        axis.plot([low, high], [low, high], linestyle="--", color=OKABE_ITO[3], linewidth=1.2)
        axis.set_title(label)
        axis.set_xlabel("Truth")
        axis.set_ylabel("Pred")
        axis.grid(alpha=0.2)
        corr = np.corrcoef(truth_valid, pred_valid)[0, 1] if len(truth_valid) > 1 else float("nan")
        metrics[label] = {
            "status": "ok",
            "num_valid_entries": int(len(truth_valid)),
            "pearson": float(corr) if np.isfinite(corr) else float("nan"),
            "mean_abs_diff": float(np.mean(np.abs(pred_valid - truth_valid))),
        }

    fig.suptitle("A/B-basis neutrino truth vs prediction")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def plot_prediction_slot_roundtrip(
    pred_events: ak.Array,
    prefix: str,
    output_path: Path,
    max_scatter: int,
) -> dict[str, Any]:
    array_pairs: list[tuple[np.ndarray | None, np.ndarray | None, str]] = []
    valid_masks: list[np.ndarray] = []
    for slot in (0, 1):
        p4, slot_valid = prediction_slot_p4(pred_events, prefix, slot)
        for component in ("energy", "pt", "eta", "phi"):
            stored = prediction_field_component(pred_events, prefix, slot, component)
            rebuilt = p4_component(p4, component) if p4 is not None else None
            array_pairs.append((stored, rebuilt, f"slot{slot} {component}"))
            valid_masks.append(slot_valid)

    combined_valid = np.ones(len(pred_events), dtype=bool)
    if "flags_valid" in pred_events.fields:
        combined_valid &= to_numpy(pred_events["flags_valid"], bool)

    metrics: dict[str, Any] = {}
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=220)
    for axis, (stored, rebuilt, label), slot_valid in zip(axes.flat, array_pairs, valid_masks):
        if stored is None or rebuilt is None:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "missing"}
            continue

        valid = np.isfinite(stored) & np.isfinite(rebuilt) & combined_valid & slot_valid
        valid &= ~np.isclose(stored, DEFAULT_FLOAT)
        valid &= ~np.isclose(rebuilt, DEFAULT_FLOAT)
        if not np.any(valid):
            axis.text(0.5, 0.5, "no valid entries", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
            metrics[label] = {"status": "no_valid_entries"}
            continue

        stored_valid = stored[valid]
        rebuilt_valid = rebuilt[valid]
        sampled = choose_sample_indices(len(stored_valid), max_scatter)
        axis.scatter(stored_valid[sampled], rebuilt_valid[sampled], s=4, alpha=0.25, color=OKABE_ITO[5], linewidths=0)
        low = float(min(np.min(stored_valid), np.min(rebuilt_valid)))
        high = float(max(np.max(stored_valid), np.max(rebuilt_valid)))
        axis.plot([low, high], [low, high], linestyle="--", color=OKABE_ITO[3], linewidth=1.2)
        axis.set_title(label)
        axis.set_xlabel("Stored slot kinematics")
        axis.set_ylabel("Rebuilt from p4")
        axis.grid(alpha=0.2)
        metrics[label] = {
            "status": "ok",
            "num_valid_entries": int(len(stored_valid)),
            "max_abs_diff": float(np.max(np.abs(rebuilt_valid - stored_valid))),
            "mean_abs_diff": float(np.mean(np.abs(rebuilt_valid - stored_valid))),
        }

    fig.suptitle(f"{prefix} slot kinematics round-trip")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_events = ak.from_parquet(args.prediction_parquet)
    export_events = ak.from_parquet(args.export_raw_parquet)
    pred_region_subset = prediction_subset_for_export_region(pred_events, args.export_raw_parquet)
    ab_truth_pred_summary = plot_ab_truth_vs_pred(
        pred_region_subset,
        args.output_dir / "ab_truth_vs_pred.png",
        args.max_scatter,
    )
    print(f"[consistency] ab_truth_vs_pred={ab_truth_pred_summary}", flush=True)
    pred_roundtrip_summary = plot_prediction_slot_roundtrip(
        pred_region_subset,
        "pred_invisible",
        args.output_dir / "pred_invisible_slot_roundtrip.png",
        args.max_scatter,
    )
    print(f"[consistency] pred_invisible_slot_roundtrip={pred_roundtrip_summary}", flush=True)
    aligned_pred, aligned_export, summary = align_events(pred_events, export_events, export_path=args.export_raw_parquet)
    print(f"[consistency] alignment={summary}", flush=True)
    if summary["matched_rows"] == 0:
        raise ValueError(
            "Unable to align any prediction rows with the chosen export parquet. "
            f"alignment={summary}"
        )
    if summary["export_predicted_rows"] == 0:
        export_name = args.export_raw_parquet.name
        hint = ""
        if export_name in {"filtered___ee.parquet", "filtered___emu.parquet", "filtered___mumu.parquet", "filtered___hadhad.parquet"}:
            hint = (
                " This looks like a common central region parquet. For signal-process sanity checks, compare against "
                "the fine exported signal region that actually contains EveNet predicted rows, e.g. "
                "`filtered___Ztautau_ee.parquet`, `filtered___Ztautau_emu.parquet`, `filtered___Ztautau_mumu.parquet`, "
                "`filtered___Ztautau_pipi.parquet`, `filtered___Ztautau_pirho.parquet`, or `filtered___Ztautau_rhorho.parquet`."
            )
        raise ValueError(
            "The chosen export parquet contains zero rows with `evenet_has_prediction=True`, so there is nothing to "
            "align against the prediction parquet." + hint
        )

    class_yield_summary = plot_class_yields(
        aligned_pred,
        aligned_export,
        args.output_dir / "class_yield_consistency.png",
    )

    visible_fields = [
        ("tau_vis_prong_slot0_energy", "tau_vis_prong_slot0_energy", "slot0 energy"),
        ("tau_vis_prong_slot0_pt", "tau_vis_prong_slot0_pt", "slot0 pt"),
        ("tau_vis_prong_slot0_eta", "tau_vis_prong_slot0_eta", "slot0 eta"),
        ("tau_vis_prong_slot0_phi", "tau_vis_prong_slot0_phi", "slot0 phi"),
        ("tau_vis_prong_slot1_energy", "tau_vis_prong_slot1_energy", "slot1 energy"),
        ("tau_vis_prong_slot1_pt", "tau_vis_prong_slot1_pt", "slot1 pt"),
        ("tau_vis_prong_slot1_eta", "tau_vis_prong_slot1_eta", "slot1 eta"),
        ("tau_vis_prong_slot1_phi", "tau_vis_prong_slot1_phi", "slot1 phi"),
    ]
    visible_summary = scatter_consistency_plot(
        aligned_pred,
        aligned_export,
        visible_fields,
        args.output_dir / "visible_tau_consistency.png",
        max_scatter=args.max_scatter,
        title="Visible tau slot consistency",
    )
    print(f"[consistency] visible_tau={visible_summary}", flush=True)

    invisible_fields = []
    for slot in (0, 1):
        for component, label_component in (("energy", "energy"), ("pt", "pt"), ("eta", "eta"), ("phi", "phi"), ("E", "E"), ("px", "px"), ("py", "py"), ("pz", "pz")):
            pred_field = f"pred_invisible_slot{slot}_{component}"
            export_field = pred_field
            if pred_field in aligned_pred.fields and export_field in aligned_export.fields:
                invisible_fields.append((pred_field, export_field, f"slot{slot} {label_component}"))
    invisible_summary = scatter_consistency_plot(
        aligned_pred,
        aligned_export,
        invisible_fields,
        args.output_dir / "pred_invisible_consistency.png",
        max_scatter=args.max_scatter,
        title="Predicted invisible slot consistency",
    )
    print(f"[consistency] pred_invisible={invisible_summary}", flush=True)

    expected_ab = build_expected_ab_fields(aligned_pred)
    ab_summary: dict[str, Any] = {"status": "missing_slot_metadata"}
    if expected_ab:
        ab_pred = ak.Array(expected_ab)
        ab_fields_all = [
            ("expected_lead_a_visible_energy", "lead_a_visible_p4", "lead_a visible energy", "E"),
            ("expected_lead_a_missing_E", "lead_a_missing_p4", "lead_a missing E", "E"),
            ("expected_lead_b_visible_energy", "lead_b_visible_p4", "lead_b visible energy", "E"),
            ("expected_lead_b_missing_E", "lead_b_missing_p4", "lead_b missing E", "E"),
        ]
        ab_fields = [field_info for field_info in ab_fields_all if field_info[0] in expected_ab]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=220)
        ab_summary = {}
        for axis, (pred_field, export_field, label, component) in zip(axes.flat, ab_fields):
            export_values = export_vector_component(aligned_export[export_field], component)
            pred_values = to_numpy(ab_pred[pred_field], np.float64)
            valid = np.isfinite(pred_values) & np.isfinite(export_values)
            valid &= ~np.isclose(pred_values, DEFAULT_FLOAT)
            valid &= ~np.isclose(export_values, DEFAULT_FLOAT)
            if not np.any(valid):
                axis.text(0.5, 0.5, "no valid entries", ha="center", va="center", fontsize=10, transform=axis.transAxes)
                axis.set_title(label)
                axis.set_xticks([])
                axis.set_yticks([])
                ab_summary[label] = {"status": "no_valid_entries"}
                continue
            pred_values = pred_values[valid]
            export_values = export_values[valid]
            sampled = choose_sample_indices(len(pred_values), args.max_scatter)
            axis.scatter(pred_values[sampled], export_values[sampled], s=4, alpha=0.25, color=OKABE_ITO[1], linewidths=0)
            low = float(min(np.min(pred_values), np.min(export_values)))
            high = float(max(np.max(pred_values), np.max(export_values)))
            axis.plot([low, high], [low, high], linestyle="--", color=OKABE_ITO[3], linewidth=1.2)
            axis.set_title(label)
            axis.set_xlabel("Prediction + slot_for_a/b")
            axis.set_ylabel("Export a/b field")
            axis.grid(alpha=0.2)
            ab_summary[label] = {
                "status": "ok",
                "num_valid_entries": int(len(pred_values)),
                "max_abs_diff": float(np.max(np.abs(export_values - pred_values))),
                "mean_abs_diff": float(np.mean(np.abs(export_values - pred_values))),
            }
        for axis in axes.flat[len(ab_fields):]:
            axis.text(0.5, 0.5, "not requested", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle("Central a/b remapping consistency")
        fig.tight_layout()
        fig.savefig(args.output_dir / "ab_remap_consistency.png", bbox_inches="tight")
        plt.close(fig)
        print(f"[consistency] ab_remap={ab_summary}", flush=True)

        ab_invisible_kinematics_fields_all = [
            ("expected_lead_a_missing_energy", "lead_a_missing_p4", "lead_a missing energy", "energy"),
            ("expected_lead_a_missing_pt", "lead_a_missing_p4", "lead_a missing pt", "pt"),
            ("expected_lead_a_missing_eta", "lead_a_missing_p4", "lead_a missing eta", "eta"),
            ("expected_lead_a_missing_phi", "lead_a_missing_p4", "lead_a missing phi", "phi"),
            ("expected_lead_b_missing_energy", "lead_b_missing_p4", "lead_b missing energy", "energy"),
            ("expected_lead_b_missing_pt", "lead_b_missing_p4", "lead_b missing pt", "pt"),
            ("expected_lead_b_missing_eta", "lead_b_missing_p4", "lead_b missing eta", "eta"),
            ("expected_lead_b_missing_phi", "lead_b_missing_p4", "lead_b missing phi", "phi"),
        ]
        ab_invisible_kinematics_fields = [
            field_info for field_info in ab_invisible_kinematics_fields_all if field_info[0] in expected_ab
        ]
        fig_kin, axes_kin = plt.subplots(2, 4, figsize=(16, 8), dpi=220)
        ab_invisible_kinematics_summary: dict[str, Any] = {}
        for axis, (pred_field, export_field, label, component) in zip(axes_kin.flat, ab_invisible_kinematics_fields):
            export_values = export_vector_component(aligned_export[export_field], component)
            pred_values = to_numpy(ab_pred[pred_field], np.float64)
            valid = np.isfinite(pred_values) & np.isfinite(export_values)
            valid &= ~np.isclose(pred_values, DEFAULT_FLOAT)
            valid &= ~np.isclose(export_values, DEFAULT_FLOAT)
            if not np.any(valid):
                axis.text(0.5, 0.5, "no valid entries", ha="center", va="center", fontsize=10, transform=axis.transAxes)
                axis.set_title(label)
                axis.set_xticks([])
                axis.set_yticks([])
                ab_invisible_kinematics_summary[label] = {"status": "no_valid_entries"}
                continue
            pred_values = pred_values[valid]
            export_values = export_values[valid]
            sampled = choose_sample_indices(len(pred_values), args.max_scatter)
            axis.scatter(pred_values[sampled], export_values[sampled], s=4, alpha=0.25, color=OKABE_ITO[2], linewidths=0)
            low = float(min(np.min(pred_values), np.min(export_values)))
            high = float(max(np.max(pred_values), np.max(export_values)))
            axis.plot([low, high], [low, high], linestyle="--", color=OKABE_ITO[3], linewidth=1.2)
            axis.set_title(label)
            axis.set_xlabel("Prediction + slot_for_a/b")
            axis.set_ylabel("Export a/b field")
            axis.grid(alpha=0.2)
            ab_invisible_kinematics_summary[label] = {
                "status": "ok",
                "num_valid_entries": int(len(pred_values)),
                "max_abs_diff": float(np.max(np.abs(export_values - pred_values))),
                "mean_abs_diff": float(np.mean(np.abs(export_values - pred_values))),
            }
        for axis in axes_kin.flat[len(ab_invisible_kinematics_fields):]:
            axis.text(0.5, 0.5, "not requested", ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        fig_kin.suptitle("Central a/b invisible kinematics consistency")
        fig_kin.tight_layout()
        fig_kin.savefig(args.output_dir / "ab_invisible_kinematics_consistency.png", bbox_inches="tight")
        plt.close(fig_kin)
        ab_summary["invisible_kinematics"] = ab_invisible_kinematics_summary
        print(f"[consistency] ab_invisible_kinematics={ab_invisible_kinematics_summary}", flush=True)

    with (args.output_dir / "export_consistency_summary.json").open("w") as handle:
        json.dump(
            {
                "alignment": summary,
                "class_yields": class_yield_summary,
                "visible_tau": visible_summary,
                "pred_invisible": invisible_summary,
                "ab_remap": ab_summary,
                "ab_truth_vs_pred": ab_truth_pred_summary,
                "pred_invisible_slot_roundtrip": pred_roundtrip_summary,
                "prediction_parquet": str(args.prediction_parquet),
                "export_raw_parquet": str(args.export_raw_parquet),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
