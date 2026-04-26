#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


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


def align_events(pred_events: ak.Array, export_events: ak.Array) -> tuple[ak.Array, ak.Array, dict[str, Any]]:
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
        "prediction_rows": int(len(pred_events)),
        "export_predicted_rows": int(len(export_pred)),
        "matched_rows": int(len(aligned_pred)),
        "missing_prediction_rows_in_export": missing_pred,
        "extra_export_rows": extra_export,
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
            axis.set_visible(False)
            metrics[label] = {"status": "missing"}
            continue

        pred_values = to_numpy(pred_events[pred_field], np.float64)
        export_values = to_numpy(export_events[export_field], np.float64)
        valid = np.isfinite(pred_values) & np.isfinite(export_values)
        valid &= ~np.isclose(pred_values, DEFAULT_FLOAT)
        valid &= ~np.isclose(export_values, DEFAULT_FLOAT)
        if not np.any(valid):
            axis.set_visible(False)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_events = ak.from_parquet(args.prediction_parquet)
    export_events = ak.from_parquet(args.export_raw_parquet)
    aligned_pred, aligned_export, summary = align_events(pred_events, export_events)

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

    invisible_fields = [
        ("pred_invisible_slot0_E", "pred_invisible_slot0_E", "slot0 E"),
        ("pred_invisible_slot0_px", "pred_invisible_slot0_px", "slot0 px"),
        ("pred_invisible_slot0_py", "pred_invisible_slot0_py", "slot0 py"),
        ("pred_invisible_slot0_pz", "pred_invisible_slot0_pz", "slot0 pz"),
        ("pred_invisible_slot1_E", "pred_invisible_slot1_E", "slot1 E"),
        ("pred_invisible_slot1_px", "pred_invisible_slot1_px", "slot1 px"),
        ("pred_invisible_slot1_py", "pred_invisible_slot1_py", "slot1 py"),
        ("pred_invisible_slot1_pz", "pred_invisible_slot1_pz", "slot1 pz"),
    ]
    invisible_summary = scatter_consistency_plot(
        aligned_pred,
        aligned_export,
        invisible_fields,
        args.output_dir / "pred_invisible_consistency.png",
        max_scatter=args.max_scatter,
        title="Predicted invisible slot consistency",
    )

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
                axis.set_visible(False)
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
            axis.set_visible(False)
        fig.suptitle("Central a/b remapping consistency")
        fig.tight_layout()
        fig.savefig(args.output_dir / "ab_remap_consistency.png", bbox_inches="tight")
        plt.close(fig)

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
                axis.set_visible(False)
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
            axis.set_visible(False)
        fig_kin.suptitle("Central a/b invisible kinematics consistency")
        fig_kin.tight_layout()
        fig_kin.savefig(args.output_dir / "ab_invisible_kinematics_consistency.png", bbox_inches="tight")
        plt.close(fig_kin)
        ab_summary["invisible_kinematics"] = ab_invisible_kinematics_summary

    with (args.output_dir / "export_consistency_summary.json").open("w") as handle:
        json.dump(
            {
                "alignment": summary,
                "class_yields": class_yield_summary,
                "visible_tau": visible_summary,
                "pred_invisible": invisible_summary,
                "ab_remap": ab_summary,
                "prediction_parquet": str(args.prediction_parquet),
                "export_raw_parquet": str(args.export_raw_parquet),
            },
            handle,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
