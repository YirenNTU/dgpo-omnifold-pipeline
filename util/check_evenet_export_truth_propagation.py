#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables, get_observable_names
from utils.common_functions import rebuild_p4


vector.register_awkward()

TARGET_COMPONENTS = ("energy", "pt", "eta", "phi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether truth-related observables drift between prediction parquet "
            "and exported truth-neutrino parquet."
        )
    )
    parser.add_argument("--prediction-parquet", nargs="+", required=True, help="Prediction parquet file(s), dir(s), or glob(s).")
    parser.add_argument("--export-parquet", nargs="+", required=True, help="Exported truth-neutrino parquet file(s), dir(s), or glob(s).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON summary path.")
    parser.add_argument("--max-entries", type=int, default=None, help="Optional row cap after alignment.")
    return parser.parse_args()


def resolve_parquet_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        expanded = Path(item).expanduser()
        if expanded.is_dir():
            final_prediction_paths = sorted(expanded.glob("*__evenet_pred.parquet"))
            matched = final_prediction_paths if final_prediction_paths else sorted(expanded.glob("*.parquet"))
            paths.extend(matched)
            continue
        matches = sorted(glob.glob(str(expanded)))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(expanded)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise FileNotFoundError("No parquet inputs found.")
    return unique


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(values.fields)
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


def load_events(paths: list[Path]) -> ak.Array:
    arrays = []
    for path in paths:
        events = ak.from_parquet(path)
        for field in events.fields:
            if field.endswith("_p4"):
                events[field] = rebuild_vector(events[field])
        arrays.append(events)
    if not arrays:
        return ak.Array([])
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def make_alignment_keys(events: ak.Array) -> np.ndarray:
    if "source_event_key" in events.fields:
        return ak.to_numpy(events["source_event_key"], allow_missing=False).astype(np.int64)
    if {"source_sample_index", "source_event_index"}.issubset(set(events.fields)):
        sample_index = ak.to_numpy(events["source_sample_index"], allow_missing=False).astype(np.int64)
        event_index = ak.to_numpy(events["source_event_index"], allow_missing=False).astype(np.int64)
        return sample_index * np.int64(10**12) + event_index
    raise ValueError("Need source_event_key or source_sample_index + source_event_index for alignment.")


def build_momentum4d(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> ak.Array:
    return vector.zip(
        {
            "px": np.asarray(px, dtype=np.float64),
            "py": np.asarray(py, dtype=np.float64),
            "pz": np.asarray(pz, dtype=np.float64),
            "E": np.asarray(energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )


def slot_field_values(events: ak.Array, prefix: str, slot: int, component: str) -> np.ndarray:
    field = f"{prefix}_slot{slot}_{component}"
    if field not in events.fields:
        raise ValueError(f"Missing field '{field}'.")
    return ak.to_numpy(events[field], allow_missing=False).astype(np.float64)


def slot_p4(events: ak.Array, prefix: str, slot: int) -> ak.Array:
    energy = slot_field_values(events, prefix, slot, "energy")
    pt = slot_field_values(events, prefix, slot, "pt")
    eta = slot_field_values(events, prefix, slot, "eta")
    phi = slot_field_values(events, prefix, slot, "phi")
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        energy,
    )


def prediction_truth_like_observables(events: ak.Array) -> dict[str, np.ndarray]:
    visible_a = slot_p4(events, "tau_vis_prong", 0)
    visible_b = slot_p4(events, "tau_vis_prong", 1)
    missing_a = slot_p4(events, "target_invisible", 0)
    missing_b = slot_p4(events, "target_invisible", 1)
    tau_a = visible_a + missing_a
    tau_b = visible_b + missing_b
    observables = build_observables(tau_a, tau_b, visible_a, visible_b)
    return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}


def export_truth_like_observables(events: ak.Array) -> dict[str, np.ndarray]:
    required = {"reco_tau_a_p4", "reco_tau_b_p4", "lead_a_visible_p4", "lead_b_visible_p4"}
    if not required.issubset(set(events.fields)):
        raise ValueError(f"Export parquet is missing required fields: {sorted(required - set(events.fields))}")
    observables = build_observables(
        events["reco_tau_a_p4"],
        events["reco_tau_b_p4"],
        events["lead_a_visible_p4"],
        events["lead_b_visible_p4"],
    )
    return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return {
            "count": 0,
            "exact_equal": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "rmse": None,
            "corr": None,
        }
    left = left[finite]
    right = right[finite]
    diff = right - left
    corr = None
    if left.size >= 2:
        matrix = np.corrcoef(left, right)
        value = matrix[0, 1]
        corr = float(value) if np.isfinite(value) else None
    return {
        "count": int(left.size),
        "exact_equal": bool(np.array_equal(left, right)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "corr": corr,
    }


def main() -> None:
    args = parse_args()
    pred_events = load_events(resolve_parquet_inputs(args.prediction_parquet))
    export_events = load_events(resolve_parquet_inputs(args.export_parquet))

    pred_keys = make_alignment_keys(pred_events)
    export_keys = make_alignment_keys(export_events)
    export_index_by_key = {int(key): index for index, key in enumerate(export_keys.tolist())}
    pred_indices = []
    export_indices = []
    for pred_index, key in enumerate(pred_keys.tolist()):
        export_index = export_index_by_key.get(int(key))
        if export_index is None:
            continue
        pred_indices.append(pred_index)
        export_indices.append(export_index)
    if not pred_indices:
        raise ValueError("No overlapping rows found between prediction parquet and export parquet.")

    pred_indices_np = np.asarray(pred_indices, dtype=np.int64)
    export_indices_np = np.asarray(export_indices, dtype=np.int64)
    if args.max_entries is not None and args.max_entries > 0:
        pred_indices_np = pred_indices_np[: args.max_entries]
        export_indices_np = export_indices_np[: args.max_entries]

    pred_aligned = pred_events[pred_indices_np]
    export_aligned = export_events[export_indices_np]

    pred_truth_like = prediction_truth_like_observables(pred_aligned)
    export_truth_like = export_truth_like_observables(export_aligned)

    summary: dict[str, Any] = {
        "prediction_rows": int(len(pred_events)),
        "export_rows": int(len(export_events)),
        "overlap_rows": int(len(pred_aligned)),
        "field_checks": {},
        "observable_checks": {},
    }

    field_pairs = [
        ("source_slot_for_a", "source_slot_for_a"),
        ("source_slot_for_b", "source_slot_for_b"),
    ]
    for slot in (0, 1):
        for component in TARGET_COMPONENTS:
            field_pairs.append((f"tau_vis_prong_slot{slot}_{component}", f"tau_vis_prong_slot{slot}_{component}"))
            field_pairs.append((f"target_invisible_slot{slot}_{component}", f"target_invisible_slot{slot}_{component}"))
    for left_field, right_field in field_pairs:
        if left_field not in pred_aligned.fields or right_field not in export_aligned.fields:
            continue
        left = ak.to_numpy(pred_aligned[left_field], allow_missing=False)
        right = ak.to_numpy(export_aligned[right_field], allow_missing=False)
        summary["field_checks"][left_field] = compare_arrays(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))

    for observable in get_observable_names():
        entry: dict[str, Any] = {}
        truth_field = f"truth_{observable}"
        if truth_field in pred_aligned.fields and truth_field in export_aligned.fields:
            pred_truth = ak.to_numpy(pred_aligned[truth_field], allow_missing=False).astype(np.float64)
            export_truth = ak.to_numpy(export_aligned[truth_field], allow_missing=False).astype(np.float64)
            entry["stored_truth_prediction_vs_export"] = compare_arrays(pred_truth, export_truth)
        if observable in pred_truth_like and observable in export_truth_like:
            entry["prediction_truth_like_vs_export_reco"] = compare_arrays(
                pred_truth_like[observable],
                export_truth_like[observable],
            )
        if entry:
            summary["observable_checks"][observable] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[check-evenet-export-truth-propagation] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
