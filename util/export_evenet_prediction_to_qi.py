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
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables
from utils.common_functions import rebuild_p4
from build_evenet_input_from_parquet import expand_samples, parse_config, read_yaml, source_event_key_array


vector.register_awkward()

DEFAULT_FLOAT = -99.0
DEFAULT_REGIONS = ("baseline", "hadhad", "ee", "mumu", "emu")


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
        return p4, valid

    if all(f"{prefix}_slot{slot}_{name}" in events.fields for name in ("energy", "pt", "eta", "phi")):
        p4 = vector.zip(
            {
                "pt": component("pt"),
                "eta": component("eta"),
                "phi": component("phi"),
                "E": component("energy"),
            }
        )
        return p4, valid

    if all(f"{prefix}_slot{slot}_{name}" in events.fields for name in ("log_energy", "log_pt", "eta", "phi")):
        p4 = vector.zip(
            {
                "pt": np.expm1(component("log_pt")),
                "eta": component("eta"),
                "phi": component("phi"),
                "E": np.expm1(component("log_energy")),
            }
        )
        return p4, valid

    available = [field for field in events.fields if field.startswith(f"{prefix}_slot{slot}_")]
    raise KeyError(
        f"Cannot build p4 for {prefix}_slot{slot}. Need E/px/py/pz, energy/pt/eta/phi, "
        f"or log_energy/log_pt/eta/phi. Available: {available}"
    )


def choose_by_slot(slot0, slot1, choose_slot1: np.ndarray):
    return ak.where(ak.Array(choose_slot1), slot1, slot0)


def finite_p4_mask(p4: ak.Array) -> np.ndarray:
    return (
        np.isfinite(ak.to_numpy(p4.px, allow_missing=False))
        & np.isfinite(ak.to_numpy(p4.py, allow_missing=False))
        & np.isfinite(ak.to_numpy(p4.pz, allow_missing=False))
        & np.isfinite(ak.to_numpy(p4.E, allow_missing=False))
    )


def zero_p4(num_events: int) -> ak.Array:
    zeros = np.zeros(num_events, dtype=np.float64)
    return vector.zip({"px": zeros, "py": zeros, "pz": zeros, "E": zeros})


def default_float_array(num_events: int, dtype=np.float32) -> np.ndarray:
    return np.full(num_events, DEFAULT_FLOAT, dtype=dtype)


def default_object_array(num_events: int, value: str = "unpredicted") -> ak.Array:
    return ak.Array([value] * num_events)


def default_field_like(field_values: ak.Array, num_events: int):
    array = ak.to_numpy(field_values, allow_missing=False)
    if array.dtype == np.bool_:
        return np.zeros(num_events, dtype=bool)
    if np.issubdtype(array.dtype, np.integer):
        return np.full(num_events, -1, dtype=array.dtype)
    if np.issubdtype(array.dtype, np.floating):
        return default_float_array(num_events, dtype=array.dtype)
    return default_object_array(num_events)


def safe_delta_r(first: ak.Array, second: ak.Array) -> np.ndarray:
    values = ak.to_numpy(first.deltaR(second), allow_missing=False).astype(np.float64)
    return np.where(np.isfinite(values), values, 1.0e9)


def infer_slot_to_central_leg(
    central_events: ak.Array,
    visible_slot0: ak.Array,
    visible_slot1: ak.Array,
    valid_slot0: np.ndarray,
    valid_slot1: np.ndarray,
) -> dict[str, Any]:
    if "lead_a_visible_p4" not in central_events.fields or "lead_b_visible_p4" not in central_events.fields:
        raise ValueError(
            "Cannot export EveNet slots back to the central a/b convention without "
            "lead_a_visible_p4 and lead_b_visible_p4 in the central source parquet."
        )

    lead_a = rebuild_vector(central_events["lead_a_visible_p4"])
    lead_b = rebuild_vector(central_events["lead_b_visible_p4"])

    score_direct = safe_delta_r(visible_slot0, lead_a) + safe_delta_r(visible_slot1, lead_b)
    score_swapped = safe_delta_r(visible_slot1, lead_a) + safe_delta_r(visible_slot0, lead_b)
    both_valid = valid_slot0 & valid_slot1 & finite_p4_mask(lead_a) & finite_p4_mask(lead_b)
    use_slot1_for_a = (score_swapped < score_direct) & both_valid

    delta_r_a = np.where(
        use_slot1_for_a,
        safe_delta_r(visible_slot1, lead_a),
        safe_delta_r(visible_slot0, lead_a),
    ).astype(np.float32)
    delta_r_b = np.where(
        use_slot1_for_a,
        safe_delta_r(visible_slot0, lead_b),
        safe_delta_r(visible_slot1, lead_b),
    ).astype(np.float32)
    delta_r_a = np.where(both_valid, delta_r_a, np.nan).astype(np.float32)
    delta_r_b = np.where(both_valid, delta_r_b, np.nan).astype(np.float32)

    return {
        "use_slot1_for_a": use_slot1_for_a,
        "delta_r_a": delta_r_a,
        "delta_r_b": delta_r_b,
        "aligned_by": "visible_deltaR_minimization",
    }


def build_predicted_reconstruction(
    central_pred_events: ak.Array,
    pred_events: ak.Array,
) -> tuple[dict[str, Any], dict[str, Any]]:
    visible0, visible0_valid = p4_from_components(pred_events, "tau_vis_prong", 0)
    visible1, visible1_valid = p4_from_components(pred_events, "tau_vis_prong", 1)
    pred_missing0, pred_missing0_valid = p4_from_components(pred_events, "pred_invisible", 0)
    pred_missing1, pred_missing1_valid = p4_from_components(pred_events, "pred_invisible", 1)

    alignment = infer_slot_to_central_leg(
        central_pred_events,
        visible0,
        visible1,
        visible0_valid,
        visible1_valid,
    )
    use_slot1_for_a = alignment["use_slot1_for_a"]
    use_slot1_for_b = ~use_slot1_for_a

    evenet_visible_a = choose_by_slot(visible0, visible1, use_slot1_for_a)
    evenet_visible_b = choose_by_slot(visible0, visible1, use_slot1_for_b)
    evenet_missing_a_raw = choose_by_slot(pred_missing0, pred_missing1, use_slot1_for_a)
    evenet_missing_b_raw = choose_by_slot(pred_missing0, pred_missing1, use_slot1_for_b)
    pred_valid_a = np.where(use_slot1_for_a, pred_missing1_valid, pred_missing0_valid)
    pred_valid_b = np.where(use_slot1_for_a, pred_missing0_valid, pred_missing1_valid)

    lead_a_visible = rebuild_vector(central_pred_events["lead_a_visible_p4"])
    lead_b_visible = rebuild_vector(central_pred_events["lead_b_visible_p4"])

    # EveNet slots are ordered by visible-particle kind, not by central a/b.
    # Restore the central convention here: lead_a is tau+ and lead_b is tau-.
    reco_tau_a = evenet_visible_a + evenet_missing_a_raw
    reco_tau_b = evenet_visible_b + evenet_missing_b_raw
    lead_a_missing = reco_tau_a - lead_a_visible
    lead_b_missing = reco_tau_b - lead_b_visible

    alignment_valid = np.isfinite(alignment["delta_r_a"]) & np.isfinite(alignment["delta_r_b"])
    flags_valid = pred_valid_a & pred_valid_b & alignment_valid & finite_p4_mask(reco_tau_a) & finite_p4_mask(reco_tau_b)
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
        "mmc_likelihood": np.zeros(len(central_pred_events), dtype=np.float32),
        "neutrino_method": ak.Array(["EveNet"] * len(central_pred_events)),
        "evenet_has_prediction": np.ones(len(central_pred_events), dtype=bool),
        "evenet_slot_for_a": np.where(use_slot1_for_a, 1, 0).astype(np.int8),
        "evenet_slot_for_b": np.where(use_slot1_for_a, 0, 1).astype(np.int8),
        "evenet_leg_match_deltaR_a": alignment["delta_r_a"],
        "evenet_leg_match_deltaR_b": alignment["delta_r_b"],
    }
    for obs_name, obs_values in observables.items():
        values[obs_name] = ak.where(flags_valid, obs_values, np.nan)
    for field in pred_events.fields:
        if field.startswith("evenet_") or field.startswith("pred_invisible_") or field.startswith("target_invisible_"):
            values[field] = pred_events[field]

    metrics = {
        "num_predicted_events": int(len(central_pred_events)),
        "valid_predicted_events": int(np.sum(flags_valid)),
        "valid_predicted_fraction": float(np.mean(flags_valid)) if len(flags_valid) else 0.0,
        "slot_alignment": alignment["aligned_by"],
        "slot_swap_fraction": float(np.mean(use_slot1_for_a)) if len(central_pred_events) else 0.0,
        "median_deltaR_a": finite_median(alignment["delta_r_a"]),
        "median_deltaR_b": finite_median(alignment["delta_r_b"]),
        "p95_deltaR_a": finite_percentile(alignment["delta_r_a"], 95),
        "p95_deltaR_b": finite_percentile(alignment["delta_r_b"], 95),
    }
    return values, metrics


def default_evenet_columns(full_events: ak.Array, pred_values: dict[str, Any]) -> dict[str, Any]:
    num_events = len(full_events)
    defaults: dict[str, Any] = {
        "lead_a_visible_p4": rebuild_vector(full_events["lead_a_visible_p4"])
        if "lead_a_visible_p4" in full_events.fields
        else zero_p4(num_events),
        "lead_b_visible_p4": rebuild_vector(full_events["lead_b_visible_p4"])
        if "lead_b_visible_p4" in full_events.fields
        else zero_p4(num_events),
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
        return vector.zip({"px": px, "py": py, "pz": pz, "E": energy})

    if isinstance(base_values, ak.Array):
        try:
            array = ak.to_numpy(base_values, allow_missing=False)
            pred_array = ak.to_numpy(pred_values, allow_missing=False)
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


def write_qi_tree(events: ak.Array, output_root: Path, sample_name: str, regions: list[str]) -> dict[str, int]:
    sample_dir = output_root / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"raw": int(len(events))}
    ak.to_parquet(events, sample_dir / "filtered___raw.parquet", compression="snappy")

    for region in regions:
        if region == "raw":
            continue
        cut_field = f"{region}_cut"
        if cut_field not in events.fields:
            continue
        region_events = events[events[cut_field] == 1]
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
    for source_index, (expanded_sample, events, parent_sample) in enumerate(expanded_entries):
        mapping[source_index] = {
            "expanded_name": expanded_sample.name,
            "parent_key": expanded_sample.key,
            "parent_name": parent_sample.name,
            "is_data": bool(parent_sample.is_data),
            "events": events,
        }
    return mapping


def selected_events_by_source_index(analysis_config_path: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    analysis_cfg = read_yaml(analysis_config_path)
    samples, subcategories, _ = parse_config(analysis_config_path)
    selected_events = {
        sample_key: load_concat_events(expand_paths(sample.input_files))
        for sample_key, sample in samples.items()
    }
    expanded = expand_samples(samples, selected_events, subcategories)
    mc_entries = []
    data_entries = []
    for expanded_sample, events in expanded:
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
        for field in ("source_sample_index", "source_event_key")
        if field not in pred_events.fields
    ]
    if missing:
        raise ValueError(
            f"{pred_path} is missing source id columns {missing}. Rebuild the EveNet input and rerun "
            "prediction with the updated ml_pipeline scripts so preprocessing shuffles/splits can be "
            "merged back to analysis.yaml raw_files safely."
        )


def event_keys(events: ak.Array) -> np.ndarray:
    return source_event_key_array(events)


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


def export_config_prediction(
    pred_path: Path,
    source_mapping: dict[int, dict[str, Any]],
    analysis_cfg: dict,
    output_root: Path,
    regions: list[str],
    prediction_split_fraction: float | None,
    output_label: str,
    summary_label: str,
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
        selected_subset = source_info["events"]
        if "source_event_index" in pred_subset.fields:
            selected_indices = ak.to_numpy(pred_subset["source_event_index"], allow_missing=False).astype(np.int64)
            if np.any(selected_indices < 0) or np.any(selected_indices >= len(selected_subset)):
                raise ValueError(
                    f"{pred_path} has source_event_index outside the selected source range for "
                    f"{source_info['expanded_name']}."
                )
        else:
            pred_keys = ak.to_numpy(pred_subset["source_event_key"], allow_missing=False)
            selected_keys = event_keys(selected_subset)
            selected_indices = map_by_event_key_occurrence(
                full_keys=selected_keys,
                subset_keys=pred_keys,
                context=f"{source_info['expanded_name']} prediction source rows",
            )
        selected_pred_source = selected_subset[selected_indices]

        group = grouped.setdefault(
            parent_name,
            {
                "parent_key": parent_key,
                "parent_name": parent_name,
                "is_data": source_info["is_data"],
                "pred_parts": [],
                "source_parts": [],
                "source_infos": [],
                "selected_index_parts": [],
                "expanded_samples": [],
            },
        )
        group["pred_parts"].append(pred_subset)
        group["source_parts"].append(selected_pred_source)
        group["source_infos"].append(source_info)
        group["selected_index_parts"].append(selected_indices)
        group["expanded_samples"].append(source_info["expanded_name"])

    for parent_name, group in grouped.items():
        raw_files = sample_raw_files(analysis_cfg, group["parent_key"], parent_name)
        full_events = load_concat_events(raw_files)
        pred_group = ak.concatenate(group["pred_parts"], axis=0) if len(group["pred_parts"]) > 1 else group["pred_parts"][0]
        source_group = ak.concatenate(group["source_parts"], axis=0) if len(group["source_parts"]) > 1 else group["source_parts"][0]
        split_fraction = None if group["is_data"] else prediction_split_fraction
        full_index_parts = []
        for source_info, selected_indices in zip(group["source_infos"], group["selected_index_parts"]):
            selected_to_full = map_subset_rows_to_full(
                full_events=full_events,
                subset_events=source_info["events"],
                context=f"{source_info['expanded_name']} selected rows",
            )
            full_index_parts.append(selected_to_full[selected_indices])
        full_indices = np.concatenate(full_index_parts).astype(np.int64) if full_index_parts else np.array([], dtype=np.int64)
        evenet_events, metrics = with_evenet_reconstruction(
            full_events=full_events,
            central_pred_events=source_group,
            pred_events=pred_group,
            full_indices=full_indices,
            prediction_split_fraction=split_fraction,
        )
        sample_output_root = output_root / output_label
        counts = write_qi_tree(evenet_events, sample_output_root, parent_name, regions)
        metrics["region_counts"] = counts
        metrics["parent_sample"] = parent_name
        metrics["expanded_samples"] = group["expanded_samples"]
        metrics["raw_files"] = raw_files
        metrics["is_data"] = group["is_data"]
        summary["samples"][parent_name] = metrics

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
