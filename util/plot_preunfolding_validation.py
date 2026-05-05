#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import vector
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
UTIL_ROOT = REPO_ROOT / "ml_pipeline" / "util"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(UTIL_ROOT))

from parquet_plot_common import choose_bins
from plot_style import channel_latex_label
from plot_qi_method_comparison import (
    event_weights,
    is_background_like_region,
    json_safe,
    load_events as _load_events_base,
    method_color,
    method_display_name,
    parquet_for,
    physics_observable_specs,
    plot_physics_data_mc_comparisons,
    rebuild_vector,
    sanitize_filename,
)
from quantum.observables_builder import build_observables, get_observable_names


vector.register_awkward()

# Column prefixes that appear in large EveNet exports but are never read by
# this script.  Skipping them at load time avoids OOM on oversized parquets.
_SKIP_COLUMN_PREFIXES = (
    "pred_invisible_slot",
    "tau_vis_prong_slot",
    "tau_vis_target_slot",
)


def parquet_columns_to_load(path: Path) -> list[str] | None:
    try:
        schema = pq.read_schema(path)
        return [f.name for f in schema if not f.name.startswith(_SKIP_COLUMN_PREFIXES)]
    except Exception:
        return None


def rebuild_event_vectors(events: ak.Array) -> ak.Array:
    for field in events.fields:
        if field.endswith("_p4"):
            events[field] = rebuild_vector(events[field])
    return events


def iter_event_batches(path: Path, batch_size: int, max_entries: int | None = None):
    columns = parquet_columns_to_load(path)
    if batch_size <= 0:
        events = ak.from_parquet(path, columns=columns)
        if max_entries is not None:
            events = events[:max_entries]
        yield rebuild_event_vectors(events)
        return

    parquet = pq.ParquetFile(path)
    remaining = None if max_entries is None else max(0, int(max_entries))
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        events = ak.from_arrow(record_batch)
        if remaining is not None and len(events) > remaining:
            events = events[:remaining]
        if len(events) == 0:
            continue
        yield rebuild_event_vectors(events)
        if remaining is not None:
            remaining -= len(events)
            if remaining <= 0:
                break


def load_events(path: Path, max_entries: int | None = None) -> ak.Array:
    """Load a parquet, dropping heavy slot-level columns not used by this script."""
    columns = parquet_columns_to_load(path)
    events = ak.from_parquet(path, columns=columns)
    if max_entries is not None:
        events = events[:max_entries]
    return rebuild_event_vectors(events)


DEFAULT_FLOAT = -99.0
OKABE_ITO_BLACK = "#000000"
TARGET_INVISIBLE_COMPONENTS = ("pt", "eta", "phi")
IGNORED_CHANNEL_REGIONS = {"hadhad"}
BASELINE_HADHAD_FINE_CHANNELS = (
    "pipi",
    "pirho",
    "rhopi",
    "rhorho",
)
BASELINE_FINE_REGION_TO_PARENT = {
    "ee": ("ee", "Ztautau_ee"),
    "emu": ("emu", "Ztautau_emu"),
    "mumu": ("mumu", "Ztautau_mumu"),
    "pipi": ("hadhad", "Ztautau_pipi"),
    "pirho": ("hadhad", "Ztautau_pirho"),
    "rhopi": ("hadhad", "Ztautau_rhopi"),
    "rhorho": ("hadhad", "Ztautau_rhorho"),
    "Ztautau_pipi": ("hadhad", "Ztautau_pipi"),
    "Ztautau_pirho": ("hadhad", "Ztautau_pirho"),
    "Ztautau_rhopi": ("hadhad", "Ztautau_rhopi"),
    "Ztautau_rhorho": ("hadhad", "Ztautau_rhorho"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-unfolding validation plots from exported central-schema parquet trees. "
            "Produces MC truth-vs-reco observable comparisons and data-vs-MC control plots."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method spec in the form Label:/path/to/export_tree. Repeat for Baseline/EveNet variants.",
    )
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Legacy shortcut for --method Baseline:<dir>.")
    parser.add_argument("--evenet-dir", type=Path, default=None, help="Legacy shortcut for --method EveNet:<dir>.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and JSON summaries.")
    parser.add_argument("--signal-sample-name", default="Ztautau", help="Signal MC sample directory name.")
    parser.add_argument("--data-sample-name", default="data94", help="Data sample directory name for control plots.")
    parser.add_argument(
        "--mc-sample-names",
        nargs="+",
        default=["Ztautau", "Zll", "Zqq"],
        help="MC sample directory names for control plots.",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help=(
            "Optional region filter. If omitted, each method uses all native exported regions found under "
            "<method-root>/<sample>/filtered___*.parquet."
        ),
    )
    parser.add_argument(
        "--truth-observables",
        nargs="+",
        default=None,
        help="Optional subset of signal truth/reco observables. Defaults to theta_cm and all cos_theta observables.",
    )
    parser.add_argument(
        "--control-observables",
        nargs="+",
        default=None,
        help="Optional subset of data-vs-MC control observables. Defaults to the existing method-comparison list.",
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-truth-reco",
        dest="normalize_truth_reco",
        action="store_true",
        default=True,
        help="Normalize truth-vs-reco distributions to unit area. This is the default.",
    )
    normalize_group.add_argument(
        "--no-normalize-truth-reco",
        dest="normalize_truth_reco",
        action="store_false",
        help="Plot absolute weighted yields for truth-vs-reco distributions.",
    )
    parser.add_argument(
        "--reco-observable-source",
        choices=["auto", "stored", "recompute"],
        default="auto",
        help=(
            "How to obtain reco truth-vs-reco observables. "
            "'auto' prefers stored parquet columns and falls back to recomputing from p4; "
            "'stored' requires the parquet column; "
            "'recompute' always rebuilds theta_cm/cos_theta_* from reco/visible p4."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for independent pre-unfolding plot blocks. "
            "Default 1 preserves serial behavior."
        ),
    )
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=50000,
        help="Parquet rows per streaming batch for large truth-vs-reco inputs.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None
    )
    return parser.parse_args()


def parse_method_specs(args: argparse.Namespace) -> dict[str, Path]:
    specs = list(args.method or [])
    if args.baseline_dir is not None:
        specs.append(f"Baseline:{args.baseline_dir}")
    if args.evenet_dir is not None:
        specs.append(f"EveNet:{args.evenet_dir}")
    if not specs:
        raise ValueError("Provide at least one --method Label:/path, or legacy --baseline-dir/--evenet-dir.")

    methods: dict[str, Path] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid --method '{spec}'. Expected Label:/path/to/tree.")
        label, path = spec.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Invalid --method '{spec}': empty label.")
        if label in methods:
            raise ValueError(f"Duplicate method label '{label}'.")
        methods[label] = Path(path).expanduser()
    return methods


def baseline_neutrino_reco_path(root: Path, sample_name: str, region: str) -> Path:
    return root / "neutrino_reco" / region / f"{sample_name}_reconstructed_neutrinos.parquet"


def neutrino_reco_root(root: Path) -> Path | None:
    if root.name in {"neutrino_reco", "neutrino_solutions"} and root.exists():
        return root
    for dirname in ("neutrino_solutions", "neutrino_reco"):
        candidate = root / dirname
        if candidate.exists():
            return candidate
    return None


def reconstructed_neutrino_paths(root: Path, sample_name: str, region: str) -> list[Path]:
    reco_root = neutrino_reco_root(root)
    if reco_root is None:
        return []
    if sample_name == "Ztautau":
        if region in BASELINE_FINE_REGION_TO_PARENT:
            _, signal_name = BASELINE_FINE_REGION_TO_PARENT[region]
            region_dir = region.removeprefix("Ztautau_")
            candidate = reco_root / region_dir / f"{signal_name}_reconstructed_neutrinos.parquet"
            return [candidate] if candidate.exists() else []

    region_dir = reco_root / region
    if not region_dir.exists():
        return []

    exact = region_dir / f"{sample_name}_reconstructed_neutrinos.parquet"
    if exact.exists():
        return [exact]

    matched = sorted(Path(path) for path in glob.glob(str(region_dir / f"{sample_name}*_reconstructed_neutrinos.parquet")))
    return matched


def method_event_paths(root: Path, sample_name: str, region: str) -> list[Path]:
    baseline_candidates = reconstructed_neutrino_paths(root, sample_name, region)
    if baseline_candidates:
        return baseline_candidates

    export_candidate = parquet_for(root, sample_name, region)
    if export_candidate.exists():
        return [export_candidate]
    return []


def load_method_events(root: Path, sample_name: str, region: str) -> ak.Array | None:
    paths = method_event_paths(root, sample_name, region)
    if not paths:
        return None
    arrays = [load_events(path) for path in paths]
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def truth_reco_event_paths(root: Path, sample_name: str, region: str) -> list[Path]:
    if neutrino_reco_root(root) is not None:
        return method_event_paths(root, sample_name, region)

    method_paths = method_event_paths(root, sample_name, region)
    if method_paths and not region.startswith("Ztautau_"):
        return method_paths

    raw_candidate = parquet_for(root, sample_name, "raw")
    if raw_candidate.exists() and expected_truth_classes_for_region(region):
        return [raw_candidate]

    return method_paths


def load_truth_reco_method_events(root: Path, sample_name: str, region: str, max_entries: int=None) -> ak.Array | None:
    paths = truth_reco_event_paths(root, sample_name, region)
    if not paths:
        return None
    arrays = [load_events(path, max_entries) for path in paths]
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def iter_truth_reco_method_event_batches(
    root: Path,
    sample_name: str,
    region: str,
    batch_size: int,
    max_entries: int | None = None,
):
    paths = truth_reco_event_paths(root, sample_name, region)
    if not paths:
        return
    remaining = None if max_entries is None else max(0, int(max_entries))
    for path in paths:
        path_limit = remaining
        for events in iter_event_batches(path, batch_size=batch_size, max_entries=path_limit):
            yield events
            if remaining is not None:
                remaining -= len(events)
                if remaining <= 0:
                    return


def observable_bins_from_limits(observable: str, low: float, high: float) -> np.ndarray:
    if observable == "theta_cm":
        return np.linspace(0.0, 1.0, 41)
    if observable.startswith("cos_theta_"):
        return np.linspace(-1.0, 1.0, 41)
    if observable.endswith("_phi"):
        return np.linspace(-np.pi, np.pi, 41)
    if observable.endswith("_eta"):
        bound = max(abs(low), abs(high), 1.0)
        return np.linspace(-bound, bound, 41)
    if any(observable.endswith(suffix) for suffix in ("_px", "_py", "_pz")):
        bound = max(abs(low), abs(high), 1.0)
        return np.linspace(-bound, bound, 41)
    bounded_low = min(low, 0.0) if observable.endswith(("_E", "_pt", "_mass")) else low
    if not np.isfinite(bounded_low) or not np.isfinite(high) or bounded_low == high:
        high = max(high, 1.0)
        bounded_low = 0.0 if observable.endswith(("_E", "_pt", "_mass")) else -1.0
    if observable.endswith(("_E", "_pt", "_mass")):
        bounded_low = 0.0
        high = max(high, 1.0)
    return np.linspace(bounded_low, high, 41)


def init_streaming_summary(
    observable: str,
    bins: np.ndarray,
    normalize: bool,
    include_2d: bool,
    edges2d: np.ndarray | None,
    limits2d: tuple[float, float] | None,
) -> dict[str, Any]:
    return {
        "observable": observable,
        "bins": bins,
        "normalize": normalize,
        "truth_hist": np.zeros(len(bins) - 1, dtype=np.float64),
        "reco_hist": np.zeros(len(bins) - 1, dtype=np.float64),
        "hist2d": np.zeros((len(edges2d) - 1, len(edges2d) - 1), dtype=np.float64) if include_2d and edges2d is not None else None,
        "edges2d": edges2d,
        "centers2d": histogram_centers(edges2d) if include_2d and edges2d is not None else None,
        "limits2d": limits2d,
        "num_events": 0,
        "sumw": 0.0,
        "sumw2": 0.0,
        "sum_truth": 0.0,
        "sum_reco": 0.0,
        "sum_diff": 0.0,
        "sum_absdiff": 0.0,
        "sum_sqdiff": 0.0,
        "sum_truth2": 0.0,
        "sum_reco2": 0.0,
        "sum_truthreco": 0.0,
        "truth_count": 0,
        "valid_count": 0,
        "truth_finite": 0,
        "reco_finite": 0,
        "positive_weight": 0,
        "flags_valid": 0,
    }


def finalize_streaming_summary(state: dict[str, Any]) -> dict[str, Any]:
    truth_hist = state["truth_hist"].copy()
    reco_hist = state["reco_hist"].copy()
    hist2d = None if state["hist2d"] is None else state["hist2d"].copy()
    if state["normalize"]:
        truth_total = np.sum(truth_hist)
        reco_total = np.sum(reco_hist)
        hist2d_total = np.sum(hist2d) if hist2d is not None else 0.0
        if truth_total > 0:
            truth_hist /= truth_total
        if reco_total > 0:
            reco_hist /= reco_total
        if hist2d is not None and hist2d_total > 0:
            hist2d /= hist2d_total

    sumw = state["sumw"]
    sumw2 = state["sumw2"]
    if sumw <= 0:
        truth_mean = reco_mean = bias = mae = rmse = pearson = pearson_unc = neff = float("nan")
    else:
        truth_mean = state["sum_truth"] / sumw
        reco_mean = state["sum_reco"] / sumw
        bias = state["sum_diff"] / sumw
        mae = state["sum_absdiff"] / sumw
        rmse = math.sqrt(max(state["sum_sqdiff"] / sumw, 0.0))
        var_truth = max(state["sum_truth2"] / sumw - truth_mean ** 2, 0.0)
        var_reco = max(state["sum_reco2"] / sumw - reco_mean ** 2, 0.0)
        cov = state["sum_truthreco"] / sumw - truth_mean * reco_mean
        denom = math.sqrt(var_truth * var_reco)
        pearson = float(cov / denom) if denom > 0 else float("nan")
        neff = float(sumw ** 2 / sumw2) if sumw2 > 0 else 0.0
        if np.isfinite(pearson) and neff > 3.0:
            clipped = float(np.clip(pearson, -0.999999, 0.999999))
            pearson_unc = float((1.0 - clipped ** 2) / math.sqrt(neff - 3.0))
        else:
            pearson_unc = float("nan")

    return {
        "truth_hist": truth_hist,
        "reco_hist": reco_hist,
        "hist2d": hist2d,
        "edges2d": state["edges2d"],
        "centers2d": state["centers2d"],
        "limits2d": state["limits2d"],
        "num_events": int(state["num_events"]),
        "weight_sum": float(sumw),
        "truth_mean": float(truth_mean),
        "reco_mean": float(reco_mean),
        "bias": float(bias),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "pearson_unc": float(pearson_unc),
        "neff": float(neff),
        "truth_count": int(state["truth_count"]),
        "valid_count": int(state["valid_count"]),
        "truth_finite": int(state["truth_finite"]),
        "reco_finite": int(state["reco_finite"]),
        "positive_weight": int(state["positive_weight"]),
        "flags_valid": int(state["flags_valid"]),
    }


def stream_truth_reco_region_summary(
    root: Path,
    signal_sample_name: str,
    region: str,
    observable_specs: list[tuple[str, str]],
    reco_observable_source: str,
    require_reco_valid_flags: bool,
    normalize: bool,
    load_batch_size: int,
    max_entries: int | None,
):
    selection_totals = {
        "input_events": 0,
        "selected_events": 0,
        "expected_truth_classes": sorted(expected_truth_classes_for_region(region)),
        "class_filter_applied": False,
        "correct_assignment_required": region.startswith("Ztautau_"),
    }
    first_events = None
    range_state: dict[str, dict[str, Any]] = {}

    for batch in iter_truth_reco_method_event_batches(
        root,
        signal_sample_name,
        region,
        batch_size=load_batch_size,
        max_entries=max_entries,
    ):
        if first_events is None:
            first_events = batch[:1] if len(batch) > 0 else batch
        selected_events, selection_info = select_truth_reco_events(batch, region)
        selection_totals["input_events"] += int(selection_info["input_events"])
        selection_totals["selected_events"] += int(selection_info["selected_events"])
        selection_totals["class_filter_applied"] = selection_totals["class_filter_applied"] or bool(selection_info["class_filter_applied"])
        if len(selected_events) == 0:
            continue

        for observable, _ in observable_specs:
            state = range_state.setdefault(
                observable,
                {
                    "missing_reco": False,
                    "missing_truth": False,
                    "truth_count": 0,
                    "valid_count": 0,
                    "truth_finite": 0,
                    "reco_finite": 0,
                    "positive_weight": 0,
                    "flags_valid": 0,
                    "low": float("inf"),
                    "high": float("-inf"),
                    "limits2d": None,
                },
            )
            reco_values_full = truth_reco_observable_values(selected_events, observable, source_mode=reco_observable_source)
            truth_values_full = truth_observable_values(selected_events, observable)
            if reco_values_full is None:
                state["missing_reco"] = True
                continue
            if truth_values_full is None:
                state["missing_truth"] = True
                continue

            weights_full = event_weights(selected_events)
            truth_mask = valid_truth_reco_mask(truth_values_full, truth_values_full, weights_full)
            valid_mask = valid_truth_reco_mask(truth_values_full, reco_values_full, weights_full)
            reco_event_mask = valid_reco_event_mask(selected_events)
            if reco_event_mask is not None and require_reco_valid_flags:
                truth_mask &= reco_event_mask
                valid_mask &= reco_event_mask

            state["truth_count"] += int(np.count_nonzero(truth_mask))
            state["valid_count"] += int(np.count_nonzero(valid_mask))
            state["truth_finite"] += int(np.count_nonzero(np.isfinite(truth_values_full)))
            state["reco_finite"] += int(np.count_nonzero(np.isfinite(reco_values_full)))
            state["positive_weight"] += int(np.count_nonzero(np.isfinite(weights_full) & (weights_full > 0)))
            state["flags_valid"] += int(np.count_nonzero(reco_event_mask)) if reco_event_mask is not None else 0

            if np.any(truth_mask):
                truth_values = truth_values_full[truth_mask]
                state["low"] = min(state["low"], float(np.nanmin(truth_values)))
                state["high"] = max(state["high"], float(np.nanmax(truth_values)))
            if np.any(valid_mask):
                truth_valid = truth_values_full[valid_mask]
                reco_valid = reco_values_full[valid_mask]
                state["low"] = min(state["low"], float(np.nanmin(reco_valid)))
                state["high"] = max(state["high"], float(np.nanmax(reco_valid)))
                if is_spin_observable(observable) or observable.startswith("lead_") or observable.startswith("reco_tau_"):
                    low2d, high2d = observable_2d_limits(observable, truth_valid, reco_valid)
                    if state["limits2d"] is None:
                        state["limits2d"] = (low2d, high2d)
                    else:
                        state["limits2d"] = (
                            min(state["limits2d"][0], low2d),
                            max(state["limits2d"][1], high2d),
                        )

    if first_events is None:
        return None, selection_totals

    summaries: dict[str, Any] = {}
    grouped_records: list[dict[str, Any]] = []
    streaming_states: dict[str, dict[str, Any]] = {}
    for observable, xlabel in observable_specs:
        state = range_state.get(observable)
        if state is None:
            continue
        if state["missing_reco"] or state["missing_truth"] or state["truth_count"] <= 0 or state["valid_count"] <= 0:
            continue
        bins = observable_bins_from_limits(observable, state["low"], state["high"])
        include_2d = is_spin_observable(observable) or observable.startswith("lead_") or observable.startswith("reco_tau_")
        limits2d = state["limits2d"]
        edges2d = np.linspace(limits2d[0], limits2d[1], 51) if include_2d and limits2d is not None else None
        streaming_states[observable] = init_streaming_summary(observable, bins, normalize, include_2d, edges2d, limits2d)

    if not streaming_states:
        return {"region_summary": {}, "grouped_records": [], "first_events": first_events}, selection_totals

    for batch in iter_truth_reco_method_event_batches(
        root,
        signal_sample_name,
        region,
        batch_size=load_batch_size,
        max_entries=max_entries,
    ):
        selected_events, _ = select_truth_reco_events(batch, region)
        if len(selected_events) == 0:
            continue
        for observable, _ in observable_specs:
            if observable not in streaming_states:
                continue
            reco_values_full = truth_reco_observable_values(selected_events, observable, source_mode=reco_observable_source)
            truth_values_full = truth_observable_values(selected_events, observable)
            if reco_values_full is None or truth_values_full is None:
                continue
            weights_full = event_weights(selected_events)
            truth_mask = valid_truth_reco_mask(truth_values_full, truth_values_full, weights_full)
            valid_mask = valid_truth_reco_mask(truth_values_full, reco_values_full, weights_full)
            reco_event_mask = valid_reco_event_mask(selected_events)
            if reco_event_mask is not None and require_reco_valid_flags:
                truth_mask &= reco_event_mask
                valid_mask &= reco_event_mask
            if not np.any(truth_mask) or not np.any(valid_mask):
                continue
            state = streaming_states[observable]
            bins = state["bins"]
            state["truth_hist"] += np.histogram(truth_values_full[truth_mask], bins=bins, weights=weights_full[truth_mask])[0].astype(np.float64)
            truth_valid = truth_values_full[valid_mask]
            reco_valid = reco_values_full[valid_mask]
            weight_valid = weights_full[valid_mask]
            state["reco_hist"] += np.histogram(reco_valid, bins=bins, weights=weight_valid)[0].astype(np.float64)
            if state["hist2d"] is not None and state["edges2d"] is not None:
                state["hist2d"] += np.histogram2d(truth_valid, reco_valid, bins=[state["edges2d"], state["edges2d"]], weights=weight_valid)[0].astype(np.float64)
            state["num_events"] += int(np.count_nonzero(valid_mask))
            state["sumw"] += float(np.sum(weight_valid))
            state["sumw2"] += float(np.sum(weight_valid ** 2))
            state["sum_truth"] += float(np.sum(weight_valid * truth_valid))
            state["sum_reco"] += float(np.sum(weight_valid * reco_valid))
            diff = reco_valid - truth_valid
            state["sum_diff"] += float(np.sum(weight_valid * diff))
            state["sum_absdiff"] += float(np.sum(weight_valid * np.abs(diff)))
            state["sum_sqdiff"] += float(np.sum(weight_valid * diff ** 2))
            state["sum_truth2"] += float(np.sum(weight_valid * truth_valid ** 2))
            state["sum_reco2"] += float(np.sum(weight_valid * reco_valid ** 2))
            state["sum_truthreco"] += float(np.sum(weight_valid * truth_valid * reco_valid))

    for observable, xlabel in observable_specs:
        if observable not in streaming_states:
            continue
        final_state = finalize_streaming_summary(streaming_states[observable])
        summaries[observable] = {
            "plot": None,
            "plot_2d": None,
            "normalize": normalize,
            "reco_observable_source": reco_observable_source,
            "num_events": final_state["num_events"],
            "weight_sum": final_state["weight_sum"],
            "truth_mean": final_state["truth_mean"],
            "reco_mean": final_state["reco_mean"],
            "bias": final_state["bias"],
            "mae": final_state["mae"],
            "rmse": final_state["rmse"],
            "pearson": final_state["pearson"],
            "pearson_unc": final_state["pearson_unc"],
            "neff": final_state["neff"],
        }
        grouped_records.append(
            {
                "region": region,
                "observable": observable,
                "xlabel": xlabel,
                "bins": streaming_states[observable]["bins"],
                "truth_hist": final_state["truth_hist"],
                "reco_hist": final_state["reco_hist"],
                "edges": final_state["edges2d"],
                "centers": final_state["centers2d"],
                "hist2d": final_state["hist2d"],
                "limits": final_state["limits2d"],
                "num_events": final_state["num_events"],
            }
        )

    return {"region_summary": summaries, "grouped_records": grouped_records, "first_events": first_events}, selection_totals


def truth_observable_specs(requested: list[str] | None) -> list[tuple[str, str]]:
    names = requested or list(get_observable_names())
    return [(name, observable_latex_label(name)) for name in names]


def missing_observable_specs(requested: list[str] | None = None) -> list[tuple[str, str]]:
    names = requested or [
        *(f"lead_a_missing_{component}" for component in ("E", "px", "py", "pz", "pt", "eta", "phi")),
        *(f"lead_b_missing_{component}" for component in ("E", "px", "py", "pz", "pt", "eta", "phi")),
    ]
    return [(name, observable_latex_label(name)) for name in names]


def reco_tau_observable_specs(requested: list[str] | None = None) -> list[tuple[str, str]]:
    names = requested or [
        *(f"reco_tau_a_{component}" for component in ("E", "px", "py", "pz", "pt", "eta", "phi", "mass")),
        *(f"reco_tau_b_{component}" for component in ("E", "px", "py", "pz", "pt", "eta", "phi", "mass")),
    ]
    return [(name, observable_latex_label(name)) for name in names]


def visible_tau_observable_specs(requested: list[str] | None = None) -> list[tuple[str, str]]:
    names = requested or [
        *(f"lead_a_visible_{component}" for component in ("E", "px", "py", "pz", "pt", "eta", "phi", "mass")),
        *(f"lead_b_visible_{component}" for component in ("E", "px", "py", "pz", "pt", "eta", "phi", "mass")),
    ]
    return [(name, observable_latex_label(name)) for name in names]


def observable_latex_label(name: str) -> str:
    if name == "theta_cm":
        return r"$2\arccos|\cos\theta_{\mathrm{CM}}|/\pi$"
    if name == "mtautau":
        return r"$m_{\tau\tau}$ [GeV]"
    if name.startswith("cos_theta_A_") and "_times_" not in name:
        axis = name.removeprefix("cos_theta_A_")
        return rf"$\cos\theta_{{A,{axis}}}$"
    if name.startswith("cos_theta_B_") and "_times_" not in name:
        axis = name.removeprefix("cos_theta_B_")
        return rf"$\cos\theta_{{B,{axis}}}$"
    if "_times_" in name and name.startswith("cos_theta_A_"):
        left, right = name.split("_times_")
        axis_a = left.removeprefix("cos_theta_A_")
        axis_b = right.removeprefix("cos_theta_B_")
        return rf"$\cos\theta_{{A,{axis_a}}}\times\cos\theta_{{B,{axis_b}}}$"
    if name.startswith("lead_a_missing_"):
        component = name.removeprefix("lead_a_missing_")
        return missing_component_label("a", component)
    if name.startswith("lead_b_missing_"):
        component = name.removeprefix("lead_b_missing_")
        return missing_component_label("b", component)
    if name.startswith("reco_tau_a_"):
        component = name.removeprefix("reco_tau_a_")
        return reco_tau_component_label("a", component)
    if name.startswith("reco_tau_b_"):
        component = name.removeprefix("reco_tau_b_")
        return reco_tau_component_label("b", component)
    if name.startswith("lead_a_visible_"):
        component = name.removeprefix("lead_a_visible_")
        return visible_tau_component_label("a", component)
    if name.startswith("lead_b_visible_"):
        component = name.removeprefix("lead_b_visible_")
        return visible_tau_component_label("b", component)
    return name.replace("_", " ")


def missing_component_label(leg: str, component: str) -> str:
    label_map = {
        "E": "E",
        "px": "p_x",
        "py": "p_y",
        "pz": "p_z",
        "pt": "p_T",
        "eta": r"\eta",
        "phi": r"\phi",
    }
    rendered = label_map.get(component, component)
    return rf"${rendered}(\nu_{{{leg}}}^{{reco}})$"


def reco_tau_component_label(leg: str, component: str) -> str:
    label_map = {
        "E": "E",
        "px": "p_x",
        "py": "p_y",
        "pz": "p_z",
        "pt": "p_T",
        "eta": r"\eta",
        "phi": r"\phi",
        "mass": "m",
    }
    rendered = label_map.get(component, component)
    return rf"${rendered}(\tau_{{{leg}}}^{{reco}})$"


def visible_tau_component_label(leg: str, component: str) -> str:
    label_map = {
        "E": "E",
        "px": "p_x",
        "py": "p_y",
        "pz": "p_z",
        "pt": "p_T",
        "eta": r"\eta",
        "phi": r"\phi",
        "mass": "m",
    }
    rendered = label_map.get(component, component)
    return rf"${rendered}(\tau_{{{leg}}}^{{vis}})$"


def is_spin_observable(name: str) -> bool:
    return name == "theta_cm" or name.startswith("cos_theta_")


def canonical_summary_region(region: str) -> str:
    mapping = {
        "pipi": "Ztautau_pipi",
        "pirho": "Ztautau_pirho",
        "rhopi": "Ztautau_rhopi",
        "Ztautau_pipi": "Ztautau_pipi",
        "Ztautau_pirho": "Ztautau_pirho",
        "Ztautau_rhopi": "Ztautau_rhopi",
        "Ztautau_rhorho": "Ztautau_rhorho",
    }
    return mapping.get(region, region)


def comparison_channel_group(region: str) -> str:
    """
    inputs:
      region: str, native region name from one method output.
    outputs:
      str, loose channel group used only for diagnostic comparison panels.
    goal:
      Draw broad and class-qualified versions of the same visible channel
      together, e.g. ee and Ztautau_ee, without changing summary-row labels.
    """
    return region.removeprefix("Ztautau_")


def broad_region_label(region: str) -> str:
    """
    inputs:
      region: str, broad reconstruction region or grouped diagnostic channel.
    outputs:
      str, display label that does not imply a truth decay class.
    goal:
      Keep broad regions like ee/emu/mumu visually distinct from truth-class
      labels like Ztautau_ee.
    """
    mapping = {
        "ee": r"$ee$",
        "emu": r"$e\mu$",
        "mue": r"$\mu e$",
        "mumu": r"$\mu\mu$",
        "pipi": r"$\pi\pi$",
        "pirho": r"$\pi\rho$",
        "rhopi": r"$\rho\pi$",
        "rhorho": r"$\rho\rho$",
    }
    return mapping.get(region, region.replace("_", r"\_"))


def expected_truth_classes_for_region(region: str) -> set[str]:
    mapping = {
        "ee": {"Ztautau_ee"},
        "emu": {"Ztautau_emu", "Ztautau_mue"},
        "mumu": {"Ztautau_mumu"},
        "Ztautau_ee": {"Ztautau_ee"},
        "Ztautau_emu": {"Ztautau_emu"},
        "Ztautau_mue": {"Ztautau_mue"},
        "Ztautau_mumu": {"Ztautau_mumu"},
        "pipi": {"Ztautau_pipi"},
        "pirho": {"Ztautau_pirho"},
        "rhopi": {"Ztautau_rhopi"},
        "Ztautau_pipi": {"Ztautau_pipi"},
        "Ztautau_pirho": {"Ztautau_pirho"},
        "Ztautau_rhopi": {"Ztautau_rhopi"},
        "Ztautau_rhorho": {"Ztautau_rhorho"},
    }
    return mapping.get(region, set())


def select_truth_reco_events(events: ak.Array, region: str) -> tuple[ak.Array, dict[str, Any]]:
    expected_classes = expected_truth_classes_for_region(region)
    require_correct_assignment = region.startswith("Ztautau_")
    info = {
        "input_events": int(len(events)),
        "expected_truth_classes": sorted(expected_classes),
        "class_filter_applied": False,
        "correct_assignment_required": require_correct_assignment,
        "selected_events": int(len(events)),
    }
    if not expected_classes:
        return events, info

    fields = set(events.fields)
    truth_names = (
        np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
        if "evenet_truth_class_name" in fields
        else None
    )
    pred_names = (
        np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
        if "evenet_pred_class_name" in fields
        else None
    )

    if truth_names is None and pred_names is None:
        return events, info

    mask = np.ones(len(events), dtype=bool)
    if truth_names is not None:
        mask &= np.isin(truth_names, list(expected_classes))
        info["class_filter_applied"] = True
    if pred_names is not None and truth_names is not None and require_correct_assignment:
        mask &= pred_names == truth_names
        info["class_filter_applied"] = True
    elif pred_names is not None and truth_names is None:
        mask &= np.isin(pred_names, list(expected_classes))
        info["class_filter_applied"] = True

    selected = events[mask]
    info["selected_events"] = int(len(selected))
    return selected, info


def summary_region_latex_label(region: str) -> str:
    if not region.startswith("Ztautau_"):
        return broad_region_label(region)
    return channel_latex_label(region)


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sum(values * weights) / total)


def weighted_rmse(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(weights * (pred - truth) ** 2) / total))


def weighted_mae(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    return float(np.sum(weights * np.abs(pred - truth)) / total)


def weighted_covariance(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    mean_x = weighted_mean(x, weights)
    mean_y = weighted_mean(y, weights)
    return float(np.sum(weights * (x - mean_x) * (y - mean_y)) / total)


def weighted_variance(x: np.ndarray, weights: np.ndarray) -> float:
    return weighted_covariance(x, x, weights)


def weighted_pearson(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    cov = weighted_covariance(truth, pred, weights)
    var_truth = weighted_variance(truth, weights)
    var_pred = weighted_variance(pred, weights)
    denom = math.sqrt(max(var_truth, 0.0) * max(var_pred, 0.0))
    if denom <= 0:
        return float("nan")
    return float(cov / denom)


def effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    sumsq = float(np.sum(weights ** 2))
    if total <= 0 or sumsq <= 0:
        return 0.0
    return float(total ** 2 / sumsq)


def pearson_uncertainty(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    correlation = weighted_pearson(truth, pred, weights)
    if not np.isfinite(correlation):
        return float("nan")
    neff = effective_sample_size(weights)
    if neff <= 3.0:
        return float("nan")
    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    sigma_z = 1.0 / math.sqrt(neff - 3.0)
    return float((1.0 - clipped ** 2) * sigma_z)


def valid_truth_reco_mask(truth: np.ndarray, reco: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mask = np.isfinite(truth) & np.isfinite(reco) & np.isfinite(weights) & (weights > 0)
    mask &= ~np.isclose(truth, DEFAULT_FLOAT)
    mask &= ~np.isclose(reco, DEFAULT_FLOAT)
    return mask


def valid_reco_event_mask(events: ak.Array) -> np.ndarray | None:
    if "flags_valid" not in events.fields:
        return None
    return to_numpy(events["flags_valid"], np.bool_)


def build_momentum4d(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> ak.Array:
    return ak.zip(
        {
            "px": np.asarray(px, dtype=np.float64),
            "py": np.asarray(py, dtype=np.float64),
            "pz": np.asarray(pz, dtype=np.float64),
            "E": np.asarray(energy, dtype=np.float64),
        },
        with_name="Momentum4D",
    )


def massless_p4_from_pt_eta_phi(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> ak.Array:
    """
    inputs:
      pt/eta/phi: np.ndarray, invisible momentum coordinates.
    outputs:
      ak.Array Momentum4D, massless four-vector.
    goal:
      Support EveNet prediction/export parquets whose invisible target stores
      only pt/eta/phi by reconstructing the neutrino energy.
    """
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        pt * np.cosh(eta),
    )


def choose_component_by_slot(events: ak.Array, prefix: str, slot_indices: np.ndarray, component: str) -> np.ndarray | None:
    slot0_name = f"{prefix}_slot0_{component}"
    slot1_name = f"{prefix}_slot1_{component}"
    fields = set(events.fields)
    if slot0_name not in fields or slot1_name not in fields:
        return None
    slot0 = ak.to_numpy(events[slot0_name], allow_missing=False).astype(np.float64)
    slot1 = ak.to_numpy(events[slot1_name], allow_missing=False).astype(np.float64)
    return np.where(slot_indices == 0, slot0, slot1)


def remapped_slot_p4(events: ak.Array, prefix: str, leg: str) -> ak.Array | None:
    slot_field = f"source_slot_for_{leg}"
    if slot_field not in events.fields:
        return None
    slot_indices = ak.to_numpy(events[slot_field], allow_missing=False).astype(np.int64)

    px = choose_component_by_slot(events, prefix, slot_indices, "px")
    py = choose_component_by_slot(events, prefix, slot_indices, "py")
    pz = choose_component_by_slot(events, prefix, slot_indices, "pz")
    energy = choose_component_by_slot(events, prefix, slot_indices, "E")
    if px is not None and py is not None and pz is not None and energy is not None:
        return build_momentum4d(px, py, pz, energy)

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
        )
    if pt is not None and eta is not None and phi is not None:
        return massless_p4_from_pt_eta_phi(pt, eta, phi)

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
        )
    if log_pt is not None and eta is not None and phi is not None:
        return massless_p4_from_pt_eta_phi(np.expm1(log_pt), eta, phi)
    return None


def truth_visible_field(fields: set[str], leg: str) -> str:
    """
    inputs:
      fields: set[str], available parquet fields.
      leg: str, "a" or "b".
    outputs:
      str, preferred visible-p4 field name for truth comparisons.
    goal:
      Use truth_visible_* when available, otherwise reuse central lead_*_visible_p4.
    """
    direct = f"truth_visible_{leg}_p4"
    return direct if direct in fields else f"lead_{leg}_visible_p4"


def target_invisible_slot_fields(leg: str) -> list[str]:
    """
    inputs:
      leg: str, "a" or "b".
    outputs:
      list[str], slot-level target invisible fields needed for fallback truth p4.
    goal:
      Keep slot-fallback requirements in one place so skip messages and value
      extraction stay aligned.
    """
    return [
        f"target_invisible_slot{slot}_{component}"
        for slot in (0, 1)
        for component in TARGET_INVISIBLE_COMPONENTS
    ] + [f"source_slot_for_{leg}"]


def truth_tau_p4(events: ak.Array, leg: str) -> ak.Array | None:
    """
    inputs:
      events: ak.Array, exported parquet events.
      leg: str, "a" or "b".
    outputs:
      Momentum4D array for truth tau p4, or None if the parquet cannot supply it.
    goal:
      Resolve truth tau p4 using the cleanest available source:
      truth_tau_* first, truth_missing_* + visible second, slot target fallback last.
    """
    fields = set(events.fields)
    direct = f"truth_tau_{leg}_p4"
    if direct in fields:
        return events[direct]

    visible = truth_visible_field(fields, leg)
    missing = f"truth_missing_{leg}_p4"
    if missing in fields and visible in fields:
        return events[missing] + events[visible]

    remapped_missing = remapped_slot_p4(events, "target_invisible", leg)
    if remapped_missing is not None and visible in fields:
        return remapped_missing + events[visible]
    return None


def truth_visible_p4(events: ak.Array, leg: str) -> ak.Array | None:
    """
    inputs:
      events: ak.Array, exported parquet events.
      leg: str, "a" or "b".
    outputs:
      Momentum4D array for truth visible tau p4, or None if unavailable.
    goal:
      Monitor the visible tau inputs independently from neutrino reconstruction.
    """
    field = f"truth_visible_{leg}_p4"
    return events[field] if field in events.fields else None


def reco_with_truth_neutrino_p4(events: ak.Array, leg: str) -> ak.Array | None:
    """
    inputs:
      events: ak.Array, exported parquet events.
      leg: str, "a" or "b".
    outputs:
      Momentum4D tau p4 built from current visible input plus truth neutrino.
    goal:
      Provide an upper-limit reconstruction that keeps visible-input effects
      but removes neutrino-regression errors.
    """
    fields = set(events.fields)
    visible_field = f"lead_{leg}_visible_p4"
    if visible_field not in fields:
        return None
    truth_missing_field = f"truth_missing_{leg}_p4"
    if truth_missing_field in fields:
        return events[visible_field] + events[truth_missing_field]
    remapped_missing = remapped_slot_p4(events, "target_invisible", leg)
    if remapped_missing is not None:
        return events[visible_field] + remapped_missing
    return None


def truth_tau_requirements(fields: set[str], leg: str) -> str:
    """
    inputs:
      fields: set[str], available parquet fields.
      leg: str, "a" or "b".
    outputs:
      str, human-readable reason describing which truth-tau source is available or missing.
    goal:
      Print concise skip/debug messages that mirror truth_tau_p4 resolution.
    """
    direct = f"truth_tau_{leg}_p4"
    if direct in fields:
        return f"truth tau field '{direct}' available"

    missing = f"truth_missing_{leg}_p4"
    visible = truth_visible_field(fields, leg)
    p4_missing = [field for field in (missing, visible) if field not in fields]
    if not p4_missing:
        return f"fallback truth tau fields '{missing}' + '{visible}' available"

    slot_missing = [
        field_name for field_name in target_invisible_slot_fields(leg) + [visible] if field_name not in fields
    ]
    if not slot_missing:
        return f"using slot fallback target_invisible + '{visible}'"
    return (
        f"missing direct truth tau field '{direct}', missing fallback truth-tau fields {p4_missing}, "
        f"and slot fallback fields {slot_missing}"
    )


def observable_bins(observable: str, values_by_name: dict[str, np.ndarray]) -> np.ndarray:
    if observable == "theta_cm":
        return np.linspace(0.0, 1.0, 41)
    if observable.startswith("cos_theta_"):
        return np.linspace(-1.0, 1.0, 41)
    if observable.endswith("_phi"):
        return np.linspace(-np.pi, np.pi, 41)
    return choose_bins(values_by_name, num_bins=40)


def observable_2d_limits(observable: str, truth_values: np.ndarray | None = None, reco_values: np.ndarray | None = None) -> tuple[float, float]:
    if observable == "theta_cm":
        return 0.0, 1.0
    if observable.startswith("cos_theta_"):
        return -1.0, 1.0
    combined = []
    if truth_values is not None:
        combined.append(np.asarray(truth_values, dtype=np.float64))
    if reco_values is not None:
        combined.append(np.asarray(reco_values, dtype=np.float64))
    if not combined:
        raise ValueError(f"Observable '{observable}' is not configured for 2D truth-vs-reco limits.")
    values = np.concatenate([arr[np.isfinite(arr)] for arr in combined if arr.size > 0])
    if values.size == 0:
        return -1.0, 1.0
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if observable.endswith("_phi"):
        return -float(np.pi), float(np.pi)
    if observable.endswith("_eta"):
        bound = max(abs(low), abs(high), 1.0)
        return -bound, bound
    if any(observable.endswith(suffix) for suffix in ("_px", "_py", "_pz")):
        bound = max(abs(low), abs(high), 1.0)
        return -bound, bound
    span = high - low
    if span <= 0:
        pad = max(abs(high) * 0.1, 1.0)
        return low - pad, high + pad
    pad = max(0.08 * span, 1.0e-3)
    return low - pad, high + pad


def weighted_hist(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, normalize: bool) -> np.ndarray:
    hist = np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)
    if normalize:
        total = np.sum(hist)
        if total > 0:
            hist = hist / total
    return hist


def weighted_hist2d(
    truth: np.ndarray,
    reco: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    """
    inputs:
      truth/reco: np.ndarray, matched truth and reconstructed values.
      weights: np.ndarray, positive event weights.
      edges: np.ndarray, common x/y bin edges.
      normalize: bool, whether to normalize to unit area.
    outputs:
      np.ndarray, weighted 2D histogram indexed as truth-bin, reco-bin.
    goal:
      Cache compact 2D diagnostic content so grouped comparison figures do
      not need to keep full event arrays alive.
    """
    hist = np.histogram2d(truth, reco, bins=[edges, edges], weights=weights)[0].astype(np.float64)
    if normalize:
        total = np.sum(hist)
        if total > 0:
            hist = hist / total
    return hist


def histogram_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def region_marker(region: str) -> str:
    mapping = {
        "baseline": "o",
        "hadhad": "s",
        "ee": "^",
        "mumu": "D",
        "emu": "v",
        "other": "P",
    }
    return mapping.get(region, "o")


def format_method_region_label(region: str) -> str:
    return region.replace("_", " ")


def method_marker(method: str, index: int) -> str:
    named = {
        "Baseline": "o",
        "EveNet": "s",
        "EveNet-Pretrain": "D",
        "EveNet-Scratch": "^",
        "Pretrain": "D",
        "Scratch": "^",
    }
    fallback = ["o", "s", "D", "^", "v", "P", "X"]
    return named.get(method, fallback[index % len(fallback)])


def render_grouped_truth_reco_panels(
    records_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    output_root: Path,
    output_dir: Path,
    normalize: bool,
    log_label: str,
) -> dict[tuple[str, str], str]:
    """
    inputs:
      records_by_group: dict, keyed by (channel_group, observable), with
        compact per-method/region histograms.
      output_root: Path, directory for this plot block.
      output_dir: Path, top-level output directory used for relative paths.
      normalize: bool, whether histograms are unit-normalized.
      log_label: str, human-readable block label for progress messages.
    outputs:
      dict[(str, str), str], relative combined-plot path by group key.
    goal:
      Produce one diagnostic figure per similar channel and observable, with
      the 1D truth/reco comparison on the left and one 2D truth-vs-reco block
      per method/region to the right.
    """
    combined_dir = output_root / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[tuple[str, str], str] = {}

    for (channel_group, observable), records in sorted(records_by_group.items()):
        if not records:
            continue
        records = deduplicate_grouped_records(records, channel_group, observable, log_label)

        num_2d_blocks = max(1, len(records))
        fig_width = max(13.2, 5.4 + 3.35 * num_2d_blocks)
        fig, axes = plt.subplots(
            1,
            1 + num_2d_blocks,
            figsize=(fig_width, 5.4),
            dpi=180,
            gridspec_kw={"width_ratios": [1.25, *([1.0] * num_2d_blocks)]},
        )
        ax1d = axes[0]
        axes2d = list(axes[1:])
        all_lows = [record["limits"][0] for record in records if record.get("limits") is not None]
        all_highs = [record["limits"][1] for record in records if record.get("limits") is not None]
        plot_low = float(np.nanmin(all_lows)) if all_lows else -1.0
        plot_high = float(np.nanmax(all_highs)) if all_highs else 1.0
        xlabel = records[0]["xlabel"]
        hist2d_max = max(
            (
                float(np.nanmax(record["hist2d"]))
                for record in records
                if record.get("hist2d") is not None and np.any(np.isfinite(record["hist2d"]))
            ),
            default=0.0,
        )
        colorbar_mesh = None

        legend_handles: list[Line2D] = []
        for index, record in enumerate(records):
            color = method_color(record["method"], record["method_index"])
            if index > 0 and any(
                previous["method"] == record["method"] for previous in records[:index]
            ):
                color = plt.get_cmap("tab20")(index % 20)
            label = f"{method_display_name(record['method'])}: {record['region']}"
            ax1d.step(
                record["bins"][:-1],
                record["truth_hist"],
                where="post",
                color=color,
                linestyle="--",
                linewidth=1.25,
                alpha=0.65,
            )
            ax1d.step(
                record["bins"][:-1],
                record["reco_hist"],
                where="post",
                color=color,
                linestyle="-",
                linewidth=1.7,
                alpha=0.95,
            )

            hist2d = record.get("hist2d")
            ax2d = axes2d[index]
            if hist2d is not None and np.nanmax(hist2d) > 0:
                colorbar_mesh = ax2d.pcolormesh(
                    record["edges"],
                    record["edges"],
                    hist2d.T,
                    cmap="Blues",
                    shading="auto",
                    vmin=0.0,
                    vmax=hist2d_max if hist2d_max > 0 else None,
                )
            else:
                ax2d.text(
                    0.5,
                    0.5,
                    "No 2D entries",
                    transform=ax2d.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            ax2d.plot([plot_low, plot_high], [plot_low, plot_high], color=OKABE_ITO_BLACK, linestyle="--", linewidth=1.0)
            ax2d.set_xlim(plot_low, plot_high)
            ax2d.set_ylim(plot_low, plot_high)
            ax2d.set_xlabel(f"Truth {xlabel}")
            if index == 0:
                ax2d.set_ylabel(f"Reco {xlabel}")
            else:
                ax2d.set_ylabel("")
                ax2d.tick_params(labelleft=False)
            ax2d.grid(alpha=0.18)
            ax2d.set_title(label, fontsize=9)
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    label=label,
                )
            )

        ax1d.set_xlabel(xlabel)
        ax1d.set_ylabel("Normalized yield" if normalize else "Weighted yield")
        ax1d.grid(alpha=0.22)
        ax1d.set_title("1D truth/reco")
        ax1d.plot([], [], color=OKABE_ITO_BLACK, linestyle="--", label="Truth")
        ax1d.plot([], [], color=OKABE_ITO_BLACK, linestyle="-", label="Reco")

        if colorbar_mesh is not None:
            colorbar_axis = fig.add_axes([0.018, 0.18, 0.012, 0.56])
            fig.colorbar(
                colorbar_mesh,
                cax=colorbar_axis,
                label="Normalized yield" if normalize else "Weighted yield",
            )
            colorbar_axis.yaxis.set_ticks_position("left")
            colorbar_axis.yaxis.set_label_position("left")

        fig.suptitle(f"{broad_region_label(channel_group)}: {xlabel}", y=0.90)
        style_handles = [
            Line2D([0], [0], color=OKABE_ITO_BLACK, linestyle="--", linewidth=1.5, label="Truth 1D"),
            Line2D([0], [0], color=OKABE_ITO_BLACK, linestyle="-", linewidth=1.5, label="Reco 1D"),
        ]
        fig.legend(
            handles=style_handles + legend_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(4, len(style_handles) + len(legend_handles)),
            fontsize=8,
        )
        fig.tight_layout(rect=(0.055, 0.0, 1.0, 0.86))

        channel_dir = combined_dir / sanitize_filename(channel_group)
        channel_dir.mkdir(parents=True, exist_ok=True)
        plot_path = channel_dir / f"{sanitize_filename(observable)}.png"
        fig.savefig(plot_path)
        plt.close(fig)

        relative_path = str(plot_path.relative_to(output_dir))
        paths[(channel_group, observable)] = relative_path
        print(
            f"[preunfolding] wrote grouped {log_label} channel={channel_group} "
            f"observable={observable} plot={plot_path}",
            flush=True,
        )

    return paths


def deduplicate_grouped_records(
    records: list[dict[str, Any]],
    channel_group: str,
    observable: str,
    log_label: str,
) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(record["method"], []).append(record)

    selected: list[dict[str, Any]] = []
    for method, method_records in by_method.items():
        if len(method_records) == 1:
            selected.append(method_records[0])
            continue

        def record_rank(record: dict[str, Any]) -> tuple[int, int, int]:
            exact_group = record["region"] == channel_group
            broad_region = not record["region"].startswith("Ztautau_")
            num_events = int(record.get("num_events") or 0)
            return (0 if exact_group else 1, 0 if broad_region else 1, -num_events)

        chosen = sorted(method_records, key=record_rank)[0]
        dropped = [record["region"] for record in method_records if record is not chosen]
        print(
            f"[preunfolding] deduplicate grouped {log_label} channel={channel_group} "
            f"observable={observable} method={method} kept={chosen['region']} dropped={dropped}",
            flush=True,
        )
        selected.append(chosen)

    return sorted(selected, key=lambda record: (record["method_index"], record["region"]))


def discover_method_regions(root: Path, sample_name: str, preferred_regions: list[str] | None = None) -> list[str]:
    discovered: list[str] = []
    sample_dir = root / sample_name
    raw_candidate = parquet_for(root, sample_name, "raw")
    if raw_candidate.exists():
        try:
            raw_fields = set(pq.ParquetFile(raw_candidate).schema_arrow.names)
        except Exception as error:
            print(f"[preunfolding] warning failed to inspect raw parquet schema path={raw_candidate}: {error}", flush=True)
            raw_fields = set()
        for field in sorted(raw_fields):
            if not field.endswith("_cut"):
                continue
            region = field.removesuffix("_cut")
            if region in {"raw", "baseline", "hadhad"} or region in IGNORED_CHANNEL_REGIONS:
                continue
            if expected_truth_classes_for_region(region):
                discovered.append(region)

    if sample_dir.exists():
        for path in sorted(sample_dir.glob("filtered___*.parquet")):
            region = path.stem.removeprefix("filtered___")
            if region == "raw" or region in IGNORED_CHANNEL_REGIONS:
                continue
            discovered.append(region)

    reco_root = neutrino_reco_root(root)
    if reco_root is not None:
        if sample_name == "Ztautau":
            for fine_region in ("ee", "emu", "mumu", *BASELINE_HADHAD_FINE_CHANNELS):
                if reconstructed_neutrino_paths(root, sample_name, fine_region):
                    discovered.append(fine_region)
        else:
            for region_dir in sorted(path for path in reco_root.iterdir() if path.is_dir()):
                if reconstructed_neutrino_paths(root, sample_name, region_dir.name):
                    discovered.append(region_dir.name)

    discovered = list(dict.fromkeys(discovered))

    if preferred_regions is None:
        return discovered

    preferred = [region for region in preferred_regions if region in discovered]
    extras = [region for region in discovered if region not in preferred]
    return preferred + extras


def plot_truth_vs_reco_by_method_and_region(
    method_regions: dict[str, list[str]],
    methods: dict[str, Path],
    signal_sample_name: str,
    observable_specs: list[tuple[str, str]],
    output_dir: Path,
    normalize: bool,
    reco_observable_source: str,
    subdir_name: str = "truth_vs_reco",
    log_label: str = "truth-vs-reco",
    require_reco_valid_flags: bool = True,
    max_entries: int = None,
    load_batch_size: int = 0,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    truth_dir = output_dir / subdir_name
    truth_dir.mkdir(parents=True, exist_ok=True)
    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for method_index, (method, root) in enumerate(methods.items()):
        regions = method_regions.get(method, [])
        print(f"[preunfolding] {log_label} method={method} native_regions={regions}", flush=True)
        method_summary: dict[str, Any] = {}

        for region in regions:
            print(f"  [region] method={method} region={region}", flush=True)
            paths = truth_reco_event_paths(root, signal_sample_name, region)
            if not paths:
                print(f"    [skip] missing signal parquet paths for method={method} region={region}", flush=True)
                continue
            print(
                f"    [load] method={method} sample={signal_sample_name} region={region} "
                f"paths={[str(path) for path in paths]}",
                flush=True,
            )
            if load_batch_size > 0:
                try:
                    streamed, selection_info = stream_truth_reco_region_summary(
                        root=root,
                        signal_sample_name=signal_sample_name,
                        region=region,
                        observable_specs=observable_specs,
                        reco_observable_source=reco_observable_source,
                        require_reco_valid_flags=require_reco_valid_flags,
                        normalize=normalize,
                        load_batch_size=load_batch_size,
                        max_entries=max_entries,
                    )
                except Exception as error:
                    print(
                        f"    [error] failed streaming load method={method} region={region}: {error}",
                        flush=True,
                    )
                    continue
                if selection_info["class_filter_applied"]:
                    print(
                        f"    [class-filter] method={method} region={region} "
                        f"input={selection_info['input_events']} selected={selection_info['selected_events']} "
                        f"expected_truth_classes={selection_info['expected_truth_classes']} "
                        f"correct_assignment_required={selection_info['correct_assignment_required']}",
                        flush=True,
                    )
                if streamed is None or not streamed["region_summary"]:
                    print(f"    [skip] no streamed observables remain after filtering method={method} region={region}", flush=True)
                    continue
                region_summary = streamed["region_summary"]
                for record in streamed["grouped_records"]:
                    print(
                        f"    [collect] method={method} region={region} observable={record['observable']} "
                        f"group={comparison_channel_group(region)}",
                        flush=True,
                    )
                    grouped_records.setdefault((comparison_channel_group(region), record["observable"]), []).append(
                        {
                            "method": method,
                            "method_index": method_index,
                            "region": region,
                            "observable": record["observable"],
                            "xlabel": record["xlabel"],
                            "bins": record["bins"],
                            "truth_hist": record["truth_hist"],
                            "reco_hist": record["reco_hist"],
                            "edges": record["edges"],
                            "centers": record["centers"],
                            "hist2d": record["hist2d"],
                            "limits": record["limits"],
                            "num_events": record["num_events"],
                        }
                    )
                method_summary[region] = region_summary
                continue
            try:
                events = load_truth_reco_method_events(root, signal_sample_name, region, max_entries)
            except Exception as error:
                print(
                    f"    [error] failed to load events method={method} region={region}: {error}",
                    flush=True,
                )
                continue
            if events is None:
                print(f"    [skip] failed to load events after path resolution method={method} region={region}", flush=True)
                continue
            try:
                events, selection_info = select_truth_reco_events(events, region)
            except Exception as error:
                print(
                    f"    [error] failed event selection method={method} region={region}: {error}",
                    flush=True,
                )
                continue
            if selection_info["class_filter_applied"]:
                print(
                    f"    [class-filter] method={method} region={region} "
                    f"input={selection_info['input_events']} selected={selection_info['selected_events']} "
                    f"expected_truth_classes={selection_info['expected_truth_classes']} "
                    f"correct_assignment_required={selection_info['correct_assignment_required']}",
                    flush=True,
                )
            if len(events) == 0:
                print(f"    [skip] no events remain after class filter method={method} region={region}", flush=True)
                continue
            region_summary: dict[str, Any] = {}

            for observable, xlabel in observable_specs:
                try:
                    reco_values_full = truth_reco_observable_values(
                        events,
                        observable,
                        source_mode=reco_observable_source,
                    )
                    truth_values_full = truth_observable_values(events, observable)
                except Exception as error:
                    print(
                        f"    [error] failed observable extraction method={method} region={region} "
                        f"observable={observable}: {error}",
                        flush=True,
                    )
                    continue
                if reco_values_full is None or truth_values_full is None:
                    print(
                        f"    [skip] method={method} region={region} observable={observable} "
                        f"reco_source={'missing' if reco_values_full is None else 'available'} "
                        f"truth_source={'missing' if truth_values_full is None else 'available'} "
                        f"reco_requirements=\"{reco_observable_requirements(events, observable, reco_observable_source)}\" "
                        f"truth_requirements=\"{truth_observable_requirements(events, observable)}\"",
                        flush=True,
                    )
                    continue

                weights_full = event_weights(events)
                truth_mask = valid_truth_reco_mask(truth_values_full, truth_values_full, weights_full)
                valid_mask = valid_truth_reco_mask(truth_values_full, reco_values_full, weights_full)
                reco_event_mask = valid_reco_event_mask(events)
                if reco_event_mask is not None and require_reco_valid_flags:
                    truth_mask &= reco_event_mask
                    valid_mask &= reco_event_mask
                if not np.any(truth_mask):
                    print(
                        f"    [skip] method={method} region={region} observable={observable} "
                        f"truth values are all invalid/default after masking "
                        f"(events={len(events)}, truth_finite={int(np.count_nonzero(np.isfinite(truth_values_full)))})",
                        flush=True,
                    )
                    continue
                if not np.any(valid_mask):
                    print(
                        f"    [skip] method={method} region={region} observable={observable} "
                        f"no valid truth/reco entries after masking "
                        f"(events={len(events)}, truth_finite={int(np.count_nonzero(np.isfinite(truth_values_full)))}, "
                        f"reco_finite={int(np.count_nonzero(np.isfinite(reco_values_full)))}, "
                        f"positive_weight={int(np.count_nonzero(np.isfinite(weights_full) & (weights_full > 0)))}, "
                        f"flags_valid={int(np.count_nonzero(reco_event_mask)) if reco_event_mask is not None else 'n/a'})",
                        flush=True,
                    )
                    continue

                truth_valid = truth_values_full[valid_mask]
                reco_valid = reco_values_full[valid_mask]
                weight_valid = weights_full[valid_mask]
                values_for_bins: dict[str, np.ndarray] = {
                    "truth": truth_values_full[truth_mask],
                    "reco": reco_valid,
                }
                bins = observable_bins(observable, values_for_bins)
                truth_hist = weighted_hist(truth_values_full[truth_mask], weights_full[truth_mask], bins, normalize)
                reco_hist = weighted_hist(reco_valid, weight_valid, bins, normalize)

                hist2d = None
                edges2d = None
                centers2d = None
                limits2d = None
                if (
                    is_spin_observable(observable)
                    or observable.startswith("lead_")
                    or observable.startswith("reco_tau_")
                ):
                    low, high = observable_2d_limits(observable, truth_valid, reco_valid)
                    edges2d = np.linspace(low, high, 51)
                    centers2d = histogram_centers(edges2d)
                    hist2d = weighted_hist2d(truth_valid, reco_valid, weight_valid, edges2d, normalize)
                    limits2d = (low, high)

                print(
                    f"    [collect] method={method} region={region} observable={observable} "
                    f"group={comparison_channel_group(region)}",
                    flush=True,
                )

                region_summary[observable] = {
                    "plot": None,
                    "plot_2d": None,
                    "normalize": normalize,
                    "reco_observable_source": reco_observable_source,
                    "num_events": int(np.count_nonzero(valid_mask)),
                    "weight_sum": float(np.sum(weight_valid)),
                    "truth_mean": weighted_mean(truth_valid, weight_valid),
                    "reco_mean": weighted_mean(reco_valid, weight_valid),
                    "bias": weighted_mean(reco_valid - truth_valid, weight_valid),
                    "mae": weighted_mae(truth_valid, reco_valid, weight_valid),
                    "rmse": weighted_rmse(truth_valid, reco_valid, weight_valid),
                    "pearson": weighted_pearson(truth_valid, reco_valid, weight_valid),
                    "pearson_unc": pearson_uncertainty(truth_valid, reco_valid, weight_valid),
                    "neff": effective_sample_size(weight_valid),
                }
                grouped_records.setdefault((comparison_channel_group(region), observable), []).append(
                    {
                        "method": method,
                        "method_index": method_index,
                        "region": region,
                        "observable": observable,
                        "xlabel": xlabel,
                        "bins": bins,
                        "truth_hist": truth_hist,
                        "reco_hist": reco_hist,
                        "edges": edges2d,
                        "centers": centers2d,
                        "hist2d": hist2d,
                        "limits": limits2d,
                        "num_events": int(np.count_nonzero(valid_mask)),
                    }
                )

            if region_summary:
                method_summary[region] = region_summary

        if method_summary:
            summary[method] = method_summary

    grouped_plot_paths = render_grouped_truth_reco_panels(
        grouped_records,
        truth_dir,
        output_dir,
        normalize,
        log_label,
    )
    for method_info in summary.values():
        for region, region_info in method_info.items():
            channel_group = comparison_channel_group(region)
            for observable, metrics in region_info.items():
                plot_path = grouped_plot_paths.get((channel_group, observable))
                if plot_path is not None:
                    metrics["plot"] = plot_path
                    metrics["plot_2d"] = plot_path
                    metrics["combined_channel_group"] = channel_group

    return summary


def run_truth_reco_plot_block(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    inputs:
      task: dict[str, Any], serializable arguments for one truth-vs-reco plot block.
    outputs:
      tuple(str, dict), block key and the generated summary.
    goal:
      Allow independent pre-unfolding validation blocks to run in separate
      worker processes without sharing matplotlib state.
    """
    summary = plot_truth_vs_reco_by_method_and_region(
        method_regions=task["method_regions"],
        methods=task["methods"],
        signal_sample_name=task["signal_sample_name"],
        observable_specs=task["observable_specs"],
        output_dir=task["output_dir"],
        normalize=task["normalize"],
        reco_observable_source=task["reco_observable_source"],
        subdir_name=task["subdir_name"],
        log_label=task["log_label"],
        require_reco_valid_flags=task.get("require_reco_valid_flags", True),
        max_entries=task.get("max_entries", None),
        load_batch_size=task.get("load_batch_size", 0),
    )
    return task["key"], summary


def truth_reco_observable_values(events: ak.Array, observable: str, source_mode: str = "auto") -> np.ndarray | None:
    if source_mode not in {"auto", "stored", "recompute", "truth-neutrino"}:
        raise ValueError(f"Unsupported reco observable source mode '{source_mode}'.")

    if source_mode in {"auto", "stored"} and observable in events.fields:
        return to_numpy(events[observable], np.float64)
    if source_mode == "stored":
        return None
    if source_mode == "truth-neutrino":
        if observable.startswith("lead_") and "_missing_" in observable:
            parts = observable.split("_")
            if len(parts) >= 4:
                leg = parts[1]
                component = parts[-1]
                field = f"truth_missing_{leg}_p4"
                if field in events.fields:
                    return missing_p4_component_values(events[field], component)
                remapped = remapped_slot_p4(events, "target_invisible", leg)
                if remapped is not None:
                    return missing_p4_component_values(remapped, component)
            return None
        if observable.startswith("reco_tau_"):
            parts = observable.split("_")
            if len(parts) >= 4:
                leg = parts[2]
                component = parts[-1]
                tau_p4 = reco_with_truth_neutrino_p4(events, leg)
                if tau_p4 is not None:
                    return missing_p4_component_values(tau_p4, component)
            return None
        required_visible = {"lead_a_visible_p4", "lead_b_visible_p4"}
        if not required_visible.issubset(set(events.fields)):
            return None
        tau_a = reco_with_truth_neutrino_p4(events, "a")
        tau_b = reco_with_truth_neutrino_p4(events, "b")
        if tau_a is None or tau_b is None:
            return None
        observables = build_observables(
            tau_a,
            tau_b,
            events["lead_a_visible_p4"],
            events["lead_b_visible_p4"],
        )
        if observable not in observables:
            return None
        return to_numpy(observables[observable], np.float64)
    if observable.startswith("lead_") and "_missing_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            component = parts[-1]
            field = f"lead_{leg}_missing_p4"
            if field in events.fields:
                return missing_p4_component_values(events[field], component)
        return None
    if observable.startswith("lead_") and "_visible_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            component = parts[-1]
            field = f"lead_{leg}_visible_p4"
            if field in events.fields:
                return missing_p4_component_values(events[field], component)
        return None
    if observable.startswith("reco_tau_"):
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[2]
            component = parts[-1]
            field = f"reco_tau_{leg}_p4"
            if field in events.fields:
                return missing_p4_component_values(events[field], component)
        return None
    required_fields = {"reco_tau_a_p4", "reco_tau_b_p4", "lead_a_visible_p4", "lead_b_visible_p4"}
    if not required_fields.issubset(set(events.fields)):
        return None
    observables = build_observables(
        events["reco_tau_a_p4"],
        events["reco_tau_b_p4"],
        events["lead_a_visible_p4"],
        events["lead_b_visible_p4"],
    )
    if observable not in observables:
        return None
    return to_numpy(observables[observable], np.float64)


def truth_observable_values(events: ak.Array, observable: str) -> np.ndarray | None:
    truth_field = f"truth_{observable}"
    if truth_field in events.fields:
        return to_numpy(events[truth_field], np.float64)

    if observable.startswith("lead_") and "_missing_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            component = parts[-1]
            field = f"truth_missing_{leg}_p4"
            if field in events.fields:
                return missing_p4_component_values(events[field], component)
            remapped = remapped_slot_p4(events, "target_invisible", leg)
            if remapped is not None:
                return missing_p4_component_values(remapped, component)
        return None
    if observable.startswith("lead_") and "_visible_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            component = parts[-1]
            truth_visible = truth_visible_p4(events, leg)
            if truth_visible is not None:
                return missing_p4_component_values(truth_visible, component)
        return None
    if observable.startswith("reco_tau_"):
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[2]
            component = parts[-1]
            truth_tau = truth_tau_p4(events, leg)
            if truth_tau is not None:
                return missing_p4_component_values(truth_tau, component)
        return None

    fields = set(events.fields)
    truth_vis_a_field = truth_visible_field(fields, "a")
    truth_vis_b_field = truth_visible_field(fields, "b")
    truth_tau_a = truth_tau_p4(events, "a")
    truth_tau_b = truth_tau_p4(events, "b")
    if truth_tau_a is None or truth_tau_b is None or truth_vis_a_field not in fields or truth_vis_b_field not in fields:
        return None
    observables = build_observables(
        truth_tau_a,
        truth_tau_b,
        events[truth_vis_a_field],
        events[truth_vis_b_field],
    )
    if observable not in observables:
        return None
    return to_numpy(observables[observable], np.float64)


def reco_observable_requirements(events: ak.Array, observable: str, source_mode: str) -> str:
    fields = set(events.fields)
    if source_mode in {"auto", "stored"} and observable in fields:
        return "stored column available"
    if source_mode == "stored":
        return f"missing stored column '{observable}'"
    if source_mode == "truth-neutrino":
        if observable.startswith("lead_") and "_missing_" in observable:
            parts = observable.split("_")
            if len(parts) >= 4:
                leg = parts[1]
                field = f"truth_missing_{leg}_p4"
                if field in fields:
                    return f"truth-neutrino field '{field}' available"
                missing = [field_name for field_name in target_invisible_slot_fields(leg) if field_name not in fields]
                if missing:
                    return f"missing truth-neutrino field '{field}' and slot fallback fields {missing}"
                return f"using slot fallback target_invisible + source_slot_for_{leg}"
        required = ["lead_a_visible_p4", "lead_b_visible_p4", "truth_missing_a_p4", "truth_missing_b_p4"]
        missing = [field for field in required if field not in fields]
        if missing:
            slot_missing = [
                field_name
                for leg in ("a", "b")
                for field_name in target_invisible_slot_fields(leg)
                if field_name not in fields
            ]
            return f"missing truth-neutrino recompute fields {missing}; slot fallback missing {slot_missing}"
        return "truth-neutrino recompute fields available"
    if observable.startswith("lead_") and "_missing_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            field = f"lead_{leg}_missing_p4"
            return f"required field '{field}' {'available' if field in fields else 'missing'}"
    if observable.startswith("lead_") and "_visible_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            field = f"lead_{leg}_visible_p4"
            return f"required field '{field}' {'available' if field in fields else 'missing'}"
    if observable.startswith("reco_tau_"):
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[2]
            field = f"reco_tau_{leg}_p4"
            return f"required field '{field}' {'available' if field in fields else 'missing'}"
    required = ["reco_tau_a_p4", "reco_tau_b_p4", "lead_a_visible_p4", "lead_b_visible_p4"]
    missing = [field for field in required if field not in fields]
    if missing:
        return f"missing recompute fields {missing}"
    return "recompute fields available"


def truth_observable_requirements(events: ak.Array, observable: str) -> str:
    fields = set(events.fields)
    truth_field = f"truth_{observable}"
    if truth_field in fields:
        return f"stored truth column '{truth_field}' available"
    if observable.startswith("lead_") and "_missing_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            field = f"truth_missing_{leg}_p4"
            if field in fields:
                return f"required field '{field}' available"
            missing = [field_name for field_name in target_invisible_slot_fields(leg) if field_name not in fields]
            if missing:
                return f"missing truth missing field '{field}' and slot fallback fields {missing}"
            return f"using slot fallback target_invisible + source_slot_for_{leg}"
    if observable.startswith("lead_") and "_visible_" in observable:
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[1]
            field = f"truth_visible_{leg}_p4"
            return f"required field '{field}' {'available' if field in fields else 'missing'}"
    if observable.startswith("reco_tau_"):
        parts = observable.split("_")
        if len(parts) >= 4:
            leg = parts[2]
            return truth_tau_requirements(fields, leg)
    truth_vis_a_field = truth_visible_field(fields, "a")
    truth_vis_b_field = truth_visible_field(fields, "b")
    required = ["truth_missing_a_p4", "truth_missing_b_p4", truth_vis_a_field, truth_vis_b_field]
    missing = [field for field in required if field not in fields]
    if missing:
        return f"missing truth recompute fields {missing}"
    return "truth recompute fields available"


def missing_p4_component_values(values: ak.Array, component: str) -> np.ndarray:
    if component == "E":
        return to_numpy(values.E, np.float64)
    if component == "pt":
        return to_numpy(values.pt, np.float64)
    if component == "eta":
        return to_numpy(values.eta, np.float64)
    if component == "phi":
        return to_numpy(values.phi, np.float64)
    return to_numpy(getattr(values, component), np.float64)


def metric_value_and_uncertainty(metrics: dict[str, Any], metric: str) -> tuple[float, float]:
    if metric == "pearson":
        return float(metrics.get("pearson", np.nan)), float(metrics.get("pearson_unc", np.nan))
    if metric == "bias":
        return float(metrics.get("bias", np.nan)), float("nan")
    if metric == "mae":
        return float(metrics.get("mae", np.nan)), float("nan")
    if metric == "rmse":
        return float(metrics.get("rmse", np.nan)), float("nan")
    raise ValueError(f"Unsupported summary metric '{metric}'.")


def summary_metric_label(metric: str) -> str:
    labels = {
        "pearson": "Weighted Pearson r",
        "bias": "Weighted bias",
        "mae": "Weighted MAE",
        "rmse": "Weighted RMSE",
    }
    return labels[metric]


def metric_precision(value: float, uncertainty: float) -> float:
    if not np.isfinite(value) or not np.isfinite(uncertainty) or value == 0:
        return float("nan")
    return abs(uncertainty / value) * 100.0


def deduplicate_summary_rows(rows: list[dict[str, Any]], observable: str, log_label: str) -> list[dict[str, Any]]:
    """
    inputs:
      rows: list[dict], candidate rows for one observable summary plot.
      observable: str, observable name for debug logging.
      log_label: str, summary block label for debug logging.
    outputs:
      list[dict], at most one row per (method, displayed summary region).
    goal:
      Avoid drawing duplicate method markers when broad and fine native regions
      collapse onto the same displayed summary channel.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["summary_region"]), []).append(row)

    selected_rows: list[dict[str, Any]] = []
    for (method, summary_region), group_rows in grouped.items():
        if len(group_rows) == 1:
            selected_rows.append(group_rows[0])
            continue

        def row_rank(row: dict[str, Any]) -> tuple[int, int]:
            exact_region = row["region"] == summary_region
            num_events = int(row.get("num_events") or 0)
            return (0 if exact_region else 1, -num_events)

        chosen = sorted(group_rows, key=row_rank)[0]
        dropped = [row["region"] for row in group_rows if row is not chosen]
        print(
            f"[preunfolding] deduplicate {log_label} observable={observable} "
            f"method={method} summary_region={summary_region} "
            f"kept={chosen['region']} dropped={dropped}",
            flush=True,
        )
        selected_rows.append(chosen)

    return selected_rows


def plot_truth_metric_summary(
    truth_summary: dict[str, Any],
    methods: dict[str, Path],
    observable_specs: list[tuple[str, str]],
    output_dir: Path,
    metric: str = "pearson",
    subdir_name: str = "truth_vs_reco_summary",
    log_label: str = "truth-summary",
) -> dict[str, Any]:
    summary_dir = output_dir / subdir_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_summary: dict[str, Any] = {}
    method_names = list(methods)

    all_regions = []
    for method_info in truth_summary.values():
        all_regions.extend(canonical_summary_region(region) for region in method_info.keys())
    unique_regions = list(dict.fromkeys(all_regions))
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=method_marker(method, index),
            color=method_color(method, index),
            markerfacecolor=method_color(method, index),
            markeredgecolor=method_color(method, index),
            markersize=7,
            linestyle="None",
            label=method_display_name(method),
        )
        for index, method in enumerate(method_names)
    ]

    for observable, _ in observable_specs:
        rows: list[dict[str, Any]] = []
        for method_index, method in enumerate(method_names):
            for region, region_info in truth_summary.get(method, {}).items():
                metrics = region_info.get(observable)
                if not metrics:
                    continue
                value, uncertainty = metric_value_and_uncertainty(metrics, metric)
                if not np.isfinite(value):
                    continue
                summary_region = canonical_summary_region(region)
                rows.append(
                    {
                        "method": method,
                        "method_index": method_index,
                        "region": region,
                        "summary_region": summary_region,
                        "value": value,
                        "uncertainty": uncertainty,
                        "precision": metric_precision(value, uncertainty),
                        "num_events": metrics.get("num_events"),
                    }
                )

        if not rows:
            continue
        rows = deduplicate_summary_rows(rows, observable, log_label)

        active_regions = [
            region
            for region in unique_regions
            if region != "baseline"
            and region not in IGNORED_CHANNEL_REGIONS
            and any(row["summary_region"] == region for row in rows)
        ]
        rows = [row for row in rows if row["summary_region"] in set(active_regions)]
        if not rows:
            continue
        fig_height = max(4.0, 0.7 * len(active_regions) + 1.8)
        fig, ax = plt.subplots(figsize=(10.6, fig_height), dpi=200)
        y_base = np.arange(len(active_regions), dtype=np.float64)
        region_to_index = {region: index for index, region in enumerate(active_regions)}

        x_values = np.array([row["value"] for row in rows], dtype=np.float64)
        x_unc = np.array([row["uncertainty"] for row in rows if np.isfinite(row["uncertainty"])], dtype=np.float64)
        xmin = float(np.nanmin(x_values))
        xmax = float(np.nanmax(x_values))
        span = xmax - xmin
        pad = max(0.08 * span, 0.01 if metric == "pearson" else 1.0e-3)
        if x_unc.size > 0:
            pad += float(np.nanmax(x_unc))
        ax.set_xlim(xmin - pad, xmax + pad)

        x_text_value = 1.03
        for region in active_regions:
            region_rows = [row for row in rows if row["summary_region"] == region]
            region_rows.sort(key=lambda row: row["method_index"])
            offsets = np.linspace(-0.24, 0.24, len(region_rows)) if len(region_rows) > 1 else np.array([0.0])
            for offset, row in zip(offsets, region_rows):
                y = y_base[region_to_index[region]] + offset
                color = method_color(row["method"], row["method_index"])
                xerr = row["uncertainty"] if np.isfinite(row["uncertainty"]) else None
                ax.errorbar(
                    row["value"],
                    y,
                    xerr=xerr,
                    fmt=method_marker(row["method"], row["method_index"]),
                    color=color,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    capsize=2.5 if xerr is not None else 0.0,
                    markersize=6.5,
                    lw=1.2,
                )
                if np.isfinite(row["uncertainty"]):
                    value_text = f"{row['value']:.3f} ± {row['uncertainty']:.3f}"
                else:
                    value_text = f"{row['value']:.3f}"
                ax.text(
                    x_text_value,
                    y,
                    value_text,
                    color=color,
                    fontsize=8,
                    va="center",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )

        ax.text(x_text_value, 1.02, r"$r \pm \sigma_r$", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
        ax.set_yticks(y_base)
        ax.set_yticklabels([summary_region_latex_label(region) for region in active_regions])
        ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.18, linestyle=":")
        for separator in np.arange(len(active_regions) - 1, dtype=np.float64) + 0.5:
            ax.axhline(separator, color="#D9D9D9", linewidth=0.8, zorder=0)
        ax.set_xlabel(summary_metric_label(metric))
        ax.set_ylabel("Channel / Region")
        ax.set_title(f"{observable_latex_label(observable)}: reco vs truth summary")

        ax.legend(
            handles=method_handles,
            title="Methods",
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=min(len(method_handles), 4),
        )

        fig.subplots_adjust(right=0.78, top=0.8, left=0.16, bottom=0.16)
        plot_path = summary_dir / f"{sanitize_filename(observable)}_{metric}.png"
        fig.savefig(plot_path)
        plt.close(fig)
        plot_summary[observable] = {
            "plot": str(plot_path.relative_to(output_dir)),
            "metric": metric,
            "num_points": len(rows),
            "methods": sorted({row["method"] for row in rows}),
            "regions": sorted({row["summary_region"] for row in rows}),
        }
        print(f"[preunfolding] wrote {log_label} observable={observable} metric={metric} plot={plot_path}", flush=True)

    return plot_summary


def write_report(
    output_dir: Path,
    methods: dict[str, Path],
    truth_summary: dict[str, Any],
    truth_metric_summary: dict[str, Any],
    truth_neutrino_summary: dict[str, Any],
    truth_neutrino_metric_summary: dict[str, Any],
    missing_summary: dict[str, Any],
    missing_metric_summary: dict[str, Any],
    reco_tau_summary: dict[str, Any],
    reco_tau_metric_summary: dict[str, Any],
    visible_tau_summary: dict[str, Any],
    visible_tau_metric_summary: dict[str, Any],
    control_summary: dict[str, Any],
    method_regions: dict[str, list[str]],
) -> None:
    lines = [
        "# Pre-Unfolding Validation",
        "",
        "## Methods",
        "",
        "| Method | Root | Native regions |",
        "|---|---|---|",
    ]
    for method, root in methods.items():
        native_regions = ", ".join(method_regions.get(method, [])) or "n/a"
        lines.append(f"| {method} | `{root}` | {native_regions} |")

    lines.extend(
        [
            "",
            "## Truth-vs-Reco Coverage",
            "",
            "| Method | Region | Observables with plots |",
            "|---|---|---:|",
        ]
    )
    for method, method_info in truth_summary.items():
        for region, region_info in method_info.items():
            lines.append(f"| {method} | {region} | {len(region_info)} |")

    lines.extend(
        [
            "",
            "## Truth-Neutrino Upper-Limit Coverage",
            "",
            "| Method | Region | Observables with plots |",
            "|---|---|---:|",
        ]
    )
    for method, method_info in truth_neutrino_summary.items():
        for region, region_info in method_info.items():
            lines.append(f"| {method} | {region} | {len(region_info)} |")

    lines.extend(
        [
            "",
            "## Missing-Neutrino Truth-vs-Reco Coverage",
            "",
            "| Method | Region | Observables with plots |",
            "|---|---|---:|",
        ]
    )
    for method, method_info in missing_summary.items():
        for region, region_info in method_info.items():
            lines.append(f"| {method} | {region} | {len(region_info)} |")

    lines.extend(
        [
            "",
            "## Reco-Tau Truth-vs-Reco Coverage",
            "",
            "| Method | Region | Observables with plots |",
            "|---|---|---:|",
        ]
    )
    for method, method_info in reco_tau_summary.items():
        for region, region_info in method_info.items():
            lines.append(f"| {method} | {region} | {len(region_info)} |")

    lines.extend(
        [
            "",
            "## Visible-Tau Input Truth-vs-Reco Coverage",
            "",
            "| Method | Region | Observables with plots |",
            "|---|---|---:|",
        ]
    )
    for method, method_info in visible_tau_summary.items():
        for region, region_info in method_info.items():
            lines.append(f"| {method} | {region} | {len(region_info)} |")

    control_count = 0
    for method_info in control_summary.values():
        for region_info in method_info.values():
            control_count += len(region_info.get("plots", []))

    lines.extend(
        [
            "",
            "## Control-Plot Coverage",
            "",
            f"- Data-vs-MC control plots: {control_count}",
            f"- Truth summary plots: {len(truth_metric_summary)}",
            f"- Truth-neutrino upper-limit summary plots: {len(truth_neutrino_metric_summary)}",
            f"- Missing-neutrino summary plots: {len(missing_metric_summary)}",
            f"- Reco-tau summary plots: {len(reco_tau_metric_summary)}",
            f"- Visible-tau input summary plots: {len(visible_tau_metric_summary)}",
            "",
            "## Generated PNG Files",
            "",
        ]
    )
    for path in sorted(output_dir.rglob("*.png")):
        lines.append(f"- `{path.relative_to(output_dir)}`")

    (output_dir / "preunfolding_validation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    methods = parse_method_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preunfolding] output_dir={args.output_dir}", flush=True)
    print(f"[preunfolding] methods={ {label: str(path) for label, path in methods.items()} }", flush=True)
    print(f"[preunfolding] reco_observable_source={args.reco_observable_source}", flush=True)

    method_regions = {
        method: discover_method_regions(root, args.signal_sample_name, args.regions)
        for method, root in methods.items()
    }
    observable_specs = truth_observable_specs(args.truth_observables)
    missing_specs = missing_observable_specs()
    reco_tau_specs = reco_tau_observable_specs()
    visible_tau_specs = visible_tau_observable_specs()
    print(f"[preunfolding] signal_sample={args.signal_sample_name}", flush=True)
    print(f"[preunfolding] method_regions={method_regions}", flush=True)
    print(f"[preunfolding] truth_observables={[name for name, _ in observable_specs]}", flush=True)
    print(f"[preunfolding] missing_observables={[name for name, _ in missing_specs]}", flush=True)
    print(f"[preunfolding] reco_tau_observables={[name for name, _ in reco_tau_specs]}", flush=True)
    print(f"[preunfolding] visible_tau_observables={[name for name, _ in visible_tau_specs]}", flush=True)
    print(f"[preunfolding] num_workers={args.num_workers}", flush=True)

    plot_block_tasks = [
        {
            "key": "truth",
            "method_regions": method_regions,
            "methods": methods,
            "signal_sample_name": args.signal_sample_name,
            "observable_specs": observable_specs,
            "output_dir": args.output_dir,
            "normalize": args.normalize_truth_reco,
            "reco_observable_source": args.reco_observable_source,
            "subdir_name": "truth_vs_reco",
            "log_label": "truth-vs-reco",
            "require_reco_valid_flags": True,
            "max_entries": args.max_entries,
            "load_batch_size": args.load_batch_size,
        },
        {
            "key": "truth_neutrino",
            "method_regions": method_regions,
            "methods": methods,
            "signal_sample_name": args.signal_sample_name,
            "observable_specs": observable_specs,
            "output_dir": args.output_dir,
            "normalize": args.normalize_truth_reco,
            "reco_observable_source": "truth-neutrino",
            "subdir_name": "truth_neutrino_upper_limit",
            "log_label": "truth-neutrino-upper-limit",
            "require_reco_valid_flags": False,
            "max_entries": args.max_entries,
            "load_batch_size": args.load_batch_size,
        },
        {
            "key": "missing",
            "method_regions": method_regions,
            "methods": methods,
            "signal_sample_name": args.signal_sample_name,
            "observable_specs": missing_specs,
            "output_dir": args.output_dir,
            "normalize": args.normalize_truth_reco,
            "reco_observable_source": "recompute",
            "subdir_name": "missing_truth_vs_reco",
            "log_label": "missing-truth-vs-reco",
            "require_reco_valid_flags": True,
            "max_entries": args.max_entries,
            "load_batch_size": args.load_batch_size,

        },
        {
            "key": "reco_tau",
            "method_regions": method_regions,
            "methods": methods,
            "signal_sample_name": args.signal_sample_name,
            "observable_specs": reco_tau_specs,
            "output_dir": args.output_dir,
            "normalize": args.normalize_truth_reco,
            "reco_observable_source": "recompute",
            "subdir_name": "reco_tau_truth_vs_reco",
            "log_label": "reco-tau-truth-vs-reco",
            "require_reco_valid_flags": True,
            "max_entries": args.max_entries,
            "load_batch_size": args.load_batch_size,

        },
        {
            "key": "visible_tau",
            "method_regions": method_regions,
            "methods": methods,
            "signal_sample_name": args.signal_sample_name,
            "observable_specs": visible_tau_specs,
            "output_dir": args.output_dir,
            "normalize": args.normalize_truth_reco,
            "reco_observable_source": "recompute",
            "subdir_name": "visible_tau_truth_vs_reco",
            "log_label": "visible-tau-truth-vs-reco",
            "require_reco_valid_flags": False,
            "max_entries": args.max_entries,
            "load_batch_size": args.load_batch_size,

        },
    ]
    if args.num_workers > 1:
        print(
            f"[preunfolding] launching parallel truth/reco blocks "
            f"workers={args.num_workers} blocks={len(plot_block_tasks)}",
            flush=True,
        )
        plot_block_results: dict[str, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(run_truth_reco_plot_block, task) for task in plot_block_tasks]
            for future in as_completed(futures):
                key, block_summary = future.result()
                plot_block_results[key] = block_summary
                print(f"[preunfolding] finished parallel block={key} methods={list(block_summary)}", flush=True)
        truth_summary = plot_block_results.get("truth", {})
        truth_neutrino_summary = plot_block_results.get("truth_neutrino", {})
        missing_summary = plot_block_results.get("missing", {})
        reco_tau_summary = plot_block_results.get("reco_tau", {})
        visible_tau_summary = plot_block_results.get("visible_tau", {})
    else:
        plot_block_results = {
            key: summary
            for key, summary in (run_truth_reco_plot_block(task) for task in plot_block_tasks)
        }
        truth_summary = plot_block_results["truth"]
        truth_neutrino_summary = plot_block_results["truth_neutrino"]
        missing_summary = plot_block_results["missing"]
        reco_tau_summary = plot_block_results["reco_tau"]
        visible_tau_summary = plot_block_results["visible_tau"]

    print(f"[preunfolding] finished truth-vs-reco methods={list(truth_summary)}", flush=True)
    truth_metric_summary = plot_truth_metric_summary(
        truth_summary=truth_summary,
        methods=methods,
        observable_specs=observable_specs,
        output_dir=args.output_dir,
        metric="pearson",
        subdir_name="truth_vs_reco_summary",
        log_label="truth-summary",
    )
    print(f"[preunfolding] finished truth summary plots observables={list(truth_metric_summary)}", flush=True)
    print(f"[preunfolding] finished truth-neutrino upper-limit methods={list(truth_neutrino_summary)}", flush=True)
    truth_neutrino_metric_summary = plot_truth_metric_summary(
        truth_summary=truth_neutrino_summary,
        methods=methods,
        observable_specs=observable_specs,
        output_dir=args.output_dir,
        metric="pearson",
        subdir_name="truth_neutrino_upper_limit_summary",
        log_label="truth-neutrino-upper-limit-summary",
    )
    print(
        f"[preunfolding] finished truth-neutrino upper-limit summary plots "
        f"observables={list(truth_neutrino_metric_summary)}",
        flush=True,
    )
    print(f"[preunfolding] finished missing truth-vs-reco methods={list(missing_summary)}", flush=True)
    missing_metric_summary = plot_truth_metric_summary(
        truth_summary=missing_summary,
        methods=methods,
        observable_specs=missing_specs,
        output_dir=args.output_dir,
        metric="pearson",
        subdir_name="missing_truth_vs_reco_summary",
        log_label="missing-summary",
    )
    print(f"[preunfolding] finished missing summary plots observables={list(missing_metric_summary)}", flush=True)
    print(f"[preunfolding] finished reco-tau truth-vs-reco methods={list(reco_tau_summary)}", flush=True)
    reco_tau_metric_summary = plot_truth_metric_summary(
        truth_summary=reco_tau_summary,
        methods=methods,
        observable_specs=reco_tau_specs,
        output_dir=args.output_dir,
        metric="pearson",
        subdir_name="reco_tau_truth_vs_reco_summary",
        log_label="reco-tau-summary",
    )
    print(f"[preunfolding] finished reco-tau summary plots observables={list(reco_tau_metric_summary)}", flush=True)
    print(f"[preunfolding] finished visible-tau truth-vs-reco methods={list(visible_tau_summary)}", flush=True)
    visible_tau_metric_summary = plot_truth_metric_summary(
        truth_summary=visible_tau_summary,
        methods=methods,
        observable_specs=visible_tau_specs,
        output_dir=args.output_dir,
        metric="pearson",
        subdir_name="visible_tau_truth_vs_reco_summary",
        log_label="visible-tau-summary",
    )
    print(f"[preunfolding] finished visible-tau summary plots observables={list(visible_tau_metric_summary)}", flush=True)

    print(
        f"[preunfolding] control-plot inputs data_sample={args.data_sample_name} "
        f"mc_samples={list(args.mc_sample_names)} control_observables={args.control_observables or 'default'}",
        flush=True,
    )
    control_observables = None
    if args.control_observables is not None:
        control_observables = [observable for observable in args.control_observables if not is_spin_observable(observable)]
    else:
        control_observables = [name for name, _, _ in physics_observable_specs(None) if not is_spin_observable(name)]
    control_summary: dict[str, Any] = {}
    for method, root in methods.items():
        native_regions = method_regions.get(method, [])
        if not native_regions:
            print(f"[preunfolding] skip control plots method={method} no native regions discovered", flush=True)
            control_summary[method] = {}
            continue
        if neutrino_reco_root(root) is None:
            method_control_summary = plot_physics_data_mc_comparisons(
                methods={method: root},
                output_dir=args.output_dir,
                regions=native_regions,
                data_sample_name=args.data_sample_name,
                mc_sample_names=list(args.mc_sample_names),
                requested_observables=control_observables,
            )
            control_summary[method] = method_control_summary.get(method, {})
        else:
            print(f"[preunfolding] skip control plots method={method} baseline neutrino_reco layout is not yet wired into data/MC control helper", flush=True)
            control_summary[method] = {}
    control_count = sum(len(region_info.get("plots", [])) for method_info in control_summary.values() for region_info in method_info.values())
    print(f"[preunfolding] finished control plots count={control_count}", flush=True)

    summary = {
        "methods": {method: str(path) for method, path in methods.items()},
        "signal_sample_name": args.signal_sample_name,
        "method_regions": method_regions,
        "truth_observables": [name for name, _ in observable_specs],
        "missing_observables": [name for name, _ in missing_specs],
        "reco_tau_observables": [name for name, _ in reco_tau_specs],
        "visible_tau_observables": [name for name, _ in visible_tau_specs],
        "reco_observable_source": args.reco_observable_source,
        "truth_vs_reco": truth_summary,
        "truth_metric_summary": truth_metric_summary,
        "truth_neutrino_upper_limit": truth_neutrino_summary,
        "truth_neutrino_metric_summary": truth_neutrino_metric_summary,
        "missing_truth_vs_reco": missing_summary,
        "missing_metric_summary": missing_metric_summary,
        "reco_tau_truth_vs_reco": reco_tau_summary,
        "reco_tau_metric_summary": reco_tau_metric_summary,
        "visible_tau_truth_vs_reco": visible_tau_summary,
        "visible_tau_metric_summary": visible_tau_metric_summary,
        "data_mc_control": control_summary,
        "control_observable_defaults": control_observables,
    }
    with (args.output_dir / "preunfolding_validation_summary.json").open("w") as handle:
        json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
    write_report(
        args.output_dir,
        methods,
        truth_summary,
        truth_metric_summary,
        truth_neutrino_summary,
        truth_neutrino_metric_summary,
        missing_summary,
        missing_metric_summary,
        reco_tau_summary,
        reco_tau_metric_summary,
        visible_tau_summary,
        visible_tau_metric_summary,
        control_summary,
        method_regions,
    )
    print(f"[preunfolding] wrote summary_json={args.output_dir / 'preunfolding_validation_summary.json'}", flush=True)
    print(f"[preunfolding] wrote report_md={args.output_dir / 'preunfolding_validation_report.md'}", flush=True)

    print(f"[plot-preunfolding-validation] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
