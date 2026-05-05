#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import build_observables, get_observable_names
from utils.common_functions import rebuild_p4


vector.register_awkward()

TARGET_COMPONENTS = ("pt", "eta", "phi")
DEFAULT_MAX_PLOT_ENTRIES = 200000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check truth-observable origins inside MC parquet files by comparing "
            "stored truth_* values, target-invisible reconstruction, and direct truth-p4 recalculation."
        )
    )
    parser.add_argument(
        "--parquet",
        nargs="+",
        required=True,
        help="Input parquet file(s), directory/directories, or glob pattern(s).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and JSON summary.")
    parser.add_argument(
        "--observables",
        nargs="+",
        default=None,
        help="Subset of observables to compare. Defaults to theta_cm, mtautau, and all cos_theta_*.",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="Rows per parquet streaming batch.")
    parser.add_argument("--max-entries", type=int, default=None, help="Optional global row cap.")
    parser.add_argument(
        "--max-plot-entries",
        type=int,
        default=DEFAULT_MAX_PLOT_ENTRIES,
        help="Maximum valid entries kept per observable/source for plotting.",
    )
    parser.add_argument(
        "--weight-field",
        choices=["auto", "weight", "central_weight", "evenet_weight", "unit"],
        default="auto",
        help="Weight source for metrics and normalized histograms.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Optional region name. If provided and <region>_cut exists, only selected events are used.",
    )
    parser.add_argument(
        "--truth-region-only",
        action="store_true",
        help="Require truth_QI_region == 1 when that field exists.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize 1D histograms to unit area.",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    clean = []
    for char in name:
        clean.append(char if char.isalnum() or char in {"_", "-", "."} else "_")
    return "".join(clean).strip("_")


def resolve_parquet_inputs(raw_inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in raw_inputs:
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
    unique = []
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


def requested_observables(raw: list[str] | None) -> list[str]:
    if raw:
        return list(raw)
    return list(get_observable_names())


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(values.fields)
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


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


def massless_p4_from_pt_eta_phi(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> ak.Array:
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        pt * np.cosh(eta),
    )


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


def event_weights(events: ak.Array, mode: str) -> np.ndarray:
    if mode == "unit":
        return np.ones(len(events), dtype=np.float64)
    candidates = []
    if mode == "auto":
        candidates = ["weight", "central_weight", "evenet_weight"]
    else:
        candidates = [mode]
    for field in candidates:
        if field in events.fields:
            values = to_numpy(events[field], np.float64)
            values = np.where(np.isfinite(values), values, 0.0)
            return values
    return np.ones(len(events), dtype=np.float64)


def parquet_columns_to_load(path: Path, observables: list[str], region: str | None, truth_region_only: bool) -> list[str] | None:
    schema = pq.read_schema(path)
    available = {field.name for field in schema}
    columns: set[str] = set()
    for observable in observables:
        columns.add(f"truth_{observable}")
    columns.update(
        {
            "weight",
            "central_weight",
            "evenet_weight",
            "truth_tau_a_p4",
            "truth_tau_b_p4",
            "truth_missing_a_p4",
            "truth_missing_b_p4",
            "truth_visible_a_p4",
            "truth_visible_b_p4",
            "lead_a_visible_p4",
            "lead_b_visible_p4",
            "source_slot_for_a",
            "source_slot_for_b",
        }
    )
    for slot in (0, 1):
        for component in TARGET_COMPONENTS:
            columns.add(f"target_invisible_slot{slot}_{component}")
    if region is not None:
        columns.add(f"{region}_cut")
    if truth_region_only:
        columns.add("truth_QI_region")
    selected = [name for name in columns if name in available]
    return sorted(selected) if selected else None


def iter_event_batches(
    path: Path,
    observables: list[str],
    batch_size: int,
    max_entries: int | None,
    region: str | None,
    truth_region_only: bool,
):
    columns = parquet_columns_to_load(path, observables, region, truth_region_only)
    parquet = pq.ParquetFile(path)
    remaining = None if max_entries is None else max(0, int(max_entries))
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        events = ak.from_arrow(batch)
        for field in events.fields:
            if field.endswith("_p4"):
                events[field] = rebuild_vector(events[field])
        if remaining is not None and len(events) > remaining:
            events = events[:remaining]
        if len(events) == 0:
            continue
        yield events
        if remaining is not None:
            remaining -= len(events)
            if remaining <= 0:
                break


def truth_visible_field(fields: set[str], leg: str) -> str:
    direct = f"truth_visible_{leg}_p4"
    return direct if direct in fields else f"lead_{leg}_visible_p4"


def choose_component_by_slot(events: ak.Array, prefix: str, slot_indices: np.ndarray, component: str) -> np.ndarray | None:
    slot0_name = f"{prefix}_slot0_{component}"
    slot1_name = f"{prefix}_slot1_{component}"
    fields = set(events.fields)
    if slot0_name not in fields or slot1_name not in fields:
        return None
    slot0 = to_numpy(events[slot0_name], np.float64)
    slot1 = to_numpy(events[slot1_name], np.float64)
    return np.where(slot_indices == 0, slot0, slot1)


def remapped_slot_p4(events: ak.Array, prefix: str, leg: str) -> ak.Array | None:
    slot_field = f"source_slot_for_{leg}"
    if slot_field not in events.fields:
        return None
    slot_indices = to_numpy(events[slot_field], np.int64)
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
    return None


def truth_tau_p4(events: ak.Array, leg: str) -> ak.Array | None:
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


def target_tau_p4(events: ak.Array, leg: str) -> ak.Array | None:
    visible_field = f"lead_{leg}_visible_p4"
    fields = set(events.fields)
    if visible_field not in fields:
        return None
    remapped_missing = remapped_slot_p4(events, "target_invisible", leg)
    if remapped_missing is None:
        return None
    return events[visible_field] + remapped_missing


def truth_observable_values(events: ak.Array, observable: str) -> np.ndarray | None:
    field = f"truth_{observable}"
    if field in events.fields:
        return to_numpy(events[field], np.float64)
    return None


def recalculated_truth_values(events: ak.Array, observable: str) -> np.ndarray | None:
    fields = set(events.fields)
    visible_a_field = truth_visible_field(fields, "a")
    visible_b_field = truth_visible_field(fields, "b")
    if visible_a_field not in fields or visible_b_field not in fields:
        return None
    truth_tau_a = truth_tau_p4(events, "a")
    truth_tau_b = truth_tau_p4(events, "b")
    if truth_tau_a is None or truth_tau_b is None:
        return None
    observables = build_observables(
        truth_tau_a,
        truth_tau_b,
        events[visible_a_field],
        events[visible_b_field],
    )
    if observable not in observables:
        return None
    return np.asarray(observables[observable], dtype=np.float64)


def target_reconstructed_values(events: ak.Array, observable: str) -> np.ndarray | None:
    fields = set(events.fields)
    if "lead_a_visible_p4" not in fields or "lead_b_visible_p4" not in fields:
        return None
    target_tau_a = target_tau_p4(events, "a")
    target_tau_b = target_tau_p4(events, "b")
    if target_tau_a is None or target_tau_b is None:
        return None
    observables = build_observables(
        target_tau_a,
        target_tau_b,
        events["lead_a_visible_p4"],
        events["lead_b_visible_p4"],
    )
    if observable not in observables:
        return None
    return np.asarray(observables[observable], dtype=np.float64)


def apply_event_selection(events: ak.Array, *, region: str | None, truth_region_only: bool) -> ak.Array:
    mask = np.ones(len(events), dtype=bool)
    if region is not None:
        field = f"{region}_cut"
        if field in events.fields:
            mask &= to_numpy(events[field], np.int64) > 0
    if truth_region_only and "truth_QI_region" in events.fields:
        mask &= to_numpy(events["truth_QI_region"], np.int64) > 0
    return events[mask]


@dataclass
class PairStats:
    count: int = 0
    weight_sum: float = 0.0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0
    sum_diff: float = 0.0
    sum_abs_diff: float = 0.0
    sum_diff2: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        if x.size == 0:
            return
        self.count += int(x.size)
        self.weight_sum += float(np.sum(weights))
        self.sum_x += float(np.sum(weights * x))
        self.sum_y += float(np.sum(weights * y))
        self.sum_x2 += float(np.sum(weights * x * x))
        self.sum_y2 += float(np.sum(weights * y * y))
        self.sum_xy += float(np.sum(weights * x * y))
        diff = x - y
        self.sum_diff += float(np.sum(weights * diff))
        self.sum_abs_diff += float(np.sum(weights * np.abs(diff)))
        self.sum_diff2 += float(np.sum(weights * diff * diff))

    def summary(self) -> dict[str, float | int | None]:
        if self.weight_sum <= 0.0:
            return {
                "count": self.count,
                "weight_sum": self.weight_sum,
                "weighted_mean_diff": None,
                "weighted_mean_abs_diff": None,
                "weighted_rmse": None,
                "weighted_corr": None,
            }
        mean_x = self.sum_x / self.weight_sum
        mean_y = self.sum_y / self.weight_sum
        var_x = max(self.sum_x2 / self.weight_sum - mean_x * mean_x, 0.0)
        var_y = max(self.sum_y2 / self.weight_sum - mean_y * mean_y, 0.0)
        cov_xy = self.sum_xy / self.weight_sum - mean_x * mean_y
        denom = math.sqrt(var_x * var_y)
        corr = cov_xy / denom if denom > 0.0 else None
        return {
            "count": self.count,
            "weight_sum": self.weight_sum,
            "weighted_mean_diff": self.sum_diff / self.weight_sum,
            "weighted_mean_abs_diff": self.sum_abs_diff / self.weight_sum,
            "weighted_rmse": math.sqrt(max(self.sum_diff2 / self.weight_sum, 0.0)),
            "weighted_corr": corr,
        }


def sampled_append(current: list[np.ndarray], values: np.ndarray, limit: int) -> list[np.ndarray]:
    if limit <= 0 or values.size == 0:
        return current
    current_size = sum(chunk.size for chunk in current)
    if current_size >= limit:
        return current
    keep = min(limit - current_size, values.size)
    current.append(np.asarray(values[:keep]))
    return current


def finalize_sampled(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def observable_bins(observable: str, values: list[np.ndarray]) -> np.ndarray:
    non_empty = [np.asarray(chunk, dtype=np.float64) for chunk in values if np.asarray(chunk).size > 0]
    non_empty = [chunk[np.isfinite(chunk)] for chunk in non_empty if np.asarray(chunk).size > 0]
    non_empty = [chunk for chunk in non_empty if chunk.size > 0]
    if not non_empty:
        if observable == "theta_cm":
            return np.linspace(0.0, 1.0, 41)
        if observable.startswith("cos_theta_"):
            return np.linspace(-1.0, 1.0, 41)
        return np.linspace(-1.0, 1.0, 41)
    combined = np.concatenate(non_empty)
    if observable == "theta_cm":
        return np.linspace(0.0, 1.0, 41)
    if observable.startswith("cos_theta_"):
        return np.linspace(-1.0, 1.0, 41)
    low = float(np.nanmin(combined))
    high = float(np.nanmax(combined))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = low - 1.0, high + 1.0
    pad = max(0.05 * (high - low), 1.0e-3)
    return np.linspace(low - pad, high + pad, 41)


def weighted_hist(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, normalize: bool) -> np.ndarray:
    hist = np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)
    if normalize:
        total = np.sum(hist)
        if total > 0.0:
            hist /= total
    return hist


def weighted_hist2d(x: np.ndarray, y: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist = np.histogram2d(x, y, bins=[edges, edges], weights=weights)[0].astype(np.float64)
    total = np.sum(hist)
    if total > 0.0:
        hist /= total
    return hist


def plot_observable_summary(
    output_path: Path,
    observable: str,
    values_by_source: dict[str, np.ndarray],
    weights_by_source: dict[str, np.ndarray],
    pair_metrics: dict[str, dict[str, Any]],
    normalize: bool,
) -> None:
    source_order = ["stored_truth", "recalculated_truth", "target_reconstructed"]
    source_colors = {
        "stored_truth": "#000000",
        "recalculated_truth": "#009E73",
        "target_reconstructed": "#D55E00",
    }
    source_labels = {
        "stored_truth": "Stored truth_*",
        "recalculated_truth": "Recalc truth p4",
        "target_reconstructed": "Target invisible reco",
    }
    bins = observable_bins(observable, [values_by_source[name] for name in source_order if name in values_by_source])
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(19, 4.8),
        dpi=180,
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0]},
    )
    ax1d = axes[0]
    for source in source_order:
        values = values_by_source.get(source, np.array([], dtype=np.float64))
        weights = weights_by_source.get(source, np.array([], dtype=np.float64))
        if values.size == 0:
            continue
        hist = weighted_hist(values, weights, bins, normalize)
        ax1d.step(
            bins[:-1],
            hist,
            where="post",
            label=source_labels[source],
            color=source_colors[source],
            linewidth=1.7,
        )
    ax1d.set_title("1D overlay")
    ax1d.set_xlabel(observable)
    ax1d.set_ylabel("Normalized yield" if normalize else "Weighted yield")
    ax1d.grid(alpha=0.2)
    ax1d.legend(frameon=False, fontsize=8)

    pairs = [
        ("stored_truth", "recalculated_truth"),
        ("stored_truth", "target_reconstructed"),
        ("recalculated_truth", "target_reconstructed"),
    ]
    for axis, (left, right) in zip(axes[1:], pairs):
        x = values_by_source.get(left, np.array([], dtype=np.float64))
        y = values_by_source.get(right, np.array([], dtype=np.float64))
        w = weights_by_source.get(left, np.array([], dtype=np.float64))
        if x.size == 0 or y.size == 0 or w.size == 0:
            axis.text(0.5, 0.5, "No entries", ha="center", va="center", transform=axis.transAxes)
            continue
        low = float(np.nanmin(np.concatenate([x, y])))
        high = float(np.nanmax(np.concatenate([x, y])))
        if observable == "theta_cm":
            low, high = 0.0, 1.0
        elif observable.startswith("cos_theta_"):
            low, high = -1.0, 1.0
        hist2d = weighted_hist2d(x, y, w, np.linspace(low, high, 41))
        mesh = axis.pcolormesh(
            np.linspace(low, high, 41),
            np.linspace(low, high, 41),
            hist2d.T,
            cmap="Blues",
            shading="auto",
            vmin=0.0,
            vmax=float(np.nanmax(hist2d)) if np.nanmax(hist2d) > 0 else 1.0,
        )
        fig.colorbar(mesh, ax=axis, fraction=0.046, pad=0.03)
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel(source_labels[left])
        axis.set_ylabel(source_labels[right])
        axis.grid(alpha=0.18)
        metrics = pair_metrics.get(f"{left}__vs__{right}", {})
        rmse = metrics.get("weighted_rmse")
        corr = metrics.get("weighted_corr")
        axis.set_title(
            f"RMSE={rmse:.3g}, corr={corr:.3g}" if rmse is not None and corr is not None else "pair",
            fontsize=9,
        )

    fig.suptitle(observable, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    parquet_paths = resolve_parquet_inputs(args.parquet)
    observables = requested_observables(args.observables)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, PairStats]] = {
        observable: {
            "stored_truth__vs__recalculated_truth": PairStats(),
            "stored_truth__vs__target_reconstructed": PairStats(),
            "recalculated_truth__vs__target_reconstructed": PairStats(),
        }
        for observable in observables
    }
    sampled_values: dict[str, dict[str, list[np.ndarray]]] = {
        observable: {
            "stored_truth": [],
            "recalculated_truth": [],
            "target_reconstructed": [],
            "weights": [],
        }
        for observable in observables
    }

    total_rows_seen = 0
    total_rows_used = 0
    truth_source_info = {"truth_tau_a": set(), "truth_tau_b": set(), "visible_a": set(), "visible_b": set()}

    remaining = args.max_entries
    for path_index, path in enumerate(parquet_paths, start=1):
        print(f"[truth-origin-check] loading {path_index}/{len(parquet_paths)} path={path}", flush=True)
        path_max = None if remaining is None else remaining
        for batch in iter_event_batches(
            path,
            observables,
            args.batch_size,
            path_max,
            args.region,
            args.truth_region_only,
        ):
            total_rows_seen += len(batch)
            selected = apply_event_selection(batch, region=args.region, truth_region_only=args.truth_region_only)
            if len(selected) == 0:
                continue
            total_rows_used += len(selected)
            weights = event_weights(selected, args.weight_field)
            fields = set(selected.fields)
            truth_source_info["truth_tau_a"].add("truth_tau_a_p4" if "truth_tau_a_p4" in fields else ("truth_missing_a_p4+visible" if "truth_missing_a_p4" in fields else "target_slot_fallback"))
            truth_source_info["truth_tau_b"].add("truth_tau_b_p4" if "truth_tau_b_p4" in fields else ("truth_missing_b_p4+visible" if "truth_missing_b_p4" in fields else "target_slot_fallback"))
            truth_source_info["visible_a"].add(truth_visible_field(fields, "a"))
            truth_source_info["visible_b"].add(truth_visible_field(fields, "b"))

            for observable in observables:
                stored = truth_observable_values(selected, observable)
                recalculated = recalculated_truth_values(selected, observable)
                target = target_reconstructed_values(selected, observable)
                values_by_source = {
                    "stored_truth": stored,
                    "recalculated_truth": recalculated,
                    "target_reconstructed": target,
                }

                common_mask = np.ones(len(selected), dtype=bool)
                for values in values_by_source.values():
                    if values is None:
                        common_mask &= False
                    else:
                        common_mask &= np.isfinite(values)
                common_mask &= np.isfinite(weights) & (weights > 0.0)
                if not np.any(common_mask):
                    continue

                stored_masked = values_by_source["stored_truth"][common_mask]
                recalculated_masked = values_by_source["recalculated_truth"][common_mask]
                target_masked = values_by_source["target_reconstructed"][common_mask]
                weights_masked = weights[common_mask]

                stats[observable]["stored_truth__vs__recalculated_truth"].update(
                    stored_masked,
                    recalculated_masked,
                    weights_masked,
                )
                stats[observable]["stored_truth__vs__target_reconstructed"].update(
                    stored_masked,
                    target_masked,
                    weights_masked,
                )
                stats[observable]["recalculated_truth__vs__target_reconstructed"].update(
                    recalculated_masked,
                    target_masked,
                    weights_masked,
                )

                sampled_append(sampled_values[observable]["stored_truth"], stored_masked, args.max_plot_entries)
                sampled_append(sampled_values[observable]["recalculated_truth"], recalculated_masked, args.max_plot_entries)
                sampled_append(sampled_values[observable]["target_reconstructed"], target_masked, args.max_plot_entries)
                sampled_append(sampled_values[observable]["weights"], weights_masked, args.max_plot_entries)

            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    summary: dict[str, Any] = {
        "inputs": [str(path) for path in parquet_paths],
        "rows_seen": total_rows_seen,
        "rows_used": total_rows_used,
        "weight_field": args.weight_field,
        "region": args.region,
        "truth_region_only": args.truth_region_only,
        "truth_sources": {key: sorted(values) for key, values in truth_source_info.items()},
        "observables": {},
    }

    plots_dir = output_dir / "plots"
    for observable in observables:
        pair_metrics = {
            pair_name: pair_stats.summary()
            for pair_name, pair_stats in stats[observable].items()
        }
        values_by_source = {
            source: finalize_sampled(sampled_values[observable][source])
            for source in ("stored_truth", "recalculated_truth", "target_reconstructed")
        }
        weights_for_plot = finalize_sampled(sampled_values[observable]["weights"])
        weights_by_source = {
            "stored_truth": weights_for_plot,
            "recalculated_truth": weights_for_plot,
            "target_reconstructed": weights_for_plot,
        }
        plot_path = plots_dir / f"{sanitize_filename(observable)}.png"
        plot_observable_summary(
            plot_path,
            observable,
            values_by_source,
            weights_by_source,
            pair_metrics,
            args.normalize,
        )
        summary["observables"][observable] = {
            "pair_metrics": pair_metrics,
            "plot": str(plot_path.relative_to(output_dir)),
            "sampled_entries": int(weights_for_plot.size),
        }
        print(f"[truth-origin-check] wrote observable={observable} plot={plot_path}", flush=True)

    summary_path = output_dir / "truth_observable_origin_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[truth-origin-check] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
