#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
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

TAU_MASS = 1.777  # GeV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stored truth observables against observables rebuilt from "
            "target missing plus visible tau inputs."
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
        help="Optional observable subset. Defaults to theta_cm, mtautau, and all cos_theta_*.",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="Rows per parquet streaming batch.")
    parser.add_argument("--max-entries", type=int, default=None, help="Optional global row cap.")
    parser.add_argument("--region", default=None, help="Optional region name; if <region>_cut exists, keep only that selection.")
    parser.add_argument(
        "--truth-region-only",
        action="store_true",
        help="Require truth_QI_region == 1 when that field exists.",
    )
    parser.add_argument(
        "--weight-field",
        choices=["auto", "weight", "central_weight", "evenet_weight", "unit"],
        default="unit",
        help="Weight source for metrics and 1D histograms.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the 1D histograms to unit area.",
    )
    return parser.parse_args()


def sanitize_filename(name: str) -> str:
    chars = [char if char.isalnum() or char in {"_", "-", "."} else "_" for char in name]
    return "".join(chars).strip("_")


def resolve_parquet_inputs(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        expanded = Path(item).expanduser()
        if expanded.is_dir():
            paths.extend(sorted(expanded.glob("*.parquet")))
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


def requested_observables(raw: list[str] | None) -> list[str]:
    return list(raw) if raw else list(get_observable_names())


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(values.fields)
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


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


def build_momentum4d_with_mass(obj: ak.Array, mass: float) -> ak.Array:
    px = np.asarray(obj.px, dtype=np.float64)
    py = np.asarray(obj.py, dtype=np.float64)
    pz = np.asarray(obj.pz, dtype=np.float64)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return build_momentum4d(px, py, pz, energy)


def massless_p4_from_pt_eta_phi(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> ak.Array:
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        pt * np.cosh(eta),
    )


def p4_from_slot_features(events: ak.Array, prefix: str, slot: int) -> ak.Array:
    energy = to_numpy(events[f"{prefix}_slot{slot}_energy"], np.float64)
    pt = to_numpy(events[f"{prefix}_slot{slot}_pt"], np.float64)
    eta = to_numpy(events[f"{prefix}_slot{slot}_eta"], np.float64)
    phi = to_numpy(events[f"{prefix}_slot{slot}_phi"], np.float64)
    return build_momentum4d(
        pt * np.cos(phi),
        pt * np.sin(phi),
        pt * np.sinh(eta),
        energy,
    )


def massless_p4_from_slot_features(events: ak.Array, prefix: str, slot: int) -> ak.Array:
    pt = to_numpy(events[f"{prefix}_slot{slot}_pt"], np.float64)
    eta = to_numpy(events[f"{prefix}_slot{slot}_eta"], np.float64)
    phi = to_numpy(events[f"{prefix}_slot{slot}_phi"], np.float64)
    return massless_p4_from_pt_eta_phi(pt, eta, phi)


def visible_tau_p4(events: ak.Array, slot: int) -> ak.Array:
    fields = set(events.fields)
    direct_field = f"lead_{'a' if slot == 0 else 'b'}_visible_p4"
    if direct_field in fields:
        return rebuild_vector(events[direct_field])
    required = {
        f"tau_vis_prong_slot{slot}_energy",
        f"tau_vis_prong_slot{slot}_pt",
        f"tau_vis_prong_slot{slot}_eta",
        f"tau_vis_prong_slot{slot}_phi",
    }
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Missing visible-tau fields for slot {slot}: {missing}")
    return p4_from_slot_features(events, "tau_vis_prong", slot)


def target_missing_p4(events: ak.Array, slot: int) -> ak.Array:
    required = {
        f"target_invisible_slot{slot}_pt",
        f"target_invisible_slot{slot}_eta",
        f"target_invisible_slot{slot}_phi",
    }
    fields = set(events.fields)
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"Missing target-invisible fields for slot {slot}: {missing}")
    return massless_p4_from_slot_features(events, "target_invisible", slot)


def parquet_columns_to_load(path: Path, observables: list[str], region: str | None, truth_region_only: bool) -> list[str] | None:
    schema = pq.read_schema(path)
    available = {field.name for field in schema}
    columns: set[str] = set()
    for observable in observables:
        columns.add(f"truth_{observable}")
    for slot in (0, 1):
        columns.update(
            {
                f"target_invisible_slot{slot}_pt",
                f"target_invisible_slot{slot}_eta",
                f"target_invisible_slot{slot}_phi",
                f"tau_vis_prong_slot{slot}_energy",
                f"tau_vis_prong_slot{slot}_pt",
                f"tau_vis_prong_slot{slot}_eta",
                f"tau_vis_prong_slot{slot}_phi",
            }
        )
    columns.update({"lead_a_visible_p4", "lead_b_visible_p4", "weight", "central_weight", "evenet_weight"})
    if region is not None:
        columns.add(f"{region}_cut")
    if truth_region_only:
        columns.add("truth_QI_region")
    selected = [name for name in sorted(columns) if name in available]
    return selected if selected else None


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


def apply_event_selection(events: ak.Array, region: str | None, truth_region_only: bool) -> ak.Array:
    mask = np.ones(len(events), dtype=bool)
    if region is not None:
        region_field = f"{region}_cut"
        if region_field in events.fields:
            mask &= to_numpy(events[region_field], np.int64) > 0
    if truth_region_only and "truth_QI_region" in events.fields:
        mask &= to_numpy(events["truth_QI_region"], np.int64) > 0
    return events[mask]


def event_weights(events: ak.Array, mode: str) -> np.ndarray:
    if mode == "unit":
        return np.ones(len(events), dtype=np.float64)
    if mode == "auto":
        candidates = ["weight", "central_weight", "evenet_weight"]
    else:
        candidates = [mode]
    for field in candidates:
        if field in events.fields:
            weights = to_numpy(events[field], np.float64)
            return np.where(np.isfinite(weights), weights, 0.0)
    return np.ones(len(events), dtype=np.float64)


def truth_values(events: ak.Array, observable: str) -> np.ndarray:
    return to_numpy(events[f"truth_{observable}"], np.float64)


def target_reconstructed_values(events: ak.Array, observable: str) -> np.ndarray:
    visible_a = visible_tau_p4(events, 0)
    visible_b = visible_tau_p4(events, 1)
    target_a = target_missing_p4(events, 0)
    target_b = target_missing_p4(events, 1)
    tau_a = build_momentum4d_with_mass(visible_a + target_a, TAU_MASS)
    tau_b = build_momentum4d_with_mass(visible_b + target_b, TAU_MASS)
    observables = build_observables(
        tau_a_p4=tau_a,
        tau_b_p4=tau_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    return np.asarray(observables[observable], dtype=np.float64)


def observable_limits(observable: str, truth: np.ndarray, reco: np.ndarray) -> tuple[float, float]:
    if observable == "theta_cm":
        return 0.0, 1.0
    if observable.startswith("cos_theta_"):
        return -1.0, 1.0
    merged = np.concatenate([truth, reco])
    low = float(np.nanpercentile(merged, 0.5))
    high = float(np.nanpercentile(merged, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return -1.0, 1.0
    padding = 0.05 * (high - low)
    return low - padding, high + padding


def weighted_hist(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, normalize: bool) -> np.ndarray:
    hist = np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)
    if normalize:
        total = np.sum(hist)
        if total > 0.0:
            hist /= total
    return hist


def compare_summary(truth: np.ndarray, reco: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(truth) & np.isfinite(reco)
    if not np.any(finite):
        return {
            "count": 0,
            "mean_diff": None,
            "mean_abs_diff": None,
            "rmse": None,
            "corr": None,
        }
    truth = truth[finite]
    reco = reco[finite]
    diff = reco - truth
    corr = None
    if truth.size >= 2:
        corr_matrix = np.corrcoef(truth, reco)
        corr_value = corr_matrix[0, 1]
        corr = float(corr_value) if np.isfinite(corr_value) else None
    return {
        "count": int(truth.size),
        "mean_diff": float(np.mean(diff)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "corr": corr,
    }


def plot_observable(
    output_path: Path,
    observable: str,
    truth: np.ndarray,
    reco: np.ndarray,
    weights: np.ndarray,
    normalize: bool,
) -> None:
    low, high = observable_limits(observable, truth, reco)
    bins = np.linspace(low, high, 60)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=180, gridspec_kw={"width_ratios": [1.1, 1.0]})

    truth_hist = weighted_hist(truth, weights, bins, normalize)
    reco_hist = weighted_hist(reco, weights, bins, normalize)
    axes[0].step(bins[:-1], truth_hist, where="post", color="black", linewidth=1.7, label="Stored truth")
    axes[0].step(bins[:-1], reco_hist, where="post", color="#D55E00", linewidth=1.7, label="Target missing + visible")
    axes[0].set_xlabel(observable)
    axes[0].set_ylabel("Normalized yield" if normalize else "Weighted yield")
    axes[0].set_title("1D overlay")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    hist2d = np.histogram2d(truth, reco, bins=[bins, bins], weights=weights)[0].astype(np.float64)
    mesh = axes[1].pcolormesh(bins, bins, hist2d.T, cmap="Blues", shading="auto", vmin=0.0)
    fig.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.03, label="Entries")
    axes[1].plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)
    axes[1].set_xlabel(f"Stored truth {observable}")
    axes[1].set_ylabel(f"Target missing + visible {observable}")
    axes[1].set_title("2D comparison")
    axes[1].grid(alpha=0.16)

    fig.suptitle(observable)
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

    collected_truth: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_reco: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    collected_weights: dict[str, list[np.ndarray]] = {observable: [] for observable in observables}
    rows_seen = 0
    rows_used = 0

    remaining = args.max_entries
    for path_index, path in enumerate(parquet_paths, start=1):
        print(f"[check-target-missing-vs-truth] loading {path_index}/{len(parquet_paths)} path={path}", flush=True)
        path_limit = None if remaining is None else remaining
        for batch in iter_event_batches(
            path,
            observables,
            args.batch_size,
            path_limit,
            args.region,
            args.truth_region_only,
        ):
            rows_seen += len(batch)
            selected = apply_event_selection(batch, args.region, args.truth_region_only)
            if len(selected) == 0:
                continue
            rows_used += len(selected)
            weights = event_weights(selected, args.weight_field)
            for observable in observables:
                truth_field = f"truth_{observable}"
                if truth_field not in selected.fields:
                    continue
                truth = truth_values(selected, observable)
                reco = target_reconstructed_values(selected, observable)
                finite = np.isfinite(truth) & np.isfinite(reco) & np.isfinite(weights) & (weights > 0.0)
                if not np.any(finite):
                    continue
                collected_truth[observable].append(truth[finite])
                collected_reco[observable].append(reco[finite])
                collected_weights[observable].append(weights[finite])
            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    summary: dict[str, Any] = {
        "inputs": [str(path) for path in parquet_paths],
        "rows_seen": rows_seen,
        "rows_used": rows_used,
        "region": args.region,
        "truth_region_only": args.truth_region_only,
        "weight_field": args.weight_field,
        "observables": {},
    }

    for observable in observables:
        if not collected_truth[observable]:
            continue
        truth = np.concatenate(collected_truth[observable])
        reco = np.concatenate(collected_reco[observable])
        weights = np.concatenate(collected_weights[observable])
        summary["observables"][observable] = compare_summary(truth, reco)
        plot_path = output_dir / "plots" / f"{sanitize_filename(observable)}.png"
        plot_observable(plot_path, observable, truth, reco, weights, args.normalize)
        summary["observables"][observable]["plot"] = str(plot_path.relative_to(output_dir))
        print(f"[check-target-missing-vs-truth] wrote observable={observable} plot={plot_path}", flush=True)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[check-target-missing-vs-truth] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
