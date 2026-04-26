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
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
UTIL_ROOT = REPO_ROOT / "ml_pipeline" / "util"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UTIL_ROOT) not in sys.path:
    sys.path.insert(0, str(UTIL_ROOT))

from parquet_plot_common import choose_bins
from plot_qi_method_comparison import (
    event_weights,
    is_background_like_region,
    json_safe,
    load_events,
    method_color,
    method_display_name,
    parquet_for,
    physics_observable_specs,
    plot_physics_data_mc_comparisons,
    sanitize_filename,
)
from quantum.observables_builder import build_observables, get_observable_names


DEFAULT_FLOAT = -99.0
OKABE_ITO_BLACK = "#000000"
BASELINE_HADHAD_FINE_CHANNELS = (
    "Ztautau_pipi",
    "Ztautau_pirho",
    "Ztautau_rhorho",
)
BASELINE_FINE_REGION_TO_PARENT = {
    "ee": ("ee", "Ztautau_ee"),
    "emu": ("emu", "Ztautau_emu"),
    "mumu": ("mumu", "Ztautau_mumu"),
    "Ztautau_pipi": ("hadhad", "Ztautau_pipi"),
    "Ztautau_pirho": ("hadhad", "Ztautau_pirho"),
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
    parser.add_argument(
        "--normalize-truth-reco",
        action="store_true",
        help="Normalize truth-vs-reco distributions to unit area instead of plotting absolute weighted yields.",
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
    if root.name == "neutrino_reco" and root.exists():
        return root
    candidate = root / "neutrino_reco"
    if candidate.exists():
        return candidate
    return None


def reconstructed_neutrino_paths(root: Path, sample_name: str, region: str) -> list[Path]:
    reco_root = neutrino_reco_root(root)
    if reco_root is None:
        return []
    if sample_name == "Ztautau":
        if region in BASELINE_FINE_REGION_TO_PARENT:
            parent_region, signal_name = BASELINE_FINE_REGION_TO_PARENT[region]
            candidate = reco_root / parent_region / f"{signal_name}_reconstructed_neutrinos.parquet"
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


def truth_observable_specs(requested: list[str] | None) -> list[tuple[str, str]]:
    names = requested or list(get_observable_names())
    return [(name, observable_latex_label(name)) for name in names]


def observable_latex_label(name: str) -> str:
    if name == "theta_cm":
        return r"$\theta_{\mathrm{CM}}$"
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
    return name.replace("_", " ")


def is_spin_observable(name: str) -> bool:
    return name == "theta_cm" or name.startswith("cos_theta_")


def channel_latex_label(name: str) -> str:
    mapping = {
        "hadhad": r"$\tau_{\mathrm{had}}\tau_{\mathrm{had}}$",
        "ee": r"$ee$",
        "mumu": r"$\mu\mu$",
        "emu": r"$e\mu$",
        "baseline": "baseline",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("Ztautau_"):
        suffix = name.removeprefix("Ztautau_")
        token_map = {"rho": r"\rho", "pi": r"\pi", "e": "e", "mu": r"\mu"}
        tokens: list[str] = []
        index = 0
        ordered_keys = sorted(token_map, key=len, reverse=True)
        while index < len(suffix):
            matched = False
            for key in ordered_keys:
                if suffix.startswith(key, index):
                    tokens.append(token_map[key])
                    index += len(key)
                    matched = True
                    break
            if not matched:
                tokens.append(suffix[index])
                index += 1
        if tokens:
            return r"$\tau\tau\to " + " ".join(tokens) + "$"
    return name.replace("_", " ")


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


def observable_bins(observable: str, values_by_name: dict[str, np.ndarray]) -> np.ndarray:
    if observable == "theta_cm":
        return np.linspace(0.0, np.pi, 41)
    if observable.startswith("cos_theta_"):
        return np.linspace(-1.0, 1.0, 41)
    return choose_bins(values_by_name, num_bins=40)


def observable_2d_limits(observable: str) -> tuple[float, float]:
    if observable == "theta_cm":
        return 0.0, float(np.pi)
    if observable.startswith("cos_theta_"):
        return -1.0, 1.0
    raise ValueError(f"Observable '{observable}' is not configured for 2D truth-vs-reco limits.")


def weighted_hist(values: np.ndarray, weights: np.ndarray, bins: np.ndarray, normalize: bool) -> np.ndarray:
    hist = np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)
    if normalize:
        total = np.sum(hist)
        if total > 0:
            hist = hist / total
    return hist


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


def discover_method_regions(root: Path, sample_name: str, preferred_regions: list[str] | None = None) -> list[str]:
    discovered: list[str] = []
    sample_dir = root / sample_name
    if sample_dir.exists():
        for path in sorted(sample_dir.glob("filtered___*.parquet")):
            region = path.stem.removeprefix("filtered___")
            if region == "raw":
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
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    truth_dir = output_dir / "truth_vs_reco"
    truth_dir.mkdir(parents=True, exist_ok=True)

    for method, root in methods.items():
        regions = method_regions.get(method, [])
        print(f"[preunfolding] truth-vs-reco method={method} native_regions={regions}", flush=True)
        method_summary: dict[str, Any] = {}
        method_dir = truth_dir / sanitize_filename(method)
        method_dir.mkdir(parents=True, exist_ok=True)

        for region in regions:
            print(f"  [region] method={method} region={region}", flush=True)
            paths = method_event_paths(root, signal_sample_name, region)
            if not paths:
                print(f"    [skip] missing signal parquet paths for method={method} region={region}", flush=True)
                continue
            print(
                f"    [load] method={method} sample={signal_sample_name} region={region} "
                f"paths={[str(path) for path in paths]}",
                flush=True,
            )
            events = load_method_events(root, signal_sample_name, region)
            if events is None:
                print(f"    [skip] failed to load events after path resolution method={method} region={region}", flush=True)
                continue
            region_dir = method_dir / sanitize_filename(region)
            region_dir.mkdir(parents=True, exist_ok=True)
            region_summary: dict[str, Any] = {}

            for observable, xlabel in observable_specs:
                reco_values_full = truth_reco_observable_values(events, observable)
                truth_values_full = truth_observable_values(events, observable)
                if reco_values_full is None or truth_values_full is None:
                    print(
                        f"    [skip] method={method} region={region} observable={observable} "
                        f"reco_source={'missing' if reco_values_full is None else 'available'} "
                        f"truth_source={'missing' if truth_values_full is None else 'available'}",
                        flush=True,
                    )
                    continue

                weights_full = event_weights(events)
                truth_mask = valid_truth_reco_mask(truth_values_full, truth_values_full, weights_full)
                valid_mask = valid_truth_reco_mask(truth_values_full, reco_values_full, weights_full)
                if not np.any(truth_mask):
                    print(f"    [skip] method={method} region={region} observable={observable} truth values are all invalid/default", flush=True)
                    continue
                if not np.any(valid_mask):
                    print(f"    [skip] method={method} region={region} observable={observable} no valid truth/reco entries", flush=True)
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

                fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)
                ax.step(
                    bins[:-1],
                    truth_hist,
                    where="post",
                    color=OKABE_ITO_BLACK,
                    linestyle="--",
                    linewidth=1.8,
                    label="Truth",
                )
                ax.step(
                    bins[:-1],
                    reco_hist,
                    where="post",
                    linewidth=1.8,
                    color=method_color(method, 0),
                    label=method_display_name(method),
                )

                ax.set_xlabel(xlabel)
                ax.set_ylabel("Normalized yield" if normalize else "Weighted yield")
                ax.set_title(f"{method_display_name(method)} {channel_latex_label(region)}: {xlabel}")
                ax.grid(alpha=0.25)
                ax.legend(frameon=False)
                fig.tight_layout()

                plot_path = region_dir / f"{sanitize_filename(observable)}.png"
                fig.savefig(plot_path)
                plt.close(fig)

                plot_path_2d = None
                if is_spin_observable(observable):
                    low, high = observable_2d_limits(observable)
                    fig2d, ax2d = plt.subplots(figsize=(6.2, 5.8), dpi=180)
                    hist2d = ax2d.hist2d(
                        truth_valid,
                        reco_valid,
                        bins=[np.linspace(low, high, 51), np.linspace(low, high, 51)],
                        weights=weight_valid,
                        cmap="Blues",
                    )
                    ax2d.plot([low, high], [low, high], color=OKABE_ITO_BLACK, linestyle="--", linewidth=1.2)
                    ax2d.set_xlim(low, high)
                    ax2d.set_ylim(low, high)
                    ax2d.set_xlabel(f"Truth {xlabel}")
                    ax2d.set_ylabel(f"Reco {xlabel}")
                    ax2d.set_title(f"{method_display_name(method)} {channel_latex_label(region)}")
                    fig2d.colorbar(hist2d[3], ax=ax2d, label="Weighted yield")
                    fig2d.tight_layout()
                    plot_path_2d = region_dir / f"{sanitize_filename(observable)}_2d.png"
                    fig2d.savefig(plot_path_2d)
                    plt.close(fig2d)

                print(
                    f"    [write] method={method} region={region} observable={observable} plot={plot_path}",
                    flush=True,
                )

                region_summary[observable] = {
                    "plot": str(plot_path.relative_to(output_dir)),
                    "plot_2d": str(plot_path_2d.relative_to(output_dir)) if plot_path_2d is not None else None,
                    "normalize": normalize,
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

            if region_summary:
                method_summary[region] = region_summary

        if method_summary:
            summary[method] = method_summary

    return summary


def truth_reco_observable_values(events: ak.Array, observable: str) -> np.ndarray | None:
    if observable in events.fields:
        return to_numpy(events[observable], np.float64)
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

    required_fields = {"truth_missing_a_p4", "truth_missing_b_p4", "lead_a_visible_p4", "lead_b_visible_p4"}
    if not required_fields.issubset(set(events.fields)):
        return None

    truth_tau_a = events["truth_missing_a_p4"] + events["lead_a_visible_p4"]
    truth_tau_b = events["truth_missing_b_p4"] + events["lead_b_visible_p4"]
    observables = build_observables(
        truth_tau_a,
        truth_tau_b,
        events["lead_a_visible_p4"],
        events["lead_b_visible_p4"],
    )
    if observable not in observables:
        return None
    return to_numpy(observables[observable], np.float64)


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


def plot_truth_metric_summary(
    truth_summary: dict[str, Any],
    methods: dict[str, Path],
    observable_specs: list[tuple[str, str]],
    output_dir: Path,
    metric: str = "pearson",
) -> dict[str, Any]:
    summary_dir = output_dir / "truth_vs_reco_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_summary: dict[str, Any] = {}
    method_names = list(methods)

    all_regions = []
    for method_info in truth_summary.values():
        all_regions.extend(method_info.keys())
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
                rows.append(
                    {
                        "method": method,
                        "method_index": method_index,
                        "region": region,
                        "value": value,
                        "uncertainty": uncertainty,
                        "precision": metric_precision(value, uncertainty),
                        "num_events": metrics.get("num_events"),
                    }
                )

        if not rows:
            continue

        active_regions = [
            region
            for region in unique_regions
            if region != "baseline" and any(row["region"] == region for row in rows)
        ]
        rows = [row for row in rows if row["region"] in set(active_regions)]
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
            region_rows = [row for row in rows if row["region"] == region]
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
        ax.set_yticklabels([channel_latex_label(region) for region in active_regions])
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
            "regions": sorted({row["region"] for row in rows}),
        }
        print(f"[preunfolding] wrote truth-summary observable={observable} metric={metric} plot={plot_path}", flush=True)

    return plot_summary


def write_report(
    output_dir: Path,
    methods: dict[str, Path],
    truth_summary: dict[str, Any],
    truth_metric_summary: dict[str, Any],
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

    method_regions = {
        method: discover_method_regions(root, args.signal_sample_name, args.regions)
        for method, root in methods.items()
    }
    observable_specs = truth_observable_specs(args.truth_observables)
    print(f"[preunfolding] signal_sample={args.signal_sample_name}", flush=True)
    print(f"[preunfolding] method_regions={method_regions}", flush=True)
    print(f"[preunfolding] truth_observables={[name for name, _ in observable_specs]}", flush=True)

    truth_summary = plot_truth_vs_reco_by_method_and_region(
        method_regions=method_regions,
        methods=methods,
        signal_sample_name=args.signal_sample_name,
        observable_specs=observable_specs,
        output_dir=args.output_dir,
        normalize=args.normalize_truth_reco,
    )
    print(f"[preunfolding] finished truth-vs-reco methods={list(truth_summary)}", flush=True)
    truth_metric_summary = plot_truth_metric_summary(
        truth_summary=truth_summary,
        methods=methods,
        observable_specs=observable_specs,
        output_dir=args.output_dir,
        metric="pearson",
    )
    print(f"[preunfolding] finished truth summary plots observables={list(truth_metric_summary)}", flush=True)

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
        "truth_vs_reco": truth_summary,
        "truth_metric_summary": truth_metric_summary,
        "data_mc_control": control_summary,
        "control_observable_defaults": control_observables,
    }
    with (args.output_dir / "preunfolding_validation_summary.json").open("w") as handle:
        json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
    write_report(args.output_dir, methods, truth_summary, truth_metric_summary, control_summary, method_regions)
    print(f"[preunfolding] wrote summary_json={args.output_dir / 'preunfolding_validation_summary.json'}", flush=True)
    print(f"[preunfolding] wrote report_md={args.output_dir / 'preunfolding_validation_report.md'}", flush=True)

    print(f"[plot-preunfolding-validation] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
