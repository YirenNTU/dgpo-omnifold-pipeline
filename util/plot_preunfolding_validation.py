#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    cms_label,
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
from quantum.observables_builder import get_observable_names


DEFAULT_FLOAT = -99.0
OKABE_ITO_BLACK = "#000000"


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


def truth_observable_specs(requested: list[str] | None) -> list[tuple[str, str]]:
    names = requested or list(get_observable_names())
    labels = {"theta_cm": r"$\theta_{CM}$"}
    return [(name, labels.get(name, name.replace("_", " "))) for name in names]


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
    sample_dir = root / sample_name
    if not sample_dir.exists():
        return []

    discovered: list[str] = []
    for path in sorted(sample_dir.glob("filtered___*.parquet")):
        region = path.stem.removeprefix("filtered___")
        if region == "raw":
            continue
        discovered.append(region)

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
            path = parquet_for(root, signal_sample_name, region)
            if not path.exists():
                print(f"    [skip] missing signal parquet path={path}", flush=True)
                continue
            print(f"    [load] method={method} sample={signal_sample_name} region={region} path={path}", flush=True)
            events = load_events(path)
            region_dir = method_dir / sanitize_filename(region)
            region_dir.mkdir(parents=True, exist_ok=True)
            region_summary: dict[str, Any] = {}

            for observable, xlabel in observable_specs:
                truth_field = f"truth_{observable}"
                if observable not in events.fields or truth_field not in events.fields:
                    print(
                        f"    [skip] method={method} region={region} observable={observable} "
                        f"missing_fields={[field for field in (observable, truth_field) if field not in events.fields]}",
                        flush=True,
                    )
                    continue

                truth_values_full = to_numpy(events[truth_field], np.float64)
                reco_values_full = to_numpy(events[observable], np.float64)
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
                ax.set_title(f"{method_display_name(method)} {region}: {observable}")
                ax.grid(alpha=0.25)
                cms_label(ax)
                ax.legend(frameon=False)
                fig.tight_layout()

                plot_path = region_dir / f"{sanitize_filename(observable)}.png"
                fig.savefig(plot_path)
                plt.close(fig)
                print(
                    f"    [write] method={method} region={region} observable={observable} plot={plot_path}",
                    flush=True,
                )

                region_summary[observable] = {
                    "plot": str(plot_path.relative_to(output_dir)),
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

        active_regions = [region for region in unique_regions if any(row["region"] == region for row in rows)]
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

        x_text_sigma = 1.03
        x_text_precision = 1.18
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
                sigma_text = f"{row['uncertainty'] * 100:.2f}" if np.isfinite(row["uncertainty"]) else "n/a"
                precision_text = f"{row['precision']:.2f}%" if np.isfinite(row["precision"]) else "n/a"
                ax.text(
                    x_text_sigma,
                    y,
                    sigma_text,
                    color=color,
                    fontsize=8,
                    va="center",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
                ax.text(
                    x_text_precision,
                    y,
                    precision_text,
                    color=color,
                    fontsize=8,
                    va="center",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
                ax.text(
                    row["value"],
                    y + 0.06,
                    method_display_name(row["method"]),
                    color=color,
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )

        ax.text(x_text_sigma, 1.02, r"$\sigma \times 100$", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
        ax.text(x_text_precision, 1.02, "Precision", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
        ax.set_yticks(y_base)
        ax.set_yticklabels([format_method_region_label(region) for region in active_regions])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", alpha=0.18, linestyle=":")
        ax.set_xlabel(summary_metric_label(metric))
        ax.set_ylabel("Channel / Region")
        ax.set_title(f"{observable}: reco vs truth summary")
        cms_label(ax)

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
    control_summary: dict[str, Any] = {}
    for method, root in methods.items():
        native_regions = method_regions.get(method, [])
        if not native_regions:
            print(f"[preunfolding] skip control plots method={method} no native regions discovered", flush=True)
            control_summary[method] = {}
            continue
        method_control_summary = plot_physics_data_mc_comparisons(
            methods={method: root},
            output_dir=args.output_dir,
            regions=native_regions,
            data_sample_name=args.data_sample_name,
            mc_sample_names=list(args.mc_sample_names),
            requested_observables=args.control_observables,
        )
        control_summary[method] = method_control_summary.get(method, {})
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
        "control_observable_defaults": [name for name, _, _ in physics_observable_specs(args.control_observables)],
    }
    with (args.output_dir / "preunfolding_validation_summary.json").open("w") as handle:
        json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
    write_report(args.output_dir, methods, truth_summary, truth_metric_summary, control_summary, method_regions)
    print(f"[preunfolding] wrote summary_json={args.output_dir / 'preunfolding_validation_summary.json'}", flush=True)
    print(f"[preunfolding] wrote report_md={args.output_dir / 'preunfolding_validation_report.md'}", flush=True)

    print(f"[plot-preunfolding-validation] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
