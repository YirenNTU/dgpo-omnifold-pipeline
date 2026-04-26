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
        default=["baseline", "hadhad", "ee", "mumu", "emu"],
        help="Common central regions to compare before unfolding.",
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


def available_regions(methods: dict[str, Path], signal_sample_name: str, candidate_regions: list[str]) -> list[str]:
    active: list[str] = []
    for region in candidate_regions:
        if any(parquet_for(root, signal_sample_name, region).exists() for root in methods.values()):
            active.append(region)
    return active


def plot_truth_vs_reco_by_region(
    methods: dict[str, Path],
    signal_sample_name: str,
    regions: list[str],
    observable_specs: list[tuple[str, str]],
    output_dir: Path,
    normalize: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    truth_dir = output_dir / "truth_vs_reco"
    truth_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        region_dir = truth_dir / sanitize_filename(region)
        region_dir.mkdir(parents=True, exist_ok=True)
        region_summary: dict[str, Any] = {}

        events_by_method: dict[str, ak.Array] = {}
        for method, root in methods.items():
            path = parquet_for(root, signal_sample_name, region)
            if path.exists():
                events_by_method[method] = load_events(path)
        if not events_by_method:
            continue

        for observable, xlabel in observable_specs:
            truth_field = f"truth_{observable}"
            available_methods = [
                method for method, events in events_by_method.items() if observable in events.fields and truth_field in events.fields
            ]
            if not available_methods:
                continue

            truth_reference_method = available_methods[0]
            truth_events = events_by_method[truth_reference_method]
            truth_values_full = to_numpy(truth_events[truth_field], np.float64)
            truth_weights_full = event_weights(truth_events)
            truth_mask = np.isfinite(truth_values_full) & np.isfinite(truth_weights_full) & (truth_weights_full > 0)
            truth_mask &= ~np.isclose(truth_values_full, DEFAULT_FLOAT)
            if not np.any(truth_mask):
                continue

            values_for_bins: dict[str, np.ndarray] = {f"truth:{truth_reference_method}": truth_values_full[truth_mask]}
            reco_payload: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

            for method in available_methods:
                events = events_by_method[method]
                truth_values = to_numpy(events[truth_field], np.float64)
                reco_values = to_numpy(events[observable], np.float64)
                weights = event_weights(events)
                mask = valid_truth_reco_mask(truth_values, reco_values, weights)
                if not np.any(mask):
                    continue

                truth_valid = truth_values[mask]
                reco_valid = reco_values[mask]
                weight_valid = weights[mask]
                values_for_bins[f"reco:{method}"] = reco_valid
                reco_payload[method] = (
                    reco_valid,
                    weight_valid,
                    {
                        "num_events": int(np.count_nonzero(mask)),
                        "weight_sum": float(np.sum(weight_valid)),
                        "truth_mean": weighted_mean(truth_valid, weight_valid),
                        "reco_mean": weighted_mean(reco_valid, weight_valid),
                        "bias": weighted_mean(reco_valid - truth_valid, weight_valid),
                        "mae": weighted_mae(truth_valid, reco_valid, weight_valid),
                        "rmse": weighted_rmse(truth_valid, reco_valid, weight_valid),
                        "pearson": weighted_pearson(truth_valid, reco_valid, weight_valid),
                    },
                )

            if not reco_payload:
                continue

            bins = observable_bins(observable, values_for_bins)
            truth_hist = weighted_hist(truth_values_full[truth_mask], truth_weights_full[truth_mask], bins, normalize)

            fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)
            ax.step(
                bins[:-1],
                truth_hist,
                where="post",
                color=OKABE_ITO_BLACK,
                linestyle="--",
                linewidth=1.8,
                label=f"Truth ({method_display_name(truth_reference_method)})",
            )

            for method_index, method in enumerate(available_methods):
                payload = reco_payload.get(method)
                if payload is None:
                    continue
                reco_valid, weight_valid, _ = payload
                reco_hist = weighted_hist(reco_valid, weight_valid, bins, normalize)
                ax.step(
                    bins[:-1],
                    reco_hist,
                    where="post",
                    linewidth=1.8,
                    color=method_color(method, method_index),
                    label=method_display_name(method),
                )

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Normalized yield" if normalize else "Weighted yield")
            ax.set_title(f"{region}: {observable}")
            ax.grid(alpha=0.25)
            cms_label(ax)
            ax.legend(frameon=False)
            fig.tight_layout()

            plot_path = region_dir / f"{sanitize_filename(observable)}.png"
            fig.savefig(plot_path)
            plt.close(fig)

            region_summary[observable] = {
                "plot": str(plot_path.relative_to(output_dir)),
                "truth_reference_method": truth_reference_method,
                "normalize": normalize,
                "methods": {method: metrics for method, (_, _, metrics) in reco_payload.items()},
            }

        if region_summary:
            summary[region] = region_summary

    return summary


def write_report(
    output_dir: Path,
    methods: dict[str, Path],
    truth_summary: dict[str, Any],
    control_summary: dict[str, Any],
) -> None:
    lines = [
        "# Pre-Unfolding Validation",
        "",
        "## Methods",
        "",
        "| Method | Root |",
        "|---|---|",
    ]
    for method, root in methods.items():
        lines.append(f"| {method} | `{root}` |")

    lines.extend(
        [
            "",
            "## Truth-vs-Reco Coverage",
            "",
            "| Region | Observables with plots |",
            "|---|---:|",
        ]
    )
    for region, region_info in truth_summary.items():
        lines.append(f"| {region} | {len(region_info)} |")

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

    candidate_regions = ["other", *args.regions]
    regions = [region for region in available_regions(methods, args.signal_sample_name, candidate_regions) if not is_background_like_region(region)]
    observable_specs = truth_observable_specs(args.truth_observables)

    truth_summary = plot_truth_vs_reco_by_region(
        methods=methods,
        signal_sample_name=args.signal_sample_name,
        regions=regions,
        observable_specs=observable_specs,
        output_dir=args.output_dir,
        normalize=args.normalize_truth_reco,
    )

    control_summary = plot_physics_data_mc_comparisons(
        methods=methods,
        output_dir=args.output_dir,
        regions=regions,
        data_sample_name=args.data_sample_name,
        mc_sample_names=list(args.mc_sample_names),
        requested_observables=args.control_observables,
    )

    summary = {
        "methods": {method: str(path) for method, path in methods.items()},
        "signal_sample_name": args.signal_sample_name,
        "regions": regions,
        "truth_observables": [name for name, _ in observable_specs],
        "truth_vs_reco": truth_summary,
        "data_mc_control": control_summary,
        "control_observable_defaults": [name for name, _, _ in physics_observable_specs(args.control_observables)],
    }
    with (args.output_dir / "preunfolding_validation_summary.json").open("w") as handle:
        json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
    write_report(args.output_dir, methods, truth_summary, control_summary)

    print(f"[plot-preunfolding-validation] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
