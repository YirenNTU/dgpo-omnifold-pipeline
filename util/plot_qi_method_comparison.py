#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantum.observables_builder import get_observable_names
from utils.common_functions import rebuild_p4


vector.register_awkward()

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

METHOD_COLORS = {
    "EveNet": OKABE_ITO["blue"],
    "Baseline": OKABE_ITO["vermillion"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare EveNet-QI export against central MMC/algebraic baseline export."
    )
    parser.add_argument("--evenet-dir", type=Path, required=True, help="Directory containing EveNet <sample>/filtered___*.parquet.")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Directory containing baseline <sample>/filtered___*.parquet.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison plots and metrics JSON.")
    parser.add_argument("--sample-name", default="Ztautau", help="Sample directory name.")
    parser.add_argument("--regions", nargs="+", default=["hadhad", "ee", "mumu", "emu"], help="Regions to compare.")
    parser.add_argument("--max-scatter-points", type=int, default=8000, help="Maximum points per scatter panel.")
    return parser.parse_args()


def load_events(path: Path) -> ak.Array:
    events = ak.from_parquet(path)
    for field in events.fields:
        if field.endswith("_p4"):
            events[field] = rebuild_vector(events[field])
    return events


def rebuild_vector(values: ak.Array) -> ak.Array:
    fields = set(values.fields)
    if {"px", "py", "pz", "E"}.issubset(fields):
        return vector.zip({"px": values["px"], "py": values["py"], "pz": values["pz"], "E": values["E"]})
    if {"x", "y", "z", "t"}.issubset(fields):
        return rebuild_p4(values)
    return values


def parquet_for(root: Path, sample_name: str, region: str) -> Path:
    return root / sample_name / f"filtered___{region}.parquet"


def to_numpy(values: Any, dtype=np.float64) -> np.ndarray:
    return ak.to_numpy(values, allow_missing=False).astype(dtype)


def event_weights(events: ak.Array) -> np.ndarray:
    if "weight" in events.fields:
        weights = to_numpy(events["weight"])
    elif "evenet_weight" in events.fields:
        weights = to_numpy(events["evenet_weight"])
    else:
        weights = np.ones(len(events), dtype=np.float64)
    return np.where(np.isfinite(weights), weights, 0.0)


def weighted_mean_and_unc(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan, np.nan
    values = values[mask]
    weights = weights[mask]
    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    neff = weights.sum() ** 2 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else len(values)
    return float(mean), float(np.sqrt(variance / max(neff, 1.0)))


def weighted_fraction_and_unc(mask: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    valid_weights = np.where(np.isfinite(weights), weights, 0.0)
    total = np.sum(valid_weights)
    if total <= 0:
        return np.nan, np.nan
    fraction = np.sum(valid_weights[mask]) / total
    neff = total ** 2 / np.sum(valid_weights ** 2) if np.sum(valid_weights ** 2) > 0 else len(valid_weights)
    return float(fraction), float(np.sqrt(max(fraction * (1.0 - fraction), 0.0) / max(neff, 1.0)))


def p4_component(p4: ak.Array, name: str) -> np.ndarray:
    if name == "E":
        return to_numpy(p4.E)
    if name == "pt":
        return to_numpy(p4.pt)
    if name == "eta":
        return to_numpy(p4.eta)
    if name == "phi":
        return to_numpy(p4.phi)
    return to_numpy(getattr(p4, name))


def truth_missing_available(events: ak.Array) -> bool:
    return "truth_missing_a_p4" in events.fields and "truth_missing_b_p4" in events.fields


def method_region_metrics(events: ak.Array) -> dict[str, Any]:
    weights = event_weights(events)
    flags_valid = to_numpy(events["flags_valid"], bool) if "flags_valid" in events.fields else np.zeros(len(events), dtype=bool)
    metrics: dict[str, Any] = {
        "num_events": int(len(events)),
        "weighted_yield": float(np.sum(weights)),
    }
    metrics["valid_fraction"], metrics["valid_fraction_unc"] = weighted_fraction_and_unc(flags_valid, weights)

    if "theta_cm" in events.fields:
        theta_qi = to_numpy(events["theta_cm"]) * 2.0 / np.pi > 0.6
        metrics["qi_acceptance"], metrics["qi_acceptance_unc"] = weighted_fraction_and_unc(flags_valid & theta_qi, weights)

    if truth_missing_available(events):
        for leg in ("a", "b"):
            pred = rebuild_vector(events[f"lead_{leg}_missing_p4"])
            truth = rebuild_vector(events[f"truth_missing_{leg}_p4"])
            for comp in ("E", "px", "py", "pz", "pt", "eta", "phi"):
                residual = p4_component(pred, comp) - p4_component(truth, comp)
                abs_residual = np.abs(residual)
                key = f"nu_{leg}_{comp}"
                metrics[f"{key}_bias"], metrics[f"{key}_bias_unc"] = weighted_mean_and_unc(residual, weights)
                metrics[f"{key}_mae"], metrics[f"{key}_mae_unc"] = weighted_mean_and_unc(abs_residual, weights)

    for obs in get_observable_names():
        if obs in events.fields:
            metrics[f"{obs}_mean"], metrics[f"{obs}_mean_unc"] = weighted_mean_and_unc(to_numpy(events[obs]), weights)
    return metrics


def classify_evenet_region(class_name: str) -> str:
    name = str(class_name)
    if name in {"Ztautau_pipi", "Ztautau_pirho", "Ztautau_rhorho"}:
        return "hadhad"
    if name == "Ztautau_ee":
        return "ee"
    if name == "Ztautau_mumu":
        return "mumu"
    if name in {"Ztautau_emu", "Ztautau_mue"}:
        return "emu"
    return "other"


def cms_label(ax, text: str = "Work in progress") -> None:
    ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.12, 1.02, text, transform=ax.transAxes, fontsize=10, style="italic", ha="left", va="bottom")


def plot_metric_summary(metrics: dict[str, dict[str, dict[str, Any]]], output_dir: Path, regions: list[str]) -> None:
    rows = []
    for region in regions:
        for metric_key, label in [
            ("valid_fraction", "Valid fraction"),
            ("qi_acceptance", "QI acceptance"),
            ("nu_a_E_mae", r"$\nu_a$ E MAE [GeV]"),
            ("nu_b_E_mae", r"$\nu_b$ E MAE [GeV]"),
            ("nu_a_pt_mae", r"$\nu_a$ pT MAE [GeV]"),
            ("nu_b_pt_mae", r"$\nu_b$ pT MAE [GeV]"),
        ]:
            if all(metric_key in metrics[method].get(region, {}) for method in ("Baseline", "EveNet")):
                rows.append((region, metric_key, label))

    if not rows:
        return

    fig_height = max(5.0, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(8.5, fig_height), dpi=180)
    y_positions = np.arange(len(rows))
    offsets = {"Baseline": -0.13, "EveNet": 0.13}

    for method in ("Baseline", "EveNet"):
        values = []
        errors = []
        for region, metric_key, _ in rows:
            region_metrics = metrics[method][region]
            values.append(region_metrics.get(metric_key, np.nan))
            errors.append(region_metrics.get(f"{metric_key}_unc", np.nan))
        ax.errorbar(
            values,
            y_positions + offsets[method],
            xerr=errors,
            fmt="o",
            color=METHOD_COLORS[method],
            label=method,
            capsize=2.5,
            markersize=4.5,
        )

    labels = [f"{region}: {label}" for region, _, label in rows]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("Metric value")
    cms_label(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "qi_method_metric_summary.png")
    plt.close(fig)


def sample_indices(mask: np.ndarray, max_points: int) -> np.ndarray:
    indices = np.nonzero(mask)[0]
    if indices.size <= max_points:
        return indices
    rng = np.random.default_rng(12345)
    return np.sort(rng.choice(indices, size=max_points, replace=False))


def plot_neutrino_truth_scatter(
    events_by_method: dict[str, ak.Array],
    output_dir: Path,
    region: str,
    max_points: int,
) -> None:
    if not all(truth_missing_available(events) for events in events_by_method.values()):
        return

    components = ("E", "px", "py", "pz")
    fig, axes = plt.subplots(len(components), 2, figsize=(9.5, 12.0), dpi=180)
    for row, comp in enumerate(components):
        for col, leg in enumerate(("a", "b")):
            ax = axes[row, col]
            all_values = []
            for method, events in events_by_method.items():
                weights = event_weights(events)
                valid = (to_numpy(events["flags_valid"], bool) if "flags_valid" in events.fields else np.ones(len(events), dtype=bool))
                pred = rebuild_vector(events[f"lead_{leg}_missing_p4"])
                truth = rebuild_vector(events[f"truth_missing_{leg}_p4"])
                x = p4_component(truth, comp)
                y = p4_component(pred, comp)
                mask = valid & np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights >= 0)
                chosen = sample_indices(mask, max_points)
                if chosen.size == 0:
                    continue
                ax.scatter(
                    x[chosen],
                    y[chosen],
                    s=5,
                    alpha=0.35,
                    color=METHOD_COLORS[method],
                    label=method if row == 0 and col == 0 else None,
                    rasterized=True,
                )
                all_values.extend([x[chosen], y[chosen]])
            if all_values:
                merged = np.concatenate(all_values)
                lo, hi = np.nanpercentile(merged, [1, 99])
                margin = 0.08 * max(hi - lo, 1.0)
                ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], color="black", lw=1, ls="--")
                ax.set_xlim(lo - margin, hi + margin)
                ax.set_ylim(lo - margin, hi + margin)
            ax.set_xlabel(f"Truth {comp}")
            ax.set_ylabel(f"Pred {comp}")
            ax.set_title(rf"Region {region}, $\nu_{leg}$")
            ax.grid(alpha=0.2)
    cms_label(axes[0, 0])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / f"neutrino_truth_vs_pred_{region}.png")
    plt.close(fig)


def plot_cut_based_vs_evenet(events: ak.Array, output_dir: Path, regions: list[str]) -> None:
    if "evenet_pred_class_name" not in events.fields:
        return
    pred_regions = np.array([classify_evenet_region(name) for name in ak.to_list(events["evenet_pred_class_name"])], dtype=object)
    labels = list(regions) + ["other"]
    matrix = np.zeros((len(regions), len(labels)), dtype=np.float64)
    weights = event_weights(events)
    for i, region in enumerate(regions):
        cut_field = f"{region}_cut"
        if cut_field not in events.fields:
            continue
        cut_mask = to_numpy(events[cut_field], bool)
        for j, pred_region in enumerate(labels):
            matrix[i, j] = np.sum(weights[cut_mask & (pred_regions == pred_region)])

    fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=180)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_xlabel("EveNet predicted broad region")
    ax.set_ylabel("Central cut-based region")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    cms_label(ax)
    fig.colorbar(image, ax=ax, label="Weighted yield")
    fig.tight_layout()
    fig.savefig(output_dir / "cut_based_vs_evenet_region_matrix.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, dict[str, Any]]] = {"Baseline": {}, "EveNet": {}}
    raw_evenet_path = parquet_for(args.evenet_dir, args.sample_name, "raw")
    raw_evenet = load_events(raw_evenet_path) if raw_evenet_path.exists() else None

    for region in args.regions:
        region_events: dict[str, ak.Array] = {}
        for method, root in [("Baseline", args.baseline_dir), ("EveNet", args.evenet_dir)]:
            path = parquet_for(root, args.sample_name, region)
            if not path.exists():
                continue
            events = load_events(path)
            region_events[method] = events
            metrics[method][region] = method_region_metrics(events)

        if set(region_events) == {"Baseline", "EveNet"}:
            plot_neutrino_truth_scatter(region_events, args.output_dir, region, args.max_scatter_points)

    plot_metric_summary(metrics, args.output_dir, args.regions)
    if raw_evenet is not None:
        plot_cut_based_vs_evenet(raw_evenet, args.output_dir, args.regions)

    with (args.output_dir / "qi_method_comparison_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    print(f"[plot-qi-method-comparison] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
