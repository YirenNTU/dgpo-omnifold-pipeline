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
from parquet_plot_common import (
    OKABE_ITO,
    choose_bins,
    method_color,
    plot_from_histograms,
    summarize_invalid_hist_values,
)


vector.register_awkward()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare central-schema QI exports for multiple neutrino-reconstruction methods."
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help="Method spec in the form Label:/path/to/central_schema_tree. Repeat for Baseline/EveNet variants.",
    )
    parser.add_argument("--evenet-dir", type=Path, default=None, help="Legacy shortcut for --method EveNet:<dir>.")
    parser.add_argument("--baseline-dir", type=Path, default=None, help="Legacy shortcut for --method Baseline:<dir>.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison plots and metrics JSON.")
    parser.add_argument("--sample-name", default="Ztautau", help="Sample directory name.")
    parser.add_argument("--data-sample-name", default="data94", help="Data sample directory name for data-vs-MC plots.")
    parser.add_argument(
        "--mc-sample-names",
        nargs="+",
        default=["Ztautau", "Zll", "Zqq"],
        help="MC sample directory names for data-vs-MC plots.",
    )
    parser.add_argument("--regions", nargs="+", default=["hadhad", "ee", "mumu", "emu"], help="Regions to compare.")
    parser.add_argument(
        "--metric-grouping",
        choices=["region", "evenet-channel"],
        default="region",
        help=(
            "Grouping for final metric summary plots. 'region' reads filtered___<region>.parquet. "
            "'evenet-channel' groups filtered___raw.parquet by evenet_pred_class_name."
        ),
    )
    parser.add_argument(
        "--physics-observables",
        nargs="+",
        default=None,
        help=(
            "Optional observable list for data-vs-MC plots. If omitted, use QI observables plus "
            "reconstructed tau-pair kinematics."
        ),
    )
    parser.add_argument("--max-scatter-points", type=int, default=8000, help="Maximum points per scatter panel.")
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


def method_display_name(method: str) -> str:
    return method.removeprefix("EveNet-")


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


def delta_phi(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(np.arctan2(np.sin(first - second), np.cos(first - second)))


def extract_physics_observable(events: ak.Array, observable: str) -> np.ndarray:
    if observable in events.fields:
        return to_numpy(events[observable])

    if observable in {"reco_tau_pair_mass", "reco_tau_pair_pt", "reco_tau_pair_E"}:
        if "reco_tau_a_p4" not in events.fields or "reco_tau_b_p4" not in events.fields:
            return np.array([], dtype=np.float64)
        pair = rebuild_vector(events["reco_tau_a_p4"]) + rebuild_vector(events["reco_tau_b_p4"])
        component = {"reco_tau_pair_mass": "mass", "reco_tau_pair_pt": "pt", "reco_tau_pair_E": "E"}[observable]
        return p4_component(pair, component)

    if observable in {"reco_tau_delta_eta", "reco_tau_delta_phi"}:
        if "reco_tau_a_p4" not in events.fields or "reco_tau_b_p4" not in events.fields:
            return np.array([], dtype=np.float64)
        tau_a = rebuild_vector(events["reco_tau_a_p4"])
        tau_b = rebuild_vector(events["reco_tau_b_p4"])
        if observable == "reco_tau_delta_eta":
            return np.abs(p4_component(tau_a, "eta") - p4_component(tau_b, "eta"))
        return delta_phi(p4_component(tau_a, "phi"), p4_component(tau_b, "phi"))

    if observable.startswith("reco_tau_"):
        component = observable.removeprefix("reco_tau_")
        if component in {"E", "px", "py", "pz", "pt", "eta", "phi"}:
            values = []
            for leg in ("a", "b"):
                field = f"reco_tau_{leg}_p4"
                if field in events.fields:
                    values.append(p4_component(rebuild_vector(events[field]), component))
            if values:
                return np.concatenate(values)

    if observable.startswith("visible_tau_"):
        component = observable.removeprefix("visible_tau_")
        if component in {"E", "px", "py", "pz", "pt", "eta", "phi"}:
            values = []
            for leg in ("a", "b"):
                field = f"lead_{leg}_visible_p4"
                if field in events.fields:
                    values.append(p4_component(rebuild_vector(events[field]), component))
            if values:
                return np.concatenate(values)

    if observable in {"visible_tau_pair_mass", "visible_tau_pair_pt"}:
        if "lead_a_visible_p4" not in events.fields or "lead_b_visible_p4" not in events.fields:
            return np.array([], dtype=np.float64)
        pair = rebuild_vector(events["lead_a_visible_p4"]) + rebuild_vector(events["lead_b_visible_p4"])
        component = "mass" if observable.endswith("_mass") else "pt"
        return p4_component(pair, component)

    return np.array([], dtype=np.float64)


def physics_observable_specs(requested: list[str] | None) -> list[tuple[str, str, bool]]:
    if requested:
        names = requested
    else:
        names = [
            "reco_tau_pair_mass",
            "reco_tau_pair_pt",
            "reco_tau_delta_eta",
            "reco_tau_delta_phi",
            "reco_tau_E",
            "reco_tau_px",
            "reco_tau_py",
            "reco_tau_pz",
            "visible_tau_E",
            "visible_tau_px",
            "visible_tau_py",
            "visible_tau_pz",
            "visible_tau_pt",
            "visible_tau_eta",
            "visible_tau_phi",
            "visible_tau_pair_mass",
            "visible_tau_pair_pt",
            *get_observable_names(),
        ]

    labels = {
        "reco_tau_pair_mass": r"$m_{\tau\tau}^{reco}$ [GeV]",
        "reco_tau_pair_pt": r"$p_{T,\tau\tau}^{reco}$ [GeV]",
        "reco_tau_delta_eta": r"$|\Delta\eta(\tau,\tau)|$",
        "reco_tau_delta_phi": r"$|\Delta\phi(\tau,\tau)|$",
        "reco_tau_E": r"$E_\tau^{reco}$ [GeV]",
        "reco_tau_px": r"$p_{x,\tau}^{reco}$ [GeV]",
        "reco_tau_py": r"$p_{y,\tau}^{reco}$ [GeV]",
        "reco_tau_pz": r"$p_{z,\tau}^{reco}$ [GeV]",
        "visible_tau_E": r"$E_\tau^{vis}$ [GeV]",
        "visible_tau_px": r"$p_{x,\tau}^{vis}$ [GeV]",
        "visible_tau_py": r"$p_{y,\tau}^{vis}$ [GeV]",
        "visible_tau_pz": r"$p_{z,\tau}^{vis}$ [GeV]",
        "visible_tau_pt": r"$p_{T,\tau}^{vis}$ [GeV]",
        "visible_tau_eta": r"$\eta_\tau^{vis}$",
        "visible_tau_phi": r"$\phi_\tau^{vis}$",
        "visible_tau_pair_mass": r"$m_{\tau\tau}^{vis}$ [GeV]",
        "visible_tau_pair_pt": r"$p_{T,\tau\tau}^{vis}$ [GeV]",
        "theta_cm": r"$2\arccos|\cos\theta_{CM}|/\pi$",
        "mtautau": r"$m_{\tau\tau}$ [GeV]",
    }
    seen = set()
    specs = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        label = labels.get(name, name.replace("_", " "))
        log_scale = name.endswith("_mass") or name.endswith("_pt") or name.endswith("_E")
        specs.append((name, label, log_scale))
    return specs


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
        theta_qi = to_numpy(events["theta_cm"]) > 0.6
        mass_qi = (
            to_numpy(events["mtautau"]) > 80.0
            if "mtautau" in events.fields
            else np.ones(len(events), dtype=bool)
        )
        metrics["qi_acceptance"], metrics["qi_acceptance_unc"] = weighted_fraction_and_unc(
            flags_valid & theta_qi & mass_qi,
            weights,
        )

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


def is_background_like_region(region: str) -> bool:
    lowered = region.lower()
    return lowered in {"other", "others", "background"} or lowered.endswith("_others")


def summary_region_order(regions: list[str]) -> list[str]:
    unique_regions = list(dict.fromkeys(regions))
    top_regions = [region for region in unique_regions if is_background_like_region(region)]
    signal_regions = [region for region in unique_regions if region not in set(top_regions)]
    return top_regions + signal_regions


def is_background_like_channel(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"zll", "zqq", "other", "others", "background", "unpredicted"} or lowered.endswith("_others")


def predicted_channel_order(names: list[str]) -> list[str]:
    unique_names = list(dict.fromkeys(names))

    def priority(name: str) -> int:
        lowered = name.lower()
        if lowered in {"ztautau_others", "tautau_others"} or lowered.endswith("_others"):
            return 0
        if is_background_like_channel(name):
            return 1
        return 2

    return [
        name
        for _, name in sorted(
            enumerate(unique_names),
            key=lambda indexed_name: (priority(indexed_name[1]), indexed_name[0]),
        )
    ]


def channel_label(name: str) -> str:
    if name.startswith("Ztautau_"):
        return name.removeprefix("Ztautau_")
    return name


def cms_label(ax, text: str = "Work in progress") -> None:
    ax.text(0.0, 1.02, "CMS", transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")
    ax.text(0.12, 1.02, text, transform=ax.transAxes, fontsize=10, style="italic", ha="left", va="bottom")


METRIC_PLOT_SPECS = [
    ("valid_fraction", "Valid fraction", "Fraction", "{:.3f}", "fraction"),
    ("qi_acceptance", "QI acceptance", "Fraction", "{:.3f}", "fraction"),
    ("nu_a_E_mae", r"$\nu_a$ E MAE", "MAE [GeV]", "{:.3f}", "absolute"),
    ("nu_b_E_mae", r"$\nu_b$ E MAE", "MAE [GeV]", "{:.3f}", "absolute"),
    ("nu_a_pt_mae", r"$\nu_a$ pT MAE", "MAE [GeV]", "{:.3f}", "absolute"),
    ("nu_b_pt_mae", r"$\nu_b$ pT MAE", "MAE [GeV]", "{:.3f}", "absolute"),
]


def sanitize_filename(name: str) -> str:
    output = []
    for char in name:
        output.append(char if char.isalnum() else "_")
    return "_".join("".join(output).strip("_").lower().split("_"))


def metric_precision(value: float, uncertainty: float, precision_mode: str) -> float:
    if not np.isfinite(uncertainty):
        return np.nan
    if precision_mode == "fraction":
        return uncertainty * 100.0
    if not np.isfinite(value) or value == 0:
        return np.nan
    return abs(uncertainty / value) * 100.0


def finite_metric_values(
    metrics: dict[str, dict[str, dict[str, Any]]],
    method_names: list[str],
    regions: list[str],
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    errors = []
    for method in method_names:
        for region in regions:
            region_metrics = metrics.get(method, {}).get(region, {})
            values.append(region_metrics.get(metric_key, np.nan))
            errors.append(region_metrics.get(f"{metric_key}_unc", np.nan))
    return np.asarray(values, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def plot_single_metric_summary(
    metrics: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    regions: list[str],
    method_names: list[str],
    metric_key: str,
    title: str,
    xlabel: str,
    uncertainty_format: str,
    precision_mode: str,
) -> None:
    values, errors = finite_metric_values(metrics, method_names, regions, metric_key)
    if not np.any(np.isfinite(values)):
        return

    fig_height = max(3.4, 0.62 * len(regions) + 1.2)
    fig, ax = plt.subplots(figsize=(9.4, fig_height), dpi=180)
    y_positions = np.arange(len(regions), dtype=np.float64)
    offsets = np.linspace(-0.22, 0.22, len(method_names)) if len(method_names) > 1 else np.array([0.0])

    finite_values = values[np.isfinite(values)]
    finite_errors = errors[np.isfinite(errors)]
    if finite_values.size:
        span = np.nanmax(finite_values) - np.nanmin(finite_values)
        pad = max(0.12 * span, 1.0e-3)
        if finite_errors.size:
            pad += np.nanmax(finite_errors)
        xmin = np.nanmin(finite_values) - pad
        xmax = np.nanmax(finite_values) + pad
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        ax.set_xlim(xmin, xmax)

    x_text_unc = 1.035
    x_text_prec = 1.17
    for method_index, method in enumerate(method_names):
        color = method_color(method, method_index)
        label = method_display_name(method)
        method_values = []
        method_errors = []
        method_y = []
        for region_index, region in enumerate(regions):
            region_metrics = metrics.get(method, {}).get(region, {})
            value = float(region_metrics.get(metric_key, np.nan))
            uncertainty = float(region_metrics.get(f"{metric_key}_unc", np.nan))
            y = y_positions[region_index] + offsets[method_index]
            method_values.append(value)
            method_errors.append(uncertainty)
            method_y.append(y)
            if np.isfinite(value):
                uncertainty_text = uncertainty_format.format(uncertainty) if np.isfinite(uncertainty) else "n/a"
                precision = metric_precision(value, uncertainty, precision_mode)
                precision_text = f"{precision:.2f}%" if np.isfinite(precision) else "n/a"
                ax.text(
                    x_text_unc,
                    y,
                    uncertainty_text,
                    color=color,
                    fontsize=7.2,
                    va="center",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
                ax.text(
                    x_text_prec,
                    y,
                    precision_text,
                    color=color,
                    fontsize=7.2,
                    va="center",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )

        method_values = np.asarray(method_values, dtype=np.float64)
        method_errors = np.asarray(method_errors, dtype=np.float64)
        method_y = np.asarray(method_y, dtype=np.float64)
        plot_mask = np.isfinite(method_values)
        plot_errors = np.where(np.isfinite(method_errors[plot_mask]), method_errors[plot_mask], 0.0)
        ax.errorbar(
            method_values[plot_mask],
            method_y[plot_mask],
            xerr=plot_errors,
            fmt="o",
            color=color,
            label=label,
            capsize=2.5,
            markersize=5.0,
            lw=1.2,
        )

    ax.text(x_text_unc, 1.02, "Unc.", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    ax.text(x_text_prec, 1.02, "Precision", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(regions)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.18, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Channel")
    ax.set_title(title)
    cms_label(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17), ncols=min(len(method_names), 4))
    fig.subplots_adjust(right=0.76, top=0.82, left=0.16, bottom=0.16)
    fig.savefig(output_dir / f"qi_metric_{sanitize_filename(metric_key)}.png")
    plt.close(fig)


def plot_metric_summary(
    metrics: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    regions: list[str],
    method_names: list[str],
) -> None:
    specs = list(METRIC_PLOT_SPECS)
    for observable in get_observable_names():
        specs.append(
            (
                f"{observable}_mean",
                observable.replace("_", " "),
                "Weighted mean",
                "{:.3f}",
                "absolute",
            )
        )

    for metric_key, title, xlabel, uncertainty_format, precision_mode in specs:
        plot_single_metric_summary(
            metrics=metrics,
            output_dir=output_dir,
            regions=regions,
            method_names=method_names,
            metric_key=metric_key,
            title=title,
            xlabel=xlabel,
            uncertainty_format=uncertainty_format,
            precision_mode=precision_mode,
        )


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
            for method_index, (method, events) in enumerate(events_by_method.items()):
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
                    color=method_color(method, method_index),
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
    pred_channels = np.array([str(name) for name in ak.to_list(events["evenet_pred_class_name"])], dtype=object)
    labels = predicted_channel_order(pred_channels.tolist())
    if not labels:
        return
    matrix = np.zeros((len(regions), len(labels)), dtype=np.float64)
    weights = event_weights(events)
    for i, region in enumerate(regions):
        cut_field = f"{region}_cut"
        if cut_field not in events.fields:
            continue
        cut_mask = to_numpy(events[cut_field], bool)
        for j, pred_channel in enumerate(labels):
            matrix[i, j] = np.sum(weights[cut_mask & (pred_channels == pred_channel)])

    fig_width = max(8.5, 0.78 * len(labels) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), dpi=180)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([channel_label(label) for label in labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    ax.set_xlabel("EveNet predicted fine channel")
    ax.set_ylabel("Central cut-based region")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    cms_label(ax)
    fig.colorbar(image, ax=ax, label="Weighted yield")
    fig.tight_layout()
    fig.savefig(output_dir / "cut_based_vs_evenet_region_matrix.png")
    plt.close(fig)


def load_optional_region(root: Path, sample_name: str, region: str) -> ak.Array | None:
    path = parquet_for(root, sample_name, region)
    if not path.exists():
        return None
    return load_events(path)


def metric_groups_from_regions(
    methods: dict[str, Path],
    sample_name: str,
    regions: list[str],
    output_dir: Path,
    max_scatter_points: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    method_names = list(methods)
    metrics: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in method_names}
    for region in regions:
        region_events: dict[str, ak.Array] = {}
        for method, root in methods.items():
            path = parquet_for(root, sample_name, region)
            if not path.exists():
                continue
            events = load_events(path)
            region_events[method] = events
            metrics[method][region] = method_region_metrics(events)

        if len(region_events) >= 2:
            plot_neutrino_truth_scatter(region_events, output_dir, region, max_scatter_points)
    active_regions = [region for region in regions if any(region in metrics.get(method, {}) for method in method_names)]
    return metrics, active_regions


def metric_groups_from_evenet_channels(
    methods: dict[str, Path],
    sample_name: str,
    output_dir: Path,
    max_scatter_points: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
    method_names = list(methods)
    metrics: dict[str, dict[str, dict[str, Any]]] = {method: {} for method in method_names}
    raw_events_by_method: dict[str, ak.Array] = {}
    all_channels: list[str] = []

    for method, root in methods.items():
        raw_path = parquet_for(root, sample_name, "raw")
        if not raw_path.exists():
            continue
        events = load_events(raw_path)
        if "evenet_pred_class_name" not in events.fields:
            continue
        raw_events_by_method[method] = events
        all_channels.extend(str(name) for name in ak.to_list(events["evenet_pred_class_name"]))

    channels = predicted_channel_order(all_channels)
    for channel in channels:
        channel_events: dict[str, ak.Array] = {}
        for method, events in raw_events_by_method.items():
            pred_channels = np.array([str(name) for name in ak.to_list(events["evenet_pred_class_name"])], dtype=object)
            subset = events[pred_channels == channel]
            if len(subset) == 0:
                continue
            channel_events[method] = subset
            metrics[method][channel] = method_region_metrics(subset)

        if len(channel_events) >= 2:
            plot_neutrino_truth_scatter(channel_events, output_dir, sanitize_filename(channel), max_scatter_points)

    active_channels = [channel for channel in channels if any(channel in metrics.get(method, {}) for method in method_names)]
    return metrics, active_channels


def weighted_hist_and_err2(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values) & np.isfinite(weights)
    hist = np.histogram(values[mask], bins=bins, weights=weights[mask])[0].astype(np.float64)
    err2 = np.histogram(values[mask], bins=bins, weights=np.square(weights[mask]))[0].astype(np.float64)
    return hist, err2


def count_hist(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    return np.histogram(values, bins=bins)[0].astype(np.float64)


def plot_physics_data_mc_comparisons(
    methods: dict[str, Path],
    output_dir: Path,
    regions: list[str],
    data_sample_name: str,
    mc_sample_names: list[str],
    requested_observables: list[str] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    specs = physics_observable_specs(requested_observables)

    for method_index, (method, root) in enumerate(methods.items()):
        method_dir = output_dir / f"physics_data_mc_{sanitize_filename(method)}"
        method_dir.mkdir(parents=True, exist_ok=True)
        method_summary: dict[str, Any] = {}

        for region in regions:
            data_events = load_optional_region(root, data_sample_name, region)
            mc_events_by_sample = {
                sample_name: events
                for sample_name in mc_sample_names
                if (events := load_optional_region(root, sample_name, region)) is not None
            }
            if data_events is None and not mc_events_by_sample:
                continue

            region_summary: dict[str, Any] = {
                "data_sample": data_sample_name,
                "data_events": int(len(data_events)) if data_events is not None else 0,
                "mc_samples": {name: int(len(events)) for name, events in mc_events_by_sample.items()},
                "plots": [],
            }

            for observable, x_label, log_scale in specs:
                values_by_sample: dict[str, np.ndarray] = {}
                if data_events is not None:
                    values_by_sample[data_sample_name] = extract_physics_observable(data_events, observable)
                for sample_name, events in mc_events_by_sample.items():
                    values_by_sample[sample_name] = extract_physics_observable(events, observable)

                if not any(np.asarray(values).size > 0 and np.any(np.isfinite(values)) for values in values_by_sample.values()):
                    continue

                bins = choose_bins(values_by_sample, num_bins=50)
                data_hist = None
                if data_events is not None:
                    data_hist = count_hist(values_by_sample[data_sample_name], bins)

                hist_mc: dict[str, np.ndarray] = {}
                hist_mc_err2: dict[str, np.ndarray] = {}
                for sample_name, events in mc_events_by_sample.items():
                    values = values_by_sample[sample_name]
                    weights = event_weights(events)
                    if observable.startswith("reco_tau_") and observable not in {
                        "reco_tau_pair_mass",
                        "reco_tau_pair_pt",
                        "reco_tau_pair_E",
                        "reco_tau_delta_eta",
                        "reco_tau_delta_phi",
                    }:
                        weights = np.concatenate([weights, weights])
                    if observable.startswith("visible_tau_") and observable not in {
                        "visible_tau_pair_mass",
                        "visible_tau_pair_pt",
                    }:
                        weights = np.concatenate([weights, weights])
                    hist_mc[sample_name], hist_mc_err2[sample_name] = weighted_hist_and_err2(values, weights, bins)

                output_path = method_dir / f"{region}_{sanitize_filename(observable)}.png"
                plot_from_histograms(
                    hist_data=data_hist,
                    hist_mc=hist_mc,
                    hist_mc_err2=hist_mc_err2,
                    bin_edges=bins,
                    x_label=x_label,
                    title=f"{method_display_name(method)} {region}: {observable}",
                    output_path=output_path,
                    normalize=False,
                    log_scale=False,
                    invalid_summary=summarize_invalid_hist_values(values_by_sample),
                )
                region_summary["plots"].append(str(output_path.relative_to(output_dir)))

            method_summary[region] = region_summary

        summary[method] = method_summary
    return summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def build_audit_summary(
    methods: dict[str, Path],
    metrics: dict[str, dict[str, dict[str, Any]]],
    regions: list[str],
    sample_name: str,
    output_dir: Path,
    raw_matrix_available: bool,
    physics_data_mc_summary: dict[str, Any],
    metric_grouping: str,
) -> dict[str, Any]:
    generated_plots = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.png"))
    method_summary: dict[str, Any] = {}
    for method, root in methods.items():
        channel_summary: dict[str, Any] = {}
        for region in regions:
            parquet_path = parquet_for(root, sample_name, "raw" if metric_grouping == "evenet-channel" else region)
            region_metrics = metrics.get(method, {}).get(region, {})
            channel_summary[region] = {
                "parquet": str(parquet_path),
                "parquet_exists": parquet_path.exists(),
                "num_events": region_metrics.get("num_events"),
                "weighted_yield": region_metrics.get("weighted_yield"),
                "valid_fraction": region_metrics.get("valid_fraction"),
                "valid_fraction_unc": region_metrics.get("valid_fraction_unc"),
                "qi_acceptance": region_metrics.get("qi_acceptance"),
                "qi_acceptance_unc": region_metrics.get("qi_acceptance_unc"),
                "available_metric_keys": sorted(region_metrics.keys()),
            }
        method_summary[method] = {
            "root": str(root),
            "display_name": method_display_name(method),
            "channels": channel_summary,
        }

    return {
        "methods": method_summary,
        "regions": regions,
        "metric_grouping": metric_grouping,
        "sample_name": sample_name,
        "generated_plots": generated_plots,
        "physics_data_mc": physics_data_mc_summary,
        "diagnostics": {
            "per_observable_metric_plots": [name for name in generated_plots if name.startswith("qi_metric_")],
            "neutrino_truth_scatter_plots": [name for name in generated_plots if name.startswith("neutrino_truth_vs_pred_")],
            "physics_data_mc_plots": [
                name for name in generated_plots if name.startswith("physics_data_mc_")
            ],
            "cut_based_vs_evenet_region_matrix": raw_matrix_available,
            "metrics_json": "qi_method_comparison_metrics.json",
        },
    }


def format_optional_float(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value_float):
        return "n/a"
    return f"{value_float:.{precision}f}"


def write_audit_report(audit: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# QI Method Comparison Audit",
        "",
        "This report is intended to make the comparison traceable before looking at final summary plots.",
        "",
        "## Inputs",
        "",
        "| Method | Input root |",
        "|---|---|",
    ]
    for method, method_info in audit["methods"].items():
        lines.append(f"| {method} | `{method_info['root']}` |")

    lines.extend(
        [
            "",
            "## Channel Coverage",
            "",
            "| Method | Channel | Parquet | Events | Weighted yield | Valid fraction | QI acceptance |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for method, method_info in audit["methods"].items():
        for region, region_info in method_info["channels"].items():
            exists = "yes" if region_info["parquet_exists"] else "no"
            lines.append(
                "| "
                f"{method} | {region} | {exists} | "
                f"{region_info['num_events'] if region_info['num_events'] is not None else 'n/a'} | "
                f"{format_optional_float(region_info['weighted_yield'])} | "
                f"{format_optional_float(region_info['valid_fraction'])} | "
                f"{format_optional_float(region_info['qi_acceptance'])} |"
            )

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Metrics JSON: `{audit['diagnostics']['metrics_json']}`",
            f"- Per-observable metric plots: {len(audit['diagnostics']['per_observable_metric_plots'])}",
            f"- Neutrino truth scatter plots: {len(audit['diagnostics']['neutrino_truth_scatter_plots'])}",
            f"- Physics data-vs-MC plots: {len(audit['diagnostics']['physics_data_mc_plots'])}",
            f"- Cut-based vs EveNet region matrix: {'available' if audit['diagnostics']['cut_based_vs_evenet_region_matrix'] else 'not available'}",
            "",
            "## Generated Plots",
            "",
        ]
    )
    for plot_name in audit["generated_plots"]:
        lines.append(f"- `{plot_name}`")

    output_dir.joinpath("qi_method_comparison_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_method_specs(args)
    method_names = list(methods)
    candidate_regions = summary_region_order(["other", *args.regions])

    raw_for_matrix = None
    for method, root in methods.items():
        raw_path = parquet_for(root, args.sample_name, "raw")
        if raw_path.exists():
            raw_events = load_events(raw_path)
            if "evenet_pred_class_name" in raw_events.fields:
                raw_for_matrix = raw_events
                break

    if args.metric_grouping == "evenet-channel":
        metrics, active_regions = metric_groups_from_evenet_channels(
            methods=methods,
            sample_name=args.sample_name,
            output_dir=args.output_dir,
            max_scatter_points=args.max_scatter_points,
        )
        physics_regions = candidate_regions
    else:
        metrics, active_regions = metric_groups_from_regions(
            methods=methods,
            sample_name=args.sample_name,
            regions=candidate_regions,
            output_dir=args.output_dir,
            max_scatter_points=args.max_scatter_points,
        )
        physics_regions = active_regions

    plot_metric_summary(metrics, args.output_dir, active_regions, method_names)
    if raw_for_matrix is not None:
        cut_based_regions = [region for region in args.regions if not is_background_like_region(region)]
        if cut_based_regions:
            plot_cut_based_vs_evenet(raw_for_matrix, args.output_dir, cut_based_regions)
    physics_data_mc_summary = plot_physics_data_mc_comparisons(
        methods=methods,
        output_dir=args.output_dir,
        regions=physics_regions,
        data_sample_name=args.data_sample_name,
        mc_sample_names=list(args.mc_sample_names),
        requested_observables=args.physics_observables,
    )

    audit = build_audit_summary(
        methods=methods,
        metrics=metrics,
        regions=active_regions,
        sample_name=args.sample_name,
        output_dir=args.output_dir,
        raw_matrix_available=raw_for_matrix is not None,
        physics_data_mc_summary=physics_data_mc_summary,
        metric_grouping=args.metric_grouping,
    )
    with (args.output_dir / "qi_method_comparison_audit.json").open("w") as handle:
        json.dump(json_safe(audit), handle, indent=2, sort_keys=True)
    write_audit_report(json_safe(audit), args.output_dir)

    with (args.output_dir / "qi_method_comparison_metrics.json").open("w") as handle:
        json.dump(json_safe(metrics), handle, indent=2, sort_keys=True)

    print(f"[plot-qi-method-comparison] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
