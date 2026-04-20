#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

from build_evenet_input_from_parquet import merge_evenet_config, parse_config, read_yaml
from ml_pipeline_config import parse_evenet_config


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "prediction_summary"
DEFAULT_CLASS_NAME = "unselected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summary truth-metric plots for EveNet prediction parquet outputs."
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "analysis.yaml",
        help="analysis.yaml used to recover class ordering.",
    )
    parser.add_argument(
        "--evenet-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "evenet_schema.yaml",
        help="EveNet schema config used to reproduce class ordering with _others handling.",
    )
    parser.add_argument(
        "--mc-parquet",
        nargs="+",
        required=True,
        help="One or more MC prediction parquet files or glob patterns.",
    )
    parser.add_argument(
        "--data-parquet",
        nargs="*",
        default=None,
        help="Optional data prediction parquet files or glob patterns for a predicted-class comparison plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where summary plots and metrics are written.",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=None,
        help="Optional cap on the number of per-process neutrino figures.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str] | None) -> list[str]:
    if not patterns:
        return []
    output: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            output.extend(matches)
        else:
            output.append(pattern)
    return output


def load_events(paths: list[str]) -> ak.Array:
    arrays = [ak.from_parquet(path) for path in paths]
    if not arrays:
        raise ValueError("No parquet inputs found.")
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def build_class_names_from_analysis(analysis_config_path: Path, evenet_config_path: Path) -> list[str]:
    samples, subcategories, feature_config = parse_config(analysis_config_path)
    merged_evenet = parse_evenet_config(
        merge_evenet_config(read_yaml(evenet_config_path), read_yaml(analysis_config_path)),
        feature_config,
    )

    class_names: list[str] = []
    for sample_key, sample in samples.items():
        if sample.is_data:
            continue
        splits = subcategories.get(sample_key) or subcategories.get(sample.name)
        if splits:
            class_names.extend(split.name for split in splits)
            remainder_name = f"{sample.name}_others"
            if remainder_name in merged_evenet.process_topologies:
                class_names.append(remainder_name)
        else:
            class_names.append(sample.name)
    return class_names


def to_numpy(values: ak.Array, dtype=None) -> np.ndarray:
    output = ak.to_numpy(values, allow_missing=False)
    if dtype is not None:
        output = output.astype(dtype)
    return output


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0:
        return float("nan")
    return float(np.sum(weights * values) / weights_sum)


def weighted_covariance(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    mean_x = weighted_mean(x, weights)
    mean_y = weighted_mean(y, weights)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0:
        return float("nan")
    return float(np.sum(weights * (x - mean_x) * (y - mean_y)) / weights_sum)


def weighted_variance(x: np.ndarray, weights: np.ndarray) -> float:
    return weighted_covariance(x, x, weights)


def weighted_rmse(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0:
        return float("nan")
    mse = np.sum(weights * (pred - truth) ** 2) / weights_sum
    return float(np.sqrt(max(mse, 0.0)))


def weighted_r2(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    mean_truth = weighted_mean(truth, weights)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0:
        return float("nan")
    ss_res = np.sum(weights * (pred - truth) ** 2)
    ss_tot = np.sum(weights * (truth - mean_truth) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def weighted_pearson(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    cov = weighted_covariance(truth, pred, weights)
    var_truth = weighted_variance(truth, weights)
    var_pred = weighted_variance(pred, weights)
    denom = math.sqrt(max(var_truth, 0.0) * max(var_pred, 0.0))
    if denom <= 0:
        return float("nan")
    return float(cov / denom)


def weighted_ccc(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    cov = weighted_covariance(truth, pred, weights)
    mean_truth = weighted_mean(truth, weights)
    mean_pred = weighted_mean(pred, weights)
    var_truth = weighted_variance(truth, weights)
    var_pred = weighted_variance(pred, weights)
    denom = var_truth + var_pred + (mean_truth - mean_pred) ** 2
    if denom <= 0:
        return float("nan")
    return float((2.0 * cov) / denom)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def normalize_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))


def confusion_matrix_weighted(
    truth_idx: np.ndarray,
    pred_idx: np.ndarray,
    weights: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p, w in zip(truth_idx, pred_idx, weights):
        if 0 <= t < num_classes and 0 <= p < num_classes and w > 0:
            matrix[t, p] += w
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sum, out=np.zeros_like(matrix), where=row_sum > 0)


def col_normalize(matrix: np.ndarray) -> np.ndarray:
    col_sum = matrix.sum(axis=0, keepdims=True)
    return np.divide(matrix, col_sum, out=np.zeros_like(matrix), where=col_sum > 0)


def plot_confusion_summary(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> dict[str, Any]:
    row_norm = row_normalize(matrix)
    col_norm = col_normalize(matrix)
    total_weight = float(matrix.sum())
    weighted_accuracy = float(np.trace(matrix) / total_weight) if total_weight > 0 else float("nan")

    recalls = np.diag(row_norm)
    precisions = np.diag(col_norm)
    balanced_accuracy = float(np.nanmean(recalls)) if recalls.size > 0 else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(class_names) * 0.75), 8), dpi=220)
    panels = [
        (matrix, "Weighted Confusion Matrix", "{:.1f}"),
        (row_norm, "Row-Normalized Confusion Matrix", "{:.2f}"),
    ]
    for ax, (panel, title, fmt) in zip(axes, panels):
        image = ax.imshow(panel, aspect="auto", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")
        ax.set_title(title)
        if panel.size > 0:
            threshold = np.nanmax(panel) * 0.5 if np.isfinite(np.nanmax(panel)) else 0.0
            for i in range(panel.shape[0]):
                for j in range(panel.shape[1]):
                    value = panel[i, j]
                    color = "white" if value > threshold else "black"
                    ax.text(j, i, fmt.format(value), ha="center", va="center", color=color, fontsize=8)

    fig.suptitle(
        f"Event-Weighted Classification Summary\n"
        f"accuracy={weighted_accuracy:.4f}, balanced_accuracy={balanced_accuracy:.4f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    per_class = {
        class_name: {
            "weighted_recall": float(recalls[index]) if index < len(recalls) else float("nan"),
            "weighted_precision": float(precisions[index]) if index < len(precisions) else float("nan"),
            "truth_weight": float(matrix[index].sum()),
            "pred_weight": float(matrix[:, index].sum()),
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "weighted_accuracy": weighted_accuracy,
        "weighted_balanced_accuracy": balanced_accuracy,
        "total_weight": total_weight,
        "per_class": per_class,
    }


def extract_leg_components(events: ak.Array) -> dict[str, dict[str, np.ndarray]]:
    fields = set(events.fields)

    def values_for_prefix(prefix: str) -> dict[str, np.ndarray] | None:
        direct = {f"{prefix}_{name}" for name in ["E", "px", "py", "pz"]}
        if direct.issubset(fields):
            valid_field = f"{prefix}_valid"
            valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
            return {
                "valid": valid,
                "E": to_numpy(events[f"{prefix}_E"], np.float64),
                "px": to_numpy(events[f"{prefix}_px"], np.float64),
                "py": to_numpy(events[f"{prefix}_py"], np.float64),
                "pz": to_numpy(events[f"{prefix}_pz"], np.float64),
            }

        energy_field = f"{prefix}_energy"
        pt_field = f"{prefix}_pt"
        eta_field = f"{prefix}_eta"
        phi_field = f"{prefix}_phi"
        if {energy_field, pt_field, eta_field, phi_field}.issubset(fields):
            valid_field = f"{prefix}_valid"
            valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
            energy = to_numpy(events[energy_field], np.float64)
            pt = to_numpy(events[pt_field], np.float64)
            eta = to_numpy(events[eta_field], np.float64)
            phi = to_numpy(events[phi_field], np.float64)
            return {
                "valid": valid,
                "E": energy,
                "px": pt * np.cos(phi),
                "py": pt * np.sin(phi),
                "pz": pt * np.sinh(eta),
            }

        log_energy_field = f"{prefix}_log_energy"
        log_pt_field = f"{prefix}_log_pt"
        eta_field = f"{prefix}_eta"
        phi_field = f"{prefix}_phi"
        if {log_energy_field, log_pt_field, eta_field, phi_field}.issubset(fields):
            valid_field = f"{prefix}_valid"
            valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
            energy = np.expm1(to_numpy(events[log_energy_field], np.float64))
            pt = np.expm1(to_numpy(events[log_pt_field], np.float64))
            eta = to_numpy(events[eta_field], np.float64)
            phi = to_numpy(events[phi_field], np.float64)
            return {
                "valid": valid,
                "E": energy,
                "px": pt * np.cos(phi),
                "py": pt * np.sin(phi),
                "pz": pt * np.sinh(eta),
            }
        return None

    leg_map: dict[str, dict[str, np.ndarray]] = {}

    slot_prefixes = [
        ("slot0", "pred_invisible_slot0", "target_invisible_slot0"),
        ("slot1", "pred_invisible_slot1", "target_invisible_slot1"),
    ]
    for leg_name, pred_prefix, truth_prefix in slot_prefixes:
        pred = values_for_prefix(pred_prefix)
        truth = values_for_prefix(truth_prefix)
        if pred is not None and truth is not None:
            leg_map[leg_name] = {"pred": pred, "truth": truth}

    if leg_map:
        return leg_map

    paired_prefixes = [
        ("lead_a", "lead_a_missing", "truth_lead_a_missing"),
        ("lead_b", "lead_b_missing", "truth_lead_b_missing"),
    ]
    for leg_name, pred_prefix, truth_prefix in paired_prefixes:
        pred = values_for_prefix(pred_prefix)
        truth = values_for_prefix(truth_prefix)
        if pred is not None and truth is not None:
            leg_map[leg_name] = {"pred": pred, "truth": truth}

    return leg_map


def panel_limits(truth: np.ndarray, pred: np.ndarray, component: str) -> tuple[float, float]:
    merged = np.concatenate([truth, pred])
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return (-1.0, 1.0)

    low = float(np.percentile(merged, 0.5))
    high = float(np.percentile(merged, 99.5))
    if component in {"px", "py", "pz"}:
        bound = max(abs(low), abs(high))
        if bound == 0:
            bound = 1.0
        return (-bound, bound)
    if low == high:
        low -= 0.5
        high += 0.5
    return (low, high)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative.size == 0 or cumulative[-1] <= 0:
        return np.full_like(quantiles, np.nan, dtype=np.float64)
    normalized = cumulative / cumulative[-1]
    return np.interp(quantiles, normalized, values)


def make_neutrino_metrics(
    truth: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    rmse = weighted_rmse(truth, pred, weights)
    truth_std = math.sqrt(max(weighted_variance(truth, weights), 0.0))
    return {
        "weighted_ccc": weighted_ccc(truth, pred, weights),
        "weighted_r2": weighted_r2(truth, pred, weights),
        "weighted_pearson": weighted_pearson(truth, pred, weights),
        "weighted_rmse": rmse,
        "weighted_nrmse": float(rmse / truth_std) if truth_std > 0 else float("nan"),
        "weight_sum": float(np.sum(weights)),
    }


def plot_neutrino_grid(
    events: ak.Array,
    process_name: str,
    output_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    components = ["E", "px", "py", "pz"]
    leg_data = extract_leg_components(events)
    if not leg_data:
        raise ValueError("Prediction parquet does not contain recognizable truth/predicted neutrino fields.")
    leg_names = list(leg_data.keys())
    fig, axes = plt.subplots(len(leg_names), len(components), figsize=(4.5 * len(components), 4.0 * len(leg_names)), dpi=220)
    if len(leg_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    weights = to_numpy(events["evenet_weight"], np.float64)

    for row_index, leg_name in enumerate(leg_names):
        metrics[leg_name] = {}
        leg_pred = leg_data[leg_name]["pred"]
        leg_truth = leg_data[leg_name]["truth"]
        leg_valid = leg_pred["valid"] & leg_truth["valid"] & np.isfinite(weights) & (weights > 0)

        for col_index, component in enumerate(components):
            ax = axes[row_index, col_index]
            truth_values = leg_truth[component]
            pred_values = leg_pred[component]
            valid = leg_valid & finite_mask(truth_values, pred_values)
            if not np.any(valid):
                ax.set_title(f"{leg_name} {component}\nno valid events")
                ax.axis("off")
                metrics[leg_name][component] = {"weighted_ccc": float("nan")}
                continue

            truth_valid = truth_values[valid]
            pred_valid = pred_values[valid]
            weight_valid = weights[valid]
            x_min, x_max = panel_limits(truth_valid, pred_valid, component)

            hist = ax.hist2d(
                truth_valid,
                pred_valid,
                bins=80,
                range=[[x_min, x_max], [x_min, x_max]],
                weights=weight_valid,
                norm=LogNorm(),
                cmap="viridis",
            )
            fig.colorbar(hist[3], ax=ax, fraction=0.046, pad=0.04)
            ax.plot([x_min, x_max], [x_min, x_max], color="red", linestyle="--", linewidth=1.5)
            ax.set_xlabel(f"Truth {component}")
            ax.set_ylabel(f"Pred {component}")

            panel_metrics = make_neutrino_metrics(truth_valid, pred_valid, weight_valid)
            metrics[leg_name][component] = panel_metrics
            ax.set_title(
                f"{leg_name} {component}\n"
                f"CCC={panel_metrics['weighted_ccc']:.3f}, "
                f"R2={panel_metrics['weighted_r2']:.3f}, "
                f"nRMSE={panel_metrics['weighted_nrmse']:.3f}",
                fontsize=10,
            )

    fig.suptitle(f"Truth vs Predicted Neutrino Four-Momentum: {process_name}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def gather_neutrino_metrics(
    events: ak.Array,
    class_names: list[str],
    output_dir: Path,
    max_processes: int | None,
) -> dict[str, Any]:
    truth_name = np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
    valid_truth = truth_name != DEFAULT_CLASS_NAME
    process_names_present = [name for name in class_names if np.any(truth_name == name)]
    if max_processes is not None:
        process_names_present = process_names_present[:max_processes]

    summary: dict[str, Any] = {"all_processes": {}, "per_process": {}}

    all_events = events[valid_truth]
    summary["all_processes"] = plot_neutrino_grid(
        all_events,
        process_name="All Processes",
        output_path=output_dir / "neutrino_truth_vs_pred_all.png",
    )

    for process_name in process_names_present:
        process_events = events[truth_name == process_name]
        if len(process_events) == 0:
            continue
        summary["per_process"][process_name] = plot_neutrino_grid(
            process_events,
            process_name=process_name,
            output_path=output_dir / f"neutrino_truth_vs_pred_{process_name}.png",
        )

    ccc_values: list[float] = []
    for scope_metrics in [summary["all_processes"], *summary["per_process"].values()]:
        for leg_metrics in scope_metrics.values():
            for panel_metrics in leg_metrics.values():
                ccc = panel_metrics.get("weighted_ccc", float("nan"))
                if np.isfinite(ccc):
                    ccc_values.append(ccc)
    summary["consistency_score_ccc_mean"] = float(np.mean(ccc_values)) if ccc_values else float("nan")
    summary["consistency_score_ccc_mean_0to1"] = normalize_score(summary["consistency_score_ccc_mean"])
    return summary


def plot_predicted_class_comparison(
    mc_events: ak.Array,
    data_events: ak.Array,
    class_names: list[str],
    output_path: Path,
) -> None:
    mc_pred = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    mc_weight = to_numpy(mc_events["evenet_weight"], np.float64)
    data_pred = np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object)

    mc_counts = np.array([float(np.sum(mc_weight[mc_pred == name])) for name in class_names], dtype=np.float64)
    data_counts = np.array([int(np.sum(data_pred == name)) for name in class_names], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8), 6), dpi=220)
    x = np.arange(len(class_names))
    ax.bar(x - 0.2, mc_counts, width=0.4, label="Weighted MC", color="tab:blue", alpha=0.8)
    ax.bar(x + 0.2, data_counts, width=0.4, label="Data", color="black", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Events")
    ax.set_title("Predicted Class Distribution: Data vs Weighted MC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_channel_purity(
    mc_events: ak.Array,
    data_events: ak.Array | None,
    class_names: list[str],
    output_path: Path,
) -> dict[str, Any]:
    mc_pred = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    mc_truth = np.asarray(ak.to_list(mc_events["evenet_truth_class_name"]), dtype=object)
    mc_weight = to_numpy(mc_events["evenet_weight"], np.float64)

    valid_mc = np.isfinite(mc_weight) & (mc_weight > 0)
    mc_pred = mc_pred[valid_mc]
    mc_truth = mc_truth[valid_mc]
    mc_weight = mc_weight[valid_mc]

    truth_processes = [name for name in class_names if np.any(mc_truth == name)]
    stack_matrix = np.zeros((len(class_names), len(truth_processes)), dtype=np.float64)

    pred_index = {name: index for index, name in enumerate(class_names)}
    truth_index = {name: index for index, name in enumerate(truth_processes)}
    for pred_name, truth_name, weight in zip(mc_pred, mc_truth, mc_weight):
        if pred_name in pred_index and truth_name in truth_index:
            stack_matrix[pred_index[pred_name], truth_index[truth_name]] += weight

    data_counts = None
    data_unc = None
    if data_events is not None:
        data_pred = np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object)
        data_counts = np.array([float(np.sum(data_pred == name)) for name in class_names], dtype=np.float64)
        data_unc = np.sqrt(data_counts)

    fig_height = max(6, 0.45 * len(class_names) + 2.0)
    fig, ax = plt.subplots(figsize=(13, fig_height), dpi=220)
    y = np.arange(len(class_names))
    left = np.zeros(len(class_names), dtype=np.float64)
    cmap = plt.cm.get_cmap("tab20", max(len(truth_processes), 1))

    for truth_idx, truth_name in enumerate(truth_processes):
        values = stack_matrix[:, truth_idx]
        if not np.any(values > 0):
            continue
        ax.barh(
            y,
            values,
            left=left,
            height=0.75,
            color=cmap(truth_idx),
            edgecolor="white",
            linewidth=0.5,
            label=truth_name,
        )
        left += values

    if data_counts is not None and data_unc is not None:
        ax.errorbar(
            data_counts,
            y,
            xerr=data_unc,
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.4,
            capsize=3,
            markersize=5,
            label="Data",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Yield")
    ax.set_ylabel("Predicted channel")
    ax.set_title("Predicted Channel Purity: stacked truth-process yield with data overlay")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.invert_yaxis()
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    purity_summary = {}
    row_sums = stack_matrix.sum(axis=1)
    for index, class_name in enumerate(class_names):
        row = stack_matrix[index]
        total = float(row_sums[index])
        dominant_idx = int(np.argmax(row)) if row.size > 0 else -1
        purity_summary[class_name] = {
            "total_mc_yield": total,
            "dominant_truth_process": truth_processes[dominant_idx] if total > 0 and dominant_idx >= 0 else DEFAULT_CLASS_NAME,
            "dominant_fraction": float(row[dominant_idx] / total) if total > 0 and dominant_idx >= 0 else float("nan"),
            "data_yield": float(data_counts[index]) if data_counts is not None else float("nan"),
            "data_stat_unc": float(data_unc[index]) if data_unc is not None else float("nan"),
        }
    return {
        "truth_processes": truth_processes,
        "per_predicted_channel": purity_summary,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_paths = expand_paths(args.mc_parquet)
    data_paths = expand_paths(args.data_parquet)
    mc_events = load_events(mc_paths)
    data_events = load_events(data_paths) if data_paths else None

    class_names = build_class_names_from_analysis(args.analysis_config.resolve(), args.evenet_config.resolve())

    truth_idx = to_numpy(mc_events["evenet_truth_class_index"], np.int64)
    pred_idx = to_numpy(mc_events["evenet_pred_class_index"], np.int64)
    weights = to_numpy(mc_events["evenet_weight"], np.float64)
    valid_class = (truth_idx >= 0) & (truth_idx < len(class_names)) & (pred_idx >= 0) & (pred_idx < len(class_names)) & np.isfinite(weights) & (weights > 0)

    confusion_metrics = plot_confusion_summary(
        matrix=confusion_matrix_weighted(
            truth_idx[valid_class],
            pred_idx[valid_class],
            weights[valid_class],
            num_classes=len(class_names),
        ),
        class_names=class_names,
        output_path=output_dir / "classification_confusion_summary.png",
    )

    neutrino_metrics = gather_neutrino_metrics(
        events=mc_events[valid_class],
        class_names=class_names,
        output_dir=output_dir,
        max_processes=args.max_processes,
    )

    if data_events is not None:
        plot_predicted_class_comparison(
            mc_events=mc_events,
            data_events=data_events,
            class_names=class_names,
            output_path=output_dir / "predicted_class_data_vs_mc.png",
        )
    purity_metrics = plot_predicted_channel_purity(
        mc_events=mc_events,
        data_events=data_events,
        class_names=class_names,
        output_path=output_dir / "predicted_channel_purity.png",
    )

    metrics_payload = {
        "inputs": {
            "mc_parquet": [str(Path(path).resolve()) for path in mc_paths],
            "data_parquet": [str(Path(path).resolve()) for path in data_paths],
            "analysis_config": str(args.analysis_config.resolve()),
            "evenet_config": str(args.evenet_config.resolve()),
        },
        "classification": confusion_metrics,
        "purity": purity_metrics,
        "neutrino": neutrino_metrics,
    }
    with (output_dir / "summary_metrics.yaml").open("w") as handle:
        yaml.safe_dump(metrics_payload, handle, sort_keys=False)

    print(f"[summary] wrote plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
