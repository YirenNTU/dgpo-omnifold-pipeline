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
import yaml
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_PIPELINE_ROOT = REPO_ROOT / "ml_pipeline"
EVENET_ROOT = ML_PIPELINE_ROOT / "EveNet-Full"
if str(EVENET_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENET_ROOT))

from build_evenet_input_from_parquet import merge_evenet_config, parse_config, read_yaml
from ml_pipeline_config import parse_evenet_config
from evenet.dataset.preprocess import unflatten_dict


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "prediction_summary"
DEFAULT_CLASS_NAME = "unselected"
MAX_NEUTRINO_SCATTER_POINTS = 50000
OKABE_ITO = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
STACK_COLORS = OKABE_ITO[:-1]
MC_COLOR = OKABE_ITO[0]
ACCENT_COLOR = OKABE_ITO[1]
TRUTH_COLOR = OKABE_ITO[3]
DATA_COLOR = OKABE_ITO[-1]
SCATTER_COLOR = OKABE_ITO[0]
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "okabe_heat",
    ["#FAFAFA", "#D9EEF7", "#56B4E9", "#0072B2"],
)


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
        "--mc-source-converted",
        nargs="*",
        default=None,
        help="Optional original converted MC parquet file(s) used to reconstruct tau/Z kinematics.",
    )
    parser.add_argument(
        "--data-source-converted",
        nargs="*",
        default=None,
        help="Optional original converted data parquet file(s) used to reconstruct tau/Z kinematics.",
    )
    parser.add_argument(
        "--source-shape-metadata",
        type=Path,
        default=None,
        help="Optional shape_metadata.json for source converted parquet inputs.",
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


def resolve_shape_metadata_path(paths: list[str], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if not paths:
        raise ValueError("No source converted parquet paths were provided.")
    candidate = Path(paths[0]).expanduser().resolve().parent / "shape_metadata.json"
    if not candidate.exists():
        raise FileNotFoundError(
            f"shape_metadata.json not found next to {paths[0]}. Pass --source-shape-metadata explicitly."
        )
    return candidate


def load_source_converted_batch(paths: list[str], shape_metadata_path: Path) -> dict[str, np.ndarray]:
    with shape_metadata_path.open() as handle:
        shape_metadata = json.load(handle)

    reconstructed_batches: list[dict[str, np.ndarray]] = []
    for path in paths:
        table = pq.read_table(path)
        flat_batch = {key: np.asarray(value) for key, value in table.to_pydict().items()}
        reconstructed_batches.append(unflatten_dict(flat_batch, shape_metadata, drop_column_prefix=None))

    merged: dict[str, np.ndarray] = {}
    for key in reconstructed_batches[0]:
        merged[key] = np.concatenate([batch[key] for batch in reconstructed_batches], axis=0)
    return merged


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


def _plot_single_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str,
    colorbar_label: str,
) -> None:
    num_classes = len(class_names)
    fig_size = max(9.0, min(16.0, 0.75 * num_classes))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=220)
    image = ax.imshow(matrix, aspect="equal", cmap=HEATMAP_CMAP)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Truth")
    ax.set_title(title)

    if np.all(np.isfinite(matrix)):
        max_value = float(np.max(matrix)) if matrix.size > 0 else 0.0
        normalized_like = max_value <= 1.05
        text_threshold = 0.45 * max_value if max_value > 0 else 0.0
        font_size = 9 if num_classes <= 10 else 7 if num_classes <= 14 else 6
        for row in range(num_classes):
            for col in range(num_classes):
                value = float(matrix[row, col])
                if normalized_like:
                    label = f"{value:.2f}"
                else:
                    label = f"{value:.0f}" if value >= 100 else f"{value:.1f}"
                text_color = "white" if value >= text_threshold and max_value > 0 else "#1F2937"
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color=text_color,
                )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_summary(
    matrix: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    row_norm = row_normalize(matrix)
    col_norm = col_normalize(matrix)
    total_weight = float(matrix.sum())
    weighted_accuracy = float(np.trace(matrix) / total_weight) if total_weight > 0 else float("nan")

    recalls = np.diag(row_norm)
    precisions = np.diag(col_norm)
    balanced_accuracy = float(np.nanmean(recalls)) if recalls.size > 0 else float("nan")
    _plot_single_confusion_matrix(
        matrix=matrix,
        class_names=class_names,
        output_path=output_dir / "classification_confusion_weighted.png",
        title=f"Weighted Confusion Matrix\naccuracy={weighted_accuracy:.4f}",
        colorbar_label="Weighted yield",
    )
    _plot_single_confusion_matrix(
        matrix=row_norm,
        class_names=class_names,
        output_path=output_dir / "classification_confusion_row_normalized.png",
        title=f"Row-Normalized Confusion Matrix\nbalanced accuracy={balanced_accuracy:.4f}",
        colorbar_label="Row-normalized yield",
    )

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


def p4_from_energy_pt_eta_phi(values: np.ndarray) -> dict[str, np.ndarray]:
    energy = values[..., 0].astype(np.float64)
    pt = values[..., 1].astype(np.float64)
    eta = values[..., 2].astype(np.float64)
    phi = values[..., 3].astype(np.float64)
    return {
        "E": energy,
        "px": pt * np.cos(phi),
        "py": pt * np.sin(phi),
        "pz": pt * np.sinh(eta),
        "pt": pt,
        "eta": eta,
        "phi": phi,
    }


def build_full_tau_kinematics(
    pred_events: ak.Array,
    source_batch: dict[str, np.ndarray],
) -> dict[str, Any]:
    if "tau_vis_prong" not in source_batch:
        raise ValueError("Source converted parquet is missing tau_vis_prong.")
    event_index = to_numpy(pred_events["event_index"], np.int64)
    visible = source_batch["tau_vis_prong"][event_index]
    visible_mask = source_batch.get("tau_vis_prong_mask", np.ones(visible.shape[:2], dtype=bool))[event_index].astype(bool)
    visible_p4 = p4_from_energy_pt_eta_phi(visible)

    def invisible_leg(prefix: str, slot: int) -> dict[str, np.ndarray]:
        valid_name = f"{prefix}_slot{slot}_valid"
        valid = to_numpy(pred_events[valid_name], bool) if valid_name in pred_events.fields else np.ones(len(pred_events), dtype=bool)
        return {
            "valid": valid,
            "E": to_numpy(pred_events[f"{prefix}_slot{slot}_E"], np.float64),
            "px": to_numpy(pred_events[f"{prefix}_slot{slot}_px"], np.float64),
            "py": to_numpy(pred_events[f"{prefix}_slot{slot}_py"], np.float64),
            "pz": to_numpy(pred_events[f"{prefix}_slot{slot}_pz"], np.float64),
        }

    pred_invisible = [invisible_leg("pred_invisible", slot) for slot in range(2)]
    truth_invisible = [invisible_leg("target_invisible", slot) for slot in range(2)]

    def combine_tau(slot: int, invisible: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        px = visible_p4["px"][:, slot] + invisible["px"]
        py = visible_p4["py"][:, slot] + invisible["py"]
        pz = visible_p4["pz"][:, slot] + invisible["pz"]
        energy = visible_p4["E"][:, slot] + invisible["E"]
        pt = np.sqrt(px ** 2 + py ** 2)
        momentum = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        eta = np.arcsinh(np.divide(pz, np.maximum(pt, 1e-8)))
        phi = np.arctan2(py, px)
        mass2 = np.maximum(energy ** 2 - momentum ** 2, 0.0)
        return {
            "valid": visible_mask[:, slot] & invisible["valid"],
            "E": energy,
            "px": px,
            "py": py,
            "pz": pz,
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": np.sqrt(mass2),
        }

    pred_tau = [combine_tau(slot, pred_invisible[slot]) for slot in range(2)]
    truth_tau = [combine_tau(slot, truth_invisible[slot]) for slot in range(2)]

    def pair_metrics(tau_pair: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        total_px = tau_pair[0]["px"] + tau_pair[1]["px"]
        total_py = tau_pair[0]["py"] + tau_pair[1]["py"]
        total_pz = tau_pair[0]["pz"] + tau_pair[1]["pz"]
        total_E = tau_pair[0]["E"] + tau_pair[1]["E"]
        momentum2 = total_px ** 2 + total_py ** 2 + total_pz ** 2
        mass = np.sqrt(np.maximum(total_E ** 2 - momentum2, 0.0))
        delta_eta = np.abs(tau_pair[0]["eta"] - tau_pair[1]["eta"])
        delta_phi = np.abs(np.arctan2(np.sin(tau_pair[0]["phi"] - tau_pair[1]["phi"]), np.cos(tau_pair[0]["phi"] - tau_pair[1]["phi"])))
        valid = tau_pair[0]["valid"] & tau_pair[1]["valid"]
        return {"mass": mass, "delta_eta": delta_eta, "delta_phi": delta_phi, "valid": valid}

    return {
        "pred_tau": pred_tau,
        "truth_tau": truth_tau,
        "pred_pair": pair_metrics(pred_tau),
        "truth_pair": pair_metrics(truth_tau),
    }


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


def choose_scatter_indices(num_points: int, max_points: int = MAX_NEUTRINO_SCATTER_POINTS) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(7)
    return np.sort(rng.choice(num_points, size=max_points, replace=False))


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
    pred_name = np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
    truth_name = np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
    class_match = pred_name == truth_name
    legend_drawn = False

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

            match_valid = class_match[valid]
            scatter_index = choose_scatter_indices(len(truth_valid))
            match_sample = match_valid[scatter_index]

            if np.any(match_sample):
                ax.scatter(
                    truth_valid[scatter_index][match_sample],
                    pred_valid[scatter_index][match_sample],
                    s=6,
                    alpha=0.28,
                    color=SCATTER_COLOR,
                    edgecolors="none",
                    rasterized=True,
                    label="Correct class" if not legend_drawn else None,
                )
            if np.any(~match_sample):
                ax.scatter(
                    truth_valid[scatter_index][~match_sample],
                    pred_valid[scatter_index][~match_sample],
                    s=7,
                    alpha=0.38,
                    color=ACCENT_COLOR,
                    edgecolors="none",
                    rasterized=True,
                    label="Mis-ID" if not legend_drawn else None,
                )
            ax.plot([x_min, x_max], [x_min, x_max], color=TRUTH_COLOR, linestyle="--", linewidth=1.5)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ax.set_xlabel(f"Truth {component}")
            ax.set_ylabel(f"Pred {component}")
            if not legend_drawn:
                ax.legend(loc="upper left", frameon=False, fontsize=9)
                legend_drawn = True

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
    ax.bar(x - 0.2, mc_counts, width=0.4, label="Weighted MC", color=MC_COLOR, alpha=0.82)
    ax.bar(x + 0.2, data_counts, width=0.4, label="Data", color=DATA_COLOR, alpha=0.82)
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

    for truth_idx, truth_name in enumerate(truth_processes):
        values = stack_matrix[:, truth_idx]
        if not np.any(values > 0):
            continue
        ax.barh(
            y,
            values,
            left=left,
            height=0.75,
            color=STACK_COLORS[truth_idx % len(STACK_COLORS)],
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
            color=DATA_COLOR,
            ecolor=DATA_COLOR,
            elinewidth=1.4,
            capsize=3,
            markersize=5,
            label="Data",
        )

    row_sums = stack_matrix.sum(axis=1)
    visible_max = float(np.max(left)) if left.size > 0 else 0.0
    if data_counts is not None and data_unc is not None and data_counts.size > 0:
        visible_max = max(visible_max, float(np.max(data_counts + data_unc)))
    if visible_max <= 0:
        visible_max = 1.0
    text_x = visible_max * 1.02
    ax.set_xlim(0.0, visible_max * 1.22)

    for index, class_name in enumerate(class_names):
        total = float(row_sums[index])
        if total > 0 and class_name in truth_index:
            signal_purity = float(stack_matrix[index, truth_index[class_name]] / total)
            label = f"{signal_purity:.3f}"
        else:
            signal_purity = float("nan")
            label = "n/a"
        ax.text(
            text_x,
            y[index],
            label,
            va="center",
            ha="left",
            fontsize=9,
            color=DATA_COLOR,
            fontweight="semibold",
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
            "signal_purity": float(row[truth_index[class_name]] / total) if total > 0 and class_name in truth_index else float("nan"),
            "data_yield": float(data_counts[index]) if data_counts is not None else float("nan"),
            "data_stat_unc": float(data_unc[index]) if data_unc is not None else float("nan"),
        }
    return {
        "truth_processes": truth_processes,
        "per_predicted_channel": purity_summary,
    }


def choose_hist_bins(values: np.ndarray, num_bins: int = 50, symmetric: bool = False) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(-1.0, 1.0, 11)
    low = float(np.percentile(finite, 0.5))
    high = float(np.percentile(finite, 99.5))
    if symmetric:
        bound = max(abs(low), abs(high))
        if bound == 0:
            bound = 1.0
        low, high = -bound, bound
    if low == high:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, num_bins + 1)


def make_weighted_hist(values: np.ndarray, weights: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)


def make_count_hist(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.histogram(values, bins=bins)[0].astype(np.float64)
    return counts, np.sqrt(counts)


def plot_region_histogram_panel(
    ax,
    bins: np.ndarray,
    mc_stack: dict[str, np.ndarray],
    data_hist: np.ndarray | None,
    data_err: np.ndarray | None,
    truth_hist: np.ndarray | None,
    title: str,
    xlabel: str,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    bottom = np.zeros_like(centers, dtype=np.float64)

    for index, (process_name, hist) in enumerate(mc_stack.items()):
        if np.sum(hist) <= 0:
            continue
        ax.bar(
            centers,
            hist,
            width=widths,
            bottom=bottom,
            color=STACK_COLORS[index % len(STACK_COLORS)],
            edgecolor="white",
            linewidth=0.4,
            align="center",
            label=process_name,
        )
        bottom += hist

    if truth_hist is not None and np.sum(truth_hist) > 0:
        ax.step(bins[:-1], truth_hist, where="post", color=TRUTH_COLOR, linewidth=1.8, linestyle="--", label="MC truth")

    if data_hist is not None and data_err is not None:
        ax.errorbar(
            centers,
            data_hist,
            yerr=data_err,
            fmt="o",
            color=DATA_COLOR,
            markersize=3.5,
            linewidth=1.0,
            capsize=2,
            label="Data" if np.any(data_hist > 0) else None,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Yield")
    ax.grid(axis="y", linestyle=":", alpha=0.3)


def plot_region_kinematics(
    region_name: str,
    mc_events: ak.Array,
    mc_source_batch: dict[str, np.ndarray],
    data_events: ak.Array | None,
    data_source_batch: dict[str, np.ndarray] | None,
    class_names: list[str],
    output_path: Path,
) -> dict[str, Any]:
    region_mc = mc_events[np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object) == region_name]
    region_data = None
    if data_events is not None:
        region_data = data_events[np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object) == region_name]

    truth_names_mc = np.asarray(ak.to_list(region_mc["evenet_truth_class_name"]), dtype=object)
    base_mc_weights = to_numpy(region_mc["evenet_weight"], np.float64) if len(region_mc) > 0 else np.array([], dtype=np.float64)
    truth_processes = [name for name in class_names if np.any(truth_names_mc == name)]

    mc_kin = build_full_tau_kinematics(region_mc, mc_source_batch) if len(region_mc) > 0 else None
    data_kin = build_full_tau_kinematics(region_data, data_source_batch) if (region_data is not None and len(region_data) > 0 and data_source_batch is not None) else None

    panel_specs = [
        ("z_mass", "Z mass from reconstructed tau pair", r"$m_{\tau\tau}$"),
        ("delta_eta", r"$\Delta\eta(\tau,\tau)$", r"$|\Delta\eta|$"),
        ("delta_phi", r"$\Delta\phi(\tau,\tau)$", r"$|\Delta\phi|$"),
        ("tau_E", r"Tau energy", r"$E_\tau$"),
        ("tau_px", r"Tau px", r"$p_{x,\tau}$"),
        ("tau_py", r"Tau py", r"$p_{y,\tau}$"),
        ("tau_pz", r"Tau pz", r"$p_{z,\tau}$"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=220)
    axes = axes.flatten()
    summary: dict[str, Any] = {}

    for axis, (key, title, xlabel) in zip(axes, panel_specs):
        if mc_kin is None:
            axis.set_title(f"{title}\nno MC events")
            axis.axis("off")
            continue

        if key == "z_mass":
            mc_pred_values = mc_kin["pred_pair"]["mass"]
            mc_truth_values = mc_kin["truth_pair"]["mass"]
            mc_valid = mc_kin["pred_pair"]["valid"] & mc_kin["truth_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_pair"]["mass"] if data_kin is not None else None
            data_valid = data_kin["pred_pair"]["valid"] if data_kin is not None else None
            symmetric = False
        elif key == "delta_eta":
            mc_pred_values = mc_kin["pred_pair"]["delta_eta"]
            mc_truth_values = mc_kin["truth_pair"]["delta_eta"]
            mc_valid = mc_kin["pred_pair"]["valid"] & mc_kin["truth_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_pair"]["delta_eta"] if data_kin is not None else None
            data_valid = data_kin["pred_pair"]["valid"] if data_kin is not None else None
            symmetric = False
        elif key == "delta_phi":
            mc_pred_values = mc_kin["pred_pair"]["delta_phi"]
            mc_truth_values = mc_kin["truth_pair"]["delta_phi"]
            mc_valid = mc_kin["pred_pair"]["valid"] & mc_kin["truth_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_pair"]["delta_phi"] if data_kin is not None else None
            data_valid = data_kin["pred_pair"]["valid"] if data_kin is not None else None
            symmetric = False
        else:
            component = key.split("_", 1)[1]
            mc_pred_values = np.concatenate([mc_kin["pred_tau"][0][component], mc_kin["pred_tau"][1][component]])
            mc_truth_values = np.concatenate([mc_kin["truth_tau"][0][component], mc_kin["truth_tau"][1][component]])
            mc_valid = np.concatenate([
                mc_kin["pred_tau"][0]["valid"] & mc_kin["truth_tau"][0]["valid"],
                mc_kin["pred_tau"][1]["valid"] & mc_kin["truth_tau"][1]["valid"],
            ])
            if data_kin is not None:
                data_values = np.concatenate([data_kin["pred_tau"][0][component], data_kin["pred_tau"][1][component]])
                data_valid = np.concatenate([data_kin["pred_tau"][0]["valid"], data_kin["pred_tau"][1]["valid"]])
            else:
                data_values = None
                data_valid = None
            symmetric = component in {"px", "py", "pz"}
            mc_weights_for_hist = np.concatenate([base_mc_weights, base_mc_weights])
        mc_finite = mc_valid & finite_mask(mc_pred_values, mc_truth_values, mc_weights_for_hist)
        combined_for_bins = np.concatenate([
            mc_pred_values[mc_finite],
            mc_truth_values[mc_finite],
            data_values[data_valid] if data_values is not None and data_valid is not None and np.any(data_valid) else np.array([], dtype=np.float64),
        ])
        bins = choose_hist_bins(combined_for_bins, symmetric=symmetric)

        mc_stack = {}
        if key.startswith("tau_"):
            expanded_truth_names = np.concatenate([truth_names_mc, truth_names_mc])
            expanded_weights = mc_weights_for_hist
            for process_name in truth_processes:
                process_mask = (expanded_truth_names == process_name) & mc_finite
                mc_stack[process_name] = make_weighted_hist(mc_pred_values[process_mask], expanded_weights[process_mask], bins)
            truth_hist = make_weighted_hist(mc_truth_values[mc_finite], expanded_weights[mc_finite], bins)
        else:
            for process_name in truth_processes:
                process_mask = (truth_names_mc == process_name) & mc_finite
                mc_stack[process_name] = make_weighted_hist(mc_pred_values[process_mask], mc_weights_for_hist[process_mask], bins)
            truth_hist = make_weighted_hist(mc_truth_values[mc_finite], mc_weights_for_hist[mc_finite], bins)

        data_hist = data_err = None
        if data_values is not None and data_valid is not None:
            data_mask = data_valid & finite_mask(data_values)
            data_hist, data_err = make_count_hist(data_values[data_mask], bins)

        plot_region_histogram_panel(
            axis,
            bins=bins,
            mc_stack=mc_stack,
            data_hist=data_hist,
            data_err=data_err,
            truth_hist=truth_hist,
            title=title,
            xlabel=xlabel,
        )

        summary[key] = {
            "mc_reco_yield": float(sum(np.sum(hist) for hist in mc_stack.values())),
            "mc_truth_yield": float(np.sum(truth_hist)),
            "data_yield": float(np.sum(data_hist)) if data_hist is not None else float("nan"),
        }

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[-1].legend(handles, labels, loc="center")
        axes[-1].axis("off")
    fig.suptitle(f"Region {region_name}: data vs stacked MC (with MC truth reference)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def gather_region_kinematics_plots(
    mc_events: ak.Array,
    mc_source_batch: dict[str, np.ndarray] | None,
    data_events: ak.Array | None,
    data_source_batch: dict[str, np.ndarray] | None,
    class_names: list[str],
    output_dir: Path,
    max_processes: int | None,
) -> dict[str, Any]:
    if mc_source_batch is None:
        return {}

    predicted_names = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    region_names = [name for name in class_names if np.any(predicted_names == name)]
    if max_processes is not None:
        region_names = region_names[:max_processes]

    summary: dict[str, Any] = {}
    for region_name in region_names:
        summary[region_name] = plot_region_kinematics(
            region_name=region_name,
            mc_events=mc_events,
            mc_source_batch=mc_source_batch,
            data_events=data_events,
            data_source_batch=data_source_batch,
            class_names=class_names,
            output_path=output_dir / f"region_kinematics_{region_name}.png",
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_paths = expand_paths(args.mc_parquet)
    data_paths = expand_paths(args.data_parquet)
    mc_source_paths = expand_paths(args.mc_source_converted)
    data_source_paths = expand_paths(args.data_source_converted)
    mc_events = load_events(mc_paths)
    data_events = load_events(data_paths) if data_paths else None
    source_shape_metadata = resolve_shape_metadata_path(mc_source_paths or data_source_paths, args.source_shape_metadata) if (mc_source_paths or data_source_paths) else None
    mc_source_batch = load_source_converted_batch(mc_source_paths, source_shape_metadata) if mc_source_paths and source_shape_metadata is not None else None
    data_source_batch = load_source_converted_batch(data_source_paths, source_shape_metadata) if data_source_paths and source_shape_metadata is not None else None

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
        output_dir=output_dir,
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
    region_kinematics_metrics = gather_region_kinematics_plots(
        mc_events=mc_events,
        mc_source_batch=mc_source_batch,
        data_events=data_events,
        data_source_batch=data_source_batch,
        class_names=class_names,
        output_dir=output_dir,
        max_processes=args.max_processes,
    )

    metrics_payload = {
        "inputs": {
            "mc_parquet": [str(Path(path).resolve()) for path in mc_paths],
            "data_parquet": [str(Path(path).resolve()) for path in data_paths],
            "mc_source_converted": [str(Path(path).resolve()) for path in mc_source_paths],
            "data_source_converted": [str(Path(path).resolve()) for path in data_source_paths],
            "source_shape_metadata": str(source_shape_metadata.resolve()) if source_shape_metadata is not None else None,
            "analysis_config": str(args.analysis_config.resolve()),
            "evenet_config": str(args.evenet_config.resolve()),
        },
        "classification": confusion_metrics,
        "purity": purity_metrics,
        "neutrino": neutrino_metrics,
        "region_kinematics": region_kinematics_metrics,
    }
    with (output_dir / "summary_metrics.yaml").open("w") as handle:
        yaml.safe_dump(metrics_payload, handle, sort_keys=False)

    print(f"[summary] wrote plots and metrics to {output_dir}")


if __name__ == "__main__":
    main()
