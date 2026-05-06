#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_PIPELINE_ROOT = REPO_ROOT / "ml_pipeline"
EVENET_ROOT = ML_PIPELINE_ROOT / "EveNet-Full"
if str(EVENET_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENET_ROOT))

from build_evenet_input_from_parquet import expand_input_files, merge_evenet_config, parse_config, read_yaml
from ml_pipeline_config import parse_evenet_config
from parquet_plot_common import OKABE_ITO, infer_luminosity, process_color, process_latex_label


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots" / "prediction_summary"
DEFAULT_CLASS_NAME = "unselected"
DEFAULT_FLOAT = -99.0
MAX_NEUTRINO_SCATTER_POINTS = 50000
MC_COLOR = OKABE_ITO["blue"]
ACCENT_COLOR = OKABE_ITO["orange"]
TRUTH_COLOR = OKABE_ITO["vermillion"]
DATA_COLOR = OKABE_ITO["black"]
SCATTER_COLOR = OKABE_ITO["blue"]
BACKGROUND_COLOR = "#D8D8D8"
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
    parser.add_argument(
        "--unblind",
        action="store_true",
        help="If set, overlay data on data-vs-MC plots. Otherwise data inputs are ignored in summary figures.",
    )
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="If set, ignore evenet_weight and evaluate all MC events with unit weight.",
    )
    parser.add_argument(
        "--weight-source",
        choices=["auto", "evenet", "central", "event", "class", "unit"],
        default="evenet",
        help=(
            "MC weight source. 'auto' prefers prediction-parquet evenet_weight and falls back to "
            "central/event/class normalization, "
            "'evenet' uses prediction-parquet evenet_weight, "
            "'class' recomputes analysis-config class normalization, "
            "'event' uses event_weight, and 'unit' ignores weights."
        ),
    )
    return parser.parse_args()


def expand_paths(patterns: list[str] | None) -> list[str]:
    if not patterns:
        return []
    output: list[str] = []
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        if expanded.is_dir():
            final_prediction_paths = sorted(expanded.glob("*__evenet_pred.parquet"))
            paths = final_prediction_paths if final_prediction_paths else sorted(expanded.glob("*.parquet"))
            output.extend(str(path) for path in paths)
            continue

        matches = sorted(glob.glob(str(expanded)))
        if matches:
            output.extend(str(Path(match)) for match in matches)
        else:
            output.append(str(expanded))
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


def build_class_to_sample_map(samples: dict, subcategories: dict, evenet_config) -> dict[str, Any]:
    class_to_sample: dict[str, Any] = {}
    for sample_key, sample in samples.items():
        if sample.is_data:
            continue
        splits = subcategories.get(sample_key) or subcategories.get(sample.name)
        if splits:
            for split in splits:
                class_to_sample[split.name] = sample
            remainder_name = f"{sample.name}_others"
            if remainder_name in evenet_config.process_topologies:
                class_to_sample[remainder_name] = sample
        else:
            class_to_sample[sample.name] = sample
    return class_to_sample


def sample_initial_total_num_events(sample) -> int:
    parquet_files = expand_input_files(sample.input_files)
    values: list[int] = []
    for path in parquet_files:
        events = ak.from_parquet(path, columns=["initial_total_num_events"])
        if len(events) == 0 or "initial_total_num_events" not in events.fields:
            continue
        values.append(int(ak.to_numpy(events["initial_total_num_events"][:1], allow_missing=False)[0]))
    if values:
        return int(sum(values))
    return 0


def build_prediction_class_weights(
    analysis_config_path: Path,
    evenet_config_path: Path,
    class_names: list[str],
) -> dict[str, float]:
    samples, subcategories, feature_config = parse_config(analysis_config_path)
    merged_evenet = parse_evenet_config(
        merge_evenet_config(read_yaml(evenet_config_path), read_yaml(analysis_config_path)),
        feature_config,
    )
    luminosity = infer_luminosity(samples, None)
    class_to_sample = build_class_to_sample_map(samples, subcategories, merged_evenet)

    output: dict[str, float] = {}
    for class_name in class_names:
        sample = class_to_sample.get(class_name)
        if sample is None or sample.is_data or luminosity is None:
            output[class_name] = 1.0
            continue
        initial_total_num_events = sample_initial_total_num_events(sample)
        if initial_total_num_events <= 0:
            output[class_name] = 1.0
            continue
        output[class_name] = float(sample.norm_factor) / float(initial_total_num_events) * float(luminosity)
    return output


def latex_process_label(name: str) -> str:
    if name == DEFAULT_CLASS_NAME:
        return name
    return process_latex_label(name)


def latex_labels(names: list[str]) -> list[str]:
    return [latex_process_label(name) for name in names]


def is_background_like_channel(name: str) -> bool:
    lowered = name.lower()
    return (
        name == DEFAULT_CLASS_NAME
        or lowered in {"zll", "zqq"}
        or lowered.endswith("_others")
        or lowered == "others"
        or "background" in lowered
    )


def summary_channel_priority(name: str) -> int:
    lowered = name.lower()
    if lowered in {"ztautau_others", "tautau_others", "tau_tau_others"} or lowered.endswith("_others"):
        return 0
    if lowered in {"zll", "zqq"} or "background" in lowered:
        return 1
    if name == DEFAULT_CLASS_NAME:
        return 2
    return 3


def summary_channel_order(class_names: list[str]) -> list[str]:
    return [
        name
        for _, name in sorted(
            enumerate(class_names),
            key=lambda indexed_name: (summary_channel_priority(indexed_name[1]), indexed_name[0]),
        )
    ]


def signal_channel_order(class_names: list[str]) -> list[str]:
    return [name for name in class_names if not is_background_like_channel(name)]


def process_stack_color(process_name: str, index: int) -> str:
    if is_background_like_channel(process_name):
        return BACKGROUND_COLOR
    return process_color(process_name, index)


def stack_draw_order(process_names: list[str]) -> list[str]:
    signal_names = [name for name in process_names if not is_background_like_channel(name)]
    background_names = [name for name in process_names if is_background_like_channel(name)]
    return signal_names + background_names


def to_numpy(values: ak.Array, dtype=None) -> np.ndarray:
    output = ak.to_numpy(values, allow_missing=False)
    if dtype is not None:
        output = output.astype(dtype)
    return output


def class_weight_array(events: ak.Array, class_weight_map: dict[str, float] | None) -> np.ndarray | None:
    if not class_weight_map:
        return None

    class_names = None
    if "evenet_truth_class_name" in events.fields:
        class_names = np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
    elif "evenet_pred_class_name" in events.fields:
        class_names = np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
    if class_names is None:
        return None

    weights = np.ones(len(events), dtype=np.float64)
    for class_name, class_weight in class_weight_map.items():
        weights[class_names == class_name] = float(class_weight)
    return np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)


def event_weights(
    events: ak.Array,
    use_weighted: bool,
    weight_source: str = "auto",
    class_weight_map: dict[str, float] | None = None,
) -> np.ndarray:
    if not use_weighted or weight_source == "unit":
        return np.ones(len(events), dtype=np.float64)

    if weight_source == "class":
        class_weights = class_weight_array(events, class_weight_map)
        return class_weights if class_weights is not None else np.ones(len(events), dtype=np.float64)

    field_priority = {
        "auto": ("evenet_weight", "central_weight", "weight", "event_weight"),
        "evenet": ("evenet_weight", "central_weight", "weight", "event_weight"),
        "central": ("central_weight", "weight", "event_weight"),
        "event": ("event_weight", "central_weight", "weight"),
    }.get(weight_source, ("evenet_weight",))
    for field in field_priority:
        if field not in events.fields:
            continue
        weights = to_numpy(events[field], np.float64)
        valid = np.isfinite(weights) & (weights > 0)
        return np.where(valid, weights, 0.0)

    class_weights = class_weight_array(events, class_weight_map)
    if class_weights is not None:
        return class_weights
    return np.ones(len(events), dtype=np.float64)


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


def valid_physics_values(*arrays: np.ndarray) -> np.ndarray:
    mask = finite_mask(*arrays)
    for arr in arrays:
        mask &= ~np.isclose(arr, DEFAULT_FLOAT)
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
    ax.set_xticklabels(latex_labels(class_names), rotation=45, ha="right")
    ax.set_yticklabels(latex_labels(class_names))
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
    leg_map: dict[str, dict[str, np.ndarray]] = {}

    slot_prefixes = [
        ("slot0", "pred_invisible_slot0", "target_invisible_slot0"),
        ("slot1", "pred_invisible_slot1", "target_invisible_slot1"),
    ]
    for leg_name, pred_prefix, truth_prefix in slot_prefixes:
        pred = values_for_prefix(events, pred_prefix)
        truth = values_for_prefix(events, truth_prefix)
        if pred is not None and truth is not None:
            leg_map[leg_name] = {"pred": pred, "truth": truth}

    if leg_map:
        return leg_map

    paired_prefixes = [
        ("lead_a", "lead_a_missing", "truth_lead_a_missing"),
        ("lead_b", "lead_b_missing", "truth_lead_b_missing"),
    ]
    for leg_name, pred_prefix, truth_prefix in paired_prefixes:
        pred = values_for_prefix(events, pred_prefix)
        truth = values_for_prefix(events, truth_prefix)
        if pred is not None and truth is not None:
            leg_map[leg_name] = {"pred": pred, "truth": truth}

    return leg_map


def values_for_prefix(events: ak.Array, prefix: str) -> dict[str, np.ndarray] | None:
    fields = set(events.fields)
    direct = {f"{prefix}_{name}" for name in ["E", "px", "py", "pz"]}
    if direct.issubset(fields):
        valid_field = f"{prefix}_valid"
        valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
        energy = to_numpy(events[f"{prefix}_E"], np.float64)
        px = to_numpy(events[f"{prefix}_px"], np.float64)
        py = to_numpy(events[f"{prefix}_py"], np.float64)
        pz = to_numpy(events[f"{prefix}_pz"], np.float64)
        valid &= valid_physics_values(energy, px, py, pz)
        return {
            "valid": valid,
            "E": energy,
            "px": px,
            "py": py,
            "pz": pz,
        }

    energy_field = f"{prefix}_energy"
    pt_field = f"{prefix}_pt"
    eta_field = f"{prefix}_eta"
    phi_field = f"{prefix}_phi"

    def values_from_pt_eta_phi(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
        """
        inputs:
          pt/eta/phi: np.ndarray, invisible momentum features stored in prediction parquet.
          valid: np.ndarray[bool], slot validity mask.
        outputs:
          dict[str, np.ndarray], four-vector-like components for a massless neutrino.
        goal:
          Support EveNet models trained with only pt/eta/phi invisible features by
          reconstructing the missing energy from the massless-neutrino assumption.
        """
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = pt * np.cosh(eta)
        return {
            "valid": valid,
            "E": energy,
            "px": px,
            "py": py,
            "pz": pz,
        }

    if {energy_field, pt_field, eta_field, phi_field}.issubset(fields):
        valid_field = f"{prefix}_valid"
        valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
        energy = to_numpy(events[energy_field], np.float64)
        pt = to_numpy(events[pt_field], np.float64)
        eta = to_numpy(events[eta_field], np.float64)
        phi = to_numpy(events[phi_field], np.float64)
        valid &= valid_physics_values(energy, pt, eta, phi)
        return {
            "valid": valid,
            "E": energy,
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * np.sinh(eta),
        }
    if {pt_field, eta_field, phi_field}.issubset(fields):
        valid_field = f"{prefix}_valid"
        valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
        pt = to_numpy(events[pt_field], np.float64)
        eta = to_numpy(events[eta_field], np.float64)
        phi = to_numpy(events[phi_field], np.float64)
        valid &= valid_physics_values(pt, eta, phi)
        return values_from_pt_eta_phi(pt, eta, phi, valid)

    log_energy_field = f"{prefix}_log_energy"
    log_pt_field = f"{prefix}_log_pt"
    if {log_energy_field, log_pt_field, eta_field, phi_field}.issubset(fields):
        valid_field = f"{prefix}_valid"
        valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
        log_energy = to_numpy(events[log_energy_field], np.float64)
        log_pt = to_numpy(events[log_pt_field], np.float64)
        energy = np.expm1(log_energy)
        pt = np.expm1(log_pt)
        eta = to_numpy(events[eta_field], np.float64)
        phi = to_numpy(events[phi_field], np.float64)
        valid &= valid_physics_values(log_energy, log_pt, eta, phi)
        return {
            "valid": valid,
            "E": energy,
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * np.sinh(eta),
        }
    if {log_pt_field, eta_field, phi_field}.issubset(fields):
        valid_field = f"{prefix}_valid"
        valid = to_numpy(events[valid_field], bool) if valid_field in fields else np.ones(len(events), dtype=bool)
        log_pt = to_numpy(events[log_pt_field], np.float64)
        pt = np.expm1(log_pt)
        eta = to_numpy(events[eta_field], np.float64)
        phi = to_numpy(events[phi_field], np.float64)
        valid &= valid_physics_values(log_pt, eta, phi)
        return values_from_pt_eta_phi(pt, eta, phi, valid)
    return None


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


def extract_visible_tau_prong(events: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    fields = set(events.fields)
    num_events = len(events)
    values = np.full((num_events, 2, 4), np.nan, dtype=np.float64)
    valid = np.zeros((num_events, 2), dtype=bool)

    for slot in range(2):
        required_fields = {
            f"tau_vis_prong_slot{slot}_energy",
            f"tau_vis_prong_slot{slot}_pt",
            f"tau_vis_prong_slot{slot}_eta",
            f"tau_vis_prong_slot{slot}_phi",
        }
        if not required_fields.issubset(fields):
            raise ValueError(
                "Prediction parquet is missing stored visible tau information. "
                "Re-run predict_evenet_from_raw_parquet.py after updating it to store tau_vis_prong slots."
            )

        values[:, slot, 0] = to_numpy(events[f"tau_vis_prong_slot{slot}_energy"], np.float64)
        values[:, slot, 1] = to_numpy(events[f"tau_vis_prong_slot{slot}_pt"], np.float64)
        values[:, slot, 2] = to_numpy(events[f"tau_vis_prong_slot{slot}_eta"], np.float64)
        values[:, slot, 3] = to_numpy(events[f"tau_vis_prong_slot{slot}_phi"], np.float64)
        valid_field = f"tau_vis_prong_slot{slot}_valid"
        if valid_field in fields:
            valid[:, slot] = to_numpy(events[valid_field], bool)
        else:
            valid[:, slot] = finite_mask(values[:, slot, 0], values[:, slot, 1], values[:, slot, 2], values[:, slot, 3])

    return values, valid


def build_full_tau_kinematics(
    pred_events: ak.Array,
) -> dict[str, Any]:
    visible, visible_mask = extract_visible_tau_prong(pred_events)
    visible_p4 = p4_from_energy_pt_eta_phi(visible)
    visible_components = [
        {
            "E": visible_p4["E"][:, 0],
            "px": visible_p4["px"][:, 0],
            "py": visible_p4["py"][:, 0],
            "pz": visible_p4["pz"][:, 0],
            "pt": visible_p4["pt"][:, 0],
            "eta": visible_p4["eta"][:, 0],
            "phi": visible_p4["phi"][:, 0],
            "valid": visible_mask[:, 0],
        },
        {
            "E": visible_p4["E"][:, 1],
            "px": visible_p4["px"][:, 1],
            "py": visible_p4["py"][:, 1],
            "pz": visible_p4["pz"][:, 1],
            "pt": visible_p4["pt"][:, 1],
            "eta": visible_p4["eta"][:, 1],
            "phi": visible_p4["phi"][:, 1],
            "valid": visible_mask[:, 1],
        },
    ]

    def invisible_leg(prefix: str, slot: int) -> dict[str, np.ndarray]:
        values = values_for_prefix(pred_events, f"{prefix}_slot{slot}")
        if values is None:
            raise ValueError(
                f"Prediction parquet is missing recognizable neutrino fields for {prefix}_slot{slot}. "
                "Expected E/px/py/pz, energy/pt/eta/phi, or log_energy/log_pt/eta/phi."
            )
        observables = build_neutrino_observable_map(values)
        return {**observables, "valid": values["valid"]}

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

    def pair_metrics(component_pair: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        total_px = component_pair[0]["px"] + component_pair[1]["px"]
        total_py = component_pair[0]["py"] + component_pair[1]["py"]
        total_pz = component_pair[0]["pz"] + component_pair[1]["pz"]
        total_E = component_pair[0]["E"] + component_pair[1]["E"]
        momentum2 = total_px ** 2 + total_py ** 2 + total_pz ** 2
        mass = np.sqrt(np.maximum(total_E ** 2 - momentum2, 0.0))
        delta_eta = np.abs(component_pair[0]["eta"] - component_pair[1]["eta"])
        delta_phi = np.abs(
            np.arctan2(
                np.sin(component_pair[0]["phi"] - component_pair[1]["phi"]),
                np.cos(component_pair[0]["phi"] - component_pair[1]["phi"]),
            )
        )
        valid = component_pair[0]["valid"] & component_pair[1]["valid"]
        return {"mass": mass, "delta_eta": delta_eta, "delta_phi": delta_phi, "valid": valid}

    return {
        "visible_tau": visible_components,
        "visible_pair": pair_metrics(visible_components),
        "pred_invisible": pred_invisible,
        "truth_invisible": truth_invisible,
        "pred_invisible_pair": pair_metrics(pred_invisible),
        "truth_invisible_pair": pair_metrics(truth_invisible),
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


def build_neutrino_observable_map(leg_values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    px = leg_values["px"]
    py = leg_values["py"]
    pz = leg_values["pz"]
    energy = leg_values["E"]
    pt = np.sqrt(px ** 2 + py ** 2)
    eta = np.arcsinh(np.divide(pz, np.maximum(pt, 1e-8)))
    phi = np.arctan2(py, px)
    return {
        "E": energy,
        "energy": energy,
        "px": px,
        "py": py,
        "pz": pz,
        "pt": pt,
        "eta": eta,
        "phi": phi,
    }


def plot_neutrino_grid(
    events: ak.Array,
    process_name: str,
    output_path: Path,
    component_specs: list[tuple[str, str]],
    title_suffix: str,
    use_weighted: bool,
    weight_source: str,
    class_weight_map: dict[str, float] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    leg_data = extract_leg_components(events)
    if not leg_data:
        raise ValueError("Prediction parquet does not contain recognizable truth/predicted neutrino fields.")
    leg_names = list(leg_data.keys())
    num_columns = len(component_specs)
    fig, axes = plt.subplots(len(leg_names), num_columns, figsize=(4.5 * num_columns, 4.0 * len(leg_names)), dpi=220)
    if len(leg_names) == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_columns == 1:
        axes = np.expand_dims(axes, axis=1)

    metrics: dict[str, dict[str, dict[str, float]]] = {}
    weights = event_weights(
        events,
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )
    pred_name = np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
    truth_name = np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
    class_match = pred_name == truth_name
    legend_drawn = False

    for row_index, leg_name in enumerate(leg_names):
        metrics[leg_name] = {}
        leg_pred = leg_data[leg_name]["pred"]
        leg_truth = leg_data[leg_name]["truth"]
        pred_observables = build_neutrino_observable_map(leg_pred)
        truth_observables = build_neutrino_observable_map(leg_truth)
        leg_valid = leg_pred["valid"] & leg_truth["valid"] & np.isfinite(weights) & (weights > 0)

        for col_index, (component_key, component_label) in enumerate(component_specs):
            ax = axes[row_index, col_index]
            truth_values = truth_observables[component_key]
            pred_values = pred_observables[component_key]
            valid = leg_valid & finite_mask(truth_values, pred_values)
            if not np.any(valid):
                ax.set_title(f"{leg_name} {component_label}\nno valid events")
                ax.axis("off")
                metrics[leg_name][component_key] = {"weighted_ccc": float("nan")}
                continue

            truth_valid = truth_values[valid]
            pred_valid = pred_values[valid]
            weight_valid = weights[valid]
            x_min, x_max = panel_limits(truth_valid, pred_valid, component_key)

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
            ax.set_xlabel(f"Truth {component_label}")
            ax.set_ylabel(f"Pred {component_label}")
            if not legend_drawn:
                ax.legend(loc="upper left", frameon=False, fontsize=9)
                legend_drawn = True

            panel_metrics = make_neutrino_metrics(truth_valid, pred_valid, weight_valid)
            metrics[leg_name][component_key] = panel_metrics
            ax.set_title(
                f"{leg_name} {component_label}\n"
                f"CCC={panel_metrics['weighted_ccc']:.3f}, "
                f"R2={panel_metrics['weighted_r2']:.3f}, "
                f"nRMSE={panel_metrics['weighted_nrmse']:.3f}",
                fontsize=10,
            )

    fig.suptitle(f"Truth vs Predicted Neutrino {title_suffix}: {latex_process_label(process_name)}", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return metrics


def gather_neutrino_metrics(
    events: ak.Array,
    class_names: list[str],
    output_dir: Path,
    max_processes: int | None,
    use_weighted: bool,
    weight_source: str,
    class_weight_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    truth_name = np.asarray(ak.to_list(events["evenet_truth_class_name"]), dtype=object)
    valid_truth = truth_name != DEFAULT_CLASS_NAME
    process_names_present = [name for name in class_names if np.any(truth_name == name)]
    if max_processes is not None:
        process_names_present = process_names_present[:max_processes]

    cartesian_specs = [("E", "E"), ("px", "px"), ("py", "py"), ("pz", "pz")]
    kinematic_specs = [("energy", "energy"), ("pt", "pt"), ("eta", "eta"), ("phi", "phi")]
    summary: dict[str, Any] = {
        "all_processes": {"cartesian": {}, "kinematics": {}},
        "per_process": {},
    }

    all_events = events[valid_truth]
    summary["all_processes"]["cartesian"] = plot_neutrino_grid(
        all_events,
        process_name="All Processes",
        output_path=output_dir / "neutrino_truth_vs_pred_all.png",
        component_specs=cartesian_specs,
        title_suffix="Four-Momentum (Cartesian)",
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )
    summary["all_processes"]["kinematics"] = plot_neutrino_grid(
        all_events,
        process_name="All Processes",
        output_path=output_dir / "neutrino_truth_vs_pred_kinematics_all.png",
        component_specs=kinematic_specs,
        title_suffix="Kinematics",
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )

    for process_name in process_names_present:
        process_events = events[truth_name == process_name]
        if len(process_events) == 0:
            continue
        summary["per_process"][process_name] = {
            "cartesian": plot_neutrino_grid(
                process_events,
                process_name=process_name,
                output_path=output_dir / f"neutrino_truth_vs_pred_{process_name}.png",
                component_specs=cartesian_specs,
                title_suffix="Four-Momentum (Cartesian)",
                use_weighted=use_weighted,
                weight_source=weight_source,
                class_weight_map=class_weight_map,
            ),
            "kinematics": plot_neutrino_grid(
                process_events,
                process_name=process_name,
                output_path=output_dir / f"neutrino_truth_vs_pred_kinematics_{process_name}.png",
                component_specs=kinematic_specs,
                title_suffix="Kinematics",
                use_weighted=use_weighted,
                weight_source=weight_source,
                class_weight_map=class_weight_map,
            ),
        }

    ccc_values: list[float] = []
    for scope_metrics in [summary["all_processes"], *summary["per_process"].values()]:
        for view_metrics in scope_metrics.values():
            for leg_metrics in view_metrics.values():
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
    use_weighted: bool,
    weight_source: str,
    class_weight_map: dict[str, float] | None = None,
) -> None:
    mc_pred = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    mc_weight = event_weights(
        mc_events,
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )
    data_pred = np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object)

    mc_counts = np.array([float(np.sum(mc_weight[mc_pred == name])) for name in class_names], dtype=np.float64)
    data_counts = np.array([int(np.sum(data_pred == name)) for name in class_names], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.8), 6), dpi=220)
    x = np.arange(len(class_names))
    ax.bar(x - 0.2, mc_counts, width=0.4, label="Weighted MC", color=MC_COLOR, alpha=0.82)
    ax.bar(x + 0.2, data_counts, width=0.4, label="Data", color=DATA_COLOR, alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(latex_labels(class_names), rotation=45, ha="right")
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
    use_weighted: bool,
    weight_source: str,
    class_weight_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    mc_pred = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    mc_truth = np.asarray(ak.to_list(mc_events["evenet_truth_class_name"]), dtype=object)
    mc_weight = event_weights(
        mc_events,
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )

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

    fig_height = max(7, 0.58 * len(class_names) + 2.5)
    fig, ax = plt.subplots(figsize=(13.5, fig_height), dpi=220)
    y = np.arange(len(class_names))
    left = np.zeros(len(class_names), dtype=np.float64)

    for draw_idx, truth_name in enumerate(stack_draw_order(truth_processes)):
        truth_idx = truth_index[truth_name]
        values = stack_matrix[:, truth_idx]
        if not np.any(values > 0):
            continue
        ax.barh(
            y,
            values,
            left=left,
            height=0.75,
            color=process_stack_color(truth_name, draw_idx),
            edgecolor="white",
            linewidth=0.5,
            label=latex_process_label(truth_name),
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
    ax.set_xlim(0.0, visible_max * 1.18)

    def yield_label(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.0f}" if abs(value) >= 100 else f"{value:.1f}"

    for index, class_name in enumerate(class_names):
        mc_total = float(row_sums[index])
        data_yield = float(data_counts[index]) if data_counts is not None else float("nan")
        if mc_total > 0 and class_name in truth_index:
            signal_purity = float(stack_matrix[index, truth_index[class_name]] / mc_total)
            purity_label = f"{signal_purity:.3f}"
        else:
            signal_purity = float("nan")
            purity_label = "n/a"

        if np.isfinite(data_yield):
            data_minus_mc = data_yield - mc_total
            data_minus_mc_frac = data_minus_mc / mc_total if mc_total > 0 else float("nan")
            diff_label = (
                f"Δ={yield_label(data_minus_mc)}"
                + (f" ({data_minus_mc_frac:+.1%})" if np.isfinite(data_minus_mc_frac) else "")
            )
            label = (
                f"P={purity_label} | {diff_label}\n"
                f"MC={yield_label(mc_total)}  Data={yield_label(data_yield)}"
            )
        else:
            label = f"P={purity_label}\nMC={yield_label(mc_total)}"

        ax.text(
            1.01,
            y[index],
            label,
            va="center",
            ha="left",
            fontsize=8.2,
            color=DATA_COLOR,
            fontweight="semibold",
            linespacing=1.15,
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(latex_labels(class_names))
    ax.set_xlabel("Yield")
    ax.set_ylabel("Predicted channel")
    ax.set_title("Predicted Channel Purity: stacked truth-process yield with data overlay")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncols=2)
    fig.subplots_adjust(left=0.18, right=0.78, top=0.91, bottom=0.10)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    purity_summary = {}
    row_sums = stack_matrix.sum(axis=1)
    for index, class_name in enumerate(class_names):
        row = stack_matrix[index]
        total = float(row_sums[index])
        dominant_idx = int(np.argmax(row)) if row.size > 0 else -1
        data_yield = float(data_counts[index]) if data_counts is not None else float("nan")
        data_minus_mc = data_yield - total if np.isfinite(data_yield) else float("nan")
        data_minus_mc_fraction = data_minus_mc / total if total > 0 and np.isfinite(data_minus_mc) else float("nan")
        purity_summary[class_name] = {
            "total_mc_yield": total,
            "dominant_truth_process": truth_processes[dominant_idx] if total > 0 and dominant_idx >= 0 else DEFAULT_CLASS_NAME,
            "dominant_fraction": float(row[dominant_idx] / total) if total > 0 and dominant_idx >= 0 else float("nan"),
            "signal_purity": float(row[truth_index[class_name]] / total) if total > 0 and class_name in truth_index else float("nan"),
            "data_yield": data_yield,
            "data_stat_unc": float(data_unc[index]) if data_unc is not None else float("nan"),
            "data_minus_mc": data_minus_mc,
            "data_minus_mc_fraction": data_minus_mc_fraction,
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


def make_data_hist(values: np.ndarray, bins: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if weights is None:
        return make_count_hist(values, bins)
    counts = np.histogram(values, bins=bins, weights=weights)[0].astype(np.float64)
    variance = np.histogram(values, bins=bins, weights=weights * weights)[0].astype(np.float64)
    return counts, np.sqrt(variance)


def plot_region_histogram_panel(
    ax,
    bins: np.ndarray,
    mc_stack: dict[str, np.ndarray],
    data_hist: np.ndarray | None,
    data_err: np.ndarray | None,
    truth_hist: np.ndarray | None,
    title: str,
    xlabel: str,
    data_note: str | None = None,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    bottom = np.zeros_like(centers, dtype=np.float64)

    for index, process_name in enumerate(stack_draw_order(list(mc_stack))):
        hist = mc_stack[process_name]
        if np.sum(hist) <= 0:
            continue
        ax.bar(
            centers,
            hist,
            width=widths,
            bottom=bottom,
            color=process_stack_color(process_name, index),
            edgecolor="white",
            linewidth=0.4,
            align="center",
            label=latex_process_label(process_name),
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
    if data_note:
        ax.text(
            0.98,
            0.95,
            data_note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=DATA_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )


def plot_region_kinematics(
    region_name: str,
    mc_events: ak.Array,
    data_events: ak.Array | None,
    class_names: list[str],
    output_path: Path,
    use_weighted: bool,
    weight_source: str,
    class_weight_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    region_mc = mc_events[np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object) == region_name]
    region_data = None
    if data_events is not None:
        region_data = data_events[np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object) == region_name]

    truth_names_mc = np.asarray(ak.to_list(region_mc["evenet_truth_class_name"]), dtype=object)
    base_mc_weights = (
        event_weights(
            region_mc,
            use_weighted=use_weighted,
            weight_source=weight_source,
            class_weight_map=class_weight_map,
        )
        if len(region_mc) > 0
        else np.array([], dtype=np.float64)
    )
    truth_processes = [name for name in class_names if np.any(truth_names_mc == name)]

    mc_kin = build_full_tau_kinematics(region_mc) if len(region_mc) > 0 else None
    data_kin = build_full_tau_kinematics(region_data) if (region_data is not None and len(region_data) > 0) else None

    panel_specs = [
        ("z_mass", "Z mass from reconstructed tau pair", r"$m_{\tau\tau}$"),
        ("delta_eta", r"$\Delta\eta(\tau,\tau)$", r"$|\Delta\eta|$"),
        ("delta_phi", r"$\Delta\phi(\tau,\tau)$", r"$|\Delta\phi|$"),
        ("vis_delta_eta", r"$\Delta\eta(\tau_{\mathrm{vis}},\tau_{\mathrm{vis}})$", r"$|\Delta\eta_{\mathrm{vis}}|$"),
        ("vis_delta_phi", r"$\Delta\phi(\tau_{\mathrm{vis}},\tau_{\mathrm{vis}})$", r"$|\Delta\phi_{\mathrm{vis}}|$"),
        ("invis_delta_eta", r"$\Delta\eta(\tau_{\mathrm{inv}},\tau_{\mathrm{inv}})$", r"$|\Delta\eta_{\mathrm{inv}}|$"),
        ("invis_delta_phi", r"$\Delta\phi(\tau_{\mathrm{inv}},\tau_{\mathrm{inv}})$", r"$|\Delta\phi_{\mathrm{inv}}|$"),
        ("tau_E", r"Tau energy", r"$E_\tau$"),
        ("tau_px", r"Tau px", r"$p_{x,\tau}$"),
        ("tau_py", r"Tau py", r"$p_{y,\tau}$"),
        ("tau_pz", r"Tau pz", r"$p_{z,\tau}$"),
    ]

    ncols = 4
    nrows = math.ceil(len(panel_specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.5 * nrows), dpi=220)
    axes = np.atleast_1d(axes).flatten()
    summary: dict[str, Any] = {}
    region_data_count = len(region_data) if region_data is not None else 0

    for axis, (key, title, xlabel) in zip(axes, panel_specs):
        if mc_kin is None:
            axis.set_title(f"{title}\nno MC events")
            axis.axis("off")
            continue

        if key == "z_mass":
            mc_pred_values = mc_kin["pred_pair"]["mass"]
            mc_truth_values = mc_kin["truth_pair"]["mass"]
            mc_reco_valid = mc_kin["pred_pair"]["valid"]
            mc_truth_valid = mc_kin["truth_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_pair"]["mass"] if data_kin is not None else None
            data_valid = data_kin["pred_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        elif key == "delta_eta":
            mc_pred_values = mc_kin["pred_pair"]["delta_eta"]
            mc_truth_values = mc_kin["truth_pair"]["delta_eta"]
            mc_reco_valid = mc_kin["pred_pair"]["valid"]
            mc_truth_valid = mc_kin["truth_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_pair"]["delta_eta"] if data_kin is not None else None
            data_valid = data_kin["pred_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        elif key == "delta_phi":
            mc_pred_values = mc_kin["pred_pair"]["delta_phi"]
            mc_truth_values = mc_kin["truth_pair"]["delta_phi"]
            mc_reco_valid = mc_kin["pred_pair"]["valid"]
            mc_truth_valid = mc_kin["truth_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_pair"]["delta_phi"] if data_kin is not None else None
            data_valid = data_kin["pred_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        elif key == "vis_delta_eta":
            mc_pred_values = mc_kin["visible_pair"]["delta_eta"]
            mc_truth_values = mc_kin["visible_pair"]["delta_eta"]
            mc_reco_valid = mc_kin["visible_pair"]["valid"]
            mc_truth_valid = mc_kin["visible_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["visible_pair"]["delta_eta"] if data_kin is not None else None
            data_valid = data_kin["visible_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        elif key == "vis_delta_phi":
            mc_pred_values = mc_kin["visible_pair"]["delta_phi"]
            mc_truth_values = mc_kin["visible_pair"]["delta_phi"]
            mc_reco_valid = mc_kin["visible_pair"]["valid"]
            mc_truth_valid = mc_kin["visible_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["visible_pair"]["delta_phi"] if data_kin is not None else None
            data_valid = data_kin["visible_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        elif key == "invis_delta_eta":
            mc_pred_values = mc_kin["pred_invisible_pair"]["delta_eta"]
            mc_truth_values = mc_kin["truth_invisible_pair"]["delta_eta"]
            mc_reco_valid = mc_kin["pred_invisible_pair"]["valid"]
            mc_truth_valid = mc_kin["truth_invisible_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_invisible_pair"]["delta_eta"] if data_kin is not None else None
            data_valid = data_kin["pred_invisible_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        elif key == "invis_delta_phi":
            mc_pred_values = mc_kin["pred_invisible_pair"]["delta_phi"]
            mc_truth_values = mc_kin["truth_invisible_pair"]["delta_phi"]
            mc_reco_valid = mc_kin["pred_invisible_pair"]["valid"]
            mc_truth_valid = mc_kin["truth_invisible_pair"]["valid"]
            mc_weights_for_hist = base_mc_weights
            data_values = data_kin["pred_invisible_pair"]["delta_phi"] if data_kin is not None else None
            data_valid = data_kin["pred_invisible_pair"]["valid"] if data_kin is not None else None
            data_weights_for_hist = None
            symmetric = False
        else:
            component = key.split("_", 1)[1]
            mc_pred_values = np.concatenate([mc_kin["pred_tau"][0][component], mc_kin["pred_tau"][1][component]])
            mc_truth_values = np.concatenate([mc_kin["truth_tau"][0][component], mc_kin["truth_tau"][1][component]])
            mc_reco_valid = np.concatenate([
                mc_kin["pred_tau"][0]["valid"],
                mc_kin["pred_tau"][1]["valid"],
            ])
            mc_truth_valid = np.concatenate([
                mc_kin["truth_tau"][0]["valid"],
                mc_kin["truth_tau"][1]["valid"],
            ])
            if data_kin is not None:
                data_values = np.concatenate([data_kin["pred_tau"][0][component], data_kin["pred_tau"][1][component]])
                data_valid = np.concatenate([data_kin["pred_tau"][0]["valid"], data_kin["pred_tau"][1]["valid"]])
                data_weights_for_hist = np.full(len(data_values), 0.5, dtype=np.float64)
            else:
                data_values = None
                data_valid = None
                data_weights_for_hist = None
            symmetric = component in {"px", "py", "pz"}
            mc_weights_for_hist = 0.5 * np.concatenate([base_mc_weights, base_mc_weights])
        mc_reco_finite = mc_reco_valid & finite_mask(mc_pred_values, mc_weights_for_hist)
        mc_truth_finite = mc_truth_valid & finite_mask(mc_truth_values, mc_weights_for_hist)
        combined_for_bins = np.concatenate([
            mc_pred_values[mc_reco_finite],
            mc_truth_values[mc_truth_finite],
            data_values[data_valid] if data_values is not None and data_valid is not None and np.any(data_valid) else np.array([], dtype=np.float64),
        ])
        bins = choose_hist_bins(combined_for_bins, symmetric=symmetric)

        mc_stack = {}
        if key.startswith("tau_"):
            expanded_truth_names = np.concatenate([truth_names_mc, truth_names_mc])
            expanded_weights = mc_weights_for_hist
            for process_name in truth_processes:
                process_mask = (expanded_truth_names == process_name) & mc_reco_finite
                mc_stack[process_name] = make_weighted_hist(mc_pred_values[process_mask], expanded_weights[process_mask], bins)
            truth_hist = make_weighted_hist(mc_truth_values[mc_truth_finite], expanded_weights[mc_truth_finite], bins)
        else:
            for process_name in truth_processes:
                process_mask = (truth_names_mc == process_name) & mc_reco_finite
                mc_stack[process_name] = make_weighted_hist(mc_pred_values[process_mask], mc_weights_for_hist[process_mask], bins)
            truth_hist = make_weighted_hist(mc_truth_values[mc_truth_finite], mc_weights_for_hist[mc_truth_finite], bins)

        data_hist = data_err = None
        data_note = None
        if data_values is not None and data_valid is not None:
            data_mask = data_valid & finite_mask(data_values)
            if np.any(data_mask):
                data_weights = data_weights_for_hist[data_mask] if data_weights_for_hist is not None else None
                data_hist, data_err = make_data_hist(data_values[data_mask], bins, weights=data_weights)
            elif region_data_count > 0:
                data_note = "Data present, but no valid prediction\nstored for this observable"

        plot_region_histogram_panel(
            axis,
            bins=bins,
            mc_stack=mc_stack,
            data_hist=data_hist,
            data_err=data_err,
            truth_hist=truth_hist,
            title=title,
            xlabel=xlabel,
            data_note=data_note,
        )

        summary[key] = {
            "mc_reco_yield": float(sum(np.sum(hist) for hist in mc_stack.values())),
            "mc_truth_yield": float(np.sum(truth_hist)),
            "data_yield": float(np.sum(data_hist)) if data_hist is not None else float("nan"),
        }

    for axis in axes[len(panel_specs):]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center right", frameon=False)
    title_suffix = "data vs stacked MC" if data_events is not None else "stacked MC"
    fig.suptitle(f"Region {latex_process_label(region_name)}: {title_suffix} (with MC truth reference)", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def gather_region_kinematics_plots(
    mc_events: ak.Array,
    data_events: ak.Array | None,
    class_names: list[str],
    output_dir: Path,
    max_processes: int | None,
    use_weighted: bool,
    weight_source: str,
    class_weight_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    predicted_names = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    if data_events is not None:
        data_predicted_names = np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object)
        predicted_names = np.concatenate([predicted_names, data_predicted_names])
    region_names = [name for name in class_names if np.any(predicted_names == name)]
    if max_processes is not None:
        region_names = region_names[:max_processes]

    summary: dict[str, Any] = {}
    for region_name in region_names:
        summary[region_name] = plot_region_kinematics(
            region_name=region_name,
            mc_events=mc_events,
            data_events=data_events,
            class_names=class_names,
            output_path=output_dir / f"region_kinematics_{region_name}.png",
            use_weighted=use_weighted,
            weight_source=weight_source,
            class_weight_map=class_weight_map,
        )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_paths = expand_paths(args.mc_parquet)
    data_paths = expand_paths(args.data_parquet)
    mc_events = load_events(mc_paths)
    data_events = load_events(data_paths) if (data_paths and args.unblind) else None
    use_weighted = not args.unweighted
    weight_source = "unit" if args.unweighted else args.weight_source

    class_names = build_class_names_from_analysis(args.analysis_config.resolve(), args.evenet_config.resolve())
    summary_class_names = summary_channel_order(class_names)
    class_weight_map = (
        build_prediction_class_weights(
            args.analysis_config.resolve(),
            args.evenet_config.resolve(),
            class_names,
        )
        if weight_source in {"auto", "class"}
        else None
    )

    truth_idx = to_numpy(mc_events["evenet_truth_class_index"], np.int64)
    pred_idx = to_numpy(mc_events["evenet_pred_class_index"], np.int64)
    weights = event_weights(
        mc_events,
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )
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
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )

    if data_events is not None:
        plot_predicted_class_comparison(
            mc_events=mc_events,
            data_events=data_events,
            class_names=summary_class_names,
            output_path=output_dir / "predicted_class_data_vs_mc.png",
            use_weighted=use_weighted,
            weight_source=weight_source,
            class_weight_map=class_weight_map,
        )
    purity_metrics = plot_predicted_channel_purity(
        mc_events=mc_events,
        data_events=data_events,
        class_names=summary_class_names,
        output_path=output_dir / "predicted_channel_purity.png",
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )
    region_kinematics_metrics = gather_region_kinematics_plots(
        mc_events=mc_events,
        data_events=data_events,
        class_names=summary_class_names,
        output_dir=output_dir,
        max_processes=args.max_processes,
        use_weighted=use_weighted,
        weight_source=weight_source,
        class_weight_map=class_weight_map,
    )

    metrics_payload = {
        "inputs": {
            "mc_parquet": [str(Path(path).resolve()) for path in mc_paths],
            "data_parquet": [str(Path(path).resolve()) for path in data_paths],
            "analysis_config": str(args.analysis_config.resolve()),
            "evenet_config": str(args.evenet_config.resolve()),
            "use_weighted": use_weighted,
            "weight_source": weight_source,
            "class_weight_map_enabled": class_weight_map is not None,
            "summary_channel_order": summary_class_names,
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
