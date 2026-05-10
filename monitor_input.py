#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from generate_event_info_yaml import parse_feature_config
from plot_style import plot_2d_histogram_comparison, plot_data_mc_histogram_from_counts
from quantum.observables_builder import build_observables, get_observable_names

DEFAULT_SAMPLE_LIMIT = 200_000
QUANTUM_RANGES = {
    "theta_cm": (0.0, 1.0),
    "mtautau": (0.0, 150.0),
}
for _axis in ("n", "r", "k"):
    QUANTUM_RANGES[f"cos_theta_A_{_axis}"] = (-1.0, 1.0)
    QUANTUM_RANGES[f"cos_theta_B_{_axis}"] = (-1.0, 1.0)
for _axis_a in ("n", "r", "k"):
    for _axis_b in ("n", "r", "k"):
        QUANTUM_RANGES[f"cos_theta_A_{_axis_a}_times_cos_theta_B_{_axis_b}"] = (-1.0, 1.0)
del _axis, _axis_a, _axis_b


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("Install pyyaml or use JSON config.")
        return yaml.safe_load(text)
    return json.loads(text)


def parquet_files(paths: list[Path]) -> list[str]:
    files: list[str] = []
    for path in paths:
        if path.is_file() and path.suffix == ".parquet":
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(p) for p in sorted(path.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {paths}")
    return files


def build_sample_map(paths: list[Path], default_label: str, combine: bool) -> dict[str, list[str]]:
    if combine:
        return {default_label: parquet_files(paths)}

    samples: dict[str, list[str]] = {}
    for index, path in enumerate(paths):
        label = path.stem if path.is_file() else path.name
        if not label:
            label = f"{default_label}_{index}"
        if label in samples:
            label = f"{label}_{index}"
        samples[label] = parquet_files([path])
    return samples


def available_columns(file_name: str) -> set[str] | None:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(file_name).schema_arrow.names)
    except Exception:
        return None


def row_group_chunks(file_name: str, row_groups_per_chunk: int) -> list[list[int] | None]:
    if row_groups_per_chunk <= 0:
        return [None]
    try:
        import inspect

        if "row_groups" not in inspect.signature(ak.from_parquet).parameters:
            return [None]
    except Exception:
        return [None]
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(file_name)
        num_row_groups = parquet_file.num_row_groups
    except Exception:
        return [None]

    chunks: list[list[int] | None] = []
    for start in range(0, num_row_groups, row_groups_per_chunk):
        chunks.append(list(range(start, min(start + row_groups_per_chunk, num_row_groups))))
    return chunks or [None]


def read_parquet_chunk(file_name: str, columns: list[str], row_groups: list[int] | None) -> ak.Array | None:
    present = available_columns(file_name)
    requested = list(dict.fromkeys(columns))
    selected = requested if present is None else [column for column in requested if column in present]
    if not selected:
        return None
    if row_groups is not None:
        try:
            return ak.from_parquet(file_name, columns=selected, row_groups=row_groups)
        except TypeError:
            pass
    return ak.from_parquet(file_name, columns=selected)


def to_numpy(array: Any, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(ak.to_numpy(array, allow_missing=False), dtype=dtype)
    except TypeError:
        return np.asarray(ak.to_numpy(array), dtype=dtype)
    except Exception:
        return np.asarray(ak.to_list(array), dtype=dtype)


def event_weights(events: ak.Array, weight_column: str | None) -> np.ndarray | None:
    if not weight_column or weight_column not in events.fields:
        return None
    weights = to_numpy(events[weight_column], np.float64).reshape(-1)
    if len(weights) != len(events):
        return None
    return weights


def condition_values(events: ak.Array, global_idx: int) -> np.ndarray:
    conditions = to_numpy(events["conditions"], np.float64)
    if conditions.ndim != 2:
        conditions = np.asarray(conditions, dtype=np.float64).reshape((len(events), -1))
    if global_idx >= conditions.shape[1]:
        raise IndexError(f"conditions only has {conditions.shape[1]} columns; requested index {global_idx}.")
    return conditions[:, global_idx]


def sequential_values_and_weights(
    events: ak.Array,
    sequential_idx: int,
    weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    x = to_numpy(events["x"], np.float64)
    x_mask = to_numpy(events["x_mask"], bool)
    if x.ndim != 3:
        raise ValueError(f"x must be rank-3, got shape {x.shape}")
    if x_mask.shape != x.shape[:2]:
        raise ValueError(f"x_mask shape {x_mask.shape} does not match x leading shape {x.shape[:2]}")
    if sequential_idx >= x.shape[2]:
        raise IndexError(f"x only has {x.shape[2]} features; requested index {sequential_idx}.")

    values = x[:, :, sequential_idx][x_mask]
    weights = event_weights(events, weight_column)
    if weights is None:
        return values, None
    expanded_weights = np.broadcast_to(weights[:, None], x_mask.shape)[x_mask]
    return values, expanded_weights


def update_summary(summary: dict[str, Any], values: np.ndarray, sample_limit: int) -> None:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return

    summary["count"] += int(finite_values.size)
    summary["min"] = min(summary["min"], float(np.min(finite_values)))
    summary["max"] = max(summary["max"], float(np.max(finite_values)))
    summary["integer"] = bool(summary["integer"] and np.allclose(finite_values, np.round(finite_values)))

    sampled = summary["sampled"]
    if sampled.size < sample_limit:
        need = sample_limit - sampled.size
        if finite_values.size > need:
            indices = np.linspace(0, finite_values.size - 1, need, dtype=np.int64)
            finite_values = finite_values[indices]
        summary["sampled"] = np.concatenate([sampled, finite_values])


def make_bins(summary: dict[str, Any], bins: int, max_integer_bins: int) -> np.ndarray:
    if summary["count"] == 0:
        return np.linspace(0.0, 1.0, bins + 1)

    low = float(summary["min"])
    high = float(summary["max"])
    if summary["integer"]:
        int_low = math.floor(low)
        int_high = math.ceil(high)
        if int_high - int_low + 1 <= max_integer_bins:
            return np.arange(int_low - 0.5, int_high + 1.5, 1.0)

    sampled = summary["sampled"]
    if sampled.size:
        low, high = np.quantile(sampled, [0.005, 0.995])

    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(summary["min"])
        high = float(summary["max"])
    if low == high:
        pad = abs(low) if low else 1.0
        low -= 0.5 * pad
        high += 0.5 * pad

    pad = 0.05 * (high - low)
    return np.linspace(low - pad, high + pad, bins + 1)


def empty_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "min": float("inf"),
        "max": float("-inf"),
        "integer": True,
        "sampled": np.array([], dtype=np.float64),
    }


def sample_feature_values(
    events: ak.Array,
    feature_kind: str,
    feature_idx: int,
    weight_column: str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if feature_kind == "global":
        return condition_values(events, feature_idx), event_weights(events, weight_column)
    if feature_kind == "sequential":
        return sequential_values_and_weights(events, feature_idx, weight_column)
    raise ValueError(f"Unsupported feature kind: {feature_kind}")


def needed_columns(feature_kind: str, weight_column: str | None) -> list[str]:
    columns = ["conditions"] if feature_kind == "global" else ["x", "x_mask"]
    if weight_column:
        columns.append(weight_column)
    return columns


def scan_feature_range(
    sample_files: dict[str, list[str]],
    feature_kind: str,
    feature_idx: int,
    weight_column: str | None,
    max_events: int | None,
    row_groups_per_chunk: int,
) -> dict[str, Any]:
    summary = empty_summary()
    rows_by_sample: dict[str, int] = {}
    columns = needed_columns(feature_kind, weight_column)
    for sample_name, files in sample_files.items():
        remaining = max_events
        rows_by_sample[sample_name] = 0
        for file_name in files:
            for row_groups in row_group_chunks(file_name, row_groups_per_chunk):
                if remaining is not None and remaining <= 0:
                    break
                events = read_parquet_chunk(file_name, columns, row_groups)
                if events is None:
                    continue
                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                values, _ = sample_feature_values(events, feature_kind, feature_idx, weight_column)
                rows_by_sample[sample_name] += int(len(events))
                update_summary(summary, values, DEFAULT_SAMPLE_LIMIT)
                if remaining is not None:
                    remaining -= int(len(events))
                del events, values
                gc.collect()
            if remaining is not None and remaining <= 0:
                break
    return {"summary": summary, "rows_by_sample": rows_by_sample}


def histogram_feature(
    sample_files: dict[str, list[str]],
    feature_kind: str,
    feature_idx: int,
    bins: np.ndarray,
    weight_column: str | None,
    max_events: int | None,
    row_groups_per_chunk: int,
) -> dict[str, dict[str, Any]]:
    histograms: dict[str, dict[str, Any]] = {}
    columns = needed_columns(feature_kind, weight_column)
    for sample_name, files in sample_files.items():
        counts = np.zeros(len(bins) - 1, dtype=np.float64)
        sumw2 = np.zeros(len(bins) - 1, dtype=np.float64)
        rows = 0
        weighted = False
        remaining = max_events
        for file_name in files:
            for row_groups in row_group_chunks(file_name, row_groups_per_chunk):
                if remaining is not None and remaining <= 0:
                    break
                events = read_parquet_chunk(file_name, columns, row_groups)
                if events is None:
                    continue
                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                values, weights = sample_feature_values(events, feature_kind, feature_idx, weight_column)

                finite = np.isfinite(values)
                if weights is not None:
                    finite &= np.isfinite(weights)
                    weight_values = weights[finite]
                    weighted = True
                    counts += np.histogram(values[finite], bins=bins, weights=weight_values)[0]
                    sumw2 += np.histogram(values[finite], bins=bins, weights=weight_values * weight_values)[0]
                else:
                    counts_chunk = np.histogram(values[finite], bins=bins)[0].astype(np.float64)
                    counts += counts_chunk
                    sumw2 += counts_chunk

                rows += int(len(events))
                if remaining is not None:
                    remaining -= int(len(events))
                del events, values, weights
                gc.collect()
            if remaining is not None and remaining <= 0:
                break
        histograms[sample_name] = {
            "counts": counts,
            "sumw2": sumw2,
            "rows": rows,
            "weighted": weighted,
        }
    return histograms


def plot_control_histograms(
    feature_name: str,
    bins: np.ndarray,
    data_histograms: dict[str, dict[str, Any]],
    mc_histograms: dict[str, dict[str, Any]],
    output_dir: Path,
    title_prefix: str,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_counts = np.zeros(len(bins) - 1, dtype=np.float64)
    data_sumw2 = np.zeros(len(bins) - 1, dtype=np.float64)
    for payload in data_histograms.values():
        data_counts += payload["counts"]
        data_sumw2 += payload["sumw2"]
    mc_counts = {sample_name: payload["counts"] for sample_name, payload in mc_histograms.items()}
    plot_path = output_dir / f"{feature_name.replace('/', '_')}.png"
    return plot_data_mc_histogram_from_counts(
        bins,
        data_counts,
        data_sumw2,
        mc_counts,
        plot_path,
        title=f"{title_prefix}: {feature_name}",
        xlabel=feature_name,
    )


def process_control_feature(payload: dict[str, Any]) -> dict[str, Any]:
    feature_idx = payload["feature_idx"]
    feature_name = payload["feature_name"]
    feature_kind = payload["feature_kind"]
    sample_files = {**payload["data_files"], **payload["mc_files"]}
    range_result = scan_feature_range(
        sample_files,
        feature_kind,
        feature_idx,
        payload["weight_column"],
        payload["max_events"],
        payload["row_groups_per_chunk"],
    )
    bins = make_bins(range_result["summary"], payload["bins"], payload["max_integer_bins"])
    data_histograms = histogram_feature(
        payload["data_files"],
        feature_kind,
        feature_idx,
        bins,
        payload["weight_column"],
        payload["max_events"],
        payload["row_groups_per_chunk"],
    )
    mc_histograms = histogram_feature(
        payload["mc_files"],
        feature_kind,
        feature_idx,
        bins,
        payload["weight_column"],
        payload["max_events"],
        payload["row_groups_per_chunk"],
    )
    plot_path = plot_control_histograms(
        feature_name,
        bins,
        data_histograms,
        mc_histograms,
        Path(payload["output_dir"]),
        payload["title_prefix"],
    )
    return {
        "feature_kind": feature_kind,
        "feature_name": feature_name,
        "plot": plot_path,
        "rows_by_sample": range_result["rows_by_sample"],
    }


def p4_field_candidates(prefixes: tuple[str, ...], component: str) -> list[str]:
    return [f"{prefix}_{component}" for prefix in prefixes]


def first_existing_field(events: ak.Array, candidates: list[str]) -> str | None:
    fields = set(events.fields)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def component_array(events: ak.Array, candidates: list[str], slot: int | None = None) -> np.ndarray | None:
    field = first_existing_field(events, candidates)
    if field is None:
        return None
    values = to_numpy(events[field], np.float64)
    if slot is not None and values.ndim == 2 and values.shape[1] > slot:
        return values[:, slot]
    return values


def p4_from_component_prefixes(
    events: ak.Array,
    prefixes: tuple[str, ...],
    slot: int | None = None,
) -> ak.Array | None:
    components = {
        "px": component_array(events, p4_field_candidates(prefixes, "px"), slot),
        "py": component_array(events, p4_field_candidates(prefixes, "py"), slot),
        "pz": component_array(events, p4_field_candidates(prefixes, "pz"), slot),
        "E": component_array(events, p4_field_candidates(prefixes, "E"), slot),
    }
    if any(value is None for value in components.values()):
        return None
    return ak.zip(
        {
            "px": components["px"],
            "py": components["py"],
            "pz": components["pz"],
            "E": components["E"],
        },
        with_name="Momentum4D",
    )


def target_visible_columns() -> list[str]:
    columns = []
    for leg in ("a", "b"):
        visible_prefixes = (
            f"lead_{leg}_visible",
            f"visible_{leg}",
            "visible",
        )
        target_prefixes = (
            f"target_{leg}_invisible",
            f"target_{leg}_missing",
            f"target_missing_{leg}",
            f"lead_{leg}_missing",
            "target_missing",
            "target_invisible",
        )
        for component in ("px", "py", "pz", "E"):
            columns.extend(p4_field_candidates(visible_prefixes, component))
            columns.extend(p4_field_candidates(target_prefixes, component))
    return columns


def build_target_observables(events: ak.Array) -> dict[str, np.ndarray] | None:
    try:
        import vector

        vector.register_awkward()
    except Exception:
        pass

    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1)
    target_a = p4_from_component_prefixes(
        events,
        ("target_a_invisible", "target_a_missing", "target_missing_a", "lead_a_missing", "target_missing", "target_invisible"),
        slot=0,
    )
    target_b = p4_from_component_prefixes(
        events,
        ("target_b_invisible", "target_b_missing", "target_missing_b", "lead_b_missing", "target_missing", "target_invisible"),
        slot=1,
    )
    if visible_a is None or visible_b is None or target_a is None or target_b is None:
        return None
    observables = build_observables(
        tau_a_p4=visible_a + target_a,
        tau_b_p4=visible_b + target_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )
    return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}


def quantum_columns(observable: str, weight_column: str | None) -> list[str]:
    columns = target_visible_columns()
    columns.extend([f"truth_{observable}", f"baseline_{observable}"])
    if weight_column:
        columns.append(weight_column)
    return list(dict.fromkeys(columns))


def observable_values(
    events: ak.Array,
    source: str,
    observable: str,
    target_observables: dict[str, np.ndarray] | None,
) -> np.ndarray | None:
    if source == "target":
        if target_observables is None:
            return None
        return target_observables.get(observable)
    field = f"{source}_{observable}"
    if source == "truth":
        field = f"truth_{observable}"
    if source == "baseline":
        field = f"baseline_{observable}"
    if field not in events.fields:
        return None
    return to_numpy(events[field], np.float64)


def empty_comparison_state(x_edges: np.ndarray, y_edges: np.ndarray) -> dict[str, Any]:
    return {
        "counts": np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float64),
        "sumw": 0.0,
        "sumx": 0.0,
        "sumy": 0.0,
        "sumxx": 0.0,
        "sumyy": 0.0,
        "sumxy": 0.0,
        "entries": 0,
    }


def update_comparison_state(
    state: dict[str, Any],
    x_values: np.ndarray,
    y_values: np.ndarray,
    weights: np.ndarray | None,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> None:
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if weights is not None:
        finite &= np.isfinite(weights)
        weight_values = weights[finite]
    else:
        weight_values = None
    x = x_values[finite]
    y = y_values[finite]
    if x.size == 0:
        return
    state["counts"] += np.histogram2d(x, y, bins=(x_edges, y_edges), weights=weight_values)[0]
    w = np.ones_like(x, dtype=np.float64) if weight_values is None else weight_values.astype(np.float64)
    state["sumw"] += float(np.sum(w))
    state["sumx"] += float(np.sum(w * x))
    state["sumy"] += float(np.sum(w * y))
    state["sumxx"] += float(np.sum(w * x * x))
    state["sumyy"] += float(np.sum(w * y * y))
    state["sumxy"] += float(np.sum(w * x * y))
    state["entries"] += int(x.size)


def comparison_statistics(state: dict[str, Any]) -> dict[str, float]:
    sumw = float(state["sumw"])
    if sumw <= 0:
        return {"mean_x": math.nan, "mean_y": math.nan, "corr": math.nan}
    mean_x = state["sumx"] / sumw
    mean_y = state["sumy"] / sumw
    var_x = max(state["sumxx"] / sumw - mean_x * mean_x, 0.0)
    var_y = max(state["sumyy"] / sumw - mean_y * mean_y, 0.0)
    cov = state["sumxy"] / sumw - mean_x * mean_y
    corr = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else math.nan
    return {"mean_x": mean_x, "mean_y": mean_y, "corr": corr}


def plot_2d_comparison(
    observable: str,
    comparison_name: str,
    x_label: str,
    y_label: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    state: dict[str, Any],
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = comparison_statistics(state)
    plot_path = output_dir / f"{observable}__{comparison_name}.png"
    return plot_2d_histogram_comparison(
        x_edges,
        y_edges,
        state["counts"],
        plot_path,
        title=f"{comparison_name}: {observable}",
        xlabel=x_label,
        ylabel=y_label,
        entries=int(state["entries"]),
        correlation=stats["corr"],
    )


def process_quantum_observable(payload: dict[str, Any]) -> dict[str, Any]:
    observable = payload["observable"]
    low, high = QUANTUM_RANGES.get(observable, (-1.0, 1.0))
    x_edges = np.linspace(low, high, payload["bins_2d"] + 1)
    y_edges = np.linspace(low, high, payload["bins_2d"] + 1)
    comparisons = {
        "truth_vs_target": ("truth", "target", "Stored truth", "Target"),
        "truth_vs_baseline": ("truth", "baseline", "Stored truth", "Baseline"),
        "baseline_vs_target": ("baseline", "target", "Baseline", "Target"),
    }
    states = {
        name: empty_comparison_state(x_edges, y_edges)
        for name in comparisons
    }
    columns = quantum_columns(observable, payload["weight_column"])
    sample_files = {**payload["data_files"], **payload["mc_files"]}

    for files in sample_files.values():
        remaining = payload["max_events"]
        for file_name in files:
            for row_groups in row_group_chunks(file_name, payload["row_groups_per_chunk"]):
                if remaining is not None and remaining <= 0:
                    break
                events = read_parquet_chunk(file_name, columns, row_groups)
                if events is None:
                    continue
                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                target_observables = build_target_observables(events)
                weights = event_weights(events, payload["weight_column"])
                for comparison_name, (x_source, y_source, _, _) in comparisons.items():
                    x_values = observable_values(events, x_source, observable, target_observables)
                    y_values = observable_values(events, y_source, observable, target_observables)
                    if x_values is None or y_values is None:
                        continue
                    update_comparison_state(states[comparison_name], x_values, y_values, weights, x_edges, y_edges)
                if remaining is not None:
                    remaining -= int(len(events))
                del events, target_observables, weights
                gc.collect()
            if remaining is not None and remaining <= 0:
                break

    plots = {}
    output_dir = Path(payload["output_dir"])
    for comparison_name, (_, _, x_label, y_label) in comparisons.items():
        if states[comparison_name]["entries"] == 0:
            continue
        plots[comparison_name] = plot_2d_comparison(
            observable,
            comparison_name,
            x_label,
            y_label,
            x_edges,
            y_edges,
            states[comparison_name],
            output_dir,
        )
    return {
        "observable": observable,
        "plots": plots,
        "entries": {name: int(state["entries"]) for name, state in states.items()},
    }


def run_tasks(tasks: list[dict[str, Any]], workers: int, worker_func, log_key: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            print(f"[monitor-input] plotting {task[log_key]}", flush=True)
            results.append(worker_func(task))
        return results

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        future_map = {executor.submit(worker_func, task): task[log_key] for task in tasks}
        for future in as_completed(future_map):
            name = future_map[future]
            print(f"[monitor-input] finished {name}", flush=True)
            results.append(future.result())
    return results


def main() -> None:
    parser = argparse.ArgumentParser("Monitor EveNet input parquet files")
    parser.add_argument("--data-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--mc-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("monitor_plots"))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--bins-2d", type=int, default=70)
    parser.add_argument("--max-integer-bins", type=int, default=120)
    parser.add_argument("--weight-column", default="event_weight")
    parser.add_argument("--row-groups-per-chunk", type=int, default=1)
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--skip-quantum", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    feature_config = parse_feature_config(config)

    data_files = build_sample_map(args.data_dir, "data", combine=True)
    mc_files = build_sample_map(args.mc_dir, "mc", combine=False)
    common_payload = {
        "data_files": data_files,
        "mc_files": mc_files,
        "max_events": args.max_events,
        "bins": args.bins,
        "max_integer_bins": args.max_integer_bins,
        "weight_column": args.weight_column or None,
        "row_groups_per_chunk": args.row_groups_per_chunk,
    }

    all_results: dict[str, Any] = {
        "data_samples": {name: len(files) for name, files in data_files.items()},
        "mc_samples": {name: len(files) for name, files in mc_files.items()},
        "max_events": args.max_events,
        "weight_column": args.weight_column,
    }

    if not args.skip_global:
        global_dir = args.output_dir / "global_conditions"
        global_tasks = [
            {
                **common_payload,
                "feature_kind": "global",
                "feature_idx": global_idx,
                "feature_name": global_name,
                "output_dir": str(global_dir),
                "title_prefix": "Global condition",
            }
            for global_idx, global_name in enumerate(feature_config.global_fields)
        ]
        print(f"[monitor-input] global conditions={len(global_tasks)} workers={args.num_workers}", flush=True)
        global_results = run_tasks(global_tasks, args.num_workers, process_control_feature, "feature_name")
        all_results["global_conditions"] = {
            result["feature_name"]: str(Path(result["plot"]).relative_to(args.output_dir))
            for result in global_results
        }

    if not args.skip_sequential:
        sequential_dir = args.output_dir / "sequential_inputs"
        sequential_tasks = [
            {
                **common_payload,
                "feature_kind": "sequential",
                "feature_idx": sequential_idx,
                "feature_name": sequential_name,
                "output_dir": str(sequential_dir),
                "title_prefix": "Sequential input",
            }
            for sequential_idx, sequential_name in enumerate(feature_config.raw_sequential_fields)
        ]
        print(f"[monitor-input] sequential inputs={len(sequential_tasks)} workers={args.num_workers}", flush=True)
        sequential_results = run_tasks(sequential_tasks, args.num_workers, process_control_feature, "feature_name")
        all_results["sequential_inputs"] = {
            result["feature_name"]: str(Path(result["plot"]).relative_to(args.output_dir))
            for result in sequential_results
        }

    if not args.skip_quantum:
        quantum_dir = args.output_dir / "quantum_2d"
        quantum_tasks = [
            {
                **common_payload,
                "observable": observable,
                "bins_2d": args.bins_2d,
                "output_dir": str(quantum_dir),
            }
            for observable in get_observable_names()
        ]
        print(f"[monitor-input] quantum comparisons={len(quantum_tasks)} workers={args.num_workers}", flush=True)
        quantum_results = run_tasks(quantum_tasks, args.num_workers, process_quantum_observable, "observable")
        all_results["quantum_2d"] = {
            result["observable"]: {
                name: str(Path(path).relative_to(args.output_dir))
                for name, path in result["plots"].items()
            }
            for result in quantum_results
        }
        all_results["quantum_2d_entries"] = {
            result["observable"]: result["entries"]
            for result in quantum_results
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"[monitor-input] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
