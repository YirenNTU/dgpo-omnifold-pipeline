#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
import math
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import vector

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from common import (
    build_classification_lookup,
    channel_latex_label,
    process_latex_label,
)
from generate_event_info_yaml import parse_feature_config
from plot_style import plot_truth_prediction_bundle
from quantum.observables_builder import build_observables, get_observable_names

vector.register_awkward()

DEFAULT_SAMPLE_LIMIT = 200_000

METHOD_LABELS = {
    "target": "Target",
    "truth": "Stored truth",
    "baseline": "Baseline",
    "evenet": "EveNet",
}

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


def to_numpy(array: Any, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(ak.to_numpy(array, allow_missing=False), dtype=dtype)
    except TypeError:
        return np.asarray(ak.to_numpy(array), dtype=dtype)
    except Exception:
        return np.asarray(ak.to_list(array), dtype=dtype)


def wrap_phi(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi))


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


def display_process_label(label: str) -> str:
    if label in {"Data", "data", "data94"}:
        return "Data"
    if label.startswith("Ztautau_"):
        return channel_latex_label(label)
    return process_latex_label(label)


def category_lookup_payload(config: dict[str, Any]) -> dict[str, Any]:
    lookup = build_classification_lookup(config)
    return {
        "sample_default_label": lookup.sample_default_label,
        "sample_event_category_to_label": lookup.sample_event_category_to_label,
    }


def neutrino_prediction_labels(config: dict[str, Any]) -> set[str]:
    prediction_cfg = config.get("NeutrinoPrediction") or {}
    labels: set[str] = set()
    for raw_value in prediction_cfg.values():
        if isinstance(raw_value, dict):
            for nested_value in raw_value.values():
                if isinstance(nested_value, list):
                    labels.update(str(item) for item in nested_value)
                elif nested_value is not None:
                    labels.add(str(nested_value))
        elif isinstance(raw_value, list):
            labels.update(str(item) for item in raw_value)
        elif raw_value is not None:
            labels.add(str(raw_value))
    return labels


def raw_event_process_labels(
    events: ak.Array,
    sample_name: str,
    label_lookup: dict[str, Any],
) -> np.ndarray:
    if "classification_target_name" in events.fields:
        return np.asarray(ak.to_numpy(events["classification_target_name"], allow_missing=False), dtype=str)

    category_map = label_lookup.get("sample_event_category_to_label", {}).get(sample_name)
    if category_map and "event_category" in events.fields:
        categories = to_numpy(events["event_category"], np.int64)
        labels = [category_map.get(int(category), "") for category in categories]
        return np.asarray(labels, dtype=str)

    default_label = label_lookup.get("sample_default_label", {}).get(sample_name, sample_name)
    return np.asarray([default_label] * len(events), dtype=str)


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


def event_weights(events: ak.Array, weight_column: str | None) -> np.ndarray | None:
    if not weight_column or weight_column not in events.fields:
        return None
    weights = to_numpy(events[weight_column], np.float64).reshape(-1)
    if len(weights) != len(events):
        return None
    return weights


def p4_field_candidates(prefixes: tuple[str, ...], component: str) -> list[str]:
    candidates: list[str] = []

    for prefix in prefixes:
        candidates.append(f"{prefix}_{component}")

        if component == "E":
            candidates.append(f"{prefix}_energy")
            candidates.append(f"{prefix}_Energy")
            candidates.append(f"{prefix}_t")

    return candidates


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


def massless_p4_from_pt_eta_phi(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> ak.Array:
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = wrap_phi(np.asarray(phi, dtype=np.float64))

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.clip(px * px + py * py + pz * pz, 0.0, None))

    return ak.zip(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": energy,
        },
        with_name="Momentum4D",
    )


def scalar_p4_columns(prefixes: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for component in ("px", "py", "pz", "E", "energy", "t"):
        columns.extend(p4_field_candidates(prefixes, component))
    return columns


def method_columns(observable: str, weight_column: str | None) -> list[str]:
    columns: list[str] = []

    columns.extend(["classification_target_name", "event_category"])

    if weight_column:
        columns.append(weight_column)

    # Visible legs.
    columns.extend(
        scalar_p4_columns(
            (
                "lead_a_visible",
                "visible_a",
                "visible",
            )
        )
    )
    columns.extend(
        scalar_p4_columns(
            (
                "lead_b_visible",
                "visible_b",
                "visible",
            )
        )
    )

    # Target invisibles.
    columns.extend(
        scalar_p4_columns(
            (
                "target_a_invisible",
                "target_a_missing",
                "target_missing_a",
                "lead_a_missing",
                "target_missing",
                "target_invisible",
            )
        )
    )
    columns.extend(
        scalar_p4_columns(
            (
                "target_b_invisible",
                "target_b_missing",
                "target_missing_b",
                "lead_b_missing",
                "target_missing",
                "target_invisible",
            )
        )
    )

    # Truth p4, if available.
    columns.extend(scalar_p4_columns(("truth_tau_a",)))
    columns.extend(scalar_p4_columns(("truth_tau_b",)))
    columns.extend(scalar_p4_columns(("truth_a_visible", "truth_visible_a")))
    columns.extend(scalar_p4_columns(("truth_b_visible", "truth_visible_b")))

    # Stored truth/baseline observables.
    columns.append(f"truth_{observable}")
    columns.append(f"baseline_{observable}")
    columns.extend(["baseline_flags_valid", "flags_valid", "baseline_mmc_likelihood", "mmc_likelihood"])

    # EveNet delta invisibles.
    for leg in ("a", "b"):
        columns.append(f"evenet_invisible_{leg}_valid")
        for feature in ("pt", "log_pt", "eta", "phi"):
            columns.append(f"evenet_invisible_{leg}_{feature}")

    return list(dict.fromkeys(columns))


def stored_observables(events: ak.Array, source: str, observable_names: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    for name in observable_names:
        if source == "truth":
            field = f"truth_{name}"
        elif source == "baseline":
            field = f"baseline_{name}"
        else:
            field = f"{source}_{name}"

        if field in events.fields:
            out[name] = to_numpy(events[field], np.float64)

    return out


def build_target_observables(events: ak.Array) -> dict[str, np.ndarray] | None:
    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1)

    target_a = p4_from_component_prefixes(
        events,
        (
            "target_a_invisible",
            "target_a_missing",
            "target_missing_a",
            "lead_a_missing",
            "target_missing",
            "target_invisible",
        ),
        slot=0,
    )
    target_b = p4_from_component_prefixes(
        events,
        (
            "target_b_invisible",
            "target_b_missing",
            "target_missing_b",
            "lead_b_missing",
            "target_missing",
            "target_invisible",
        ),
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


def build_truth_observables(events: ak.Array, observable_names: list[str]) -> dict[str, np.ndarray] | None:
    tau_a = p4_from_component_prefixes(events, ("truth_tau_a",))
    tau_b = p4_from_component_prefixes(events, ("truth_tau_b",))
    vis_a = p4_from_component_prefixes(events, ("truth_a_visible", "truth_visible_a"))
    vis_b = p4_from_component_prefixes(events, ("truth_b_visible", "truth_visible_b"))

    if tau_a is not None and tau_b is not None and vis_a is not None and vis_b is not None:
        observables = build_observables(
            tau_a_p4=tau_a,
            tau_b_p4=tau_b,
            vis_a_p4=vis_a,
            vis_b_p4=vis_b,
        )
        return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}

    stored = stored_observables(events, "truth", observable_names)
    return stored if stored else None


def visible_pt_eta_phi(visible: ak.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.asarray(ak.to_numpy(visible.pt, allow_missing=False), dtype=np.float64)
    eta = np.asarray(ak.to_numpy(visible.eta, allow_missing=False), dtype=np.float64)
    phi = np.asarray(ak.to_numpy(visible.phi, allow_missing=False), dtype=np.float64)
    return pt, eta, phi


def evenet_invisible_p4(events: ak.Array, visible: ak.Array, leg: str) -> ak.Array | None:
    valid_field = f"evenet_invisible_{leg}_valid"
    eta_field = f"evenet_invisible_{leg}_eta"
    phi_field = f"evenet_invisible_{leg}_phi"
    pt_field = f"evenet_invisible_{leg}_pt"
    log_pt_field = f"evenet_invisible_{leg}_log_pt"

    if eta_field not in events.fields or phi_field not in events.fields:
        return None

    visible_pt, visible_eta, visible_phi = visible_pt_eta_phi(visible)

    delta_eta = to_numpy(events[eta_field], np.float64)
    delta_phi = to_numpy(events[phi_field], np.float64)

    pred_eta = visible_eta + delta_eta
    pred_phi = wrap_phi(visible_phi + delta_phi)

    if pt_field in events.fields:
        delta_pt = to_numpy(events[pt_field], np.float64)
        pred_pt = visible_pt + delta_pt
    elif log_pt_field in events.fields:
        delta_log_pt = to_numpy(events[log_pt_field], np.float64)
        pred_pt = np.expm1(np.log1p(np.clip(visible_pt, 0.0, None)) + delta_log_pt)
    else:
        return None

    pred_pt = np.clip(pred_pt, 0.0, None)

    if valid_field in events.fields:
        valid = to_numpy(events[valid_field], bool)
        pred_pt = np.where(valid, pred_pt, 0.0)
        pred_eta = np.where(valid, pred_eta, 0.0)
        pred_phi = np.where(valid, pred_phi, 0.0)

    return massless_p4_from_pt_eta_phi(pred_pt, pred_eta, pred_phi)


def build_evenet_observables(events: ak.Array) -> dict[str, np.ndarray] | None:
    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1)

    if visible_a is None or visible_b is None:
        return None

    invisible_a = evenet_invisible_p4(events, visible_a, "a")
    invisible_b = evenet_invisible_p4(events, visible_b, "b")

    if invisible_a is None or invisible_b is None:
        return None

    observables = build_observables(
        tau_a_p4=visible_a + invisible_a,
        tau_b_p4=visible_b + invisible_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )

    return {name: np.asarray(values, dtype=np.float64) for name, values in observables.items()}


def baseline_valid_mask(events: ak.Array) -> np.ndarray:
    if "baseline_flags_valid" in events.fields:
        return to_numpy(events["baseline_flags_valid"], bool).reshape(-1)

    if "flags_valid" in events.fields:
        return to_numpy(events["flags_valid"], bool).reshape(-1)

    if "baseline_mmc_likelihood" in events.fields:
        likelihood = to_numpy(events["baseline_mmc_likelihood"], np.float64).reshape(-1)
        return np.isfinite(likelihood) & (likelihood > 0.0)

    if "mmc_likelihood" in events.fields:
        likelihood = to_numpy(events["mmc_likelihood"], np.float64).reshape(-1)
        return np.isfinite(likelihood) & (likelihood > 0.0)

    return np.zeros(len(events), dtype=bool)


def evenet_valid_mask(events: ak.Array) -> np.ndarray:
    valid = np.ones(len(events), dtype=bool)

    for leg in ("a", "b"):
        field = f"evenet_invisible_{leg}_valid"
        if field in events.fields:
            valid &= to_numpy(events[field], bool).reshape(-1)

    return valid


def method_valid_masks(events: ak.Array) -> dict[str, np.ndarray]:
    return {
        "target": np.ones(len(events), dtype=bool),
        "truth": np.ones(len(events), dtype=bool),
        "baseline": baseline_valid_mask(events),
        "evenet": evenet_valid_mask(events),
    }


def build_method_observables(events: ak.Array, observable_names: list[str]) -> dict[str, dict[str, np.ndarray]]:
    methods: dict[str, dict[str, np.ndarray]] = {}

    target = build_target_observables(events)
    if target is not None:
        methods["target"] = target

    truth = build_truth_observables(events, observable_names)
    if truth is not None:
        methods["truth"] = truth

    baseline = stored_observables(events, "baseline", observable_names)
    if baseline:
        methods["baseline"] = baseline

    evenet = build_evenet_observables(events)
    if evenet is not None:
        methods["evenet"] = evenet

    return methods


def finite_common_mask(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray | None,
    base_mask: np.ndarray,
) -> np.ndarray:
    mask = base_mask & np.isfinite(x) & np.isfinite(y)

    if weight is not None:
        mask &= np.isfinite(weight)

    return mask


def make_comparison_tasks(methods: list[str], reference: str = "target") -> list[tuple[str, str, str]]:
    tasks = []
    for method in methods:
        if method == reference:
            continue
        tasks.append((f"{reference}_vs_{method}", reference, method))
    return tasks


def process_quantum_observable(payload: dict[str, Any]) -> dict[str, Any]:
    observable = payload["observable"]
    observable_names = payload["observable_names"]
    low, high = QUANTUM_RANGES.get(observable, (-1.0, 1.0))
    bins = np.linspace(low, high, payload["bins_2d"] + 1)

    comparisons = make_comparison_tasks(
        payload["comparison_methods"],
        reference=payload["reference_method"],
    )

    states: dict[str, dict[str, dict[str, Any]]] = {}
    columns = method_columns(observable, payload["weight_column"])
    sample_files = payload["sample_files"]
    allowed_labels = set(payload["neutrino_prediction_labels"])

    for sample_name, files in sample_files.items():
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

                labels = raw_event_process_labels(events, sample_name, payload["label_lookup"])

                if allowed_labels:
                    keep = np.asarray([label in allowed_labels for label in labels], dtype=bool)
                else:
                    keep = np.ones(len(events), dtype=bool)

                if not np.any(keep):
                    if remaining is not None:
                        remaining -= int(len(events))
                    del events, labels, keep
                    gc.collect()
                    continue

                weights = event_weights(events, payload["weight_column"])
                method_values = build_method_observables(events, observable_names)
                valid_masks = method_valid_masks(events)

                for comparison_name, x_method, y_method in comparisons:
                    if x_method not in method_values or y_method not in method_values:
                        continue
                    if observable not in method_values[x_method] or observable not in method_values[y_method]:
                        continue

                    x_all = method_values[x_method][observable]
                    y_all = method_values[y_method][observable]

                    base_valid = keep & valid_masks.get(x_method, np.ones(len(events), dtype=bool))
                    base_valid &= valid_masks.get(y_method, np.ones(len(events), dtype=bool))

                    common = finite_common_mask(x_all, y_all, weights, base_valid)

                    if not np.any(common):
                        continue

                    for process_name in np.unique(labels[common]):
                        process_mask = common & (labels == process_name)
                        if not np.any(process_mask):
                            continue

                        state = states.setdefault(comparison_name, {}).setdefault(
                            str(process_name),
                            {
                                "x": [],
                                "y": [],
                                "weight": [],
                                "extra": {},
                                "total": 0,
                            },
                        )

                        state["x"].append(x_all[process_mask])
                        state["y"].append(y_all[process_mask])
                        state["total"] += int(np.sum(process_mask))

                        if weights is not None:
                            state["weight"].append(weights[process_mask])

                        if payload["include_extra"]:
                            for extra_method, extra_observables in method_values.items():
                                if extra_method in {x_method, y_method}:
                                    continue
                                if observable not in extra_observables:
                                    continue

                                extra_values = extra_observables[observable][process_mask].copy()
                                extra_valid = valid_masks.get(extra_method, np.ones(len(events), dtype=bool))[process_mask]
                                extra_values = np.where(extra_valid & np.isfinite(extra_values), extra_values, np.nan)

                                label = METHOD_LABELS.get(extra_method, extra_method)
                                state["extra"].setdefault(label, []).append(extra_values)

                if remaining is not None:
                    remaining -= int(len(events))

                del events, weights, labels, keep, method_values, valid_masks
                gc.collect()

            if remaining is not None and remaining <= 0:
                break

    results: dict[str, dict[str, Any]] = {}
    output_dir = Path(payload["output_dir"])

    for comparison_name, x_method, y_method in comparisons:
        process_states = states.get(comparison_name, {})
        x_label = METHOD_LABELS.get(x_method, x_method)
        y_label = METHOD_LABELS.get(y_method, y_method)

        for process_name, state in process_states.items():
            if not state["x"] or not state["y"]:
                continue

            x_values = np.concatenate(state["x"])
            y_values = np.concatenate(state["y"])
            weights = np.concatenate(state["weight"]) if state["weight"] else None

            extra_predictions = {
                extra_label: np.concatenate(extra_chunks)
                for extra_label, extra_chunks in state["extra"].items()
                if extra_chunks
            }

            process_result = plot_truth_prediction_bundle(
                x_values,
                y_values,
                output_dir / comparison_name / process_name,
                observable,
                bins=bins,
                weight=weights,
                truth_label=x_label,
                pred_label=y_label,
                xaxis_label=observable,
                title=f"{comparison_name}: {observable}",
                summary_title=f"{display_process_label(process_name)} summary",
                total_entries=state["total"],
                extra_predictions=extra_predictions,
            )

            results.setdefault(comparison_name, {})[process_name] = process_result

    return {
        "observable": observable,
        "plots": {
            comparison_name: {
                process_name: result["plots"]
                for process_name, result in process_results.items()
            }
            for comparison_name, process_results in results.items()
        },
        "metrics": {
            comparison_name: {
                process_name: result["metrics"]
                for process_name, result in process_results.items()
            }
            for comparison_name, process_results in results.items()
        },
    }


def run_tasks(tasks: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            print(f"[method-monitor] plotting {task['observable']}", flush=True)
            results.append(process_quantum_observable(task))
        return results

    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
        future_map = {executor.submit(process_quantum_observable, task): task["observable"] for task in tasks}

        for future in as_completed(future_map):
            observable = future_map[future]
            print(f"[method-monitor] finished {observable}", flush=True)
            results.append(future.result())

    return results


def main() -> None:
    parser = argparse.ArgumentParser("Monitor EveNet method-comparison parquet files")
    parser.add_argument("--input-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("method_comparison_plots"))
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--bins-2d", type=int, default=70)
    parser.add_argument("--weight-column", default="event_weight")
    parser.add_argument("--row-groups-per-chunk", type=int, default=1)
    parser.add_argument(
        "--comparison-methods",
        nargs="+",
        default=["truth", "baseline", "evenet"],
        help="Methods to compare against --reference-method.",
    )
    parser.add_argument(
        "--reference-method",
        default="target",
        help="Reference method for comparison plots.",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Overlay other methods in 1D summary panels when available.",
    )
    parser.add_argument(
        "--combine-inputs",
        action="store_true",
        help="Combine all input dirs/files into one sample group. Default keeps one group per path.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    _feature_config = parse_feature_config(config)

    sample_files = build_sample_map(args.input_dir, "samples", combine=args.combine_inputs)
    observable_names = list(get_observable_names())

    common_payload = {
        "sample_files": sample_files,
        "max_events": args.max_events,
        "bins_2d": args.bins_2d,
        "weight_column": args.weight_column or None,
        "row_groups_per_chunk": args.row_groups_per_chunk,
        "label_lookup": category_lookup_payload(config),
        "neutrino_prediction_labels": sorted(neutrino_prediction_labels(config)),
        "comparison_methods": list(args.comparison_methods),
        "reference_method": str(args.reference_method),
        "include_extra": bool(args.include_extra),
        "observable_names": observable_names,
        "output_dir": str(args.output_dir / "quantum_method_comparison"),
    }

    tasks = [
        {
            **common_payload,
            "observable": observable,
        }
        for observable in observable_names
    ]

    print(
        f"[method-monitor] observables={len(tasks)} workers={args.num_workers} "
        f"reference={args.reference_method} comparisons={args.comparison_methods}",
        flush=True,
    )

    results = run_tasks(tasks, args.num_workers)

    summary = {
        "input_samples": {name: len(files) for name, files in sample_files.items()},
        "max_events": args.max_events,
        "weight_column": args.weight_column,
        "reference_method": args.reference_method,
        "comparison_methods": args.comparison_methods,
        "include_extra": args.include_extra,
        "neutrino_prediction_labels": sorted(neutrino_prediction_labels(config)),
        "quantum_method_comparison": {
            result["observable"]: {
                comparison: {
                    process_name: {
                        plot_name: str(Path(plot_path).relative_to(args.output_dir))
                        for plot_name, plot_path in process_plots.items()
                    }
                    for process_name, process_plots in comparison_plots.items()
                }
                for comparison, comparison_plots in result["plots"].items()
            }
            for result in results
        },
        "quantum_metrics": {
            result["observable"]: result["metrics"]
            for result in results
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[method-monitor] wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()