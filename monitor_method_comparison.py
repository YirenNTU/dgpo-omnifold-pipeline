#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import json
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


def debug_print(debug: bool, *items: Any) -> None:
    if debug:
        print("[debug]", *items, flush=True)


def warn_print(*items: Any) -> None:
    print("[warning]", *items, flush=True)


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
        return np.asarray(
            ak.to_numpy(events["classification_target_name"], allow_missing=False),
            dtype=str,
        )

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


def read_parquet_chunk(
    file_name: str,
    columns: list[str],
    row_groups: list[int] | None,
    debug: bool = False,
) -> ak.Array | None:
    present = available_columns(file_name)
    requested = list(dict.fromkeys(columns))
    selected = requested if present is None else [column for column in requested if column in present]
    missing = [] if present is None else [column for column in requested if column not in present]

    debug_print(debug, "read file =", file_name)
    debug_print(debug, "row_groups =", row_groups)
    debug_print(debug, "requested columns =", len(requested))
    debug_print(debug, "selected columns =", len(selected))
    if missing:
        debug_print(debug, "missing first 40 columns =", missing[:40])

    if not selected:
        debug_print(debug, "skip chunk: no selected columns")
        return None

    if row_groups is not None:
        try:
            events = ak.from_parquet(file_name, columns=selected, row_groups=row_groups)
            debug_print(debug, "loaded rows =", len(events), "fields =", len(events.fields))
            return events
        except TypeError:
            pass

    events = ak.from_parquet(file_name, columns=selected)
    debug_print(debug, "loaded rows =", len(events), "fields =", len(events.fields))
    return events


def event_weights(events: ak.Array, weight_column: str | None) -> np.ndarray | None:
    if not weight_column or weight_column not in events.fields:
        return None

    weights = to_numpy(events[weight_column], np.float64).reshape(-1)
    if len(weights) != len(events):
        warn_print(f"weight column length mismatch: {weight_column} has {len(weights)}, events has {len(events)}")
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


def scalar_p4_columns(prefixes: tuple[str, ...]) -> list[str]:
    columns: list[str] = []
    for component in ("px", "py", "pz", "E"):
        columns.extend(p4_field_candidates(prefixes, component))
    return columns


def first_existing_field(events: ak.Array, candidates: list[str]) -> str | None:
    fields = set(events.fields)
    for candidate in candidates:
        if candidate in fields:
            return candidate
    return None


def component_array(
    events: ak.Array,
    candidates: list[str],
    slot: int | None = None,
    debug: bool = False,
) -> np.ndarray | None:
    field = first_existing_field(events, candidates)
    if field is None:
        debug_print(debug, "missing component candidates =", candidates)
        return None

    values = to_numpy(events[field], np.float64)

    if slot is not None and values.ndim == 2 and values.shape[1] > slot:
        values = values[:, slot]

    debug_print(debug, "component field =", field, "shape =", values.shape)
    return values


def p4_from_component_prefixes(
    events: ak.Array,
    prefixes: tuple[str, ...],
    slot: int | None = None,
    debug: bool = False,
) -> ak.Array | None:
    components = {
        "px": component_array(events, p4_field_candidates(prefixes, "px"), slot, debug),
        "py": component_array(events, p4_field_candidates(prefixes, "py"), slot, debug),
        "pz": component_array(events, p4_field_candidates(prefixes, "pz"), slot, debug),
        "E": component_array(events, p4_field_candidates(prefixes, "E"), slot, debug),
    }

    if any(value is None for value in components.values()):
        debug_print(debug, "failed to build p4 for prefixes =", prefixes)
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


def vector_pt_eta_phi(p4: ak.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.asarray(ak.to_numpy(p4.pt, allow_missing=False), dtype=np.float64)
    eta = np.asarray(ak.to_numpy(p4.eta, allow_missing=False), dtype=np.float64)
    phi = np.asarray(ak.to_numpy(p4.phi, allow_missing=False), dtype=np.float64)
    return pt, eta, phi


def method_columns(
    observable_names: list[str],
    weight_column: str | None,
    methods: list[str],
    reference_method: str,
) -> list[str]:
    requested_methods = set(methods) | {reference_method}
    columns: list[str] = []

    columns.extend(["classification_target_name", "event_category"])

    if weight_column:
        columns.append(weight_column)

    # Visible legs.
    columns.extend(scalar_p4_columns(("lead_a_visible", "visible_a", "visible")))
    columns.extend(scalar_p4_columns(("lead_b_visible", "visible_b", "visible")))

    if "target" in requested_methods:
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

    if "truth" in requested_methods:
        columns.extend(scalar_p4_columns(("truth_tau_a",)))
        columns.extend(scalar_p4_columns(("truth_tau_b",)))
        columns.extend(scalar_p4_columns(("truth_a_visible", "truth_visible_a")))
        columns.extend(scalar_p4_columns(("truth_b_visible", "truth_visible_b")))
        for observable in observable_names:
            columns.append(f"truth_{observable}")

    if "baseline" in requested_methods:
        for observable in observable_names:
            columns.append(f"baseline_{observable}")
        columns.extend(["baseline_flags_valid", "flags_valid", "baseline_mmc_likelihood", "mmc_likelihood"])

    if "evenet" in requested_methods:
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
            out[name] = to_numpy(events[field], np.float64).reshape(-1)

    return out


def build_target_observables(events: ak.Array, debug: bool = False) -> dict[str, np.ndarray] | None:
    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0, debug=debug)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1, debug=debug)

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
        debug=debug,
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
        debug=debug,
    )

    if visible_a is None or visible_b is None or target_a is None or target_b is None:
        debug_print(debug, "target observables unavailable")
        return None

    observables = build_observables(
        tau_a_p4=visible_a + target_a,
        tau_b_p4=visible_b + target_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )

    out = {name: np.asarray(values, dtype=np.float64).reshape(-1) for name, values in observables.items()}
    debug_print(debug, "target observables built =", len(out))
    return out


def build_truth_observables(
    events: ak.Array,
    observable_names: list[str],
    debug: bool = False,
) -> dict[str, np.ndarray] | None:
    tau_a = p4_from_component_prefixes(events, ("truth_tau_a",), debug=debug)
    tau_b = p4_from_component_prefixes(events, ("truth_tau_b",), debug=debug)
    vis_a = p4_from_component_prefixes(events, ("truth_a_visible", "truth_visible_a"), debug=debug)
    vis_b = p4_from_component_prefixes(events, ("truth_b_visible", "truth_visible_b"), debug=debug)

    if tau_a is not None and tau_b is not None and vis_a is not None and vis_b is not None:
        observables = build_observables(
            tau_a_p4=tau_a,
            tau_b_p4=tau_b,
            vis_a_p4=vis_a,
            vis_b_p4=vis_b,
        )
        out = {name: np.asarray(values, dtype=np.float64).reshape(-1) for name, values in observables.items()}
        debug_print(debug, "truth observables rebuilt from p4 =", len(out))
        return out

    stored = stored_observables(events, "truth", observable_names)
    debug_print(debug, "truth stored observables =", len(stored))
    return stored if stored else None


def evenet_invisible_p4(events: ak.Array, visible: ak.Array, leg: str, debug: bool = False) -> ak.Array | None:
    valid_field = f"evenet_invisible_{leg}_valid"
    eta_field = f"evenet_invisible_{leg}_eta"
    phi_field = f"evenet_invisible_{leg}_phi"
    pt_field = f"evenet_invisible_{leg}_pt"
    log_pt_field = f"evenet_invisible_{leg}_log_pt"

    if eta_field not in events.fields or phi_field not in events.fields:
        debug_print(debug, f"evenet leg {leg}: missing eta/phi fields")
        return None

    visible_pt, visible_eta, visible_phi = vector_pt_eta_phi(visible)

    delta_eta = to_numpy(events[eta_field], np.float64).reshape(-1)
    delta_phi = to_numpy(events[phi_field], np.float64).reshape(-1)

    pred_eta = visible_eta + delta_eta
    pred_phi = wrap_phi(visible_phi + delta_phi)

    if pt_field in events.fields:
        delta_pt = to_numpy(events[pt_field], np.float64).reshape(-1)
        pred_pt = visible_pt + delta_pt
        debug_print(
            debug,
            f"evenet leg {leg}: using delta pt",
            "visible_pt min/max =", float(np.nanmin(visible_pt)), float(np.nanmax(visible_pt)),
            "delta_pt min/max =", float(np.nanmin(delta_pt)), float(np.nanmax(delta_pt)),
            "raw pred_pt min/max =", float(np.nanmin(pred_pt)), float(np.nanmax(pred_pt)),
        )
    elif log_pt_field in events.fields:
        delta_log_pt = to_numpy(events[log_pt_field], np.float64).reshape(-1)
        pred_pt = np.expm1(np.log1p(np.clip(visible_pt, 0.0, None)) + delta_log_pt)
        debug_print(
            debug,
            f"evenet leg {leg}: using delta log_pt",
            "raw pred_pt min/max =", float(np.nanmin(pred_pt)), float(np.nanmax(pred_pt)),
        )
    else:
        debug_print(debug, f"evenet leg {leg}: missing pt/log_pt")
        return None

    if valid_field in events.fields:
        valid = to_numpy(events[valid_field], bool).reshape(-1)
    else:
        valid = np.ones(len(events), dtype=bool)

    if len(valid) != len(pred_pt):
        raise ValueError(f"{valid_field} length {len(valid)} != pred_pt length {len(pred_pt)}")

    if np.sum(pred_pt > 0) == 0:
        warn_print(
            f"all EveNet reconstructed pt <= 0 before clipping for leg {leg};",
            "this usually means evenet_invisible_* is not a delta in normal pt space",
        )

    pred_pt = np.clip(pred_pt, 0.0, None)

    pred_pt = np.where(valid, pred_pt, 0.0)
    pred_eta = np.where(valid, pred_eta, 0.0)
    pred_phi = np.where(valid, pred_phi, 0.0)

    debug_print(
        debug,
        f"evenet leg {leg}: valid =",
        int(np.sum(valid)),
        "/",
        len(valid),
        "final pred_pt min/max =",
        float(np.nanmin(pred_pt)),
        float(np.nanmax(pred_pt)),
    )

    return massless_p4_from_pt_eta_phi(pred_pt, pred_eta, pred_phi)


def build_evenet_observables(events: ak.Array, debug: bool = False) -> dict[str, np.ndarray] | None:
    visible_a = p4_from_component_prefixes(events, ("lead_a_visible", "visible_a", "visible"), slot=0, debug=debug)
    visible_b = p4_from_component_prefixes(events, ("lead_b_visible", "visible_b", "visible"), slot=1, debug=debug)

    if visible_a is None or visible_b is None:
        debug_print(debug, "evenet observables unavailable: missing visible p4")
        return None

    invisible_a = evenet_invisible_p4(events, visible_a, "a", debug=debug)
    invisible_b = evenet_invisible_p4(events, visible_b, "b", debug=debug)

    if invisible_a is None or invisible_b is None:
        debug_print(debug, "evenet observables unavailable: missing invisible p4")
        return None

    observables = build_observables(
        tau_a_p4=visible_a + invisible_a,
        tau_b_p4=visible_b + invisible_b,
        vis_a_p4=visible_a,
        vis_b_p4=visible_b,
    )

    out = {name: np.asarray(values, dtype=np.float64).reshape(-1) for name, values in observables.items()}
    debug_print(debug, "evenet observables built =", len(out))
    return out


def baseline_valid_mask(events: ak.Array, debug: bool = False) -> np.ndarray:
    if "baseline_flags_valid" in events.fields:
        valid = to_numpy(events["baseline_flags_valid"], bool).reshape(-1)
        debug_print(debug, "baseline valid from baseline_flags_valid =", int(np.sum(valid)), "/", len(valid))
        return valid

    if "flags_valid" in events.fields:
        valid = to_numpy(events["flags_valid"], bool).reshape(-1)
        debug_print(debug, "baseline valid from flags_valid =", int(np.sum(valid)), "/", len(valid))
        return valid

    if "baseline_mmc_likelihood" in events.fields:
        likelihood = to_numpy(events["baseline_mmc_likelihood"], np.float64).reshape(-1)
        valid = np.isfinite(likelihood) & (likelihood > 0.0)
        debug_print(debug, "baseline valid from baseline_mmc_likelihood =", int(np.sum(valid)), "/", len(valid))
        return valid

    if "mmc_likelihood" in events.fields:
        likelihood = to_numpy(events["mmc_likelihood"], np.float64).reshape(-1)
        valid = np.isfinite(likelihood) & (likelihood > 0.0)
        debug_print(debug, "baseline valid from mmc_likelihood =", int(np.sum(valid)), "/", len(valid))
        return valid

    valid = np.zeros(len(events), dtype=bool)
    debug_print(debug, "baseline valid unavailable; all false")
    return valid


def evenet_valid_mask(events: ak.Array, debug: bool = False) -> np.ndarray:
    valid = np.ones(len(events), dtype=bool)

    for leg in ("a", "b"):
        field = f"evenet_invisible_{leg}_valid"
        if field in events.fields:
            leg_valid = to_numpy(events[field], bool).reshape(-1)
            valid &= leg_valid
            debug_print(debug, field, "valid =", int(np.sum(leg_valid)), "/", len(leg_valid))

    debug_print(debug, "evenet combined valid =", int(np.sum(valid)), "/", len(valid))
    return valid


def method_valid_masks(events: ak.Array, debug: bool = False) -> dict[str, np.ndarray]:
    return {
        "target": np.ones(len(events), dtype=bool),
        "truth": np.ones(len(events), dtype=bool),
        "baseline": baseline_valid_mask(events, debug),
        "evenet": evenet_valid_mask(events, debug),
    }


def build_method_observables(
    events: ak.Array,
    observable_names: list[str],
    debug: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    methods: dict[str, dict[str, np.ndarray]] = {}

    target = build_target_observables(events, debug)
    if target is not None:
        methods["target"] = target

    truth = build_truth_observables(events, observable_names, debug)
    if truth is not None:
        methods["truth"] = truth

    baseline = stored_observables(events, "baseline", observable_names)
    if baseline:
        methods["baseline"] = baseline
        debug_print(debug, "baseline stored observables =", len(baseline))
    else:
        debug_print(debug, "baseline stored observables unavailable")

    evenet = build_evenet_observables(events, debug)
    if evenet is not None:
        methods["evenet"] = evenet

    debug_print(debug, "built methods =", sorted(methods))
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
    debug = bool(payload.get("debug", False))

    low, high = QUANTUM_RANGES.get(observable, (-1.0, 1.0))
    bins = np.linspace(low, high, payload["bins_2d"] + 1)

    comparisons = make_comparison_tasks(
        payload["comparison_methods"],
        reference=payload["reference_method"],
    )

    states: dict[str, dict[str, dict[str, Any]]] = {}

    columns = method_columns(
        observable_names=observable_names,
        weight_column=payload["weight_column"],
        methods=payload["comparison_methods"],
        reference_method=payload["reference_method"],
    )

    sample_files = payload["sample_files"]

    allowed_labels = set(payload["neutrino_prediction_labels"])
    if payload.get("no_label_filter", False):
        allowed_labels = set()

    print(f"[method-monitor] observable={observable}", flush=True)
    debug_print(debug, "comparison tasks =", comparisons)
    debug_print(debug, "allowed labels =", sorted(allowed_labels) if allowed_labels else "<disabled>")
    debug_print(debug, "sample files =", {name: len(files) for name, files in sample_files.items()})
    debug_print(debug, "requested columns =", len(columns))

    for sample_name, files in sample_files.items():
        remaining = payload["max_events"]

        for file_name in files:
            debug_print(debug, "start file", sample_name, file_name)

            for row_groups in row_group_chunks(file_name, payload["row_groups_per_chunk"]):
                if remaining is not None and remaining <= 0:
                    break

                events = read_parquet_chunk(file_name, columns, row_groups, debug=debug)
                if events is None:
                    debug_print(debug, "skip: events is None")
                    continue

                if remaining is not None and len(events) > remaining:
                    events = events[:remaining]
                    debug_print(debug, "trimmed to remaining =", remaining)

                labels = raw_event_process_labels(events, sample_name, payload["label_lookup"])
                unique_labels, unique_counts = np.unique(labels, return_counts=True)
                debug_print(debug, "labels =", dict(zip(unique_labels.tolist(), unique_counts.tolist())))

                if allowed_labels:
                    keep = np.asarray([label in allowed_labels for label in labels], dtype=bool)
                else:
                    keep = np.ones(len(events), dtype=bool)

                debug_print(debug, "keep =", int(np.sum(keep)), "/", len(keep))

                if not np.any(keep):
                    if remaining is not None:
                        remaining -= int(len(events))
                    del events, labels, keep
                    gc.collect()
                    continue

                weights = event_weights(events, payload["weight_column"])
                if weights is None:
                    debug_print(debug, "weights = None")
                else:
                    debug_print(
                        debug,
                        "weights len/min/max/sum =",
                        len(weights),
                        float(np.nanmin(weights)),
                        float(np.nanmax(weights)),
                        float(np.nansum(weights)),
                    )

                method_values = build_method_observables(events, observable_names, debug=debug)
                valid_masks = method_valid_masks(events, debug=debug)

                for method_name, obs_dict in method_values.items():
                    debug_print(
                        debug,
                        method_name,
                        "n_observables =",
                        len(obs_dict),
                        "has current =",
                        observable in obs_dict,
                    )

                for comparison_name, x_method, y_method in comparisons:
                    debug_print(debug, "comparison =", comparison_name, x_method, "vs", y_method)

                    if x_method not in method_values:
                        debug_print(debug, "skip: missing x method", x_method)
                        continue

                    if y_method not in method_values:
                        debug_print(debug, "skip: missing y method", y_method)
                        continue

                    if observable not in method_values[x_method]:
                        debug_print(debug, "skip: missing observable for x", x_method, observable)
                        continue

                    if observable not in method_values[y_method]:
                        debug_print(debug, "skip: missing observable for y", y_method, observable)
                        continue

                    x_all = method_values[x_method][observable]
                    y_all = method_values[y_method][observable]

                    if len(x_all) != len(events) or len(y_all) != len(events):
                        raise ValueError(
                            f"{observable} length mismatch: "
                            f"{x_method} len={len(x_all)}, {y_method} len={len(y_all)}, events len={len(events)}"
                        )

                    base_valid = keep.copy()
                    base_valid &= valid_masks.get(x_method, np.ones(len(events), dtype=bool))
                    base_valid &= valid_masks.get(y_method, np.ones(len(events), dtype=bool))

                    common = finite_common_mask(x_all, y_all, weights, base_valid)

                    debug_print(
                        debug,
                        "common count =",
                        comparison_name,
                        int(np.sum(common)),
                        "/",
                        len(common),
                    )

                    if not np.any(common):
                        continue

                    for process_name in np.unique(labels[common]):
                        process_mask = common & (labels == process_name)

                        if not np.any(process_mask):
                            continue

                        debug_print(
                            debug,
                            "process entries =",
                            comparison_name,
                            process_name,
                            int(np.sum(process_mask)),
                        )

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

    debug_print(debug, "states comparisons =", list(states))
    for comparison_name, process_states in states.items():
        debug_print(debug, comparison_name, "processes =", list(process_states))
        for process_name, state in process_states.items():
            nx = sum(len(chunk) for chunk in state["x"])
            ny = sum(len(chunk) for chunk in state["y"])
            nw = sum(len(chunk) for chunk in state["weight"])
            debug_print(
                debug,
                comparison_name,
                process_name,
                "x/y/w/total =",
                nx,
                ny,
                nw,
                state["total"],
            )

    results: dict[str, dict[str, Any]] = {}
    output_dir = Path(payload["output_dir"])

    for comparison_name, x_method, y_method in comparisons:
        process_states = states.get(comparison_name, {})
        x_label = METHOD_LABELS.get(x_method, x_method)
        y_label = METHOD_LABELS.get(y_method, y_method)

        if not process_states:
            debug_print(debug, "no process states for", comparison_name)

        for process_name, state in process_states.items():
            if not state["x"] or not state["y"]:
                debug_print(debug, "skip plotting empty state", comparison_name, process_name)
                continue

            x_values = np.concatenate(state["x"])
            y_values = np.concatenate(state["y"])
            weights = np.concatenate(state["weight"]) if state["weight"] else None

            extra_predictions = {
                extra_label: np.concatenate(extra_chunks)
                for extra_label, extra_chunks in state["extra"].items()
                if extra_chunks
            }

            debug_print(
                debug,
                "plotting",
                comparison_name,
                process_name,
                "entries =",
                len(x_values),
                "weights =",
                None if weights is None else len(weights),
                "extras =",
                list(extra_predictions),
            )

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

            debug_print(debug, "plot result =", process_result)

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
            result = future.result()
            print(f"[method-monitor] finished {observable}", flush=True)
            results.append(result)

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
        help="Overlay other methods in summary panels when available.",
    )
    parser.add_argument(
        "--combine-inputs",
        action="store_true",
        help="Combine all input dirs/files into one sample group. Default keeps one group per path.",
    )
    parser.add_argument(
        "--no-label-filter",
        action="store_true",
        help="Do not filter to NeutrinoPrediction labels. Useful for debugging no-output problems.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics for files, columns, labels, masks, methods, and entries.",
    )
    parser.add_argument(
        "--observables",
        nargs="+",
        default=None,
        help="Optional subset of quantum observables to run.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    _feature_config = parse_feature_config(config)

    sample_files = build_sample_map(args.input_dir, "samples", combine=args.combine_inputs)

    observable_names = list(get_observable_names())
    if args.observables is not None:
        requested = set(args.observables)
        observable_names = [name for name in observable_names if name in requested]
        missing = sorted(requested - set(observable_names))
        if missing:
            raise ValueError(f"Requested observables are not known: {missing}")

    print("[method-monitor] input samples:", flush=True)
    for sample_name, files in sample_files.items():
        print(f"  {sample_name}: {len(files)} file(s)", flush=True)
        if args.debug:
            for file_name in files[:20]:
                print(f"    {file_name}", flush=True)

    print(
        f"[method-monitor] observables={len(observable_names)} workers={args.num_workers} "
        f"reference={args.reference_method} comparisons={args.comparison_methods}",
        flush=True,
    )
    print(
        f"[method-monitor] output_dir={args.output_dir} "
        f"weight_column={args.weight_column} "
        f"max_events={args.max_events} "
        f"row_groups_per_chunk={args.row_groups_per_chunk}",
        flush=True,
    )

    pred_labels = sorted(neutrino_prediction_labels(config))
    print(f"[method-monitor] neutrino_prediction_labels={pred_labels}", flush=True)
    if args.no_label_filter:
        print("[method-monitor] label filter disabled", flush=True)

    common_payload = {
        "sample_files": sample_files,
        "max_events": args.max_events,
        "bins_2d": args.bins_2d,
        "weight_column": args.weight_column or None,
        "row_groups_per_chunk": args.row_groups_per_chunk,
        "label_lookup": category_lookup_payload(config),
        "neutrino_prediction_labels": pred_labels,
        "comparison_methods": list(args.comparison_methods),
        "reference_method": str(args.reference_method),
        "include_extra": bool(args.include_extra),
        "no_label_filter": bool(args.no_label_filter),
        "debug": bool(args.debug),
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

    results = run_tasks(tasks, args.num_workers)

    summary = {
        "input_samples": {name: len(files) for name, files in sample_files.items()},
        "max_events": args.max_events,
        "weight_column": args.weight_column,
        "reference_method": args.reference_method,
        "comparison_methods": args.comparison_methods,
        "include_extra": args.include_extra,
        "no_label_filter": args.no_label_filter,
        "neutrino_prediction_labels": pred_labels,
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