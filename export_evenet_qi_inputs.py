#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import glob
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import vector
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))

from ml_pipeline.build_evenet_input_from_parquet import read_file_initial_total_num_events
from ml_pipeline.common import (
    build_classification_lookup,
    event_preselection_mask,
    post_calibrate_tau_tau,
    read_yaml,
    rebuild_vector,
    to_numpy,
)
from quantum.observables_builder import build_observables, get_observable_names

vector.register_awkward()

TAU_MASS_GEV = 1.777
CM_ENERGY_GEV = 91.2
MAX_VISIBLE_ENERGY_GEV = 91.25
METHOD_CHOICES = ("target", "baseline", "evenet")
DEFAULT_METHODS = ("target", "evenet", "baseline")
SAMPLE_ORDER = ("data94", "Zqq", "Zll", "Ztautau")


@dataclass(frozen=True)
class RawWeightInfo:
    is_data: bool
    weight_scale: float
    total_initial_num_events: float | None
    weight_source: str


def export_observable_names() -> tuple[str, ...]:
    return tuple(get_observable_names())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EveNet prediction parquets as nominal QI unfolding inputs."
    )
    parser.add_argument("--analysis-config", type=Path, default=Path("ml_pipeline/config/analysis.yaml"))
    parser.add_argument("--prediction-parquet", nargs="+", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=METHOD_CHOICES)
    parser.add_argument("--regions", nargs="+", default=None, help="Defaults to Ztautau labels listed in NeutrinoPrediction.")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--pseudo-data", action="store_true")
    parser.add_argument("--compression", default="snappy")
    return parser.parse_args()


def resolve_parquets(paths: list[Path]) -> list[Path]:
    output: list[Path] = []
    for path in paths:
        text = str(path.expanduser())
        matches = sorted(glob.glob(text))
        candidates = [Path(match) for match in matches] if matches else [Path(text)]
        for candidate in candidates:
            if candidate.is_dir():
                output.extend(sorted(candidate.glob("*__evenet_pred.*.parquet")))
            else:
                output.append(candidate)
    return [path.resolve() for path in output]


def sample_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    samples = config.get("Samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError("Analysis config is missing non-empty Samples.")
    return dict(samples)


def required_sample_value(sample_key: str, sample_cfg: dict[str, Any], field: str) -> Any:
    if field not in sample_cfg or sample_cfg[field] is None:
        raise ValueError(f"Sample '{sample_key}' is missing required field '{field}'.")
    return sample_cfg[field]


def sample_name(sample_key: str, sample_cfg: dict[str, Any]) -> str:
    return str(required_sample_value(sample_key, sample_cfg, "name"))


def sample_is_data(sample_key: str, sample_cfg: dict[str, Any]) -> bool:
    return bool(required_sample_value(sample_key, sample_cfg, "is_data"))


def sample_is_signal(sample_key: str, sample_cfg: dict[str, Any]) -> bool:
    return bool(required_sample_value(sample_key, sample_cfg, "is_signal"))


def sample_raw_files(sample_key: str, sample_cfg: dict[str, Any]) -> list[Path]:
    raw_files = sample_cfg.get("raw_files") or sample_cfg.get("raw_input_files")
    if not raw_files:
        raise ValueError(f"Sample '{sample_name(sample_key, sample_cfg)}' is missing raw_files/raw_input_files.")

    output: list[Path] = []
    for raw_file in raw_files:
        matches = sorted(glob.glob(str(raw_file)))
        output.extend(Path(match).expanduser().resolve() for match in (matches or [str(raw_file)]))
    return output


def analysis_luminosity(config: dict[str, Any]) -> float | None:
    total = 0.0
    found = False
    for sample_key, sample_cfg in sample_configs(config).items():
        if sample_is_data(sample_key, sample_cfg) and sample_cfg.get("lumi") is not None:
            total += float(sample_cfg["lumi"])
            found = True
    return total if found else None


def qi_luminosity(config: dict[str, Any]) -> float:
    luminosity = analysis_luminosity(config)
    if luminosity is None:
        raise ValueError("QI config needs GlobalConfigs.luminosity or at least one data sample lumi.")
    return luminosity


def raw_weight_info(
    config: dict[str, Any],
    sample_key: str,
    sample_cfg: dict[str, Any],
    raw_files: list[Path],
) -> RawWeightInfo:
    if sample_is_data(sample_key, sample_cfg):
        return RawWeightInfo(
            is_data=True,
            weight_scale=1.0,
            total_initial_num_events=None,
            weight_source="data_unit",
        )

    norm_factor = float(required_sample_value(sample_key, sample_cfg, "norm_factor"))
    luminosity = qi_luminosity(config)
    total_initial_num_events = 0.0
    for raw_path in raw_files:
        file_total = read_file_initial_total_num_events(str(raw_path))
        if file_total is None:
            raise ValueError(
                f"MC sample '{sample_key}' raw file '{raw_path}' is missing initial_total_num_events."
            )
        total_initial_num_events += float(file_total)
    if total_initial_num_events <= 0.0:
        raise ValueError(
            f"MC sample '{sample_key}' has non-positive summed initial_total_num_events={total_initial_num_events}."
        )

    return RawWeightInfo(
        is_data=False,
        weight_scale=luminosity * norm_factor / total_initial_num_events,
        total_initial_num_events=total_initial_num_events,
        weight_source="nominal_lumi_times_norm_over_summed_raw_initial_events",
    )


def build_raw_weight_info(config: dict[str, Any], samples: dict[str, dict[str, Any]]) -> dict[str, RawWeightInfo]:
    return {
        sample_key: raw_weight_info(config, sample_key, sample_cfg, sample_raw_files(sample_key, sample_cfg))
        for sample_key, sample_cfg in samples.items()
    }


def class_regions(config: dict[str, Any]) -> list[str]:
    lookup = build_classification_lookup(config)
    return [label for label in lookup.class_labels if label.startswith("Ztautau_")]


def neutrino_prediction_regions(config: dict[str, Any]) -> list[str]:
    prediction_cfg = config.get("NeutrinoPrediction") or {}
    regions: list[str] = []
    for value in prediction_cfg.values():
        if isinstance(value, dict):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            values = [value]

        for item in values:
            if isinstance(item, list):
                regions.extend(str(label) for label in item)
            elif item is not None:
                regions.append(str(item))

    seen: set[str] = set()
    return [region for region in regions if region.startswith("Ztautau_") and not (region in seen or seen.add(region))]


def signal_categories(config: dict[str, Any], regions: list[str]) -> dict[str, list[int]]:
    subcategories = config.get("Subcategories")
    if not isinstance(subcategories, dict) or "Ztautau" not in subcategories:
        raise ValueError("Analysis config is missing Subcategories.Ztautau.")
    categories = subcategories["Ztautau"]
    return {
        region: [int(value) for value in categories[region]]
        for region in regions
        if region.startswith("Ztautau_") and region in categories
    }


def rebuild_vectors(events: ak.Array) -> ak.Array:
    output = events
    for field in output.fields:
        if field.endswith("_p4"):
            output[field] = rebuild_vector(output[field])
    return output


def p4_from_fields(events: ak.Array, prefix: str) -> ak.Array:
    if f"{prefix}_p4" in events.fields:
        return rebuild_vector(events[f"{prefix}_p4"])
    return vector.zip(
        {
            "px": events[f"{prefix}_px"],
            "py": events[f"{prefix}_py"],
            "pz": events[f"{prefix}_pz"],
            "E": events[f"{prefix}_E"],
        }
    )


def finite(values: Any) -> np.ndarray:
    return np.isfinite(to_numpy(values, np.float64))


def finite_p4(p4: ak.Array) -> np.ndarray:
    return finite(p4.px) & finite(p4.py) & finite(p4.pz) & finite(p4.E)



def preselection_mask(events: ak.Array) -> np.ndarray:
    mask = np.ones(len(events), dtype=bool)
    fields = set(events.fields)
    if "baseline_cut" in fields:
        mask &= to_numpy(events["baseline_cut"], bool)
    mask &= event_preselection_mask(events)
    return mask

def prediction_weight(sample_key: str, sample_cfg: dict[str, Any], events: ak.Array) -> np.ndarray:
    if sample_is_data(sample_key, sample_cfg):
        return np.ones(len(events), dtype=np.float32)
    if "event_weight" not in events.fields:
        raise ValueError(
            f"Prediction parquet for MC sample '{sample_key}' is missing event_weight. "
            "Do not fall back to an inferred normalization."
        )
    return to_numpy(events["event_weight"], np.float32)


def raw_weight(info: RawWeightInfo, num_events: int) -> np.ndarray:
    return np.full(num_events, info.weight_scale, dtype=np.float32)


def class_labels(config: dict[str, Any]) -> tuple[str, ...]:
    return build_classification_lookup(config).class_labels


def label_from_index(indices: np.ndarray, labels: tuple[str, ...]) -> np.ndarray:
    output = np.full(len(indices), "", dtype=object)
    valid = (indices >= 0) & (indices < len(labels))
    if not np.all(valid):
        bad_values = sorted(set(int(index) for index in indices[~valid]))
        raise ValueError(f"EveNet class index outside configured labels: {bad_values}")
    for index, label in enumerate(labels):
        output[valid & (indices == index)] = label
    return output


def target_labels(events: ak.Array, sample_key: str, config: dict[str, Any]) -> np.ndarray:
    if "classification_target_name" in events.fields:
        return np.asarray(ak.to_list(events["classification_target_name"]), dtype=object)
    if sample_key != "Ztautau":
        return np.full(len(events), sample_key, dtype=object)
    categories = to_numpy(events["event_category"], np.int64)
    lookup = build_classification_lookup(config)
    if "Ztautau" not in lookup.sample_event_category_to_label:
        raise ValueError("Classification lookup is missing Ztautau category labels.")
    mapping = lookup.sample_event_category_to_label["Ztautau"]
    missing_categories = sorted({int(category) for category in categories if int(category) not in mapping})
    if missing_categories:
        raise ValueError(f"Ztautau event categories are not covered by Subcategories: {missing_categories}")
    return np.asarray([mapping[int(category)] for category in categories], dtype=object)


def evenet_labels(events: ak.Array, labels: tuple[str, ...]) -> np.ndarray:
    if "evenet_pred_class_name" in events.fields:
        return np.asarray(ak.to_list(events["evenet_pred_class_name"]), dtype=object)
    if "evenet_class_index" in events.fields:
        return label_from_index(to_numpy(events["evenet_class_index"], np.int64), labels)
    raise ValueError("EveNet region export requires evenet_pred_class_name or evenet_class_index.")


def target_tau_pair(events: ak.Array) -> tuple[ak.Array, ak.Array, np.ndarray]:
    vis_a = p4_from_fields(events, "lead_a_visible")
    vis_b = p4_from_fields(events, "lead_b_visible")
    tau_a = p4_from_fields(events, "target_a_invisible")
    tau_b = p4_from_fields(events, "target_b_invisible")
    valid = finite_p4(vis_a) & finite_p4(vis_b) & finite_p4(tau_a) & finite_p4(tau_b)
    return tau_a, tau_b, valid


def baseline_tau_pair(events: ak.Array) -> tuple[ak.Array | None, ak.Array | None, np.ndarray | None]:
    if "baseline_flags_valid" in events.fields:
        valid = to_numpy(events["baseline_flags_valid"], bool)
    elif "flags_valid" in events.fields:
        valid = to_numpy(events["flags_valid"], bool)
    else:
        valid = None
    return None, None, valid


def evenet_tau_pair(events: ak.Array) -> tuple[ak.Array, ak.Array, np.ndarray]:
    vis_a = p4_from_fields(events, "lead_a_visible")
    vis_b = p4_from_fields(events, "lead_b_visible")
    theta_a = vis_a.theta + events["evenet_invisible_a_theta"]
    theta_b = vis_b.theta + events["evenet_invisible_b_theta"]
    phi_a = (vis_a.phi + events["evenet_invisible_a_phi"] + math.pi) % (2 * math.pi) - math.pi
    phi_b = (vis_b.phi + events["evenet_invisible_b_phi"] + math.pi) % (2 * math.pi) - math.pi

    energy = CM_ENERGY_GEV / 2
    momentum = math.sqrt(energy * energy - TAU_MASS_GEV * TAU_MASS_GEV)
    tau_a = ak.zip(
        {"pt": momentum * np.sin(theta_a), "theta": theta_a, "phi": phi_a, "m": ak.ones_like(theta_a) * TAU_MASS_GEV},
        with_name="Momentum4D",
    )
    tau_b = ak.zip(
        {"pt": momentum * np.sin(theta_b), "theta": theta_b, "phi": phi_b, "m": ak.ones_like(theta_b) * TAU_MASS_GEV},
        with_name="Momentum4D",
    )
    tau_a, tau_b = post_calibrate_tau_tau(tau_a, tau_b)

    valid = to_numpy(events["evenet_invisible_a_valid"], bool) & to_numpy(events["evenet_invisible_b_valid"], bool)
    valid &= finite_p4(vis_a) & finite_p4(vis_b) & finite_p4(tau_a) & finite_p4(tau_b)
    return tau_a, tau_b, valid


def method_observables(events: ak.Array, method: str) -> tuple[dict[str, Any], np.ndarray]:
    names = export_observable_names()
    if method == "target":
        if all(f"truth_{name}" in events.fields or name == "mtautau" for name in names):
            output = {
                name: observable_field_values(events, name, (f"truth_{name}",))
                for name in names
            }
            valid = np.ones(len(events), dtype=bool)
            for values in output.values():
                valid &= finite(values)
            return output, valid
        required_prefixes = (
            "lead_a_visible",
            "lead_b_visible",
            "target_a_invisible",
            "target_b_invisible",
        )
        missing = [
            prefix
            for prefix in required_prefixes
            if f"{prefix}_p4" not in events.fields
            and not all(f"{prefix}_{component}" in events.fields for component in ("px", "py", "pz", "E"))
        ]
        if missing:
            missing_truth = [f"truth_{name}" for name in names if f"truth_{name}" not in events.fields]
            raise KeyError(
                "Target method needs either all truth observable fields or target four-vector fields. "
                f"Missing truth observables: {missing_truth}. Missing four-vector prefixes: {missing}."
            )
    if method == "baseline":
        output = {
            name: observable_field_values(events, name, (f"baseline_{name}", name))
            for name in names
        }
        _, _, valid = baseline_tau_pair(events)
        if valid is None:
            valid = np.ones(len(events), dtype=bool)
        for values in output.values():
            valid &= finite(values)
        return output, valid

    vis_a = p4_from_fields(events, "lead_a_visible")
    vis_b = p4_from_fields(events, "lead_b_visible")
    if method == "target":
        tau_a, tau_b, valid = target_tau_pair(events)
    elif method == "evenet":
        tau_a, tau_b, valid = evenet_tau_pair(events)
    else:
        raise ValueError(f"Unknown method {method}")

    built_observables = build_observables(tau_a, tau_b, vis_a, vis_b)
    output = {name: built_observables[name] for name in names}
    valid &= np.ones(len(events), dtype=bool)
    for values in output.values():
        valid &= finite(values)
    return {name: ak.where(valid, values, np.nan) for name, values in output.items()}, valid


def observable_field_values(events: ak.Array, observable_name: str, candidates: tuple[str, ...]) -> Any:
    for candidate in candidates:
        if candidate in events.fields:
            return events[candidate]
    if observable_name == "mtautau":
        return fixed_mtautau_values(len(events))
    raise KeyError(
        f"Missing observable '{observable_name}'. Expected one of {list(candidates)}. "
        f"Available matching fields: {matching_observable_fields(events, observable_name)}"
    )


def fixed_mtautau_values(num_events: int) -> np.ndarray:
    return np.full(num_events, CM_ENERGY_GEV, dtype=np.float32)


def matching_observable_fields(events: ak.Array, observable_name: str) -> list[str]:
    return sorted(field for field in events.fields if field == observable_name or field.endswith(f"_{observable_name}"))


def base_fields(
    events: ak.Array,
    sample_key: str,
    config: dict[str, Any],
    weights: np.ndarray,
    total_initial_num_events: float | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in [
        "event_category",
        "truth_QI_region",
        "analyzing_power_a",
        "analyzing_power_b",
        "analyzing_power",
        "initial_total_num_events",
        "nprong",
    ]:
        if name in events.fields:
            fields[name] = events[name]
    for name in events.fields:
        if name.startswith("truth_") and name not in fields:
            fields[name] = events[name]
        if name.endswith("_cut") and name not in fields:
            fields[name] = events[name]

    if weights.shape != (len(events),):
        raise ValueError(f"Weight shape {weights.shape} does not match {len(events)} events for sample '{sample_key}'.")
    fields["weight"] = weights
    fields["weight_nominal"] = weights
    if "initial_total_num_events" not in fields and "initial_num_events" in events.fields:
        fields["initial_total_num_events"] = events["initial_num_events"]
    if total_initial_num_events is not None:
        fields["initial_total_num_events"] = np.full(
            len(events),
            total_initial_num_events,
            dtype=np.float64,
        )
    if "classification_target_name" in events.fields:
        fields["classification_target_name"] = events["classification_target_name"]
    else:
        fields["classification_target_name"] = ak.Array(target_labels(events, sample_key, config).tolist())
    return fields


def region_masks(events: ak.Array, method: str, sample_key: str, regions: list[str], config: dict[str, Any]) -> dict[str, np.ndarray]:
    truth_label = target_labels(events, sample_key, config)
    pred_label = evenet_labels(events, class_labels(config)) if method == "evenet" else None
    output: dict[str, np.ndarray] = {}
    categories_cfg = signal_categories(config, regions)
    categories = to_numpy(events["event_category"], np.int64) if "event_category" in events.fields else None

    for region in regions:
        cut = f"{region}_cut"
        if region.startswith("Ztautau_") and method == "evenet":
            if pred_label is None:
                raise ValueError("Internal error: missing EveNet labels for EveNet region masks.")
            output[region] = pred_label == region
        elif cut in events.fields:
            output[region] = to_numpy(events[cut], bool)
        elif region.startswith("Ztautau_") and region in categories_cfg and categories is not None:
            output[region] = np.isin(categories, categories_cfg[region])
        elif region.startswith("Ztautau_"):
            output[region] = truth_label == region
        else:
            raise ValueError(f"Unsupported QI region '{region}'.")
    return output


def export_method_events(
    events: ak.Array,
    method: str,
    sample_key: str,
    sample_cfg: dict[str, Any],
    config: dict[str, Any],
    regions: list[str],
) -> ak.Array:
    weights = prediction_weight(sample_key, sample_cfg, events)
    output = base_fields(events, sample_key, config, weights)
    observables, valid = method_observables(events, method)
    output.update(observables)
    output["flags_valid"] = valid
    output["mmc_likelihood"] = auxiliary_field(events, method, "mmc_likelihood")

    masks = region_masks(events, method, sample_key, regions, config)
    for region, mask in masks.items():
        output[f"{region}_cut"] = mask
    return ak.Array(output)


def auxiliary_field(events: ak.Array, method: str, name: str) -> ak.Array:
    if name == "mmc_likelihood" and method in {"target", "evenet"}:
        return np.zeros(len(events), dtype=np.float32)
    candidates = [name]
    if method == "baseline":
        candidates.append(f"baseline_{name}")
    for candidate in candidates:
        if candidate in events.fields:
            return events[candidate]
    raise KeyError(f"Missing required auxiliary field '{name}' for method '{method}'.")


def invalid_float_values(num_events: int) -> np.ndarray:
    return np.full(num_events, np.nan, dtype=np.float32)


def invalid_bool_values(num_events: int) -> np.ndarray:
    return np.zeros(num_events, dtype=bool)


def invalid_likelihood_values(num_events: int) -> np.ndarray:
    return np.zeros(num_events, dtype=np.float32)


def raw_observable_values(raw_events: ak.Array, sample_key: str, name: str, require_field: bool) -> Any:
    if name in raw_events.fields:
        return raw_events[name]
    baseline_name = f"baseline_{name}"
    if baseline_name in raw_events.fields:
        return raw_events[baseline_name]
    if name == "mtautau":
        return fixed_mtautau_values(len(raw_events))
    if require_field:
        raise KeyError(f"RAW sample '{sample_key}' is missing observable '{name}' or '{baseline_name}'.")
    return invalid_float_values(len(raw_events))


def raw_auxiliary_values(raw_events: ak.Array, sample_key: str, name: str, require_field: bool) -> Any:
    if name in raw_events.fields:
        return raw_events[name]
    if require_field:
        raise KeyError(f"RAW sample '{sample_key}' is missing {name}.")
    if name == "flags_valid":
        return invalid_bool_values(len(raw_events))
    if name == "mmc_likelihood":
        return invalid_likelihood_values(len(raw_events))
    raise ValueError(f"Unsupported RAW auxiliary field '{name}'.")


def raw_region_cut_values(
    raw_events: ak.Array,
    region: str,
) -> np.ndarray:
    if not region.startswith("Ztautau_"):
        raise ValueError(f"Unsupported QI region '{region}'.")
    return invalid_bool_values(len(raw_events))


def export_raw_complement(
    raw_events: ak.Array,
    sample_key: str,
    sample_cfg: dict[str, Any],
    config: dict[str, Any],
    weight_info: RawWeightInfo,
    regions: list[str],
) -> ak.Array:
    keep = ~preselection_mask(raw_events)
    raw_events = raw_events[keep]
    require_qi_fields = sample_is_signal(sample_key, sample_cfg)
    weights = raw_weight(weight_info, len(raw_events))
    output = base_fields(
        raw_events,
        sample_key,
        config,
        weights,
        total_initial_num_events=weight_info.total_initial_num_events,
    )
    for name in export_observable_names():
        output[name] = raw_observable_values(raw_events, sample_key, name, require_qi_fields)
    output["flags_valid"] = raw_auxiliary_values(raw_events, sample_key, "flags_valid", require_qi_fields)
    output["mmc_likelihood"] = raw_auxiliary_values(raw_events, sample_key, "mmc_likelihood", require_qi_fields)
    for region in regions:
        output[f"{region}_cut"] = raw_region_cut_values(raw_events, region)
    return ak.Array(output)


def write_cutflow(sample_dir: Path, sample_name: str, events: ak.Array) -> None:
    weight_sum = float(np.sum(to_numpy(events["weight"], np.float64))) if len(events) else 0.0
    record = {
        "step": 0,
        "cut": "initial_total_num_events",
        "events": int(len(events)),
        "weighted_events": weight_sum,
        "efficiency": 1.0,
        "weighted_efficiency": 1.0,
        "relative_efficiency": 1.0,
        "weighted_relative_efficiency": 1.0,
    }
    (sample_dir / f"cutflow_{sample_name}.json").write_text(json.dumps([record], indent=2))


def numeric_field(events: ak.Array, name: str, dtype: Any, fill_value: float | int | bool) -> np.ndarray:
    if name not in events.fields:
        return np.full(len(events), fill_value, dtype=dtype)
    values = ak.fill_none(events[name], fill_value)
    return to_numpy(values, dtype)


def string_field(events: ak.Array, name: str, fill_value: str) -> ak.Array:
    if name not in events.fields:
        return ak.Array([fill_value] * len(events))
    values = ak.fill_none(events[name], fill_value)
    return ak.Array([str(value) for value in ak.to_list(values)])


def final_qi_events(events: ak.Array, sample_name: str, regions: list[str]) -> ak.Array:
    fields: dict[str, Any] = {
        "event_category": numeric_field(events, "event_category", np.int32, 0),
        "truth_QI_region": numeric_field(events, "truth_QI_region", bool, False),
        "analyzing_power_a": numeric_field(events, "analyzing_power_a", np.float32, np.nan),
        "analyzing_power_b": numeric_field(events, "analyzing_power_b", np.float32, np.nan),
        "analyzing_power": numeric_field(events, "analyzing_power", np.float32, np.nan),
        "initial_total_num_events": numeric_field(events, "initial_total_num_events", np.float64, 0.0),
        "nprong": numeric_field(events, "nprong", np.int32, 0),
        "baseline_cut": numeric_field(events, "baseline_cut", bool, False),
        "weight": numeric_field(events, "weight", np.float32, 0.0),
        "weight_nominal": numeric_field(events, "weight_nominal", np.float32, 0.0),
        "classification_target_name": string_field(events, "classification_target_name", sample_name),
    }

    for name in export_observable_names():
        fill_value = CM_ENERGY_GEV if name == "mtautau" else np.nan
        fields[name] = numeric_field(events, name, np.float32, fill_value)
        fields[f"truth_{name}"] = numeric_field(events, f"truth_{name}", np.float32, fill_value)

    fields["flags_valid"] = numeric_field(events, "flags_valid", bool, False)
    fields["mmc_likelihood"] = numeric_field(events, "mmc_likelihood", np.float32, 0.0)
    for region in regions:
        fields[f"{region}_cut"] = numeric_field(events, f"{region}_cut", bool, False)
    return ak.Array(fields)


def write_tree(events: ak.Array, sample_dir: Path, sample_name: str, regions: list[str], compression: str) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    events = final_qi_events(events, sample_name, regions)
    ak.to_parquet(events, sample_dir / "filtered___raw.parquet", compression=compression)
    for region in regions:
        cut = f"{region}_cut"
        if cut not in events.fields:
            raise KeyError(f"Cannot write region '{region}' because field '{cut}' is missing.")
        mask = to_numpy(events[cut], bool)
        ak.to_parquet(events[mask], sample_dir / f"filtered___{region}.parquet", compression=compression)
    write_cutflow(sample_dir, sample_name, events)


def write_fragment_tree(events: ak.Array, fragment_dir: Path, sample_name: str, regions: list[str], compression: str) -> None:
    write_tree(events, fragment_dir, sample_name, regions, compression)


def clean_method_outputs(output_root: Path, methods: list[str]) -> None:
    for method in methods:
        method_dir = output_root / method
        for child_name in ("processed", "_fragments"):
            child = method_dir / child_name
            if child.exists():
                shutil.rmtree(child)
        method_dir.mkdir(parents=True, exist_ok=True)


def sample_fragment_dirs(fragment_root: Path, sample_name: str) -> list[Path]:
    if not fragment_root.exists():
        return []
    output: list[tuple[int, Path]] = []
    for path in fragment_root.iterdir():
        if not path.is_dir():
            continue
        if path.name == sample_name:
            output.append((-1, path))
            continue
        prefix = f"{sample_name}_"
        if not path.name.startswith(prefix):
            continue
        suffix = path.name.removeprefix(prefix)
        if suffix.isdigit():
            output.append((int(suffix), path))
    return [path for _, path in sorted(output)]


def read_parquet_list(paths: list[Path]) -> ak.Array:
    if not paths:
        raise ValueError("Cannot merge an empty parquet list.")
    arrays = [ak.from_parquet(path) for path in paths]
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def merge_sample_fragments(
    method_dir: Path,
    sample_name: str,
    regions: list[str],
    compression: str,
) -> dict[str, int]:
    fragment_dirs = sample_fragment_dirs(method_dir / "_fragments", sample_name)
    if not fragment_dirs:
        return {}

    sample_dir = method_dir / "processed" / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for region in ["raw", *regions]:
        files = [
            fragment_dir / f"filtered___{region}.parquet"
            for fragment_dir in fragment_dirs
            if (fragment_dir / f"filtered___{region}.parquet").exists()
        ]
        if not files:
            continue
        events = read_parquet_list(files)
        ak.to_parquet(events, sample_dir / f"filtered___{region}.parquet", compression=compression)
        counts[region] = int(len(events))
        if region == "raw":
            write_cutflow(sample_dir, sample_name, events)
    return counts


def merge_fragment_outputs(
    output_root: Path,
    methods: list[str],
    samples: dict[str, dict[str, Any]],
    regions: list[str],
    compression: str,
) -> dict[str, dict[str, dict[str, int]]]:
    sample_names = list(samples)
    if "data94" not in sample_names:
        sample_names.append("data94")
    merged: dict[str, dict[str, dict[str, int]]] = {}
    for method in methods:
        method_dir = output_root / method
        method_counts: dict[str, dict[str, int]] = {}
        for sample_name_value in sample_names:
            counts = merge_sample_fragments(method_dir, sample_name_value, regions, compression)
            if counts:
                method_counts[sample_name_value] = counts
        fragment_root = method_dir / "_fragments"
        if fragment_root.exists():
            shutil.rmtree(fragment_root)
        merged[method] = method_counts
    return merged


def parquet_schema_names(path: Path) -> set[str]:
    return {field.name for field in pq.ParquetFile(path).schema_arrow}


def existing_columns(path: Path, requested: set[str]) -> list[str]:
    schema_names = parquet_schema_names(path)
    return sorted(column for column in requested if column in schema_names)


def common_export_columns(regions: list[str]) -> set[str]:
    columns = {
        "sample_key",
        "source_sample_index",
        "event_weight",
        "central_weight",
        "classification_target_name",
        "event_category",
        "truth_QI_region",
        "analyzing_power_a",
        "analyzing_power_b",
        "analyzing_power",
        "initial_total_num_events",
        "initial_num_events",
        "nprong",
        "baseline_cut",
        "flags_valid",
        "baseline_flags_valid",
        "mmc_likelihood",
        "baseline_mmc_likelihood",
    }
    columns.update(f"{region}_cut" for region in regions)
    return columns


def p4_columns(prefix: str) -> set[str]:
    return {
        f"{prefix}_p4",
        f"{prefix}_px",
        f"{prefix}_py",
        f"{prefix}_pz",
        f"{prefix}_E",
    }


def prediction_columns(methods: list[str], regions: list[str]) -> set[str]:
    columns = common_export_columns(regions)
    names = export_observable_names()
    if "target" in methods:
        columns.update(f"truth_{name}" for name in names)
        columns.update(p4_columns("lead_a_visible"))
        columns.update(p4_columns("lead_b_visible"))
        columns.update(p4_columns("target_a_invisible"))
        columns.update(p4_columns("target_b_invisible"))
    if "baseline" in methods:
        columns.update(names)
        columns.update(f"baseline_{name}" for name in names)
    if "evenet" in methods:
        columns.update(p4_columns("lead_a_visible"))
        columns.update(p4_columns("lead_b_visible"))
        columns.update(
            {
                "evenet_pred_class_name",
                "evenet_class_index",
                "evenet_invisible_a_theta",
                "evenet_invisible_b_theta",
                "evenet_invisible_a_phi",
                "evenet_invisible_b_phi",
                "evenet_invisible_a_valid",
                "evenet_invisible_b_valid",
            }
        )
    return columns


def raw_columns(regions: list[str]) -> set[str]:
    columns = common_export_columns(regions)
    names = export_observable_names()
    columns.update(names)
    columns.update(f"baseline_{name}" for name in names)
    return columns


def iter_batches(path: Path, batch_size: int, columns: set[str] | None = None):
    parquet = pq.ParquetFile(path)
    selected_columns = None if columns is None else existing_columns(path, columns)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected_columns):
        yield rebuild_vectors(ak.from_arrow(batch))


def fragment_name(sample: str, index: int) -> str:
    return sample if index == 0 else f"{sample}_{index:06d}"


def prediction_sample_keys(events: ak.Array, samples: dict[str, dict[str, Any]]) -> np.ndarray:
    if "sample_key" in events.fields:
        sample_keys = np.asarray(ak.to_list(events["sample_key"]), dtype=object)
    elif "source_sample_index" in events.fields:
        sample_order = tuple(samples.keys())
        source_indices = to_numpy(events["source_sample_index"], np.int64)
        valid = (source_indices >= 0) & (source_indices < len(sample_order))
        if not np.all(valid):
            bad_indices = sorted({int(index) for index in source_indices[~valid]})
            raise ValueError(
                "Prediction parquet source_sample_index contains values outside Samples order: "
                f"{bad_indices}."
            )
        sample_keys = np.asarray([sample_order[int(index)] for index in source_indices], dtype=object)
    else:
        available = ", ".join(events.fields[:40])
        suffix = " ..." if len(events.fields) > 40 else ""
        raise KeyError(
            "Prediction parquet needs sample_key or source_sample_index to identify sample membership. "
            f"Available fields include: {available}{suffix}"
        )

    unknown = sorted({str(key) for key in sample_keys if str(key) not in samples})
    if unknown:
        raise ValueError(f"Prediction parquet contains sample keys not present in analysis config: {unknown}")
    return sample_keys


def export_prediction_file(args: tuple[Any, ...]) -> dict[str, int]:
    pred_path, config, samples, methods, regions, output_root, batch_size, compression, pseudo_data, start_index = args
    counts: dict[str, int] = {}
    fragment_index = start_index
    for events in iter_batches(pred_path, batch_size, prediction_columns(methods, regions)):
        sample_keys = prediction_sample_keys(events, samples)
        for sample_key in sorted(set(sample_keys)):
            if sample_key not in samples:
                continue
            sample_cfg = samples[sample_key]
            if pseudo_data and sample_is_data(sample_key, sample_cfg):
                continue
            sample_events = events[sample_keys == sample_key]
            for method in methods:
                method_events = export_method_events(sample_events, method, sample_key, sample_cfg, config, regions)
                sample_dir = output_root / method / "_fragments" / fragment_name(sample_key, fragment_index)
                write_fragment_tree(method_events, sample_dir, sample_key, regions, compression)
                counts[f"{method}:{sample_key}"] = counts.get(f"{method}:{sample_key}", 0) + len(method_events)
                if pseudo_data and sample_key != "data94" and not sample_is_data(sample_key, sample_cfg):
                    pseudo_dir = output_root / method / "_fragments" / fragment_name("data94", fragment_index)
                    write_fragment_tree(method_events, pseudo_dir, "data94", regions, compression)
                    counts[f"{method}:data94"] = counts.get(f"{method}:data94", 0) + len(method_events)
            fragment_index += 1
    return counts


def export_raw_file(args: tuple[Any, ...]) -> dict[str, int]:
    raw_path, sample_key, sample_cfg, config, weight_info, methods, regions, output_root, batch_size, compression, pseudo_data, start_index = args
    counts: dict[str, int] = {}
    fragment_index = start_index
    for events in iter_batches(raw_path, batch_size, raw_columns(regions)):
        complement = export_raw_complement(events, sample_key, sample_cfg, config, weight_info, regions)
        if len(complement) == 0:
            continue
        for method in methods:
            sample_dir = output_root / method / "_fragments" / fragment_name(sample_key, fragment_index)
            write_fragment_tree(complement, sample_dir, sample_key, regions, compression)
            counts[f"{method}:{sample_key}"] = counts.get(f"{method}:{sample_key}", 0) + len(complement)
            if pseudo_data and sample_key != "data94" and not sample_is_data(sample_key, sample_cfg):
                pseudo_dir = output_root / method / "_fragments" / fragment_name("data94", fragment_index)
                write_fragment_tree(complement, pseudo_dir, "data94", regions, compression)
                counts[f"{method}:data94"] = counts.get(f"{method}:data94", 0) + len(complement)
        fragment_index += 1
    return counts


def merge_counts(items: list[dict[str, int]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            output[key] = output.get(key, 0) + value
    return output


def run_jobs(jobs: list[tuple[Any, ...]], fn, workers: int) -> dict[str, int]:
    if workers <= 1:
        return merge_counts([fn(job) for job in jobs])
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return merge_counts(results)


def write_qi_config(
    method_dir: Path,
    method: str,
    config: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    regions: list[str],
    pseudo_data: bool,
) -> Path:
    signal_cfg = signal_categories(config, regions)
    data_loaders = {}
    for sample_key in SAMPLE_ORDER:
        if sample_key not in samples and sample_key != "data94":
            continue
        if sample_key not in samples:
            raise ValueError(f"QI config needs sample '{sample_key}' in analysis config.")
        sample_cfg = samples[sample_key]
        data_loaders[sample_key] = {
            "name": sample_name(sample_key, sample_cfg),
            "is_data": sample_is_data(sample_key, sample_cfg),
            "is_signal": sample_is_signal(sample_key, sample_cfg),
        }
        if "norm_factor" in sample_cfg:
            data_loaders[sample_key]["norm_factor"] = float(sample_cfg["norm_factor"])

    qi_config = {
        "GlobalConfigs": {
            "default_output_dir": str(method_dir / "run"),
            "processed_data_dir": str(method_dir / "processed"),
            "load_regions": ["raw", *regions],
            "verbosity": 1,
            "luminosity": qi_luminosity(config),
            "signal_categories": signal_cfg,
        },
        "DataLoaders": data_loaders,
        "Processors": {
            "QIProcessor": {
                "processor_name": "QIProcessor",
                "output_dir_name": "QI_analysis",
                "asimov_data": not pseudo_data,
                "dict_region_to_signals": {region: [region] for region in regions if region in signal_cfg},
            }
        },
    }
    path = method_dir / f"config_{method}.yaml"
    path.write_text(yaml.safe_dump(qi_config, sort_keys=False))
    return path


def main() -> None:
    args = parse_args()
    config = read_yaml(args.analysis_config)
    samples = sample_configs(config)
    raw_weight_infos = build_raw_weight_info(config, samples)
    regions = args.regions or neutrino_prediction_regions(config)
    if not regions:
        regions = class_regions(config)
    output_root = args.base_dir
    output_root.mkdir(parents=True, exist_ok=True)
    clean_method_outputs(output_root, args.methods)

    prediction_paths = resolve_parquets(args.prediction_parquet)
    if not prediction_paths:
        raise FileNotFoundError("No prediction parquet files were found.")

    raw_jobs = []
    job_index = 1
    for sample_key, sample_cfg in samples.items():
        if args.pseudo_data and sample_is_data(sample_key, sample_cfg):
            continue
        for raw_path in sample_raw_files(sample_key, sample_cfg):
            raw_jobs.append((
                raw_path,
                sample_key,
                sample_cfg,
                config,
                raw_weight_infos[sample_key],
                args.methods,
                regions,
                output_root,
                args.batch_size,
                args.compression,
                args.pseudo_data,
                job_index,
            ))
            job_index += 100_000

    pred_jobs = []
    for pred_path in prediction_paths:
        pred_jobs.append((pred_path, config, samples, args.methods, regions, output_root, args.batch_size, args.compression, args.pseudo_data, job_index))
        job_index += 100_000

    raw_counts = run_jobs(raw_jobs, export_raw_file, args.num_workers)
    pred_counts = run_jobs(pred_jobs, export_prediction_file, args.num_workers)
    counts = merge_counts([raw_counts, pred_counts])
    merged_outputs = merge_fragment_outputs(output_root, args.methods, samples, regions, args.compression)

    config_paths = {}
    for method in args.methods:
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        config_paths[method] = str(write_qi_config(method_dir, method, config, samples, regions, args.pseudo_data))

    summary = {
        "prediction_files": [str(path) for path in prediction_paths],
        "methods": args.methods,
        "regions": regions,
        "counts": counts,
        "merged_outputs": merged_outputs,
        "configs": config_paths,
        "raw_weight_info": {sample_key: asdict(info) for sample_key, info in raw_weight_infos.items()},
        "pseudo_data": bool(args.pseudo_data),
    }
    (output_root / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
