#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import awkward as ak
import numpy as np
import yaml
from evenet_parquet_common import (
    FOUR_VECTOR_FEATURES,
    MAX_PART_ENERGY_GEV,
    build_central_leg_slot_indices,
    build_tau_targets,
    build_visible_tau_assumptions,
    build_momentum4d,
    extract_target_invisible_observable,
    extract_part_feature,
    extract_part_momentum_observable,
    extract_visible_tau_observable,
    features_from_p4,
    part_energy_mask,
)
from ml_pipeline_config import EveNetConfig, FeatureConfig, parse_evenet_config, parse_feature_config
from parquet_plot_common import (
    choose_bins,
    infer_luminosity,
    plot_from_histograms,
    sanitize_hist_values,
    sample_scale,
    summarize_invalid_hist_values,
)
from rich.console import Console
from rich.table import Table


console = Console()
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
GENERATED_EVENT_INFO_PATH = CONFIG_DIR / "generated_event_info.yaml"

PREDICTION_PASSTHROUGH_EXACT_FIELDS = {
    "event_category",
    "truth_QI_region",
    "analyzing_power_a",
    "analyzing_power_b",
    "analyzing_power",
    "initial_total_num_events",
}
PREDICTION_PASSTHROUGH_SUFFIXES = ("_cut",)
PREDICTION_PASSTHROUGH_PREFIXES = ("truth_",)
DEFAULT_FLOAT = -99.0
PHOTON_PDG_ID = 21
REQUIRED_MC_CONCAT_SOURCE_FIELDS = {
    "event_category",
    "truth_QI_region",
    "analyzing_power_a",
    "analyzing_power_b",
    "analyzing_power",
    "initial_total_num_events",
    "truth_theta_cm",
    "truth_mtautau",
}
REQUIRED_MC_CONCAT_SOURCE_FIELDS.update({f"truth_cos_theta_A_{axis}" for axis in ("n", "r", "k")})
REQUIRED_MC_CONCAT_SOURCE_FIELDS.update({f"truth_cos_theta_B_{axis}" for axis in ("n", "r", "k")})
REQUIRED_MC_CONCAT_SOURCE_FIELDS.update(
    {
        f"truth_cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}"
        for axis_a, axis_b in product(("n", "r", "k"), repeat=2)
    }
)

@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    is_data: bool
    is_signal: bool
    input_files: tuple[str, ...]
    norm_factor: float = 1.0
    lumi: float | None = None


@dataclass(frozen=True)
class CategorySplit:
    name: str
    categories: tuple[int, ...]


def read_yaml(path: Path) -> dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def merge_evenet_config(
    schema_config: dict,
    analysis_config: dict,
) -> dict:
    merged_config = dict(schema_config)

    normalization_cfg = analysis_config.get("Normalization")
    if normalization_cfg is not None:
        merged_config["Normalization"] = normalization_cfg
        return merged_config

    feature_tags_cfg = analysis_config.get("EveNet", {}).get("FeatureTags")
    if feature_tags_cfg is not None:
        merged_evenet = dict(merged_config.get("EveNet", {}))
        merged_evenet["FeatureTags"] = feature_tags_cfg
        merged_config["EveNet"] = merged_evenet

    return merged_config


def expand_input_files(patterns: tuple[str, ...]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            files.extend(matched)
        else:
            files.append(pattern)
    return files


def sample_uses_invisible_target(sample: Sample) -> bool:
    return sample.is_signal


def load_sample_events(sample: Sample) -> ak.Array:
    parquet_files = expand_input_files(sample.input_files)
    console.print(
        f"[bold cyan]Loading sample[/bold cyan] [white]{sample.name}[/white] "
        f"from [white]{len(parquet_files)}[/white] parquet file(s)"
    )
    arrays = [ak.from_parquet(path) for path in parquet_files]
    if not arrays:
        raise ValueError(f"No parquet inputs found for sample '{sample.name}'.")
    events = arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)
    console.print(f"  [green]Loaded[/green] [white]{len(events)}[/white] selected event(s)")
    return events


def apply_preselection(events: ak.Array) -> ak.Array:
    if "nprong" not in events.fields:
        return events

    nprong_selected = events[events["nprong"] == 2]
    console.print(
        f"  [bold cyan]Preselection[/bold cyan] nprong == 2 -> "
        f"[white]{len(nprong_selected)}[/white] / [white]{len(events)}[/white] event(s)"
    )

    tau_vis_prong_p4, tau_vis_prong_mask, _, _ = build_visible_tau_assumptions(nprong_selected)
    energy = ak.to_numpy(tau_vis_prong_p4.E, allow_missing=False)
    valid = ak.to_numpy(tau_vis_prong_mask, allow_missing=False).astype(bool)
    energy_mask = np.all(valid & np.isfinite(energy) & (energy < MAX_PART_ENERGY_GEV), axis=1)
    selected = nprong_selected[energy_mask]
    console.print(
        f"  [bold cyan]Preselection[/bold cyan] tau_vis_prong_energy < "
        f"{MAX_PART_ENERGY_GEV:g} GeV -> [white]{len(selected)}[/white] / "
        f"[white]{len(nprong_selected)}[/white] event(s)"
    )
    return selected


def parse_category_values(raw_values) -> tuple[int, ...]:
    if isinstance(raw_values, (list, tuple)):
        return tuple(int(value) for value in raw_values)
    return (int(raw_values),)


def parse_config(config_path: Path) -> tuple[dict[str, Sample], dict[str, list[CategorySplit]], FeatureConfig]:
    config = read_yaml(config_path)

    raw_samples = config.get("Samples", {})
    if not raw_samples:
        raise ValueError(f"No 'Samples' block found in {config_path}.")

    samples = {
        sample_key: Sample(
            key=sample_key,
            name=sample_cfg.get("name", sample_key),
            is_data=bool(sample_cfg.get("is_data", False)),
            is_signal=bool(sample_cfg.get("is_signal", False)),
            input_files=tuple(sample_cfg.get("input_files", [])),
            norm_factor=float(sample_cfg.get("norm_factor", 1.0)),
            lumi=sample_cfg.get("lumi"),
        )
        for sample_key, sample_cfg in raw_samples.items()
    }

    raw_subcategories = config.get("Subcategories", {})
    subcategories = {
        sample_key: [
            CategorySplit(name=split_name, categories=parse_category_values(raw_categories))
            for split_name, raw_categories in split_cfg.items()
        ]
        for sample_key, split_cfg in raw_subcategories.items()
    }
    feature_config = parse_feature_config(config)
    return samples, subcategories, feature_config


def split_sample_by_category(sample: Sample, events: ak.Array, splits: list[CategorySplit]) -> list[tuple[Sample, ak.Array]]:
    if "event_category" not in events.fields:
        raise ValueError(f"Sample '{sample.name}' is missing 'event_category' and cannot be split.")

    available_mask = ak.ones_like(events["event_category"], dtype=bool)
    outputs: list[tuple[Sample, ak.Array]] = []

    console.print(f"[bold cyan]Splitting sample[/bold cyan] [white]{sample.name}[/white] by event_category")
    for split in splits:
        category_mask = ak.zeros_like(events["event_category"], dtype=bool)
        for category in split.categories:
            category_mask = category_mask | (events["event_category"] == category)
        category_mask = available_mask & category_mask
        split_events = events[category_mask]
        if len(split_events) == 0:
            console.print(f"  [yellow]Skipping empty split[/yellow] [white]{split.name}[/white]")
            continue

        outputs.append((replace(sample, name=split.name), split_events))
        available_mask = available_mask & ~category_mask
        console.print(
            f"  [green]Created[/green] [white]{split.name}[/white] "
            f"categories={list(split.categories)} events=[white]{len(split_events)}[/white]"
        )

    remainder = events[available_mask]
    if len(remainder) > 0:
        remainder_name = f"{sample.name}_others"
        outputs.append((replace(sample, name=remainder_name), remainder))
        console.print(f"  [green]Created[/green] [white]{remainder_name}[/white] events=[white]{len(remainder)}[/white]")

    return outputs


def expand_samples(samples: dict[str, Sample], sample_events: dict[str, ak.Array], subcategories: dict[str, list[CategorySplit]]) -> list[tuple[Sample, ak.Array]]:
    expanded: list[tuple[Sample, ak.Array]] = []
    seen_names: set[str] = set()

    for sample_key, sample in samples.items():
        splits = subcategories.get(sample_key) or subcategories.get(sample.name)
        entries = split_sample_by_category(sample, sample_events[sample_key], splits) if splits else [(sample, sample_events[sample_key])]
        for split_sample, split_events in entries:
            if split_sample.name in seen_names:
                raise ValueError(f"Duplicate sample name '{split_sample.name}' after expansion.")
            seen_names.add(split_sample.name)
            expanded.append((split_sample, split_events))
    return expanded


def pad_and_flatten_part_feature(values: ak.Array, max_particles: int) -> ak.Array:
    padded = ak.pad_none(values, max_particles, axis=1, clip=True)
    filled = ak.fill_none(padded, 0)
    regular = ak.to_regular(filled, axis=1)
    return ak.values_astype(regular, np.float32)[..., np.newaxis]


def flatten_global_feature(values: ak.Array) -> ak.Array:
    filled = ak.fill_none(values, 0)
    return ak.values_astype(filled, np.float32)[..., np.newaxis]


def _p4_component(p4: ak.Array, component: str) -> ak.Array:
    fields = set(p4.fields)
    if component == "px":
        return p4["px"] if "px" in fields else p4["x"]
    if component == "py":
        return p4["py"] if "py" in fields else p4["y"]
    if component == "pz":
        return p4["pz"] if "pz" in fields else p4["z"]
    if component in {"E", "energy"}:
        return p4["E"] if "E" in fields else p4["t"]
    raise ValueError(f"Unsupported p4 component '{component}'.")


def resolve_global_feature(events: ak.Array, field_name: str) -> ak.Array:
    if field_name in events.fields:
        return events[field_name]

    if "missing_p4" in events.fields:
        missing_p4 = events["missing_p4"]
        if field_name == "missing_px":
            return _p4_component(missing_p4, "px")
        if field_name == "missing_py":
            return _p4_component(missing_p4, "py")
        if field_name == "missing_pz":
            return _p4_component(missing_p4, "pz")
        if field_name in {"missing_E", "missing_energy"}:
            return _p4_component(missing_p4, "E")
        if field_name == "missing_pt":
            px = _p4_component(missing_p4, "px")
            py = _p4_component(missing_p4, "py")
            return np.sqrt(px * px + py * py)

    preview = ", ".join(events.fields[:25])
    suffix = " ..." if len(events.fields) > 25 else ""
    raise KeyError(f"Global feature '{field_name}' is missing. Available fields include: {preview}{suffix}")


def build_input_particle_mask(events: ak.Array, remove_neutral_non_photon: bool) -> ak.Array:
    """
    inputs:
      events: ak.Array, selected central parquet events with Part_* jagged fields.
      remove_neutral_non_photon: bool, when true keep charged particles and photons only.
    outputs:
      ak.Array[bool], jagged per-particle mask aligned with Part_* collections.
    goal:
      Define the particle collection that becomes EveNet's sequential input.
    """
    mask = part_energy_mask(events)
    if not remove_neutral_non_photon:
        return mask

    missing_fields = [field for field in ("Part_charge", "Part_pdgId") if field not in events.fields]
    if missing_fields:
        raise KeyError(
            "Cannot remove neutral non-photon particles because the selected parquet is missing "
            f"{missing_fields}."
        )

    charge = events["Part_charge"]
    abs_pdg_id = abs(events["Part_pdgId"])
    keep_particle = (charge != 0) | (abs_pdg_id == PHOTON_PDG_ID)
    return mask & ak.values_astype(keep_particle, bool)


def filter_input_part_values(values: ak.Array, input_part_mask: ak.Array) -> ak.Array:
    return values[input_part_mask]


def build_point_cloud(
    events: ak.Array,
    max_particles: int,
    feature_config: FeatureConfig,
    remove_neutral_non_photon: bool,
):
    input_part_mask = build_input_particle_mask(events, remove_neutral_non_photon)
    part_p4 = filter_input_part_values(build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"],
        events["Part_fourMomentum_fCoordinates_fY"],
        events["Part_fourMomentum_fCoordinates_fZ"],
        events["Part_fourMomentum_fCoordinates_fT"],
    ), input_part_mask)
    eta = ak.where(np.isfinite(part_p4.eta), part_p4.eta, 0)

    available_momentum_features = {
        "energy": part_p4.E,
        "pt": part_p4.pt,
        "eta": eta,
        "phi": part_p4.phi,
    }

    features = []
    feature_names: list[str] = []
    for feature_name in feature_config.raw_sequential_fields:
        if feature_name in available_momentum_features:
            values = available_momentum_features[feature_name]
        elif feature_name.startswith("Part_") and feature_name[5:] in available_momentum_features:
            values = available_momentum_features[feature_name[5:]]
        else:
            values = filter_input_part_values(events[feature_name], input_part_mask)
        expanded = pad_and_flatten_part_feature(values, max_particles)
        features.append(expanded)
        feature_names.append(feature_name)

    # Point-cloud tensor shape: [event, particle slot, feature].
    x = ak.concatenate(features, axis=2)

    num_particles = ak.values_astype(ak.num(events["Part_pdgId"][input_part_mask], axis=1), np.float32)
    # Mask shape: [event, particle slot].
    x_mask = ak.Array(np.arange(max_particles)[None, :] < ak.to_numpy(num_particles, allow_missing=False)[:, None])
    return x, x_mask, num_particles, feature_names


def build_global_conditions(events: ak.Array, feature_config: FeatureConfig):
    features = []
    feature_names: list[str] = []

    for field_name in feature_config.global_fields:
        flattened = flatten_global_feature(resolve_global_feature(events, field_name))
        features.append(flattened)
        feature_names.append(field_name)

    # Global-condition tensor shape: [event, feature].
    conditions = ak.concatenate(features, axis=1)
    conditions_mask = ak.Array(np.ones((len(events), 1), dtype=bool))
    return conditions, conditions_mask, feature_names


def compute_event_totals(num_sequential_vectors, conditions_mask):
    return num_sequential_vectors + ak.values_astype(conditions_mask[:, 0], np.float32)


def infer_max_particles(
    expanded_samples: list[tuple[Sample, ak.Array]],
    override: int | None,
    remove_neutral_non_photon: bool,
) -> int:
    if override is not None:
        return override
    return max(
        int(ak.max(ak.num(events["Part_pdgId"][build_input_particle_mask(events, remove_neutral_non_photon)], axis=1)))
        for _, events in expanded_samples
    )


def select_training_samples(expanded_samples: list[tuple[Sample, ak.Array]]) -> list[tuple[Sample, ak.Array]]:
    training_samples = [(sample, events) for sample, events in expanded_samples if not sample.is_data]
    if not training_samples:
        raise ValueError("No MC samples remain for EveNet dataset building after filtering out data.")
    return training_samples


def select_data_samples(expanded_samples: list[tuple[Sample, ak.Array]]) -> list[tuple[Sample, ak.Array]]:
    return [(sample, events) for sample, events in expanded_samples if sample.is_data]


def to_numpy_array(values, dtype=None) -> np.ndarray:
    if isinstance(values, ak.Array):
        values = ak.to_numpy(values, allow_missing=False)
    else:
        values = np.asarray(values)
    if dtype is not None:
        values = values.astype(dtype)
    return values


def source_event_key_array(events: ak.Array) -> np.ndarray:
    for key in ("evtNumber", "Event_evtNumber"):
        if key in events.fields:
            return to_numpy_array(events[key], np.int64)
    return np.arange(len(events), dtype=np.int64)


def should_passthrough_prediction_field(field: str) -> bool:
    return (
        field in PREDICTION_PASSTHROUGH_EXACT_FIELDS
        or field.endswith(PREDICTION_PASSTHROUGH_SUFFIXES)
        or field.startswith(PREDICTION_PASSTHROUGH_PREFIXES)
    )


def convert_passthrough_field(values, field: str, required: bool) -> np.ndarray | None:
    nested_fields = list(getattr(values, "fields", []))
    if nested_fields:
        if required:
            raise ValueError(
                f"Required passthrough field '{field}' is a nested record/struct with subfields={nested_fields[:20]}; "
                "concat-based export requires a flat 1D scalar column."
            )
        return None

    array = to_numpy_array(values)
    if array.ndim != 1:
        if required:
            raise ValueError(
                f"Required passthrough field '{field}' has ndim={array.ndim}, shape={array.shape}; "
                "concat-based export requires a flat 1D scalar column."
            )
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        if required:
            raise ValueError(
                f"Required passthrough field '{field}' has unsupported dtype={array.dtype}; "
                "concat-based export requires a bool/int/uint/float 1D scalar column."
            )
        return None
    return array


def passthrough_prediction_fields(events: ak.Array, required_fields: set[str] | None = None) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    required_fields = required_fields or set()
    if "weight" in events.fields:
        output["central_weight"] = to_numpy_array(events["weight"], np.float32)

    for field in events.fields:
        if field == "weight":
            continue
        if not should_passthrough_prediction_field(field):
            continue
        array = convert_passthrough_field(events[field], field, required=field in required_fields)
        if array is None:
            continue
        output[field] = array
    return output


def expected_passthrough_output_fields(events: ak.Array) -> set[str]:
    expected: set[str] = set()
    if "weight" in events.fields:
        expected.add("central_weight")

    for field in events.fields:
        if field == "weight":
            continue
        if not should_passthrough_prediction_field(field):
            continue
        array = convert_passthrough_field(events[field], field, required=False)
        if array is None:
            continue
        expected.add(field)
    return expected


def required_concat_source_fields(sample: Sample) -> set[str]:
    return set(REQUIRED_MC_CONCAT_SOURCE_FIELDS) if sample_uses_invisible_target(sample) else set()


def validate_source_passthrough_contract(events: ak.Array, sample: Sample) -> set[str]:
    required = required_concat_source_fields(sample)
    missing = sorted(field for field in required if field not in events.fields)
    if missing:
        truth_like_fields = sorted(field for field in events.fields if field.startswith("truth_"))
        cut_like_fields = sorted(field for field in events.fields if field.endswith("_cut"))
        raise ValueError(
            "Selected-source parquet is missing fields required for concat-based unfolding export before EveNet "
            f"dataset building. sample='{sample.name}', missing_fields={missing[:20]}, "
            f"available_truth_fields={truth_like_fields[:20]}, available_cut_fields={cut_like_fields[:20]}. "
            "Check the exact parquet path in analysis.yaml and inspect its top-level keys."
        )
    return required


def default_concat_passthrough_fields(sample: Sample, num_events: int) -> dict[str, np.ndarray]:
    if sample.is_data:
        return {}

    defaults: dict[str, np.ndarray] = {
        "event_category": np.full(num_events, -1, dtype=np.int64),
        "initial_total_num_events": np.full(num_events, num_events, dtype=np.int64),
        "truth_QI_region": np.zeros(num_events, dtype=bool),
        "analyzing_power": np.zeros(num_events, dtype=np.float32),
        "analyzing_power_a": np.zeros(num_events, dtype=np.float32),
        "analyzing_power_b": np.zeros(num_events, dtype=np.float32),
        "truth_theta_cm": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
        "truth_mtautau": np.full(num_events, DEFAULT_FLOAT, dtype=np.float32),
    }
    for axis in ("n", "r", "k"):
        defaults[f"truth_cos_theta_A_{axis}"] = np.full(num_events, DEFAULT_FLOAT, dtype=np.float32)
        defaults[f"truth_cos_theta_B_{axis}"] = np.full(num_events, DEFAULT_FLOAT, dtype=np.float32)
    for axis_a, axis_b in product(("n", "r", "k"), repeat=2):
        defaults[f"truth_cos_theta_A_{axis_a}_times_cos_theta_B_{axis_b}"] = np.full(
            num_events, DEFAULT_FLOAT, dtype=np.float32
        )
    return defaults


def apply_default_concat_passthrough_fields(batch: dict[str, np.ndarray], sample: Sample, num_events: int) -> None:
    for field, values in default_concat_passthrough_fields(sample, num_events).items():
        batch.setdefault(field, values)


def default_missing_passthrough_array(field: str, reference: np.ndarray, num_events: int) -> np.ndarray:
    dtype = reference.dtype
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(num_events, dtype=bool)
    if np.issubdtype(dtype, np.integer):
        if field == "initial_total_num_events":
            return np.full(num_events, num_events, dtype=dtype)
        if field.startswith("truth_num_"):
            return np.zeros(num_events, dtype=dtype)
        if field == "event_category":
            return np.full(num_events, -1, dtype=dtype)
        return np.full(num_events, -1, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        if field.startswith("analyzing_power"):
            return np.zeros(num_events, dtype=dtype)
        return np.full(num_events, DEFAULT_FLOAT, dtype=dtype)
    raise ValueError(
        f"Cannot synthesize default passthrough values for field '{field}' with unsupported dtype={dtype}."
    )


def fill_missing_passthrough_batch_keys(
    batches: list[dict[str, np.ndarray]],
    batch_sample_names: list[str],
) -> None:
    reference_by_key: dict[str, np.ndarray] = {}
    for batch in batches:
        for key, values in batch.items():
            reference_by_key.setdefault(key, values)

    all_keys = sorted(reference_by_key)
    for sample_name, batch in zip(batch_sample_names, batches):
        for key in all_keys:
            if key in batch:
                continue
            if not should_passthrough_prediction_field(key):
                raise ValueError(
                    "EveNet builder found inconsistent non-passthrough batch keys across samples before NPZ writing. "
                    f"sample='{sample_name}' is missing key '{key}'."
                )
            batch[key] = default_missing_passthrough_array(key, reference_by_key[key], len(next(iter(batch.values()))))


def validate_batch_passthrough_fields(
    batch: dict[str, np.ndarray],
    events: ak.Array,
    sample: Sample,
) -> set[str]:
    required = validate_source_passthrough_contract(events, sample)
    expected = expected_passthrough_output_fields(events)
    missing_required = sorted(required - set(batch))
    if missing_required:
        raise ValueError(
            "EveNet builder found required concat-export fields in the selected-source parquet, but they were not "
            f"written into the in-memory dataset batch for sample '{sample.name}'. "
            f"missing_required_fields={missing_required[:20]}."
        )
    missing_expected = sorted(expected - set(batch))
    if missing_expected:
        raise ValueError(
            "EveNet builder failed to propagate required passthrough fields into the in-memory dataset batch for "
            f"sample '{sample.name}'. missing_fields={missing_expected[:20]}. This usually means the builder logic "
            "and the selected-source parquet schema are out of sync."
        )
    return expected


def build_dataset(
    expanded_samples: list[tuple[Sample, ak.Array]],
    max_particles: int,
    feature_config: FeatureConfig,
    evenet_config: EveNetConfig,
    include_classification: bool = True,
    remove_neutral_non_photon: bool = False,
) -> tuple[dict[str, np.ndarray], dict]:
    class_labels = [sample.name for sample, _ in expanded_samples] if include_classification else []
    class_index = {label: index for index, label in enumerate(class_labels)}

    batches = []
    batch_sample_names: list[str] = []
    batch_expected_passthrough: list[set[str]] = []
    point_cloud_feature_names: list[str] | None = None
    global_feature_names: list[str] | None = None

    for source_sample_index, (sample, events) in enumerate(expanded_samples):
        console.print(f"[bold cyan]Converting[/bold cyan] [white]{sample.name}[/white]")
        required_fields = validate_source_passthrough_contract(events, sample)

        x, x_mask, num_sequential_vectors, point_cloud_names = build_point_cloud(
            events,
            max_particles,
            feature_config,
            remove_neutral_non_photon,
        )
        conditions, conditions_mask, condition_names = build_global_conditions(events, feature_config)
        num_vectors = compute_event_totals(num_sequential_vectors, conditions_mask)

        tau_vis_prong_p4, tau_vis_prong_mask, tau_vis_rho_p4, tau_vis_rho_mask = build_visible_tau_assumptions(events)
        slot_for_a, slot_for_b = build_central_leg_slot_indices(events)
        x_invisible_p4, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target_p4, tau_vis_target_mask = build_tau_targets(
            events,
            tau_vis_prong_p4,
            tau_vis_prong_mask,
            tau_vis_rho_p4,
            tau_vis_rho_mask,
        )
        if not sample_uses_invisible_target(sample):
            x_invisible_mask = ak.Array(np.zeros((len(events), 2), dtype=bool))
            x_invisible_p4 = build_momentum4d(
                np.zeros((len(events), 2), dtype=np.float32),
                np.zeros((len(events), 2), dtype=np.float32),
                np.zeros((len(events), 2), dtype=np.float32),
                np.zeros((len(events), 2), dtype=np.float32),
            )
            tau_vis_target_mask = ak.Array(np.zeros((len(events), 2), dtype=bool))
            tau_vis_target_p4 = build_momentum4d(
                np.zeros((len(events), 2), dtype=np.float32),
                np.zeros((len(events), 2), dtype=np.float32),
                np.zeros((len(events), 2), dtype=np.float32),
                np.zeros((len(events), 2), dtype=np.float32),
            )
            num_invisible_raw = np.zeros(len(events), dtype=np.int64)
            num_invisible_valid = np.zeros(len(events), dtype=np.int64)
        tau_vis_prong = features_from_p4(tau_vis_prong_p4)
        tau_vis_rho = features_from_p4(tau_vis_rho_p4)
        tau_vis_target = features_from_p4(tau_vis_target_p4)
        x_invisible = features_from_p4(x_invisible_p4, evenet_config.invisible_features)
        batch = {
            # EveNet inputs: x [N, P, F_seq], conditions [N, F_global].
            "x": to_numpy_array(x, np.float32),
            "x_mask": to_numpy_array(x_mask, bool),
            "conditions": to_numpy_array(conditions, np.float32),
            "conditions_mask": to_numpy_array(conditions_mask, bool),
            "num_vectors": to_numpy_array(num_vectors, np.float32),
            "num_sequential_vectors": to_numpy_array(num_sequential_vectors, np.float32),
            # Truth/auxiliary targets use two canonical visible slots per event,
            # not a fixed tau- / tau+ ordering.
            "x_invisible": to_numpy_array(x_invisible, np.float32),
            "x_invisible_mask": to_numpy_array(x_invisible_mask, bool),
            "num_invisible_raw": num_invisible_raw,
            "num_invisible_valid": num_invisible_valid,
            "tau_vis_prong": to_numpy_array(tau_vis_prong, np.float32),
            "tau_vis_prong_mask": to_numpy_array(tau_vis_prong_mask, bool),
            "tau_vis_rho": to_numpy_array(tau_vis_rho, np.float32),
            "tau_vis_rho_mask": to_numpy_array(tau_vis_rho_mask, bool),
            "tau_vis_target": to_numpy_array(tau_vis_target, np.float32),
            "tau_vis_target_mask": to_numpy_array(tau_vis_target_mask, bool),
            "source_slot_for_a": slot_for_a,
            "source_slot_for_b": slot_for_b,
            # Source identifiers are carried through preprocessing shuffles/splits
            # so downstream prediction can be merged back into central parquet safely.
            "source_sample_index": np.full(len(events), source_sample_index, dtype=np.int64),
            "source_event_index": np.arange(len(events), dtype=np.int64),
            "source_event_key": source_event_key_array(events),
        }
        batch.update(passthrough_prediction_fields(events, required_fields=required_fields))
        apply_default_concat_passthrough_fields(batch, sample, len(events))
        if include_classification:
            batch["classification"] = np.full(len(events), class_index[sample.name], dtype=np.int64)
            batch["event_weight"] = np.ones(len(events), dtype=np.float32)

        batch_expected_passthrough.append(validate_batch_passthrough_fields(batch, events, sample))
        batches.append(batch)
        batch_sample_names.append(sample.name)

        if point_cloud_feature_names is None:
            point_cloud_feature_names = point_cloud_names
        if global_feature_names is None:
            global_feature_names = condition_names

    fill_missing_passthrough_batch_keys(batches, batch_sample_names)
    all_keys = sorted(set().union(*(batch.keys() for batch in batches)))

    dataset = {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in all_keys
    }

    expected_passthrough_union = sorted(set().union(*batch_expected_passthrough))
    missing_dataset_passthrough = sorted(set(expected_passthrough_union) - set(dataset))
    if missing_dataset_passthrough:
        raise ValueError(
            "EveNet builder dropped passthrough fields while constructing the final dataset. "
            f"missing_fields={missing_dataset_passthrough[:20]}."
        )

    metadata = {
        "point_cloud_features": point_cloud_feature_names,
        "global_features": global_feature_names,
        "invisible_features": list(evenet_config.invisible_features),
        "visible_tau_features": FOUR_VECTOR_FEATURES,
        "max_particles": max_particles,
        "remove_neutral_non_photon": bool(remove_neutral_non_photon),
        "num_events": int(dataset["x"].shape[0]),
        "source_samples": [sample.name for sample, _ in expanded_samples],
        "passthrough_fields": expected_passthrough_union,
    }
    if include_classification:
        metadata["class_labels"] = class_labels
    return dataset, metadata


def _lookup_feature_tag(feature_name: str, tags: dict[str, str], default: str = "none") -> str:
    if feature_name in tags:
        return tags[feature_name]
    stripped = feature_name[5:] if feature_name.startswith("Part_") else feature_name
    return tags.get(stripped, default)


def build_event_info_yaml(metadata: dict, feature_config: FeatureConfig, evenet_config: EveNetConfig) -> dict:
    class_labels = metadata["class_labels"]
    missing_processes = [label for label in class_labels if label not in evenet_config.process_topologies]
    if missing_processes:
        raise ValueError(
            "Missing EveNet process topology definitions for: "
            + ", ".join(missing_processes)
        )

    # EveNet-Full's current training/metric code assumes the event-level
    # classification head is named "signal". Keep that contract here and use
    # the class list contents for the actual process labels.
    classification_name = "signal"

    payload = {
        "INPUTS": {
            "SEQUENTIAL": {
                "Source": {
                    feature_name: _lookup_feature_tag(feature_name, evenet_config.sequential_tags)
                    for feature_name in metadata["point_cloud_features"]
                }
            },
            "GLOBAL": {
                "Conditions": {
                    feature_name: _lookup_feature_tag(feature_name, evenet_config.global_tags)
                    for feature_name in metadata["global_features"]
                }
            },
        },
        "EVENT": {
            label: {
                resonance_name: list(products)
                for resonance_name, products in evenet_config.process_topologies[label].items()
            }
            for label in class_labels
        },
        "CLASSIFICATIONS": {
            "EVENT": [classification_name]
        },
        "CLASSLABEL": {
            "EVENT": {
                classification_name: [class_labels]
            }
        },
        "GENERATIONS": {
            "Conditions": list(evenet_config.generation_conditions),
            "GlobalTargets": list(evenet_config.generation_global_targets),
            "Events": list(evenet_config.generation_events),
            "Neutrinos": {
                feature_name: evenet_config.invisible_tags.get(feature_name, "none")
                for feature_name in metadata["invisible_features"]
            },
        },
    }
    if feature_config.grouped_sequential_config is not None:
        payload["GROUPED_INPUTS"] = feature_config.grouped_sequential_config
    return payload


def write_monitor_plot(
    expanded_samples: list[tuple[Sample, ak.Array]],
    extractor,
    bins: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    luminosity: float | None,
    normalize: bool,
    log_scale: bool,
    mc_only: bool = False,
    signal_only: bool = False,
) -> None:
    hist_data = None if mc_only else np.zeros(len(bins) - 1, dtype=float)
    hist_mc: dict[str, np.ndarray] = {}
    hist_mc_err2: dict[str, np.ndarray] = {}
    raw_values_by_sample: dict[str, np.ndarray] = {}

    for sample, events in expanded_samples:
        if signal_only and not sample_uses_invisible_target(sample):
            continue
        raw_values = np.asarray(extractor(events), dtype=float)
        raw_values_by_sample[sample.name] = raw_values
        values = sanitize_hist_values(raw_values)
        hist, _ = np.histogram(values, bins=bins)

        if sample.is_data and not mc_only:
            hist_data += hist
            continue
        if sample.is_data:
            continue

        scale = sample_scale(sample, events, luminosity)
        hist_mc[sample.name] = hist.astype(float) * scale
        hist_mc_err2[sample.name] = hist.astype(float) * (scale ** 2)

    if not hist_mc and mc_only:
        return

    plot_from_histograms(
        hist_data=hist_data,
        hist_mc=hist_mc,
        hist_mc_err2=hist_mc_err2,
        bin_edges=bins,
        x_label=xlabel,
        title=title,
        output_path=output_path,
        normalize=normalize,
        log_scale=log_scale,
        invalid_summary=summarize_invalid_hist_values(raw_values_by_sample),
    )
    console.print(f"  [green]Wrote monitor[/green] [white]{output_path}[/white]")


def write_monitoring_plots(
    raw_expanded_samples: list[tuple[Sample, ak.Array]],
    expanded_samples: list[tuple[Sample, ak.Array]],
    output_dir: Path,
    luminosity: float | None,
    normalize: bool,
    feature_config: FeatureConfig,
) -> None:
    monitor_dir = output_dir / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Monitoring plots[/bold] [white]{monitor_dir}[/white]")

    write_monitor_plot(
        expanded_samples=raw_expanded_samples,
        extractor=lambda events: ak.to_numpy(events["nprong"], allow_missing=False),
        bins=np.arange(-0.5, 8.5, 1.0),
        title="nprong before preselection",
        xlabel="nprong",
        output_path=monitor_dir / "nprong_before_preselection.png",
        luminosity=luminosity,
        normalize=normalize,
        log_scale=False,
    )
    write_monitor_plot(
        expanded_samples=expanded_samples,
        extractor=lambda events: ak.to_numpy(events["nprong"], allow_missing=False),
        bins=np.arange(-0.5, 8.5, 1.0),
        title="nprong after preselection",
        xlabel="nprong",
        output_path=monitor_dir / "nprong_after_preselection.png",
        luminosity=luminosity,
        normalize=normalize,
        log_scale=False,
    )

    for field_name in feature_config.global_fields:
        values_by_sample = {
            sample.name: sanitize_hist_values(ak.to_numpy(resolve_global_feature(events, field_name), allow_missing=False))
            for sample, events in expanded_samples
        }
        if not values_by_sample:
            continue
        bins = choose_bins(values_by_sample)
        write_monitor_plot(
            expanded_samples=expanded_samples,
            extractor=lambda events, field_name=field_name: ak.to_numpy(resolve_global_feature(events, field_name), allow_missing=False),
            bins=bins,
            title=field_name,
            xlabel=field_name,
            output_path=monitor_dir / f"{field_name}.png",
            luminosity=luminosity,
            normalize=normalize,
            log_scale=True,
        )

    for plot_name, extractor, mc_only, log_scale, signal_only in [
        ("tau_vis_prong_only_energy", lambda events: extract_visible_tau_observable(events, "prong_only", "energy"), False, True, False),
        ("tau_vis_prong_only_pt", lambda events: extract_visible_tau_observable(events, "prong_only", "pt"), False, True, False),
        ("tau_vis_prong_only_eta", lambda events: extract_visible_tau_observable(events, "prong_only", "eta"), False, False, False),
        ("tau_vis_prong_only_phi", lambda events: extract_visible_tau_observable(events, "prong_only", "phi"), False, False, False),
        ("tau_vis_prong_only_mass", lambda events: extract_visible_tau_observable(events, "prong_only", "mass"), False, False, False),
        ("tau_vis_prong_energy", lambda events: extract_visible_tau_observable(events, "prong", "energy"), False, True, False),
        ("tau_vis_prong_pt", lambda events: extract_visible_tau_observable(events, "prong", "pt"), False, True, False),
        ("tau_vis_prong_eta", lambda events: extract_visible_tau_observable(events, "prong", "eta"), False, False, False),
        ("tau_vis_prong_phi", lambda events: extract_visible_tau_observable(events, "prong", "phi"), False, False, False),
        ("tau_vis_prong_mass", lambda events: extract_visible_tau_observable(events, "prong", "mass"), False, False, False),
        ("tau_vis_rho_energy", lambda events: extract_visible_tau_observable(events, "rho", "energy"), False, True, False),
        ("tau_vis_rho_pt", lambda events: extract_visible_tau_observable(events, "rho", "pt"), False, True, False),
        ("tau_vis_rho_eta", lambda events: extract_visible_tau_observable(events, "rho", "eta"), False, False, False),
        ("tau_vis_rho_phi", lambda events: extract_visible_tau_observable(events, "rho", "phi"), False, False, False),
        ("tau_vis_rho_mass", lambda events: extract_visible_tau_observable(events, "rho", "mass"), False, False, False),
        ("target_invisible_E", lambda events: extract_target_invisible_observable(events, "E"), True, True, True),
        ("target_invisible_px", lambda events: extract_target_invisible_observable(events, "px"), True, False, True),
        ("target_invisible_py", lambda events: extract_target_invisible_observable(events, "py"), True, False, True),
        ("target_invisible_pz", lambda events: extract_target_invisible_observable(events, "pz"), True, False, True),
        ("target_invisible_energy", lambda events: extract_target_invisible_observable(events, "energy"), True, True, True),
        ("target_invisible_pt", lambda events: extract_target_invisible_observable(events, "pt"), True, True, True),
        ("target_invisible_eta", lambda events: extract_target_invisible_observable(events, "eta"), True, False, True),
        ("target_invisible_phi", lambda events: extract_target_invisible_observable(events, "phi"), True, False, True),
    ]:
        values_by_sample = {
            sample.name: sanitize_hist_values(extractor(events))
            for sample, events in expanded_samples
            if not (mc_only and sample.is_data) and not (signal_only and not sample_uses_invisible_target(sample))
        }
        if not any(values.size > 0 for values in values_by_sample.values()):
            if plot_name.startswith("target_invisible"):
                console.print(f"  [yellow]Skipped monitor[/yellow] [white]{plot_name}[/white]: no target invisible entries")
            continue
        bins = choose_bins(values_by_sample)
        write_monitor_plot(
            expanded_samples=expanded_samples,
            extractor=extractor,
            bins=bins,
            title=plot_name,
            xlabel=plot_name,
            output_path=monitor_dir / f"{plot_name}.png",
            luminosity=luminosity,
            normalize=normalize,
            log_scale=log_scale,
            mc_only=mc_only,
            signal_only=signal_only,
        )

    for observable in FOUR_VECTOR_FEATURES:
        values_by_sample = {
            sample.name: sanitize_hist_values(extract_part_momentum_observable(events, observable))
            for sample, events in expanded_samples
        }
        if not any(values.size > 0 for values in values_by_sample.values()):
            continue
        write_monitor_plot(
            expanded_samples=expanded_samples,
            extractor=lambda events, observable=observable: extract_part_momentum_observable(events, observable),
            bins=choose_bins(values_by_sample),
            title=f"Part_{observable}",
            xlabel=f"Part_{observable}",
            output_path=monitor_dir / f"Part_{observable}.png",
            luminosity=luminosity,
            normalize=normalize,
            log_scale=observable not in {"eta", "phi"},
        )

    for field_name in feature_config.part_aux_fields:
        values_by_sample = {
            sample.name: sanitize_hist_values(extract_part_feature(events, field_name))
            for sample, events in expanded_samples
            if field_name in events.fields
        }
        if not any(values.size > 0 for values in values_by_sample.values()):
            continue
        write_monitor_plot(
            expanded_samples=[(sample, events) for sample, events in expanded_samples if field_name in events.fields],
            extractor=lambda events, field_name=field_name: extract_part_feature(events, field_name),
            bins=choose_bins(values_by_sample),
            title=field_name,
            xlabel=field_name,
            output_path=monitor_dir / f"{field_name}.png",
            luminosity=luminosity,
            normalize=normalize,
            log_scale=False,
        )

    for field_name in ["Part_lock"]:
        values_by_sample = {
            sample.name: sanitize_hist_values(extract_part_feature(events, field_name))
            for sample, events in expanded_samples
            if field_name in events.fields
        }
        if not any(values.size > 0 for values in values_by_sample.values()):
            continue
        write_monitor_plot(
            expanded_samples=[(sample, events) for sample, events in expanded_samples if field_name in events.fields],
            extractor=lambda events, field_name=field_name: extract_part_feature(events, field_name),
            bins=np.arange(-0.5, 3.5, 1.0),
            title=field_name,
            xlabel=field_name,
            output_path=monitor_dir / f"{field_name}.png",
            luminosity=luminosity,
            normalize=normalize,
            log_scale=False,
        )


def write_outputs(
    dataset: dict[str, np.ndarray],
    metadata: dict,
    output_dir: Path,
    stem: str,
    write_event_info: bool = False,
    feature_config: FeatureConfig | None = None,
    evenet_config: EveNetConfig | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}_metadata.json"

    np.savez_compressed(npz_path, **dataset)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    with np.load(npz_path, allow_pickle=False) as saved_npz:
        saved_keys = set(saved_npz.files)
    expected_keys = set(dataset)
    missing_saved_keys = sorted(expected_keys - saved_keys)
    extra_saved_keys = sorted(saved_keys - expected_keys)
    if missing_saved_keys or extra_saved_keys:
        raise ValueError(
            f"Saved NPZ key verification failed for {npz_path}. "
            f"missing_keys={missing_saved_keys[:20]}, extra_keys={extra_saved_keys[:20]}."
        )

    console.print(f"[green]Wrote[/green] [white]{npz_path}[/white]")
    console.print(f"[green]Wrote[/green] [white]{metadata_path}[/white]")
    if write_event_info:
        if feature_config is None or evenet_config is None:
            raise ValueError("feature_config and evenet_config are required when write_event_info=True.")
        event_info_payload = build_event_info_yaml(metadata, feature_config, evenet_config)
        event_info_path = output_dir / "event_info.yaml"
        with event_info_path.open("w") as handle:
            yaml.safe_dump(event_info_payload, handle, sort_keys=False)
        console.print(f"[green]Wrote[/green] [white]{event_info_path}[/white]")

        # Keep the latest generated schema in ml_pipeline/config so the
        # static preprocess_config.yaml wrapper can always reference it.
        GENERATED_EVENT_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GENERATED_EVENT_INFO_PATH.open("w") as handle:
            yaml.safe_dump(event_info_payload, handle, sort_keys=False)
        console.print(f"[green]Wrote[/green] [white]{GENERATED_EVENT_INFO_PATH}[/white]")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DataLoader awkward parquet files into a simple EveNet-style NPZ bundle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/config/analysis.yaml"),
        help="YAML config with Samples, feature lists, and normalization rules.",
    )
    parser.add_argument(
        "--evenet-config",
        type=Path,
        default=CONFIG_DIR / "evenet_schema.yaml",
        help="YAML config with EveNet process topology and generation settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/evenet_inputs"),
        help="Directory for the converted EveNet-style outputs.",
    )
    parser.add_argument(
        "--max-particles",
        type=int,
        default=None,
        help="Optional fixed padding size for the point cloud. Defaults to the maximum in the loaded dataset.",
    )
    parser.add_argument(
        "--remove-neutral-non-photon",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Build EveNet point-cloud inputs from charged particles plus photons only. "
            "Use --no-remove-neutral-non-photon to override the config and keep all neutral particles."
        ),
    )
    return parser


def resolve_remove_neutral_non_photon(args: argparse.Namespace, analysis_config: dict) -> bool:
    """
    inputs:
      args: argparse.Namespace, parsed command-line options.
      analysis_config: dict, analysis.yaml content.
    outputs:
      bool, final particle-collection filtering mode.
    goal:
      Let command-line flags override the optional EveNetInput config block.
    """
    if args.remove_neutral_non_photon is not None:
        return bool(args.remove_neutral_non_photon)
    return bool(analysis_config.get("EveNetInput", {}).get("remove_neutral_non_photon", False))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    console.print(f"[bold]Using config[/bold] [white]{args.config}[/white]")
    console.print(f"[bold]Using EveNet schema[/bold] [white]{args.evenet_config}[/white]")
    analysis_config = read_yaml(args.config)
    remove_neutral_non_photon = resolve_remove_neutral_non_photon(args, analysis_config)
    evenet_schema_config = read_yaml(args.evenet_config)
    samples, subcategories, feature_config = parse_config(args.config)
    evenet_config = parse_evenet_config(
        merge_evenet_config(evenet_schema_config, analysis_config),
        feature_config,
    )
    raw_sample_events = {
        sample_key: load_sample_events(sample)
        for sample_key, sample in samples.items()
    }
    sample_events = {
        sample_key: apply_preselection(raw_sample_events[sample_key])
        for sample_key in samples
    }
    raw_expanded_samples = expand_samples(samples, raw_sample_events, subcategories)
    expanded_samples = expand_samples(samples, sample_events, subcategories)

    training_samples = select_training_samples(expanded_samples)
    data_samples = select_data_samples(expanded_samples)

    max_particles = infer_max_particles(training_samples, args.max_particles, remove_neutral_non_photon)
    console.print(f"[bold]Point-cloud padding[/bold] max_particles=[white]{max_particles}[/white]")
    console.print(
        "[bold]Point-cloud particles[/bold] "
        + (
            "[white]charged particles + photons only[/white]"
            if remove_neutral_non_photon
            else "[white]all particles passing energy sanity mask[/white]"
        )
    )

    sample_table = Table(title="Expanded Samples")
    sample_table.add_column("Sample")
    sample_table.add_column("Type")
    sample_table.add_column("Events", justify="right")
    sample_table.add_column("Use in EveNet", justify="center")
    for sample, events in expanded_samples:
        sample_table.add_row(
            sample.name,
            "data" if sample.is_data else "mc",
            str(len(events)),
            "no" if sample.is_data else "yes",
        )
    console.print(sample_table)

    console.print(
        f"[bold]EveNet training samples[/bold] "
        f"[white]{', '.join(sample.name for sample, _ in training_samples)}[/white]"
    )

    mc_dataset, mc_metadata = build_dataset(
        training_samples,
        max_particles,
        feature_config,
        evenet_config,
        include_classification=True,
        remove_neutral_non_photon=remove_neutral_non_photon,
    )
    write_outputs(
        mc_dataset,
        mc_metadata,
        args.output_dir,
        stem="evenet_input",
        write_event_info=True,
        feature_config=feature_config,
        evenet_config=evenet_config,
    )

    if data_samples:
        console.print(
            f"[bold]Data payload samples[/bold] "
            f"[white]{', '.join(sample.name for sample, _ in data_samples)}[/white]"
        )
        data_dataset, data_metadata = build_dataset(
            data_samples,
            max_particles,
            feature_config,
            evenet_config,
            include_classification=False,
            remove_neutral_non_photon=remove_neutral_non_photon,
        )
        write_outputs(
            data_dataset,
            data_metadata,
            args.output_dir,
            stem="data",
            write_event_info=False,
        )
    else:
        console.print("[yellow]Skipped dataw[/yellow] no data samples configured")

    resolved_luminosity = infer_luminosity(samples, None)
    normalize = resolved_luminosity is None
    console.print(
        f"[bold]Monitoring normalization[/bold] "
        f"{'shape-only' if normalize else 'absolute-yield'}"
        + (
            f" (luminosity={resolved_luminosity} pb^-1)"
            if resolved_luminosity is not None
            else ""
        )
    )

    write_monitoring_plots(
        raw_expanded_samples=raw_expanded_samples,
        expanded_samples=expanded_samples,
        output_dir=args.output_dir,
        luminosity=resolved_luminosity,
        normalize=normalize,
        feature_config=feature_config,
    )


if __name__ == "__main__":
    main()
