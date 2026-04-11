#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, replace
from pathlib import Path

import awkward as ak
import numpy as np
import yaml
from evenet_parquet_common import (
    FOUR_VECTOR_FEATURES,
    build_tau_targets,
    build_visible_tau_assumptions,
    build_momentum4d,
    extract_target_invisible_observable,
    extract_part_feature,
    extract_part_momentum_observable,
    extract_visible_tau_observable,
    features_from_p4,
)
from ml_pipeline_config import EveNetConfig, FeatureConfig, parse_evenet_config, parse_feature_config
from parquet_plot_common import (
    choose_bins,
    infer_luminosity,
    plot_from_histograms,
    sanitize_hist_values,
    sample_scale,
)
from rich.console import Console
from rich.table import Table


console = Console()
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
GENERATED_EVENT_INFO_PATH = CONFIG_DIR / "generated_event_info.yaml"

@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    is_data: bool
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


def expand_input_files(patterns: tuple[str, ...]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if matched:
            files.extend(matched)
        else:
            files.append(pattern)
    return files


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

    selected = events[events["nprong"] == 2]
    console.print(
        f"  [bold cyan]Preselection[/bold cyan] nprong == 2 -> "
        f"[white]{len(selected)}[/white] / [white]{len(events)}[/white] event(s)"
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


def build_point_cloud(events: ak.Array, max_particles: int, feature_config: FeatureConfig):
    part_p4 = build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"],
        events["Part_fourMomentum_fCoordinates_fY"],
        events["Part_fourMomentum_fCoordinates_fZ"],
        events["Part_fourMomentum_fCoordinates_fT"],
    )
    eta = ak.where(np.isfinite(part_p4.eta), part_p4.eta, 0)

    features = []
    feature_names: list[str] = []

    available_momentum_features = {
        "energy": part_p4.E,
        "pt": part_p4.pt,
        "eta": eta,
        "phi": part_p4.phi,
    }
    for feature_name in feature_config.part_momentum_fields:
        values = available_momentum_features[feature_name]
        expanded = pad_and_flatten_part_feature(values, max_particles)
        features.append(expanded)
        feature_names.append(f"Part_{feature_name}")

    for field_name in feature_config.part_aux_fields:
        expanded = pad_and_flatten_part_feature(events[field_name], max_particles)
        features.append(expanded)
        feature_names.append(field_name)

    # Point-cloud tensor shape: [event, particle slot, feature].
    x = ak.concatenate(features, axis=2)

    num_particles = ak.values_astype(ak.num(events["Part_pdgId"], axis=1), np.float32)
    # Mask shape: [event, particle slot].
    x_mask = ak.Array(np.arange(max_particles)[None, :] < ak.to_numpy(num_particles, allow_missing=False)[:, None])
    return x, x_mask, num_particles, feature_names


def build_global_conditions(events: ak.Array, feature_config: FeatureConfig):
    features = []
    feature_names: list[str] = []

    for field_name in feature_config.global_fields:
        flattened = flatten_global_feature(events[field_name])
        features.append(flattened)
        feature_names.append(field_name)

    # Global-condition tensor shape: [event, feature].
    conditions = ak.concatenate(features, axis=1)
    conditions_mask = ak.Array(np.ones((len(events), 1), dtype=bool))
    return conditions, conditions_mask, feature_names


def compute_event_totals(num_sequential_vectors, conditions_mask):
    return num_sequential_vectors + ak.values_astype(conditions_mask[:, 0], np.float32)


def infer_max_particles(expanded_samples: list[tuple[Sample, ak.Array]], override: int | None) -> int:
    if override is not None:
        return override
    return max(int(ak.max(ak.num(events["Part_pdgId"], axis=1))) for _, events in expanded_samples)


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


def build_dataset(
    expanded_samples: list[tuple[Sample, ak.Array]],
    max_particles: int,
    feature_config: FeatureConfig,
    include_classification: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    class_labels = [sample.name for sample, _ in expanded_samples] if include_classification else []
    class_index = {label: index for index, label in enumerate(class_labels)}

    batches = []
    point_cloud_feature_names: list[str] | None = None
    global_feature_names: list[str] | None = None

    for sample, events in expanded_samples:
        console.print(f"[bold cyan]Converting[/bold cyan] [white]{sample.name}[/white]")

        x, x_mask, num_sequential_vectors, point_cloud_names = build_point_cloud(events, max_particles, feature_config)
        conditions, conditions_mask, condition_names = build_global_conditions(events, feature_config)
        num_vectors = compute_event_totals(num_sequential_vectors, conditions_mask)

        tau_vis_prong_p4, tau_vis_prong_mask, tau_vis_rho_p4, tau_vis_rho_mask = build_visible_tau_assumptions(events)
        x_invisible_p4, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target_p4, tau_vis_target_mask = build_tau_targets(
            events,
            tau_vis_prong_p4,
            tau_vis_prong_mask,
            tau_vis_rho_p4,
            tau_vis_rho_mask,
        )
        tau_vis_prong = features_from_p4(tau_vis_prong_p4)
        tau_vis_rho = features_from_p4(tau_vis_rho_p4)
        tau_vis_target = features_from_p4(tau_vis_target_p4)
        x_invisible = features_from_p4(x_invisible_p4)
        batch = {
            # EveNet inputs: x [N, P, F_seq], conditions [N, F_global].
            "x": to_numpy_array(x, np.float32),
            "x_mask": to_numpy_array(x_mask, bool),
            "conditions": to_numpy_array(conditions, np.float32),
            "conditions_mask": to_numpy_array(conditions_mask, bool),
            "num_vectors": to_numpy_array(num_vectors, np.float32),
            "num_sequential_vectors": to_numpy_array(num_sequential_vectors, np.float32),
            # Truth/auxiliary targets: [N, 2, 4] for tau- / tau+ and [N, 2] masks.
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
        }
        if include_classification:
            batch["classification"] = np.full(len(events), class_index[sample.name], dtype=np.int64)
            batch["event_weight"] = np.ones(len(events), dtype=np.float32)

        batches.append(batch)

        if point_cloud_feature_names is None:
            point_cloud_feature_names = point_cloud_names
        if global_feature_names is None:
            global_feature_names = condition_names

    dataset = {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }

    metadata = {
        "point_cloud_features": point_cloud_feature_names,
        "global_features": global_feature_names,
        "invisible_features": FOUR_VECTOR_FEATURES,
        "visible_tau_features": FOUR_VECTOR_FEATURES,
        "max_particles": max_particles,
        "num_events": int(dataset["x"].shape[0]),
        "source_samples": [sample.name for sample, _ in expanded_samples],
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

    return {
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
) -> None:
    hist_data = None if mc_only else np.zeros(len(bins) - 1, dtype=float)
    hist_mc: dict[str, np.ndarray] = {}
    hist_mc_err2: dict[str, np.ndarray] = {}

    for sample, events in expanded_samples:
        values = sanitize_hist_values(extractor(events))
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
            sample.name: sanitize_hist_values(ak.to_numpy(events[field_name], allow_missing=False))
            for sample, events in expanded_samples
            if field_name in events.fields
        }
        if not values_by_sample:
            continue
        bins = choose_bins(values_by_sample)
        write_monitor_plot(
            expanded_samples=[(sample, events) for sample, events in expanded_samples if field_name in events.fields],
            extractor=lambda events, field_name=field_name: ak.to_numpy(events[field_name], allow_missing=False),
            bins=bins,
            title=field_name,
            xlabel=field_name,
            output_path=monitor_dir / f"{field_name}.png",
            luminosity=luminosity,
            normalize=normalize,
            log_scale=True,
        )

    for plot_name, extractor, mc_only, log_scale in [
        ("tau_vis_prong_energy", lambda events: extract_visible_tau_observable(events, "prong", "energy"), False, True),
        ("tau_vis_prong_pt", lambda events: extract_visible_tau_observable(events, "prong", "pt"), False, True),
        ("tau_vis_prong_eta", lambda events: extract_visible_tau_observable(events, "prong", "eta"), False, False),
        ("tau_vis_prong_phi", lambda events: extract_visible_tau_observable(events, "prong", "phi"), False, False),
        ("tau_vis_prong_mass", lambda events: extract_visible_tau_observable(events, "prong", "mass"), False, False),
        ("tau_vis_rho_energy", lambda events: extract_visible_tau_observable(events, "rho", "energy"), False, True),
        ("tau_vis_rho_pt", lambda events: extract_visible_tau_observable(events, "rho", "pt"), False, True),
        ("tau_vis_rho_eta", lambda events: extract_visible_tau_observable(events, "rho", "eta"), False, False),
        ("tau_vis_rho_phi", lambda events: extract_visible_tau_observable(events, "rho", "phi"), False, False),
        ("tau_vis_rho_mass", lambda events: extract_visible_tau_observable(events, "rho", "mass"), False, False),
        ("target_invisible_energy", lambda events: extract_target_invisible_observable(events, "energy"), True, False),
        ("target_invisible_pt", lambda events: extract_target_invisible_observable(events, "pt"), True, False),
        ("target_invisible_eta", lambda events: extract_target_invisible_observable(events, "eta"), True, False),
        ("target_invisible_phi", lambda events: extract_target_invisible_observable(events, "phi"), True, False),
        ("target_invisible_mass", lambda events: extract_target_invisible_observable(events, "mass"), True, False),
    ]:
        values_by_sample = {
            sample.name: sanitize_hist_values(extractor(events))
            for sample, events in expanded_samples
            if not (mc_only and sample.is_data)
        }
        if not any(values.size > 0 for values in values_by_sample.values()):
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
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    console.print(f"[bold]Using config[/bold] [white]{args.config}[/white]")
    console.print(f"[bold]Using EveNet schema[/bold] [white]{args.evenet_config}[/white]")
    samples, subcategories, feature_config = parse_config(args.config)
    evenet_config = parse_evenet_config(read_yaml(args.evenet_config), feature_config)
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

    training_samples = select_training_samples(expanded_samples)
    data_samples = select_data_samples(expanded_samples)

    max_particles = infer_max_particles(training_samples, args.max_particles)
    console.print(f"[bold]Point-cloud padding[/bold] max_particles=[white]{max_particles}[/white]")

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

    mc_dataset, mc_metadata = build_dataset(training_samples, max_particles, feature_config, include_classification=True)
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
        data_dataset, data_metadata = build_dataset(data_samples, max_particles, feature_config, include_classification=False)
        write_outputs(
            data_dataset,
            data_metadata,
            args.output_dir,
            stem="data",
            write_event_info=False,
        )
    else:
        console.print("[yellow]Skipped dataw[/yellow] no data samples configured")


if __name__ == "__main__":
    main()
