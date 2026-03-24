#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, replace
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from evenet_parquet_common import (
    FOUR_VECTOR_FEATURES,
    build_tau_targets,
    build_visible_tau_assumptions,
    build_momentum4d,
)
from rich.console import Console
from rich.table import Table


console = Console()


PART_FIELDS = [
    "Part_charge",
    "Part_pdgId",
    "Part_vtxIdx",
    "Part_hpcShowerEnergy",
    "Part_hpcShowerTheta",
    "Part_hpcShowerPhi",
    "Part_hpcParticleCode",
    "Part_hpcNumLayers",
    "Part_hpcLayerHitPattern",
    "Part_hpcNumAssociatedShowers",
    "Part_hpcTotalShowerEnergy",
    "Part_hacShowerEnergy",
    "Part_hacShowerTheta",
    "Part_hacShowerPhi",
    "Part_hacParticleCode",
    "Part_hacNumTowers",
    "Part_hacTowerHitPattern",
    "Part_hacNumAssociatedShowers",
    "Part_hacTotalShowerEnergy",
    "Part_sticShowerEnergy",
    "Part_sticShowerTheta",
    "Part_sticShowerPhi",
    "Part_sticNumTowers",
    "Part_sticChargedTag",
    "Part_sticSiliconVertexPos",
    "Part_hemisphere",
]

GLOBAL_FIELDS = [
    "Event_totalChargedEnergy",
    "Event_totalEMEnergy",
    "Event_totalHadronicEnergy",
    "thrust_Mag",
    "charged_E",
    "missing_px",
    "missing_py",
    "missing_pt",
    "isolation_angle",
    "thrust_x",
    "thrust_y",
    "thrust_z",
]

@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    is_data: bool
    input_files: tuple[str, ...]


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


def parse_config(config_path: Path) -> tuple[dict[str, Sample], dict[str, list[CategorySplit]]]:
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
    return samples, subcategories


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


def component_names(base_name: str, array_ndim: int, trailing_shape: tuple[int, ...]) -> list[str]:
    if array_ndim == 2:
        return [base_name]

    flat_size = int(np.prod(trailing_shape))
    if flat_size == 1:
        return [base_name]
    return [f"{base_name}_{index}" for index in range(flat_size)]


def pad_and_flatten_part_feature(values: ak.Array, max_particles: int) -> np.ndarray:
    padded = ak.pad_none(values, max_particles, axis=1, clip=True)
    filled = ak.fill_none(padded, 0)
    numpy_values = ak.to_numpy(filled)
    if numpy_values.ndim == 2:
        return numpy_values[..., np.newaxis].astype(np.float32)
    return numpy_values.reshape(numpy_values.shape[0], numpy_values.shape[1], -1).astype(np.float32)


def flatten_global_feature(values: ak.Array) -> np.ndarray:
    filled = ak.fill_none(values, 0)
    numpy_values = ak.to_numpy(filled)
    if numpy_values.ndim == 1:
        return numpy_values[:, np.newaxis].astype(np.float32)
    return numpy_values.reshape(numpy_values.shape[0], -1).astype(np.float32)


def build_point_cloud(events: ak.Array, max_particles: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    part_p4 = build_momentum4d(
        events["Part_fourMomentum_fCoordinates_fX"],
        events["Part_fourMomentum_fCoordinates_fY"],
        events["Part_fourMomentum_fCoordinates_fZ"],
        events["Part_fourMomentum_fCoordinates_fT"],
    )
    eta = ak.where(np.isfinite(part_p4.eta), part_p4.eta, 0)

    features = []
    feature_names: list[str] = []

    momentum_features = {
        "Part_energy": part_p4.E,
        "Part_pt": part_p4.pt,
        "Part_eta": eta,
        "Part_phi": part_p4.phi,
    }
    for feature_name, values in momentum_features.items():
        expanded = pad_and_flatten_part_feature(values, max_particles)
        features.append(expanded)
        feature_names.extend(component_names(feature_name, expanded.ndim, expanded.shape[2:]))

    for field_name in PART_FIELDS:
        expanded = pad_and_flatten_part_feature(events[field_name], max_particles)
        features.append(expanded)
        feature_names.extend(component_names(field_name, expanded.ndim, expanded.shape[2:]))

    x = np.concatenate(features, axis=2).astype(np.float32)

    num_particles = ak.to_numpy(ak.num(events["Part_pdgId"], axis=1), allow_missing=False).astype(np.float32)
    x_mask = (np.arange(max_particles)[None, :] < num_particles[:, None])
    return x, x_mask, num_particles, feature_names


def build_global_conditions(events: ak.Array) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    feature_names: list[str] = []

    for field_name in GLOBAL_FIELDS:
        flattened = flatten_global_feature(events[field_name])
        features.append(flattened)
        feature_names.extend(component_names(field_name, flattened.ndim, flattened.shape[1:]))

    conditions = np.concatenate(features, axis=1).astype(np.float32)
    conditions_mask = np.ones((len(events), 1), dtype=bool)
    return conditions, conditions_mask, feature_names


def compute_event_totals(num_sequential_vectors: np.ndarray, conditions_mask: np.ndarray) -> np.ndarray:
    return num_sequential_vectors + conditions_mask[:, 0].astype(np.float32)


def infer_max_particles(expanded_samples: list[tuple[Sample, ak.Array]], override: int | None) -> int:
    if override is not None:
        return override
    return max(int(ak.max(ak.num(events["Part_pdgId"], axis=1))) for _, events in expanded_samples)


def build_dataset(expanded_samples: list[tuple[Sample, ak.Array]], max_particles: int) -> tuple[dict[str, np.ndarray], dict]:
    class_labels = [sample.name for sample, _ in expanded_samples]
    class_index = {label: index for index, label in enumerate(class_labels)}

    batches = []
    point_cloud_feature_names: list[str] | None = None
    global_feature_names: list[str] | None = None

    for sample, events in expanded_samples:
        console.print(f"[bold cyan]Converting[/bold cyan] [white]{sample.name}[/white]")

        x, x_mask, num_sequential_vectors, point_cloud_names = build_point_cloud(events, max_particles)
        conditions, conditions_mask, condition_names = build_global_conditions(events)
        num_vectors = compute_event_totals(num_sequential_vectors, conditions_mask)

        tau_vis_prong, tau_vis_prong_mask, tau_vis_rho, tau_vis_rho_mask = build_visible_tau_assumptions(events)
        x_invisible, x_invisible_mask, num_invisible_raw, num_invisible_valid, tau_vis_target, tau_vis_target_mask = build_tau_targets(
            events,
            tau_vis_prong,
            tau_vis_prong_mask,
            tau_vis_rho,
            tau_vis_rho_mask,
        )
        classification = np.full(len(events), class_index[sample.name], dtype=np.int64)
        event_weight = np.ones(len(events), dtype=np.float32)

        batches.append(
            {
                "x": x,
                "x_mask": x_mask,
                "conditions": conditions,
                "conditions_mask": conditions_mask,
                "classification": classification,
                "event_weight": event_weight,
                "num_vectors": num_vectors.astype(np.float32),
                "num_sequential_vectors": num_sequential_vectors.astype(np.float32),
                "x_invisible": x_invisible,
                "x_invisible_mask": x_invisible_mask,
                "num_invisible_raw": num_invisible_raw,
                "num_invisible_valid": num_invisible_valid,
                "tau_vis_prong": tau_vis_prong,
                "tau_vis_prong_mask": tau_vis_prong_mask,
                "tau_vis_rho": tau_vis_rho,
                "tau_vis_rho_mask": tau_vis_rho_mask,
                "tau_vis_target": tau_vis_target,
                "tau_vis_target_mask": tau_vis_target_mask,
            }
        )

        if point_cloud_feature_names is None:
            point_cloud_feature_names = point_cloud_names
        if global_feature_names is None:
            global_feature_names = condition_names

    dataset = {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }

    metadata = {
        "class_labels": class_labels,
        "point_cloud_features": point_cloud_feature_names,
        "global_features": global_feature_names,
        "invisible_features": FOUR_VECTOR_FEATURES,
        "visible_tau_features": FOUR_VECTOR_FEATURES,
        "max_particles": max_particles,
        "num_events": int(dataset["x"].shape[0]),
    }
    return dataset, metadata


def build_event_info_yaml(metadata: dict) -> dict:
    return {
        "INPUTS": {
            "SEQUENTIAL": {
                "Source": {feature_name: "none" for feature_name in metadata["point_cloud_features"]}
            },
            "GLOBAL": {
                "Conditions": {feature_name: "none" for feature_name in metadata["global_features"]}
            },
        },
        "CLASSIFICATIONS": {
            "EVENT": ["category_subcategory"]
        },
        "CLASSLABEL": {
            "EVENT": {
                "category_subcategory": [metadata["class_labels"]]
            }
        },
        "GENERATIONS": {
            "Neutrinos": {feature_name: "none" for feature_name in metadata["invisible_features"]}
        },
    }


def sanitize_plot_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def choose_bins(values_by_sample: dict[str, np.ndarray], num_bins: int = 60) -> np.ndarray:
    non_empty = [values for values in values_by_sample.values() if values.size > 0]
    if not non_empty:
        return np.linspace(0, 1, 2)
    merged = np.concatenate(non_empty)

    low, high = np.percentile(merged, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.min(merged))
        high = float(np.max(merged))
        if low == high:
            low -= 0.5
            high += 0.5
    return np.linspace(low, high, num_bins + 1)


def plot_step_histograms(values_by_sample: dict[str, np.ndarray], bins: np.ndarray, title: str, xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=220)
    for sample_name, values in values_by_sample.items():
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, label=sample_name)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Entries")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    console.print(f"  [green]Wrote monitor[/green] [white]{output_path}[/white]")


def write_preselection_plot(sample_name: str, raw_events: ak.Array, selected_events: ak.Array, output_dir: Path) -> None:
    if "nprong" not in raw_events.fields:
        return

    raw_values = sanitize_plot_values(ak.to_numpy(raw_events["nprong"], allow_missing=False))
    selected_values = sanitize_plot_values(ak.to_numpy(selected_events["nprong"], allow_missing=False))
    bins = np.arange(-0.5, max(np.max(raw_values, initial=0), np.max(selected_values, initial=0)) + 1.5, 1.0)
    if bins.size < 2:
        bins = np.arange(-0.5, 3.5, 1.0)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=220)
    ax.hist(raw_values, bins=bins, histtype="step", linewidth=2, label="before preselection")
    ax.hist(selected_values, bins=bins, histtype="step", linewidth=2, label="after nprong == 2")
    ax.set_title(f"{sample_name}: nprong preselection")
    ax.set_xlabel("nprong")
    ax.set_ylabel("Entries")
    ax.legend(loc="best")
    fig.tight_layout()
    output_path = output_dir / f"{sample_name}_nprong_preselection.png"
    fig.savefig(output_path)
    plt.close(fig)
    console.print(f"  [green]Wrote monitor[/green] [white]{output_path}[/white]")


def write_monitoring_plots(
    raw_sample_events: dict[str, ak.Array],
    selected_sample_events: dict[str, ak.Array],
    expanded_samples: list[tuple[Sample, ak.Array]],
    output_dir: Path,
) -> None:
    monitor_dir = output_dir / "monitoring"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Monitoring plots[/bold] [white]{monitor_dir}[/white]")

    for sample_key, raw_events in raw_sample_events.items():
        write_preselection_plot(sample_key, raw_events, selected_sample_events[sample_key], monitor_dir)

    for field_name in GLOBAL_FIELDS:
        values_by_sample = {
            sample.name: sanitize_plot_values(ak.to_numpy(events[field_name], allow_missing=False))
            for sample, events in expanded_samples
            if field_name in events.fields
        }
        if not values_by_sample:
            continue
        bins = choose_bins(values_by_sample)
        plot_step_histograms(
            values_by_sample=values_by_sample,
            bins=bins,
            title=field_name,
            xlabel=field_name,
            output_path=monitor_dir / f"{field_name}.png",
        )

    tau_vis_prong_pt = {}
    tau_vis_rho_pt = {}
    invisible_pt = {}
    for sample, events in expanded_samples:
        tau_vis_prong, tau_vis_prong_mask, tau_vis_rho, tau_vis_rho_mask = build_visible_tau_assumptions(events)
        x_invisible, x_invisible_mask, _, _, _, _ = build_tau_targets(
            events,
            tau_vis_prong,
            tau_vis_prong_mask,
            tau_vis_rho,
            tau_vis_rho_mask,
        )

        tau_vis_prong_pt[sample.name] = sanitize_plot_values(tau_vis_prong[..., 1][tau_vis_prong_mask])
        tau_vis_rho_pt[sample.name] = sanitize_plot_values(tau_vis_rho[..., 1][tau_vis_rho_mask])
        invisible_pt[sample.name] = sanitize_plot_values(x_invisible[..., 1][x_invisible_mask])

    for plot_name, values_by_sample in [
        ("tau_vis_prong_pt", tau_vis_prong_pt),
        ("tau_vis_rho_pt", tau_vis_rho_pt),
        ("target_invisible_pt", invisible_pt),
    ]:
        if not any(values.size > 0 for values in values_by_sample.values()):
            continue
        bins = choose_bins(values_by_sample)
        plot_step_histograms(
            values_by_sample=values_by_sample,
            bins=bins,
            title=plot_name,
            xlabel=plot_name,
            output_path=monitor_dir / f"{plot_name}.png",
        )


def write_outputs(dataset: dict[str, np.ndarray], metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "evenet_input.npz"
    metadata_path = output_dir / "evenet_input_metadata.json"
    event_info_path = output_dir / "event_info.yaml"

    np.savez_compressed(npz_path, **dataset)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    with event_info_path.open("w") as handle:
        yaml.safe_dump(build_event_info_yaml(metadata), handle, sort_keys=False)

    console.print(f"[green]Wrote[/green] [white]{npz_path}[/white]")
    console.print(f"[green]Wrote[/green] [white]{metadata_path}[/white]")
    console.print(f"[green]Wrote[/green] [white]{event_info_path}[/white]")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DataLoader awkward parquet files into a simple EveNet-style NPZ bundle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/config/analysis.yaml"),
        help="YAML config with Samples and optional Subcategories.",
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
    samples, subcategories = parse_config(args.config)
    raw_sample_events = {
        sample_key: load_sample_events(sample)
        for sample_key, sample in samples.items()
    }
    sample_events = {
        sample_key: apply_preselection(raw_sample_events[sample_key])
        for sample_key in samples
    }
    expanded_samples = expand_samples(samples, sample_events, subcategories)

    write_monitoring_plots(raw_sample_events, sample_events, expanded_samples, args.output_dir)

    max_particles = infer_max_particles(expanded_samples, args.max_particles)
    console.print(f"[bold]Point-cloud padding[/bold] max_particles=[white]{max_particles}[/white]")

    sample_table = Table(title="Expanded Samples")
    sample_table.add_column("Sample")
    sample_table.add_column("Type")
    sample_table.add_column("Events", justify="right")
    for sample, events in expanded_samples:
        sample_table.add_row(sample.name, "data" if sample.is_data else "mc", str(len(events)))
    console.print(sample_table)

    dataset, metadata = build_dataset(expanded_samples, max_particles)
    write_outputs(dataset, metadata, args.output_dir)


if __name__ == "__main__":
    main()
