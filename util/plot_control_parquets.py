#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import awkward as ak
import numpy as np
import yaml
from evenet_parquet_common import (
    FOUR_VECTOR_FEATURES,
    PART_AUX_FIELDS,
    extract_target_invisible_observable,
    extract_part_feature,
    extract_part_momentum_observable,
    extract_visible_tau_observable,
)
from parquet_plot_common import (
    choose_bins,
    get_initial_total_num_events,
    infer_luminosity,
    plot_from_histograms,
    sanitize_hist_values,
    sample_scale,
)
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class Sample:
    name: str
    is_data: bool
    is_signal: bool
    input_files: tuple[str, ...]
    norm_factor: float = 1.0
    lumi: float | None = None


@dataclass(frozen=True)
class PlotSpec:
    name: str
    x_label: str
    title: str
    bins: np.ndarray | None
    extractor: Callable[[ak.Array], np.ndarray]
    log_scale: bool = True


@dataclass(frozen=True)
class CategorySplit:
    name: str
    categories: tuple[int, ...]


console = Console()


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
    arrays = []
    parquet_files = expand_input_files(sample.input_files)
    console.print(
        f"[bold cyan]Loading sample[/bold cyan] [white]{sample.name}[/white] "
        f"from [white]{len(parquet_files)}[/white] parquet file(s)"
    )
    for parquet_path in parquet_files:
        console.print(f"  [dim]- {parquet_path}[/dim]")
        arrays.append(ak.from_parquet(parquet_path))

    if not arrays:
        raise ValueError(f"No parquet inputs found for sample '{sample.name}'.")
    if len(arrays) == 1:
        events = arrays[0]
    else:
        events = ak.concatenate(arrays, axis=0)

    console.print(
        f"  [green]Loaded[/green] [white]{len(events)}[/white] selected event(s), "
        f"initial_total_num_events=[white]{get_initial_total_num_events(events)}[/white]"
    )
    return events


def _parse_category_values(raw_values) -> tuple[int, ...]:
    if isinstance(raw_values, (list, tuple)):
        return tuple(int(value) for value in raw_values)
    return (int(raw_values),)


def parse_plot_config(config_path: Path) -> tuple[dict[str, Sample], dict[str, list[CategorySplit]]]:
    config = read_yaml(config_path)

    raw_samples = config.get("Samples", {})
    if not raw_samples:
        raise ValueError(f"No 'Samples' section found in {config_path}.")

    samples = {}
    for sample_key, sample_cfg in raw_samples.items():
        samples[sample_key] = Sample(
            name=sample_cfg.get("name", sample_key),
            is_data=bool(sample_cfg.get("is_data", False)),
            is_signal=bool(sample_cfg.get("is_signal", False)),
            input_files=tuple(sample_cfg.get("input_files", [])),
            norm_factor=float(sample_cfg.get("norm_factor", 1.0)),
            lumi=sample_cfg.get("lumi"),
        )

    raw_subcategories = config.get("Subcategories", {})
    legacy_signal_categories = config.get("signal_categories")
    if legacy_signal_categories is not None and "Ztautau" not in raw_subcategories:
        raw_subcategories = dict(raw_subcategories)
        raw_subcategories["Ztautau"] = legacy_signal_categories

    subcategories: dict[str, list[CategorySplit]] = {}
    for sample_key, split_cfg in raw_subcategories.items():
        subcategories[sample_key] = [
            CategorySplit(name=split_name, categories=_parse_category_values(category_values))
            for split_name, category_values in split_cfg.items()
        ]

    return samples, subcategories


def split_sample_by_event_category(
    sample: Sample,
    events: ak.Array,
    category_splits: list[CategorySplit],
) -> list[tuple[Sample, ak.Array]]:
    if "event_category" not in events.fields:
        raise ValueError(
            f"Sample '{sample.name}' cannot be split into subcategories because 'event_category' is missing."
        )

    available_mask = ak.ones_like(events["event_category"], dtype=bool)
    split_samples: list[tuple[Sample, ak.Array]] = []

    console.print(f"[bold cyan]Splitting sample[/bold cyan] [white]{sample.name}[/white] by event_category")

    for category_split in category_splits:
        category_mask = ak.zeros_like(events["event_category"], dtype=bool)
        for category in category_split.categories:
            category_mask = category_mask | (events["event_category"] == category)
        category_mask = available_mask & category_mask

        split_events = events[category_mask]
        if len(split_events) == 0:
            console.print(
                f"  [yellow]Skipping empty split[/yellow] [white]{category_split.name}[/white] "
                f"categories={list(category_split.categories)}"
            )
            continue

        console.print(
            f"  [green]Created[/green] [white]{category_split.name}[/white] "
            f"categories={list(category_split.categories)} events=[white]{len(split_events)}[/white]"
        )
        split_samples.append((replace(sample, name=category_split.name), split_events))
        available_mask = available_mask & ~category_mask

    remainder_events = events[available_mask]
    if len(remainder_events) > 0:
        console.print(
            f"  [green]Created[/green] [white]{sample.name}_others[/white] "
            f"events=[white]{len(remainder_events)}[/white]"
        )
        split_samples.append((replace(sample, name=f"{sample.name}_others"), remainder_events))

    return split_samples


def expand_samples(
    samples: dict[str, Sample],
    loaded_events: dict[str, ak.Array],
    subcategories: dict[str, list[CategorySplit]],
) -> list[tuple[Sample, ak.Array]]:
    expanded_samples: list[tuple[Sample, ak.Array]] = []
    seen_names: set[str] = set()

    for sample_key, sample in samples.items():
        split_cfg = subcategories.get(sample_key) or subcategories.get(sample.name)
        if split_cfg:
            split_entries = split_sample_by_event_category(sample, loaded_events[sample_key], split_cfg)
        else:
            split_entries = [(sample, loaded_events[sample_key])]

        for split_sample, split_events in split_entries:
            if split_sample.name in seen_names:
                raise ValueError(f"Duplicate sample name '{split_sample.name}' after subcategory expansion.")
            seen_names.add(split_sample.name)
            expanded_samples.append((split_sample, split_events))

    return expanded_samples


def extract_isolation_angle(events: ak.Array) -> np.ndarray:
    return ak.to_numpy(events["isolation_angle"], allow_missing=False)


def extract_erad(events: ak.Array) -> np.ndarray:
    return ak.to_numpy(events["E_rad"], allow_missing=False)


def extract_prad(events: ak.Array) -> np.ndarray:
    return ak.to_numpy(events["P_rad"], allow_missing=False)


def extract_charged_e(events: ak.Array) -> np.ndarray:
    return ak.to_numpy(events["charged_E"], allow_missing=False)


def extract_missing_pt(events: ak.Array) -> np.ndarray:
    return ak.to_numpy(events["missing_pt"], allow_missing=False)


def extract_nprong(events: ak.Array) -> np.ndarray:
    return ak.to_numpy(events["nprong"], allow_missing=False)


def extract_n_neutral(events: ak.Array) -> np.ndarray:
    neutral_mask = events["Part_charge"] == 0
    return ak.to_numpy(ak.sum(neutral_mask, axis=-1), allow_missing=False)


def extract_thrust_neglog1m(events: ak.Array) -> np.ndarray:
    thrust = events["thrust_Mag"]
    return ak.to_numpy(-np.log10(1 - thrust + 1e-10), allow_missing=False)


def extract_tau_vis_prong_pt(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "prong", "pt")


def extract_tau_vis_prong_energy(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "prong", "energy")


def extract_tau_vis_rho_pt(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "rho", "pt")


def extract_tau_vis_rho_energy(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "rho", "energy")


def extract_tau_vis_prong_eta(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "prong", "eta")


def extract_tau_vis_prong_phi(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "prong", "phi")


def extract_tau_vis_prong_mass(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "prong", "mass")


def extract_tau_vis_rho_eta(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "rho", "eta")


def extract_tau_vis_rho_phi(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "rho", "phi")


def extract_tau_vis_rho_mass(events: ak.Array) -> np.ndarray:
    return extract_visible_tau_observable(events, "rho", "mass")


def truth_array(events: ak.Array, field_name: str) -> np.ndarray:
    if field_name not in events.fields:
        return np.array([], dtype=float)
    return ak.to_numpy(ak.fill_none(events[field_name], np.nan), allow_missing=False)


def invariant_mass(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    mass2 = energy ** 2 - px ** 2 - py ** 2 - pz ** 2
    return np.sqrt(np.clip(mass2, a_min=0.0, a_max=None))


def extract_truth_tau_pt(events: ak.Array) -> np.ndarray:
    px = truth_array(events, "truth_tau_px")
    py = truth_array(events, "truth_tau_py")
    return np.sqrt(px ** 2 + py ** 2)


def extract_truth_anti_tau_pt(events: ak.Array) -> np.ndarray:
    px = truth_array(events, "truth_anti_tau_px")
    py = truth_array(events, "truth_anti_tau_py")
    return np.sqrt(px ** 2 + py ** 2)


def extract_truth_tau_pair_pt(events: ak.Array) -> np.ndarray:
    px = truth_array(events, "truth_tau_px") + truth_array(events, "truth_anti_tau_px")
    py = truth_array(events, "truth_tau_py") + truth_array(events, "truth_anti_tau_py")
    return np.sqrt(px ** 2 + py ** 2)


def extract_truth_tau_pair_mass(events: ak.Array) -> np.ndarray:
    px = truth_array(events, "truth_tau_px") + truth_array(events, "truth_anti_tau_px")
    py = truth_array(events, "truth_tau_py") + truth_array(events, "truth_anti_tau_py")
    pz = truth_array(events, "truth_tau_pz") + truth_array(events, "truth_anti_tau_pz")
    energy = truth_array(events, "truth_tau_E") + truth_array(events, "truth_anti_tau_E")
    return invariant_mass(px, py, pz, energy)


def extract_truth_nunu_pt(events: ak.Array) -> np.ndarray:
    px = truth_array(events, "truth_nu_tau_px") + truth_array(events, "truth_anti_nu_tau_px")
    py = truth_array(events, "truth_nu_tau_py") + truth_array(events, "truth_anti_nu_tau_py")
    return np.sqrt(px ** 2 + py ** 2)


def extract_target_invisible_pt(events: ak.Array) -> np.ndarray:
    return extract_target_invisible_observable(events, "pt")


def extract_target_invisible_energy(events: ak.Array) -> np.ndarray:
    return extract_target_invisible_observable(events, "energy")


def extract_target_invisible_eta(events: ak.Array) -> np.ndarray:
    return extract_target_invisible_observable(events, "eta")


def extract_target_invisible_phi(events: ak.Array) -> np.ndarray:
    return extract_target_invisible_observable(events, "phi")


def extract_target_invisible_mass(events: ak.Array) -> np.ndarray:
    return extract_target_invisible_observable(events, "mass")


def default_plot_specs() -> list[PlotSpec]:
    specs = [
        PlotSpec(
            name="isolation_angle",
            x_label="Isolation angle [deg]",
            title="Isolation angle",
            bins=np.linspace(140, 180, 81),
            extractor=extract_isolation_angle,
        ),
        PlotSpec(
            name="erad",
            x_label="E_rad",
            title="E_rad",
            bins=np.linspace(0, 2, 81),
            extractor=extract_erad,
        ),
        PlotSpec(
            name="prad",
            x_label="P_rad",
            title="P_rad",
            bins=np.linspace(0, 2, 81),
            extractor=extract_prad,
        ),
        PlotSpec(
            name="charged_e",
            x_label="Charged energy [GeV]",
            title="Charged energy",
            bins=np.linspace(0, 100, 81),
            extractor=extract_charged_e,
        ),
        PlotSpec(
            name="missing_pt",
            x_label="Missing pT [GeV]",
            title="Missing pT",
            bins=np.linspace(0, 100, 81),
            extractor=extract_missing_pt,
        ),
        PlotSpec(
            name="nprong",
            x_label="nprong",
            title="nprong",
            bins=np.arange(1.5, 7.6, 1.0),
            extractor=extract_nprong,
            log_scale=False,
        ),
        PlotSpec(
            name="n_neutral",
            x_label="Number of neutral particles",
            title="Number of neutral particles",
            bins=np.arange(-0.5, 10.6, 1.0),
            extractor=extract_n_neutral,
            log_scale=False,
        ),
        PlotSpec(
            name="thrust_neglog1m",
            x_label="-log10(1 - thrust)",
            title="-log10(1 - thrust)",
            bins=np.linspace(0, 10, 81),
            extractor=extract_thrust_neglog1m,
        ),
        PlotSpec(
            name="tau_vis_prong_energy",
            x_label="Visible tau energy from prongs [GeV]",
            title="Visible tau energy from prongs",
            bins=None,
            extractor=extract_tau_vis_prong_energy,
        ),
        PlotSpec(
            name="tau_vis_prong_pt",
            x_label="Visible tau pT from prongs [GeV]",
            title="Visible tau pT from prongs",
            bins=None,
            extractor=extract_tau_vis_prong_pt,
        ),
        PlotSpec(
            name="tau_vis_prong_eta",
            x_label="Visible tau eta from prongs",
            title="Visible tau eta from prongs",
            bins=np.linspace(-5, 5, 81),
            extractor=extract_tau_vis_prong_eta,
            log_scale=False,
        ),
        PlotSpec(
            name="tau_vis_prong_phi",
            x_label="Visible tau phi from prongs",
            title="Visible tau phi from prongs",
            bins=np.linspace(-np.pi, np.pi, 81),
            extractor=extract_tau_vis_prong_phi,
            log_scale=False,
        ),
        PlotSpec(
            name="tau_vis_prong_mass",
            x_label="Visible tau mass from prongs [GeV]",
            title="Visible tau mass from prongs",
            bins=np.linspace(0, 3, 81),
            extractor=extract_tau_vis_prong_mass,
            log_scale=False,
        ),
        PlotSpec(
            name="tau_vis_rho_energy",
            x_label="Visible tau energy from prongs + nearby photons [GeV]",
            title="Visible tau energy from prongs + nearby photons",
            bins=None,
            extractor=extract_tau_vis_rho_energy,
        ),
        PlotSpec(
            name="tau_vis_rho_pt",
            x_label="Visible tau pT from prongs + nearby photons [GeV]",
            title="Visible tau pT from prongs + nearby photons",
            bins=None,
            extractor=extract_tau_vis_rho_pt,
        ),
        PlotSpec(
            name="tau_vis_rho_eta",
            x_label="Visible tau eta from prongs + nearby photons",
            title="Visible tau eta from prongs + nearby photons",
            bins=np.linspace(-5, 5, 81),
            extractor=extract_tau_vis_rho_eta,
            log_scale=False,
        ),
        PlotSpec(
            name="tau_vis_rho_phi",
            x_label="Visible tau phi from prongs + nearby photons",
            title="Visible tau phi from prongs + nearby photons",
            bins=np.linspace(-np.pi, np.pi, 81),
            extractor=extract_tau_vis_rho_phi,
            log_scale=False,
        ),
        PlotSpec(
            name="tau_vis_rho_mass",
            x_label="Visible tau mass from prongs + nearby photons [GeV]",
            title="Visible tau mass from prongs + nearby photons",
            bins=np.linspace(0, 3, 81),
            extractor=extract_tau_vis_rho_mass,
            log_scale=False,
        ),
    ]

    for observable in FOUR_VECTOR_FEATURES:
        specs.append(
            PlotSpec(
                name=f"Part_{observable}",
                x_label=f"Part_{observable}",
                title=f"Part_{observable}",
                bins=None,
                extractor=lambda events, observable=observable: extract_part_momentum_observable(events, observable),
                log_scale=observable not in {"eta", "phi"},
            )
        )
    for field_name in PART_AUX_FIELDS:
        specs.append(
            PlotSpec(
                name=field_name,
                x_label=field_name,
                title=field_name,
                bins=None,
                extractor=lambda events, field_name=field_name: extract_part_feature(events, field_name),
                log_scale=False,
            )
        )
    return specs


def default_truth_plot_specs() -> list[PlotSpec]:
    return [
        PlotSpec(
            name="truth_tau_pt",
            x_label="Truth tau pT [GeV]",
            title="Truth tau pT",
            bins=np.linspace(0, 60, 81),
            extractor=extract_truth_tau_pt,
        ),
        PlotSpec(
            name="truth_anti_tau_pt",
            x_label="Truth anti-tau pT [GeV]",
            title="Truth anti-tau pT",
            bins=np.linspace(0, 60, 81),
            extractor=extract_truth_anti_tau_pt,
        ),
        PlotSpec(
            name="truth_tau_pair_pt",
            x_label="Truth tau-pair pT [GeV]",
            title="Truth tau-pair pT",
            bins=np.linspace(0, 30, 81),
            extractor=extract_truth_tau_pair_pt,
            log_scale=False,
        ),
        PlotSpec(
            name="truth_tau_pair_mass",
            x_label="Truth tau-pair mass [GeV]",
            title="Truth tau-pair mass",
            bins=np.linspace(0, 120, 81),
            extractor=extract_truth_tau_pair_mass,
        ),
        PlotSpec(
            name="truth_nunu_pt",
            x_label="Truth neutrino-pair pT [GeV]",
            title="Truth neutrino-pair pT",
            bins=np.linspace(0, 30, 81),
            extractor=extract_truth_nunu_pt,
            log_scale=False,
        ),
        PlotSpec(
            name="target_invisible_energy",
            x_label="Target invisible energy [GeV]",
            title="Target invisible energy",
            bins=None,
            extractor=extract_target_invisible_energy,
            log_scale=False,
        ),
        PlotSpec(
            name="target_invisible_pt",
            x_label="Target invisible pT [GeV]",
            title="Target invisible pT",
            bins=None,
            extractor=extract_target_invisible_pt,
            log_scale=False,
        ),
        PlotSpec(
            name="target_invisible_eta",
            x_label="Target invisible eta",
            title="Target invisible eta",
            bins=np.linspace(-5, 5, 81),
            extractor=extract_target_invisible_eta,
            log_scale=False,
        ),
        PlotSpec(
            name="target_invisible_phi",
            x_label="Target invisible phi",
            title="Target invisible phi",
            bins=np.linspace(-np.pi, np.pi, 81),
            extractor=extract_target_invisible_phi,
            log_scale=False,
        ),
        PlotSpec(
            name="target_invisible_mass",
            x_label="Target invisible mass [GeV]",
            title="Target invisible mass",
            bins=np.linspace(0, 3, 81),
            extractor=extract_target_invisible_mass,
            log_scale=False,
        ),
    ]


def make_control_plots(
    config_path: Path,
    output_dir: Path,
    luminosity: float | None = None,
    normalize: bool | None = None,
):
    samples, subcategories = parse_plot_config(config_path)
    console.print(f"[bold]Using config[/bold] [white]{config_path}[/white]")
    loaded_events = {sample_key: load_sample_events(sample) for sample_key, sample in samples.items()}
    expanded_samples = expand_samples(samples, loaded_events, subcategories)

    resolved_luminosity = infer_luminosity(samples, luminosity)
    if normalize is None:
        normalize = resolved_luminosity is None

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]Output directory[/bold] [white]{output_dir}[/white]")
    console.print(
        f"[bold]Normalization mode[/bold] "
        f"{'shape-only' if normalize else 'absolute-yield'}"
        + (
            f" (luminosity={resolved_luminosity} pb^-1)"
            if resolved_luminosity is not None
            else ""
        )
    )

    sample_table = Table(title="Expanded Samples")
    sample_table.add_column("Sample")
    sample_table.add_column("Type")
    sample_table.add_column("Selected events", justify="right")
    sample_table.add_column("Initial events", justify="right")
    for sample, sample_events in expanded_samples:
        sample_table.add_row(
            sample.name,
            "data" if sample.is_data else ("signal" if sample.is_signal else "background"),
            str(len(sample_events)),
            str(get_initial_total_num_events(sample_events)),
        )
    console.print(sample_table)

    for plot_spec in default_plot_specs():
        console.print(f"[bold cyan]Processing plot[/bold cyan] [white]{plot_spec.name}[/white]")
        bins = plot_spec.bins
        if bins is None:
            values_by_sample = {
                sample.name: sanitize_hist_values(plot_spec.extractor(sample_events))
                for sample, sample_events in expanded_samples
            }
            bins = choose_bins(values_by_sample)
        hist_data = np.zeros(len(bins) - 1, dtype=float)
        hist_mc: dict[str, np.ndarray] = {}
        hist_mc_err2: dict[str, np.ndarray] = {}

        for sample, sample_events in expanded_samples:
            values = sanitize_hist_values(plot_spec.extractor(sample_events))
            hist, _ = np.histogram(values, bins=bins)

            if sample.is_data:
                hist_data += hist
                continue

            scale = sample_scale(sample, sample_events, resolved_luminosity)
            hist_mc[sample.name] = hist.astype(float) * scale
            hist_mc_err2[sample.name] = hist.astype(float) * (scale ** 2)

        plot_from_histograms(
            hist_data=hist_data,
            hist_mc=hist_mc,
            hist_mc_err2=hist_mc_err2,
            bin_edges=bins,
            x_label=plot_spec.x_label,
            title=plot_spec.title,
            output_path=output_dir / f"{plot_spec.name}.png",
            normalize=normalize,
            log_scale=plot_spec.log_scale,
        )
        console.print(
            f"  [green]Wrote[/green] [white]{output_dir / f'{plot_spec.name}.png'}[/white]"
        )

    for plot_spec in default_truth_plot_specs():
        console.print(f"[bold magenta]Processing truth plot[/bold magenta] [white]{plot_spec.name}[/white]")
        hist_mc: dict[str, np.ndarray] = {}
        hist_mc_err2: dict[str, np.ndarray] = {}
        bins = plot_spec.bins

        values_by_sample: dict[str, np.ndarray] = {}
        for sample, sample_events in expanded_samples:
            if sample.is_data:
                continue

            values = sanitize_hist_values(plot_spec.extractor(sample_events))
            if values.size == 0:
                continue
            values_by_sample[sample.name] = values

        if not values_by_sample:
            console.print("  [yellow]Skipped[/yellow] no valid truth branches found")
            continue
        if bins is None:
            bins = choose_bins(values_by_sample)

        for sample, sample_events in expanded_samples:
            if sample.is_data:
                continue

            values = values_by_sample.get(sample.name)
            if values is None or values.size == 0:
                continue

            hist, _ = np.histogram(values, bins=bins)
            scale = sample_scale(sample, sample_events, resolved_luminosity)
            hist_mc[sample.name] = hist.astype(float) * scale
            hist_mc_err2[sample.name] = hist.astype(float) * (scale ** 2)

        plot_from_histograms(
            hist_data=None,
            hist_mc=hist_mc,
            hist_mc_err2=hist_mc_err2,
            bin_edges=bins,
            x_label=plot_spec.x_label,
            title=plot_spec.title,
            output_path=output_dir / f"{plot_spec.name}.png",
            normalize=normalize,
            log_scale=plot_spec.log_scale,
        )
        console.print(
            f"  [green]Wrote[/green] [white]{output_dir / f'{plot_spec.name}.png'}[/white]"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple data/MC control plots from DataLoader parquet outputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/config/analysis.yaml"),
        help="YAML config containing a Samples block.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Users/tihsu/PycharmProjects/lep_tree_ana/ml_pipeline/plots"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--luminosity",
        type=float,
        default=None,
        help="Override luminosity in pb^-1. If omitted, uses the data sample lumi when present.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Force shape-only normalization instead of absolute yield scaling.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    make_control_plots(
        config_path=args.config,
        output_dir=args.output_dir,
        luminosity=args.luminosity,
        normalize=True if args.normalize else None,
    )


if __name__ == "__main__":
    main()
