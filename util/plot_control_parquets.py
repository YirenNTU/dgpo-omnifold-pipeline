#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml
from evenet_parquet_common import build_tau_targets, build_visible_tau_assumptions
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
    bins: np.ndarray
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


def get_initial_total_num_events(events: ak.Array) -> int:
    if "initial_total_num_events" in events.fields and len(events) > 0:
        return int(ak.to_numpy(events["initial_total_num_events"][:1])[0])
    return int(len(events))


def infer_luminosity(samples: dict[str, Sample], fallback: float | None) -> float | None:
    if fallback is not None:
        return fallback

    for sample in samples.values():
        if sample.is_data and sample.lumi is not None:
            return float(sample.lumi)
    return None


def sample_scale(sample: Sample, events: ak.Array, luminosity: float | None) -> float:
    if sample.is_data:
        return 1.0

    if luminosity is None:
        return 1.0

    initial_total_num_events = get_initial_total_num_events(events)
    if initial_total_num_events <= 0:
        raise ValueError(f"Sample '{sample.name}' has non-positive initial_total_num_events.")
    return sample.norm_factor / initial_total_num_events * luminosity


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


def make_ratio_axes():
    return plt.subplots(
        2,
        1,
        dpi=220,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )


def sanitize_hist_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def plot_from_histograms(
    hist_data: np.ndarray | None,
    hist_mc: dict[str, np.ndarray],
    hist_mc_err2: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    x_label: str,
    title: str,
    output_path: Path,
    normalize: bool,
    log_scale: bool,
):
    has_data = hist_data is not None
    if has_data:
        fig, (ax, ax_ratio) = make_ratio_axes()
    else:
        fig, ax = plt.subplots(1, 1, dpi=220, figsize=(8, 5))
        ax_ratio = None

    num_bins = len(bin_edges) - 1
    total_mc = np.zeros(num_bins, dtype=float)
    total_mc_err2 = np.zeros(num_bins, dtype=float)

    sum_mc_yields = sum(float(np.sum(hist)) for hist in hist_mc.values())
    if normalize and sum_mc_yields <= 0:
        normalize = False

    colors = plt.cm.tab10.colors
    for index, sample_name in enumerate(hist_mc):
        hist = hist_mc[sample_name].astype(float)
        err2 = hist_mc_err2[sample_name].astype(float)
        if normalize:
            hist = hist / sum_mc_yields
            err2 = err2 / (sum_mc_yields ** 2)

        ax.bar(
            bin_edges[:-1],
            hist,
            bottom=total_mc,
            width=np.diff(bin_edges),
            align="edge",
            label=sample_name,
            color=colors[index % len(colors)],
            alpha=0.75,
            edgecolor="black",
        )
        total_mc += hist
        total_mc_err2 += err2

    total_mc_err = np.sqrt(total_mc_err2)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.fill_between(
        bin_centers,
        total_mc - total_mc_err,
        total_mc + total_mc_err,
        step="mid",
        color="gray",
        alpha=0.35,
        label="MC unc.",
    )

    ax.set_ylabel("Normalized events" if normalize else "Events")
    ax.set_title(title)
    ax.set_xlabel(x_label if not has_data else "")

    if has_data:
        data_hist = hist_data.astype(float)
        data_total = float(np.sum(data_hist))
        if normalize and data_total > 0:
            data_plot = data_hist / data_total
            data_err = np.sqrt(data_hist) / data_total
        else:
            data_plot = data_hist
            data_err = np.sqrt(data_hist)

        ax.errorbar(
            bin_centers,
            data_plot,
            yerr=data_err,
            fmt="o",
            color="black",
            label="Data",
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(
                data_plot,
                total_mc,
                out=np.full_like(data_plot, np.nan, dtype=float),
                where=total_mc > 0,
            )
            ratio_err = np.divide(
                data_err,
                total_mc,
                out=np.full_like(data_err, np.nan, dtype=float),
                where=total_mc > 0,
            )

        ax_ratio.errorbar(bin_centers, ratio, yerr=ratio_err, fmt="o", color="black")
        ax_ratio.axhline(1.0, color="gray", linestyle=":")
        ax_ratio.set_ylabel("Data/MC")
        ax_ratio.set_xlabel(x_label)
        ax_ratio.set_ylim(0.5, 1.5)

    ax.legend(loc="best")

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-1 if normalize else 1)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


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
    tau_vis_prong, tau_vis_prong_mask, _, _ = build_visible_tau_assumptions(events)
    return tau_vis_prong[..., 1][tau_vis_prong_mask]


def extract_tau_vis_rho_pt(events: ak.Array) -> np.ndarray:
    _, _, tau_vis_rho, tau_vis_rho_mask = build_visible_tau_assumptions(events)
    return tau_vis_rho[..., 1][tau_vis_rho_mask]


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
    tau_vis_prong, tau_vis_prong_mask, tau_vis_rho, tau_vis_rho_mask = build_visible_tau_assumptions(events)
    x_invisible, x_invisible_mask, _, _, _, _ = build_tau_targets(
        events,
        tau_vis_prong,
        tau_vis_prong_mask,
        tau_vis_rho,
        tau_vis_rho_mask,
    )
    return x_invisible[..., 1][x_invisible_mask]


def default_plot_specs() -> list[PlotSpec]:
    return [
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
            name="tau_vis_prong_pt",
            x_label="Visible tau pT from prongs [GeV]",
            title="Visible tau pT from prongs",
            bins=np.linspace(0, 60, 81),
            extractor=extract_tau_vis_prong_pt,
        ),
        PlotSpec(
            name="tau_vis_rho_pt",
            x_label="Visible tau pT from prongs + nearby photons [GeV]",
            title="Visible tau pT from prongs + nearby photons",
            bins=np.linspace(0, 60, 81),
            extractor=extract_tau_vis_rho_pt,
        ),
    ]


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
            name="target_invisible_pt",
            x_label="Target invisible pT [GeV]",
            title="Target invisible pT",
            bins=np.linspace(0, 30, 81),
            extractor=extract_target_invisible_pt,
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
        hist_data = np.zeros(len(plot_spec.bins) - 1, dtype=float)
        hist_mc: dict[str, np.ndarray] = {}
        hist_mc_err2: dict[str, np.ndarray] = {}

        for sample, sample_events in expanded_samples:
            values = sanitize_hist_values(plot_spec.extractor(sample_events))
            hist, _ = np.histogram(values, bins=plot_spec.bins)

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
            bin_edges=plot_spec.bins,
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

        for sample, sample_events in expanded_samples:
            if sample.is_data:
                continue

            values = sanitize_hist_values(plot_spec.extractor(sample_events))
            if values.size == 0:
                continue

            hist, _ = np.histogram(values, bins=plot_spec.bins)
            scale = sample_scale(sample, sample_events, resolved_luminosity)
            hist_mc[sample.name] = hist.astype(float) * scale
            hist_mc_err2[sample.name] = hist.astype(float) * (scale ** 2)

        if not hist_mc:
            console.print("  [yellow]Skipped[/yellow] no valid truth branches found")
            continue

        plot_from_histograms(
            hist_data=None,
            hist_mc=hist_mc,
            hist_mc_err2=hist_mc_err2,
            bin_edges=plot_spec.bins,
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
