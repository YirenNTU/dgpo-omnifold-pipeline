#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import yaml


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
    for parquet_path in expand_input_files(sample.input_files):
        arrays.append(ak.from_parquet(parquet_path))

    if not arrays:
        raise ValueError(f"No parquet inputs found for sample '{sample.name}'.")
    if len(arrays) == 1:
        return arrays[0]
    return ak.concatenate(arrays, axis=0)


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


def make_ratio_axes():
    return plt.subplots(
        2,
        1,
        dpi=220,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )


def plot_from_histograms(
    hist_data: np.ndarray,
    hist_mc: dict[str, np.ndarray],
    hist_mc_err2: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    x_label: str,
    title: str,
    output_path: Path,
    normalize: bool,
    log_scale: bool,
):
    fig, (ax, ax_ratio) = make_ratio_axes()

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
    ax.set_ylabel("Normalized events" if normalize else "Events")
    ax.set_title(title)
    ax.legend(loc="best")

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
    ]


def parse_samples(config_path: Path) -> dict[str, Sample]:
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
    return samples


def make_control_plots(
    config_path: Path,
    output_dir: Path,
    luminosity: float | None = None,
    normalize: bool | None = None,
):
    samples = parse_samples(config_path)
    loaded_events = {sample_key: load_sample_events(sample) for sample_key, sample in samples.items()}

    resolved_luminosity = infer_luminosity(samples, luminosity)
    if normalize is None:
        normalize = resolved_luminosity is None

    output_dir.mkdir(parents=True, exist_ok=True)

    for plot_spec in default_plot_specs():
        hist_data = np.zeros(len(plot_spec.bins) - 1, dtype=float)
        hist_mc: dict[str, np.ndarray] = {}
        hist_mc_err2: dict[str, np.ndarray] = {}

        for sample_key, sample in samples.items():
            values = plot_spec.extractor(loaded_events[sample_key])
            hist, _ = np.histogram(values, bins=plot_spec.bins)

            if sample.is_data:
                hist_data += hist
                continue

            scale = sample_scale(sample, loaded_events[sample_key], resolved_luminosity)
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
