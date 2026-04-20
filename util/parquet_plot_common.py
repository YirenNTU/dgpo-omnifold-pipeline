from __future__ import annotations

from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


def get_initial_total_num_events(events: ak.Array) -> int:
    if "initial_total_num_events" in events.fields and len(events) > 0:
        return int(ak.to_numpy(events["initial_total_num_events"][:1])[0])
    return int(len(events))


def infer_luminosity(samples: dict, fallback: float | None) -> float | None:
    if fallback is not None:
        return fallback

    for sample in samples.values():
        if sample.is_data and sample.lumi is not None:
            return float(sample.lumi)
    return None


def sample_scale(sample, events: ak.Array, luminosity: float | None) -> float:
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


def sanitize_hist_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def summarize_invalid_hist_values(values_by_sample: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for sample_name, values in values_by_sample.items():
        values = np.asarray(values, dtype=float)
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        if nan_count > 0 or inf_count > 0:
            summary[sample_name] = {"nan": nan_count, "inf": inf_count}
    return summary


def choose_bins(values_by_sample: dict[str, np.ndarray], num_bins: int = 60) -> np.ndarray:
    non_empty = [sanitize_hist_values(values) for values in values_by_sample.values() if np.asarray(values).size > 0]
    non_empty = [values for values in non_empty if values.size > 0]
    if not non_empty:
        return np.linspace(0, 1, 2)

    merged = np.concatenate(non_empty)
    rounded = np.round(merged)
    if np.allclose(merged, rounded, atol=1e-6) and np.unique(rounded).size <= 30:
        low = int(np.min(rounded))
        high = int(np.max(rounded))
        return np.arange(low - 0.5, high + 1.5, 1.0)

    low = float(np.min(merged))
    high = float(np.max(merged))
    if low == high:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, num_bins + 1)


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
    invalid_summary: dict[str, dict[str, int]] | None = None,
):
    def render_single_plot(path: Path, use_log_scale: bool) -> None:
        has_data = hist_data is not None
        if has_data:
            fig, (ax, ax_ratio) = make_ratio_axes()
        else:
            fig, ax = plt.subplots(1, 1, dpi=220, figsize=(8, 5))
            ax_ratio = None

        num_bins = len(bin_edges) - 1
        total_mc = np.zeros(num_bins, dtype=float)
        total_mc_err2 = np.zeros(num_bins, dtype=float)

        use_normalize = normalize
        sum_mc_yields = sum(float(np.sum(hist)) for hist in hist_mc.values())
        if use_normalize and sum_mc_yields <= 0:
            use_normalize = False

        colors = plt.cm.tab10.colors
        for index, sample_name in enumerate(hist_mc):
            hist = hist_mc[sample_name].astype(float)
            err2 = hist_mc_err2[sample_name].astype(float)
            if use_normalize:
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

        ax.set_ylabel("Normalized events" if use_normalize else "Events")
        ax.set_title(title)
        ax.set_xlabel(x_label if not has_data else "")

        if has_data:
            data_hist = hist_data.astype(float)
            data_total = float(np.sum(data_hist))
            if use_normalize and data_total > 0:
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

        if invalid_summary:
            summary_lines = ["Dropped invalid entries:"]
            for sample_name, counts in invalid_summary.items():
                parts = []
                if counts.get("nan", 0) > 0:
                    parts.append(f"NaN={counts['nan']}")
                if counts.get("inf", 0) > 0:
                    parts.append(f"Inf={counts['inf']}")
                if parts:
                    summary_lines.append(f"{sample_name}: {', '.join(parts)}")
            if len(summary_lines) > 1:
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(summary_lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
                )

        if use_log_scale:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-1 if use_normalize else 1)
        else:
            ax.set_ylim(bottom=0)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    render_single_plot(output_path, log_scale)
    if not log_scale:
        log_output_path = output_path.with_name(f"{output_path.stem}_log{output_path.suffix}")
        render_single_plot(log_output_path, True)
