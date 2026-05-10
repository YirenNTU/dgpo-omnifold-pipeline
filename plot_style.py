from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

process_color_map = "tab20"


def set_plot_style() -> None:
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))


def _flat_finite_pair(
    truth: Any,
    pred: Any,
    weight: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth_values = to_numpy(truth)
    pred_values = to_numpy(pred)
    n_values = min(len(truth_values), len(pred_values))
    truth_values = truth_values[:n_values]
    pred_values = pred_values[:n_values]
    mask = np.isfinite(truth_values) & np.isfinite(pred_values)
    weights = None
    if weight is not None:
        weights = to_numpy(weight)[:n_values]
        mask &= np.isfinite(weights)
        weights = weights[mask]
    return truth_values[mask], pred_values[mask], weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.mean(values))
    weight_sum = float(np.sum(weights))
    if weight_sum == 0:
        return float("nan")
    return float(np.sum(weights * values) / weight_sum)


def prediction_metrics(
    truth: Any,
    pred: Any,
    *,
    bins: int | np.ndarray = 60,
    weight: Any | None = None,
) -> dict[str, float]:
    truth_values, pred_values, weights = _flat_finite_pair(truth, pred, weight)
    if truth_values.size == 0:
        return {"entries": 0.0, "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "jsd": float("nan")}

    residual = pred_values - truth_values
    mae = _weighted_mean(np.abs(residual), weights)
    rmse = math.sqrt(_weighted_mean(residual * residual, weights))
    if weights is None:
        pearson = float(np.corrcoef(truth_values, pred_values)[0, 1]) if truth_values.size > 1 else float("nan")
    else:
        mean_truth = _weighted_mean(truth_values, weights)
        mean_pred = _weighted_mean(pred_values, weights)
        weight_sum = float(np.sum(weights))
        cov = float(np.sum(weights * (truth_values - mean_truth) * (pred_values - mean_pred)) / weight_sum)
        var_truth = float(np.sum(weights * (truth_values - mean_truth) ** 2) / weight_sum)
        var_pred = float(np.sum(weights * (pred_values - mean_pred) ** 2) / weight_sum)
        pearson = cov / math.sqrt(var_truth * var_pred) if var_truth > 0 and var_pred > 0 else float("nan")

    edges = auto_bins([truth_values, pred_values], n_bins=bins) if isinstance(bins, int) else np.asarray(bins, dtype=float)
    truth_hist, _ = np.histogram(truth_values, bins=edges, weights=weights)
    pred_hist, _ = np.histogram(pred_values, bins=edges, weights=weights)
    truth_prob = truth_hist.astype(float)
    pred_prob = pred_hist.astype(float)
    if np.sum(truth_prob) > 0:
        truth_prob /= np.sum(truth_prob)
    if np.sum(pred_prob) > 0:
        pred_prob /= np.sum(pred_prob)
    mixture = 0.5 * (truth_prob + pred_prob)
    truth_terms = truth_prob > 0
    pred_terms = pred_prob > 0
    kl_truth = float(np.sum(truth_prob[truth_terms] * np.log(truth_prob[truth_terms] / mixture[truth_terms])))
    kl_pred = float(np.sum(pred_prob[pred_terms] * np.log(pred_prob[pred_terms] / mixture[pred_terms])))
    jsd = 0.5 * (kl_truth + kl_pred)

    return {
        "entries": float(truth_values.size),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "jsd": float(jsd),
    }


def plot_truth_prediction_bundle(
    truth: Any,
    pred: Any,
    output_dir: str | Path,
    name: str,
    *,
    bins: int | np.ndarray = 60,
    weight: Any | None = None,
    truth_label: str = "Truth",
    pred_label: str = "Prediction",
    xaxis_label: str | None = None,
    title: str | None = None,
    summary_title: str | None = None,
    total_entries: int | None = None,
    extra_predictions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write summary and diagnostics plots for one truth/pred pair.

    extra_predictions are overlaid only in the 1D, residual, relative residual, and profile panels.
    """
    import json
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    set_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    xaxis_label = xaxis_label or name
    title = title or name
    summary_title = summary_title or f"{title} summary"
    total_entries = int(total_entries) if total_entries is not None else min(len(to_numpy(truth)), len(to_numpy(pred)))
    truth_values, pred_values, weights = _flat_finite_pair(truth, pred, weight)
    metrics = prediction_metrics(truth_values, pred_values, bins=bins, weight=weights)
    valid_entries = int(metrics["entries"])
    metrics["total_entries"] = float(total_entries)
    metrics["valid_entries"] = float(valid_entries)
    metrics["valid_ratio"] = float(valid_entries / total_entries) if total_entries else float("nan")
    if truth_values.size == 0:
        metrics_path = output_dir / f"{name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        return {"metrics": metrics, "metrics_json": str(metrics_path), "plots": {}}

    edges = auto_bins([truth_values, pred_values], n_bins=bins) if isinstance(bins, int) else np.asarray(bins, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    paths: dict[str, str] = {}

    fig, (axis, text_axis) = plt.subplots(
        1,
        2,
        figsize=(10.2, 5.8),
        gridspec_kw={"width_ratios": [3.2, 1.35], "wspace": 0.16},
    )
    counts, x_edges, y_edges = np.histogram2d(truth_values, pred_values, bins=(edges, edges), weights=weights)
    mesh = axis.pcolormesh(x_edges, y_edges, np.ma.masked_where(counts.T <= 0, counts.T), cmap="viridis")
    axis.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="white", linestyle="--", linewidth=1.1)
    axis.set_xlabel(truth_label)
    axis.set_ylabel(pred_label)
    axis.set_title(title, loc="left")
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label("Weighted entries")
    text_axis.axis("off")
    text_axis.text(0.0, 0.98, summary_title, ha="left", va="top", fontsize=14, fontweight="bold")
    text_axis.text(
        0.0,
        0.80,
        "\n".join(
            [
                f"Total entries: {total_entries}",
                f"Valid entries: {valid_entries}",
                f"Valid ratio: {metrics['valid_ratio']:.3f}",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    text_axis.text(0.0, 0.56, "Metrics", ha="left", va="top", fontsize=12, fontweight="bold")
    text_axis.text(
        0.0,
        0.45,
        "\n".join(
            [
                f"RMSE: {metrics['rmse']:.4g}",
                f"MAE: {metrics['mae']:.4g}",
                f"Pearson: {metrics['pearson']:.4g}",
                f"JSD: {metrics['jsd']:.4g}",
            ]
        ),
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    output_path = output_dir / f"{name}_summary.png"
    save_figure(fig, output_path)
    plt.close(fig)
    paths["summary"] = str(output_path)

    extra_predictions = extra_predictions or {}
    prediction_map = {pred_label: pred}
    prediction_map.update(extra_predictions)
    clean_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {
        label: _flat_finite_pair(truth, values, weight)
        for label, values in prediction_map.items()
    }
    all_distribution_values = [truth_values]
    all_distribution_values.extend(values for _, values, _ in clean_predictions.values())
    distribution_edges = auto_bins(all_distribution_values, n_bins=bins) if isinstance(bins, int) else edges

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.2),
        gridspec_kw={"height_ratios": [1.0, 1.0], "wspace": 0.28, "hspace": 0.34},
    )
    dist_axis, residual_axis = axes[0]
    profile_axis, relative_axis = axes[1]

    dist_axis.hist(truth_values, bins=distribution_edges, weights=weights, histtype="step", linewidth=1.6, label=truth_label)
    for label, (_, values, clean_weight) in clean_predictions.items():
        dist_axis.hist(values, bins=distribution_edges, weights=clean_weight, histtype="step", linewidth=1.4, label=label)
    dist_axis.set_ylabel("Weighted entries")
    dist_axis.set_title("1D distributions", loc="left")
    dist_axis.legend(frameon=False, fontsize=9)

    residual_n_bins = bins if isinstance(bins, int) else max(len(np.asarray(bins, dtype=float)) - 1, 1)
    residual_lines = []
    relative_lines = []
    for label, (clean_truth, clean_pred, clean_weight) in clean_predictions.items():
        residual = clean_pred - clean_truth
        relative = np.divide(residual, clean_truth, out=np.full_like(residual, np.nan), where=np.abs(clean_truth) > 1e-12)
        finite_residual = np.isfinite(residual)
        if np.any(finite_residual):
            residual_axis.hist(
                residual[finite_residual],
                bins=auto_bins([residual[finite_residual]], n_bins=residual_n_bins),
                weights=None if clean_weight is None else clean_weight[finite_residual],
                histtype="step",
                linewidth=1.3,
                label=label,
            )
            residual_lines.append(
                f"{label}: mean={np.nanmean(residual[finite_residual]):.3g}, std={np.nanstd(residual[finite_residual]):.3g}"
            )
        finite_relative = np.isfinite(relative)
        if np.any(finite_relative):
            relative_axis.hist(
                relative[finite_relative],
                bins=auto_bins([relative[finite_relative]], n_bins=residual_n_bins),
                weights=None if clean_weight is None else clean_weight[finite_relative],
                histtype="step",
                linewidth=1.3,
                label=label,
            )
            relative_lines.append(
                f"{label}: mean={np.nanmean(relative[finite_relative]):.3g}, std={np.nanstd(relative[finite_relative]):.3g}"
            )

        profile_mean = np.full(len(centers), np.nan)
        profile_err = np.full(len(centers), np.nan)
        for index in range(len(centers)):
            in_bin = (clean_truth >= edges[index]) & (clean_truth < edges[index + 1])
            if index == len(centers) - 1:
                in_bin = (clean_truth >= edges[index]) & (clean_truth <= edges[index + 1])
            if not np.any(in_bin):
                continue
            bin_residual = residual[in_bin]
            bin_weights = None if clean_weight is None else clean_weight[in_bin]
            profile_mean[index] = _weighted_mean(bin_residual, bin_weights)
            if bin_weights is None:
                profile_err[index] = np.nanstd(bin_residual) / math.sqrt(max(np.sum(np.isfinite(bin_residual)), 1))
            else:
                mean = profile_mean[index]
                weight_sum = np.sum(bin_weights)
                variance = np.sum(bin_weights * (bin_residual - mean) ** 2) / weight_sum if weight_sum > 0 else np.nan
                profile_err[index] = math.sqrt(variance / max(np.sum(in_bin), 1)) if np.isfinite(variance) else np.nan
        valid_profile = np.isfinite(profile_mean)
        if np.any(valid_profile):
            profile_axis.errorbar(
                centers[valid_profile],
                profile_mean[valid_profile],
                yerr=profile_err[valid_profile],
                fmt="o",
                markersize=3,
                linewidth=0.9,
                label=label,
            )

    residual_axis.axvline(0.0, color="gray", linestyle="--", linewidth=0.9)
    residual_axis.set_xlabel(f"Prediction - {truth_label}")
    residual_axis.set_ylabel("Weighted entries")
    residual_axis.set_title("Residuals", loc="left")
    residual_axis.legend(frameon=False, fontsize=9)
    if residual_lines:
        residual_axis.text(
            0.98,
            0.96,
            "\n".join(residual_lines),
            transform=residual_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
        )

    profile_axis.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    profile_axis.set_xlabel(truth_label)
    profile_axis.set_ylabel(f"Prediction - {truth_label}")
    profile_axis.set_title("Residual profile", loc="left")
    profile_axis.legend(frameon=False, fontsize=9)

    relative_axis.axvline(0.0, color="gray", linestyle="--", linewidth=0.9)
    relative_axis.set_xlabel(f"(Prediction - {truth_label}) / {truth_label}")
    relative_axis.set_ylabel("Weighted entries")
    relative_axis.set_title("Relative residuals", loc="left")
    relative_axis.legend(frameon=False, fontsize=9)
    if relative_lines:
        relative_axis.text(
            0.98,
            0.96,
            "\n".join(relative_lines),
            transform=relative_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
        )

    fig.suptitle(title, x=0.08, ha="left", fontsize=13, fontweight="bold")
    output_path = output_dir / f"{name}_diagnostics.png"
    save_figure(fig, output_path)
    plt.close(fig)
    paths["diagnostics"] = str(output_path)

    metrics_path = output_dir / f"{name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return {"metrics": metrics, "metrics_json": str(metrics_path), "plots": paths}


def plot_data_mc_histogram_from_counts(
    bins: np.ndarray,
    data_counts: np.ndarray,
    data_sumw2: np.ndarray,
    mc_counts: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Weighted entries",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    set_plot_style()
    bins = np.asarray(bins, dtype=float)
    data_counts = np.asarray(data_counts, dtype=float)
    data_sumw2 = np.asarray(data_sumw2, dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)

    fig, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.1], "hspace": 0.05},
    )

    cmap = plt.get_cmap(process_color_map)
    colors = cmap(np.linspace(0.0, 1.0, max(len(mc_counts), 1)))
    mc_sum = np.zeros_like(data_counts, dtype=float)
    stack_base = np.zeros_like(data_counts, dtype=float)

    for color, (label, counts) in zip(colors, mc_counts.items()):
        counts = np.asarray(counts, dtype=float)
        axis.bar(
            bins[:-1],
            counts,
            width=widths,
            bottom=stack_base,
            align="edge",
            alpha=0.82,
            color=color,
            edgecolor="black",
            linewidth=0.25,
            label=label,
        )
        stack_base += counts
        mc_sum += counts

    data_errors = np.sqrt(np.clip(data_sumw2, 0.0, None))
    axis.errorbar(
        centers,
        data_counts,
        yerr=data_errors,
        fmt="o",
        color="black",
        markersize=3.5,
        linewidth=0.9,
        label="data",
    )

    ratio = np.divide(data_counts, mc_sum, out=np.full_like(data_counts, np.nan), where=mc_sum != 0)
    ratio_error = np.divide(data_errors, mc_sum, out=np.full_like(data_errors, np.nan), where=mc_sum != 0)
    ratio_axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ratio_axis.errorbar(
        centers,
        ratio,
        yerr=ratio_error,
        fmt="o",
        color="black",
        markersize=3.0,
        linewidth=0.8,
    )

    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    ratio_axis.set_ylabel("Data / MC")
    ratio_axis.set_xlabel(xlabel)
    ratio_axis.set_ylim(0.0, 2.0)
    ratio_axis.grid(axis="y", alpha=0.22)

    save_figure(fig, output_path)
    plt.close(fig)
    return str(output_path)


def plot_2d_histogram_comparison(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    counts: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    entries: int | None = None,
    correlation: float | None = None,
    zlabel: str = "Weighted entries",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    set_plot_style()
    counts = np.asarray(counts, dtype=float).T
    masked_counts = np.ma.masked_where(counts <= 0, counts)

    fig, axis = plt.subplots(figsize=(6.6, 5.8))
    mesh = axis.pcolormesh(x_edges, y_edges, masked_counts, cmap="viridis", norm=LogNorm())
    low = max(float(x_edges[0]), float(y_edges[0]))
    high = min(float(x_edges[-1]), float(y_edges[-1]))
    axis.plot([low, high], [low, high], color="white", linestyle="--", linewidth=1.1, alpha=0.9)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.set_ylim(y_edges[0], y_edges[-1])
    axis.set_aspect("equal", adjustable="box")

    stat_lines = []
    if entries is not None:
        stat_lines.append(f"entries={entries}")
    if correlation is not None and np.isfinite(correlation):
        stat_lines.append(f"r={correlation:.3f}")
    if stat_lines:
        axis.text(
            0.03,
            0.97,
            "\n".join(stat_lines),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )

    colorbar = fig.colorbar(mesh, ax=axis, pad=0.02)
    colorbar.set_label(zlabel)
    save_figure(fig, output_path)
    plt.close(fig)
    return str(output_path)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, ak.Array):
        x = ak.to_numpy(ak.flatten(x, axis=None))
    return np.asarray(x, dtype=float).reshape(-1)


def finite_values(values: Any, weights: Any | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    values = to_numpy(values)

    if weights is None:
        mask = np.isfinite(values)
        return values[mask], None

    weights = to_numpy(weights)
    if len(values) != len(weights):
        raise ValueError(f"value/weight length mismatch: {len(values)} vs {len(weights)}")

    mask = np.isfinite(values) & np.isfinite(weights)
    return values[mask], weights[mask]


def auto_bins(
    arrays: list[Any],
    *,
    n_bins: int = 50,
    quantile_range: tuple[float, float] = (0.005, 0.995),
    integer: bool | None = None,
) -> np.ndarray:
    values = []
    for arr in arrays:
        arr = to_numpy(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)

    if not values:
        return np.linspace(0.0, 1.0, n_bins + 1)

    merged = np.concatenate(values)

    if integer is None:
        unique = np.unique(merged)
        integer = len(unique) <= 40 and np.allclose(unique, np.round(unique))

    if integer:
        lo = int(np.nanmin(merged))
        hi = int(np.nanmax(merged))
        return np.arange(lo - 0.5, hi + 1.5, 1.0)

    lo, hi = np.quantile(merged, quantile_range)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(merged)), float(np.nanmax(merged))

    if lo == hi:
        pad = abs(lo) if lo else 1.0
        lo -= 0.5 * pad
        hi += 0.5 * pad

    pad = 0.05 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def plot_data_mc_comparison(
    data: ak.Array,
    mc: dict[str, ak.Array],
    bin: int | np.ndarray,
    xaxis_label: str,
    yaxis_label: str,
    *,
    data_weight: ak.Array | None = None,
    mc_weight: dict[str, ak.Array] | None = None,
    title: str | None = None,
) -> Any:
    """Data vs stacked MC with ratio panel.

    data: ak.Array
    mc: dict[str, ak.Array]
    """
    from matplotlib import pyplot as plt

    set_plot_style()

    bins = bin
    labels = list(mc)

    data_values, data_weights = finite_values(data, data_weight)

    mc_values = []
    mc_weights = []
    for label in labels:
        w = None if mc_weight is None else mc_weight.get(label)
        values, weights = finite_values(mc[label], w)
        mc_values.append(values)
        mc_weights.append(weights)

    if isinstance(bins, int):
        bins = auto_bins([data_values, *mc_values], n_bins=bins)

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(6.8, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
    )

    cmap = plt.get_cmap(process_color_map)
    colors = cmap(np.linspace(0.0, 1.0, max(len(labels), 1)))

    ax.hist(
        mc_values,
        bins=bins,
        weights=mc_weights,
        stacked=True,
        histtype="stepfilled",
        alpha=0.82,
        color=colors[: len(labels)],
        label=labels,
    )

    data_counts, edges = np.histogram(data_values, bins=bins, weights=data_weights)
    if data_weights is None:
        data_var, _ = np.histogram(data_values, bins=edges)
    else:
        data_var, _ = np.histogram(data_values, bins=edges, weights=data_weights * data_weights)
    data_err = np.sqrt(data_var)

    mc_counts = []
    for values, weights in zip(mc_values, mc_weights):
        counts, _ = np.histogram(values, bins=edges, weights=weights)
        mc_counts.append(counts.astype(float))

    mc_sum = np.sum(mc_counts, axis=0) if mc_counts else np.zeros_like(data_counts, dtype=float)
    centers = 0.5 * (edges[1:] + edges[:-1])

    ax.errorbar(
        centers,
        data_counts,
        yerr=data_err,
        fmt="o",
        color="black",
        markersize=3.0,
        linewidth=0.8,
        label="Data",
    )

    ratio = np.divide(data_counts, mc_sum, out=np.full_like(data_counts, np.nan, dtype=float), where=mc_sum > 0)
    ratio_err = np.divide(data_err, mc_sum, out=np.full_like(data_err, np.nan, dtype=float), where=mc_sum > 0)

    rax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    rax.errorbar(
        centers,
        ratio,
        yerr=ratio_err,
        fmt="o",
        color="black",
        markersize=2.5,
        linewidth=0.7,
    )

    ax.set_ylabel(yaxis_label)
    rax.set_ylabel("Data / MC")
    rax.set_xlabel(xaxis_label)
    rax.set_ylim(0.0, 2.0)
    rax.grid(axis="y", alpha=0.25)

    if title:
        ax.set_title(title, loc="left")

    ax.legend(frameon=False, ncol=2)
    return fig


def plot_1D_comparison(
    reference: Any,
    target: Any,
    bin: int | np.ndarray = 50,
    xaxis_label: str = "",
    yaxis_label: str = "Events",
    *,
    reference_label: str = "reference",
    target_label: str = "target",
    reference_weight: Any | None = None,
    target_weight: Any | None = None,
    title: str | None = None,
) -> Any:
    """Overlay reference and target distributions with target/reference ratio."""
    from matplotlib import pyplot as plt

    set_plot_style()

    ref, ref_w = finite_values(reference, reference_weight)
    tar, tar_w = finite_values(target, target_weight)

    bins = auto_bins([ref, tar], n_bins=bin) if isinstance(bin, int) else np.asarray(bin, dtype=float)

    ref_counts, edges = np.histogram(ref, bins=bins, weights=ref_w)
    tar_counts, _ = np.histogram(tar, bins=edges, weights=tar_w)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(6.8, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
    )

    ax.hist(ref, bins=edges, weights=ref_w, histtype="step", linewidth=1.4, label=reference_label)
    ax.hist(tar, bins=edges, weights=tar_w, histtype="step", linewidth=1.4, label=target_label)

    ratio = np.divide(tar_counts, ref_counts, out=np.full_like(tar_counts, np.nan, dtype=float), where=ref_counts > 0)

    rax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    rax.plot(centers, ratio, "o-", markersize=2.5, linewidth=0.8)

    ax.set_ylabel(yaxis_label)
    rax.set_ylabel(f"{target_label}\n/ {reference_label}")
    rax.set_xlabel(xaxis_label)
    rax.grid(axis="y", alpha=0.25)

    if title:
        ax.set_title(title, loc="left")

    ax.legend(frameon=False)
    return fig


def plot_residual(
    reference: Any,
    target: Any,
    bin: int | np.ndarray = 50,
    xaxis_label: str = "target - reference",
    yaxis_label: str = "Events",
    *,
    weight: Any | None = None,
    title: str | None = None,
) -> Any:
    from matplotlib import pyplot as plt

    set_plot_style()

    ref = to_numpy(reference)
    tar = to_numpy(target)
    n = min(len(ref), len(tar))
    ref = ref[:n]
    tar = tar[:n]

    if weight is not None:
        w = to_numpy(weight)[:n]
        mask = np.isfinite(ref) & np.isfinite(tar) & np.isfinite(w)
        w = w[mask]
    else:
        mask = np.isfinite(ref) & np.isfinite(tar)
        w = None

    delta = tar[mask] - ref[mask]
    bins = auto_bins([delta], n_bins=bin) if isinstance(bin, int) else np.asarray(bin, dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.hist(delta, bins=bins, weights=w, histtype="stepfilled", alpha=0.75)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.9)

    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)

    if title:
        ax.set_title(title, loc="left")

    return fig


def plot_profile(
    reference: Any,
    target: Any,
    bin: int | np.ndarray = 20,
    xaxis_label: str = "reference",
    yaxis_label: str = "target - reference",
    *,
    weight: Any | None = None,
    title: str | None = None,
) -> Any:
    from matplotlib import pyplot as plt

    set_plot_style()

    ref = to_numpy(reference)
    tar = to_numpy(target)
    n = min(len(ref), len(tar))
    ref = ref[:n]
    tar = tar[:n]

    if weight is not None:
        w = to_numpy(weight)[:n]
        mask = np.isfinite(ref) & np.isfinite(tar) & np.isfinite(w)
        w = w[mask]
    else:
        mask = np.isfinite(ref) & np.isfinite(tar)
        w = None

    x = ref[mask]
    y = tar[mask] - ref[mask]

    bins = auto_bins([x], n_bins=bin) if isinstance(bin, int) else np.asarray(bin, dtype=float)
    centers = 0.5 * (bins[1:] + bins[:-1])

    mean = np.full(len(centers), np.nan)
    err = np.full(len(centers), np.nan)

    for i in range(len(centers)):
        in_bin = (x >= bins[i]) & (x < bins[i + 1])
        if i == len(centers) - 1:
            in_bin = (x >= bins[i]) & (x <= bins[i + 1])

        if not np.any(in_bin):
            continue

        yy = y[in_bin]
        if w is None:
            mean[i] = np.nanmean(yy)
            err[i] = np.nanstd(yy) / np.sqrt(np.sum(np.isfinite(yy)))
        else:
            ww = w[in_bin]
            mean[i] = np.average(yy, weights=ww)
            variance = np.average((yy - mean[i]) ** 2, weights=ww)
            err[i] = np.sqrt(variance / max(np.sum(in_bin), 1))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    valid = np.isfinite(mean)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.errorbar(centers[valid], mean[valid], yerr=err[valid], fmt="o-", markersize=3, linewidth=0.9)

    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)

    if title:
        ax.set_title(title, loc="left")

    return fig


def plot_2D_comparison(
    reference: Any,
    target: Any,
    *,
    bin: int | np.ndarray = 50,
    xaxis_label: str = "reference",
    yaxis_label: str = "target",
    title: str | None = None,
    weight: Any | None = None,
) -> dict[str, Any]:
    """Response matrix plus related 1D plots.

    Returns:
      {
        "2D_response": fig,
        "1D_overlay": fig,
        "1D_residual": fig,
        "1D_profile": fig,
      }
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    set_plot_style()

    ref = to_numpy(reference)
    tar = to_numpy(target)
    n = min(len(ref), len(tar))
    ref = ref[:n]
    tar = tar[:n]

    if weight is not None:
        w = to_numpy(weight)[:n]
        mask = np.isfinite(ref) & np.isfinite(tar) & np.isfinite(w)
        w = w[mask]
    else:
        mask = np.isfinite(ref) & np.isfinite(tar)
        w = None

    ref = ref[mask]
    tar = tar[mask]

    if len(ref) == 0:
        return {}

    bins = auto_bins([ref, tar], n_bins=bin) if isinstance(bin, int) else np.asarray(bin, dtype=float)

    out: dict[str, Any] = {}

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    image = ax.hist2d(ref, tar, bins=[bins, bins], weights=w, cmin=1, norm=LogNorm())

    ax.plot([bins[0], bins[-1]], [bins[0], bins[-1]], "--", color="red", linewidth=1.0)
    fig.colorbar(image[3], ax=ax, label="Events")
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)

    if title:
        ax.set_title(title, loc="left")

    out["2D_response"] = fig

    out["1D_overlay"] = plot_1D_comparison(
        ref,
        tar,
        bins,
        xaxis_label=xaxis_label,
        reference_label="reference",
        target_label="target",
        reference_weight=w,
        target_weight=w,
        title=None if title is None else f"{title}: 1D",
    )

    out["1D_residual"] = plot_residual(
        ref,
        tar,
        50,
        xaxis_label=f"{yaxis_label} - {xaxis_label}",
        weight=w,
        title=None if title is None else f"{title}: residual",
    )

    out["1D_profile"] = plot_profile(
        ref,
        tar,
        20,
        xaxis_label=xaxis_label,
        yaxis_label=f"{yaxis_label} - {xaxis_label}",
        weight=w,
        title=None if title is None else f"{title}: profile",
    )

    return out
