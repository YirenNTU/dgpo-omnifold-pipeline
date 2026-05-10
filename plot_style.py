from __future__ import annotations

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