"""W&B monitoring plots for the latent-constraint autoencoder.

Mirrors the EveNet generation metrics style (``evenet/network/metrics/generation.py``)
so the AE's panels look identical to the main model's neutrino plots:

- **1D overlay** per kinematic feature: truth as outlined bars, reconstruction as
  a dashed line + markers, density-normalized, with a Jensen-Shannon distance
  (JSD) reported per neutrino slot.
- **2D pred-vs-truth** histogram per feature (``pcolormesh`` + colorbar).
- a derived **pT** overlay (``sqrt(px^2+py^2)`` in cartesian, ``expm1(log_pt)``
  in spherical) and a **latent-z** distribution.

Figures are logged under the same ``generation-invisible/*`` category the main
model uses, so they land in the same W&B group.

The accumulator stores fixed-size histogram counts and is reduced across ranks
(sum) before plotting on rank 0 -- exactly like ``GenerationMetrics.reduce_across_gpus``
-- so it issues identical, rank-symmetric collectives and never desyncs DDP.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless (NERSC compute nodes have no display)
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402

import torch  # noqa: E402

# Same palette as evenet/network/metrics/generation.py for visual consistency.
_COLORS = [
    "#40B0A6", "#6D8EF7", "#6E579A", "#A38E89", "#A5C8DD",
    "#CD5582", "#E1BE6A", "#E89A7A", "#EC6B2D",
]
_SLOT_NAMES = ("nu", "anti-nu")


def _feature_bins(name: str, cartesian: bool) -> np.ndarray:
    """Fixed physics bins per feature (must be identical across ranks)."""
    if cartesian:
        return np.linspace(-300.0, 300.0, 61)  # px, py, pz [GeV]
    if name == "log_pt":
        return np.linspace(0.0, 7.0, 61)        # log1p(pt), pt up to ~1100 GeV
    if name == "eta":
        return np.linspace(-6.0, 6.0, 61)
    if name == "phi":
        return np.linspace(-np.pi, np.pi, 61)
    return np.linspace(-10.0, 10.0, 61)


_PT_BINS = np.linspace(0.0, 300.0, 61)
_Z_BINS = np.linspace(-5.0, 5.0, 61)


def _pt_from_features(kin: np.ndarray, cartesian: bool) -> np.ndarray:
    """Physical pT from a ``(..., nu_kin_dim)`` feature array."""
    if cartesian:
        return np.sqrt(kin[..., 0] ** 2 + kin[..., 1] ** 2)
    return np.expm1(kin[..., 0])  # log1p(pt) -> pt


class ReconPlotState:
    """Accumulates truth/reconstruction histograms for the AE monitoring plots.

    All buffers are fixed-size float64 count arrays so they can be summed across
    ranks with a single rank-symmetric all-reduce per buffer.
    """

    def __init__(self, feature_names: Sequence[str], *, cartesian: bool) -> None:
        self.feature_names = list(feature_names)
        self.cartesian = bool(cartesian)
        self.bins = {n: _feature_bins(n, cartesian) for n in self.feature_names}
        nslot = len(_SLOT_NAMES)
        nfeat = len(self.feature_names)
        nb = len(next(iter(self.bins.values()))) - 1
        # [feature][slot] 1D counts (truth / pred) and [feature][slot] 2D counts.
        self.truth_1d = {n: np.zeros((nslot, nb)) for n in self.feature_names}
        self.pred_1d = {n: np.zeros((nslot, nb)) for n in self.feature_names}
        self.hist2d = {n: np.zeros((nslot, nb, nb)) for n in self.feature_names}
        self.pt_truth = np.zeros((nslot, len(_PT_BINS) - 1))
        self.pt_pred = np.zeros((nslot, len(_PT_BINS) - 1))
        self.z_hist = np.zeros(len(_Z_BINS) - 1)

    @torch.no_grad()
    def update(
        self,
        truth_phys: torch.Tensor,   # (B, 2, F) physical
        recon_phys: torch.Tensor,   # (B, 2, F) physical
        slot_mask: torch.Tensor,    # (B, 2)
        z: torch.Tensor,            # (B, latent_dim)
    ) -> None:
        t = truth_phys.detach().float().cpu().numpy()
        p = recon_phys.detach().float().cpu().numpy()
        m = slot_mask.detach().bool().cpu().numpy()  # (B, 2)
        for s in range(len(_SLOT_NAMES)):
            sel = m[:, s]
            if not np.any(sel):
                continue
            ts, ps = t[sel, s, :], p[sel, s, :]
            for fi, name in enumerate(self.feature_names):
                edges = self.bins[name]
                self.truth_1d[name][s] += np.histogram(ts[:, fi], bins=edges)[0]
                self.pred_1d[name][s] += np.histogram(ps[:, fi], bins=edges)[0]
                self.hist2d[name][s] += np.histogram2d(
                    ps[:, fi], ts[:, fi], bins=[edges, edges]
                )[0]
            self.pt_truth[s] += np.histogram(_pt_from_features(ts, self.cartesian), bins=_PT_BINS)[0]
            self.pt_pred[s] += np.histogram(_pt_from_features(ps, self.cartesian), bins=_PT_BINS)[0]
        self.z_hist += np.histogram(z.detach().float().cpu().numpy().reshape(-1), bins=_Z_BINS)[0]

    def all_reduce(self, device: torch.device, world_size: int) -> None:
        """Sum every count buffer across ranks (rank-symmetric collectives)."""
        if world_size <= 1:
            return
        buffers = []
        for d in (self.truth_1d, self.pred_1d):
            buffers += [d[n][s] for n in self.feature_names for s in range(len(_SLOT_NAMES))]
        buffers += [self.hist2d[n][s] for n in self.feature_names for s in range(len(_SLOT_NAMES))]
        buffers += [self.pt_truth[s] for s in range(len(_SLOT_NAMES))]
        buffers += [self.pt_pred[s] for s in range(len(_SLOT_NAMES))]
        buffers.append(self.z_hist)
        for arr in buffers:
            t = torch.as_tensor(arr, device=device, dtype=torch.float64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            arr[...] = t.cpu().numpy()

    # ------------------------------------------------------------------
    # rendering (mirrors generation.py: plot_histogram_func / plot_histogram2d_func)
    # ------------------------------------------------------------------
    def _overlay_1d(self, truth_counts, pred_counts, edges, xlabel):
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        fig, ax = plt.subplots()
        jsd = {}
        for s, slot in enumerate(_SLOT_NAMES):
            color = _COLORS[s % len(_COLORS)]
            pc, tc = pred_counts[s], truth_counts[s]
            if np.sum(pc) > 0:
                ax.plot(centers, pc / (np.sum(pc) * widths), color=color,
                        linestyle="--", marker="o", linewidth=2, markersize=5,
                        label=f"{slot} (Pred)")
            if np.sum(tc) > 0:
                ax.bar(centers, tc / (np.sum(tc) * widths), width=widths, color=color,
                       alpha=0.7, edgecolor=color, fill=False, label=f"{slot} (Truth)")
            if np.sum(pc) > 0 and np.sum(tc) > 0:
                jsd[slot] = float(jensenshannon(tc / np.sum(tc), pc / np.sum(pc)))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()
        return fig, jsd

    def _hist2d(self, counts, edges, title):
        centers = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(centers, centers, indexing="ij")
        pcm = ax.pcolormesh(X, Y, counts, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, label="Counts")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Truth")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    def build_figures(self) -> tuple[dict, dict]:
        """Return ``(figs, jsd)``: ``figs`` keyed by tag, ``jsd`` scalar per series."""
        figs: dict = {}
        jsd_results: dict = {}
        for name in self.feature_names:
            edges = self.bins[name]
            fig, jsd = self._overlay_1d(self.truth_1d[name], self.pred_1d[name], edges, name)
            figs[f"neutrino-{name}-1d"] = fig
            for slot, score in jsd.items():
                jsd_results[f"neutrino-{name}-{slot}"] = score
            for s, slot in enumerate(_SLOT_NAMES):
                if np.sum(self.hist2d[name][s]) > 0:
                    figs[f"neutrino-2D_{name}_{slot}"] = self._hist2d(
                        self.hist2d[name][s], edges, f"2D {name} - {slot}")
        fig_pt, jsd_pt = self._overlay_1d(self.pt_truth, self.pt_pred, _PT_BINS, "pT [GeV]")
        figs["neutrino-pt-1d"] = fig_pt
        for slot, score in jsd_pt.items():
            jsd_results[f"neutrino-pt-{slot}"] = score
        # latent z distribution (single series)
        zc = 0.5 * (_Z_BINS[:-1] + _Z_BINS[1:])
        zw = np.diff(_Z_BINS)
        figz, axz = plt.subplots()
        if np.sum(self.z_hist) > 0:
            axz.bar(zc, self.z_hist / (np.sum(self.z_hist) * zw), width=zw,
                    color=_COLORS[0], alpha=0.7, edgecolor=_COLORS[0], fill=False)
        axz.set_xlabel("latent z")
        axz.set_ylabel("Frequency")
        figz.tight_layout()
        figs["neutrino-latent_z-1d"] = figz
        return figs, jsd_results
