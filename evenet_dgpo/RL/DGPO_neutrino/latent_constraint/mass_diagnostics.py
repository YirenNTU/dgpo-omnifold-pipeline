"""Physics diagnostics for latent-constraint validation: neutrino momentum + W/top mass.

Answers "is this latent constraint good enough?" with observables DGPO actually
cares about (the same ``val_mass/*`` panel the DGPO trainer uses):

- **Neutrino momentum**: |p| distribution of truth vs AE-reconstructed neutrinos
  vs SHUFFLED pairing (each event given another event's truth neutrinos).
- **W mass**: assigned lepton + neutrino invariant mass (truth / recon / shuffled).
- **Top mass**: assigned b + W invariant mass (truth / recon / shuffled).

The shuffled series is the "broken pairing" reference: a useful event-aware
latent must reconstruct neutrinos whose mass peaks track TRUTH, while the
shuffled peaks stay smeared. Scalars logged every epoch (W&B ``val_mass/*``):

    w_jsd_recon / top_jsd_recon / p_jsd_recon          -> should FALL during training
    w_jsd_shuffled / top_jsd_shuffled / p_jsd_shuffled -> should stay LARGE (reference)

Mass math replicates ``dgpo_trainer._val_mass_reconstruction_masses`` (ground-truth
``assignments-indices`` pick b/lepton from the physical point cloud; first four pc
features are ``logE, logPt, eta, phi``; neutrinos are massless). Requires the
``assignments-*`` columns in the batch: set ``Assignment: {include: true}`` in the
latent-constraint config Components so the data pipeline keeps them.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.distance import jensenshannon

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from RL.DGPO_neutrino.rewards import (  # noqa: E402
    cartesian_to_log_pt_eta_phi,
    get_event_valid_mask,
)

_SERIES = ("truth", "recon", "shuffled")


# ----------------------------------------------------------------------
# four-vector helpers (mirror dgpo_trainer, kept local to avoid the heavy import)
# ----------------------------------------------------------------------
def _pc_row_to_4vec(pc_row: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """First four point-cloud features ``logE, logPt, eta, phi`` -> ``(E, px, py, pz)``."""
    log_e, log_pt, eta, phi = pc_row[..., :4].unbind(dim=-1)
    pt = torch.expm1(log_pt.clamp(-10.0, 10.0))
    e = torch.expm1(log_e.clamp(-10.0, 10.0))
    return e, pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta)


def _nu_kin_to_4vec(kin: Tensor, *, cartesian: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Neutrino slot kinematics -> massless four-vector [GeV]."""
    if cartesian:
        log_pt, eta, phi = cartesian_to_log_pt_eta_phi(kin[..., 0], kin[..., 1], kin[..., 2])
    else:
        log_pt, eta, phi = kin[..., 0], kin[..., 1], kin[..., 2]
    pt = torch.expm1(log_pt.clamp(-10.0, 10.0))
    return (
        pt * torch.cosh(eta),
        pt * torch.cos(phi),
        pt * torch.sin(phi),
        pt * torch.sinh(eta),
    )


def _mass(e: Tensor, px: Tensor, py: Tensor, pz: Tensor) -> Tensor:
    return torch.sqrt(torch.clamp(e * e - px * px - py * py - pz * pz, min=0.0))


@torch.no_grad()
def compute_w_top_masses(
    batch: Mapping[str, Any],
    nu_kin: Tensor,
    *,
    cartesian: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """W and top masses [GeV] for valid TT2L events (both tops, flattened).

    ``nu_kin`` is ``(B, 2, 3)`` PHYSICAL neutrino kinematics (truth, recon, or
    shuffled). Point cloud ``x`` is physical log-space; no denormalization here.
    Returns empty arrays when the assignment columns are absent/invalid.
    """
    assign = batch.get("assignments-indices")
    assign_m = batch.get("assignments-mask")
    if not isinstance(assign, Tensor) or assign.dim() != 3:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if not isinstance(assign_m, Tensor):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    device = nu_kin.device
    dtype = nu_kin.dtype
    B = int(batch["x"].shape[0])
    if assign.shape[0] != B or assign.shape[1] < 2 or assign.shape[2] < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    pc = batch["x"].to(device=device, dtype=dtype)
    assign = assign.to(device=device)
    b_idx = torch.arange(B, device=device)
    idx_ok = (assign[..., :2] >= 0).all(dim=-1)
    event_ok = (
        (get_event_valid_mask(dict(batch), B, device, dtype).reshape(B) > 0)
        & (assign_m.to(device=device) > 0).all(dim=-1)
        & idx_ok.all(dim=-1)
    )
    if not bool(event_ok.any().item()):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    nu1 = _nu_kin_to_4vec(nu_kin[:, 0, :], cartesian=cartesian)
    nu2 = _nu_kin_to_4vec(nu_kin[:, 1, :], cartesian=cartesian)

    w_masses, top_masses = [], []
    for r in range(2):
        b_pc = _pc_row_to_4vec(pc[b_idx, assign[:, r, 0].long().clamp_min(0)])
        l_pc = _pc_row_to_4vec(pc[b_idx, assign[:, r, 1].long().clamp_min(0)])
        nu = nu1 if r == 0 else nu2
        w4 = tuple(a + b for a, b in zip(l_pc, nu))
        top4 = tuple(a + b for a, b in zip(b_pc, w4))
        w_masses.append(_mass(*w4))
        top_masses.append(_mass(*top4))

    w_all = torch.stack(w_masses, dim=-1)[event_ok].reshape(-1)
    t_all = torch.stack(top_masses, dim=-1)[event_ok].reshape(-1)
    w_np = w_all.detach().float().cpu().numpy()
    t_np = t_all.detach().float().cpu().numpy()
    finite = np.isfinite(w_np) & np.isfinite(t_np)
    return w_np[finite], t_np[finite]


@torch.no_grad()
def nu_momentum_magnitude(nu_kin: Tensor, *, cartesian: bool) -> np.ndarray:
    """|p| [GeV] of both neutrino slots, flattened to 1D numpy."""
    if cartesian:
        p = torch.sqrt((nu_kin ** 2).sum(dim=-1).clamp_min(0.0))
    else:
        pt = torch.expm1(nu_kin[..., 0].clamp(-10.0, 10.0))
        p = pt * torch.cosh(nu_kin[..., 1])
    p_np = p.reshape(-1).detach().float().cpu().numpy()
    return p_np[np.isfinite(p_np)]


# ----------------------------------------------------------------------
# accumulating state (rank-symmetric all_reduce, DGPO val_mass-style figures)
# ----------------------------------------------------------------------
class MassDiagState:
    """Histogram accumulator for |p| / W mass / top mass, series truth|recon|shuffled."""

    def __init__(self, *, cartesian: bool) -> None:
        self.cartesian = bool(cartesian)
        self.edges = {
            "nu_momentum": np.linspace(0.0, 400.0, 81),
            "w_mass": np.linspace(0.0, 200.0, 81),
            "top_mass": np.linspace(80.0, 320.0, 81),
        }
        self.counts = {
            obs: {s: np.zeros(len(e) - 1, dtype=np.float64) for s in _SERIES}
            for obs, e in self.edges.items()
        }

    @torch.no_grad()
    def update(
        self,
        batch: Mapping[str, Any],
        truth_kin: Tensor,
        recon_kin: Tensor,
        shuffled_kin: Tensor,
    ) -> None:
        """Accumulate one validation batch (all kins PHYSICAL, ``(B, 2, 3)``)."""
        kins = {"truth": truth_kin, "recon": recon_kin, "shuffled": shuffled_kin}
        for s, kin in kins.items():
            p = nu_momentum_magnitude(kin, cartesian=self.cartesian)
            self.counts["nu_momentum"][s] += np.histogram(p, bins=self.edges["nu_momentum"])[0]
            w_m, t_m = compute_w_top_masses(batch, kin, cartesian=self.cartesian)
            self.counts["w_mass"][s] += np.histogram(w_m, bins=self.edges["w_mass"])[0]
            self.counts["top_mass"][s] += np.histogram(t_m, bins=self.edges["top_mass"])[0]

    def all_reduce(self, device: torch.device, world_size: int) -> None:
        """Sum histogram counts across ranks (call on EVERY rank, symmetric)."""
        if world_size <= 1:
            return
        import torch.distributed as dist

        for obs in self.counts:
            for s in _SERIES:
                t = torch.as_tensor(self.counts[obs][s], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                self.counts[obs][s] = t.cpu().numpy()

    # -------------------------------------------------------------- outputs
    def _jsd(self, obs: str, series: str) -> float:
        """Jensen-Shannon distance of ``series`` vs truth (0 = identical)."""
        a = self.counts[obs]["truth"]
        b = self.counts[obs][series]
        if a.sum() <= 0 or b.sum() <= 0:
            return float("nan")
        return float(jensenshannon(a / a.sum(), b / b.sum()))

    def jsd_scalars(self) -> dict[str, float]:
        """W&B scalars (``val_mass/*``): recon JSDs should fall, shuffled stay high."""
        tag = {"nu_momentum": "p", "w_mass": "w", "top_mass": "top"}
        out: dict[str, float] = {}
        for obs, t in tag.items():
            out[f"{t}_jsd_recon"] = self._jsd(obs, "recon")
            out[f"{t}_jsd_shuffled"] = self._jsd(obs, "shuffled")
        return out

    def build_figures(self) -> dict[str, Any]:
        """Density overlays (Truth / AE recon / Shuffled pairing) per observable."""
        titles = {
            "nu_momentum": (r"Neutrino $|p|$", r"$|p_\nu|$ [GeV]"),
            "w_mass": (r"W mass (lepton + $\nu$)", r"$m_{\ell\nu}$ [GeV]"),
            "top_mass": (r"Top mass (b + W)", r"$m_{b\ell\nu}$ [GeV]"),
        }
        figs: dict[str, Any] = {}
        for obs, (title, xlabel) in titles.items():
            edges = self.edges[obs]
            centers = 0.5 * (edges[:-1] + edges[1:])
            width = float(edges[1] - edges[0])
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            styles = {
                "truth": dict(linewidth=2.0, marker="o", markersize=3),
                "recon": dict(linewidth=2.0, marker="s", markersize=3),
                "shuffled": dict(linewidth=1.6, linestyle="--", marker="^", markersize=3),
            }
            labels = {"truth": "Truth", "recon": "AE recon", "shuffled": "Shuffled pairing"}
            for s in _SERIES:
                c = self.counts[obs][s]
                norm = c.sum() * width + 1e-12
                ax.plot(centers, c / norm, label=labels[s], **styles[s])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Density")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            figs[obs] = fig
        return figs
