"""Tests for the latent-constraint physics diagnostics (|p|, W/top mass, shuffling).

    python -m unittest RL.DGPO_neutrino.latent_constraint.test_mass_diagnostics
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from RL.DGPO_neutrino.latent_constraint.mass_diagnostics import (
    MassDiagState,
    compute_w_top_masses,
    nu_momentum_magnitude,
)

N_VIS, N_PART = 7, 12


def _pc_row(log_e, log_pt, eta, phi) -> torch.Tensor:
    row = torch.zeros(N_VIS)
    row[0], row[1], row[2], row[3] = log_e, log_pt, eta, phi
    return row


def _batch_with_assignments(bsz: int) -> dict[str, torch.Tensor]:
    """Events where slot 0/1 hold the b, slot 2/3 the leptons (both tops)."""
    x = torch.zeros(bsz, N_PART, N_VIS)
    g = torch.Generator().manual_seed(0)
    for i in range(bsz):
        for j in range(4):
            pt = 40.0 + 60.0 * torch.rand((), generator=g)
            eta = 2.0 * torch.rand((), generator=g) - 1.0
            phi = 2 * torch.pi * torch.rand((), generator=g) - torch.pi
            e = pt * torch.cosh(eta) * 1.05  # slightly massive
            x[i, j] = _pc_row(torch.log1p(e), torch.log1p(pt), eta, phi)
    assign = torch.full((bsz, 2, 2), -1.0)
    assign[:, 0, 0], assign[:, 0, 1] = 0.0, 2.0  # top 1: b=slot0, lepton=slot2
    assign[:, 1, 0], assign[:, 1, 1] = 1.0, 3.0  # top 2: b=slot1, lepton=slot3
    return {
        "x": x,
        "x_mask": torch.ones(bsz, N_PART),
        "assignments-indices": assign,
        "assignments-mask": torch.ones(bsz, 2),
    }


def _truth_nu(bsz: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    pt = 20.0 + 50.0 * torch.rand(bsz, 2, generator=g)
    eta = 2.0 * torch.rand(bsz, 2, generator=g) - 1.0
    phi = 2 * torch.pi * torch.rand(bsz, 2, generator=g) - torch.pi
    return torch.stack([torch.log1p(pt), eta, phi], dim=-1)  # (B, 2, 3) spherical


class TestMassDiagnostics(unittest.TestCase):
    def test_masses_finite_and_shaped(self) -> None:
        b = _batch_with_assignments(32)
        nu = _truth_nu(32)
        w_m, t_m = compute_w_top_masses(b, nu, cartesian=False)
        self.assertEqual(len(w_m), 64)  # 2 tops per event
        self.assertEqual(len(t_m), 64)
        self.assertTrue(np.isfinite(w_m).all() and np.isfinite(t_m).all())
        self.assertTrue((t_m >= w_m - 1e-3).all())  # adding the b can't reduce mass

    def test_missing_assignments_returns_empty(self) -> None:
        b = _batch_with_assignments(8)
        b.pop("assignments-indices")
        w_m, t_m = compute_w_top_masses(b, _truth_nu(8), cartesian=False)
        self.assertEqual(len(w_m), 0)

    def test_momentum_magnitude(self) -> None:
        nu = _truth_nu(16)
        p = nu_momentum_magnitude(nu, cartesian=False)
        self.assertEqual(len(p), 32)
        self.assertTrue((p > 0).all())
        # cartesian path agrees with spherical: |p| = pt*cosh(eta)
        pt = torch.expm1(nu[..., 0])
        xyz = torch.stack(
            [pt * torch.cos(nu[..., 2]), pt * torch.sin(nu[..., 2]), pt * torch.sinh(nu[..., 1])],
            dim=-1,
        )
        p_cart = nu_momentum_magnitude(xyz, cartesian=True)
        np.testing.assert_allclose(np.sort(p), np.sort(p_cart), rtol=1e-4)

    def test_state_update_jsd_and_figures(self) -> None:
        state = MassDiagState(cartesian=False)
        b = _batch_with_assignments(64)
        truth = _truth_nu(64)
        perm = torch.randperm(64, generator=torch.Generator().manual_seed(2))
        # recon = truth + small noise; shuffled = wrong pairing
        recon = truth + 0.05 * torch.randn(truth.shape, generator=torch.Generator().manual_seed(3))
        state.update(b, truth, recon, truth[perm])
        scal = state.jsd_scalars()
        for k in ("w_jsd_recon", "top_jsd_recon", "p_jsd_recon",
                  "w_jsd_shuffled", "top_jsd_shuffled", "p_jsd_shuffled"):
            self.assertIn(k, scal)
            self.assertTrue(np.isfinite(scal[k]))
        # near-truth recon must beat the shuffled reference on the masses
        self.assertLess(scal["w_jsd_recon"], scal["w_jsd_shuffled"] + 1e-9)
        figs = state.build_figures()
        self.assertEqual(set(figs), {"nu_momentum", "w_mass", "top_mass"})
        import matplotlib.pyplot as plt
        for f in figs.values():
            plt.close(f)

    def test_all_reduce_single_rank_noop(self) -> None:
        state = MassDiagState(cartesian=False)
        state.update(_batch_with_assignments(16), _truth_nu(16), _truth_nu(16), _truth_nu(16))
        before = {o: {s: c.copy() for s, c in d.items()} for o, d in state.counts.items()}
        state.all_reduce(torch.device("cpu"), world_size=1)
        for o in before:
            for s in before[o]:
                np.testing.assert_array_equal(before[o][s], state.counts[o][s])


if __name__ == "__main__":
    unittest.main(verbosity=2)
