"""Sanity tests for the object-token-bottleneck latent-constraint AE (v2).

No dataset or GPU needed (tiny normalization.pt synthesized in a temp dir):

    python -m unittest RL.DGPO_neutrino.latent_constraint.test_object_token_ae
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from RL.DGPO_neutrino.latent_constraint.object_token_ae import (
    ObjectTokenBottleneckAutoencoder,
    load_checkpoint,
    save_checkpoint,
)

N_VIS, N_COND, N_INV, N_PART, TOK = 7, 10, 3, 12, 16


def _write_fake_normalization(path: Path) -> None:
    torch.save(
        {
            "input_mean": {"Source": torch.zeros(N_VIS), "Conditions": torch.zeros(N_COND)},
            "input_std": {"Source": torch.ones(N_VIS), "Conditions": torch.ones(N_COND)},
            "invisible_mean": {"Source": torch.zeros(N_INV)},
            "invisible_std": {"Source": torch.ones(N_INV)},
        },
        str(path),
    )


def _fake_batch(bsz: int, *, n_valid: int = N_PART) -> dict[str, torch.Tensor]:
    x_mask = torch.zeros(bsz, N_PART)
    x_mask[:, :n_valid] = 1.0
    return {
        "x": torch.randn(bsz, N_PART, N_VIS),
        "x_mask": x_mask,
        "conditions": torch.randn(bsz, N_COND),
        "conditions_mask": torch.ones(bsz, 1),
        "event_token": torch.randn(bsz, TOK) * 3.0 + 1.0,
        "object_token": torch.randn(bsz, N_PART, TOK) * 2.0 - 0.5,
        "x_invisible": torch.randn(bsz, 2, N_INV) * 0.5,
        "x_invisible_mask": torch.ones(bsz, 2),
    }


class TestObjectTokenBottleneckAE(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.norm_file = Path(self.tmp.name) / "normalization.pt"
        _write_fake_normalization(self.norm_file)
        torch.manual_seed(0)
        self.model = ObjectTokenBottleneckAutoencoder(
            normalization_file=str(self.norm_file),
            token_dim=TOK,
            nu_kin_dim=N_INV,
            latent_dim=8,
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            phi_index=2,
        )
        self.model.set_token_stats(torch.ones(TOK), torch.full((TOK,), 3.0))
        self.model.set_object_token_stats(torch.full((TOK,), -0.5), torch.full((TOK,), 2.0))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_shapes(self) -> None:
        b = _fake_batch(6)
        z = self.model.encode_latent(b)
        self.assertEqual(tuple(z.shape), (6, 8))
        nu_reco, tok_reco = self.model.decode(z)
        self.assertEqual(tuple(nu_reco.shape), (6, 2, N_INV))
        self.assertEqual(tuple(tok_reco.shape), (6, TOK))
        z2, nu2 = self.model(b)
        self.assertEqual(tuple(nu2.shape), (6, 2, N_INV))

    def test_loss_and_metrics(self) -> None:
        self.model.train()
        loss, metrics = self.model.reconstruction_loss(_fake_batch(6))
        self.assertTrue(torch.isfinite(loss))
        for k in ("recon_mse", "recon_nu_mse", "recon_token_mse", "latent_rms"):
            self.assertIn(k, metrics)
        self.assertNotIn("token_weight", metrics)  # no loss weight anymore
        # loss composition: plain sum nu + token (implicit weight 1)
        self.assertAlmostEqual(
            float(loss),
            float(metrics["recon_nu_mse"] + metrics["recon_token_mse"]),
            places=5,
        )

    def test_neutrino_residuals(self) -> None:
        """Per-component physical MSE: ~0 for perfect reco, grows when perturbed."""
        b = _fake_batch(6)
        truth_phys = self.model.neutrino_kin_from_batch(b)
        nu_norm_truth = self.model._normalize_neutrinos(truth_phys)  # decodes back to truth
        res0 = self.model._neutrino_residuals(nu_norm_truth, b)
        names = self.model._res_names
        self.assertEqual(names, ("pt", "eta", "phi"))  # spherical config
        for nm in names:
            self.assertIn(f"res_mse/{nm}", res0)
            self.assertLess(float(res0[f"res_mse/{nm}"]), 1e-3)  # near-perfect roundtrip
        # Perturbed reconstruction -> strictly larger residual, still finite.
        res1 = self.model._neutrino_residuals(nu_norm_truth + 1.0, b)
        for nm in names:
            self.assertGreater(float(res1[f"res_mse/{nm}"]), float(res0[f"res_mse/{nm}"]))
            self.assertTrue(torch.isfinite(res1[f"res_mse/{nm}"]))
        # Residuals surface in the loss metrics (so they get logged/aggregated).
        _, metrics = self.model.reconstruction_loss(b)
        for nm in names:
            self.assertIn(f"res_mse/{nm}", metrics)

    def test_object_mask_ignores_padding(self) -> None:
        """z must be invariant to the token values of padded (masked-out) objects."""
        b = _fake_batch(4, n_valid=N_PART - 3)  # last 3 objects are padding
        z0 = self.model.encode_latent(b, detach_neutrinos=True)
        b2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in b.items()}
        b2["object_token"][:, N_PART - 3:, :] += 17.0  # perturb only padded slots
        z1 = self.model.encode_latent(b2, detach_neutrinos=True)
        self.assertLess(float((z0 - z1).abs().max()), 1e-5)

    def test_grad_flows_to_neutrinos_with_frozen_encoder(self) -> None:
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        b = _fake_batch(6)
        nu = b.pop("x_invisible").clone().requires_grad_(True)
        b["nu_kin"] = nu
        z = self.model.encode_latent(b)
        z.sum().backward()
        self.assertIsNotNone(nu.grad)
        self.assertGreater(float(nu.grad.abs().sum()), 0.0)
        # event/object tokens are detached context: no encoder-parameter grads
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_detach_neutrinos(self) -> None:
        b = _fake_batch(6)
        nu = b.pop("x_invisible").clone().requires_grad_(True)
        b["nu_kin"] = nu
        z = self.model.encode_latent(b, detach_neutrinos=True)
        self.assertFalse(z.requires_grad)

    def test_missing_object_token_raises(self) -> None:
        b = _fake_batch(6)
        b.pop("object_token")
        with self.assertRaises(KeyError):
            self.model.encode_latent(b)

    def test_missing_event_token_raises(self) -> None:
        b = _fake_batch(6)
        b.pop("event_token")
        with self.assertRaises(KeyError):
            self.model.encode_latent(b)

    def test_checkpoint_roundtrip(self) -> None:
        self.model.train()
        self.model.reconstruction_loss(_fake_batch(8))
        ckpt = Path(self.tmp.name) / "obj.ckpt"
        save_checkpoint(ckpt, self.model, normalization_file=self.norm_file, epoch=1)
        model2, meta = load_checkpoint(ckpt, device="cpu")
        self.assertIsInstance(model2, ObjectTokenBottleneckAutoencoder)
        self.assertTrue(torch.allclose(model2.token_std, self.model.token_std))
        self.assertTrue(torch.allclose(model2.obj_token_std, self.model.obj_token_std))
        b = _fake_batch(5)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(
                    self.model.encode_latent(b, detach_neutrinos=True),
                    model2.encode_latent(b, detach_neutrinos=True),
                    atol=1e-6,
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
