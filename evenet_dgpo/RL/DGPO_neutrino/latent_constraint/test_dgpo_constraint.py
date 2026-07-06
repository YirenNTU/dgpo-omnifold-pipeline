"""Tests for the latent-SWD DGPO constraint (encode + SWD-ratio core).

Exercises the latent-specific core with the object-token bottleneck encoder:
the SWD ratio and the encode-from-kinematics path (finite, differentiable wrt
the predicted neutrinos, frozen-encoder grad isolation). No dataset / GPU /
DGPO stack required.

    python -m unittest RL.DGPO_neutrino.latent_constraint.test_dgpo_constraint
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
)
from RL.DGPO_neutrino.latent_constraint.dgpo_constraint import (
    LatentSWDConfig,
    LatentSWDState,
    _validate_dgpo_constraint_resume,
    latent_swd_constraint_from_kin,
    swd_ratio_constraint,
    sync_projection_constraint_C_across_ranks,
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


def _backbone(batch_size: int) -> dict[str, torch.Tensor]:
    x_mask = torch.ones(batch_size, N_PART)
    x_mask[:, -3:] = 0.0
    return {
        "x": torch.randn(batch_size, N_PART, N_VIS),
        "x_mask": x_mask,
        "conditions": torch.randn(batch_size, N_COND),
        "conditions_mask": torch.ones(batch_size, 1),
        "event_token": torch.randn(batch_size, TOK),
        "object_token": torch.randn(batch_size, N_PART, TOK),
    }


def _make_model(norm_path: Path, *, dropout: float = 0.0) -> ObjectTokenBottleneckAutoencoder:
    return ObjectTokenBottleneckAutoencoder(
        normalization_file=str(norm_path),
        token_dim=TOK,
        nu_kin_dim=N_INV,
        d_model=32,
        latent_dim=4,
        num_layers=2,
        num_heads=4,
        dropout=dropout,
    )


class TestSWDRatio(unittest.TestCase):
    def test_shift_increases_ratio(self) -> None:
        torch.manual_seed(0)
        z_truth = torch.randn(256, 4)
        c_same, _ = swd_ratio_constraint(
            z_truth.clone() + 0.01, z_truth, num_projections=128, eps=1e-6
        )
        c_shift, _ = swd_ratio_constraint(
            z_truth + 5.0, z_truth, num_projections=128, eps=1e-6
        )
        self.assertTrue(torch.isfinite(c_same) and torch.isfinite(c_shift))
        self.assertGreater(float(c_shift), float(c_same))

    def test_differentiable_pred_only(self) -> None:
        z_truth = torch.randn(128, 4)
        z_pred = torch.randn(128, 4, requires_grad=True)
        z_truth_g = torch.randn(128, 4, requires_grad=True)
        c, _ = swd_ratio_constraint(z_pred, z_truth_g, num_projections=64, eps=1e-6)
        c.backward()
        self.assertIsNotNone(z_pred.grad)
        self.assertTrue(torch.isfinite(z_pred.grad).all())
        self.assertGreater(float(z_pred.grad.abs().sum()), 0.0)
        self.assertIsNone(z_truth_g.grad)  # truth branch detached

    def test_seed_makes_estimator_deterministic(self) -> None:
        # Common random numbers: same seed -> identical projections + null split ->
        # the constraint is a deterministic function of the inputs (the property CPO
        # relies on so its theta_old vs theta_adam comparison reflects the step, not noise).
        torch.manual_seed(0)
        z_truth = torch.randn(200, 4)
        z_pred = torch.randn(200, 4)
        c_a, _ = swd_ratio_constraint(z_pred, z_truth, num_projections=64, eps=1e-6, seed=123)
        c_b, _ = swd_ratio_constraint(z_pred, z_truth, num_projections=64, eps=1e-6, seed=123)
        self.assertEqual(float(c_a), float(c_b))
        c_c, _ = swd_ratio_constraint(z_pred, z_truth, num_projections=64, eps=1e-6, seed=124)
        self.assertNotEqual(float(c_a), float(c_c))  # different seed -> different draw

    def test_seed_isolates_param_step_signal(self) -> None:
        # With CRN, a tiny perturbation of z_pred (the analogue of one optimizer step)
        # produces a small constraint change, not one dominated by estimator variance.
        torch.manual_seed(1)
        z_truth = torch.randn(256, 4)
        z_pred = torch.randn(256, 4)
        c0, _ = swd_ratio_constraint(z_pred, z_truth, num_projections=64, eps=1e-6, seed=7)
        c1, _ = swd_ratio_constraint(z_pred + 1e-3, z_truth, num_projections=64, eps=1e-6, seed=7)
        delta_crn = abs(float(c1) - float(c0))
        # Same perturbation but independent random draws (legacy behaviour): the
        # measured delta is inflated by projection/split noise.
        c0_noseed, _ = swd_ratio_constraint(z_pred, z_truth, num_projections=64, eps=1e-6)
        c1_noseed, _ = swd_ratio_constraint(z_pred + 1e-3, z_truth, num_projections=64, eps=1e-6)
        delta_noise = abs(float(c1_noseed) - float(c0_noseed))
        self.assertLess(delta_crn, max(delta_noise, 1e-9))


class TestFromKin(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        norm_path = Path(self._tmp.name) / "normalization.pt"
        _write_fake_normalization(norm_path)
        torch.manual_seed(0)
        model = _make_model(norm_path)
        cfg = LatentSWDConfig(
            enabled=True,
            checkpoint_file="",
            normalization_file=str(norm_path),
            margin=1.0,
            eps=1e-6,
            min_samples=8,
            apply_to="all_candidates",
            num_projections=64,
        )
        self.state = LatentSWDState(model=model, cfg=cfg)
        self.state.freeze()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_constraint_finite_and_differentiable(self) -> None:
        n = 64
        batch_sel = _backbone(n)
        pred_kin = torch.randn(n, 2, N_INV, requires_grad=True)
        truth_kin = torch.randn(n, 2, N_INV)
        c, diag = latent_swd_constraint_from_kin(self.state, batch_sel, pred_kin, truth_kin)
        self.assertTrue(torch.isfinite(c))
        for key in (
            "latent_constraint/swd_pred_truth",
            "latent_constraint/swd_truth_truth",
            "latent_constraint/swd_ratio",
            "latent_constraint/C_norm",
        ):
            self.assertIn(key, diag)
        c.backward()
        self.assertIsNotNone(pred_kin.grad)
        self.assertTrue(torch.isfinite(pred_kin.grad).all())
        self.assertGreater(float(pred_kin.grad.abs().sum()), 0.0)  # grad reaches predicted nu

    def test_frozen_encoder_has_no_param_grad(self) -> None:
        n = 32
        batch_sel = _backbone(n)
        pred_kin = torch.randn(n, 2, N_INV, requires_grad=True)
        truth_kin = torch.randn(n, 2, N_INV)
        c, _ = latent_swd_constraint_from_kin(self.state, batch_sel, pred_kin, truth_kin)
        c.backward()
        self.assertTrue(all(p.grad is None for p in self.state.model.parameters()))

    def test_missing_tokens_raise(self) -> None:
        n = 16
        pred_kin = torch.randn(n, 2, N_INV, requires_grad=True)
        truth_kin = torch.randn(n, 2, N_INV)
        for missing in ("event_token", "object_token"):
            batch_sel = _backbone(n)
            batch_sel.pop(missing)
            with self.assertRaises(KeyError):
                latent_swd_constraint_from_kin(self.state, batch_sel, pred_kin, truth_kin)


class TestIntegration(unittest.TestCase):
    def test_resume_constraint_type_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_dgpo_constraint_resume(
                {"constraint_type": "discriminator_wasserstein"},
                expected_type="latent_swd",
            )

    def test_checkpoint_payload_records_provenance(self) -> None:
        cfg = LatentSWDConfig(
            enabled=True,
            checkpoint_file="enc.ckpt",
            normalization_file="norm.pt",
            margin=1.0,
            eps=1e-6,
            min_samples=8,
            apply_to="all_candidates",
            num_projections=32,
        )
        state = LatentSWDState(model=None, cfg=cfg)  # type: ignore[arg-type]
        payload = state.checkpoint_payload()
        self.assertEqual(payload["constraint_type"], "latent_swd")
        self.assertEqual(payload["checkpoint_file"], "enc.ckpt")
        self.assertEqual(payload["normalization_file"], "norm.pt")

    def test_sync_c_identity_single_rank(self) -> None:
        c = sync_projection_constraint_C_across_ranks(
            1.25, device=torch.device("cpu"), world_size=1
        )
        self.assertEqual(c, 1.25)

    def test_encoder_stays_eval_during_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            norm_path = Path(tmp) / "normalization.pt"
            _write_fake_normalization(norm_path)
            model = _make_model(norm_path, dropout=0.5)
            model.train()
            cfg = LatentSWDConfig(
                enabled=True,
                checkpoint_file="",
                normalization_file="",
                margin=1.0,
                eps=1e-6,
                min_samples=8,
                apply_to="all_candidates",
                num_projections=32,
            )
            state = LatentSWDState(model=model, cfg=cfg)
            batch_sel = _backbone(16)
            pred_kin = torch.randn(16, 2, N_INV, requires_grad=True)
            truth_kin = torch.randn(16, 2, N_INV)
            latent_swd_constraint_from_kin(state, batch_sel, pred_kin, truth_kin)
            self.assertFalse(state.model.training)


class TestProjectionConstraintConfig(unittest.TestCase):
    def test_default_type_is_latent_swd(self) -> None:
        from RL.DGPO_neutrino.projection_cpo import resolve_projection_constraint_config

        cfg = resolve_projection_constraint_config(
            {"projection_constraint": {"latent_swd": {"checkpoint_file": "x.ckpt", "margin": 0.5}}}
        )
        self.assertEqual(cfg.type, "latent_swd")
        self.assertEqual(cfg.latent_swd.margin, 0.5)
        self.assertEqual(cfg.epsilon, 0.5)  # epsilon defaults to the margin

    def test_non_latent_swd_type_rejected(self) -> None:
        from RL.DGPO_neutrino.projection_cpo import resolve_projection_constraint_config

        with self.assertRaises(ValueError):
            resolve_projection_constraint_config(
                {"projection_constraint": {"type": "discriminator_wasserstein"}}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
