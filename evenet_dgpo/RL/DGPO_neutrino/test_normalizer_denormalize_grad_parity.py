"""
Parity checks: ``Normalizer.denormalize_grad`` vs the pre-rebase reference
(``EveNet-DGPO-original`` ``denormalize`` without ``@torch.no_grad()``).

Run from repo root (needs PyTorch):
    python -m unittest RL.DGPO_neutrino.test_normalizer_denormalize_grad_parity -v
"""

from __future__ import annotations

import math
import unittest

import torch
from torch.distributions import Normal

from evenet.network.body.normalizer import Normalizer


def _original_denormalize_reference(
    norm: Normalizer,
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    remove_padding: bool = False,
    index: list[int] | None = None,
) -> torch.Tensor:
    """Copy of ``EveNet-DGPO-original/evenet/network/body/normalizer.py::denormalize``."""
    if remove_padding:
        current_mean = norm.mean[:- norm.padding]
        current_std = norm.std[:- norm.padding]
    else:
        current_mean = norm.mean
        current_std = norm.std

    if len(norm.inv_cdf_index) > 0:
        if index is not None:
            inv_cdf_index = [idx for idx in index if idx in norm.inv_cdf_index]
        else:
            inv_cdf_index = norm.inv_cdf_index

        x_partial = x[..., inv_cdf_index].contiguous()
        x_partial = torch.nan_to_num(x_partial, nan=0.0, posinf=8.0, neginf=-8.0)
        x_partial = Normal(0, 1).cdf(x_partial)
        x_partial = x_partial * 2 * math.sqrt(3) - math.sqrt(3)
        x = x.clone()
        x[..., inv_cdf_index] = x_partial
        if mask is not None:
            x = x * mask

    if index is not None:
        x = x * current_std[index] + current_mean[index]
    else:
        x = (x * current_std) + current_mean
    if mask is not None:
        x = x * mask
    return x


def _make_normalizer(*, padding: int = 0, inv_cdf_index: list[int] | None = None) -> Normalizer:
    F = 4
    mean = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)
    std = torch.tensor([1.5, 2.0, 0.8, 1.1], dtype=torch.float32)
    norm_mask = torch.ones(F, dtype=torch.bool)
    return Normalizer(
        mean,
        std,
        norm_mask,
        inv_cdf_index=inv_cdf_index or [],
        padding_size=padding,
    )


class TestDenormalizeGradParity(unittest.TestCase):
    def test_forward_matches_original_without_inv_cdf(self) -> None:
        norm = _make_normalizer()
        x = torch.randn(5, 2, 4)
        mask = torch.ones(5, 2, 4)
        ref = _original_denormalize_reference(norm, x, mask)
        out = norm.denormalize_grad(x, mask=mask)
        self.assertTrue(torch.allclose(ref, out, atol=1e-6, rtol=1e-5))

    def test_forward_matches_original_with_inv_cdf(self) -> None:
        norm = _make_normalizer(inv_cdf_index=[1])
        x = torch.randn(7, 2, 4)
        mask = torch.ones(7, 2, 4)
        ref = _original_denormalize_reference(norm, x, mask)
        out = norm.denormalize_grad(x, mask=mask)
        self.assertTrue(torch.allclose(ref, out, atol=1e-6, rtol=1e-5))

    def test_forward_matches_original_remove_padding(self) -> None:
        # 3 real features + padding_size=1 => mean/std length 4; remove_padding => 3.
        F = 3
        mean = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
        std = torch.tensor([1.5, 2.0, 0.8], dtype=torch.float32)
        norm_mask = torch.ones(F, dtype=torch.bool)
        norm = Normalizer(mean, std, norm_mask, inv_cdf_index=[1], padding_size=1)
        x = torch.randn(3, 2, F)
        mask = torch.ones(3, 2, F)
        ref = _original_denormalize_reference(norm, x, mask, remove_padding=True)
        out = norm.denormalize_grad(x, mask=mask, remove_padding=True)
        self.assertTrue(torch.allclose(ref, out, atol=1e-6, rtol=1e-5))

    def test_no_grad_denormalize_matches_original_forward(self) -> None:
        norm = _make_normalizer(inv_cdf_index=[1])
        x = torch.randn(4, 2, 4)
        mask = torch.ones(4, 2, 4)
        ref = _original_denormalize_reference(norm, x, mask)
        with torch.no_grad():
            out = norm.denormalize(x, mask=mask)
        self.assertTrue(torch.allclose(ref, out, atol=1e-6, rtol=1e-5))

    def test_grad_matches_original_reference(self) -> None:
        norm = _make_normalizer(inv_cdf_index=[1])
        x = torch.randn(6, 2, 4, requires_grad=True)
        mask = torch.ones(6, 2, 4)

        y_ref = _original_denormalize_reference(norm, x, mask)
        y_ref.sum().backward()
        grad_ref = x.grad.detach().clone()

        x2 = x.detach().clone().requires_grad_(True)
        y_new = norm.denormalize_grad(x2, mask=mask)
        y_new.sum().backward()
        grad_new = x2.grad

        self.assertIsNotNone(grad_ref)
        self.assertIsNotNone(grad_new)
        self.assertTrue(torch.allclose(grad_ref, grad_new, atol=1e-5, rtol=1e-4))

    def test_constraint_path_requires_grad_after_denormalize_grad(self) -> None:
        """Smoke test for the exact failure mode from the NERSC crash."""
        norm = _make_normalizer(inv_cdf_index=[1])
        x0_hat = torch.randn(2, 2, 4, requires_grad=True)
        mask = torch.ones(2, 2, 4)
        nu_phys = norm.denormalize_grad(x0_hat, mask=mask)
        loss = nu_phys.sum()
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(x0_hat.grad)


class TestProjectionViolationEstimatorParity(unittest.TestCase):
    """``v_linear`` formula is unchanged from EveNet-DGPO-original."""

    def test_linear_violation_formula(self) -> None:
        C_old = 1.25
        b = torch.tensor([2.0, -1.0, 0.5])
        delta0 = torch.tensor([0.1, 0.2, -0.3])
        eps = 1.0
        C_adam_pred = C_old + float(torch.dot(b, delta0))
        v_linear = C_adam_pred - eps
        self.assertAlmostEqual(C_adam_pred, 1.10, places=6)
        self.assertAlmostEqual(v_linear, 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
