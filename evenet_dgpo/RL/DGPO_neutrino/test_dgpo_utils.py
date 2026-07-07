"""
Unit tests for RL/DGPO_neutrino/dgpo_utils.py.

Run from repo root:
    python RL/DGPO_neutrino/test_dgpo_utils.py
"""

from __future__ import annotations

import ast
import numpy as np
import os
import sys
import types
import unittest

import torch

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)

# ``diffusion_sampler`` imports ``debug_tool``, which pulls Lightning/torchmetrics/torchvision.
# Stub it so these unit tests only need PyTorch.
_dbg = types.ModuleType("evenet.utilities.debug_tool")


def _noop_time_decorator(name=None):
    def _wrapper(func):
        return func

    return _wrapper


_dbg.time_decorator = _noop_time_decorator
sys.modules.setdefault("evenet.utilities.debug_tool", _dbg)

from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma

from RL.DGPO_neutrino.dgpo_utils import (
    build_dgpo_loss,
    compute_per_event_advantage,
    predict_x0_normalized_from_velocity_diffusion,
    repeat_batch_for_candidates,
)


def _load_generate_neutrino_candidates():
    trainer_path = os.path.join(_REPO_ROOT, "RL", "DGPO_neutrino", "dgpo_trainer.py")
    with open(trainer_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    module = ast.parse(source, filename=trainer_path)
    func_node = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_neutrino_candidates"
    )
    fn_module = ast.Module(body=[func_node], type_ignores=[])
    code = compile(fn_module, trainer_path, "exec")
    namespace = {
        "Any": object,
        "DDIMSampler": object,
        "Tensor": torch.Tensor,
        "partial": __import__("functools").partial,
        "repeat_batch_for_candidates": repeat_batch_for_candidates,
        "torch": torch,
    }
    exec(code, namespace)
    return namespace["generate_neutrino_candidates"]


generate_neutrino_candidates = _load_generate_neutrino_candidates()


class TestPerEventAdvantage(unittest.TestCase):
    def test_zscore_within_event(self):
        torch.manual_seed(0)
        K, B = 8, 4
        rewards = torch.randn(K, B)
        advantages, weights = compute_per_event_advantage(rewards, eps=1e-6)
        self.assertEqual(advantages.shape, (K, B))
        self.assertEqual(weights.shape, (K, B))
        torch.testing.assert_close(weights, advantages.abs())
        mean_per_event = advantages.mean(dim=0)
        std_per_event = advantages.std(dim=0, unbiased=False)
        torch.testing.assert_close(mean_per_event, torch.zeros(B), rtol=0, atol=1e-5)
        torch.testing.assert_close(std_per_event, torch.ones(B), rtol=0, atol=1e-5)


class TestRepeatBatchForCandidates(unittest.TestCase):
    def test_shape_and_tiling(self):
        B, P, F = 4, 20, 7
        K = 3
        batch = {
            "x": torch.arange(B * P * F, dtype=torch.float32).reshape(B, P, F),
            "meta": "skip",
        }
        out = repeat_batch_for_candidates(batch, K)
        self.assertEqual(out["x"].shape, (K * B, P, F))
        self.assertEqual(out["meta"], "skip")
        for k in range(K):
            self.assertTrue(torch.equal(out["x"][k * B], batch["x"][0]))


class TestVelocityToX0Reconstruction(unittest.TestCase):
    """Algebra matches ``DDIMSampler`` velocity branch (`diffusion_sampler.py`)."""

    def test_recover_x0_from_velocity_under_forward_noise(self):
        torch.manual_seed(41)
        b, n, f = 2, 3, 5
        x0 = torch.randn(b, n, f)
        eps_true = torch.randn(b, n, f)
        t_flat = torch.rand(b)
        _, alpha, sigma = get_logsnr_alpha_sigma(t_flat, shape=(b, 1, 1))
        x_t = alpha * x0 + sigma * eps_true
        v_pred = (eps_true - x_t * sigma) / alpha.clamp(min=1e-8)
        x0_hat, _, _ = predict_x0_normalized_from_velocity_diffusion(x_t, v_pred, t_flat)
        torch.testing.assert_close(x0_hat, x0, rtol=1e-4, atol=1e-4)


class TestBuildDGPOLoss(unittest.TestCase):
    """Velocity DGPO loss: detached gate, pure main term."""

    def test_gradient_through_L_cur_and_finite(self):
        torch.manual_seed(2)
        K, B = 4, 3
        L_cur = torch.randn(K, B, requires_grad=True)
        L_ref = torch.randn(K, B)
        advantages = torch.randn(K, B)
        loss, diag = build_dgpo_loss(
            L_cur, L_ref, advantages, beta_dgpo=0.5, K=K
        )
        self.assertTrue(loss.requires_grad)
        self.assertFalse(diag["L_cur_mean"].requires_grad)
        loss.backward()
        self.assertIsNotNone(L_cur.grad)
        self.assertTrue(torch.isfinite(loss).item())

    def test_L_ref_not_updated(self):
        K, B = 2, 2
        L_cur = torch.randn(K, B, requires_grad=True)
        L_ref = torch.randn(K, B, requires_grad=True)
        advantages = torch.ones(K, B)
        loss, _ = build_dgpo_loss(L_cur, L_ref, advantages, beta_dgpo=1.0, K=K)
        loss.backward()
        self.assertIsNone(L_ref.grad)

    def test_sgd_decreases_loss(self):
        torch.manual_seed(3)
        K, B = 4, 2
        L_ref = torch.ones(K, B) * 0.5
        L_cur = (torch.ones(K, B) * 0.5).requires_grad_(True)
        advantages = torch.tensor([[-1.2, -0.8], [-0.3, -0.1], [0.3, 0.1], [1.2, 0.8]])
        opt = torch.optim.SGD([L_cur], lr=0.05)
        losses = []
        for _ in range(200):
            opt.zero_grad()
            loss, _ = build_dgpo_loss(
                L_cur, L_ref, advantages, beta_dgpo=1.0, K=K
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_positive_advantage_drives_L_cur_down(self):
        K, B = 2, 1
        L_ref = torch.ones(K, B) * 0.5
        L_cur = (torch.ones(K, B) * 0.5).requires_grad_(True)
        advantages = torch.tensor([[1.0], [-1.0]])
        opt = torch.optim.SGD([L_cur], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss, _ = build_dgpo_loss(
                L_cur, L_ref, advantages, beta_dgpo=1.0, K=K
            )
            loss.backward()
            opt.step()
        self.assertLess(L_cur[0, 0].item(), 0.5)

    def test_diag_finite_means(self):
        """Loss diagnostics expose only mean L_cur / L_ref (for W&B); should be finite."""
        K, B = 4, 1
        L_cur = torch.randn(K, B, requires_grad=True)
        L_ref = torch.randn(K, B)
        advantages = torch.randn(K, B)
        _, diag = build_dgpo_loss(L_cur, L_ref, advantages, beta_dgpo=0.2, K=K)
        self.assertTrue(torch.isfinite(diag["L_cur_mean"]))
        self.assertTrue(torch.isfinite(diag["L_ref_mean"]))
        for k in ("w_e_mean", "w_e_std", "w_e_min", "w_e_max"):
            self.assertTrue(torch.isfinite(diag[k]), msg=k)

    def test_equal_L_cur_L_ref_backward_ok(self):
        """When L_cur == L_ref, Delta=0, w_e=0.5; loss stays finite and backprops to L_cur."""
        K, B = 4, 2
        # Use a leaf tensor; ``ones * scalar`` is non-leaf, so .grad is not stored on L.
        L = torch.full((K, B), 0.3, dtype=torch.float32, requires_grad=True)
        advantages = torch.randn(K, B)
        loss, _ = build_dgpo_loss(L, L, advantages, beta_dgpo=1.0, K=K)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(L.grad)


class TestGenerateNeutrinoCandidates(unittest.TestCase):
    def test_parallel_chains_batches_sampler_calls(self):
        class FakeNormalizer:
            def denormalize(self, x, noise_mask, remove_padding=False):
                return x

        class FakeModel:
            invisible_input_dim = 3
            invisible_normalizer = FakeNormalizer()

            def __init__(self):
                self.call_batch_sizes = []

            def predict_diffusion_vector(self, noise_x, cond_x, time, mode, noise_mask=None):
                self.call_batch_sizes.append((noise_x.shape[0], cond_x["x"].shape[0], noise_mask.shape[0]))
                return torch.zeros_like(noise_x)

        class FakeSampler:
            def __init__(self):
                self.data_shapes = []

            def sample(
                self,
                data_shape,
                pred_fn,
                normalize_fn=None,
                num_steps=20,
                eta=1.0,
                noise_mask=None,
                use_tqdm=False,
                process_name="Sampling",
                remove_padding=False,
            ):
                self.data_shapes.append(data_shape)
                pred_fn(
                    noise_x=torch.zeros(data_shape, dtype=torch.float32),
                    time=torch.zeros((data_shape[0],), dtype=torch.float32),
                )
                out = torch.arange(int(np.prod(data_shape)), dtype=torch.float32).reshape(data_shape)
                if normalize_fn is not None:
                    out = normalize_fn.denormalize(out, noise_mask, remove_padding=remove_padding)
                return out

        B, N_nu, F, K = 3, 2, 3, 5
        batch = {
            "x": torch.randn(B, 4, 6),
            "x_mask": torch.ones(B, 4),
            "conditions": torch.randn(B, 2),
            "conditions_mask": torch.ones(B),
            "classification": torch.zeros(B, dtype=torch.long),
            "x_invisible": torch.randn(B, N_nu, F),
            "x_invisible_mask": torch.ones(B, N_nu),
        }
        model = FakeModel()
        sampler = FakeSampler()

        out = generate_neutrino_candidates(
            model,
            batch,
            sampler,
            K=K,
            num_ddim_steps=4,
            device=torch.device("cpu"),
            parallel_chains=2,
        )

        self.assertEqual(out.shape, (K, B, N_nu, F))
        self.assertEqual(sampler.data_shapes, [(2 * B, N_nu, F), (2 * B, N_nu, F), (B, N_nu, F)])
        self.assertEqual(model.call_batch_sizes, [(2 * B, 2 * B, 2 * B), (2 * B, 2 * B, 2 * B), (B, B, B)])


if __name__ == "__main__":
    unittest.main()
