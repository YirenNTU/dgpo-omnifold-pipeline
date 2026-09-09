"""
Unit tests for RL/DGPO_neutrino/dgpo_utils.py.

Run from repo root:
    python RL/DGPO_neutrino/test_dgpo_utils.py
"""

from __future__ import annotations

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
    REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
    REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
    adaptive_trust_backtracking_scales,
    adaptive_trust_update_scale,
    build_dgpo_loss,
    build_reference_trust_loss,
    compute_per_event_advantage,
    predict_x0_normalized_from_velocity_diffusion,
    rebase_trainable_params_for_extragradient_,
    reset_adam_first_moment,
    repeat_batch_for_candidates,
    sample_cosine_vp_path_kl_timesteps,
)
from RL.DGPO_neutrino.sampling import generate_neutrino_candidates


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

    def test_leave_one_out_unscaled_preserves_reward_scale(self):
        rewards = torch.tensor([[1.0], [2.0], [7.0]])
        advantages, weights = compute_per_event_advantage(
            rewards,
            estimator="leave_one_out_unscaled",
        )
        expected = torch.tensor([[-3.5], [-2.0], [5.5]])
        torch.testing.assert_close(advantages, expected)
        torch.testing.assert_close(weights, expected.abs())

    def test_leave_one_out_requires_multiple_candidates(self):
        with self.assertRaisesRegex(ValueError, "K>=2"):
            compute_per_event_advantage(
                torch.ones(1, 3),
                estimator="leave_one_out_unscaled",
            )


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

    def test_tensor_key_filter_omits_unused_targets_before_tiling(self):
        batch = {
            "x": torch.randn(4, 3, 2),
            "x_mask": torch.ones(4, 3),
            "large_unused_target": torch.randn(4, 100, 20),
            "meta": "kept",
        }
        out = repeat_batch_for_candidates(
            batch,
            8,
            tensor_keys={"x", "x_mask"},
        )
        self.assertEqual(out["x"].shape[0], 32)
        self.assertEqual(out["x_mask"].shape[0], 32)
        self.assertNotIn("large_unused_target", out)
        self.assertEqual(out["meta"], "kept")


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

    def test_event_microbatch_gradient_matches_full_batch(self):
        """Event chunks preserve the full-batch DGPO loss and gradient."""
        torch.manual_seed(23)
        K, B = 8, 5
        base = torch.randn(K, B)
        L_ref = torch.randn(K, B)
        advantages = torch.randn(K, B)

        full_cur = base.clone().requires_grad_(True)
        full_loss, _ = build_dgpo_loss(
            full_cur,
            L_ref,
            advantages,
            beta_dgpo=0.7,
            K=K,
        )
        full_loss.backward()

        chunked_cur = base.clone().requires_grad_(True)
        chunked_loss = chunked_cur.new_zeros(())
        for start, stop in ((0, 2), (2, 5)):
            loss, _ = build_dgpo_loss(
                chunked_cur[:, start:stop],
                L_ref[:, start:stop],
                advantages[:, start:stop],
                beta_dgpo=0.7,
                K=K,
            )
            chunked_loss = chunked_loss + loss * float(stop - start) / float(B)
        chunked_loss.backward()

        torch.testing.assert_close(chunked_loss, full_loss)
        torch.testing.assert_close(chunked_cur.grad, full_cur.grad)

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


class TestReferenceTrustLoss(unittest.TestCase):
    def test_sequential_vp_backward_matches_combined_gradient(self):
        """Two independent backwards preserve the exact summed objective."""

        torch.manual_seed(97)
        K, B, features = 3, 4, 5
        combined_model = torch.nn.Linear(features, features)
        sequential_model = torch.nn.Linear(features, features)
        sequential_model.load_state_dict(combined_model.state_dict())

        main_inputs = torch.randn(K * B, features)
        main_targets = torch.randn(K * B, features)
        trust_inputs = torch.randn(K * B, features)
        trust_reference = torch.randn(K * B, 1, features)
        trust_mask = torch.ones(K * B, 1, 1)
        reference_losses = torch.rand(K, B) + 0.25
        advantages = torch.randn(K, B)
        trust_coefficient = 0.37
        accumulation_weight = 0.125

        def _main_loss(module):
            prediction = module(main_inputs)
            current_losses = (
                (prediction - main_targets).square().mean(dim=-1).reshape(K, B)
            )
            loss, _ = build_dgpo_loss(
                current_losses,
                reference_losses,
                advantages,
                beta_dgpo=1.0,
                K=K,
            )
            return loss

        def _trust_loss(module):
            current_velocity = module(trust_inputs).reshape(K * B, 1, features)
            loss, _ = build_reference_trust_loss(
                current_velocity,
                trust_reference,
                trust_mask,
                L_ref_2d=reference_losses,
                objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
                path_kl_normalizer=20.0,
            )
            return loss

        combined_total = (
            _main_loss(combined_model)
            + trust_coefficient * _trust_loss(combined_model)
        ) * accumulation_weight
        combined_total.backward()

        sequential_main = _main_loss(sequential_model)
        (sequential_main * accumulation_weight).backward()
        sequential_trust = _trust_loss(sequential_model)
        (trust_coefficient * sequential_trust * accumulation_weight).backward()

        torch.testing.assert_close(
            combined_total.detach(),
            (
                sequential_main.detach()
                + trust_coefficient * sequential_trust.detach()
            )
            * accumulation_weight,
        )
        for combined_parameter, sequential_parameter in zip(
            combined_model.parameters(), sequential_model.parameters()
        ):
            torch.testing.assert_close(
                combined_parameter.grad,
                sequential_parameter.grad,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_cosine_vp_path_kl_sampler_is_stratified_and_normalized(self):
        torch.manual_seed(101)
        count, batch_size, total_strata, offset = 3, 7, 5, 1
        times, normalizer = sample_cosine_vp_path_kl_timesteps(
            count,
            batch_size,
            total_strata=total_strata,
            stratum_offset=offset,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(times.shape, (count, batch_size))
        self.assertEqual(times.dtype, torch.float32)
        self.assertAlmostEqual(normalizer, 20.0, places=10)
        self.assertTrue(torch.all((times >= 0.0) & (times <= 1.0)))

        # Under rho(t)=w(t)/Z, the CDF is the normalized change in
        # log(sigma(t)^2).  Every returned row must stay in its requested
        # global stratum even after the nonlinear inverse-CDF transform.
        _, _, sigma = get_logsnr_alpha_sigma(times.reshape(-1))
        log_sigma_sq = sigma.double().square().log().reshape(count, batch_size)
        log_sigma_sq_at_zero = -torch.nn.functional.softplus(
            torch.tensor(20.0, dtype=torch.float64)
        )
        quantiles = (log_sigma_sq - log_sigma_sq_at_zero) / normalizer
        for local_index in range(count):
            lower = (offset + local_index) / total_strata
            upper = (offset + local_index + 1) / total_strata
            self.assertTrue(torch.all(quantiles[local_index] >= lower - 2e-6))
            self.assertTrue(torch.all(quantiles[local_index] <= upper + 2e-6))

    def test_cosine_vp_path_kl_sampler_validates_partition_and_dtype(self):
        with self.assertRaisesRegex(ValueError, "within total_strata"):
            sample_cosine_vp_path_kl_timesteps(
                3,
                2,
                total_strata=4,
                stratum_offset=2,
            )
        with self.assertRaisesRegex(TypeError, "floating-point"):
            sample_cosine_vp_path_kl_timesteps(2, 2, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "logsnr_min"):
            sample_cosine_vp_path_kl_timesteps(
                2,
                2,
                logsnr_min=20.0,
                logsnr_max=-20.0,
            )

    def test_vp_path_kl_uses_dimension_sum_and_keeps_legacy_diagnostics(self):
        model_v = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [1.0, 5.0]],
            ],
            requires_grad=True,
        )
        ref_v = torch.zeros_like(model_v, requires_grad=True)
        mask = torch.tensor([[[1.0], [0.0]], [[1.0], [1.0]]])
        normalizer = 20.0

        loss, diagnostics = build_reference_trust_loss(
            model_v,
            ref_v,
            mask,
            L_ref_2d=torch.full((1, 2), 2.0),
            objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
            path_kl_normalizer=normalizer,
        )

        # Active per-row squared sums are 1^2+2^2=5 and
        # 2^2+0^2+1^2+5^2=30: 0.5 * 20 * mean([5, 30]) = 175.
        self.assertEqual(float(loss), 175.0)
        self.assertEqual(float(diagnostics["reference_trust/vp_path_kl"]), 175.0)
        self.assertEqual(float(diagnostics["reference_trust/loss"]), 175.0)
        self.assertAlmostEqual(
            float(diagnostics["reference_trust/velocity_mse"]),
            35.0 / 6.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(diagnostics["reference_trust/velocity_mse_ratio"]),
            (35.0 / 6.0) / 2.0,
            places=6,
        )

        loss.backward()
        self.assertIsNotNone(model_v.grad)
        self.assertIsNone(ref_v.grad)
        expected_gradient = normalizer / 2.0 * model_v.detach() * mask
        torch.testing.assert_close(model_v.grad, expected_gradient)

    def test_vp_path_kl_requires_positive_scalar_normalizer(self):
        args = (
            torch.ones(2, 1, 2),
            torch.zeros(2, 1, 2),
            torch.ones(2, 1, 1),
        )
        with self.assertRaisesRegex(ValueError, "requires path_kl_normalizer"):
            build_reference_trust_loss(
                *args,
                objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
            )
        for invalid in (0.0, float("nan"), torch.ones(2)):
            with self.subTest(normalizer=invalid):
                with self.assertRaisesRegex(ValueError, "path_kl_normalizer"):
                    build_reference_trust_loss(
                        *args,
                        objective=REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL,
                        path_kl_normalizer=invalid,
                    )

    def test_legacy_reference_trust_objective_is_unchanged(self):
        model_v = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
        ref_v = torch.zeros_like(model_v)
        loss, diagnostics = build_reference_trust_loss(
            model_v,
            ref_v,
            torch.ones(1, 1, 1),
            objective=REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE,
        )
        self.assertEqual(float(loss), 2.5)
        self.assertNotIn("reference_trust/vp_path_kl", diagnostics)

    def test_adaptive_backtracking_scales_are_absolute_and_decreasing(self):
        scales = adaptive_trust_backtracking_scales(
            0.8,
            factor=0.5,
            max_backtracks=4,
        )
        self.assertEqual(scales, (0.4, 0.2, 0.1, 0.05))

    def test_adaptive_backtracking_rejects_invalid_factor(self):
        with self.assertRaisesRegex(ValueError, "factor"):
            adaptive_trust_backtracking_scales(
                1.0,
                factor=1.0,
                max_backtracks=4,
            )

    def test_adaptive_boundary_scale(self):
        scale, hit = adaptive_trust_update_scale(
            0.0004,
            delta=0.001,
            warning_fraction=0.8,
        )
        self.assertEqual(scale, 1.0)
        self.assertFalse(hit)

        scale, hit = adaptive_trust_update_scale(
            0.0009,
            delta=0.001,
            warning_fraction=0.8,
        )
        self.assertAlmostEqual(scale, 0.5)
        self.assertFalse(hit)

        scale, hit = adaptive_trust_update_scale(
            0.001,
            delta=0.001,
            warning_fraction=0.8,
        )
        self.assertEqual(scale, 0.0)
        self.assertTrue(hit)

    def test_reset_adam_first_moment_preserves_second_moment(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        second_before = optimizer.state[parameter]["exp_avg_sq"].clone()

        reset_count = reset_adam_first_moment(optimizer)

        self.assertEqual(reset_count, 1)
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg"],
            torch.zeros_like(parameter),
        )
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg_sq"],
            second_before,
        )

    def test_extragradient_rebase_applies_lookahead_gradient_from_anchor(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        anchor = {"weight": torch.tensor([[1.0]])}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss = (model(torch.ones(1, 1)) - 4.0).square().mean()
        loss.backward()
        lookahead_gradient = model.weight.grad.detach().clone()
        self.assertEqual(float(lookahead_gradient), -4.0)

        count = rebase_trainable_params_for_extragradient_(model, anchor)
        optimizer.step()

        self.assertEqual(count, 1)
        # The gradient was evaluated at weight=2, but the SGD update starts
        # from the incumbent weight=1: 1 - 0.1*(-4) = 1.4.
        torch.testing.assert_close(model.weight, torch.tensor([[1.4]]))

    def test_extragradient_rebase_rejects_incomplete_snapshot(self):
        model = torch.nn.Linear(1, 1, bias=True)
        with self.assertRaisesRegex(KeyError, "keys differ"):
            rebase_trainable_params_for_extragradient_(
                model,
                {"weight": model.weight.detach().clone()},
            )

    def test_shared_policy_reference_match_is_zero(self):
        velocity = torch.randn(6, 2, 3)
        loss, diagnostics = build_reference_trust_loss(
            velocity,
            velocity.clone(),
            torch.ones(6, 2, 1),
        )
        self.assertAlmostEqual(float(loss), 0.0)
        self.assertAlmostEqual(
            float(diagnostics["reference_trust/velocity_mse"]), 0.0
        )

    def test_mask_and_reference_detach(self):
        model_v = torch.randn(4, 3, 2, requires_grad=True)
        ref_v = torch.randn(4, 3, 2, requires_grad=True)
        mask = torch.ones(4, 3, 1)
        mask[:, 2, :] = 0.0
        loss, diagnostics = build_reference_trust_loss(
            model_v,
            ref_v,
            mask,
            L_ref_2d=torch.ones(2, 2),
        )
        loss.backward()
        self.assertIsNotNone(model_v.grad)
        self.assertIsNone(ref_v.grad)
        self.assertTrue(
            torch.isfinite(
                diagnostics["reference_trust/velocity_mse_ratio"]
            )
        )

    def test_event_microbatch_weighting_preserves_masked_full_batch(self):
        torch.manual_seed(29)
        B, N, F = 5, 2, 3
        base = torch.randn(B, N, F)
        reference = torch.randn(B, N, F)
        mask = torch.tensor(
            [
                [[1.0], [1.0]],
                [[1.0], [0.0]],
                [[1.0], [1.0]],
                [[0.0], [1.0]],
                [[1.0], [1.0]],
            ]
        )

        full_cur = base.clone().requires_grad_(True)
        full_loss, _ = build_reference_trust_loss(full_cur, reference, mask)
        full_loss.backward()

        chunked_cur = base.clone().requires_grad_(True)
        chunked_loss = chunked_cur.new_zeros(())
        full_mask_mass = mask.sum()
        for start, stop in ((0, 2), (2, 5)):
            event_weight = float(stop - start) / float(B)
            local_loss, _ = build_reference_trust_loss(
                chunked_cur[start:stop],
                reference[start:stop],
                mask[start:stop],
            )
            trust_correction = (
                mask[start:stop].sum() / full_mask_mass / event_weight
            )
            chunked_loss = (
                chunked_loss + event_weight * trust_correction * local_loss
            )
        chunked_loss.backward()

        torch.testing.assert_close(chunked_loss, full_loss)
        torch.testing.assert_close(chunked_cur.grad, full_cur.grad)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            build_reference_trust_loss(
                torch.randn(4, 2, 3),
                torch.randn(4, 2, 5),
                torch.ones(4, 2, 1),
            )

    def test_mask_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "noise_mask"):
            build_reference_trust_loss(
                torch.randn(4, 2, 3),
                torch.randn(4, 2, 3),
                torch.ones(4, 4, 1),
            )


class TestGenerateNeutrinoCandidates(unittest.TestCase):
    def test_k8_rollout_disables_autograd(self):
        class FakeNormalizer:
            def denormalize(self, x, noise_mask, remove_padding=False):
                return x

        class GradAwareModel(torch.nn.Module):
            invisible_input_dim = 3

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(2.0))
                self.invisible_normalizer = FakeNormalizer()
                self.grad_enabled_during_forward: list[bool] = []

            def predict_diffusion_vector(
                self,
                noise_x,
                cond_x,
                time,
                mode,
                noise_mask=None,
            ):
                self.grad_enabled_during_forward.append(torch.is_grad_enabled())
                return noise_x * self.scale

        class OneStepSampler:
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
                noise = torch.ones(data_shape, dtype=torch.float32, requires_grad=True)
                return pred_fn(
                    noise_x=noise,
                    time=torch.zeros((data_shape[0],), dtype=torch.float32),
                )

        B, N_nu, F, K = 2, 2, 3, 8
        batch = {
            "x": torch.randn(B, 4, 6, requires_grad=True),
            "x_mask": torch.ones(B, 4),
            "conditions": torch.randn(B, 2, requires_grad=True),
            "conditions_mask": torch.ones(B),
            "classification": torch.zeros(B, dtype=torch.long),
            "x_invisible": torch.randn(B, N_nu, F, requires_grad=True),
            "x_invisible_mask": torch.ones(B, N_nu),
        }
        model = GradAwareModel()

        out = generate_neutrino_candidates(
            model,
            batch,
            OneStepSampler(),
            K=K,
            num_ddim_steps=20,
            device=torch.device("cpu"),
            parallel_chains=K,
        )

        self.assertEqual(out.shape, (K, B, N_nu, F))
        self.assertFalse(out.requires_grad)
        self.assertEqual(out.grad_fn, None)
        self.assertEqual(model.grad_enabled_during_forward, [False])
        self.assertIsNone(model.scale.grad)

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

        # The production K==parallel_chains path is one sampler call and returns
        # the grouped tensor directly without a Python candidate-group loop.
        model_all = FakeModel()
        sampler_all = FakeSampler()
        expanded_batch = repeat_batch_for_candidates(batch, K)
        out_all = generate_neutrino_candidates(
            model_all,
            batch,
            sampler_all,
            K=K,
            num_ddim_steps=4,
            device=torch.device("cpu"),
            parallel_chains=K,
            expanded_batch=expanded_batch,
        )
        self.assertEqual(out_all.shape, (K, B, N_nu, F))
        self.assertEqual(sampler_all.data_shapes, [(K * B, N_nu, F)])
        self.assertEqual(model_all.call_batch_sizes, [(K * B, K * B, K * B)])


if __name__ == "__main__":
    unittest.main()
