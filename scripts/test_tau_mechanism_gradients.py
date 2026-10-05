"""CPU checks of full-call weighting, gradient isolation and reconstruction."""

from __future__ import annotations

from pathlib import Path
import ast
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))
from RL.DGPO_neutrino.tau_mechanism_gradients import (
    NativeGradientAccumulator, native_reward_event_terms,
    normalized_time_labels, observed_visible_opening_labels,
    summarize_replicas, velocity_reference_event_terms,
)
# Execute the production pure loss functions while avoiding optional trainer
# imports (Lightning/torchvision) that are unrelated to these CPU math checks.
native_source = Path(__file__).resolve().parents[1] / "evenet_dgpo/RL/DGPO_neutrino/dgpo_utils.py"
native_tree = ast.parse(native_source.read_text())
native_names = {"REFERENCE_TRUST_OBJECTIVE_VELOCITY_MSE", "REFERENCE_TRUST_OBJECTIVE_VP_PATH_KL", "VALID_REFERENCE_TRUST_OBJECTIVES"}
native_nodes = [node for node in native_tree.body if
               (isinstance(node, ast.FunctionDef) and node.name in {"build_dgpo_loss", "build_reference_trust_loss"})
               or (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in native_names for target in node.targets))]
native_namespace = {"torch": torch, "Tensor": torch.Tensor}
exec(compile(ast.Module(body=native_nodes, type_ignores=[]), str(native_source), "exec"), native_namespace)
build_dgpo_loss = native_namespace["build_dgpo_loss"]
build_reference_trust_loss = native_namespace["build_reference_trust_loss"]


def accumulator(**kwargs):
    return NativeGradientAccumulator(condition_names=("observed_a", "observed_b"), noise_band_names=("low", "high"), **kwargs)


class TestNativeMechanismGradients(unittest.TestCase):
    def test_condition_strata_read_only_observed_geometry_and_fixed_edges(self):
        a = torch.tensor([[1., 0., 0.]] * 5)
        b = torch.tensor([[-1., 0., 0.], [-.5, 3**.5/2, 0.], [0., 1., 0.], [.5, 3**.5/2, 0.], [1., 0., 0.]])
        batch = {f"lead_{leg}_visible_{axis}": value[:, j] for leg, value in (("a", a), ("b", b)) for j, axis in enumerate(("px", "py", "pz"))}
        labels = observed_visible_opening_labels(batch)
        self.assertEqual(labels.tolist(), [0, 0, 1, 2, 2])
        batch["event_id"] = torch.arange(5)
        batch["truth_tau_direction"] = torch.full((5, 3), float("nan"))
        self.assertTrue(torch.equal(labels, observed_visible_opening_labels(batch)))
        del batch["lead_a_visible_px"]
        with self.assertRaises(ValueError):
            observed_visible_opening_labels(batch)

    def test_native_time_labels_keep_boundaries_and_reject_out_of_range(self):
        times = torch.tensor([0., .7/3, 2*.7/3, .7], dtype=torch.float64)
        self.assertEqual(normalized_time_labels(times).tolist(), [0, 1, 2, 2])
        for values in (torch.tensor([.71]), torch.tensor([-.01]), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                normalized_time_labels(values)

    def test_full_gate_and_unequal_cell_mass_reconstruct_native_autograd(self):
        parameter = torch.nn.Parameter(torch.tensor([0.7, -0.5], dtype=torch.float64))
        unused = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
        saved = torch.tensor([17.0, 19.0], dtype=torch.float64)
        parameter.grad = saved.clone()
        x = torch.tensor([[[1., 2.], [3., -1.], [4., .5]], [[-.5, 2.], [1., 1.], [2., -1.]]], dtype=torch.float64)
        current = (x @ parameter).square()
        reference = torch.tensor([[.1, .4, .7], [.3, .6, .9]], dtype=torch.float64)
        advantage = torch.tensor([[1., -2., .5], [-1., 2., -.5]], dtype=torch.float64)
        native, _ = build_dgpo_loss(current, reference, advantage, .8, 2)
        gate = torch.sigmoid(.8 / 2 * (advantage * (current.detach() - reference)).sum(0)).detach()
        terms = native_reward_event_terms(current, advantage, gate)
        self.assertTrue(torch.allclose(terms.sum(), native))
        velocity = (x @ parameter).reshape(6, 1)
        ref_velocity = torch.zeros_like(velocity)
        mask = torch.tensor([[1.], [0.], [1.], [1.], [1.], [1.]], dtype=torch.float64)
        trust, _ = build_reference_trust_loss(velocity, ref_velocity, mask)
        reference_terms = velocity_reference_event_terms(velocity, ref_velocity, mask, K=2, B=3)
        self.assertTrue(torch.allclose(reference_terms.sum(), trust))
        trace = accumulator(parameters=(parameter, unused), dtype=torch.float64)
        trace.capture_call(reward_terms_b=terms, reference_terms_b=reference_terms,
                           condition_labels=torch.tensor([0, 0, 1]), noise_labels=torch.tensor([0, 1, 1]), weight=.25)
        self.assertTrue(torch.equal(parameter.grad, saved))
        self.assertIsNone(unused.grad)
        actual = torch.autograd.grad(.25 * (native + .6 * trust), (parameter, unused), allow_unused=True, retain_graph=True)
        actual_flat = torch.cat([actual[0], torch.zeros(1, dtype=torch.float64)])
        report = trace.summarize(actual_total_gradient=actual_flat, trust_coefficient=.6,
                                 parameter_blocks={"used": [(0, 2)], "unused": [(2, 3)]})
        self.assertTrue(report["reconstruction"]["passed"])
        self.assertLess(report["reconstruction"]["relative_error"], 1e-12)
        self.assertEqual(report["event_draw_counts"]["observed_b__low"], 0)
        self.assertIsNone(report["components"]["reward"]["parameter_blocks"]["unused"]["cells"]["coherence"])
        parameter.grad = None
        (.25 * (native + .6 * trust)).backward()
        self.assertTrue(torch.allclose(parameter.grad, actual[0], atol=1e-12, rtol=1e-12))
        self.assertIsNone(unused.grad)

    def test_microbatch_valid_element_correction_retains_full_denominator(self):
        full = torch.arange(12., dtype=torch.float64).reshape(6, 2).requires_grad_()
        mask = torch.tensor([[1., 1.], [1., 0.], [0., 0.], [1., 1.], [0., 1.], [1., 0.]], dtype=torch.float64)
        native, _ = build_reference_trust_loss(full, torch.zeros_like(full), mask)
        total = 0
        for start, stop in ((0, 1), (1, 3)):
            chunk = full.reshape(2, 3, 2)[:, start:stop].reshape(-1, 2)
            chunk_mask = mask.reshape(2, 3, 2)[:, start:stop].reshape(-1, 2)
            event_weight = (stop - start) / 3
            correction = chunk_mask.sum() / mask.sum() / event_weight
            terms = velocity_reference_event_terms(chunk, torch.zeros_like(chunk), chunk_mask, K=2, B=stop-start, weight_correction=correction)
            total = total + event_weight * terms.sum()
        self.assertTrue(torch.allclose(total, native))
        self.assertTrue(torch.allclose(torch.autograd.grad(total, full, retain_graph=True)[0], torch.autograd.grad(native, full)[0], atol=1e-12, rtol=1e-12))

    def test_weighted_cells_show_cancellation_without_cell_renormalization(self):
        trace = accumulator(vector_size=2, dtype=torch.float64)
        zero = torch.zeros(2, dtype=torch.float64)
        trace.add_flat("observed_a", "low", reward=torch.tensor([2., 0.]), reference=zero, weight=.5)
        trace.add_flat("observed_b", "high", reward=torch.tensor([-1., 0.]), reference=zero, weight=.5)
        report = trace.summarize(actual_total_gradient=torch.tensor([.5, 0.]), trust_coefficient=1.)
        family = report["components"]["reward"]["conditions"]
        self.assertAlmostEqual(family["coherence"], 1/3)
        self.assertEqual(family["pairs"]["observed_a__observed_b"]["cosine"], -1.)
        self.assertAlmostEqual(family["pairs"]["observed_a__observed_b"]["cross_dot"], -.5)

    def test_failed_reconstruction_hides_cancellation_interpretation(self):
        trace = accumulator(vector_size=2)
        trace.add_flat("observed_a", "low", reward=torch.tensor([1., 0.]), reference=torch.tensor([0., 1.]))
        report = trace.summarize(actual_total_gradient=torch.tensor([1., 0.]), trust_coefficient=1.)
        self.assertFalse(report["reconstruction"]["passed"])
        self.assertNotIn("components", report)

    def test_streamed_16_rank_reduction_uses_detached_sum_and_preserves_grad(self):
        trace = accumulator(vector_size=2)
        reward = torch.tensor([1., 2.], requires_grad=True)
        trace.add_flat("observed_a", "low", reward=reward, reference=torch.zeros(2), event_draw_count=3, weight=.5)
        shapes = []
        def all_sum(value):
            self.assertFalse(value.requires_grad)
            shapes.append(tuple(value.shape))
            value.mul_(16)
        trace.reduce_(world_size=16, reducer=all_sum)
        self.assertTrue(torch.equal(trace.total("reward"), reward.detach() * .5))
        self.assertEqual(trace.event_draw_counts["observed_a", "low"], 48.)
        self.assertEqual(trace.weighted_event_draw_mass["observed_a", "low"], 24.)
        self.assertEqual(shapes, [(2,)] * 8 + [(9,)])
        self.assertIsNone(reward.grad)
        with self.assertRaises(RuntimeError):
            trace.reduce_(world_size=16, reducer=all_sum)
        with self.assertRaises(RuntimeError):
            trace.add_flat("observed_a", "low", reward=reward, reference=torch.zeros(2))

    def test_cross_replica_self_dot_does_not_replace_noisy_norm(self):
        replicas = [accumulator(vector_size=2, dtype=torch.float64) for _ in range(4)]
        for trace, vector in zip(replicas, ([1., 4.], [1., -4.], [1., 4.], [1., -4.])):
            trace.add_flat("observed_a", "low", reward=torch.tensor(vector), reference=torch.zeros(2))
        report = summarize_replicas(replicas)
        total = report["components"]["reward"]["total"]
        self.assertAlmostEqual(total["cross_replica_self_dot"], -13/3)
        self.assertAlmostEqual(total["half_mean_cosine"], 1.)
        self.assertEqual(total["replica_norms"]["0"], 17**.5)
        self.assertNotIn("population_gradient_norm", total)

    def test_rejects_grad_gate_invalid_labels_and_nonfinite_vectors(self):
        with self.assertRaises(ValueError):
            native_reward_event_terms(torch.ones(2, 3), torch.ones(2, 3), torch.ones(3, requires_grad=True))
        parameter = torch.nn.Parameter(torch.ones(2))
        trace = accumulator(parameters=(parameter,))
        with self.assertRaises(ValueError):
            trace.capture_call(reward_terms_b=parameter, reference_terms_b=parameter,
                               condition_labels=torch.tensor([0, 2]), noise_labels=torch.tensor([0, 1]))
        with self.assertRaises(FloatingPointError):
            trace.add_flat("observed_a", "low", reward=torch.tensor([float("nan"), 1.]), reference=torch.zeros(2))
        with self.assertRaises(RuntimeError):
            trace.reduce_(world_size=16)


if __name__ == "__main__":
    unittest.main()
