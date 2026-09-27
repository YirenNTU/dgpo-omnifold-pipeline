from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_block_natural_gradient as experiment


def settings() -> dict:
    return {
        "reward_source_dir": "/reward",
        "gradient_source_dir": "/gradient",
        "output_dir": "/output",
        "expected_reward_schema": "reward-schema",
        "expected_gradient_schema": "gradient-schema",
        "expected_direction_schema": "direction-schema",
        "expected_source_wandb_run": "entity/project/c4a91e07",
        "expected_gradient_wandb_id": "h4grad01",
        "expected_policy_step": 1110,
        "workers": 16,
        "cpus_per_worker": 2,
        "K": 8,
        "curvature_events": 2048,
        "curvature_selection_seed": 11,
        "curvature_candidate_seed": 12,
        "curvature_timesteps": 8,
        "curvature_path_seed": 13,
        "basis_probe_rms": 1.0e-6,
        "maximum_metric_condition": 100.0,
        "ridge_floor_fraction": 1.0e-6,
        "zero_block_norm": 1.0e-12,
        "vanilla_parameter_rms": 1.0e-6,
        "distance_match_tolerance": 0.05,
        "distance_match_iterations": 3,
        "minimum_parameter_rms": 1.0e-8,
        "maximum_parameter_rms": 3.0e-5,
        "minimum_decisive_efficiency_gain": 2.0,
        "judge_events": 2048,
        "judge_selection_seed": 14,
        "judge_rollout_seed": 15,
        "score_batch_size": 32768,
        "save_direction_vectors": True,
        "upload_direction_vectors": False,
        "wandb": {"enabled": False},
    }


def h4grad_manifest() -> dict:
    return {
        "realizations": 4,
        "events_per_worker": 512,
        "workers": 16,
        "probe_events": 2048,
        "event_selection_seed": 2026091501,
        "probe_selection_seed": 2026091504,
    }


class TestBlockNaturalGradientProtocol(unittest.TestCase):
    def test_validates_exact_geometry_contract(self):
        cfg = experiment._validated_settings(settings())
        self.assertEqual(cfg["workers"], 16)
        self.assertEqual(cfg["K"], 8)
        self.assertEqual(cfg["curvature_timesteps"], 8)

    def test_rejects_changes_to_predeclared_contract(self):
        for key, value in (
            ("workers", 8),
            ("K", 4),
            ("curvature_timesteps", 4),
            ("curvature_events", 2047),
            ("judge_events", 2047),
            ("maximum_metric_condition", 1.0),
            ("distance_match_tolerance", 1.0),
            ("basis_probe_rms", 0.0),
        ):
            bad = copy.deepcopy(settings())
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment._validated_settings(bad)

    def test_upload_requires_saved_vectors(self):
        bad = settings()
        bad["save_direction_vectors"] = False
        bad["upload_direction_vectors"] = True
        with self.assertRaises(ValueError):
            experiment._validated_settings(bad)

    def test_curvature_and_judge_exclude_all_h4grad01_events(self):
        final_indices = torch.arange(42000)
        cfg = settings()
        curvature, judge, unused = experiment.select_unused_event_indices(
            final_indices, h4grad_manifest(), cfg
        )
        old_gradient, old_probe = (
            experiment.gradient_repro.select_disjoint_event_indices(
                final_indices, h4grad_manifest()
            )
        )
        old = set(torch.cat((old_gradient, old_probe)).tolist())
        self.assertEqual(len(curvature), cfg["curvature_events"])
        self.assertEqual(len(judge), cfg["judge_events"])
        self.assertEqual(unused, len(final_indices) - len(old))
        self.assertTrue(old.isdisjoint(curvature.tolist()))
        self.assertTrue(old.isdisjoint(judge.tolist()))
        self.assertTrue(set(curvature.tolist()).isdisjoint(judge.tolist()))

    def test_parameter_layout_creates_disjoint_gradient_block_basis(self):
        first = torch.nn.Parameter(torch.ones(2))
        second = torch.nn.Parameter(torch.ones(1))
        layout = [
            {"name": "left", "shape": [2], "start": 0, "stop": 2, "block": "A"},
            {"name": "right", "shape": [1], "start": 2, "stop": 3, "block": "B"},
        ]
        blocks = experiment.validate_parameter_layout(
            layout, [("left", first), ("right", second)], 3
        )
        gradient = torch.tensor([1.0, 2.0, 4.0])
        names, basis, q = experiment.build_block_basis(
            gradient, blocks, norm_floor=1.0e-12
        )
        self.assertEqual(names, ["A", "B"])
        self.assertAlmostEqual(experiment.vector_rms(basis[0]), 1.0, places=6)
        self.assertAlmostEqual(experiment.vector_rms(basis[1]), 1.0, places=6)
        self.assertEqual(float(torch.dot(basis[0], basis[1])), 0.0)
        self.assertTrue(torch.all(q > 0.0))

    def test_ridge_is_selected_from_condition_number_without_judge(self):
        ridge, diagnostics = experiment.ridge_for_condition(
            torch.diag(torch.tensor([1.0, 100.0], dtype=torch.float64)),
            maximum_condition=10.0,
            floor_fraction=1.0e-6,
        )
        self.assertAlmostEqual(ridge, 10.0)
        self.assertAlmostEqual(diagnostics["regularized_condition"], 10.0)

    def test_natural_solve_downweights_high_curvature_block(self):
        basis = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        direction, coefficients, diagnostics = (
            experiment.solve_block_natural_direction(
                torch.diag(torch.tensor([100.0, 1.0], dtype=torch.float64)),
                torch.tensor([1.0, 1.0], dtype=torch.float64),
                basis,
                maximum_condition=1000.0,
                floor_fraction=1.0e-9,
            )
        )
        self.assertGreater(float(coefficients[1]), 50.0 * float(coefficients[0]))
        self.assertGreater(float(direction.sum()), 0.0)
        self.assertGreater(diagnostics["linear_descent_derivative"], 0.0)

    def test_primary_decision_requires_distance_match_and_signed_descent(self):
        decisive = experiment.classify_result(
            zero_gap=0.24,
            vanilla_plus_gap=0.23,
            natural_plus_gap=0.21,
            natural_minus_gap=0.25,
            vp_distance_ratio=1.01,
            distance_tolerance=0.05,
            decisive_gain=2.0,
        )
        self.assertEqual(
            decisive["decision"],
            "block_policy_conditioning_is_material_bottleneck",
        )
        invalid = experiment.classify_result(
            zero_gap=0.24,
            vanilla_plus_gap=0.23,
            natural_plus_gap=0.21,
            natural_minus_gap=0.20,
            vp_distance_ratio=1.01,
            distance_tolerance=0.05,
            decisive_gain=2.0,
        )
        self.assertEqual(
            invalid["decision"], "invalid_distance_match_or_natural_sign"
        )
        no_improvement = experiment.classify_result(
            zero_gap=0.24,
            vanilla_plus_gap=0.25,
            natural_plus_gap=0.245,
            natural_minus_gap=0.26,
            vp_distance_ratio=1.01,
            distance_tolerance=0.05,
            decisive_gain=2.0,
        )
        self.assertEqual(
            no_improvement["decision"], "block_geometry_is_not_sufficient"
        )


if __name__ == "__main__":
    unittest.main()
