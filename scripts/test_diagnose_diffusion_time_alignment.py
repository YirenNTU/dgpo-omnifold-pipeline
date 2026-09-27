from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_diffusion_time_alignment as experiment


def settings() -> dict:
    return {
        "reward_source_dir": "/reward",
        "gradient_source_dir": "/gradient",
        "geometry_source_dir": "/geometry",
        "output_dir": "/output",
        "expected_reward_schema": "reward-schema",
        "expected_gradient_schema": "gradient-schema",
        "expected_direction_schema": "direction-schema",
        "expected_geometry_schema": "geometry-schema",
        "expected_source_wandb_run": "entity/project/c4a91e07",
        "expected_gradient_wandb_id": "h4grad01",
        "expected_policy_step": 1110,
        "workers": 16,
        "cpus_per_worker": 2,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "K": 8,
        "beta": 1.0,
        "timesteps_per_band": 8,
        "time_bands": [
            {"name": "low", "min": 0.0, "max": 0.175},
            {"name": "mid_low", "min": 0.175, "max": 0.35},
            {"name": "mid_high", "min": 0.35, "max": 0.525},
            {"name": "high", "min": 0.525, "max": 0.7},
        ],
        "candidate_seed": 21,
        "gradient_seed": 22,
        "minimum_split_cosine": 0.8,
        "distance_candidate_seed": 23,
        "distance_timesteps": 8,
        "distance_path_seed": 24,
        "full_parameter_rms": 1.0e-6,
        "distance_match_tolerance": 0.05,
        "distance_match_iterations": 3,
        "minimum_parameter_rms": 1.0e-8,
        "maximum_parameter_rms": 3.0e-5,
        "judge_events": 2048,
        "judge_selection_seed": 25,
        "judge_rollout_seed": 26,
        "score_batch_size": 32768,
        "minimum_decisive_efficiency_gain": 2.0,
        "save_direction_vectors": True,
        "upload_direction_vectors": False,
        "wandb": {"enabled": False},
    }


def gradient_manifest() -> dict:
    return {
        "realizations": 4,
        "events_per_worker": 512,
        "workers": 16,
        "probe_events": 2048,
        "event_selection_seed": 101,
        "probe_selection_seed": 102,
    }


def geometry_manifest() -> dict:
    return {
        "curvature_events": 2048,
        "curvature_selection_seed": 103,
        "judge_events": 2048,
        "judge_selection_seed": 104,
    }


class TestDiffusionTimeAlignment(unittest.TestCase):
    def test_validates_exact_equal_band_contract(self):
        cfg = experiment._validated_settings(settings())
        self.assertEqual(tuple(row["name"] for row in cfg["time_bands"]), experiment.EXPECTED_BANDS)
        self.assertEqual(cfg["time_bands"][0]["min"], 0.0)
        self.assertEqual(cfg["time_bands"][-1]["max"], 0.7)

    def test_rejects_nonproduction_shape_or_bad_bands(self):
        for key, value in (
            ("workers", 8),
            ("events_per_worker", 256),
            ("event_microbatch_size", 64),
            ("K", 4),
            ("timesteps_per_band", 4),
            ("distance_timesteps", 4),
            ("minimum_split_cosine", 1.1),
        ):
            bad = copy.deepcopy(settings())
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment._validated_settings(bad)
        bad = copy.deepcopy(settings())
        bad["time_bands"][1]["min"] = 0.2
        with self.assertRaises(ValueError):
            experiment._validated_settings(bad)
        bad = copy.deepcopy(settings())
        bad["time_bands"][0]["max"] = 0.1
        bad["time_bands"][1]["min"] = 0.1
        with self.assertRaises(ValueError):
            experiment._validated_settings(bad)

    def test_new_judge_excludes_both_previous_experiments(self):
        final_indices = torch.arange(42000)
        panels = experiment.select_protocol_indices(
            final_indices, gradient_manifest(), geometry_manifest(), settings()
        )
        old_gradient, old_probe = (
            experiment.gradient_repro.select_disjoint_event_indices(
                final_indices, gradient_manifest()
            )
        )
        old_curvature, old_judge, _ = (
            experiment.geometry.select_unused_event_indices(
                final_indices, gradient_manifest(), geometry_manifest()
            )
        )
        previously_used = set(
            torch.cat((old_gradient, old_probe, old_curvature, old_judge)).tolist()
        )
        self.assertEqual(len(panels["direction"]), 8192)
        self.assertEqual(len(panels["distance"]), 2048)
        self.assertEqual(len(panels["judge"]), 2048)
        self.assertTrue(previously_used.isdisjoint(panels["judge"].tolist()))

    def test_time_summary_reconstructs_full_and_cancellation(self):
        halves = {
            "low": (torch.tensor([4.0, 0.0]), torch.tensor([4.0, 0.0])),
            "mid_low": (torch.tensor([0.0, 2.0]), torch.tensor([0.0, 2.0])),
            "mid_high": (torch.tensor([-2.0, 0.0]), torch.tensor([-2.0, 0.0])),
            "high": (torch.tensor([0.0, -2.0]), torch.tensor([0.0, -2.0])),
        }
        directions, diagnostics = experiment.summarize_time_directions(halves)
        self.assertTrue(torch.allclose(directions["full"], torch.tensor([0.5, 0.0])))
        self.assertAlmostEqual(diagnostics["full_split_cosine"], 1.0)
        self.assertLess(diagnostics["cancellation_ratio"], 1.0)
        self.assertIn("low_high", diagnostics["pairwise_cosines"])

    def test_decision_requires_reproducibility_distance_and_twofold_gain(self):
        decisive = experiment.classify_result(
            low_split_cosine=0.9,
            full_split_cosine=0.95,
            minimum_split_cosine=0.8,
            full_to_saved_cosine=0.9,
            low_distance_matched=True,
            rest_distance_matched=True,
            full_plus_gap_delta=-0.002,
            low_plus_gap_delta=-0.005,
            rest_plus_gap_delta=-0.001,
            low_plus_beats_minus=True,
            decisive_gain=2.0,
        )
        self.assertEqual(
            decisive["decision"], "low_noise_credit_is_materially_diluted"
        )
        invalid = experiment.classify_result(
            low_split_cosine=0.7,
            full_split_cosine=0.95,
            minimum_split_cosine=0.8,
            full_to_saved_cosine=0.9,
            low_distance_matched=True,
            rest_distance_matched=True,
            full_plus_gap_delta=-0.002,
            low_plus_gap_delta=-0.005,
            rest_plus_gap_delta=-0.001,
            low_plus_beats_minus=True,
            decisive_gain=2.0,
        )
        self.assertEqual(
            invalid["decision"], "invalid_or_underpowered_time_decomposition"
        )


if __name__ == "__main__":
    unittest.main()
