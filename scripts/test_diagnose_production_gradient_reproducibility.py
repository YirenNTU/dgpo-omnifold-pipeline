from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_production_gradient_reproducibility as experiment


def settings() -> dict:
    return {
        "source_dir": "/source",
        "output_dir": "/output",
        "expected_source_schema": "schema",
        "expected_source_wandb_run": "entity/project/c4a91e07",
        "expected_policy_step": 1110,
        "workers": 16,
        "cpus_per_worker": 2,
        "realizations": 4,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "K": 8,
        "gradient_timesteps": 8,
        "beta": 1.0,
        "policy_eval_t_min": 0.0,
        "policy_eval_t_max": 0.7,
        "event_selection_seed": 1,
        "candidate_seed": 2,
        "gradient_seed": 3,
        "probe_events": 2048,
        "probe_selection_seed": 4,
        "probe_rollout_seed": 5,
        "probe_epsilon_rms": 1e-6,
        "score_batch_size": 32768,
        "time_diagnostic_events_per_worker": 8,
        "save_direction_vectors": True,
        "upload_direction_vectors": False,
        "wandb": {"enabled": False},
    }


class TestProductionGradientProtocol(unittest.TestCase):
    def test_validates_exact_four_gradient_contract(self):
        cfg = experiment._validated_settings(settings())
        self.assertEqual(cfg["realizations"], 4)
        self.assertEqual(cfg["events_per_worker"], 512)
        self.assertEqual(cfg["gradient_timesteps"], 8)

    def test_rejects_nonproduction_sharding_shape(self):
        for key, value in (
            ("realizations", 3),
            ("workers", 8),
            ("events_per_worker", 256),
            ("event_microbatch_size", 127),
            ("K", 1),
            ("gradient_timesteps", 4),
            ("probe_epsilon_rms", 0.0),
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

    def test_gradient_and_probe_event_identities_are_disjoint(self):
        cfg = settings()
        total = (
            cfg["realizations"] * cfg["workers"] * cfg["events_per_worker"]
            + cfg["probe_events"]
            + 19
        )
        gradient, probe = experiment.select_disjoint_event_indices(
            torch.arange(total), cfg
        )
        self.assertEqual(
            len(gradient),
            cfg["realizations"] * cfg["workers"] * cfg["events_per_worker"],
        )
        self.assertEqual(len(probe), cfg["probe_events"])
        self.assertTrue(set(gradient.tolist()).isdisjoint(probe.tolist()))

    def test_reproducibility_summary_and_half_mean(self):
        gradients = [
            torch.tensor([1.0, 0.00, 0.5]),
            torch.tensor([1.0, 0.02, 0.5]),
            torch.tensor([1.0, -0.01, 0.5]),
            torch.tensor([1.0, 0.01, 0.5]),
        ]
        summary = experiment.summarize_reproducibility(
            gradients, {"first": [(0, 2)], "last": [(2, 3)]}
        )
        self.assertGreater(summary["mean_pairwise_cosine"], 0.99)
        self.assertGreater(summary["half_mean_cosine"], 0.99)
        self.assertEqual(tuple(summary["gbar"].shape), (3,))
        self.assertIn("first", summary["blocks"])
        self.assertEqual(
            experiment.production_variance_status(
                summary["mean_pairwise_cosine"], summary["min_pairwise_cosine"]
            ),
            "production_gradient_reproducible",
        )

    def test_variance_diagnosis_uses_gbar_probe(self):
        reproducibility = {
            "mean_pairwise_cosine": 0.4,
            "min_pairwise_cosine": 0.1,
        }
        probes = {}
        for index, delta in enumerate((-0.001, 0.001, -0.002, 0.0), start=1):
            probes[f"g{index}"] = {
                "plus_gap_delta": delta,
                "plus": {"judge_auc_gap": 0.24 + delta},
                "minus": {"judge_auc_gap": 0.25},
            }
        probes["gbar"] = {
            "plus_gap_delta": -0.01,
            "plus": {"judge_auc_gap": 0.23},
            "minus": {"judge_auc_gap": 0.26},
        }
        self.assertEqual(
            experiment.diagnose(reproducibility, probes),
            "monte_carlo_variance_materially_loses_h4_signal",
        )


if __name__ == "__main__":
    unittest.main()
