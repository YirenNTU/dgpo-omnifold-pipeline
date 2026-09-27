from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_reward_epsilon_sweep as sweep


class TestEpsilonSweep(unittest.TestCase):
    def test_settings_require_strictly_increasing_positive_epsilons(self):
        base = {
            "source_dir": "/source", "output_dir": "/output",
            "expected_source_schema": "schema", "expected_policy_step": 1,
            "workers": 16, "cpus_per_worker": 2,
            "epsilons_rms": [1e-7, 3e-7], "K": 8,
            "gradient_events": 16, "gradient_blocks": 2,
            "gradient_timesteps": 1, "gradient_seed": 1, "beta": 1.0,
            "policy_eval_t_min": 0.0, "policy_eval_t_max": 0.7,
            "rollout_seed": 2, "score_batch_size": 32, "physics_bins": 60,
            "response_bins": 12, "response_bootstrap_replicates": 10,
            "response_bootstrap_seed": 3, "wandb": {"enabled": False},
        }
        self.assertEqual(sweep._validated_settings(base)["epsilons_rms"], [1e-7, 3e-7])
        for values in ([3e-7, 1e-7], [1e-7, 1e-7], [0.0, 1e-7]):
            bad = copy.deepcopy(base)
            bad["epsilons_rms"] = values
            with self.assertRaises(ValueError):
                sweep._validated_settings(bad)

    def test_decision_requires_all_three_metrics_and_no_bad_component(self):
        zero = {
            "judge_auc_gap": 0.2,
            "response_mean_abs_bin_offset": 1.0,
            "jsd": {"mean": 0.10},
        }
        plus = {
            "judge_auc_gap": 0.19,
            "response_mean_abs_bin_offset": 0.9,
            "jsd": {"mean": 0.09},
        }
        bootstrap = {"theta": {"ci95_low": -0.02, "ci95_high": 0.01}}
        self.assertTrue(sweep.decide_epsilon(zero, plus, bootstrap)["eligible"])
        bad = copy.deepcopy(bootstrap)
        bad["theta"]["ci95_low"] = 0.001
        decision = sweep.decide_epsilon(zero, plus, bad)
        self.assertFalse(decision["eligible"])
        self.assertEqual(decision["significantly_worse_response_components"], ["theta"])

    def test_jsd_summary_uses_only_current_policy(self):
        result = sweep.summarize_jsd({
            "val_ztautau/jsd/current/a": 0.1,
            "val_ztautau/jsd/current/b": 0.3,
            "val_ztautau/jsd/ref/a": 0.9,
            "else": 4.0,
        })
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertEqual(result["count"], 2.0)

    def test_selection_uses_largest_fully_eligible_radius(self):
        rows = [
            {"epsilon_rms": 1e-7, "decision": {"eligible": True}},
            {"epsilon_rms": 3e-7, "decision": {"eligible": False}},
            {"epsilon_rms": 1e-6, "decision": {"eligible": True}},
        ]
        self.assertEqual(sweep.select_epsilon(rows)["epsilon_rms"], 1e-6)
        self.assertIsNone(sweep.select_epsilon([
            {"epsilon_rms": 1e-7, "decision": {"eligible": False}},
        ]))


if __name__ == "__main__":
    unittest.main()
