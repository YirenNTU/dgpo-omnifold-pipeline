"""Contract tests for the saturated-vs-BA70 launch pair."""

from __future__ import annotations

import copy
import unittest

import train_dgpo_h4_minimum_sufficient_ablation as launch
from train_neutrino_backend import read_overlay_yaml


class TestMinimumSufficientAblationLauncher(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "epsilon_rms": 1.0e-6,
            "temperature": 2.0,
            "tempering": 0.5,
        }

    def resolved(self, path):
        config = read_overlay_yaml(path)
        launch.apply_epsilon_result(config, self.result)
        return config

    def test_configs_are_paired_and_start_from_c4_last(self) -> None:
        saturated = self.resolved(launch.HISTORICAL_CONTROL)
        undertrained = self.resolved(launch.BA70)
        launch.assert_contract(undertrained)
        launch.assert_paired_configs(undertrained, self.result)
        self.assertEqual(
            saturated["options"]["Training"]["model_checkpoint_load_path"],
            str(launch.SOURCE_CHECKPOINT),
        )
        self.assertEqual(
            undertrained["options"]["Training"]["model_checkpoint_load_path"],
            str(launch.SOURCE_CHECKPOINT),
        )
        self.assertEqual(
            launch._training_signature(saturated),
            launch._training_signature(undertrained),
        )

    def test_contract_rejects_checkpoint50_and_unmatched_changes(self) -> None:
        config = self.resolved(launch.BA70)
        config["options"]["Training"]["model_checkpoint_load_path"] = (
            "/pscratch/not-the-c4-last/checkpoints/step=50.ckpt"
        )
        with self.assertRaisesRegex(ValueError, "old-classifier DGPO last"):
            launch.assert_contract(config)

        config = self.resolved(launch.BA70)
        config["dgpo"]["adaptive_omnifold"]["recalibration"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "outside the classifier"):
            launch.assert_paired_configs(config, self.result)

    def test_ba70_rule_is_exact_and_fail_closed(self) -> None:
        config = self.resolved(launch.BA70)
        fit = config["dgpo"]["adaptive_omnifold"]["recalibration"]["fit"]
        self.assertEqual(fit["min_steps_per_fold"], 1000)
        self.assertEqual(fit["minimum_sufficient_balanced_accuracy"], 0.70)
        self.assertEqual(fit["minimum_sufficient_confidence_z"], 0.0)
        self.assertEqual(fit["minimum_sufficient_required_consecutive"], 3)
        for key, bad in (
            ("minimum_sufficient_balanced_accuracy", 0.5),
            ("minimum_sufficient_confidence_z", -1.0),
            ("minimum_sufficient_required_consecutive", 0),
        ):
            changed = copy.deepcopy(config)
            changed["dgpo"]["adaptive_omnifold"]["recalibration"]["fit"][key] = bad
            with self.assertRaises(ValueError):
                launch.assert_contract(changed)


if __name__ == "__main__":
    unittest.main()
