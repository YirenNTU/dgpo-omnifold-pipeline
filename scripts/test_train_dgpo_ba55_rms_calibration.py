from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_dgpo_ba55_rms_calibration as launch
from train_neutrino_backend import read_overlay_yaml


EPSILON = {
    "epsilon_rms": 1.0e-6,
    "temperature": 2.0,
    "tempering": 0.5,
}


class TestBa55RmsCalibrationLauncher(unittest.TestCase):
    def _config(self, path: Path) -> dict:
        config = read_overlay_yaml(path)
        launch.apply_policy_lr_only(config, EPSILON)
        return config

    def test_all_predeclared_arms_pass_and_are_paired(self):
        native = self._config(launch.NATIVE)
        calibrated = self._config(launch.CALIBRATED)
        lr1e5 = self._config(launch.LR1E5)
        launch.assert_contract(native)
        launch.assert_contract(calibrated)
        launch.assert_contract(lr1e5)
        launch.assert_paired_configs(native, EPSILON)
        launch.assert_paired_configs(calibrated, EPSILON)
        launch.assert_paired_configs(lr1e5, EPSILON)
        self.assertEqual(
            launch._paired_signature(native),
            launch._paired_signature(calibrated),
        )
        self.assertEqual(
            native["dgpo"]["adaptive_omnifold"]["recalibration"]["tempering"],
            0.75,
        )
        self.assertEqual(lr1e5["options"]["Training"]["learning_rate"], 1.0e-5)
        self.assertEqual(
            launch._lr_agnostic_paired_signature(native),
            launch._lr_agnostic_paired_signature(lr1e5),
        )

    def test_lr_arm_rejects_non_lr_training_change(self):
        lr1e5 = self._config(launch.LR1E5)
        lr1e5["dgpo"]["K"] = 16
        with self.assertRaises(ValueError):
            launch.assert_paired_configs(lr1e5, EPSILON)

    def test_contract_rejects_target_or_reward_changes(self):
        calibrated = self._config(launch.CALIBRATED)
        mutations = (
            (("dgpo", "parameter_update_rms_calibration", "target_rms"), 1.0e-4),
            (("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"), 2),
            (("dgpo", "adaptive_omnifold", "recalibration", "tempering"), 0.5),
            (("dgpo", "adaptive_omnifold", "audit_fit", "min_steps"), 999),
        )
        for path, value in mutations:
            bad = copy.deepcopy(calibrated)
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                launch.assert_contract(bad)

    def test_endpoint_requires_saturated_1000_step_cold_audit(self):
        row = {
            "global_step": 1,
            "raw_auc_gap": 0.2,
            "raw_audit_training_steps": 1200,
            "raw_audit_saturated": 1.0,
        }
        checkpoint = {
            "dgpo_adaptive_omnifold_state": {"probe_history": [row]}
        }
        self.assertIs(launch._assert_endpoint_audit(checkpoint), row)
        for key, value in (
            ("raw_audit_training_steps", 999),
            ("raw_audit_saturated", 0.0),
        ):
            bad = copy.deepcopy(checkpoint)
            bad["dgpo_adaptive_omnifold_state"]["probe_history"][0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                launch._assert_endpoint_audit(bad)


if __name__ == "__main__":
    unittest.main()
