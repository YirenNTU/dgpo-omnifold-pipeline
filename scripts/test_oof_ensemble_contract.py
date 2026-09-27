from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "config/dgpo_omnifold_ztautau_10pct_oof_r1f2_control.yaml"
ENSEMBLE = ROOT / "config/dgpo_omnifold_ztautau_10pct_oof_r1f5_ensemble.yaml"
DIAGNOSTIC = ROOT / "config/dgpo_10pct_oof_ensemble_diagnostic.yaml"
CONTROL_TAG = (
    "dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_"
    "fullfit_gradlife_step320_seed42_oof_r1f2_control"
)
ENSEMBLE_TAG = (
    "dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_"
    "fullfit_gradlife_step320_seed42_oof_r1f5_ensemble"
)


def _read(path):
    return yaml.safe_load(path.read_text())


def _scrub(value, *, filename, run_tag):
    if isinstance(value, dict):
        return {
            key: _scrub(item, filename=filename, run_tag=run_tag)
            for key, item in value.items()
            if key != "crossfit_folds"
        }
    if isinstance(value, list):
        return [
            _scrub(item, filename=filename, run_tag=run_tag)
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(filename, "<ARM_OVERLAY>").replace(
            run_tag, "<ARM_RUN>"
        )
    return value


class TestOofEnsembleContract(unittest.TestCase):
    def test_dgpo_arms_change_only_fold_count_and_destinations(self):
        control, ensemble = _read(CONTROL), _read(ENSEMBLE)
        control_recal = control["dgpo"]["adaptive_omnifold"]["recalibration"]
        ensemble_recal = ensemble["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(control_recal["crossfit_folds"], 2)
        self.assertEqual(ensemble_recal["crossfit_folds"], 5)
        self.assertEqual(control_recal["crossfit_repeats"], 1)
        self.assertEqual(ensemble_recal["crossfit_repeats"], 1)
        self.assertEqual(
            control["options"]["Training"]["model_checkpoint_load_path"],
            ensemble["options"]["Training"]["model_checkpoint_load_path"],
        )
        self.assertNotEqual(
            control["options"]["Training"]["model_checkpoint_save_path"],
            ensemble["options"]["Training"]["model_checkpoint_save_path"],
        )
        self.assertEqual(
            _scrub(control, filename=CONTROL.name, run_tag=CONTROL_TAG),
            _scrub(ensemble, filename=ENSEMBLE.name, run_tag=ENSEMBLE_TAG),
        )

    def test_fixed_policy_screen_is_paired_and_focused(self):
        diagnostic = _read(DIAGNOSTIC)
        self.assertEqual(diagnostic["crossfit_repeat_arms"], [1])
        self.assertEqual(diagnostic["crossfit_fold_arms"], [2, 5])
        self.assertEqual(diagnostic["training_seeds"], [20260906, 20260907])
        self.assertFalse(diagnostic["run_controls"])
        self.assertEqual(diagnostic["expected_policy_step"], 320)
        self.assertIn("fullfit", diagnostic["overlay_config"])


if __name__ == "__main__":
    unittest.main()
