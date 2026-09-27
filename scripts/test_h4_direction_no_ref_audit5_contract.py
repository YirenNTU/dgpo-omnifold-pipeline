"""Static contract for the no-reference H4 direction pilot."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_neutrino_backend import read_overlay_yaml


CONFIG_PATH = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_direction_no_ref_audit5_100step.yaml"
)
PARENT_PATH = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_minimal_alpha1_frozen_20step.yaml"
)


class TestH4DirectionNoReferenceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = read_overlay_yaml(CONFIG_PATH)
        cls.parent = read_overlay_yaml(PARENT_PATH)

    def test_small_fixed_reward_protocol(self) -> None:
        config = self.config
        experiment = config["experiment"]
        dgpo = config["dgpo"]
        recal = dgpo["adaptive_omnifold"]["recalibration"]

        self.assertEqual(
            experiment["protocol"],
            "h4-direction-no-reference-audit5-100step-v1",
        )
        self.assertEqual(
            (config["options"]["Training"]["epochs"], dgpo["steps_per_epoch"]),
            (10, 10),
        )
        self.assertEqual(
            (recal["crossfit_folds"], recal["crossfit_repeats"]), (2, 1)
        )
        self.assertEqual(
            (recal["min_iterations"], recal["max_iterations"]), (2, 2)
        )
        self.assertFalse(recal["iteration_one_only"])
        self.assertEqual(recal["max_reward_rounds"], 1)
        self.assertEqual(experiment["reference_snapshot_count"], 1)
        self.assertEqual(recal["tempering"], 1.0)

    def test_reference_trust_is_not_applied(self) -> None:
        trust = self.config["dgpo"]["reference_trust"]
        self.assertFalse(trust["enabled"])
        self.assertEqual(trust["coefficient"], 0.0)
        self.assertFalse(trust["adaptive_boundary"]["enabled"])

    def test_audits_are_measurement_only_every_five_epochs(self) -> None:
        adaptive = self.config["dgpo"]["adaptive_omnifold"]
        trigger = adaptive["trigger"]
        audit = adaptive["audit_fit"]

        self.assertTrue(adaptive["log_only"])
        self.assertEqual(adaptive["monitor_mode"], "raw_only")
        self.assertEqual(adaptive["staleness_every_n_epochs"], 5)
        self.assertIsNone(adaptive["staleness_every_n_steps"])
        self.assertTrue(adaptive["fixed_audit_panel"])
        self.assertTrue(adaptive["cache_event_inputs"])
        self.assertTrue(trigger["raw_audit_enabled"])
        self.assertFalse(trigger.get("fixed_schedule_log_raw_audit", False))
        self.assertFalse(trigger["warm_start_classifier"])
        self.assertFalse(trigger["rollback_to_best_on_plateau"])
        self.assertFalse(trigger["require_audit_saturation"])
        self.assertEqual(audit["steps"], 3000)
        self.assertEqual(audit["min_steps"], 3000)
        self.assertEqual(audit["validation_interval_steps"], 40)
        self.assertEqual(audit["progress_every_n_steps"], 40)
        self.assertFalse(audit["require_saturation"])
        self.assertFalse(audit["fail_if_unsaturated"])
        self.assertEqual(
            self.config["experiment"]["cold_h4_audit_min_optimizer_updates"],
            3000,
        )
        self.assertEqual(
            self.config["experiment"]["cold_h4_audit_fixed_optimizer_updates"],
            3000,
        )
        self.assertEqual(
            self.config["experiment"]["cold_h4_audit_policy_steps"],
            [0, 50, 100],
        )

    def test_gradient_direction_uses_the_fresh_audit_cadence(self) -> None:
        dgpo = self.config["dgpo"]
        monitor = dgpo["gradient_conflict"]

        self.assertTrue(monitor["enabled"])
        self.assertFalse(monitor["monitor_refit_lifecycle"])
        self.assertEqual(monitor["every_n_steps"], 50)
        self.assertEqual(monitor["blocks"], 8)
        self.assertEqual(monitor["events_per_block"], 512)
        self.assertEqual(
            self.config["experiment"]["gradient_direction_policy_steps"],
            [50, 100],
        )
        self.assertTrue(dgpo["log_parameter_update_rms"])
        self.assertTrue(dgpo["fail_on_skipped_optimizer_step"])

    def test_unrelated_training_settings_stay_fixed(self) -> None:
        current = self.config
        parent = self.parent
        self.assertEqual(current["dgpo"]["beta"], 1.0)
        self.assertFalse(current["dgpo"]["tarp"]["enabled"])
        for key in (
            "learning_rate",
            "learning_rate_body",
            "weight_decay",
            "decoupled_weight_decay",
            "learning_rate_warm_up_factor",
            "Components",
        ):
            self.assertEqual(
                current["options"]["Training"][key],
                parent["options"]["Training"][key],
                key,
            )

    def test_wandb_identity_is_readable(self) -> None:
        wb = self.config["logger"]["wandb"]
        self.assertEqual(wb["id"], "h4dir001")
        self.assertEqual(wb["group"], "H4 policy projection")
        self.assertEqual(
            wb["run_name"],
            "Does the DGPO direction track a fresh audit? | two-fold H4 | no reference trust",
        )
        self.assertLessEqual(len(wb["run_name"]), 96)
        self.assertNotIn("_", wb["run_name"])


if __name__ == "__main__":
    unittest.main()
