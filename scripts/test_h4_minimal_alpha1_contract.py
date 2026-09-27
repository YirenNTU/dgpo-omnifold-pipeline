"""Static contract for the minimal alpha=1 frozen-reference H4 pilot."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_neutrino_backend import read_overlay_yaml


MINIMAL = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_minimal_alpha1_frozen_20step.yaml"
)
PARENT = ROOT / (
    "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble_5round.yaml"
)


class TestMinimalAlpha1Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = read_overlay_yaml(MINIMAL)
        cls.parent = read_overlay_yaml(PARENT)

    def test_exact_minimal_training_mechanism(self) -> None:
        config = self.config
        experiment = config["experiment"]
        dgpo = config["dgpo"]
        adaptive = dgpo["adaptive_omnifold"]
        recal = adaptive["recalibration"]

        self.assertEqual(
            experiment["protocol"], "h4-minimal-alpha1-frozen-20step-v1"
        )
        self.assertEqual(
            (experiment["rounds"], experiment["policy_updates_per_round"]),
            (1, 20),
        )
        self.assertEqual(
            (config["options"]["Training"]["epochs"], dgpo["steps_per_epoch"]),
            (2, 10),
        )
        self.assertEqual(dgpo["beta"], 1.0)
        self.assertEqual(dgpo["beta_kl"], 0.0)
        self.assertNotIn("\n  beta_kl:", MINIMAL.read_text())
        self.assertEqual(dgpo["advantage_estimator"], "leave_one_out_unscaled")
        self.assertEqual(dgpo["K"], 8)
        self.assertEqual(
            (
                dgpo["reference_trust"]["coefficient"],
                dgpo["reference_trust"]["objective"],
            ),
            (1.0, "velocity_mse"),
        )

        self.assertEqual(
            (recal["crossfit_repeats"], recal["crossfit_folds"]), (1, 2)
        )
        self.assertEqual(
            (recal["min_iterations"], recal["max_iterations"]), (1, 1)
        )
        self.assertTrue(recal["iteration_one_only"])
        self.assertEqual(recal["tempering"], 1.0)
        self.assertEqual(recal["adaptive_tempering"], {"enabled": False})
        self.assertEqual(recal["minimum_ess_fraction"], 0.0)

    def test_reward_and_reference_are_installed_once(self) -> None:
        adaptive = self.config["dgpo"]["adaptive_omnifold"]
        trigger = adaptive["trigger"]
        recal = adaptive["recalibration"]

        self.assertTrue(recal["bootstrap_on_start"])
        self.assertEqual(recal["max_reward_rounds"], 1)
        self.assertFalse(recal["refit_once_on_resume"])
        self.assertFalse(recal["scheduled_refit_fail_closed"])
        self.assertFalse(recal["reset_optimizer_state_on_install"])
        self.assertFalse(recal["reset_adam_first_moment_on_install"])
        self.assertEqual(recal["warm_start_iterations"], [])
        self.assertFalse(recal["warm_start_from_iteration_one"])
        self.assertGreater(trigger["max_reward_age_epochs"], 2)
        self.assertGreater(trigger["required_consecutive_checks"], 2)
        self.assertFalse(trigger["rollback_to_best_on_plateau"])

    def test_only_valid_cold_h4_audits_can_decide(self) -> None:
        experiment = self.config["experiment"]
        audit = self.config["dgpo"]["adaptive_omnifold"]["audit_fit"]

        self.assertEqual(experiment["cold_h4_audit_policy_steps"], [0, 10, 20])
        self.assertEqual(experiment["cold_h4_audit_min_optimizer_updates"], 1000)
        self.assertEqual(audit["min_steps"], 1000)
        self.assertTrue(audit["fail_if_unsaturated"])
        self.assertTrue(audit["disjoint_final_audit"])
        self.assertEqual(audit["checkpoint_selection_metric"], "balanced_accuracy")

    def test_deleted_mechanisms_stay_disabled(self) -> None:
        dgpo = self.config["dgpo"]
        recal = dgpo["adaptive_omnifold"]["recalibration"]

        self.assertFalse(dgpo["gradient_conflict"]["enabled"])
        self.assertFalse(dgpo["tarp"]["enabled"])
        self.assertFalse(dgpo["ztautau_metrics"]["enabled"])
        self.assertFalse(dgpo["variance_regularization"]["enabled"])
        self.assertFalse(dgpo["reference_trust"]["adaptive_boundary"]["enabled"])
        self.assertEqual(dgpo["projection_constraint"]["type"], "none")
        self.assertFalse(recal["acceptance_audit_enabled"])
        self.assertFalse(recal["topology_acceptance_audit_enabled"])
        self.assertFalse(recal["ess_aware_checkpoint_selection"]["enabled"])

        consensus = self.config["reward_config"]["omnifold"].get(
            "candidate_consensus"
        )
        self.assertTrue(
            consensus is None or consensus.get("apply_to_reward") is False
        )

    def test_native_adamw_settings_match_parent(self) -> None:
        current = self.config["options"]["Training"]
        parent = self.parent["options"]["Training"]
        for key in (
            "learning_rate",
            "learning_rate_body",
            "weight_decay",
            "decoupled_weight_decay",
            "learning_rate_warm_up_factor",
            "Components",
        ):
            self.assertEqual(current[key], parent[key], key)
        self.assertEqual(self.config["dgpo"]["lr_schedule"], self.parent["dgpo"]["lr_schedule"])

    def test_wandb_identity_is_readable_and_resumable(self) -> None:
        wb = self.config["logger"]["wandb"]
        name = wb["run_name"]
        self.assertEqual(wb["id"], "h4min001")
        self.assertEqual(
            name,
            "Can alpha 1 close the H4 gap? | two-fold frozen H4 | step-1110 pilot",
        )
        self.assertLessEqual(len(name), 96)
        self.assertNotIn("_", name)
        self.assertEqual(wb["resume"], "allow")
        self.assertFalse(wb["fresh_run"])


if __name__ == "__main__":
    unittest.main()
