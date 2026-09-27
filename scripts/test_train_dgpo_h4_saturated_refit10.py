from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_saturated_refit10 import (  # noqa: E402
    AUDIT_STEPS,
    DEFAULT,
    REFRESH_STEPS,
    SEED_BUNDLE,
    TRACE_STEPS,
    _seed_bundle,
    assert_contract,
    summarize_endpoint,
)
from train_neutrino_backend import read_overlay_yaml  # noqa: E402


class SaturatedRefit10ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_overlay_yaml(DEFAULT)
        self.config["experiment"]["resolved_epsilon_rms"] = self.config[
            "options"
        ]["Training"]["learning_rate"]

    def test_checked_in_config_satisfies_contract(self) -> None:
        assert_contract(self.config)

    def test_matches_h4rep01_source_seed_and_mathematical_loss(self) -> None:
        dgpo = self.config["dgpo"]
        experiment = self.config["experiment"]
        trust = dgpo["reference_trust"]
        self.assertEqual(_seed_bundle(self.config), SEED_BUNDLE)
        self.assertEqual(experiment["source_policy_step"], 1110)
        self.assertFalse(experiment["h4_reward_formula_changed"])
        self.assertFalse(experiment["reference_formula_changed"])
        self.assertEqual(dgpo["advantage_estimator"], "leave_one_out_unscaled")
        self.assertEqual(dgpo["K"], 8)
        self.assertEqual(dgpo["beta"], 1.0)
        self.assertEqual(dgpo["beta_kl"], 0.0)
        self.assertEqual(trust["objective"], "velocity_mse")
        self.assertEqual(trust["coefficient"], 1.0)

    def test_reward_and_reference_refresh_together_without_adam_reset(self) -> None:
        experiment = self.config["experiment"]
        adaptive = self.config["dgpo"]["adaptive_omnifold"]
        trigger = adaptive["trigger"]
        recal = adaptive["recalibration"]
        self.assertEqual(tuple(experiment["reward_refresh_steps"]), REFRESH_STEPS)
        self.assertTrue(experiment["reference_anchor_dynamics_changed"])
        self.assertTrue(trigger["fixed_schedule_skip_staleness_audit"])
        self.assertTrue(trigger["fixed_schedule_log_raw_audit"])
        self.assertEqual(trigger["max_reward_age_epochs"], 1)
        self.assertEqual(recal["max_reward_rounds"], 5)
        self.assertFalse(recal["reset_optimizer_state_on_install"])
        self.assertFalse(recal["reset_adam_first_moment_on_install"])
        self.assertTrue(recal["fit"]["require_saturation"])

    def test_audits_and_gradient_lifecycle_are_pinned(self) -> None:
        dgpo = self.config["dgpo"]
        adaptive = dgpo["adaptive_omnifold"]
        self.assertEqual(tuple(self.config["experiment"]["audit_steps"]), AUDIT_STEPS)
        self.assertEqual(
            tuple(dgpo["gradient_transfer_trace"]["update_end_steps"]),
            TRACE_STEPS,
        )
        self.assertTrue(dgpo["gradient_conflict"]["monitor_refit_lifecycle"])
        self.assertEqual(adaptive["audit_fit"]["min_steps"], 1000)
        self.assertTrue(adaptive["audit_fit"]["fail_if_unsaturated"])
        self.assertFalse(adaptive["trigger"]["warm_start_classifier"])

    def test_contract_rejects_causal_confounders(self) -> None:
        mutations = (
            (
                (
                    "dgpo",
                    "adaptive_omnifold",
                    "recalibration",
                    "reset_optimizer_state_on_install",
                ),
                True,
            ),
            (("dgpo", "reference_trust", "coefficient"), 0.5),
            (("dgpo", "advantage_estimator"), "zscore"),
            (
                (
                    "dgpo",
                    "adaptive_omnifold",
                    "recalibration",
                    "fit",
                    "require_saturation",
                ),
                False,
            ),
            (
                (
                    "dgpo",
                    "adaptive_omnifold",
                    "trigger",
                    "fixed_schedule_log_raw_audit",
                ),
                False,
            ),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.config)
                cursor = changed
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaises(ValueError):
                    assert_contract(changed)

    @staticmethod
    def _checkpoint(gaps: list[float], *, training_steps: int = 1200) -> dict:
        rows = [
            {
                "global_step": step,
                "fixed_schedule_diagnostic_only": 1.0,
                "raw_auc": 0.5 + gap,
                "raw_auc_gap": gap,
                "raw_audit_training_steps": training_steps,
                "raw_audit_saturated": 1.0,
            }
            for step, gap in zip(AUDIT_STEPS, gaps)
        ]
        return {
            "global_step": 50,
            "dgpo_reward_round_id": 5,
            "dgpo_adaptive_omnifold_state": {"probe_history": rows},
        }

    def test_endpoint_uses_late_window_and_reports_stability(self) -> None:
        report = summarize_endpoint(
            self._checkpoint([0.36, 0.35, 0.34, 0.32, 0.31, 0.30])
        )
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["late_window_mean_gap"], 0.31)
        self.assertAlmostEqual(report["delta_vs_step0"], -0.05)
        self.assertAlmostEqual(report["total_variation"], 0.06)
        self.assertEqual(report["worst_excess_vs_step0"], 0.0)

    def test_endpoint_rejects_undertrained_audit(self) -> None:
        with self.assertRaisesRegex(ValueError, "updates=999"):
            summarize_endpoint(
                self._checkpoint(
                    [0.36, 0.35, 0.34, 0.32, 0.31, 0.30],
                    training_steps=999,
                )
            )


if __name__ == "__main__":
    unittest.main()
