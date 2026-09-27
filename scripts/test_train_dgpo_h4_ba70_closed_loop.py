from __future__ import annotations

import copy
from pathlib import Path
import sys
import types
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch"] = types.ModuleType("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_h4_ba70_closed_loop import (  # noqa: E402
    AUDIT_STEPS,
    DEFAULT,
    OLD_CONFIG,
    _audit_trajectory,
    assert_contract,
)
from train_neutrino_backend import read_overlay_yaml  # noqa: E402


class BA70ClosedLoopContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_overlay_yaml(DEFAULT)
        self.config["experiment"]["resolved_epsilon_rms"] = self.config[
            "options"
        ]["Training"]["learning_rate"]

    def test_checked_in_config_satisfies_contract(self) -> None:
        assert_contract(self.config)

    def test_reward_stop_retains_exact_ba70_after_1000_step_floor(self) -> None:
        fit = self.config["dgpo"]["adaptive_omnifold"]["recalibration"]["fit"]
        self.assertEqual(fit["min_steps_per_fold"], 1000)
        self.assertEqual(fit["minimum_sufficient_balanced_accuracy"], 0.70)
        self.assertEqual(fit["minimum_sufficient_confidence_z"], 0.0)
        self.assertEqual(fit["minimum_sufficient_required_consecutive"], 3)
        self.assertFalse(fit["require_saturation"])

    def test_previous_1000_step_config_remains_resumable(self) -> None:
        old_config = read_overlay_yaml(OLD_CONFIG)
        old_config["experiment"]["resolved_epsilon_rms"] = old_config[
            "options"
        ]["Training"]["learning_rate"]
        assert_contract(old_config)

    def test_single_audit_is_saturated_and_selection_blind(self) -> None:
        adaptive = self.config["dgpo"]["adaptive_omnifold"]
        trigger = adaptive["trigger"]
        audit = adaptive["audit_fit"]
        self.assertTrue(trigger["fixed_schedule_skip_staleness_audit"])
        self.assertTrue(trigger["fixed_schedule_log_raw_audit"])
        self.assertFalse(trigger["warm_start_classifier"])
        self.assertEqual(audit["repeats"], 1)
        self.assertEqual(audit["min_steps"], 1000)
        self.assertTrue(audit["disjoint_final_audit"])
        self.assertTrue(audit["fail_if_unsaturated"])

    def test_policy_adam_is_continuous_after_weights_only_start(self) -> None:
        dgpo = self.config["dgpo"]
        recal = dgpo["adaptive_omnifold"]["recalibration"]
        self.assertEqual(dgpo["checkpoint_load_mode"], "weights_only")
        self.assertFalse(recal["reset_optimizer_state_on_install"])
        self.assertFalse(recal["reset_adam_first_moment_on_install"])

    def test_contract_rejects_trajectory_changing_mutations(self) -> None:
        mutations = (
            (("dgpo", "adaptive_omnifold", "recalibration", "reset_optimizer_state_on_install"), True),
            (("dgpo", "adaptive_omnifold", "trigger", "fixed_schedule_skip_staleness_audit"), False),
            (("dgpo", "adaptive_omnifold", "trigger", "fixed_schedule_log_raw_audit"), False),
            (("dgpo", "adaptive_omnifold", "audit_fit", "repeats"), 2),
            (("dgpo", "adaptive_omnifold", "recalibration", "fit", "minimum_sufficient_balanced_accuracy"), 0.75),
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

    def test_final_audit_summary_uses_all_six_boundaries(self) -> None:
        gaps = [0.20, 0.19, 0.17, 0.18, 0.14, 0.12]
        rows = [
            {
                "global_step": step,
                "fixed_schedule_diagnostic_only": 1.0,
                "raw_auc_gap": gap,
                "raw_audit_saturated": 1.0,
                "raw_audit_repeats": 1.0,
            }
            for step, gap in zip(AUDIT_STEPS, gaps)
        ]
        summary = _audit_trajectory(
            {"dgpo_adaptive_omnifold_state": {"probe_history": rows}}
        )
        self.assertTrue(summary["primary_success"])
        self.assertLess(summary["slope_per_10_steps"], 0.0)
        self.assertEqual(summary["audit_steps"], AUDIT_STEPS)


if __name__ == "__main__":
    unittest.main()
