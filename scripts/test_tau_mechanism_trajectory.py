"""CPU observer checks for extended trajectories and a common independent judge."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_tau_reward_transfer as base_tests
from RL.DGPO_neutrino.tau_reward_transfer import (
    MECHANISM_RELATIVE_STEPS, RELATIVE_STEPS, TauRewardTransferProbe,
)


class FakeEnsemble:
    def __init__(self):
        self.judge_head = torch.nn.Linear(2, 1).eval().requires_grad_(False)
        self.selections = []

    @contextmanager
    def temporary_reward(self, reward, selection):
        if selection != "judge":
            raise ValueError("Only the common fifth judge is an evaluation head")
        self.selections.append(selection)
        original = reward.head
        clocks = (reward.round_id, reward.denominator_step)
        reward.head = self.judge_head
        try:
            yield self.judge_head
        finally:
            reward.head = original
            if clocks != (reward.round_id, reward.denominator_step):
                raise ValueError("Scoring modified the teacher clocks")


class MechanismTrajectoryTests(unittest.TestCase):
    def fixture(self, output, budget=50):
        cycle, cfg = base_tests.ProbeTests().fixture(output)
        cfg.update(mechanism_protocol=True,
                   relative_steps=[step for step in MECHANISM_RELATIVE_STEPS if step <= budget],
                   audit_relative_steps=[step for step in (0, 50, 200, 500) if step <= budget],
                   reward_arm="ensemble", ensemble_directory=None)
        return cycle, cfg

    def run_trajectory(self, probe, cycle, budget):
        metrics = {"train/optimizer_step_ran": 1,
                   "gradient_transfer/reconstruction_actual/relative_error": 0}
        with base_tests.ProbeTests().mock_terms():
            probe.start()
            for relative in range(1, budget + 1):
                cycle.progress = relative
                probe.before_update({})
                probe.after_update(1920 + relative, metrics, cycle.reward.round_id)

    def test_longer_endpoints_require_opt_in_and_complete_declared_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            for budget in (50, 200, 500):
                cycle, cfg = self.fixture(Path(tmp), budget)
                probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
                self.assertEqual(probe.relative_steps[-1], budget)
                self.assertIn(f"+{budget}", probe.results["primary_endpoint"])
                if budget > 50:
                    with self.assertRaises(ValueError):
                        TauRewardTransferProbe(cycle, dict(cfg, mechanism_protocol=False), source_step=1920, reward_round=20)
            cycle, cfg = self.fixture(Path(tmp), 500)
            for steps in ([0, 50, 500], list(MECHANISM_RELATIVE_STEPS[:-1]), [0, 1, 50], [0, True, 50]):
                with self.assertRaises(ValueError):
                    TauRewardTransferProbe(cycle, dict(cfg, relative_steps=steps), source_step=1920, reward_round=20)

    def test_500_step_probe_measures_all_declared_endpoints_and_labels_own_head_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp), 500)
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
            self.run_trajectory(probe, cycle, 500)
            report = json.loads((probe.output / "report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(tuple(map(int, report["measurements"])), MECHANISM_RELATIVE_STEPS)
            self.assertEqual(cycle.audit_steps, [1920, 1970, 2120, 2420])
            self.assertEqual(report["reward_evaluator"], "installed_training_head")
            self.assertFalse(report["comparable_across_arms"])
            self.assertIn("incomparable", report["reward_comparison_scope"])
            self.assertEqual(report["measurements"]["500"]["reward"]["delta_mean"], 5.)
            with self.assertRaises(ValueError):
                probe.after_update(2421, {"train/optimizer_step_ran": 1}, 20)

    def test_common_judge_rescores_same_candidates_preserves_training_rewards_and_controls_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            ensemble = FakeEnsemble()
            teacher = cycle.reward.head
            head_before = copy.deepcopy(teacher.state_dict())
            reference_before = copy.deepcopy(cycle.reference.state_dict())
            candidates, score_calls, candidate_records = [], [], []
            generate = cycle.generate
            def retain_generated(*args):
                result = generate(*args)
                candidates.append(result)
                if args[1].name == "candidates":
                    candidate_records.append(result)
                return result
            cycle.generate = retain_generated
            def score(judged, candidate_folder, output, step, phase):
                self.assertIs(cycle.reward.head, ensemble.judge_head)
                original = candidate_records[-1]
                self.assertIs(judged["deltas"], original["deltas"])
                self.assertFalse(np.shares_memory(judged["rewards"], original["rewards"]))
                self.assertEqual(candidate_folder.name, "candidates")
                score_calls.append((step, phase))
                judged["rewards"][:] = 3 - .02 * cycle.progress
                return {"auc": .7, "bce": .65, "mean": float(judged["rewards"].mean())}
            cycle._score_transfer_head = score
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20, ensemble=ensemble)
            self.run_trajectory(probe, cycle, 50)
            report = json.loads((probe.output / "report.json").read_text())
            self.assertEqual(report["reward_evaluator"], "common_independent_judge")
            self.assertTrue(report["comparable_across_arms"])
            self.assertEqual(report["conclusion"], "supports_local_failure")
            final = report["measurements"]["50"]
            self.assertEqual(final["reward"]["delta_mean"], -1.)
            self.assertEqual(final["training_reward"]["delta_mean"], .5)
            self.assertEqual([step for step, _ in score_calls], [1920 + step for step in RELATIVE_STEPS])
            self.assertTrue(all(phase == "common_judge" for _, phase in score_calls))
            self.assertEqual(len(candidates), len(RELATIVE_STEPS) + 1)
            self.assertIs(cycle.reward.head, teacher)
            self.assertEqual((cycle.reward.round_id, cycle.reward.denominator_step), (20, 1880))
            for name, value in head_before.items():
                torch.testing.assert_close(teacher.state_dict()[name], value)
            for name, value in reference_before.items():
                torch.testing.assert_close(cycle.reference.state_dict()[name], value)
            with np.load(probe.output / "step-50/measurements.npz") as saved:
                np.testing.assert_equal(saved["training_reward"], .5)
                np.testing.assert_equal(saved["judge_reward"], 2.)
                np.testing.assert_array_equal(saved["reward"], saved["judge_reward"])

    def test_common_judge_loader_and_scoring_error_restore_training_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            ensemble = FakeEnsemble()
            teacher = cycle.reward.head
            loaded = []
            module = ModuleType("RL.DGPO_neutrino.tau_classifier_ensemble_probe")
            class FakeLoader:
                @staticmethod
                def load(path, reward, device):
                    loaded.append((path, reward, device))
                    return ensemble
            module.TauClassifierEnsemble = FakeLoader
            cfg["ensemble_directory"] = Path(tmp) / "common-ensemble"
            with patch.dict(sys.modules, {module.__name__: module}):
                probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
            self.assertEqual(len(loaded), 1)
            def fail(*args):
                raise RuntimeError("fixture judge scoring failed")
            cycle._score_transfer_head = fail
            with base_tests.ProbeTests().mock_terms(), self.assertRaisesRegex(RuntimeError, "judge scoring failed"):
                probe.start()
            self.assertIs(cycle.reward.head, teacher)
            probe._assert_frozen()

    def test_mutating_common_judge_fails_frozen_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            ensemble = FakeEnsemble()
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20, ensemble=ensemble)
            with torch.no_grad():
                ensemble.judge_head.weight.add_(1)
            with self.assertRaisesRegex(ValueError, "judge weights changed"):
                probe._assert_frozen()

    def test_offrank_zero_judge_scores_features_without_collected_physics_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            cycle.rank = 1
            ensemble = FakeEnsemble()
            teacher = cycle.reward.head
            def score(generated, *args):
                self.assertEqual(set(generated), {"features"})
                self.assertIs(cycle.reward.head, ensemble.judge_head)
                return None
            cycle._score_transfer_head = score
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20, ensemble=ensemble)
            judged, metrics = probe._score_common_judge({"features": np.ones((5, 2))}, Path(tmp) / "candidates", Path(tmp), 0)
            self.assertEqual(set(judged), {"features"})
            self.assertEqual(metrics, {})
            self.assertIs(cycle.reward.head, teacher)


if __name__ == "__main__":
    unittest.main()
