"""CPU checks for fixed-head tau transfer; no dataset, Ray, or remote jobs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))
from RL.DGPO_neutrino.tau_reward_transfer import (
    RELATIVE_STEPS, TauRewardTransferProbe, matrix_errors, paired_statistics, reward_statistics,
)


class StatisticsTests(unittest.TestCase):
    def test_errors_and_weighted_reward_statistics(self):
        delta = np.arange(9).reshape(3, 3)
        errors = matrix_errors(delta)
        self.assertEqual(errors["nn_error"], 8)
        self.assertAlmostEqual(errors["error"]**2,
                               errors["diagonal_error"]**2+errors["offdiagonal_error"]**2)
        r = np.array([[0., 2.], [4., 4.]])
        result = reward_statistics(r, np.array([3., 1.]))
        self.assertEqual(result["mean"], 1.75)
        self.assertEqual(result["best_of_k"], 2.5)
        self.assertEqual(result["worst_of_k"], 1.)
        self.assertEqual(result["median"], 2.)
        self.assertEqual(result["within_condition_std"], .75)
        self.assertGreater(result["global_std"], result["within_condition_std"])
        self.assertGreaterEqual(result["within_condition_ess_fraction"], .5)
        self.assertLessEqual(result["within_condition_ess_fraction"], 1.)

    def test_paired_constant_reward_gain_and_cij_improvement(self):
        n = 7
        before = np.zeros((n, 8)); after = before+2
        truth = np.zeros((n, 9)); old = np.ones((n, 9)); new = old*.5
        result = paired_statistics(before, after, old, new, truth, np.ones(n), replicates=20)
        self.assertEqual(result["reward"]["delta_mean"], 2)
        self.assertAlmostEqual(result["reward"]["delta_lo95"], 2)
        self.assertAlmostEqual(result["reward"]["delta_hi95"], 2)
        self.assertEqual(result["reward"]["fraction_improving"], 1)
        self.assertAlmostEqual(result["cij"]["error_changes"]["error"]["mean"], -1.5)
        self.assertAlmostEqual(result["cij"]["error_changes"]["nn_error"]["mean"], -.5)

    def test_joint_resampling_and_determinism(self):
        rng = np.random.default_rng(42)
        before = rng.normal(size=(12, 8)); after = before+rng.normal(size=(12, 1))
        truth = rng.normal(size=(12, 9)); old = truth+rng.normal(size=(12, 9)); new = truth+old*.2
        args = before, after, old, new, truth, np.arange(1, 13)
        first = paired_statistics(*args, replicates=30, seed=5)
        self.assertEqual(first, paired_statistics(*args, replicates=30, seed=5))
        zero = paired_statistics(before, before, old, old, truth, np.ones(12), replicates=30)
        self.assertEqual(zero["reward"]["delta_lo95"], 0)
        for row in zero["cij"]["error_changes"].values():
            self.assertEqual(row, dict(mean=0., lo95=0., hi95=0.))

    def test_invalid_inputs_fail_without_dropping_events(self):
        r = np.zeros((3, 8)); t = np.zeros((3, 9))
        with self.assertRaises(ValueError):
            paired_statistics(r, r[:2], t, t, t, np.ones(3))
        bad = r.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            paired_statistics(r, bad, t, t, t, np.ones(3))
        for w in (np.zeros(3), np.array([1, -1, 1])):
            with self.assertRaises(ValueError):
                paired_statistics(r, r, t, t, t, w)
        result = paired_statistics(r, r, t, t, t, np.array([1., 0, 0]), replicates=20)
        self.assertEqual(result["reward"]["delta_mean"], 0)


class ProbeTests(unittest.TestCase):
    def mock_terms(self):
        # Avoid importing optional EveNet/Lightning machinery for an observer's
        # CPU unit test; production uses the cycle's real shared Cij function.
        module = ModuleType("RL.DGPO_neutrino.conditional_tau_cycle")
        module.cij_terms = lambda *args: np.zeros((5, 9))
        return patch.dict(sys.modules, {module.__name__: module})

    def fixture(self, output):
        n = 5
        panel = dict(source_ids=np.arange(n), truth_deltas=np.zeros((n, 2, 2)),
                     event_weight=np.ones(n))
        head = torch.nn.Linear(2, 1).eval().requires_grad_(False)
        reward = SimpleNamespace(head=head, round_id=20, denominator_step=1880)
        cycle = SimpleNamespace(validation=panel, cfg=dict(validation_candidates=8, validation_seed=42),
                                output=output, world=1, rank=0, device=torch.device("cpu"), reward=reward)
        cycle.reference = torch.nn.Linear(2, 1).eval().requires_grad_(False)
        cycle.actor = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        cycle.actor.train(); cycle.actor[0].eval()
        cycle.progress = 0
        cycle.calls = []
        def generate(panel, folder, candidates, seed, physics):
            cycle.calls.append((candidates, seed, physics))
            # Simulate inner generator RNG activity and its top-level train
            # restoration, which would erase mixed modes without the wrapper.
            torch.rand(1); np.random.random()
            cycle.actor.train(True)
            return dict(rewards=np.full((n, 8), cycle.progress*.01),
                        cij=np.ones((n, 9))*(1-cycle.progress*.001),
                        deltas=np.ones((n, 8, 2, 2))*cycle.progress)
        cycle.generate = generate
        cycle.emit = lambda payload, step: cycle.calls.append((payload, step))
        cycle.audit_steps = []
        def audit(generated, folder, step):
            cycle.audit_steps.append(step)
            return dict(auc=.6, bce=.65, fit_total_steps=1000, fit_best_steps=950)
        cycle.transfer_audit = audit
        cfg = dict(allow_cpu_fixture=True, bootstrap_replicates=10, gradient_trace_enabled=True,
                   marginal_relative_steps=[])
        return cycle, cfg

    def test_complete_native_probe_and_saved_all_candidate_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            cfg["audit_relative_steps"] = [0, 50]
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
            metrics = {"train/optimizer_step_ran": 1,
                       "gradient_transfer/reconstruction_actual/relative_error": 0,
                       "gradient_transfer/norm/h4": 2.}
            torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()
            with self.mock_terms():
                probe.start()
                self.assertTrue(cycle.actor.training)
                self.assertFalse(cycle.actor[0].training)
                for step in range(1, 51):
                    probe.before_update({})
                    cycle.progress = step
                    probe.after_update(1920+step, metrics, 20)
            report = json.loads((probe.output/"report.json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual(report["conclusion"], "supports_local_transfer")
            self.assertEqual(cycle.audit_steps, [1920, 1970])
            self.assertEqual(report["measurements"]["50"]["fresh_audit"]["auc"], .6)
            self.assertEqual(list(map(int, report["measurements"])), list(RELATIVE_STEPS))
            with np.load(probe.output/"step-50/measurements.npz") as saved:
                self.assertEqual(saved["reward"].shape, (5, 8))
                self.assertEqual(saved["deltas"].shape, (5, 8, 2, 2))
                self.assertEqual(saved["truth_cij"].shape, (5, 9))
            self.assertEqual(len([x for x in cycle.calls if x == (8, 42, True)]), len(RELATIVE_STEPS)+1)
            self.assertEqual(report["measurements"]["0"]["no_update_replay_max_abs_error"], 0)
            self.assertEqual(report["measurements"]["50"]["gradient"]["norm/reward"], 2.)
            torch.testing.assert_close(torch.get_rng_state(), torch_state)
            for left, right in zip(np.random.get_state(), numpy_state):
                np.testing.assert_equal(left, right)

    def test_no_update_replay_failure_and_baseline_marginal_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            original_generate = cycle.generate
            calls = []
            def broken(*args):
                result = original_generate(*args)
                calls.append(1)
                if len(calls) == 2: result["rewards"][0, 0] += .01
                return result
            cycle.generate = broken
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
            with self.mock_terms(), self.assertRaisesRegex(ValueError, "replay failed"):
                probe.start()
            cycle.generate = original_generate
            cfg["marginal_relative_steps"] = [0]
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
            marginal = ModuleType("RL.DGPO_neutrino.diagnostics.tau_marginals")
            def monitor(panel, current, baseline, folder):
                self.assertTrue(baseline.is_file())
                return {"tau/marginal/baseline_available": 1}, {}
            marginal.monitor = monitor
            with self.mock_terms(), patch.dict(sys.modules, {marginal.__name__: marginal}):
                probe.start()
            self.assertEqual(probe.results["measurements"]["0"]["marginals"]["metrics"],
                             {"tau/marginal/baseline_available": 1})
            logged = [row for row in cycle.calls if isinstance(row[0], dict)][-1][0]
            self.assertEqual(logged["tau/transfer/marginal/baseline_available"], 1)

    def test_contract_and_update_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            probe = TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)
            with self.assertRaises(ValueError): probe.before_update({})
            with self.mock_terms():
                probe.start()
                with self.assertRaises(ValueError): probe.start()
                with self.assertRaises(ValueError): probe.after_update(1921, {}, 20)
                with self.assertRaises(ValueError): probe.after_update(1921, {"train/optimizer_step_ran": 1}, 21)
                with self.assertRaises(ValueError): probe.after_update(1922, {"train/optimizer_step_ran": 1}, 20)
                with self.assertRaises(ValueError): probe.after_update(1921, {"train/optimizer_step_ran": 1}, 20)
            cycle.reward.denominator_step = 1920
            with self.assertRaises(ValueError): probe._assert_frozen()
            cycle.reward.denominator_step = 1880
            with torch.no_grad(): cycle.reference.bias.add_(1)
            with self.assertRaises(ValueError): probe._assert_frozen()
            cycle.reference = torch.nn.Linear(2, 1).eval().requires_grad_(False)
            with torch.no_grad(): cycle.reward.head.bias.add_(1)
            with self.assertRaises(ValueError): probe._assert_frozen()

    def test_production_worker_and_endpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cycle, cfg = self.fixture(Path(tmp))
            with self.assertRaises(ValueError):
                TauRewardTransferProbe(cycle, {}, source_step=1920, reward_round=20)
            with self.assertRaises(ValueError):
                TauRewardTransferProbe(cycle, dict(cfg, relative_steps=[0, 50]), source_step=1920, reward_round=20)
            cycle.cfg["validation_candidates"] = 1
            with self.assertRaises(ValueError):
                TauRewardTransferProbe(cycle, cfg, source_step=1920, reward_round=20)


if __name__ == "__main__":
    unittest.main()
