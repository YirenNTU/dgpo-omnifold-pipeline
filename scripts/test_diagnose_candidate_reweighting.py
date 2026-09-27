from __future__ import annotations

import copy
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))

# Keep the math tests independent of the optional Lightning/torchvision stack.
_debug_tool = types.ModuleType("evenet.utilities.debug_tool")


def _noop_time_decorator(name=None):
    del name

    def wrapper(function):
        return function

    return wrapper


_debug_tool.time_decorator = _noop_time_decorator
sys.modules.setdefault("evenet.utilities.debug_tool", _debug_tool)

import diagnose_candidate_reweighting as diagnostic


def settings():
    return {
        "source_dir": "/source", "output_dir": "/output",
        "expected_source_schema": "schema", "expected_policy_step": 1110,
        "workers": 16, "cpus_per_worker": 2, "K": 8,
        "score_batch_size": 32, "tilt_exponents": [0.25, 0.5, 0.75, 1.0],
        "physics_bins": 20, "response_bins": 4,
        "response_bootstrap_replicates": 10, "response_bootstrap_seed": 1,
        "primary": {
            "source": "h4_ensemble", "exponent": 0.75,
            "require_independent_judge_improvement": True,
            "require_target_jsd_mean_improvement": True,
            "minimum_target_components_improved": 3,
        },
        "legacy_control": {
            "enabled": True, "base_config": "base.yaml", "overlay_config": "old.yaml",
        },
        "wandb": {"enabled": True, "required": True, "id": "fixed"},
    }


class TestCandidateReweighting(unittest.TestCase):
    def test_settings_pin_production_exponent_and_live_id(self):
        self.assertEqual(
            diagnostic._validated_settings(settings())["tilt_exponents"],
            [0.25, 0.5, 0.75, 1.0],
        )
        for values in ([0.25, 1.0], [0.75, 0.5], [0.75, 0.75], [0.0, 0.75]):
            bad = copy.deepcopy(settings())
            bad["tilt_exponents"] = values
            with self.assertRaises(ValueError):
                diagnostic._validated_settings(bad)
        bad = copy.deepcopy(settings())
        bad["wandb"].pop("id")
        with self.assertRaisesRegex(ValueError, "fixed run id"):
            diagnostic._validated_settings(bad)
        bad = copy.deepcopy(settings())
        bad["primary"]["exponent"] = 1.0
        with self.assertRaisesRegex(ValueError, "reviewed H4 ensemble"):
            diagnostic._validated_settings(bad)

    def test_tilt_is_event_normalized_and_selects_higher_logit(self):
        logits = torch.tensor([
            [-2.0, 3.0],
            [0.0, 2.0],
            [2.0, 1.0],
        ])
        uniform = diagnostic.candidate_tilt_weights(logits, 0.0)
        torch.testing.assert_close(uniform, torch.full((2, 3), 1.0 / 3.0, dtype=torch.float64))
        weights = diagnostic.candidate_tilt_weights(logits, 0.75)
        torch.testing.assert_close(weights.sum(1), torch.ones(2, dtype=torch.float64))
        self.assertEqual(int(weights[0].argmax()), 2)
        self.assertEqual(int(weights[1].argmax()), 0)
        self.assertLess(diagnostic.weight_diagnostics(weights.numpy())["event_ess_fraction_mean"], 1.0)

    def test_weighted_jsd_improves_when_reward_selects_truthlike_candidate(self):
        truth = np.array([0.0, 1.0, 2.0, 3.0])
        candidates = np.stack([truth, truth + 10.0], axis=1)
        uniform = np.full((4, 2), 0.5)
        selected = np.tile(np.array([1.0, 0.0]), (4, 1))
        observables = {"target/tau_a_delta_theta": (truth, candidates)}
        baseline = diagnostic.weighted_physics_metrics(observables, uniform, bins=10)
        corrected = diagnostic.weighted_physics_metrics(observables, selected, bins=10)
        self.assertGreater(baseline["target_jsd_mean"], 0.0)
        self.assertAlmostEqual(corrected["target_jsd_mean"], 0.0)

    def test_weighted_response_rewards_truth_aligned_candidate(self):
        truth = np.tile(np.linspace(-1.0, 1.0, 20)[:, None], (1, 4))
        candidates = np.stack((truth, truth + 1.0), axis=1)
        uniform = np.full((20, 2), 0.5)
        selected = np.tile(np.array([1.0, 0.0]), (20, 1))
        baseline, _ = diagnostic.weighted_response_matrix_metrics(
            truth, candidates, uniform, bins=4
        )
        corrected, events = diagnostic.weighted_response_matrix_metrics(
            truth, candidates, selected, bins=4
        )
        for name in diagnostic.TARGET_COMPONENTS:
            self.assertLess(
                corrected[name]["mean_abs_bin_offset"],
                baseline[name]["mean_abs_bin_offset"],
            )
            self.assertEqual(events[name].shape, (20,))

    def test_periodic_correlation_is_invariant_to_full_phi_turns(self):
        theta = np.linspace(-1.0, 1.0, 30)
        phi = np.linspace(-np.pi, np.pi, 30, endpoint=False)
        truth = np.stack((theta, phi, theta**2, -phi), axis=1)
        candidates = np.stack((truth, truth.copy()), axis=1)
        candidates[:, 1, 1] += 2.0 * np.pi
        candidates[:, 1, 3] -= 2.0 * np.pi
        metrics = diagnostic.correlation_metrics(
            truth, candidates, np.full((30, 2), 0.5)
        )
        self.assertLess(metrics["mean_abs_pair_error"], 1.0e-12)
        self.assertLess(metrics["mean_abs_log_std_ratio"], 1.0e-12)

    def test_diagnosis_separates_reward_direction_from_proxy_mismatch(self):
        def row(judge, target, topology=0.2, corr=0.2, response=1.0):
            return {
                "judge": {"auc_gap": judge},
                "physics": {
                    "target_jsd_mean": target,
                    "topology_jsd_mean": topology,
                    "observables": {
                        f"target/{name}": {"jsd": target}
                        for name in diagnostic.TARGET_COMPONENTS
                    },
                },
                "correlation": {"mean_abs_pair_error": corr},
                "response_mean_abs_bin_offset": response,
            }

        baseline = row(0.3, 0.2)
        useful = diagnostic.compare_to_uniform(baseline, row(0.2, 0.1))
        self.assertEqual(
            useful["finding"],
            "candidate_direction_useful_dgpo_transfer_is_next_suspect",
        )
        proxy = diagnostic.compare_to_uniform(baseline, row(0.2, 0.3))
        self.assertEqual(proxy["finding"], "classifier_proxy_improves_without_target_closure")
        unsupported = diagnostic.compare_to_uniform(baseline, row(0.4, 0.1))
        self.assertEqual(unsupported["finding"], "classifier_candidate_direction_not_supported")

        # The coordinate response is a non-production proxy and cannot veto
        # the predeclared judge + target-JSD diagnosis.
        useful_with_bad_response = diagnostic.compare_to_uniform(
            baseline, row(0.2, 0.1, response=100.0)
        )
        self.assertEqual(
            useful_with_bad_response["finding"],
            "candidate_direction_useful_dgpo_transfer_is_next_suspect",
        )


if __name__ == "__main__":
    unittest.main()
