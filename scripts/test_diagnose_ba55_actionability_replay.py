from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_ba55_actionability_replay as replay


def settings():
    return {
        "source_dir": "/source",
        "output_dir": "/output",
        "expected_source_schema": "schema",
        "expected_policy_step": 1110,
        "reward_stage": "ba_lcb_55",
        "selected_members": [
            {"seed": 20260913, "fold": 1},
            {"seed": 20260913, "fold": 2},
        ],
        "workers": 16,
        "cpus_per_worker": 2,
        "probe_events": 1024,
        "K": 8,
        "gradient_events": 1024,
        "gradient_blocks": 8,
        "gradient_timesteps": 4,
        "gradient_seed": 1,
        "beta": 1.0,
        "policy_eval_t_min": 0.0,
        "policy_eval_t_max": 0.7,
        "raw_tempering": 0.75,
        "step_rms_values": [1e-6, 1e-5],
        "rollout_seeds": list(range(11, 19)),
        "score_batch_size": 32768,
        "minimum_plus_beats_zero_fraction": 0.75,
        "minimum_plus_beats_minus_fraction": 1.0,
        "bootstrap_seed": 3,
        "bootstrap_replicates": 1000,
        "wandb": {"enabled": False, "required": False},
    }


def physics_settings():
    cfg = settings()
    cfg["step_rms_values"] = [3e-5]
    cfg["minimum_plus_beats_minus_fraction"] = 0.75
    cfg["physics_endpoint"] = {
        "enabled": True,
        "target_components": list(replay.DEFAULT_TARGET_COMPONENTS),
        "physics_bins": 60,
        "response_bins": 12,
    }
    return cfg


class TestBA55ActionabilityReplay(unittest.TestCase):
    def test_contract_pins_two_folds_one_repeat_and_radii(self):
        cfg = replay._validated_settings(settings())
        self.assertEqual(
            [replay._member_key(row) for row in cfg["selected_members"]],
            [(20260913, 1), (20260913, 2)],
        )
        self.assertEqual(cfg["step_rms_values"], [1e-6, 1e-5])
        bad = copy.deepcopy(settings())
        bad["selected_members"][1]["seed"] = 20260914
        with self.assertRaisesRegex(ValueError, "one-repeat replay"):
            replay._validated_settings(bad)

    @staticmethod
    def _rows(delta, *, nonfinite=0.0):
        return [
            {
                "rollout_seed": seed,
                "zero_judge_auc_gap": 0.30,
                "plus_judge_auc_gap": 0.30 + delta,
                "minus_judge_auc_gap": 0.31,
                "candidate_nonfinite_fraction_max_rank": nonfinite,
            }
            for seed in range(8)
        ]

    def test_decision_selects_reliable_larger_radius(self):
        cfg = replay._validated_settings(settings())
        sweep = {
            "1e-06": {"seeds": self._rows(-0.001)},
            "1e-05": {"seeds": self._rows(-0.006)},
        }
        result = replay.diagnose_sweep(sweep, cfg)
        self.assertEqual(result["finding"], "larger_ba55_radius_supported")
        self.assertEqual(result["selected_step_rms"], 1e-5)

    def test_nonfinite_candidates_disqualify_radius(self):
        cfg = replay._validated_settings(settings())
        sweep = {
            "1e-06": {"seeds": self._rows(-0.001)},
            "1e-05": {"seeds": self._rows(-0.006, nonfinite=0.01)},
        }
        result = replay.diagnose_sweep(sweep, cfg)
        self.assertEqual(result["finding"], "base_ba55_radius_preferred")
        self.assertEqual(result["selected_step_rms"], 1e-6)
        self.assertFalse(result["radii"]["1e-05"]["reliable"])

    @staticmethod
    def _physics_rows(*, plus_delta=-0.01, minus_delta=0.01):
        rows = []
        for seed in range(8):
            zero_components = {
                component: 0.10 + 0.001 * seed
                for component in replay.DEFAULT_TARGET_COMPONENTS
            }
            zero_mean = sum(zero_components.values()) / 4.0
            rows.append({
                "rollout_seed": seed,
                "zero_target_jsd": zero_components,
                "plus_target_jsd": {
                    key: value + plus_delta for key, value in zero_components.items()
                },
                "minus_target_jsd": {
                    key: value + minus_delta for key, value in zero_components.items()
                },
                "zero_target_jsd_mean": zero_mean,
                "plus_target_jsd_mean": zero_mean + plus_delta,
                "minus_target_jsd_mean": zero_mean + minus_delta,
                "candidate_nonfinite_fraction_max_rank": 0.0,
            })
        return rows

    def test_physics_endpoint_accepts_replicated_target_improvement(self):
        cfg = replay._validated_settings(physics_settings())
        sweep = {"3e-05": {"seeds": self._physics_rows()}}
        result = replay.diagnose_physics_endpoint(sweep, cfg)
        self.assertEqual(
            result["finding"], "ba55_h4_physics_improvement_supported"
        )
        self.assertTrue(result["reliable"])
        self.assertLess(result["plus_minus_zero"]["ci90_high"], 0.0)
        self.assertEqual(result["plus_beats_zero_fraction"], 1.0)

    def test_physics_endpoint_rejects_wrong_direction(self):
        cfg = replay._validated_settings(physics_settings())
        sweep = {
            "3e-05": {
                "seeds": self._physics_rows(plus_delta=0.005, minus_delta=-0.005)
            }
        }
        result = replay.diagnose_physics_endpoint(sweep, cfg)
        self.assertEqual(
            result["finding"], "ba55_h4_physics_improvement_not_replicated"
        )
        self.assertFalse(result["reliable"])

    def test_physics_endpoint_requires_one_predeclared_radius(self):
        cfg = physics_settings()
        cfg["step_rms_values"] = [1e-5, 3e-5]
        with self.assertRaisesRegex(ValueError, "exactly one predeclared RMS"):
            replay._validated_settings(cfg)


if __name__ == "__main__":
    unittest.main()
