from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_dgpo_h4_resume50_refit10_m4 as resume
from train_dgpo_h4_fresh_ensemble_pilot import apply_epsilon_result
from train_neutrino_backend import read_overlay_yaml


class TestH4Resume50Refit10M4(unittest.TestCase):
    def setUp(self):
        self.config = read_overlay_yaml(resume.DEFAULT)
        apply_epsilon_result(
            self.config,
            {"epsilon_rms": 3.0e-7, "temperature": 2.0, "tempering": 0.5},
        )

    def test_contract_preserves_adam_and_refits_four_members_every_ten(self):
        resume.assert_contract(self.config)
        recal = self.config["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertFalse(recal["reset_optimizer_state_on_install"])
        self.assertTrue(recal["refit_once_on_resume"])
        self.assertTrue(recal["refit_once_fail_closed"])
        self.assertEqual(recal["crossfit_repeats"] * recal["crossfit_folds"], 4)
        self.assertIsNone(recal["max_reward_rounds"])
        self.assertEqual(
            self.config["options"]["Training"]["epochs"],
            resume.UNBOUNDED_EPOCH_SENTINEL,
        )
        self.assertTrue(
            self.config["dgpo"]["adaptive_omnifold"]["trigger"][
                "fixed_schedule_skip_staleness_audit"
            ]
        )
        self.assertEqual(self.config["logger"]["wandb"]["id"], "h4e10m4fs")

    def test_contract_rejects_optimizer_reset_and_undertrained_judge(self):
        for path, value in (
            (
                (
                    "dgpo",
                    "adaptive_omnifold",
                    "recalibration",
                    "reset_optimizer_state_on_install",
                ),
                True,
            ),
            (("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"), 3),
            (("dgpo", "adaptive_omnifold", "audit_fit", "min_steps"), 300),
            (("dgpo", "adaptive_omnifold", "trigger", "max_reward_age_epochs"), 2),
            (
                (
                    "dgpo",
                    "adaptive_omnifold",
                    "trigger",
                    "fixed_schedule_skip_staleness_audit",
                ),
                False,
            ),
            (("options", "Training", "epochs"), 100),
        ):
            bad = copy.deepcopy(self.config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                resume.assert_contract(bad)

    def test_source_requires_step50_round3_and_populated_adam_moments(self):
        increments = [
            {"state": {"weight": torch.ones(1)}} for _ in range(6)
        ]
        payload = {
            "state_dict": {"policy": torch.ones(1)},
            "dgpo_checkpoint_version": 1,
            "dgpo_next_epoch": 5,
            "dgpo_epoch_step": 0,
            "dgpo_optimizer_state_dict": {
                "optimizer": {
                    "state": {
                        0: {
                            "step": torch.tensor(50.0),
                            "exp_avg": torch.ones(1),
                            "exp_avg_sq": torch.ones(1),
                        }
                    },
                    "param_groups": [],
                },
                "scheduler": {"last_epoch": 50},
            },
            "dgpo_ref_state_dict": {"policy": torch.ones(1)},
            "dgpo_round_ref_state_dict": {"policy": torch.ones(1)},
            "dgpo_round_ref_sha256": "a" * 64,
            "dgpo_omnifold_reward_metadata": {},
            "dgpo_omnifold_reward_stack": {
                "reward": {
                    "increments": increments,
                    "increment_iterations": [1] * 6,
                    "increment_coefficients": [1.0 / 6.0] * 6,
                    "warm_start_state": {
                        "protocol": {
                            "scheme": "condition_sha256_v1",
                            "folds": 2,
                            "repeats": 3,
                        }
                    },
                }
            },
            "dgpo_adaptive_omnifold_state": {"reward_round_id": 3},
            "global_step": 50,
            "epoch": 4,
            "dgpo_reward_round_id": 3,
        }
        resume.assert_source_checkpoint(payload)

        bad_step = copy.deepcopy(payload)
        bad_step["global_step"] = 40
        with self.assertRaises(ValueError):
            resume.assert_source_checkpoint(bad_step)

        no_moments = copy.deepcopy(payload)
        no_moments["dgpo_optimizer_state_dict"]["optimizer"]["state"] = {}
        with self.assertRaisesRegex(ValueError, "AdamW"):
            resume.assert_source_checkpoint(no_moments)

        five_members = copy.deepcopy(payload)
        five_members["dgpo_omnifold_reward_stack"]["reward"]["increments"].pop()
        with self.assertRaisesRegex(ValueError, "six-member"):
            resume.assert_source_checkpoint(five_members)


if __name__ == "__main__":
    unittest.main()
