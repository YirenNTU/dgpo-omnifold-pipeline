from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))

import train_dgpo_h4_fresh_ensemble_pilot as pilot
from train_neutrino_backend import read_overlay_yaml
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    AdaptiveOmniFoldState,
    reward_round_budget_exhausted,
    resolve_adaptive_config,
    scheduled_raw_refit_due,
)


class TestFreshEnsemblePilot(unittest.TestCase):
    def setUp(self):
        self.config = read_overlay_yaml(pilot.DEFAULT)
        self.result = {"epsilon_rms": 3e-7, "temperature": 2.0, "tempering": 0.5}
        pilot.apply_epsilon_result(self.config, self.result)

    def test_overlay_resolves_to_exact_experiment_contract(self):
        pilot.assert_contract(self.config)
        recal = self.config["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(recal["max_reward_rounds"], 5)
        self.assertEqual(recal["crossfit_repeats"] * recal["crossfit_folds"], 4)
        self.assertEqual(recal["warm_start_iterations"], [])
        self.assertFalse(self.config["dgpo"]["reference_trust"]["adaptive_boundary"]["enabled"])
        self.assertEqual(self.config["options"]["Training"]["learning_rate"], 3e-7)

    def test_ess15_overlay_is_a_single_change_paired_experiment(self):
        config = read_overlay_yaml(pilot.ESS15_DEFAULT)
        pilot.apply_epsilon_result(config, self.result)
        pilot.assert_contract(config)
        recal = config["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(
            recal["adaptive_tempering"],
            {
                "enabled": True,
                "target_ess_fraction": 0.15,
                "minimum": 0.10,
                "grid_steps": 14,
                "inherit_previous": False,
            },
        )
        self.assertEqual(recal["minimum_ess_fraction"], 0.0)
        self.assertFalse(recal["ess_aware_checkpoint_selection"]["enabled"])
        self.assertEqual(config["logger"]["wandb"]["id"], "h4e15r01")
        resolved = resolve_adaptive_config(config["dgpo"])
        self.assertTrue(resolved.adaptive_tempering_enabled)
        self.assertEqual(resolved.target_ess_fraction, 0.15)
        self.assertEqual(resolved.minimum_tempering, 0.10)
        self.assertEqual(resolved.tempering_grid_steps, 14)
        self.assertFalse(resolved.inherit_previous_tempering)

        for path, value in (
            (("dgpo", "beta"), 0.5),
            (
                (
                    "dgpo", "adaptive_omnifold", "recalibration",
                    "adaptive_tempering", "target_ess_fraction",
                ),
                0.20,
            ),
            (
                (
                    "dgpo", "adaptive_omnifold", "recalibration",
                    "ess_aware_checkpoint_selection", "enabled",
                ),
                True,
            ),
            (
                ("options", "Training", "model_checkpoint_save_path"),
                self.config["options"]["Training"]["model_checkpoint_save_path"],
            ),
        ):
            bad = copy.deepcopy(config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                pilot.assert_contract(bad)

    def test_long_reward_overlay_changes_only_reward_round_budget(self):
        config = read_overlay_yaml(pilot.LONG_REWARD_DEFAULT)
        pilot.apply_epsilon_result(config, self.result)
        pilot.assert_contract(config)
        experiment = config["experiment"]
        recal = config["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(experiment["rounds"], 1)
        self.assertEqual(experiment["policy_updates_per_round"], 50)
        self.assertEqual(experiment["reward_lifetime_updates"], 50)
        self.assertEqual(
            experiment["control_wandb_run"], pilot.LONG_REWARD_CONTROL_RUN
        )
        self.assertEqual(recal["max_reward_rounds"], 1)
        self.assertTrue(recal["adaptive_tempering"]["enabled"])
        self.assertEqual(config["logger"]["wandb"]["id"], "h4e50r01")

        for path, value in (
            (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
            (("dgpo", "adaptive_omnifold", "trigger", "max_reward_age_epochs"), 2),
            (("dgpo", "adaptive_omnifold", "recalibration", "reset_optimizer_state_on_install"), False),
            (("experiment", "reward_lifetime_updates"), 40),
            (("logger", "wandb", "id"), "h4e15r01"),
            (
                ("nersc", "ray", "results_dir"),
                read_overlay_yaml(pilot.ESS15_DEFAULT)["nersc"]["ray"][
                    "results_dir"
                ],
            ),
        ):
            bad = copy.deepcopy(config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                pilot.assert_contract(bad)

    def test_eight_member_overlay_changes_only_crossfit_repeats(self):
        config = read_overlay_yaml(pilot.LONG_REWARD_8_DEFAULT)
        pilot.apply_epsilon_result(config, self.result)
        pilot.assert_contract(config)
        experiment = config["experiment"]
        recal = config["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(experiment["rounds"], 1)
        self.assertEqual(experiment["policy_updates_per_round"], 50)
        self.assertEqual(experiment["ensemble_seeds"], 4)
        self.assertEqual(
            experiment["control_wandb_run"], pilot.LONG_REWARD_8_CONTROL_RUN
        )
        self.assertEqual(recal["crossfit_repeats"], 4)
        self.assertEqual(recal["crossfit_folds"], 2)
        self.assertEqual(recal["max_reward_rounds"], 1)
        self.assertEqual(config["logger"]["wandb"]["id"], "h4e50m8")

        for path, value in (
            (("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"), 3),
            (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
            (("dgpo", "adaptive_omnifold", "recalibration", "adaptive_tempering", "target_ess_fraction"), 0.20),
            (("experiment", "ensemble_seeds"), 3),
            (("logger", "wandb", "id"), "h4e50r01"),
        ):
            bad = copy.deepcopy(config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                pilot.assert_contract(bad)

    def test_six_member_overlay_changes_only_crossfit_repeats(self):
        config = read_overlay_yaml(pilot.LONG_REWARD_6_DEFAULT)
        pilot.apply_epsilon_result(config, self.result)
        pilot.assert_contract(config)
        experiment = config["experiment"]
        recal = config["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(experiment["rounds"], 1)
        self.assertEqual(experiment["policy_updates_per_round"], 50)
        self.assertEqual(experiment["ensemble_seeds"], 3)
        self.assertEqual(
            experiment["control_wandb_run"], pilot.LONG_REWARD_6_CONTROL_RUN
        )
        self.assertEqual(recal["crossfit_repeats"], 3)
        self.assertEqual(recal["crossfit_folds"], 2)
        self.assertEqual(recal["max_reward_rounds"], 1)
        self.assertEqual(config["logger"]["wandb"]["id"], "h4e50m6")

        for path, value in (
            (("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"), 4),
            (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
            (("dgpo", "adaptive_omnifold", "recalibration", "adaptive_tempering", "target_ess_fraction"), 0.20),
            (("experiment", "ensemble_seeds"), 4),
            (("logger", "wandb", "id"), "h4e50r01"),
        ):
            bad = copy.deepcopy(config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                pilot.assert_contract(bad)

    def test_six_member_refit20_preserves_policy_adam(self):
        config = read_overlay_yaml(pilot.REFIT20_6_DEFAULT)
        pilot.apply_epsilon_result(config, self.result)
        pilot.assert_contract(config)
        experiment = config["experiment"]
        adaptive = config["dgpo"]["adaptive_omnifold"]
        recal = adaptive["recalibration"]
        self.assertEqual(experiment["rounds"], 3)
        self.assertEqual(experiment["policy_updates_per_round"], 20)
        self.assertEqual(experiment["reward_refresh_steps"], [0, 20, 40])
        self.assertEqual(experiment["final_reward_updates"], 10)
        self.assertEqual(
            experiment["control_wandb_run"], pilot.REFIT20_6_CONTROL_RUN
        )
        self.assertEqual(adaptive["staleness_every_n_epochs"], 1)
        self.assertEqual(adaptive["trigger"]["max_reward_age_epochs"], 2)
        self.assertEqual(recal["crossfit_repeats"], 3)
        self.assertEqual(recal["crossfit_folds"], 2)
        self.assertEqual(recal["max_reward_rounds"], 3)
        self.assertFalse(recal["reset_optimizer_state_on_install"])
        self.assertFalse(recal["bootstrap_on_start"])
        self.assertEqual(config["dgpo"]["checkpoint_load_mode"], "resume")
        self.assertTrue(config["dgpo"]["pinned_classifier_restart"])
        self.assertEqual(
            Path(config["options"]["Training"]["model_checkpoint_load_path"]),
            pilot.H4E50M6_BOOTSTRAP,
        )
        self.assertEqual(config["logger"]["wandb"]["id"], "h4e20m6")

        resolved = resolve_adaptive_config(config["dgpo"])
        state = AdaptiveOmniFoldState(reward_round_id=1, installed_at_epoch=-1)
        self.assertFalse(
            scheduled_raw_refit_due(state, cfg=resolved, epoch=0)[0]
        )
        self.assertTrue(
            scheduled_raw_refit_due(state, cfg=resolved, epoch=1)[0]
        )
        state.last_recalibration_epoch = 1
        self.assertFalse(
            scheduled_raw_refit_due(state, cfg=resolved, epoch=2)[0]
        )
        self.assertTrue(
            scheduled_raw_refit_due(state, cfg=resolved, epoch=3)[0]
        )

        for path, value in (
            (("dgpo", "adaptive_omnifold", "trigger", "max_reward_age_epochs"), 1),
            (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
            (("dgpo", "adaptive_omnifold", "recalibration", "reset_optimizer_state_on_install"), True),
            (("dgpo", "adaptive_omnifold", "recalibration", "bootstrap_on_start"), True),
            (("dgpo", "checkpoint_load_mode"), "weights_only"),
            (("dgpo", "pinned_classifier_restart"), False),
            (("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"), 2),
            (("experiment", "reward_refresh_steps"), [0, 10, 30]),
            (("logger", "wandb", "id"), "h4e50m6"),
        ):
            bad = copy.deepcopy(config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                pilot.assert_contract(bad)

    def test_inherited_six_member_checkpoint_contract(self):
        increments = [
            {
                "state": {"weight": torch.ones(1)},
                "base_digest": "frozen-body",
                "packing_spec": {"shape": [1]},
            }
            for _ in range(6)
        ]
        payload = {
            "state_dict": {"policy": torch.ones(1)},
            "dgpo_checkpoint_version": 1,
            "dgpo_optimizer_state_dict": {},
            "dgpo_ref_state_dict": {},
            "dgpo_round_ref_state_dict": {"policy": torch.ones(1)},
            "dgpo_round_ref_sha256": "paired",
            "dgpo_omnifold_reward_metadata": {},
            "dgpo_omnifold_reward_stack": {
                "reward": {
                    "warm_start_state": {
                        "outer_partition": None,
                        "protocol": {
                            "scheme": "condition_sha256_v1",
                            "folds": 2,
                            "repeats": 3,
                        },
                        "models": [],
                    },
                    "increments": increments,
                    "increment_iterations": [1] * 6,
                    "increment_coefficients": [1.0 / 6.0] * 6,
                }
            },
            "dgpo_adaptive_omnifold_state": {
                "reward_round_id": 1,
                "raw_monitor_state": {
                    "state": {"weight": torch.ones(1)},
                    "protocol": {"seed": 1},
                },
            },
            "global_step": 0,
            "epoch": -1,
            "dgpo_next_epoch": 0,
            "dgpo_reward_round_id": 1,
        }
        self.assertTrue(pilot.assert_inherited_six_member_checkpoint(payload))
        no_monitor = copy.deepcopy(payload)
        no_monitor["dgpo_adaptive_omnifold_state"]["raw_monitor_state"] = {}
        self.assertFalse(
            pilot.assert_inherited_six_member_checkpoint(no_monitor)
        )
        partial_monitor = copy.deepcopy(payload)
        partial_monitor["dgpo_adaptive_omnifold_state"]["raw_monitor_state"] = {
            "protocol": {"seed": 1}
        }
        with self.assertRaisesRegex(ValueError, "partial raw monitor"):
            pilot.assert_inherited_six_member_checkpoint(partial_monitor)
        bad = copy.deepcopy(payload)
        bad["global_step"] = 50
        with self.assertRaises(ValueError):
            pilot.assert_inherited_six_member_checkpoint(bad)
        bad = copy.deepcopy(payload)
        bad["dgpo_omnifold_reward_stack"]["reward"]["warm_start_state"][
            "outer_partition"
        ] = {"schema": "condition-hash-80-20-v1", "seed": 42}
        with self.assertRaisesRegex(ValueError, "outer single-pool partition"):
            pilot.assert_inherited_six_member_checkpoint(bad)
        bad = copy.deepcopy(payload)
        bad["dgpo_omnifold_reward_stack"]["reward"]["increments"].pop()
        with self.assertRaises(ValueError):
            pilot.assert_inherited_six_member_checkpoint(bad)

    def test_contract_rejects_classifier_memory_and_sixth_round(self):
        for path, value in (
            (("dgpo", "adaptive_omnifold", "recalibration", "warm_start_iterations"), [1]),
            (("dgpo", "adaptive_omnifold", "trigger", "warm_start_classifier"), True),
            (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 6),
            (("dgpo", "reference_trust", "adaptive_boundary", "enabled"), True),
            (("dgpo", "fail_on_skipped_optimizer_step"), False),
            (("dgpo", "adaptive_omnifold", "recalibration", "fit", "topology_warmup_learning_rate"), 0.001),
            (("dgpo", "adaptive_omnifold", "recalibration", "adaptive_tempering", "enabled"), True),
        ):
            bad = copy.deepcopy(self.config)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                pilot.assert_contract(bad)

    def test_round_budget_counts_bootstrap_and_stops_at_five(self):
        resolved = resolve_adaptive_config(self.config["dgpo"])
        self.assertEqual(resolved.max_reward_rounds, 5)
        self.assertTrue(resolved.scheduled_refit_fail_closed)
        state = AdaptiveOmniFoldState(reward_round_id=4)
        self.assertFalse(
            reward_round_budget_exhausted(state, max_reward_rounds=5)
        )
        state.reward_round_id = 5
        self.assertTrue(
            reward_round_budget_exhausted(state, max_reward_rounds=5)
        )
        self.assertFalse(
            reward_round_budget_exhausted(state, max_reward_rounds=None)
        )

    def test_epsilon_report_must_have_an_eligible_read_only_selection(self):
        base = {
            "schema": self.config["experiment"]["expected_epsilon_schema"],
            "policy_global_step": 1110, "classifier_fits": 0, "policy_updates": 0,
            "selected_epsilon_rms": None, "eligible_epsilons_rms": [],
            "temperature": 2.0,
            "sweep": [
                {
                    "epsilon_rms": 3e-7,
                    "decision": {
                        "eligible": False, "pass_judge": True,
                        "pass_physics_jsd": True,
                        "judge_auc_gap_delta": -0.001,
                        "physics_mean_jsd_delta": -0.002,
                        "response_mean_abs_bin_offset_delta": 0.003,
                    },
                    "plus": {"anchor_gradient_cosine": 0.999},
                },
                {
                    "epsilon_rms": 1e-6,
                    "decision": {
                        "eligible": False, "pass_judge": True,
                        "pass_physics_jsd": True,
                        "judge_auc_gap_delta": -0.004,
                        "physics_mean_jsd_delta": -0.001,
                        "response_mean_abs_bin_offset_delta": 0.012,
                    },
                    "plus": {"anchor_gradient_cosine": 0.993},
                },
                {
                    "epsilon_rms": 3e-6,
                    "decision": {
                        "eligible": False, "pass_judge": True,
                        "pass_physics_jsd": False,
                        "judge_auc_gap_delta": -0.002,
                        "physics_mean_jsd_delta": 0.004,
                        "response_mean_abs_bin_offset_delta": 0.031,
                    },
                    "plus": {"anchor_gradient_cosine": 0.96},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(base))
            result = pilot.read_epsilon_result(path, self.config["experiment"])
            self.assertEqual(result["tempering"], 0.5)
            self.assertEqual(result["epsilon_rms"], 1e-6)
            self.assertFalse(result["strict_joint_eligible"])
            bad = copy.deepcopy(base)
            bad["sweep"][0]["decision"]["pass_judge"] = False
            bad["sweep"][1]["plus"]["anchor_gradient_cosine"] = 0.98
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                pilot.read_epsilon_result(path, self.config["experiment"])


if __name__ == "__main__":
    unittest.main()
