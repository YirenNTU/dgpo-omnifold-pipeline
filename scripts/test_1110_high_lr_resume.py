"""High-LR continuation contracts; no remote access or training launch."""
import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import test_227e4975_fixed_reward_resume as base
from RL.DGPO_neutrino import dgpo_trainer
from RL.DGPO_neutrino.model_utils import (
    parse_dgpo_resume_from_checkpoint, select_dgpo_training_state,
)


class HighLearningRateResumeTests(unittest.TestCase):
    def setUp(self):
        self.old = base.config("dgpo_1110_step0_fixed_reward_lr5e5.yaml")
        self.new = base.config("dgpo_1110_step0_fixed_reward_lr5e5_resume.yaml")

    def test_own_latest_checkpoint_not_upstream_step_zero(self):
        old_training = self.old["options"]["Training"]
        training = self.new["options"]["Training"]
        expected = old_training["model_checkpoint_save_path"] + "/last.ckpt"
        self.assertEqual(training["model_checkpoint_load_path"], expected)
        self.assertNotEqual(expected, old_training["model_checkpoint_load_path"])
        self.assertEqual(self.new["nersc"]["reproducibility"]["source_checkpoint"], expected)
        self.assertEqual(self.new["dgpo"]["checkpoint_load_mode"], "resume")
        self.assertTrue(self.new["dgpo"]["auto_resume_from_last"])
        self.assertFalse(self.new["dgpo"]["pinned_classifier_restart"])
        self.assertFalse(self.new["dgpo"]["best_source_start_new_experiment"])
        self.assertIsNone(self.new["dgpo"]["auto_resume_best_source_checkpoint_dir"])
        self.assertIsNone(self.new["experiment"]["source_policy_step"])
        self.assertEqual(self.new["experiment"]["source_run"], "h4s0lr5e5")

    def test_only_resume_controls_change(self):
        expected = copy.deepcopy(self.old)
        expected["options"]["Training"]["model_checkpoint_load_path"] = (
            self.old["options"]["Training"]["model_checkpoint_save_path"] + "/last.ckpt"
        )
        expected["dgpo"]["auto_resume_from_last"] = True
        expected["dgpo"]["lr_schedule"]["resume_use_config"] = False
        expected["logger"]["wandb"].update({
            "id": None, "resume": "never", "fresh_run": True,
            "run_name": "Can longer training absorb fixed reward? | H4 FiLM | 5e-5 cosine | continuation",
        })
        for key in self.new:
            if key not in {"nersc", "experiment"}:
                self.assertEqual(self.new[key], expected[key], key)
        self.assertEqual(self.new["nersc"]["ray"], self.old["nersc"]["ray"])
        self.assertIn("config/dgpo_1110_step0_fixed_reward_lr5e5_resume.yaml",
                      self.new["nersc"]["execution"]["command"])
        self.assertFalse(self.new["dgpo"]["lr_schedule"]["resume_use_config"])

    # Apply the existing full 15000-step no-audit/refit contract to this config.
    test_no_scheduled_audit_or_refit_at_any_training_step = (
        base.FixedRewardResumeTests.test_no_scheduled_audit_or_refit_at_any_training_step
    )

    def test_saved_policy_clock_and_installed_reward_are_not_reset(self):
        checkpoint = {
            "dgpo_checkpoint_version": 1, "epoch": 89, "dgpo_next_epoch": 90,
            "global_step": 900, "dgpo_epoch_step": 0,
            "dgpo_optimizer_state_dict": {"sentinel": "saved optimizer"},
            "dgpo_omnifold_reward_stack": {"sentinel": "fixed reward"},
            "dgpo_round_ref_state_dict": {"sentinel": "matching reference"},
        }
        selected = select_dgpo_training_state(
            checkpoint, load_mode=self.new["dgpo"]["checkpoint_load_mode"])
        self.assertIs(selected, checkpoint)
        self.assertEqual(parse_dgpo_resume_from_checkpoint(selected), (90, 900))

    def test_real_wandb_init_creates_new_run_without_changing_training_resume(self):
        w = self.new["logger"]["wandb"]
        wb = mock.Mock()
        wb.run.step = 0
        with mock.patch.dict("sys.modules", {"wandb": wb}), \
                mock.patch.dict("os.environ", {"WANDB_DISABLED": "false", "WANDB_RUN_ID": "h4s0lr5e5"}), \
                mock.patch("uuid.uuid4", return_value=SimpleNamespace(hex="1234abcd5678abcd")), \
                mock.patch.object(dgpo_trainer, "_dgpo_wandb_yaml_section", return_value=(w, "logger.wandb")), \
                mock.patch.object(dgpo_trainer, "global_config", SimpleNamespace(to_logger=lambda: self.new)), \
                mock.patch.object(dgpo_trainer, "_wandb_define_axes"), \
                mock.patch.object(dgpo_trainer, "_dgpo_wandb_publish_metric_docs"):
            self.assertTrue(dgpo_trainer._start_wandb_run())
        init = wb.init.call_args.kwargs
        self.assertEqual(init["id"], "1234abcd")
        self.assertEqual(init["resume"], "never")
        self.assertEqual(init["name"], w["run_name"])
        self.assertLessEqual(len(init["name"]), 96)
        self.assertEqual(init["group"], self.old["logger"]["wandb"]["group"])
        self.assertEqual(self.new["dgpo"]["checkpoint_load_mode"], "resume")


if __name__ == "__main__":
    unittest.main()
