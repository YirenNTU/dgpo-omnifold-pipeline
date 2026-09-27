"""Step-0 replay contracts; no remote access or training launch."""
import copy
from types import SimpleNamespace
from unittest import mock

import test_227e4975_fixed_reward_resume as continuation
from RL.DGPO_neutrino import dgpo_trainer
from RL.DGPO_neutrino.model_utils import parse_dgpo_resume_from_checkpoint, select_dgpo_training_state
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    AdaptiveOmniFoldState, resolve_adaptive_config, step_zero_raw_audit_enabled,
)


class StepZeroFixedRewardTests(continuation.FixedRewardResumeTests):
    def setUp(self):
        super().setUp()
        self.new = continuation.config("dgpo_1110_step0_fixed_reward_lr5e5.yaml")

    def test_full_resume_same_output_and_wandb(self):
        # Override continuation semantics: exact step-0 source, separate outputs.
        d = self.new["dgpo"]
        t = self.new["options"]["Training"]
        source = self.old["experiment"]["retained_classifier_checkpoint"]
        self.assertEqual(t["model_checkpoint_load_path"], source)
        self.assertEqual(self.new["nersc"]["reproducibility"]["source_checkpoint"], source)
        self.assertEqual(self.new["experiment"]["source_policy_step"], 0)
        self.assertTrue(source.endswith("/dgpo-epoch=-1-next_ep=0-step=0.ckpt"))
        self.assertNotEqual(t["model_checkpoint_save_path"], str(continuation.Path(source).parent))
        self.assertEqual(d["checkpoint_load_mode"], "resume")
        self.assertFalse(d["auto_resume_from_last"])
        self.assertFalse(d["pinned_classifier_restart"])
        self.assertFalse(d["best_source_start_new_experiment"])
        self.assertIsNone(d["auto_resume_best_source_checkpoint_dir"])
        self.assertIsNone(t["pretrain_model_load_path"])
        w = self.new["logger"]["wandb"]
        self.assertEqual(w["id"], "h4s0lr5e5")
        self.assertEqual(w["resume"], "never")
        self.assertFalse(w["fresh_run"])
        self.assertLessEqual(len(w["run_name"]), 96)
        # Exercise real wandb.init argument resolution without contacting W&B.
        wb = mock.Mock()
        wb.run.step = 0
        with mock.patch.dict("sys.modules", {"wandb": wb}), \
                mock.patch.dict("os.environ", {"WANDB_DISABLED": "false"}), \
                mock.patch.object(dgpo_trainer, "_dgpo_wandb_yaml_section", return_value=(w, "logger.wandb")), \
                mock.patch.object(dgpo_trainer, "global_config", SimpleNamespace(to_logger=lambda: self.new)), \
                mock.patch.object(dgpo_trainer, "_wandb_define_axes"), \
                mock.patch.object(dgpo_trainer, "_dgpo_wandb_publish_metric_docs"):
            self.assertTrue(dgpo_trainer._start_wandb_run())
        self.assertEqual(wb.init.call_args.kwargs["id"], w["id"])
        self.assertEqual(wb.init.call_args.kwargs["name"], w["run_name"])
        self.assertEqual(wb.init.call_args.kwargs["group"], w["group"])
        self.assertEqual(wb.init.call_args.kwargs["resume"], "never")

    def test_bootstrap_state_is_inherited_without_reset_or_initial_audit(self):
        cfg = resolve_adaptive_config(self.new["dgpo"])
        state = AdaptiveOmniFoldState(reward_round_id=1, raw_monitor_baseline_pending=True)
        checkpoint = {"dgpo_checkpoint_version": 1, "dgpo_next_epoch": 0,
                      "epoch": -1, "global_step": 0, "dgpo_epoch_step": 0,
                      "dgpo_adaptive_omnifold_state": state.to_dict(),
                      "dgpo_omnifold_reward_stack": {"sentinel": "installed"},
                      "dgpo_round_ref_state_dict": {"sentinel": "matching"},
                      "dgpo_optimizer_state_dict": {"sentinel": "saved"}}
        selected = select_dgpo_training_state(checkpoint, load_mode=self.new["dgpo"]["checkpoint_load_mode"])
        self.assertIs(selected, checkpoint)
        self.assertEqual(parse_dgpo_resume_from_checkpoint(selected), (0, 0))
        self.assertFalse(step_zero_raw_audit_enabled(cfg))
        before = copy.deepcopy(state.to_dict())
        self.assertTrue(dgpo_trainer._clear_unused_raw_monitor_baseline(state, cfg))
        before["raw_monitor_baseline_pending"] = False
        self.assertEqual(state.to_dict(), before)
        self.assertFalse(dgpo_trainer._clear_unused_raw_monitor_baseline(state, cfg))

    def test_required_startup_baseline_is_not_silently_removed(self):
        cfg = resolve_adaptive_config(self.old["dgpo"])
        state = AdaptiveOmniFoldState(raw_monitor_baseline_pending=True)
        self.assertFalse(dgpo_trainer._clear_unused_raw_monitor_baseline(state, cfg))
        self.assertTrue(state.raw_monitor_baseline_pending)
        # Preserve the pre-existing fixed-schedule skip behavior too.
        cfg = SimpleNamespace(fixed_schedule_skip_staleness_audit=True,
                              fixed_schedule_log_raw_audit=False)
        self.assertTrue(dgpo_trainer._clear_unused_raw_monitor_baseline(state, cfg))
