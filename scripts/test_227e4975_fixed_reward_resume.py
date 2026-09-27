"""Local resume/config checks; no Ray launch, classifier fit, or real policy run."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evenet_dgpo"))

from train_neutrino_backend import deep_update, read_overlay_yaml, read_yaml
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    AdaptiveOmniFoldState, resolve_adaptive_config, should_probe_training_boundary,
)


def config(name):
    return deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                       read_overlay_yaml(ROOT / "config" / name))


class FixedRewardResumeTests(unittest.TestCase):
    def setUp(self):
        self.old = config("dgpo_h4_kinematic_adaln_depth3.yaml")
        self.new = config("dgpo_227e4975_fixed_reward_resume.yaml")

    def test_full_resume_same_output_and_wandb(self):
        d = self.new["dgpo"]
        t = self.new["options"]["Training"]
        self.assertEqual(d["checkpoint_load_mode"], "resume")
        self.assertTrue(d["auto_resume_from_last"])
        self.assertFalse(d["pinned_classifier_restart"])
        self.assertFalse(d["best_source_start_new_experiment"])
        self.assertIsNone(d["auto_resume_best_source_checkpoint_dir"])
        self.assertIsNone(t["pretrain_model_load_path"])
        self.assertEqual(t["model_checkpoint_load_path"],
                         t["model_checkpoint_save_path"] + "/last.ckpt")
        w = self.new["logger"]["wandb"]
        self.assertEqual(w["id"], "227e4975")
        self.assertEqual(w["resume"], "must")
        self.assertFalse(w["fresh_run"])
        self.assertEqual(w["run_name"], self.old["logger"]["wandb"]["run_name"])

    def test_no_scheduled_audit_or_refit_at_any_training_step(self):
        d = self.new["dgpo"]
        a = resolve_adaptive_config(d)
        state = AdaptiveOmniFoldState()
        epochs = self.new["options"]["Training"]["epochs"]
        self.assertFalse(d["unbounded_training"])
        self.assertEqual(epochs, 1500)
        self.assertGreater(a.staleness_every_n_epochs, epochs)
        self.assertTrue(a.enabled)  # Preserve the round reference and payloads.
        self.assertTrue(a.log_only)
        self.assertFalse(a.raw_audit_enabled)
        self.assertFalse(a.baseline_probe_on_start)
        self.assertFalse(a.bootstrap_on_start)
        self.assertFalse(a.refit_once_on_resume)
        self.assertIsNone(a.max_reward_age_epochs)
        self.assertFalse(a.fixed_schedule_log_raw_audit)
        self.assertFalse(a.classifier_trust_enabled)
        self.assertFalse(a.raw_rollback_to_best_on_plateau)
        steps = d["steps_per_epoch"]
        for step in range(1, epochs * steps + 1):
            self.assertFalse(should_probe_training_boundary(
                state, cfg=a, epoch=(step - 1) // steps, global_step=step,
                epoch_end=step % steps == 0))

    def test_validation_loss_and_model_are_unchanged(self):
        allowed = {"checkpoint_load_mode", "auto_resume_from_last", "adaptive_omnifold", "lr_schedule"}
        for key, value in self.old["dgpo"].items():
            if key not in allowed:
                self.assertEqual(self.new["dgpo"][key], value, key)
        for key in ("platform", "network", "reward_config"):
            self.assertEqual(self.new[key], self.old[key], key)
        for key, value in self.old["options"]["Training"].items():
            if key not in {"model_checkpoint_load_path", "model_checkpoint_save_path",
                           "learning_rate", "learning_rate_body", "Components"}:
                self.assertEqual(self.new["options"]["Training"][key], value, key)
        changed_components = {"GlobalEmbedding", "PET", "TruthGeneration",
                              "InvisibleInputProjector", "GroupedSequentialEmbedding"}
        old_components = self.old["options"]["Training"]["Components"]
        new_components = self.new["options"]["Training"]["Components"]
        self.assertEqual(set(old_components), set(new_components))
        for name, values in old_components.items():
            for key, value in values.items():
                if name in changed_components and key == "learning_rate":
                    self.assertEqual(new_components[name][key], 5e-5)
                else:
                    self.assertEqual(new_components[name][key], value, (name, key))
        d = self.new["dgpo"]
        self.assertEqual(d["validation_every_n_epochs"], 10)
        self.assertEqual(d["validation_full_every_n_epochs"], 10)
        self.assertEqual(d["validation_K"], 16)
        self.assertEqual(d["validation_num_ddim_steps"], 20)
        self.assertEqual(d["validation_max_batches"], 4)
        self.assertEqual(d["lr_schedule"]["total_epochs"], 1500)
        self.assertEqual(d["reference_trust"]["coefficient"], 1)
        self.assertFalse(self.new["options"]["Training"]["EMA"]["enable"])
        self.assertEqual(self.new["platform"]["number_of_workers"], 16)

    def test_actual_policy_groups_use_requested_lrs_and_full_cosine(self):
        from types import SimpleNamespace
        from unittest import mock
        import torch
        from RL.DGPO_neutrino import dgpo_trainer

        class Config(dict):
            __getattr__ = dict.__getitem__

        # Include default optimizer_group assignments, as the real loader does.
        options = deep_update(read_yaml(ROOT / "config/evenet_defaults/options.yaml"),
                              self.new["options"])
        training = Config(options["Training"])
        self.assertEqual(training.learning_rate, 5e-5)
        self.assertEqual(training.learning_rate_body, 5e-5)
        model = torch.nn.Module()
        for name in ("GlobalEmbedding", "PET", "TruthGeneration",
                     "InvisibleInputProjector", "GroupedSequentialEmbedding"):
            setattr(model, name, torch.nn.Linear(1, 1))
        model.PET.angular_conditioning = torch.nn.Linear(1, 1)
        model.TruthGeneration.visible_conditioning = torch.nn.Linear(1, 1)
        with mock.patch.object(dgpo_trainer, "global_config", SimpleNamespace(
                options=SimpleNamespace(Training=training))), \
                mock.patch.object(dgpo_trainer, "_unwrap_core_evenet", return_value=model):
            optimizer = dgpo_trainer.build_optimizer(
                model, steps_per_epoch=self.new["dgpo"]["steps_per_epoch"], warmup_steps=1,
                is_rank0=False, lr_schedule=self.new["dgpo"]["lr_schedule"],
                conditioning_learning_rates=self.new["dgpo"]["conditioning_learning_rates"],
            )
        expected = {"body": 5e-5, "generation": 5e-5, "projector": 5e-5,
                    "angular_conditioning": 1e-4, "visible_conditioning": 1e-4}
        self.assertEqual({pg["group_name"]: pg["lr"] for pg in optimizer.param_groups}, expected)
        self.assertEqual(optimizer.cosine_state["decay_groups"], [True] * 5)
        self.assertEqual(optimizer.cosine_state["total_steps"], 15000)
        self.assertTrue(self.new["dgpo"]["lr_schedule"]["resume_use_config"])
        for index, pg in enumerate(optimizer.param_groups):
            self.assertAlmostEqual(expected[pg["group_name"]] * optimizer._cosine_factor(15000, index),
                                   expected[pg["group_name"]] * .1)

    def test_reward_representation_and_optimizer_install_rules_unchanged(self):
        old = self.old["dgpo"]["adaptive_omnifold"]["recalibration"]
        new = self.new["dgpo"]["adaptive_omnifold"]["recalibration"]
        for key, value in old.items():
            if key not in {"bootstrap_on_start", "refit_once_on_resume",
                           "scheduled_refit_fail_closed"}:
                self.assertEqual(new[key], value, key)
        self.assertEqual(self.new["experiment"]["cold_h4_audit_policy_steps"], [])


if __name__ == "__main__":
    unittest.main()
