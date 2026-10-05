"""Source/config guards for the isolated real-case step1920 diagnostic."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.diagnose_tau_reward_transfer import (
    RELATIVE_STEPS, TRAIN_DIRECTORY, VALIDATION_DIRECTORY, WAND_NAME,
    command_for_runtime, configure, read_mapping, source_metadata,
    validate_settings, validate_source_runtime,
)


ROOT = Path(__file__).resolve().parents[1]


def saved_state():
    return dict(
        global_step=1920, epoch=191, dgpo_next_epoch=192, dgpo_epoch_step=0,
        state_dict={"model.original.weight": 1}, dgpo_ref_state_dict={"model.original.weight": 1},
        dgpo_optimizer_state_dict={
            "optimizer": {"state": {0: {"exp_avg": 1, "exp_avg_sq": 2}},
                "param_groups": [{"group_name": "body", "lr": 4.820682456e-5,
                                  "weight_decay": 0.001, "params": [0]}]},
            "scheduler": {"last_epoch": 1920, "base_lrs": [5e-5]},
            "lr_schedule": {"kind": "cosine", "total_steps": 15000, "min_lr_ratio": 0.1}},
        dgpo_omnifold_reward_stack={
            "kind": "conditional_tau_bound30", "schema_version": 1,
            "policy_conditioning_contract": "saved_tau_denominator_label0_v1",
            "round_id": 20, "denominator_step": 1880, "last_refit_epoch": 187,
            "condition_mean": [0], "condition_scale": [1], "normalization_file": "/pinned/norm.yaml",
            "source_checkpoint": "/pinned/raw1110.ckpt",
            "head": {"head_kind": "film", "head_depth": 3, "condition_width": 256,
                "condition_hidden": 256, "relative_dim": 6, "ratio_bound": 30,
                "cross_attention": True, "state_dict": {"head.weight": 1}}})


def source_runtime():
    return {
        "reward_config": {"type": "conditional_tau", "conditional_tau": {"bundle_file": "/pinned/initial.pt"}},
        "platform": {"number_of_workers": 16, "use_gpu": True, "batch_size": 512,
            "resources_per_worker": {"GPU": 1, "CPU": 15},
            "data_parquet_dir": TRAIN_DIRECTORY, "data_parquet_val_dir": VALIDATION_DIRECTORY},
        "network": {"Body": {"PET": {"visible_angular_fourier": {"fourier_enabled": False}}},
            "VisibleConditioning": {"diffusion_fourier_enabled": False, "width": 64},
            "TruthGeneration": {"num_layers": 3, "hidden_dim": 256}},
        "options": {"Training": {"epochs": 1500, "total_epochs": 1500,
            "EMA": {"enable": False, "use_for_generation": False},
            "Components": {"TruthGeneration": {"learning_rate": 5e-5}}}},
        "dgpo": {"K": 8, "num_ddim_steps": 20, "num_train_timesteps": 8,
            "policy_eval_t_min": 0, "policy_eval_t_max": 0.7, "steps_per_epoch": 10,
            "advantage_estimator": "leave_one_out_unscaled", "beta": 1,
            "adaptive_omnifold": {"enabled": False}, "checkpoint_transfer": {"enabled": True},
            "reference_trust": {"enabled": True, "coefficient": 1, "objective": "velocity_mse"},
            "conditioning_learning_rates": {"visible_conditioning": 1e-4},
            "lr_schedule": {"type": "cosine", "total_epochs": 1500, "resume_use_config": False},
            "gradient_conflict": {"enabled": True}, "tarp": {"enabled": True},
            "tau_ratio": {"workers": 16, "production_cross_attention": True,
                "policy_conditioning_contract": "saved_tau_denominator_label0_v1",
                "validation_panel": "/pinned/val.npz", "train_panel": "/pinned/train.npz",
                "generation_batch_size": 1024, "validation_candidates": 8,
                "refit_every_epochs": 5, "fit": {"batch_size": 1024, "epochs": 250, "min_steps": 1000, "lr": 0.0002}}},
        "logger": {"local": {}, "wandb": {"project": "nu2flow-RL", "entity": "ytchou97-university-of-washington",
            "id": "6d8bb6c2", "run_name": "previous", "tags": ["DGPO"]}},
        "nersc": {"ray": {}, "execution": {}},
        "experiment": {"actor_updates": 0, "classifier_only": True},
    }


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.settings = read_mapping(ROOT / "config/tau_reward_transfer_1920.yaml")

    def test_declared_source(self):
        validate_settings(self.settings)
        self.assertEqual(self.settings["relative_steps"], RELATIVE_STEPS)
        self.assertTrue(self.settings["checkpoint"].endswith("dgpo-epoch=191-next_ep=192-step=1920.ckpt"))
        self.assertNotIn("last.ckpt", self.settings["checkpoint"])
        self.assertEqual(self.settings["wandb_name"], WAND_NAME)
        self.assertLess(len(WAND_NAME), 96)

    def test_full_source_clocks_and_moments(self):
        meta = source_metadata(saved_state())
        self.assertEqual(meta["inherited_denominator_step"], 1880)
        self.assertEqual(meta["inherited_reward_round"], 20)
        self.assertEqual(meta["scheduler_last_epoch"], 1920)
        self.assertFalse(meta["rng_data_iterator_restored"])
        self.assertEqual(meta["optimizer_groups"][0]["lr"], 4.820682456e-5)

    def test_source_rejects_other_checkpoint_or_head(self):
        changes = [("global_step", 1780), ("epoch", 192), ("dgpo_next_epoch", 0), ("dgpo_epoch_step", 3)]
        for key, value in changes:
            with self.subTest(key=key):
                state = saved_state(); state[key] = value
                with self.assertRaises(ValueError): source_metadata(state)
        for key, value in (("cross_attention", False), ("head_depth", 6), ("ratio_bound", 15)):
            with self.subTest(key=key):
                state = saved_state(); state["dgpo_omnifold_reward_stack"]["head"][key] = value
                with self.assertRaises(ValueError): source_metadata(state)

    def test_source_rejects_changed_reference_or_unpopulated_optimizer(self):
        state = saved_state()
        state["dgpo_ref_state_dict"]["model.TruthGeneration.visible_conditioning.token_readout.output.weight"] = 1
        with self.assertRaises(ValueError): source_metadata(state)
        state = saved_state(); state["dgpo_optimizer_state_dict"]["optimizer"]["state"] = {}
        with self.assertRaises(ValueError): source_metadata(state)
        state = saved_state(); state["dgpo_optimizer_state_dict"]["scheduler"]["last_epoch"] = 0
        with self.assertRaises(ValueError): source_metadata(state)
        state = saved_state(); state["dgpo_omnifold_reward_stack"]["policy_conditioning_contract"] = "wrong"
        with self.assertRaises(ValueError): source_metadata(state)

    def test_preserves_native_structure_objective_and_schedule(self):
        original = source_runtime()
        before = copy.deepcopy(original)
        out = configure(original, self.settings, source_metadata(saved_state()),
                        pinned=Path("/probe/source.ckpt"), output=Path("/probe"))
        self.assertEqual(original, before)
        for key in ("network", "platform", "reward_config"):
            self.assertEqual(out[key], before[key])
        for key in ("Components", "EMA", "epochs", "total_epochs"):
            self.assertEqual(out["options"]["Training"][key], before["options"]["Training"][key])
        for key in ("lr_schedule", "conditioning_learning_rates", "reference_trust", "K", "num_ddim_steps", "num_train_timesteps", "beta", "advantage_estimator"):
            self.assertEqual(out["dgpo"][key], before["dgpo"][key])
        self.assertEqual(out["dgpo"]["tau_ratio"]["fit"], before["dgpo"]["tau_ratio"]["fit"])
        self.assertEqual(out["dgpo"]["gradient_transfer_trace"]["update_end_steps"], [1921, 1925, 1930, 1940, 1955, 1970])
        self.assertEqual(out["dgpo"]["gradient_transfer_trace"]["counterfactual_trust_coefficients"], [1.0])
        self.assertFalse(out["dgpo"]["checkpoint_transfer"]["enabled"])
        self.assertFalse(out["dgpo"]["tau_ratio"]["reward_cij_probe"])
        self.assertTrue(out["dgpo"]["tau_ratio"]["transfer_probe"]["enabled"])
        self.assertTrue(out["dgpo"]["tau_ratio"]["transfer_probe"]["gradient_trace_enabled"])
        self.assertEqual(out["dgpo"]["tau_ratio"]["transfer_probe"]["expected_source_reward_round"], 20)
        self.assertEqual(out["options"]["Training"]["model_checkpoint_load_path"], "/probe/source.ckpt")
        self.assertFalse(out["nersc"]["submit"])
        self.assertNotIn("actor_updates", out["experiment"])

    def test_rejects_changed_real_setup(self):
        mutate = [
            lambda c: c["platform"].update(number_of_workers=1),
            lambda c: c["platform"].update(data_parquet_dir="/raw/train"),
            lambda c: c["dgpo"]["reference_trust"].update(coefficient=0),
            lambda c: c["dgpo"].update(K=16),
            lambda c: c["dgpo"]["lr_schedule"].update(resume_use_config=True),
            lambda c: c["options"]["Training"]["EMA"].update(use_for_generation=True),
            lambda c: c["network"]["VisibleConditioning"].update(diffusion_fourier_enabled=True),
            lambda c: c["dgpo"]["tau_ratio"]["fit"].update(min_steps=0),
        ]
        for index, change in enumerate(mutate):
            with self.subTest(index=index):
                cfg = source_runtime(); change(cfg)
                with self.assertRaises(ValueError): validate_source_runtime(cfg)

    def test_new_run_logging_and_unambiguous_aliases(self):
        cfg = source_runtime(); cfg["wandb"] = {"project": "nu2flow-RL", "id": "old", "name": "old"}
        out = configure(cfg, self.settings, source_metadata(saved_state()), pinned=Path("/p/source.ckpt"), output=Path("/p"))
        for wb in (out["logger"]["wandb"], out["wandb"]):
            self.assertIsNone(wb["id"])
            self.assertEqual(wb["resume"], "never")
            self.assertTrue(wb["fresh_run"])
            self.assertEqual(wb["run_name"], WAND_NAME)
            self.assertEqual(wb["name"], WAND_NAME)

    def test_absolute_stop_command_no_scheduler_reset(self):
        command = command_for_runtime(Path("/p/runtime.yaml"), Path("/p"))
        self.assertEqual(command[command.index("--max-steps") + 1], "1970")
        self.assertEqual(command[command.index("--ray-dir") + 1], "/p/ray_results")

    def test_prepare_only_never_calls_subprocess(self):
        # Exercise actual preparation with synthetic full source, including the
        # production manifest validator interface, without importing Torch/Ray.
        import sys
        from types import ModuleType
        from scripts import diagnose_tau_reward_transfer as launch
        torch = ModuleType("torch"); torch.load = lambda *a, **k: saved_state()
        filtered = ModuleType("evenet.dataset.filtered_data")
        filtered.validate_filtered_dataset = lambda p: {"rows": 416701 if p == TRAIN_DIRECTORY else 119002}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.ckpt"; checkpoint.touch()
            runtime = root / "original.yaml"
            import yaml
            runtime.write_text(yaml.safe_dump(source_runtime()))
            settings = dict(self.settings, checkpoint=str(checkpoint), source_runtime=str(runtime), output_root=str(root / "outputs"))
            config = root / "settings.yaml"; config.write_text(yaml.safe_dump(settings))
            with patch.dict(sys.modules, {"torch": torch, "evenet.dataset.filtered_data": filtered}), \
                    patch.object(sys, "argv", ["diagnose_tau_reward_transfer.py", str(config), "--prepare-only"]), \
                    patch.object(launch.subprocess, "run") as run, patch("builtins.print"):
                launch.main()
                launch.main()
                run.assert_not_called()
            folders = list((root / "outputs").iterdir())
            self.assertEqual(len(folders), 2)
            self.assertNotEqual(folders[0], folders[1])
            self.assertEqual(read_mapping(runtime), source_runtime())
            for folder in folders:
                self.assertTrue((folder / "source.ckpt").is_file())
                self.assertTrue((folder / "source_runtime.yaml").is_file())
                self.assertEqual(read_mapping(folder / "runtime.yaml")["dgpo"]["tau_ratio"]["transfer_probe"]["source_step"], 1920)


if __name__ == "__main__":
    unittest.main()
