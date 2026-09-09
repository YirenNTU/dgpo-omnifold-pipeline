from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from prepare_dgpo_sbatch_resume import check_10pct_protocol, prepare_resume
from train_neutrino_backend import REPO_ROOT, deep_update, read_yaml


class TestSbatchResume(unittest.TestCase):
    def setUp(self):
        self.base = REPO_ROOT / "config/train_diffusion_nersc.yaml"
        self.overlay = REPO_ROOT / "config/dgpo_omnifold_ztautau_10pct_resume_v26.yaml"
        self.original = deep_update(read_yaml(self.base), read_yaml(self.overlay))

    def test_resume_snapshot_and_preserved_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "old/checkpoints/epoch200.ckpt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"test fixture, not a real model")
            alias = source.parent / "last.ckpt"
            alias.symlink_to(source.name)
            output = root / "new_job"
            runtime = prepare_resume(self.base, self.overlay, alias, output)
            cfg = read_yaml(runtime)
            training = cfg["options"]["Training"]
            self.assertEqual(training["model_checkpoint_load_path"], str(source))
            self.assertEqual(training["model_checkpoint_save_path"], str(output / "checkpoints"))
            self.assertIsNone(training["pretrain_model_load_path"])
            self.assertEqual(cfg["dgpo"]["checkpoint_load_mode"], "resume")
            self.assertFalse(cfg["dgpo"]["auto_resume_from_last"])
            self.assertTrue(cfg["logger"]["wandb"]["fresh_run"])
            self.assertEqual(cfg["logger"]["wandb"]["resume"], "never")
            self.assertIsNone(cfg["logger"]["wandb"]["id"])
            self.assertEqual(cfg["dgpo"]["adaptive_omnifold"], self.original["dgpo"]["adaptive_omnifold"])
            for key in ("lr_schedule", "reference_trust", "validation_every_n_epochs", "validation_full_every_n_epochs"):
                self.assertEqual(cfg["dgpo"][key], self.original["dgpo"][key])
            self.assertEqual(training["weight_decay"], 0.001)
            self.assertIn(str(source.parent), cfg["dgpo"]["global_best_checkpoint_search_dirs"])
            old_source = Path(self.original["options"]["Training"]["model_checkpoint_load_path"]).parent
            self.assertIn(str(old_source), cfg["dgpo"]["global_best_checkpoint_search_dirs"])
            self.assertIn("event_info", cfg)
            self.assertIn("resonance", cfg)
            self.assertTrue(Path(cfg["options"]["default"]).is_absolute())
            before = runtime.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare_resume(self.base, self.overlay, alias, output)
            self.assertEqual(before, runtime.read_bytes())
            with self.assertRaises(ValueError):
                prepare_resume(self.base, self.overlay, alias, source.parent.parent)

    def test_missing_checkpoint_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "new"
            with self.assertRaises(FileNotFoundError):
                prepare_resume(self.base, self.overlay, Path(tmp) / "missing.ckpt", output)
            self.assertFalse(output.exists())

    def test_wrong_fraction_data_backbone_and_pool_fail(self):
        mutations = [
            ("nersc", "reproducibility", "dgpo_training_fraction", 0.01),
            ("nersc", "reproducibility", "diffusion_pretrain_fraction", 0.01),
            ("platform", "data_parquet_dir", "/wrong/1pct/train"),
            ("platform", "data_parquet_val_dir", "/full/validation"),
            ("reward_config", "omnifold", "backbone_checkpoint", "/1pct/last.ckpt"),
            ("dgpo", "adaptive_omnifold", "trigger", "probe_max_events", 40000),
            ("options", "Dataset", "dataset_limit", 0.1),
        ]
        check_10pct_protocol(self.original)
        for *keys, value in mutations:
            with self.subTest(keys=keys):
                cfg = copy.deepcopy(self.original)
                target = cfg
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaises(ValueError):
                    check_10pct_protocol(cfg)

    def test_interactive_overlay_resumes_latest_in_place(self):
        cfg = self.original
        training = cfg["options"]["Training"]
        save_dir = Path(training["model_checkpoint_save_path"])
        self.assertTrue(cfg["dgpo"]["auto_resume_from_last"])
        self.assertEqual(cfg["dgpo"]["checkpoint_load_mode"], "resume")
        self.assertEqual(Path(training["model_checkpoint_load_path"]), save_dir / "last.ckpt")
        self.assertEqual(Path(cfg["logger"]["local"]["save_dir"]), save_dir.parent / "logs")
        self.assertEqual(Path(cfg["nersc"]["ray"]["results_dir"]), save_dir.parent / "ray_results")
        self.assertTrue(cfg["logger"]["wandb"]["fresh_run"])
        self.assertEqual(cfg["logger"]["wandb"]["resume"], "never")
        adaptive = cfg["dgpo"]["adaptive_omnifold"]
        self.assertEqual(adaptive["trigger"]["best_scope"], "global")
        self.assertFalse(adaptive["recalibration"]["bootstrap_on_start"])
        self.assertFalse(adaptive["recalibration"]["refit_once_on_resume"])


if __name__ == "__main__":
    unittest.main()
