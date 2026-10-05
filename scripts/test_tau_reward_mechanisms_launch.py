"""Isolation and causal-comparison guards for the step1920 mechanism series."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import yaml

from scripts import diagnose_tau_reward_mechanisms as launch
from scripts.test_tau_reward_transfer_launch import saved_state, source_runtime


ROOT = Path(__file__).resolve().parents[1]


def recorded_native_manifest(ensemble_directory, source_checkpoint="/pinned/source1920.ckpt"):
    folder = Path(ensemble_directory) / "native_training"
    folder.mkdir(parents=True)
    ranks = []
    for rank in range(16):
        rows = 416701 // 16 + (rank < 416701 % 16)
        sizes = [512] * (rows // 512) + ([rows % 512] if rows % 512 else [])
        ranks.append({"rank": rank, "rows": rows, "batches": len(sizes), "batch_sizes": sizes,
            "partial_batch_indices": [index for index, size in enumerate(sizes) if size < 512],
            "local_shuffle_seed": 42 + rank, "replay_fingerprint": f"rank{rank}",
            "batch_fingerprints": [f"rank{rank}batch{index}" for index in range(len(sizes))], "error": None})
        (folder / f"rank-{rank:02d}.pt").touch()
    manifest = {"kind": "tau_native_training_batch_replay", "schema_version": 1, "complete": True,
        "source_policy_step": 1920, "world": 16, "expected_events": 416701, "global_rows": 416701,
        "source_checkpoint": source_checkpoint,
        "batch_size": 512, "base_shuffle_seed": 42, "actor_updates": 0,
        "all_native_tensor_fields_preserved": True, "ranks": ranks,
        "minimum_rank_batches": min(row["batches"] for row in ranks),
        "maximum_rank_batches": max(row["batches"] for row in ranks),
        "tail_recipe": "Preserve original partial tails and independently cycle each rank",
        "global_replay_fingerprint": "complete-native-replay"}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return manifest


class MechanismLaunchTests(unittest.TestCase):
    def setUp(self):
        self.settings = launch.read_mapping(ROOT / "config/tau_reward_mechanisms_1920.yaml")
        self.metadata = launch.source_metadata(saved_state())

    def configured(self, *, stage="diagnose", arm="inherited", updates=50, ensemble_directory=None):
        return launch.configure(source_runtime(), self.settings, self.metadata,
            pinned=Path("/series/diagnose/source.ckpt"), output=Path("/series/diagnose"),
            stage=stage, arm=arm, updates=updates, ensemble_directory=ensemble_directory)

    def test_source_and_predeclared_settings(self):
        launch.validate_settings(self.settings)
        self.assertEqual(self.settings["trajectory_updates"], [50, 200, 500])
        self.assertEqual(self.settings["condition_edges"], [-1.0, -0.5, 0.5, 1.0])
        self.assertEqual(self.settings["gradient_events_per_rank"], 512)
        self.assertEqual(self.settings["ensemble_aggregation"], "mean_bounded_log_ratio")
        self.assertNotIn("last.ckpt", self.settings["checkpoint"])

    def test_inherited_probe_preserves_full_native_state_contract(self):
        original = source_runtime()
        before = copy.deepcopy(original)
        cfg = launch.configure(original, self.settings, self.metadata,
            pinned=Path("/series/source.ckpt"), output=Path("/series"))
        self.assertEqual(original, before)
        for key in ("network", "platform", "reward_config"):
            self.assertEqual(cfg[key], before[key])
        for key in ("Components", "EMA", "epochs", "total_epochs"):
            self.assertEqual(cfg["options"]["Training"][key], before["options"]["Training"][key])
        for key in ("reference_trust", "conditioning_learning_rates", "lr_schedule", "K",
                    "num_ddim_steps", "num_train_timesteps", "beta", "advantage_estimator"):
            self.assertEqual(cfg["dgpo"][key], before["dgpo"][key])
        tau = cfg["dgpo"]["tau_ratio"]
        self.assertEqual(tau["fit"], before["dgpo"]["tau_ratio"]["fit"])
        self.assertEqual(tau["refit_every_epochs"], before["dgpo"]["tau_ratio"]["refit_every_epochs"])
        probe = tau["mechanism_probe"]
        self.assertTrue(probe["enabled"])
        self.assertEqual(probe["mode"], "diagnose")
        self.assertEqual(probe["updates"], 0)
        self.assertFalse(probe["reference_recenter"])
        self.assertFalse(probe["periodic_refits"])
        self.assertEqual(probe["expected_source_reward_round"], 20)
        self.assertEqual(probe["expected_source_denominator_step"], 1880)
        self.assertFalse(tau["transfer_probe"]["enabled"])
        self.assertFalse(cfg["dgpo"]["checkpoint_transfer"]["enabled"])
        self.assertEqual(cfg["dgpo"]["checkpoint_load_mode"], "resume")
        self.assertFalse(cfg["nersc"]["submit"])
        self.assertNotIn("actor_updates", cfg["experiment"])

    def test_all_native_stage_caps_and_endpoints(self):
        for stage, expected in (("diagnose", 1921), ("ensemble", 1920)):
            command = launch.command_for_runtime(Path("/series/runtime.yaml"), Path("/series"), stage=stage)
            self.assertEqual(command[command.index("--max-steps") + 1], str(expected))
            self.assertEqual(self.configured(stage=stage)["dgpo"]["tau_ratio"]["mechanism_probe"]["updates"], 0)
            self.assertEqual(self.configured(stage=stage)["dgpo"]["tau_ratio"]["mechanism_probe"]["evaluation_events"],
                             32768 if stage == "diagnose" else 119002)
        for updates in (50, 200, 500):
            with self.subTest(updates=updates):
                command = launch.command_for_runtime(Path("/series/runtime.yaml"), Path("/series"), stage="trajectory", updates=updates)
                self.assertEqual(command[command.index("--max-steps") + 1], str(1920 + updates))
                probe = self.configured(stage="trajectory", updates=updates)["dgpo"]["tau_ratio"]["mechanism_probe"]
                self.assertEqual(probe["relative_steps"][-1], updates)
                self.assertEqual(probe["audit_relative_steps"][-1], updates)
                self.assertEqual(probe["evaluation_events"], 119002)

    def test_fresh_arm_requires_saved_artifacts_and_common_judge_can_be_shared(self):
        for stage in ("diagnose", "trajectory"):
            for arm in ("member0", "ensemble"):
                with self.subTest(stage=stage, arm=arm):
                    with self.assertRaises(ValueError):
                        self.configured(stage=stage, arm=arm)
                    cfg = self.configured(stage=stage, arm=arm, ensemble_directory=Path("/series/fit/ensemble"))
                    self.assertEqual(cfg["dgpo"]["tau_ratio"]["mechanism_probe"]["reward_arm"], arm)
        cfg = self.configured(stage="trajectory", arm="inherited", ensemble_directory=Path("/series/fit/ensemble"))
        self.assertEqual(cfg["dgpo"]["tau_ratio"]["mechanism_probe"]["ensemble_directory"], "/series/fit/ensemble")

    def test_rejects_changed_population_reference_objective_or_policy_batch(self):
        mutations = [
            lambda c: c["platform"].update(number_of_workers=1),
            lambda c: c["platform"].update(batch_size=1024),
            lambda c: c["platform"].update(data_parquet_dir="/raw/train"),
            lambda c: c["dgpo"]["reference_trust"].update(coefficient=0),
            lambda c: c["dgpo"].update(K=16),
            lambda c: c["dgpo"].update(normalize_advantages=True),
            lambda c: c["dgpo"]["lr_schedule"].update(resume_use_config=True),
            lambda c: c["options"]["Training"]["EMA"].update(use_for_generation=True),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                cfg = source_runtime()
                mutate(cfg)
                with self.assertRaises(ValueError):
                    launch.validate_native_runtime(cfg)

    def test_readable_names_and_new_run_aliases(self):
        cfg = source_runtime()
        cfg["wandb"] = {"project": "nu2flow-RL", "id": "old", "name": "old"}
        out = launch.configure(cfg, self.settings, self.metadata,
            pinned=Path("/series/source.ckpt"), output=Path("/series"), stage="trajectory", arm="ensemble",
            updates=500, ensemble_directory=Path("/series/fit/ensemble"))
        name = launch.display_name("trajectory", arm="ensemble", updates=500)
        for wb in (out["logger"]["wandb"], out["wandb"]):
            self.assertIsNone(wb["id"])
            self.assertEqual(wb["resume"], "never")
            self.assertTrue(wb["fresh_run"])
            self.assertEqual(wb["name"], name)
            self.assertEqual(wb["run_name"], name)
        for stage in ("diagnose", "ensemble", "trajectory"):
            self.assertLess(len(launch.display_name(stage)), 96)

    def test_artifacts_must_belong_to_declared_series(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series = root / "series"
            ensemble = series / "fit" / "ensemble"
            ensemble.mkdir(parents=True)
            (ensemble / "ensemble.pt").touch()
            settings = dict(self.settings, series_root=str(series))
            with self.assertRaises(ValueError):
                launch.resolve_ensemble_directory(settings, ensemble, arm="member0", stage="trajectory")
            recorded_native_manifest(ensemble, source_checkpoint=self.settings["checkpoint"])
            self.assertEqual(launch.resolve_ensemble_directory(settings, ensemble, arm="member0", stage="trajectory"), ensemble.resolve())
            with self.assertRaises(ValueError):
                launch.resolve_ensemble_directory(settings, root, arm="ensemble", stage="diagnose")

    def test_native_replay_preflight_rejects_incomplete_capture_and_lost_tails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = recorded_native_manifest(root)
            accepted = launch.validate_native_training_manifest(root)
            self.assertEqual(accepted["global_rows"], 416701)
            with self.assertRaises(ValueError):
                launch.validate_native_training_manifest(root, source_checkpoint="/another/step1920.ckpt")
            path = root / "native_training" / "manifest.json"
            for mutation in (
                lambda m: m.update(complete=False),
                lambda m: m.update(global_rows=416700),
                lambda m: m.update(world=1),
                lambda m: m["ranks"][0].update(partial_batch_indices=[]),
            ):
                changed = copy.deepcopy(manifest)
                mutation(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    launch.validate_native_training_manifest(root)
            path.write_text(json.dumps(manifest))
            (root / "native_training" / "rank-15.pt").unlink()
            with self.assertRaises(ValueError):
                launch.validate_native_training_manifest(root)

    def test_unique_ensemble_selection_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = dict(self.settings, series_root=str(root))
            for index in (1, 2):
                invocation = root / f"ensemble-{index}"
                ensemble = invocation / "ensemble"
                ensemble.mkdir(parents=True)
                (ensemble / "ensemble.pt").touch()
                recorded_native_manifest(ensemble, source_checkpoint=self.settings["checkpoint"])
                reports = invocation / "mechanisms"
                reports.mkdir()
                (reports / "ensemble_diagnostic.json").write_text(json.dumps({"complete": True}))
                if index == 1:
                    self.assertEqual(launch.resolve_ensemble_directory(settings, Path("unique"),
                        arm="ensemble", stage="trajectory"), ensemble.resolve())
            with self.assertRaises(ValueError):
                launch.resolve_ensemble_directory(settings, Path("unique"), arm="ensemble", stage="trajectory")

    def test_ensemble_preflight_rejects_undertraining_and_changed_source(self):
        payload = dict(kind="tau_classifier_ensemble_diagnostic", schema_version=1, complete=True,
            source_policy_step=1920, fresh_denominator_step=1920, inherited_denominator_step=1880,
            single_control_member=0, aggregation="mean_training_time_bounded_log_ratio",
            training_population=416701, training_candidates=1,
            inherited_clocks={"round_id": 20, "denominator_step": 1880, "last_refit_epoch": 187},
            reward_contract={"normalization_file": "/pinned/norm.yaml", "source_checkpoint": "/pinned/raw1110.ckpt",
                "policy_conditioning_contract": launch.POLICY_CONTRACT},
            members=[{"seed": index, "fit": {"minimum_fit_steps_met": True, "total_steps": 1000}} for index in range(4)],
            judge={"seed": 4, "fit": {"minimum_fit_steps_met": True, "total_steps": 1000}})
        launch.validate_ensemble_metadata(payload, self.metadata)
        mutated = copy.deepcopy(payload)
        mutated["judge"]["fit"]["total_steps"] = 999
        with self.assertRaises(ValueError):
            launch.validate_ensemble_metadata(mutated, self.metadata)
        mutated = copy.deepcopy(payload)
        mutated["inherited_clocks"]["round_id"] = 21
        with self.assertRaises(ValueError):
            launch.validate_ensemble_metadata(mutated, self.metadata)
        mutated = copy.deepcopy(payload)
        mutated["reward_contract"]["normalization_file"] = "/different/norm.yaml"
        with self.assertRaises(ValueError):
            launch.validate_ensemble_metadata(mutated, self.metadata)

    def test_prepare_only_pins_new_folders_without_subprocess_or_ray(self):
        torch = ModuleType("torch")
        torch.load = lambda *a, **k: saved_state()
        filtered = ModuleType("evenet.dataset.filtered_data")
        filtered.validate_filtered_dataset = lambda p: {"rows": 416701 if p == launch.TRAIN_DIRECTORY else 119002}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.ckpt"
            checkpoint.touch()
            runtime = root / "source.yaml"
            runtime.write_text(yaml.safe_dump(source_runtime()))
            settings = dict(self.settings, checkpoint=str(checkpoint), source_runtime=str(runtime), series_root=str(root / "series"))
            config = root / "settings.yaml"
            config.write_text(yaml.safe_dump(settings))
            with patch.dict(sys.modules, {"torch": torch, "evenet.dataset.filtered_data": filtered}), \
                    patch.object(launch.subprocess, "run") as run, patch("builtins.print"):
                launch.main([str(config), "prepare"])
                launch.main([str(config), "trajectory", "--updates", "200", "--prepare-only"])
                run.assert_not_called()
            folders = sorted((root / "series").iterdir())
            self.assertEqual(len(folders), 2)
            self.assertNotEqual(folders[0], folders[1])
            self.assertEqual(launch.read_mapping(runtime), source_runtime())
            for folder in folders:
                self.assertTrue((folder / "source.ckpt").is_file())
                self.assertTrue((folder / "source_runtime.yaml").is_file())
                manifest = json.loads((folder / launch.MANIFEST).read_text())
                self.assertTrue(manifest["prepared_only"])
                cfg = launch.read_mapping(folder / "runtime.yaml")
                self.assertFalse(cfg["dgpo"]["tau_ratio"]["mechanism_probe"]["reference_recenter"])

    def test_summary_does_not_treat_preparation_as_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "diagnose-prepared"
            folder.mkdir()
            (folder / launch.MANIFEST).write_text(json.dumps({"stage": "diagnose", "reward_arm": "inherited",
                "relative_updates": 0, "prepared_only": True}))
            settings = dict(self.settings, series_root=str(root))
            report = launch.summarize(settings)
            self.assertEqual(len(report["invocations"]), 1)
            self.assertEqual(report["invocations"][0]["execution_evidence"], "no result artifact recorded")
            self.assertEqual(report["invocations"][0]["result_files"], [])

    def test_summary_exposes_partial_last_point_without_inventing_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "trajectory"
            (folder / "mechanisms").mkdir(parents=True)
            (folder / launch.MANIFEST).write_text(json.dumps({"stage": "trajectory", "reward_arm": "inherited",
                "relative_updates": 200, "prepared_only": False}))
            (folder / "mechanisms" / "transfer_report.json").write_text(json.dumps({
                "complete": False, "measurements": {"0": {"reward": {"delta_mean": 0}},
                    "50": {"reward": {"delta_mean": 0.2, "delta_lo95": 0.1, "delta_hi95": 0.3},
                           "cij": {"error": 0.4}}}}))
            result = launch.summarize(dict(self.settings, series_root=str(root)))["invocations"][0]
            self.assertEqual(result["last_relative_step"], 50)
            self.assertFalse(result["reports"]["transfer_report.json"]["complete"])
            self.assertEqual(result["reports"]["transfer_report.json"]["endpoint"]["reward"]["delta_mean"], 0.2)

    def test_cross_arm_comparison_pairs_identities_under_common_judge(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, arm in enumerate(("member0", "ensemble")):
                path = root / f"{arm}.npz"
                np.savez(path, source_ids=np.arange(4), weight=np.ones(4), truth_cij=np.ones((4, 9)),
                    generated_cij=np.ones((4, 9))*(2-index), judge_reward=np.ones((4, 8))*index)
                records.append({"stage": "trajectory", "reward_arm": arm, "output": str(root / arm),
                    "source_checkpoint": "/pinned/source1920.ckpt", "ensemble_directory": "/series/ensemble",
                    "endpoint_measurements": str(path), "validation_seed": 42, "candidates": 8,
                    "native_replay_valid": True, "native_replay_fingerprint": "shared-native-replay",
                    "last_relative_step": 50})
            compared = launch._paired_cross_arm(self.settings, records)
            self.assertEqual(len(compared), 1)
            self.assertEqual(compared[0]["status"], "paired")
            self.assertEqual(compared[0]["judge_reward"]["delta_mean"], 1.0)
            self.assertEqual(compared[0]["judge_reward"]["delta_lo95"], 1.0)
            self.assertLess(compared[0]["cij_error_changes"]["error"]["mean"], 0)
            records[1]["validation_seed"] = 99
            self.assertEqual(launch._paired_cross_arm(self.settings, records), [])
            records[1]["validation_seed"] = 42
            records[1]["native_replay_valid"] = False
            self.assertEqual(launch._paired_cross_arm(self.settings, records), [])
            records[1]["native_replay_valid"] = True
            np.savez(root / "ensemble.npz", source_ids=np.arange(4)+1, weight=np.ones(4), truth_cij=np.ones((4, 9)),
                generated_cij=np.ones((4, 9)), judge_reward=np.ones((4, 8)))
            with self.assertRaises(ValueError):
                launch._paired_cross_arm(self.settings, records)


class RepeatedDirectionLaunchTests(unittest.TestCase):
    def setUp(self):
        self.settings = launch.read_mapping(ROOT / "config/tau_reward_direction_repeats_1920.yaml")
        self.metadata = launch.source_metadata(saved_state())

    def test_repeat_mode_preserves_source_and_fixed_head_contract(self):
        launch.validate_settings(self.settings)
        original = source_runtime()
        cfg = launch.configure(original, self.settings, self.metadata, pinned=Path("/repeat/source.ckpt"),
                               output=Path("/repeat"), stage="repeat_diagnose")
        for key in ("network", "platform", "reward_config"):
            self.assertEqual(cfg[key], original[key])
        for key in ("reference_trust", "lr_schedule", "conditioning_learning_rates", "K", "num_train_timesteps"):
            self.assertEqual(cfg["dgpo"][key], original["dgpo"][key])
        probe = cfg["dgpo"]["tau_ratio"]["mechanism_probe"]
        self.assertEqual(probe["mode"], "repeat_diagnose")
        self.assertEqual(probe["draw_repeats"], 4)
        self.assertEqual(probe["gradient_repeats"], 2)
        self.assertEqual(probe["evaluation_seeds"], [202610041, 202610042])
        self.assertEqual(probe["direction_rms_fractions"], [0.01, 0.03, 0.1])
        self.assertEqual(probe["evaluation_events"], 32768)
        self.assertEqual(probe["audit_relative_steps"], [])
        self.assertEqual(probe["updates"], 0)
        self.assertEqual(probe["persistent_updates"], 0)
        self.assertFalse(probe["reference_recenter"])
        self.assertFalse(probe["periodic_refits"])
        self.assertEqual(probe["evaluator"], "installed_training_head")
        self.assertIsNone(probe["ensemble_directory"])
        self.assertNotIn("ensemble_fit_seeds", probe)
        self.assertEqual(cfg["experiment"]["protocol"], launch.REPEAT_PROTOCOL)
        self.assertIn("inherited frozen installed head", cfg["experiment"]["primary"])
        self.assertEqual(cfg["experiment"]["direction_gradient_replica"], 0)
        self.assertIn("not averaging", cfg["experiment"]["gradient_repeat_role"])
        self.assertIn("draw_index", cfg["experiment"]["native_mc_seed_recipe"])
        self.assertFalse(cfg["nersc"]["submit"])
        name = launch.display_name("repeat_diagnose")
        self.assertLess(len(name), 96)
        self.assertEqual(cfg["logger"]["wandb"]["name"], name)
        self.assertIsNone(cfg["logger"]["wandb"]["id"])
        command = launch.command_for_runtime(Path("/repeat/runtime.yaml"), Path("/repeat"), stage="repeat_diagnose")
        self.assertEqual(command[command.index("--max-steps") + 1], "1921")

    def test_repeated_settings_reject_invalid_counts_seeds_and_radii(self):
        mutations = (
            ("draw_repeats", True), ("draw_repeats", 0), ("draw_repeats", 4.0),
            ("gradient_repeats", 1), ("gradient_repeats", True),
            ("evaluation_seeds", [1, 1]), ("evaluation_seeds", [-1, 2]),
            ("evaluation_seeds", [True, 2]), ("evaluation_seeds", [[1], 2]),
            ("evaluation_seeds", [1]), ("evaluation_seeds", "1,2"),
            ("direction_rms_fractions", [0.01, float("nan"), 0.1]),
            ("direction_rms_fractions", [0.01, float("inf"), 0.1]),
            ("direction_rms_fractions", [0.01, 0, 0.1]),
            ("direction_rms_fractions", [0.01, True, 0.1]),
            ("direction_rms_fractions", [1.0]), ("direction_rms_fractions", "0.01"),
            ("diagnostic_evaluation_events", 32769), ("ensemble_directory", "/an/ensemble"),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                settings = copy.deepcopy(self.settings)
                settings[key] = value
                with self.assertRaises(ValueError):
                    launch.validate_settings(settings)

    def test_protocol_and_arm_selection_are_explicit(self):
        old = launch.read_mapping(ROOT / "config/tau_reward_mechanisms_1920.yaml")
        for settings, stage in ((old, "repeat_diagnose"), (self.settings, "diagnose"), (self.settings, "trajectory")):
            with self.subTest(stage=stage):
                with self.assertRaises(ValueError):
                    launch.configure(source_runtime(), settings, self.metadata, pinned=Path("/r/source.ckpt"),
                                     output=Path("/r"), stage=stage)
        for arm, directory in (("member0", None), ("ensemble", Path("/an/ensemble")), ("inherited", Path("/an/ensemble"))):
            with self.subTest(arm=arm, directory=directory):
                with self.assertRaises(ValueError):
                    launch.configure(source_runtime(), self.settings, self.metadata, pinned=Path("/r/source.ckpt"),
                                     output=Path("/r"), stage="repeat_diagnose", arm=arm, ensemble_directory=directory)

    def test_repeat_prepare_only_never_launches_subprocess_or_ray(self):
        torch = ModuleType("torch")
        torch.load = lambda *a, **k: saved_state()
        filtered = ModuleType("evenet.dataset.filtered_data")
        filtered.validate_filtered_dataset = lambda p: {"rows": 416701 if p == launch.TRAIN_DIRECTORY else 119002}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "source.ckpt"
            checkpoint.touch()
            runtime = root / "source.yaml"
            runtime.write_text(yaml.safe_dump(source_runtime()))
            settings = dict(self.settings, checkpoint=str(checkpoint), source_runtime=str(runtime), series_root=str(root / "series"))
            config = root / "settings.yaml"
            config.write_text(yaml.safe_dump(settings))
            with patch.dict(sys.modules, {"torch": torch, "evenet.dataset.filtered_data": filtered}), \
                    patch.object(launch.subprocess, "run") as run, patch("builtins.print"):
                launch.main([str(config), "prepare"])
                launch.main([str(config), "repeat_diagnose", "--prepare-only"])
                run.assert_not_called()
            for folder in (root / "series").iterdir():
                cfg = launch.read_mapping(folder / "runtime.yaml")
                self.assertEqual(cfg["dgpo"]["tau_ratio"]["mechanism_probe"]["mode"], "repeat_diagnose")
                self.assertEqual(cfg["dgpo"]["tau_ratio"]["mechanism_probe"]["updates"], 0)
                self.assertTrue(json.loads((folder / launch.MANIFEST).read_text())["prepared_only"])

    def test_summary_reads_repeated_report_without_converting_missing_draws_into_success(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "repeat_diagnose-one"
            (folder / "mechanisms").mkdir(parents=True)
            (folder / launch.MANIFEST).write_text(json.dumps({"stage": "repeat_diagnose", "reward_arm": "inherited",
                "relative_updates": 0, "prepared_only": False}))
            (folder / "mechanisms" / "local_direction_repeats.json").write_text(json.dumps({
                "complete": False, "actual_updates": 0, "draw_repeats": 4, "completed_draws": 2,
                "evaluation_seeds": [202610041, 202610042],
                "summary": {"screening": "Four draws are not proof", "valid_draws": 2}}))
            summary = launch.summarize(dict(self.settings, series_root=directory))
            report = summary["invocations"][0]["reports"]["local_direction_repeats.json"]
            self.assertFalse(report["complete"])
            self.assertEqual(report["completed_draws"], 2)
            self.assertEqual(report["summary"]["valid_draws"], 2)
            self.assertEqual(summary["paired_cross_arm"], [])


if __name__ == "__main__":
    unittest.main()
