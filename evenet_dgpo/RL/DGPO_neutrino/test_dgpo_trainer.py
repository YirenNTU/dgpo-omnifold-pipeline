"""Focused regression tests for DGPO trainer orchestration helpers."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)

# Keep this orchestration test independent of optional torchvision imports.
_dbg = types.ModuleType("evenet.utilities.debug_tool")


def _noop_time_decorator(name=None):
    def _wrapper(func):
        return func

    return _wrapper


_dbg.time_decorator = _noop_time_decorator
sys.modules.setdefault("evenet.utilities.debug_tool", _dbg)

from RL.DGPO_neutrino import dgpo_trainer
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    build_reference_trust_pool,
)


class _OrderChangingShard:
    """A restarted iterator deliberately returns a different event order."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.calls = 0

    def iter_torch_batches(self, **_kwargs):
        self.calls += 1
        order = torch.arange(len(self.batch["x"]))
        if self.calls % 2 == 0:
            order = order.flip(0)
        yield {
            key: value.index_select(0, order)
            for key, value in self.batch.items()
        }


class _ChunkedShard:
    def __init__(self, batch: dict[str, torch.Tensor], chunk_size: int) -> None:
        self.batch = batch
        self.chunk_size = int(chunk_size)

    def iter_torch_batches(self, **_kwargs):
        count = len(self.batch["x"])
        for start in range(0, count, self.chunk_size):
            stop = min(start + self.chunk_size, count)
            yield {key: value[start:stop] for key, value in self.batch.items()}


class _Policy(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)


class TestWandbClocks(unittest.TestCase):
    def test_checkpoint_transfer_metrics_survive_compact_profiles(self):
        key = "checkpoint_transfer/heldout/delta_mean"
        self.assertTrue(dgpo_trainer._wandb_critical_keep(key))
        self.assertTrue(dgpo_trainer._wandb_simplified_keep(key, .1))
        self.assertEqual(dgpo_trainer._wandb_train_payload({key: .1})[key], .1)

    @staticmethod
    def _live_progress_logger(wb, *, is_rank0=True):
        # Execute the actual nested production callback without launching Ray,
        # loading datasets, or replacing the phase-validation path with a mock.
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(dgpo_trainer.dgpo_train_loop))
        callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == "_log_omnifold_fit_progress")
        harness = ast.parse(
            "def make_logger(wandb_mod, is_rank0):\n"
            "    global_step = 170\n"
            "    omnifold_live_log_index = 0\n"
            "    omnifold_phase_ids = _OMNIFOLD_LIVE_PHASE_IDS\n"
            "    classifier_loss_tracker = _ClassifierFitLossTracker()\n"
            "    classifier_plot_tracker = _ClassifierTrainingPlotTracker()\n"
        )
        harness.body[0].body.extend([callback, ast.Return(value=ast.Name(id=callback.name, ctx=ast.Load()))])
        namespace = dict(vars(dgpo_trainer))
        exec(compile(ast.fix_missing_locations(harness), "<live-progress-regression>", "exec"), namespace)
        return namespace["make_logger"](wb, is_rank0)

    def setUp(self):
        dgpo_trainer._wandb_reset_step_tracker()
        patcher = mock.patch.object(dgpo_trainer, "_dgpo_wandb_yaml_section",
                                    return_value=({"profile": "critical"}, "logger.wandb"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_baseline_fit_train_monitor_and_validation_have_distinct_rows(self):
        wb = mock.Mock()
        dgpo_trainer._wandb_log_validation(wb, {"val_ztautau/jsd/current/topology/cos_opening": .02},
                                          epoch=-1, wandb_step=0)
        for index in range(1, 101):
            dgpo_trainer._wandb_log_auxiliary(wb, {
                "omnifold_live/log_index": index,
                "omnifold_live/meta/dgpo_epoch": -1,
                "omnifold_live/meta/fit_step": index,
                "omnifold_live/residual_reward/validation_loss": .69,
            }, current_global_step=0)
        dgpo_trainer._wandb_log_step(wb, {"epoch": 0, "train/loss/total": .1}, step=1)
        dgpo_trainer._wandb_log_with_step(wb, {
            "staleness/raw_auc": .60, "staleness/epoch": 0,
            "staleness/global_step": 5,
        }, step=dgpo_trainer._wandb_epoch_end_step(5))
        dgpo_trainer._wandb_log_validation(wb, {"val_ztautau/jsd/current/topology/cos_opening": .01},
                                          epoch=0, wandb_step=dgpo_trainer._wandb_epoch_end_step(10))
        dgpo_trainer._wandb_log_step(wb, {"epoch": 1, "train/loss/total": .09}, step=11)
        calls = wb.log.call_args_list
        self.assertEqual([c.kwargs["step"] for c in calls], list(range(len(calls))))
        self.assertTrue(all(c.kwargs["commit"] is True for c in calls))
        self.assertEqual((calls[0].args[0]["global_step"], calls[0].args[0]["epoch"]), (0, -1))
        self.assertTrue(all(c.args[0]["global_step"] == 0 for c in calls[1:101]))
        self.assertEqual([c.args[0]["global_step"] for c in calls[-4:]], [1, 5, 10, 11])
        self.assertEqual([c.args[0]["epoch"] for c in calls[-4:]], [0, 0, 0, 1])
        self.assertNotIn("train/loss/total", calls[0].args[0])

    def test_new_wandb_run_preserves_checkpoint_training_clock(self):
        wb = mock.Mock()
        # Checkpoint step 70 is NOT W&B transport row 70, nor classifier step 3474.
        dgpo_trainer._wandb_log_step(wb, {"epoch": 7, "train/loss/total": .1}, step=71)
        self.assertEqual(wb.log.call_args.kwargs["step"], 0)
        self.assertEqual(wb.log.call_args.args[0]["global_step"], 71)
        dgpo_trainer._wandb_reset_step_tracker(next_row=100)
        dgpo_trainer._wandb_log_step(wb, {"epoch": 7, "train/loss/total": .09}, step=72)
        self.assertEqual(wb.log.call_args.kwargs["step"], 100)

    def test_classifier_loss_curves_keep_local_steps_and_separate_refits(self):
        wb = mock.Mock()
        settings = {"profile": "critical", "classifier_loss_curves": False,
                    "classifier_loss_curves_raw": True}
        with mock.patch.object(dgpo_trainer, '_dgpo_wandb_yaml_section', return_value=(settings, 'logger.wandb')):
            logger = self._live_progress_logger(wb)
            for phase, fold, step, validate in (
                ('residual_reward', 1, 10, True),
                ('residual_reward', 1, 20, False),
                ('residual_reward', 2, 10, True),
                ('raw_staleness_monitor', 0, 10, True),
                ('raw_staleness_monitor', 0, 20, False),
                ('raw_staleness_monitor', 0, 10, True),
            ):
                logger(phase, {'step': step, 'fold': fold, 'iteration': 1,
                              'training_loss': .68, 'training_balanced_accuracy': .51,
                              'validation_loss': .69, 'validation_balanced_accuracy': .52,
                              'validation_auc': .53,
                              'learning_rate': 1e-3 if step <= 10 else 2e-4,
                              'topology_training_stage': 1 if step <= 10 else 2,
                              'validation_evaluated': validate}, epoch_value=16)
        wb.plot.line_series.assert_not_called()
        rows = [call.args[0] for call in wb.log.call_args_list]
        loss_keys = [next(k for k in row if k.startswith('classifier_fit/') and k.endswith('/training_loss')) for row in rows]
        self.assertEqual(loss_keys[0], loss_keys[1])
        self.assertEqual(loss_keys[3], loss_keys[4])
        self.assertEqual(len(set(loss_keys)), 4)
        self.assertNotEqual(loss_keys[3], loss_keys[5])
        definitions = {c.args[0]: c.kwargs for c in wb.define_metric.call_args_list}
        for index, (row, loss_key) in enumerate(zip(rows, loss_keys)):
            definition = definitions[loss_key]
            self.assertEqual(row[definition['step_metric']], [10,20,10,10,20,10][index])
            self.assertFalse(definition['hidden'])
            self.assertFalse(definition['step_sync'])
            self.assertEqual(row['global_step'], 170)
            self.assertEqual(row['epoch'], 16)
            self.assertIn(loss_key.replace('/training_loss', '/training_balanced_accuracy'), row)
            self.assertIn(loss_key.replace('/training_loss', '/learning_rate'), row)
            self.assertIn(
                loss_key.replace('/training_loss', '/topology_training_stage'),
                row,
            )
            has_val = loss_key.replace('/training_loss','/validation_loss') in row
            self.assertEqual(has_val, index not in (1,4))
            if has_val:
                self.assertIn(loss_key.replace('/training_loss', '/validation_balanced_accuracy'), row)
                self.assertIn(loss_key.replace('/training_loss', '/validation_auc'), row)

    def test_classifier_training_charts_use_logged_steps_from_repeat1_fold1(self):
        import wandb

        wb = mock.Mock()
        wb.plot = wandb.plot
        settings = {"profile": "critical", "classifier_loss_curves": True}
        with mock.patch.object(
            dgpo_trainer,
            "_dgpo_wandb_yaml_section",
            return_value=(settings, "logger.wandb"),
        ):
            logger = self._live_progress_logger(wb)
            for iteration, repeat, fold, step, loss in (
                (1, 1, 1, 10, 0.68),
                (1, 1, 1, 20, 0.64),
                (1, 1, 2, 999, 0.31),  # another fold: ignored
                (2, 2, 1, 777, 0.25),  # another repeat: ignored
                (2, 1, 2, 888, 0.22),  # reports first, but is ignored
                (2, 1, 1, 7, 0.61),
                (2, 1, 1, 17, 0.57),
            ):
                logger(
                    "residual_reward",
                    {
                        "step": step,
                        "fold": fold,
                        "repeat": repeat,
                        "iteration": iteration,
                        "training_loss": loss,
                        "validation_loss": loss + 0.01,
                        "validation_balanced_accuracy": 0.70,
                        "validation_auc": 0.75,
                        "learning_rate": 1e-4,
                        "logit_class_mean_separation": 0.25 + loss,
                        "parameter_rms_direct_topology_head": 0.02,
                        "gradient_rms_direct_topology_head": 0.003,
                        "gradient_to_parameter_rms_ratio_direct_topology_head": 0.15,
                        "validation_evaluated": True,
                    },
                    epoch_value=16,
                )
        rows = [call.args[0] for call in wb.log.call_args_list]
        key = "Classifier training/Reward/Train loss"
        charts = [row[key] for row in rows if key in row]
        self.assertEqual(len(charts), 4)
        latest = charts[-1]
        self.assertEqual(latest.spec.string_fields["title"],
                         "Reward classifier · Train loss · DGPO step 170")
        self.assertEqual(
            latest.table.data,
            [
                [10, "Iteration 1", 0.68],
                [20, "Iteration 1", 0.64],
                [7, "Iteration 2", 0.61],
                [17, "Iteration 2", 0.57],
            ],
        )
        separation_key = "Classifier training/Reward/Logit class separation"
        separation_charts = [row[separation_key] for row in rows if separation_key in row]
        self.assertEqual(
            separation_charts[-1].table.data,
            [
                [10, "Iteration 1", 0.93],
                [20, "Iteration 1", 0.89],
                [7, "Iteration 2", 0.86],
                [17, "Iteration 2", 0.82],
            ],
        )
        self.assertIn(
            "omnifold_live/residual_reward/gradient_rms_direct_topology_head",
            rows[-1],
        )
        self.assertFalse(any(k.startswith("classifier_fit/") for row in rows for k in row))

    def test_global_confirmation_real_progress_callbacks_and_axes(self):
        wb = mock.Mock()
        logger = self._live_progress_logger(wb)
        with tempfile.TemporaryDirectory() as tmp:
            candidate, incumbent = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
            ckpt = Path(tmp) / "best.ckpt"
            torch.save({"state_dict": incumbent.state_dict()}, ckpt)
            def judge(pool, phase):
                for step in (10, 20):
                    logger(phase, {"step": step, "validation_auc": .67,
                                   "saturated": float(step == 20)}, epoch_value=16)
                return {"raw_auc_gap": .17 if pool == "candidate" else .19, "raw_audit_saturated": 1.}
            accepted, _ = dgpo_trainer._confirm_global_raw_candidate(
                model=candidate, comparison_model=incumbent, best_checkpoint=ckpt,
                materialize_pair=lambda *_: (None, "candidate", "incumbent"),
                fit_judge=judge, min_delta=.001,
            )
        self.assertTrue(accepted)
        rows = [call.args[0] for call in wb.log.call_args_list]
        self.assertEqual([row["omnifold_live/meta/phase_id"] for row in rows], [9, 9, 10, 10])
        self.assertEqual([row["omnifold_live/log_index"] for row in rows], [1, 2, 3, 4])
        self.assertTrue(all(row["global_step"] == 170 for row in rows))
        self.assertTrue(all(row["omnifold_live/meta/dgpo_epoch"] == 16 for row in rows))
        self.assertIn("omnifold_live/global_best_candidate/validation_auc", rows[0])
        self.assertIn("omnifold_live/global_best_incumbent/validation_auc", rows[-1])

    def test_live_progress_without_wandb_and_nonzero_rank(self):
        logger = self._live_progress_logger(None)
        for phase in ("global_best_candidate", "global_best_incumbent"):
            logger(phase, {"step": 10}, epoch_value=16)
        with self.assertRaisesRegex(ValueError, "unknown OmniFold progress phase"):
            logger("typo", {}, epoch_value=16)
        wb = mock.Mock()
        self._live_progress_logger(wb, is_rank0=False)("global_best_candidate", {}, epoch_value=16)
        wb.log.assert_not_called()

    def test_live_progress_labels_actual_optimizer_learning_rates(self):
        logger = self._live_progress_logger(None)
        with mock.patch.object(dgpo_trainer._log, "info") as log:
            logger("raw_staleness_audit", {
                "step": 70, "learning_rate": 2e-4,
                "optimizer_group_lr_0": 2e-4,
                "optimizer_group_lr_head": 2e-4,
                "optimizer_group_lr_backbone": 1e-5,
                "optimizer_group_lr_decoder": 5e-5,
                "optimizer_group_lr_adapter": 5e-5,
            }, epoch_value=-1)
        args = log.call_args.args
        message = args[0] % args[1:]
        self.assertIn("base_lr=0.0002", message)
        self.assertIn("actual_lrs=[head=0.0002, backbone=1e-05, decoder=5e-05, adapter=5e-05]", message)
        self.assertNotIn("actual_lrs=[0=", message)

    def test_all_classifier_diagnostic_charts_survive_critical_profile(self):
        for role in ("Reward", "Fresh audit"):
            for _metric, label, _validation in dgpo_trainer._ClassifierTrainingPlotTracker._METRICS:
                key = f"Classifier training/{role}/{label}"
                self.assertIn(key, dgpo_trainer._WANDB_CLASSIFIER_TRAINING_CHARTS)
                self.assertTrue(dgpo_trainer._wandb_critical_keep(key))

    def test_critical_axes_hide_metadata_and_do_not_forward_fill_stale_epochs(self):
        wb = mock.Mock()
        dgpo_trainer._wandb_define_axes(wb, critical=True)
        definitions = {c.args[0]: c.kwargs for c in wb.define_metric.call_args_list}
        self.assertEqual(definitions["*"]["step_metric"], "global_step")
        self.assertTrue(definitions["*"]["hidden"])
        self.assertEqual(definitions["staleness/raw_auc"]["step_metric"], "global_step")
        self.assertEqual(definitions["val_ztautau/topology/cos_opening"]["step_metric"], "epoch")
        self.assertFalse(definitions["staleness/raw_auc"]["step_sync"])
        self.assertTrue(definitions["omnifold_live/*"]["hidden"])
        self.assertEqual(definitions["omnifold_live/*"]["step_metric"], "omnifold_live/log_index")
        self.assertNotIn("omnifold_live/residual_reward/training_loss", definitions)
        for key in dgpo_trainer._WANDB_CLASSIFIER_TRAINING_CHARTS:
            self.assertFalse(definitions[key]["hidden"])

    def test_compact_classifier_logging_uses_fixed_phase_metrics_only(self):
        wb = mock.Mock()
        settings = {"profile": "critical", "classifier_loss_curves": False}
        with mock.patch.object(
            dgpo_trainer,
            "_dgpo_wandb_yaml_section",
            return_value=(settings, "logger.wandb"),
        ):
            logger = self._live_progress_logger(wb)
            logger(
                "residual_reward",
                {
                    "step": 10,
                    "parameter_update_rms_adapter": 0.00005,
                    "update_to_parameter_rms_ratio_decoder": 0.001,
                    "optimizer_group_lr_0": 0.0002,
                    "stability/probe/bce_delta": 0.01,
                    "stability/representation_probe/fourier/holdout_auc": 0.7,
                    "stability/layer/representation/fourier/activation_rms/rankmean": 0.2,
                    "stability/layer/bank.decoder.norm/pre_norm_variance_min/rankmin": 0.02,
                    "iteration": 1,
                    "fold": 1,
                    "training_loss": 0.68,
                    "training_balanced_accuracy": 0.61,
                    "validation_loss": 0.69,
                    "validation_balanced_accuracy": 0.60,
                    "validation_auc": 0.66,
                    "threshold_reached": 1.0,
                    "validation_oriented_balanced_accuracy": 0.70,
                    "validation_balanced_accuracy_lcb": 0.70,
                    "validation_balanced_accuracy_lcb_standard_error": 0.001,
                    "validation_balanced_accuracy_lcb_streak": 3.0,
                    "validation_evaluated": True,
                },
                epoch_value=0,
            )
        row = wb.log.call_args.args[0]
        self.assertFalse(any(key.startswith("classifier_fit/") for key in row))
        for metric in ("parameter_update_rms_adapter", "update_to_parameter_rms_ratio_decoder", "optimizer_group_lr_0",
                       "stability/probe/bce_delta", "stability/layer/bank.decoder.norm/pre_norm_variance_min/rankmin",
                       "stability/representation_probe/fourier/holdout_auc",
                       "stability/layer/representation/fourier/activation_rms/rankmean"):
            self.assertTrue(any(key.endswith("/" + metric) for key in row), metric)
        for metric in (
            "training_loss",
            "training_balanced_accuracy",
            "validation_loss",
            "validation_balanced_accuracy",
            "validation_auc",
            "threshold_reached",
            "validation_oriented_balanced_accuracy",
            "validation_balanced_accuracy_lcb",
            "validation_balanced_accuracy_lcb_standard_error",
            "validation_balanced_accuracy_lcb_streak",
        ):
            self.assertIn(f"omnifold_live/residual_reward/{metric}", row)

    def test_critical_profile_filters_dormant_ablation_and_duplicate_plots(self):
        payload = {
            "global_step": 10, "epoch": 0, "staleness/raw_auc": .60,
            "reference_trust/velocity_mse": .002,
            "reference_trust/velocity_mse_ratio": .04,
            "reference_trust/delta": .1, "omnifold/candidate/residual_closure_auc": .504,
            "omnifold/fit/iter01/validation_auc": .64,
            "val_ztautau/topology/cos_opening": 1,
            "val_neutrino/by_process/Z/phi_rmse": .4,
            "reference_trust/extragradient/enabled": 0,
            "val_ztautau/residual/ref/topology/cos_opening/mean": .2,
            "omnifold_live/meta/fit_step": 300,
            "staleness/raw_classifier_warm_started": 0,
            "staleness/raw_no_improvement_patience": 24,
            "staleness/raw_audit_training_ready": 1,
            "staleness/raw_audit_training_min_steps": 600,
            "staleness/raw_audit_training_steps": 660,
            "staleness/raw_audit_training_epochs": 110,
        }
        clean = dgpo_trainer._wandb_apply_simplified_profile(payload)
        self.assertEqual(len(clean), len(payload) - 3)
        self.assertIn("omnifold_live/meta/fit_step", clean)
        self.assertIn("reference_trust/velocity_mse", clean)
        self.assertIn("reference_trust/velocity_mse_ratio", clean)
        self.assertEqual(clean["staleness/raw_no_improvement_patience"], 24)
        self.assertNotIn("reference_trust/extragradient/enabled", clean)

    def test_mid_epoch_zero_resume_does_not_relabel_policy_as_baseline(self):
        self.assertTrue(dgpo_trainer._should_log_pretraining_baseline(0, 0))
        for epoch, step in ((0, 5), (1, 10), (7, 70)):
            self.assertFalse(dgpo_trainer._should_log_pretraining_baseline(epoch, step))


class TestSinglePoolScaling(unittest.TestCase):
    def test_one_dataset_read_and_no_external_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.parquet").touch()
            pool = object()
            with mock.patch.object(dgpo_trainer, "register_dataset", return_value=(pool, 100)) as register:
                train, val, n_train, n_val = dgpo_trainer._prepare_single_pool_datasets(
                    base_dir=root, base_val_dir=root, process_fn=None, platform_info=None,
                    dataset_options={"dataset_limit": 1.0, "val_dataset_limit": 1.0},
                )
                self.assertIs(train, val)
                self.assertEqual((n_train, n_val), (100, 100))
                register.assert_called_once()
                self.assertEqual(register.call_args.kwargs["dataset_limit"], 1.0)

    def test_external_validation_or_second_subsampling_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for val_dir, limit in ((root / "external", 1.0), (root, 0.01)):
                with self.assertRaises(ValueError):
                    dgpo_trainer._prepare_single_pool_datasets(
                        base_dir=root, base_val_dir=val_dir, process_fn=None, platform_info=None,
                        dataset_options={"dataset_limit": limit, "val_dataset_limit": 1.0},
                    )

    def test_pool_generation_can_drain_longer_shards_but_training_requires_all(self):
        for require_all, expected_op, global_flag in (
            (False, dgpo_trainer.dist.ReduceOp.MAX, 1),
            (True, dgpo_trainer.dist.ReduceOp.MIN, 0),
        ):
            def reduce_flag(flag, *, op):
                self.assertEqual(op, expected_op)
                flag.fill_(global_flag)
            with mock.patch.object(dgpo_trainer.dist, "all_reduce", side_effect=reduce_flag):
                batch, more = dgpo_trainer._next_batch_synced(
                    iter([]), world_size=2, device=torch.device("cpu"), require_all_ranks=require_all,
                )
            self.assertIsNone(batch)
            self.assertEqual(more, not require_all)

    def test_scaling_yamls_use_matched_live_checkpoints_and_disjoint_outputs(self):
        configs = []
        for percent in (1, 10):
            with self.subTest(percent=percent):
                configs.append(self._assert_scaling_config(percent))
        for section, key in (("ray", "results_dir"),):
            self.assertNotEqual(configs[0]["nersc"][section][key], configs[1]["nersc"][section][key])
        self.assertNotEqual(configs[0]["options"]["Training"]["model_checkpoint_save_path"],
                            configs[1]["options"]["Training"]["model_checkpoint_save_path"])
        self.assertNotEqual(configs[0]["logger"]["wandb"]["run_name"], configs[1]["logger"]["wandb"]["run_name"])

    def _assert_scaling_config(self, percent):
        from scripts.train_neutrino_backend import read_yaml, deep_update
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
        root = Path(__file__).resolve().parents[3]
        overlay = root / f"config/dgpo_omnifold_ztautau_{percent}pct_scaling_raw_plateau_vpkl.yaml"
        cfg = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"), read_yaml(overlay))
        adaptive = resolve_adaptive_config(cfg["dgpo"])
        self.assertTrue(adaptive.single_pool_train_validation)
        self.assertEqual(cfg["platform"]["data_parquet_dir"], cfg["platform"]["data_parquet_val_dir"])
        self.assertIn(f"diffusion_train_{percent}pct_seed42", cfg["platform"]["data_parquet_dir"])
        training = cfg["options"]["Training"]
        self.assertIn(f"diffusion_pretrain_{percent}pct_seed42", training["model_checkpoint_load_path"])
        self.assertFalse(training["EMA"]["replace_model_after_load"])
        self.assertFalse(training["EMA"]["use_for_generation"])
        self.assertEqual(cfg["logger"]["wandb"]["profile"], "critical")
        self.assertEqual(cfg["logger"]["wandb"]["resume"], "never")
        self.assertEqual(cfg["nersc"]["reproducibility"]["diffusion_pretrain_fraction"], percent / 100)
        self.assertEqual(training["model_checkpoint_load_path"], cfg["reward_config"]["omnifold"]["backbone_checkpoint"])
        self.assertEqual(cfg["dgpo"]["checkpoint_load_mode"], "weights_only")
        self.assertIsNone(cfg["dgpo"]["auto_resume_best_source_checkpoint_dir"])
        self.assertIsNone(cfg["dgpo"]["auto_resume_fallback_checkpoint_path"])
        self.assertFalse(adaptive.refit_once_on_resume)
        self.assertEqual(adaptive.staleness_every_n_steps, 5)
        self.assertEqual(adaptive.required_consecutive_epochs, 5)
        self.assertEqual(adaptive.trust_delta_max, 0.1)
        self.assertTrue(adaptive.raw_monitor_warm_start)
        for cap in (adaptive.pool_events, adaptive.refit_score_events, adaptive.probe_max_events,
                    adaptive.classifier_trust_probe_max_events):
            self.assertIsNone(cap)
        # No accidental change to network, optimizer settings, epoch/step budget.
        control = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
                             read_yaml(root / "config/dgpo_omnifold_ztautau_fullpretrain_40pct_raw_plateau_vpkl_monitor1_cheaptrust.yaml"))
        self.assertEqual(cfg["network"], control["network"])
        for key in ("learning_rate", "weight_decay", "epochs"):
            self.assertEqual(training[key], control["options"]["Training"][key])
        return cfg


class TestMidEpochResume(unittest.TestCase):
    def test_mid_epoch_checkpoint_runs_only_remaining_steps(self):
        checkpoint = {"epoch": 0, "dgpo_next_epoch": 0, "global_step": 5, "dgpo_epoch_step": 5}
        progress = dgpo_trainer._resume_logical_epoch_step(checkpoint, 10)
        steps = list(range(progress + 1, 11))
        self.assertEqual(steps, [6, 7, 8, 9, 10])

    def test_legacy_and_completed_snapshots_start_epoch_at_zero(self):
        self.assertEqual(dgpo_trainer._resume_logical_epoch_step(None, 10), 0)
        self.assertEqual(dgpo_trainer._resume_logical_epoch_step({"dgpo_next_epoch": 2}, 10), 0)
        self.assertEqual(dgpo_trainer._resume_logical_epoch_step({"dgpo_epoch_step": 0}, None), 0)

    def test_inconsistent_mid_epoch_metadata_fails(self):
        for budget, next_epoch, progress in ((None, 0, 5), (5, 0, 5), (10, 1, 5), (10, 0, -1)):
            with self.subTest(budget=budget, next_epoch=next_epoch, progress=progress):
                with self.assertRaises(ValueError):
                    dgpo_trainer._resume_logical_epoch_step(
                        {"epoch": 0, "dgpo_next_epoch": next_epoch, "dgpo_epoch_step": progress}, budget,
                    )


class TestPinnedRestartSelection(unittest.TestCase):
    def test_first_start_applies_pinned_resume_without_last_ckpt(self):
        self.assertTrue(
            dgpo_trainer._should_apply_pinned_classifier_restart(
                pinned=True, load_mode="resume", auto_resume_checkpoint=None,
                best_source_dir=None,
            )
        )

    def test_existing_last_ckpt_skips_pinned_reset(self):
        self.assertFalse(
            dgpo_trainer._should_apply_pinned_classifier_restart(
                pinned=True, load_mode="resume",
                auto_resume_checkpoint="/tmp/this/last.ckpt",
                best_source_dir=None,
            )
        )

    def test_weights_only_first_start_cannot_pin(self):
        with self.assertRaises(ValueError):
            dgpo_trainer._should_apply_pinned_classifier_restart(
                pinned=True, load_mode="weights_only", auto_resume_checkpoint=None,
                best_source_dir=None,
            )


class TestBestPointNewExperiment(unittest.TestCase):
    def test_pinned_restart_keeps_classifiers_but_not_historical_scores(self):
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldState
        a = {'single_pool_train_validation': True, 'single_pool_split_seed': 42,
             'recalibration': {'seed': 7, 'crossfit_folds': 2}}
        cache = {'outer_partition': {'schema': 'condition-hash-80-20-v1', 'seed': 42},
                 'protocol': {'scheme': 'condition_sha256_v1', 'seed': 7, 'folds': 2},
                 'models': [{'iteration': 1, 'fold': i, 'state': {'w': torch.ones(1)}} for i in (1, 2)]}
        increments = [
            {'state': {'w': torch.ones(1)}, 'base_digest': 'base',
             'packing_spec': {'shape': [1]}}
            for _ in range(2)
        ]
        previous = AdaptiveOmniFoldState(reward_round_id=3, baseline_auc_gap=.1, trigger_threshold=.02,
                                        raw_monitor_state={'protocol': {'seed': 42}, 'state': {'w': torch.ones(1)}})
        p = {'dgpo_adaptive_omnifold_state': previous.to_dict(),
             'dgpo_omnifold_reward_stack': {'reward': {
                 'warm_start_state': cache, 'increments': increments,
                 'increment_iterations': [1, 1],
                 'increment_coefficients': [.5, .5],
             }},
             'dgpo_ref_state_dict': {}, 'dgpo_round_ref_state_dict': {},
             'dgpo_round_ref_sha256': 'test', 'dgpo_omnifold_reward_metadata': {},
             'state_dict': {'w': torch.ones(1)}, 'dgpo_optimizer_state_dict': {'old': 1}}
        result = dgpo_trainer._prepare_pinned_classifier_restart(p, {'adaptive_omnifold': a})
        self.assertIs(result['dgpo_omnifold_reward_stack'], p['dgpo_omnifold_reward_stack'])
        self.assertEqual(result['global_step'], 0)
        self.assertNotIn('dgpo_optimizer_state_dict', result)
        state = result['dgpo_adaptive_omnifold_state']
        self.assertTrue(state['raw_monitor_state']['inherit_monitor_as_is'])
        self.assertNotIn('recertify_inherited_weights', state['raw_monitor_state'])
        self.assertFalse(state['raw_monitor_baseline_pending'])
        self.assertEqual(state['probe_history'], [])
        self.assertNotIn('inherit_monitor_as_is', p['dgpo_adaptive_omnifold_state']['raw_monitor_state'])
        # The inherited reference keeps its installed stack but restarts the
        # global-best trust schedule at age zero instead of failing the resume.
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import clamp_fixed_trust_radius_after_resume
        restarted = AdaptiveOmniFoldState.from_dict(state)
        cfg = SimpleNamespace(trust_boundary_enabled=True, trust_radius_mode="best_decay",
                              trust_delta_max=0.1, trust_delta_floor=0.02, trust_best_decay_factor=0.9)
        clamp_fixed_trust_radius_after_resume(restarted, cfg=cfg, initialize_round_decay=True)
        self.assertEqual(restarted.trust_best_decay_count, 0)
        self.assertEqual(restarted.trust_best_decay_round_id, 3)
        self.assertEqual(restarted.trust_current_delta, 0.1)
        self.assertEqual(restarted.last_decision, 'new_experiment_from_best')
        self.assertIsNone(restarted.policy_warmup_protocol)
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
            start_inherited_round_policy_warmup, policy_round_warmup_metrics,
        )
        warmup_cfg = SimpleNamespace(policy_warmup_steps=10, policy_warmup_start_factor=.1)
        with self.assertRaises(ValueError):
            policy_round_warmup_metrics(restarted, cfg=warmup_cfg)
        metrics = start_inherited_round_policy_warmup(restarted, cfg=warmup_cfg, restart=True)
        self.assertAlmostEqual(metrics['train/round_warmup/lr_scale'], .1)
        self.assertEqual(restarted.policy_warmup_round_id, 3)
        resume = AdaptiveOmniFoldState.from_dict(restarted.to_dict())
        resume.raw_monitor_baseline_pending = False
        resume.last_decision = 'raw_improved'
        resume.policy_warmup_protocol = None
        resume.policy_warmup_round_id = -1
        resume.policy_warmup_completed_updates = 0
        resume.raw_patience_warmup_updates_seen = 0
        self.assertEqual(start_inherited_round_policy_warmup(resume, cfg=warmup_cfg, global_step=30), {})
        metrics = start_inherited_round_policy_warmup(resume, cfg=warmup_cfg, global_step=0)
        self.assertAlmostEqual(metrics['train/round_warmup/lr_scale'], .1)

        repeated = copy.deepcopy(p)
        repeated_cfg = copy.deepcopy(a)
        repeated_cfg['recalibration']['crossfit_repeats'] = 5
        repeated_cache = repeated['dgpo_omnifold_reward_stack']['reward']['warm_start_state']
        repeated_cache['protocol']['repeats'] = 5
        repeated_cache['protocol']['repeat_seed_stride'] = 104729
        repeated_cache['models'] = []
        repeated_reward = repeated['dgpo_omnifold_reward_stack']['reward']
        repeated_reward['increments'] = [copy.deepcopy(increments[0]) for _ in range(10)]
        repeated_reward['increment_iterations'] = [1] * 10
        repeated_reward['increment_coefficients'] = [.1] * 10
        repeated_result = dgpo_trainer._prepare_pinned_classifier_restart(
            repeated, {'adaptive_omnifold': repeated_cfg}
        )
        self.assertEqual(
            len(
                repeated_result['dgpo_omnifold_reward_stack']['reward']
                ['increments']
            ),
            10,
        )

        p['dgpo_omnifold_reward_stack']['reward']['increments'].pop()
        with self.assertRaisesRegex(ValueError, 'every serialized'):
            dgpo_trainer._prepare_pinned_classifier_restart(p, {'adaptive_omnifold': a})

    def test_pinned_restart_rebuilds_only_missing_non_warm_raw_monitor(self):
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldState

        adaptive = {
            'single_pool_train_validation': False,
            'single_pool_split_seed': 42,
            'trigger': {'warm_start_classifier': False},
            'recalibration': {'seed': 7, 'crossfit_folds': 2, 'crossfit_repeats': 3},
        }
        increments = [
            {'state': {'w': torch.ones(1)}, 'base_digest': 'base',
             'packing_spec': {'shape': [1]}}
            for _ in range(6)
        ]
        cache = {
            'outer_partition': None,
            'protocol': {
                'scheme': 'condition_sha256_v1', 'seed': 7, 'folds': 2,
                'repeats': 3,
            },
            'models': [],
        }
        previous = AdaptiveOmniFoldState(
            reward_round_id=1, baseline_auc_gap=.1, trigger_threshold=.02,
            raw_monitor_state={},
        )
        checkpoint = {
            'dgpo_adaptive_omnifold_state': previous.to_dict(),
            'dgpo_omnifold_reward_stack': {'reward': {
                'warm_start_state': cache,
                'increments': increments,
                'increment_iterations': [1] * 6,
                'increment_coefficients': [1.0 / 6.0] * 6,
            }},
            'dgpo_ref_state_dict': {}, 'dgpo_round_ref_state_dict': {},
            'dgpo_round_ref_sha256': 'paired',
            'dgpo_omnifold_reward_metadata': {},
            'state_dict': {'w': torch.ones(1)},
        }

        result = dgpo_trainer._prepare_pinned_classifier_restart(
            checkpoint, {'adaptive_omnifold': adaptive}
        )
        state = result['dgpo_adaptive_omnifold_state']
        self.assertEqual(state['raw_monitor_state'], {})
        self.assertTrue(state['raw_monitor_baseline_pending'])
        self.assertIs(
            result['dgpo_omnifold_reward_stack'],
            checkpoint['dgpo_omnifold_reward_stack'],
        )

        requires_monitor = copy.deepcopy(adaptive)
        requires_monitor['trigger']['warm_start_classifier'] = True
        with self.assertRaisesRegex(ValueError, 'warm_start_classifier=true'):
            dgpo_trainer._prepare_pinned_classifier_restart(
                checkpoint, {'adaptive_omnifold': requires_monitor}
            )

    def test_reset_clock_optimizer_and_trust_without_changing_reward_pair(self):
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
            AdaptiveOmniFoldState, clamp_fixed_trust_radius_after_resume,
        )
        previous = AdaptiveOmniFoldState(
            reward_round_id=7, baseline_auc_gap=0.01, trigger_threshold=0.02,
            raw_best_epoch=8, raw_best_auc_gap=0.15, raw_no_improvement_streak=2,
            trust_current_delta=0.04, trust_radius_decay_step=9,
            trust_radius_decay_round_id=7, resume_refit_once_completed=True,
            resume_refit_once_id="v17", installed_at_epoch=8,
            raw_monitor_state={"state": {"weight": torch.ones(1)}},
            probe_history=[{"epoch": 8.0}],
        )
        checkpoint = {
            "epoch": 9, "dgpo_next_epoch": 10, "global_step": 100,
            "dgpo_adaptive_omnifold_state": previous.to_dict(),
            "state_dict": {"policy": torch.tensor([2.0])},
            "dgpo_round_ref_state_dict": {"policy": torch.tensor([1.0])},
            "dgpo_omnifold_reward_stack": {"id": 7},
            "dgpo_optimizer_state_dict": {"old": True}, "ema_state_dict": {},
            "dgpo_reward_round_id": 7, "dgpo_round_ref_sha256": "paired",
        }
        result = dgpo_trainer._prepare_best_point_restart(checkpoint)
        self.assertEqual((result["epoch"], result["dgpo_next_epoch"], result["global_step"]), (-1, 0, 0))
        for key in ("state_dict", "dgpo_round_ref_state_dict", "dgpo_omnifold_reward_stack"):
            self.assertIs(result[key], checkpoint[key])
        self.assertNotIn("dgpo_optimizer_state_dict", result)
        self.assertNotIn("ema_state_dict", result)
        self.assertIn("dgpo_optimizer_state_dict", checkpoint)  # parent untouched
        state = AdaptiveOmniFoldState.from_dict(result["dgpo_adaptive_omnifold_state"])
        self.assertTrue(state.raw_monitor_baseline_pending)
        self.assertTrue(state.resume_refit_once_completed)
        self.assertEqual(state.raw_no_improvement_streak, 0)
        self.assertEqual(state.probe_history, [])
        self.assertEqual(state.installed_at_epoch, -1)
        self.assertEqual(state.reward_round_id, 7)
        torch.testing.assert_close(state.raw_monitor_state["state"]["weight"], torch.ones(1))
        cfg = SimpleNamespace(trust_boundary_enabled=True, trust_radius_mode="round_decay",
                              trust_delta_max=0.1, trust_delta_floor=0.02, trust_round_decay_factor=0.9)
        clamp_fixed_trust_radius_after_resume(state, cfg=cfg)
        self.assertEqual(state.trust_radius_decay_step, 0)
        self.assertEqual(state.trust_current_delta, 0.1)
        # A regular restart restores the serialized schedule, not another reset.
        state.trust_radius_decay_step = 1
        state.trust_current_delta = 0.09
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        clamp_fixed_trust_radius_after_resume(restored, cfg=cfg)
        self.assertAlmostEqual(restored.trust_current_delta, 0.09)


class TestDistributedAutoResume(unittest.TestCase):
    def _resolve(self, **kwargs):
        options = dict(enabled=True, fallback_checkpoint_path=None,
                       best_source_checkpoint_dir="parent", world_size=2,
                       device=torch.device("cpu"))
        options.update(kwargs)
        return dgpo_trainer._resolve_distributed_auto_resume_checkpoint("output", **options)

    def test_rank_zero_resolves_once_and_broadcasts_canonical_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "best.ckpt"
            selected.touch()
            with mock.patch.object(dgpo_trainer.dist, "is_initialized", return_value=True), \
                 mock.patch.object(dgpo_trainer.dist, "get_rank", return_value=0), \
                 mock.patch.object(dgpo_trainer, "resolve_dgpo_auto_resume_checkpoint", return_value=selected) as resolve, \
                 mock.patch.object(dgpo_trainer.dist, "broadcast_object_list") as broadcast, \
                 mock.patch.object(dgpo_trainer.dist, "all_reduce"):
                self.assertEqual(self._resolve(), selected)
                resolve.assert_called_once_with("output", enabled=True,
                    fallback_checkpoint_path=None, best_source_checkpoint_dir="parent")
                self.assertEqual(broadcast.call_args.args[0], [str(selected), None])

    def test_peer_uses_broadcast_without_resolving_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "best.ckpt"
            selected.touch()
            def receive(message, **_kwargs):
                message[:] = [str(selected), None]
            with mock.patch.object(dgpo_trainer.dist, "is_initialized", return_value=True), \
                 mock.patch.object(dgpo_trainer.dist, "get_rank", return_value=1), \
                 mock.patch.object(dgpo_trainer, "resolve_dgpo_auto_resume_checkpoint") as resolve, \
                 mock.patch.object(dgpo_trainer.dist, "broadcast_object_list", side_effect=receive), \
                 mock.patch.object(dgpo_trainer.dist, "all_reduce"):
                self.assertEqual(self._resolve(), selected)
                resolve.assert_not_called()

    def test_rank_zero_failure_is_broadcast_before_raising(self):
        with mock.patch.object(dgpo_trainer.dist, "is_initialized", return_value=True), \
             mock.patch.object(dgpo_trainer.dist, "get_rank", return_value=0), \
             mock.patch.object(dgpo_trainer, "resolve_dgpo_auto_resume_checkpoint", side_effect=FileNotFoundError("best missing")), \
             mock.patch.object(dgpo_trainer.dist, "broadcast_object_list") as broadcast:
            with self.assertRaisesRegex(RuntimeError, "best missing"):
                self._resolve()
            self.assertIn("best missing", broadcast.call_args.args[0][1])

    def test_inconsistent_visibility_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = Path(tmp) / "best.ckpt"
            selected.touch()
            with mock.patch.object(dgpo_trainer.dist, "is_initialized", return_value=True), \
                 mock.patch.object(dgpo_trainer.dist, "get_rank", return_value=0), \
                 mock.patch.object(dgpo_trainer, "resolve_dgpo_auto_resume_checkpoint", return_value=selected), \
                 mock.patch.object(dgpo_trainer.dist, "broadcast_object_list"), \
                 mock.patch.object(dgpo_trainer.dist, "all_reduce", side_effect=lambda value, **_: value.zero_()):
                with self.assertRaisesRegex(RuntimeError, "not visible on every worker"):
                    self._resolve()

    def test_single_worker_delegates_without_collectives(self):
        with mock.patch.object(dgpo_trainer, "resolve_dgpo_auto_resume_checkpoint", return_value=None) as resolve, \
             mock.patch.object(dgpo_trainer.dist, "broadcast_object_list") as broadcast:
            self.assertIsNone(self._resolve(world_size=1))
            resolve.assert_called_once()
            broadcast.assert_not_called()


class TestFreshWandbRun(unittest.TestCase):
    def test_fresh_id_overrides_previous_run_and_environment(self):
        wb = mock.Mock()
        wb.run.step = 0
        settings = {"project": "test", "id": "old-id", "resume": "must", "fresh_run": True}
        with mock.patch.dict(sys.modules, {"wandb": wb}), \
             mock.patch.dict(os.environ, {"WANDB_RUN_ID": "old-id", "WANDB_RESUME": "must", "WANDB_DISABLED": ""}), \
             mock.patch.object(dgpo_trainer, "_dgpo_wandb_yaml_section", return_value=(settings, "logger.wandb")), \
             mock.patch.object(dgpo_trainer.global_config, "to_logger", return_value={}), \
             mock.patch.object(dgpo_trainer, "_wandb_define_axes"), \
             mock.patch.object(dgpo_trainer, "_wandb_critical_enabled", return_value=True), \
             mock.patch.object(dgpo_trainer, "_dgpo_wandb_publish_metric_docs"):
            self.assertTrue(dgpo_trainer._start_wandb_run())
            first_id = wb.init.call_args.kwargs["id"]
            self.assertRegex(first_id, r"^[0-9a-f]{8}$")
            self.assertEqual(wb.init.call_args.kwargs["resume"], "never")
            self.assertTrue(dgpo_trainer._start_wandb_run())
            self.assertNotEqual(wb.init.call_args.kwargs["id"], first_id)
            self.assertEqual(os.environ["WANDB_RUN_ID"], "old-id")


class TestRoundOptimizerReset(unittest.TestCase):
    def _optimizer(self):
        parameter = torch.nn.Parameter(torch.tensor([1., -2.]))
        optimizer = torch.optim.AdamW([parameter], lr=.02, weight_decay=.1, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
        for _ in range(3):
            parameter.grad = torch.tensor([.5, -.2])
            optimizer.step()
            scheduler.step()
        return parameter, optimizer, scheduler

    def test_complete_reset_preserves_weights_schedule_and_matches_fresh_adamw(self):
        import copy
        parameter, optimizer, scheduler = self._optimizer()
        before = parameter.detach().clone()
        schedule = copy.deepcopy(scheduler.state_dict())
        group_settings = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
        cfg = SimpleNamespace(reset_optimizer_state_on_install=True,
                              trust_reset_adam_first_moment=True)
        for _ in range(2):
            metrics = dgpo_trainer._reset_optimizer_after_reward_install(
                optimizer, cfg=cfg, accepted=True,
            )
            self.assertEqual(metrics["reference_trust/adam_full_state_reset"], 1.)
            self.assertFalse(optimizer.state)
            self.assertIsNone(parameter.grad)
            torch.testing.assert_close(parameter, before)
            self.assertEqual(scheduler.state_dict(), schedule)
            self.assertEqual({k: v for k, v in optimizer.param_groups[0].items() if k != "params"}, group_settings)
            fresh_parameter = torch.nn.Parameter(before.clone())
            fresh = torch.optim.AdamW([fresh_parameter], lr=group_settings["lr"],
                                      weight_decay=.1, amsgrad=True)
            parameter.grad = torch.tensor([-.3, .7])
            fresh_parameter.grad = parameter.grad.clone()
            optimizer.step()
            fresh.step()
            torch.testing.assert_close(parameter, fresh_parameter)
            self.assertEqual(optimizer.state[parameter]["step"].item(), 1.)
            before = parameter.detach().clone()

    def test_rejected_install_leaves_optimizer_untouched(self):
        parameter, optimizer, _ = self._optimizer()
        before = {k: v.clone() for k, v in optimizer.state[parameter].items()}
        grad = parameter.grad.clone()
        metrics = dgpo_trainer._reset_optimizer_after_reward_install(
            optimizer, cfg=SimpleNamespace(reset_optimizer_state_on_install=True), accepted=False,
        )
        self.assertEqual(metrics, {})
        for key, value in before.items():
            torch.testing.assert_close(optimizer.state[parameter][key], value)
        torch.testing.assert_close(parameter.grad, grad)

    def test_install_starts_checkpointed_warmup_without_changing_lr_or_decay(self):
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
            AdaptiveOmniFoldState, advance_policy_round_warmup, policy_round_warmup_metrics,
        )
        parameter, optimizer, scheduler = self._optimizer()
        initial_lr = optimizer.param_groups[0]["lr"]
        initial_weights = parameter.detach().clone()
        scheduler_step = scheduler.last_epoch
        cfg = SimpleNamespace(reset_optimizer_state_on_install=True,
                              trust_reset_adam_first_moment=False,
                              policy_warmup_steps=20, policy_warmup_start_factor=.1)
        state = AdaptiveOmniFoldState(reward_round_id=1)
        metrics = dgpo_trainer._reset_optimizer_after_reward_install(
            optimizer, cfg=cfg, accepted=True, adaptive_state=state,
        )
        self.assertEqual(metrics["train/round_warmup/lr_scale"], .1)
        self.assertEqual(optimizer.param_groups[0]["lr"], initial_lr)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], .1)
        self.assertEqual(scheduler.last_epoch, scheduler_step)
        torch.testing.assert_close(parameter, initial_weights)
        advance_policy_round_warmup(state, cfg=cfg, accepted=True)
        state = AdaptiveOmniFoldState.from_dict(state.to_dict())
        before = policy_round_warmup_metrics(state, cfg=cfg)
        self.assertEqual(dgpo_trainer._reset_optimizer_after_reward_install(
            optimizer, cfg=cfg, accepted=False, adaptive_state=state,
        ), {})
        self.assertEqual(policy_round_warmup_metrics(state, cfg=cfg), before)
        state.reward_round_id = 2
        metrics = dgpo_trainer._reset_optimizer_after_reward_install(
            optimizer, cfg=cfg, accepted=True, adaptive_state=state,
        )
        self.assertEqual(metrics["train/round_warmup/lr_scale"], .1)
        self.assertEqual(state.policy_warmup_completed_updates, 0)

    def test_legacy_first_moment_reset_keeps_variance_and_step(self):
        parameter, optimizer, _ = self._optimizer()
        variance = optimizer.state[parameter]["exp_avg_sq"].clone()
        step = optimizer.state[parameter]["step"].clone()
        metrics = dgpo_trainer._reset_optimizer_after_reward_install(
            optimizer, cfg=SimpleNamespace(reset_optimizer_state_on_install=False,
                                           trust_reset_adam_first_moment=True), accepted=True,
        )
        self.assertEqual(metrics["reference_trust/adam_first_moments_reset"], 1.)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], torch.zeros_like(parameter))
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], variance)
        torch.testing.assert_close(optimizer.state[parameter]["step"], step)


class TestDgpoWeightDecayResume(unittest.TestCase):
    def test_resume_keeps_new_decay_and_restores_adam_moments(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                old_param = torch.nn.Parameter(torch.tensor([1.0]))
                old_opt = torch.optim.AdamW([old_param], lr=0.1, weight_decay=0.0)
                old_schedule = torch.optim.lr_scheduler.LambdaLR(old_opt, lambda _: 1.0)
                old_param.grad = torch.ones_like(old_param)
                old_opt.step()
                old_schedule.step()
                saved = {"optimizer": old_opt.state_dict(), "scheduler": old_schedule.state_dict()}

                new_param = torch.nn.Parameter(old_param.detach().clone())
                new_opt = torch.optim.AdamW([new_param], lr=0.1, weight_decay=0.001)
                schedule = torch.optim.lr_scheduler.LambdaLR(new_opt, lambda _: 1.0)
                wrapped = dgpo_trainer._DgpoOptimizerWithSchedule(new_opt, schedule)
                wrapped.load_state_dict(saved["optimizer"] if legacy else saved)
                self.assertEqual(wrapped.param_groups[0]["weight_decay"], 0.001)
                torch.testing.assert_close(new_opt.state[new_param]["exp_avg"], old_opt.state[old_param]["exp_avg"])
                if not legacy:
                    self.assertEqual(schedule.last_epoch, old_schedule.last_epoch)
                # With moments cleared and a zero gradient, AdamW must still
                # apply parameter decay on an accepted optimizer update.
                new_opt.state.clear()
                before = new_param.detach().clone()
                new_param.grad = torch.zeros_like(new_param)
                wrapped.step()
                torch.testing.assert_close(new_param, before * (1 - 0.1 * 0.001))


class TestDgpoCosineResume(unittest.TestCase):
    def _make(self, cosine=True, total=100, floor=.1):
        params = [torch.nn.Parameter(torch.tensor([1.])), torch.nn.Parameter(torch.tensor([2.]))]
        opt = torch.optim.AdamW([{"params": [params[0]], "lr": .01},
                                {"params": [params[1]], "lr": .001}], weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: min(1., s / 10), lambda s: 1.])
        wrapper = dgpo_trainer._DgpoOptimizerWithSchedule(
            opt, scheduler, cosine_config={"total_steps": total, "min_lr_ratio": floor} if cosine else None,
            warmup_steps=10, warmup_groups=[True, False],
        )
        return params, wrapper

    def _advance(self, params, wrapper, steps):
        for _ in range(steps):
            for p in params:
                p.grad = torch.ones_like(p)
            wrapper.step()
            wrapper.scheduler_step()

    def test_cold_warmup_decay_floor_and_no_restart(self):
        params, wrapper = self._make()
        self.assertEqual([pg["lr"] for pg in wrapper.param_groups], [0., .001])
        self._advance(params, wrapper, 10)
        self.assertEqual([pg["lr"] for pg in wrapper.param_groups], [.01, .001])
        lrs = []
        for _ in range(100):
            self._advance(params, wrapper, 1)
            lrs.append(wrapper.param_groups[0]["lr"])
        self.assertTrue(all(b <= a for a, b in zip(lrs, lrs[1:])))
        self.assertAlmostEqual(lrs[-1], .001)
        self.assertAlmostEqual(wrapper.param_groups[1]["lr"], .0001)

    def test_migration_continuous_and_repeated_resume_exact(self):
        import copy
        old_params, old = self._make(cosine=False)
        self._advance(old_params, old, 30)
        saved = copy.deepcopy(old.state_dict())
        params, wrapper = self._make()
        wrapper.load_state_dict(saved)
        self.assertEqual(wrapper.scheduler.last_epoch, 30)
        self.assertEqual(wrapper.cosine_state["start_step"], 30)
        self.assertEqual([g["lr"] for g in wrapper.param_groups], [.01, .001])
        torch.testing.assert_close(wrapper.state[params[0]]["exp_avg"], old.state[old_params[0]]["exp_avg"])
        self._advance(params, wrapper, 10)
        resumed_params, resumed = self._make()
        resumed.load_state_dict(copy.deepcopy(wrapper.state_dict()))
        self.assertEqual(resumed.cosine_state, wrapper.cosine_state)
        for _ in range(80):
            self.assertEqual([g["lr"] for g in resumed.param_groups], [g["lr"] for g in wrapper.param_groups])
            self._advance(params, wrapper, 1)
            self._advance(resumed_params, resumed, 1)

    def test_migration_during_initial_warmup_and_round_reset(self):
        import copy
        params, old = self._make(cosine=False)
        self._advance(params, old, 5)
        _, wrapper = self._make()
        wrapper.load_state_dict(copy.deepcopy(old.state_dict()))
        self.assertEqual([g["lr"] for g in wrapper.param_groups], [.005, .001])
        before = copy.deepcopy(wrapper.state_dict())
        dgpo_trainer._reset_optimizer_after_reward_install(
            wrapper, cfg=SimpleNamespace(reset_optimizer_state_on_install=True,
                                         trust_reset_adam_first_moment=False), accepted=True,
        )
        self.assertEqual(wrapper.cosine_state, before["lr_schedule"])
        self.assertEqual(wrapper.scheduler.state_dict(), before["scheduler"])
        self.assertEqual([g["lr"] for g in wrapper.param_groups], [.005, .001])

    def test_missing_clock_or_protocol_changes_fail_closed(self):
        import copy
        _, wrapper = self._make()
        saved = copy.deepcopy(wrapper.state_dict())
        for total, floor in ((200, .1), (100, .2)):
            _, changed = self._make(total=total, floor=floor)
            with self.assertRaisesRegex(ValueError, "protocol mismatch"):
                changed.load_state_dict(saved)
        _, constant = self._make(cosine=False)
        with self.assertRaisesRegex(ValueError, "checkpoint uses cosine"):
            constant.load_state_dict(saved)
        with self.assertRaisesRegex(ValueError, "requires optimizer and scheduler"):
            wrapper.load_state_dict(saved["optimizer"])

    def test_invalid_cosine_settings_and_exhausted_migration_fail(self):
        for total, floor in ((10, .1), (True, .1), (20.5, .1), (100, 0), (100, float("nan"))):
            with self.subTest(total=total, floor=floor), self.assertRaises(ValueError):
                self._make(total=total, floor=floor)
        params, old = self._make(cosine=False)
        self._advance(params, old, 100)
        _, new = self._make()
        with self.assertRaisesRegex(ValueError, "must exceed"):
            new.load_state_dict(old.state_dict())

    def test_build_optimizer_wires_grouped_cosine_and_fallback(self):
        class Config(dict):
            __getattr__ = dict.__getitem__
        model = torch.nn.Module()
        model.Body = torch.nn.Linear(1, 1)
        model.unassigned = torch.nn.Parameter(torch.ones(1))
        opt_cfg = Config(learning_rate=.01, weight_decay=.001, Components={
            "Body": {"optimizer_group": "body", "learning_rate": .001,
                     "weight_decay": .001, "warm_up": False},
        })
        with mock.patch.object(dgpo_trainer, "global_config", SimpleNamespace(
            options=SimpleNamespace(Training=opt_cfg)
        )), mock.patch.object(dgpo_trainer, "_unwrap_core_evenet", return_value=model):
            wrapper = dgpo_trainer.build_optimizer(
                model, steps_per_epoch=10, warmup_steps=10, is_rank0=False,
                lr_schedule={"type": "cosine", "total_steps": 1500, "min_lr_ratio": .1},
            )
        self.assertEqual(wrapper.cosine_state["warmup_groups"], [False, True])
        self.assertEqual([g["lr"] for g in wrapper.param_groups], [.001, 0.])
        self._advance(list(model.parameters()), wrapper, 10)
        self.assertEqual([g["lr"] for g in wrapper.param_groups], [.001, .01])

    def test_legacy_cosine_checkpoint_defaults_to_decaying_all_groups(self):
        params, original = self._make()
        self._advance(params, original, 30)
        saved = copy.deepcopy(original.state_dict())
        del saved["lr_schedule"]["decay_groups"]
        _, resumed = self._make()
        resumed.load_state_dict(saved)
        self.assertEqual(resumed.cosine_state["decay_groups"], [True, True])
        self.assertEqual([g["lr"] for g in original.param_groups],
                         [g["lr"] for g in resumed.param_groups])


class TestExplicitResumeLearningRateChange(unittest.TestCase):
    def _make(self, *, higher_lr=False, all_groups=False):
        names = ["body", "generation", "projector", "angular_conditioning", "visible_conditioning"]
        params = [torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64)) for _ in names]
        rates = [5e-5 if higher_lr else 1e-6] * 3 + [1e-4] * 2
        optimizer = torch.optim.AdamW([
            {"params": [p], "lr": lr, "group_name": name}
            for p, lr, name in zip(params, rates, names, strict=True)
        ], weight_decay=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda _: 1.] * len(names))
        wrapper = dgpo_trainer._DgpoOptimizerWithSchedule(
            optimizer, scheduler, warmup_steps=1, warmup_groups=[False] * len(names),
            cosine_config={"total_steps": 15000, "min_lr_ratio": .1,
                           "groups": names if all_groups else names[-2:]},
        )
        return params, wrapper

    def test_override_keeps_moments_and_clock_but_uses_new_base_rates_and_decay_scope(self):
        params, original = self._make()
        advance = TestDgpoCosineResume()._advance
        advance(params, original, 326)
        saved = copy.deepcopy(original.state_dict())
        restored_params, restored = self._make(higher_lr=True, all_groups=True)
        restored.load_state_dict(saved, use_config_lr_schedule=True)
        self.assertEqual(restored.scheduler.last_epoch, 326)
        self.assertEqual(restored.scheduler._step_count, original.scheduler._step_count)
        self.assertEqual(restored.cosine_state["start_step"], 1)
        self.assertEqual(restored.cosine_state["decay_groups"], [True] * 5)
        self.assertEqual(restored.scheduler.base_lrs, [5e-5] * 3 + [1e-4] * 2)
        for index, (p, q, pg) in enumerate(zip(params, restored_params, restored.param_groups, strict=True)):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(restored.state[q][key], original.state[p][key], rtol=0, atol=0)
            base = 5e-5 if index < 3 else 1e-4
            self.assertEqual(pg["initial_lr"], base)
            self.assertAlmostEqual(pg["lr"], base * restored._cosine_factor(326, index))
            if index < 3:
                self.assertGreater(pg["lr"], original.param_groups[index]["lr"] * 49)
            else:
                self.assertEqual(pg["lr"], original.param_groups[index]["lr"])
        self.assertEqual(restored.scheduler.get_last_lr(), [pg["lr"] for pg in restored.param_groups])
        self.assertEqual(saved["scheduler"]["base_lrs"], [1e-6] * 3 + [1e-4] * 2)

    def test_repeated_resume_and_transaction_restore_do_not_restart_or_lose_override(self):
        advance = TestDgpoCosineResume()._advance
        params, original = self._make()
        advance(params, original, 30)
        params, active = self._make(higher_lr=True, all_groups=True)
        active.load_state_dict(copy.deepcopy(original.state_dict()), use_config_lr_schedule=True)
        advance(params, active, 20)
        saved = copy.deepcopy(active.state_dict())
        # Both subsequent startup resumes and exact transaction restores work.
        for override in (False, True):
            _, restored = self._make(higher_lr=True, all_groups=True)
            restored.load_state_dict(copy.deepcopy(saved), use_config_lr_schedule=override)
            self.assertEqual(restored.cosine_state, saved["lr_schedule"])
            self.assertEqual(restored.scheduler.state_dict(), saved["scheduler"])
            self.assertEqual([pg["lr"] for pg in restored.param_groups],
                             [pg["lr"] for pg in active.param_groups])
        # A rollback on the SAME object restores the old clock/rates exactly.
        advance(params, active, 10)
        active.load_state_dict(saved)
        self.assertEqual(active.scheduler.state_dict(), saved["scheduler"])
        # End at the original absolute horizon, not 15000 steps after resume.
        active.scheduler.last_epoch = 14999
        active._apply_cosine_lr()
        advance(params, active, 2)
        for pg in active.param_groups:
            expected = 1e-5 if "conditioning" in pg["group_name"] else 5e-6
            self.assertAlmostEqual(pg["lr"], expected)

    def test_default_resume_stays_strict_and_restores_saved_rates(self):
        _, original = self._make()
        saved = copy.deepcopy(original.state_dict())
        _, changed_scope = self._make(higher_lr=True, all_groups=True)
        with self.assertRaisesRegex(ValueError, "decay_groups"):
            changed_scope.load_state_dict(saved)
        _, same_scope = self._make(higher_lr=True)
        same_scope.load_state_dict(saved)
        self.assertEqual(same_scope.scheduler.base_lrs, [1e-6] * 3 + [1e-4] * 2)

    def test_override_requires_saved_clock_and_same_group_identity(self):
        _, original = self._make()
        for invalid in ("missing_clock", "different_groups", "bare_optimizer"):
            saved = copy.deepcopy(original.state_dict())
            if invalid == "missing_clock":
                del saved["scheduler"]["last_epoch"]
            elif invalid == "different_groups":
                saved["optimizer"]["param_groups"][0]["group_name"] = "wrong_group"
            else:
                saved = saved["optimizer"]
            _, resumed = self._make(higher_lr=True, all_groups=True)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "resume_use_config"):
                resumed.load_state_dict(saved, use_config_lr_schedule=True)


class TestPolicyEpochCosine(unittest.TestCase):
    def test_epoch_horizon_resolves_to_policy_updates_without_mutating_config(self):
        config = {"type": "cosine", "total_epochs": 1500, "total_steps": None, "min_lr_ratio": .1}
        for steps in (10, 13):
            with self.subTest(steps=steps):
                resolved = dgpo_trainer._resolve_dgpo_lr_schedule(config, steps_per_epoch=steps)
                self.assertEqual(resolved["total_steps"], 1500 * steps)
                self.assertEqual(resolved["total_epochs"], 1500)
        self.assertIsNone(config["total_steps"])

    def test_step_based_and_constant_schedules_remain_unchanged(self):
        for cfg in ({"type": "constant", "total_steps": 1500},
                    {"type": "cosine", "total_steps": 1500, "min_lr_ratio": .1}):
            self.assertEqual(dgpo_trainer._resolve_dgpo_lr_schedule(cfg, steps_per_epoch=10), cfg)

    def test_ambiguous_invalid_horizons_do_not_silently_shorten_schedule(self):
        for cfg in ({"type": "cosine", "total_epochs": 1500, "total_steps": 1500},
                    {"type": "constant", "total_epochs": 1500},
                    *({"type": "cosine", "total_epochs": v} for v in (0, -1, True, 1.5))):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                dgpo_trainer._resolve_dgpo_lr_schedule(cfg, steps_per_epoch=10)

    def test_resume_override_is_explicit_boolean_and_cosine_only(self):
        for cfg in ({"type": "cosine", "resume_use_config": "false"},
                    {"type": "constant", "resume_use_config": True}):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, "resume_use_config"):
                dgpo_trainer._resolve_dgpo_lr_schedule(cfg, steps_per_epoch=10)


class TestPolicyConditioningLearningRates(unittest.TestCase):
    def _make(self, rates=None, *, cosine=False, configure_model=None, schedule=None):
        class Config(dict):
            __getattr__ = dict.__getitem__

        model = torch.nn.Module()
        model.PET = torch.nn.Module()
        model.PET.pretrained = torch.nn.Linear(2, 2)
        model.PET.angular_conditioning = torch.nn.Linear(2, 2)
        model.TruthGeneration = torch.nn.Module()
        model.TruthGeneration.pretrained = torch.nn.Linear(2, 2)
        model.TruthGeneration.visible_conditioning = torch.nn.Sequential(
            torch.nn.Linear(2, 3), torch.nn.SiLU(), torch.nn.Linear(3, 2))
        model.unassigned = torch.nn.Parameter(torch.ones(1))
        model.double()
        if configure_model is not None:
            configure_model(model)
        options = Config(learning_rate=1e-6, weight_decay=.001, Components={
            "PET": {"optimizer_group": "body", "learning_rate": 1e-7,
                    "weight_decay": .002, "warm_up": False},
            "TruthGeneration": {"optimizer_group": "generation", "learning_rate": 1e-6,
                                "weight_decay": .003, "warm_up": False},
        })
        with mock.patch.object(dgpo_trainer, "global_config", SimpleNamespace(
                options=SimpleNamespace(Training=options))), \
                mock.patch.object(dgpo_trainer, "_unwrap_core_evenet", return_value=model):
            optimizer = dgpo_trainer.build_optimizer(
                model, steps_per_epoch=10, warmup_steps=2, is_rank0=False,
                conditioning_learning_rates=rates,
                lr_schedule=schedule if schedule is not None else (
                    {"type": "cosine", "total_steps": 20, "min_lr_ratio": .1} if cosine else None),
            )
        return model, optimizer

    def test_branch_groups_have_exclusive_ownership_and_real_adamw_updates(self):
        model, optimizer = self._make({"angular_conditioning": 1e-5, "visible_conditioning": 1e-5})
        groups = {g["group_name"]: g for g in optimizer.param_groups}
        expected = {"body": ("PET.pretrained", 1e-7, .002),
                    "generation": ("TruthGeneration.pretrained", 1e-6, .003),
                    "angular_conditioning": ("PET.angular_conditioning", 1e-5, .002),
                    "visible_conditioning": ("TruthGeneration.visible_conditioning", 1e-5, .003)}
        for name, (path, lr, wd) in expected.items():
            self.assertEqual(groups[name]["lr"], lr)
            self.assertEqual(groups[name]["weight_decay"], wd)
            self.assertEqual({id(p) for p in groups[name]["params"]},
                             {id(p) for p in model.get_submodule(path).parameters()})
        assigned = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {id(p) for p in model.parameters() if p.requires_grad})
        before = {id(p): p.detach().clone() for p in model.parameters()}
        sum(p.sum() for p in model.parameters()).backward()
        optimizer.step()
        for group in optimizer.param_groups:
            for p in group["params"]:
                old = before[id(p)]
                expected_update = group["lr"] * (1 / (1 + group["eps"]) + group["weight_decay"] * old)
                torch.testing.assert_close(old - p.detach(), expected_update, rtol=1e-7, atol=1e-14)

    def test_no_override_and_single_override_preserve_parent_behavior(self):
        model, optimizer = self._make()
        groups = {g["group_name"]: g for g in optimizer.param_groups}
        self.assertEqual(set(groups), {"body", "generation", "__fallback__"})
        self.assertEqual({id(p) for p in groups["body"]["params"]},
                         {id(p) for p in model.PET.parameters()})
        self.assertEqual({id(p) for p in groups["generation"]["params"]},
                         {id(p) for p in model.TruthGeneration.parameters()})
        model, optimizer = self._make({"visible_conditioning": 1e-5})
        groups = {g["group_name"]: g for g in optimizer.param_groups}
        self.assertNotIn("angular_conditioning", groups)
        self.assertEqual({id(p) for p in groups["body"]["params"]},
                         {id(p) for p in model.PET.parameters()})

    def test_scheduler_and_matching_checkpoint_resume_keep_branch_rates(self):
        rates = {"angular_conditioning": 1e-5, "visible_conditioning": 1e-5}
        model, optimizer = self._make(rates, cosine=True)
        for _ in range(9):
            optimizer.zero_grad()
            sum(p.square().sum() for p in model.parameters()).backward()
            optimizer.step()
            optimizer.scheduler_step()
        restored, resumed = self._make(dict(reversed(list(rates.items()))), cosine=True)
        restored.load_state_dict(model.state_dict())
        resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertEqual(resumed.scheduler.state_dict(), optimizer.scheduler.state_dict())
        self.assertEqual([g["group_name"] for g in resumed.param_groups],
                         [g["group_name"] for g in optimizer.param_groups])
        for _ in range(11):
            for m, opt in ((model, optimizer), (restored, resumed)):
                opt.zero_grad()
                sum(p.square().sum() for p in m.parameters()).backward()
                opt.step()
                opt.scheduler_step()
        for p, q in zip(model.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        groups = {g["group_name"]: g for g in resumed.param_groups}
        self.assertAlmostEqual(groups["angular_conditioning"]["lr"], 1e-6)
        self.assertAlmostEqual(groups["visible_conditioning"]["lr"], 1e-6)
        self.assertAlmostEqual(groups["body"]["lr"], 1e-8)

    def test_branch_only_cosine_1500_epochs_preserves_pretrained_rates(self):
        rates = {"angular_conditioning": 1e-4, "visible_conditioning": 1e-4}
        schedule = {"type": "cosine", "total_epochs": 1500, "total_steps": None,
                    "min_lr_ratio": .1, "groups": list(rates)}
        _, optimizer = self._make(rates, schedule=schedule)
        self.assertEqual(optimizer.cosine_state["total_steps"], 15000)
        for index, (group, base) in enumerate(zip(optimizer.param_groups, optimizer.scheduler.base_lrs, strict=True)):
            expected = base * .1 if group["group_name"] in rates else base
            self.assertAlmostEqual(base * optimizer._cosine_factor(15000, index), expected)
        # Resume mid-decay and reward refit must not restart the cosine clock.
        model, optimizer = self._make(rates, schedule={**schedule, "total_epochs": 2})
        advance = TestDgpoCosineResume()._advance
        advance(list(model.parameters()), optimizer, 9)
        saved = copy.deepcopy(optimizer.state_dict())
        restored, resumed = self._make(rates, schedule={**schedule, "total_epochs": 2})
        restored.load_state_dict(model.state_dict())
        resumed.load_state_dict(saved)
        dgpo_trainer._reset_optimizer_after_reward_install(
            resumed, cfg=SimpleNamespace(reset_optimizer_state_on_install=True,
                                         trust_reset_adam_first_moment=False), accepted=True)
        self.assertEqual(resumed.cosine_state, saved["lr_schedule"])
        advance(list(restored.parameters()), resumed, 11)
        groups = {g["group_name"]: g for g in resumed.param_groups}
        for name in rates:
            self.assertAlmostEqual(groups[name]["lr"], 1e-5)
        self.assertEqual(groups["body"]["lr"], 1e-7)
        self.assertEqual(groups["generation"]["lr"], 1e-6)

    def test_invalid_decay_groups_and_changed_resume_scope_rejected(self):
        rates = {"angular_conditioning": 1e-4, "visible_conditioning": 1e-4}
        schedule = {"type": "cosine", "total_epochs": 1500, "min_lr_ratio": .1}
        for groups in ([], "visible_conditioning", ["typo"],
                       ["visible_conditioning", "visible_conditioning"], [1]):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "cosine groups"):
                self._make(rates, schedule={**schedule, "groups": groups})
        _, all_groups = self._make(rates, schedule=schedule)
        _, branches_only = self._make(rates, schedule={**schedule, "groups": list(rates)})
        with self.assertRaisesRegex(ValueError, "decay_groups"):
            branches_only.load_state_dict(all_groups.state_dict())

    def test_invalid_or_inactive_branch_override_is_rejected(self):
        for rates in ({"typo": 1e-5}, [], {"visible_conditioning": True},
                      {"visible_conditioning": 0}, {"visible_conditioning": -1},
                      {"visible_conditioning": float("nan")},
                      {"visible_conditioning": float("inf")}):
            with self.subTest(rates=rates), self.assertRaisesRegex(ValueError, "conditioning_learning_rates"):
                self._make(rates)
        with self.assertRaisesRegex(ValueError, "requires enabled"):
            self._make({"visible_conditioning": 1e-5}, configure_model=lambda m:
                       setattr(m.TruthGeneration, "visible_conditioning", None))
        with self.assertRaisesRegex(ValueError, "requires trainable"):
            self._make({"visible_conditioning": 1e-5}, configure_model=lambda m:
                       m.TruthGeneration.visible_conditioning.requires_grad_(False))

    def test_named_rates_survive_compact_wandb_and_use_policy_clock(self):
        wandb = mock.Mock()
        dgpo_trainer._wandb_define_axes(wandb, critical=True)
        wandb.define_metric.assert_any_call("train/lr/scheduled/*", step_metric="global_step",
                                          step_sync=False, hidden=False)
        for name in ("angular_conditioning", "visible_conditioning", "body", "generation"):
            key = f"train/lr/scheduled/{name}"
            self.assertTrue(dgpo_trainer._wandb_critical_keep(key))
            self.assertTrue(dgpo_trainer._wandb_simplified_keep(key, 1e-5))


class TestPairedAdaptivePool(unittest.TestCase):
    def test_matched_periodic_audit_routes_to_diagnostic_with_training_pool(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(dgpo_trainer.dgpo_train_loop))
        callback = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                        and node.name == '_selection_blind_fixed_schedule_audit')
        cfg = SimpleNamespace(
            audit_fit={'training_population': 'omnifold_fold', 'training_fold': 1},
            crossfit_folds=2, seed=20260920, probe_seed=20260921,
            periodic_pair_features_enabled=True, visible_pair_rest_frame_enabled=False,
        )
        train_pool = SimpleNamespace(n_events=208355)
        evaluation_pool = SimpleNamespace(n_events=118992)
        shard, loader, policy, builder = object(), object(), object(), object()
        materialize = mock.Mock(return_value=train_pool)
        fit = mock.Mock(return_value={'raw_auc': .6, 'raw_auc_gap': .1,
                                     'raw_audit_saturated': 1., 'raw_audit_training_ready': 0.,
                                     'raw_audit_uses_omnifold_fold': 1., 'raw_audit_training_fold': 1.})
        namespace = dict(vars(dgpo_trainer))
        namespace.update(
            adaptive_cfg=cfg, omnifold_train_shard=shard, omnifold_train_loader_cfg=loader,
            model=policy, sampler=None, device=torch.device('cpu'), world_size=16, rank=0,
            num_ddim_val=20, probe_panel_seed=42, is_rank0=False, score_pool=evaluation_pool,
            omnifold_source=SimpleNamespace(model_builder=builder), epoch=4, global_step=50,
            adaptive_state=SimpleNamespace(probe_history=[{
                'fixed_schedule_diagnostic_only': 1., 'global_step': 0., 'raw_auc_gap': .01,
            }], reward_round_id=0),
            _materialize_adaptive_omnifold_pool=materialize, fit_raw_policy_audit=fit,
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[callback], type_ignores=[])),
                     '<matched-audit-production-callback>', 'exec'), namespace)
        with mock.patch.object(dgpo_trainer, "_materialize_adaptive_omnifold_pool", materialize):
            result = namespace[callback.name]()
        self.assertEqual(materialize.call_args.args, (shard, loader))
        self.assertIs(materialize.call_args.kwargs['model'], policy)
        self.assertEqual(materialize.call_args.kwargs['training_crossfit_fold'], (2, 1, 20260920))
        self.assertIsNone(materialize.call_args.kwargs['quota_events'])
        self.assertEqual(materialize.call_args.kwargs['world_size'], 16)
        self.assertIs(fit.call_args.kwargs['training_pool'], train_pool)
        self.assertIs(fit.call_args.kwargs['pool'], evaluation_pool)
        self.assertIsNone(fit.call_args.kwargs['warm_start_cache'])
        self.assertEqual(result['staleness/trigger_recalibration'], 0.)
        self.assertEqual(result['audit/trajectory_points'], 1.)
        self.assertEqual(result['audit/raw_auc_gap_change_from_step0'], 0.)

        # Exercise the actual branch predicate: intermediate five-epoch audits
        # must not fall through to the legacy small-pool raw plateau monitor.
        route = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and ast.unparse(node.test).startswith('diagnostic_raw_only or'))
        expression = compile(ast.Expression(route.test), '<matched-audit-route>', 'eval')
        self.assertTrue(eval(expression, {'diagnostic_raw_only': False, 'adaptive_cfg': cfg,
                                          'direct_fixed_schedule_refit': False}))
        self.assertFalse(eval(expression, {'diagnostic_raw_only': False, 'adaptive_cfg': cfg,
                                           'direct_fixed_schedule_refit': True}))
        cfg.audit_fit.clear()
        self.assertFalse(eval(expression, {'diagnostic_raw_only': False, 'adaptive_cfg': cfg,
                                           'direct_fixed_schedule_refit': False}))

    def test_raw_audit_generates_only_exact_reward_training_fold_and_refreshes(self):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
            _identity_crossfit_splits, _crossfit_repeat_seed, pack_event_inputs,
        )
        n = 400
        ids = torch.arange(n, dtype=torch.float32)
        batch = {
            'x': ids[:, None, None].expand(n, 2, 3).clone(),
            'x_mask': torch.ones(n, 2, 1),
            'conditions': ids[:, None, None].expand(n, 1, 2).clone(),
            'conditions_mask': torch.ones(n, 1),
            'x_invisible': ids[:, None, None].expand(n, 2, 2).clone(),
            'x_invisible_mask': torch.ones(n, 2),
        }
        packed, _ = pack_event_inputs(batch)
        seed = _crossfit_repeat_seed(20260920, 1)
        fit_idx, _ = _identity_crossfit_splits(packed, folds=2, seed=seed)[0]
        source = _OrderChangingShard(batch)
        fixed = dgpo_trainer._FixedEventInputShard(source)
        policy = _Policy(0.)
        observed = []

        def generate(model, data, _sampler, **kwargs):
            observed.append(set(data['x'][:, 0, 0].tolist()))
            return torch.randn(1, len(data['x']), 2, 2) + model.offset

        def collect(shard):
            return dgpo_trainer._materialize_adaptive_omnifold_pool(
                shard, {'batch_size': n}, model=policy, sampler=None,
                device=torch.device('cpu'), world_size=1, rank=0,
                quota_events=None, num_ddim_steps=20, seed=42,
                training_crossfit_fold=(2, 1, seed),
            )

        with mock.patch.object(dgpo_trainer, 'generate_neutrino_candidates', side_effect=generate):
            first = collect(fixed)
            policy.offset = 5.
            second = collect(fixed)
        expected = set(ids[fit_idx].tolist())
        self.assertEqual(observed, [expected, expected])
        self.assertEqual(set(first.truth[:, 0].tolist()), expected)
        self.assertEqual(source.calls, 1)
        torch.testing.assert_close(first.packed_event, second.packed_event, atol=0, rtol=0)
        torch.testing.assert_close(second.candidates - first.candidates, torch.full_like(first.candidates, 5.))

        # A rank whose local input contains only the excluded fold must still
        # participate in all-gather with correctly shaped empty tensors.
        other = torch.tensor([i for i in range(n) if i not in expected])
        empty_shard = _ChunkedShard({key: value[other] for key, value in batch.items()}, n)
        def gather(local, **kwargs):
            self.assertEqual(local['truth'].shape, (0, 4))
            self.assertEqual(local['candidates'].shape, (0, 1, 4))
            return {'packed_event': first.packed_event, 'truth': first.truth,
                    'candidates': first.candidates, 'packing_spec': first.packing_spec.to_dict()}
        with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.adaptive.gather_pool_across_ranks', side_effect=gather), \
                mock.patch.object(dgpo_trainer, 'generate_neutrino_candidates') as gen:
            collect(empty_shard)
            gen.assert_not_called()

    def test_fixed_inputs_reuse_identities_but_regenerate_current_policy(self):
        n = 40
        identities = torch.arange(n, dtype=torch.float32)
        batch = {
            'x': identities[:, None, None].expand(n, 2, 3).clone(),
            'x_mask': torch.ones(n, 2, 1),
            'conditions': identities[:, None, None].expand(n, 1, 2).clone(),
            'conditions_mask': torch.ones(n, 1),
            'x_invisible': torch.randn(n, 2, 2),
            'x_invisible_mask': torch.ones(n, 2),
        }
        source = _OrderChangingShard(batch)
        fixed = dgpo_trainer._FixedEventInputShard(source)
        policy = _Policy(0.)
        def generate(model, data, _sampler, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            return torch.randn(1, len(data['x']), 2, 2) + model.offset
        def collect():
            return dgpo_trainer._materialize_adaptive_omnifold_pool(
                fixed, {'batch_size': n}, model=policy, sampler=None,
                device=torch.device('cpu'), world_size=1, rank=0,
                quota_events=32, num_ddim_steps=20, seed=42,
            )
        with mock.patch.object(dgpo_trainer, 'generate_neutrino_candidates', side_effect=generate) as gen:
            first = collect()
            # Neither original tensors nor returned batch mutations can alter
            # the fixed panel. The latest policy, however, must still be used.
            batch['x'].fill_(-123.)
            returned = next(fixed.iter_torch_batches(batch_size=n))
            returned['x'].fill_(999.)
            policy.offset = 5.
            second = collect()
        self.assertEqual(source.calls, 1)
        self.assertEqual(gen.call_count, 2)
        torch.testing.assert_close(first.packed_event, second.packed_event, atol=0, rtol=0)
        torch.testing.assert_close(first.truth, second.truth, atol=0, rtol=0)
        torch.testing.assert_close(second.candidates - first.candidates, torch.full_like(first.candidates, 5.))
        with self.assertRaisesRegex(ValueError, 'unchanged generation loader'):
            list(fixed.iter_torch_batches(batch_size=20))

    def test_fixed_input_cache_retains_tail_for_validation(self):
        batch = {'x': torch.arange(35).reshape(35,1,1), 'truth': torch.arange(35)}
        fixed = dgpo_trainer._FixedEventInputShard(_ChunkedShard(batch, 16))
        for _ in range(2):
            chunks = list(fixed.iter_torch_batches(batch_size=16))
            self.assertEqual([len(chunk['x']) for chunk in chunks], [16,16,3])
            self.assertEqual(torch.cat([chunk['truth'] for chunk in chunks]).tolist(), list(range(35)))

    def test_current_and_reference_share_one_iterator_and_ddim_noise(self):
        n_events = 32
        identity = torch.arange(n_events, dtype=torch.float32)
        batch = {
            "x": identity[:, None, None].expand(n_events, 2, 3).clone(),
            "x_mask": torch.ones(n_events, 2, 1),
            "conditions": identity[:, None, None].expand(n_events, 1, 2).clone(),
            "conditions_mask": torch.ones(n_events, 1),
            "x_invisible": torch.randn(n_events, 2, 2),
            "x_invisible_mask": torch.ones(n_events, 2),
        }
        shard = _OrderChangingShard(batch)
        current_policy = _Policy(offset=1.0)
        reference_policy = _Policy(offset=11.0)
        reference_policy.eval()

        def _fake_generate(model, input_batch, _sampler, **kwargs):
            self.assertEqual(kwargs["K"], 1)
            count = int(input_batch["x"].shape[0])
            noise = torch.randn(1, count, 2, 2, device=kwargs["device"])
            return noise + float(model.offset)

        with mock.patch.object(
            dgpo_trainer,
            "generate_neutrino_candidates",
            side_effect=_fake_generate,
        ):
            current_pool, trust_current_pool, reference_pool = (
                dgpo_trainer._materialize_adaptive_omnifold_pool(
                    shard,
                    {"batch_size": n_events},
                    model=current_policy,
                    paired_reference_model=reference_policy,
                    sampler=None,
                    device=torch.device("cpu"),
                    world_size=1,
                    rank=0,
                    quota_events=n_events,
                    paired_reference_quota_events=30,
                    num_ddim_steps=20,
                    seed=1234,
                )
            )

        self.assertEqual(shard.calls, 1)
        self.assertTrue(current_policy.training)
        self.assertFalse(reference_policy.training)
        self.assertTrue(
            torch.equal(
                trust_current_pool.packed_event,
                reference_pool.packed_event,
            )
        )
        self.assertEqual(current_pool.n_events, n_events)
        self.assertEqual(trust_current_pool.n_events, 30)
        torch.testing.assert_close(
            reference_pool.candidates - trust_current_pool.candidates,
            torch.full_like(trust_current_pool.candidates, 10.0),
        )
        paired_classifier_pool = build_reference_trust_pool(
            trust_current_pool,
            reference_pool,
        )
        torch.testing.assert_close(
            paired_classifier_pool.truth,
            reference_pool.candidates[:, 0],
        )

    def test_reference_generation_stops_at_its_smaller_quota(self):
        n_events = 96
        identity = torch.arange(n_events, dtype=torch.float32)
        batch = {
            "x": identity[:, None, None].expand(n_events, 2, 3).clone(),
            "x_mask": torch.ones(n_events, 2, 1),
            "conditions": identity[:, None, None].expand(n_events, 1, 2).clone(),
            "conditions_mask": torch.ones(n_events, 1),
            "x_invisible": torch.randn(n_events, 2, 2),
            "x_invisible_mask": torch.ones(n_events, 2),
        }
        shard = _ChunkedShard(batch, chunk_size=32)
        current_policy = _Policy(offset=1.0)
        reference_policy = _Policy(offset=11.0)
        calls = {"current": 0, "reference": 0}

        def _fake_generate(model, input_batch, _sampler, **kwargs):
            key = "reference" if model is reference_policy else "current"
            calls[key] += 1
            count = int(input_batch["x"].shape[0])
            noise = torch.randn(1, count, 2, 2, device=kwargs["device"])
            return noise + float(model.offset)

        with mock.patch.object(
            dgpo_trainer,
            "generate_neutrino_candidates",
            side_effect=_fake_generate,
        ):
            current_pool, trust_current_pool, reference_pool = (
                dgpo_trainer._materialize_adaptive_omnifold_pool(
                    shard,
                    {"batch_size": 32},
                    model=current_policy,
                    paired_reference_model=reference_policy,
                    sampler=None,
                    device=torch.device("cpu"),
                    world_size=1,
                    rank=0,
                    quota_events=96,
                    paired_reference_quota_events=32,
                    num_ddim_steps=20,
                    seed=1234,
                )
            )

        self.assertEqual(calls, {"current": 3, "reference": 1})
        self.assertEqual(current_pool.n_events, 96)
        self.assertEqual(trust_current_pool.n_events, 32)
        self.assertEqual(reference_pool.n_events, 32)
        torch.testing.assert_close(
            reference_pool.candidates - trust_current_pool.candidates,
            torch.full_like(trust_current_pool.candidates, 10.0),
        )


class TestRawBestPolicyRollback(unittest.TestCase):
    def test_global_resolution_checks_exact_step_and_source_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "old"
            source.mkdir()
            target = source / "dgpo-epoch=1-next_ep=1-step=15.ckpt"
            payload = {"state_dict": {"weight": torch.ones(1)}, "epoch": 1,
                       "global_step": 15, "dgpo_next_epoch": 1,
                       "dgpo_adaptive_omnifold_state": {"probe_history": [
                           {"raw_auc_gap": .1, "raw_audit_saturated": 1., "epoch": 1,
                            "global_step": 15, "checkpoint_next_epoch": 1}]}}
            torch.save(payload, target)
            state = SimpleNamespace(raw_best_epoch=1, raw_best_global_step=15,
                                    raw_best_next_epoch=1, raw_best_auc_gap=.1, raw_best_checkpoint="")
            resolved = dgpo_trainer._resolve_raw_best_policy_checkpoint(
                state, root / "new", global_scope=True, source_dirs=[source])
            self.assertEqual(resolved, target.resolve())
            payload["global_step"] = 20
            torch.save(payload, target)
            with self.assertRaisesRegex(ValueError, "metadata/raw AUC mismatch"):
                dgpo_trainer._resolve_raw_best_policy_checkpoint(state, root, global_scope=True)
            state.raw_best_checkpoint = ""
            with self.assertRaises(FileNotFoundError):
                dgpo_trainer._resolve_raw_best_policy_checkpoint(state, root, global_scope=True)

    def test_confirmation_restores_reference_even_on_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = torch.nn.Linear(1, 1)
            reference = torch.nn.Linear(1, 1)
            best = torch.nn.Linear(1, 1)
            checkpoint = Path(tmp) / "best.ckpt"
            torch.save({"state_dict": best.state_dict()}, checkpoint)
            reference_before = {k: v.clone() for k, v in reference.state_dict().items()}
            candidate_before = {k: v.clone() for k, v in candidate.state_dict().items()}
            calls = []
            def materialize(current, incumbent):
                torch.testing.assert_close(incumbent.weight, best.weight)
                return "full", "candidate", "incumbent"
            def judge(pool, phase):
                calls.append((pool, phase))
                torch.testing.assert_close(reference.weight, reference_before["weight"])
                return {"raw_auc_gap": .1 if pool == "candidate" else .12, "raw_audit_saturated": 1.}
            accepted, d = dgpo_trainer._confirm_global_raw_candidate(
                model=candidate, comparison_model=reference, best_checkpoint=checkpoint,
                materialize_pair=materialize, fit_judge=judge, min_delta=.001)
            self.assertTrue(accepted)
            self.assertEqual(len(calls), 2)
            self.assertEqual(d["staleness/global_best/confirmation_valid"], 1.)
            for k, v in candidate.state_dict().items():
                torch.testing.assert_close(v, candidate_before[k])
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                dgpo_trainer._confirm_global_raw_candidate(
                    model=candidate, comparison_model=reference, best_checkpoint=checkpoint,
                    materialize_pair=mock.Mock(side_effect=RuntimeError("generation failed")),
                    fit_judge=judge, min_delta=.001)
            for k, v in reference.state_dict().items():
                torch.testing.assert_close(v, reference_before[k])

    def test_confirmation_warm_starts_two_isolated_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = torch.nn.Linear(1, 1)
            reference = torch.nn.Linear(1, 1)
            checkpoint = Path(tmp) / "best.ckpt"
            torch.save({"state_dict": reference.state_dict()}, checkpoint)
            shared = {"protocol": {"seed": 42}, "state": {"weight": torch.tensor([3.])}}
            seen = []
            def judge(pool, phase, *, warm_start_cache):
                self.assertIsNot(warm_start_cache, shared)
                torch.testing.assert_close(warm_start_cache["state"]["weight"], torch.tensor([3.]))
                seen.append(phase)
                warm_start_cache["state"]["weight"].add_(100)
                warm_start_cache.clear()
                return {"raw_auc_gap": .1 if pool == "candidate" else .12, "raw_audit_saturated": 1.}
            accepted, _ = dgpo_trainer._confirm_global_raw_candidate(
                model=candidate, comparison_model=reference, best_checkpoint=checkpoint,
                materialize_pair=lambda *args: ("full", "candidate", "incumbent"),
                fit_judge=judge, min_delta=.001, initial_judge_cache=shared)
            self.assertTrue(accepted)
            self.assertEqual(seen, ["global_best_candidate", "global_best_incumbent"])
            torch.testing.assert_close(shared["state"]["weight"], torch.tensor([3.]))

    def test_global_rewind_keeps_cosine_clock_and_moments_until_install(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
            wrapped = dgpo_trainer._DgpoOptimizerWithSchedule(
                optimizer, scheduler, cosine_config={"total_steps": 100, "min_lr_ratio": .1},
                warmup_steps=1,
            )
            for _ in range(5):
                model(torch.ones(1, 1)).sum().backward()
                wrapped.step()
                wrapped.scheduler_step()
                wrapped.zero_grad()
            best = torch.nn.Linear(1, 1)
            path = Path(tmp) / "best.ckpt"
            torch.save({"state_dict": best.state_dict()}, path)
            saved = copy.deepcopy(wrapped.state_dict())
            dgpo_trainer._rewind_policy_for_raw_best_refit(model, wrapped, path, clear_optimizer=False)
            self.assertEqual(wrapped.scheduler.state_dict(), saved["scheduler"])
            self.assertEqual(wrapped.cosine_state, saved["lr_schedule"])
            self.assertEqual(wrapped.param_groups[0]["lr"], saved["optimizer"]["param_groups"][0]["lr"])
            self.assertTrue(wrapped.state)
            dgpo_trainer._reset_optimizer_after_reward_install(
                wrapped, cfg=SimpleNamespace(reset_optimizer_state_on_install=True,
                                             trust_reset_adam_first_moment=False), accepted=True)
            self.assertFalse(wrapped.state)
            self.assertEqual(wrapped.scheduler.state_dict(), saved["scheduler"])
            self.assertEqual(wrapped.cosine_state, saved["lr_schedule"])

    def test_resolves_legacy_best_epoch_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_root = Path(tmp)
            expected = save_root / "dgpo-epoch=3-next_ep=4-step=40.ckpt"
            expected.touch()
            state = SimpleNamespace(
                raw_best_checkpoint="",
                raw_best_epoch=3,
            )

            resolved = dgpo_trainer._resolve_raw_best_policy_checkpoint(
                state,
                save_root,
            )

            self.assertEqual(resolved, expected.resolve())
            self.assertEqual(state.raw_best_checkpoint, str(expected.resolve()))

    def test_rewind_restores_policy_and_clears_optimizer_and_ema(self):
        class _EMA:
            def __init__(self):
                self.calls = []

            def update(self, model, *, decay_):
                self.calls.append(
                    (
                        float(decay_),
                        {
                            key: value.detach().clone()
                            for key, value in model.state_dict().items()
                        },
                    )
                )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "best.ckpt"
            best_model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                best_model.weight.fill_(2.0)
                best_model.bias.fill_(3.0)
            torch.save({"state_dict": best_model.state_dict()}, checkpoint_path)

            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
            model(torch.ones(2, 2)).sum().backward()
            optimizer.step()
            self.assertGreater(len(optimizer.state), 0)
            ema_save = _EMA()
            ema_rollout = _EMA()

            loaded, cleared, ema_restored = (
                dgpo_trainer._rewind_policy_for_raw_best_refit(
                    model,
                    optimizer,
                    checkpoint_path,
                    ema_save=ema_save,
                    ema_rollout=ema_rollout,
                )
            )

            self.assertEqual(loaded, len(best_model.state_dict()))
            self.assertGreater(cleared, 0)
            self.assertEqual(len(optimizer.state), 0)
            self.assertTrue(ema_restored)
            torch.testing.assert_close(model.weight, best_model.weight)
            torch.testing.assert_close(model.bias, best_model.bias)
            for ema in (ema_save, ema_rollout):
                self.assertEqual(len(ema.calls), 1)
                self.assertEqual(ema.calls[0][0], 0.0)
                torch.testing.assert_close(
                    ema.calls[0][1]["weight"],
                    best_model.weight,
                )


if __name__ == "__main__":
    unittest.main()
