"""Focused regression tests for DGPO trainer orchestration helpers."""

from __future__ import annotations

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


class TestBestPointNewExperiment(unittest.TestCase):
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


class TestPairedAdaptivePool(unittest.TestCase):
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
