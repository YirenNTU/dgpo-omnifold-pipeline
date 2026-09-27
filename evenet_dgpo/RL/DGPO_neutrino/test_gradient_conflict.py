"""CPU regressions for statistical direction, estimator parity and read-only probes."""
from __future__ import annotations

import ast
import copy
import inspect
import math
import random
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import torch

from . import gradient_conflict as gc
from . import dgpo_trainer as trainer
from .omnifold_ztautau.adaptive import (
    AdaptiveOmniFoldPool, AdaptiveOmniFoldState, resolve_adaptive_config,
    update_raw_plateau_controller, raw_staleness_patience,
)
from .omnifold_ztautau.evenet_ratio import EventPackingSpec, _identity_crossfit_splits
from .omnifold_ztautau.test_rest_frame import ablation_payload


class Policy(torch.nn.Module):
    invisible_input_dim = 2
    invisible_padding = 0

    def __init__(self, value=.1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        self.unused = torch.nn.Parameter(torch.tensor(3.))
        self.dropout = torch.nn.Dropout(.25)
        self.register_buffer("counter", torch.zeros(()))

    def invisible_normalizer(self, *, x, mask):
        return x * mask

    def predict_diffusion_vector(self, *, noise_x, cond_x, time, mode, noise_mask):
        return self.weight * noise_x + 0.01 * time[:, None, None]

    def forward(self, x):
        return self.weight * x + self.unused * 0


class Judge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(-1.))

    def forward(self, packed, candidate):
        return self.scale * candidate.sum(-1)


def fixture():
    n = 512
    pool = AdaptiveOmniFoldPool(
        packed_event=torch.arange(n, dtype=torch.float32)[:, None] / n,
        truth=torch.ones(n, 4), candidates=torch.zeros(n, 1, 4),
        packing_spec=EventPackingSpec({"x": (1,)}),
        policy_noise_mask=torch.ones(n, 2, dtype=torch.bool),
    )
    judge = Judge()
    cache = {"protocol": {"schema": "raw-monitor-condition-60-20-20-v1", "seed": 42, "folds": 5},
             "state": copy.deepcopy(judge.state_dict())}
    cfg = gc.GradientConflictConfig(enabled=True, events_per_block=4, event_microbatch_size=2)
    policy, ref = Policy(), Policy(.05)

    def generate(core, batch, sampler, **kwargs):
        assert not torch.is_grad_enabled()
        return torch.randn(kwargs["K"], len(batch["x"]), 2, 2) + core.weight

    return dict(cfg=cfg, model=policy, core=policy, reference=ref, pool=pool,
                monitor_factory=Judge, monitor_cache=cache,
                reward_compute=lambda candidates, batch: -candidates.sum((-1, -2)),
                generate=generate, evaluate=trainer.policy_evaluation_step, sampler=None,
                device=torch.device("cpu"), dtype=torch.float32, rank=0, world_size=1,
                K=8, beta=1., num_ddim_steps=20, rollout_parallel_chains=4, num_train_timesteps=2,
                t_min=0., t_max=.7, advantage_estimator="leave_one_out_unscaled", adv_clip_max=None,
                trust_coefficient=1., global_step=5, reward_round_id=2, raw_auc=.7, raw_ready=True)


class TestStatistics(unittest.TestCase):
    def metrics(self, blocks):
        vectors = [torch.tensor(v, dtype=torch.float32) for block in blocks for v in block]
        return gc.summarize_gradient_gram(gc.gradient_gram(vectors), gc.GradientConflictConfig())

    def test_aligned_and_opposed(self):
        metrics = self.metrics([[[1., 2.], [-2., -4.], [3., 6.]]] * 8)
        p = "gradient_conflict/"
        self.assertAlmostEqual(metrics[p + "omnifold_staleness/cosine"], -1)
        self.assertEqual(metrics[p + "omnifold_staleness/conflict"], 1)
        self.assertEqual(metrics[p + "omnifold_trust/alignment"], 1)
        self.assertEqual(metrics[p + "omnifold_staleness/conclusive"], 1)

    def test_zero_gradient_is_not_conflict(self):
        metrics = self.metrics([[[1., 2.], [0., 0.], [0., 0.]]] * 8)
        self.assertTrue(math.isnan(metrics["gradient_conflict/omnifold_staleness/cosine"]))
        self.assertEqual(metrics["gradient_conflict/omnifold_staleness/conclusive"], 0)
        self.assertEqual(metrics["gradient_conflict/omnifold_staleness/conflict"], 0)

    def test_total_gradient_projection_and_cancellation(self):
        for trust, expected in ((0., 1.), (-.9, .1), (-1., 0.), (-2., -1.)):
            metrics = self.metrics([[[1., 0.], [1., 0.], [trust, 0.]]] * 8)
            self.assertAlmostEqual(metrics["gradient_conflict/total_on_omnifold/projection_ratio"], expected, places=6)
            self.assertEqual(metrics["gradient_conflict/total_on_omnifold/opposed"], float(expected < 0))

    def test_common_noise_is_not_evidence_of_alignment(self):
        # Perfect paired alignment but orthogonal independent block directions.
        blocks = [[np.eye(8)[i], np.eye(8)[i], np.eye(8)[i]] for i in range(8)]
        metrics = self.metrics(blocks)
        self.assertAlmostEqual(metrics["gradient_conflict/omnifold_staleness/cosine"], 1)
        self.assertAlmostEqual(metrics["gradient_conflict/omnifold_staleness/cross_dot"], 0)
        self.assertEqual(metrics["gradient_conflict/omnifold_staleness/conclusive"], 0)

    def test_cosine_of_mean_not_mean_cosine(self):
        blocks = [[[10., 0.], [10., 0.], [0., 0.]]] * 4 + [[[0., 1.], [0., -1.], [0., 0.]]] * 4
        metrics = self.metrics(blocks)
        self.assertAlmostEqual(metrics["gradient_conflict/omnifold_staleness/cosine"], 99 / 101)

    def test_jackknife_matches_explicit_deletion(self):
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=(8, 5)), rng.normal(size=(8, 5))
        matrix = a @ b.T
        estimate, lo, hi = gc._cross_interval(matrix, .95)
        brute = lambda x: (x.sum() - np.trace(x)) / (len(x) * (len(x) - 1))
        self.assertAlmostEqual(estimate, brute(matrix))
        deleted = np.array([brute(np.delete(np.delete(matrix, i, 0), i, 1)) for i in range(8)])
        se = np.sqrt(7 / 8 * np.sum((deleted - deleted.mean()) ** 2))
        self.assertAlmostEqual(hi - estimate, gc.student_t.ppf(.975, 7) * se)
        self.assertAlmostEqual(estimate - lo, hi - estimate)

    def test_config_validation_and_default_off(self):
        self.assertFalse(gc.resolve_gradient_conflict_config(None).enabled)
        for payload in ({"blocks": 2}, {"blocks": 9}, {"enabled": "true"}, {"confidence": 1},
                        {"events_per_block": 0}, {"norm_floor": float("nan")}, {"typo": 1}):
            with self.assertRaises(ValueError):
                gc.resolve_gradient_conflict_config(payload)


class TestProbe(unittest.TestCase):
    def test_current_and_legacy_identity_stable_monitor_protocols(self):
        for schema in (
            "raw-monitor-condition-60-20-20-v1",
            "raw-monitor-condition-80-20-v1",
            "raw-monitor-condition-split-v1",
        ):
            kwargs = fixture()
            kwargs["monitor_cache"]["protocol"]["schema"] = schema
            self.assertEqual(gc.probe_gradients(**kwargs)["gradient_conflict/ran"], 1.)

        kwargs = fixture()
        kwargs["monitor_cache"]["protocol"]["schema"] = "random-row-split-v1"
        with self.assertRaisesRegex(ValueError, "identity-stable raw monitor split"):
            gc.probe_gradients(**kwargs)

    def test_trainer_lifecycle_uses_frozen_judge_and_separate_series(self):
        # Execute the real nested wiring with tiny CPU policy/judge fixtures.
        tree = ast.parse(inspect.getsource(trainer.dgpo_train_loop))
        function = copy.deepcopy(next(n for n in ast.walk(tree)
                                      if isinstance(n, ast.FunctionDef) and n.name == "_measure_gradient_phase"))
        function.body = [n for n in function.body if not isinstance(n, ast.Nonlocal)]
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        kw = fixture()
        state = AdaptiveOmniFoldState(reward_round_id=2, policy_warmup_round_id=2,
                                     raw_monitor_state=copy.deepcopy(kw["monitor_cache"]))
        environment = dict(kw)
        environment.update({
            "copy": copy, "adaptive_state": state,
            "adaptive_cfg": SimpleNamespace(audit_fit={}, raw_rollback_to_best_on_plateau=False),
            "gradient_conflict_cfg": replace(kw["cfg"], monitor_refit_lifecycle=True),
            "gradient_cache": {}, "global_step": 10, "epoch": 0, "is_rank0": False,
            "current_raw_monitor_cache": state.raw_monitor_state,
            "_unwrap_core_evenet": lambda model: model,
            "round_ref_model": kw["reference"], "raw_probe": {
                "raw_auc": .7,
                "raw_audit_training_ready": 1.,
                "raw_audit_saturated": 1.,
            },
            "omnifold_source": SimpleNamespace(model_builder=object()),
            "reward_agg": SimpleNamespace(compute=lambda candidates, batch: (kw["reward_compute"](candidates, batch), {})),
            "generate_neutrino_candidates": kw["generate"], "policy_evaluation_step": kw["evaluate"],
            "num_ddim": kw["num_ddim_steps"], "policy_eval_t_min_cfg": kw["t_min"],
            "policy_eval_t_max_cfg": kw["t_max"], "adv_clip_max_cfg": None,
            "reference_trust_coefficient": 1.,
        })
        exec(compile(module, "<gradient-lifecycle-wiring>", "exec"), environment)
        measure = environment["_measure_gradient_phase"]
        emitted = []
        original = gc.phase_metrics
        def capture(metrics, phase):
            emitted.append((phase, dict(metrics)))
            return original(metrics, phase)
        # The extracted production closure imports through the runtime ``RL``
        # package path. Patch that exact module so the test does not depend on
        # whether pytest collected this file as ``DGPO_neutrino`` or
        # ``RL.DGPO_neutrino``.
        with mock.patch("RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio.peft_bank_factory", return_value=Judge), mock.patch("RL.DGPO_neutrino.gradient_conflict.phase_metrics", side_effect=capture):
            measure("pre_refit", kw["pool"])
            kw["reference"].load_state_dict(kw["core"].state_dict())
            state.reward_round_id = state.policy_warmup_round_id = 3
            measure("post_install", kw["pool"])
            self.assertEqual(emitted[-1][1]["gradient_conflict/trust/norm"], 0.)
            self.assertEqual(state.gradient_post_install_probe_round_id, 3)
            saved = copy.deepcopy(state.gradient_lifecycle_monitor_state)
            state.raw_monitor_state["state"]["scale"].fill_(-9.)
            torch.testing.assert_close(saved["state"]["scale"], state.gradient_lifecycle_monitor_state["state"]["scale"])
            environment["global_step"] = 20
            state.policy_warmup_completed_updates = 10
            measure("post_warmup", kw["pool"])
        self.assertEqual(emitted[-1][1]["gradient_conflict/judge_reused_from_install"], 1.)
        self.assertEqual(emitted[-1][1]["gradient_conflict/judge_fit_global_step"], 10)
        self.assertEqual(emitted[-1][1]["gradient_conflict/raw_auc_is_current"], 0.)
        self.assertEqual(state.gradient_warmup_probe_round_id, 3)
        self.assertEqual(state.gradient_lifecycle_monitor_state, {})
        self.assertEqual(len({m["gradient_conflict/panel_sha256"] for _, m in emitted}), 1)

    def test_recentered_reference_zero_trust_and_common_panel(self):
        kwargs = fixture()
        before = gc.probe_gradients(**kwargs)
        self.assertGreater(before["gradient_conflict/trust/norm"], 0)
        kwargs["reference"].load_state_dict(kwargs["core"].state_dict())
        after = gc.probe_gradients(**kwargs)
        self.assertEqual(after["gradient_conflict/trust/norm"], 0.)
        self.assertEqual(after["gradient_conflict/trust/reliable"], 0.)
        self.assertEqual(after["gradient_conflict/panel_sha256"], before["gradient_conflict/panel_sha256"])
        self.assertAlmostEqual(after["gradient_conflict/total_on_omnifold/projection_ratio"], 1.)

    def test_global_event_and_mask_weighting_with_unequal_rank_counts(self):
        kwargs = fixture()
        kwargs["pool"].policy_noise_mask[::2, 1] = False
        # Deterministic per-event draws make partition/microbatch parity exact.
        def generate(core, batch, sampler, **kw):
            return (batch["x"][:, 0][None, :, None, None]
                    + torch.arange(kw["K"])[:, None, None, None] * .1).expand(-1, -1, 2, 2)
        def evaluate(core, reference, batch, candidates, **kw):
            b = len(batch["x"])
            return trainer.policy_evaluation_step(core, reference, batch, candidates,
                t=torch.full((b,), .2), eps_rep=torch.full((kw["K"] * b, 2, 2), .3), **kw)
        kwargs.update(generate=generate, evaluate=evaluate)
        captured = []
        real_gram = gc.gradient_gram
        def gram(vectors):
            captured.extend(v.clone() for v in vectors)
            return real_gram(vectors)
        with mock.patch.object(gc, "gradient_gram", side_effect=gram):
            gc.probe_gradients(**kwargs)
        serial = torch.stack(captured)
        shards = []
        for rank in range(3):  # Four global events => rank counts 2,1,1.
            local = []
            with mock.patch.object(gc.dist, "all_reduce", side_effect=lambda v, op: local.append(v.clone())):
                gc.probe_gradients(**{**kwargs, "rank": rank, "world_size": 3})
            shards.append(torch.stack(local))
        torch.testing.assert_close(sum(shards), serial, atol=1e-7, rtol=1e-5)

    def test_actual_policy_evaluation_and_read_only_reproducibility(self):
        kwargs = fixture()
        model = kwargs["core"]
        model.train()
        model.dropout.eval()  # Preserve mixed modes as well as top-level mode.
        model.weight.grad = torch.tensor(8.)
        model.unused.grad = torch.tensor(9.)
        state = copy.deepcopy(model.state_dict())
        ref_state = copy.deepcopy(kwargs["reference"].state_dict())
        cache = copy.deepcopy(kwargs["monitor_cache"])
        rng = torch.get_rng_state().clone()
        np_rng, py_rng = np.random.get_state(), random.getstate()
        models = []
        def factory():
            judge = Judge()
            judge.scale.data.fill_(123.)  # Cached fitted value must replace this.
            models.append(judge)
            return judge
        kwargs["monitor_factory"] = factory
        first = gc.probe_gradients(**kwargs)
        second = gc.probe_gradients(**kwargs)
        self.assertEqual(first["gradient_conflict/ran"], 1)
        self.assertAlmostEqual(first["gradient_conflict/omnifold_staleness/cosine"], 1., places=5)
        for key in first:
            if key.endswith("/seconds") or (isinstance(first[key], float) and math.isnan(first[key])):
                continue
            self.assertEqual(first[key], second[key], key)
        for key, value in state.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        for key, value in ref_state.items():
            torch.testing.assert_close(kwargs["reference"].state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(model.weight.grad, torch.tensor(8.))
        torch.testing.assert_close(model.unused.grad, torch.tensor(9.))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        np.testing.assert_equal(np.random.get_state(), np_rng)
        self.assertEqual(random.getstate(), py_rng)
        torch.testing.assert_close(kwargs["monitor_cache"]["state"]["scale"], cache["state"]["scale"])
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in models for p in m.parameters()))

    def test_exact_masks_and_heldout_unique_events(self):
        kwargs = fixture()
        seen, masks = [], []
        kwargs["pool"].policy_noise_mask[::2, 1] = False
        generate = kwargs["generate"]
        def capture(core, batch, sampler, **kw):
            seen.extend(batch["x"][:, 0].tolist())
            masks.extend(batch["x_invisible_mask"].tolist())
            return generate(core, batch, sampler, **kw)
        kwargs["generate"] = capture
        gc.probe_gradients(**kwargs)
        train, val = _identity_crossfit_splits(kwargs["pool"].identity_inputs, folds=5, seed=42)[0]
        rows = [round(v * 512) for v in seen]
        self.assertEqual(len(set(rows)), 32)
        self.assertTrue(set(rows).issubset(set(val.tolist())))
        self.assertFalse(set(rows) & set(train.tolist()))
        for row, mask in zip(rows, masks):
            self.assertEqual(mask, kwargs["pool"].policy_noise_mask[row].tolist())

    def test_missing_or_unready_skips_no_forward(self):
        kwargs = fixture()
        kwargs["raw_ready"] = False
        kwargs["generate"] = mock.Mock(side_effect=AssertionError)
        self.assertEqual(gc.probe_gradients(**kwargs)["gradient_conflict/skipped_unready"], 1)
        kwargs["raw_ready"] = True
        kwargs["pool"] = replace(kwargs["pool"], policy_noise_mask=None)
        self.assertEqual(gc.probe_gradients(**kwargs)["gradient_conflict/skipped_missing_masks"], 1)

    def test_nonfinite_skip_and_exception_restore_rng_modes(self):
        kwargs = fixture()
        kwargs["reward_compute"] = lambda c, b: torch.full(c.shape[:2], float("nan"))
        rng = torch.get_rng_state().clone()
        result = gc.probe_gradients(**kwargs)
        self.assertEqual(result["gradient_conflict/skipped_nonfinite"], 1)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertTrue(kwargs["core"].training)
        kwargs = fixture()
        kwargs["evaluate"] = mock.Mock(side_effect=RuntimeError("injected error"))
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "injected"):
            gc.probe_gradients(**kwargs)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        self.assertTrue(kwargs["core"].training)

    def test_no_optimizer_and_before_rollback_integration(self):
        source = inspect.getsource(gc.probe_gradients)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer", source.replace("AdamW-preconditioned", ""))
        tree = ast.parse(inspect.getsource(trainer.dgpo_train_loop))
        cycle = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_adaptive_cycle")
        calls = {n.func.id: n.lineno for n in ast.walk(cycle) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in ("probe_gradients", "update_raw_plateau_controller")}
        self.assertLess(calls["probe_gradients"], calls["update_raw_plateau_controller"])


class TestWandbAndConfig(unittest.TestCase):
    def test_train_payload_preserves_gradient_diagnostic_namespaces(self):
        raw = {
            "gradient_transfer/h4/norm": 1.25,
            "gradient_conflict/omnifold_staleness/cosine": 0.75,
        }
        payload = trainer._wandb_train_payload(raw)
        self.assertEqual(payload, raw)
        self.assertEqual(
            {key: value for key, value in payload.items()
             if trainer._wandb_simplified_keep(key, value)},
            raw,
        )

    def test_gradient_transfer_survives_complete_wandb_logging_pipeline(self):
        raw = {
            "gradient_transfer/ran": 1.0,
            "gradient_transfer/h4/norm": 1.25,
            "gradient_transfer/adamw_descent_on_h4/cosine": 0.75,
            "gradient_transfer/reconstruction_actual/relative_error": 1.0e-7,
            "unrelated_internal_value": 99.0,
        }
        fake_wandb = mock.Mock()
        trainer._wandb_reset_step_tracker()
        with (
            mock.patch.object(trainer, "_wandb_critical_enabled", return_value=False),
            mock.patch.object(trainer, "_wandb_simplified_enabled", return_value=True),
        ):
            trainer._wandb_log_step(
                fake_wandb,
                trainer._wandb_train_payload(raw),
                step=5,
            )
        logged = fake_wandb.log.call_args.args[0]
        self.assertEqual(logged["global_step"], 5)
        for key in raw:
            if key.startswith("gradient_transfer/"):
                self.assertEqual(logged[key], raw[key])
        self.assertNotIn("train/unrelated_internal_value", logged)

    def test_forward_yaml_and_controller_preserve_global_best_without_rollback(self):
        import yaml
        root = Path(__file__).resolve().parents[3] / "config"
        base = ablation_payload()
        payload = yaml.safe_load((root / "dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit.yaml").read_text())
        cfg = resolve_adaptive_config(payload["dgpo"])
        self.assertFalse(cfg.raw_rollback_to_best_on_plateau)
        self.assertEqual(cfg.raw_best_scope, "global")
        self.assertEqual(cfg.raw_global_max_failed_rounds, 0)
        self.assertEqual([raw_staleness_patience(cfg, global_step=s) for s in (0, 99, 100, 299, 300, 599, 600)], [4, 4, 8, 8, 16, 16, 24])
        self.assertEqual(payload["dgpo"]["checkpoint_load_mode"], "weights_only")
        self.assertFalse(payload["dgpo"]["auto_resume_from_last"])
        self.assertEqual(payload["options"]["Training"]["model_checkpoint_load_path"], base["options"]["Training"]["model_checkpoint_load_path"])
        self.assertNotEqual(payload["options"]["Training"]["model_checkpoint_save_path"], base["options"]["Training"]["model_checkpoint_save_path"])
        for key in ("reference_trust", "lr_schedule", "validation_every_n_epochs"):
            self.assertEqual(payload["dgpo"][key], base["dgpo"][key])
        for key in ("recalibration", "audit_fit"):
            self.assertEqual(payload["dgpo"]["adaptive_omnifold"][key], base["dgpo"]["adaptive_omnifold"][key])
        self.assertTrue(payload["logger"]["wandb"]["fresh_run"])
        state = AdaptiveOmniFoldState()
        probe = {"raw_auc_gap": .1, "raw_audit_saturated": 1., "raw_audit_training_ready": 1.}
        update_raw_plateau_controller(state, probe, cfg=cfg, epoch=0, global_step=0, checkpoint_path="/saved/step0.ckpt")
        for i in range(1, 9):
            state.install(baseline_auc_gap=.2, cfg=cfg, epoch=i, round_id=i)
            state.raw_global_refit_pending = True
            step = i * 100
            for j in range(raw_staleness_patience(cfg, global_step=step)):
                fired, _ = update_raw_plateau_controller(state, {**probe, "raw_auc_gap": .2}, cfg=cfg,
                                                         epoch=i, global_step=step+j)
            self.assertTrue(fired)
            self.assertFalse(state.raw_global_stop_requested)
            self.assertEqual(state.raw_best_global_step, 0)
            self.assertEqual(state.raw_best_checkpoint, "/saved/step0.ckpt")
        self.assertEqual(state.raw_global_failed_rounds, 8)
        bad = copy.deepcopy(payload["dgpo"])
        bad["adaptive_omnifold"]["trigger"]["global_max_failed_rounds"] = -1
        with self.assertRaises(ValueError):
            resolve_adaptive_config(bad)

    def test_lifecycle_cadence_resume_and_phase_isolation(self):
        cfg = gc.GradientConflictConfig(enabled=True, monitor_refit_lifecycle=True)
        state = AdaptiveOmniFoldState(reward_round_id=3, policy_warmup_round_id=3)
        self.assertEqual(gc.lifecycle_probe_due(cfg, state, warmup_steps=10), "post_install")
        state.gradient_post_install_probe_round_id = 3
        state.gradient_lifecycle_monitor_state = {"state": {"weight": torch.ones(2)}}
        state.policy_warmup_completed_updates = 5
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertIsNone(gc.lifecycle_probe_due(cfg, restored, warmup_steps=10))
        torch.testing.assert_close(restored.gradient_lifecycle_monitor_state["state"]["weight"], torch.ones(2))
        restored.policy_warmup_completed_updates = 10
        self.assertEqual(gc.lifecycle_probe_due(cfg, restored, warmup_steps=10), "post_warmup")
        restored.gradient_warmup_probe_round_id = 3
        self.assertIsNone(gc.lifecycle_probe_due(cfg, restored, warmup_steps=10))
        self.assertIsNone(gc.lifecycle_probe_due(replace(cfg, enabled=False), state, warmup_steps=10))
        metrics = {"gradient_conflict/trust/norm": 0., "gradient_conflict/panel_sha256": "abcd"*16}
        pre = gc.phase_metrics(metrics, "pre_refit")
        post = gc.phase_metrics(metrics, "post_install")
        self.assertFalse(set(pre) & set(post))
        self.assertEqual(trainer._wandb_sanitize_log_dict(post), post)
        wb = mock.Mock()
        trainer._wandb_define_axes(wb, critical=True)
        definitions = {c.args[0]: c.kwargs for c in wb.define_metric.call_args_list}
        self.assertEqual(definitions["gradient_conflict/post_install/trust/norm"]["step_metric"], "global_step")
        with self.assertRaises(ValueError):
            gc.resolve_gradient_conflict_config({"monitor_refit_lifecycle": "true"})

    def test_critical_and_simplified_keep_and_axes(self):
        key = "gradient_conflict/omnifold_staleness/cosine"
        self.assertTrue(trainer._wandb_critical_keep(key))
        self.assertTrue(trainer._wandb_simplified_keep(key, -.8))
        wb = mock.Mock()
        trainer._wandb_define_axes(wb, critical=True)
        definitions = {c.args[0]: c.kwargs for c in wb.define_metric.call_args_list}
        self.assertEqual(definitions[key]["step_metric"], "global_step")
        self.assertFalse(definitions[key]["hidden"])
        payload = {key: -.8, "gradient_conflict/panel_sha256": "abcd" * 16,
                   "gradient_conflict/staleness/split_cosine": float("nan")}
        clean = trainer._wandb_sanitize_log_dict(payload)
        self.assertEqual(clean["gradient_conflict/panel_sha256"], "abcd" * 16)
        self.assertNotIn("gradient_conflict/staleness/split_cosine", clean)

    def test_active_yaml_budget_and_preserved_training_protocol(self):
        payload = ablation_payload()
        cfg = gc.resolve_gradient_conflict_config(payload["dgpo"]["gradient_conflict"])
        adaptive = resolve_adaptive_config(payload["dgpo"])
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.events_per_block * cfg.blocks, 4096)
        self.assertEqual(cfg.every_n_steps, 10)
        self.assertEqual(adaptive.staleness_every_n_steps, 5)
        self.assertEqual(cfg.every_n_steps % adaptive.staleness_every_n_steps, 0)
        self.assertEqual(payload["dgpo"]["validation_every_n_epochs"], 5)
        self.assertEqual(payload["dgpo"]["validation_full_every_n_epochs"], 5)
        self.assertTrue(adaptive.cache_event_inputs)
        self.assertEqual(payload["dgpo"]["K"], 8)
        self.assertEqual(payload["dgpo"]["reference_trust"]["coefficient"], 1.)
        self.assertIn("step=320.ckpt", payload["options"]["Training"]["model_checkpoint_load_path"])

    def test_mask_sidecar_survives_pool_operations_without_changing_identity(self):
        pool = fixture()["pool"]
        with_mask = pool.select(torch.tensor([5, 2])).to(torch.device("cpu")).prefix(1)
        self.assertEqual(tuple(with_mask.policy_noise_mask.shape), (1, 2))
        torch.testing.assert_close(pool.identity_inputs, replace(pool, policy_noise_mask=None).identity_inputs)


def distributed_probe_worker(rendezvous, rank):
    """Separate interpreters: real Gloo/DDP reducer regression, no training data."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        kwargs = fixture()
        policy = kwargs["core"]
        ddp = torch.nn.parallel.DistributedDataParallel(policy)
        ddp(torch.ones(1)).sum().backward()
        gradients = [p.grad.clone() for p in policy.parameters()]
        weights = [p.detach().clone() for p in policy.parameters()]
        metrics = gc.probe_gradients(**{**kwargs, "model": ddp, "world_size": 2, "rank": rank})
        assert metrics["gradient_conflict/ran"] == 1
        for p, g, w in zip(policy.parameters(), gradients, weights):
            torch.testing.assert_close(p.grad, g, atol=0, rtol=0)
            torch.testing.assert_close(p, w, atol=0, rtol=0)
        ddp(torch.ones(1)).sum().backward()
        for p, g in zip(policy.parameters(), gradients):
            torch.testing.assert_close(p.grad, 2 * g)
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


class TestDistributedProbe(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("DGPO_TEST_DISTRIBUTED") == "1", "opt-in localhost Gloo test")
    def test_real_ddp_before_probe_after_probe(self):
        with tempfile.TemporaryDirectory(prefix="gradient-conflict-gloo-") as directory:
            rendezvous = Path(directory, "rendezvous").as_uri()
            code = ("import sys,torch; "
                    "lib=torch.library.Library('torchvision','DEF'); "
                    "lib.define('nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor'); "
                    "from RL.DGPO_neutrino.test_gradient_conflict import distributed_probe_worker; "
                    "distributed_probe_worker(sys.argv[1],int(sys.argv[2]))")
            workers = [subprocess.Popen([sys.executable, "-c", code, rendezvous, str(rank)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for rank in range(2)]
            try:
                for worker in workers:
                    output, _ = worker.communicate(timeout=50)
                    self.assertEqual(worker.returncode, 0, output)
            finally:
                for worker in workers:
                    if worker.poll() is None:
                        worker.kill()
                        worker.communicate()


if __name__ == "__main__":
    unittest.main()
