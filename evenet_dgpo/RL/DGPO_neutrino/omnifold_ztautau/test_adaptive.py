"""Contract tests for adaptive Ztautau OmniFold orchestration."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from RL.DGPO_neutrino.omnifold_ztautau import adaptive as adaptive_module
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    AdaptiveOmniFoldPool,
    AdaptiveOmniFoldState,
    adaptive_audit_protocol_signature,
    resolve_adaptive_config,
    reward_refit_due_to_age,
    run_adaptive_refit,
    should_probe_epoch,
    update_controller,
    validate_adaptive_pairing,
)
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
from RL.DGPO_neutrino.model_utils import state_dict_sha256


def _config(
    *,
    log_only: bool = False,
    candidates_per_event: int = 1,
    retrain_auc_margin: float = 0.01,
    acceptance_audit_enabled: bool = True,
    acceptance_max_balanced_accuracy: float = 0.51,
    residual_min_auc_gain: float = 1.0e-3,
    require_audit_saturation: bool = False,
    max_reward_age_epochs: int | None = 20,
    required_consecutive_epochs: int = 1,
    retrain_cooldown_epochs: int = 0,
):
    return resolve_adaptive_config(
        {
            "adaptive_omnifold": {
                "enabled": True,
                "log_only": log_only,
                "staleness_every_n_epochs": 2,
                "pool_generation_batch_size": 1024,
                "trigger": {
                    "retrain_auc_margin": retrain_auc_margin,
                    "max_reward_age_epochs": max_reward_age_epochs,
                    "required_consecutive_epochs": required_consecutive_epochs,
                    "retrain_cooldown_epochs": retrain_cooldown_epochs,
                    "require_audit_saturation": require_audit_saturation,
                    "probe_max_events": 100,
                },
                "audit_fit": {
                    "batch_size": 64,
                    "validation_interval_epochs": 1.0,
                    "validation_patience_epochs": 5.0,
                },
                "recalibration": {
                    "candidates_per_event": candidates_per_event,
                    "refit_once_on_resume": True,
                    "refit_once_id": "test_regularized_v1",
                    "pool_selection_seed": 73,
                    "score_pool_events": 200,
                    "min_iterations": 2,
                    "max_iterations": 4,
                    "acceptance_audit_enabled": acceptance_audit_enabled,
                    "acceptance_max_balanced_accuracy": (
                        acceptance_max_balanced_accuracy
                    ),
                    "residual_min_auc_gain": residual_min_auc_gain,
                },
            }
        }
    )


class TestAdaptiveConfig(unittest.TestCase):
    def test_k1_is_independent_of_dgpo_group_size(self) -> None:
        cfg = _config()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.candidates_per_event, 1)
        self.assertEqual(cfg.staleness_every_n_epochs, 2)
        self.assertEqual(cfg.pool_generation_batch_size, 1024)
        self.assertEqual(cfg.pool_selection_seed, 73)
        self.assertEqual(cfg.probe_max_events, 100)
        self.assertEqual(cfg.refit_score_events, 200)
        self.assertEqual(cfg.audit_fit["batch_size"], 64)
        self.assertEqual(cfg.audit_fit["validation_patience_epochs"], 5.0)
        self.assertFalse(cfg.require_audit_saturation)
        self.assertEqual(cfg.retrain_auc_margin, 0.01)
        self.assertEqual(cfg.max_reward_age_epochs, 20)
        self.assertTrue(cfg.acceptance_audit_enabled)
        self.assertEqual(cfg.acceptance_max_balanced_accuracy, 0.51)
        self.assertEqual(cfg.crossfit_folds, 2)
        self.assertAlmostEqual(cfg.residual_min_auc_gain, 1.0e-3)
        self.assertFalse(cfg.bootstrap_on_start)
        self.assertTrue(cfg.bootstrap_fail_closed)
        self.assertTrue(cfg.refit_once_on_resume)
        self.assertEqual(cfg.refit_once_id, "test_regularized_v1")

    def test_refit_rejects_more_than_one_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidates_per_event=1"):
            _config(candidates_per_event=2)

    def test_probe_schedule_includes_epoch_minus_one(self) -> None:
        self.assertTrue(should_probe_epoch(-1, 7))
        self.assertFalse(should_probe_epoch(0, 2))
        self.assertTrue(should_probe_epoch(1, 2))

    def test_reward_age_forces_refit_only_at_configured_limit(self) -> None:
        cfg = _config(max_reward_age_epochs=20)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=9, round_id=1)
        due, age = reward_refit_due_to_age(
            state,
            epoch=19,
            max_reward_age_epochs=cfg.max_reward_age_epochs,
        )
        self.assertFalse(due)
        self.assertEqual(age, 10)
        due, age = reward_refit_due_to_age(
            state,
            epoch=29,
            max_reward_age_epochs=cfg.max_reward_age_epochs,
        )
        self.assertTrue(due)
        self.assertEqual(age, 20)

    def test_reward_age_trigger_can_be_disabled(self) -> None:
        cfg = _config(max_reward_age_epochs=None)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        due, age = reward_refit_due_to_age(
            state,
            epoch=999,
            max_reward_age_epochs=cfg.max_reward_age_epochs,
        )
        self.assertFalse(due)
        self.assertEqual(age, 1000)

    def test_auc_power_reports_when_half_percent_gap_is_underpowered(self) -> None:
        diagnostics = adaptive_module._auc_power_diagnostics(
            observed_weighted_gap=0.005,
            audit_events=10_000,
            ess_fraction=1.0,
            retrain_auc_margin=0.005,
            alpha=0.05,
            target_power=0.80,
        )
        self.assertLess(diagnostics["audit_power_at_retrain_margin"], 0.30)
        self.assertGreater(diagnostics["audit_minimum_detectable_auc_gap"], 0.005)
        self.assertEqual(diagnostics["audit_power_sufficient"], 0.0)

    def test_auc_power_reaches_target_with_large_heldout_audit(self) -> None:
        diagnostics = adaptive_module._auc_power_diagnostics(
            observed_weighted_gap=0.005,
            audit_events=60_000,
            ess_fraction=1.0,
            retrain_auc_margin=0.005,
            alpha=0.05,
            target_power=0.80,
        )
        self.assertGreater(diagnostics["audit_power_at_retrain_margin"], 0.80)
        self.assertLess(diagnostics["audit_minimum_detectable_auc_gap"], 0.005)
        self.assertEqual(diagnostics["audit_power_sufficient"], 1.0)


class TestAdaptiveController(unittest.TestCase):
    def test_two_consecutive_crossings_are_required_when_configured(self) -> None:
        cfg = _config(
            retrain_auc_margin=0.005,
            required_consecutive_epochs=2,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=1)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.02},
            cfg=cfg,
            epoch=4,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "threshold_exceeded")
        self.assertEqual(state.probe_exceedance_streak, 1)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.021},
            cfg=cfg,
            epoch=9,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")
        self.assertEqual(state.probe_exceedance_streak, 2)

    def test_cooldown_requires_fresh_post_cooldown_crossings(self) -> None:
        cfg = _config(
            retrain_auc_margin=0.005,
            required_consecutive_epochs=2,
            retrain_cooldown_epochs=10,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=20, round_id=2)
        state.last_recalibration_epoch = 20

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.03},
            cfg=cfg,
            epoch=25,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "cooldown")
        self.assertEqual(diagnostics["staleness/cooldown_active"], 1.0)
        self.assertEqual(state.probe_exceedance_streak, 0)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.03},
            cfg=cfg,
            epoch=30,
        )
        self.assertFalse(trigger)
        self.assertEqual(state.probe_exceedance_streak, 1)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.031},
            cfg=cfg,
            epoch=35,
        )
        self.assertTrue(trigger)
        self.assertEqual(state.probe_exceedance_streak, 2)

    def test_trigger_compares_to_installed_round_baseline(self) -> None:
        cfg = _config(retrain_auc_margin=0.005)
        state = AdaptiveOmniFoldState()
        state.install(
            baseline_auc_gap=0.020,
            cfg=cfg,
            epoch=-1,
            round_id=0,
        )
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.024,
            },
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(trigger)
        self.assertAlmostEqual(diagnostics["staleness/trigger_threshold"], 0.025)
        self.assertAlmostEqual(state.baseline_auc_gap, 0.020)

        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.028,
            },
            cfg=cfg,
            epoch=19,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")
        self.assertAlmostEqual(
            diagnostics["staleness/previous_audit_auc_gap"], 0.024
        )
        self.assertAlmostEqual(diagnostics["staleness/trigger_threshold"], 0.025)
        self.assertAlmostEqual(state.baseline_auc_gap, 0.020)
        self.assertAlmostEqual(state.previous_audit_auc_gap, 0.028)

    def test_trigger_is_exactly_baseline_plus_configured_margin(self) -> None:
        cfg = _config(retrain_auc_margin=0.005)
        state = AdaptiveOmniFoldState()
        state.install(
            baseline_auc_gap=0.020,
            cfg=cfg,
            epoch=-1,
            round_id=0,
        )
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.025,
            },
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(trigger)
        self.assertAlmostEqual(
            diagnostics["staleness/required_auc_gap_increase"],
            0.005,
        )
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.025001,
            },
            cfg=cfg,
            epoch=19,
        )
        self.assertTrue(trigger)

    def test_ess_is_diagnostic_only(self) -> None:
        cfg = _config(retrain_auc_margin=0.005)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.020, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.021, "ess_fraction": 0.0},
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "healthy")

    def test_unsaturated_audit_gap_can_trigger_when_configured(self) -> None:
        cfg = _config(require_audit_saturation=False)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 0.0,
                "ess_fraction": 1.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")

    def test_required_saturation_blocks_trigger_until_audit_saturates(self) -> None:
        cfg = _config(require_audit_saturation=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 0.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "audit_unsaturated")
        self.assertEqual(state.probe_exceedance_streak, 0)

        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 1.0,
            },
            cfg=cfg,
            epoch=3,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")

    def test_confirmed_threshold_crossing_can_stop_before_saturation(self) -> None:
        cfg = _config(require_audit_saturation=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 0.0,
                "audit_threshold_reached": 1.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")

    def test_validation_crossing_needs_final_split_confirmation(self) -> None:
        cfg = _config(require_audit_saturation=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.015,
                "audit_saturated": 0.0,
                "audit_threshold_reached": 1.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "audit_unsaturated")

    def test_log_only_never_authorizes_refit(self) -> None:
        cfg = _config(log_only=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1)
        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.2, "ess_fraction": 1.0},
            cfg=cfg,
            epoch=1,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "stale_log_only")

    def test_state_round_trip_preserves_fixed_baseline(self) -> None:
        cfg = _config()
        state = AdaptiveOmniFoldState()
        state.install(
            baseline_auc_gap=0.08,
            cfg=cfg,
            epoch=3,
            round_id=2,
        )
        state.audit_protocol_signature = adaptive_audit_protocol_signature(cfg)
        state.resume_refit_once_completed = True
        state.resume_refit_once_id = cfg.refit_once_id
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(restored.reward_round_id, 2)
        self.assertEqual(restored.trigger_threshold, state.trigger_threshold)
        self.assertEqual(restored.baseline_auc_gap, state.baseline_auc_gap)
        self.assertTrue(restored.resume_refit_once_completed)
        self.assertEqual(restored.resume_refit_once_id, cfg.refit_once_id)
        self.assertEqual(
            restored.audit_protocol_signature,
            state.audit_protocol_signature,
        )

    def test_old_checkpoint_migrates_previous_audit_from_history(self) -> None:
        restored = AdaptiveOmniFoldState.from_dict(
            {
                "reward_round_id": 5,
                "baseline_auc_gap": 0.019,
                "trigger_threshold": 0.024,
                "probe_history": [
                    {"weighted_auc_gap": 0.012},
                    {"weighted_auc_gap": 0.007},
                ],
            }
        )
        self.assertAlmostEqual(restored.previous_audit_auc_gap, 0.007)

    def test_protocol_change_invalidates_only_audit_baseline(self) -> None:
        restored = AdaptiveOmniFoldState.from_dict(
            {
                "reward_round_id": 5,
                "baseline_auc_gap": 0.019,
                "trigger_threshold": 0.024,
            }
        )
        self.assertTrue(restored.calibrated)
        restored.invalidate_audit_baseline(reason="protocol_changed")
        self.assertEqual(restored.reward_round_id, 5)
        self.assertFalse(restored.calibrated)
        self.assertEqual(restored.last_decision, "protocol_changed")

class TestAdaptivePool(unittest.TestCase):
    def test_pool_contract_is_four_dimensional_k1(self) -> None:
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        self.assertEqual(pool.n_events, 32)
        with self.assertRaisesRegex(ValueError, "exactly K=1"):
            AdaptiveOmniFoldPool(
                packed_event=pool.packed_event,
                truth=pool.truth,
                candidates=torch.randn(32, 2, 4),
                packing_spec=spec,
            )

    def test_fresh_audit_fits_only_the_weighted_judge(self) -> None:
        cfg = _config()
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )

        class _Builder:
            def __init__(self) -> None:
                self.discarded: list[str] = []

            def make_classifier(self, *_args, **_kwargs):
                return torch.nn.Linear(1, 1)

            def discard_bank(self, name: str) -> None:
                self.discarded.append(name)

        def _result(*, auc: float, accuracy: float):
            return SimpleNamespace(
                auc=auc,
                auc_gap=abs(auc - 0.5),
                balanced_accuracy=accuracy,
                fit_diagnostics=SimpleNamespace(
                    saturated=True,
                    threshold_reached=False,
                    validation_loss=0.69,
                    validation_balanced_accuracy=accuracy,
                    validation_auc=auc,
                ),
                fit_events=19,
                audit_events=6,
            )

        builder = _Builder()
        log_weight = torch.linspace(-1.0, 1.0, 32).reshape(32, 1)
        with mock.patch.object(
            adaptive_module,
            "fit_independent_evenet_audit",
            return_value=_result(auc=0.53, accuracy=0.54),
        ) as fit:
            metrics = adaptive_module.fit_fresh_audit(
                pool=pool,
                log_weight=log_weight,
                model_builder=builder,
                cfg=cfg,
                device=torch.device("cpu"),
                seed=123,
            )

        self.assertEqual(fit.call_count, 1)
        audit_fit_config = fit.call_args_list[0].kwargs["fit_config"]
        self.assertEqual(audit_fit_config.batch_size, 64)
        self.assertEqual(audit_fit_config.validation_interval_steps, 1)
        self.assertEqual(audit_fit_config.validation_patience_evaluations, 5)
        weighted_gen_weight = fit.call_args_list[0].kwargs["gen_weight"]
        self.assertFalse(torch.allclose(weighted_gen_weight, torch.ones_like(weighted_gen_weight)))
        self.assertAlmostEqual(metrics["judge_auc_weighted"], 0.53)
        self.assertNotIn("judge_auc_raw", metrics)
        self.assertEqual(builder.discarded, ["audit"])


class TestAtomicAdaptiveInstall(unittest.TestCase):
    def test_resume_pairing_rejects_a_different_round_reference(self) -> None:
        round_ref = torch.nn.Linear(2, 2)
        digest = state_dict_sha256(round_ref)
        source = SimpleNamespace(
            reward_round_id=2,
            reference_kind="state_dict_sha256",
            policy_reference_sha256=digest,
        )
        state = AdaptiveOmniFoldState(reward_round_id=2)
        validate_adaptive_pairing(
            reward_source=source,
            state=state,
            round_ref_model=round_ref,
            checkpoint={
                "dgpo_reward_round_id": 2,
                "dgpo_round_ref_sha256": digest,
            },
            where="test",
        )
        source.policy_reference_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "round-reference"):
            validate_adaptive_pairing(
                reward_source=source,
                state=state,
                round_ref_model=round_ref,
                where="test",
            )

    def test_initial_bootstrap_skips_missing_incumbent_and_installs_candidate(
        self,
    ) -> None:
        cfg = _config(
            acceptance_audit_enabled=False,
            residual_min_auc_gain=0.01,
        )
        state = AdaptiveOmniFoldState()
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        round_ref = torch.nn.Linear(2, 2)
        policy = torch.nn.Linear(2, 2)
        snapshot = {
            key: value.detach().clone() for key, value in policy.state_dict().items()
        }

        class _Stack:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def assert_frozen(self):
                return None

        class _Source:
            model_builder = object()

            def __init__(self):
                self.is_installed = False
                self.installed = None

            @property
            def frozen_reward(self):
                raise AssertionError("cold bootstrap must not score an incumbent")

            def replace_stack(self, stack, **kwargs):
                self.installed = (stack, kwargs)
                self.is_installed = True

        source = _Source()
        fit_result = SimpleNamespace(
            diagnostics=(
                SimpleNamespace(saturated=True, validation_auc=0.58),
                SimpleNamespace(saturated=True, validation_auc=0.51),
            ),
            iterations=2,
        )
        progress_events: list[tuple[str, int]] = []

        def _fit_side_effect(**kwargs):
            kwargs["progress_callback"]({"iteration": 1.0, "step": 10.0})
            return fit_result

        with (
            mock.patch.object(
                adaptive_module,
                "fit_residual_ratio_stack",
                side_effect=_fit_side_effect,
            ),
            mock.patch.object(
                adaptive_module.FrozenResidualRatioReward,
                "from_fit_result",
                return_value=_Stack(),
            ),
            mock.patch.object(
                adaptive_module,
                "score_reward_on_pool",
                return_value=torch.zeros(32, 1),
            ) as score_reward,
            mock.patch.object(
                adaptive_module,
                "fit_fresh_audit",
                side_effect=AssertionError("acceptance audit must be disabled"),
            ) as fresh_audit,
        ):
            diagnostics = run_adaptive_refit(
                state=state,
                cfg=cfg,
                reward_source=source,
                round_ref_model=round_ref,
                policy_snapshot_state_dict=snapshot,
                fit_pool=pool,
                score_pool=pool,
                epoch=-1,
                device=torch.device("cpu"),
                world_size=1,
                progress_callback=lambda phase, row: progress_events.append(
                    (phase, int(row["step"]))
                ),
            )

        self.assertEqual(diagnostics["omnifold/accepted"], 1.0)
        self.assertEqual(diagnostics["omnifold/initial_bootstrap"], 1.0)
        self.assertEqual(
            diagnostics["omnifold/accept_reason"],
            "candidate reached saturated cross-fit residual closure; "
            "fresh acceptance audit disabled",
        )
        self.assertEqual(state.reward_round_id, 1)
        self.assertIsNotNone(source.installed)
        self.assertEqual(source.installed[1]["round_id"], 1)
        score_reward.assert_not_called()
        fresh_audit.assert_not_called()
        self.assertEqual(
            progress_events,
            [("residual_reward", 10)],
        )
        for key, expected in snapshot.items():
            torch.testing.assert_close(round_ref.state_dict()[key], expected)

    def test_accepted_stack_moves_reward_and_round_reference_together(self) -> None:
        cfg = _config()
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.08, cfg=cfg, epoch=-1, round_id=0)
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        round_ref = torch.nn.Linear(2, 2)
        policy = torch.nn.Linear(2, 2)
        snapshot = {
            key: value.detach().clone() for key, value in policy.state_dict().items()
        }

        class _Stack:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def assert_frozen(self):
                return None

        class _Source:
            model_builder = object()
            frozen_reward = object()

            def __init__(self):
                self.installed = None

            def replace_stack(self, stack, **kwargs):
                self.installed = (stack, kwargs)

        source = _Source()
        fit_result = SimpleNamespace(
            diagnostics=(SimpleNamespace(saturated=True),),
            iterations=1,
        )
        candidate_audit = {
            "weighted_auc_gap": 0.01,
            "audit_observed_auc_gap": 0.01,
            "audit_balanced_accuracy": 0.50,
            "audit_saturated": 1.0,
        }
        cheap_baseline_audit = {
            "weighted_auc_gap": 0.03,
            "audit_observed_auc_gap": 0.03,
            "audit_balanced_accuracy": 0.50,
            "audit_saturated": 1.0,
        }
        audit_results = iter((candidate_audit, cheap_baseline_audit))
        progress_events: list[tuple[str, int]] = []

        def _fit_side_effect(**kwargs):
            kwargs["progress_callback"]({"iteration": 1.0, "step": 10.0})
            return fit_result

        def _audit_side_effect(**kwargs):
            kwargs["progress_callback"]({"iteration": 1.0, "step": 10.0})
            return next(audit_results)

        with (
            mock.patch.object(
                adaptive_module,
                "fit_residual_ratio_stack",
                side_effect=_fit_side_effect,
            ),
            mock.patch.object(
                adaptive_module.FrozenResidualRatioReward,
                "from_fit_result",
                return_value=_Stack(),
            ),
            mock.patch.object(
                adaptive_module,
                "score_reward_on_pool",
                side_effect=lambda _stack, scored_pool, **_kwargs: torch.zeros(
                    scored_pool.n_events, 1
                ),
            ),
            mock.patch.object(
                adaptive_module,
                "fit_fresh_audit",
                side_effect=_audit_side_effect,
            ),
        ):
            diagnostics = run_adaptive_refit(
                state=state,
                cfg=cfg,
                reward_source=source,
                round_ref_model=round_ref,
                policy_snapshot_state_dict=snapshot,
                fit_pool=pool,
                score_pool=pool,
                baseline_pool=pool.prefix(30),
                epoch=1,
                device=torch.device("cpu"),
                world_size=1,
                progress_callback=lambda phase, row: progress_events.append(
                    (phase, int(row["step"]))
                ),
            )
        self.assertEqual(diagnostics["omnifold/accepted"], 1.0)
        self.assertEqual(state.reward_round_id, 1)
        self.assertAlmostEqual(state.baseline_auc_gap, 0.03)
        self.assertAlmostEqual(state.trigger_threshold, 0.04)
        self.assertEqual(diagnostics["omnifold/baseline/probe_events"], 30.0)
        self.assertIsNotNone(source.installed)
        self.assertEqual(
            progress_events,
            [
                ("residual_reward", 10),
                ("acceptance_audit", 10),
                ("baseline_audit", 10),
            ],
        )
        self.assertEqual(source.installed[1]["round_id"], 1)
        for key, expected in snapshot.items():
            torch.testing.assert_close(round_ref.state_dict()[key], expected)


if __name__ == "__main__":
    unittest.main()
