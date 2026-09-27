"""Tests for installing a deliberately undertrained, signal-bearing reward."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from . import adaptive as ad
from . import evenet_ratio as ratio
from .ratio_fit import ConditionalRatioMLP, RatioFitConfig, fit_density_ratio
from .test_adaptive import _config
from .. import dgpo_trainer as trainer


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, condition, sample):
        return self.bias.expand(sample.shape[:-1])


def _member(*, reached: bool = True, steps: int = 1002):
    return SimpleNamespace(
        saturated=False,
        threshold_reached=reached,
        threshold_reason=(
            "validation_balanced_accuracy_lcb" if reached else None
        ),
        loss=0.55,
        balanced_accuracy=0.70,
        steps_completed=steps,
    )


class TestMinimumSufficientReward(unittest.TestCase):
    def test_ba_stop_uses_internal_validation_without_a_separate_evaluator(self) -> None:
        torch.manual_seed(11)
        model = ConditionalRatioMLP(2, 1, hidden_dim=8, hidden_layers=1)
        condition = torch.zeros(16, 2)
        positive = torch.ones(16, 1)
        negative = -torch.ones(16, 1)
        validation = (
            condition,
            positive,
            torch.ones(16),
            condition,
            negative,
            torch.ones(16),
        )
        diagnostics = fit_density_ratio(
            model,
            *validation,
            RatioFitConfig(
                steps=None,
                batch_size=4,
                sampling="independent_epoch_shuffle",
                min_steps=4,
                validation_interval_steps=4,
                validation_patience_evaluations=20,
                validation_batch_size=16,
            ),
            seed=11,
            validation=validation,
            stop_when_validation_balanced_accuracy_lcb_exceeds=0.70,
            validation_balanced_accuracy_lcb_confidence_z=0.0,
            validation_balanced_accuracy_lcb_events_per_class=16,
            validation_balanced_accuracy_lcb_required_consecutive=2,
        )
        self.assertTrue(diagnostics.threshold_reached)
        self.assertFalse(diagnostics.saturated)
        self.assertGreaterEqual(diagnostics.steps_completed, 4)

    def test_four_non_saturated_members_can_be_minimum_sufficient(self) -> None:
        diag = ratio.ResidualIterationDiagnostics(
            1,
            (_member(),) * 4,
            0.693,
            0.55,
            0.70,
            0.75,
            0.143,
            True,
        )
        self.assertFalse(diag.saturated)
        self.assertTrue(diag.minimum_sufficient)

    def test_stack_passes_ba70_rule_to_every_fold_and_fails_closed(self) -> None:
        condition = torch.arange(160, dtype=torch.float32).reshape(80, 2)
        sample = torch.zeros(80, 4)
        calls = []

        def fit(model, *args, **kwargs):
            calls.append(kwargs)
            with torch.no_grad():
                model.bias.add_(0.1)
            return _member()

        with mock.patch(
            "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
            side_effect=fit,
        ), mock.patch.object(
            ratio,
            "_weighted_binary_score_metrics",
            return_value=(0.55, 0.70, 0.75),
        ):
            result = ratio.fit_residual_ratio_stack(
                model_factory=_Tiny,
                data_condition=condition,
                data_sample=sample,
                gen_condition=condition,
                gen_sample=sample,
                iterations=1,
                min_iterations=1,
                iteration_one_only=True,
                fit_config=RatioFitConfig(
                    steps=None,
                    batch_size=4,
                    min_steps=1000,
                    require_saturation=False,
                ),
                tempering=1.0,
                seed=42,
                crossfit_repeats=2,
                crossfit_folds=2,
                validation_data_condition=condition,
                validation_data_sample=sample,
                validation_gen_condition=condition,
                validation_gen_sample=sample,
                minimum_sufficient_balanced_accuracy=0.70,
                minimum_sufficient_confidence_z=0.0,
                minimum_sufficient_required_consecutive=3,
            )
        self.assertEqual(len(calls), 4)
        for kwargs in calls:
            self.assertEqual(
                kwargs["stop_when_validation_balanced_accuracy_lcb_exceeds"],
                0.70,
            )
            self.assertEqual(
                kwargs["validation_balanced_accuracy_lcb_confidence_z"], 0.0
            )
            self.assertEqual(
                kwargs[
                    "validation_balanced_accuracy_lcb_required_consecutive"
                ],
                3,
            )
            self.assertEqual(
                kwargs["validation_balanced_accuracy_lcb_events_per_class"],
                80,
            )
        self.assertTrue(result.diagnostics[0].minimum_sufficient)
        self.assertFalse(result.diagnostics[0].saturated)

        with self.assertRaisesRegex(RuntimeError, "minimum-sufficient"):
            with mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
                return_value=_member(reached=False),
            ):
                ratio.fit_residual_ratio_stack(
                    model_factory=_Tiny,
                    data_condition=condition,
                    data_sample=sample,
                    gen_condition=condition,
                    gen_sample=sample,
                    iterations=1,
                    min_iterations=1,
                    iteration_one_only=True,
                    fit_config=RatioFitConfig(
                        steps=None,
                        batch_size=4,
                        min_steps=1000,
                        require_saturation=False,
                    ),
                    tempering=1.0,
                    seed=42,
                    validation_data_condition=condition,
                    validation_data_sample=sample,
                    validation_gen_condition=condition,
                    validation_gen_sample=sample,
                    minimum_sufficient_balanced_accuracy=0.70,
                    minimum_sufficient_required_consecutive=3,
                )

    def test_adaptive_installs_ready_unsaturated_reward_and_logs_it(self) -> None:
        cfg = replace(
            _config(
                monitor_mode="raw_plateau_refit",
                raw_audit_enabled=True,
                acceptance_audit_enabled=False,
            ),
            iteration_one_only=True,
            min_iterations=1,
            max_iterations=1,
            crossfit_repeats=2,
            crossfit_folds=2,
            fit={
                "require_saturation": False,
                "minimum_sufficient_balanced_accuracy": 0.70,
                "minimum_sufficient_confidence_z": 0.0,
                "minimum_sufficient_required_consecutive": 3,
            },
        )
        spec = ratio.EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = ad.AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        source = SimpleNamespace(model_builder=mock.Mock(), replace_stack=mock.Mock())
        stack = mock.Mock(spec=["to", "eval", "assert_frozen"])
        stack.to.return_value = stack
        stack.eval.return_value = stack
        diag = ratio.ResidualIterationDiagnostics(
            1,
            tuple(_member(steps=1000 + i) for i in range(4)),
            0.693,
            0.55,
            0.70,
            0.75,
            0.143,
            True,
        )
        result = SimpleNamespace(
            diagnostics=(diag,),
            iterations=1,
            train_log_weight=torch.zeros(32),
            validation_log_weight=torch.zeros(32),
        )
        with mock.patch.object(
            ad, "fit_residual_ratio_stack", return_value=result
        ) as fitting, mock.patch.object(
            ad.FrozenResidualRatioReward, "from_fit_result", return_value=stack
        ):
            metrics = ad.run_adaptive_refit(
                state=ad.AdaptiveOmniFoldState(),
                cfg=cfg,
                reward_source=source,
                round_ref_model=torch.nn.Linear(2, 2),
                policy_snapshot_state_dict=torch.nn.Linear(2, 2).state_dict(),
                fit_pool=pool,
                score_pool=pool,
                epoch=-1,
                device=torch.device("cpu"),
                world_size=1,
                global_step=0,
            )
        self.assertEqual(
            fitting.call_args.kwargs["minimum_sufficient_balanced_accuracy"],
            0.70,
        )
        source.replace_stack.assert_called_once()
        self.assertEqual(metrics["omnifold/accepted"], 1.0)
        self.assertEqual(metrics["omnifold/all_fits_saturated"], 0.0)
        self.assertEqual(metrics["omnifold/all_fits_ready"], 1.0)
        self.assertEqual(
            metrics["omnifold/fit/iter01/threshold_reached_folds"], 4.0
        )
        self.assertEqual(metrics["omnifold/fit/iter01/fit_steps_min"], 1000.0)
        self.assertIn("minimum-sufficient", metrics["omnifold/accept_reason"])
        for key, value in metrics.items():
            if key.startswith("omnifold/minimum_sufficient/") or key in {
                "omnifold/all_fits_ready"
            }:
                self.assertTrue(trainer._wandb_critical_keep(key))
                self.assertTrue(trainer._wandb_simplified_keep(key, value))


if __name__ == "__main__":
    unittest.main()
