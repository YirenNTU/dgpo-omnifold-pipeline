"""Production iteration-one-only ablation; legacy closure remains fail-closed."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

from . import adaptive as ad
from . import evenet_ratio as ratio
from .ratio_fit import RatioFitConfig
from .test_adaptive import _config
from .. import dgpo_trainer as trainer


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, condition, sample):
        return self.bias.expand(sample.shape[:-1])


class TestIterationOne(unittest.TestCase):
    def run_fit(
        self,
        *,
        enabled=True,
        auc=.8,
        saturated=True,
        saved=None,
        log_ratio_clip=None,
        fixed_budget=False,
    ):
        condition = torch.arange(160, dtype=torch.float32).reshape(80, 2)
        sample = torch.zeros(80, 4)
        calls = []
        def fit(model, dc, ds, dw, gc, gs, gw, config, *args, **kwargs):
            calls.append(float(model.bias))
            self.assertTrue(torch.equal(gw, torch.ones_like(gw)))
            with torch.no_grad():
                model.bias.add_(.1)
            return SimpleNamespace(saturated=saturated, loss=.4,
                                   balanced_accuracy=.8, steps_completed=20)
        with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio', side_effect=fit), \
             mock.patch.object(ratio, '_weighted_binary_score_metrics', return_value=(.45,.8,auc)):
            result = ratio.fit_residual_ratio_stack(model_factory=Tiny,
                data_condition=condition, data_sample=sample,
                gen_condition=condition, gen_sample=sample,
                iterations=2 if fixed_budget else 1,
                min_iterations=2 if fixed_budget else 1,
                iteration_one_only=enabled,
                fixed_iteration_budget=fixed_budget,
                fit_config=RatioFitConfig(steps=40, batch_size=4, require_saturation=True),
                tempering=1., seed=42, warm_start_iterations=(1,),
                warm_start_state=saved,
                log_ratio_clip=log_ratio_clip,
                validation_data_condition=condition, validation_data_sample=sample,
                validation_gen_condition=condition, validation_gen_sample=sample)
        return result, calls

    def test_exactly_two_folds_and_next_round_inherits_without_accumulating_weights(self):
        first, calls = self.run_fit()
        self.assertEqual(calls, [0., 0.])
        self.assertEqual(first.iterations, 1)
        self.assertEqual(first.checkpoint_iterations, (1, 1))
        self.assertEqual(first.checkpoint_coefficients, (.5, .5))
        self.assertEqual(first.iteration_temperatures, (1.,))
        torch.testing.assert_close(first.train_log_weight, torch.full((80,), .1))
        reward = ratio.FrozenResidualRatioReward.from_fit_result(first, tempering=1.)
        reward.assert_frozen()
        torch.testing.assert_close(reward(torch.zeros(80,2), torch.zeros(80,4)), torch.full((80,), .1))
        adaptive_reward = ratio.FrozenResidualRatioReward.from_fit_result(
            replace(first, iteration_temperatures=(.5,)),
            tempering=1.,
        )
        torch.testing.assert_close(
            adaptive_reward(torch.zeros(80, 2), torch.zeros(80, 4)),
            torch.full((80,), .05),
        )
        second, calls = self.run_fit(saved=first.warm_start_state)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(abs(v-.1)<1e-6 for v in calls))
        self.assertEqual(second.diagnostics[0].warm_started_folds, (1, 2))
        torch.testing.assert_close(second.train_log_weight, torch.full((80,), .2))

    def test_legacy_cap_still_raises_and_signal_and_saturation_still_required(self):
        with self.assertRaisesRegex(RuntimeError, 'held-out no-op'):
            self.run_fit(enabled=False)
        with self.assertRaisesRegex(RuntimeError, 'AUC gate'):
            self.run_fit(auc=.5)
        with self.assertRaisesRegex(RuntimeError, 'saturat'):
            self.run_fit(saturated=False)

    def test_fixed_two_iterations_return_useful_stack_without_closure(self):
        result, calls = self.run_fit(enabled=False, fixed_budget=True)
        self.assertEqual(len(calls), 4)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.checkpoint_iterations, (1, 1, 2, 2))
        self.assertTrue(all(d.accepted for d in result.diagnostics))
        with self.assertRaisesRegex(RuntimeError, 'AUC gate'):
            self.run_fit(enabled=False, fixed_budget=True, auc=.5)

    def test_weight_metrics_are_shift_invariant_and_not_clipped(self):
        w = torch.tensor([0., 1., 4., 10.], dtype=torch.float64)
        a, b = ad._iteration_one_weight_metrics(w), ad._iteration_one_weight_metrics(w+10000)
        self.assertEqual(a, b)
        self.assertLess(a['ess_fraction'], .3)
        self.assertGreater(a['top_1pct_mass'], .99)
        unit = ad._iteration_one_weight_metrics(torch.zeros(100))
        self.assertAlmostEqual(unit['ess_fraction'], 1.)
        self.assertAlmostEqual(unit['top_1pct_mass'], .01)

    def test_bounded_log_ratio_matches_training_and_frozen_reward(self):
        result, _ = self.run_fit(log_ratio_clip=.05)
        torch.testing.assert_close(
            result.train_log_weight, torch.full((80,), .05)
        )
        reward = ratio.FrozenResidualRatioReward.from_fit_result(
            result, tempering=1., log_ratio_clip=.05
        )
        torch.testing.assert_close(
            reward(torch.zeros(80, 2), torch.zeros(80, 4)),
            torch.full((80,), .05),
        )

        model = Tiny()
        positive = {'bias': torch.tensor(3.)}
        negative = {'bias': torch.tensor(-3.)}
        sequential = ratio.FrozenResidualRatioReward(
            model,
            (positive, negative),
            checkpoint_iterations=(1, 2),
            log_ratio_clip=1.,
        )
        # Sequential clipping reproduces the weights propagated between
        # OmniFold iterations: clamp(clamp(3)-3)=-1, not clamp(3-3)=0.
        torch.testing.assert_close(
            sequential(torch.zeros(4, 2), torch.zeros(4, 4)),
            torch.full((4,), -1.),
        )
        adaptive = ratio.FrozenResidualRatioReward(
            model,
            (positive, negative),
            tempering=1.,
            checkpoint_iterations=(1, 2),
            iteration_temperatures=(.25, .75),
        )
        self.assertEqual(adaptive.iteration_temperatures, (.25, .75))
        torch.testing.assert_close(
            adaptive(torch.zeros(4, 2), torch.zeros(4, 4)),
            torch.full((4,), -1.5),
        )

    def test_adaptive_alpha_is_independent_and_can_increase_later(self):
        current = torch.zeros(100)
        spike = torch.zeros(100)
        spike[0] = 10.
        (
            first_alpha,
            first_train,
            first_validation,
            _first_train_ess,
            _first_validation_ess,
            first_reached,
        ) = ratio._select_ess_tempering(
            train_log_weight=current,
            validation_log_weight=current,
            train_increment=spike,
            validation_increment=spike,
            maximum=.75,
            minimum=.1,
            target_ess_fraction=.8,
            grid_steps=14,
            log_ratio_clip=None,
        )
        self.assertTrue(first_reached)
        self.assertLess(first_alpha, .75)

        # A later residual can flatten the cumulative tail. Its alpha search is
        # independent, so it may return to full strength.
        flattening = -first_train / .75
        (
            later_alpha,
            _later_train,
            _later_validation,
            later_train_ess,
            later_validation_ess,
            later_reached,
        ) = ratio._select_ess_tempering(
            train_log_weight=first_train,
            validation_log_weight=first_validation,
            train_increment=flattening,
            validation_increment=-first_validation / .75,
            maximum=.75,
            minimum=.1,
            target_ess_fraction=.8,
            grid_steps=14,
            log_ratio_clip=None,
        )
        self.assertTrue(later_reached)
        self.assertEqual(later_alpha, .75)
        self.assertGreater(later_alpha, first_alpha)
        self.assertAlmostEqual(later_train_ess, 1.)
        self.assertAlmostEqual(later_validation_ess, 1.)

    def test_inherited_tempering_reuses_previous_alpha(self):
        condition = torch.arange(160, dtype=torch.float32).reshape(80, 2)
        sample = torch.zeros(80, 4)

        def fit(*_args, **_kwargs):
            return SimpleNamespace(
                saturated=True,
                loss=.4,
                balanced_accuracy=.8,
                steps_completed=20,
            )

        def score(_model, population, _sample, _batch_size):
            return torch.full((len(population),), .15)

        metrics = mock.Mock(side_effect=[
            (.4, .7, .7),
            (.4, .7, .7),
            (.6, .5, .5),
        ])

        with mock.patch(
            'RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio',
            side_effect=fit,
        ), mock.patch.object(
            ratio, '_score_population',
            side_effect=score,
        ), mock.patch.object(
            ratio, '_weighted_binary_score_metrics',
            metrics,
        ), mock.patch.object(
            ratio, '_select_ess_tempering',
            return_value=(.35, torch.zeros(80), torch.zeros(80), .9, .9, True),
        ) as select_alpha:
            result = ratio.fit_residual_ratio_stack(
                model_factory=Tiny,
                data_condition=condition,
                data_sample=sample,
                gen_condition=condition,
                gen_sample=sample,
                iterations=3,
                min_iterations=1,
                fit_config=RatioFitConfig(
                    steps=None,
                    min_steps=1,
                    batch_size=16,
                    sampling='independent_epoch_shuffle',
                    restore_best=True,
                    require_saturation=True,
                ),
                tempering=.75,
                seed=7,
                crossfit_folds=2,
                residual_min_auc_gain=.02,
                validation_data_condition=condition,
                validation_data_sample=sample,
                validation_gen_condition=condition,
                validation_gen_sample=sample,
                warm_start_iterations=(1,),
                warm_start_from_iteration_one=True,
                adaptive_tempering=True,
                target_ess_fraction=.3,
                minimum_tempering=.1,
                tempering_grid_steps=14,
                inherit_previous_tempering=True,
                minimum_ess_fraction=.3,
            )
        self.assertEqual(select_alpha.call_count, 1)
        self.assertEqual(result.iteration_temperatures, (.35, .35))
        self.assertEqual(
            [item.applied_tempering for item in result.diagnostics[:2]],
            [.35, .35],
        )

    def test_ess_aware_checkpoint_prefers_earlier_smooth_snapshot(self):
        class ScaleRatio(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, condition, sample):
                # condition[:,0]=1 marks data rows; gen rows stay 0. A large
                # scale both separates classes and spikes the first gen weight.
                label = condition[:, 0]
                logits = self.scale * (2. * label - 1.)
                peak = torch.zeros_like(logits)
                peak[0] = self.scale * 8.
                return logits + peak

        def snapshot(step, scale, loss):
            model = ScaleRatio()
            with torch.no_grad():
                model.scale.fill_(scale)
            return {
                'step': float(step),
                'loss': float(loss),
                'balanced_accuracy': .7,
                'auc': .7,
                'state': {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            }

        n = 64
        data_condition = torch.ones(n, 2)
        gen_condition = torch.zeros(n, 2)
        sample = torch.zeros(n, 4)
        log_weight = torch.zeros(n)
        selected = ratio._select_ess_aware_fold_checkpoint(
            ScaleRatio(),
            (
                snapshot(10, .5, .40),
                snapshot(40, 6., .10),  # loss-best, but too sharp for ESS
            ),
            holdout_condition=gen_condition,
            holdout_sample=sample,
            holdout_log_weight=log_weight,
            validation_data_condition=data_condition,
            validation_data_sample=sample,
            validation_gen_condition=gen_condition,
            validation_gen_sample=sample,
            validation_log_weight=log_weight,
            score_batch_size=32,
            min_auc=.02,
            tempering_maximum=.75,
            tempering_minimum=.1,
            target_ess_fraction=.2,
            tempering_grid_steps=14,
            adaptive_tempering=True,
            log_ratio_clip=2.5,
        )
        self.assertEqual(selected['step'], 10.)
        self.assertFalse(selected['used_loss_best_fallback'])
        self.assertEqual(selected['loss_best_step'], 40.)
        self.assertGreater(selected['applied_tempering'], .1)
        self.assertTrue(selected['ess_target_reached'])
        self.assertAlmostEqual(float(selected['state']['scale']), .5)

    def test_each_residual_increment_must_pass_ess_before_commit(self):
        condition = torch.arange(160, dtype=torch.float32).reshape(80, 2)
        sample = torch.zeros(80, 4)
        score_call = 0

        def fit(*_args, **_kwargs):
            return SimpleNamespace(
                saturated=True,
                loss=.4,
                balanced_accuracy=.8,
                steps_completed=20,
            )

        def score(_model, population, _sample, _batch_size):
            nonlocal score_call
            phase = score_call % 3
            score_call += 1
            values = torch.zeros(len(population))
            # Per fold: OOF-gen, validation-data, validation-gen.
            if phase != 1:
                values[0] = 10.
            return values

        with mock.patch(
            'RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio',
            side_effect=fit,
        ), mock.patch.object(
            ratio, '_score_population', side_effect=score
        ), mock.patch.object(
            ratio,
            '_weighted_binary_score_metrics',
            return_value=(.45, .8, .8),
        ), self.assertRaisesRegex(RuntimeError, 'ESS gate'):
            ratio.fit_residual_ratio_stack(
                model_factory=Tiny,
                data_condition=condition,
                data_sample=sample,
                gen_condition=condition,
                gen_sample=sample,
                iterations=1,
                min_iterations=1,
                iteration_one_only=True,
                fit_config=RatioFitConfig(
                    steps=40,
                    batch_size=4,
                    require_saturation=True,
                ),
                tempering=1.,
                seed=42,
                minimum_ess_fraction=.9,
                validation_data_condition=condition,
                validation_data_sample=sample,
                validation_gen_condition=condition,
                validation_gen_sample=sample,
            )
        for key in (
            'omnifold_live/residual_reward/train_ess_fraction',
            'omnifold_live/residual_reward/validation_ess_fraction',
        ):
            self.assertTrue(trainer._wandb_critical_keep(key))
            self.assertTrue(trainer._wandb_simplified_keep(key, .5))

    def test_atomic_install_without_false_closure_or_extra_classifier(self):
        cfg = replace(_config(monitor_mode='raw_only', log_only=True, raw_audit_enabled=True,
                              acceptance_audit_enabled=False),
                      iteration_one_only=True, min_iterations=1, max_iterations=1,
                      reward_classifier={'decoder_layers': 1,
                                         'periodic_pair_features': False})
        spec = ratio.EventPackingSpec({'x':(2,3), 'x_mask':(2,1),
                                      'conditions':(1,2), 'conditions_mask':(1,)})
        pool = ad.AdaptiveOmniFoldPool(packed_event=torch.randn(32,spec.width),
            truth=torch.randn(32,4), candidates=torch.randn(32,1,4), packing_spec=spec)
        reference, policy = torch.nn.Linear(2,2), torch.nn.Linear(2,2)
        builder = mock.Mock()
        builder.make_classifier.return_value = Tiny()
        source = SimpleNamespace(model_builder=builder, replace_stack=mock.Mock())
        stack = mock.Mock(spec=['to','eval','assert_frozen'])
        stack.to.return_value = stack; stack.eval.return_value = stack
        diag = ratio.ResidualIterationDiagnostics(1,
            (SimpleNamespace(saturated=True,loss=.4,balanced_accuracy=.8),)*2,
            .693, .45, .8, .8, .243, True)
        result = SimpleNamespace(diagnostics=(diag,), iterations=1,
            train_log_weight=torch.zeros(32), validation_log_weight=torch.zeros(32))
        with mock.patch.object(ad,'fit_residual_ratio_stack',return_value=result) as fitting, \
             mock.patch.object(ad.FrozenResidualRatioReward,'from_fit_result',return_value=stack), \
             mock.patch.object(ad,'fit_fresh_audit',side_effect=AssertionError('extra fit')):
            metrics = ad.run_adaptive_refit(state=ad.AdaptiveOmniFoldState(), cfg=cfg,
                reward_source=source, round_ref_model=reference,
                policy_snapshot_state_dict=policy.state_dict(), fit_pool=pool, score_pool=pool,
                epoch=-1, device=torch.device('cpu'), world_size=1, global_step=0)
        self.assertTrue(fitting.call_args.kwargs['iteration_one_only'])
        fitting.call_args.kwargs['model_factory']()
        builder.make_classifier.assert_called_once_with(
            pool.packing_spec,
            'adaptive_reward',
            reset=True,
            decoder_layers=1,
            periodic_pair_features=False,
        )
        source.replace_stack.assert_called_once()
        self.assertEqual(metrics['omnifold/accepted'], 1.)
        self.assertEqual(metrics['omnifold/closure_evaluated'], 0.)
        self.assertNotIn('omnifold/candidate/residual_closure_auc', metrics)
        self.assertIn('closure not evaluated', metrics['omnifold/accept_reason'])
        for key in metrics:
            if key.startswith(
                ('omnifold/iteration1_monitor/', 'omnifold/weight_guard/')
            ):
                self.assertTrue(trainer._wandb_critical_keep(key))
                self.assertTrue(trainer._wandb_simplified_keep(key, metrics[key]))

    def test_low_ess_candidate_is_not_installed(self):
        cfg = replace(
            _config(
                monitor_mode='raw_plateau_refit',
                raw_audit_enabled=True,
                acceptance_audit_enabled=False,
            ),
            iteration_one_only=True,
            min_iterations=1,
            max_iterations=1,
            minimum_ess_fraction=.9,
        )
        spec = ratio.EventPackingSpec({
            'x': (2, 3),
            'x_mask': (2, 1),
            'conditions': (1, 2),
            'conditions_mask': (1,),
        })
        pool = ad.AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        source = SimpleNamespace(model_builder=object(), replace_stack=mock.Mock())
        stack = mock.Mock(spec=['to', 'eval', 'assert_frozen'])
        stack.to.return_value = stack
        stack.eval.return_value = stack
        diag = ratio.ResidualIterationDiagnostics(
            1,
            (SimpleNamespace(saturated=True, loss=.4, balanced_accuracy=.8),) * 2,
            .693,
            .45,
            .8,
            .8,
            .243,
            True,
        )
        concentrated = torch.cat((torch.tensor([10.]), torch.zeros(31)))
        result = SimpleNamespace(
            diagnostics=(diag,),
            iterations=1,
            train_log_weight=concentrated,
            validation_log_weight=concentrated,
        )
        with mock.patch.object(
            ad, 'fit_residual_ratio_stack', return_value=result
        ), mock.patch.object(
            ad.FrozenResidualRatioReward, 'from_fit_result', return_value=stack
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
                device=torch.device('cpu'),
                world_size=1,
                global_step=0,
            )
        self.assertEqual(metrics['omnifold/accepted'], 0.)
        self.assertEqual(metrics['omnifold/weight_guard/passed'], 0.)
        self.assertIn('failed ESS guard', metrics['omnifold/accept_reason'])
        source.replace_stack.assert_not_called()

    def test_yaml_changes_only_iteration_protocol_and_output_locations(self):
        root = Path(__file__).resolve().parents[4]/'config'
        old = yaml.safe_load((root/'dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_fullfit.yaml').read_text())
        new = yaml.safe_load((root/'dgpo_omnifold_ztautau_10pct_iteration1_only.yaml').read_text())
        expected = copy.deepcopy(old['dgpo'])
        expected['adaptive_omnifold']['recalibration'].update(
            iteration_one_only=True, max_iterations=1, residual_closure_schedule=[], tempering=1.)
        self.assertEqual(new['dgpo'], expected)
        cfg = ad.resolve_adaptive_config(new['dgpo'])
        self.assertTrue(cfg.iteration_one_only)
        self.assertEqual(cfg.tempering, 1.)
        self.assertEqual(cfg.warm_start_iterations, (1,))
        for key in ('platform','network','reward_config'):
            self.assertEqual(new[key], old[key])
        tr = new['options']['Training']
        self.assertEqual(tr['model_checkpoint_load_path'], old['options']['Training']['model_checkpoint_load_path'])
        self.assertIn('step=320.ckpt', tr['model_checkpoint_load_path'])
        self.assertNotEqual(tr['model_checkpoint_save_path'], old['options']['Training']['model_checkpoint_save_path'])
        self.assertEqual(new['dgpo']['checkpoint_load_mode'], 'weights_only')
        self.assertFalse(new['dgpo']['auto_resume_from_last'])
        for changes in ({'max_iterations':2}, {'residual_closure_schedule':[{'start_step':0,'max_auc':.55}]},
                        {'iteration_one_only':'true'}, {'acceptance_audit_enabled':True}):
            bad = copy.deepcopy(new['dgpo'])
            bad['adaptive_omnifold']['recalibration'].update(changes)
            with self.assertRaises(ValueError):
                ad.resolve_adaptive_config(bad)

    def test_dual_classifier_config_separates_reward_and_monitor(self):
        root = Path(__file__).resolve().parents[4] / 'config'
        overlay = yaml.safe_load(
            (
                root
                / 'dgpo_omnifold_ztautau_10pct_dual_classifier_t075_trust02_step260.yaml'
            ).read_text()
        )
        adaptive = overlay['dgpo']['adaptive_omnifold']
        cfg = ad.resolve_adaptive_config(overlay['dgpo'])
        self.assertFalse(cfg.iteration_one_only)
        self.assertEqual((cfg.min_iterations, cfg.max_iterations), (1, 10))
        self.assertEqual(cfg.residual_closure_schedule, ((0, .52),))
        self.assertEqual(cfg.log_ratio_clip, 2.5)
        self.assertEqual(cfg.minimum_ess_fraction, .3)
        self.assertTrue(cfg.adaptive_tempering_enabled)
        self.assertEqual(cfg.target_ess_fraction, .3)
        self.assertEqual(cfg.minimum_tempering, .1)
        self.assertEqual(cfg.tempering_grid_steps, 14)
        self.assertTrue(cfg.inherit_previous_tempering)
        self.assertTrue(cfg.ess_aware_checkpoint_selection)
        self.assertEqual(cfg.ess_aware_max_checkpoints, 16)
        self.assertTrue(cfg.ess_aware_first_residual_only)
        self.assertEqual(cfg.reward_classifier['decoder_layers'], 1)
        self.assertFalse(cfg.reward_classifier['periodic_pair_features'])
        self.assertTrue(adaptive['audit_fit']['periodic_pair_features'])
        self.assertTrue(adaptive['audit_fit']['visible_pair_rest_frame'])
        bad = copy.deepcopy(overlay['dgpo'])
        bad['adaptive_omnifold']['recalibration']['reward_classifier'][
            'unknown_feature'
        ] = True
        with self.assertRaisesRegex(ValueError, 'unsupported keys'):
            ad.resolve_adaptive_config(bad)


if __name__ == '__main__':
    unittest.main()
