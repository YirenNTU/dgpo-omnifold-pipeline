"""Same-scope/same-LR residual fits and forward-only refit failure handling."""
import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

from . import evenet_ratio as ratio
from .adaptive import resolve_adaptive_config
from .ratio_fit import RatioFitConfig
from .test_rest_frame import batch, classifier
from .test_ztautau_omnifold import _FakeZtautauBackbone
from .. import dgpo_trainer as trainer


class TestFullFitResidual(unittest.TestCase):
    def test_all_iterations_keep_trainable_scope_lr_and_complete_first_fold_state(self):
        torch.manual_seed(42)
        data = {k: v.repeat(20, *([1] * (v.ndim - 1))) for k, v in batch().items()}
        data['conditions'] = torch.randn_like(data['conditions'])
        packed, spec = ratio.pack_event_inputs(data, include_pairwise_context=True,
                                               include_visible_pair_rest_frame=True)
        samples = torch.randn(len(packed), 4)
        template = _FakeZtautauBackbone()
        factory = lambda: classifier(spec, template=template, layers=2, dropout=.15)
        expected = {n for n, p in factory().named_parameters() if p.requires_grad}
        cfg = RatioFitConfig(steps=None, min_steps=1, batch_size=4,
            drop_last_batch=True, sampling='independent_epoch_shuffle',
            learning_rate=2e-4, backbone_learning_rate=1e-5, weight_decay=.0005,
            gradient_clip_norm=5., require_saturation=True)
        previous = None
        for round_index in range(2):
            first, calls = {}, []
            def fit(model, dc, ds, dw, gc, gs, gw, fc, *args, **kwargs):
                iteration, fold = len(calls) // 2 + 1, len(calls) % 2 + 1
                calls.append((iteration, fold))
                model.train()
                self.assertFalse(getattr(model, '_residual_last_block_only', False))
                self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad}, expected)
                self.assertTrue(all(b.training for b in model.bank.decoder.blocks))
                self.assertTrue(all(p.requires_grad for p in model.backbone.GroupedSequentialEmbedding.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.backbone.InvisibleInputProjector.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.backbone.PET.adapters.parameters()))
                self.assertEqual((fc.learning_rate, fc.backbone_learning_rate, fc.decoder_learning_rate),
                                 (2e-4, 1e-5, None))
                self.assertEqual((fc.weight_decay, fc.gradient_clip_norm), (.0005, 5.))
                cold = round_index == 0 and iteration == 1
                self.assertEqual(fc.min_steps, 1000 if cold else 10 * (len(dc) // 4))
                inherited = (first[fold] if iteration > 1 else
                             next((x['state'] for x in previous['models'] if x['fold'] == fold), None)
                             if previous else None)
                if inherited is not None:
                    for key,value in model.state_dict().items():
                        torch.testing.assert_close(value, inherited[key], atol=0, rtol=0)
                    self.assertGreater(model.bank.output.bias.abs().sum().item(), 0.)
                with torch.no_grad():
                    for p in model.parameters():
                        if p.requires_grad:
                            p.add_(.01 * len(calls))
                if iteration == 1:
                    first[fold] = copy.deepcopy(model.state_dict())
                return SimpleNamespace(saturated=True, loss=.6, balanced_accuracy=.7,
                                       steps_completed=fc.min_steps)
            with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio', side_effect=fit), \
                 mock.patch.object(ratio, 'fit_staged_residual_classifier', side_effect=AssertionError('staging must be disabled')), \
                 mock.patch.object(ratio, '_weighted_binary_score_metrics',
                                   side_effect=[(.6,.7,.7),(.65,.6,.6),(.693,.5,.5)]):
                result = ratio.fit_residual_ratio_stack(model_factory=factory,
                    data_condition=packed, data_sample=samples, gen_condition=packed, gen_sample=samples,
                    iterations=3, fit_config=cfg, tempering=1., seed=42,
                    warm_start_iterations=(1,), warm_start_from_iteration_one=True,
                    warm_start_state=previous, later_iteration_train_mode='full',
                    min_steps_per_fold=1000, warm_start_min_epochs_per_fold=10,
                    validation_interval_epochs=2., validation_patience_epochs=10.,
                    validation_data_condition=packed, validation_data_sample=samples,
                    validation_gen_condition=packed, validation_gen_sample=samples)
            self.assertEqual(calls, [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)])
            previous = result.warm_start_state
            self.assertEqual(len(previous['models']), 2)
            for saved in previous['models']:
                self.assertEqual(saved['iteration'], 1)
                for k,v in saved['state'].items():
                    torch.testing.assert_close(v, first[saved['fold']][k], atol=0, rtol=0)

    def test_yaml_contract_and_no_other_algorithm_changes(self):
        directory = Path(__file__).resolve().parents[4] / 'config'
        old = yaml.safe_load((directory / 'dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_staged.yaml').read_text())
        new = yaml.safe_load((directory / 'dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_fullfit.yaml').read_text())
        expected = copy.deepcopy(old['dgpo'])
        expected['adaptive_omnifold']['recalibration']['tempering'] = .75
        fit = expected['adaptive_omnifold']['recalibration']['fit']
        fit['later_iteration_train_mode'] = 'full'
        del fit['later_iteration_learning_rate'], fit['later_iteration_decoder_learning_rate']
        expected['adaptive_omnifold']['recalibration']['residual_closure_schedule'] = [
            {'start_step':s,'max_auc':v} for s,v in ((0,.55),(100,.53),(300,.52),(600,.51))]
        self.assertEqual(new['dgpo'], expected)
        self.assertEqual(new['options']['Training']['model_checkpoint_load_path'], old['options']['Training']['model_checkpoint_load_path'])
        self.assertNotEqual(new['options']['Training']['model_checkpoint_save_path'], old['options']['Training']['model_checkpoint_save_path'])
        for key in ('platform', 'network', 'reward_config'):
            self.assertEqual(new[key], old[key])
        cfg = resolve_adaptive_config(new['dgpo'])
        self.assertFalse(cfg.raw_rollback_to_best_on_plateau)
        self.assertEqual(cfg.residual_min_auc_gain, .01)
        self.assertEqual(cfg.tempering, .75)
        self.assertEqual(cfg.fit['later_iteration_train_mode'], 'full')
        self.assertNotIn('later_iteration_learning_rate', cfg.fit)

    def test_rejected_refit_checks_actual_rollback_not_global_best_scope(self):
        self.assertEqual(trainer._raw_refit_failure_decision(rollback_applied=False), 'forward_recenter_deferred')
        with self.assertRaisesRegex(RuntimeError, 'after policy rollback'):
            trainer._raw_refit_failure_decision(rollback_applied=True)
        tree = ast.parse(inspect.getsource(trainer))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == '_raw_refit_failure_decision']
        self.assertEqual(len(calls), 1)
        self.assertIn('staleness/raw_best_rollback_applied', ast.unparse(calls[0]))
        self.assertNotIn('raw_best_scope', ast.unparse(calls[0]))

    def test_critical_logging_keeps_stage_decisions_and_refit_failure_reason(self):
        for suffix in ('fit_stage', 'selected_stage', 'stage_a_validation_loss', 'stage_b_validation_loss', 'accepted'):
            key = 'omnifold_live/residual_reward/' + suffix
            self.assertTrue(trainer._wandb_critical_keep(key))
            self.assertTrue(trainer._wandb_simplified_keep(key, 1.))
        for key in ('omnifold/accept_reason', 'omnifold/recalibrations_rejected',
                    'staleness/decision', 'staleness/raw_best_rollback_applied'):
            self.assertTrue(trainer._wandb_critical_keep(key))


if __name__ == '__main__':
    unittest.main()
