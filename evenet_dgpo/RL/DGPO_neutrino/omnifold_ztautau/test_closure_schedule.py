import ast
import copy
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import yaml

from . import adaptive as a
from .evenet_ratio import EventPackingSpec
from .test_adaptive import _config
from .. import dgpo_trainer as trainer


SCHEDULE = ((0, .55), (100, .53), (300, .52), (600, .51))


class TestClosureSchedule(unittest.TestCase):
    def test_fixed_budget_installs_two_useful_increments_without_closure(self):
        cfg = replace(_config(acceptance_audit_enabled=False),
                      fixed_iteration_budget=True, min_iterations=2, max_iterations=2)
        spec = EventPackingSpec({'x':(2,3),'x_mask':(2,1),'conditions':(1,2),'conditions_mask':(1,)})
        pool = a.AdaptiveOmniFoldPool(packed_event=torch.randn(32,spec.width),
            truth=torch.randn(32,4), candidates=torch.randn(32,1,4), packing_spec=spec)
        result = SimpleNamespace(diagnostics=tuple(
            SimpleNamespace(saturated=True, validation_auc=auc, accepted=True)
            for auc in (.8, .646)), iterations=2)
        stack = mock.Mock(assert_frozen=mock.Mock())
        stack.to.return_value = stack
        stack.eval.return_value = stack
        source = SimpleNamespace(model_builder=object(), is_installed=True, replace_stack=mock.Mock())
        reference = torch.nn.Linear(2,2)
        with mock.patch.object(a, 'fit_residual_ratio_stack', return_value=result) as fit, \
             mock.patch.object(a.FrozenResidualRatioReward, 'from_fit_result', return_value=stack):
            metrics = a.run_adaptive_refit(state=a.AdaptiveOmniFoldState(), cfg=cfg,
                reward_source=source, round_ref_model=reference,
                policy_snapshot_state_dict=copy.deepcopy(reference.state_dict()),
                fit_pool=pool, score_pool=pool, epoch=-1, global_step=0,
                device=torch.device('cpu'), world_size=1, enforce_round_acceptance=False)
        self.assertTrue(fit.call_args.kwargs['fixed_iteration_budget'])
        self.assertEqual(metrics['omnifold/accepted'], 1.)
        self.assertEqual(metrics['omnifold/closure_evaluated'], 0.)
        self.assertNotIn('omnifold/candidate/residual_closure_auc', metrics)
        source.replace_stack.assert_called_once()

    def test_boundaries_saved_step_and_legacy(self):
        cfg = replace(_config(residual_min_auc_gain=.01), residual_closure_schedule=SCHEDULE)
        for step, expected in ((0,.55),(99,.55),(100,.53),(299,.53),(300,.52),(599,.52),(600,.51),(1500,.51)):
            self.assertEqual(a.residual_closure_auc_limit(cfg, global_step=step), expected)
        # The caller passes the restored DGPO clock, not imported source-policy age.
        saved_checkpoint = {'dgpo_global_step': 320}
        self.assertEqual(a.residual_closure_auc_limit(cfg, global_step=saved_checkpoint['dgpo_global_step']), .52)
        self.assertEqual(a.residual_closure_auc_limit(cfg, global_step=0), .55)
        for invalid in (None, -1, True, 1.5):
            with self.assertRaises(ValueError):
                a.residual_closure_auc_limit(cfg, global_step=invalid)
        self.assertEqual(a.residual_closure_auc_limit(_config(residual_min_auc_gain=.01)), .51)

    def test_yaml_and_invalid_schedules(self):
        path = Path(__file__).resolve().parents[4] / 'config/dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_fullfit.yaml'
        data = yaml.safe_load(path.read_text())['dgpo']
        cfg = a.resolve_adaptive_config(data)
        self.assertEqual(cfg.residual_closure_schedule, SCHEDULE)
        for bad in ('bad', [{'start_step':1,'max_auc':.55}],
                    [{'start_step':0,'max_auc':.5}], [{'start_step':0,'max_auc':float('nan')}],
                    [{'start_step':True,'max_auc':.55}], [{'start_step':0,'max_auc':True}],
                    [{'start_step':0,'max_auc':.55},{'start_step':0,'max_auc':.53}],
                    [{'start_step':0,'max_auc':.55},{'start_step':1,'max_auc':.56}]):
            modified = copy.deepcopy(data)
            modified['adaptive_omnifold']['recalibration']['residual_closure_schedule'] = bad
            with self.assertRaises(ValueError):
                a.resolve_adaptive_config(modified)

    def test_same_limit_reaches_fit_and_install_and_is_logged_on_rejection(self):
        cfg = replace(_config(acceptance_audit_enabled=False, residual_min_auc_gain=.01),
                      residual_closure_schedule=SCHEDULE)
        spec = EventPackingSpec({'x':(2,3),'x_mask':(2,1),'conditions':(1,2),'conditions_mask':(1,)})
        pool = a.AdaptiveOmniFoldPool(packed_event=torch.randn(32,spec.width),
            truth=torch.randn(32,4), candidates=torch.randn(32,1,4), packing_spec=spec)
        result = SimpleNamespace(diagnostics=(SimpleNamespace(saturated=True, validation_auc=.8),
                                              SimpleNamespace(saturated=True, validation_auc=.54)), iterations=1)
        stack = mock.Mock(assert_frozen=mock.Mock())
        stack.to.return_value = stack
        stack.eval.return_value = stack
        for step, expected, failure in ((30,True,False),(100,False,False),(300,False,True)):
            source = SimpleNamespace(model_builder=object(), is_installed=True, replace_stack=mock.Mock())
            reference = torch.nn.Linear(2,2)
            before = copy.deepcopy(reference.state_dict())
            snapshot = copy.deepcopy(torch.nn.Linear(2,2).state_dict())
            with mock.patch.object(a,'fit_residual_ratio_stack',return_value=result,
                    side_effect=RuntimeError('did not produce a held-out no-op') if failure else None) as fit, \
                 mock.patch.object(a.FrozenResidualRatioReward,'from_fit_result',return_value=stack):
                metrics = a.run_adaptive_refit(state=a.AdaptiveOmniFoldState(), cfg=cfg,
                    reward_source=source, round_ref_model=reference, policy_snapshot_state_dict=snapshot,
                    fit_pool=pool,score_pool=pool,epoch=2,global_step=step,
                    device=torch.device('cpu'),world_size=1,enforce_round_acceptance=False)
            limit = a.residual_closure_auc_limit(cfg,global_step=step)
            self.assertAlmostEqual(fit.call_args.kwargs['residual_min_auc_gain'],limit-.5)
            self.assertEqual(metrics['omnifold/residual_closure_auc_limit'],limit)
            self.assertEqual(metrics['omnifold/refit_global_step'],step)
            self.assertEqual(metrics['omnifold/accepted'],float(expected))
            self.assertEqual(source.replace_stack.called,expected)
            for k,v in reference.state_dict().items():
                torch.testing.assert_close(v,(snapshot if expected else before)[k])

    def test_all_trainer_refit_paths_forward_clock_and_metrics_survive_filter(self):
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(trainer)))
                 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_adaptive_refit']
        self.assertEqual(len(calls),4)
        for call in calls:
            value = next(k.value for k in call.keywords if k.arg=='global_step')
            self.assertEqual(ast.unparse(value),'int(global_step)')
        for key in ('omnifold/residual_closure_auc_limit','omnifold/refit_global_step',
                    'omnifold/residual_closure_schedule_enabled'):
            self.assertTrue(trainer._wandb_critical_keep(key))


if __name__ == '__main__':
    unittest.main()
