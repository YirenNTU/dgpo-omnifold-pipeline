"""CPU transactions and AST guards for the sixteen-worker mechanism driver."""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import json
import logging
import math
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'evenet_dgpo'))
from RL.DGPO_neutrino import tau_reward_mechanisms as worker

# Use the actual production optimizer/scheduler wrapper without unrelated
# Lightning/vision/Ray trainer imports.
TRAINER = REPO / 'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'
TREE = ast.parse(TRAINER.read_text())
wrapper_node = next(node for node in TREE.body if isinstance(node, ast.ClassDef)
                    and node.name == '_DgpoOptimizerWithSchedule')
wrapper_namespace = dict(torch=torch, copy=copy, math=math, Mapping=Mapping,
                         Any=Any, _log=logging.getLogger(__name__))
exec(compile(ast.Module(body=[wrapper_node], type_ignores=[]), str(TRAINER), 'exec'), wrapper_namespace)
NativeOptimizer = wrapper_namespace['_DgpoOptimizerWithSchedule']


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(2, 2)
        self.unused = torch.nn.Parameter(torch.tensor([2.]))
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(.2)
        self.register_buffer('calls', torch.zeros(()))

    def forward(self, x):
        self.calls.add_(1)
        return self.dropout(self.bn(self.used(x)))


def optimizer_for(actor):
    base = torch.optim.AdamW(actor.parameters(), lr=.03, weight_decay=.1)
    schedule = torch.optim.lr_scheduler.LambdaLR(base, lambda step: 1 / (step + 1))
    return NativeOptimizer(base, schedule)


def nested_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            nested_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class NativeWorkerTests(unittest.TestCase):
    def actor_fixture(self):
        actor = TinyActor()
        optimizer = optimizer_for(actor)
        actor(torch.tensor([[1., 2.], [2., -1.], [3., .5]])).square().sum().backward()
        optimizer.step(); optimizer.scheduler_step()
        actor.bn.eval()
        actor.unused.grad = None
        return actor, optimizer

    def test_transaction_rolls_back_buffers_optimizer_scheduler_rng_modes_and_grad_on_error(self):
        actor, optimizer = self.actor_fixture()
        state = copy.deepcopy(actor.state_dict())
        optimization = copy.deepcopy(optimizer.state_dict())
        grads = [None if p.grad is None else p.grad.clone() for p in actor.parameters()]
        modes = [module.training for module in actor.modules()]
        torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        with self.assertRaisesRegex(RuntimeError, 'intentional rollback'):
            with worker.native_transaction(actor, optimizer, seed=77):
                actor.train()
                optimizer.zero_grad(set_to_none=True)
                actor(torch.randn(5, 2)).square().sum().backward()
                actor.unused.grad = torch.ones_like(actor.unused)
                optimizer.step(); optimizer.scheduler_step()
                optimizer.param_groups[0]['weight_decay'] = .9
                torch.rand(3); np.random.random(3); random.random()
                raise RuntimeError('intentional rollback')
        nested_equal(self, actor.state_dict(), state)
        nested_equal(self, optimizer.state_dict(), optimization)
        self.assertEqual([module.training for module in actor.modules()], modes)
        for p, grad in zip(actor.parameters(), grads):
            if grad is None:
                self.assertIsNone(p.grad)
            else:
                self.assertTrue(torch.equal(p.grad, grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for a, b in zip(np.random.get_state(), numpy_rng):
            np.testing.assert_equal(a, b)
        self.assertEqual(random.getstate(), python_rng)

    def test_adamw_proposal_is_actual_inherited_clipped_step_and_leaves_no_change(self):
        actor, optimizer = self.actor_fixture()
        parameters = tuple(actor.parameters())
        before_actor, before_optimization = copy.deepcopy(actor.state_dict()), copy.deepcopy(optimizer.state_dict())
        gradient = torch.linspace(-3, 2, sum(p.numel() for p in parameters))
        present = [p is not actor.unused for p in parameters]
        twin = copy.deepcopy(actor)
        twin_optimizer = optimizer_for(twin)
        twin_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        twin_parameters = tuple(twin.parameters())
        initial = worker.parameter_vector(twin_parameters)
        twin_optimizer.zero_grad(set_to_none=True)
        offset = 0
        for p, used in zip(twin_parameters, present):
            p.grad = gradient[offset:offset + p.numel()].reshape(p.shape).clone() if used else None
            offset += p.numel()
        torch.nn.utils.clip_grad_norm_(twin_parameters, .4, error_if_nonfinite=True)
        twin_optimizer.step()
        expected = worker.parameter_vector(twin_parameters) - initial
        proposed = worker.adamw_proposal(actor, optimizer, parameters, gradient, present, clip_norm=.4, seed=3)
        self.assertTrue(torch.equal(proposed, expected))
        self.assertGreater(float(proposed.norm()), 0)
        nested_equal(self, actor.state_dict(), before_actor)
        nested_equal(self, optimizer.state_dict(), before_optimization)
        self.assertEqual(optimizer.scheduler.last_epoch, before_optimization['scheduler']['last_epoch'])

    def test_normalized_signed_descent_and_invalid_displacement_are_safe(self):
        gradient = torch.tensor([3., 4., -5.])
        delta = worker.normalized_descent(gradient, 2e-5)
        self.assertAlmostEqual(float(delta.double().square().mean().sqrt()), 2e-5, places=11)
        self.assertLess(float(torch.dot(delta, gradient)), 0)
        self.assertGreater(float(torch.dot(-delta, gradient)), 0)
        self.assertIsNone(worker.normalized_descent(torch.zeros(3), 1e-5))
        for bad in (torch.tensor([float('nan')]), torch.tensor([float('inf')])):
            with self.assertRaises(ValueError):
                worker.normalized_descent(bad, 1e-5)
        actor, _ = self.actor_fixture()
        original = copy.deepcopy(actor.state_dict())
        for bad in (torch.zeros(1), torch.full((sum(p.numel() for p in actor.parameters()),), float('nan'))):
            with self.assertRaises(ValueError):
                worker.add_displacement(tuple(actor.parameters()), bad)
            nested_equal(self, actor.state_dict(), original)

    def test_matched_sampling_couples_all_rng_restores_rng_and_keeps_native_update(self):
        actor, optimizer = self.actor_fixture()
        original = worker.parameter_vector(tuple(actor.parameters()))
        scheduler_step = optimizer.scheduler.last_epoch
        torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        def draws():
            return torch.rand(4), np.random.random(4), [random.random() for _ in range(4)]
        with worker.matched_sampling(831):
            first = draws()
            optimizer.zero_grad(set_to_none=True)
            for parameter in actor.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step(); optimizer.scheduler_step()
        with worker.matched_sampling(831):
            same = draws()
        with worker.matched_sampling(832):
            different = draws()
        self.assertTrue(torch.equal(first[0], same[0]))
        np.testing.assert_array_equal(first[1], same[1])
        self.assertEqual(first[2], same[2])
        self.assertFalse(torch.equal(first[0], different[0]))
        self.assertFalse(np.array_equal(first[1], different[1]))
        self.assertNotEqual(first[2], different[2])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for a, b in zip(np.random.get_state(), numpy_rng):
            np.testing.assert_equal(a, b)
        self.assertEqual(random.getstate(), python_rng)
        self.assertFalse(torch.equal(worker.parameter_vector(tuple(actor.parameters())), original))
        self.assertEqual(optimizer.scheduler.last_epoch, scheduler_step + 1)
        self.assertTrue(all(parameter.grad is not None for parameter in actor.parameters()))

    def test_matched_sampling_restores_rng_after_exception(self):
        torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        with self.assertRaisesRegex(RuntimeError, 'sampling error'):
            with worker.matched_sampling(134):
                torch.rand(3); np.random.random(3); random.random()
                raise RuntimeError('sampling error')
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
        for a, b in zip(np.random.get_state(), numpy_rng):
            np.testing.assert_equal(a, b)
        self.assertEqual(random.getstate(), python_rng)

    def driver_fixture(self, directory, *, arm='inherited', mode='diagnose', native_step=None):
        actor, optimizer = self.actor_fixture()
        n = 6
        panel = dict(source_ids=np.arange(n), event_weight=np.ones(n), truth_deltas=np.zeros((n, 2, 2)),
                     visible_a=np.tile([2., 1., 0., 0.], (n, 1)),
                     visible_b=np.tile([2., 0., 1., 0.], (n, 1)))
        reference = copy.deepcopy(actor).eval().requires_grad_(False)
        head = torch.nn.Linear(2, 1).eval().requires_grad_(False)
        reward = SimpleNamespace(head=head, round_id=20, denominator_step=1880)
        cycle = SimpleNamespace(actor=actor, reference=reference, reward=reward, world=1, rank=0,
            device=torch.device('cpu'), validation=panel, cfg={'validation_seed': 123}, output=directory,
            emit=lambda *args: None)
        cfg = dict(allow_cpu_fixture=True, source_step=1920, expected_source_reward_round=20,
            expected_source_denominator_step=1880, reference_recenter=False, periodic_refits=False,
            mode=mode, reward_arm=arm, ensemble_directory=str(directory / 'ensemble'),
            output_directory=str(directory / 'diagnostic'), evaluation_events=n,
            bootstrap_replicates=10, bootstrap_seed=42, gradient_repeats=2,
            condition_groups=3, condition_edges=[-1., -.5, .5, 1.],
            noise_edges=[0., .7/3, 2*.7/3, .7], reconstruction_tolerance=1e-5,
            direction_rms_fractions=[1.])
        checkpoint = {'dgpo_optimizer_state_dict': copy.deepcopy(optimizer.state_dict())}
        step = native_step if native_step is not None else lambda *args, **kwargs: {'native': True}
        driver = worker.TauRewardMechanismDriver(cycle, cfg, native_step=step, optimizer=optimizer,
                                                 checkpoint=checkpoint, source_step=1920)
        return driver, cycle, optimizer, cfg, checkpoint

    def test_fresh_teacher_requires_artifact_and_bad_lifecycle_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for arm in ('member0', 'ensemble', 'all'):
                with self.assertRaisesRegex(ValueError, 'completed ensemble artifact'):
                    self.driver_fixture(directory, arm=arm)
            driver, cycle, optimizer, cfg, checkpoint = self.driver_fixture(directory)
            for field in ('reference_recenter', 'periodic_refits'):
                with self.assertRaisesRegex(ValueError, 'must not recenter'):
                    worker.TauRewardMechanismDriver(cycle, dict(cfg, **{field: True}), native_step=lambda: None,
                        optimizer=optimizer, checkpoint=checkpoint, source_step=1920)
            self.assertFalse(driver.prepare(191))
            self.assertEqual((cycle.reward.round_id, cycle.reward.denominator_step), (20, 1880))

    def test_saved_judge_scoring_keeps_separate_own_head_rewards(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _, _ = self.driver_fixture(Path(temporary))
            teacher = cycle.reward.head
            fifth = torch.nn.Linear(2, 1).eval().requires_grad_(False)
            @contextmanager
            def temporary_reward(reward, selection):
                self.assertEqual(selection, 'judge')
                reward.head = fifth
                try:
                    yield
                finally:
                    reward.head = teacher
            driver.ensemble = SimpleNamespace(temporary_reward=temporary_reward)
            def generate(panel, folder, k, seed, physics):
                self.assertEqual(k, 8)
                folder.mkdir(parents=True, exist_ok=True)
                return dict(rewards=np.ones((6, 8)), deltas=np.zeros((6, 8, 2, 2)), cij=np.zeros((6, 9)))
            cycle.generate = generate
            def score(generated, *args, **kwargs):
                self.assertIs(cycle.reward.head, fifth)
                generated['rewards'][:] = 7.
                return {}
            cycle._score_transfer_head = score
            mocked = ModuleType('RL.DGPO_neutrino.conditional_tau_cycle')
            mocked.cij_terms = lambda *args: np.zeros((6, 9))
            with patch.dict(sys.modules, {mocked.__name__: mocked}):
                _, result = driver._evaluate('shared-candidates')
            np.testing.assert_equal(result['training_reward'], 1.)
            np.testing.assert_equal(result['rewards'], 7.)
            self.assertIs(cycle.reward.head, teacher)
            with np.load(driver.output / 'shared-candidates/measurements.npz') as saved:
                np.testing.assert_equal(saved['training_reward'], 1.)
                np.testing.assert_equal(saved['reward'], 7.)

    def test_member0_trajectory_installs_single_head_without_reference_or_optimizer_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, optimizer, _, _ = self.driver_fixture(Path(temporary), mode='trajectory')
            driver.cfg.update(reward_arm='member0', updates=50)
            old_reference = copy.deepcopy(cycle.reference.state_dict())
            old_optimization = copy.deepcopy(optimizer.state_dict())
            calls = []
            def install(reward, selection, *, policy_step, epoch):
                self.assertEqual(selection, 'single')
                self.assertEqual((policy_step, epoch), (1920, 191))
                calls.append('install')
                reward.round_id += 1
                reward.denominator_step = policy_step
            driver.ensemble = SimpleNamespace(install_into_reward=install)
            class Observer:
                def __init__(self, observed_cycle, options, **kwargs):
                    self.options = options
                    self.assertions = kwargs
                    calls.append('observer')
                def start(self):
                    calls.append('baseline')
            with patch('RL.DGPO_neutrino.tau_reward_transfer.TauRewardTransferProbe', Observer):
                self.assertFalse(driver.prepare(191))
            self.assertEqual(calls, ['install', 'observer', 'baseline'])
            self.assertEqual(driver.observer.assertions['reward_round'], 21)
            self.assertTrue(driver.observer.options['mechanism_protocol'])
            nested_equal(self, cycle.reference.state_dict(), old_reference)
            nested_equal(self, optimizer.state_dict(), old_optimization)

    def test_trajectory_calls_native_step_with_paired_step_and_rank_seed_and_preserves_applied_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            def native(actor, reference, ema_rollout, ema_save, batch, optimizer, sampler, reward, **kwargs):
                calls.append(dict(kwargs=kwargs, batch=batch,
                    draws=(torch.rand(3), np.random.random(3), [random.random() for _ in range(3)])))
                optimizer.zero_grad(set_to_none=True)
                for parameter in actor.parameters():
                    parameter.grad = torch.ones_like(parameter)
                optimizer.step(); optimizer.scheduler_step()
                return {'train/optimizer_step_ran': 1., 'regular_native_metric': 17.}
            driver, cycle, optimizer, cfg, _ = self.driver_fixture(Path(temporary), mode='trajectory', native_step=native)
            batch = {'x': torch.ones(3, 2)}
            before = worker.parameter_vector(tuple(cycle.actor.parameters()))
            clock = optimizer.scheduler.last_epoch
            kwargs = dict(global_step=1920, reference_trust_coefficient=.5, grad_clip_norm=1., marker='unchanged')
            result = driver.native_step(cycle.actor, cycle.reference, None, None, batch, optimizer, None, cycle.reward, **kwargs)
            self.assertEqual(result, {'train/optimizer_step_ran': 1., 'regular_native_metric': 17.})
            self.assertEqual(calls[0]['kwargs'], kwargs)
            self.assertIs(calls[0]['batch'], batch)
            self.assertNotIn('mechanism_read_only', calls[0]['kwargs'])
            self.assertFalse(torch.equal(worker.parameter_vector(tuple(cycle.actor.parameters())), before))
            self.assertEqual(optimizer.scheduler.last_epoch, clock + 1)
            seed = cfg['bootstrap_seed'] + cycle.rank
            with worker.matched_sampling(seed):
                expected = (torch.rand(3), np.random.random(3), [random.random() for _ in range(3)])
            self.assertTrue(torch.equal(calls[0]['draws'][0], expected[0]))
            np.testing.assert_array_equal(calls[0]['draws'][1], expected[1])
            self.assertEqual(calls[0]['draws'][2], expected[2])
            driver.native_step(cycle.actor, cycle.reference, None, None, batch, optimizer, None, cycle.reward,
                               **dict(kwargs, global_step=1921))
            with worker.matched_sampling(seed + 100003):
                expected_next = (torch.rand(3), np.random.random(3), [random.random() for _ in range(3)])
            self.assertTrue(torch.equal(calls[1]['draws'][0], expected_next[0]))
            np.testing.assert_array_equal(calls[1]['draws'][1], expected_next[1])
            self.assertEqual(calls[1]['draws'][2], expected_next[2])
            cycle.rank = 1
            driver.native_step(cycle.actor, cycle.reference, None, None, batch, optimizer, None, cycle.reward, **kwargs)
            with worker.matched_sampling(cfg['bootstrap_seed'] + 1):
                expected_rank = torch.rand(3)
            self.assertTrue(torch.equal(calls[2]['draws'][0], expected_rank))

    def test_driver_training_iterator_uses_replay_copies_without_live_ray(self):
        from RL.DGPO_neutrino.tau_native_batch_replay import NativeTrainingReplay
        with tempfile.TemporaryDirectory() as temporary:
            driver, _, _, _, _ = self.driver_fixture(Path(temporary))
            stored = {'x': torch.tensor([[1., 2.]]), 'conditions': {'mask': torch.tensor([True])}}
            driver.replay = NativeTrainingReplay({}, {'batches': [stored], 'rank': 0, 'world': 1})
            class RayShard:
                def iter_torch_batches(self, **kwargs):
                    raise AssertionError('Replay must avoid the live Ray iterator')
            iterator = driver.training_iterator(RayShard(), {'batch_size': 512})
            first = next(iterator)
            first['x'].fill_(99)
            first['conditions']['mask'].fill_(False)
            second = next(iterator)
            self.assertFalse(first['x'].data_ptr() == second['x'].data_ptr())
            self.assertTrue(torch.equal(second['x'], torch.tensor([[1., 2.]])))
            self.assertTrue(bool(second['conditions']['mask'][0]))
            self.assertTrue(torch.equal(stored['x'], second['x']))
            renewed = driver.training_iterator(RayShard(), {})
            self.assertTrue(torch.equal(next(renewed)['x'], stored['x']))
            self.assertEqual(driver.replay.next_update, 3)

    def test_driver_reuses_candidates_replica0_gradient_presence_and_restores_actor(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            def step(*args, **kwargs):
                actor, optimizer = args[0], args[5]
                trace = kwargs['mechanism_capture']
                parameters = tuple(actor.parameters())
                width = sum(p.numel() for p in parameters)
                replica = len(calls)
                candidates = torch.arange(8 * 3 * 4.).reshape(8, 3, 2, 2) if replica == 0 else kwargs['mechanism_candidates']
                if replica == 0:
                    self.assertIsNone(kwargs['mechanism_candidates'])
                else:
                    self.assertIs(kwargs['mechanism_candidates'], calls[0]['candidates'])
                reward = torch.ones(width) * (replica + 1)
                reference = torch.ones(width) * -.1
                trace.add_flat('visible_middle', 'mid_t', reward=reward, reference=reference)
                trace.reduce_(world_size=1)
                total = reward + kwargs['reference_trust_coefficient'] * reference
                present = [True] * len(parameters) if replica == 0 else [False] * len(parameters)
                calls.append(dict(candidates=candidates, total=total, present=present, read_only=kwargs['mechanism_read_only']))
                # Deliberate mutations prove the driver transaction, not this
                # fixture's step, protects model/optimizer state.
                with torch.no_grad():
                    actor.used.weight.add_(2.)
                    actor.calls.add_(5.)
                optimizer.zero_grad(set_to_none=True)
                return dict(mechanism_candidates=candidates, mechanism_vectors={'actual_unclipped': total},
                            mechanism_gradient_present=present)
            driver, cycle, optimizer, _, _ = self.driver_fixture(Path(temporary), native_step=step)
            original_actor, original_optimizer = copy.deepcopy(cycle.actor.state_dict()), copy.deepcopy(optimizer.state_dict())
            baseline = dict(rewards=np.zeros((6, 8)), training_reward=np.zeros((6, 8)),
                cij=np.zeros((6, 9)), truth_cij=np.zeros((6, 9)), deltas=np.zeros((6, 8, 2, 2)))
            driver._evaluate = lambda name: (cycle.validation, baseline)
            driver._compare = lambda *args: {'reward': {'delta_mean': 0.}}
            proposals = []
            def proposal(actor, opt, params, gradient, present, **kwargs):
                proposals.append((gradient.clone(), list(present)))
                return torch.ones_like(gradient) * 1e-5
            with patch.object(worker, 'adamw_proposal', proposal):
                result = driver.native_step(cycle.actor, cycle.reference, None, None, {}, optimizer, None, None,
                                            reference_trust_coefficient=.5, grad_clip_norm=1.)
            self.assertTrue(result['_tau_mechanism_diagnosis_complete'])
            self.assertEqual(result['train/optimizer_step_ran'], 0.)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call['read_only'] for call in calls))
            self.assertTrue(torch.equal(proposals[0][0], calls[0]['total']))
            self.assertEqual(proposals[0][1], calls[0]['present'])
            self.assertEqual(proposals[1][1], calls[0]['present'])
            nested_equal(self, cycle.actor.state_dict(), original_actor)
            nested_equal(self, optimizer.state_dict(), original_optimizer)
            report = json.loads((driver.output / 'gradient_diagnosis.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['actual_updates'], 0)
            self.assertFalse(report['comparable_across_arms'])
            self.assertIn('NOT function-distance matched', report['scope'])

    def test_offrank_zero_evaluation_only_requires_feature_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _, _ = self.driver_fixture(Path(temporary))
            cycle.rank = 1
            @contextmanager
            def temporary_reward(*args):
                yield
            driver.ensemble = SimpleNamespace(temporary_reward=temporary_reward)
            cycle.generate = lambda *args: {'features': np.ones((6, 2))}
            def score(generated, *args, **kwargs):
                self.assertEqual(set(generated), {'features'})
                return None
            cycle._score_transfer_head = score
            panel, generated = driver._evaluate('offrank')
            self.assertEqual(len(panel['source_ids']), 6)
            self.assertEqual(set(generated), {'features', 'rewards'})
            self.assertIsNone(generated['rewards'])

    def test_native_read_only_return_precedes_clipping_adamw_scheduler_and_ema(self):
        train = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'train_step')
        guard = next(node for node in train.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name) and node.test.id == 'mechanism_read_only')
        self.assertTrue(any(isinstance(node, ast.Return) for node in guard.body))
        guard_source = ast.unparse(guard)
        self.assertIn('train/optimizer_step_ran', guard_source)
        self.assertIn('0.0', guard_source)
        self.assertIn('mechanism_candidates', guard_source)
        self.assertIn('mechanism_capture.reduce_', guard_source)
        for node in ast.walk(train):
            if not isinstance(node, ast.Call):
                continue
            called = ast.unparse(node.func)
            if called in {'optimizer.step', 'optimizer.scheduler_step', '_grad_norm_pre_clip_and_clip_active', 'ema_rollout.update', 'ema_save.update'}:
                self.assertGreater(node.lineno, guard.end_lineno)
        capture = next(node for node in ast.walk(train) if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == 'mechanism_capture.capture_call')
        self.assertLess(capture.lineno, guard.lineno)
        # Production captures share the existing no_sync context, retaining
        # ordinary backward and reducing detached traces only afterward.
        contexts = [node for node in ast.walk(train) if isinstance(node, ast.IfExp)
                    and isinstance(node.body, ast.Call) and ast.unparse(node.body.func) == 'model.no_sync']
        def native_no_sync_or(node):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                return False
            names = {term.id for term in node.values if isinstance(term, ast.Name)}
            has_last_guard = any(isinstance(term, ast.UnaryOp) and isinstance(term.op, ast.Not)
                                 and isinstance(term.operand, ast.Name) and term.operand.id == 'is_last_backward'
                                 for term in node.values)
            return {'endpoint_active', 'gradient_transfer_active'} <= names and has_last_guard
        self.assertTrue(any(any(native_no_sync_or(node) for node in ast.walk(context.test)) for context in contexts))
        mechanism_source = Path(worker.__file__).read_text()
        self.assertNotIn('prepare_reward_transfer(', mechanism_source)
        self.assertNotIn('cycle.refit(', mechanism_source)
        complete = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Call) and ast.unparse(node.test.func) == 'metrics.get'
                        and any(isinstance(arg, ast.Constant) and arg.value == '_tau_mechanism_diagnosis_complete' for arg in node.test.args))
        self.assertTrue(any(isinstance(node, ast.Return) for node in complete.body))
        self.assertNotIn('_dgpo_save_last_ckpt', ast.unparse(complete))

    def test_stage_caps_match_launcher_before_prepare(self):
        cap = next(node.value for node in ast.walk(TREE) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'expected_cap' for target in node.targets))
        expression = compile(ast.Expression(cap), str(TRAINER), 'eval')
        for mode, expected in (('ensemble', 1920), ('diagnose', 1921), ('trajectory', 1970)):
            actual = eval(expression, {'global_step': 1920, 'mode': mode, 'tau_mechanism_cfg': {'updates': 50}})
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
