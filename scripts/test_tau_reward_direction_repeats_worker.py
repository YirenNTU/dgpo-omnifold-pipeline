"""CPU checks for repeated, rollback-only native direction measurements."""
from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'evenet_dgpo'))
from RL.DGPO_neutrino import tau_reward_mechanisms as worker
from scripts import test_tau_reward_mechanisms_worker as legacy_fixtures
nested_equal, TRAINER = legacy_fixtures.nested_equal, legacy_fixtures.TRAINER


class DirectionRepeatsTests(unittest.TestCase):
    def fixture(self, directory, native_step=None):
        driver, cycle, optimizer, cfg, checkpoint = legacy_fixtures.NativeWorkerTests().driver_fixture(directory)
        cycle.validation['source_ids'] = np.array([f'1:{100+i}' for i in range(6)])
        cfg.update(mode='repeat_diagnose', draw_repeats=4, gradient_repeats=2,
                   evaluation_seeds=[202610041, 202610042], direction_rms_fractions=[.01, .03, .1],
                   updates=0, persistent_updates=0)
        cfg.pop('ensemble_directory')
        driver = worker.TauRewardMechanismDriver(cycle, cfg, native_step=native_step or driver.step,
                                                optimizer=optimizer, checkpoint=checkpoint, source_step=1920)
        return driver, cycle, optimizer, cfg

    @staticmethod
    def batch(draw, rows=3):
        return dict(x=torch.full((rows, 2), float(draw)),
                    source_sample_index=torch.zeros(rows, dtype=torch.int64),
                    source_event_key=torch.arange(draw*rows, (draw+1)*rows, dtype=torch.int64))

    def test_four_fresh_native_draws_share_only_within_replica_and_restore_everything(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls, evaluations, proposals, normalized, emitted = [], [], [], [], []
            def native(*args, **kwargs):
                actor, batch, optimizer = args[0], args[4], args[5]
                parameters = tuple(p for p in actor.parameters() if p.requires_grad)
                width = sum(p.numel() for p in parameters)
                incoming = kwargs['mechanism_candidates']
                candidates = torch.rand(8, len(batch['x']), 2, 2) if incoming is None else incoming
                reward = torch.rand(width) + .1
                reference = torch.full((width,), -.05)
                trace = kwargs['mechanism_capture']
                trace.add_flat('visible_middle', 'mid_t', reward=reward, reference=reference)
                trace.reduce_(world_size=1)
                total = reward + kwargs['reference_trust_coefficient']*reference
                calls.append(dict(incoming=incoming, candidates=candidates, total=total.clone(), reward=reward.clone(),
                                  global_step=kwargs['global_step'], batch=batch, read_only=kwargs['mechanism_read_only']))
                # Deliberate fixture mutations must not escape the transaction.
                optimizer.zero_grad(set_to_none=True)
                for p in parameters:
                    p.grad = torch.ones_like(p)
                optimizer.step(); optimizer.scheduler_step()
                actor.calls.add_(1.)
                actor.train()
                return dict(mechanism_candidates=candidates, mechanism_vectors={'actual_unclipped': total},
                            mechanism_gradient_present=[True] * len(parameters))
            driver, cycle, optimizer, cfg = self.fixture(Path(temporary), native)
            cycle.emit = lambda payload, step: emitted.append((payload, step))
            original_actor, original_optimization = copy.deepcopy(cycle.actor.state_dict()), copy.deepcopy(optimizer.state_dict())
            original_modes = [module.training for module in cycle.actor.modules()]
            original_grads = [None if p.grad is None else p.grad.clone() for p in cycle.actor.parameters()]
            original_torch, original_numpy = torch.get_rng_state().clone(), np.random.get_state()
            parameters = tuple(cycle.actor.parameters())
            origin = worker.parameter_vector(parameters)
            def evaluate(name, seed=None):
                self.assertIn(seed, cfg['evaluation_seeds'])
                evaluations.append((name, seed))
                delta = (worker.parameter_vector(parameters)-origin).double().sum().item()
                noise = np.random.default_rng(seed).normal(0, .01, (6, 8, 2, 2))
                generated = dict(deltas=noise+delta, rewards=(noise+delta).sum(axis=(-1, -2)),
                                 training_reward=(noise+delta).sum(axis=(-1, -2)),
                                 cij=np.zeros((6, 9)), truth_cij=np.zeros((6, 9)))
                return cycle.validation, generated
            driver._evaluate = evaluate
            original_proposal = worker.adamw_proposal
            original_descent = worker.normalized_descent
            def proposal(*args, **kwargs):
                proposals.append(args[3].clone())
                return original_proposal(*args, **kwargs)
            def descent(gradient, rms):
                normalized.append(gradient.clone())
                return original_descent(gradient, rms)
            results = []
            with patch.object(worker, 'adamw_proposal', proposal), patch.object(worker, 'normalized_descent', descent):
                for draw in range(4):
                    results.append(driver.native_step(cycle.actor, cycle.reference, None, None, self.batch(draw),
                        optimizer, None, cycle.reward, global_step=1920,
                        reference_trust_coefficient=.5, grad_clip_norm=1.))
                    nested_equal(self, cycle.actor.state_dict(), original_actor)
                    nested_equal(self, optimizer.state_dict(), original_optimization)
                    self.assertEqual([module.training for module in cycle.actor.modules()], original_modes)
                    for p, grad in zip(parameters, original_grads):
                        self.assertIsNone(p.grad) if grad is None else self.assertTrue(torch.equal(p.grad, grad))
            self.assertTrue(all(result.get('_tau_mechanism_read_only_pending') for result in results[:3]))
            self.assertTrue(results[3]['_tau_mechanism_diagnosis_complete'])
            self.assertTrue(all(result['train/optimizer_step_ran'] == 0 for result in results))
            self.assertTrue(torch.equal(torch.get_rng_state(), original_torch))
            for a, b in zip(np.random.get_state(), original_numpy):
                np.testing.assert_equal(a, b)
            self.assertEqual(len(calls), 8)
            for draw in range(4):
                first, second = calls[draw*2:draw*2+2]
                self.assertIsNone(first['incoming'])
                self.assertIs(second['incoming'], first['candidates'])
                self.assertTrue(torch.equal(proposals[draw], first['total']))
                self.assertTrue(torch.equal(normalized[draw*2], first['reward']))
                self.assertTrue(torch.equal(normalized[draw*2+1], first['total']))
                self.assertEqual((first['global_step'], second['global_step']), (1920, 1920))
                self.assertTrue(first['read_only'] and second['read_only'])
            self.assertFalse(torch.equal(calls[0]['candidates'], calls[2]['candidates']))
            self.assertEqual(sum('/baseline/' in name for name, _ in evaluations), 2)
            self.assertEqual(sum('/zero-replay/' in name for name, _ in evaluations), 2)
            self.assertEqual(len(evaluations), 4+4*3*3*2*2)
            report = json.loads((driver.output/'local_direction_repeats.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['completed_draws'], 4)
            self.assertEqual(report['actual_updates'], 0)
            self.assertEqual(report['evaluator'], 'installed_training_head')
            self.assertEqual(report['draws']['0']['native_batch']['global_event_count'], 3)
            self.assertEqual(report['draws']['0']['direction_gradient_replica'], 0)
            self.assertEqual(len(report['draws']['0']['interventions']), 18)
            for direction, matrix in report['cross_draw_direction_cosines'].items():
                self.assertIn(direction, ('raw_reward', 'raw_total', 'native_adamw'))
                self.assertEqual(len(matrix), 4)
                for i in range(4):
                    self.assertAlmostEqual(matrix[str(i)][str(i)], 1.)
            for intervention in report['draws']['0']['interventions'].values():
                self.assertEqual(set(intervention['evaluations']), set(map(str, cfg['evaluation_seeds'])))
                for value in intervention['evaluations'].values():
                    self.assertNotIn('condition_transfer', value)
                    self.assertIn('measurement_valid', value)
                    self.assertIn('realized_requested_cosine', value['motion'])
            self.assertEqual(len(emitted), 4)
            self.assertEqual([payload['tau/local_direction/draw_index'] for payload, _ in emitted], [0., 1., 2., 3.])
            self.assertTrue(all('tau/local_direction/raw_reward/rms0.01/seed202610041/plus_gain' in payload for payload, _ in emitted))
            self.assertTrue(all(len(payload) < 160 for payload, _ in emitted))
            self.assertTrue(all('tau/local_direction/raw_total/cosine_to_draw0' in payload for payload, _ in emitted))
            self.assertTrue(all('tau/local_direction/fixed_candidate_noise_cosine_reward' in payload for payload, _ in emitted))
            self.assertTrue(all('tau/local_direction/reconstruction_relative_error_max' in payload for payload, _ in emitted))
            self.assertNotIn('tau/local_direction/raw_reward/min_previous_draw_cosine', emitted[0][0])
            self.assertIn('tau/local_direction/raw_reward/min_previous_draw_cosine', emitted[1][0])
            self.assertFalse(driver.repeat_baselines)
            self.assertFalse(driver.repeat_direction_vectors)

    def test_identifiers_preserve_precision_reject_overlap_and_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            batch = self.batch(0)
            batch['source_event_key'] += 2**55
            recorded = driver._repeat_batch_identity(batch)
            self.assertIn(str(2**55), recorded['ranks']['0']['source_ids'][0])
            with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
                driver._repeat_batch_identity(batch)
            malformed = self.batch(1)
            malformed['source_event_key'] = malformed['source_event_key'].float()
            with self.assertRaisesRegex(ValueError, 'integer precision'):
                driver._repeat_batch_identity(malformed)
            validation = self.batch(3)
            validation['source_sample_index'].fill_(1)
            validation['source_event_key'] = torch.arange(100, 103)
            with self.assertRaisesRegex(ValueError, 'validation identities'):
                driver._repeat_batch_identity(validation)

    @staticmethod
    def full_identity_batch(start=10, rows=3):
        # Real parquet IDs may share both event_key AND file_index. Only the
        # entire four-column key distinguishes these native event rows.
        return dict(x=torch.zeros(rows, 2),
                    source_sample_index=torch.ones(rows, dtype=torch.int64),
                    source_event_key=torch.full((rows,), 8186255, dtype=torch.int64),
                    source_file_index=torch.zeros(rows, dtype=torch.int64),
                    source_event_index=torch.arange(start, start+rows, dtype=torch.int64))

    def test_full_ids_allow_same_file_base_collisions_and_false_holdout_prefix_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            cycle.validation['source_ids'] = np.array([f'1:8186255:0:{100+i}' for i in range(6)])
            first = driver._repeat_batch_identity(self.full_identity_batch())
            second = driver._repeat_batch_identity(self.full_identity_batch(start=13))
            self.assertEqual(first['global_unique_event_count'], 3)
            self.assertEqual(second['global_unique_event_count'], 3)
            self.assertEqual(first['ranks']['0']['source_ids'],
                             ['1:8186255:0:10', '1:8186255:0:11', '1:8186255:0:12'])
            self.assertEqual(first['ranks']['0']['source_id_columns'],
                             ['source_sample_index', 'source_event_key', 'source_file_index', 'source_event_index'])
            self.assertEqual(len(driver.repeat_seen_ids), 6)
            self.assertIn('1:8186255:0:10', driver.repeat_seen_ids)
            with self.assertRaisesRegex(ValueError, 'nonoverlapping'):
                driver._repeat_batch_identity(self.full_identity_batch())

    def test_full_ids_still_reject_true_local_duplicates_and_true_holdout_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            cycle.validation['source_ids'] = np.array([f'1:8186255:0:{100+i}' for i in range(6)])
            duplicate = self.full_identity_batch()
            duplicate['source_event_index'][1] = duplicate['source_event_index'][0]
            with self.assertRaisesRegex(ValueError, 'unique.*(full|physical|event)|Duplicate full'):
                driver._repeat_batch_identity(duplicate)
            self.assertFalse(driver.repeat_seen_ids)
            with self.assertRaisesRegex(ValueError, 'validation identities'):
                driver._repeat_batch_identity(self.full_identity_batch(start=100))
            self.assertFalse(driver.repeat_seen_ids)

    def test_full_ids_across_ranks_accept_base_collisions_but_reject_full_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            cycle.validation['source_ids'] = np.array([f'1:8186255:0:{1000+i}' for i in range(6)])
            def independent_rank(record):
                remote = copy.deepcopy(record)
                remote['rank'] = 1
                remote['source_ids'] = ['1:8186255:0:100', '1:8186255:0:101', '1:8186255:0:102']
                return [record, remote]
            driver._gather_records = independent_rank
            recorded = driver._repeat_batch_identity(self.full_identity_batch())
            self.assertEqual(recorded['global_event_count'], 6)
            self.assertEqual(recorded['global_unique_event_count'], 6)
            driver.repeat_seen_ids.clear()
            def duplicated_rank(record):
                remote = copy.deepcopy(record); remote['rank'] = 1
                return [record, remote]
            driver._gather_records = duplicated_rank
            with self.assertRaisesRegex(ValueError, 'Duplicate full source identities across native ranks'):
                driver._repeat_batch_identity(self.full_identity_batch())
            self.assertFalse(driver.repeat_seen_ids)

    def test_identity_schema_must_match_ranks_and_validation_without_prefix_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            cycle.validation['source_ids'] = np.array([f'1:8186255:0:{100+i}' for i in range(6)])
            def mismatched_rank(record):
                remote = copy.deepcopy(record); remote['rank'] = 1
                remote['source_id_columns'] = record['source_id_columns'][:2]
                remote['source_ids'] = [':'.join(identity.split(':')[:2]) for identity in record['source_ids']]
                return [record, remote]
            driver._gather_records = mismatched_rank
            with self.assertRaisesRegex(ValueError, '(schema|columns)'):
                driver._repeat_batch_identity(self.full_identity_batch())
            self.assertFalse(driver.repeat_seen_ids)
            driver._gather_records = lambda record: [record]
            with self.assertRaisesRegex(ValueError, '(schema|columns)'):
                driver._repeat_batch_identity(self.batch(0))
            malformed = self.full_identity_batch()
            del malformed['source_file_index']
            with self.assertRaisesRegex(ValueError, '(schema|columns)'):
                driver._repeat_batch_identity(malformed)
            self.assertFalse(driver.repeat_seen_ids)

    def test_identity_schema_cannot_change_between_native_draws(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            cycle.validation['source_ids'] = np.array([f'1:8186255:0:{100+i}' for i in range(6)])
            driver._repeat_batch_identity(self.full_identity_batch())
            before = driver.repeat_seen_ids.copy()
            cycle.validation['source_ids'] = np.array([f'1:{100+i}' for i in range(6)])
            with self.assertRaisesRegex(ValueError, '(schema|columns)'):
                driver._repeat_batch_identity(self.batch(1))
            self.assertEqual(driver.repeat_seen_ids, before)

    def test_invalid_identity_stops_before_sampling_or_optimizer_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            native_calls = []
            def native(*args, **kwargs):
                native_calls.append(True)
                raise AssertionError('Identity failure must occur before native computation')
            driver, cycle, optimizer, _ = self.fixture(Path(temporary), native)
            cycle.validation['source_ids'] = np.array([f'1:8186255:0:{100+i}' for i in range(6)])
            actor_before = copy.deepcopy(cycle.actor.state_dict())
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            invalid = self.full_identity_batch()
            invalid['source_event_index'].fill_(10)
            with patch.object(driver, '_repeat_anchor_baselines') as evaluate:
                with self.assertRaisesRegex(ValueError, 'Duplicate full source identities'):
                    driver.native_step(cycle.actor, cycle.reference, None, None, invalid,
                        optimizer, None, cycle.reward, global_step=1920,
                        reference_trust_coefficient=.5, grad_clip_norm=1.)
                evaluate.assert_not_called()
            self.assertFalse(native_calls)
            self.assertEqual(driver.repeat_draw, 0)
            nested_equal(self, cycle.actor.state_dict(), actor_before)
            nested_equal(self, optimizer.state_dict(), optimizer_before)

    def test_invalid_local_heldout_schema_is_gathered_before_worker_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            cycle.validation['source_ids'] = np.array(['1:100', '1:101'])
            gathered = []
            def gather(record):
                gathered.append(copy.deepcopy(record))
                return [record]
            driver._gather_records = gather
            with self.assertRaisesRegex(ValueError, 'columns'):
                driver._repeat_batch_identity(self.full_identity_batch())
            self.assertEqual(len(gathered), 1)
            self.assertIsNotNone(gathered[0]['error'])
            self.assertFalse(driver.repeat_seen_ids)

    def test_heldout_identity_views_must_agree_on_every_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, _, _, _ = self.fixture(Path(temporary))
            for field, altered in (('heldout_event_count', 7), ('heldout_event_id_fingerprint', 'different-panel')):
                def gather(record):
                    remote = copy.deepcopy(record); remote['rank'] = 1
                    remote['source_ids'] = ['0:10', '0:11', '0:12']
                    remote[field] = altered
                    return [record, remote]
                driver._gather_records = gather
                with self.assertRaisesRegex(ValueError, 'Held-out source identity views differ across native ranks'):
                    driver._repeat_batch_identity(self.batch(0))
                self.assertFalse(driver.repeat_seen_ids)

    def test_repeat_requires_exact_source_clock_frozen_head_and_no_teacher_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, optimizer, _ = self.fixture(Path(temporary))
            self.assertIsNone(driver.ensemble)
            with self.assertRaisesRegex(ValueError, 'source step'):
                driver.native_step(cycle.actor, cycle.reference, None, None, self.batch(0), optimizer,
                                   None, cycle.reward, global_step=1921)
            with torch.no_grad():
                cycle.reward.head.weight.add_(.1)
            with self.assertRaisesRegex(ValueError, 'classifier or reward clocks'):
                driver._assert_repeat_frozen()

    def test_evaluate_explicit_seed_reaches_outer_and_inner_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            driver, cycle, _, _ = self.fixture(Path(temporary))
            seeds = []
            def generate(panel, folder, k, seed, physics, **kwargs):
                self.assertEqual(kwargs, {'retain_features': False})
                seeds.append(seed)
                folder.mkdir(parents=True, exist_ok=True)
                return dict(rewards=np.zeros((6, 8)), deltas=np.zeros((6, 8, 2, 2)), cij=np.zeros((6, 9)))
            cycle.generate = generate
            cycle_module = ModuleType('RL.DGPO_neutrino.conditional_tau_cycle')
            cycle_module.cij_terms = lambda *args: np.zeros((6, 9))
            transfer_module = ModuleType('RL.DGPO_neutrino.checkpoint_transfer')
            @contextmanager
            def isolated_evaluation(actor, seed):
                self.assertIs(actor, cycle.actor)
                self.assertEqual(seed, 456+cycle.rank)
                yield
            transfer_module.isolated_evaluation = isolated_evaluation
            with patch.dict(sys.modules, {cycle_module.__name__: cycle_module, transfer_module.__name__: transfer_module}):
                driver._evaluate('seed-forwarding', seed=456)
            self.assertEqual(seeds, [456])

    def test_realized_precision_fidelity_and_physical_motion_not_nominal_rms(self):
        requested = torch.tensor([1e-9, -1e-9])
        unresolved = worker.displacement_measurements(requested, torch.zeros_like(requested))
        self.assertFalse(unresolved['perturbation_valid'])
        self.assertEqual(unresolved['parameter_update_rms'], 0.)
        wrong = worker.displacement_measurements(requested, -requested)
        self.assertFalse(wrong['perturbation_valid'])
        self.assertEqual(wrong['realized_requested_cosine'], -1.)
        good = worker.displacement_measurements(requested, requested)
        self.assertTrue(good['perturbation_valid'])
        self.assertEqual(good['realized_requested_norm_ratio'], 1.)
        panel = dict(visible_a=np.tile([3., 1., 0., 0.], (2, 1)),
                     visible_b=np.tile([3., -1., 0., 0.], (2, 1)))
        original = np.zeros((2, 8, 2, 2))
        changed = original.copy(); changed[..., 1] = .01
        motion = worker.generated_motion({'deltas': original}, {'deltas': changed}, panel)
        self.assertTrue(motion['motion_resolved'])
        self.assertAlmostEqual(motion['tau_direction_angular_rms_rad'], .01, places=10)
        self.assertAlmostEqual(motion['delta_phi_wrapped_rms_rad'], .01)
        self.assertFalse(worker.generated_motion({'deltas': original}, {'deltas': original}, panel)['motion_resolved'])

    def test_actual_preprocessing_keeps_integer_source_ids_but_policy_excludes_them(self):
        preprocessing = REPO/'evenet_dgpo/evenet/dataset/preprocess.py'
        tree = ast.parse(preprocessing.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ('process_event_batch', 'unflatten_dict')]
        namespace = dict(np=np)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(preprocessing), 'exec'), namespace)
        flat = {'x:0:0': np.array([1., 2.], dtype=np.float32),
                'x:0:1': np.array([3., 4.], dtype=np.float32),
                'x_mask:0': np.array([True, False]),
                'source_sample_index': np.zeros(2, dtype=np.int64),
                'source_event_key': np.array([2**55+1, 2**55+2], dtype=np.int64),
                'EXTRA/unused': np.ones(2), 'regression-unused': np.ones(2)}
        output = namespace['process_event_batch'](flat, {'x': [1, 2], 'x_mask': [1]}, namespace['unflatten_dict'],
                                                    drop_column_prefix=['EXTRA/', 'regression-'])
        self.assertEqual(output['source_event_key'].dtype, np.int64)
        np.testing.assert_array_equal(output['source_event_key'], flat['source_event_key'])
        np.testing.assert_array_equal(output['x'], np.array([[[1., 3.]], [[2., 4.]]], dtype=np.float32))
        self.assertEqual(output['x_mask'].dtype, np.bool_)
        self.assertNotIn('EXTRA/unused', output)
        tree = ast.parse(TRAINER.read_text())
        nodes = [node for node in tree.body if (isinstance(node, ast.Assign) and
                 any(isinstance(target, ast.Name) and target.id == '_DGPO_POLICY_BATCH_TENSOR_KEYS' for target in node.targets))
                 or (isinstance(node, ast.FunctionDef) and node.name == '_dgpo_policy_conditioning_batch')]
        namespace = dict(torch=torch, Tensor=torch.Tensor, Any=object)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), 'exec'), namespace)
        tensors = {key: torch.from_numpy(value) for key, value in output.items()}
        tensors.update(conditions=torch.zeros(2, 1), conditions_mask=torch.ones(2, 1, dtype=torch.bool),
                       x_invisible_mask=torch.ones(2, 2, dtype=torch.bool))
        policy = namespace['_dgpo_policy_conditioning_batch'](tensors)
        self.assertNotIn('source_event_key', policy)
        self.assertNotIn('source_sample_index', policy)
        self.assertIs(policy['x'], tensors['x'])

    def test_trainer_pending_hook_executes_before_native_clock_and_state_changes(self):
        tree = ast.parse(TRAINER.read_text())
        loop = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'dgpo_train_loop')
        pending = next(node for node in ast.walk(loop) if isinstance(node, ast.If)
                       and '_tau_mechanism_read_only_pending' in ast.unparse(node.test))
        done = next(node for node in ast.walk(loop) if isinstance(node, ast.If)
                    and '_tau_mechanism_diagnosis_complete' in ast.unparse(node.test))
        self.assertTrue(any(isinstance(node, ast.Continue) for node in pending.body))
        self.assertLess(pending.lineno, done.lineno)
        mutations = [node.lineno for node in ast.walk(loop)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == 'advance_policy_round_warmup' and node.lineno > pending.lineno]
        self.assertTrue(mutations)
        self.assertLess(pending.lineno, min(mutations))
        source = ast.unparse(loop)
        self.assertIn("mode in ('diagnose', 'repeat_diagnose')", source)
        # Execute the actual Continue node to establish that no after-hook
        # clock/log/state mutation runs for three pending draws.
        body = ast.parse('for metrics in draws:\n    pass\n').body[0]
        body.body = [copy.deepcopy(pending), ast.parse('changes.append(metrics)').body[0]]
        namespace = dict(draws=[{'_tau_mechanism_read_only_pending': True} for _ in range(3)]+[{}], changes=[])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[])), '<pending-hook>', 'exec'), namespace)
        self.assertEqual(namespace['changes'], [{}])


if __name__ == '__main__':
    unittest.main()
