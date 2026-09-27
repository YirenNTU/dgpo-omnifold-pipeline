from dataclasses import dataclass
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

import diagnose_residual_weights as diagnostic


@dataclass
class Pool:
    packed_event: torch.Tensor
    truth: torch.Tensor
    candidates: torch.Tensor

    @property
    def n_events(self):
        return len(self.truth)

    def select(self, indices):
        return Pool(self.packed_event[indices], self.truth[indices], self.candidates[indices])


class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4) * .1)

    def forward(self, condition, sample):
        return (sample * self.weight).sum(-1)


class TestWeightDiagnostic(unittest.TestCase):
    def test_weight_stats_shift_invariance_and_extreme_values(self):
        x = torch.tensor([0., 1., -1., 2.], dtype=torch.float64)
        a, b = diagnostic.mean_one(x), diagnostic.mean_one(x + 10000)
        torch.testing.assert_close(a, b)
        self.assertAlmostEqual(float(a.mean()), 1.)
        unit = diagnostic.weight_stats(torch.zeros(100))
        self.assertAlmostEqual(unit['ess'], 100.)
        extreme = diagnostic.weight_stats(torch.tensor([-10000., 10000.]))
        self.assertAlmostEqual(extreme['ess'], 1.)
        self.assertAlmostEqual(extreme['top_1pct_mass'], 1.)
        for bad in [torch.tensor([]), torch.tensor([float('nan')]), torch.tensor([float('inf')])]:
            with self.assertRaises(ValueError):
                diagnostic.weight_stats(bad)

    def test_null_metrics_and_weight_tail_loss(self):
        zero = torch.zeros(100)
        m = diagnostic.score_metrics(zero, zero, zero)
        self.assertAlmostEqual(m['bce'], math.log(2), places=6)
        self.assertAlmostEqual(m['auc'], .5)
        self.assertAlmostEqual(m['negative_bce_top_weight_1pct_fraction'], .01)
        with self.assertRaises(ValueError):
            diagnostic.score_metrics(zero, zero[:2], zero)

    def test_cap_uses_training_only_and_leaves_sources_unchanged(self):
        train = torch.arange(100, dtype=torch.float64) / 10
        val = torch.zeros(40)
        original = train.clone()
        a, b, cap = diagnostic.intervention('capped', train, val, .9)
        _, _, cap2 = diagnostic.intervention('capped', train, val + torch.arange(40), .9)
        self.assertEqual(cap, cap2)
        self.assertLessEqual(float(a.exp().max()), cap)
        torch.testing.assert_close(train, original)
        self.assertGreater(diagnostic.weight_stats(a)['ess'], diagnostic.weight_stats(train)['ess'])
        for arm in ('unit_raw', 'truth_null'):
            t, v, c = diagnostic.intervention(arm, train, val, .9)
            self.assertEqual(int(torch.count_nonzero(t)), 0)
            self.assertEqual(int(torch.count_nonzero(v)), 0)
            self.assertIsNone(c)

    def test_opposite_fold_counterfactual_can_expose_target_mismatch(self):
        # Both fold weights are evaluated on the same held-out events.
        alpha = .75
        z1 = torch.tensor([-4., 4.], dtype=torch.float64)
        z2 = -z1
        f = -diagnostic.mean_one(alpha * z2).log()
        train_like = diagnostic.score_metrics(f, f, alpha * z2)['bce']
        ensemble = diagnostic.score_metrics(f, f, alpha * (z1 + z2) / 2)['bce']
        self.assertLess(train_like, math.log(2))
        self.assertGreater(ensemble, 1.)

    def test_paired_protocol_summary_uses_shared_seed_and_iteration(self):
        def result(repeats, folds, bce, ess, jsd, updates):
            physics = {
                'target/x': {'weighted_jsd': jsd},
                'reco/x': {'weighted_jsd': jsd + .1},
                'topology/x': {'weighted_jsd': jsd + .2},
            }
            return {
                'seed': 7,
                'crossfit_repeats': repeats,
                'crossfit_folds': folds,
                'iterations': [{
                    'iteration': 1,
                    'ensemble_validation': {'bce': bce, 'auc': .7},
                    'member_fit_minus_oof_bce': .1,
                    'train_weights_if_applied': {'ess_fraction': ess},
                    'validation_weights_if_applied': {'ess_fraction': ess},
                    'classifier_updates': updates,
                    'classifier_wall_time_seconds': float(updates),
                    'train_physics_if_applied': physics,
                    'validation_physics_if_applied': physics,
                }],
            }

        rows = diagnostic.paired_protocol_comparisons([
            result(1, 2, .7, .5, .3, 20),
            result(1, 5, .6, .6, .2, 50),
        ])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['control'], 'r1f2')
        self.assertEqual(rows[0]['ensemble'], 'r1f5')
        self.assertAlmostEqual(rows[0]['validation_bce_delta'], -.1)
        self.assertAlmostEqual(rows[0]['train_target_weighted_jsd_delta'], -.1)
        self.assertEqual(rows[0]['classifier_updates_ratio'], 2.5)

    def test_config_validation(self):
        cfg = yaml.safe_load((diagnostic.ROOT / 'config/dgpo_10pct_residual_weight_diagnostic.yaml').read_text())
        diagnostic.validate_settings(cfg)
        self.assertEqual(cfg['expected_policy_step'], 320)
        self.assertIsNone(cfg['pool_events'])
        for field, value in [('workers', 0), ('training_seeds', [1, 1]), ('cap_quantile', 1), ('diagnostic_iterations', 1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                diagnostic.validate_settings({**cfg, field: value})
        paired = yaml.safe_load(
            (
                diagnostic.ROOT
                / 'config/dgpo_10pct_oof_ensemble_diagnostic.yaml'
            ).read_text()
        )
        validated = diagnostic.validate_settings(paired)
        self.assertEqual(validated['crossfit_repeat_arms'], [1])
        self.assertEqual(validated['crossfit_fold_arms'], [2, 5])
        self.assertEqual(
            diagnostic.resolved_protocol_arms(validated),
            [(1, 2), (1, 5)],
        )
        self.assertFalse(validated['run_controls'])
        with self.assertRaises(ValueError):
            diagnostic.validate_settings({
                **paired, 'run_controls': True,
            })
        with self.assertRaises(ValueError):
            diagnostic.validate_settings({
                **paired, 'crossfit_fold_arms': [1, 5],
            })

    def test_preflight_is_readonly_pins_sources_and_rejects_overwrite(self):
        cfg = yaml.safe_load((diagnostic.ROOT / 'config/dgpo_10pct_residual_weight_diagnostic.yaml').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('policy', 'backbone', 'normalization', 'data', 'configs'):
                (root / name).mkdir()
            checkpoint = root / 'policy/source.ckpt'
            torch.save({'state_dict': {'w': torch.ones(2)}, 'global_step': 320}, checkpoint)
            backbone = root / 'backbone/source.ckpt'
            torch.save({'state_dict': {'w': torch.ones(2)}}, backbone)
            normalizer = root / 'normalization/n.pt'
            torch.save({'mean': torch.zeros(2)}, normalizer)
            (root / 'data/part.parquet').write_bytes(b'preflight-metadata-only')
            runtime = {'options': {'Training': {'model_checkpoint_load_path': str(checkpoint),
                        'EMA': {'replace_model_after_load': True}},
                        'Dataset': {'normalization_file': str(normalizer)}},
                       'platform': {'data_parquet_dir': str(root / 'data')},
                       'reward_config': {'omnifold': {'backbone_checkpoint': str(backbone)}},
                       'dgpo': {'adaptive_omnifold': {'single_pool_train_validation': True,
                            'recalibration': {'crossfit_folds': 2, 'warm_start_from_iteration_one': True,
                                              'fit': {'later_iteration_train_mode': 'full'}}}}}
            base = root / 'configs/base.yaml'
            base.write_text(yaml.safe_dump(runtime))
            overlay = root / 'configs/overlay.yaml'
            overlay.write_text('{}\n')
            cfg.update(base_config=str(base), overlay_config=str(overlay), output_dir=str(root / 'output'))
            prepared = diagnostic.prepare(cfg)
            self.assertFalse((root / 'output').exists())
            self.assertFalse(prepared['runtime']['options']['Training']['EMA']['replace_model_after_load'])
            diagnostic.verify_sources(prepared)
            with self.assertRaisesRegex(ValueError, 'Wrong source policy step'):
                diagnostic.prepare({**cfg, 'expected_policy_step': 260})
            with self.assertRaisesRegex(ValueError, 'tensor digest'):
                diagnostic.prepare({**cfg, 'expected_policy_sha256': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                diagnostic.prepare({**cfg, 'output_dir': str(root / 'policy/new_output')})
            (root / 'output').mkdir()
            with self.assertRaises(FileExistsError):
                diagnostic.prepare(cfg)
            with normalizer.open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'Source changed'):
                diagnostic.verify_sources(prepared)

    def test_production_observer_and_real_control_fits(self):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import fit_residual_ratio_stack
        from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig
        torch.manual_seed(2)
        train = Pool(torch.randn(80, 2), torch.ones(80, 4), -torch.ones(80, 1, 4))
        val = Pool(torch.randn(20, 2), torch.ones(20, 4), -torch.ones(20, 1, 4))
        cfg = dict(score_batch_size=8, tempering=.75, diagnostic_iterations=3,
                   control_steps=3, control_evaluate_every=2, cap_quantile=.995)
        fc = RatioFitConfig(steps=4, batch_size=8, sampling='independent_epoch_shuffle',
                            drop_last_batch=True, learning_rate=.001, validation_batch_size=8,
                            validation_interval_steps=1, validation_patience_evaluations=10)
        args = dict(model_factory=Classifier, data_condition=train.packed_event, data_sample=train.truth,
                    gen_condition=train.packed_event, gen_sample=train.candidates,
                    validation_data_condition=val.packed_event, validation_data_sample=val.truth,
                    validation_gen_condition=val.packed_event, validation_gen_sample=val.candidates,
                    iterations=3, fit_config=fc, tempering=.75, seed=17,
                    warm_start_iterations=(1,), warm_start_from_iteration_one=True,
                    crossfit_seed=42, crossfit_partition='identity', residual_min_auc_gain=.01)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = diagnostic.Capture(cfg, train, val, root, 0)
            with self.assertRaises(diagnostic.DiagnosticLimit):
                fit_residual_ratio_stack(**args, diagnostic_callback=capture)
            self.assertEqual(len(capture.iterations), 3)
            self.assertEqual(len(capture.inputs), 6)
            # Inheritance captured includes the complete corresponding first-fold state.
            restored = torch.load(root / 'i1_f1_restored_best.pt', weights_only=True)
            self.assertEqual(diagnostic.replay.state_digest(restored['state']),
                             diagnostic.replay.state_digest(capture.inputs[(2, 1, 1)]['initial']))
            results = diagnostic.run_controls(capture, Classifier, torch.device('cpu'))
            self.assertEqual(len(results), 20)
            for i in (2, 3):
                for f in (1, 2):
                    arms = [r for r in results if (r['iteration'], r['fold']) == (i, f)]
                    self.assertEqual(len({r['initial_sha256'] for r in arms}), 1)
                    for arm in arms:
                        self.assertEqual([r['step'] for r in arm['evaluations']], [0, 2, 3])
                        if arm['arm'] == 'truth_null':
                            self.assertAlmostEqual(arm['evaluations'][-1]['own_objective']['auc'], .5)
            report = json.loads((root / 'i2_f1_restored_best.json').read_text())
            self.assertIn('fit_eval', report)
            self.assertIn('oof_eval', report)
            self.assertIn('validation_training_weight_path', report)
            self.assertIn('validation_physics_if_applied', capture.iterations[0])

    def test_repeated_observer_assembles_per_event_oof_ensemble(self):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import fit_residual_ratio_stack
        from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig

        torch.manual_seed(7)
        train = Pool(torch.arange(160).reshape(80, 2).float(),
                     torch.ones(80, 4), -torch.ones(80, 1, 4))
        val = Pool(torch.arange(40).reshape(20, 2).float(),
                   torch.ones(20, 4), -torch.ones(20, 1, 4))
        cfg = dict(
            score_batch_size=16,
            tempering=.75,
            diagnostic_iterations=1,
            control_steps=1,
            control_evaluate_every=1,
            cap_quantile=.995,
            crossfit_repeats=3,
            crossfit_folds=2,
            protocol='r3f2',
            physics_bins=10,
            workers=1,
        )
        fit_cfg = RatioFitConfig(
            steps=2,
            batch_size=8,
            validation_batch_size=16,
            validation_interval_steps=1,
            validation_patience_evaluations=4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            capture = diagnostic.Capture(
                cfg, train, val, Path(tmp), 0, seed=17
            )
            with self.assertRaises(diagnostic.DiagnosticLimit):
                fit_residual_ratio_stack(
                    model_factory=Classifier,
                    data_condition=train.packed_event,
                    data_sample=train.truth,
                    gen_condition=train.packed_event,
                    gen_sample=train.candidates,
                    iterations=1,
                    min_iterations=1,
                    iteration_one_only=True,
                    fit_config=fit_cfg,
                    tempering=.75,
                    seed=17,
                    crossfit_seed=42,
                    crossfit_folds=2,
                    crossfit_repeats=3,
                    crossfit_partition='identity',
                    validation_data_condition=val.packed_event,
                    validation_data_sample=val.truth,
                    validation_gen_condition=val.packed_event,
                    validation_gen_sample=val.candidates,
                    diagnostic_callback=capture,
                )
            self.assertEqual(len(capture.inputs), 6)
            self.assertEqual(
                {(repeat, fold) for _, repeat, fold in capture.inputs},
                {(repeat, fold) for repeat in range(1, 4) for fold in (1, 2)},
            )
            row = capture.iterations[0]
            self.assertEqual(row['crossfit_repeats'], 3)
            self.assertIn('oof_repeat_rms_dispersion', row)
            self.assertIn('classifier_gpu_hours', row)
            self.assertTrue(
                (Path(tmp) / 'i1_r3_f2_restored_best.json').is_file()
            )

    def test_readonly_observer_does_not_change_fit(self):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import fit_residual_ratio_stack
        from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig
        seen = []
        states = []
        for enabled in (False, True):
            torch.manual_seed(1)
            c = torch.randn(40, 2)
            def observer(event, row):
                seen.append(event)
                if event == 'before_fit':
                    diagnostic.replay.state_digest(row['model'].state_dict())
            # Null problem closes after iteration 1; accept first artificially
            # is unnecessary: both runs must raise exactly the same gate failure.
            model_states = []
            def factory():
                m = Classifier()
                model_states.append(m)
                return m
            with self.assertRaisesRegex(RuntimeError, 'first residual classifier failed'):
                fit_residual_ratio_stack(model_factory=factory, data_condition=c, data_sample=torch.ones(40, 4),
                    gen_condition=c, gen_sample=torch.ones(40, 1, 4), iterations=2,
                    fit_config=RatioFitConfig(steps=4, batch_size=8, validation_interval_steps=1,
                                              validation_patience_evaluations=10), tempering=.75, seed=7,
                    validation_data_condition=c, validation_data_sample=torch.ones(40, 4),
                    validation_gen_condition=c, validation_gen_sample=torch.ones(40, 1, 4),
                    diagnostic_callback=observer if enabled else None)
            states.append([diagnostic.replay.state_digest(m.state_dict()) for m in model_states])
        self.assertEqual(states[0], states[1])
        self.assertEqual(seen, ['before_fit', 'fold_scored', 'before_fit', 'fold_scored', 'iteration_scored'])


if __name__ == '__main__':
    unittest.main()
