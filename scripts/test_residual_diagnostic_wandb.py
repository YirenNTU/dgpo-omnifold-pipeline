import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import residual_diagnostic_wandb as wb
import diagnose_residual_weights as diagnostic


class FakeRun:
    id = 'test-diagnostic'
    url = 'https://wandb.ai/test/project/runs/test-diagnostic'
    disabled = False
    offline = False

    def __init__(self):
        self.summary, self.logs, self.definitions, self.saved, self.finishes = {}, [], [], [], []

    def log(self, data, **kwargs):
        assert 'step' not in kwargs
        self.logs.append(data)

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def save(self, path, **kwargs):
        self.saved.append(path)

    def finish(self, **kwargs):
        self.finishes.append(kwargs)


class FakeSDK:
    def __init__(self):
        self.calls = []
        self.run = FakeRun()

    def init(self, **kwargs):
        self.calls.append(kwargs)
        self.run.id = kwargs['id']
        return self.run


def config(root):
    return {'output_dir': str(root), 'control_steps': 400, 'training_seeds': [1],
            'diagnostic_iterations': 3,
            'wandb': {'enabled': True, 'entity': 'test', 'project': 'project', 'name': 'diagnostic'}}


class TestDiagnosticWandb(unittest.TestCase):
    def test_only_rank_zero_initializes_fresh_online_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            for rank in (1, 2):
                tracker = wb.DiagnosticWandb(config(tmp), rank=rank, sdk=sdk)
                tracker.start()
                tracker.emit('ignored', 0, {'loss': 1})
                tracker.finish(0)
            self.assertFalse(sdk.calls)
            tracker = wb.DiagnosticWandb(config(tmp), sdk=sdk)
            with patch.dict('os.environ', {'WANDB_RUN_ID': 'old-production', 'WANDB_MODE': 'disabled'}):
                tracker.start()
            args = sdk.calls[0]
            self.assertNotEqual(args['id'], 'old-production')
            self.assertEqual(args['resume'], 'never')
            self.assertEqual(args['mode'], 'online')
            self.assertFalse(args['save_code'])
            self.assertEqual(args['job_type'], 'residual-weight-diagnostic')

    def test_axis_isolation_and_no_stale_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb(config(tmp), sdk=sdk)
            tracker.start()
            row = {'iteration': 1, 'fold': 1, 'step': 100, 'training_loss': .6,
                   'validation_loss': .7, 'validation_auc': .65, 'validation_evaluated': 0}
            tracker.production(1, row)
            tracker.production(1, {**row, 'step': 110, 'validation_evaluated': 1})
            tracker.production(1, {**row, 'fold': 2, 'step': 10})
            tracker.production(2, {**row, 'step': 10})
            tracker.restored(1, 1, 1, {'validation_production': {'bce': .65}})
            self.assertNotIn('production/s1/i1/f1/validation_loss', sdk.run.logs[0])
            self.assertIn('production/s1/i1/f1/validation_loss', sdk.run.logs[1])
            self.assertIn('production/s1/i1/f2/fit_step', sdk.run.logs[2])
            self.assertIn('production/s2/i1/f1/fit_step', sdk.run.logs[3])
            self.assertIn('restored/s1/f1/iteration', sdk.run.logs[4])
            self.assertEqual([r['diagnostic/event_index'] for r in sdk.run.logs], [1, 2, 3, 4, 5])
            self.assertFalse(tracker.failed)

    def test_controls_cadence_and_nonfinite_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb(config(tmp), sdk=sdk)
            tracker.start()
            tracker.control_progress(1, 2, 1, 'original', {'step': 1, 'training_loss': .6})
            self.assertEqual(len(sdk.run.logs), 0)
            tracker.control_progress(1, 2, 1, 'original', {'step': 10, 'training_loss': .6, 'gradient_norm': float('nan')})
            tracker.control_evaluation(1, 2, 1, 'original', {'step': 10, 'own_objective': {'bce': .7},
                                                         'original_weights_same_samples': {'auc': .6}})
            self.assertEqual(len(sdk.run.logs), 2)
            self.assertFalse(any('gradient_norm' in k for k in sdk.run.logs[0]))
            self.assertNotIn('control/s1/i2/f1/original/training_loss', sdk.run.logs[1])

    def test_final_upload_allowlist_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('summary.json', 'report.json', 'pool.pt', 'runtime.yaml', 'manifest.json'):
                (root / name).write_text('{}')
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb(config(root), sdk=sdk)
            tracker.start()
            tracker.finish(1)
            self.assertEqual({Path(p).name for p in sdk.run.saved}, {'summary.json', 'report.json'})
            self.assertEqual(sdk.run.finishes, [{'exit_code': 1}])
            self.assertEqual(sdk.run.summary['diagnostic/status'], 'failed')

    def test_logging_error_does_not_break_training_collectives(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb(config(tmp), sdk=sdk)
            tracker.start()
            with patch.object(sdk.run, 'log', side_effect=RuntimeError('logging error')), self.assertLogs(level='ERROR'):
                tracker.emit('x', 0, {'loss': .7})
            self.assertTrue(tracker.failed)
            self.assertEqual(json.loads((Path(tmp) / 'wandb_status.json').read_text())['status'],
                             'logging_failed_local_results_preserved')

    def test_init_error_is_explicit_and_disabled_mode_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb({**config(tmp), 'wandb': {'enabled': False}}, sdk=sdk)
            tracker.start()
            self.assertFalse(sdk.calls)
            tracker = wb.DiagnosticWandb(config(tmp), sdk=sdk)
            with patch.object(sdk, 'init', side_effect=RuntimeError('no login')), self.assertLogs(level='ERROR'):
                with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                    tracker.start()
            for bad in ({'enabled': 'true'}, {**config(tmp)['wandb'], 'resume': 'allow'},
                        {**config(tmp)['wandb'], 'api_key': 'do-not-store'}):
                with self.assertRaises(ValueError):
                    wb.validate_wandb(bad)

    def test_upload_existing_uses_only_saved_json_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = config(root)
            (root / 'manifest.json').write_text(json.dumps(cfg))
            folder = root / 'seed_1'
            folder.mkdir()
            (folder / 'production_fit.jsonl').write_text(json.dumps(
                {'iteration': 1, 'fold': 1, 'step': 10, 'training_loss': .6}) + '\n{"partial":')
            (folder / 'iteration_1.json').write_text('{"fold_logit_rms_difference": 0.2}')
            (folder / 'i1_f1_restored_best.json').write_text('{"fit_eval": {"bce": 0.5}}')
            (folder / 'control_i2_f1_original.json').write_text(json.dumps({'evaluations': [
                {'step': 0, 'own_objective': {'bce': .7}, 'original_weights_same_samples': {'auc': .6}}]}))
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb(cfg, sdk=sdk)
            with patch.object(diagnostic, 'DiagnosticWandb', return_value=tracker), \
                 patch.object(diagnostic, 'worker', side_effect=AssertionError('must not train')), \
                 patch.object(diagnostic.replay, 'read_checkpoint', side_effect=AssertionError('must not load policy')):
                self.assertEqual(diagnostic.upload_existing(cfg), 0)
            self.assertEqual(len(sdk.calls), 1)
            self.assertEqual(len(sdk.run.logs), 5)
            self.assertEqual(sdk.run.summary['diagnostic/status'], 'partial_results_uploaded')
            self.assertEqual(sdk.run.finishes, [{'exit_code': 0}])

    def test_repeated_protocol_namespaces_and_dynamic_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {
                **config(root),
                'crossfit_repeat_arms': [1, 2],
                'run_controls': False,
            }
            (root / 'manifest.json').write_text(json.dumps(cfg))
            for protocol, repeats in (('r1f2', 1), ('r2f2', 2)):
                folder = root / protocol / 'seed_1'
                folder.mkdir(parents=True)
                (folder / 'production_fit.jsonl').write_text(json.dumps({
                    'iteration': 1, 'repeat': repeats, 'fold': 1,
                    'step': 10, 'training_loss': .6,
                }) + '\n')
                (folder / 'iteration_1.json').write_text(
                    '{"oof_repeat_rms_dispersion": 0.2}'
                )
                (folder / f'i1_r{repeats}_f1_restored_best.json').write_text(
                    '{"fit_eval": {"bce": 0.5}}'
                )
            sdk = FakeSDK()
            tracker = wb.DiagnosticWandb(cfg, sdk=sdk)
            with patch.object(
                diagnostic, 'DiagnosticWandb', return_value=tracker
            ):
                self.assertEqual(diagnostic.upload_existing(cfg), 0)
            keys = {key for row in sdk.run.logs for key in row}
            self.assertTrue(
                any(key.startswith('production/r1f2/') for key in keys)
            )
            self.assertTrue(
                any(key.startswith('production/r2f2/') for key in keys)
            )
            self.assertTrue(any(key.startswith('stack/r2f2/') for key in keys))


if __name__ == '__main__':
    unittest.main()
