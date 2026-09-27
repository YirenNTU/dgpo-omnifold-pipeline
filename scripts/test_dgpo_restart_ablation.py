import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from train_dgpo_restart_ablation import check_pair, config_path, read_yaml, verify_source


class RestartAblationTests(unittest.TestCase):
    def setUp(self):
        self.a = read_yaml(config_path('inherit'))
        self.b = read_yaml(config_path('restart'))

    def test_configs_are_matched(self):
        check_pair(self.a, self.b)
        self.assertEqual(self.a['dgpo']['adaptive_omnifold']['recalibration']['warm_start_iterations'], [1])
        self.assertFalse(self.a['dgpo']['adaptive_omnifold']['recalibration']['warm_start_from_iteration_one'])
        for config in (self.a, self.b):
            fit = config['dgpo']['adaptive_omnifold']['recalibration']['fit']
            self.assertEqual(fit['min_steps_per_fold'], 1000)
            self.assertEqual(fit['warm_start_min_epochs_per_fold'], 10)
            audit = config['dgpo']['adaptive_omnifold']['audit_fit']
            self.assertEqual(audit['training_readiness'], {
                'cold_start_min_epochs': 250, 'warm_start_min_epochs': 5})
            self.assertGreaterEqual((199760 // audit['batch_size']) *
                                    audit['training_readiness']['cold_start_min_epochs'], 1000)

    def test_other_training_difference_rejected(self):
        self.b['dgpo']['adaptive_omnifold']['audit_fit']['learning_rate'] *= 2
        with self.assertRaises(ValueError):
            check_pair(self.a, self.b)

    def test_later_iteration_inheritance_rejected(self):
        r = self.a['dgpo']['adaptive_omnifold']['recalibration']
        r['warm_start_iterations'] = [1, 2]
        with self.assertRaises(ValueError):
            check_pair(self.a, self.b)
        r['warm_start_iterations'] = [1]
        r['warm_start_from_iteration_one'] = True
        with self.assertRaises(ValueError):
            check_pair(self.a, self.b)

    def test_i1_only_inheritance_resume_and_longer_patience(self):
        c = read_yaml(config_path('inherit').parent/'dgpo_omnifold_ztautau_10pct_i1inherit_best_resume.yaml')
        d = c['dgpo']; a = d['adaptive_omnifold']; r = a['recalibration']
        self.assertFalse(d['tarp']['enabled'])
        self.assertEqual(r['warm_start_iterations'], [1])
        self.assertFalse(r['warm_start_from_iteration_one'])
        self.assertTrue(a['trigger']['rollback_to_best_on_plateau'])
        self.assertEqual(a['trigger']['required_consecutive_checks'], 12)
        self.assertEqual(a['staleness_every_n_steps'], 10)
        self.assertEqual(r['decoder_hidden_dim'], 128)
        self.assertEqual(r['decoder_layers'], 1)
        self.assertFalse(r['topology_fourier_embedding'])
        self.assertTrue(d['ztautau_metrics']['enabled'])

    def test_confirmed_step260_p12_no_tarp_control(self):
        c = read_yaml(config_path('inherit').parent/'dgpo_omnifold_ztautau_10pct_step260_i1inherit_p12.yaml')
        d = c['dgpo']; a = d['adaptive_omnifold']; r = a['recalibration']
        self.assertEqual(d['checkpoint_load_mode'], 'resume')
        self.assertTrue(d['pinned_classifier_restart'])
        self.assertFalse(d['auto_resume_from_last'])
        self.assertTrue(c['options']['Training']['model_checkpoint_load_path'].endswith('/dgpo-epoch=25-next_ep=26-step=260.ckpt'))
        self.assertEqual(r['warm_start_iterations'], [1])
        self.assertFalse(r['warm_start_from_iteration_one'])
        self.assertFalse(r['bootstrap_on_start'])
        self.assertTrue(a['baseline_probe_on_start'])
        self.assertTrue(a['trigger']['rollback_to_best_on_plateau'])
        self.assertEqual(a['trigger']['required_consecutive_checks'], 12)
        self.assertEqual(a['staleness_every_n_steps'], 10)
        self.assertFalse(d['tarp']['enabled'])
        self.assertTrue(d['ztautau_metrics']['enabled'])
        self.assertEqual(d['validation_every_n_epochs'], 3)
        self.assertEqual(a['audit_fit']['training_readiness']['cold_start_min_epochs'], 250)
        self.assertEqual(r['fit']['min_steps_per_fold'], 1000)
        self.assertTrue(c['logger']['wandb']['fresh_run'])

    def test_standalone_launcher_selects_requested_config(self):
        from train_dgpo_restart_ablation import main
        target = config_path('inherit').parent/'dgpo_omnifold_ztautau_10pct_step260_i1inherit_p12.yaml'
        with patch('sys.argv', ['launcher', '--config', str(target)]), \
             patch('train_dgpo_restart_ablation.verify_source') as verify, \
             patch('train_dgpo_restart_ablation.build_runtime_config', return_value=target), \
             patch('train_dgpo_restart_ablation.subprocess.run') as run:
            main()
        self.assertEqual(verify.call_args.args[0]['dgpo']['adaptive_omnifold']['trigger']['required_consecutive_checks'], 12)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--overlay-config')+1], str(target.resolve()))

    def test_shared_output_rejected(self):
        self.b['options']['Training']['model_checkpoint_save_path'] = self.a['options']['Training']['model_checkpoint_save_path']
        with self.assertRaises(ValueError):
            check_pair(self.a, self.b)

    def test_cold_partition_drift_rejected(self):
        self.b['dgpo']['adaptive_omnifold']['recalibration']['crossfit_partition'] = 'auto'
        with self.assertRaises(ValueError):
            check_pair(self.a, self.b)

    def test_best0_resume_preserves_protocol_and_restarts_future_fits(self):
        c = read_yaml(config_path('restart').parent/'dgpo_omnifold_ztautau_10pct_restart_best_resume.yaml')
        d = c['dgpo']; a = d['adaptive_omnifold']
        self.assertEqual(d['checkpoint_load_mode'], 'resume')
        self.assertTrue(d['auto_resume_from_last'])
        self.assertTrue(d['auto_resume_fallback_checkpoint_path'].endswith('/dgpo-epoch=-1-next_ep=0-step=0.ckpt'))
        # Full resume must retain the old audit protocol/global-best provenance.
        legacy = read_yaml(config_path('restart').parent/'dgpo_omnifold_ztautau_10pct_old_classifier_resume_p8.yaml')
        self.assertEqual(a['audit_fit'], legacy['dgpo']['adaptive_omnifold']['audit_fit'])
        self.assertNotIn('training_readiness', a['audit_fit'])
        self.assertEqual(a['recalibration']['warm_start_iterations'], [])
        self.assertEqual(a['recalibration']['fit']['min_steps_per_fold'], 1000)
        self.assertFalse(a['recalibration']['bootstrap_on_start'])
        self.assertFalse(a['recalibration']['refit_once_on_resume'])
        self.assertFalse(a['baseline_probe_on_start'])
        self.assertTrue(a['trigger']['rollback_to_best_on_plateau'])
        self.assertNotEqual(c['options']['Training']['model_checkpoint_save_path'],
                            self.b['options']['Training']['model_checkpoint_save_path'])

    def test_source_and_fresh_output_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.a)
            root = Path(tmp)
            source = root/'source.ckpt'
            output = root/'new'
            c['options']['Training']['model_checkpoint_load_path'] = str(source)
            c['options']['Training']['model_checkpoint_save_path'] = str(output)
            c['reward_config']['omnifold']['backbone_checkpoint'] = str(source)
            c['platform']['data_parquet_dir'] = tmp
            c['platform']['data_parquet_val_dir'] = tmp
            payload = dict(epoch=25, global_step=260, dgpo_next_epoch=26, state_dict={'w': torch.ones(2)})
            torch.save(payload, source)
            verify_source(c)
            payload['global_step'] = 330
            torch.save(payload, source)
            with self.assertRaises(ValueError):
                verify_source(c)
            output.mkdir()
            (output/'existing.ckpt').touch()
            with self.assertRaises(FileExistsError):
                verify_source(c)


if __name__ == '__main__':
    unittest.main()
