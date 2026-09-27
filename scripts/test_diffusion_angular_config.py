"""CPU-only contract tests; does not start Ray or training."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from train_neutrino_backend import read_yaml, read_overlay_yaml, deep_update, build_runtime_config

ROOT = Path(__file__).resolve().parents[1]


class AngularFineTuneConfigTest(unittest.TestCase):
    def test_historical_optimization_continuation(self):
        cfg = read_overlay_yaml(ROOT / 'config/train_diffusion_angular_10pct_legacy_setup.yaml')
        t = cfg['options']['Training']
        self.assertEqual(t['pretrain_model_load_path'], '/pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_lr5e4/checkpoints/last.ckpt')
        self.assertIsNone(t['model_checkpoint_load_path'])
        self.assertEqual((t['epochs'], t['total_epochs']), (1000, 1000))
        self.assertEqual(t['learning_rate_warm_up_factor'], 1)
        self.assertFalse(t['scale_lr_with_world_size'])
        self.assertEqual(t['learning_rate_factor'], 1)
        for name in ['PET', 'GlobalEmbedding', 'ObjectEncoder', 'TruthGeneration',
                     'GroupedSequentialEmbedding', 'InvisibleInputProjector']:
            self.assertEqual(t['Components'][name]['learning_rate'], 8e-4)
            self.assertTrue(t['Components'][name]['warm_up'])
            self.assertEqual(t['Components'][name]['optimizer_type'].lower(), 'adamw')
        self.assertEqual(t['EarlyStopping']['patience'], 25)
        self.assertEqual(t['EarlyStopping']['min_delta'], 0)
        self.assertEqual(t['diffusion_every_n_epochs'], 20)
        self.assertFalse(t['EMA']['enable'])
        self.assertFalse(t['JointCoverage']['enable'])
        self.assertEqual(t['freeze_modules'], [])
        self.assertEqual(cfg['platform']['batch_size'], 2048)
        self.assertEqual(cfg['platform']['number_of_workers'], 16)
        self.assertTrue(cfg['network']['Body']['PET']['visible_angular_fourier']['enabled'])
        self.assertEqual(cfg['logger']['wandb']['id'], 'diffang10oldopt1')

    def test_lr5e4_500_epoch_restart(self):
        cfg = deep_update(
            read_yaml(ROOT / 'config/train_diffusion_nersc.yaml'),
            read_overlay_yaml(ROOT / 'config/train_diffusion_angular_10pct_lr5e4.yaml'))
        t = cfg['options']['Training']
        previous = read_overlay_yaml(ROOT / 'config/train_diffusion_angular_10pct_highlr.yaml')
        self.assertEqual(t['pretrain_model_load_path'],
                         '/pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_highlr/checkpoints/last.ckpt')
        self.assertEqual(cfg['experiment']['parent_wandb_run'], 'diffang10lr1')
        self.assertEqual(cfg['experiment']['source_stage'], 'angular_diffusion_highlr_last_checkpoint')
        self.assertIsNone(t['model_checkpoint_load_path'])
        self.assertEqual((t['epochs'], t['total_epochs']), (500, 500))
        self.assertEqual(t['learning_rate_warm_up_factor'], 5)
        self.assertFalse(t['scale_lr_with_world_size'])
        self.assertEqual(t['learning_rate'], 5e-4)
        self.assertEqual(t['learning_rate_body'], 2.5e-4)
        self.assertEqual(t['learning_rate_median'], 2.5e-4)
        for name in ['PET', 'GlobalEmbedding', 'ObjectEncoder']:
            self.assertEqual(t['Components'][name]['learning_rate'], 2.5e-4)
            self.assertTrue(t['Components'][name]['warm_up'])
        for name in ['TruthGeneration', 'GroupedSequentialEmbedding', 'InvisibleInputProjector']:
            self.assertEqual(t['Components'][name]['learning_rate'], 5e-4)
            self.assertTrue(t['Components'][name]['warm_up'])
        self.assertEqual(t['precision'], '32-true')
        self.assertEqual(t['freeze_modules'], [])
        self.assertFalse(t['EMA']['enable'])
        self.assertFalse(t['JointCoverage']['enable'])
        self.assertEqual(t['EarlyStopping']['patience'], 50)
        self.assertEqual(t['EarlyStopping']['monitor'], 'val/loss')
        self.assertEqual(t['EarlyStopping']['min_delta'], 1e-4)
        self.assertEqual(cfg['platform']['batch_size'], 2048)
        self.assertEqual(cfg['platform']['number_of_workers'], 16)
        self.assertEqual(cfg['logger']['wandb']['id'], 'diffang10lr5e4')
        self.assertNotEqual(t['model_checkpoint_save_path'], previous['options']['Training']['model_checkpoint_save_path'])

    def test_highlr_weights_only_schedule(self):
        cfg = read_overlay_yaml(ROOT / 'config/train_diffusion_angular_10pct_highlr.yaml')
        t = cfg['options']['Training']
        self.assertIsNone(t['model_checkpoint_load_path'])
        self.assertTrue(t['pretrain_model_load_path'].endswith('/epoch=390_train=0.1764_val=0.1745.ckpt'))
        self.assertEqual((t['epochs'], t['total_epochs']), (200, 200))
        self.assertEqual(t['learning_rate_warm_up_factor'], 5)
        self.assertFalse(t['scale_lr_with_world_size'])
        for name in ['PET', 'GlobalEmbedding', 'ObjectEncoder']:
            self.assertEqual(t['Components'][name]['learning_rate'], 1e-4)
            self.assertTrue(t['Components'][name]['warm_up'])
        for name in ['TruthGeneration', 'GroupedSequentialEmbedding', 'InvisibleInputProjector']:
            self.assertEqual(t['Components'][name]['learning_rate'], 2e-4)
            self.assertTrue(t['Components'][name]['warm_up'])
        self.assertEqual(t['freeze_modules'], [])
        self.assertFalse(t['EMA']['enable'])
        self.assertFalse(t['JointCoverage']['enable'])
        self.assertEqual(t['EarlyStopping']['patience'], 75)
        self.assertEqual(cfg['platform']['batch_size'], 2048)
        self.assertEqual(cfg['logger']['wandb']['id'], 'diffang10lr1')

    def test_resume130_has_new_run_and_exact_checkpoint(self):
        cfg = read_overlay_yaml(ROOT / 'config/train_diffusion_angular_10pct_resume130.yaml')
        t = cfg['options']['Training']
        self.assertEqual(cfg['logger']['wandb']['id'], 'diffang10r130')
        self.assertIsNone(t['pretrain_model_load_path'])
        self.assertEqual(t['model_checkpoint_load_path'], '/pscratch/sd/y/yiren/Ztautau/diffusion_angular_10pct_seed42/checkpoints/epoch=130_train=0.1952_val=0.2003.ckpt')
        self.assertNotEqual(Path(t['model_checkpoint_load_path']).parent, Path(t['model_checkpoint_save_path']))
        self.assertEqual(t['epochs'], 500)
        self.assertEqual(t['EarlyStopping']['patience'], 75)
        self.assertEqual(cfg['platform']['batch_size'], 2048)

    def setUp(self):
        self.cfg = deep_update(
            read_yaml(ROOT / 'config/train_diffusion_nersc.yaml'),
            read_overlay_yaml(ROOT / 'config/train_diffusion_angular_10pct_nersc.yaml'),
        )

    def test_weights_only_raw_and_fresh(self):
        t = self.cfg['options']['Training']
        self.assertIsNone(t['model_checkpoint_load_path'])
        self.assertEqual(t['precision'], '32-true')
        self.assertEqual(t['pretrain_model_load_path'],
                         '/pscratch/sd/y/yiren/checkpoints.20M.a4.last.ckpt')
        self.assertEqual(t['pretrain_model_load_path'], read_yaml(
            ROOT / 'config/train_diffusion_nersc.yaml')['options']['Training']['pretrain_model_load_path'])
        self.assertNotIn('1110', t['pretrain_model_load_path'])
        self.assertFalse(t['EMA']['enable'])
        self.assertFalse(t['EMA']['use_for_generation'])
        self.assertEqual(t['freeze_modules'], [])
        self.assertNotEqual(t['model_checkpoint_save_path'], str(Path(t['pretrain_model_load_path']).parent))

    def test_runtime_config_is_in_shared_experiment_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = deep_update(self.cfg, {'options': {'Training': {
                'model_checkpoint_save_path': str(Path(tmp) / 'experiment/checkpoints')}}})
            with patch('train_neutrino_backend.read_yaml', return_value=cfg):
                runtime = build_runtime_config(base_config=ROOT / 'config/train_diffusion_nersc.yaml',
                                               overlay_config=None, backend='pure-evenet')
            self.assertTrue(runtime.is_file())
            self.assertEqual(runtime.parent.parent, Path(tmp).resolve() / 'experiment/runtime_configs')
            resolved = read_yaml(runtime)
            self.assertFalse(resolved['rl']['enabled'])
            self.assertEqual(resolved['options']['Training']['precision'], '32-true')

    def test_objective_and_conditioning(self):
        t = self.cfg['options']['Training']
        self.assertEqual(t['Components']['TruthGeneration']['low_noise_weight'], 1)
        self.assertFalse(t['Components']['Classification']['include'])
        a = self.cfg['network']['Body']['PET']['visible_angular_fourier']
        self.assertEqual(a, dict(enabled=True, theta_source='Part_eta', phi_source='Part_phi'))
        self.assertNotIn('dgpo', self.cfg)

    def test_data_lr_and_diagnostics(self):
        p = self.cfg['platform']
        t = self.cfg['options']['Training']
        self.assertEqual(p['number_of_workers'], 16)
        self.assertEqual(p['batch_size'], 2048)
        self.assertIn('diffusion_train_10pct_seed42/train', p['data_parquet_dir'])
        self.assertIn('stic_filtered', p['data_parquet_val_dir'])
        self.assertEqual(self.cfg['options']['Dataset']['dataset_limit'], 1)
        self.assertFalse(t['scale_lr_with_world_size'])
        self.assertEqual(t['Components']['PET']['learning_rate'], 2e-5)
        self.assertEqual(t['Components']['TruthGeneration']['learning_rate'], 1e-4)
        self.assertEqual(t['JointCoverage']['every_n_epochs'], 25)
        self.assertFalse(t['JointCoverage']['enable'])
        self.assertEqual(t['epochs'], 500)
        self.assertEqual(t['total_epochs'], 500)
        self.assertEqual(t['EarlyStopping']['patience'], 75)
        self.assertEqual(self.cfg['logger']['wandb']['project'], 'nu2flow-RL')
        self.assertLess(len(self.cfg['logger']['wandb']['run_name']), 96)


if __name__ == '__main__':
    unittest.main()
