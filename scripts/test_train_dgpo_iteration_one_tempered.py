"""CPU-only launch guard tests; no NERSC data or W&B connection."""
import copy
import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
import yaml
import train_dgpo_iteration_one_tempered as launcher
from train_neutrino_backend import deep_update


def _source_cfg(directory):
    cfg = yaml.safe_load(launcher.DEFAULT.read_text())
    p = Path(directory)
    source = p / 'dgpo-epoch=25-next_ep=26-step=260.ckpt'
    state = {'weight': torch.tensor([1., 2.])}
    torch.save({'epoch': 25, 'global_step': 260, 'dgpo_next_epoch': 26,
                'state_dict': state,
                'ema_state_dict': {'weight': torch.zeros(2)}}, source)
    cfg['options']['Training']['model_checkpoint_load_path'] = str(source)
    cfg['options']['Training']['model_checkpoint_save_path'] = str(
        p / 't075_trust05_step260' / 'checkpoints')
    cfg['nersc']['reproducibility']['source_policy_sha256'] = launcher.state_digest(state)
    cfg['reward_config']['omnifold']['backbone_checkpoint'] = str(source)
    cfg['platform']['data_parquet_dir'] = str(p)
    cfg['platform']['data_parquet_val_dir'] = str(p)
    return cfg, p, state


class TestTemperedLaunch(unittest.TestCase):
    def test_overlay_is_control_plus_intended_knobs(self):
        root = launcher.ROOT
        control = yaml.safe_load(launcher.CONTROL.read_text())
        overlay = yaml.safe_load(launcher.DEFAULT.read_text())
        expected = copy.deepcopy(control['dgpo'])
        expected['auto_resume_from_last'] = True
        expected['reference_trust']['coefficient'] = 0.5
        expected['adaptive_omnifold']['trigger']['raw_improvement_min_delta'] = 0.005
        expected['adaptive_omnifold']['trigger']['rollback_to_best_on_plateau'] = True
        expected['adaptive_omnifold']['recalibration']['tempering'] = 0.75
        self.assertEqual(overlay['dgpo'], expected)
        self.assertTrue(overlay['options']['Training']['model_checkpoint_load_path'].endswith(
            'dgpo-epoch=25-next_ep=26-step=260.ckpt'))
        self.assertNotEqual(overlay['options']['Training']['model_checkpoint_load_path'],
                            control['options']['Training']['model_checkpoint_load_path'])
        self.assertEqual(overlay['nersc']['reproducibility']['source_dgpo_global_step'], 260)
        self.assertEqual(
            overlay['nersc']['reproducibility']['source_policy_sha256'],
            'dc66b775e474070c12aa9f3bc5f67436921fa1dd36d78202ac4f9ed75d376378')
        merged = deep_update(
            yaml.safe_load((root / 'config/train_diffusion_nersc.yaml').read_text()), overlay)
        d = merged['dgpo']
        self.assertEqual(d['beta_kl'], 0.0)
        self.assertEqual(d['checkpoint_load_mode'], 'weights_only')
        self.assertIsNone(d['auto_resume_fallback_checkpoint_path'])
        out = merged['options']['Training']['model_checkpoint_save_path']
        self.assertIn('t075_trust05_step260', out)
        self.assertNotIn('step320', out)
        self.assertNotIn('iteration1_only_step320', out)
        self.assertNotIn('122b9d84', out)
        self.assertIn('ResidualTempering075', merged['logger']['wandb']['tags'])
        self.assertNotIn('ResidualTempering1', merged['logger']['wandb']['tags'])
        self.assertNotIn('ForwardRefitNoRollback', merged['logger']['wandb']['tags'])
        launcher.assert_nersc_16gpu(overlay)
        launcher.assert_nersc_16gpu(merged)
        self.assertEqual(merged['platform']['number_of_workers'], 16)
        self.assertEqual(merged['platform']['resources_per_worker']['GPU'], 1)
        self.assertTrue(merged['platform']['use_gpu'])
        self.assertEqual(merged['nersc']['execution']['workers'], 16)
        self.assertIn('event_info', merged)

    def test_refuses_non_16gpu_layout(self):
        overlay = yaml.safe_load(launcher.DEFAULT.read_text())
        overlay['platform']['number_of_workers'] = 8
        with self.assertRaises(ValueError):
            launcher.assert_nersc_16gpu(overlay)
        overlay = yaml.safe_load(launcher.DEFAULT.read_text())
        overlay['dgpo']['adaptive_omnifold']['recalibration']['fit']['batch_size'] = 1024
        with self.assertRaises(ValueError):
            launcher.assert_nersc_16gpu(overlay)

    def test_live_ray_requires_address_and_16_gpus(self):
        env = {key: value for key, value in os.environ.items() if key != 'RAY_ADDRESS'}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                launcher.assert_live_ray_16gpu()
        fake_ray = mock.Mock()
        fake_ray.cluster_resources.return_value = {'GPU': 4}
        with mock.patch.dict(os.environ, {'RAY_ADDRESS': '10.0.0.1:6379'}):
            with mock.patch.dict(sys.modules, {'ray': fake_ray}):
                with self.assertRaises(RuntimeError):
                    launcher.assert_live_ray_16gpu()
                fake_ray.cluster_resources.return_value = {'GPU': 16}
                launcher.assert_live_ray_16gpu()
                fake_ray.init.assert_called()

    def test_refuses_control_overlay_as_config(self):
        with mock.patch.object(sys, 'argv', ['train_dgpo_iteration_one_tempered.py',
                                              '--config', str(launcher.CONTROL)]):
            with self.assertRaises(ValueError):
                launcher.main()

    def test_first_start_checks_source_and_empty_output(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, p, _ = _source_cfg(directory)
            with mock.patch('builtins.print'):
                self.assertEqual(launcher.verify(cfg), 'first_start')
            for key, value in [('source_dgpo_global_step', 320), ('source_policy_sha256', '0' * 64)]:
                bad = copy.deepcopy(cfg)
                bad['nersc']['reproducibility'][key] = value
                with self.assertRaises(ValueError):
                    launcher.verify(bad)
            wrong_clock = copy.deepcopy(cfg)
            torch.save({'epoch': 31, 'global_step': 320, 'dgpo_next_epoch': 32,
                        'state_dict': {'weight': torch.tensor([1., 2.])}},
                       Path(cfg['options']['Training']['model_checkpoint_load_path']))
            with self.assertRaises(ValueError):
                launcher.verify(wrong_clock)
            torch.save({'epoch': 25, 'global_step': 260, 'dgpo_next_epoch': 26,
                        'state_dict': {'weight': torch.tensor([1., 2.])},
                        'ema_state_dict': {'weight': torch.zeros(2)}},
                       Path(cfg['options']['Training']['model_checkpoint_load_path']))
            out = Path(cfg['options']['Training']['model_checkpoint_save_path'])
            out.mkdir(parents=True)
            torch.save({}, out / 'orphan.ckpt')
            with self.assertRaises(FileExistsError):
                launcher.verify(cfg)

    def test_resume_accepts_complete_last_ckpt(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, p, _ = _source_cfg(directory)
            out = Path(cfg['options']['Training']['model_checkpoint_save_path'])
            out.mkdir(parents=True)
            torch.save({
                'global_step': 40, 'dgpo_next_epoch': 4, 'dgpo_epoch_step': 0,
                'state_dict': {'weight': torch.ones(2)},
                'dgpo_checkpoint_version': 1, 'dgpo_optimizer_state_dict': {},
                'dgpo_round_ref_state_dict': {}, 'dgpo_adaptive_omnifold_state': {},
                'dgpo_omnifold_reward_stack': {'ok': True},
            }, out / 'last.ckpt')
            with mock.patch('builtins.print'):
                self.assertEqual(launcher.verify(cfg), 'resume')

    def test_incomplete_last_ckpt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, p, _ = _source_cfg(directory)
            out = Path(cfg['options']['Training']['model_checkpoint_save_path'])
            out.mkdir(parents=True)
            torch.save({'global_step': 1, 'state_dict': {}}, out / 'last.ckpt')
            with self.assertRaises(ValueError):
                launcher.verify(cfg)

    def test_refuses_step320_source_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, p, _ = _source_cfg(directory)
            cfg['options']['Training']['model_checkpoint_load_path'] = str(
                p / 'dgpo-epoch=31-next_ep=32-step=320.ckpt')
            with self.assertRaises(ValueError):
                launcher.verify(cfg)

    def test_refuses_control_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, p, _ = _source_cfg(directory)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(
                p / 'dgpo_omnifold_10pct_iteration1_only_step320_seed42' / 'checkpoints'
            )
            with self.assertRaises(ValueError):
                launcher.verify(cfg)

    def test_trust02_overlay_keeps_tempering_and_inherits_9073ef9e_stack(self):
        root = launcher.ROOT
        overlay = yaml.safe_load(launcher.TRUST02.read_text())
        d = overlay['dgpo']
        self.assertEqual(d['reference_trust']['coefficient'], 0.2)
        self.assertEqual(d['adaptive_omnifold']['recalibration']['tempering'], 0.75)
        self.assertFalse(d['adaptive_omnifold']['recalibration']['bootstrap_on_start'])
        self.assertEqual(d['checkpoint_load_mode'], 'resume')
        self.assertTrue(d['pinned_classifier_restart'])
        self.assertTrue(d['auto_resume_from_last'])
        self.assertEqual(
            d['adaptive_omnifold']['audit_fit']['training_readiness']['cold_start_min_epochs'],
            167,
        )
        self.assertEqual(
            d['adaptive_omnifold']['audit_fit']['training_readiness']['warm_start_min_epochs'],
            5,
        )
        self.assertEqual(d['adaptive_omnifold']['trigger']['required_consecutive_checks'], 8)
        self.assertEqual(
            d['adaptive_omnifold']['trigger']['patience_schedule'][0],
            {'start_step': 0, 'required_consecutive_checks': 8},
        )
        self.assertEqual(
            d['adaptive_omnifold']['trigger']['patience_schedule'][1],
            {'start_step': 100, 'required_consecutive_checks': 16},
        )
        load = overlay['options']['Training']['model_checkpoint_load_path']
        self.assertTrue(load.endswith(launcher.TRUST02_SNAPSHOT))
        self.assertIn('t075_trust05_step260', load)
        self.assertNotIn('last.ckpt', load)
        out = overlay['options']['Training']['model_checkpoint_save_path']
        self.assertIn('t075_trust02_step260', out)
        self.assertNotIn('t075_trust05_step260', out)
        launcher.assert_knobs(overlay)
        merged = deep_update(
            yaml.safe_load((root / 'config/train_diffusion_nersc.yaml').read_text()), overlay)
        launcher.assert_nersc_16gpu(merged)

    def test_trust02_knobs_refuse_9073ef9e_output_and_last_ckpt(self):
        overlay = yaml.safe_load(launcher.TRUST02.read_text())
        overlay['options']['Training']['model_checkpoint_save_path'] = (
            '/tmp/t075_trust05_step260/checkpoints')
        with self.assertRaises(ValueError):
            launcher.assert_knobs(overlay)
        overlay = yaml.safe_load(launcher.TRUST02.read_text())
        overlay['options']['Training']['model_checkpoint_load_path'] = (
            overlay['options']['Training']['model_checkpoint_load_path'].replace(
                launcher.TRUST02_SNAPSHOT, 'last.ckpt'))
        with self.assertRaises(ValueError):
            launcher.assert_knobs(overlay)

    def test_trust02_first_start_requires_complete_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            source = p / 't075_trust05_step260' / 'checkpoints' / launcher.TRUST02_SNAPSHOT
            source.parent.mkdir(parents=True)
            state = {'weight': torch.tensor([1., 2.])}
            payload = {
                'epoch': -1, 'global_step': 0, 'dgpo_next_epoch': 0,
                'state_dict': state,
                'dgpo_adaptive_omnifold_state': {},
                'dgpo_ref_state_dict': {}, 'dgpo_round_ref_state_dict': {},
                'dgpo_round_ref_sha256': 'x',
                'dgpo_omnifold_reward_stack': {'reward': {}},
                'dgpo_omnifold_reward_metadata': {},
            }
            torch.save(payload, source)
            cfg = yaml.safe_load(launcher.TRUST02.read_text())
            cfg['options']['Training']['model_checkpoint_load_path'] = str(source)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(
                p / 't075_trust02_step260' / 'checkpoints')
            cfg['nersc']['reproducibility']['source_policy_sha256'] = launcher.state_digest(state)
            cfg['reward_config']['omnifold']['backbone_checkpoint'] = str(source)
            cfg['platform']['data_parquet_dir'] = str(p)
            cfg['platform']['data_parquet_val_dir'] = str(p)
            with mock.patch('builtins.print'):
                self.assertEqual(launcher.verify(cfg), 'first_start')
            incomplete = copy.deepcopy(cfg)
            torch.save({'epoch': -1, 'global_step': 0, 'dgpo_next_epoch': 0,
                        'state_dict': state}, source)
            with self.assertRaises(ValueError):
                launcher.verify(incomplete)

    def test_dual_classifier_overlay_is_fresh_same_start_and_separated(self):
        overlay = yaml.safe_load(launcher.DUAL_CLASSIFIER.read_text())
        dgpo = overlay['dgpo']
        adaptive = dgpo['adaptive_omnifold']
        recal = adaptive['recalibration']
        self.assertEqual(dgpo['checkpoint_load_mode'], 'resume')
        self.assertTrue(dgpo.get('pinned_classifier_restart', False))
        self.assertFalse(recal['bootstrap_on_start'])
        self.assertEqual(
            overlay['nersc']['reproducibility']['source_dgpo_global_step'], 0
        )
        self.assertEqual(
            overlay['nersc']['reproducibility']['source_classifier_epoch'], -1
        )
        self.assertEqual(
            overlay['nersc']['reproducibility']['source_classifier_global_step'], 0
        )
        load = overlay['options']['Training']['model_checkpoint_load_path']
        self.assertTrue(load.endswith(launcher.DUAL_INHERIT_SNAPSHOT))
        self.assertIn(launcher.DUAL_INHERIT_TAG, load)
        self.assertNotIn('last.ckpt', load)
        self.assertIn(
            launcher.DIFFUSION_PRETRAIN,
            overlay['reward_config']['omnifold']['backbone_checkpoint'],
        )
        self.assertIn(launcher.DUAL_OUTPUT_TAG, overlay['options']['Training']['model_checkpoint_save_path'])
        self.assertEqual(overlay['logger']['wandb']['resume'], 'allow')
        self.assertFalse(overlay['logger']['wandb']['fresh_run'])
        self.assertEqual(
            overlay['logger']['wandb']['id'],
            launcher.DUAL_WANDB_ID,
        )
        self.assertNotEqual(overlay['logger']['wandb']['id'], 'd17994d9')
        self.assertFalse(dgpo['reference_trust']['adaptive_boundary']['enabled'])
        self.assertEqual(adaptive['trigger']['best_scope'], 'round')
        self.assertTrue(adaptive['trigger']['rollback_to_best_on_plateau'])
        self.assertEqual(adaptive['trigger']['required_consecutive_checks'], 4)
        self.assertEqual(
            adaptive['trigger']['patience_schedule'],
            [
                {'start_step': 0, 'required_consecutive_checks': 4},
                {'start_step': 100, 'required_consecutive_checks': 8},
            ],
        )
        self.assertFalse(recal['iteration_one_only'])
        self.assertEqual(
            (recal['min_iterations'], recal['max_iterations']), (1, 10)
        )
        self.assertEqual(
            recal['residual_closure_schedule'],
            [{'start_step': 0, 'max_auc': 0.52}],
        )
        self.assertEqual(dgpo['reference_trust']['coefficient'], 0.2)
        self.assertEqual(recal['log_ratio_clip'], 2.5)
        self.assertEqual(recal['minimum_ess_fraction'], 0.3)
        self.assertEqual(
            recal['adaptive_tempering'],
            {
                'enabled': True,
                'target_ess_fraction': 0.3,
                'minimum': 0.1,
                'grid_steps': 14,
                'inherit_previous': True,
            },
        )
        self.assertEqual(
            recal['ess_aware_checkpoint_selection'],
            {
                'enabled': True,
                'max_checkpoints': 16,
                'first_residual_only': True,
            },
        )
        self.assertEqual(
            recal['reward_classifier'],
            {
                'head_dropout': 0.15,
                'decoder_hidden_dim': 128,
                'decoder_layers': 1,
                'decoder_heads': 4,
                'periodic_pair_features': False,
                'topology_fourier_embedding': False,
                'topology_conditioning': False,
                'visible_pair_rest_frame': False,
            },
        )
        for key in (
            'periodic_pair_features',
            'topology_fourier_embedding',
            'topology_conditioning',
            'visible_pair_rest_frame',
        ):
            self.assertTrue(adaptive['audit_fit'][key])
        self.assertIn('RoundBestRollback', overlay['logger']['wandb']['tags'])
        self.assertIn('NoHardTrustBoundary', overlay['logger']['wandb']['tags'])
        self.assertIn('InitialRawPatience4', overlay['logger']['wandb']['tags'])
        self.assertIn('PinnedClassifierRestart', overlay['logger']['wandb']['tags'])
        launcher.assert_knobs(overlay)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['reference_trust']['adaptive_boundary']['enabled'] = True
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['trigger']['best_scope'] = 'global'
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['trigger']['patience_schedule'][0][
            'required_consecutive_checks'
        ] = 8
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['checkpoint_load_mode'] = 'weights_only'
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'bootstrap_on_start'
        ] = True
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['options']['Training']['model_checkpoint_load_path'] = (
            broken['options']['Training']['model_checkpoint_load_path'].replace(
                launcher.DUAL_INHERIT_SNAPSHOT, 'last.ckpt'
            )
        )
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'reward_classifier'
        ]['decoder_layers'] = 2
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'adaptive_tempering'
        ]['target_ess_fraction'] = .1
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'adaptive_tempering'
        ]['inherit_previous'] = False
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'ess_aware_checkpoint_selection'
        ]['first_residual_only'] = False
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'ess_aware_checkpoint_selection'
        ]['enabled'] = False
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)
        broken = copy.deepcopy(overlay)
        broken['dgpo']['adaptive_omnifold']['recalibration'][
            'residual_closure_schedule'
        ][0]['max_auc'] = .51
        with self.assertRaises(ValueError):
            launcher.assert_knobs(broken)

    def test_dual_classifier_verify_needs_complete_d17994d9_epoch_minus1_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            source = (
                p / launcher.DUAL_INHERIT_TAG / 'checkpoints'
                / launcher.DUAL_INHERIT_SNAPSHOT
            )
            source.parent.mkdir(parents=True)
            state = {'weight': torch.tensor([1., 2.])}
            payload = {
                'epoch': -1,
                'global_step': 0,
                'dgpo_next_epoch': 0,
                'state_dict': state,
                'dgpo_adaptive_omnifold_state': {},
                'dgpo_ref_state_dict': {},
                'dgpo_round_ref_state_dict': {},
                'dgpo_round_ref_sha256': 'x',
                'dgpo_omnifold_reward_stack': {'reward': {}},
                'dgpo_omnifold_reward_metadata': {},
            }
            torch.save(payload, source)
            cfg = yaml.safe_load(launcher.DUAL_CLASSIFIER.read_text())
            cfg['options']['Training']['model_checkpoint_load_path'] = str(source)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(
                p / launcher.DUAL_OUTPUT_TAG / 'checkpoints'
            )
            cfg['nersc']['reproducibility']['source_policy_sha256'] = (
                launcher.state_digest(state)
            )
            cfg['reward_config']['omnifold']['backbone_checkpoint'] = str(
                p / launcher.DIFFUSION_PRETRAIN
            )
            Path(cfg['reward_config']['omnifold']['backbone_checkpoint']).parent.mkdir(
                parents=True, exist_ok=True
            )
            Path(cfg['reward_config']['omnifold']['backbone_checkpoint']).write_text('x')
            cfg['platform']['data_parquet_dir'] = str(p)
            cfg['platform']['data_parquet_val_dir'] = str(p)
            with mock.patch('builtins.print'):
                self.assertEqual(launcher.verify(cfg), 'first_start')
            incomplete = copy.deepcopy(cfg)
            torch.save(
                {
                    'epoch': -1,
                    'global_step': 0,
                    'dgpo_next_epoch': 0,
                    'state_dict': state,
                },
                source,
            )
            with self.assertRaises(ValueError):
                launcher.verify(incomplete)
            wrong = copy.deepcopy(cfg)
            torch.save(payload, source)
            wrong['options']['Training']['model_checkpoint_load_path'] = str(
                p / 'diffusion_pretrain_10pct_seed42' / 'checkpoints' / 'last.ckpt'
            )
            with self.assertRaises(ValueError):
                launcher.assert_knobs(wrong)

    def test_next_round_patience_schedule_resolves_from_step(self):
        trigger = yaml.safe_load(launcher.DUAL_CLASSIFIER.read_text())[
            'dgpo'
        ]['adaptive_omnifold']['trigger']
        self.assertEqual(
            launcher._effective_patience(trigger, global_step=0),
            4,
        )
        self.assertEqual(
            launcher._effective_patience(trigger, global_step=99),
            4,
        )
        self.assertEqual(
            launcher._effective_patience(trigger, global_step=100),
            8,
        )


if __name__ == '__main__':
    unittest.main()
