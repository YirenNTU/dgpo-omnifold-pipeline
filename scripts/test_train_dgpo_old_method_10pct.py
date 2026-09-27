"""CPU-only guards for the 10% old-method hard ablation launcher."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

import train_dgpo_old_method_10pct as launcher
from train_neutrino_backend import deep_update, read_overlay_yaml


class OldMethod10pctTests(unittest.TestCase):
    def setUp(self):
        self.overlay = yaml.safe_load(launcher.DEFAULT.read_text())
        self.control = yaml.safe_load(launcher.CONTROL_5PCT.read_text())

    def test_overlay_inheritance_deep_merges_without_leaking_directive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'base.yaml').write_text(
                'top:\n  keep: 1\n  replace: old\n'
            )
            (root / 'child.yaml').write_text(
                'extends_overlay: base.yaml\n'
                'top:\n  replace: new\n'
                'added: true\n'
            )
            self.assertEqual(
                read_overlay_yaml(root / 'child.yaml'),
                {
                    'top': {'keep': 1, 'replace': 'new'},
                    'added': True,
                },
            )

    def test_overlay_matches_nc4shnpg_controller_contract(self):
        launcher.assert_knobs(self.overlay)
        adaptive = self.overlay['dgpo']['adaptive_omnifold']
        trigger = adaptive['trigger']
        recal = adaptive['recalibration']
        trust = self.overlay['dgpo']['reference_trust']

        self.assertEqual(adaptive['monitor_mode'], 'weighted_and_raw')
        self.assertFalse(adaptive['single_pool_train_validation'])
        self.assertEqual(adaptive['staleness_every_n_epochs'], 5)
        self.assertEqual(adaptive['audit_fit']['head_dropout'], 0.25)
        self.assertFalse(adaptive['audit_fit'].get('train_layernorm'))
        self.assertFalse(adaptive['audit_fit'].get('train_encoder'))
        self.assertTrue(adaptive['audit_fit']['train_grouped_sequential_embedding'])
        self.assertTrue(adaptive['audit_fit']['train_invisible_projector'])
        self.assertFalse(adaptive['audit_fit'].get('train_backbone'))
        self.assertTrue(adaptive['audit_fit']['asymmetric_attention'])
        self.assertEqual(trigger['retrain_auc_margin'], 0.005)
        self.assertEqual(trigger['required_consecutive_epochs'], 1)
        self.assertEqual(trigger['max_reward_age_epochs'], 20)
        self.assertFalse(trigger.get('rollback_to_best_on_plateau', False))
        self.assertFalse(trigger.get('warm_start_classifier', False))
        self.assertTrue(trigger['raw_audit_enabled'])
        self.assertEqual(trigger['probe_max_events'], 250000)

        self.assertEqual(trust['coefficient'], 1.0)
        self.assertEqual(trust['objective'], 'velocity_mse')
        self.assertFalse(trust['adaptive_boundary']['enabled'])

        self.assertEqual(recal['tempering'], 0.75)
        self.assertEqual((recal['min_iterations'], recal['max_iterations']), (2, 5))
        self.assertTrue(recal['acceptance_audit_enabled'])
        self.assertEqual(recal['acceptance_max_balanced_accuracy'], 0.52)
        self.assertEqual(recal['residual_min_auc_gain'], 0.01)
        self.assertEqual(recal['score_pool_events'], 250000)
        self.assertEqual(recal['warm_start_iterations'], [])
        self.assertFalse(recal.get('warm_start_from_iteration_one', False))
        self.assertTrue(recal['bootstrap_on_start'])
        self.assertFalse(recal.get('refit_once_on_resume', False))
        self.assertTrue(self.overlay['dgpo']['auto_resume_from_last'])
        self.assertEqual(self.overlay['dgpo']['checkpoint_load_mode'], 'weights_only')
        self.assertIsNone(self.overlay['dgpo'].get('auto_resume_fallback_checkpoint_path'))
        self.assertEqual(recal['crossfit_partition'], 'auto')
        self.assertIsNone(recal['log_ratio_clip'])
        self.assertFalse(recal['adaptive_tempering']['enabled'])
        self.assertEqual((recal['decoder_hidden_dim'], recal['decoder_layers']), (128, 1))
        self.assertTrue(recal['train_grouped_sequential_embedding'])
        self.assertTrue(recal['train_invisible_projector'])
        self.assertEqual(recal['head_dropout'], 0.25)
        self.assertEqual(adaptive['audit_fit']['head_dropout'], 0.25)
        self.assertGreaterEqual(recal['fit']['min_steps'], 1)
        self.assertTrue(recal['fit']['require_saturation'])
        # Audit mirrors OmniFold fit protocol + decoder; only require_saturation
        # stays false because fit_fresh_audit forces cheap probes.
        for key in (
            'learning_rate', 'backbone_learning_rate', 'min_steps',
            'gradient_clip_norm', 'batch_size', 'sampling', 'restore_best',
        ):
            self.assertEqual(
                adaptive['audit_fit'][key], recal['fit'][key], msg=key
            )
        for key in (
            'decoder_hidden_dim', 'decoder_layers', 'decoder_heads', 'head_dropout',
        ):
            self.assertEqual(
                adaptive['audit_fit'][key], recal[key], msg=key
            )
        self.assertFalse(adaptive['audit_fit']['require_saturation'])
        self.assertEqual(
            self.overlay['options']['Training']['Components']['GroupedSequentialEmbedding'][
                'learning_rate'
            ],
            self.overlay['options']['Training']['Components']['InvisibleInputProjector'][
                'learning_rate'
            ],
        )
        self.assertFalse(recal['periodic_pair_features'])
        self.assertFalse(recal['topology_fourier_embedding'])
        self.assertFalse(recal['visible_pair_rest_frame'])
        self.assertEqual(self.overlay['options']['Training']['learning_rate'], 0.0001)
        self.assertEqual(self.overlay['logger']['wandb']['id'], launcher.WANDB_ID)
        self.assertEqual(self.overlay['logger']['wandb']['resume'], 'allow')
        self.assertFalse(self.overlay['logger']['wandb']['fresh_run'])
        self.assertTrue(self.overlay['dgpo']['ztautau_metrics']['log_images'])

        # Core controller knobs match the original 5% control YAML.
        control_adaptive = self.control['dgpo']['adaptive_omnifold']
        control_trigger = control_adaptive['trigger']
        control_recal = control_adaptive['recalibration']
        self.assertEqual(
            adaptive['staleness_every_n_epochs'],
            control_adaptive['staleness_every_n_epochs'],
        )
        self.assertEqual(
            trigger['retrain_auc_margin'], control_trigger['retrain_auc_margin']
        )
        self.assertEqual(
            trigger['required_consecutive_epochs'],
            control_trigger['required_consecutive_epochs'],
        )
        self.assertEqual(
            trigger['max_reward_age_epochs'], control_trigger['max_reward_age_epochs']
        )
        self.assertEqual(recal['tempering'], control_recal['tempering'])
        self.assertEqual(
            (recal['min_iterations'], recal['max_iterations']),
            (control_recal['min_iterations'], control_recal['max_iterations']),
        )
        # 10%+group-embed arm uses a slightly looser acceptance gate than nc4shnpg.
        self.assertEqual(control_recal['acceptance_max_balanced_accuracy'], 0.51)
        self.assertEqual(recal['acceptance_max_balanced_accuracy'], 0.52)
        self.assertEqual(
            self.overlay['dgpo']['reference_trust']['coefficient'],
            self.control['dgpo']['reference_trust']['coefficient'],
        )

    def test_merged_runtime_keeps_old_method_knobs(self):
        base = yaml.safe_load(launcher.BASE.read_text())
        merged = deep_update(base, self.overlay)
        launcher.assert_knobs(merged)
        self.assertEqual(merged['platform']['number_of_workers'], 16)
        self.assertEqual(merged['platform']['resources_per_worker']['GPU'], 1)
        self.assertTrue(merged['platform']['use_gpu'])
        self.assertIn(launcher.STIC_TRAIN, merged['platform']['data_parquet_dir'])
        self.assertIn(launcher.HELD_OUT_VAL, merged['platform']['data_parquet_val_dir'])
        self.assertIn(launcher.OUTPUT_TAG, merged['options']['Training']['model_checkpoint_save_path'])
        self.assertIn('OldMethodHardAblation', merged['logger']['wandb']['tags'])
        self.assertIn('NoHardTrustBoundary', merged['logger']['wandb']['tags'])
        self.assertIn('NoClassifierInherit', merged['logger']['wandb']['tags'])
        self.assertTrue(merged['dgpo']['tarp']['enabled'])

    def test_rejects_hard_trust_or_inherit(self):
        bad = copy.deepcopy(self.overlay)
        bad['dgpo']['reference_trust']['adaptive_boundary']['enabled'] = True
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

        bad = copy.deepcopy(self.overlay)
        bad['dgpo']['adaptive_omnifold']['recalibration']['warm_start_iterations'] = [1]
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

        bad = copy.deepcopy(self.overlay)
        bad['dgpo']['pinned_classifier_restart'] = True
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

        bad = copy.deepcopy(self.overlay)
        bad['dgpo']['adaptive_omnifold']['recalibration']['crossfit_partition'] = 'random'
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

        bad = copy.deepcopy(self.overlay)
        bad['dgpo']['adaptive_omnifold']['single_pool_train_validation'] = True
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

        bad = copy.deepcopy(self.overlay)
        bad['dgpo']['adaptive_omnifold']['monitor_mode'] = 'raw_plateau_refit'
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

    def test_rejects_wrong_data_or_output(self):
        bad = copy.deepcopy(self.overlay)
        bad['platform']['data_parquet_val_dir'] = bad['platform']['data_parquet_dir']
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

        bad = copy.deepcopy(self.overlay)
        bad['options']['Training']['model_checkpoint_save_path'] = (
            '/tmp/dgpo_omnifold_10pct_dual_classifier_seed42/checkpoints'
        )
        with self.assertRaises(ValueError):
            launcher.assert_knobs(bad)

    def test_resolve_launch_mode_first_start_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'checkpoints'
            out.mkdir()
            cfg = copy.deepcopy(self.overlay)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(out)
            self.assertEqual(
                launcher.resolve_launch_mode(cfg, check_filesystem=True),
                'first_start',
            )
            incomplete = {key: 1 for key in launcher.RECOVERY_KEYS if key != 'state_dict'}
            torch.save(incomplete, out / 'last.ckpt')
            with self.assertRaises(ValueError):
                launcher.resolve_launch_mode(cfg, check_filesystem=True)
            complete = {key: 0 for key in launcher.RECOVERY_KEYS}
            complete.update(
                {
                    'global_step': 12,
                    'dgpo_next_epoch': 3,
                    'dgpo_epoch_step': 0,
                    'dgpo_reward_round_id': 1,
                }
            )
            torch.save(complete, out / 'last.ckpt')
            self.assertEqual(
                launcher.resolve_launch_mode(cfg, check_filesystem=True),
                'resume',
            )

    def test_resolve_launch_mode_rejects_orphan_ckpts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'checkpoints'
            out.mkdir()
            torch.save({'state_dict': {}}, out / 'step=0.ckpt')
            cfg = copy.deepcopy(self.overlay)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(out)
            with self.assertRaises(FileExistsError):
                launcher.resolve_launch_mode(cfg, check_filesystem=True)

    def test_h4_parity_fork_overlay_contract(self):
        fork = yaml.safe_load(launcher.H4_PARITY.read_text())
        launcher.assert_knobs(fork)
        training = fork['options']['Training']
        adaptive = fork['dgpo']['adaptive_omnifold']
        recal = fork['dgpo']['adaptive_omnifold']['recalibration']
        self.assertEqual(fork['dgpo']['checkpoint_load_mode'], 'resume')
        self.assertTrue(fork['dgpo']['auto_resume_from_last'])
        self.assertFalse(recal['bootstrap_on_start'])
        self.assertTrue(recal['refit_once_on_resume'])
        self.assertFalse(recal['refit_once_fail_closed'])
        self.assertEqual(recal['refit_once_id'], launcher.H4_PARITY_REFIT_ONCE_ID)
        self.assertEqual(recal['head_dropout'], 0.15)
        self.assertTrue(recal['periodic_pair_features'])
        self.assertTrue(recal['topology_fourier_embedding'])
        self.assertFalse(recal['topology_direct_logit'])
        self.assertEqual(recal['topology_context_residual_scale'], 1.0)
        self.assertFalse(recal['topology_conditioning'])
        self.assertFalse(recal['visible_pair_rest_frame'])
        self.assertEqual(recal['topology_max_harmonic'], 4)
        self.assertFalse(recal['topology_include_theta_pair'])
        self.assertEqual(recal['topology_hidden_dim'], 64)
        self.assertEqual(recal['topology_embedding_dim'], 32)
        self.assertEqual(recal['topology_fusion_hidden_dim'], 64)
        self.assertEqual(recal['topology_dropout'], 0.15)
        self.assertEqual(recal['warm_start_iterations'], [1])
        self.assertTrue(recal['warm_start_from_iteration_one'])
        self.assertEqual(recal['crossfit_partition'], 'identity')
        self.assertEqual(recal['fit']['min_steps'], 1)
        self.assertEqual(recal['fit']['min_steps_per_fold'], 1000)
        self.assertEqual(recal['fit']['warm_start_min_epochs_per_fold'], 10)
        self.assertEqual(recal['fit']['min_epochs'], 1)
        self.assertFalse(recal['fit']['enforce_min_epochs'])
        self.assertEqual(recal['fit']['validation_patience_epochs'], 15.0)
        self.assertEqual(recal['fit']['topology_warmup_steps'], 0)
        self.assertEqual(recal['fit']['topology_body_unfreeze_step'], 0)
        self.assertEqual(recal['fit']['topology_warmup_learning_rate'], 0.001)
        self.assertEqual(recal['fit']['validation_min_delta'], 0.0001)
        self.assertFalse(recal['train_layernorm'])
        self.assertFalse(recal['train_encoder'])
        self.assertFalse(recal['train_backbone'])
        audit = adaptive['audit_fit']
        self.assertEqual(adaptive['staleness_every_n_epochs'], 1)
        self.assertTrue(adaptive['fixed_audit_panel'])
        self.assertTrue(adaptive['cache_event_inputs'])
        self.assertEqual(
            adaptive['trigger']['required_consecutive_epochs'],
            2,
        )
        self.assertEqual(adaptive['trigger']['probe_seed'], recal['seed'])
        self.assertTrue(adaptive['trigger']['warm_start_classifier'])
        self.assertEqual(adaptive['trigger']['retrain_cooldown_epochs'], 5)
        self.assertTrue(audit['periodic_pair_features'])
        self.assertTrue(audit['topology_fourier_embedding'])
        self.assertEqual(audit['topology_max_harmonic'], 4)
        self.assertFalse(audit['topology_direct_logit'])
        self.assertEqual(audit['topology_context_residual_scale'], 1.0)
        self.assertFalse(audit['topology_include_theta_pair'])
        self.assertEqual(audit['head_dropout'], 0.15)
        self.assertEqual(audit['min_steps'], 300)
        self.assertEqual(audit['validation_patience_epochs'], 15.0)
        self.assertEqual(
            audit['training_readiness'],
            {'cold_start_min_epochs': 100, 'warm_start_min_epochs': 5},
        )
        self.assertTrue(audit['disjoint_final_audit'])
        self.assertEqual(audit['topology_warmup_steps'], 0)
        self.assertEqual(audit['topology_body_unfreeze_step'], 0)
        self.assertEqual(audit['topology_warmup_learning_rate'], 0.001)
        self.assertFalse(audit['train_layernorm'])
        self.assertFalse(audit['train_encoder'])
        self.assertTrue(audit['train_grouped_sequential_embedding'])
        self.assertTrue(audit['train_invisible_projector'])
        self.assertFalse(audit['train_backbone'])
        self.assertTrue(audit['asymmetric_attention'])
        self.assertEqual(audit['topology_hidden_dim'], 64)
        self.assertEqual(audit['topology_embedding_dim'], 32)
        self.assertEqual(audit['topology_fusion_hidden_dim'], 64)
        self.assertEqual(audit['topology_dropout'], 0.15)
        self.assertIn(launcher.PARENT_LAST, training['model_checkpoint_load_path'])
        self.assertIn(
            launcher.H4_PARITY_OUTPUT_TAG, training['model_checkpoint_save_path']
        )
        self.assertNotEqual(
            Path(training['model_checkpoint_load_path']).parent,
            Path(training['model_checkpoint_save_path']),
        )
        self.assertEqual(fork['logger']['wandb']['id'], launcher.H4_PARITY_WANDB_ID)
        self.assertFalse(fork['logger']['wandb']['classifier_loss_curves'])
        self.assertIn('ForkFromC4a91e07', fork['logger']['wandb']['tags'])
        self.assertIn('CompactWandbCoreMetrics', fork['logger']['wandb']['tags'])
        self.assertIn('HeadDropout015', fork['logger']['wandb']['tags'])
        self.assertIn('PeriodicPairPhiH4', fork['logger']['wandb']['tags'])
        self.assertIn('NoDirectFourierShortcut', fork['logger']['wandb']['tags'])
        self.assertIn('FixedPanelWarmRawMonitor', fork['logger']['wandb']['tags'])
        self.assertIn('FrozenLayerNorm', fork['logger']['wandb']['tags'])
        self.assertIn('FrozenGlobalEmbedding', fork['logger']['wandb']['tags'])
        self.assertIn('NoThetaPair', fork['logger']['wandb']['tags'])

    def test_legacy_finetuned_body_ablation_contract(self):
        config = read_overlay_yaml(launcher.FINETUNED_BODY)
        launcher.assert_knobs(config)
        adaptive = config['dgpo']['adaptive_omnifold']
        recal = adaptive['recalibration']
        source = config['options']['Training']['model_checkpoint_load_path']
        self.assertIn(launcher.DIFFUSION_PRETRAIN, source)
        self.assertEqual(
            config['reward_config']['omnifold']['backbone_checkpoint'],
            source,
        )
        self.assertTrue(recal['body_only_checkpoint'])
        self.assertTrue(recal['bootstrap_on_start'])
        self.assertFalse(recal['refit_once_on_resume'])
        self.assertFalse(recal['periodic_pair_features'])
        self.assertFalse(recal['topology_fourier_embedding'])
        self.assertEqual(recal['head_dropout'], 0.25)
        self.assertFalse(recal['train_layernorm'])
        self.assertFalse(recal['train_encoder'])
        self.assertTrue(recal['train_grouped_sequential_embedding'])
        self.assertTrue(recal['train_invisible_projector'])
        self.assertFalse(recal['train_backbone'])
        self.assertEqual(
            config['logger']['wandb']['id'],
            launcher.FINETUNED_BODY_WANDB_ID,
        )
        self.assertFalse(config['logger']['wandb']['classifier_loss_curves'])

    def test_resolve_launch_mode_finetuned_body_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'parent.ckpt'
            torch.save({'state_dict': {'model.weight': torch.ones(1)}}, source)
            config = read_overlay_yaml(launcher.FINETUNED_BODY)
            config['options']['Training']['model_checkpoint_load_path'] = str(source)
            config['options']['Training']['model_checkpoint_save_path'] = str(
                root / launcher.FINETUNED_BODY_OUTPUT_TAG / 'checkpoints'
            )
            self.assertEqual(
                launcher.resolve_launch_mode(config, check_filesystem=True),
                'body_ablation_start',
            )

    def test_resolve_launch_mode_fork_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / 'parent' / 'checkpoints'
            parent.mkdir(parents=True)
            out = Path(tmp) / f'fork_{launcher.H4_PARITY_OUTPUT_TAG}' / 'checkpoints'
            out.mkdir(parents=True)
            complete = {key: 0 for key in launcher.RECOVERY_KEYS}
            complete.update(
                {
                    'global_step': 40,
                    'dgpo_next_epoch': 8,
                    'dgpo_epoch_step': 2,
                    'dgpo_reward_round_id': 3,
                }
            )
            parent_last = parent / 'last.ckpt'
            torch.save(complete, parent_last)
            cfg = copy.deepcopy(yaml.safe_load(launcher.H4_PARITY.read_text()))
            cfg['options']['Training']['model_checkpoint_load_path'] = str(parent_last)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(out)
            self.assertEqual(
                launcher.resolve_launch_mode(cfg, check_filesystem=True),
                'fork_start',
            )
            torch.save(complete, out / 'last.ckpt')
            self.assertEqual(
                launcher.resolve_launch_mode(cfg, check_filesystem=True),
                'resume',
            )

    def test_check_only_launcher(self):
        with mock.patch('train_dgpo_old_method_10pct.subprocess.run') as run:
            launcher.main(['--check-only', '--skip-filesystem-checks'])
        run.assert_not_called()

    def test_launch_invokes_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp) / 'runtime.yaml'
            dep = Path(tmp) / 'dep.yaml'
            dep.write_text('ok: true\n')
            runtime.write_text(f'event_info:\n  default: {dep}\n')
            with mock.patch(
                'train_dgpo_old_method_10pct.build_runtime_config', return_value=runtime
            ), mock.patch(
                'train_dgpo_old_method_10pct.verify_paths'
            ), mock.patch(
                'train_dgpo_old_method_10pct.resolve_launch_mode', return_value='first_start'
            ), mock.patch(
                'train_dgpo_old_method_10pct.subprocess.run'
            ) as run:
                launcher.main(['--skip-filesystem-checks'])
            command = run.call_args.args[0]
            self.assertEqual(
                command[command.index('--overlay-config') + 1],
                str(launcher.DEFAULT.resolve()),
            )
            self.assertEqual(command[command.index('--backend') + 1], 'dgpo-evenet')


if __name__ == '__main__':
    unittest.main()
