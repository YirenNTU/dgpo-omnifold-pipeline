import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np

from scripts.conditional_tau_preprocessing import (
    apply_masked_feature, fit_masked_feature, layout, particle_view,
    preprocessing_report, ratio_group_report,
)
from scripts.run_conditional_tau_ratio import command, read_config


class MaskedConditionTest(unittest.TestCase):
    def test_worker_trains_validates_and_saves_normalization(self):
        # CPU smoke test of the real worker; distributed transport is mocked.
        import torch
        from scripts.train_conditional_spin_ratio import training_worker
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/'config/conditional_tau_ratio_masked_norm_10pct.yaml')['classifier'].copy()
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            prepared = directory/'prepared.npz'
            np.savez(prepared, condition=rng.normal(size=(32,24)).astype('float32'),
                candidate_truth=rng.normal(size=(32,15)).astype('float32'),
                candidate_generated=rng.normal(size=(32,15)).astype('float32'),
                event_weight=np.ones(32), split=np.r_[np.zeros(16),np.ones(16)],
                condition_mean=np.zeros(24,dtype='float32'),condition_scale=np.ones(24,dtype='float32'))
            cfg.update(prepared=str(prepared), checkpoint=str(directory/'best.pt'),
                epochs=2, batch_size=8, representation='tau', packing_spec={'shapes':{}})
            context = Mock()
            context.get_world_rank.return_value = 0
            context.get_world_size.return_value = 1
            with patch('ray.train.get_context', return_value=context), \
                 patch('ray.train.torch.get_device', return_value=torch.device('cpu')), \
                 patch('ray.train.torch.prepare_model', side_effect=lambda m:m), \
                 patch('torch.distributed.all_reduce'), patch('torch.distributed.broadcast'), \
                 patch('ray.train.report') as report:
                training_worker(cfg)
            saved = torch.load(cfg['checkpoint'], weights_only=True)
            self.assertEqual(saved['condition_normalization'], 'masked_feature')
            self.assertEqual(report.call_count, 2)
            self.assertIn('val_ratio_ess_fraction', report.call_args.args[0])

    def fixture(self):
        # Deliberately nonstandard packing order: no hard-coded 464 offset.
        spec = {'shapes': {'conditions':[1], 'x_mask':[2], 'x':[2,2], 'conditions_mask':[1]}}
        spans, width = layout(spec)
        values = np.zeros((4, width + 16), dtype=np.float32)
        values[:, spans['x_mask'][0]] = [[1,0],[1,1],[1,1],[1,0]]
        values[:, spans['conditions_mask'][0]] = [[1],[1],[0],[1]]
        values[:, spans['conditions'][0]] = [[2],[4],[1e9],[1000]]
        values[:, spans['x'][0]] = [[2,0,1e9,1e9], [4,1,6,0], [8,1,10,0], [1000,1,1e9,1e9]]
        values[:, width:] = np.eye(16, dtype=np.float32)[:4]
        return spec, values, np.array([True,True,True,False])

    def test_fit_uses_only_valid_training_particles_and_shares_slot_statistics(self):
        spec, values, fit = self.fixture()
        mean, scale = fit_masked_feature(values, fit, spec)
        xs = layout(spec)[0]['x'][0]
        np.testing.assert_allclose(mean[xs], [6,0,6,0])
        np.testing.assert_allclose(scale[xs], [np.std([2,4,6,8,10]),1]*2)
        perturbed = values.copy()
        perturbed[-1, xs] *= 100
        m2, s2 = fit_masked_feature(perturbed, fit, spec)
        np.testing.assert_array_equal(mean, m2)
        np.testing.assert_array_equal(scale, s2)

    def test_padding_zero_masks_and_binary_features_preserved(self):
        spec, values, fit = self.fixture()
        mean, scale = fit_masked_feature(values, fit, spec)
        out = apply_masked_feature(values, mean, scale, spec)
        x, mask = particle_view(out, spec)
        np.testing.assert_array_equal(x[~mask], np.zeros((2,2)))
        np.testing.assert_array_equal(x[...,1][mask], [0,1,0,1,0,1])
        spans, width = layout(spec)
        np.testing.assert_array_equal(out[:,width:], values[:,width:])
        self.assertEqual(out[2,spans['conditions'][0]][0], 0)
        report = preprocessing_report(out, np.array([0,0,1,2]), spec)
        self.assertEqual(report['splits'][0]['valid_feature_clip_fraction'], 0.)
        self.assertGreater(report['splits'][2]['valid_feature_clip_fraction'], 0.)
        self.assertTrue(all(r['padding_nonzero_values']==0 for r in report['splits']))

    def test_common_value_has_same_transform_in_different_slots(self):
        spec, values, fit = self.fixture()
        mean, scale = fit_masked_feature(values, fit, spec)
        values[1,layout(spec)[0]['x'][0]] = [7,0,7,0]
        out, _ = particle_view(apply_masked_feature(values, mean, scale, spec), spec)
        np.testing.assert_array_equal(out[1,0], out[1,1])

    def test_group_normalization_exposes_condition_mass_shift(self):
        r = ratio_group_report(np.ones(4), np.log([2,2,1,1]), np.zeros(4),
            np.array([11,11,22,22]), np.array([2,2,3,3]))
        group = next(x for x in r['groups'] if x['group']=='category_11')
        self.assertAlmostEqual(group['log_mean_ratio'], np.log(2))
        self.assertAlmostEqual(group['base_fraction'], .5)
        self.assertAlmostEqual(group['reweighted_fraction'], 2/3)

    def test_ablation_keeps_all_classifier_hyperparameters(self):
        root = Path(__file__).resolve().parents[1]
        baseline = read_config(root/'config/conditional_tau_ratio_10pct.yaml')
        new = read_config(root/'config/conditional_tau_ratio_masked_norm_10pct.yaml')
        settings = new['classifier'].copy()
        self.assertEqual(settings.pop('condition_normalization'), 'masked_feature')
        self.assertEqual(settings, baseline['classifier'])
        self.assertEqual(new['platform'], baseline['platform'])
        cmd = command(new, 'train')
        self.assertEqual(cmd[cmd.index('--condition-normalization')+1], 'masked_feature')
        self.assertEqual(cmd[cmd.index('--baseline-directory')+1], baseline['experiment']['output'])


if __name__ == '__main__':
    unittest.main()
