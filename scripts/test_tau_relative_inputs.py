import copy
from pathlib import Path
import unittest

import numpy as np
import torch

from scripts.tau_relative_inputs import relative_angles, attach_relative_inputs, paired_input_contract
from scripts.train_conditional_spin_ratio import ConditionalSpinMLP, paired_loss
from scripts.run_conditional_tau_ratio import read_config, command


def p4(theta, phi):
    theta, phi = np.broadcast_arrays(np.asarray(theta), np.asarray(phi))
    return np.stack((np.full_like(theta, 45.6), np.sin(theta)*np.cos(phi),
                     np.sin(theta)*np.sin(phi), np.cos(theta)), axis=-1)


class RelativeInputsTest(unittest.TestCase):
    def test_angles_and_periodic_boundary(self):
        v = p4([1., .7], [np.pi-.01, .2])
        t = p4([1.1, .9], [-np.pi+.02, .6])
        features = relative_angles(t, t, v, v)
        expected = np.array([[.1, np.sin(.03), np.cos(.03)], [.2, np.sin(.4), np.cos(.4)]])
        np.testing.assert_allclose(features[:, :3], expected, atol=1e-14)
        np.testing.assert_allclose(features[:, 3:], expected, atol=1e-14)
        # Candidate == visible gives no angular displacement in both classes.
        np.testing.assert_allclose(relative_angles(v, v, v, v), [[0,0,1,0,0,1]]*2, atol=1e-14)

    def test_canonical_p4_not_raw_wrapped_offset(self):
        v = p4([.2], [.5])
        a = p4([.3], [.7])
        b = p4([.3+2*np.pi], [.7+2*np.pi])
        np.testing.assert_allclose(relative_angles(a,a,v,v), relative_angles(b,b,v,v), atol=1e-14)

    def test_invalid_direction_rejected(self):
        with self.assertRaisesRegex(ValueError, 'zero-momentum'):
            relative_angles(np.zeros((2,4)), np.ones((2,4)), np.ones((2,4)), np.ones((2,4)))

    def arrays(self):
        rng = np.random.default_rng(8)
        return dict(split=np.array([0]*12+[1]*4+[2]*4),
            relative_truth=rng.normal(size=(20,6)), relative_generated=rng.normal(size=(20,6)),
            candidate_truth=rng.normal(size=(20,23)).astype('float32'),
            candidate_generated=rng.normal(size=(20,23)).astype('float32'))

    def test_fit_only_shared_stats_and_mmd_geometry_unchanged(self):
        a = self.arrays(); b = copy.deepcopy(a)
        b['relative_truth'][12:] = 1e8
        b['relative_generated'][12:] = -1e8
        before = copy.deepcopy(a)
        pa, pb = attach_relative_inputs(a), attach_relative_inputs(b)
        self.assertEqual(pa, pb)
        for key in ('candidate_truth','candidate_generated'):
            np.testing.assert_array_equal(a[key][:, -15:], before[key][:, -15:])
            np.testing.assert_array_equal(a[key][:, :-21], before[key][:, :-15])
            self.assertEqual(a[key].shape, (20,29))
        fit = np.concatenate((a['candidate_truth'][:12,-21:-15], a['candidate_generated'][:12,-21:-15]))
        np.testing.assert_allclose(fit.mean(0), 0., atol=1e-7)
        np.testing.assert_allclose(fit.std(0), 1., atol=1e-6)

    def test_matched_initialization_rng_and_new_input_learning(self):
        torch.manual_seed(42)
        base = ConditionalSpinMLP(7, hidden=16, dropout=.05, candidate_dim=23)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new = ConditionalSpinMLP(7, hidden=16, dropout=.05, candidate_dim=29, relative_dim=6)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            actual = new.state_dict()[key]
            if key == 'spin_encoder.0.weight':
                actual = torch.cat((actual[:, :-21], actual[:, -15:]), 1)
            torch.testing.assert_close(actual, value, rtol=0, atol=0)
        c, a, b = torch.randn(16,7), torch.randn(16,23), torch.randn(16,23)
        def add(x):
            return torch.cat((x[:, :-15], torch.randn(16,6), x[:, -15:]), 1)
        aa, bb = add(a), add(b)
        base.eval(); new.eval()
        torch.testing.assert_close(base(c,a), new(c,aa), rtol=1e-6, atol=1e-7)
        loss, _, _ = paired_loss(new, c, aa, bb, torch.ones(16))
        loss.backward()
        self.assertGreater(new.spin_encoder[0].weight.grad[:, -21:-15].norm().item(), 0.)
        torch.optim.AdamW(new.parameters(), lr=2e-4).step()
        self.assertGreater(new.spin_encoder[0].weight[:, -21:-15].norm().item(), 0.)
        # Saved architecture metadata is sufficient to reload the expanded head.
        restored = ConditionalSpinMLP(7, 16, .05, 29, relative_dim=6).eval()
        restored.load_state_dict(new.state_dict())
        torch.testing.assert_close(restored(c, aa), new(c, aa))

    def test_yaml_matches_control_and_no_resampling(self):
        root = Path(__file__).resolve().parents[1]
        cfg = read_config(root/'config/conditional_tau_relative_input_10pct.yaml')
        old = read_config(root/'config/conditional_tau_backbone_alignment_10pct.yaml')
        for key, value in old['classifier'].items():
            if key == 'batch_size':
                self.assertEqual(cfg['classifier'][key], 1024)
                continue
            self.assertEqual(cfg['classifier'][key], value)
        self.assertEqual(cfg['classifier']['backbone_cache'], old['alignment']['cache'])
        args = command(cfg, 'train')
        self.assertIn('--relative-angle-input', args)
        self.assertEqual(args[args.index('--mmd-coefficient')+1], '0.0')
        self.assertEqual(args[args.index('--baseline-run')+1], '5ugyt4h0')
        self.assertEqual(args[args.index('--workers')+1], '16')
        self.assertEqual(args[args.index('--batch-size')+1], '1024')
        self.assertIn('--allow-baseline-batch-size-mismatch', args)
        self.assertIn('--prepare-only', command(cfg, 'prepare'))

    def test_condition_only_paired_weight_contract(self):
        rng = np.random.default_rng(9)
        data = dict(condition=rng.normal(size=(60,4)), split=np.repeat([0,1,2],20),
                    event_weight=np.exp(rng.normal(size=60)))
        report = paired_input_contract(data, {'shapes':{'x':[4]}})
        np.testing.assert_allclose(
            [r['condition_only_auc'] for r in report['splits']], .5,
            rtol=0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'Identity or target'):
            paired_input_contract(data, {'shapes':{'source_event_key':[]}})

    def test_single_candidate_calls_share_condition_and_weighted_bce(self):
        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, condition, candidate):
                self.calls.append((condition, candidate))
                return candidate[:, 0]
        model = Recorder()
        c = torch.randn(8,3)
        a, b = torch.randn(8,15), torch.randn(8,15)
        w = torch.arange(1.,9.)
        loss, positive, negative = paired_loss(model, c, a, b, w)
        self.assertEqual(len(model.calls), 2)
        self.assertIs(model.calls[0][0], model.calls[1][0])
        self.assertIs(model.calls[0][1], a)
        self.assertIs(model.calls[1][1], b)
        expected = (w * (torch.nn.functional.softplus(-positive)+torch.nn.functional.softplus(negative))/2).mean()
        torch.testing.assert_close(loss, expected)


if __name__ == '__main__':
    unittest.main()
