"""CPU contract tests; no cluster, W&B writes, or policy training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.tau_backbone_alignment import joint_mmd, candidate_hidden, merge_features, attach_features
from scripts.run_tau_backbone_alignment import arm_command
from scripts.run_conditional_tau_ratio import read_config


class NativePathStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.invisible_normalizer = SimpleNamespace(mean=torch.zeros(2))
        self.TruthGeneration = nn.Module()
        self.TruthGeneration.generator = nn.Linear(4, 2)
        self.seen = []

    def invisible_coordinate_normalizer(self, batch):
        return lambda x, mask: x / 2

    def predict_diffusion_vector(self, x, batch, time, mode, noise_mask):
        assert time.shape == (len(x),) and not time.any()
        assert mode == 'neutrino' and not self.training
        self.seen.append(x.clone())
        z = torch.cat((x, batch['c'].expand(-1, 2, -1)), -1)
        # Include visible tokens to check that the extractor returns only the
        # last two candidate tokens, preserving their charge/slot order.
        return self.TruthGeneration.generator(torch.cat((torch.zeros_like(z), z), 1))


class AlignmentTests(unittest.TestCase):
    def test_native_clean_path_and_hook_cleanup(self):
        model = NativePathStub().requires_grad_(False)
        before = copy.deepcopy(model.state_dict())
        candidate = torch.arange(12.).reshape(3, 2, 2)
        batch = {'c':torch.ones(3, 1, 2)}
        result = candidate_hidden(model, batch, candidate)
        self.assertEqual(result.shape, (3, 8))
        np.testing.assert_equal(result.reshape(3, 2, 4)[..., :2], candidate.numpy()/2)
        np.testing.assert_equal(candidate_hidden(model, batch, candidate), result)
        self.assertEqual(len(model.TruthGeneration.generator._forward_pre_hooks), 0)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))

    def test_joint_detects_wrong_conditional_with_matching_marginal(self):
        c = torch.tensor([[-1.], [-1.], [1.], [1.]])
        a, b = c.clone(), -c.clone()
        w, q = torch.ones(4), torch.zeros(4, requires_grad=True)
        matched = joint_mmd(c, a, a, q, w)
        wrong = joint_mmd(c, a, b, q, w)
        marginal = joint_mmd(torch.zeros_like(c), a, b, q, w)
        self.assertLess(abs(matched.item()), 1e-6)
        self.assertLess(abs(marginal.item()), 1e-6)
        self.assertGreater(wrong.item(), .1)

    def test_ratio_gradient_and_stable_offset(self):
        torch.manual_seed(1)
        c, a, b = torch.randn(9, 3), torch.randn(9, 15), torch.randn(9, 15)
        q = torch.randn(9, requires_grad=True)
        w = torch.tensor([0., 1., 2., 1., 1., 3., 1., 2., 1.])
        loss = joint_mmd(c, a, b, q, w)
        grad, = torch.autograd.grad(loss, q)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(), 1e-5)
        self.assertAlmostEqual(grad[0].item(), 0.)
        self.assertAlmostEqual(loss.item(), joint_mmd(c, a, b, q+1000, w).item(), places=5)
        self.assertEqual(joint_mmd(c, a, b, q, torch.zeros_like(w)).item(), 0.)

    def test_metric_geometry_is_not_trainable(self):
        torch.manual_seed(2)
        c = torch.randn(8, 4, requires_grad=True)
        a = torch.randn(8, 15, requires_grad=True)
        q = torch.randn(8, requires_grad=True)
        joint_mmd(c, a, -a, q, torch.ones(8)).backward()
        self.assertIsNone(c.grad)
        self.assertIsNone(a.grad)
        self.assertGreater(q.grad.norm().item(), 0.)

    def test_actual_head_backward_and_zero_coefficient_control(self):
        from scripts.train_conditional_spin_ratio import ConditionalSpinMLP, paired_loss
        torch.manual_seed(3)
        model = ConditionalSpinMLP(5, hidden=16, dropout=0., candidate_dim=31)
        control = copy.deepcopy(model)
        c, a, b = torch.randn(16, 5), torch.randn(16, 31), torch.randn(16, 31)
        w = torch.ones(16)
        bce, _, q = paired_loss(model, c, a, b, w)
        mmd = joint_mmd(c, a[:, -15:], b[:, -15:], q, w)
        gb, = torch.autograd.grad(bce, q, retain_graph=True)
        gm, = torch.autograd.grad(mmd, q, retain_graph=True)
        self.assertGreater(gb.norm().item(), 0.)
        self.assertGreater(gm.norm().item(), 0.)
        (bce + 0*mmd).backward()
        paired_loss(control, c, a, b, w)[0].backward()
        for p, ref in zip(model.parameters(), control.parameters()):
            torch.testing.assert_close(p.grad, ref.grad, rtol=1e-6, atol=1e-8)
        model.zero_grad(set_to_none=True)
        bce, _, q = paired_loss(model, c, a, b, w)
        (bce + joint_mmd(c, a[:, -15:], b[:, -15:], q, w)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        before = model.head[-1].weight.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=2e-4).step()
        self.assertFalse(torch.equal(model.head[-1].weight, before))

    def test_16_rank_partition_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.arange(99*8, dtype=np.float32).reshape(99, 8)
            for rank in range(16):
                idx = np.arange(rank, 99, 16)
                np.savez(root/f'train-{rank:02d}.npz', positions=idx, truth=x[idx], generated=-x[idx])
            a, b = merge_features(root, 'train', 99, 16)
            np.testing.assert_equal(a, x)
            np.testing.assert_equal(b, -x)
            np.savez(root/'train-00.npz', positions=[1], truth=x[[1]], generated=-x[[1]])
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                merge_features(root, 'train', 99, 16)

    def test_cache_identity_alignment_and_tau_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {'weights':'raw_state_dict_only', 'sources':{'train':'/train', 'test':'/test'}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'COMPLETE').touch()
            for name, identity in (('train','a'), ('test','b')):
                np.savez(root/f'{name}.npz', truth=np.ones((1,8)), generated=np.zeros((1,8)), source_ids=[identity])
            arrays = dict(split=np.array([0,2]), source_ids=np.array(['a','b']),
                          tau_truth=np.ones((2,15)), tau_generated=-np.ones((2,15)))
            attach_features(arrays, root, '/train', '/test')
            np.testing.assert_equal(arrays['candidate_generated'][:, -15:], arrays['tau_generated'])
            arrays['source_ids'] = arrays['source_ids'][::-1]
            with self.assertRaisesRegex(ValueError, 'identities/order'):
                attach_features(arrays, root, '/train', '/test')

    def test_yaml_two_matched_arms(self):
        cfg = read_config(Path(__file__).resolve().parents[1]/'config/conditional_tau_backbone_alignment_10pct.yaml')
        a, b = arm_command(cfg, 'bce'), arm_command(cfg, 'mmd')
        for key in ('--workers','--epochs','--batch-size','--seed','--lr','--backbone-cache','--train-source','--test-source'):
            self.assertEqual(a[a.index(key)+1], b[b.index(key)+1])
        self.assertEqual(a[a.index('--workers')+1], '16')
        self.assertEqual(a[a.index('--mmd-coefficient')+1], '0.0')
        self.assertEqual(b[b.index('--mmd-coefficient')+1], '1.0')
        self.assertGreater(cfg['classifier']['patience'], cfg['classifier']['epochs'])


if __name__ == '__main__':
    unittest.main()
