"""Run the actual native generation method with tiny CPU sampler fixtures."""
import ast
from contextlib import contextmanager
import copy
import logging
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'evenet_dgpo'))
SOURCE = REPO / 'evenet_dgpo/RL/DGPO_neutrino/conditional_tau_cycle.py'


def actual_generate():
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ConditionalTauCycle')
    method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'generate'))
    context = dict(torch=torch, np=np, log=logging.getLogger(__name__), barrier=lambda world: None,
                   panel_batch=lambda panel, rows, spec, device: {'rows': rows},
                   cij_terms=lambda panel, rows, delta: np.tile(np.arange(9), (len(rows), 1)))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), context)
    return context['generate']


class GenerationRetentionTests(unittest.TestCase):
    def evaluate(self, output, *, retain_features=True, rank=0, world=1):
        calls = dict(feature=0, reward=0, rng=[])
        def features(batch, delta):
            calls['feature'] += 1
            if not retain_features:
                raise AssertionError('No standalone feature pass needed for read-only rewards')
            return None, np.zeros((len(batch['rows']), 3))
        def compute(delta, batch):
            calls['reward'] += 1
            return delta.sum(dim=(-1, -2))
        def sample(actor, batch, sampler, *, K, num_ddim_steps, device, parallel_chains):
            self.assertEqual(num_ddim_steps, 20)
            self.assertEqual(parallel_chains, 1)
            rows = torch.tensor(batch['rows']).view(1, -1, 1, 1)
            return torch.ones(K, len(batch['rows']), 2, 2) * rows
        @contextmanager
        def isolated_rng(device, seed):
            calls['rng'].append(seed)
            yield
        sampling = ModuleType('RL.DGPO_neutrino.sampling'); sampling.generate_neutrino_candidates = sample
        noise = ModuleType('scripts.tau_fresh_negatives'); noise.isolated_rng = isolated_rng
        actor = torch.nn.Linear(1, 1); actor.train()
        cycle = SimpleNamespace(actor=actor, reward=SimpleNamespace(spec=None, features=features, compute=compute),
                                rank=rank, world=world, cfg={'generation_batch_size': 2}, sampler=None, device=torch.device('cpu'))
        with patch.dict(sys.modules, {sampling.__name__: sampling, noise.__name__: noise}):
            result = actual_generate()(cycle, {'source_ids': np.arange(5)}, output, 8, 77, True,
                                       retain_features=retain_features)
        self.assertTrue(actor.training)
        self.assertEqual(calls['rng'], [77+rank])
        return result, calls

    def test_feature_free_and_original_paths_have_identical_all_k_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            retained, calls = self.evaluate(Path(directory) / 'retained')
            minimal, minimal_calls = self.evaluate(Path(directory) / 'minimal', retain_features=False)
            self.assertEqual(calls['feature'], 3)
            self.assertEqual(minimal_calls['feature'], 0)
            self.assertEqual(calls['reward'], minimal_calls['reward'])
            self.assertIn('features', retained)
            self.assertNotIn('features', minimal)
            for key in ('deltas', 'rewards', 'cij'):
                np.testing.assert_array_equal(retained[key], minimal[key])
            with np.load(Path(directory) / 'minimal/rank-00.npz') as saved:
                self.assertNotIn('features', saved.files)
                self.assertEqual(saved['rewards'].shape, (5, 8))

    def test_feature_free_numerical_bundle_merges_rank_zero_and_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            # Tiny completed rank1 shard replaces only the synchronization in
            # this CPU fixture; the actual production method merges the files.
            rows = np.array([1, 3]); delta = np.ones((2, 8, 2, 2)) * rows[:, None, None, None]
            np.savez(output/'rank-01.npz', positions=rows, deltas=delta,
                     rewards=delta.sum(axis=(-1, -2)), cij=np.tile(np.arange(9), (2, 1)))
            result, _ = self.evaluate(output, retain_features=False, rank=0, world=2)
            np.testing.assert_array_equal(result['deltas'][:, 0, 0, 0], np.arange(5))
            # Rank1 requires neither unused features nor rank0 physics merges.
            nonzero, _ = self.evaluate(output, retain_features=False, rank=1, world=2)
            self.assertEqual(nonzero, {})

    def test_no_feature_no_physics_is_rejected_instead_of_empty_bundle(self):
        with self.assertRaisesRegex(ValueError, 'physics/reward'):
            actual_generate()(None, {}, Path('/not-used'), 8, 42, False, retain_features=False)


if __name__ == '__main__':
    unittest.main()
