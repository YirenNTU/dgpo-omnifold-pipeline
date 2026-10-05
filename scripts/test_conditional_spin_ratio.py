import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from scripts.train_conditional_spin_ratio import (
    ConditionalSpinMLP, event_split, generated_log_ratio, join_pools, paired_loss,
    read_bundle, resolve_test_events, spin_features, tau_features, tau_moment_report,
    verify_paired_parquet,
)
from scripts.sample_conditional_spin_train import merge_shards
from scripts.diagnose_reweighted_cij import features as cij_features


class ConditionalSpinRatioTest(unittest.TestCase):
    def test_density_ratio_scores_generated_events_directly(self):
        positive=np.array([5.,6.]); negative=np.array([-1.,2.])
        np.testing.assert_array_equal(generated_log_ratio(positive,negative),negative)

    def test_same_condition_pair_and_candidate_information(self):
        a = np.array([[1., 0., 0.], [0., 1., 0.]])
        b = np.array([[0., 0., 1.], [1., 0., 0.]])
        kappas = np.array([[1., -0.5], [0.5, 0.5]])
        f = spin_features(a, b, kappas)
        np.testing.assert_array_equal(f[:, :3], a)
        np.testing.assert_array_equal(f[:, 3:6], b)
        self.assertEqual(f.shape, (2, 24))
        self.assertEqual(f[0, 8], 1.)  # a_k*b_n
        self.assertEqual(f[0, 17], -2.)  # a_k*b_n/(kappa_a*kappa_b)
        np.testing.assert_allclose(9 * f[:, 15:24], cij_features(a, b, kappas))
        model = ConditionalSpinMLP(4)
        c = torch.randn(2, 4)
        score = model(c, torch.from_numpy(f))
        self.assertEqual(score.shape, (2,))
        model.eval()
        loss, positive, negative = paired_loss(model, c, torch.from_numpy(f),
            torch.from_numpy(f), torch.ones(2))
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(positive, negative)

    def test_tau_features_preserve_both_directions_and_joint_products(self):
        a = np.array([[45.6, 1., 0., 0.], [45.6, 0., 2., 0.]])
        b = np.array([[45.6, 0., 0., 3.], [45.6, 4., 0., 0.]])
        f = tau_features(a, b)
        self.assertEqual(f.shape, (2,15))
        np.testing.assert_array_equal(f[0, :6], [1,0,0,0,0,1])
        self.assertEqual(f[0, 8], 1.)

    def test_tau_moment_report_uses_candidate_ratio_and_channels(self):
        truth = np.zeros((8,15), dtype=np.float32)
        generated = truth.copy()
        generated[::2,0] = 1
        arrays = dict(tau_truth=truth, tau_generated=generated,
            event_weight=np.ones(8), category=np.array([11]*4+[12]*4),
            visible_pt_sum=np.arange(8, dtype=float))
        scores = np.array([-8.,0.,-8.,0.,-8.,0.,-8.,0.])
        report = tau_moment_report(arrays, scores, np.ones(8,dtype=bool))
        self.assertGreaterEqual(len(report['groups']),3)
        self.assertLess(report['groups'][0]['l2_error_change'],0)
        self.assertLess(report['groups'][1]['l2_error_change'],0)

    def test_external_test_never_sets_normalization_or_fit_split(self):
        n = 500
        ids = np.array([f'1:{i}' for i in range(n)])
        zero = np.zeros((n, 3), dtype=np.float64)
        sample = dict(condition=np.column_stack((np.arange(n), np.ones(n),
                    np.eye(16, dtype=np.float32)[np.arange(n) % 16])).astype('float32'),
            candidate_truth=np.zeros((n, 24), dtype='float32'),
            candidate_generated=np.zeros((n, 24), dtype='float32'),
            tau_truth=np.zeros((n, 15), dtype='float32'),
            tau_generated=np.zeros((n, 15), dtype='float32'),
            truth_a=zero, truth_b=zero, sample_a=zero, sample_b=zero,
            base_log_ratio=np.zeros(n), category=np.full(n, 11),
            event_weight=np.ones(n), kappas=np.ones((n, 2)), source_ids=ids)
        test = {k:v[:50].copy() for k,v in sample.items()}
        test['source_ids'] = np.array([f'2:{i}' for i in range(50)])
        test['condition'][:, 0] = 1e8
        joined = join_pools(sample, test, 42)
        self.assertTrue(np.all(joined['split'][n:] == 2))
        self.assertGreater((joined['split'][:n] == 1).sum(), 2)
        self.assertGreater(np.abs(joined['condition'][n:, 0]).min(), 10.)
        np.testing.assert_array_equal(joined['split'][:n], event_split(ids, 42))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            join_pools(sample, sample, 42)

    def test_saved_candidate_parquet_alignment_and_paired_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; events = root / 'events'
            source.mkdir(); events.mkdir()
            n = 32
            va = np.array([12., 4., 5., 8.])
            vb = np.array([12., -4., -5., -8.])
            rows = {'source_sample_index': np.ones(n, dtype=np.int64),
                'source_event_key': np.arange(n, dtype=np.int64),
                'event_weight': np.ones(n), 'event_category':np.full(n, 11),
                'analyzing_power_a': np.ones(n), 'analyzing_power_b': np.ones(n)}
            for prefix,p4 in [('lead_a_visible',va),('lead_b_visible',vb)]:
                for component,value in zip(('E','px','py','pz'),p4):
                    rows[f'{prefix}_{component}'] = np.full(n, value)
            pq.write_table(pa.table(rows), events / 'part.parquet')
            truth = np.zeros((n,2,2), dtype='float32')
            truth[:,0,0] = 0.05
            gen = truth[:,None].copy(); gen[:,:,0,0] += .02
            np.savez(source / 'candidates.npz', source_sample_index=np.ones(n, dtype=np.int64),
                source_event_key=np.arange(n, dtype=np.int64), truth_deltas=truth,
                deltas=gen, log_ratio=np.zeros((n,1)))
            torch.save(dict(condition=torch.ones(n,4), deltas=torch.from_numpy(gen)), source / 'pool.pt')
            (source / 'manifest.json').write_text(json.dumps(dict(events=n,weight_mode='joint',
                event_source=str(events.resolve()))))
            self.assertEqual(resolve_test_events(source).resolve(), events.resolve())
            with self.assertRaisesRegex(ValueError, 'differs from saved event_source'):
                resolve_test_events(source, source)
            arrays,_=read_bundle(source,events)
            self.assertEqual(arrays['condition'].shape,(n,20))
            self.assertEqual(arrays['candidate_truth'].shape,(n,24))
            self.assertTrue(np.isfinite(arrays['candidate_generated']).all())
            self.assertGreater(np.max(np.abs(arrays['candidate_truth']-arrays['candidate_generated'])),0)
            tau_arrays,_=read_bundle(source,events,'tau')
            self.assertEqual(tau_arrays['candidate_truth'].shape,(n,15))
            np.testing.assert_array_equal(tau_arrays['candidate_truth'],tau_arrays['tau_truth'])
            self.assertGreater(np.max(np.abs(tau_arrays['candidate_truth']-tau_arrays['candidate_generated'])),0)
            (source / 'manifest.json').write_text(json.dumps(dict(events=n+1,weight_mode='joint',
                event_source=str(events.resolve()))))
            with self.assertRaisesRegex(ValueError, 'differs from manifest events'):
                read_bundle(source, events, 'tau')

    def test_missing_source_metadata_requires_full_parquet_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            source, events = Path(temp) / 'source', Path(temp) / 'events'
            source.mkdir(); events.mkdir()
            (source / 'manifest.json').write_text(json.dumps({'events':2}))
            self.assertEqual(resolve_test_events(source, events), events)
            ids = np.array(['1:10','1:20'])
            condition = np.array([[2.,3.],[4.,5.]], dtype=np.float32)
            truth = np.zeros((2,2,2), dtype=np.float32)
            rebuilt = {'condition':torch.from_numpy(condition),
                'truth':torch.from_numpy(truth),
                'source_sample_index':torch.tensor([1,1]),
                'source_event_key':torch.tensor([10,20])}
            with patch('scripts.sample_1110_cij.validation_pool', return_value=rebuilt):
                report = verify_paired_parquet(events, {'truth_deltas':truth},
                    condition, ids, ['source_sample_index','source_event_key'], {})
                self.assertTrue(report['packed_visible_match'])
                with self.assertRaisesRegex(ValueError, 'condition does not match'):
                    verify_paired_parquet(events, {'truth_deltas':truth}, condition+1,
                        ids, ['source_sample_index','source_event_key'], {})
                wrong = truth.copy(); wrong[0,0,0] = 1.
                with self.assertRaisesRegex(ValueError, 'truth tau target does not match'):
                    verify_paired_parquet(events, {'truth_deltas':wrong}, condition,
                        ids, ['source_sample_index','source_event_key'], {})

    def test_parallel_sample_merge_preserves_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            torch.save(dict(positions=torch.tensor([0,2]),draws=torch.ones(2,1,2,2)),
                       directory/'rank-000.pt')
            torch.save(dict(positions=torch.tensor([1,3]),draws=2*torch.ones(2,1,2,2)),
                       directory/'rank-001.pt')
            samples=merge_shards(directory,4,2)
            np.testing.assert_array_equal(samples[:,0,0,0],[1,2,1,2])
            torch.save(dict(positions=torch.tensor([0,3]),draws=2*torch.ones(2,1,2,2)),
                       directory/'rank-001.pt')
            with self.assertRaisesRegex(ValueError,'Missing, duplicate'):
                merge_shards(directory,4,2)


if __name__ == '__main__':
    unittest.main()
