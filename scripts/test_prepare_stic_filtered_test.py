import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prepare_stic_filtered_test import exclusion_mask, filter_file, filtered_runtime


FEATURES = ['Part_sticShowerEnergy', 'Part_sticNumTowers', 'Part_sticChargedTag']


def sample_table():
    return pa.table({
        'x_mask:0': [True, True, True, False],
        'x:0:0': np.array([0., 0., np.nextafter(np.float32(0), np.float32(1)), 0.], dtype=np.float32),
        'x:0:1': np.array([2., 1051420096., 2., 1.e10], dtype=np.float32),
        'x:0:2': np.array([1., 1040744128., 0., 1.e10], dtype=np.float32),
        'x_invisible:0:0': np.array([1., 2., 3., 4.], dtype=np.float32),
        'source_event_index': [101, 102, 103, 104],
    }).replace_schema_metadata({b'test': b'preserve'})


class TestFilteredStic(unittest.TestCase):
    def test_excludes_entire_event_once_and_retains_padding_subnormals(self):
        mask, reasons = exclusion_mask(sample_table().to_batches()[0], FEATURES)
        self.assertEqual(mask.tolist(), [False, True, False, False])
        self.assertEqual(len(reasons), 1)
        self.assertEqual(len(reasons[1]), 2)

    def test_kept_rows_columns_schema_and_original_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'original.parquet', Path(directory)/'filtered.parquet'
            table = sample_table()
            pq.write_table(table, source)
            before = source.read_bytes()
            audit = io.StringIO()
            result = filter_file(source, target, FEATURES, audit, batch_rows=1)
            self.assertEqual(result['events_removed'], 1)
            self.assertEqual(result['rows_out'], 3)
            self.assertTrue(pq.read_table(target).equals(table.take([0,2,3]), check_metadata=True))
            self.assertEqual(source.read_bytes(), before)
            row = json.loads(audit.getvalue())
            self.assertEqual(row['source_ids']['source_event_index'], '102')
            with self.assertRaises(FileExistsError):
                filter_file(source, target, FEATURES, io.StringIO())

    def test_nonfinite_is_excluded_and_missing_mask_fails(self):
        data = sample_table().to_pydict()
        data['x:0:2'][0] = float('nan')
        mask, _ = exclusion_mask(pa.table(data).to_batches()[0], FEATURES)
        self.assertTrue(mask[0])
        del data['x_mask:0']
        with self.assertRaisesRegex(ValueError, 'Missing particle mask'):
            exclusion_mask(pa.table(data).to_batches()[0], FEATURES)

    def test_runtime_isolated_cold_start_preserves_fit_and_normalization(self):
        original = {'platform': {'data_parquet_dir':'old', 'data_parquet_val_dir':'old'},
          'options': {'Dataset': {'normalization_file':'original_norm.pt'},
                      'Training': {'EMA': {}, 'model_checkpoint_save_path':'old'}},
          'reward_config': {'omnifold': {'bundle_file':'old_reward'}},
          'dgpo': {'auto_resume_from_last': True, 'adaptive_omnifold':
                   {'recalibration': {'fit': {'learning_rate':.0002, 'batch_size':32768}}}},
          'logger': {'wandb': {'id':'old_id'}, 'local': {}},
          'nersc': {'ray': {}, 'execution': {'command':'old_command'}}}
        result = filtered_runtime(original, Path('/new_test'), Path('/pretrained.ckpt'))
        self.assertFalse(result['dgpo']['auto_resume_from_last'])
        self.assertEqual(result['dgpo']['checkpoint_load_mode'], 'weights_only')
        self.assertEqual(result['platform']['data_parquet_dir'], result['platform']['data_parquet_val_dir'])
        self.assertEqual(result['options']['Dataset'], original['options']['Dataset'])
        self.assertEqual(result['dgpo']['adaptive_omnifold']['recalibration']['fit'],
                         original['dgpo']['adaptive_omnifold']['recalibration']['fit'])
        self.assertIsNone(result['reward_config']['omnifold']['bundle_file'])
        self.assertNotIn('id', result['logger']['wandb'])
        self.assertEqual(original['platform']['data_parquet_dir'], 'old')


if __name__ == '__main__':
    unittest.main()
