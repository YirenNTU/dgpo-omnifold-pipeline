import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diagnose_stic_parquet import TARGETS, scan_file


class TestSticParquet(unittest.TestCase):
    def table(self):
        columns = {}
        for slot in range(2):
            columns[f'x_mask:{slot}'] = np.array([True, True, False])
            for feature in range(5):
                columns[f'x:{slot}:{feature}'] = np.zeros(3, dtype=np.float32)
        columns['source_event_index'] = np.array([10, 11, 12], dtype=np.int64)
        return columns

    def scan(self, columns):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.parquet'
            pq.write_table(pa.table(columns), path)
            return scan_file(path, list(TARGETS), batch_rows=1)

    def test_direct_large_value_preserved_and_source_recorded(self):
        columns = self.table()
        columns['x:1:4'][1] = 1040744128
        result = self.scan(columns)
        stat = result['statistics']['Part_sticChargedTag']
        self.assertEqual(stat['large'], 1)
        self.assertEqual(stat['max'], 1040744128.)
        row = result['examples'][0]
        self.assertEqual((row['file_row'], row['particle_slot']), (1, 1))
        self.assertEqual(row['source_ids']['source_event_index'], '11')

    def test_padding_does_not_flag_dataset(self):
        columns = self.table()
        columns['x:1:4'][2] = 1.e12
        result = self.scan(columns)
        self.assertEqual(result['statistics']['Part_sticChargedTag']['large'], 0)
        self.assertEqual(result['statistics']['Part_sticChargedTag']['valid'], 4)
        self.assertEqual(result['examples'], [])

    def test_nonfinite_and_subnormal_are_counted_and_json_safe(self):
        import json
        columns = self.table()
        columns['x:0:0'][0] = np.nan
        columns['x:0:1'][1] = np.nextafter(np.float32(0), np.float32(1))
        result = self.scan(columns)
        self.assertEqual(result['statistics'][TARGETS[0]]['nonfinite'], 1)
        self.assertEqual(result['statistics'][TARGETS[1]]['subnormal'], 1)
        json.dumps(result, allow_nan=False)

    def test_missing_schema_fails_instead_of_reporting_clean(self):
        columns = self.table()
        del columns['x_mask:1']
        with self.assertRaisesRegex(ValueError, 'missing particle mask'):
            self.scan(columns)


if __name__ == '__main__':
    unittest.main()
