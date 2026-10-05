import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.check_parquet_analyzing_power import inspect


class InventoryTests(unittest.TestCase):
    def test_streaming_and_missing_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            pq.write_table(pa.table(dict(event_category=[1]*5, lead_a_pdgId=[11]*5,
                lead_b_pdgId=[211]*5, analyzing_power_a=[-.33, -.34, None, float('nan'), 0.],
                analyzing_power_b=[1., 1., 1., 1., 2.])), path/'one.parquet')
            pq.write_table(pa.table(dict(unrelated=[1, 2])), path/'two.parquet')
            (path/'shape_metadata.json').write_text('{}')
            report = inspect(path, batch_size=2)
            self.assertEqual(report['rows'], 7)
            self.assertEqual(report['negative_product_events'], 2)
            self.assertEqual(report['missing_field_rows']['analyzing_power_a'], 2)
            self.assertEqual(report['invalid_power_rows']['analyzing_power_a'], 5)
            self.assertEqual(report['invalid_power_rows']['analyzing_power_b'], 3)
            self.assertEqual(report['null_field_rows']['analyzing_power_a'], 1)
            self.assertEqual(sum(r['count'] for r in report['combinations']), 7)
            json.dumps(report, allow_nan=False)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                inspect(tmp)

    def test_category_product_and_mass_without_pdg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.parquet'
            data = dict(event_category=[41,42,24,23,44,41,99],
                analyzing_power_a=[-.34,-.34,.41,.41,-.34,.34,1.],
                analyzing_power_b=[1.,.41,-.34,-.33,-.34,1.,1.],
                analyzing_power=[-.34,-.1394,-.1394,-.1353,.1156,0.,None])
            for leg in ('a','b'):
                for key, val in [('E',2.),('px',0.),('py',0.),('pz',0.)]:
                    data[f'truth_{leg}_visible_{key}'] = [val]*7
            data['truth_a_visible_E'][0] = 0.
            data['truth_a_visible_px'][0] = 1.
            data['truth_b_visible_E'][0] = None
            pq.write_table(pa.table(data), path)
            report = inspect(path, 2)
            checks = report['consistency_checks']
            self.assertEqual(checks['category_leg_mismatches'], 1)
            self.assertEqual(checks['product_mismatches'], 1)
            self.assertEqual(checks['unknown_category_rows'], 1)
            self.assertEqual(checks['product_uncheckable'], 1)
            mass = next(r for r in report['truth_visible_mass'] if r['event_category']==41 and r['leg']=='a')
            self.assertEqual(mass['status']['negative_mass2_rows'], 1)
            self.assertEqual(mass['mass2_GeV2_quantiles']['0'], -1.)
            self.assertEqual(mass['nonnegative_mass_GeV_quantiles']['0.5'], 2.)
            json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
