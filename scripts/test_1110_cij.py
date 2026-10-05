import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import torch
from scripts.diagnose_1110_cij import export, KEYS


class ExportTest(unittest.TestCase):
    def fixture(self):
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        b = dict(schema='h4-ratio-health-v1', model_state={'w':torch.ones(1)},
            packing_spec={'shapes':{k:[1] for k in KEYS}},
            identity_condition=x, test_condition=x[2:],
            split_indices={'fit':torch.tensor([0]), 'early_stop':torch.tensor([1]), 'test':torch.tensor([2,3])},
            truth_logits=torch.tensor([0.,1.]), gen_logits=torch.tensor([-2.,3.]),
            test_truth=torch.zeros(2,4), test_generated=torch.ones(2,4))
        order=[3,1,2,0]
        table=pa.table({**{k:x[order,j].numpy() for j,k in enumerate(KEYS)},
            'source_sample_index':[1]*4,'source_event_key':order})
        return b, table

    def test_export_preserves_raw_weights_and_aligns(self):
        b,t=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'c.npz'
            self.assertEqual(export(b,t,path),2)
            with np.load(path) as a:
                np.testing.assert_array_equal(a['source_event_key'],[2,3])
                np.testing.assert_array_equal(a['log_ratio'][:,0],[-2,3])
                self.assertEqual(a['deltas'].shape,(2,1,2,2))

    def test_overlap_rejected(self):
        b,t=self.fixture(); b['split_indices']['fit']=torch.tensor([2])
        with self.assertRaises(ValueError): export(b,t,Path('/unused.npz'))


if __name__=='__main__': unittest.main()
