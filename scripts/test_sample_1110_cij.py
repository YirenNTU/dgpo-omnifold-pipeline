import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from scripts.sample_1110_cij import validation_pool, merge_parts, weight_summary


class SamplingTest(unittest.TestCase):
    def test_weight_logging(self):
        stats=weight_summary(np.zeros((10,1)))
        self.assertAlmostEqual(stats['ess_fraction'],1.)
        self.assertAlmostEqual(stats['max_weight_mass'],.1)
        self.assertLess(weight_summary([0.,100.])['ess_fraction'],.51)

    def test_full_validation_and_overlap(self):
        from evenet.dataset.preprocess import flatten_dict
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs
        data=dict(x=np.arange(12,dtype=np.float32).reshape(3,2,2),
                  x_mask=np.ones((3,2),bool),conditions=np.ones((3,1),np.float32),
                  conditions_mask=np.ones((3,1),bool),x_invisible=np.zeros((3,2,2),np.float32),
                  x_invisible_mask=np.array([[1,1],[1,1],[1,0]],bool),
                  source_sample_index=np.ones(3,np.int64),source_event_key=np.zeros(3,np.int64),
                  source_file_index=np.arange(3,dtype=np.int64),source_event_index=np.zeros(3,np.int64))
        packed,spec=pack_event_inputs({k:torch.as_tensor(v) for k,v in data.items()})
        bundle=dict(packing_spec=spec.to_dict(),identity_condition=packed,
                    split_indices={'fit':torch.tensor([0]),'early_stop':torch.tensor([2]),'test':torch.tensor([1])})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);table,meta=flatten_dict(data)
            pq.write_table(table,path/'part.parquet')
            (path/'shape_metadata.json').write_text(json.dumps(meta))
            pool=validation_pool(path,bundle)
        self.assertEqual(len(pool['truth']),2)
        self.assertEqual(pool['invalid_target_rows'],1)
        self.assertEqual(pool['overlap']['fit'].tolist(),[True,False])

    def test_merge_reorders_and_rejects_duplicates(self):
        p=dict(positions=torch.tensor([1,0]),draws=torch.zeros(2,1,2,2),logits=torch.tensor([[2.],[3.]]))
        _,s=merge_parts([p],2,1)
        np.testing.assert_array_equal(s[:,0],[3,2])
        with self.assertRaises(ValueError):merge_parts([p,p],2,1)


if __name__=='__main__': unittest.main()
