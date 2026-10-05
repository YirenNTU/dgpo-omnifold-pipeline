import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.rescore_film_cij import validate_stack,log_cij_reports


class StackTests(unittest.TestCase):
    def fixture(self):
        return {'global_step':0,'dgpo_omnifold_reward_stack':{'reward':{
            'increments':[{'packing_spec':{'shapes':{}}}], 'tempering':1}}}

    def test_raw_saved_stack(self):
        self.assertEqual(validate_stack(self.fixture())['reward']['tempering'],1)

    def test_transformed_or_wrong_step_rejected(self):
        for key,value in [('tempering',.75),('log_ratio_clip',5),('iteration_temperatures',[1,.8])]:
            c=self.fixture();c['dgpo_omnifold_reward_stack']['reward'][key]=value
            with self.assertRaises(ValueError):validate_stack(c)
        c=self.fixture();c['global_step']=320
        with self.assertRaises(ValueError):validate_stack(c)

    def test_logging_distinguishes_full_truth_and_matched_target(self):
        class Run:
            def __init__(self):self.summary={};self.logged=[];self.saved=[]
            def log(self,value):self.logged.append(value)
            def save(self,path,**kwargs):self.saved.append(path)
        run=Run()
        full={'C_frobenius_error_change':dict(value=.1,ci95=[0,.2])}
        matched={'C_frobenius_error_change':dict(value=-.2,ci95=[-.3,-.1])}
        report=dict(events=100,candidates=1,comparison=full,
                    reference_comparisons={'matched_target':dict(comparison=matched)},
                    diagnostics=dict(weight_health={'event_ess_fraction':.08},
                        inverse_abs_product_quantiles={'0.99':50},
                        influence_top1pct_fraction={'reweighted':[[.1]*3]*3}))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'truth_convention_check.json').write_text(json.dumps(dict(passed=False,max_error=.002,events_above_tolerance=1)))
            for mode in ('raw','calibrated'):
                for suffix in ('','_matched','_angular_moments'):
                    (root/f'{mode}{suffix}.json').write_text(json.dumps(report))
            with patch('wandb.Image',side_effect=lambda path:path):
                log_cij_reports(run,root)
        self.assertEqual(run.summary['cij/raw/full_truth/error_change/value'],.1)
        self.assertEqual(run.summary['cij/raw/matched_target/error_change/value'],-.2)
        self.assertFalse(run.summary['cij/physics_unfolded'])
        self.assertEqual(len(run.saved),11)


if __name__=='__main__':unittest.main()
