import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
import yaml
import test_monitor_restart_plateau as trial


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Linear(1,1).requires_grad_(False)
        self.output=torch.nn.Linear(1,1)
    def forward(self,condition,sample):
        return self.output(sample).squeeze(-1)


class TestPlateau(unittest.TestCase):
    def config(self):
        return yaml.safe_load((Path(__file__).resolve().parents[1]/'config/dgpo_10pct_monitor_restart_plateau.yaml').read_text())

    def test_settings_and_invalid_caps(self):
        c=self.config();trial.settings(c)
        for changes in ({'minimum_updates':6000},{'seeds':[1,1]},{'validation_interval':0}):
            with self.assertRaises(ValueError):trial.settings({**c,**changes})

    def test_paired_metrics_and_weights(self):
        s={'truth_logits':torch.ones(20),'raw_logits':-torch.ones(20)}
        z=trial.paired_bce(s,s)
        self.assertEqual(z['cold_minus_warm'],0)
        self.assertEqual(z['normal_95_high'],0)
        self.assertAlmostEqual(trial.weight_metrics(torch.zeros(100))['ess_fraction'],1)
        a=trial.weight_metrics(torch.tensor([0.,1.,4.]));b=trial.weight_metrics(torch.tensor([100.,101.,104.]))
        self.assertEqual(a,b)

    def test_real_fit_plateau_restoration_cap_and_frozen_backbone(self):
        torch.manual_seed(10)
        data={'train_condition':torch.zeros(32,1),'train_truth':torch.ones(32,1),
              'train_raw':-torch.ones(32,1,1),'validation_condition':torch.zeros(16,1),
              'validation_truth':torch.ones(16,1),'validation_raw':-torch.ones(16,1,1)}
        with tempfile.TemporaryDirectory() as tmp:
            cfg={**self.config(),'output_dir':tmp,'steps':8,'batch_size':8,'microbatch_size':4,
                 'score_batch_size':8,'minimum_updates':2,'validation_interval':2,
                 'patience_evaluations':2,'learning_rate':1e-12}
            with mock.patch('builtins.print'):
                result=trial.run_arm(Tiny(),'cold',data,cfg)
                capped=trial.run_arm(Tiny(),'capped',data,{**cfg,'minimum_updates':7})
            self.assertTrue(result['saturated'])
            self.assertLess(result['steps_completed'],8)
            self.assertTrue(result['frozen_backbone_unchanged'])
            self.assertTrue(capped['cap_hit_without_plateau'])
            self.assertEqual(capped['steps_completed'],8)
            self.assertTrue((Path(tmp)/'cold/selected_monitor.pt').is_file())
            self.assertTrue((Path(tmp)/'cold/selected_scores.pt').is_file())


if __name__=='__main__':unittest.main()
