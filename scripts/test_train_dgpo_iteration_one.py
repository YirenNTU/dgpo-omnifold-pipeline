"""CPU-only launch guard tests; no NERSC data or W&B connection."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import torch
import yaml
import train_dgpo_iteration_one as launcher
from train_neutrino_backend import deep_update


class TestLaunch(unittest.TestCase):
    def test_merged_config_has_required_sections(self):
        root = launcher.ROOT
        base = yaml.safe_load((root/'config/train_diffusion_nersc.yaml').read_text())
        overlay = yaml.safe_load(launcher.DEFAULT.read_text())
        merged = deep_update(base, overlay)
        for name in ('event_info','resonance','options','network','dgpo'):
            self.assertIn(name, merged)
        self.assertEqual(merged['dgpo']['adaptive_omnifold']['recalibration']['tempering'], 1.)

    def test_guard_validates_step_live_digest_and_new_destination(self):
        cfg = yaml.safe_load(launcher.DEFAULT.read_text())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            source = p/'source.ckpt'
            state = {'weight':torch.tensor([1.,2.])}
            torch.save({'global_step':320, 'state_dict':state,
                        'ema_state_dict':{'weight':torch.zeros(2)}},source)
            cfg['options']['Training']['model_checkpoint_load_path'] = str(source)
            cfg['options']['Training']['model_checkpoint_save_path'] = str(p/'new')
            cfg['nersc']['reproducibility']['source_policy_sha256'] = launcher.state_digest(state)
            cfg['reward_config']['omnifold']['backbone_checkpoint'] = str(source)
            cfg['platform']['data_parquet_dir'] = str(p)
            cfg['platform']['data_parquet_val_dir'] = str(p)
            with mock.patch('builtins.print'):
                launcher.verify(cfg)
            for key,value in [('source_dgpo_global_step',260), ('source_policy_sha256','0'*64)]:
                bad=copy.deepcopy(cfg);bad['nersc']['reproducibility'][key]=value
                with self.assertRaises(ValueError):launcher.verify(bad)
            (p/'new').mkdir()
            torch.save({},p/'new/existing.ckpt')
            with self.assertRaises(FileExistsError):launcher.verify(cfg)


if __name__ == '__main__':
    unittest.main()
