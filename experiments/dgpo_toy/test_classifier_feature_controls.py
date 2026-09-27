import copy
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.dgpo_toy.classifier_feature_controls import load_controls
from experiments.dgpo_toy.closed_loop_lab import ARTIFACTS, score_metrics
from experiments.dgpo_toy.truth_pretrain import atomic_checkpoint, atomic_json


class FeatureControlTests(unittest.TestCase):
    def fixture(self, folder):
        panel = {s: {'c': torch.zeros(4, 1), 'positive': torch.ones(4, 3),
                     'negative': -torch.ones(4, 3)} for s in ('train','validation','test')}
        score = {'positive': torch.zeros(4), 'negative': torch.zeros(4)}
        (folder/'classifiers').mkdir()
        fit = {'width': 32, 'cold_start': True, 'minimum_steps': 32000, 'max_steps': 64000,
               'check_every': 100, 'patience_checks': 20, 'min_delta': 1e-4,
               'test': {'step0__plain': {**score_metrics(score), 'valid': True}}}
        atomic_json(folder/'report.json', {'plan': {'classifier_seed': 83}, 'fit': fit})
        atomic_checkpoint(folder/'step0_panels.pt', panel)
        atomic_checkpoint(folder/'classifiers/test_scores.pt', {'step0__plain': score})
        plan = {'classifier_seed': 83, 'classifier_width': 32, 'minimum_fit_steps': 32000,
                'max_fit_steps': 64000, 'saved_classifier_controls': {
                    'saved_plain': {'directory': str(folder), 'key': 'step0__plain', 'current_dataset': 'step0'}}}
        return plan, {'step0': panel}

    def test_controls_load_and_all_three_splits_are_checked(self):
        with tempfile.TemporaryDirectory(prefix='feature_controls_', dir=ARTIFACTS) as tmp:
            plan, panels = self.fixture(Path(tmp))
            scores, metadata = load_controls(plan, panels)
            self.assertEqual(set(scores), {'saved_plain'})
            self.assertEqual(metadata['saved_plain']['width'], 32)
            for split in ('train','validation','test'):
                broken = copy.deepcopy(panels)
                broken['step0'][split]['negative'][0, 0] = 9
                with self.assertRaises(ValueError):
                    load_controls(plan, broken)

    def test_seed_width_and_stopping_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix='feature_controls_', dir=ARTIFACTS) as tmp:
            plan, panels = self.fixture(Path(tmp))
            for field, value in [('classifier_seed', 84), ('classifier_width', 128),
                                  ('minimum_fit_steps', 8000), ('max_fit_steps', 32000)]:
                with self.assertRaises(ValueError):
                    load_controls({**plan, field: value}, panels)


if __name__ == '__main__':
    unittest.main()
