import math
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.dgpo_toy import closed_loop_lab as lab
from experiments.dgpo_toy.classifier_capacity_comparison import context_bce, capacity_contrasts, decision
from experiments.dgpo_toy.nonperiodic_cube import Data, panel
from experiments.dgpo_toy.conditional import Config


class CapacityComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_null_bce_and_interaction(self):
        null = {'positive': torch.zeros(5), 'negative': torch.zeros(5)}
        self.assertTrue(torch.allclose(context_bce(null), torch.full((5,), math.log(2.), dtype=torch.double)))
        stronger = {'positive': torch.ones(5), 'negative': -torch.ones(5)}
        values = capacity_contrasts({'narrow_plain': null, 'narrow_both': stronger,
                                    'wide_plain': stronger, 'wide_both': stronger})
        self.assertTrue((values['width_interaction'] < 0).all())
        self.assertTrue(torch.equal(values['width_interaction'], values['narrow_fourier_minus_plain']))

    def test_unequal_or_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            context_bce({'positive': torch.zeros(5), 'negative': torch.zeros(4)})
        with self.assertRaises(ValueError):
            context_bce({'positive': torch.tensor([float('nan')]), 'negative': torch.zeros(1)})

    def test_decision_never_labels_undertraining_as_failure(self):
        c = {name: {'hi95': -.01} for name in ('width_interaction', 'narrow_fourier_minus_plain', 'narrow_plain_minus_chance')}
        self.assertEqual(decision(c, valid=False, margin=.002), 'inconclusive_fit_budget')
        self.assertEqual(decision(c, valid=True, margin=.002), 'supports_larger_fourier_advantage_at_narrow_width')
        c['width_interaction']['hi95'] = .001
        self.assertTrue(decision(c, valid=True, margin=.002).startswith('narrow_plain_still_detects'))

    def test_width_metadata_checkpoint_and_pair_matching(self):
        data = Data(Config(dimensions=3, context_dim=1))
        panels = {'step0': {split: panel(data, 32, seed) for split, seed in
                           [('train', 11), ('validation', 12), ('test', 13)]}}
        initial_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory(prefix='width_test_', dir=lab.ARTIFACTS) as tmp:
            fit, _ = lab.fit_classifiers(panels, ['plain', 'both'], Path(tmp)/'fit', lambda _: None,
                width=32, max_steps=1, min_steps=1, check_every=1)
            self.assertTrue(torch.equal(initial_rng, torch.get_rng_state()))
            self.assertEqual(fit['width'], 32)
            states = [torch.load(Path(tmp)/f'fit/step0__{mode}_best.pt', weights_only=True) for mode in ('plain', 'both')]
            self.assertTrue(all(s['width'] == 32 and s['model']['net.0.weight'].shape == (32, 36) for s in states))
            self.assertEqual(fit['parameter_count'], sum(p.numel() for p in lab.FeatureCritic(width=32).parameters()))
        # The fitting function itself does not perturb the caller's RNG.
        with torch.random.fork_rng():
            torch.set_rng_state(initial_rng)
            narrow = lab.FeatureCritic('plain', 32)
            both = lab.FeatureCritic('both', 32)
            self.assertEqual(sum(p.numel() for p in narrow.parameters()), sum(p.numel() for p in both.parameters()))


if __name__ == '__main__':
    unittest.main()
