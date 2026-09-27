import copy
import unittest
from unittest.mock import patch

import torch

from experiments.dgpo_toy import conditional as native
from experiments.dgpo_toy.cube_swap import swaps, modes
from experiments.dgpo_toy.nonperiodic_cube import Data
from experiments.dgpo_toy.width_signal_attribution import replay_panels, content_contrasts, branch_content_contrasts


class WidthAttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        data = Data(native.Config(dimensions=3, context_dim=1))
        with patch.object(native, 'ddim', side_effect=lambda model, c, z, steps: z):
            panels, _, pool = swaps(None, data, 4, 128, 512, 77, return_pool=True)
        return data, panels, pool

    def test_replay_is_bitwise_equal_and_rng_independent(self):
        data, panels, pool = self.fixture()
        before = torch.get_rng_state().clone()
        replay = replay_panels(pool, data.centers, 128, 77)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for label in panels:
            for key in panels[label]:
                self.assertTrue(torch.equal(panels[label][key], replay[label][key]), (label, key))
        self.assertTrue(torch.equal(modes(replay['A']['negative']), modes(replay['C']['negative'])))
        self.assertTrue(torch.equal(modes(replay['B']['negative']), modes(replay['D']['negative'])))

    def test_wrong_modes_and_counts_are_rejected(self):
        data, _, pool = self.fixture()
        broken = copy.deepcopy(pool)
        broken['ids'][0, 0] = (broken['ids'][0, 0] + 1) % 8
        with self.assertRaises(ValueError):
            replay_panels(broken, data.centers, 128, 77)
        broken = copy.deepcopy(pool)
        broken['counts'][0, 0] = 1
        with self.assertRaises(ValueError):
            replay_panels(broken, data.centers, 128, 77)

    def test_interaction_cancels_judge_baseline_bce(self):
        values = {name: {k: torch.full((4,), offset) for k in ('A','B','C','D')}
                  for name, offset in [('narrow_plain', .8), ('narrow_both', .7),
                                       ('wide_plain', .9), ('wide_both', .75)]}
        values['narrow_both']['B'] -= .1
        contrasts = content_contrasts(values)
        self.assertTrue(torch.allclose(contrasts['width_interaction/B'], torch.full((4,), -.1)))
        self.assertTrue(torch.equal(contrasts['width_interaction/C'], torch.zeros(4)))

    def test_branch_contrasts_keep_both_raw_inputs_and_compare_score_changes(self):
        values = {name: {k: torch.full((4,), offset) for k in ('A','B','C','D')}
                  for name, offset in [('narrow_plain', .8), ('narrow_both', .7),
                                       ('c_only', .9), ('y_only', .75)]}
        values['c_only']['B'] -= .1
        values['y_only']['C'] -= .05
        contrasts = branch_content_contrasts(values)
        self.assertEqual(len(contrasts), 15)
        self.assertTrue(torch.allclose(contrasts['B/c_only_minus_y_only'], torch.full((4,), -.1)))
        self.assertTrue(torch.allclose(contrasts['C/c_only_minus_y_only'], torch.full((4,), .05)))
        self.assertTrue(torch.equal(contrasts['D/c_only_minus_y_only'], torch.zeros(4)))


if __name__ == '__main__':
    unittest.main()
