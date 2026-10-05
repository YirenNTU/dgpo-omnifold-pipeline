import unittest
import numpy as np
from scripts.tau_reward_cij_probe import compare_reweighting


class RewardProbeTests(unittest.TestCase):
    def test_equal_scores_recover_unweighted(self):
        rng = np.random.default_rng(42)
        q = rng.normal(size=(8, 4, 9)); t = rng.normal(size=(8, 9))
        r = compare_reweighting(t, q, np.zeros((8, 4)), np.arange(1, 9.))
        for arm in r['arms'].values():
            np.testing.assert_allclose(arm['cij'], r['arms']['unweighted']['cij'])

    def test_condition_only_score_does_not_fake_conditional_gain(self):
        q = np.zeros((2, 2, 9)); q[1] = 1
        r = compare_reweighting(np.zeros((2, 9)), q, np.array([[0., 0.], [3., 3.]]), np.ones(2))
        self.assertGreater(r['arms']['global_ratio']['condition_mass_tv'], .4)
        self.assertAlmostEqual(r['arms']['within_condition']['condition_mass_tv'], 0)
        np.testing.assert_allclose(r['arms']['within_condition']['cij'], r['arms']['unweighted']['cij'])

    def test_correct_candidate_preference_improves_nn(self):
        q = np.zeros((2, 2, 9)); q[:, 1, 8] = -1
        t = np.zeros((2, 9)); t[:, 8] = -1
        r = compare_reweighting(t, q, np.array([[0., 3.], [0., 3.]]), np.ones(2))
        self.assertLess(r['arms']['within_condition']['nn_error'], r['arms']['unweighted']['nn_error'])


if __name__ == '__main__': unittest.main()
