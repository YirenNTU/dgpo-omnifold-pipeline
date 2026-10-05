"""CPU-only tests of scan resampling and uncertainty accounting (no ROOT)."""
import unittest
import numpy as np
from scripts.diagnose_tau_unfolded_bias_scan import evaluate_case
from scripts.tau_bias_sampling import target_probabilities
from scripts.tau_unfold_core import moment


class BiasScanTests(unittest.TestCase):
    def setUp(self):
        self.z = np.array([-1., -.5, .5, 1.])
        self.reco = np.arange(4)
        self.p = np.ones(4) / 4

    def evaluate(self, offsets, probability=None, seed=42):
        def estimate(counts, replica):
            value, sigma = moment(counts, np.diag(counts), self.z)
            return value + offsets[replica], sigma
        return evaluate_case(estimate, self.reco, self.z, self.z,
                             self.p if probability is None else probability,
                             2000, 100, len(offsets)-1, seed)

    def test_identity_and_truth_injection(self):
        for target in [-.5, 0., .5]:
            p, _ = target_probabilities(self.z, np.ones(4), target)
            row, arrays = self.evaluate([0., 0., 0.], p)
            self.assertAlmostEqual(row['truth_exact'], target, places=8)
            self.assertAlmostEqual(row['expected_bias'], 0, places=12)
            self.assertEqual(row['response_mc_sigma'], 0.)
            np.testing.assert_allclose(arrays['fixed_estimate'], arrays['sampled_truth'])
            np.testing.assert_array_equal(arrays['fixed_estimate'], arrays['joint_estimate'])
            self.assertAlmostEqual(row['expected_combined_sigma'], row['expected_stat_sigma'])

    def test_pairing_and_combined_variance(self):
        offsets = [0., -.03, .01, .02]
        row, a = self.evaluate(offsets)
        other, b = self.evaluate(offsets)
        for key in a: np.testing.assert_array_equal(a[key], b[key])
        self.assertAlmostEqual(row['response_mc_sigma'], np.std(offsets[1:], ddof=1))
        np.testing.assert_allclose(a['joint_estimate']-a['fixed_estimate'],
                                   np.array(offsets)[a['response_replica']])
        np.testing.assert_allclose(a['joint_combined_sigma']**2,
                                   a['fixed_stat_sigma']**2+row['response_mc_sigma']**2)

    def test_bias_is_not_added_to_uncertainty(self):
        good, _ = self.evaluate([0., 0., 0.])
        biased, _ = self.evaluate([1., 1., 1.])
        self.assertAlmostEqual(biased['expected_bias'], 1.)
        self.assertEqual(good['expected_combined_sigma'], biased['expected_combined_sigma'])
        self.assertEqual(biased['stat_coverage68'], 0.)
        self.assertEqual(good['stat_centered_coverage68'], biased['stat_centered_coverage68'])

    def test_invalid_replica_not_silently_dropped(self):
        with self.assertRaises(ValueError): self.evaluate([0., float('nan'), 0.])

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError): self.evaluate([0., 0., 0.], np.ones(4))


if __name__ == '__main__': unittest.main()
