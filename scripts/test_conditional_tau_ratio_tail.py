import unittest
import numpy as np
from scripts.diagnose_conditional_tau_ratio_tail import normalized_weights, moments, analyze


class TailDiagnosticTest(unittest.TestCase):
    def test_extreme_logit_is_stable_and_zero_base_stays_zero(self):
        w, _ = normalized_weights(np.array([1., 1., 0.]), np.array([10000., 0., 20000.]))
        np.testing.assert_array_equal(w, [1., 0., 0.])

    def test_deletion_reports_target_shift(self):
        result = moments(np.array([[0.], [2.]]), np.array([[0.], [2.]]),
            np.ones(2), np.zeros(2), np.array([True, False]))
        self.assertEqual(result['error_change'], 0.)
        self.assertEqual(result['target_shift'], 1.)
        self.assertEqual(result['error_reweighted_to_original_target'], 1.)

    def test_single_event_failure_and_identity_guard(self):
        n = 40
        a = np.tile([1., 0., 0.], (n, 1))
        b = a.copy(); b[20] = [0., 1., 0.]
        arrays = dict(split=np.r_[np.zeros(20), np.full(20, 2)], source_ids=np.arange(n).astype(str),
            event_weight=np.ones(n), kappas=np.ones((n, 2)), truth_a=a, truth_b=a,
            sample_a=a, sample_b=b, tau_truth=np.zeros((n, 15)), tau_generated=np.zeros((n, 15)),
            condition=np.zeros((n, 3)), candidate_generated=np.zeros((n, 15)), category=np.full(n, 11))
        arrays['tau_generated'][20, 0] = 1.
        q = np.r_[30., np.zeros(19)]
        scores = dict(source_ids=arrays['source_ids'][20:], log_ratio=q, generated_logits=q,
            truth_logits=np.zeros(20), saved_h4_log_ratio=np.zeros(20))
        result = analyze(arrays, scores)
        self.assertGreater(result['original_health']['max_mass'], .999)
        arm = next(x for x in result['removal_arms'] if x['removed'] == 1)
        self.assertAlmostEqual(arm['health']['ess'], 19.)
        self.assertEqual(arm['moments']['cij']['error_reweighted'], 0.)
        scores['source_ids'] = scores['source_ids'][::-1]
        with self.assertRaisesRegex(ValueError, 'source IDs'):
            analyze(arrays, scores)


if __name__ == '__main__':
    unittest.main()
