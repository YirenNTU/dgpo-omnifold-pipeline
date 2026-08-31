"""Regression tests for event pairing in DGPO validation monitoring."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)

from RL.DGPO_neutrino.monitoring import (
    append_reference_prediction_arrays,
    paired_finite_truth_pred,
    scheduled_epoch,
    select_truth_pred_by_class,
    validation_schedule_tier,
)


class TestMonitoringPairing(unittest.TestCase):
    def test_reference_pass_does_not_append_truth_twice(self) -> None:
        destination = {
            "phi_truth": [np.array([1.0, 2.0])],
            "phi_pred": [np.array([1.1, 2.1])],
            "phi_ref": [],
        }
        append_reference_prediction_arrays(
            destination,
            {
                "phi_pred": np.array([0.9, 1.9]),
                "phi_truth": np.array([1.0, 2.0]),
            },
        )
        self.assertEqual(len(destination["phi_truth"]), 1)
        self.assertEqual(len(destination["phi_ref"]), 1)
        np.testing.assert_array_equal(destination["phi_ref"][0], [0.9, 1.9])

    def test_misaligned_arrays_fail_instead_of_silent_truncation(self) -> None:
        with self.assertRaisesRegex(ValueError, "preserve event pairing"):
            paired_finite_truth_pred(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                context="truth/pred monitoring",
            )

    def test_nonfinite_rows_are_removed_as_pairs(self) -> None:
        truth, pred = paired_finite_truth_pred(
            np.array([1.0, np.nan, 3.0]),
            np.array([1.1, 2.0, np.inf]),
            context="truth/pred monitoring",
        )
        np.testing.assert_array_equal(truth, [1.0])
        np.testing.assert_array_equal(pred, [1.1])

    def test_class_selection_preserves_candidate_row_pairing(self) -> None:
        truth, pred = select_truth_pred_by_class(
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.array([11.0, 21.0, 31.0, 41.0]),
            np.array([0, 1, 0, 1]),
            class_id=1,
        )
        np.testing.assert_array_equal(truth, [20.0, 40.0])
        np.testing.assert_array_equal(pred, [21.0, 41.0])

    def test_class_selection_rejects_misaligned_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "lost alignment"):
            select_truth_pred_by_class(
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1]),
                np.array([0]),
                class_id=0,
            )


class TestMonitoringCadence(unittest.TestCase):
    def test_full_validation_replaces_coincident_cheap_validation(self) -> None:
        self.assertIsNone(
            validation_schedule_tier(
                3,
                cheap_every_n_epochs=5,
                full_every_n_epochs=20,
            )
        )
        self.assertEqual(
            validation_schedule_tier(
                4,
                cheap_every_n_epochs=5,
                full_every_n_epochs=20,
            ),
            "cheap",
        )
        self.assertEqual(
            validation_schedule_tier(
                19,
                cheap_every_n_epochs=5,
                full_every_n_epochs=20,
            ),
            "full",
        )

    def test_train_monitor_can_include_initial_epoch(self) -> None:
        self.assertTrue(scheduled_epoch(0, 20, include_epoch_zero=True))
        self.assertFalse(scheduled_epoch(1, 20, include_epoch_zero=True))
        self.assertTrue(scheduled_epoch(19, 20, include_epoch_zero=True))


if __name__ == "__main__":
    unittest.main()
