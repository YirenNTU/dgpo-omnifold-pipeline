#!/usr/bin/env python3

import unittest

import compare_nested_residual_series as comparison


def _report(name, rank, *, auc, delta, reliable=True):
    phase = "old_base" if rank == 0 else f"rank{rank}_residual"
    return {
        "series_arm": {"name": name, "conditional_residual_rank": rank},
        "source_wandb_run": "entity/project/c4a91e07",
        "policy_global_step": 1110,
        "judge_classifier_overrides": {"topology_max_harmonic": 4},
        "classification": {
            "fully_trained": {
                "final_audit_uncalibrated": {
                    "auc": auc,
                    "balanced_accuracy": auc - 0.01,
                }
            }
        },
        "classifier_members": [
            {
                "stage_diagnostics": {
                    phase: {
                        "balanced_accuracy": auc + 0.02,
                        "validation_balanced_accuracy": auc - 0.01,
                    }
                }
            },
            {
                "stage_diagnostics": {
                    phase: {
                        "balanced_accuracy": auc + 0.01,
                        "validation_balanced_accuracy": auc - 0.02,
                    }
                }
            },
        ],
        "decision": {
            "stages": {
                "fully_trained": {
                    "3e-05": {
                        "plus_minus_zero": {
                            "plus_minus_zero": {
                                "mean": delta,
                                "ci90_low": delta - 0.001,
                                "ci90_high": delta + 0.001,
                            },
                            "plus_beats_zero_fraction": 1.0,
                            "plus_beats_minus_fraction": 1.0,
                        },
                        "reliable": reliable,
                    }
                }
            }
        },
        "signal_alignment_with_independent_judge": {
            "fully_trained": {"advantage_cosine": 0.5}
        },
        "gradient_alignment_with_independent_judge": {"fully_trained": 0.4},
    }


class NestedResidualSeriesComparisonTest(unittest.TestCase):
    def test_selects_lowest_rank_that_improves_both_endpoints(self):
        result = comparison.compare(
            {
                "old": _report("old", 0, auc=0.55, delta=-0.001),
                "rank2": _report("rank2", 2, auc=0.60, delta=-0.004),
                "rank4": _report("rank4", 4, auc=0.63, delta=-0.005),
            },
            gate_radius=3.0e-5,
        )
        self.assertEqual(result["selected_arm"], "rank2")
        self.assertEqual(result["finding"], "nested_residual_supported")

    def test_rejects_discrimination_without_reliable_actionability(self):
        result = comparison.compare(
            {
                "old": _report("old", 0, auc=0.55, delta=-0.001),
                "rank2": _report(
                    "rank2", 2, auc=0.60, delta=-0.004, reliable=False
                ),
                "rank4": _report(
                    "rank4", 4, auc=0.63, delta=-0.005, reliable=False
                ),
            },
            gate_radius=3.0e-5,
        )
        self.assertIsNone(result["selected_arm"])
        self.assertEqual(result["finding"], "no_nested_residual_arm_passed")


if __name__ == "__main__":
    unittest.main()
