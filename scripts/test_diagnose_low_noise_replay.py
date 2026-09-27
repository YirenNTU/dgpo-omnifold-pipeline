from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_low_noise_replay as experiment


def settings() -> dict:
    return {
        "reward_source_dir": "/reward",
        "time_source_dir": "/time",
        "output_dir": "/output",
        "expected_reward_schema": "reward-schema",
        "expected_time_schema": "time-schema",
        "expected_direction_schema": "direction-schema",
        "expected_source_wandb_run": "entity/project/c4a91e07",
        "expected_time_wandb_id": "h4time01",
        "expected_policy_step": 1110,
        "workers": 16,
        "cpus_per_worker": 2,
        "K": 8,
        "score_batch_size": 32768,
        "rollout_seeds": list(range(101, 109)),
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 201,
        "confidence_level": 0.90,
        "minimum_low_over_full_gain": 1.25,
        "minimum_consistency_fraction": 0.75,
        "wandb": {"enabled": False},
    }


def gradient_manifest() -> dict:
    return {
        "realizations": 4,
        "events_per_worker": 512,
        "workers": 16,
        "probe_events": 2048,
        "event_selection_seed": 301,
        "probe_selection_seed": 302,
    }


def geometry_manifest() -> dict:
    return {
        "curvature_events": 2048,
        "curvature_selection_seed": 303,
        "judge_events": 2048,
        "judge_selection_seed": 304,
    }


def replication_rows(
    *, full: float, low: float, rest: float, high: float
) -> list[dict]:
    deltas = {"full": full, "low": low, "rest": rest, "high": high}
    rows = []
    for index, seed in enumerate(settings()["rollout_seeds"]):
        panels = {}
        for panel in experiment.PANELS:
            panels[panel] = {
                "zero": {"auc_gap": 0.25},
                "arms": {
                    arm: {
                        "plus_gap_delta": value,
                        "plus_beats_minus": arm == "low",
                    }
                    for arm, value in deltas.items()
                },
            }
        rows.append({"seed_index": index, "seed": seed, "panels": panels})
    return rows


class TestLowNoiseReplay(unittest.TestCase):
    def test_validates_exact_confirmation_contract(self):
        cfg = experiment._validated_settings(settings())
        self.assertEqual(cfg["workers"], 16)
        self.assertEqual(cfg["K"], 8)
        self.assertEqual(len(cfg["rollout_seeds"]), 8)
        self.assertEqual(cfg["confidence_level"], 0.90)

    def test_rejects_changed_scale_and_invalid_replication_settings(self):
        for key, value in (
            ("workers", 8),
            ("K", 4),
            ("bootstrap_replicates", 999),
            ("confidence_level", 1.0),
            ("minimum_low_over_full_gain", 1.0),
            ("minimum_consistency_fraction", 0.49),
        ):
            bad = copy.deepcopy(settings())
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment._validated_settings(bad)
        for seeds in (
            list(range(7)),
            [1, 2, 3, 4, 5, 6, 7, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ):
            bad = copy.deepcopy(settings())
            bad["rollout_seeds"] = seeds
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                experiment._validated_settings(bad)

    def test_confirmation_panels_are_exact_and_disjoint_from_direction(self):
        panels = experiment.select_confirmation_indices(
            torch.arange(42000),
            gradient_manifest(),
            geometry_manifest(),
            direction_events=8192,
        )
        self.assertEqual(len(panels["direction"]), 8192)
        self.assertEqual(len(panels["h4grad01_probe"]), 2048)
        self.assertEqual(len(panels["h4natg01_judge"]), 2048)
        self.assertEqual(len(panels["union"]), 4096)
        self.assertTrue(
            set(panels["direction"].tolist()).isdisjoint(
                panels["union"].tolist()
            )
        )
        self.assertTrue(
            set(panels["h4grad01_probe"].tolist()).isdisjoint(
                panels["h4natg01_judge"].tolist()
            )
        )

    def test_saved_directions_require_layout_and_matched_rms_provenance(self):
        names = {"full", "low", "rest", "mid_low", "mid_high", "high"}
        layout = [{
            "name": "weight",
            "block": "model",
            "shape": [4],
            "start": 0,
            "stop": 4,
        }]
        payload = {
            "schema": "direction-schema",
            "policy_global_step": 1110,
            "parameter_layout": layout,
            "directions": {
                name: torch.arange(1, 5, dtype=torch.float16) for name in names
            },
            "matched_parameter_rms": {name: 1.0e-6 for name in names},
        }
        matched = {
            name: {"parameter_rms": 1.0e-6, "passed": True} for name in names
        }
        result = experiment.validate_direction_payload(
            payload,
            expected_schema="direction-schema",
            expected_policy_step=1110,
            expected_layout=layout,
            matched_distances=matched,
        )
        self.assertEqual(set(result["directions"]), names)
        bad = copy.deepcopy(payload)
        bad["matched_parameter_rms"]["low"] = 2.0e-6
        with self.assertRaises(ValueError):
            experiment.validate_direction_payload(
                bad,
                expected_schema="direction-schema",
                expected_policy_step=1110,
                expected_layout=layout,
                matched_distances=matched,
            )

    def test_bootstrap_is_deterministic_and_exact_for_constant_values(self):
        first = experiment.percentile_mean_interval(
            [-0.5] * 8, replicates=1000, confidence=0.90, seed=401
        )
        second = experiment.percentile_mean_interval(
            [-0.5] * 8, replicates=1000, confidence=0.90, seed=401
        )
        self.assertEqual(first, second)
        self.assertEqual(first["lower"], -0.5)
        self.assertEqual(first["upper"], -0.5)

    def test_strong_replication_requires_ordering_consistency_and_intervals(self):
        cfg = experiment._validated_settings(settings())
        result = experiment.summarize_replications(
            replication_rows(
                full=-0.002, low=-0.003, rest=-0.001, high=0.001
            ),
            panel="union",
            cfg=cfg,
        )
        self.assertEqual(
            result["decision"],
            "noise_dependent_credit_ordering_replicates",
        )
        self.assertAlmostEqual(result["low_over_full_gain"], 1.5)
        self.assertEqual(result["low_beats_full_fraction"], 1.0)
        self.assertEqual(result["high_harmful_fraction"], 1.0)
        self.assertLess(result["low_minus_full_interval"]["upper"], 0.0)
        self.assertGreater(result["high_delta_interval"]["lower"], 0.0)

    def test_partial_and_failed_replication_are_distinct(self):
        cfg = experiment._validated_settings(settings())
        partial = experiment.summarize_replications(
            replication_rows(
                full=-0.002, low=-0.0022, rest=-0.001, high=0.001
            ),
            panel="union",
            cfg=cfg,
        )
        self.assertEqual(partial["decision"], "pattern_persists_but_uncertain")
        failed = experiment.summarize_replications(
            replication_rows(
                full=-0.002, low=-0.001, rest=-0.001, high=-0.001
            ),
            panel="union",
            cfg=cfg,
        )
        self.assertEqual(failed["decision"], "h4time01_ordering_not_replicated")


if __name__ == "__main__":
    unittest.main()
