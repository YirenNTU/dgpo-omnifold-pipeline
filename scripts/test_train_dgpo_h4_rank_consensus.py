from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_dgpo_h4_rank_consensus as experiment
from train_neutrino_backend import read_overlay_yaml


def _epsilon_report(config: dict) -> dict:
    return {
        "schema": config["experiment"]["expected_epsilon_schema"],
        "policy_global_step": 1110,
        "classifier_fits": 0,
        "policy_updates": 0,
        "temperature": 2.0,
        "sweep": [
            {
                "epsilon_rms": 3.0e-7,
                "plus": {"anchor_gradient_cosine": 0.999},
                "decision": {
                    "pass_judge": True,
                    "pass_physics_jsd": True,
                    "eligible": True,
                    "judge_auc_gap_delta": -0.01,
                    "physics_mean_jsd_delta": -0.02,
                    "response_mean_abs_bin_offset_delta": -0.03,
                },
            }
        ],
    }


class RankConsensusContractTest(unittest.TestCase):
    def _resolved(self, path: Path) -> dict:
        config = read_overlay_yaml(path)
        experiment.apply_epsilon_result(
            config,
            {
                "epsilon_rms": 3.0e-7,
                "temperature": 2.0,
                "tempering": 0.5,
            },
        )
        return config

    def test_both_checked_in_configs_satisfy_contract(self) -> None:
        audit = self._resolved(experiment.AUDIT_CONFIG)
        gated = self._resolved(experiment.GATED_CONFIG)
        experiment.assert_contract(audit)
        experiment.assert_contract(gated)
        self.assertFalse(
            audit["reward_config"]["omnifold"]["candidate_consensus"][
                "apply_to_reward"
            ]
        )
        self.assertTrue(
            gated["reward_config"]["omnifold"]["candidate_consensus"][
                "apply_to_reward"
            ]
        )

    def test_arms_are_matched_except_gate_and_provenance(self) -> None:
        audit = self._resolved(experiment.AUDIT_CONFIG)
        gated = self._resolved(experiment.GATED_CONFIG)
        self.assertEqual(
            experiment._trajectory_payload(audit),
            experiment._trajectory_payload(gated),
        )

    def test_contract_rejects_scientifically_material_mutation(self) -> None:
        config = self._resolved(experiment.GATED_CONFIG)
        for path, value in (
            (("dgpo", "K"), 4),
            (("dgpo", "adaptive_omnifold", "recalibration", "crossfit_repeats"), 3),
            (("reward_config", "omnifold", "candidate_consensus", "minimum_sign_agreement"), 0.5),
            (("reward_config", "omnifold", "candidate_consensus", "temporal_history_rounds"), 0),
            (("reward_config", "omnifold", "candidate_consensus", "fixed_panel_candidates"), 16),
            (("logger", "wandb", "id"), "wrong"),
        ):
            with self.subTest(path=path):
                changed = copy.deepcopy(config)
                cursor = changed
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaises(ValueError):
                    experiment.assert_contract(changed)

    def test_check_only_reads_epsilon_and_never_launches(self) -> None:
        config = read_overlay_yaml(experiment.AUDIT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps(_epsilon_report(config)))
            with mock.patch.object(experiment.subprocess, "run") as run:
                status = experiment.main(
                    [
                        "--config",
                        str(experiment.AUDIT_CONFIG),
                        "--epsilon-report",
                        str(report),
                        "--check-only",
                        "--skip-filesystem-checks",
                    ]
                )
        self.assertEqual(status, 0)
        run.assert_not_called()

    def test_rank_audit_report_requires_all_five_fixed_panel_rounds(self) -> None:
        rows = [
            {
                "reward_round_id": float(round_id),
                "panel_events_per_rank": 128.0,
                "candidates_per_event": 8.0,
                "temporal_available": float(round_id > 1),
            }
            for round_id in range(1, 6)
        ]
        report = experiment._rank_audit_report(
            {"dgpo_omnifold_reward_stack": {"rank_audit_history": rows}}
        )
        self.assertEqual(report["rows"], rows)
        with self.assertRaises(RuntimeError):
            experiment._rank_audit_report(
                {"dgpo_omnifold_reward_stack": {"rank_audit_history": rows[:-1]}}
            )


if __name__ == "__main__":
    unittest.main()
