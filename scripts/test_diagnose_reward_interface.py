from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))

# ``dgpo_utils`` reaches Lightning/torchvision only through the timing helper.
# Keep these reward-interface unit tests independent of that optional stack.
_debug_tool = types.ModuleType("evenet.utilities.debug_tool")


def _noop_time_decorator(name=None):
    del name

    def _wrapper(function):
        return function

    return _wrapper


_debug_tool.time_decorator = _noop_time_decorator
sys.modules.setdefault("evenet.utilities.debug_tool", _debug_tool)

import diagnose_reward_interface as experiment
from RL.DGPO_neutrino import reward_interface


class TestRewardInterfaceMath(unittest.TestCase):
    def test_centered_rank_is_tie_safe_and_event_centered(self):
        values = torch.tensor([[1.0, 3.0], [2.0, 1.0], [2.0, 2.0]])
        ranks = reward_interface.centered_candidate_rank(values)
        torch.testing.assert_close(
            ranks[:, 0], torch.tensor([-1.0, 0.5, 0.5])
        )
        torch.testing.assert_close(ranks.sum(0), torch.zeros(2))
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(
            reward_interface.centered_candidate_rank(values[permutation]),
            ranks[permutation],
        )

    def test_reward_arms_preserve_order_but_change_event_scale(self):
        logits = torch.tensor([
            [-2.0, -0.01],
            [0.0, 0.00],
            [2.0, 0.01],
        ])
        arms = reward_interface.reward_advantage_arms(
            logits, temperature=2.0, raw_tempering=0.75, zscore_epsilon=1e-12
        )
        self.assertEqual(set(arms), set(reward_interface.REWARD_ARMS))
        for value in arms.values():
            torch.testing.assert_close(value.sum(0), torch.zeros(2), atol=1e-6, rtol=0)
            self.assertTrue(torch.equal(value.argmax(0), logits.argmax(0)))
        # Raw LOO retains the 200x event-scale contrast. Z-score and rank remove it.
        raw_ratio = float(arms["raw_loo"][:, 0].norm() / arms["raw_loo"][:, 1].norm())
        self.assertGreater(raw_ratio, 100.0)
        self.assertAlmostEqual(
            float(arms["event_zscore"][:, 0].norm() / arms["event_zscore"][:, 1].norm()),
            1.0,
            places=4,
        )
        torch.testing.assert_close(arms["centered_rank"][:, 0], arms["centered_rank"][:, 1])

    def test_temperature_uses_held_out_bce_objective(self):
        truth = torch.tensor([8.0, 8.0, -8.0])
        generated = torch.tensor([-8.0, -8.0, 8.0])
        result = reward_interface.fit_scalar_temperature(truth, generated)
        self.assertGreaterEqual(result["temperature"], 0.05)
        self.assertLessEqual(result["temperature"], 20.0)
        self.assertLessEqual(result["bce_after"], result["bce_before"] + 1e-12)

    def test_member_metrics_detect_opposite_ordering(self):
        first = torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        metrics = reward_interface.member_ordering_metrics(torch.stack([first, -first]))
        self.assertAlmostEqual(metrics["pairwise_rank_cosine_mean"], -1.0)
        # Exact zeros are ties rather than sign disagreements; four of six
        # candidate/event positions still reverse sign.
        self.assertAlmostEqual(metrics["candidate_sign_disagreement_fraction"], 2.0 / 3.0)

    def test_paired_metrics_detect_temporal_rank_reversal(self):
        current = torch.tensor([[-1.0, 2.0], [0.0, 0.0], [1.0, -2.0]])
        metrics = reward_interface.paired_ordering_metrics(current, -current)
        self.assertAlmostEqual(metrics["rank_cosine"], -1.0)
        self.assertAlmostEqual(metrics["advantage_cosine"], -1.0)
        self.assertAlmostEqual(metrics["winner_agreement"], 0.0)
        self.assertAlmostEqual(
            metrics["candidate_sign_agreement_fraction"], 0.0
        )

    def test_consensus_gate_preserves_unanimous_raw_loo_direction(self):
        current = torch.tensor([[-2.0, 1.0], [0.0, 0.0], [2.0, -1.0]])
        members = torch.stack((current, 1.1 * current, 0.9 * current, current))
        gated, metrics = reward_interface.consensus_gated_advantage(
            current,
            members,
            minimum_sign_agreement=0.75,
            uncertainty_scale=0.0,
        )
        expected = reward_interface.compute_per_event_advantage(
            current, estimator="leave_one_out_unscaled"
        )[0]
        torch.testing.assert_close(gated, expected)
        self.assertEqual(metrics["hard_gate_pass_fraction"], 1.0)
        self.assertEqual(metrics["zeroed_fraction"], 0.0)
        self.assertAlmostEqual(metrics["rms_ratio"], 1.0)

    def test_consensus_gate_zeros_split_candidate_directions(self):
        current = torch.tensor([[-1.0], [0.0], [1.0]])
        members = torch.stack((current, current, -current, -current))
        gated, metrics = reward_interface.consensus_gated_advantage(
            current,
            members,
            minimum_sign_agreement=0.75,
            uncertainty_scale=1.0,
        )
        torch.testing.assert_close(gated, torch.zeros_like(gated))
        self.assertEqual(metrics["hard_gate_pass_fraction"], 0.0)
        self.assertEqual(metrics["zeroed_fraction"], 1.0)

    def test_consensus_uncertainty_is_invariant_to_member_logit_scale(self):
        current = torch.tensor([[-2.0, 1.0], [0.5, 0.0], [1.5, -1.0]])
        members = torch.stack(
            (0.05 * current, 0.2 * current, 5.0 * current, 20.0 * current)
        )
        gated, metrics = reward_interface.consensus_gated_advantage(
            current,
            members,
            minimum_sign_agreement=0.75,
            uncertainty_scale=1.0,
        )
        expected = reward_interface.compute_per_event_advantage(
            current, estimator="leave_one_out_unscaled"
        )[0]
        torch.testing.assert_close(gated, expected, atol=1.0e-6, rtol=1.0e-6)
        self.assertEqual(metrics["hard_gate_pass_fraction"], 1.0)
        self.assertAlmostEqual(metrics["uncertainty_weight_mean"], 1.0, places=6)

    def test_decision_table_separates_calibration_from_event_scaling(self):
        def row(raw, calibrated, zscore, rank):
            output = {}
            for arm, passing in zip(reward_interface.REWARD_ARMS, (raw, calibrated, zscore, rank)):
                output[arm] = {
                    "minus_judge_auc_gap": 0.2,
                    "zero_judge_auc_gap": 0.1,
                    "plus_judge_auc_gap": 0.05 if passing else 0.15,
                    "minus_response_mean_abs_bin_offset": 2.0,
                    "zero_response_mean_abs_bin_offset": 1.0,
                    "plus_response_mean_abs_bin_offset": 0.8 if passing else 1.2,
                }
            return output

        self.assertEqual(
            reward_interface.reward_interface_diagnosis(row(False, True, True, True))["finding"],
            "scalar_scale_requires_finite_step_relinearization",
        )
        self.assertEqual(
            reward_interface.reward_interface_diagnosis(row(False, False, True, True))["finding"],
            "within_event_scale_or_tail_heterogeneity",
        )


class TestRewardInterfaceProtocol(unittest.TestCase):
    def test_balanced_accuracy_lcb_and_stage_labels(self):
        lcb, standard_error = experiment._balanced_accuracy_lcb(
            0.70, events_per_class=20_000, confidence_z=1.96
        )
        self.assertAlmostEqual(standard_error, math.sqrt(0.125 / 20_000))
        self.assertLess(lcb, 0.70)
        self.assertGreater(lcb, 0.69)
        self.assertEqual(experiment._trajectory_stage_label(0.55), "ba_lcb_55")
        self.assertEqual(experiment._trajectory_stage_label(0.70), "ba_lcb_70")

    def test_paired_signal_metrics_detect_judge_direction(self):
        judge = torch.tensor([
            [-2.0, 0.0, 3.0],
            [0.0, 2.0, 1.0],
            [2.0, 1.0, -1.0],
        ])
        aligned = experiment._paired_signal_metrics(judge * 3.0, judge)
        reversed_signal = experiment._paired_signal_metrics(-judge, judge)
        self.assertAlmostEqual(aligned["advantage_cosine"], 1.0)
        self.assertAlmostEqual(aligned["rank_cosine"], 1.0)
        self.assertAlmostEqual(aligned["winner_agreement"], 1.0)
        self.assertAlmostEqual(reversed_signal["advantage_cosine"], -1.0)

    def test_trajectory_decision_requires_local_signed_judge_improvement(self):
        stages = ["ba_lcb_55", "fully_trained"]
        signal = {
            "ba_lcb_55": {"advantage_cosine": 0.4},
            "fully_trained": {"advantage_cosine": -0.2},
        }
        gradients = {"ba_lcb_55": 0.3, "fully_trained": -0.1}
        signed = {
            "ba_lcb_55": {
                "minus": {"judge_auc_gap": 0.12},
                "zero": {"judge_auc_gap": 0.10},
                "plus": {"judge_auc_gap": 0.08},
            },
            "fully_trained": {
                "minus": {"judge_auc_gap": 0.08},
                "zero": {"judge_auc_gap": 0.10},
                "plus": {"judge_auc_gap": 0.12},
            },
        }
        result = experiment._trajectory_diagnosis(
            stages,
            signal_alignment=signal,
            gradient_alignment=gradients,
            signed_probes=signed,
        )
        self.assertEqual(
            result["finding"],
            "early_stopping_repairs_classifier_to_dgpo_direction",
        )
        self.assertEqual(result["best_local_stage"], "ba_lcb_55")

    def test_json_sanitizer_replaces_nonfinite_values_recursively(self):
        import json

        payload = {
            "python": float("nan"),
            "numpy": np.array([1.0, np.inf]),
            "tensor": torch.tensor([2.0, float("-inf")]),
        }
        sanitized = experiment._jsonable(payload)
        self.assertIsNone(sanitized["python"])
        self.assertEqual(sanitized["numpy"], [1.0, None])
        self.assertEqual(sanitized["tensor"], [2.0, None])
        json.dumps(sanitized, allow_nan=False)

    def test_ratio_fit_weights_match_truth_and_candidate_population_shapes(self):
        from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import _flatten_population

        condition = torch.randn(7, 5)
        truth = torch.randn(7, 4)
        candidates = torch.randn(7, 1, 4)
        populations = experiment._unit_weighted_populations(
            condition, truth, candidates
        )
        self.assertEqual(populations[2].shape, truth.shape[:-1])
        self.assertEqual(populations[5].shape, candidates.shape[:-1])
        self.assertEqual(populations[2].shape, (7,))
        self.assertEqual(populations[5].shape, (7, 1))
        _, flat_truth, flat_truth_weight = _flatten_population(*populations[:3])
        _, flat_candidates, flat_candidate_weight = _flatten_population(*populations[3:])
        self.assertEqual(flat_truth.shape, (7, 4))
        self.assertEqual(flat_candidates.shape, (7, 4))
        self.assertEqual(flat_truth_weight.shape, (7,))
        self.assertEqual(flat_candidate_weight.shape, (7,))

    def test_worker_uses_supported_ray_dataset_shard_api(self):
        source = inspect.getsource(experiment._worker)
        self.assertIn('ray.train.get_dataset_shard("pool")', source)
        self.assertNotIn("context.get_dataset_shard", source)

    def test_identity_partition_keeps_duplicates_together(self):
        generator = torch.Generator().manual_seed(4)
        identity = torch.randn(2000, 5, generator=generator)
        identity[100] = identity[0]
        identity[900] = identity[0]
        fractions = {
            "reward_fit": 0.50,
            "early_stop": 0.15,
            "calibration": 0.10,
            "judge_fit": 0.15,
            "final_audit": 0.10,
        }
        parts = experiment.identity_partitions(identity, fractions=fractions, seed=17)
        membership = {}
        for name, indices in parts.items():
            for index in indices.tolist():
                membership[index] = name
        self.assertEqual(membership[0], membership[100])
        self.assertEqual(membership[0], membership[900])
        self.assertEqual(len(membership), len(identity))
        self.assertEqual(sum(len(value) for value in parts.values()), len(identity))

    def test_response_summary_recognizes_perfect_and_shifted_matrix(self):
        truth = np.tile(np.array([[-0.8, -2.0, -0.6, -1.0],
                                  [-0.2, -0.5, 0.0, 0.0],
                                  [0.4, 0.8, 0.6, 1.2],
                                  [0.9, 2.2, 1.0, 2.5]]), (50, 1))
        perfect = np.repeat(truth[:, None, :], 3, axis=1)
        shifted = perfect.copy()
        shifted[:, :, (0, 2)] += 0.5
        exact, scores = experiment.response_matrix_metrics(truth, perfect, bins=4)
        moved, _ = experiment.response_matrix_metrics(truth, shifted, bins=4)
        self.assertEqual(set(scores), set(exact))
        for component in exact:
            self.assertAlmostEqual(exact[component]["diagonal_fraction"], 1.0)
            self.assertAlmostEqual(exact[component]["mean_abs_bin_offset"], 0.0)
        self.assertGreater(
            moved["tau_a_delta_theta"]["mean_abs_bin_offset"],
            exact["tau_a_delta_theta"]["mean_abs_bin_offset"],
        )

    def test_checked_in_config_pins_c4_last_and_one_increment_protocol(self):
        path = experiment.ROOT / "config/dgpo_10pct_c4a91e07_h4_reward_interface.yaml"
        cfg = yaml.safe_load(path.read_text())
        validated = experiment._validated_settings(cfg)
        self.assertEqual(validated["expected_policy_step"], 1110)
        self.assertIn("c4a91e07", validated["source_wandb_run"])
        self.assertEqual(validated["training_seeds"], [20260913, 20260914])
        self.assertEqual(validated["folds"], 2)
        self.assertEqual(validated["K"], 8)
        self.assertEqual(validated["raw_tempering"], 0.75)
        self.assertEqual(validated["classifier_fit"]["min_steps"], 1000)
        self.assertEqual(validated["classifier_fit"]["steps"], 3000)
        self.assertEqual(
            validated["classifier_fit"]["checkpoint_selection_metric"],
            "balanced_accuracy",
        )
        self.assertFalse("hard_trust" in validated)
        with self.assertRaises(ValueError):
            experiment._validated_settings({**cfg, "training_seeds": [1]})
        with self.assertRaises(ValueError):
            experiment._validated_settings({**cfg, "K": 1})

    def test_trajectory_config_is_one_same_path_ablation(self):
        path = (
            experiment.ROOT
            / "config/dgpo_10pct_c4a91e07_h4_weak_classifier_trajectory.yaml"
        )
        cfg = yaml.safe_load(path.read_text())
        validated = experiment._validated_settings(cfg)
        trajectory = validated["classifier_trajectory"]
        self.assertEqual(
            trajectory["balanced_accuracy_lcb_targets"], [0.55, 0.60, 0.70]
        )
        self.assertEqual(trajectory["confidence_z"], 1.96)
        self.assertEqual(trajectory["reward_arm"], "raw_loo")
        self.assertEqual(validated["expected_policy_step"], 1110)
        self.assertTrue(validated["wandb"]["enabled"])
        source = inspect.getsource(experiment._worker)
        self.assertIn("captured_trajectory", source)
        self.assertIn("clone_state(_model.state_dict())", source)
        self.assertIn("_run_classifier_trajectory_analysis", source)
        with self.assertRaisesRegex(ValueError, "unique and increasing"):
            experiment._validated_settings({
                **cfg,
                "classifier_trajectory": {
                    **trajectory,
                    "balanced_accuracy_lcb_targets": [0.70, 0.55],
                },
            })

    def test_h2_bandwidth_actionability_config_uses_h4_judge(self):
        root = experiment.ROOT / "config"
        h2 = experiment._validated_settings(yaml.safe_load(
            (root / "dgpo_10pct_c4a91e07_h2_bandwidth_actionability.yaml").read_text()
        ))
        self.assertEqual(h2["reward_classifier_overrides"], {"topology_max_harmonic": 2})
        self.assertEqual(h2["judge_classifier_overrides"], {"topology_max_harmonic": 4})
        self.assertEqual(h2["training_seeds"], [20260913])
        self.assertEqual(h2["actionability_sweep"]["stages"], ["fully_trained"])
        self.assertEqual(h2["actionability_sweep"]["step_rms_values"][-2:], [3e-5, 1e-4])
        self.assertEqual(len(h2["actionability_sweep"]["rollout_seeds"]), 8)
        self.assertEqual(
            h2["historical_actionability_reference"]["saturated_h4_runs"],
            ["aqbszk1r", "qlhcslov", "h4grad01"],
        )
        self.assertEqual(
            h2["historical_actionability_reference"]["curriculum_gate_radius"],
            3e-5,
        )
        with self.assertRaisesRegex(ValueError, "unsupported keys"):
            experiment._validated_settings({
                **h2, "reward_classifier_overrides": {"learning_rate": 1e-4}
            })
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            experiment._validated_settings({
                **h2, "reward_classifier_overrides": {"topology_max_harmonic": 0}
            })

    def test_conditional_h2_series_skips_old_base_for_rank_arms(self):
        root = experiment.ROOT / "config"
        loaded = {}
        for arm in ("old", "rank2", "rank4"):
            loaded[arm] = experiment._validated_settings(yaml.safe_load(
                (root / f"dgpo_10pct_c4a91e07_nested_residual_{arm}.yaml").read_text()
            ))

        self.assertEqual(
            [loaded[arm]["reward_classifier_overrides"]["conditional_residual_rank"]
             for arm in ("old", "rank2", "rank4")],
            [0, 2, 4],
        )
        self.assertNotIn("conditional_residual_fit", loaded["old"])
        self.assertFalse(loaded["old"]["conditional_residual_only"])
        for arm in ("rank2", "rank4"):
            self.assertNotIn("conditional_residual_fit", loaded[arm])
            self.assertTrue(loaded[arm]["conditional_residual_only"])
            self.assertEqual(loaded[arm]["classifier_fit"]["steps"], 3000)
            self.assertEqual(loaded[arm]["classifier_fit"]["min_steps"], 1000)
            self.assertFalse(
                loaded[arm]["reward_classifier_overrides"][
                    "topology_fourier_embedding"
                ]
            )
        for cfg in loaded.values():
            self.assertEqual(cfg["classifier_fit"]["min_steps"], 1000)
            self.assertEqual(
                cfg["classifier_fit"]["checkpoint_selection_metric"], "loss"
            )
            self.assertEqual(
                cfg["classifier_trajectory"]["balanced_accuracy_lcb_targets"], []
            )
            self.assertEqual(
                cfg["actionability_sweep"]["stages"], ["fully_trained"]
            )
            self.assertEqual(
                cfg["judge_classifier_overrides"]["topology_max_harmonic"], 4
            )
            self.assertTrue(
                cfg["judge_classifier_overrides"]["topology_fourier_embedding"]
            )
            self.assertEqual(
                cfg["judge_classifier_overrides"]["conditional_residual_rank"], 0
            )
        self.assertEqual(loaded["old"]["classifier_fit"]["steps"], 1000)
        self.assertEqual(
            loaded["old"]["reward_classifier_overrides"]["head_dropout"], 0.25
        )
        bad = {**loaded["rank2"], "conditional_residual_only": False}
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            experiment._validated_settings(bad)
        both = {
            **loaded["rank2"],
            "conditional_residual_fit": loaded["rank2"]["classifier_fit"],
        }
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            experiment._validated_settings(both)
        bad_selection = {
            **loaded["rank2"],
            "classifier_fit": {
                **loaded["rank2"]["classifier_fit"],
                "checkpoint_selection_metric": "balanced_accuracy",
            },
        }
        with self.assertRaisesRegex(ValueError, "validation loss"):
            experiment._validated_settings(bad_selection)

    def test_actionability_sweep_selects_reliable_larger_radius(self):
        settings = {
            "stages": ["ba_lcb_55"],
            "primary_stage": "ba_lcb_55",
            "step_rms_values": [1.0e-6, 1.0e-5],
            "minimum_plus_beats_zero_fraction": 0.75,
            "minimum_plus_beats_minus_fraction": 1.0,
            "bootstrap_seed": 17,
            "bootstrap_replicates": 1000,
        }

        def rows(delta):
            return [
                {
                    "rollout_seed": seed,
                    "zero_judge_auc_gap": 0.30,
                    "plus_judge_auc_gap": 0.30 + delta,
                    "minus_judge_auc_gap": 0.31,
                }
                for seed in range(8)
            ]

        sweep = {
            "ba_lcb_55": {
                "1e-06": {"step_rms": 1.0e-6, "seeds": rows(-0.001)},
                "1e-05": {"step_rms": 1.0e-5, "seeds": rows(-0.006)},
            }
        }
        result = experiment._actionability_sweep_diagnosis(sweep, settings)
        self.assertEqual(result["finding"], "larger_ba55_radius_supported")
        self.assertEqual(result["selected_step_rms"], 1.0e-5)

    def test_preflight_forces_weights_only_and_pins_exact_last_checkpoint(self):
        checked_in = yaml.safe_load(
            (experiment.ROOT / "config/dgpo_10pct_c4a91e07_h4_reward_interface.yaml").read_text()
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("source/checkpoints", "backbone", "normalization", "data", "config"):
                (root / name).mkdir(parents=True, exist_ok=True)
            checkpoint = root / "source/checkpoints/last.ckpt"
            torch.save({"state_dict": {"w": torch.ones(2)}, "global_step": 1110}, checkpoint)
            backbone = root / "backbone/last.ckpt"
            torch.save({"state_dict": {"w": torch.zeros(2)}}, backbone)
            normalization = root / "normalization/n.pt"
            torch.save({"mean": torch.zeros(2)}, normalization)
            (root / "data/part.parquet").write_bytes(b"metadata-only")
            h4 = {
                "periodic_pair_features": True,
                "topology_fourier_embedding": True,
                "topology_conditioning": False,
                "visible_pair_rest_frame": False,
                "topology_max_harmonic": 4,
                "topology_include_theta_pair": False,
                "topology_direct_logit": False,
            }
            runtime = {
                "options": {"Training": {
                    "model_checkpoint_load_path": str(checkpoint),
                    "EMA": {"replace_model_after_load": True},
                }, "Dataset": {"normalization_file": str(normalization)}},
                "platform": {"data_parquet_dir": str(root / "data")},
                "reward_config": {"omnifold": {"backbone_checkpoint": str(backbone)}},
                "dgpo": {
                    "auto_resume_from_last": True,
                    "checkpoint_load_mode": "resume",
                    "adaptive_omnifold": {"recalibration": h4},
                },
            }
            base = root / "config/base.yaml"
            overlay = root / "config/overlay.yaml"
            base.write_text(yaml.safe_dump(runtime))
            overlay.write_text("{}\n")
            settings = {
                **checked_in,
                "base_config": str(base),
                "overlay_config": str(overlay),
                "expected_policy_checkpoint": str(checkpoint),
                "output_dir": str(root / "output"),
                "wandb": {"enabled": False},
            }
            prepared = experiment.prepare(settings)
            self.assertEqual(prepared["policy_checkpoint"], str(checkpoint.resolve()))
            self.assertFalse(prepared["runtime"]["dgpo"]["auto_resume_from_last"])
            self.assertEqual(prepared["runtime"]["dgpo"]["checkpoint_load_mode"], "weights_only")
            self.assertFalse(
                prepared["runtime"]["options"]["Training"]["EMA"]["replace_model_after_load"]
            )
            experiment.verify_sources(prepared)
            with self.assertRaisesRegex(ValueError, "declared c4a91e07 last.ckpt"):
                experiment.prepare({
                    **settings,
                    "expected_policy_checkpoint": str(root / "source/checkpoints/other.ckpt"),
                })

    def test_signed_direction_plus_is_gradient_descent_and_restores_zero(self):
        model = torch.nn.Linear(2, 1, bias=False)
        parameters = tuple(model.parameters())
        anchor = tuple(parameter.detach().clone() for parameter in parameters)
        direction = torch.tensor([3.0, 4.0])
        scale = experiment._assign_direction(
            parameters, anchor, direction, sign=1, epsilon_rms=1.0e-3
        )
        delta = torch.cat([(parameter - base).reshape(-1) for parameter, base in zip(parameters, anchor)])
        self.assertLess(float(torch.dot(delta, direction)), 0.0)
        self.assertAlmostEqual(float(delta.square().mean().sqrt()), 1.0e-3, places=7)
        self.assertGreater(scale, 0.0)
        experiment._assign_direction(parameters, anchor, direction, sign=0, epsilon_rms=1.0e-3)
        torch.testing.assert_close(parameters[0], anchor[0])


if __name__ == "__main__":
    unittest.main()
