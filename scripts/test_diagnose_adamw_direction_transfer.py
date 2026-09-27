from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagnose_adamw_direction_transfer as experiment


def settings() -> dict:
    return {
        "base_config": "/base.yaml",
        "source_checkpoint": "/source.ckpt",
        "source_runtime": "/runtime.yaml",
        "event_pool": "/pool.pt",
        "output_dir": "/output",
        "expected_source_wandb_id": "h4xfer01",
        "expected_policy_step": 20,
        "expected_reward_round_id": 1,
        "workers": 16,
        "cpus_per_worker": 2,
        "K": 8,
        "events_per_worker": 512,
        "event_microbatch_size": 128,
        "gradient_timesteps": 8,
        "gradient_candidate_seed": 1,
        "gradient_seed": 2,
        "panel_selection_seed": 3,
        "curvature_events": 2048,
        "curvature_candidate_seed": 4,
        "curvature_timesteps": 8,
        "curvature_path_seed": 5,
        "native_parameter_rms": 1e-6,
        "distance_match_tolerance": 0.05,
        "distance_match_iterations": 4,
        "minimum_parameter_rms": 1e-8,
        "maximum_parameter_rms": 3e-5,
        "judge_events": 2048,
        "judge_rollout_seeds": list(range(11, 19)),
        "score_batch_size": 32768,
        "save_direction_vectors": True,
        "upload_direction_vectors": False,
        "wandb": {"enabled": False},
    }


class TestAdamWDirectionTransfer(unittest.TestCase):
    def test_protocol_is_pinned(self):
        cfg = experiment._validated_settings(settings())
        self.assertEqual(cfg["workers"], 16)
        self.assertEqual(cfg["events_per_worker"], 512)
        self.assertEqual(len(cfg["judge_rollout_seeds"]), 8)

    def test_rejects_mathematical_or_power_changes(self):
        for key, value in (
            ("workers", 8),
            ("K", 4),
            ("events_per_worker", 256),
            ("event_microbatch_size", 64),
            ("gradient_timesteps", 4),
            ("judge_rollout_seeds", [1]),
            ("curvature_events", 2047),
            ("judge_events", 2047),
            ("native_parameter_rms", 0.0),
            ("distance_match_tolerance", 0.5),
        ):
            bad = copy.deepcopy(settings())
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                experiment._validated_settings(bad)

    def test_live_wandb_retry_keeps_the_predeclared_run(self):
        live = copy.deepcopy(settings())
        live["wandb"] = {
            "enabled": True,
            "required": True,
            "id": "h4opt01",
            "resume": "allow",
        }
        self.assertEqual(
            experiment._validated_settings(live)["wandb"]["resume"], "allow"
        )
        live["wandb"]["resume"] = "never"
        with self.assertRaises(ValueError):
            experiment._validated_settings(live)

    def test_panel_selection_is_disjoint(self):
        cfg = settings()
        cfg["total_gradient_events"] = cfg["workers"] * cfg["events_per_worker"]
        final = torch.arange(14000)
        gradient, curvature, judge = experiment.select_panels(final, cfg)
        self.assertEqual(len(gradient), 8192)
        self.assertEqual(len(curvature), 2048)
        self.assertEqual(len(judge), 2048)
        self.assertTrue(set(gradient.tolist()).isdisjoint(curvature.tolist()))
        self.assertTrue(set(gradient.tolist()).isdisjoint(judge.tolist()))
        self.assertTrue(set(curvature.tolist()).isdisjoint(judge.tolist()))

    def test_analytic_native_direction_matches_torch_adamw(self):
        parameter = torch.nn.Parameter(torch.tensor([1.5, -0.5, 0.25]))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4, betas=(0.8, 0.95), eps=1e-7, weight_decay=0.03)
        for gradient in (torch.tensor([0.2, -0.4, 0.6]), torch.tensor([-0.3, 0.1, 0.5])):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = gradient.clone()
            optimizer.step()
        gradient = torch.tensor([0.7, -0.2, 0.1])
        before = parameter.detach().clone()
        expected = experiment._adamw_parameter_descent(
            parameter, gradient, optimizer.state[parameter], optimizer.param_groups[0],
            keep_first_moment=True, keep_second_moment=True, keep_clock=True,
            use_weight_decay=True,
        )
        parameter.grad = gradient.clone()
        optimizer.step()
        observed = before - parameter.detach()
        self.assertTrue(torch.allclose(expected, observed, rtol=2e-5, atol=1e-8))

    def test_counterfactual_builder_does_not_mutate_state_or_parameter(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.1)
        parameter.grad = torch.tensor([0.2, -0.3])
        optimizer.step()
        before_parameter = parameter.detach().clone()
        before_m = optimizer.state[parameter]["exp_avg"].clone()
        directions, diagnostics = experiment.adamw_counterfactual_directions(
            (parameter,), optimizer, torch.tensor([0.4, -0.1])
        )
        self.assertEqual(tuple(directions), experiment.ARM_ORDER)
        self.assertTrue(torch.equal(parameter, before_parameter))
        self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"], before_m))
        self.assertEqual(diagnostics["optimizer_state_step_min"], 1)

    def test_decision_requires_paired_raw_advantage(self):
        summaries = {
            name: {
                "plus_gap_mean": value,
                "plus_beats_minus_count": 8,
                "improvement_zero_minus_plus": {"mean": 0.02},
            }
            for name, value in zip(experiment.ARM_ORDER, (.25, .245, .21, .22, .20))
        }
        decision = experiment.classify_optimizer_result(
            summaries,
            {"ci95_low": 0.02},
            all_distance_matches=True,
        )
        self.assertEqual(decision, "stale_first_moment_is_material_optimizer_bottleneck")
        inconclusive = experiment.classify_optimizer_result(
            summaries,
            {"ci95_low": -0.001},
            all_distance_matches=True,
        )
        self.assertEqual(inconclusive, "optimizer_transform_not_established")


if __name__ == "__main__":
    unittest.main()
