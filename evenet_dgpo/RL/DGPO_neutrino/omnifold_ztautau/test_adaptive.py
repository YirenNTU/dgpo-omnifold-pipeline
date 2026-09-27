"""Contract tests for adaptive Ztautau OmniFold orchestration."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from RL.DGPO_neutrino.omnifold_ztautau import adaptive as adaptive_module
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
    AdaptiveOmniFoldPool,
    AdaptiveOmniFoldState,
    auc_null_standard_error,
    adaptive_audit_protocol_signature,
    adaptive_trust_policy_lr_scale,
    baseline_probe_auc_gap,
    build_reference_trust_pool,
    clamp_fixed_trust_radius_after_resume,
    evaluate_round_auc_change,
    evaluate_signed_direction_probe,
    evaluate_signed_direction_recovery,
    install_adaptive_trust_round,
    record_extragradient_rejection,
    record_reference_trust_attempt,
    record_reference_trust_distance,
    resolve_adaptive_config,
    reward_refit_due_to_age,
    run_adaptive_refit,
    should_probe_epoch,
    should_skip_incumbent_probe,
    should_run_raw_only_monitor,
    trust_region_exhausted,
    update_controller,
    update_classifier_trust_controller,
    update_empirical_trust_radius_from_audit,
    update_raw_plateau_controller,
    update_trust_trajectory_candidate,
    validate_adaptive_pairing,
)
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EventPackingSpec,
    fit_independent_evenet_audit,
)
from RL.DGPO_neutrino.model_utils import state_dict_sha256


def _config(
    *,
    log_only: bool = False,
    monitor_mode: str = "weighted_and_raw",
    candidates_per_event: int = 1,
    retrain_auc_margin: float = 0.01,
    acceptance_audit_enabled: bool = True,
    acceptance_max_balanced_accuracy: float = 0.51,
    residual_min_auc_gain: float = 1.0e-3,
    require_audit_saturation: bool = False,
    max_reward_age_epochs: int | None = 20,
    required_consecutive_epochs: int = 1,
    retrain_cooldown_epochs: int = 0,
    trust_boundary_enabled: bool = False,
    trust_radius_mode: str = "auc_scaled",
    trust_delta_max: float = 1.0e-3,
    trust_delta_floor: float = 5.0e-6,
    trust_round_decay_factor: float = 0.9,
    trust_distance: str = "velocity_mse_ratio",
    trust_initial_raw_auc_gap: float | None = None,
    trust_empirical_radius_enabled: bool = False,
    trust_empirical_bidirectional: bool = False,
    trust_empirical_allow_expansion: bool = True,
    trust_cross_round_nonexpanding: bool = False,
    trust_policy_lr_scaling_enabled: bool = False,
    trust_policy_lr_scale_floor: float = 0.1,
    trust_refit_on_exhaustion: bool = False,
    trust_exhaustion_scale_window_steps: int = 50,
    trust_round_acceptance_enabled: bool = False,
    trust_round_acceptance_confidence_z: float = 1.96,
    trust_round_plateau_patience: int = 2,
    trust_trajectory_search_enabled: bool = False,
    trust_failed_direction_patience: int = 3,
    trust_signed_direction_probe_enabled: bool = False,
    trust_signed_direction_probe_scales: tuple[float, ...] = (0.25, 0.5, 1.0),
    trust_signed_direction_recovery_enabled: bool = False,
    trust_extragradient_enabled: bool = False,
    trust_extragradient_lookahead_scale: float = 0.5,
    reference_trust_coefficient: float = 1.0,
    trust_reset_adam_first_moment_on_zero_step: bool = False,
    beta: float = 1.0,
    raw_audit_enabled: bool = False,
    raw_improvement_min_delta: float = 0.0,
    raw_rollback_to_best_on_plateau: bool = False,
    classifier_trust_enabled: bool = False,
    classifier_trust_max_balanced_accuracy: float = 0.525,
    classifier_trust_every_n_epochs: int = 2,
    classifier_trust_probe_max_events: int | None = 100,
    classifier_trust_validation_patience_epochs: float = 5.0,
    classifier_trust_unsafe_early_stop_enabled: bool = False,
    classifier_trust_unsafe_early_stop_confidence_z: float = 1.96,
    classifier_trust_unsafe_early_stop_required_consecutive: int = 2,
    refit_once_fail_closed: bool = False,
    topology_acceptance_repeats: int = 1,
    fixed_audit_panel: bool = False,
    pool_data_parquet_dir: str | None = None,
    reset_adam_first_moment_on_install: bool | None = None,
    warm_start_iterations: tuple[int, ...] = (),
    crossfit_repeats: int = 1,
):
    payload = {
            "reference_trust": {
                "enabled": True,
                "coefficient": reference_trust_coefficient,
                "objective": (
                    "vp_path_kl"
                    if trust_distance == "vp_path_kl"
                    else "velocity_mse"
                ),
                "adaptive_boundary": {
                    "enabled": trust_boundary_enabled,
                    "radius_mode": trust_radius_mode,
                    "distance": trust_distance,
                    "delta_max": trust_delta_max,
                    "delta_floor": trust_delta_floor,
                    "round_decay_factor": trust_round_decay_factor,
                    "initial_raw_auc_gap": trust_initial_raw_auc_gap,
                    "warning_fraction": 0.8,
                    "adaptive_power": 2.0,
                    "auc_confidence_z": 1.96,
                    "auc_stop_gap": 0.01,
                    "reset_adam_first_moment_on_install": True,
                    "reset_adam_first_moment_on_zero_step": (
                        trust_reset_adam_first_moment_on_zero_step
                    ),
                    "enforcement": "post_step_backtracking",
                    "backtrack_factor": 0.5,
                    "max_backtracks": 16,
                    "probe_events_per_rank": 64,
                    "fixed_probe_per_reward_round": True,
                    "interior_fraction": 0.8,
                    "radius_calibration": {
                        "enabled": trust_empirical_radius_enabled,
                        "bidirectional": trust_empirical_bidirectional,
                        "allow_expansion": trust_empirical_allow_expansion,
                        "cross_round_nonexpanding": (
                            trust_cross_round_nonexpanding
                        ),
                        "scale_policy_lr": trust_policy_lr_scaling_enabled,
                        "policy_lr_scale_floor": trust_policy_lr_scale_floor,
                        "refit_on_exhaustion": trust_refit_on_exhaustion,
                        "exhaustion_distance_fraction": 0.98,
                        "exhaustion_mean_update_scale": 0.01,
                        "exhaustion_scale_window_steps": (
                            trust_exhaustion_scale_window_steps
                        ),
                        "round_acceptance_enabled": (
                            trust_round_acceptance_enabled
                        ),
                        "round_acceptance_confidence_z": (
                            trust_round_acceptance_confidence_z
                        ),
                        "round_plateau_patience": trust_round_plateau_patience,
                        "trajectory_search_enabled": (
                            trust_trajectory_search_enabled
                        ),
                        "failed_direction_patience": (
                            trust_failed_direction_patience
                        ),
                        "signed_direction_probe": {
                            "enabled": trust_signed_direction_probe_enabled,
                            "scales": list(trust_signed_direction_probe_scales),
                            "recover_reverse_only": (
                                trust_signed_direction_recovery_enabled
                            ),
                        },
                        "lookahead_extragradient": {
                            "enabled": trust_extragradient_enabled,
                            "scale": trust_extragradient_lookahead_scale,
                        },
                        "safety_factor": 0.5,
                        "expand_factor": 1.25,
                        "shrink_factor": 0.5,
                        "target_acceptance_rate": 0.5,
                        "target_update_scale": 0.05,
                        "safe_audits_required": 2,
                        "attempt_window_steps": 50,
                        "distance_window_steps": 10,
                        "min_distance_samples": 5,
                        "confidence_z": 1.96,
                    },
                },
            },
            "beta": beta,
            "adaptive_omnifold": {
                "enabled": True,
                "log_only": log_only,
                "monitor_mode": monitor_mode,
                "staleness_every_n_epochs": 2,
                "fixed_audit_panel": fixed_audit_panel,
                "pool_generation_batch_size": 1024,
                "trigger": {
                    "retrain_auc_margin": retrain_auc_margin,
                    "raw_audit_enabled": raw_audit_enabled,
                    "raw_improvement_min_delta": raw_improvement_min_delta,
                    "rollback_to_best_on_plateau": (
                        raw_rollback_to_best_on_plateau
                    ),
                    "max_reward_age_epochs": max_reward_age_epochs,
                    "required_consecutive_epochs": required_consecutive_epochs,
                    "retrain_cooldown_epochs": retrain_cooldown_epochs,
                    "require_audit_saturation": require_audit_saturation,
                    "probe_max_events": 100,
                },
                "classifier_trust": {
                    "enabled": classifier_trust_enabled,
                    "every_n_epochs": classifier_trust_every_n_epochs,
                    "probe_max_events": classifier_trust_probe_max_events,
                    "max_balanced_accuracy": (
                        classifier_trust_max_balanced_accuracy
                    ),
                    "confidence_z": 1.96,
                    "required_consecutive_epochs": 1,
                    "require_saturation": True,
                    "audit_fit": {
                        "validation_patience_epochs": (
                            classifier_trust_validation_patience_epochs
                        ),
                    },
                    "unsafe_early_stop": {
                        "enabled": classifier_trust_unsafe_early_stop_enabled,
                        "confidence_z": (
                            classifier_trust_unsafe_early_stop_confidence_z
                        ),
                        "required_consecutive_validations": (
                            classifier_trust_unsafe_early_stop_required_consecutive
                        ),
                    },
                },
                "audit_fit": {
                    "batch_size": 64,
                    "validation_interval_epochs": 1.0,
                    "validation_patience_epochs": 5.0,
                },
                "recalibration": {
                    "candidates_per_event": candidates_per_event,
                    "train_parquet_dir": pool_data_parquet_dir,
                    "refit_once_on_resume": True,
                    "refit_once_fail_closed": refit_once_fail_closed,
                    "refit_once_id": "test_regularized_v1",
                    "pool_selection_seed": 73,
                    "score_pool_events": 200,
                    "min_iterations": 2,
                    "max_iterations": 4,
                    "warm_start_iterations": list(warm_start_iterations),
                    "fit": {"weight_decay": 0.0005},
                    "acceptance_audit_enabled": acceptance_audit_enabled,
                    "acceptance_max_balanced_accuracy": (
                        acceptance_max_balanced_accuracy
                    ),
                    "topology_acceptance_repeats": (
                        topology_acceptance_repeats
                    ),
                    "crossfit_repeats": crossfit_repeats,
                    "residual_min_auc_gain": residual_min_auc_gain,
                },
            }
        }
    if reset_adam_first_moment_on_install is not None:
        payload["adaptive_omnifold"]["recalibration"][
            "reset_adam_first_moment_on_install"
        ] = bool(reset_adam_first_moment_on_install)
    return resolve_adaptive_config(payload)


class TestEpochRefitAblation(unittest.TestCase):
    def test_10pct_eight_misses_and_independent_ten_epoch_validation(self):
        import yaml
        from RL.DGPO_neutrino.dgpo_trainer import validation_schedule_tier
        path = Path(__file__).resolve().parents[4] / "config/dgpo_omnifold_ztautau_10pct_resume_v26.yaml"
        dg = yaml.safe_load(path.read_text())["dgpo"]
        cfg = resolve_adaptive_config(dg)
        self.assertEqual(cfg.staleness_every_n_steps, 5)
        self.assertEqual(cfg.staleness_every_n_epochs, 1)
        self.assertEqual(cfg.required_consecutive_epochs, 8)
        state = AdaptiveOmniFoldState(raw_global_initialized=True, raw_best_auc_gap=.1)
        triggers = []
        for step in range(1, 41):
            epoch, end = (step - 1)//10, step % 10 == 0
            if adaptive_module.should_probe_training_boundary(state, cfg=cfg, epoch=epoch,
                                                               global_step=step, epoch_end=end):
                fired, _ = update_raw_plateau_controller(
                    state, {"raw_auc_gap": .12, "raw_audit_saturated": 1.},
                    cfg=cfg, epoch=epoch, global_step=step,
                )
                if fired:
                    triggers.append(step)
        self.assertEqual(triggers, [40])
        tiers = [validation_schedule_tier(e, cheap_every_n_epochs=dg["validation_every_n_epochs"],
                                          full_every_n_epochs=dg["validation_full_every_n_epochs"])
                 for e in range(20)]
        self.assertEqual(tiers, ([None] * 9 + ["full"]) * 2)

    def test_v26_resume_configs_preserve_training_protocol_and_start_new_wandb(self):
        import copy
        import yaml
        from scripts.train_neutrino_backend import deep_update
        root = Path(__file__).resolve().parents[4] / "config"
        sources = {"1pct": "dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml",
                   "10pct": "dgpo_omnifold_ztautau_10pct_velocity_mse_trust_ablation.yaml"}
        for budget, source in sources.items():
            cold = yaml.safe_load((root / source).read_text())
            resumed = yaml.safe_load((root / f"dgpo_omnifold_ztautau_{budget}_resume_v26.yaml").read_text())
            expected = copy.deepcopy(cold["dgpo"])
            expected["checkpoint_load_mode"] = "resume"
            expected["adaptive_omnifold"]["recalibration"]["bootstrap_on_start"] = False
            if budget == "10pct":
                expected["auto_resume_from_last"] = True
                expected["lr_schedule"] = {"type": "cosine", "total_steps": 1500, "min_lr_ratio": 0.1}
                expected["validation_every_n_epochs"] = 10
                expected["validation_full_every_n_epochs"] = 10
                expected["adaptive_omnifold"]["recalibration"]["policy_warmup_steps"] = 10
                expected["tarp"]["enabled"] = False
                expected["ztautau_metrics"]["log_images"] = False
                old_root = cold["options"]["Training"]["model_checkpoint_save_path"]
                expected["global_best_checkpoint_search_dirs"] = [old_root, old_root.replace("_v26_", "_v26_resume1_")]
                expected["adaptive_omnifold"]["trigger"].update({
                    "raw_improvement_min_delta": .001, "best_scope": "global",
                    "global_confirm_candidates": False, "global_max_failed_rounds": 2,
                    "pause_patience_during_warmup": True,
                    "required_consecutive_checks": 8,
                })
            self.assertEqual(resumed["dgpo"], expected)
            self.assertEqual(resumed["platform"], cold["platform"])
            self.assertEqual(resumed["network"], cold["network"])
            self.assertEqual(resumed["reward_config"], cold["reward_config"])
            old_path = cold["options"]["Training"]["model_checkpoint_save_path"]
            source_path = (old_path.replace("_v26_", "_v26_resume2_")
                           if budget == "10pct" else old_path)
            training = resumed["options"]["Training"]
            if budget == "10pct":
                self.assertEqual(training["model_checkpoint_load_path"], training["model_checkpoint_save_path"] + "/last.ckpt")
            else:
                self.assertEqual(training["model_checkpoint_load_path"], source_path + "/last.ckpt")
            self.assertNotEqual(training["model_checkpoint_save_path"], old_path)
            self.assertNotEqual(training["model_checkpoint_save_path"], source_path)
            output_suffix = "v26_resume3_" if budget == "10pct" else "v26_resume1_"
            self.assertIn(output_suffix, training["model_checkpoint_save_path"])
            output_root = str(Path(training["model_checkpoint_save_path"]).parent)
            self.assertEqual(resumed["nersc"]["ray"]["results_dir"], output_root + "/ray_results")
            self.assertEqual(resumed["logger"]["local"]["save_dir"], output_root + "/logs")
            wb = resumed["logger"]["wandb"]
            self.assertEqual(wb["resume"], "never")
            self.assertTrue(wb["fresh_run"])
            self.assertIsNone(wb["id"])
            self.assertNotEqual(wb["run_name"], cold["logger"]["wandb"]["run_name"])
            runtime = deep_update(yaml.safe_load((root / "train_diffusion_nersc.yaml").read_text()), resumed)
            self.assertTrue({"event_info", "resonance", "options", "network"}.issubset(runtime))
            cfg = resolve_adaptive_config(runtime["dgpo"])
            self.assertFalse(cfg.bootstrap_on_start)
            self.assertFalse(cfg.refit_once_on_resume)
            self.assertEqual(adaptive_module.adaptive_audit_protocol_signature(cfg),
                             adaptive_module.adaptive_audit_protocol_signature(resolve_adaptive_config(cold["dgpo"])))

    def test_10pct_v23_matches_1pct_protocol_without_cross_budget_paths(self):
        import copy
        import yaml
        from scripts.train_neutrino_backend import deep_update

        root = Path(__file__).resolve().parents[4] / "config"
        one = yaml.safe_load((root / "dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml").read_text())
        ten = yaml.safe_load((root / "dgpo_omnifold_ztautau_10pct_velocity_mse_trust_ablation.yaml").read_text())
        previous = yaml.safe_load((root / "dgpo_omnifold_ztautau_10pct_scaling_raw_plateau_vpkl.yaml").read_text())
        expected_dgpo = copy.deepcopy(ten["dgpo"])
        self.assertEqual(expected_dgpo["adaptive_omnifold"]["recalibration"]["fit"].pop("anomaly_detection_steps"), 0)
        ten_fit = expected_dgpo["adaptive_omnifold"]["recalibration"]["fit"]
        one_fit = one["dgpo"]["adaptive_omnifold"]["recalibration"]["fit"]
        for key, expected in (("batch_size", 16384), ("train_microbatch_size_per_rank", 1024)):
            self.assertEqual(ten_fit[key], expected)
            self.assertEqual(ten_fit[key], 4 * one_fit[key])
            ten_fit[key] = one_fit[key]
        self.assertEqual(expected_dgpo["adaptive_omnifold"]["recalibration"]["residual_min_auc_gain"], .01)
        self.assertEqual(one["dgpo"]["adaptive_omnifold"]["recalibration"]["residual_min_auc_gain"], .02)
        expected_dgpo["adaptive_omnifold"]["recalibration"]["residual_min_auc_gain"] = .02
        ten_audit = expected_dgpo["adaptive_omnifold"]["audit_fit"]
        one_audit = one["dgpo"]["adaptive_omnifold"]["audit_fit"]
        for key, expected in (("batch_size", 32768), ("train_microbatch_size_per_rank", 2048)):
            self.assertEqual(ten_audit[key], expected)
            self.assertEqual(ten_audit[key], 4 * one_audit[key])
            ten_audit[key] = one_audit[key]
        self.assertEqual(expected_dgpo["adaptive_omnifold"]["trigger"]["probe_max_events"], 250000)
        expected_dgpo["adaptive_omnifold"]["trigger"]["probe_max_events"] = one["dgpo"]["adaptive_omnifold"]["trigger"]["probe_max_events"]
        self.assertEqual(expected_dgpo, one["dgpo"])
        self.assertEqual(ten["network"], one["network"])
        self.assertEqual(ten["options"]["Dataset"], one["options"]["Dataset"])
        expected_training = copy.deepcopy(ten["options"]["Training"])
        for key in ("model_checkpoint_load_path", "model_checkpoint_save_path"):
            expected_training[key] = one["options"]["Training"][key]
        self.assertEqual(expected_training, one["options"]["Training"])
        for key in ("data_parquet_dir", "data_parquet_val_dir"):
            self.assertEqual(ten["platform"][key], "/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train")
            self.assertNotEqual(ten["platform"][key], previous["platform"][key])
        self.assertEqual(ten["options"]["Training"]["model_checkpoint_load_path"], "/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt")
        self.assertEqual(ten["dgpo"]["checkpoint_load_mode"], "weights_only")
        self.assertFalse(ten["dgpo"]["auto_resume_from_last"])
        self.assertTrue(ten["dgpo"]["adaptive_omnifold"]["recalibration"]["bootstrap_on_start"])
        self.assertEqual(ten["reward_config"]["omnifold"]["backbone_checkpoint"], previous["reward_config"]["omnifold"]["backbone_checkpoint"])
        self.assertNotIn("1pct", yaml.safe_dump(ten))
        self.assertNotEqual(ten["options"]["Training"]["model_checkpoint_save_path"], previous["options"]["Training"]["model_checkpoint_save_path"])
        self.assertEqual(ten["nersc"]["reproducibility"]["dgpo_training_fraction"], .10)
        self.assertFalse(ten["options"]["Training"]["EMA"]["replace_model_after_load"])
        merged = deep_update(yaml.safe_load((root / "train_diffusion_nersc.yaml").read_text()), ten)
        self.assertTrue({"event_info", "resonance", "network", "options"}.issubset(merged))
        cfg = resolve_adaptive_config(merged["dgpo"])
        self.assertTrue(cfg.reset_optimizer_state_on_install)
        self.assertTrue(cfg.raw_rollback_to_best_on_plateau)
        self.assertEqual(cfg.required_consecutive_epochs, 5)
        self.assertEqual(cfg.policy_warmup_steps, 20)
        self.assertEqual(cfg.policy_warmup_start_factor, .1)
        self.assertEqual(cfg.staleness_every_n_steps, 5)
        self.assertEqual(cfg.trust_best_decay_factor, .9)
        self.assertAlmostEqual(.5 + cfg.residual_min_auc_gain, .51)
        self.assertEqual(cfg.probe_max_events, 250000)
        self.assertIsNone(cfg.pool_events)
        self.assertIsNone(cfg.refit_score_events)
        for key, value in (("pool_events", 250000), ("score_pool_events", 250000),
                           ("train_parquet_dir", "/different/source")):
            invalid = copy.deepcopy(merged["dgpo"])
            invalid["adaptive_omnifold"]["recalibration"][key] = value
            with self.assertRaises(ValueError):
                resolve_adaptive_config(invalid)
        invalid = copy.deepcopy(merged["dgpo"])
        invalid["adaptive_omnifold"]["trigger"]["probe_max_events"] = 0
        with self.assertRaises(ValueError):
            resolve_adaptive_config(invalid)

    def test_velocity_mse_ablation_matches_original_5pct_core(self):
        import yaml
        from RL.DGPO_neutrino.dgpo_utils import build_reference_trust_loss

        root = Path(__file__).resolve().parents[4] / "config"
        baseline = yaml.safe_load((root / "dgpo_omnifold_ztautau.yaml").read_text())["dgpo"]
        candidate = yaml.safe_load((root / "dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml").read_text())["dgpo"]
        for key in ("advantage_estimator", "K", "beta", "beta_kl", "num_ddim_steps",
                    "num_train_timesteps", "policy_eval_t_min", "policy_eval_t_max",
                    "adv_clip_max", "grad_clip_norm", "steps_per_epoch"):
            with self.subTest(key=key):
                self.assertEqual(candidate[key], baseline[key])
        self.assertFalse(candidate["sequential_vp_trust_backward"])
        self.assertFalse(baseline.get("sequential_vp_trust_backward", False))
        old_trust, new_trust = baseline["reference_trust"], candidate["reference_trust"]
        self.assertEqual(new_trust["coefficient"], old_trust["coefficient"])
        self.assertEqual(new_trust["objective"], old_trust.get("objective", "velocity_mse"))
        self.assertNotIn("vp_path_kl", new_trust)
        # Explicit new selection must preserve both the legacy loss and gradient.
        old_v = torch.tensor([[1., 3.], [2., 4.]], requires_grad=True)
        new_v = old_v.detach().clone().requires_grad_(True)
        ref = torch.zeros_like(old_v)
        mask = torch.tensor([[1., 0.], [1., 1.]])
        old_loss, _ = build_reference_trust_loss(old_v, ref, mask)
        new_loss, _ = build_reference_trust_loss(new_v, ref, mask, objective=new_trust["objective"])
        torch.testing.assert_close(new_loss, old_loss)
        torch.testing.assert_close(torch.autograd.grad(new_loss, new_v)[0],
                                   torch.autograd.grad(old_loss, old_v)[0])

    def test_velocity_mse_trust_ablation_uses_step_patience_rollback_and_best_decay(self):
        import yaml
        root = Path(__file__).resolve().parents[4]
        payload = yaml.safe_load((root / "config/dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml").read_text())
        cfg = resolve_adaptive_config(payload["dgpo"])
        trust = payload["dgpo"]["reference_trust"]
        self.assertEqual(trust["objective"], "velocity_mse")
        self.assertEqual(trust["coefficient"], 1.0)
        self.assertTrue(trust["adaptive_boundary"]["enabled"])
        self.assertEqual(cfg.trust_distance, "velocity_mse_ratio")
        self.assertEqual(cfg.trust_radius_mode, "best_decay")
        self.assertEqual(cfg.trust_delta_max, .1)
        self.assertEqual(cfg.trust_delta_floor, .02)
        self.assertEqual(cfg.trust_best_decay_factor, .9)
        self.assertNotIn("round_decay_factor", trust["adaptive_boundary"])
        state = AdaptiveOmniFoldState()
        for step in range(25):
            state.install(baseline_auc_gap=.2, cfg=cfg, epoch=step, round_id=step + 1)
            for _ in range(2):
                install_adaptive_trust_round(state, cfg=cfg, raw_auc=.7, raw_auc_se=.001)
                self.assertAlmostEqual(state.trust_current_delta, .1)
                self.assertEqual(state.trust_best_decay_count, 0)
            state = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(trust["adaptive_boundary"]["enforcement"], "post_step_backtracking")
        self.assertFalse(cfg.classifier_trust_enabled)
        self.assertFalse(cfg.periodic_pair_features_enabled)
        self.assertFalse(cfg.topology_fourier_embedding_enabled)
        self.assertEqual(payload["dgpo"]["adaptive_omnifold"]["recalibration"]["decoder_hidden_dim"], 128)
        self.assertEqual(payload["dgpo"]["adaptive_omnifold"]["recalibration"]["decoder_layers"], 1)
        self.assertEqual(payload["dgpo"]["adaptive_omnifold"]["recalibration"]["decoder_heads"], 4)
        self.assertTrue(cfg.raw_rollback_to_best_on_plateau)
        self.assertIsNone(cfg.max_reward_age_epochs)
        self.assertEqual(cfg.staleness_every_n_steps, 5)
        self.assertEqual(cfg.required_consecutive_epochs, 5)
        self.assertTrue(cfg.raw_monitor_warm_start)
        self.assertTrue(cfg.reset_optimizer_state_on_install)
        self.assertFalse(cfg.trust_reset_adam_first_moment)
        training = payload["options"]["Training"]
        self.assertEqual(training["weight_decay"], .001)
        self.assertTrue(training["decoupled_weight_decay"])
        for component in ("InvisibleInputProjector", "GroupedSequentialEmbedding",
                          "GlobalEmbedding", "PET", "TruthGeneration"):
            self.assertEqual(training["Components"][component]["weight_decay"], .001)
        # Classifier regularization remains independent of DGPO's AdamW settings.
        self.assertEqual(cfg.fit["weight_decay"], .0005)
        self.assertEqual(cfg.audit_fit["weight_decay"], .0005)
        self.assertEqual(cfg.warm_start_iterations, (1, 2))
        state = AdaptiveOmniFoldState(raw_best_auc_gap=.1)
        check_steps, fired_steps = [], []
        for step in range(1, 26):
            epoch, end = (step - 1) // 10, step % 10 == 0
            due, _ = adaptive_module.scheduled_raw_refit_due(state, cfg=cfg, epoch=epoch)
            self.assertFalse(due)
            if not adaptive_module.should_probe_training_boundary(
                state, cfg=cfg, epoch=epoch, global_step=step, epoch_end=end,
            ):
                continue
            check_steps.append(step)
            fired, _ = update_raw_plateau_controller(
                state, {"raw_auc_gap": .12, "raw_audit_saturated": 1},
                cfg=cfg, epoch=epoch, global_step=step,
            )
            if fired:
                fired_steps.append(step)
            state = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(check_steps, [5, 10, 15, 20, 25])
        self.assertEqual(fired_steps, [25])
        self.assertEqual(payload["dgpo"]["adaptive_omnifold"]["recalibration"]["head_dropout"], .15)
        self.assertEqual(cfg.audit_fit["head_dropout"], .15)
        self.assertAlmostEqual(.5 + cfg.residual_min_auc_gain, .52)
        self.assertFalse(payload["dgpo"]["auto_resume_from_last"])
        self.assertEqual(payload["dgpo"]["checkpoint_load_mode"], "weights_only")
        self.assertEqual(payload["options"]["Training"]["model_checkpoint_load_path"], "/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_1pct_seed42/checkpoints/epoch=172_train=0.2060_val=0.1993.ckpt")
        self.assertFalse(payload["options"]["Training"]["EMA"]["replace_model_after_load"])
        self.assertTrue(payload["dgpo"]["adaptive_omnifold"]["recalibration"]["bootstrap_on_start"])
        self.assertNotIn("restart_from_bootstrap", payload["dgpo"])
        self.assertIn("v26_rawplateau5_patience5_returnbest_notopology_d128_l1_velocity_mse_trust10_globalbestdecay09_warm20from10_dropout15", payload["options"]["Training"]["model_checkpoint_save_path"])

    def _config(self):
        import yaml
        path = Path(__file__).resolve().parents[4] / "config/dgpo_omnifold_ztautau_1pct_epoch_refit_nohardtrust_ablation.yaml"
        payload = yaml.safe_load(path.read_text())
        return payload, resolve_adaptive_config(payload["dgpo"])

    def test_ablation_config_keeps_kl_without_boundary_or_rollback(self):
        payload, cfg = self._config()
        self.assertAlmostEqual(0.5 + cfg.residual_min_auc_gain, 0.55)
        recalibration = payload["dgpo"]["adaptive_omnifold"]["recalibration"]
        self.assertEqual(recalibration["head_dropout"], 0.35)
        self.assertEqual(recalibration["topology_dropout"], 0.35)
        self.assertEqual(cfg.audit_fit["head_dropout"], 0.25)
        self.assertEqual(cfg.audit_fit["topology_dropout"], 0.25)
        trust = payload["dgpo"]["reference_trust"]
        self.assertTrue(trust["enabled"])
        self.assertEqual(trust["coefficient"], 1.0)
        self.assertEqual(trust["objective"], "vp_path_kl")
        self.assertFalse(trust["adaptive_boundary"]["enabled"])
        self.assertFalse(cfg.classifier_trust_enabled)
        self.assertFalse(cfg.raw_rollback_to_best_on_plateau)
        self.assertTrue(cfg.single_pool_train_validation)
        self.assertIsNone(cfg.probe_max_events)
        self.assertIsNone(cfg.pool_events)
        self.assertIsNone(cfg.refit_score_events)
        self.assertEqual(payload["dgpo"]["adaptive_omnifold"]["recalibration"]["warm_start_iterations"], [1, 2])
        self.assertEqual(payload["dgpo"]["checkpoint_every_n_epochs"], 1)
        self.assertIn("v19_epoch_refit_nohardtrust", payload["options"]["Training"]["model_checkpoint_save_path"])

    def test_each_epoch_refits_even_when_raw_improves_and_survives_resume(self):
        _, cfg = self._config()
        state = AdaptiveOmniFoldState(last_recalibration_epoch=-1)
        state.install(baseline_auc_gap=0.0, cfg=cfg, epoch=-1, round_id=1)
        refits = []
        for step in range(1, 41):
            epoch, end = (step - 1) // 10, step % 10 == 0
            if not adaptive_module.should_probe_training_boundary(
                state, cfg=cfg, epoch=epoch, global_step=step, epoch_end=end,
            ):
                continue
            self.assertTrue(end)
            raw_trigger, _ = update_raw_plateau_controller(
                state, {"raw_auc_gap": 0.1, "raw_audit_saturated": 1},
                cfg=cfg, epoch=epoch, global_step=step,
            )
            self.assertFalse(raw_trigger)
            due, _ = adaptive_module.scheduled_raw_refit_due(state, cfg=cfg, epoch=epoch)
            self.assertTrue(due)
            refits.append(step)
            state.last_recalibration_epoch = epoch
            state.install(baseline_auc_gap=0.0, cfg=cfg, epoch=epoch, round_id=state.reward_round_id + 1)
            state = AdaptiveOmniFoldState.from_dict(state.to_dict())
            due, _ = adaptive_module.scheduled_raw_refit_due(state, cfg=cfg, epoch=epoch)
            self.assertFalse(due)
        self.assertEqual(refits, [10, 20, 30, 40])

    def test_baseline_and_opt_out_do_not_schedule_refit(self):
        _, cfg = self._config()
        state = AdaptiveOmniFoldState(last_recalibration_epoch=-1)
        for epoch, baseline, active_cfg in (
            (-1, False, cfg), (4, True, cfg),
            (99, False, replace(cfg, max_reward_age_epochs=None)),
            (0, False, replace(cfg, retrain_cooldown_epochs=3)),
        ):
            due, _ = adaptive_module.scheduled_raw_refit_due(
                state, cfg=active_cfg, epoch=epoch, baseline_only=baseline,
            )
            self.assertFalse(due)

    def test_rejected_refit_attempt_enforces_age_retry_cooldown(self):
        _, cfg = self._config()
        cfg = replace(cfg, max_reward_age_epochs=1, retrain_cooldown_epochs=3)
        state = AdaptiveOmniFoldState(
            installed_at_epoch=0,
            last_recalibration_epoch=0,
            last_recalibration_attempt_epoch=10,
        )
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())

        due, diagnostics = adaptive_module.scheduled_raw_refit_due(
            restored,
            cfg=cfg,
            epoch=11,
        )
        self.assertFalse(due)
        self.assertEqual(diagnostics["staleness/age_refit_due"], 1.0)
        self.assertEqual(
            diagnostics["staleness/age_refit_cooldown_active"],
            1.0,
        )
        self.assertEqual(
            diagnostics["staleness/last_refit_attempt_epoch"],
            10.0,
        )

        due, diagnostics = adaptive_module.scheduled_raw_refit_due(
            restored,
            cfg=cfg,
            epoch=13,
        )
        self.assertTrue(due)
        self.assertEqual(
            diagnostics["staleness/age_refit_cooldown_active"],
            0.0,
        )


class TestGlobalRawRollback(unittest.TestCase):
    def test_warmup_intervals_do_not_consume_patience_across_resume(self):
        cfg = replace(self._cfg(), policy_warmup_steps=20,
                      raw_pause_patience_during_warmup=True, required_consecutive_epochs=6)
        state = AdaptiveOmniFoldState(reward_round_id=5, raw_global_initialized=True,
                                      raw_best_auc_gap=.18, raw_global_refit_pending=True)
        adaptive_module.start_policy_round_warmup(state, cfg=cfg)
        for update in range(1, 51):
            adaptive_module.advance_policy_round_warmup(state, cfg=cfg, accepted=True)
            if update % 5:
                continue
            fired, metrics = update_raw_plateau_controller(
                state, {"raw_auc_gap": .19, "raw_audit_saturated": 1.},
                cfg=cfg, epoch=update//10, global_step=235+update)
            self.assertEqual(metrics["staleness/patience_paused_for_warmup"], float(update <= 20))
            self.assertEqual(state.raw_no_improvement_streak, max(0, (update-20)//5))
            self.assertEqual(fired, update == 50)
            state = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(state.raw_global_failed_rounds, 1)
        self.assertFalse(state.raw_global_stop_requested)

    def test_warmup_still_accepts_improvement_and_legacy_resume_preserves_finished_warmup(self):
        cfg = replace(self._cfg(), policy_warmup_steps=20, raw_pause_patience_during_warmup=True)
        state = AdaptiveOmniFoldState(reward_round_id=4, raw_global_initialized=True,
                                      raw_best_auc_gap=.18)
        adaptive_module.start_policy_round_warmup(state, cfg=cfg)
        fired, metrics = update_raw_plateau_controller(
            state, {"raw_auc_gap": .17, "raw_audit_saturated": 1.}, cfg=cfg,
            epoch=1, global_step=10, global_confirmation=True)
        self.assertFalse(fired)
        self.assertEqual(metrics["staleness/global_best/improved"], 1.)
        self.assertEqual(state.raw_best_global_step, 10)
        legacy = state.to_dict()
        legacy.pop("raw_patience_warmup_updates_seen")
        legacy["policy_warmup_completed_updates"] = 20
        restored = AdaptiveOmniFoldState.from_dict(legacy)
        _, metrics = update_raw_plateau_controller(
            restored, {"raw_auc_gap": .18, "raw_audit_saturated": 1.},
            cfg=cfg, epoch=1, global_step=15)
        self.assertEqual(metrics["staleness/patience_paused_for_warmup"], 0.)
        self.assertEqual(restored.raw_no_improvement_streak, 1)
        restored.reward_round_id += 1
        adaptive_module.start_policy_round_warmup(restored, cfg=cfg)
        self.assertEqual(restored.raw_patience_warmup_updates_seen, 0)

    def test_confirmation_budget_preserves_raw_protocol_and_enforces_both_minima(self):
        import yaml
        from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
        root = Path(__file__).resolve().parents[4]
        payload = yaml.safe_load((root / "config/dgpo_omnifold_ztautau_10pct_resume_v26.yaml").read_text())
        cfg = resolve_adaptive_config(payload["dgpo"])
        self.assertFalse(cfg.raw_global_confirm_candidates)
        self.assertEqual(cfg.raw_global_confirmation_fit, {})
        # Exercise the optional confirmation budget independently of the live
        # resume configuration, which now disables candidate confirmation.
        cfg = replace(cfg, raw_global_confirm_candidates=True,
                      raw_global_confirmation_warm_start=True,
                      raw_global_confirmation_fit={"min_steps": 512, "min_epochs": 25,
                                                   "enforce_min_epochs": True})
        legacy = replace(cfg, raw_global_confirmation_fit={}, raw_global_confirmation_warm_start=False,
                         raw_pause_patience_during_warmup=False)
        self.assertEqual(adaptive_audit_protocol_signature(cfg), adaptive_audit_protocol_signature(legacy))
        self.assertTrue(cfg.raw_pause_patience_during_warmup)
        self.assertTrue(cfg.raw_global_confirmation_warm_start)
        block = {**cfg.audit_fit, **cfg.raw_global_confirmation_fit}
        # Current pool: 6 updates/epoch. Larger pool: epoch floor must take over.
        for n_train, expected in ((199669, 512), (1000000, 750)):
            fit = build_fit_config(block, n_train=n_train, n_validation=50000)
            self.assertEqual(fit.min_steps, expected)
        raw_fit = build_fit_config(cfg.audit_fit, n_train=199669, n_validation=50331)
        self.assertEqual(raw_fit.min_steps, 0)

    def _cfg(self):
        return replace(_config(monitor_mode="raw_plateau_refit", raw_audit_enabled=True,
                               require_audit_saturation=True, required_consecutive_epochs=5),
                       staleness_every_n_steps=5, raw_best_scope="global",
                       raw_global_confirm_candidates=True, raw_improvement_min_delta=.001,
                       raw_rollback_to_best_on_plateau=True)

    def _check(self, state, step, gap, *, confirm=None, saturated=1.):
        return update_raw_plateau_controller(
            state, {"raw_auc_gap": gap, "raw_audit_saturated": saturated}, cfg=self._cfg(),
            epoch=(step - 1)//10, global_step=step,
            checkpoint_path=f"/snap/step{step}.ckpt", global_confirmation=confirm,
        )

    def test_migration_uses_all_rounds_and_resume_retains_best(self):
        state = AdaptiveOmniFoldState(raw_best_auc_gap=.2, raw_best_epoch=4, raw_best_global_step=50,
                                    raw_best_checkpoint="/snap/step50.ckpt", trust_best_global_auc_gap=.1)
        state.probe_history = [
            {"raw_auc_gap": .1, "raw_audit_saturated": 1, "epoch": 1, "global_step": 15,
             "checkpoint_next_epoch": 1, "reward_round_id": 1},
            {"raw_auc_gap": .2, "raw_audit_saturated": 1, "epoch": 4, "global_step": 50,
             "checkpoint_next_epoch": 5, "reward_round_id": 2},
            {"raw_auc_gap": .01, "raw_audit_saturated": 0, "epoch": 5, "global_step": 55},
        ]
        adaptive_module.initialize_global_raw_best(state)
        self.assertEqual((state.raw_best_auc_gap, state.raw_best_global_step, state.raw_best_next_epoch), (.1, 15, 1))
        self.assertEqual(state.raw_best_checkpoint, "")
        state.install(baseline_auc_gap=.3, cfg=self._cfg(), epoch=5, round_id=3)
        self.assertEqual(state.raw_best_global_step, 15)
        self.assertEqual(state.raw_best_auc_gap, .1)
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        adaptive_module.initialize_global_raw_best(restored)
        self.assertEqual(restored.raw_best_global_step, 15)

    def test_missing_global_history_fails_not_runner_up(self):
        state = AdaptiveOmniFoldState(trust_best_global_auc_gap=.01, probe_history=[
            {"raw_auc_gap": .1, "raw_audit_saturated": 1, "epoch": 0, "global_step": 10},
        ])
        with self.assertRaisesRegex(ValueError, "missing the inherited"):
            adaptive_module.initialize_global_raw_best(state)

    def test_round_refit_does_not_reset_global_and_stops_after_two_failed_rounds(self):
        state = AdaptiveOmniFoldState()
        self._check(state, 5, .1)
        for step in (10, 15, 20, 25):
            fired, _ = self._check(state, step, .12)
            self.assertFalse(fired)
        fired, _ = self._check(state, 30, .12)
        self.assertTrue(fired)
        self.assertEqual(state.raw_global_failed_rounds, 0)
        for round_id in (1, 2):
            state.install(baseline_auc_gap=.2, cfg=self._cfg(), epoch=3, round_id=round_id)
            state.raw_global_refit_pending = True
            for offset in range(1, 6):
                fired, d = self._check(state, 30 + (round_id - 1)*25 + offset*5, .11)
            self.assertEqual(state.raw_best_global_step, 5)
            self.assertEqual(state.raw_global_failed_rounds, round_id)
            self.assertEqual(fired, round_id < 2)
        self.assertTrue(state.raw_global_stop_requested)
        self.assertEqual(d["staleness/decision"], "raw_global_stagnation_stop")
        self.assertTrue(AdaptiveOmniFoldState.from_dict(state.to_dict()).raw_global_stop_requested)

    def test_confirmation_effect_floor_invalid_and_duplicate_checks(self):
        state = AdaptiveOmniFoldState()
        self._check(state, 5, .1)
        self._check(state, 10, .0995, confirm=True)  # smaller than effect-size floor
        self.assertEqual(state.raw_best_global_step, 5)
        self._check(state, 15, .09, confirm=False)
        self.assertEqual(state.raw_best_global_step, 5)
        self._check(state, 20, .09, confirm=None)  # incomplete comparison resets streak
        self.assertEqual(state.raw_no_improvement_streak, 0)
        state.raw_global_failed_rounds = 1
        state.raw_global_refit_pending = True
        _, d = self._check(state, 25, .09, confirm=True)
        self.assertEqual(state.raw_best_global_step, 25)
        self.assertEqual(state.raw_global_failed_rounds, 0)
        self.assertFalse(state.raw_global_refit_pending)
        self.assertEqual(d["staleness/global_best/improved"], 1.)
        with self.assertRaisesRegex(ValueError, "strictly newer"):
            self._check(state, 25, .08, confirm=True)


class TestStepStaleness(unittest.TestCase):
    schedule = ((0, 6), (100, 10), (300, 16), (600, 24))

    def _cfg(self):
        return replace(_config(monitor_mode="raw_plateau_refit", raw_audit_enabled=True,
                               require_audit_saturation=True, required_consecutive_epochs=5),
                       staleness_every_n_steps=5)

    def test_schedule_parser_validation_and_legacy_default(self):
        parse = adaptive_module._raw_patience_schedule
        valid = [{'start_step': s, 'required_consecutive_checks': p}
                 for s, p in self.schedule]
        self.assertEqual(parse(valid), self.schedule)
        self.assertEqual(parse([SimpleNamespace(**entry) for entry in valid]), parse(valid))
        self.assertEqual(parse(None), ())
        self.assertEqual(parse([]), ())
        invalid = [
            {}, '4,6,8', [True], [{'start_step': 0}],
            [{'start_step': True, 'required_consecutive_checks': 4}],
            [{'start_step': 0, 'required_consecutive_checks': False}],
            [{'start_step': 0, 'required_consecutive_checks': 0}],
            [{'start_step': 0.0, 'required_consecutive_checks': 4}],
            [{'start_step': 5, 'required_consecutive_checks': 4}],
            [valid[0], valid[0]], [valid[0], valid[2], valid[1]],
            [valid[0], {'start_step': 100, 'required_consecutive_checks': 3}],
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'patience_schedule'):
                parse(value)
        self.assertEqual(adaptive_module.raw_staleness_patience(self._cfg(), global_step=-1), 5)
        with self.assertRaisesRegex(ValueError, 'requires step-based'):
            resolve_adaptive_config({'adaptive_omnifold': {'trigger': {'patience_schedule': valid}}})

    def test_patience_stages_and_trigger_counts(self):
        cfg = replace(self._cfg(), raw_patience_schedule=self.schedule)
        for step, expected in ((0, 6), (99, 6), (100, 10), (299, 10), (300, 16), (599, 16), (600, 24), (1500, 24)):
            self.assertEqual(adaptive_module.raw_staleness_patience(cfg, global_step=step), expected)
        with self.assertRaisesRegex(ValueError, 'global_step'):
            adaptive_module.raw_staleness_patience(cfg, global_step=-1)
        for start, patience in self.schedule:
            state = AdaptiveOmniFoldState(raw_best_auc_gap=.1)
            for missed in range(1, patience + 1):
                step = start + missed * 5
                fired, metrics = update_raw_plateau_controller(
                    state, {'raw_auc_gap': .12, 'raw_audit_saturated': 1.},
                    cfg=cfg, epoch=step//10, global_step=step,
                )
                self.assertEqual(fired, missed == patience)
                self.assertEqual(metrics['staleness/raw_no_improvement_patience'], patience)
                self.assertEqual(state.probe_history[-1]['raw_no_improvement_patience'], patience)
                state = AdaptiveOmniFoldState.from_dict(state.to_dict())

    def test_schedule_boundary_resume_and_round_install_preserve_clock(self):
        cfg = replace(self._cfg(), raw_best_scope='global',
                      raw_patience_schedule=self.schedule)
        state = AdaptiveOmniFoldState(raw_best_auc_gap=.1, raw_best_global_step=5,
                                      raw_best_epoch=0, raw_global_initialized=True)
        for step in range(75, 121, 5):
            fired, _ = update_raw_plateau_controller(
                state, {'raw_auc_gap': .12, 'raw_audit_saturated': 1.},
                cfg=cfg, epoch=step//10, global_step=step,
            )
            self.assertEqual(fired, step == 120)  # step 100 already uses patience 10
            state = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(state.raw_no_improvement_streak, 10)
        # Installing a reward after rewinding only policy weights resets the
        # streak, not the training clock or the global-best checkpoint metadata.
        state.install(baseline_auc_gap=.02, cfg=cfg, epoch=60, round_id=2)
        self.assertEqual(state.raw_no_improvement_streak, 0)
        self.assertEqual(state.raw_best_global_step, 5)
        _, metrics = update_raw_plateau_controller(
            state, {'raw_auc_gap': .12, 'raw_audit_saturated': 1.},
            cfg=cfg, epoch=60, global_step=605,
        )
        self.assertEqual(metrics['staleness/raw_no_improvement_patience'], 24)
        self.assertEqual(state.raw_no_improvement_streak, 1)

    def test_schedule_keeps_warmup_eligibility_and_improvement_rules(self):
        cfg = replace(self._cfg(), raw_patience_schedule=self.schedule,
                      policy_warmup_steps=10, raw_pause_patience_during_warmup=True)
        state = AdaptiveOmniFoldState(raw_best_auc_gap=.1, reward_round_id=2)
        adaptive_module.start_policy_round_warmup(state, cfg=cfg)
        fired, metrics = update_raw_plateau_controller(
            state, {'raw_auc_gap': .12, 'raw_audit_saturated': 1.},
            cfg=cfg, epoch=60, global_step=605,
        )
        self.assertFalse(fired)
        self.assertEqual(metrics['staleness/patience_paused_for_warmup'], 1.)
        self.assertEqual(state.raw_no_improvement_streak, 0)
        for gap, saturated in ((.09, 1.), (.12, 0.), (float('nan'), 1.)):
            state = AdaptiveOmniFoldState(raw_best_auc_gap=.1, raw_no_improvement_streak=23)
            fired, _ = update_raw_plateau_controller(
                state, {'raw_auc_gap': gap, 'raw_audit_saturated': saturated},
                cfg=cfg, epoch=60, global_step=605,
            )
            self.assertFalse(fired)
            self.assertEqual(state.raw_no_improvement_streak, 0)

    def test_five_checks_trigger_at_step_25_and_resume_does_not_duplicate(self):
        cfg = self._cfg()
        state = AdaptiveOmniFoldState(raw_best_auc_gap=0.1)
        checks, triggers = [], []
        for step in range(1, 26):
            epoch, end = (step - 1) // 10, step % 10 == 0
            if not adaptive_module.should_probe_training_boundary(
                state, cfg=cfg, epoch=epoch, global_step=step, epoch_end=end,
            ):
                continue
            checks.append(step)
            fired, _ = update_raw_plateau_controller(
                state, {"raw_auc_gap": 0.12, "raw_audit_saturated": 1},
                cfg=cfg, epoch=epoch, global_step=step,
                checkpoint_next_epoch=epoch + int(end),
            )
            if fired:
                triggers.append(step)
            state = AdaptiveOmniFoldState.from_dict(state.to_dict())
            self.assertFalse(adaptive_module.should_probe_training_boundary(
                state, cfg=cfg, epoch=epoch, global_step=step, epoch_end=end,
            ))
        self.assertEqual(checks, [5, 10, 15, 20, 25])
        self.assertEqual(triggers, [25])
        self.assertEqual(state.raw_no_improvement_streak, 5)
        self.assertEqual(state.probe_history[-1]["checkpoint_next_epoch"], 2)

    def test_improvement_or_invalid_check_breaks_streak(self):
        cfg = self._cfg()
        for gap, saturated in ((0.09, 1), (0.2, 0), (float("nan"), 1)):
            with self.subTest(gap=gap, saturated=saturated):
                state = AdaptiveOmniFoldState(raw_best_auc_gap=0.1, raw_no_improvement_streak=4)
                fired, _ = update_raw_plateau_controller(
                    state, {"raw_auc_gap": gap, "raw_audit_saturated": saturated},
                    cfg=cfg, epoch=2, global_step=25,
                )
                self.assertFalse(fired)
                self.assertEqual(state.raw_no_improvement_streak, 0)

    def test_legacy_epoch_mode_has_no_mid_epoch_checks(self):
        cfg = replace(self._cfg(), staleness_every_n_steps=None, staleness_every_n_epochs=2)
        state = AdaptiveOmniFoldState()
        for epoch, end, expected in ((0, True, False), (1, False, False), (1, True, True)):
            self.assertEqual(adaptive_module.should_probe_training_boundary(
                state, cfg=cfg, epoch=epoch, global_step=20, epoch_end=end,
            ), expected)

    def test_step_history_keeps_all_300_checks_of_150_epochs(self):
        cfg = self._cfg()
        state = AdaptiveOmniFoldState()
        for index in range(300):
            update_raw_plateau_controller(
                state, {"raw_auc_gap": 0.4 - index * 0.001, "raw_audit_saturated": 1},
                cfg=cfg, epoch=index // 2, global_step=(index + 1) * 5,
            )
        self.assertEqual(len(state.probe_history), 300)

    def test_cold_bootstrap_saves_pending_raw_baseline_without_refitting_reward(self):
        cfg = self._cfg()
        state = AdaptiveOmniFoldState(reward_round_id=1, trust_radius_decay_step=0,
                                     trust_current_delta=0.1)
        with self.assertRaisesRegex(ValueError, "calibrated"):
            state.mark_bootstrap_complete(cfg=cfg)
        state.install(baseline_auc_gap=0.0, cfg=cfg, epoch=-1, round_id=1)
        state.mark_bootstrap_complete(cfg=cfg)
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertTrue(restored.raw_monitor_baseline_pending)
        self.assertTrue(restored.resume_refit_once_completed)
        self.assertEqual(restored.reward_round_id, 1)
        self.assertEqual(restored.trust_radius_decay_step, 0)
        self.assertEqual(restored.trust_current_delta, 0.1)
        fired, _ = update_raw_plateau_controller(
            restored, {"raw_auc_gap": 0.3, "raw_audit_saturated": 1},
            cfg=cfg, epoch=-1, global_step=0,
        )
        self.assertFalse(fired)
        self.assertEqual(restored.raw_best_auc_gap, 0.3)
        self.assertEqual(restored.last_staleness_step, 0)

    def test_legacy_weighted_bootstrap_does_not_request_raw_baseline(self):
        cfg = _config()
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=1)
        state.mark_bootstrap_complete(cfg=cfg)
        self.assertFalse(state.raw_monitor_baseline_pending)


class TestAdaptiveConfig(unittest.TestCase):
    def test_baseline_probe_uses_the_active_controller_metric(self) -> None:
        probe = {"weighted_auc_gap": 0.01, "raw_auc_gap": 0.20}
        self.assertEqual(baseline_probe_auc_gap(probe, cfg=_config()), 0.01)
        self.assertEqual(
            baseline_probe_auc_gap(
                probe,
                cfg=_config(
                    monitor_mode="raw_plateau_refit",
                    raw_audit_enabled=True,
                    acceptance_audit_enabled=False,
                ),
            ),
            0.20,
        )

    def test_monitor_readiness_requires_safe_raw_protocol_and_changes_signature(self):
        import copy
        import yaml
        path = Path(__file__).resolve().parents[4] / "config/dgpo_omnifold_ztautau_10pct_arch_fourier_conditioning.yaml"
        payload = yaml.safe_load(path.read_text())["dgpo"]
        cfg = resolve_adaptive_config(payload)
        signature = adaptive_audit_protocol_signature(cfg)
        changed = copy.deepcopy(payload)
        changed["adaptive_omnifold"]["audit_fit"]["training_readiness"]["cold_start_min_epochs"] = 101
        self.assertNotEqual(signature, adaptive_audit_protocol_signature(resolve_adaptive_config(changed)))
        for section, field in (("trigger", "warm_start_classifier"), ("trigger", "require_audit_saturation")):
            invalid = copy.deepcopy(payload)
            invalid["adaptive_omnifold"][section][field] = False
            with self.assertRaises(ValueError):
                resolve_adaptive_config(invalid)

    def test_selective_warm_start_config(self) -> None:
        self.assertEqual(_config().warm_start_iterations, ())
        cfg = _config(warm_start_iterations=(1, 2))
        self.assertEqual(cfg.warm_start_iterations, (1, 2))
        self.assertEqual(cfg.fit["weight_decay"], 0.0005)
        for invalid in ((0,), (5,), (1, 1), (True,), (1.5,)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _config(warm_start_iterations=invalid)

    def test_k1_is_independent_of_dgpo_group_size(self) -> None:
        cfg = _config()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.monitor_mode, "weighted_and_raw")
        self.assertEqual(cfg.candidates_per_event, 1)
        self.assertEqual(cfg.staleness_every_n_epochs, 2)
        self.assertEqual(cfg.pool_generation_batch_size, 1024)
        self.assertEqual(cfg.pool_selection_seed, 73)
        self.assertEqual(cfg.probe_max_events, 100)
        self.assertEqual(cfg.refit_score_events, 200)
        self.assertEqual(cfg.audit_fit["batch_size"], 64)
        self.assertEqual(cfg.audit_fit["validation_patience_epochs"], 5.0)
        self.assertFalse(cfg.require_audit_saturation)
        self.assertEqual(cfg.retrain_auc_margin, 0.01)
        self.assertFalse(cfg.raw_audit_enabled)
        self.assertEqual(cfg.max_reward_age_epochs, 20)
        self.assertTrue(cfg.acceptance_audit_enabled)
        self.assertEqual(cfg.acceptance_max_balanced_accuracy, 0.51)
        self.assertEqual(cfg.crossfit_folds, 2)
        self.assertEqual(cfg.crossfit_repeats, 1)
        self.assertEqual(_config(crossfit_repeats=5).crossfit_repeats, 5)
        with self.assertRaises(ValueError):
            _config(crossfit_repeats=0)
        self.assertAlmostEqual(cfg.residual_min_auc_gain, 1.0e-3)
        self.assertFalse(cfg.bootstrap_on_start)
        self.assertTrue(cfg.bootstrap_fail_closed)
        self.assertTrue(cfg.refit_once_on_resume)
        self.assertFalse(cfg.refit_once_fail_closed)
        self.assertEqual(cfg.refit_once_id, "test_regularized_v1")
        self.assertIsNone(cfg.pool_data_parquet_dir)
        self.assertFalse(cfg.trust_boundary_enabled)

    def test_omnifold_can_use_an_independent_training_directory(self) -> None:
        cfg = _config(pool_data_parquet_dir=" /data/full-train ")
        self.assertEqual(cfg.pool_data_parquet_dir, "/data/full-train")

    def test_adaptive_trust_config_is_opt_in(self) -> None:
        cfg = _config(trust_boundary_enabled=True)
        self.assertTrue(cfg.trust_boundary_enabled)
        self.assertAlmostEqual(cfg.trust_delta_max, 1.0e-3)
        self.assertAlmostEqual(cfg.trust_delta_floor, 5.0e-6)
        self.assertAlmostEqual(cfg.trust_warning_fraction, 0.8)
        self.assertAlmostEqual(cfg.trust_adaptive_power, 2.0)
        self.assertTrue(cfg.trust_reset_adam_first_moment)
        self.assertEqual(cfg.trust_enforcement, "post_step_backtracking")
        self.assertAlmostEqual(cfg.trust_backtrack_factor, 0.5)
        self.assertEqual(cfg.trust_max_backtracks, 16)
        self.assertEqual(cfg.trust_probe_events_per_rank, 64)
        self.assertTrue(cfg.trust_fixed_probe_per_reward_round)
        self.assertAlmostEqual(cfg.trust_interior_fraction, 0.8)
        self.assertFalse(cfg.trust_reset_adam_first_moment_on_zero_step)
        self.assertFalse(cfg.trust_cross_round_nonexpanding)
        self.assertFalse(cfg.trust_policy_lr_scaling_enabled)
        self.assertAlmostEqual(cfg.trust_policy_lr_scale_floor, 0.1)
        self.assertFalse(cfg.trust_empirical_radius_enabled)

    def test_reward_install_momentum_reset_is_independent_of_boundary(self) -> None:
        default_cfg = _config(trust_boundary_enabled=False)
        enabled_cfg = _config(
            trust_boundary_enabled=False,
            reset_adam_first_moment_on_install=True,
        )
        self.assertFalse(default_cfg.trust_reset_adam_first_moment)
        self.assertTrue(enabled_cfg.trust_reset_adam_first_moment)

    def test_raw_only_monitor_requires_log_only_and_raw_audit(self) -> None:
        cfg = _config(
            log_only=True,
            monitor_mode="raw_only",
            raw_audit_enabled=True,
        )
        self.assertEqual(cfg.monitor_mode, "raw_only")

        with self.assertRaisesRegex(ValueError, "raw_audit_enabled"):
            _config(log_only=True, monitor_mode="raw_only")
        with self.assertRaisesRegex(ValueError, "log_only=true"):
            _config(monitor_mode="raw_only", raw_audit_enabled=True)

    def test_raw_only_monitor_never_bypasses_pair_certification(self) -> None:
        cfg = _config(
            log_only=True,
            monitor_mode="raw_only",
            raw_audit_enabled=True,
        )
        state = AdaptiveOmniFoldState()
        self.assertFalse(should_run_raw_only_monitor(state, cfg=cfg))

        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=54, round_id=1)
        before = repr(state.to_dict())
        self.assertTrue(should_run_raw_only_monitor(state, cfg=cfg))
        self.assertFalse(
            should_run_raw_only_monitor(state, cfg=cfg, baseline_only=True)
        )
        self.assertFalse(
            should_run_raw_only_monitor(state, cfg=cfg, force_refit=True)
        )
        self.assertFalse(should_skip_incumbent_probe(cfg=cfg))
        self.assertTrue(should_skip_incumbent_probe(cfg=cfg, force_refit=True))
        self.assertFalse(
            should_skip_incumbent_probe(cfg=_config(), force_refit=True)
        )
        self.assertEqual(repr(state.to_dict()), before)

    def test_raw_plateau_refit_and_classifier_trust_config(self) -> None:
        cfg = _config(
            monitor_mode="raw_plateau_refit",
            raw_audit_enabled=True,
            classifier_trust_enabled=True,
            classifier_trust_unsafe_early_stop_enabled=True,
            required_consecutive_epochs=2,
            raw_rollback_to_best_on_plateau=True,
            fixed_audit_panel=True,
            require_audit_saturation=True,
        )
        self.assertFalse(cfg.log_only)
        self.assertEqual(cfg.monitor_mode, "raw_plateau_refit")
        self.assertTrue(cfg.classifier_trust_enabled)
        self.assertEqual(cfg.classifier_trust_every_n_epochs, 2)
        self.assertEqual(cfg.classifier_trust_probe_max_events, 100)
        self.assertEqual(
            cfg.classifier_trust_audit_fit["validation_patience_epochs"],
            5.0,
        )
        self.assertTrue(cfg.classifier_trust_unsafe_early_stop_enabled)
        self.assertAlmostEqual(
            cfg.classifier_trust_unsafe_early_stop_confidence_z,
            1.96,
        )
        self.assertEqual(
            cfg.classifier_trust_unsafe_early_stop_required_consecutive_validations,
            2,
        )
        self.assertEqual(cfg.required_consecutive_epochs, 2)
        self.assertTrue(cfg.raw_rollback_to_best_on_plateau)
        self.assertAlmostEqual(
            cfg.classifier_trust_max_balanced_accuracy,
            0.525,
        )
        with self.assertRaisesRegex(ValueError, "log_only=false"):
            _config(
                log_only=True,
                monitor_mode="raw_plateau_refit",
                raw_audit_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "must be a multiple"):
            _config(
                monitor_mode="raw_plateau_refit",
                raw_audit_enabled=True,
                classifier_trust_enabled=True,
                classifier_trust_every_n_epochs=3,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            _config(
                monitor_mode="raw_plateau_refit",
                raw_audit_enabled=True,
                classifier_trust_enabled=True,
                classifier_trust_probe_max_events=101,
            )
        with self.assertRaisesRegex(ValueError, "fixed_audit_panel=true"):
            _config(
                monitor_mode="raw_plateau_refit",
                raw_audit_enabled=True,
                raw_rollback_to_best_on_plateau=True,
                fixed_audit_panel=False,
            )

    def test_fixed_vp_path_kl_boundary_is_supported(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_radius_mode="fixed",
            trust_distance="vp_path_kl",
        )
        self.assertEqual(cfg.trust_radius_mode, "fixed")
        self.assertEqual(cfg.trust_distance, "vp_path_kl")
        state = AdaptiveOmniFoldState()
        diagnostics = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.80,
            raw_auc_se=0.001,
        )
        self.assertAlmostEqual(state.trust_current_delta, cfg.trust_delta_max)
        self.assertEqual(diagnostics["reference_trust/radius_mode_fixed"], 1.0)

        state = AdaptiveOmniFoldState()
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.505,
            raw_auc_se=0.002,
        )
        self.assertAlmostEqual(state.trust_current_delta, cfg.trust_delta_max)

    def test_classifier_trust_uses_its_own_fit_overrides(self) -> None:
        cfg = _config(
            monitor_mode="raw_plateau_refit",
            raw_audit_enabled=True,
            classifier_trust_enabled=True,
            classifier_trust_validation_patience_epochs=3.0,
            classifier_trust_unsafe_early_stop_enabled=True,
        )
        pool = SimpleNamespace(candidates=torch.zeros(32, 1, 4))
        audit_result = {
            "weighted_auc_gap": 0.01,
            "judge_auc_weighted": 0.51,
            "audit_balanced_accuracy": 0.51,
            "audit_saturated": 1.0,
            "audit_unsafe_balanced_accuracy_lcb_reached": 0.0,
            "audit_validation_loss": 0.69,
            "audit_validation_auc": 0.51,
            "audit_validation_oriented_balanced_accuracy": 0.51,
            "audit_validation_balanced_accuracy_lcb": 0.50,
            "audit_validation_balanced_accuracy_lcb_standard_error": 0.01,
            "audit_validation_balanced_accuracy_lcb_streak": 0.0,
            "audit_fit_events": 25.0,
            "audit_test_events": 7.0,
            "audit_probe_events": 32.0,
            "audit_auc_null_se_approx": 0.01,
            "audit_weighted_auc_gap_z_approx": 1.0,
            "audit_weighted_auc_gap_pvalue_approx": 0.3,
        }
        with mock.patch.object(
            adaptive_module,
            "fit_fresh_audit",
            return_value=audit_result,
        ) as fit:
            adaptive_module.fit_reference_trust_audit(
                pool=pool,
                model_builder=object(),
                cfg=cfg,
                device=torch.device("cpu"),
                seed=7,
            )

        self.assertEqual(
            fit.call_args.kwargs["fit_overrides"],
            {"validation_patience_epochs": 3.0},
        )
        self.assertAlmostEqual(
            fit.call_args.kwargs["early_stop_balanced_accuracy_lcb"],
            0.525,
        )
        self.assertEqual(
            fit.call_args.kwargs[
                "early_stop_balanced_accuracy_required_consecutive"
            ],
            2,
        )

    def test_empirical_radius_is_independently_opt_in_and_locks_beta(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
        )
        self.assertTrue(cfg.trust_empirical_radius_enabled)
        self.assertTrue(cfg.trust_empirical_bidirectional)
        self.assertTrue(cfg.trust_empirical_allow_expansion)
        self.assertFalse(cfg.trust_refit_on_exhaustion)
        self.assertAlmostEqual(cfg.trust_empirical_safety_factor, 0.5)
        self.assertAlmostEqual(cfg.trust_empirical_expand_factor, 1.25)
        self.assertAlmostEqual(cfg.trust_empirical_shrink_factor, 0.5)
        self.assertAlmostEqual(cfg.trust_empirical_target_acceptance_rate, 0.5)
        self.assertAlmostEqual(cfg.trust_empirical_target_update_scale, 0.05)
        self.assertEqual(cfg.trust_empirical_safe_audits_required, 2)
        self.assertEqual(cfg.trust_empirical_attempt_window_steps, 50)
        self.assertEqual(cfg.trust_exhaustion_scale_window_steps, 50)
        self.assertEqual(cfg.trust_empirical_distance_window_steps, 10)
        self.assertEqual(cfg.trust_empirical_min_distance_samples, 5)
        self.assertFalse(cfg.trust_round_acceptance_enabled)
        self.assertEqual(cfg.trust_round_plateau_patience, 2)
        self.assertFalse(cfg.trust_trajectory_search_enabled)
        self.assertEqual(cfg.trust_failed_direction_patience, 3)
        self.assertFalse(cfg.trust_signed_direction_recovery_enabled)
        self.assertFalse(cfg.trust_extragradient_enabled)
        self.assertAlmostEqual(
            cfg.trust_extragradient_lookahead_scale,
            0.5,
        )
        self.assertEqual(cfg.topology_acceptance_repeats, 1)
        with self.assertRaisesRegex(ValueError, "requires dgpo.beta=1.0"):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                beta=0.75,
            )

    def test_trajectory_search_requires_raw_audit(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires trigger.raw_audit_enabled"):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                trust_trajectory_search_enabled=True,
                raw_audit_enabled=False,
            )

    def test_signed_direction_probe_resolves_scales_and_prerequisites(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_trajectory_search_enabled=True,
            trust_signed_direction_probe_enabled=True,
            trust_signed_direction_recovery_enabled=True,
            trust_signed_direction_probe_scales=(0.2, 0.5, 1.0),
            raw_audit_enabled=True,
        )
        self.assertTrue(cfg.trust_signed_direction_probe_enabled)
        self.assertTrue(cfg.trust_signed_direction_recovery_enabled)
        self.assertEqual(
            cfg.trust_signed_direction_probe_scales,
            (0.2, 0.5, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "requires trajectory_search_enabled"):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                trust_signed_direction_probe_enabled=True,
                raw_audit_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "requires the signed-direction probe"):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                trust_trajectory_search_enabled=True,
                trust_signed_direction_recovery_enabled=True,
                raw_audit_enabled=True,
            )

    def test_signed_direction_probe_requires_unique_scales_including_one(self) -> None:
        common = {
            "trust_boundary_enabled": True,
            "trust_empirical_radius_enabled": True,
            "trust_trajectory_search_enabled": True,
            "trust_signed_direction_probe_enabled": True,
            "raw_audit_enabled": True,
        }
        with self.assertRaisesRegex(ValueError, "must include 1.0"):
            _config(
                **common,
                trust_signed_direction_probe_scales=(0.25, 0.5),
            )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            _config(
                **common,
                trust_signed_direction_probe_scales=(0.25, 1.0, 1.0),
            )

    def test_lookahead_extragradient_is_opt_in_and_fail_closed(self) -> None:
        enabled = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_refit_on_exhaustion=True,
            trust_round_acceptance_enabled=True,
            trust_trajectory_search_enabled=True,
            trust_extragradient_enabled=True,
            trust_extragradient_lookahead_scale=0.5,
            raw_audit_enabled=True,
        )
        self.assertTrue(enabled.trust_extragradient_enabled)
        self.assertAlmostEqual(
            enabled.trust_extragradient_lookahead_scale,
            0.5,
        )
        with self.assertRaisesRegex(
            ValueError,
            "look-ahead extragradient requires",
        ):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                trust_extragradient_enabled=True,
                raw_audit_enabled=True,
            )
        with self.assertRaisesRegex(
            ValueError,
            "scale must lie in",
        ):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                trust_refit_on_exhaustion=True,
                trust_round_acceptance_enabled=True,
                trust_trajectory_search_enabled=True,
                trust_extragradient_enabled=True,
                trust_extragradient_lookahead_scale=0.0,
                raw_audit_enabled=True,
            )
        with self.assertRaisesRegex(
            ValueError,
            "reference_trust.coefficient=1.0",
        ):
            _config(
                trust_boundary_enabled=True,
                trust_empirical_radius_enabled=True,
                trust_refit_on_exhaustion=True,
                trust_round_acceptance_enabled=True,
                trust_trajectory_search_enabled=True,
                trust_extragradient_enabled=True,
                reference_trust_coefficient=0.5,
                raw_audit_enabled=True,
            )

    def test_fixed_trajectory_panel_changes_audit_protocol_signature(self) -> None:
        legacy = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            raw_audit_enabled=True,
        )
        fixed_panel = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_trajectory_search_enabled=True,
            raw_audit_enabled=True,
        )
        self.assertNotEqual(
            adaptive_audit_protocol_signature(legacy),
            adaptive_audit_protocol_signature(fixed_panel),
        )

    def test_explicit_fixed_panel_changes_audit_protocol_signature(self) -> None:
        changing = _config(raw_audit_enabled=True)
        fixed = _config(raw_audit_enabled=True, fixed_audit_panel=True)
        self.assertFalse(changing.fixed_audit_panel)
        self.assertTrue(fixed.fixed_audit_panel)
        self.assertNotEqual(
            adaptive_audit_protocol_signature(changing),
            adaptive_audit_protocol_signature(fixed),
        )

    def test_refit_rejects_more_than_one_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidates_per_event=1"):
            _config(candidates_per_event=2)

    def test_versioned_resume_refit_can_fail_closed(self) -> None:
        cfg = _config(refit_once_fail_closed=True)
        self.assertTrue(cfg.refit_once_fail_closed)

    def test_probe_schedule_includes_epoch_minus_one(self) -> None:
        self.assertTrue(should_probe_epoch(-1, 7))
        self.assertFalse(should_probe_epoch(0, 2))
        self.assertTrue(should_probe_epoch(1, 2))

    def test_reward_age_forces_refit_only_at_configured_limit(self) -> None:
        cfg = _config(max_reward_age_epochs=20)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=9, round_id=1)
        due, age = reward_refit_due_to_age(
            state,
            epoch=19,
            max_reward_age_epochs=cfg.max_reward_age_epochs,
        )
        self.assertFalse(due)
        self.assertEqual(age, 10)
        due, age = reward_refit_due_to_age(
            state,
            epoch=29,
            max_reward_age_epochs=cfg.max_reward_age_epochs,
        )
        self.assertTrue(due)
        self.assertEqual(age, 20)

    def test_reward_age_trigger_can_be_disabled(self) -> None:
        cfg = _config(max_reward_age_epochs=None)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        due, age = reward_refit_due_to_age(
            state,
            epoch=999,
            max_reward_age_epochs=cfg.max_reward_age_epochs,
        )
        self.assertFalse(due)
        self.assertEqual(age, 1000)

    def test_auc_power_reports_when_half_percent_gap_is_underpowered(self) -> None:
        diagnostics = adaptive_module._auc_power_diagnostics(
            observed_weighted_gap=0.005,
            audit_events=10_000,
            ess_fraction=1.0,
            retrain_auc_margin=0.005,
            alpha=0.05,
            target_power=0.80,
        )
        self.assertLess(diagnostics["audit_power_at_retrain_margin"], 0.30)
        self.assertGreater(diagnostics["audit_minimum_detectable_auc_gap"], 0.005)
        self.assertEqual(diagnostics["audit_power_sufficient"], 0.0)

    def test_auc_power_reaches_target_with_large_heldout_audit(self) -> None:
        diagnostics = adaptive_module._auc_power_diagnostics(
            observed_weighted_gap=0.005,
            audit_events=60_000,
            ess_fraction=1.0,
            retrain_auc_margin=0.005,
            alpha=0.05,
            target_power=0.80,
        )
        self.assertGreater(diagnostics["audit_power_at_retrain_margin"], 0.80)
        self.assertLess(diagnostics["audit_minimum_detectable_auc_gap"], 0.005)
        self.assertEqual(diagnostics["audit_power_sufficient"], 1.0)


class TestAdaptiveController(unittest.TestCase):
    def test_unready_monitor_cannot_seed_global_best_or_resume_history(self):
        from RL.DGPO_neutrino.model_utils import _completed_raw_monitor_records
        cfg = replace(_config(monitor_mode="raw_plateau_refit", raw_audit_enabled=True,
                              require_audit_saturation=True), raw_best_scope="global",
                      staleness_every_n_steps=5)
        cfg.audit_fit["training_readiness"] = {"cold_start_min_epochs": 100, "warm_start_min_epochs": 5}
        state = AdaptiveOmniFoldState(reward_round_id=1)
        fired, metrics = update_raw_plateau_controller(
            state, {"raw_auc_gap": .006, "raw_audit_saturated": 1., "raw_audit_training_ready": 0.},
            cfg=cfg, epoch=-1, global_step=0,
        )
        self.assertFalse(fired)
        self.assertFalse(math.isfinite(state.raw_best_auc_gap))
        self.assertEqual(metrics["staleness/decision"], "raw_monitor_training_unready")
        self.assertEqual(_completed_raw_monitor_records({"probe_history": state.probe_history}), [])
        _, metrics = update_raw_plateau_controller(
            state, {"raw_auc_gap": .2, "raw_audit_saturated": 1., "raw_audit_training_ready": 1.},
            cfg=cfg, epoch=0, global_step=5,
        )
        self.assertEqual(state.raw_best_auc_gap, .2)
        self.assertEqual(metrics["staleness/global_best/improved"], 1.)

    def test_raw_plateau_uses_best_so_far_and_two_consecutive_misses(self) -> None:
        cfg = _config(
            monitor_mode="raw_plateau_refit",
            raw_audit_enabled=True,
            required_consecutive_epochs=2,
            require_audit_saturation=True,
            raw_improvement_min_delta=0.0,
        )
        state = AdaptiveOmniFoldState(reward_round_id=3)

        fired, first = update_raw_plateau_controller(
            state,
            {"raw_auc_gap": 0.10, "raw_audit_saturated": 1.0},
            cfg=cfg,
            epoch=1,
            global_step=20,
            checkpoint_path="/tmp/raw-best-epoch1.ckpt",
        )
        self.assertFalse(fired)
        self.assertEqual(first["staleness/decision"], "raw_improved")
        self.assertAlmostEqual(state.raw_best_auc_gap, 0.10)
        self.assertEqual(state.raw_best_epoch, 1)
        self.assertEqual(state.raw_best_global_step, 20)
        self.assertEqual(
            state.raw_best_checkpoint,
            "/tmp/raw-best-epoch1.ckpt",
        )

        fired, _ = update_raw_plateau_controller(
            state,
            {"raw_auc_gap": 0.11, "raw_audit_saturated": 1.0},
            cfg=cfg,
            epoch=2,
        )
        self.assertFalse(fired)
        self.assertEqual(state.raw_no_improvement_streak, 1)

        fired, improved = update_raw_plateau_controller(
            state,
            {"raw_auc_gap": 0.09, "raw_audit_saturated": 1.0},
            cfg=cfg,
            epoch=3,
            global_step=40,
            checkpoint_path="/tmp/raw-best-epoch3.ckpt",
        )
        self.assertFalse(fired)
        self.assertEqual(improved["staleness/decision"], "raw_improved")
        self.assertEqual(state.raw_no_improvement_streak, 0)
        self.assertEqual(state.raw_best_epoch, 3)
        self.assertEqual(state.raw_best_global_step, 40)
        self.assertEqual(
            state.raw_best_checkpoint,
            "/tmp/raw-best-epoch3.ckpt",
        )

        fired, _ = update_raw_plateau_controller(
            state,
            {"raw_auc_gap": 0.091, "raw_audit_saturated": 1.0},
            cfg=cfg,
            epoch=4,
        )
        self.assertFalse(fired)
        fired, plateau = update_raw_plateau_controller(
            state,
            {"raw_auc_gap": 0.092, "raw_audit_saturated": 1.0},
            cfg=cfg,
            epoch=5,
        )
        self.assertTrue(fired)
        self.assertEqual(
            plateau["staleness/decision"],
            "raw_plateau_recalibrate",
        )
        self.assertEqual(plateau["staleness/raw_best_epoch"], 3.0)
        self.assertEqual(
            state.raw_best_checkpoint,
            "/tmp/raw-best-epoch3.ckpt",
        )

    def test_legacy_raw_best_epoch_is_recovered_from_probe_history(self) -> None:
        state = AdaptiveOmniFoldState.from_dict(
            {
                "reward_round_id": 4,
                "raw_best_auc_gap": 0.08,
                "probe_history": [
                    {
                        "epoch": 5.0,
                        "reward_round_id": 4.0,
                        "raw_auc_gap": 0.10,
                    },
                    {
                        "epoch": 6.0,
                        "reward_round_id": 4.0,
                        "raw_auc_gap": 0.08,
                    },
                    {
                        "epoch": 7.0,
                        "reward_round_id": 4.0,
                        "raw_auc_gap": 0.09,
                    },
                ],
            }
        )
        self.assertEqual(state.raw_best_epoch, 6)
        self.assertEqual(state.raw_best_global_step, -1)
        self.assertEqual(state.raw_best_checkpoint, "")

    def test_unsaturated_raw_probe_does_not_advance_plateau(self) -> None:
        cfg = _config(
            monitor_mode="raw_plateau_refit",
            raw_audit_enabled=True,
            required_consecutive_epochs=2,
            require_audit_saturation=True,
        )
        state = AdaptiveOmniFoldState(
            reward_round_id=2,
            raw_best_auc_gap=0.08,
        )
        fired, diagnostics = update_raw_plateau_controller(
            state,
            {"raw_auc_gap": 0.12, "raw_audit_saturated": 0.0},
            cfg=cfg,
            epoch=4,
        )
        self.assertFalse(fired)
        self.assertEqual(state.raw_no_improvement_streak, 0)
        self.assertEqual(
            diagnostics["staleness/decision"],
            "raw_monitor_unsaturated",
        )

    def test_classifier_trust_uses_oriented_balanced_accuracy_and_saturation(
        self,
    ) -> None:
        cfg = _config(
            monitor_mode="raw_plateau_refit",
            raw_audit_enabled=True,
            classifier_trust_enabled=True,
        )
        state = AdaptiveOmniFoldState(reward_round_id=7)

        fired, diagnostics = update_classifier_trust_controller(
            state,
            {
                "reference_trust_balanced_accuracy": 0.46,
                "reference_trust_test_events": 100_000,
                "reference_trust_saturated": 1.0,
            },
            cfg=cfg,
            epoch=8,
        )
        self.assertTrue(fired)
        self.assertAlmostEqual(
            diagnostics["classifier_trust/oriented_balanced_accuracy"],
            0.54,
        )
        self.assertAlmostEqual(
            diagnostics["classifier_trust/estimated_total_variation"],
            0.08,
        )

        state.classifier_trust_exceedance_streak = 0
        fired, diagnostics = update_classifier_trust_controller(
            state,
            {
                "reference_trust_balanced_accuracy": 0.54,
                "reference_trust_test_events": 100_000,
                "reference_trust_saturated": 0.0,
            },
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(fired)
        self.assertEqual(
            diagnostics["classifier_trust/decision_eligible"],
            0.0,
        )

        fired, diagnostics = update_classifier_trust_controller(
            state,
            {
                "reference_trust_balanced_accuracy": 0.54,
                "reference_trust_test_events": 100_000,
                "reference_trust_saturated": 0.0,
                "reference_trust_unsafe_balanced_accuracy_lcb_reached": 1.0,
            },
            cfg=cfg,
            epoch=10,
        )
        self.assertTrue(fired)
        self.assertEqual(
            diagnostics["classifier_trust/unsafe_early_stop"],
            1.0,
        )
        self.assertEqual(
            diagnostics["classifier_trust/decision_eligible"],
            1.0,
        )

    def test_extragradient_rejection_restores_failure_semantics(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_refit_on_exhaustion=True,
            trust_round_acceptance_enabled=True,
            trust_trajectory_search_enabled=True,
            trust_extragradient_enabled=True,
            trust_failed_direction_patience=3,
            raw_audit_enabled=True,
        )
        state = AdaptiveOmniFoldState(
            trust_current_delta=2.0e-5,
            trust_failed_direction_streak=1,
            trust_round_rollbacks=4,
            trust_distance_window=[1.0e-5],
            trust_step_acceptance_window=[1.0],
            trust_step_scale_window=[0.25],
        )

        metrics = record_extragradient_rejection(
            state,
            cfg=cfg,
            reason="final_classifier_gate_failed",
        )

        self.assertEqual(state.trust_failed_direction_streak, 2)
        self.assertEqual(state.trust_round_rollbacks, 5)
        self.assertEqual(state.recalibrations_rejected, 1)
        self.assertAlmostEqual(state.trust_current_delta, 1.0e-5)
        self.assertFalse(state.trust_round_stop_requested)
        self.assertEqual(state.trust_distance_window, [])
        self.assertEqual(
            metrics[
                "reference_trust/extragradient/rejection_reason_code"
            ],
            3.0,
        )

        terminal = record_extragradient_rejection(
            state,
            cfg=cfg,
            reason="trust_backtracking_failed",
        )
        self.assertTrue(state.trust_round_stop_requested)
        self.assertEqual(
            terminal["reference_trust/extragradient/stop_requested"],
            1.0,
        )

        lookahead = record_extragradient_rejection(
            AdaptiveOmniFoldState(trust_current_delta=2.0e-5),
            cfg=cfg,
            reason="lookahead_classifier_gate_failed",
        )
        self.assertEqual(
            lookahead[
                "reference_trust/extragradient/rejection_reason_code"
            ],
            4.0,
        )

    def test_reverse_only_recovery_requires_saturation_and_honors_patience(
        self,
    ) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_trajectory_search_enabled=True,
            trust_signed_direction_probe_enabled=True,
            trust_signed_direction_recovery_enabled=True,
            trust_failed_direction_patience=3,
            raw_audit_enabled=True,
        )
        state = AdaptiveOmniFoldState(trust_failed_direction_streak=1)
        probe = {
            "reference_trust/signed_probe/completed": 1.0,
            "reference_trust/signed_probe/all_raw_audits_saturated": 1.0,
            "reference_trust/signed_probe/decision_code": -1.0,
        }

        decision = evaluate_signed_direction_recovery(
            state,
            cfg=cfg,
            signed_probe=probe,
        )

        self.assertEqual(
            decision["reference_trust/signed_probe/recovery_triggered"],
            1.0,
        )
        self.assertEqual(
            decision["reference_trust/signed_probe/recovery_attempt"],
            2.0,
        )
        self.assertEqual(
            decision["reference_trust/signed_probe/recovery_stop_requested"],
            0.0,
        )
        self.assertEqual(state.trust_failed_direction_streak, 1)

        probe["reference_trust/signed_probe/all_raw_audits_saturated"] = 0.0
        unsaturated = evaluate_signed_direction_recovery(
            state,
            cfg=cfg,
            signed_probe=probe,
        )
        self.assertEqual(
            unsaturated["reference_trust/signed_probe/recovery_triggered"],
            0.0,
        )

        probe["reference_trust/signed_probe/all_raw_audits_saturated"] = 1.0
        state.trust_failed_direction_streak = 2
        terminal = evaluate_signed_direction_recovery(
            state,
            cfg=cfg,
            signed_probe=probe,
        )
        self.assertEqual(
            terminal["reference_trust/signed_probe/recovery_attempt"],
            3.0,
        )
        self.assertEqual(
            terminal["reference_trust/signed_probe/recovery_stop_requested"],
            1.0,
        )

    def test_signed_probe_detects_reverse_only_improvement_and_misalignment(
        self,
    ) -> None:
        metrics = evaluate_signed_direction_probe(
            [
                {
                    "signed_scale": 0.5,
                    "raw_auc": 0.62,
                    "raw_auc_gap": 0.12,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.30,
                },
                {
                    "signed_scale": 1.0,
                    "raw_auc": 0.64,
                    "raw_auc_gap": 0.14,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.40,
                },
                {
                    "signed_scale": -0.5,
                    "raw_auc": 0.58,
                    "raw_auc_gap": 0.08,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.05,
                },
                {
                    "signed_scale": -1.0,
                    "raw_auc": 0.59,
                    "raw_auc_gap": 0.09,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.00,
                },
            ],
            anchor_raw_auc_gap=0.10,
            anchor_raw_auc_se=0.001,
            anchor_reward_mean=0.10,
            confidence_z=1.96,
        )

        self.assertEqual(
            metrics["reference_trust/signed_probe/decision_code"], -1.0
        )
        self.assertEqual(
            metrics["reference_trust/signed_probe/best_overall_scale"], -0.5
        )
        self.assertEqual(
            metrics["reference_trust/signed_probe/reward_raw_misaligned"], 1.0
        )
        self.assertGreater(
            metrics[
                "reference_trust/signed_probe/candidate/minus_0p5/improvement"
            ],
            0.0,
        )

    def test_signed_probe_detects_smaller_positive_step(self) -> None:
        metrics = evaluate_signed_direction_probe(
            [
                {
                    "signed_scale": 0.25,
                    "raw_auc": 0.57,
                    "raw_auc_gap": 0.07,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.11,
                },
                {
                    "signed_scale": 1.0,
                    "raw_auc": 0.61,
                    "raw_auc_gap": 0.11,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.12,
                },
                {
                    "signed_scale": -0.25,
                    "raw_auc": 0.61,
                    "raw_auc_gap": 0.11,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.09,
                },
                {
                    "signed_scale": -1.0,
                    "raw_auc": 0.63,
                    "raw_auc_gap": 0.13,
                    "raw_auc_se": 0.001,
                    "reward_mean": 0.08,
                },
            ],
            anchor_raw_auc_gap=0.10,
            anchor_raw_auc_se=0.001,
            anchor_reward_mean=0.10,
            confidence_z=1.96,
        )

        self.assertEqual(
            metrics["reference_trust/signed_probe/decision_code"], 1.0
        )
        self.assertEqual(
            metrics["reference_trust/signed_probe/best_positive_scale"], 0.25
        )
        self.assertEqual(
            metrics[
                "reference_trust/signed_probe/smaller_positive_step_improves"
            ],
            1.0,
        )
        self.assertEqual(
            metrics["reference_trust/signed_probe/reward_raw_misaligned"], 1.0
        )

    def test_cross_round_radius_cannot_reexpand(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_cross_round_nonexpanding=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.003, cfg=cfg, epoch=0, round_id=1)
        first = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.55,
            raw_auc_se=0.001,
        )
        first_delta = float(first["reference_trust/delta"])

        state.install(baseline_auc_gap=0.003, cfg=cfg, epoch=2, round_id=2)
        second = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.60,
            raw_auc_se=0.001,
        )

        self.assertGreater(
            second["reference_trust/cross_round/delta_before_cap"],
            first_delta,
        )
        self.assertEqual(
            second["reference_trust/cross_round/cap_active"], 1.0
        )
        self.assertAlmostEqual(second["reference_trust/delta"], first_delta)

    def test_policy_lr_tracks_sqrt_radius_without_rebasing(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_cross_round_nonexpanding=True,
            trust_policy_lr_scaling_enabled=True,
            trust_policy_lr_scale_floor=0.1,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.003, cfg=cfg, epoch=0, round_id=1)
        first = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.60,
            raw_auc_se=0.001,
        )
        reference_delta = float(first["reference_trust/delta"])
        self.assertAlmostEqual(
            state.trust_policy_lr_reference_delta, reference_delta
        )

        state.install(baseline_auc_gap=0.003, cfg=cfg, epoch=2, round_id=2)
        second = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.55,
            raw_auc_se=0.001,
        )
        lr_diagnostics = adaptive_trust_policy_lr_scale(state, cfg=cfg)
        expected = max(
            cfg.trust_policy_lr_scale_floor,
            math.sqrt(float(second["reference_trust/delta"]) / reference_delta),
        )
        self.assertAlmostEqual(
            lr_diagnostics["reference_trust/policy_lr/scale"], expected
        )
        self.assertAlmostEqual(
            lr_diagnostics["reference_trust/policy_lr/reference_delta"],
            reference_delta,
        )

    def test_fixed_radius_resume_removes_inherited_live_expansion(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
            trust_empirical_allow_expansion=False,
            trust_refit_on_exhaustion=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.0036, cfg=cfg, epoch=54, round_id=6)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.5414,
            raw_auc_se=0.0013,
        )
        auc_scaled_delta = state.trust_current_delta
        state.trust_current_delta = 1.25 * auc_scaled_delta
        state.trust_empirical_safe_streak = 2

        diagnostics = clamp_fixed_trust_radius_after_resume(state, cfg=cfg)

        self.assertEqual(
            diagnostics["reference_trust/resume_radius_clamp_applied"], 1.0
        )
        self.assertAlmostEqual(state.trust_current_delta, auc_scaled_delta)
        self.assertEqual(state.trust_empirical_safe_streak, 0)

    def test_configured_fixed_radius_overrides_checkpoint_value_on_resume(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_radius_mode="fixed",
        )
        state = AdaptiveOmniFoldState()
        state.trust_current_delta = 0.01

        diagnostics = clamp_fixed_trust_radius_after_resume(state, cfg=cfg)

        self.assertEqual(
            diagnostics["reference_trust/resume_radius_clamp_applied"], 1.0
        )
        self.assertAlmostEqual(
            diagnostics["reference_trust/resume_radius_before"], 0.01
        )
        self.assertAlmostEqual(
            diagnostics["reference_trust/resume_radius_after"],
            cfg.trust_delta_max,
        )
        self.assertAlmostEqual(state.trust_current_delta, cfg.trust_delta_max)

    def test_fixed_radius_safe_audits_never_expand(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
            trust_empirical_allow_expansion=False,
            trust_refit_on_exhaustion=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.0036, cfg=cfg, epoch=-1, round_id=1)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.5414,
            raw_auc_se=0.0013,
        )
        original_delta = state.trust_current_delta
        for _ in range(50):
            record_reference_trust_attempt(
                state,
                cfg=cfg,
                accepted=False,
                update_scale=0.0,
                distance=float("nan"),
            )
        for distance in (1.2e-5, 1.3e-5, 1.4e-5, 1.3e-5, 1.2e-5):
            record_reference_trust_distance(state, cfg=cfg, distance=distance)
        for _ in range(2):
            diagnostics = update_empirical_trust_radius_from_audit(
                state,
                {
                    "weighted_auc_gap": 0.004,
                    "audit_auc_null_se_approx": 0.004,
                },
                cfg=cfg,
            )

        self.assertEqual(diagnostics["reference_trust/empirical/allow_expansion"], 0.0)
        self.assertEqual(diagnostics["reference_trust/empirical/radius_action"], 0.0)
        self.assertAlmostEqual(state.trust_current_delta, original_delta)
        self.assertEqual(state.trust_empirical_expansions, 0)

    def test_fixed_radius_exhaustion_requires_boundary_and_starved_step(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_empirical_allow_expansion=False,
            trust_refit_on_exhaustion=True,
        )
        state = AdaptiveOmniFoldState(trust_current_delta=1.0e-4)
        state.trust_distance_window = [7.9e-5]
        state.trust_step_scale_window = [0.0] * 5

        exhausted, diagnostics = trust_region_exhausted(state, cfg=cfg)

        self.assertTrue(exhausted)
        self.assertGreaterEqual(
            diagnostics["reference_trust/exhaustion/distance_fraction"],
            0.98,
        )
        state.trust_step_scale_window = [0.1] * 5
        exhausted, _ = trust_region_exhausted(state, cfg=cfg)
        self.assertFalse(exhausted)
        state.trust_step_scale_window = [0.0] * 5
        state.trust_distance_window = [7.0e-5]
        exhausted, _ = trust_region_exhausted(state, cfg=cfg)
        self.assertFalse(exhausted)

    def test_exhaustion_uses_only_the_configured_recent_scale_window(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_empirical_allow_expansion=False,
            trust_refit_on_exhaustion=True,
            trust_exhaustion_scale_window_steps=10,
        )
        state = AdaptiveOmniFoldState(trust_current_delta=1.0e-4)
        state.trust_distance_window = [7.9e-5]
        state.trust_step_scale_window = [0.2] * 40 + [0.0] * 10

        exhausted, diagnostics = trust_region_exhausted(state, cfg=cfg)

        self.assertTrue(exhausted)
        self.assertEqual(
            diagnostics["reference_trust/exhaustion/attempt_count"], 10.0
        )
        self.assertEqual(
            diagnostics["reference_trust/exhaustion/available_attempt_count"],
            50.0,
        )
        self.assertEqual(
            diagnostics["reference_trust/exhaustion/mean_update_scale"], 0.0
        )

    def test_round_auc_guard_distinguishes_improvement_plateau_and_regression(
        self,
    ) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_round_acceptance_enabled=True,
            trust_round_acceptance_confidence_z=1.96,
        )
        state = AdaptiveOmniFoldState(
            trust_current_raw_auc_gap=0.08,
            trust_current_raw_auc_se=0.002,
        )

        improved = evaluate_round_auc_change(
            state,
            cfg=cfg,
            raw_auc=0.56,
            raw_auc_se=0.002,
        )
        plateau = evaluate_round_auc_change(
            state,
            cfg=cfg,
            raw_auc=0.579,
            raw_auc_se=0.002,
        )
        regressed = evaluate_round_auc_change(
            state,
            cfg=cfg,
            raw_auc=0.59,
            raw_auc_se=0.002,
        )
        bypassed = evaluate_round_auc_change(
            state,
            cfg=cfg,
            raw_auc=0.59,
            raw_auc_se=0.002,
            enforce=False,
        )

        self.assertEqual(improved["decision"], "improved")
        self.assertEqual(plateau["decision"], "plateau")
        self.assertEqual(regressed["decision"], "regressed")
        self.assertEqual(bypassed["decision"], "bypass")
        self.assertGreater(
            improved["improvement"], improved["required_improvement"]
        )

    def test_trajectory_keeps_lowest_fixed_panel_raw_auc_checkpoint(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_trajectory_search_enabled=True,
            raw_audit_enabled=True,
        )
        state = AdaptiveOmniFoldState(reward_round_id=4)
        first = update_trust_trajectory_candidate(
            state,
            cfg=cfg,
            raw_auc_gap=0.08,
            raw_auc_se=0.002,
            epoch=10,
            global_step=100,
            checkpoint_path="/tmp/step100.ckpt",
            audit_saturated=True,
        )
        worse = update_trust_trajectory_candidate(
            state,
            cfg=cfg,
            raw_auc_gap=0.09,
            raw_auc_se=0.002,
            epoch=12,
            global_step=120,
            checkpoint_path="/tmp/step120.ckpt",
            audit_saturated=True,
        )
        best = update_trust_trajectory_candidate(
            state,
            cfg=cfg,
            raw_auc_gap=0.05,
            raw_auc_se=0.002,
            epoch=14,
            global_step=140,
            checkpoint_path="/tmp/step140.ckpt",
            audit_saturated=True,
        )
        self.assertEqual(first["reference_trust/trajectory/best_updated"], 1.0)
        self.assertEqual(worse["reference_trust/trajectory/best_updated"], 0.0)
        self.assertEqual(best["reference_trust/trajectory/best_updated"], 1.0)
        self.assertEqual(state.trust_trajectory_best_global_step, 140)
        self.assertEqual(
            state.trust_trajectory_best_checkpoint,
            "/tmp/step140.ckpt",
        )
        self.assertEqual(state.trust_trajectory_candidate_count, 3)
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(
            restored.trust_trajectory_best_checkpoint,
            state.trust_trajectory_best_checkpoint,
        )

    def test_unsaturated_raw_audit_is_not_a_trajectory_candidate(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_trajectory_search_enabled=True,
            raw_audit_enabled=True,
        )
        state = AdaptiveOmniFoldState(reward_round_id=2)
        diagnostics = update_trust_trajectory_candidate(
            state,
            cfg=cfg,
            raw_auc_gap=0.01,
            raw_auc_se=0.002,
            epoch=3,
            global_step=30,
            checkpoint_path="/tmp/step30.ckpt",
            audit_saturated=False,
        )
        self.assertEqual(diagnostics["reference_trust/trajectory/eligible"], 0.0)
        self.assertEqual(state.trust_trajectory_candidate_count, 0)
        self.assertEqual(state.trust_trajectory_best_checkpoint, "")

    def test_repeated_topology_audit_aggregates_three_fit_seeds(self) -> None:
        cfg = _config(topology_acceptance_repeats=3)
        calls: list[int] = []
        progress: list[tuple[str, int]] = []

        def _single_audit(**kwargs):
            calls.append(int(kwargs["seed"]))
            kwargs["progress_callback"]({"step": 10.0})
            index = len(calls) - 1
            return {
                "topology_audit_auc": 0.502 + 0.001 * index,
                "topology_audit_auc_gap": 0.002 + 0.001 * index,
                "topology_audit_balanced_accuracy": 0.501 + 0.001 * index,
                "topology_audit_saturated": 1.0,
                "topology_audit_same_architecture": 1.0,
                "topology_audit_fit_events": 60.0,
                "topology_audit_test_events": 20.0,
            }

        with mock.patch.object(
            adaptive_module,
            "fit_fresh_topology_audit",
            side_effect=_single_audit,
        ):
            metrics = adaptive_module.fit_repeated_topology_audit(
                pool=object(),
                log_weight=torch.zeros(1),
                model_builder=object(),
                cfg=cfg,
                device=torch.device("cpu"),
                seed=17,
                progress_callback=lambda phase, row: progress.append(
                    (phase, int(row["repeat"]))
                ),
            )

        self.assertEqual(calls, [17, 104_746, 209_475])
        self.assertEqual(
            progress,
            [
                ("topology_acceptance_audit", 1),
                ("topology_acceptance_audit", 2),
                ("topology_acceptance_audit", 3),
            ],
        )
        self.assertEqual(metrics["topology_audit_repeats"], 3.0)
        self.assertEqual(metrics["topology_audit_same_architecture"], 1.0)
        self.assertAlmostEqual(metrics["topology_audit_auc_gap"], 0.003)
        self.assertGreater(metrics["topology_audit_auc_gap_se"], 0.0)
        self.assertAlmostEqual(
            metrics["topology_audit_repeat_03_auc_gap"], 0.004
        )

    def test_topology_confirmation_reuses_the_omnifold_model_builder(self) -> None:
        builder = object()
        audit_result = {
            "judge_auc_weighted": 0.503,
            "weighted_auc_gap": 0.003,
            "audit_balanced_accuracy": 0.502,
            "audit_saturated": 1.0,
            "audit_fit_events": 60.0,
            "audit_test_events": 20.0,
        }
        with mock.patch.object(
            adaptive_module,
            "fit_fresh_audit",
            return_value=audit_result,
        ) as fresh:
            metrics = adaptive_module.fit_fresh_topology_audit(
                pool=object(),
                log_weight=torch.zeros(1),
                model_builder=builder,
                cfg=_config(),
                device=torch.device("cpu"),
                seed=31,
            )

        self.assertIs(fresh.call_args.kwargs["model_builder"], builder)
        self.assertEqual(metrics["topology_audit_same_architecture"], 1.0)
        self.assertAlmostEqual(metrics["topology_audit_auc_gap"], 0.003)

    def test_three_audits_reuse_primary_acceptance_as_repeat_one(self) -> None:
        cfg = _config(topology_acceptance_repeats=3)
        initial = {
            "judge_auc_weighted": 0.502,
            "weighted_auc_gap": 0.002,
            "audit_balanced_accuracy": 0.501,
            "audit_saturated": 1.0,
            "audit_fit_events": 60.0,
            "audit_test_events": 20.0,
        }
        extra = {
            "topology_audit_auc": 0.503,
            "topology_audit_auc_gap": 0.003,
            "topology_audit_balanced_accuracy": 0.502,
            "topology_audit_saturated": 1.0,
            "topology_audit_same_architecture": 1.0,
            "topology_audit_fit_events": 60.0,
            "topology_audit_test_events": 20.0,
        }
        with mock.patch.object(
            adaptive_module,
            "fit_fresh_topology_audit",
            return_value=extra,
        ) as fresh:
            metrics = adaptive_module.fit_repeated_topology_audit(
                pool=object(),
                log_weight=torch.zeros(1),
                model_builder=object(),
                cfg=cfg,
                device=torch.device("cpu"),
                seed=41,
                initial_audit=initial,
            )

        self.assertEqual(fresh.call_count, 2)
        self.assertEqual(
            [call.kwargs["seed"] for call in fresh.call_args_list],
            [41, 104_770],
        )
        self.assertAlmostEqual(
            metrics["topology_audit_repeat_01_auc_gap"], 0.002
        )

    def test_bidirectional_safe_audits_expand_a_starved_live_radius(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.0036, cfg=cfg, epoch=-1, round_id=1)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.5414,
            raw_auc_se=0.0013,
        )
        original_delta = state.trust_current_delta
        for _ in range(50):
            record_reference_trust_attempt(
                state,
                cfg=cfg,
                accepted=False,
                update_scale=0.0,
                distance=float("nan"),
            )
        for distance in (1.2e-5, 1.3e-5, 1.4e-5, 1.3e-5, 1.2e-5):
            record_reference_trust_distance(state, cfg=cfg, distance=distance)

        first = update_empirical_trust_radius_from_audit(
            state,
            {
                # Below the installed baseline+0.01 threshold by point estimate,
                # but not by a 95% UCB: the safe streak guards expansion.
                "weighted_auc_gap": 0.004,
                "audit_auc_null_se_approx": 0.004,
            },
            cfg=cfg,
        )
        self.assertEqual(first["reference_trust/empirical/point_safe"], 1.0)
        self.assertEqual(first["reference_trust/empirical/updated"], 0.0)

        second = update_empirical_trust_radius_from_audit(
            state,
            {
                "weighted_auc_gap": 0.005,
                "audit_auc_null_se_approx": 0.004,
            },
            cfg=cfg,
        )
        self.assertEqual(second["reference_trust/empirical/radius_action"], 1.0)
        self.assertEqual(second["reference_trust/empirical/updated"], 1.0)
        self.assertAlmostEqual(
            state.trust_current_delta,
            min(cfg.trust_delta_max, 1.25 * original_delta),
        )
        self.assertEqual(state.trust_empirical_expansions, 1)

    def test_bidirectional_safe_audit_does_not_expand_healthy_throughput(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.0036, cfg=cfg, epoch=-1, round_id=1)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.5414,
            raw_auc_se=0.0013,
        )
        original_delta = state.trust_current_delta
        for _ in range(10):
            record_reference_trust_attempt(
                state,
                cfg=cfg,
                accepted=True,
                update_scale=0.5,
                distance=1.0e-5,
            )
        for _ in range(2):
            diagnostics = update_empirical_trust_radius_from_audit(
                state,
                {
                    "weighted_auc_gap": 0.004,
                    "audit_auc_null_se_approx": 0.004,
                },
                cfg=cfg,
            )
        self.assertEqual(
            diagnostics["reference_trust/empirical/throughput_starved"], 0.0
        )
        self.assertAlmostEqual(state.trust_current_delta, original_delta)
        self.assertEqual(state.trust_empirical_expansions, 0)

    def test_unsaturated_audit_cannot_expand_radius(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.0036, cfg=cfg, epoch=-1, round_id=1)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.5414,
            raw_auc_se=0.0013,
        )
        original_delta = state.trust_current_delta
        for _ in range(50):
            record_reference_trust_attempt(
                state,
                cfg=cfg,
                accepted=False,
                update_scale=0.0,
                distance=float("nan"),
            )
        for distance in (1.0e-5,) * 5:
            record_reference_trust_distance(state, cfg=cfg, distance=distance)
        for _ in range(2):
            diagnostics = update_empirical_trust_radius_from_audit(
                state,
                {
                    "weighted_auc_gap": 0.001,
                    "audit_auc_null_se_approx": 0.001,
                    "audit_saturated": 0.0,
                },
                cfg=cfg,
            )
        self.assertEqual(
            diagnostics["reference_trust/empirical/audit_saturated"], 0.0
        )
        self.assertAlmostEqual(state.trust_current_delta, original_delta)

    def test_attempt_windows_are_bounded_and_checkpointed(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_empirical_bidirectional=True,
        )
        state = AdaptiveOmniFoldState()
        for index in range(80):
            record_reference_trust_attempt(
                state,
                cfg=cfg,
                accepted=index % 2 == 0,
                update_scale=0.25 if index % 2 == 0 else 0.0,
                distance=1.0e-5,
            )
        self.assertEqual(len(state.trust_step_acceptance_window), 50)
        self.assertEqual(len(state.trust_step_scale_window), 50)
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(
            restored.trust_step_acceptance_window,
            state.trust_step_acceptance_window,
        )
        self.assertEqual(restored.trust_step_scale_window, state.trust_step_scale_window)

    def test_new_reward_round_invalidates_fixed_probe_payload(self) -> None:
        cfg = _config(trust_boundary_enabled=True)
        state = AdaptiveOmniFoldState(
            reward_round_id=2,
            trust_probe_payload={"x_t": torch.ones(1)},
            trust_probe_round_id=2,
        )
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=5, round_id=3)
        self.assertIsNone(state.trust_probe_payload)
        self.assertEqual(state.trust_probe_round_id, -1)

    def test_fixed_probe_payload_survives_same_round_checkpoint_state(self) -> None:
        payload = {
            "x_t": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "reference_loss_count": 2,
        }
        state = AdaptiveOmniFoldState(
            reward_round_id=4,
            trust_probe_payload=payload,
            trust_probe_round_id=4,
        )
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(restored.trust_probe_round_id, 4)
        self.assertIsNotNone(restored.trust_probe_payload)
        torch.testing.assert_close(
            restored.trust_probe_payload["x_t"],
            payload["x_t"],
        )

    def test_empirical_unsafe_audit_caps_only_later_rounds(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=1)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.807,
            raw_auc_se=0.0,
        )
        original_delta = state.trust_current_delta
        for distance in (0.00038, 0.00040, 0.00042, 0.00041, 0.00039):
            record_reference_trust_distance(
                state,
                cfg=cfg,
                distance=distance,
            )
        diagnostics = update_empirical_trust_radius_from_audit(
            state,
            {
                "weighted_auc_gap": 0.03,
                "audit_auc_null_se_approx": 0.001,
            },
            cfg=cfg,
        )
        self.assertEqual(diagnostics["reference_trust/empirical/status"], 1.0)
        self.assertEqual(diagnostics["reference_trust/empirical/updated"], 1.0)
        # Do not retroactively strand the current policy outside a newly
        # shrunken boundary.
        self.assertAlmostEqual(state.trust_current_delta, original_delta)
        cap = state.trust_empirical_delta_cap
        self.assertGreater(cap, cfg.trust_delta_floor)
        self.assertLess(cap, 0.00025)

        next_round = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.807,
            raw_auc_se=0.0,
        )
        self.assertAlmostEqual(next_round["reference_trust/delta"], cap)
        self.assertEqual(
            next_round["reference_trust/empirical_delta_cap_active"], 1.0
        )

    def test_empirical_safe_or_ambiguous_audit_never_expands_radius(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=1)
        for distance in (1.0e-4,) * 5:
            record_reference_trust_distance(
                state,
                cfg=cfg,
                distance=distance,
            )
        safe = update_empirical_trust_radius_from_audit(
            state,
            {
                "weighted_auc_gap": 0.011,
                "audit_auc_null_se_approx": 0.001,
            },
            cfg=cfg,
        )
        self.assertEqual(safe["reference_trust/empirical/status"], -1.0)
        self.assertFalse(math.isfinite(state.trust_empirical_delta_cap))

        ambiguous = update_empirical_trust_radius_from_audit(
            state,
            {
                "weighted_auc_gap": 0.020,
                "audit_auc_null_se_approx": 0.002,
            },
            cfg=cfg,
        )
        self.assertEqual(
            ambiguous["reference_trust/empirical/status"], 0.0
        )
        self.assertFalse(math.isfinite(state.trust_empirical_delta_cap))

    def test_empirical_distance_window_is_bounded_and_checkpointed(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
        )
        state = AdaptiveOmniFoldState()
        for index in range(20):
            record_reference_trust_distance(
                state,
                cfg=cfg,
                distance=float(index),
            )
        self.assertEqual(len(state.trust_distance_window), 10)
        self.assertEqual(state.trust_distance_window[0], 10.0)
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(restored.trust_distance_window, state.trust_distance_window)

    def test_disabling_empirical_calibration_ignores_a_restored_cap(self) -> None:
        cfg = _config(trust_boundary_enabled=True)
        state = AdaptiveOmniFoldState(trust_empirical_delta_cap=7.0e-6)
        diagnostics = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.807,
            raw_auc_se=0.0,
        )
        self.assertAlmostEqual(
            diagnostics["reference_trust/delta"], cfg.trust_delta_max
        )
        self.assertEqual(
            diagnostics["reference_trust/empirical_delta_cap_active"], 0.0
        )

    def test_trust_radius_shrinks_quadratically_with_raw_auc_gap(self) -> None:
        cfg = _config(trust_boundary_enabled=True)
        state = AdaptiveOmniFoldState()
        se = auc_null_standard_error(100_000, 100_000)

        first = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.807,
            raw_auc_se=se,
        )
        self.assertAlmostEqual(first["reference_trust/delta"], 1.0e-3)

        second = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.538,
            raw_auc_se=se,
        )
        self.assertLess(second["reference_trust/delta"], 2.0e-5)
        self.assertGreaterEqual(second["reference_trust/delta"], 5.0e-6)

    def test_trust_radius_uses_floor_at_statistical_closure(self) -> None:
        cfg = _config(trust_boundary_enabled=True)
        state = AdaptiveOmniFoldState()
        se = auc_null_standard_error(100_000, 100_000)
        install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.807,
            raw_auc_se=se,
        )
        diagnostics = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.505,
            raw_auc_se=se,
        )
        self.assertAlmostEqual(
            diagnostics["reference_trust/delta"], cfg.trust_delta_floor
        )
        self.assertTrue(state.trust_statistically_closed)

    def test_resume_can_preserve_original_bootstrap_gap(self) -> None:
        cfg = _config(
            trust_boundary_enabled=True,
            trust_initial_raw_auc_gap=0.307,
        )
        state = AdaptiveOmniFoldState()
        diagnostics = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.538,
            raw_auc_se=0.0,
        )
        expected = 1.0e-3 * (0.038 / 0.307) ** 2
        self.assertAlmostEqual(
            diagnostics["reference_trust/delta"], expected
        )

    def test_trust_radius_preview_has_no_state_side_effect(self) -> None:
        cfg = _config(trust_boundary_enabled=True)
        state = AdaptiveOmniFoldState()
        diagnostics = install_adaptive_trust_round(
            state,
            cfg=cfg,
            raw_auc=0.70,
            raw_auc_se=0.002,
            commit=False,
        )
        self.assertTrue(math.isfinite(diagnostics["reference_trust/delta"]))
        self.assertTrue(math.isnan(state.trust_current_delta))
        self.assertTrue(math.isnan(state.trust_initial_effective_raw_auc_gap))

    def test_two_consecutive_crossings_are_required_when_configured(self) -> None:
        cfg = _config(
            retrain_auc_margin=0.005,
            required_consecutive_epochs=2,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=1)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.02},
            cfg=cfg,
            epoch=4,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "threshold_exceeded")
        self.assertEqual(state.probe_exceedance_streak, 1)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.021},
            cfg=cfg,
            epoch=9,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")
        self.assertEqual(state.probe_exceedance_streak, 2)

    def test_cooldown_requires_fresh_post_cooldown_crossings(self) -> None:
        cfg = _config(
            retrain_auc_margin=0.005,
            required_consecutive_epochs=2,
            retrain_cooldown_epochs=10,
        )
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=20, round_id=2)
        state.last_recalibration_epoch = 20

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.03},
            cfg=cfg,
            epoch=25,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "cooldown")
        self.assertEqual(diagnostics["staleness/cooldown_active"], 1.0)
        self.assertEqual(state.probe_exceedance_streak, 0)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.03},
            cfg=cfg,
            epoch=30,
        )
        self.assertFalse(trigger)
        self.assertEqual(state.probe_exceedance_streak, 1)

        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.031},
            cfg=cfg,
            epoch=35,
        )
        self.assertTrue(trigger)
        self.assertEqual(state.probe_exceedance_streak, 2)

    def test_trigger_compares_to_installed_round_baseline(self) -> None:
        cfg = _config(retrain_auc_margin=0.005)
        state = AdaptiveOmniFoldState()
        state.install(
            baseline_auc_gap=0.020,
            cfg=cfg,
            epoch=-1,
            round_id=0,
        )
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.024,
            },
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(trigger)
        self.assertAlmostEqual(diagnostics["staleness/trigger_threshold"], 0.025)
        self.assertAlmostEqual(state.baseline_auc_gap, 0.020)

        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.028,
            },
            cfg=cfg,
            epoch=19,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")
        self.assertAlmostEqual(
            diagnostics["staleness/previous_audit_auc_gap"], 0.024
        )
        self.assertAlmostEqual(diagnostics["staleness/trigger_threshold"], 0.025)
        self.assertAlmostEqual(state.baseline_auc_gap, 0.020)
        self.assertAlmostEqual(state.previous_audit_auc_gap, 0.028)

    def test_trigger_is_exactly_baseline_plus_configured_margin(self) -> None:
        cfg = _config(retrain_auc_margin=0.005)
        state = AdaptiveOmniFoldState()
        state.install(
            baseline_auc_gap=0.020,
            cfg=cfg,
            epoch=-1,
            round_id=0,
        )
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.025,
            },
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(trigger)
        self.assertAlmostEqual(
            diagnostics["staleness/required_auc_gap_increase"],
            0.005,
        )
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.025001,
            },
            cfg=cfg,
            epoch=19,
        )
        self.assertTrue(trigger)

    def test_ess_is_diagnostic_only(self) -> None:
        cfg = _config(retrain_auc_margin=0.005)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.020, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.021, "ess_fraction": 0.0},
            cfg=cfg,
            epoch=9,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "healthy")

    def test_unsaturated_audit_gap_can_trigger_when_configured(self) -> None:
        cfg = _config(require_audit_saturation=False)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 0.0,
                "ess_fraction": 1.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")

    def test_required_saturation_blocks_trigger_until_audit_saturates(self) -> None:
        cfg = _config(require_audit_saturation=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 0.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "audit_unsaturated")
        self.assertEqual(state.probe_exceedance_streak, 0)

        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 1.0,
            },
            cfg=cfg,
            epoch=3,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")

    def test_confirmed_threshold_crossing_can_stop_before_saturation(self) -> None:
        cfg = _config(require_audit_saturation=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.08,
                "audit_saturated": 0.0,
                "audit_threshold_reached": 1.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertTrue(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "recalibrate")

    def test_validation_crossing_needs_final_split_confirmation(self) -> None:
        cfg = _config(require_audit_saturation=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1, round_id=0)
        trigger, diagnostics = update_controller(
            state,
            {
                "weighted_auc_gap": 0.015,
                "audit_saturated": 0.0,
                "audit_threshold_reached": 1.0,
            },
            cfg=cfg,
            epoch=1,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "audit_unsaturated")

    def test_log_only_never_authorizes_refit(self) -> None:
        cfg = _config(log_only=True)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.01, cfg=cfg, epoch=-1)
        trigger, diagnostics = update_controller(
            state,
            {"weighted_auc_gap": 0.2, "ess_fraction": 1.0},
            cfg=cfg,
            epoch=1,
        )
        self.assertFalse(trigger)
        self.assertEqual(diagnostics["staleness/decision"], "stale_log_only")

    def test_state_round_trip_preserves_fixed_baseline(self) -> None:
        cfg = _config()
        state = AdaptiveOmniFoldState()
        state.install(
            baseline_auc_gap=0.08,
            cfg=cfg,
            epoch=3,
            round_id=2,
        )
        state.audit_protocol_signature = adaptive_audit_protocol_signature(cfg)
        state.resume_refit_once_completed = True
        state.resume_refit_once_id = cfg.refit_once_id
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        self.assertEqual(restored.reward_round_id, 2)
        self.assertEqual(restored.trigger_threshold, state.trigger_threshold)
        self.assertEqual(restored.baseline_auc_gap, state.baseline_auc_gap)
        self.assertTrue(restored.resume_refit_once_completed)
        self.assertEqual(restored.resume_refit_once_id, cfg.refit_once_id)
        self.assertEqual(
            restored.audit_protocol_signature,
            state.audit_protocol_signature,
        )

    def test_old_checkpoint_migrates_previous_audit_from_history(self) -> None:
        restored = AdaptiveOmniFoldState.from_dict(
            {
                "reward_round_id": 5,
                "baseline_auc_gap": 0.019,
                "trigger_threshold": 0.024,
                "probe_history": [
                    {"weighted_auc_gap": 0.012},
                    {"weighted_auc_gap": 0.007},
                ],
            }
        )
        self.assertAlmostEqual(restored.previous_audit_auc_gap, 0.007)

    def test_protocol_change_invalidates_only_audit_baseline(self) -> None:
        restored = AdaptiveOmniFoldState.from_dict(
            {
                "reward_round_id": 5,
                "baseline_auc_gap": 0.019,
                "trigger_threshold": 0.024,
            }
        )
        self.assertTrue(restored.calibrated)
        restored.invalidate_audit_baseline(reason="protocol_changed")
        self.assertEqual(restored.reward_round_id, 5)
        self.assertFalse(restored.calibrated)
        self.assertEqual(restored.last_decision, "protocol_changed")

class TestAdaptivePool(unittest.TestCase):
    def test_readiness_reaches_raw_monitor_but_not_classifier_trust(self):
        cfg = replace(_config(raw_audit_enabled=True), single_pool_train_validation=True)
        policy = {"cold_start_min_epochs": 100, "warm_start_min_epochs": 5}
        cfg.audit_fit["training_readiness"] = policy
        spec = EventPackingSpec({"x": (1, 1), "x_mask": (1, 1),
                                 "conditions": (1, 1), "conditions_mask": (1,)})
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.arange(200, dtype=torch.float32)[:, None].expand(-1, spec.width).clone(),
            truth=torch.zeros(200, 4), candidates=torch.zeros(200, 1, 4), packing_spec=spec,
        )
        result = SimpleNamespace(
            auc=.6, auc_gap=.1, balanced_accuracy=.57, warm_started=False,
            fit_diagnostics=SimpleNamespace(saturated=True, steps_completed=660),
            fit_events=160, audit_events=40,
            training_ready=True, training_min_steps=600, training_steps_per_epoch=6,
        )
        builder = SimpleNamespace(make_classifier=lambda *args, **kwargs: torch.nn.Linear(1, 1))
        with mock.patch.object(adaptive_module, "fit_independent_evenet_audit", return_value=result) as fit:
            raw = adaptive_module.fit_raw_policy_audit(
                pool=pool, model_builder=builder, cfg=cfg, device=torch.device("cpu"), seed=123,
            )
            self.assertEqual(fit.call_args.kwargs["training_readiness"], policy)
            self.assertTrue(torch.equal(fit.call_args.kwargs["gen_weight"], torch.ones(200, 1)))
            self.assertEqual(raw["raw_audit_training_ready"], 1.)
            self.assertEqual(raw["raw_audit_training_min_steps"], 600.)
            self.assertEqual(raw["raw_audit_training_epochs"], 110.)
            adaptive_module.fit_reference_trust_audit(
                pool=pool, model_builder=builder, cfg=cfg, device=torch.device("cpu"), seed=123,
            )
            self.assertNotIn("training_readiness", fit.call_args.kwargs)

    def test_single_pool_refit_uses_disjoint_identity_stable_splits(self):
        spec = EventPackingSpec({"x": (1, 1), "x_mask": (1, 1),
                                 "conditions": (1, 1), "conditions_mask": (1,)})
        ids = torch.arange(200, dtype=torch.float32)
        pool = AdaptiveOmniFoldPool(
            packed_event=ids[:, None].expand(-1, spec.width).clone(),
            truth=ids[:, None].expand(-1, 4).clone(),
            candidates=ids[:, None, None].expand(-1, 1, 4).clone(), packing_spec=spec,
        )
        fit, val = adaptive_module.split_single_classifier_pool(pool, seed=42)
        fit_ids, val_ids = set(fit.truth[:, 0].tolist()), set(val.truth[:, 0].tolist())
        self.assertTrue(fit_ids.isdisjoint(val_ids))
        self.assertEqual(fit_ids | val_ids, set(ids.tolist()))
        reverse_fit, reverse_val = adaptive_module.split_single_classifier_pool(pool.select(torch.arange(199, -1, -1)), seed=42)
        self.assertEqual(fit_ids, set(reverse_fit.truth[:, 0].tolist()))
        self.assertEqual(val_ids, set(reverse_val.truth[:, 0].tolist()))
        cfg = replace(_config(monitor_mode="raw_plateau_refit", raw_audit_enabled=True, acceptance_audit_enabled=False),
                      single_pool_train_validation=True, single_pool_split_seed=42,
                      warm_start_iterations=(1, 2))
        source = SimpleNamespace(model_builder=mock.Mock(), is_installed=True,
                                 frozen_reward=SimpleNamespace(packing_spec=spec, warm_start_state={"legacy": True}))
        with mock.patch.object(adaptive_module, "peft_bank_factory", return_value=lambda: torch.nn.Linear(1, 1)), \
             mock.patch.object(adaptive_module, "fit_residual_ratio_stack", side_effect=RuntimeError("split test sentinel")) as mocked_fit:
            with self.assertRaisesRegex(RuntimeError, "split test sentinel"):
                run_adaptive_refit(state=AdaptiveOmniFoldState(), cfg=cfg, reward_source=source,
                                   round_ref_model=torch.nn.Linear(1, 1), policy_snapshot_state_dict={},
                                   fit_pool=pool, score_pool=pool, epoch=-1, device=torch.device("cpu"), world_size=1)
        self.assertEqual(set(mocked_fit.call_args.kwargs["data_sample"][:, 0].tolist()), fit_ids)
        self.assertEqual(set(mocked_fit.call_args.kwargs["validation_data_sample"][:, 0].tolist()), val_ids)
        self.assertIsNone(mocked_fit.call_args.kwargs["warm_start_state"])

    def test_reference_trust_pool_pairs_reference_against_current(self) -> None:
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        packed = torch.randn(12, spec.width)
        current_candidates = torch.randn(12, 1, 4)
        reference_candidates = torch.randn(12, 1, 4)
        current = AdaptiveOmniFoldPool(
            packed_event=packed,
            truth=torch.randn(12, 4),
            candidates=current_candidates,
            packing_spec=spec,
        )
        reference = AdaptiveOmniFoldPool(
            packed_event=packed.clone(),
            truth=torch.randn(12, 4),
            candidates=reference_candidates,
            packing_spec=spec,
        )

        paired = build_reference_trust_pool(current, reference)

        torch.testing.assert_close(paired.truth, reference_candidates[:, 0])
        torch.testing.assert_close(paired.candidates, current_candidates)
        torch.testing.assert_close(paired.packed_event, packed)

        mismatched_reference = AdaptiveOmniFoldPool(
            packed_event=packed.clone(),
            truth=reference.truth,
            candidates=reference.candidates,
            packing_spec=spec,
        )
        mismatched_reference.packed_event[0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "event identities"):
            build_reference_trust_pool(current, mismatched_reference)

    def test_pool_contract_is_four_dimensional_k1(self) -> None:
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        self.assertEqual(pool.n_events, 32)
        with self.assertRaisesRegex(ValueError, "exactly K=1"):
            AdaptiveOmniFoldPool(
                packed_event=pool.packed_event,
                truth=pool.truth,
                candidates=torch.randn(32, 2, 4),
                packing_spec=spec,
            )

    def test_fresh_audit_fits_only_the_weighted_judge(self) -> None:
        cfg = _config()
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )

        class _Builder:
            def __init__(self) -> None:
                self.discarded: list[str] = []

            def make_classifier(self, *_args, **_kwargs):
                return torch.nn.Linear(1, 1)

            def discard_bank(self, name: str) -> None:
                self.discarded.append(name)

        def _result(*, auc: float, accuracy: float):
            return SimpleNamespace(
                auc=auc,
                auc_gap=abs(auc - 0.5),
                balanced_accuracy=accuracy,
                fit_diagnostics=SimpleNamespace(
                    saturated=True,
                    threshold_reached=False,
                    validation_loss=0.69,
                    validation_balanced_accuracy=accuracy,
                    validation_auc=auc,
                ),
                fit_events=19,
                audit_events=6,
            )

        builder = _Builder()
        log_weight = torch.linspace(-1.0, 1.0, 32).reshape(32, 1)
        with mock.patch.object(
            adaptive_module,
            "fit_independent_evenet_audit",
            return_value=_result(auc=0.53, accuracy=0.54),
        ) as fit:
            metrics = adaptive_module.fit_fresh_audit(
                pool=pool,
                log_weight=log_weight,
                model_builder=builder,
                cfg=cfg,
                device=torch.device("cpu"),
                seed=123,
            )

        self.assertEqual(fit.call_count, 1)
        audit_fit_config = fit.call_args_list[0].kwargs["fit_config"]
        self.assertEqual(audit_fit_config.batch_size, 64)
        self.assertEqual(audit_fit_config.validation_interval_steps, 1)
        self.assertEqual(audit_fit_config.validation_patience_evaluations, 5)
        weighted_gen_weight = fit.call_args_list[0].kwargs["gen_weight"]
        self.assertFalse(torch.allclose(weighted_gen_weight, torch.ones_like(weighted_gen_weight)))
        self.assertFalse(
            fit.call_args_list[0].kwargs["reuse_early_stop_for_audit"]
        )
        self.assertAlmostEqual(metrics["judge_auc_weighted"], 0.53)
        self.assertEqual(metrics["audit_reused_validation_for_final"], 0.0)
        self.assertNotIn("judge_auc_raw", metrics)
        self.assertEqual(builder.discarded, ["audit"])

        # Fit-block dropout controls must reach the actual monitor constructor,
        # independently of the shared builder's reward classifier defaults.
        cfg.audit_fit.update(head_dropout=0.25, topology_dropout=0.25,
                             decoder_hidden_dim=128, decoder_layers=1,
                             periodic_pair_features=False, topology_fourier_embedding=False,
                             topology_conditioning=False)
        with mock.patch.object(builder, "make_classifier", wraps=builder.make_classifier) as make, \
             mock.patch.object(adaptive_module, "fit_independent_evenet_audit",
                               return_value=_result(auc=0.53, accuracy=0.54)) as fit:
            adaptive_module.fit_fresh_audit(
                pool=pool, log_weight=log_weight, model_builder=builder, cfg=cfg,
                device=torch.device("cpu"), seed=123,
            )
            fit.call_args.kwargs["model_factory"]()
            make.assert_called_once_with(
                pool.packing_spec, "audit", reset=True,
                head_dropout=0.25, topology_dropout=0.25,
                decoder_hidden_dim=128, decoder_layers=1,
                periodic_pair_features=False, topology_fourier_embedding=False,
                topology_conditioning=False,
            )

    def test_independent_audit_can_use_train_validation_only(self) -> None:
        n_events = 100
        condition = torch.randn(n_events, 3)
        truth = torch.randn(n_events, 4)
        generated = torch.randn(n_events, 1, 4)
        weight = torch.ones(n_events, 1)

        with (
            mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
                return_value=SimpleNamespace(saturated=True),
            ),
            mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio._score_population",
                side_effect=lambda _model, cond, _sample, _batch: torch.zeros(
                    cond.shape[0]
                ),
            ),
        ):
            result = fit_independent_evenet_audit(
                model_factory=lambda: torch.nn.Linear(1, 1),
                data_condition=condition,
                data_sample=truth,
                gen_condition=condition,
                gen_sample=generated,
                gen_weight=weight,
                fit_config=SimpleNamespace(
                    validation_batch_size=128,
                    require_saturation=False,
                ),
                seed=123,
                reuse_early_stop_for_audit=True,
            )

        self.assertEqual(result.fit_events, 80)
        self.assertEqual(result.early_stop_events, 20)
        self.assertEqual(result.audit_events, 20)
        self.assertAlmostEqual(result.auc, 0.5)

    def test_raw_audit_reuses_pool_but_trains_an_unweighted_fresh_judge(self) -> None:
        cfg = _config(raw_audit_enabled=True)
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        weighted = {
            "weighted_auc_gap": 0.01,
            "judge_auc_weighted": 0.51,
            "audit_balanced_accuracy": 0.50,
            "audit_saturated": 1.0,
            "audit_validation_loss": 0.69,
            "audit_validation_auc": 0.51,
            "audit_fit_events": 19.0,
            "audit_test_events": 6.0,
            "audit_probe_events": 32.0,
        }
        raw = {
            **weighted,
            "weighted_auc_gap": 0.16,
            "judge_auc_weighted": 0.66,
            "audit_balanced_accuracy": 0.64,
            "audit_validation_loss": 0.61,
            "audit_validation_auc": 0.65,
        }
        log_weight = torch.linspace(-1.0, 1.0, 32).reshape(32, 1)
        phases: list[str] = []
        raw_monitor_cache: dict[str, object] = {}
        source = SimpleNamespace(frozen_reward=object(), model_builder=object())
        with (
            mock.patch.object(
                adaptive_module,
                "score_reward_on_pool",
                return_value=log_weight,
            ),
            mock.patch.object(
                adaptive_module,
                "fit_fresh_audit",
                side_effect=(weighted, raw),
            ) as fit,
        ):
            metrics = adaptive_module.probe_installed_reward(
                source,
                pool,
                cfg=cfg,
                device=torch.device("cpu"),
                seed=123,
                raw_warm_start_cache=raw_monitor_cache,
                progress_callback=lambda phase, _row: phases.append(phase),
            )

        self.assertEqual(fit.call_count, 2)
        self.assertTrue(
            fit.call_args_list[0].kwargs["reuse_validation_for_final"]
        )
        self.assertTrue(
            fit.call_args_list[1].kwargs["reuse_validation_for_final"]
        )
        torch.testing.assert_close(fit.call_args_list[0].kwargs["log_weight"], log_weight)
        torch.testing.assert_close(
            fit.call_args_list[1].kwargs["log_weight"],
            torch.zeros_like(log_weight),
        )
        self.assertIsNone(fit.call_args_list[1].kwargs["early_stop_auc_gap"])
        self.assertNotIn("warm_start_cache", fit.call_args_list[0].kwargs)
        self.assertIs(
            fit.call_args_list[1].kwargs["warm_start_cache"],
            raw_monitor_cache,
        )
        self.assertAlmostEqual(metrics["raw_auc"], 0.66)
        self.assertAlmostEqual(metrics["raw_auc_gap"], 0.16)
        self.assertAlmostEqual(metrics["raw_balanced_accuracy"], 0.64)
        self.assertTrue(math.isnan(metrics["raw_auc_null_se_approx"]))
        fit.call_args_list[0].kwargs["progress_callback"]({})
        fit.call_args_list[1].kwargs["progress_callback"]({})
        self.assertEqual(phases, ["staleness_audit", "raw_staleness_audit"])


class TestPolicyRoundWarmup(unittest.TestCase):
    def test_resume_changes_length_only_for_unstarted_round(self):
        adaptive_module.start_policy_round_warmup(self.state, cfg=self.cfg)
        cfg = replace(self.cfg, policy_warmup_steps=10)
        before = self.state.to_dict()
        self.assertTrue(adaptive_module.migrate_unstarted_policy_warmup_after_resume(self.state, cfg=cfg))
        expected = {**before, "policy_warmup_protocol": {"steps": 10, "start_factor": .1}}
        self.assertEqual(self.state.to_dict(), expected)
        self.assertFalse(adaptive_module.migrate_unstarted_policy_warmup_after_resume(self.state, cfg=cfg))
        self.assertAlmostEqual(adaptive_module.policy_round_warmup_metrics(self.state, cfg=cfg)["train/round_warmup/lr_scale"], .1)
        for updates in (1, 20):
            state = AdaptiveOmniFoldState.from_dict(before)
            state.policy_warmup_completed_updates = updates
            snapshot = state.to_dict()
            with self.assertRaises(ValueError):
                adaptive_module.migrate_unstarted_policy_warmup_after_resume(state, cfg=cfg)
            self.assertEqual(state.to_dict(), snapshot)

    def setUp(self):
        import yaml
        root = Path(__file__).resolve().parents[4]
        self.payload = yaml.safe_load((root / "config/dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml").read_text())
        self.cfg = resolve_adaptive_config(self.payload["dgpo"])
        self.state = AdaptiveOmniFoldState(reward_round_id=1)

    def test_twenty_accepted_updates_with_resume_and_duplicate_install(self):
        adaptive_module.start_policy_round_warmup(self.state, cfg=self.cfg)
        for index in range(23):
            metrics = adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg)
            self.assertAlmostEqual(metrics["train/round_warmup/lr_scale"], .1 + .9 * min(1., index / 19))
            self.assertEqual(self.state.policy_warmup_completed_updates, min(index, 20))
            adaptive_module.advance_policy_round_warmup(self.state, cfg=self.cfg, accepted=False)
            self.assertEqual(self.state.policy_warmup_completed_updates, min(index, 20))
            self.state = AdaptiveOmniFoldState.from_dict(self.state.to_dict())
            # Repeating the install hook at the same reference cannot restart warmup.
            adaptive_module.start_policy_round_warmup(self.state, cfg=self.cfg)
            self.assertEqual(adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg), metrics)
            adaptive_module.advance_policy_round_warmup(self.state, cfg=self.cfg, accepted=True)
        self.state.reward_round_id = 2
        adaptive_module.start_policy_round_warmup(self.state, cfg=self.cfg)
        self.assertEqual(self.state.policy_warmup_completed_updates, 0)
        self.assertEqual(adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg)["train/round_warmup/lr_scale"], .1)

    def test_disabled_legacy_configs_are_noops_and_invalid_config_fails(self):
        import copy
        from dataclasses import replace
        disabled = replace(self.cfg, policy_warmup_steps=0)
        self.assertEqual(adaptive_module.start_policy_round_warmup(self.state, cfg=disabled), {})
        self.assertEqual(adaptive_module.policy_round_warmup_metrics(self.state, cfg=disabled), {})
        adaptive_module.advance_policy_round_warmup(self.state, cfg=disabled, accepted=True)
        self.assertEqual(self.state.policy_warmup_round_id, -1)
        for key, values in (("policy_warmup_steps", (-1, True, 2.5)),
                            ("policy_warmup_start_factor", (0., -1., 1.1, float("nan"), float("inf")))):
            for value in values:
                payload = copy.deepcopy(self.payload["dgpo"])
                payload["adaptive_omnifold"]["recalibration"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_adaptive_config(payload)

    def test_mismatched_resume_and_missing_install_fail_closed(self):
        from dataclasses import replace
        with self.assertRaises(ValueError):
            adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg)
        self.assertEqual(adaptive_module.start_inherited_round_policy_warmup(self.state, cfg=self.cfg), {})
        with self.assertRaises(ValueError):
            adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg)
        adaptive_module.start_policy_round_warmup(self.state, cfg=self.cfg)
        with self.assertRaises(ValueError):
            adaptive_module.policy_round_warmup_metrics(self.state, cfg=replace(self.cfg, policy_warmup_steps=10))
        self.state.policy_warmup_completed_updates = -1
        with self.assertRaises(ValueError):
            adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg)

    def test_inherited_round_starts_warmup_without_a_new_install(self):
        self.state.last_decision = "new_experiment_from_best"
        metrics = adaptive_module.start_inherited_round_policy_warmup(self.state, cfg=self.cfg)
        self.assertAlmostEqual(metrics["train/round_warmup/lr_scale"], .1)
        self.assertEqual(self.state.policy_warmup_round_id, 1)
        self.assertEqual(self.state.policy_warmup_completed_updates, 0)
        restored = AdaptiveOmniFoldState.from_dict(self.state.to_dict())
        restored.raw_monitor_baseline_pending = False
        restored.last_decision = "raw_improved"
        self.assertEqual(adaptive_module.start_inherited_round_policy_warmup(restored, cfg=self.cfg), {})
        self.assertEqual(
            adaptive_module.policy_round_warmup_metrics(restored, cfg=self.cfg)["train/round_warmup/completed_updates"],
            0.0,
        )

    def test_step0_last_ckpt_after_baseline_still_starts_warmup(self):
        self.state.last_decision = "raw_improved"
        self.assertEqual(
            adaptive_module.start_inherited_round_policy_warmup(self.state, cfg=self.cfg, global_step=30),
            {},
        )
        with self.assertRaises(ValueError):
            adaptive_module.policy_round_warmup_metrics(self.state, cfg=self.cfg)
        metrics = adaptive_module.start_inherited_round_policy_warmup(
            self.state, cfg=self.cfg, global_step=0,
        )
        self.assertAlmostEqual(metrics["train/round_warmup/lr_scale"], .1)
        self.assertEqual(self.state.policy_warmup_round_id, 1)


class TestBestDecayTrustRadius(unittest.TestCase):
    def setUp(self):
        import yaml
        root = Path(__file__).resolve().parents[4]
        self.payload = yaml.safe_load((root / "config/dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml").read_text())
        self.cfg = resolve_adaptive_config(self.payload["dgpo"])
        self.state = AdaptiveOmniFoldState(reward_round_id=1)
        install_adaptive_trust_round(self.state, cfg=self.cfg, raw_auc=.7, raw_auc_se=.001)
        self.shrink(0, distance=float("nan"), **{"staleness/raw_auc_gap": .2})

    def shrink(self, step, distance=0., **overrides):
        diagnostics = {"staleness/raw_improved": 1., "staleness/raw_audit_saturated": 1.,
                       "staleness/raw_best_auc_gap_before": .2,
                       "staleness/raw_auc_gap": .2 - .001 * step / 5, **overrides}
        return adaptive_module.shrink_trust_radius_on_raw_best(
            self.state, cfg=self.cfg, diagnostics=diagnostics,
            current_distance=distance, global_step=step,
        )

    def test_no_shrink_for_baseline_misses_unsaturated_or_duplicate_steps(self):
        self.assertEqual(self.state.trust_best_global_auc_gap, .2)
        self.assertEqual(self.state.trust_best_decay_count, 0)
        for step, overrides in enumerate(({"staleness/raw_auc_gap": .2},
                          {"staleness/raw_auc_gap": .3},
                          {"staleness/raw_auc_gap": float("nan")},
                          {"staleness/raw_audit_saturated": 0}), start=1):
            self.shrink(step, distance=float("nan"), **overrides)
            self.assertEqual(self.state.trust_best_decay_count, 0)
        self.shrink(5)
        self.assertAlmostEqual(self.state.trust_current_delta, .09)
        self.shrink(5, distance=float("nan"))
        self.assertEqual(self.state.trust_best_decay_count, 1)

    def test_global_record_survives_round_reset_and_local_recovery_does_not_shrink(self):
        self.shrink(5, **{"staleness/raw_auc_gap": .1})
        self.assertEqual(self.state.trust_best_global_auc_gap, .1)
        self.state.install(baseline_auc_gap=.25, cfg=self.cfg, epoch=1, round_id=2)
        install_adaptive_trust_round(self.state, cfg=self.cfg, raw_auc=.75, raw_auc_se=.001)
        self.state = AdaptiveOmniFoldState.from_dict(self.state.to_dict())
        # Both are local-round improvements, but neither beats the global record.
        for step, gap in ((10, .25), (15, .15), (20, .1)):
            diag = self.shrink(step, distance=float("nan"), **{"staleness/raw_auc_gap": gap})
            self.assertEqual(diag["reference_trust/best_decay/new_best"], 0.)
            self.assertEqual(self.state.trust_best_global_auc_gap, .1)
            self.assertEqual(self.state.trust_best_decay_count, 1)
        self.shrink(25, **{"staleness/raw_auc_gap": .09})
        self.assertEqual(self.state.trust_best_decay_count, 2)
        self.assertAlmostEqual(self.state.trust_current_delta, .081)
        self.assertEqual(self.state.trust_best_global_auc_gap, .09)

    def test_floor_serialization_and_reference_install_do_not_reset_or_double_decay(self):
        for count in range(1, 25):
            self.shrink(count * 5)
            target = max(.02, .1 * .9 ** count)
            self.assertAlmostEqual(self.state.trust_current_delta, target)
            self.state = AdaptiveOmniFoldState.from_dict(self.state.to_dict())
            clamp_fixed_trust_radius_after_resume(self.state, cfg=self.cfg)
            self.state.install(baseline_auc_gap=.2, cfg=self.cfg, epoch=count, round_id=count + 1)
            # Baseline registration and new references are not improvements.
            for _ in range(2):
                install_adaptive_trust_round(self.state, cfg=self.cfg, raw_auc=.7, raw_auc_se=.001)
                self.assertAlmostEqual(self.state.trust_current_delta, target)
                self.assertEqual(self.state.trust_best_decay_count, count)

    def test_feasible_shrink_preserved_on_resume_then_applied_after_recenter(self):
        diag = self.shrink(5, distance=.085)
        self.assertAlmostEqual(diag["reference_trust/best_decay/target"], .09)
        self.assertAlmostEqual(self.state.trust_current_delta, .085 / .9)
        self.assertEqual(diag["reference_trust/best_decay/feasibility_limited"], 1.)
        self.state = AdaptiveOmniFoldState.from_dict(self.state.to_dict())
        clamp_fixed_trust_radius_after_resume(self.state, cfg=self.cfg)
        install_adaptive_trust_round(self.state, cfg=self.cfg, raw_auc=.65, raw_auc_se=.001)
        self.assertAlmostEqual(self.state.trust_current_delta, .085 / .9)
        before = self.state.to_dict()
        preview = install_adaptive_trust_round(self.state, cfg=self.cfg, raw_auc=.65,
                                              raw_auc_se=.001, round_id=2, commit=False)
        self.assertEqual(self.state.to_dict(), before)
        self.assertAlmostEqual(preview["reference_trust/delta"], .09)
        self.state.install(baseline_auc_gap=.15, cfg=self.cfg, epoch=1, round_id=2)
        install_adaptive_trust_round(self.state, cfg=self.cfg, raw_auc=.65, raw_auc_se=.001)
        self.assertAlmostEqual(self.state.trust_current_delta, .09)

    def test_near_boundary_never_expands_or_excludes_current_policy(self):
        diag = self.shrink(5, distance=.099)
        self.assertEqual(self.state.trust_current_delta, .1)
        self.assertEqual(diag["reference_trust/best_decay/shrink_applied"], 0.)
        self.assertEqual(self.state.trust_best_decay_count, 1)
        for distance in (float("nan"), float("inf"), -.01, .1, .2):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                self.shrink(10, distance=distance)
            self.assertEqual(self.state.trust_best_decay_count, 1)

    def test_invalid_config_and_changed_resume_protocol_fail_closed(self):
        import copy
        for factor in (0., 1., -1., float("nan"), float("inf")):
            config = copy.deepcopy(self.payload["dgpo"])
            config["reference_trust"]["adaptive_boundary"]["best_decay_factor"] = factor
            with self.assertRaises(ValueError):
                resolve_adaptive_config(config)
        config = copy.deepcopy(self.payload["dgpo"])
        config["reference_trust"]["adaptive_boundary"]["fixed_probe_per_reward_round"] = False
        with self.assertRaises(ValueError):
            resolve_adaptive_config(config)
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            clamp_fixed_trust_radius_after_resume(self.state, cfg=replace(self.cfg, trust_best_decay_factor=.8))
        cold = AdaptiveOmniFoldState()
        self.assertEqual(clamp_fixed_trust_radius_after_resume(cold, cfg=self.cfg,
                                                             initialize_round_decay=False), {})
        with self.assertRaisesRegex(ValueError, "fresh reference"):
            clamp_fixed_trust_radius_after_resume(cold, cfg=self.cfg)
        # A live radius without a best-decay schedule is a mid-run protocol switch.
        switched = AdaptiveOmniFoldState(reward_round_id=3, baseline_auc_gap=.1,
                                         trigger_threshold=.02, trust_current_delta=.05)
        with self.assertRaisesRegex(ValueError, "fresh reference"):
            clamp_fixed_trust_radius_after_resume(switched, cfg=self.cfg)
        self.assertIsNone(switched.trust_best_decay_protocol)

    def test_best_point_restart_opts_inherited_reference_in_at_age_zero(self):
        # Installed, calibrated reference whose trust schedule was reset by a
        # best-point / pinned-classifier restart: no live radius, no global record.
        self.shrink(5)
        restart = AdaptiveOmniFoldState(
            reward_round_id=self.state.reward_round_id,
            baseline_auc_gap=self.state.baseline_auc_gap,
            trigger_threshold=self.state.trigger_threshold,
        )
        self.assertFalse(math.isfinite(restart.trust_current_delta))
        diag = clamp_fixed_trust_radius_after_resume(restart, cfg=self.cfg)
        self.assertEqual(diag["reference_trust/best_decay/inherited_reference_age_zero"], 1.)
        self.assertEqual(restart.trust_best_decay_count, 0)
        self.assertEqual(restart.trust_best_decay_round_id, self.state.reward_round_id)
        self.assertEqual(restart.trust_best_decay_protocol, self.state.trust_best_decay_protocol)
        self.assertAlmostEqual(restart.trust_current_delta, self.cfg.trust_delta_max)
        # Idempotent across serialization and repeated resumes; no second reset.
        restored = AdaptiveOmniFoldState.from_dict(restart.to_dict())
        for _ in range(3):
            clamp_fixed_trust_radius_after_resume(restored, cfg=self.cfg)
        self.assertEqual(restored.to_dict(), restart.to_dict())
        # The fresh schedule then shrinks from the initial radius on a new global best.
        self.state = restored
        self.shrink(5)
        self.assertEqual(self.state.trust_best_decay_count, 1)
        self.assertAlmostEqual(self.state.trust_current_delta, .09)


class TestRoundDecayTrustRadius(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _config(
            trust_boundary_enabled=True, trust_radius_mode="round_decay",
            trust_distance="vp_path_kl", trust_delta_max=0.1,
            trust_delta_floor=0.02,
        )

    def install_radius(self, state, **kwargs):
        return install_adaptive_trust_round(
            state, cfg=self.cfg, raw_auc=0.7, raw_auc_se=0.001, **kwargs,
        )

    def test_bootstrap_and_only_new_reference_rounds_advance_schedule(self) -> None:
        state = AdaptiveOmniFoldState()
        self.assertEqual(clamp_fixed_trust_radius_after_resume(
            state, cfg=self.cfg, initialize_round_decay=False,
        ), {})
        self.assertEqual(state.trust_radius_decay_step, -1)
        for step in range(22):
            # Preview before atomic reference install must not change state.
            before = state.to_dict()
            expected = max(0.02, 0.1 * 0.9 ** step)
            preview = self.install_radius(state, round_id=step + 1, commit=False)
            self.assertEqual(state.to_dict(), before)
            self.assertAlmostEqual(preview["reference_trust/delta"], expected)
            state.install(baseline_auc_gap=0.2, cfg=self.cfg, epoch=step, round_id=step + 1)
            self.install_radius(state)
            self.assertEqual(state.trust_radius_decay_step, step)
            self.assertAlmostEqual(state.trust_current_delta, expected)
            # Repeated baseline installs/monitor callbacks do not decay twice.
            self.install_radius(state)
            self.assertEqual(state.trust_radius_decay_step, step)
            self.assertAlmostEqual(state.trust_current_delta, expected)
        self.assertEqual(state.trust_current_delta, 0.02)

    def test_legacy_resume_starts_at_zero_then_preserves_schedule_and_probe(self) -> None:
        state = AdaptiveOmniFoldState.from_dict({
            "reward_round_id": 12, "trust_current_delta": 0.05,
            "recalibration_count": 11, "trust_probe_payload": {"sentinel": 4},
            "trust_probe_round_id": 12,
        })
        clamp_fixed_trust_radius_after_resume(state, cfg=self.cfg)
        self.assertEqual(state.trust_radius_decay_step, 0)
        self.assertEqual(state.trust_current_delta, 0.1)
        self.assertEqual(state.trust_probe_payload, {"sentinel": 4})
        state.install(baseline_auc_gap=0.15, cfg=self.cfg, epoch=20, round_id=13)
        self.install_radius(state)
        self.assertAlmostEqual(state.trust_current_delta, 0.09)
        restored = AdaptiveOmniFoldState.from_dict(state.to_dict())
        for _ in range(3):
            clamp_fixed_trust_radius_after_resume(restored, cfg=self.cfg)
        self.assertEqual(restored.trust_radius_decay_step, 1)
        self.assertAlmostEqual(restored.trust_current_delta, 0.09)
        # Policy rollback and audit-baseline invalidation do not reset age.
        restored.raw_plateau_rollbacks += 1
        restored.invalidate_audit_baseline(reason="test_rollback")
        self.install_radius(restored)
        self.assertEqual(restored.trust_radius_decay_step, 1)
        restored.install(baseline_auc_gap=0.15, cfg=self.cfg, epoch=21, round_id=14)
        self.install_radius(restored)
        self.assertAlmostEqual(restored.trust_current_delta, 0.081)

    def test_resume_rejects_inconsistent_state_or_changed_schedule(self) -> None:
        state = AdaptiveOmniFoldState(reward_round_id=5, trust_current_delta=0.05)
        clamp_fixed_trust_radius_after_resume(state, cfg=self.cfg)
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            clamp_fixed_trust_radius_after_resume(
                state, cfg=replace(self.cfg, trust_round_decay_factor=0.8),
            )
        with self.assertRaisesRegex(ValueError, "cannot rewind"):
            self.install_radius(state, round_id=4)
        state.trust_current_delta = 0.05
        with self.assertRaisesRegex(ValueError, "does not match"):
            clamp_fixed_trust_radius_after_resume(state, cfg=self.cfg)
        legacy = AdaptiveOmniFoldState(trust_current_delta=0.2)
        with self.assertRaisesRegex(ValueError, "shrinking an installed"):
            clamp_fixed_trust_radius_after_resume(legacy, cfg=self.cfg)
        self.assertEqual(legacy.trust_radius_decay_step, -1)

    def test_rejects_invalid_decay_and_conflicting_controllers(self) -> None:
        for factor in (0.0, 1.0, -0.1, float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                _config(trust_boundary_enabled=True, trust_radius_mode="round_decay",
                        trust_round_decay_factor=factor)
        for flag in ("trust_empirical_radius_enabled", "trust_cross_round_nonexpanding",
                     "trust_policy_lr_scaling_enabled", "trust_round_acceptance_enabled",
                     "trust_trajectory_search_enabled", "trust_extragradient_enabled"):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                _config(trust_boundary_enabled=True, trust_radius_mode="round_decay",
                        **{flag: True})

    def test_atomic_refits_decay_only_after_success(self) -> None:
        cfg = replace(self.cfg, acceptance_audit_enabled=False)
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.2, cfg=cfg, epoch=0, round_id=7)
        state.trust_current_delta = 0.05
        clamp_fixed_trust_radius_after_resume(state, cfg=cfg)
        spec = EventPackingSpec({
            "x": (2, 3), "x_mask": (2, 1),
            "conditions": (1, 2), "conditions_mask": (1,),
        })
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width), truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4), packing_spec=spec,
        )
        reference, policy = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        snapshot = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        source = SimpleNamespace(model_builder=object(), replace_stack=mock.Mock())
        stack = mock.Mock(spec=["to", "eval", "assert_frozen"])
        stack.to.return_value = stack
        stack.eval.return_value = stack
        expected_step = 0
        for outcome in ("fit_failed", "closure_rejected", "install_failed", "accepted", "accepted"):
            with self.subTest(outcome=outcome):
                previous_reference = {k: v.clone() for k, v in reference.state_dict().items()}
                fit_result = SimpleNamespace(
                    diagnostics=(
                        SimpleNamespace(saturated=True, validation_auc=0.58),
                        SimpleNamespace(saturated=True, validation_auc=(
                            0.55 if outcome == "closure_rejected" else 0.50
                        )),
                    ), iterations=1,
                )
                source.replace_stack.reset_mock()
                source.replace_stack.side_effect = (
                    RuntimeError("install failed") if outcome == "install_failed" else None
                )
                with mock.patch.object(
                    adaptive_module, "fit_residual_ratio_stack", return_value=fit_result,
                    side_effect=(RuntimeError("did not saturate") if outcome == "fit_failed" else None),
                ), mock.patch.object(
                    adaptive_module.FrozenResidualRatioReward, "from_fit_result", return_value=stack,
                ):
                    args = dict(
                        state=state, cfg=cfg, reward_source=source,
                        round_ref_model=reference, policy_snapshot_state_dict=snapshot,
                        fit_pool=pool, score_pool=pool, epoch=5,
                        device=torch.device("cpu"), world_size=1,
                    )
                    if outcome == "install_failed":
                        with self.assertRaisesRegex(RuntimeError, "install failed"):
                            run_adaptive_refit(**args)
                    else:
                        diagnostics = run_adaptive_refit(**args)
                        self.assertEqual(diagnostics["omnifold/accepted"], float(outcome == "accepted"))
                if outcome == "accepted":
                    expected_step += 1
                    self.assertEqual(diagnostics["reference_trust/round_decay/step"], expected_step)
                else:
                    for key, value in previous_reference.items():
                        torch.testing.assert_close(reference.state_dict()[key], value)
                    if outcome != "install_failed":
                        source.replace_stack.assert_not_called()
                self.assertEqual(state.trust_radius_decay_step, expected_step)
                self.assertEqual(state.reward_round_id, 7 + expected_step)
                self.assertAlmostEqual(state.trust_current_delta, 0.1 * 0.9 ** expected_step)


class TestAtomicAdaptiveInstall(unittest.TestCase):
    def test_trajectory_plateau_rolls_back_each_direction_and_stops_after_three(
        self,
    ) -> None:
        cfg = _config(
            acceptance_audit_enabled=False,
            residual_min_auc_gain=0.01,
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_round_acceptance_enabled=True,
            trust_trajectory_search_enabled=True,
            trust_failed_direction_patience=3,
            raw_audit_enabled=True,
        )
        state = AdaptiveOmniFoldState(
            reward_round_id=2,
            trust_current_raw_auc_gap=0.08,
            trust_current_raw_auc_se=0.001,
            trust_current_delta=1.0e-4,
        )
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        round_ref = torch.nn.Linear(2, 2)
        snapshot = {
            key: value.detach().clone()
            for key, value in torch.nn.Linear(2, 2).state_dict().items()
        }

        class _Stack:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def assert_frozen(self):
                return None

        class _Source:
            model_builder = object()
            is_installed = True

            def replace_stack(self, *_args, **_kwargs):
                raise AssertionError("plateau candidate must not be installed")

        fit_result = SimpleNamespace(
            diagnostics=(
                # 0.579 is only a 0.001 gap improvement over the incumbent;
                # the z=1.96 combined-SE gate classifies it as a plateau.
                SimpleNamespace(saturated=True, validation_auc=0.579),
                SimpleNamespace(saturated=True, validation_auc=0.501),
            ),
            iterations=1,
        )
        with (
            mock.patch.object(
                adaptive_module,
                "fit_residual_ratio_stack",
                return_value=fit_result,
            ),
            mock.patch.object(
                adaptive_module.FrozenResidualRatioReward,
                "from_fit_result",
                return_value=_Stack(),
            ),
            mock.patch.object(
                adaptive_module,
                "auc_null_standard_error",
                return_value=0.001,
            ),
        ):
            outcomes = [
                run_adaptive_refit(
                    state=state,
                    cfg=cfg,
                    reward_source=_Source(),
                    round_ref_model=round_ref,
                    policy_snapshot_state_dict=snapshot,
                    fit_pool=pool,
                    score_pool=pool,
                    epoch=epoch,
                    device=torch.device("cpu"),
                    world_size=1,
                )
                for epoch in (5, 7, 9)
            ]

        self.assertEqual(
            [
                item["reference_trust/round_acceptance/rollback_required"]
                for item in outcomes
            ],
            [1.0, 1.0, 1.0],
        )
        self.assertEqual(
            [
                item["reference_trust/round_acceptance/stop_requested"]
                for item in outcomes
            ],
            [0.0, 0.0, 1.0],
        )
        self.assertEqual(state.trust_failed_direction_streak, 3)
        self.assertEqual(state.trust_round_rollbacks, 3)
        self.assertAlmostEqual(state.trust_current_delta, 1.25e-5)

    def test_significant_round_auc_regression_rejects_and_requests_rollback(
        self,
    ) -> None:
        cfg = _config(
            acceptance_audit_enabled=False,
            residual_min_auc_gain=0.01,
            trust_boundary_enabled=True,
            trust_empirical_radius_enabled=True,
            trust_round_acceptance_enabled=True,
        )
        state = AdaptiveOmniFoldState(
            reward_round_id=2,
            trust_current_raw_auc_gap=0.08,
            trust_current_raw_auc_se=0.001,
            trust_current_delta=1.0e-4,
        )
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        round_ref = torch.nn.Linear(2, 2)
        snapshot = {
            key: value.detach().clone()
            for key, value in torch.nn.Linear(2, 2).state_dict().items()
        }

        class _Stack:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def assert_frozen(self):
                return None

        class _Source:
            model_builder = object()
            is_installed = True

            def replace_stack(self, *_args, **_kwargs):
                raise AssertionError("regressed candidate must not be installed")

        fit_result = SimpleNamespace(
            diagnostics=(
                SimpleNamespace(saturated=True, validation_auc=0.60),
                SimpleNamespace(saturated=True, validation_auc=0.501),
            ),
            iterations=1,
        )
        with (
            mock.patch.object(
                adaptive_module,
                "fit_residual_ratio_stack",
                return_value=fit_result,
            ),
            mock.patch.object(
                adaptive_module.FrozenResidualRatioReward,
                "from_fit_result",
                return_value=_Stack(),
            ),
            mock.patch.object(
                adaptive_module,
                "auc_null_standard_error",
                return_value=0.001,
            ),
        ):
            diagnostics = run_adaptive_refit(
                state=state,
                cfg=cfg,
                reward_source=_Source(),
                round_ref_model=round_ref,
                policy_snapshot_state_dict=snapshot,
                fit_pool=pool,
                score_pool=pool,
                epoch=5,
                device=torch.device("cpu"),
                world_size=1,
            )

        self.assertEqual(diagnostics["omnifold/accepted"], 0.0)
        self.assertEqual(
            diagnostics[
                "reference_trust/round_acceptance/rollback_required"
            ],
            1.0,
        )
        self.assertEqual(state.reward_round_id, 2)
        self.assertEqual(state.trust_round_regressions, 1)
        self.assertEqual(state.trust_round_rollbacks, 1)
        self.assertAlmostEqual(state.trust_current_delta, 5.0e-5)

    def test_resume_pairing_rejects_a_different_round_reference(self) -> None:
        round_ref = torch.nn.Linear(2, 2)
        digest = state_dict_sha256(round_ref)
        source = SimpleNamespace(
            reward_round_id=2,
            reference_kind="state_dict_sha256",
            policy_reference_sha256=digest,
        )
        state = AdaptiveOmniFoldState(reward_round_id=2)
        validate_adaptive_pairing(
            reward_source=source,
            state=state,
            round_ref_model=round_ref,
            checkpoint={
                "dgpo_reward_round_id": 2,
                "dgpo_round_ref_sha256": digest,
            },
            where="test",
        )
        source.policy_reference_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "round-reference"):
            validate_adaptive_pairing(
                reward_source=source,
                state=state,
                round_ref_model=round_ref,
                where="test",
            )

    def test_initial_bootstrap_skips_missing_incumbent_and_installs_candidate(
        self,
    ) -> None:
        cfg = _config(
            acceptance_audit_enabled=False,
            residual_min_auc_gain=0.01,
        )
        state = AdaptiveOmniFoldState()
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        round_ref = torch.nn.Linear(2, 2)
        policy = torch.nn.Linear(2, 2)
        snapshot = {
            key: value.detach().clone() for key, value in policy.state_dict().items()
        }

        class _Stack:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def assert_frozen(self):
                return None

        class _Source:
            model_builder = object()

            def __init__(self):
                self.is_installed = False
                self.installed = None

            @property
            def frozen_reward(self):
                raise AssertionError("cold bootstrap must not score an incumbent")

            def replace_stack(self, stack, **kwargs):
                self.installed = (stack, kwargs)
                self.is_installed = True

        source = _Source()
        fit_result = SimpleNamespace(
            diagnostics=(
                SimpleNamespace(saturated=True, validation_auc=0.58),
                SimpleNamespace(saturated=True, validation_auc=0.51),
            ),
            iterations=2,
        )
        progress_events: list[tuple[str, int]] = []

        def _fit_side_effect(**kwargs):
            kwargs["progress_callback"]({"iteration": 1.0, "step": 10.0})
            return fit_result

        with (
            mock.patch.object(
                adaptive_module,
                "fit_residual_ratio_stack",
                side_effect=_fit_side_effect,
            ),
            mock.patch.object(
                adaptive_module.FrozenResidualRatioReward,
                "from_fit_result",
                return_value=_Stack(),
            ),
            mock.patch.object(
                adaptive_module,
                "score_reward_on_pool",
                return_value=torch.zeros(32, 1),
            ) as score_reward,
            mock.patch.object(
                adaptive_module,
                "fit_fresh_audit",
                side_effect=AssertionError("acceptance audit must be disabled"),
            ) as fresh_audit,
        ):
            diagnostics = run_adaptive_refit(
                state=state,
                cfg=cfg,
                reward_source=source,
                round_ref_model=round_ref,
                policy_snapshot_state_dict=snapshot,
                fit_pool=pool,
                score_pool=pool,
                epoch=-1,
                device=torch.device("cpu"),
                world_size=1,
                progress_callback=lambda phase, row: progress_events.append(
                    (phase, int(row["step"]))
                ),
            )

        self.assertEqual(diagnostics["omnifold/accepted"], 1.0)
        self.assertEqual(diagnostics["omnifold/initial_bootstrap"], 1.0)
        self.assertEqual(
            diagnostics["omnifold/accept_reason"],
            "candidate reached saturated cross-fit residual closure; "
            "fresh acceptance audit disabled",
        )
        self.assertEqual(state.reward_round_id, 1)
        self.assertIsNotNone(source.installed)
        self.assertEqual(source.installed[1]["round_id"], 1)
        score_reward.assert_not_called()
        fresh_audit.assert_not_called()
        self.assertEqual(
            progress_events,
            [("residual_reward", 10)],
        )
        for key, expected in snapshot.items():
            torch.testing.assert_close(round_ref.state_dict()[key], expected)

    def test_accepted_stack_moves_reward_and_round_reference_together(self) -> None:
        cfg = replace(_config(trust_boundary_enabled=True, warm_start_iterations=(1,)),
                      warm_start_from_iteration_one=True)
        cfg.fit.update(warm_start_min_epochs_per_fold=10,
                       validation_interval_epochs=2., validation_patience_epochs=10.,
                       later_iteration_learning_rate=5e-5,
                       later_iteration_train_mode='last_decoder_and_output')
        state = AdaptiveOmniFoldState()
        state.install(baseline_auc_gap=0.08, cfg=cfg, epoch=-1, round_id=0)
        spec = EventPackingSpec(
            {
                "x": (2, 3),
                "x_mask": (2, 1),
                "conditions": (1, 2),
                "conditions_mask": (1,),
            }
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=torch.randn(32, spec.width),
            truth=torch.randn(32, 4),
            candidates=torch.randn(32, 1, 4),
            packing_spec=spec,
        )
        round_ref = torch.nn.Linear(2, 2)
        policy = torch.nn.Linear(2, 2)
        snapshot = {
            key: value.detach().clone() for key, value in policy.state_dict().items()
        }

        class _Stack:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def assert_frozen(self):
                return None

        class _Source:
            model_builder = object()
            frozen_reward = object()

            def __init__(self):
                self.installed = None

            def replace_stack(self, stack, **kwargs):
                self.installed = (stack, kwargs)

        source = _Source()
        warm_cache = {"protocol": {"seed": cfg.seed}, "models": []}
        source.is_installed = True
        source.frozen_reward = SimpleNamespace(
            packing_spec=spec, warm_start_state=warm_cache,
        )
        fit_result = SimpleNamespace(
            diagnostics=(
                SimpleNamespace(saturated=True, validation_auc=0.58),
                SimpleNamespace(saturated=True, validation_auc=0.501),
            ),
            iterations=1,
        )
        candidate_audit = {
            "weighted_auc_gap": 0.01,
            "audit_observed_auc_gap": 0.01,
            "audit_balanced_accuracy": 0.50,
            "audit_saturated": 1.0,
        }
        cheap_baseline_audit = {
            "weighted_auc_gap": 0.03,
            "audit_observed_auc_gap": 0.03,
            "audit_balanced_accuracy": 0.50,
            "audit_saturated": 1.0,
        }
        audit_results = iter((candidate_audit, cheap_baseline_audit))
        progress_events: list[tuple[str, int]] = []

        def _fit_side_effect(**kwargs):
            self.assertEqual(kwargs["warm_start_iterations"], (1,))
            self.assertTrue(kwargs['warm_start_from_iteration_one'])
            self.assertEqual(kwargs['later_iteration_learning_rate'], 5e-5)
            self.assertEqual(kwargs['later_iteration_train_mode'], 'last_decoder_and_output')
            self.assertIs(kwargs["warm_start_state"], warm_cache)
            self.assertEqual(kwargs["crossfit_seed"], cfg.seed)
            self.assertEqual(kwargs["fit_config"].weight_decay, 0.0005)
            self.assertEqual(kwargs["warm_start_min_epochs_per_fold"], 10)
            self.assertEqual(kwargs["validation_interval_epochs"], 2.)
            self.assertEqual(kwargs["validation_patience_epochs"], 10.)
            self.assertNotIn("resume_state", kwargs)
            kwargs["progress_callback"]({"iteration": 1.0, "step": 10.0})
            return fit_result

        def _audit_side_effect(**kwargs):
            self.assertNotIn("warm_start_state", kwargs)
            self.assertNotIn("resume_state", kwargs)
            kwargs["progress_callback"]({"iteration": 1.0, "step": 10.0})
            return next(audit_results)

        with (
            mock.patch.object(
                adaptive_module,
                "fit_residual_ratio_stack",
                side_effect=_fit_side_effect,
            ),
            mock.patch.object(
                adaptive_module.FrozenResidualRatioReward,
                "from_fit_result",
                return_value=_Stack(),
            ),
            mock.patch.object(
                adaptive_module,
                "score_reward_on_pool",
                side_effect=lambda _stack, scored_pool, **_kwargs: torch.zeros(
                    scored_pool.n_events, 1
                ),
            ),
            mock.patch.object(
                adaptive_module,
                "fit_fresh_audit",
                side_effect=_audit_side_effect,
            ),
        ):
            diagnostics = run_adaptive_refit(
                state=state,
                cfg=cfg,
                reward_source=source,
                round_ref_model=round_ref,
                policy_snapshot_state_dict=snapshot,
                fit_pool=pool,
                score_pool=pool,
                baseline_pool=pool.prefix(30),
                epoch=1,
                device=torch.device("cpu"),
                world_size=1,
                progress_callback=lambda phase, row: progress_events.append(
                    (phase, int(row["step"]))
                ),
            )
        self.assertEqual(diagnostics["omnifold/accepted"], 1.0)
        self.assertEqual(state.reward_round_id, 1)
        self.assertAlmostEqual(state.baseline_auc_gap, 0.03)
        self.assertAlmostEqual(state.trigger_threshold, 0.04)
        self.assertTrue(math.isfinite(state.trust_current_delta))
        self.assertAlmostEqual(
            diagnostics["reference_trust/raw_auc"], 0.58
        )
        self.assertEqual(diagnostics["omnifold/baseline/probe_events"], 30.0)
        self.assertIsNotNone(source.installed)
        self.assertEqual(
            progress_events,
            [
                ("residual_reward", 10),
                ("acceptance_audit", 10),
                ("baseline_audit", 10),
            ],
        )
        self.assertEqual(source.installed[1]["round_id"], 1)
        for key, expected in snapshot.items():
            torch.testing.assert_close(round_ref.state_dict()[key], expected)


if __name__ == "__main__":
    unittest.main()
