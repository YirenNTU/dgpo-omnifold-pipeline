"""Focused tests for the standalone Ztautau OmniFold integration."""

from __future__ import annotations

import json
import copy
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow.parquet as pq
import torch
from torch import Tensor, nn

from evenet.dataset.preprocess import flatten_dict
from evenet.network.body.adapter import Adapter
from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio as evenet_ratio_module
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterRatioClassifier,
    EvenetAdapterModelBuilder,
    FrozenResidualRatioReward,
    PeriodicPairAuditClassifier,
    fit_residual_ratio_stack,
    pack_event_inputs,
    periodic_tau_pair_features,
    unpack_event_inputs,
)
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldPool
from RL.DGPO_neutrino.omnifold_ztautau import dgpo_reward as dgpo_reward_module
from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import (
    REWARD_CHECKPOINT_KEY,
    ZtautauOmniFoldReward,
    load_ztautau_omnifold_reward,
    sha256_file,
    validate_omnifold_reward_startup,
)
from RL.DGPO_neutrino.omnifold_ztautau import stage as ztautau_stage
from RL.DGPO_neutrino.omnifold_ztautau.stage import load_pool
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import (
    ConditionalRatioMLP,
    RatioFitConfig,
    _EpochShuffleBatcher,
    _uses_fast_ratio_learning_rate,
    fit_density_ratio,
)
from RL.DGPO_neutrino.rewards import RewardAggregator


class _IdentityNormalizer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return x if mask is None else x * mask


class _FakeGlobalEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 8)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.projection(x) * mask


class _FakePET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.use_adapter = True
        self.num_layers = 1
        self.projection_dim = 8
        self.feature_embedding = nn.Linear(4, 8)
        self.transformer_blocks = nn.ModuleList([nn.Identity()])
        self.adapters = nn.ModuleList(
            [Adapter(8, bottleneck=4, dropout=0.0)]
        )
        self.last_attn_mask: Tensor | None = None

    def forward(
        self,
        *,
        input_features: Tensor,
        input_points: Tensor,
        mask: Tensor,
        time: Tensor,
        attn_mask: Tensor | None,
        time_masking: Tensor,
        adapters: nn.ModuleList | None,
    ) -> Tensor:
        del input_points, time, time_masking
        self.last_attn_mask = attn_mask
        encoded = self.feature_embedding(input_features)
        for adapter in self.adapters if adapters is None else adapters:
            encoded = adapter(encoded)
        return encoded * mask


class _FakeZtautauBackbone(nn.Module):
    invisible_input_dim = 2
    sequential_input_dim = 4
    invisible_padding = 0
    local_feature_indices = [0, 1]

    def __init__(self) -> None:
        super().__init__()
        self.sequential_normalizer = _IdentityNormalizer(4)
        self.invisible_normalizer = _IdentityNormalizer(2)
        self.global_normalizer = _IdentityNormalizer(2)
        self.GroupedSequentialEmbedding = nn.Linear(4, 4)
        self.InvisibleInputProjector = nn.Linear(2, 4)
        self.GlobalEmbedding = _FakeGlobalEmbedding()
        self.PET = _FakePET()
        self.network_cfg = SimpleNamespace(
            Body=SimpleNamespace(
                PET=SimpleNamespace(hidden_dim=8, adapter_bottleneck=4),
                GlobalEmbedding=SimpleNamespace(hidden_dim=8),
                ObjectEncoder=SimpleNamespace(num_attention_heads=2),
            ),
            Classification=SimpleNamespace(
                hidden_dim=8,
                num_classification_layers=1,
                num_attention_heads=2,
                dropout=0.0,
            ),
            TruthGeneration=SimpleNamespace(max_position_length=8),
        )

    def project_sequential_inputs(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.GroupedSequentialEmbedding(x) * mask

    def project_invisible_inputs(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.InvisibleInputProjector(x) * mask


def _event_batch(batch_size: int = 3) -> dict[str, Tensor]:
    return {
        "x": torch.randn(batch_size, 5, 4),
        "x_mask": torch.ones(batch_size, 5, 1, dtype=torch.bool),
        "conditions": torch.randn(batch_size, 2),
        "conditions_mask": torch.ones(batch_size, dtype=torch.bool),
    }


def _event_batch_with_pair_context(batch_size: int = 3) -> dict[str, Tensor]:
    batch = _event_batch(batch_size)
    ones = torch.ones(batch_size)
    zeros = torch.zeros(batch_size)
    batch.update(
        {
            "lead_a_visible_px": ones,
            "lead_a_visible_py": zeros,
            "lead_a_visible_pz": zeros,
            "lead_b_visible_px": -ones,
            "lead_b_visible_py": zeros,
            "lead_b_visible_pz": zeros,
        }
    )
    return batch


class TestRawMonitorWarmStart(unittest.TestCase):
    def _ready_fit(self, cache, *, readiness=None, short_result=False, steps=None):
        class NullRatio(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(.25))

            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1]) * 0.

        condition = torch.arange(96, dtype=torch.float32)[:, None]
        sample = torch.zeros(96, 4)
        config = RatioFitConfig(
            steps=steps, batch_size=16, drop_last_batch=True, weight_decay=0.,
            sampling="independent_epoch_shuffle", validation_interval_steps=1,
            validation_patience_evaluations=2, restore_best=True,
        )
        observed = []

        def train(*args, **kwargs):
            observed.append(args[7])
            diag = fit_density_ratio(*args, **kwargs)
            return evenet_ratio_module.replace(diag, steps_completed=0) if short_result else diag

        with mock.patch("RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio", side_effect=train):
            result = evenet_ratio_module.fit_independent_evenet_audit(
                model_factory=NullRatio, data_condition=condition, data_sample=sample,
                gen_condition=condition, gen_sample=sample[:, None, :],
                gen_weight=torch.ones(96, 1), fit_config=config, seed=123,
                reuse_early_stop_for_audit=True, identity_split_seed=42,
                warm_start_cache=cache,
                training_readiness=readiness or {"cold_start_min_epochs": 3, "warm_start_min_epochs": 1},
            )
        return result, observed[0]

    def test_readiness_uses_actual_cold_and_certified_warm_event_epochs(self):
        cache = {}
        cold, cold_cfg = self._ready_fit(cache)
        self.assertFalse(cold.warm_started)
        self.assertTrue(cold.training_ready)
        epoch_steps = cold.fit_events // cold_cfg.batch_size
        self.assertEqual(cold.training_steps_per_epoch, epoch_steps)
        self.assertEqual(cold_cfg.min_steps, 3 * epoch_steps)
        self.assertGreaterEqual(cold.fit_diagnostics.steps_completed, cold_cfg.min_steps + 1)
        self.assertTrue(cold_cfg.require_saturation)
        # A trained null classifier is allowed: there is no forced AUC threshold.
        self.assertEqual(cold.auc, .5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "certified_monitor.pt"
            torch.save(cache, path)
            restored = torch.load(path, weights_only=False)
        warm, warm_cfg = self._ready_fit(restored)
        self.assertTrue(warm.warm_started)
        self.assertTrue(warm.training_ready)
        self.assertEqual(warm_cfg.min_steps, epoch_steps)
        self.assertGreaterEqual(warm.fit_diagnostics.steps_completed, epoch_steps + 1)
        self.assertLess(warm.fit_diagnostics.steps_completed, cold.fit_diagnostics.steps_completed)

    def test_legacy_or_changed_monitor_training_protocol_requires_cold_fit(self):
        cache = {}
        self._ready_fit(cache)
        cache.pop("training_policy")
        legacy, config = self._ready_fit(cache)
        self.assertFalse(legacy.warm_started)
        self.assertEqual(config.min_steps, 3 * legacy.training_steps_per_epoch)
        changed, config = self._ready_fit(cache, readiness={"cold_start_min_epochs": 4, "warm_start_min_epochs": 1})
        self.assertFalse(changed.warm_started)
        self.assertEqual(config.min_steps, 4 * changed.training_steps_per_epoch)

    def test_unqualified_monitor_fit_preserves_previous_cache(self):
        cache = {}
        self._ready_fit(cache)
        before = dgpo_reward_module.payload_sha256(cache)
        with self.assertRaisesRegex(RuntimeError, "required training budget"):
            self._ready_fit(cache, short_result=True)
        self.assertEqual(dgpo_reward_module.payload_sha256(cache), before)
        empty = {}
        with self.assertRaisesRegex(ValueError, "min_steps"):
            self._ready_fit(empty, steps=2)
        self.assertEqual(empty, {})

    def test_explicit_new_baseline_recertifies_inherited_weights_with_cold_budget(self):
        cache = {}
        self._ready_fit(cache)
        cache.pop('training_policy')
        cache['recertify_inherited_weights'] = True
        inherited, cfg = self._ready_fit(cache)
        self.assertTrue(inherited.warm_started)
        self.assertTrue(inherited.training_ready)
        self.assertEqual(cfg.min_steps, 3 * inherited.training_steps_per_epoch)
        self.assertNotIn('recertify_inherited_weights', cache)
        warm, cfg = self._ready_fit(cache)
        self.assertTrue(warm.warm_started)
        self.assertEqual(cfg.min_steps, warm.training_steps_per_epoch)

    def test_inherited_monitor_as_is_keeps_weights_on_protocol_change(self):
        cache = {}
        self._ready_fit(cache)
        cache.pop('training_policy')
        cache['inherit_monitor_as_is'] = True
        inherited, cfg = self._ready_fit(cache)
        self.assertTrue(inherited.warm_started)
        self.assertTrue(inherited.training_ready)
        self.assertEqual(cfg.min_steps, inherited.training_steps_per_epoch)
        self.assertNotIn('inherit_monitor_as_is', cache)
        warm, cfg = self._ready_fit(cache)
        self.assertTrue(warm.warm_started)
        self.assertEqual(cfg.min_steps, warm.training_steps_per_epoch)

    def test_inherit_as_is_rejects_changed_split_and_recertify(self):
        cache = {}
        self._ready_fit(cache)
        cache['protocol']['seed'] += 1
        cache['inherit_monitor_as_is'] = True
        with self.assertRaisesRegex(ValueError, 'split protocol mismatch'):
            self._ready_fit(cache)
        cache = {}
        self._ready_fit(cache)
        cache['inherit_monitor_as_is'] = True
        cache['recertify_inherited_weights'] = True
        with self.assertRaisesRegex(ValueError, 'cannot recertify and inherit-as-is'):
            self._ready_fit(cache)

    def test_inherited_recertification_rejects_changed_split(self):
        cache = {}
        self._ready_fit(cache)
        cache['protocol']['seed'] += 1
        cache['recertify_inherited_weights'] = True
        with self.assertRaisesRegex(ValueError, 'split protocol mismatch'):
            self._ready_fit(cache)

    def test_invalid_monitor_training_readiness_is_rejected(self):
        for settings in ({}, {"cold_start_min_epochs": True, "warm_start_min_epochs": 1},
                         {"cold_start_min_epochs": float("nan"), "warm_start_min_epochs": 1},
                         {"cold_start_min_epochs": 1, "warm_start_min_epochs": 2},
                         {"cold_start_min_epochs": 3, "warm_start_min_epochs": 0}):
            with self.assertRaises(ValueError):
                evenet_ratio_module.validate_monitor_training_readiness(settings)

    def _fit(self, cache, condition, *, fail=False, seed=123, split_seed=None):
        observed = []
        def train(model, data_condition, *args, **kwargs):
            # New optimizer is constructed by fit_density_ratio on every call.
            # Only the classifier state is transferred into a new model.
            observed.append((model.weight.detach().clone(), set(data_condition[:, 0].tolist())))
            if fail:
                raise RuntimeError("fit failed")
            model.weight.data.add_(1.0)
            return SimpleNamespace(saturated=True)
        def factory():
            model = nn.Linear(1, 1, bias=False)
            model.weight.data.zero_()
            return model
        with mock.patch("RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio", side_effect=train), \
             mock.patch.object(evenet_ratio_module, "_score_population",
                               side_effect=lambda model, cond, sample, batch: torch.zeros(len(cond))):
            result = evenet_ratio_module.fit_independent_evenet_audit(
                model_factory=factory, data_condition=condition, data_sample=condition,
                gen_condition=condition, gen_sample=condition[:, None, :],
                gen_weight=torch.ones(len(condition), 1),
                fit_config=SimpleNamespace(validation_batch_size=32), seed=seed,
                reuse_early_stop_for_audit=True, warm_start_cache=cache,
                identity_split_seed=split_seed,
            )
        return result, observed[0]

    def test_finetune_survives_serialization_and_pool_reorder_growth(self):
        cache = {}
        condition = torch.arange(200, dtype=torch.float32)[:, None]
        first, (cold_weight, first_fit) = self._fit(cache, condition)
        self.assertFalse(first.warm_started)
        self.assertEqual(cold_weight.item(), 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw-monitor.pt"
            torch.save(cache, path)
            restored = torch.load(path, weights_only=False)
        grown = torch.arange(399, -1, -1, dtype=torch.float32)[:, None]
        second, (warm_weight, second_fit) = self._fit(restored, grown)
        self.assertTrue(second.warm_started)
        self.assertEqual(warm_weight.item(), 1.0)
        self.assertEqual(first_fit, second_fit.intersection(range(200)))
        self.assertEqual(restored["state"]["weight"].device.type, "cpu")
        self.assertEqual(cache["state"]["weight"].item(), 1.0)
        self.assertEqual(restored["state"]["weight"].item(), 2.0)
        # Trust/default audits never receive this cache and remain cold.
        fresh, (weight, _) = self._fit(None, condition)
        self.assertFalse(fresh.warm_started)
        self.assertEqual(weight.item(), 0.0)

    def test_failed_fit_does_not_replace_cache(self):
        cache = {}
        condition = torch.arange(200, dtype=torch.float32)[:, None]
        self._fit(cache, condition)
        before = copy.deepcopy(cache)
        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            self._fit(cache, condition, fail=True)
        self.assertEqual(cache["protocol"], before["protocol"])
        torch.testing.assert_close(cache["state"]["weight"], before["state"]["weight"])

    def test_split_protocol_change_reinitializes(self):
        cache = {}
        condition = torch.arange(200, dtype=torch.float32)[:, None]
        self._fit(cache, condition)
        result, (weight, _) = self._fit(cache, condition, seed=456)
        self.assertFalse(result.warm_started)
        self.assertEqual(weight.item(), 0.0)

    def test_invalid_weights_fail_before_fit(self):
        condition = torch.arange(200, dtype=torch.float32)[:, None]
        for bad in (torch.zeros(2, 1), torch.full((1, 1), float("nan"))):
            cache = {}
            self._fit(cache, condition)
            cache["state"]["weight"] = bad
            with self.assertRaises((ValueError, FloatingPointError)):
                self._fit(cache, condition)

    def test_single_pool_trust_and_raw_share_validation_but_not_weights(self):
        condition = torch.arange(200, dtype=torch.float32)[:, None]
        cache = {}
        _, (_, raw_fit) = self._fit(cache, condition, seed=123, split_seed=42)
        cold, (weight, trust_fit) = self._fit(None, condition.flip(0), seed=789, split_seed=42)
        self.assertEqual(raw_fit, trust_fit)
        self.assertFalse(cold.warm_started)
        self.assertEqual(weight.item(), 0.0)

    def test_disjoint_warm_monitor_never_trains_on_final_audit_identities(self):
        condition = torch.arange(500, dtype=torch.float32)[:, None]
        cache: dict[str, object] = {}
        observed: dict[str, set[float]] = {}

        def train(
            _model,
            data_condition,
            _data_sample,
            _data_weight,
            _gen_condition,
            _gen_sample,
            _gen_weight,
            _fit_config,
            _seed,
            validation,
            **_kwargs,
        ):
            observed["train"] = set(data_condition[:, 0].tolist())
            observed["early_stop"] = set(validation[0][:, 0].tolist())
            return SimpleNamespace(saturated=True)

        def score(_model, score_condition, _sample, _batch):
            observed.setdefault("final_audit", set()).update(
                score_condition[:, 0].tolist()
            )
            return torch.zeros(len(score_condition))

        def fit_once():
            with mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
                side_effect=train,
            ), mock.patch.object(
                evenet_ratio_module,
                "_score_population",
                side_effect=score,
            ):
                return evenet_ratio_module.fit_independent_evenet_audit(
                    model_factory=lambda: nn.Linear(1, 1, bias=False),
                    data_condition=condition,
                    data_sample=condition,
                    gen_condition=condition,
                    gen_sample=condition[:, None, :],
                    gen_weight=torch.ones(len(condition), 1),
                    fit_config=SimpleNamespace(validation_batch_size=32),
                    seed=123,
                    audit_fraction=0.20,
                    early_stop_fraction=0.20,
                    reuse_early_stop_for_audit=False,
                    warm_start_cache=cache,
                    identity_split_seed=42,
                )

        cold = fit_once()
        self.assertFalse(cold.warm_started)
        self.assertEqual(
            cache["protocol"]["schema"],
            "raw-monitor-condition-60-20-20-v1",
        )
        self.assertTrue(observed["train"].isdisjoint(observed["early_stop"]))
        self.assertTrue(observed["train"].isdisjoint(observed["final_audit"]))
        self.assertTrue(
            observed["early_stop"].isdisjoint(observed["final_audit"])
        )
        self.assertEqual(
            observed["train"]
            | observed["early_stop"]
            | observed["final_audit"],
            set(condition[:, 0].tolist()),
        )
        warm = fit_once()
        self.assertTrue(warm.warm_started)


class TestEventPacking(unittest.TestCase):
    def test_round_trip_is_lossless(self) -> None:
        batch = _event_batch()
        packed, spec = pack_event_inputs(batch)
        restored = unpack_event_inputs(packed, spec)
        for key, expected in batch.items():
            torch.testing.assert_close(restored[key], expected)

    def test_enriched_pool_projects_back_to_legacy_reward_spec(self) -> None:
        batch = _event_batch_with_pair_context()
        legacy_packed, legacy_spec = pack_event_inputs(batch)
        enriched_packed, enriched_spec = pack_event_inputs(
            batch, include_pairwise_context=True
        )
        pool = AdaptiveOmniFoldPool(
            packed_event=enriched_packed,
            truth=torch.zeros(3, 4),
            candidates=torch.zeros(3, 1, 4),
            packing_spec=enriched_spec,
        )
        projected = pool.repack(legacy_spec)
        torch.testing.assert_close(projected.packed_event, legacy_packed)
        self.assertEqual(projected.packing_spec, legacy_spec)

    def test_periodic_pair_features_match_back_to_back_geometry(self) -> None:
        packed, spec = pack_event_inputs(
            _event_batch_with_pair_context(), include_pairwise_context=True
        )
        features = periodic_tau_pair_features(packed, torch.zeros(3, 4), spec)
        torch.testing.assert_close(features[:, 0], torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(features[:, 1], -torch.ones(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(features[:, 2], -torch.ones(3), atol=1e-6, rtol=0)

        fourier = periodic_tau_pair_features(
            packed,
            torch.zeros(3, 4),
            spec,
            max_harmonic=4,
            include_theta_pair=True,
        )
        self.assertEqual(tuple(fourier.shape), (3, 11))
        for harmonic in range(4):
            torch.testing.assert_close(
                fourier[:, 2 * harmonic], torch.zeros(3), atol=1e-6, rtol=0
            )
            expected_cosine = -torch.ones(3) if harmonic % 2 == 0 else torch.ones(3)
            torch.testing.assert_close(
                fourier[:, 2 * harmonic + 1], expected_cosine, atol=1e-6, rtol=0
            )
        torch.testing.assert_close(fourier[:, 8], -torch.ones(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(fourier[:, 9:], torch.zeros(3, 2), atol=1e-6, rtol=0)


class TestZtautauRatioClassifier(unittest.TestCase):
    def test_architecture_ablation_configs_keep_monitor_fixed_and_outputs_separate(self) -> None:
        import inspect
        import yaml
        from pathlib import Path
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config, adaptive_audit_protocol_signature
        from scripts.train_neutrino_backend import deep_update
        root = Path(__file__).resolve().parents[4] / "config"
        packed, spec = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
        configs = [yaml.safe_load((root / f"dgpo_omnifold_ztautau_10pct_arch_{arm}.yaml").read_text())
                   for arm in ("control", "fourier_conditioning", "large_deep_dropout")]
        self.assertEqual(len({c["options"]["Training"]["model_checkpoint_save_path"] for c in configs}), 3)
        # Fourier now starts at diffusion pretrain; historical controls stay at step 260.
        sources = [c["options"]["Training"]["model_checkpoint_load_path"] for c in configs]
        self.assertEqual(sources[0], sources[2])
        self.assertTrue(sources[0].endswith("dgpo-epoch=25-next_ep=26-step=260.ckpt"))
        self.assertEqual(sources[1], "/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/epoch=214_train=0.1519_val=0.1278.ckpt")
        signatures = [adaptive_audit_protocol_signature(resolve_adaptive_config(c["dgpo"])) for c in configs]
        self.assertEqual(signatures[0], signatures[2])
        self.assertNotEqual(signatures[0], signatures[1])  # New training budget, unchanged monitor architecture.
        accepted = set(inspect.signature(EvenetAdapterRatioClassifier).parameters)
        for index, c in enumerate(configs):
            merged = deep_update(yaml.safe_load((root / "train_diffusion_nersc.yaml").read_text()), c)
            self.assertTrue({"event_info", "resonance", "network", "options"}.issubset(merged))
            self.assertEqual(c["dgpo"]["checkpoint_load_mode"], "weights_only")
            self.assertFalse(c["dgpo"]["auto_resume_from_last"])
            self.assertFalse(c["options"]["Training"]["EMA"]["replace_model_after_load"])
            self.assertEqual(c["dgpo"]["validation_full_every_n_epochs"], 10)
            self.assertTrue(c["logger"]["wandb"]["fresh_run"])
            a = c["dgpo"]["adaptive_omnifold"]
            self.assertTrue(a["recalibration"]["bootstrap_on_start"])
            expected_warm_iterations = [] if index == 2 else [1, 2]
            self.assertEqual(a["recalibration"]["warm_start_iterations"], expected_warm_iterations)
            resolved = resolve_adaptive_config(c["dgpo"])
            self.assertEqual(resolved.warm_start_iterations, tuple(expected_warm_iterations))
            if index == 1:
                self.assertIsNone(merged["options"]["Training"]["pretrain_model_load_path"])
                self.assertFalse(merged["options"]["Training"]["EMA"]["use_for_generation"])
                self.assertIsNone(merged["dgpo"]["auto_resume_best_source_checkpoint_dir"])
                self.assertFalse(merged["dgpo"]["best_source_start_new_experiment"])
                self.assertIsNone(merged["dgpo"]["auto_resume_fallback_checkpoint_path"])
                self.assertEqual(merged["dgpo"]["global_best_checkpoint_search_dirs"], [])
                self.assertIsNone(merged["reward_config"]["omnifold"]["bundle_file"])
                self.assertEqual(merged["reward_config"]["omnifold"]["backbone_checkpoint"],
                                 "/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt")
                self.assertTrue(a["trigger"]["warm_start_classifier"])
                self.assertTrue(resolved.raw_monitor_warm_start)
                self.assertTrue(a["baseline_probe_on_start"])
                self.assertFalse(a["recalibration"]["refit_once_on_resume"])
                self.assertTrue(a["recalibration"]["reset_optimizer_state_on_install"])
                self.assertTrue(a["recalibration"]["train_grouped_sequential_embedding"])
                self.assertTrue(a["recalibration"]["train_invisible_projector"])
                self.assertEqual(merged["dgpo"]["lr_schedule"],
                                 {"type": "cosine", "total_steps": 1500, "min_lr_ratio": .1})
                self.assertEqual(merged["dgpo"]["reference_trust"]["adaptive_boundary"]["delta_max"], .1)
                self.assertEqual(merged["options"]["Training"]["weight_decay"], .001)
                self.assertEqual(merged["logger"]["wandb"]["resume"], "never")
                self.assertIsNone(merged["logger"]["wandb"]["id"])
                self.assertEqual(a["audit_fit"]["training_readiness"],
                                 {"cold_start_min_epochs": 100, "warm_start_min_epochs": 5})
                original_monitor = dict(a["audit_fit"])
                original_monitor.pop("training_readiness")
                self.assertEqual(original_monitor, configs[0]["dgpo"]["adaptive_omnifold"]["audit_fit"])
                self.assertEqual(a["recalibration"]["fit"]["min_steps_per_fold"], 1000)
                self.assertEqual(resolved.crossfit_partition, "identity")
                self.assertEqual(resolved.fit["validation_patience_epochs"], 10.)
                self.assertTrue(resolved.fit["require_saturation"])
                self.assertTrue(resolved.fit["restore_best"])
                self.assertIsNone(resolved.fit["safety_max_epochs"])
                fit_cfg = ztautau_stage.build_fit_config(
                    resolved.fit, n_train=320000, n_validation=80000,
                    max_batch_population=160000,
                )
                for fraction in (.49, .5, .51):
                    fold_cfg = evenet_ratio_module._scaled_crossfit_config(
                        fit_cfg, fraction,
                        min_steps_per_fold=resolved.fit["min_steps_per_fold"],
                    )
                    fold_cfg.validate()
                    self.assertEqual(fold_cfg.min_steps, 1000)
                    self.assertIsNone(fold_cfg.steps)
                    self.assertEqual(fold_cfg.validation_patience_evaluations, 10)
                run_name = c["logger"]["wandb"]["run_name"]
                self.assertIn("minfold1000_warm12", run_name)
                self.assertIn("monitorcold100", run_name)
                self.assertIn("frompretrain214", run_name)
                self.assertNotIn("from260", run_name)
                run_root = Path("/pscratch/sd/y/yiren/Ztautau") / run_name
                self.assertEqual(c["options"]["Training"]["model_checkpoint_save_path"], str(run_root / "checkpoints"))
                self.assertEqual(c["nersc"]["ray"]["results_dir"], str(run_root / "ray_results"))
                self.assertEqual(c["logger"]["local"]["save_dir"], str(run_root / "logs"))
                self.assertIn(str(run_root / "ray_results"), c["nersc"]["execution"]["command"])
            self.assertFalse(a["trigger"]["global_confirm_candidates"])
            self.assertEqual(a["staleness_every_n_steps"], 5)
            rec = {k:v for k,v in a["recalibration"].items() if k in accepted}
            for monitor in (False, True):
                settings = {**rec, **{k:v for k,v in a["audit_fit"].items() if k in accepted}} if monitor else rec
                model = EvenetAdapterRatioClassifier(_FakeZtautauBackbone(), spec, **settings)
                self.assertEqual(model.bank.decoder.hidden_dim, 128 if monitor or index != 2 else 256)
                self.assertEqual(len(model.bank.decoder.blocks), 1 if monitor or index != 2 else 2)
                self.assertEqual(model._head_dropout, .15 if monitor or index != 2 else .25)
                self.assertEqual(model.bank.topology_conditioning, not monitor and index == 1)
                self.assertEqual(model.bank.pairwise_feature_dim, 11 if not monitor and index == 1 else 0)
                logits = model(packed, torch.randn(3, 4))
                logits.sum().backward()
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_fourier_decoder_conditioning_gradients_and_roundtrip(self) -> None:
        import copy
        packed, spec = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
        template = _FakeZtautauBackbone()
        def build(spec):
            return EvenetAdapterRatioClassifier(
                copy.deepcopy(template), spec, periodic_pair_features=True,
                topology_fourier_embedding=True, topology_conditioning=True,
                topology_max_harmonic=4, topology_include_theta_pair=True,
                topology_hidden_dim=16, topology_embedding_dim=8, topology_dropout=0.,
                decoder_hidden_dim=8, decoder_layers=2, decoder_heads=2, head_dropout=0.,
                adapter_bottleneck=4,
            )
        model = build(spec)
        self.assertFalse(hasattr(model.bank, "fusion"))
        self.assertEqual(model.bank.output.in_features, 16)
        candidate = torch.randn(3, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(4):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(packed, candidate), torch.tensor([0., 1., 0.]))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.bank.topology_encoder.parameters()))
        model.eval()
        expected = model(packed, candidate).detach()
        self.assertEqual(model(packed, torch.randn(3, 2, 4)).shape, (3, 2))
        payload = model.peft_payload()
        self.assertTrue(payload["classifier_config"]["topology_conditioning"])
        restored = EvenetAdapterRatioClassifier.from_peft_payload(payload, model_builder=build, device=torch.device("cpu"))
        torch.testing.assert_close(restored(packed, candidate), expected)

    def test_clean_v23_reward_and_staleness_architecture(self) -> None:
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[4] / "config"
        packed, spec = pack_event_inputs(_event_batch())  # No physics/topology context.
        for filename in ("dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml",
                         "dgpo_omnifold_ztautau_10pct_velocity_mse_trust_ablation.yaml"):
            config = yaml.safe_load((root / filename).read_text())["dgpo"]["adaptive_omnifold"]
            recal = config["recalibration"]
            for dropout in (recal["head_dropout"], config["audit_fit"]["head_dropout"]):
                with self.subTest(filename=filename, dropout=dropout):
                    model = EvenetAdapterRatioClassifier(
                        _FakeZtautauBackbone(), spec,
                        periodic_pair_features=recal["periodic_pair_features"],
                        topology_fourier_embedding=recal["topology_fourier_embedding"],
                        decoder_hidden_dim=recal["decoder_hidden_dim"],
                        decoder_layers=recal["decoder_layers"],
                        decoder_heads=recal["decoder_heads"], head_dropout=dropout,
                        train_grouped_sequential_embedding=True,
                        train_invisible_projector=True,
                    )
                    self.assertEqual(model.bank.decoder.hidden_dim, 128)
                    self.assertEqual(len(model.bank.decoder.blocks), 1)
                    self.assertEqual(model.bank.pairwise_feature_dim, 0)
                    self.assertEqual(model.bank.output.in_features, 256)
                    self.assertFalse(hasattr(model.bank, "topology_encoder"))
                    self.assertFalse(hasattr(model.bank, "fusion"))
                    logits = model(packed, torch.randn(3, 4))
                    logits.sum().backward()
                    self.assertEqual(logits.shape, (3,))
                    self.assertTrue(torch.isfinite(logits).all())
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                        for p in model.parameters()))

    def test_empty_visible_event_and_nonfinite_padding_stay_finite(self) -> None:
        batch = _event_batch(batch_size=3)
        batch["x"][0] = float("nan")
        batch["x_mask"][0] = False
        packed, spec = pack_event_inputs(batch)
        model = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            asymmetric_attention=True,
            train_grouped_sequential_embedding=True,
            train_invisible_projector=True,
            head_dropout=0.25,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        block = model.bank.decoder.blocks[0]
        self.assertEqual(float(block.self_attn.dropout), 0.0)
        self.assertEqual(float(block.cross_attn.dropout), 0.0)
        self.assertEqual(float(block.ffn[2].p), 0.25)
        candidate = torch.randn(3, 4)
        target = torch.tensor([0.0, 1.0, 0.0])
        weight = torch.tensor([0.003, 2.994, 0.003])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        position_gradient = None
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            logits = model(packed, candidate)
            loss = (
                weight
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
            ).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )
            position_gradient = (
                model.bank.position_encoder.position_embedding.weight.grad
            )
            optimizer.step()

        attention_mask = model.backbone.PET.last_attn_mask
        self.assertIsNotNone(attention_mask)
        self.assertEqual(attention_mask.shape, (3, 7, 7))
        full_valid = torch.cat(
            (
                batch["x_mask"].squeeze(-1),
                torch.ones(3, 2, dtype=torch.bool),
            ),
            dim=1,
        )
        combined_blocked = attention_mask | (~full_valid[:, None, :])
        self.assertFalse(combined_blocked.all(dim=-1).any())
        # Only padded visible queries get a safe invisible key.
        self.assertFalse(attention_mask[0, 0, 5])
        self.assertTrue(attention_mask[1, 0, 5])
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsNotNone(position_gradient)
        self.assertGreater(float(position_gradient.abs().sum()), 0.0)

    def test_open_adaln_gates_sparse_memory_peaked_weights_stay_finite(self) -> None:
        """Regression for acceptance-audit step≈3 finite-fwd / NaN-bwd.

        Force AdaLN gates open (cold start normally delays this ~2 Adam steps),
        mix empty-visible and single-visible rows, and use tempered-like mean-one
        weights. Decoder attention dropout must stay off or this graph flakes.
        """
        batch = _event_batch(batch_size=4)
        batch["x"][0] = float("nan")
        batch["x_mask"][0] = False
        batch["x_mask"][1, 1:] = False
        packed, spec = pack_event_inputs(batch)
        model = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            asymmetric_attention=True,
            train_grouped_sequential_embedding=True,
            train_invisible_projector=True,
            head_dropout=0.25,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        model.train()
        for block in model.bank.decoder.blocks:
            nn.init.normal_(block.modulation.proj.weight, std=0.05)
            nn.init.normal_(block.modulation.proj.bias, std=0.05)
        # Exercise the backward through open gates, rather than the null
        # output head which intentionally blocks every deeper gradient.
        nn.init.normal_(model.bank.output.weight, std=0.05)
        candidate = torch.randn(4, 4)
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        raw = torch.tensor([1.0e-3, 1.0e3, 1.0, 1.0])
        weight = raw * (raw.numel() / raw.sum())
        for seed in (0, 1, 2, 3, 4):
            torch.manual_seed(seed)
            model.zero_grad(set_to_none=True)
            logits = model(packed, candidate)
            loss = (
                weight
                * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
            ).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(logits).all())
            bad = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            self.assertEqual(bad, [], msg=f"seed={seed} nonfinite grads: {bad}")
            position_grad = model.bank.position_encoder.position_embedding.weight.grad
            self.assertIsNotNone(position_grad)
            self.assertGreater(float(position_grad.abs().sum()), 0.0)

    def test_fourier_topology_embedding_fuses_with_decoder_and_roundtrips(self) -> None:
        packed, spec = pack_event_inputs(
            _event_batch_with_pair_context(), include_pairwise_context=True
        )
        backbone = _FakeZtautauBackbone()

        def build(restored_spec):
            return EvenetAdapterRatioClassifier(
                backbone,
                restored_spec,
                periodic_pair_features=True,
                topology_fourier_embedding=True,
                topology_max_harmonic=4,
                topology_include_theta_pair=True,
                topology_hidden_dim=16,
                topology_embedding_dim=8,
                topology_fusion_hidden_dim=12,
                topology_dropout=0.0,
                decoder_hidden_dim=8,
                decoder_layers=1,
                decoder_heads=2,
                adapter_bottleneck=4,
            )

        model = build(spec)
        self.assertEqual(model.bank.pairwise_feature_dim, 11)
        self.assertEqual(model.bank.output.in_features, 12)
        self.assertEqual(model(packed, torch.randn(3, 5, 4)).shape, (3, 5))
        payload = model.peft_payload()
        self.assertEqual(payload["classifier_config"]["topology_max_harmonic"], 4)
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            payload,
            model_builder=build,
            device=torch.device("cpu"),
        )
        self.assertTrue(restored._topology_fourier_embedding)
        self.assertEqual(restored.bank.pairwise_feature_dim, 11)

    def test_plain_fourier_head_opens_deep_gradient_after_first_update(self) -> None:
        """Zero output delays feature gradients, but is not a permanent freeze."""
        torch.manual_seed(42)
        packed, spec = pack_event_inputs(
            _event_batch_with_pair_context(), include_pairwise_context=True
        )
        model = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(), spec,
            periodic_pair_features=True, topology_fourier_embedding=True,
            topology_max_harmonic=4, topology_include_theta_pair=False,
            topology_direct_logit=False, topology_conditioning=False,
            topology_hidden_dim=16, topology_embedding_dim=8,
            topology_fusion_hidden_dim=12, topology_dropout=0.0,
            decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
            adapter_bottleneck=4,
        )
        self.assertEqual(model.bank.pairwise_feature_dim, 9)
        self.assertFalse(hasattr(model.bank, "topology_output"))
        candidate = torch.randn(3, 4)
        target = torch.tensor([1.0, 0.0, 1.0])
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4
        )
        logits = model(packed, candidate)
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, target).backward()
        self.assertGreater(float(model.bank.output.weight.grad.norm()), 0)
        self.assertEqual(float(model.bank.topology_encoder[1].weight.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.binary_cross_entropy_with_logits(
            model(packed, candidate), target
        ).backward()
        self.assertGreater(float(model.bank.topology_encoder[1].weight.grad.norm()), 0)
        self.assertTrue(all(
            torch.isfinite(p.grad).all() for p in model.parameters()
            if p.grad is not None
        ))

    def test_direct_h4_logit_stages_and_roundtrips(self) -> None:
        import copy

        packed, spec = pack_event_inputs(
            _event_batch_with_pair_context(), include_pairwise_context=True
        )
        template = _FakeZtautauBackbone()

        def build(restored_spec):
            return EvenetAdapterRatioClassifier(
                copy.deepcopy(template),
                restored_spec,
                periodic_pair_features=True,
                topology_fourier_embedding=True,
                topology_direct_logit=True,
                topology_context_residual_scale=0.1,
                topology_max_harmonic=4,
                topology_hidden_dim=16,
                topology_embedding_dim=8,
                topology_fusion_hidden_dim=12,
                topology_dropout=0.0,
                train_grouped_sequential_embedding=True,
                train_invisible_projector=True,
                train_layernorm=False,
                train_encoder=False,
                decoder_hidden_dim=8,
                decoder_layers=1,
                decoder_heads=2,
                adapter_bottleneck=4,
            )

        model = build(spec)
        self.assertEqual(model.bank.pairwise_feature_dim, 9)
        self.assertEqual(model.trainable_parameter_counts["topology_output"], 10)
        with torch.no_grad():
            model.bank.topology_output.weight.zero_()
            model.bank.topology_output.bias.fill_(1.0)
            model.bank.output.weight.zero_()
            model.bank.output.bias.fill_(2.0)

        candidate = torch.randn(3, 4)
        model.configure_topology_training_stage(1)
        stage1_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            stage1_names,
            {"bank.topology_output.weight", "bank.topology_output.bias"},
        )
        torch.testing.assert_close(
            model(packed, candidate),
            torch.ones(3),
            atol=1e-6,
            rtol=0,
        )

        model.configure_topology_training_stage(2)
        stage2_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("bank.decoder.") for name in stage2_names))
        self.assertTrue(
            any(name.startswith("backbone.PET.adapters.") for name in stage2_names)
        )
        self.assertFalse(
            any(
                name.startswith("backbone.GroupedSequentialEmbedding.")
                for name in stage2_names
            )
        )
        torch.testing.assert_close(
            model.eval()(packed, candidate),
            torch.full((3,), 1.2),
            atol=1e-6,
            rtol=0,
        )

        model.configure_topology_training_stage(3)
        stage3_names = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any(
                name.startswith("backbone.GroupedSequentialEmbedding.")
                for name in stage3_names
            )
        )
        self.assertTrue(
            any(
                name.startswith("backbone.InvisibleInputProjector.")
                for name in stage3_names
            )
        )
        self.assertFalse(
            any(name.startswith("backbone.GlobalEmbedding.") for name in stage3_names)
        )

        payload = model.peft_payload()
        self.assertTrue(payload["classifier_config"]["topology_direct_logit"])
        self.assertEqual(
            payload["classifier_config"]["topology_context_residual_scale"],
            0.1,
        )
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            payload,
            model_builder=build,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            restored(packed, candidate),
            torch.full((3,), 1.2),
            atol=1e-6,
            rtol=0,
        )

    def test_low_rank_conditional_residual_stages_and_roundtrips(self) -> None:
        import copy

        packed, spec = pack_event_inputs(
            _event_batch_with_pair_context(), include_pairwise_context=True
        )
        template = _FakeZtautauBackbone()

        def build(restored_spec):
            return EvenetAdapterRatioClassifier(
                copy.deepcopy(template),
                restored_spec,
                periodic_pair_features=True,
                topology_fourier_embedding=False,
                topology_max_harmonic=2,
                conditional_residual_rank=2,
                asymmetric_attention=True,
                topology_dropout=0.0,
                train_grouped_sequential_embedding=True,
                train_invisible_projector=True,
                decoder_hidden_dim=8,
                decoder_layers=1,
                decoder_heads=2,
                head_dropout=0.0,
                adapter_bottleneck=4,
            )

        model = build(spec)
        self.assertEqual(model.bank.pairwise_feature_dim, 5)
        self.assertEqual(model.bank.output.in_features, 16)
        self.assertEqual(model.bank.conditional_context.out_features, 2)
        self.assertEqual(model.bank.conditional_candidate.out_features, 2)
        with torch.no_grad():
            model.bank.output.weight.zero_()
            model.bank.output.bias.fill_(0.7)

        candidate = torch.randn(3, 4)
        model.configure_conditional_residual_training_stage(1)
        stage1_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertFalse(any("conditional_" in name for name in stage1_names))
        torch.testing.assert_close(
            model.eval()(packed, candidate),
            torch.full((3,), 0.7),
            atol=1e-6,
            rtol=0,
        )

        model.configure_conditional_residual_training_stage(2)
        stage2_names = {
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(
            stage2_names,
            {
                "bank.conditional_context.weight",
                "bank.conditional_candidate.weight",
            },
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
        target = torch.tensor([0.0, 1.0, 0.0])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(packed, candidate), target
            )
            loss.backward()
            self.assertTrue(
                all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )
            optimizer.step()
        self.assertGreater(
            float(model.bank.conditional_candidate.weight.abs().sum()), 0.0
        )
        self.assertEqual(model(packed, torch.randn(3, 4, 4)).shape, (3, 4))

        model.eval()
        expected = model(packed, candidate).detach()
        payload = model.peft_payload()
        self.assertEqual(
            payload["classifier_config"]["conditional_residual_rank"], 2
        )
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            payload, model_builder=build, device=torch.device("cpu")
        )
        torch.testing.assert_close(restored(packed, candidate), expected)

    def test_periodic_pair_classifier_handles_single_and_grouped_candidates(self) -> None:
        packed, spec = pack_event_inputs(
            _event_batch_with_pair_context(), include_pairwise_context=True
        )
        model = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            periodic_pair_features=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        self.assertEqual(model.bank.output.in_features, 19)
        self.assertEqual(model(packed, torch.randn(3, 4)).shape, (3,))
        self.assertEqual(model(packed, torch.randn(3, 5, 4)).shape, (3, 5))
        specialist = PeriodicPairAuditClassifier(spec, hidden_dim=8, dropout=0.0)
        self.assertEqual(specialist(packed, torch.randn(3, 5, 4)).shape, (3, 5))
        payload = model.peft_payload()
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            payload,
            model_builder=lambda restored_spec: EvenetAdapterRatioClassifier(
                model.backbone,
                restored_spec,
                periodic_pair_features=True,
                decoder_hidden_dim=8,
                decoder_layers=1,
                decoder_heads=2,
                adapter_bottleneck=4,
            ),
            device=torch.device("cpu"),
        )
        self.assertTrue(restored._periodic_pair_features)
        self.assertEqual(restored.bank.output.in_features, 19)

    def test_full_finetune_uses_only_registered_internal_pet_adapters(self) -> None:
        packed, spec = pack_event_inputs(_event_batch())
        backbone = _FakeZtautauBackbone()
        model = EvenetAdapterRatioClassifier(
            backbone,
            spec,
            train_backbone=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        self.assertFalse(hasattr(model.bank, "pet_adapters"))
        self.assertFalse(
            any(key.startswith("pet_adapters.") for key in model.bank.state_dict())
        )
        self.assertGreater(
            model.trainable_parameter_counts["internal_pet_adapters"], 0
        )
        model.train()
        self.assertTrue(backbone.training)
        model.eval()
        self.assertFalse(backbone.training)
        model(packed, torch.randn(3, 4))

    def test_last_pet_block_training_scope_and_payload(self) -> None:
        _packed, spec = pack_event_inputs(_event_batch())
        backbone = _FakeZtautauBackbone()
        backbone.PET.transformer_blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.2)),
            nn.Sequential(nn.Linear(8, 8), nn.Dropout(0.2)),
        ])
        model = EvenetAdapterRatioClassifier(
            backbone, spec, train_last_pet_block=True,
            train_grouped_sequential_embedding=False,
            train_invisible_projector=False,
            decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
            adapter_bottleneck=4,
        )
        for name, parameter in backbone.named_parameters():
            expected = name.startswith(("PET.transformer_blocks.1.", "PET.adapters."))
            self.assertEqual(parameter.requires_grad, expected, name)
        model.train()
        self.assertFalse(backbone.training)
        self.assertFalse(backbone.PET.transformer_blocks[0].training)
        self.assertTrue(backbone.PET.transformer_blocks[-1].training)
        self.assertTrue(backbone.PET.adapters.training)
        # Exercise the unfrozen tensors without depending on the fake PET's
        # forward (which intentionally skips its placeholder blocks).
        before = backbone.PET.transformer_blocks[-1][0].weight.detach().clone()
        optimizer = torch.optim.AdamW(
            [p for p in backbone.parameters() if p.requires_grad], lr=1e-3
        )
        backbone.PET.transformer_blocks[-1](torch.ones(3, 8)).square().sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, backbone.PET.transformer_blocks[-1][0].weight))
        self.assertIsNone(backbone.PET.transformer_blocks[0][0].weight.grad)
        payload = model.peft_payload()
        self.assertTrue(payload["classifier_config"]["train_last_pet_block"])
        self.assertTrue(any("transformer_blocks.1." in key for key in payload["body"]))
        self.assertFalse(any("transformer_blocks.0." in key for key in payload["body"]))
        rebuilt = EvenetAdapterRatioClassifier(
            copy.deepcopy(backbone), spec, train_last_pet_block=True,
            train_grouped_sequential_embedding=False,
            train_invisible_projector=False,
            decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
            adapter_bottleneck=4,
        )
        with torch.no_grad():
            rebuilt.backbone.PET.transformer_blocks[-1][0].weight.zero_()
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            payload, model_builder=lambda unused: rebuilt, device=torch.device("cpu")
        )
        self.assertEqual(set(restored.peft_payload()["body"]), set(payload["body"]))
        self.assertTrue(torch.equal(
            restored.backbone.PET.transformer_blocks[-1][0].weight,
            backbone.PET.transformer_blocks[-1][0].weight,
        ))
        model.eval()
        self.assertFalse(backbone.PET.transformer_blocks[-1].training)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            EvenetAdapterRatioClassifier(
                backbone, spec, train_backbone=True, train_last_pet_block=True,
                decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
                adapter_bottleneck=4,
            )

    def test_new_classifier_uses_full_self_attention(self) -> None:
        packed, spec = pack_event_inputs(_event_batch())
        backbone = _FakeZtautauBackbone()
        model = EvenetAdapterRatioClassifier(
            backbone,
            spec,
            asymmetric_attention=False,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        model(packed, torch.randn(3, 4))
        self.assertIsNone(backbone.PET.last_attn_mask)

    def test_legacy_classifier_can_restore_asymmetric_attention(self) -> None:
        packed, spec = pack_event_inputs(_event_batch())
        backbone = _FakeZtautauBackbone()
        model = EvenetAdapterRatioClassifier(
            backbone,
            spec,
            asymmetric_attention=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        model(packed, torch.randn(3, 4))
        self.assertIsNotNone(backbone.PET.last_attn_mask)
        mask = backbone.PET.last_attn_mask
        self.assertEqual(tuple(mask.shape), (3, 7, 7))
        self.assertTrue(mask[:, :5, 5:].all())
        self.assertFalse(mask[:, 5:, :].any())

    def test_accepts_four_dimensional_k1_and_grouped_candidates(self) -> None:
        packed, spec = pack_event_inputs(_event_batch())
        model = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        self.assertEqual(model.candidate_width, 4)
        self.assertEqual(model(packed, torch.randn(3, 4)).shape, (3,))
        self.assertEqual(model(packed, torch.randn(3, 5, 4)).shape, (3, 5))

    def test_trainable_body_snapshots_stay_loadable_after_freeze(self) -> None:
        _packed, spec = pack_event_inputs(_event_batch(batch_size=2))
        classifier = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            train_backbone=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        first = classifier.state_dict()
        body_keys = sorted(key for key in first if key.startswith("_backbone."))
        self.assertTrue(body_keys)
        self.assertTrue(classifier.peft_payload()["body"])
        second = {key: value.detach().clone() for key, value in first.items()}
        second[body_keys[0]].add_(1.0)
        frozen = FrozenResidualRatioReward(classifier, (first, second))
        frozen.assert_frozen()
        frozen._load_checkpoint(0)
        frozen._load_checkpoint(1)
        restored = classifier.state_dict()[body_keys[0]]
        self.assertTrue(torch.equal(restored, second[body_keys[0]]))

    def test_invisible_projector_trains_without_opening_the_frozen_backbone(self) -> None:
        _packed, spec = pack_event_inputs(_event_batch(batch_size=2))
        classifier = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            train_invisible_projector=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        body_names = {
            name for name, _parameter in classifier._trainable_backbone_parameters()
        }
        self.assertTrue(
            {
                "InvisibleInputProjector.weight",
                "InvisibleInputProjector.bias",
            }.issubset(body_names)
        )
        self.assertTrue(any(name.startswith("PET.adapters.") for name in body_names))
        self.assertEqual(body_names, set(classifier.peft_payload()["body"]))
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.backbone.InvisibleInputProjector.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.backbone.PET.adapters.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in classifier.backbone.PET.feature_embedding.parameters()
            )
        )

    def test_grouped_sequential_embedding_trains_without_opening_pet_attention(self) -> None:
        packed, spec = pack_event_inputs(_event_batch(batch_size=2))
        classifier = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            train_grouped_sequential_embedding=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        body_names = {
            name for name, _parameter in classifier._trainable_backbone_parameters()
        }
        self.assertTrue(
            {
                "GroupedSequentialEmbedding.weight",
                "GroupedSequentialEmbedding.bias",
            }.issubset(body_names)
        )
        self.assertEqual(body_names, set(classifier.peft_payload()["body"]))
        self.assertTrue(
            classifier.peft_payload()["classifier_config"][
                "train_grouped_sequential_embedding"
            ]
        )
        self.assertGreater(
            classifier.trainable_parameter_counts[
                "grouped_sequential_embedding"
            ],
            0,
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.backbone.GroupedSequentialEmbedding.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in classifier.backbone.PET.feature_embedding.parameters()
            )
        )
        self.assertGreater(
            classifier.trainable_parameter_counts["position_encoder"], 0
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.bank.position_encoder.parameters()
            )
        )
        classifier.train()
        classifier(packed, torch.randn(2, 4)).sum().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in classifier.backbone.GroupedSequentialEmbedding.parameters()
            )
        )

    def test_legacy_payload_roundtrip_does_not_add_new_flags(self) -> None:
        _packed, spec = pack_event_inputs(_event_batch(batch_size=2))
        backbone = _FakeZtautauBackbone()

        def build(_spec):
            return EvenetAdapterRatioClassifier(
                backbone,
                _spec,
                decoder_hidden_dim=8,
                decoder_layers=1,
                decoder_heads=2,
                adapter_bottleneck=4,
            )

        original = build(spec).peft_payload()
        original["classifier_config"].pop(
            "train_grouped_sequential_embedding"
        )
        original["classifier_config"].pop("train_invisible_projector")
        original["classifier_config"].pop("asymmetric_attention")
        original["classifier_config"].pop("periodic_pair_features")
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            original,
            model_builder=build,
            device=torch.device("cpu"),
        )
        roundtrip = restored.peft_payload()
        self.assertNotIn(
            "train_grouped_sequential_embedding",
            roundtrip["classifier_config"],
        )
        self.assertNotIn(
            "train_invisible_projector", roundtrip["classifier_config"]
        )
        self.assertNotIn("asymmetric_attention", roundtrip["classifier_config"])
        self.assertNotIn("periodic_pair_features", roundtrip["classifier_config"])
        self.assertTrue(
            all(name.startswith("PET.adapters.") for name in roundtrip["body"])
        )


class TestResidualRatioStack(unittest.TestCase):
    def test_fold_epoch_controls_use_actual_global_batches(self) -> None:
        cfg = RatioFitConfig(
            steps=None, min_steps=1, batch_size=16, train_microbatch_size_per_rank=2,
            sampling="independent_epoch_shuffle", drop_last_batch=True,
            validation_interval_steps=40, validation_patience_evaluations=10,
            restore_best=True, require_saturation=True,
        )
        # The parent has 20 full batches; proportional rounding would give
        # 11 for 161/320, but the actual fold has only 10 complete batches.
        for n_train, epoch_steps in ((159, 9), (161, 10)):
            for warm in (False, True):
                scaled = evenet_ratio_module._scaled_crossfit_config(
                    cfg, n_train / 320, n_train=n_train,
                    min_steps_per_fold=0 if warm else 1000,
                    min_epochs_per_fold=10 if warm else None,
                    validation_interval_epochs=2., validation_patience_epochs=10.,
                )
                self.assertEqual(scaled.min_steps, 10 * epoch_steps if warm else 1000)
                self.assertEqual(scaled.validation_interval_steps, 2 * epoch_steps)
                self.assertEqual(scaled.validation_patience_evaluations, 5)
                self.assertTrue(scaled.restore_best and scaled.require_saturation)
                self.assertEqual(scaled.train_microbatch_size_per_rank, 2)
                scaled.validate()
        tail = evenet_ratio_module._scaled_crossfit_config(
            replace(cfg, drop_last_batch=False), .5, n_train=161,
            min_epochs_per_fold=10, validation_interval_epochs=2., validation_patience_epochs=10.,
        )
        self.assertEqual((tail.min_steps, tail.validation_interval_steps), (110, 22))
        # A separate explicit minimum remains respected even for warm fits.
        larger = evenet_ratio_module._scaled_crossfit_config(
            replace(cfg, min_steps=400), .5, n_train=161, min_epochs_per_fold=10,
        )
        self.assertEqual(larger.min_steps, 200)

    def test_fold_epoch_controls_reject_unsafe_budgets(self) -> None:
        cfg = RatioFitConfig(steps=None, batch_size=16, drop_last_batch=True,
                             sampling="independent_epoch_shuffle", validation_interval_steps=2)
        for key in ('min_epochs_per_fold', 'validation_interval_epochs', 'validation_patience_epochs'):
            for value in (0, -1, True, '10', float('nan'), float('inf')):
                with self.assertRaisesRegex(ValueError, key):
                    evenet_ratio_module._scaled_crossfit_config(cfg, .5, n_train=161, **{key: value})
        with self.assertRaisesRegex(ValueError, 'no complete'):
            evenet_ratio_module._scaled_crossfit_config(cfg, .5, n_train=15, min_epochs_per_fold=10)
        with self.assertRaisesRegex(ValueError, 'step budget'):
            evenet_ratio_module._scaled_crossfit_config(
                replace(cfg, steps=150), .5, n_train=161, min_epochs_per_fold=10)
        with self.assertRaisesRegex(ValueError, 'independent_epoch_shuffle'):
            evenet_ratio_module._scaled_crossfit_config(
                replace(cfg, sampling='independent_with_replacement'), .5,
                n_train=161, min_epochs_per_fold=10)

    def test_warm_epoch_minimum_only_after_matching_fold_loaded(self) -> None:
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))
            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1])
        condition = torch.arange(160, dtype=torch.float32).reshape(80, 2)
        sample = torch.zeros(80, 4)
        cfg = RatioFitConfig(steps=None, min_steps=1, batch_size=4, drop_last_batch=True,
                             sampling='independent_epoch_shuffle', validation_interval_steps=40,
                             validation_patience_evaluations=10, require_saturation=True)

        def run(saved=None, closure=2):
            observed = []
            def fit(model, dc, ds, dw, gc, gs, gw, config, *args, **kwargs):
                warm = model.bias.item() != 0.
                epoch_steps = len(dc) // cfg.batch_size
                self.assertEqual(config.min_steps, 10 * epoch_steps if warm else 1000)
                self.assertEqual(config.validation_interval_steps, 2 * epoch_steps)
                self.assertEqual(config.validation_patience_evaluations, 5)
                self.assertNotIn('resume_state', kwargs)
                observed.append(warm)
                with torch.no_grad():
                    model.bias.fill_(len(observed))
                return SimpleNamespace(saturated=True, loss=.6, balanced_accuracy=.7, steps_completed=config.min_steps)
            with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio', side_effect=fit), \
                 mock.patch.object(evenet_ratio_module, '_weighted_binary_score_metrics',
                                   side_effect=[(.6, .7, .7)] * (closure - 1) + [(.693, .5, .5)]):
                result = fit_residual_ratio_stack(
                    model_factory=Tiny, data_condition=condition, data_sample=sample,
                    gen_condition=condition, gen_sample=sample, iterations=3,
                    fit_config=cfg, tempering=1., seed=17, crossfit_seed=42,
                    warm_start_iterations=(1, 2, 3), warm_start_state=saved,
                    min_steps_per_fold=1000, warm_start_min_epochs_per_fold=10,
                    validation_interval_epochs=2., validation_patience_epochs=10.,
                    validation_data_condition=condition, validation_data_sample=sample,
                    validation_gen_condition=condition, validation_gen_sample=sample,
                )
            return result, observed

        cold, observed = run()
        self.assertEqual(observed, [False] * 4)
        saved = copy.deepcopy(cold.warm_start_state)
        saved['models'] = [m for m in saved['models'] if (m['iteration'], m['fold']) != (1, 2)]
        _, observed = run(saved, closure=3)
        self.assertEqual(observed, [True, False, True, True, False, False])
        saved['protocol']['seed'] += 1
        _, observed = run(saved)
        self.assertEqual(observed, [False] * 4)

    def test_current_iteration_one_initializes_all_later_folds_and_only_it_crosses_rounds(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))
                self.feature = nn.Parameter(torch.zeros(()))
            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1])
        condition = torch.arange(160, dtype=torch.float32).reshape(80, 2)
        sample = torch.zeros(80, 4)
        cfg = RatioFitConfig(steps=None, min_steps=1, batch_size=4, drop_last_batch=True,
                             sampling='independent_epoch_shuffle', validation_interval_steps=40,
                             validation_patience_evaluations=5, require_saturation=True,
                             learning_rate=2e-4, backbone_learning_rate=1e-5)

        def run(saved=None, offset=0):
            observed, models = [], []
            def fit(model, dc, ds, dw, gc, gs, gw, config, *args, **kwargs):
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                self.assertNotIn('resume_state', kwargs)
                value = model.bias.item()
                self.assertEqual(model.feature.item(), 10 * value)
                self.assertEqual(config.min_steps, 10 * (len(dc) // 4) if value else 1000)
                # Only later ITERATIONS get a lower LR; cross-round warm
                # iteration 1 must retain the original rate.
                self.assertEqual(config.learning_rate, 2e-4 if len(observed) < 2 else 5e-5)
                self.assertEqual(config.backbone_learning_rate, 1e-5)
                self.assertEqual(config.weight_decay, cfg.weight_decay)
                observed.append((value, set(dc[:, 0].tolist())))
                models.append(model)
                with torch.no_grad():
                    model.bias.fill_(offset + len(observed))
                    model.feature.fill_(10 * (offset + len(observed)))
                return SimpleNamespace(saturated=True, loss=.6, balanced_accuracy=.7,
                                       steps_completed=config.min_steps)
            with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio', side_effect=fit), \
                 mock.patch.object(evenet_ratio_module, '_weighted_binary_score_metrics',
                                   side_effect=[(.6,.7,.7),(.6,.7,.7),(.693,.5,.5)]):
                result = fit_residual_ratio_stack(
                    model_factory=Tiny, data_condition=condition, data_sample=sample,
                    gen_condition=condition, gen_sample=sample, iterations=3,
                    fit_config=cfg, tempering=1., seed=17, crossfit_seed=42,
                    warm_start_iterations=(1,), warm_start_state=saved,
                    warm_start_from_iteration_one=True,
                    later_iteration_learning_rate=5e-5,
                    min_steps_per_fold=1000, warm_start_min_epochs_per_fold=10,
                    validation_interval_epochs=2., validation_patience_epochs=10.,
                    validation_data_condition=condition, validation_data_sample=sample,
                    validation_gen_condition=condition, validation_gen_sample=sample,
                )
            self.assertEqual(len({id(m) for m in models}), 6)
            # The active stack still includes both accepted residuals; only
            # the initialization cache is restricted to iteration 1.
            self.assertEqual(result.checkpoint_iterations, (1, 1, 2, 2))
            self.assertEqual([(x['iteration'], x['fold']) for x in result.warm_start_state['models']], [(1,1),(1,2)])
            self.assertEqual([x['state']['bias'].item() for x in result.warm_start_state['models']],
                             [offset+1, offset+2])
            for index in range(2, 6):
                self.assertEqual(observed[index][1], observed[index % 2][1])
            self.assertTrue(observed[0][1].isdisjoint(observed[1][1]))
            return result, [x[0] for x in observed]

        first, values = run()
        self.assertEqual(values, [0., 0., 1., 2., 1., 2.])
        saved = copy.deepcopy(first.warm_start_state)
        # Old all-iteration caches must not override the new current-round base.
        saved['models'].append({'iteration':2, 'fold':1,
                                'state':{'bias':torch.tensor(999.), 'feature':torch.tensor(9990.)}})
        digest = dgpo_reward_module.payload_sha256(saved)
        _, values = run(saved, offset=100)
        self.assertEqual(values, [1., 2., 101., 102., 101., 102.])
        self.assertEqual(dgpo_reward_module.payload_sha256(saved), digest)
        saved['models'] = [x for x in saved['models'] if (x['iteration'],x['fold']) != (1,2)]
        _, values = run(saved, offset=200)
        self.assertEqual(values, [1., 0., 201., 202., 201., 202.])
        saved['protocol']['seed'] += 1
        _, values = run(saved)
        self.assertEqual(values, [0., 0., 1., 2., 1., 2.])

    def test_absolute_fold_floor_survives_unequal_fold_scaling(self) -> None:
        cfg = RatioFitConfig(steps=None, min_steps=1, validation_interval_steps=10,
                             restore_best=True)
        for fraction in (.49, .5, .51):
            scaled = evenet_ratio_module._scaled_crossfit_config(cfg, fraction, min_steps_per_fold=1000)
            self.assertEqual(scaled.min_steps, 1000)
            self.assertTrue(scaled.restore_best)
        for invalid in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                evenet_ratio_module._scaled_crossfit_config(cfg, .5, min_steps_per_fold=invalid)
        with self.assertRaises(ValueError):
            evenet_ratio_module._scaled_crossfit_config(RatioFitConfig(steps=1500), .5, min_steps_per_fold=1000)

    def test_identity_folds_match_legacy_without_inheriting_weights(self) -> None:
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))
            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1])
        condition = torch.arange(80, dtype=torch.float32).reshape(40, 2)
        sample = torch.zeros(40, 4)
        cfg = RatioFitConfig(steps=None, min_steps=1, validation_batch_size=40, require_saturation=True)
        recorded = []
        for selected, partition, floor in (((1, 2), "auto", 0), ((), "identity", 1000)):
            fits = []
            def fit(model, dc, ds, dw, gc, gs, gw, config, *args, **kwargs):
                self.assertEqual(float(model.bias.detach()), 0.)
                if floor:
                    self.assertEqual(config.min_steps, 1000)
                fits.append(dc.clone())
                with torch.no_grad():
                    model.bias.fill_(9.)
                return SimpleNamespace(saturated=True, loss=.6, balanced_accuracy=.7, steps_completed=1000)
            with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio', side_effect=fit), \
                 mock.patch.object(evenet_ratio_module, '_weighted_binary_score_metrics', side_effect=[(.6,.7,.7),(.693,.5,.5)]):
                result = fit_residual_ratio_stack(
                    model_factory=Tiny, data_condition=condition, data_sample=sample,
                    gen_condition=condition, gen_sample=sample, iterations=2,
                    fit_config=cfg, tempering=1., seed=20260906, crossfit_seed=20260906,
                    warm_start_iterations=selected, crossfit_partition=partition,
                    min_steps_per_fold=floor,
                    validation_data_condition=condition, validation_data_sample=sample,
                    validation_gen_condition=condition, validation_gen_sample=sample,
                )
            self.assertTrue(all(not d.warm_started_folds for d in result.diagnostics))
            recorded.append(fits)
        self.assertEqual(len(recorded[0]), 4)
        self.assertEqual(len(recorded[1]), 4)
        for old, new in zip(*recorded):
            torch.testing.assert_close(old, new)

    def test_fit_does_not_early_stop_before_fold_floor(self) -> None:
        class NullRatio(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))
            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1]) * 0.
        cfg = evenet_ratio_module._scaled_crossfit_config(
            RatioFitConfig(steps=None, min_steps=1, batch_size=8,
                           sampling="independent_epoch_shuffle",
                           validation_interval_steps=2, validation_patience_evaluations=1,
                           validation_min_delta=.001, restore_best=True, require_saturation=True),
            .5, min_steps_per_fold=8)
        condition, sample, weight = torch.zeros(16, 2), torch.zeros(16, 4), torch.ones(16)
        result = fit_density_ratio(NullRatio(), condition, sample, weight, condition, sample, weight,
                                   cfg, 42, (condition, sample, weight, condition, sample, weight))
        self.assertEqual(result.steps_completed, 8)
        self.assertTrue(result.saturated)

    def test_warm_refit_keeps_training_floor_patience_and_fresh_adam(self) -> None:
        class NullRatio(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(.25))

            def forward(self, condition, sample):
                # Constant validation loss exercises the earliest allowed stop.
                return self.bias.expand(sample.shape[:-1]) * 0.

        condition = torch.arange(80, dtype=torch.float32).reshape(40, 2)
        sample = torch.zeros(40, 4)
        cfg = RatioFitConfig(
            steps=None, min_steps=1, batch_size=8, weight_decay=0.,
            sampling="independent_epoch_shuffle", validation_interval_steps=2,
            validation_patience_evaluations=3, validation_min_delta=.001,
            validation_batch_size=40, restore_best=True, require_saturation=True,
        )
        optimizers = []
        original_init = torch.optim.AdamW.__init__

        def record_optimizer(optimizer, *args, **kwargs):
            original_init(optimizer, *args, **kwargs)
            self.assertFalse(optimizer.state)
            optimizers.append(optimizer)

        def run(saved=None, seed=17, warm_epochs=None, from_first=False):
            # Keep the actual fitter/optimizer/early-stopping loop. Only the
            # outer acceptance metric is stubbed to exercise two iterations.
            with mock.patch.object(torch.optim.AdamW, "__init__", record_optimizer), \
                 mock.patch.object(evenet_ratio_module, "_weighted_binary_score_metrics",
                                   side_effect=[(.6, .7, .7), (.693, .5, .5)]):
                return fit_residual_ratio_stack(
                    model_factory=NullRatio,
                    data_condition=condition, data_sample=sample,
                    gen_condition=condition, gen_sample=sample,
                    iterations=2, fit_config=cfg, tempering=1., seed=seed,
                    warm_start_iterations=(1,) if from_first else (1, 2), warm_start_state=saved,
                    warm_start_from_iteration_one=from_first,
                    later_iteration_learning_rate=5e-5 if from_first else None,
                    crossfit_seed=42, crossfit_partition="identity",
                    min_steps_per_fold=8,
                    **({"warm_start_min_epochs_per_fold": warm_epochs,
                        "validation_interval_epochs": 2., "validation_patience_epochs": 10.}
                       if warm_epochs is not None else {}),
                    validation_data_condition=condition, validation_data_sample=sample,
                    validation_gen_condition=condition, validation_gen_sample=sample,
                )

        cold = run()
        frozen = FrozenResidualRatioReward.from_fit_result(cold, tempering=1.)
        warm = run(saved=frozen.warm_start_state, seed=18)
        self.assertEqual([d.warm_started_folds for d in cold.diagnostics], [(), ()])
        self.assertEqual([d.warm_started_folds for d in warm.diagnostics], [(1, 2), (1, 2)])
        fits = [fold for result in (cold, warm) for iteration in result.diagnostics
                for fold in iteration.fold_diagnostics]
        self.assertEqual(len(optimizers), 8)
        self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 8)
        for fit, optimizer in zip(fits, optimizers):
            self.assertTrue(fit.saturated)
            # Patience must not accumulate during the minimum-training window.
            self.assertGreaterEqual(fit.steps_completed, 8 + 3 - 1)
            self.assertTrue(optimizer.state)
            for state in optimizer.state.values():
                self.assertEqual(int(state["step"].item()), fit.steps_completed)

        shortened = run(saved=frozen.warm_start_state, seed=19, warm_epochs=2)
        pairs = evenet_ratio_module._identity_crossfit_splits(condition, folds=2, seed=42)
        self.assertEqual(len(optimizers), 12)
        self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 12)
        for iteration in shortened.diagnostics:
            self.assertEqual(iteration.warm_started_folds, (1, 2))
            for fit, (train, _) in zip(iteration.fold_diagnostics, pairs):
                epoch_steps = (len(train) + cfg.batch_size - 1) // cfg.batch_size
                # First eligible check at 2 epochs, then 4 further checks at
                # 2-epoch intervals. Not an unconditional stop at the floor.
                self.assertEqual(fit.steps_completed, 10 * epoch_steps)
                self.assertTrue(fit.saturated)

        current_first = run(seed=20, warm_epochs=2, from_first=True)
        self.assertEqual([d.warm_started_folds for d in current_first.diagnostics], [(), (1, 2)])
        self.assertEqual(len(optimizers), 16)
        self.assertEqual(len({id(optimizer) for optimizer in optimizers}), 16)
        for index, optimizer in enumerate(optimizers[-4:]):
            self.assertEqual(optimizer.param_groups[0]['lr'], cfg.learning_rate if index < 2 else 5e-5)
            self.assertEqual(optimizer.param_groups[0]['weight_decay'], cfg.weight_decay)
        for fit, (train, _) in zip(current_first.diagnostics[1].fold_diagnostics, pairs):
            self.assertEqual(fit.steps_completed, 10 * ((len(train) + cfg.batch_size - 1) // cfg.batch_size))
            self.assertTrue(fit.saturated)

    def test_crossfit_fold_checkpoints_are_averaged_per_iteration(self) -> None:
        class _TinyRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logit = nn.Parameter(torch.zeros(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.logit.expand(sample.shape[:-1])

        classifier = _TinyRatio()
        with torch.no_grad():
            classifier.logit.fill_(1.0)
        first = {
            key: value.detach().clone()
            for key, value in classifier.state_dict().items()
        }
        with torch.no_grad():
            classifier.logit.fill_(3.0)
        second = {
            key: value.detach().clone()
            for key, value in classifier.state_dict().items()
        }
        reward = FrozenResidualRatioReward(
            classifier,
            (first, second),
            tempering=0.75,
            checkpoint_coefficients=(0.5, 0.5),
            checkpoint_iterations=(1, 1),
        )
        score = reward(torch.zeros(5, 2), torch.zeros(5, 4))
        torch.testing.assert_close(score, torch.full((5,), 1.5))
        self.assertEqual(reward.num_iterations, 1)
        self.assertEqual(reward.num_checkpoints, 2)

    def test_each_residual_starts_from_a_fresh_classifier(self) -> None:
        class _TinyRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logit = nn.Parameter(torch.randn(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.logit.expand(sample.shape[:-1])

        fitted_models: list[nn.Module] = []

        def _fit(model, *args, **kwargs):
            del args, kwargs
            fitted_models.append(model)
            with torch.no_grad():
                model.logit.fill_(float((len(fitted_models) - 1) // 2 + 1))
            return SimpleNamespace(
                saturated=True,
                loss=0.6,
                balanced_accuracy=0.7,
                steps_completed=1,
            )

        condition = torch.zeros(8, 2)
        sample = torch.zeros(8, 4)
        fit_config = SimpleNamespace(
            require_saturation=True,
            validation_batch_size=4,
        )
        with mock.patch.object(
            evenet_ratio_module,
            "_score_population",
            side_effect=lambda model, _condition, population, _batch_size: (
                model.logit.detach().expand(population.shape[:-1]).clone()
            ),
        ), mock.patch(
            "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
            side_effect=_fit,
        ), mock.patch.object(
            evenet_ratio_module,
            "_weighted_binary_score_metrics",
            side_effect=[
                # AUC alone controls outer usefulness; worse-than-null BCE is
                # retained as a diagnostic and must not reject this residual.
                (0.80, 0.70, 0.70),
                (float(torch.log(torch.tensor(2.0))), 0.50, 0.50),
            ],
        ):
            result = fit_residual_ratio_stack(
                model_factory=_TinyRatio,
                data_condition=condition,
                data_sample=sample,
                gen_condition=condition,
                gen_sample=sample,
                iterations=3,
                fit_config=fit_config,
                tempering=1.0,
                seed=17,
                min_iterations=2,
                stop_balanced_accuracy=0.55,
                validation_data_condition=condition,
                validation_data_sample=sample,
                validation_gen_condition=condition,
                validation_gen_sample=sample,
            )

        self.assertEqual(len(fitted_models), 4)
        self.assertEqual(len({id(model) for model in fitted_models}), 4)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.checkpoint_coefficients, (0.5, 0.5))
        self.assertEqual(result.checkpoint_iterations, (1, 1))
        self.assertTrue(result.diagnostics[0].accepted)
        self.assertLess(result.diagnostics[0].validation_loss_gain, 0.0)
        torch.testing.assert_close(result.train_log_weight, torch.ones(8))

    def test_warm_start_selected_iterations_keeps_folds_and_unreached_history(self) -> None:
        class TinyRatio(nn.Module):
            def __init__(self):
                super().__init__()
                self.logit = nn.Parameter(torch.tensor(-10.0))

            def forward(self, condition, sample):
                return self.logit.expand(sample.shape[:-1])

        condition = torch.arange(64, dtype=torch.float32).reshape(32, 2)
        sample = torch.zeros(32, 4)
        fit_config = SimpleNamespace(require_saturation=True, validation_batch_size=8)

        def run(*, saved=None, seed=17, order=None, closure_iteration=3,
                selected=(1, 2), trained_offset=0):
            observed, progress = [], []
            if order is None:
                order = torch.arange(32)

            def fit(model, positive_condition, *args, **kwargs):
                self.assertNotIn("resume_state", kwargs)
                self.assertTrue(model.logit.requires_grad)
                observed.append((model.logit.item(), set(positive_condition[:, 0].tolist())))
                model.logit.data.fill_(float(trained_offset + len(observed)))
                return SimpleNamespace(saturated=True, loss=0.6, balanced_accuracy=0.7, steps_completed=1)

            with mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio", side_effect=fit
            ), mock.patch.object(
                evenet_ratio_module, "_weighted_binary_score_metrics",
                side_effect=[(0.6, 0.7, 0.7)] * (closure_iteration - 1) + [(0.693, 0.5, 0.5)],
            ):
                result = fit_residual_ratio_stack(
                    model_factory=TinyRatio, data_condition=condition[order], data_sample=sample[order],
                    gen_condition=condition[order], gen_sample=sample[order, None, :],
                    iterations=4, fit_config=fit_config, tempering=1.0, seed=seed,
                    validation_data_condition=condition, validation_data_sample=sample,
                    validation_gen_condition=condition, validation_gen_sample=sample[:, None, :],
                    warm_start_iterations=selected, warm_start_state=saved, crossfit_seed=42,
                    progress_callback=progress.append,
                )
            return result, observed, progress

        first, cold, _ = run()
        frozen = FrozenResidualRatioReward.from_fit_result(first, tempering=1.0)
        saved = frozen.warm_start_state
        original_digest = dgpo_reward_module.payload_sha256(saved)
        second, warm, progress = run(saved=saved, seed=18, order=torch.arange(31, -1, -1))
        self.assertEqual([row[0] for row in cold], [-10.0] * 6)
        self.assertEqual([row[0] for row in warm], [1.0, 2.0, 3.0, 4.0, -10.0, -10.0])
        self.assertEqual([row[1] for row in cold], [row[1] for row in warm])
        self.assertTrue(cold[0][1].isdisjoint(cold[1][1]))
        self.assertEqual([d.warm_started_folds for d in second.diagnostics], [(1, 2), (1, 2), ()])
        self.assertEqual([p["warm_started_folds"] for p in progress], [2.0, 2.0, 0.0])
        self.assertEqual([(m["iteration"], m["fold"]) for m in saved["models"]], [(1, 1), (1, 2), (2, 1), (2, 2)])
        self.assertEqual(dgpo_reward_module.payload_sha256(saved), original_digest)
        # New cumulative weights are recomputed from zero, never initialized
        # with the previous reward stack's accumulated log weights.
        torch.testing.assert_close(first.train_log_weight, second.train_log_weight.flip(0))

        # A selected closure-only classifier remains available for fine tuning
        # but is not an active reward increment.
        closure, _, _ = run(closure_iteration=2)
        self.assertEqual(closure.iterations, 1)
        self.assertEqual(len(closure.checkpoints), 2)
        self.assertEqual(len(closure.warm_start_state["models"]), 4)

        mismatched = copy.deepcopy(saved)
        mismatched["protocol"]["seed"] += 1
        _, skipped, _ = run(saved=mismatched)
        self.assertEqual([row[0] for row in skipped], [-10.0] * 6)

        # The new all-iteration arm warms EVERY matching fold, not iteration
        # i-1 or the opposite fold; a previously unseen iteration stays cold.
        all_ids = (1, 2, 3, 4)
        history, _, _ = run(selected=all_ids, closure_iteration=3)
        expanded, observed, _ = run(selected=all_ids, saved=history.warm_start_state,
                                   closure_iteration=4, seed=20, order=torch.arange(31, -1, -1))
        self.assertEqual([row[0] for row in observed], [1., 2., 3., 4., 5., 6., -10., -10.])
        self.assertEqual([d.warm_started_folds for d in expanded.diagnostics], [(1, 2)] * 3 + [()])
        self.assertEqual(len(expanded.warm_start_state["models"]), 8)
        incoming_hash = dgpo_reward_module.payload_sha256(expanded.warm_start_state)
        # A shorter middle round updates iterations 1/2 and retains 3/4 only
        # as initialization, never as extra active reward increments.
        shorter, _, _ = run(selected=all_ids, saved=expanded.warm_start_state,
                            closure_iteration=2, trained_offset=100)
        self.assertEqual(len(shorter.checkpoints), 2)
        self.assertEqual(shorter.checkpoint_iterations, (1, 1))
        self.assertEqual(len(shorter.warm_start_state["models"]), 8)
        self.assertEqual(dgpo_reward_module.payload_sha256(expanded.warm_start_state), incoming_hash)
        last, observed, _ = run(selected=all_ids, saved=shorter.warm_start_state, closure_iteration=4,
                                order=torch.arange(31, -1, -1))
        self.assertEqual([row[0] for row in observed], [101., 102., 103., 104., 5., 6., 7., 8.])
        self.assertEqual([d.warm_started_folds for d in last.diagnostics], [(1, 2)] * 4)
        # A protocol mismatch invalidates history, including unused entries.
        bad_history = copy.deepcopy(shorter.warm_start_state)
        bad_history["protocol"]["seed"] += 1
        fresh, observed, _ = run(selected=all_ids, saved=bad_history, closure_iteration=2)
        self.assertEqual([row[0] for row in observed], [-10.] * 4)
        self.assertEqual(len(fresh.warm_start_state["models"]), 4)

    def test_identity_folds_keep_duplicate_conditions_together(self) -> None:
        unique = torch.arange(128, dtype=torch.float32).reshape(64, 2)
        condition = torch.cat([unique, unique])
        pairs = evenet_ratio_module._identity_crossfit_splits(condition, folds=2, seed=42)
        for fit, holdout in pairs:
            self.assertTrue(set(condition[fit, 0].tolist()).isdisjoint(condition[holdout, 0].tolist()))

    def test_repeated_crossfit_averages_five_oof_predictions_and_warm_starts(self) -> None:
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))

            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1])

        condition = torch.arange(400, dtype=torch.float32).reshape(200, 2)
        sample = torch.zeros(200, 4)
        cfg = RatioFitConfig(
            steps=1,
            batch_size=8,
            validation_interval_steps=1,
            validation_patience_evaluations=1,
            require_saturation=True,
        )

        def run(saved=None, offset=0):
            calls, iteration_rows = [], []

            def fit(model, dc, ds, dw, gc, gs, gw, config, seed, *args, **kwargs):
                calls.append((float(model.bias), int(seed), set(dc[:, 0].tolist())))
                with torch.no_grad():
                    model.bias.fill_(offset + len(calls))
                return SimpleNamespace(
                    saturated=True,
                    loss=.6,
                    balanced_accuracy=.7,
                    steps_completed=1,
                )

            def observer(event, row):
                if event == "iteration_scored":
                    iteration_rows.append(row)

            with mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
                side_effect=fit,
            ), mock.patch.object(
                evenet_ratio_module,
                "_weighted_binary_score_metrics",
                return_value=(.6, .7, .7),
            ):
                result = fit_residual_ratio_stack(
                    model_factory=Tiny,
                    data_condition=condition,
                    data_sample=sample,
                    gen_condition=condition,
                    gen_sample=sample,
                    iterations=1,
                    min_iterations=1,
                    iteration_one_only=True,
                    fit_config=cfg,
                    tempering=1.,
                    seed=17,
                    crossfit_seed=42,
                    crossfit_folds=2,
                    crossfit_repeats=5,
                    crossfit_partition="identity",
                    warm_start_iterations=(1,),
                    warm_start_state=saved,
                    validation_data_condition=condition,
                    validation_data_sample=sample,
                    validation_gen_condition=condition,
                    validation_gen_sample=sample,
                    diagnostic_callback=observer,
                )
            return result, calls, iteration_rows[0]

        result, calls, row = run()
        self.assertEqual(len(calls), 10)
        expected_repeat_oof = []
        for repeat in range(1, 6):
            repeat_oof = torch.empty(200)
            split_seed = evenet_ratio_module._crossfit_repeat_seed(42, repeat)
            for fold, (_, holdout) in enumerate(
                evenet_ratio_module._identity_crossfit_splits(
                    condition, folds=2, seed=split_seed
                ),
                start=1,
            ):
                repeat_oof[holdout] = (repeat - 1) * 2 + fold
            expected_repeat_oof.append(repeat_oof)
        expected = torch.stack(expected_repeat_oof).mean(0)
        torch.testing.assert_close(row["oof_logit"].reshape(-1), expected)
        torch.testing.assert_close(
            row["validation_gen_logit"],
            torch.full_like(row["validation_gen_logit"], 5.5),
        )
        self.assertEqual(result.checkpoint_coefficients, (0.1,) * 10)
        self.assertEqual(result.checkpoint_iterations, (1,) * 10)
        self.assertEqual(result.diagnostics[0].warm_started_folds, ())
        self.assertEqual(result.warm_start_state["protocol"]["repeats"], 5)
        self.assertEqual(
            {
                (entry["repeat"], entry["fold"])
                for entry in result.warm_start_state["models"]
            },
            {(repeat, fold) for repeat in range(1, 6) for fold in (1, 2)},
        )
        reward = FrozenResidualRatioReward.from_fit_result(result, tempering=1.)
        torch.testing.assert_close(
            reward(condition, sample),
            torch.full((200,), 5.5),
        )

        warmed, warm_calls, _ = run(copy.deepcopy(result.warm_start_state), offset=20)
        self.assertEqual([value for value, _, _ in warm_calls], list(range(1, 11)))
        self.assertEqual(warmed.diagnostics[0].warm_started_folds, tuple(range(1, 11)))

        legacy = copy.deepcopy(result.warm_start_state)
        legacy["protocol"].pop("repeats")
        legacy["protocol"].pop("repeat_seed_stride")
        for entry in legacy["models"]:
            entry.pop("repeat")
        single, _, _ = run(legacy)
        self.assertEqual(single.diagnostics[0].warm_started_folds, ())

    def test_five_fold_keeps_held_out_scores_and_five_trainings(self) -> None:
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))

            def forward(self, condition, sample):
                return self.bias.expand(sample.shape[:-1])

        condition = torch.arange(400, dtype=torch.float32).reshape(200, 2)
        sample = torch.zeros(200, 4)
        cfg = RatioFitConfig(
            steps=1,
            batch_size=8,
            validation_interval_steps=1,
            validation_patience_evaluations=1,
            require_saturation=True,
        )

        calls, iteration_rows = [], []

        def fit(model, dc, ds, dw, gc, gs, gw, config, seed, *args, **kwargs):
            fit_ids = set(dc[:, 0].tolist())
            calls.append(fit_ids)
            with torch.no_grad():
                model.bias.fill_(float(len(calls)))
            return SimpleNamespace(
                saturated=True,
                loss=.6,
                balanced_accuracy=.7,
                steps_completed=1,
            )

        def observer(event, row):
            if event == "fold_scored":
                fit_ids = set(row["fit_index"].tolist())
                holdout_ids = set(row["holdout_index"].tolist())
                self.assertTrue(fit_ids.isdisjoint(holdout_ids))
                self.assertEqual(fit_ids | holdout_ids, set(range(200)))
            if event == "iteration_scored":
                iteration_rows.append(row)

        with mock.patch(
            "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
            side_effect=fit,
        ), mock.patch.object(
            evenet_ratio_module,
            "_weighted_binary_score_metrics",
            return_value=(.6, .7, .7),
        ):
            result = fit_residual_ratio_stack(
                model_factory=Tiny,
                data_condition=condition,
                data_sample=sample,
                gen_condition=condition,
                gen_sample=sample,
                iterations=1,
                min_iterations=1,
                iteration_one_only=True,
                fit_config=cfg,
                tempering=1.,
                seed=17,
                crossfit_seed=42,
                crossfit_folds=5,
                crossfit_repeats=1,
                crossfit_partition="identity",
                warm_start_iterations=(1,),
                validation_data_condition=condition,
                validation_data_sample=sample,
                validation_gen_condition=condition,
                validation_gen_sample=sample,
                diagnostic_callback=observer,
            )

        self.assertEqual(len(calls), 5)
        expected = torch.empty(200)
        for fold, (_, holdout) in enumerate(
            evenet_ratio_module._identity_crossfit_splits(
                condition, folds=5, seed=42
            ),
            start=1,
        ):
            expected[holdout] = fold
        torch.testing.assert_close(
            iteration_rows[0]["oof_logit"].reshape(-1), expected
        )
        torch.testing.assert_close(
            iteration_rows[0]["validation_gen_logit"],
            torch.full_like(iteration_rows[0]["validation_gen_logit"], 3.0),
        )
        self.assertEqual(result.checkpoint_coefficients, (0.2,) * 5)
        self.assertEqual(result.checkpoint_iterations, (1,) * 5)
        self.assertEqual(len(result.warm_start_state["models"]), 5)
        self.assertEqual(result.warm_start_state["protocol"]["folds"], 5)
        self.assertNotIn("repeats", result.warm_start_state["protocol"])

    def test_warm_start_cache_survives_reward_serialization_without_affecting_scores(self) -> None:
        batch = _event_batch(batch_size=4)
        packed, spec = pack_event_inputs(batch)
        backbone = _FakeZtautauBackbone()

        def factory(packing_spec):
            return EvenetAdapterRatioClassifier(
                copy.deepcopy(backbone), packing_spec,
                decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
                adapter_bottleneck=4,
            )

        classifier = factory(spec)
        snapshot = copy.deepcopy(classifier.state_dict())
        cache = {
            "protocol": {
                "scheme": "condition_sha256_v1",
                "seed": 42,
                "folds": 2,
                "repeats": 5,
                "repeat_seed_stride": 104729,
                "condition_width": packed.shape[-1],
            },
            "models": [
                {
                    "iteration": 1,
                    "repeat": repeat,
                    "fold": fold,
                    "state": snapshot,
                }
                for repeat in range(1, 6) for fold in (1, 2)
            ],
        }
        reward = FrozenResidualRatioReward(classifier, (snapshot,), warm_start_state=cache)
        candidate = torch.zeros(4, 1, 4)
        expected = reward(packed, candidate).clone()
        payload = reward.serializable_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reward.pt"
            torch.save(payload, path)
            restored = FrozenResidualRatioReward.from_serializable_payload(
                torch.load(path, map_location="cpu", weights_only=False),
                model_builder=factory, device=torch.device("cpu"),
            )
        torch.testing.assert_close(restored(packed, candidate), expected)
        self.assertEqual(dgpo_reward_module.payload_sha256(payload), dgpo_reward_module.payload_sha256(restored.serializable_payload()))
        restored.assert_frozen()
        self.assertEqual(restored.warm_start_state["protocol"]["repeats"], 5)
        self.assertEqual(len(restored.warm_start_state["models"]), 10)
        for value in restored.warm_start_state["models"][0]["state"].values():
            self.assertEqual(value.device.type, "cpu")
        # The training cache is detached from the caller's tensors.
        for value in cache["models"][0]["state"].values():
            value.zero_()
        self.assertEqual(dgpo_reward_module.payload_sha256(payload), dgpo_reward_module.payload_sha256(restored.serializable_payload()))

    def test_first_residual_below_auc_gate_is_rejected(self) -> None:
        class _TinyRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logit = nn.Parameter(torch.zeros(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.logit.expand(sample.shape[:-1])

        condition = torch.zeros(8, 2)
        sample = torch.zeros(8, 4)
        fit_config = SimpleNamespace(
            require_saturation=True,
            validation_batch_size=4,
            validation_min_delta=5.0e-4,
        )
        diagnostic = SimpleNamespace(
            saturated=True,
            loss=0.69,
            balanced_accuracy=0.5,
            steps_completed=1,
        )
        with (
            mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
                return_value=diagnostic,
            ),
            mock.patch.object(
                evenet_ratio_module,
                "_score_population",
                side_effect=lambda _model, condition, _sample, _batch: torch.zeros(
                    len(condition)
                ),
            ),
            mock.patch.object(
                evenet_ratio_module,
                "_weighted_binary_score_metrics",
                return_value=(0.715, 0.42, 0.42),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "AUC gate"):
                fit_residual_ratio_stack(
                    model_factory=_TinyRatio,
                    data_condition=condition,
                    data_sample=sample,
                    gen_condition=condition,
                    gen_sample=sample,
                    iterations=2,
                    fit_config=fit_config,
                    tempering=0.75,
                    seed=17,
                    min_iterations=2,
                    stop_balanced_accuracy=0.55,
                    validation_data_condition=condition,
                    validation_data_sample=sample,
                    validation_gen_condition=condition,
                    validation_gen_sample=sample,
                )

    def test_crossfit_scores_training_events_only_out_of_fold(self) -> None:
        class _IdentityAwareRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logit = nn.Parameter(torch.ones(()))
                self.fit_ids: set[int] = set()

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.logit.expand(sample.shape[:-1])

        scored_oof_ids: set[int] = set()

        def _fit(model, positive_condition, *args, **kwargs):
            del args, kwargs
            model.fit_ids = {
                int(value) for value in positive_condition[:, 0].tolist()
            }
            return SimpleNamespace(
                saturated=True,
                loss=0.6,
                balanced_accuracy=0.7,
                steps_completed=1,
            )

        def _score(model, condition, population, _batch_size):
            del population
            ids = {int(value) for value in condition[:, 0].tolist()}
            if ids and max(ids) < 100:
                self.assertTrue(ids.isdisjoint(model.fit_ids))
                scored_oof_ids.update(ids)
            return model.logit.detach().expand(condition.shape[0]).clone()

        train_condition = torch.stack(
            (torch.arange(12, dtype=torch.float32), torch.zeros(12)), dim=1
        )
        validation_condition = train_condition + 100.0
        sample = torch.zeros(12, 4)
        fit_config = SimpleNamespace(
            require_saturation=True,
            validation_batch_size=4,
            validation_min_delta=5.0e-4,
        )
        with (
            mock.patch(
                "RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio",
                side_effect=_fit,
            ),
            mock.patch.object(
                evenet_ratio_module,
                "_score_population",
                side_effect=_score,
            ),
            mock.patch.object(
                evenet_ratio_module,
                "_weighted_binary_score_metrics",
                side_effect=[(0.60, 0.70, 0.70), (0.693147, 0.50, 0.50)],
            ),
        ):
            fit_residual_ratio_stack(
                model_factory=_IdentityAwareRatio,
                data_condition=train_condition,
                data_sample=sample,
                gen_condition=train_condition,
                gen_sample=sample,
                iterations=2,
                fit_config=fit_config,
                tempering=0.75,
                seed=31,
                min_iterations=2,
                stop_balanced_accuracy=0.55,
                validation_data_condition=validation_condition,
                validation_data_sample=sample,
                validation_gen_condition=validation_condition,
                validation_gen_sample=sample,
            )
        self.assertEqual(scored_oof_ids, set(range(12)))


class TestEvenetAdapterModelBuilder(unittest.TestCase):
    def test_body_only_checkpoint_discards_truth_head_and_reopens_selected_inputs(
        self,
    ) -> None:
        class TruthHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.position_encoder = nn.Linear(4, 4)

        template = _FakeZtautauBackbone()
        template.TruthGeneration = TruthHead()

        def load_finetuned_weights(model, *_args, **_kwargs):
            model.GroupedSequentialEmbedding.weight.data.fill_(2.0)
            model.InvisibleInputProjector.weight.data.fill_(3.0)
            model.TruthGeneration.position_encoder.weight.data.fill_(9.0)
            return {"state_dict": {}}

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "finetuned.ckpt"
            checkpoint.touch()
            with mock.patch.object(
                evenet_ratio_module,
                "_config_with_pet_adapters",
                return_value=SimpleNamespace(),
            ), mock.patch(
                "RL.DGPO_neutrino.model_utils.build_evenet_on_device",
                return_value=template,
            ), mock.patch(
                "RL.DGPO_neutrino.model_utils.load_weights_like_configure_model",
                side_effect=load_finetuned_weights,
            ):
                builder = EvenetAdapterModelBuilder(
                    config=SimpleNamespace(),
                    normalization_dict={},
                    checkpoint_path=checkpoint,
                    device=torch.device("cpu"),
                    body_only_checkpoint=True,
                    train_grouped_sequential_embedding=True,
                    train_invisible_projector=True,
                )

        self.assertTrue(builder.body_only_checkpoint)
        self.assertIsNone(builder._position_state)
        self.assertFalse(hasattr(builder.backbone, "TruthGeneration"))
        torch.testing.assert_close(
            builder.backbone.GroupedSequentialEmbedding.weight,
            torch.full_like(
                builder.backbone.GroupedSequentialEmbedding.weight,
                2.0,
            ),
        )
        torch.testing.assert_close(
            builder.backbone.InvisibleInputProjector.weight,
            torch.full_like(
                builder.backbone.InvisibleInputProjector.weight,
                3.0,
            ),
        )

        _packed, spec = pack_event_inputs(_event_batch(batch_size=2))
        with mock.patch(
            "RL.DGPO_neutrino.model_utils.build_evenet_on_device",
            return_value=_FakeZtautauBackbone(),
        ):
            classifier = builder.make_classifier(spec, "body_ablation")
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.backbone.GroupedSequentialEmbedding.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.backbone.InvisibleInputProjector.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in classifier.backbone.GlobalEmbedding.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in classifier.backbone.PET.adapters.parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in classifier.backbone.PET.feature_embedding.parameters()
            )
        )

    def test_monitor_architecture_override_reaches_builder(self) -> None:
        builder = mock.Mock()
        overrides = dict(head_dropout=.15, decoder_hidden_dim=128, decoder_layers=1,
                         decoder_heads=4, periodic_pair_features=False,
                         topology_fourier_embedding=True, topology_direct_logit=True,
                         topology_context_residual_scale=.1,
                         topology_conditioning=False)
        evenet_ratio_module.peft_bank_factory(builder, "spec", "audit", classifier_overrides=overrides)()
        builder.make_classifier.assert_called_once_with("spec", "audit", reset=True, **overrides)
        for invalid in (
            {"decoder_layers": 0},
            {"decoder_hidden_dim": True},
            {"topology_conditioning": "false"},
            {"topology_context_residual_scale": 1.1},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                evenet_ratio_module.peft_bank_factory(builder, "spec", classifier_overrides=invalid)

    def test_monitor_dropout_override_does_not_change_reward_defaults(self) -> None:
        builder = mock.Mock()
        spec = object()
        reward = evenet_ratio_module.peft_bank_factory(builder, spec, "adaptive_reward")
        reward()
        builder.make_classifier.assert_called_once_with(spec, "adaptive_reward", reset=True)
        builder.make_classifier.reset_mock()
        audit = evenet_ratio_module.peft_bank_factory(
            builder, spec, "audit", classifier_overrides={"head_dropout": .25, "topology_dropout": .25}
        )
        audit()
        builder.make_classifier.assert_called_once_with(
            spec, "audit", reset=True, head_dropout=.25, topology_dropout=.25
        )
        builder.reset_audit_bank.assert_not_called()

    def test_default_audit_factory_preserves_legacy_reset(self) -> None:
        builder = mock.Mock()
        spec = object()
        evenet_ratio_module.peft_bank_factory(builder, spec, "audit")()
        builder.reset_audit_bank.assert_called_once_with(spec)
        builder.make_classifier.assert_not_called()
        for invalid in (-.1, 1., float("nan")):
            with self.assertRaises(ValueError):
                evenet_ratio_module.peft_bank_factory(
                    builder, spec, "audit", classifier_overrides={"head_dropout": invalid}
                )

    def test_fresh_backbone_rebuilds_without_deepcopy(self) -> None:
        template = _FakeZtautauBackbone()
        # Mirrors the non-pickleable runtime view present in the real EveNet
        # model that caused the NERSC bootstrap failure.
        template.runtime_feature_keys = {"visible": 1}.keys()
        pretrained = {
            key: value.detach().cpu().clone()
            for key, value in template.state_dict().items()
        }
        rebuilt = _FakeZtautauBackbone()
        for parameter in rebuilt.parameters():
            parameter.data.zero_()

        builder = EvenetAdapterModelBuilder.__new__(EvenetAdapterModelBuilder)
        builder._backbone = template
        builder._config = SimpleNamespace()
        builder._normalization_dict = {}
        builder._device = torch.device("cpu")
        builder._pretrained_body = pretrained

        with mock.patch(
            "RL.DGPO_neutrino.model_utils.build_evenet_on_device",
            return_value=rebuilt,
        ) as build:
            fresh = builder._fresh_backbone()

        build.assert_called_once_with(
            builder._config,
            builder._normalization_dict,
            builder._device,
        )
        self.assertIs(fresh, rebuilt)
        self.assertIsNot(fresh, template)
        for key, expected in pretrained.items():
            torch.testing.assert_close(fresh.state_dict()[key], expected)
        self.assertFalse(fresh.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in fresh.parameters()))


class TestRatioFitProgress(unittest.TestCase):
    def test_anomaly_identifies_backward_operation_with_finite_forward(self) -> None:
        class PoisonGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, grad):
                return torch.full_like(grad, float("nan"))

        class Ratio(nn.Module):
            def __init__(self, poison_positive):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))
                self.poison_positive = poison_positive

            def forward(self, condition, sample):
                value = self.weight * sample[..., 0]
                if bool((sample[0, 0] > 0).item()) == self.poison_positive:
                    value = PoisonGradient.apply(value)
                return value

        condition = torch.zeros(8, 1)
        for positive in (True, False):
            with self.subTest(positive=positive):
                model = Ratio(positive)
                label = "positive" if positive else "negative"
                with warnings.catch_warnings(record=True), self.assertRaisesRegex(
                    FloatingPointError,
                    f"first_bad_rank=0.*autograd_detail=class={label}.*PoisonGradientBackward",
                ), mock.patch.object(torch.optim.AdamW, "step") as optimizer_step:
                    fit_density_ratio(
                        model, condition, torch.ones(8, 1), torch.ones(8),
                        condition, -torch.ones(8, 1), torch.ones(8),
                        RatioFitConfig(steps=1, batch_size=8, anomaly_detection_steps=10),
                        seed=13,
                    )
                optimizer_step.assert_not_called()
                self.assertEqual(model.weight.item(), 1.0)
                self.assertIsNone(model.weight.grad)

    def test_anomaly_setting_reaches_fit_config(self) -> None:
        config = ztautau_stage.build_fit_config(
            {"batch_size": 8, "anomaly_detection_steps": 10},
            n_train=32, n_validation=16,
        )
        self.assertEqual(config.anomaly_detection_steps, 10)
        with self.assertRaisesRegex(ValueError, "anomaly_detection_steps"):
            RatioFitConfig(anomaly_detection_steps=-1).validate()
        staged = ztautau_stage.build_fit_config(
            {
                "batch_size": 8,
                "min_steps": 300,
                "topology_warmup_steps": 100,
                "topology_body_unfreeze_step": 300,
                "topology_warmup_learning_rate": 1.0e-3,
            },
            n_train=32,
            n_validation=16,
        )
        self.assertEqual(staged.topology_warmup_steps, 100)
        self.assertEqual(staged.topology_body_unfreeze_step, 300)
        self.assertEqual(staged.topology_warmup_learning_rate, 1.0e-3)
        fold = evenet_ratio_module._scaled_crossfit_config(
            staged,
            0.5,
            min_steps_per_fold=300,
        )
        self.assertEqual(fold.min_steps, 300)
        self.assertEqual(fold.topology_warmup_steps, 100)
        self.assertEqual(fold.topology_body_unfreeze_step, 300)

    def test_topology_fit_schedule_rebuilds_optimizer_and_gates_body(self) -> None:
        class StagedRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.direct = nn.Parameter(torch.zeros(()))
                self.head = nn.Parameter(torch.zeros(()))
                self.body = nn.Parameter(torch.zeros(()))
                self.stage_calls: list[int] = []

            def configure_topology_training_stage(self, stage: int) -> None:
                self.stage_calls.append(stage)
                self.requires_grad_(False)
                self.direct.requires_grad_(True)
                if stage >= 2:
                    self.head.requires_grad_(True)
                if stage >= 3:
                    self.body.requires_grad_(True)

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return (self.direct + self.head + self.body) * sample[..., 0]

        model = StagedRatio()
        condition = torch.zeros(8, 1)
        positive = torch.ones(8, 1)
        negative = -torch.ones(8, 1)
        weights = torch.ones(8)
        validation = (
            condition,
            positive,
            weights,
            condition,
            negative,
            weights,
        )
        optimizer_lrs: list[float] = []
        original_init = torch.optim.AdamW.__init__

        def record_optimizer(optimizer, *args, **kwargs):
            original_init(optimizer, *args, **kwargs)
            optimizer_lrs.append(float(kwargs["lr"]))

        evaluator = mock.Mock(
            side_effect=[
                (0.70, 0.50, 0.50),
                (0.69, 0.55, 0.55),
                (0.68, 0.60, 0.60),
                (0.67, 0.65, 0.65),
            ]
        )
        with mock.patch.object(torch.optim.AdamW, "__init__", record_optimizer):
            fit_density_ratio(
                model,
                condition,
                positive,
                weights,
                condition,
                negative,
                weights,
                RatioFitConfig(
                    steps=3,
                    batch_size=8,
                    learning_rate=1.0e-3,
                    min_steps=2,
                    validation_interval_steps=1,
                    validation_patience_evaluations=10,
                    validation_batch_size=8,
                    topology_warmup_steps=1,
                    topology_body_unfreeze_step=2,
                    topology_warmup_learning_rate=1.0e-2,
                ),
                seed=17,
                validation=validation,
                validation_evaluator=evaluator,
            )

        self.assertEqual(model.stage_calls, [1, 2, 3])
        self.assertEqual(optimizer_lrs, [1.0e-2, 1.0e-3, 1.0e-3])

    def test_grouped_embedding_and_invisible_projector_share_fast_lr(self) -> None:
        self.assertTrue(
            _uses_fast_ratio_learning_rate(
                "backbone.GroupedSequentialEmbedding.projection.weight"
            )
        )
        self.assertTrue(
            _uses_fast_ratio_learning_rate(
                "backbone.InvisibleInputProjector.weight"
            )
        )
        self.assertTrue(
            _uses_fast_ratio_learning_rate(
                "backbone.PET.adapters.0.down.weight"
            )
        )
        self.assertTrue(
            _uses_fast_ratio_learning_rate("bank.decoder.output.weight")
        )
        self.assertFalse(
            _uses_fast_ratio_learning_rate(
                "bank.position_encoder.position_embedding.weight"
            )
        )
        self.assertFalse(
            _uses_fast_ratio_learning_rate(
                "backbone.PET.feature_embedding.weight"
            )
        )

    def test_epoch_shuffle_visits_each_row_once_per_epoch(self) -> None:
        batcher = _EpochShuffleBatcher(size=10, batch_size=4, seed=17)
        first_epoch = [batcher.draw(step) for step in range(3)]
        second_epoch = [batcher.draw(step) for step in range(3, 6)]
        self.assertEqual([len(batch) for batch in first_epoch], [4, 4, 2])
        self.assertEqual([len(batch) for batch in second_epoch], [4, 4, 2])
        self.assertEqual(sorted(torch.cat(first_epoch).tolist()), list(range(10)))
        self.assertEqual(sorted(torch.cat(second_epoch).tolist()), list(range(10)))

    def test_epoch_shuffle_can_drop_short_last_batch(self) -> None:
        batcher = _EpochShuffleBatcher(
            size=10, batch_size=4, seed=17, drop_last=True
        )
        first_epoch = [batcher.draw(step) for step in range(2)]
        second_epoch = [batcher.draw(step) for step in range(2, 4)]
        self.assertEqual([len(batch) for batch in first_epoch], [4, 4])
        self.assertEqual([len(batch) for batch in second_epoch], [4, 4])
        self.assertEqual(len(torch.unique(torch.cat(first_epoch))), 8)
        self.assertEqual(len(torch.unique(torch.cat(second_epoch))), 8)

    def test_null_max_epochs_builds_an_unbounded_epoch_fit(self) -> None:
        config = ztautau_stage.build_fit_config(
            {
                "batch_size": 32,
                "train_microbatch_size_per_rank": 7,
                "sampling": "independent_epoch_shuffle",
                "safety_max_epochs": None,
                "min_epochs": 1,
                "validation_interval_epochs": 1,
                "validation_patience_epochs": 10,
                "gradient_clip_norm": 1.0,
                "require_saturation": True,
            },
            n_train=100,
            n_validation=20,
        )
        self.assertIsNone(config.steps)
        self.assertEqual(config.train_microbatch_size_per_rank, 7)
        self.assertEqual(config.min_steps, 4)
        self.assertEqual(config.validation_interval_steps, 4)
        self.assertEqual(config.validation_patience_evaluations, 10)
        self.assertEqual(config.gradient_clip_norm, 1.0)

    def test_drop_last_uses_only_complete_batches_for_epoch_length(self) -> None:
        config = ztautau_stage.build_fit_config(
            {
                "batch_size": 32,
                "drop_last_batch": True,
                "safety_max_epochs": None,
                "min_epochs": 1,
                "validation_interval_epochs": 1,
                "validation_patience_epochs": 10,
            },
            n_train=100,
            n_validation=20,
        )
        self.assertTrue(config.drop_last_batch)
        self.assertEqual(config.min_steps, 3)
        self.assertEqual(config.validation_interval_steps, 3)
        self.assertEqual(config.validation_patience_evaluations, 10)

    def test_small_pool_batch_cap_fits_every_fold_on_16_workers(self) -> None:
        from RL.DGPO_neutrino.omnifold_ztautau import ratio_fit

        with mock.patch.object(ratio_fit, "distributed_context", return_value=(0, 16)):
            config = ztautau_stage.build_fit_config(
                {"batch_size": 32768, "drop_last_batch": True},
                n_train=21000, n_validation=5000, max_batch_population=10499,
            )
            self.assertEqual(config.batch_size, 10496)
            self.assertEqual(config.batch_size % 16, 0)
            batch = _EpochShuffleBatcher(size=10499, batch_size=config.batch_size, seed=42, drop_last=True).draw(0)
            self.assertEqual(len(batch), 10496)
            large = ztautau_stage.build_fit_config(
                {"batch_size": 32768, "drop_last_batch": True},
                n_train=210000, n_validation=50000, max_batch_population=104990,
            )
            self.assertEqual(large.batch_size, 32768)
            with self.assertRaisesRegex(ValueError, "GPU worker count"):
                ztautau_stage.build_fit_config({}, n_train=20, n_validation=5, max_batch_population=15)

    def test_training_microbatches_reconstruct_the_full_batch_update(self) -> None:
        class _LinearRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(3, 1)

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                return self.linear(torch.cat((condition, sample), dim=-1)).squeeze(-1)

        torch.manual_seed(31)
        full_model = _LinearRatio()
        micro_model = copy.deepcopy(full_model)
        condition = torch.randn(8, 2)
        positive = torch.randn(8, 1) + 0.4
        negative = torch.randn(8, 1) - 0.3
        positive_weight = torch.linspace(0.5, 1.5, 8)
        negative_weight = torch.linspace(1.5, 0.5, 8)
        common = dict(
            steps=1,
            batch_size=8,
            learning_rate=2.0e-3,
            weight_decay=0.0,
            sampling="independent_epoch_shuffle",
        )
        full_diag = fit_density_ratio(
            full_model,
            condition,
            positive,
            positive_weight,
            condition,
            negative,
            negative_weight,
            RatioFitConfig(**common),
            seed=41,
        )
        micro_diag = fit_density_ratio(
            micro_model,
            condition,
            positive,
            positive_weight,
            condition,
            negative,
            negative_weight,
            RatioFitConfig(
                **common,
                train_microbatch_size_per_rank=2,
            ),
            seed=41,
        )
        for full_parameter, micro_parameter in zip(
            full_model.parameters(), micro_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                full_parameter,
                micro_parameter,
                rtol=1.0e-6,
                atol=1.0e-7,
            )
        self.assertAlmostEqual(full_diag.loss, micro_diag.loss, places=6)
        self.assertAlmostEqual(
            full_diag.balanced_accuracy,
            micro_diag.balanced_accuracy,
            places=6,
        )

    def test_checkpoint_can_select_discrimination_before_later_calibration(self) -> None:
        class _LinearRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.weight * sample[..., 0]

        model = _LinearRatio()
        condition = torch.zeros(16, 1)
        positive = torch.ones(16, 1)
        negative = -torch.ones(16, 1)
        population = (
            condition,
            positive,
            torch.ones(16),
            condition,
            negative,
            torch.ones(16),
        )
        evaluations = iter(
            ((0.693, 0.50, float("nan")), (0.80, 0.60, float("nan")),
             (0.90, 0.70, float("nan")))
        )
        diagnostics = fit_density_ratio(
            model,
            *population,
            RatioFitConfig(
                steps=2,
                batch_size=16,
                learning_rate=0.1,
                weight_decay=0.0,
                sampling="independent_epoch_shuffle",
                min_steps=2,
                validation_interval_steps=1,
                validation_patience_evaluations=10,
                validation_batch_size=16,
                restore_best=True,
                checkpoint_selection_metric="balanced_accuracy",
            ),
            seed=23,
            validation=population,
            validation_evaluator=lambda _model: next(evaluations),
        )
        self.assertEqual(diagnostics.best_step, 2)
        self.assertAlmostEqual(diagnostics.validation_loss, 0.90)
        self.assertAlmostEqual(diagnostics.validation_balanced_accuracy, 0.70)
        self.assertGreater(abs(float(model.weight)), 0.0)

    def test_auc_gap_threshold_stops_an_unbounded_fit_after_one_epoch(self) -> None:
        model = ConditionalRatioMLP(
            condition_dim=2,
            sample_dim=1,
            hidden_dim=8,
            hidden_layers=1,
        )
        condition = torch.zeros(16, 2)
        positive = torch.ones(16, 1)
        negative = -torch.ones(16, 1)
        validation = (
            condition,
            positive,
            torch.ones(16),
            condition,
            negative,
            torch.ones(16),
        )
        diagnostics = fit_density_ratio(
            model,
            *validation,
            RatioFitConfig(
                steps=None,
                batch_size=4,
                sampling="independent_epoch_shuffle",
                min_steps=4,
                validation_interval_steps=4,
                validation_patience_evaluations=10,
                validation_batch_size=16,
                restore_best=True,
            ),
            seed=11,
            validation=validation,
            validation_evaluator=lambda _model: (0.69, 0.60, 0.53),
            stop_when_validation_auc_gap_exceeds=0.02,
        )
        self.assertTrue(diagnostics.threshold_reached)
        self.assertFalse(diagnostics.saturated)
        self.assertEqual(diagnostics.steps_completed, 4)
        self.assertFalse(diagnostics.hit_step_cap)

    def test_auc_gap_threshold_cannot_bypass_minimum_steps(self) -> None:
        model = ConditionalRatioMLP(
            condition_dim=2,
            sample_dim=1,
            hidden_dim=8,
            hidden_layers=1,
        )
        condition = torch.zeros(16, 2)
        positive = torch.ones(16, 1)
        negative = -torch.ones(16, 1)
        validation = (
            condition,
            positive,
            torch.ones(16),
            condition,
            negative,
            torch.ones(16),
        )
        diagnostics = fit_density_ratio(
            model,
            *validation,
            RatioFitConfig(
                steps=None,
                batch_size=4,
                sampling="independent_epoch_shuffle",
                min_steps=12,
                validation_interval_steps=4,
                validation_patience_evaluations=10,
                validation_batch_size=16,
                restore_best=True,
            ),
            seed=11,
            validation=validation,
            validation_evaluator=lambda _model: (0.69, 0.60, 0.53),
            stop_when_validation_auc_gap_exceeds=0.02,
        )
        self.assertTrue(diagnostics.threshold_reached)
        self.assertFalse(diagnostics.saturated)
        self.assertEqual(diagnostics.steps_completed, 12)

    def test_balanced_accuracy_lcb_stops_after_two_confirmations(self) -> None:
        model = ConditionalRatioMLP(
            condition_dim=2,
            sample_dim=1,
            hidden_dim=8,
            hidden_layers=1,
        )
        condition = torch.zeros(16, 2)
        positive = torch.ones(16, 1)
        negative = -torch.ones(16, 1)
        validation = (
            condition,
            positive,
            torch.ones(16),
            condition,
            negative,
            torch.ones(16),
        )
        diagnostics = fit_density_ratio(
            model,
            *validation,
            RatioFitConfig(
                steps=None,
                batch_size=4,
                sampling="independent_epoch_shuffle",
                min_steps=4,
                validation_interval_steps=4,
                validation_patience_evaluations=10,
                validation_batch_size=16,
                restore_best=True,
            ),
            seed=11,
            validation=validation,
            validation_evaluator=lambda _model: (0.69, 0.54, 0.53),
            stop_when_validation_balanced_accuracy_lcb_exceeds=0.525,
            validation_balanced_accuracy_lcb_confidence_z=1.96,
            validation_balanced_accuracy_lcb_events_per_class=10_000,
            validation_balanced_accuracy_lcb_required_consecutive=2,
        )
        self.assertTrue(diagnostics.threshold_reached)
        self.assertTrue(diagnostics.unsafe_balanced_accuracy_lcb_reached)
        self.assertEqual(
            diagnostics.threshold_reason,
            "validation_balanced_accuracy_lcb",
        )
        self.assertFalse(diagnostics.saturated)
        self.assertEqual(diagnostics.steps_completed, 8)
        self.assertEqual(
            diagnostics.validation_balanced_accuracy_lcb_streak,
            2,
        )
        self.assertGreater(
            diagnostics.validation_balanced_accuracy_lcb,
            0.525,
        )

    def test_balanced_accuracy_lcb_keeps_normal_patience_when_safe(self) -> None:
        model = ConditionalRatioMLP(
            condition_dim=2,
            sample_dim=1,
            hidden_dim=8,
            hidden_layers=1,
        )
        condition = torch.zeros(16, 2)
        positive = torch.ones(16, 1)
        negative = -torch.ones(16, 1)
        validation = (
            condition,
            positive,
            torch.ones(16),
            condition,
            negative,
            torch.ones(16),
        )
        diagnostics = fit_density_ratio(
            model,
            *validation,
            RatioFitConfig(
                steps=None,
                batch_size=4,
                sampling="independent_epoch_shuffle",
                min_steps=4,
                validation_interval_steps=4,
                validation_patience_evaluations=2,
                validation_batch_size=16,
                restore_best=True,
            ),
            seed=11,
            validation=validation,
            validation_evaluator=lambda _model: (0.69, 0.525, 0.51),
            stop_when_validation_balanced_accuracy_lcb_exceeds=0.525,
            validation_balanced_accuracy_lcb_confidence_z=1.96,
            validation_balanced_accuracy_lcb_events_per_class=10_000,
            validation_balanced_accuracy_lcb_required_consecutive=2,
        )
        self.assertTrue(diagnostics.saturated)
        self.assertFalse(diagnostics.threshold_reached)
        self.assertFalse(diagnostics.unsafe_balanced_accuracy_lcb_reached)
        self.assertEqual(diagnostics.steps_completed, 8)
        self.assertEqual(
            diagnostics.validation_balanced_accuracy_lcb_streak,
            0,
        )

    def test_min_delta_controls_patience_without_discarding_a_weak_best(self) -> None:
        class _LinearRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.weight * sample[..., 0]

        model = _LinearRatio()
        condition = torch.zeros(32, 1)
        positive = torch.full((32, 1), 0.01)
        negative = torch.full((32, 1), -0.01)
        null_loss = float(torch.log(torch.tensor(2.0)))
        diagnostics = fit_density_ratio(
            model,
            condition,
            positive,
            torch.ones(32),
            condition,
            negative,
            torch.ones(32),
            RatioFitConfig(
                steps=1,
                batch_size=16,
                learning_rate=1.0,
                weight_decay=0.0,
                min_steps=1,
                validation_interval_steps=1,
                validation_patience_evaluations=1,
                validation_min_delta=1.0,
                validation_batch_size=16,
                restore_best=True,
            ),
            seed=17,
            validation=(
                condition,
                positive,
                torch.ones(32),
                condition,
                negative,
                torch.ones(32),
            ),
        )
        self.assertTrue(diagnostics.saturated)
        self.assertEqual(diagnostics.best_step, 1)
        self.assertLess(diagnostics.validation_loss, null_loss)
        self.assertNotEqual(float(model.weight.detach()), 0.0)

    def test_zero_output_step_zero_is_preserved_as_null_checkpoint(self) -> None:
        class _ZeroRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logit = nn.Parameter(torch.zeros(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.logit.expand(sample.shape[:-1])

        model = _ZeroRatio()
        condition = torch.zeros(16, 2)
        sample = torch.zeros(16, 1)
        diagnostics = fit_density_ratio(
            model,
            condition,
            sample,
            torch.ones(16),
            condition,
            sample,
            torch.ones(16),
            RatioFitConfig(
                steps=2,
                batch_size=4,
                learning_rate=1.0e-3,
                weight_decay=0.0,
                min_steps=1,
                validation_interval_steps=1,
                validation_patience_evaluations=1,
                validation_batch_size=8,
                restore_best=True,
            ),
            seed=9,
            validation=(
                condition,
                sample,
                torch.ones(16),
                condition,
                sample,
                torch.ones(16),
            ),
        )
        self.assertEqual(diagnostics.best_step, 0)
        self.assertAlmostEqual(
            diagnostics.validation_loss,
            float(torch.log(torch.tensor(2.0))),
            places=6,
        )
        self.assertAlmostEqual(
            diagnostics.validation_balanced_accuracy, 0.5, places=6
        )
        self.assertEqual(float(model.logit.detach()), 0.0)

    def test_reports_every_ten_steps_and_at_final_step(self) -> None:
        torch.manual_seed(7)
        model = ConditionalRatioMLP(
            condition_dim=2,
            sample_dim=1,
            hidden_dim=8,
            hidden_layers=1,
        )
        condition = torch.randn(32, 2)
        positive = torch.randn(32, 1) + 0.5
        negative = torch.randn(32, 1) - 0.5
        progress: list[dict[str, float]] = []
        fit_density_ratio(
            model,
            condition,
            positive,
            torch.ones(32),
            condition,
            negative,
            torch.ones(32),
            RatioFitConfig(
                steps=21,
                batch_size=4,
                progress_interval_steps=10,
            ),
            seed=11,
            progress_callback=progress.append,
        )
        self.assertEqual([int(row["step"]) for row in progress], [10, 20, 21])
        self.assertTrue(all("training_loss" in row for row in progress))

    def test_nonfinite_gradient_fails_before_optimizer_step(self) -> None:
        class _NaNRatio(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, condition: Tensor, sample: Tensor) -> Tensor:
                del condition
                return self.weight * sample[..., 0]

        model = _NaNRatio()
        condition = torch.zeros(8, 1)
        positive = torch.full((8, 1), float("nan"))
        negative = torch.zeros(8, 1)
        with self.assertRaisesRegex(
            FloatingPointError,
            "before optimizer.step.*first_bad_rank=0.*location=positive_sample",
        ):
            fit_density_ratio(
                model,
                condition,
                positive,
                torch.ones(8),
                condition,
                negative,
                torch.ones(8),
                RatioFitConfig(
                    steps=1,
                    batch_size=8,
                    gradient_clip_norm=1.0,
                ),
                seed=13,
            )
        self.assertTrue(torch.isfinite(model.weight).all())

    def test_gradient_clip_configuration_requires_positive_finite_norm(self) -> None:
        for invalid in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "gradient_clip_norm"):
                RatioFitConfig(gradient_clip_norm=invalid).validate()


class TestK1PoolContract(unittest.TestCase):
    def test_load_pool_requires_exactly_one_four_dimensional_candidate(self) -> None:
        packed, spec = pack_event_inputs(_event_batch())
        payload = {
            "schema_version": 1,
            "kind": "ztautau_omnifold_k1_pool",
            "packing_spec": spec.to_dict(),
            "candidates_per_event": 1,
            "packed_event": packed,
            "truth": torch.randn(3, 4),
            "candidate": torch.randn(3, 1, 4),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.pt"
            torch.save(payload, path)
            loaded = load_pool(path)
            self.assertEqual(tuple(loaded["candidate"].shape), (3, 1, 4))

            payload["candidate"] = torch.randn(3, 2, 4)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "truth \\(N,4\\)"):
                load_pool(path)

    def test_materialize_pool_uses_shared_sampler_with_k1(self) -> None:
        events = _event_batch(batch_size=4)
        events.update(
            {
                "x_invisible": torch.randn(4, 2, 2),
                "x_invisible_mask": torch.ones(4, 2),
            }
        )
        arrays = {key: value.numpy() for key, value in events.items()}
        table, shapes = flatten_dict(arrays)

        class _FakePolicy(nn.Module):
            invisible_input_dim = 2

        fake_policy = _FakePolicy()
        fake_bundle = SimpleNamespace(model=fake_policy)
        calls: list[int] = []

        def _fake_generate(model, batch, sampler, **kwargs):
            del model, sampler
            calls.append(int(kwargs["K"]))
            return batch["x_invisible"].unsqueeze(0) + 0.25

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "train-diffusion"
            input_dir.mkdir()
            pq.write_table(table, input_dir / "part.parquet")
            metadata = root / "shape_metadata.json"
            metadata.write_text(json.dumps(shapes))
            output = root / "pool.pt"
            policy_checkpoint = root / "policy.ckpt"
            policy_checkpoint.write_bytes(b"fake-policy")
            with (
                mock.patch.object(
                    ztautau_stage,
                    "load_evenet_model_for_dgpo",
                    return_value=fake_bundle,
                ),
                mock.patch.object(
                    ztautau_stage,
                    "generate_neutrino_candidates",
                    side_effect=_fake_generate,
                ),
            ):
                payload = ztautau_stage.materialize_pool(
                    train_config=root / "train.yaml",
                    policy_checkpoint=policy_checkpoint,
                    input_dir=input_dir,
                    shape_metadata_path=metadata,
                    output_path=output,
                    device=torch.device("cpu"),
                    batch_size=3,
                    num_ddim_steps=5,
                    max_events=None,
                    seed=17,
                )
            self.assertEqual(calls, [1, 1])
            self.assertEqual(tuple(payload["candidate"].shape), (4, 1, 4))
            torch.testing.assert_close(
                payload["candidate"][:, 0],
                payload["truth"] + 0.25,
            )
            self.assertEqual(payload["source"]["seed"], 17)


class _FakeFrozenReward(nn.Module):
    def __init__(self, packing_spec) -> None:
        super().__init__()
        self.packing_spec = packing_spec
        self.num_iterations = 2

    def assert_frozen(self) -> None:
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("fake reward is not frozen")

    def forward(self, packed_event: Tensor, candidate_bk4: Tensor) -> Tensor:
        return candidate_bk4.sum(dim=-1) + packed_event[:, :1]


class _FakeConsensusFrozenReward(_FakeFrozenReward):
    def __init__(self, packing_spec) -> None:
        super().__init__(packing_spec)
        self.num_iterations = 1
        self.num_checkpoints = 4

    def member_logits(self, packed_event: Tensor, candidate_bk4: Tensor) -> Tensor:
        base = self.forward(packed_event, candidate_bk4)
        return torch.stack((base, base, -base, -base))

    def ensemble_from_member_logits(self, member_logits: Tensor) -> Tensor:
        # Keep a nonzero current direction while member votes remain split.
        return member_logits[0]


class TestDgpoOmniFoldReward(unittest.TestCase):
    def test_candidate_consensus_gate_changes_only_disagreed_advantages(self) -> None:
        batch = _event_batch(batch_size=2)
        packed, spec = pack_event_inputs(batch)
        del packed
        source = ZtautauOmniFoldReward(
            _FakeConsensusFrozenReward(spec).eval(),
            bundle_sha256="b" * 64,
            policy_reference_sha256="a" * 64,
            base_digest="c" * 64,
            stack_sha256="d" * 64,
            bundle_schema_version=1,
            device=torch.device("cpu"),
            reward_round_id=1,
            candidate_consensus={
                "enabled": True,
                "apply_to_reward": True,
                "temporal_history_rounds": 0,
                "minimum_sign_agreement": 0.75,
                "uncertainty_scale": 1.0,
                "epsilon": 1.0e-8,
                "fixed_panel_events_per_rank": 2,
                "fixed_panel_candidates": 3,
            },
        )
        # The first validation call uses a different validation_K and must not
        # capture the fixed DGPO training panel.
        validation_batch = dict(batch)
        source.compute(torch.randn(3, 2, 2, 2), validation_batch)
        self.assertFalse(
            any(
                key.startswith("reward_rank_audit/")
                for key in source.last_interface_metrics()
            )
        )
        candidates = torch.randn(3, 2, 2, 2)
        training_batch = dict(batch)
        training_batch["_dgpo_reward_context"] = "policy_update"
        scores = source.compute(candidates, training_batch)
        torch.testing.assert_close(scores, torch.zeros_like(scores))
        metrics = source.last_interface_metrics()
        self.assertEqual(metrics["reward_consensus/applied"], 1.0)
        self.assertEqual(metrics["reward_consensus/live/zeroed_fraction"], 1.0)
        self.assertEqual(
            metrics["reward_rank_audit/fixed_panel/reward_round_id"], 1.0
        )

    def test_simplified_wandb_profile_keeps_topology_acceptance_evidence(self) -> None:
        from RL.DGPO_neutrino import dgpo_trainer

        expected = {
            "omnifold/topology_acceptance_audit_enabled",
            "omnifold/topology_acceptance_max_auc_gap",
            "omnifold/topology_acceptance_repeats",
            "omnifold/candidate/topology_audit_auc",
            "omnifold/candidate/topology_audit_auc_gap",
            "omnifold/candidate/topology_audit_balanced_accuracy",
            "omnifold/candidate/topology_audit_saturated",
            "omnifold/candidate/topology_audit_same_architecture",
            "omnifold/candidate/topology_audit_fit_events",
            "omnifold/candidate/topology_audit_test_events",
            "omnifold/candidate/topology_audit_auc_gap_se",
            "omnifold/candidate/topology_audit_repeat_01_auc_gap",
            "reference_trust/round_acceptance/improvement",
            "reference_trust/round_acceptance/stop_requested",
        }
        self.assertTrue(
            all(
                dgpo_trainer._wandb_simplified_keep(key, 1.0)
                for key in expected
            )
        )

    def test_in_dgpo_bootstrap_forwards_trainable_projector_flag(self) -> None:
        from RL.DGPO_neutrino import dgpo_trainer

        config = SimpleNamespace(
            reward_config=SimpleNamespace(
                type="omnifold",
                weight=1.0,
                omnifold=SimpleNamespace(
                    bootstrap_in_dgpo=True,
                    bundle_file=None,
                    backbone_checkpoint="unused-by-mock.ckpt",
                    candidate_consensus={
                        "enabled": True,
                        "apply_to_reward": False,
                    },
                ),
            ),
            dgpo=SimpleNamespace(
                adaptive_omnifold=SimpleNamespace(
                    recalibration=SimpleNamespace(
                        body_only_checkpoint=True,
                        train_last_pet_block=True,
                        train_grouped_sequential_embedding=True,
                        train_invisible_projector=True,
                        periodic_pair_features=True,
                        topology_fourier_embedding=True,
                        topology_conditioning=True,
                    )
                )
            ),
        )
        with mock.patch.object(dgpo_trainer, "global_config", config), mock.patch.object(
            dgpo_reward_module,
            "build_uninstalled_ztautau_omnifold_reward",
            return_value=mock.sentinel.reward,
        ) as build_reward:
            aggregator = dgpo_trainer.build_reward_aggregator(
                nn.Identity(),
                torch.device("cpu"),
                normalization_dict={},
            )

        classifier_config = build_reward.call_args.kwargs["classifier_config"]
        self.assertEqual(
            build_reward.call_args.kwargs["candidate_consensus"],
            {"enabled": True, "apply_to_reward": False},
        )
        self.assertTrue(classifier_config["body_only_checkpoint"])
        self.assertTrue(classifier_config["train_last_pet_block"])
        self.assertTrue(
            classifier_config["train_grouped_sequential_embedding"]
        )
        self.assertTrue(classifier_config["train_invisible_projector"])
        self.assertTrue(classifier_config["topology_conditioning"])
        self.assertIs(aggregator.sources[0][0], mock.sentinel.reward)

    def _reward(self, batch: dict[str, Tensor], policy_sha: str = "a" * 64):
        _packed, spec = pack_event_inputs(batch)
        stack = _FakeFrozenReward(spec).eval()
        return ZtautauOmniFoldReward(
            stack,
            bundle_sha256="b" * 64,
            policy_reference_sha256=policy_sha,
            base_digest="c" * 64,
            stack_sha256="d" * 64,
            bundle_schema_version=1,
            device=torch.device("cpu"),
        )

    def test_scores_every_k_candidate_without_reading_event_truth(self) -> None:
        batch = _event_batch(batch_size=3)
        batch["x_invisible"] = torch.randn(3, 2, 2)
        batch["x_invisible_mask"] = torch.ones(3, 2)
        candidates = torch.randn(5, 3, 2, 2)
        reward = self._reward(batch)
        first = reward.compute(candidates, batch)
        changed_truth = dict(batch)
        changed_truth["x_invisible"] = torch.randn_like(batch["x_invisible"]) * 100.0
        second = reward.compute(candidates, changed_truth)
        self.assertEqual(tuple(first.shape), (5, 3))
        torch.testing.assert_close(first, second)

    def test_uninstalled_source_accepts_first_in_dgpo_stack_atomically(self) -> None:
        batch = _event_batch(batch_size=2)
        _packed, spec = pack_event_inputs(batch)
        classifier = EvenetAdapterRatioClassifier(
            _FakeZtautauBackbone(),
            spec,
            train_backbone=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        frozen = FrozenResidualRatioReward(
            classifier,
            (classifier.state_dict(),),
        )
        source = ZtautauOmniFoldReward(
            None,
            bundle_sha256="b" * 64,
            policy_reference_sha256="a" * 64,
            base_digest=classifier.base_digest,
            stack_sha256="",
            bundle_schema_version=1,
            device=torch.device("cpu"),
        )
        self.assertFalse(source.is_installed)
        with self.assertRaisesRegex(RuntimeError, "before the in-process"):
            source.compute(torch.randn(3, 2, 2, 2), batch)
        source.replace_stack(
            frozen,
            round_id=1,
            reference_sha256="e" * 64,
        )
        self.assertTrue(source.is_installed)
        self.assertEqual(source.reward_round_id, 1)
        self.assertEqual(source.iterations, 1)

    def test_checkpoint_metadata_and_reference_pairing_fail_closed(self) -> None:
        batch = _event_batch(batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.ckpt"
            policy.write_bytes(b"policy-a")
            policy_sha = sha256_file(policy)
            reward = self._reward(batch, policy_sha=policy_sha)
            aggregator = RewardAggregator()
            aggregator.add(reward, 1.0)
            metadata = aggregator.checkpoint_metadata()
            self.assertIsNotNone(metadata)
            validate_omnifold_reward_startup(
                checkpoint=None,
                current_metadata=metadata,
                policy_checkpoint=policy,
            )

            wrong_policy = root / "wrong.ckpt"
            wrong_policy.write_bytes(b"policy-b")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_omnifold_reward_startup(
                    checkpoint=None,
                    current_metadata=metadata,
                    policy_checkpoint=wrong_policy,
                )

            uninstalled = copy.deepcopy(metadata)
            uninstalled_meta = uninstalled["sources"][0]["metadata"]
            uninstalled_meta["reward_round_id"] = 0
            uninstalled_meta["iterations"] = 0
            uninstalled_meta["stack_sha256"] = ""
            # A fresh in-process bootstrap has no denominator yet. Its frozen
            # classifier backbone may differ from weights-only warm-start policy.
            validate_omnifold_reward_startup(
                checkpoint=None,
                current_metadata=uninstalled,
                policy_checkpoint=wrong_policy,
            )

            validate_omnifold_reward_startup(
                checkpoint={
                    "dgpo_checkpoint_version": 1,
                    REWARD_CHECKPOINT_KEY: metadata,
                },
                current_metadata=metadata,
                policy_checkpoint=wrong_policy,
            )
            migrated = copy.deepcopy(metadata)
            migrated["sources"][0]["metadata"]["bundle_sha256"] = "f" * 64
            validate_omnifold_reward_startup(
                checkpoint={
                    "dgpo_checkpoint_version": 1,
                    REWARD_CHECKPOINT_KEY: metadata,
                },
                current_metadata=migrated,
                policy_checkpoint=policy,
                allow_source_bundle_migration=True,
            )
            with self.assertRaisesRegex(ValueError, "different OmniFold reward"):
                validate_omnifold_reward_startup(
                    checkpoint={
                        "dgpo_checkpoint_version": 1,
                        REWARD_CHECKPOINT_KEY: metadata,
                    },
                    current_metadata=migrated,
                    policy_checkpoint=policy,
                )
            changed = dict(metadata)
            changed["sources"] = [dict(metadata["sources"][0], weight=2.0)]
            with self.assertRaisesRegex(ValueError, "different OmniFold reward"):
                validate_omnifold_reward_startup(
                    checkpoint={
                        "dgpo_checkpoint_version": 1,
                        REWARD_CHECKPOINT_KEY: metadata,
                    },
                    current_metadata=changed,
                    policy_checkpoint=policy,
                )

    def test_bundle_loader_restores_the_serialized_ratio_stack(self) -> None:
        batch = _event_batch(batch_size=2)
        packed, spec = pack_event_inputs(batch)
        backbone = _FakeZtautauBackbone()
        classifier = EvenetAdapterRatioClassifier(
            backbone,
            spec,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
        )
        with torch.no_grad():
            classifier.bank.output.weight.fill_(0.1)
        first_fold = {
            key: value.detach().clone()
            for key, value in classifier.state_dict().items()
        }
        with torch.no_grad():
            classifier.bank.output.weight.fill_(0.2)
        second_fold = {
            key: value.detach().clone()
            for key, value in classifier.state_dict().items()
        }
        frozen = FrozenResidualRatioReward(
            classifier,
            (first_fold, second_fold),
            tempering=1.0,
            checkpoint_coefficients=(0.5, 0.5),
            checkpoint_iterations=(1, 1),
            iteration_temperatures=(0.5,),
        )
        reward_payload = frozen.serializable_payload()
        self.assertEqual(reward_payload["iteration_temperatures"], [0.5])
        class _FakeBuilder:
            def __init__(self):
                self.default_hidden = 8
                self.default_layers = 1
                self.default_heads = 2
                self.default_bottleneck = 4
                self.backbone = backbone
                self._adapter_bottleneck = 4

            def make_classifier(self, packing_spec, **kwargs):
                return EvenetAdapterRatioClassifier(
                    backbone,
                    packing_spec,
                    head_dropout=kwargs.get("head_dropout"),
                    decoder_hidden_dim=kwargs.get(
                        "decoder_hidden_dim", self.default_hidden
                    ),
                    decoder_layers=kwargs.get(
                        "decoder_layers", self.default_layers
                    ),
                    decoder_heads=kwargs.get(
                        "decoder_heads", self.default_heads
                    ),
                    adapter_bottleneck=kwargs.get(
                        "adapter_bottleneck", self.default_bottleneck
                    ),
                    bank_name=kwargs.get("name"),
                )

        fake_builder = _FakeBuilder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path = root / "omnifold_reward.pt"
            backbone_path = root / "backbone.ckpt"
            backbone_path.write_bytes(b"fake-backbone")
            torch.save(
                {
                    "schema_version": 1,
                    "kind": "ztautau_evenet_omnifold_reward",
                    "candidates_per_event_for_fit": 1,
                    "classifier": {
                        "adapter_placement": "internal",
                        "adapter_bottleneck": 4,
                        "decoder_hidden_dim": 8,
                        "decoder_layers": 1,
                        "decoder_heads": 2,
                        "head_dropout": 0.0,
                    },
                    "provenance": {"policy_reference_sha256": "d" * 64},
                    "reward": reward_payload,
                },
                bundle_path,
            )
            with mock.patch.object(
                dgpo_reward_module,
                "EvenetAdapterModelBuilder",
                return_value=fake_builder,
            ):
                loaded = load_ztautau_omnifold_reward(
                    bundle_file=bundle_path,
                    backbone_checkpoint=backbone_path,
                    training_config=object(),
                    normalization_dict={},
                    device=torch.device("cpu"),
                    expected_iterations=1,
                )
                consensus_loaded = load_ztautau_omnifold_reward(
                    bundle_file=bundle_path,
                    backbone_checkpoint=backbone_path,
                    training_config=object(),
                    normalization_dict={},
                    device=torch.device("cpu"),
                    expected_iterations=1,
                    candidate_consensus={
                        "enabled": True,
                        "apply_to_reward": True,
                        "temporal_history_rounds": 1,
                        "minimum_sign_agreement": 0.75,
                        "uncertainty_scale": 1.0,
                        "epsilon": 1.0e-8,
                        "fixed_panel_events_per_rank": 2,
                        "fixed_panel_candidates": 3,
                    },
                )
        candidates = torch.randn(3, 2, 2, 2)
        batch["x_invisible"] = torch.randn(2, 2, 2)
        batch["x_invisible_mask"] = torch.ones(2, 2)
        self.assertEqual(tuple(loaded.compute(candidates, batch).shape), (3, 2))
        self.assertEqual(loaded.iterations, 1)
        self.assertEqual(loaded.frozen_reward.num_checkpoints, 2)
        self.assertEqual(loaded.frozen_reward.iteration_temperatures, (0.5,))
        loaded.replace_stack(
            loaded.frozen_reward,
            round_id=1,
            reference_sha256="e" * 64,
        )
        dynamic_payload = loaded.stack_payload()
        loaded.load_stack_payload(dynamic_payload)
        resumed_payload = loaded.stack_payload()
        self.assertEqual(
            resumed_payload["stack_sha256"], dynamic_payload["stack_sha256"]
        )
        self.assertEqual(
            resumed_payload["reward"]["iteration_temperatures"], [0.5]
        )
        self.assertEqual(loaded.reward_round_id, 1)
        self.assertEqual(loaded.reference_kind, "state_dict_sha256")
        self.assertEqual(loaded.policy_reference_sha256, "e" * 64)

        # The fixed candidate panel, previous reward, and audit trajectory are
        # all recovery state. Verify a round boundary survives serialization.
        consensus_loaded.replace_stack(
            consensus_loaded.frozen_reward,
            round_id=1,
            reference_sha256="1" * 64,
        )
        consensus_training_batch = dict(batch)
        consensus_training_batch["_dgpo_reward_context"] = "policy_update"
        first_consensus = consensus_loaded.compute(
            candidates, consensus_training_batch
        )
        consensus_loaded.replace_stack(
            consensus_loaded.frozen_reward,
            round_id=2,
            reference_sha256="2" * 64,
        )
        second_consensus = consensus_loaded.compute(
            candidates, consensus_training_batch
        )
        consensus_payload = consensus_loaded.stack_payload()
        self.assertEqual(len(consensus_payload["consensus_history"]), 1)
        self.assertEqual(len(consensus_payload["rank_audit_history"]), 2)
        self.assertEqual(
            tuple(consensus_payload["rank_audit_panel"]["candidates"].shape),
            (2, 3, 4),
        )
        consensus_loaded.load_stack_payload(consensus_payload)
        torch.testing.assert_close(
            consensus_loaded.compute(candidates, consensus_training_batch),
            second_consensus,
        )
        self.assertEqual(len(consensus_loaded.rank_audit_history), 2)
        self.assertEqual(tuple(first_consensus.shape), (3, 2))
        legacy_payload = copy.deepcopy(dynamic_payload)
        for increment in legacy_payload["reward"]["increments"]:
            increment.pop("classifier_config", None)
        legacy_payload["stack_sha256"] = dgpo_reward_module.payload_sha256(
            legacy_payload["reward"]
        )
        fake_builder.default_hidden = 4
        with self.assertRaisesRegex(
            ValueError,
            "predates the internal-adapter classifier schema",
        ):
            loaded.load_stack_payload(legacy_payload)
        corrupted = dict(dynamic_payload)
        corrupted["stack_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            loaded.load_stack_payload(corrupted)

        migrated_source = copy.deepcopy(dynamic_payload)
        migrated_source["source_bundle_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different reward bundle"):
            loaded.load_stack_payload(migrated_source)
        migration_target = ZtautauOmniFoldReward(
            None,
            bundle_sha256="a" * 64,
            policy_reference_sha256="d" * 64,
            base_digest=str(migrated_source["reward"]["base_digest"]),
            stack_sha256="",
            bundle_schema_version=1,
            device=torch.device("cpu"),
            model_builder=fake_builder,
        )
        migration_target.load_stack_payload(
            migrated_source,
            allow_source_bundle_migration=True,
        )
        self.assertTrue(migration_target.is_installed)

    def test_resume_keeps_parent_body_keys_when_fork_opens_encoder(self) -> None:
        """Fork YAML may set train_encoder/LN; parent restore must not expand body."""

        batch = _event_batch(batch_size=2)
        _, spec = pack_event_inputs(batch)
        parent_backbone = _FakeZtautauBackbone()
        parent = EvenetAdapterRatioClassifier(
            parent_backbone,
            spec,
            train_layernorm=False,
            train_encoder=False,
            train_grouped_sequential_embedding=True,
            train_invisible_projector=True,
            decoder_hidden_dim=8,
            decoder_layers=1,
            decoder_heads=2,
            adapter_bottleneck=4,
            head_dropout=0.25,
            bank_name="adaptive_reward",
        )
        self.assertFalse(
            any(name.startswith("GlobalEmbedding") for name in parent._backbone_state_keys)
        )
        checkpoint = {
            key: value.detach().clone() for key, value in parent.state_dict().items()
        }
        frozen = FrozenResidualRatioReward(
            parent,
            (checkpoint,),
            tempering=0.75,
            checkpoint_coefficients=(1.0,),
            checkpoint_iterations=(1,),
        )
        reward_payload = frozen.serializable_payload()
        stack_sha = dgpo_reward_module.payload_sha256(reward_payload)

        class _ForkBuilder:
            """Simulates a fork builder whose defaults open LN + GlobalEmbedding."""

            def make_classifier(self, packing_spec, **kwargs):
                train_layernorm = kwargs.get("train_layernorm", True)
                train_encoder = kwargs.get("train_encoder", True)
                return EvenetAdapterRatioClassifier(
                    _FakeZtautauBackbone(),
                    packing_spec,
                    train_layernorm=bool(train_layernorm),
                    train_encoder=bool(train_encoder),
                    train_grouped_sequential_embedding=bool(
                        kwargs.get("train_grouped_sequential_embedding", True)
                    ),
                    train_invisible_projector=bool(
                        kwargs.get("train_invisible_projector", True)
                    ),
                    train_backbone=bool(kwargs.get("train_backbone", False)),
                    head_dropout=kwargs.get("head_dropout", 0.15),
                    decoder_hidden_dim=int(kwargs.get("decoder_hidden_dim", 8)),
                    decoder_layers=int(kwargs.get("decoder_layers", 1)),
                    decoder_heads=int(kwargs.get("decoder_heads", 2)),
                    adapter_bottleneck=int(kwargs.get("adapter_bottleneck", 4)),
                    bank_name=kwargs.get("name"),
                    # Keep the same body digest as the parent payload; production
                    # builders share one pretrained snapshot, so this matches.
                    base_digest=str(reward_payload["base_digest"]),
                )

        target = ZtautauOmniFoldReward(
            None,
            bundle_sha256="a" * 64,
            policy_reference_sha256="d" * 64,
            base_digest=str(reward_payload["base_digest"]),
            stack_sha256="",
            bundle_schema_version=1,
            device=torch.device("cpu"),
            model_builder=_ForkBuilder(),
        )
        payload = {
            "schema_version": 1,
            "kind": "ztautau_adaptive_omnifold_stack",
            "source_bundle_sha256": "b" * 64,
            "stack_sha256": stack_sha,
            "reward_round_id": 1,
            "policy_reference_sha256": "c" * 64,
            "reference_kind": "state_dict_sha256",
            "reward": reward_payload,
        }
        # Without the restore-flag fix this raises backbone trainable-state mismatch.
        target.load_stack_payload(payload, allow_source_bundle_migration=True)
        self.assertTrue(target.is_installed)
        self.assertEqual(target.stack_payload()["stack_sha256"], stack_sha)
        restored = target.frozen_reward.classifier
        self.assertFalse(restored._train_encoder)
        self.assertFalse(restored._train_layernorm)
        self.assertFalse(
            any(
                name.startswith("GlobalEmbedding")
                for name in restored._backbone_state_keys
            )
        )


if __name__ == "__main__":
    unittest.main()
