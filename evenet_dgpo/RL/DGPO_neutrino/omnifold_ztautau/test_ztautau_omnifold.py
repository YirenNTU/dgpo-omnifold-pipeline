"""Focused tests for the standalone Ztautau OmniFold integration."""

from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pyarrow.parquet as pq
import torch
from torch import Tensor, nn

from evenet.dataset.preprocess import flatten_dict
from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio as evenet_ratio_module
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterRatioClassifier,
    EvenetAdapterModelBuilder,
    FrozenResidualRatioReward,
    fit_residual_ratio_stack,
    pack_event_inputs,
    unpack_event_inputs,
)
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
            [evenet_ratio_module.Adapter(8, bottleneck=4, dropout=0.0)]
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
        return x * mask

    def project_invisible_inputs(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.InvisibleInputProjector(x) * mask


def _event_batch(batch_size: int = 3) -> dict[str, Tensor]:
    return {
        "x": torch.randn(batch_size, 5, 4),
        "x_mask": torch.ones(batch_size, 5, 1, dtype=torch.bool),
        "conditions": torch.randn(batch_size, 2),
        "conditions_mask": torch.ones(batch_size, dtype=torch.bool),
    }


class TestEventPacking(unittest.TestCase):
    def test_round_trip_is_lossless(self) -> None:
        batch = _event_batch()
        packed, spec = pack_event_inputs(batch)
        restored = unpack_event_inputs(packed, spec)
        for key, expected in batch.items():
            torch.testing.assert_close(restored[key], expected)


class TestZtautauRatioClassifier(unittest.TestCase):
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
        self.assertEqual(tuple(backbone.PET.last_attn_mask.shape), (7, 7))

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
        original["classifier_config"].pop("train_invisible_projector")
        original["classifier_config"].pop("asymmetric_attention")
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            original,
            model_builder=build,
            device=torch.device("cpu"),
        )
        roundtrip = restored.peft_payload()
        self.assertNotIn(
            "train_invisible_projector", roundtrip["classifier_config"]
        )
        self.assertNotIn("asymmetric_attention", roundtrip["classifier_config"])
        self.assertTrue(
            all(name.startswith("PET.adapters.") for name in roundtrip["body"])
        )


class TestResidualRatioStack(unittest.TestCase):
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


class TestDgpoOmniFoldReward(unittest.TestCase):
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
                ),
            ),
            dgpo=SimpleNamespace(
                adaptive_omnifold=SimpleNamespace(
                    recalibration=SimpleNamespace(
                        train_invisible_projector=True,
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
        self.assertTrue(classifier_config["train_invisible_projector"])
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
        )
        reward_payload = frozen.serializable_payload()
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
        candidates = torch.randn(3, 2, 2, 2)
        batch["x_invisible"] = torch.randn(2, 2, 2)
        batch["x_invisible_mask"] = torch.ones(2, 2)
        self.assertEqual(tuple(loaded.compute(candidates, batch).shape), (3, 2))
        self.assertEqual(loaded.iterations, 1)
        self.assertEqual(loaded.frozen_reward.num_checkpoints, 2)
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
        self.assertEqual(loaded.reward_round_id, 1)
        self.assertEqual(loaded.reference_kind, "state_dict_sha256")
        self.assertEqual(loaded.policy_reference_sha256, "e" * 64)
        legacy_payload = copy.deepcopy(dynamic_payload)
        for increment in legacy_payload["reward"]["increments"]:
            increment.pop("classifier_config", None)
        legacy_payload["stack_sha256"] = dgpo_reward_module.payload_sha256(
            legacy_payload["reward"]
        )
        fake_builder.default_hidden = 4
        loaded.load_stack_payload(legacy_payload)
        self.assertEqual(
            loaded.frozen_reward.classifier._decoder_hidden_dim,
            8,
        )
        self.assertEqual(
            loaded.stack_payload()["stack_sha256"],
            legacy_payload["stack_sha256"],
        )
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

if __name__ == "__main__":
    unittest.main()
