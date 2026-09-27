"""Two-stage residual fitting, best-stage restore, LR groups and launch contract."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest import mock

import torch
import yaml

from . import evenet_ratio as ratio
from . import ratio_fit
from .adaptive import resolve_adaptive_config
from .test_rest_frame import batch, classifier
from ..dgpo_trainer import _ClassifierFitLossTracker


def fixture():
    packed, spec = ratio.pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
    packed = packed.repeat(4, 1)
    sample = torch.randn(len(packed), 4)
    weights = torch.ones(len(packed))
    inputs = (packed, sample, weights, packed, sample + .4, weights)
    model = classifier(spec, layers=2, dropout=.15)
    cfg = ratio_fit.RatioFitConfig(steps=6, batch_size=4, learning_rate=5e-5,
                                   backbone_learning_rate=1e-5, min_steps=1,
                                   sampling="independent_epoch_shuffle", drop_last_batch=True,
                                   validation_interval_steps=1, validation_patience_evaluations=10,
                                   validation_min_delta=1e-4, restore_best=True)
    return model, inputs, cfg, inputs


class TestStagedResidual(unittest.TestCase):
    def test_selection_and_stage_b_starts_from_best_a_without_reset(self):
        for b_loss, expected_stage in ((.7, 1), (.59995, 1), (.55, 2)):
            with self.subTest(b_loss=b_loss):
                model, inputs, cfg, validation = fixture()
                original = copy.deepcopy(model.state_dict())
                saved, calls, progress = {}, [], []

                def fit(m, *args, progress_callback=None):
                    config, seed, val = args[-3:]
                    stage = len(calls) + 1
                    calls.append(stage)
                    self.assertIs(val, validation)
                    self.assertEqual(seed, 42)
                    for got, expected in zip(args[:6], inputs):
                        self.assertIs(got, expected)
                    m.train()
                    trainable = [n for n,p in m.named_parameters() if p.requires_grad]
                    if stage == 1:
                        self.assertTrue(all(n.startswith("bank.output.") for n in trainable))
                        self.assertEqual(m.bank.output.weight.abs().sum().item(), 0.)
                        self.assertTrue(all(not block.training for block in m.bank.decoder.blocks))
                        self.assertIsNone(config.decoder_learning_rate)
                    else:
                        for k,v in m.state_dict().items():
                            torch.testing.assert_close(v, saved[1][k], atol=0, rtol=0)
                        self.assertTrue(any(n.startswith("bank.decoder.blocks.1.") for n in trainable))
                        self.assertFalse(m.bank.decoder.blocks[0].training)
                        self.assertTrue(m.bank.decoder.blocks[1].training)
                        self.assertEqual(config.decoder_learning_rate, 1e-5)
                        self.assertEqual(config.learning_rate, 5e-5)
                    self.assertFalse(any(p.requires_grad for p in m.backbone.parameters()))
                    with torch.no_grad():
                        for p in m.parameters():
                            if p.requires_grad:
                                p.add_(.1 * stage)
                    saved[stage] = copy.deepcopy(m.state_dict())
                    progress_callback({"step": 1., "training_loss": .6})
                    return ratio_fit.RatioFitDiagnostics(loss=.5, balanced_accuracy=.6, steps=None,
                        steps_completed=10 * stage, best_step=3, validation_loss=.6 if stage == 1 else b_loss,
                        validation_auc=.7, saturated=True,
                        validation_history=({"step": 3., "loss": .6 if stage == 1 else b_loss},))

                with mock.patch.object(ratio_fit, "fit_density_ratio", side_effect=fit):
                    diag = ratio.fit_staged_residual_classifier(model, *inputs, cfg, 42, validation,
                        decoder_learning_rate=1e-5, progress_callback=progress.append)
                self.assertEqual(calls, [1, 2])
                self.assertEqual(diag.selected_stage, expected_stage)
                self.assertEqual(diag.steps_completed, 30)
                self.assertEqual(diag.best_step, 3 if expected_stage == 1 else 13)
                self.assertEqual([r["fit_stage"] for r in progress[:2]], [1., 2.])
                self.assertEqual(progress[-1]["selected_stage"], expected_stage)
                for k,v in model.state_dict().items():
                    torch.testing.assert_close(v, saved[expected_stage][k], atol=0, rtol=0)
                    if not k.startswith(("bank.output.", "bank.decoder.blocks.1.")):
                        torch.testing.assert_close(v, original[k], atol=0, rtol=0)
                self.assertFalse(model.training)

    def test_real_training_parameter_groups_frozen_features_and_payload(self):
        torch.manual_seed(42)
        model, inputs, cfg, validation = fixture()
        before = copy.deepcopy(model.state_dict())
        template = copy.deepcopy(model.backbone)
        original_init = torch.optim.AdamW.__init__
        optimizers = []
        def record(optimizer, *args, **kwargs):
            original_init(optimizer, *args, **kwargs)
            optimizers.append(optimizer)
        with mock.patch.object(torch.optim.AdamW, "__init__", record):
            diag = ratio.fit_staged_residual_classifier(model, *inputs, cfg, 42, validation, decoder_learning_rate=1e-5)
        self.assertEqual(len(optimizers), 2)
        output_ids = {id(p) for p in model.bank.output.parameters()}
        decoder_ids = {id(p) for p in model.bank.decoder.blocks[-1].parameters()}
        groups_a = {id(p):group["lr"] for group in optimizers[0].param_groups for p in group["params"]}
        groups_b = {id(p):group["lr"] for group in optimizers[1].param_groups for p in group["params"]}
        self.assertEqual(set(groups_a), output_ids)
        self.assertEqual(set(groups_b), output_ids | decoder_ids)
        self.assertTrue(all(groups_b[p] == 5e-5 for p in output_ids))
        self.assertTrue(all(groups_b[p] == 1e-5 for p in decoder_ids))
        for key,value in model.state_dict().items():
            if not key.startswith(("bank.output.", "bank.decoder.blocks.1.")):
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertEqual(set(model.state_dict()), set(before))
        self.assertLessEqual(diag.validation_loss, diag.stage_a_validation_loss)
        self.assertEqual(diag.steps_completed, 12)
        # Frozen inherited body tensors remain part of portable reward state.
        self.assertTrue(any("GroupedSequentialEmbedding" in k for k in model.state_dict()))
        payload = model.peft_payload()
        restored = ratio.EvenetAdapterRatioClassifier.from_peft_payload(
            payload, model_builder=lambda spec: classifier(spec, template=template, layers=2, dropout=.15), device=torch.device("cpu"))
        restored.eval()
        torch.testing.assert_close(model(inputs[0], inputs[1]), restored(inputs[0], inputs[1]), atol=0, rtol=0)

    def test_stack_stages_only_later_iterations_and_keeps_iteration_one_cache(self):
        model, inputs, cfg, validation = fixture()
        cfg = replace(cfg, steps=None, min_steps=1, require_saturation=True)
        spec = model.packing_spec
        normal_calls, staged_calls = [], []
        def normal(m, *args, **kwargs):
            normal_calls.append(m)
            self.assertFalse(getattr(m, "_residual_last_block_only", False))
            return ratio_fit.RatioFitDiagnostics(loss=.6, balanced_accuracy=.6, steps=None, steps_completed=1, saturated=True)
        def staged(m, *args, **kwargs):
            staged_calls.append(m)
            self.assertEqual(kwargs["decoder_learning_rate"], 1e-5)
            self.assertEqual(args[-3].learning_rate, 5e-5)
            m.configure_residual_last_block_training(output_only=True)
            return ratio_fit.RatioFitDiagnostics(loss=.6, balanced_accuracy=.6, steps=None, steps_completed=2, saturated=True, selected_stage=1)
        with mock.patch.object(ratio_fit, "fit_density_ratio", side_effect=normal), mock.patch.object(ratio, "fit_staged_residual_classifier", side_effect=staged), mock.patch.object(ratio, "_weighted_binary_score_metrics", side_effect=[(.6,.6,.7),(.69,.5,.5)]):
            result = ratio.fit_residual_ratio_stack(model_factory=lambda: classifier(spec, layers=2),
                data_condition=inputs[0], data_sample=inputs[1], gen_condition=inputs[3], gen_sample=inputs[4],
                iterations=2, fit_config=cfg, tempering=1., seed=42,
                identity_condition=torch.arange(len(inputs[0]), dtype=torch.float32).reshape(-1, 1),
                warm_start_iterations=(1,), warm_start_from_iteration_one=True,
                later_iteration_learning_rate=5e-5, later_iteration_train_mode="output_then_last_decoder",
                later_iteration_decoder_learning_rate=1e-5,
                validation_data_condition=inputs[0], validation_data_sample=inputs[1],
                validation_gen_condition=inputs[3], validation_gen_sample=inputs[4])
        self.assertEqual((len(normal_calls), len(staged_calls)), (2, 2))
        self.assertEqual(len(result.warm_start_state["models"]), 2)
        self.assertTrue(all(saved["iteration"] == 1 for saved in result.warm_start_state["models"]))

    def test_validation_failures_and_stage_series(self):
        model, inputs, cfg, val = fixture()
        for lr in (None, True, -1., float("nan"), 1e-3):
            with self.assertRaises(ValueError):
                ratio.fit_staged_residual_classifier(model, *inputs, cfg, 42, val, decoder_learning_rate=lr)
        with self.assertRaises(ValueError):
            ratio.fit_staged_residual_classifier(model, *inputs, replace(cfg, restore_best=False), 42, val, decoder_learning_rate=1e-5)
        with self.assertRaises(ValueError):
            replace(cfg, decoder_learning_rate=float("nan")).validate()
        with mock.patch.object(ratio_fit, "fit_density_ratio", return_value=ratio_fit.RatioFitDiagnostics(
                loss=.6, balanced_accuracy=.6, steps=None, validation_loss=.6, saturated=False)):
            with self.assertRaisesRegex(RuntimeError, "stage 1 did not saturate"):
                ratio.fit_staged_residual_classifier(model, *inputs, replace(cfg, require_saturation=True), 42, val, decoder_learning_rate=1e-5)
        tracker, wb = _ClassifierFitLossTracker(), mock.Mock()
        a = tracker.payload(wb, "residual_reward", {"step": 1, "training_loss": .6, "fit_stage": 1}, global_step=0, epoch=-1)
        b = tracker.payload(wb, "residual_reward", {"step": 1, "training_loss": .5, "fit_stage": 2}, global_step=0, epoch=-1)
        self.assertFalse(set(a) & set(b))
        self.assertTrue(all("_stage1/" in k for k in a))
        self.assertTrue(all("_stage2/" in k for k in b))

    def test_new_yaml_changes_only_residual_fit_and_output_paths(self):
        root = Path(__file__).resolve().parents[4] / "config"
        old = yaml.safe_load((root / "dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit.yaml").read_text())
        new = yaml.safe_load((root / "dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_staged.yaml").read_text())
        expected = copy.deepcopy(old["dgpo"])
        expected["adaptive_omnifold"]["recalibration"]["fit"].update(
            later_iteration_train_mode="output_then_last_decoder", later_iteration_decoder_learning_rate=1e-5)
        self.assertEqual(new["dgpo"], expected)
        for key in ("platform", "network", "reward_config"):
            self.assertEqual(new[key], old[key])
        cfg = resolve_adaptive_config(new["dgpo"])
        self.assertFalse(cfg.raw_rollback_to_best_on_plateau)
        self.assertEqual(cfg.fit["later_iteration_decoder_learning_rate"], 1e-5)
        self.assertEqual(new["options"]["Training"]["model_checkpoint_load_path"], old["options"]["Training"]["model_checkpoint_load_path"])
        self.assertNotEqual(new["options"]["Training"]["model_checkpoint_save_path"], old["options"]["Training"]["model_checkpoint_save_path"])
        for lr in (None, -1., 1e-3, True):
            bad = copy.deepcopy(new["dgpo"])
            bad["adaptive_omnifold"]["recalibration"]["fit"]["later_iteration_decoder_learning_rate"] = lr
            with self.assertRaises(ValueError):
                resolve_adaptive_config(bad)


if __name__ == "__main__":
    unittest.main()
