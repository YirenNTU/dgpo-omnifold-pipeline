"""Physics, leakage, architecture, serialization and launch-contract regressions."""
from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import vector
import yaml

from .rest_frame import (
    REST_FRAME_DIM, REST_FRAME_KEY, REST_FRAME_FEATURE_NAMES,
    boost_to_rest, mass_squared, visible_pair_rest_features,
)
from .evenet_ratio import (
    EvenetAdapterModelBuilder, EvenetAdapterRatioClassifier, pack_event_inputs,
    peft_bank_factory, unpack_event_inputs,
    event_identity_inputs, _identity_crossfit_splits,
    fit_independent_evenet_audit,
    fit_residual_ratio_stack,
)
from .test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch_with_pair_context
from .adaptive import AdaptiveOmniFoldPool, split_single_classifier_pool, resolve_adaptive_config, fit_raw_policy_audit
from .ratio_fit import RatioFitConfig, _uses_fast_ratio_learning_rate, fit_density_ratio


def legs():
    a = torch.tensor([[12., 3., 4., 8.], [7., -2., 1., 4.], [4., 0., 0., 3.]], dtype=torch.float64)
    b = torch.tensor([[10., -2., -1., -6.], [9., 4., 1., -5.], [4., 0., 0., -3.]], dtype=torch.float64)
    return a, b


def batch():
    result = _event_batch_with_pair_context()
    for leg, p4 in zip(("a", "b"), legs()):
        for index, component in enumerate(("E", "px", "py", "pz")):
            result[f"lead_{leg}_visible_{component}"] = p4[:, index].float()
    return result


def classifier(spec, *, rest=True, template=None, layers=1, dropout=0.):
    return EvenetAdapterRatioClassifier(
        copy.deepcopy(template) if template is not None else _FakeZtautauBackbone(), spec,
        periodic_pair_features=True, topology_fourier_embedding=True,
        topology_conditioning=True, topology_max_harmonic=4, topology_include_theta_pair=True,
        visible_pair_rest_frame=rest, topology_hidden_dim=16, topology_embedding_dim=8,
        topology_dropout=dropout, head_dropout=dropout, decoder_hidden_dim=8, decoder_layers=layers,
        decoder_heads=2, train_grouped_sequential_embedding=True, train_invisible_projector=True,
    )


def ablation_payload():
    root = Path(__file__).resolve().parents[4]
    return yaml.safe_load((root/'config/dgpo_omnifold_ztautau_10pct_arch_fourier_visible_rest.yaml').read_text())


class TestLorentzBoost(unittest.TestCase):
    def test_sign_zero_boost_and_invariant_mass(self):
        a, b = legs()
        frame = a + b
        result = boost_to_rest(frame, frame)
        torch.testing.assert_close(result[:, 1:], torch.zeros_like(result[:, 1:]), atol=1e-12, rtol=0)
        torch.testing.assert_close(result[:, 0], mass_squared(frame).sqrt())
        torch.testing.assert_close(mass_squared(boost_to_rest(a, frame)), mass_squared(a))
        rest = torch.zeros_like(frame)
        rest[:, 0] = 3.
        torch.testing.assert_close(boost_to_rest(a, rest), a, atol=0, rtol=0)

    def test_agrees_with_independent_vector_library(self):
        a, b = legs()
        for q, f in ((a, a + b), (b, a + b), (a, a)):
            qv = vector.array(dict(E=q[:, 0].numpy(), px=q[:, 1].numpy(), py=q[:, 2].numpy(), pz=q[:, 3].numpy()))
            fv = vector.array(dict(E=f[:, 0].numpy(), px=f[:, 1].numpy(), py=f[:, 2].numpy(), pz=f[:, 3].numpy()))
            expected = qv.boostCM_of_p4(fv)
            expected = np.stack((expected.E, expected.px, expected.py, expected.pz), -1)
            torch.testing.assert_close(boost_to_rest(q, f), torch.from_numpy(expected), atol=1e-11, rtol=1e-11)

    def test_broadcast_inverse_and_gradcheck(self):
        a, b = legs()
        frame = a + b
        q = torch.stack((a, b), 1)
        result = boost_to_rest(q, frame[:, None, :])
        inverse = frame.clone()
        inverse[:, 1:] *= -1
        torch.testing.assert_close(boost_to_rest(result, inverse[:, None, :]), q)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, f: boost_to_rest(x, f),
            (a.clone().requires_grad_(), frame.clone().requires_grad_()),
        ))

    def test_nonphysical_and_nonfinite_frame_rejected(self):
        a, b = legs()
        for bad in (torch.tensor([1., 0., 0., 1.]), torch.tensor([-2., 0., 0., 0.]),
                    torch.tensor([1., 2., 0., 0.]), torch.tensor([float('nan'), 0., 0., 0.])):
            with self.assertRaises(ValueError):
                boost_to_rest(a, bad)


class TestVisibleFrameFeatures(unittest.TestCase):
    def test_dimensions_names_and_angle_bounds(self):
        a, b = legs()
        f = visible_pair_rest_features(a, b)
        self.assertEqual(REST_FRAME_DIM, 8)
        self.assertEqual(tuple(f.shape), (3, REST_FRAME_DIM))
        self.assertTrue(torch.isfinite(f).all())
        self.assertTrue((f[:, 5:8].abs() <= 1.).all())
        self.assertEqual(len(set(REST_FRAME_FEATURE_NAMES)), REST_FRAME_DIM)
        self.assertFalse(any('valid' in name for name in REST_FRAME_FEATURE_NAMES))
        self.assertEqual(REST_FRAME_FEATURE_NAMES, (
            'log1p_pair_mass_gev', 'beta_z', 'beta_transverse',
            'asinh_mass2_a_over_pair_mass2', 'asinh_mass2_b_over_pair_mass2',
            'cos_a_cs_z', 'cos_a_cs_x', 'cos_a_cs_y',
        ))

    def test_all_eight_channels_match_independent_direct_calculation(self):
        # Nondegenerate physical events, independently boosted with vector.
        # Covers the transforms/axis conventions, not only boost or dimensions.
        a, b = (q[:2].numpy() for q in legs())
        expected = []
        def v(q):
            return vector.obj(E=q[0], px=q[1], py=q[2], pz=q[3])
        def spatial(q):
            return np.array([q.px, q.py, q.pz])
        def unit(q):
            return q / np.linalg.norm(q)
        def m2(q):
            return q[0] ** 2 - np.dot(q[1:], q[1:])
        for qa, qb in zip(a, b):
            total = qa + qb
            frame = v(total)
            mass = np.sqrt(m2(total))
            astar = v(qa).boostCM_of_p4(frame)
            plus = unit(spatial(v([1., 0., 0., 1.]).boostCM_of_p4(frame)))
            minus = unit(spatial(v([1., 0., 0., -1.]).boostCM_of_p4(frame)))
            z, x = unit(plus - minus), unit(plus + minus)
            y = np.cross(z, x)
            direction = unit(spatial(astar))
            expected.append([
                np.log1p(mass),
                total[3] / total[0], np.linalg.norm(total[1:3]) / total[0],
                np.arcsinh(m2(qa) / mass**2), np.arcsinh(m2(qb) / mass**2),
                np.dot(direction, z), np.dot(direction, x), np.dot(direction, y),
            ])
        torch.testing.assert_close(
            visible_pair_rest_features(torch.from_numpy(a), torch.from_numpy(b)),
            torch.tensor(np.array(expected)), atol=1e-11, rtol=1e-11,
        )

    def test_zero_recoil_has_no_arbitrary_transverse_axis(self):
        f = visible_pair_rest_features(*legs())
        torch.testing.assert_close(f[-1, 6:8], torch.zeros(2, dtype=f.dtype))
        self.assertAlmostEqual(float(f[-1, 5]), 1.)
        self.assertAlmostEqual(float(f[-1, 1]), 0.)

    def test_rotation_about_beam_leaves_scalars_unchanged(self):
        a, b = legs()
        angle = .73
        def rotate(q):
            out = q.clone()
            out[:, 1] = np.cos(angle) * q[:, 1] - np.sin(angle) * q[:, 2]
            out[:, 2] = np.sin(angle) * q[:, 1] + np.cos(angle) * q[:, 2]
            return out
        torch.testing.assert_close(visible_pair_rest_features(rotate(a), rotate(b)),
                                   visible_pair_rest_features(a, b), atol=1e-10, rtol=1e-10)

    def test_invalid_finite_frame_neutral_and_nonfinite_fails(self):
        a = torch.tensor([[1., 0., 0., 1.], [1., 2., 0., 0.], [0., 0., 0., 0.]])
        f = visible_pair_rest_features(a, a)
        torch.testing.assert_close(f, torch.zeros_like(f))
        a[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            visible_pair_rest_features(a, a)

    def test_feature_gradients_are_finite(self):
        a, b = legs()
        a, b = a[:2].requires_grad_(), b[:2].requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(visible_pair_rest_features, (a, b)))


class TestRestFrameClassifier(unittest.TestCase):
    def test_partial_residual_fit_modes_optimizer_and_frozen_checkpoint_tensors(self):
        torch.manual_seed(42)
        data = {k: v.repeat(8, *([1] * (v.ndim - 1))) for k, v in batch().items()}
        packed, spec = pack_event_inputs(data, include_pairwise_context=True, include_visible_pair_rest_frame=True)
        template = _FakeZtautauBackbone()
        first = classifier(spec, template=template, layers=2, dropout=.15)
        # Simulate a fitted iteration 1 with open AdaLN gates and nonzero logits.
        with torch.no_grad():
            first.bank.output.weight.normal_(std=.1)
            first.bank.output.bias.fill_(.3)
            for block in first.bank.decoder.blocks:
                block.modulation.proj.weight.normal_(std=.05)
        source = {k:v.detach().clone() for k,v in first.state_dict().items()}
        residual = classifier(spec, template=template, layers=2, dropout=.15)
        residual.load_state_dict(source, strict=True)
        residual.configure_residual_last_block_training()
        allowed = ('bank.decoder.blocks.1.', 'bank.output.')
        trainable = {n:p for n,p in residual.named_parameters() if p.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all(n.startswith(allowed) for n in trainable))
        self.assertTrue(any(n.startswith(allowed[0]) for n in trainable))
        self.assertEqual(residual.trainable_parameter_counts['total'], sum(p.numel() for p in trainable.values()))
        self.assertEqual(residual.trainable_parameter_counts['invisible_projector'], 0)
        self.assertEqual(residual.trainable_parameter_counts['grouped_sequential_embedding'], 0)
        self.assertEqual(set(residual.state_dict()), set(source))
        for name, value in residual.state_dict().items():
            torch.testing.assert_close(value, torch.zeros_like(value) if name.startswith('bank.output.') else source[name])
        for mode in (True, False, True):
            residual.train(mode)
            self.assertEqual(residual.training, mode)
            self.assertTrue(all(not m.training for m in residual.backbone.modules()))
            for name, module in residual.bank.named_modules():
                active = name.startswith(('decoder.blocks.1', 'output'))
                self.assertEqual(module.training, mode if active else False, name)
        pos, neg = torch.randn(len(packed), 4), torch.randn(len(packed), 4) + .8
        captured = []
        hook = residual.bank.decoder.blocks[0].register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach().clone()))
        with torch.no_grad():
            torch.testing.assert_close(residual(packed, pos), torch.zeros(len(packed)))
            residual(packed, pos)
        hook.remove()
        torch.testing.assert_close(captured[0], captured[1], atol=0, rtol=0)
        before = {k:v.detach().clone() for k,v in residual.state_dict().items()}
        body_before = {k:v.detach().clone() for k,v in residual.backbone.state_dict().items()}
        optimizers = []
        original_init = torch.optim.AdamW.__init__
        def record(optimizer, *args, **kwargs):
            original_init(optimizer, *args, **kwargs)
            optimizers.append(optimizer)
        weights = torch.ones(len(packed))
        cfg = RatioFitConfig(steps=8, batch_size=8, learning_rate=5e-5,
                             backbone_learning_rate=1e-5, weight_decay=.0005,
                             sampling='independent_epoch_shuffle', min_steps=1,
                             validation_interval_steps=2, validation_patience_evaluations=20,
                             restore_best=False)
        with mock.patch.object(torch.optim.AdamW, '__init__', record):
            fit_density_ratio(residual, packed, pos, weights, packed, neg, weights, cfg, 42,
                              (packed, pos, weights, packed, neg, weights))
        self.assertEqual(len(optimizers), 1)
        self.assertEqual({id(p) for g in optimizers[0].param_groups for p in g['params']},
                         {id(p) for p in trainable.values()})
        self.assertTrue(all(g['lr'] == 5e-5 for g in optimizers[0].param_groups))
        after = residual.state_dict()
        for name, value in before.items():
            if not name.startswith(allowed):
                torch.testing.assert_close(after[name], value, atol=0, rtol=0)
        for prefix in allowed:
            self.assertTrue(any(not torch.equal(after[k], before[k]) for k in before if k.startswith(prefix)))
        for name, value in residual.backbone.state_dict().items():
            torch.testing.assert_close(value, body_before[name], atol=0, rtol=0)
        # Frozen inherited body tensors survive both state_dict and PEFT export.
        exported = residual.peft_payload()
        self.assertTrue(any(k.startswith('GroupedSequentialEmbedding.') for k in exported['body']))
        self.assertTrue(any(k.startswith('InvisibleInputProjector.') for k in exported['body']))
        self.assertTrue(any(k.startswith('PET.adapters.') for k in exported['body']))
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            exported, model_builder=lambda s: classifier(s, template=template, layers=2, dropout=.15),
            device=torch.device('cpu')).eval()
        residual.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(packed, pos), residual(packed, pos))
        # Freezing weights must not sever candidate-input derivatives used by
        # policy-gradient diagnostics / differentiable reward evaluation.
        differentiable_sample = pos.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(residual(packed, differentiable_sample).sum(), differentiable_sample)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, source[key], atol=0, rtol=0)
        # A separate raw monitor and the original iteration 1 remain unfrozen.
        for model in (first, classifier(spec, template=template, layers=2, dropout=.15)):
            model.train()
            self.assertTrue(all(p.requires_grad for p in model.backbone.GroupedSequentialEmbedding.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.bank.decoder.blocks[0].parameters()))

    def test_residual_stack_partially_trains_only_iterations_after_one(self):
        from types import SimpleNamespace
        from . import evenet_ratio as module
        data = {k: v.repeat(20, *([1] * (v.ndim - 1))) for k, v in batch().items()}
        data['conditions'] = torch.randn_like(data['conditions'])
        packed, spec = pack_event_inputs(data, include_pairwise_context=True, include_visible_pair_rest_frame=True)
        sample = torch.randn(len(packed), 4)
        template = _FakeZtautauBackbone()
        cfg = RatioFitConfig(steps=None, min_steps=1, batch_size=4, drop_last_batch=True,
                             sampling='independent_epoch_shuffle', learning_rate=2e-4,
                             validation_interval_steps=20, validation_patience_evaluations=5,
                             require_saturation=True)
        first_states, calls = {}, []
        def fit(model, dc, ds, dw, gc, gs, gw, fit_config, *args, **kwargs):
            iteration, fold = len(calls) // 2 + 1, len(calls) % 2 + 1
            calls.append((iteration, fold))
            model.train()
            self.assertEqual(getattr(model, '_residual_last_block_only', False), iteration > 1)
            self.assertEqual(fit_config.learning_rate, 2e-4 if iteration == 1 else 5e-5)
            self.assertEqual(fit_config.min_steps, 1000 if iteration == 1 else 10 * (len(dc) // 4))
            self.assertEqual(model.bank.output.weight.abs().sum().item(), 0.)
            if iteration > 1:
                for key, value in model.state_dict().items():
                    if not key.startswith('bank.output.'):
                        torch.testing.assert_close(value, first_states[fold][key], atol=0, rtol=0)
                self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
                self.assertFalse(model.bank.decoder.blocks[0].training)
                self.assertTrue(model.bank.decoder.blocks[-1].training)
            with torch.no_grad():
                model.bank.output.bias.fill_(float(len(calls)))
                model.bank.decoder.blocks[-1].modulation.proj.bias.add_(.1 * len(calls))
            if iteration == 1:
                first_states[fold] = {k:v.detach().clone() for k,v in model.state_dict().items()}
            return SimpleNamespace(saturated=True, loss=.6, balanced_accuracy=.7, steps_completed=fit_config.min_steps)
        with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.ratio_fit.fit_density_ratio', side_effect=fit), \
             mock.patch.object(module, '_weighted_binary_score_metrics',
                               side_effect=[(.6,.7,.7),(.6,.7,.7),(.693,.5,.5)]):
            result = fit_residual_ratio_stack(
                model_factory=lambda: classifier(spec, template=template, layers=2, dropout=.15),
                data_condition=packed, data_sample=sample, gen_condition=packed, gen_sample=sample,
                iterations=3, fit_config=cfg, tempering=1., seed=42,
                warm_start_iterations=(1,), warm_start_from_iteration_one=True,
                later_iteration_learning_rate=5e-5, later_iteration_train_mode='last_decoder_and_output',
                min_steps_per_fold=1000, warm_start_min_epochs_per_fold=10,
                validation_interval_epochs=2., validation_patience_epochs=10.,
                validation_data_condition=packed, validation_data_sample=sample,
                validation_gen_condition=packed, validation_gen_sample=sample)
        self.assertEqual(calls, [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2)])
        self.assertEqual([d.warm_started_folds for d in result.diagnostics], [(),(1,2),(1,2)])
        self.assertEqual(len(result.warm_start_state['models']), 2)
        for saved in result.warm_start_state['models']:
            for key, value in saved['state'].items():
                torch.testing.assert_close(value, first_states[saved['fold']][key], atol=0, rtol=0)

    def test_production_builder_shares_architecture_inputs_but_not_trainable_weights(self):
        payload = ablation_payload()
        cfg = resolve_adaptive_config(payload['dgpo'])
        recal = payload['dgpo']['adaptive_omnifold']['recalibration']
        settings = {k: v for k, v in recal.items()
                    if k in inspect.signature(EvenetAdapterModelBuilder).parameters}
        packed, spec = pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
        with tempfile.NamedTemporaryFile(suffix='.ckpt') as checkpoint, \
             mock.patch('RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio._config_with_pet_adapters', side_effect=lambda c, _: c), \
             mock.patch('RL.DGPO_neutrino.model_utils.build_evenet_on_device', side_effect=lambda *args: _FakeZtautauBackbone()), \
             mock.patch('RL.DGPO_neutrino.model_utils.load_weights_like_configure_model'):
            builder = EvenetAdapterModelBuilder(config=object(), normalization_dict={},
                checkpoint_path=checkpoint.name, device=torch.device('cpu'), **settings)
            reward = builder.make_classifier(spec, 'adaptive_reward').eval()
            monitor = peft_bank_factory(builder, spec, 'audit')().eval()
            self.assertEqual(reward.peft_payload()['classifier_config'], monitor.peft_payload()['classifier_config'])
            self.assertEqual(reward.packing_spec, monitor.packing_spec)
            self.assertEqual(reward.trainable_parameter_counts, monitor.trainable_parameter_counts)
            self.assertEqual(reward.bank.pairwise_feature_dim, 11)
            self.assertEqual(monitor.bank.rest_frame_encoder[1].in_features, 8)
            self.assertEqual(monitor.bank.rest_frame_encoder[1].out_features, 64)
            self.assertEqual(monitor.bank.rest_frame_encoder[4].out_features, 32)
            self.assertTrue(monitor.bank.topology_conditioning)
            self.assertEqual(monitor.bank.output.in_features, 256)
            self.assertEqual(len(monitor.bank.decoder.blocks), 2)
            self.assertEqual(len(reward.bank.decoder.blocks), 2)
            self.assertEqual(monitor.bank.decoder.hidden_dim, 128)
            for block in monitor.bank.decoder.blocks:
                self.assertEqual(block.self_attn.num_heads, 4)
                self.assertEqual(block.cross_attn.num_heads, 4)
                self.assertEqual(block.self_attn.dropout, .15)
                self.assertEqual(block.cross_attn.dropout, .15)
            for name, p in reward.named_parameters():
                other = dict(monitor.named_parameters())[name]
                self.assertEqual(p.requires_grad, other.requires_grad)
                if p.requires_grad:
                    self.assertNotEqual(p.data_ptr(), other.data_ptr())
            # Matching complete state gives identical outputs on the same
            # packed visible inputs and candidates (test only, not weight reuse).
            with torch.no_grad():
                reward.bank.output.weight.normal_(std=.1)
                for block in reward.bank.decoder.blocks:
                    block.modulation.proj.weight.normal_(std=.1)
            monitor.load_state_dict(reward.state_dict(), strict=True)
            candidate = torch.randn(3, 2, 4)
            torch.testing.assert_close(reward(packed, candidate), monitor(packed, candidate), atol=0, rtol=0)
            # Both decoder layers participate in backprop once the initially
            # zero AdaLN gates open. This is test-only initialization.
            reward(packed, candidate).square().mean().backward()
            for block in reward.bank.decoder.blocks:
                for parameter in (block.self_attn.in_proj_weight,
                                  block.cross_attn.in_proj_weight,
                                  block.ffn[0].weight, block.modulation.proj.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            reward.zero_grad(set_to_none=True)

            # Exercise the real raw-monitor fitting path, not just its factory.
            # Only the synthetic-test training budget is shortened; architecture
            # and inputs are taken from the actual launch YAML.
            data = {k: v.repeat(20, *([1] * (v.ndim - 1))) for k, v in batch().items()}
            data['conditions'] = torch.randn_like(data['conditions'])
            events, pool_spec = pack_event_inputs(data, include_pairwise_context=True, include_visible_pair_rest_frame=True)
            samples = torch.randn(60, 4)
            pool = AdaptiveOmniFoldPool(events, samples, samples[:, None, :], pool_spec)
            cache = {}
            before = {k: v.detach().clone() for k, v in reward.state_dict().items()}
            captures = []
            reset = builder.reset_audit_bank
            def capture(s):
                model = reset(s)
                captures.append(model)
                return model
            with mock.patch.object(builder, 'reset_audit_bank', side_effect=capture):
                for index in range(2):
                    result = fit_raw_policy_audit(
                        pool=pool, model_builder=builder, cfg=cfg, device=torch.device('cpu'),
                        seed=42, warm_start_cache=cache, use_training_readiness=False,
                        fit_overrides={'steps': 2, 'batch_size': 8, 'train_microbatch_size_per_rank': 8,
                                       'validation_interval_steps': 1, 'validation_batch_size': 64},
                    )
                    self.assertEqual(result['raw_classifier_warm_started'], float(index == 1))
            self.assertEqual(len(captures), 2)
            for fitted in captures:
                self.assertEqual(fitted.peft_payload()['classifier_config'], reward.peft_payload()['classifier_config'])
                self.assertEqual(fitted.packing_spec, spec)
            self.assertTrue(any(k.startswith('bank.rest_frame_encoder.') for k in cache['state']))
            self.assertTrue(any(k.startswith('bank.topology_encoder.') for k in cache['state']))
            self.assertTrue(any(k.startswith('bank.decoder.blocks.1.') for k in cache['state']))
            for key, value in reward.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_real_builder_separates_reward_and_clean_monitor(self):
        _, spec = pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
        with tempfile.NamedTemporaryFile(suffix='.ckpt') as checkpoint, \
             mock.patch('RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio._config_with_pet_adapters', side_effect=lambda cfg, _: cfg), \
             mock.patch('RL.DGPO_neutrino.model_utils.build_evenet_on_device', side_effect=lambda *args: _FakeZtautauBackbone()), \
             mock.patch('RL.DGPO_neutrino.model_utils.load_weights_like_configure_model'):
            builder = EvenetAdapterModelBuilder(
                config=object(), normalization_dict={}, checkpoint_path=checkpoint.name,
                device=torch.device('cpu'), visible_pair_rest_frame=True,
                periodic_pair_features=True, topology_fourier_embedding=True, topology_conditioning=True,
                topology_max_harmonic=4, topology_include_theta_pair=True,
                decoder_hidden_dim=8, decoder_layers=1, decoder_heads=2,
                train_grouped_sequential_embedding=True, train_invisible_projector=True,
            )
            reward = builder.make_classifier(spec)
            monitor = peft_bank_factory(builder, spec, 'audit', classifier_overrides={
                'visible_pair_rest_frame': False, 'periodic_pair_features': False,
                'topology_fourier_embedding': False, 'topology_conditioning': False,
            })()
        self.assertTrue(reward.bank.visible_pair_rest_frame)
        self.assertEqual(reward.bank.rest_frame_encoder[0].normalized_shape, (8,))
        self.assertEqual(reward.bank.rest_frame_encoder[1].in_features, 8)
        self.assertEqual(reward.bank.decoder.blocks[0].modulation.proj.in_features, 8 + 32 + 32)
        self.assertFalse(monitor.bank.visible_pair_rest_frame)
        self.assertFalse(hasattr(monitor.bank, 'rest_frame_encoder'))
        self.assertEqual(monitor.bank.decoder.blocks[0].modulation.proj.in_features, 8)
        self.assertIsNot(reward.backbone, monitor.backbone)

    def test_packing_has_no_truth_or_supplied_feature_leakage(self):
        data = batch()
        old, old_spec = pack_event_inputs(data, include_pairwise_context=True)
        packed, spec = pack_event_inputs(data, include_pairwise_context=True, include_visible_pair_rest_frame=True)
        self.assertEqual(spec.width, old_spec.width + REST_FRAME_DIM + 2)
        torch.testing.assert_close(packed[:, :old_spec.width], old, atol=0, rtol=0)
        poisoned = {**data, 'truth_tau_a_p4': torch.full((3, 4), float('nan')),
                    'x_invisible': torch.randn(3, 2, 2), REST_FRAME_KEY: torch.full((3, REST_FRAME_DIM), 999.)}
        repeated, _ = pack_event_inputs(poisoned, spec)
        torch.testing.assert_close(packed, repeated, atol=0, rtol=0)
        del poisoned['lead_a_visible_E']
        with self.assertRaisesRegex(KeyError, 'visible E,px,py,pz'):
            pack_event_inputs(poisoned, spec)

    def test_derived_rest_features_preserve_identity_folds_and_repack(self):
        data = batch()
        # Unique original conditions, with identical geometry, across 60 events.
        data = {k: v.repeat(20, *([1] * (v.ndim - 1))) for k, v in data.items()}
        data['conditions'] = torch.randn_like(data['conditions'])
        old, old_spec = pack_event_inputs(data, include_pairwise_context=True)
        new, new_spec = pack_event_inputs(data, include_pairwise_context=True, include_visible_pair_rest_frame=True)
        torch.testing.assert_close(event_identity_inputs(new, new_spec), old, atol=0, rtol=0)
        for folds in (2, 5):
            old_folds = _identity_crossfit_splits(old, folds=folds, seed=42)
            new_folds = _identity_crossfit_splits(event_identity_inputs(new, new_spec), folds=folds, seed=42)
            for left, right in zip(old_folds, new_folds):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
        truth, candidates = torch.randn(60, 4), torch.randn(60, 1, 4)
        old_pool = AdaptiveOmniFoldPool(old, truth, candidates, old_spec)
        new_pool = AdaptiveOmniFoldPool(new, truth, candidates, new_spec)
        for left, right in zip(split_single_classifier_pool(old_pool, seed=42), split_single_classifier_pool(new_pool, seed=42)):
            torch.testing.assert_close(left.truth, right.truth, atol=0, rtol=0)
        torch.testing.assert_close(new_pool.repack(old_spec).packed_event, old)
        # P4 energies are retained so repacking can recompute, not trust a derived column.
        reversed_spec = type(new_spec)(dict(reversed(list(new_spec.shapes.items()))))
        torch.testing.assert_close(new_pool.repack(reversed_spec).repack(new_spec).packed_event, new)

    def test_classifier_forward_candidate_gradient_and_complete_warm_load(self):
        packed, spec = pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
        template = _FakeZtautauBackbone()
        model = classifier(spec, template=template).eval()
        self.assertGreater(model.trainable_parameter_counts['rest_frame_encoder'], 0)
        self.assertTrue(_uses_fast_ratio_learning_rate('bank.rest_frame_encoder.1.weight'))
        # Exercise the whole learned path, not the initial exactly-null readout.
        with torch.no_grad():
            model.bank.output.weight.normal_(std=.1)
            model.bank.decoder.blocks[0].modulation.proj.weight.normal_(std=.1)
        candidate = torch.randn(3, 2, 4, requires_grad=True)
        scores = model(packed, candidate)
        self.assertEqual(tuple(scores.shape), (3, 2))
        scores.square().sum().backward()
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.abs().sum()), 0)
        for group in (model.bank.rest_frame_encoder, model.bank.topology_encoder,
                      model.backbone.GroupedSequentialEmbedding, model.backbone.InvisibleInputProjector,
                      model.backbone.PET.adapters):
            self.assertTrue(all(p.requires_grad for p in group.parameters()))
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in group.parameters()))
        warm = classifier(spec, template=template).eval()
        warm.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(warm(packed, candidate.detach()), scores.detach())
        # A fresh AdamW updates the inherited rest-frame weights too.
        before = warm.bank.rest_frame_encoder[1].weight.detach().clone()
        optimizer = torch.optim.AdamW(warm.parameters(), lr=.0002)
        self.assertFalse(optimizer.state)
        warm(packed, candidate.detach()).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, warm.bank.rest_frame_encoder[1].weight))

    def test_raw_fitter_uses_canonical_ids_and_warm_cache(self):
        data = {k: v.repeat(20, *([1] * (v.ndim - 1))) for k, v in batch().items()}
        data['conditions'] = torch.randn_like(data['conditions'])
        packed, spec = pack_event_inputs(data, include_pairwise_context=True, include_visible_pair_rest_frame=True)
        ids = event_identity_inputs(packed, spec)
        fit_idx, val_idx = _identity_crossfit_splits(ids, folds=5, seed=42)[0]
        cache = {}
        sample = torch.randn(60, 4)
        cfg = RatioFitConfig(steps=2, batch_size=8, validation_interval_steps=1,
                             validation_patience_evaluations=2,
                             validation_batch_size=64, restore_best=True)
        def fit():
            return fit_independent_evenet_audit(
                model_factory=lambda: classifier(spec), data_condition=packed, data_sample=sample,
                gen_condition=packed, gen_sample=sample[:, None, :], gen_weight=torch.ones(60, 1),
                fit_config=cfg, seed=123, reuse_early_stop_for_audit=True,
                identity_split_seed=42, identity_condition=ids, warm_start_cache=cache,
            )
        first = fit()
        self.assertFalse(first.warm_started)
        self.assertEqual(first.fit_events, len(fit_idx))
        self.assertEqual(first.early_stop_events, len(val_idx))
        self.assertEqual(cache['protocol']['identity_scheme'], 'exclude-visible-pair-rest-v1')
        self.assertTrue(any(k.startswith('bank.rest_frame_encoder.') for k in cache['state']))
        self.assertTrue(fit().warm_started)

    def test_identical_candidates_cannot_be_distinguished_by_event_context(self):
        packed, spec = pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
        model = classifier(spec).eval()
        with torch.no_grad():
            model.bank.output.weight.normal_()
            model.bank.decoder.blocks[0].modulation.proj.weight.normal_()
        candidate = torch.randn(3, 4)
        together = model(packed, torch.stack((candidate, candidate), 1))
        torch.testing.assert_close(together[:, 0], together[:, 1], atol=0, rtol=0)
        changed = packed.clone()
        changed[:, -REST_FRAME_DIM:] += .1
        self.assertGreater(float((model(changed, candidate) - model(packed, candidate)).abs().max()), 0.)

    def test_payload_roundtrip_and_old_architecture_fail_closed(self):
        packed, spec = pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
        template = _FakeZtautauBackbone()
        model = classifier(spec, template=template).eval()
        payload = model.peft_payload()
        self.assertTrue(payload['classifier_config']['visible_pair_rest_frame'])
        restored = EvenetAdapterRatioClassifier.from_peft_payload(
            payload, model_builder=lambda s: classifier(s, template=template), device=torch.device('cpu'),
        ).eval()
        self.assertEqual(payload['classifier_config'], restored.peft_payload()['classifier_config'])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        with self.assertRaisesRegex(ValueError, 'unknown keys'):
            EvenetAdapterRatioClassifier.from_peft_payload(
                payload, model_builder=lambda s: classifier(s, rest=False, template=template), device=torch.device('cpu'))
        clean, clean_spec = pack_event_inputs(batch(), include_pairwise_context=True)
        with self.assertRaisesRegex(ValueError, 'versioned rest-frame'):
            classifier(clean_spec)
        for legacy_key, legacy_dim in (('visible_pair_rest_v1', 18), ('visible_pair_rest_v2', 15)):
            legacy_spec = type(spec)({
                (legacy_key if k == REST_FRAME_KEY else k):
                ((legacy_dim,) if k == REST_FRAME_KEY else shape)
                for k, shape in spec.shapes.items()
            })
            with self.assertRaisesRegex(ValueError, 'versioned rest-frame'):
                classifier(legacy_spec)
        old = classifier(clean_spec, rest=False, template=template)
        self.assertNotIn('visible_pair_rest_frame', old.peft_payload()['classifier_config'])

    def test_monitor_override_and_new_config_preserve_control(self):
        root = Path(__file__).resolve().parents[4]
        original = yaml.safe_load((root/'config/dgpo_omnifold_ztautau_10pct_arch_fourier_conditioning.yaml').read_text())
        new = yaml.safe_load((root/'config/dgpo_omnifold_ztautau_10pct_arch_fourier_visible_rest.yaml').read_text())
        self.assertNotEqual(original['options']['Training']['model_checkpoint_load_path'], new['options']['Training']['model_checkpoint_load_path'])
        source = Path(new['options']['Training']['model_checkpoint_load_path'])
        self.assertEqual(source.name, 'dgpo-epoch=31-next_ep=32-step=320.ckpt')
        self.assertIn('dgpo_omnifold_10pct_v26_resume3_', str(source))
        self.assertEqual(new['nersc']['reproducibility']['source_wandb_run'].split('/')[-1], 'f6b4ec46')
        self.assertNotEqual(original['options']['Training']['model_checkpoint_save_path'], new['options']['Training']['model_checkpoint_save_path'])
        a = new['dgpo']['adaptive_omnifold']
        cfg = resolve_adaptive_config(new['dgpo'])
        self.assertTrue(cfg.visible_pair_rest_frame_enabled)
        self.assertTrue(cfg.raw_monitor_warm_start)
        self.assertTrue(cfg.cache_event_inputs and cfg.fixed_audit_panel)
        self.assertEqual(cfg.single_pool_split_seed, 42)
        self.assertEqual(cfg.pool_selection_seed, 42)
        self.assertEqual(cfg.seed, 20260906)
        self.assertEqual(cfg.crossfit_partition, 'identity')
        self.assertTrue(new['logger']['wandb']['classifier_loss_curves'])
        for fit_block in (a['audit_fit'], a['recalibration']['fit']):
            self.assertTrue(fit_block['drop_last_batch'])
            self.assertEqual(fit_block['sampling'], 'independent_epoch_shuffle')
            self.assertEqual(fit_block['validation_interval_epochs'], 2.)
            self.assertEqual(fit_block['validation_patience_epochs'], 10.)
        self.assertEqual(cfg.warm_start_iterations, (1,))
        self.assertTrue(cfg.warm_start_from_iteration_one)
        for bad in ('true', 1, None):
            invalid = copy.deepcopy(new['dgpo'])
            invalid['adaptive_omnifold']['recalibration']['warm_start_from_iteration_one'] = bad
            with self.assertRaisesRegex(ValueError, 'must be a boolean'):
                resolve_adaptive_config(invalid)
        invalid = copy.deepcopy(new['dgpo'])
        invalid['adaptive_omnifold']['recalibration']['warm_start_iterations'] = [1, 2]
        with self.assertRaisesRegex(ValueError, 'requires warm_start_iterations'):
            resolve_adaptive_config(invalid)
        self.assertEqual(cfg.raw_patience_schedule, ((0, 6), (100, 10), (300, 16), (600, 24)))
        self.assertIn('visible_rest8_samearch_l2_psched_', new['options']['Training']['model_checkpoint_save_path'])
        expected_fit = dict(resolve_adaptive_config(original['dgpo']).fit)
        expected_fit.update(warm_start_min_epochs_per_fold=10, validation_interval_epochs=2.,
                            later_iteration_learning_rate=5e-5,
                            later_iteration_train_mode='last_decoder_and_output')
        self.assertEqual(cfg.fit, expected_fit)
        self.assertEqual(cfg.fit['min_steps_per_fold'], 1000)
        self.assertEqual(cfg.fit['learning_rate'], 2e-4)
        self.assertEqual(cfg.fit['backbone_learning_rate'], 1e-5)
        self.assertEqual(a['audit_fit']['learning_rate'], 2e-4)
        self.assertNotIn('later_iteration_train_mode', a['audit_fit'])
        for mode in ('head_only', None, True):
            invalid = copy.deepcopy(new['dgpo'])
            invalid['adaptive_omnifold']['recalibration']['fit']['later_iteration_train_mode'] = mode
            with self.assertRaisesRegex(ValueError, 'later_iteration_train_mode'):
                resolve_adaptive_config(invalid)
        for bad_lr in (0, -1, True, '0.00005', float('nan'), float('inf')):
            invalid = copy.deepcopy(new['dgpo'])
            invalid['adaptive_omnifold']['recalibration']['fit']['later_iteration_learning_rate'] = bad_lr
            with self.assertRaisesRegex(ValueError, 'later_iteration_learning_rate'):
                resolve_adaptive_config(invalid)
        invalid = copy.deepcopy(new['dgpo'])
        invalid['adaptive_omnifold']['recalibration']['warm_start_from_iteration_one'] = False
        with self.assertRaisesRegex(ValueError, 'later_iteration_.* requires'):
            resolve_adaptive_config(invalid)
        self.assertEqual(a['audit_fit']['training_readiness'],
                         {'cold_start_min_epochs': 250, 'warm_start_min_epochs': 5})
        from .stage import build_fit_config
        raw_fit = build_fit_config(a['audit_fit'], n_train=200000, n_validation=50000)
        self.assertEqual(raw_fit.validation_interval_steps, 12)  # 6 global batches per epoch
        self.assertEqual(raw_fit.validation_patience_evaluations, 5)
        epoch_steps = 200096 // raw_fit.batch_size
        self.assertEqual(epoch_steps * a['audit_fit']['training_readiness']['cold_start_min_epochs'], 1500)
        self.assertEqual(epoch_steps * a['audit_fit']['training_readiness']['warm_start_min_epochs'], 30)
        for invalid_value in (0, -1, True, '10', float('nan'), float('inf')):
            invalid = copy.deepcopy(new['dgpo'])
            invalid['adaptive_omnifold']['recalibration']['fit']['warm_start_min_epochs_per_fold'] = invalid_value
            with self.assertRaisesRegex(ValueError, 'warm_start_min_epochs_per_fold'):
                resolve_adaptive_config(invalid)
        self.assertEqual(cfg.staleness_every_n_steps, 5)
        architecture_keys = set(inspect.signature(EvenetAdapterModelBuilder).parameters)
        self.assertFalse(architecture_keys.intersection(a['audit_fit']))
        self.assertTrue(a['recalibration']['visible_pair_rest_frame'])
        self.assertTrue(a['recalibration']['topology_fourier_embedding'])
        self.assertEqual(a['recalibration']['decoder_layers'], 2)
        self.assertTrue(cfg.bootstrap_on_start and cfg.bootstrap_fail_closed)
        self.assertTrue(cfg.baseline_probe_on_start)
        self.assertIsNone(new['reward_config']['omnifold']['bundle_file'])
        self.assertEqual(new['dgpo']['checkpoint_load_mode'], 'weights_only')
        self.assertFalse(new['dgpo']['auto_resume_from_last'])
        self.assertEqual(new['dgpo']['lr_schedule']['type'], 'cosine')
        self.assertEqual(new['dgpo']['reference_trust'], original['dgpo']['reference_trust'])
        self.assertFalse(new['options']['Training']['EMA']['replace_model_after_load'])
        # An already-loaded DGPO checkpoint must not import epoch/optimizer,
        # old OmniFold stacks or raw-monitor caches into this fresh DGPO run.
        from RL.DGPO_neutrino.model_utils import select_dgpo_training_state, parse_dgpo_resume_from_checkpoint
        state = select_dgpo_training_state({'epoch': 31, 'global_step': 320,
            'dgpo_next_epoch': 32, 'dgpo_optimizer_state_dict': {'old': 1},
            'dgpo_adaptive_omnifold_state': {'old': 1},
            'dgpo_omnifold_reward_stack': {'old': 1}}, load_mode=new['dgpo']['checkpoint_load_mode'])
        self.assertIsNone(state)
        self.assertEqual(parse_dgpo_resume_from_checkpoint(state), (0, 0))
        invalid = copy.deepcopy(new['dgpo'])
        invalid['adaptive_omnifold']['recalibration']['visible_pair_rest_frame'] = 'false'
        with self.assertRaisesRegex(ValueError, 'must be a boolean'):
            resolve_adaptive_config(invalid)
        invalid = copy.deepcopy(new['dgpo'])
        invalid['adaptive_omnifold']['cache_event_inputs'] = 'false'
        with self.assertRaisesRegex(ValueError, 'cache_event_inputs must be a boolean'):
            resolve_adaptive_config(invalid)
        _, spec = pack_event_inputs(batch(), include_pairwise_context=True, include_visible_pair_rest_frame=True)
        builder = mock.Mock()
        factory = peft_bank_factory(builder, spec, 'audit', classifier_overrides={'visible_pair_rest_frame': False})
        factory()
        builder.make_classifier.assert_called_once_with(spec, 'audit', reset=True, visible_pair_rest_frame=False)
        # Every actual adaptive-pool path must forward the packing flag, including bootstrap/refit/rollback.
        tree = ast.parse((root/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == '_materialize_adaptive_omnifold_pool']
        self.assertEqual(len(calls), 16)
        for call in calls:
            self.assertIn(ast.unparse(call.args[0]), ('omnifold_train_shard', 'omnifold_val_shard'))
            flag = next(k.value for k in call.keywords if k.arg == 'include_visible_pair_rest_frame')
            self.assertEqual(ast.unparse(flag), 'adaptive_cfg.visible_pair_rest_frame_enabled')


if __name__ == '__main__':
    unittest.main()
