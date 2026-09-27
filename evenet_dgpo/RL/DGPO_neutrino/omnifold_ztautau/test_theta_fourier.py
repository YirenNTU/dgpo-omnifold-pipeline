"""Theta Fourier feature, checkpoint and next-run configuration contracts."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch_with_pair_context
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (EvenetAdapterRatioClassifier, EvenetAdapterModelBuilder,
    pack_event_inputs, periodic_tau_pair_features, periodic_tau_pair_feature_dim,
    peft_bank_factory)


def test_theta_fourier_preserves_old_channels_and_has_candidate_gradients():
    packed, spec = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    candidate = torch.randn(3, 5, 4, requires_grad=True)
    old = periodic_tau_pair_features(packed, candidate, spec, max_harmonic=4)
    new = periodic_tau_pair_features(packed, candidate, spec, max_harmonic=4, theta_fourier=True)
    assert new.shape == (3, 5, 25)
    assert periodic_tau_pair_feature_dim(4, theta_fourier=True) == 25
    torch.testing.assert_close(new[..., :9], old, atol=0, rtol=0)
    linear = periodic_tau_pair_features(packed, candidate, spec, max_harmonic=4, include_theta_pair=True)
    delta = linear[..., -2] * torch.pi
    total = (linear[..., -1] + 1) * torch.pi
    for k in range(1, 5):
        expected = torch.stack(((k*delta).sin(), (k*delta).cos(), (k*total).sin(), (k*total).cos()), -1)
        torch.testing.assert_close(new[..., 9+4*(k-1):9+4*k], expected, atol=5e-6, rtol=1e-5)
    new[..., 9:].sum().backward()
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad[..., [0, 2]].abs().sum() > 0
    assert candidate.grad[..., [1, 3]].abs().sum() == 0


def test_theta_fourier_checkpoint_roundtrip():
    packed, spec = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    template = _FakeZtautauBackbone()
    def build(spec):
        return EvenetAdapterRatioClassifier(copy.deepcopy(template), spec,
            periodic_pair_features=True, topology_fourier_embedding=True,
            topology_theta_fourier=True, topology_max_harmonic=4,
            topology_hidden_dim=16, topology_embedding_dim=8, topology_fusion_hidden_dim=12,
            topology_dropout=0., head_dropout=0., decoder_hidden_dim=8,
            decoder_layers=1, decoder_heads=2, adapter_bottleneck=4)
    model = build(spec).eval()
    with torch.no_grad():
        model.bank.output.weight.normal_(std=.1)
    payload = model.peft_payload()
    assert payload['classifier_config']['topology_theta_fourier'] is True
    restored = EvenetAdapterRatioClassifier.from_peft_payload(payload, model_builder=build, device=torch.device('cpu')).eval()
    candidate = torch.randn(3, 4)
    torch.testing.assert_close(model(packed, candidate), restored(packed, candidate))
    assert restored.bank.pairwise_feature_dim == 25


def test_audit_factory_theta_override(tmp_path):
    _, spec = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    checkpoint = tmp_path / 'raw1110.ckpt'
    checkpoint.touch()
    with mock.patch('RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio._config_with_pet_adapters', return_value=SimpleNamespace()), \
         mock.patch('RL.DGPO_neutrino.model_utils.build_evenet_on_device', return_value=_FakeZtautauBackbone()), \
         mock.patch('RL.DGPO_neutrino.model_utils.load_weights_like_configure_model', return_value={}):
        builder = EvenetAdapterModelBuilder(config=SimpleNamespace(), normalization_dict={},
            checkpoint_path=checkpoint, device=torch.device('cpu'),
            periodic_pair_features=True, topology_fourier_embedding=True,
            topology_max_harmonic=4, decoder_hidden_dim=8, decoder_layers=1,
            decoder_heads=2, adapter_bottleneck=4)
        model = peft_bank_factory(builder, spec, 'theta-audit', classifier_overrides={'topology_theta_fourier': True})()
    assert model._topology_theta_fourier
    assert model.bank.pairwise_feature_dim == 25


def test_resolved_experiment_keeps_source_and_latest_fit():
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'scripts'))
    from train_neutrino_backend import read_overlay_yaml, deep_update, read_yaml
    from .adaptive import resolve_adaptive_config
    c = deep_update(read_yaml(root / 'config/train_diffusion_nersc.yaml'),
                    read_overlay_yaml(root / 'config/dgpo_h4_angular_conditioning_1110.yaml'))
    training = c['options']['Training']
    assert training['model_checkpoint_load_path'].endswith('dgpo-epoch=110-next_ep=111-step=1110.ckpt')
    assert c['dgpo']['checkpoint_load_mode'] == 'weights_only'
    assert not c['dgpo']['auto_resume_from_last']
    assert c['dgpo']['unbounded_training']
    assert not c['dgpo']['endpoint_kl']['enabled']
    assert c['dgpo']['reference_trust']['enabled']
    assert c['dgpo']['reference_trust']['objective'] == 'velocity_mse'
    assert c['dgpo']['reference_trust']['coefficient'] == 1.0
    assert not c['dgpo']['reference_trust']['adaptive_boundary']['enabled']
    assert c['platform']['number_of_workers'] == 16
    assert not training['EMA']['replace_model_after_load']
    assert c['network']['Body']['PET']['visible_angular_fourier']['enabled']
    cfg = resolve_adaptive_config(c['dgpo'])
    assert cfg.topology_theta_fourier
    for block in (c['dgpo']['adaptive_omnifold']['audit_fit'],
                  c['dgpo']['adaptive_omnifold']['recalibration']['fit']):
        assert block['fourier_output_standardization']
        assert block['min_steps'] == 300
        assert block['validation_min_delta'] == .001
        assert block['lr_scheduler'] == 'constant'
    assert len(c['logger']['wandb']['run_name']) < 96


def test_epoch_zero_resume_retains_classifier_without_refitting():
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'scripts'))
    from train_neutrino_backend import read_overlay_yaml
    c = read_overlay_yaml(root / 'config/dgpo_h4_angular_conditioning_resume_epoch0.yaml')
    assert c['options']['Training']['model_checkpoint_load_path'].endswith('dgpo-epoch=-1-next_ep=0-step=0.ckpt')
    assert c['dgpo']['checkpoint_load_mode'] == 'resume'
    assert not c['dgpo']['auto_resume_from_last']
    recal = c['dgpo']['adaptive_omnifold']['recalibration']
    assert not recal['bootstrap_on_start']
    assert not recal['refit_once_on_resume']
    assert c['dgpo']['adaptive_omnifold']['staleness_every_n_epochs'] == 5
    assert c['dgpo']['adaptive_omnifold']['staleness_every_n_steps'] is None
    assert c['dgpo']['validation_every_n_epochs'] == 10
    assert recal['topology_theta_fourier']
    assert not c['dgpo']['endpoint_kl']['enabled']
    assert c['dgpo']['reference_trust']['objective'] == 'velocity_mse'
    assert c['dgpo']['reference_trust']['enabled']
    assert c['dgpo']['reference_trust']['coefficient'] == 1.0
    assert c['logger']['wandb']['id'] == 'h4angc1'
    assert not c['logger']['wandb']['fresh_run']


def test_trained_diffusion_visible_fourier_config():
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'scripts'))
    from train_neutrino_backend import read_overlay_yaml
    from .adaptive import resolve_adaptive_config
    c = read_overlay_yaml(root / 'config/dgpo_h4_visible_fourier_diffang.yaml')
    t = c['options']['Training']
    assert t['model_checkpoint_load_path'].endswith('diffusion_angular_10pct_lr5e4/checkpoints/last.ckpt')
    assert t['pretrain_model_load_path'] is None
    assert not t['EMA']['enable']
    assert not t['EMA']['use_for_generation']
    assert c['network']['Body']['PET']['visible_angular_fourier']['enabled']
    assert c['dgpo']['checkpoint_load_mode'] == 'weights_only'
    assert not c['dgpo']['auto_resume_from_last']
    assert c['dgpo']['adaptive_omnifold']['recalibration']['bootstrap_on_start']
    a = resolve_adaptive_config(c['dgpo'])
    assert a.topology_theta_fourier and a.topology_fourier_embedding_enabled
    assert (a.min_iterations, a.max_iterations) == (2, 2)
    assert a.fixed_iteration_budget
    assert not a.ess_aware_checkpoint_selection
    assert not a.iteration_one_only and not a.log_only
    assert a.monitor_mode == 'raw_plateau_refit'
    assert a.max_reward_age_epochs == 20
    assert a.max_reward_rounds is None
    assert a.scheduled_refit_fail_closed
    assert a.fixed_schedule_log_raw_audit
    assert a.fixed_schedule_skip_staleness_audit
    from ..gradient_conflict import resolve_gradient_conflict_config
    assert not resolve_gradient_conflict_config(c['dgpo']['gradient_conflict']).enabled
    for fit in [c['dgpo']['adaptive_omnifold']['audit_fit'],
                c['dgpo']['adaptive_omnifold']['recalibration']['fit']]:
        assert not fit['diagnostic_enabled']
        assert not fit['representation_diagnostic_enabled']
        assert not fit['representation_path_enabled']
        assert not fit['log_parameter_updates']
        assert fit['representation_export_dir'] is None
        assert fit['validation_interval_epochs'] == 1
        assert fit['validation_patience_epochs'] == 25
        assert fit['min_steps'] == 0
        assert fit['steps'] is None
        assert not fit['enforce_min_epochs']
        assert fit['fourier_output_standardization']
        assert fit['restore_best']
        assert fit['checkpoint_selection_metric'] == 'loss'
    from .adaptive import AdaptiveOmniFoldState, reward_refit_due_to_age
    assert a.fit['min_steps_per_fold'] == 0
    assert a.fit['max_steps_per_fold'] is None
    assert a.fit['warm_start_min_epochs_per_fold'] is None
    state = AdaptiveOmniFoldState()
    state.installed_at_epoch = -1
    assert not reward_refit_due_to_age(state, epoch=18, max_reward_age_epochs=20)[0]
    assert reward_refit_due_to_age(state, epoch=19, max_reward_age_epochs=20)[0]
    state.last_recalibration_epoch = 19
    assert not reward_refit_due_to_age(state, epoch=38, max_reward_age_epochs=20)[0]
    assert reward_refit_due_to_age(state, epoch=39, max_reward_age_epochs=20)[0]
    for b in [c['dgpo']['adaptive_omnifold']['recalibration'], c['dgpo']['adaptive_omnifold']['audit_fit']]:
        assert b['periodic_pair_features'] and b['topology_fourier_embedding']
        assert b['topology_theta_fourier'] and b['topology_max_harmonic'] == 4
        assert b['train_last_pet_block'] and not b['train_backbone']
    assert c['dgpo']['reference_trust']['objective'] == 'velocity_mse'
    assert c['dgpo']['reference_trust']['coefficient'] == 1
    assert not c['dgpo']['endpoint_kl']['enabled']
    assert c['dgpo']['adaptive_omnifold']['staleness_every_n_epochs'] == 5
    assert c['platform']['number_of_workers'] == 16
    assert c['logger']['wandb']['id'] == 'h4visang1'
    assert not c['logger']['wandb']['classifier_loss_curves']
    assert c['logger']['wandb']['classifier_loss_curves_raw']


def test_visible_fourier_resume_requires_this_runs_last_checkpoint(tmp_path):
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'scripts'))
    from train_neutrino_backend import read_overlay_yaml
    from ..model_utils import resolve_dgpo_auto_resume_checkpoint
    c = read_overlay_yaml(root / 'config/dgpo_h4_visible_fourier_diffang_resume.yaml')
    d = c['dgpo']
    t = c['options']['Training']
    assert d['checkpoint_load_mode'] == 'resume'
    assert d['auto_resume_from_last']
    assert t['model_checkpoint_load_path'] == t['model_checkpoint_save_path'] + '/last.ckpt'
    assert d['auto_resume_fallback_checkpoint_path'] == t['model_checkpoint_load_path']
    assert not d['gradient_conflict']['enabled']
    assert not d['adaptive_omnifold']['recalibration']['refit_once_on_resume']
    import pytest
    with pytest.raises(FileNotFoundError):
        resolve_dgpo_auto_resume_checkpoint(tmp_path, enabled=True,
            fallback_checkpoint_path=tmp_path / 'last.ckpt')


def test_classifier_receives_all_raw_visible_tokens_with_h4():
    from evenet.network.body.angular_conditioning import VisibleAngularFourier
    from .test_ztautau_omnifold import _FakePET

    class AngularPET(_FakePET):
        def __init__(self):
            super().__init__()
            self.angular_conditioning = VisibleAngularFourier(['energy', 'pt', 'eta', 'phi'], 8, 'eta', 'phi')
            with torch.no_grad():
                self.angular_conditioning.projection.weight.normal_(std=.1)

        def forward(self, *, visible_raw, **kwargs):
            self.seen = visible_raw.detach().clone()
            encoded = super().forward(**kwargs)
            n = visible_raw.shape[1]
            residual = self.angular_conditioning(visible_raw, kwargs['mask'][:, :n])
            return encoded + torch.nn.functional.pad(residual, (0, 0, 0, encoded.shape[1]-n))

    batch = _event_batch_with_pair_context()
    batch['x_mask'][:, -1] = False
    batch['x'][:, -1] = float('nan')
    packed, spec = pack_event_inputs(batch, include_pairwise_context=True)
    backbone = _FakeZtautauBackbone()
    backbone.PET = AngularPET()
    model = EvenetAdapterRatioClassifier(backbone, spec,
        periodic_pair_features=True, topology_fourier_embedding=True,
        topology_theta_fourier=True, topology_max_harmonic=4,
        head_dropout=0., decoder_hidden_dim=8, decoder_layers=1,
        decoder_heads=2, adapter_bottleneck=4).eval()
    candidate = torch.randn(3, 4)
    logits = model(packed, candidate)
    assert torch.isfinite(logits).all()
    expected = torch.where(batch['x_mask'], batch['x'], torch.zeros_like(batch['x']))
    torch.testing.assert_close(backbone.PET.seen, expected)
    assert backbone.PET.seen.shape[1] == 5  # No four-particle truncation.
    assert model.bank.pairwise_feature_dim == 25
    assert not backbone.PET.angular_conditioning.projection.weight.requires_grad
