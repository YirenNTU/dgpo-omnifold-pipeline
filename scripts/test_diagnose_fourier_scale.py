import copy

import numpy as np
import pytest
import torch

from diagnose_fourier_scale import branch_scale, load_trained_weights, paired_batch, summarize
from evenet.network.body.test_fourier_integration import SmallDiffusion


class DiagnosticModel(SmallDiffusion):
    def shared_step(self, batch, size, *args, **kwargs):
        noise = torch.randn_like(batch['x'])
        time = torch.rand(size)
        pred = self(batch['x'] + noise)[1]
        return {'generations': {'neutrino': dict(vector=pred, truth=noise[:, :1],
                    time=time, mask=torch.ones(size, 1, 1, dtype=torch.bool))}}


def trained_model():
    torch.manual_seed(7)
    model = DiagnosticModel('input').eval()
    with torch.no_grad():
        model.PET.angular_conditioning.projection.weight.normal_(std=.2)
    return model


def test_real_pet_scale_zero_matches_branch_disabled_and_restores_after_exception():
    model = trained_model()
    x = torch.randn(3, 4, 4)
    branch = model.PET.angular_conditioning
    original = model(x)[1].detach()
    with branch_scale(branch, 0):
        zero = model(x)[1].detach()
    model.PET.angular_conditioning = None
    disabled = model(x)[1].detach()
    model.PET.angular_conditioning = branch
    torch.testing.assert_close(zero, disabled, atol=0, rtol=0)
    assert not torch.allclose(original, zero)
    with pytest.raises(RuntimeError), branch_scale(branch, 2):
        raise RuntimeError('probe')
    torch.testing.assert_close(model(x)[1], original, atol=0, rtol=0)


def test_paired_evaluation_repeatable_and_does_not_change_weights():
    model = trained_model()
    saved = copy.deepcopy(model.state_dict())
    batch = {'x': torch.randn(3, 4, 4)}
    args = (model, model.PET.angular_conditioning, batch, [0., .5, 1., 2.], 42, 0, 'cpu')
    a, b = paired_batch(*args), paired_batch(*args)
    for scale in a:
        np.testing.assert_array_equal(a[scale], b[scale])
    assert np.all(a[1.][:, 1] == 0)
    assert a[0.][:, 1].sum() > 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
    assert not model.PET.angular_conditioning._forward_hooks


def test_loader_rejects_missing_branch_and_loads_trained_weights():
    model = trained_model()
    state = copy.deepcopy(model.state_dict())
    load_trained_weights(model, {'state_dict': {'model.' + k: v for k, v in state.items()}})
    state.pop('PET.angular_conditioning.projection.weight')
    with pytest.raises(ValueError, match='missing'):
        load_trained_weights(model, {'state_dict': state})
    with pytest.raises(ValueError, match='nonzero'):
        zero = DiagnosticModel('input')
        load_trained_weights(zero, {'state_dict': zero.state_dict()})


def test_persistent_frequency_bank_does_not_count_as_a_trained_readout():
    model = SmallDiffusion('readout')
    assert torch.count_nonzero(model.state_dict()['PET.angular_conditioning.harmonics']) > 0
    with pytest.raises(ValueError, match='nonzero'):
        load_trained_weights(model, {'state_dict': model.state_dict()})


def test_padded_channels_are_excluded_like_production_loss():
    model = trained_model()
    model.invisible_padding = 1
    result = paired_batch(model, model.PET.angular_conditioning,
        {'x': torch.randn(2, 4, 4)}, [0., .5, 1., 2.], 42, 0, 'cpu')
    for stats in result.values():
        np.testing.assert_array_equal(stats[:, 2], [3, 3])


def test_pooled_metrics_use_target_counts_not_mean_of_batch_means():
    reference = np.array([[[2., 0., 2.], [18., 0., 6.]]])
    changed = np.array([[[4., 2., 2.], [12., 6., 6.]]])
    result = summarize({1.: reference, 0.: changed})
    assert result['1.0']['velocity_mse'] == 2.5
    assert result['0.0']['velocity_mse'] == 2.
    assert result['0.0']['paired_loss_delta_vs_scale1'] == -.5
    assert result['0.0']['velocity_delta_rms_vs_scale1'] == 1.
