"""Masking, bandwidth and causal routing checks for the actual Fourier readout."""
import math

import pytest
import torch

from evenet.network.body.angular_conditioning import VisibleAngularFourier
from evenet.network.body.test_fourier_integration import SmallDiffusion


def readout(bank=(1, 2, 4, 8)):
    branch = VisibleAngularFourier(['eta', 'phi'], 8, 'eta', 'phi',
        placement='output', projection_type='cross_attention', mlp_dim=12,
        harmonics=bank, log_diagnostics=True).eval()
    with torch.no_grad():
        branch.projection.weight.normal_(std=.1)
    return branch


def inputs():
    torch.manual_seed(71)
    return (torch.randn(2, 5, 8), torch.randn(2, 3, 2),
            torch.tensor([[[1], [1], [0]], [[0], [0], [0]]], dtype=torch.bool),
            torch.tensor([[[1], [1], [0], [1], [0]], [[0], [0], [0], [1], [1]]], dtype=torch.bool),
            torch.full((2, 5, 1), .6))


def test_readout_updates_only_valid_invisible_and_handles_empty_memory():
    h, raw, visible, mask, t = inputs()
    raw[~visible.expand_as(raw)] = float('nan')
    branch = readout()
    out = branch.inject(h, raw, visible, target_mask=mask, time=t)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :3], h[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(out[0, 4], h[0, 4], rtol=0, atol=0)
    torch.testing.assert_close(out[1], h[1], rtol=0, atol=0)
    assert not torch.allclose(out[0, 3], h[0, 3])
    out.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in branch.parameters() if p.grad is not None)


def test_memory_permutation_padding_and_phi_periodicity():
    h, raw, visible, mask, t = inputs()
    b = readout()
    reference = b.inject(h, raw, visible, target_mask=mask, time=t)
    altered = raw.clone()
    altered[~visible.expand_as(raw)] = 1e6
    altered[..., 1] += 2 * math.pi
    permutation = [2, 0, 1]
    other = b.inject(h, altered[:, permutation], visible[:, permutation], target_mask=mask, time=t)
    torch.testing.assert_close(other, reference, atol=2e-6, rtol=1e-5)


def test_noisy_queries_and_time_reach_the_readout():
    h, raw, visible, mask, t = inputs()
    b = readout()
    h.requires_grad_()
    t.requires_grad_()
    residual = b.inject(h, raw, visible, target_mask=mask, time=t) - h
    residual.square().sum().backward()
    assert h.grad[0, 3].norm() > 0
    assert t.grad[0, 3].norm() > 0
    assert h.grad[:, :3].count_nonzero() == 0


def test_bank_pair_has_equal_parameter_shapes_and_correct_persistent_frequencies():
    low, wide = readout((1, 2, 3, 4)), readout()
    assert {k: p.shape for k, p in low.named_parameters()} == {
        k: p.shape for k, p in wide.named_parameters()}
    assert wide.state_dict()['harmonics'].tolist() == [1, 2, 4, 8]
    raw = torch.tensor([[[.4, .31]]])
    mask = torch.ones(1, 1, 1, dtype=torch.bool)
    assert not torch.allclose(low.features(raw, mask), wide.features(raw, mask))


def test_readout_bypasses_pet_blocks_and_reaches_generation_head():
    model = SmallDiffusion('readout').eval()
    x = torch.randn(3, 4, 4)
    captures = []
    hook = model.PET.transformer_blocks[-1].register_forward_hook(
        lambda m, args, output: captures.append(output.detach().clone()))
    h0, p0 = model(x)
    with torch.no_grad():
        model.PET.angular_conditioning.projection.weight.normal_(std=.1)
    h1, p1 = model(x)
    hook.remove()
    torch.testing.assert_close(captures[0], captures[1], rtol=0, atol=0)
    torch.testing.assert_close(h0[:, :3], h1[:, :3], rtol=0, atol=0)
    assert not torch.allclose(h0[:, 3:], h1[:, 3:])
    assert not torch.allclose(p0, p1)
    p1.square().mean().backward()
    assert model.PET.feature_embedding[0].weight.grad.norm() > 0
    assert model.PET.angular_conditioning.attention.in_proj_weight.grad.norm() > 0


@pytest.mark.parametrize('bank', [[], [0, 1], [1, 1], [1.5, 2], [float('nan')]])
def test_invalid_banks_fail_closed(bank):
    with pytest.raises(ValueError, match='Harmonics'):
        readout(bank)
