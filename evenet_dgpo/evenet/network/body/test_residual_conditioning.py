"""Direct visible readout: identity start, shared repeated access, masks and learning."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from evenet.network.body.visible_conditioning import VisibleConditioning, visible_conditioning_spec
from evenet.network.body.token_conditioning import TokenSpecificConditioning
from evenet.network.body.test_visible_conditioning import spec, head_inputs, open_heads
from evenet.network.body.test_conditioning_depth import make_head
from evenet.utilities.fourier_integration import load_conditioning_ablation_weights


def residual_head(block="last"):
    model = make_head(3, kinematics=True, identity=1)
    model.visible_conditioning = VisibleConditioning(
        **dict(spec(kinematics=True), num_layers=3),
        token_readout=dict(enabled=True, width=12, heads=3, mode="residual", block=block))
    return model


def velocity_head():
    model = make_head(3, kinematics=True, identity=1)
    model.visible_conditioning = VisibleConditioning(
        **dict(spec(kinematics=True), num_layers=3),
        token_readout=dict(enabled=True, width=12, heads=3, mode="velocity",
                           block="output", output_dim=4))
    return model


def wrap(head):
    model = nn.Module(); model.TruthGeneration = head
    return model


@pytest.mark.parametrize("block", ["last", "all"])
def test_exact_raw_initial_function_input_gradient_and_iterative_updates(block):
    torch.manual_seed(311)
    base = make_head(3, kinematics=True, identity=1)
    open_heads(base.visible_conditioning)
    new = residual_head(block)
    checkpoint = {"state_dict": {"model." + k: v for k, v in wrap(base).state_dict().items()}}
    # Deliberately disagreeing EMA must never be used by the raw loader.
    checkpoint["ema_state_dict"] = {k: torch.zeros_like(v) for k, v in checkpoint["state_dict"].items()}
    load_conditioning_ablation_weights(wrap(new), checkpoint)
    args = head_inputs(True)
    args["x"].requires_grad_()
    for _ in range(4):
        expected, actual = base(**args), new(**args)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        g0 = torch.autograd.grad(expected.square().sum(), args["x"])[0]
        g1 = torch.autograd.grad(actual.square().sum(), args["x"])[0]
        torch.testing.assert_close(g0, g1, rtol=0, atol=0)
        args["x"] = (args["x"] + .02 * torch.cat((actual, actual), -1)).detach().requires_grad_()


def test_last_and_all_same_parameters_shared_rng_and_classifier_unchanged():
    torch.manual_seed(89)
    last = residual_head("last")
    rng = torch.get_rng_state()
    torch.manual_seed(89)
    all_blocks = residual_head("all")
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in last.parameters()) == sum(p.numel() for p in all_blocks.parameters())
    for name, value in last.state_dict().items():
        torch.testing.assert_close(value, all_blocks.state_dict()[name], rtol=0, atol=0)
    torch.manual_seed(91)
    VisibleConditioning(**spec())
    rng = torch.get_rng_state()
    torch.manual_seed(91)
    VisibleConditioning(**spec(), token_readout=dict(enabled=True, mode="residual", block="all"))
    assert torch.equal(rng, torch.get_rng_state())
    cfg = SimpleNamespace(VisibleConditioning=dict(classifier_enabled=True, diffusion_enabled=True,
                          diffusion_token_readout=dict(enabled=True, mode="residual", block="all")))
    kwargs = dict(feature_names=spec()["feature_names"], token_dim=4, hidden_dim=8, num_layers=3, n_branches=2)
    assert "token_readout" not in visible_conditioning_spec(cfg, target="classifier", **kwargs)
    assert visible_conditioning_spec(cfg, target="diffusion", **kwargs)["token_readout"]["mode"] == "residual"


def test_residual_masks_empty_memory_permutation_query_and_visible_dependence():
    torch.manual_seed(12)
    branch = TokenSpecificConditioning(8, 12, width=12, heads=3, mode="residual", block="all")
    nn.init.normal_(branch.output.weight, std=.1)
    q = torch.randn(3, 2, 8, requires_grad=True)
    memory = torch.randn(3, 4, 12, requires_grad=True)
    qm = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    mm = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    qdirty = q.masked_fill(~qm[..., None], float("nan"))
    mdirty = memory.masked_fill(~mm[..., None], float("nan"))
    out = branch(qdirty, mdirty, qm, mm)
    assert out.shape == q.shape and torch.isfinite(out).all()
    assert out[1].count_nonzero() == 0 and out[2, 1].count_nonzero() == 0
    torch.testing.assert_close(out, branch(qdirty, mdirty.flip(1), qm, mm.flip(1)), atol=1e-7, rtol=1e-5)
    assert not torch.allclose(out[0, 0], out[0, 1])
    assert not torch.allclose(out, branch(qdirty, mdirty + .3, qm, mm))
    assert all(torch.isfinite(value) for value in branch.diagnostics.values())
    out.square().sum().backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(memory.grad).all()
    assert q.grad[~qm].count_nonzero() == 0 and memory.grad[~mm].count_nonzero() == 0


@pytest.mark.parametrize("block,calls", [("last", 1), ("all", 3)])
def test_effective_path_learning_injection_sites_diagnostics_and_resume(block, calls):
    torch.manual_seed(71)
    model = residual_head(block)
    args = head_inputs(True)
    branch = model.visible_conditioning.token_readout
    target = torch.randn(3, 2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    for step in range(4):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(**args)[:, -2:], target).backward()
        assert branch.output.weight.grad.norm() > 0  # No dead path behind identity-initialized blocks.
        if step == 0:
            assert branch.attention.in_proj_weight.grad.count_nonzero() == 0
        optimizer.step()
    assert branch.attention.in_proj_weight.grad.norm() > 0
    assert model.visible_conditioning.encoder[0].weight.grad.norm() > 0
    seen = []
    hook = branch.register_forward_hook(lambda *a: seen.append(1))
    actual = model(**args)
    hook.remove()
    assert len(seen) == calls
    sites = [2] if block == "last" else [0, 1, 2]
    logs = model.visible_conditioning.diagnostics
    assert {k for k in logs if k.endswith("token_residual_rms")} == {
        f"block_{i}/token_residual_rms" for i in sites}
    assert all(logs[f"block_{i}/token_residual_rms"] > 0 for i in sites)
    restored = residual_head(block)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(actual, restored(**args), rtol=0, atol=0)
    # Snapshot/restore AdamW as well as parameters; the next update must replay.
    other_optimizer = torch.optim.AdamW(restored.parameters(), lr=.003)
    other_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for net, opt in ((model, optimizer), (restored, other_optimizer)):
        opt.zero_grad()
        nn.functional.mse_loss(net(**args)[:, -2:], target).backward()
        opt.step()
    torch.testing.assert_close(model(**args), restored(**args), rtol=0, atol=0)
    # Turning off only the extra route changes generated states, not visible slots.
    restored.visible_conditioning.token_readout = None
    with torch.no_grad():
        active, disabled = model(**args), restored(**args)
    torch.testing.assert_close(active[:, :4], disabled[:, :4], atol=0, rtol=0)
    assert not torch.allclose(active[:, -2:], disabled[:, -2:])


@pytest.mark.parametrize("options", [dict(mode="invalid"), dict(block="first"), dict(mode="film", block="all")])
def test_invalid_route_fails_instead_of_silently_reusing_legacy(options):
    with pytest.raises(ValueError):
        TokenSpecificConditioning(8, 12, width=12, heads=3, **options)


def test_common_source_rejects_already_trained_token_readout():
    model = wrap(residual_head())
    with pytest.raises(ValueError, match="precede token-readout"):
        load_conditioning_ablation_weights(model, {"state_dict": model.state_dict()})


def test_velocity_probe_exact_start_then_learns_only_invisible_output():
    torch.manual_seed(419)
    base = make_head(3, kinematics=True, identity=1)
    open_heads(base.visible_conditioning)
    probe = velocity_head()
    checkpoint = {"state_dict": {"model." + k: v for k, v in wrap(base).state_dict().items()}}
    load_conditioning_ablation_weights(wrap(probe), checkpoint)
    args = head_inputs(True)
    with torch.no_grad():
        expected, actual = base(**args), probe(**args)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    branch = probe.visible_conditioning.token_readout
    probe.requires_grad_(False)
    branch.requires_grad_(True)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=.01)
    target = torch.randn_like(actual[:, -2:])
    for step in range(4):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(probe(**args)[:, -2:], target)
        loss.backward()
        assert branch.output.weight.grad.norm() > 0
        if step == 0:
            assert branch.attention.in_proj_weight.grad.count_nonzero() == 0
        optimizer.step()
    assert branch.attention.in_proj_weight.grad.norm() > 0
    with torch.no_grad():
        changed = probe(**args)
    torch.testing.assert_close(changed[:, :4], expected[:, :4], rtol=0, atol=0)
    assert not torch.allclose(changed[:, -2:], expected[:, -2:])
    assert set(probe.visible_conditioning.diagnostics) >= {
        "velocity_residual_rms", "velocity_residual_to_base_rms"}
    assert all(torch.isfinite(value) for value in probe.visible_conditioning.diagnostics.values())


def test_velocity_spec_derives_output_dimension_and_rejects_ambiguous_routes():
    cfg = SimpleNamespace(VisibleConditioning=dict(
        diffusion_enabled=True,
        diffusion_token_readout=dict(enabled=True, mode="velocity", block="output")))
    kwargs = dict(target="diffusion", feature_names=spec()["feature_names"], token_dim=4,
                  hidden_dim=8, num_layers=3, n_branches=2)
    with pytest.raises(ValueError, match="output dimension"):
        visible_conditioning_spec(cfg, **kwargs)
    derived = visible_conditioning_spec(cfg, output_dim=4, **kwargs)
    assert derived["token_readout"]["output_dim"] == 4
    for options in (dict(mode="velocity", block="last", output_dim=4),
                    dict(mode="velocity", block="output"),
                    dict(mode="residual", block="output")):
        with pytest.raises(ValueError):
            TokenSpecificConditioning(8, 12, width=12, heads=3, **options)
