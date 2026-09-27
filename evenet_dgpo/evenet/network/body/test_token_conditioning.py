"""Diffusion-only token FiLM: zero-start, masks, state compatibility and gradients."""
import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from evenet.network.body.visible_conditioning import VisibleConditioning, visible_conditioning_spec
from evenet.network.body.token_conditioning import TokenSpecificConditioning
from evenet.network.body.test_visible_conditioning import head, head_inputs, spec, open_heads
from evenet.utilities.tool import safe_load_state
from evenet.utilities.fourier_integration import optimizer_parameters
from RL.DGPO_neutrino.model_utils import (
    make_reference_model, make_round_reference_model, state_dict_sha256,
    prepare_step_zero_architecture_bootstrap,
)


def token_head():
    model = head(True, True)
    model.visible_conditioning = VisibleConditioning(
        **spec(kinematics=True), token_readout=dict(enabled=True, width=12, heads=3, block="last"))
    return model


def test_zero_start_is_exact_even_with_trained_global_film_and_ddim_updates():
    torch.manual_seed(9)
    base, new = head(True, True), token_head()
    open_heads(base.visible_conditioning)  # Preserve a nonzero, already-trained FiLM.
    safe_load_state(new, base.state_dict(), verbose=False)
    args = head_inputs(True)
    for _ in range(6):
        old, actual = base(**args), new(**args)
        torch.testing.assert_close(old, actual, rtol=0, atol=0)
        # A fixed-noise iterative state update remains identical at initialization.
        args["x"] = args["x"] + .02 * torch.cat((actual, actual), -1)
    wrapper = nn.Module(); wrapper.TruthGeneration = new
    paths = ["TruthGeneration", "TruthGeneration.visible_conditioning"]
    groups = [optimizer_parameters(wrapper, [path], paths) for path in paths]
    ids = [id(p) for group in groups for p in group]
    assert len(ids) == len(set(ids)) == len(list(wrapper.parameters()))
    assert {id(p) for p in new.visible_conditioning.token_readout.parameters()} <= {id(p) for p in groups[1]}


def test_token_branch_does_not_advance_common_rng_or_change_classifier_spec():
    torch.manual_seed(19)
    base = VisibleConditioning(**spec())
    expected_rng = torch.get_rng_state()
    torch.manual_seed(19)
    new = VisibleConditioning(**spec(), token_readout=dict(enabled=True))
    assert torch.equal(expected_rng, torch.get_rng_state())
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    cfg = SimpleNamespace(VisibleConditioning=dict(classifier_enabled=True, diffusion_enabled=True,
                          diffusion_token_readout=dict(enabled=True, block="last")))
    kwargs = dict(feature_names=spec()["feature_names"], token_dim=4, hidden_dim=8, num_layers=2, n_branches=2)
    assert "token_readout" not in visible_conditioning_spec(cfg, target="classifier", **kwargs)
    assert "token_readout" in visible_conditioning_spec(cfg, target="diffusion", **kwargs)


def test_query_dependency_masks_empty_memory_and_memory_permutation():
    torch.manual_seed(17)
    branch = TokenSpecificConditioning(8, 12, width=12, heads=3)
    nn.init.normal_(branch.output.weight, std=.02)
    q = torch.randn(3, 2, 8, requires_grad=True)
    mem = torch.randn(3, 4, 12, requires_grad=True)
    qm = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    mm = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    q_dirty = q.masked_fill(~qm[..., None], float("nan"))
    m_dirty = mem.masked_fill(~mm[..., None], float("nan"))
    out = branch(q_dirty, m_dirty, qm, mm)
    perm = branch(q_dirty, m_dirty.flip(1), qm, mm.flip(1))
    for a, b in zip(out, perm):
        assert torch.isfinite(a).all() and a[1].count_nonzero() == 0 and a[2, 1].count_nonzero() == 0
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
        assert not torch.allclose(a[0, 0], a[0, 1])
    sum(a.square().sum() for a in out).backward()
    assert torch.isfinite(q.grad).all() and torch.isfinite(mem.grad).all()
    assert q.grad[~qm].count_nonzero() == 0 and mem.grad[~mm].count_nonzero() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())


def test_last_block_only_gradient_and_state_roundtrip():
    torch.manual_seed(33)
    model = token_head()
    args = head_inputs(True)
    before = model(**args).detach()
    branch = model.visible_conditioning.token_readout
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(4):
        opt.zero_grad()
        model(**args)[:, -2:].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        opt.step()
    assert branch.output.weight.grad.norm() > 0
    assert branch.attention.in_proj_weight.grad.norm() > 0
    assert model.visible_conditioning.encoder[0].weight.grad.norm() > 0
    assert not torch.allclose(before[:, -2:], model(**args)[:, -2:])
    seen = []
    hooks = [block.register_forward_pre_hook(lambda _, a, kw: seen.append(kw["modulation"][0].ndim), with_kwargs=True)
             for block in model.gen_transformer_blocks]
    model(**args)
    for hook in hooks: hook.remove()
    assert seen == [2, 3]
    restored = token_head(); restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model(**args), restored(**args), rtol=0, atol=0)


def test_step_zero_three_block_identity_initialization_is_not_a_dead_gradient_path():
    from evenet.network.body.test_conditioning_depth import make_head
    from evenet.network.body.token_conditioning import TokenSpecificConditioning
    torch.manual_seed(81)
    model = make_head(3, kinematics=True, identity=1)
    model.visible_conditioning.token_readout = TokenSpecificConditioning(8, 12, width=12, heads=3)
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    args = head_inputs(True)
    for _ in range(5):
        opt.zero_grad()
        model(**args)[:, -2:].square().mean().backward()
        opt.step()
    branch = model.visible_conditioning.token_readout
    assert branch.output.weight.grad.norm() > 0
    assert branch.attention.in_proj_weight.grad.norm() > 0


def test_frozen_references_keep_exact_hash_and_function_not_policy_only_parameters():
    def wrap(value):
        model = nn.Module(); model.TruthGeneration = value
        return model
    original = wrap(head(True, True)).eval()
    policy = wrap(token_head())
    checkpoint = dict(dgpo_checkpoint_version=1, dgpo_ref_state_dict=original.state_dict(),
                      dgpo_round_ref_state_dict=original.state_dict())
    with mock.patch("RL.DGPO_neutrino.model_utils.build_evenet_on_device", side_effect=lambda *a: wrap(token_head())):
        ref = make_reference_model(policy, None, {}, torch.device("cpu"), checkpoint)
        round_ref = make_round_reference_model(ref, None, {}, torch.device("cpu"), checkpoint)
    for saved in (ref, round_ref):
        assert state_dict_sha256(saved) == state_dict_sha256(original)
        assert saved.TruthGeneration.visible_conditioning.token_readout is None
        assert not any(p.requires_grad for p in saved.parameters())
        args = head_inputs(True)
        with torch.no_grad():
            torch.testing.assert_close(original.TruthGeneration(**args), saved.TruthGeneration(**args), rtol=0, atol=0)
    assert policy.TruthGeneration.visible_conditioning.token_readout is not None


def bootstrap():
    return dict(state_dict={}, dgpo_next_epoch=0, global_step=0, dgpo_epoch_step=0,
                dgpo_checkpoint_version=1, epoch=-1, dgpo_ref_state_dict={"old": "ref"},
                dgpo_round_ref_state_dict={"old": "round"}, dgpo_round_ref_sha256="original hash",
                dgpo_omnifold_reward_stack={"original": "classifiers"}, dgpo_omnifold_reward_metadata={},
                dgpo_adaptive_omnifold_state={"reward_round_id": 1},
                dgpo_optimizer_state_dict={"optimizer": {"state": {}}, "scheduler": {"last_epoch": 0}})


def test_bootstrap_only_rebuilds_empty_optimizer_never_resets_later_training():
    source = bootstrap()
    result = prepare_step_zero_architecture_bootstrap(source)
    assert result == {k: v for k, v in source.items() if k != "dgpo_optimizer_state_dict"}
    assert "dgpo_optimizer_state_dict" in source
    for key, value in (("global_step", 1), ("dgpo_next_epoch", 1), ("dgpo_epoch_step", 1)):
        with pytest.raises(ValueError, match="step zero"):
            prepare_step_zero_architecture_bootstrap(dict(source, **{key: value}))
    nonempty = copy.deepcopy(source)
    nonempty["dgpo_optimizer_state_dict"]["optimizer"]["state"][0] = {"step": 1}
    with pytest.raises(ValueError, match="empty Adam"):
        prepare_step_zero_architecture_bootstrap(nonempty)
