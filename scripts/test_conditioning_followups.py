"""Synthetic CPU checks of the two real-data experiment implementations."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))
from evenet.network.body.pair_attention import VisiblePairAttentionBias
from evenet.network.body.visible_conditioning import visible_conditioning_spec
from evenet.network.body.relation_conditioning import load_relation_weights
from evenet.network.heads.generation.generation_head import EventGenerationHead
from evenet.utilities.fourier_integration import optimizer_parameters
from train_neutrino_backend import deep_update, read_yaml, read_overlay_yaml


def make_head(*, context=True, pair=False, scale=True):
    settings = dict(feature_names=["Part_eta", "Part_phi", "Part_energy", "Part_pt"],
                    token_dim=8, hidden_dim=8, num_layers=3, n_branches=2,
                    feature_mode="kinematics", numerical_features=["Part_energy", "Part_pt"],
                    width=12, film_scale_enabled=scale)
    if context:
        settings["relation_adapter"] = dict(enabled=True, mode="context", width=12)
    if pair:
        settings["pair_attention"] = dict(enabled=True, heads=2, width=8)
    return EventGenerationHead(input_dim=8, projection_dim=8, num_global_cond=8,
        num_classes=2, output_dim=2, num_layers=3, num_heads=2, dropout=0.,
        layer_scale=True, layer_scale_init=.3, drop_probability=0., feature_drop=0.,
        visible_conditioning=settings)


def wrap(head):
    result = nn.Module()
    result.TruthGeneration = head
    return result


def inputs():
    raw = torch.randn(3, 4, 4)
    raw[..., 2:] = raw[..., 2:].abs()
    visible_mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)[..., None]
    full_mask = torch.cat((visible_mask, torch.ones(3, 2, 1, dtype=torch.bool)), dim=1)
    attn = torch.zeros(6, 6, dtype=torch.bool)
    attn[:4, 4:] = True
    # Existing visible-first mask: padded query rows may read the valid invisible keys.
    attn = attn[None].expand(3, -1, -1) & full_mask
    return dict(x=torch.randn(3, 6, 8), global_cond=torch.randn(3, 1, 8),
        global_cond_mask=torch.ones(3, 1, 1), num_x=None, x_mask=full_mask,
        time=torch.rand(3), label=torch.zeros(3, 1, dtype=torch.long), attn_mask=attn,
        time_masking=torch.cat((torch.zeros_like(visible_mask), torch.ones(3, 2, 1, dtype=torch.bool)), dim=1),
        visible_raw=raw, visible_tokens=torch.randn(3, 4, 8), visible_mask=visible_mask,
        visible_normalized=(raw - 2.) / 3.)


def trained_modulations(head):
    branch = head.visible_conditioning
    with torch.no_grad():
        heads = list(branch.modulations)
        if branch.relation_adapter is not None:
            heads += list(branch.relation_adapter.outputs)
        for projection in heads:
            projection.weight.normal_(std=.04)
            projection.bias.normal_(std=.02)


def test_shift_only_matches_explicit_gamma_removal_preserves_beta_layerscale_and_checkpoint():
    torch.manual_seed(17)
    baseline = make_head()
    trained_modulations(baseline)
    shift = make_head(scale=False)
    shift.load_state_dict(baseline.state_dict(), strict=True)
    manual = copy.deepcopy(baseline)
    with torch.no_grad():
        branch = manual.visible_conditioning
        for head in [*branch.modulations, *branch.relation_adapter.outputs]:
            head.weight.view(2, 2, 8, -1)[:, 0].zero_()
            head.bias.view(2, 2, 8)[:, 0].zero_()
    args = inputs()
    actual, expected = shift(**args), manual(**args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual[:, -2:], baseline(**args)[:, -2:])
    assert shift.visible_conditioning.diagnostics["effective_scale_rms"] == 0
    assert shift.visible_conditioning.diagnostics["scale_rms"] > 0
    actual[:, -2:].square().mean().backward()
    for head in [*shift.visible_conditioning.modulations,
                 *shift.visible_conditioning.relation_adapter.outputs]:
        grad = head.weight.grad.view(2, 2, 8, -1)
        assert grad[:, 0].count_nonzero() == 0
        assert grad[:, 1].abs().sum() > 0
    for old, new in zip(baseline.gen_transformer_blocks, shift.gen_transformer_blocks):
        assert torch.equal(old.layer_scale1.gamma, new.layer_scale1.gamma)
        assert torch.equal(old.layer_scale2.gamma, new.layer_scale2.gamma)
    restored = make_head(scale=False)
    restored.load_state_dict(shift.state_dict(), strict=True)
    torch.testing.assert_close(actual, restored(**args), rtol=0, atol=0)


def test_pair_zero_start_strict_loading_rng_shared_weights_and_input_gradient():
    torch.manual_seed(42)
    source = make_head(context=False)
    trained_modulations(source)
    torch.manual_seed(99)
    control = make_head()
    rng = torch.get_rng_state()
    torch.manual_seed(99)
    paired = make_head(pair=True)
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in control.state_dict().items():
        assert torch.equal(v, paired.state_dict()[k])
    state = {"model." + k: v.clone() for k, v in wrap(source).state_dict().items()}
    state["model.famo.w.neutrino"] = torch.tensor(.2)
    for model in (control, paired):
        load_relation_weights(wrap(model), dict(state_dict=state))
    args = inputs()
    args["x"].requires_grad_()
    expected, actual = control(**args), paired(**args)
    torch.testing.assert_close(expected, source(**args), rtol=0, atol=0)
    # A differentiable float mask can select another SDPA kernel. The new
    # logits are exactly zero, but FP32 evaluation need not be bitwise identical.
    torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
    a = torch.autograd.grad(expected.square().sum(), args["x"])[0]
    b = torch.autograd.grad(actual.square().sum(), args["x"])[0]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    bad = dict(state)
    bad.pop(next(k for k in bad if "generator.weight" in k))
    with pytest.raises(ValueError, match="missing_shared"):
        load_relation_weights(wrap(make_head(pair=True)), dict(state_dict=bad))
    with pytest.raises(ValueError, match="before relation-adapter"):
        load_relation_weights(wrap(make_head(pair=True)), dict(state_dict=wrap(paired).state_dict()))


@pytest.mark.parametrize("n_visible", [1, 4])
def test_pair_geometry_mask_symmetry_permutation_and_periodicity(n_visible):
    torch.manual_seed(8)
    branch = VisiblePairAttentionBias(heads=2, layers=3, width=8)
    for head in branch.outputs:
        nn.init.normal_(head.weight, std=.3)
    valid = torch.ones(3, n_visible, dtype=torch.bool)
    valid[1] = False
    valid[0, 2:] = False
    values = [torch.randn(3, n_visible) for _ in range(4)]
    values[2:] = [v.abs() for v in values[2:]]
    values = [v.masked_fill(~valid, float("nan")).requires_grad_() for v in values]
    biases = branch(*values, valid)
    reordered = branch(*(v.flip(1) for v in values), valid.flip(1))
    rotated_values = list(values)
    rotated_values[1] = rotated_values[1] + 2 * torch.pi
    rotated = branch(*rotated_values, valid)
    for a, b, c in zip(biases, reordered, rotated):
        assert torch.isfinite(a).all() and a[1].count_nonzero() == 0
        assert a.diagonal(dim1=-2, dim2=-1).count_nonzero() == 0
        torch.testing.assert_close(a, a.transpose(-2, -1), rtol=0, atol=0)
        torch.testing.assert_close(a, b.flip((-2, -1)), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)
    sum(b.square().sum() for b in biases).backward()
    for v in values:
        assert torch.isfinite(v.grad).all()
        assert v.grad[~valid].count_nonzero() == 0
    bad = [v.detach().clone() for v in values]
    bad[0][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite valid"):
        branch(*bad, valid)


def test_pair_route_learns_from_invisible_loss_keeps_masks_and_roundtrips_optimizer():
    torch.manual_seed(4)
    model = make_head(pair=True)
    args = inputs()
    target = torch.randn(3, 2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    branch = model.visible_conditioning.pair_attention
    for step in range(4):
        optimizer.zero_grad()
        nn.functional.mse_loss(model(**args)[:, -2:], target).backward()
        for head in branch.outputs:
            assert head.weight.grad.abs().sum() > 0
        if step > 0:
            assert branch.encoder[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    active = model(**args)
    seen = []
    def capture(module, positional, kwargs):
        seen.append(kwargs)
    hooks = [b.attn.register_forward_pre_hook(capture, with_kwargs=True) for b in model.gen_transformer_blocks]
    model(**args)
    for h in hooks:
        h.remove()
    for kw in seen[:2]:
        mask = kw["attn_mask"].view(3, 2, 6, 6)
        assert mask[0, :, :2, 4:].isneginf().all()
        assert mask[:, :, 4:, :4].count_nonzero() == 0
        assert kw["key_padding_mask"][0, 2:4].isneginf().all()
    assert seen[2]["attn_mask"].dtype == torch.bool
    restored = make_head(pair=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(active, restored(**args), rtol=0, atol=0)
    other = torch.optim.AdamW(restored.parameters(), lr=.01)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for net, opt in ((model, optimizer), (restored, other)):
        opt.zero_grad()
        nn.functional.mse_loss(net(**args)[:, -2:], target).backward()
        opt.step()
    torch.testing.assert_close(model(**args), restored(**args), rtol=0, atol=0)
    restored.visible_conditioning.pair_attention = None
    disabled = restored(**args)
    assert not torch.allclose(disabled[:, -2:], model(**args)[:, -2:])
    # This route changes visible states, unlike the older invisible-only residual.
    assert not torch.allclose(disabled[0, :2], model(**args)[0, :2])


def test_new_parameters_belong_to_visible_optimizer_once():
    model = wrap(make_head(pair=True))
    paths = ["TruthGeneration", "TruthGeneration.visible_conditioning"]
    groups = [optimizer_parameters(model, [p], paths) for p in paths]
    ids = [id(p) for group in groups for p in group]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    pair_ids = {id(p) for p in model.TruthGeneration.visible_conditioning.pair_attention.parameters()}
    assert pair_ids <= {id(p) for p in groups[1]}


def test_factory_is_diffusion_only_and_derives_heads_from_actual_generator():
    cfg = SimpleNamespace(VisibleConditioning=dict(diffusion_enabled=True, classifier_enabled=True,
        diffusion_film_scale_enabled=False, diffusion_pair_attention=dict(enabled=True)))
    kwargs = dict(feature_names=["Part_eta", "Part_phi", "Part_energy", "Part_pt"],
                  token_dim=8, hidden_dim=8, num_layers=3, n_branches=2, attention_heads=2)
    classifier = visible_conditioning_spec(cfg, target="classifier", **kwargs)
    diffusion = visible_conditioning_spec(cfg, target="diffusion", **kwargs)
    assert "film_scale_enabled" not in classifier and "pair_attention" not in classifier
    assert diffusion["film_scale_enabled"] is False
    assert diffusion["pair_attention"]["heads"] == 2
    cfg.VisibleConditioning["diffusion_film_scale_enabled"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        visible_conditioning_spec(cfg, target="diffusion", **kwargs)


@pytest.mark.parametrize("arm", ["context_shift_only", "pair_attention"])
def test_resolved_configs_match_control_and_keep_jobs_isolated(arm):
    def config(name):
        return deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                           read_overlay_yaml(ROOT / f"config/train_diffusion_{name}.yaml"))
    control, current = config("relation_context"), config(arm)
    assert current["platform"] == control["platform"]
    assert current["platform"]["number_of_workers"] == 16
    assert current["platform"]["batch_size"] == 2048
    training = copy.deepcopy(current["options"]["Training"])
    assert training["model_checkpoint_save_path"] != control["options"]["Training"]["model_checkpoint_save_path"]
    training["model_checkpoint_save_path"] = control["options"]["Training"]["model_checkpoint_save_path"]
    assert training == control["options"]["Training"]
    assert training["epochs"] == training["total_epochs"] == 50
    assert training["strict_relation_source"] and training["model_checkpoint_save_last"] is True
    assert not training["EMA"]["enable"] and not current["rl"]["enabled"]
    assert training["model_checkpoint_load_path"] is None
    network = copy.deepcopy(current["network"])
    visible = network["VisibleConditioning"]
    assert visible.pop("diffusion_film_scale_enabled") is (arm == "pair_attention")
    if arm == "pair_attention":
        assert visible.pop("diffusion_pair_attention") == dict(enabled=True, width=32, seed=20260930)
    assert network == control["network"]
    assert current["experiment"]["matched_control_wandb_run"] == "relcontext02"
    assert current["logger"]["wandb"]["id"] == {"context_shift_only": "ctxshift02", "pair_attention": "pairattn02"}[arm]
    assert current["nersc"]["ray"]["results_dir"] != control["nersc"]["ray"]["results_dir"]
    assert f"--overlay-config config/train_diffusion_{arm}.yaml" in current["nersc"]["execution"]["command"]
    assert len(current["logger"]["wandb"]["run_name"]) < 96


def test_endpoint_contrast_sign_pairing_and_bootstrap_reproducibility():
    import numpy as np
    from compare_conditioning_endpoints import paired_joint_w1
    t = np.array([[.01, .02], [.03, .04], [.05, .06], [.07, .08]])
    exact = np.repeat(t[:, None, :], 3, axis=1)
    bad = exact + .2
    same = paired_joint_w1(t, bad, bad, bootstrap=50)
    assert same["candidate_minus_control_w1"] == 0 and same["paired_event_ci95"] == [0., 0.]
    better = paired_joint_w1(t, bad, exact, bootstrap=50)
    worse = paired_joint_w1(t, exact, bad, bootstrap=50)
    assert better == paired_joint_w1(t, bad, exact, bootstrap=50)
    assert better["candidate_minus_control_w1"] == pytest.approx(-.2)
    assert worse["candidate_minus_control_w1"] == pytest.approx(.2)
    np.testing.assert_allclose(better["paired_event_ci95"], -np.array(worse["paired_event_ci95"])[::-1])
    with pytest.raises(ValueError, match="aligned"):
        paired_joint_w1(t, bad, exact[:2], bootstrap=50)
