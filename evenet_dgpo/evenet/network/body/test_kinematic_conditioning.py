"""Angle + momentum conditioning without full-feature/global expansion."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evenet.network.body.normalizer import Normalizer
from evenet.network.body.visible_conditioning import VisibleConditioning, visible_conditioning_spec
from evenet.network.body.test_visible_conditioning import spec, inputs, open_heads, classifier_factory
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _event_batch_with_pair_context


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numeric_mapping_masks_and_other_features_not_expanded(dtype):
    torch.manual_seed(10)
    branch = VisibleConditioning(**spec(kinematics=True)).to(dtype)
    raw, tokens, mask = inputs()
    raw, tokens = raw.to(dtype), tokens.to(dtype)
    normed = (raw - 10.) / 7.
    raw[~mask.expand_as(raw)] = float("nan")
    normed[~mask.expand_as(normed)] = float("nan")
    tokens[~mask.expand_as(tokens)] = float("nan")
    captures = []
    hook = branch.encoder.register_forward_pre_hook(lambda _m, args: captures.append(args[0].detach().clone()))
    open_heads(branch)
    out = branch(raw, tokens, mask, normalized_raw=normed)
    hook.remove()
    # LN(tokens):4, angular Fourier:16, numeric scalar:1, numeric Fourier:8.
    expected = torch.where(mask, normed[..., :1], 0.)
    torch.testing.assert_close(captures[0][..., 20:21], expected)
    phase = expected * expected.new_tensor([.25, .5, 1., 2.])
    torch.testing.assert_close(captures[0][..., 21:29], torch.cat((phase.sin(), phase.cos()), -1))
    changed_category = raw.clone(); changed_category[..., 3] += 13.
    changed_normalized = normed.clone(); changed_normalized[..., 1:] += 20.
    shifted = raw.clone(); shifted[..., 2] += 2 * torch.pi
    other = branch(changed_category, tokens, mask, normalized_raw=changed_normalized)
    periodic = branch(shifted, tokens, mask, normalized_raw=normed)
    permuted = branch(raw.flip(1), tokens.flip(1), mask.flip(1), normalized_raw=normed.flip(1))
    for a_layer, b_layer, c_layer, d_layer in zip(out, other, periodic, permuted):
        for a, b, c, d in zip(a_layer, b_layer, c_layer, d_layer):
            assert torch.isfinite(a).all() and a[1].count_nonzero() == 0
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(a, c, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(a, d, rtol=1e-5, atol=1e-6)
    changed = normed.clone(); changed[..., 0] += .9
    assert not torch.allclose(out[0][0], branch(raw, tokens, mask, normalized_raw=changed)[0][0])
    sum(value.square().sum() for layer in out for value in layer).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    assert branch.encoder[0].weight.grad.norm() > 0
    assert "numerical_input_rms" in branch.diagnostics


def test_selected_numeric_features_require_known_schema_and_normalization():
    settings = spec(kinematics=True)
    for change in (dict(numerical_features=[]), dict(numerical_features=["energy", "energy"]),
                   dict(numerical_features=["missing"]), dict(numerical_features=["Part_phi"]),
                   dict(feature_mode="full"), dict(numerical_frequencies=[0., 1.]),
                   dict(numerical_frequencies=[float("nan")])):
        with pytest.raises(ValueError):
            VisibleConditioning(**{**settings, **change})
    branch = VisibleConditioning(**settings)
    raw, tokens, mask = inputs()
    with pytest.raises(ValueError, match="normalized observed inputs"):
        branch(raw, tokens, mask)
    with pytest.raises(ValueError, match="normalized observed inputs"):
        branch(raw, tokens, mask, normalized_raw=raw[..., :2])


def test_classifier_uses_checkpoint_normalization_before_grouped_projection():
    batch = _event_batch_with_pair_context()
    packed, packing = pack_event_inputs(batch, include_pairwise_context=True)
    model = classifier_factory(True, kinematics=True)(packing)
    model.backbone.sequential_normalizer = Normalizer(torch.tensor([3., 0., 0., 0.]),
        torch.tensor([2., 1., 1., 1.]), torch.tensor([1, 1, 1, 0], dtype=torch.bool))
    captures = []
    hook = model.bank.visible_conditioning.register_forward_pre_hook(
        lambda _m, args, kwargs: captures.append((args, kwargs)), with_kwargs=True)
    model(packed, torch.randn(3, 4))
    args, kwargs = captures[0]
    assert set(kwargs) == {"normalized_raw"}
    torch.testing.assert_close(kwargs["normalized_raw"][..., 0], (batch["x"][..., 0] - 3.) / 2.)
    assert not torch.allclose(args[1], kwargs["normalized_raw"])
    hook.remove()


def test_saved_kinematic_and_angle_classifiers_override_opposite_live_mode():
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    for kinematics in (False, True):
        model = classifier_factory(True, kinematics)(packing).eval()
        open_heads(model.bank.visible_conditioning)
        with torch.no_grad(): model.bank.output.weight.normal_(std=.1)
        payload = model.peft_payload()
        restored = EvenetAdapterRatioClassifier.from_peft_payload(payload,
            model_builder=classifier_factory(True, not kinematics), device=torch.device("cpu"))
        assert restored.bank.visible_conditioning.feature_mode == ("kinematics" if kinematics else "angles")
        assert restored.peft_payload()["classifier_config"] == payload["classifier_config"]
        with torch.no_grad():
            candidate = torch.randn(3, 4)
            torch.testing.assert_close(model(packed, candidate), restored(packed, candidate), rtol=1e-6, atol=1e-7)


def test_kinematic_overlay_schema_scope_and_unchanged_training_setup():
    from train_neutrino_backend import read_yaml, read_overlay_yaml, deep_update
    root = Path(__file__).resolve().parents[4]
    baseline = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
                           read_overlay_yaml(root / "config/dgpo_h4_visible_adaln.yaml"))
    cfg = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
                     read_overlay_yaml(root / "config/dgpo_h4_kinematic_adaln.yaml"))
    # Generate the input schema from the existing analysis rather than copying it.
    import importlib.util
    import sys
    module_spec = importlib.util.spec_from_file_location("generate_event_info_yaml", root / "generate_event_info_yaml.py")
    generator = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = generator
    module_spec.loader.exec_module(generator)
    schema = generator.parse_feature_config(read_yaml(root / "config/analysis.yaml"))
    settings = cfg["network"]["VisibleConditioning"]
    resolved = visible_conditioning_spec(SimpleNamespace(VisibleConditioning=settings), target="diffusion",
        feature_names=schema.raw_sequential_fields, token_dim=len(schema.projected_sequential_feature_names),
        hidden_dim=256, num_layers=1, n_branches=2)
    branch = VisibleConditioning(**resolved)
    assert resolved["numerical_features"] == ["Part_energy", "Part_pt"]
    assert resolved["harmonics"] == [1, 2, 3, 4]
    assert branch.encoder[0].in_features == 41 and branch.event_encoder[0].in_features == 65
    assert len(branch.modulations) == 1
    base_branch = VisibleConditioning(**{k: v for k, v in resolved.items()
        if k not in ("feature_mode", "numerical_features", "numerical_frequencies")})
    assert sum(p.numel() for p in branch.parameters()) - sum(p.numel() for p in base_branch.parameters()) == 1152
    # Test all actual raw fields, including two numerical dimensions, with gradients.
    open_heads(branch)
    raw = torch.randn(2, 5, len(schema.raw_sequential_fields))
    tokens = torch.randn(2, 5, len(schema.projected_sequential_feature_names))
    normed = raw.clone().requires_grad_()
    output = branch(raw, tokens, torch.ones(2, 5, 1), normalized_raw=normed)
    sum(v.square().sum() for layer in output for v in layer).backward()
    for index, name in enumerate(schema.raw_sequential_fields):
        assert bool(normed.grad[..., index].count_nonzero()) == (name in ("Part_energy", "Part_pt"))
    assert cfg["dgpo"] == baseline["dgpo"]
    assert cfg["platform"] == baseline["platform"] and cfg["platform"]["number_of_workers"] == 16
    assert cfg["network"]["Body"] == baseline["network"]["Body"]
    for key in baseline["options"]["Training"]:
        if key != "model_checkpoint_save_path":
            assert cfg["options"]["Training"][key] == baseline["options"]["Training"][key]
    assert cfg["logger"]["wandb"]["id"] == "h4kfilm1"
    assert cfg["logger"]["wandb"]["resume"] == "never" and cfg["logger"]["wandb"]["fresh_run"]
    assert cfg["logger"]["wandb"]["id"] != baseline["logger"]["wandb"]["id"]


def test_disabled_and_angle_modes_preserve_legacy_spec_and_state():
    settings = spec()
    branch = VisibleConditioning(**settings)
    assert "feature_mode" not in branch.spec and "numerical_frequencies" not in branch.state_dict()
    cfg = SimpleNamespace(VisibleConditioning=dict(diffusion_enabled=False, feature_mode="kinematics"))
    assert visible_conditioning_spec(cfg, target="diffusion", feature_names=settings["feature_names"],
        token_dim=4, hidden_dim=8, num_layers=2, n_branches=2) is None
