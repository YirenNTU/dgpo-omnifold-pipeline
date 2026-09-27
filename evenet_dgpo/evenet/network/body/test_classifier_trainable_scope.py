"""Pretrained body frozen except last PET block; new conditioning can learn."""
import copy
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from evenet.network.body.angular_conditioning import VisibleAngularFourier
from evenet.network.body.embedding import PETBody
from evenet.network.body.test_visible_conditioning import VisibleBackbone, spec
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterModelBuilder, EvenetAdapterRatioClassifier, FrozenResidualRatioReward,
    configure_adapter_training, pack_event_inputs, peft_bank_factory,
)
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig, fit_density_ratio
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _event_batch_with_pair_context


def template(kinematics=False):
    torch.manual_seed(23)
    model = VisibleBackbone()
    model.network_cfg.VisibleConditioning = dict(classifier_enabled=True, width=12)
    if kinematics:
        model.network_cfg.VisibleConditioning.update(feature_mode="kinematics", numerical_features=["energy"])
    model.PET = PETBody(num_feat=4, num_keep=4, feature_drop=0., projection_dim=8,
        local=False, K=1, num_local=0, num_layers=2, num_heads=2,
        drop_probability=0., talking_head=False, layer_scale=False,
        layer_scale_init=1., dropout=0., mode="all", use_adapter=True,
        adapter_bottleneck=4, angular_conditioning=VisibleAngularFourier(
            spec()["feature_names"], 8, "Part_eta", "Part_phi"))
    return model


def classifier(backbone, packing, train_angular=True):
    return EvenetAdapterRatioClassifier(copy.deepcopy(backbone), packing,
        train_backbone=False, train_last_pet_block=True,
        train_invisible_projector=True, train_angular_conditioning=train_angular,
        train_grouped_sequential_embedding=False, train_encoder=False, train_layernorm=False,
        decoder_layers=3, decoder_hidden_dim=8, decoder_heads=2, adapter_bottleneck=4,
        head_dropout=0., topology_dropout=0., periodic_pair_features=True,
        topology_fourier_embedding=True, topology_max_harmonic=4,
        topology_hidden_dim=16, topology_embedding_dim=8, topology_fusion_hidden_dim=16)


@pytest.mark.parametrize("kinematics", [False, True])
def test_scope_updates_real_pet_and_projector_without_changing_frozen_weights(kinematics):
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    backbone = template(kinematics)
    model = classifier(backbone, packing)
    allowed = ("PET.transformer_blocks.1.", "PET.adapters.",
               "PET.angular_conditioning.", "InvisibleInputProjector.")
    before = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    for name, parameter in model.backbone.named_parameters():
        assert parameter.requires_grad == name.startswith(allowed), name
    model.train()
    assert not model.backbone.PET.transformer_blocks[0].training
    assert model.backbone.PET.transformer_blocks[1].training
    assert model.backbone.PET.angular_conditioning.training
    assert not model.backbone.GlobalEmbedding.training
    candidate = torch.randn(3, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    for _ in range(6):
        optimizer.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(packed, candidate), torch.tensor([0., 1., 0.])).backward()
        optimizer.step()
    for name, parameter in model.backbone.named_parameters():
        if not name.startswith(allowed):
            assert parameter.grad is None, name
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    for prefix in allowed:
        pairs = [(n, p) for n, p in model.backbone.named_parameters() if n.startswith(prefix)]
        assert any(not torch.equal(p, before[n]) for n, p in pairs), prefix
        assert any(p.grad is not None and p.grad.norm() > 0 for _, p in pairs), prefix
    assert model.bank.visible_conditioning.encoder[0].weight.grad.norm() > 0
    assert all(layer.weight.grad.norm() > 0 for layer in model.bank.visible_conditioning.modulations)
    assert model.trainable_parameter_counts["angular_conditioning"] > 0
    assert model.trainable_parameter_counts["invisible_projector"] > 0
    payload = model.peft_payload()
    assert payload["classifier_config"]["train_angular_conditioning"]
    assert any(n.startswith("PET.angular_conditioning.") for n in payload["body"])
    assert not any(n.startswith("PET.transformer_blocks.0.") for n in payload["body"])
    members = tuple(copy.deepcopy(model).eval() for _ in range(4))
    reward = FrozenResidualRatioReward(members, checkpoint_coefficients=(.5,) * 4,
        checkpoint_iterations=(1, 1, 2, 2), tempering=.75)
    restored = FrozenResidualRatioReward.from_serializable_payload(reward.serializable_payload(),
        model_builder=lambda packing: classifier(backbone, packing), device=torch.device("cpu"))
    torch.testing.assert_close(reward(packed, candidate), restored(packed, candidate), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kinematics", [False, True])
def test_actual_fit_angular_lr_and_frozen_parameter_exclusion(kinematics):
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    model = classifier(template(kinematics), packing)
    rows, optimizers = [], []
    actual_adamw = torch.optim.AdamW
    def capture_optimizer(params, **kwargs):
        result = actual_adamw(params, **kwargs)
        optimizers.append(result)
        return result
    cfg = RatioFitConfig(steps=6, min_steps=0, batch_size=3, learning_rate=2e-4,
        backbone_learning_rate=1e-5, decoder_learning_rate=5e-5, adapter_learning_rate=5e-5,
        progress_interval_steps=1, diagnostic_enabled=False, representation_diagnostic_enabled=False)
    with mock.patch("torch.optim.AdamW", side_effect=capture_optimizer):
        fit_density_ratio(model, packed, torch.randn(3, 4), torch.ones(3),
            packed, torch.randn(3, 4), torch.ones(3), cfg, seed=11, progress_callback=rows.append)
    groups = {group["group_name"]: group for group in optimizers[0].param_groups}
    assert rows[-1]["optimizer_group_lr_angular_conditioning"] == 2e-4
    assert rows[-1]["optimizer_group_lr_visible_conditioning"] == 2e-4
    for name, rate in (("backbone", 1e-5), ("decoder", 5e-5), ("adapter", 5e-5)):
        assert groups[name]["lr"] == rate
    assert {id(p) for p in groups["angular_conditioning"]["params"]} == {
        id(p) for p in model.backbone.PET.angular_conditioning.parameters()}
    assert set(map(id, model.backbone.InvisibleInputProjector.parameters())) <= set(map(id, groups["head"]["params"]))
    ids = [id(p) for group in groups.values() for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    frozen = {id(p) for p in model.backbone.parameters() if not p.requires_grad}
    assert not frozen.intersection(ids)


def test_builder_propagates_flag_independent_folds_and_old_payload(tmp_path):
    backbone = template()
    checkpoint = tmp_path / "pretrain.ckpt"
    checkpoint.touch()
    cfg = SimpleNamespace(network=copy.deepcopy(backbone.network_cfg))
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    with mock.patch("RL.DGPO_neutrino.model_utils.build_evenet_on_device", side_effect=lambda *a: copy.deepcopy(backbone)), \
         mock.patch("RL.DGPO_neutrino.model_utils.load_weights_like_configure_model"):
        builder = EvenetAdapterModelBuilder(config=cfg, normalization_dict={},
            checkpoint_path=checkpoint, device=torch.device("cpu"),
            train_last_pet_block=True, train_invisible_projector=True, train_angular_conditioning=True,
            decoder_layers=3, decoder_hidden_dim=8, decoder_heads=2, adapter_bottleneck=4,
            periodic_pair_features=True, topology_fourier_embedding=True, head_dropout=0., topology_dropout=0.)
        first = builder.make_classifier(packing, "fold1")
        second = peft_bank_factory(builder, packing, "audit", classifier_overrides={"train_angular_conditioning": True})()
        assert first._train_angular_conditioning and second._train_angular_conditioning
        assert not set(map(id, first.parameters())).intersection(map(id, second.parameters()))
        old = builder.make_classifier(packing, "old", train_angular_conditioning=False).eval()
        payload = old.peft_payload()
        assert "train_angular_conditioning" not in payload["classifier_config"]
        assert not any(n.startswith("PET.angular_conditioning.") for n in payload["body"])
        restored = EvenetAdapterRatioClassifier.from_peft_payload(payload,
            model_builder=lambda packing: builder.make_classifier(packing, "restore",
                train_angular_conditioning=payload["classifier_config"].get("train_angular_conditioning", False)),
            device=torch.device("cpu"))
        assert restored.peft_payload()["classifier_config"] == payload["classifier_config"]
        candidate = torch.randn(3, 4)
        torch.testing.assert_close(old(packed, candidate), restored(packed, candidate), rtol=1e-6, atol=1e-7)


def test_missing_fourier_branch_rejected_without_unfreezing_whole_pet():
    model = VisibleBackbone()
    with pytest.raises(ValueError, match="PET.angular_conditioning"):
        configure_adapter_training(model, train_angular_conditioning=True)
