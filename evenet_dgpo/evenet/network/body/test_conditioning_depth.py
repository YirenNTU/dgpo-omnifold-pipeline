"""Three-block migration: pretrained function, learning, and frozen reward."""
import copy
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from evenet.network.body.test_visible_conditioning import (
    spec, head_inputs, open_heads, classifier_factory,
)
from evenet.network.evenet_model import EveNetModel
from evenet.control.global_config import DotDict
from evenet.network.heads.generation.generation_head import EventGenerationHead
from evenet.utilities.tool import safe_load_state
from evenet.utilities.fourier_integration import optimizer_parameters
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import FrozenResidualRatioReward, pack_event_inputs
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _event_batch_with_pair_context
from RL.DGPO_neutrino.model_utils import load_weights_like_configure_model, select_dgpo_training_state


STEP1110_SOURCE = (
    "/pscratch/sd/y/yiren/Ztautau/"
    "dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/"
    "checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt"
)
CLASSIFIER_PRETRAIN_SOURCE = (
    "/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt"
)


def make_head(depth, *, conditioning=True, kinematics=False, layer_scale=True, identity=None):
    settings = {**spec(kinematics=kinematics), "num_layers": depth} if conditioning else None
    return EventGenerationHead(input_dim=8, projection_dim=8, num_global_cond=8,
        num_classes=2, output_dim=4, num_layers=depth, num_heads=2, dropout=0.,
        layer_scale=layer_scale, layer_scale_init=1e-5, drop_probability=0.,
        feature_drop=0., visible_conditioning=settings, identity_init_from_layer=identity)


@pytest.mark.parametrize("kinematics", [False, True])
@pytest.mark.parametrize("trained_conditioning", [False, True])
@pytest.mark.parametrize("layer_scale", [False, True])
def test_shallow_checkpoint_preserves_velocity_input_gradient_and_anchor(kinematics, trained_conditioning, layer_scale):
    torch.manual_seed(72)
    base = make_head(1, conditioning=trained_conditioning, kinematics=kinematics, layer_scale=layer_scale)
    if trained_conditioning:
        open_heads(base.visible_conditioning)
    # Emulate trained LayerScale: the original block must not be reset.
    if layer_scale:
        with torch.no_grad():
            base.gen_transformer_blocks[0].layer_scale1.gamma.fill_(.15)
            base.gen_transformer_blocks[0].layer_scale2.gamma.fill_(.2)
    new = make_head(3, kinematics=kinematics, layer_scale=layer_scale, identity=1)
    safe_load_state(new, {"model." + k: v for k, v in base.state_dict().items()}, verbose=False)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
    args = head_inputs(kinematics)
    args["x"].requires_grad_()
    base.eval(); new.eval()
    original, actual = base(**args), new(**args)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    grad0 = torch.autograd.grad(original[:, -2:].square().sum(), args["x"])[0]
    grad1 = torch.autograd.grad(actual[:, -2:].square().sum(), args["x"])[0]
    torch.testing.assert_close(grad1, grad0, rtol=0, atol=0)
    for block in new.gen_transformer_blocks[1:]:
        assert block.attn.out_proj.weight.count_nonzero() == 0
        assert block.mlp[-1].weight.count_nonzero() == 0
        assert block.attn.in_proj_weight.count_nonzero() > 0
        assert block.mlp[0].weight.count_nonzero() > 0
        if layer_scale:
            assert torch.all(block.layer_scale1.gamma == 1)
            assert torch.all(block.layer_scale2.gamma == 1)
    # Rebuild + load, as the frozen reference does, must use the same new weights.
    reference = make_head(3, kinematics=kinematics, layer_scale=layer_scale, identity=1)
    reference.load_state_dict(new.state_dict()); reference.eval().requires_grad_(False)
    # Frozen MHA may select a different CPU fast path; allow float32 roundoff.
    torch.testing.assert_close(new(**args), reference(**args), rtol=1e-6, atol=3e-7)


@pytest.mark.parametrize("kinematics", [False, True])
def test_all_three_diffusion_blocks_learn_and_full_resume_keeps_new_weights(kinematics):
    torch.manual_seed(23)
    base = make_head(1, conditioning=False)
    new = make_head(3, kinematics=kinematics, identity=1)
    safe_load_state(new, base.state_dict(), verbose=False)
    args = head_inputs(kinematics)
    target = torch.randn(3, 2, 4)
    opt = torch.optim.AdamW(new.parameters(), lr=.002)
    for step in range(5):
        opt.zero_grad()
        nn.functional.mse_loss(new(**args)[:, -2:], target).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
        for block in new.gen_transformer_blocks[1:]:
            assert block.attn.out_proj.weight.grad.norm() > 0
            assert block.mlp[-1].weight.grad.norm() > 0
            if step == 0:
                # Zero output projections delay, but do not permanently block, inner gradients.
                assert block.mlp[0].weight.grad.count_nonzero() == 0
        opt.step()
    for block, modulation in zip(new.gen_transformer_blocks, new.visible_conditioning.modulations):
        assert block.mlp[0].weight.grad.norm() > 0
        assert block.attn.in_proj_weight.grad.norm() > 0
        assert modulation.weight.grad.norm() > 0
    assert new.visible_conditioning.encoder[0].weight.grad.norm() > 0
    assert new.visible_conditioning.event_encoder[0].weight.grad.norm() > 0
    restored = make_head(3, kinematics=kinematics, identity=1)
    safe_load_state(restored, new.state_dict(), verbose=False)
    torch.testing.assert_close(restored(**args), new(**args), rtol=0, atol=0)
    assert restored.gen_transformer_blocks[2].attn.out_proj.weight.count_nonzero() > 0
    wrapper = nn.Module(); wrapper.TruthGeneration = new
    paths = ["TruthGeneration", "TruthGeneration.visible_conditioning"]
    params = [p for path in paths for p in optimizer_parameters(wrapper, [path], paths)]
    assert len(params) == len({id(p) for p in params}) == len(list(wrapper.parameters()))


@pytest.mark.parametrize("kinematics", [False, True])
def test_three_block_classifier_gradients_and_four_member_reward_roundtrip(kinematics):
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    builder = classifier_factory(True, kinematics, num_layers=3)
    model = builder(packing)
    candidate = torch.randn(3, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(6):
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(model(packed, candidate), torch.tensor([0., 1., 0.])).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert len(model.bank.decoder.blocks) == len(model.bank.visible_conditioning.modulations) == 3
    for block, modulation in zip(model.bank.decoder.blocks, model.bank.visible_conditioning.modulations):
        assert block.self_attn.in_proj_weight.grad.norm() > 0
        assert block.cross_attn.in_proj_weight.grad.norm() > 0
        assert block.ffn[0].weight.grad.norm() > 0
        assert modulation.weight.grad.norm() > 0
    assert model.bank.visible_conditioning.encoder[0].weight.grad.norm() > 0
    members = tuple(copy.deepcopy(model).eval() for _ in range(4))
    reward = FrozenResidualRatioReward(members, checkpoint_coefficients=(.5,) * 4,
        checkpoint_iterations=(1, 1, 2, 2), tempering=.75)
    restored = FrozenResidualRatioReward.from_serializable_payload(reward.serializable_payload(),
        model_builder=builder, device=torch.device("cpu"))
    torch.testing.assert_close(reward(packed, candidate), restored(packed, candidate), rtol=1e-6, atol=1e-7)
    assert all(not p.requires_grad for p in restored.parameters())


@pytest.mark.parametrize("bad_index", [-1, 3, 1.5, True])
def test_identity_init_index_is_explicit_and_validated(bad_index):
    with pytest.raises(ValueError, match="identity_init_from_layer"):
        make_head(3, identity=bad_index)


@pytest.mark.parametrize("kinematics", [False, True])
@pytest.mark.parametrize("policy_load", [False, True])
def test_step1110_raw_load_ignores_ema_and_discards_old_experiment_state(tmp_path, kinematics, policy_load):
    torch.manual_seed(71)
    base = make_head(1, conditioning=False)
    # Use the real checkpoint selection/loading path, including a conflicting EMA.
    checkpoint = dict(
        dgpo_checkpoint_version=1, global_step=1110, epoch=110,
        state_dict={"model." + k: v.clone() for k, v in base.state_dict().items()},
        ema_state_dict={"model." + k: v.clone() + 9 for k, v in base.state_dict().items()},
        dgpo_optimizer_state_dict={"old": True}, dgpo_ref_state_dict={"old": True},
        dgpo_round_ref_state_dict={"old": True}, dgpo_omnifold_reward_stack={"old": True},
        dgpo_adaptive_omnifold_state={"old": True}, dgpo_next_epoch=111,
    )
    path = tmp_path / "step1110.ckpt"
    torch.save(checkpoint, path)
    cfg = DotDict({"options": {"Training": {
        "EMA": {"enable": False, "replace_model_after_load": False}}}})
    new = make_head(3, kinematics=kinematics, identity=1)
    loaded = load_weights_like_configure_model(new, path, torch.device("cpu"), cfg,
        for_dgpo_training=policy_load)
    assert loaded["global_step"] == 1110
    assert select_dgpo_training_state(loaded, load_mode="weights_only") is None
    args = head_inputs(kinematics)
    base.eval(); new.eval()
    torch.testing.assert_close(new(**args), base(**args), rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["visible", "kinematic"])
def test_resolved_depth_config_preserves_feature_and_training_controls(arm):
    from train_neutrino_backend import read_yaml, read_overlay_yaml, deep_update
    root = Path(__file__).resolve().parents[4]
    def resolve(suffix):
        return deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
            read_overlay_yaml(root / f"config/dgpo_h4_{arm}_adaln{suffix}.yaml"))
    old, new = resolve(""), resolve("_depth3")
    assert new["network"]["TruthGeneration"]["num_layers"] == 3
    assert new["network"]["TruthGeneration"]["identity_init_from_layer"] == 1
    assert "identity_init_from_layer" in inspect.getsource(EveNetModel.__init__)
    assert new["network"]["VisibleConditioning"] == old["network"]["VisibleConditioning"]
    expected_dgpo = copy.deepcopy(old["dgpo"])
    expected_dgpo["conditioning_learning_rates"] = {
        "angular_conditioning": 1e-4, "visible_conditioning": 1e-4,
    }
    expected_dgpo["unbounded_training"] = False
    expected_dgpo["lr_schedule"] = {
        "type": "cosine", "total_epochs": 1500, "total_steps": None,
        "min_lr_ratio": .1, "groups": ["angular_conditioning", "visible_conditioning"],
    }
    for scope in ("recalibration", "audit_fit"):
        expected_dgpo["adaptive_omnifold"][scope].update(
            decoder_layers=3, train_backbone=False, train_last_pet_block=True,
            train_encoder=False, train_layernorm=False,
            train_grouped_sequential_embedding=False, train_invisible_projector=True,
            train_angular_conditioning=True,
        )
    expected_dgpo["adaptive_omnifold"]["audit_fit"].update(
        training_population="omnifold_fold", training_fold=1,
    )
    assert new["dgpo"] == expected_dgpo
    for key, value in old["options"]["Training"].items():
        if key not in ("model_checkpoint_save_path", "model_checkpoint_load_path", "epochs", "total_epochs"):
            assert new["options"]["Training"][key] == value
    assert new["options"]["Training"]["epochs"] == new["options"]["Training"]["total_epochs"] == 1500
    assert new["dgpo"]["steps_per_epoch"] == 10
    assert new["options"]["Training"]["model_checkpoint_load_path"] == STEP1110_SOURCE
    assert new["nersc"]["reproducibility"]["source_checkpoint"] == STEP1110_SOURCE
    assert new["experiment"]["source_run"] == "c4a91e07"
    assert new["experiment"]["source_policy_step"] == 1110
    assert new["dgpo"]["checkpoint_load_mode"] == "weights_only"
    assert not new["dgpo"]["auto_resume_from_last"]
    assert new["dgpo"]["adaptive_omnifold"]["recalibration"]["bootstrap_on_start"]
    assert new["reward_config"]["omnifold"]["bootstrap_in_dgpo"]
    assert new["reward_config"]["omnifold"]["bundle_file"] is None
    classifier_source = new["reward_config"]["omnifold"]["backbone_checkpoint"]
    assert classifier_source == CLASSIFIER_PRETRAIN_SOURCE
    assert classifier_source != new["options"]["Training"]["model_checkpoint_load_path"]
    assert new["nersc"]["reproducibility"]["classifier_backbone_checkpoint"] == classifier_source
    ema = new["options"]["Training"]["EMA"]
    assert not any(ema[k] for k in ("enable", "replace_model_after_load", "use_for_generation", "use_ema_during_training_eval"))
    assert new["platform"] == old["platform"] and new["platform"]["number_of_workers"] == 16
    assert new["logger"]["wandb"]["id"] != old["logger"]["wandb"]["id"]
    assert new["logger"]["wandb"]["fresh_run"] and new["logger"]["wandb"]["resume"] == "never"
    assert "_depth3_1110/" in new["options"]["Training"]["model_checkpoint_save_path"]
    assert new["logger"]["wandb"]["id"] == ("h4vad3s1110" if arm == "visible" else "h4kfd3s1110")
    assert "_depth3.yaml" in new["nersc"]["execution"]["command"]
