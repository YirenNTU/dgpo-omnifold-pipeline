"""CPU contracts for nonlinear visible-only conditioning and legacy rewards."""
import copy
import inspect
from pathlib import Path
import sys
from unittest import mock

import pytest
import torch
from torch import nn

from evenet.network.body.visible_conditioning import VisibleConditioning, modulate_visible_condition
from evenet.network.heads.generation.generation_head import EventGenerationHead
from evenet.network.evenet_model import EveNetModel
from evenet.utilities.tool import safe_load_state
from evenet.utilities.fourier_integration import optimizer_parameters
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterRatioClassifier, FrozenResidualRatioReward, pack_event_inputs,
)
from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import (
    _FakeZtautauBackbone, _event_batch_with_pair_context,
)


def spec(n_branches=2, kinematics=False):
    result = dict(feature_names=["energy", "Part_eta", "Part_phi", "type"],
                token_dim=4, hidden_dim=8, num_layers=2, n_branches=n_branches, width=12)
    if kinematics:
        result.update(feature_mode="kinematics", numerical_features=["energy"])
    return result


def inputs():
    raw, tokens = torch.randn(3, 4, 4), torch.randn(3, 4, 4)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)[..., None]
    return raw, tokens, mask


def open_heads(branch):
    with torch.no_grad():
        for head in branch.modulations:
            head.weight.normal_(std=.03)
            head.bias.fill_(.02)


def test_masks_empty_permutation_periodicity_and_gradients():
    branch = VisibleConditioning(**spec())
    raw, tokens, mask = inputs()
    raw[~mask.expand_as(raw)] = float("nan")
    tokens[~mask.expand_as(tokens)] = float("nan")
    out = branch(raw, tokens, mask)
    assert all(x.count_nonzero() == 0 for layer in out for x in layer)
    sum(x.sum() for layer in out for x in layer).backward()
    assert branch.modulations[0].weight.grad.norm() > 0
    open_heads(branch)
    out = branch(raw, tokens, mask)
    shifted = raw.clone(); shifted[..., 2] += 2 * torch.pi
    for expected, actual, permuted in zip(out, branch(shifted, tokens, mask), branch(raw.flip(1), tokens.flip(1), mask.flip(1))):
        for a, b, c in zip(expected, actual, permuted):
            assert torch.isfinite(a).all() and a[1].count_nonzero() == 0
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(a, c, rtol=1e-5, atol=1e-6)
    branch.zero_grad()
    sum(x.square().sum() for layer in out for x in layer).backward()
    assert branch.encoder[0].weight.grad.norm() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    changed = raw.clone(); changed[:, :2, 2] += .7
    assert not torch.allclose(out[0][0], branch(changed, tokens, mask)[0][0])


def head(enabled, kinematics=False):
    return EventGenerationHead(input_dim=8, projection_dim=8, num_global_cond=8,
        num_classes=2, output_dim=4, num_layers=2, num_heads=2, dropout=0.,
        layer_scale=False, layer_scale_init=1., drop_probability=0., feature_drop=0.,
        visible_conditioning=spec(kinematics=kinematics) if enabled else None)


def head_inputs(kinematics=False):
    raw, tokens, vmask = inputs()
    mask = torch.cat((vmask, torch.ones(3, 2, 1, dtype=torch.bool)), dim=1)
    attn = torch.zeros(6, 6, dtype=torch.bool); attn[:4, 4:] = True
    # Nonempty invisible keys are available for padded visible-query rows.
    attn = attn[None].expand(3, -1, -1) & mask
    args = dict(x=torch.randn(3, 6, 8), global_cond=torch.randn(3, 1, 8),
        global_cond_mask=torch.ones(3, 1, 1), num_x=None, x_mask=mask,
        time=torch.rand(3), label=torch.zeros(3, 1, dtype=torch.long), attn_mask=attn,
        time_masking=torch.cat((torch.zeros_like(vmask), torch.ones(3, 2, 1, dtype=torch.bool)), dim=1),
        visible_raw=raw, visible_tokens=tokens, visible_mask=vmask)
    if kinematics:
        args.update(visible_normalized=(raw - 2.) / 3.)
    return args


@pytest.mark.parametrize("kinematics", [False, True])
def test_diffusion_zero_start_gradient_path_optimizer_and_roundtrip(kinematics):
    torch.manual_seed(31)
    base, new = head(False), head(True, kinematics)
    safe_load_state(new, base.state_dict(), verbose=False)
    args = head_inputs(kinematics)
    old, actual = base(**args), new(**args)
    torch.testing.assert_close(old, actual, atol=0, rtol=0)
    args_no_context = {k: v for k, v in args.items() if not k.startswith("visible_")}
    with pytest.raises(ValueError, match="observed context"):
        new(**args_no_context)
    opt = torch.optim.AdamW(new.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad()
        pred = new(**args)
        pred[:, -2:].square().mean().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
        opt.step()
    assert new.visible_conditioning.modulations[0].weight.grad.norm() > 0
    assert new.visible_conditioning.encoder[0].weight.grad.norm() > 0
    assert not torch.allclose(old[:, -2:], new(**args)[:, -2:])
    restored = head(True, kinematics); restored.load_state_dict(new.state_dict())
    torch.testing.assert_close(new(**args), restored(**args), atol=0, rtol=0)
    wrapper = nn.Module(); wrapper.TruthGeneration = new
    paths = ["TruthGeneration", "TruthGeneration.visible_conditioning"]
    groups = [optimizer_parameters(wrapper, [p], paths) for p in paths]
    ids = [id(p) for group in groups for p in group]
    assert len(set(ids)) == len(ids) == len(list(wrapper.parameters()))
    x = torch.randn(3, 6, 8)
    y = modulate_visible_condition(x, torch.ones(3, 8), torch.ones(3, 8), args["time_masking"])
    torch.testing.assert_close(y[:, :4], x[:, :4], atol=0, rtol=0)


class VisibleBackbone(_FakeZtautauBackbone):
    def _raw_sequential_feature_names(self):
        return spec()["feature_names"]


def classifier_factory(enabled, kinematics=False, num_layers=2):
    torch.manual_seed(21)
    template = VisibleBackbone()
    template.network_cfg.VisibleConditioning = dict(classifier_enabled=enabled, width=12)
    if kinematics:
        template.network_cfg.VisibleConditioning.update(feature_mode="kinematics", numerical_features=["energy"])
    def build(packing):
        return EvenetAdapterRatioClassifier(copy.deepcopy(template), packing,
            periodic_pair_features=True, topology_fourier_embedding=True,
            topology_max_harmonic=4, topology_hidden_dim=16, topology_embedding_dim=8,
            topology_fusion_hidden_dim=16, topology_dropout=0.,
            decoder_hidden_dim=8, decoder_layers=num_layers, decoder_heads=2, head_dropout=0.,
            adapter_bottleneck=4)
    return build


@pytest.mark.parametrize("kinematics", [False, True])
def test_classifier_gradients_late_fusion_independent_banks_and_saved_architecture(kinematics):
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    enabled, disabled = classifier_factory(True, kinematics), classifier_factory(False)
    model = enabled(packing)
    assert model.bank.visible_conditioning is not None
    assert hasattr(model.bank, "fusion") and not model.bank.topology_conditioning
    second = enabled(packing)
    assert not ({id(p) for p in model.bank.parameters()} & {id(p) for p in second.bank.parameters()})
    candidates = torch.randn(3, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(6):
        opt.zero_grad()
        logits = model(packed, candidates)
        nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert model.bank.visible_conditioning.encoder[0].weight.grad.norm() > 0
    model.eval()
    payload = model.peft_payload()
    # A new saved classifier restores even if the live config has disabled it.
    restored = EvenetAdapterRatioClassifier.from_peft_payload(payload, model_builder=disabled, device=torch.device("cpu"))
    for name, tensor in payload["state"].items():
        torch.testing.assert_close(tensor, restored.bank.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(packed, candidates), restored(packed, candidates), rtol=1e-6, atol=1e-7)
    assert restored.bank.visible_conditioning is not None
    # Conversely an old frozen classifier never acquires new conditioning.
    old = disabled(packing).eval()
    with torch.no_grad(): old.bank.output.weight.normal_(std=.1)
    old_payload = old.peft_payload()
    legacy = EvenetAdapterRatioClassifier.from_peft_payload(old_payload, model_builder=enabled, device=torch.device("cpu"))
    assert legacy.bank.visible_conditioning is None
    with torch.no_grad():
        torch.testing.assert_close(old(packed, candidates), legacy(packed, candidates), rtol=1e-6, atol=1e-7)
    assert legacy.peft_payload()["classifier_config"] == old_payload["classifier_config"]
    assert legacy.peft_payload()["state"].keys() == old_payload["state"].keys()
    # Attaching a zero branch to a trained classifier is also an exact no-op.
    with torch.no_grad():
        before = old(packed, candidates)
        old.bank.configure_visible_conditioning(spec(3))
        torch.testing.assert_close(old(packed, candidates), before, rtol=0, atol=0)


def test_production_context_wiring_is_visible_pre_pet_not_truth():
    for method, raw_source in ((EveNetModel.forward, "x['x']"), (EveNetModel.predict_diffusion_vector, "cond_x['x']")):
        source = inspect.getsource(method)
        assert f"visible_raw={raw_source}, visible_tokens=input_point_cloud, visible_mask=input_point_cloud_mask" in source
    signature = inspect.signature(VisibleConditioning.forward)
    assert list(signature.parameters) == ["self", "raw", "tokens", "mask", "normalized_raw", "return_tokens"]
    assert signature.parameters["return_tokens"].default is False  # optional visible-only memory, never truth


@pytest.mark.parametrize("kinematics", [False, True])
def test_production_supervised_and_sampling_paths_use_same_observed_context(kinematics):
    from evenet.network.body.test_fourier_integration import SmallDiffusion
    from evenet.network.body.normalizer import Normalizer
    class Identity(nn.Module):
        def forward(self, x, **kwargs):
            return x
    class Global(nn.Linear):
        def forward(self, x, **kwargs):
            return super().forward(x)
    model = EveNetModel.__new__(EveNetModel)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.include_global_generation = model.include_point_cloud_generation = False
    model.include_neutrino_generation = True
    model.neutrino_position_encode = False
    model.invisible_input_dim = 4
    model.local_feature_indices = [0, 1]
    model.GroupedSequentialEmbedding = None
    model.InvisibleInputProjector = Identity()
    model.invisible_padding = 0
    model.sequential_normalizer = Identity()
    model.global_normalizer = Identity()
    if kinematics:
        class Project(nn.Module):
            def forward(self, x, mask):
                return 2. * x * mask
        model.GroupedSequentialEmbedding = Project()
        model.sequential_normalizer = Normalizer(torch.tensor([3., 0., 0., 0.]),
            torch.tensor([2., 1., 1., 1.]), torch.tensor([1, 1, 1, 0], dtype=torch.bool))
        model.global_normalizer = Normalizer(torch.tensor([5., -3.]),
            torch.tensor([4., 2.]), torch.ones(2, dtype=torch.bool))
    model.invisible_normalizer = Identity()
    model.GlobalEmbedding = Global(2, 8)
    model.PET = SmallDiffusion("none").PET
    model.TruthGeneration = head(True, kinematics)
    model.eval()
    open_heads(model.TruthGeneration.visible_conditioning)
    context = []
    hook = model.TruthGeneration.visible_conditioning.register_forward_pre_hook(
        lambda _m, args: context.append(tuple(x.detach().clone() for x in args)))
    extra_context = []
    extra_hook = model.TruthGeneration.visible_conditioning.register_forward_pre_hook(
        lambda _m, _args, kwargs: extra_context.append({k: v.detach().clone() for k, v in kwargs.items()}),
        with_kwargs=True)
    batch = dict(x=torch.randn(2, 4, 4), x_mask=torch.ones(2, 4, dtype=torch.bool),
        conditions=torch.randn(2, 2), conditions_mask=torch.ones(2, 1),
        x_invisible=torch.randn(2, 2, 4), x_invisible_mask=torch.ones(2, 2, dtype=torch.bool))
    time, noise, mask = torch.full((2,), .4), torch.randn(2, 2, 4), torch.ones(2, 2, 1)
    a = model.predict_diffusion_vector(noise, batch, time, "neutrino", mask)
    batch["x_invisible"] = torch.full((2, 2, 4), float("nan"))
    b = model.predict_diffusion_vector(noise, batch, time, "neutrino", mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    batch["x_invisible"] = torch.randn(2, 2, 4)
    pred = model(batch, time, schedules=[("neutrino_generation", True)])["generations"]["neutrino"]["vector"]
    assert torch.isfinite(pred).all() and pred.shape == (2, 2, 4)
    pred.square().mean().backward()
    assert model.TruthGeneration.visible_conditioning.encoder[0].weight.grad.norm() > 0
    for captured in context[1:]:
        for old, new in zip(context[0], captured):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
    if kinematics:
        for captured in extra_context:
            torch.testing.assert_close(captured["normalized_raw"][..., 0], (batch["x"][..., 0] - 3.) / 2.)
            torch.testing.assert_close(context[0][1], captured["normalized_raw"] * 2.)
        for captured in extra_context[1:]:
            for key, value in captured.items():
                torch.testing.assert_close(value, extra_context[0][key], rtol=0, atol=0)
    hook.remove()
    extra_hook.remove()


@pytest.mark.parametrize("kinematics", [False, True])
def test_frozen_two_fold_two_iteration_reward_payload_and_candidate_independence(kinematics):
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    build = classifier_factory(True, kinematics)
    members = tuple(build(packing).eval() for _ in range(4))
    for member in members:
        open_heads(member.bank.visible_conditioning)
        with torch.no_grad():
            member.bank.output.weight.normal_(std=.1)
            for block in member.bank.decoder.blocks:
                block.modulation.proj.bias.normal_(std=.1)
    candidate = torch.randn(3, 4)
    contexts = []
    hook = members[0].bank.visible_conditioning.register_forward_hook(
        lambda _m, _x, out: contexts.append(torch.cat(out[0], dim=-1).detach().clone()))
    members[0](packed, candidate)
    members[0](packed, candidate + 3.)
    torch.testing.assert_close(contexts[0], contexts[1], rtol=0, atol=0)
    hook.remove()
    reward = FrozenResidualRatioReward(members, checkpoint_coefficients=(.5,) * 4,
        checkpoint_iterations=(1, 1, 2, 2), tempering=.75)
    payload = reward.serializable_payload()
    restored = FrozenResidualRatioReward.from_serializable_payload(payload, model_builder=classifier_factory(False), device=torch.device("cpu"))
    torch.testing.assert_close(reward(packed, candidate), restored(packed, candidate), rtol=1e-6, atol=1e-7)
    assert all(not p.requires_grad for p in restored.parameters())


@pytest.mark.parametrize("kinematics", [False, True])
def test_real_classifier_fit_optimizer_groups_and_cheap_progress(kinematics):
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig, fit_density_ratio
    packed, packing = pack_event_inputs(_event_batch_with_pair_context(), include_pairwise_context=True)
    model = classifier_factory(True, kinematics)(packing)
    rows, states = [], []
    cfg = RatioFitConfig(steps=6, min_steps=0, batch_size=3, learning_rate=2e-4,
        backbone_learning_rate=1e-5, decoder_learning_rate=5e-5, adapter_learning_rate=5e-5,
        progress_interval_steps=1, diagnostic_enabled=False, representation_diagnostic_enabled=False)
    fit_density_ratio(model, packed, torch.randn(3, 4), torch.ones(3),
        packed, torch.randn(3, 4), torch.ones(3), cfg, seed=11, progress_callback=rows.append,
        checkpoint_callback=states.append)
    assert rows and rows[-1]["optimizer_group_lr_visible_conditioning"] == 2e-4
    assert rows[-1]["optimizer_group_lr_decoder"] == 5e-5
    assert rows[-1]["visible_conditioning/encoder_grad_norm_post_clip"] > 0
    assert rows[-1]["visible_conditioning/modulations_grad_norm_post_clip"] > 0
    assert rows[-1]["visible_conditioning/scale_rms"] > 0
    if kinematics:
        assert rows[-1]["visible_conditioning/numerical_input_rms"] > 0
    else:
        assert "visible_conditioning/numerical_input_rms" not in rows[-1]


def test_wandb_keeps_lightweight_branch_diagnostics_in_production_callback():
    from RL.DGPO_neutrino import dgpo_trainer
    from RL.DGPO_neutrino.test_dgpo_trainer import TestWandbClocks
    wb = mock.Mock()
    dgpo_trainer._wandb_reset_step_tracker()
    with mock.patch.object(dgpo_trainer, "_dgpo_wandb_yaml_section", return_value=({"profile": "critical"}, "logger.wandb")):
        logger = TestWandbClocks._live_progress_logger(wb)
        logger("residual_reward", {"step": 1, "visible_conditioning/scale_rms": .01,
            "visible_conditioning/encoder_grad_norm_post_clip": .02,
            "visible_conditioning/numerical_input_rms": .5,
            "optimizer_group_lr_visible_conditioning": 2e-4}, epoch_value=0)
    row = wb.log.call_args.args[0]
    assert row["omnifold_live/residual_reward/visible_conditioning/scale_rms"] == .01
    assert row["omnifold_live/residual_reward/visible_conditioning/numerical_input_rms"] == .5
    for key, value in row.items():
        if "visible_conditioning" in key:
            assert dgpo_trainer._wandb_critical_keep(key)
            assert dgpo_trainer._wandb_simplified_keep(key, value)
    assert dgpo_trainer._wandb_critical_keep("train/visible_conditioning/encoder_grad_norm_post_clip")


def test_experiment_retains_raw_weights_budget_and_sixteen_gpu_config():
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "scripts"))
    try:
        from train_neutrino_backend import read_yaml, read_overlay_yaml, deep_update
        resolved = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
                               read_overlay_yaml(root / "config/dgpo_h4_visible_adaln.yaml"))
    finally:
        sys.path.pop(0)
    assert resolved["platform"]["number_of_workers"] == 16
    network = resolved["network"]
    assert network["VisibleConditioning"]["diffusion_enabled"]
    assert network["VisibleConditioning"]["classifier_enabled"]
    assert network["Body"]["PET"]["visible_angular_fourier"]["enabled"]
    training = resolved["options"]["Training"]
    assert "diffusion_angular_10pct_lr5e4/checkpoints/last.ckpt" in training["model_checkpoint_load_path"]
    assert not training["EMA"]["enable"] and not training["EMA"]["replace_model_after_load"]
    dgpo = resolved["dgpo"]
    assert dgpo["checkpoint_load_mode"] == "weights_only" and not dgpo["auto_resume_from_last"]
    assert dgpo["reference_trust"]["objective"] == "velocity_mse"
    assert dgpo["reference_trust"]["coefficient"] == 1 and not dgpo["endpoint_kl"]["enabled"]
    adaptive = dgpo["adaptive_omnifold"]
    assert adaptive["recalibration"]["min_iterations"] == adaptive["recalibration"]["max_iterations"] == 2
    assert adaptive["trigger"]["max_reward_age_epochs"] == 20
    from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
    for block in (adaptive["audit_fit"], adaptive["recalibration"]["fit"]):
        fit = build_fit_config(block, n_train=20000, n_validation=2000)
        fit.validate()
        assert not fit.diagnostic_enabled and not fit.representation_diagnostic_enabled
        assert fit.min_steps == 0 and fit.steps is None and fit.restore_best
