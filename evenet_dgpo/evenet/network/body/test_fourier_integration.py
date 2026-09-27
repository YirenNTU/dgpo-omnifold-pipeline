"""Small CPU integration checks using the production PET and generation head."""
import copy

import pytest
import torch
from torch import nn

from evenet.network.body.angular_conditioning import VisibleAngularFourier
from evenet.network.body.embedding import PETBody
from evenet.network.heads.generation.generation_head import EventGenerationHead
from evenet.utilities.fourier_integration import (
    load_no_fourier_weights, optimizer_parameters, paired_diffusion_rng,
)


class SmallDiffusion(nn.Module):
    def __init__(self, arm):
        super().__init__()
        branch = None if arm == "none" else VisibleAngularFourier(
            ["energy", "pt", "eta", "phi"], 8, "eta", "phi",
            placement="output" if arm in ("output", "readout") else "input",
            projection_type=("cross_attention" if arm == "readout" else
                             "mlp" if arm in ("output", "input_mlp") else "linear"),
            mlp_dim=12, log_diagnostics=True, harmonics=[1, 2, 4, 8] if arm == "readout" else None)
        self.PET = PETBody(num_feat=4, num_keep=4, feature_drop=0., projection_dim=8,
            local=False, K=1, num_local=0, num_layers=2, num_heads=2,
            drop_probability=0., talking_head=False, layer_scale=False,
            layer_scale_init=1., dropout=.1, mode="all", angular_conditioning=branch)
        self.TruthGeneration = EventGenerationHead(input_dim=8, projection_dim=8,
            num_global_cond=8, num_classes=2, output_dim=4, num_layers=1,
            num_heads=2, dropout=.1, layer_scale=False, layer_scale_init=1.,
            drop_probability=0., feature_drop=0.)

    def forward(self, x):
        # 3 visible (one padded) + 1 invisible token; same mask as production.
        mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool).view(1, 4, 1).expand(x.shape[0], -1, -1)
        attn = torch.zeros(4, 4, dtype=torch.bool)
        attn[:3, 3:] = True
        time = torch.full((x.shape[0],), .4)
        tmask = torch.tensor([0, 0, 0, 1]).view(1, 4, 1).expand(x.shape[0], -1, -1)
        h = self.PET(x, x, mask, time, attn_mask=attn, time_masking=tmask, visible_raw=x[:, :3])
        pred = self.TruthGeneration(h, torch.ones(x.shape[0], 1, 8),
            torch.ones(x.shape[0], 1, 1), None, mask, time,
            torch.zeros(x.shape[0], 1, dtype=torch.long), attn_mask=attn, time_masking=tmask)
        return h, pred[:, 3:]


@pytest.mark.parametrize("arm", ["input", "output", "input_mlp", "readout"])
@pytest.mark.parametrize("training", [False, True])
def test_raw_load_initial_prediction_and_gradient_equivalence(arm, training):
    torch.manual_seed(17)
    base, new = SmallDiffusion("none"), SmallDiffusion(arm)
    checkpoint = {"state_dict": {"model." + k: v for k, v in base.state_dict().items()}}
    load_no_fourier_weights(new, checkpoint)
    base.train(training)
    new.train(training)
    x = torch.randn(3, 4, 4)
    outputs = []
    for model in (base, new):
        with paired_diffusion_rng(42, training=training, epoch=0, batch_idx=0, rank=0, device="cpu"):
            outputs.append(model(x)[1])
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    outputs[1].square().mean().backward()
    branch = new.PET.angular_conditioning
    assert branch.projection.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None)
    assert branch.diagnostics["residual_to_base_rms"] == 0


def test_late_branch_bypasses_pet_blocks_but_reaches_prediction_and_backbone_gradient():
    torch.manual_seed(8)
    base, late = SmallDiffusion("none"), SmallDiffusion("output")
    load_no_fourier_weights(late, {"state_dict": base.state_dict()})
    base.eval(); late.eval()
    with torch.no_grad():
        late.PET.angular_conditioning.projection.weight.normal_(std=.1)
    x = torch.randn(3, 4, 4)
    captures = []
    handles = [m.PET.transformer_blocks[-1].register_forward_hook(
        lambda _m, _a, y: captures.append(y.detach().clone())) for m in (base, late)]
    h0, p0 = base(x)
    h1, p1 = late(x)
    for handle in handles: handle.remove()
    torch.testing.assert_close(captures[0], captures[1], rtol=0, atol=0)
    torch.testing.assert_close(h0[:, 3:], h1[:, 3:], rtol=0, atol=0)
    torch.testing.assert_close(h0[:, 2], h1[:, 2], rtol=0, atol=0)
    assert not torch.allclose(h0[:, :2], h1[:, :2])
    assert not torch.allclose(p0, p1)
    p1.square().mean().backward()
    assert late.PET.feature_embedding[0].weight.grad.norm() > 0
    assert late.PET.angular_conditioning.encoder[0].weight.grad.norm() > 0


def test_adapter_masks_nan_padding_after_learned_bias_and_is_periodic():
    b = VisibleAngularFourier(["eta", "phi"], 8, "eta", "phi",
                              placement="output", projection_type="mlp")
    with torch.no_grad(): b.projection.weight.normal_()
    x = torch.tensor([[[1., .3], [float("nan"), float("nan")]]])
    mask = torch.tensor([[True, False]])
    y = b(x, mask)
    assert torch.isfinite(y).all() and y[:, 1].count_nonzero() == 0
    x[..., 1] += 2 * torch.pi
    torch.testing.assert_close(b(x, mask), y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("arm", ["output", "input_mlp", "readout"])
def test_separate_optimizer_ownership_and_two_updates(arm):
    m = SmallDiffusion(arm)
    paths = ["PET", "TruthGeneration", "PET.angular_conditioning"]
    groups = [optimizer_parameters(m, [p], paths) for p in paths]
    ids = [id(p) for group in groups for p in group]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in m.parameters()}
    opts = [torch.optim.AdamW(g, lr=1e-3) for g in groups]
    x = torch.randn(3, 4, 4)
    for _ in range(2):
        for opt in opts: opt.zero_grad()
        m(x)[1].square().mean().backward()
        for opt in opts: opt.step()
    assert m.PET.angular_conditioning.encoder[0].weight.grad.norm() > 0
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_source_loader_rejects_partial_wrong_shape_or_already_fourier():
    base = SmallDiffusion("none")
    state = copy.deepcopy(base.state_dict())
    del state["PET.feature_embedding.0.weight"]
    with pytest.raises(ValueError, match="missing"):
        load_no_fourier_weights(SmallDiffusion("output"), {"state_dict": state})
    state = copy.deepcopy(base.state_dict())
    state["PET.feature_embedding.0.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape_mismatch"):
        load_no_fourier_weights(SmallDiffusion("output"), {"state_dict": state})
    with pytest.raises(ValueError, match="no-Fourier"):
        load_no_fourier_weights(base, {"state_dict": SmallDiffusion("input").state_dict()})


def test_paired_rng_isolation_validation_fixed_and_train_changes():
    def draw(training, epoch):
        with paired_diffusion_rng(42, training=training, epoch=epoch, batch_idx=2, rank=0, device="cpu"):
            return torch.randn(8)
    state = torch.random.get_rng_state().clone()
    torch.testing.assert_close(draw(False, 0), draw(False, 10), rtol=0, atol=0)
    assert not torch.equal(draw(True, 0), draw(True, 1))
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)


def test_engine_optimizers_keep_backbone_and_branch_trainable_without_overlap():
    import logging
    import lightning as L
    from evenet.control.global_config import DotDict
    from evenet.engine import EveNetEngine
    engine = EveNetEngine.__new__(EveNetEngine)
    L.LightningModule.__init__(engine)
    engine.model = SmallDiffusion("output")
    engine.l = logging.getLogger("fourier-test")
    engine.world_size = 1
    engine.total_events = engine.total_val_events = 8
    engine.hyper_par_cfg = dict(batch_size=4, epoch=2, warm_up_factor=1)
    engine.config = DotDict(dict(options=dict(Training=dict(scale_lr_with_world_size=False,
        ProgressiveTraining=dict(stages=[dict(name="diffusion", epoch_ratio=1.,
            transition_ratio=0., loss_weights={"generation-truth": [1., 1.]})])))))
    engine.include_famo = False
    engine.model_parts = {name: dict(modules=[path], lr=lr, weight_decay=.001,
        warm_up=True, optimizer_type="adamW", decoupled_wd=True)
        for name, path, lr in [("body", "PET", 2e-5), ("generation", "TruthGeneration", 1e-4),
                                ("fourier", "PET.angular_conditioning", 2e-5)]}
    optimizers, schedulers = engine.configure_optimizers()
    ids = [id(p) for opt in optimizers for group in opt.param_groups for p in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in engine.model.parameters()}
    assert all(p.requires_grad for p in engine.model.parameters())
    assert [s['name'] for s in schedulers] == ['lr-body', 'lr-generation', 'lr-fourier']
