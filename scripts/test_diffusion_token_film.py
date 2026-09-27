"""Supervised conditioning screen: same source/function, data, noise and budget."""
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from train_neutrino_backend import deep_update, read_overlay_yaml, read_yaml
from evenet.network.body.test_visible_conditioning import head, head_inputs
from evenet.network.body.test_token_conditioning import token_head
from evenet.utilities.fourier_integration import load_conditioning_ablation_weights

ROOT = Path(__file__).resolve().parents[1]


def config(arm):
    return deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                       read_overlay_yaml(ROOT / f"config/train_diffusion_token_film_{arm}.yaml"))


def test_supervised_arms_match_and_no_rl_or_extra_physics_weighting():
    configs = [config(a) for a in ("pet_only", "global", "last")]
    for c in configs:
        t, p = c["options"]["Training"], c["platform"]
        assert t["strict_conditioning_ablation_source"] and not t["strict_no_fourier_source"]
        assert t["pretrain_model_load_path"].endswith("dgpo-epoch=-1-next_ep=0-step=0.ckpt")
        assert t["model_checkpoint_load_path"] is None
        assert not t["EMA"]["enable"] and not t["EMA"]["replace_model_after_load"]
        assert not t["JointCoverage"]["enable"]
        assert t["Components"]["TruthGeneration"]["low_noise_weight"] == 1
        assert t["epochs"] == t["total_epochs"] == 50 < t["EarlyStopping"]["patience"]
        assert t["paired_diffusion_seed"] == 420024
        assert t["freeze_modules"] == []
        assert not t["scale_lr_with_world_size"]
        assert p["number_of_workers"] == 16 and p["batch_size"] == 2048 and p["ordered_data"]
        assert p["resources_per_worker"]["GPU"] == 1 and p["use_gpu"] is True
        assert c["nersc"]["nodes"] * c["nersc"]["gpus_per_node"] == 16
        assert not c["network"]["VisibleConditioning"]["classifier_enabled"]
        assert c["network"]["TruthGeneration"]["num_layers"] == 3
        assert "--backend pure-evenet" in c["nersc"]["execution"]["command"]
    for c in configs[1:]:
        assert c["platform"] == configs[0]["platform"]
        t = copy.deepcopy(c["options"]["Training"])
        t["model_checkpoint_save_path"] = configs[0]["options"]["Training"]["model_checkpoint_save_path"]
        assert t == configs[0]["options"]["Training"]


def wrap(module):
    model = nn.Module(); model.TruthGeneration = module
    return model


def test_strict_raw_loader_preserves_initial_function_for_all_three_arms():
    source = wrap(head(True, True)).eval()
    checkpoint = {"state_dict": {"model." + k: v for k, v in source.state_dict().items()}}
    args = head_inputs(True)
    for arm in (wrap(head(False)), wrap(head(True, True)), wrap(token_head())):
        load_conditioning_ablation_weights(arm, checkpoint)
        arm.eval()
        with torch.no_grad():
            torch.testing.assert_close(source.TruthGeneration(**args), arm.TruthGeneration(**args), rtol=0, atol=0)
    broken = copy.deepcopy(checkpoint)
    broken["state_dict"]["model.TruthGeneration.visible_conditioning.modulations.0.bias"].fill_(.1)
    with pytest.raises(ValueError, match="zero-output"):
        load_conditioning_ablation_weights(wrap(head(False)), broken)
    broken = copy.deepcopy(checkpoint)
    broken["state_dict"].pop("model.TruthGeneration.generator.weight")
    with pytest.raises(ValueError, match="missing"):
        load_conditioning_ablation_weights(wrap(token_head()), broken)


@pytest.mark.parametrize("arm", ["pet_only", "global", "last"])
def test_supervised_real_optimizer_separates_conditioning_once(arm):
    import logging
    import lightning as L
    from evenet.engine import EveNetEngine
    from evenet.control.global_config import DotDict
    c = config(arm)
    engine = EveNetEngine.__new__(EveNetEngine)
    L.LightningModule.__init__(engine)
    engine.model = wrap(token_head() if arm == "last" else head(arm == "global", True))
    engine.l = logging.getLogger("token-film-test")
    engine.world_size = 16
    engine.total_events = engine.total_val_events = 327680
    engine.hyper_par_cfg = dict(batch_size=2048, epoch=50, warm_up_factor=5.)
    engine.config = DotDict(dict(options=dict(Training=dict(scale_lr_with_world_size=False,
        ProgressiveTraining=dict(stages=[dict(name="diffusion", epoch_ratio=1.,
            transition_ratio=0., loss_weights={"generation-truth": [1., 1.]})])))))
    engine.include_famo = False
    parts = {"generation": dict(modules=["TruthGeneration"], lr=1e-4, weight_decay=.001,
                               warm_up=True, optimizer_type="adamW", decoupled_wd=True)}
    if arm != "pet_only":
        parts["visible_conditioning"] = dict(parts["generation"], modules=["TruthGeneration.visible_conditioning"])
    engine.model_parts = parts
    optimizers, schedulers = engine.configure_optimizers()
    ids = [id(p) for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(engine.model.parameters()))
    assert len(schedulers) == len(parts)
