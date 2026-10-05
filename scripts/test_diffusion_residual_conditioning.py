"""Validate new supervised arms without launching training or accessing NERSC."""
import copy
import logging
from pathlib import Path

import pytest
import lightning as L

from train_neutrino_backend import deep_update, read_yaml, read_overlay_yaml
from evenet.engine import EveNetEngine
from evenet.control.global_config import DotDict
from evenet.network.body.test_residual_conditioning import residual_head, velocity_head, wrap

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                       read_overlay_yaml(ROOT / "config" / name))


@pytest.mark.parametrize("block", ["last", "all"])
def test_new_arms_only_change_routing_and_names_not_supervised_training(block):
    baseline = config("train_diffusion_token_film_global.yaml")
    current = config(f"train_diffusion_residual_{block}.yaml")
    assert current["platform"] == baseline["platform"]
    assert current["options"]["Dataset"] == baseline["options"]["Dataset"]
    training = copy.deepcopy(current["options"]["Training"])
    training["model_checkpoint_save_path"] = baseline["options"]["Training"]["model_checkpoint_save_path"]
    assert training == baseline["options"]["Training"]
    network = copy.deepcopy(current["network"])
    routing = network["VisibleConditioning"]["diffusion_token_readout"]
    assert routing == dict(enabled=True, mode="residual", block=block, width=64, heads=4)
    network["VisibleConditioning"]["diffusion_token_readout"] = baseline["network"]["VisibleConditioning"]["diffusion_token_readout"]
    assert network == baseline["network"]
    assert current["platform"]["number_of_workers"] == 16
    assert current["platform"]["batch_size"] == 2048
    assert not current["rl"]["enabled"] and not training["EMA"]["enable"]
    assert training["precision"] == "32-true"
    assert training["epochs"] == training["total_epochs"] == 50
    assert training["model_checkpoint_load_path"] is None
    assert training["pretrain_model_load_path"].endswith("dgpo-epoch=-1-next_ep=0-step=0.ckpt")
    assert current["logger"]["wandb"]["id"] == f"diffres{block}1"
    assert current["logger"]["wandb"]["project"] == "EveNet"
    assert len(current["logger"]["wandb"]["run_name"]) <= 96
    assert f"--overlay-config config/train_diffusion_residual_{block}.yaml" in current["nersc"]["execution"]["command"]
    assert "--backend pure-evenet" in current["nersc"]["execution"]["command"]


@pytest.mark.parametrize("block", ["last", "all"])
def test_actual_optimizer_owns_new_branch_once_with_same_schedule(block):
    engine = EveNetEngine.__new__(EveNetEngine)
    L.LightningModule.__init__(engine)
    engine.model = wrap(residual_head(block))
    engine.l = logging.getLogger("residual-test")
    engine.world_size = 16
    engine.total_events = engine.total_val_events = 327680
    engine.hyper_par_cfg = dict(batch_size=2048, epoch=50, warm_up_factor=5.)
    engine.config = DotDict(dict(options=dict(Training=dict(scale_lr_with_world_size=False,
        ProgressiveTraining=dict(stages=[dict(name="diffusion", epoch_ratio=1.,
            transition_ratio=0., loss_weights={"generation-truth": [1., 1.]})])))))
    engine.include_famo = False
    settings = dict(lr=1e-4, weight_decay=.001, warm_up=True, optimizer_type="adamW", decoupled_wd=True)
    engine.model_parts = {"generation": dict(settings, modules=["TruthGeneration"]),
                          "visible_conditioning": dict(settings, modules=["TruthGeneration.visible_conditioning"])}
    optimizers, schedulers = engine.configure_optimizers()
    ids = [id(p) for opt in optimizers for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(engine.model.parameters()))
    assert len(schedulers) == 2
    branch_ids = {id(p) for p in engine.model.TruthGeneration.visible_conditioning.token_readout.parameters()}
    containing = [opt for opt in optimizers if branch_ids.intersection(
        id(p) for group in opt.param_groups for p in group["params"])]
    assert len(containing) == 1
    assert branch_ids <= {id(p) for group in containing[0].param_groups for p in group["params"]}


def test_velocity_probe_config_is_a_frozen_matched_16_gpu_screen():
    baseline = config("train_diffusion_token_film_global.yaml")
    current = config("train_diffusion_velocity_residual_probe.yaml")
    assert current["platform"] == baseline["platform"]
    assert current["options"]["Dataset"] == baseline["options"]["Dataset"]
    training = current["options"]["Training"]
    readout = current["network"]["VisibleConditioning"]["diffusion_token_readout"]
    assert readout == dict(enabled=True, mode="velocity", block="output", width=64, heads=4)
    assert training["train_only_modules"] == ["TruthGeneration.visible_conditioning.token_readout"]
    assert training["epochs"] == training["total_epochs"] == 20
    assert training["learning_rate_warm_up_factor"] == 2.0
    assert training["precision"] == "32-true"
    assert training["EarlyStopping"]["patience"] > training["epochs"]
    assert training["pretrain_model_load_path"].endswith("dgpo-epoch=-1-next_ep=0-step=0.ckpt")
    assert training["model_checkpoint_load_path"] is None
    assert not training["EMA"]["enable"] and not current["rl"]["enabled"]
    branch = training["Components"]["TruthGeneration.visible_conditioning"]
    assert branch["learning_rate"] == 5e-4 and branch["optimizer_type"] == "adamW"
    assert current["platform"]["number_of_workers"] == 16
    assert current["platform"]["batch_size"] == 2048
    assert current["logger"]["wandb"]["id"] == "diffvelres1"
    assert current["logger"]["wandb"]["group"] == "Diffusion conditioning architecture"
    assert "--backend pure-evenet" in current["nersc"]["execution"]["command"]


def test_velocity_probe_optimizer_and_train_mode_are_branch_only():
    engine = EveNetEngine.__new__(EveNetEngine)
    L.LightningModule.__init__(engine)
    engine.model = wrap(velocity_head())
    branch = engine.model.TruthGeneration.visible_conditioning.token_readout
    engine.model.requires_grad_(False)
    branch.requires_grad_(True)
    engine.l = logging.getLogger("velocity-probe-test")
    engine.world_size = 16
    engine.total_events = engine.total_val_events = 327680
    engine.hyper_par_cfg = dict(batch_size=2048, epoch=20, warm_up_factor=2.)
    engine.config = DotDict(dict(options=dict(Training=dict(
        train_only_modules=["TruthGeneration.visible_conditioning.token_readout"],
        scale_lr_with_world_size=False,
        ProgressiveTraining=dict(stages=[dict(name="diffusion", epoch_ratio=1.,
            transition_ratio=0., loss_weights={"generation-truth": [1., 1.]})])))))
    engine.include_famo = False
    engine.eval_metrics_every_n_epochs = 1
    settings = dict(lr=5e-4, weight_decay=.001, warm_up=True,
                    optimizer_type="adamW", decoupled_wd=True)
    engine.model_parts = {
        "generation": dict(settings, modules=["TruthGeneration"]),
        "visible_conditioning": dict(settings,
            modules=["TruthGeneration.visible_conditioning"]),
    }
    optimizers, schedulers = engine.configure_optimizers()
    params = [p for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    assert {id(p) for p in params} == {id(p) for p in branch.parameters()}
    assert len(optimizers) == len(schedulers) == 1
    engine.model.train()
    engine.on_train_epoch_start()
    assert not engine.model.TruthGeneration.training
    assert branch.training
