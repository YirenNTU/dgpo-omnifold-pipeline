"""Classifier-local EveNet schedule: fit clocks, recovery and diagnostics."""
import copy
from dataclasses import replace

import pytest
import torch

from .ratio_fit import RatioFitConfig, classifier_lr_multiplier, fit_density_ratio
from .stage import build_fit_config
from .test_stability_diagnostic import TinyClassifier, population, config


def test_half_cosine_warmup_floor_and_no_restart():
    cfg = RatioFitConfig(lr_scheduler="cosine", lr_warmup_epochs=1,
                         lr_cosine_epochs=3, lr_min_ratio=.1)
    assert [classifier_lr_multiplier(cfg, s, 10) for s in (0, 5, 10)] == [0, .5, 1]
    assert classifier_lr_multiplier(cfg, 20, 10) == pytest.approx(.55)
    assert classifier_lr_multiplier(cfg, 30, 10) == pytest.approx(.1)
    assert classifier_lr_multiplier(cfg, 100, 10) == pytest.approx(.1)


def test_zero_floor_matches_evenet_huggingface_schedule():
    from transformers import get_cosine_schedule_with_warmup
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01)
    sch = get_cosine_schedule_with_warmup(opt, 10, 30, num_cycles=.5)
    cfg = RatioFitConfig(lr_scheduler="cosine", lr_cosine_epochs=3, lr_min_ratio=0)
    for step in range(31):
        assert opt.param_groups[0]["lr"] == pytest.approx(.01 * classifier_lr_multiplier(cfg, step, 10))
        opt.step()
        sch.step()


@pytest.mark.parametrize("changes", [dict(lr_scheduler="bad"), dict(lr_warmup_epochs=-1),
    dict(lr_cosine_epochs=1), dict(lr_min_ratio=1.1), dict(lr_min_ratio=float("nan"))])
def test_invalid_schedule(changes):
    with pytest.raises(ValueError):
        replace(RatioFitConfig(), **changes).validate()


def test_config_builder_keeps_unbounded_stopping():
    cfg = build_fit_config(dict(steps=None, min_steps=1000, lr_scheduler="cosine",
        lr_warmup_epochs=1, lr_cosine_epochs=20, lr_min_ratio=.1), n_train=16000, n_validation=1000)
    assert cfg.steps is None and cfg.min_steps == 1000
    assert cfg.lr_scheduler == "cosine"


def test_fold_epoch_clock_and_wandb_filters():
    from .evenet_ratio import _scaled_crossfit_config
    from ..dgpo_trainer import _wandb_critical_keep
    cfg = RatioFitConfig(steps=None, batch_size=8, sampling="independent_epoch_shuffle",
                         min_steps=1000, validation_interval_steps=2,
                         validation_patience_evaluations=10, lr_scheduler="cosine")
    folded = _scaled_crossfit_config(cfg, .5, n_train=8, min_epochs_per_fold=1)
    assert folded.lr_cosine_epochs == cfg.lr_cosine_epochs
    assert classifier_lr_multiplier(folded, 1, 1) == 1
    assert classifier_lr_multiplier(cfg, 1, 2) == .5
    for metric in ('scheduler_multiplier', 'scheduler_epoch', 'optimizer_group_lr_adapter'):
        assert _wandb_critical_keep('omnifold_live/raw_staleness_audit/' + metric)


def _scheduler_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://' + rendezvous,
                                         rank=rank, world_size=2)
    try:
        rows = []
        fit_density_ratio(TinyClassifier(), *population(), config(steps=4,
            diagnostic_enabled=False, lr_scheduler='cosine', lr_cosine_epochs=3),
            23, validation=population(), progress_callback=rows.append)
        lrs = [r['optimizer_group_lr_0'] for r in rows]
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, lrs)
        assert gathered[0] == gathered[1]
        assert lrs[:3] == pytest.approx([0, .005, .01])
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_scheduler_uses_global_epoch_clock(tmp_path):
    torch.multiprocessing.spawn(_scheduler_worker, args=(str(tmp_path / 'scheduler-rendezvous'),),
                                nprocs=2, join=True)


def test_fit_resume_and_diagnostics_equivalence(tmp_path):
    torch.manual_seed(4)
    original = TinyClassifier()
    cfg = config(steps=8, lr_scheduler="cosine", lr_warmup_epochs=1,
                 lr_cosine_epochs=3, checkpoint_interval_steps=1,
                 diagnostic_enabled=False, diagnostic_snapshot_dir=str(tmp_path))
    states, rows, rng_states = [], [], {}
    def save(state):
        states.append(copy.deepcopy(state))
        rng_states[state["steps_completed"]] = torch.get_rng_state().clone()
    model = copy.deepcopy(original)
    torch.manual_seed(123)
    fit_density_ratio(model, *population(), cfg, 23, validation=population(),
                      checkpoint_callback=save,
                      progress_callback=rows.append)
    # 16 rows / GLOBAL batch 8 = 2 updates per epoch.
    assert [r["optimizer_group_lr_0"] for r in rows[:3]] == pytest.approx([0, .005, .01])
    assert rows[-1]["optimizer_group_lr_0"] == pytest.approx(.001)
    saved = next(x for x in states if x["steps_completed"] == 4)
    resumed = copy.deepcopy(original)
    # Isolate schedule/optimizer recovery from the caller-owned dropout RNG.
    torch.set_rng_state(rng_states[4])
    fit_density_ratio(resumed, *population(), cfg, 23, validation=population(), resume_state=saved)
    for name, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name
    diagnosed = copy.deepcopy(original)
    torch.manual_seed(123)
    fit_density_ratio(diagnosed, *population(), replace(cfg, diagnostic_enabled=True),
                      23, validation=population())
    for name, value in model.state_dict().items():
        assert torch.equal(value, diagnosed.state_dict()[name]), name
