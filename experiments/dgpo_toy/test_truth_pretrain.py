"""Full observed targets, finite dataset, loss-only stopping and exact resume."""
from dataclasses import replace

import pytest
import torch

from experiments.dgpo_toy.conditional import Config, Distribution, Denoiser, alpha_sigma, generator
from experiments.dgpo_toy.test_fixed_reward import assert_nested_equal
from experiments.dgpo_toy.truth_pretrain import (prepare_dataset, noisy_target, sample_loss, validation,
                             make_panel, early_stop_update, run)


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def small():
    return replace(Config(), dimensions=6, hidden=12, train_events=16,
        validation_events=16, test_events=16, fit_batch=8, eval_events=8)


def test_fixed_dataset_is_truth_split_and_reload_does_not_resample(tmp_path, monkeypatch):
    calls = []
    original = Distribution.sample
    def record(self, c, rng, *, truth):
        calls.append((len(c), truth))
        return original(self, c, rng, truth=truth)
    monkeypatch.setattr(Distribution, "sample", record)
    cfg = small()
    path = tmp_path/"data.pt"
    first = prepare_dataset(path, cfg, 17)
    second = prepare_dataset(path, cfg, 17)
    assert calls == [(16, True)]*3
    assert_nested_equal(first, second)
    assert not torch.equal(first["train"]["condition"], first["validation"]["condition"])
    assert not torch.equal(first["test"]["target"], first["validation"]["target"])
    with pytest.raises(ValueError, match="configuration"):
        prepare_dataset(path, cfg, 18)


def test_velocity_target_is_observed_sample_not_teacher():
    y = torch.randn(20, 6, generator=generator(5))
    x, t, v = noisy_target(y, generator(6))
    a, s = alpha_sigma(t[:, None])
    torch.testing.assert_close(a*x-s*v, y)
    cfg = small()
    model = Denoiser(cfg)
    c = torch.zeros(20, cfg.context_dim)
    loss = sample_loss(model, (c, x, t, v))
    assert torch.equal(loss, (model(x, t, c)-v).square().mean())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_validation_reproducible_rng_independent_and_all_time_bins(tmp_path):
    cfg = small()
    dataset = prepare_dataset(tmp_path/"data.pt", cfg, 17)
    panel = make_panel(dataset["validation"], 21017)
    model = Denoiser(cfg)
    state = torch.random.get_rng_state().clone()
    a, b = validation(model, panel), validation(model, panel)
    assert a == b and torch.equal(state, torch.random.get_rng_state())
    assert sum(x["count"] for x in a["time_bins"]) == cfg.validation_events
    weighted = sum(x["count"]*x["mse"] for x in a["time_bins"] if x["count"])
    assert weighted/cfg.validation_events == pytest.approx(a["velocity_mse"])


def test_early_stopping_significance_and_accumulation():
    assert early_stop_update(.99995, 1., 0, 1e-4) == (1., 1)
    assert early_stop_update(.9998, 1., 1, 1e-4) == (.9998, 0)
    assert early_stop_update(1.2, 1., 2, 1e-4) == (1., 3)


def test_train_stops_without_hard_budget_and_never_uses_teacher(tmp_path, monkeypatch):
    import experiments.dgpo_toy.conditional as core
    monkeypatch.setattr(core, "pretrain", lambda *a, **k: pytest.fail("Gaussian teacher is forbidden"))
    monkeypatch.setattr(Distribution, "nominal_log_ratio", lambda *a: pytest.fail("Oracle reward is forbidden"))
    cfg = small()
    data = tmp_path/"dataset.pt"
    prepare_dataset(data, cfg, 17)
    before = data.read_bytes()
    monkeypatch.setattr(Distribution, "sample", lambda *a, **k: pytest.fail("No new targets after dataset preparation"))
    result = run(tmp_path/"run", dataset_path=data, cfg=cfg, patience=2, min_delta=100., structure_every=2)
    assert result["state"] == "completed" and result["stop_reason"] == "validation_early_stopping"
    assert result["completed_epochs"] == 3 and result["completed_steps"] == 6
    assert result["max_steps"] is None and result["max_epochs"] is None and result["teacher"] is None
    rows = [x for x in result["history"] if x["phase"] == "epoch"]
    assert result["best_validation_mse"] == min(x["validation"]["velocity_mse"] for x in rows)
    assert data.read_bytes() == before
    selected = torch.load(tmp_path/"run/best_model.pt", weights_only=True)
    assert selected["weights"] == "raw" and selected["step"] == result["best_step"]


def test_epoch_resume_preserves_model_optimizer_rng_and_early_stop(tmp_path):
    cfg = small()
    kw = dict(dataset_path=tmp_path/"data.pt", cfg=cfg, patience=2, min_delta=100., structure_every=2)
    run(tmp_path/"full", **kw)
    paused = run(tmp_path/"part", stop_after_epoch=1, **kw)
    assert paused["state"] == "paused"
    run(tmp_path/"resume", resume_from=tmp_path/"part/last_state.pt", **kw)
    full = torch.load(tmp_path/"full/last_state.pt", weights_only=True)
    resumed = torch.load(tmp_path/"resume/last_state.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "initial", "best_model", "best_loss", "anchor", "stale_epochs", "step", "epoch"):
        assert_nested_equal(full[key], resumed[key])
    assert full["report"]["history"] == resumed["report"]["history"]
    assert full["report"]["endpoints"] == resumed["report"]["endpoints"]
    assert all(int(x["step"]) == 6 for x in full["optimizer"]["state"].values())


@pytest.mark.parametrize("kwargs", [{"patience": 0}, {"min_delta": -1}, {"min_delta": float("nan")}, {"structure_every": 0}])
def test_invalid_stopping_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run(tmp_path/"run", dataset_path=tmp_path/"data.pt", cfg=small(), **kwargs)
