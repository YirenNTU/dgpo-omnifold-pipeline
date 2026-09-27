"""Measurement, shared supervision, early-stop and exact-resume contract tests."""
from dataclasses import replace
import json

import pytest
import torch

from experiments.dgpo_toy import cube_truth_fourier as runner
from experiments.dgpo_toy.conditional import generator
from experiments.dgpo_toy.parity_cube import CubeDistribution, ModeReward
from experiments.dgpo_toy.test_fixed_reward import assert_nested_equal


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def small():
    return replace(runner.DEFAULT_CONFIG, hidden=8, train_events=16, validation_events=16,
        test_events=16, fit_batch=8, eval_events=512, candidates=2, ddim_steps=2)


def test_fixed_truth_dataset_is_shared_reused_and_not_reference(tmp_path, monkeypatch):
    calls = []
    original = CubeDistribution.sample
    def record(self, c, rng, *, truth=False):
        calls.append(truth)
        return original(self, c, rng, truth=truth)
    monkeypatch.setattr(CubeDistribution, "sample", record)
    a = runner.prepare_truth_dataset(tmp_path / "dataset.pt", small(), 17)
    b = runner.prepare_truth_dataset(tmp_path / "dataset.pt", small(), 17)
    assert calls == [True, True, True]
    assert_nested_equal(a, b)
    assert a["metadata"]["truth"] is True
    assert not torch.equal(a["train"]["condition"], a["validation"]["condition"])
    with pytest.raises(ValueError, match="dataset differs"):
        runner.prepare_truth_dataset(tmp_path / "dataset.pt", small(), 23)


def test_truth_sample_contains_condition_dependent_joint_not_wrong_reference():
    cfg = small(); data = runner.truth_data(cfg); rng = generator(31)
    c = data.contexts(65536, rng); y = data.sample(c, rng, truth=True)
    moment = (data.condition_signal(c[:, 0]) * y.sign().prod(-1)).mean()
    assert .38 < float(moment) < .42
    assert float(y.mean(0).abs().max()) < .02
    assert float((y[:, 0] * y[:, 1]).mean().abs()) < .02


def test_initial_pair_matches_and_fourier_starts_learning_in_first_pretrain_batch(tmp_path):
    cfg = small(); models, matching = runner.initial_pair(cfg, 17)
    assert matching["velocity_exact"] and matching["samples_exact"]
    assert len(set(matching["parameter_count"].values())) == 1
    original = {b: {k: v.clone() for k, v in m.state_dict().items()} for b, m in models.items()}
    dataset = runner.prepare_truth_dataset(tmp_path / "data.pt", cfg, 17)
    opts = {b: torch.optim.AdamW(m.parameters(), lr=cfg.fit_lr) for b, m in models.items()}
    runner.train_epoch(models, opts, runner.BASES, dataset["train"], cfg, generator(19))
    assert models["raw"].condition_adapter.weight.count_nonzero() == 0
    assert models["fourier"].condition_adapter.weight.count_nonzero() > 0
    for b in runner.BASES:
        assert any(not torch.equal(v, original[b][k]) for k, v in models[b].state_dict().items())


def test_both_arms_receive_identical_batches_times_noise_and_targets(tmp_path, monkeypatch):
    cfg = small(); models, _ = runner.initial_pair(cfg, 17)
    dataset = runner.prepare_truth_dataset(tmp_path / "data.pt", cfg, 17)
    seen = {b: [] for b in runner.BASES}; handles = []
    for b, model in models.items():
        handles.append(model.register_forward_pre_hook(
            lambda m, args, name=b: seen[name].append(tuple(x.detach().clone() for x in args))))
    original = runner.noisy_target; draws = []
    def record(y, rng):
        result = original(y, rng); draws.append(result)
        return result
    monkeypatch.setattr(runner, "noisy_target", record)
    monkeypatch.setattr(ModeReward, "forward", lambda *a: pytest.fail("No reward allowed in pretraining"))
    opts = {b: torch.optim.AdamW(m.parameters(), lr=cfg.fit_lr) for b, m in models.items()}
    runner.train_epoch(models, opts, runner.BASES, dataset["train"], cfg, generator(19))
    assert_nested_equal(seen["raw"], seen["fourier"])
    assert len(draws) == len(seen["raw"]) == 2
    for handle in handles:
        handle.remove()


def test_joint_metric_detects_shuffled_condition_despite_preserved_marginals():
    cfg = replace(small(), eval_events=8192, candidates=16)
    data = runner.truth_data(cfg)
    panel, truth = runner.evaluate(None, data, cfg, 13)
    shuffled = panel["samples"][torch.randperm(cfg.eval_events, generator=generator(23))]
    wrong = runner.generation_metrics(shuffled, panel["condition"], data)
    assert truth["conditional_mode_tv"] < .08
    assert wrong["conditional_mode_tv"] > .23
    assert truth["condition_moment_abs_error"] < .02
    assert wrong["condition_moment_abs_error"] > .35
    assert truth["near_corner_fraction"] == wrong["near_corner_fraction"]
    assert truth["marginal_mean_abs_max"] == pytest.approx(wrong["marginal_mean_abs_max"])
    comparison = runner.paired_tv_comparison(
        {"samples": shuffled, "condition": panel["condition"]}, panel, data, replicates=32)
    assert comparison["raw_minus_fourier_tv"] > .15
    assert comparison["bootstrap_lo95"] > .15


def test_paired_bootstrap_preserves_context_units_and_identity():
    cfg = small(); data = runner.truth_data(cfg)
    panel, _ = runner.evaluate(None, data, cfg, 13)
    identical = runner.paired_tv_comparison(panel, panel, data, replicates=16)
    assert identical["raw_minus_fourier_tv"] == identical["bootstrap_lo95"] == identical["bootstrap_hi95"] == 0
    other, _ = runner.evaluate(None, data, cfg, 14)
    first = runner.paired_tv_comparison(panel, other, data, replicates=16)
    doubled = {**panel, "samples": panel["samples"].repeat_interleave(2, dim=1)}
    doubled_other = {**other, "samples": other["samples"].repeat_interleave(2, dim=1)}
    assert first == runner.paired_tv_comparison(doubled, doubled_other, data, replicates=16)
    with pytest.raises(ValueError, match="midpoint"):
        runner.generation_metrics(panel["samples"], panel["condition"].flip(0), data)


def test_early_stop_checkpoint_selection_and_matched_budget(tmp_path):
    cfg = small(); dataset = runner.prepare_truth_dataset(tmp_path / "data.pt", cfg, 17)
    rows = []
    result = runner.fit_pair(tmp_path / "pair", dataset, cfg, 17, patience=2, min_delta=100.,
        monitor_every=2, emit=rows.append)
    assert result["state"] == "completed"
    assert result["matched_budget"]["available_epochs"] == 3
    assert result["early_stop"]["available_epochs"] == 3
    for basis in runner.BASES:
        assert result["arms"][basis]["steps"] == 6
        assert result["arms"][basis]["epochs"] == 3
        expected = min(r["validation"]["velocity_mse"] for r in rows if r["basis"] == basis and r["phase"] == "pretrain")
        assert result["arms"][basis]["best_mse"] == expected
        saved = torch.load(tmp_path / "pair" / f"early_stop_{basis}_model.pt", weights_only=True)
        assert saved["trained_on"] == "complete_truth" and saved["weights"] == "raw_no_ema"


def test_exact_epoch_boundary_resume(tmp_path):
    cfg = small(); dataset = runner.prepare_truth_dataset(tmp_path / "data.pt", cfg, 17)
    kw = dict(patience=2, min_delta=100., monitor_every=2, emit=lambda row: None)
    runner.fit_pair(tmp_path / "full", dataset, cfg, 17, **kw)
    result = runner.fit_pair(tmp_path / "resumed", dataset, cfg, 17, stop_after_epoch=1, **kw)
    assert result["state"] == "paused_test_only"
    assert "matched_budget" not in result
    runner.fit_pair(tmp_path / "resumed", dataset, cfg, 17, resume=True, **kw)
    full = torch.load(tmp_path / "full/last_state.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/last_state.pt", weights_only=True)
    assert_nested_equal(full, resumed)


def test_independent_stops_do_not_confuse_matched_budget(tmp_path, monkeypatch):
    cfg = small(); dataset = runner.prepare_truth_dataset(tmp_path / "data.pt", cfg, 17)
    scores = {"raw": iter([.9, .9, .9]), "fourier": iter([.9, .8, .7, .7, .7])}
    monkeypatch.setattr(runner, "validation", lambda model, panel: {
        "velocity_mse": next(scores[model.basis], .7), "time_bins": []})
    result = runner.fit_pair(tmp_path / "pair", dataset, cfg, 17, patience=2, min_delta=.01,
        monitor_every=10, emit=lambda row: None)
    assert result["matched_budget"]["available_epochs"] == 3
    assert result["matched_budget"]["raw"]["available_steps"] == 6
    assert result["matched_budget"]["fourier"]["available_steps"] == 6
    assert result["early_stop"]["raw"]["available_steps"] == 6
    assert result["early_stop"]["fourier"]["available_steps"] == 10
    assert result["stop_reason"] == "independent_validation_early_stopping"


def test_runner_and_resume_no_duplicate_completed_training(tmp_path, monkeypatch):
    kw = dict(cfg=small(), seeds=(17,), patience=1, min_delta=100., wandb_mode="disabled")
    result = runner.run(tmp_path / "run", **kw)
    assert result["state"] == "completed"
    assert result["decision"]["seed_replication"] == "pilot_only"
    assert result["contract"]["max_steps"] is None and result["contract"]["max_epochs"] is None
    assert result["truth_sampling_floor"]["samples"] == 1024
    with pytest.raises(ValueError, match="--resume"):
        runner.run(tmp_path / "run", **kw)
    monkeypatch.setattr(runner, "fit_pair", lambda *a, **k: pytest.fail("Completed run must not retrain"))
    assert runner.run(tmp_path / "run", resume=True, **kw) == result
    assert json.loads((tmp_path / "run/report.json").read_text())["state"] == "completed"


def test_decision_requires_every_seed_and_shape_guardrail():
    comparison = {"raw_minus_fourier_tv": .1, "bootstrap_lo95": .09}
    generation = {"near_corner_fraction": .95}
    pair = {"state": "completed", "early_stop": {"comparison": comparison,
        "raw": {"generation": generation}, "fourier": {"generation": generation}},
        "matched_budget": {"comparison": comparison}}
    pairs = {str(s): pair for s in (17, 23, 41)}
    assert runner.decisions(pairs, (17, 23, 41))["decision"] == "supports"
    assert runner.decisions(pairs, (17, 23, 41))["seed_replication"] == "three_or_more"
    import copy
    pairs = copy.deepcopy(pairs)
    pairs["41"] = copy.deepcopy(pair)
    pairs["41"]["early_stop"]["fourier"]["generation"] = {"near_corner_fraction": .9}
    result = runner.decisions(pairs, (17, 23, 41))
    assert result["primary_material_gain_all_seeds"] and result["decision"] == "unresolved"
    assert runner.decisions({}, (17, 23, 41))["state"] == "pending"


@pytest.mark.parametrize("changes", [{"patience": 0}, {"min_delta": float("nan")},
    {"seeds": (17, 17)}, {"cfg": replace(small(), eval_events=300)}])
def test_reject_invalid_contract(tmp_path, changes):
    with pytest.raises(ValueError):
        runner.run(tmp_path / "run", **{ "cfg": small(), "wandb_mode": "disabled", **changes})
