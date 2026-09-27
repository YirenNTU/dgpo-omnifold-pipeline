"""Frozen measured reference ratio, native objective, and exact continuation."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from experiments.dgpo_toy import cube_pretrained_dgpo as runner
from experiments.dgpo_toy.conditional import generator
from experiments.dgpo_toy.cube_truth_fourier import DEFAULT_CONFIG, initial_pair, truth_data
from experiments.dgpo_toy.test_fixed_reward import assert_nested_equal


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.fixture
def source(tmp_path):
    cfg = replace(DEFAULT_CONFIG, hidden=8, train_events=16, validation_events=16,
        test_events=16, fit_batch=8, eval_events=512, ddim_steps=2)
    models, _ = initial_pair(cfg, 17)
    # A nonzero learned-branch fixture checks that loading never resets the adapter.
    with torch.no_grad():
        models["fourier"].condition_adapter.weight.normal_(0, .05, generator=generator(3))
    path = tmp_path / "source"; (path / "seed17").mkdir(parents=True)
    torch.save({"model": models["fourier"].state_dict(), "config": asdict(cfg),
        "basis": "fourier", "condition_frequency": 8, "trained_on": "complete_truth",
        "weights": "raw_no_ema", "epoch": 42, "step": 10752}, path / "seed17/early_stop_fourier_model.pt")
    (path / "report.json").write_text(json.dumps({"state": "completed", "pairs": {"17": {"state": "completed"}}}))
    return path


def options():
    return dict(seeds=(17,), steps=4, eval_every=2, calibration_contexts=256,
        calibration_candidates=4, endpoint_contexts=512, endpoint_candidates=2, wandb_mode="disabled")


def test_truth_table_is_normalized_and_integrates_continuous_target():
    p = runner.target_table()
    assert p.shape == (256, 8) and (p > 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(256, dtype=torch.float64))
    c = ((torch.arange(65536, dtype=torch.float64) + .5) / 65536 * 2 - 1)[:, None]
    expected = truth_data(DEFAULT_CONFIG).probabilities(c[:, 0], .9).reshape(256, -1, 8).mean(1)
    torch.testing.assert_close(p, expected, atol=5e-9, rtol=1e-7)


def test_ratio_uses_measured_source_and_is_zero_only_if_reference_matches_truth():
    p = runner.target_table()
    # Undo the declared pseudocount so smoothed q equals p exactly.
    counts = p * (100000 + 8 * .5) - .5
    perfect = runner.FrozenModeRatio(counts)
    assert float(perfect.log_ratios.abs().max()) < 1e-6
    uniform = runner.FrozenModeRatio(torch.full((256, 8), 1000.))
    assert float(uniform.log_ratios.abs().max()) > 1
    q = uniform.reference_probabilities
    torch.testing.assert_close(q * uniform.log_ratios.double().exp(), p, atol=1e-8, rtol=1e-7)
    assert not list(uniform.parameters())


def test_reward_shapes_broadcasting_and_condition_boundaries():
    critic = runner.FrozenModeRatio(torch.ones(256, 8))
    y = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])[None].expand(3, -1, -1)
    c = torch.tensor([[-1.], [0.], [1.]])[:, None]
    r = critic(y, c)
    assert r.shape == (3, 2) and torch.isfinite(r).all()
    assert torch.equal(runner.bin_ids(c).flatten(), torch.tensor([0, 128, 255]))
    assert torch.equal(runner.mode_ids(y[0]), torch.tensor([0, 7]))


def test_loading_preserves_learned_fourier_weights_and_uses_fresh_policy_settings(source):
    cfg, model, saved, _ = runner.load_source(source, 17, 3000, 100)
    assert_nested_equal(model.state_dict(), saved["model"])
    assert model.condition_adapter.weight.count_nonzero() > 0
    assert all(not p.requires_grad for p in model.parameters())
    assert (cfg.policy_lr, cfg.batch, cfg.candidates, cfg.timesteps) == (1e-4, 64, 8, 4)
    assert cfg.policy_steps == 3000 and saved["step"] == 10752


def test_preflight_recovers_truth_for_an_independent_uniform_reference_panel():
    n = 512
    c = ((torch.arange(n, dtype=torch.float32) + .5) / n * 2 - 1)[:, None]
    corners = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    panel = {"condition": c, "samples": corners[None].expand(n, -1, -1)}
    counts = torch.full((256, 8), 1000.)
    critic = runner.FrozenModeRatio(2 * counts)
    result = runner.reward_preflight(panel, critic, counts, counts)
    assert result["baseline_mode_tv"] > .25
    assert result["ideal_reweighted_mode_tv"] < 1e-12
    assert result["ideal_reward_headroom"] > 0
    assert result["mean_ratio"] == pytest.approx(1.)
    assert result["split_calibration_centered_reward_cosine"] == pytest.approx(1.)
    assert result["split_calibration_reference_tv"] == 0


def test_calibration_counts_are_reproducible_and_do_not_change_source(source):
    cfg, model, saved, _ = runner.load_source(source, 17, 4, 2)
    a = runner.calibration_counts(model, cfg, 19, contexts=256, candidates=4)
    b = runner.calibration_counts(model, cfg, 19, contexts=256, candidates=4)
    assert torch.equal(a, b)
    assert a.shape == (256, 8) and int(a.sum()) == 1024
    assert torch.equal(a.sum(-1), torch.full((256,), 4))
    assert_nested_equal(model.state_dict(), saved["model"])


def test_independent_half_joint_metric_has_no_plugin_floor():
    # For balanced iid uniform modes against uniform truth, construct arbitrary
    # probability residuals with exactly opposite candidate-half errors. The
    # cross score may be negative; it must not clamp to zero.
    n = 512; c = ((torch.arange(n, dtype=torch.float32) + .5) / n * 2 - 1)[:, None]
    corners = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    generator_ = generator(25)
    y = corners[torch.randint(8, (n, 2), generator=generator_)]
    panel = {"samples": y, "condition": c}
    score = runner.split_joint_squared_error(panel)
    probs = torch.nn.functional.one_hot(runner.mode_ids(y), 8).double().reshape(256, 2, 2, 8)
    target = truth_data(DEFAULT_CONFIG).probabilities(c[:, 0].double(), .9).reshape(256, 2, 8).mean(1)
    expected = ((probs[:, :, 0].mean(1) - target) * (probs[:, :, 1].mean(1) - target)).sum(-1).mean()
    assert score == pytest.approx(float(expected), abs=1e-12)
    repeated = {**panel, "samples": y.repeat_interleave(2, dim=1)}
    assert runner.split_joint_squared_error(repeated) == score
    with pytest.raises(ValueError, match="even"):
        runner.split_joint_squared_error({**panel, "samples": y[:, :1]})


def test_finite_native_dgpo_smoke_with_immutable_source_reward_and_valid_clocks(source, tmp_path):
    raw_file = source / "seed17/early_stop_fourier_model.pt"
    before = raw_file.read_bytes()
    result = runner.run(source, tmp_path / "run", **options())
    arm = result["arms"]["17"]
    assert result["state"] == "completed" and arm["completed_steps"] == 4
    assert arm["source_unchanged"] and arm["reward_unchanged"]
    assert raw_file.read_bytes() == before
    assert result["contract"]["new_pretrain_steps"] == 0 and result["contract"]["classifier"] is None
    state = torch.load(tmp_path / "run/seed17/policy_state.pt", weights_only=True)
    assert len(state["history"]) == 4
    assert state["velocity_coefficient"] == 1.
    assert all(int(x["step"]) == 4 for x in state["optimizer"]["state"].values())
    assert "main_gradient_norm" in state["history"][-1]
    assert arm["evaluations"]["4"]["metrics"]["samples"] == 1024
    assert all(torch.isfinite(v).all() for v in state["model"].values())


def test_interruption_resume_matches_uninterrupted_native_updates(source, tmp_path, monkeypatch):
    native = runner.policy_train
    runner.run(source, tmp_path / "full", **options())
    def interrupted(*args, **kwargs):
        save = kwargs["checkpoint_callback"]
        def fail_after_save(step, model, opt, rng, history):
            save(step, model, opt, rng, history)
            if step == 2:
                raise RuntimeError("test interruption")
        kwargs["checkpoint_callback"] = fail_after_save
        return native(*args, **kwargs)
    monkeypatch.setattr(runner, "policy_train", interrupted)
    with pytest.raises(RuntimeError, match="test interruption"):
        runner.run(source, tmp_path / "resumed", **options())
    monkeypatch.setattr(runner, "policy_train", native)
    result = runner.run(source, tmp_path / "resumed", resume=True, **options())
    assert result["state"] == "completed" and "error" not in result
    assert "test interruption" in result["previous_failures"][0]
    full = torch.load(tmp_path / "full/seed17/policy_state.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/seed17/policy_state.pt", weights_only=True)
    assert_nested_equal(full, resumed)
    other = json.loads((tmp_path / "full/report.json").read_text())
    assert result["arms"]["17"]["evaluations"] == other["arms"]["17"]["evaluations"]


def test_failed_final_endpoint_resumes_without_extra_updates(source, tmp_path, monkeypatch):
    original = runner.endpoint_changes
    def fail(*args):
        raise RuntimeError("test endpoint failure")
    monkeypatch.setattr(runner, "endpoint_changes", fail)
    with pytest.raises(RuntimeError, match="test endpoint failure"):
        runner.run(source, tmp_path / "run", **options())
    monkeypatch.setattr(runner, "endpoint_changes", original)
    monkeypatch.setattr(runner, "policy_train", lambda *a, **k: pytest.fail("No extra optimizer updates allowed"))
    result = runner.run(source, tmp_path / "run", resume=True, **options())
    assert result["state"] == "completed" and result["arms"]["17"]["completed_steps"] == 4


@pytest.mark.parametrize("changes", [{"steps": 0}, {"endpoint_candidates": 3},
    {"calibration_contexts": 300}, {"seeds": (17, 17)}])
def test_bad_contract_rejected(source, tmp_path, changes):
    with pytest.raises(ValueError):
        runner.run(source, tmp_path / "bad", **{**options(), **changes})


@pytest.mark.parametrize("field,value", [("basis", "raw"), ("trained_on", "reference"), ("weights", "ema")])
def test_wrong_source_kind_rejected(source, field, value):
    path = source / "seed17/early_stop_fourier_model.pt"
    saved = torch.load(path, weights_only=True); saved[field] = value; torch.save(saved, path)
    with pytest.raises(ValueError, match="Fourier complete-truth"):
        runner.load_source(source, 17, 4, 2)
