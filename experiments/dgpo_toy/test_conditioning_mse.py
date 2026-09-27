"""Small contract checks, not an experimental training run."""
from dataclasses import replace

import pytest
import torch

from experiments.dgpo_toy import conditioning_mse as toy


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small():
    return replace(toy.Config(), train_events=32, validation_events=32, test_events=64,
                   train_probe_events=16, hidden=8, batch_size=16, patience=2, min_delta=100.)


@pytest.mark.parametrize("case", toy.CASES+toy.FREQUENCY_CASES)
def test_function_scale_and_finiteness(case):
    c = ((torch.arange(100000, dtype=torch.float64)+.5)/100000*2-1)[:, None]
    mean = toy.mean_function(c, case)
    assert torch.isfinite(mean).all()
    assert abs(float(mean.mean())) < 1e-7
    assert float(mean.square().mean()) == pytest.approx(1., abs=1e-7)


def test_shared_conditions_and_data_noise_across_cases():
    cfg = small()
    datasets = {case: toy.make_dataset(cfg, case) for case in toy.CASES}
    reference = datasets["linear"]
    for case, data in datasets.items():
        for split in ("train", "validation", "test"):
            c = data[split]["condition"]
            assert torch.equal(c, reference[split]["condition"])
            noise = data[split]["target"]-toy.mean_function(c, case)
            original = reference[split]["target"]-toy.mean_function(c, "linear")
            torch.testing.assert_close(noise, original, atol=3e-7, rtol=0)
    assert not torch.equal(reference["train"]["condition"], reference["validation"]["condition"])
    again = toy.make_dataset(cfg, "linear")
    assert torch.equal(reference["train"]["target"], again["train"]["target"])


def test_fourier_depends_only_on_condition_and_is_unit_variance():
    c = ((torch.arange(10000)+.5)/10000*2-1)[:, None]
    features = toy.condition_features(c, "fourier")
    assert features.shape == (10000, 8)
    torch.testing.assert_close(features.mean(0), torch.zeros(8), atol=2e-6, rtol=0)
    torch.testing.assert_close(features.square().mean(0), torch.ones(8), atol=2e-6, rtol=0)
    assert toy.condition_features(c, "raw").count_nonzero() == 0


def test_initial_functions_match_and_both_receive_same_noise(monkeypatch):
    cfg = small()
    models = toy.initial_pair(cfg, 17)
    dataset = toy.make_dataset(cfg, "periodic")
    c = dataset["train"]["condition"]
    x, t, target = toy.noisy_target(dataset["train"]["target"], toy.generator(18))
    assert torch.equal(models["raw"](x, t, c), models["fourier"](x, t, c))
    assert len({sum(p.numel() for p in m.parameters()) for m in models.values()}) == 1
    seen = {arm: [] for arm in toy.ARMS}
    for arm, model in models.items():
        model.register_forward_pre_hook(lambda m, args, arm=arm: seen[arm].append(
            tuple(value.detach().clone() for value in args)))
    draws = []
    real_noisy_target = toy.noisy_target
    def record(y, rng):
        result = real_noisy_target(y, rng)
        draws.append(result)
        return result
    monkeypatch.setattr(toy, "noisy_target", record)
    # Oracle and truth function must never be called inside the training loop.
    monkeypatch.setattr(toy, "oracle_velocity", lambda *a: pytest.fail("Teacher leakage"))
    monkeypatch.setattr(toy, "mean_function", lambda *a: pytest.fail("Truth leakage"))
    opts = {a: torch.optim.AdamW(m.parameters(), lr=cfg.lr) for a, m in models.items()}
    toy.train_epoch(models, opts, dataset["train"], cfg, toy.generator(19))
    assert len(draws) == len(seen["raw"]) == len(seen["fourier"]) == 2
    for raw, fourier in zip(seen["raw"], seen["fourier"]):
        assert all(torch.equal(a, b) for a, b in zip(raw, fourier))
    assert models["raw"].condition_adapter.weight.count_nonzero() == 0
    assert models["fourier"].condition_adapter.weight.count_nonzero() > 0


@pytest.mark.parametrize("case", toy.CASES+toy.FREQUENCY_CASES)
def test_analytic_teacher_and_irreducible_error(case):
    cfg = replace(small(), test_events=262144)
    dataset = toy.make_dataset(cfg, case)
    panel = toy.make_panel(dataset["test"], 78, case, cfg)
    residual = (panel["target"]-panel["oracle"]).double()
    # Analytic conditional mean: the error is orthogonal to x_t and c.
    a, _ = toy.alpha_sigma(panel["t"][:, None])
    centered_x = panel["x"]-a*toy.mean_function(panel["condition"], case)
    assert abs(float(residual.mean())) < .004
    assert abs(float((residual*centered_x).mean())) < .004
    assert abs(float((residual*panel["condition"]).mean())) < .004
    empirical = float(residual.square().mean())
    floor = float(panel["floor"].mean())
    assert empirical == pytest.approx(floor, abs=.006)
    # For the nearly full cosine time interval the mean Bayes risk is ~noise_width.
    assert floor == pytest.approx(cfg.noise_width, abs=.004)
    shifted_mse = float((residual-.2).square().mean())
    assert shifted_mse-empirical == pytest.approx(.04, abs=.002)


def test_oracle_matches_closed_form_at_unit_width():
    c = torch.tensor([[-.8], [.2], [.7]])
    x = torch.tensor([[-8.], [1.], [3.]])
    t = torch.tensor([0., .5, 1.])
    mean, floor = toy.oracle_velocity(x, t, c, "linear", 1.)
    _, s = toy.alpha_sigma(t[:, None])
    torch.testing.assert_close(mean, -s*toy.mean_function(c, "linear"))
    torch.testing.assert_close(floor, torch.ones_like(floor))


def test_paired_interval_and_decisions():
    raw = torch.ones(100, dtype=torch.float64)
    same = toy.paired_comparison(raw, raw, .005)
    assert same["fourier_minus_raw_mse"] == same["lo95"] == same["hi95"] == 0
    assert same["decision"] == "unresolved"
    assert toy.paired_comparison(raw, raw-.1, .005)["decision"] == "fourier_better"
    assert toy.paired_comparison(raw, raw+.1, .005)["decision"] == "fourier_worse"


def test_frequency_banks_have_equal_scale_size_and_exact_initial_predictions(monkeypatch):
    c = ((torch.arange(10000)+.5)/10000*2-1)[:, None]
    for arm in toy.FREQUENCY_ARMS[1:]:
        features = toy.condition_features(c, arm)
        assert features.shape == (10000, 8)
        torch.testing.assert_close(features.mean(0), torch.zeros(8), atol=2e-6, rtol=0)
        torch.testing.assert_close(features.square().mean(0), torch.ones(8), atol=2e-6, rtol=0)
    cfg = small()
    models = toy.initial_pair(cfg, 17, toy.FREQUENCY_ARMS)
    original_pair = toy.initial_pair(cfg, 17)
    data = toy.make_dataset(cfg, "chirp")["train"]
    x, t, _ = toy.noisy_target(data["target"], toy.generator(15))
    pred = models["raw"](x, t, data["condition"])
    assert len({sum(p.numel() for p in model.parameters()) for model in models.values()}) == 1
    seen = {arm: [] for arm in models}
    for arm, model in models.items():
        assert torch.equal(pred, model(x, t, data["condition"]))
        if arm in original_pair:
            for name, value in model.state_dict().items():
                assert torch.equal(value, original_pair[arm].state_dict()[name])
        model.register_forward_pre_hook(lambda m, args, arm=arm: seen[arm].append(
            tuple(value.detach().clone() for value in args)))
    monkeypatch.setattr(toy, "mean_function", lambda *a: pytest.fail("No truth input"))
    monkeypatch.setattr(toy, "oracle_velocity", lambda *a: pytest.fail("No teacher target"))
    opts = {arm: torch.optim.AdamW(model.parameters(), lr=cfg.lr) for arm, model in models.items()}
    toy.train_epoch(models, opts, data, cfg, toy.generator(16))
    for arm in models:
        assert len(seen[arm]) == 2
        for baseline, candidate in zip(seen["raw"], seen[arm]):
            assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
        if arm != "raw":
            assert models[arm].condition_adapter.weight.count_nonzero() > 0


def test_frequency_runner_metadata_and_prespecified_contrasts(tmp_path, monkeypatch):
    monkeypatch.setattr(toy, "plot_design", lambda *args: None)
    monkeypatch.setattr(toy, "plot_results", lambda *args: None)
    cfg = replace(small(), max_epochs=1)
    report = toy.run(tmp_path, cfg, ("chirp",), (17,), toy.FREQUENCY_ARMS)
    assert report["frequency_banks"]["fourier_high"] == [3., 6., 9., 12.]
    assert report["arms"] == list(toy.FREQUENCY_ARMS)
    assert report["primary_contrast"] == "fourier_high_minus_fourier"
    assert report["decisions"]["chirp"]["contrast"] == report["primary_contrast"]
    assert not report["decisions"]["chirp"]["rescue_checks_by_seed"]["17"]["all_checks_pass"]
    result = report["pairs"]["chirp/17"]
    assert result["epochs"] == 1
    assert len(result["history"]) == 8
    assert all(value is None for value in result["first_patience_plateau"].values())
    comparison = result["comparisons"]["fourier_high_minus_fourier"]
    expected = result["endpoint"]["fourier_high"]["test"]["velocity_mse"] - result["endpoint"]["fourier"]["test"]["velocity_mse"]
    assert comparison["candidate_minus_baseline_mse"] == pytest.approx(expected, abs=1e-12)
    assert comparison["baseline"] == "fourier" and comparison["candidate"] == "fourier_high"
    assert "fourier_minus_raw_mse" not in comparison
    checkpoint = torch.load(tmp_path/"chirp/17/best_fourier_high.pt", weights_only=True)
    assert checkpoint["arm"] == "fourier_high"
    assert checkpoint["frequencies"] == [3., 6., 9., 12.]


def test_paired_early_stop_selection_and_test_is_not_used_for_stopping(tmp_path, monkeypatch):
    cfg = small()
    dataset = toy.make_dataset(cfg, "linear")
    rows, panels = [], []
    original = toy.make_panel
    def record(split, seed, case, config):
        panels.append((seed, len(rows)))
        return original(split, seed, case, config)
    monkeypatch.setattr(toy, "make_panel", record)
    result = toy.fit_pair(tmp_path, dataset, cfg, "linear", 17, rows.append)
    assert result["epochs"] == 2
    assert result["initial_velocity_exact"]
    assert result["stop_reason"] == "paired_validation_early_stop"
    assert all(value["epoch"] == 2 for value in result["first_patience_plateau"].values())
    assert len(rows) == 6  # Epochs 0,1,2, both arms.
    assert panels == [(71017, 0), (72017, 0), (74017, 6)]
    for arm in toy.ARMS:
        candidate = min((r for r in rows if r["arm"] == arm), key=lambda r: r["validation"]["velocity_mse"])
        assert result["endpoint"][arm]["selected_epoch"] == candidate["epoch"]
        assert result["endpoint"][arm]["validation_mse"] == candidate["validation"]["velocity_mse"]
        saved = torch.load(tmp_path/f"best_{arm}.pt", weights_only=True)
        assert saved["weights"] == "raw_no_ema"
        assert saved["epoch"] == candidate["epoch"]


def test_budget_limit_does_not_claim_convergence(tmp_path):
    cfg = replace(small(), max_epochs=1)
    result = toy.fit_pair(tmp_path, toy.make_dataset(cfg, "bump"), cfg, "bump", 17, lambda row: None)
    assert result["epochs"] == 1
    assert result["stop_reason"] == "budget_limit_not_convergence"


def test_runner_reports_pilot_without_claiming_convergence(tmp_path, monkeypatch):
    monkeypatch.setattr(toy, "plot_design", lambda *args: None)
    monkeypatch.setattr(toy, "plot_results", lambda *args: None)
    result = toy.run(tmp_path, replace(small(), max_epochs=1), ("linear",), (17,))
    assert result["state"] == "completed"
    assert result["oracle_used_in_training"] is False
    assert result["decisions"]["linear"]["replication"] == "pilot_only"
    assert result["decisions"]["linear"]["both_arms_plateaued"] is False
    assert (tmp_path/"progress.jsonl").is_file()
    with pytest.raises(ValueError, match="Existing experiment"):
        toy.run(tmp_path, small(), ("linear",), (17,))


def test_measurement_bins_cover_panel_and_reconstruct_mse():
    cfg = replace(small(), test_events=4096)
    panel = toy.make_panel(toy.make_dataset(cfg, "bump")["test"], 89, "bump", cfg)
    model = toy.initial_pair(cfg, 17)["raw"]
    result, errors = toy.evaluate(model, panel)
    for key in ("time_bins", "condition_bins"):
        assert sum(r["count"] for r in result[key]) == len(errors)
        reconstructed = sum(r["count"]*r["velocity_mse"] for r in result[key])/len(errors)
        assert reconstructed == pytest.approx(result["velocity_mse"], abs=1e-12)


@pytest.mark.parametrize("changes", [dict(lr=0), dict(lr=float("nan")), dict(noise_width=0),
                                    dict(max_epochs=0), dict(min_delta=-1), dict(patience=0)])
def test_invalid_configs(changes):
    with pytest.raises(ValueError):
        toy.validate_config(replace(small(), **changes))
