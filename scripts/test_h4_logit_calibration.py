import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evenet_dgpo"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from RL.DGPO_neutrino.omnifold_ztautau import logit_calibration as lc
import diagnose_h4_logit_calibration as runner


def gaussian_scores(n=10000, seed=19):
    rng = np.random.default_rng(seed)
    # p=N(0.8,1), q=N(-0.8,1): exact log(p/q)=1.6*x.
    t = 1.6 * rng.normal(0.8, 1, n)
    g = 1.6 * rng.normal(-0.8, 1, n)
    return (t + 0.4) / 0.55, (g + 0.4) / 0.55


def bundle_fixture(n=1000):
    t, g = gaussian_scores(n)
    identity = torch.arange((n + 40) * 2, dtype=torch.float32).reshape(n + 40, 2)
    return dict(schema="h4-ratio-health-v1", seed=42,
                split_protocol={"schema": "raw-monitor-condition-60-20-20-v1", "seed": 42},
                identity_condition=identity, truth_logits=torch.tensor(t), gen_logits=torch.tensor(g),
                split_indices=dict(fit=torch.arange(20), early_stop=torch.arange(20, 40),
                                   test=torch.arange(40, n + 40)),
                fit_diagnostics=dict(best_step=932, steps_completed=1036, validation_auc=float("nan")))


def save_source(tmp_path, n=1000):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(bundle_fixture(n), source / "best_classifier_and_test.pt")
    (source / "COMPLETE").write_text("h4-ratio-health-v1\n")
    return source


def test_known_affine_distortion_is_recovered():
    t, g = gaussian_scores(30000)
    fit = lc.fit_affine(t, g)
    assert fit["eligible"]
    assert fit["a"] == pytest.approx(0.55, abs=0.025)
    assert fit["b"] == pytest.approx(-0.4, abs=0.05)
    assert fit["curve"][-1]["bce"] < fit["curve"][0]["bce"]
    assert all(y["bce"] <= x["bce"] + 1e-10 for x, y in zip(fit["curve"], fit["curve"][1:]))


def test_held_out_improvement_preserves_auc():
    t, g = gaussian_scores()
    report, cal, test = lc.run_experiment(t, g, np.arange(len(t)), bootstrap=100)
    assert not np.intersect1d(cal, test).size
    assert report["decision"] == "exploratory_calibration_improvement"
    assert report["evaluation"]["uncertainty"]["bce_delta_ci95"][1] < 0
    assert report["evaluation"]["delta"]["auc"] == pytest.approx(0, abs=1e-12)
    assert report["protocol"]["confirmatory"] is False
    assert report["dgpo_interface"]["measured"] is False
    json.dumps(report, allow_nan=False)


def test_evaluation_scores_cannot_change_fitted_parameters():
    t, g = gaussian_scores(1000)
    groups = np.arange(len(t))
    original, cal, test = lc.run_experiment(t, g, groups, bootstrap=20)
    # Keep calibration rows fixed; radically change only held-out scores.
    changed_t, changed_g = t.copy(), g.copy()
    changed_t[test], changed_g[test] = -100, 100
    changed, _, _ = lc.run_experiment(changed_t, changed_g, groups, bootstrap=20)
    assert changed["fit"] == original["fit"]
    assert changed["evaluation"]["raw"]["bce"] != original["evaluation"]["raw"]["bce"]


def test_source_rejects_both_row_and_identity_leakage():
    bundle = bundle_fixture()
    t, g, groups, rows = lc.validate_source(bundle)
    assert len(t) == len(g) == len(groups) == len(rows) == 1000
    bundle["split_indices"]["test"][0] = 0
    with pytest.raises(ValueError, match="overlaps"):
        lc.validate_source(bundle)
    bundle = bundle_fixture()
    bundle["identity_condition"][40] = bundle["identity_condition"][0]
    with pytest.raises(ValueError, match="identity overlaps"):
        lc.validate_source(bundle)


def test_identity_groups_and_duplicate_splitting_are_stable():
    rows = np.array([[0., 1.], [-0., 1.], [1., 1.], [2., 1.], [3., 1.], [4., 1.]])
    groups = lc.identity_groups(rows, chunk_size=2)
    assert groups[0] == groups[1]
    np.testing.assert_array_equal(groups, lc.identity_groups(rows, chunk_size=5))
    cal, test = lc.split_groups(groups)
    assert not np.intersect1d(groups[cal], groups[test]).size
    permutation = np.array([5, 2, 0, 4, 1, 3])
    c2, _ = lc.split_groups(groups[permutation])
    assert set(groups[cal]) == set(groups[permutation][c2])


def test_constant_score_is_an_explicit_degenerate_case():
    fit = lc.fit_affine(np.full(20, 3.), np.full(20, 3.))
    assert fit["a"] == 1 and fit["b"] == -3
    assert fit["constant_scores"] and not fit["eligible"]
    assert fit["curve"][-1]["bce"] == pytest.approx(math.log(2))


def test_reverse_ranking_is_not_repaired_by_flipping_labels():
    fit = lc.fit_affine(-np.ones(20), np.ones(20))
    assert fit["a"] >= lc.MIN_SLOPE
    assert fit["slope_at_lower_bound"] and not fit["eligible"]


@pytest.mark.parametrize("truth,generated", [
    (np.ones(20), -np.ones(20)),
    (np.r_[np.ones(19), 0.], np.r_[-np.ones(19), 0.]),
])
def test_separable_calibration_does_not_get_certified(truth, generated):
    fit = lc.fit_affine(truth, generated)
    assert fit["separated_or_quasiseparated_calibration"] and not fit["eligible"]


def test_extreme_finite_scores_and_bootstrap_remain_finite():
    t, g = np.zeros(100), np.zeros(100)
    t[0], g[0] = -1000, 1000
    result = lc.compare(t, g, np.arange(100), 0.5, -0.2, bootstrap=30)
    assert result["raw"]["ratio"]["mean_ratio"] is None
    assert result["raw"]["ratio"]["ess"] == pytest.approx(1)
    json.dumps(result, allow_nan=False)


def test_bce_brier_reliability_and_relative_se():
    groups = np.arange(100)
    result = lc.measure(np.zeros(100), np.zeros(100), groups)
    assert result["bce"] == pytest.approx(math.log(2))
    assert result["brier"] == 0.25 and result["ece"] == 0
    assert sum(row["count"] for row in result["reliability"]) == 200
    assert result["ratio"]["ess"] == pytest.approx(100)
    assert result["ratio"]["log_mean_ratio"] == pytest.approx(0)
    scores = np.linspace(-1, 1, 100)
    expected_se = np.exp(scores).std(ddof=1) / math.sqrt(100) / np.exp(scores).mean()
    assert lc.ratio_metrics(scores, groups)["mean_ratio_relative_se"] == pytest.approx(expected_se)


def test_duplicate_events_use_cluster_uncertainty():
    t, g = gaussian_scores(100)
    single = lc.compare(t, g, np.arange(100), 0.7, 0.2, bootstrap=30)
    repeated = lc.compare(np.repeat(t, 2), np.repeat(g, 2), np.repeat(np.arange(100), 2),
                          0.7, 0.2, bootstrap=30)
    np.testing.assert_allclose(single["uncertainty"]["bce_delta_ci95"], repeated["uncertainty"]["bce_delta_ci95"])
    assert single["uncertainty"]["paired_bce_se"] == pytest.approx(repeated["uncertainty"]["paired_bce_se"])


def test_intercept_changes_ratio_normalization_not_concentration():
    _, g = gaussian_scores(100)
    groups = np.arange(100)
    raw = lc.ratio_metrics(g, groups)
    shifted = lc.ratio_metrics(g + 3, groups)
    assert shifted["log_mean_ratio"] == pytest.approx(raw["log_mean_ratio"] + 3)
    for key in ("ess", "ess_fraction", "top1pct_mass", "max_weight_mass", "mean_ratio_relative_se"):
        assert shifted[key] == pytest.approx(raw[key])


def test_unconverged_fit_is_not_a_positive_decision():
    t, g = gaussian_scores(1000)
    report, _, _ = lc.run_experiment(t, g, np.arange(1000), max_iterations=1, bootstrap=20)
    assert not report["fit"]["success"] and not report["fit"]["eligible"]
    assert report["decision"] == "inconclusive_fit_or_rank_check"


def test_affine_interface_matches_production_loo_and_gate():
    from RL.DGPO_neutrino.dgpo_utils import compute_per_event_advantage, build_dgpo_loss
    rng = torch.Generator().manual_seed(22)
    scores = torch.randn(8, 16, generator=rng, dtype=torch.float64)
    current = torch.randn(8, 16, generator=rng, dtype=torch.float64)
    reference = torch.randn(8, 16, generator=rng, dtype=torch.float64)
    a, b = 0.4, -1.3
    raw, _ = compute_per_event_advantage(scores, estimator="leave_one_out_unscaled")
    calibrated, _ = compute_per_event_advantage(a * scores + b, estimator="leave_one_out_unscaled")
    torch.testing.assert_close(calibrated, a * raw)
    loss, diagnostics = build_dgpo_loss(current, reference, calibrated, 1, 8)
    statistic = (raw * (current - reference)).mean(0)
    gate = torch.sigmoid(a * statistic)
    torch.testing.assert_close(diagnostics["w_e_mean"], gate.mean())
    torch.testing.assert_close(loss, (gate * calibrated * current).mean())


@pytest.mark.parametrize("field", ["truth_logits", "gen_logits"])
def test_nonfinite_input_rejected(field):
    bundle = bundle_fixture()
    bundle[field][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        lc.validate_source(bundle)


def test_k_greater_than_one_is_not_silently_flattened():
    bundle = bundle_fixture()
    bundle["truth_logits"] = bundle["truth_logits"].repeat(2)
    bundle["gen_logits"] = bundle["gen_logits"].repeat(2)
    with pytest.raises(ValueError, match="K=1"):
        lc.validate_source(bundle)


def test_rerun_overwrites_only_its_own_files(tmp_path):
    source, output = tmp_path / "source", tmp_path / "result"
    source.mkdir()
    output.mkdir()
    (output / "report.json").write_text("old")
    (output / "my_notes.txt").write_text("keep")
    path, replaced = runner.prepare_output(output, source)
    assert replaced == ["report.json"] and path == output
    assert (output / "my_notes.txt").read_text() == "keep"
    for invalid in (source, source / "nested", source.parent):
        with pytest.raises(ValueError, match="separate"):
            runner.prepare_output(invalid, source)


def test_full_cpu_cli_and_repeat(tmp_path):
    source = save_source(tmp_path)
    original = (source / "best_classifier_and_test.pt").read_bytes()
    output = tmp_path / "result"
    cmd = [sys.executable, str(Path(runner.__file__)), str(source), "--output", str(output),
           "--bootstrap", "20", "--no-wandb"]
    env = dict(os.environ, SLURM_PROCID="0", MPLCONFIGDIR=str(tmp_path / "mpl"))
    for _ in range(2):
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "report.json").read_text())
    assert (output / "COMPLETE").read_text().strip() == lc.SCHEMA
    assert report["source"]["source_fit_diagnostics"]["validation_auc"] is None
    assert report["protocol"]["classifier_fits"] == report["protocol"]["policy_updates"] == 0
    saved = np.load(output / "scores.npz")
    mask = saved["calibration_mask"]
    assert not np.intersect1d(saved["identity_group"][mask], saved["identity_group"][~mask]).size
    assert (source / "best_classifier_and_test.pt").read_bytes() == original
    for name in ("reliability.png", "log_ratio.png", "calibration_fit.png"):
        assert (output / name).stat().st_size > 1000


def test_wandb_logs_calibration_separately_and_keeps_event_data_local(tmp_path, monkeypatch):
    source = save_source(tmp_path, 400)
    captured, logs, saves = {}, [], []
    class Summary(dict):
        def update(self, mapping):
            super().update(mapping)
    run = SimpleNamespace(summary=Summary(), define_metric=lambda *a, **k: None,
                          log=logs.append, save=lambda path, **kw: saves.append(path),
                          finish=lambda **kw: captured.update(finish=kw))
    def init(**kwargs):
        captured.update(kwargs)
        return run
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Image=lambda path: path))
    monkeypatch.setenv("SLURM_PROCID", "0")
    assert runner.main([str(source), "--output", str(tmp_path / "result"), "--bootstrap", "20"]) == 0
    assert captured["id"] == "h4logcal1" and captured["resume"] == "allow"
    assert captured["name"] == runner.RUN_NAME
    assert run.summary["complete"] and run.summary["phase"] == "complete"
    assert "evaluation/delta/bce" in run.summary
    assert "evaluation/bce_delta_ci95/hi95" in run.summary
    assert all(Path(path).name != "scores.npz" for path in saves)
    fitting = [row for row in logs if "calibration_fit/iteration" in row]
    assert fitting and all(not any(key.startswith("evaluation/") for key in row) for row in fitting)


def test_incomplete_input_does_not_create_output(tmp_path):
    output = tmp_path / "result"
    with pytest.raises(SystemExit):
        runner.main([str(tmp_path / "missing"), "--output", str(output), "--no-wandb"])
    assert not output.exists()
