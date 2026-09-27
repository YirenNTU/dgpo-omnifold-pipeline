"""Fixed-reward replay: no classifier fitting or silent blanket gate bypass."""
from dataclasses import asdict, replace
import json

import pytest
import torch

from experiments.dgpo_toy.conditional import Config, Denoiser, Distribution, generator
from experiments.dgpo_toy.low_ess_recovery import JointFourierClassifier
from experiments.dgpo_toy.fixed_reward import (run, load_source, unchanged,
    REQUIRED_GATES, WAIVED_GATES, panel_summary)


@pytest.fixture
def saved(tmp_path):
    torch.set_num_threads(1)
    cfg = replace(Config(), dimensions=6, hidden=8, classifier_hidden=8,
                  policy_steps=2, eval_events=8, batch=4, ddim_steps=3, eval_every=1)
    source = tmp_path/"source"/"reward.pt"
    source.parent.mkdir()
    initial, critic = Denoiser(cfg), JointFourierClassifier(cfg)
    torch.save({"config": asdict(cfg), "initial": initial.state_dict(),
                "classifier": critic.state_dict(), "selected_step": 123,
                "feature_mode": "joint3"}, source)
    report = {"config": asdict(cfg), "fit": {"selected_step": 123},
              "feature_mode": "joint3", "confirmation": {"ess_fraction": .001},
              "validity": {**dict.fromkeys(REQUIRED_GATES, True),
                           **dict.fromkeys(WAIVED_GATES, False)}}
    source.with_name("report.json").write_text(json.dumps(report))
    return source


def test_load_exact_frozen_weights_and_differentiable_inputs(saved):
    cfg, initial, critic, payload, _ = load_source(saved)
    assert unchanged(initial, payload["initial"])
    assert unchanged(critic, payload["classifier"])
    assert not critic.training and not initial.training
    assert all(not p.requires_grad for p in critic.parameters())
    c = Distribution(cfg).contexts(4, generator(17))
    y = torch.randn(4, cfg.dimensions, requires_grad=True)
    grad = torch.autograd.grad(critic(y, c, Distribution(cfg)).sum(), y)[0]
    assert grad.norm() > 0


def test_policy_run_waives_only_named_fidelity_gates_and_preserves_source(saved, tmp_path):
    before = saved.read_bytes(), saved.with_name("report.json").read_bytes()
    result = run(saved, tmp_path/"policy", smoke=True)
    assert result["decision"] == "smoke_only"
    assert set(result["endpoints"]) == {"dgpo", "pathwise"}
    assert result["frozen_source_unchanged"]
    assert result["source_validity"]["ratio_normalization"] is False
    assert result["validity"]["low_ess"] is True
    assert len(result["policy_histories"]["dgpo"]) == 2
    assert (tmp_path/"policy"/"dgpo.pt").exists()
    assert before == (saved.read_bytes(), saved.with_name("report.json").read_bytes())


def test_remaining_setup_gate_is_not_silently_ignored(saved, tmp_path):
    path = saved.with_name("report.json")
    evidence = json.loads(path.read_text())
    evidence["validity"]["low_ess"] = False
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="low_ess"):
        run(saved, tmp_path/"policy")


def test_mismatched_source_report_and_overwrite_source_rejected(saved):
    with pytest.raises(ValueError, match="separate"):
        run(saved, saved.parent)
    path = saved.with_name("report.json")
    evidence = json.loads(path.read_text())
    evidence["fit"]["selected_step"] += 1
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="selections"):
        load_source(saved)


def test_nonfinite_panel_stops_before_interpretation():
    with pytest.raises(FloatingPointError):
        panel_summary(torch.tensor([[float("nan")]]), torch.zeros(1, 1))


def test_longer_run_exactly_replays_prefix_and_saves_full_state(saved, tmp_path):
    short, longer = tmp_path/"short", tmp_path/"long"
    run(saved, short)
    result = run(saved, longer, policy_steps=4, replay_from=short, endpoint_seed=92017)
    assert all(v["exact_weights"] and v["exact_history"]
               for v in result["replay_verification"].values())
    assert result["stream_seeds"]["endpoint"] == 92017
    for arm in ("dgpo", "pathwise"):
        state = torch.load(longer/(arm+"_state.pt"), weights_only=True)
        assert state["step"] == 4 and len(state["history"]) == 4
        assert state["optimizer"]["state"]
        assert all(float(v["step"]) == 4 for v in state["optimizer"]["state"].values())
        assert state["rng"].dtype == torch.uint8


def test_replay_rejects_changed_settings_and_mismatched_prefix(saved, tmp_path):
    short = tmp_path/"short"
    run(saved, short)
    with pytest.raises(ValueError, match="only policy_steps"):
        run(saved, tmp_path/"bad_seed", seed=29, policy_steps=4, replay_from=short)
    with pytest.raises(ValueError, match="separate"):
        run(saved, short, policy_steps=4, replay_from=short)
    path = short/"dgpo.pt"
    checkpoint = torch.load(path, weights_only=True)
    next(iter(checkpoint["model"].values())).add_(1.)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="prefix"):
        run(saved, tmp_path/"bad_weights", policy_steps=4, replay_from=short)


def test_policy_budget_and_endpoint_stream_validation(saved, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        run(saved, tmp_path/"zero", policy_steps=0)
    with pytest.raises(ValueError, match="stream"):
        run(saved, tmp_path/"overlap", endpoint_seed=90017)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_resume_equals_uninterrupted_model_optimizer_rng_and_history(saved, tmp_path):
    short, full, resumed = tmp_path/"short", tmp_path/"full", tmp_path/"resumed"
    run(saved, short)
    run(saved, full, policy_steps=4, endpoint_seed=93017)
    result = run(saved, resumed, policy_steps=4, endpoint_seed=93017, resume_from=short)
    for arm in ("dgpo", "pathwise"):
        a = torch.load(full/(arm+"_state.pt"), weights_only=True)
        b = torch.load(resumed/(arm+"_state.pt"), weights_only=True)
        assert_nested_equal(a, b)
        assert result["resume_verification"][arm]["monitor_boundary_exact"]
        assert result["resume_verification"][arm]["new_optimizer_updates"] == 2
        assert "gain_since_resume" in result["endpoints"][arm]
    rows = [json.loads(x) for x in (resumed/"progress.jsonl").read_text().splitlines()]
    assert all(x["step"] > 2 for x in rows if x["phase"] == "policy")


def test_resume_rejects_wrong_clock_and_reset_settings(saved, tmp_path):
    short = tmp_path/"short"
    run(saved, short)
    with pytest.raises(ValueError, match="separate"):
        run(saved, short, policy_steps=4, resume_from=short)
    with pytest.raises(ValueError, match="combined"):
        run(saved, tmp_path/"mix", policy_steps=4, replay_from=short, resume_from=short)
    with pytest.raises(ValueError, match="preserve"):
        run(saved, tmp_path/"seed", seed=29, policy_steps=4, resume_from=short)
    path = short/"dgpo_state.pt"
    state = torch.load(path, weights_only=True)
    next(iter(state["optimizer"]["state"].values()))["step"].zero_()
    torch.save(state, path)
    with pytest.raises(ValueError, match="clock"):
        run(saved, tmp_path/"clock", policy_steps=4, resume_from=short)
