"""Saved no-Fourier ablation provenance, matched objective and measurement tests."""
import copy
from dataclasses import replace
import json

import pytest
import torch

from experiments.dgpo_toy.conditional import (
    Classifier, Distribution, generator, make_panel, classifier_metrics, policy_train,
)
from experiments.dgpo_toy.fixed_reward import load_source, run as fixed_run
from experiments.dgpo_toy.net_reward_kl import run as endpoint_run
from experiments.dgpo_toy.velocity_kl import run as velocity_run
from experiments.dgpo_toy.direct_density import EndpointRatio, DensityValidityError
from experiments.dgpo_toy.weak_classifier import (
    run, load_plain, load_control, setup_gates, paired_scores, signal_metrics, endpoint_decision,
)
from experiments.dgpo_toy.test_fixed_reward import saved, assert_nested_equal


@pytest.fixture
def plain(saved, tmp_path):
    cfg, initial, _, payload, _ = load_source(saved)
    # Keep validation small in BOTH source files, not by skipping provenance.
    path = saved.with_name("report.json")
    evidence = json.loads(path.read_text())
    for obj in (payload["config"], evidence["config"]):
        obj["validation_events"] = 16
    torch.save(payload, saved)
    path.write_text(json.dumps(evidence))
    cfg = replace(cfg, validation_events=16)
    critic = Classifier(cfg, False).eval().requires_grad_(False)
    metrics = classifier_metrics(critic, make_panel(initial, Distribution(cfg), cfg, 16, 20017), Distribution(cfg))
    p = tmp_path/"plain"/"models.pt"
    p.parent.mkdir()
    torch.save({"config": payload["config"], "initial": payload["initial"],
                "classifiers": {"plain": critic.state_dict()}}, p)
    p.with_name("report.json").write_text(json.dumps({"config": payload["config"], "seed": 17,
        "classifiers": {"plain": {"selected_step": 9900, "validation_bce": metrics["bce"],
            "history": [{"step": 9900, **metrics}], "parameters": sum(x.numel() for x in critic.parameters())}}}))
    return p


def test_load_plain_exact_frozen_no_fourier_and_same_generator(saved, plain):
    cfg, _, _, payload, _ = load_source(saved)
    model, bundle, fit, _ = load_plain(plain, payload, cfg, 17)
    assert fit["selected_step"] == 9900 and not model.fourier
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert model.encoder[0].in_features == cfg.context_dim+cfg.dimensions
    assert all(torch.equal(v, bundle["classifiers"]["plain"][k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize("change", ["generator", "seed", "config", "selection", "nonfinite"])
def test_plain_provenance_rejects_mismatch(saved, plain, change):
    cfg, _, _, payload, _ = load_source(saved)
    a = torch.load(plain, weights_only=True)
    b = json.loads(plain.with_name("report.json").read_text())
    if change == "generator":
        next(iter(a["initial"].values())).add_(1)
    elif change == "seed":
        b["seed"] = 29
    elif change == "config":
        a["config"]["kappa"] = b["config"]["kappa"] = 9.
    elif change == "selection":
        b["classifiers"]["plain"]["selected_step"] = 9700
    else:
        a["classifiers"]["plain"]["head.weight"].fill_(float("nan"))
    torch.save(a, plain)
    plain.with_name("report.json").write_text(json.dumps(b))
    with pytest.raises(ValueError):
        load_plain(plain, payload, cfg, 17)


def test_ess_gate_requires_information_and_actual_concentration_contrast():
    confirmation = {"weak": {"auc": .9, "bce": .35, "ess_fraction": .1},
                    "strong": {"auc": .95, "bce": .2, "ess_fraction": .002}}
    signal = {"centered_weak_strong_cosine": .4, "weak": {"informative_group_fraction": 1.}}
    assert all(setup_gates(confirmation, signal, True).values())
    confirmation["weak"].update(auc=.5, ess_fraction=1.)
    assert not setup_gates(confirmation, signal, True)["weaker_but_informative"]
    assert not setup_gates(confirmation, signal, True)["higher_ess"]


def test_pair_scoring_preserves_rng_and_separates_global_from_within_k(saved, plain):
    cfg, initial, strong, payload, _ = load_source(saved)
    weak, *_ = load_plain(plain, payload, cfg, 17)
    state = torch.random.get_rng_state().clone()
    first = paired_scores(initial, weak, strong, Distribution(cfg), cfg, 108017)
    second = paired_scores(initial, weak, strong, Distribution(cfg), cfg, 108017)
    assert_nested_equal(first, second)
    assert torch.equal(state, torch.random.get_rng_state())
    values = torch.tensor([[0., .1], [10., 10.1]])
    metrics = signal_metrics({"weak": values, "strong": values})
    assert metrics["weak"]["within_k_ess_fraction"] > .99
    assert metrics["weak"]["ess_fraction"] < .51
    assert metrics["centered_weak_strong_cosine"] == pytest.approx(1.)


@pytest.mark.parametrize("mode", ["none", "endpoint", "velocity"])
def test_smoke_no_fit_frozen_sources_and_unchanged_training(saved, plain, tmp_path, monkeypatch, mode):
    no_kl = tmp_path/"no_kl"
    fixed_run(saved, no_kl)
    control = no_kl
    if mode == "endpoint":
        control = tmp_path/"endpoint"
        endpoint_run(saved, no_kl, control, policy_steps=2)
    elif mode == "velocity":
        control = tmp_path/"velocity"
        velocity_run(saved, no_kl, control, coefficient=1., policy_steps=2)
    source_bytes = saved.read_bytes(), plain.read_bytes(), (control/"dgpo.pt").read_bytes()
    import experiments.dgpo_toy.low_ess_recovery as fitting
    monkeypatch.setattr(fitting, "fit_streaming", lambda *a, **k: pytest.fail("Must not fit classifier"))
    out = tmp_path/"ablation"
    result = run(saved, control, out, plain_checkpoint=plain, regularization=mode, policy_steps=2, smoke=True)
    assert result["decision"] == "smoke_only" and result["state"] == "completed"
    assert result["completed_steps"] == 2 and result["new_classifier_fit_updates"] == 0
    assert result["classifier_selected_step"] == 9900 and result["frozen_models_unchanged"]
    assert result["classifier_provenance"]["validation_reproduced"]
    assert result["comparison_budgets_matched"]
    assert "strong_gain" in result["endpoints"]["weak_policy"]
    assert source_bytes == (saved.read_bytes(), plain.read_bytes(), (control/"dgpo.pt").read_bytes())
    checkpoint = torch.load(out/"dgpo_state.pt", weights_only=True)
    assert checkpoint["step"] == len(checkpoint["history"]) == 2
    assert checkpoint["velocity_coefficient"] == result["velocity_coefficient"] == (1. if mode == "velocity" else 0.)
    if mode == "velocity":
        assert checkpoint["endpoint_controller"] is None
        assert result["density_settings"] is None
        assert result["velocity_proxy"]["exact_endpoint_kl"] is False
        first, last = result["policy_history"]
        assert first["velocity_penalty"] == first["velocity_gradient_norm"] == 0.
        assert last["velocity_penalty"] == pytest.approx(.5*last["velocity_mse"])
        assert last["velocity_penalty"] > 0.
        assert last["total_loss"] == pytest.approx(last["dgpo_loss"]+last["velocity_penalty"])
        assert "main_velocity_gradient_cosine" in last
        assert "endpoint_kl_score_mean" not in last
    assert all(float(x["step"]) == 2 for x in checkpoint["optimizer"]["state"].values())
    # A direct call without extra judge measurements must give identical training.
    cfg, initial, _, payload, _ = load_source(saved)
    weak, *_ = load_plain(plain, payload, cfg, 17)
    final = {}
    def capture(step, model, optimizer, rng, history):
        final.update(model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng.get_state().clone())
    policy_train("dgpo", initial, weak, Distribution(cfg), cfg, 17, 90017, lambda _:None, capture,
                 endpoint_controller=EndpointRatio() if mode == "endpoint" else None,
                 velocity_coefficient=1. if mode == "velocity" else 0.)
    for key in ("model", "optimizer", "rng"):
        assert_nested_equal(checkpoint[key], final[key])


@pytest.mark.parametrize("change", ["report_coefficient", "state_coefficient", "incomplete", "decision"])
def test_velocity_control_rejects_unmatched_or_incomplete(saved, plain, tmp_path, change):
    fixed_run(saved, tmp_path/"no_kl")
    control = tmp_path/"velocity"
    velocity_run(saved, tmp_path/"no_kl", control, policy_steps=2)
    report = json.loads((control/"report.json").read_text())
    state = torch.load(control/"dgpo.pt", weights_only=True)
    if change == "report_coefficient":
        report["velocity_coefficient"] = .5
    elif change == "state_coefficient":
        state["velocity_coefficient"] = 0.
    elif change == "incomplete":
        report["policy_histories"]["velocity_kl"].pop()
    else:
        report["decision"] = "running"
    (control/"report.json").write_text(json.dumps(report))
    torch.save(state, control/"dgpo.pt")
    cfg, *_ = load_source(saved)
    with pytest.raises(ValueError):
        load_control(control, saved, cfg, 17, "velocity")


def test_bad_setup_does_not_train(saved, plain, tmp_path, monkeypatch):
    control = tmp_path/"control"
    fixed_run(saved, control)
    import experiments.dgpo_toy.weak_classifier as module
    monkeypatch.setattr(module, "setup_gates", lambda *a:{"higher_ess": False})
    monkeypatch.setattr(module, "policy_train", lambda *a, **k: pytest.fail("Invalid setup must not train"))
    # Keep the independent setup panel small without granting smoke's bypass.
    original = module.make_panel
    monkeypatch.setattr(module, "make_panel", lambda m,d,c,n,s: original(m,d,c,min(n,64),s))
    result = run(saved, control, tmp_path/"bad", plain_checkpoint=plain, regularization="none", policy_steps=2)
    assert result["state"] == "stopped_setup" and result["completed_steps"] == 0
    assert not (tmp_path/"bad"/"dgpo_state.pt").exists()


def test_mislabelled_plain_weights_fail_validation(saved, plain, tmp_path):
    control = tmp_path/"control"
    fixed_run(saved, control)
    bundle = torch.load(plain, weights_only=True)
    bundle["classifiers"]["plain"]["head.bias"].add_(1.)
    torch.save(bundle, plain)
    with pytest.raises(ValueError, match="reproduce"):
        run(saved, control, tmp_path/"bad", plain_checkpoint=plain, regularization="none", policy_steps=2)


def test_post_update_density_failure_preserves_previous_full_state(saved, plain, tmp_path, monkeypatch):
    fixed_run(saved, tmp_path/"no_kl")
    endpoint_run(saved, tmp_path/"no_kl", tmp_path/"control", policy_steps=2)
    original = EndpointRatio.refresh
    def fail(self, *args):
        if self.policy_step == 2:
            raise DensityValidityError("injected post-update failure")
        return original(self, *args)
    monkeypatch.setattr(EndpointRatio, "refresh", fail)
    result = run(saved, tmp_path/"control", tmp_path/"stopped", plain_checkpoint=plain, policy_steps=2, smoke=True)
    state = torch.load(tmp_path/"stopped"/"dgpo_state.pt", weights_only=True)
    assert result["state"] == "stopped_density_validity" and result["completed_steps"] == 1
    assert state["step"] == len(state["history"]) == state["endpoint_controller"]["policy_step"] == 1
    assert all(float(x["step"]) == 1 for x in state["optimizer"]["state"].values())
    assert not result["comparison_budgets_matched"]


def test_success_uses_common_judge_and_marginals_do_not_veto_pure_transfer():
    endpoints = {"initial": {"structure": {"fourier_moment_rmse": .5}},
        "weak_policy": {"strong_gain": {"gain": .3, "lo95": .2},
            "structure_change": {"hi95": -.01},
            "structure": {"fourier_moment_rmse": .3, "marginal_mean_absmax": 5.,
                          "marginal_variance_error_absmax": 5., "pair_covariance_absmax": 5.}}}
    comparisons = {"strong_reward": {"lo95": .1}, "structure": {"hi95": -.01}}
    assert endpoint_decision(endpoints, comparisons, "none")[0] == "supports_weak_signal_structural_progress"
    assert endpoint_decision(endpoints, comparisons, "endpoint")[0] == "supports_weak_signal_reward_transfer"
    endpoints["weak_policy"]["strong_gain"]["gain"] = 0.
    assert endpoint_decision(endpoints, comparisons, "none")[0] == "no_decisive_weak_signal_advantage"


def test_source_and_plain_output_protected(saved, plain):
    for out in (saved.parent, plain.parent):
        with pytest.raises(ValueError, match="separate"):
            run(saved, saved.parent/"control", out, plain_checkpoint=plain)
