"""Direct density mathematics and unchanged DGPO plumbing."""
import copy
from dataclasses import replace
import math

import pytest
import torch

from experiments.dgpo_toy.conditional import Denoiser, Distribution, ddim, generator, initialize, policy_train, dgpo_objective
from experiments.dgpo_toy.direct_density import DDIMDensity, DensitySettings, DensityValidityError, EndpointRatio
from experiments.dgpo_toy.low_ess_recovery import neural_log_density
from experiments.dgpo_toy.fixed_reward import load_source, run as fixed_run
from experiments.dgpo_toy.net_reward_kl import run
from experiments.dgpo_toy.test_fixed_reward import saved, assert_nested_equal
from experiments.dgpo_toy.test_conditional import small_config
from experiments.dgpo_toy.test_velocity_kl import objective_inputs


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def density_fixture():
    cfg = small_config()
    model = initialize(Denoiser, cfg, 17)
    data = Distribution(cfg)
    rng = generator(21)
    c = data.contexts(7, rng).double()
    z = torch.randn(7, cfg.dimensions, generator=rng, dtype=torch.float64)
    return cfg, model, c, z


def test_analytic_jacobian_inverse_and_existing_independent_density():
    cfg, model, c, z = density_fixture()
    backend = DDIMDensity(model, cfg.ddim_steps)
    assert max(backend.validate_autodiff(c, z).values()) < 1e-12
    y, ld, _ = backend.forward_density(z, c)
    inverse, info = backend.inverse(y, c)
    torch.testing.assert_close(inverse, z, rtol=1e-9, atol=1e-9)
    lp, diag = backend.log_prob(y, c)
    torch.testing.assert_close(lp, -.5*(z.square()+math.log(2*math.pi)).sum(-1)-ld, rtol=1e-9, atol=1e-9)
    independent, _ = neural_log_density(backend.model, y, c, torch.zeros_like(y), cfg.ddim_steps)
    torch.testing.assert_close(lp, independent, rtol=1e-9, atol=1e-9)
    assert diag["forward_residual_max"] < 1e-8


def test_affine_gaussian_ground_truth_and_ratio_orientation():
    cfg, model, c, z = density_fixture()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[-1].bias.fill_(.3)
    backend = DDIMDensity(model, cfg.ddim_steps)
    scale = torch.tensor(1., dtype=torch.float64)
    mean = torch.zeros(cfg.dimensions, dtype=torch.float64)
    for _, a, b, s in backend.schedule:
        scale = a*scale
        mean = a*mean+b*s*float(model.network[-1].bias[0])
    y = scale*z+mean
    lp, _ = backend.log_prob(y, c)
    exact = -.5*(((y-mean)/scale).square()+math.log(2*math.pi)).sum(-1)-cfg.dimensions*scale.log()
    torch.testing.assert_close(lp, exact, rtol=1e-9, atol=1e-9)
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.network[-1].bias.zero_()
    ratio = EndpointRatio()
    ratio.refresh(model, ref, cfg.ddim_steps)
    actual = ratio(y, c)
    expected = exact-(-.5*((y/scale).square()+math.log(2*math.pi)).sum(-1)-cfg.dimensions*scale.log())
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)


def test_invalid_map_and_inverse_fail_closed():
    cfg, model, c, z = density_fixture()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(100)
    with pytest.raises(DensityValidityError, match="not certified"):
        DDIMDensity(model, cfg.ddim_steps)
    cfg, model, c, z = density_fixture()
    backend = DDIMDensity(model, cfg.ddim_steps, DensitySettings(inverse_iterations=1))
    with pytest.raises(DensityValidityError, match="error tolerance"):
        backend.log_prob(z, c)


def test_exact_reward_cancels_at_target():
    cfg, model, c, z = density_fixture()
    ref = copy.deepcopy(model)
    with torch.no_grad():
        model.network[-1].bias.add_(.2)
    ratio = EndpointRatio()
    ratio.refresh(model, ref, cfg.ddim_steps)
    y, _, _ = ratio.current.forward_density(z, c)
    # Here p IS current: independently evaluated log p - log reference.
    lp, _ = neural_log_density(ratio.current.model, y, c, torch.zeros_like(y), cfg.ddim_steps)
    lr, _ = neural_log_density(ratio.reference.model, y, c, torch.zeros_like(y), cfg.ddim_steps)
    net = lp-lr-ratio(y, c)
    assert float(net.abs().max()) < 1e-9


def test_zero_density_preserves_loss_gradient_rng_and_detaches():
    args = objective_inputs()
    model, ref, critic, data, cfg, *_ = args
    ratio = EndpointRatio()
    state = torch.random.get_rng_state().clone()
    ratio.update(model, ref, data, cfg, 1)
    loss, diag = dgpo_objective(*args)
    other, other_diag = dgpo_objective(*args, endpoint_ratio=ratio)
    assert torch.equal(loss, other)
    assert other_diag["endpoint_kl_score_mean"] == other_diag["endpoint_kl_score_std"] == 0
    for a, b in zip(torch.autograd.grad(loss, tuple(model.parameters())),
                    torch.autograd.grad(other, tuple(model.parameters()))):
        assert torch.equal(a, b)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(p.grad is None for module in (ref, critic, ratio.current.model, ratio.reference.model) for p in module.parameters())
    assert args[6].grad is None


def test_nonzero_ratio_detached_and_disabled_branch_unchanged():
    args = objective_inputs()
    with torch.no_grad():
        args[0].network[-1].bias.add_(.1)
    ratio = EndpointRatio()
    ratio.update(args[0], args[1], args[3], args[4], 1)
    base, _ = dgpo_objective(*args)
    off, _ = dgpo_objective(*args, endpoint_ratio=ratio, endpoint_coefficient=0.)
    assert torch.equal(base, off)
    on, diag = dgpo_objective(*args, endpoint_ratio=ratio)
    assert diag["endpoint_kl_score_std"] > 0
    on.backward()
    assert args[6].grad is None
    assert all(p.grad is None for module in (args[1], args[2], ratio.current.model, ratio.reference.model) for p in module.parameters())


def test_full_state_resume_exact(saved):
    cfg, model, critic, _, _ = load_source(saved)
    cfg = replace(cfg, policy_steps=4, eval_every=2)
    data = Distribution(cfg)
    def train(config, resume=None):
        controller = EndpointRatio()
        snapshots = []
        def checkpoint(step, model, optimizer, rng, history):
            snapshots.append(copy.deepcopy({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": rng.get_state(), "history": history,
                "endpoint_controller": controller.state_dict()}))
        policy_train("dgpo", model, critic, data, config, 17, 90017, lambda _: None,
                     checkpoint, resume, endpoint_controller=controller)
        return snapshots
    full = train(cfg)
    short = train(replace(cfg, policy_steps=2))
    resumed = train(cfg, short[-1])
    assert_nested_equal(full[-1], resumed[-1])


def test_runner_completion_and_validity_stop(saved, tmp_path, monkeypatch):
    before = saved.read_bytes()
    control = tmp_path/"control"
    fixed_run(saved, control)
    completed = run(saved, control, tmp_path/"completed", policy_steps=2)
    assert completed["state"] == "completed" and completed["completed_steps"] == 2
    assert completed["first_update_matches_no_kl"] and completed["frozen_source_unchanged"]
    assert saved.read_bytes() == before
    original = EndpointRatio.update
    def fail(self, *args):
        if args[-1] == 2:
            raise DensityValidityError("injected validity failure")
        return original(self, *args)
    monkeypatch.setattr(EndpointRatio, "update", fail)
    stopped = run(saved, control, tmp_path/"stopped", policy_steps=2)
    assert stopped["state"] == "stopped_density_validity" and stopped["completed_steps"] == 1
    assert stopped["decision"] == "unresolved_density_validity_stop"
    assert not stopped["comparison_budgets_matched"]
    state = torch.load(tmp_path/"stopped"/"dgpo_state.pt", weights_only=True)
    assert len(state["history"]) == state["step"] == state["endpoint_controller"]["policy_step"] == 1
    assert all(float(s["step"]) == 1 for s in state["optimizer"]["state"].values())


def test_post_update_failure_retains_previous_full_state(saved, tmp_path, monkeypatch):
    control = tmp_path/"control"
    fixed_run(saved, control)
    original = EndpointRatio.refresh
    def fail_after_second(self, *args):
        # update refreshes BEFORE increment; checkpoint refreshes AFTER.
        if self.policy_step == 2:
            raise DensityValidityError("injected post-update failure")
        return original(self, *args)
    monkeypatch.setattr(EndpointRatio, "refresh", fail_after_second)
    result = run(saved, control, tmp_path/"post_failure", policy_steps=2)
    state = torch.load(tmp_path/"post_failure"/"dgpo_state.pt", weights_only=True)
    attempted = torch.load(tmp_path/"post_failure"/"uncertified_attempt.pt", weights_only=True)
    assert state["step"] == result["completed_steps"] == 1
    assert attempted["step"] == result["stop"]["attempted_step"] == 2
    assert state["endpoint_controller"]["policy_step"] == 1
    assert all(float(s["step"]) == 1 for s in state["optimizer"]["state"].values())
    assert len(state["history"]) == 1


def test_runner_full_resume_preserves_updates(saved, tmp_path):
    short_control, full_control = tmp_path/"control2", tmp_path/"control4"
    fixed_run(saved, short_control)
    fixed_run(saved, full_control, policy_steps=4)
    run(saved, short_control, tmp_path/"short", policy_steps=2)
    run(saved, full_control, tmp_path/"full", policy_steps=4)
    result = run(saved, full_control, tmp_path/"resume", policy_steps=4, resume_from=tmp_path/"short")
    full = torch.load(tmp_path/"full"/"dgpo_state.pt", weights_only=True)
    resumed = torch.load(tmp_path/"resume"/"dgpo_state.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "endpoint_controller"):
        assert_nested_equal(full[key], resumed[key])
    # A shorter run adds terminal-only measurements, never training changes.
    for a, b in zip(full["history"], resumed["history"]):
        for key in a.keys() & b.keys():
            assert a[key] == b[key]
    assert result["state"] == "completed" and result["completed_steps"] == 4


def test_continuation_with_old_controls_labels_budgets_and_preserves_state(saved, tmp_path):
    control = tmp_path/"control2"
    fixed_run(saved, control)
    from experiments.dgpo_toy.velocity_kl import run as velocity_run
    velocity_run(saved, control, tmp_path/"velocity2", policy_steps=2)
    run(saved, control, tmp_path/"short", policy_steps=2)
    original = (tmp_path/"short"/"dgpo_state.pt").read_bytes()
    result = run(saved, control, tmp_path/"extended", policy_steps=4, comparator_steps=2,
                 velocity_comparator=tmp_path/"velocity2", resume_from=tmp_path/"short", endpoint_seed=97017)
    # Compare against an uninterrupted trajectory with the same short controls.
    run(saved, control, tmp_path/"full", policy_steps=4, comparator_steps=2, endpoint_seed=97017)
    full = torch.load(tmp_path/"full"/"dgpo_state.pt", weights_only=True)
    continued = torch.load(tmp_path/"extended"/"dgpo_state.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "endpoint_controller"):
        assert_nested_equal(full[key], continued[key])
    assert original == (tmp_path/"short"/"dgpo_state.pt").read_bytes()
    assert result["start_step"] == result["new_updates_completed"] == result["requested_new_updates"] == 2
    assert result["completed_steps"] == 4 and not result["comparison_budgets_matched"]
    assert result["resume_verification"]["monitor_boundary_exact"]
    assert result["resume_verification"]["history_prefix_exact"]
    assert {k:v["step"] for k,v in result["endpoints"].items()} == {
        "initial": 0, "resume_start": 2, "no_kl": 2, "velocity_kl": 2, "net_reward_kl": 4}
    assert "gain_since_resume" in result["endpoints"]["net_reward_kl"]
    assert "structure_change_since_resume" in result["endpoints"]["net_reward_kl"]
    assert not any(x["matched_update_budget"] for x in result["comparisons"].values())
    import json
    progress=[json.loads(x) for x in (tmp_path/"extended"/"progress.jsonl").read_text().splitlines()]
    assert [x["step"] for x in progress if x["phase"]=="policy"] == [3,4]
    assert next(x for x in progress if x["phase"]=="resume_first_update_verified")["step"] == 3


def test_resume_rejects_boundary_clock_and_changed_density_settings(saved, tmp_path):
    control = tmp_path/"control"
    fixed_run(saved, control)
    run(saved, control, tmp_path/"short", policy_steps=2)
    with pytest.raises(ValueError, match="direct-density settings"):
        run(saved, control, tmp_path/"bad_coefficient", policy_steps=4, comparator_steps=2,
            resume_from=tmp_path/"short", settings=DensitySettings(coefficient=.5))
    checkpoint=tmp_path/"short"/"dgpo_state.pt"
    state=torch.load(checkpoint,weights_only=True)
    state["history"][-1]["monitor_reward"] += 1.
    torch.save(state,checkpoint)
    with pytest.raises(ValueError,match="boundary monitor"):
        run(saved,control,tmp_path/"bad_monitor",policy_steps=4,comparator_steps=2,resume_from=tmp_path/"short")
