"""Estimator sign, full DDIM gradients, checkpoint and distributed contracts."""
import copy
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

# Avoid importing optional graphics dependencies through timing decorators.
debug = types.ModuleType("evenet.utilities.debug_tool")
debug.time_decorator = lambda **kwargs: lambda function: function
sys.modules.setdefault("evenet.utilities.debug_tool", debug)

from RL.DGPO_neutrino.endpoint_kl import (
    EndpointKLConfig, EndpointKLController, validate_endpoint_protocol,
)
from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
from evenet.utilities.diffusion_sampler import DDIMSampler
from evenet.network.body.normalizer import Normalizer


class ToyPolicy(nn.Module):
    invisible_input_dim = 1

    def __init__(self, shift=0.2):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(shift))
        self.invisible_normalizer = Normalizer(torch.tensor([0.3]), torch.tensor([1.7]), torch.tensor([True]))

    def predict_diffusion_vector(self, noise_x, time, **kwargs):
        return noise_x * 0.1 + self.shift * time[:, None, None]

    def forward(self, x):
        return self.shift * x


def batch(n):
    return {"x": torch.zeros(n, 1, 1), "x_mask": torch.ones(n, 1),
            "conditions": torch.zeros(n, 1), "conditions_mask": torch.ones(n),
            "x_invisible": torch.full((n, 1, 1), 999.), "x_invisible_mask": torch.ones(n, 1)}


@pytest.mark.parametrize("checkpoint_steps", [False, True])
@pytest.mark.parametrize("x0_mode", ["legacy", "stable_v"])
def test_full_ddim_gradient_matches_finite_difference(checkpoint_steps, x0_mode):
    model = ToyPolicy()
    sampler = DDIMSampler(torch.device("cpu"), x0_mode=x0_mode)
    def evaluate(differentiable):
        torch.manual_seed(12)
        return generate_neutrino_candidates(model, batch(4), sampler, K=1,
            num_ddim_steps=5, device=torch.device("cpu"),
            differentiable=differentiable, checkpoint_steps=checkpoint_steps).mean()
    value = evaluate(True)
    grad = torch.autograd.grad(value, model.shift)[0]
    assert grad.abs() > 0.01
    with torch.no_grad():
        old = model.shift.clone()
        model.shift.copy_(old + 0.002)
        plus = evaluate(False)
        model.shift.copy_(old - 0.002)
        minus = evaluate(False)
        model.shift.copy_(old)
    # Legacy high-noise subtraction has appreciable float32 round-off.
    torch.testing.assert_close(grad, (plus - minus) / 0.004, rtol=0.03, atol=0.03)
    assert not evaluate(False).requires_grad


def test_normalizer_candidate_gradient_and_values():
    normalizer = Normalizer(torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4]), torch.tensor([True, True]), inv_cdf_index=[1])
    x = torch.tensor([[[0.3, 0.4]]], requires_grad=True)
    y = normalizer.forward_grad(x)
    torch.testing.assert_close(y, normalizer(x.detach()))
    gradient = torch.autograd.grad(y.sum(), x)[0]
    assert torch.isfinite(gradient).all() and (gradient.abs() > 0).all()
    torch.testing.assert_close(normalizer.denormalize_grad(y), normalizer.denormalize(y.detach()))


class LinearCritic(nn.Module):
    packing_spec = None

    def __init__(self):
        super().__init__()
        self.output = nn.Linear(1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, condition, candidate):
        return self.output(candidate).squeeze(-1)


class ToySource:
    reward_round_id = 1

    def make_endpoint_kl_classifier(self):
        return LinearCritic()


def gaussian_sample(model, b, sampler, **kwargs):
    assert torch.count_nonzero(b["x_invisible"]) == 0  # no truth leakage
    with torch.set_grad_enabled(kwargs.get("differentiable", False)):
        return (model.shift + torch.randn(1, len(b["x"]), 1, 1))


def pack(b, spec):
    return b["x"].flatten(1), None


def run_controller(controller, model, reference, step=0):
    with patch("RL.DGPO_neutrino.sampling.generate_neutrino_candidates", gaussian_sample), \
         patch("RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio.pack_event_inputs", pack):
        c = controller.config
        return controller.backward(model=model, reference=reference, source=ToySource(),
            batch=batch(c.fit_events_per_rank+c.validation_events_per_rank+c.actor_events_per_rank),
            sampler=None, num_ddim_steps=20, device=torch.device("cpu"), global_step=step,
            reduce_gradients=lambda model: None)


def test_learned_gaussian_kl_has_correct_actor_gradient_and_resume():
    # q=N(mu,1), reference=N(0,1): true KL=mu^2/2 and dKL/dmu=mu.
    cfg = EndpointKLConfig(enabled=True, bootstrap_steps=100, fit_steps=10,
        fit_events_per_rank=2048, validation_events_per_rank=1024, actor_events_per_rank=1024,
        microbatch_size=128, learning_rate=0.08, backbone_learning_rate=0.08,
        weight_decay=0.)
    model, reference = ToyPolicy(1.), ToyPolicy(0.)
    reference.requires_grad_(False)
    controller = EndpointKLController(cfg)
    rng = torch.random.get_rng_state().clone()
    metrics = run_controller(controller, model, reference)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert 0.65 < model.shift.grad.item() < 1.4
    assert 0.2 < metrics["endpoint_kl/estimate"] < 0.8
    assert metrics["endpoint_kl/validation_bce"] < 0.66
    assert reference.shift.grad is None
    assert all(p.grad is None for p in controller.params)
    assert model.training
    state = copy.deepcopy(controller.state_dict())
    resumed = EndpointKLController(cfg, state)
    model.shift.grad = None
    next_metrics = run_controller(controller, model, reference, step=1)
    grad = model.shift.grad.clone()
    model.shift.grad = None
    resumed_metrics = run_controller(resumed, model, reference, step=1)
    torch.testing.assert_close(model.shift.grad, grad)
    assert next_metrics["endpoint_kl/estimate"] == resumed_metrics["endpoint_kl/estimate"]
    assert resumed.updates == controller.updates


def test_off_does_no_sampling_or_initialization():
    controller = EndpointKLController(EndpointKLConfig())
    metrics = controller.backward(model=None, reference=None, source=None, batch=None,
        sampler=None, num_ddim_steps=20, device=torch.device("cpu"),
        global_step=0, reduce_gradients=None)
    assert metrics == {} and controller.classifier is None


def test_config_and_switches():
    assert not EndpointKLConfig.parse().enabled
    with pytest.raises(ValueError):
        EndpointKLConfig.parse({"enabled": "false"})
    with pytest.raises(ValueError):
        EndpointKLConfig.parse({"microbatch_size": 0})
    with pytest.raises(ValueError):
        validate_endpoint_protocol(EndpointKLConfig(enabled=True), {})
    validate_endpoint_protocol(EndpointKLConfig(), {})
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from train_dgpo_endpoint_kl import make_config
    on, off = make_config(True), make_config(False)
    assert on["dgpo"]["endpoint_kl"]["enabled"]
    assert not off["dgpo"]["endpoint_kl"]["enabled"]
    assert on["logger"]["wandb"]["id"] != off["logger"]["wandb"]["id"]
    assert on["options"]["Training"]["model_checkpoint_save_path"] != off["options"]["Training"]["model_checkpoint_save_path"]
    assert on["platform"]["number_of_workers"] == 16


def test_16_rank_gradient_weighting():
    # Equal per-rank actor budgets: averaging local means equals global mean.
    x = torch.arange(16 * 8, dtype=torch.float64).reshape(16, 8) / 128
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    local = [torch.autograd.grad((theta * row).square().mean(), theta)[0] for row in x]
    global_grad = torch.autograd.grad((theta * x).square().mean(), theta)[0]
    torch.testing.assert_close(torch.stack(local).mean(), global_grad)


def test_full_resume_launcher_preserves_training_setup():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from train_dgpo_endpoint_kl import make_config
    fresh = make_config()
    resumed = make_config(resume=True, total_steps=1000)
    assert resumed['dgpo']['checkpoint_load_mode'] == 'resume'
    assert resumed['dgpo']['auto_resume_from_last']
    assert resumed['dgpo']['endpoint_kl'] == fresh['dgpo']['endpoint_kl']
    assert resumed['dgpo']['adaptive_omnifold']['audit_fit'] == fresh['dgpo']['adaptive_omnifold']['audit_fit']
    assert not resumed['dgpo']['adaptive_omnifold']['recalibration']['bootstrap_on_start']
    assert resumed['options']['Training']['epochs'] == 100
    assert resumed['options']['Training']['model_checkpoint_load_path'].endswith('/h4_endpoint_kl_on/checkpoints/last.ckpt')
    assert resumed['logger']['wandb']['id'] == fresh['logger']['wandb']['id']
    assert resumed['logger']['wandb']['resume'] == 'must'
    unlimited = make_config(resume=True)
    assert unlimited['dgpo']['unbounded_training'] is True
    assert unlimited['experiment']['policy_updates_per_round'] is None
    assert unlimited['dgpo']['validation_every_n_epochs'] == 10
    assert unlimited['dgpo']['adaptive_omnifold']['staleness_every_n_epochs'] == 10
    assert not resumed['dgpo'].get('unbounded_training', False)
    with pytest.raises(ValueError):
        make_config(resume=True, total_steps=105)


@pytest.mark.parametrize("enabled", [False, True])
def test_reward_and_audit_use_latest_classifier_setup(enabled):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from train_dgpo_endpoint_kl import make_config
    from train_h4_output_scaling import validated_config
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
    from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _scaled_crossfit_config
    from dataclasses import asdict
    c = make_config(enabled=enabled)
    runtime = resolve_adaptive_config(c["dgpo"])
    latest = validated_config("ratio-health")["dgpo"]["adaptive_omnifold"]["audit_fit"]
    expected = build_fit_config(latest, n_train=250000, n_validation=100000)
    expected = replace(expected, min_steps=300, representation_export_dir=None,
                       diagnostic_snapshot_dir=None, ratio_audit_export_dir=None)
    for block in (runtime.fit, runtime.audit_fit):
        fit = build_fit_config(block, n_train=250000, n_validation=100000)
        fit.validate()
        assert asdict(fit) == asdict(expected)
    # Architecture hand-off: reward builder reads recalibration, audit overrides
    # read audit_fit. Compare every shared architecture setting to latest H4.
    recal = c["dgpo"]["adaptive_omnifold"]["recalibration"]
    keys = [k for k in latest if k.startswith("topology_") or k.startswith("train_")]
    keys += ["decoder_hidden_dim", "decoder_layers", "decoder_heads", "head_dropout",
             "periodic_pair_features", "visible_pair_rest_frame", "asymmetric_attention"]
    for key in keys:
        if key in recal:
            assert recal[key] == latest[key] == runtime.audit_fit[key], key
    assert runtime.crossfit_folds == 2 and runtime.crossfit_repeats == 1
    assert not runtime.baseline_probe_on_start
    from RL.DGPO_neutrino.monitoring import validation_schedule_tier
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import should_probe_epoch
    dgpo = c["dgpo"]
    assert runtime.staleness_every_n_steps is None
    assert runtime.staleness_every_n_epochs == 10
    assert dgpo["steps_per_epoch"] == 10
    assert dgpo["gradient_conflict"]["every_n_steps"] == 100
    assert c["experiment"]["cold_h4_audit_policy_steps"] == [100]
    for epoch in range(10):
        audit_due = should_probe_epoch(epoch, runtime.staleness_every_n_epochs)
        tier = validation_schedule_tier(epoch,
            cheap_every_n_epochs=dgpo["validation_every_n_epochs"],
            full_every_n_epochs=dgpo["validation_full_every_n_epochs"])
        assert audit_due == (epoch == 9)
        assert tier == ("full" if epoch == 9 else None)
    assert runtime.audit_fit["validation_interval_epochs"] == 1
    assert runtime.fit["validation_interval_epochs"] == 1
    for fraction in (0.49, 0.5, 0.51):
        fold = _scaled_crossfit_config(expected, fraction,
            min_steps_per_fold=300, max_steps_per_fold=3000,
            n_train=125000, validation_interval_epochs=1,
            validation_patience_epochs=10)
        assert (fold.min_steps, fold.steps) == (300, 3000)
        assert fold.validation_patience_evaluations == 10
        assert fold.validation_interval_steps == 125000 // fold.batch_size
        assert fold.fourier_output_standardization
    legacy = _scaled_crossfit_config(expected, 0.5, min_steps_per_fold=300)
    assert legacy.steps == 1500  # Other experiments keep their original behavior.


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_reject_invalid_absolute_fold_cap(cap):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _scaled_crossfit_config
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig
    with pytest.raises(ValueError, match="max_steps_per_fold"):
        _scaled_crossfit_config(RatioFitConfig(), .5, max_steps_per_fold=cap)


def test_h4_candidate_gradient_through_normalizer_without_fourier():
    from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
    packed, spec = pack_event_inputs(_event_batch(2))
    backbone = _FakeZtautauBackbone()
    backbone.invisible_normalizer = Normalizer(torch.zeros(2), torch.ones(2), torch.ones(2, dtype=torch.bool))
    classifier = EvenetAdapterRatioClassifier(backbone, spec)
    classifier.endpoint_candidate_grad = True
    with torch.no_grad():
        classifier.bank.output.weight.fill_(0.1)
    params = list(classifier.parameters())
    for p in params:
        p.requires_grad_(False)
    classifier.eval()
    candidate = torch.randn(2, 4, requires_grad=True)
    gradient = torch.autograd.grad(classifier(packed, candidate).sum(), candidate)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_endpoint_state_is_in_policy_checkpoint_and_logging_survives_filters():
    from RL.DGPO_neutrino.model_utils import build_lightning_compatible_checkpoint
    from RL.DGPO_neutrino.dgpo_trainer import _wandb_train_payload, _wandb_critical_keep, _wandb_simplified_keep
    model = ToyPolicy()
    controller = EndpointKLController(EndpointKLConfig(enabled=True))
    model._endpoint_kl_controller = controller
    config = types.SimpleNamespace(options=types.SimpleNamespace(Training={"EMA": {"enable": False}}))
    saved = build_lightning_compatible_checkpoint(model, None, config)
    assert saved["dgpo_endpoint_kl_state"]["config"]["enabled"]
    assert all("endpoint" not in key for key in saved["state_dict"])
    key = "endpoint_kl/estimate"
    assert _wandb_train_payload({key: 1.})[key] == 1.
    assert _wandb_critical_keep(key) and _wandb_simplified_keep(key, 1.)


def _distributed_worker(rank, rendezvous):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from RL.DGPO_neutrino.dgpo_trainer import _all_reduce_accumulated_gradients
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = ToyPolicy(1.)
        wrapped = DistributedDataParallel(model)
        reference = ToyPolicy(0.).requires_grad_(False)
        cfg = EndpointKLConfig(enabled=True, bootstrap_steps=40, fit_steps=5,
            fit_events_per_rank=256, validation_events_per_rank=128, actor_events_per_rank=64,
            microbatch_size=32, learning_rate=0.08, backbone_learning_rate=0.08, weight_decay=0.)
        controller = EndpointKLController(cfg)
        with wrapped.no_sync():
            wrapped(torch.tensor(float(1 + 2 * rank))).backward()
        with wrapped.no_sync(), \
             patch("RL.DGPO_neutrino.sampling.generate_neutrino_candidates", gaussian_sample), \
             patch("RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio.pack_event_inputs", pack):
            controller.backward(model=model, reference=reference, source=ToySource(),
                batch=batch(448), sampler=None, num_ddim_steps=20,
                device=torch.device("cpu"), global_step=0,
                reduce_gradients=lambda m: _all_reduce_accumulated_gradients(m, world_size=2))
        _all_reduce_accumulated_gradients(model, world_size=2)
        # Main gradient mean=2; Gaussian critic actor gradient is its slope.
        slope = controller.classifier.output.weight.item()
        torch.testing.assert_close(model.shift.grad, torch.tensor(2. + slope))
        state = torch.cat([p.detach().flatten() for p in controller.params])
        gathered = [torch.empty_like(state) for _ in range(2)]
        dist.all_gather(gathered, state)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(__import__("os").environ.get("RUN_ENDPOINT_DISTRIBUTED") != "1", reason="explicit multiprocess CPU check")
def test_two_rank_ddp_main_plus_endpoint_gradient(tmp_path):
    import torch.multiprocessing as mp
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    mp.spawn(_distributed_worker, args=(rendezvous,), nprocs=2, join=True)
