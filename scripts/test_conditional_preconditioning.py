"""CPU numerical/integration contracts; does not execute a real-case experiment."""
import copy
import math
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))
from evenet.network.body.normalizer import Normalizer
from evenet.network.body.conditional_preconditioning import (
    ConditionalPreconditioner, ConditionalTargetNormalizer, file_sha256)
from train_neutrino_backend import read_overlay_yaml, read_yaml, deep_update
from run_conditional_preconditioning import validate_config, validate_calibration_data


def normalizer(width, std=1.):
    return Normalizer(torch.zeros(width), torch.full((width,), float(std)), torch.ones(width, dtype=torch.bool))


def make_preconditioner(**kwargs):
    return ConditionalPreconditioner(["Part_eta", "Part_phi", "Part_energy", "Part_pt"],
                                    normalizer(4), normalizer(2), normalizer(2, .2), width=16, **kwargs)


def batch(n=8):
    torch.manual_seed(14)
    raw = torch.randn(n, 5, 4)
    valid = torch.ones(n, 5, dtype=torch.bool)
    valid[:, -1] = False
    return dict(x=raw, x_mask=valid, conditions=torch.randn(n, 2),
                conditions_mask=torch.ones(n, 1, dtype=torch.bool),
                x_invisible=torch.randn(n, 2, 2) * .2,
                x_invisible_mask=torch.ones(n, 2, dtype=torch.bool))


def open_head(model):
    with torch.no_grad():
        model.output.weight.normal_(std=.15)
        model.output.bias.add_(torch.linspace(-.4, .4, 14))
        model.fitted.fill_(True)


def test_identity_initialization_rng_and_invertible_full_covariance():
    torch.manual_seed(21)
    rng = torch.random.get_rng_state()
    model = make_preconditioner()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.identity_initialized()
    b = batch()
    mu, factor = model(b)
    torch.testing.assert_close(mu, torch.zeros_like(mu), rtol=0, atol=0)
    torch.testing.assert_close(factor, torch.eye(4).expand(len(mu), -1, -1), atol=2e-7, rtol=0)
    with pytest.raises(RuntimeError, match="Fit/load"):
        model.coordinates(b)
    open_head(model)
    mu, factor = model.coordinates(b)
    covariance = factor @ factor.mT
    assert covariance[:, 0, 2].abs().max() > 0  # actual cross-tau correlation
    assert (torch.linalg.eigvalsh(covariance.double()) > 0).all()
    target = torch.randn(len(mu), 2, 2)
    z = model.encode(target, mu, factor)
    torch.testing.assert_close(model.decode(z, mu, factor), target, atol=2e-6, rtol=2e-6)


def test_covariance_floor_survives_extreme_outputs():
    model = make_preconditioner().double()
    b = {k: v.double() if v.is_floating_point() else v for k, v in batch().items()}
    with torch.no_grad():
        model.output.bias[4:] = torch.tensor([-100., 100., -100., 100., -100., -100., 100., -100., 100., -100.])
    _, factor = model(b)
    eigenvalues = torch.linalg.eigvalsh(factor @ factor.mT)
    assert eigenvalues.min() >= model.spec["scale_floor"] ** 2 * .999999
    assert torch.isfinite(factor).all()


def test_only_observations_permutation_periodicity_masks_and_bad_inputs():
    model = make_preconditioner()
    open_head(model)
    b = batch()
    expected = model.coordinates(b)
    # Targets, assignments and noisy candidates cannot change coordinates.
    changed = {**b, "x_invisible": torch.full_like(b["x_invisible"], float("nan")),
               "assignments-indices": torch.randn(8, 7), "noise_x": torch.randn(8, 2, 2)}
    for old, new in zip(expected, model.coordinates(changed)):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    permuted = {**b, "x": b["x"].flip(1), "x_mask": b["x_mask"].flip(1)}
    periodic = copy.deepcopy(b)
    periodic["x"][..., 1] += 2 * math.pi
    padded = copy.deepcopy(b)
    padded["x"][:, -1] = float("nan")
    for candidate in (permuted, periodic, padded):
        for old, new in zip(expected, model.coordinates(candidate)):
            torch.testing.assert_close(old, new, atol=2e-6, rtol=2e-6)
    empty = {**b, "x_mask": torch.zeros_like(b["x_mask"]), "x": torch.full_like(b["x"], float("nan"))}
    assert all(torch.isfinite(x).all() for x in model.coordinates(empty))
    invalid = copy.deepcopy(b)
    invalid["x"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="observed"):
        model(invalid)
    invalid = copy.deepcopy(b)
    invalid["x_invisible_mask"][0, 0] = False
    with pytest.raises(ValueError, match="both tau"):
        model.gaussian_loss(invalid)


def test_gaussian_fit_gradient_learns_and_frozen_diffusion_roundtrip():
    model = make_preconditioner()
    model.requires_grad_(True)
    b = batch(32)
    # Numerical unit fixture, not a scientific toy experiment.
    b["x_invisible"][:, 1, :] = .8 * b["x_invisible"][:, 0, :] + .01 * torch.randn(32, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    before = model.gaussian_loss(b)[0].mean().item()
    for _ in range(30):
        nll, _, _ = model.gaussian_loss(b, coordinates=model(b))
        optimizer.zero_grad()
        nll.mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert model.gaussian_loss(b)[0].mean().item() < before - .5
    model.fitted.fill_(True)
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    adapter = ConditionalTargetNormalizer(model.target_normalizer, model, b)
    z = adapter(b["x_invisible"], b["x_invisible_mask"][..., None])
    z.requires_grad_()
    physical = adapter.denormalize_grad(z, b["x_invisible_mask"][..., None])
    torch.testing.assert_close(physical, b["x_invisible"], rtol=2e-5, atol=2e-6)
    physical.square().mean().backward()
    assert z.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("field", ["Part_sticNumTowers", "Part_sticChargedTag"])
def test_rejects_finite_stic_outliers_only_on_valid_particles(field):
    names = ["Part_eta", "Part_phi", "Part_energy", "Part_pt", field]
    model = ConditionalPreconditioner(names, normalizer(5), normalizer(2), normalizer(2), width=16)
    b = batch()
    b["x"] = torch.cat((b["x"], torch.zeros_like(b["x"][..., :1])), -1)
    b["x"][:, -1, -1] = 1.078e9  # padded detector values must remain ignored
    model(b)
    b["x"][0, 0, -1] = 1.078e9
    with pytest.raises(ValueError, match="STIC-filtered"):
        model(b)


def test_calibration_cannot_transfer_from_raw_to_filtered_data(tmp_path):
    platform = dict(data_parquet_dir=str(tmp_path / "filtered/train"),
                    data_parquet_val_dir=str(tmp_path / "filtered/val"))
    artifact = {"fit_protocol": {"platform": dict(platform)}}
    validate_calibration_data(artifact, platform)
    artifact["fit_protocol"]["platform"]["data_parquet_dir"] = str(tmp_path / "raw/train")
    with pytest.raises(ValueError, match="fit fresh coordinates"):
        validate_calibration_data(artifact, platform)


def test_calibration_nll_matches_multivariate_normal():
    model, b = make_preconditioner(), batch()
    open_head(model)
    mu, factor = model(b)
    expected = -torch.distributions.MultivariateNormal(mu, scale_tril=factor).log_prob(
        model.target_normalizer(b["x_invisible"]).flatten(1))
    actual, _, _ = model.gaussian_loss(b)
    torch.testing.assert_close(actual, expected)


def test_calibration_provenance_complete_and_checkpoint_contract(tmp_path):
    source = tmp_path / "raw.pt"
    source.write_bytes(b"pinned raw source")
    model = make_preconditioner()
    open_head(model)
    payload = dict(schema="conditional-preconditioning-v1", spec=model.spec,
                   state_dict=model.state_dict(), source_sha256=file_sha256(source), fit_complete=True)
    artifact = tmp_path / "calibration.pt"
    torch.save(payload, artifact)
    restored = make_preconditioner()
    restored.load_calibration(artifact, source)
    assert not any(p.requires_grad for p in restored.parameters())
    for a, b in zip(model.coordinates(batch()), restored.coordinates(batch())):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    # Frozen transform is fully contained in model weights, no artifact dependency.
    other = make_preconditioner()
    other.load_state_dict(restored.state_dict())
    assert bool(other.fitted)
    changed = make_preconditioner(scale_floor=.02)
    with pytest.raises(ValueError, match="contract differs"):
        changed.load_state_dict(restored.state_dict())
    source.write_bytes(b"different source")
    with pytest.raises(ValueError, match="identical raw"):
        restored.load_calibration(artifact, source)
    payload["fit_complete"] = False
    torch.save(payload, artifact)
    with pytest.raises(ValueError, match="not completed"):
        restored.load_calibration(artifact, source)


def test_configuration_and_run_names():
    cfg = deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                      read_overlay_yaml(ROOT / "config/train_diffusion_conditional_preconditioning.yaml"))
    control = deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                          read_overlay_yaml(ROOT / "config/train_diffusion_relation_context_filtered.yaml"))
    validate_config(cfg)
    assert cfg["platform"] == control["platform"]
    assert cfg["platform"]["require_filtered_data"] is True
    assert cfg["logger"]["wandb"]["id"] == "condpre02"
    assert cfg["preconditioning_fit"]["wandb"]["id"] == "condcoordfit02"
    assert control["logger"]["wandb"]["id"] == "relcontext02"
    assert cfg["options"]["Dataset"] == control["options"]["Dataset"]
    training = cfg["options"]["Training"]
    assert training["epochs"] == training["total_epochs"] == 50
    assert cfg["preconditioning_fit"]["epochs"] == 20
    assert cfg["network"]["VisibleConditioning"] == control["network"]["VisibleConditioning"]
    assert training["JointCoverage"] == control["options"]["Training"]["JointCoverage"]
    assert not training["EMA"]["enable"]
    assert not cfg["rl"]["enabled"]
    for name in (cfg["logger"]["wandb"]["run_name"], cfg["preconditioning_fit"]["wandb"]["name"]):
        assert len(name) <= 96 and len(name.split(" | ")) == 3
    cfg["platform"]["number_of_workers"] = 1
    with pytest.raises(ValueError, match="16 workers"):
        validate_config(cfg)


def small_model(enabled=True):
    pytest.importorskip("lightning")
    from evenet.control.global_config import DotDict
    from evenet.network.evenet_model import EveNetModel
    network = read_yaml(ROOT / "config/evenet_defaults/network-20M.yaml")
    for component in network["Body"].values():
        if isinstance(component, dict):
            for key in ("hidden_dim", "initial_embedding_dim", "position_embedding_dim"):
                if key in component:
                    component[key] = 8
            for key in ("num_layers", "num_embedding_layers", "num_encoder_layers"):
                if key in component:
                    component[key] = 1
            for key in ("num_heads", "num_attention_heads"):
                if key in component:
                    component[key] = 2
            component.update(dropout=0., feature_drop=0., drop_probability=0.)
    network["Body"]["PET"].update(enable_local_embedding=False, local_point_index=[2, 3])
    network["TruthGeneration"].update(hidden_dim=8, num_layers=2, num_heads=2,
                                     dropout=0., feature_drop=0., drop_probability=0.)
    network["ConditionalPreconditioning"] = dict(enabled=enabled, width=16)
    names = ["Part_eta", "Part_phi", "Part_energy", "Part_pt"]
    event = dict(input_types={"Source": "SEQUENTIAL", "Global": "GLOBAL"},
                 input_features={"Source": [dict(name=x, normalize=True) for x in names],
                                 "Global": [dict(name=x, normalize=True) for x in ["g1", "g2"]]},
                 sequential_inv_cdf_index=[], invisible_inv_cdf_index=[], num_classes_total=2)
    norm = dict(input_mean={"Source": torch.zeros(4), "Global": torch.zeros(2)},
                input_std={"Source": torch.ones(4), "Global": torch.ones(2)},
                invisible_mean={"Source": torch.zeros(2)}, invisible_std={"Source": torch.full((2,), .2)})
    return EveNetModel(DotDict(dict(network=network, options={}, event_info=event)),
                       torch.device("cpu"), neutrino_generation=True, normalization_dict=norm)


def test_actual_evenet_forward_sampler_velocity_alignment_and_frozen_coordinates():
    from evenet.utilities.diffusion_sampler import add_noise
    model = small_model().eval()
    open_head(model.conditional_preconditioning)
    b, t = batch(4), torch.tensor([.02, .2, .7, .98])
    captured = {}
    def record(target, time):
        captured["target"] = target.detach().clone()
        captured["noise"], captured["velocity"] = add_noise(target, time)
        return captured["noise"], captured["velocity"]
    with patch("evenet.network.evenet_model.add_noise", side_effect=record):
        out = model(b, t)["generations"]["neutrino"]
    expected = model.invisible_coordinate_normalizer(b)(b["x_invisible"], b["x_invisible_mask"][..., None])
    torch.testing.assert_close(captured["target"], expected, rtol=0, atol=0)
    torch.testing.assert_close(out["truth"], captured["velocity"], rtol=0, atol=0)
    no_truth = {k: v for k, v in b.items() if k != "x_invisible"}
    prediction = model.predict_diffusion_vector(captured["noise"], no_truth, t, "neutrino",
                                                b["x_invisible_mask"][..., None])
    torch.testing.assert_close(out["vector"], prediction, rtol=0, atol=0)
    (out["vector"] - out["truth"]).square().mean().backward()
    assert all(p.grad is None for p in model.conditional_preconditioning.parameters())
    assert model.TruthGeneration.generator.weight.grad.abs().sum() > 0
    restored = small_model().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.predict_diffusion_vector(captured["noise"], no_truth, t, "neutrino",
                              b["x_invisible_mask"][..., None]), prediction, rtol=0, atol=0)
    old = small_model(False)
    assert old.invisible_coordinate_normalizer(b) is old.invisible_normalizer


def test_strict_source_then_calibration_preserves_every_backbone_tensor(tmp_path):
    from evenet.network.body.visible_conditioning import VisibleConditioning
    from evenet.network.body.relation_conditioning import load_relation_weights
    source, target = small_model(False), small_model(True)
    spec = dict(feature_names=["Part_eta", "Part_phi", "Part_energy", "Part_pt"],
                token_dim=4, hidden_dim=8, num_layers=2, n_branches=2, width=16,
                feature_mode="kinematics", numerical_features=["Part_energy", "Part_pt"])
    source.TruthGeneration.visible_conditioning = VisibleConditioning(**spec)
    target.TruthGeneration.visible_conditioning = VisibleConditioning(
        **spec, relation_adapter=dict(enabled=True, mode="context", width=16))
    # Source owns the normalizer even if an external normalization file changed.
    source.invisible_normalizer.std.fill_(.31)
    checkpoint = {"state_dict": {"model." + k: v for k, v in source.state_dict().items()}}
    load_relation_weights(target, checkpoint)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    torch.testing.assert_close(target.conditional_preconditioning.target_normalizer.std,
                               source.invisible_normalizer.std, rtol=0, atol=0)
    assert target.conditional_preconditioning.identity_initialized()
    bad = copy.deepcopy(checkpoint)
    del bad["state_dict"]["model.TruthGeneration.generator.weight"]
    with pytest.raises(ValueError, match="missing_shared"):
        load_relation_weights(target, bad)


def test_weights_only_loader_cannot_drop_coordinate_system_or_normalizers():
    from evenet.utilities.tool import safe_load_state
    source, restored = small_model(), small_model()
    open_head(source.conditional_preconditioning)
    source.invisible_normalizer.std.fill_(.31)
    saved = {"model." + k: v for k, v in source.state_dict().items()}
    safe_load_state(restored, saved, verbose=False)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="matching model configuration"):
        safe_load_state(small_model(False), saved, verbose=False)
    del saved["model.conditional_preconditioning.target_normalizer.std"]
    with pytest.raises(ValueError, match="Incomplete"):
        safe_load_state(restored, saved, verbose=False)


def test_real_candidate_sampling_applies_inverse_for_parallel_chains_and_gradients():
    pytest.importorskip("lightning")
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from evenet.utilities.diffusion_sampler import DDIMSampler
    model = small_model().eval()
    open_head(model.conditional_preconditioning)
    b = batch(3)
    class FixedResidualSampler(DDIMSampler):
        def sample(self, data_shape, pred_fn, normalize_fn, **kwargs):
            z = torch.ones(data_shape, requires_grad=kwargs.get("differentiable", False)) * .4
            method = normalize_fn.denormalize_grad if kwargs.get("differentiable") else normalize_fn.denormalize
            return method(z, kwargs["noise_mask"])
    sampler = FixedResidualSampler(torch.device("cpu"))
    sequential = generate_neutrino_candidates(model, b, sampler, K=3, num_ddim_steps=2,
                                               device=torch.device("cpu"), parallel_chains=1)
    parallel = generate_neutrino_candidates(model, b, sampler, K=3, num_ddim_steps=2,
                                             device=torch.device("cpu"), parallel_chains=3)
    expected = model.invisible_coordinate_normalizer(b).denormalize(torch.full((3, 2, 2), .4))
    torch.testing.assert_close(sequential, expected[None].expand_as(sequential), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(parallel, sequential, rtol=1e-6, atol=1e-7)
    differentiable = generate_neutrino_candidates(model, b, sampler, K=1, num_ddim_steps=2,
        device=torch.device("cpu"), differentiable=True)
    assert differentiable.requires_grad


def test_ddim_oracle_clean_estimate_is_recovered_in_physical_coordinates():
    pytest.importorskip("lightning")
    from evenet.utilities.diffusion_sampler import DDIMSampler, get_logsnr_alpha_sigma
    model, b = make_preconditioner(), batch(4)
    open_head(model)
    adapter = ConditionalTargetNormalizer(model.target_normalizer, model, b)
    z0 = adapter(b["x_invisible"])
    def oracle(noise_x, time):
        _, a, s = get_logsnr_alpha_sigma(time, (len(time), 1, 1))
        return (a * noise_x - z0) / s
    generated = DDIMSampler(torch.device("cpu"), x0_mode="stable_v").sample(
        z0.shape, oracle, normalize_fn=adapter, num_steps=20)
    # Finite endpoint sigma(t=0) is ~4.5e-5; allow that numerical residual.
    torch.testing.assert_close(generated, b["x_invisible"], atol=3e-4, rtol=3e-4)
