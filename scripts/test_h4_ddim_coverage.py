import importlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_h4_ddim_coverage as exp
from test_h4_topology_resolution import fixture


@pytest.fixture
def sampler_module(monkeypatch):
    # The timing decorator otherwise imports Lightning/torchvision. Exercise
    # the actual sampler and production candidate interface without that stack.
    stub = types.ModuleType("evenet.utilities.debug_tool")
    stub.time_decorator = lambda **kwargs: lambda fn: fn
    monkeypatch.setitem(sys.modules, "evenet.utilities.debug_tool", stub)
    return importlib.import_module("evenet.utilities.diffusion_sampler")


def panel_fixture(n=32):
    b = fixture()
    return dict(condition=b["test_condition"][:n], truth=b["test_truth"][:n],
                test_rows=torch.arange(n), pool_rows=torch.arange(n), packing_spec=b["packing_spec"])


def original_loop(module, noise, pred, steps, mask=None):
    x = noise.clone()
    if mask is not None:
        x = x * mask
    shape = (len(x), 1, 1)
    for step in range(steps, 0, -1):
        t = torch.ones(len(x)) * step / steps
        _, a, s = module.get_logsnr_alpha_sigma(t, shape)
        _, ap, sp = module.get_logsnr_alpha_sigma(torch.ones(len(x)) * (step - 1) / steps, shape)
        v = pred(noise_x=x, time=t)
        eps = v * a + x * s
        x0 = (x - s * eps) / a
        x = ap * x0 + sp * eps
        if mask is not None:
            x = x * mask
    return x


@pytest.mark.parametrize("steps", [1, 20, 100, 200])
def test_default_sampler_is_bitwise_legacy(sampler_module, steps):
    model = lambda noise_x, time: .1 * noise_x + time[:, None, None]
    torch.manual_seed(4)
    noise = torch.randn(3, 2, 2)
    expected = original_loop(sampler_module, noise, model, steps)
    torch.manual_seed(4)
    sampler = sampler_module.DDIMSampler("cpu")
    actual = sampler.sample(noise.shape, model, num_steps=steps)
    assert sampler.x0_mode == "legacy"
    assert torch.equal(expected, actual)


def test_stable_high_noise_inversion(sampler_module):
    torch.manual_seed(42017)
    x, v = torch.randn(2000, 2, 2), torch.randn(2000, 2, 2)
    _, a, s = sampler_module.get_logsnr_alpha_sigma(torch.ones(len(x)), (len(x), 1, 1))
    _, ap, sp = sampler_module.get_logsnr_alpha_sigma(torch.zeros(len(x)), (len(x), 1, 1))
    # Independent FP64 VP rotation evaluated with the same schedule constants.
    reference = ap.double() * (a.double()*x.double() - s.double()*v.double()) + sp.double() * (a.double()*v.double() + s.double()*x.double())
    results = {}
    for mode in ("legacy", "stable_v"):
        sampler = exp.make_replay_sampler(x[None], mode)
        results[mode] = sampler.sample(x.shape, lambda **kwargs: v, num_steps=1)
    stable_error = (results["stable_v"].double() - reference).abs().max()
    legacy_error = (results["legacy"].double() - reference).abs().max()
    assert stable_error < 1e-6
    assert legacy_error > 1e-3
    assert stable_error < legacy_error / 1000
    with pytest.raises(ValueError, match="x0_mode"):
        sampler_module.DDIMSampler("cpu", x0_mode="typo")


@pytest.mark.parametrize("mode", ["legacy", "stable_v"])
def test_mask_denormalization_and_noise_not_mutated(sampler_module, mode):
    noise = torch.randn(1, 4, 2, 2)
    saved = noise.clone()
    mask = torch.ones(4, 2, 1)
    mask[:, 1] = 0
    class Normalizer:
        def denormalize(self, x, received_mask, remove_padding):
            assert received_mask is mask and remove_padding is True
            assert (x[:, 1] == 0).all()
            return x + 7
    sampler = exp.make_replay_sampler(noise, mode)
    out = sampler.sample((4, 2, 2), lambda noise_x, time: noise_x*.2,
                         normalize_fn=Normalizer(), noise_mask=mask, remove_padding=True)
    assert (out[:, 1] == 7).all()
    assert torch.equal(noise, saved)
    with pytest.raises(ValueError, match="exhausted"):
        sampler.prior_sde((4, 2, 2))


def test_noise_bank_matches_old_chain_layout_without_global_rng_changes():
    torch.manual_seed(99)
    rng_before = torch.get_rng_state()
    bank = exp.make_noise_bank(5, 8, 3, "cpu", 42)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(bank, exp.make_noise_bank(5, 8, 3, "cpu", 42))
    torch.manual_seed(42)
    expected = torch.cat([torch.stack([torch.randn(size, 2, 2) for _ in range(8)]).permute(1, 0, 2, 3)
                          for size in [3, 2]])
    assert torch.equal(bank, expected)


def test_all_four_arms_use_identical_noise_through_production_interface(sampler_module):
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    noise = exp.make_noise_bank(3, 2, 3, "cpu", 42).permute(1, 0, 2, 3)
    class Model(torch.nn.Module):
        invisible_input_dim = 2
        invisible_normalizer = None
        def __init__(self):
            super().__init__()
            self.first_inputs, self.calls = [], 0
        def predict_diffusion_vector(self, noise_x, time, **kwargs):
            assert not torch.is_grad_enabled()
            if torch.all(time == 1):
                self.first_inputs.append(noise_x.clone())
            self.calls += 1
            return .05*noise_x
    batch = dict(x_invisible=torch.zeros(3, 2, 2), x_invisible_mask=torch.ones(3, 2, dtype=torch.bool))
    for mode, steps in exp.ARMS.values():
        model = Model().eval().requires_grad_(False)
        sampler = exp.make_replay_sampler(noise, mode)
        generated = generate_neutrino_candidates(model, batch, sampler, K=2,
            num_ddim_steps=steps, device=torch.device("cpu"), parallel_chains=1)
        assert generated.shape == (2, 3, 2, 2)
        assert torch.equal(torch.stack(model.first_inputs), noise)
        assert model.calls == 2*steps and sampler.draws_used == 2


def test_sixteen_rank_merge_handles_unequal_shards():
    n, k = 35, 8
    values = torch.arange(n*k*4).float().reshape(n, k, 4)
    parts = [dict(positions=torch.arange(rank, n, 16), generated=values[rank::16]) for rank in range(16)]
    assert torch.equal(exp.merge_parts(parts, n, k, "generated"), values)
    parts[-1] = parts[0]
    with pytest.raises(ValueError, match="Missing/duplicate"):
        exp.merge_parts(parts, n, k, "generated")


def test_paired_intervals_cluster_events_not_candidates():
    truth = np.zeros((2, 2))
    before = np.ones((2, 128, 2))
    after = before.copy()
    after[0] = 0
    rows = exp.paired_comparison(truth, before, after, 1000, 1)
    for row in rows:
        assert row["after_minus_before_draw_fraction"] == .5
        assert row["paired_event_se"] == .5  # NOT iid 256-draw SE.
        assert row["delta_ci95"] == [0., 1.]
        assert row["absolute_truth_gap_change"] == -.5
        assert row["events_with_nonzero_paired_difference"] == 1
    same = exp.paired_comparison(truth, before, before, 1000, 1)
    assert all(row["after_minus_before_draw_fraction"] == 0 for row in same)
    assert rows == exp.paired_comparison(truth, before, after, 1000, 1)
    with pytest.raises(ValueError, match="Unaligned"):
        exp.paired_comparison(truth, before, after[:, :1])


def test_end_to_end_analysis_json_plot_and_comparisons(tmp_path):
    panel = panel_fixture()
    report = dict(arms={}, comparisons={})
    saved_angles = {}
    for index, arm in enumerate(exp.ARMS):
        generated = panel["truth"][:, None].repeat(1, 8, 1).clone()
        generated[:, :, 1] += .01*index
        result, truth_angles, candidate_angles = exp.analyze_arm(panel, generated)
        report["arms"][arm] = result
        saved_angles[arm] = candidate_angles
        exp.update_comparisons(report, truth_angles, saved_angles, dict(bootstrap=20, bootstrap_seed=42))
        for metric in result["topology"].values():
            assert np.all(np.diff(metric["cdf"]["generated"]) >= 0)
        assert result["target_marginals"]["tau_a_delta_phi"]["w1_radians"] == pytest.approx(.01*index)
    assert len(report["comparisons"]) == 4
    assert report["arms"]["legacy20"]["topology"]["joint_radius"]["w1_radians"] == pytest.approx(0)
    exp.write_json(tmp_path / "report.json", report)
    assert json.loads((tmp_path / "report.json").read_text())["arms"]
    exp.plot_cdfs(report, tmp_path)
    assert (tmp_path / "cdf.png").stat().st_size > 1000


def test_invalid_finite_target_is_reported_not_cut():
    panel = panel_fixture()
    generated = panel["truth"][:, None].repeat(1, 2, 1)
    generated[0, 0, 0] = 4  # theta outside [0,pi], retained as a diagnostic.
    result, _, _ = exp.analyze_arm(panel, generated)
    assert result["invalid_direction_inputs"]["a_theta_outside_0_pi"] == 1
    assert result["coverage"]["events"] == len(generated)


@pytest.fixture
def source_fixture(tmp_path):
    meta = dict(arm="step1110", weights="raw_state_dict_only", events=32, workers=16,
                batch_size=16, seed=42017, K=128, ddim_steps=20, policy_updates=0,
                classifier_fits=0, checkpoint="/checkpoint1110.ckpt")
    raw = dict(options=dict(Training=dict(model_checkpoint_load_path=meta["checkpoint"])),
               dgpo=dict(num_ddim_steps=5, validation_num_ddim_steps=20))
    torch.save(panel_fixture(), tmp_path / "panel.pt")
    (tmp_path / "runtime.yaml").write_text(yaml.safe_dump(raw))
    exp.write_json(tmp_path / "manifest.json", meta)
    return tmp_path, meta, raw


def test_source_contract(source_fixture):
    tmp_path, meta, _ = source_fixture
    with pytest.raises(ValueError, match="incomplete"):
        exp.load_source(tmp_path, 16, 16)
    (tmp_path / "COMPLETE").write_text("yes")
    loaded, _, runtime = exp.load_source(tmp_path, 16, 16)
    assert loaded["seed"] == 42017
    assert runtime["options"]["Training"]["model_checkpoint_load_path"] == meta["checkpoint"]
    with pytest.raises(ValueError, match="workers"):
        exp.load_source(tmp_path, 8, 16)
    meta["weights"] = "ema"
    exp.write_json(tmp_path / "manifest.json", meta)
    with pytest.raises(ValueError, match="weights: saved='ema', expected='raw_state_dict_only'"):
        exp.load_source(tmp_path, 16, 16)


@pytest.mark.parametrize("missing", [("arm",), ("weights",), ("arm", "weights")])
def test_legacy_h4cov1_without_labels_is_accepted_not_relabelled(source_fixture, capsys, missing):
    path, meta, _ = source_fixture
    for key in missing:
        del meta[key]
    exp.write_json(path / "manifest.json", meta)
    original = (path / "manifest.json").read_bytes()
    (path / "COMPLETE").write_text("h4-spike-coverage-v1\n")
    loaded, panel, _ = exp.load_source(path, 16, 16)
    assert loaded == meta and len(panel["truth"]) == 32
    assert all(key not in loaded for key in missing)
    assert (path / "manifest.json").read_bytes() == original
    assert "Legacy manifest" in capsys.readouterr().out


@pytest.mark.parametrize("key,value", [("arm", "pretrainedfull"), ("weights", "ema"),
    ("K", 8), ("ddim_steps", 5), ("policy_updates", 1), ("classifier_fits", 1),
    ("workers", 8), ("batch_size", 8)])
def test_source_mismatches_name_the_actual_field(source_fixture, key, value):
    path, meta, _ = source_fixture
    meta[key] = value
    exp.write_json(path / "manifest.json", meta)
    (path / "COMPLETE").write_text("complete")
    with pytest.raises(ValueError, match=f"{key}: saved={value!r}, expected="):
        exp.load_source(path, 16, 16)


@pytest.mark.parametrize("key", ["K", "ddim_steps", "policy_updates", "classifier_fits", "workers",
                                    "batch_size", "events", "seed", "checkpoint"])
def test_required_source_fields_still_report_missing(source_fixture, key):
    path, meta, _ = source_fixture
    del meta[key]
    exp.write_json(path / "manifest.json", meta)
    (path / "COMPLETE").write_text("complete")
    with pytest.raises(ValueError, match=f"{key}: missing"):
        exp.load_source(path, 16, 16)


def test_legacy_manifest_does_not_skip_runtime_check(source_fixture):
    path, meta, raw = source_fixture
    del meta["arm"], meta["weights"]
    exp.write_json(path / "manifest.json", meta)
    (path / "COMPLETE").write_text("complete")
    raw["options"]["Training"]["model_checkpoint_load_path"] = "/different.ckpt"
    (path / "runtime.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="runtime checkpoint='/different.ckpt'"):
        exp.load_source(path, 16, 16)


def test_rerun_overwrites_only_owned_outputs(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "panel.pt").write_text("untouched")
    (output / "user.txt").write_text("untouched")
    (output / "COMPLETE").write_text("old")
    (output / "report.json").write_text("old")
    _, replaced = exp.prepare_output(output, source)
    assert set(replaced) == {"COMPLETE", "report.json"}
    assert (source / "panel.pt").read_text() == (output / "user.txt").read_text() == "untouched"
    for path in (source, source / "child", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            exp.prepare_output(path, source)


def test_cli_help_and_metadata():
    result = subprocess.run([sys.executable, exp.__file__, "--help"], capture_output=True, text=True, check=True)
    assert "--ray-address" in result.stdout
    assert list(exp.ARMS) == ["legacy20", "stable20", "stable100", "stable200"]
    assert len(exp.RUN_NAME) < 96 and "_" not in exp.RUN_NAME


@pytest.mark.parametrize("nonfinite", [False, True])
def test_worker_lifecycle_on_cpu_with_mocked_cluster(tmp_path, monkeypatch, sampler_module, nonfinite):
    """Run worker orchestration and real sampling/reporting; no Ray/GPU claim."""
    ray = types.ModuleType("ray")
    train = types.ModuleType("ray.train")
    ray_torch = types.ModuleType("ray.train.torch")
    train.get_context = lambda: types.SimpleNamespace(get_world_rank=lambda: 0, get_world_size=lambda: 1)
    ray_torch.get_device = lambda: torch.device("cpu")
    ray.train, train.torch = train, ray_torch
    for name, module in (("ray", ray), ("ray.train", train), ("ray.train.torch", ray_torch)):
        monkeypatch.setitem(sys.modules, name, module)
    config_module = types.ModuleType("evenet.control.global_config")
    config_module.global_config = types.SimpleNamespace(load_yaml=lambda path: None, dgpo={})
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    model_module = types.ModuleType("RL.DGPO_neutrino.model_utils")
    class Model(torch.nn.Module):
        invisible_input_dim = 2
        invisible_normalizer = None
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.1))
        def predict_diffusion_vector(self, noise_x, cond_x, **kwargs):
            assert not self.training and not self.weight.requires_grad
            assert not cond_x["x_invisible"].any()  # no truth leakage
            assert self.weight.item() == pytest.approx(.2)  # raw, not EMA
            return noise_x * (float("nan") if nonfinite else self.weight)
    model_module.build_evenet_on_device = lambda *args: Model()
    model_module.load_normalization_dict = lambda *args: {}
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    packing = types.ModuleType("RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio")
    packing.EventPackingSpec = types.SimpleNamespace(from_dict=lambda x: x)
    packing.unpack_event_inputs = lambda x, spec: unpack_for_test(x, spec)
    monkeypatch.setitem(sys.modules, packing.__name__, packing)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    barriers = []
    monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(True))
    monkeypatch.setattr(exp, "plot_cdfs", lambda *args: None)
    torch.save(panel_fixture(4), tmp_path / "panel.pt")
    torch.save(dict(global_step=1110, state_dict={"weight": torch.tensor(.2)},
                    ema_state_dict={"weight": torch.tensor(999.)}), tmp_path / "source.ckpt")
    cfg = dict(workers=1, output=str(tmp_path), runtime="unused.yaml", checkpoint=str(tmp_path / "source.ckpt"),
               events=4, K=2, seed=42, batch_size=2, bootstrap=20, bootstrap_seed=42, wandb=False)
    precision_before = torch.get_float32_matmul_precision()
    try:
        if nonfinite:
            with pytest.raises(ValueError, match="Nonfinite sampler"):
                exp.worker(cfg)
            assert not (tmp_path / "COMPLETE").exists()
            return
        exp.worker(cfg)
    finally:
        torch.set_float32_matmul_precision(precision_before)
    assert len(barriers) == 8
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["complete"] and len(report["arms"]) == 4 and len(report["comparisons"]) == 4
    assert (tmp_path / "COMPLETE").read_text().strip() == exp.SCHEMA
    noise = torch.load(tmp_path / "initial_noise.pt", weights_only=True)
    assert torch.equal(noise, exp.make_noise_bank(4, 2, 2, "cpu", 10042))
    for arm in exp.ARMS:
        assert torch.load(tmp_path / f"{arm}.pt", weights_only=True)["generated"].shape == (4, 2, 4)


def unpack_for_test(x, spec):
    return exp.unpack(dict(test_condition=x, packing_spec=spec))
