import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_global_film_coverage as exp
from diagnose_h4_ddim_coverage import analyze_arm, paired_comparison
from test_h4_ddim_coverage import panel_fixture


def test_dry_run_preserves_architecture_and_never_needs_source_files(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run([sys.executable, str(Path(exp.__file__)), "--dry-run",
                             "--checkpoint", "/nonexistent/checkpoint.ckpt", "--output", str(output)],
                            text=True, capture_output=True, check=True)
    d = yaml.safe_load(result.stdout)
    assert not output.exists()
    assert d["evaluation"]["checkpoint_files_verified"] is False
    assert d["evaluation"]["policy_updates"] == d["evaluation"]["classifier_fits"] == 0
    assert d["evaluation"]["protocol"] == exp.PROTOCOL
    c = d["runtime"]
    assert c["network"]["TruthGeneration"]["num_layers"] == 3
    assert c["network"]["Body"]["PET"]["visible_angular_fourier"]["enabled"]
    assert c["network"]["VisibleConditioning"]["diffusion_enabled"]
    assert not c["network"]["VisibleConditioning"]["diffusion_token_readout"]["enabled"]
    assert not c["options"]["Training"]["EMA"]["enable"]
    assert c["options"]["Training"]["model_checkpoint_load_path"] is None
    assert c["options"]["Training"]["pretrain_model_load_path"] is None


def source_fixture():
    return dict(epoch=98, global_step=1312,
                state_dict={"model.PET.angular_conditioning.weight": torch.ones(2, 2),
                            "model.TruthGeneration.visible_conditioning.modulations.0.weight": torch.ones(2, 2)},
                ema_state_dict={"not_raw": torch.zeros(2)}, optimizer_states=[{"discard": True}])


def test_source_snapshot_has_only_raw_weights_and_recorded_training_clock(tmp_path):
    path = tmp_path / "last.ckpt"
    source = source_fixture()
    torch.save(source, path)
    meta, saved = exp.inspect_checkpoint(path)
    assert meta["epoch"] == 98 and meta["global_step"] == 1312
    assert set(saved) == {"state_dict", "epoch", "global_step"}
    for key in source["state_dict"]:
        assert torch.equal(saved["state_dict"][key], source["state_dict"][key])


@pytest.mark.parametrize("fault", ["dgpo", "no_film", "no_fourier", "token_readout", "half", "nan", "no_epoch"])
def test_rejects_wrong_checkpoint_before_sampling(tmp_path, fault):
    s = source_fixture()
    film = "model.TruthGeneration.visible_conditioning.modulations.0.weight"
    if fault == "dgpo":
        s["dgpo_checkpoint_version"] = 1
    elif fault == "no_film":
        del s["state_dict"][film]
    elif fault == "no_fourier":
        del s["state_dict"]["model.PET.angular_conditioning.weight"]
    elif fault == "token_readout":
        s["state_dict"]["model.TruthGeneration.visible_conditioning.token_readout.weight"] = torch.ones(2)
    elif fault == "half":
        s["state_dict"][film] = s["state_dict"][film].half()
    elif fault == "nan":
        s["state_dict"][film][0, 0] = float("nan")
    else:
        del s["epoch"]
    path = tmp_path / "bad.ckpt"
    torch.save(s, path)
    with pytest.raises(ValueError):
        exp.inspect_checkpoint(path)


def baseline_fixture(tmp_path, monkeypatch):
    panel = panel_fixture(8)
    protocol = {**exp.PROTOCOL, "events": 8, "K": 4, "bootstrap": 20}
    monkeypatch.setattr(exp, "PROTOCOL", protocol)
    panel_path = tmp_path / "panel.pt"
    torch.save(panel, panel_path)
    generated = panel["truth"].float()[:, None, :].repeat(1, 4, 1)
    generated[:, :, 1] += .02
    result, _, angles = analyze_arm(panel, generated)
    path = tmp_path / "epoch-0100.json"
    report = dict(config={**protocol, "panel_path": str(panel_path)}, completed_epochs=100,
                  result=result, global_step=1300)
    path.write_text(json.dumps(report))
    torch.save(dict(generated=generated, angles=torch.from_numpy(angles)), path.with_suffix(".pt"))
    return path, panel, report


def test_saved_reference_reconstructs_and_retains_event_order(tmp_path, monkeypatch):
    path, panel, report = baseline_fixture(tmp_path, monkeypatch)
    restored, saved, old, metrics = exp.load_baseline(path)
    assert torch.equal(restored["pool_rows"], panel["pool_rows"])
    assert old == report
    assert saved["generated"].shape == (8, 4, 4)
    assert "topology/joint_radius/w1_radians" in metrics


@pytest.mark.parametrize("fault", ["noise_seed", "K", "epoch", "duplicate_ids", "truth_values", "candidates", "saved_angles", "report_metric"])
def test_rejects_mismatched_baseline_protocol_and_artifacts(tmp_path, monkeypatch, fault):
    path, panel, report = baseline_fixture(tmp_path, monkeypatch)
    if fault in ("noise_seed", "K"):
        report["config"]["seed" if fault == "noise_seed" else "K"] += 1
    elif fault == "epoch":
        report["completed_epochs"] = 75
    elif fault in ("duplicate_ids", "truth_values"):
        if fault == "duplicate_ids":
            panel["pool_rows"][1] = panel["pool_rows"][0]
        else:
            panel["truth"][0, 1] += .1
        torch.save(panel, report["config"]["panel_path"])
    elif fault == "report_metric":
        report["result"]["topology"]["joint_radius"]["w1_radians"] += .1
    else:
        saved = torch.load(path.with_suffix(".pt"), weights_only=True)
        saved["generated" if fault == "candidates" else "angles"][0, 0, 1] += .1
        torch.save(saved, path.with_suffix(".pt"))
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        exp.load_baseline(path)


def test_decision_requires_paired_truth_gap_and_radius_not_just_any_hit():
    panel = panel_fixture(8)
    # Give all events exactly back-to-back truth; generate a displaced baseline.
    panel["truth"] = torch.zeros_like(panel["truth"])
    old = torch.zeros(8, 4, 4)
    old[:, :, 1] = .02
    new = torch.zeros_like(old)
    a, truth, before = analyze_arm(panel, old)
    b, _, after = analyze_arm(panel, new)
    current = dict(result=b, paired=paired_comparison(truth, before, after, 50, 42017))
    decision = exp.decision(a, current)
    assert decision["primary_gap_ci95_below_zero"]
    assert decision["coverage_and_radius_improved"]
    same = dict(result=a, paired=paired_comparison(truth, before, before, 50, 42017))
    assert not exp.decision(a, same)["coverage_and_radius_improved"]
    reverse = dict(result=a, paired=paired_comparison(truth, after, before, 50, 42017))
    assert not exp.decision(b, reverse)["primary_gap_ci95_below_zero"]
