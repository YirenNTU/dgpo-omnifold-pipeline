from pathlib import Path
import sys
import copy
import json
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_dgpo_checkpoint_transfer import build_probe_config, source_metadata, DEFAULT_CONFIG, main, validate_probe_runtime
from train_neutrino_backend import REPO_ROOT, absolutize_default_paths, deep_update, read_overlay_yaml, read_yaml


def checkpoint():
    return {"state_dict": {"weight": 1}, "dgpo_ref_state_dict": {"weight": 2},
            "dgpo_round_ref_state_dict": {"weight": 3}, "dgpo_omnifold_reward_stack": {"models": [1]},
            "dgpo_adaptive_omnifold_state": {"baseline_auc_gap": .38, "trigger_threshold": .05},
            "dgpo_checkpoint_version": 1, "global_step": 1137, "epoch": 113,
            "dgpo_next_epoch": 113, "dgpo_epoch_step": 7, "dgpo_reward_round_id": 7,
            "dgpo_optimizer_state_dict": {"optimizer": {"state": {0: {}}, "param_groups": []},
                "scheduler": {"last_epoch": 1137}, "lr_schedule": {"total_steps": 15000}}}


def test_full_resume_config_keeps_algorithm_and_source_untouched(tmp_path):
    base = REPO_ROOT / "config/train_diffusion_nersc.yaml"
    original = deep_update(read_yaml(base), read_overlay_yaml(REPO_ROOT / "config/dgpo_global_film_diffusion_last.yaml"))
    original = absolutize_default_paths(original, base.parent)
    ckpt = checkpoint()
    before = copy.deepcopy(ckpt)
    metadata = source_metadata(ckpt)
    cfg = build_probe_config(base, DEFAULT_CONFIG, pinned=tmp_path / "source.ckpt", output=tmp_path, metadata=metadata)
    assert ckpt == before
    assert cfg["network"] == original["network"]
    assert cfg["reward_config"] == original["reward_config"]
    assert cfg["platform"] == original["platform"]
    assert cfg["dgpo"]["reference_trust"] == original["dgpo"]["reference_trust"]
    for key in ("K", "num_ddim_steps", "num_train_timesteps", "advantage_estimator", "beta", "beta_kl", "steps_per_epoch",
                "policy_eval_t_min", "policy_eval_t_max", "grad_clip_norm", "conditioning_learning_rates", "lr_schedule"):
        assert cfg["dgpo"][key] == original["dgpo"][key]
    assert cfg["dgpo"]["gradient_transfer_trace"]["update_end_steps"] == [1138, 1142, 1157]
    assert cfg["dgpo"]["checkpoint_transfer"]["events_per_rank"] * 16 == 32768
    assert not cfg["dgpo"]["adaptive_omnifold"]["recalibration"]["refit_once_on_resume"]
    assert not cfg["dgpo"]["adaptive_omnifold"]["recalibration"]["refit_once_fail_closed"]
    assert cfg["logger"]["wandb"]["fresh_run"] and cfg["logger"]["wandb"]["id"] is None
    assert cfg["options"]["Training"]["model_checkpoint_save_path"] == str(tmp_path / "checkpoints")
    assert cfg["options"]["Training"]["Components"] == original["options"]["Training"]["Components"]


@pytest.mark.parametrize("key", ["dgpo_round_ref_state_dict", "dgpo_omnifold_reward_stack", "dgpo_optimizer_state_dict", "dgpo_next_epoch"])
def test_missing_saved_state_rejected(key):
    state = checkpoint()
    state.pop(key)
    with pytest.raises(ValueError):
        source_metadata(state)


def test_pending_controller_action_rejected():
    state = checkpoint()
    state["dgpo_adaptive_omnifold_state"]["raw_monitor_baseline_pending"] = True
    with pytest.raises(ValueError, match="controller action"):
        source_metadata(state)


def test_native_config_validation_detects_inherited_refit_conflict():
    cfg = read_overlay_yaml(DEFAULT_CONFIG)
    validate_probe_runtime(cfg)
    cfg["dgpo"]["adaptive_omnifold"]["recalibration"]["refit_once_fail_closed"] = True
    with pytest.raises(ValueError, match="refit_once_fail_closed requires refit_once_on_resume"):
        validate_probe_runtime(cfg)


def test_invalid_config_stops_before_checkpoint_io_or_ray(tmp_path, monkeypatch):
    import yaml
    cfg = read_overlay_yaml(DEFAULT_CONFIG)
    cfg["dgpo"]["adaptive_omnifold"]["recalibration"]["refit_once_fail_closed"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["probe", "--config", str(invalid),
        "--checkpoint", str(tmp_path / "nonexistent.ckpt"), "--output-root", str(tmp_path / "outputs")])
    with mock.patch("diagnose_dgpo_checkpoint_transfer.subprocess.run") as run:
        with pytest.raises(ValueError, match="refit_once_fail_closed requires refit_once_on_resume"):
            main()
        run.assert_not_called()
    assert not (tmp_path / "outputs").exists()


def test_launcher_dry_run_pins_source_and_does_not_launch(tmp_path, monkeypatch):
    import torch
    import yaml
    source = tmp_path / "last.ckpt"
    torch.save(checkpoint(), source)
    monkeypatch.setattr(sys, "argv", ["probe", "--checkpoint", str(source),
        "--output-root", str(tmp_path / "outputs"), "--dry-run"])
    with mock.patch("diagnose_dgpo_checkpoint_transfer.subprocess.run") as run:
        main()
        run.assert_not_called()
    output, = (tmp_path / "outputs").iterdir()
    metadata = json.loads((output / "source_metadata.json").read_text())
    assert metadata["global_step"] == 1137
    runtime = yaml.safe_load((output / "runtime.yaml").read_text())
    assert runtime["dgpo"]["gradient_transfer_trace"]["update_end_steps"] == [1138, 1142, 1157]
    assert runtime["options"]["Training"]["model_checkpoint_load_path"] == str(output / "source.ckpt")
    # Simulate the production writer's atomic replacement of last.ckpt. The
    # diagnostic must retain the pinned snapshot, not follow the mutable alias.
    replacement = tmp_path / "replacement.ckpt"
    torch.save({**checkpoint(), "global_step": 1140}, replacement)
    replacement.replace(source)
    assert torch.load(output / "source.ckpt", weights_only=False)["global_step"] == 1137
    assert torch.load(source, weights_only=False)["global_step"] == 1140
