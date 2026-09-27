"""Matched A/B and fresh-audit contracts; never launch jobs."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import yaml

from train_dgpo_token_film import configuration
from train_neutrino_backend import deep_update, read_yaml
from RL.DGPO_neutrino import dgpo_trainer as trainer
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config, step_zero_raw_audit_enabled
from evenet.network.body.test_token_conditioning import token_head


def test_only_network_flag_changes_and_training_protocol_stays_matched():
    control, last = configuration("control"), configuration("last")
    network = copy.deepcopy(control["network"])
    network["VisibleConditioning"]["diffusion_token_readout"]["enabled"] = True
    assert network == last["network"]
    for key in ("dgpo", "reward_config", "platform"):
        assert control[key] == last[key]
    training = dict(control["options"]["Training"])
    training["model_checkpoint_save_path"] = last["options"]["Training"]["model_checkpoint_save_path"]
    assert training == last["options"]["Training"]
    for cfg in (control, last):
        dg, tr = cfg["dgpo"], cfg["options"]["Training"]
        assert tr["model_checkpoint_load_path"].endswith("dgpo-epoch=-1-next_ep=0-step=0.ckpt")
        assert not tr["EMA"]["enable"] and not tr["EMA"]["replace_model_after_load"]
        assert dg["reference_trust"]["coefficient"] == 1
        assert dg["step_zero_architecture_bootstrap"] and not dg["auto_resume_from_last"]
        assert dg["conditioning_learning_rates"]["visible_conditioning"] == 1e-4
        assert tr["learning_rate"] == 5e-5
        assert dg["lr_schedule"]["total_epochs"] == tr["epochs"] == 1500
        assert cfg["experiment"]["policy_update_budget"] == 1000
        assert cfg["experiment"]["classifier_fit_count"] == cfg["experiment"]["reward_fit_count"] == 0
        assert cfg["experiment"]["endpoint_selection"] == "absolute_policy_step1000"
        assert "--max-steps 1000" in cfg["nersc"]["execution"]["command"]
        assert cfg["platform"]["number_of_workers"] == 16
        assert cfg["platform"]["resources_per_worker"]["GPU"] == 1
        assert cfg["platform"]["use_gpu"] is True
        assert cfg["nersc"]["nodes"] * cfg["nersc"]["gpus_per_node"] == 16
        assert dg["validation_every_n_epochs"] == 10
        a = resolve_adaptive_config(dg)
        assert a.log_only and a.staleness_every_n_epochs > 1500
        assert not a.bootstrap_on_start and not a.refit_once_on_resume
        assert not step_zero_raw_audit_enabled(a)
        assert not a.raw_audit_enabled
        assert len(cfg["logger"]["wandb"]["run_name"]) <= 96
    assert control["logger"]["wandb"]["id"] != last["logger"]["wandb"]["id"]


def test_real_optimizer_groups_include_new_branch_once_at_existing_branch_lr():
    cfg = configuration("last")
    options = deep_update(read_yaml(Path(__file__).resolve().parents[1] / "config/evenet_defaults/options.yaml"), cfg["options"])
    class Config(dict):
        __getattr__ = dict.__getitem__
    model = torch.nn.Module()
    for name in ("GlobalEmbedding", "PET", "InvisibleInputProjector", "GroupedSequentialEmbedding"):
        setattr(model, name, torch.nn.Linear(1, 1))
    model.PET.angular_conditioning = torch.nn.Linear(1, 1)
    model.TruthGeneration = token_head()
    global_cfg = SimpleNamespace(options=SimpleNamespace(Training=Config(options["Training"])))
    with mock.patch.object(trainer, "global_config", global_cfg):
        opt = trainer.build_optimizer(model, steps_per_epoch=10, warmup_steps=1, is_rank0=False,
                lr_schedule=cfg["dgpo"]["lr_schedule"],
                conditioning_learning_rates=cfg["dgpo"]["conditioning_learning_rates"])
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    group = next(pg for pg in opt.param_groups if pg["group_name"] == "visible_conditioning")
    assert group["initial_lr"] == 1e-4
    assert {id(p) for p in model.TruthGeneration.visible_conditioning.token_readout.parameters()} <= {id(p) for p in group["params"]}


@pytest.mark.parametrize("k", [1, 16])
def test_all_sample_reward_is_not_best_of_k_and_reduces_by_sample_count(k):
    rewards = torch.arange(k * 5, dtype=torch.float32).reshape(k, 5)
    mask = torch.tensor([True, True, False, True, False])
    rewards[:, ~mask] = float("nan")
    best, total, count = trainer._validation_reward_sample_stats(rewards, mask)
    assert total / count == pytest.approx(float(rewards[:, mask].mean()))
    torch.testing.assert_close(best, rewards[:, mask].max(0).values)
    if k > 1:
        assert best.mean() > total / count
    # Unequal rank populations: sum totals/counts, not rank averages.
    pieces = [trainer._validation_reward_sample_stats(rewards[:, start:stop], mask[start:stop])
              for start, stop in ((0, 1), (1, 5))]
    assert sum(x[1] for x in pieces) / sum(x[2] for x in pieces) == pytest.approx(total / count)
    for key in ("val/reward/all_sample_mean", "val/reward/best_of_k_mean"):
        assert trainer._wandb_simplified_keep(key, 1.)
        assert trainer._wandb_critical_keep(key)


def test_fresh_judges_match_have_disjoint_test_and_never_update_policy_or_reward():
    audits = [configuration(arm, audit_checkpoint=Path(f"/source/{arm}.ckpt"), expected_step=1000)
              for arm in ("control", "last")]
    assert audits[0]["dgpo"] == audits[1]["dgpo"]
    for cfg in audits:
        assert cfg["platform"]["number_of_workers"] == 16
        assert cfg["platform"]["resources_per_worker"]["GPU"] == 1
        assert cfg["platform"]["use_gpu"] is True
        assert cfg["nersc"]["nodes"] * cfg["nersc"]["gpus_per_node"] == 16
        assert cfg["experiment"]["classifier_only"]
        assert cfg["experiment"]["policy_updates_per_round"] == 0
        assert cfg["dgpo"]["checkpoint_load_mode"] == "weights_only"
        assert not cfg["dgpo"]["step_zero_architecture_bootstrap"]
        a = resolve_adaptive_config(cfg["dgpo"], classifier_only=True)
        assert not a.bootstrap_on_start and not a.baseline_probe_on_start
        assert not a.raw_monitor_warm_start
        assert a.audit_fit["disjoint_final_audit"] and a.audit_fit["restore_best"]
        assert a.audit_fit["training_population"] == "omnifold_fold"
        assert a.audit_fit["training_fold"] == 1
        assert a.audit_fit["min_steps"] == 0 and a.audit_fit["steps"] is None
        assert a.audit_fit["validation_patience_epochs"] == 25
        assert a.audit_fit["decoder_layers"] == 3
        assert cfg["experiment"]["protocol"] == "h4-matched-fold-fresh-audit-v1"
        assert cfg["experiment"]["primary_endpoint"] == "staleness/raw_auc_gap"
        assert cfg["experiment"]["policy_update_budget"] == 0
        assert cfg["experiment"]["source_policy_step"] == 1000
        assert not cfg["experiment"]["checkpoint_metadata_verified"]
        assert "matched_fold_audit_step1000" in cfg["options"]["Training"]["model_checkpoint_save_path"]
        assert "-matched-1000" in cfg["logger"]["wandb"]["id"]
        assert "fold 1" in cfg["logger"]["wandb"]["run_name"]
        assert len(cfg["logger"]["wandb"]["run_name"]) <= 96
        assert not cfg["options"]["Training"]["EMA"]["enable"]
        assert not cfg["options"]["Training"]["EMA"]["replace_model_after_load"]
    with pytest.raises(ValueError, match="expected-step"):
        configuration("last", audit_checkpoint=Path("/source/last.ckpt"))


@pytest.mark.parametrize("arm", ["control", "last"])
def test_audit_matches_latest_and_original_population_not_the_old_probe_split(arm):
    from train_h4_checkpoint_audit import configuration as latest_configuration
    from train_neutrino_backend import read_overlay_yaml
    root = Path(__file__).resolve().parents[1]
    cfg = configuration(arm, audit_checkpoint=Path("/source.ckpt"), expected_step=1000)
    latest = latest_configuration()
    original = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
                           read_overlay_yaml(root / "config/dgpo_h4_kinematic_adaln_depth3.yaml"))
    a = resolve_adaptive_config(cfg["dgpo"], classifier_only=True)
    for reference in (latest, original):
        other = resolve_adaptive_config(reference["dgpo"], classifier_only=True)
        assert a.audit_fit == other.audit_fit
        for key in ("seed", "probe_seed", "pool_selection_seed", "crossfit_partition",
                    "crossfit_folds", "crossfit_repeats", "pool_data_parquet_dir", "pool_events",
                    "probe_max_events", "pool_generation_batch_size", "cache_event_inputs",
                    "fixed_audit_panel", "periodic_pair_features_enabled", "visible_pair_rest_frame_enabled"):
            assert getattr(a, key) == getattr(other, key), key
        assert cfg["options"]["Dataset"] == reference["options"]["Dataset"]
        assert cfg["reward_config"] == reference["reward_config"]
        assert cfg["dgpo"]["validation_num_ddim_steps"] == reference["dgpo"]["validation_num_ddim_steps"] == 20
    # Historical counts are documentation, not hard-coded training quotas.
    assert a.pool_events is None
    assert cfg["experiment"]["historical_population_events"] == latest["experiment"]["historical_population_events"]


@pytest.mark.parametrize("arm", ["control", "last"])
@pytest.mark.parametrize("audit", [False, True])
def test_launcher_passes_training_budget_but_never_limits_classifier_fit(tmp_path, monkeypatch, arm, audit):
    import train_dgpo_token_film as launcher
    argv = ["train_dgpo_token_film.py", "--arm", arm]
    if audit:
        state = {"model.weight": torch.ones(1)}
        if arm == "last":
            state["model.TruthGeneration.visible_conditioning.token_readout.output.weight"] = torch.ones(1)
        actual = tmp_path / "step1000.ckpt"
        torch.save({"state_dict": state, "global_step": 1000, "epoch": 99}, actual)
        link = tmp_path / "last.ckpt"
        link.symlink_to(actual)
        argv.extend(["--audit-checkpoint", str(link), "--expected-step", "1000"])
    monkeypatch.setattr(launcher.sys, "argv", argv)
    launched = []
    def capture(command, *, cwd, check):
        cfg = yaml.safe_load(Path(command[command.index("--overlay-config") + 1]).read_text())
        assert check and cwd == launcher.ROOT
        assert cfg["platform"]["number_of_workers"] == 16
        if audit:
            assert "--max-steps" not in command
            assert cfg["options"]["Training"]["model_checkpoint_load_path"] == str(actual)
            assert cfg["nersc"]["reproducibility"]["source_checkpoint"] == str(actual)
            assert cfg["experiment"]["source_policy_epoch"] == 99
            assert cfg["experiment"]["checkpoint_metadata_verified"]
            a = resolve_adaptive_config(cfg["dgpo"], classifier_only=True)
            assert a.audit_fit["training_population"] == "omnifold_fold"
        else:
            assert command[command.index("--max-steps") + 1] == "1000"
            assert cfg["dgpo"]["lr_schedule"]["total_epochs"] == 1500
        launched.append(cfg)
    monkeypatch.setattr(launcher.subprocess, "run", capture)
    launcher.main()
    assert len(launched) == 1


@pytest.mark.parametrize("arm", ["control", "last"])
@pytest.mark.parametrize("audit", [False, True])
def test_token_dry_run_never_loads_checkpoints_or_starts_training(monkeypatch, capsys, arm, audit):
    import train_dgpo_token_film as launcher
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not load checkpoints or launch work")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    argv = ["train_dgpo_token_film.py", "--arm", arm, "--dry-run"]
    if audit:
        argv.extend(["--audit-checkpoint", "/does/not/exist.ckpt", "--expected-step", "1000"])
    monkeypatch.setattr(launcher.sys, "argv", argv)
    launcher.main()
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert cfg["experiment"]["classifier_only"] == audit
    assert cfg["experiment"]["policy_update_budget"] == (0 if audit else 1000)


@pytest.mark.parametrize("bad_step,token_state", [(999, True), (1000, False)])
def test_audit_rejects_wrong_step_or_wrong_policy_arm(tmp_path, monkeypatch, bad_step, token_state):
    import train_dgpo_token_film as launcher
    state = {"weight": torch.ones(1)}
    if token_state:
        state["TruthGeneration.visible_conditioning.token_readout.weight"] = torch.ones(1)
    path = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": state, "global_step": bad_step, "epoch": 99}, path)
    monkeypatch.setattr(launcher.sys, "argv", ["train_dgpo_token_film.py", "--arm", "last",
                                              "--audit-checkpoint", str(path), "--expected-step", "1000"])
    with mock.patch.object(launcher.subprocess, "run") as launch:
        with pytest.raises(ValueError, match="global_step|architecture"):
            launcher.main()
        launch.assert_not_called()
