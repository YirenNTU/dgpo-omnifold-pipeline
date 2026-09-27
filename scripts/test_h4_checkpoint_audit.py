"""Fresh latest-policy audit: no fitted classifier reuse or DGPO training."""
from pathlib import Path
import ast
from collections.abc import Mapping
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from train_h4_checkpoint_audit import configuration, inspect_checkpoint, DEFAULT_CHECKPOINT
from train_dgpo_token_film import configuration as token_configuration
from train_neutrino_backend import deep_update, read_overlay_yaml, read_yaml
from RL.DGPO_neutrino import dgpo_trainer as trainer
from RL.DGPO_neutrino.omnifold_ztautau import adaptive
from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
from RL.DGPO_neutrino.model_utils import select_dgpo_training_state


def test_latest_policy_only_cold_classifier_and_matched_fit():
    cfg = configuration()
    original = token_configuration("control", audit_checkpoint=Path("/source.ckpt"), expected_step=10)
    dg, tr, ex = cfg["dgpo"], cfg["options"]["Training"], cfg["experiment"]
    assert tr["model_checkpoint_load_path"] == str(DEFAULT_CHECKPOINT)
    assert dg["checkpoint_load_mode"] == "weights_only"
    assert not dg["auto_resume_from_last"] and not dg["step_zero_architecture_bootstrap"]
    assert not tr["EMA"]["enable"] and not tr["EMA"]["replace_model_after_load"]
    assert ex["classifier_only"] and ex["classifier_fit_count"] == 1
    assert ex["policy_updates_per_round"] == ex["reward_fit_count"] == 0
    assert ex["source_policy_step"] is None and not ex["checkpoint_metadata_verified"]
    a = resolve_adaptive_config(dg, classifier_only=True)
    assert not a.bootstrap_on_start and not a.refit_once_on_resume and not a.baseline_probe_on_start
    assert not a.raw_monitor_warm_start
    assert a.audit_fit.get("repeats", 1) == 1
    assert a.audit_fit["disjoint_final_audit"] and a.audit_fit["restore_best"]
    assert a.audit_fit["steps"] is None and a.audit_fit["min_steps"] == 0
    assert a.audit_fit["validation_patience_epochs"] == 25
    assert a.audit_fit["checkpoint_selection_metric"] == "loss"
    assert a.audit_fit["decoder_layers"] == 3
    fit = dict(dg["adaptive_omnifold"]["audit_fit"])
    assert fit["training_population"] == "omnifold_fold" and fit["training_fold"] == 1
    assert fit == original["dgpo"]["adaptive_omnifold"]["audit_fit"]
    assert not a.fixed_schedule_log_raw_audit and not a.raw_audit_enabled
    assert cfg["reward_config"] == original["reward_config"]
    assert cfg["network"] == original["network"]
    assert cfg["platform"]["number_of_workers"] == 16
    assert cfg["platform"]["resources_per_worker"]["GPU"] == 1
    assert cfg["logger"]["wandb"]["fresh_run"]
    assert cfg["logger"]["wandb"]["id"] is None
    assert len(cfg["logger"]["wandb"]["run_name"]) <= 96
    assert cfg["logger"]["wandb"]["classifier_loss_curves_raw"]
    assert ex["comparison_run"] == "227e4975"
    assert str(DEFAULT_CHECKPOINT).endswith("dgpo-epoch=277-next_ep=278-step=2780.ckpt")
    assert select_dgpo_training_state({"state_dict": {}, "dgpo_omnifold_reward_stack": {"fitted": True},
                                      "global_step": 2780}, load_mode=dg["checkpoint_load_mode"]) is None


def test_reads_actual_step_and_pins_last_symlink(tmp_path):
    actual = tmp_path / "epoch=277-step=2780.ckpt"
    torch.save({"state_dict": {"model.weight": torch.ones(1)}, "global_step": 2780, "epoch": 277}, actual)
    last = tmp_path / "last.ckpt"
    last.symlink_to(actual)
    metadata = inspect_checkpoint(last)
    assert metadata == {"path": str(actual), "global_step": 2780, "epoch": 277}
    cfg = configuration(last, tmp_path / "output", metadata=metadata)
    assert cfg["options"]["Training"]["model_checkpoint_load_path"] == str(actual)
    assert cfg["experiment"]["source_policy_step"] == 2780
    assert cfg["experiment"]["source_policy_epoch"] == 277
    assert cfg["experiment"]["checkpoint_metadata_verified"]


@pytest.mark.parametrize("payload", [
    {"state_dict": {}, "global_step": 0, "epoch": -1},
    {"state_dict": {"weight": torch.ones(1)}, "epoch": 20},
    {"state_dict": {"TruthGeneration.visible_conditioning.token_readout.weight": torch.ones(1)},
     "global_step": 10, "epoch": 0},
])
def test_rejects_missing_policy_progress_or_wrong_architecture(tmp_path, payload):
    path = tmp_path / "bad.ckpt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        inspect_checkpoint(path)


def test_dry_run_never_loads_remote_files_or_launches(monkeypatch, capsys):
    import train_h4_checkpoint_audit as launcher
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not load checkpoints or submit work")
    monkeypatch.setattr(launcher, "inspect_checkpoint", forbidden)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.sys, "argv", ["train_h4_checkpoint_audit.py", "--dry-run"])
    launcher.main()
    assert "classifier_only: true" in capsys.readouterr().out


def test_original_data_identity_sampler_classifier_and_budget_are_preserved():
    root = Path(__file__).resolve().parents[1]
    original = deep_update(read_yaml(root / "config/train_diffusion_nersc.yaml"),
                           read_overlay_yaml(root / "config/dgpo_h4_kinematic_adaln_depth3.yaml"))
    cfg = configuration()
    old = resolve_adaptive_config(original["dgpo"])
    new = resolve_adaptive_config(cfg["dgpo"], classifier_only=True)
    assert old.audit_fit == new.audit_fit
    for key in ("seed", "probe_seed", "pool_selection_seed", "crossfit_partition",
                "crossfit_folds", "crossfit_repeats", "pool_data_parquet_dir", "pool_events",
                "probe_max_events", "pool_generation_batch_size", "cache_event_inputs",
                "fixed_audit_panel", "periodic_pair_features_enabled", "visible_pair_rest_frame_enabled"):
        assert getattr(old, key) == getattr(new, key), key
    assert original["platform"] == cfg["platform"]
    assert original["options"]["Dataset"] == cfg["options"]["Dataset"]
    assert original["reward_config"] == cfg["reward_config"]
    assert original["dgpo"]["validation_num_ddim_steps"] == cfg["dgpo"]["validation_num_ddim_steps"] == 20
    network = dict(cfg["network"])
    visible = dict(network["VisibleConditioning"])
    assert not visible.pop("diffusion_token_readout")["enabled"]
    network["VisibleConditioning"] = visible
    assert network == original["network"]


def test_standalone_context_does_not_relax_normal_training_or_warm_start():
    cfg = configuration()
    with pytest.raises(ValueError, match="omnifold_fold audit requires"):
        resolve_adaptive_config(cfg["dgpo"])
    cfg["dgpo"]["adaptive_omnifold"]["trigger"]["warm_start_classifier"] = True
    with pytest.raises(ValueError):
        resolve_adaptive_config(cfg["dgpo"], classifier_only=True)


@pytest.mark.parametrize("source", ["latest", "control", "last", "legacy_probe"])
def test_actual_terminal_path_materializes_correct_populations_and_exits(source):
    matched = source != "legacy_probe"
    cfg = configuration(metadata={"path": "/raw-2780.ckpt", "global_step": 2780, "epoch": 277}) if source == "latest" else (
        token_configuration("last" if source == "last" else "control", audit_checkpoint=Path("/raw.ckpt"), expected_step=1000)
    )
    if not matched:
        # Retain coverage of old saved probe-split configs, not a new launch mode.
        cfg["dgpo"]["adaptive_omnifold"]["audit_fit"]["training_population"] = "probe_split"
        cfg["experiment"]["protocol"] = "h4-token-conditioning-fresh-audit-v1"
    a = resolve_adaptive_config(cfg["dgpo"], classifier_only=True)
    evaluation = SimpleNamespace(n_events=118992)
    training = SimpleNamespace(n_events=208355)
    results = {
        "raw_auc": .85, "raw_balanced_accuracy": .79, "raw_audit_training_steps": 2040,
        "raw_audit_fit_events": 208355 if matched else 71396,
        "raw_audit_early_stop_events": 59465 if matched else 23798,
        "raw_audit_test_events": 59527 if matched else 23798,
        "raw_audit_probe_events": 118992, "raw_audit_steps_per_epoch": 12 if matched else 4,
        "raw_audit_uses_omnifold_fold": int(matched), "raw_audit_training_fold": int(matched),
        "raw_audit_validation_interval_steps": 12 if matched else 4,
        "raw_audit_patience_evaluations": 25,
    }
    tree = ast.parse(Path(trainer.__file__).read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name) and node.test.id == "classifier_only")
    # Execute the real terminal branch, replacing only GPU sampling and the fit.
    wrapper = ast.parse("def terminal():\n    pass\n")
    wrapper.body[0].body = branch.body
    ast.fix_missing_locations(wrapper)
    with mock.patch.object(trainer, "_materialize_adaptive_omnifold_pool",
                           side_effect=[evaluation, training] if matched else [evaluation]) as generate, \
            mock.patch.object(adaptive, "fit_raw_policy_audit", return_value=results) as fit:
        ns = dict(trainer.__dict__)
        ns.update(
            Mapping=Mapping, adaptive_cfg=a, experiment_cfg=cfg["experiment"],
            global_config=SimpleNamespace(experiment=cfg["experiment"]),
            omnifold_source=SimpleNamespace(model_builder=object()),
            omnifold_val_shard="external-validation", omnifold_train_shard="omnifold-training",
            omnifold_val_loader_cfg={"batch_size": 2048}, omnifold_train_loader_cfg={"batch_size": 2048},
            checkpoint_load_mode="weights_only", start_epoch=0, global_step=0,
            model=object(), sampler=object(), device=torch.device("cpu"), world_size=16, rank=0,
            is_rank0=True, num_ddim_val=20, wandb_mod=SimpleNamespace(run=SimpleNamespace(summary={})),
            wandb_active=True, _barrier=mock.Mock(), _finish_wandb_run=mock.Mock(),
            _wandb_log_step=mock.Mock(), _log_omnifold_fit_progress=mock.Mock(),
        )
        exec(compile(wrapper, "<real-classifier-only-path>", "exec"), ns)
        ns["terminal"]()
        assert generate.call_count == (2 if matched else 1)
        assert generate.call_args_list[0].args[0] == "external-validation"
        assert generate.call_args_list[0].kwargs["seed"] == a.probe_seed
        if matched:
            call = generate.call_args_list[1]
            assert call.args[0] == "omnifold-training"
            assert call.kwargs["training_crossfit_fold"] == (2, 1, a.seed)
            assert call.kwargs["quota_events"] is None
            assert call.kwargs["seed"] == a.probe_seed + 2_000_003
            assert call.kwargs["world_size"] == 16 and call.kwargs["num_ddim_steps"] == 20
            assert fit.call_args.kwargs["training_pool"] is training
        else:
            assert "training_pool" not in fit.call_args.kwargs
        fit.assert_called_once()
        assert fit.call_args.kwargs["pool"] is evaluation
        assert fit.call_args.kwargs["warm_start_cache"] is None
        logged = ns["_wandb_log_step"].call_args.args[1]
        assert logged["classifier_only/policy_updates"] == logged["classifier_only/reward_fits"] == 0
        assert logged["classifier_only/source_policy_step"] == (2780 if source == "latest" else 1000)
        for key in results:
            if "events" in key or "epoch" in key or "fold" in key or "interval" in key or "patience" in key:
                name = "staleness/" + key
                assert trainer._wandb_simplified_keep(name, logged[name]), name
                assert trainer._wandb_critical_keep(name), name
        ns["_finish_wandb_run"].assert_called_once_with(True)


def test_custom_source_has_accurate_run_name():
    cfg = configuration(Path("/custom.ckpt"), metadata={"path": "/custom.ckpt", "global_step": 1200, "epoch": 119})
    assert "step 1200" in cfg["logger"]["wandb"]["run_name"]
    assert "2780" not in cfg["logger"]["wandb"]["run_name"]
