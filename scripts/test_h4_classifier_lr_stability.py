"""Regression checks for the single-variable classifier LR replay."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_h4_classifier_lr_stability import validated_config, DIAGNOSTIC_CONFIG, SCHEDULER_CONFIG, OLD_CHECKPOINT_CONFIG
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig, fit_density_ratio
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


def test_replay_contract_and_config_parsing():
    config = validated_config()
    fit = build_fit_config(config["dgpo"]["adaptive_omnifold"]["audit_fit"],
                           n_train=20000, n_validation=2000)
    assert fit.adapter_learning_rate == fit.decoder_learning_rate == 5e-5
    assert fit.decoder_learning_rate_scope == "all"
    assert fit.log_parameter_updates
    assert fit.steps is None and fit.min_steps == 1000


def test_old_checkpoint_keeps_classifier_data_and_schedule_identical():
    current = validated_config(SCHEDULER_CONFIG)
    old = validated_config(OLD_CHECKPOINT_CONFIG)
    assert old['experiment']['source_policy_step'] == 1110
    assert old['experiment']['source_wandb_run'].endswith('/c4a91e07')
    assert old['options']['Training']['model_checkpoint_load_path'].endswith('step=1110.ckpt')
    assert old['logger']['wandb']['id'] == 'h4clfold1'
    assert old['platform'] == current['platform']
    assert old['dgpo']['checkpoint_load_mode'] == 'weights_only'
    assert not old['dgpo']['auto_resume_from_last']
    old_adaptive = old['dgpo']['adaptive_omnifold']
    new_adaptive = current['dgpo']['adaptive_omnifold']
    assert old_adaptive['recalibration'] == new_adaptive['recalibration']
    old_fit, new_fit = dict(old_adaptive['audit_fit']), dict(new_adaptive['audit_fit'])
    assert old_fit.pop('diagnostic_snapshot_dir') != new_fit.pop('diagnostic_snapshot_dir')
    assert old_fit == new_fit
    assert old['options']['Training']['model_checkpoint_save_path'] != current['options']['Training']['model_checkpoint_save_path']
    assert old['nersc']['ray']['results_dir'] != current['nersc']['ray']['results_dir']


def test_old_checkpoint_rejects_wrong_source():
    import train_h4_classifier_lr_stability as launcher
    config = validated_config(OLD_CHECKPOINT_CONFIG)
    config['options']['Training']['model_checkpoint_load_path'] = '/wrong/step=1110.ckpt'
    original = launcher.read_overlay_yaml
    with patch.object(launcher, 'read_overlay_yaml', side_effect=lambda path: config if path == OLD_CHECKPOINT_CONFIG else original(path)):
        with pytest.raises(ValueError, match='pinned old-DGPO'):
            validated_config(OLD_CHECKPOINT_CONFIG)


@pytest.mark.parametrize('path', [DIAGNOSTIC_CONFIG, SCHEDULER_CONFIG, OLD_CHECKPOINT_CONFIG])
def test_runtime_classifier_budget_guard_accepts_replays(path):
    # Execute the actual terminal-path guard without loading a GPU policy/data.
    import ast
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py').read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name) and node.test.id == 'classifier_only')
    index = next(i for i, node in enumerate(branch.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'classifier_lr_replay' for t in node.targets))
    guard = ast.Module(body=branch.body[index:index + 3], type_ignores=[])
    resolved = validated_config(path)
    namespace = dict(_dgpo_cfg_get=lambda cfg, key, default: cfg.get(key, default),
                     global_config=SimpleNamespace(experiment=resolved['experiment']),
                     adaptive_cfg=SimpleNamespace(audit_fit=resolved['dgpo']['adaptive_omnifold']['audit_fit']))
    exec(compile(guard, '<classifier-budget-guard>', 'exec'), namespace)
    assert namespace['classifier_lr_replay']


def test_diagnostic_overlay_preserves_lr_and_has_separate_run():
    base = validated_config()
    diag = validated_config(DIAGNOSTIC_CONFIG)
    assert base['logger']['wandb']['id'] == 'h4clflr1'
    assert diag['logger']['wandb']['id'] == 'h4clfd1'
    fit = build_fit_config(diag['dgpo']['adaptive_omnifold']['audit_fit'], n_train=20000, n_validation=2000)
    assert fit.diagnostic_enabled and fit.diagnostic_max_snapshots == 2
    assert fit.adapter_learning_rate == fit.decoder_learning_rate == 5e-5
    assert fit.learning_rate == 2e-4 and fit.backbone_learning_rate == 1e-5


def test_scheduler_overlay_enables_both_paths_without_changing_peak_lr():
    resolved = validated_config(SCHEDULER_CONFIG)
    adaptive = resolved['dgpo']['adaptive_omnifold']
    assert resolved['logger']['wandb']['id'] == 'h4clfcos1'
    for block in (adaptive['audit_fit'], adaptive['recalibration']['fit']):
        fit = build_fit_config(block, n_train=20000, n_validation=2000)
        fit.validate()
        assert fit.lr_scheduler == 'cosine'
        assert fit.lr_warmup_epochs == 1 and fit.lr_cosine_epochs == 250
        assert fit.lr_min_ratio == .1 and fit.steps is None
    fit = build_fit_config(adaptive['audit_fit'], n_train=20000, n_validation=2000)
    assert fit.adapter_learning_rate == fit.decoder_learning_rate == 5e-5
    assert fit.learning_rate == 2e-4 and fit.backbone_learning_rate == 1e-5


def test_full_dgpo_scheduler_overlay_preserves_parent_except_schedule():
    from train_neutrino_backend import read_overlay_yaml
    root = Path(__file__).resolve().parents[1]
    parent = read_overlay_yaml(root / 'config/dgpo_omnifold_ztautau_10pct_h4_lastblock_no_ref_100step.yaml')
    resolved = read_overlay_yaml(root / 'config/dgpo_h4_lastblock_classifier_cosine.yaml')
    assert not resolved['experiment']['classifier_only']
    assert resolved['platform'] == parent['platform']
    assert resolved['logger']['wandb']['id'] != parent['logger']['wandb']['id']
    for key in ('audit_fit', 'recalibration'):
        block = dict(resolved['dgpo']['adaptive_omnifold'][key])
        old = parent['dgpo']['adaptive_omnifold'][key]
        if key == 'recalibration':
            block, old = dict(block['fit']), old['fit']
        assert block.pop('lr_scheduler') == 'cosine'
        assert block.pop('lr_warmup_epochs') == 1
        assert block.pop('lr_cosine_epochs') == 250
        assert block.pop('lr_min_ratio') == .1
        assert block == old


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_invalid_adapter_lr(rate):
    with pytest.raises(ValueError):
        RatioFitConfig(adapter_learning_rate=rate).validate()


@pytest.mark.parametrize("log_updates", [False, True])
def test_group_lrs_and_actual_updates(log_updates):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.PET = nn.Module()
            self.backbone.PET.adapters = nn.Linear(1, 1)
            self.backbone.PET.transformer_blocks = nn.Linear(1, 1)
            self.bank = nn.Module()
            self.bank.decoder = nn.Module()
            self.bank.decoder.blocks = nn.Linear(1, 1)
            self.bank.decoder.input = nn.Linear(1, 1)
            self.bank.output = nn.Linear(1, 1)
            self.bank.position_encoder = nn.Linear(1, 1)

        def forward(self, condition, sample):
            return sum(layer(sample).squeeze(-1) for layer in self.modules()
                       if isinstance(layer, nn.Linear))

    model = Model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizers, rows = [], []
    adamw = torch.optim.AdamW

    def capture(*args, **kwargs):
        optimizer = adamw(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    condition = torch.zeros(8, 1)
    with patch("torch.optim.AdamW", side_effect=capture):
        fit_density_ratio(model, condition, torch.ones(8, 1), torch.ones(8),
                          condition, -torch.ones(8, 1), torch.ones(8),
                          RatioFitConfig(steps=1, batch_size=8, learning_rate=2e-4,
                                         backbone_learning_rate=1e-5,
                                         adapter_learning_rate=5e-5, decoder_learning_rate=5e-5,
                                         decoder_learning_rate_scope="all", log_parameter_updates=log_updates,
                                         progress_interval_steps=1), seed=23,
                          progress_callback=rows.append)
    entries = [(id(p), g["lr"]) for g in optimizers[0].param_groups for p in g["params"]]
    assert len(entries) == len(set(pid for pid, _ in entries)) == len(before)
    rates = dict(entries)
    for name, parameter in model.named_parameters():
        expected = (5e-5 if name.startswith(("bank.decoder.", "backbone.PET.adapters."))
                    else 1e-5 if name.startswith(("backbone.", "bank.position_encoder.")) else 2e-4)
        assert rates[id(parameter)] == expected
    for name, rate in {"head": 2e-4, "backbone": 1e-5, "decoder": 5e-5, "adapter": 5e-5}.items():
        assert rows[-1][f"optimizer_group_lr_{name}"] == rate
    if not log_updates:
        assert "parameter_update_rms_adapter" not in rows[-1]
        return
    delta = torch.cat([(p.detach() - before[n]).flatten() for n, p in model.named_parameters()
                       if n.startswith("backbone.PET.adapters.")])
    assert rows[-1]["parameter_update_rms_adapter"] == pytest.approx(delta.square().mean().sqrt().item())
    assert rows[-1]["parameter_update_rms_adapter"] > 0
