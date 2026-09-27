from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_dgpo_h4_lastblock_no_ref import CONFIG, assert_contract
from train_neutrino_backend import read_overlay_yaml


def test_live_policy_contract():
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config
    from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _scaled_crossfit_config

    c = read_overlay_yaml(CONFIG)
    assert_contract(c)
    runtime = resolve_adaptive_config(c["dgpo"])
    assert runtime.min_iterations == runtime.max_iterations == 1
    assert runtime.iteration_one_only
    assert runtime.log_only and runtime.monitor_mode == "raw_only"
    for block in (runtime.fit, runtime.audit_fit):
        fit = build_fit_config(block, n_train=250000, n_validation=100000)
        assert fit.steps is None
        assert fit.min_steps == 1000
        assert fit.validation_interval_steps == 250000 // fit.batch_size
        assert fit.validation_patience_evaluations == 10
        assert fit.checkpoint_selection_metric == "loss"
    fold = _scaled_crossfit_config(
        build_fit_config(runtime.fit, n_train=250000, n_validation=100000),
        0.5, min_steps_per_fold=1000, n_train=125000,
        min_epochs_per_fold=1, validation_interval_epochs=1,
        validation_patience_epochs=10,
    )
    assert fold.steps is None
    assert fold.min_steps == 1000
    assert fold.validation_interval_steps == 125000 // fold.batch_size
    assert fold.validation_patience_evaluations == 10
    parent = read_overlay_yaml(ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_direction_no_ref_audit5_100step.yaml")
    for key in ("learning_rate", "learning_rate_body", "weight_decay", "Components", "decoupled_weight_decay"):
        assert c["options"]["Training"][key] == parent["options"]["Training"][key]
    assert c["dgpo"]["adaptive_omnifold"]["audit_fit"]["disjoint_final_audit"]


@pytest.mark.parametrize("path,value", [
    (("experiment", "classifier_only"), True),
    (("dgpo", "reference_trust", "enabled"), True),
    (("dgpo", "adaptive_omnifold", "recalibration", "max_reward_rounds"), 2),
    (("dgpo", "adaptive_omnifold", "recalibration", "max_iterations"), 2),
    (("dgpo", "adaptive_omnifold", "recalibration", "iteration_one_only"), False),
    (("dgpo", "adaptive_omnifold", "recalibration", "train_last_pet_block"), False),
    (("dgpo", "adaptive_omnifold", "audit_fit", "train_backbone"), True),
    (("dgpo", "adaptive_omnifold", "audit_fit", "steps"), 800),
    (("platform", "number_of_workers"), 8),
    (("dgpo", "log_parameter_update_rms"), False),
    (("dgpo", "gradient_conflict", "enabled"), False),
    (("experiment", "retain_installed_classifier"), False),
    (("dgpo", "adaptive_omnifold", "baseline_probe_on_start"), True),
    (("dgpo", "adaptive_omnifold", "audit_fit", "validation_interval_steps"), 40),
    (("dgpo", "adaptive_omnifold", "recalibration", "fit", "steps"), 3000),
])
def test_rejects_wrong_mechanism(path, value):
    c = deepcopy(read_overlay_yaml(CONFIG))
    node = c
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        assert_contract(c)


def test_wandb_preserves_gradient_diagnostics():
    from RL.DGPO_neutrino import dgpo_trainer as trainer
    from unittest.mock import Mock

    keys = [
        "train/grad/global_norm_pre_clip", "train/grad/clip_active",
        "train/parameter_update_rms",
        "gradient_conflict/omnifold/norm", "gradient_conflict/staleness/norm",
        "gradient_conflict/omnifold_staleness/cosine",
        "gradient_conflict/omnifold_staleness/conclusive",
        "gradient_conflict/omnifold/split_cosine",
        "gradient_conflict/staleness/split_cosine",
    ]
    for key in keys:
        assert trainer._wandb_simplified_keep(key, 1.0), key
    wb = Mock()
    trainer._wandb_define_axes(wb, critical=False)
    for key in keys[3:]:
        wb.define_metric.assert_any_call(key, step_metric="global_step", step_sync=False, hidden=False)
    for key in ("frozen_classifier/ensemble/auc", "frozen_classifier/fold01/auc"):
        assert trainer._wandb_simplified_keep(key, .8)
        assert trainer._wandb_critical_keep(key)
    wb.define_metric.assert_any_call("frozen_classifier/*", step_metric="global_step", step_sync=False, hidden=False)


def test_no_hidden_step_zero_classifier_fits():
    from dataclasses import replace
    from unittest.mock import Mock
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import (
        resolve_adaptive_config, bootstrap_baseline_pool, step_zero_raw_audit_enabled,
    )

    cfg = resolve_adaptive_config(read_overlay_yaml(CONFIG)["dgpo"])
    pool = Mock()
    assert bootstrap_baseline_pool(cfg, pool) is None
    assert not step_zero_raw_audit_enabled(cfg)
    pool.prefix.assert_not_called()
    enabled = replace(cfg, baseline_probe_on_start=True)
    assert bootstrap_baseline_pool(enabled, pool) is pool.prefix.return_value
    assert step_zero_raw_audit_enabled(enabled)


def test_frozen_classifier_auc_measures_policy_change_without_training():
    import torch
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldPool, frozen_classifier_metrics
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, FrozenResidualRatioReward
    from dataclasses import replace

    spec = EventPackingSpec({"x": (1, 1)})

    class Score(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.packing_spec = spec

        def forward(self, condition, sample):
            return self.scale * sample[..., 0]

    model = Score()
    stack = FrozenResidualRatioReward(model, (model.state_dict(), model.state_dict()),
                                     checkpoint_iterations=(1, 1), checkpoint_coefficients=(.5, .5))
    stack.eval()
    pool = AdaptiveOmniFoldPool(packed_event=torch.zeros(8, spec.width), truth=torch.ones(8, 4),
                               candidates=-torch.ones(8, 1, 4), packing_spec=spec)
    before = {k: v.clone() for k, v in stack.state_dict().items()}
    metrics = frozen_classifier_metrics(stack, pool, row_budget=3)
    matched = frozen_classifier_metrics(stack, replace(pool, candidates=pool.truth.unsqueeze(1)), row_budget=3)
    for label in ("ensemble", "fold01", "fold02"):
        assert metrics[f"frozen_classifier/{label}/auc"] == 1.0
        assert matched[f"frozen_classifier/{label}/auc"] == .5
    for key, value in stack.state_dict().items():
        torch.testing.assert_close(value, before[key])
    stack.assert_frozen()
