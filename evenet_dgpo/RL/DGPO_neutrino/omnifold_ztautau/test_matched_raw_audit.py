"""Raw audit: exact reward-fold training identities, independent external test."""
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest
import torch

from RL.DGPO_neutrino.omnifold_ztautau import adaptive, evenet_ratio, ratio_fit
from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config


ROOT = Path(__file__).resolve().parents[4]


def resolved(overlay="dgpo_h4_visible_adaln_depth3.yaml"):
    from train_neutrino_backend import deep_update, read_yaml, read_overlay_yaml
    config = deep_update(read_yaml(ROOT / "config/train_diffusion_nersc.yaml"),
                         read_overlay_yaml(ROOT / "config" / overlay))
    return config, adaptive.resolve_adaptive_config(config["dgpo"])


def pool(start, count):
    spec = evenet_ratio.EventPackingSpec({"x": (1, 1), "x_mask": (1, 1),
                                         "conditions": (1, 1), "conditions_mask": (1,)})
    ids = torch.arange(start, start + count, dtype=torch.float32)
    return adaptive.AdaptiveOmniFoldPool(
        packed_event=ids[:, None].expand(-1, spec.width).clone(),
        truth=ids[:, None].expand(-1, 4).clone(),
        candidates=(ids + .1)[:, None, None].expand(-1, 1, 4).clone(), packing_spec=spec,
    )


class NullRatio(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, condition, sample):
        return self.bias.expand(sample.shape[:-1])


@pytest.mark.parametrize("overlay", ["dgpo_h4_visible_adaln_depth3.yaml", "dgpo_h4_kinematic_adaln_depth3.yaml"])
def test_both_configs_use_fixed_fold_without_changing_omnifold(overlay):
    config, cfg = resolved(overlay)
    assert cfg.audit_fit["training_population"] == "omnifold_fold"
    assert cfg.audit_fit["training_fold"] == 1
    assert cfg.audit_fit["disjoint_final_audit"]
    assert cfg.fit["min_steps"] == cfg.audit_fit["min_steps"] == 0
    assert cfg.fit["validation_patience_epochs"] == cfg.audit_fit["validation_patience_epochs"] == 25
    assert cfg.fit["steps"] is cfg.audit_fit["steps"] is None
    assert cfg.crossfit_folds == 2 and cfg.crossfit_repeats == 1
    assert cfg.min_iterations == cfg.max_iterations == 2
    assert cfg.staleness_every_n_epochs == 5
    assert cfg.max_reward_age_epochs == 20
    assert cfg.fixed_schedule_log_raw_audit and cfg.cache_event_inputs
    assert config["platform"]["number_of_workers"] == 16
    assert config["platform"]["data_parquet_dir"] != config["platform"]["data_parquet_val_dir"]


def test_legacy_config_keeps_probe_split():
    _, cfg = resolved("dgpo_h4_visible_adaln.yaml")
    assert cfg.audit_fit.get("training_population", "probe_split") == "probe_split"


@pytest.mark.parametrize("key,value", [("training_fold", 0), ("training_fold", 3),
                                       ("training_population", "typo"), ("disjoint_final_audit", False)])
def test_invalid_matched_audit_config_fails_before_training(key, value):
    config, _ = resolved()
    config["dgpo"]["adaptive_omnifold"]["audit_fit"][key] = value
    with pytest.raises(ValueError):
        adaptive.resolve_adaptive_config(config["dgpo"])


def test_local_hash_matches_reward_fold_across_sixteen_unequal_shards():
    inputs = pool(0, 1003).identity_inputs
    seed = evenet_ratio._crossfit_repeat_seed(20260920, 1)
    fit, heldout = evenet_ratio._identity_crossfit_splits(inputs, folds=2, seed=seed)[0]
    labels = torch.cat([evenet_ratio._local_identity_fold_labels(chunk, folds=2, seed=seed)
                        for chunk in torch.tensor_split(inputs, 16)])
    assert torch.equal(torch.where(labels != 0)[0], fit)
    assert torch.equal(torch.where(labels == 0)[0], heldout)
    assert torch.equal(labels.flip(0), evenet_ratio._local_identity_fold_labels(inputs.flip(0), folds=2, seed=seed))


def test_external_split_keeps_all_training_rows_and_is_identity_stable():
    train, evaluation = pool(0, 200), pool(1000, 300)
    combined, indices = adaptive._external_raw_audit_population(train, evaluation, seed=42)
    sets = [set(combined.truth[idx, 0].tolist()) for idx in indices]
    assert sets[0] == set(range(200))
    assert sets[1] | sets[2] == set(range(1000, 1300))
    assert all(sets[i].isdisjoint(sets[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    combined2, indices2 = adaptive._external_raw_audit_population(
        train.select(torch.arange(199, -1, -1)), evaluation.select(torch.arange(299, -1, -1)), seed=42,
    )
    assert sets == [set(combined2.truth[idx, 0].tolist()) for idx in indices2]


def test_real_raw_fit_trains_only_selected_fold_and_tests_only_final_half():
    _, cfg = resolved()
    cfg = replace(cfg, audit_fit={**cfg.audit_fit, "batch_size": 16, "steps": 3,
                                 "validation_interval_steps": 1, "validation_batch_size": 128,
                                 "adapter_learning_rate": None, "decoder_learning_rate": None,
                                 "fourier_output_standardization": False})
    train, evaluation = pool(0, 200), pool(1000, 120)
    combined, splits = adaptive._external_raw_audit_population(train, evaluation, seed=42 + 7919)
    expected = [set(combined.truth[idx, 0].tolist()) for idx in splits]
    observed, models = {}, []
    real_fit = ratio_fit.fit_density_ratio
    real_score = evenet_ratio._score_population

    def fit(*args, **kwargs):
        observed["fit"] = set(args[1][:, 0].tolist())
        observed["early"] = set(args[9][0][:, 0].tolist())
        result = real_fit(*args, **kwargs)
        observed["fitted"] = True
        return result

    def score(model, conditions, samples, batch):
        if observed.get("fitted"):
            observed.setdefault("final", set()).update(conditions[:, 0].tolist())
        return real_score(model, conditions, samples, batch)

    def factory(*args, **kwargs):
        def make_model():
            model = NullRatio()
            models.append(model)
            return model
        return make_model

    with mock.patch.object(adaptive, "peft_bank_factory", side_effect=factory), \
            mock.patch.object(ratio_fit, "fit_density_ratio", side_effect=fit), \
            mock.patch.object(evenet_ratio, "_score_population", side_effect=score):
        result = adaptive.fit_raw_policy_audit(
            pool=evaluation, training_pool=train, model_builder=object(), cfg=cfg,
            device=torch.device("cpu"), seed=42,
        )
    assert [observed["fit"], observed["early"], observed["final"]] == expected
    assert result["raw_audit_fit_events"] == 200
    assert result["raw_audit_probe_events"] == 120
    assert result["raw_audit_early_stop_events"] + result["raw_audit_test_events"] == 120
    assert result["raw_audit_steps_per_epoch"] == 12
    assert result["raw_audit_training_steps"] == 3
    assert result["raw_audit_training_epochs"] == .25
    assert result["raw_classifier_warm_started"] == 0
    assert result["raw_audit_uses_omnifold_fold"] == 1
    assert len(models) == 1


def test_budget_uses_whole_selected_fold_not_sixty_percent_of_validation():
    _, cfg = resolved()
    fit = build_fit_config(cfg.audit_fit, n_train=208350, n_validation=59501)
    assert fit.validation_interval_steps == 12
    assert fit.validation_patience_evaluations == 25
    assert fit.min_steps == 0 and fit.steps is None
    assert fit.restore_best and fit.checkpoint_selection_metric == "loss"


def test_missing_training_fold_never_silently_falls_back_to_small_pool():
    _, cfg = resolved()
    with pytest.raises(ValueError, match="freshly generated training fold"):
        adaptive.fit_raw_policy_audit(pool=pool(0, 100), cfg=cfg, model_builder=object(),
                                     device=torch.device("cpu"), seed=42)


@pytest.mark.parametrize("indices", [
    (torch.arange(60), torch.arange(60, 80), torch.arange(79, 99)),
    (torch.arange(60), torch.arange(60, 80), torch.arange(81, 101)),
])
def test_explicit_split_overlap_or_out_of_range_fails(indices):
    p = pool(0, 100)
    with pytest.raises(ValueError, match="disjoint and cover"):
        evenet_ratio.fit_independent_evenet_audit(
            model_factory=NullRatio, data_condition=p.packed_event, data_sample=p.truth,
            gen_condition=p.packed_event, gen_sample=p.candidates, gen_weight=torch.ones(100, 1),
            fit_config=ratio_fit.RatioFitConfig(steps=1, batch_size=16), seed=42,
            explicit_split_indices=indices,
        )
