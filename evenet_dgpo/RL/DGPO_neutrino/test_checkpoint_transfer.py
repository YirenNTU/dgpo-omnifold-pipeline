from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL.DGPO_neutrino.checkpoint_transfer import (
    CheckpointTransferProbe, event_fingerprints, isolated_evaluation, paired_statistics,
)


def batch(start=0, count=6):
    return {"x": torch.arange(start, start + count).float().reshape(count, 1, 1),
            "x_mask": torch.ones(count, 1, dtype=torch.bool),
            "conditions": torch.zeros(count, 1), "conditions_mask": torch.ones(count, 1, dtype=torch.bool),
            "x_invisible": torch.zeros(count, 2, 2), "x_invisible_mask": torch.ones(count, 2, dtype=torch.bool)}


def test_paired_stats_use_every_candidate_and_event_clusters():
    before = np.zeros((50, 4))
    after = np.arange(50)[:, None] / 100 + np.array([-3, -1, 1, 3])[None, :]
    stats = paired_statistics(before, after, replicates=500, seed=1)
    assert stats["delta_mean"] == pytest.approx(.245)
    # Replicating candidates must NOT narrow the interval.
    duplicate = paired_statistics(np.tile(before, (1, 10)), np.tile(after, (1, 10)), replicates=500, seed=1)
    assert duplicate["delta_lo95"] == pytest.approx(stats["delta_lo95"])
    assert duplicate["delta_hi95"] == pytest.approx(stats["delta_hi95"])
    assert stats["delta_lo95"] > 0
    with pytest.raises(ValueError, match="Non-finite"):
        paired_statistics(before, after * np.nan, replicates=10, seed=1)


def test_evaluation_preserves_rng_mixed_modes_and_parameters_on_error():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
    model.train()
    model[0].eval()
    modes = [m.training for m in model.modules()]
    state = torch.random.get_rng_state().clone()
    py_state, np_state = random.getstate(), np.random.get_state()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(RuntimeError):
        with isolated_evaluation(model, 123):
            torch.randn(20)
            random.random()
            np.random.random(20)
            assert not any(m.training for m in model.modules())
            raise RuntimeError("test")
    assert torch.equal(state, torch.random.get_rng_state())
    assert py_state == random.getstate()
    assert np.array_equal(np_state[1], np.random.get_state()[1])
    assert modes == [m.training for m in model.modules()]
    for key, value in weights.items():
        assert torch.equal(value, model.state_dict()[key])


class Shard:
    def iter_torch_batches(self, **kwargs):
        yield batch(100, 4)
        yield batch(104, 4)


def make_probe(tmp_path, scorer=None, overrides=None):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    group = {"group_name": "body", "lr": 1e-4, "weight_decay": .001, "betas": (.9, .999), "eps": 1e-8}
    optimizer = SimpleNamespace(param_groups=[group], scheduler=SimpleNamespace(last_epoch=98))
    cfg = {"relative_steps": [0, 1, 5, 20], "source_step": 100, "source_reward_round": 7,
           "world_size": 1, "output_directory": str(tmp_path), "noise_seed": 42, "events_per_rank": 6,
           "evaluation_batch_size": 2, "bootstrap_replicates": 100, "bootstrap_seed": 3,
           "reconstruction_tolerance": .001}
    cfg.update(overrides or {})
    def score(b):
        noise = torch.randn(8, len(b["x"]), 2, 2)
        samples = noise + model.weight.item()
        return samples, samples.mean(dim=(2, 3))
    logs = []
    probe = CheckpointTransferProbe(cfg, model=model, score=scorer or score, validation_shard=Shard(),
        rank=0, world_size=1, source_step=100, reward_round=7, optimizer=optimizer,
        checkpoint={"dgpo_optimizer_state_dict": {"optimizer": {"param_groups": [group]},
                                                 "scheduler": {"last_epoch": 98}}},
        log=lambda metrics, step: logs.append((metrics, step)))
    return probe, model, logs


def test_multibatch_paired_probe_and_actual_relative_endpoints(tmp_path):
    probe, model, logs = make_probe(tmp_path)
    assert len(probe.ids["heldout"]) == 6
    assert len(probe.panels["heldout"]) == 2
    probe.before_update(batch())
    baseline = probe.baselines["heldout"].copy()
    trace = {"train/optimizer_step_ran": 1.,
             "gradient_transfer/reconstruction_actual/relative_error": 1e-6,
             "gradient_transfer/critical_lambda": float("inf")}
    for delta in (1, 5, 20):
        model.weight.data.fill_(delta / 100)
        probe.after_update(100 + delta, trace, 7)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["conclusion"] == "supports_local_transfer"
    assert report["measurements"]["heldout/+20"]["delta_mean"] == pytest.approx(.2, abs=1e-7)
    assert report["measurements"]["update_batch/+1"]["delta_mean"] == pytest.approx(.01, abs=1e-7)
    assert report["gradient_traces"]["20"]["gradient_transfer/critical_lambda"] is None
    assert np.array_equal(baseline, probe.baselines["heldout"])
    assert [s for _, s in logs] == [100, 100, 101, 101, 105, 105, 120, 120]
    saved = np.load(tmp_path / "heldout_step20_rank00.npz")
    assert saved["rewards"].shape == (6, 8)
    assert saved["event_ids"].tolist() == probe.ids["heldout"]
    with pytest.raises(ValueError, match="overlaps"):
        probe.before_update(batch(100))
    with pytest.raises(ValueError, match="round changed"):
        probe.after_update(121, trace, 8)
    with pytest.raises(ValueError, match="Skipped"):
        probe.after_update(121, {"train/optimizer_step_ran": 0}, 7)
    with pytest.raises(ValueError, match="reconstruction"):
        probe.after_update(120, {**trace, "gradient_transfer/reconstruction_actual/relative_error": .1}, 7)


def test_identity_excludes_truth_candidate_but_includes_condition():
    b = batch()
    keys = event_fingerprints(b)
    b["x_invisible"] += 1
    assert keys == event_fingerprints(b)
    b["conditions"] += 1
    assert keys != event_fingerprints(b)


def test_nondeterministic_zero_update_replay_rejected(tmp_path):
    state = [0]
    def score(b):
        state[0] += 1
        return torch.ones(8, len(b["x"]), 2, 2) * state[0], torch.ones(8, len(b["x"]))
    with pytest.raises(ValueError, match="replay failed"):
        make_probe(tmp_path, score)


def test_fifty_update_endpoint_and_shared_initial_samples(tmp_path):
    cfg = {'relative_steps': [0, 1, 5, 20, 35, 50], 'conditioning_ablation': True,
           'update_cache_directory': str(tmp_path/'cache'), 'update_cache_mode': 'write', 'update_seed': 8}
    a, ma, _ = make_probe(tmp_path/'A', overrides=cfg)
    a.before_update(batch())
    b, mb, _ = make_probe(tmp_path/'B', overrides={**cfg, 'update_cache_mode': 'read',
                                                  'paired_panel_directory': str(tmp_path/'A')})
    b.before_update(batch())
    trace = {'train/optimizer_step_ran': 1., 'gradient_transfer/reconstruction_actual/relative_error': 0.}
    for probe, model in ((a, ma), (b, mb)):
        model.weight.data.fill_(.1)
        probe.after_update(150, trace, 7)
    report = json.loads((tmp_path/'B'/'report.json').read_text())
    assert report['primary_endpoint'] == 50
    assert report['status'] == 'complete'
    assert report['measurements']['heldout/+0']['matched_control_initial_max_abs_error'] == 0
    # A changed initial function must fail BEFORE training B or C.
    def wrong_score(x):
        samples = torch.randn(8, len(x['x']), 2, 2) + 1
        return samples, samples.mean((2, 3))
    with pytest.raises(ValueError, match='initial sampling function'):
        make_probe(tmp_path/'C', scorer=wrong_score, overrides={**cfg, 'update_cache_mode': 'read',
                    'paired_panel_directory': str(tmp_path/'A')})
