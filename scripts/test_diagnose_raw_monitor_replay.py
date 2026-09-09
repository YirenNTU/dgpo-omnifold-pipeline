from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

SCRIPT = Path(__file__).with_name("diagnose_raw_monitor_replay.py")
spec = importlib.util.spec_from_file_location("diagnose_raw_monitor_replay", SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def checkpoint(state=None):
    return {
        "epoch": 25, "global_step": 260,
        "state_dict": {"model.weight": torch.tensor([[1., 2.]])} if state is None else state,
        "dgpo_adaptive_omnifold_state": {"raw_monitor_state": {
            "protocol": {"schema": "raw-monitor-condition-split-v1", "folds": 5, "seed": 42},
            "state": {"weight": torch.ones(1, 2)},
        }},
        "dgpo_omnifold_reward_stack": {
            "source_bundle_sha256": "a" * 64,
            "reward": {"base_digest": "b" * 64, "increments": [{
                "packing_spec": {"shapes": {"x": [2]}}}]},
        },
    }


class TestCheckpointGuards(unittest.TestCase):
    def test_digest_complete_and_order_independent(self):
        a = {"scalar": torch.tensor(1.), "vector": torch.arange(100.)}
        b = {k: v.clone() for k, v in reversed(list(a.items()))}
        self.assertEqual(replay.state_digest(a), replay.state_digest(b))
        b["vector"][99] += 1  # Unlike legacy sampled base_digest, includes every value.
        self.assertNotEqual(replay.state_digest(a), replay.state_digest(b))

    def test_monitor_provenance_and_missing_cache(self):
        c = checkpoint()
        cache, _, _ = replay.monitor_payload(c)
        self.assertEqual(len(cache["state"]), 1)
        for bad in ({}, {"protocol": {"seed": 42}, "state": cache["state"]}):
            c["dgpo_adaptive_omnifold_state"]["raw_monitor_state"] = bad
            with self.assertRaises(ValueError):
                replay.monitor_payload(c)

    def test_rejects_changed_condition_width(self):
        c = checkpoint()
        c["dgpo_adaptive_omnifold_state"]["raw_monitor_state"]["training_policy"] = {"condition_width": 3}
        with self.assertRaisesRegex(ValueError, "width"):
            replay.monitor_payload(c)

    def test_loaded_policy_exact_and_famo_exception(self):
        m = torch.nn.Linear(2, 1, bias=False)
        saved = {"model.weight": m.weight.detach().clone(), "model.famo.w.task": torch.zeros(1)}
        replay.verify_policy_loaded(m, saved)
        with torch.no_grad():
            m.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "different"):
            replay.verify_policy_loaded(m, saved)
        with self.assertRaisesRegex(ValueError, "missing"):
            replay.verify_policy_loaded(m, {})

    def test_monitor_load_checks_before_mutating(self):
        m = torch.nn.Linear(2, 1, bias=False)
        initial = m.weight.detach().clone()
        for state in ({}, {"weight": torch.ones(3)}, {"weight": torch.full_like(m.weight, float("nan"))}):
            with self.assertRaises(ValueError):
                replay.strict_monitor_load(m, state)
            self.assertTrue(torch.equal(initial, m.weight))
        replay.strict_monitor_load(m, {"weight": torch.ones_like(m.weight)})
        self.assertFalse(m.training)
        self.assertTrue(torch.equal(m.weight, torch.ones_like(m.weight)))

    def test_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "report.json"
            replay._exclusive_json(p, {"x": 1})
            with self.assertRaises(FileExistsError):
                replay._exclusive_json(p, {"x": 2})
            p = Path(tmp) / "scores.pt"
            replay._exclusive_torch_save(p, {"x": torch.ones(1)})
            with self.assertRaises(FileExistsError):
                replay._exclusive_torch_save(p, {})

    def test_prepare_no_training_or_overwrite(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = checkpoint()
            old, new = root / "old.ckpt", root / "new.ckpt"
            replay._exclusive_torch_save(old, c)
            replay._exclusive_torch_save(new, c)
            before = old.read_bytes(), new.read_bytes()
            raw = {
                "options": {"Training": {"EMA": {}}, "Dataset": {"normalization_file": "normalization.pt"}},
                "platform": {"data_parquet_dir": "train", "data_parquet_val_dir": "train"},
                "reward_config": {"omnifold": {"backbone_checkpoint": "backbone.ckpt"}},
                "dgpo": {"num_ddim_steps": 20, "adaptive_omnifold": {"single_pool_split_seed": 42}},
            }
            base, overlay = root / "base.yaml", root / "overlay.yaml"
            base.write_text(yaml.safe_dump(raw))
            overlay.write_text("{}")
            cfg = {"base_config": str(base), "old_overlay": str(overlay), "new_overlay": str(overlay),
                   "policy_checkpoint": str(old), "old_checkpoint": str(old), "new_checkpoint": str(new),
                   "expected_policy_state_sha256": replay.state_digest(c["state_dict"]),
                   "output_dir": str(root / "output"), "workers": 1, "pool_events": 1000,
                   "generation_batch_size": 32, "score_batch_size": 32, "minimum_validation_events": 10}
            prepared = replay.prepare(cfg)
            self.assertTrue((Path(prepared["output_dir"]) / "manifest.json").is_file())
            self.assertEqual(before, (old.read_bytes(), new.read_bytes()))
            with self.assertRaises(FileExistsError):
                replay.prepare(cfg)
            c["state_dict"]["model.weight"] += 1
            with new.open("wb") as f:
                torch.save(c, f)
            cfg["output_dir"] = str(root / "other")
            with self.assertRaisesRegex(ValueError, "policy differs"):
                replay.prepare(cfg)
            self.assertFalse((root / "other").exists())


class TestPairedReplay(unittest.TestCase):
    def test_worker_end_to_end_on_one_fixed_pool(self):
        """Run the worker orchestration with fake heavy models, real splits/scoring/I/O."""
        import json
        from types import SimpleNamespace as NS
        from contextlib import ExitStack
        import ray.train
        import ray.train.torch
        from RL.DGPO_neutrino import model_utils, dgpo_trainer
        from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldPool
        from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import payload_sha256
        from evenet.control.global_config import global_config, DotDict

        class Judge(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, condition, sample):
                return sample[..., 0] * self.scale

        class Builder:
            base_digest = "b" * 64
            def __init__(self, **kwargs):
                pass
            def make_classifier(self, *args, **kwargs):
                return Judge()

        n = 1000
        g = torch.Generator().manual_seed(19)
        fields = {"x": torch.randn(n, 2, 3, generator=g), "x_mask": torch.ones(n, 2, 1, dtype=torch.bool),
                  "conditions": torch.randn(n, 1, 2, generator=g), "conditions_mask": torch.ones(n, 1, 1, dtype=torch.bool)}
        _, clean_spec = evenet_ratio.pack_event_inputs(fields)
        for leg in ("a", "b"):
            for axis in ("x", "y", "z"):
                fields[f"lead_{leg}_visible_p{axis}"] = torch.randn(n, generator=g)
        packed, full_spec = evenet_ratio.pack_event_inputs(fields, include_pairwise_context=True)
        pool = AdaptiveOmniFoldPool(packed, torch.ones(n, 4), -torch.ones(n, 1, 4), full_spec)
        policy = torch.nn.Linear(2, 1, bias=False)
        c = checkpoint({"model.weight": policy.weight.detach().clone()})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backbone = root / "backbone.ckpt"
            replay._exclusive_torch_save(backbone, {"state_dict": {"x": torch.ones(1)}})
            with backbone.open("rb") as f:
                backbone_hash = replay.hashlib.file_digest(f, "sha256").hexdigest()
            identity = {"schema_version": 2, "kind": "ztautau_in_dgpo_omnifold_bootstrap",
                        "base_digest": Builder.base_digest, "policy_reference_sha256": backbone_hash}
            cfg = {"output_dir": str(root), "pool_events": n, "minimum_validation_events": 10,
                   "expected_policy_state_sha256": replay.state_digest(c["state_dict"]),
                   "generation_batch_size": 32, "ddim_steps": 20, "score_batch_size": 17,
                   "seed": 42, "provenance": {}, "new_runtime": "new.yaml", "old_runtime": "old.yaml"}
            for label, scale, spec in (("old", 1., clean_spec), ("new", 0., full_spec)):
                cp = deepcopy(c)
                cache = cp["dgpo_adaptive_omnifold_state"]["raw_monitor_state"]
                cache["state"] = {"scale": torch.tensor(scale)}
                cp["dgpo_omnifold_reward_stack"]["reward"]["increments"][0]["packing_spec"] = spec.to_dict()
                path = root / f"{label}.ckpt"
                replay._exclusive_torch_save(path, cp)
                cfg[label + "_checkpoint"] = str(path)
                cfg["provenance"][label] = {"packing_spec": spec.to_dict(), "split_protocol": cache["protocol"],
                    "legacy_base_digest": Builder.base_digest, "source_bundle_sha256": payload_sha256(identity),
                    "monitor_state_sha256": replay.state_digest(cache["state"])}
            cfg["policy_checkpoint"] = cfg["old_checkpoint"]
            originals = {label: Path(cfg[label + "_checkpoint"]).read_bytes() for label in ("old", "new")}
            runtime = NS(dgpo=NS(adaptive_omnifold=NS(recalibration={}, audit_fit={})),
                         reward_config=NS(omnifold=NS(backbone_checkpoint=str(backbone))))
            ctx = NS(get_world_rank=lambda: 0, get_world_size=lambda: 1)
            with ExitStack() as stack:
                stack.enter_context(patch.object(ray.train, "get_context", return_value=ctx))
                stack.enter_context(patch.object(ray.train.torch, "get_device", return_value=torch.device("cpu")))
                stack.enter_context(patch.object(ray.train, "get_dataset_shard", return_value=object()))
                report_call = stack.enter_context(patch.object(ray.train, "report"))
                stack.enter_context(patch.object(global_config, "load_yaml"))
                stack.enter_context(patch.object(global_config, "_global_config", DotDict({"dgpo": {}})))
                stack.enter_context(patch.object(model_utils, "load_evenet_model_for_dgpo", return_value=NS(model=policy)))
                stack.enter_context(patch.object(model_utils, "load_training_config", return_value=runtime))
                stack.enter_context(patch.object(model_utils, "load_normalization_dict", return_value={}))
                generation = stack.enter_context(patch.object(dgpo_trainer, "_materialize_adaptive_omnifold_pool", return_value=pool))
                stack.enter_context(patch.object(evenet_ratio, "EvenetAdapterModelBuilder", Builder))
                stack.enter_context(patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("must not train")))
                replay.replay_worker(cfg)
            generation.assert_called_once()
            report_call.assert_called_once()
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["metrics"]["old"]["auc"], 1.)
            self.assertEqual(report["metrics"]["new"]["auc"], .5)
            self.assertEqual(report["classifier_fits"], 0)
            self.assertTrue(report["policy_unchanged"])
            self.assertTrue((root / "pool.pt").is_file())
            self.assertTrue((root / "scores.pt").is_file())
            for label in ("old", "new"):
                self.assertEqual(originals[label], Path(cfg[label + "_checkpoint"]).read_bytes())

    def test_intersection_is_never_union_and_preserves_duplicates(self):
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _identity_crossfit_splits
        g = torch.Generator().manual_seed(7)
        old = torch.randn(2000, 4, generator=g)
        new = torch.cat((old, torch.randn(2000, 2, generator=g)), dim=1)
        old, new = torch.cat((old, old)), torch.cat((new, new))
        p = {"folds": 5, "seed": 42}
        idx, counts = replay.shared_validation_indices([old, new], [p, p], minimum=20)
        old_train, old_val = _identity_crossfit_splits(old, **p)[0]
        new_train, new_val = _identity_crossfit_splits(new, **p)[0]
        self.assertEqual(set(idx.tolist()), set(old_val.tolist()) & set(new_val.tolist()))
        self.assertFalse(set(idx.tolist()) & set(old_train.tolist()))
        self.assertFalse(set(idx.tolist()) & set(new_train.tolist()))
        self.assertLess(len(idx), min(counts))
        keep = set(idx.tolist())
        self.assertTrue(all((i + 2000 in keep) for i in keep if i < 2000))
        order = torch.randperm(len(old), generator=g)
        reordered, _ = replay.shared_validation_indices([old[order], new[order]], [p, p], minimum=20)
        self.assertEqual(set(order[reordered].tolist()), keep)
        with self.assertRaisesRegex(ValueError, "Increase pool_events"):
            replay.shared_validation_indices([old, new], [p, p], minimum=len(old))

    def test_repacking_does_not_shift_clean_features(self):
        from types import SimpleNamespace
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, pack_event_inputs
        fields = {"x": torch.randn(10, 2, 3), "x_mask": torch.ones(10, 2, 1, dtype=torch.bool),
                  "conditions": torch.randn(10, 1, 2), "conditions_mask": torch.ones(10, 1, 1, dtype=torch.bool)}
        clean, clean_spec = pack_event_inputs(fields)
        for leg in ("a", "b"):
            for axis in ("x", "y", "z"):
                fields[f"lead_{leg}_visible_p{axis}"] = torch.randn(10)
        packed, spec = pack_event_inputs(fields, include_pairwise_context=True)
        pool = SimpleNamespace(packed_event=packed, packing_spec=spec)
        self.assertTrue(torch.equal(clean, replay.repack_pool(pool, clean_spec)))

    def test_real_score_path_unweighted_and_eval_only(self):
        class Judge(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.))
            def forward(self, condition, sample):
                self.assert_eval = not self.training
                return sample[..., 0] * self.scale
        model = Judge()
        condition = torch.zeros(20, 3)
        truth, raw = torch.ones(20, 1), -torch.ones(20, 1, 1)
        with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No optimization allowed")):
            metrics, scores = replay.score_monitor(model, condition, truth, raw, 7)
            null, _ = replay.score_monitor(model, condition, truth, truth[:, None], 7)
        self.assertEqual(metrics["auc"], 1.)
        self.assertEqual(metrics["balanced_accuracy"], 1.)
        self.assertEqual(null["auc"], .5)
        self.assertEqual(null["balanced_accuracy"], .5)
        self.assertTrue(model.assert_eval)
        self.assertIsNone(model.scale.grad)
        self.assertEqual(len(scores["truth_logits"]), 20)


if __name__ == "__main__":
    unittest.main()
