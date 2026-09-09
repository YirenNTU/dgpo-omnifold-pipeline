from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

SCRIPT = Path(__file__).with_name("ablate_raw_monitor_initialization.py")
spec = importlib.util.spec_from_file_location("ablate_raw_monitor_initialization", SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def settings(root):
    return {"replay_dir": str(root / "replay"), "output_dir": str(root / "ablation"),
            "workers": 1, "steps": 4, "seed": 42, "batch_size": 16, "microbatch_size": 8,
            "score_batch_size": 16, "learning_rate": .0002, "weight_decay": .0005,
            "gradient_clip_norm": 5., "evaluation_steps": [0, 1, 2, 4],
            "minimum_validation_events": 2, "positive_control_auc_tolerance": .005}


def simple_states():
    cold = {"bank.output.weight": torch.zeros(1, 4), "bank.decoder.weight": torch.ones(4, 4),
            "bank.position_encoder.weight": torch.ones(2, 4),
            **{prefix + "weight": torch.full((2, 2), float(i + 1))
               for i, prefix in enumerate(ablation.FEATURE_PREFIXES.values())}}
    return cold, {k: v + 7 for k, v in cold.items()}


class TestInitialization(unittest.TestCase):
    def test_exact_transplant_and_source_immutability(self):
        cold, old = simple_states()
        hashes = ablation.replay.state_digest(cold), ablation.replay.state_digest(old)
        a, b, c = (ablation.arm_initial_state(arm, cold, old) for arm in ablation.ARMS)
        for k in cold:
            self.assertTrue(torch.equal(a[k], cold[k]))
            self.assertTrue(torch.equal(c[k], old[k]))
            expected = old[k] if k.startswith(tuple(ablation.FEATURE_PREFIXES.values())) else cold[k]
            self.assertTrue(torch.equal(b[k], expected))
        b["bank.output.weight"] += 100
        self.assertEqual(hashes, (ablation.replay.state_digest(cold), ablation.replay.state_digest(old)))

    def test_rejects_mismatch_nonfinite_missing_feature_groups(self):
        cold, old = simple_states()
        bad = deepcopy(old); bad.pop(next(iter(bad)))
        with self.assertRaisesRegex(ValueError, "keys"):
            ablation.arm_initial_state("B_old_features", cold, bad)
        for value in (torch.zeros(9), torch.full((1, 4), float("nan"))):
            bad = deepcopy(old); bad["bank.output.weight"] = value
            with self.assertRaises(ValueError):
                ablation.arm_initial_state("B_old_features", cold, bad)
        for prefix in ablation.FEATURE_PREFIXES.values():
            subset = {k: v for k, v in cold.items() if not k.startswith(prefix)}
            with self.assertRaisesRegex(ValueError, "No trainable"):
                ablation.arm_initial_state("A_cold", subset, subset)
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            ablation.arm_initial_state("D", cold, old)

    def test_settings_reject_unequal_geometry_and_invalid_budget(self):
        cfg = settings(Path("/tmp/test"))
        ablation.validate_settings(cfg)
        for updates in ({"workers": 3}, {"steps": 0}, {"evaluation_steps": []},
                        {"evaluation_steps": [1, 4]}, {"evaluation_steps": [0, 4, 4]},
                        {"evaluation_steps": [0, 3]}, {"learning_rate": float("nan")},
                        {"weight_decay": -1}, {"seed": True}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                ablation.validate_settings({**cfg, **updates})


def fixture(root):
    """A synthetic complete replay artifact, including exact old/new hash splits."""
    from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _event_batch_with_pair_context
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import pack_event_inputs, _identity_crossfit_splits
    cfg = settings(root)
    source = Path(cfg["replay_dir"]); source.mkdir()
    torch.manual_seed(9)
    batch = _event_batch_with_pair_context(1000)
    old_condition, old_spec = pack_event_inputs(batch)
    condition, new_spec = pack_event_inputs(batch, include_pairwise_context=True)
    protocol = {"schema": "raw-monitor-condition-split-v1", "seed": 42, "folds": 5}
    conditions = [old_condition, condition]
    val, counts = ablation.replay.shared_validation_indices(conditions, [protocol, protocol], minimum=2)
    _, old_state = simple_states()
    policy = {"model.weight": torch.tensor([1., 2.])}
    policy_hash = ablation.replay.state_digest(policy)
    old_hash = ablation.replay.state_digest(old_state)
    checkpoint_path = root / "checkpoints" / "old.ckpt"; checkpoint_path.parent.mkdir()
    checkpoint = {"state_dict": policy, "global_step": 260, "epoch": 25,
                  "dgpo_adaptive_omnifold_state": {"raw_monitor_state": {"protocol": protocol, "state": old_state}},
                  "dgpo_omnifold_reward_stack": {"source_bundle_sha256": "source", "reward": {
                      "base_digest": "body", "increments": [{"packing_spec": old_spec.to_dict()}]}}}
    ablation.replay._exclusive_torch_save(checkpoint_path, checkpoint)
    backbone = root / "pretrain" / "backbone.ckpt"; backbone.parent.mkdir()
    normalization = root / "normalization.pt"
    ablation.replay._exclusive_torch_save(backbone, {"weight": torch.ones(1)})
    ablation.replay._exclusive_torch_save(normalization, {"mean": torch.zeros(1)})
    provenance = {label: {"checkpoint": str(checkpoint_path), "monitor_state_sha256": old_hash,
                          "split_protocol": protocol, "packing_spec": packing.to_dict(),
                          "legacy_base_digest": "body", "source_bundle_sha256": "source"}
                  for label, packing in (("old", old_spec), ("new", new_spec))}
    manifest = {"provenance": provenance, "pool_events": 1000, "expected_policy_state_sha256": policy_hash}
    report = {"policy_loaded_verified": True, "policy_unchanged": True, "policy_updates": 0,
              "classifier_fits": 0, "loaded_policy_sha256": "loaded", "pool_events": 1000,
              "common_validation_events": len(val), "metrics": {"old": {
                  "monitor_state_sha256": old_hash, "backbone_file_sha256": ablation.file_digest(backbone), "auc": .67}}}
    ablation.replay._exclusive_json(source / "manifest.json", manifest)
    ablation.replay._exclusive_json(source / "report.json", report)
    ablation.replay._exclusive_torch_save(source / "scores.pt", {"validation_indices": val})
    ablation.replay._exclusive_torch_save(source / "pool.pt", {"schema_version": 1,
        "policy_state_sha256": "loaded", "packing_spec": new_spec.to_dict(), "packed_event": condition,
        "truth": torch.randn(1000, 4), "raw": torch.randn(1000, 1, 4)})
    with (source / "old_runtime.yaml").open("x") as f:
        yaml.safe_dump({"reward_config": {"omnifold": {"backbone_checkpoint": str(backbone)}},
                        "options": {"Dataset": {"normalization_file": str(normalization)},
                                    "Training": {"EMA": {"replace_model_after_load": False}}}}, f)
    expected_train, _ = _identity_crossfit_splits(old_condition, folds=5, seed=42)[0]
    return cfg, expected_train, val


class TestPreflight(unittest.TestCase):
    def test_preflight_is_read_only_and_split_preserves_old_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, expected_train, expected_val = fixture(root)
            before = {p: ablation.file_digest(p) for p in root.rglob("*") if p.is_file()}
            prepared, splits = ablation.prepare(cfg)
            self.assertFalse(Path(cfg["output_dir"]).exists())
            self.assertTrue(torch.equal(splits["train"], expected_train))
            self.assertTrue(torch.equal(splits["validation"], expected_val))
            self.assertFalse(torch.isin(splits["train"], expected_val).any())
            # Crucial: DO NOT put the rest of the old validation fold into training.
            self.assertLess(len(splits["train"]) + len(expected_val), 1000)
            ablation.create_output(prepared, splits)
            self.assertEqual(before, {p: ablation.file_digest(p) for p in before})
            ablation.verify_sources(prepared)
            with self.assertRaises(FileExistsError):
                ablation.prepare(cfg)
            with self.assertRaises(FileExistsError):
                ablation.create_output(prepared, splits)

    def test_rejects_tampered_pool_policy_checkpoint_monitor_split_backbone(self):
        for target in ("pool_policy", "policy", "monitor", "split", "backbone"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); cfg, _, _ = fixture(root); source = Path(cfg["replay_dir"])
                if target == "pool_policy":
                    p = source / "pool.pt"; x = ablation.tensor_read(p); x["policy_state_sha256"] = "wrong"
                elif target in ("policy", "monitor"):
                    p = root / "checkpoints" / "old.ckpt"; x = ablation.replay.read_checkpoint(p)
                    if target == "policy": x["state_dict"]["model.weight"] = torch.zeros(2)
                    else: x["dgpo_adaptive_omnifold_state"]["raw_monitor_state"]["state"]["bank.output.weight"] = torch.ones(1, 4)
                elif target == "split":
                    p = source / "scores.pt"; x = ablation.tensor_read(p)
                    x["validation_indices"] = x["validation_indices"].flip(0)
                else:
                    p = root / "pretrain" / "backbone.ckpt"; x = {"weight": torch.zeros(1)}
                # Test fixture replacement; never rewrite a memory-mapped open source.
                replacement = p.with_suffix(".replacement")
                ablation.replay._exclusive_torch_save(replacement, x)
                replacement.replace(p)
                with self.assertRaises(ValueError):
                    ablation.prepare(cfg)
                self.assertFalse(Path(cfg["output_dir"]).exists())

    def test_rejects_output_overlap_small_fold_and_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, _, _ = fixture(root)
            with self.assertRaisesRegex(ValueError, "separate"):
                ablation.prepare({**cfg, "output_dir": str(Path(cfg["replay_dir"]) / "child")})
            with self.assertRaisesRegex(ValueError, "smaller"):
                ablation.prepare({**cfg, "batch_size": 2000})
            prepared, _ = ablation.prepare(cfg)
            with (Path(cfg["replay_dir"]) / "report.json").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "Protected source changed"):
                ablation.verify_sources(prepared)


class TestTraining(unittest.TestCase):
    def test_complete_three_arm_worker_on_cached_pool(self):
        from types import SimpleNamespace
        from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone
        from RL.DGPO_neutrino.omnifold_ztautau import evenet_ratio, dgpo_reward
        from RL.DGPO_neutrino import model_utils
        from RL.DGPO_neutrino.omnifold_ztautau.adaptive import AdaptiveOmniFoldPool
        import ray.train
        import ray.train.torch
        torch.set_num_threads(1)
        torch.manual_seed(90)
        base = _FakeZtautauBackbone()
        class AttrDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        def attr(x):
            return AttrDict({k: attr(v) for k, v in x.items()}) if isinstance(x, dict) else x
        def classifier(packing):
            return evenet_ratio.EvenetAdapterRatioClassifier(deepcopy(base), packing,
                train_grouped_sequential_embedding=True, train_invisible_projector=True,
                decoder_hidden_dim=16, decoder_layers=1, decoder_heads=2, head_dropout=.15)
        class Builder:
            def __init__(self, *, config, normalization_dict, checkpoint_path, device,
                         train_grouped_sequential_embedding=False, train_invisible_projector=False,
                         train_backbone=False, **kwargs):
                self.base_digest = "body"
                self._pretrained_body = ablation.clone_state(base.state_dict())
            def make_classifier(self, packing, reset=True):
                return classifier(packing)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg, _, val = fixture(root); source = Path(cfg["replay_dir"])
            manifest = ablation.json_read(source / "manifest.json")
            report = ablation.json_read(source / "report.json")
            runtime = yaml.safe_load((source / "old_runtime.yaml").read_text())
            runtime["dgpo"] = {"float32_matmul_precision": "highest", "adaptive_omnifold": {
                "recalibration": {"train_grouped_sequential_embedding": True,
                                  "train_invisible_projector": True, "train_backbone": False},
                "audit_fit": {"head_dropout": .15, "decoder_hidden_dim": 16, "decoder_layers": 1, "decoder_heads": 2}}}
            with (source / "old_runtime.yaml").open("w") as f: yaml.safe_dump(runtime, f)
            raw = ablation.tensor_read(source / "pool.pt")
            packing = evenet_ratio.EventPackingSpec.from_dict(manifest["provenance"]["old"]["packing_spec"])
            with ablation.seeded(999, torch.device("cpu")): old_model = classifier(packing)
            with torch.no_grad():
                for p in old_model.parameters(): p.add_(.01)
            old = ablation.clone_state(old_model.state_dict())
            pool = AdaptiveOmniFoldPool(packed_event=raw["packed_event"], truth=raw["truth"],
                                       candidates=raw["raw"], packing_spec=evenet_ratio.EventPackingSpec.from_dict(raw["packing_spec"]))
            condition = ablation.replay.repack_pool(pool, packing)
            metrics, _ = ablation.replay.score_monitor(old_model, condition[val], pool.truth[val], pool.candidates[val], 16)
            identity = dgpo_reward.payload_sha256({"schema_version": 2, "kind": "ztautau_in_dgpo_omnifold_bootstrap",
                "base_digest": "body", "policy_reference_sha256": report["metrics"]["old"]["backbone_file_sha256"]})
            for meta in manifest["provenance"].values():
                meta.update(monitor_state_sha256=ablation.replay.state_digest(old), source_bundle_sha256=identity)
            report["metrics"]["old"].update(metrics)
            checkpoint_path = root / "checkpoints" / "old.ckpt"
            cp = ablation.replay.read_checkpoint(checkpoint_path)
            cp["dgpo_adaptive_omnifold_state"]["raw_monitor_state"]["state"] = old
            cp["dgpo_omnifold_reward_stack"]["source_bundle_sha256"] = identity
            replacement = checkpoint_path.with_suffix(".replacement")
            ablation.replay._exclusive_torch_save(replacement, cp); replacement.replace(checkpoint_path)
            for name, value in (("manifest", manifest), ("report", report)):
                with (source / f"{name}.json").open("w") as f: json.dump(value, f)
            prepared, splits = ablation.prepare(cfg)
            ablation.create_output(prepared, splits)
            context = SimpleNamespace(get_world_rank=lambda: 0, get_world_size=lambda: 1)
            with patch.object(ray.train, "get_context", return_value=context), \
                 patch.object(ray.train.torch, "get_device", return_value=torch.device("cpu")), \
                 patch.object(ray.train, "report") as reporter, \
                 patch.object(model_utils, "load_training_config", return_value=attr(runtime)), \
                 patch.object(model_utils, "load_normalization_dict", return_value={}), \
                 patch.object(model_utils, "load_evenet_model_for_dgpo", side_effect=AssertionError("No policy loading")), \
                 patch.object(evenet_ratio, "EvenetAdapterModelBuilder", Builder):
                ablation.ablation_worker(prepared)
            final = ablation.json_read(Path(cfg["output_dir"]) / "report.json")
            self.assertEqual(final["classifier_fits"], 3)
            self.assertEqual(final["policy_updates"], 0)
            self.assertEqual(final["policy_generations"], 0)
            self.assertTrue(final["protected_sources_unchanged"])
            self.assertEqual(set(final["arms"]), set(ablation.ARMS))
            summary = ablation.json_read(Path(cfg["output_dir"]) / "summary.json")
            self.assertEqual([r["step"] for r in summary["matched_step_comparison"]], cfg["evaluation_steps"])
            self.assertEqual(summary["matched_step_comparison"][0]["B_minus_A_auc"], 0.)
            starts = {arm: ablation.tensor_read(Path(cfg["output_dir"]) / arm / "initial_monitor.pt") for arm in ablation.ARMS}
            for k in starts["A_cold"]:
                expected = old[k] if k.startswith(tuple(ablation.FEATURE_PREFIXES.values())) else starts["A_cold"][k]
                self.assertTrue(torch.equal(starts["B_old_features"][k], expected))
                self.assertTrue(torch.equal(starts["C_full_warm"][k], old[k]))
            for arm in ablation.ARMS: self.assertEqual(final["arms"][arm]["steps_completed"], 4)
            reporter.assert_called_once()
            ablation.verify_sources(prepared)
            bad_control = {**prepared, "output_dir": str(root / "bad_control"),
                           "expected_old_auc": metrics["auc"] + .1}
            ablation.create_output(bad_control, splits)
            with patch.object(ray.train, "get_context", return_value=context), \
                 patch.object(ray.train.torch, "get_device", return_value=torch.device("cpu")), \
                 patch.object(model_utils, "load_training_config", return_value=attr(runtime)), \
                 patch.object(model_utils, "load_normalization_dict", return_value={}), \
                 patch.object(evenet_ratio, "EvenetAdapterModelBuilder", Builder), \
                 patch.object(ablation, "run_arm", side_effect=AssertionError("No training on invalid control")), \
                 self.assertRaisesRegex(ValueError, "reproduces replay AUC"):
                ablation.ablation_worker(bad_control)
            self.assertFalse((root / "bad_control" / "A_cold").exists())

    def test_real_classifier_fixed_budget_gradients_and_rng_reproducibility(self):
        from RL.DGPO_neutrino.omnifold_ztautau.test_ztautau_omnifold import _FakeZtautauBackbone, _event_batch
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EvenetAdapterRatioClassifier, pack_event_inputs
        torch.set_num_threads(1)
        torch.manual_seed(12)
        condition, packing = pack_event_inputs(_event_batch(64))
        truth, raw = torch.randn(64, 4), torch.randn(64, 1, 4) + .5
        base = _FakeZtautauBackbone()
        def make():
            return EvenetAdapterRatioClassifier(deepcopy(base), packing,
                train_grouped_sequential_embedding=True, train_invisible_projector=True,
                decoder_hidden_dim=16, decoder_layers=1, decoder_heads=2, head_dropout=.15)
        with ablation.seeded(42, torch.device("cpu")):
            model = make()
        cold = ablation.clone_state(model.state_dict())
        data = {"train_condition": condition[:48], "train_truth": truth[:48], "train_raw": raw[:48],
                "validation_condition": condition[48:], "validation_truth": truth[48:], "validation_raw": raw[48:]}
        observed_modes = []
        handle = model.register_forward_pre_hook(lambda m, args: observed_modes.append((torch.is_grad_enabled(), m.training)))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = settings(Path(tmp)); Path(cfg["output_dir"]).mkdir()
            result = ablation.run_arm(model, "A_cold", data, cfg)
            self.assertEqual(result["steps_completed"], 4)
            self.assertEqual([r["step"] for r in result["evaluations"]], [0, 1, 2, 4])
            self.assertEqual(result["initial"]["auc"], .5)
            self.assertTrue(result["frozen_backbone_unchanged"])
            self.assertTrue(all(training for grad, training in observed_modes if grad))
            for group in ablation.FEATURE_PREFIXES:
                # Fake PET need not couple visible features, but every configured
                # parameter must be tracked; genuine gradient values are never invented.
                stat = result["final"]["module_updates"][group]
                self.assertGreater(stat["parameter_count"], 0)
                self.assertTrue(torch.isfinite(torch.tensor(stat["single_step_relative_update"])))
            stat = result["final"]["module_updates"]["invisible_projector"]
            self.assertGreater(stat["gradient_l2_after_clip"], 0)
            self.assertGreater(stat["single_step_update_l2"], 0)
            first_final = ablation.clone_state(model.state_dict())
            other = make(); ablation.replay.strict_monitor_load(other, cold)
            # Change ambient RNG: run_arm must still produce exactly the same fit.
            torch.manual_seed(9876)
            second = ablation.run_arm(other, "repeat", data, cfg)
            self.assertEqual(result["final"]["auc"], second["final"]["auc"])
            self.assertEqual(ablation.replay.state_digest(first_final), ablation.replay.state_digest(other.state_dict()))
            self.assertTrue((Path(cfg["output_dir"]) / "A_cold" / "final_monitor.pt").is_file())
            self.assertEqual(json.loads((Path(cfg["output_dir"]) / "A_cold" / "report.json").read_text())["steps_completed"], 4)
        handle.remove()

    def test_update_tracker_reports_missing_gradient_and_true_single_step_delta(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2)) for _ in range(4)])
            def named_parameters(self, *args, **kwargs):
                names = [p.replace("_backbone.", "backbone.", 1) + "weight" for p in ablation.FEATURE_PREFIXES.values()] + ["bank.output.weight"]
                return iter(zip(names, self.p))
        model = M(); tracker = ablation.UpdateTracker(model)
        with torch.no_grad(): model.p[0].add_(1.)
        model.p[0].grad = torch.full_like(model.p[0], 2.)
        stats = tracker.observe(record=True)
        self.assertEqual(stats["grouped_sequential_embedding"]["gradient_present_count"], 2)
        self.assertAlmostEqual(stats["grouped_sequential_embedding"]["single_step_relative_update"], 1.)
        self.assertEqual(stats["invisible_projector"]["gradient_present_count"], 0)
        stats = tracker.observe(record=True)
        self.assertEqual(stats["grouped_sequential_embedding"]["single_step_update_l2"], 0.)


if __name__ == "__main__":
    unittest.main()
