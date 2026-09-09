"""Unit tests for ``model_utils`` (config + optional full load if data paths exist).

Run from repo root::

    python RL/DGPO_neutrino/test_model_utils.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

# Repo root must be on ``sys.path`` before ``evenet`` / third-party imports (run as file or ``-m``).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_root_s = str(_REPO_ROOT)
if _root_s not in sys.path:
    sys.path.insert(0, _root_s)

# ``diffusion_sampler`` imports ``debug_tool`` → Lightning/torchvision; stub for lightweight tests.
_dbg = types.ModuleType("evenet.utilities.debug_tool")


def _noop_time_decorator(name=None):
    def _wrapper(func):
        return func

    return _wrapper


_dbg.time_decorator = _noop_time_decorator
sys.modules.setdefault("evenet.utilities.debug_tool", _dbg)

import torch

# Load sibling module by path — ``RL`` is not a package (no ``RL/__init__.py``), so avoid ``from RL...``.
_mu_path = Path(__file__).resolve().parent / "model_utils.py"
_mu_name = "dgpo_neutrino_model_utils"
_spec = importlib.util.spec_from_file_location(_mu_name, _mu_path)
assert _spec is not None and _spec.loader is not None
mu = importlib.util.module_from_spec(_spec)
sys.modules[_mu_name] = mu
_spec.loader.exec_module(mu)

_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class TestResolveCheckpointPath(unittest.TestCase):
    @staticmethod
    def _config(*, resume=None, pretrain=None):
        return types.SimpleNamespace(
            options=types.SimpleNamespace(
                Training=types.SimpleNamespace(
                    model_checkpoint_load_path=resume,
                    pretrain_model_load_path=pretrain,
                )
            )
        )

    def test_configured_missing_resume_fails_instead_of_random_init(self) -> None:
        cfg = self._config(resume="/definitely/missing/dgpo-last.ckpt")
        with self.assertRaisesRegex(FileNotFoundError, "model_checkpoint_load_path"):
            mu.resolve_checkpoint_path(cfg, None)

    def test_existing_configured_resume_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "resume.ckpt"
            checkpoint.touch()
            cfg = self._config(resume=str(checkpoint))
            self.assertEqual(mu.resolve_checkpoint_path(cfg, None), checkpoint.resolve())


class TestCheckpointWeightSource(unittest.TestCase):
    def _load(self, *, replace, dgpo_checkpoint=False, training=True, live=True):
        config = types.SimpleNamespace(options=types.SimpleNamespace(
            Training={"EMA": {"enable": True, "replace_model_after_load": replace}},
        ))
        model = torch.nn.Linear(1, 1, bias=False)
        checkpoint = {"ema_state_dict": {"model.weight": torch.tensor([[9.0]])}}
        if live:
            checkpoint["state_dict"] = {"model.weight": torch.tensor([[2.0]])}
        if dgpo_checkpoint:
            checkpoint["dgpo_checkpoint_version"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pretrain.ckpt"
            torch.save(checkpoint, path)
            mu.load_weights_like_configure_model(
                model, path, torch.device("cpu"), config, for_dgpo_training=training,
            )
        return model.weight.detach().item()

    def test_live_pretrain_selected_for_policy_and_classifier(self):
        for training in (True, False):
            with self.subTest(training=training):
                self.assertEqual(self._load(replace=False, training=training), 2.0)

    def test_dgpo_resume_always_restores_live_state(self):
        self.assertEqual(self._load(replace=True, dgpo_checkpoint=True), 2.0)

    def test_other_configs_can_still_explicitly_select_ema(self):
        self.assertEqual(self._load(replace=True), 9.0)

    def test_missing_live_state_does_not_silently_fall_back_to_ema(self):
        with self.assertRaises(KeyError):
            self._load(replace=False, live=False)


class TestSelectDgpoTrainingState(unittest.TestCase):
    def test_resume_preserves_full_checkpoint_state(self) -> None:
        checkpoint = {"global_step": 100, "dgpo_optimizer_state_dict": {}}
        self.assertIs(
            mu.select_dgpo_training_state(checkpoint, load_mode="resume"),
            checkpoint,
        )

    def test_weights_only_discards_training_state(self) -> None:
        checkpoint = {"global_step": 100, "dgpo_adaptive_omnifold_state": {}}
        self.assertIsNone(
            mu.select_dgpo_training_state(checkpoint, load_mode="weights_only")
        )

    def test_unknown_mode_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_load_mode"):
            mu.select_dgpo_training_state({}, load_mode="freshish")


class TestResolveDgpoAutoResumeCheckpoint(unittest.TestCase):
    def test_disabled_does_not_require_save_directory(self) -> None:
        self.assertIsNone(
            mu.resolve_dgpo_auto_resume_checkpoint(None, enabled=False)
        )

    def test_enabled_requires_save_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_checkpoint_save_path"):
            mu.resolve_dgpo_auto_resume_checkpoint(None, enabled=True)

    def test_fresh_run_has_no_auto_resume_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                mu.resolve_dgpo_auto_resume_checkpoint(tmp, enabled=True)
            )

    def test_saved_last_checkpoint_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "dgpo-epoch=-1-next_ep=0-step=0.ckpt"
            snapshot.write_bytes(b"complete omnifold state")
            (root / "last.ckpt").symlink_to(snapshot.name)
            self.assertEqual(
                mu.resolve_dgpo_auto_resume_checkpoint(root, enabled=True),
                snapshot.resolve(),
            )

    def test_fallback_bootstrap_checkpoint_starts_a_new_output_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new_output = root / "trust05"
            bootstrap = root / "trust01" / "dgpo-epoch=-1-next_ep=0-step=0.ckpt"
            bootstrap.parent.mkdir()
            bootstrap.write_bytes(b"omnifold bootstrap")
            self.assertEqual(
                mu.resolve_dgpo_auto_resume_checkpoint(
                    new_output,
                    enabled=True,
                    fallback_checkpoint_path=bootstrap,
                ),
                bootstrap.resolve(),
            )

    def test_new_branch_last_takes_precedence_over_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            new_output = root / "trust05"
            new_output.mkdir()
            new_last = new_output / "last.ckpt"
            new_last.write_bytes(b"new branch")
            fallback = root / "trust01.ckpt"
            fallback.write_bytes(b"old branch")
            self.assertEqual(
                mu.resolve_dgpo_auto_resume_checkpoint(
                    new_output,
                    enabled=True,
                    fallback_checkpoint_path=fallback,
                ),
                new_last.resolve(),
            )

    def test_missing_explicit_fallback_fails_instead_of_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "fallback checkpoint"):
                mu.resolve_dgpo_auto_resume_checkpoint(
                    Path(tmp) / "new-output",
                    enabled=True,
                    fallback_checkpoint_path=Path(tmp) / "missing.ckpt",
                )


class TestBestRawAucResume(unittest.TestCase):
    @staticmethod
    def _row(epoch, gap, *, saturated=1.0):
        return dict(epoch=float(epoch), global_step=float((epoch + 1) * 10),
                    raw_auc_gap=gap, raw_audit_saturated=saturated)

    def _snapshot(self, root, epoch, history, *, legacy=False, step=None, next_epoch=None):
        step = (epoch + 1) * 10 if step is None else step
        next_epoch = epoch + 1 if next_epoch is None else next_epoch
        path = root / mu.dgpo_snapshot_checkpoint_name(
            last_completed_epoch=epoch, dgpo_next_epoch=next_epoch, global_step=step,
        )
        payload = dict(
            state_dict={"policy": torch.tensor([float(epoch)])},
            epoch=epoch, global_step=step, dgpo_next_epoch=next_epoch,
            dgpo_epoch_step=step % 10 if next_epoch == epoch else 0,
            dgpo_checkpoint_version=1, dgpo_optimizer_state_dict={},
            dgpo_ref_state_dict={}, dgpo_round_ref_state_dict={},
            dgpo_round_ref_sha256="test", dgpo_omnifold_reward_metadata={},
            dgpo_omnifold_reward_stack={},
            dgpo_adaptive_omnifold_state={"probe_history": history},
        )
        torch.save(payload, path, _use_new_zipfile_serialization=not legacy)
        return path

    def test_best_from_history_not_last_or_current_round_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = self._snapshot(root, 0, [self._row(0, 0.12)])
            history = [self._row(0, 0.12), self._row(1, 0.2), self._row(2, 0.18)]
            last = self._snapshot(root, 2, history)
            (root / "last.ckpt").symlink_to(last.name)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = mu.resolve_dgpo_auto_resume_checkpoint(
                root / "new", enabled=True, best_source_checkpoint_dir=root,
            )
            self.assertEqual(result, best.resolve())
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)
            restored = torch.load(result, weights_only=False)
            self.assertEqual(restored["dgpo_next_epoch"], 1)

    def test_invalid_unsaturated_and_tied_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = self._snapshot(root, 1, [self._row(1, 0.15)])
            history = [self._row(1, 0.15), self._row(2, 0.15),
                       self._row(3, 0.01, saturated=0.0),
                       self._row(4, float("nan")), self._row(5, -0.01),
                       self._row(6, 0.7), {}, None]
            last = self._snapshot(root, 6, history)
            (root / "last.ckpt").symlink_to(last.name)
            self.assertEqual(mu.resolve_best_raw_auc_checkpoint(root), best.resolve())

    def test_own_last_has_priority_without_parent_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = root / "last.ckpt"
            last.touch()
            self.assertEqual(mu.resolve_dgpo_auto_resume_checkpoint(
                root, enabled=True, best_source_checkpoint_dir=root / "missing-parent",
            ), last.resolve())

    def test_missing_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(FileNotFoundError, "best-source last"):
                mu.resolve_dgpo_auto_resume_checkpoint(
                    root / "new", enabled=True, best_source_checkpoint_dir=root,
                )

    def test_missing_best_does_not_choose_runner_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = self._snapshot(root, 1, [self._row(0, 0.1), self._row(1, 0.2)])
            (root / "last.ckpt").symlink_to(last.name)
            with self.assertRaisesRegex(FileNotFoundError, "recorded best.*missing"):
                mu.resolve_best_raw_auc_checkpoint(root)

    def test_no_eligible_records_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            last = self._snapshot(root, 0, [self._row(0, 0.1, saturated=0)])
            (root / "last.ckpt").symlink_to(last.name)
            with self.assertRaisesRegex(ValueError, "no completed saturated"):
                mu.resolve_best_raw_auc_checkpoint(root)

    def test_incomplete_best_fails_closed(self):
        for missing in ("dgpo_optimizer_state_dict", "dgpo_round_ref_state_dict",
                        "dgpo_omnifold_reward_stack", "dgpo_omnifold_reward_metadata"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                best = self._snapshot(root, 0, [self._row(0, 0.1)])
                (root / "last.ckpt").symlink_to(best.name)
                payload = torch.load(best, weights_only=False)
                del payload[missing]
                torch.save(payload, best)
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    mu.resolve_best_raw_auc_checkpoint(root)

    def test_epoch_step_and_score_must_match(self):
        for kind in ("epoch", "global_step", "dgpo_next_epoch", "score"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                best = self._snapshot(root, 0, [self._row(0, 0.1)])
                last = self._snapshot(root, 1, [self._row(0, 0.1), self._row(1, 0.2)])
                (root / "last.ckpt").symlink_to(last.name)
                payload = torch.load(best, weights_only=False)
                if kind == "score":
                    payload["dgpo_adaptive_omnifold_state"]["probe_history"][0]["raw_auc_gap"] = 0.3
                else:
                    payload[kind] = 123
                torch.save(payload, best)
                with self.assertRaisesRegex(ValueError, "mismatch|corroborate"):
                    mu.resolve_best_raw_auc_checkpoint(root)

    def test_legacy_non_mmap_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = self._snapshot(root, 0, [self._row(0, 0.1)], legacy=True)
            (root / "last.ckpt").symlink_to(best.name)
            self.assertEqual(mu.resolve_best_raw_auc_checkpoint(root), best.resolve())

    def test_initial_baseline_is_a_recoverable_best_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = self._snapshot(root, -1, [self._row(-1, 0.1)])
            last = self._snapshot(root, 0, [self._row(-1, 0.1), self._row(0, 0.2)])
            (root / "last.ckpt").symlink_to(last.name)
            self.assertEqual(mu.resolve_best_raw_auc_checkpoint(root), best.resolve())

    def test_mid_epoch_best_is_selected_with_correct_resume_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"epoch": 0, "global_step": 5, "checkpoint_next_epoch": 0,
                   "raw_auc_gap": 0.1, "raw_audit_saturated": 1}
            best = self._snapshot(root, 0, [row], step=5, next_epoch=0)
            last = self._snapshot(root, 0, [row, self._row(0, 0.2)])
            (root / "last.ckpt").symlink_to(last.name)
            self.assertEqual(mu.resolve_best_raw_auc_checkpoint(root), best.resolve())
            payload = torch.load(best, weights_only=False)
            self.assertEqual(payload["dgpo_epoch_step"], 5)
            del payload["dgpo_epoch_step"]
            torch.save(payload, best)
            with self.assertRaisesRegex(ValueError, "within-epoch progress"):
                mu.resolve_best_raw_auc_checkpoint(root)

    def test_same_output_or_conflicting_fallback_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "separate output"):
                mu.resolve_dgpo_auto_resume_checkpoint(
                    root, enabled=True, best_source_checkpoint_dir=root,
                )
            with self.assertRaisesRegex(ValueError, "either best-source"):
                mu.resolve_dgpo_auto_resume_checkpoint(
                    root / "new", enabled=True, best_source_checkpoint_dir=root,
                    fallback_checkpoint_path=root / "fallback.ckpt",
                )


class TestDgpoSnapshotCheckpoint(unittest.TestCase):
    def test_snapshot_name_records_epoch_and_step(self) -> None:
        self.assertEqual(
            mu.dgpo_snapshot_checkpoint_name(
                last_completed_epoch=4,
                dgpo_next_epoch=5,
                global_step=50,
            ),
            "dgpo-epoch=4-next_ep=5-step=50.ckpt",
        )

    def test_last_pointer_preserves_legacy_file_and_tracks_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_last = root / "last.ckpt"
            legacy_last.write_bytes(b"legacy")
            first = root / "dgpo-epoch=4-next_ep=5-step=50.ckpt"
            first.write_bytes(b"first")
            last = mu.update_last_checkpoint_pointer(first)
            preserved = root / "dgpo-preserved-last-before-snapshot-mode.ckpt"
            self.assertEqual(preserved.read_bytes(), b"legacy")
            self.assertTrue(last.is_symlink())
            self.assertEqual(last.resolve(), first.resolve())

            second = root / "dgpo-epoch=9-next_ep=10-step=100.ckpt"
            second.write_bytes(b"second")
            mu.update_last_checkpoint_pointer(second)
            self.assertEqual(last.resolve(), second.resolve())
            self.assertEqual(preserved.read_bytes(), b"legacy")


class TestParseDgpoResume(unittest.TestCase):
    def test_lightning_ckpt_resets_schedule(self) -> None:
        ckpt = {"state_dict": {}, "pytorch-lightning_version": "2.0", "global_step": 999}
        self.assertEqual(mu.parse_dgpo_resume_from_checkpoint(ckpt), (0, 0))

    def test_dgpo_v1(self) -> None:
        ckpt = {
            "dgpo_checkpoint_version": 1,
            "pytorch-lightning_version": "2.0",
            "dgpo_next_epoch": 7,
            "global_step": 1400,
        }
        self.assertEqual(mu.parse_dgpo_resume_from_checkpoint(ckpt), (7, 1400))

    def test_legacy_minimal(self) -> None:
        ckpt = {"state_dict": {}, "epoch": 3, "global_step": 99}
        self.assertEqual(mu.parse_dgpo_resume_from_checkpoint(ckpt), (4, 99))

    def test_is_lightning_trainer_checkpoint(self) -> None:
        self.assertTrue(mu.is_lightning_trainer_checkpoint({"optimizer_states": []}))
        self.assertTrue(mu.is_lightning_trainer_checkpoint({"pytorch-lightning_version": "2.1"}))
        self.assertFalse(mu.is_lightning_trainer_checkpoint({"state_dict": {}, "epoch": 5}))
        self.assertFalse(mu.is_lightning_trainer_checkpoint({
            "dgpo_checkpoint_version": 1,
            "pytorch-lightning_version": "2.1",
            "dgpo_next_epoch": 3,
        }))


class TestGenerationUsesEmaShadow(unittest.TestCase):
    def test_ema_disabled(self) -> None:
        self.assertFalse(mu.generation_uses_ema_shadow({"enable": False}))

    def test_explicit_live_policy_generation(self) -> None:
        self.assertFalse(
            mu.generation_uses_ema_shadow(
                {"enable": True, "use_for_generation": False}
            )
        )

    def test_explicit_ema_generation(self) -> None:
        self.assertTrue(
            mu.generation_uses_ema_shadow(
                {"enable": True, "use_for_generation": True}
            )
        )

    def test_live_generation_does_not_allocate_rollout_ema(self) -> None:
        config = types.SimpleNamespace(
            options=types.SimpleNamespace(
                Training={
                    "EMA": {"enable": True, "use_for_generation": False}
                }
            )
        )
        self.assertIsNone(mu.make_ema_rollout(torch.nn.Linear(2, 2), config))

    def test_ema_generation_allocates_rollout_ema(self) -> None:
        config = types.SimpleNamespace(
            options=types.SimpleNamespace(
                Training={
                    "EMA": {"enable": True, "use_for_generation": True}
                }
            )
        )
        self.assertIsNotNone(mu.make_ema_rollout(torch.nn.Linear(2, 2), config))


class _DummyNeutrinoPolicy(torch.nn.Module):
    def __init__(self, *, dropout: float) -> None:
        super().__init__()
        self.GroupedSequentialEmbedding = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(dropout)
        )
        self.GlobalEmbedding = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(dropout)
        )
        self.PET = torch.nn.ModuleDict(
            {
                "attention": torch.nn.MultiheadAttention(
                    2, 1, dropout=dropout, batch_first=True
                )
            }
        )
        self.TruthGeneration = torch.nn.Sequential(
            torch.nn.Linear(2, 2), torch.nn.Dropout(dropout)
        )


class TestDeterministicDgpoPolicyGuard(unittest.TestCase):
    def test_accepts_zero_dropout_policy(self) -> None:
        mu.assert_dgpo_neutrino_policy_deterministic(
            _DummyNeutrinoPolicy(dropout=0.0)
        )

    def test_rejects_train_mode_policy_dropout(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stochastic layers remain"):
            mu.assert_dgpo_neutrino_policy_deterministic(
                _DummyNeutrinoPolicy(dropout=0.1)
            )


class TestFamoStateDictInjection(unittest.TestCase):
    def test_injects_missing_keys(self) -> None:
        sd: dict[str, object] = {"model.foo": torch.zeros(1)}
        n = mu.inject_default_famo_state_dict_keys(sd)
        self.assertEqual(n, 5)
        for task in mu.FAMO_STATE_DICT_TASKS:
            self.assertIn(f"model.famo.w.{task}", sd)

    def test_idempotent_when_present(self) -> None:
        sd = {f"model.famo.w.{task}": torch.tensor([0.0]) for task in mu.FAMO_STATE_DICT_TASKS}
        self.assertEqual(mu.inject_default_famo_state_dict_keys(sd), 0)


class TestCheckpointRewardMetadata(unittest.TestCase):
    def test_omnifold_identity_is_saved(self) -> None:
        model = torch.nn.Linear(2, 2)
        config = types.SimpleNamespace(
            options=types.SimpleNamespace(Training={"EMA": {"enable": False}})
        )
        metadata = {"schema_version": 1, "sources": [{"name": "omnifold"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.ckpt"
            mu.save_lightning_compatible_checkpoint(
                path,
                model,
                None,
                config,
                last_completed_epoch=0,
                dgpo_next_epoch=1,
                global_step=3,
                dgpo_omnifold_reward_metadata=metadata,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["dgpo_omnifold_reward_metadata"], metadata)

    def test_adaptive_reward_reference_pair_is_saved_together(self) -> None:
        model = torch.nn.Linear(2, 2)
        round_ref = torch.nn.Linear(2, 2)
        config = types.SimpleNamespace(
            options=types.SimpleNamespace(Training={"EMA": {"enable": False}})
        )
        state = {"reward_round_id": 3, "trigger_threshold": 0.08}
        stack = {"schema_version": 1, "reward_round_id": 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adaptive.ckpt"
            mu.save_lightning_compatible_checkpoint(
                path,
                model,
                None,
                config,
                last_completed_epoch=2,
                dgpo_next_epoch=3,
                global_step=11,
                round_ref_model=round_ref,
                reward_round_id=3,
                dgpo_adaptive_omnifold_state=state,
                dgpo_omnifold_reward_stack=stack,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["dgpo_reward_round_id"], 3)
        self.assertEqual(payload["dgpo_adaptive_omnifold_state"], state)
        self.assertEqual(payload["dgpo_omnifold_reward_stack"], stack)
        self.assertEqual(
            payload["dgpo_round_ref_sha256"],
            mu.state_dict_sha256(round_ref),
        )

    def test_state_dict_digest_changes_with_weights(self) -> None:
        model = torch.nn.Linear(2, 2)
        before = mu.state_dict_sha256(model)
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertNotEqual(before, mu.state_dict_sha256(model))


class TestLoadTrainingConfig(unittest.TestCase):
    def test_dgpo_k_from_yaml(self) -> None:
        if not _CONFIG.is_file():
            self.skipTest(f"config not found: {_CONFIG}")
        try:
            cfg = mu.load_training_config(_CONFIG)
        except OSError as exc:
            self.skipTest(f"config load failed (check options.default paths): {exc}")
        self.assertEqual(int(cfg.dgpo.K), 8)


class TestModelLoadOptional(unittest.TestCase):
    """Integration checks; skipped when config, defaults, or normalization paths are unavailable."""

    norm_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        if not _CONFIG.is_file():
            raise unittest.SkipTest(f"config not found: {_CONFIG}")
        try:
            cls.cfg = mu.load_training_config(_CONFIG)
        except OSError as exc:
            raise unittest.SkipTest(f"config load failed: {exc}") from exc
        cls.norm_path = Path(str(cls.cfg.options.Dataset.normalization_file)).expanduser()
        if not cls.norm_path.is_file():
            raise unittest.SkipTest(f"normalization_file not on disk: {cls.norm_path}")

    def test_reference_is_frozen_and_matches_state_dict(self) -> None:
        device = torch.device("cpu")
        bundle = mu.load_evenet_model_for_dgpo(_CONFIG, device, checkpoint_path=None)
        ref = mu.make_reference_model(
            bundle.model,
            bundle.config,
            bundle.normalization_dict,
            device,
        )
        self.assertEqual(ref.state_dict().keys(), bundle.model.state_dict().keys())
        self.assertEqual(mu.count_trainable_params(ref), 0)


if __name__ == "__main__":
    unittest.main()
