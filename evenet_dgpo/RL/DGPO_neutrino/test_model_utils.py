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
