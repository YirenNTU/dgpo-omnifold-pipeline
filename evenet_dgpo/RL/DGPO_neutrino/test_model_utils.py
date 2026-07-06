"""Unit tests for ``model_utils`` (config + optional full load if data paths exist).

Run from repo root::

    python RL/DGPO_neutrino/test_model_utils.py
"""

from __future__ import annotations

import importlib.util
import sys
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
