"""Unit tests for EMA configuration helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


_EMA_PATH = Path(__file__).with_name("ema.py")
_SPEC = importlib.util.spec_from_file_location("evenet_ema_under_test", _EMA_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EMA_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EMA_MODULE)

resolve_ema_update_every_n_steps = _EMA_MODULE.resolve_ema_update_every_n_steps


class TestResolveEmaUpdateEveryNSteps(unittest.TestCase):
    def test_canonical_key(self) -> None:
        self.assertEqual(
            resolve_ema_update_every_n_steps({"update_every_n_steps": 5}),
            5,
        )

    def test_canonical_key_takes_precedence(self) -> None:
        self.assertEqual(
            resolve_ema_update_every_n_steps(
                {"update_every_n_steps": 7, "update_step": 3}
            ),
            7,
        )

    def test_legacy_key_fallback(self) -> None:
        self.assertEqual(resolve_ema_update_every_n_steps({"update_step": 4}), 4)

    def test_default_interval(self) -> None:
        self.assertEqual(resolve_ema_update_every_n_steps({}), 1)

    def test_invalid_interval_is_rejected(self) -> None:
        for value in (0, -1, 1.5, True, "not-an-integer"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_ema_update_every_n_steps(
                        {"update_every_n_steps": value}
                    )


if __name__ == "__main__":
    unittest.main()
