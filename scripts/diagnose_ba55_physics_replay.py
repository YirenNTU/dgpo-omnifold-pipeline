#!/usr/bin/env python3
"""Fail-closed entry point for the H4 BA55 physics-first artifact replay."""

from __future__ import annotations

import diagnose_ba55_actionability_replay as replay


EXPECTED_SCHEMA = "c4a91e07-h4-ba55-physics-actionability-replay-v1"


def main() -> int:
    observed = getattr(replay, "PHYSICS_REPLAY_SCHEMA", None)
    if observed != EXPECTED_SCHEMA:
        raise RuntimeError(
            "diagnose_ba55_actionability_replay.py is stale: expected physics "
            f"schema {EXPECTED_SCHEMA!r}, found {observed!r}. Sync both replay "
            "scripts before launching."
        )
    return replay.main()


if __name__ == "__main__":
    raise SystemExit(main())
