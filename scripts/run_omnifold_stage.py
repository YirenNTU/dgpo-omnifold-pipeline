#!/usr/bin/env python3
"""Launch the standalone K=1 Ztautau OmniFold stage."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENET_DGPO_ROOT = REPO_ROOT / "evenet_dgpo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=REPO_ROOT / "config" / "train_diffusion_nersc.yaml",
    )
    parser.add_argument(
        "--omnifold-config",
        type=Path,
        default=REPO_ROOT / "config" / "omnifold_ztautau.yaml",
    )
    parser.add_argument("--stage", choices=("all", "pool", "fit", "check"), default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild-pools", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        (str(EVENET_DGPO_ROOT), env.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    command = [
        sys.executable,
        "-m",
        "RL.DGPO_neutrino.omnifold_ztautau.stage",
        "--train-config",
        str(args.train_config.expanduser().resolve()),
        "--config",
        str(args.omnifold_config.expanduser().resolve()),
        "--stage",
        args.stage,
    ]
    if args.device:
        command.extend(("--device", args.device))
    if args.rebuild_pools:
        command.append("--rebuild-pools")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
