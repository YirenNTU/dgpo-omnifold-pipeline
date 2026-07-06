#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for scripts/train_neutrino_backend.py. "
        "Install the EveNet / DGPO Python requirements first."
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENET_DGPO_ROOT = REPO_ROOT / "evenet_dgpo"
DEFAULT_OVERLAY = REPO_ROOT / "config" / "dgpo_ztautau_overlay.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping YAML at {path}, got {type(data)!r}.")
    return data


def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_runtime_config(
    *,
    base_config: Path,
    overlay_config: Path | None,
    backend: str,
) -> Path:
    merged = read_yaml(base_config)
    if overlay_config is not None:
        merged = deep_update(merged, read_yaml(overlay_config))
    merged.setdefault("compat", {})
    merged["compat"]["backend"] = backend
    merged["compat"]["repo_root"] = str(REPO_ROOT)
    merged.setdefault("rl", {})
    merged["rl"]["enabled"] = backend == "dgpo-evenet"
    runtime_dir = Path(tempfile.mkdtemp(prefix="ztautau_dgpo_runtime_"))
    runtime_path = runtime_dir / f"{backend}_runtime.yaml"
    with runtime_path.open("w") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False)
    return runtime_path


def command_for_backend(backend: str, runtime_config: Path) -> list[str]:
    python_exe = sys.executable
    if backend == "pure-evenet":
        return [python_exe, str(EVENET_DGPO_ROOT / "evenet" / "train.py"), str(runtime_config)]
    if backend == "dgpo-evenet":
        return [python_exe, str(EVENET_DGPO_ROOT / "RL" / "DGPO_neutrino" / "dgpo_trainer.py"), str(runtime_config)]
    raise ValueError(f"Unsupported backend={backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch pure EveNet or DGPO-EveNet neutrino training with a shared runtime config."
    )
    parser.add_argument("--backend", choices=("pure-evenet", "dgpo-evenet"), required=True)
    parser.add_argument("--base-config", type=Path, required=True, help="Base training YAML from the current EveNet workflow.")
    parser.add_argument(
        "--overlay-config",
        type=Path,
        default=DEFAULT_OVERLAY,
        help="Optional overlay YAML. Defaults to the Ztautau DGPO compatibility overlay.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not apply any overlay; run the base config exactly as provided.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the underlying train entrypoint.",
    )
    args = parser.parse_args()

    runtime_config = build_runtime_config(
        base_config=args.base_config.resolve(),
        overlay_config=None if args.no_overlay else args.overlay_config.resolve(),
        backend=args.backend,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(EVENET_DGPO_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    command = command_for_backend(args.backend, runtime_config)
    if args.extra_args:
        passthrough = list(args.extra_args)
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]
        command.extend(passthrough)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
