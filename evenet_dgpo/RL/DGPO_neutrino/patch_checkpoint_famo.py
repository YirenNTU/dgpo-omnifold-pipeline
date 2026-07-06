"""One-time script to inject default FAMO keys into existing DGPO checkpoints.

EveNetEngine.configure_model always registers a FAMO submodule on EveNetModel
(even when turn_on=false), producing ``model.famo.w.<task>`` keys in the engine
state_dict. Checkpoints saved before automatic injection are missing those keys,
which causes Lightning strict=True load to fail.

New DGPO runs inject FAMO keys automatically in
:func:`RL.DGPO_neutrino.model_utils.build_lightning_compatible_checkpoint`.

Usage (run on NERSC):
    python RL/DGPO_neutrino/patch_checkpoint_famo.py /path/to/last.ckpt
    python RL/DGPO_neutrino/patch_checkpoint_famo.py /path/to/checkpoints_dir/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from RL.DGPO_neutrino.model_utils import inject_default_famo_state_dict_keys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)


def patch_checkpoint(path: Path) -> bool:
    """Add default famo keys to a single checkpoint. Returns True if patched."""
    _log.info("Loading %s ...", path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        _log.warning("  No 'state_dict' key — skipping.")
        return False

    n_injected = inject_default_famo_state_dict_keys(ckpt["state_dict"])
    if n_injected <= 0:
        _log.info("  Already has famo keys — skipping.")
        return False

    torch.save(ckpt, str(path))
    _log.info("  Patched: injected %s famo key(s).", n_injected)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch DGPO checkpoints with default FAMO keys.")
    parser.add_argument("path", help="Path to a .ckpt file or a directory of .ckpt files.")
    args = parser.parse_args()

    target = Path(args.path).expanduser().resolve()
    if not target.exists():
        _log.error("Path does not exist: %s", target)
        sys.exit(1)

    if target.is_file():
        ckpt_files = [target]
    else:
        ckpt_files = sorted(target.glob("*.ckpt"))
        if not ckpt_files:
            _log.error("No .ckpt files found in %s", target)
            sys.exit(1)

    patched = 0
    for f in ckpt_files:
        if patch_checkpoint(f):
            patched += 1

    _log.info("Done. Patched %d / %d checkpoint(s).", patched, len(ckpt_files))


if __name__ == "__main__":
    main()
