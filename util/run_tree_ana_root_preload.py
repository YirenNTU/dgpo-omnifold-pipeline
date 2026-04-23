#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
TREE_ANA = REPO_ROOT / "bin" / "tree_ana"
LEGACY_IMPORT_PATHS = (
    REPO_ROOT,
    REPO_ROOT / "processor",
    REPO_ROOT / "RegionSelections",
    REPO_ROOT / "quantum",
)


def set_thread_defaults() -> None:
    # ROOT/cppyy can be fragile when imported after large threaded runtimes.
    defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_MAX_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main() -> None:
    set_thread_defaults()
    for path in reversed(LEGACY_IMPORT_PATHS):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        import ROOT  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ROOT before running bin/tree_ana. "
            "This is an environment issue, not an EveNet parquet-content issue."
        ) from exc

    sys.argv = [str(TREE_ANA), *sys.argv[1:]]
    runpy.run_path(str(TREE_ANA), run_name="__main__")


if __name__ == "__main__":
    main()
