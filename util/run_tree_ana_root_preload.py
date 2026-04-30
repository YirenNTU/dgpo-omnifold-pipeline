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


def maybe_load_roounfold() -> None:
    roounfold_lib = os.environ.get("ROOUNFOLD_LIB")
    if not roounfold_lib:
        return

    import ROOT

    load_status = ROOT.gSystem.Load(roounfold_lib)
    if load_status < 0:
        raise RuntimeError(
            f"Failed to load RooUnfold library from ROOUNFOLD_LIB={roounfold_lib}. "
            "Set ROOUNFOLD_LIB to the full path of libRooUnfold.so."
        )


def patch_qi_unfold_observables() -> None:
    """
    inputs:
      None. Reads QI_EXCLUDE_UNFOLD_OBSERVABLES from the environment.
    outputs:
      None. Patches quantum.observables_builder.get_observable_names in this process.
    goal:
      Keep the central QIProcessor untouched while allowing the ml_pipeline
      wrapper to skip non-SDM observables, such as mtautau, in RooUnfold.
    """
    excluded_text = os.environ.get("QI_EXCLUDE_UNFOLD_OBSERVABLES", "mtautau")
    excluded = {item.strip() for item in excluded_text.split(",") if item.strip()}
    if not excluded:
        return

    import quantum.observables_builder as observables_builder

    original_get_observable_names = observables_builder.get_observable_names

    def get_filtered_observable_names():
        return [name for name in original_get_observable_names() if name not in excluded]

    observables_builder.get_observable_names = get_filtered_observable_names
    print(
        "[run-tree-ana-root-preload] excluding unfold observables: "
        f"{sorted(excluded)}",
        flush=True,
    )


def patch_preserve_parquet_weights() -> None:
    """
    inputs:
      None. Reads QI_PRESERVE_PARQUET_WEIGHTS from the environment.
    outputs:
      None. Patches processor.DataLoader.DataLoader.postprocess in this process.
    goal:
      Keep nominal DataLoader code unchanged while allowing the ml_pipeline QI
      wrapper to unfold already-normalized parquet exports with their stored
      event weights, including EveNet test-split corrections.
    """
    enabled = os.environ.get("QI_PRESERVE_PARQUET_WEIGHTS", "1").lower()
    if enabled in {"0", "false", "no", "off"}:
        return

    from processor import DataLoader as DataLoaderModule

    original_postprocess = DataLoaderModule.DataLoader.postprocess

    def postprocess_preserve_parquet_weights(self):
        parquet_inputs = all(str(path).endswith(".parquet") for path in self.input_files)
        has_stored_weights = bool(self.data) and all("weight" in events.fields for events in self.data.values())
        if not parquet_inputs or not has_stored_weights:
            return original_postprocess(self)

        for events in self.data.values():
            events["weight_nominal"] = events["weight"]
        self.current_variation = ("nominal", 0.0)

    DataLoaderModule.DataLoader.postprocess = postprocess_preserve_parquet_weights
    print(
        "[run-tree-ana-root-preload] preserving parquet event weights in DataLoader.postprocess",
        flush=True,
    )


def main() -> None:
    set_thread_defaults()
    os.environ.setdefault("TREE_ANA_DIR", str(REPO_ROOT))
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
    maybe_load_roounfold()
    patch_qi_unfold_observables()
    patch_preserve_parquet_weights()

    sys.argv = [str(TREE_ANA), *sys.argv[1:]]
    runpy.run_path(str(TREE_ANA), run_name="__main__")


if __name__ == "__main__":
    main()
