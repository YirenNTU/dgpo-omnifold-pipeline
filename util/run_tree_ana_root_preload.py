#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
TREE_ANA = REPO_ROOT / "bin" / "tree_ana"
LEGACY_IMPORT_PATHS = (
    REPO_ROOT,
    REPO_ROOT / "processor",
    REPO_ROOT / "RegionSelections",
    REPO_ROOT / "quantum",
)


def create_parser() -> argparse.ArgumentParser:
    """
    inputs:
      None.
    outputs:
      argparse.ArgumentParser for wrapper-only options plus tree_ana passthrough basics.
    goal:
      Add ml_pipeline conveniences, such as multi-process region splitting,
      without changing bin/tree_ana arguments.
    """
    parser = argparse.ArgumentParser(description="Run tree_ana with ml_pipeline ROOT/QI patches.")
    parser.add_argument("--config-yaml", "-c", default=None, help="YAML file for tree_ana configuration.")
    parser.add_argument("--log-level", "-l", default=None, help="tree_ana log level.")
    parser.add_argument("--output-dir", "-o", default=None, help="tree_ana output directory.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of region-parallel QI worker processes.")
    return parser


def build_tree_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    """
    inputs:
      args: argparse.Namespace, parsed wrapper arguments.
      passthrough: list[str], unknown arguments that should be passed to tree_ana.
    outputs:
      list[str], command-line arguments accepted by bin/tree_ana.
    goal:
      Strip wrapper-only options while preserving normal tree_ana invocation semantics.
    """
    tree_args: list[str] = []
    if args.config_yaml is not None:
        tree_args.extend(["-c", args.config_yaml])
    if args.log_level is not None:
        tree_args.extend(["-l", args.log_level])
    if args.output_dir is not None:
        tree_args.extend(["-o", args.output_dir])
    tree_args.extend(passthrough)
    return tree_args


def read_config(path: str) -> dict:
    """
    inputs:
      path: str, YAML config path.
    outputs:
      dict, parsed tree_ana configuration.
    goal:
      Keep config loading local to the wrapper so parallel mode can write
      temporary per-region configs.
    """
    import yaml

    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def write_config(config: dict, path: Path) -> None:
    """
    inputs:
      config: dict, tree_ana configuration to serialize.
      path: pathlib.Path, destination YAML path.
    outputs:
      None.
    goal:
      Materialize one temporary config per worker chunk.
    """
    import yaml

    with open(path, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def chunk_regions(regions: list[str], num_workers: int) -> list[list[str]]:
    """
    inputs:
      regions: list[str], QI regions from dict_region_to_signals.
      num_workers: int, requested worker count.
    outputs:
      list[list[str]], non-empty balanced region chunks.
    goal:
      Split QI unfolding by region because each region writes independent output
      directories and response matrices.
    """
    worker_count = max(1, min(num_workers, len(regions)))
    chunks = [[] for _ in range(worker_count)]
    for index, region in enumerate(regions):
        chunks[index % worker_count].append(region)
    return [chunk for chunk in chunks if chunk]


def build_chunk_config(config: dict, regions: list[str]) -> dict:
    """
    inputs:
      config: dict, full tree_ana configuration.
      regions: list[str], regions assigned to one worker.
    outputs:
      dict, reduced config containing only the assigned QI regions.
    goal:
      Avoid loading and unfolding unused regions in each worker while keeping
      signal-category splitting correct for backgrounds.
    """
    chunk_config = copy.deepcopy(config)
    processors = chunk_config["Processors"]
    qi_config = processors["QIProcessor"]
    full_region_to_signals = qi_config["dict_region_to_signals"]
    qi_config["dict_region_to_signals"] = {region: full_region_to_signals[region] for region in regions}

    global_configs = chunk_config.setdefault("GlobalConfigs", {})
    global_configs["load_regions"] = ["raw", *regions]

    used_signals = {
        signal
        for region in regions
        for signal in full_region_to_signals[region]
    }
    signal_categories = global_configs.get("signal_categories")
    if isinstance(signal_categories, dict):
        global_configs["signal_categories"] = {
            name: value for name, value in signal_categories.items() if name in used_signals
        }
    return chunk_config


def output_root_from_args(args: argparse.Namespace, config: dict) -> Path:
    """
    inputs:
      args: argparse.Namespace, parsed wrapper arguments.
      config: dict, full tree_ana configuration.
    outputs:
      pathlib.Path, tree_ana output root.
    goal:
      Match bin/tree_ana output-dir resolution so merged parallel outputs land
      where the single-process run would have written them.
    """
    if args.output_dir is not None:
        return Path(args.output_dir)
    return Path(config.get("GlobalConfigs", {}).get("default_output_dir", "./output/"))


def merge_qi_outputs(chunk_roots: list[Path], final_root: Path) -> None:
    """
    inputs:
      chunk_roots: list[pathlib.Path], per-worker tree_ana output roots.
      final_root: pathlib.Path, final tree_ana output root.
    outputs:
      None.
    goal:
      Reconstruct the usual QI_analysis directory after region-parallel workers
      finish in isolated output roots.
    """
    final_qi_dir = final_root / "QI_analysis"
    final_qi_dir.mkdir(parents=True, exist_ok=True)
    result_text: list[str] = []

    for chunk_root in chunk_roots:
        chunk_qi_dir = chunk_root / "QI_analysis"
        result_path = chunk_qi_dir / "results.txt"
        if result_path.exists():
            result_text.append(result_path.read_text())
        for child in (chunk_qi_dir.iterdir() if chunk_qi_dir.exists() else []):
            if child.name == "results.txt":
                continue
            destination = final_qi_dir / child.name
            if child.is_dir():
                shutil.copytree(child, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(child, destination)

    if result_text:
        (final_qi_dir / "results.txt").write_text("\n".join(result_text))


def run_parallel(args: argparse.Namespace, passthrough: list[str]) -> None:
    """
    inputs:
      args: argparse.Namespace, parsed wrapper arguments.
      passthrough: list[str], extra arguments for bin/tree_ana.
    outputs:
      None.
    goal:
      Run QI unfolding in multiple independent ROOT processes by splitting
      dict_region_to_signals across temporary configs.
    """
    if args.config_yaml is None:
        raise ValueError("--num-workers requires an explicit -c/--config-yaml path.")

    config = read_config(args.config_yaml)
    region_to_signals = config.get("Processors", {}).get("QIProcessor", {}).get("dict_region_to_signals", {})
    regions = list(region_to_signals)
    if not regions:
        raise ValueError("Cannot run parallel QI: config has no Processors.QIProcessor.dict_region_to_signals.")

    chunks = chunk_regions(regions, args.num_workers)
    final_root = output_root_from_args(args, config)
    work_root = final_root / "_qi_parallel_work"
    work_root.mkdir(parents=True, exist_ok=True)
    print(
        "[run-tree-ana-root-preload] running region-parallel QI "
        f"workers={len(chunks)} regions={len(regions)}",
        flush=True,
    )

    processes: list[tuple[list[str], subprocess.Popen]] = []
    chunk_roots: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="qi_parallel_", dir=str(work_root)) as tmp_dir:
        tmp_path = Path(tmp_dir)
        for index, chunk in enumerate(chunks):
            chunk_config = build_chunk_config(config, chunk)
            chunk_config_path = tmp_path / f"chunk_{index:03d}.yaml"
            write_config(chunk_config, chunk_config_path)

            chunk_root = tmp_path / f"output_{index:03d}"
            chunk_roots.append(chunk_root)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "-c",
                str(chunk_config_path),
                "-o",
                str(chunk_root),
                *([] if args.log_level is None else ["-l", args.log_level]),
                *passthrough,
            ]
            env = os.environ.copy()
            env["QI_PARALLEL_CHILD"] = "1"
            print(
                "[run-tree-ana-root-preload] start worker "
                f"{index + 1}/{len(chunks)} regions={chunk}",
                flush=True,
            )
            processes.append((chunk, subprocess.Popen(command, env=env)))

        failures: list[tuple[list[str], int]] = []
        for chunk, process in processes:
            return_code = process.wait()
            if return_code != 0:
                failures.append((chunk, return_code))

        if failures:
            details = ", ".join(f"regions={chunk} exit={code}" for chunk, code in failures)
            raise RuntimeError(f"One or more QI parallel workers failed: {details}")

        merge_qi_outputs(chunk_roots, final_root)
        print(
            "[run-tree-ana-root-preload] merged parallel QI outputs into "
            f"{final_root / 'QI_analysis'}",
            flush=True,
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


def patch_parquet_column_filter() -> None:
    """
    Patch DataLoader.load_events_from_parquet to skip nested struct columns
    (p4 fields) that QI unfolding never reads.  This dramatically reduces
    peak RSS when the pretrain parquet is large.
    Disable by setting QI_PARQUET_COLUMN_FILTER=0 in the environment.
    """
    if os.environ.get("QI_PARQUET_COLUMN_FILTER", "1").lower() in {"0", "false", "no", "off"}:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from processor import DataLoader as DataLoaderModule
    except Exception:
        return

    original_load = DataLoaderModule.load_events_from_parquet

    def _load_flat_columns_only(file_path, columns=None):
        if columns is None:
            try:
                schema = pq.read_schema(file_path)
                columns = [
                    field.name for field in schema
                    if not pa.types.is_struct(field.type)
                ]
            except Exception:
                pass
        return original_load(file_path, columns=columns)

    DataLoaderModule.load_events_from_parquet = _load_flat_columns_only
    print(
        "[run-tree-ana-root-preload] parquet column filter active: "
        "nested struct (p4) columns will not be loaded",
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
    parser = create_parser()
    args, passthrough = parser.parse_known_args()
    if args.num_workers > 1 and os.environ.get("QI_PARALLEL_CHILD") != "1":
        set_thread_defaults()
        run_parallel(args, passthrough)
        return

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
    patch_parquet_column_filter()
    patch_preserve_parquet_weights()

    sys.argv = [str(TREE_ANA), *build_tree_args(args, passthrough)]
    runpy.run_path(str(TREE_ANA), run_name="__main__")


if __name__ == "__main__":
    main()
