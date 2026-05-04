#!/usr/bin/env python3
"""Convert sharded EveNet NPZ inputs to bounded-memory parquet shards."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib import format as npy_format


REPO_DIR = Path(__file__).resolve().parents[2]
ML_PIPELINE_DIR = Path(__file__).resolve().parents[1]
EVENET_DIR = ML_PIPELINE_DIR / "EveNet-Full"
sys.path.insert(0, str(EVENET_DIR))

from evenet.control.global_config import global_config  # noqa: E402
from preprocessing.helper import (  # noqa: E402
    PostProcessor,
    build_log_scale_plan,
    event_split_indices,
    load_npz,
    process_dict,
    slice_event_dict,
)


LOGGER = logging.getLogger("preprocess_evenet_shards")
DEFAULT_MISSING_FLOAT = -99.0
CORE_MODEL_KEYS = {
    "x",
    "x_mask",
    "conditions",
    "conditions_mask",
    "x_invisible",
    "x_invisible_mask",
    "event_weight",
}


def generate_assignment_names(event_info):
    """
    inputs:
      event_info: EveNet EventInfo object, loaded from preprocess_config.yaml.
    outputs:
      assignment_names: list[str], flattened assignment labels.
      assignment_map: list[tuple], original product-particle mapping.
    goal:
      Mirror EveNet-Full preprocessing label construction without importing the
      CLI entrypoint.
    """
    assignment_names = []
    assignment_map = []

    for process_name, children in event_info.product_particles.items():
        for particle_name, daughter_particles in children.items():
            assignment_names.append(f"TARGETS/{process_name}/{particle_name}")
            assignment_map.append((process_name, particle_name, daughter_particles))

    return assignment_names, assignment_map


def parse_split_ratio(value: str) -> tuple[float, float, float]:
    """
    inputs:
      value: str, comma-separated train,val,test fractions.
    outputs:
      split_ratio: tuple[float, float, float], validated to sum to one.
    goal:
      Keep the CLI compatible with EveNet-Full/preprocessing/preprocess.py.
    """
    try:
        split_ratio = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("Use --split-ratio train,val,test, e.g. 0.4,0.1,0.5") from exc

    if len(split_ratio) != 3:
        raise ValueError("--split-ratio must have exactly three values: train,val,test")
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"--split-ratio must sum to one, got {split_ratio}")

    return split_ratio


def load_shards(manifest_path: Path, key: str) -> list[Path]:
    """
    inputs:
      manifest_path: Path, evenet_input_shards_manifest.json from step 1.
      key: str, manifest list key such as training_shards or data_shards.
    outputs:
      shards: list[Path], absolute paths to NPZ shards.
    goal:
      Resolve relative manifest entries while keeping the sharded step-1 output
      as the single source of truth.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "evenet_input_shards_v1":
        raise ValueError(f"Unsupported shard manifest format: {manifest.get('format')}")

    base_dir = manifest_path.parent
    shards = [base_dir / item for item in manifest.get(key, [])]
    missing = [str(path) for path in shards if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {key} files: {missing[:5]}")

    return shards


def read_npz_header_schema(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """
    inputs:
      path: Path, NPZ shard.
    outputs:
      schema: dict[str, (shape, dtype)], parsed from embedded NPY headers.
    goal:
      Discover the global shard schema without materializing large arrays.
    """
    schema: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            key = name.removesuffix(".npy")
            with archive.open(name) as stream:
                version = npy_format.read_magic(stream)
                shape, _, dtype = npy_format._read_array_header(stream, version)
            schema[key] = (tuple(shape), np.dtype(dtype))
    return schema


def build_reference_schema(shards: list[Path]) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """
    inputs:
      shards: list[Path], training NPZ shards.
    outputs:
      reference: dict[str, (tail_shape, dtype)], union schema excluding event axis.
    goal:
      Make independently built NPZ shards look like one globally consistent
      dataset before converting them to parquet.
    """
    reference: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    conflicts: list[str] = []
    for shard in shards:
        for key, (shape, dtype) in read_npz_header_schema(shard).items():
            tail_shape = tuple(shape[1:])
            if key not in reference:
                reference[key] = (tail_shape, dtype)
                continue
            ref_shape, ref_dtype = reference[key]
            if ref_shape != tail_shape or ref_dtype != dtype:
                conflicts.append(
                    f"{key}: reference(shape={ref_shape}, dtype={ref_dtype}) "
                    f"vs {shard.name}(shape={tail_shape}, dtype={dtype})"
                )

    if conflicts:
        raise ValueError("Inconsistent NPZ shard schema:\n  " + "\n  ".join(conflicts[:20]))
    return reference


def default_missing_array(key: str, tail_shape: tuple[int, ...], dtype: np.dtype, num_events: int) -> np.ndarray:
    """
    inputs:
      key: str, missing array name.
      tail_shape: tuple[int, ...], shape after the event axis.
      dtype: np.dtype, target dtype.
      num_events: int, number of events in this shard.
    outputs:
      values: np.ndarray, synthetic defaults matching the global schema.
    goal:
      Fill optional passthrough fields that are absent in one shard but present
      in another, matching the defaults used by the EveNet input builder.
    """
    shape = (num_events,) + tuple(tail_shape)
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(shape, dtype=bool)
    if np.issubdtype(dtype, np.integer):
        if key == "initial_total_num_events":
            return np.full(shape, num_events, dtype=dtype)
        if key.startswith("truth_num_"):
            return np.zeros(shape, dtype=dtype)
        return np.full(shape, -1, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        if key.startswith("analyzing_power"):
            return np.zeros(shape, dtype=dtype)
        return np.full(shape, DEFAULT_MISSING_FLOAT, dtype=dtype)
    raise ValueError(f"Cannot synthesize missing array for {key}: dtype={dtype}")


def align_to_reference_schema(
    data: dict[str, np.ndarray],
    reference_schema: dict[str, tuple[tuple[int, ...], np.dtype]],
    shard_path: Path,
) -> dict[str, np.ndarray]:
    """
    inputs:
      data: dict[str, np.ndarray], loaded NPZ shard.
      reference_schema: dict[str, (tail_shape, dtype)], global training schema.
      shard_path: Path, current shard path used for diagnostics.
    outputs:
      data: dict[str, np.ndarray], with missing optional arrays filled.
    goal:
      Prevent parquet/shape_metadata drift caused by independently built NPZ
      shards having different passthrough columns.
    """
    num_events = len(data["x"]) if "x" in data else len(next(iter(data.values())))
    missing_required = sorted(CORE_MODEL_KEYS - set(data))
    if missing_required:
        raise ValueError(f"{shard_path} is missing core EveNet keys: {missing_required}")

    for key, (tail_shape, dtype) in reference_schema.items():
        if key in data:
            actual_tail = tuple(data[key].shape[1:])
            if actual_tail != tail_shape:
                raise ValueError(
                    f"{shard_path} has inconsistent shape for {key}: "
                    f"{actual_tail} vs reference {tail_shape}"
                )
            continue
        data[key] = default_missing_array(key, tail_shape, dtype, num_events)
    return data


def split_output_path(store_dir: Path, split_name: str, shard_index: int) -> Path:
    """
    inputs:
      store_dir: Path, base EveNet parquet directory.
      split_name: str, one of train/val/test.
      shard_index: int, source NPZ shard index.
    outputs:
      path: Path, destination parquet path.
    goal:
      Keep train shards directly under store_dir for the existing train config,
      while validation and test shards live in explicit subdirectories.
    """
    if split_name == "train":
        return store_dir / f"train_{shard_index:06d}.parquet"
    return store_dir / split_name / f"{split_name}_{shard_index:06d}.parquet"


def write_chunk_table(chunks: list[pa.Table], out_path: Path, *, shuffle_seed: int | None) -> dict:
    """
    inputs:
      chunks: list[pa.Table], processed chunks for one shard/split.
      out_path: Path, destination parquet.
      shuffle_seed: int | None, optional row shuffle seed.
    outputs:
      summary: dict, row count and output path.
    goal:
      Materialize only one shard/split table at a time to avoid the old
      all-shards-in-memory concatenation.
    """
    if not chunks:
        return {"path": str(out_path), "rows": 0, "written": False}

    table = pa.concat_tables(chunks)
    if shuffle_seed is not None and table.num_rows > 1:
        order = np.random.default_rng(shuffle_seed).permutation(table.num_rows)
        table = table.take(pa.array(order))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    summary = {
        "path": str(out_path),
        "rows": table.num_rows,
        "size_mb": table.nbytes / 1024 / 1024,
        "written": True,
    }
    del table
    return summary


def preprocess_shards(
    *,
    shards: list[Path],
    data_shards: list[Path],
    store_dir: Path,
    split_ratio: tuple[float, float, float],
    unique_process_ids,
    assignment_keys,
    verbose: bool,
) -> dict:
    """
    inputs:
      shards: list[Path], EveNet NPZ shards from build_evenet_input_from_parquet.
      data_shards: list[Path], optional data-only NPZ shards for inference.
      store_dir: Path, output directory for train parquet shards and metadata.
      split_ratio: tuple[float, float, float], event-level train/val/test split.
      unique_process_ids: list[str], class labels from EveNet event info.
      assignment_keys: list[str], assignment labels for preprocessing statistics.
      verbose: bool, print progress lines.
    outputs:
      summary: dict, written parquet paths and row counts.
    goal:
      Run EveNet preprocessing in bounded memory by loading one NPZ shard,
      writing its parquet shards immediately, and then releasing memory.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "val").mkdir(parents=True, exist_ok=True)
    (store_dir / "test").mkdir(parents=True, exist_ok=True)
    (store_dir / "data").mkdir(parents=True, exist_ok=True)

    shape_metadata = None
    train_stats = PostProcessor(global_config)
    log_scale_plan = build_log_scale_plan(global_config)
    reference_schema = build_reference_schema(shards)
    rng = np.random.default_rng(42)
    summary = {
        "store_dir": str(store_dir),
        "split_ratio": list(split_ratio),
        "train": [],
        "val": [],
        "test": [],
        "data": [],
    }

    LOGGER.info("Applying np.log1p to log-scale features:%s", log_scale_plan.description())

    for shard_index, shard_path in enumerate(shards):
        if verbose:
            print(f"[preprocess-shards] loading {shard_index + 1}/{len(shards)} {shard_path}")

        data = align_to_reference_schema(load_npz(shard_path), reference_schema, shard_path)
        n_events = len(data["x"])
        split_indices = dict(zip(("train", "val", "test"), event_split_indices(n_events, split_ratio, rng)))

        for split_name, indices in split_indices.items():
            if len(indices) == 0:
                continue

            pdict = slice_event_dict(data, indices, n_events)
            chunks: list[pa.Table] = []
            shape_metadata = process_dict(
                pdict,
                global_config=global_config,
                unique_process_ids=unique_process_ids,
                assignment_keys=assignment_keys,
                log_scale_plan=log_scale_plan,
                statistics=train_stats if split_name == "train" else None,
                shape_metadata=shape_metadata,
                store_chunks=chunks,
            )

            out_path = split_output_path(store_dir, split_name, shard_index)
            written = write_chunk_table(chunks, out_path, shuffle_seed=31 + shard_index)
            summary[split_name].append(written)
            if verbose and written["written"]:
                print(
                    "[preprocess-shards] wrote "
                    f"{split_name} rows={written['rows']} size={written['size_mb']:.2f} MB -> {out_path}"
                )

            del pdict, chunks
            gc.collect()

        del data, split_indices
        gc.collect()

    for shard_index, shard_path in enumerate(data_shards):
        if verbose:
            print(f"[preprocess-shards] loading data {shard_index + 1}/{len(data_shards)} {shard_path}")

        data = align_to_reference_schema(load_npz(shard_path), reference_schema, shard_path)
        chunks: list[pa.Table] = []
        shape_metadata = process_dict(
            data,
            global_config=global_config,
            unique_process_ids=unique_process_ids,
            assignment_keys=assignment_keys,
            log_scale_plan=log_scale_plan,
            statistics=None,
            shape_metadata=shape_metadata,
            store_chunks=chunks,
        )
        out_path = store_dir / "data" / f"data_{shard_index:06d}.parquet"
        written = write_chunk_table(chunks, out_path, shuffle_seed=None)
        summary["data"].append(written)
        if verbose and written["written"]:
            print(
                "[preprocess-shards] wrote "
                f"data rows={written['rows']} size={written['size_mb']:.2f} MB -> {out_path}"
            )

        del data, chunks
        gc.collect()

    with (store_dir / "shape_metadata.json").open("w") as stream:
        json.dump(shape_metadata, stream)
    PostProcessor.merge([train_stats], saved_results_path=store_dir)

    summary_path = store_dir / "preprocess_shards_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"[preprocess-shards] wrote {summary_path}")
        print(f"[preprocess-shards] wrote {store_dir / 'normalization.pt'}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess sharded EveNet NPZ files into bounded-memory parquet shards."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to evenet_input_shards_manifest.json from build_evenet_input_from_parquet.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="EveNet preprocessing YAML, usually config/preprocess_config.yaml.",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        required=True,
        help="Output directory for train parquet shards, val/test subdirs, and normalization.pt.",
    )
    parser.add_argument(
        "--split-ratio",
        default="0.4,0.1,0.5",
        help="Event split ratio train,val,test. Default: 0.4,0.1,0.5.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Do not convert data_shards from the manifest into store-dir/data.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    global_config.load_yaml(str(args.config))
    split_ratio = parse_split_ratio(args.split_ratio)
    shards = load_shards(args.manifest, "training_shards")
    if not shards:
        raise ValueError(f"No training_shards found in {args.manifest}")
    data_shards = [] if args.skip_data else load_shards(args.manifest, "data_shards")

    key_a, key_b = global_config.event_info.classification_names[0].split("/")
    process_ids = global_config.event_info.class_label[key_a][key_b][0]
    assignment_keys, _ = generate_assignment_names(global_config.event_info)

    preprocess_shards(
        shards=shards,
        data_shards=data_shards,
        store_dir=args.store_dir,
        split_ratio=split_ratio,
        unique_process_ids=process_ids,
        assignment_keys=assignment_keys,
        verbose=True,
    )


if __name__ == "__main__":
    main()
