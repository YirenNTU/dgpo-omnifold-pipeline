#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


from evenet.control.global_config import global_config
from preprocessing.helper import (
    PostProcessor,
    build_log_scale_plan,
    event_split_indices,
    process_dict,
    slice_event_dict,
)

def ensure_global_config_loaded(config_path: Path) -> None:
    if not getattr(global_config, "loaded", False):
        global_config.load_yaml(str(config_path))

def parse_split_ratio(value: str) -> tuple[float, float, float]:
    parts = tuple(float(item) for item in value.split(","))
    if len(parts) != 3:
        raise ValueError("--split-ratio must have exactly three values: train,val,test")
    if not np.isclose(sum(parts), 1.0):
        raise ValueError(f"--split-ratio must sum to one, got {parts}")
    return parts

def event_info_class_labels() -> list[str]:
    key_a, key_b = global_config.event_info.classification_names[0].split("/")
    return list(global_config.event_info.class_label[key_a][key_b][0])

def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    return manifest

def resolve_shard_groups(manifest_path: Path, manifest: dict[str, Any], *, include_data: bool) -> tuple[list[Path], list[Path]]:
    base_dir = manifest_path.parent
    training_shards: list[Path] = []
    data_shards: list[Path] = []
    samples = manifest.get("samples") or {}
    shards = manifest.get("shards") or {}
    for sample_key, shard_entries in shards.items():
        sample_cfg = samples.get(sample_key) or {}
        target = data_shards if bool(sample_cfg.get("is_data", False)) else training_shards
        if target is data_shards and not include_data:
            continue
        for entry in shard_entries:
            target.append((base_dir / entry["path"]).resolve())
    return training_shards, data_shards

def generate_assignment_names(event_info):
    assignment_names = []
    for process_name, children in event_info.product_particles.items():
        for particle_name, daughter_particles in children.items():
            assignment_names.append((process_name, particle_name, daughter_particles))
    return [f"TARGETS/{process_name}/{particle_name}" for process_name, particle_name, _ in assignment_names]

def numeric_array_from_field(values: ak.Array) -> np.ndarray | None:
    if "var *" in str(ak.type(values)):
        return None
    if getattr(values, "fields", []):
        return None
    try:
        array = ak.to_numpy(values, allow_missing=False)
    except Exception:
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    return np.ascontiguousarray(array)


def read_numeric_schema(path: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    parquet = pq.ParquetFile(path)
    schema: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for record_batch in parquet.iter_batches(batch_size=1):
        events = ak.from_arrow(record_batch)
        for field in events.fields:
            array = numeric_array_from_field(events[field])
            if array is None:
                continue
            schema[field] = (tuple(array.shape[1:]), array.dtype)
        break
    return schema

def build_reference_schema(shards: list[Path]) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    reference: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for path in shards:
        schema = read_numeric_schema(path)
        for key, (tail_shape, dtype) in schema.items():
            if key not in reference:
                reference[key] = (tail_shape, dtype)
                continue
            ref_shape, ref_dtype = reference[key]
            if ref_shape != tail_shape:
                raise ValueError(
                    f"Reference schema mismatch for {key}: {ref_shape} vs {tail_shape} in {path}"
                )
            if ref_dtype != dtype:
                reference[key] = (ref_shape, np.promote_types(ref_dtype, dtype))
    return reference

def default_missing_array(key: str, tail_shape: tuple[int, ...], dtype: np.dtype, num_events: int, *, is_data: bool) -> np.ndarray:
    shape = (num_events,) + tail_shape
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(shape, dtype=bool)
    if np.issubdtype(dtype, np.integer):
        if key == "classification":
            return np.full(shape, -1 if is_data else 0, dtype=dtype)
        return np.zeros(shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return np.zeros(shape, dtype=dtype)
    raise ValueError(f"Unsupported dtype for default array: key={key}, dtype={dtype}")


def load_parquet_numeric_dict(path: Path, reference_schema: dict[str, tuple[tuple[int, ...], np.dtype]], *, is_data: bool) -> dict[str, np.ndarray]:
    events = ak.from_parquet(path)
    data: dict[str, np.ndarray] = {}
    for field in events.fields:
        array = numeric_array_from_field(events[field])
        if array is None:
            continue
        data[field] = np.ascontiguousarray(array)
    if not data:
        raise ValueError(f"No numeric arrays found in {path}")
    num_events = len(next(iter(data.values())))
    for key, (tail_shape, dtype) in reference_schema.items():
        if key in data:
            if tuple(data[key].shape[1:]) != tail_shape:
                raise ValueError(
                    f"{path} has inconsistent shape for {key}: {data[key].shape[1:]} vs {tail_shape}"
                )
            if data[key].dtype != dtype:
                data[key] = np.ascontiguousarray(data[key].astype(dtype, copy=False))
            continue
        data[key] = default_missing_array(key, tail_shape, dtype, num_events, is_data=is_data)
    return data

def split_output_path(store_dir: Path, split_name: str, shard_index: int) -> Path:
    return store_dir / split_name / f"{split_name}_{shard_index:06d}.parquet"

def write_chunk_table(chunks: list[pa.Table], out_path: Path, *, shuffle_seed: int | None) -> dict[str, Any]:
    if not chunks:
        return {"path": str(out_path), "rows": 0, "written": False}
    table = pa.concat_tables(chunks)
    if shuffle_seed is not None and table.num_rows > 1:
        order = np.random.default_rng(shuffle_seed).permutation(table.num_rows)
        table = table.take(pa.array(order))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    return {
        "path": str(out_path),
        "rows": table.num_rows,
        "size_mb": table.nbytes / 1024 / 1024,
        "written": True,
    }


def process_training_shard_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(payload["config_path"])
    ensure_global_config_loaded(config_path)
    shard_index = int(payload["shard_index"])
    shard_path = Path(payload["shard_path"])
    store_dir = Path(payload["store_dir"])
    split_ratio = tuple(payload["split_ratio"])
    reference_schema = payload["reference_schema"]
    unique_process_ids = payload["unique_process_ids"]
    assignment_keys = payload["assignment_keys"]
    verbose = bool(payload["verbose"])

    train_stats = PostProcessor(global_config)
    log_scale_plan = build_log_scale_plan(global_config)
    rng = np.random.default_rng(int(payload["split_seed"]) + shard_index)
    if verbose:
        print(f"[lite-preprocess] loading training shard {shard_index} {shard_path}", flush=True)

    data = load_parquet_numeric_dict(shard_path, reference_schema, is_data=False)
    n_events = len(data["x"])
    split_indices = dict(zip(("train", "val", "test"), event_split_indices(n_events, split_ratio, rng)))
    result = {"shard_index": shard_index, "train": [], "val": [], "test": [], "shape_metadata": None, "train_stats": train_stats}

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
            shape_metadata=result["shape_metadata"],
            store_chunks=chunks,
        )
        if shape_metadata is not None:
            result["shape_metadata"] = shape_metadata
        out_path = split_output_path(store_dir, split_name, shard_index)
        result[split_name].append(
            write_chunk_table(chunks, out_path, shuffle_seed=int(payload["shuffle_seed"]) + shard_index)
        )
        del pdict, chunks
        gc.collect()
    del data, split_indices
    gc.collect()
    result["train_stats"] = train_stats
    return result

def process_data_shard_worker(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(payload["config_path"])
    ensure_global_config_loaded(config_path)
    shard_index = int(payload["shard_index"])
    shard_path = Path(payload["shard_path"])
    store_dir = Path(payload["store_dir"])
    reference_schema = payload["reference_schema"]
    unique_process_ids = payload["unique_process_ids"]
    assignment_keys = payload["assignment_keys"]
    verbose = bool(payload["verbose"])
    log_scale_plan = build_log_scale_plan(global_config)
    if verbose:
        print(f"[lite-preprocess] loading data shard {shard_index} {shard_path}", flush=True)

    data = load_parquet_numeric_dict(shard_path, reference_schema, is_data=True)
    chunks: list[pa.Table] = []
    process_dict(
        data,
        global_config=global_config,
        unique_process_ids=unique_process_ids,
        assignment_keys=assignment_keys,
        log_scale_plan=log_scale_plan,
        statistics=None,
        shape_metadata=None,
        store_chunks=chunks,
    )
    out_path = store_dir / "data" / f"data_{shard_index:06d}.parquet"
    written = write_chunk_table(chunks, out_path, shuffle_seed=None)
    del data, chunks
    gc.collect()
    return {"shard_index": shard_index, "data": [written]}


def run_tasks(task_payloads: list[dict[str, Any]], worker_fn, num_workers: int, label: str) -> list[dict[str, Any]]:
    if not task_payloads:
        return []
    worker_count = max(1, min(int(num_workers), len(task_payloads)))
    if worker_count == 1:
        return [worker_fn(payload) for payload in task_payloads]
    print(f"[lite-preprocess] {label} workers={worker_count} tasks={len(task_payloads)}", flush=True)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker_fn, payload) for payload in task_payloads]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[lite-preprocess] {label} done {done}/{len(task_payloads)} shard={result['shard_index']}",
                flush=True,
            )
    return sorted(results, key=lambda item: item["shard_index"])

def preprocess_shards(
    *,
    training_shards: list[Path],
    data_shards: list[Path],
    store_dir: Path,
    split_ratio: tuple[float, float, float],
    unique_process_ids: list[str],
    assignment_keys: list[str],
    config_path: Path,
    num_workers: int,
    split_seed: int,
    shuffle_seed: int,
    verbose: bool,
) -> dict[str, Any]:
    store_dir.mkdir(parents=True, exist_ok=True)
    (store_dir / "train").mkdir(parents=True, exist_ok=True)
    (store_dir / "val").mkdir(parents=True, exist_ok=True)
    (store_dir / "test").mkdir(parents=True, exist_ok=True)
    (store_dir / "data").mkdir(parents=True, exist_ok=True)

    reference_schema = build_reference_schema(training_shards)
    summary = {
        "store_dir": str(store_dir),
        "split_ratio": list(split_ratio),
        "train": [],
        "val": [],
        "test": [],
        "data": []
    }

    common_payload = {
        "config_path": str(config_path),
        "store_dir": str(store_dir),
        "split_ratio": list(split_ratio),
        "reference_schema": reference_schema,
        "unique_process_ids": list(unique_process_ids),
        "assignment_keys": list(assignment_keys),
        "verbose": verbose,
        "split_seed": split_seed,
        "shuffle_seed": shuffle_seed,
    }

    train_payloads = [
        {**common_payload, "shard_index": shard_index, "shard_path": str(shard_path)}
        for shard_index, shard_path in enumerate(training_shards)
    ]
    train_results = run_tasks(train_payloads, process_training_shard_worker, num_workers, "training")
    shape_metadata = None
    train_stats: list[PostProcessor] = []
    for result in train_results:
        train_stats.append(result["train_stats"])
        if shape_metadata is None:
            shape_metadata = result["shape_metadata"]
        elif result["shape_metadata"] is not None and shape_metadata != result["shape_metadata"]:
            raise AssertionError("Shape metadata mismatch across worker results.")
        for split_name in ("train", "val", "test"):
            summary[split_name].extend(result[split_name])

    data_payloads = [
        {**common_payload, "shard_index": shard_index, "shard_path": str(shard_path)}
        for shard_index, shard_path in enumerate(data_shards)
    ]
    data_results = run_tasks(data_payloads, process_data_shard_worker, num_workers, "data")
    for result in data_results:
        summary["data"].extend(result["data"])

    with (store_dir / "shape_metadata.json").open("w") as stream:
        json.dump(shape_metadata, stream)
    PostProcessor.merge(train_stats, saved_results_path=store_dir)

    summary_path = store_dir / "preprocess_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if verbose:
        print(f"[lite-preprocess] wrote {summary_path}")
        print(f"[lite-preprocess] wrote {store_dir / 'normalization.pt'}")
    return summary



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess ml_pipeline_lite EveNet parquet shards into shuffled train/val/test parquet files. "
            "This stage keeps event weights as-is and only applies the standard EveNet preprocessing transforms."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.json from ml_pipeline_lite/build_evenet_input_from_parquet.py.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml_pipeline_lite/config/preprocess_config.yaml"),
        help="EveNet preprocessing config.",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        required=True,
        help="Output directory for preprocessed parquet files and normalization.pt.",
    )
    parser.add_argument(
        "--split-ratio",
        default="0.4,0.1,0.5",
        help="Event split ratio train,val,test. Default: 0.4,0.1,0.5.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data shards.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of CPU worker processes.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for event-level split shuffling.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=31,
        help="Seed for per-output parquet shuffling.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    config_path = args.config
    ensure_global_config_loaded(config_path)
    split_ratio = parse_split_ratio(args.split_ratio)
    manifest_path = args.manifest
    manifest = read_manifest(manifest_path)

    training_shards, data_shards = resolve_shard_groups(manifest_path, manifest, include_data=not args.skip_data)
    manifest_class_labels = list(manifest.get("class_labels", []))
    config_class_labels = event_info_class_labels()
    process_ids = manifest_class_labels or config_class_labels
    if manifest_class_labels and manifest_class_labels != config_class_labels:
        print(
            "[lite-preprocess] warning: manifest class_labels differ from preprocess event_info class labels. "
            "Using manifest class_labels for preprocessing statistics. "
            "Regenerate generated_event_info.yaml before training to keep class ordering aligned.",
            flush=True,
        )

    assignment_keys = generate_assignment_names(global_config.event_info)

    preprocess_shards(
        training_shards=training_shards,
        data_shards=data_shards,
        store_dir=args.store_dir,
        split_ratio=split_ratio,
        unique_process_ids=process_ids,
        assignment_keys=assignment_keys,
        config_path=config_path,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
        shuffle_seed=args.shuffle_seed,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()