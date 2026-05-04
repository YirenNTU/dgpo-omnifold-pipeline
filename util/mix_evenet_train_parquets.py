#!/usr/bin/env python3
"""Mix EveNet train parquet shards without changing shared normalization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SIDECAR_FILES = (
    "normalization.pt",
    "shape_metadata.json",
    "preprocess_shards_manifest.json",
)
SIDECAR_DIRS = ("val", "test", "data")
DEFAULT_FLOAT = -99.0


def discover_train_files(input_dir: Path) -> list[Path]:
    """
    inputs:
      input_dir: Path, directory containing train_*.parquet shards.
    outputs:
      files: list[Path], sorted train parquet files.
    goal:
      Keep validation/test/data parquet files out of the training mix.
    """
    files = sorted(input_dir.glob("train_*.parquet"))
    if not files:
        files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train parquet files found in {input_dir}")
    return files


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """
    inputs:
      output_dir: Path, destination mixed parquet directory.
      overwrite: bool, allow replacing existing mixed outputs.
    outputs:
      None.
    goal:
      Avoid accidental mixing into an old directory unless explicitly allowed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("train_*.parquet"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains train_*.parquet. "
            "Pass --overwrite or choose a new --output-dir."
        )
    for path in existing:
        path.unlink()


def copy_sidecars(input_dir: Path, output_dir: Path, *, overwrite: bool, copy_non_train: bool) -> None:
    """
    inputs:
      input_dir: Path, source preprocessing output directory.
      output_dir: Path, destination mixed output directory.
      overwrite: bool, allow replacing existing copied directories.
      copy_non_train: bool, copy val/test/data subdirectories.
    outputs:
      None.
    goal:
      Preserve shared normalization and metadata while replacing only the train
      parquet shards.
    """
    for name in SIDECAR_FILES:
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    if not copy_non_train:
        return

    for name in SIDECAR_DIRS:
        source = input_dir / name
        destination = output_dir / name
        if not source.exists():
            continue
        if destination.exists() and overwrite:
            shutil.rmtree(destination)
        shutil.copytree(source, destination, dirs_exist_ok=overwrite)


def promote_arrow_type(left: pa.DataType, right: pa.DataType) -> pa.DataType:
    """
    inputs:
      left/right: pa.DataType, two observed column types.
    outputs:
      dtype: pa.DataType, common type for mixed parquet output.
    goal:
      Allow harmless preprocessing dtype drift, such as float32 vs float64.
    """
    if left == right:
        return left
    if pa.types.is_floating(left) and pa.types.is_floating(right):
        return pa.float64()
    if pa.types.is_integer(left) and pa.types.is_integer(right):
        return pa.int64() if pa.types.is_signed_integer(left) or pa.types.is_signed_integer(right) else pa.uint64()
    if pa.types.is_integer(left) and pa.types.is_floating(right):
        return pa.float64()
    if pa.types.is_floating(left) and pa.types.is_integer(right):
        return pa.float64()
    raise TypeError(f"Cannot promote incompatible parquet column types: {left} vs {right}")


def build_reference_schema(files: list[Path]) -> pa.Schema:
    """
    inputs:
      files: list[Path], source train parquet files.
    outputs:
      schema: pa.Schema, union schema with promoted numeric types.
    goal:
      Make mixing robust to independently written train parquet shards.
    """
    field_by_name: dict[str, pa.Field] = {}
    column_order: list[str] = []
    conflicts: list[str] = []

    for path in files:
        for field in pq.ParquetFile(path).schema_arrow:
            if field.name not in field_by_name:
                field_by_name[field.name] = field
                column_order.append(field.name)
                continue
            old_field = field_by_name[field.name]
            try:
                promoted_type = promote_arrow_type(old_field.type, field.type)
            except TypeError as exc:
                conflicts.append(f"{field.name}: {old_field.type} vs {field.type} in {path.name}: {exc}")
                continue
            field_by_name[field.name] = pa.field(field.name, promoted_type, nullable=old_field.nullable or field.nullable)

    if conflicts:
        raise ValueError("Inconsistent train parquet schema:\n  " + "\n  ".join(conflicts[:20]))
    return pa.schema([field_by_name[name] for name in column_order])


def default_column_array(field: pa.Field, num_rows: int) -> pa.Array:
    """
    inputs:
      field: pa.Field, missing column field definition.
      num_rows: int, number of rows to synthesize.
    outputs:
      array: pa.Array, default values compatible with training input.
    goal:
      Fill rare missing passthrough columns rather than failing during mixing.
    """
    dtype = field.type
    name = field.name
    if pa.types.is_boolean(dtype):
        values = np.zeros(num_rows, dtype=bool)
    elif pa.types.is_integer(dtype):
        default = 0 if name.startswith("truth_num_") else -1
        values = np.full(num_rows, default, dtype=np.int64)
    elif pa.types.is_floating(dtype):
        default = 0.0 if name.startswith("analyzing_power") else DEFAULT_FLOAT
        values = np.full(num_rows, default, dtype=np.float64)
    else:
        raise TypeError(f"Cannot synthesize missing column {name} with type {dtype}")
    return pa.array(values, type=dtype)


def align_table_to_schema(table: pa.Table, reference_schema: pa.Schema) -> pa.Table:
    """
    inputs:
      table: pa.Table, source batch.
      reference_schema: pa.Schema, target schema/order.
    outputs:
      table: pa.Table, columns reordered/cast and missing columns filled.
    goal:
      Ensure pa.concat_tables always sees identical schemas.
    """
    arrays = []
    for field in reference_schema:
        if field.name in table.column_names:
            arrays.append(table[field.name].cast(field.type, safe=False))
        else:
            arrays.append(default_column_array(field, table.num_rows))
    return pa.Table.from_arrays(arrays, schema=reference_schema)


def parquet_batch_iter(path: Path, read_batch_size: int, reference_schema: pa.Schema):
    """
    inputs:
      path: Path, source parquet file.
      read_batch_size: int, rows per streaming batch.
      reference_schema: pa.Schema, target mixed schema.
    outputs:
      iterator yielding pyarrow.Table batches.
    goal:
      Stream parquet rows so the mixer never needs all training events in RAM.
    """
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=read_batch_size):
        yield align_table_to_schema(pa.Table.from_batches([batch]), reference_schema)


def write_mixed_table(
    batches: list[pa.Table],
    output_dir: Path,
    output_index: int,
    rng: np.random.Generator,
    compression: str,
) -> dict:
    """
    inputs:
      batches: list[pa.Table], accumulated interleaved input batches.
      output_dir: Path, destination directory.
      output_index: int, mixed shard index.
      rng: np.random.Generator, row-shuffle source.
      compression: str, parquet compression codec.
    outputs:
      summary: dict, output path and row count.
    goal:
      Shuffle rows within each mixed shard after interleaving process-specific
      source shards.
    """
    table = pa.concat_tables(batches)
    if table.num_rows > 1:
        order = rng.permutation(table.num_rows)
        table = table.take(pa.array(order))

    output_path = output_dir / f"train_{output_index:06d}.parquet"
    pq.write_table(table, output_path, compression=compression)
    summary = {
        "path": str(output_path),
        "rows": int(table.num_rows),
        "size_mb": float(table.nbytes / 1024 / 1024),
    }
    del table
    return summary


def mix_train_parquets(
    *,
    input_dir: Path,
    output_dir: Path,
    rows_per_output: int,
    read_batch_size: int,
    seed: int,
    overwrite: bool,
    copy_non_train: bool,
    compression: str,
) -> dict:
    """
    inputs:
      input_dir: Path, sharded EveNet preprocessing output.
      output_dir: Path, mixed training output.
      rows_per_output: int, approximate rows per mixed train parquet.
      read_batch_size: int, rows read from each source file per turn.
      seed: int, deterministic file and row shuffle seed.
      overwrite: bool, allow replacing existing output shards.
      copy_non_train: bool, copy val/test/data and shared metadata.
      compression: str, parquet compression codec.
    outputs:
      summary: dict, source files and mixed shard metadata.
    goal:
      Rebucket process-ordered train parquet files into mixed shards so Ray's
      file-level shuffling sees files with a healthier process composition.
    """
    if rows_per_output <= 0:
        raise ValueError("--rows-per-output must be positive")
    if read_batch_size <= 0:
        raise ValueError("--read-batch-size must be positive")

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Use a separate --output-dir so the original train shards are preserved.")

    files = discover_train_files(input_dir)
    reference_schema = build_reference_schema(files)
    prepare_output_dir(output_dir, overwrite=overwrite)
    copy_sidecars(input_dir, output_dir, overwrite=overwrite, copy_non_train=copy_non_train)

    rng = np.random.default_rng(seed)
    file_order = list(files)
    rng.shuffle(file_order)
    iterators = {path: parquet_batch_iter(path, read_batch_size, reference_schema) for path in file_order}
    active = list(file_order)

    output_index = 0
    buffered_rows = 0
    buffered_batches: list[pa.Table] = []
    outputs: list[dict] = []

    while active:
        rng.shuffle(active)
        next_active: list[Path] = []
        for path in active:
            try:
                batch = next(iterators[path])
            except StopIteration:
                continue

            buffered_batches.append(batch)
            buffered_rows += batch.num_rows
            next_active.append(path)

            if buffered_rows >= rows_per_output:
                outputs.append(
                    write_mixed_table(
                        buffered_batches,
                        output_dir,
                        output_index,
                        rng,
                        compression,
                    )
                )
                output_index += 1
                buffered_rows = 0
                buffered_batches = []

        active = next_active

    if buffered_batches:
        outputs.append(
            write_mixed_table(
                buffered_batches,
                output_dir,
                output_index,
                rng,
                compression,
            )
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "rows_per_output": rows_per_output,
        "read_batch_size": read_batch_size,
        "num_input_files": len(files),
        "num_output_files": len(outputs),
        "input_files": [str(path) for path in files],
        "outputs": outputs,
        "copied_non_train": bool(copy_non_train),
    }
    (output_dir / "mix_evenet_train_parquets_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mix sharded EveNet training parquet files while preserving shared normalization."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing train_*.parquet plus normalization.pt and shape_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for mixed train_*.parquet files.",
    )
    parser.add_argument(
        "--rows-per-output",
        type=int,
        default=100000,
        help="Approximate rows per mixed output parquet. Default: 100000.",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=8192,
        help="Rows read from each source parquet per interleaving turn. Default: 8192.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--compression", default="snappy", help="Parquet compression codec. Default: snappy.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing mixed train shards.")
    parser.add_argument(
        "--no-copy-non-train",
        action="store_true",
        help="Do not copy val/test/data directories into the mixed output directory.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = mix_train_parquets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rows_per_output=args.rows_per_output,
        read_batch_size=args.read_batch_size,
        seed=args.seed,
        overwrite=args.overwrite,
        copy_non_train=not args.no_copy_non_train,
        compression=args.compression,
    )
    print(
        "[mix-evenet-train] wrote "
        f"{summary['num_output_files']} mixed train parquet file(s) to {summary['output_dir']}"
    )
    print(f"[mix-evenet-train] summary={Path(summary['output_dir']) / 'mix_evenet_train_parquets_summary.json'}")


if __name__ == "__main__":
    main()
