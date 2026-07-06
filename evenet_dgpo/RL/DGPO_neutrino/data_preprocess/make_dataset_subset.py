#!/usr/bin/env python3
"""Materialize a fixed Parquet train/val subset from a full training directory.

Reads ``subset_dataset_object.yaml`` in this directory (``RL/DGPO_neutrino/data_preprocess/``), takes a fraction of
all rows (optionally shuffled), splits the kept rows by ``val_split`` on subset indices,
and writes one Parquet file per split plus optional ``shape_metadata.json`` /
``normalization.pt`` copies.

This mirrors the semantics of ``evenet/shared.py`` when using a single Parquet directory:
``dataset_limit`` is applied first, then ``val_split`` on the limited row stream.

Usage::

    python RL/DGPO_neutrino/data_preprocess/make_dataset_subset.py RL/DGPO_neutrino/data_preprocess/subset_dataset_object.yaml
    python RL/DGPO_neutrino/data_preprocess/make_dataset_subset.py path/to/subset_config.yaml --force
"""

from __future__ import annotations

import argparse
import inspect
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "make_dataset_subset.py requires pyarrow. Install pyarrow in your environment."
    ) from e

_log = logging.getLogger(__name__)

SHAPE_METADATA = "shape_metadata.json"
NORMALIZATION_PT = "normalization.pt"
TRAIN_OUT_NAME = "subset_train.parquet"
VAL_OUT_NAME = "subset_val.parquet"


def _parquet_writer_init_param_names() -> set[str]:
    sig = inspect.signature(pq.ParquetWriter.__init__)
    return {name for name in sig.parameters if name != "self"}


def _filter_parquet_writer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = _parquet_writer_init_param_names()
    return {k: v for k, v in kwargs.items() if k in allowed}


def _normalize_compression_for_writer(codec: Any) -> str | None:
    if codec is None:
        return None
    if isinstance(codec, str):
        name = codec.upper()
    else:
        name = str(getattr(codec, "name", codec)).upper()
        if name.startswith("COMPRESSION."):
            name = name.split(".")[-1]
    if name in ("UNCOMPRESSED", "NONE", "UNKNOWN"):
        return "none"
    return name.lower()


def _infer_parquet_write_options_from_file(path: Path) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    if meta is None or meta.num_row_groups < 1:
        return opts
    rg0 = meta.row_group(0)
    opts["row_group_size"] = rg0.num_rows
    if rg0.num_columns > 0:
        comp = _normalize_compression_for_writer(rg0.column(0).compression)
        if comp is not None:
            opts["compression"] = comp
    fv = getattr(meta, "format_version", None)
    if callable(fv):
        fv = fv()
    if isinstance(fv, tuple) and len(fv) >= 2:
        opts["version"] = f"{fv[0]}.{fv[1]}"
    elif isinstance(fv, str) and fv:
        opts["version"] = fv
    return opts


def _open_parquet_writer(
    out_path: Path,
    schema: pa.Schema,
    writer_kwargs: dict[str, Any],
) -> pq.ParquetWriter:
    try:
        return pq.ParquetWriter(str(out_path), schema=schema, **writer_kwargs)
    except Exception as e:
        if writer_kwargs:
            _log.warning(
                "ParquetWriter rejected options %s (%s); retrying with schema only",
                writer_kwargs,
                e,
            )
        return pq.ParquetWriter(str(out_path), schema=schema)


def _list_train_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No *.parquet files in {data_dir} (non-recursive; matches evenet/shared.py)."
        )
    return files


def _count_rows(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        meta = pq.ParquetFile(p).metadata
        if meta is None:
            raise ValueError(f"No metadata for {p}")
        total += meta.num_rows
    return total


def _validate_schemas(paths: list[Path], *, check_metadata: bool = False) -> pa.Schema:
    ref = pq.ParquetFile(paths[0]).schema_arrow
    for p in paths[1:]:
        other = pq.ParquetFile(p).schema_arrow
        if not ref.equals(other, check_metadata=check_metadata):
            raise ValueError(f"Schema mismatch between {paths[0]} and {p}")
    return ref


def _copy_sidecars_if_requested(
    source_dir: Path,
    train_dir: Path,
    val_dir: Path,
    *,
    copy_sidecars: bool,
) -> None:
    """Copy source sidecars into both output dirs.

    Copies shape_metadata.json (which, on an augmented *_evttok / *_evttok_obj mirror,
    already describes ``event_token: [D]`` and ``object_token: [P, D]``) and
    normalization.pt, PLUS any other non-parquet sidecar present in ``source_dir`` --
    mirroring ``augment_event_token.py`` so the current augmented version's sidecars are
    never silently dropped. The subset carries the token columns verbatim (row/column
    copy), so together the downstream pipeline reconstructs them via ``unflatten_dict``.
    """
    if not copy_sidecars:
        return
    # Explicit essentials first (logged), then any remaining non-parquet sidecar.
    seen: set[str] = set()
    ordered = [SHAPE_METADATA, NORMALIZATION_PT]
    extras = sorted(
        p.name
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix != ".parquet" and p.name not in ordered
    )
    for name in ordered + extras:
        if name in seen:
            continue
        seen.add(name)
        src = source_dir / name
        if not src.is_file():
            if name in (SHAPE_METADATA, NORMALIZATION_PT):
                _log.info("Sidecar not found, skip: %s", src)
            continue
        for d in (train_dir, val_dir):
            d.mkdir(parents=True, exist_ok=True)
            dst = d / name
            shutil.copy2(str(src), str(dst))
            _log.info("Copied sidecar %s -> %s", src, dst)


def _route_subset_positions_to_tables(
    taken: pa.RecordBatch,
    subset_positions: np.ndarray,
    *,
    val_start: int,
    val_end: int,
) -> tuple[pa.Table | None, pa.Table | None]:
    """Split *taken* rows by subset index positions (0..n_keep-1)."""
    if len(subset_positions) != taken.num_rows:
        raise ValueError("subset_positions length must match batch size")
    train_mask = (subset_positions < val_start) | (subset_positions >= val_end)
    val_mask = (subset_positions >= val_start) & (subset_positions < val_end)
    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    train_tb = pa.Table.from_batches([taken.take(train_idx)]) if train_idx.size else None
    val_tb = pa.Table.from_batches([taken.take(val_idx)]) if val_idx.size else None
    return train_tb, val_tb


def _run_sequential(
    paths: list[Path],
    *,
    ref_schema: pa.Schema,
    n_keep: int,
    val_start: int,
    val_end: int,
    batch_size: int,
    train_path: Path,
    val_path: Path,
    writer_kwargs: dict[str, Any],
) -> tuple[int, int]:
    """First n_keep rows in file order; subset position equals global index in [0, n_keep)."""
    train_w = _open_parquet_writer(train_path, ref_schema, writer_kwargs)
    val_w = _open_parquet_writer(val_path, ref_schema, writer_kwargs)
    train_rows = val_rows = 0
    global_row = 0
    try:
        for src in paths:
            pf = pq.ParquetFile(src)
            for batch in pf.iter_batches(batch_size=batch_size):
                n = batch.num_rows
                # Batch covers global indices [global_row, global_row + n).
                g_start = global_row
                g_end = min(global_row + n, n_keep)
                if g_end > g_start:
                    local_lo = g_start - global_row
                    local_len = g_end - g_start
                    slice_batch = batch.slice(local_lo, local_len)
                    positions = np.arange(g_start, g_end, dtype=np.int64)
                    train_tb, val_tb = _route_subset_positions_to_tables(
                        slice_batch, positions, val_start=val_start, val_end=val_end
                    )
                    if train_tb is not None and train_tb.num_rows > 0:
                        train_w.write_table(train_tb)
                        train_rows += train_tb.num_rows
                    if val_tb is not None and val_tb.num_rows > 0:
                        val_w.write_table(val_tb)
                        val_rows += val_tb.num_rows
                global_row += n
                if global_row >= n_keep:
                    break
            if global_row >= n_keep:
                break
    finally:
        train_w.close()
        val_w.close()
    return train_rows, val_rows


def _run_shuffled(
    paths: list[Path],
    *,
    ref_schema: pa.Schema,
    sorted_global_indices: np.ndarray,
    n_keep: int,
    val_start: int,
    val_end: int,
    batch_size: int,
    train_path: Path,
    val_path: Path,
    writer_kwargs: dict[str, Any],
) -> tuple[int, int]:
    train_w = _open_parquet_writer(train_path, ref_schema, writer_kwargs)
    val_w = _open_parquet_writer(val_path, ref_schema, writer_kwargs)
    train_rows = val_rows = 0
    file_start = 0
    try:
        for src in paths:
            pf = pq.ParquetFile(src)
            n_file = int(pf.metadata.num_rows) if pf.metadata else 0
            file_end = file_start + n_file

            for batch in pf.iter_batches(batch_size=batch_size):
                n = batch.num_rows
                b0 = file_start
                b1 = file_start + n
                lo = int(np.searchsorted(sorted_global_indices, b0, side="left"))
                hi = int(np.searchsorted(sorted_global_indices, b1, side="left"))
                if lo < hi:
                    globals_needed = sorted_global_indices[lo:hi]
                    local_rows = (globals_needed - b0).astype(np.int32, copy=False)
                    taken = batch.take(pa.array(local_rows))
                    subset_positions = np.arange(lo, hi, dtype=np.int64)
                    train_tb, val_tb = _route_subset_positions_to_tables(
                        taken, subset_positions, val_start=val_start, val_end=val_end
                    )
                    if train_tb is not None and train_tb.num_rows > 0:
                        train_w.write_table(train_tb)
                        train_rows += train_tb.num_rows
                    if val_tb is not None and val_tb.num_rows > 0:
                        val_w.write_table(val_tb)
                        val_rows += val_tb.num_rows
                file_start = b1
            if file_start != file_end:
                _log.warning(
                    "Row count mismatch for %s: expected file_end=%s got %s",
                    src,
                    file_end,
                    file_start,
                )
            file_start = file_end
    finally:
        train_w.close()
        val_w.close()

    if train_rows + val_rows != n_keep:
        _log.warning(
            "Shuffled output rows %s + %s = %s, expected n_keep=%s",
            train_rows,
            val_rows,
            train_rows + val_rows,
            n_keep,
        )
    return train_rows, val_rows


def load_subset_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fixed train/val Parquet subset from subset_dataset_object.yaml"
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "subset_dataset_object.yaml",
        help="Path to the subset config YAML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output Parquet files if present.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65_536,
        help="PyArrow iter_batches batch size (default 65536).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg_path = args.config.expanduser().resolve()
    cfg = load_subset_config(cfg_path)

    source = cfg.get("source") or {}
    output = cfg.get("output") or {}
    subset = cfg.get("subset") or {}

    data_dir = Path(str(source.get("data_parquet_dir", ""))).expanduser().resolve()
    train_dir = Path(str(output.get("train_dir", ""))).expanduser().resolve()
    val_dir = Path(str(output.get("val_dir", ""))).expanduser().resolve()
    copy_sidecars = bool(output.get("copy_sidecars", True))

    fraction = float(subset.get("fraction", 1.0))
    val_split = subset.get("val_split", [0.7, 1.0])
    seed = int(subset.get("seed", 42))
    shuffle = bool(subset.get("shuffle", False))

    if not data_dir.is_dir():
        raise SystemExit(f"source.data_parquet_dir is not a directory: {data_dir}")
    if not train_dir or not val_dir:
        raise SystemExit("output.train_dir and output.val_dir are required.")
    if not (0.0 < fraction <= 1.0):
        raise SystemExit(f"subset.fraction must be in (0, 1], got {fraction}")
    if (
        not isinstance(val_split, (list, tuple))
        or len(val_split) != 2
    ):
        raise SystemExit("subset.val_split must be a list of two floats [start, end).")
    a, b = float(val_split[0]), float(val_split[1])
    # Allow an empty val range (a == b) so all kept rows go to train_dir -- useful when
    # validation comes from a separate dir (e.g. a mg5_test slice) instead of a train split.
    if not (0.0 <= a <= b <= 1.0):
        raise SystemExit(f"subset.val_split must satisfy 0 <= start <= end <= 1, got {val_split}")
    if a == b:
        _log.info("subset.val_split=%s is an empty range -> all kept rows go to train_dir.", val_split)

    train_out = train_dir / TRAIN_OUT_NAME
    val_out = val_dir / VAL_OUT_NAME
    if train_out.exists() or val_out.exists():
        if not args.force:
            raise SystemExit(
                f"Output exists ({train_out} or {val_out}). Pass --force to overwrite."
            )

    paths = _list_train_parquet_files(data_dir)
    ref_schema = _validate_schemas(paths)
    total_rows = _count_rows(paths)
    if total_rows == 0:
        raise SystemExit(f"No rows found under {data_dir} (*.parquet).")
    n_keep = max(1, int(total_rows * fraction)) if fraction < 1.0 else total_rows
    n_keep = min(n_keep, total_rows)

    val_start = int(n_keep * a)
    val_end = int(n_keep * b)
    if val_end <= val_start:
        _log.warning(
            "Empty validation range: val_start=%s val_end=%s (n_keep=%s). All kept rows go to train.",
            val_start,
            val_end,
            n_keep,
        )

    inferred = _infer_parquet_write_options_from_file(paths[0])
    writer_kwargs = _filter_parquet_writer_kwargs(inferred)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    if train_out.exists():
        train_out.unlink()
    if val_out.exists():
        val_out.unlink()

    _log.info("Source rows (all shards): %s", f"{total_rows:,}")
    _log.info("Keeping rows: %s (fraction=%s)", f"{n_keep:,}", fraction)
    _log.info(
        "Val subset index range: [%s, %s) of %s (val_split=%s)",
        val_start,
        val_end,
        n_keep,
        val_split,
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        sorted_global_indices = np.sort(
            rng.choice(total_rows, size=n_keep, replace=False)
        )
        tr, vr = _run_shuffled(
            paths,
            ref_schema=ref_schema,
            sorted_global_indices=sorted_global_indices,
            n_keep=n_keep,
            val_start=val_start,
            val_end=val_end,
            batch_size=args.batch_size,
            train_path=train_out,
            val_path=val_out,
            writer_kwargs=writer_kwargs,
        )
    else:
        tr, vr = _run_sequential(
            paths,
            ref_schema=ref_schema,
            n_keep=n_keep,
            val_start=val_start,
            val_end=val_end,
            batch_size=args.batch_size,
            train_path=train_out,
            val_path=val_out,
            writer_kwargs=writer_kwargs,
        )

    if tr + vr != n_keep:
        _log.warning(
            "Train+val row count %s + %s = %s differs from n_keep=%s",
            tr,
            vr,
            tr + vr,
            n_keep,
        )

    _copy_sidecars_if_requested(
        data_dir,
        train_dir,
        val_dir,
        copy_sidecars=copy_sidecars,
    )

    _log.info("Wrote train: %s (%s rows)", train_out, f"{tr:,}")
    _log.info("Wrote val:   %s (%s rows)", val_out, f"{vr:,}")
    _log.info("Done. Set platform.data_parquet_dir=%s", train_dir)
    _log.info("        platform.data_parquet_val_dir=%s", val_dir)
    _log.info("        options.Dataset.dataset_limit=1.0")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
