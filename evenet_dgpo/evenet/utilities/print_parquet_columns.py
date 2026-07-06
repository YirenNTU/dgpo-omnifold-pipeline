#!/usr/bin/env python3
r"""
Print Parquet schema (column names and Arrow types) using PyArrow, consistent with EveNet
data I/O: ``preprocessing/preprocess.py`` and ``evenet/playground_new.py`` use
``pyarrow.parquet``; training reads the same files via ``ray.data.read_parquet`` in
``evenet/shared.py``.

**Inspect what is inside columns** (actual values, not only dtypes):

- ``--only COL ... --head N`` — print the first *N* rows for those columns only (small read).
- ``--only COL ... --scan`` — stream the whole file in batches and print per-column summaries
  (min, max, mean, std, nulls; for integers, distinct count and top value frequencies if tractable).

Parquet **footer** metadata: ``--stats`` (no data read).

Examples::

  python evenet/utilities/print_parquet_columns.py /path/to/dataset_parquet_dir/
  python evenet/utilities/print_parquet_columns.py /path/to/file.parquet \\
      --only syst_jes syst_met_px syst_met_py syst_tag --head 20
  python evenet/utilities/print_parquet_columns.py /path/to/file.parquet \\
      --only syst_jes syst_met_px syst_met_py syst_tag --scan
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import pyarrow.types as pat


def _collect_parquet_files(path: Path, *, recursive: bool) -> list[Path]:
    """Return sorted Parquet file paths for a file or directory."""
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Not a .parquet file: {path}")
        return [path]
    if path.is_dir():
        pattern = "**/*.parquet" if recursive else "*.parquet"
        files = sorted(path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No Parquet files under {path} (pattern {pattern!r})")
        return files
    raise FileNotFoundError(f"Path does not exist: {path}")


def _schema_field_names(schema: Any) -> set[str]:
    return {schema.field(i).name for i in range(len(schema))}


def _aggregate_footer_statistics(pf: pq.ParquetFile, names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Merge per-row-group Parquet statistics for named columns (metadata only)."""
    out: Dict[str, Dict[str, Any]] = {
        n: {"mins": [], "maxs": [], "null_count": 0, "num_values": 0} for n in names
    }
    want = set(names)
    meta = pf.metadata
    for rgi in range(meta.num_row_groups):
        rg = meta.row_group(rgi)
        for ci in range(rg.num_columns):
            ch = rg.column(ci)
            path = ch.path_in_schema
            if isinstance(path, (list, tuple)):
                col_name = ".".join(str(p) for p in path)
            else:
                col_name = str(path)
            if col_name not in want:
                continue
            st = ch.statistics
            if st is None:
                continue
            if getattr(st, "has_min_max", False) and st.min is not None and st.max is not None:
                out[col_name]["mins"].append(st.min)
                out[col_name]["maxs"].append(st.max)
            nc = getattr(st, "null_count", None)
            if nc is not None:
                out[col_name]["null_count"] += int(nc)
            nv = getattr(st, "num_values", None)
            if nv is not None:
                out[col_name]["num_values"] += int(nv)
    return out


def _format_footer_stats(agg: Dict[str, Any]) -> str:
    parts: List[str] = []
    mins: List[Any] = agg.get("mins") or []
    maxs: List[Any] = agg.get("maxs") or []
    if mins and maxs:
        parts.append(f"min={min(mins)!r}, max={max(maxs)!r}")
    else:
        parts.append("min/max not in Parquet footer (or statistics disabled)")
    parts.append(f"null_count={agg['null_count']:,}")
    if agg.get("num_values", 0) > 0:
        parts.append(f"num_values(sum chunks)={agg['num_values']:,}")
    return ", ".join(parts)


def _validate_only_columns(pf: pq.ParquetFile, only: Sequence[str]) -> None:
    schema = pf.schema_arrow
    all_names = _schema_field_names(schema)
    missing = [n for n in only if n not in all_names]
    if missing:
        raise ValueError(
            f"Column(s) not in schema: {missing}. "
            f"(File has {len(all_names)} top-level fields.)"
        )


def print_parquet_head(parquet_path: Path, *, columns: List[str], n: int, show_rows: bool) -> None:
    """Read first *n* rows for *columns* and print them (actual cell values)."""
    pf = pq.ParquetFile(parquet_path)
    _validate_only_columns(pf, columns)
    table = pq.read_table(parquet_path, columns=list(columns))
    table = table.slice(0, min(n, table.num_rows))
    print(parquet_path)
    if show_rows:
        md = pf.metadata
        if md is not None and md.num_rows is not None:
            print(f"  rows (file): {md.num_rows:,}")
    print(f"  showing first {table.num_rows:,} row(s), columns: {', '.join(columns)}")
    col_arrays = [table.column(j) for j in range(table.num_columns)]
    widths = [max(len(columns[j]), 12) for j in range(len(columns))]
    header = "  " + "  ".join(str(columns[j]).ljust(widths[j]) for j in range(len(columns)))
    print(header)
    print("  " + "  ".join("-" * widths[j] for j in range(len(columns))))
    for i in range(table.num_rows):
        cells = []
        for j, w in enumerate(widths):
            v = col_arrays[j][i].as_py()
            cells.append(repr(v).ljust(w))
        print("  " + "  ".join(cells))


class _Welford:
    """Online mean and variance for streaming floats (ignores NaN; skips nulls)."""

    def __init__(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update_array(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        mask = np.isfinite(x)
        for v in x[mask]:
            self._n += 1
            d = v - self._mean
            self._mean += d / self._n
            d2 = v - self._mean
            self._m2 += d * d2

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._n < 2:
            return float("nan")
        return math.sqrt(self._m2 / (self._n - 1))


def _numpy1d(col: Any) -> Tuple[np.ndarray, int]:
    """Convert Arrow column chunk to 1d array; returns (values, null_count)."""
    n = len(col)
    if n == 0:
        return np.array([], dtype=np.float64), 0
    raw = col.to_pylist()
    nulls = sum(1 for v in raw if v is None)
    return np.asarray(raw, dtype=np.float64), nulls


def scan_parquet_columns(parquet_path: Path, *, columns: List[str], batch_size: int) -> None:
    """Stream all row groups and print numeric summaries and integer value distributions."""
    pf = pq.ParquetFile(parquet_path)
    _validate_only_columns(pf, columns)
    schema = pf.schema_arrow
    field_by_name = {schema.field(i).name: schema.field(i) for i in range(len(schema))}

    # Per-column accumulators
    mins: Dict[str, float] = {c: float("inf") for c in columns}
    maxs: Dict[str, float] = {c: float("-inf") for c in columns}
    nulls: Dict[str, int] = {c: 0 for c in columns}
    welford: Dict[str, _Welford] = {c: _Welford() for c in columns}
    int_counters: Dict[str, Counter] = {}
    int_freq_stopped: Dict[str, bool] = {}
    MAX_DISTINCT_KEYS = 10_000

    total_rows = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        total_rows += batch.num_rows
        for j, name in enumerate(columns):
            col = batch.column(j)
            f = field_by_name[name]
            ty = f.type
            is_int = pat.is_integer(ty) or pat.is_unsigned_integer(ty)

            if is_int:
                raw = col.to_pylist()
                for v in raw:
                    if v is None:
                        nulls[name] += 1
                        continue
                    iv = int(v)
                    mins[name] = min(mins[name], float(iv))
                    maxs[name] = max(maxs[name], float(iv))
                    if int_freq_stopped.get(name):
                        continue
                    if name not in int_counters:
                        int_counters[name] = Counter()
                    ctr = int_counters[name]
                    if len(ctr) >= MAX_DISTINCT_KEYS and iv not in ctr:
                        int_freq_stopped[name] = True
                        del int_counters[name]
                    else:
                        ctr[iv] += 1
                continue

            arr, nc = _numpy1d(col)
            nulls[name] += nc
            if arr.size == 0:
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size:
                mins[name] = min(mins[name], float(np.min(finite)))
                maxs[name] = max(maxs[name], float(np.max(finite)))
            welford[name].update_array(arr)

    print(parquet_path)
    md = pf.metadata
    if md is not None and md.num_rows is not None:
        print(f"  rows (metadata): {md.num_rows:,}")
    print(f"  rows (scanned): {total_rows:,}")
    print("  per-column (data scan):")
    for name in columns:
        f = field_by_name[name]
        t = str(f.type)
        ty = f.type
        is_int = pat.is_integer(ty) or pat.is_unsigned_integer(ty)
        if is_int:
            n_non_null = total_rows - nulls[name]
            parts = [
                f"dtype={t}",
                f"min={mins[name]:.0f}" if math.isfinite(mins[name]) else "min=n/a",
                f"max={maxs[name]:.0f}" if math.isfinite(maxs[name]) else "max=n/a",
                f"nulls={nulls[name]:,}",
                f"non_null={n_non_null:,}",
            ]
            if name in int_counters:
                ctr = int_counters[name]
                parts.append(f"n_distinct={len(ctr)}")
                top = ctr.most_common(15)
                freq = ", ".join(f"{k!r}:{v:,}" for k, v in top)
                parts.append(f"value_counts={{{freq}}}")
            elif int_freq_stopped.get(name):
                parts.append(f"value_counts omitted (>{MAX_DISTINCT_KEYS} distinct values)")
            print(f"    {name}: " + ", ".join(parts))
        else:
            w = welford[name]
            parts = [
                f"dtype={t}",
                f"min={mins[name]:.6g}" if math.isfinite(mins[name]) else "min=n/a",
                f"max={maxs[name]:.6g}" if math.isfinite(maxs[name]) else "max=n/a",
                f"mean={w.mean:.6g}" if w.count else "mean=n/a",
                f"std={w.std:.6g}" if w.count > 1 else "std=n/a",
                f"non_null={w.count:,}",
                f"nulls={nulls[name]:,}",
            ]
            print(f"    {name}: " + ", ".join(parts))


def print_parquet_schema(
    parquet_path: Path,
    *,
    show_rows: bool,
    only: Optional[List[str]],
    show_stats: bool,
) -> None:
    """Load schema metadata; optionally restrict columns and print footer statistics."""
    pf = pq.ParquetFile(parquet_path)
    schema = pf.schema_arrow
    all_names = _schema_field_names(schema)

    if only:
        missing = [n for n in only if n not in all_names]
        if missing:
            raise ValueError(
                f"Column(s) not in schema: {missing}. "
                f"(File has {len(all_names)} top-level fields.)"
            )

    print(parquet_path)
    if show_rows:
        md = pf.metadata
        nrows = md.num_rows if md is not None else None
        if nrows is not None:
            print(f"  rows: {nrows:,}")

    name_to_field = {schema.field(i).name: schema.field(i) for i in range(len(schema))}
    if only:
        fields = [name_to_field[n] for n in only]
    else:
        fields = [schema.field(i) for i in range(len(schema))]

    print(f"  columns ({len(fields)}):")
    for field in fields:
        print(f"    {field.name}: {field.type}")

    if show_stats and only:
        agg = _aggregate_footer_statistics(pf, only)
        print("  Parquet footer statistics (merged across row groups):")
        for name in only:
            print(f"    {name}: {_format_footer_stats(agg[name])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Parquet: schema, optional footer stats, sample rows (--head), "
            "or full data summaries (--scan). Uses PyArrow."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        metavar="DIR_OR_FILE",
        help=(
            "Directory containing *.parquet shards (usual EveNet layout), or a single .parquet file"
        ),
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="If path is a directory, include Parquet files in subdirectories",
    )
    parser.add_argument(
        "--no-rows",
        action="store_true",
        help="Do not print row count from file metadata",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="COL",
        default=None,
        help="Restrict to these column names (required for --head, --scan, and --stats)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print merged Parquet footer min/max/null_count when available (requires --only)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        metavar="N",
        help="Print first N rows of actual values (requires --only; reads only those columns)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Stream entire file(s) and print per-column data summaries (requires --only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=262_144,
        metavar="ROWS",
        help="Row batch size for --scan (default: 262144)",
    )
    args = parser.parse_args(argv)
    if args.stats and not args.only:
        parser.error("--stats requires --only COL [COL ...]")
    if args.head is not None and not args.only:
        parser.error("--head N requires --only COL [COL ...]")
    if args.scan and not args.only:
        parser.error("--scan requires --only COL [COL ...]")
    if args.head is not None and args.scan:
        parser.error("--head and --scan cannot be used together")
    if args.head is not None and args.head < 1:
        parser.error("--head N must be >= 1")

    try:
        files = _collect_parquet_files(args.path, recursive=args.recursive)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    show_rows = not args.no_rows

    for i, fpath in enumerate(files):
        if i:
            print()
        try:
            if args.head is not None:
                print_parquet_head(fpath, columns=list(args.only), n=args.head, show_rows=show_rows)
            elif args.scan:
                scan_parquet_columns(fpath, columns=list(args.only), batch_size=args.batch_size)
            else:
                print_parquet_schema(
                    fpath,
                    show_rows=show_rows,
                    only=args.only,
                    show_stats=args.stats,
                )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
