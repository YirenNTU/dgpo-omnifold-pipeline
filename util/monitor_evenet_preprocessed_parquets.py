#!/usr/bin/env python3
"""Monitor sharded EveNet preprocessing parquet outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_MONITOR_COLUMNS = (
    "event_weight",
    "num_vectors",
    "num_sequential_vectors",
    "central_weight",
    "classification",
    "source_sample_index",
)
REQUIRED_TRAIN_COLUMNS = ("classification", "event_weight")
SIDECAR_FILES = ("normalization.pt", "shape_metadata.json")


def expand_parquet_files(input_dir: Path, split: str) -> list[Path]:
    """
    inputs:
      input_dir: Path, preprocessing output directory.
      split: str, split name such as train, val, test, or data.
    outputs:
      files: list[Path], concrete parquet files for the split.
    goal:
      Keep the monitor aligned with the sharded preprocessing directory layout.
    """
    if split == "train":
        patterns = ("train_*.parquet", "*.parquet")
        for pattern in patterns:
            files = sorted(input_dir.glob(pattern))
            files = [path for path in files if path.parent == input_dir]
            if files:
                return files
        return []
    return sorted((input_dir / split).glob("*.parquet"))


def load_class_labels(path: Path | None) -> list[str]:
    """
    inputs:
      path: Path | None, optional evenet_input_shards_manifest.json.
    outputs:
      labels: list[str], class labels if available.
    goal:
      Make class-monitor plots readable without requiring EveNet runtime imports.
    """
    if path is None:
        return []
    payload = json.loads(path.expanduser().read_text())
    labels = payload.get("class_labels", [])
    return [str(label) for label in labels]


def table_column_to_numpy(table: pa.Table, column: str) -> np.ndarray:
    """
    inputs:
      table: pa.Table, small table batch.
      column: str, column to extract.
    outputs:
      values: np.ndarray, column values.
    goal:
      Convert pyarrow arrays consistently for numeric monitoring.
    """
    return np.asarray(table[column].combine_chunks().to_numpy(zero_copy_only=False))


def sampled_column_values(path: Path, columns: list[str], max_rows: int, batch_size: int) -> dict[str, np.ndarray]:
    """
    inputs:
      path: Path, parquet file.
      columns: list[str], columns to sample.
      max_rows: int, maximum sampled rows.
      batch_size: int, pyarrow read batch size.
    outputs:
      values_by_column: dict[str, np.ndarray], sampled numeric values.
    goal:
      Inspect distributions without reading a full large shard into memory.
    """
    parquet = pq.ParquetFile(path)
    available = [column for column in columns if column in parquet.schema_arrow.names]
    values: dict[str, list[np.ndarray]] = {column: [] for column in available}
    rows_seen = 0

    for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        take = min(table.num_rows, max_rows - rows_seen)
        if take <= 0:
            break
        table = table.slice(0, take)
        for column in available:
            values[column].append(table_column_to_numpy(table, column))
        rows_seen += take
        if rows_seen >= max_rows:
            break

    return {
        column: np.concatenate(chunks) if chunks else np.array([])
        for column, chunks in values.items()
    }


def full_column_counts(path: Path, column: str, batch_size: int) -> Counter:
    """
    inputs:
      path: Path, parquet file.
      column: str, integer-like categorical column.
      batch_size: int, pyarrow read batch size.
    outputs:
      counts: Counter, value counts over the full file.
    goal:
      Compute exact class/source composition while still streaming.
    """
    counts: Counter = Counter()
    parquet = pq.ParquetFile(path)
    if column not in parquet.schema_arrow.names:
        return counts

    for batch in parquet.iter_batches(columns=[column], batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        values = table_column_to_numpy(table, column)
        valid = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else values
        counts.update(int(value) for value in valid)
    return counts


def auto_numeric_columns(schema_names: list[str], requested: list[str], max_extra: int) -> list[str]:
    """
    inputs:
      schema_names: list[str], parquet columns.
      requested: list[str], user-requested columns.
      max_extra: int, maximum auto-selected flattened feature columns.
    outputs:
      columns: list[str], numeric-like column names to sample.
    goal:
      Include common event-level columns plus a few flattened feature columns
      such as x:0:0 and conditions:0.
    """
    columns: list[str] = []
    for column in requested:
        if column in schema_names and column not in columns:
            columns.append(column)

    prefixes = ("x:", "conditions:", "x_invisible:")
    for column in schema_names:
        if len(columns) >= len(requested) + max_extra:
            break
        if column.startswith(prefixes) and column not in columns:
            columns.append(column)
    return columns


def class_name(index: int, class_labels: list[str]) -> str:
    """
    inputs:
      index: int, class index.
      class_labels: list[str], optional labels.
    outputs:
      label: str, human-readable class label.
    goal:
      Keep plots useful even when the manifest is unavailable.
    """
    if 0 <= index < len(class_labels):
        return class_labels[index]
    return f"class_{index}"


def plot_row_counts(split_summary: dict, output_path: Path) -> None:
    """
    inputs:
      split_summary: dict, per-file row counts.
      output_path: Path, plot destination.
    outputs:
      None.
    goal:
      Show obvious empty or wildly imbalanced parquet shards.
    """
    rows = [item["rows"] for item in split_summary["files"]]
    labels = [Path(item["path"]).stem for item in split_summary["files"]]
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(rows)), 4.5))
    ax.bar(np.arange(len(rows)), rows, color="#4477AA")
    ax.set_ylabel("Rows")
    ax.set_title(f"{split_summary['split']} rows per parquet")
    ax.set_xticks(np.arange(len(rows))[:: max(1, len(rows) // 20)])
    ax.set_xticklabels(labels[:: max(1, len(rows) // 20)], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_class_totals(class_counts: Counter, class_labels: list[str], output_path: Path) -> None:
    """
    inputs:
      class_counts: Counter, total class counts.
      class_labels: list[str], optional class labels.
      output_path: Path, plot destination.
    outputs:
      None.
    goal:
      Monitor global process balance after preprocessing.
    """
    if not class_counts:
        return
    indices = sorted(class_counts)
    labels = [class_name(index, class_labels) for index in indices]
    counts = [class_counts[index] for index in indices]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(indices)), 5))
    ax.bar(np.arange(len(indices)), counts, color="#228833")
    ax.set_ylabel("Events")
    ax.set_title("Class counts")
    ax.set_xticks(np.arange(len(indices)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_class_fraction_heatmap(file_class_counts: list[Counter], class_labels: list[str], output_path: Path) -> None:
    """
    inputs:
      file_class_counts: list[Counter], class counts for each train file.
      class_labels: list[str], optional class labels.
      output_path: Path, plot destination.
    outputs:
      None.
    goal:
      Check whether each train shard contains a healthy process mixture.
    """
    class_indices = sorted({index for counts in file_class_counts for index in counts})
    if not class_indices or not file_class_counts:
        return

    heatmap = np.zeros((len(class_indices), len(file_class_counts)), dtype=float)
    for file_index, counts in enumerate(file_class_counts):
        total = sum(counts.values())
        if total <= 0:
            continue
        for row_index, class_index in enumerate(class_indices):
            heatmap[row_index, file_index] = counts[class_index] / total

    fig, ax = plt.subplots(figsize=(max(8, 0.18 * len(file_class_counts)), max(4, 0.32 * len(class_indices))))
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, float(np.max(heatmap))))
    ax.set_xlabel("Train parquet index")
    ax.set_ylabel("Class")
    ax.set_title("Class fraction per train parquet")
    ax.set_yticks(np.arange(len(class_indices)))
    ax.set_yticklabels([class_name(index, class_labels) for index in class_indices])
    ax.set_xticks(np.arange(len(file_class_counts))[:: max(1, len(file_class_counts) // 20)])
    ax.set_xticklabels(np.arange(len(file_class_counts))[:: max(1, len(file_class_counts) // 20)])
    fig.colorbar(image, ax=ax, label="Fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_numeric_histograms(values_by_column: dict[str, list[np.ndarray]], output_dir: Path) -> dict[str, dict]:
    """
    inputs:
      values_by_column: dict[str, list[np.ndarray]], sampled values per column.
      output_dir: Path, histogram directory.
    outputs:
      summary: dict, finite/non-finite counts and ranges.
    goal:
      Catch corrupted flattened features, masks, or weights before training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for column, chunks in values_by_column.items():
        values = np.concatenate(chunks) if chunks else np.array([])
        finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else np.array([])
        summary[column] = {
            "sampled": int(len(values)),
            "finite": int(len(finite)),
            "nonfinite": int(len(values) - len(finite)),
            "min": float(np.min(finite)) if len(finite) else None,
            "max": float(np.max(finite)) if len(finite) else None,
            "mean": float(np.mean(finite)) if len(finite) else None,
        }
        if len(finite) == 0:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.hist(finite, bins=80, histtype="stepfilled", color="#4477AA", alpha=0.75)
        ax.set_xlabel(column)
        ax.set_ylabel("Sampled entries")
        ax.set_title(column.replace(":", " "))
        fig.tight_layout()
        safe_name = column.replace(":", "_").replace("/", "_")
        fig.savefig(output_dir / f"{safe_name}.png", dpi=160)
        plt.close(fig)
    return summary


def monitor_split(
    input_dir: Path,
    split: str,
    class_labels: list[str],
    output_dir: Path,
    monitor_columns: list[str],
    max_rows_per_file: int,
    batch_size: int,
) -> dict:
    """
    inputs:
      input_dir: Path, preprocessing output directory.
      split: str, split to monitor.
      class_labels: list[str], optional labels.
      output_dir: Path, destination for plots.
      monitor_columns: list[str], numeric columns requested by user.
      max_rows_per_file: int, sample cap per file for histograms.
      batch_size: int, pyarrow read batch size.
    outputs:
      summary: dict, rows, schema checks, class counts, and numeric summaries.
    goal:
      Validate one preprocessed parquet split without loading it all at once.
    """
    files = expand_parquet_files(input_dir, split)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"split": split, "files": [], "missing_required_columns": [], "schema_mismatches": []}
    if not files:
        summary["warning"] = f"No parquet files found for split={split}"
        return summary

    reference_schema = None
    total_class_counts: Counter = Counter()
    file_class_counts: list[Counter] = []
    numeric_samples: dict[str, list[np.ndarray]] = defaultdict(list)

    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        schema_names = parquet.schema_arrow.names
        rows = int(parquet.metadata.num_rows)
        summary["files"].append({"path": str(path), "rows": rows, "columns": len(schema_names)})

        if reference_schema is None:
            reference_schema = schema_names
        elif schema_names != reference_schema:
            summary["schema_mismatches"].append(str(path))

        if split == "train":
            missing = [column for column in REQUIRED_TRAIN_COLUMNS if column not in schema_names]
            if missing:
                summary["missing_required_columns"].append({"path": str(path), "missing": missing})

        columns = auto_numeric_columns(schema_names, monitor_columns, max_extra=8)
        sampled = sampled_column_values(path, columns, max_rows=max_rows_per_file, batch_size=batch_size)
        for column, values in sampled.items():
            numeric_samples[column].append(values)

        class_counts = full_column_counts(path, "classification", batch_size=batch_size)
        if class_counts:
            total_class_counts.update(class_counts)
            file_class_counts.append(class_counts)

        if file_index % 10 == 0:
            print(f"[monitor-preprocessed] {split} {file_index + 1}/{len(files)} rows={rows} {path}", flush=True)

    plot_row_counts(summary, split_dir / "rows_per_file.png")
    plot_class_totals(total_class_counts, class_labels, split_dir / "class_counts.png")
    if split == "train":
        plot_class_fraction_heatmap(file_class_counts, class_labels, split_dir / "class_fraction_per_file.png")
    summary["class_counts"] = {
        class_name(index, class_labels): int(total_class_counts[index])
        for index in sorted(total_class_counts)
    }
    summary["numeric"] = plot_numeric_histograms(numeric_samples, split_dir / "numeric_hists")
    summary["total_rows"] = int(sum(item["rows"] for item in summary["files"]))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor EveNet preprocessed parquet shards.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Preprocessed parquet directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for monitor plots and JSON.")
    parser.add_argument(
        "--shard-manifest",
        type=Path,
        default=None,
        help="Optional evenet_input_shards_manifest.json used for class labels.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test", "data"],
        choices=["train", "val", "test", "data"],
        help="Splits to monitor.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=list(DEFAULT_MONITOR_COLUMNS),
        help="Numeric columns to sample for histograms.",
    )
    parser.add_argument("--max-rows-per-file", type=int, default=20000, help="Histogram sample cap per parquet file.")
    parser.add_argument("--batch-size", type=int, default=50000, help="Pyarrow streaming batch size.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = load_class_labels(args.shard_manifest)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "class_labels": class_labels,
        "sidecars": {name: (input_dir / name).exists() for name in SIDECAR_FILES},
        "splits": {},
    }

    for split in args.splits:
        summary["splits"][split] = monitor_split(
            input_dir=input_dir,
            split=split,
            class_labels=class_labels,
            output_dir=output_dir,
            monitor_columns=args.columns,
            max_rows_per_file=args.max_rows_per_file,
            batch_size=args.batch_size,
        )

    summary_path = output_dir / "preprocessed_monitor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[monitor-preprocessed] wrote {summary_path}")


if __name__ == "__main__":
    main()
