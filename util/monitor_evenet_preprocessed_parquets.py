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
INVISIBLE_FEATURE_NAMES = ("energy", "pt", "eta", "phi")
FOUR_VECTOR_FEATURE_NAMES = ("energy", "pt", "eta", "phi", "mass")


def load_shape_metadata(input_dir: Path) -> dict:
    """
    inputs:
      input_dir: Path, preprocessing output directory.
    outputs:
      metadata: dict, shape_metadata.json payload or empty dict.
    goal:
      Reconstruct structured EveNet tensors from flattened parquet columns for
      monitoring plots that resemble the original ntuple-preparation monitors.
    """
    path = input_dir / "shape_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def flatten_column_name(base: str, indices: tuple[int, ...]) -> str:
    """
    inputs:
      base: str, original tensor name.
      indices: tuple[int, ...], flattened tensor indices.
    outputs:
      column: str, parquet column name produced by EveNet flatten_dict.
    goal:
      Keep parquet flattened-column reconstruction in one place.
    """
    return base + ":" + ":".join(str(index) for index in indices)


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


def sample_tables(files: list[Path], columns: list[str], max_rows: int, batch_size: int) -> list[pa.Table]:
    """
    inputs:
      files: list[Path], parquet files.
      columns: list[str], requested flattened columns.
      max_rows: int, total rows to sample over this split.
      batch_size: int, pyarrow read batch size.
    outputs:
      tables: list[pa.Table], sampled tables.
    goal:
      Build structured monitor plots from a bounded sample, independent of the
      number or size of parquet shards.
    """
    tables: list[pa.Table] = []
    rows_seen = 0
    for path in files:
        if rows_seen >= max_rows:
            break
        parquet = pq.ParquetFile(path)
        available = [column for column in columns if column in parquet.schema_arrow.names]
        if not available:
            continue
        for batch in parquet.iter_batches(columns=available, batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            take = min(table.num_rows, max_rows - rows_seen)
            if take <= 0:
                break
            tables.append(table.slice(0, take))
            rows_seen += take
            if rows_seen >= max_rows:
                break
    return tables


def concat_sampled_columns(tables: list[pa.Table], column: str) -> np.ndarray:
    """
    inputs:
      tables: list[pa.Table], sampled tables.
      column: str, column to concatenate.
    outputs:
      values: np.ndarray, sampled values or empty array.
    goal:
      Simplify structured monitor extraction from many sampled tables.
    """
    chunks = [table_column_to_numpy(table, column) for table in tables if column in table.column_names]
    return np.concatenate(chunks) if chunks else np.array([])


def collect_structured_monitor_columns(shape_metadata: dict, schema_names: list[str]) -> list[str]:
    """
    inputs:
      shape_metadata: dict, tensor shapes from preprocessing.
      schema_names: list[str], available parquet columns.
    outputs:
      columns: list[str], flattened columns needed for structured monitors.
    goal:
      Read only the columns needed to reconstruct x, masks, conditions, and
      invisible targets.
    """
    schema = set(schema_names)
    columns = ["classification", "event_weight", "num_vectors", "num_sequential_vectors"]
    for base in ("x_mask", "x_invisible_mask"):
        shape = tuple(shape_metadata.get(base, []))
        if len(shape) == 1:
            for index in range(shape[0]):
                columns.append(flatten_column_name(base, (index,)))

    for base in ("x", "x_invisible"):
        shape = tuple(shape_metadata.get(base, []))
        if len(shape) == 2:
            for slot in range(shape[0]):
                for feature in range(shape[1]):
                    columns.append(flatten_column_name(base, (slot, feature)))

    shape = tuple(shape_metadata.get("conditions", []))
    if len(shape) == 1:
        for feature in range(shape[0]):
            columns.append(flatten_column_name("conditions", (feature,)))

    return [column for column in dict.fromkeys(columns) if column in schema]


def finite_values(values: np.ndarray) -> np.ndarray:
    """
    inputs:
      values: np.ndarray, numeric values.
    outputs:
      values: np.ndarray, finite values only.
    goal:
      Keep plots readable and summarize invalid values separately elsewhere.
    """
    values = np.asarray(values)
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        return np.array([])
    return values[np.isfinite(values)]


def valid_flat_feature_values(
    tables: list[pa.Table],
    shape_metadata: dict,
    base: str,
    feature_index: int,
    mask_base: str | None,
) -> np.ndarray:
    """
    inputs:
      tables: list[pa.Table], sampled split tables.
      shape_metadata: dict, tensor shapes.
      base: str, tensor name such as x or x_invisible.
      feature_index: int, last-axis feature index.
      mask_base: str | None, optional mask tensor.
    outputs:
      values: np.ndarray, flattened valid feature values.
    goal:
      Plot EveNet tensor features after applying their masks, matching the
      original ntuple monitor spirit.
    """
    shape = tuple(shape_metadata.get(base, []))
    if len(shape) != 2 or feature_index >= shape[1]:
        return np.array([])
    values: list[np.ndarray] = []
    for table in tables:
        for slot in range(shape[0]):
            column = flatten_column_name(base, (slot, feature_index))
            if column not in table.column_names:
                continue
            slot_values = table_column_to_numpy(table, column)
            if mask_base is not None:
                mask_column = flatten_column_name(mask_base, (slot,))
                if mask_column in table.column_names:
                    mask = table_column_to_numpy(table, mask_column).astype(bool)
                    slot_values = slot_values[mask]
            values.append(slot_values)
    return finite_values(np.concatenate(values) if values else np.array([]))


def condition_feature_values(tables: list[pa.Table], shape_metadata: dict, feature_index: int) -> np.ndarray:
    """
    inputs:
      tables: list[pa.Table], sampled split tables.
      shape_metadata: dict, tensor shapes.
      feature_index: int, condition feature index.
    outputs:
      values: np.ndarray, finite condition feature values.
    goal:
      Plot global-condition features after preprocessing/log scaling.
    """
    shape = tuple(shape_metadata.get("conditions", []))
    if len(shape) != 1 or feature_index >= shape[0]:
        return np.array([])
    column = flatten_column_name("conditions", (feature_index,))
    return finite_values(np.concatenate([table_column_to_numpy(table, column) for table in tables if column in table.column_names]))


def mask_multiplicity(tables: list[pa.Table], shape_metadata: dict, mask_base: str) -> np.ndarray:
    """
    inputs:
      tables: list[pa.Table], sampled split tables.
      shape_metadata: dict, tensor shapes.
      mask_base: str, mask tensor name.
    outputs:
      counts: np.ndarray, number of valid slots per event.
    goal:
      Monitor valid object multiplicities after preprocessing.
    """
    shape = tuple(shape_metadata.get(mask_base, []))
    if len(shape) != 1:
        return np.array([])
    counts: list[np.ndarray] = []
    for table in tables:
        slot_masks = []
        for slot in range(shape[0]):
            column = flatten_column_name(mask_base, (slot,))
            if column in table.column_names:
                slot_masks.append(table_column_to_numpy(table, column).astype(bool))
        if slot_masks:
            counts.append(np.sum(np.stack(slot_masks, axis=1), axis=1))
    return np.concatenate(counts) if counts else np.array([])


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


def choose_overlay_bins(values_by_split: dict[str, np.ndarray], bins: int = 80) -> np.ndarray | None:
    """
    inputs:
      values_by_split: dict[str, np.ndarray], finite values per split.
      bins: int, number of bins.
    outputs:
      edges: np.ndarray | None, robust common bin edges.
    goal:
      Make split-overlay histograms directly comparable.
    """
    all_values = np.concatenate([values for values in values_by_split.values() if len(values)])
    all_values = finite_values(all_values)
    if len(all_values) == 0:
        return None
    low, high = np.percentile(all_values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.min(all_values)), float(np.max(all_values))
    if low == high:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, bins + 1)


def plot_split_overlay(
    values_by_split: dict[str, np.ndarray],
    output_path: Path,
    title: str,
    xlabel: str,
    *,
    bins: np.ndarray | None = None,
    density: bool = True,
    log_scale: bool = False,
) -> dict[str, int]:
    """
    inputs:
      values_by_split: dict[str, np.ndarray], finite values per split.
      output_path: Path, plot destination.
      title/xlabel: str, plot labels.
      bins: np.ndarray | None, optional shared bin edges.
      density/log_scale: bool, plotting controls.
    outputs:
      counts: dict[str, int], number of plotted entries per split.
    goal:
      Reproduce the original monitor style of comparing distributions across
      samples, here using train/val/test/data splits.
    """
    clean = {split: finite_values(values) for split, values in values_by_split.items()}
    clean = {split: values for split, values in clean.items() if len(values)}
    if not clean:
        return {}
    if bins is None:
        bins = choose_overlay_bins(clean)
    if bins is None:
        return {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for split, values in clean.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=1.7, density=density, label=f"{split} ({len(values)})")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Entries")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {split: int(len(values)) for split, values in clean.items()}


def plot_structured_preprocess_monitors(
    split_samples: dict[str, dict],
    shape_metadata: dict,
    output_dir: Path,
    max_x_features: int,
    max_condition_features: int,
) -> dict:
    """
    inputs:
      split_samples: dict[str, dict], sampled split tables and summaries.
      shape_metadata: dict, tensor shapes.
      output_dir: Path, output directory.
      max_x_features: int, maximum point-cloud features to draw.
      max_condition_features: int, maximum condition features to draw.
    outputs:
      summary: dict, plotted-entry counts.
    goal:
      Produce monitor plots that look at the actual preprocessed EveNet tensors,
      not just raw flattened parquet columns.
    """
    structured_dir = output_dir / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for mask_base, title in (("x_mask", "valid sequential vectors"), ("x_invisible_mask", "valid invisible targets")):
        values_by_split = {
            split: mask_multiplicity(payload["tables"], shape_metadata, mask_base)
            for split, payload in split_samples.items()
        }
        max_count = max((int(np.max(values)) for values in values_by_split.values() if len(values)), default=0)
        bins = np.arange(-0.5, max_count + 1.5, 1.0)
        summary[mask_base] = plot_split_overlay(
            values_by_split,
            structured_dir / f"{mask_base}_multiplicity.png",
            title,
            "valid slots per event",
            bins=bins,
            density=False,
        )

    for column in ("num_vectors", "num_sequential_vectors", "event_weight"):
        values_by_split = {
            split: concat_sampled_columns(payload["tables"], column)
            for split, payload in split_samples.items()
        }
        summary[column] = plot_split_overlay(
            values_by_split,
            structured_dir / f"{column}.png",
            column,
            column,
            density=column == "event_weight",
            log_scale=column == "event_weight",
        )

    x_shape = tuple(shape_metadata.get("x", []))
    if len(x_shape) == 2:
        for feature_index in range(min(x_shape[1], max_x_features)):
            label = FOUR_VECTOR_FEATURE_NAMES[feature_index] if feature_index < len(FOUR_VECTOR_FEATURE_NAMES) else f"feature_{feature_index}"
            values_by_split = {
                split: valid_flat_feature_values(payload["tables"], shape_metadata, "x", feature_index, "x_mask")
                for split, payload in split_samples.items()
            }
            summary[f"x_{feature_index}"] = plot_split_overlay(
                values_by_split,
                structured_dir / "x_features" / f"x_feature_{feature_index:02d}_{label}.png",
                f"x feature {feature_index}: {label}",
                f"x[{feature_index}] {label}",
                log_scale=feature_index in {0, 1},
            )

    invisible_shape = tuple(shape_metadata.get("x_invisible", []))
    if len(invisible_shape) == 2:
        for feature_index in range(invisible_shape[1]):
            label = INVISIBLE_FEATURE_NAMES[feature_index] if feature_index < len(INVISIBLE_FEATURE_NAMES) else f"feature_{feature_index}"
            values_by_split = {
                split: valid_flat_feature_values(payload["tables"], shape_metadata, "x_invisible", feature_index, "x_invisible_mask")
                for split, payload in split_samples.items()
            }
            summary[f"x_invisible_{feature_index}"] = plot_split_overlay(
                values_by_split,
                structured_dir / "x_invisible_features" / f"x_invisible_{feature_index:02d}_{label}.png",
                f"x_invisible feature {feature_index}: {label}",
                f"x_invisible[{feature_index}] {label}",
                log_scale=feature_index in {0, 1},
            )

    condition_shape = tuple(shape_metadata.get("conditions", []))
    if len(condition_shape) == 1:
        for feature_index in range(min(condition_shape[0], max_condition_features)):
            values_by_split = {
                split: condition_feature_values(payload["tables"], shape_metadata, feature_index)
                for split, payload in split_samples.items()
            }
            summary[f"conditions_{feature_index}"] = plot_split_overlay(
                values_by_split,
                structured_dir / "conditions" / f"condition_{feature_index:02d}.png",
                f"condition feature {feature_index}",
                f"conditions[{feature_index}]",
            )

    return summary


def monitor_split(
    input_dir: Path,
    split: str,
    class_labels: list[str],
    shape_metadata: dict,
    output_dir: Path,
    monitor_columns: list[str],
    max_rows_per_file: int,
    max_rows_per_split: int,
    batch_size: int,
) -> dict:
    """
    inputs:
      input_dir: Path, preprocessing output directory.
      split: str, split to monitor.
      class_labels: list[str], optional labels.
      shape_metadata: dict, preprocessing tensor shapes.
      output_dir: Path, destination for plots.
      monitor_columns: list[str], numeric columns requested by user.
      max_rows_per_file: int, sample cap per file for histograms.
      max_rows_per_split: int, sample cap per split for structured plots.
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
    split_sample_tables: list[pa.Table] = []

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

    if reference_schema is not None and shape_metadata:
        structured_columns = collect_structured_monitor_columns(shape_metadata, reference_schema)
        split_sample_tables = sample_tables(files, structured_columns, max_rows=max_rows_per_split, batch_size=batch_size)

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
    summary["structured_sample_rows"] = int(sum(table.num_rows for table in split_sample_tables))
    summary["_sample_tables"] = split_sample_tables
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
    parser.add_argument("--max-rows-per-split", type=int, default=200000, help="Structured monitor sample cap per split.")
    parser.add_argument("--max-x-features", type=int, default=12, help="Maximum x feature indices to plot.")
    parser.add_argument("--max-condition-features", type=int, default=16, help="Maximum condition feature indices to plot.")
    parser.add_argument("--batch-size", type=int, default=50000, help="Pyarrow streaming batch size.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class_labels = load_class_labels(args.shard_manifest)
    shape_metadata = load_shape_metadata(input_dir)
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "class_labels": class_labels,
        "shape_metadata_available": bool(shape_metadata),
        "sidecars": {name: (input_dir / name).exists() for name in SIDECAR_FILES},
        "splits": {},
    }

    split_samples: dict[str, dict] = {}
    for split in args.splits:
        split_summary = monitor_split(
            input_dir=input_dir,
            split=split,
            class_labels=class_labels,
            shape_metadata=shape_metadata,
            output_dir=output_dir,
            monitor_columns=args.columns,
            max_rows_per_file=args.max_rows_per_file,
            max_rows_per_split=args.max_rows_per_split,
            batch_size=args.batch_size,
        )
        split_samples[split] = {"tables": split_summary.pop("_sample_tables", [])}
        summary["splits"][split] = split_summary

    if shape_metadata:
        summary["structured"] = plot_structured_preprocess_monitors(
            split_samples,
            shape_metadata,
            output_dir,
            max_x_features=args.max_x_features,
            max_condition_features=args.max_condition_features,
        )

    summary_path = output_dir / "preprocessed_monitor_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[monitor-preprocessed] wrote {summary_path}")


if __name__ == "__main__":
    main()
