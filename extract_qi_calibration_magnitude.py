#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from plot_style import channel_latex_label, method_color
except Exception:
    def channel_latex_label(channel: str) -> str:
        if channel.startswith("Ztautau_"):
            return channel.removeprefix("Ztautau_").replace("_", r"\_")
        return channel.replace("_", r"\_")

    _FALLBACK_COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    def method_color(method: str, index: int) -> str:
        return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


DEFAULT_ANALYSIS_CONFIG = Path("ml_pipeline/config/analysis.yaml")
CALIBRATION_FIELD = "calibration_deltaR_sum"
DEFAULT_SIGNAL_SAMPLE = "Ztautau"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the exported post-calibration direction-change magnitude "
            "(deltaR_a + deltaR_b) per selected channel and method."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help=(
            "Method spec Label:/path/to/exported/method or Label:/path/to/exported/method/processed. "
            "Repeat for each method to compare."
        ),
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--sample-name", default=DEFAULT_SIGNAL_SAMPLE)
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--weight-field", default="weight")
    parser.add_argument("--load-batch-size", type=int, default=100_000)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_method_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    methods: list[tuple[str, Path]] = []
    for spec in specs:
        name, separator, raw_path = spec.partition(":")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"Invalid --method '{spec}'. Expected Label:/path/to/exported/method")
        methods.append((name.strip(), Path(raw_path).expanduser().resolve()))
    return methods


def processed_dir(path: Path) -> Path:
    if path.name == "processed":
        return path
    if (path / "processed").is_dir():
        return path / "processed"
    return path


def read_analysis_channels(path: Path) -> list[str]:
    try:
        import yaml
    except ModuleNotFoundError:
        return []

    if not path.exists():
        return []
    config = yaml.safe_load(path.read_text()) or {}
    prediction_cfg = config.get("NeutrinoPrediction") or {}
    channels: list[str] = []
    seen: set[str] = set()
    for value in prediction_cfg.values():
        items = value if isinstance(value, list) else value.values() if isinstance(value, dict) else [value]
        for item in items:
            channel = str(item)
            if channel.startswith("Ztautau_") and channel not in seen:
                channels.append(channel)
                seen.add(channel)
    return channels


def parquet_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return {field.name for field in pq.ParquetFile(path).schema_arrow}


def iter_parquet_batches(paths: list[Path], requested_columns: set[str], batch_size: int):
    import awkward as ak
    import pyarrow.parquet as pq

    for path in paths:
        available = parquet_columns(path)
        columns = sorted(requested_columns & available)
        if not columns:
            continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            yield ak.from_arrow(batch)


def to_numpy(values: Any, dtype: Any) -> np.ndarray:
    import awkward as ak

    return np.asarray(ak.to_numpy(values, allow_missing=False), dtype=dtype)


def calibration_region_path(processed_path: Path, sample_name: str, channel: str) -> Path:
    return processed_path / sample_name / f"filtered___{channel}.parquet"


def summarize_channel(
    processed_path: Path,
    sample_name: str,
    channel: str,
    weight_field: str,
    batch_size: int,
) -> dict[str, Any]:
    parquet_path = calibration_region_path(processed_path, sample_name, channel)
    if not parquet_path.is_file():
        return {
            "channel": channel,
            "mean_calibration_deltaR_sum": float("nan"),
            "num_events": 0,
            "num_finite_events": 0,
        }

    weighted_sum = 0.0
    weight_sum = 0.0
    num_events = 0
    num_finite_events = 0
    for events in iter_parquet_batches([parquet_path], {CALIBRATION_FIELD, weight_field}, batch_size):
        calibration = to_numpy(events[CALIBRATION_FIELD], np.float64) if CALIBRATION_FIELD in events.fields else np.full(len(events), np.nan)
        weights = to_numpy(events[weight_field], np.float64) if weight_field in events.fields else np.ones(len(events), dtype=np.float64)
        num_events += len(events)
        finite = np.isfinite(calibration) & np.isfinite(weights) & (weights > 0.0)
        num_finite_events += int(np.sum(finite))
        if np.any(finite):
            weighted_sum += float(np.sum(calibration[finite] * weights[finite]))
            weight_sum += float(np.sum(weights[finite]))

    mean_value = weighted_sum / weight_sum if weight_sum > 0.0 else float("nan")
    return {
        "channel": channel,
        "mean_calibration_deltaR_sum": mean_value,
        "num_events": int(num_events),
        "num_finite_events": int(num_finite_events),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]], channels: list[str], output_prefix: Path) -> dict[str, Any]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    method_to_index = {method: index for index, method in enumerate(methods)}
    x = np.arange(len(channels), dtype=np.float64)
    group_width = min(0.82, 0.18 * len(methods) + 0.20)
    bar_width = group_width / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(12.0, 1.0 * len(channels) + 3.5), 6.8), dpi=200)
    ymax = 0.0
    for method_index, method in enumerate(methods):
        method_rows = {row["channel"]: row for row in rows if row["method"] == method}
        values = np.array(
            [method_rows.get(channel, {}).get("mean_calibration_deltaR_sum", np.nan) for channel in channels],
            dtype=np.float64,
        )
        ymax = max(ymax, float(np.nanmax(values)) if np.any(np.isfinite(values)) else 0.0)
        offset = x - group_width / 2.0 + (method_index + 0.5) * bar_width
        finite = np.isfinite(values)
        ax.bar(
            offset[finite],
            values[finite],
            width=bar_width * 0.9,
            color=method_color(method, method_index),
            edgecolor="none",
            label=method,
            alpha=0.90,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([channel_latex_label(channel) for channel in channels], rotation=30, ha="right")
    ax.set_ylabel(r"Weighted mean of $\Delta R_a + \Delta R_b$")
    ax.set_xlabel("Selected channel")
    ax.set_title("Post-calibration direction change magnitude")
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.set_ylim(0.0, max(0.02, ymax) * 1.18)
    ax.legend(frameon=False, loc="upper right", ncol=max(1, min(4, len(methods))))
    fig.tight_layout()

    plot_path = output_prefix.parent / f"{output_prefix.name}.png"
    fig.savefig(plot_path)
    fig.savefig(plot_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"plot": str(plot_path), "methods": methods, "channels": channels}


def main() -> None:
    args = parse_args()
    method_specs = parse_method_specs(args.method)
    channels = args.channels if args.channels is not None else read_analysis_channels(args.analysis_config)
    if not channels:
        raise ValueError("No channels were provided or discovered from the analysis config.")

    rows: list[dict[str, Any]] = []
    for method, raw_path in method_specs:
        method_processed_dir = processed_dir(raw_path)
        if not method_processed_dir.is_dir():
            raise FileNotFoundError(f"Cannot find processed directory for method '{method}': {raw_path}")
        sample_dir = method_processed_dir / args.sample_name
        if not sample_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find sample directory '{args.sample_name}' for method '{method}' under {method_processed_dir}"
            )
        for channel in channels:
            summary = summarize_channel(
                method_processed_dir,
                args.sample_name,
                channel,
                args.weight_field,
                args.load_batch_size,
            )
            summary["method"] = method
            rows.append(summary)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.parent / f"{args.output_prefix.name}.json"
    csv_path = args.output_prefix.parent / f"{args.output_prefix.name}.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_csv(rows, csv_path)

    if not args.no_plots:
        plot_info = plot_summary(rows, channels, args.output_prefix)
        plot_json_path = args.output_prefix.parent / f"{args.output_prefix.name}_plots.json"
        plot_json_path.write_text(json.dumps(plot_info, indent=2, sort_keys=True) + "\n")
        print(f"[qi-calibration] wrote_plot_summary={plot_json_path}", flush=True)

    print(f"[qi-calibration] wrote_json={json_path}", flush=True)
    print(f"[qi-calibration] wrote_csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
