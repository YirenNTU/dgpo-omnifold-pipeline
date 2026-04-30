#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import channel_latex_label, method_color


RESULT_LINE_ASYMMETRIC_RE = re.compile(
    r"^\s*(?P<name>[^:]+):\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"\+(?P<err_up>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"/-(?P<err_down>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
RESULT_LINE_SYMMETRIC_RE = re.compile(
    r"^\s*(?P<name>[^:=]+)\s*(?::|=)\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?:±|\+/-)\s*"
    r"(?P<err>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)
SECTION_RE = re.compile(
    r"(?:(?P<source>Unfolded|Truth|Final|Nominal)\s+)?"
    r"(?P<group>B and C matrices|BC matrices|Quantum results|Final metrics|Metrics):$",
    re.IGNORECASE,
)
METHOD_MARKERS = ("o", "D", "^", "s", "P", "X")
DEFAULT_IGNORED_REGIONS = {"hadhad"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract final unfolded QI measurements from central QIProcessor results.txt."
    )
    parser.add_argument(
        "--results-txt",
        type=Path,
        default=None,
        help="Path to QI_analysis/results.txt produced by central QIProcessor.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        help=(
            "Optional method spec in the form Label:/path/to/results.txt. Repeat for Baseline/Pretrain/Scratch. "
            "When provided, --results-txt is ignored."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="Output path prefix. Writes <prefix>.json and <prefix>.csv.",
    )
    parser.add_argument(
        "--keep-truth",
        action="store_true",
        help="Also keep Truth sections. By default only Unfolded measurements are written.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write JSON/CSV tables; skip summary plots.",
    )
    parser.add_argument(
        "--keep-hadhad",
        action="store_true",
        help="Keep broad hadhad rows. By default they are ignored in favor of fine channels such as pipi/pirho/rhopi.",
    )
    return parser.parse_args()


def parse_method_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """
    inputs:
      args: argparse.Namespace, parsed command-line options.
    outputs:
      list[(method_label, results_path)], one entry per QIProcessor result text.
    goal:
      Support both the legacy single-file mode and multi-method comparison mode.
    """
    if args.method:
        methods: list[tuple[str, Path]] = []
        for spec in args.method:
            if ":" not in spec:
                raise ValueError(f"Invalid --method '{spec}'. Expected Label:/path/to/results.txt.")
            label, path = spec.split(":", 1)
            methods.append((label.strip(), Path(path)))
        return methods
    if args.results_txt is None:
        raise ValueError("Provide either --results-txt or one or more --method Label:/path/to/results.txt options.")
    return [("Measurement", args.results_txt)]


def parse_results_text(
    text: str,
    method: str,
    keep_truth: bool = False,
    ignored_regions: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    inputs:
      text: str, content of central QIProcessor results.txt.
      method: str, user-facing method label for this result file.
      keep_truth: bool, include Truth sections in addition to Unfolded sections.
      ignored_regions: set[str] | None, broad regions to drop from final tables/plots.
    outputs:
      list[dict], one row per final B/C or quantum-result value.
    goal:
      Convert central human-readable final measurements into stable JSON/CSV rows.
    """
    rows: list[dict[str, Any]] = []
    ignored_regions = ignored_regions or set()
    region: str | None = None
    signal: str | None = None
    section_source: str | None = None
    section_group: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Region: "):
            region = line.removeprefix("Region: ").strip()
            signal = None
            section_source = None
            section_group = None
            continue

        match = re.match(r"Unfolding results for signal (?P<signal>.+?) in region (?P<region>.+):$", line)
        if match:
            signal = match.group("signal").strip()
            region = match.group("region").strip()
            section_source = None
            section_group = None
            continue

        match = SECTION_RE.match(line)
        if match:
            section_source = canonical_source_name(match.group("source") or "Unfolded")
            section_group = canonical_group_name(match.group("group"))
            continue

        parsed_value = parse_measurement_line(raw_line)
        if parsed_value is None or region is None or signal is None or section_source is None or section_group is None:
            continue
        if section_source == "Truth" and not keep_truth:
            continue
        if region in ignored_regions:
            continue

        rows.append(
            {
                "region": region,
                "signal": signal,
                "method": method,
                "source": section_source,
                "group": section_group,
                "parameter": parsed_value["name"],
                "value": parsed_value["value"],
                "err_up": parsed_value["err_up"],
                "err_down": parsed_value["err_down"],
            }
        )

    return rows


def canonical_source_name(source: str) -> str:
    """
    inputs:
      source: str, source label printed by QIProcessor.
    outputs:
      str, stable source label stored in JSON/CSV.
    goal:
      Keep the extractor compatible with nominal output variants such as
      "Final" or "Nominal" while preserving the historical "Unfolded" label.
    """
    normalized = source.strip().lower()
    if normalized in {"final", "nominal"}:
        return "Unfolded"
    if normalized == "truth":
        return "Truth"
    return "Unfolded"


def canonical_group_name(group: str) -> str:
    """
    inputs:
      group: str, results section label printed by QIProcessor.
    outputs:
      str, compact group name used by output tables and plot filenames.
    goal:
      Map updated nominal final-metric headings onto the existing BC/quantum
      grouping without hard-coding every downstream plot path.
    """
    normalized = group.strip().lower()
    if "b and c" in normalized or normalized.startswith("bc"):
        return "BC"
    return "quantum"


def parse_measurement_line(raw_line: str) -> dict[str, float | str] | None:
    """
    inputs:
      raw_line: str, one candidate measurement line from results.txt.
    outputs:
      dict with name/value/err_up/err_down, or None if the line is not a measurement.
    goal:
      Accept both historical asymmetric "x +a/-b" lines and newer nominal
      symmetric "x ± a" / "x +/- a" final-metric lines.
    """
    match = RESULT_LINE_ASYMMETRIC_RE.match(raw_line)
    if match:
        return {
            "name": match.group("name").strip(),
            "value": float(match.group("value")),
            "err_up": float(match.group("err_up")),
            "err_down": float(match.group("err_down")),
        }

    match = RESULT_LINE_SYMMETRIC_RE.match(raw_line)
    if match:
        err = float(match.group("err"))
        return {
            "name": match.group("name").strip(),
            "value": float(match.group("value")),
            "err_up": err,
            "err_down": err,
        }
    return None


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["method", "region", "signal", "source", "group", "parameter", "value", "err_up", "err_down"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return safe.strip("_") or "measurement"


def method_marker(method: str, method_index: int) -> str:
    return METHOD_MARKERS[method_index % len(METHOD_MARKERS)]


def channel_label(region: str, signal: str) -> str:
    return channel_latex_label(canonical_channel_key(region, signal))


def canonical_channel_key(region: str, signal: str) -> str:
    """
    inputs:
      region: str, QIProcessor region name from results.txt.
      signal: str, QIProcessor signal name from results.txt.
    outputs:
      str, canonical channel key used for y-axis grouping.
    goal:
      Put baseline bare channels such as pipi/pirho/rhopi on the same y-axis row
      as EveNet channels named Ztautau_pipi/Ztautau_pirho/Ztautau_rhopi.
    """
    raw = signal if signal.startswith("Ztautau_") else region
    if raw.startswith("Ztautau_"):
        channel = raw.removeprefix("Ztautau_")
    else:
        channel = raw
    if channel in {"pipi", "pirho", "rhopi", "rhorho", "ee", "emu", "mue", "mumu"}:
        return f"Ztautau_{channel}"
    return region


def parameter_label(parameter: str) -> str:
    if parameter.startswith("C_") and len(parameter) == 4:
        return rf"$C_{{{parameter[-2]}{parameter[-1]}}}$"
    if parameter.startswith("B_") and len(parameter) == 4:
        return rf"$B_{{{parameter[-2]},{parameter[-1]}}}$"
    if parameter == "Concurrence":
        return "Concurrence"
    match = re.match(r"C_?(?P<a>[a-z]{2}) (?P<op>[+-]) C_?(?P<b>[a-z]{2})$", parameter)
    if match:
        return rf"$C_{{{match.group('a')}}} {match.group('op')} C_{{{match.group('b')}}}$"
    return parameter.replace("_", " ")


def format_value_unc(row: dict[str, Any]) -> str:
    value = row["value"]
    err_up = row["err_up"]
    err_down = row["err_down"]
    if math.isclose(err_up, err_down, rel_tol=0.05, abs_tol=5.0e-4):
        return f"{value:.3f} ± {0.5 * (err_up + err_down):.3f}"
    return f"{value:.3f} +{err_up:.3f}/-{err_down:.3f}"


def plot_measurement_summaries(rows: list[dict[str, Any]], output_prefix: Path) -> dict[str, Any]:
    """
    inputs:
      rows: list[dict], parsed final measurements with method/region/signal/value/uncertainty.
      output_prefix: Path, prefix used to create the summary-plot directory.
    outputs:
      dict, plot metadata keyed by measurement parameter.
    goal:
      Render final B/C and quantum measurements with the same compact summary
      style used by pre-unfolding validation plots.
    """
    plot_dir = output_prefix.parent / f"{output_prefix.name}_summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(row["method"] for row in rows))
    method_index = {method: index for index, method in enumerate(methods)}
    groups = list(dict.fromkeys(row["group"] for row in rows))
    plot_summary: dict[str, Any] = {}

    for group in groups:
        group_rows = [row for row in rows if row["group"] == group]
        parameters = list(dict.fromkeys(row["parameter"] for row in group_rows))
        for parameter in parameters:
            parameter_rows = [row for row in group_rows if row["parameter"] == parameter]
            channel_keys = list(dict.fromkeys(canonical_channel_key(row["region"], row["signal"]) for row in parameter_rows))
            y_base = np.arange(len(channel_keys), dtype=np.float64)
            channel_index = {key: index for index, key in enumerate(channel_keys)}

            values = np.array([row["value"] for row in parameter_rows], dtype=np.float64)
            err_up = np.array([row["err_up"] for row in parameter_rows], dtype=np.float64)
            err_down = np.array([row["err_down"] for row in parameter_rows], dtype=np.float64)
            xmin = float(np.nanmin(values - err_down))
            xmax = float(np.nanmax(values + err_up))
            span = xmax - xmin
            pad = max(0.12 * span, 0.02)

            fig_height = max(4.0, 0.58 * len(channel_keys) + 1.8)
            fig, ax = plt.subplots(figsize=(10.6, fig_height), dpi=200)
            ax.set_xlim(xmin - pad, xmax + pad)

            x_text_value = 1.03
            for key in channel_keys:
                channel_rows = [
                    row
                    for row in parameter_rows
                    if canonical_channel_key(row["region"], row["signal"]) == key
                ]
                channel_rows.sort(key=lambda row: method_index[row["method"]])
                offsets = np.linspace(-0.24, 0.24, len(channel_rows)) if len(channel_rows) > 1 else np.array([0.0])
                for offset, row in zip(offsets, channel_rows):
                    index = method_index[row["method"]]
                    y = y_base[channel_index[key]] + offset
                    color = method_color(row["method"], index)
                    ax.errorbar(
                        row["value"],
                        y,
                        xerr=np.array([[row["err_down"]], [row["err_up"]]]),
                        fmt=method_marker(row["method"], index),
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        capsize=2.5,
                        markersize=6.5,
                        lw=1.2,
                    )
                    ax.text(
                        x_text_value,
                        y,
                        format_value_unc(row),
                        color=color,
                        fontsize=8,
                        va="center",
                        ha="left",
                        transform=ax.get_yaxis_transform(),
                        clip_on=False,
                    )

            ax.text(x_text_value, 1.02, "value ± unc.", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
            ax.set_yticks(y_base)
            ax.set_yticklabels([channel_label(channel, channel) for channel in channel_keys])
            ax.invert_yaxis()
            ax.grid(axis="y", alpha=0.18, linestyle=":")
            for separator in np.arange(len(channel_keys) - 1, dtype=np.float64) + 0.5:
                ax.axhline(separator, color="#D9D9D9", linewidth=0.8, zorder=0)
            ax.set_xlabel(parameter_label(parameter))
            ax.set_ylabel("Channel / Region")
            ax.set_title(f"{parameter_label(parameter)} final measurement")

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker=method_marker(method, index),
                    color=method_color(method, index),
                    markerfacecolor=method_color(method, index),
                    markeredgecolor=method_color(method, index),
                    markersize=7,
                    linestyle="None",
                    label=method,
                )
                for method, index in method_index.items()
            ]
            ax.legend(
                handles=handles,
                title="Methods",
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.16),
                ncol=min(len(handles), 4),
            )
            fig.subplots_adjust(right=0.78, top=0.82, left=0.16, bottom=0.16)

            plot_path = plot_dir / f"{sanitize_filename(group)}_{sanitize_filename(parameter)}.png"
            fig.savefig(plot_path)
            plt.close(fig)
            plot_summary[f"{group}:{parameter}"] = {
                "plot": str(plot_path),
                "num_points": len(parameter_rows),
                "methods": methods,
            }
            print(f"[extract-qi-final] wrote_plot={plot_path}", flush=True)

    return plot_summary


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    ignored_regions = set() if args.keep_hadhad else DEFAULT_IGNORED_REGIONS
    for method, results_path in parse_method_specs(args):
        rows.extend(
            parse_results_text(
                results_path.read_text(),
                method=method,
                keep_truth=args.keep_truth,
                ignored_regions=ignored_regions,
            )
        )
    if not rows:
        raise ValueError(
            "No final measurements found. "
            "Check that QIProcessor finished and wrote Unfolded B/C or Quantum result sections."
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_csv(rows, csv_path)
    if not args.no_plots:
        plot_summary = plot_measurement_summaries(rows, args.output_prefix)
        plot_json_path = args.output_prefix.parent / f"{args.output_prefix.name}_summary_plots.json"
        plot_json_path.write_text(json.dumps(plot_summary, indent=2, sort_keys=True) + "\n")
        print(f"[extract-qi-final] wrote_plot_summary={plot_json_path}")

    print(f"[extract-qi-final] rows={len(rows)}")
    print(f"[extract-qi-final] wrote_json={json_path}")
    print(f"[extract-qi-final] wrote_csv={csv_path}")


if __name__ == "__main__":
    main()
