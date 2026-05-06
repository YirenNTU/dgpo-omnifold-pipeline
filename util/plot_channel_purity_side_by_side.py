#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np

from plot_style import OKABE_ITO, channel_latex_label, method_color, process_color, process_latex_label


DATA_COLOR = OKABE_ITO["black"]
BACKGROUND_COLOR = "#D8D8D8"
DEFAULT_CLASS_NAME = "unselected"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "channel_purity_side_by_side.png"
OPENXML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

BASELINE_PROCESS_COLUMN_MAP = {
    "C": "Zqq",
    "D": "Zll",
    "E": "Ztautau_pipi",
    "F": "Ztautau_pirho",
    "G": "Ztautau_rhopi",
    "H": "Ztautau_rhorho",
    "I": "Ztautau_pie",
    "J": "Ztautau_epi",
    "K": "Ztautau_pimu",
    "L": "Ztautau_mupi",
    "M": "Ztautau_rhoe",
    "N": "Ztautau_erho",
    "O": "Ztautau_rhomu",
    "P": "Ztautau_murho",
    "Q": "Ztautau_ee",
    "R": "Ztautau_mumu",
    "S": "Ztautau_emu",
    "T": "Ztautau_mue",
    "U": "Other",
}

CHANNEL_ALIASES = {
    "pi_el": "pie",
    "el_pi": "epi",
    "pi_mu": "pimu",
    "mu_pi": "mupi",
    "rho_el": "rhoe",
    "el_rho": "erho",
    "rho_mu": "rhomu",
    "mu_rho": "murho",
}

SIGNAL_CHANNEL_KEYS = {
    "pipi",
    "pirho",
    "rhopi",
    "rhorho",
    "pie",
    "epi",
    "pimu",
    "mupi",
    "rhoe",
    "erho",
    "rhomu",
    "murho",
    "ee",
    "mumu",
    "emu",
    "mue",
}

STACK_PRIORITY = {
    "Zqq": 0,
    "Zll": 1,
    "Ztautau_pipi": 2,
    "Ztautau_pirho": 3,
    "Ztautau_rhopi": 4,
    "Ztautau_rhorho": 5,
    "Ztautau_pie": 6,
    "Ztautau_epi": 7,
    "Ztautau_pimu": 8,
    "Ztautau_mupi": 9,
    "Ztautau_rhoe": 10,
    "Ztautau_erho": 11,
    "Ztautau_rhomu": 12,
    "Ztautau_murho": 13,
    "Ztautau_ee": 14,
    "Ztautau_mumu": 15,
    "Ztautau_emu": 16,
    "Ztautau_mue": 17,
    "Other": 18,
}

BASELINE_CHANNEL_ORDER = [
    "zee",
    "zmumu",
    "ee",
    "mumu",
    "emu",
    "mue",
    "pipi",
    "pirho",
    "rhopi",
    "pie",
    "epi",
    "pimu",
    "mupi",
    "rhoe",
    "erho",
    "rhomu",
    "murho",
    "baseline",
]


@dataclass
class MethodPlotData:
    name: str
    channel_order: list[str]
    stack_matrix: dict[str, dict[str, float]]
    total_mc: dict[str, float]
    data_yield: dict[str, float]
    purity: dict[str, float]
    data_over_mc: dict[str, float]
    is_baseline: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and prediction channel-purity yields side by side."
    )
    parser.add_argument(
        "--baseline-xlsx",
        type=Path,
        required=True,
        help="Baseline yield workbook, e.g. baseline_yield.xlsx.",
    )
    parser.add_argument(
        "--prediction-method",
        action="append",
        default=[],
        metavar="NAME:MC_PATH[:DATA_PATH]",
        help=(
            "Prediction method definition. MC_PATH and optional DATA_PATH can be parquet files, "
            "directories, or glob patterns."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="Optional explicit channel order using canonical names such as pipi, emu, pie.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output figure path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional sidecar JSON summary. Defaults to <output>.json.",
    )
    parser.add_argument(
        "--title",
        default="Predicted Channel Purity Comparison",
        help="Figure title.",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[str]:
    resolved: list[str] = []
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        if expanded.is_dir():
            final_prediction_paths = sorted(expanded.glob("*__evenet_pred.parquet"))
            paths = final_prediction_paths if final_prediction_paths else sorted(expanded.glob("*.parquet"))
            resolved.extend(str(path.resolve()) for path in paths)
            continue
        matches = sorted(glob.glob(str(expanded)))
        if matches:
            resolved.extend(str(Path(match).resolve()) for match in matches)
        else:
            resolved.append(str(expanded.resolve()))
    return resolved


def load_events(paths: list[str]) -> ak.Array:
    import awkward as ak

    arrays = [ak.from_parquet(path) for path in paths]
    if not arrays:
        raise ValueError("No parquet inputs found.")
    return arrays[0] if len(arrays) == 1 else ak.concatenate(arrays, axis=0)


def event_weights(events: ak.Array) -> np.ndarray:
    if "evenet_weight" not in events.fields:
        raise ValueError("Prediction comparison requires evenet_weight in prediction parquet.")
    weights = np.asarray(ak.to_numpy(events["evenet_weight"]), dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0)
    return np.where(valid, weights, 0.0)


def canonical_channel_name(name: str) -> str | None:
    text = str(name).strip()
    if not text or text == DEFAULT_CLASS_NAME:
        return None
    lowered = text.lower()
    lowered = CHANNEL_ALIASES.get(lowered, lowered)
    if lowered.startswith("ztautau_"):
        lowered = lowered.removeprefix("ztautau_")
    if lowered in {"zqq", "zll"}:
        return lowered
    return lowered


def canonical_process_name(name: str) -> str | None:
    text = str(name).strip()
    if not text or text == DEFAULT_CLASS_NAME:
        return None
    lowered = text.lower()
    lowered = CHANNEL_ALIASES.get(lowered, lowered)
    if lowered in {"zqq", "z→qq"}:
        return "Zqq"
    if lowered in {"zll", "z→ℓℓ", "z→ll"}:
        return "Zll"
    if lowered in {"other", "other bkg", "other_bkg"}:
        return "Other"
    if lowered.startswith("ztautau_"):
        channel = lowered.removeprefix("ztautau_")
        if channel in SIGNAL_CHANNEL_KEYS:
            return f"Ztautau_{channel}"
    if lowered in SIGNAL_CHANNEL_KEYS:
        return f"Ztautau_{lowered}"
    return text


def signal_process_for_channel(channel: str) -> str | None:
    if channel in SIGNAL_CHANNEL_KEYS:
        return f"Ztautau_{channel}"
    if channel in {"zee", "zmumu", "zll"}:
        return "Zll"
    if channel == "zqq":
        return "Zqq"
    return None


def method_channel_order(methods: list[MethodPlotData], explicit_channels: list[str] | None) -> list[str]:
    if explicit_channels:
        return [canonical_channel_name(channel) or channel for channel in explicit_channels]
    ordered: list[str] = []
    seen: set[str] = set()
    for channel in BASELINE_CHANNEL_ORDER:
        if channel not in seen and any(channel in method.channel_order for method in methods):
            ordered.append(channel)
            seen.add(channel)
    for method in methods:
        for channel in method.channel_order:
            if channel not in seen:
                ordered.append(channel)
                seen.add(channel)
    return ordered


def stack_draw_order(process_names: list[str]) -> list[str]:
    return sorted(process_names, key=lambda name: (STACK_PRIORITY.get(name, 999), name))


def display_channel_label(channel: str) -> str:
    if channel == "zee":
        return r"$Z\to ee$"
    if channel == "zmumu":
        return r"$Z\to\mu\mu$"
    if channel == "zll":
        return process_latex_label("Zll")
    if channel == "zqq":
        return process_latex_label("Zqq")
    return channel_latex_label(channel)


def parse_prediction_method(spec: str) -> tuple[str, list[str], list[str]]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid --prediction-method '{spec}'. Use NAME:MC_PATH[:DATA_PATH]."
        )
    name = parts[0].strip()
    mc_path = parts[1].strip()
    data_path = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""
    if not name or not mc_path:
        raise ValueError(
            f"Invalid --prediction-method '{spec}'. NAME and MC_PATH are required."
        )
    mc_paths = expand_paths([mc_path])
    data_paths = expand_paths([data_path]) if data_path else []
    return name, mc_paths, data_paths


def cell_reference_to_column(cell_ref: str) -> str:
    letters = []
    for char in cell_ref:
        if char.isalpha():
            letters.append(char)
        else:
            break
    return "".join(letters)


def read_xlsx_first_sheet_rows(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as workbook_zip:
        workbook = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        sheets = workbook.find(f"{OPENXML_NS}sheets")
        if sheets is None or len(sheets) == 0:
            raise ValueError(f"No sheets found in workbook: {path}")
        first_sheet = sheets[0]
        relationship_id = first_sheet.attrib[f"{REL_NS}id"]

        rels = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
        sheet_target = None
        for relation in rels:
            if relation.attrib.get("Id") == relationship_id:
                sheet_target = relation.attrib.get("Target")
                break
        if sheet_target is None:
            raise ValueError(f"Cannot resolve first sheet in workbook: {path}")

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            shared_root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in shared_root:
                text_parts = [node.text or "" for node in item.iter(f"{OPENXML_NS}t")]
                shared_strings.append("".join(text_parts))

        sheet_root = ET.fromstring(workbook_zip.read(f"xl/{sheet_target}"))

        def parse_cell_value(cell: ET.Element) -> str:
            value_node = cell.find(f"{OPENXML_NS}v")
            if value_node is None:
                return ""
            value = value_node.text or ""
            if cell.attrib.get("t") == "s":
                return shared_strings[int(value)]
            return value

        rows: list[dict[str, str]] = []
        for row in sheet_root.iter(f"{OPENXML_NS}row"):
            row_data: dict[str, str] = {}
            for cell in row.iter(f"{OPENXML_NS}c"):
                reference = cell.attrib.get("r", "")
                column = cell_reference_to_column(reference)
                row_data[column] = parse_cell_value(cell)
            rows.append(row_data)
        return rows


def parse_baseline_workbook(path: Path) -> MethodPlotData:
    rows = read_xlsx_first_sheet_rows(path)
    stack_matrix: dict[str, dict[str, float]] = {}
    total_mc: dict[str, float] = {}
    data_yield: dict[str, float] = {}
    purity: dict[str, float] = {}
    data_over_mc: dict[str, float] = {}
    channel_order: list[str] = []

    for row in rows:
        channel_raw = row.get("A", "").strip()
        if not channel_raw or channel_raw in {"Region", "Column groups", "Highlighting", "Notes"}:
            continue
        channel = canonical_channel_name(channel_raw)
        if channel is None:
            continue
        channel_order.append(channel)

        process_values: dict[str, float] = {}
        for column, process_name in BASELINE_PROCESS_COLUMN_MAP.items():
            value_text = row.get(column, "").strip()
            process_values[process_name] = float(value_text) if value_text else 0.0

        mc_total = float(row.get("V", "0") or 0.0)
        data_count = float(row.get("B", "nan") or float("nan"))
        ratio = float(row.get("W", "nan") or float("nan"))
        signal_process = signal_process_for_channel(channel)
        signal_yield = process_values.get(signal_process, 0.0) if signal_process is not None else float("nan")

        stack_matrix[channel] = process_values
        total_mc[channel] = mc_total
        data_yield[channel] = data_count
        purity[channel] = signal_yield / mc_total if signal_process is not None and mc_total > 0 else float("nan")
        data_over_mc[channel] = ratio if np.isfinite(ratio) else (data_count / mc_total if mc_total > 0 else float("nan"))

    return MethodPlotData(
        name="Baseline",
        channel_order=channel_order,
        stack_matrix=stack_matrix,
        total_mc=total_mc,
        data_yield=data_yield,
        purity=purity,
        data_over_mc=data_over_mc,
        is_baseline=True,
    )


def summarize_prediction_method(name: str, mc_paths: list[str], data_paths: list[str]) -> MethodPlotData:
    mc_events = load_events(mc_paths)
    mc_pred = np.asarray(ak.to_list(mc_events["evenet_pred_class_name"]), dtype=object)
    mc_truth = np.asarray(ak.to_list(mc_events["evenet_truth_class_name"]), dtype=object)
    mc_weight = event_weights(mc_events)

    valid_mc = np.isfinite(mc_weight) & (mc_weight > 0)
    mc_pred = mc_pred[valid_mc]
    mc_truth = mc_truth[valid_mc]
    mc_weight = mc_weight[valid_mc]

    stack_matrix: dict[str, dict[str, float]] = {}
    total_mc: dict[str, float] = {}
    data_yield: dict[str, float] = {}
    purity: dict[str, float] = {}
    data_over_mc: dict[str, float] = {}

    observed_channels: list[str] = []
    for pred_name, truth_name, weight in zip(mc_pred, mc_truth, mc_weight):
        channel = canonical_channel_name(pred_name)
        process_name = canonical_process_name(truth_name)
        if channel is None or process_name is None:
            continue
        if channel not in stack_matrix:
            observed_channels.append(channel)
            stack_matrix[channel] = {}
        stack_matrix[channel][process_name] = stack_matrix[channel].get(process_name, 0.0) + float(weight)

    for channel in observed_channels:
        total = float(sum(stack_matrix[channel].values()))
        total_mc[channel] = total
        signal_process = signal_process_for_channel(channel)
        signal_yield = stack_matrix[channel].get(signal_process, 0.0) if signal_process is not None else float("nan")
        purity[channel] = signal_yield / total if signal_process is not None and total > 0 else float("nan")

    if data_paths:
        data_events = load_events(data_paths)
        data_pred = np.asarray(ak.to_list(data_events["evenet_pred_class_name"]), dtype=object)
        for pred_name in data_pred:
            channel = canonical_channel_name(pred_name)
            if channel is None:
                continue
            data_yield[channel] = data_yield.get(channel, 0.0) + 1.0

    for channel in observed_channels:
        mc_total = total_mc.get(channel, 0.0)
        count = data_yield.get(channel, float("nan"))
        data_over_mc[channel] = count / mc_total if mc_total > 0 and np.isfinite(count) else float("nan")

    return MethodPlotData(
        name=name,
        channel_order=observed_channels,
        stack_matrix=stack_matrix,
        total_mc=total_mc,
        data_yield=data_yield,
        purity=purity,
        data_over_mc=data_over_mc,
    )


def all_process_names(methods: list[MethodPlotData]) -> list[str]:
    found: set[str] = set()
    for method in methods:
        for values in method.stack_matrix.values():
            found.update(values.keys())
    return stack_draw_order(list(found))


def style_for_method(method_name: str, method_index: int, is_baseline: bool) -> dict[str, Any]:
    return {
        "color": method_color(method_name, method_index),
        "linestyle": "--" if is_baseline else "-",
        "marker": "o",
        "hatch": "///" if is_baseline else None,
        "alpha": 0.85 if is_baseline else 0.92,
    }


def plot_comparison(
    methods: list[MethodPlotData],
    channels: list[str],
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    process_names = all_process_names(methods)
    num_methods = len(methods)
    x = np.arange(len(channels), dtype=np.float64)
    group_width = min(0.84, 0.20 * num_methods + 0.18)
    bar_width = group_width / max(num_methods, 1)

    fig = plt.figure(figsize=(max(13.5, 1.05 * len(channels) + 4.0), 10.8), dpi=220)
    gs = fig.add_gridspec(3, 1, height_ratios=[5.3, 1.8, 1.8], hspace=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_purity = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_ratio = fig.add_subplot(gs[2, 0], sharex=ax_main)

    component_legend_handles: list[Any] = []
    component_legend_labels: list[str] = []
    method_legend_handles: list[Any] = []
    method_legend_labels: list[str] = []

    max_yield = 0.0
    summary: dict[str, Any] = {"channels": channels, "methods": {}}

    for method_index, method in enumerate(methods):
        method_style = style_for_method(method.name, method_index, method.is_baseline)
        x_offset = x - group_width / 2.0 + (method_index + 0.5) * bar_width
        bottoms = np.zeros(len(channels), dtype=np.float64)

        for process_index, process_name in enumerate(process_names):
            values = np.array(
                [method.stack_matrix.get(channel, {}).get(process_name, 0.0) for channel in channels],
                dtype=np.float64,
            )
            if not np.any(values > 0):
                continue
            bars = ax_main.bar(
                x_offset,
                values,
                width=bar_width * 0.95,
                bottom=bottoms,
                color=BACKGROUND_COLOR if process_name == "Other" else process_color(process_name, process_index),
                edgecolor=method_style["color"],
                linewidth=1.0,
                alpha=method_style["alpha"],
                hatch=method_style["hatch"],
                zorder=2,
            )
            if process_name not in component_legend_labels:
                component_legend_handles.append(bars[0])
                component_legend_labels.append(process_name)
            bottoms += values

        total_values = np.array([method.total_mc.get(channel, 0.0) for channel in channels], dtype=np.float64)
        data_values = np.array([method.data_yield.get(channel, np.nan) for channel in channels], dtype=np.float64)
        purity_values = np.array([method.purity.get(channel, np.nan) for channel in channels], dtype=np.float64)
        ratio_values = np.array([method.data_over_mc.get(channel, np.nan) for channel in channels], dtype=np.float64)

        max_yield = max(max_yield, float(np.nanmax(total_values)) if np.any(np.isfinite(total_values)) else 0.0)
        if np.any(np.isfinite(data_values)):
            max_yield = max(max_yield, float(np.nanmax(data_values)))

        data_mask = np.isfinite(data_values)
        if np.any(data_mask):
            ax_main.scatter(
                x_offset[data_mask],
                data_values[data_mask],
                s=18,
                color=method_style["color"],
                marker=method_style["marker"],
                facecolors="white",
                linewidths=1.0,
                zorder=4,
            )

        purity_line, = ax_purity.plot(
            x,
            purity_values,
            color=method_style["color"],
            linestyle=method_style["linestyle"],
            marker=method_style["marker"],
            linewidth=1.8,
            markersize=4.0,
            label=method.name,
        )
        ax_ratio.plot(
            x,
            ratio_values,
            color=method_style["color"],
            linestyle=method_style["linestyle"],
            marker=method_style["marker"],
            linewidth=1.8,
            markersize=4.0,
            label=method.name,
        )
        method_legend_handles.append(purity_line)
        method_legend_labels.append(method.name)

        summary["methods"][method.name] = {
            "is_baseline": bool(method.is_baseline),
            "per_channel": {
                channel: {
                    "stack": {process_name: float(method.stack_matrix.get(channel, {}).get(process_name, 0.0)) for process_name in process_names},
                    "total_mc_yield": float(method.total_mc.get(channel, 0.0)),
                    "data_yield": float(method.data_yield.get(channel, np.nan)),
                    "signal_purity": float(method.purity.get(channel, np.nan)),
                    "data_over_mc": float(method.data_over_mc.get(channel, np.nan)),
                }
                for channel in channels
            },
        }

    ax_main.set_title(title)
    ax_main.set_ylabel("Yield")
    ax_main.grid(axis="y", linestyle=":", alpha=0.28)
    ax_main.set_ylim(0.0, max(1.0, max_yield) * 1.18)

    ax_purity.set_ylabel("Purity")
    ax_purity.set_ylim(0.0, 1.05)
    ax_purity.grid(axis="y", linestyle=":", alpha=0.28)
    ax_purity.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.5)

    ratio_values_all = [
        method.data_over_mc.get(channel, np.nan)
        for method in methods
        for channel in channels
    ]
    finite_ratio_values = np.asarray([value for value in ratio_values_all if np.isfinite(value)], dtype=np.float64)
    ratio_upper = 2.0 if finite_ratio_values.size == 0 else max(1.6, min(3.0, float(np.nanmax(finite_ratio_values) * 1.15)))
    ax_ratio.set_ylabel("Data/MC")
    ax_ratio.set_ylim(0.0, ratio_upper)
    ax_ratio.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_ratio.grid(axis="y", linestyle=":", alpha=0.28)

    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([display_channel_label(channel) for channel in channels], rotation=30, ha="right")
    ax_ratio.set_xlabel("Predicted channel")
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_purity.get_xticklabels(), visible=False)

    component_labels_display = [process_latex_label(label) if label != "Other" else "Other bkg" for label in component_legend_labels]
    first_legend = ax_main.legend(
        component_legend_handles,
        component_labels_display,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="MC truth components",
    )
    ax_main.add_artist(first_legend)
    ax_purity.legend(method_legend_handles, method_legend_labels, loc="upper left", ncols=max(1, len(method_legend_handles)), frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    methods: list[MethodPlotData] = [parse_baseline_workbook(args.baseline_xlsx.resolve())]
    for method_spec in args.prediction_method:
        method_name, mc_paths, data_paths = parse_prediction_method(method_spec)
        methods.append(summarize_prediction_method(method_name, mc_paths, data_paths))

    channels = method_channel_order(methods, args.channels)
    summary = plot_comparison(
        methods=methods,
        channels=channels,
        output_path=args.output.resolve(),
        title=args.title,
    )

    summary_json = args.summary_json.resolve() if args.summary_json is not None else args.output.resolve().with_suffix(".json")
    with summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[channel-purity-compare] wrote figure to {args.output.resolve()}")
    print(f"[channel-purity-compare] wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
