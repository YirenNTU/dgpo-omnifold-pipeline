#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_PIPELINE_DIR = REPO_ROOT / "ml_pipeline"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ML_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_PIPELINE_DIR))

try:
    from ml_pipeline.plot_style import (
        OKABE_ITO,
        channel_latex_label,
        method_color,
        process_color,
        process_latex_label,
    )
except ModuleNotFoundError as exc:
    if exc.name != "awkward":
        raise

    OKABE_ITO = {
        "black": "#000000",
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
    }
    METHOD_COLORS = {
        "Baseline": OKABE_ITO["vermillion"],
        "EveNet": OKABE_ITO["blue"],
        "Target": OKABE_ITO["reddish_purple"],
        "Truth": OKABE_ITO["orange"],
    }
    METHOD_COLOR_CYCLE = (
        OKABE_ITO["vermillion"],
        OKABE_ITO["blue"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["sky_blue"],
    )
    PROCESS_COLOR_CYCLE = (
        "#4477AA",
        "#EE6677",
        "#228833",
        "#CCBB44",
        "#66CCEE",
        "#AA3377",
        "#EE7733",
        "#009988",
        "#BBBBBB",
    )
    PROCESS_LATEX_LABELS = {
        "Ztautau_pipi": r"$\tau\tau\to\pi\pi$",
        "Ztautau_pirho": r"$\tau\tau\to\pi\rho$",
        "Ztautau_rhopi": r"$\tau\tau\to\rho\pi$",
        "Ztautau_rhorho": r"$\tau\tau\to\rho\rho$",
        "Ztautau_pie": r"$\tau\tau\to\pi e$",
        "Ztautau_epi": r"$\tau\tau\to e\pi$",
        "Ztautau_pimu": r"$\tau\tau\to\pi\mu$",
        "Ztautau_mupi": r"$\tau\tau\to\mu\pi$",
        "Ztautau_rhoe": r"$\tau\tau\to\rho e$",
        "Ztautau_erho": r"$\tau\tau\to e\rho$",
        "Ztautau_rhomu": r"$\tau\tau\to\rho\mu$",
        "Ztautau_murho": r"$\tau\tau\to\mu\rho$",
        "Ztautau_ee": r"$\tau\tau\to ee$",
        "Ztautau_mumu": r"$\tau\tau\to\mu\mu$",
        "Ztautau_emu": r"$\tau\tau\to e\mu$",
        "Ztautau_mue": r"$\tau\tau\to\mu e$",
        "Ztautau_others": r"$Z\to\tau\tau$ other",
        "Zll": r"$Z\to\ell\ell$",
        "Zqq": r"$Z\to q\bar{q}$",
    }

    def method_color(method: str, method_index: int) -> str:
        return METHOD_COLORS.get(method, METHOD_COLOR_CYCLE[method_index % len(METHOD_COLOR_CYCLE)])

    def process_color(process_name: str, process_index: int = 0) -> str:
        return PROCESS_COLOR_CYCLE[process_index % len(PROCESS_COLOR_CYCLE)]

    def process_latex_label(sample_name: str) -> str:
        return PROCESS_LATEX_LABELS.get(sample_name, sample_name.replace("_", r"\_"))

    def channel_latex_label(name: str) -> str:
        channel = name.removeprefix("Ztautau_")
        return PROCESS_LATEX_LABELS.get(f"Ztautau_{channel}", name.replace("_", r"\_"))

DATA_COLOR = OKABE_ITO["black"]
BACKGROUND_COLOR = "#D8D8D8"
DEFAULT_CLASS_NAME = "unselected"
DEFAULT_OUTPUT = ML_PIPELINE_DIR / "plots" / "channel_purity_side_by_side.png"
DEFAULT_BASELINE_XLSX = REPO_ROOT / "data" / "baseline_yield.xlsx"
DEFAULT_ANALYSIS_CONFIG = ML_PIPELINE_DIR / "config" / "analysis.yaml"
OPENXML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
METHOD_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h")

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

CHANNEL_ORDER = [
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
    "rhorho",
]

PROCESS_ORDER = [
    "Ztautau_pipi",
    "Ztautau_pirho",
    "Ztautau_rhopi",
    "Ztautau_rhorho",
    "Ztautau_pie",
    "Ztautau_epi",
    "Ztautau_pimu",
    "Ztautau_mupi",
    "Ztautau_rhoe",
    "Ztautau_erho",
    "Ztautau_rhomu",
    "Ztautau_murho",
    "Ztautau_ee",
    "Ztautau_mumu",
    "Ztautau_emu",
    "Ztautau_mue",
    "Ztautau_others",
    "Zll",
    "Zqq",
    "Other",
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
        description="Compare baseline and exported QI channel-purity yields side by side."
    )
    parser.add_argument(
        "--baseline-xlsx",
        type=Path,
        default=DEFAULT_BASELINE_XLSX,
        help="Baseline yield workbook. Pass 'none' to skip it.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory produced by export_evenet_qi_inputs.py.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="Exported method definition. PATH may be a method directory or its processed directory.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Method directory names under --base-dir. Defaults to export_summary.json methods.",
    )
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--data-sample-name", default="data94")
    parser.add_argument("--mc-sample-names", nargs="+", default=["Ztautau", "Zll", "Zqq"])
    parser.add_argument("--channels", nargs="*", default=None)
    parser.add_argument("--load-batch-size", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--title", default="Channel Purity Comparison")
    return parser.parse_args()


def baseline_xlsx_arg(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    text = str(path)
    if text.lower() in {"", "none", "null", "skip"}:
        return None
    return Path(path)


def method_display_name(method: str) -> str:
    mapping = {
        "baseline": "Baseline",
        "evenet": "EveNet",
        "target": "Target",
        "truth": "Truth",
    }
    return mapping.get(method, method)


def read_export_summary_methods(base_dir: Path) -> list[str]:
    summary_path = base_dir / "export_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        methods = summary.get("methods")
        if isinstance(methods, list):
            return [str(method) for method in methods]
    return sorted(path.name for path in base_dir.iterdir() if (path / "processed").is_dir())


def resolve_method_specs(args: argparse.Namespace, include_baseline_xlsx: bool) -> list[tuple[str, Path]]:
    if args.method:
        output: list[tuple[str, Path]] = []
        for spec in args.method:
            name, separator, path_text = spec.partition(":")
            if not separator or not name.strip() or not path_text.strip():
                raise ValueError(f"Invalid --method '{spec}'. Use NAME:PATH.")
            output.append((name.strip(), Path(path_text).expanduser().resolve()))
        return output

    if args.base_dir is None:
        return []

    base_dir = args.base_dir.expanduser().resolve()
    method_names = args.methods if args.methods is not None else read_export_summary_methods(base_dir)
    if args.methods is None and include_baseline_xlsx:
        method_names = [method for method in method_names if method != "baseline"]
    return [(method_display_name(method), base_dir / method) for method in method_names]


def read_analysis_channels(path: Path) -> list[str]:
    try:
        import yaml
    except ModuleNotFoundError:
        return CHANNEL_ORDER.copy()

    if not path.exists():
        return CHANNEL_ORDER.copy()
    config = yaml.safe_load(path.read_text()) or {}
    regions: list[str] = []
    prediction_cfg = config.get("NeutrinoPrediction") or {}
    for value in prediction_cfg.values():
        if isinstance(value, list):
            regions.extend(str(item) for item in value)
        elif isinstance(value, dict):
            regions.extend(str(item) for item in value.values())
    output: list[str] = []
    seen: set[str] = set()
    for region in regions:
        channel = canonical_channel_name(region)
        if channel is not None and channel in SIGNAL_CHANNEL_KEYS and channel not in seen:
            output.append(channel)
            seen.add(channel)
    return output or CHANNEL_ORDER.copy()


def channel_to_region(channel: str) -> str:
    if channel.startswith("Ztautau_"):
        return channel
    return f"Ztautau_{channel}"


def canonical_channel_name(name: str) -> str | None:
    text = str(name).strip()
    if not text or text == DEFAULT_CLASS_NAME:
        return None
    lowered = CHANNEL_ALIASES.get(text.lower(), text.lower())
    if lowered.startswith("ztautau_"):
        lowered = lowered.removeprefix("ztautau_")
    if lowered in {"zqq", "zll"}:
        return lowered
    return lowered


def canonical_process_name(name: str) -> str | None:
    text = str(name).strip()
    if not text or text == DEFAULT_CLASS_NAME:
        return None
    lowered = CHANNEL_ALIASES.get(text.lower(), text.lower())
    if lowered in {"zqq", "z->qq", "zqqbar"}:
        return "Zqq"
    if lowered in {"zll", "z->ll", "z->ell ell"}:
        return "Zll"
    if lowered in {"other", "other bkg", "other_bkg"}:
        return "Other"
    if lowered.startswith("ztautau_"):
        channel = lowered.removeprefix("ztautau_")
        if channel in SIGNAL_CHANNEL_KEYS or channel == "others":
            return f"Ztautau_{channel}"
    if lowered in SIGNAL_CHANNEL_KEYS or lowered == "others":
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


def is_background_like_process(name: str) -> bool:
    lowered = name.lower()
    return (
        name == DEFAULT_CLASS_NAME
        or lowered in {"zll", "zqq"}
        or lowered.endswith("_others")
        or lowered == "others"
        or lowered == "other"
        or "background" in lowered
    )


def method_channel_order(methods: list[MethodPlotData], explicit_channels: list[str] | None) -> list[str]:
    if explicit_channels:
        return [canonical_channel_name(channel) or channel for channel in explicit_channels]
    ordered: list[str] = []
    seen: set[str] = set()
    for channel in CHANNEL_ORDER:
        if channel not in seen and any(channel in method.channel_order for method in methods):
            ordered.append(channel)
            seen.add(channel)
    for method in methods:
        for channel in method.channel_order:
            if channel in SIGNAL_CHANNEL_KEYS and channel not in seen:
                ordered.append(channel)
                seen.add(channel)
    return ordered


def stack_draw_order(process_names: list[str]) -> list[str]:
    known = [name for name in PROCESS_ORDER if name in process_names]
    extra_signal = sorted(
        name for name in process_names if name not in PROCESS_ORDER and not is_background_like_process(name)
    )
    extra_background = sorted(
        name for name in process_names if name not in PROCESS_ORDER and is_background_like_process(name)
    )
    return [*known, *extra_signal, *extra_background]


def process_stack_color(process_name: str, index: int) -> str:
    if is_background_like_process(process_name):
        return BACKGROUND_COLOR
    return process_color(process_name, index)


def display_channel_label(channel: str) -> str:
    if channel == "zll":
        return process_latex_label("Zll")
    if channel == "zqq":
        return process_latex_label("Zqq")
    return channel_latex_label(channel)


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
        if channel is None or channel not in SIGNAL_CHANNEL_KEYS:
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


def processed_dir(method_path: Path) -> Path:
    if method_path.name == "processed":
        return method_path
    if (method_path / "processed").is_dir():
        return method_path / "processed"
    return method_path


def sample_dirs(processed_path: Path, sample_name: str) -> list[Path]:
    if not processed_path.is_dir():
        return []
    prefix = f"{sample_name}_"
    return sorted(
        path
        for path in processed_path.iterdir()
        if path.is_dir() and (path.name == sample_name or path.name.startswith(prefix))
    )


def region_files(processed_path: Path, sample_name: str, region: str) -> list[Path]:
    output: list[Path] = []
    for sample_dir in sample_dirs(processed_path, sample_name):
        path = sample_dir / f"filtered___{region}.parquet"
        if path.exists():
            output.append(path)
    return output


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


def numeric_values(events: ak.Array, name: str, default: float) -> np.ndarray:
    import awkward as ak

    if name not in events.fields:
        return np.full(len(events), default, dtype=np.float64)
    return np.asarray(ak.to_numpy(ak.fill_none(events[name], default)), dtype=np.float64)


def string_values(events: ak.Array, name: str, default: str) -> np.ndarray:
    import awkward as ak

    if name not in events.fields:
        return np.full(len(events), default, dtype=object)
    return np.asarray([str(value) for value in ak.to_list(ak.fill_none(events[name], default))], dtype=object)


def add_weighted_events(
    stack: dict[str, float],
    events: ak.Array,
    sample_name: str,
) -> None:
    weights = numeric_values(events, "weight", 1.0)
    labels = string_values(events, "classification_target_name", sample_name)
    for label, weight in zip(labels, weights):
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        process_name = canonical_process_name(label)
        if process_name is None:
            process_name = canonical_process_name(sample_name)
        if process_name is None:
            continue
        stack[process_name] = stack.get(process_name, 0.0) + float(weight)


def data_count(events: ak.Array) -> float:
    return float(len(events))


def summarize_exported_method(
    name: str,
    method_path: Path,
    channels: list[str],
    data_sample_name: str,
    mc_sample_names: list[str],
    batch_size: int,
) -> MethodPlotData:
    processed_path = processed_dir(method_path)
    if not processed_path.is_dir():
        raise FileNotFoundError(f"Cannot find processed directory for method '{name}': {method_path}")

    stack_matrix: dict[str, dict[str, float]] = {}
    total_mc: dict[str, float] = {}
    data_yield: dict[str, float] = {}
    purity: dict[str, float] = {}
    data_over_mc: dict[str, float] = {}
    observed_channels: list[str] = []

    for channel in channels:
        region = channel_to_region(channel)
        stack: dict[str, float] = {}
        for sample_name in mc_sample_names:
            files = region_files(processed_path, sample_name, region)
            for events in iter_parquet_batches(files, {"weight", "classification_target_name"}, batch_size):
                add_weighted_events(stack, events, sample_name)

        data_total = 0.0
        data_files = region_files(processed_path, data_sample_name, region)
        for events in iter_parquet_batches(data_files, {"weight"}, batch_size):
            data_total += data_count(events)

        if not stack and data_total == 0.0:
            continue

        observed_channels.append(channel)
        stack_matrix[channel] = stack
        total = float(sum(stack.values()))
        total_mc[channel] = total
        data_yield[channel] = data_total if data_files else float("nan")
        signal_process = signal_process_for_channel(channel)
        signal_yield = stack.get(signal_process, 0.0) if signal_process is not None else float("nan")
        purity[channel] = signal_yield / total if signal_process is not None and total > 0 else float("nan")
        data_over_mc[channel] = data_total / total if total > 0 and data_files else float("nan")

    return MethodPlotData(
        name=name,
        channel_order=observed_channels,
        stack_matrix=stack_matrix,
        total_mc=total_mc,
        data_yield=data_yield,
        purity=purity,
        data_over_mc=data_over_mc,
        is_baseline=name.lower() == "baseline",
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
        "marker": METHOD_MARKERS[method_index % len(METHOD_MARKERS)],
        "hatch": "///" if is_baseline else None,
        "alpha": 0.85 if is_baseline else 0.92,
    }


def finite_nanmax(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    return float(np.max(finite_values)) if finite_values.size else 0.0


def plot_comparison(
    methods: list[MethodPlotData],
    channels: list[str],
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    process_names = all_process_names(methods)
    num_methods = len(methods)
    x = np.arange(len(channels), dtype=np.float64)
    group_width = min(0.84, 0.20 * num_methods + 0.18)
    bar_width = group_width / max(num_methods, 1)

    fig = plt.figure(figsize=(max(13.5, 1.05 * len(channels) + 4.0), 12.2), dpi=220)
    gs = fig.add_gridspec(4, 1, height_ratios=[5.6, 1.55, 1.55, 1.55], hspace=0.08)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_purity = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_ratio = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_signal = fig.add_subplot(gs[3, 0], sharex=ax_main)

    component_legend_handles: list[Any] = []
    component_legend_labels: list[str] = []
    method_legend_handles: list[Any] = []
    method_legend_labels: list[str] = []
    method_lower_legend_handles: list[Any] = []
    method_lower_legend_labels: list[str] = []

    max_yield = 0.0
    summary: dict[str, Any] = {"channels": channels, "methods": {}}

    for method_index, method in enumerate(methods):
        method_style = style_for_method(method.name, method_index, method.is_baseline)
        x_offset = x - group_width / 2.0 + (method_index + 0.5) * bar_width
        bottoms = np.zeros(len(channels), dtype=np.float64)

        exact_signal_yields = np.zeros(len(channels), dtype=np.float64)
        for process_index, process_name in enumerate(process_names):
            values = np.array(
                [method.stack_matrix.get(channel, {}).get(process_name, 0.0) for channel in channels],
                dtype=np.float64,
            )
            if not np.any(values > 0):
                continue
            bar_color = process_stack_color(process_name, process_index)
            bars = ax_main.bar(
                x_offset,
                values,
                width=bar_width * 0.95,
                bottom=bottoms,
                color=bar_color,
                edgecolor=bar_color if method.is_baseline else "white",
                linewidth=1.0,
                alpha=method_style["alpha"],
                hatch=method_style["hatch"],
                zorder=2,
            )
            if process_name not in component_legend_labels:
                component_legend_handles.append(bars[0])
                component_legend_labels.append(process_name)
            bottoms += values
            for channel_index, channel in enumerate(channels):
                if process_name == signal_process_for_channel(channel):
                    exact_signal_yields[channel_index] = values[channel_index]

        total_values = np.array([method.total_mc.get(channel, 0.0) for channel in channels], dtype=np.float64)
        data_values = np.array([method.data_yield.get(channel, np.nan) for channel in channels], dtype=np.float64)
        purity_values = np.array([method.purity.get(channel, np.nan) for channel in channels], dtype=np.float64)
        ratio_values = np.array([method.data_over_mc.get(channel, np.nan) for channel in channels], dtype=np.float64)

        max_yield = max(max_yield, finite_nanmax(total_values), finite_nanmax(data_values))

        data_unc = np.sqrt(np.clip(data_values, a_min=0.0, a_max=None))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_unc = np.divide(
                data_unc,
                total_values,
                out=np.full_like(data_unc, np.nan, dtype=np.float64),
                where=total_values > 0,
            )

        data_mask = np.isfinite(data_values)
        if np.any(data_mask):
            ax_main.errorbar(
                x_offset[data_mask],
                data_values[data_mask],
                yerr=data_unc[data_mask],
                fmt=method_style["marker"],
                color=DATA_COLOR,
                ecolor=DATA_COLOR,
                elinewidth=1.1,
                capsize=2.5,
                markersize=4.6,
                markerfacecolor=DATA_COLOR,
                markeredgecolor=DATA_COLOR,
                zorder=4,
            )

        bars = ax_purity.bar(
            x_offset,
            purity_values,
            width=bar_width * 0.82,
            color=method_style["color"],
            edgecolor=method_style["color"],
            linewidth=0.8,
            alpha=0.78,
            hatch=method_style["hatch"],
            zorder=2,
        )

        method_lower_legend_handles.append(bars[0])
        method_lower_legend_labels.append(method.name)
        ratio_mask = np.isfinite(ratio_values)
        if np.any(ratio_mask):
            ax_ratio.errorbar(
                x_offset[ratio_mask],
                ratio_values[ratio_mask],
                yerr=ratio_unc[ratio_mask],
                fmt=method_style["marker"],
                color=method_style["color"],
                ecolor=method_style["color"],
                elinewidth=1.0,
                capsize=2.3,
                markersize=4.4,
                markerfacecolor=method_style["color"],
                markeredgecolor=method_style["color"],
                linestyle="None",
                zorder=3,
            )
        ax_signal.bar(
            x_offset,
            exact_signal_yields,
            width=bar_width * 0.82,
            color=method_style["color"],
            edgecolor=method_style["color"],
            linewidth=0.8,
            alpha=0.78,
            hatch=method_style["hatch"],
            zorder=2,
        )
        method_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=DATA_COLOR,
                linestyle="None",
                marker=method_style["marker"],
                markerfacecolor=DATA_COLOR,
                markeredgecolor=DATA_COLOR,
                markersize=6.0,
            )
        )
        method_legend_labels.append(method.name)

        summary["methods"][method.name] = {
            "is_baseline": bool(method.is_baseline),
            "per_channel": {
                channel: {
                    "stack": {
                        process_name: float(method.stack_matrix.get(channel, {}).get(process_name, 0.0))
                        for process_name in process_names
                    },
                    "total_mc_yield": float(method.total_mc.get(channel, 0.0)),
                    "data_yield": float(method.data_yield.get(channel, np.nan)),
                    "signal_purity": float(method.purity.get(channel, np.nan)),
                    "exact_signal_yield": float(exact_signal_yields[channels.index(channel)]),
                    "data_over_mc": float(method.data_over_mc.get(channel, np.nan)),
                }
                for channel in channels
            },
        }

    ax_main.set_title(title)
    ax_main.set_ylabel("Yield")
    ax_main.grid(axis="y", linestyle=":", alpha=0.28)
    ax_main.set_ylim(0.0, max(1.0, max_yield) * 1.30)

    ax_purity.set_ylabel("Purity")
    ax_purity.set_ylim(0.0, 1.05)
    ax_purity.grid(axis="y", linestyle=":", alpha=0.28)
    ax_purity.axhline(0.5, color="gray", linestyle=":", linewidth=0.9, alpha=0.5)

    ax_ratio.set_ylabel("Data/MC")
    ax_ratio.set_ylim(0.8, 1.2)
    ax_ratio.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax_ratio.grid(axis="y", linestyle=":", alpha=0.28)

    ax_signal.set_ylabel("Signal")
    ax_signal.grid(axis="y", linestyle=":", alpha=0.28)
    ax_signal.set_ylim(bottom=0.0)

    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([display_channel_label(channel) for channel in channels], rotation=30, ha="right")
    ax_signal.set_xticks(x)
    ax_signal.set_xticklabels([display_channel_label(channel) for channel in channels], rotation=30, ha="right")
    ax_signal.set_xlabel("Selected channel")
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_purity.get_xticklabels(), visible=False)
    plt.setp(ax_ratio.get_xticklabels(), visible=False)

    component_labels_display = [
        process_latex_label(label) if label != "Other" else "Other bkg"
        for label in component_legend_labels
    ]
    if component_legend_handles:
        first_legend = ax_main.legend(
            component_legend_handles,
            component_labels_display,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            frameon=False,
            title="MC truth components",
            ncols=min(max(4, math.ceil(len(component_legend_labels) / 2)), len(component_legend_labels)),
            fontsize=9.5,
            title_fontsize=10.5,
        )
        ax_main.add_artist(first_legend)
    second_legend = ax_main.legend(
        method_legend_handles,
        method_legend_labels,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        title="Methods",
        ncols=1,
        fontsize=9.5,
        title_fontsize=10.5,
    )
    ax_main.add_artist(second_legend)
    ax_purity.legend(
        method_lower_legend_handles,
        method_lower_legend_labels,
        loc="upper right",
        frameon=False,
        ncols=max(1, min(4, len(method_legend_handles))),
        fontsize=8.0,
        title_fontsize=8.3,
    )
    ax_signal.legend(
        method_lower_legend_handles,
        method_lower_legend_labels,
        loc="upper right",
        frameon=False,
        ncols=max(1, min(4, len(method_legend_handles))),
        fontsize=8.0,
        title_fontsize=8.3,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    args = parse_args()
    baseline_path = baseline_xlsx_arg(args.baseline_xlsx)
    methods: list[MethodPlotData] = []
    if baseline_path is not None:
        methods.append(parse_baseline_workbook(baseline_path.expanduser().resolve()))

    scan_channels = (
        [canonical_channel_name(channel) or channel for channel in args.channels]
        if args.channels
        else read_analysis_channels(args.analysis_config.expanduser().resolve())
    )
    method_specs = resolve_method_specs(args, include_baseline_xlsx=baseline_path is not None)
    for method_name, method_path in method_specs:
        methods.append(
            summarize_exported_method(
                method_name,
                method_path,
                scan_channels,
                args.data_sample_name,
                args.mc_sample_names,
                args.load_batch_size,
            )
        )

    if not methods:
        raise ValueError("No baseline workbook or exported methods were provided.")

    channels = method_channel_order(methods, args.channels)
    if not channels:
        raise ValueError("No channels with yields were found.")

    output_path = args.output.expanduser().resolve()
    summary = plot_comparison(
        methods=methods,
        channels=channels,
        output_path=output_path,
        title=args.title,
    )

    summary_json = (
        args.summary_json.expanduser().resolve()
        if args.summary_json is not None
        else output_path.with_suffix(".json")
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[channel-purity-compare] wrote figure to {output_path}")
    print(f"[channel-purity-compare] wrote summary to {summary_json}")


if __name__ == "__main__":
    main()
