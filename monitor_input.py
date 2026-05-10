#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from plot_style import plot_data_mc_comparison

EPS = 1e-12


def parquet_files(paths: list[Path]) -> list[str]:
    files: list[str] = []
    for path in paths:
        if path.is_file() and path.suffix == ".parquet":
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(p) for p in sorted(path.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {paths}")
    return files

def read_events(paths: list[Path], max_events: int) -> ak.Array:
    arrays = []
    for file_name in parquet_files(paths):
        arrays.append(ak.from_parquet(file_name))
    events = ak.concatenate(arrays)
    if max_events is not None:
        events = events[:max_events]
    return events


def main() -> None:
    parser = argparse.ArgumentParser("Simple parquet monitor plots")
    parser.add_argument("--data-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--mc-dir", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("monitor_plots"))
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    events = read_events(args.data_dir + args.mc_dir, args.max_events)



if __name__ == "__main__":
    main()