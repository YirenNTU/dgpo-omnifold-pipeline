#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import vector
import yaml

from common import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evenet input from parquet"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for lite parquet shards and monitoring.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Rows per parquet record-batch read.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=100000,
        help="Selected rows per output shard.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers. Each worker processes one input parquet file.",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.analysis_config)
    feature_config = parse_feature_config(config)

if __name__ == "__main__":
    main()