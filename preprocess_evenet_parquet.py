#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


from evenet.control.global_config import global_config
from preprocessing.helper import (
    PostProcessor,
    build_log_scale_plan,
    event_split_indices,
    process_dict,
    slice_event_dict,
)

parser = argparse.ArgumentParser(
    description=(
        "Preprocess ml_pipeline_lite EveNet parquet shards into shuffled train/val/test parquet files. "
        "This stage keeps event weights as-is and only applies the standard EveNet preprocessing transforms."
    )
)
parser.add_argument(
    "--manifest",
    type=Path,
    required=True,
    help="Path to manifest.json from ml_pipeline_lite/build_evenet_input_from_parquet.py.",
)
parser.add_argument(
    "--config",
    type=Path,
    default=Path("ml_pipeline_lite/config/preprocess_config.yaml"),
    help="EveNet preprocessing config.",
)
parser.add_argument(
    "--store-dir",
    type=Path,
    required=True,
    help="Output directory for preprocessed parquet files and normalization.pt.",
)
parser.add_argument(
    "--split-ratio",
    default="0.4,0.1,0.5",
    help="Event split ratio train,val,test. Default: 0.4,0.1,0.5.",
)
parser.add_argument(
    "--skip-data",
    action="store_true",
    help="Skip data shards.",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="Number of CPU worker processes.",
)
parser.add_argument(
    "--split-seed",
    type=int,
    default=42,
    help="Seed for event-level split shuffling.",
)
parser.add_argument(
    "--shuffle-seed",
    type=int,
    default=31,
    help="Seed for per-output parquet shuffling.",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
