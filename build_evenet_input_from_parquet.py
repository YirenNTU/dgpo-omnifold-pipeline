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

from common import (
    read_yaml, process_latex_label, build_classification_lookup,
    event_preselection_mask, build_input_particle_mask
)
from generate_event_info_yaml import parse_feature_config
from ml_pipeline.common import build_classification_lookup
from ml_pipeline_lite.build_evenet_input_from_parquet import build_input_particle_mask

vector.register_awkward()


@dataclass(frozen=True)
class Sample:
    key: str
    name: str
    is_data: bool
    is_signal: bool
    files: tuple[str, ...]
    file_source: str
    norm_factor: float | None
    lumi: float | None
    total_initial_num_events: float | None
    plot_label: str

def parse_samples(config: dict[str, Any], selected_keys: list[str] | None) -> list[Sample]:
    selected = set(selected_keys or [])
    samples: list[Sample] = []
    for key, sample_cfg in config["Samples"].items():
        if selected and key not in selected:
            continue
        input_files = sample_cfg["input_files"]
        samples.append(
            Sample(
                key=key,
                name=str(sample_cfg.get("name", key)),
                is_data=bool(sample_cfg.get("is_data", False)),
                is_signal=bool(sample_cfg.get("is_signal", False)),
                files=tuple(str(item) for item in input_files),
                file_source="input_files",
                norm_factor=float(sample_cfg.get("norm_factor", 1)),
                lumi=float(sample_cfg.get("lumi", 1)),
                total_initial_num_events=None,
                plot_label=process_latex_label(str(sample_cfg.get("name", key)))
            )
        )
    return samples


def read_file_initial_total_num_events(path: str) -> float | None:
    parquet = pq.ParquetFile(path)
    for record_batch in parquet.iter_batches(batch_size=1, columns=["initial_total_num_events"]):
        values = ak.to_numpy(ak.from_arrow(record_batch)["initial_total_num_events"], allow_missing=False)
        if len(values) == 0:
            continue
        return float(values[0])
    return None

def attach_sample_total_intial_events(samples: list[Sample]) -> list[Sample]:
    resolved_samples: list[Sample] = []
    for sample in samples:
        if sample.is_data:
            resolved_samples.append(replace(sample))
            continue
        total = 0.0
        for path in sample.files:
            file_total = read_file_initial_total_num_events(path)
            if file_total is None:
                raise ValueError(
                    f"Sample '{sample.key}' file '{path}' is missing initial_total_num_events."
                )
            total += float(file_total)
        resolved_samples.append(replace(sample, total_initial_num_events=total))
    return resolved_samples

def infer_luminosity(samples: list[Sample]) -> float | None:
    data_lumis = [sample.lumi for sample in samples if sample.is_data and sample.lumi is not None]
    return sum(data_lumis) if data_lumis else None

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build evenet input from parquet"
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("ml_pipeline/config/analysis.yaml"),
        help="Analysis YAML with Samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for lite parquet shards and monitoring.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Optional subset of sample keys from analysis.yaml.",
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
    parser.add_argument(
        "--remove_neutral_non_photon",
        action="store_true",
        default=False,
        help="Remove non-photon candidates from input point cloud",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.analysis_config)
    feature_config = parse_feature_config(config)
    invisible_features = tuple(str(key) for key in config["Normalization"].get("Invisible"))
    remove_neutral_non_photon = args.remove_neutral_non_photon
    samples = attach_sample_total_intial_events(
        parse_samples(config, args.samples)
    )

    selected_keys = {sample.key for sample in samples}
    luminosity = infer_luminosity(samples)
    classification_lookup = build_classification_lookup(config, selected_keys)
    sample_index_lookup = {sample.key: index for index, sample in enumerate(samples)}

    # configure multi-cpu jobs
    jobs: list[tuple[Sample, str, int]] = []
    for sample in samples:
        files = sample.files
        for file_index, file_path in enumerate(files):
            jobs.append((sample, file_path, file_index))

    max_particles = 0
    max_scan_columns = ["nprong", "Part_fourMomentum_fCoordinates_fT", "Part_pdgId", "Part_charge"]
    for sample in samples:
        if sample.is_data:
            continue
        for file_path in sample.files:
            parquet = pq.ParquetFile(file_path)
            schema_names = {field.name for field in parquet.schema_arrow}
            scan_columns = [column for column in max_scan_columns if column in schema_names]
            row_offset = 0
            for record_batch in parquet.iter_batches(batch_size=args.batch_size, columns=scan_columns):
                events = ak.from_arrow(record_batch)
                mask = event_preselection_mask(events)
                if np.any(mask):
                    selected = events[mask]
                    input_part_mask = build_input_particle_mask(selected, remove_neutral_non_photon)
                    num_particles = ak.sum(input_part_mask, axis=1)
                    print(num_particles)
                    print(max(num_particles))
                    if np.max(num_particles) > max_particles:
                        max_particles = num_particles
                row_offset += len(events)
    print(f"[ml_pipeline_lite] output_dir={output_dir}")
    print(f"[ml_pipeline_lite] samples={[sample.key for sample in samples]}")
    print(f"[ml_pipeline_lite] class_labels={list(classification_lookup.class_labels)}")
    print(f"[ml_pipeline_lite] jobs={len(jobs)} workers={args.num_workers}")
    print(f"[ml_pipeline_lite] invisible_features={list(invisible_features)}")
    print(f"[ml_pipeline_lite] remove_neutral_non_photon={remove_neutral_non_photon}")
    print(f"[ml_pipeline_lite] max_particles={max_particles}")


if __name__ == "__main__":
    main()