#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENET_DGPO_ROOT = REPO_ROOT / "evenet_dgpo"
if str(EVENET_DGPO_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENET_DGPO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evenet_dgpo.evenet.dataset.preprocess import (
    flatten_column_name,
    unflatten_dict,
)
from evenet_dgpo.RL.DGPO_neutrino.model_utils import load_evenet_model_for_dgpo


def read_shape_metadata(input_dir: Path) -> dict[str, list[int]]:
    path = input_dir / "shape_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"shape_metadata.json not found in {input_dir}")
    return json.loads(path.read_text())


def to_torch_batch(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = torch.from_numpy(np.asarray(value))
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        output[key] = tensor.to(device=device)
    return output


def load_flat_batch(record_batch: pa.RecordBatch) -> dict[str, np.ndarray]:
    table = pa.Table.from_batches([record_batch])
    pydict = table.to_pydict()
    return {key: np.asarray(value) for key, value in pydict.items()}


def flatten_token_array(base: str, values: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    if values.ndim == 2:
        for feature_index in range(values.shape[1]):
            output[flatten_column_name(base, (feature_index,))] = values[:, feature_index]
        return output
    if values.ndim == 3:
        for slot_index in range(values.shape[1]):
            for feature_index in range(values.shape[2]):
                output[flatten_column_name(base, (slot_index, feature_index))] = values[:, slot_index, feature_index]
        return output
    raise ValueError(f"Unsupported token array shape for {base}: {values.shape}")


def augment_file(
    *,
    model: torch.nn.Module,
    device: torch.device,
    input_path: Path,
    output_path: Path,
    shape_metadata: dict[str, list[int]],
    batch_size: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    parquet = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    event_token_shape: tuple[int, ...] | None = None
    object_token_shape: tuple[int, ...] | None = None

    with torch.no_grad():
        for record_batch in parquet.iter_batches(batch_size=batch_size):
            flat_batch = load_flat_batch(record_batch)
            batch_np = unflatten_dict(flat_batch, shape_metadata, drop_column_prefix=None)
            batch_torch = to_torch_batch(batch_np, device=device)
            time = torch.zeros(batch_torch["x"].shape[0], device=device, dtype=torch.float32)
            outputs = model(
                batch_torch,
                time=time,
                schedules=[("deterministic", True)],
            )
            event_token = outputs.get("event_token")
            object_token = outputs.get("object_token")
            if event_token is None or object_token is None:
                raise RuntimeError("Model did not produce event_token/object_token during deterministic forward.")

            event_token_np = event_token.detach().cpu().numpy().astype(np.float32, copy=False)
            object_token_np = object_token.detach().cpu().numpy().astype(np.float32, copy=False)
            event_token_shape = tuple(event_token_np.shape[1:])
            object_token_shape = tuple(object_token_np.shape[1:])

            augmented = dict(flat_batch)
            augmented.update(flatten_token_array("event_token", event_token_np))
            augmented.update(flatten_token_array("object_token", object_token_np))
            table = pa.table(augmented)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += table.num_rows

    if writer is not None:
        writer.close()

    if event_token_shape is None or object_token_shape is None:
        raise RuntimeError(f"No rows were processed while augmenting {input_path}")

    return total_rows, event_token_shape, object_token_shape


def copy_sidecars(
    *,
    input_dir: Path,
    output_dir: Path,
    shape_metadata: dict[str, list[int]],
    event_token_shape: tuple[int, ...],
    object_token_shape: tuple[int, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        if path.name == "shape_metadata.json":
            continue
        shutil.copy2(path, output_dir / path.name)

    updated_shape = dict(shape_metadata)
    updated_shape["event_token"] = [int(v) for v in event_token_shape]
    updated_shape["object_token"] = [int(v) for v in object_token_shape]
    (output_dir / "shape_metadata.json").write_text(json.dumps(updated_shape))


def augment_split(
    *,
    config_path: Path,
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    shape_metadata = read_shape_metadata(input_dir)
    bundle = load_evenet_model_for_dgpo(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model = bundle.model.eval()

    event_shape: tuple[int, ...] | None = None
    object_shape: tuple[int, ...] | None = None
    for parquet_path in sorted(input_dir.glob("*.parquet")):
        out_path = output_dir / parquet_path.name
        _, file_event_shape, file_object_shape = augment_file(
            model=model,
            device=device,
            input_path=parquet_path,
            output_path=out_path,
            shape_metadata=shape_metadata,
            batch_size=batch_size,
        )
        event_shape = file_event_shape if event_shape is None else event_shape
        object_shape = file_object_shape if object_shape is None else object_shape
        if event_shape != file_event_shape or object_shape != file_object_shape:
            raise ValueError("Inconsistent token shapes across parquet files.")

    if event_shape is None or object_shape is None:
        raise RuntimeError(f"No parquet files found in {input_dir}")

    copy_sidecars(
        input_dir=input_dir,
        output_dir=output_dir,
        shape_metadata=shape_metadata,
        event_token_shape=event_shape,
        object_token_shape=object_shape,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment EveNet parquet splits with event_token and object_token.")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="Base directory containing split subdirectories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Base directory where augmented splits are written.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.train_config.expanduser().resolve()

    for split in args.splits:
        augment_split(
            config_path=config_path,
            checkpoint_path=checkpoint,
            input_dir=(args.input_dir / split).resolve(),
            output_dir=(args.output_dir / split).resolve(),
            batch_size=args.batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
