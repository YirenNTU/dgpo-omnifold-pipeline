#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

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


def parse_device_list(raw: str | None, *, use_gpu: bool) -> list[str]:
    if raw is None:
        if use_gpu and torch.cuda.is_available():
            count = torch.cuda.device_count()
            return [f"cuda:{index}" for index in range(count)] or ["cpu"]
        return ["cpu"]

    value = raw.strip()
    if not value:
        return ["cpu"]
    if value.lower() == "auto":
        if use_gpu and torch.cuda.is_available():
            count = torch.cuda.device_count()
            return [f"cuda:{index}" for index in range(count)] or ["cpu"]
        return ["cpu"]
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_worker_devices(
    *,
    requested_devices: str | None,
    requested_workers: int | None,
    explicit_device: str | None,
    use_gpu: bool,
) -> list[str]:
    if explicit_device and requested_devices is None and requested_workers is None:
        return [explicit_device]

    devices = parse_device_list(requested_devices, use_gpu=use_gpu)
    if explicit_device and requested_devices is None:
        devices = [explicit_device]

    if requested_workers is None or requested_workers <= 0:
        return devices

    if not devices:
        return ["cpu"]

    if len(devices) >= requested_workers:
        return devices[:requested_workers]

    if all(device.startswith("cuda:") for device in devices):
        return devices

    repeats = math.ceil(requested_workers / len(devices))
    return (devices * repeats)[:requested_workers]


def augment_file(
    *,
    model: torch.nn.Module,
    device: torch.device,
    input_path: Path,
    output_path: Path,
    shape_metadata: dict[str, list[int]],
    batch_size: int,
    progress_desc: str | None = None,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    parquet = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    event_token_shape: tuple[int, ...] | None = None
    object_token_shape: tuple[int, ...] | None = None
    total_batches = parquet.metadata.num_rows // batch_size
    if parquet.metadata.num_rows % batch_size:
        total_batches += 1

    with torch.no_grad():
        batch_iterator = parquet.iter_batches(batch_size=batch_size)
        if progress_desc is not None:
            batch_iterator = tqdm(
                batch_iterator,
                total=total_batches,
                desc=progress_desc,
                leave=False,
                dynamic_ncols=True,
            )
        for record_batch in batch_iterator:
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


def augment_files_on_device(
    *,
    config_path: str,
    checkpoint_path: str,
    input_paths: list[str],
    output_paths: list[str],
    shape_metadata: dict[str, list[int]],
    batch_size: int,
    device_name: str,
) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    device = torch.device(device_name)
    bundle = load_evenet_model_for_dgpo(
        config_path=Path(config_path),
        checkpoint_path=Path(checkpoint_path),
        device=device,
    )
    model = bundle.model.eval()
    results: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for input_path, output_path in zip(input_paths, output_paths, strict=True):
        progress_desc = None
        if len(input_paths) == 1:
            progress_desc = f"{Path(input_path).parent.name}/{Path(input_path).name} [{device_name}]"
        results.append(
            augment_file(
                model=model,
                device=device,
                input_path=Path(input_path),
                output_path=Path(output_path),
                shape_metadata=shape_metadata,
                batch_size=batch_size,
                progress_desc=progress_desc,
            )
        )
    return results


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
        if path.name == "shape_metadata.json" or path.suffix == ".parquet":
            continue
        shutil.copy2(path, output_dir / path.name)

    updated_shape = dict(shape_metadata)
    updated_shape["event_token"] = [int(v) for v in event_token_shape]
    updated_shape["object_token"] = [int(v) for v in object_token_shape]
    (output_dir / "shape_metadata.json").write_text(json.dumps(updated_shape))


def validate_augmented_split(output_dir: Path) -> None:
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No augmented parquet files found in {output_dir}")
    schema = pq.ParquetFile(parquet_paths[0]).schema_arrow
    column_names = set(schema.names)
    if not any(name == "event_token" or name.startswith("event_token:") for name in column_names):
        raise RuntimeError(
            f"Augmented split {output_dir} is missing event_token columns after export."
        )
    if not any(name == "object_token" or name.startswith("object_token:") for name in column_names):
        raise RuntimeError(
            f"Augmented split {output_dir} is missing object_token columns after export."
        )


def augment_split(
    *,
    config_path: Path,
    checkpoint_path: Path,
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    worker_devices: list[str],
) -> None:
    shape_metadata = read_shape_metadata(input_dir)
    event_shape: tuple[int, ...] | None = None
    object_shape: tuple[int, ...] | None = None
    parquet_paths = sorted(input_dir.glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found in {input_dir}")

    output_paths = [output_dir / parquet_path.name for parquet_path in parquet_paths]
    device_names = worker_devices or ["cpu"]
    num_workers = min(len(device_names), len(parquet_paths))
    device_names = device_names[:num_workers]

    if num_workers <= 1:
        results = augment_files_on_device(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            input_paths=[str(path) for path in parquet_paths],
            output_paths=[str(path) for path in output_paths],
            shape_metadata=shape_metadata,
            batch_size=batch_size,
            device_name=device_names[0],
        )
    else:
        chunks_input: list[list[str]] = [[] for _ in range(num_workers)]
        chunks_output: list[list[str]] = [[] for _ in range(num_workers)]
        for index, (input_path, output_path) in enumerate(zip(parquet_paths, output_paths, strict=True)):
            bucket = index % num_workers
            chunks_input[bucket].append(str(input_path))
            chunks_output[bucket].append(str(output_path))

        ctx = get_context("spawn")
        futures = []
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            for device_name, input_chunk, output_chunk in zip(device_names, chunks_input, chunks_output, strict=True):
                if not input_chunk:
                    continue
                futures.append(
                    executor.submit(
                        augment_files_on_device,
                        config_path=str(config_path),
                        checkpoint_path=str(checkpoint_path),
                        input_paths=input_chunk,
                        output_paths=output_chunk,
                        shape_metadata=shape_metadata,
                        batch_size=batch_size,
                        device_name=device_name,
                    )
                )

            results = []
            progress = tqdm(
                futures,
                total=len(futures),
                desc=f"augment {input_dir.name}",
                leave=True,
                dynamic_ncols=True,
            )
            for future in progress:
                results.extend(future.result())

    for _, file_event_shape, file_object_shape in results:
        event_shape = file_event_shape if event_shape is None else event_shape
        object_shape = file_object_shape if object_shape is None else object_shape
        if event_shape != file_event_shape or object_shape != file_object_shape:
            raise ValueError("Inconsistent token shapes across parquet files.")

    if event_shape is None or object_shape is None:
        raise RuntimeError(f"No rows were processed while augmenting {input_dir}")

    copy_sidecars(
        input_dir=input_dir,
        output_dir=output_dir,
        shape_metadata=shape_metadata,
        event_token_shape=event_shape,
        object_token_shape=object_shape,
    )
    validate_augmented_split(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment EveNet parquet splits with event_token and object_token.")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="Base directory containing split subdirectories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Base directory where augmented splits are written.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None, help="Single-device fallback, for example cuda:0 or cpu.")
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated device list for parallel token export, or 'auto' to use all visible GPUs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel export workers. Defaults to the number of selected devices.",
    )
    parser.add_argument(
        "--use-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether token export should target GPUs when available.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.train_config.expanduser().resolve()
    worker_devices = resolve_worker_devices(
        requested_devices=args.devices,
        requested_workers=args.num_workers,
        explicit_device=args.device,
        use_gpu=args.use_gpu,
    )
    print(
        f"[augment_event_tokens] using {len(worker_devices)} worker(s) on devices: {', '.join(worker_devices)}",
        flush=True,
    )

    split_iterator = tqdm(args.splits, desc="AD token export", leave=True, dynamic_ncols=True)
    for split in split_iterator:
        split_iterator.set_postfix_str(split)
        augment_split(
            config_path=config_path,
            checkpoint_path=checkpoint,
            input_dir=(args.input_dir / split).resolve(),
            output_dir=(args.output_dir / split).resolve(),
            batch_size=args.batch_size,
            worker_devices=worker_devices,
        )


if __name__ == "__main__":
    main()
