import numpy as np
import torch
import time
from collections.abc import Mapping

from evenet.dataset.types import Batch, Source, AssignmentTargets


def flatten_column_name(base: str, indices: tuple[int, ...]) -> str:
    return base if not indices else base + ":" + ":".join(str(index) for index in indices)


def flatten_dict(
    data: Mapping[str, object],
    parent_key: str = "",
    sep: str = ":",
) -> dict:
    """Flatten nested mappings and arrays into EveNet parquet column names.

    Array-like values are flattened into ``base:i:j`` columns so they align with
    the original EveNet parquet/``shape_metadata.json`` convention.
    """
    items: dict[str, object] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(flatten_dict(value, parent_key=new_key, sep=sep))
        else:
            array = np.asarray(value)
            if array.ndim <= 1:
                items[new_key] = value
                continue
            trailing_shape = array.shape[1:]
            for index in np.ndindex(trailing_shape):
                items[flatten_column_name(new_key, tuple(int(i) for i in index))] = array[(slice(None),) + index]
    return items


def _stack_unflattened_columns(
    flat_batch: Mapping[str, np.ndarray],
    base: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for index in np.ndindex(shape):
        column = flatten_column_name(base, tuple(int(i) for i in index))
        if column not in flat_batch:
            raise KeyError(f"Missing flattened column {column!r} for base={base!r} shape={shape}.")
        columns.append(np.asarray(flat_batch[column]))

    if not columns:
        raise ValueError(f"Cannot reconstruct {base!r}: empty shape metadata {shape}.")

    num_events = int(np.asarray(columns[0]).shape[0])
    stacked = np.stack(columns, axis=1)
    return stacked.reshape((num_events,) + shape)


def unflatten_dict(
    flat_batch: Mapping[str, np.ndarray],
    shape_metadata: Mapping[str, list[int] | tuple[int, ...]],
    drop_column_prefix: list[str] | str | None = None,
) -> dict[str, np.ndarray]:
    """Reconstruct structured EveNet tensors from flattened parquet columns."""
    if isinstance(drop_column_prefix, str):
        drop_prefixes = [drop_column_prefix]
    else:
        drop_prefixes = list(drop_column_prefix or [])

    output: dict[str, np.ndarray] = {}
    consumed: set[str] = set()

    for base, raw_shape in shape_metadata.items():
        if any(base.startswith(prefix) for prefix in drop_prefixes):
            continue
        shape = tuple(int(item) for item in raw_shape)
        if len(shape) == 0:
            if base in flat_batch:
                output[base] = np.asarray(flat_batch[base])
                consumed.add(base)
            continue
        output[base] = _stack_unflattened_columns(flat_batch, base, shape)
        consumed.update(flatten_column_name(base, tuple(int(i) for i in index)) for index in np.ndindex(shape))

    for key, value in flat_batch.items():
        if key in consumed:
            continue
        if any(key.startswith(prefix) for prefix in drop_prefixes):
            continue
        output[key] = np.asarray(value)

    return output


def process_event_batch_old(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ####################################
    # Input: Point Cloud and Conditions
    ####################################
    point_cloud = batch['sources-0-data']
    point_cloud_mask = batch['sources-0-mask']

    conditions = batch['sources-1-data']
    conditions_mask = batch['sources-1-mask']

    point_cloud = np.array([np.vstack(event) for event in point_cloud], dtype=np.float32)
    point_cloud_mask = np.array([np.vstack(event).ravel() for event in point_cloud_mask], dtype=np.float32)

    conditions = np.array([np.vstack(event) for event in conditions], dtype=np.float32)
    conditions = conditions.reshape(conditions.shape[0], -1)
    conditions_mask = np.array([np.vstack(event).ravel() for event in conditions_mask], dtype=np.float32)

    ########################################
    # Target: Classification and Regression
    ########################################
    classification_target = batch['classification-EVENT/signal']
    regression_target = {k: batch[k] for k in batch if k.startswith('regression-') and k.endswith('-data')}
    regression_mask = {k: batch[k] for k in batch if k.startswith('regression-') and k.endswith('-mask')}
    regression_values = np.array([np.vstack(event).ravel() for event in regression_target.values()], dtype=np.float32)
    regression_values = regression_values.swapaxes(0, 1)
    regression_mask = np.array([np.vstack(event).ravel() for event in regression_mask.values()], dtype=np.float32)
    regression_mask = regression_mask.swapaxes(0, 1)

    ########################################
    # Target: Assignment
    ########################################
    # Step 1: Select and sort assignment index keys
    assignment_index_keys = sorted(
        k for k in batch if k.startswith('assignments-') and k.endswith('-indices')
    )

    assignment_mask = np.array([
        np.vstack(batch[k]).ravel()
        for k in batch if k.startswith('assignments-') and k.endswith('-mask')
    ], dtype=np.float32).swapaxes(0, 1)

    # Step 2: Collect index arrays and determine max number of children
    assignment_index_list = [np.vstack(batch[k]) for k in assignment_index_keys]
    max_num_children = max(indices.shape[1] for indices in assignment_index_list)

    # Step 3: Pad each array to (num_events, max_num_children) with -2 (temporary marker)
    assignment_index_padded = [
        np.pad(indices, ((0, 0), (0, max_num_children - indices.shape[1])), constant_values=-2)
        for indices in assignment_index_list
    ]

    # Step 4: Stack into a single array of shape (num_events, num_assignment_types, max_num_children)
    assignment_indices = np.stack(assignment_index_padded, axis=1, dtype=np.float32)

    # Step 5: Create assignment mask: True where original, False where padded
    assignment_indices_mask = (assignment_indices != -2)

    # Step 6: Replace padding marker -2 with actual padding value -1
    assignment_indices[assignment_indices == -2] = -1

    ########################################
    # Target: Generation
    ########################################
    num_vectors = batch['num_vectors']
    num_sequential_vectors = batch['num_sequential_vectors']

    batch_out = {
        'x': point_cloud,
        'x_mask': point_cloud_mask,
        'conditions': conditions,
        'conditions_mask': conditions_mask,

        'classification': classification_target,
        'regression': regression_values,
        'regression_mask': regression_mask,

        'num_vectors': num_vectors,
        'num_sequential_vectors': num_sequential_vectors,

        'assignment_indices': assignment_indices,
        'assignment_indices_mask': assignment_indices_mask,
        'assignment_mask': assignment_mask,
    }

    return {
        k: v.astype(np.float32) for k, v in batch_out.items()
    }


def process_event_batch(batch: dict[str, np.ndarray], shape_metadata: dict, unflatten, drop_column_prefix: str = None) -> dict[str, np.ndarray]:
    return unflatten(batch, shape_metadata, drop_column_prefix=drop_column_prefix)


def convert_batch_to_torch_tensor(batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """
    Convert a batch of data from numpy arrays to torch tensors.
    :param batch: Batch of data as a dictionary with numpy arrays.
    :return: Batch of data as a dictionary with torch tensors.
    """
    return {k: torch.tensor(v) for k, v in batch.items()}
