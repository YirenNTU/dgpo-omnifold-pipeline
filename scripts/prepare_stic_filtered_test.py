#!/usr/bin/env python3
"""Make an isolated event-filtered dataset and cold-start diagnostic runtime.

Only exclude events with nonfinite or abs(value)>1e6 STIC tower/tag values
on valid particles. Preserve source data, surviving rows, and normalization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

FIELDS = ('Part_sticNumTowers', 'Part_sticChargedTag')


def exclusion_mask(batch, feature_names, threshold=1.e6):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be finite and positive')
    if any(feature_names.count(name) != 1 for name in FIELDS):
        raise ValueError('STIC fields missing or ambiguous in checkpoint feature ordering')
    names = set(batch.schema.names)
    slots = sorted({int(m[1]) for name in names if (m := re.fullmatch(r'x:(\d+):(\d+)', name))})
    if not slots:
        raise ValueError('Expected flattened training parquet x:slot:feature columns')
    rejected = np.zeros(batch.num_rows, dtype=bool)
    reasons = {}
    for slot in slots:
        mask_name = f'x_mask:{slot}'
        if mask_name not in names:
            raise ValueError(f'Missing particle mask {mask_name}')
        mask_column = batch.column(mask_name)
        if mask_column.null_count:
            raise ValueError(f'Null particle mask {mask_name}')
        valid = mask_column.to_numpy(zero_copy_only=False) > .5
        for field in FIELDS:
            column = f'x:{slot}:{feature_names.index(field)}'
            if column not in names:
                raise ValueError(f'Missing STIC field {column}')
            values = batch.column(column).to_numpy(zero_copy_only=False)
            finite = np.isfinite(values)
            bad = valid & (~finite | (np.abs(values) > threshold))
            rejected |= bad
            for row in np.flatnonzero(bad):
                value = values[row]
                reasons.setdefault(int(row), []).append({
                    'particle_slot': slot, 'field': field, 'column': column,
                    'value': float(value) if finite[row] else str(value)})
    return rejected, reasons


def filter_file(source, destination, feature_names, audit, *, batch_rows=8192, threshold=1.e6):
    if destination.exists():
        raise FileExistsError(destination)
    parquet = pq.ParquetFile(source)
    count = removed = 0
    with pq.ParquetWriter(destination, parquet.schema_arrow, compression='snappy') as writer:
        for batch in parquet.iter_batches(batch_size=batch_rows):
            rejected, reasons = exclusion_mask(batch, feature_names, threshold)
            identifiers = [key for key in ('source_sample_index', 'source_event_index', 'source_event_key')
                           if key in batch.schema.names]
            for row, flags in reasons.items():
                audit.write(json.dumps({'source_file': str(source), 'file_row': count + row,
                    'source_ids': {key: str(batch.column(key)[row].as_py()) for key in identifiers},
                    'reasons': flags}, allow_nan=False) + '\n')
            kept = batch.filter(pa.array(~rejected))
            if kept.num_rows:
                writer.write_batch(kept)
            count += batch.num_rows
            removed += int(rejected.sum())
    return {'source': str(source), 'output': str(destination), 'rows_in': count,
            'events_removed': removed, 'rows_out': count-removed}


def filtered_runtime(original, root, checkpoint):
    config = deepcopy(original)
    config['platform']['data_parquet_dir'] = str(root / 'train')
    config['platform']['data_parquet_val_dir'] = str(root / 'train')
    training = config['options']['Training']
    training['model_checkpoint_load_path'] = str(checkpoint)
    training['pretrain_model_load_path'] = None
    training['model_checkpoint_save_path'] = str(root / 'checkpoints')
    training['EMA']['replace_model_after_load'] = False
    training['EMA']['use_for_generation'] = False
    config['reward_config']['omnifold'].update(backbone_checkpoint=str(checkpoint),
                                             bundle_file=None, bootstrap_in_dgpo=True)
    config['dgpo'].update(auto_resume_from_last=False, auto_resume_best_source_checkpoint_dir=None,
                         auto_resume_fallback_checkpoint_path=None, best_source_start_new_experiment=False,
                         checkpoint_load_mode='weights_only')
    config['dgpo']['adaptive_omnifold']['recalibration']['refit_once_on_resume'] = False
    config['logger']['wandb'].update(run_name=root.name, resume='never')
    config['logger']['wandb'].pop('id', None)
    config['logger']['local'].update(save_dir=str(root / 'logs'), name=root.name, version='stic_filtered_test')
    config['nersc']['ray']['results_dir'] = str(root / 'ray_results')
    config['nersc']['execution'].pop('command', None)
    config['stic_filter_test'] = {'normalization': 'unchanged from source runtime',
                                  'manifest': str(root / 'filter_manifest.json')}
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True, help='Merged runtime from the original failed capture')
    parser.add_argument('--policy-report', type=Path, required=True, help='Pins checkpoint and feature ordering')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; refuses overwrite')
    parser.add_argument('--batch-rows', type=int, default=8192)
    args = parser.parse_args()
    if args.batch_rows < 1:
        parser.error('--batch-rows must be positive')
    with args.runtime.open() as stream:
        runtime = yaml.safe_load(stream)
    with args.policy_report.open() as stream:
        policy = json.load(stream)
    source = Path(runtime['platform']['data_parquet_dir']).resolve(strict=True)
    checkpoint = Path(policy['checkpoint']).resolve(strict=True)
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != policy['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since diffusion test; refusing a confounded comparison')
    root = args.output_dir.resolve()
    if root == source or source in root.parents or root in source.parents:
        parser.error('Output must be separate from source dataset')
    files = [source] if source.is_file() else sorted(source.rglob('*.parquet'))
    if not files:
        parser.error(f'No parquet files under {source}')
    root.mkdir(parents=True, exist_ok=False)
    data_dir = root / 'train'
    data_dir.mkdir()
    results = []
    with (root / 'removed_events.jsonl').open('x') as audit:
        for index, path in enumerate(files):
            result = filter_file(path, data_dir / f'part-{index:05d}.parquet',
                                 policy['raw_sequential_feature_names'], audit, batch_rows=args.batch_rows)
            results.append(result)
            print(json.dumps(result), flush=True)
    kept = sum(row['rows_out'] for row in results)
    if not kept:
        raise ValueError('No surviving events; no runnable runtime will be written')
    copied_sidecars = []
    for name in ('shape_metadata.json', 'normalization.pt'):
        sidecar = (source if source.is_dir() else source.parent) / name
        if sidecar.is_file():
            shutil.copy2(sidecar, data_dir / name)
            copied_sidecars.append(name)
    manifest = {'complete': True, 'source': str(source), 'rows_in': sum(r['rows_in'] for r in results),
                'rows_out': kept, 'events_removed': sum(r['events_removed'] for r in results),
                'rule': 'Exclude whole event if any valid particle has nonfinite or abs(value)>1e6 in sticNumTowers/sticChargedTag',
                'subnormal_only_events': 'retained; not established as the crash trigger',
                'normalization': 'preserved, not recomputed', 'sidecars_copied': copied_sidecars,
                'source_runtime': str(args.runtime.resolve()), 'policy_report': str(args.policy_report.resolve()),
                'files': results}
    with (root / 'filter_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    updated = filtered_runtime(runtime, root, checkpoint)
    with (root / 'runtime.yaml').open('x') as stream:
        yaml.safe_dump(updated, stream, sort_keys=False)
    print(f'READY: removed {manifest["events_removed"]} events; retained {kept}. Original unchanged.')
    print(f'Runtime: {root / "runtime.yaml"}')


if __name__ == '__main__':
    main()
