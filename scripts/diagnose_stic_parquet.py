#!/usr/bin/env python3
"""Read STIC values directly from on-disk EveNet training parquet, before Ray/Torch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import yaml

TARGETS = ('Part_sticShowerEnergy', 'Part_sticShowerTheta', 'Part_sticShowerPhi',
           'Part_sticNumTowers', 'Part_sticChargedTag')


def scan_file(path, feature_names, *, batch_rows=8192, threshold=1.e6, max_examples=10):
    """Project only STIC, masks and source identifiers; never normalize or clamp."""
    if batch_rows < 1 or threshold <= 0 or max_examples < 0:
        raise ValueError('Invalid scan limits')
    if any(feature_names.count(name) != 1 for name in TARGETS):
        raise ValueError('Expected unique STIC names in policy feature mapping')
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    slots = sorted({int(m[1]) for name in names if (m := re.fullmatch(r'x:(\d+):(\d+)', name))})
    if not slots:
        raise ValueError(f'{path}: not flattened EveNet training parquet (no x:slot:feature columns)')
    columns = []
    mapping = []
    for slot in slots:
        mask = f'x_mask:{slot}'
        if mask not in names:
            raise ValueError(f'{path}: missing particle mask {mask}')
        columns.append(mask)
        for feature in TARGETS:
            column = f'x:{slot}:{feature_names.index(feature)}'
            if column not in names:
                raise ValueError(f'{path}: missing {column}')
            columns.append(column)
            mapping.append((slot, feature, column, mask))
    ids = [name for name in ('source_sample_index', 'source_event_index', 'source_event_key') if name in names]
    columns.extend(ids)
    stats = {name: dict(valid=0, nonfinite=0, large=0, nonbinary=0, subnormal=0,
                       min=None, max=None) for name in TARGETS}
    examples = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        arrays = {name: batch.column(name).to_numpy(zero_copy_only=False) for name in columns}
        for slot, feature, column, mask in mapping:
            valid = np.asarray(arrays[mask]) > .5
            values = arrays[column]
            finite = np.isfinite(values)
            large = finite & (np.abs(values) > threshold)
            nonbinary = finite & (values != 0) & (values != 1) if feature.endswith('ChargedTag') else np.zeros(len(values), dtype=bool)
            subnormal = finite & (values != 0) & (np.abs(values) < np.finfo(np.float32).tiny)
            stat = stats[feature]
            stat['valid'] += int(valid.sum())
            for key, test in [('nonfinite', ~finite), ('large', large), ('nonbinary', nonbinary), ('subnormal', subnormal)]:
                stat[key] += int((valid & test).sum())
            good = values[valid & finite]
            if good.size:
                low, high = float(good.min()), float(good.max())
                stat['min'] = low if stat['min'] is None else min(stat['min'], low)
                stat['max'] = high if stat['max'] is None else max(stat['max'], high)
            flagged = valid & (~finite | large | nonbinary | subnormal)
            for row in np.flatnonzero(flagged)[:max(0, max_examples-len(examples))]:
                value = values[row]
                item = {'file_row': offset + int(row), 'particle_slot': slot, 'feature': feature,
                        'column': column, 'arrow_type': str(parquet.schema_arrow.field(column).type),
                        'value': float(value) if np.isfinite(value) else str(value),
                        'stic_values': {n: (float(arrays[f'x:{slot}:{feature_names.index(n)}'][row])
                                          if np.isfinite(arrays[f'x:{slot}:{feature_names.index(n)}'][row])
                                          else str(arrays[f'x:{slot}:{feature_names.index(n)}'][row])) for n in TARGETS},
                        'source_ids': {name: str(arrays[name][row]) for name in ids}}
                examples.append(item)
        offset += batch.num_rows
    return {'file': str(path), 'rows': offset, 'statistics': stats, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--policy-report', type=Path, required=True, help='Provides actual checkpoint feature ordering')
    parser.add_argument('--parquet-dir', type=Path, help='Defaults to runtime platform.data_parquet_dir')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report, no overwrite')
    parser.add_argument('--batch-rows', type=int, default=8192)
    parser.add_argument('--large-threshold', type=float, default=1.e6)
    parser.add_argument('--max-files', type=int, help='Optional bounded scan; omitted scans all files')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new report path')
    if args.max_files is not None and args.max_files < 1:
        parser.error('--max-files must be positive')
    with args.runtime.open() as stream:
        runtime = yaml.safe_load(stream)
    with args.policy_report.open() as stream:
        policy = json.load(stream)
    root = args.parquet_dir or Path(runtime['platform']['data_parquet_dir'])
    files = [root] if root.is_file() else sorted(root.rglob('*.parquet'))
    if not files:
        parser.error(f'No parquet files found in {root}')
    selected = files if args.max_files is None else files[:args.max_files]
    report = {'dataset': str(root), 'feature_mapping_source': str(args.policy_report),
              'files_found': len(files), 'files_selected': len(selected),
              'large_threshold': args.large_threshold, 'results': [], 'errors': [],
              'note': 'Reads stored EveNet x columns directly, not original detector parquet. Nonbinary tag/subnormal counts are flags, not proof of corruption. Schema mapping uses the policy report; source metadata still needs verification. Matching values alone do not prove captured event identity.'}
    for index, path in enumerate(selected, 1):
        try:
            result = scan_file(path, policy['raw_sequential_feature_names'],
                               batch_rows=args.batch_rows, threshold=args.large_threshold)
            report['results'].append(result)
            counts = {n: {k: v for k,v in s.items() if k in ('large','nonfinite','nonbinary','subnormal')}
                      for n,s in result['statistics'].items()}
            print(f'[{index}/{len(selected)}] {path.name} rows={result["rows"]} {counts}', flush=True)
        except Exception as exc:
            report['errors'].append({'file': str(path), 'error': f'{type(exc).__name__}: {exc}'})
            print(f'ERROR {path}: {exc}', flush=True)
    report['complete_scan'] = not report['errors'] and len(selected) == len(files)
    totals = {n: {key: sum(r['statistics'][n][key] for r in report['results'])
                  for key in ('valid','large','nonfinite','nonbinary','subnormal')} for n in TARGETS}
    report['totals'] = totals
    report['suspicious_values_found'] = any(s[k] for s in totals.values() for k in ('large','nonfinite','nonbinary','subnormal'))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete_scan': report['complete_scan'], 'totals': totals, 'errors': report['errors']}, indent=2))
    print(f'Report: {args.output}')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
