"""Read-only inventory of Ztautau analyzer assignments; no physics certification."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


FIELDS = ('event_category', 'lead_a_pdgId', 'lead_b_pdgId',
          'analyzing_power_a', 'analyzing_power_b')
IDS = ('source_sample_index', 'source_event_key', 'source_file_index', 'source_event_index')
# Explicit category labels from config/analysis.yaml; not inferred from observed kappas.
CHANNELS = {11: ('pi','pi'), 12: ('pi','rho'), 21: ('rho','pi'),
            13: ('pi','e'), 31: ('e','pi'), 14: ('pi','mu'), 41: ('mu','pi'),
            23: ('rho','e'), 32: ('e','rho'), 42: ('mu','rho'), 24: ('rho','mu'),
            22: ('rho','rho'), 33: ('e','e'), 44: ('mu','mu'),
            34: ('e','mu'), 43: ('mu','e')}
EXPECTED = {'pi': 1., 'rho': .41, 'e': -.33, 'mu': -.34}
P4 = tuple(f'truth_{leg}_visible_{v}' for leg in ('a','b') for v in ('E','px','py','pz'))


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def scalar(value):
    if value is None:
        return '<null>'
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    return '<non-scalar>'


def valid_power(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 < abs(value) <= 1)


def inspect(path, batch_size=32768):
    path = Path(path)
    files = sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'No parquet files in {path}')
    combinations, missing_counts, null_counts, invalid_counts = (Counter() for _ in range(4))
    kappas = {leg: Counter() for leg in ('a', 'b')}
    assignments = {}
    examples, schemas = [], []
    rows, negative_products = 0, 0
    checks = Counter()
    masses, mass_status = {}, {}
    for file in files:
        parquet = pq.ParquetFile(file)
        names = parquet.schema_arrow.names
        present = [key for key in (*FIELDS, *IDS, 'analyzing_power', *P4) if key in names]
        relevant = [key for key in names if any(token in key.lower() for token in
                    ('pdg', 'charge', 'prong', 'decay', 'category', 'analyzing', 'visible', 'truth_tau'))]
        schemas.append(dict(file=str(file), rows=parquet.metadata.num_rows,
                            missing_fields=[key for key in FIELDS if key not in names],
                            relevant_columns={key: str(parquet.schema_arrow.field(key).type) for key in relevant}))
        for key in FIELDS:
            if key not in names:
                missing_counts[key] += parquet.metadata.num_rows
        offset = 0
        # If all relevant fields are absent, one arbitrary column preserves row accounting.
        for batch in parquet.iter_batches(batch_size=batch_size, columns=present or names[:1]):
            data = batch.to_pydict()
            for i in range(batch.num_rows):
                raw = {key: data[key][i] if key in present else '<missing>' for key in FIELDS}
                values = {key: scalar(value) for key, value in raw.items()}
                combinations[tuple(values[key] for key in FIELDS)] += 1
                reasons = []
                category = values['event_category']
                channel = CHANNELS.get(category)
                if channel is None:
                    checks['unknown_category_rows'] += 1
                else:
                    for leg, species in zip(('a','b'), channel):
                        value = raw[f'analyzing_power_{leg}']
                        if valid_power(value):
                            checks['category_leg_checks'] += 1
                            if not math.isclose(value, EXPECTED[species], rel_tol=0, abs_tol=1e-6):
                                checks['category_leg_mismatches'] += 1
                                reasons.append(f'{leg}: expected {EXPECTED[species]} for {species}; review convention')
                        else:
                            checks['category_leg_uncheckable'] += 1
                product = data['analyzing_power'][i] if 'analyzing_power' in data else None
                if all(valid_power(raw[f'analyzing_power_{leg}']) for leg in ('a','b')) and finite_number(product):
                    checks['product_checks'] += 1
                    expected_product = raw['analyzing_power_a'] * raw['analyzing_power_b']
                    if not math.isclose(product, expected_product, rel_tol=0, abs_tol=1e-6):
                        checks['product_mismatches'] += 1
                        reasons.append(f'product: stored {product}, expected {expected_product}')
                else:
                    checks['product_uncheckable'] += 1
                for leg in ('a','b'):
                    group = (category, leg)
                    status = mass_status.setdefault(group, Counter())
                    keys = [f'truth_{leg}_visible_{v}' for v in ('E','px','py','pz')]
                    if not all(key in data for key in keys):
                        status['missing_p4_rows'] += 1
                        continue
                    p = [data[key][i] for key in keys]
                    if not all(finite_number(v) for v in p):
                        status['nonfinite_or_null_rows'] += 1
                        continue
                    # Double arithmetic cannot recover precision lost upstream.
                    m2 = float(p[0])**2 - sum(float(v)**2 for v in p[1:])
                    if not math.isfinite(m2):
                        status['nonfinite_mass2_rows'] += 1
                        continue
                    status['nonpositive_energy_rows'] += p[0] <= 0
                    status['negative_mass2_rows'] += m2 < 0
                    masses.setdefault(group, []).append(m2)
                for key, value in raw.items():
                    if value is None:
                        null_counts[key] += 1
                for leg in ('a', 'b'):
                    key = f'analyzing_power_{leg}'
                    value = raw[key]
                    kappas[leg][values[key]] += 1
                    if not valid_power(value):
                        invalid_counts[key] += 1
                        reasons.append(key + ': missing/null/nonfinite/zero/out-of-range/non-numeric')
                    group = (leg, values['event_category'], values[f'lead_{leg}_pdgId'])
                    assignments.setdefault(group, Counter())[values[key]] += 1
                if all(valid_power(raw[f'analyzing_power_{leg}']) for leg in ('a', 'b')):
                    negative_products += raw['analyzing_power_a'] * raw['analyzing_power_b'] < 0
                if reasons and len(examples) < 20:
                    examples.append(dict(file=str(file), file_row=offset+i, values=values,
                                         source_ids={key: scalar(data[key][i]) for key in IDS if key in present},
                                         reasons=reasons))
            offset += batch.num_rows
        rows += offset
    def counts(counter):
        return [dict(value=value, count=n) for value, n in counter.most_common()]
    mass_reports = []
    for (category, leg), status in mass_status.items():
        m2 = np.asarray(masses.get((category, leg), []), dtype=float)
        positive = m2[m2 >= 0]
        def quantiles(array):
            return {str(q): float(np.quantile(array, q)) for q in (0,.01,.1,.5,.9,.99,1)} if len(array) else {}
        channel = CHANNELS.get(category)
        mass_reports.append(dict(event_category=category, leg=leg,
            expected_analyzer=channel[0 if leg == 'a' else 1] if channel else 'unknown',
            status=dict(status), finite_p4_mass2_rows=len(m2),
            mass2_GeV2_quantiles=quantiles(m2),
            nonnegative_mass_GeV_quantiles=quantiles(np.sqrt(positive))))
    return dict(
        source=str(path), rows=rows, files=schemas, complete_scan=True,
        missing_field_rows=dict(missing_counts), null_field_rows=dict(null_counts),
        invalid_power_rows=dict(invalid_counts), invalid_examples=examples,
        powers={leg: counts(counter) for leg, counter in kappas.items()},
        negative_product_events=int(negative_products),
        consistency_checks={key: checks[key] for key in (
            'unknown_category_rows', 'category_leg_checks', 'category_leg_mismatches',
            'category_leg_uncheckable', 'product_checks', 'product_mismatches', 'product_uncheckable')},
        truth_visible_mass=mass_reports,
        comparison_convention=dict(category_source='config/analysis.yaml:Subcategories.Ztautau',
            expected_powers=EXPECTED, absolute_tolerance=1e-6,
            scope='Checks consistency with channel constants; does NOT establish charge signs or truth/reco category provenance.'),
        combinations=[dict(zip(FIELDS, values), count=n) for values, n in combinations.most_common()],
        multiple_power_groups=[dict(leg=g[0], event_category=g[1], pdgId=g[2], powers=counts(c))
                               for g, c in assignments.items() if len(c) > 1],
        interpretation=[
            'Counts use all rows, without cuts, weights, rounding, or changes to stored kappa.',
            'Multiple powers in a category/PDG group are review flags, NOT proven mistakes.',
            'PDG fields may describe a track rather than the composite visible analyzer.',
            'Category labels use the explicit repository map; no charge-sign conversions are applied.',
            'Visible masses are clues, not particle identification. Negative mass-squared values are reported, not clipped.',
            'This inventory does not certify analyzer identity, acceptance, or physical Cij closure.',
        ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parquet', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32768)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.output.suffix.lower() != '.json':
        parser.error('output must be a .json report, not a parquet file')
    report = inspect(args.parquet, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in
                      ('rows', 'complete_scan', 'missing_field_rows', 'null_field_rows',
                       'invalid_power_rows', 'powers', 'negative_product_events')}, indent=2))
    print('\n=== CATEGORY / PRODUCT CHECKS ===')
    print(json.dumps(report['consistency_checks'], indent=2))
    print('\n=== TRUTH VISIBLE MASS BY CATEGORY AND LEG ===')
    print(json.dumps(report['truth_visible_mass'], indent=2))
    print('\nMost frequent category / PDG A / PDG B / kappa A / kappa B combinations:')
    for row in report['combinations'][:40]:
        print(row)
    print('Groups with multiple powers:', len(report['multiple_power_groups']))
    print('FULL REPORT:', args.output)
    print('Inventory only; physics convention not yet certified.')


if __name__ == '__main__':
    main()
