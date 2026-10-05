"""Read-only paired reward accounting, shared by saved toy and native panels.

No policy updates or reward transformations. Candidate splits protect group
selection from reusing the same Monte Carlo noise for outcome evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def validate_rewards(before, early, late):
    values = [np.asarray(x, dtype=np.float64) for x in (before, early, late)]
    if any(x.shape != values[0].shape for x in values):
        raise ValueError('Paired endpoints must have identical shapes')
    if values[0].ndim != 2 or min(values[0].shape) < 2:
        raise ValueError('Require at least two conditions and two candidate draws')
    if values[0].shape[1] % 2:
        raise ValueError('Candidate count must be even for equal split cross-fitting')
    if not all(np.isfinite(x).all() for x in values):
        raise ValueError('Rewards must be finite')
    return values


def intervals(columns, *, repeats=2000, seed=290929):
    columns = np.asarray(columns, dtype=np.float64)
    n = len(columns)
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, columns.shape[1]))
    for start in range(0, repeats, 64):
        count = min(64, repeats-start)
        ids = rng.integers(n, size=(count, n))
        draws[start:start+count] = columns[ids].mean(1)
    return [{'mean': float(m), 'lo95': float(lo), 'hi95': float(hi)}
            for m, lo, hi in zip(columns.mean(0), *np.quantile(draws, [.025, .975], axis=0))]


def analyze(before, early, late, *, repeats=2000):
    if repeats < 100:
        raise ValueError('At least 100 bootstrap replicates')
    before, early, late = validate_rewards(before, early, late)
    g = (early-before).mean(1)
    h = (late-early).mean(1)
    metrics = dict(zip(('early_gain', 'later_change', 'net_gain'),
                       intervals(np.stack((g, h, g+h), 1), repeats=repeats)))
    loss = np.maximum(-h, 0)
    erased = np.minimum(np.maximum(g, 0), loss)
    additional = loss-erased
    gained = np.maximum(h, 0)
    accounting = {'new_gains': float(gained.mean()), 'erased_early_gains': float(erased.mean()),
                  'additional_damage': float(additional.mean()),
                  'reconstruction_error': float(abs((gained-erased-additional).mean()-h.mean())),
                  'scope': 'Exact observed mean accounting; nonlinear clipping is not unbiased causal attribution'}
    crossfit = {}
    for beneficiary in (True, False):
        population = np.zeros(len(g))
        contributions = np.zeros((len(g), 3))
        for assignment, evaluation in ((slice(0, None, 2), slice(1, None, 2)),
                                       (slice(1, None, 2), slice(0, None, 2))):
            chosen = ((early[:, assignment]-before[:, assignment]).mean(1) > 0) == beneficiary
            ge = (early[:, evaluation]-before[:, evaluation]).mean(1)
            he = (late[:, evaluation]-early[:, evaluation]).mean(1)
            population += .5*chosen
            contributions += .5*chosen[:, None]*np.stack((ge, he, ge+he), 1)
        frac = float(population.mean())
        crossfit['early_beneficiaries' if beneficiary else 'other_conditions'] = {
            'fraction': frac,
            'population_contributions': dict(zip(('early_gain', 'later_change', 'net_gain'),
                intervals(contributions, repeats=repeats))),
            'conditional_means': dict(zip(('early_gain', 'later_change', 'net_gain'),
                (contributions.mean(0)/frac).tolist())) if frac else None}
    ratio = float((g+h).mean()/g.mean()) if g.mean() > 0 else None
    return {'conditions': len(g), 'candidates': before.shape[1], 'metrics': metrics,
            'late_retention_fraction': ratio, 'accounting': accounting,
            'crossfit_groups': crossfit,
            'fractions': {'early_positive': float((g>0).mean()),
                          'later_negative': float((h<0).mean()),
                          'net_negative': float((g+h<0).mean())},
            'strong_aggregate_regression': metrics['early_gain']['lo95'] > .01
                and metrics['later_change']['hi95'] < -.01 and ratio is not None and ratio < .5,
            'uncertainty': 'Pointwise condition-bootstrap, fixed models/noise; deterministic toy grid intervals descriptive only'}


def native_panel(directory, steps=(0, 5, 20), panel='heldout', expected_ranks=16):
    """Align IDs even if native shard/event order changed. Require complete ranks."""
    all_values, anchor = [], None
    rank_set = None
    for step in steps:
        files = sorted(Path(directory).glob(f'{panel}_step{step:02d}_rank*.npz'))
        ranks = {p.stem.rsplit('rank', 1)[1] for p in files}
        if ranks != {f'{i:02d}' for i in range(expected_ranks)}:
            raise ValueError('Missing native ranks; do not analyze a partial distributed panel')
        if not files or (rank_set is not None and ranks != rank_set):
            raise ValueError('Missing or mismatched endpoint ranks')
        rank_set = ranks
        ids, rewards = [], []
        for file in files:
            with np.load(file, allow_pickle=False) as data:
                ids.extend(data['event_ids'].tolist())
                rewards.append(data['rewards'].copy())
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate event identities')
        values = np.concatenate(rewards)
        if len(ids) != len(values):
            raise ValueError('Mismatched rewards and event IDs')
        if anchor is None:
            anchor = ids
        if set(ids) != set(anchor):
            raise ValueError('Endpoint event identities differ')
        position = {key: i for i, key in enumerate(ids)}
        all_values.append(values[[position[key] for key in anchor]])
    return all_values


def toy_campaign(root, *, repeats=2000):
    import torch
    rows = {}
    for name in ('round04', 'round07', 'round08', 'round09', 'round10', 'round12'):
        directory = Path(root)/name
        plan = json.loads((directory/'plan.json').read_text())
        base = torch.load(directory/'baseline_endpoint.pt', weights_only=True, map_location='cpu')
        for arm in plan['conditioning']:
            points = [torch.load(directory/f'{arm}_endpoint{s}.pt', weights_only=True, map_location='cpu')
                      for s in (25, 100, 300, 1000)]
            if any(not torch.equal(p['contexts'], base['contexts']) for p in points):
                raise ValueError('Toy condition identities differ')
            r0 = base['rewards']['full'].numpy()
            r = [p['rewards']['full'].numpy() for p in points]
            result = analyze(r0, r[0], r[-1], repeats=repeats)
            result['fixed_horizon_means'] = {str(s): float(x.mean()) for s, x in zip((0,25,100,300,1000), [r0,*r])}
            result['source'] = str(directory)
            rows[f'{name}/{arm}'] = result
    return {'state': 'completed', 'scope': 'Post-hoc reanalysis; no fitting/updates; same fixed full-truth critic',
            'arms': rows, 'any_aggregate_regression': any(x['strong_aggregate_regression'] for x in rows.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--native', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    from .closed_loop_lab import toy_path
    from .truth_pretrain import atomic_json
    output = toy_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = (analyze(*native_panel(args.directory), repeats=args.replicates) if args.native
              else toy_campaign(args.directory, repeats=args.replicates))
    if args.native:
        result.update(source=str(args.directory.resolve()), expected_ranks=16,
                      scope='Native production rewards; event IDs aligned; no toy simulation or updates')
    atomic_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
