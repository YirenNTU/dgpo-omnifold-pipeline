"""Inspect saved conditional tau ratios and paired tail-removal sensitivity on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np


def normalized_weights(base, logits):
    base, logits = np.asarray(base, float), np.asarray(logits, float)
    if base.shape != logits.shape or not np.isfinite(base).all() or not np.isfinite(logits).all():
        raise ValueError('Invalid aligned weights/logits')
    if (base < 0).any() or not (base > 0).any():
        raise ValueError('Require nonnegative base weights with positive total')
    lw = np.full(base.shape, -np.inf)
    positive = base > 0
    lw[positive] = np.log(base[positive]) + logits[positive]
    shifted = np.exp(lw - lw.max())
    total = shifted.sum()
    return shifted / total, float(lw.max() + np.log(total) - np.log(base.sum()))


def health(base, logits):
    w, log_mean = normalized_weights(base, logits)
    return dict(events=len(w), ess=float(1 / np.square(w).sum()),
        ess_fraction=float(1 / np.square(w).sum() / len(w)),
        max_mass=float(w.max()), top1pct_mass=float(np.sort(w)[-max(1, int(np.ceil(len(w)*.01))):].sum()),
        log_mean_ratio=log_mean,
        logit_quantiles={str(q):float(np.quantile(logits, q)) for q in (0, .5, .95, .99, .999, 1)})


def moments(truth, generated, base, logits, keep):
    w, _ = normalized_weights(base[keep], logits[keep])
    target = np.average(truth[keep], weights=base[keep], axis=0)
    full_target = np.average(truth, weights=base, axis=0)
    raw = np.average(generated[keep], weights=base[keep], axis=0)
    fit = np.sum(generated[keep] * w[:, None], axis=0)
    e0, e1 = float(np.linalg.norm(raw-target)), float(np.linalg.norm(fit-target))
    return dict(target=target.tolist(), unweighted=raw.tolist(), reweighted=fit.tolist(),
        error_unweighted=e0, error_reweighted=e1, error_change=e1-e0,
        error_unweighted_to_original_target=float(np.linalg.norm(raw-full_target)),
        error_reweighted_to_original_target=float(np.linalg.norm(fit-full_target)),
        target_shift=float(np.linalg.norm(target-full_target)))


def analyze(arrays, scores):
    test = arrays['split'] == 2
    fit = arrays['split'] == 0
    ids = arrays['source_ids'][test]
    if not np.array_equal(ids, scores['source_ids']):
        raise ValueError('test_scores and prepared arrays have different source IDs/order')
    if not np.array_equal(scores['generated_logits'], scores['log_ratio']):
        raise ValueError('Raw ratio must be the generated-event logit')
    n = len(ids)
    if n < 20 or not fit.any():
        raise ValueError('Need fit pool and at least 20 held-out events')
    base = arrays['event_weight'][test]
    logits = np.asarray(scores['log_ratio'], float)
    w, _ = normalized_weights(base, logits)
    order = np.argsort(-w, kind='stable')
    powers = arrays['kappas'][test]
    if not np.isfinite(powers).all() or (powers == 0).any():
        raise ValueError('Invalid analyzing powers')
    angular_truth = np.einsum('ni,nj->nij', arrays['truth_a'][test], arrays['truth_b'][test]).reshape(n, 9)
    angular_gen = np.einsum('ni,nj->nij', arrays['sample_a'][test], arrays['sample_b'][test]).reshape(n, 9)
    families = dict(tau=(arrays['tau_truth'][test], arrays['tau_generated'][test]),
        angular=(angular_truth, angular_gen),
        cij=(9*angular_truth/np.prod(powers, axis=1)[:, None],
             9*angular_gen/np.prod(powers, axis=1)[:, None]))
    for truth, gen in families.values():
        if not np.isfinite(truth).all() or not np.isfinite(gen).all():
            raise ValueError('Nonfinite moment features')
    arms = []
    for count in sorted(set((0, 1, 10, int(np.ceil(n*.001)), int(np.ceil(n*.01))))):
        keep = np.ones(n, dtype=bool)
        keep[order[:count]] = False
        arms.append(dict(removed=count, removed_original_mass=float(w[~keep].sum()),
            health=health(base[keep], logits[keep]),
            moments={name:moments(t, g, base, logits, keep) for name,(t,g) in families.items()}))
    ranges = {}
    for name in ('condition', 'candidate_generated'):
        values = arrays[name][fit]
        ranges[name] = (values.min(axis=0), values.max(axis=0))
    top = []
    for i in order[:20]:
        record = dict(source_id=str(ids[i]), test_index=int(i), mass=float(w[i]),
            category=int(arrays['category'][test][i]), base_weight=float(base[i]),
            truth_logit=float(scores['truth_logits'][i]), generated_logit=float(logits[i]),
            saved_h4_logit=float(scores['saved_h4_log_ratio'][i]),
            tau_truth=arrays['tau_truth'][test][i].tolist(),
            tau_generated=arrays['tau_generated'][test][i].tolist())
        for name, (lo, hi) in ranges.items():
            value = arrays[name][test][i]
            outside = np.flatnonzero((value < lo) | (value > hi))
            record[name] = dict(max_abs=float(np.abs(value).max()),
                outside_fit_range=[dict(index=int(j), value=float(value[j]),
                    fit_min=float(lo[j]), fit_max=float(hi[j])) for j in outside],
                values=value.tolist())
        top.append(record)
    return dict(events=n, top_events=top, removal_arms=arms,
        original_health=health(base, logits),
        scope='Fixed classifier, fixed samples, raw logits; no retraining or generator updates',
        convention='Cij uses the existing matched-target 9*mean(a_i*b_j/(kappa_a*kappa_b)) estimator',
        limitations=['Tail removal is a diagnostic, sample-dependent selection, not an unbiased correction.',
            'Both paired truth and generated events are removed; original-target errors are also reported.',
            'Condition values are saved standardized/clipped model inputs, not original raw parquet values.',
            'Coordinate-wise fit-range checks cannot certify joint coverage or logit calibration.',
            'Point estimates only; no inference from bootstrap with near-one ESS.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/conditional_tau_ratio_10pct.yaml')
    parser.add_argument('--directory', type=Path, help='Override saved classifier output directory')
    parser.add_argument('--source-run', default='qelc17rx')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    import yaml
    directory = args.directory or Path(yaml.safe_load(args.config.read_text())['experiment']['output'])
    with np.load(directory/'prepared.npz', allow_pickle=False) as f:
        arrays = {k:f[k] for k in f.files}
    with np.load(directory/'test_scores.npz', allow_pickle=False) as f:
        scores = {k:f[k] for k in f.files}
    report = analyze(arrays, scores)
    report.update(source_run=args.source_run, source_directory=str(directory.resolve()))
    output = directory / ('tail-diagnostic-' + uuid.uuid4().hex[:10])
    output.mkdir()
    path = output/'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    rows = []
    for arm in report['removal_arms']:
        row = [arm['removed'], arm['health']['ess'], arm['health']['max_mass']]
        row += [arm['moments'][k]['error_change'] for k in ('tau','angular','cij')]
        rows.append(row)
        print(dict(zip(('removed','ess','max_mass','tau_error_change','angular_error_change','cij_error_change'), row)), flush=True)
    if not args.no_wandb:
        import wandb
        with wandb.init(entity='ytchou97-university-of-washington', project='nu2flow-RL',
                name='Does one event dominate the ratio? | conditional tau MLP | tail sensitivity',
                group='Conditional ratio closure', config={'source_run':args.source_run,
                    'source_directory':str(directory.resolve()), 'analysis_only':True}, dir=str(output)) as run:
            run.log({'tail/removal':wandb.Table(columns=['removed','ess','max_mass',
                'tau_error_change','angular_error_change','cij_error_change'], data=rows)})
            for arm in report['removal_arms']:
                prefix = f'drop_{arm["removed"]}'
                for key in ('ess','ess_fraction','max_mass','log_mean_ratio'):
                    run.summary[f'{prefix}/{key}'] = arm['health'][key]
                for key in ('tau','angular','cij'):
                    run.summary[f'{prefix}/{key}_error_change'] = arm['moments'][key]['error_change']
            run.save(str(path), base_path=str(output), policy='now')
            print('WANDB:',run.url, flush=True)
    print('REPORT:',path, flush=True)


if __name__ == '__main__':
    main()
