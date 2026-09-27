#!/usr/bin/env python3
"""300-update frozen-head replay: BCE attribution and density-ratio diagnostics."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from replay_h4_frozen_readout import DEFAULT, fit_head, measure, prepare_features, validate_panel

NAME = 'Where does BCE improvement come from? | frozen H4 ratio | 300-update diagnosis'
STEPS = (0, 50, 100, 200, 300)


def attribution(before, after, labels):
    """Signed balanced BCE gain; concentration of positive gains is separate from losses."""
    y = labels.bool()
    weights = torch.where(y, .5 / y.sum(), .5 / (~y).sum()).double()
    delta = before.double() - after.double()
    gain = weights * delta
    positive, negative = gain.clamp_min(0), (-gain).clamp_min(0)
    k = max(1, math.ceil(len(y) * .01))
    total = float(positive.sum())
    return dict(balanced_bce_gain=float(gain.sum()), gross_gain=total,
                gross_deterioration=float(negative.sum()),
                balanced_fraction_improved=float(weights[delta > 0].sum()),
                top1pct_positive_gain_share=float(positive.topk(k).values.sum()) / total if total else 0.,
                median_sample_gain=float(delta.median()))


def ratio_metrics(logits):
    """On generated rows only. No tempering/clipping. Use log-space for heavy tails."""
    s = logits.double()
    normalized = torch.softmax(s, dim=0)
    return dict(ess_fraction=float(1 / (len(s) * normalized.square().sum())),
                log_mean_ratio=float(torch.logsumexp(s, 0) - math.log(len(s))),
                top1pct_weight_mass=float(normalized.topk(max(1, math.ceil(len(s) * .01))).values.sum()),
                log_ratio_p01=float(torch.quantile(s, .01)),
                log_ratio_p50=float(torch.quantile(s, .5)),
                log_ratio_p99=float(torch.quantile(s, .99)), log_ratio_max=float(s.max()))


def diagnostics(logits, baseline, y, fit, norms, cutoff):
    losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction='none')
    old_losses = F.binary_cross_entropy_with_logits(baseline, y.float(), reduction='none')
    result = measure(logits, y, fit)
    for split, mask in [('fit', fit), ('holdout', ~fit)]:
        for k, v in attribution(old_losses[mask], losses[mask], y[mask]).items():
            result[f'{split}/from_initial/{k}'] = v
        for label, select in [('truth', y), ('generated', ~y)]:
            chosen = mask & select
            result[f'{split}/{label}/bce'] = float(losses[chosen].mean())
            tail = chosen & (norms > cutoff)
            core = chosen & (norms <= cutoff)
            for group, rows in [('high_norm', tail), ('core', core)]:
                result[f'{split}/{label}/{group}/rows'] = int(rows.sum())
                if rows.any():
                    result[f'{split}/{label}/{group}/bce'] = float(losses[rows].mean())
        for k, v in ratio_metrics(logits[mask & ~y]).items():
            result[f'{split}/generated_ratio/{k}'] = v
    return result


def run_panel(panel, output, run=None, steps=300):
    validate_panel(panel)
    if panel['capture_step'] != 50 or panel['world_size'] != 16:
        raise ValueError('Requires step-50 panel from the 16-GPU capture')
    x = panel['features']['fourier']
    y, fit = panel['target'], panel['fit_mask']
    standardized, mean, scale = prepare_features(x, fit, True)
    norms = standardized.norm(dim=1)
    cutoff = float(torch.quantile(norms[fit], .99))
    report = dict(scope='Fixed diagnostic panel, not final audit; no policy updates',
                  primary_endpoint='held-out balanced BCE at update 200',
                  secondary_endpoint='BCE at 50/100/300, loss attribution and ratio tails',
                  standardization_fit_only=True, feature_norm_fit_p99=cutoff, arms={})
    snapshots = dict(target=y, fit_mask=fit, feature_norm=norms, arms={})
    for arm, seed, full in [('minibatch_seed17',17,False), ('minibatch_seed29',29,False),
                            ('minibatch_seed43',43,False), ('full_batch',17,True)]:
        scores, records = {}, []
        if run:
            run.define_metric(arm + '/step')
            run.define_metric(arm + '/*', step_metric=arm + '/step')
        def snapshot(step, logits):
            if step in STEPS:
                scores[step] = logits
            metrics = diagnostics(logits, scores[0], y, fit, norms, cutoff)
            metrics['step'] = step
            if step in STEPS:
                records.append(metrics)
            if run:
                run.log({arm + '/' + k: v for k, v in metrics.items()})
        fitted = fit_head(x, y, fit, standardized=True, seed=seed, full_batch=full,
                         steps=steps, lr=2e-4, weight_decay=1e-3, batch_size=1024,
                         eval_every=25, snapshot_callback=snapshot)
        transitions = {}
        for start, end in [(0,50),(50,100),(50,200),(200,300)]:
            if start not in scores or end not in scores:
                continue
            a = F.binary_cross_entropy_with_logits(scores[start], y.float(), reduction='none')
            b = F.binary_cross_entropy_with_logits(scores[end], y.float(), reduction='none')
            for split, mask in [('fit',fit),('holdout',~fit)]:
                key = f'{start}_to_{end}/{split}'
                transitions[key] = attribution(a[mask], b[mask], y[mask])
                if run:
                    run.summary.update({f'{arm}/transition/{key}/{k}': v for k,v in transitions[key].items()})
        report['arms'][arm] = dict(curve=records, transitions=transitions, optimizer_curve=fitted['curve'])
        snapshots['arms'][arm] = scores
        print(arm, {r['step']: r['holdout_bce'] for r in records}, flush=True)
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (output / 'sample_scores.pt').open('xb') as stream:
        torch.save(snapshots, stream)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    panel = torch.load(args.panels / 'panel_step0050.pt', weights_only=True, map_location='cpu')
    validate_panel(panel)
    output = args.output or args.panels.parent / 'readout_bce300'
    output.mkdir(parents=True, exist_ok=False)
    run = None
    try:
        if not args.offline:
            import wandb
            run = wandb.init(entity='ytchou97-university-of-washington', project='nu2flow-RL',
                id='h4bceq1', resume='never', name=NAME, group='H4 frozen readout diagnosis',
                tags=['H4','BCEPrimary','FrozenReadout','NoPolicyUpdate','CPUReplay'],
                config=dict(steps=300, primary_update=200, source_policy_step=1110,
                    capture_step=50, panel_source=str(args.panels), lr=2e-4,
                    weight_decay=1e-3, batch_size=1024, seeds=[17,29,43],
                    historical_run='h4readb1',
                    historical_full_batch_holdout_bce={'50':.6911433339,'200':.6907104254,'300':.6907894611},
                    interpretation='BCE primary; historical AUC gate is not a density-ratio success criterion'))
        run_panel(panel, output, run)
        if run:
            run.summary.update(completed=True, policy_updates=0)
            run.save(str(output / 'report.json'), base_path=str(output))
            run.save(str(output / 'sample_scores.pt'), base_path=str(output))
    except Exception:
        if run:
            run.finish(exit_code=1)
        raise
    else:
        if run:
            run.finish()


if __name__ == '__main__':
    main()
