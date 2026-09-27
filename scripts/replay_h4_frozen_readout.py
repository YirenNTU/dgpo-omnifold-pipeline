#!/usr/bin/env python3
"""Read-only replay of detached H4 panels: balanced BCE/AdamW versus nested ridge."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evenet_dgpo'))
import torch
import torch.nn.functional as F
from RL.DGPO_neutrino.omnifold_ztautau.representation_diagnostic import binary_auc, nested_ridge_readout

DEFAULT = Path('/pscratch/sd/y/yiren/Ztautau/h4_frozen_readout_capture/panels')
NAME = 'Can a linear head read early features? | frozen H4 | AdamW versus ridge'
HISTORY = {
    'h4clf02': {'kind': 'raw Fourier linear shortcut plus existing fusion',
                 'first_val_auc_085_update': 1280, 'updates': 3000,
                 'final_test_auc': .9075213401269238,
                 'conclusion': 'Failed historical acceleration screen; not a matched control.'},
    'h4clflb1': {'kind': 'last-block nonlinear Fourier late fusion',
                  'updates': 3000, 'final_test_auc': .9069309568522148,
                  'conclusion': 'Different trainable modules/regularization; not an architecture-only control.'},
    'h4clfpath1': {'kind': 'nested ridge diagnostic, unchanged H4 training',
                    'step50_encoder_probe_auc': .80906,
                    'step50_fusion_probe_auc': .49527,
                    'conclusion': 'Small diagnostic panel, not whole-audit AUC; motivates frozen readout test.'},
    'oldr2h201': {'kind': 'legacy old-base only', 'updates': 1700,
                   'train_ba': .76645, 'val_ba': .50935,
                   'conclusion': 'Longer legacy fitting overfits; residual stage never executed.'},
}
SETTINGS = dict(steps=3000, lr=2e-4, weight_decay=1e-3, batch_size=1024,
                seeds=[17, 29, 43], captures=[0, 50, 100],
                branches=['raw_fourier', 'fourier'], eval_every=25,
                primary_capture=50, primary_update=200, ridge_min_auc=.65,
                ridge_tolerance=.02, head_gain_min=.03)


def validate_panel(panel):
    if panel.get('schema') != 1 or panel.get('capture_step') not in SETTINGS['captures']:
        raise ValueError('Unsupported panel schema/capture')
    y, fit, inner = (panel[k] for k in ('target', 'fit_mask', 'inner_mask'))
    if any(v.dtype != torch.bool or v.ndim != 1 or len(v) != len(y) for v in (y, fit, inner)):
        raise ValueError('Invalid labels/context partitions')
    for mask in (fit, ~fit, fit & inner, fit & ~inner):
        if min(int((mask & y).sum()), int((mask & ~y).sum())) < 8:
            raise ValueError('Insufficient independent split/class rows')
    for branch in SETTINGS['branches']:
        x = panel['features'][branch]
        if x.ndim != 2 or len(x) != len(y) or not torch.isfinite(x).all():
            raise ValueError('Invalid frozen features')
    logits = panel['current_logits']
    if logits.shape != y.shape or not torch.isfinite(logits).all():
        raise ValueError('Invalid original-head logits')


def prepare_features(features, fit, standardized):
    x = features.detach().cpu().float().clone()
    mean = x[fit].mean(0) if standardized else torch.zeros(x.shape[1])
    scale = x[fit].std(0, unbiased=False).clamp_min(1e-6) if standardized else torch.ones(x.shape[1])
    return (x - mean) / scale, mean, scale


def measure(logits, y, fit):
    result = {}
    for name, mask in [('fit', fit), ('holdout', ~fit)]:
        s, labels = logits[mask], y[mask]
        losses = F.binary_cross_entropy_with_logits(s, labels.float(), reduction='none')
        result[name + '_bce'] = float((losses[labels].mean() + losses[~labels].mean()) / 2)
        result[name + '_auc'] = binary_auc(s, labels)
        result[name + '_logit_rms'] = float(s.square().mean().sqrt())
    result['bce_generalization_gap'] = result['holdout_bce'] - result['fit_bce']
    return result


def fit_head(features, y, fit, *, standardized, seed, steps=3000,
             lr=2e-4, weight_decay=1e-3, batch_size=1024, eval_every=25, callback=None,
             full_batch=False, snapshot_callback=None):
    """Only two new leaf tensors train. No feature gradients or outer-holdout selection."""
    x, mean, scale = prepare_features(features, fit, standardized)
    y, fit = y.cpu(), fit.cpu()
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.AdamW([w, b], lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    pos, neg = (torch.where(fit & label)[0] for label in (y, ~y))
    records = []
    grad_norm = update_norm = 0.
    for step in range(steps + 1):
        if step:
            # Identical balanced minibatch indices across all arms for a given seed.
            optimizer.zero_grad(set_to_none=True)
            if full_batch:
                # Exact expectation of balanced minibatch BCE, even if class counts differ.
                positive = F.softplus(-(x[pos] @ w + b)).mean()
                negative = F.softplus(x[neg] @ w + b).mean()
                loss = (positive + negative) / 2
            else:
                indices = torch.cat([idx[torch.randint(len(idx), (batch_size // 2,), generator=generator)]
                                     for idx in (pos, neg)])
                loss = F.binary_cross_entropy_with_logits(x[indices] @ w + b, y[indices].float())
            loss.backward()
            grad_norm = float((w.grad.square().sum() + b.grad.square()).sqrt())
            old_w, old_b = w.detach().clone(), b.detach().clone()
            optimizer.step()
            update_norm = float(((w.detach() - old_w).square().sum() + (b.detach() - old_b).square()).sqrt())
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                row = dict(step=step, **measure(x @ w + b, y, fit),
                           gradient_norm=grad_norm, update_norm=update_norm,
                           weight_norm=float(w.norm()), lr=lr,
                           rows_processed=step * (int(fit.sum()) if full_batch else batch_size))
            if not all(torch.isfinite(torch.tensor(v)) for v in row.values()):
                raise ValueError('Nonfinite readout result')
            records.append(row)
            if callback:
                callback(row)
            if snapshot_callback:
                with torch.no_grad():
                    snapshot_callback(step, (x @ w + b).detach().clone())
    return dict(curve=records, mean=mean.tolist(), scale=scale.tolist(),
                weights=w.detach().tolist(), bias=float(b.detach()))


def decide(arms, ridge_auc, current_auc):
    """Predeclared descriptive decision; seeds vary only minibatches, not populations."""
    if ridge_auc < SETTINGS['ridge_min_auc']:
        return 'inconclusive: early frozen ridge signal did not replicate'
    passed = {}
    for scaling in ('raw', 'standardized'):
        endpoints = [next(row for row in a['curve'] if row['step'] == SETTINGS['primary_update'])
                     for key, a in arms.items() if key.startswith(scaling + '/')]
        passed[scaling] = len(endpoints) == 3 and all(
            r['holdout_auc'] >= ridge_auc - SETTINGS['ridge_tolerance']
            and r['holdout_auc'] >= current_auc + SETTINGS['head_gain_min']
            and r['holdout_bce'] < .69314718056 for r in endpoints)
    if passed['raw']:
        return 'supports accessible encoder signal; motivates matched end-to-end readout-path test'
    if passed['standardized']:
        return 'supports scaling/conditioning sensitivity; shortcut alone is not yet supported'
    return 'no rapid BCE readout recovery; inspect objective/optimizer/conditioning before architecture changes'


def batch_decision(arms, ridge_auc, current_auc):
    """Fixed step-200 endpoint plus prespecified 200..300 stability window."""
    threshold = max(ridge_auc - .02, current_auc + .03)
    results = {}
    for name, arm in arms.items():
        window = [r for r in arm['curve'] if 200 <= r['step'] <= 300]
        endpoint = next(r for r in window if r['step'] == 200)
        aucs = [r['holdout_auc'] for r in window]
        results[name] = dict(auc200=endpoint['holdout_auc'], min_auc=min(aucs),
            auc_range=max(aucs) - min(aucs),
            passed=all(r['holdout_auc'] >= threshold and r['holdout_bce'] < .69314718056 for r in window)
                   and max(aucs) - min(aucs) <= .05)
    full = results['full_batch']['passed']
    minibatch = all(v['passed'] for k, v in results.items() if k.startswith('minibatch/'))
    decision = ('inconclusive: ridge signal did not replicate' if ridge_auc < .65 else
                'supports minibatch variability as a contributor on this frozen panel' if full and not minibatch else
                'both pass; prior instability not reproduced' if full else
                'full-batch does not suffice; investigate conditioning/effective step size/objective')
    return dict(decision=decision, threshold=threshold, arms=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panels', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=DEFAULT.parent / 'readout_replay')
    parser.add_argument('--offline', action='store_true', help='Write local report without W&B')
    parser.add_argument('--quick', action='store_true', help='300 readout updates, separate W&B/output; reuse existing captures')
    parser.add_argument('--batch-ablation', action='store_true', help='Step-50 standardized encoder: matched minibatch versus exact full-batch BCE')
    args = parser.parse_args()
    settings = dict(SETTINGS)
    if args.batch_ablation:
        settings.update(steps=300, captures=[50], branches=['fourier'],
                        intervention='balanced minibatch versus exact full-batch BCE')
        if args.output == DEFAULT.parent / 'readout_replay':
            args.output = args.panels.parent / 'readout_batch300'
    if args.quick:
        settings['steps'] = 300
        if args.output == DEFAULT.parent / 'readout_replay':
            args.output = args.panels.parent / 'readout_replay_quick300'
    torch.set_num_threads(1)
    panels = []
    for step in settings['captures']:
        panel = torch.load(args.panels / f'panel_step{step:04d}.pt', map_location='cpu', weights_only=True)
        validate_panel(panel)
        if panel['capture_step'] != step or panel['world_size'] != 16:
            raise ValueError('Production comparison requires correct capture steps and 16-GPU panel')
        panels.append(panel)
    # Verify the fixed panel partition/labels were preserved across captures.
    for panel in panels[1:]:
        for key in ('target', 'fit_mask', 'inner_mask'):
            if not torch.equal(panel[key], panels[0][key]):
                raise ValueError('Capture panels changed row alignment')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(settings=settings, historical_context=HISTORY, panel_source=str(args.panels),
                  scope='diagnostic validation subpanel, not an independent final audit',
                  captured_fit_config=panels[0]['fit_config'],
                  panel_rows=len(panels[0]['target']), captures={})
    if args.batch_ablation:
        report['matched_previous_result'] = dict(run='h4readq1', capture=50, update=200,
            standardized_auc_by_seed={'17': .829964, '29': .882074, '43': .693570},
            standardized_auc300_by_seed={'17': .876570, '29': .869995, '43': .574730},
            ridge_auc=.8423476502082095,
            note='Historical context only; rerun identical minibatch arms as the matched control.')
    run = None
    try:
        if not args.offline:
            import wandb
            run = wandb.init(entity='ytchou97-university-of-washington', project='nu2flow-RL',
                id='h4readb1' if args.batch_ablation else ('h4readq1' if args.quick else 'h4readout1'), resume='never',
                name='Does minibatch noise cause instability? | frozen H4 | full-batch BCE' if args.batch_ablation else
                     ('Can a linear head escape the plateau? | frozen H4 | 300-update screen' if args.quick else NAME),
                group='H4 frozen readout diagnosis',
                tags=['H4', 'FrozenReadout', 'NoPolicyUpdate', 'CPUReplay'],
                config=dict(**report, source_policy_step=1110))
        for panel in panels:
            step = panel['capture_step']
            y, fit, inner = (panel[k] for k in ('target', 'fit_mask', 'inner_mask'))
            current = measure(panel['current_logits'], y, fit)
            capture = dict(current_head=current, branches={})
            if run:
                run.log({f'frozen/encoder_step{step}/original_head/{k}': v for k, v in current.items()})
            for branch in settings['branches']:
                x = panel['features'][branch]
                ridge = nested_ridge_readout(x, y, fit, inner)
                if not ridge['valid']:
                    raise ValueError('Invalid nested ridge control')
                arms = {}
                root = f'frozen/encoder_step{step}/{branch}'
                if run:
                    run.log({root + '/ridge/' + k: v for k, v in ridge.items()})
                for standardized in ((True,) if args.batch_ablation else (False, True)):
                    for seed in SETTINGS['seeds']:
                        label = ('minibatch' if args.batch_ablation else ('standardized' if standardized else 'raw')) + f'/seed{seed}'
                        prefix = root + '/' + label
                        if run:
                            run.define_metric(prefix + '/step')
                            run.define_metric(prefix + '/*', step_metric=prefix + '/step')
                        def log(row, prefix=prefix, ridge_auc=ridge['holdout_auc'], current_auc=current['holdout_auc']):
                            if run:
                                metrics = {prefix + '/' + k: v for k, v in row.items()}
                                metrics[prefix + '/auc_minus_ridge'] = row['holdout_auc'] - ridge_auc
                                metrics[prefix + '/auc_minus_original_head'] = row['holdout_auc'] - current_auc
                                run.log(metrics)
                        arms[label] = fit_head(x, y, fit, standardized=standardized, seed=seed,
                            **{k: settings[k] for k in ('steps', 'lr', 'weight_decay', 'batch_size', 'eval_every')}, callback=log)
                        print(f'capture={step} branch={branch} arm={label} final={arms[label]["curve"][-1]}', flush=True)
                if args.batch_ablation:
                    prefix = root + '/full_batch'
                    if run:
                        run.define_metric(prefix + '/step')
                        run.define_metric(prefix + '/*', step_metric=prefix + '/step')
                    def log_full(row):
                        if run:
                            run.log({prefix + '/' + k: v for k, v in row.items()})
                    # No RNG-dependent operation: one deterministic run, not three fake replicates.
                    arms['full_batch'] = fit_head(x, y, fit, standardized=True, seed=17,
                        full_batch=True, **{k: settings[k] for k in ('steps', 'lr', 'weight_decay', 'batch_size', 'eval_every')},
                        callback=log_full)
                capture['branches'][branch] = dict(ridge=ridge, arms=arms)
            report['captures'][str(step)] = capture
            with (args.output / f'capture_{step}.json').open('x') as stream:
                json.dump(capture, stream, indent=2, allow_nan=False)
        primary = report['captures']['50']
        result = primary['branches']['fourier']
        if args.batch_ablation:
            report['batch_comparison'] = batch_decision(result['arms'], result['ridge']['holdout_auc'], primary['current_head']['holdout_auc'])
            report['decision'] = report['batch_comparison']['decision']
        else:
            report['decision'] = decide(result['arms'], result['ridge']['holdout_auc'], primary['current_head']['holdout_auc'])
        with (args.output / 'report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        if run:
            run.summary['decision'] = report['decision']
            run.summary['completed'] = True
            run.summary['policy_updates'] = 0
            run.save(str(args.output / '*.json'), base_path=str(args.output))
        print(report['decision'], flush=True)
    except Exception:
        if run:
            run.finish(exit_code=1)
        raise
    else:
        if run:
            run.finish()


if __name__ == '__main__':
    main()
