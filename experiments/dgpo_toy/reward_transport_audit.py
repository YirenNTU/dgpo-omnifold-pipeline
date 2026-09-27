"""Cold, paired full-truth audits for the reward-transport diagnostic.

These classifiers never supply a policy reward. Validation alone selects weights.
"""
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import conditional as native
from .nonperiodic_cube import Critic, classification, panel
from .nonperiodic_cube_audit import scores
from .truth_pretrain import atomic_checkpoint, atomic_json


def audit_contrasts(predictions, repeats=300):
    """Paired context bootstrap; uncertainty conditional on the fitted classifiers."""
    arrays = {k: {s: v.detach().cpu().numpy() for s, v in d.items()}
              for k, d in predictions.items()}
    n = len(arrays['baseline']['positive'])
    labels = np.r_[np.ones(n), np.zeros(n)]

    def metrics(p, ids):
        positive, negative = p['positive'][ids], p['negative'][ids]
        auc = roc_auc_score(labels, np.r_[positive, negative])
        bce = .5 * (np.logaddexp(0., -positive) + np.logaddexp(0., negative)).mean()
        return np.array([auc, abs(auc - .5), bce])

    names = [k for k in arrays if k != 'baseline']
    contrasts = [(k, 'baseline') for k in names]
    contrasts += [(k, 'A') for k in names if k != 'A' and 'A' in arrays]
    if 'B' in arrays and 'C' in arrays:
        contrasts.append(('C', 'B'))
    full = np.arange(n)
    point = {k: metrics(p, full) for k, p in arrays.items()}
    draws = {f'{a}_minus_{b}': [] for a, b in contrasts}
    rng = np.random.default_rng(391017)
    for _ in range(repeats):
        ids = rng.integers(0, n, n)
        sampled = {k: metrics(p, ids) for k, p in arrays.items()}
        for a, b in contrasts:
            draws[f'{a}_minus_{b}'].append(sampled[a] - sampled[b])
    result = {}
    for a, b in contrasts:
        key = f'{a}_minus_{b}'
        quantiles = np.quantile(draws[key], [.025, .975], axis=0)
        result[key] = {name: {'delta': float(point[a][i] - point[b][i]),
                             'lo95': float(quantiles[0, i]), 'hi95': float(quantiles[1, i])}
                       for i, name in enumerate(('auc', 'auc_gap', 'bce'))}
    return result


def fit_audits(policies, data, output, emit, max_steps=16000):
    """All arms share cold initialization, minibatch indices, conditions and truth."""
    if not policies:
        raise ValueError('At least one policy is required for a cold audit')
    pairing_anchor = next(iter(policies))
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'generating_panels', 'cold_start': True, 'seed': 41,
              'max_steps': max_steps, 'minimum_steps': 2000, 'patience_checks': 20,
              'check_every': 100, 'min_delta': 1e-4, 'history': [],
              'scope': 'Full original truth, not shape-matched training target; single audit seed'}
    atomic_json(output / 'report.json', report)
    try:
        panels = {}
        for name, policy in policies.items():
            panels[name] = {split: panel(data, n, seed, policy) for split, n, seed in
                           [('train', 32768, 351041), ('validation', 8192, 361041),
                            ('test', 16384, 371041)]}
            if name != pairing_anchor:
                for split, p in panels[name].items():
                    for key in ('c', 'positive'):
                        if not torch.equal(p[key], panels[pairing_anchor][split][key]):
                            raise ValueError('Audit panels lost condition/truth pairing')
            atomic_checkpoint(output / f'{name}_panels.pt', panels[name])
            emit({'phase': 'audit_panel', 'arm': name, 'step': 0})
        with torch.random.fork_rng():
            torch.manual_seed(41)
            initial = Critic(True)
        models = {k: copy.deepcopy(initial) for k in policies}
        opts = {k: torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.001)
                for k, m in models.items()}
        best = {k: math.inf for k in models}
        anchor, stale, selected, weights = best.copy(), dict.fromkeys(models, 0), {}, {}
        rng = native.generator(381041)
        report['state'] = 'training_cold_audits'
        for step in range(1, max_steps + 1):
            ids = torch.randint(32768, (256,), generator=rng)
            losses = {}
            for name, model in models.items():
                p = panels[name]['train']
                logits = model(torch.cat([p['positive'][ids], p['negative'][ids]]),
                               torch.cat([p['c'][ids]] * 2))
                loss = F.binary_cross_entropy_with_logits(
                    logits, torch.cat([torch.ones(256), torch.zeros(256)]))
                opts[name].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opts[name].step()
                losses[name] = float(loss.detach())
            if step % 100 == 0 or step == max_steps:
                row = {'step': step, 'arms': {}}
                for name, model in models.items():
                    stats, _ = classification(model, panels[name]['validation'])
                    value = stats['bce']
                    if value < best[name]:
                        best[name], selected[name] = value, step
                        weights[name] = copy.deepcopy(model.state_dict())
                    if value < anchor[name] - 1e-4:
                        anchor[name], stale[name] = value, 0
                    else:
                        stale[name] += 1
                    row['arms'][name] = {'validation_bce': value, 'validation_auc': stats['auc'],
                                        'train_minibatch_bce': losses[name],
                                        'selected_step': selected[name], 'stale_checks': stale[name]}
                    emit({'phase': 'audit', 'arm': name, 'step': step, **row['arms'][name]})
                    atomic_checkpoint(output / f'{name}_training_state.pt', {
                        'model': model.state_dict(), 'optimizer': opts[name].state_dict(),
                        'rng': rng.get_state(), 'step': step, 'best_model': weights[name],
                        'selected_step': selected[name], 'stale_checks': stale[name]})
                report['history'].append(row)
                report['step'] = step
                atomic_json(output / 'report.json', report)
                if step >= 2000 and min(stale.values()) >= 20:
                    break
        predictions, report['test'] = {}, {}
        for name, model in models.items():
            model.load_state_dict(weights[name])
            model.eval()
            stats, _ = classification(model, panels[name]['test'])
            report['test'][name] = {**stats, 'auc_gap': abs(stats['auc'] - .5),
                                    'fit_steps': step, 'selected_step': selected[name],
                                    'plateau': step >= 2000 and stale[name] >= 20}
            predictions[name] = scores(model, panels[name]['test'])
            atomic_checkpoint(output / f'{name}_best.pt', {
                'model': model.state_dict(), 'selected_step': selected[name]})
            emit({'phase': 'audit_test', 'arm': name, 'step': step, **report['test'][name]})
        atomic_checkpoint(output / 'test_scores.pt', predictions)
        # A trajectory can audit just one policy per checkpoint.  Its temporal
        # contrasts are computed afterward, across paired checkpoint panels.
        report['contrasts'] = audit_contrasts(predictions) if 'baseline' in predictions else {}
        report['state'] = ('completed' if all(s['plateau'] for s in report['test'].values())
                           else 'budget_exhausted_inconclusive')
        return report
    except BaseException as exc:
        report.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        atomic_json(output / 'report.json', report)
