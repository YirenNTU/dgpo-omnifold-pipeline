"""Read-only paired-score analysis of the closed-loop rounds 2 and 3.

Only the derived report is written. Never refit, select or change checkpoints.
"""
import argparse
import json
from pathlib import Path

import torch

from .closed_loop_lab import compare_scores, toy_path
from .truth_pretrain import atomic_json


def compare(directory):
    directory = toy_path(directory)
    early, late = directory/'round02', directory/'round03'
    a, b = [json.loads((p/'report.json').read_text()) for p in (early, late)]
    for key in ('classifier_seed', 'datasets'):
        if a['plan'][key] != b['plan'][key]:
            raise ValueError(f'Changed factor other than fit budget: {key}')
    panels_equal = {}
    for dataset in a['plan']['datasets']:
        p, q = [torch.load(folder/f'{dataset}_panels.pt', map_location='cpu', weights_only=True)
                for folder in (early, late)]
        panels_equal[dataset] = all(torch.equal(p[split][key], q[split][key])
                                   for split in p for key in p[split])
    if not all(panels_equal.values()):
        raise ValueError('Evaluation populations differ')
    prior_rows = {(r['arm'], r['step']): r for r in a['fit']['history'] if r['arm'].endswith('__plain')}
    current_rows = {(r['arm'], r['step']): r for r in b['fit']['history']}
    # Before the early stop the entire training path must agree; stopped flag may differ.
    prefix_equal = all(all(row[k] == current_rows[key][k]
                         for k in ('train_bce', 'validation_bce', 'gradient_norm', 'selected_step'))
                       for key, row in prior_rows.items())
    if not prefix_equal:
        raise ValueError('Budget replay changed the earlier training trajectory')
    old_scores, new_scores = [torch.load(folder/'classifiers/test_scores.pt',
                              map_location='cpu', weights_only=True) for folder in (early, late)]
    all_scores, pairs = {}, []
    for dataset in a['plan']['datasets']:
        plain, both = f'{dataset}__plain', f'{dataset}__both'
        short, long, strong = f'{dataset}_short_plain', f'{dataset}_long_plain', f'{dataset}_fourier'
        all_scores.update({short: old_scores[plain], long: new_scores[plain], strong: old_scores[both]})
        pairs.extend([(long, short), (strong, long)])
    contrasts = compare_scores(all_scores, pairs, repeats=2000)
    endpoint = contrasts['step1000_long_plain minus step1000_short_plain']
    result = {'panel_equality': panels_equal, 'exact_prefix_training_replay': prefix_equal,
              'contrasts': contrasts,
              'endpoint_early_stop_false_negative_supported':
                  endpoint['bce']['hi95'] < -.002 and endpoint['auc_gap']['lo95'] > 0,
              'scope': 'Eight simultaneous contrasts per metric; fixed fitted classifiers. '
                       'The short endpoint passed the old patience criterion but failed this longer-budget challenge.'}
    atomic_json(late/'budget_comparison.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    compare(p.parse_args().directory)
