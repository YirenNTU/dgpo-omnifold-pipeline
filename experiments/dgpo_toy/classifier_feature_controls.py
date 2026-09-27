"""Exact saved-control loading for toy classifier input-branch experiments."""
import json
import math

import torch

from .reward_transfer_comparison import assert_same_panel


def load_controls(plan, panels):
    from .closed_loop_lab import toy_path, score_metrics
    scores, metadata = {}, {}
    for alias, spec in plan['saved_classifier_controls'].items():
        folder = toy_path(spec['directory'])
        report = json.loads((folder/'report.json').read_text())
        old_plan, fit = report['plan'], report['fit']
        key = spec['key']
        dataset, mode = key.rsplit('__', 1)
        if (fit.get('width', 128) != plan.get('classifier_width', 128)
                or old_plan['classifier_seed'] != plan['classifier_seed']
                or not fit['cold_start'] or not fit['test'][key]['valid']):
            raise ValueError('Saved classifier width/seed/adequacy mismatch')
        if (fit['minimum_steps'] != plan['minimum_fit_steps']
                or fit['max_steps'] != plan['max_fit_steps']
                or fit['check_every'] != 100 or fit['patience_checks'] != 20
                or fit['min_delta'] != 1e-4):
            raise ValueError('Saved classifier stopping protocol mismatch')
        original = torch.load(folder/f'{dataset}_panels.pt', map_location='cpu', weights_only=True)
        for split in ('train', 'validation', 'test'):
            assert_same_panel(original[split], panels[spec['current_dataset']][split], include_negative=True)
        score = torch.load(folder/'classifiers/test_scores.pt', map_location='cpu', weights_only=True)[key]
        if not math.isclose(score_metrics(score)['bce'], fit['test'][key]['bce'], abs_tol=1e-10):
            raise ValueError('Selected saved scores do not match report')
        scores[alias] = score
        metadata[alias] = {'directory': str(folder), 'key': key, 'feature': mode,
                           'width': fit.get('width', 128), **fit['test'][key]}
    return scores, metadata
