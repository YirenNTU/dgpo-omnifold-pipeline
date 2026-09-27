"""Rescore saved source swaps to identify what narrow classifiers detect."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from . import conditional as native
from .closed_loop_lab import FeatureCritic, load_source, predictions, score_metrics, toy_path
from .classifier_signal_attribution import (
    bce_by_condition, cell_means, probabilities, score_decomposition, simultaneous_mean_intervals)
from .cube_swap import modes, truth_shape
from .nonperiodic_cube import Data
from .reward_transfer_comparison import assert_same_panel
from .truth_pretrain import atomic_checkpoint, atomic_json


def replay_panels(pool, centers, samples, seed):
    """Replay the original cube_swap RNG transaction without generating diffusion."""
    c, y, ids, p = (pool[k] for k in ('c', 'y', 'ids', 'p'))
    grid, donors, dimensions = y.shape
    if dimensions != 3 or not 1 <= samples <= donors or not torch.isfinite(y).all():
        raise ValueError('Invalid saved pool')
    if not torch.equal(ids, modes(y)):
        raise ValueError('Mode IDs do not match saved samples')
    counts = torch.stack([torch.bincount(row, minlength=8) for row in ids])
    if not torch.equal(counts, pool['counts']) or counts.min() < 16:
        raise ValueError('Sparse or inconsistent donor counts')
    rng = native.generator(seed)
    # The original draw of DDIM initial noise precedes conditional donor draws.
    torch.randn(grid, donors, 3, generator=rng)
    trng = native.generator(seed + 100000)
    target_ids = torch.multinomial(p, samples, replacement=True, generator=trng)
    positive_ids = torch.multinomial(p, samples, replacement=True, generator=trng)
    a = truth_shape(target_ids, centers, trng)
    positive = truth_shape(positive_ids, centers, trng)
    d = y[:, :samples].clone()
    b = truth_shape(ids[:, :samples], centers, native.generator(seed + 200000))
    shaped = torch.empty_like(a)
    for j in range(grid):
        for mode in range(8):
            slots = (target_ids[j] == mode).nonzero().flatten()
            choices = (ids[j] == mode).nonzero().flatten()
            draws = torch.randint(len(choices), (len(slots),), generator=rng)
            shaped[j, slots] = y[j, choices[draws]]
    if not torch.equal(modes(a), modes(shaped)) or not torch.equal(modes(b), modes(d)):
        raise ValueError('Counterfactual mode pairing failed')
    cc = c[:, None].expand(-1, samples, -1).reshape(-1, 1)
    return {name: {'c': cc, 'positive': positive.reshape(-1, 3), 'negative': value.reshape(-1, 3)}
            for name, value in [('A', a), ('B', b), ('C', shaped), ('D', d)]}


def content_contrasts(bces):
    result = {}
    for component in ('B', 'C', 'D'):
        for width in ('narrow', 'wide'):
            both, plain = (bces[f'{width}_{feature}'] for feature in ('both', 'plain'))
            result[f'{width}/{component}/fourier_minus_plain'] = (
                both[component] - both['A'] - plain[component] + plain['A'])
        result[f'width_interaction/{component}'] = (
            result[f'narrow/{component}/fourier_minus_plain']
            - result[f'wide/{component}/fourier_minus_plain'])
    return result


def branch_content_contrasts(bces):
    pairs = [('c_only', 'narrow_plain'), ('y_only', 'narrow_plain'),
             ('narrow_both', 'c_only'), ('narrow_both', 'y_only'), ('c_only', 'y_only')]
    return {f'{component}/{left}_minus_{right}':
            bces[left][component] - bces[left]['A'] - bces[right][component] + bces[right]['A']
            for component in ('B', 'C', 'D') for left, right in pairs}


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan['kind'] not in ('width_signal_attribution', 'branch_signal_attribution') or not 1 <= plan['round'] <= 20:
        raise ValueError('Invalid toy-only plan')
    campaign = toy_path(plan['campaign'])
    output = toy_path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'process.json', {'pid': os.getpid(), 'start_time': time.time()})
    def emit(row):
        atomic_json(output/'status.json', {'state': 'running', 'round': plan['round'],
                    'active': row, 'elapsed_seconds': time.monotonic()-start})
        print(json.dumps(row), flush=True)
    try:
        old = toy_path(campaign/plan['pool_source'])
        old_plan = json.loads((old/'plan.json').read_text())
        old_report = json.loads((old/'report.json').read_text())
        if old_report['state'] != 'completed':
            raise ValueError('Source attribution incomplete')
        pool = torch.load(old/'step0_pool.pt', map_location='cpu', weights_only=True)
        historical = torch.load(old/'attribution_arrays.pt', map_location='cpu', weights_only=True)['0']
        cfg, _ = load_source()
        data = Data(cfg)
        if not torch.equal(pool['p'], data.probabilities(pool['c'][:, 0], .9)):
            raise ValueError('Source diagnostic truth changed')
        panels = replay_panels(pool, data.centers, old_plan['samples_per_condition'], old_plan['sample_seed'])
        grid = len(pool['c'])
        ty = truth_shape(torch.arange(8)[None, :, None].expand(grid, -1, old_plan['truth_shape_per_cell']),
                         data.centers, native.generator(old_plan['sample_seed']+300000))
        q = probabilities(pool['ids'])
        report = {'state': 'running', 'round': plan['round'], 'judges': {}, 'scope': plan['scope']}
        arrays, bces, third_order = {}, {}, {}
        anchor = None
        for name, spec in plan['judges'].items():
            emit({'phase': 'score_saved_judge', 'judge': name})
            directory = toy_path(campaign/spec['directory'])
            fit = json.loads((directory/'classifiers/fit_report.json').read_text())
            key = 'step0__'+spec['feature']
            if not fit['test'][key]['valid'] or not fit['cold_start'] or fit['seed'] != 83:
                raise ValueError('Invalid source judge fit')
            train_panels = torch.load(directory/'step0_panels.pt', map_location='cpu', weights_only=True)
            if anchor is None:
                anchor = train_panels
            for split in ('train', 'validation', 'test'):
                assert_same_panel(anchor[split], train_panels[split], include_negative=True)
            ck = torch.load(directory/f'classifiers/{key}_best.pt', map_location='cpu', weights_only=True)
            if ck['feature'] != spec['feature'] or ck.get('width', 128) != spec['width']:
                raise ValueError('Judge feature/width mismatch')
            model = FeatureCritic(spec['feature'], width=spec['width']).eval().requires_grad_(False)
            model.load_state_dict(ck['model'])
            scores = {label: predictions(model, panel) for label, panel in panels.items()}
            if name.startswith('wide_'):
                for label in panels:
                    for key_score in ('positive', 'negative'):
                        if not torch.equal(scores[label][key_score], historical['judges'][spec['feature']]['scores'][label][key_score]):
                            raise ValueError('Saved wide predictions do not replay exactly')
            bces[name] = {label: bce_by_condition(score, grid) for label, score in scores.items()}
            truth_means, gen_means = cell_means(model, pool, ty)
            decomposition = score_decomposition(pool['p'], q, truth_means, gen_means, data.centers)
            third_order[name] = decomposition['order3_mode_logit_gap']
            report['judges'][name] = {'width': spec['width'], 'feature': spec['feature'],
                'fit': fit['test']['step0__'+spec['feature']],
                'swaps': {label: score_metrics(score) for label, score in scores.items()},
                'logit_gap': {key: float(value.mean()) for key, value in decomposition.items()}}
            arrays[name] = {'scores': scores, 'bce_by_condition': bces[name], 'decomposition': decomposition}
        contrast_function = (branch_content_contrasts if plan['kind'] == 'branch_signal_attribution'
                             else content_contrasts)
        report['bce_contrasts'] = simultaneous_mean_intervals(contrast_function(bces),
            plan['bootstrap_repeats'], plan['bootstrap_seed'])
        report['order3_descriptive_intervals'] = simultaneous_mean_intervals(third_order,
            plan['bootstrap_repeats'], plan['bootstrap_seed']+1)
        report.update(state='completed', elapsed_seconds=time.monotonic()-start,
                      wide_score_replay_exact=True, all_training_panels_exact=True,
                      classifier_fits=0, policy_updates=0, new_diffusion_draws=0)
        atomic_checkpoint(output/'arrays.pt', arrays)
        atomic_json(output/'report.json', report)
        title = '# Fourier input branch x classifier content' if plan['kind'] == 'branch_signal_attribution' else '# Width x classifier content'
        lines = [title, '',
            'Frozen judges, not refits. A truth/truth; B generated modes/truth shape;',
            'C truth modes/generated shape; D unchanged generator. Descriptive grid uncertainty only.', '',
            '| Judge | A AUC | B AUC | C AUC | D AUC | B BCE | C BCE | order3 logit gap |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name, record in report['judges'].items():
            ss = record['swaps']
            values = [ss[k]['auc'] for k in ('A','B','C','D')] + [ss[k]['bce'] for k in ('B','C')]
            values += [record['logit_gap']['order3_mode_logit_gap']]
            lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
        lines += ['', 'Primary component interactions:', '']
        for label in ('B','C','D'):
            key = label+'/c_only_minus_y_only' if plan['kind'] == 'branch_signal_attribution' else 'width_interaction/'+label
            value = report['bce_contrasts'][key]
            lines.append(f"- {label}: {value['mean']:+.6f} [{value['lo95']:+.6f}, {value['hi95']:+.6f}].")
        lines += ['', f"Elapsed {report['elapsed_seconds']:.2f}s; zero training and zero new diffusion draws.", plan['scope'], '']
        (output/'SUMMARY.md').write_text('\n'.join(lines))
        atomic_json(output/'status.json', {'state': 'awaiting_review', 'round': plan['round'],
                    'result_state': 'completed', 'elapsed_seconds': report['elapsed_seconds']})
    except BaseException as exc:
        atomic_json(output/'status.json', {'state': 'failed_or_interrupted', 'error': repr(exc)})
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.plan, args.output)
