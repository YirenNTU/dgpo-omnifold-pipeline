"""Post-hoc paired fixed-reward comparison. No training or new diffusion draws."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .closed_loop_lab import HISTORY, assert_pairing, toy_path
from .nonperiodic_cube import Critic
from .truth_pretrain import atomic_checkpoint, atomic_json


def curve_average(times, values):
    """Per-context trapezoidal average on saved milestones, NOT a dense integral."""
    t = torch.as_tensor(times, dtype=torch.float64)
    x = torch.stack(values).double()
    if t.ndim != 1 or len(t) != len(x) or len(t) < 2 or t[0] != 0 or not (t.diff() > 0).all():
        raise ValueError('Need increasing milestones beginning at zero')
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite reward curve')
    return ((x[:-1] + x[1:]) * .5 * t.diff()[:, None]).sum(0) / t[-1]


def paired_intervals(series, repeats=2000, seed=831093):
    """IID test contexts, joint nonstudentized max-deviation bootstrap family."""
    if repeats < 100 or not series:
        raise ValueError('Need a nonempty contrast family and >=100 repeats')
    names = list(series)
    values = np.stack([series[n].double().numpy() for n in names], 1)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Invalid context-level contrasts')
    means = values.mean(0)
    rng = np.random.default_rng(seed)
    deviations = np.empty(repeats)
    for i in range(repeats):
        sampled = values[rng.integers(len(values), size=len(values))].mean(0)
        deviations[i] = np.max(np.abs(sampled - means))
    radius = float(np.quantile(deviations, .95))
    return {name: {'mean': float(value), 'lo95': float(value-radius),
                   'hi95': float(value+radius)} for name, value in zip(names, means)}


def assert_same_panel(left, right, *, include_negative=False):
    for key in ('c', 'positive', 'negative') if include_negative else ('c', 'positive'):
        if not torch.equal(left[key], right[key]):
            raise ValueError('Unpaired panel field: '+key)


def check_factor(left, right, expected):
    a, b = ({'encoder_gain': 1., **v} for v in (left, right))
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    if changed != {expected}:
        raise ValueError('Unexpected conditioning difference: '+str(changed))


@torch.no_grad()
def score(critic, panel):
    value = torch.cat([critic(y, c) for y, c in
                       zip(panel['negative'].split(1024), panel['c'].split(1024))]).double()
    if value.ndim != 1 or not torch.isfinite(value).all():
        raise ValueError('Nonfinite or incorrectly shaped reward')
    return value


def gain_distribution(values):
    values = values.double()
    positive = values.clamp_min(0)
    k = max(1, int(np.ceil(len(values)*.01)))
    return {'mean': float(values.mean()), 'median': float(values.median()),
            'q05': float(values.quantile(.05)), 'q95': float(values.quantile(.95)),
            'fraction_positive': float((values > 0).double().mean()),
            'top1pct_share_of_positive_gain': (float(positive.topk(k).values.sum()/positive.sum())
                                               if positive.sum() > 0 else None)}


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan['kind'] != 'reward_transfer_comparison' or not 1 <= plan['round'] <= 20:
        raise ValueError('Invalid toy-only analysis plan')
    output = toy_path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'process.json', {'pid': os.getpid(), 'start_time': time.time()})
    start = time.monotonic()
    atomic_json(output/'status.json', {'state': 'running', 'round': plan['round']})
    try:
        reward_path = HISTORY/'rewards/full_reward.pt'
        ck = torch.load(reward_path, map_location='cpu', weights_only=True)
        critic = Critic(True).eval().requires_grad_(False)
        critic.load_state_dict(ck['model'])
        fits = json.loads((HISTORY/'rewards/fit_report.json').read_text())
        if not fits['arms']['full']['valid']:
            raise ValueError('Invalid frozen reward fit')
        times = [0, *plan['milestones']]
        records, gains, points = {}, {}, {}
        baseline_panel = baseline_grid = baseline_scores = None
        for name, spec in plan['sources'].items():
            folder = toy_path(spec['directory'])
            old = json.loads((folder/'report.json').read_text())
            old_plan = json.loads((folder/'plan.json').read_text())
            if old['state'] != 'completed' or not old['valid']:
                raise ValueError('Source not completed and valid: '+name)
            if old['reward'] != str(reward_path) or old['velocity_coefficient'] != 1.:
                raise ValueError('Different reward or reference coefficient')
            match = old['initial_matching'][spec['arm']]
            if not match['velocity_exact'] or not match['samples_exact']:
                raise ValueError('Nonmatching initial policy')
            if old_plan['milestones'] != plan['milestones']:
                raise ValueError('Different saved milestones')
            if records:
                first = next(iter(records.values()))
                if old['config'] != first['report']['config']:
                    raise ValueError('Different training config')
                for key in ('initialization_seed','policy_seed','monitor_seed','panel_seed','classifier_seed'):
                    if old_plan[key] != first['plan'][key]:
                        raise ValueError('Different seed/sampling clock: '+key)
            base = torch.load(folder/'baseline_step0_panels.pt', map_location='cpu', weights_only=True)
            grid = torch.load(folder/'baseline_endpoint.pt', map_location='cpu', weights_only=True)
            if baseline_panel is None:
                baseline_panel, baseline_grid = base, grid
                baseline_scores = score(critic, base['test'])
            else:
                for split in ('train','validation','test'):
                    assert_same_panel(base[split], baseline_panel[split], include_negative=True)
                if not torch.equal(grid['contexts'], baseline_grid['contexts']) or not torch.equal(
                        grid['rewards']['full'], baseline_grid['rewards']['full']):
                    raise ValueError('Nonmatching source endpoint panel')
            records[name] = {'folder': folder, 'report': old, 'plan': old_plan, 'arm': spec['arm']}
            gains[name], points[name] = {0: torch.zeros_like(baseline_scores)}, {}
            for step in plan['milestones']:
                panels = torch.load(folder/f"{spec['arm']}_step{step}_panels.pt", map_location='cpu', weights_only=True)
                assert_pairing({'baseline': baseline_panel, 'current': panels})
                gain = score(critic, panels['test']) - baseline_scores
                gains[name][step] = gain
                grid_after = torch.load(folder/f"{spec['arm']}_endpoint{step}.pt", map_location='cpu', weights_only=True)
                if not torch.equal(grid_after['contexts'], baseline_grid['contexts']):
                    raise ValueError('Endpoint condition grid differs')
                grid_gain = (grid_after['rewards']['full'].double()-baseline_grid['rewards']['full'].double()).mean(1)
                point = old['points'][f"{spec['arm']}_{step}"]
                if abs(float(grid_gain.mean())-point['reward_gain']['gain']) > 1e-7:
                    raise ValueError('Saved reward mean does not replay')
                audit = old['audits'][str(step)]['test'][spec['arm']+'__both']
                if not audit['valid'] or audit['fit_steps'] < 8000:
                    raise ValueError('Inadequate saved diagnostic audit')
                points[name][str(step)] = {
                    'iid_test_gain': gain_distribution(gain),
                    'grid_conditional_mean_gain': gain_distribution(grid_gain),
                    'reward_decomposition': point['reward_decomposition'],
                    'mode_tv': point['metrics']['mode_tv'],
                    'parity_mae': point['metrics']['parity_mae'],
                    'fixed_probe_velocity_mse': point['conditioning']['fixed_probe_velocity_mse'],
                    'fresh_audit': {k: v for k,v in audit.items() if k != 'weight'},
                }
                print(f"Scored {name} step{step}: mean gain {gain.mean():+.6f}", flush=True)
                atomic_json(output/'status.json', {'state':'running','round':plan['round'],
                                                   'active':{'arm':name,'step':step}})
        primary, curves, temporal = {}, {}, {}
        for contrast in plan['contrasts']:
            name, left, right = (contrast[k] for k in ('name','left','right'))
            a,b = records[left],records[right]
            check_factor(a['plan']['conditioning'][a['arm']], b['plan']['conditioning'][b['arm']], contrast['factor'])
            primary[name] = gains[left][times[-1]]-gains[right][times[-1]]
            curves[name] = curve_average(times, [gains[left][t]-gains[right][t] for t in times])
            for t in times[1:]:
                temporal[f'{name}/step{t}'] = gains[left][t]-gains[right][t]
        primary_ci = paired_intervals(primary, plan['bootstrap_repeats'], plan['bootstrap_seed'])
        curve_ci = paired_intervals(curves, plan['bootstrap_repeats'], plan['bootstrap_seed']+1)
        temporal_ci = paired_intervals(temporal, plan['bootstrap_repeats'], plan['bootstrap_seed']+2)
        decisions = {n: ('supports_reward_gain' if v['lo95'] > plan['material_reward_margin'] else
                         'opposes_reward_gain' if v['hi95'] < -plan['material_reward_margin'] else
                         'unresolved_material_gain') for n,v in primary_ci.items()}
        report = {'state':'completed','round':plan['round'],'valid':True,
                  'primary':primary_ci,'curve_average':curve_ci,'temporal':temporal_ci,
                  'decisions':decisions,'points':points,'scope':plan['scope'],
                  'reward':str(reward_path),'test_contexts':len(baseline_scores),
                  'baseline_test_mean_reward':float(baseline_scores.mean()),
                  'validity':{'matched_config_and_seeds':True,'initial_functions_exact':True,
                              'source_samples_exact':True,'all_panels_paired':True,
                              'one_declared_factor_per_contrast':True,'saved_grid_reward_replay':True},
                  'elapsed_seconds':time.monotonic()-start}
        atomic_checkpoint(output/'paired_reward_gains.pt', {'contexts':baseline_panel['test']['c'],
                          'gains':gains,'primary':primary,'curve_average':curves})
        atomic_json(output/'report.json', report)
        lines = ['# Round13: fixed-reward absorption on saved paired samples','',
                 'No model training, new generation, seed replication or reward transformation.',
                 'Post-hoc exploratory comparison; intervals condition on fitted policies and critic.','',
                 '| Contrast | Final reward difference [95%] | Sparse curve-average difference [95%] | Decision |',
                 '| --- | --- | --- | --- |']
        for n,v in primary_ci.items():
            c = curve_ci[n]
            lines.append(f"| {n} | {v['mean']:+.6f} [{v['lo95']:+.6f}, {v['hi95']:+.6f}] | "
                         f"{c['mean']:+.6f} [{c['lo95']:+.6f}, {c['hi95']:+.6f}] | {decisions[n]} |")
        lines += ['', 'Primary simultaneous family: three final contrasts. Curve and temporal families separate.',
                  'Material margin .01 is exploratory, chosen after seeing earlier grid point estimates.',
                  'Curve average uses only0/25/100/300/1000, not a dense learning curve or measured time-to-target.',
                  'Same update/rollout budget does not mean matched velocity distance or wall-clock FLOPs.',
                  'IID test: one paired draw/context. Positive sample-gain fraction is not per-condition expected improvement.',
                  'Grid conditional averages use512 draws at64 nodes; grid variation is descriptive, not training-seed uncertainty.',
                  f"Elapsed {report['elapsed_seconds']:.2f}s. Full diagnostics: report.json.", '']
        (output/'SUMMARY.md').write_text('\n'.join(lines))
        atomic_json(output/'status.json', {'state':'awaiting_review','result_state':'completed',
                                           'round':plan['round'],'elapsed_seconds':report['elapsed_seconds']})
        return report
    except Exception as exc:
        atomic_json(output/'status.json', {'state':'failed','round':plan['round'],'error':repr(exc)})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    run(args.plan, args.output)
