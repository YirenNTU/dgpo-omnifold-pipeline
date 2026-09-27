"""Complete the Fourier x FiLM factorial with ONE new raw-shift trajectory.

Reuse the native closed-loop runner and three immutable completed controls.
Preflight is read-only; only an explicit invocation without --preflight trains.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch

from . import closed_loop_lab as lab
from . import reward_transfer_comparison as paired
from .nonperiodic_cube import Critic
from .truth_pretrain import atomic_checkpoint, atomic_json

DEFAULT_PLAN = lab.ROOT / 'experiments/dgpo_toy/plans/fourier_film_factorial.json'
CELLS = {
    'raw_shift': ('raw', 'shift'),
    'raw_film': ('raw', 'film'),
    'fourier_shift': ('fourier', 'shift'),
    'fourier_film': ('fourier', 'film'),
}
MATCH_FIELDS = ('initialization_seed', 'policy_seed', 'monitor_seed', 'panel_seed',
                'classifier_seed', 'audit_features', 'minimum_fit_steps',
                'max_fit_steps', 'milestones')


def load_json(path):
    return json.loads(Path(path).read_text())


def architecture(cell):
    basis, modulation = CELLS[cell]
    return dict(basis=basis, modulation=modulation, nonlinear=True,
                normalization=True, layers=3, width=32)


def validate_plan(plan):
    lab.validate_plan(plan)
    if plan['kind'] != 'rl' or plan['conditioning'] != {'raw_shift': architecture('raw_shift')}:
        raise ValueError('Train only the missing raw-shift cell from its matched initialization')
    if plan['milestones'] != [25, 100, 300, 1000] or plan['audit_features'] != ['both']:
        raise ValueError('Keep the archived milestone and fresh-H4 audit contract')
    if plan['minimum_fit_steps'] != 8000 or plan['max_fit_steps'] != 32000:
        raise ValueError('Keep the matched 8000..32000 fresh-classifier budget')
    spec = plan['factorial']
    if set(spec['sources']) != set(CELLS) - {'raw_shift'}:
        raise ValueError('Provide exactly three saved factorial controls')
    if spec['material_reward_margin'] != .01 or plan['bootstrap_repeats'] != 2000:
        raise ValueError('Do not silently change the declared interval or material-gain rule')
    control = plan['external_control']
    if control != {**spec['sources']['raw_film'], 'changed_factor': 'modulation'}:
        raise ValueError('The training control must be saved raw FiLM, changing only modulation')


def checked_record(cell, spec, plan, cfg):
    folder = lab.toy_path(spec['directory'])
    report, old_plan = load_json(folder/'report.json'), load_json(folder/'plan.json')
    if report.get('state') != 'completed' or not report.get('valid'):
        raise ValueError(f'Incomplete or invalid control: {cell}')
    if any(old_plan.get(key) != plan.get(key) for key in MATCH_FIELDS):
        raise ValueError(f'Control sampling/initialization/audit clock mismatch: {cell}')
    arm = spec['arm']
    if old_plan['conditioning'][arm] != architecture(cell):
        raise ValueError(f'Wrong factorial architecture or hidden intervention: {cell}')
    if report['config'] != asdict(cfg) or report['velocity_coefficient'] != 1.:
        raise ValueError(f'Control objective/training config mismatch: {cell}')
    if Path(report['reward']).resolve() != (lab.HISTORY/'rewards/full_reward.pt').resolve():
        raise ValueError('Do not mix full-truth, shape-matched or other reward classifiers')
    matching = report['initial_matching'][arm]
    if not matching['velocity_exact'] or not matching['samples_exact']:
        raise ValueError(f'Initial generator did not match: {cell}')
    for step in (0, *plan['milestones']):
        name = 'baseline' if step == 0 else arm
        audit = report['audits'][str(step)]['test'][name+'__both']
        if not audit['valid'] or not audit['plateau'] or audit['fit_steps'] < 8000:
            raise ValueError(f'Inadequate diagnostic audit: {cell}/{step}')
        if not (folder/f'{name}_step{step}_panels.pt').is_file():
            raise FileNotFoundError(f'Missing saved audit panel: {cell}/{step}')
        if step and not (folder/f'{arm}_endpoint{step}.pt').is_file():
            raise FileNotFoundError(f'Missing saved endpoint: {cell}/{step}')
    state = torch.load(folder/f'{arm}_step1000.pt', map_location='cpu', weights_only=True)
    if (state['step'] != 1000 or len(state['history']) != 1000 or
            state['conditioning'] != architecture(cell) or
            state['config'] != asdict(cfg) or state['velocity_coefficient'] != 1.):
        raise ValueError(f'Saved checkpoint does not match the completed trajectory: {cell}')
    return {'folder': folder, 'arm': arm, 'report': report, 'plan': old_plan}


def matched_sources(plan, new_folder=None):
    """Validate provenance plus actual baseline arrays, not only configuration text."""
    validate_plan(plan)
    lab.validate_external_control(plan)
    with torch.random.fork_rng():
        cfg, source = lab.load_source()
    cfg = replace(cfg, policy_steps=1000, eval_every=25, eval_events=512)
    specs = dict(plan['factorial']['sources'])
    if new_folder is not None:
        specs['raw_shift'] = {'directory': str(new_folder), 'arm': 'raw_shift'}
    records = {cell: checked_record(cell, spec, plan, cfg) for cell, spec in specs.items()}
    baseline = baseline_grid = baseline_scores = None
    for cell, record in records.items():
        folder = record['folder']
        pp = torch.load(folder/'baseline_step0_panels.pt', map_location='cpu', weights_only=True)
        grid = torch.load(folder/'baseline_endpoint.pt', map_location='cpu', weights_only=True)
        scores = torch.load(folder/'test_scores.pt', map_location='cpu', weights_only=True)
        scores = scores['baseline__both_step0']
        if baseline is None:
            baseline, baseline_grid, baseline_scores = pp, grid, scores
        else:
            for split in ('train', 'validation', 'test'):
                paired.assert_same_panel(pp[split], baseline[split], include_negative=True)
            if not torch.equal(grid['contexts'], baseline_grid['contexts']) or not torch.equal(
                    grid['rewards']['full'], baseline_grid['rewards']['full']):
                raise ValueError(f'Baseline grid does not replay: {cell}')
            if any(not torch.equal(scores[label], baseline_scores[label])
                   for label in ('positive', 'negative')):
                raise ValueError(f'Baseline fresh-classifier predictions do not replay: {cell}')
    return cfg, source, records, baseline, baseline_grid


def preflight(plan):
    cfg, source, records, baseline, _ = matched_sources(plan)
    policies = {}
    with torch.random.fork_rng():
        for cell in CELLS:
            torch.manual_seed(plan['initialization_seed'])
            policies[cell] = lab.ConditioningAblation(cfg, source.state_dict(), **architecture(cell)).eval()
        reward = Critic(True)
        reward.load_state_dict(torch.load(lab.HISTORY/'rewards/full_reward.pt',
                                         map_location='cpu', weights_only=True)['model'])
    if not load_json(lab.HISTORY/'rewards/fit_report.json')['arms']['full']['valid']:
        raise ValueError('Frozen reward classifier lacks an adequate fit')
    matching = lab.film.previous.verify_matching(source, policies, cfg)
    return {'state': 'ready_not_started', 'new_policy_trajectories': 1,
            'reused_controls': list(records), 'initial_matching': matching,
            'test_contexts': len(baseline['test']['c']), 'config': asdict(cfg),
            'velocity_coefficient': 1., 'training_started': False}


def factorial_contrasts(gains):
    """Positive interaction means Fourier increases the benefit of enabling scale."""
    if set(gains) != set(CELLS):
        raise ValueError('Need all four factorial cells')
    rs, rf, fs, ff = (gains[k].double() for k in CELLS)
    if rs.ndim != 1 or any(v.shape != rs.shape or not torch.isfinite(v).all()
                           for v in (rs, rf, fs, ff)):
        raise ValueError('Need paired finite per-context gains')
    return {
        'film_effect_raw': rf-rs,
        'film_effect_fourier': ff-fs,
        'fourier_effect_shift': fs-rs,
        'fourier_effect_film': ff-rf,
        'interaction': (ff-fs)-(rf-rs),
    }


def classify_interval(value, margin, *, interaction=False):
    if value['lo95'] > margin:
        return 'supports_positive_interaction' if interaction else 'supports_material_reward_gain'
    if value['hi95'] < -margin:
        return 'supports_negative_interaction' if interaction else 'supports_material_reward_loss'
    if interaction and value['lo95'] >= -margin and value['hi95'] <= margin:
        return 'interaction_within_declared_small_effect_band'
    return 'unresolved'


def analyze(plan, output):
    cfg, _, records, baseline, baseline_grid = matched_sources(plan, output/'raw_shift')
    with torch.random.fork_rng():
        critic = Critic(True).eval().requires_grad_(False)
        critic.load_state_dict(torch.load(lab.HISTORY/'rewards/full_reward.pt',
                                         map_location='cpu', weights_only=True)['model'])
    baseline_reward = paired.score(critic, baseline['test'])
    gains, points = {}, {}
    times = [0, *plan['milestones']]
    for cell, record in records.items():
        folder, arm, old = record['folder'], record['arm'], record['report']
        gains[cell] = {0: torch.zeros_like(baseline_reward)}
        points[cell] = {'0': {'iid_test_gain': paired.gain_distribution(gains[cell][0]),
                              'metrics': old['baseline'], 'fixed_probe_velocity_mse': 0.,
                              'fresh_audit': old['audits']['0']['test']['baseline__both']}}
        for step in plan['milestones']:
            panels = torch.load(folder/f'{arm}_step{step}_panels.pt', map_location='cpu', weights_only=True)
            lab.assert_pairing({'baseline': baseline, 'current': panels})
            gain = paired.score(critic, panels['test'])-baseline_reward
            grid = torch.load(folder/f'{arm}_endpoint{step}.pt', map_location='cpu', weights_only=True)
            if not torch.equal(grid['contexts'], baseline_grid['contexts']):
                raise ValueError('Different endpoint context grid')
            grid_gain = (grid['rewards']['full'].double()-baseline_grid['rewards']['full'].double()).mean(1)
            point = old['points'][f'{arm}_{step}']
            if abs(float(grid_gain.mean())-point['reward_gain']['gain']) > 1e-7:
                raise ValueError('Saved reward grid fails replay')
            gains[cell][step] = gain
            points[cell][str(step)] = {
                'iid_test_gain': paired.gain_distribution(gain),
                'grid_conditional_mean_gain': paired.gain_distribution(grid_gain),
                'metrics': point['metrics'], 'reward_decomposition': point['reward_decomposition'],
                'fixed_probe_velocity_mse': point['conditioning']['fixed_probe_velocity_mse'],
                'fresh_audit': old['audits'][str(step)]['test'][arm+'__both'],
            }
    by_time = {t: factorial_contrasts({cell: gains[cell][t] for cell in CELLS}) for t in times}
    curves = {name: paired.curve_average(times, [by_time[t][name] for t in times])
              for name in by_time[1000]}
    repeats, seed = plan['bootstrap_repeats'], plan['factorial']['bootstrap_seed']
    intervals = paired.paired_intervals(by_time[1000], repeats, seed)
    decisions = {name: classify_interval(value, plan['factorial']['material_reward_margin'],
                                         interaction=name == 'interaction')
                 for name, value in intervals.items()}
    result = {'state': 'completed', 'primary': intervals, 'decisions': decisions,
              'curve_average': paired.paired_intervals(curves, repeats, seed+1),
              'points': points, 'config': asdict(cfg), 'test_contexts': len(baseline_reward),
              'milestones': times, 'baseline_reward': float(baseline_reward.mean()),
              'sources': {cell: {'directory': str(r['folder']), 'arm': r['arm']}
                          for cell, r in records.items()},
              'scope': plan['scope'], 'validity': {'all_four_cells_paired': True,
                  'initial_samples_exact': True, 'source_audit_predictions_exact': True,
                  'saved_reward_grid_replays': True, 'all_fresh_audits_adequate': True,
                  'velocity_coefficient': 1., 'one_new_policy_trajectory': True}}
    atomic_checkpoint(output/'paired_reward_gains.pt', {'contexts': baseline['test']['c'],
                      'gains': gains, 'primary': by_time[1000], 'curve_average': curves})
    atomic_json(output/'report.json', result)
    (output/'SUMMARY.md').write_text(summarize(result))
    plot(result, output)
    return result


def summarize(result):
    lines = ['# Fourier x FiLM: one missing cell, four paired trajectories', '',
             'Only raw shift was newly trained. Fixed full-truth H4; velocity half-MSE coefficient1.',
             'Primary: interaction = (Fourier FiLM - Fourier shift) - (raw FiLM - raw shift).', '',
             '| Policy | Held-out fixed reward gain | Fresh AUC | Fresh BCE | Mode TV | Velocity probe MSE |',
             '|---|---:|---:|---:|---:|---:|']
    for cell in CELLS:
        p = result['points'][cell]['1000']
        a = p['fresh_audit']
        lines.append(f"| {cell} | {p['iid_test_gain']['mean']:+.6f} | {a['auc']:.6f} | "
                     f"{a['bce']:.6f} | {p['metrics']['mode_tv']:.6f} | {p['fixed_probe_velocity_mse']:.6f} |")
    lines += ['', '| Contrast | Difference [simultaneous95%] | Decision |', '|---|---|---|']
    for name, v in result['primary'].items():
        lines.append(f"| {name} | {v['mean']:+.6f} [{v['lo95']:+.6f}, {v['hi95']:+.6f}] | {result['decisions'][name]} |")
    lines += ['', 'The family contains four simple effects and their interaction; material margin=.01.',
              'A positive interaction is extra FiLM benefit with Fourier, not proof it is necessary.',
              'Unresolved is NOT zero interaction. The small-effect label applies only to the declared margin.',
              'All-sample reward, fresh discrimination and joint structure are separate measurements.',
              f"Test reward:{result['test_contexts']} paired IID contexts, one draw/context. Training uses K8; no best-of-K endpoint.",
              'Equal updates/initial outputs do not match active rank, effective capacity, FLOPs or velocity distance.',
              'Curve-average intervals are a separate family on only0/25/100/300/1000, not dense time-to-target.',
              'Adaptive single-training-seed exploration; bootstrap excludes training/architecture-selection uncertainty.',
              'Stop and review. No automatic relation-access test, extra seeds, longer training or production launch.', '']
    return '\n'.join(lines)


def plot(result, output):
    os.environ['MPLCONFIGDIR'] = str(output/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    times = result['milestones']
    for cell in CELLS:
        rows = [result['points'][cell][str(t)] for t in times]
        series = ([r['iid_test_gain']['mean'] for r in rows],
                  [r['fresh_audit']['auc'] if r['fresh_audit']['valid'] else float('nan') for r in rows],
                  [r['metrics']['mode_tv'] for r in rows],
                  [r['fixed_probe_velocity_mse'] for r in rows])
        for ax, values in zip(axes.flat, series):
            ax.plot(times, values, 'o-', label=cell.replace('_', ' '))
    for ax, label in zip(axes.flat, ('Held-out fixed reward gain', 'Fresh H4 AUC',
                                    'Joint mode TV (grid)', 'Reference velocity probe MSE')):
        ax.set(xlabel='Policy updates', ylabel=label)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fourier x FiLM | same full H4 reward | velocity-MSE coefficient 1')
    fig.tight_layout()
    fig.savefig(output/'factorial_curves.png', dpi=150)
    plt.close(fig)


def run(plan_path, output, wandb_mode='offline', *, analysis_only=False):
    plan = load_json(plan_path)
    output = lab.toy_path(output)
    validate_plan(plan)
    if analysis_only:
        if load_json(output/'plan.json') != plan:
            raise ValueError('Reanalysis must use the saved unmodified plan')
    else:
        readiness = preflight(plan)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'plan.json', plan)
        atomic_json(output/'preflight.json', readiness)
    started = time.monotonic()
    status = {'state': 'running', 'training_started': not analysis_only}
    atomic_json(output/'status.json', status)
    try:
        if not analysis_only:
            result = lab.run(output/'plan.json', output/'raw_shift', wandb_mode)
            if result['state'] != 'completed' or not result['valid']:
                status.update(state='awaiting_review', result_state=result['state'])
                (output/'SUMMARY.md').write_text('Incomplete factorial: new arm has an inadequate audit.\n'
                    'No interaction conclusion; see raw_shift/SUMMARY.md. No automatic retry.\n')
                return result
            from .plot_closed_loop import plot as plot_fits
            plot_fits(output/'raw_shift')
        result = analyze(plan, output)
        status.update(state='awaiting_review', result_state='completed', decisions=result['decisions'])
        return result
    except BaseException as exc:
        status.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wandb-mode', choices=['offline', 'disabled'], default='offline')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--analysis-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)  # Reproduce the saved CPU sampling/fit path.
    plan = load_json(args.plan)
    output = args.output or lab.ROOT/plan['factorial']['output']
    lab.toy_path(output)
    if args.preflight:
        print(json.dumps(preflight(plan), indent=2), flush=True)
    else:
        run(args.plan, output, args.wandb_mode, analysis_only=args.analysis_only)


if __name__ == '__main__':
    main()
