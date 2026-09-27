"""Dense fresh-H4 AUC trajectory; replay the successful learned-reward FiLM toy.

Only measurement cadence changes. No production edits or automatic job launch.
See FILM_AUC_TRAJECTORY_PROTOCOL.md for the fixed question and interpretation.
"""
import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from . import film_conditioning as film
from .coverage_budget import flatten_metrics
from .nonperiodic_cube import Data
from .reward_transport_audit import fit_audits
from .truth_pretrain import atomic_checkpoint, atomic_json

MILESTONES = (0, 100, 250, 500, 750, 1000, 2000, 3000)
RUN_NAME = 'Is fresh AUC monotonic? | learned H4 | nonlinear FiLM | dense early audits'
ARM = 'mlp_c'


def validate_milestones(values):
    values = tuple(values)
    if (len(values) < 2 or values[0] != 0
            or any(type(v) is not int for v in values)
            or tuple(sorted(set(values))) != values):
        raise ValueError('Milestones must start at 0, then contain increasing positive integers')
    return values


class PairedScores:
    """Exact weighted AUC with tie handling; bootstrap repeated context IDs.

    Sorting is done once: repeating an observation changes its multiplicity,
    not the ordering of its score. This matches explicitly resampled ROC AUC.
    """
    def __init__(self, scores):
        p, q = [np.asarray(scores[k], dtype=np.float64) for k in ('positive', 'negative')]
        if p.ndim != 1 or p.shape != q.shape or len(p) < 2 or not np.isfinite([p, q]).all():
            raise ValueError('Need finite, paired 1D positive/negative scores of equal length')
        self.n = len(p)
        joined = np.r_[p, q]
        order = np.argsort(joined, kind='stable')
        self.context = order % self.n
        self.positive = order < self.n
        self.starts = np.r_[0, np.flatnonzero(np.diff(joined[order]) != 0) + 1]
        self.loss = .5 * (np.logaddexp(0., -p) + np.logaddexp(0., q))

    def metrics(self, counts):
        weights = counts[self.context]
        p = np.add.reduceat(weights * self.positive, self.starts)
        q = np.add.reduceat(weights * ~self.positive, self.starts)
        auc = float(np.dot(p, np.cumsum(q) - .5*q) / float(counts.sum())**2)
        return np.array([auc, abs(auc-.5), float(np.dot(self.loss, counts)/counts.sum())])


def trajectory_analysis(points, predictions, *, repeats=2000):
    """Paired evaluation uncertainty, conditional on all trained models.

    The simultaneous band covers adjacent and versus-start contrasts within
    each metric. Absence of a resolved increase is NOT proof of monotonicity.
    """
    steps = [p['policy_step'] for p in points]
    if len(points) < 2 or steps[0] != 0 or steps != sorted(set(steps)):
        raise ValueError('Need ordered distinct policy checkpoints including step 0')
    if not all(p['valid'] for p in points):
        return {'state': 'audit_inconclusive', 'invalid_steps': [p['policy_step'] for p in points if not p['valid']],
                'point_estimate_monotonic': None, 'interpretation': 'Do not bridge invalid audits'}
    if type(repeats) is not int or repeats < 100:
        raise ValueError('Use at least 100 bootstrap repeats')
    scorers = [PairedScores(predictions[s]) for s in steps]
    n = scorers[0].n
    if any(s.n != n for s in scorers):
        raise ValueError('Checkpoint score panels are not paired')
    point = np.stack([s.metrics(np.ones(n)) for s in scorers])
    for row, measured in zip(points, point):
        if not np.allclose([row['auc'], row['auc_gap'], row['bce']], measured, atol=2e-7, rtol=0):
            raise ValueError('Saved test metrics and scores disagree')
    pairs = list(dict.fromkeys([(i, i-1) for i in range(1, len(steps))]
                               + [(i, 0) for i in range(1, len(steps))]))
    delta = np.stack([point[a]-point[b] for a, b in pairs])
    draws = np.empty((repeats, len(pairs), 3))
    rng = np.random.default_rng(491017)
    for j in range(repeats):
        counts = np.bincount(rng.integers(n, size=n), minlength=n)
        metrics = np.stack([s.metrics(counts) for s in scorers])
        draws[j] = [metrics[a]-metrics[b] for a, b in pairs]
    radius = np.quantile(np.max(np.abs(draws-delta), axis=1), .95, axis=0)
    contrasts = []
    names = ('auc', 'auc_gap', 'bce')
    for i, (a, b) in enumerate(pairs):
        entry = {'from_step': steps[b], 'to_step': steps[a], 'adjacent': a == b+1,
                 'versus_start': b == 0}
        for k, name in enumerate(names):
            lo, hi = np.quantile(draws[:, i, k], [.025, .975])
            entry[name] = {'delta': float(delta[i, k]), 'pointwise_lo95': float(lo),
                           'pointwise_hi95': float(hi),
                           'simultaneous_lo95': float(delta[i, k]-radius[k]),
                           'simultaneous_hi95': float(delta[i, k]+radius[k])}
        contrasts.append(entry)
    increases = [c for c in contrasts if c['adjacent'] and c['auc']['simultaneous_lo95'] > 0]
    early = [c for c in contrasts if c['versus_start'] and c['to_step'] < steps[-1]
             and c['auc_gap']['simultaneous_lo95'] > 0]
    final = next(c for c in contrasts if c['versus_start'] and c['to_step'] == steps[-1])
    improves = final['auc_gap']['simultaneous_hi95'] < 0
    return {'state': 'completed', 'steps': steps, 'bootstrap_repeats': repeats,
            'point_estimate_monotonic': bool(np.all(np.diff(point[:, 0]) <= 1e-12)),
            'gap_point_estimate_monotonic': bool(np.all(np.diff(point[:, 1]) <= 1e-12)),
            'resolved_auc_increase_intervals': [[c['from_step'], c['to_step']] for c in increases],
            'resolved_early_worse_than_start': [c['to_step'] for c in early],
            'resolved_final_gap_improvement': bool(improves),
            'resolved_rise_then_recovery': bool(early and improves),
            'contrasts': contrasts,
            'interpretation': ('resolved_nonmonotonic_AUC_on_measured_grid' if increases else
                               'no_resolved_AUC_increase_not_proof_of_monotonicity'),
            'scope': ('Paired context bootstrap; simultaneous adjacent/versus-start contrasts per metric. '
                      'Conditional on fitted models; excludes training-seed uncertainty and between-checkpoint behavior.')}


def write_curve(output, points):
    fields = ('policy_step', 'auc', 'auc_gap', 'bce', 'valid', 'fit_steps', 'selected_step', 'plateau')
    with (output/'auc_trajectory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: p[k] for k in fields} for p in points)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, null in zip(axes, ('auc', 'bce'), (.5, np.log(2))):
        x = [p['policy_step'] for p in points]
        y = [p[key] if p['valid'] else np.nan for p in points]
        ax.plot(x, y, 'o-', label='Fresh held-out audit')
        invalid = [p for p in points if not p['valid']]
        if invalid:
            ax.scatter([p['policy_step'] for p in invalid], [p[key] for p in invalid],
                       marker='x', color='red', label='Audit inconclusive')
        ax.axhline(null, color='gray', ls='--', alpha=.6)
        ax.set(xlabel='DGPO policy updates', ylabel=key.upper())
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Learned H4 + nonlinear FiLM | fresh cold fits; paired evaluation; no smoothing')
    fig.tight_layout()
    fig.savefig(output/'auc_trajectory.png', dpi=160)
    plt.close(fig)


def verify_panels(first, current):
    a = torch.load(first, map_location='cpu', weights_only=True)
    b = torch.load(current, map_location='cpu', weights_only=True)
    for split in ('train', 'validation', 'test'):
        for key in ('c', 'positive'):
            if not torch.equal(a[split][key], b[split][key]):
                raise ValueError('Audit condition/truth identities changed between policy checkpoints')


def prepare(reward_source, control_source, milestones):
    milestones = validate_milestones(milestones)
    cfg, initial, rewards, prior, _ = film.load_inputs(reward_source, control_source, milestones[-1])
    initial_policy = film.make_models(initial, cfg, width=32, arms=film.LEARNED_ARMS)[ARM]
    matching = film.previous.verify_matching(initial, {ARM: initial_policy}, cfg)
    return cfg, initial, initial_policy, rewards, prior, matching


def run(reward_source, control_source, output, *, milestones=MILESTONES,
        audit_max_steps=16000, bootstrap_repeats=2000, wandb_mode='online', run_id=None):
    milestones = validate_milestones(milestones)
    if audit_max_steps < 2000 or bootstrap_repeats < 100:
        raise ValueError('Audit maximum >=2000 and bootstrap repeats >=100 required')
    cfg, initial, initial_policy, rewards, prior, matching = prepare(reward_source, control_source, milestones)
    data = Data(cfg)
    starts = {k: copy.deepcopy(m.state_dict()) for k, m in
              {'source': initial, 'initial_policy': initial_policy, 'reward': rewards['A']}.items()}
    report = {'state': 'prepared_not_started', 'question': 'Is fresh audit AUC monotonic on the measured policy-step grid?',
              'config': asdict(cfg), 'milestones': list(milestones), 'source': prior['source'],
              'reward_source': str(reward_source.resolve()), 'control_source': str(control_source.resolve()),
              'reward_arm': 'A', 'arm': ARM, 'width': 32, 'velocity_coefficient': 1.,
              'reward_target': film.experiment_spec('A')['reward_target'], 'initial_matching': matching,
              'policy_seeds': film.transport.SEEDS, 'encoder_seed': 451017,
              'audit_spec': {'cold_seed': 41, 'minimum_steps': 2000, 'maximum_steps': audit_max_steps,
                             'check_every': 100, 'patience_checks': 20, 'min_delta': 1e-4,
                             'train_events': 32768, 'validation_events': 8192, 'test_events': 16384,
                             'stopping': 'Independent patience at each policy checkpoint, not joint arm stopping'},
              'points': [], 'bootstrap_repeats': bootstrap_repeats,
              'scope': 'One policy seed and one cold-audit seed; no inference about unmeasured steps or real-case convergence.'}
    output.mkdir(parents=True, exist_ok=False)
    wb, started, saved = None, time.monotonic(), None

    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        report['active'] = {k: row[k] for k in ('phase', 'policy_step', 'step') if k in row}
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'report.json', report)
        if wb is not None:
            if row['phase'].startswith('audit'):
                payload = flatten_metrics(row, f"audit/policy{row['policy_step']}/")
            else:
                payload = flatten_metrics(row, row['phase']+'/')
            wb.log(payload)
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        if wandb_mode != 'disabled':
            import wandb
            wb = wandb.init(project='dgpo-toy', mode=wandb_mode, dir=str(output.resolve()),
                            id=run_id, name=RUN_NAME, group='Conditional reward transport',
                            tags=['dense-audit', 'learned-h4-reward', 'nonlinear-film', 'velocity-mse-1', 'raw-no-ema'],
                            config=copy.deepcopy(report))
            report['wandb'] = {'id': wb.id, 'directory': wb.dir, 'mode': wandb_mode, 'name': RUN_NAME}
            for phase in ('policy', 'validation', 'structure', 'conditioning', 'adjacent', 'versus_start'):
                wb.define_metric(phase+'/step')
                wb.define_metric(phase+'/*', step_metric=phase+'/step')
            for step in milestones:
                wb.define_metric(f'audit/policy{step}/step')
                wb.define_metric(f'audit/policy{step}/*', step_metric=f'audit/policy{step}/step')
        atomic_checkpoint(output/'frozen_reward.pt', torch.load(reward_source/'classifier.pt', weights_only=True))
        probe = film.previous.condition_probe(initial, cfg)
        for step in milestones:
            stage = output/f'round_{step}'
            stage.mkdir()
            current = initial_policy
            if step:
                report['state'] = 'training_dgpo'

                def checkpoint(update, model, optimizer, rng, history):
                    nonlocal saved
                    if update == 1 or update % 100 == 0 or update == step:
                        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'rng': rng.get_state(), 'history': history, 'step': update,
                                 'config': asdict(replace(cfg, policy_steps=step)), 'route': ARM, 'width': 32,
                                 'velocity_coefficient': 1., 'reward_arm': 'A', 'source': prior['source']}
                        atomic_checkpoint(output/'mlp_c_last.pt', state)
                        if update == step:
                            atomic_checkpoint(stage/'mlp_c_state.pt', state)
                            saved = copy.deepcopy(state)

                current, history = film.native.policy_train(
                    'dgpo', initial_policy, rewards['A'], data, replace(cfg, policy_steps=step),
                    film.transport.SEEDS['policy'], film.transport.SEEDS['native_monitor'],
                    emit, checkpoint, resume_state=saved, velocity_coefficient=1.)
                if saved['step'] != step or len(history) != step:
                    raise RuntimeError('Lost policy update history at measurement boundary')
            else:
                atomic_checkpoint(stage/'mlp_c_state.pt', {'model': initial_policy.state_dict(), 'step': 0,
                                                         'reward_arm': 'A', 'velocity_coefficient': 1.})
            emit({'phase': 'conditioning', 'step': step, **film.diagnostics(current, initial, probe)})
            structure, _ = film.transport.assess(current, rewards, data, film.transport.SEEDS['endpoint'])
            emit({'phase': 'structure', 'step': step, **structure})
            report['state'] = 'cold_auditing'
            audit = fit_audits({ARM: current}, data, stage/'cold_audit',
                               lambda row: emit({**row, 'policy_step': step}), max_steps=audit_max_steps)
            verify_panels(output/'round_0/cold_audit/mlp_c_panels.pt', stage/'cold_audit/mlp_c_panels.pt')
            stats = audit['test'][ARM]
            point = {'policy_step': step, **{k: stats[k] for k in
                     ('auc', 'auc_gap', 'bce', 'fit_steps', 'selected_step', 'plateau')},
                     'valid': audit['state'] == 'completed' and stats['plateau'] and stats['fit_steps'] >= 2000}
            report['points'].append(point)
            emit({'phase': 'validation', 'step': step, **point})
            write_curve(output, report['points'])
        report['state'] = 'analyzing_monotonicity'
        emit({'phase': 'analysis', 'step': milestones[-1]})
        predictions = {s: torch.load(output/f'round_{s}/cold_audit/test_scores.pt', weights_only=True)[ARM]
                       for s in milestones}
        report['decision'] = trajectory_analysis(report['points'], predictions, repeats=bootstrap_repeats)
        for contrast in report['decision'].get('contrasts', []):
            for phase, include in (('adjacent', contrast['adjacent']),
                                   ('versus_start', contrast['versus_start'])):
                if include:
                    emit({'phase': phase, 'step': contrast['to_step'], **contrast})
        unchanged = {k: film.previous.equal_state(m.state_dict(), starts[k]) for k, m in
                     {'source': initial, 'initial_policy': initial_policy, 'reward': rewards['A']}.items()}
        report['fixed_states_unchanged'] = unchanged
        if not all(unchanged.values()):
            raise RuntimeError('Original source/reference or frozen reward was mutated')
        report['state'] = report['decision']['state']
        atomic_json(output/'monotonicity.json', report['decision'])
        emit({'phase': 'decision', 'step': milestones[-1],
              **{k: v for k, v in report['decision'].items() if not isinstance(v, (dict, list))}})
        return report
    except BaseException as exc:
        report.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'report.json', report)
        if wb is not None:
            wb.summary['state'] = report['state']
            wb.summary.update(flatten_metrics({k: v for k, v in report.get('decision', {}).items()
                                               if not isinstance(v, (list, dict))}, 'decision/'))
            wb.finish(exit_code=int(report['state'] == 'failed_or_interrupted'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reward-source', type=Path, default=film.previous.DEFAULT_SOURCE)
    parser.add_argument('--control-source', type=Path, default=film.CONTROL_SOURCE)
    parser.add_argument('--output', type=Path, default=Path('artifacts/dgpo_toy/film_auc_trajectory_v1'))
    parser.add_argument('--milestones', nargs='+', type=int, default=list(MILESTONES))
    parser.add_argument('--audit-max-steps', type=int, default=16000)
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='online')
    parser.add_argument('--run-id')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        milestones = validate_milestones(args.milestones)
        if args.audit_max_steps < 2000 or args.bootstrap_repeats < 100:
            raise ValueError('Audit maximum >=2000 and bootstrap repeats >=100 required')
    except ValueError as exc:
        parser.error(str(exc))
    if args.preflight:
        cfg, _, _, rewards, prior, matching = prepare(args.reward_source, args.control_source, milestones)
        print(json.dumps({'state': 'preflight_passed_not_started', 'milestones': milestones,
                          'source': prior['source'], 'reward_arm': 'A', 'reward_frozen':
                          all(not p.requires_grad for p in rewards['A'].parameters()),
                          'initial_matching': matching, 'config': asdict(cfg), 'wandb_name': RUN_NAME}, indent=2))
        return
    run(args.reward_source, args.control_source, args.output, milestones=milestones,
        audit_max_steps=args.audit_max_steps, bootstrap_repeats=args.bootstrap_repeats,
        wandb_mode=args.wandb_mode, run_id=args.run_id)


if __name__ == '__main__':
    main()
