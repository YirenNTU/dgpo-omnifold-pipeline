"""A/B/C: learned score, mode-averaged score, and estimated oracle mode ratio.

Only toy code. Same Fourier policy, native DGPO and half velocity-MSE penalty=1.
See REWARD_TRANSPORT_PROTOCOL.md for approximation and decision contracts.
"""
import argparse
import copy
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from . import conditional as native
from .coverage_budget import flatten_metrics
from .cube_lockdown import ConditionDenoiser, verify_initial
from .cube_swap import modes
from .nonperiodic_cube import Critic, Data
from .reward_transport_audit import fit_audits
from .truth_pretrain import atomic_checkpoint, atomic_json

ARMS = ('A', 'B', 'C')
RUN_NAME = 'Can reward move mode probabilities? | Fourier cube | A/B/C reward | V MSE 1'
SEEDS = {'calibration': 301017, 'calibration_validation': 311017,
         'endpoint': 321017, 'structure_monitor': 331017, 'policy': 17,
         'native_monitor': 284017}


def interpolate(nodes, table, contexts):
    """Linear interpolation in the actual continuous c, including both boundaries."""
    if not torch.isfinite(contexts).all() or ((contexts < nodes[0]) | (contexts > nodes[-1])).any():
        raise ValueError('Condition outside calibration domain')
    index = torch.searchsorted(nodes, contexts.contiguous(), right=True).sub(1).clamp(0, len(nodes) - 2)
    fraction = (contexts - nodes[index]) / (nodes[index + 1] - nodes[index])
    return table[index] + fraction[..., None] * (table[index + 1] - table[index])


class ModeTableReward(nn.Module):
    def __init__(self, nodes, values, *, oracle=False):
        super().__init__()
        if nodes.ndim != 1 or len(nodes) < 2 or values.shape != (len(nodes), 8):
            raise ValueError('Need ordered context nodes and eight mode values each')
        if not torch.isfinite(values).all() or not torch.isfinite(nodes).all() or not (nodes.diff() > 0).all():
            raise ValueError('Nonfinite or unordered calibration table')
        if oracle and ((values <= 0).any() or not torch.allclose(values.sum(-1), torch.ones(len(nodes), dtype=values.dtype))):
            raise ValueError('Oracle denominator must be positive normalized actual-generator probabilities')
        self.register_buffer('nodes', nodes.detach().clone())
        self.register_buffer('values', values.detach().clone())
        self.oracle = oracle

    def forward(self, y, c, data=None):
        contexts = c.expand(*y.shape[:-1], 1)[..., 0]
        table = interpolate(self.nodes.to(y), self.values.to(y), contexts)
        if self.oracle:
            if data is None:
                raise ValueError('Oracle numerator requires the declared toy truth distribution')
            # No uniform-reference assumption and no post-fit shift/scale/tempering.
            table = data.probabilities(contexts, .9).to(table).log() - table.log()
        return table.gather(-1, modes(y)[..., None]).squeeze(-1)


@torch.no_grad()
def generate(model, c, k, seed, steps=50):
    z = torch.randn(len(c), k, 3, generator=native.generator(seed))
    y = torch.cat([native.ddim(model, cc[:, None], zz, steps)
                   for cc, zz in zip(c.split(8), z.split(8))])
    if not torch.isfinite(y).all():
        raise FloatingPointError('Nonfinite generated samples')
    return y


def cells(y, reward):
    """Empty endpoint modes are data, not exceptions. Their means are unavailable."""
    ids = modes(y)
    count = torch.zeros(len(y), 8, dtype=torch.long).scatter_add_(1, ids, torch.ones_like(ids))
    total = torch.zeros(len(y), 8, dtype=torch.float64).scatter_add_(1, ids, reward.double())
    square = torch.zeros_like(total).scatter_add_(1, ids, reward.double().square())
    mean = total / count.clamp_min(1)
    variance = (square / count.clamp_min(1) - mean.square()).clamp_min(0)
    return {'counts': count, 'q': count.double() / y.shape[1], 'mean': mean,
            'variance': variance, 'mean_available': count > 0}


def centered_cosine(a, b):
    a, b = a.double(), b.double()
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    norm = a.norm() * b.norm()
    return float((a * b).sum() / norm) if norm > 0 else None


@torch.no_grad()
def calibrate(initial, critic, data, nodes=257, draws=2048, validation_draws=1024):
    if nodes < 3 or draws < 2 or draws % 2 or validation_draws < 1:
        raise ValueError('Need >=3 nodes, even calibration draws and positive validation draws')
    c = torch.linspace(-1, 1, nodes)[:, None]
    y = generate(initial, c, draws, SEEDS['calibration'])
    r = critic(y, c[:, None])
    stats = cells(y, r)
    halves = [cells(yy, rr) for yy, rr in zip(y.chunk(2, dim=1), r.chunk(2, dim=1))]
    # No pseudocount: pretraining is expected to cover every mode. Report a failed
    # calibration instead of filling an unobserved mode with an invented mean.
    min_count = min(int(h['counts'].min()) for h in halves)
    if min_count < 32:
        raise ValueError(f'Calibration insufficient: half-panel minimum mode count={min_count}')
    B = ModeTableReward(c[:, 0], stats['mean']).eval()
    C = ModeTableReward(c[:, 0], stats['q'], oracle=True).eval()
    midpoint = (c[:-1] + c[1:]) / 2
    vy = generate(initial, midpoint, validation_draws, SEEDS['calibration_validation'])
    vr = critic(vy, midpoint[:, None])
    validation = cells(vy, vr)
    if int(validation['counts'].min()) < 32:
        raise ValueError('Independent calibration validation has insufficient mode coverage')
    target = data.probabilities(c[:, 0], .9).double()
    oracle_halves = [target.log() - h['q'].log() for h in halves]
    b_cosine = centered_cosine(halves[0]['mean'], halves[1]['mean'])
    c_cosine = centered_cosine(*oracle_halves)
    predicted_mean = interpolate(B.nodes, B.values, midpoint[:, 0])
    predicted_q = interpolate(C.nodes, C.values, midpoint[:, 0])
    # Includes MC uncertainty of both estimate and independent validation.
    mean_rms = float((predicted_mean - validation['mean']).square().mean().sqrt())
    mean_scale = float((predicted_mean - predicted_mean.mean(-1, keepdim=True)).square().mean().sqrt())
    relative_error = mean_rms / max(mean_scale, 1e-12)
    q_rms = float((predicted_q - validation['q']).square().mean().sqrt())
    cr = C(vy, midpoint[:, None], data)
    weights = cr.double().exp()
    weighted_q = torch.zeros(len(midpoint), 8, dtype=torch.float64).scatter_add_(1, modes(vy), weights)
    weighted_q /= weights.sum(-1, keepdim=True)
    p = data.probabilities(midpoint[:, 0], .9).double()
    tv_before = float((validation['q'] - p).abs().sum(-1).mean() / 2)
    tv_after = float((weighted_q - p).abs().sum(-1).mean() / 2)
    gates = {'split_B_reliable': b_cosine is not None and b_cosine >= .90,
             'split_C_reliable': c_cosine is not None and c_cosine >= .90,
             'B_interpolation_error_small': relative_error <= .25,
             'q_interpolation_error_small': q_rms <= .025,
             'independent_C_reweighting_improves': tv_before - tv_after >= .01}
    report = {'nodes': nodes, 'draws_per_node': draws, 'half_min_count': min_count,
              'validation_midpoints': len(midpoint), 'validation_draws': validation_draws,
              'validation_min_count': int(validation['counts'].min()),
              'B_split_centered_cosine': b_cosine, 'C_split_centered_cosine': c_cosine,
              'B_validation_mean_rms': mean_rms, 'B_validation_relative_rms': relative_error,
              'q_validation_rms': q_rms, 'C_validation_mean_ratio': float(weights.mean()),
              'C_reweighting_mode_tv_before': tv_before, 'C_reweighting_mode_tv_after': tv_after,
              'gates': gates, 'passed': all(gates.values()),
              'scope': 'Estimated node-conditional scores/probabilities; linear interpolation, not exact density'}
    tensors = {'nodes': c[:, 0], 'mean_scores': stats['mean'], 'q_initial': stats['q'],
               'counts': stats['counts'], 'halves': halves,
               'validation_contexts': midpoint, 'validation': validation}
    return {'A': critic, 'B': B, 'C': C}, tensors, report


def sample_metrics(y, c, data):
    signs = torch.where(y >= 0, 1., -1.)
    count = cells(y, torch.zeros(y.shape[:-1]))['counts']
    q = count.double() / y.shape[1]
    p = data.probabilities(c[:, 0], .9).double()
    tv = (q - p).abs().sum(-1) / 2
    parity = signs.prod(-1).mean(-1)
    shape_distance = (y - signs).norm(dim=-1)
    metrics = {'mode_tv': float(tv.mean()), 'mode_rmse': float((q - p).square().mean().sqrt()),
               'parity_mae': float((parity - .8 * data.condition_signal(c[:, 0])).abs().mean()),
               'corner_fraction': float((shape_distance < .5).float().mean()),
               'corner_distance_rms': float(shape_distance.square().mean().sqrt()),
               'low_order_sign_moment_max': float(torch.maximum(signs.mean(1).abs().max(),
                                                   (signs * signs.roll(1, -1)).mean(1).abs().max())),
               'missing_mode_cells': int((count == 0).sum()),
               'sparse_mode_cells': int((count < 16).sum()), 'min_cell_count': int(count.min()),
               'max_abs_coordinate': float(y.abs().max())}
    return metrics, {'q': q, 'p': p, 'counts': count, 'mode_tv_by_context': tv}


@torch.no_grad()
def assess(model, rewards, data, seed, *, grid=128, k=1024):
    c = ((torch.arange(grid) + .5) / grid * 2 - 1)[:, None]
    y = generate(model, c, k, seed)
    metrics, payload = sample_metrics(y, c, data)
    values = {name: reward(y, c[:, None], data).detach() for name, reward in rewards.items()}
    metrics['reward_means'] = {name: float(value.double().mean()) for name, value in values.items()}
    payload.update(contexts=c, rewards=values, learned_cells=cells(y, values['A']))
    return metrics, payload


def endpoint_contrast(after, before):
    """Same generated noise/c grid. Point TV contrasts, not training-seed CIs."""
    return {'mode_tv_delta': float((after['mode_tv_by_context'] - before['mode_tv_by_context']).mean()),
            'reward_gains': {name: native.paired_gain(value, before['rewards'][name])
                             for name, value in after['rewards'].items()},
            'interval_scope': 'Normal intervals from paired context means on fixed grid; not training-seed uncertainty'}


def learned_decomposition(current, baseline):
    mean = baseline['learned_cells']['mean']
    if not baseline['learned_cells']['mean_available'].all():
        return {'available': False, 'reason': 'Missing source mode scores'}
    mode = float(((current['q'] - baseline['q']) * mean).sum(-1).mean())
    total = float(current['rewards']['A'].double().mean() - baseline['rewards']['A'].double().mean())
    return {'available': True, 'total': total, 'mode_probability': mode, 'within_mode_shape': total - mode,
            'scope': 'Frozen reference-shape score decomposition; includes interactions, not KL attribution'}


def decisions(report):
    base = report['baseline']
    success = {name: (base['mode_tv'] - arm['endpoint']['mode_tv'] >= .01
                      and arm['endpoint']['corner_fraction'] >= base['corner_fraction'] - .02
                      and arm['endpoint']['missing_mode_cells'] == 0)
               for name, arm in report['arms'].items()}
    if success.get('B') and not success.get('A'):
        interpretation = 'supports_within_mode_score_variation_as_actionable_contributor'
    elif success.get('C') and not success.get('A') and not success.get('B'):
        interpretation = 'supports_relative_mode_score_fidelity_as_actionable_contributor'
    elif not any(success.values()):
        interpretation = 'reward_error_not_sufficient_at_this_budget_transport_unresolved'
    else:
        interpretation = 'mixed_or_control_improves_inspect_matched_contrasts'
    audit = report.get('audit', {})
    return {'pilot_mode_transport': success, 'interpretation': interpretation,
            'fresh_audit_valid': audit.get('state') == 'completed',
            'fresh_auc_gap_improves': {
                name: (audit.get('state') == 'completed' and
                       audit['contrasts'][f'{name}_minus_baseline']['auc_gap']['hi95'] < 0)
                for name in ARMS} if audit else {},
            'scope': 'Exploratory single-seed finite-budget diagnostic; mode transport is NOT full closure'}


def load_source(source, classifier_source, steps):
    if steps < 1:
        raise ValueError('Positive policy budget required')
    prior = json.loads((classifier_source / 'report.json').read_text())
    if (prior['state'] != 'completed' or prior.get('velocity_coefficient') != 1.
            or not prior['classifier']['gate']['passed']
            or Path(prior['source']).resolve() != source.resolve()):
        raise ValueError('Use completed shape-matched classifier from this exact source')
    saved = torch.load(source / 'reference.pt', map_location='cpu', weights_only=True)
    if saved.get('trained_on') != 'uniform cube reference, not full truth':
        raise ValueError('Wrong pretraining lineage')
    cfg = replace(native.Config(**saved['config']), policy_steps=steps, eval_every=100, eval_events=512)
    for key, value in asdict(cfg).items():
        if key != 'policy_steps' and prior['config'][key] != value:
            raise ValueError(f'Configuration changed versus matched control: {key}')
    if (cfg.dimensions, cfg.context_dim, cfg.ddim_steps) != (3, 1, 50):
        raise ValueError('This diagnostic is the existing continuous nonperiodic DDIM50 cube')
    initial = native.Denoiser(cfg)
    initial.load_state_dict(saved['model'])
    initial.eval()
    ck = torch.load(classifier_source / 'classifier.pt', map_location='cpu', weights_only=True)
    if ck['selected_step'] != prior['classifier']['selected_step']:
        raise ValueError('Frozen classifier selection mismatch')
    critic = Critic(True)
    critic.load_state_dict(ck['model'])
    critic.eval().requires_grad_(False)
    return cfg, initial, critic, ck, prior


def run(source, classifier_source, output, *, steps=1000, wandb_mode='offline', run_id=None,
        audit_max_steps=16000, skip_audit=False):
    if audit_max_steps < 2000:
        raise ValueError('Cold audits require an available budget of at least 2000 steps')
    cfg, initial, critic, ck, prior = load_source(source, classifier_source, steps)
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'calibrating', 'source': str(source.resolve()),
              'classifier_source': str(classifier_source.resolve()), 'classifier_selected_step': ck['selected_step'],
              'config': asdict(cfg), 'seeds': SEEDS, 'basis': 'fourier', 'velocity_coefficient': 1.,
              'reward_definitions': {'A': 'frozen shape-matched learned score',
                                     'B': 'estimated E_q0[score | c, mode], interpolated',
                                     'C': 'log p_truth(mode|c) - log q_hat_initial(mode|c)'},
              'audit_requested': not skip_audit, 'arms': {},
              'scope': 'Diagnostic uses toy mode labels; not a physics-prior-free production intervention'}
    started = time.monotonic()
    wb = None

    def emit(row):
        line = json.dumps(row, allow_nan=False)
        with (output / 'progress.jsonl').open('a') as stream:
            stream.write(line + '\n')
        report['active'] = {k: row[k] for k in ('phase', 'arm', 'step') if k in row}
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'report.json', report)
        if wb is not None:
            wb.log(flatten_metrics(row, row['phase'] + '/' + row.get('arm', 'all') + '/'))
        print(line, flush=True)

    try:
        if wandb_mode != 'disabled':
            import wandb
            wb = wandb.init(project='dgpo-toy', mode=wandb_mode, dir=str(output.resolve()),
                            id=run_id, name=RUN_NAME, group='Conditional reward transport',
                            tags=['A-B-C', 'fourier', 'raw-no-ema', 'velocity-mse-1', 'fixed-reward'],
                            config=copy.deepcopy(report))
            report['wandb'] = {'id': wb.id, 'mode': wandb_mode, 'directory': wb.dir, 'name': RUN_NAME}
            for phase in ('policy', 'structure', 'audit'):
                for name in ('baseline', *ARMS):
                    prefix = f'{phase}/{name}'
                    wb.define_metric(prefix + '/step')
                    wb.define_metric(prefix + '/*', step_metric=prefix + '/step')
        atomic_checkpoint(output / 'classifier.pt', ck)
        initial_state = copy.deepcopy(initial.state_dict())
        critic_state = copy.deepcopy(critic.state_dict())
        data = Data(cfg)
        models = {name: ConditionDenoiser(cfg, 'fourier', initial.state_dict()).eval() for name in ARMS}
        report['initial_matching'] = verify_initial(initial, models, cfg)
        rewards, calibration, report['calibration'] = calibrate(initial, critic, data)
        atomic_checkpoint(output / 'calibration.pt', calibration)
        emit({'phase': 'calibration', 'step': 0, **report['calibration']})
        if not report['calibration']['passed']:
            report['state'] = 'stopped_calibration_gate'
            return report
        reward_states = {name: copy.deepcopy(r.state_dict()) for name, r in rewards.items()}
        report['baseline'], base = assess(initial, rewards, data, SEEDS['endpoint'])
        atomic_checkpoint(output / 'baseline_endpoint.pt', base)
        emit({'phase': 'structure', 'arm': 'baseline', 'step': 0, **report['baseline']})
        endpoints, policies = {}, {'baseline': initial}
        report['state'] = 'training_dgpo'
        for name, model in models.items():
            report['arms'][name] = {'state': 'running', 'structure_history': []}

            def checkpoint(step, current, opt, rng, history):
                if step % 100 == 0 or step == steps:
                    # Persist before evaluation, so even a collapsed endpoint is recoverable.
                    atomic_checkpoint(output / f'{name}_last.pt', {
                        'model': current.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state(),
                        'history': history, 'step': step, 'config': asdict(cfg), 'basis': 'fourier',
                        'reward_arm': name, 'velocity_coefficient': 1., 'source': report['source'],
                        'classifier_source': report['classifier_source']})
                    stats, _ = assess(current, rewards, data, SEEDS['structure_monitor'], grid=64, k=256)
                    report['arms'][name]['structure_history'].append({'step': step, **stats})
                    emit({'phase': 'structure', 'arm': name, 'step': step, **stats})

            final, history = native.policy_train(
                'dgpo', model, rewards[name], data, cfg, SEEDS['policy'], SEEDS['native_monitor'],
                lambda row: emit({**row, 'arm': name}), checkpoint, velocity_coefficient=1.)
            stats, current = assess(final, rewards, data, SEEDS['endpoint'])
            policies[name], endpoints[name] = final, current
            report['arms'][name].update(state='completed', steps=len(history), endpoint=stats,
                                       contrast_to_source=endpoint_contrast(current, base),
                                       learned_reward_decomposition=learned_decomposition(current, base))
            atomic_checkpoint(output / f'{name}_endpoint.pt', current)
            emit({'phase': 'endpoint', 'arm': name, 'step': steps, **stats,
                  'contrast_to_source': report['arms'][name]['contrast_to_source'],
                  'learned_reward_decomposition': report['arms'][name]['learned_reward_decomposition']})
        report['contrasts'] = {f'{a}_minus_{b}': endpoint_contrast(endpoints[a], endpoints[b])
                               for a, b in [('B', 'A'), ('C', 'A'), ('C', 'B')]}
        if prior['config']['policy_steps'] == steps:
            old = torch.load(classifier_source / 'fourier_last.pt', map_location='cpu', weights_only=True)
            report['A_replays_previous_fourier_exactly'] = all(
                torch.equal(v, old['model'][k]) for k, v in policies['A'].state_dict().items())
        report['source_unchanged'] = all(torch.equal(v, initial_state[k]) for k, v in initial.state_dict().items())
        report['rewards_unchanged'] = {name: all(torch.equal(v, reward_states[name][k]) for k, v in r.state_dict().items())
                                       for name, r in rewards.items()}
        report['classifier_unchanged'] = all(torch.equal(v, critic_state[k]) for k, v in critic.state_dict().items())
        if not report['source_unchanged'] or not all(report['rewards_unchanged'].values()) or not report['classifier_unchanged']:
            raise RuntimeError('Frozen source/reward state changed')
        report['decision'] = decisions(report)
        if not skip_audit:
            report['state'] = 'cold_auditing'
            atomic_json(output / 'report.json', report)
            report['audit'] = fit_audits(policies, data, output / 'cold_audit', emit, audit_max_steps)
        report['decision'] = decisions(report)
        report['state'] = ('completed_without_audit' if skip_audit else
                           'completed' if report['audit']['state'] == 'completed' else 'audit_inconclusive')
        emit({'phase': 'decision', 'step': steps, **report['decision']})
        return report
    except BaseException as exc:
        report.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'report.json', report)
        if wb is not None:
            wb.summary['state'] = report['state']
            wb.finish(exit_code=int(report['state'] == 'failed_or_interrupted'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('artifacts/dgpo_toy/nonperiodic_cube_classifier_extended_v1'))
    p.add_argument('--classifier-source', type=Path, default=Path('artifacts/dgpo_toy/shape_matched_dgpo_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--audit-max-steps', type=int, default=16000)
    p.add_argument('--skip-audit', action='store_true', help='Diagnostic only; no classifier-closure conclusion')
    p.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'), default='offline')
    p.add_argument('--run-id')
    p.add_argument('--preflight', action='store_true', help='Read source metadata only; no output or training')
    args = p.parse_args()
    if args.steps < 1 or args.audit_max_steps < 2000:
        p.error('Positive policy steps and >=2000 audit maximum required')
    torch.set_num_threads(2)
    if args.preflight:
        cfg, _, _, ck, _ = load_source(args.source, args.classifier_source, args.steps)
        print(json.dumps({'state': 'preflight_passed_not_started', 'config': asdict(cfg),
                          'classifier_selected_step': ck['selected_step'], 'wandb_name': RUN_NAME}, indent=2))
        return
    report = run(args.source, args.classifier_source, args.output, steps=args.steps,
                 wandb_mode=args.wandb_mode, run_id=args.run_id,
                 audit_max_steps=args.audit_max_steps, skip_audit=args.skip_audit)
    print(json.dumps({'state': report['state'], 'decision': report.get('decision')}))


if __name__ == '__main__':
    main()
