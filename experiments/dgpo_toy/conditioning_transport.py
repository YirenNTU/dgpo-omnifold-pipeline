"""Fast, function-matched conditioning screen with the frozen reward-transport C.

Only the injection route changes. See CONDITIONING_TRANSPORT_PROTOCOL.md.
No production changes, reward fits, recalibration, EMA or automatic job submission.
"""
import argparse
import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from . import conditional as native
from . import reward_transport as transport
from .coverage_budget import flatten_metrics
from .cube_lockdown import ConditionDenoiser, condition_features
from .nonperiodic_cube import Data
from .reward_transport_audit import audit_contrasts, fit_audits
from .truth_pretrain import atomic_checkpoint, atomic_json

ARMS = ('first_add', 'deep_add', 'deep_film')
RUN_NAME = 'Can deeper conditioning move modes? | fixed mode reward | add vs FiLM | V MSE 1'
DEFAULT_SOURCE = Path('artifacts/dgpo_toy/reward_transport_abc_v1_retry1')


class DeepConditionDenoiser(ConditionDenoiser):
    """Linear, zero-output condition heads at all three pre-SiLU hidden layers.

    Keep the pretrained activation/normalization path and the existing first
    adapter. No truth function, mode label, new frequencies or timestep encoder.
    """
    def __init__(self, cfg, route, state):
        if route not in ('deep_add', 'deep_film'):
            raise ValueError(route)
        super().__init__(cfg, 'fourier', state)
        self.route = route
        self.condition_shifts = nn.ModuleList([nn.Linear(8, cfg.hidden, bias=False) for _ in range(2)])
        self.condition_scales = nn.ModuleList(
            [nn.Linear(8, cfg.hidden, bias=False) for _ in range(3)] if route == 'deep_film' else [])
        for head in (*self.condition_shifts, *self.condition_scales):
            nn.init.zeros_(head.weight)

    def _predict(self, x, t, c, diagnostics=False):
        t = t.expand(x.shape[:-1])
        c = c.expand(*x.shape[:-1], c.shape[-1])
        tf = torch.stack([t, (math.pi*t).sin(), (math.pi*t).cos(),
                          (2*math.pi*t).sin(), (2*math.pi*t).cos()], -1)
        features = condition_features(c, 'fourier')
        hidden = torch.cat([x, c, tf], -1)
        stats = {}
        shifts = (self.condition_adapter, *self.condition_shifts)
        for layer in range(3):
            original = self.network[2*layer](hidden)
            beta = shifts[layer](features)
            gamma = self.condition_scales[layer](features) if self.condition_scales else None
            modified = original + beta if gamma is None else original * (1 + gamma) + beta
            hidden = self.network[2*layer+1](modified)
            if diagnostics:
                rms = original.square().mean().sqrt().clamp_min(1e-12)
                stats[f'layer{layer}'] = {
                    'residual_over_hidden_rms': float((modified-original).square().mean().sqrt()/rms),
                    'shift_rms': float(beta.square().mean().sqrt()),
                    'scale_rms': float(gamma.square().mean().sqrt()) if gamma is not None else 0.}
        v = native.alpha_sigma(t)[1][..., None] * self.network[-1](hidden)
        return v, hidden, stats

    def predict_features(self, x, t, c):
        v, hidden, _ = self._predict(x, t, c)
        return v, hidden


def make_models(initial, cfg):
    # Architecture construction must not consume an experiment's random stream.
    with torch.random.fork_rng():
        torch.manual_seed(401017)
        return {'first_add': ConditionDenoiser(cfg, 'fourier', initial.state_dict()).eval(),
                **{name: DeepConditionDenoiser(cfg, name, initial.state_dict()).eval()
                   for name in ARMS[1:]}}


@torch.no_grad()
def verify_matching(initial, models, cfg):
    rng = native.generator(800017)
    c = 2*torch.rand(256, 1, generator=rng)-1
    x = torch.randn(256, 8, 3, generator=rng)
    t = torch.rand(256, 8, generator=rng)
    expected_v = initial(x, t, c[:, None])
    expected_y = native.ddim(initial, c[:, None], x, cfg.ddim_steps)
    base_count = sum(p.numel() for p in initial.parameters())
    result = {}
    for name, model in models.items():
        count = sum(p.numel() for p in model.parameters())
        result[name] = {'velocity_exact': torch.equal(model(x, t, c[:, None]), expected_v),
                        'samples_exact': torch.equal(native.ddim(model, c[:, None], x, cfg.ddim_steps), expected_y),
                        'parameters': count, 'added_parameters': count-base_count}
        if not result[name]['velocity_exact'] or not result[name]['samples_exact']:
            raise ValueError(f'Initial function matching failed: {name}')
    return result


def load_previous(directory, steps):
    prior = json.loads((directory / 'report.json').read_text())
    if (prior['state'] != 'completed' or not prior['calibration']['passed']
            or prior['velocity_coefficient'] != 1. or prior['basis'] != 'fourier'
            or prior['seeds'] != transport.SEEDS):
        raise ValueError('Requires the completed, calibrated Fourier A/B/C reward experiment')
    cfg, initial, critic, ck, _ = transport.load_source(
        Path(prior['source']), Path(prior['classifier_source']), steps)
    for key, value in asdict(cfg).items():
        if key != 'policy_steps' and prior['config'][key] != value:
            raise ValueError(f'Changed policy setting: {key}')
    cached_ck = torch.load(directory / 'classifier.pt', map_location='cpu', weights_only=True)
    if (cached_ck['selected_step'] != ck['selected_step']
            or not equal_state(cached_ck['model'], critic.state_dict())):
        raise ValueError('Cached frozen classifier does not match its source')
    table = torch.load(directory / 'calibration.pt', map_location='cpu', weights_only=True)
    rewards = {'A': critic,
               'B': transport.ModeTableReward(table['nodes'], table['mean_scores']).eval(),
               'C': transport.ModeTableReward(table['nodes'], table['q_initial'], oracle=True).eval()}
    return cfg, initial, rewards, prior


def equal_state(a, b):
    return a.keys() == b.keys() and all(torch.equal(v, b[k]) for k, v in a.items())


@torch.no_grad()
def condition_probe(initial, cfg):
    """Fixed reference-generated noisy states; own RNG, no truth in model input."""
    rng = native.generator(411017)
    c = 2*torch.rand(64, 1, generator=rng)-1
    z = torch.randn(64, 8, 3, generator=rng)
    y = native.ddim(initial, c[:, None], z, cfg.ddim_steps)
    t = .7*torch.rand(64, 8, generator=rng)
    eps = torch.randn(y.shape, generator=rng)
    a, s = native.alpha_sigma(t)
    return a[..., None]*y+s[..., None]*eps, t, c[:, None]


@torch.no_grad()
def route_diagnostics(model, initial, probe):
    x, t, c = probe
    if isinstance(model, DeepConditionDenoiser):
        v, _, layers = model._predict(x, t, c, diagnostics=True)
    else:
        cc = c.expand(*x.shape[:-1], 1)
        tf = torch.stack([t, (math.pi*t).sin(), (math.pi*t).cos(),
                          (2*math.pi*t).sin(), (2*math.pi*t).cos()], -1)
        h = model.network[0](torch.cat([x, cc, tf], -1))
        beta = model.condition_adapter(condition_features(cc, 'fourier'))
        layers = {'layer0': {'residual_over_hidden_rms': float(
            beta.square().mean().sqrt()/h.square().mean().sqrt().clamp_min(1e-12)),
            'shift_rms': float(beta.square().mean().sqrt()), 'scale_rms': 0.}}
        v = model(x, t, c)
    blocks = {'backbone': [], 'condition': []}
    for name, p in model.named_parameters():
        blocks['backbone' if name.startswith('network.') else 'condition'].append(p)
    grads = {name: math.sqrt(sum(float(p.grad.square().sum()) for p in params if p.grad is not None))
             for name, params in blocks.items()}
    return {'layers': layers, 'last_step_postclip_gradient_norm': grads,
            'fixed_probe_velocity_mse': float((v-initial(x, t, c)).square().mean())}


def tv_contrast(after, before):
    d = (after['mode_tv_by_context']-before['mode_tv_by_context']).double()
    delta, se = float(d.mean()), float(d.std(unbiased=True)/math.sqrt(len(d)))
    return {'delta': delta, 'lo95': delta-1.96*se, 'hi95': delta+1.96*se,
            'scope': 'Paired fixed-context endpoint variation; NOT training-seed uncertainty'}


def decisions(report):
    baseline, arms = report['baseline'], report['arms']
    transport_pass = {name: (baseline['mode_tv']-arm['endpoint']['mode_tv'] >= .01
                             and arm['endpoint']['corner_fraction'] >= baseline['corner_fraction']-.02
                             and arm['endpoint']['missing_mode_cells'] == 0)
                      for name, arm in arms.items()}
    beats_control = {name: (transport_pass[name]
                            and report['contrasts'][f'{name}_minus_first_add']['delta'] <= -.005
                            and report['contrasts'][f'{name}_minus_first_add']['hi95'] < 0)
                     for name in ARMS[1:]}
    adequate = report['config']['policy_steps'] >= 1000
    interpretation = ('short_budget_inconclusive' if not adequate else
                      'supports_conditioning_route_as_actionable_contributor' if any(beats_control.values()) else
                      'tested_routes_not_sufficient_at_this_budget')
    audit = report.get('audit', {})
    valid = audit.get('state') == 'completed'
    return {'interpretation': interpretation, 'pilot_mode_transport': transport_pass,
            'route_beats_first_add': beats_control, 'budget_adequate_for_screen': adequate,
            'fresh_audit_valid': valid,
            'fresh_auc_gap_improves': {name: valid and audit['contrasts'][f'{name}_minus_baseline']['auc_gap']['hi95'] < 0
                                       for name in ARMS},
            'scope': 'Single-seed architecture screen, unequal added parameters. Not full closure or a unique-cause proof.'}


def run(directory, output, *, steps=1000, wandb_mode='offline', run_id=None,
        audit_max_steps=16000, skip_audit=False):
    if audit_max_steps < 2000:
        raise ValueError('Cold audit budget must allow at least 2000 updates')
    cfg, initial, rewards, prior = load_previous(directory, steps)
    models = make_models(initial, cfg)
    matching = verify_matching(initial, models, cfg)
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'preparing', 'reward_source': str(directory.resolve()),
              'source': prior['source'], 'config': asdict(cfg), 'seeds': transport.SEEDS,
              'policy_reward': 'C: frozen estimated log p_truth(mode|c)/q_initial(mode|c)',
              'condition_features': 'unchanged sqrt(2) sin/cos(pi*c*[1,2,4,8])',
              'velocity_coefficient': 1., 'initial_matching': matching,
              'arms': {}, 'audit_requested': not skip_audit,
              'scope': 'Route/capacity package comparison, not parameter-matched. Truth only enters toy reward and evaluation.'}
    started, wb = time.monotonic(), None

    def emit(row):
        line = json.dumps(row, allow_nan=False)
        with (output / 'progress.jsonl').open('a') as stream:
            stream.write(line+'\n')
        report['active'] = {k: row[k] for k in ('phase', 'arm', 'step') if k in row}
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output / 'report.json', report)
        if wb is not None:
            wb.log(flatten_metrics(row, row['phase']+'/'+row.get('arm', 'all')+'/'))
        print(line, flush=True)

    try:
        if wandb_mode != 'disabled':
            import wandb
            wb = wandb.init(project='dgpo-toy', mode=wandb_mode, dir=str(output.resolve()),
                            name=RUN_NAME, id=run_id, group='Conditional reward transport',
                            tags=['conditioning-route', 'fixed-mode-reward', 'raw-no-ema', 'velocity-mse-1'],
                            config=copy.deepcopy(report))
            report['wandb'] = {'id': wb.id, 'mode': wandb_mode, 'directory': wb.dir, 'name': RUN_NAME}
            for phase in ('policy', 'structure', 'conditioning', 'audit'):
                for name in ('baseline', *ARMS):
                    prefix = f'{phase}/{name}'
                    wb.define_metric(prefix+'/step')
                    wb.define_metric(prefix+'/*', step_metric=prefix+'/step')
        source_state = copy.deepcopy(initial.state_dict())
        reward_states = {name: copy.deepcopy(r.state_dict()) for name, r in rewards.items()}
        data, probe = Data(cfg), condition_probe(initial, cfg)
        report['baseline'], baseline = transport.assess(initial, rewards, data, transport.SEEDS['endpoint'])
        cached = torch.load(directory / 'baseline_endpoint.pt', map_location='cpu', weights_only=True)
        report['cached_baseline_exact'] = (torch.equal(baseline['q'], cached['q'])
            and all(torch.equal(v, cached['rewards'][k]) for k, v in baseline['rewards'].items()))
        if not report['cached_baseline_exact']:
            raise ValueError('Cached calibration/source baseline is not replayable; stop before updates')
        atomic_checkpoint(output / 'baseline_endpoint.pt', baseline)
        emit({'phase': 'structure', 'arm': 'baseline', 'step': 0, **report['baseline']})
        endpoints, policies = {}, {'baseline': initial}
        report['state'] = 'training_dgpo'
        for name, model in models.items():
            arm_start = time.monotonic()
            report['arms'][name] = {'state': 'running', 'structure_history': []}
            emit({'phase': 'conditioning', 'arm': name, 'step': 0,
                  **route_diagnostics(model, initial, probe)})

            def checkpoint(step, current, opt, rng, history):
                if step == 1 or step % 100 == 0 or step == steps:
                    atomic_checkpoint(output / f'{name}_last.pt', {
                        'model': current.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state(),
                        'history': history, 'step': step, 'config': asdict(cfg), 'route': name,
                        'velocity_coefficient': 1., 'reward_source': report['reward_source'],
                        'source': report['source'], 'reward_arm': 'C'})
                    emit({'phase': 'conditioning', 'arm': name, 'step': step,
                          **route_diagnostics(current, initial, probe)})
                if step % 100 == 0 or step == steps:
                    stats, _ = transport.assess(current, rewards, data, transport.SEEDS['structure_monitor'], grid=64, k=256)
                    report['arms'][name]['structure_history'].append({'step': step, **stats})
                    emit({'phase': 'structure', 'arm': name, 'step': step, **stats})

            final, history = native.policy_train(
                'dgpo', model, rewards['C'], data, cfg, transport.SEEDS['policy'], transport.SEEDS['native_monitor'],
                lambda row: emit({**row, 'arm': name}), checkpoint, velocity_coefficient=1.)
            stats, endpoint = transport.assess(final, rewards, data, transport.SEEDS['endpoint'])
            endpoints[name], policies[name] = endpoint, final
            report['arms'][name].update(state='completed', steps=len(history), endpoint=stats,
                wall_seconds=time.monotonic()-arm_start,
                contrast_to_source=transport.endpoint_contrast(endpoint, baseline))
            atomic_checkpoint(output / f'{name}_endpoint.pt', endpoint)
            if name == 'first_add' and steps == prior['config']['policy_steps']:
                old = torch.load(directory / 'C_last.pt', map_location='cpu', weights_only=True)
                report['first_add_replays_previous_C_exactly'] = equal_state(final.state_dict(), old['model'])
                if not report['first_add_replays_previous_C_exactly']:
                    raise ValueError('Matched control did not reproduce previous C; do not interpret route contrasts')
            emit({'phase': 'endpoint', 'arm': name, 'step': steps, **stats})
        report['contrasts'] = {f'{a}_minus_{b}': tv_contrast(endpoints[a], endpoints[b])
                               for a, b in [('deep_add', 'first_add'), ('deep_film', 'first_add'), ('deep_film', 'deep_add')]}
        report['source_unchanged'] = equal_state(initial.state_dict(), source_state)
        report['rewards_unchanged'] = {name: equal_state(r.state_dict(), reward_states[name]) for name, r in rewards.items()}
        if not report['source_unchanged'] or not all(report['rewards_unchanged'].values()):
            raise RuntimeError('Frozen source/reward changed')
        report['decision'] = decisions(report)
        if not skip_audit:
            report['state'] = 'cold_auditing'
            atomic_json(output / 'report.json', report)
            report['audit'] = fit_audits(policies, data, output / 'cold_audit', emit, audit_max_steps)
            predictions = torch.load(output / 'cold_audit/test_scores.pt', map_location='cpu', weights_only=True)
            control_comparisons = audit_contrasts({'baseline': predictions['first_add'],
                                                  **{k: predictions[k] for k in ARMS[1:]}})
            report['audit']['route_contrasts'] = {k.replace('_minus_baseline', '_minus_first_add'): v
                                                  for k, v in control_comparisons.items()}
        report['decision'] = decisions(report)
        report['state'] = ('completed_without_audit' if skip_audit else
                           'completed' if report['audit']['state'] == 'completed' else 'audit_inconclusive')
        emit({'phase': 'decision', 'step': steps, **report['decision']})
        return report
    except BaseException as exc:
        report.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output / 'report.json', report)
        if wb is not None:
            wb.summary['state'] = report['state']
            wb.finish(exit_code=int(report['state'] == 'failed_or_interrupted'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reward-source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--audit-max-steps', type=int, default=16000)
    p.add_argument('--skip-audit', action='store_true', help='Screen only, never classifier-closure evidence')
    p.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'), default='offline')
    p.add_argument('--run-id')
    p.add_argument('--preflight', action='store_true', help='Read and function-check only; no output or updates')
    args = p.parse_args()
    if args.steps < 1 or args.audit_max_steps < 2000:
        p.error('Positive policy budget and audit maximum >=2000 required')
    torch.set_num_threads(2)
    if args.preflight:
        cfg, initial, _, _ = load_previous(args.reward_source, args.steps)
        print(json.dumps({'state': 'preflight_passed_not_started', 'config': asdict(cfg),
                          'initial_matching': verify_matching(initial, make_models(initial, cfg), cfg),
                          'wandb_name': RUN_NAME}, indent=2))
        return
    result = run(args.reward_source, args.output, steps=args.steps, audit_max_steps=args.audit_max_steps,
                 skip_audit=args.skip_audit, wandb_mode=args.wandb_mode, run_id=args.run_id)
    print(json.dumps({'state': result['state'], 'decision': result.get('decision')}))


if __name__ == '__main__':
    main()
