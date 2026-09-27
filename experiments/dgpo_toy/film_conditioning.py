"""FiLM milestone comparison with frozen oracle C or learned H4 A reward.

Native DGPO and its frozen reference survive every stage boundary unchanged.
See FILM_CONDITIONING_PROTOCOL.md and LEARNED_FILM_CONDITIONING_PROTOCOL.md.
Toy-only; no production or remote actions.
"""
import argparse
import copy
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

from . import conditional as native
from . import conditioning_transport as previous
from . import reward_transport as transport
from .coverage_budget import flatten_metrics
from .cube_lockdown import condition_features
from .nonperiodic_cube import Data
from .reward_transport_audit import audit_contrasts, fit_audits
from .truth_pretrain import atomic_checkpoint, atomic_json

ARMS = ('linear', 'mlp_c', 'mlp_ct')
LEARNED_ARMS = ('linear', 'mlp_c')
RUN_NAME = 'Can nonlinear FiLM improve transport? | condition vs time | fixed reward | V MSE 1'
LEARNED_RUN_NAME = 'Does FiLM transfer learned reward? | frozen H4 | linear vs nonlinear | V MSE 1'
CONTROL_SOURCE = Path('artifacts/dgpo_toy/conditioning_transport_v1')


def experiment_spec(reward_arm):
    if reward_arm not in ('A', 'C'):
        raise ValueError('Choose A (frozen learned H4) or C (frozen oracle mode reward)')
    learned = reward_arm == 'A'
    return {
        'arms': LEARNED_ARMS if learned else ARMS,
        'run_name': LEARNED_RUN_NAME if learned else RUN_NAME,
        'policy_reward': ('frozen A shape-matched learned H4 logit' if learned else
                          'frozen C estimated oracle mode log ratio'),
        'reward_target': ('Shape-matched diagnostic target, not original full truth' if learned else
                          'Analytic truth mode numerator / calibrated initial generator denominator'),
        'tags': (['nonlinear-film', 'learned-h4-reward', 'shape-matched-reward'] if learned else
                 ['nonlinear-film', 'time-conditioning', 'fixed-mode-reward']) +
                ['velocity-mse-1', 'raw-no-ema'],
    }


def time_features(t):
    return torch.stack([t, (math.pi*t).sin(), (math.pi*t).cos(),
                         (2*math.pi*t).sin(), (2*math.pi*t).cos()], -1)


class NonlinearFiLM(native.Denoiser):
    def __init__(self, cfg, state, *, use_time=False, width=32):
        if width < 2:
            raise ValueError('Condition encoder width must be >=2')
        super().__init__(cfg)
        self.load_state_dict(state)
        self.use_time, self.width = use_time, width
        self.condition_input = nn.Linear(8, width)
        self.condition_hidden = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        # Only NEW encoder features are normalized, never pretrained activations.
        self.condition_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition_heads = nn.ModuleList([nn.Linear(width, 2*cfg.hidden) for _ in range(3)])
        for head in self.condition_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Construct last so c-only and c,t have exactly the same common weights.
        self.condition_time = nn.Linear(5, width, bias=False) if use_time else None

    def modulation(self, c, t):
        t = t.expand(c.shape[:-1])
        z = self.condition_input(condition_features(c, 'fourier'))
        if self.condition_time is not None:
            z = z + self.condition_time(time_features(t))
        z = self.condition_norm(self.condition_hidden(z))
        return [head(z).chunk(2, -1) for head in self.condition_heads], z

    def _predict(self, x, t, c, diagnostics=False):
        t = t.expand(x.shape[:-1])
        c = c.expand(*x.shape[:-1], c.shape[-1])
        modulation, z = self.modulation(c, t)
        hidden = torch.cat([x, c, time_features(t)], -1)
        stats = {}
        for layer, (gamma, beta) in enumerate(modulation):
            original = self.network[2*layer](hidden)
            modified = original * (1+gamma) + beta
            hidden = self.network[2*layer+1](modified)
            if diagnostics:
                rms = original.square().mean().sqrt().clamp_min(1e-12)
                stats[f'layer{layer}'] = {
                    'residual_over_hidden_rms': float((modified-original).square().mean().sqrt()/rms),
                    'shift_rms': float(beta.square().mean().sqrt()),
                    'scale_rms': float(gamma.square().mean().sqrt())}
        if diagnostics:
            stats['encoder'] = {'feature_rms': float(z.square().mean().sqrt())}
        return native.alpha_sigma(t)[1][..., None]*self.network[-1](hidden), hidden, stats

    def predict_features(self, x, t, c):
        v, hidden, _ = self._predict(x, t, c)
        return v, hidden


def make_models(initial, cfg, width=32, *, arms=ARMS):
    if tuple(arms) not in (ARMS, LEARNED_ARMS):
        raise ValueError('Use the prespecified linear/c-only pair or linear/c-only/c,t trio')
    models = {'linear': previous.make_models(initial, cfg)['deep_film']}
    for name in arms[1:]:
        with torch.random.fork_rng():
            torch.manual_seed(451017)
            models[name] = NonlinearFiLM(cfg, initial.state_dict(), use_time=name == 'mlp_ct', width=width).eval()
    return models


@torch.no_grad()
def diagnostics(model, initial, probe):
    x, t, c = probe
    v, _, layers = model._predict(x, t, c, diagnostics=True)
    groups = {'backbone': [], 'encoder': [], 'time_encoder': [], 'modulation': []}
    for name, p in model.named_parameters():
        group = ('backbone' if name.startswith('network.') else
                 'encoder' if name.startswith(('condition_input.', 'condition_hidden.')) else
                 'time_encoder' if name.startswith('condition_time.') else 'modulation')
        groups[group].append(p)
    grads = {g: math.sqrt(sum(float(p.grad.square().sum()) for p in ps if p.grad is not None))
             for g, ps in groups.items()}
    result = {'layers': layers, 'last_step_postclip_gradient_norm': grads,
              'fixed_probe_velocity_mse': float((v-initial(x, t, c)).square().mean())}
    if isinstance(model, NonlinearFiLM):
        cc = c.expand(*x.shape[:-1], 1)
        m1, _ = model.modulation(cc, t)
        m2, _ = model.modulation(cc, .7-t)
        result['time_modulation_difference_rms'] = math.sqrt(
            sum(float((a-b).square().mean()) for l1, l2 in zip(m1, m2) for a, b in zip(l1, l2))/6)
        # DGPO samples t<=.7 while DDIM starts at t=1: expose extrapolation,
        # without silently changing the inherited training-time distribution.
        result['coefficient_rms_by_time'] = {}
        contexts = c.reshape(-1, 1)
        for key, value in (('t0', 0.), ('t035', .35), ('t070', .7), ('t100', 1.)):
            coeffs, _ = model.modulation(contexts, contexts.new_full(contexts.shape[:-1], value))
            result['coefficient_rms_by_time'][key] = math.sqrt(
                sum(float(v.square().mean()) for pair in coeffs for v in pair)/6)
    return result


def validate_milestones(milestones):
    values = tuple(milestones)
    if not values or any(not isinstance(v, int) or v < 1 for v in values) or tuple(sorted(set(values))) != values:
        raise ValueError('Milestones must be distinct increasing positive integers')
    return values


def load_inputs(directory, control_directory, final_step):
    cfg, initial, rewards, prior = previous.load_previous(directory, final_step)
    control = json.loads((control_directory/'report.json').read_text())
    if (control['state'] != 'completed' or control['source'] != prior['source']
            or Path(control['reward_source']).resolve() != directory.resolve()
            or control['velocity_coefficient'] != 1.):
        raise ValueError('Linear control must use the completed same-source conditioning experiment')
    for key, value in asdict(cfg).items():
        if key != 'policy_steps' and value != control['config'][key]:
            raise ValueError(f'Control setting mismatch: {key}')
    return cfg, initial, rewards, prior, control


def decision(baseline, result, *, reward_arm='C'):
    arms = experiment_spec(reward_arm)['arms']
    audit = result.get('audit', {})
    valid = audit.get('state') == 'completed'
    source_improves = {name: valid and audit['contrasts'][f'{name}_minus_baseline']['auc_gap']['hi95'] < 0
                       for name in arms}
    control_improves = {name: valid and audit['route_contrasts'][f'{name}_minus_linear']['auc_gap']['hi95'] < 0
                        for name in arms[1:]}
    modes = {name: (baseline['mode_tv']-arm['endpoint']['mode_tv'] >= .01
                    and arm['endpoint']['corner_fraction'] >= baseline['corner_fraction']-.02
                    and arm['endpoint']['missing_mode_cells'] == 0)
             for name, arm in result['arms'].items()}
    fresh = {name: source_improves[name] and control_improves[name] for name in arms[1:]}
    full = {name: fresh[name] and modes[name] for name in arms[1:]}
    result = {'fresh_audit_valid': valid, 'fresh_gap_improves_vs_source': source_improves,
            'fresh_gap_improves_vs_linear': control_improves, 'mode_transport_pass': modes,
            'combined_pass': full,
            'scope': 'Exploratory single-seed package comparison; intervals are not training-seed replication'}
    if reward_arm == 'A':
        # Full-truth fresh discrimination is primary; mode diagnostics do not veto it.
        result['primary_pass'] = fresh
        result['interpretation'] = (
            'audit_inconclusive' if not valid else
            'supports_nonlinear_learned_H4_transfer_at_this_budget' if fresh['mlp_c'] else
            'learned_H4_transfers_but_nonlinear_advantage_unresolved' if source_improves['mlp_c'] else
            'nonlinear_package_not_sufficient_at_this_budget')
    else:
        result['time_branch_improves_vs_c_only'] = (
            valid and audit['time_contrast']['mlp_ct_minus_mlp_c']['auc_gap']['hi95'] < 0)
    return result


def run(directory, control_directory, output, *, milestones=(1000, 2000, 3000), width=32,
        wandb_mode='offline', run_id=None, audit_max_steps=16000, reward_arm='C'):
    spec = experiment_spec(reward_arm)
    arms = spec['arms']
    milestones = validate_milestones(milestones)
    if audit_max_steps < 2000:
        raise ValueError('Need a cold-audit maximum of at least 2000')
    cfg, initial, rewards, prior, old_control = load_inputs(directory, control_directory, milestones[-1])
    models = make_models(initial, cfg, width, arms=arms)
    matching = previous.verify_matching(initial, models, cfg)
    frozen_classifier = torch.load(directory/'classifier.pt', map_location='cpu', weights_only=True)
    output.mkdir(parents=True, exist_ok=False)
    report = {'state': 'preparing', 'source': prior['source'], 'reward_source': str(directory.resolve()),
              'control_source': str(control_directory.resolve()), 'config': asdict(cfg),
              'milestones': list(milestones), 'primary_endpoint_step': milestones[-1],
              'width': width, 'encoder_initialization_seed': 451017, 'policy_seeds': transport.SEEDS,
              'velocity_coefficient': 1., 'policy_reward': spec['policy_reward'],
              'reward_arm': reward_arm, 'arms': list(arms), 'reward_target': spec['reward_target'],
              'classifier_checkpoint': str((directory/'classifier.pt').resolve()),
              'classifier_source': prior['classifier_source'],
              'classifier_selected_step': frozen_classifier['selected_step'],
              'primary_endpoint_rule': ('fresh audit gap improves vs source and linear; topology secondary'
                  if reward_arm == 'A' else 'fresh audit gap improves vs source and linear plus mode transport screen'),
              'initial_matching': matching, 'rounds': {},
              'scope': 'Same initial function; unequal parameter counts. Stage boundaries preserve optimizer, RNG and original reference.'}
    wb, started = None, time.monotonic()

    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        report['active'] = {k: row[k] for k in ('phase', 'milestone', 'arm', 'step') if k in row}
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'report.json', report)
        if wb is not None:
            audit_phase = row['phase'].startswith('audit')
            prefix = (f"{row['phase']}/round{row.get('milestone', 0)}/{row.get('arm', 'all')}/"
                      if audit_phase else f"{row['phase']}/{row.get('arm', 'all')}/")
            wb.log(flatten_metrics(row, prefix))
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        if wandb_mode != 'disabled':
            import wandb
            wb = wandb.init(project='dgpo-toy', mode=wandb_mode, dir=str(output.resolve()),
                            id=run_id, name=spec['run_name'], group='Conditional reward transport',
                            tags=spec['tags'],
                            config=copy.deepcopy(report))
            report['wandb'] = {'id': wb.id, 'directory': wb.dir, 'mode': wandb_mode, 'name': spec['run_name']}
            for phase in ('policy', 'structure', 'conditioning', 'endpoint', 'validation'):
                for name in ('baseline', *arms):
                    prefix = f'{phase}/{name}'
                    wb.define_metric(prefix+'/step')
                    wb.define_metric(prefix+'/*', step_metric=prefix+'/step')
            for milestone in milestones:
                for name in ('baseline', *arms):
                    prefix = f'audit/round{milestone}/{name}'
                    wb.define_metric(prefix+'/step')
                    wb.define_metric(prefix+'/*', step_metric=prefix+'/step')
        source_state = copy.deepcopy(initial.state_dict())
        model_starts = {k: copy.deepcopy(v.state_dict()) for k, v in models.items()}
        reward_states = {k: copy.deepcopy(v.state_dict()) for k, v in rewards.items()}
        if reward_arm == 'A':
            atomic_checkpoint(output/'frozen_reward.pt', frozen_classifier)
        data, probe = Data(cfg), previous.condition_probe(initial, cfg)
        report['baseline'], baseline = transport.assess(initial, rewards, data, transport.SEEDS['endpoint'])
        cached = torch.load(directory/'baseline_endpoint.pt', map_location='cpu', weights_only=True)
        report['cached_baseline_exact'] = (torch.equal(cached['q'], baseline['q']) and
            all(torch.equal(v, cached['rewards'][k]) for k, v in baseline['rewards'].items()))
        if not report['cached_baseline_exact']:
            raise ValueError('Cached baseline mismatch; no updates allowed')
        atomic_checkpoint(output/'baseline_endpoint.pt', baseline)
        emit({'phase': 'structure', 'arm': 'baseline', 'step': 0, **report['baseline']})
        resume_states = dict.fromkeys(arms, None)
        for milestone in milestones:
            stage = output/f'round_{milestone}'
            stage.mkdir()
            result = {'state': 'training', 'arms': {}}
            report['rounds'][str(milestone)] = result
            stage_cfg = replace(cfg, policy_steps=milestone)
            endpoints, policies = {}, {'baseline': initial}
            for name, model in models.items():
                report['state'] = 'training_dgpo'
                result['arms'][name] = {'state': 'running', 'structure_history': []}
                start_step = resume_states[name]['step'] if resume_states[name] is not None else 0
                arm_started = time.monotonic()

                def checkpoint(step, current, opt, rng, history):
                    if step == 1 or step % 100 == 0 or step == milestone:
                        state = {'model': current.state_dict(), 'optimizer': opt.state_dict(),
                                 'rng': rng.get_state(), 'history': history, 'step': step,
                                 'config': asdict(stage_cfg), 'route': name, 'width': width,
                                 'velocity_coefficient': 1., 'source': report['source'],
                                 'reward_source': report['reward_source'], 'reward_arm': reward_arm}
                        atomic_checkpoint(output/f'{name}_last.pt', state)
                        emit({'phase': 'conditioning', 'arm': name, 'milestone': milestone, 'step': step,
                              **diagnostics(current, initial, probe)})
                        if step == milestone:
                            atomic_checkpoint(stage/f'{name}_state.pt', state)
                            resume_states[name] = copy.deepcopy(state)
                    if step % 100 == 0 or step == milestone:
                        stats, _ = transport.assess(current, rewards, data, transport.SEEDS['structure_monitor'], grid=64, k=256)
                        result['arms'][name]['structure_history'].append({'step': step, **stats})
                        emit({'phase': 'structure', 'arm': name, 'milestone': milestone, 'step': step, **stats})

                if resume_states[name] is not None and resume_states[name]['reward_arm'] != reward_arm:
                    raise ValueError('Milestone continuation must preserve its frozen reward')
                final, history = native.policy_train(
                    'dgpo', model, rewards[reward_arm], data, stage_cfg, transport.SEEDS['policy'], transport.SEEDS['native_monitor'],
                    lambda row: emit({**row, 'arm': name, 'milestone': milestone}), checkpoint,
                    resume_state=resume_states[name], velocity_coefficient=1.)
                if len(history) != milestone or resume_states[name]['step'] != milestone:
                    raise RuntimeError('Stage lost its optimizer-step history')
                if reward_arm == 'C' and name == 'linear' and milestone == old_control['config']['policy_steps']:
                    old = torch.load(control_directory/'deep_film_last.pt', map_location='cpu', weights_only=True)
                    report['linear_replays_previous_film_exactly'] = previous.equal_state(final.state_dict(), old['model'])
                    if not report['linear_replays_previous_film_exactly']:
                        raise ValueError('Linear control mismatch; do not interpret new arms')
                stats, endpoint = transport.assess(final, rewards, data, transport.SEEDS['endpoint'])
                endpoints[name], policies[name] = endpoint, final
                result['arms'][name].update(state='completed', start_step=start_step, step=milestone,
                    updates_this_stage=milestone-start_step, wall_seconds=time.monotonic()-arm_started,
                    endpoint=stats, contrast_to_source=transport.endpoint_contrast(endpoint, baseline))
                decomposition = {}
                if reward_arm == 'A':
                    decomposition['learned_reward_decomposition'] = transport.learned_decomposition(endpoint, baseline)
                    result['arms'][name].update(decomposition)
                atomic_checkpoint(stage/f'{name}_endpoint.pt', endpoint)
                emit({'phase': 'endpoint', 'arm': name, 'milestone': milestone, 'step': milestone, **stats, **decomposition})
            pairs = [(name, 'linear') for name in arms[1:]]
            if 'mlp_ct' in arms:
                pairs.append(('mlp_ct', 'mlp_c'))
            result['contrasts'] = {f'{a}_minus_{b}': previous.tv_contrast(endpoints[a], endpoints[b])
                                   for a, b in pairs}
            report['state'], result['state'] = 'cold_auditing', 'cold_auditing'
            atomic_json(output/'report.json', report)
            result['audit'] = fit_audits(policies, data, stage/'cold_audit',
                lambda row: emit({**row, 'milestone': milestone}), audit_max_steps)
            # A separate policy-step axis joins fresh endpoints across audit rounds.
            # Raw audit/* curves keep their own classifier-fit step axes.
            for name, stats in result['audit'].get('test', {}).items():
                emit({'phase': 'validation', 'arm': name, 'milestone': milestone,
                      'step': milestone, **stats,
                      'valid': result['audit']['state'] == 'completed' and stats['plateau']})
            scores = torch.load(stage/'cold_audit/test_scores.pt', map_location='cpu', weights_only=True)
            comparisons = audit_contrasts({'baseline': scores['linear'], **{k: scores[k] for k in arms[1:]}})
            result['audit']['route_contrasts'] = {k.replace('_minus_baseline', '_minus_linear'): v for k, v in comparisons.items()}
            if 'mlp_ct' in arms:
                comparisons = audit_contrasts({'baseline': scores['mlp_c'], 'mlp_ct': scores['mlp_ct']})
                result['audit']['time_contrast'] = {k.replace('_minus_baseline', '_minus_mlp_c'): v for k, v in comparisons.items()}
            result['decision'] = decision(report['baseline'], result, reward_arm=reward_arm)
            result['state'] = 'completed' if result['audit']['state'] == 'completed' else 'audit_inconclusive'
            atomic_json(stage/'report.json', result)
            emit({'phase': 'round_decision', 'milestone': milestone, 'step': milestone, **result['decision']})
        report['source_unchanged'] = previous.equal_state(initial.state_dict(), source_state)
        report['initial_models_unchanged'] = {k: previous.equal_state(v.state_dict(), model_starts[k]) for k, v in models.items()}
        report['rewards_unchanged'] = {k: previous.equal_state(v.state_dict(), reward_states[k]) for k, v in rewards.items()}
        if not report['source_unchanged'] or not all(report['initial_models_unchanged'].values()) or not all(report['rewards_unchanged'].values()):
            raise RuntimeError('Fixed initial reference or reward state changed')
        report['decision'] = report['rounds'][str(milestones[-1])]['decision']
        report['state'] = 'completed' if all(r['state'] == 'completed' for r in report['rounds'].values()) else 'audit_inconclusive'
        emit({'phase': 'decision', 'step': milestones[-1], **report['decision']})
        return report
    except BaseException as exc:
        report.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(output/'report.json', report)
        if wb is not None:
            wb.summary['state'] = report['state']
            if 'decision' in report:
                wb.summary.update(flatten_metrics(report['decision'], 'decision/'))
                final_audit = report['rounds'][str(milestones[-1])]['audit']
                wb.summary.update(flatten_metrics(final_audit.get('test', {}), 'final_audit/'))
            wb.finish(exit_code=int(report['state'] == 'failed_or_interrupted'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reward-source', type=Path, default=previous.DEFAULT_SOURCE)
    p.add_argument('--control-source', type=Path, default=CONTROL_SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reward-arm', choices=('A', 'C'), default='C',
                   help='A: frozen learned H4, linear/c-only pair. C: original oracle three-arm experiment.')
    p.add_argument('--milestones', type=int, nargs='+', default=[1000, 2000, 3000])
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--audit-max-steps', type=int, default=16000)
    p.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'), default='offline')
    p.add_argument('--run-id')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    try:
        milestones = validate_milestones(args.milestones)
    except ValueError as exc:
        p.error(str(exc))
    if args.width < 2 or args.audit_max_steps < 2000:
        p.error('Width >=2 and audit maximum >=2000 required')
    torch.set_num_threads(2)
    spec = experiment_spec(args.reward_arm)
    if args.preflight:
        cfg, initial, rewards, prior, _ = load_inputs(args.reward_source, args.control_source, milestones[-1])
        ck = torch.load(args.reward_source/'classifier.pt', map_location='cpu', weights_only=True)
        print(json.dumps({'state': 'preflight_passed_not_started', 'milestones': milestones,
            'reward_arm': args.reward_arm, 'policy_reward': spec['policy_reward'], 'reward_target': spec['reward_target'],
            'source_checkpoint': str(Path(prior['source'])/'reference.pt'),
            'classifier_checkpoint': str(args.reward_source/'classifier.pt'),
            'classifier_selected_step': ck['selected_step'],
            'classifier_frozen': all(not p.requires_grad for p in rewards['A'].parameters()),
            'initial_matching': previous.verify_matching(initial,
                make_models(initial, cfg, args.width, arms=spec['arms']), cfg),
            'wandb_name': spec['run_name']}, indent=2))
        return
    result = run(args.reward_source, args.control_source, args.output, milestones=milestones,
        width=args.width, wandb_mode=args.wandb_mode, run_id=args.run_id,
        audit_max_steps=args.audit_max_steps, reward_arm=args.reward_arm)
    print(json.dumps({'state': result['state'], 'decision': result.get('decision')}))


if __name__ == '__main__':
    main()
