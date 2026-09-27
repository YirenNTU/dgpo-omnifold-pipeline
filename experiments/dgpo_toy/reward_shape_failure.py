"""One-variable failure probe: full-truth versus shape-matched learned H4.

Same nonlinear FiLM diffusion, native DGPO, frozen reference and velocity MSE=1.
No job is launched by importing or using --preflight. See the paired protocol.
"""
import argparse
import copy
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import conditional as native
from . import film_conditioning as film
from . import reward_transport as transport
from .coverage_budget import flatten_metrics
from .cube_swap import modes, swaps
from .film_auc_trajectory import PairedScores, verify_panels
from .nonperiodic_cube import Critic, Data, classification
from .reward_transport_audit import fit_audits
from .truth_pretrain import atomic_checkpoint, atomic_json

ARMS = ('full', 'shape')
# Step zero is a single shared cold fit; each positive milestone fits both arms.
# Concentrate audits before step 300; finish at the user's 1000-update endpoint.
# Measurement does not reset the policy optimizer, RNG, or reference.
MILESTONES = (25, 50, 100, 150, 200, 300, 1000)
RUN_NAME = 'Why reward rises without closure? | full vs shape-matched H4 | nonlinear FiLM | V MSE 1'
SOURCE = Path('artifacts/dgpo_toy/nonperiodic_cube_classifier_extended_v1/reference.pt')


def prepare(source, milestones):
    milestones = film.validate_milestones(milestones)
    ck = torch.load(source, map_location='cpu', weights_only=True)
    if ck.get('trained_on') != 'uniform cube reference, not full truth':
        raise ValueError('Use the original velocity-pretrained uniform-cube reference')
    cfg = replace(native.Config(**ck['config']), policy_steps=milestones[-1],
                  eval_every=100, eval_events=512)
    if (cfg.dimensions, cfg.context_dim, cfg.hidden, cfg.ddim_steps,
            cfg.batch, cfg.candidates, cfg.timesteps, cfg.policy_lr) != (3, 1, 128, 50, 64, 8, 4, 1e-4):
        raise ValueError('Source must preserve the successful FiLM toy configuration')
    source_model = native.Denoiser(cfg).eval()
    source_model.load_state_dict(ck['model'])
    with torch.random.fork_rng():
        torch.manual_seed(451017)
        initial = film.NonlinearFiLM(cfg, source_model.state_dict(), width=32).eval()
    matching = film.previous.verify_matching(source_model, {'nonlinear_film': initial}, cfg)
    return cfg, source_model, initial, matching


def paired_targets(swap_panels):
    """A/C have identical designed modes; D is the shared negative population."""
    full, shape, negative = (swap_panels[k]['negative'] for k in ('A', 'C', 'D'))
    c = swap_panels['D']['c']
    if not all(torch.equal(c, swap_panels[k]['c']) for k in ('A', 'C')):
        raise ValueError('Reward conditions must match')
    if not torch.equal(modes(full), modes(shape)):
        raise ValueError('Positive mode identities must match, not only their probabilities')
    if not all(torch.isfinite(x).all() for x in (c, full, shape, negative)):
        raise ValueError('Nonfinite reward training samples')
    return {name: {'c': c, 'positive': positive, 'negative': negative}
            for name, positive in zip(ARMS, (full, shape))}


def make_reward_panels(source_model, data, output, emit):
    panels, health = {}, {}
    for split, grid, n, seed in [('train', 128, 256, 241017),
                                ('validation', 128, 64, 251017), ('test', 256, 128, 261017)]:
        p, h = swaps(source_model, data, grid, n, 1024, seed)
        panels[split], health[split] = paired_targets(p), h
        atomic_checkpoint(output/f'{split}_panels.pt', panels[split])
        emit({'phase': 'reward_panels', 'arm': 'both', 'step': 0, 'split': split,
              'events': grid*n, 'min_donor_count': h['min_cell_count']})
    return panels, health


def fit_rewards(panels, output, emit, max_steps=16000):
    """Paired H4 fits, same initialization/batches and joint patience budget."""
    with torch.random.fork_rng():
        torch.manual_seed(23)
        initial = Critic(True)
    models = {name: copy.deepcopy(initial) for name in ARMS}
    opts = {name: torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.001)
            for name, m in models.items()}
    rng, labels = native.generator(40023), torch.cat([torch.ones(256), torch.zeros(256)])
    best = dict.fromkeys(ARMS, math.inf)
    anchor, stale, weights, selected = best.copy(), dict.fromkeys(ARMS, 0), {}, {}
    history = []
    for step in range(1, max_steps+1):
        ids = torch.randint(len(panels['train']['full']['c']), (256,), generator=rng)
        train_losses = {}
        for name, model in models.items():
            p = panels['train'][name]
            loss = F.binary_cross_entropy_with_logits(model(
                torch.cat([p['positive'][ids], p['negative'][ids]]), torch.cat([p['c'][ids]]*2)), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite reward classifier loss')
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
            train_losses[name] = float(loss.detach())
        if step % 100 == 0 or step == max_steps:
            row = {'step': step, 'arms': {}}
            for name, model in models.items():
                stats, _ = classification(model, panels['validation'][name])
                if stats['bce'] < best[name]:
                    best[name], weights[name], selected[name] = stats['bce'], copy.deepcopy(model.state_dict()), step
                if stats['bce'] < anchor[name]-1e-4:
                    anchor[name], stale[name] = stats['bce'], 0
                else:
                    stale[name] += 1
                row['arms'][name] = {**stats, 'selected_step': selected[name],
                                     'stale_checks': stale[name], 'train_bce': train_losses[name]}
                emit({'phase': 'reward_fit', 'arm': name, 'step': step, **row['arms'][name]})
                atomic_checkpoint(output/f'{name}_training_state.pt', {
                    'model': model.state_dict(), 'optimizer': opts[name].state_dict(),
                    'rng': rng.get_state(), 'step': step, 'best_model': weights[name],
                    'selected_step': selected[name], 'stale_checks': stale[name]})
            history.append(row)
            atomic_json(output/'fit_report.json', {'state': 'fitting', 'history': history})
            if step >= 2000 and min(stale.values()) >= 20:
                break
    report = {'history': history, 'steps': step, 'arms': {}}
    for name, model in models.items():
        model.load_state_dict(weights[name])
        model.eval().requires_grad_(False)
        stats, _ = classification(model, panels['test'][name])
        valid = step >= 2000 and stale[name] >= 20 and stats['auc'] > .55 and stats['bce'] < math.log(2)-.01
        report['arms'][name] = {**stats, 'valid': valid, 'plateau': stale[name] >= 20,
                                'selected_step': selected[name], 'fit_steps': step}
        atomic_checkpoint(output/f'{name}_reward.pt', {'model': model.state_dict(),
            'selected_step': selected[name], 'target': name, 'test': stats})
        emit({'phase': 'reward_test', 'arm': name, 'step': step, **report['arms'][name]})
    report['valid'] = all(v['valid'] for v in report['arms'].values())
    report['state'] = 'completed' if report['valid'] else 'inconclusive_reward_fit'
    atomic_json(output/'fit_report.json', report)
    return models, report


@torch.no_grad()
def measure(model, rewards, data, seed=321017, *, grid=128, k=1024):
    c = ((torch.arange(grid)+.5)/grid*2-1)[:, None]
    y = transport.generate(model, c, k, seed)
    metrics, payload = transport.sample_metrics(y, c, data)
    values = {name: r(y, c[:, None]).detach() for name, r in rewards.items()}
    metrics['rewards'] = {name: {'all_mean': float(r.double().mean()),
                                'best_of_16_mean': float(r[:, :16].max(1).values.double().mean()),
                                **native.weight_health(r)} for name, r in values.items()}
    payload.update(contexts=c, rewards=values,
                   cells={name: transport.cells(y, r) for name, r in values.items()})
    return metrics, payload


def decompose(after, before, reward_name):
    if not torch.equal(after['contexts'], before['contexts']):
        raise ValueError('Decomposition needs paired condition grids')
    cell = before['cells'][reward_name]
    if not cell['mean_available'].all():
        return {'available': False, 'reason': 'Missing baseline mode'}
    mode = float(((after['q']-before['q'])*cell['mean']).sum(-1).mean())
    total = float((after['rewards'][reward_name]-before['rewards'][reward_name]).double().mean())
    return {'available': True, 'total_gain': total, 'mode_probability': mode,
            'within_mode_shape_and_interaction': total-mode,
            'scope': 'Frozen source cell-score decomposition, not unique optimizer or KL causality'}


def paired_audit_contrasts(predictions, repeats=2000):
    """Simultaneous final-endpoint baseline and between-arm paired intervals."""
    if repeats < 100:
        raise ValueError('At least 100 bootstrap repeats')
    scorers = {name: PairedScores(p) for name, p in predictions.items()}
    n = scorers['baseline'].n
    if any(s.n != n for s in scorers.values()):
        raise ValueError('Mismatched paired audit populations')
    pairs = [('full', 'baseline'), ('shape', 'baseline'), ('full', 'shape')]
    point = {k: s.metrics(np.ones(n)) for k, s in scorers.items()}
    delta = np.stack([point[a]-point[b] for a, b in pairs])
    draws = np.empty((repeats, len(pairs), 3))
    rng = np.random.default_rng(501017)
    for i in range(repeats):
        counts = np.bincount(rng.integers(n, size=n), minlength=n)
        values = {k: s.metrics(counts) for k, s in scorers.items()}
        draws[i] = [values[a]-values[b] for a, b in pairs]
    radius = np.quantile(np.max(np.abs(draws-delta), axis=1), .95, axis=0)
    return {f'{a}_minus_{b}': {name: {'delta': float(delta[j, i]),
             'lo95': float(delta[j, i]-radius[i]), 'hi95': float(delta[j, i]+radius[i])}
             for i, name in enumerate(('auc', 'auc_gap', 'bce'))}
            for j, (a, b) in enumerate(pairs)}


def decide(valid, contrasts, gains):
    if not valid:
        return {'interpretation': 'inconclusive_audit', 'failure_reproduced': None}
    full, shape = (contrasts[f'{name}_minus_baseline'] for name in ARMS)
    reward_moves = gains['full']['lo95'] > .01
    # Failure to reject zero is NOT evidence of plateau: exclude meaningful rescue.
    no_material_closure = full['auc_gap']['lo95'] > -.005 and full['bce']['hi95'] < .005
    failure = reward_moves and no_material_closure
    rescue = shape['auc_gap']['hi95'] < -.005 and shape['bce']['lo95'] > 0
    difference = contrasts['full_minus_shape']['auc_gap']['lo95'] > 0
    return {'failure_reproduced': failure, 'full_reward_gain_resolved': reward_moves,
            'full_material_closure_excluded': no_material_closure,
            'full_audit_worsening_resolved': full['auc_gap']['lo95'] > 0,
            'shape_control_rescue': rescue, 'between_arm_difference_resolved': difference,
            'interpretation': ('supports_reward_target_package_as_contributor' if failure and rescue and difference
                else 'failure_reproduced_core_unresolved' if failure
                else 'no_failure_reproduced_or_precision_insufficient'),
            'scope': 'Single seed; learned score scale, ESS and accuracy can change with target. Not unique shape/KL causality.'}


def run(source, output, *, milestones=MILESTONES, max_fit_steps=16000,
        bootstrap_repeats=2000, wandb_mode='offline', run_id=None):
    if max_fit_steps < 2000 or bootstrap_repeats < 100:
        raise ValueError('Require >=2000 maximum fit updates and >=100 bootstrap repeats')
    cfg, source_model, initial, matching = prepare(source, milestones)
    milestones = film.validate_milestones(milestones)
    data, start = Data(cfg), time.monotonic()
    report = {'state': 'prepared_not_started', 'source': str(source.resolve()),
        'question': 'Does full-truth reward reintroduce reward gain without fresh closure under working FiLM?',
        'config': asdict(cfg), 'initial_matching': matching, 'milestones': list(milestones),
        'primary_endpoint': milestones[-1], 'velocity_coefficient': 1., 'conditioning': 'nonlinear c-only FiLM width32 three blocks',
        'intervention': 'Only reward positive within-mode shape; modes, negatives, fit settings and policy setup paired',
        'audit': {'min_steps': 2000, 'max_steps': max_fit_steps, 'seed': 41, 'cold': True,
                  'policy_steps': [0, *milestones], 'planned_cold_fits': 1+len(ARMS)*len(milestones),
                  'shared_baseline': True},
        'bootstrap_repeats': bootstrap_repeats, 'rounds': {},
        'scope': 'Local diagnostic. Shape matching uses known toy modes, not a production prescription.'}
    output.mkdir(parents=True, exist_ok=False)
    wb = None

    def emit(row):
        report['active'] = {k: row[k] for k in ('phase', 'arm', 'step', 'milestone') if k in row}
        report['elapsed_seconds'] = time.monotonic()-start
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        atomic_json(output/'report.json', report)
        if wb is not None:
            prefix = (f"audit/policy{row['milestone']}/{row['arm']}" if row['phase'].startswith('audit')
                      else f"{row['phase']}/{row.get('arm', 'both')}")
            wb.log(flatten_metrics(row, prefix+'/'))
        print(json.dumps(row, allow_nan=False), flush=True)

    def audit_one(name, policy, step):
        folder = output/f'round_{step}'/name/'cold_audit'
        result = fit_audits({name: policy}, data, folder,
            lambda row: emit({**row, 'milestone': step}), max_fit_steps)
        if step:
            verify_panels(output/'round_0/baseline/cold_audit/baseline_panels.pt', folder/f'{name}_panels.pt')
        stats = result['test'][name]
        stats['valid'] = result['state'] == 'completed' and stats['plateau']
        emit({'phase': 'validation', 'arm': name, 'step': step, **stats})
        scores = torch.load(folder/'test_scores.pt', map_location='cpu', weights_only=True)[name]
        return stats, scores

    try:
        if wandb_mode != 'disabled':
            import wandb
            wb = wandb.init(project='dgpo-toy', name=RUN_NAME, id=run_id, mode=wandb_mode,
                dir=str(output.resolve()), group='Conditional reward transport',
                tags=['paired-reward-target', 'nonlinear-film', 'velocity-mse-1', 'raw-no-ema'], config=copy.deepcopy(report))
            report['wandb'] = {'id': wb.id, 'name': RUN_NAME, 'mode': wandb_mode}
            for phase in ('reward_fit', 'reward_test', 'policy', 'conditioning', 'structure', 'validation', 'endpoint'):
                for arm in (*ARMS, 'baseline'):
                    prefix = f'{phase}/{arm}'
                    wb.define_metric(prefix+'/step')
                    wb.define_metric(prefix+'/*', step_metric=prefix+'/step')
            for step in (0, *milestones):
                for arm in ('baseline',) if step == 0 else ARMS:
                    prefix = f'audit/policy{step}/{arm}'
                    wb.define_metric(prefix+'/step')
                    wb.define_metric(prefix+'/*', step_metric=prefix+'/step')
        reward_dir = output/'rewards'
        reward_dir.mkdir()
        report['state'] = 'preparing_paired_rewards'
        panels, report['panel_health'] = make_reward_panels(source_model, data, reward_dir, emit)
        rewards, report['reward_fit'] = fit_rewards(panels, reward_dir, emit, max_fit_steps)
        del panels
        if not report['reward_fit']['valid']:
            report['state'] = 'inconclusive_reward_fit'
            return report
        originals = {name: copy.deepcopy(m.state_dict()) for name, m in
                     {'source': source_model, 'initial': initial, **rewards}.items()}
        report['baseline'], baseline = measure(initial, rewards, data)
        atomic_checkpoint(output/'baseline_endpoint.pt', baseline)
        report['baseline_audit'], base_scores = audit_one('baseline', initial, 0)
        if not report['baseline_audit']['valid']:
            report['state'] = 'audit_inconclusive'
            return report
        # Plot both arm trajectories from their identical, once-audited source.
        # These are aliases of one fit, not two independent baseline measurements.
        for arm in ARMS:
            emit({'phase': 'validation', 'arm': arm, 'step': 0,
                  **report['baseline_audit'], 'shared_baseline': True})
        probe = film.previous.condition_probe(initial, cfg)
        states, final_scores, final_gains = dict.fromkeys(ARMS, None), {}, {}
        for step in milestones:
            report['rounds'][str(step)] = {}
            for arm in ARMS:
                report['state'] = 'training_dgpo'
                stage = output/f'round_{step}'/arm
                stage.mkdir(parents=True)

                def checkpoint(update, model, opt, rng, history):
                    if update == 1 or update % 100 == 0 or update == step:
                        state = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                            'rng': rng.get_state(), 'history': history, 'step': update,
                            'config': asdict(replace(cfg, policy_steps=step)), 'velocity_coefficient': 1.,
                            'reward_arm': arm, 'source': str(source.resolve()), 'route': 'mlp_c', 'width': 32}
                        atomic_checkpoint(output/f'{arm}_last.pt', state)
                        emit({'phase': 'conditioning', 'arm': arm, 'step': update,
                              **film.diagnostics(model, initial, probe)})
                        if update == step:
                            atomic_checkpoint(stage/'state.pt', state)
                            states[arm] = copy.deepcopy(state)

                if states[arm] is not None and states[arm]['reward_arm'] != arm:
                    raise ValueError('Cannot resume a policy under another reward')
                policy, _ = native.policy_train('dgpo', initial, rewards[arm], data,
                    replace(cfg, policy_steps=step), 17, 284017,
                    lambda row: emit({**row, 'arm': arm}), checkpoint,
                    resume_state=states[arm], velocity_coefficient=1.)
                metrics, endpoint = measure(policy, rewards, data)
                gain = native.paired_gain(endpoint['rewards'][arm], baseline['rewards'][arm])
                stats = {'metrics': metrics, 'own_reward_gain': gain,
                         'decomposition': decompose(endpoint, baseline, arm)}
                report['rounds'][str(step)][arm] = stats
                atomic_checkpoint(stage/'endpoint.pt', endpoint)
                emit({'phase': 'endpoint', 'arm': arm, 'step': step, **stats})
                report['state'] = 'cold_auditing'
                stats['audit'], final_scores[arm] = audit_one(arm, policy, step)
                final_gains[arm] = gain
        final = report['rounds'][str(milestones[-1])]
        valid = all(v['audit']['valid'] for v in final.values())
        report['contrasts'] = paired_audit_contrasts({'baseline': base_scores, **final_scores}, bootstrap_repeats)
        report['decision'] = decide(valid, report['contrasts'], final_gains)
        report['fixed_states_unchanged'] = all(film.previous.equal_state(m.state_dict(), originals[name])
            for name, m in {'source': source_model, 'initial': initial, **rewards}.items())
        if not report['fixed_states_unchanged']:
            raise RuntimeError('Frozen reward/source was mutated')
        report['state'] = 'completed' if valid else 'audit_inconclusive'
        emit({'phase': 'decision', 'step': milestones[-1], **report['decision']})
        return report
    except BaseException as exc:
        report.update(state='failed_or_interrupted', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        atomic_json(output/'report.json', report)
        if wb is not None:
            wb.summary['state'] = report['state']
            wb.summary.update(flatten_metrics(report.get('decision', {}), 'decision/'))
            wb.finish(exit_code=int(report['state'] == 'failed_or_interrupted'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--milestones', type=int, nargs='+', default=list(MILESTONES),
                   help='Positive policy audit steps; a shared step-0 cold audit is always included.')
    p.add_argument('--max-fit-steps', type=int, default=16000)
    p.add_argument('--bootstrap-repeats', type=int, default=2000)
    p.add_argument('--wandb-mode', choices=('offline', 'online', 'disabled'), default='offline')
    p.add_argument('--run-id')
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    if a.max_fit_steps < 2000 or a.bootstrap_repeats < 100:
        p.error('max-fit-steps >=2000, bootstrap-repeats >=100 required')
    torch.set_num_threads(2)
    if a.preflight:
        cfg, _, _, matching = prepare(a.source, a.milestones)
        print(json.dumps({'state': 'preflight_passed_not_started', 'source': str(a.source.resolve()),
            'config': asdict(cfg), 'initial_matching': matching, 'arms': ARMS,
            'milestones': a.milestones, 'audit_steps': [0, *a.milestones],
            'planned_cold_fits': 1+len(ARMS)*len(a.milestones), 'name': RUN_NAME}, indent=2))
        return
    result = run(a.source, a.output, milestones=a.milestones, max_fit_steps=a.max_fit_steps,
        bootstrap_repeats=a.bootstrap_repeats, wandb_mode=a.wandb_mode, run_id=a.run_id)
    print(json.dumps({'state': result['state'], 'decision': result.get('decision')}))


if __name__ == '__main__':
    main()
