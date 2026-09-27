"""One preregistered round of the local-only conditional-cube research loop.

The assistant reviews SUMMARY.md before scheduling another round. This module
never chooses experiments from test scores, launches remote jobs, or changes
the inherited DGPO objective. See CLOSED_LOOP_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import conditional as native
from . import film_conditioning as film
from . import reward_shape_failure as previous
from .cube_lockdown import condition_features
from .film_auc_trajectory import PairedScores
from .nonperiodic_cube import Critic, Data, panel
from .truth_pretrain import atomic_checkpoint, atomic_json

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts/dgpo_toy'
SOURCE = ROOT / previous.SOURCE
HISTORY = ARTIFACTS / 'reward_shape_failure_early1k_v1'
FEATURES = ('plain', 'c_only', 'y_only', 'both')


def toy_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ARTIFACTS.resolve()):
        raise ValueError('All experiment outputs must stay under artifacts/dgpo_toy')
    return path


class FeatureCritic(Critic):
    """Same MLP and initialization; only which Fourier input slots are active differs."""
    def __init__(self, features='both', width=128):
        if features not in FEATURES:
            raise ValueError(features)
        super().__init__(True, width)
        self.feature_mode = features

    def features(self, y, c):
        value = super().features(y, c)
        mask = value.new_tensor({'plain': [0, 0, 0, 0], 'c_only': [1, 0, 0, 0],
                                 'y_only': [0, 1, 1, 1], 'both': [1, 1, 1, 1]}[self.feature_mode])
        return torch.cat((value[..., :4], value[..., 4:] * mask.repeat(8)), -1)


class ConditioningAblation(film.NonlinearFiLM):
    """Factorize basis, modulation and encoder while preserving initial function.

    'raw' is repeated standardized c (rank one), NOT the old zero 'raw'
    feature implementation. Same nominal parameter count is not equal effective
    capacity. Encoder LayerNorm is a separate nonlinear operation.
    """
    def __init__(self, cfg, state, *, basis='fourier', modulation='film',
                 nonlinear=True, normalization=True, layers=3, width=32, encoder_gain=1.):
        if basis not in ('fourier', 'raw', 'polynomial') or modulation not in ('film', 'shift', 'scale'):
            raise ValueError('Unknown conditioning factor')
        if layers not in (1, 3):
            raise ValueError('Use one or three active modulation blocks')
        if not math.isfinite(encoder_gain) or encoder_gain <= 0:
            raise ValueError('Fixed encoder gain must be positive and finite')
        super().__init__(cfg, state, width=width)
        self.basis, self.modulation_type = basis, modulation
        self.nonlinear, self.normalization, self.layers = nonlinear, normalization, layers
        self.encoder_gain = float(encoder_gain)

    def modulation(self, c, t):
        features = (c.expand(*c.shape[:-1], 8) * math.sqrt(3)
                    if self.basis == 'raw' else condition_features(c, self.basis))
        z = self.condition_input(features)
        z = (self.condition_hidden(z) if self.nonlinear else self.condition_hidden[1](z))
        if self.normalization:
            z = self.condition_norm(z)
        if self.encoder_gain != 1.:
            z = z * self.encoder_gain
        result = []
        for i, head in enumerate(self.condition_heads):
            gamma, beta = head(z).chunk(2, -1)
            if self.modulation_type == 'shift' or i >= self.layers:
                gamma = gamma * 0
            if self.modulation_type == 'scale' or i >= self.layers:
                beta = beta * 0
            result.append((gamma, beta))
        return result, z


@torch.no_grad()
def initial_encoder_rms_gain(cfg, state, *, seed, spec, grid=2048):
    """One constant gain from initialized c-only features; no y, truth or reward.

    Unlike LayerNorm, this does not center or normalize each event and is never
    recalibrated as training evolves. Default model parameters/state are unchanged.
    """
    if spec.get('normalization', True) or spec.get('encoder_gain', 1.) != 1. or grid < 64:
        raise ValueError('Calibrate only the unnormalized unit-gain encoder')
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ConditioningAblation(cfg, state, **spec).eval()
    c = ((torch.arange(grid)+.5)/grid*2-1)[:,None]
    _, z = model.modulation(c, torch.zeros(grid))
    rms = float(z.square().mean().sqrt())
    if not math.isfinite(rms) or rms < 1e-8:
        raise ValueError('Degenerate initialized encoder RMS')
    return {'gain':1./rms,'initial_encoder_rms':rms,'target_rms':1.,'grid':grid,
            'rule':'Frozen scalar reciprocal of initial global encoder RMS over midpoint c grid [-1,1]',
            'uses_truth_or_reward':False,'recalibrated_during_training':False}


@torch.no_grad()
def predictions(model, p):
    return {s: torch.cat([model(y, c) for y, c in zip(p[s].split(1024), p['c'].split(1024))])
            for s in ('positive', 'negative')}


def score_metrics(scores):
    metric = PairedScores(scores)
    auc, gap, bce = metric.metrics(np.ones(metric.n))
    return {'auc': float(auc), 'auc_gap': float(gap), 'bce': float(bce),
            'weight': native.weight_health(scores['negative'])}


def compare_scores(scores, pairs, *, repeats=1000):
    """Simultaneous paired-context intervals over the declared contrasts, per metric."""
    if not pairs:
        return {}
    if repeats < 100:
        raise ValueError('At least 100 bootstrap repetitions')
    scorers = {k: PairedScores(v) for k, v in scores.items()}
    n = next(iter(scorers.values())).n
    if any(s.n != n for s in scorers.values()):
        raise ValueError('Unpaired sample lengths')
    point = {k: s.metrics(np.ones(n)) for k, s in scorers.items()}
    delta = np.stack([point[a]-point[b] for a, b in pairs])
    draws = np.empty((repeats, len(pairs), 3))
    rng = np.random.default_rng(901027)
    for j in range(repeats):
        counts = np.bincount(rng.integers(n, size=n), minlength=n)
        values = {k: s.metrics(counts) for k, s in scorers.items()}
        draws[j] = [values[a]-values[b] for a, b in pairs]
    radius = np.quantile(np.max(np.abs(draws-delta), axis=1), .95, axis=0)
    return {f'{a} minus {b}': {key: {'delta': float(delta[j, i]),
                    'lo95': float(delta[j, i]-radius[i]), 'hi95': float(delta[j, i]+radius[i])}
             for i, key in enumerate(('auc', 'auc_gap', 'bce'))} for j, (a, b) in enumerate(pairs)}


def assert_pairing(panels):
    anchor = next(iter(panels.values()))
    for split in ('train', 'validation', 'test'):
        for item in panels.values():
            for key in ('c', 'positive'):
                if not torch.equal(anchor[split][key], item[split][key]):
                    raise ValueError('Lost condition/truth pairing; comparisons would not be paired')


def fit_classifiers(panels, feature_modes, output, emit, *, seed=83, max_steps=32000,
                    min_steps=2000, patience=20, check_every=100, min_delta=1e-4,
                    width=128, model_factory=FeatureCritic):
    """Cold fits, shared initialization and batch identities, validation-only selection.

    Every arm has the same maximum budget and its own identical early-stop rule.
    Training longer than patience does not reset another arm. Test is evaluated
    once after selection. Explicit RNG makes audits invisible to policy sampling.
    """
    output = toy_path(output)
    output.mkdir(parents=True, exist_ok=False)
    assert_pairing(panels)
    if max_steps < min_steps or min_steps < 1 or type(width) is not int or width < 1:
        raise ValueError('Invalid fit budget')
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        initial = model_factory(feature_modes[0], width=width)
    models, caches, opts = {}, {}, {}
    for dataset, pp in panels.items():
        for feature in feature_modes:
            name = f'{dataset}__{feature}'
            m = copy.deepcopy(initial)
            m.feature_mode = feature
            models[name] = m
            opts[name] = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.001)
            with torch.no_grad():
                caches[name] = {split: {s: m.features(p[s], p['c']) for s in ('positive', 'negative')}
                                for split, p in pp.items() if split != 'test'}
    best, anchor = dict.fromkeys(models, math.inf), dict.fromkeys(models, math.inf)
    stale, selected, weights, finished = dict.fromkeys(models, 0), {}, {}, {}
    rng = native.generator(seed+380000)
    labels = torch.cat([torch.ones(256), torch.zeros(256)])
    train_n = len(next(iter(panels.values()))['train']['c'])
    report = {'state': 'fitting', 'seed': seed, 'width': width,
              'model_type': type(initial).__name__,
              'parameter_count': sum(p.numel() for p in initial.parameters()),
              'max_steps': max_steps, 'minimum_steps': min_steps,
              'check_every': check_every, 'patience_checks': patience, 'min_delta': min_delta,
              'cold_start': True, 'shared_initialization_and_batches': True, 'history': [],
              'selection': 'exact minimum validation BCE; test never selects checkpoint',
              'uncertainty': 'Bootstrap conditional on trained models, not across training seeds'}
    for step in range(1, max_steps+1):
        ids = torch.randint(train_n, (256,), generator=rng)
        losses = {}
        for name, m in models.items():
            if name in finished:
                continue
            features = caches[name]['train']
            loss = F.binary_cross_entropy_with_logits(m.net(torch.cat(
                [features['positive'][ids], features['negative'][ids]])).squeeze(-1), labels)
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            opts[name].step()
            losses[name] = (float(loss.detach()), float(norm))
        if step % check_every == 0 or step == max_steps:
            for name, m in models.items():
                if name in finished:
                    continue
                with torch.no_grad():
                    pred = {s: torch.cat([m.net(x).squeeze(-1) for x in xx.split(1024)])
                            for s, xx in caches[name]['validation'].items()}
                stats = score_metrics(pred)
                value = stats['bce']
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite validation BCE')
                if value < best[name]:
                    best[name], selected[name], weights[name] = value, step, copy.deepcopy(m.state_dict())
                if value < anchor[name]-min_delta:
                    anchor[name], stale[name] = value, 0
                else:
                    stale[name] += 1
                if step >= min_steps and stale[name] >= patience:
                    finished[name] = step
                row = {'phase': 'classifier', 'arm': name, 'step': step,
                       'validation_bce': value, 'validation_auc': stats['auc'],
                       'train_bce': losses[name][0], 'gradient_norm': losses[name][1],
                       'selected_step': selected[name], 'stale_checks': stale[name],
                       'stopped': name in finished}
                report['history'].append(row)
                emit(row)
                atomic_checkpoint(output/f'{name}_last.pt', {'model': m.state_dict(),
                    'optimizer': opts[name].state_dict(), 'rng': rng.get_state(), 'step': step,
                    'best_model': weights[name], 'best_bce': best[name], 'anchor': anchor[name],
                    'stale': stale[name], 'selected_step': selected[name], 'feature': m.feature_mode,
                    'width': width})
            report['step'] = step
            atomic_json(output/'fit_report.json', report)
            if len(finished) == len(models):
                break
    result_scores, report['test'] = {}, {}
    for name, m in models.items():
        m.load_state_dict(weights[name])
        m.eval().requires_grad_(False)
        dataset = name.rsplit('__', 1)[0]
        result_scores[name] = predictions(m, panels[dataset]['test'])
        stats = score_metrics(result_scores[name])
        report['test'][name] = {**stats, 'fit_steps': finished.get(name, step),
                               'selected_step': selected[name], 'best_validation_bce': best[name],
                               'plateau': name in finished, 'valid': name in finished}
        atomic_checkpoint(output/f'{name}_best.pt', {'model': m.state_dict(),
                            'feature': m.feature_mode, 'selected_step': selected[name], 'width': width})
        emit({'phase': 'classifier_test', 'arm': name, 'step': step, **report['test'][name]})
    atomic_checkpoint(output/'test_scores.pt', result_scores)
    report['state'] = 'completed' if len(finished) == len(models) else 'inconclusive_fit_budget'
    atomic_json(output/'fit_report.json', report)
    return report, result_scores


def load_source():
    ck = torch.load(SOURCE, map_location='cpu', weights_only=True)
    if ck.get('trained_on') != 'uniform cube reference, not full truth':
        raise ValueError('Wrong pretraining source')
    cfg = native.Config(**ck['config'])
    model = native.Denoiser(cfg)
    model.load_state_dict(ck['model'])
    return cfg, model.eval()


def generate_panels(policy, data, seed):
    return {split: panel(data, n, seed+offset, policy) for split, n, offset in
            [('train', 32768, 10000), ('validation', 8192, 20000), ('test', 16384, 30000)]}


def classifier_round(plan, output, emit):
    cfg, source = load_source()
    data, panels = Data(cfg), {}
    for name, spec in plan['datasets'].items():
        if 'panels' in spec:
            panels[name] = torch.load(toy_path(spec['panels']), map_location='cpu', weights_only=True)
        else:
            kind = spec.get('policy', 'source')
            if kind not in ('source', 'ideal'):
                raise ValueError('Use saved paired panels for existing policy trajectories')
            panels[name] = generate_panels(source if kind == 'source' else None, data, plan['panel_seed'])
        atomic_checkpoint(output/f'{name}_panels.pt', panels[name])
        emit({'phase': 'panels', 'arm': name, 'step': 0})
    saved_scores, saved_metadata = {}, {}
    if 'saved_classifier_controls' in plan:
        from .classifier_feature_controls import load_controls
        saved_scores, saved_metadata = load_controls(plan, panels)
    fits, scores = fit_classifiers(panels, plan['features'], output/'classifiers', emit,
                                  seed=plan['classifier_seed'], max_steps=plan['max_fit_steps'],
                                  min_steps=plan.get('minimum_fit_steps', 2000),
                                  width=plan.get('classifier_width', 128))
    if set(scores) & set(saved_scores):
        raise ValueError('Saved classifier aliases collide with new fits')
    comparisons = compare_scores({**scores, **saved_scores}, plan['contrasts'], repeats=plan['bootstrap_repeats'])
    valid = all(v['valid'] for v in fits['test'].values())
    result = {'state': 'completed' if valid else 'inconclusive_fit_budget', 'fit': fits,
              'contrasts': comparisons, 'valid': valid}
    if saved_metadata:
        result['saved_classifier_controls'] = saved_metadata
    if 'classifier_width_controls' in plan:
        from .classifier_capacity_comparison import compare_saved_width
        result['width_comparison'] = compare_saved_width(plan, panels, fits, scores)
    return result


def rl_round(plan, output, emit):
    if 'external_control' in plan:
        validate_external_control(plan)
    cfg, source = load_source()
    milestones = film.validate_milestones(plan['milestones'])
    cfg = replace(cfg, policy_steps=milestones[-1], eval_every=25, eval_events=512)
    gain_calibration = None
    if 'encoder_gain_calibration' in plan:
        if len(plan['conditioning']) != 1:
            raise ValueError('Fixed RMS calibration requires one gain arm')
        calibrated_spec = next(iter(plan['conditioning'].values()))
        gain_calibration = initial_encoder_rms_gain(cfg, source.state_dict(),
            seed=plan['initialization_seed'], spec={**calibrated_spec,'encoder_gain':1.},
            grid=plan['encoder_gain_calibration']['grid'])
        if not math.isclose(calibrated_spec['encoder_gain'],gain_calibration['gain'],rel_tol=1e-10):
            raise ValueError('Saved fixed gain does not match the preregistered initialization-only rule')
    data, policies = Data(cfg), {}
    for name, spec in plan['conditioning'].items():
        with torch.random.fork_rng():
            torch.manual_seed(plan['initialization_seed'])
            policies[name] = ConditioningAblation(cfg, source.state_dict(), **spec).eval()
    matching = film.previous.verify_matching(source, policies, cfg)
    reward_ck = torch.load(HISTORY/'rewards/full_reward.pt', map_location='cpu', weights_only=True)
    reward = Critic(True).eval().requires_grad_(False)
    reward.load_state_dict(reward_ck['model'])
    fits = json.loads((HISTORY/'rewards/fit_report.json').read_text())
    if not fits['arms']['full']['valid']:
        raise ValueError('Source frozen reward fit was not adequate')
    report = {'config': asdict(cfg), 'initial_matching': matching, 'velocity_coefficient': 1.,
              'reward': str(HISTORY/'rewards/full_reward.pt'), 'points': {}, 'audits': {}}
    if gain_calibration is not None:
        report['encoder_gain_calibration'] = gain_calibration
    scores_all, audit_panels = {}, {}
    base_metrics, baseline = previous.measure(source, {'full': reward}, data, grid=64, k=512)
    atomic_checkpoint(output/'baseline_endpoint.pt', baseline)
    report['baseline'] = base_metrics
    for step in (0, *milestones):
        at_step = {'baseline': source} if step == 0 else {}
        if step:
            for name, initial in policies.items():
                state_path = output/f'{name}_last.pt'
                resume = torch.load(state_path, map_location='cpu', weights_only=True) if state_path.exists() else None
                def checkpoint(s, m, opt, rng, history):
                    if s % 25 == 0 or s == step:
                        state = {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state(),
                                 'history': history, 'step': s, 'velocity_coefficient': 1.,
                                 'config': asdict(cfg), 'conditioning': plan['conditioning'][name]}
                        atomic_checkpoint(state_path, state)
                        if s == step:
                            atomic_checkpoint(output/f'{name}_step{s}.pt', state)
                model, _ = native.policy_train('dgpo', initial, reward, data,
                    replace(cfg, policy_steps=step), plan['policy_seed'], plan['monitor_seed'],
                    lambda row: emit({**row, 'arm': name}), checkpoint,
                    resume_state=resume, velocity_coefficient=1.)
                at_step[name] = model
                metrics, after = previous.measure(model, {'full': reward}, data, grid=64, k=512)
                atomic_checkpoint(output/f'{name}_endpoint{step}.pt', after)
                report['points'][f'{name}_{step}'] = {'metrics': metrics,
                    'reward_gain': native.paired_gain(after['rewards']['full'], baseline['rewards']['full']),
                    'reward_decomposition': previous.decompose(after, baseline, 'full'),
                    'conditioning': film.diagnostics(model, source, film.previous.condition_probe(source, cfg))}
                emit({'phase': 'endpoint', 'arm': name, 'step': step,
                      **report['points'][f'{name}_{step}']})
        panels = {name: generate_panels(model, data, plan['panel_seed']) for name, model in at_step.items()}
        if step == 0:
            audit_panels = panels
        assert_pairing({**audit_panels, **panels})
        for name, pp in panels.items():
            atomic_checkpoint(output/f'{name}_step{step}_panels.pt', pp)
        audit, score = fit_classifiers(panels, plan.get('audit_features', ['both']),
                output/f'audit{step}', lambda row: emit({**row, 'policy_step': step}),
                seed=plan['classifier_seed'], max_steps=plan['max_fit_steps'],
                min_steps=plan.get('minimum_fit_steps', 2000))
        report['audits'][str(step)] = audit
        scores_all.update({f'{name}_step{step}': ss for name, ss in score.items()})
        if step == 0 and 'external_control' in plan and audit['state'] == 'completed':
            check_external_baseline(plan, report, scores_all)
        atomic_json(output/'rl_progress.json', report)
        if audit['state'] != 'completed':
            report.update(state='inconclusive_audit_fit', valid=False)
            return report
    pairs = []
    arms = list(policies)
    for feature in plan.get('audit_features', ['both']):
        baseline_key = f'baseline__{feature}_step0'
        for arm in arms:
            for i, step in enumerate(milestones):
                key = f'{arm}__{feature}_step{step}'
                pairs.append([key, baseline_key])
                if i:
                    pairs.append([key, f'{arm}__{feature}_step{milestones[i-1]}'])
        for arm in arms[1:]:
            pairs.append([f'{arm}__{feature}_step{milestones[-1]}', f'{arms[0]}__{feature}_step{milestones[-1]}'])
    report['contrasts'] = compare_scores(scores_all, pairs, repeats=plan['bootstrap_repeats'])
    atomic_checkpoint(output/'test_scores.pt', scores_all)
    if 'external_control' in plan:
        external = compare_external_control(plan, output, report, scores_all)
        report['external_control'] = external
        report['contrasts'].update(external['contrasts'])
    report.update(state='completed', valid=True)
    return report


def validate_external_control(plan):
    """Reuse a saved one-factor control only when sampling/fit/training clocks match."""
    spec = plan['external_control']
    directory = toy_path(spec['directory'])
    old = json.loads((directory/'plan.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    if report['state'] != 'completed' or not report['valid']:
        raise ValueError('External control lacks valid completed audits')
    fields = ('initialization_seed','policy_seed','monitor_seed','panel_seed','classifier_seed',
              'audit_features','minimum_fit_steps','max_fit_steps','milestones')
    if any(plan.get(k) != old.get(k) for k in fields):
        raise ValueError('Control seed, sampling, fit budget or milestone mismatch')
    if len(plan['conditioning']) != 1 or spec['changed_factor'] not in ('modulation','nonlinear','normalization','layers','encoder_gain'):
        raise ValueError('This reuse contract permits one declared conditioning-factor arm only')
    arm = next(iter(plan['conditioning']))
    new_cfg = {'encoder_gain':1., **plan['conditioning'][arm]}
    old_cfg = {'encoder_gain':1., **old['conditioning'][spec['arm']]}
    changed = {k for k in set(new_cfg)|set(old_cfg) if new_cfg.get(k) != old_cfg.get(k)}
    if changed != {spec['changed_factor']}:
        raise ValueError('The reused control differs in more than the declared factor')
    return directory, report


def check_external_baseline(plan, report, scores):
    directory, old = validate_external_control(plan)
    for key in ('config','reward','velocity_coefficient'):
        if report[key] != old[key]:
            raise ValueError(f'Reused control changes {key}')
    old_scores = torch.load(directory/'test_scores.pt',map_location='cpu',weights_only=True)
    # Exact repeated source-audit predictions verify the saved sampling/fit path.
    for feature in plan['audit_features']:
        base = f'baseline__{feature}_step0'
        for label in ('positive','negative'):
            if not torch.equal(scores[base][label],old_scores[base][label]):
                raise ValueError('Baseline audit replay differs; do not use saved control')
    return directory, old_scores


def compare_external_control(plan, output, report, scores):
    directory, old_scores = check_external_baseline(plan, report, scores)
    current_arm = next(iter(plan['conditioning']))
    control_arm = plan['external_control']['arm']
    merged, pairs = dict(scores), []
    for step in plan['milestones']:
        current = torch.load(output/f'{current_arm}_step{step}_panels.pt',map_location='cpu',weights_only=True)
        control = torch.load(directory/f'{control_arm}_step{step}_panels.pt',map_location='cpu',weights_only=True)
        assert_pairing({'new':current,'saved_control':control})
        for feature in plan['audit_features']:
            key = f'{control_arm}__{feature}_step{step}'
            name = 'saved_control/'+key
            merged[name] = old_scores[key]
            pairs.append([f'{current_arm}__{feature}_step{step}',name])
    return {'directory':str(directory),'arm':control_arm,'baseline_predictions_exact':True,
            'all_panels_paired':True,'only_changed_factor':plan['external_control']['changed_factor'],
            'contrasts':compare_scores(merged,pairs,repeats=plan['bootstrap_repeats'])}


def summarize(plan, result):
    lines = [f"# Round {plan['round']}: {plan['question']}", '',
             f"State: {result['state']}. Elapsed: {result['elapsed_seconds']:.1f} seconds.", '',
             f"Primary endpoint: {plan['primary_endpoint']}", '',
             '## Evidence', '', '| Classifier | AUC | BCE | Fit / selected steps | Adequate |',
             '|---|---:|---:|---:|---|']
    fits = {'': result['fit']} if 'fit' in result else result.get('audits', {})
    for checkpoint, fit in fits.items():
        for name, value in fit['test'].items():
            lines.append(f"| {checkpoint} {name} | {value['auc']:.6f} | {value['bce']:.6f} | "
                         f"{value['fit_steps']} / {value['selected_step']} | {value['valid']} |")
    for name, value in result.get('saved_classifier_controls', {}).items():
        lines.append(f"| saved {name} | {value['auc']:.6f} | {value['bce']:.6f} | "
                     f"{value['fit_steps']} / {value['selected_step']} | {value['valid']} |")
    lines.extend(['', '## Matched contrasts', ''])
    for pair, values in result.get('contrasts', {}).items():
        gap, bce = values['auc_gap'], values['bce']
        lines.append(f"- {pair}: gap Δ {gap['delta']:+.6f} [{gap['lo95']:+.6f}, {gap['hi95']:+.6f}]; "
                     f"BCE Δ {bce['delta']:+.6f} [{bce['lo95']:+.6f}, {bce['hi95']:+.6f}].")
    if 'width_comparison' in result:
        comparison = result['width_comparison']
        lines.extend(['', '## Width x Fourier comparison', '',
                      'Decision: '+comparison['decision'], comparison['limitation'], ''])
        for name, value in comparison['bce_contrasts'].items():
            lines.append(f"- {name}: {value['mean']:+.6f} [{value['lo95']:+.6f}, {value['hi95']:+.6f}].")
    lines.extend(['', '## Decision boundary', '', plan['decision_rule'], '',
        'Unresolved if any required fit lacks its validation plateau. Near-chance undertraining is not closure.',
        'Exploratory bootstrap intervals condition on the fitted classifiers; they do not include training-seed uncertainty.',
        'No claims transfer automatically to EveNet. Summarize and approve the next one-question plan before launching it.',
        'User requests matched single-seed exploration, not seed replication; training-seed robustness remains unmeasured.', ''])
    return '\n'.join(lines)


def validate_plan(plan):
    required = ['round', 'kind', 'question', 'primary_endpoint', 'decision_rule', 'classifier_seed',
                'panel_seed', 'max_fit_steps', 'bootstrap_repeats', 'run_name']
    for key in required:
        if key not in plan:
            raise ValueError(f'Missing preregistration: {key}')
    if type(plan['round']) is not int or not 1 <= plan['round'] <= 20:
        raise ValueError('Hard cap: twenty rounds')
    if plan['max_fit_steps'] < 2000 or plan['max_fit_steps'] > 64000:
        raise ValueError('Fit budget must be 2000..64000; nonplateau is inconclusive')
    if not 2000 <= plan.get('minimum_fit_steps', 2000) <= plan['max_fit_steps']:
        raise ValueError('Minimum fit budget must be >=2000 and no greater than maximum')
    if len(plan['run_name']) >= 96 or not plan['question'] or plan['kind'] not in ('classifier', 'rl'):
        raise ValueError('Invalid round metadata')
    if plan['bootstrap_repeats'] < 100:
        raise ValueError('At least 100 bootstrap repetitions')
    if plan['kind'] == 'classifier':
        if type(plan.get('classifier_width', 128)) is not int or plan.get('classifier_width', 128) < 1:
            raise ValueError('Classifier width must be a positive integer')
        if not set(plan['features']).issubset(FEATURES) or not plan['features'] or not plan['datasets']:
            raise ValueError('Invalid classifier arms')
    else:
        film.validate_milestones(plan['milestones'])
        if plan['milestones'][-1] > 3000:
            raise ValueError('This loop caps each policy trajectory at 3000 updates')


def run(plan_path, output, wandb_mode='offline'):
    output = toy_path(output)
    plan = json.loads(Path(plan_path).read_text())
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'process.json', {'pid': os.getpid(), 'start_time': time.time()})
    start, wb = time.monotonic(), None
    status = {'state': 'running', 'round': plan['round'], 'question': plan['question']}
    atomic_json(output/'status.json', status)
    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        status.update(elapsed_seconds=time.monotonic()-start,
                      active={k: row[k] for k in ('phase', 'arm', 'step', 'policy_step') if k in row})
        atomic_json(output/'status.json', status)
        if wb is not None:
            from .coverage_budget import flatten_metrics
            prefix = f"{row['phase']}/p{row.get('policy_step', 0)}/{row.get('arm', 'all')}"
            wb.log(flatten_metrics(row, prefix+'/'))
        print(json.dumps(row, allow_nan=False), flush=True)
    try:
        if wandb_mode != 'disabled':
            # Keep all SDK state local to this explicitly authorized toy output.
            for key in ('WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
                os.environ[key] = str(output/'wandb_state'/key.lower())
                Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
            import wandb
            wb = wandb.init(project='dgpo-toy', group='Cube conditioning closed loop', mode=wandb_mode,
                name=plan['run_name'], dir=str(output), config=plan,
                tags=['toy-only', 'fresh-classifier', f"round-{plan['round']}"])
        result = (classifier_round if plan['kind'] == 'classifier' else rl_round)(plan, output, emit)
        result.update(elapsed_seconds=time.monotonic()-start, plan=plan)
        atomic_json(output/'report.json', result)
        (output/'SUMMARY.md').write_text(summarize(plan, result))
        status.update(state='awaiting_review', result_state=result['state'], elapsed_seconds=result['elapsed_seconds'])
        if wb is not None:
            wb.summary.update({'result_state': result['state'], 'round': plan['round']})
        return result
    except BaseException as exc:
        status.update(state='failed_or_interrupted', error=repr(exc), elapsed_seconds=time.monotonic()-start)
        raise
    finally:
        atomic_json(output/'status.json', status)
        if wb is not None:
            wb.finish(exit_code=1 if status['state'] == 'failed_or_interrupted' else 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--wandb-mode', choices=['offline', 'disabled'], default='offline')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    if args.threads < 1 or args.threads > 4:
        p.error('Local CPU budget: one to four threads')
    torch.set_num_threads(args.threads)
    if args.preflight:
        validate_plan(json.loads(args.plan.read_text()))
        toy_path(args.output)
        print('Plan valid; no training started.')
    else:
        run(args.plan, args.output, args.wandb_mode)
