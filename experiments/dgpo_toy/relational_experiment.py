"""Toy-only, human-reviewed relation-access classifier -> fixed-reward DGPO.

No auto-launch of stage 2, remote execution, reward transformation or seed search.
See RELATIONAL_CONDITIONING.md. Preflight does not fit or update any model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import time

import torch

from . import closed_loop_lab as lab
from . import conditional as native
from .classifier_signal_attribution import bce_by_condition, simultaneous_mean_intervals, walsh
from .cube_swap import modes
from .nonperiodic_cube import Data, panel
from .relational_conditioning import (REPRESENTATIONS, RelationalCritic, RelationalData,
    RelationalPolicy, VisibleSource, decode_condition, encode_visible, lift_panels,
    rotate_visible, verify_initial)
from .reward_transport import generate, cells
from .truth_pretrain import atomic_checkpoint, atomic_json
from .width_signal_attribution import replay_panels

ROOT = lab.ROOT


def artifact(path):
    return lab.toy_path(ROOT / path)


def validate_plan(plan):
    if plan.get('schema') != 'relational_cube_v1' or plan.get('kind') not in ('classifier', 'rl'):
        raise ValueError('Expected the prespecified relational cube plan')
    for field in ('question', 'primary_endpoint', 'decision_rule', 'scope', 'run_name'):
        if not plan.get(field):
            raise ValueError('Missing '+field)
    if len(plan['run_name']) >= 96 or len(plan['run_name'].split(' | ')) != 3:
        raise ValueError('Use a short three-part readable run name')
    if plan.get('wandb_mode', 'disabled') not in ('offline', 'disabled'):
        raise ValueError('This local study supports offline or disabled W&B only')
    fit = plan['fit']
    if not 2000 <= fit['min_steps'] <= fit['max_steps'] <= 64000:
        raise ValueError('Adequate classifier budget required (2000..64000)')
    if fit['width'] < 2 or fit['check_every'] < 1 or fit['patience'] < 1 or fit['min_delta'] <= 0:
        raise ValueError('Invalid classifier fit settings')
    if fit['min_steps'] < 32000 or fit['max_steps'] != 64000:
        raise ValueError('This relation-access study uses the declared32k..64k adequacy budget')
    if plan['bootstrap_repeats'] < 100:
        raise ValueError('At least 100 bootstrap draws')
    if plan['kind'] == 'classifier':
        if fit['width'] != 32:
            raise ValueError('Keep the matched width32 classifier, not a capacity search')
        if plan['representations'] != list(REPRESENTATIONS) or plan['material_bce_margin'] != .002:
            raise ValueError('Preserve three classifier arms and BCE margin .002')
        artifact(plan['panels']); artifact(plan['swap_directory'])
    else:
        if fit['width'] != 128:
            raise ValueError('Keep width128 fresh audit classifiers')
        if plan['milestones'] != [25, 100, 300, 1000] or plan['velocity_coefficient'] != 1.:
            raise ValueError('Preserve fixed coefficient1 and the 1000-update protocol')
        if plan['reward_gain_margin'] != .01:
            raise ValueError('Preserve fixed-reward material margin .01')
        if plan['eval_contexts'] < 512 or plan['eval_candidates'] != 8:
            raise ValueError('Use enough independent evaluation contexts and all K8 candidates')
        if plan['structure_grid'] < 16 or plan['structure_candidates'] < 128:
            raise ValueError('Insufficient conditional-structure panel')
        artifact(plan['classifier_result'])


def resolved_setup(plan, cfg, wandb_mode, representation=None):
    """Expose the actual shared-runner constants, separately from source config."""
    return {
        'kind': plan['kind'], 'source_checkpoint': str(lab.SOURCE),
        'source_pretraining': 'uniform cube reference, not full truth',
        'source_frozen': True, 'raw_weights_not_ema': True,
        'learned_condition': 'two visible unit vectors; never scalar c or truth g(c)',
        'classifier': {**plan['fit'], 'optimizer': 'AdamW', 'lr': 3e-4,
            'weight_decay': .001, 'clip_norm': 1., 'batch_examples': 512,
            'paired_contexts_per_batch': 256, 'hidden_layers': 3,
            'activation': 'GELU', 'output_fourier_k': [1, 2, 3, 4],
            'selection': 'exact minimum validation BCE; test never selects weights'},
        'policy': None if plan['kind'] == 'classifier' else {
            'arms': ['raw', representation], 'optimizer': 'AdamW',
            'lr': cfg.policy_lr, 'weight_decay': cfg.weight_decay, 'clip_norm': 1.,
            'schedule': 'constant', 'updates': 1000, 'batch_contexts': cfg.batch,
            'candidates': cfg.candidates, 'timesteps': cfg.timesteps,
            'ddim_steps': cfg.ddim_steps, 'film_blocks': 3,
            'encoder': '20->32->32; two SiLU; LayerNorm',
            'reference_penalty': '1 * 0.5 * mean((v - v_reference)^2)',
            'same_frozen_reward_for_both_arms': True, 'refit': False,
            'audit_steps': [0, *plan['milestones']]},
        'wandb': {'mode': wandb_mode, 'name': plan['run_name'],
                  'group': 'Conditional relation access'},
        'next_stage_automatic': False,
    }


def checked_lift(original, seed):
    """Check finite paired data before any expensive classifier fit."""
    if set(original) != {'train', 'validation', 'test'}:
        raise ValueError('Expected the archived train/validation/test panels')
    for split, p in original.items():
        n = len(p['c'])
        for key, width in (('c', 1), ('positive', 3), ('negative', 3)):
            if p[key].shape != (n, width) or not torch.isfinite(p[key]).all():
                raise ValueError(f'Invalid shape or nonfinite data: {split}/{key}')
        if not n or (p['c'].abs() > 1).any():
            raise ValueError('Empty panel or condition outside the archived domain')
    lifted = lift_panels(original, seed)
    for split, p in lifted.items():
        if (decode_condition(p['c']) - original[split]['c']).abs().max() > 1e-5:
            raise ValueError('Visible encoding lost the archived condition')
        if any(not torch.equal(p[k], original[split][k]) for k in ('positive', 'negative')):
            raise ValueError('Condition encoding must not change either class sample')
    return lifted


def fit(panels, representations, output, emit, plan):
    settings = dict(plan['fit'])
    seed = settings.pop('seed')
    return lab.fit_classifiers(panels, representations, output, emit, seed=seed,
        model_factory=RelationalCritic, **settings)


def load_critic(directory, key):
    record = json.loads((directory/'fit_report.json').read_text())
    if record.get('model_type') != 'RelationalCritic':
        raise ValueError('Wrong classifier architecture')
    ck = torch.load(directory/f'{key}_best.pt', map_location='cpu', weights_only=True)
    if ck['feature'] != key.rsplit('__',1)[-1] or ck['width'] != record['width']:
        raise ValueError('Selected classifier checkpoint branch/width mismatch')
    model = RelationalCritic(ck['feature'], ck['width']).eval().requires_grad_(False)
    model.load_state_dict(ck['model'], strict=True)
    return model, record['test'][key]


@torch.no_grad()
def rotation_diagnostic(model, p):
    original = lab.predictions(model, p)
    result = {}
    for angle in (math.pi/7, math.pi/2, math.pi):
        rotated = lab.predictions(model, {**p, 'c': rotate_visible(p['c'], angle)})
        delta = torch.cat([rotated[s]-original[s] for s in ('positive','negative')])
        result[f'{angle:.6f}'] = {'logit_difference_rms': float(delta.square().mean().sqrt()),
            'logit_difference_max': float(delta.abs().max()),
            'bce_change': lab.score_metrics(rotated)['bce']-lab.score_metrics(original)['bce']}
    return result


def classify_decision(fits, contrasts, content, margin):
    adequate = all(v['valid'] for v in fits['test'].values())
    benefit, mode_benefit = {}, {}
    for rep in ('particle', 'relative'):
        key = f'source__{rep} minus source__raw'
        benefit[rep] = (adequate and contrasts[key]['bce']['hi95'] < -margin
                        and contrasts[key]['auc']['delta'] > 0)
        mode = content['contrasts'][f'{rep}_minus_raw/B_minus_A']
        mode_benefit[rep] = (adequate and mode['hi95'] < -margin and
            content['judges'][rep]['B']['auc'] > content['judges']['raw']['B']['auc'])
    relative_pair = contrasts.get('source__relative minus source__particle', {})
    relative_advantage = (adequate and benefit['relative'] and
        relative_pair['bce']['hi95'] < -margin and relative_pair['auc']['delta'] > 0)
    mode_pair = content['contrasts'].get('relative_minus_particle/B_minus_A', {})
    relative_mode_advantage = (adequate and mode_pair.get('hi95', math.inf) < -margin and
        content['judges']['relative']['B']['auc'] > content['judges']['particle']['B']['auc'])
    return {'adequate': adequate, 'material_bce_benefit': benefit, 'mode_benefit': mode_benefit,
        'relative_vs_particle_bce_benefit': relative_advantage,
        'relative_vs_particle_mode_benefit': relative_mode_advantage,
        'eligible_for_rl': {r: benefit[r] and mode_benefit[r] for r in benefit},
        'interpretation': ('unresolved_fit_budget' if not adequate else
            'supports_relative_joint_access_advantage' if relative_advantage and relative_mode_advantage else
            'supports_relative_bce_advantage_content_unresolved' if relative_advantage
            else 'supports_particle_basis_access' if benefit['particle'] else 'unresolved_relation_access_advantage'),
        'next_action': 'Review SUMMARY.md before explicitly launching any RL; never select a representation automatically.'}


def content_diagnostic(plan, models, output):
    directory = artifact(plan['swap_directory'])
    old = json.loads((directory/'plan.json').read_text())
    if json.loads((directory/'report.json').read_text())['state'] != 'completed':
        raise ValueError('Saved swap experiment incomplete')
    pool = torch.load(directory/'step0_pool.pt', map_location='cpu', weights_only=True)
    cfg, _ = lab.load_source()
    data = Data(cfg)
    if not torch.equal(pool['p'], data.probabilities(pool['c'][:,0], .9)):
        raise ValueError('Diagnostic truth changed')
    panels = replay_panels(pool, data.centers, old['samples_per_condition'], old['sample_seed'])
    c = panels['A']['c']
    rotation = (2*torch.rand(c.shape, generator=native.generator(plan['rotation_seed']+100))-1)*math.pi
    visible = encode_visible(c, rotation)
    judges, arrays, series = {}, {}, {}
    for rep, model in models.items():
        scores = {name: lab.predictions(model, {**pp,'c':visible}) for name, pp in panels.items()}
        judges[rep] = {name: lab.score_metrics(ss) for name, ss in scores.items()}
        arrays[rep] = {name: bce_by_condition(ss, len(pool['c'])) for name, ss in scores.items()}
    for rep, control in (('particle', 'raw'), ('relative', 'raw'), ('relative', 'particle')):
        for label in ('B','C','D'):
            series[f'{rep}_minus_{control}/{label}_minus_A'] = (
                arrays[rep][label]-arrays[rep]['A']-arrays[control][label]+arrays[control]['A'])
    atomic_checkpoint(output/'content_arrays.pt', arrays)
    return {'judges':judges, 'contrasts':simultaneous_mean_intervals(series, plan['bootstrap_repeats'], 920041),
        'labels': {'A':'truth modes/truth shape', 'B':'generated modes/truth shape',
                   'C':'truth modes/generated shape', 'D':'unchanged generator'},
        'scope':'Frozen-judge content probe, not retrained on swaps; intervals resample fixed grid nodes and are descriptive.'}


def classifier_stage(plan, output, emit):
    original = torch.load(artifact(plan['panels']), map_location='cpu', weights_only=True)
    panels = checked_lift(original, plan['rotation_seed'])
    atomic_checkpoint(output/'panels.pt', panels)
    fits, scores = fit({'source':panels}, REPRESENTATIONS, output/'classifiers', emit, plan)
    pairs = [('source__particle','source__raw'),('source__relative','source__raw'),
             ('source__relative','source__particle')]
    contrasts = lab.compare_scores(scores, pairs, repeats=plan['bootstrap_repeats'])
    models = {r:load_critic(output/'classifiers','source__'+r)[0] for r in REPRESENTATIONS}
    content = content_diagnostic(plan, models, output)
    decision = classify_decision(fits, contrasts, content, plan['material_bce_margin'])
    return {'state':'completed' if decision['adequate'] else 'inconclusive_fit_budget',
        'source_checkpoint':str(lab.SOURCE), 'archived_panels':str(artifact(plan['panels'])),
        'fit':fits, 'contrasts':contrasts, 'content':content, 'decision':decision,
        'rotation':{r:rotation_diagnostic(m,panels['test']) for r,m in models.items()},
        'paired_original_y_unchanged':all(torch.equal(original[s][k],panels[s][k])
            for s in original for k in ('positive','negative')),
        'source_roundtrip_c_max':max(float((decode_condition(p['c'])-p['scalar_c']).abs().max()) for p in panels.values()),
        'policy_updates':0}


def load_reward(plan, representation):
    directory = artifact(plan['classifier_result'])
    report = json.loads((directory/'report.json').read_text())
    saved_plan = json.loads((directory/'plan.json').read_text())
    validate_plan(saved_plan)
    if (saved_plan['kind'] != 'classifier' or report['state'] != 'completed'
            or not report['decision']['eligible_for_rl'].get(representation, False)):
        raise ValueError('Stage 1 does not yet support this relation representation; review, do not launch RL')
    model, fit_info = load_critic(directory/'classifiers', 'source__'+representation)
    if not fit_info['valid'] or fit_info['bce'] >= math.log(2)-.002:
        raise ValueError('Selected fixed reward is not an adequate useful classifier')
    provenance = {'directory':str(directory), 'representation':representation, 'fit':fit_info,
                  'selection':'Explicit reviewer choice after stage-1 BCE and mode-content gates; no automatic test-score winner.'}
    return model, provenance


def make_policies(source, cfg, representation, seed):
    result = {}
    for rep in ('raw', representation):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            result[rep] = RelationalPolicy(source, cfg, rep).eval()
    return result


def audit_panels(model, data, seed):
    return {split:panel(data,n,seed+offset,model) for split,n,offset in
            [('train',32768,10000),('validation',8192,20000),('test',16384,30000)]}


@torch.no_grad()
def reward_panel(model, reward, data, cfg, plan):
    rng = native.generator(plan['evaluation_seed'])
    c = data.contexts(plan['eval_contexts'],rng)
    noise = torch.randn(len(c),plan['eval_candidates'],3,generator=rng)
    values = []
    for cc, zz in zip(c.split(64),noise.split(64)):
        yy = native.ddim(model,cc[:,None],zz,cfg.ddim_steps)
        values.append(reward(yy,cc[:,None]))
    values = torch.cat(values)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite held-out rewards')
    return values


@torch.no_grad()
def structure_panel(model, reward, data, cfg, plan):
    grid, k = plan['structure_grid'], plan['structure_candidates']
    scalar = ((torch.arange(grid)+.5)/grid*2-1)[:,None]
    rotation = (2*torch.rand(scalar.shape,generator=native.generator(plan['structure_seed']))-1)*math.pi
    visible = encode_visible(scalar,rotation)
    y = generate(model,visible,k,plan['structure_seed']+1,steps=cfg.ddim_steps)
    r = reward(y,visible[:,None]); stats = cells(y,r)
    q, p = stats['q'], data.base.probabilities(scalar[:,0],.9).double()
    ids = modes(y); h, orders = walsh(data.centers)
    halves = [torch.nn.functional.one_hot(ii,8).double().mean(1) for ii in ids.chunk(2,1)]
    cross = ((halves[0]-p)@h)*((halves[1]-p)@h)
    metrics = {'mode_tv':float(.5*(q-p).abs().sum(-1).mean()),
        'within_mode_rms':float((y-data.centers[ids]).square().sum(-1).mean().sqrt()),
        'missing_mode_cells':int((stats['counts']==0).sum())}
    metrics.update({f'order{o}_l2_debiased':float(cross[:,orders==o].sum(-1).mean()/8) for o in (1,2,3)})
    return metrics, {'q':q,'p':p,'rewards':r,'cells':stats,'visible':visible}


@torch.no_grad()
def policy_diagnostics(model, reference, data, cfg, seed):
    rng = native.generator(seed)
    c = data.contexts(512,rng)
    y = native.ddim(reference,c,torch.randn(512,3,generator=rng),cfg.ddim_steps)
    t = .7*torch.rand(4,512,generator=rng)
    eps = torch.randn(4,512,3,generator=rng)
    a,s = native.alpha_sigma(t[...,None]); x = a*y[None]+s*eps
    v = model(x,t,c[None]); ref = reference(x,t,c[None])
    _,_,branch = model.correction(x,t,c[None],diagnostics=True)
    return {'fixed_reference_velocity_mse':float((v-ref).square().mean()), 'branch':branch,
        'postclip_gradient_norm':{group:math.sqrt(sum(float(p.grad.square().sum())
            for name,p in model.named_parameters() if p.grad is not None and name.startswith(prefixes)))
            for group,prefixes in [('backbone',('network.',)),
                ('encoder',('condition_input.','condition_hidden.')),('heads',('condition_heads.',))]}}


def rl_stage(plan, output, emit, representation, review_note):
    if representation not in ('particle','relative') or not review_note.strip():
        raise ValueError('Explicit representation and stage-1 review note required')
    reward, provenance = load_reward(plan,representation)
    cfg, source = lab.load_source()
    cfg = replace(cfg,policy_steps=1000,eval_every=25,eval_events=512)
    data = RelationalData(Data(cfg))
    policies = make_policies(source,cfg,representation,plan['initialization_seed'])
    matching = verify_initial(source,cfg,policies)
    reference = VisibleSource(source)
    reward_state = {k:v.clone() for k,v in reward.state_dict().items()}
    for name,initial in policies.items():
        atomic_checkpoint(output/f'{name}_initial.pt',{'model':initial.state_dict(),'config':asdict(cfg),
            'step':0,'representation':name,'source_checkpoint':str(lab.SOURCE),
            'schema':'relational_cube_v1','velocity_coefficient':1.,'reward':provenance})
    atomic_checkpoint(output/'fixed_reward.pt',{'model':reward_state,'feature':representation,
        'width':reward.net[0].out_features,'provenance':provenance})
    result = {'state':'running','config':asdict(cfg),'reward':provenance,'review_note':review_note,
        'source_checkpoint':str(lab.SOURCE),
        'trainable_parameters':{name:sum(p.numel() for p in model.parameters()) for name,model in policies.items()},
        'initial_matching':matching,'points':{},'audits':{},'velocity_coefficient':1.,
        'reference_definition':'immutable archived source; penalty is HALF velocity MSE, not endpoint KL'}
    reward_arrays, fresh_scores, structure_payloads = {}, {}, {}
    for step in (0,*plan['milestones']):
        active = {'baseline':reference} if step == 0 else {}
        if step:
            for name, initial in policies.items():
                path = output/f'{name}_last.pt'
                resume = torch.load(path,map_location='cpu',weights_only=True) if path.exists() else None
                def checkpoint(s,m,opt,rng,history):
                    if s % 25 == 0 or s == step:
                        saved = {'model':m.state_dict(),'optimizer':opt.state_dict(),'rng':rng.get_state(),
                            'history':history,'step':s,'velocity_coefficient':1.,'config':asdict(cfg),
                            'representation':name,'schema':'relational_cube_v1','reward':provenance}
                        atomic_checkpoint(path,saved)
                        if s == step:
                            atomic_checkpoint(output/f'{name}_step{s}.pt',saved)
                if resume and (resume['representation'] != name or resume['reward'] != provenance):
                    raise ValueError('Resume state changed policy representation or fixed reward')
                current,_ = native.policy_train('dgpo',initial,reward,data,replace(cfg,policy_steps=step),
                    plan['policy_seed'],plan['monitor_seed'],lambda row:emit({**row,'arm':name}),
                    checkpoint,resume_state=resume,velocity_coefficient=1.)
                active[name] = current
        for name, model in active.items():
            key = f'{name}_{step}'
            rr = reward_panel(model,reward,data,cfg,plan)
            reward_arrays[key] = rr.double().mean(-1)
            structure_stats,payload = structure_panel(model,reward,data,cfg,plan)
            structure_payloads[key] = payload
            entry = {'reward_mean':float(rr.double().mean()),'reward_std':float(rr.std(unbiased=False)),
                'weight':native.weight_health(rr),'structure':structure_stats}
            if step:
                gain = reward_arrays[key]-reward_arrays['baseline_0']
                entry['reward_gain'] = float(gain.mean())
                entry['positive_context_fraction'] = float((gain>0).double().mean())
                entry['reward_gain_quantiles'] = {str(q):float(torch.quantile(gain,q)) for q in (.01,.5,.99)}
                before = structure_payloads['baseline_0']
                available = bool(before['cells']['mean_available'].all())
                entry['reward_content'] = {'available':available}
                if available:
                    mode = float(((payload['q']-before['q'])*before['cells']['mean']).sum(-1).mean())
                    total = float((payload['rewards']-before['rewards']).double().mean())
                    entry['reward_content'].update(mode_probability=mode,shape_and_interaction=total-mode,
                        scope='Frozen-source cell decomposition on fixed grid, not unique causal attribution')
                entry['conditioning'] = policy_diagnostics(model,reference,data,cfg,plan['probe_seed'])
            result['points'][key] = entry
            emit({'phase':'endpoint','arm':name,'step':step,**entry})
        pp = {name:audit_panels(model,data,plan['audit_panel_seed']) for name,model in active.items()}
        lab.assert_pairing(pp)
        for name,p in pp.items():
            atomic_checkpoint(output/f'{name}_step{step}_panels.pt',p)
        audit,scores = fit(pp,['relative'],output/f'audit{step}',
            lambda row:emit({**row,'policy_step':step}),plan)
        result['audits'][str(step)] = audit
        fresh_scores.update({f'{name}_step{step}':ss for name,ss in scores.items()})
        for key,value in audit['test'].items():
            emit({'phase':'fresh_validation','arm':key,'step':step,**value})
        atomic_checkpoint(output/'reward_arrays.pt',reward_arrays)
        atomic_checkpoint(output/'fresh_scores.pt',fresh_scores)
        atomic_checkpoint(output/'structure_arrays.pt',structure_payloads)
        atomic_json(output/'rl_progress.json',result)
    series = {f'{rep}_gain_1000':reward_arrays[f'{rep}_1000']-reward_arrays['baseline_0'] for rep in policies}
    series['enhanced_minus_raw_1000'] = reward_arrays[f'{representation}_1000']-reward_arrays['raw_1000']
    result['reward_contrasts'] = simultaneous_mean_intervals(series,plan['bootstrap_repeats'],910041)
    result['reward_uncertainty'] = 'Paired IID evaluation-context bootstrap; all K8 averaged within context; conditional on fitted models.'
    audit_pairs = [(f'{rep}__relative_step{s}','baseline__relative_step0') for s in plan['milestones'] for rep in policies]
    audit_pairs += [(f'{representation}__relative_step1000','raw__relative_step1000')]
    result['audit_contrasts'] = lab.compare_scores(fresh_scores,audit_pairs,repeats=plan['bootstrap_repeats'])
    result['audits_valid'] = all(v['valid'] for audit in result['audits'].values() for v in audit['test'].values())
    result['reward_unchanged'] = all(torch.equal(v,reward_state[k]) for k,v in reward.state_dict().items())
    if not result['reward_unchanged']:
        raise ValueError('Frozen classifier changed')
    interval = result['reward_contrasts']['enhanced_minus_raw_1000']
    result['decision'] = {'reward_transfer_supports':interval['lo95'] > plan['reward_gain_margin'],
        'fresh_audit_valid':result['audits_valid'],
        'interpretation':'supports_more_reward_transfer' if interval['lo95'] > plan['reward_gain_margin'] else 'unresolved_material_transfer_advantage',
        'scope':'Reward transfer does not imply joint improvement/closure; report structure and fresh BCE independently. Unequal reference motion is not controlled away.'}
    result['state'] = 'completed' if result['audits_valid'] else 'completed_with_inconclusive_audits'
    return result


def summarize(plan,result):
    lines = ['# '+plan['question'],'',f"State: {result['state']}",'',plan['primary_endpoint'],'']
    if plan['kind'] == 'classifier':
        lines += ['| Condition branch | Test BCE | AUC | Updates / selected | Adequate |',
                  '|---|---:|---:|---:|---|']
        for name,value in result['fit']['test'].items():
            lines.append(f"| {name} | {value['bce']:.6f} | {value['auc']:.6f} | {value['fit_steps']} / {value['selected_step']} | {value['valid']} |")
        lines += ['', '## Frozen-judge content (not new fits)', '',
                  '| Branch | Mode-only B AUC | Shape-only C AUC |', '|---|---:|---:|']
        for name,value in result['content']['judges'].items():
            lines.append(f"| {name} | {value['B']['auc']:.6f} | {value['C']['auc']:.6f} |")
        contrasts = {k:v['bce'] for k,v in result['contrasts'].items()}
    else:
        lines += ['| Policy | Mean fixed reward | Gain from source | Order3 error |', '|---|---:|---:|---:|']
        for name,value in result['points'].items():
            lines.append(f"| {name} | {value['reward_mean']:.6f} | {value.get('reward_gain',0):+.6f} | {value['structure']['order3_l2_debiased']:.6f} |")
        contrasts = result['reward_contrasts']
        lines += ['', '## Fresh held-out classifiers (not the fixed reward)', '',
                  '| Policy step | Policy | BCE | AUC | Fit / selected updates | Adequate |',
                  '|---:|---|---:|---:|---:|---|']
        for step, audit in result['audits'].items():
            for name, value in audit['test'].items():
                lines.append(f"| {step} | {name} | {value['bce']:.6f} | {value['auc']:.6f} | "
                    f"{value['fit_steps']} / {value['selected_step']} | {value['valid']} |")
    lines += ['', '## Paired primary contrasts','']
    for name,value in contrasts.items():
        point = value['delta'] if 'delta' in value else value['mean']
        lines.append(f"- {name}: {point:+.6f} [{value['lo95']:+.6f}, {value['hi95']:+.6f}].")
    lines += ['', '## Decision','', '```json',json.dumps(result['decision'],indent=2),'```','',
              'No next stage launched. Review this result before the next iteration.',
              'Single training seed, exploratory evidence; bootstrap is not seed replication.',
              plan['scope'],'']
    return '\n'.join(lines)


def preflight(plan,representation=None):
    validate_plan(plan)
    cfg,source = lab.load_source()
    expected = {'dimensions':3,'context_dim':1,'policy_lr':1e-4,'weight_decay':.001,
                'batch':64,'candidates':8,'timesteps':4,'ddim_steps':50}
    if any(getattr(cfg,key) != value for key,value in expected.items()):
        raise ValueError('Archived source training settings differ from the declared protocol')
    policies = make_policies(source,cfg,representation or 'relative',plan.get('initialization_seed',451017))
    if not representation:
        policies.update(make_policies(source,cfg,'particle',plan.get('initialization_seed',451017)))
    result = {'state':'preflight_passed_not_started','initial_matching':verify_initial(source,cfg,policies),
              'resolved_policy_settings':expected, 'training_started': False,
              'setup': resolved_setup(plan, cfg, plan.get('wandb_mode', 'disabled'), representation)}
    if plan['kind'] == 'classifier':
        original = torch.load(artifact(plan['panels']),map_location='cpu',weights_only=True)
        pp = checked_lift(original,plan['rotation_seed'])
        sizes = {s:len(p['c']) for s,p in pp.items()}
        if sizes != {'train':32768,'validation':8192,'test':16384}:
            raise ValueError('Source split sizes differ from the matched archived experiment')
        if not (artifact(plan['swap_directory'])/'step0_pool.pt').is_file():
            raise FileNotFoundError('Missing saved mode/shape diagnostic pool')
        result.update(panel_sizes=sizes,output_fourier_identical=True,training_started=False)
    else:
        if not representation:
            result.update(state='planned_awaiting_stage1_review',training_started=False)
        else:
            _,provenance = load_reward(plan,representation)
            result['reward'] = provenance
    return result


def run(plan,output,wandb_mode=None,representation=None,review_note=''):
    validate_plan(plan)
    checks = preflight(plan,representation)
    wandb_mode = wandb_mode or plan.get('wandb_mode', 'disabled')
    if wandb_mode not in ('offline', 'disabled'):
        raise ValueError('Expected offline or disabled logging')
    if plan['kind'] == 'rl' and (not representation or not review_note.strip()):
        raise ValueError('Review stage 1, then provide --representation and --review-note')
    output = artifact(output)
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',plan)
    checks['setup']['wandb']['mode'] = wandb_mode
    atomic_json(output/'preflight.json',checks)
    atomic_json(output/'runtime.json',checks['setup'])
    atomic_json(output/'process.json',{'pid':os.getpid(),'start_time':time.time()})
    start,wb = time.monotonic(),None
    status = {'state':'running','kind':plan['kind']}
    atomic_json(output/'status.json',status)
    defined_metrics = set()
    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        status.update(active={k:row[k] for k in ('phase','arm','step','policy_step') if k in row},
                      elapsed_seconds=time.monotonic()-start)
        atomic_json(output/'status.json',status)
        if wb is not None:
            from .coverage_budget import flatten_metrics
            prefix = f"{row['phase']}/p{row.get('policy_step',0)}/{row.get('arm','all')}"
            if prefix not in defined_metrics:
                wb.define_metric(prefix+'/step')
                wb.define_metric(prefix+'/*',step_metric=prefix+'/step')
                defined_metrics.add(prefix)
            wb.log(flatten_metrics(row,prefix+'/'))
        print(json.dumps(row,allow_nan=False),flush=True)
    try:
        print(json.dumps({'phase': 'setup', **checks['setup']}, allow_nan=False), flush=True)
        if wandb_mode == 'offline':
            for key in ('WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR'):
                os.environ[key] = str(output/'wandb_state'/key.lower())
                Path(os.environ[key]).mkdir(parents=True,exist_ok=True)
            import wandb
            wb = wandb.init(project='dgpo-toy',group='Conditional relation access',name=plan['run_name'],
                mode='offline',dir=str(output),config={**plan,'review_note':review_note,'representation':representation},
                tags=['toy-only','relation-access','raw-no-ema',plan['kind']])
        result = (classifier_stage(plan,output,emit) if plan['kind']=='classifier' else
                  rl_stage(plan,output,emit,representation,review_note))
        result['elapsed_seconds'] = time.monotonic()-start
        atomic_json(output/'report.json',result)
        (output/'SUMMARY.md').write_text(summarize(plan,result))
        # A plotting dependency/failure cannot erase a completed scientific result.
        try:
            from .plot_relational_experiment import plot
            result['plots'] = plot(output)
        except Exception as exc:
            result['plot_error'] = repr(exc)
        atomic_json(output/'report.json',result)
        status.update(state='awaiting_review',result_state=result['state'])
        if wb is not None:
            wb.summary.update({'result_state':result['state'],'decision':result['decision']})
        return result
    except BaseException as exc:
        status.update(state='failed_or_interrupted',error=repr(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic()-start
        atomic_json(output/'status.json',status)
        if wb is not None:
            wb.finish(exit_code=int(status['state']=='failed_or_interrupted'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--representation',choices=['particle','relative'])
    parser.add_argument('--review-note',default='')
    parser.add_argument('--wandb-mode',choices=['offline','disabled'],
                        help='Override plan logging (default: local files, W&B disabled)')
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = json.loads(args.plan.read_text())
    if args.preflight:
        result = preflight(plan,args.representation)
        if args.wandb_mode:
            result['setup']['wandb']['mode'] = args.wandb_mode
    else:
        result = run(plan,args.output or plan['output'],args.wandb_mode,args.representation,args.review_note)
    print(json.dumps({'state':result['state'],'decision':result.get('decision'),
                      **(result if args.preflight else {})},indent=2))


if __name__ == '__main__':
    main()
