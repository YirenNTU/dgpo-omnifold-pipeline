"""Reward-blind functional-radius contraction of saved toy policies; no RL updates."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

import torch

from . import closed_loop_lab as lab
from . import conditional as native
from . import reward_shape_failure as previous
from .nonperiodic_cube import Critic, Data, panel
from .reward_transfer_comparison import assert_same_panel, paired_intervals, score
from .truth_pretrain import atomic_checkpoint, atomic_json


def interpolate_state(initial, endpoint, fraction):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or initial.keys() != endpoint.keys():
        raise ValueError('Invalid contraction or state schema')
    mixed = {}
    for k,a in initial.items():
        b = endpoint[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('State shape/dtype mismatch')
        if not a.is_floating_point():
            if not torch.equal(a,b):raise ValueError('Nonfloating buffer changed')
            mixed[k] = a.clone()
        elif fraction == 0:
            mixed[k] = a.clone()
        elif fraction == 1:
            mixed[k] = b.clone()
        else:
            mixed[k] = a+(b-a)*fraction
    return mixed


def solve_radius(distance, target, *, grid=17, iterations=24, relative_tolerance=.002):
    """Select the first source-to-endpoint crossing, without any reward access."""
    if target <= 0 or not math.isfinite(target) or grid < 3:
        raise ValueError('Invalid matching target/grid')
    scan = [{'fraction':i/(grid-1),'distance':float(distance(i/(grid-1)))} for i in range(grid)]
    if not all(math.isfinite(p['distance']) and p['distance'] >= 0 for p in scan):
        raise ValueError('Invalid probe distance')
    if scan[0]['distance'] > 1e-12 or scan[-1]['distance'] < target:
        raise ValueError('Target not bracketed by source and endpoint')
    right = next(i for i,p in enumerate(scan) if p['distance'] >= target)
    if right == 0:raise ValueError('Nonzero source distance')
    lo,hi = scan[right-1]['fraction'],scan[right]['fraction']
    for _ in range(iterations):
        mid = (lo+hi)/2
        if distance(mid) < target:lo = mid
        else:hi = mid
    value = float(distance(hi))
    if abs(value/target-1) > relative_tolerance:
        raise ValueError('Radius matching failed')
    return {'fraction':hi,'distance':value,'target':target,'scan':scan,
            'scan_monotonic':all(b['distance']>=a['distance'] for a,b in zip(scan,scan[1:]))}


@torch.no_grad()
def make_probe(source, cfg, n, k, seed):
    rng = native.generator(seed)
    c = 2*torch.rand(n,1,generator=rng)-1
    z = torch.randn(n,k,3,generator=rng)
    y = torch.cat([native.ddim(source,cc[:,None],zz,cfg.ddim_steps)
                   for cc,zz in zip(c.split(128),z.split(128))])
    t = .7*torch.rand(n,k,generator=rng)
    eps = torch.randn(y.shape,generator=rng)
    a,s = native.alpha_sigma(t)
    x = a[...,None]*y+s[...,None]*eps
    x,t,c = x.flatten(0,1),t.flatten(),c[:,None].expand(-1,k,-1).flatten(0,1)
    v = torch.cat([source(xx,tt,cc) for xx,tt,cc in zip(x.split(1024),t.split(1024),c.split(1024))])
    return x,t,c,v


@torch.no_grad()
def velocity_distance(model, probe):
    x,t,c,ref = probe
    v = torch.cat([model(xx,tt,cc) for xx,tt,cc in zip(x.split(1024),t.split(1024),c.split(1024))])
    return float((v-ref).double().square().mean())


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if plan['kind'] != 'matched_velocity_replay' or not 1 <= plan['round'] <= 20:
        raise ValueError('Invalid replay plan')
    output = lab.toy_path(output)
    output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'process.json',{'pid':os.getpid(),'start_time':time.time()})
    def emit(row):
        print(json.dumps(row),flush=True)
        atomic_json(output/'status.json',{'state':'running','round':plan['round'],'active':row})
    try:
        records = {}
        for name,spec in plan['sources'].items():
            folder = lab.toy_path(spec['directory'])
            records[name] = {'folder':folder,'arm':spec['arm'],
                'plan':json.loads((folder/'plan.json').read_text()),
                'report':json.loads((folder/'report.json').read_text())}
        full,shift = records['full'],records['shift']
        lab.validate_external_control(shift['plan'])
        for key in ('config','reward','velocity_coefficient'):
            if full['report'][key] != shift['report'][key]:raise ValueError('Unmatched '+key)
        cfg,source = lab.load_source()
        cfg = native.Config(**full['report']['config'])
        data = Data(cfg)
        models,initials = {},{}
        for name,r in records.items():
            with torch.random.fork_rng():
                torch.manual_seed(r['plan']['initialization_seed'])
                model = lab.ConditioningAblation(cfg,source.state_dict(),**r['plan']['conditioning'][r['arm']]).eval()
            initials[name] = copy.deepcopy(model.state_dict())
            lab.film.previous.verify_matching(source,{name:model},cfg)
            ck = torch.load(r['folder']/f"{r['arm']}_step1000.pt",map_location='cpu',weights_only=True)
            if ck['step'] != 1000 or ck['velocity_coefficient'] != 1. or ck['conditioning'] != r['plan']['conditioning'][r['arm']]:
                raise ValueError('Wrong endpoint checkpoint')
            model.load_state_dict(ck['model'])
            models[name] = model
        full_endpoint = copy.deepcopy(models['full'].state_dict())
        emit({'phase':'velocity_calibration'})
        cp = make_probe(source,cfg,plan['probe_contexts'],plan['probe_candidates'],plan['calibration_seed'])
        vp = make_probe(source,cfg,plan['probe_contexts'],plan['probe_candidates'],plan['validation_seed'])
        original = {n:velocity_distance(m,cp) for n,m in models.items()}
        if original['full'] <= original['shift']:
            raise ValueError('Full endpoint not farther on declared calibration probe; contraction premise fails')
        def distance(f):
            models['full'].load_state_dict(interpolate_state(initials['full'],full_endpoint,f))
            return velocity_distance(models['full'],cp)
        match = solve_radius(distance,original['shift'])
        models['full'].load_state_dict(interpolate_state(initials['full'],full_endpoint,match['fraction']))
        validation = {n:velocity_distance(m,vp) for n,m in models.items()}
        ratio = validation['full']/validation['shift']
        radius_valid = abs(ratio-1) <= plan['validation_relative_tolerance']
        radius = {'original_calibration':original,'matching':match,'validation':validation,
                  'validation_ratio':ratio,'valid':radius_valid,'reward_used_for_selection':False}
        atomic_json(output/'radius.json',radius)
        emit({'phase':'radius_selected','fraction':match['fraction'],'validation_ratio':ratio,'valid':radius_valid})
        if not radius_valid:
            result={'state':'inconclusive_radius_validation','valid':False,'radius':radius,
                    'elapsed_seconds':time.monotonic()-start}
            atomic_json(output/'report.json',result)
            atomic_json(output/'status.json',{'state':'awaiting_review','round':plan['round'],**result})
            return result
        reward_path = lab.HISTORY/'rewards/full_reward.pt'
        if full['report']['reward'] != str(reward_path):raise ValueError('Wrong frozen reward')
        reward = Critic(True).eval().requires_grad_(False)
        reward.load_state_dict(torch.load(reward_path,map_location='cpu',weights_only=True)['model'])
        seed = full['plan']['panel_seed']
        saved_base = torch.load(full['folder']/'baseline_step0_panels.pt',map_location='cpu',weights_only=True)
        baseline_test = panel(data,16384,seed+30000,source)
        assert_same_panel(baseline_test,saved_base['test'],include_negative=True)
        shift_panels = torch.load(shift['folder']/f"{shift['arm']}_step1000_panels.pt",map_location='cpu',weights_only=True)
        replay = panel(data,16384,seed+30000,models['shift'])
        assert_same_panel(replay,shift_panels['test'],include_negative=True)
        emit({'phase':'generate_matched_panels'})
        pp = lab.generate_panels(models['full'],data,seed)
        lab.assert_pairing({'baseline':saved_base,'matched_full':pp,'shift':shift_panels})
        atomic_checkpoint(output/'matched_full_panels.pt',pp)
        base_score = score(reward,baseline_test)
        scores = {'full':score(reward,pp['test']),'shift':score(reward,replay)}
        contrasts = paired_intervals({'full_minus_shift':scores['full']-scores['shift'],
                    'full_minus_source':scores['full']-base_score,'shift_minus_source':scores['shift']-base_score},
                    plan['bootstrap_repeats'],plan['bootstrap_seed'])
        grid_base = torch.load(full['folder']/'baseline_endpoint.pt',map_location='cpu',weights_only=True)
        metrics,grid = previous.measure(models['full'],{'full':reward},data,grid=64,k=512)
        atomic_checkpoint(output/'matched_full_grid.pt',grid)
        atomic_checkpoint(output/'matched_full_policy.pt',{'model':models['full'].state_dict(),
                          'conditioning':full['plan']['conditioning'][full['arm']],
                          'fraction':match['fraction'],'parent_step':1000,'policy_updates':0})
        diagnostic = {'metrics':metrics,'reward_decomposition':previous.decompose(grid,grid_base,'full')}
        atomic_json(output/'reward_progress.json',{'contrasts':contrasts,'radius':radius,'diagnostic':diagnostic})
        # One new adequately trained audit; saved shift audit already fits its exact unchanged samples.
        audit,audit_scores = lab.fit_classifiers({'matched_full':pp},['both'],output/'audit',emit,
            seed=full['plan']['classifier_seed'],min_steps=plan['minimum_fit_steps'],max_steps=plan['max_fit_steps'])
        saved_scores = torch.load(shift['folder']/'test_scores.pt',map_location='cpu',weights_only=True)
        if not shift['report']['audits']['1000']['test'][shift['arm']+'__both']['valid']:
            raise ValueError('Invalid saved control audit')
        audit_contrast = lab.compare_scores({'matched_full':audit_scores['matched_full__both'],
                          'shift':saved_scores[shift['arm']+'__both_step1000']},
                          [['matched_full','shift']],repeats=plan['bootstrap_repeats'])
        primary = contrasts['full_minus_shift']
        decision = ('supports_advantage_at_matched_distance' if primary['lo95']>plan['material_reward_margin'] else
                    'opposes_advantage' if primary['hi95'] < -plan['material_reward_margin'] else 'unresolved')
        report = {'state':'completed','round':plan['round'],'valid':True,'radius':radius,
                  'contrasts':contrasts,'decision':decision,'diagnostic':diagnostic,
                  'audit':audit,'audit_contrast':audit_contrast,'audit_valid':audit['state']=='completed',
                  'saved_control_audit_reused':True,'baseline_and_control_samples_exact':True,
                  'elapsed_seconds':time.monotonic()-start,'scope':plan['scope']}
        atomic_json(output/'report.json',report)
        lines = ['# Round14: reward at matched reference-velocity distance','',
                 f"Decision: {decision}. Full contraction fraction {match['fraction']:.6f}.",
                 f"Independent validation distance ratio: {ratio:.6f}.",
                 'No RL updates. Radius selected without rewards; original source reference unchanged.','']
        for n,v in contrasts.items():lines.append(f"- {n}: {v['mean']:+.6f} [{v['lo95']:+.6f}, {v['hi95']:+.6f}].")
        lines += ['',f"Fresh matched-full audit valid: {report['audit_valid']}; saved unchanged shift audit reused.",
                  'This is a radial endpoint counterfactual, not a trained matched-radius optimizer or exact KL.',
                  'Matched velocity MSE is only a reference-probe norm, not equal endpoint-distribution distance.',
                  f"Elapsed {report['elapsed_seconds']:.1f}s.",'']
        (output/'SUMMARY.md').write_text('\n'.join(lines))
        atomic_json(output/'status.json',{'state':'awaiting_review','round':plan['round'],'result_state':'completed',
                                         'elapsed_seconds':report['elapsed_seconds']})
        return report
    except Exception as exc:
        atomic_json(output/'status.json',{'state':'failed','round':plan['round'],'error':repr(exc)})
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);run(args.plan,args.output)
