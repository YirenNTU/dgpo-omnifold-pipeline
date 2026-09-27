"""Read-only toy score comparison: fixed global gain versus saved LayerNorm route."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from .closed_loop_lab import assert_pairing, compare_scores, toy_path
from .truth_pretrain import atomic_json


def within_margins(contrast,margins):
    return all(contrast[k]['lo95'] >= -v and contrast[k]['hi95'] <= v for k,v in margins.items())


def run(plan_path,output):
    plan=json.loads(plan_path.read_text())
    if plan['kind']!='gain_norm_comparison' or not 1<=plan['round']<=20:
        raise ValueError('Invalid toy comparison plan')
    output=toy_path(output);output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'process.json',{'pid':os.getpid(),'start_time':time.time()})
    start=time.monotonic()
    records={}
    try:
        for name,spec in plan['sources'].items():
            folder=toy_path(spec['directory'])
            r=json.loads((folder/'report.json').read_text());p=json.loads((folder/'plan.json').read_text())
            if r['state']!='completed' or not r['valid']:
                raise ValueError('Incomplete source round')
            if not r['audits']['1000']['test'][spec['arm']+'__both']['valid']:
                raise ValueError('Invalid endpoint fresh fit')
            records[name]={'folder':folder,'report':r,'plan':p,'arm':spec['arm'],
                'scores':torch.load(folder/'test_scores.pt',map_location='cpu',weights_only=True)}
        gain,ln=records['gain'],records['layernorm']
        for k in ('initialization_seed','policy_seed','monitor_seed','panel_seed','classifier_seed',
                  'audit_features','minimum_fit_steps','max_fit_steps','milestones'):
            if gain['plan'][k]!=ln['plan'][k]:raise ValueError('Unmatched setting:'+k)
        for k in ('config','reward','velocity_coefficient'):
            if gain['report'][k]!=ln['report'][k]:raise ValueError('Unmatched objective/config:'+k)
        for label in ('positive','negative'):
            if not torch.equal(gain['scores']['baseline__both_step0'][label],ln['scores']['baseline__both_step0'][label]):
                raise ValueError('Source-audit replay not exact')
        specs=[{'encoder_gain':1.,**r['plan']['conditioning'][r['arm']]} for r in (gain,ln)]
        changed={k for k in specs[0] if specs[0][k]!=specs[1][k]}
        if changed!={'normalization','encoder_gain'}:raise ValueError('Unexpected architecture differences')
        for step in gain['plan']['milestones']:
            pp={n:torch.load(v['folder']/f"{v['arm']}_step{step}_panels.pt",map_location='cpu',weights_only=True) for n,v in records.items()}
            assert_pairing(pp)
        primary_scores={n:v['scores'][v['arm']+'__both_step1000'] for n,v in records.items()}
        contrast=compare_scores(primary_scores,[['gain','layernorm']],repeats=plan['bootstrap_repeats'])['gain minus layernorm']
        result={'state':'completed','round':plan['round'],'contrast':contrast,
                'within_declared_metric_margins':within_margins(contrast,plan['equivalence_margins']),
                'margins':plan['equivalence_margins'],'baseline_predictions_exact':True,'all_panels_paired':True,
                'changed_components':sorted(changed),'elapsed_seconds':time.monotonic()-start,'scope':plan['scope']}
        atomic_json(output/'report.json',result)
        lines=['# Fixed gain versus LayerNorm: matched saved-score analysis','',
               f"Both metric intervals within declared margins: {result['within_declared_metric_margins']}",'']
        for k,v in contrast.items():lines.append(f"- {k}: delta {v['delta']:+.6f},95% interval [{v['lo95']:+.6f},{v['hi95']:+.6f}].")
        lines+=['','Only compares audit metrics conditional on fitted models, not distribution equivalence or seed robustness.',
                'Endpoint point estimates were already observed before this adaptive comparison; no new training occurred.','']
        (output/'SUMMARY.md').write_text('\n'.join(lines))
        atomic_json(output/'status.json',{'state':'awaiting_review','round':plan['round'],'elapsed_seconds':result['elapsed_seconds']})
        return result
    except Exception as exc:
        atomic_json(output/'status.json',{'state':'failed','error':repr(exc)})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(1);run(args.plan,args.output)
