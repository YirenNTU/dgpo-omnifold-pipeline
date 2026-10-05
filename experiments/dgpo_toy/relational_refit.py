"""Toy-only periodic cold reward/ref-source refresh with a fixed external judge.

The frozen-reward control is an exact archived trajectory, not a new seed. The
first interval must replay it bitwise before any cold classifier is installed.
This is a reward/reference-refresh analogue, not full two-stage OmniFold.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from . import closed_loop_lab as lab
from . import conditional as native
from . import relational_experiment as rel
from . import relational_retention as retention
from .classifier_signal_attribution import simultaneous_mean_intervals
from .nonperiodic_cube import Data
from .relational_conditioning import RelationalData
from .truth_pretrain import atomic_checkpoint, atomic_json


def state_at_boundary(model, optimizer, rng, history, reward_source, representation='raw'):
    return {'model':copy.deepcopy(model.state_dict()),
            'optimizer':copy.deepcopy(optimizer.state_dict()),
            'rng':rng.get_state().clone(), 'history':copy.deepcopy(history),
            'step':len(history), 'velocity_coefficient':1.,
            'reward_source':reward_source, 'representation':representation}


def assert_replay(current, archived):
    """A reused control needs exact weights, moments, RNG and step identity."""
    def equal(a,b):
        if isinstance(a,torch.Tensor):
            return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict):
            return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):
            return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    for key in ('model','optimizer','rng','step','velocity_coefficient'):
        if not equal(current[key],archived[key]):
            raise AssertionError(f'Archived control mismatch: {key}')


def run(args):
    source_dir=lab.toy_path(args.source)
    output=lab.toy_path(args.output)
    original=json.loads((source_dir/'plan.json').read_text())
    if args.arm not in original['arms'] or original['schema']!='relational_retention_v1':
        raise ValueError('Select an existing reviewed arm, not a new branch or source')
    cfg,source=lab.load_source()
    cfg=replace(cfg,eval_every=25,eval_events=512)
    data=RelationalData(Data(cfg))
    initial=retention.make_models(source,cfg,original)[args.arm]
    judge,provenance=retention.installed_reward(original)
    plan={**original,'schema':'relational_periodic_refit_v1','steps':300,
          'source_round':str(source_dir),'output':str(output),
          'question':'Does refreshing residual reward/reference every100 updates help?',
          'run_name':'Can fresh rewards sustain progress? | periodic refit 100 | fixed judge and cold audit',
          'refit_steps':[100,200], 'fixed_judge':provenance,
          'optimizer':'Preserve AdamW moments, update count and policy RNG across refits',
          'reference':'Refresh to exactly the policy generating each classifier negative panel',
          'audit_panel_seed':original['audit_panel_seed']+1000000,
          'primary_endpoint':'Fresh endpoint BCE/AUC versus archived fixed control at300; '
                             'fixed-judge paired reward and explicit joint moments are secondary',
          'decision_rule':'Improvement requires all fits valid; paired fresh BCE contrast lower95>0 '
                          'and AUC contrast upper95<0. No test-based checkpoint selection.',
          'scope':'Local toy reward/reference refresh, NOT full detector/truth OmniFold; '
                  'cannot separate reference reset from classifier refresh causally',
          'review_note':args.review_note}
    plan['reference_only']=args.reference_only
    plan['active_arm']=args.arm
    plan['arms']=[args.arm]
    plan['architecture']='Inherited exact source arm, including its anchored basis if present'
    plan['evaluation_seed']=930301
    evaluation={**original,'evaluation_seed':plan['evaluation_seed']}
    plan['milestones']=[0,1,5,10,20,50,100,105,120,150,200,205,220,250,300]
    if args.arm=='relative':
        plan.update(run_name='Can relation-aware refit improve joint? | relative FiLM | cold reward every100')
    if args.reference_only:
        plan.update(question='Does recentering the velocity reference alone explain the refit benefit?',
            run_name='Is reference refresh sufficient? | fixed critic | recenter every100',
            refit_steps=[],reference_reset_steps=[100,200],
            scope='Diagnostic fixed-reward optimization with moving velocity reference; '
                  'NOT a matched density-ratio-to-current-reference truth objective. '
                  'Compare with round5 to isolate the additional cold-classifier effect.')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'plan.json',plan)
    started=time.monotonic()
    report={'state':'running','plan':plan,'points':{},'refits':{}}
    arrays={}
    tracker=None
    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        if tracker is not None:
            from .coverage_budget import flatten_metrics
            tracker.log(flatten_metrics(row,f"{row['phase']}/round{row.get('reward_round',0)}/"))
        atomic_json(output/'status.json',{'state':report['state'],'active':row,
                                         'elapsed_seconds':time.monotonic()-started})
        print(json.dumps(row,allow_nan=False),flush=True)

    @torch.no_grad()
    def endpoint(arm,step,model):
        rr=rel.reward_panel(model,judge,data,cfg,evaluation)
        key=f'{arm}_{step}'
        arrays[key]=rr
        item={'fixed_judge_mean':float(rr.double().mean()),
              'fixed_judge_gain':float((rr-arrays.get('baseline_0',rr)).double().mean()),
              'fixed_judge_weight':native.weight_health(rr)}
        if step in (0,300):
            item['structure'],payload=rel.structure_panel(model,judge,data,cfg,original)
            atomic_checkpoint(output/f'{key}_structure.pt',payload)
        report['points'][key]=item
        atomic_checkpoint(output/'fixed_judge_arrays.pt',arrays)
        atomic_json(output/'report.json',report)
        emit({'phase':'fixed_judge','arm':arm,'step':step,**item})

    try:
        tracker=retention.tracker_for(output,plan)
        endpoint('baseline',0,initial)
        current=copy.deepcopy(initial)
        critic=judge
        resume=None
        current_provenance=provenance
        for reward_round,stop in enumerate((100,200,300)):
            # policy_train copies this exact origin into its frozen reference,
            # then restores the actor and AdamW from resume without resetting RNG.
            reference=copy.deepcopy(current).eval().requires_grad_(False)
            atomic_checkpoint(output/f'reference_round{reward_round}.pt',{
                'model':reference.state_dict(),'step':stop-100,
                'reward_source':current_provenance})
            boundary={}
            def checkpoint(step,model,opt,rng,history):
                if step not in (1,5,10,20,50,100,105,120,150,200,205,220,250,300):
                    return
                endpoint('periodic',step,model)
                state=state_at_boundary(model,opt,rng,history,current_provenance,args.arm)
                state.update(reference=str(output/f'reference_round{reward_round}.pt'),
                             reward_round=reward_round)
                atomic_checkpoint(output/f'periodic_step{step}.pt',state)
                if step==stop:
                    boundary.update(state)
            current,_=native.policy_train('dgpo',reference,critic,data,
                replace(cfg,policy_steps=stop),original['policy_seed'],original['monitor_seed'],
                lambda row:emit({**row,'reward_round':reward_round}),checkpoint,
                resume_state=resume,velocity_coefficient=1.)
            resume=boundary
            if stop==100:
                archived=torch.load(source_dir/f'{args.arm}_step100.pt',map_location='cpu',weights_only=True)
                assert_replay(resume,archived)
                report['first_interval_bitwise_replay']=True
            if stop==300:
                break
            if args.reference_only:
                emit({'phase':'reference_only_refresh','reward_round':reward_round+1,
                      'step':stop,'classifier_unchanged':True,'optimizer_reset':False,
                      'reference_reset':True})
                continue
            pp={'policy':rel.audit_panels(current,data,original['audit_panel_seed'])}
            fitdir=output/f'refit_step{stop}'
            fit,scores=rel.fit(pp,['relative'],fitdir,
                lambda row:emit({**row,'phase':'refit_classifier','reward_round':reward_round+1,
                                 'policy_step':stop}),original)
            critic,info=rel.load_critic(fitdir,'policy__relative')
            if not info['valid']:
                raise ValueError('Fresh refit did not reach the prespecified fit-validity gate')
            current_provenance={'directory':str(fitdir),'key':'policy__relative',
                                'negative_policy_step':stop,'fit':info}
            report['refits'][str(stop)]=current_provenance
            atomic_checkpoint(output/f'refit_step{stop}_scores.pt',scores)
            atomic_json(output/'report.json',report)
            emit({'phase':'refit_installed','reward_round':reward_round+1,'step':stop,
                  'fit':info,'optimizer_reset':False,'reference_reset':True})

        # Existing control is valid only after exact first-interval replay above.
        fixed=copy.deepcopy(initial)
        for step in (100,200,300):
            state=torch.load(source_dir/f'{args.arm}_step{step}.pt',map_location='cpu',weights_only=True)
            fixed.load_state_dict(state['model'],strict=True)
            endpoint('fixed',step,fixed)
        report['fixed_judge_contrasts']=simultaneous_mean_intervals({
            f'periodic_minus_fixed_{s}':(arrays[f'periodic_{s}']-arrays[f'fixed_{s}']).double().mean(-1)
            for s in (100,200,300)},2000,929991)
        # New independent audit panels, not any reward-fit panel. Every endpoint
        # receives the same data sizes, initialization, batch IDs and stop rule.
        pp={name:rel.audit_panels(model,data,plan['audit_panel_seed'])
            for name,model in [('baseline',initial),('fixed',fixed),('periodic',current)]}
        lab.assert_pairing(pp)
        fit,scores=rel.fit(pp,['relative'],output/'fresh_endpoint_audits',
            lambda row:emit({**row,'phase':'fresh_endpoint_audit'}),plan)
        report['fresh_fit']=fit
        report['fresh_contrasts']=lab.compare_scores(scores,[
            ('periodic__relative','fixed__relative'),
            ('periodic__relative','baseline__relative'),
            ('fixed__relative','baseline__relative')],repeats=2000)
        valid=all(item['valid'] for item in fit['test'].values())
        primary=report['fresh_contrasts']['periodic__relative minus fixed__relative']
        report['primary_success']=valid and primary['auc']['hi95']<0 and primary['bce']['lo95']>0
        report['state']='completed_awaiting_review'
        report['elapsed_seconds']=time.monotonic()-started
        atomic_checkpoint(output/'fresh_endpoint_scores.pt',scores)
        atomic_json(output/'report.json',report)
        emit({'phase':'decision','primary_success':report['primary_success'],
              'fresh_contrasts':report['fresh_contrasts'],
              'fixed_judge_contrasts':report['fixed_judge_contrasts']})
    except BaseException as exc:
        report.update(state='interrupted_or_failed',error=repr(exc))
        atomic_json(output/'report.json',report)
        raise
    finally:
        if tracker is not None:
            tracker.finish()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-note',required=True)
    p.add_argument('--reference-only',action='store_true',
                   help='Diagnostic control: recenter reference but never refresh classifier')
    p.add_argument('--arm',choices=('raw','relative'),default='raw',
                   help='Use an existing reviewed arm from the source round without changing its basis')
    args=p.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__=='__main__':
    main()
