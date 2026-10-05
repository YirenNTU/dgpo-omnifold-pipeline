"""Local, reviewed iterations on conditioning and fixed-reward retention.

Uses the unchanged native DGPO objective. No remote jobs or automatic next
rounds. A round saves its plan before training; later rounds require a new CLI
invocation and a review note. Fresh audits are a separate explicit operation.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
import time
from pathlib import Path

import torch

from . import closed_loop_lab as lab
from . import conditional as native
from . import relational_experiment as rel
from .classifier_signal_attribution import simultaneous_mean_intervals
from .nonperiodic_cube import Data
from .relational_conditioning import RelationalData, RelationalPolicy, VisibleSource, verify_initial, as_frozen_buffers
from .truth_pretrain import atomic_checkpoint, atomic_json


POINTS = (0, 1, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000)
WINDOWS = ((5, 20), (20, 100), (100, 300), (300, 1000))


class AnchoredBasisPolicy(torch.nn.Module):
    """Switch learned condition basis WITHOUT changing the initial generator.

    v = frozen_old + (trainable_new_basis - frozen_initial_new_basis).
    Raw is algebraically identical to the old policy. Relative changes only
    access/tangent geometry, with an explicit frozen offset (not ordinary
    unanchored deployment). No truth, mode labels, or scalar c enter the branch.
    """
    def __init__(self, fitted, representation):
        super().__init__()
        self.origin=as_frozen_buffers(fitted)
        self.network=copy.deepcopy(fitted)
        self.network.representation=representation
        self.anchor=as_frozen_buffers(self.network)

    def predict_features(self,x,t,c):
        current,hidden=self.network.predict_features(x,t,c)
        return self.origin(x,t,c)+(current-self.anchor(x,t,c)),hidden

    def forward(self,x,t,c):
        return self.predict_features(x,t,c)[0]


def retention_contrasts(arrays, arms, repeats=2000):
    """Fixed windows, not a selected peak; simultaneous context-bootstrap CIs."""
    baseline = arrays['baseline_0'].double().mean(-1)
    columns = {}
    for arm in arms:
        for early, late in WINDOWS:
            if f'{arm}_{late}' not in arrays:
                continue
            gain = arrays[f'{arm}_{early}'].double().mean(-1)-baseline
            change = (arrays[f'{arm}_{late}']-arrays[f'{arm}_{early}']).double().mean(-1)
            prefix = f'{arm}/{early}_to_{late}'
            columns[prefix+'/gain'] = gain
            columns[prefix+'/change'] = change
            # late_gain < half early_gain, tested directly without dividing.
            columns[prefix+'/half_retention'] = change+.5*gain
    ci = simultaneous_mean_intervals(columns, repeats, 929170)
    decisions = {}
    for key in columns:
        if not key.endswith('/gain'):
            continue
        prefix = key.rsplit('/', 1)[0]
        decisions[prefix] = (ci[prefix+'/gain']['lo95'] > .01 and
                            ci[prefix+'/change']['hi95'] < -.01 and
                            ci[prefix+'/half_retention']['hi95'] < 0)
    return {'contrasts': ci, 'failure_windows': decisions,
            'any_failure': any(decisions.values()),
            'rule': 'Simultaneous lower early gain > .01, upper later change < -.01, '
                    'upper late_gain-minus-half-early_gain < 0. Fixed windows only.',
            'scope': 'One trained trajectory; context-bootstrap is not training-seed replication.'}


def default_plan(args):
    plan = json.loads((lab.ROOT/'experiments/dgpo_toy/plans/relational_rl.json').read_text())
    plan.update(schema='relational_retention_v1', review_note=args.review_note,
                output=str(args.output), arms=args.arms, steps=args.steps,
                milestones=[s for s in POINTS if s <= args.steps],
                reward_representation='relative', frozen_correction=args.frozen_correction,
                question='Does relational conditioning reproduce loss of early fixed-reward gains?',
                run_name='Can conditioning retain reward? | relational DGPO | fixed critic and V MSE 1',
                wandb_mode='offline', audit_steps='separate explicit endpoint operation',
                primary_endpoint='Fixed-window simultaneous reward-retention test; no peak selection.',
                decision_rule='Apply retention_contrasts, then review before another intervention.',
                scope='Local toy only; same truth, fixed source, frozen critic and native DGPO. '
                      'Relative features supply a structural prior. Not production root-cause proof.')
    plan.update(initial_checkpoint=str(args.initial_checkpoint) if args.initial_checkpoint else None,
                reward_directory=str(args.reward_directory) if args.reward_directory else None,
                reward_key=args.reward_key, anchored_basis=args.anchored_basis)
    # Distinct from audit_panel_seed+10000/20000/30000. The legacy970093
    # collided with the classifier training panel's latent-condition prefix.
    # Archived plans remain immutable for replay; new rounds get a separate RNG.
    plan['evaluation_seed']=930301
    if args.initial_checkpoint:
        plan.update(question='Does a fresh residual reward lose its early gain on the same actor?',
            run_name='Can residual reward still transfer? | refit relational critic | V MSE 1',
            restart='Load exact raw policy weights, new AdamW and reference at this policy; not full-state resume')
    if args.anchored_basis:
        plan.update(question='Can relation access rescue residual reward retention at the SAME endpoint?',
            run_name='Can relation access rescue retention? | matched residual reward | anchored FiLM basis',
            scope='Same saved raw endpoint, frozen residual critic, fresh AdamW and V MSE1. '
                  'Frozen offsets preserve initial velocity/samples across basis switch. '
                  'Raw is an algebraic replay control; relative changes trainable access. Toy only.')
    return plan


def make_models(source, cfg, plan):
    result = {}
    if plan.get('anchored_basis'):
        if not plan.get('initial_checkpoint') or plan.get('frozen_correction'):
            raise ValueError('Anchored basis needs an existing endpoint and unchanged trainable backbone')
        state=torch.load(lab.toy_path(Path(plan['initial_checkpoint'])),map_location='cpu',weights_only=True)
        with torch.random.fork_rng():
            torch.manual_seed(plan['initialization_seed'])
            fitted=RelationalPolicy(source,cfg,state['representation']).eval()
        fitted.load_state_dict(state['model'],strict=True)
        return {arm:AnchoredBasisPolicy(fitted,arm).eval() for arm in plan['arms']}
    for arm in plan['arms']:
        with torch.random.fork_rng():
            torch.manual_seed(plan['initialization_seed'])
            result[arm] = RelationalPolicy(source, cfg, arm).eval()
        if plan.get('frozen_correction'):
            # Native training unfreezes Parameters. Buffers deliberately keep the
            # shared x,t correction frozen; the condition encoder/FiLM still learn.
            from .relational_conditioning import as_frozen_buffers
            result[arm].network = as_frozen_buffers(result[arm].network)
        if plan.get('initial_checkpoint'):
            state=torch.load(lab.toy_path(Path(plan['initial_checkpoint'])),map_location='cpu',weights_only=True)
            if len(plan['arms']) != 1 or state['representation'] != arm:
                raise ValueError('A refit trajectory keeps its existing architecture; no unmatched branch swap')
            result[arm].load_state_dict(state['model'],strict=True)
    return result


def tracker_for(output, plan):
    for key in ('WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR'):
        folder=output/'wandb_state'/key.lower()
        folder.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(folder)
    import wandb
    return wandb.init(project='dgpo-toy-local', mode='offline', dir=str(output),
        name=plan['run_name'], group='Relational reward retention',
        tags=['toy-only', 'fixed-critic', 'raw-no-EMA', 'velocity-MSE-1'],
        config=plan, settings=wandb.Settings(disable_git=True))


def installed_reward(plan):
    if plan.get('reward_directory'):
        folder=lab.toy_path(Path(plan['reward_directory']))
        reward,info=rel.load_critic(folder,plan['reward_key'])
        if not info['valid']:
            raise ValueError('Cannot install an inadequately fitted fresh classifier')
        return reward,{'directory':str(folder),'key':plan['reward_key'],'fit':info,
            'selection':'Fresh classifier from reviewed preceding endpoint; fixed for this round'}
    return rel.load_reward(plan,'relative')


def run(args):
    output = lab.toy_path(args.output)
    plan = default_plan(args)
    cfg, source = lab.load_source()
    cfg = replace(cfg, policy_steps=args.steps, eval_every=25, eval_events=512)
    reward,provenance=installed_reward(plan)
    data = RelationalData(Data(cfg))
    models = make_models(source, cfg, plan)
    reference = (copy.deepcopy(next(iter(models.values()))).eval().requires_grad_(False)
                 if plan.get('initial_checkpoint') else VisibleSource(source))
    matching = ({'state':'exact saved endpoint loaded; refit resets reference and AdamW, not generator',
                 'checkpoint':plan['initial_checkpoint']}
                if plan.get('initial_checkpoint') else verify_initial(source, cfg, models))
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'plan.json', plan)
    start = time.monotonic()
    arrays, report = {}, {'state':'running', 'plan':plan, 'config':asdict(cfg),
                         'initial_matching':matching, 'reward':provenance, 'points':{}}
    reward_weights = {k:v.clone() for k,v in reward.state_dict().items()}
    tracker = None
    def emit(row):
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if tracker is not None:
            from .coverage_budget import flatten_metrics
            tracker.log(flatten_metrics(row, f"{row['phase']}/{row.get('arm','shared')}/"))
        report['elapsed_seconds'] = time.monotonic()-start
        atomic_json(output/'status.json', {'state':report['state'], 'active':row,
                                         'elapsed_seconds':report['elapsed_seconds']})
        print(json.dumps(row, allow_nan=False), flush=True)

    @torch.no_grad()
    def endpoint(arm, step, model):
        rr = rel.reward_panel(model, reward, data, cfg, plan)
        key = f'{arm}_{step}'
        arrays[key] = rr
        gain = rr.double()-arrays.get('baseline_0', rr).double()
        item = {'reward_mean':float(rr.double().mean()), 'gain':float(gain.mean()),
                'reward_std':float(rr.std(unbiased=False)), 'weight':native.weight_health(rr)}
        if step in (0, 300, args.steps):
            stats, payload = rel.structure_panel(model, reward, data, cfg, plan)
            item['structure'] = stats
            atomic_checkpoint(output/f'{key}_structure.pt', payload)
        report['points'][key] = item
        atomic_checkpoint(output/'reward_arrays.pt', arrays)
        atomic_json(output/'report.json', report)
        emit({'phase':'paired_endpoint', 'arm':arm, 'step':step, **item})

    try:
        tracker = tracker_for(output, plan)
        endpoint('baseline', 0, reference)
        # Two replays and both initial models must agree, including held-out reward.
        for name, model in models.items():
            check = rel.reward_panel(model, reward, data, cfg, plan)
            if not torch.equal(check, arrays['baseline_0']):
                raise ValueError('Initial held-out samples/rewards differ')
            atomic_checkpoint(output/f'{name}_initial.pt', {'model':model.state_dict(),
                'config':asdict(cfg), 'step':0, 'representation':name})
            def checkpoint(step, current, opt, rng, history):
                if step not in plan['milestones']:
                    return
                endpoint(name, step, current)
                state = {'model':current.state_dict(), 'optimizer':opt.state_dict(),
                    'rng':rng.get_state(), 'history':history, 'step':step,
                    'velocity_coefficient':1., 'config':asdict(cfg),
                    'representation':name, 'reward':provenance}
                atomic_checkpoint(output/f'{name}_step{step}.pt', state)
            native.policy_train('dgpo', model, reward, data, cfg,
                plan['policy_seed'], plan['monitor_seed'],
                lambda row:emit({**row, 'arm':name}), checkpoint,
                velocity_coefficient=1.)
        report['reward_unchanged'] = all(torch.equal(v,reward_weights[k]) for k,v in reward.state_dict().items())
        if not report['reward_unchanged']:
            raise ValueError('Fixed reward changed')
        report['retention'] = retention_contrasts(arrays, plan['arms'])
        if len(plan['arms']) == 2:
            a,b = plan['arms']
            report['arm_contrasts'] = simultaneous_mean_intervals({f'{b}_minus_{a}_{s}':
                (arrays[f'{b}_{s}']-arrays[f'{a}_{s}']).double().mean(-1)
                for s in plan['milestones'] if s},2000,929171)
        report['state'] = 'completed_awaiting_review'
        report['elapsed_seconds'] = time.monotonic()-start
        atomic_json(output/'report.json', report)
        emit({'phase':'decision', 'failure_reproduced':report['retention']['any_failure']})
        plot(output)
    except BaseException as exc:
        report.update(state='interrupted_or_failed', error=repr(exc))
        atomic_json(output/'report.json', report)
        raise
    finally:
        if tracker is not None:
            tracker.finish()
    return report


def confirm(args):
    """One new IID/noise panel on prespecified saved0/5/20, zero policy updates."""
    output=lab.toy_path(args.output)
    plan=json.loads((output/'plan.json').read_text())
    target=output/'confirmation'
    target.mkdir(exist_ok=False)
    evaluation={**plan,'evaluation_seed':929927,'eval_contexts':16384,'eval_candidates':32}
    atomic_json(target/'plan.json',{'evaluation':evaluation,'steps':[0,5,20],
        'question':'Does the candidate5->20 loss persist on independent conditions and draws?',
        'selection':'Window selected on the first panel before drawing this panel; no new training seed',
        'thresholds':'Same gain>.01, loss>.01 and >50% erasure with simultaneous95 intervals'})
    cfg,source=lab.load_source()
    models=make_models(source,cfg,plan)
    baseline=(copy.deepcopy(next(iter(models.values()))) if plan.get('initial_checkpoint') else VisibleSource(source))
    reward,_=installed_reward(plan)
    data=RelationalData(Data(cfg))
    started=time.monotonic()
    arrays={'baseline_0':rel.reward_panel(baseline,reward,data,cfg,evaluation)}
    for arm,model in models.items():
        for step in (5,20):
            saved=torch.load(output/f'{arm}_step{step}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model'])
            arrays[f'{arm}_{step}']=rel.reward_panel(model,reward,data,cfg,evaluation)
            print(arm,step,float((arrays[f'{arm}_{step}']-arrays['baseline_0']).mean()),flush=True)
            atomic_checkpoint(target/'arrays.pt',arrays)
    result={'state':'completed','retention':retention_contrasts(arrays,plan['arms']),
        'elapsed_seconds':time.monotonic()-started,'contexts':16384,'candidates':32,
        'scope':'Independent evaluation conditions/noise, same fitted models; not independent training replication'}
    atomic_json(target/'report.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)
    return result


def audit(args):
    output = lab.toy_path(args.output)
    plan = json.loads((output/'plan.json').read_text())
    cfg, source = lab.load_source()
    models = make_models(source, cfg, plan)
    policies = {'baseline':(copy.deepcopy(next(iter(models.values()))).eval()
                           if plan.get('initial_checkpoint') else VisibleSource(source))}
    for name, model in models.items():
        state = torch.load(output/f'{name}_step{plan["steps"]}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'])
        policies[name] = model.eval()
    data = RelationalData(Data(cfg))
    pp = {name:rel.audit_panels(m,data,plan['audit_panel_seed']) for name,m in policies.items()}
    lab.assert_pairing(pp)
    for name,panel in pp.items():
        atomic_checkpoint(output/f'audit_{name}_panels.pt', panel)
    tracker=tracker_for(output,{**plan,
        'run_name':'Does fixed reward improve distribution? | fresh relational classifier | endpoint audit'})
    def emit(row):
        with (output/'audit_progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        from .coverage_budget import flatten_metrics
        tracker.log(flatten_metrics(row,f"audit/{row['arm']}/"))
        print(json.dumps(row,allow_nan=False),flush=True)
    started=time.monotonic()
    try:
        fits,scores=rel.fit(pp,['relative'],output/'fresh_audits',emit,plan)
    finally:
        tracker.finish()
    contrasts=lab.compare_scores(scores,[(f'{a}__relative','baseline__relative')
        for a in plan['arms']], repeats=2000)
    result={'state':'completed','fit':fits,'contrasts':contrasts,
            'elapsed_seconds':time.monotonic()-started,
            'scope':'Fresh width128 relative classifiers, matched heldout conditions and truth; '
                    'not the frozen training critic. Test never selects checkpoint.'}
    atomic_checkpoint(output/'fresh_scores.pt', scores)
    atomic_json(output/'audit_report.json',result)
    return result


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((output/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for arm in report['plan']['arms']:
        steps=[0]+[s for s in report['plan']['milestones'] if s]
        gains=[0]+[report['points'][f'{arm}_{s}']['gain'] for s in steps[1:]]
        for ax in axes:
            ax.plot(steps,gains,'.-',label=arm)
    axes[0].set_xlim(0,100);axes[0].set_title('Early updates')
    axes[1].set_title('Full trajectory')
    for ax in axes:
        ax.set_xlabel('DGPO updates');ax.set_ylabel('Paired fixed reward gain')
        ax.axhline(0,color='grey',lw=.7);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Relational conditioning | same critic, source, V MSE coefficient 1')
    fig.tight_layout();fig.savefig(output/'reward_curve.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,choices=(300,1000),default=1000)
    p.add_argument('--arms',nargs='+',choices=('raw','particle','relative'),default=['raw','relative'])
    p.add_argument('--frozen-correction',action='store_true')
    p.add_argument('--anchored-basis',action='store_true')
    p.add_argument('--initial-checkpoint',type=Path)
    p.add_argument('--reward-directory',type=Path)
    p.add_argument('--reward-key',default='source__relative')
    p.add_argument('--review-note',default='')
    p.add_argument('--audit',action='store_true')
    p.add_argument('--confirm',action='store_true')
    args=p.parse_args()
    if not args.audit and not args.confirm and not args.review_note.strip():
        p.error('A reviewed, one-question rationale is required before each new round')
    torch.set_num_threads(1)
    confirm(args) if args.confirm else audit(args) if args.audit else run(args)


if __name__=='__main__':
    main()
