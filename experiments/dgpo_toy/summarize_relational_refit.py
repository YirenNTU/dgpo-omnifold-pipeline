"""Read-only segment diagnostics and plots for the reviewed periodic toy run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from . import closed_loop_lab as lab
from . import conditional as native
from . import relational_experiment as rel
from . import relational_retention as retention
from .nonperiodic_cube import Data
from .relational_conditioning import RelationalData
from .classifier_signal_attribution import simultaneous_mean_intervals, score_decomposition
from .cube_swap import truth_shape
from .truth_pretrain import atomic_json,atomic_checkpoint


def run(output,against=None,skip_segments=False):
    output=lab.toy_path(output)
    report=json.loads((output/'report.json').read_text())
    plan=report['plan']
    if report['state']!='completed_awaiting_review':
        raise ValueError('Wait for the paired endpoint audits to finish')
    source_plan=json.loads((Path(plan['source_round'])/'plan.json').read_text())
    evaluation={**source_plan,'evaluation_seed':plan['evaluation_seed']}
    cfg,source=lab.load_source();data=RelationalData(Data(cfg))
    model=retention.make_models(source,cfg,source_plan)[plan.get('active_arm','raw')]
    first,_=retention.installed_reward(source_plan)
    arrays,columns={},{}
    for origin in (() if skip_segments else (0,100,200)):
        reward=(first if origin==0 or plan.get('reference_only') else
                rel.load_critic(output/f'refit_step{origin}','policy__relative')[0])
        for delta in (0,5,20,100):
            step=origin+delta
            path=(output/'reference_round0.pt') if step==0 else output/f'periodic_step{step}.pt'
            ck=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(ck['model'],strict=True)
            arrays[f'{origin}/{delta}']=rel.reward_panel(model,reward,data,cfg,evaluation)
        base=arrays[f'{origin}/0']
        for delta in (5,20,100):
            columns[f'round{origin//100}/gain{delta}']=(arrays[f'{origin}/{delta}']-base).double().mean(-1)
        columns[f'round{origin//100}/change5to20']=(arrays[f'{origin}/20']-arrays[f'{origin}/5']).double().mean(-1)
    result={'question':'Does each installed reward show early gain then loss?',
            'posthoc':True,'selection':'All three intervals, fixed0/5/20/100; no selected peak',
            'scope':'Descriptive within-round rewards; cannot compare reward levels across refits',
            'intervals':simultaneous_mean_intervals(columns,2000,930017) if columns else {},
            'segments_evaluated':not skip_segments}
    if not skip_segments:
        atomic_checkpoint(output/'segment_reward_arrays.pt',arrays)
        atomic_json(output/'segment_report.json',result)
    decompose_fixed_judge(output,first,data)
    plot(output,report)
    if against is not None:
        compare=(compare_refresh_arms if plan.get('reference_only') else compare_conditioning)
        compare(output,lab.toy_path(against))
    print(json.dumps(result,allow_nan=False),flush=True)


def compare_conditioning(relative,raw):
    a=json.loads((relative/'report.json').read_text())
    b=json.loads((raw/'report.json').read_text())
    if a['plan'].get('active_arm')!='relative' or b['plan'].get('active_arm','raw')!='raw':
        raise ValueError('Expected relative versus raw conditioning with the same refit schedule')
    for key in ('audit_panel_seed','fit','refit_steps','policy_seed','evaluation_seed','steps'):
        if a['plan'][key]!=b['plan'][key]:
            raise ValueError(f'Conditioning comparison changes {key}')
    aa=torch.load(relative/'fresh_endpoint_scores.pt',map_location='cpu',weights_only=True)
    bb=torch.load(raw/'fresh_endpoint_scores.pt',map_location='cpu',weights_only=True)
    for label in ('positive','negative'):
        if not torch.equal(aa['baseline__relative'][label],bb['baseline__relative'][label]):
            raise AssertionError('Different baseline audit or data')
    ar=torch.load(relative/'fixed_judge_arrays.pt',map_location='cpu',weights_only=True)
    br=torch.load(raw/'fixed_judge_arrays.pt',map_location='cpu',weights_only=True)
    if not torch.equal(ar['baseline_0'],br['baseline_0']):
        raise AssertionError('Different initial generation/judge panel')
    result={'state':'completed','baseline_replay_bitwise':True,
            'fresh_contrasts':lab.compare_scores({'relative':aa['periodic__relative'],
                'raw':bb['periodic__relative']},[('relative','raw')],repeats=2000),
            'fixed_judge_contrast':simultaneous_mean_intervals({'relative_minus_raw':
                (ar['periodic_300']-br['periodic_300']).double().mean(-1)},2000,930127),
            'structure':{name:r['points']['periodic_300']['structure']
                         for name,r in [('relative',a),('raw',b)]},
            'scope':'Posthoc paired comparison of two reviewed periodic-refit arms, same '
                    'initial function. Relative uses the declared frozen-offset construction; '
                    'not an equal-effective-capacity or training-seed replication claim.'}
    atomic_json(relative/'conditioning_comparison.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)


def compare_refresh_arms(reference_only,both):
    a=json.loads((reference_only/'report.json').read_text())
    b=json.loads((both/'report.json').read_text())
    if not a['plan'].get('reference_only') or b['plan'].get('reference_only'):
        raise ValueError('Expected reference-only vs full refresh')
    if a['plan']['audit_panel_seed']!=b['plan']['audit_panel_seed'] or a['plan']['fit']!=b['plan']['fit']:
        raise ValueError('Different audit data or classifier setup')
    aa=torch.load(reference_only/'fresh_endpoint_scores.pt',map_location='cpu',weights_only=True)
    bb=torch.load(both/'fresh_endpoint_scores.pt',map_location='cpu',weights_only=True)
    # Exactly identical predictions from both unchanged anchor fits validate
    # replay of the panel, fitting initialization and batch stream across jobs.
    for anchor in ('baseline__relative','fixed__relative'):
        for label in ('positive','negative'):
            if not torch.equal(aa[anchor][label],bb[anchor][label]):
                raise AssertionError('Cold audit anchor did not replay bitwise')
    scores={'reference_only':aa['periodic__relative'],'both':bb['periodic__relative']}
    ar=torch.load(reference_only/'fixed_judge_arrays.pt',map_location='cpu',weights_only=True)
    br=torch.load(both/'fixed_judge_arrays.pt',map_location='cpu',weights_only=True)
    for anchor in ('baseline_0','fixed_300'):
        if not torch.equal(ar[anchor],br[anchor]):
            raise AssertionError('Frozen judge panel is not paired')
    result={'state':'completed','cold_anchor_replay_bitwise':True,
            'fresh_contrasts':lab.compare_scores(scores,[('both','reference_only')],repeats=2000),
            'fixed_judge_contrast':simultaneous_mean_intervals({'both_minus_reference_only':
                (br['periodic_300']-ar['periodic_300']).double().mean(-1)},2000,930121),
            'scope':'Additional effect of cold classifier refresh given the same reference-reset '
                    'schedule. One training seed; paired heldout population inference only.'}
    atomic_json(reference_only/'refresh_comparison.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)


@torch.no_grad()
def decompose_fixed_judge(output,judge,data):
    """Exact saved-grid reward identity; not a causal or KL decomposition."""
    pools={name:torch.load(output/f'{name}_structure.pt',map_location='cpu',weights_only=True)
           for name in ('baseline_0','fixed_300','periodic_300')}
    visible=pools['baseline_0']['visible'];grid=len(visible);draws=512
    ids=torch.arange(8)[None,:,None].expand(grid,-1,draws)
    ty=truth_shape(ids,data.centers,native.generator(930117))
    cc=visible[:,None,None,:].expand(grid,8,draws,4).reshape(-1,4)
    means=torch.cat([judge(y,c) for y,c in zip(ty.reshape(-1,3).split(2048),cc.split(2048))])
    means=means.reshape(grid,8,draws).double().mean(-1)
    decompositions={}
    for name,pool in pools.items():
        if not torch.equal(pool['visible'],visible) or not pool['cells']['mean_available'].all():
            raise ValueError('Unpaired grid or missing mode mean in reward attribution')
        decompositions[name]=score_decomposition(pool['p'],pool['q'],means,
                                                pool['cells']['mean'],data.centers)
    components=['order1_mode_logit_gap','order2_mode_logit_gap','order3_mode_logit_gap',
                'within_mode_shape_logit_gap','total_logit_gap']
    columns={f'{arm}/{part}':decompositions['baseline_0'][part]-decompositions[arm][part]
             for arm in ('fixed_300','periodic_300') for part in components}
    for arm in ('fixed_300','periodic_300'):
        direct=(pools[arm]['rewards'].double().mean(-1)-pools['baseline_0']['rewards'].double().mean(-1))
        torch.testing.assert_close(direct,columns[f'{arm}/total_logit_gap'],rtol=1e-7,atol=1e-8)
    report={'posthoc':True,'fixed_grid_nodes':grid,'truth_shape_draws_per_mode':draws,
            'contributions':simultaneous_mean_intervals(columns,2000,930118),
            'scope':'Reward-gain identity on saved64-node quadrature grid, decomposed relative '
                    'to truth within-mode shape. Signed terms can cancel. Condition-bootstrap '
                    'intervals are descriptive, not IID population or training-seed uncertainty. '
                    'This grid differs from the main4096-context IID reward panel; do not mix totals.',
            'oracle_usage':'Truth shape used for diagnostic attribution ONLY, never training.'}
    atomic_checkpoint(output/'reward_attribution_arrays.pt',decompositions)
    atomic_json(output/'reward_attribution.json',report)


def plot(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for arm,color in [('fixed','#777777'),('periodic','#067a78')]:
        if arm=='fixed':
            xx=[0,1,5,10,20,50,100,200,300]
            # The first100 replay was checked bitwise; before100 use that same
            # trajectory's scores. Later control scores were recomputed on THIS
            # run's panel, which may differ from the source run's legacy seed.
            yy=[0]+[report['points'][f'{"periodic" if s<100 else "fixed"}_{s}']['fixed_judge_gain']
                     for s in xx[1:]]
        else:
            xx=[0]+sorted(int(k.rsplit('_',1)[1]) for k in report['points'] if k.startswith('periodic_'))
            yy=[0]+[report['points'][f'periodic_{s}']['fixed_judge_gain'] for s in xx[1:]]
        axes[0].plot(xx,yy,'.-',color=color,label=arm)
    for step in (100,200):axes[0].axvline(step,color='#067a78',alpha=.3,ls='--')
    axes[0].set(xlabel='DGPO updates',ylabel='Paired gain under unchanged judge',
                title='Fixed reward vs cold refit every100')
    axes[0].legend();axes[0].grid(alpha=.2)
    stats=report['fresh_fit']['test']
    labels=['Baseline','Fixed','Reference100' if report['plan'].get('reference_only') else 'Refit100']
    auc=[stats[f'{s}__relative']['auc'] for s in ('baseline','fixed','periodic')]
    axes[1].plot(labels,auc,'o',markersize=8,color='#325d99')
    for i,v in enumerate(auc):axes[1].annotate(f'{v:.4f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center')
    axes[1].set(ylabel='Fresh classifier test AUC',title='Independent cold audits at300')
    axes[1].margins(y=.5,x=.3);axes[1].grid(axis='y',alpha=.2)
    axes[0].set_title('Reference recenter every 100' if report['plan'].get('reference_only') else
                      'Cold reward/reference refit every 100')
    fig.suptitle(f"{report['plan'].get('active_arm','raw').title()} conditioning | native DGPO + velocity MSE coefficient 1")
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--against',type=Path)
    p.add_argument('--skip-segments',action='store_true',help='Endpoint attribution/comparison only; no extra DDIM panels')
    args=p.parse_args();torch.set_num_threads(1);run(args.output,args.against,args.skip_segments)


if __name__=='__main__':main()
