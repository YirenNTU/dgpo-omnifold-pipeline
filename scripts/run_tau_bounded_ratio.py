"""User-launched 16-GPU bounded BCE against completed post-hoc cap30 control."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import yaml
from scripts.run_tau_fresh_negatives import prepare as prepare_control, panel_worker, panel_report
from scripts.run_tau_ratio_objectives import run_arm, finish_reports
from scripts.tau_bounded_ratio import bound_metrics


def read_settings(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['workers']!=16 or cfg['batch_size']!=1024 or cfg['ratio_bound']!=30
            or cfg['baseline_run']!='pzq0nl1i' or cfg['panel_run']!='fzekzrmr'
            or cfg['feature_batch_size']<1 or cfg['bootstrap']<2000):
        raise ValueError('Requires fixed FiLM control, cap30, same K64 panel and 16 GPUs x1024')
    name=cfg['logger']['name']
    if len(name)>96 or not 3<=len(name.split(' | '))<=5:
        raise ValueError('Invalid W&B display name')
    return cfg


def make_config(settings,baseline,source,output,arm):
    if baseline.get('ratio_bound') is not None or baseline.get('fresh_negatives'):
        raise ValueError('Control must be unbounded fixed-negative BCE')
    if baseline.get('head_kind')!='film' or baseline.get('ratio_objective')!='bce':
        raise ValueError('Control must be FiLM BCE')
    cfg=copy.deepcopy(baseline)
    cfg.update(output=str(output),prepared=str(output/'prepared.npz'),checkpoint=str(output/'best.pt'),
        baseline_directory=str(source),baseline_run=settings['baseline_run'],
        ratio_bound=settings['ratio_bound'],fresh_negatives=False,
        # Used only by the existing frozen-backbone K64 feature extractor. No sampling.
        fresh_runtime=settings['fresh_runtime'],fresh_generator_checkpoint=settings['fresh_generator_checkpoint'],
        feature_batch_size=settings['feature_batch_size'],panel_directory=settings['panel_directory'],
        panel_run=settings['panel_run'],bootstrap=settings['bootstrap'],run_name=settings['logger']['name'],
        wandb_group=settings['logger']['group'],policy_updates=0,backbone_updates=0,generated_samples=0,
        initialization='Fresh head with same seed and parameter tensors; only output map changes.',
        ratio_transform='log r = log30 - softplus(log29 - latent); paired BCE on log r; global weights only',
        target='Bounded ratio, not exact population p/q; deliberate bias-variance tradeoff',
        primary='K64 Cij error versus old classifier post-hoc cap30; paired event bootstrap; no test-based selection',
        tags=['tau','FiLM','bounded-BCE','cap30','fixed-negatives','16-gpu','raw-1110','no-policy-update'])
    return cfg


def bounded_panel_report(inputs,data,logits,bootstrap,seed):
    if not np.isfinite(logits).all() or np.max(logits)>np.log(30)+1e-6:
        raise ValueError('K64 scoring did not preserve the trained bound')
    report=panel_report(inputs,data,logits,bootstrap,seed)
    report['arms']['bounded']=report['arms'].pop('fresh_raw')
    report['arms'].pop('fresh_cap30')
    report['comparisons']={k.replace('fresh_raw','bounded'):v for k,v in report['comparisons'].items()
                           if 'fresh_cap30' not in k}
    report['primary']='bounded_minus_old_cap30; improvement must also be assessed against unweighted'
    report['ratio_bound']=30
    # All nine entries, without promoting pointwise intervals to simultaneous claims.
    target=np.asarray(report['truth_C']).reshape(9)
    for row in report['arms'].values():
        row['absolute_component_error']=np.abs(np.asarray(row['C']).reshape(9)-target).tolist()
    return report


def group_report(inputs,data,logits,edges):
    from scripts.tau_tail_attribution import candidate_weights
    base=inputs['weight'];cats=inputs['category']
    bins=np.searchsorted(edges,inputs['visible_pt_sum'],side='right')
    groups=[('all',np.ones(len(base),bool))]
    groups += [(f'category_{cat}_pt_{b}',(cats==cat)&(bins==b)) for cat in np.unique(cats) for b in range(4)]
    rows=[]
    for name,s in [('unweighted',np.zeros_like(logits)),('old_cap30',np.minimum(data['logits'],np.log(30))),('bounded',logits)]:
        w,_=candidate_weights(base,s)
        num=np.einsum('nk,nkd->nd',w,data['cij']);den=w.sum(1)
        for label,take in groups:
            if not take.any() or base[take].sum()<=0: continue
            if den[take].sum()<=0: raise ValueError('Zero group ratio mass')
            target=np.average(inputs['truth_cij'][take],weights=base[take],axis=0)
            estimate=num[take].sum(0)/den[take].sum()
            rows.append(dict(arm=name,group=label,events=int(take.sum()),
                C=estimate.tolist(),truth_C=target.tolist(),error=float(np.linalg.norm(estimate-target)),
                base_mass=float(base[take].sum()/base.sum()),weighted_mass=float(den[take].sum())))
    return dict(groups=rows,scope='Descriptive fixed category x fit-only pT strata; no subgroup significance claim')


def finish(cfg,arrays,p,q,scores,run,settings,*,extra_score_arms=None,panel_reporter=None):
    import ray
    finish_reports(cfg,arrays,p,q,scores,run,settings,
                   extra_score_arms={'old_cap30':np.minimum(scores['log_ratio'],np.log(30)),
                                     **(extra_score_arms or {})})
    for k,v in bound_metrics(p,q,arrays['event_weight'][arrays['split']==2],30).items():
        run.summary['test_bound/'+k]=v
    run.summary['phase']='fixed_K64_rescore'
    task=ray.remote(num_gpus=1,num_cpus=1,max_calls=1)(panel_worker)
    ray.get([task.remote(cfg,rank) for rank in range(16)])
    panel=Path(cfg['panel_directory']);output=Path(cfg['output'])
    with np.load(panel/'inputs.npz',allow_pickle=False) as f:
        inputs={k:f[k] for k in ('source_ids','weight','truth_cij','category','visible_pt_sum')}
    with np.load(panel/'samples_and_scores.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],inputs['source_ids']): raise ValueError('Panel IDs differ')
        data={k:f[k] for k in ('logits','cij')}
    logits=np.empty_like(data['logits']);seen=np.zeros(len(logits),int)
    for rank in range(16):
        with np.load(output/f'panel-{rank:02d}.npz',allow_pickle=False) as f:
            idx=f['positions']
            if not np.array_equal(f['source_ids'],inputs['source_ids'][idx]): raise ValueError('Shard IDs differ')
            logits[idx]=f['logits'];np.add.at(seen,idx,1)
    if not np.all(seen==1): raise ValueError('Missing/duplicate panel score rows')
    np.savez(output/'fixed_panel_scores.npz',source_ids=inputs['source_ids'],logits=logits)
    report=(panel_reporter or bounded_panel_report)(inputs,data,logits,cfg['bootstrap'],cfg['seed'])
    verify_panel_report(report,panel)
    groups=group_report(inputs,data,logits,cfg['condition_pt_edges'])
    for name,payload in [('fixed_K64_report.json',report),('fixed_K64_groups.json',groups)]:
        path=output/name;path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    publish_panel_report(report,groups,output,run)


def verify_panel_report(report,panel):
    from scripts.diagnose_tau_cij_components import verify_replay
    replay=dict(truth=np.asarray(report['truth_C']).reshape(9).tolist(),arms=[])
    for key,name in (('unweighted','unweighted'),('old_raw','raw'),('old_cap30','cap30')):
        r=report['arms'][key]
        replay['arms'].append(dict(arm=name,cij=np.asarray(r['C']).reshape(9).tolist(),error=r['error'],event_ess=r['event_ess']))
    verify_replay(replay,json.loads((panel/'cap_confirmation_report.json').read_text()))


def publish_panel_report(report,groups,output,run):
    import wandb
    for name in ('fixed_K64_report.json','fixed_K64_groups.json'):
        run.save(str(output/name),base_path=str(output),policy='now')
    for arm,r in report['arms'].items():
        for key in ('error','event_ess','candidate_ess','max_candidate_mass','log_mean_ratio'):
            run.summary[f'K64/{arm}/{key}']=r[key]
    for label,r in report['comparisons'].items():
        run.summary['K64/'+label]=r['error_change']
        run.summary['K64/'+label+'_ci95']=r['error_change_ci95']
    # W&B 0.19 SummaryDict.update accepts a mapping, not dict-style kwargs.
    run.summary.update(dict(source_endpoints_verified=True,generated_samples=0,backbone_updates=0,policy_updates=0))
    run.log({'K64/Cij':wandb.Table(columns=['arm','i','j','Cij','truth'],data=[
        [name,i,j,r['C'][i][j],report['truth_C'][i][j]] for name,r in report['arms'].items() for i in range(3) for j in range(3)]),
        'K64/condition_groups':wandb.Table(columns=['arm','group','events','error','base_mass','weighted_mass'],
            data=[[r[k] for k in ('arm','group','events','error','base_mass','weighted_mass')] for r in groups['groups']])})


def recover_report(settings,directory):
    """Publish completed JSON endpoints only. No fit, model load, Ray or scoring."""
    directory=Path(directory).resolve()
    cfg=json.loads((directory/'manifest.json').read_text())
    for key in ('baseline_run','panel_run','ratio_bound','workers','batch_size'):
        if cfg.get(key)!=settings[key]: raise ValueError('Recovery protocol mismatch: '+key)
    if cfg.get('fresh_negatives') or cfg.get('head_kind')!='film' or cfg.get('ratio_objective')!='bce':
        raise ValueError('Not the fixed-negative bounded BCE experiment')
    if Path(cfg['panel_directory']).resolve()!=Path(settings['panel_directory']).resolve():
        raise ValueError('Recovery panel changed')
    report=json.loads((directory/'fixed_K64_report.json').read_text())
    groups=json.loads((directory/'fixed_K64_groups.json').read_text())
    if report.get('ratio_bound')!=30 or report['events']!=119002 or report['candidates']!=64:
        raise ValueError('Incomplete bounded K64 report')
    verify_panel_report(report,Path(cfg['panel_directory']))
    parent=json.loads((directory/'wandb.json').read_text())
    recovery=directory/('report-recovery-'+uuid.uuid4().hex[:10]);recovery.mkdir()
    import wandb
    logger=settings['logger']
    with wandb.init(entity=logger['entity'],project=logger['project'],group=logger['group'],
            name='Recover bounded ratio results | FiLM cap30 | saved K64 | report only',
            config=dict(source_directory=str(directory),source_run=parent['id'],
                        action='Republish already computed endpoints; no fitting or inference',
                        baseline_run=cfg['baseline_run'],panel_run=cfg['panel_run'],ratio_bound=30),
            dir=str(recovery),mode='online',tags=['report-recovery','no-training','saved-16-GPU-panel']) as run:
        (recovery/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
        run.summary.update(dict(phase='publishing',source_run=parent['id'],classifier_fits=0,inference_calls=0))
        try:
            publish_panel_report(report,groups,directory,run)
            run.summary['phase']='complete'
            (recovery/'COMPLETE').write_text('Saved endpoint reporting recovered; original run unchanged\n')
        except BaseException:
            run.summary['phase']='failed'
            raise
        print('RECOVERED REPORT:',run.url,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('phase',choices=('prepare','train','report'),nargs='?',default='train')
    p.add_argument('--directory',type=Path,help='Existing bounded-* output for report-only recovery')
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args=p.parse_args();settings=read_settings(args.config)
    if args.phase=='report':
        if args.directory is None: p.error('report requires --directory')
        recover_report(settings,args.directory)
        return
    if args.directory is not None: p.error('--directory is only for report-only recovery')
    source,baseline,arrays,scores=prepare_control(settings)
    if args.phase=='train':
        run_arm(settings,'bounded',source,baseline,arrays,scores,args.ray_address,
                config_builder=make_config,reporter=finish)


if __name__=='__main__':main()
