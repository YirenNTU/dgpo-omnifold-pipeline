"""Compare condition-scale corrections on existing scored K64 samples; no training."""
import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_tau_cij_components import load_source, verify_replay
from scripts.tau_conditional_normalization import analyze


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['source_run'] != 'fzekzrmr' or cfg['expected_events'] != 119002
            or cfg['expected_candidates'] != 64 or cfg['source_workers'] != 16
            or cfg['prefixes'] != [32,64] or cfg['cap'] != 30
            or cfg['bootstrap'] < 2000 or cfg['cpu_threads'] < 1):
        raise ValueError('Requires pinned full fzekzrmr panel, K32/64 and >=2000 bootstraps')
    name=cfg['logger']['name']
    if len(name)>96 or not 3 <= len(name.split(' | ')) <= 5:
        raise ValueError('Invalid W&B display name')
    return cfg


def plots(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    prefixes=list(report['prefixes'])
    for i,row in enumerate(report['prefixes'][prefixes[0]]['arms'][:5]):
        name=row['arm']; rows=[report['prefixes'][k]['arms'][i] for k in prefixes]
        axs[0].plot([int(k) for k in prefixes],[r['error'] for r in rows],'-o',label=name)
    axs[0].set(xlabel='Candidates per condition',ylabel='Cij Frobenius error',title='Same saved scores; no fitting')
    axs[0].legend(fontsize=7)
    for name in ('event_normalized','disjoint_normalized'):
        rows=[next(r for r in report['prefixes'][k]['arms'] if r['arm']==name)['comparisons']['raw'] for k in prefixes]
        x=np.array([int(k) for k in prefixes])+(0 if name=='event_normalized' else 1)
        ci=np.array([r['error_change_ci95'] for r in rows])
        axs[1].vlines(x,ci[:,0],ci[:,1]); axs[1].plot(x,[r['error_change'] for r in rows],'o',label=name)
    axs[1].axhline(0,color='black',linestyle='--')
    axs[1].set(xlabel='Candidates per condition (offset for clarity)',ylabel='Error change minus raw',title='Paired event bootstrap, pointwise 95%')
    axs[1].legend(fontsize=8)
    fig.savefig(output/'normalization_comparison.png',dpi=160); plt.close(fig)


def log_report(run,report,output):
    import wandb
    components=[]; matrices=[]
    for k,panel in report['prefixes'].items():
        for row in panel['arms']:
            matrices.append([int(k),row['arm'],*row['cij']])
            prefix=f'K{k}/{row["arm"]}'
            for key in ('error','event_ess','candidate_ess','event_mass_tv','category_mass_tv','max_candidate_mass'):
                run.summary[prefix+'/'+key]=row[key]
            for baseline,comp in row['comparisons'].items():
                stem=prefix+'/minus_'+baseline
                run.summary[stem+'/error_change']=comp['error_change']
                run.summary[stem+'/lo95'],run.summary[stem+'/hi95']=comp['error_change_ci95']
        for key,value in panel['denominator'].items():
            run.summary[f'K{k}/denominator/{key}']=value
        for c in panel['components_vs_raw']:
            components.append([int(k),c['arm'],c['component'],c['absolute_error_change'],
                               *c['pointwise_ci95'],*c['simultaneous_ci95'],c['status']])
    run.log({'Cij/matrices':wandb.Table(columns=['K','arm',*report['axes']],data=matrices),
        'Cij/components_vs_raw':wandb.Table(columns=['K','arm','component','absolute_error_change',
        'point_lo95','point_hi95','simultaneous_lo95','simultaneous_hi95','status'],data=components),
        'Cij/normalization_comparison':wandb.Image(str(output/'normalization_comparison.png'))})
    for name in ('normalization_report.json','normalization_comparison.png'):
        run.save(str(output/name),base_path=str(output),policy='now')
    run.summary.update(dict(phase='complete',source_endpoints_verified=True,
        report_path=str(output/'normalization_report.json')))


def execute(cfg,output,run=None):
    manifest,inputs,data,prior,filtered=load_source(cfg)
    def progress(stage):
        print('[conditional normalization]',stage,flush=True)
        if run: run.log({'phase':stage})
    report,arrays=analyze(inputs,data,cfg,progress)
    verify_replay(dict(truth=report['truth'],arms=report['prefixes']['64']['arms'][:3]),prior)
    report.update(settings=cfg,source_manifest=manifest,filtered_input=filtered,
        source_endpoints_verified=True,classifier_fits=0,policy_updates=0,generated_samples=0,
        independent_event_generalization_tested=False)
    (output/'normalization_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'denominators.npz',**arrays)
    plots(report,output)
    if run: log_report(run,report,output)
    (output/'COMPLETE').write_text('Read-only fixed-panel diagnostic complete; source unchanged\n')
    print('REPORT:',output/'normalization_report.json',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path); p.add_argument('--no-wandb',action='store_true')
    args=p.parse_args(); cfg=read_settings(args.config)
    output=Path(cfg['output_root'])/('normalization-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):
        if args.no_wandb: execute(cfg,output)
        else:
            import wandb
            with wandb.init(**cfg['logger'],config=cfg,dir=str(output),mode='online',
                            tags=['saved-16-GPU-panel','CPU-analysis','no-training','conditional-normalization']) as run:
                (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
                run.summary.update(dict(phase='loading',classifier_fits=0,policy_updates=0,generated_samples=0))
                try: execute(cfg,output,run)
                except BaseException:
                    run.summary['phase']='failed'
                    raise
                print('WANDB:',run.url,flush=True)


if __name__=='__main__': main()
