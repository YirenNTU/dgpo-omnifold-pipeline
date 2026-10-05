"""Replay saved16-GPU K64 results: Cij component uncertainty and event accounting."""
import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tau_cij_components import analyze, AXES


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if (cfg['expected_events'] != 119002 or cfg['expected_candidates'] != 64
            or cfg['source_workers'] != 16 or cfg['cap'] != 30
            or cfg['bootstrap'] < 2000 or cfg['cpu_threads'] < 1 or cfg['top_count'] < 1):
        raise ValueError('Requires full16-GPU K64 cap30 source and at least2000 event bootstraps')
    if len(cfg['logger']['name']) > 96 or len(cfg['logger']['name'].split(' | ')) < 3:
        raise ValueError('Invalid W&B name')
    return cfg


def load_source(cfg):
    source = Path(cfg['source'])
    if not (source/'COMPLETE').is_file():
        raise ValueError('Confirmation source not complete')
    if json.loads((source/'wandb.json').read_text())['id'] != cfg['source_run']:
        raise ValueError('Source run mismatch')
    m = json.loads((source/'manifest.json').read_text())
    if (m['conditions'] != cfg['expected_events'] or m['prefixes'] != [64]
            or m['workers'] != cfg['source_workers'] or m['cap'] != cfg['cap']
            or m['events'] != cfg['events'] or m['weights'] != 'raw_state_dict_only'
            or m['classifier_run'] != 'pzq0nl1i' or m['inherited_candidates'] != 0
            or m['classifier_fits'] != 0 or m['policy_updates'] != 0):
        raise ValueError('Frozen raw1110/FiLM/fresh K64 protocol mismatch')
    from evenet_dgpo.evenet.dataset.filtered_data import validate_filtered_dataset
    filtered = validate_filtered_dataset(cfg['events'])
    if filtered['rows'] != cfg['expected_events']:
        raise ValueError('Filtered population mismatch')
    with np.load(source/'inputs.npz', allow_pickle=False) as f:
        inputs = {key:f[key] for key in ('source_ids','weight','truth_cij','category')}
    with np.load(source/'samples_and_scores.npz', allow_pickle=False) as f:
        if not np.array_equal(inputs['source_ids'], f['source_ids']):
            raise ValueError('Sample identities or ordering mismatch')
        data = {key:f[key] for key in ('logits','cij')}
    if data['logits'].shape != (cfg['expected_events'],cfg['expected_candidates']):
        raise ValueError('Incomplete candidate population')
    previous = json.loads((source/'cap_confirmation_report.json').read_text())
    return m, inputs, data, previous, filtered


def verify_replay(report, previous):
    if not np.allclose(report['truth'],previous['truth'],atol=1e-9,rtol=1e-8):
        raise ValueError('Truth endpoint did not reproduce')
    for row in report['arms']:
        old = next(r for r in previous['arms'] if r['arm'] == row['arm'])
        for key in ('cij','error','event_ess'):
            if not np.allclose(row[key],old[key],atol=1e-9,rtol=1e-8):
                raise ValueError('Saved endpoint did not reproduce: '+row['arm']+'/'+key)


def plots(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax, name in zip(axs,('raw','cap30')):
        rows = [r for r in report['components'] if r['arm']==name]
        vals = np.array([r['absolute_error_change'] for r in rows])
        band = np.array([r['simultaneous_ci95'] for r in rows])
        ax.vlines(np.arange(9),band[:,0],band[:,1],color='tab:blue',label='Simultaneous95% / family18')
        ax.scatter(np.arange(9),vals,color='tab:blue')
        ax.axhline(0,color='black',linestyle='--')
        ax.set(xticks=range(9),xticklabels=AXES,title=name,ylabel='Absolute error change vs unweighted (negative better)')
        ax.legend(fontsize=8)
    fig.savefig(output/'component_intervals.png',dpi=160); plt.close(fig)
    fig, axs = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for a,name in enumerate(('raw','cap30')):
        rows=report['attribution'][name]
        ax=axs[a,0]; x=np.arange(9)
        ax.bar(x-.18,[r['within_net'] for r in rows],width=.36,label='Within condition')
        ax.bar(x+.18,[r['between_net'] for r in rows],width=.36,label='Between conditions')
        ax.scatter(x,[r['net'] for r in rows],color='black',label='Total',zorder=3)
        ax.axhline(0,color='black',linestyle='--')
        ax.set(xticks=x,xticklabels=AXES,title=name+' / exact observed accounting',ylabel='Absolute error change contribution')
        ax.legend(fontsize=8)
        ax=axs[a,1]
        grid=np.array([[np.nan if v is None else v for v in row] for row in report['tradeoffs'][name]])
        im=ax.imshow(grid,vmin=0,vmax=1,cmap='magma')
        ax.set(xticks=x,xticklabels=AXES,yticks=x,yticklabels=AXES,
               xlabel='Simultaneously harmed component',ylabel='Helped component',title=name+' / helpful-contribution overlap')
        fig.colorbar(im,ax=ax,label='Fraction of helpful contribution')
    fig.savefig(output/'event_accounting.png',dpi=160); plt.close(fig)


def log_report(run, report, output):
    import wandb
    rows=[]
    for r in report['components']:
        lo,hi=r['simultaneous_ci95']
        rows.append([r['arm'],r['component'],r['truth'],r['unweighted'],r['estimate'],
                     r['absolute_error_change'],*r['pointwise_ci95'],lo,hi,r['status']])
        prefix=r['arm']+'/'+r['component']
        run.summary.update({prefix+'/absolute_error_change':r['absolute_error_change'],
                            prefix+'/simultaneous_lo95':lo,prefix+'/simultaneous_hi95':hi,
                            prefix+'/status':r['status']})
    run.log({'Cij/components':wandb.Table(columns=['arm','component','truth','unweighted','estimate',
        'absolute_error_change','point_lo95','point_hi95','simultaneous_lo95','simultaneous_hi95','status'],data=rows)})
    top, concentration, accounting, groups=[],[],[],[]
    for name, records in report['attribution'].items():
        for r in records:
            accounting.append([name,r['component'],r['helpful_sum'],r['harmful_sum'],r['net'],
                               r['within_net'],r['between_net'],r['harmful_event_fraction']])
            for c in r['concentration']:
                concentration.append([name,r['component'],c['events'],c['fraction_of_all_harm'],
                                      c['descriptive_change_without_these_terms']])
            for direction, events in r['top_events'].items():
                for e in events:
                    top.append([name,r['component'],direction,e['source_id'],e['category'],
                                e['contribution'],e['within'],e['between'],*e['all_components']])
        for g in report['categories'][name]:
            groups.append([name,g['category'],g['events'],g['base_mass'],g['weighted_mass'],*g['absolute_error_contribution']])
    run.log({'events/accounting':wandb.Table(columns=['arm','component','helpful_sum','harmful_sum','net',
        'within','between','harmful_event_fraction'],data=accounting),
        'events/concentration':wandb.Table(columns=['arm','component','events','fraction_of_harm',
        'accounting_remainder_NOT_deletion_estimate'],data=concentration),
        'events/top':wandb.Table(columns=['arm','component','direction','source_id','category','contribution',
        'within','between',*AXES],data=top),
        'events/categories':wandb.Table(columns=['arm','category','events','base_mass','weighted_mass',*AXES],data=groups)})
    for name in ('component_intervals.png','event_accounting.png'):
        run.log({'Cij/'+name.removesuffix('.png'):wandb.Image(str(output/name))})
    for name in ('component_report.json','component_intervals.png','event_accounting.png'):
        run.save(str(output/name),base_path=str(output),policy='now')
    # Large per-event arrays stay local; source IDs are available in the top-event table.
    run.summary.update(dict(phase='complete',source_endpoints_verified=True,generated_samples=0,
                            classifier_fits=0,policy_updates=0,report_path=str(output/'component_report.json')))


def execute(cfg, output, run=None):
    manifest, inputs, data, prior, filtered=load_source(cfg)
    def progress(stage):
        print('[Cij components]',stage,flush=True)
        if run: run.log({'phase':stage})
    report,arrays=analyze(inputs,data,cfg,progress)
    verify_replay(report,prior)
    report.update(settings=cfg,source_manifest=manifest,filtered_input=filtered,
                  source_endpoints_verified=True,generated_samples=0,classifier_fits=0,policy_updates=0,
                  independent_event_generalization_tested=False)
    np.savez_compressed(output/'event_contributions.npz',**arrays)
    (output/'component_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plots(report,output)
    if run: log_report(run,report,output)
    (output/'COMPLETE').write_text('Saved-panel component diagnostics complete; source unchanged\n')
    print('REPORT:',output/'component_report.json',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--no-wandb',action='store_true')
    args=p.parse_args(); cfg=read_settings(args.config)
    output=Path(cfg['output_root'])/('components-'+uuid.uuid4().hex[:10]); output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=cfg['cpu_threads']):
        if args.no_wandb: execute(cfg,output)
        else:
            import wandb
            logger=cfg['logger']
            with wandb.init(entity=logger['entity'],project=logger['project'],name=logger['name'],group=logger['group'],
                            config=cfg,dir=str(output),mode='online',
                            tags=['saved-16-GPU-panel','CPU-analysis','Cij','no-training','fixed-cap30']) as run:
                (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
                run.summary.update(dict(phase='loading',generated_samples=0,classifier_fits=0,policy_updates=0))
                try: execute(cfg,output,run)
                except BaseException:
                    run.summary['phase']='failed'
                    raise
                print('WANDB:',run.url,flush=True)


if __name__=='__main__': main()
