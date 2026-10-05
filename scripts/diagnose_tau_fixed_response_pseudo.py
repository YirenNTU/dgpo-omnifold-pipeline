"""Frozen expanded response, fixed original test population, paired Poisson trials."""
import argparse
import json
from pathlib import Path
import sys
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'evenet_dgpo')]
import numpy as np
import yaml
from scripts.run_tau_cij_unfolding import load_samples
from scripts.tau_bias_sampling import AXES
from scripts.tau_unfold_core import SVDResponse,indices,moment,response_edges,summarize_pseudoexperiments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--response-source',type=Path,required=True)
    p.add_argument('--no-wandb',action='store_true')
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text())
    source_dir=args.response_source
    if not (source_dir/'manifest.json').exists():
        ready=json.loads((source_dir/'ready_samples.json').read_text())
        if ready.get('complete') is not True:raise ValueError('Expanded samples not ready')
        source_dir=Path(ready['sample_directory'])
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT,'RooUnfoldSvd'):ROOT.gSystem.Load(cfg.get('roounfold_library') or 'libRooUnfold')
    if not hasattr(ROOT,'RooUnfoldSvd'):raise RuntimeError('RooUnfold unavailable')
    panel,products,provenance=load_samples(Path(cfg['sample_source']))
    extra,extproducts,extprovenance=load_samples(source_dir)
    if extprovenance.get('population')!='filtered_full_validation_remainder':raise ValueError('Wrong expanded population')
    for arm in ('pretrain','dgpo'):
        a,b=provenance['arms'][arm],extprovenance['arms'][arm]
        if Path(a['checkpoint']).resolve()!=Path(b['checkpoint']).resolve() or a['global_step']!=b['global_step'] or a['epoch']!=b['epoch']:
            raise ValueError('Different models in response and test')
    for key in ('packing_spec','ddim_steps','process_label','weights'):
        if provenance.get(key)!=extprovenance.get(key):raise ValueError(f'{key} differs')
    if np.intersect1d(panel['source_ids'],extra['source_ids']).size:raise ValueError('Response/test overlap')
    kap=np.asarray(panel['kappas']);ekap=np.asarray(extra['kappas'])
    groups,group=np.unique(kap,axis=0,return_inverse=True)
    egroups,egroup=np.unique(ekap,axis=0,return_inverse=True)
    if not np.array_equal(groups,egroups):raise ValueError('Different kappa groups')
    w=np.asarray(panel['event_weight'],float);ew=np.asarray(extra['event_weight'],float)
    if any(not np.isfinite(v).all() or (v<0).any() or v.sum()<=0 for v in (w,ew)):raise ValueError('Invalid MC weights')
    rng=np.random.default_rng(cfg['seed']);old_response=[];test=[]
    for g in range(len(groups)):
        ii=np.flatnonzero(group==g);rng.shuffle(ii);cut=int(len(ii)*cfg['response_fraction'])
        old_response.extend(ii[:cut]);test.extend(ii[cut:])
    old_response=np.sort(old_response);test=np.sort(test)
    N=len(test);B=int(cfg['repeats'])
    if B<2:raise ValueError('Need multiple pseudoexperiments')
    probabilities=w[test]/w[test].sum()
    output=Path(cfg['output'])/('fixed-response-pseudo-'+uuid.uuid4().hex[:10]);output.mkdir(parents=True)
    np.savez_compressed(output/'identities.npz',test_ids=panel['source_ids'][test],response_ids=extra['source_ids'])
    report=dict(complete=False,config=cfg,source=provenance,response_source=extprovenance,test_events=N,
        response_events=len(ew),repeats=B,rows=[],binning={},
        scope='Fixed full response and fixed empirical nominal test population. Statistical-only conditional coverage. '
        'No response bootstrap, no added MC or systematic variance. Repeated resampling cannot decide whether the '
        'original empirical test-population offset is an underlying physical bias or a finite-panel fluctuation. '
        'Centered coverage tests reported statistical variance, not truth closure. mean_mc_se is simulation precision '
        'of the ensemble mean, NOT an uncertainty on the physics measurement. No model or k tuning.')
    run=None
    try:
        if not args.no_wandb:
            import wandb
            settings=dict(cfg['wandb']);settings['name']='Are statistical errors calibrated? | fixed large response | paired nominal pseudo-data'
            run=wandb.init(**settings,config=dict(**cfg,response_source=str(source_dir)),dir=str(output))
        for j,axis in enumerate(AXES):
            edges,info=response_edges({a:v[old_response,j] for a,v in products.items()},group[old_response],
                w[old_response],cfg['bins'],cfg.get('min_response_entries',20))
            report['binning'][axis]=info;nb=len(edges)-1;k=min(cfg['svd_k'],nb)
            coeff=((9/np.prod(groups,axis=1))[:,None]*((edges[:-1]+edges[1:])/2)).ravel()
            z=9*products['truth'][test,j]/np.prod(kap[test],axis=1)
            truth=float(probabilities@z)
            truthbins=group[test]*nb+indices(products['truth'][test,j],edges)
            binned=float(probabilities@coeff[truthbins])
            for arm in ('pretrain','dgpo'):
                ti=indices(extproducts['truth'][:,j],edges);ri=indices(extproducts[arm][:,j],edges)
                banks=[]
                for g in range(len(groups)):
                    take=egroup==g;banks.append(SVDResponse(ROOT,ti[take],ri[take],ew[take],nb))
                reco=group[test]*nb+indices(products[arm][test,j],edges)
                def estimate(counts):
                    values=[];cov=np.zeros((len(coeff),len(coeff)))
                    for g,bank in enumerate(banks):
                        block=slice(g*nb,(g+1)*nb);v,c=bank.unfold(counts[block],k)
                        values.extend(v);cov[block,block]=c
                    return moment(np.asarray(values),cov,coeff)
                expected_counts=N*np.bincount(reco,weights=probabilities,minlength=len(coeff))
                expected,expected_sigma=estimate(expected_counts)
                rng=np.random.default_rng(cfg['seed']+1701)
                trials=[];sample_truth=[]
                for b in range(B):
                    counts=rng.poisson(N*probabilities)
                    if counts.sum()==0:raise ValueError('Empty pseudo-data')
                    measured=np.bincount(reco,weights=counts,minlength=len(coeff))
                    trials.append(estimate(measured));sample_truth.append(float(counts@z/counts.sum()))
                trials=np.asarray(trials)
                row=dict(component=axis,arm=arm,bins=nb,k=k,truth_exact=truth,truth_binned=binned,
                    binning_shift=binned-truth,expected_unfolded=expected,expected_sigma=expected_sigma,
                    **summarize_pseudoexperiments(trials[:,0],trials[:,1],truth,expected))
                row['coverage68_binned']=float(np.mean(np.abs(trials[:,0]-binned)<=trials[:,1]))
                row['paired_sample_truth_residual_std']=float(np.std(trials[:,0]-sample_truth,ddof=1))
                report['rows'].append(row)
                np.savez_compressed(output/f'{axis}_{arm}_trials.npz',estimate=trials[:,0],sigma=trials[:,1],
                    sampled_truth=sample_truth,truth_exact=truth,truth_binned=binned,expected_unfolded=expected)
                print(f'{axis} {arm}: residual={row["mean_residual"]:+.4f} pull={row["pull_mean"]:+.2f}/{row["pull_std"]:.2f} coverage={row["coverage68"]:.2f} centered={row["centered_coverage68"]:.2f}',flush=True)
                if run:
                    run.log({'pseudo/'+key:value for key,value in row.items() if isinstance(value,(int,float))})
                    run.summary.update(dict(component=axis,arm=arm,completed_cases=len(report['rows'])))
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        for field,ideal in [('mean_residual',0),('coverage68',.6827),('centered_coverage68',.6827),('pull_std',1),('std_over_mean_sigma',1)]:
            fig=Figure(figsize=(10,4),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
            for arm in ('pretrain','dgpo'):
                rows=[r for r in report['rows'] if r['arm']==arm]
                ax.plot(AXES,[r[field] for r in rows],'o-',label=arm)
            ax.axhline(ideal,color='k',ls='--');ax.set(title=field+' | fixed response, statistical only');ax.legend()
            path=output/f'{field}.png';fig.savefig(path,dpi=140)
            if run:run.log({'pseudo/plots/'+field:wandb.Image(str(path))})
        report['complete']=True
        if run:
            rows=report['rows'];keys=list(rows[0])
            run.log({'pseudo/results':wandb.Table(columns=keys,data=[[r[k] for k in keys] for r in rows])})
            run.summary.update(dict(complete=True,repeats=B,test_events=N,response_events=len(ew)))
    finally:
        path=output/'fixed_response_report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False))
        print('REPORT:',path,flush=True)
        if run:
            run.save(str(path),base_path=str(output));run.finish(exit_code=0 if report['complete'] else 1)


if __name__=='__main__':main()
