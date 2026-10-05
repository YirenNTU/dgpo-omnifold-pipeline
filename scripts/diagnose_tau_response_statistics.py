"""Fixed-test, fixed-binning response-size diagnostic. No new model samples."""
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
from scripts.tau_unfold_core import SVDResponse, indices, moment, response_edges, nested_sets
from scripts.tau_bias_sampling import AXES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--no-wandb',action='store_true')
    parser.add_argument('--response-source',type=Path,help='Completed full-validation-remainder sample directory; fixed original test is retained')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if args.response_source and not (args.response_source/'manifest.json').exists():
        ready=json.loads((args.response_source/'ready_samples.json').read_text())
        if ready.get('complete') is not True:raise ValueError('Expanded samples not ready')
        args.response_source=Path(ready['sample_directory'])
    fractions=[.25,.5,1.];repeats=int(cfg['response_bootstraps'])
    if repeats<2:raise ValueError('Need at least two response bootstraps')
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT,'RooUnfoldSvd'):ROOT.gSystem.Load(cfg.get('roounfold_library') or 'libRooUnfold')
    if not hasattr(ROOT,'RooUnfoldSvd'):raise RuntimeError('RooUnfold unavailable')
    panel,products,source=load_samples(Path(cfg['sample_source']))
    w=np.asarray(panel['event_weight'],float);kap=np.asarray(panel['kappas'],float)
    groups,group=np.unique(kap,axis=0,return_inverse=True)
    if not np.isfinite(w).all() or (w<0).any() or not w.sum()>0:raise ValueError('Invalid MC weights')
    rng=np.random.default_rng(cfg['seed']);response=[];test=[]
    for g in range(len(groups)):
        ii=np.flatnonzero(group==g);rng.shuffle(ii);cut=int(len(ii)*cfg['response_fraction'])
        response.extend(ii[:cut]);test.extend(ii[cut:])
    response=np.sort(response);test=np.sort(test)
    external_source=None
    if args.response_source:
        extra,extra_products,external_source=load_samples(args.response_source)
        if external_source.get('population')!='filtered_full_validation_remainder':
            raise ValueError('External response must be verified full-validation remainder')
        for arm in ('pretrain','dgpo'):
            a,b=source['arms'][arm],external_source['arms'][arm]
            if (Path(a['checkpoint']).resolve()!=Path(b['checkpoint']).resolve()
                    or a['global_step']!=b['global_step'] or a['epoch']!=b['epoch']):
                raise ValueError('Response model differs from the fixed test model')
        for key in ('packing_spec','ddim_steps','process_label','weights'):
            if source.get(key)!=external_source.get(key):raise ValueError(f'Response/test {key} differs')
        if np.intersect1d(panel['source_ids'],extra['source_ids']).size:
            raise ValueError('Extra response overlaps the original validation population')
        if not np.array_equal(np.unique(extra['kappas'],axis=0),groups):
            raise ValueError('External kappa groups differ; review population before comparing')
        ew=np.asarray(extra['event_weight'])
        if not np.isfinite(ew).all() or (ew<0).any() or ew.sum()<=0:raise ValueError('Invalid extra MC weights')
        nold=len(w)
        products={key:np.concatenate((products[key],extra_products[key])) for key in products}
        panel['source_ids']=np.concatenate((panel['source_ids'],extra['source_ids']))
        w=np.concatenate((w,extra['event_weight']));kap=np.concatenate((kap,extra['kappas']))
        groups,group=np.unique(kap,axis=0,return_inverse=True)
        response=np.arange(nold,len(w))
    if np.intersect1d(panel['source_ids'][response],panel['source_ids'][test]).size:raise ValueError('Split overlap')
    subsets=nested_sets(group,response,fractions,cfg['seed']+901)
    output=Path(cfg['output'])/('response-statistics-'+uuid.uuid4().hex[:10]);output.mkdir(parents=True)
    np.savez_compressed(output/'event_splits.npz',test_ids=panel['source_ids'][test],
                        **{f'response_{f}':panel['source_ids'][ii] for f,ii in subsets.items()})
    report=dict(complete=False,config=cfg,source=source,external_response_source=external_source,fractions=fractions,rows=[],binning={},
        test_events=len(test),max_response_events=len(response),bootstrap_repeats=repeats,
        scope='Fixed independent nominal test, nested stratified response subsets, common response-only bins and k. '
        'Bootstrap is within each subset; no new independent events. The test panel is fixed: offsets are closure '
        'residuals, not ensemble estimates of physical bias. Response-only bootstrap intervals exclude test '
        'statistics and systematics. Any failed replica invalidates that scale summary; no cherry-picking. '
        'Subset sizes and bootstrap draws are correlated. No full coverage claim.')
    run=None
    try:
        if not args.no_wandb:
            import wandb
            settings=dict(cfg['wandb']);settings['name']='Does more response data reduce offsets? | fixed test | 25-50-100% MC | no model update'
            run=wandb.init(**settings,config=dict(**cfg,response_sizes=fractions),dir=str(output))
        for j,axis in enumerate(AXES):
            # Same bins as the previous fold-0 diagnostic, not retuned per scale.
            if args.response_source:
                # Recover original fold-0 response IDs solely for the old binning.
                # New MC sizes must not change the measurement definition.
                brng=np.random.default_rng(cfg['seed']);original_response=[]
                for g in np.unique(group[:nold]):
                    ids=np.flatnonzero(group[:nold]==g);brng.shuffle(ids)
                    original_response.extend(ids[:int(len(ids)*cfg['response_fraction'])])
                edge_pool=np.sort(original_response)
            else:edge_pool=response
            edges,info=response_edges({a:v[edge_pool,j] for a,v in products.items()},group[edge_pool],
                w[edge_pool],cfg['bins'],cfg.get('min_response_entries',20))
            report['binning'][axis]=info
            nb=len(edges)-1;k=min(cfg['svd_k'],nb);centers=(edges[:-1]+edges[1:])/2
            coeff=((9/np.prod(groups,axis=1))[:,None]*centers[None,:]).ravel()
            bins={a:indices(v[:,j],edges) for a,v in products.items()}
            tw=w[test]*len(test)/w[test].sum()
            th=np.bincount(group[test]*nb+bins['truth'][test],weights=tw,minlength=len(groups)*nb)
            truth=float(th@coeff/th.sum())
            exact=float(np.average(9*products['truth'][test,j]/np.prod(kap[test],axis=1),weights=tw))
            for f,ii in subsets.items():
                # Paired response resampling across models/components and nested prefixes.
                brng=np.random.default_rng(cfg['seed']+902)
                full_mult=brng.poisson(1,(repeats,len(response)))
                mult=full_mult[:,np.searchsorted(response,ii)]
                for arm in ('pretrain','dgpo'):
                    measured=np.bincount(group[test]*nb+bins[arm][test],weights=tw,minlength=len(groups)*nb)
                    def estimate(multipliers):
                        values=[];cov=np.zeros((len(coeff),len(coeff)));condition=[]
                        for g in range(len(groups)):
                            mask=group[ii]==g;ids=ii[mask];weights=w[ids]*multipliers[mask]
                            b=SVDResponse(ROOT,bins['truth'][ids],bins[arm][ids],weights,nb)
                            block=slice(g*nb,(g+1)*nb);v,c=b.unfold(measured[block],k)
                            values.extend(v);cov[block,block]=c
                            joint=np.bincount(bins[arm][ids]*nb+bins['truth'][ids],weights=weights,minlength=nb*nb).reshape(nb,nb)
                            condition.append(np.linalg.cond(joint/joint.sum(axis=0)[None,:]))
                        value,sigma=moment(np.asarray(values),cov,coeff)
                        return value,sigma,float(max(condition))
                    failures=[];trials=[];nominal=None
                    try:nominal=estimate(np.ones(len(ii)))
                    except ValueError as exc:failures.append(dict(replica='nominal',error=str(exc)))
                    for b,m in enumerate(mult):
                        try:trials.append((b,*estimate(m)))
                        except ValueError as exc:failures.append(dict(replica=b,error=str(exc)))
                    row=dict(component=axis,arm=arm,response_fraction=f,response_events=len(ii),test_events=len(test),
                        bins=nb,k=k,truth_binned=truth,truth_exact=exact,binning_shift=truth-exact,
                        valid=not failures,failed_replicas=sum(x['replica']!='nominal' for x in failures),
                        nominal_failed=nominal is None,unfolded=None,residual_to_binned=None,sigma_stat=None,
                        sigma_response_mc=None,bootstrap_mean_residual=None,response_interval_lo=None,response_interval_hi=None,
                        max_condition_number=None)
                    if nominal is not None:
                        row.update(unfolded=nominal[0],residual_to_binned=nominal[0]-truth,
                                   sigma_stat=nominal[1],max_condition_number=nominal[2])
                    if not failures:
                        vals=np.asarray(trials)[:,1]
                        row.update(sigma_response_mc=float(vals.std(ddof=1)),bootstrap_mean_residual=float(vals.mean()-truth),
                            response_interval_lo=float(np.quantile(vals,.025)-truth),response_interval_hi=float(np.quantile(vals,.975)-truth))
                    report['rows'].append(row)
                    np.savez_compressed(output/f'{axis}_{arm}_{f}_replicas.npz',replicas=np.asarray(trials))
                    (output/f'{axis}_{arm}_{f}_failures.json').write_text(json.dumps(failures,indent=2))
                    print(f'{axis} {arm} size={f}: valid={row["valid"]} residual={row["residual_to_binned"]} MC sigma={row["sigma_response_mc"]}',flush=True)
                    if run:
                        run.log({'response_size/'+key:value for key,value in row.items() if isinstance(value,(int,float)) and np.isfinite(value)})
                        run.summary.update(dict(component=axis,arm=arm,response_fraction=f,completed_cases=len(report['rows'])))
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        for field in ('residual_to_binned','sigma_response_mc','sigma_stat'):
            fig=Figure(figsize=(12,10),layout='constrained');FigureCanvasAgg(fig)
            for ax,axis in zip(fig.subplots(3,3).ravel(),AXES):
                for arm in ('pretrain','dgpo'):
                    rows=[r for r in report['rows'] if r['component']==axis and r['arm']==arm]
                    ax.plot([r['response_events'] for r in rows],
                            [r[field] if r['valid'] else np.nan for r in rows],'o-',label=arm)
                    if field=='residual_to_binned':
                        good=[r for r in rows if r['valid']]
                        ax.fill_between([r['response_events'] for r in good],[r['response_interval_lo'] for r in good],
                                        [r['response_interval_hi'] for r in good],alpha=.12)
                if field=='residual_to_binned':ax.axhline(0,color='k',ls='--')
                ax.set(title=axis,xlabel='Response events',ylabel=field);ax.legend(fontsize=7)
            fig.suptitle('Fixed nominal test | missing points = invalid replicas | bands: response MC only')
            path=output/f'{field}.png';fig.savefig(path,dpi=140)
            if run:run.log({'response_size/plots/'+field:wandb.Image(str(path))})
        report['complete']=True
        if run:
            # Condition numbers may be infinite; represent them explicitly in the table.
            rows=report['rows'];cols=list(rows[0])
            run.log({'response_size/results':wandb.Table(columns=cols,data=[[str(r[c]) if isinstance(r[c],float) and not np.isfinite(r[c]) else r[c] for c in cols] for r in rows])})
            run.summary.update(dict(complete=True,invalid_cases=sum(not r['valid'] for r in rows)))
    finally:
        def clean(x):
            if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
            if isinstance(x,list):return [clean(v) for v in x]
            if isinstance(x,float) and not np.isfinite(x):return str(x)
            return x
        path=output/'response_statistics_report.json';path.write_text(json.dumps(clean(report),indent=2,allow_nan=False))
        print('REPORT:',path,flush=True)
        if run:
            run.save(str(path),base_path=str(output));run.finish(exit_code=0 if report['complete'] else 1)


if __name__=='__main__':main()
