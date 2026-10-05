"""Frozen saved samples -> independent response/test split -> RooUnfold uncertainty scan.

Selected-population moment study, not inclusive acceptance correction.
Poisson pseudo-experiments match the statistical covariance supplied to RooUnfold.
"""
import argparse
import json
from pathlib import Path
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'evenet_dgpo')]
import numpy as np
import yaml
from scripts.tau_bias_sampling import target_probabilities, AXES
from scripts.tau_unfold_core import SVDResponse, indices, moment, response_edges


def load_samples(source):
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, angles, tau_from_deltas, SOURCE_ID_COLUMNS
    import pyarrow.parquet as pq
    manifest = json.loads((source/'manifest.json').read_text())
    if 'sample_source' in manifest:
        source = Path(manifest['sample_source'])
    with np.load(source/'panel.npz', allow_pickle=False) as f:
        panel = dict(f)
    expanded=manifest.get('population')=='filtered_full_validation_remainder'
    expected=manifest.get('event_count') if expanded else 119002
    if expanded and manifest.get('complete') is not True:
        raise ValueError('Expanded sample generation is not complete')
    if not isinstance(expected,int) or expected<1 or len(panel['source_ids']) != expected or len(np.unique(panel['source_ids'])) != expected:
        raise ValueError('Saved population size/unique identities do not match its manifest')
    directory = Path(manifest['events'])
    if validate_filtered_dataset(directory)['rows'] != expected:
        raise ValueError('Require the verified filtered population matching the manifest')
    names = pq.ParquetFile(next(directory.glob('*.parquet'))).schema_arrow.names
    keys = [k for k in SOURCE_ID_COLUMNS if k in names]
    prefixes = ('truth_tau_a','truth_tau_b','truth_a_visible','truth_b_visible')
    table = read_event_table(directory, keys+[p+'_'+c for p in prefixes for c in ('E','px','py','pz')])
    raw = {k:np.asarray(table[k].to_numpy()) for k in table.column_names}
    order = align(source_ids(raw, keys), panel['source_ids'])
    p4 = lambda p: np.stack([raw[p+'_'+c][order] for c in ('E','px','py','pz')], -1)
    a,b = angles(*[p4(p) for p in prefixes])
    products = {'truth':(a[:,:,None]*b[:,None,:]).reshape(-1,9)}
    for arm in ('pretrain','dgpo'):
        merged = source/f'{arm}_samples.npz'
        if merged.exists():
            with np.load(merged, allow_pickle=False) as f:
                if not np.array_equal(f['source_ids'],panel['source_ids']):
                    raise ValueError('Sample IDs differ')
                delta = f['deltas']
        else:
            delta = np.empty((len(a),2,2)); seen=[]
            for rank in range(manifest['workers']):
                with np.load(source/f'{arm}-rank{rank:02d}.npz',allow_pickle=False) as f:
                    delta[f['positions']] = f['deltas']; seen.extend(f['positions'].tolist())
            if sorted(seen) != list(range(len(a))): raise ValueError('Missing/duplicate samples')
        va,vb = panel['visible_a'],panel['visible_b']
        aa,bb = angles(tau_from_deltas(va,delta[:,0]),tau_from_deltas(vb,delta[:,1]),va,vb)
        products[arm] = (aa[:,:,None]*bb[:,None,:]).reshape(-1,9)
    return panel, products, manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path); p.add_argument('--no-wandb',action='store_true')
    args=p.parse_args(); cfg=yaml.safe_load(args.config.read_text())
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT,'RooUnfoldSvd'):
        ROOT.gSystem.Load(cfg.get('roounfold_library') or 'libRooUnfold')
    if not hasattr(ROOT,'RooUnfoldSvd'): raise RuntimeError('Activate ROOT/RooUnfold or set roounfold_library')
    nbin,k = cfg['bins'],cfg['svd_k']
    if not 1 <= k <= nbin or cfg['repeats'] < 2 or cfg['response_bootstraps'] < 2:
        raise ValueError('Invalid SVD/repetition settings')
    panel, products, provenance = load_samples(Path(cfg['sample_source']))
    w=np.asarray(panel['event_weight'],float); kap=np.asarray(panel['kappas'],float)
    if not np.isfinite(kap).all() or (kap==0).any() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid weights/kappas')
    groups, group = np.unique(kap,axis=0,return_inverse=True)
    rng=np.random.default_rng(cfg['seed']); response=[]; test=[]
    if not 0 < cfg['response_fraction'] < 1: raise ValueError('Invalid split')
    for g in range(len(groups)):
        ids=np.flatnonzero(group==g); rng.shuffle(ids)
        cut=int(len(ids)*cfg['response_fraction'])
        if cut<2 or len(ids)-cut<2: raise ValueError('Insufficient decay-group events')
        response.extend(ids[:cut]); test.extend(ids[cut:])
    response=np.sort(response); test=np.sort(test)
    output=Path(cfg['output'])/('unfold-'+uuid.uuid4().hex[:10]); output.mkdir(parents=True)
    np.savez_compressed(output/'split.npz',response_ids=panel['source_ids'][response],test_ids=panel['source_ids'][test])
    (output/'manifest.json').write_text(json.dumps(dict(config=cfg,source=provenance),indent=2))
    run=None
    if not args.no_wandb:
        import wandb
        run=wandb.init(**cfg['wandb'],config=dict(**cfg,source=provenance),dir=str(output))
    rows=[]; binning={}
    try:
        G=len(groups); N=len(test)
        boot_rng=np.random.default_rng(cfg['seed']+1)
        bootstrap=boot_rng.poisson(1,(cfg['response_bootstraps'],len(response)))
        for j,axis in enumerate(AXES):
            edges,diagnostic=response_edges({a:v[response,j] for a,v in products.items()},
                group[response],w[response],cfg['bins'],cfg.get('min_response_entries',20))
            nbin=len(edges)-1; k=min(cfg['svd_k'],nbin)
            diagnostic.update(svd_k=k,kappa_pairs=groups.tolist())
            binning[axis]=diagnostic
            (output/'binning.json').write_text(json.dumps(binning,indent=2))
            print(f'[response {axis}] bins {cfg["bins"]} -> {nbin}; SVD k={k}; edges={edges.tolist()}',flush=True)
            centers=(edges[1:]+edges[:-1])/2
            coeff=((9/np.prod(groups,axis=1))[:,None]*centers[None,:]).ravel()
            bins={a:indices(v[:,j],edges) for a,v in products.items()}
            z=9*products['truth'][test,j]/np.prod(kap[test],axis=1)
            probabilities=[('nominal',None,w[test]/w[test].sum())]
            for target in cfg['targets']:
                prob,_=target_probabilities(z,w[test],target)
                probabilities.append((str(target),target,prob))
            for arm in ('pretrain','dgpo'):
                responses=[]; boot_responses=[]
                for g in range(G):
                    mask=group[response]==g; ii=response[mask]
                    def build(mult):
                        return SVDResponse(ROOT,bins['truth'][ii],bins[arm][ii],w[ii]*mult,nbin)
                    try:
                        responses.append(build(np.ones(len(ii))))
                        boot_responses.append([build(v[mask]) for v in bootstrap])
                    except ValueError as exc:
                        raise ValueError(f'{axis} {arm} kappa={groups[g].tolist()}: {exc}') from exc
                reco_index=group[test]*nbin+bins[arm][test]
                truth_index=group[test]*nbin+bins['truth'][test]
                def estimate(counts, bank):
                    values=[]; covariance=np.zeros((G*nbin,G*nbin))
                    for g,r in enumerate(bank):
                        block=slice(g*nbin,(g+1)*nbin)
                        x,c=r.unfold(counts[block],k); values.extend(x); covariance[block,block]=c
                    return moment(np.asarray(values),covariance,coeff)
                for label,target,prob in probabilities:
                    expected=N*np.bincount(reco_index,weights=prob,minlength=G*nbin)
                    truth_exact=float(prob@z)
                    truth_binned=float(prob@coeff[truth_index])
                    central,sigma=estimate(expected,responses)
                    response_values=[estimate(expected,[boot_responses[g][b] for g in range(G)])[0]
                                     for b in range(cfg['response_bootstraps'])]
                    response_sigma=float(np.std(response_values,ddof=1))
                    # Same random event counts for both arms: paired Poisson experiments.
                    trials_rng=np.random.default_rng(cfg['seed']+100+j)
                    trials=[]
                    for _ in range(cfg['repeats']):
                        event_counts=trials_rng.poisson(N*prob)
                        counts=np.bincount(reco_index,weights=event_counts,minlength=G*nbin)
                        trials.append(estimate(counts,responses))
                    trials=np.asarray(trials); sig=trials[:,1]
                    pulls=np.divide(trials[:,0]-truth_exact,sig,out=np.full(len(sig),np.nan),where=sig>0)
                    row=dict(component=axis,arm=arm,target=target,truth=truth_exact,truth_binned=truth_binned,
                        response_bins=nbin,svd_k=k,
                        binning_shift=truth_binned-truth_exact,unfolded=central,bias=central-truth_exact,
                        reco=float(prob@(9*products[arm][test,j]/np.prod(kap[test],axis=1))),
                        sigma_stat=sigma,sigma_response_mc=response_sigma,
                        sigma_stat_mc_quadrature=float(np.hypot(sigma,response_sigma)),
                        empirical_stat_std=float(trials[:,0].std(ddof=1)),
                        coverage68_stat_fixed_response=float(np.mean(np.abs(trials[:,0]-truth_exact)<=sig)),
                        mean_bias_trials=float(trials[:,0].mean()-truth_exact),
                        pull_mean=float(np.nanmean(pulls)),pull_std=float(np.nanstd(pulls,ddof=1)),
                        sampling_ess_fraction=float(1/(N*np.sum(prob**2))))
                    rows.append(row)
                    np.savez_compressed(output/f'{axis}_{arm}_{label}_trials.npz',estimates=trials,response_estimates=response_values)
                    (output/'report.json').write_text(json.dumps(dict(rows=rows,complete=False),indent=2,allow_nan=False))
                    if run:
                        run.log({'unfold/'+key:val for key,val in row.items() if isinstance(val,(float,int))})
                        run.summary.update(dict(current_component=axis,current_arm=arm,current_target=label,
                                                completed_cases=len(rows),phase='unfolding'))
                    print(f'{axis} {arm} target={label}: C={central:+.4f} truth={truth_exact:+.4f} stat={sigma:.4f} MC={response_sigma:.4f}',flush=True)
        report=dict(complete=True,rows=rows,config=cfg,source=provenance,binning=binning,response_events=len(response),test_events=N,
            scope='Fixed selected population and predictions. Full truth reference. Kappa-pair stratified SVD. '
            'Poisson pseudo-data; independent event split. Coverage is statistical-only conditional on fixed response. '
            'MC uncertainty is separate Poisson response bootstrap; quadrature excludes model/regularization systematics. '
            'No inclusive efficiency correction. No joint 9-component covariance claimed.')
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        for field in ('unfolded','bias','sigma_stat_mc_quadrature','coverage68_stat_fixed_response'):
            fig=Figure(figsize=(12,10),layout='constrained'); FigureCanvasAgg(fig)
            for ax,axis in zip(fig.subplots(3,3).ravel(),AXES):
                for arm in ('pretrain','dgpo'):
                    series=sorted([r for r in rows if r['component']==axis and r['arm']==arm and r['target'] is not None],key=lambda r:r['truth'])
                    ax.plot([r['truth'] for r in series],[r[field] for r in series],'o-',label=arm)
                if field=='unfolded': ax.plot([-.5,.5],[-.5,.5],'k--')
                if field=='bias': ax.axhline(0,color='k',ls='--')
                if field.startswith('coverage'): ax.axhline(.6827,color='k',ls='--'); ax.set_ylim(0,1)
                ax.set(title=axis,xlabel='Full truth C',ylabel=field); ax.legend(fontsize=7)
            path=output/f'{field}.png'; fig.savefig(path,dpi=140)
            if run: run.log({'unfold/plots/'+field:wandb.Image(str(path))})
        if run:
            run.log({'unfold/results':wandb.Table(columns=list(rows[0]),data=[[r[k] for k in rows[0]] for r in rows])})
            run.summary['complete']=True; run.summary['phase']='complete'
        print('FULL REPORT:',output/'report.json',flush=True)
    finally:
        if run:
            if (output/'report.json').exists(): run.save(str(output/'report.json'),base_path=str(output))
            if (output/'binning.json').exists(): run.save(str(output/'binning.json'),base_path=str(output))
            run.finish(exit_code=0 if sys.exc_info()[0] is None else 1)


if __name__=='__main__': main()
