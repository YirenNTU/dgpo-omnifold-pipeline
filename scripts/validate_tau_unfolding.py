"""Validate the measurement tool, without changing either generator.

Exact inversion is a synthetic correctness oracle, NOT a recommended physics
estimator. SVD regularization effects are recorded separately from hard tests.
"""
import argparse
import json
from pathlib import Path
import sys
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'evenet_dgpo')]
import numpy as np
import yaml
from scripts.tau_unfold_core import SVDResponse, indices, moment, response_edges


def bank_from_matrix(ROOT, migration, truth):
    # Rows=reconstructed, columns=truth. Preserve non-symmetric orientation.
    r,t=np.indices(migration.shape)
    return SVDResponse(ROOT,t.ravel(),r.ravel(),(migration*truth[None,:]).ravel(),len(truth))


def check(label, actual, expected, rtol=1e-6, atol=1e-8):
    actual,expected=np.asarray(actual),np.asarray(expected)
    return dict(test=label,passed=bool(np.allclose(actual,expected,rtol=rtol,atol=atol)),
                max_absolute_error=float(np.max(np.abs(actual-expected))),rtol=rtol,atol=atol)


def synthetic(ROOT, repeats=300):
    tests=[]; diagnostics=[]
    truth=np.array([1200.,2400.,1800.,3000.])
    shifted=np.array([1800.,900.,2800.,1900.])
    matrix=np.array([[.8,.1,0,.05],[.2,.7,.15,0],[0,.2,.75,.2],[0,0,.1,.75]])
    for name,R in [('identity',np.eye(4)),('asymmetric_migration',matrix)]:
        bank=bank_from_matrix(ROOT,R,truth)
        for label,t in [('nominal',truth),('shifted',shifted)]:
            measured=R@t
            x,cov=bank.unfold(measured,4,method='invert')
            inv=np.linalg.inv(R); expected_cov=inv@np.diag(measured)@inv.T
            tests.append(check(name+'/'+label+'/counts',x,t))
            tests.append(check(name+'/'+label+'/covariance',cov,expected_cov))
            coeff=np.array([-6.,-2.,2.,6.])
            value,sigma=moment(x,cov,coeff)
            tests.append(check(name+'/'+label+'/moment',value,t@coeff/t.sum()))
            tests.append(check(name+'/'+label+'/covariance_symmetry',cov,cov.T))
            rng=np.random.default_rng(427)
            trials=[]
            for _ in range(repeats):
                toy=rng.poisson(measured)
                v,c=bank.unfold(toy,4,method='invert'); trials.append(moment(v,c,coeff))
            trials=np.asarray(trials); target=t@coeff/t.sum()
            diagnostics.append(dict(test=name+'/'+label+'/inverse_poisson',
                sigma_analytic=sigma,empirical_std=float(trials[:,0].std(ddof=1)),
                mean_minus_truth=float(trials[:,0].mean()-target),
                coverage68=float(np.mean(np.abs(trials[:,0]-target)<=trials[:,1]))))
            # SVD is regularized even at k=number of bins; not an exact inverse.
            for k in (2,4):
                v,c=bank.unfold(measured,k)
                m,s=moment(v,c,coeff)
                diagnostics.append(dict(test=name+'/'+label+'/svd',k=k,
                    moment=m,truth=target,bias=m-target,sigma=s,
                    relative_histogram_error=float(np.linalg.norm(v-t)/np.linalg.norm(t))))
    # Combined normalization across signed kappa groups, with full block covariance.
    coeff=np.concatenate((9*np.array([-.75,-.25,.25,.75]),
                          -9/.34*np.array([-.75,-.25,.25,.75])))
    counts=np.concatenate((truth,shifted)); cov=np.diag(counts)
    value,sigma=moment(counts,cov,coeff)
    tests.append(check('mixed_kappa/combined_mean',value,np.sum(counts*coeff)/counts.sum()))
    eps=.01; jac=[]
    for i in range(len(counts)):
        delta=np.zeros_like(counts);delta[i]=eps
        jac.append((moment(counts+delta,cov,coeff)[0]-moment(counts-delta,cov,coeff)[0])/(2*eps))
    tests.append(check('mixed_kappa/normalization_jacobian',sigma**2,np.array(jac)@cov@np.array(jac)))
    return tests,diagnostics


def real_closure(ROOT,cfg,fold=0):
    from scripts.run_tau_cij_unfolding import load_samples
    from scripts.tau_bias_sampling import AXES
    panel,products,provenance=load_samples(Path(cfg['sample_source']))
    w=np.asarray(panel['event_weight'],float); kap=np.asarray(panel['kappas'],float)
    groups,group=np.unique(kap,axis=0,return_inverse=True)
    rng=np.random.default_rng(cfg['seed']); response=[]; test=[]
    for g in range(len(groups)):
        ids=np.flatnonzero(group==g);rng.shuffle(ids);cut=int(len(ids)*cfg['response_fraction'])
        response.extend(ids[:cut]);test.extend(ids[cut:])
    response=np.sort(response);test=np.sort(test)
    if fold==1: response,test=test,response
    elif fold!=0: raise ValueError('Only the two complementary folds are supported')
    if np.intersect1d(panel['source_ids'][response],panel['source_ids'][test]).size:
        raise ValueError('Response/test identity overlap')
    rows=[]; diagnostics={}
    for j,axis in enumerate(AXES):
        edges,info=response_edges({a:v[response,j] for a,v in products.items()},group[response],
                                 w[response],cfg['bins'],cfg.get('min_response_entries',20))
        nb=len(edges)-1;centers=(edges[1:]+edges[:-1])/2
        idx={a:indices(v[:,j],edges) for a,v in products.items()}
        diagnostics[axis]=info
        for arm in ('pretrain','dgpo'):
            bank=[]; matrices=[]
            for g in range(len(groups)):
                ii=response[group[response]==g]
                bank.append(SVDResponse(ROOT,idx['truth'][ii],idx[arm][ii],w[ii],nb))
                joint=np.bincount(idx[arm][ii]*nb+idx['truth'][ii],weights=w[ii],minlength=nb*nb).reshape(nb,nb)
                matrices.append(joint/joint.sum(axis=0)[None,:])
            for population,ii in [('response_self',response),('independent_nominal',test)]:
                # Scale both truth and reco to the same expected event count.
                weights=w[ii]*len(ii)/w[ii].sum()
                truth=np.bincount(group[ii]*nb+idx['truth'][ii],weights=weights,minlength=len(groups)*nb)
                measured=np.bincount(group[ii]*nb+idx[arm][ii],weights=weights,minlength=len(groups)*nb)
                coeff=((9/np.prod(groups,axis=1))[:,None]*centers[None,:]).ravel()
                target=float(truth@coeff/truth.sum())
                exact=float(np.average(9*products['truth'][ii,j]/np.prod(kap[ii],axis=1),weights=weights))
                for k in sorted(set((min(cfg['svd_k'],nb),nb))):
                    values=[];cov=np.zeros((len(truth),len(truth)))
                    for g,b in enumerate(bank):
                        block=slice(g*nb,(g+1)*nb)
                        v,c=b.unfold(measured[block],k);values.extend(v);cov[block,block]=c
                    values=np.asarray(values);value,sigma=moment(values,cov,coeff)
                    row=dict(component=axis,arm=arm,population=population,bins=nb,k=k,fold=fold,
                        response_events=len(response),test_events=len(test),
                        unfolded=value,truth_binned=target,truth_exact=exact,
                        bias_to_binned=value-target,binning_shift=target-exact,sigma_stat=sigma,
                        normalization_ratio=float(values.sum()/truth.sum()),
                        relative_histogram_error=float(np.linalg.norm(values-truth)/np.linalg.norm(truth)),
                        minimum_cov_eigenvalue=float(np.linalg.eigvalsh((cov+cov.T)/2).min()),
                        max_response_condition_number=float(max(np.linalg.cond(m) for m in matrices)))
                    rows.append(row)
                    print(f'fold={fold} {axis} {arm} {population} k={k}: bias to binned={value-target:+.5f}; binning={target-exact:+.5f}',flush=True)
    return rows,diagnostics,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--synthetic-only',action='store_true')
    parser.add_argument('--no-wandb',action='store_true')
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text())
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if not hasattr(ROOT,'RooUnfoldSvd'): ROOT.gSystem.Load(cfg.get('roounfold_library') or 'libRooUnfold')
    if not hasattr(ROOT,'RooUnfoldInvert'): raise RuntimeError('RooUnfold SVD and Invert classes required')
    output=Path(cfg['output'])/('validation-'+uuid.uuid4().hex[:10]);output.mkdir(parents=True)
    run=None;report={'complete':False,'config':cfg}
    try:
        if not args.no_wandb:
            import wandb
            settings=dict(cfg['wandb']);settings['name']='Is the unfolding tool correct? | exact controls | nominal closure | no model update'
            run=wandb.init(**settings,config=cfg,dir=str(output))
        tests,diagnostic=synthetic(ROOT)
        report.update(tests=tests,synthetic_diagnostics=diagnostic,hard_tests_passed=all(t['passed'] for t in tests))
        for t in tests: print(t,flush=True)
        if not report['hard_tests_passed']:
            raise RuntimeError('Exact synthetic controls failed; do not interpret physics unfolding results')
        if not args.synthetic_only:
            rows=[];binning={}
            for fold in (0,1):
                fold_rows,fold_binning,source=real_closure(ROOT,cfg,fold=fold)
                rows.extend(fold_rows);binning[str(fold)]=fold_binning
            report.update(real_closure=rows,binning=binning,source=source)
            report['test_design']='Two complementary response/test folds. All 119002 events tested once per model. Folds share the finite source pool in opposite roles; not independent replicas and not 238004 unique events. Do not combine their uncertainties as independent.'
        report['complete']=True
        report['scope']='Exact inversion controls test API, response orientation and covariance. SVD outputs are diagnostics, not tuned parameters. Self closure is not physics validation. Covariance here is statistical only; real source MC statistics are not included.'
        if run:
            run.log({'validation/tests':wandb.Table(columns=list(tests[0]),data=[[t[k] for k in tests[0]] for t in tests])})
            keys=sorted(set().union(*(d.keys() for d in diagnostic)))
            run.log({'validation/synthetic_diagnostics':wandb.Table(columns=keys,data=[[d.get(k) for k in keys] for d in diagnostic])})
            if not args.synthetic_only:
                run.log({'validation/nominal_closure':wandb.Table(columns=list(rows[0]),data=[[r[k] for k in rows[0]] for r in rows])})
        if not args.synthetic_only:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from scripts.tau_bias_sampling import AXES
            fig=Figure(figsize=(12,5),layout='constrained');FigureCanvasAgg(fig)
            for ax,pop in zip(fig.subplots(1,2),('response_self','independent_nominal')):
                for arm in ('pretrain','dgpo'):
                    for fold,style in ((0,'o-'),(1,'s--')):
                        selected=[next(r for r in rows if r['component']==axis and r['arm']==arm
                            and r['fold']==fold and r['population']==pop and r['k']==min(cfg['svd_k'],r['bins'])) for axis in AXES]
                        ax.plot(AXES,[r['bias_to_binned'] for r in selected],style,
                            color='tab:blue' if arm=='pretrain' else 'tab:orange',label=f'{arm} fold {fold}')
                ax.axhline(0,color='k',ls='--');ax.set(title=pop,ylabel='Unfolded minus binned truth');ax.legend()
            fig.suptitle('Tool validation | configured SVD k | not a model-ranking plot')
            path=output/'nominal_closure.png';fig.savefig(path,dpi=140)
            if run:run.log({'validation/closure_plot':wandb.Image(str(path))})
        if run:
            run.summary.update(dict(complete=True,hard_tests_passed=True))
    finally:
        # JSON encodes nonfinite condition numbers as strings, not misleading numbers.
        def clean(x):
            if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
            if isinstance(x,list): return [clean(v) for v in x]
            if isinstance(x,float) and not np.isfinite(x): return str(x)
            return x
        path=output/'validation_report.json';path.write_text(json.dumps(clean(report),indent=2,allow_nan=False))
        print('VALIDATION REPORT:',path,flush=True)
        if run:
            run.summary['hard_tests_passed']=report.get('hard_tests_passed',False)
            run.save(str(path),base_path=str(output));run.finish(exit_code=0 if report['complete'] else 1)


if __name__=='__main__':main()
