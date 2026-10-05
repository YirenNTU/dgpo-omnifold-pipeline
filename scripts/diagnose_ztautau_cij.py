"""Bridge Ztautau parquet + keyed H4 candidate bundle to paired Cij reports."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_reweighted_cij import analyze, plot_report, prepare

ANALYSIS_VERSION='matched-cij-v2'


def read_event_table(path, columns):
    """Do not let Arrow interpret shape_metadata.json as a parquet fragment."""
    import pyarrow.parquet as pq
    path=Path(path)
    files=sorted(path.glob('*.parquet')) if path.is_dir() else [path]
    if not files: raise ValueError(f'No parquet files in {path}')
    return pq.read_table([str(f) for f in files],columns=columns)


def boost(p,beta):
    """Active Lorentz boost, p=[E,px,py,pz]; rest frame uses negative velocity."""
    p=np.asarray(p,dtype=np.float64)
    beta=np.asarray(beta,dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(beta).all():
        raise ValueError('Nonfinite four-vector or boost')
    b2=(beta*beta).sum(-1)
    if (b2>=1).any(): raise ValueError('Non-timelike boost')
    gamma=1/np.sqrt(1-b2)
    dot=(p[...,1:]*beta).sum(-1)
    factor=np.divide(gamma-1,b2,out=np.zeros_like(b2),where=b2>0)*dot+gamma*p[...,0]
    return np.concatenate(((gamma*(p[...,0]+dot))[...,None],p[...,1:]+factor[...,None]*beta),-1)


def unit(v):
    norm=np.linalg.norm(v,axis=-1,keepdims=True)
    if not np.isfinite(v).all() or (norm<1e-12).any(): raise ValueError('Undefined direction / beam-collinear basis')
    return v/norm


def angles(ta,tb,va,vb):
    """NumPy equivalent of TT2L Core.analyze; common A helicity basis, k,r,n."""
    # Promote BEFORE forming beta or E^2-p^2: relativistic boosts amplify
    # float32 cancellation. Casting only the final cosines cannot repair it.
    ta,tb,va,vb=[np.asarray(p,dtype=np.float64) for p in (ta,tb,va,vb)]
    pair=ta+tb
    if (pair[...,0]<=0).any(): raise ValueError('Invalid parent energy')
    velocity=-pair[...,1:]/pair[...,0,None]
    a,b,ca,cb=[boost(p,velocity) for p in (ta,tb,va,vb)]
    k=unit(a[...,1:]); beam=np.zeros_like(k); beam[...,2]=1
    r=unit(beam-k[...,2,None]*k); n=unit(np.cross(r,k))
    basis=np.stack((k,r,n),-2)
    da=unit(boost(ca,-a[...,1:]/a[...,0,None])[...,1:])
    db=unit(boost(cb,-b[...,1:]/b[...,0,None])[...,1:])
    return np.einsum('...ij,...j->...i',basis,da),np.einsum('...ij,...j->...i',basis,db)


def tau_from_deltas(visible,delta):
    visible=np.asarray(visible,dtype=np.float64)
    delta=np.asarray(delta,dtype=np.float64)
    p=visible[...,1:]
    theta=np.arctan2(np.linalg.norm(p[...,:2],axis=-1),p[...,2])+delta[...,0]
    phi=np.arctan2(p[...,1],p[...,0])+delta[...,1]
    mag=np.sqrt((91.2/2)**2-1.777**2)
    return np.stack((np.full_like(theta,45.6),mag*np.sin(theta)*np.cos(phi),mag*np.sin(theta)*np.sin(phi),mag*np.cos(theta)),-1)


def calibrate(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    direction=unit(a[...,1:]-b[...,1:])
    p=np.sqrt(45.6**2-1.777**2)*direction
    return np.concatenate((np.full_like(p[...,:1],45.6),p),-1),np.concatenate((np.full_like(p[...,:1],45.6),-p),-1)


def ids(sample,event):
    return np.array([f'{s}:{e}' for s,e in zip(sample,event)])


SOURCE_ID_COLUMNS=('source_sample_index','source_event_key','source_file_index','source_event_index')


def source_ids(columns, keys=None):
    keys=keys or [k for k in SOURCE_ID_COLUMNS if k in columns]
    if not set(SOURCE_ID_COLUMNS[:2])<=set(keys): raise ValueError('Missing base source identity')
    arrays=[]
    for key in keys:
        a=np.asarray(columns[key]).reshape(-1)
        if a.dtype.kind not in 'iu':
            raise ValueError(f'{key} must retain integer precision, got {a.dtype}')
        arrays.append(a)
    if len({len(a) for a in arrays})!=1: raise ValueError('Source identity lengths differ')
    return np.array([':'.join(str(int(v)) for v in row) for row in zip(*arrays)])


def align(source_ids,wanted):
    if len(np.unique(source_ids))!=len(source_ids) or len(np.unique(wanted))!=len(wanted):
        raise ValueError('Duplicate composite event IDs')
    lookup={key:i for i,key in enumerate(source_ids)}
    missing=set(wanted)-set(lookup)
    if missing: raise ValueError(f'{len(missing)} candidate events absent from parquet')
    return np.array([lookup[key] for key in wanted])


def load_core(repo):
    # Use the user-supplied implementation only for a parity check, no ROOT/unfold import.
    sys.path.insert(0,str(repo.resolve()))
    from analysis_core.core import Core
    return Core


def check_core(core,p4s):
    import vector
    vectors=[vector.array(dict(E=p[:,0],px=p[:,1],py=p[:,2],pz=p[:,3])) for p in p4s]
    df=core(*vectors).analyze()
    reference=[df[[f'cos_theta_{leg}_{axis}' for axis in ('k','r','n')]].to_numpy() for leg in ('A','B')]
    actual=angles(*p4s)
    err=max(float(np.max(np.abs(x-y))) for x,y in zip(reference,actual))
    if not np.isfinite(err) or err>1e-7: raise ValueError(f'TT2L Core parity failed: {err}')
    return err


def truth_comparison(actual,stored,event_ids,tolerance):
    difference=np.abs(np.stack(actual,1)-np.stack(stored,1))
    if not np.isfinite(difference).all():raise ValueError('Nonfinite truth-angle comparison')
    worst=np.unravel_index(np.argmax(difference),difference.shape)
    i,leg,axis=worst
    return dict(arithmetic='float64',tolerance=float(tolerance),
        max_error=float(difference.max()),
        absolute_error_quantiles={str(q):float(np.quantile(difference,q)) for q in (.5,.95,.99,.999)},
        events_above_tolerance=int((difference.max(axis=(1,2))>tolerance).sum()),
        worst=dict(event_id=str(event_ids[i]),leg=('A','B')[leg],axis=('k','r','n')[axis],
                   recomputed=float(actual[leg][i,axis]),stored=float(stored[leg][i,axis])),
        passed=bool(difference.max()<=tolerance))


def reconstructed_targets(va,vb,ta,tb,truth_deltas):
    """Oracle references for measurement ONLY; never passed to a model/weight.

    The generator predicts tau-direction offsets, not tau energies or true
    visible momenta. Apply its exact observable map to the true offsets too.
    This is a matched target, not an irreducible lower bound on physics error.
    """
    oa=tau_from_deltas(va,truth_deltas[:,0])
    ob=tau_from_deltas(vb,truth_deltas[:,1])
    return dict(truth_tau_reco_visible=angles(ta,tb,va,vb),
                raw=angles(oa,ob,va,vb),
                calibrated=angles(*calibrate(oa,ob),va,vb))


def moment_diagnostics(data,kappas,mode):
    """Weight ESS is not the uncertainty of a moment divided by small kappas."""
    num,den,health=prepare(data,kappas,mode)
    normalized=den/den.sum(0)
    mean=num.sum(0)/den.sum(0)[:,None]
    # Event-level influence including the self-normalized denominator term.
    influence=(num-den[...,None]*mean[None])/den.sum(0)[None,:,None]
    absolute=np.abs(influence)
    take=max(1,int(np.ceil(.01*len(num))))
    total=absolute.sum(0)
    shares=np.divide(np.sort(absolute,axis=0)[-take:].sum(0),total,
                     out=np.zeros_like(total),where=total>0)
    inverse=np.abs(1/np.prod(kappas,axis=-1))
    def quantiles(x):
        return {str(q):float(np.quantile(x,q)) for q in (0,.01,.5,.95,.99,1)}
    return dict(
        analyzing_power_abs_quantiles={leg:quantiles(np.abs(kappas[:,i])) for i,leg in enumerate(('A','B'))},
        inverse_abs_product_quantiles=quantiles(inverse),
        negative_product_fraction=float((np.prod(kappas,axis=-1)<0).mean()),
        weight_health=health,
        base_weight_ess_fraction=float(1/(len(num)*np.square(normalized[:,0]).sum())),
        reweighted_event_mass_top1pct=float(np.sort(normalized[:,2])[-take:].sum()),
        influence_top1pct_fraction={name:shares[i].reshape(3,3).tolist()
            for i,name in enumerate(('stored_truth','unweighted','reweighted'))},
        interpretation='Per-cell top 1% absolute event influence; not a second ESS, not a cut. '
                       'Small analyzing powers can amplify variance even when ratio ESS is acceptable.',
        analyzing_power_definition='Upstream passthrough; decay-channel/polarimeter validity is not established by this script.')


def reference_decomposition(report):
    """Exact signed matrix accounting; no causal attribution from error norms."""
    refs=report['reference_comparisons']
    chain=[('stored_truth',np.asarray(report['results']['truth']['C']))]
    for name in ('recomputed_truth','truth_tau_reco_visible','matched_target'):
        chain.append((name,np.asarray(refs[name]['results']['truth']['C'])))
    shifts={f'{a}_to_{b}':(y-x).tolist() for (a,x),(b,y) in zip(chain,chain[1:])}
    target=chain[-1][1];truth=chain[0][1]
    closure={}
    for arm in ('unweighted','reweighted'):
        sample=np.asarray(report['results'][arm]['C'])
        closure[arm]=dict(sample_minus_matched=(sample-target).tolist(),
                          matched_minus_stored=(target-truth).tolist(),
                          max_accounting_error=float(np.max(np.abs((sample-target)+(target-truth)-(sample-truth)))))
    return dict(shifts=shifts,closure=closure,
        scope='Exact signed C-matrix decomposition. Frobenius norms do not add; '
              'matched-target gap is not an irreducible physics floor or proof of a cause.')


def run(args):
    required=['source_sample_index','source_event_key','event_weight','analyzing_power_a','analyzing_power_b']
    if getattr(args,'channel_diagnostics',False):
        required.append('event_category')
    for prefix in ('lead_a_visible','lead_b_visible','truth_tau_a','truth_tau_b','truth_a_visible','truth_b_visible'):
        required += [prefix+'_'+c for c in ('E','px','py','pz')]
    required += [f'truth_cos_theta_{leg}_{axis}' for leg in ('A','B') for axis in ('k','r','n')]
    if args.selection: required.append(args.selection)
    with np.load(args.candidates,allow_pickle=False) as f:
        bundle={k:f[k] for k in f.files}
    id_keys=[k for k in SOURCE_ID_COLUMNS if k in bundle]
    required+= [k for k in id_keys if k not in required]
    table=read_event_table(args.events,required)
    columns={k:np.asarray(table[k].to_numpy()) for k in required}
    needed={'source_sample_index','source_event_key','deltas','log_ratio','truth_deltas'}
    if not needed<=set(bundle): raise ValueError(f'Candidate bundle missing {needed-set(bundle)}')
    event_ids=source_ids(bundle,id_keys)
    order=align(source_ids(columns,id_keys),event_ids)
    columns={k:v[order] for k,v in columns.items()}
    selected=np.ones(len(order),dtype=bool) if not args.selection else columns[args.selection].astype(bool)
    if not selected.any(): raise ValueError('Empty selection')
    columns={k:v[selected] for k,v in columns.items()}
    bundle={k:v[selected] for k,v in bundle.items() if k in needed}
    event_ids=event_ids[selected]
    def p4(prefix): return np.stack([columns[prefix+'_'+c] for c in ('E','px','py','pz')],-1).astype(np.float64)
    va,vb,ta,tb,tva,tvb=[p4(p) for p in ('lead_a_visible','lead_b_visible','truth_tau_a','truth_tau_b','truth_a_visible','truth_b_visible')]
    n=len(va); delta=bundle['deltas']
    if delta.ndim!=4 or delta.shape[0]!=n or delta.shape[2:]!=(2,2) or not np.isfinite(delta).all():
        raise ValueError('deltas must be finite [N,K,2,2], physical radians in a,b order')
    expected=np.stack((np.stack((np.arctan2(np.linalg.norm(ta[:,1:3],axis=-1),ta[:,3])-np.arctan2(np.linalg.norm(va[:,1:3],axis=-1),va[:,3]),np.arctan2(ta[:,2],ta[:,1])-np.arctan2(va[:,2],va[:,1])),-1),
                       np.stack((np.arctan2(np.linalg.norm(tb[:,1:3],axis=-1),tb[:,3])-np.arctan2(np.linalg.norm(vb[:,1:3],axis=-1),vb[:,3]),np.arctan2(tb[:,2],tb[:,1])-np.arctan2(vb[:,2],vb[:,1])),-1)),1)
    if bundle['truth_deltas'].shape!=(n,2,2) or not np.isfinite(bundle['truth_deltas']).all():
        raise ValueError('truth_deltas must be finite [N,2,2]')
    difference=bundle['truth_deltas']-expected
    difference[...,1]=np.arctan2(np.sin(difference[...,1]),np.cos(difference[...,1]))
    if np.max(np.abs(difference))>args.tolerance:
        raise ValueError('Truth delta check failed: wrong event/leg order, normalization or source')
    # The production calculation is entirely repository-local NumPy code.
    # External TT2L is an optional development parity check, not a dependency.
    reference_repo=getattr(args,'tt2l_repo',None)
    core_error=(check_core(load_core(reference_repo),[p[:min(64,n)] for p in (ta,tb,tva,tvb)])
                if reference_repo is not None else None)
    truth=angles(ta,tb,tva,tvb)
    stored=[np.stack([columns[f'truth_cos_theta_{leg}_{axis}'] for axis in ('k','r','n')],-1) for leg in ('A','B')]
    comparison=truth_comparison(truth,stored,event_ids,args.tolerance)
    comparison['source_p4_dtypes']={k:str(v.dtype) for k,v in columns.items()
        if k.startswith(('truth_tau_','truth_a_visible_','truth_b_visible_'))}
    args.output.mkdir(parents=True,exist_ok=True)
    comparison_path=args.output/'truth_convention_check.json'
    comparison_path.write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    truth_error=comparison['max_error']
    print('TRUTH CHECK:',json.dumps(comparison),flush=True)
    mismatch_allowed=bool(getattr(args,'allow_truth_mismatch',False))
    comparison['mismatch_explicitly_allowed']=mismatch_allowed
    comparison['truth_used']='stored parquet truth cosines; no replacement or event removal'
    comparison_path.write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
    if not comparison['passed'] and not mismatch_allowed:
        raise ValueError(f'Truth-angle mismatch after float64 calculation: max error {truth_error}; '
                         f'see {comparison_path}. Precision or convention cause unresolved; Cij not computed.')
    if not comparison['passed']:
        print('WARNING: proceeding with stored truth despite unresolved truth-angle mismatch (explicit override).',flush=True)
    sa=tau_from_deltas(va[:,None],delta[:,:,0]); sb=tau_from_deltas(vb[:,None],delta[:,:,1])
    kappas=np.stack([columns['analyzing_power_a'],columns['analyzing_power_b']],-1)*np.array(args.kappa_signs)
    targets=reconstructed_targets(va,vb,ta,tb,bundle['truth_deltas'])
    args.output.mkdir(parents=True,exist_ok=True)
    reports={}
    for name,parents in [('raw',(sa,sb)),('calibrated',calibrate(sa,sb))]:
        aa,bb=angles(*parents,va[:,None],vb[:,None])
        data=dict(event_id=event_ids,truth_a=stored[0],truth_b=stored[1],sample_a=aa,sample_b=bb,log_ratio=bundle['log_ratio'],event_weight=columns['event_weight'])
        refs=dict(recomputed_truth=truth,truth_tau_reco_visible=targets['truth_tau_reco_visible'],
                  matched_target=targets[name])
        if getattr(args,'channel_diagnostics',False):
            from scripts.cij_channel_diagnostics import channel_diagnostics
            channel_report=channel_diagnostics(data,kappas,columns['event_category'],
                args.weight_mode,args.bootstrap,args.seed,references=refs)
            channel_report['provenance']=dict(events=str(args.events),candidates=str(args.candidates),
                selection=args.selection,calibration=name,kappa_signs=args.kappa_signs,
                truth_angle_mismatch_allowed=mismatch_allowed,truth_angle_max_error=truth_error)
            (args.output/f'{name}_channels.json').write_text(json.dumps(channel_report,indent=2,allow_nan=False)+'\n')
            print('CHANNEL DECOMPOSITION',name,json.dumps(channel_report['decomposition']),flush=True)
        print(f'CIJ {name}: {n} events, {delta.shape[1]} candidates; paired bootstrap {args.bootstrap}',flush=True)
        report=analyze(data,kappas,args.weight_mode,args.bootstrap,args.seed,references=refs)
        report['analysis_version']=ANALYSIS_VERSION
        report['reference_label']='Stored full truth'
        report['provenance']=dict(events=str(args.events),candidates=str(args.candidates),selection=args.selection,
            tt2l_core_max_error=core_error,external_reference_checked=reference_repo is not None,
            angle_implementation='ml_pipeline/scripts/diagnose_ztautau_cij.py:angles',
            stored_truth_max_error=truth_error,truth_convention_passed=comparison['passed'],
            truth_mismatch_allowed=mismatch_allowed,truth_source='stored parquet truth cosines',
            kappa_signs=args.kappa_signs,
            estimator='9 * weighted mean(cosA*cosB / (signed kappaA*kappaB)); explicit inverse-product estimator',
            calibration=name,weights='same candidate weights in both reconstructions; no rescoring',
            target_definition='Same reconstructed visible p4, fixed E_tau=45.6 and m_tau=1.777 GeV, '
                'same raw/calibrated map; replace only generated deltas with saved physical truth deltas',
            generator_target='Tau direction offsets, NOT a neutrino four-vector or a learned tau energy',
            basis='TT2L common A helicity basis; fixed +z beam axis in pair CM; no additional charge/axis flips',
            acceptance='No acceptance correction/unfolding; selected-sample moment comparison only')
        report['diagnostics']=moment_diagnostics(data,kappas,args.weight_mode)
        report['reference_decomposition']=reference_decomposition(report)
        report['limitations'] += [
            'Stored full-truth comparison mixes generator, visible reconstruction and fixed-energy effects.',
            'Matched target tests the implemented direction-to-observable map, not fully unfolded physics Cij.',
            'K=1 joint weighting changes the empirical event mixture; it is not eventwise conditional normalization.',
            'Stored analyzing powers are unchanged covariates; a ratio only guarantees moments of variables it correctly models.',
        ]
        (args.output/(name+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        plot_report(report,args.output/(name+'.png'))
        matched={**report,**report['reference_comparisons']['matched_target'],
                 'reference_label':'Matched truth target',
                 'provenance':{**report['provenance'],
                    'truth_source':'saved physical truth deltas through the same reconstruction map as samples',
                    'stored_truth_reference_retained_in':name+'.json'},
                 'plot_title':f'{name}: matched reconstruction Cij moments — no unfolding'}
        (args.output/(name+'_matched.json')).write_text(json.dumps(matched,indent=2,allow_nan=False)+'\n')
        plot_report(matched,args.output/(name+'_matched.png'))
        # Sensitivity only: remove the inverse-power amplification, NOT the
        # physics definition in the main result. This is 9<cosA cosB>, NOT Cij.
        angular=analyze(data,[1,1],args.weight_mode,args.bootstrap,args.seed,references=refs)
        angular.update(analysis_version=ANALYSIS_VERSION,
                       provenance={**report['provenance'],
                           'estimator':'9 * weighted mean(cosA*cosB); no analyzing-power division'},
                       observable='9 * mean(cosA_i*cosB_j); no analyzing-power division, NOT physical Cij',
                       plot_title=f'{name}: angular moments only — NOT physical Cij')
        (args.output/(name+'_angular_moments.json')).write_text(json.dumps(angular,indent=2,allow_nan=False)+'\n')
        reports[name]=report['comparison']
        reports[name+'_matched']=matched['comparison']
    print(json.dumps({key:value['C_frobenius_error_change'] for key,value in reports.items()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=Path,required=True,help='Converted parquet file or dataset directory')
    p.add_argument('--candidates',type=Path,required=True,help='Keyed NPZ; physical deltas and log weights in identical candidate order')
    p.add_argument('--tt2l-repo',type=Path,help='Optional development parity check; not needed on NERSC')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',help='Predeclared boolean cut column; applied to all arms')
    p.add_argument('--channel-diagnostics',action='store_true',help='Channel closure and fixed-channel-mixture comparison from saved scores')
    p.add_argument('--kappa-signs',type=int,nargs=2,choices=[-1,1],required=True,help='Explicit conversion from stored analyzer powers to signed A/B convention')
    p.add_argument('--weight-mode',choices=['joint','conditional'],required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tolerance',type=float,default=1e-4)
    p.add_argument('--allow-truth-mismatch',action='store_true',
                   help='Explicit exploratory override; retain stored truth and record unresolved finite angle mismatch')
    run(p.parse_args())


if __name__=='__main__': main()
