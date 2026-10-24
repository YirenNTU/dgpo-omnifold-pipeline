"""Paired held-out ratio correction: raw tau structure and calibrated Cij.

No test-side score fitting or model selection. Candidate normalization is
explicit: historical global ratios or a declared within-event-only experiment.
Full-MC Cij is a cross-map reference, never a substitute for matched closure.
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scripts.tau_tihsu_cij import AXES, EDGES, bin_indices, from_deltas, analyzing_powers
from scripts.tau_general_spin_angles import panel_protocol, target_spin_observables, prediction_spin_observables
from scripts.tau_tail_attribution import candidate_weights
from scripts.tau_candidate_reweighting import evaluation_panel, reweight_candidates, validate_weighting
from evenet_dgpo.RL.DGPO_neutrino.diagnostics.tau_marginals import observables, probability, distances


def products(obs):
    return np.stack([obs[f'cos_theta_A_{a}_times_cos_theta_B_{b}'] for a in AXES for b in AXES], -1)


def bin_contributions(x):
    if not np.isfinite(x).all() or np.any(np.abs(x)>1+1e-6):
        raise ValueError('Invalid spin projections/products')
    bins = bin_indices(x)
    valid = (bins>=0)&(bins<10)
    centers = (EDGES[:-1]+EDGES[1:])/2
    return np.where(valid, centers[np.clip(bins,0,9)],0), valid.astype(float)


def cij_report(panel, draws, logits, full_truth, bootstrap, seed, progress=None, *,
               reweighting_mode='global', event_weighting='dataset'):
    validate_weighting(reweighting_mode, event_weighting)
    panel = evaluation_panel(panel, event_weighting)
    w = np.asarray(panel['event_weight'],float)
    kap = analyzing_powers(panel['category'])
    np.testing.assert_allclose(kap,panel['kappas'],rtol=0,atol=1e-6)
    n,k = draws.shape[:2]
    if draws.shape != (len(w),k,2,2) or np.shape(logits)!=(n,k) or k<1 or bootstrap<20:
        raise ValueError('Wrong candidate/score/bootstrap layout')
    if np.max(logits)>np.log(30)+1e-5: raise ValueError('Score exceeds single cap30')
    target = products(target_spin_observables(panel))
    gen = products(prediction_spin_observables(panel, draws))
    q, log_mean = reweight_candidates(w, logits, reweighting_mode)
    native,_ = candidate_weights(w,np.zeros_like(logits))
    weights = {'unweighted':native, 'reweighted':q}
    t,tv = bin_contributions(target); f,fv = bin_contributions(np.asarray(full_truth))
    if np.shape(full_truth)!=(n,9): raise ValueError('Full-truth reference must be event-aligned nine products')
    g,gv = bin_contributions(gen)
    bank = {}; channels = {}; channel_weights = {}; comparisons = []
    for ch in np.unique(panel['category']):
        take = panel['category']==ch; key=str(int(ch)); size=int(take.sum())
        if w[take].sum()<=0: raise ValueError('Zero-weight decay channel')
        channel_weights[key] = float(w[take].sum()/w.sum())
        coefficient = 9/np.prod(kap[take][0])
        pairs = {'matched':(w[take,None]*t[take],w[take,None]*tv[take]),
                 'full_truth':(w[take,None]*f[take],w[take,None]*fv[take])}
        for arm,ww in weights.items():
            pairs[arm] = ((ww[take,:,None]*g[take]).sum(1),(ww[take,:,None]*gv[take]).sum(1))
        def estimate(parts, idx=None):
            out={}
            for name,(num,den) in parts.items():
                numerator,denominator=(num.sum(0),den.sum(0)) if idx is None else (num[idx].sum(0),den[idx].sum(0))
                if np.any(denominator<=0): raise ValueError('Empty in-range channel histogram')
                out[name]=coefficient*numerator/denominator
            return out
        nominal = estimate(pairs)
        channels[key] = dict(events=size,truth_mass=channel_weights[key],analyzing_powers=kap[take][0].tolist(),
            matrices={name:value.reshape(3,3).tolist() for name,value in nominal.items()},
            arms={name:dict(matched_error=float(np.linalg.norm(nominal[name]-nominal['matched'])),
                           full_truth_distance=float(np.linalg.norm(nominal[name]-nominal['full_truth'])),
                           channel_mass=float(weights[name][take].sum()),
                           signed_matched_delta=(nominal[name]-nominal['matched']).reshape(3,3).tolist())
                  for name in weights},
            reference_map_gap=float(np.linalg.norm(nominal['matched']-nominal['full_truth'])))
        bank[key]=(pairs,coefficient,size)
    rng=np.random.default_rng(seed)
    component_changes={key:[] for key in bank}
    for repeat in range(bootstrap):
        change=0.
        for key,(pairs,coefficient,size) in bank.items():
            idx=rng.integers(0,size,size)
            means={}
            for name,(num,den) in pairs.items():
                if name=='full_truth':continue
                denominator=den[idx].sum(0)
                if np.any(denominator<=0):raise ValueError('Empty bootstrap channel histogram')
                means[name]=coefficient*num[idx].sum(0)/denominator
            change+=channel_weights[key]*(np.linalg.norm(means['reweighted']-means['matched'])-
                                           np.linalg.norm(means['unweighted']-means['matched']))
            component_changes[key].append(np.abs(means['reweighted']-means['matched'])-
                                          np.abs(means['unweighted']-means['matched']))
        comparisons.append(change)
        if progress is not None and (repeat+1)%25==0:progress(repeat+1)
    errors={arm:sum(channel_weights[ch]*v['arms'][arm]['matched_error'] for ch,v in channels.items()) for arm in weights}
    delta=errors['reweighted']-errors['unweighted']; ci=np.quantile(comparisons,[.025,.975])
    for key,rows in component_changes.items():
        channels[key]['matched_abs_error_change_ci95']=np.quantile(rows,[.025,.975],axis=0).tolist()
    event=q.sum(1)
    base_event = w / w.sum()
    conditional = np.divide(q, event[:, None], out=np.zeros_like(q), where=event[:, None]>0)
    within_ess = 1 / np.square(conditional[event>0]).sum(1)
    report=dict(protocol=panel_protocol(panel), axes=list(AXES),bin_edges=EDGES.tolist(),channels=channels,
        source_kappa_convention=str(panel.get('source_kappa_convention','tihsu-charge-signed')),
        evaluation_kappa_convention='tihsu-charge-signed',
        fixed_channel_weights=channel_weights,matched_error=errors,
        change=dict(delta=float(delta),ci95=ci.tolist()),
        weighting=dict(mode=reweighting_mode, event_weighting=event_weighting,
            event_mass_max_abs_change=float(np.max(np.abs(event-base_event))),
            fixed_event_mass=reweighting_mode=='within_event',
            formula=('q[e,k] = base[e]/sum(base) * exp(score[e,k])/sum_j exp(score[e,j])'
                     if reweighting_mode=='within_event' else 'q[e,k] proportional to base[e]/K * exp(score[e,k])')),
        ratio=dict(event_ess=float(1/(event@event)),candidate_ess=float(1/np.square(q).sum()),
                   within_event_ess_mean=float(np.average(within_ess, weights=base_event[event>0])),
                   within_event_ess_min=float(within_ess.min()),
                   event_mass_max_abs_change=float(np.max(np.abs(event-base_event))),
                   top1pct_event_mass=float(np.sort(event)[-max(1,int(np.ceil(n*.01))):].sum()),
                   mean_ratio=float(np.exp(log_mean)),cap_fraction=float(np.mean(logits>np.log(29.7)))),
        primary='Fixed truth-channel-mass weighted mean of nine-component matched Cij Frobenius errors; paired change CI95 upper < 0.',
        uncertainty='Whole-event bootstrap stratified by decay channel; all K draws travel together. Conditional on fitted head and saved candidates; excludes training-seed uncertainty.',
        reference='Full MC truth uses actual tau/visible p4; its distance is descriptive across different reconstruction maps, not a closure claim.',
        scope='Filtered validation, calibrated pre-unfolding Cij; no ROOT, central QI selection, acceptance correction or concurrence claim.')
    from scripts.tau_general_spin_angles import ROOT_PROTOCOL
    if report['protocol'] == ROOT_PROTOCOL:
        report.update(target_map='Root exporter target: full MC tau p4 plus reconstructed visible, no target calibration',
            prediction_map='Root exporter evenet: offsets plus reconstructed visible, root common.py calibration',
            scope='Filtered validation, root target/evenet pre-unfolding Cij; no central QI cuts or unfolding.',
            map_limit='Target and prediction use different reconstruction maps; even perfect offsets need not have zero target distance.')
    return report,weights,target,gen


def distribution_report(panel,draws,weights,target,gen):
    truth=observables(panel,panel['truth_deltas'])
    series=[observables(panel,draws[:,j]) for j in range(draws.shape[1])]
    records={}; w=panel['event_weight']
    for name,values in truth.items():
        generated=np.stack([row[name] for row in series],1)
        if name.startswith('target/'):
            edges=np.linspace(-.2,.2,61)
        elif name.endswith('cos_opening'):edges=np.linspace(-1,1,61)
        elif name.endswith('_phi'):edges=np.linspace(-np.pi,np.pi,61)
        else:edges=np.linspace(0,np.pi,61)
        p=probability(values,w,edges)
        records[name]=dict(edges=edges.tolist(),truth=p.tolist(),arms={
            arm:dict(probability=(q:=probability(generated.ravel(),weight.ravel(),edges)).tolist(),**distances(q,p))
            for arm,weight in weights.items()})
    for name,transform in [('topology/cos_opening',lambda x:np.log10(np.maximum(1+x,1e-12))),
                           ('topology/delta_phi_to_pi',lambda x:np.log10(np.maximum(x,1e-12)))]:
        edges=np.linspace(-12,1,105); values=transform(truth[name])
        generated=transform(np.stack([row[name] for row in series],1)); p=probability(values,w,edges)
        records[name+'_log10']=dict(edges=edges.tolist(),truth=p.tolist(),arms={
            arm:dict(probability=(q:=probability(generated.ravel(),weight.ravel(),edges)).tolist(),**distances(q,p))
            for arm,weight in weights.items()})
    for j,axis in enumerate(a+b for a in AXES for b in AXES):
        p=probability(target[:,j],w,EDGES)
        records['spin_products/'+axis]=dict(edges=EDGES.tolist(),truth=p.tolist(),arms={
            arm:dict(probability=(q:=probability(gen[:,:,j].ravel(),weight.ravel(),EDGES)).tolist(),**distances(q,p))
            for arm,weight in weights.items()})
        for ch in np.unique(panel['category']):
            take=panel['category']==ch
            cp=probability(target[take,j],w[take],EDGES)
            records[f'spin_by_channel/{int(ch)}/{axis}']=dict(edges=EDGES.tolist(),truth=cp.tolist(),arms={
                arm:dict(probability=(cq:=probability(gen[take,:,j].ravel(),weight[take].ravel(),EDGES)).tolist(),
                         **distances(cq,cp)) for arm,weight in weights.items()})
    # All six offset pairs with explicit under/overflow on each axis.
    names=[name for name in truth if name.startswith('target/')]
    if len(names)!=4:raise ValueError('Expected all four target angles')
    edges=np.linspace(-.2,.2,25); side=len(edges)+1
    def index(x):return np.clip(np.searchsorted(edges,x,side='right'),0,side-1)
    for left,right in combinations(names,2):
        ti=index(truth[left])*side+index(truth[right])
        gi=index(np.stack([r[left] for r in series],1))*side+index(np.stack([r[right] for r in series],1))
        p=np.bincount(ti,weights=w,minlength=side*side);p=p/p.sum()
        arms={}
        for arm,weight in weights.items():
            q=np.bincount(gi.ravel(),weights=weight.ravel(),minlength=side*side);q=q/q.sum()
            arms[arm]=dict(probability=q.reshape(side,side).tolist(),**distances(q,p))
        records['joint/'+left.split('/')[-1]+'__'+right.split('/')[-1]]=dict(
            edges=edges.tolist(),truth=p.reshape(side,side).tolist(),arms=arms,
            overflow='First/last row and column contain under/overflow, included in TV/JSD.')
    return records


def write_report(output,panel,draws,logits,full_truth,bootstrap,seed,progress=None, *,
                 reweighting_mode='global', event_weighting='dataset'):
    out=Path(output)
    panel = evaluation_panel(panel, event_weighting)
    cij,weights,target,gen=cij_report(panel,draws,logits,full_truth,bootstrap,seed,progress,
        reweighting_mode=reweighting_mode, event_weighting=event_weighting)
    hist=distribution_report(panel,draws,weights,target,gen)
    np.savez(out/'candidate_weights.npz', source_ids=panel['source_ids'],
        event_mass=panel['event_weight']/np.sum(panel['event_weight']),
        unweighted=weights['unweighted'], reweighted=weights['reweighted'],
        reweighting_mode=np.array(reweighting_mode), event_weighting=np.array(event_weighting))
    summary={'cij/'+arm:value for arm,value in cij['matched_error'].items()}
    summary.update({'cij/change':cij['change']['delta'],'cij/change_ci_low':cij['change']['ci95'][0],
                    'cij/change_ci_high':cij['change']['ci95'][1],
                    'cij/supported_improvement':cij['change']['ci95'][1]<0,'policy_updates':0})
    for group in ('target','joint','spin_products'):
        for arm in weights:
            summary[group+'/'+arm+'_mean_tv']=float(np.mean([r['arms'][arm]['tv'] for name,r in hist.items() if name.startswith(group+'/')]))
    summary['tau/point_guard_passed']=all(summary[g+'/reweighted_mean_tv']<=summary[g+'/unweighted_mean_tv'] for g in ('target','joint'))
    for arm in weights:
        summary[f'spin_by_channel/{arm}_mean_tv']=float(np.mean([
            row['arms'][arm]['tv'] for name,row in hist.items() if name.startswith('spin_by_channel/')]))
    summary['spin/point_guard_passed']=summary['spin_by_channel/reweighted_mean_tv']<=summary['spin_by_channel/unweighted_mean_tv']
    topology_names = ['topology/'+name for name in
        ('cos_opening','delta_phi_to_pi','cos_opening_log10','delta_phi_to_pi_log10')]
    for name in topology_names:
        for metric in ('tv', 'jsd'):
            for arm in weights:
                summary[f'{name}/{arm}_{metric}'] = hist[name]['arms'][arm][metric]
            summary[f'{name}/change_{metric}'] = (hist[name]['arms']['reweighted'][metric]
                                                 - hist[name]['arms']['unweighted'][metric])
    summary['topology/point_guard_passed'] = all(summary[name+'/change_tv']<=1e-12 for name in topology_names)
    summary['supports_next_policy_test']=bool(summary['cij/supported_improvement'] and
        summary['tau/point_guard_passed'] and summary['spin/point_guard_passed'])
    if reweighting_mode == 'within_event':
        summary['supports_next_policy_test'] &= summary['topology/point_guard_passed']
    summary.update(reweighting_mode=reweighting_mode, evaluation_event_weighting=event_weighting,
                   event_mass_preserved=bool(cij['weighting']['event_mass_max_abs_change']<1e-12))
    summary.update({'ratio/'+k:v for k,v in cij['ratio'].items()})
    for ch,row in cij['channels'].items():
        for arm,values in row['arms'].items():
            for key in ('matched_error','full_truth_distance','channel_mass'):
                summary[f'channel_{ch}/{arm}/{key}']=values[key]
        for family,matrix in row['matrices'].items():
            for i,left in enumerate(AXES):
                for j,right in enumerate(AXES):
                    summary[f'channel_{ch}/{family}/C_{left}{right}']=matrix[i][j]
    report=dict(complete=True,cij=cij,summary=summary,
        distributions={name:{arm:{k:v for k,v in row.items() if k!='probability'} for arm,row in h['arms'].items()} for name,h in hist.items()},
        selection='Internal validation BCE only after >=1000 updates; external physics never selects weights.',
        tau_guard='Marginal/joint and per-channel spin-product mean-TV point estimates must not worsen. These guards have no confidence intervals and are not proof of distribution closure.',
        evaluation_population='Previously inspected filtered validation; exploratory, not a new blind final test.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'histograms.json').write_text(json.dumps(hist,allow_nan=False)+'\n')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    plots=[]
    groups=['target','reco','topology','spin_products']+[f'spin_by_channel/{ch}' for ch in cij['channels']]
    for group in groups:
        names=[name for name in hist if name.startswith(group+'/')]
        if not names:continue
        fig=Figure(figsize=(12,3*((len(names)+1)//2)),layout='constrained');FigureCanvasAgg(fig)
        axes=fig.subplots((len(names)+1)//2,2,squeeze=False).ravel()
        for ax,name in zip(axes,names):
            h=hist[name];edges=np.array(h['edges'])
            ax.stairs(np.array(h['truth'])[1:-1],edges,label='Matched target',color='black')
            for arm,row in h['arms'].items():ax.stairs(np.array(row['probability'])[1:-1],edges,label=arm)
            ax.set(title=name.replace('spin_by_channel/','channel '),ylabel='Probability per bin');ax.legend(fontsize=8)
        for ax in axes[len(names):]:ax.set_visible(False)
        path=out/(group.replace('/','-')+'.png');fig.savefig(path,dpi=140);plots.append(path)
    names=[name for name in hist if name.startswith('joint/')]
    fig=Figure(figsize=(12,3*len(names)),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(len(names),2,squeeze=False)
    for row,name in zip(axes,names):
        h=hist[name];p=np.array(h['truth'])
        ds=[np.array(h['arms'][arm]['probability'])-p for arm in weights];scale=max(np.max(np.abs(v)) for v in ds) or 1.
        for ax,arm,diff in zip(row,weights,ds):
            im=ax.imshow(diff.T,origin='lower',cmap='coolwarm',vmin=-scale,vmax=scale)
            ax.set(title=name+' '+arm+' minus target',xlabel='First angle bin incl. flow',ylabel='Second angle bin incl. flow')
            fig.colorbar(im,ax=ax)
    path=out/'tau_joint_differences.png';fig.savefig(path,dpi=140);plots.append(path)
    channels=list(cij['channels'])
    scale=max(np.max(np.abs(row['arms'][arm]['signed_matched_delta']))
              for row in cij['channels'].values() for arm in weights) or 1.
    for arm in weights:
        fig=Figure(figsize=(14,3.5*((len(channels)+3)//4)),layout='constrained');FigureCanvasAgg(fig)
        axes=fig.subplots((len(channels)+3)//4,4,squeeze=False).ravel()
        for ax,ch in zip(axes,channels):
            diff=np.array(cij['channels'][ch]['arms'][arm]['signed_matched_delta'])
            im=ax.imshow(diff,cmap='coolwarm',vmin=-scale,vmax=scale)
            ax.set(title=f'Channel {ch}: {arm} - matched',xticks=range(3),yticks=range(3),xticklabels=AXES,yticklabels=AXES)
            for i in range(3):
                for j in range(3):ax.text(j,i,f'{diff[i,j]:+.3f}',ha='center',va='center',fontsize=8)
            fig.colorbar(im,ax=ax)
        for ax in axes[len(channels):]:ax.set_visible(False)
        path=out/f'cij_{arm}_minus_matched.png';fig.savefig(path,dpi=140);plots.append(path)
    return report,plots
