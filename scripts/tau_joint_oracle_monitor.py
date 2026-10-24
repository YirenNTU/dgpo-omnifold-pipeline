"""Portable raw-policy geometry monitoring; NumPy only until plotting.

Observed visible p3 + physical angular offsets, ordered lab-frame legs. This is
not a spin basis or a full tau rest-frame reconstruction. Never applies weights.
"""
import json
from pathlib import Path
import numpy as np

DIRECT = ['leg_a_delta_theta','leg_a_delta_phi','leg_b_delta_theta','leg_b_delta_phi']
JOINT = ['cos_opening','acoplanarity','polar_sum']


def wrap(x):
    return np.arctan2(np.sin(x),np.cos(x))


def angle(a,b):
    return np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1))


def geometry(deltas,visible):
    d=np.asarray(deltas,dtype=float);v=np.asarray(visible,dtype=float)
    if d.ndim!=4 or d.shape[2:]!=(2,2) or v.shape!=(len(d),2,3):
        raise ValueError('Expected N,K,2,2 physical offsets and N,2,3 visible p3')
    if not np.isfinite(d).all() or not np.isfinite(v).all() or (np.linalg.norm(v,axis=-1)<=0).any():
        raise ValueError('Nonfinite offsets or undefined visible directions')
    vt=np.arctan2(np.hypot(v[...,0],v[...,1]),v[...,2]);vp=np.arctan2(v[...,1],v[...,0])
    th=vt[:,None]+d[...,0];ph=vp[:,None]+d[...,1]
    u=np.stack((np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)),-1)
    ua,ub=u[...,0,:],u[...,1,:]
    axis=ua-ub;norm=np.linalg.norm(axis,axis=-1,keepdims=True)
    valid=((th>0)&(th<np.pi)).all((1,2)) & (norm[...,0]>1e-12).all(1)
    # Undefined axes are exposed in the validity mask, never assigned a direction.
    axis=np.divide(axis,norm,out=np.full_like(axis,np.nan),where=norm>1e-12)
    deficit=angle(ua,-ub)
    signed=th.sum(-1)-np.pi
    values=dict(cos_opening=2*np.sin(deficit/2)**2-1,opening_deficit=deficit,
        acoplanarity=np.pi-abs(wrap(ph[...,0]-ph[...,1])),polar_sum=abs(signed),
        signed_polar=signed,axis=axis)
    for j,key in enumerate(DIRECT):
        x=d[...,j//2,j%2];values[key]=wrap(x) if j%2 else x
    for j,key in enumerate(('axis_x','axis_y','axis_z')):values[key]=axis[...,j]
    return values,valid


def summarize(current,truth,selected,edges):
    if not selected.any():raise ValueError('No common physical events')
    metrics={};hist={}
    for key,edge in edges.items():
        x=current[key][selected];t=truth[key][selected].ravel();e=np.asarray(edge)
        if not np.isfinite(x).all() or not np.isfinite(t).all():raise ValueError('Nonfinite selected geometry')
        full=np.r_[-np.inf,e,np.inf]
        p=np.histogram(x,full)[0]/x.size;q=np.histogram(t,full)[0]/len(t)
        circular=key in ('leg_a_delta_phi','leg_b_delta_phi')
        if circular:
            center=np.arctan2(np.sin(x).mean(1),np.cos(x).mean(1))
            residual=wrap(x-center[:,None]);err=wrap(x-t[:,None])
        else:
            residual=x-x.mean(1,keepdims=True);err=x-t[:,None]
        spread=np.sqrt(np.mean(residual**2,1))
        metrics[key]=dict(tv=float(abs(p-q).sum()/2),error_rms=float(np.sqrt(np.mean(err**2))),
            spread_rms=float(np.sqrt(np.mean(spread**2))),spread_median=float(np.median(spread)),
            spread_p10=float(np.quantile(spread,.1)),spread_zero_fraction=float(np.mean(spread<1e-10)),
            lower_tail=float(p[0]),upper_tail=float(p[-1]))
        hist[key]=dict(edges=e.tolist(),prediction=p.tolist(),truth=q.tolist())
    axis_error=angle(current['axis'][selected],truth['axis'][selected])
    ax=current['axis'][selected]
    metrics['axis']=dict(error_mean_deg=float(axis_error.mean()*180/np.pi),
        error_rms_deg=float(np.sqrt(np.mean(axis_error**2))*180/np.pi),
        chord_spread_rms=float(np.sqrt(np.mean(np.sum((ax-ax.mean(1,keepdims=True))**2,axis=-1)))))
    return metrics,hist


def export_joint_monitor(panel,draws,folder,root,step,spec):
    folder,root=Path(folder),Path(root)
    v=np.stack([panel['visible_'+leg][:,1:] for leg in 'ab'],1)
    ids=panel['source_ids'];td=panel['truth_deltas']
    if len(ids)!=119002 or len(np.unique(ids))!=len(ids) or draws.shape!=(len(ids),8,2,2):
        raise ValueError('Expected complete filtered K8 validation panel')
    proto=root/'joint_monitor_protocol.json'
    if proto.exists():
        if json.loads(proto.read_text())!=spec:raise ValueError('Joint monitor protocol changed')
    else:
        if step:raise ValueError('Missing joint step0 protocol')
        proto.write_text(json.dumps(spec,indent=2))
    basefolder=root/'eval-step-00000000'
    if step:
        if not (basefolder/'JOINT_COMPLETE').exists():raise ValueError('Missing complete raw joint baseline')
        with np.load(basefolder/'direct_samples.npz') as z:
            for key,value in [('source_ids',ids),('truth_deltas',td)]:np.testing.assert_array_equal(z[key],value)
            baseline=z['deltas']
        with np.load(basefolder/'joint_inputs.npz') as z:np.testing.assert_array_equal(z['visible_p3'],v)
    else:baseline=draws
    truth,tvalid=geometry(td[:,None],v);b,bvalid=geometry(baseline,v);p,pvalid=geometry(draws,v)
    selected=tvalid & bvalid & pvalid
    edges={**spec['edges'],'signed_polar':np.linspace(-.08,.08,41).tolist(),
           **{k:np.linspace(-1,1,41).tolist() for k in ('axis_x','axis_y','axis_z')}}
    bm,bh=summarize(b,truth,selected,edges);pm,ph=summarize(p,truth,selected,edges)
    ratios={key:{field:(pm[key][field]/bm[key][field] if bm[key][field]>0 else None)
        for field in pm[key] if 'spread' in field and 'fraction' not in field} for key in pm}
    report=dict(step=step,events_input=len(ids),events_selected=int(selected.sum()),
        invalid_truth=int((~tvalid).sum()),invalid_baseline=int((~bvalid).sum()),invalid_current=int((~pvalid).sum()),
        weighting='Raw uniform K8; no General or oracle reweighting',
        baseline=bm,current=pm,spread_ratio=ratios,histograms=ph,
        scope='Development validation; comparisons recomputed on common physical events. Marginal/spread checks do not establish spin closure or no collapse.')
    with open(folder/'joint_inputs.pending','wb') as f:
        np.savez_compressed(f,source_ids=ids,visible_p3=v,selected=selected)
    (folder/'joint_inputs.pending').replace(folder/'joint_inputs.npz')
    (folder/'joint_metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,4,figsize=(17,11),constrained_layout=True)
    for ax,key in zip(axs.flat,edges):
        h=ph[key];e=np.array(h['edges']);width=np.diff(e)
        ax.stairs(np.array(h['truth'])[1:-1]/width,e,color='black',label='MC target')
        ax.stairs(np.array(bh[key]['prediction'])[1:-1]/width,e,label='Epoch287',linestyle='--')
        ax.stairs(np.array(h['prediction'])[1:-1]/width,e,label=f'Step {step}')
        ax.set(title=key,xlabel='Value',ylabel='Density');ax.legend(fontsize=8)
    for ax in list(axs.flat)[len(edges):]:ax.axis('off')
    fig.savefig(folder/'joint_distributions.png',dpi=140);plt.close(fig)
    (folder/'JOINT_COMPLETE').write_text('Raw policy joint geometry; local RooUnfoldSvd evaluation pending\n')
    logged={f'tau/joint/{k}/{f}':val for k,row in pm.items() for f,val in row.items()}
    logged.update({f'tau/joint/{k}/{f}_over_initial':val for k,row in ratios.items() for f,val in row.items() if val is not None})
    logged['tau/joint/invalid_event_fraction']=float((~pvalid).mean())
    return logged
