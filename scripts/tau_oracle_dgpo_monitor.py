"""Raw-policy K8 angular monitoring; no classifier/oracle weighting in evaluation."""
import json
import shutil
from pathlib import Path
import numpy as np

TARGETS = ['leg_a_delta_theta','leg_a_delta_phi','leg_b_delta_theta','leg_b_delta_phi']


def inherit_monitor_protocol(parent, child):
    """Inherit fixed bins and baseline together; never refit bins on continuation."""
    parent,child=Path(parent),Path(child)
    source=parent/'direct_monitor_protocol.json'
    spec=json.loads(source.read_text())
    baseline=parent/'eval-step-00000000'
    if not all((baseline/f).is_file() for f in ('DIRECT_COMPLETE','direct_samples.npz','direct_metrics.json')):
        raise ValueError('Incomplete parent direct-monitor baseline')
    saved=json.loads((baseline/'direct_metrics.json').read_text())
    if saved['step']!=0 or len(spec['edges'])!=4:raise ValueError('Invalid step-zero monitor protocol')
    for i,key in enumerate(TARGETS):
        if spec['edges'][i]!=saved['histograms'][key]['edges']:
            raise ValueError('Parent monitor bins differ from step-zero baseline')
    child.mkdir(parents=True,exist_ok=True)
    link=child/'eval-step-00000000'
    if link.exists():
        if link.resolve()!=baseline.resolve():raise ValueError('Child baseline belongs to another run')
    else:link.symlink_to(baseline.resolve())
    destination=child/source.name
    if destination.exists():
        if json.loads(destination.read_text())!=spec:raise ValueError('Child monitor protocol differs')
    else:
        tmp=destination.with_suffix('.pending');shutil.copy2(source,tmp);tmp.replace(destination)

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def summarize(draws, truth, edges):
    n,k,_,_ = draws.shape
    if k != 8 or truth.shape != (n,2,2) or not np.isfinite(draws).all():
        raise ValueError('Require finite raw K8 policy samples')
    metrics, hist = {}, {}
    for j,key in enumerate(TARGETS):
        leg,coord = divmod(j,2)
        x,t = draws[:,:,leg,coord].astype(float),truth[:,leg,coord].astype(float)
        if coord:
            x,t = wrap(x),wrap(t)
            center = np.arctan2(np.sin(x).mean(1),np.cos(x).mean(1))
            residual = wrap(x-center[:,None])
        else:
            residual = x-x.mean(1,keepdims=True)
        spread = np.sqrt(np.mean(residual**2,axis=1))
        e = np.r_[-np.inf,edges[j],np.inf]
        p = np.histogram(x,bins=e)[0]/(n*k);q=np.histogram(t,bins=e)[0]/n
        metrics.update({f'{key}/tv':float(.5*abs(p-q).sum()),
            f'{key}/spread_rms_rad':float(np.sqrt(np.mean(spread**2))),
            f'{key}/spread_median_rad':float(np.median(spread)),
            f'{key}/spread_p10_rad':float(np.quantile(spread,.1))})
        hist[key]=dict(edges=list(edges[j]),prediction=p.tolist(),truth=q.tolist())
    return metrics,hist


def export_monitor(panel, draws, folder, root, step):
    folder,root=Path(folder),Path(root)
    n=len(panel['source_ids'])
    if n!=119002 or len(np.unique(panel['source_ids']))!=n:
        raise ValueError('Expected complete filtered validation population')
    protocol=root/'direct_monitor_protocol.json'
    if not protocol.exists():
        if step!=0:raise ValueError('Missing step-zero monitor protocol')
        edges=[]
        for j in range(4):
            t=panel['truth_deltas'][:,j//2,j%2]
            if j%2:t=wrap(t)
            low,high=np.quantile(t,[.005,.995]);edges.append(np.linspace(low,high,41).tolist())
        protocol.write_text(json.dumps(dict(edges=edges,method='RooUnfoldSvd',bins=8,kreg=5,
            response_bootstraps=300,seed=4203,weighting='Uniform K8; one total contribution/event'),indent=2))
    spec=json.loads(protocol.read_text())
    values,hist=summarize(draws,panel['truth_deltas'],spec['edges'])
    baseline=root/'eval-step-00000000/direct_metrics.json'
    if step:
        old=json.loads(baseline.read_text())['metrics']
        for key,value in list(values.items()):
            if 'spread_' in key and old[key]>0:values[key+'_over_initial']=value/old[key]
    # Atomic exports: COMPLETE is written only after both arrays and metrics.
    with open(folder/'direct_samples.pending','wb') as f:
        np.savez_compressed(f,deltas=draws,truth_deltas=panel['truth_deltas'],
            source_ids=panel['source_ids'],category=panel['category'],event_weight=panel['event_weight'])
    (folder/'direct_samples.pending').replace(folder/'direct_samples.npz')
    (folder/'direct_metrics.json').write_text(json.dumps(dict(step=step,metrics=values,histograms=hist),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,key in zip(axs.flat,TARGETS):
        h=hist[key];e=np.asarray(h['edges']);width=np.diff(e)
        ax.stairs(np.array(h['truth'])[1:-1]/width,e,label='MC target',color='black')
        ax.stairs(np.array(h['prediction'])[1:-1]/width,e,label=f'Policy step {step}')
        if step:
            h0=json.loads(baseline.read_text())['histograms'][key]
            ax.stairs(np.array(h0['prediction'])[1:-1]/width,e,label='Epoch287',linestyle='--')
        ax.set(title=key,xlabel='Offset [rad]',ylabel='Density');ax.legend()
    fig.savefig(folder/'direct_distributions.png',dpi=150);plt.close(fig)
    (folder/'DIRECT_COMPLETE').write_text('Raw unweighted policy samples; not oracle-selected\n')
    return {'tau/direct/'+k:v for k,v in values.items()}
