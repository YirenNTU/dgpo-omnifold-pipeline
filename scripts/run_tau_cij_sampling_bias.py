"""16-GPU raw checkpoint sampling once, followed by paired truth-bin resampling."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts.tau_bias_sampling import sampling_scan, AXES, polarization_moments


def load_raw(model, source):
    expected = {k.removeprefix('model.'):v for k,v in source['state_dict'].items()}
    actual = model.state_dict()
    missing = set(actual)-set(expected)
    extra = {k for k in expected if k not in actual and not k.startswith('famo.w.')}
    if missing or extra:
        raise ValueError(f'Raw architecture mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    model.load_state_dict({k:expected[k] for k in actual}, strict=True)


@torch.no_grad()
def worker(cfg):
    import ray.train
    from ray.train.torch import get_device
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    from RL.DGPO_neutrino.conditional_tau_cycle import panel_batch
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from evenet.utilities.diffusion_sampler import DDIMSampler
    ctx = ray.train.get_context(); rank = ctx.get_world_rank(); world = ctx.get_world_size()
    device = get_device()
    global_config.load_yaml(cfg['runtime'])
    torch.set_float32_matmul_precision('highest')
    model = build_evenet_on_device(global_config, load_normalization_dict(global_config), device)
    source = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    load_raw(model, source); del source
    model.eval().requires_grad_(False)
    with np.load(cfg['panel'], allow_pickle=False) as f:
        panel = {k:f[k] for k in f.files}
    spec = EventPackingSpec.from_dict(cfg['packing_spec'])
    idx = np.arange(rank, len(panel['source_ids']), world)
    torch.manual_seed(cfg['seed']+rank)
    sampler = DDIMSampler(device=device)
    draws = []
    for start in range(0,len(idx),cfg['batch_size']):
        ii = idx[start:start+cfg['batch_size']]
        batch = panel_batch(panel, ii, spec, device)
        batch['x_invisible'] = torch.zeros(len(ii),2,2,device=device)
        batch['x_invisible_mask'] = torch.ones(len(ii),2,dtype=torch.bool,device=device)
        pred = generate_neutrino_candidates(model,batch,sampler,K=1,
            num_ddim_steps=cfg['ddim_steps'],device=device,parallel_chains=1)
        draws.append(pred[0].cpu().numpy())
        print(f'[{cfg["arm"]} rank={rank}] {start+len(ii)}/{len(idx)}',flush=True)
    np.savez(Path(cfg['output'])/f'{cfg["arm"]}-rank{rank:02d}.npz', positions=idx,deltas=np.concatenate(draws))
    ray.train.report({'events':len(idx)})


def analyze(output, cfg, run):
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, angles, tau_from_deltas, SOURCE_ID_COLUMNS
    from scripts.diagnose_reweighted_cij import features
    sample_dir = Path(cfg.get('sample_source',output))
    with np.load(sample_dir/'panel.npz',allow_pickle=False) as f:
        panel = {k:f[k] for k in f.files}
    n = len(panel['source_ids'])
    if n != 119002:
        raise ValueError('Expected the complete filtered validation panel (119002 events)')
    import pyarrow.parquet as pq
    first = next(Path(cfg['events']).glob('*.parquet'))
    names = pq.ParquetFile(first).schema_arrow.names
    keys = [k for k in SOURCE_ID_COLUMNS if k in names]
    columns = keys + [p+'_'+c for p in ('truth_tau_a','truth_tau_b','truth_a_visible','truth_b_visible') for c in ('E','px','py','pz')]
    table = read_event_table(cfg['events'],columns)
    raw = {k:np.asarray(table[k].to_numpy()) for k in columns}
    order = align(source_ids(raw,keys),panel['source_ids'])
    def p4(prefix): return np.stack([raw[prefix+'_'+c][order] for c in ('E','px','py','pz')],-1)
    full_a,full_b = angles(*[p4(p) for p in ('truth_tau_a','truth_tau_b','truth_a_visible','truth_b_visible')])
    va,vb = panel['visible_a'],panel['visible_b']
    def contribution(delta):
        a,b = angles(tau_from_deltas(va,delta[:,0]),tau_from_deltas(vb,delta[:,1]),va,vb)
        return features(a,b,panel['kappas']),polarization_moments(a,b,panel['kappas'])
    matched, matched_b = contribution(panel['truth_deltas'])
    truth = dict(full=features(full_a,full_b,panel['kappas']),matched=matched)
    moments = {'truth/full':polarization_moments(full_a,full_b,panel['kappas']),'truth/matched':matched_b}
    predictions = {}
    for arm in ('pretrain','dgpo'):
        path = sample_dir/f'{arm}_samples.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as f:
                if not np.array_equal(f['source_ids'],panel['source_ids']): raise ValueError('Sample ID mismatch')
                delta = f['deltas']
        else:
            delta = np.empty((n,2,2),np.float32); seen = []
            for rank in range(cfg['workers']):
                with np.load(sample_dir/f'{arm}-rank{rank:02d}.npz',allow_pickle=False) as f:
                    ii = f['positions']; delta[ii] = f['deltas']; seen.extend(ii.tolist())
            if sorted(seen) != list(range(n)): raise ValueError('Missing/duplicate sampling events')
            np.savez_compressed(output/f'{arm}_samples.npz',source_ids=panel['source_ids'],deltas=delta)
        predictions[arm],moments[arm] = contribution(delta)
    products = (full_a[:,:,None]*full_b[:,None,:]).reshape(n,9)
    report = sampling_scan(products,truth,predictions,panel['event_weight'],cfg['seed'],cfg['repeats'],
        amplitude=cfg.get('amplitude',.5),moments=moments,targets=cfg.get('targets'),
        min_ess_fraction=cfg.get('min_ess_fraction',.1))
    report['provenance'] = cfg
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scripts.tau_bias_plots import direct_plots
    for key,path in direct_plots(report,output).items():
        if run:
            import wandb
            run.log({key:wandb.Image(str(path))})
    valid_cases=[c for c in report['cases'] if 'C' in c]
    fig = Figure(figsize=(16,9),layout='constrained'); FigureCanvasAgg(fig)
    axes = fig.subplots(2,2)
    for row,ref in enumerate(truth):
        for col,metric in enumerate(('absolute','response')):
            ax = axes[row,col]
            for arm in predictions:
                values = [c['metrics'][f'{ref}/{arm}/{metric}']['mean'] for c in valid_cases]
                ax.plot(range(len(valid_cases)),values,'o-',label=arm)
            ax.set(title=f'{ref} | {metric} matrix error',ylabel='Frobenius norm (lower is better)')
            ax.set_xticks(range(len(valid_cases)),[c['case'] for c in valid_cases],rotation=90,fontsize=5)
            ax.legend()
    path = output/'matrix_errors.png'; fig.savefig(path,dpi=140)
    if run:
        import wandb
        run.log({'bias/matrix_errors':wandb.Image(str(path))})
    for ref in truth:
        fig = Figure(figsize=(12,11),layout='constrained'); FigureCanvasAgg(fig)
        axes = fig.subplots(3,3).ravel()
        for j,ax in enumerate(axes):
            lim = .01
            for arm in predictions:
                x = [np.array(c['delta_C']['truth/'+ref]).flat[j] for c in valid_cases]
                y = [np.array(c['delta_C'][arm]).flat[j] for c in valid_cases]
                ax.scatter(x,y,label=arm,s=18)
                lim = max(lim,np.max(np.abs(x)),np.max(np.abs(y)))
            ax.plot([-lim,lim],[-lim,lim],'k--',linewidth=.8)
            ax.set(title=AXES[j],xlabel='Truth delta C',ylabel='Generated delta C'); ax.legend()
        fig.suptitle(f'{ref} reference | all {len(valid_cases)} sampling variants | fixed raw models')
        path=output/f'response_{ref}.png'; fig.savefig(path,dpi=140)
        if run:
            import wandb
            run.log({f'bias/response/{ref}':wandb.Image(str(path))})
    if run:
        import wandb
        rows=[]
        for case in valid_cases:
            for key,m in case['metrics'].items():
                rows.append([case['case'],key,m['mean'],*m['resampling_interval95']])
        run.log({'bias/results':wandb.Table(columns=['case','metric','mean','resampling_lo95','resampling_hi95'],data=rows)})
        if cfg.get('targets') is not None:
            rows=[]
            for c in report['cases']:
                j=AXES.index(c['target_component']) if c.get('target_component') else None
                rows.append([c['case'],c.get('status'),c.get('requested_target'),c.get('expected_target'),
                    None if j is None or 'C' not in c else np.asarray(c['C']['truth/full']).ravel()[j],
                    c.get('sampling_ess_fraction'),c.get('max_probability'),c.get('top1pct_probability'),c.get('expected_unique_fraction'),c.get('reason','')])
            run.log({'bias/target_support':wandb.Table(columns=['case','status','requested_C','expected_C','sampled_C','ESS_fraction','max_probability','top1pct_mass','expected_unique_fraction','reason'],data=rows)})
            run.summary['bias/unsupported_cases']=sum(c.get('status')=='unsupported' for c in report['cases'])
            run.summary['bias/low_support_cases']=sum(c.get('status')=='low_support' for c in report['cases'])
        for ref in truth:
            for arm in predictions:
                value = np.linalg.norm(np.array(report['nominal_full_panel_C'][arm])-np.array(report['nominal_full_panel_C']['truth/'+ref]))
                run.summary[f'bias/nominal/{ref}/{arm}/absolute_error'] = float(value)
        run.save(str(output/'report.json'),base_path=str(output))
        run.summary['phase']='complete'
    print('FULL REPORT:',output/'report.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('--analysis-only',type=Path,help='Reuse completed samples; no Ray/inference')
    p.add_argument('--no-wandb',action='store_true')
    args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    reuse = args.analysis_only or cfg.get('reuse_samples')
    if not reuse and cfg['workers'] != 16: raise ValueError('Use 16 GPUs for inference')
    from evenet.dataset.filtered_data import validate_filtered_dataset
    if reuse:
        requested = cfg
        source=Path(reuse).resolve(strict=True)
        cfg=json.loads((source/'manifest.json').read_text())
        source=Path(cfg.get('sample_source',source)).resolve(strict=True)
        # Inherit ALL model/data/sampling provenance. Override analysis settings only.
        for key in ('amplitude','repeats','seed','wandb','targets','min_ess_fraction'):
            if key in requested: cfg[key]=requested[key]
        output=Path(requested.get('output',cfg['output']))/('analysis-'+uuid.uuid4().hex[:10])
        output.mkdir(parents=True)
        cfg.update(sample_source=str(source),output=str(output),inference_reused=True)
        (output/'manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    if validate_filtered_dataset(cfg['events'])['rows'] != 119002: raise ValueError('Wrong filtered validation population')
    if not reuse:
        output=Path(cfg['output'])/('probe-'+uuid.uuid4().hex[:10]); output.mkdir(parents=True)
        root=Path(cfg['dgpo_root'])
        shutil.copy2(root/'validation_panel.npz',output/'panel.npz')
        bundle=torch.load(root/'initial_tau_reward.pt',map_location='cpu',weights_only=False)
        cfg['packing_spec']=bundle['head']['packing_spec']; del bundle
        cfg['arms']={}
        for arm,checkpoint,runtime in [('dgpo',root/'checkpoints/last.ckpt',root/'runtime.yaml'),
                ('pretrain',Path(cfg['pretrain_checkpoint']),Path(cfg['pretrain_runtime']))]:
            resolved=checkpoint.resolve(strict=True)
            # Snapshot at launch: a running trainer cannot change this experiment's weights.
            snapshot=output/f'{arm}.ckpt'; shutil.copy2(resolved,snapshot)
            source=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
            if 'state_dict' not in source: raise ValueError('Missing raw state_dict')
            if arm=='dgpo' and not source.get('dgpo_checkpoint_version'): raise ValueError('Expected a DGPO checkpoint')
            if arm=='pretrain' and source.get('dgpo_checkpoint_version'): raise ValueError('Expected supervised pretrain')
            raw=yaml.safe_load(runtime.read_text())
            raw['compat']={'backend':'dgpo-evenet','repo_root':str(ROOT)}
            raw.setdefault('rl',{})['enabled']=True
            raw['options']['Training']['model_checkpoint_load_path']=str(snapshot)
            raw['options']['Training']['pretrain_model_load_path']=None
            runtime_copy=output/f'{arm}_runtime.yaml'; runtime_copy.write_text(yaml.safe_dump(raw))
            cfg['arms'][arm]=dict(checkpoint=str(snapshot),source_checkpoint=str(resolved),
                runtime=str(runtime_copy),global_step=int(source.get('global_step',-1)),epoch=int(source.get('epoch',-1)))
            del source
        cfg.update(weights='raw_state_dict_only',candidates=1,process_label=0,output=str(output),
                   scope='Overall pretrain-to-DGPO comparison; not isolated architecture causality; development validation')
        (output/'manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(cfg['arms'],indent=2),flush=True)
    run=None
    if not args.no_wandb:
        import wandb
        run=wandb.init(**cfg['wandb'],config=cfg,dir=str(output),job_type='sampling-bias')
    try:
        if not reuse:
            import ray
            from ray.train import RunConfig,ScalingConfig
            from ray.train.torch import TorchTrainer
            ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto',runtime_env={'env_vars':{
                'PYTHONPATH':os.pathsep.join([str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')])}})
            if ray.cluster_resources().get('GPU',0)<16: raise ValueError('Need 16 allocated GPUs')
            for arm,values in cfg['arms'].items():
                if run: run.summary['phase']='sampling_'+arm
                worker_cfg={**cfg,**values,'arm':arm,'panel':str(output/'panel.npz')}
                TorchTrainer(worker,train_loop_config=worker_cfg,
                    scaling_config=ScalingConfig(num_workers=16,use_gpu=True),
                    run_config=RunConfig(name=arm,storage_path=str(output/'ray_results'))).fit()
        if run: run.summary['phase']='sampling_bias_analysis'
        analyze(output,cfg,run)
    except BaseException:
        if run: run.finish(exit_code=1)
        raise
    else:
        if run: run.finish()


if __name__=='__main__': main()
