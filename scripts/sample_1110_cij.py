"""Fresh step1110 raw-policy draws scored by the saved h4ratio1 classifier."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid
import math
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'evenet_dgpo'), str(ROOT)]
import numpy as np
import torch
import yaml
from scripts.diagnose_1110_cij import BASE, SOURCE, POLICY, export

RUN_NAME = 'Does H4 reweighting help? | step-1110 raw diffusion | full validation K=1'


def weight_summary(logits):
    values=np.asarray(logits,dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all(): raise ValueError('Invalid logits')
    w=np.exp(values-values.max());w/=w.sum()
    return dict(events=len(values),logit_mean=float(values.mean()),logit_std=float(values.std()),
                ess_fraction=float(1/(len(w)*np.square(w).sum())),
                max_weight_mass=float(w.max()),
                top1pct_mass=float(np.sort(w)[-max(1,math.ceil(.01*len(w))):].sum()))


def validation_pool(directory, bundle):
    """Read the full converted validation population, retaining source identities."""
    import pyarrow.parquet as pq
    from evenet.dataset.preprocess import unflatten_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EventPackingSpec, pack_event_inputs, event_identity_inputs)
    from RL.DGPO_neutrino.omnifold_ztautau.logit_calibration import identity_groups
    from scripts.diagnose_ztautau_cij import source_ids, SOURCE_ID_COLUMNS
    directory = Path(directory)
    metadata = json.loads((directory/'shape_metadata.json').read_text())
    spec = EventPackingSpec.from_dict(bundle['packing_spec'])
    needed = set(spec.shapes) | {'x_invisible','x_invisible_mask','source_sample_index','source_event_key'}
    needed |= {f'lead_{leg}_visible_{c}' for leg in ('a','b') for c in ('E','px','py','pz')}
    conditions=[]; truths=[]; samples=[]; events=[]; rows=0; rejected=0
    extra_ids={}; id_schema=None
    for file in sorted(directory.glob('*.parquet')):
        parquet = pq.ParquetFile(file)
        present=[k for k in SOURCE_ID_COLUMNS[2:] if k in parquet.schema_arrow.names]
        if id_schema is not None and present!=id_schema: raise ValueError('Inconsistent source identity columns across parquet files')
        id_schema=present
        needed.update(present)
        columns = [c for c in parquet.schema_arrow.names if c.split(':')[0] in needed]
        for rb in parquet.iter_batches(batch_size=2048,columns=columns):
            flat={k:np.asarray(v) for k,v in rb.to_pydict().items()}
            batch={k:torch.as_tensor(v) for k,v in unflatten_dict(flat,metadata).items()}
            valid=batch['x_invisible_mask'].reshape(len(rb),-1).bool().all(1)
            rows+=len(rb); rejected+=int((~valid).sum())
            if not valid.any(): continue
            batch={k:v[valid] for k,v in batch.items()}
            packed,_=pack_event_inputs(batch,spec)
            truth=batch['x_invisible'].reshape(-1,2,2).float()
            if not torch.isfinite(packed).all() or not torch.isfinite(truth).all():
                raise ValueError('Nonfinite validation inputs')
            conditions.append(packed);truths.append(truth)
            samples.append(batch['source_sample_index']);events.append(batch['source_event_key'])
            for key in present: extra_ids.setdefault(key,[]).append(batch[key])
    if not conditions: raise ValueError('No valid validation events')
    c=torch.cat(conditions); s=torch.cat(samples).reshape(-1); e=torch.cat(events).reshape(-1)
    extra_ids={k:torch.cat(v).reshape(-1) for k,v in extra_ids.items()}
    identity_columns=dict(source_sample_index=s,source_event_key=e,**extra_ids)
    keys=source_ids(identity_columns)
    unique,counts=np.unique(keys,return_counts=True)
    if (counts>1).any():
        examples=unique[counts>1][:5].tolist()
        raise ValueError(f'Duplicate full source identities; columns={list(identity_columns)}, '
                         f'duplicate_groups={int((counts>1).sum())}, examples={examples}. No rows were discarded.')
    current=identity_groups(event_identity_inputs(c,spec))
    if 'identity_condition' not in bundle:
        return dict(condition=c,truth=torch.cat(truths),**identity_columns,overlap={},
                    rows_read=rows,invalid_target_rows=rejected)
    historical=identity_groups(bundle['identity_condition'])
    test_groups=historical[np.asarray(bundle['split_indices']['test'])]
    if not np.isin(test_groups,current).all():
        raise ValueError('Historical test identities not recovered in validation: check source/packing before claiming held-out status')
    overlap={name:torch.from_numpy(np.isin(current,historical[np.asarray(idx)]))
             for name,idx in bundle['split_indices'].items()}
    return dict(condition=c,truth=torch.cat(truths),source_sample_index=s,source_event_key=e,
                **extra_ids,overlap=overlap,rows_read=rows,invalid_target_rows=rejected)


def merge_parts(parts, n, k):
    seen = torch.cat([p['positions'] for p in parts])
    if sorted(seen.tolist()) != list(range(n)):
        raise ValueError('Missing/duplicate event positions')
    draws = torch.empty(n, k, 2, 2)
    logits = torch.empty(n, k)
    for part in parts:
        ids = part['positions']
        if part['draws'].shape != (len(ids), k, 2, 2) or part['logits'].shape != (len(ids), k):
            raise ValueError('Invalid shard shape')
        if not torch.isfinite(part['draws']).all() or not torch.isfinite(part['logits']).all():
            raise ValueError('Nonfinite shard')
        draws[ids] = part['draws']; logits[ids] = part['logits']
    return draws.numpy(), logits.numpy()


@torch.no_grad()
def worker(cfg):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from diagnose_h4_spike_coverage import load_raw_state
    from diagnose_h4_classifier_calibration import build_classifier, score, verify_logits, original_shard

    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg['runtime'])
    raw = yaml.safe_load(Path(cfg['runtime']).read_text())
    torch.set_float32_matmul_precision(str(raw['dgpo'].get('float32_matmul_precision', 'medium')))
    b = torch.load(cfg['bundle'], map_location='cpu', weights_only=True)
    model = build_classifier(raw, b['model_state'], b['packing_spec'], device)
    start, stop = original_shard(len(b['test_truth']), rank, world)
    old_positions = torch.arange(start, stop)
    c = b['test_condition'][old_positions]
    # Verify reconstruction against historical held-out scores before scoring new draws.
    old = score(model, c, b['test_generated'][old_positions].reshape(-1,4), device, cfg['score_batch'])
    verify_logits(old, b['gen_logits'].reshape(-1)[old_positions])
    pool=torch.load(Path(cfg['output'])/'validation_pool.pt',map_location='cpu',weights_only=True)
    start,stop=original_shard(len(pool['truth']),rank,world)
    positions=torch.arange(start,stop)
    c=pool['condition'][positions]
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device).eval()
    source = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    load_raw_state(policy, source, 'step1110')
    del source
    policy.requires_grad_(False)
    spec = EventPackingSpec.from_dict(b['packing_spec'])
    sampler = DDIMSampler(device=device)
    torch.manual_seed(cfg['seed'] + rank)
    draws, logits = [], []
    started=time.monotonic()
    rounds=math.ceil(math.ceil(cfg['conditions']/world)/cfg['batch_size'])
    for round_index in range(rounds):
        offset=round_index*cfg['batch_size']
        condition = c[offset:offset+cfg['batch_size']]
        if not len(condition):
            ray.train.report({'sampling/rank0_events_done':len(c),
                              'sampling/rank0_fraction':1.,'sampling/batch_index':round_index+1})
            continue
        batch = unpack_event_inputs(condition.to(device), spec)
        batch['x_invisible'] = torch.zeros(len(condition),2,2,device=device)
        batch['x_invisible_mask'] = torch.ones(len(condition),2,dtype=torch.bool,device=device)
        generated = generate_neutrino_candidates(policy,batch,sampler,K=cfg['candidates'],
            num_ddim_steps=cfg['ddim_steps'],device=device,parallel_chains=1)
        draws.append(generated.permute(1,0,2,3).cpu())
        logits.append(torch.stack([score(model,condition,g.reshape(-1,4),device,cfg['score_batch'])
                                   for g in generated],1))
        print(f'[Cij rank={rank}] {offset+len(condition)}/{len(c)} events, K={cfg["candidates"]}',flush=True)
        ray.train.report({'sampling/rank0_events_done':offset+len(condition),
                          'sampling/rank0_fraction':(offset+len(condition))/len(c),
                          'sampling/batch_index':round_index+1,
                          'sampling/rank0_elapsed_seconds':time.monotonic()-started})
    torch.save(dict(positions=positions,draws=torch.cat(draws),logits=torch.cat(logits)),
               Path(cfg['output'])/f'rank-{rank:03d}.pt')
    ray.train.report({'events':len(c),'generated_samples':len(c)*cfg['candidates']})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--events',type=Path,default=BASE/'diffusion_val_20pct_seed42_stic_filtered_test1/val')
    p.add_argument('--output',type=Path,default=BASE/'h4_step1110_cij_fresh')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--candidates',type=int,choices=[1],default=1,
                   help='Exactly one fresh prediction per held-out condition')
    p.add_argument('--batch-size',type=int,default=1024,
                   help='Generation events per GPU batch (K=1); historical score verification retains its original batch size')
    p.add_argument('--seed',type=int,default=42929)
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    p.add_argument('--run-id',help='New W&B ID; omitted means a new run on every launch')
    p.add_argument('--no-wandb',action='store_true',help='Explicitly disable default online W&B logging')
    args=p.parse_args()
    if min(args.workers,args.candidates,args.batch_size)<1: p.error('Sizes must be positive')
    if args.output.resolve()==args.source.resolve(): p.error('Output cannot overwrite source')
    if not (args.source/'COMPLETE').is_file(): raise ValueError('Incomplete classifier export')
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    from diagnose_h4_spike_coverage import audit_ddim_steps
    raw=deep_update(read_yaml(ROOT/'config/train_diffusion_nersc.yaml'),
                    read_yaml(args.source.parent/'resolved_ratio_experiment.yaml'))
    raw=absolutize_default_paths(raw,ROOT/'config')
    if raw['options']['Training']['model_checkpoint_load_path']!=str(POLICY):
        raise ValueError('Not canonical step1110 source')
    raw.setdefault('compat',{}).update(backend='dgpo-evenet',repo_root=str(ROOT))
    raw.setdefault('rl',{})['enabled']=True
    bundle_path=args.source/'best_classifier_and_test.pt'
    b=torch.load(bundle_path,map_location='cpu',weights_only=True)
    if len(b['test_truth'])<args.workers: raise ValueError('More workers than conditions')
    # A unique attempt directory avoids mixing interrupted or simultaneous shards.
    output=args.output.resolve()/('sample-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    pool=validation_pool(args.events,b)
    torch.save(pool,output/'validation_pool.pt')
    n=len(pool['truth'])
    runtime=output/'runtime.yaml'
    runtime.write_text(yaml.safe_dump(raw,sort_keys=False))
    cfg=dict(runtime=str(runtime),bundle=str(bundle_path.resolve()),checkpoint=str(POLICY),output=str(output),
        workers=args.workers,candidates=args.candidates,batch_size=args.batch_size,seed=args.seed,
        score_batch=int(b['fit_config']['validation_batch_size']),ddim_steps=audit_ddim_steps(raw),
        conditions=n,total_samples=n*args.candidates,weights='raw_state_dict_only',classifier_fits=0,policy_updates=0,
        event_source=str(args.events.resolve()),condition_population='Full 10pct diffusion validation dataset; not wholly classifier-held-out',
        rows_read=pool['rows_read'],invalid_target_rows=pool['invalid_target_rows'],
        classifier_overlap_counts={k:int(v.sum()) for k,v in pool['overlap'].items()})
    (output/'manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(cfg,indent=2),flush=True)
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    ray.init(address=args.ray_address,runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(
        [str(ROOT/'evenet_dgpo'),str(ROOT/'scripts'),str(ROOT),os.environ.get('PYTHONPATH','')])}})
    if ray.cluster_resources().get('GPU',0)<args.workers: raise ValueError('Insufficient cluster GPUs')
    run=None
    callbacks=[]
    if not args.no_wandb:
        import wandb
        from ray.tune import Callback
        run=wandb.init(entity='ytchou97-university-of-washington',project='nu2flow-RL',
            id=args.run_id,resume='never',mode='online',name=RUN_NAME,
            group='H4 Cij reweighting',tags=['step1110','raw-only','K1','16-gpu' if args.workers==16 else 'multi-gpu','no-training'],
            config=cfg,dir=str(output))
        run.summary.update({'phase':'sampling','cij/computed':False})
        (output/'wandb.json').write_text(json.dumps({'id':run.id,'url':run.url})+'\n')
        print('WANDB:',run.url,flush=True)
        class Progress(Callback):
            def on_trial_result(self,iteration,trials,trial,result,**info):
                metrics={k:v for k,v in result.items() if k.startswith('sampling/')}
                if metrics:run.log(metrics)
        callbacks=[Progress()]
    try:
        finish_sampling(cfg,args,pool,output,n,run,callbacks)
    except BaseException:
        if run:
            run.summary['phase']='failed'
            run.finish(exit_code=1)
        raise
    else:
        if run:run.finish(exit_code=0)


def finish_sampling(cfg,args,pool,output,n,run,callbacks):
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    TorchTrainer(train_loop_per_worker=worker,train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=args.workers,use_gpu=True),
        run_config=RunConfig(name='fresh-cij',storage_path=str(output/'ray_results'),callbacks=callbacks,
                             failure_config=FailureConfig(max_failures=0))).fit()
    parts=[torch.load(output/f'rank-{i:03d}.pt',map_location='cpu',weights_only=True) for i in range(args.workers)]
    draws,logits=merge_parts(parts,n,args.candidates)
    arrays=dict(source_sample_index=pool['source_sample_index'].numpy(),
                source_event_key=pool['source_event_key'].numpy(),truth_deltas=pool['truth'].numpy(),
                **{k:pool[k].numpy() for k in ('source_file_index','source_event_index') if k in pool},
                **{f'classifier_{k}_overlap':v.numpy() for k,v in pool['overlap'].items()})
    np.savez_compressed(output/'candidates.npz',**arrays,deltas=draws,log_ratio=logits)
    heldout=~(pool['overlap']['fit']|pool['overlap']['early_stop']).numpy()
    if heldout.any():
        np.savez_compressed(output/'candidates_classifier_heldout.npz',
            **{k:v[heldout] for k,v in arrays.items()},deltas=draws[heldout],log_ratio=logits[heldout])
    summary={'full_validation':weight_summary(logits)}
    if heldout.any():summary['classifier_heldout']=weight_summary(logits[heldout])
    (output/'score_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if run:
        run.log({f'{population}/{k}':v for population,stats in summary.items() for k,v in stats.items()})
        run.summary.update({'phase':'samples_and_scores_complete','sampling/events_complete':n,
                            'sampling/samples_complete':n*args.candidates,'cij/computed':False,
                            'output/candidates':str(output/'candidates.npz')})
    (output/'COMPLETE').write_text('fresh-step1110-h4-cij-v1\n')
    print('CANDIDATES:',output/'candidates.npz',flush=True)


if __name__=='__main__': main()
