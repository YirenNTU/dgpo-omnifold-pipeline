"""User-launched 16-GPU inference on filtered, disjoint full-validation remainder."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'scripts'),str(REPO/'evenet_dgpo')]
import numpy as np
import yaml


def prepare(cfg):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.prepare_stic_filtered_test import exclusion_mask
    from scripts.diagnose_ztautau_cij import SOURCE_ID_COLUMNS, source_ids, read_event_table, align
    from scripts.sample_1110_cij import validation_pool
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from evenet.control.global_config import global_config
    original=Path(cfg['reference_samples']).resolve(strict=True)
    manifest=json.loads((original/'manifest.json').read_text())
    validate_filtered_dataset(manifest['events'])
    source=Path(cfg['source']).resolve(strict=True);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    files=sorted(source.rglob('*.parquet'))
    if not files:raise ValueError('No full-validation parquet files')
    with np.load(original/'panel.npz',allow_pickle=False) as f:excluded=set(f['source_ids'].tolist())
    if len(excluded)!=119002:raise ValueError('Expected original 119002 unique validation identities')
    names=pq.ParquetFile(next(Path(manifest['events']).glob('*.parquet'))).schema_arrow.names
    keys=[k for k in SOURCE_ID_COLUMNS if k in names]
    # Inspect the pinned runtime's raw feature ordering; no guessed feature indices.
    global_config.load_yaml(manifest['arms']['dgpo']['runtime'])
    feature_names=list(global_config.event_info.raw_sequential_feature_names)
    filtered=root/'filtered';data=filtered/'val';panel_path=root/'panel.npz'
    contract=dict(source=str(source),reference_samples=str(original),identity_keys=keys,
                  packing_spec=manifest['packing_spec'],raw_features=feature_names)
    if (filtered/'filter_manifest.json').exists():
        saved=json.loads((filtered/'filter_manifest.json').read_text())
        if saved.get('contract')!=contract:raise ValueError('Prepared data belongs to a different source/config')
        validate_filtered_dataset(data)
    else:
        data.mkdir(parents=True,exist_ok=False)
        seen=set();found=set();total=removed=stic=overlap=0
        with (filtered/'removed_events.jsonl').open('x') as audit:
            for number,path in enumerate(files):
                pf=pq.ParquetFile(path)
                if any(k not in pf.schema_arrow.names for k in keys):raise ValueError('Full source identity schema differs')
                with pq.ParquetWriter(data/f'part-{number:05d}.parquet',pf.schema_arrow,compression='snappy') as writer:
                    offset=0
                    for batch in pf.iter_batches(batch_size=8192):
                        ids=source_ids({k:batch.column(k).to_numpy(zero_copy_only=False) for k in keys},keys)
                        if len(np.unique(ids))!=len(ids) or seen.intersection(ids):raise ValueError('Duplicate source identities; no deduplication')
                        seen.update(ids);old=np.array([i in excluded for i in ids]);found.update(ids[old])
                        bad,reasons=exclusion_mask(batch,feature_names)
                        reject=old|bad
                        kept=batch.filter(pa.array(~reject))
                        if len(kept):writer.write_batch(kept)
                        for row,why in reasons.items():audit.write(json.dumps(dict(file=str(path),row=offset+row,event_id=str(ids[row]),reasons=why))+'\n')
                        total+=len(batch);removed+=int(reject.sum());stic+=int(bad.sum());overlap+=int(old.sum());offset+=len(batch)
                print(f'[filter] {number+1}/{len(files)} read={total} retained={total-removed} excluded_old={overlap} STIC={stic}',flush=True)
        if found!=excluded:raise ValueError(f'Original validation identities not recovered: {len(excluded-found)} missing; cannot certify disjoint population')
        if total==removed:raise ValueError('No new events')
        shutil.copy2(source/'shape_metadata.json',data/'shape_metadata.json')
        result=dict(complete=True,source=str(source),output=str(data.resolve()),rows_in=total,rows_out=total-removed,
                    events_removed=removed,previous_validation_excluded=overlap,stic_flagged=stic,
                    rule='Exclude prior validation identities OR invalid STIC on valid particles; union counts',
                    normalization='Unchanged pinned model runtimes; no recomputation',contract=contract)
        (filtered/'filter_manifest.json').write_text(json.dumps(result,indent=2))
        validate_filtered_dataset(data)
    if not panel_path.exists():
        pool=validation_pool(data,{'packing_spec':manifest['packing_spec']})
        if pool['invalid_target_rows']:raise ValueError('Invalid target masks; no silent population change')
        ids=source_ids(pool,keys)
        if excluded.intersection(ids):raise ValueError('Overlap with original validation')
        cols=keys+['event_category','event_weight','analyzing_power_a','analyzing_power_b']
        cols += [f'lead_{leg}_visible_{c}' for leg in ('a','b') for c in ('E','px','py','pz')]
        table=read_event_table(data,cols);arrays={k:np.asarray(table[k].to_numpy()) for k in cols}
        order=align(source_ids(arrays,keys),ids)
        payload=dict(raw_condition=pool['condition'].numpy(),truth_deltas=pool['truth'].numpy(),source_ids=ids,
                     category=arrays['event_category'][order],event_weight=arrays['event_weight'][order],
                     kappas=np.stack([arrays['analyzing_power_a'][order],arrays['analyzing_power_b'][order]],-1))
        for leg in ('a','b'):payload['visible_'+leg]=np.stack([arrays[f'lead_{leg}_visible_{c}'][order] for c in ('E','px','py','pz')],-1)
        if not np.isfinite(payload['kappas']).all() or (payload['kappas']==0).any():raise ValueError('Invalid kappas')
        if not np.isfinite(payload['event_weight']).all() or (payload['event_weight']<0).any():raise ValueError('Invalid weights')
        temp=root/'panel.pending.npz';np.savez(temp,**payload);temp.replace(panel_path)
    with np.load(panel_path,allow_pickle=False) as f:
        ids=f['source_ids'];n=len(ids)
        if len(np.unique(ids))!=n or excluded.intersection(ids):raise ValueError('Cached panel IDs invalid')
    if n!=validate_filtered_dataset(data)['rows']:raise ValueError('Panel count differs from filtered parquet')
    return root,data,panel_path,manifest,n


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path)
    p.add_argument('phase',choices=('prepare','generate'),nargs='?',default='generate')
    p.add_argument('--no-wandb',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if cfg['workers']!=16 or cfg['batch_size']<1:raise ValueError('Require 16 GPUs and positive batch size')
    root,data,panel,old,n=prepare(cfg)
    print(f'READY: {n} new events; original validation excluded; pinned raw models: {old["arms"]}',flush=True)
    if args.phase=='prepare':return
    import ray
    from ray.train import RunConfig,ScalingConfig
    from ray.train.torch import TorchTrainer
    from scripts.run_tau_cij_sampling_bias import worker
    ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto',runtime_env={'env_vars':{
        'PYTHONPATH':os.pathsep.join([str(REPO),str(REPO/'scripts'),str(REPO/'evenet_dgpo')])}})
    if ray.cluster_resources().get('GPU',0)<16:raise ValueError('Need 16 allocated GPUs')
    out=root/('samples-'+uuid.uuid4().hex[:10]);out.mkdir()
    shutil.copy2(panel,out/'panel.npz')
    manifest=dict(old);manifest.pop('sample_source',None)
    manifest.update(events=str(data.resolve()),output=str(out.resolve()),workers=16,batch_size=cfg['batch_size'],
                    seed=cfg['seed'],event_count=n,candidates=1,weights='raw_state_dict_only',
                    complete=False,population='filtered_full_validation_remainder',reference_samples=cfg['reference_samples'])
    # Use the immutable snapshots/runtimes from the original comparison, never last.ckpt.
    for values in manifest['arms'].values():
        Path(values['checkpoint']).resolve(strict=True);Path(values['runtime']).resolve(strict=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    run=None
    try:
        if not args.no_wandb:
            import wandb
            run=wandb.init(**cfg['wandb'],config=manifest,dir=str(out))
        for arm,values in manifest['arms'].items():
            if run:run.summary['phase']='generate_'+arm
            wc={**manifest,**values,'arm':arm,'panel':str(out/'panel.npz')}
            TorchTrainer(worker,train_loop_config=wc,scaling_config=ScalingConfig(num_workers=16,use_gpu=True),
                run_config=RunConfig(name=arm,storage_path=str(out/'ray_results'))).fit()
            delta=np.empty((n,2,2),np.float32);seen=[]
            for rank in range(16):
                with np.load(out/f'{arm}-rank{rank:02d}.npz',allow_pickle=False) as f:
                    ii=f['positions'];values=f['deltas']
                    if values.shape!=(len(ii),2,2) or not np.isfinite(values).all():raise ValueError('Invalid predictions')
                    delta[ii]=values;seen.extend(ii.tolist())
            if sorted(seen)!=list(range(n)):raise ValueError('Missing/duplicate prediction positions')
            with np.load(out/'panel.npz',allow_pickle=False) as f:ids=f['source_ids']
            np.savez_compressed(out/f'{arm}_samples.npz',source_ids=ids,deltas=delta)
            if run:run.log({f'samples/{arm}/events':n})
        manifest['complete']=True;(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        ready=root/'ready_samples.pending.json'
        ready.write_text(json.dumps(dict(complete=True,sample_directory=str(out.resolve()),events=n),indent=2))
        ready.replace(root/'ready_samples.json')
        if run:run.summary.update(dict(phase='complete',complete=True,events=n,sample_directory=str(out)))
        print('SAMPLES READY:',out,flush=True)
    finally:
        if run:
            run.save(str(out/'manifest.json'),base_path=str(out))
            run.finish(exit_code=0 if manifest['complete'] else 1)


if __name__=='__main__':main()
