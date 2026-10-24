"""User-launched, staged TTbar General test. Never launches DGPO or allocations."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.ttbar_general_classifier.core import (
    GeneralClassifier, geometry, event_partition, nu_p4, mixture, physics,
    bin_ids, contribution, validate_cache_part, NU2FLOW_EDGES, required_extra, top_oracle_inputs, p4_from_columns, oracle_reference_p4,
)


def read_config(path):
    c=yaml.safe_load(Path(path).read_text())
    if c['seed']!=42 or c['policy'].get('ema',False):raise ValueError('This protocol fixes seed42 and raw policy')
    if c['mixture']['oracle'] not in ('top_p4','reco_top_truthnu_p4') or c['mixture']['alpha']!=.3 or c['mixture']['temperature']!=1.:raise ValueError('This protocol fixes 70% General + 30% declared top oracle, T1')
    return c


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)


def atomic_torch(path,obj):
    temp=Path(str(path)+'.tmp');torch.save(obj,temp);temp.replace(path)


def eve_imports(c):
    root=Path(c['evenet_root']).expanduser().resolve()
    if not (root/'evenet/train.py').is_file():raise FileNotFoundError(root)
    sys.path.insert(0,str(root))
    import evenet
    if not str(Path(evenet.__file__).resolve()).startswith(str(root)):
        raise RuntimeError('Wrong EveNet import; use a fresh Python process')
    return root


def resolve_yaml(path):
    def merge(a,b):
        out=copy.deepcopy(a)
        for k,v in b.items():out[k]=merge(out[k],v) if isinstance(v,dict) and isinstance(out.get(k),dict) else v
        return out
    path=Path(path);a=yaml.safe_load(path.read_text())
    for k,v in list(a.items()):
        if isinstance(v,dict) and 'default' in v:
            v=dict(v);base=v.pop('default');a[k]=merge(yaml.safe_load((path.parent/base).read_text()),v)
    return a


def choose_checkpoint(c, requested):
    folder=Path(c['policy']['checkpoint_dir'])
    if requested=='best':
        choices=[]
        for p in folder.glob('epoch=*_val=*.ckpt'):
            m=re.search(r'val=([0-9.]+)\.ckpt$',p.name)
            if m:choices.append((float(m[1]),p))
        if not choices:raise FileNotFoundError('No concrete completed checkpoints; do not use stale last.ckpt')
        return min(choices,key=lambda t:t[0])[1].resolve()
    if not requested:raise ValueError('cache requires --checkpoint best or a concrete checkpoint path')
    p=Path(requested).expanduser().resolve()
    if not p.is_file():raise FileNotFoundError(p)
    return p


def prepare(c, requested):
    eve_imports(c)
    pinned=c['policy'].get('checkpoint')
    if pinned and requested not in (None,pinned):raise ValueError('Config pins a concrete policy; do not override it')
    requested=requested or pinned
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    manifest=out/'cache_manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['experiment']!=c:raise ValueError('Config changed; preserve cache and choose a new output')
        if requested not in (None,'best') and str(choose_checkpoint(c,requested))!=saved['source_checkpoint']:
            raise ValueError('Cache pinned to another checkpoint')
        print('Reusing pinned source:',saved['source_checkpoint'],flush=True)
        return saved
    source=choose_checkpoint(c,requested)
    runtime=resolve_yaml(c['policy']['config'])
    t=runtime['options']['Training']
    if runtime['options']['Dataset']['dataset_limit']!=1.:raise ValueError('Requires full MG5 training data')
    if t['Components']['Assignment']['include'] or t['EMA']['enable']:
        raise ValueError('Requires current assignment-off/EMA-off supervised baseline')
    if not runtime['network']['Body']['InvisibleInputProjector']['enabled']:
        raise ValueError('Requires transplanted native-width diffusion')
    inputs={}
    for role,key in [('train','data_parquet_dir'),('validation','data_parquet_val_dir')]:
        folder=Path(runtime['platform'][key]).resolve()
        files=sorted(folder.glob('*.parquet'))
        if not files:raise FileNotFoundError(f'No parquet in {folder}')
        meta=json.loads((folder/'shape_metadata.json').read_text())
        expected={'x':[12,7],'x_mask':[12],'conditions':[10],'conditions_mask':[1],'x_invisible':[2,3],'x_invisible_mask':[2]}
        for name,shape in expected.items():
            if meta.get(name)!=shape:raise ValueError(f'{role}: schema {name} differs; adapt explicitly')
        for name in required_extra()+['assignments-mask']:
            if name not in meta:raise ValueError(f'{role}: missing required top-oracle field {name}')
        inputs[role]={'directory':str(folder),'files':[str(f) for f in files],
                      'file_sizes':[f.stat().st_size for f in files], 'metadata':meta}
    if set(inputs['train']['files']) & set(inputs['validation']['files']):raise ValueError('Train/validation file overlap')
    shutil.copy2(source,out/'policy.ckpt')
    norm=Path(runtime['options']['Dataset']['normalization_file']);shutil.copy2(norm,out/'normalization.pt')
    runtime['options']['Dataset']['normalization_file']=str(out/'normalization.pt')
    runtime['options']['Training']['model_checkpoint_load_path']=str(out/'policy.ckpt')
    (out/'policy.yaml').write_text(yaml.safe_dump(runtime,sort_keys=False))
    saved=dict(schema=1,source_checkpoint=str(source),experiment=c,inputs=inputs,
               identity='population + source file + absolute row; no native event identity exported',
               limitation='MG5 test was used for diffusion validation; developmental evaluation, not a blind final test')
    write_json(manifest,saved)
    return saved


@torch.no_grad()
def extract_features(model,batch,candidate):
    mask=torch.ones((*candidate.shape[:2],1),device=candidate.device)
    normalized=model.invisible_normalizer(candidate,mask=mask)
    captured=[]
    hook=model.TruthGeneration.generator.register_forward_pre_hook(
        lambda module,args:captured.append(args[0][:,-2:].detach()))
    observed={k:v for k,v in batch.items() if k in ('x','x_mask','conditions','conditions_mask','classification','num_sequential_vectors')}
    try:model.predict_diffusion_vector(normalized,observed,torch.zeros(len(candidate),device=candidate.device),'neutrino',noise_mask=mask)
    finally:hook.remove()
    if len(captured)!=1:raise RuntimeError('Expected one clean pre-velocity hidden state')
    h=captured[0].flatten(1).cpu().numpy()
    return np.concatenate((h,geometry(candidate.cpu().numpy())),1)


@torch.no_grad()
def observed_condition(model,batch):
    xm=batch['x_mask'][...,None];gm=batch['conditions_mask'][...,None]
    x=model.sequential_normalizer(batch['x'],mask=xm)
    g=model.global_normalizer(batch['conditions'][:,None],mask=gm)
    # Preserve observed MET, all visible particles, charges, and both masks.
    return torch.cat((x.flatten(1),xm.flatten(1),g.flatten(1),gm.flatten(1)),1).cpu().numpy()


def load_policy(c,device):
    eve_imports(c)
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device,load_normalization_dict
    out=Path(c['output']);global_config.load_yaml(out/'policy.yaml')
    model=build_evenet_on_device(global_config,load_normalization_dict(global_config),device)
    # Supervised Engine registers these training-only tensors even with FAMO off.
    # Reproduce that layout to enforce strict checkpoint completeness.
    from evenet.network.loss.famo import FAMO
    f=global_config.options.Training.FAMO
    tasks=f.get('detailed_loss_list') if f.get('turn_on',False) and f.get('detailed_loss',False) else ['classification','regression','assignment','generation','segmentation']
    model.register_module('famo',FAMO(task_list=tasks,device=device,lr=f.get('lr',.025),turn_on=f.get('turn_on',False),logits_bound=f.get('logits_bound',1.)))
    ckpt=torch.load(out/'policy.ckpt',map_location='cpu',weights_only=False)
    state=ckpt['state_dict']
    clean={k.removeprefix('model.'):v for k,v in state.items() if k.startswith('model.')}
    if not clean:clean=state
    # Fail on incomplete FiLM/projector or an assignment-enabled old model.
    model.load_state_dict(clean,strict=True)
    model.eval().requires_grad_(False)
    return model


def tensor_batch(a,device):
    keys=('x','x_mask','conditions','conditions_mask','classification','x_invisible','x_invisible_mask','num_sequential_vectors')
    return {key:torch.as_tensor(value,device=device,dtype=(torch.long if key in ('classification','num_sequential_vectors') else torch.bool if key.endswith('_mask') else torch.float32)) for key,value in a.items() if key in keys}


def sharded_batches(source,batch_size,columns,rank,world,file_index):
    # Full MG5 can be a SINGLE parquet. Shard row groups, not just files.
    offset=0
    for rg in range(source.num_row_groups):
        n=source.metadata.row_group(rg).num_rows
        if source.num_row_groups>=world and (file_index+rg)%world!=rank:
            offset+=n;continue
        for bi,record in enumerate(source.iter_batches(batch_size=batch_size,columns=columns,row_groups=[rg])):
            start=offset;offset+=record.num_rows
            if source.num_row_groups<world and (file_index+rg+bi)%world!=rank:continue
            yield start,record


def worker_device():
    from ray.train.torch import get_device
    device=get_device()
    if device.type!='cuda':raise RuntimeError('This stage requires an allocated GPU Ray worker')
    torch.cuda.set_device(device)
    return device


def cache_worker(c, *, generation_model=None, feature_model=None, positive_cache_directory=None):
    import ray.train
    import pyarrow.parquet as pq
    ctx=ray.train.get_context();rank=ctx.get_world_rank();world=ctx.get_world_size()
    torch.set_num_threads(2);device=worker_device()
    model=load_policy(c,device) if generation_model is None else generation_model
    feature_model=model if feature_model is None else feature_model
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.dgpo_trainer import generate_neutrino_candidates
    from experiments.ttbar_dgpo_general.preprocessing_bridge import native_preprocessor
    unflatten_dict = native_preprocessor(c['evenet_root'])
    from reasoning_analysis.event_cuts import compute_base_recon_cut
    out=Path(c['output']);manifest=json.loads((out/'cache_manifest.json').read_text())
    entries=[]
    for role,spec in manifest['inputs'].items():
        k=c['cache']['train_candidates'] if role=='train' else c['cache']['evaluation_candidates']
        for fi,path in enumerate(spec['files']):
            if Path(path).stat().st_size!=spec['file_sizes'][fi]:raise ValueError('Source parquet changed')
            source=pq.ParquetFile(path)
            wanted={'x','x_mask','conditions','conditions_mask','classification','x_invisible','x_invisible_mask','num_sequential_vectors','event_weight','event_valid','assignments-mask'}
            cols=[col for col in source.schema_arrow.names if col.split(':')[0] in wanted or col.split(':')[0] in required_extra()]
            for offset,record in sharded_batches(source,c['cache']['batch_size'],cols,rank,world,fi):
                bi=offset
                folder=out/'cache'/role/f'file-{fi:05d}-row-{offset:012d}'
                if (folder/'part.json').exists():
                    entries.append(str(folder.relative_to(out)));continue
                raw={name:record.column(i).to_numpy(zero_copy_only=False) for i,name in enumerate(record.schema.names)}
                a=unflatten_dict(raw,spec['metadata'])
                valid=np.asarray(a['x_invisible_mask']).astype(bool).all(1)
                if 'event_valid' in a:valid &= np.asarray(a['event_valid']).reshape(-1)>0
                if 'event_weight' in a and not np.allclose(a['event_weight'],1):
                    raise ValueError('Nonunit MG5 event weights; weighted-population protocol must be declared')
                ids=np.array([f'{role}:{Path(path).name}:{offset+i}' for i in np.flatnonzero(valid)])
                if not len(ids):raise ValueError('Empty valid chunk; investigate parquet validity')
                a={key:np.asarray(value)[valid] for key,value in a.items()}
                if not np.all(a['classification']==0):raise ValueError('Expected MG5 TTbar class0; refusing another process')
                batch=tensor_batch(a,device)
                seed=int.from_bytes(hashlib.blake2b(f'{c["seed"]}:{role}:{fi}:{bi}'.encode(),digest_size=4).digest(),'little')
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                truth=a['x_invisible'].astype('float32')
                candidates=generate_neutrino_candidates(model,batch,DDIMSampler(device),K=k,
                    num_ddim_steps=c['cache']['ddim_steps'],device=device,rollout_chunk_k=1).permute(1,0,2,3)
                if positive_cache_directory is not None:
                    old=Path(positive_cache_directory)/folder.relative_to(out)
                    fields=json.loads((old/'part.json').read_text())['fields']
                    if not np.array_equal(np.load(old/fields['ids']),ids) or not np.array_equal(np.load(old/fields['truth']),truth):
                        raise ValueError('Frozen positive-cache event/target mismatch')
                    ft=np.load(old/fields['truth_features'])
                    condition=np.load(old/fields['condition'])
                else:
                    ft=extract_features(feature_model,batch,batch['x_invisible'])
                    condition=observed_condition(feature_model,batch)
                fg=np.stack([extract_features(feature_model,batch,candidates[:,j]) for j in range(k)],1)
                payload=dict(ids=ids,truth=truth,candidates=candidates.cpu().numpy(),condition=condition,
                    labels=a['classification'].astype('int64'),truth_features=ft,generated_features=fg,
                    partition=event_partition(ids,c['seed']) if role=='validation' else np.full(len(ids),-1,dtype='int8'))
                if role=='validation':
                    payload['physics_keep']=compute_base_recon_cut(a['x'])[0] & a['assignments-mask'].astype(bool).all(1)
                payload['oracle_valid']=a['assignments-mask'].astype(bool).all(1)
                payload.update({key:value for key,value in a.items() if key.startswith('EXTRA/')})
                validate_cache_part(payload)
                folder.mkdir(parents=True,exist_ok=True)
                fields={}
                for i,(key,value) in enumerate(payload.items()):
                    name=f'array-{i:02d}.npy';np.save(folder/name,value);fields[key]=name
                write_json(folder/'part.json',dict(fields=fields,events=len(ids),input_events=record.num_rows,
                    source_file=path,batch=bi,role=role))
                entries.append(str(folder.relative_to(out)))
                print(f'cache {role} rank{rank} file{fi} batch{bi}: {len(ids)} events K{k}',flush=True)
    write_json(out/f'cache-rank-{rank}.json',dict(parts=entries))


def parts(out,role=None):
    manifest=json.loads((Path(out)/'CACHE_COMPLETE.json').read_text())
    for rel in manifest['parts']:
        if role and rel.startswith('cache/') and not rel.startswith(f'cache/{role}/'):continue
        folder=Path(out)/rel;m=json.loads((folder/'part.json').read_text())
        if role is None or m['role']==role:
            yield folder,m,{k:np.load(folder/v,mmap_mode='r') for k,v in m['fields'].items()}


class CachedRows:
    """Classifier-only arrays, loaded once; batches never reopen cache files.

    Each rank keeps a copy so existing global shuffle/rank slicing is unchanged.
    Evaluation still reads the complete K-candidate cache through ``parts``.
    """
    fields=('condition','truth_features','generated_features','labels')

    def __init__(self,out,role,selection=False):
        out=Path(out);items=[];n=0
        manifest=json.loads((out/'CACHE_COMPLETE.json').read_text())
        for rel in manifest['parts']:
            if rel.startswith('cache/') and not rel.startswith(f'cache/{role}/'):continue
            folder=out/rel;m=json.loads((folder/'part.json').read_text())
            if m['role']!=role:continue
            idx=np.flatnonzero(np.load(folder/m['fields']['partition'],mmap_mode='r')==0) if selection else np.arange(m['events'])
            if not len(idx):continue
            items.append((folder,m,idx));n+=len(idx)
        self.n=n;self.arrays={}
        if not n:raise ValueError('Empty classifier population')
        offset=0
        for folder,m,idx in items:
            for key in self.fields:
                arr=np.load(folder/m['fields'][key],mmap_mode='r')
                # Identical paired BCE population: retain candidate0 only.
                # Slice K BEFORE copying validation features into RAM.
                if key=='generated_features':arr=arr[:,0]
                if key not in self.arrays:
                    self.arrays[key]=np.empty((n,*arr.shape[1:]),dtype=arr.dtype)
                target=self.arrays[key]
                if arr.shape[1:]!=target.shape[1:] or arr.dtype!=target.dtype:
                    raise ValueError(f'Inconsistent classifier cache schema: {key} in {folder}')
                target[offset:offset+len(idx)]=arr[idx]
                del arr
            offset+=len(idx)
        self.nbytes=sum(a.nbytes for a in self.arrays.values())
        print(f'Classifier {role}: {n} events loaded into CPU RAM ({self.nbytes/2**30:.2f} GiB per worker)',flush=True)

    def batch(self,indices):
        return {key:arr[indices] for key,arr in self.arrays.items()}


def normalization(c):
    sums=squares=None;n=0;ps=np.zeros(4);pn=0;classes=set()
    for _,_,a in parts(c['output'],'train'):
        f=np.concatenate((a['truth_features'],a['generated_features'][:,0]),0).astype(float)
        if sums is None:sums=np.zeros(f.shape[1]);squares=np.zeros_like(sums)
        sums+=f.sum(0);squares+=(f*f).sum(0);n+=len(f)
        p=oracle_reference_p4(a['truth'],a,c.get('mixture',{}).get('oracle','top_p4'))[a['oracle_valid']]
        if not np.isfinite(p).all():raise ValueError('Nonfinite oracle reference p4')
        ps+=(p*p).sum((0,1));pn+=p.shape[0]*2
        classes.update(np.unique(a['labels']).tolist())
    if classes!={0}:raise ValueError('Protocol is full MG5 TTbar process label0; adapt other labels explicitly')
    mean=sums/n;std=np.sqrt(np.maximum(squares/n-mean**2,1e-12))
    # One scale for spatial axes preserves rotations; energy has its own RMS.
    if pn==0:raise ValueError('No valid top-oracle training events')
    rms=np.sqrt(ps/pn);spatial=float(np.sqrt((ps[1:]/pn).mean()))
    oracle_scale=np.array([rms[0],spatial,spatial,spatial])
    if (oracle_scale<=0).any():raise ValueError('Undefined oracle scales')
    first=next(parts(c['output'],'train'))[2]
    return dict(mean=torch.tensor(mean,dtype=torch.float32),scale=torch.tensor(std,dtype=torch.float32),
        oracle_scale=oracle_scale.tolist(),oracle_reference=c.get('mixture',{}).get('oracle','top_p4'),condition_dim=first['condition'].shape[1],candidate_dim=len(mean),
        classes=1,source='full classifier training population only; shared xyz RMS, separate energy RMS')


def fit_worker(c, *, initial_state=None, log_callback=None):
    import ray.train
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    resume=c.pop('_resume',False)
    ctx=ray.train.get_context();rank=ctx.get_world_rank();world=ctx.get_world_size()
    device=worker_device();torch.set_num_threads(2);torch.manual_seed(c['seed'])
    out=Path(c['output']);st=torch.load(out/'normalization-classifier.pt',weights_only=True)
    model=GeneralClassifier(st['condition_dim'],st['candidate_dim'],**c['head']).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state,strict=True)
    ddp=DDP(model,device_ids=[device.index]);train=CachedRows(out,'train');val=CachedRows(out,'validation',True)
    fit=c['fit'];opt=torch.optim.AdamW(model.parameters(),lr=fit['lr'],weight_decay=fit['weight_decay'])
    mean,scale=st['mean'].to(device),st['scale'].to(device)
    saved=None;start_epoch=0
    last=out/'classifier-last.pt'
    if resume:
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['config']!=c or saved['world_size']!=world:raise ValueError('Resume requires identical config/world size')
        model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'][rank]['cpu']);torch.cuda.set_rng_state(saved['rng'][rank]['cuda'],device)
        start_epoch=saved['metrics']['epoch']
    elif last.exists():raise ValueError('Interrupted classifier fit exists; use --resume')
    run=None
    if rank==0 and log_callback is None:
        import wandb
        run=wandb.init(**c['wandb'],config=c,resume='allow' if resume else 'never')
    best=float('inf');stale=0;step=0;history=[];plateau=False
    if saved:
        best=saved['metrics']['best_validation_bce'];stale=saved['metrics']['stale_epochs'];step=saved['metrics']['steps'];history=saved['history']
    for epoch in range(start_epoch,fit['max_epochs']):
        lr=fit['min_lr']+(fit['lr']-fit['min_lr'])*.5*(1+math.cos(math.pi*min(epoch,fit['cosine_epochs'])/fit['cosine_epochs']))
        for group in opt.param_groups:group['lr']=lr
        ddp.train();order=np.random.default_rng(c['seed']+epoch).permutation(train.n)
        sums=torch.zeros(2,device=device,dtype=torch.float64);gb=fit['batch_size']*world
        for start in range(0,train.n,gb):
            block=order[start:start+gb];valid=len(block)
            padded=np.pad(block,(0,gb-valid),mode='wrap');idx=padded[rank::world]
            weight=torch.tensor((np.arange(rank,gb,world)<valid).astype('float32'),device=device)
            a=train.batch(idx);cond=torch.tensor(a['condition'],device=device);labels=torch.tensor(a['labels'],device=device)
            pos=(torch.tensor(a['truth_features'],device=device)-mean)/scale
            neg=(torch.tensor(a['generated_features'],device=device)-mean)/scale
            # One DDP forward contains both populations; globally paired and class balanced.
            s=ddp(torch.cat((cond,cond)),torch.cat((pos,neg)),torch.cat((labels,labels)))
            p,q=s.chunk(2);loss=(torch.nn.functional.softplus(-p)+torch.nn.functional.softplus(q))/2
            opt.zero_grad(set_to_none=True);(world*(loss*weight).sum()/valid).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
            sums[0]+=(loss.detach()*weight).sum();sums[1]+=weight.sum()
        dist.all_reduce(sums)
        model.eval();vs=torch.zeros(2,device=device,dtype=torch.float64)
        with torch.no_grad():
            vi=np.arange(rank,val.n,world)
            for start in range(0,len(vi),fit['batch_size']):
                a=val.batch(vi[start:start+fit['batch_size']]);co=torch.tensor(a['condition'],device=device);la=torch.tensor(a['labels'],device=device)
                p=model(co,(torch.tensor(a['truth_features'],device=device)-mean)/scale,la)
                q=model(co,(torch.tensor(a['generated_features'],device=device)-mean)/scale,la)
                vs[0]+=(torch.nn.functional.softplus(-p)+torch.nn.functional.softplus(q)).sum()/2;vs[1]+=len(p)
        dist.all_reduce(vs);v=float(vs[0]/vs[1]);tr=float(sums[0]/sums[1])
        improved=v<best-fit['min_delta']
        if improved:best=v;stale=0
        else:stale+=1
        row=dict(epoch=epoch+1,steps=step,train_bce=tr,validation_bce=v,best_validation_bce=best,lr=lr,stale_epochs=stale)
        history.append(row)
        rng=[None]*world
        dist.all_gather_object(rng,dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device)))
        if rank==0:
            state=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},stats=st,config=c,metrics=row,optimizer=opt.state_dict(),rng=rng,world_size=world,history=history)
            if improved:atomic_torch(out/'classifier-best.pt',state)
            atomic_torch(out/'classifier-last.pt',state);write_json(out/'fit_history.json',history)
            if log_callback is None:run.log(row,step=step)
            else:log_callback(row)
            print(row,flush=True)
        if stale>=fit['patience'] and step>=fit['minimum_steps']:plateau=True;break
    if rank==0:
        write_json(out/'FIT_COMPLETE.json',dict(plateau=plateau,best_validation_bce=best,epochs=len(history),steps=step))
        if run is not None:
            run.summary['validation_plateau']=plateau;run.finish()
    dist.barrier()


def ray_launch(fn,c):
    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    eve_imports(c)
    # A one-node interactive allocation does not start Ray automatically.
    # Honour an explicitly configured cluster; otherwise create a local runtime
    # instead of searching for an unrelated/stale Ray session.
    address=os.environ.get('RAY_ADDRESS') or 'local'
    if address=='local':
        if int(os.environ.get('SLURM_JOB_NUM_NODES','1'))!=1:
            raise ValueError('Multi-node execution requires an explicitly configured RAY_ADDRESS')
        if torch.cuda.device_count()<c['workers']:
            raise RuntimeError(f"Need {c['workers']} visible GPUs; found {torch.cuda.device_count()}. Run inside the GPU allocation.")
    ray.init(address=address,runtime_env={'env_vars':{'PYTHONPATH':str(ROOT)+os.pathsep+c['evenet_root']}})
    try:
        trainer=TorchTrainer(fn,train_loop_config=c,scaling_config=ScalingConfig(num_workers=c['workers'],use_gpu=True,
            resources_per_worker={'CPU':c['cpus_per_worker'],'GPU':1}))
        trainer.fit()
    finally:
        ray.shutdown()


def make_cache(c,checkpoint):
    prepare(c,checkpoint)
    out=Path(c['output'])
    if (out/'CACHE_COMPLETE.json').exists():
        if not (out/'normalization-classifier.pt').exists():atomic_torch(out/'normalization-classifier.pt',normalization(c))
        print('Cache already complete; no regeneration');return
    ray_launch(cache_worker,c)
    entries=[]
    for rank in range(c['workers']):entries.extend(json.loads((out/f'cache-rank-{rank}.json').read_text())['parts'])
    entries=sorted(entries)
    if len(entries)!=len(set(entries)):raise ValueError('Duplicate cache shards')
    totals={};seen=set()
    for rel in entries:
        folder=out/rel;m=json.loads((folder/'part.json').read_text());ids=np.load(folder/m['fields']['ids'])
        if seen.intersection(ids):raise ValueError('Duplicate identities')
        seen.update(ids);totals[m['role']]=totals.get(m['role'],0)+len(ids)
    write_json(out/'CACHE_COMPLETE.json',dict(parts=entries,events=totals))
    atomic_torch(out/'normalization-classifier.pt',normalization(c))


def evaluate(c,score_fn=None):
    # This stage can run locally after downloading cached files; never loads diffusion.
    from scripts import tau_candidate_svd_evaluation as svd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    out=Path(c['output']);dest=out/'evaluation'
    # Fail before computing plots if genuine RooUnfoldSvd is not available.
    svd.load_root()
    if dest.exists():raise ValueError('Preserve existing evaluation; use a new output or remove only incomplete outputs manually')
    status=json.loads((out/'FIT_COMPLETE.json').read_text())
    if not status['plateau']:raise ValueError('Classifier hit safety cap without validation plateau; do not call saturated')
    payload=torch.load(out/'classifier-best.pt',map_location='cpu',weights_only=True);st=payload['stats']
    model=None
    if score_fn is None:
        model=GeneralClassifier(st['condition_dim'],st['candidate_dim'],**c['head']);model.load_state_dict(payload['state_dict']);model.eval()
    arrays={};positive=[];negative=[]
    with torch.no_grad():
        for _,_,a in parts(out,'validation'):
            keep=(a['partition']!=0)&a['physics_keep'].astype(bool)
            if not keep.any():continue
            if score_fn is None:
                cond=torch.tensor(a['condition'][keep]);labels=torch.tensor(a['labels'][keep]);fg=a['generated_features'][keep]
                logits=np.stack([model(cond,(torch.tensor(fg[:,j])-st['mean'])/st['scale'],labels).numpy() for j in range(fg.shape[1])],1)
                pos=model(cond,(torch.tensor(a['truth_features'][keep])-st['mean'])/st['scale'],labels).numpy()
            else:
                pos,logits=score_fn(a,keep)
                if logits.shape!=a['generated_features'][keep].shape[:2] or pos.shape!=(int(keep.sum()),):raise ValueError('Score/cache shapes differ')
            positive.extend(pos)
            negative.extend(logits[:,0])
            for key in ('ids','truth','candidates','partition'):
                arrays.setdefault(key,[]).append(np.asarray(a[key][keep]))
            arrays.setdefault('logits',[]).append(logits)
            for key in a:
                if key.startswith('EXTRA/'):arrays.setdefault(key,[]).append(np.asarray(a[key][keep]))
    arrays={k:np.concatenate(v) for k,v in arrays.items()};n=len(arrays['ids'])
    extra={k:v for k,v in arrays.items() if k.startswith('EXTRA/')}
    if st.get('oracle_reference','top_p4')!=c['mixture']['oracle']:raise ValueError('Oracle normalization belongs to a different reference; do not reuse it silently')
    candidate_top,truth_top=top_oracle_inputs(arrays['candidates'],extra,arrays['truth'],c['mixture']['oracle'])
    general,mix,distance=mixture(arrays['logits'],candidate_top,truth_top,st['oracle_scale'],**{k:c['mixture'][k] for k in ('alpha','temperature')})
    weights=[np.full_like(general,1/general.shape[1]),general,mix]
    names=['Raw diffusion','General','General70 + oracle30']
    extra={k:v for k,v in arrays.items() if k.startswith('EXTRA/')}
    observables=physics(arrays['truth'],arrays['candidates'],extra,include_mc_top_mass=c['mixture']['oracle']=='top_p4')
    report=dict(protocol=c,evaluation_events=n,source=json.loads((out/'cache_manifest.json').read_text())['source_checkpoint'],
                auc=float(roc_auc_score(np.r_[np.ones(len(positive)),np.zeros(len(negative))],np.r_[positive,negative])),
                oracle_scale=st['oracle_scale'],oracle_is_truth_assisted=True,oracle=c['mixture']['oracle'],oracle_target=c.get('protocol',{}).get('oracle_target'),oracle_distance='mean normalized squared difference over both ordered legs and E,px,py,pz',
                top_reconstruction='truth-assigned reconstructed b/leptons; excludes assignment uncertainty',
                selection='native compute_base_recon_cut AND both assignments-mask; common fixed events across arms',
                uncertainty='RooUnfoldSvd; whole-event H.T@H data covariance; separate paired event response-MC bootstrap',
                nu2flow_scope='Native 4x4 binning and kreg7; absolute dilepton delta-phi/pi explicitly used instead of the notebook signed expression. K8 event covariance and independent response/measurement/reference partitions replace legacy same-sample covariance.',
                top_mass_references={'reco_truthnu':'reco b + reco lepton + MC neutrino, matching legacy scoring plot','mc_top':('unaltered MC top, separate output; sparse reference bins are reported unsupported' if c['mixture']['oracle']=='top_p4' else 'no separate MC-top mass plot; native nu2flow joint observables retain their original parton-truth reference')},
                physics_partitions={str(i):int((arrays['partition']==i).sum()) for i in (1,2,3)},
                edges='native nu2flow 4x4 bins, kreg7; neutrino native interior bins with tail folding, kreg5; top mass 28bins with tail folding, kreg5',
                limitation='Developmental MG5 test, previously used in diffusion model selection; not blind coverage certification',
                targets={},weight_health={})
    dest.mkdir(parents=True)
    np.savez_compressed(dest/'weights-and-splits.npz',ids=arrays['ids'],partition=arrays['partition'],general=general,mixture=mix)
    for name,w in zip(names,weights):
        raw=arrays['candidates'];center=(w[...,None,None]*raw).sum(1)
        spread=(w[...,None,None]*(raw-center[:,None])**2).sum(1)
        # Phi spread uses circular variance, never linear discontinuity across +/-pi.
        circular=1-np.abs((w[...,None]*np.exp(1j*raw[...,2])).sum(1))
        report['weight_health'][name]=dict(mean_ess=float(np.mean(1/(w*w).sum(1))),
            mean_max_weight=float(w.max(1).mean()),conditional_logpt_eta_variance=spread[:,:,:2].mean(0).tolist(),
            conditional_phi_circular_variance=circular.mean(0).tolist())
    report['negative_pt_candidate_fraction']=float((arrays['candidates'][...,0]<0).mean())
    for key,(truth,gen,edges,kreg) in observables.items():
        folder=dest/key;folder.mkdir();nb=len(edges)-1
        t=bin_ids(truth,edges);b=bin_ids(gen,edges)
        hs=[contribution(b,w,nb) for w in weights]
        r,m,q=[arrays['partition']==i for i in (1,2,3)]
        try:
            analysis=svd.evaluate([h[r] for h in hs],[h[m] for h in hs],t[r],t[m],t[q],edges,folder,
                names=names,kreg=kreg,reps=c['evaluation']['response_bootstraps'],seed=4203,pairs=[(0,1),(0,2),(1,2)])
            report['targets'][key]=dict(status='complete',analysis=analysis)
        except (ValueError,np.linalg.LinAlgError) as e:
            # Preserve unsupported bins/failures, never change kreg or merge to manufacture improvement.
            report['targets'][key]=dict(status='unsupported',reason=str(e))
        fig,ax=plt.subplots(figsize=(7,4))
        display_edges=np.asarray(edges).copy();step=display_edges[2]-display_edges[1]
        if not np.isfinite(display_edges[0]):display_edges[0]=display_edges[1]-step
        if not np.isfinite(display_edges[-1]):display_edges[-1]=display_edges[-2]+step
        x=(display_edges[:-1]+display_edges[1:])/2
        xlabel='Flattened nu2flow bin' if key.startswith('mtt_x_') else ('Mass [GeV]' if 'mass' in key else 'pT [GeV]' if key.endswith('_pt') else 'eta' if key.endswith('_eta') else 'phi [rad]')
        ax.step(x,np.bincount(t[q],minlength=nb)/q.sum(),where='mid',label='Truth',color='black')
        for name,h in zip(names,hs):ax.step(x,h[m].mean(0),where='mid',label=name)
        ax.set(title=key,xlabel=xlabel,ylabel='Probability');ax.legend();fig.tight_layout();fig.savefig(folder/'distribution.png',dpi=160);plt.close(fig)
        if report['targets'][key]['status']=='complete':
            rows=analysis['models'];fig,ax=plt.subplots(figsize=(7,4))
            saved_arrays=np.load(folder/'unfolding_arrays.npz')
            for row in rows:ax.plot(x,np.sqrt(np.diag(saved_arrays[f'model{rows.index(row)}_shape_covariance'])),label=row['name'])
            ax.set(title=key,xlabel=xlabel,ylabel='Unfolded probability uncertainty');ax.legend();fig.tight_layout();fig.savefig(folder/'uncertainty.png',dpi=160);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,4));ax.step(x,saved_arrays['reference_probability'],where='mid',label='Truth',color='black')
            for j,row in enumerate(rows):
                ax.errorbar(x,row['unfolded_probability'],yerr=np.sqrt(np.diag(saved_arrays[f'model{j}_shape_covariance'])),fmt='.',label=row['name'])
            ax.set(title=key,xlabel=xlabel,ylabel='Unfolded probability');ax.legend();fig.tight_layout();fig.savefig(folder/'unfolded.png',dpi=160);plt.close(fig)
            fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
            for j,ax in enumerate(axes):
                matrix=saved_arrays[f'model{j}_response_matrix'];matrix=matrix/matrix.sum(0,keepdims=True)
                im=ax.imshow(matrix,origin='lower',vmin=0,vmax=1,aspect='auto');ax.set(title=names[j],xlabel='Truth bin',ylabel='Reco bin')
            fig.colorbar(im,ax=axes,label='P(reco bin | truth bin)');fig.savefig(folder/'response.png',dpi=160);plt.close(fig)
            saved_arrays.close()
        write_json(dest/'report.json',report)
        print('evaluation',key,report['targets'][key]['status'],flush=True)
    report['dgpo_ready']='manual review required; no automatic training launch'
    # Save per-target progress so long local evaluations remain inspectable.
    # Edges are stored in NPZ; JSON contains no nonfinite edges.
    write_json(dest/'report.json',report)
    html='<h1>TTbar General and oracle reweighting</h1><p>Frozen diffusion; truth-assisted oracle. Top observables use truth-assigned reconstructed objects. See report for closure, covariance and limitations.</p>'
    for key in observables:
        html+=f'<h2>{key}</h2><img width="650" src="{key}/distribution.png">'
        if report['targets'][key]['status']=='complete':
            html+=f'<img width="650" src="{key}/uncertainty.png"><img width="650" src="{key}/unfolded.png"><img width="1100" src="{key}/response.png">'
            html+='<table><tr><th>Model</th><th>TV distance</th><th>Data sigma RSS</th><th>Response sigma RSS</th><th>Total sigma RSS</th><th>Closure chi2/ndof</th></tr>'
            for row in report['targets'][key]['analysis']['models']:
                cl=row['closure'];html+=f'<tr><td>{row["name"]}</td><td>{row["tv"]:.4g}</td><td>{row["data_shape_rss"]:.4g}</td><td>{row["response_shape_rss"]:.4g}</td><td>{row["total_shape_rss"]:.4g}</td><td>{cl["chi2"]:.3g}/{cl["ndof"]}</td></tr>'
            html+='</table>'
        else:html+='<p>Unsupported SVD: '+report['targets'][key]['reason']+'</p>'
    (dest/'index.html').write_text(html)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config');parser.add_argument('stage',choices=['validate','cache','fit','evaluate'])
    parser.add_argument('--resume',action='store_true',help='Resume an interrupted classifier fit including optimizer and RNG')
    parser.add_argument('--checkpoint');parser.add_argument('--output',help='Local copy of the experiment output for cached evaluation')
    args=parser.parse_args();c=read_config(args.config)
    if args.output:c['output']=str(Path(args.output).resolve())
    if args.stage=='validate':
        assert c['workers']==4 and c['cache']['evaluation_candidates']==8
        assert c['head']==dict(dropout=.1,ratio_bound=30.,classes=1)
        assert isinstance(c['fit']['batch_size'],int) and c['fit']['batch_size']>0
        print('Config valid; no generation, training or allocation started.');return
    if args.stage=='cache':make_cache(c,args.checkpoint)
    elif args.stage=='fit':
        if not (Path(c['output'])/'CACHE_COMPLETE.json').exists():raise ValueError('Cache is incomplete; finish the cache stage before fit')
        if (Path(c['output'])/'FIT_COMPLETE.json').exists():raise ValueError('Existing fit; choose a new run rather than overwrite')
        c['_resume']=args.resume
        ray_launch(fit_worker,c)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(2):evaluate(c)

if __name__=='__main__':main()
