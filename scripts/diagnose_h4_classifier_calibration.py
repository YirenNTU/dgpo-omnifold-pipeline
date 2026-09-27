#!/usr/bin/env python3
"""Tail attribution and frozen-classifier response to symmetric production calibration."""
import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from diagnose_h4_calibration_impact import directions, calibrate, separation
from diagnose_h4_ratio_tail import unpack, weight_metrics
from diagnose_h4_spike_coverage import load_artifact, write_json, ROOT
from diagnose_h4_topology_resolution import reconstruct


def projected_deltas(fields, z):
    """Physical direction projection, then inverse of production delta encoding."""
    shape = z.shape
    u = calibrate(directions(fields, z.cpu().numpy().reshape(len(z), -1, 4)))
    values = []
    for j, leg in enumerate(('a', 'b')):
        p = np.stack([fields[f'lead_{leg}_visible_{axis}'].numpy().reshape(-1) for axis in ('px','py','pz')], -1)
        theta_vis = np.arctan2(np.hypot(p[:,0],p[:,1]),p[:,2])[:,None]
        phi_vis = np.arctan2(p[:,1],p[:,0])[:,None]
        v = u[:,:,j]
        theta = np.arctan2(np.hypot(v[:,:,0],v[:,:,1]),v[:,:,2])
        delta_phi = np.arctan2(v[:,:,1],v[:,:,0])-phi_vis
        values.extend([theta-theta_vis, np.arctan2(np.sin(delta_phi),np.cos(delta_phi))])
    out = torch.from_numpy(np.stack(values,-1).reshape(shape)).to(z.dtype)
    rebuilt = directions(fields,out.numpy().reshape(len(z),-1,4))
    if np.max(separation(u,rebuilt)) > 2e-6:
        raise ValueError('Projected delta encoding failed direction round-trip')
    return out


def describe(x):
    if len(x)==0:
        return None
    return dict(zip(('mean','q01','median','q99'), [float(np.mean(x)), *np.quantile(x,[.01,.5,.99]).tolist()]))


def tail_report(b, include_calibration=True):
    fields=unpack(b)
    g=b['test_generated'].reshape(-1,4)
    topo=reconstruct(fields,g)
    features={k:topo[k] for k in ('acoplanarity','acollinearity')}
    if include_calibration:
        u=directions(fields,g.numpy()[:,None,:])
        features['calibration_shift']=separation(u,calibrate(u)).mean((1,2))
    for leg in ('a','b'):
        p=np.stack([fields[f'lead_{leg}_visible_{a}'].numpy().reshape(-1) for a in ('px','py','pz')],-1)
        features[f'visible_{leg}_pt']=np.hypot(p[:,0],p[:,1])
        features[f'visible_{leg}_cos_theta']=p[:,2]/np.linalg.norm(p,axis=1)
    logits=b['gen_logits'].double().reshape(-1)
    w=torch.softmax(logits,0).numpy()
    invalid_theta=np.zeros(len(g),dtype=bool)
    for j,leg in enumerate(('a','b')):
        p=np.stack([fields[f'lead_{leg}_visible_{axis}'].double().numpy().reshape(-1) for axis in ('px','py','pz')],-1)
        theta=np.arctan2(np.hypot(p[:,0],p[:,1]),p[:,2])+g[:,2*j].double().numpy()
        invalid_theta|=(theta<0)|(theta>np.pi)
    order=np.argsort(-logits.numpy(),kind='stable')
    groups={}
    for name,count in [('top20',20),('top1pct',math.ceil(len(g)*.01))]:
        mask=np.zeros(len(g),dtype=bool); mask[order[:count]]=True
        groups[name]=dict(events=int(mask.sum()), weight_mass=float(w[mask].sum()),
            features={k:dict(top=describe(v[mask]),rest=describe(v[~mask])) for k,v in features.items()})
    correlations={k:float(spearmanr(logits.numpy(),v).statistic) if np.ptp(v)>0 and torch.std(logits)>0 else None for k,v in features.items()}
    regions=[]
    for threshold in (1e-6,1e-5,1e-4,1e-3,1e-2):
        for name,mask in [('acoplanarity',features['acoplanarity']<threshold),
                          ('acollinearity',features['acollinearity']<threshold),
                          ('joint',(features['acoplanarity']<threshold)&(features['acollinearity']<threshold))]:
            regions.append(dict(region=name,threshold_radians=threshold,events=int(mask.sum()),
                event_fraction=float(mask.mean()),raw_weight_mass=float(w[mask].sum())))
    return dict(groups=groups,spearman_logit=correlations,regions=regions,
        invalid_theta=dict(events=int(invalid_theta.sum()),raw_weight_mass=float(w[invalid_theta].sum()),
                           test_rows=np.flatnonzero(invalid_theta).tolist()),
        scope='Descriptive association, not causal feature attribution. No clipping or removal.',
        caveat=('For equal-magnitude tau directions calibration shift is half the opening deficit; these are not independent signals.' if include_calibration else
                'Calibration association omitted. All candidates retained; angular coordinates follow the saved trigonometric feature convention, with noncanonical theta explicitly flagged.'))


def metrics(t,g):
    t,g=t.double().reshape(-1),g.double().reshape(-1)
    if len(t)!=len(g) or not torch.isfinite(t).all() or not torch.isfinite(g).all():
        raise ValueError('Invalid aligned logits')
    return dict(bce=float(.5*(torch.nn.functional.softplus(-t).mean()+torch.nn.functional.softplus(g).mean())),
        auc=float(roc_auc_score(np.r_[np.ones(len(t)),np.zeros(len(g))],torch.cat([t,g]).numpy())),
        truth_logits=describe(t.numpy()),gen_logits=describe(g.numpy()),
        score_weight_concentration=weight_metrics(g))


def verify_logits(observed, expected):
    observed,expected=observed.cpu().double().reshape(-1),expected.cpu().double().reshape(-1)
    if observed.shape!=expected.shape:
        raise ValueError(f'Original logit shape mismatch: replay={tuple(observed.shape)} saved={tuple(expected.shape)}; no calibrated inference permitted')
    delta=(observed-expected).abs()
    if not torch.isfinite(observed).all() or not torch.allclose(observed,expected,atol=1e-4,rtol=1e-4):
        raise ValueError('Original logits do not reproduce saved classifier: '
            f'max_abs={float(delta.max()):.8g}, mean_abs={float(delta.mean()):.8g}, '
            f'replay_range=[{float(observed.min()):.8g},{float(observed.max()):.8g}], '
            f'saved_range=[{float(expected.min()):.8g},{float(expected.max()):.8g}]; '
            'no calibrated inference permitted')
    return float(delta.max())


def original_shard(n,rank,world):
    """Exact contiguous leading-dimension shard used by ratio export."""
    if world<1 or not 0<=rank<world or n<0:
        raise ValueError('invalid shard arguments')
    return (n*rank)//world,(n*(rank+1))//world


def paired_bce_change(scores):
    losses={stage:.5*(torch.nn.functional.softplus(-scores['truth_'+stage].double())+
                     torch.nn.functional.softplus(scores['gen_'+stage].double()))
            for stage in ('before','after')}
    delta=losses['after']-losses['before']
    return dict(after_minus_before=float(delta.mean()),
                paired_event_se=float(delta.std()/math.sqrt(len(delta))) if len(delta)>1 else None)


CLASSIFIER_DEFAULTS={
    'adapter_bottleneck':16,'body_only_checkpoint':False,
    'train_layernorm':False,'train_encoder':False,
    'train_grouped_sequential_embedding':False,'train_invisible_projector':False,
    'train_backbone':False,'train_last_pet_block':False,
    'asymmetric_attention':False,'periodic_pair_features':False,
    'topology_fourier_embedding':False,'topology_conditioning':False,
    'topology_pair_token':False,'relation_token_count':0,
    'visible_pair_rest_frame':False,'topology_max_harmonic':1,
    'topology_include_theta_pair':False,'topology_hidden_dim':64,
    'topology_embedding_dim':32,'topology_fusion_hidden_dim':64,
    'topology_dropout':.15,'topology_direct_logit':False,
    'topology_context_residual_scale':1.,'conditional_residual_rank':0,
    'head_dropout':.1,'decoder_hidden_dim':256,'decoder_layers':2,'decoder_heads':8,
}
AUDIT_OVERRIDE_KEYS=(
    'head_dropout','topology_dropout','decoder_hidden_dim','decoder_layers','decoder_heads',
    'periodic_pair_features','topology_fourier_embedding','topology_direct_logit',
    'topology_context_residual_scale','topology_conditioning','topology_pair_token','relation_token_count','visible_pair_rest_frame',
    'topology_max_harmonic','topology_include_theta_pair','topology_hidden_dim',
    'topology_embedding_dim','topology_fusion_hidden_dim','train_layernorm','train_encoder',
    'train_grouped_sequential_embedding','train_invisible_projector','train_backbone',
    'train_last_pet_block','asymmetric_attention')


def resolve_classifier_settings(raw):
    adaptive=raw['dgpo']['adaptive_omnifold']
    recal=adaptive['recalibration']
    base={key:recal.get(key,default) for key,default in CLASSIFIER_DEFAULTS.items()}
    audit=adaptive.get('audit_fit',{})
    overrides={key:audit[key] for key in AUDIT_OVERRIDE_KEYS if audit.get(key) is not None}
    checkpoint=raw['reward_config']['omnifold'].get('backbone_checkpoint')
    if not checkpoint:
        raise ValueError('Runtime has no reward_config.omnifold.backbone_checkpoint')
    return base,overrides,Path(checkpoint).expanduser().resolve()


def build_classifier(raw, state, spec, device):
    # Rebuild through the production builder. The exported model_state contains
    # the classifier bank and trainable body tensors, but intentionally omits
    # the frozen shared EveNet body. That body must come from the exact original
    # backbone checkpoint before the exported state is restored.
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EvenetAdapterModelBuilder,EventPackingSpec)
    base,overrides,checkpoint=resolve_classifier_settings(raw)
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    builder=EvenetAdapterModelBuilder(config=global_config,
        normalization_dict=load_normalization_dict(global_config),
        checkpoint_path=checkpoint,device=device,**base)
    model=builder.make_classifier(EventPackingSpec.from_dict(spec),'audit',reset=True,**overrides).to(device)
    actual=model.state_dict()
    if set(actual)!=set(state):
        raise ValueError(f'Classifier state keys differ: missing={set(actual)-set(state)}, extra={set(state)-set(actual)}')
    model.load_state_dict(state,strict=True)
    for key,value in model.state_dict().items():
        if value.dtype!=state[key].dtype or not torch.equal(value.cpu(),state[key].cpu()):
            raise ValueError(f'Classifier state mismatch: {key}')
    return model.eval().requires_grad_(False)


def prepare_output(path):
    """Overwrite only files owned by this diagnostic; preserve unknown files."""
    path=Path(path)
    if path.exists() and not path.is_dir():raise ValueError(f'Output is not a directory: {path}')
    path.mkdir(parents=True,exist_ok=True)
    owned=('tail_attribution.json','manifest.json','frozen_classifier_report.json',
           'scores_and_projected_candidates.pt','COMPLETE')
    for name in owned:
        target=path/name
        if target.exists():target.unlink()
    for target in path.glob('rank-*.pt'):target.unlink()
    ray_results=path/'ray_results'
    if ray_results.exists():shutil.rmtree(ray_results)


@torch.no_grad()
def score(model,c,z,device,batch_size):
    return torch.cat([model(c[i:i+batch_size].to(device),z[i:i+batch_size].to(device)).reshape(-1).cpu()
                      for i in range(0,len(c),batch_size)])


def worker(cfg):
    import ray.train
    import ray.train.torch
    import yaml
    from evenet.control.global_config import global_config
    ctx=ray.train.get_context(); rank=ctx.get_world_rank(); world=ctx.get_world_size()
    device=ray.train.torch.get_device()
    global_config.load_yaml(cfg['runtime'])
    raw=yaml.safe_load(Path(cfg['runtime']).read_text())
    torch.set_float32_matmul_precision(str(raw['dgpo'].get('float32_matmul_precision','medium')))
    b=load_artifact(cfg['artifact'])
    model=build_classifier(raw,b['model_state'],b['packing_spec'],device)
    start,stop=original_shard(len(b['test_truth']),rank,world)
    ids=torch.arange(start,stop)
    c=b['test_condition'][ids]; t=b['test_truth'][ids].reshape(-1,4); g=b['test_generated'][ids].reshape(-1,4)
    out={'ids':ids}
    # Exact replay requires both the original contiguous distributed shard and
    # the exported fit's validation batch size. A different row grouping can
    # select a different GPU GEMM/attention kernel and fail a strict logit check.
    score_batch=cfg['saved_validation_batch_size']
    out['truth_before']=score(model,c,t,device,score_batch)
    out['gen_before']=score(model,c,g,device,score_batch)
    verify_logits(out['truth_before'],b['truth_logits'].reshape(-1)[ids])
    verify_logits(out['gen_before'],b['gen_logits'].reshape(-1)[ids])
    print(f'[rank {rank}] original logits verified: rows={start}:{stop} batch={score_batch}',flush=True)
    fields=unpack(dict(test_condition=c,packing_spec=b['packing_spec']))
    tp,gp=projected_deltas(fields,t),projected_deltas(fields,g)
    # Forward recomputes candidate-dependent Fourier/decoder features. The
    # event context and saved training-only standardizer remain unchanged.
    out['truth_after']=score(model,c,tp,device,score_batch)
    out['gen_after']=score(model,c,gp,device,score_batch)
    out['truth_projected']=tp;out['gen_projected']=gp
    torch.save(out,Path(cfg['output'])/f'rank-{rank:03d}.pt')
    torch.distributed.barrier()
    ray.train.report({'rank0_scored_events':len(ids) if rank==0 else 0})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--score',action='store_true');p.add_argument('--runtime',type=Path)
    p.add_argument('--skip-calibration-association',action='store_true',help='Retain and flag noncanonical theta; describe saved angular features without performing calibration.')
    p.add_argument('--workers',type=int,default=16);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS'))
    p.add_argument('--no-wandb',action='store_true');p.add_argument('--run-id',default='h4calclf1')
    args=p.parse_args()
    if min(args.workers,args.batch_size)<1:p.error('Positive workers/batch size required')
    if args.score and (args.runtime is None or not args.runtime.is_file()):p.error('--score requires saved coverage runtime.yaml')
    ray=None
    RunConfig=ScalingConfig=FailureConfig=TorchTrainer=None
    if args.score:
        if not args.ray_address:
            p.error('--score requires an active Ray cluster: source scripts/nersc/start_interactive_ray.sh or pass --ray-address')
        import ray
        # Import all Ray Train modules before reserving an output directory or
        # starting W&B. On NERSC/Shifter this import can be slow on a cold
        # filesystem; an interruption must not leave a misleading partial run.
        from ray.train import RunConfig,ScalingConfig,FailureConfig
        from ray.train.torch import TorchTrainer
        try:
            ray.init(address=args.ray_address,runtime_env={'env_vars':{
                'PYTHONPATH':os.pathsep.join([str(ROOT/'evenet_dgpo'),str(ROOT/'scripts'),str(ROOT),os.environ.get('PYTHONPATH','')])}})
        except (ConnectionError,ValueError) as exc:
            p.error(f'Cannot connect to Ray at {args.ray_address}: {exc}')
        available_gpus=float(ray.cluster_resources().get('GPU',0) or 0)
        if available_gpus<args.workers:
            p.error(f'Ray cluster has {available_gpus:g} GPUs; --workers requires {args.workers}')
    b=load_artifact(args.directory)
    if args.workers>len(b['test_truth']):p.error('More workers than events')
    saved_score_batch=int(b.get('fit_config',{}).get('validation_batch_size',0))
    if args.score and saved_score_batch<1:p.error('Artifact is missing a positive fit_config.validation_batch_size')
    tail=tail_report(b,include_calibration=not args.skip_calibration_association)
    if args.output.resolve()==args.directory.resolve():p.error('Output may not overwrite the source artifact')
    prepare_output(args.output);write_json(args.output/'tail_attribution.json',tail)
    print(json.dumps(tail,indent=2,allow_nan=False),flush=True)
    if not args.score:return
    cfg=dict(artifact=str(args.directory.resolve()),output=str(args.output.resolve()),runtime=str(args.runtime.resolve()),
             workers=args.workers,requested_batch_size=args.batch_size,
             saved_validation_batch_size=saved_score_batch,ray_address=args.ray_address,
             classifier_fits=0,policy_updates=0)
    write_json(args.output/'manifest.json',cfg)
    run=None;success=False
    try:
        if not args.no_wandb:
            import wandb
            run=wandb.init(entity='ytchou97-university-of-washington',project='nu2flow-RL',id=args.run_id,resume='allow',
                name='What does calibration remove? | frozen H4 classifier | paired test',group='H4 ratio diagnostics',config=cfg)
            run.summary['phase']='verify_original_then_score_calibrated'
        TorchTrainer(train_loop_per_worker=worker,train_loop_config=cfg,
            scaling_config=ScalingConfig(num_workers=args.workers,use_gpu=True),
            run_config=RunConfig(name='frozen-calibration',storage_path=str(args.output.resolve()/'ray_results'),failure_config=FailureConfig(max_failures=0))).fit()
        n=len(b['test_truth']); merged={k:torch.empty(n) for k in ('truth_before','gen_before','truth_after','gen_after')}
        merged.update(truth_projected=torch.empty(n,4),gen_projected=torch.empty(n,4));seen=[]
        for rank in range(args.workers):
            part=torch.load(args.output/f'rank-{rank:03d}.pt',map_location='cpu',weights_only=True)
            seen.extend(part['ids'].tolist())
            for key in merged:merged[key][part['ids']]=part[key]
        if sorted(seen)!=list(range(n)):raise ValueError('Missing/duplicate event rows')
        result=dict(before=metrics(merged['truth_before'],merged['gen_before']),after=metrics(merged['truth_after'],merged['gen_after']),
            paired_bce_change=paired_bce_change(merged),
            logit_changes={side:describe((merged[side+'_after']-merged[side+'_before']).numpy()) for side in ('truth','gen')},
            original_max_abs_error={side:verify_logits(merged[side+'_before'],b[side+'_logits']) for side in ('truth','gen')},
            limitations=['Frozen-classifier response only, not Cij or closure.','Projection changes both classes and can be out of training distribution.',
                        'Post-projection exp(logit) is a score-derived weight, NOT a validated projected density ratio.',
                        'Reused test; no calibration, clipping, feature selection or refitting.'])
        write_json(args.output/'frozen_classifier_report.json',result)
        torch.save(merged,args.output/'scores_and_projected_candidates.pt')
        if run:
            run.log({f'paired_bce/{k}':v for k,v in result['paired_bce_change'].items() if v is not None})
            for stage in ('before','after'):
                run.log({f'{stage}/bce':result[stage]['bce'],f'{stage}/auc':result[stage]['auc'],
                    **{f'{stage}/score_weights/{k}':v for k,v in result[stage]['score_weight_concentration'].items() if v is not None}})
            run.summary['phase']='complete'
        (args.output/'COMPLETE').write_text('h4-classifier-calibration-v1\n')
        print(json.dumps(result,indent=2,allow_nan=False));success=True
    finally:
        if run:run.finish(exit_code=0 if success else 1)


if __name__=='__main__':main()
