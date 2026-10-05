"""Rescore fixed raw step1110 samples with the saved kinematic FiLM reward stack."""
import argparse
import json
import os
from pathlib import Path
import sys
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'evenet_dgpo'),str(ROOT)]
import numpy as np
import torch
import yaml
from sample_1110_cij import BASE, POLICY, validation_pool, weight_summary
from diagnose_ztautau_cij import source_ids, align, SOURCE_ID_COLUMNS, ANALYSIS_VERSION, run as cij_run

CHECKPOINT=BASE/'h4_kinematic_adaln_depth3_1110/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt'
RECOVERY_NAME='Does reweighting improve matched Cij? | saved FiLM scores | reconstruction controls'


def analysis_only(args):
    """Recover completed scoring even when the previous Cij stage failed."""
    output=args.analysis_only.resolve()
    args.candidates=output/'candidates.npz'
    cfg=json.loads((output/'manifest.json').read_text())
    if args.kappa_signs is None: raise ValueError('Cij requires explicit --kappa-signs')
    with np.load(args.candidates,allow_pickle=False) as f:
        if len(f['deltas'])!=cfg['events'] or f['log_ratio'].shape!=(cfg['events'],1):
            raise ValueError('Saved scores are incomplete or misaligned')
        metrics=weight_summary(f['log_ratio'])
    args.output=output;args.weight_mode='joint';args.seed=42;args.tolerance=1e-4;args.tt2l_repo=None
    import wandb
    with wandb.init(entity='ytchou97-university-of-washington',project='nu2flow-RL',mode='online',
        name=RECOVERY_NAME,
        group='H4 Cij reweighting',config={**cfg,'analysis_only':True,'kappa_signs':args.kappa_signs,
            'allow_truth_mismatch':args.allow_truth_mismatch,'analysis_version':ANALYSIS_VERSION,
            'analysis_events':str(args.events),'selection':args.selection,'bootstrap':args.bootstrap,
            'channel_diagnostics':getattr(args,'channel_diagnostics',False)},dir=str(output)) as run:
        print('WANDB:',run.url,flush=True)
        run.summary.update({'phase':'cij_analysis','cij/computed':False})
        run.log({f'raw_ratio/{k}':v for k,v in metrics.items()})
        cij_run(args)
        log_cij_reports(run,output)
        run.summary.update({'phase':'complete','cij/computed':True,'output':str(output)})
        (output/'COMPLETE').write_text(ANALYSIS_VERSION+'\n')


def log_truth_check(run,output):
    check=json.loads((output/'truth_convention_check.json').read_text())
    run.summary.update({'truth_check/passed':check['passed'],
        'truth_check/max_error':check['max_error'],
        'truth_check/events_above_tolerance':check['events_above_tolerance'],
        'truth_check/mismatch_explicitly_allowed':check.get('mismatch_explicitly_allowed',False),
        'truth_check/truth_used':check.get('truth_used','stored parquet truth cosines')})


def log_cij_reports(run,output):
    """Publish explicit references and full numerical reports, not only PNGs."""
    import wandb
    log_truth_check(run,output)
    run.summary.update({'cij/analysis_version':ANALYSIS_VERSION,
                        'cij/physics_unfolded':False})
    paths=[output/'truth_convention_check.json']
    for mode in ('raw','calibrated'):
        report=json.loads((output/f'{mode}.json').read_text())
        run.summary['cij/events']=report['events']
        run.summary['cij/candidates']=report['candidates']
        for reference,comparison in [('full_truth',report['comparison']),
                *[(name,item['comparison']) for name,item in report['reference_comparisons'].items()]]:
            for key,value in comparison['C_frobenius_error_change'].items():
                run.summary[f'cij/{mode}/{reference}/error_change/{key}']=value
        # Backward compatible names; explicitly marked as full-truth comparison.
        for key,value in report['comparison']['C_frobenius_error_change'].items():
            run.summary[f'cij/{mode}/error_change/{key}']=value
        run.summary[f'cij/{mode}/legacy_error_reference']='stored full truth'
        diag=report['diagnostics']
        run.summary[f'cij/{mode}/reweighted_event_ess_fraction']=diag['weight_health']['event_ess_fraction']
        run.summary[f'cij/{mode}/inverse_kappa_abs_p99']=diag['inverse_abs_product_quantiles']['0.99']
        run.summary[f'cij/{mode}/influence_top1pct_fraction']=diag['influence_top1pct_fraction']['reweighted']
        angular=json.loads((output/f'{mode}_angular_moments.json').read_text())
        for ref,item in [('full_truth',angular),('matched_target',angular['reference_comparisons']['matched_target'])]:
            for key,value in item['comparison']['C_frobenius_error_change'].items():
                run.summary[f'angular_moments/{mode}/{ref}/error_change/{key}']=value
        for suffix,label in [('', 'full_truth'),('_matched','matched_target')]:
            path=output/f'{mode}{suffix}.png'
            run.log({f'cij/{mode}/{label}':wandb.Image(str(path))})
            paths.extend([path,path.with_suffix('.json')])
        paths.append(output/f'{mode}_angular_moments.json')
        channel_path=output/f'{mode}_channels.json'
        if channel_path.exists() and getattr(run.config,'channel_diagnostics',False):
            channel=json.loads(channel_path.read_text())
            table=[]
            moment_rows=[]
            for item in channel['channels']:
                report=item.get('report',{})
                matched=report.get('reference_comparisons',{}).get('matched_target',{})
                change=matched.get('comparison',{}).get('C_frobenius_error_change',{})
                ci=change.get('ci95',[None,None])
                table.append([item['event_category'],item['events'],item['base_fraction'],
                    item['reweighted_fraction'],change.get('value'),*ci,
                    report.get('weight_health',{}).get('event_ess_fraction'),item['status']])
                moments=item.get('angular_moments',{}).get('references',{}).get('matched_target')
                if moments:
                    for i,axis_a in enumerate(('k','r','n')):
                        for j,axis_b in enumerate(('k','r','n')):
                            row=[item['event_category'],axis_a+axis_b,item['events']]
                            for arm in ('truth','unweighted','reweighted'):
                                value=moments['results'][arm]
                                row.extend([value['moment'][i][j],value['moment_ci95'][0][i][j],
                                            value['moment_ci95'][1][i][j]])
                            row.extend([moments['absolute_error_change'][i][j],
                                moments['absolute_error_change_ci95'][0][i][j],
                                moments['absolute_error_change_ci95'][1][i][j]])
                            moment_rows.append(row)
                    run.summary[f'angular_channels/{mode}/{item["event_category"]}/matched_error_change']=moments['error_change']
            if moment_rows:
                run.log({f'angular_channels/{mode}/matched_target':wandb.Table(columns=[
                    'category','component','events','target','target_lo95','target_hi95',
                    'unweighted','unweighted_lo95','unweighted_hi95',
                    'reweighted','reweighted_lo95','reweighted_hi95',
                    'absolute_error_change','change_lo95','change_hi95'],data=moment_rows)})
            run.log({f'cij/{mode}/channels':wandb.Table(columns=['category','events','base_fraction',
                'reweighted_fraction','matched_error_change','lo95','hi95','ess_fraction','status'],data=table)})
            for ref,errors in channel['decomposition'].get('errors',{}).items():
                for arm,value in errors.items():
                    run.summary[f'cij/{mode}/channel_mix/{ref}/{arm}_error']=value
            paths.append(channel_path)
    for path in paths:
        run.save(str(path),base_path=str(output),policy='now')


def validate_stack(checkpoint):
    if int(checkpoint.get('global_step',-1))!=0: raise ValueError('Requires pre-update step0 classifier checkpoint')
    stack=checkpoint['dgpo_omnifold_reward_stack']
    reward=stack['reward']
    if not reward.get('increments'): raise ValueError('No fitted classifiers in checkpoint')
    if reward.get('log_ratio_clip') is not None or float(reward['tempering'])!=1:
        raise ValueError('Requested raw ratio, but checkpoint is clipped or tempered')
    if any(float(t)!=1 for t in reward.get('iteration_temperatures',[])):
        raise ValueError('Checkpoint has per-iteration tempering')
    return stack


@torch.no_grad()
def worker(cfg):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.dgpo_reward import build_uninstalled_ztautau_omnifold_reward
    ctx=ray.train.get_context();rank=ctx.get_world_rank();world=ctx.get_world_size()
    device=ray.train.torch.get_device()
    global_config.load_yaml(cfg['runtime'])
    raw=yaml.safe_load(Path(cfg['runtime']).read_text())
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    stack=validate_stack(checkpoint)
    reward=build_uninstalled_ztautau_omnifold_reward(
        backbone_checkpoint=raw['reward_config']['omnifold']['backbone_checkpoint'],
        training_config=global_config,normalization_dict=load_normalization_dict(global_config),
        device=device,classifier_config=raw['dgpo']['adaptive_omnifold']['recalibration'],
        candidate_consensus=stack.get('candidate_consensus'))
    reward.load_stack_payload(stack,allow_source_bundle_migration=False)
    model=reward.frozen_reward.eval()
    pool=torch.load(Path(cfg['output'])/'pool.pt',map_location='cpu',weights_only=True)
    positions=torch.arange(rank,len(pool['condition']),world)
    scores=[]
    for offset in range(0,len(positions),cfg['batch_size']):
        ids=positions[offset:offset+cfg['batch_size']]
        scores.append(model(pool['condition'][ids].to(device),pool['deltas'][ids].reshape(len(ids),1,4).to(device)).reshape(-1).cpu())
        print(f'[FiLM score rank={rank}] {offset+len(ids)}/{len(positions)}',flush=True)
    torch.save(dict(positions=positions,logits=torch.cat(scores)),Path(cfg['output'])/f'rank-{rank:03d}.pt')
    ray.train.report({'scored_events':len(positions)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,default=BASE/'h4_step1110_cij_fresh/sample-e1b8977b0c')
    p.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    p.add_argument('--events',type=Path,default=BASE/'diffusion_val_20pct_seed42_stic_filtered_test1/val')
    p.add_argument('--output',type=Path,default=BASE/'h4_film_step1110_cij')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--kappa-signs',type=int,nargs=2,choices=[-1,1])
    p.add_argument('--selection')
    p.add_argument('--channel-diagnostics',action='store_true')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--allow-truth-mismatch',action='store_true',
                   help='Proceed with stored truth while recording the unresolved angle discrepancy')
    p.add_argument('--analysis-only',type=Path,help='Existing rescore directory; skip Ray, model loading and GPU scoring')
    args=p.parse_args()
    if args.analysis_only is not None:
        analysis_only(args)
        return
    if min(args.workers,args.batch_size)<1:p.error('Positive sizes required')
    if not (args.samples/'COMPLETE').is_file():raise ValueError('Incomplete sample source')
    source=json.loads((args.samples/'manifest.json').read_text())
    if source['checkpoint']!=str(POLICY) or source['weights']!='raw_state_dict_only' or source['candidates']!=1:
        raise ValueError('Requires raw step1110 K1 samples')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    stack=validate_stack(checkpoint)
    spec=stack['reward']['increments'][0]['packing_spec']
    pool=validation_pool(args.events,{'packing_spec':spec})
    with np.load(args.samples/'candidates.npz') as f:
        # Historical classifier overlap flags do NOT apply to this new reward.
        arrays={k:f[k] for k in SOURCE_ID_COLUMNS if k in f}
        arrays.update(deltas=f['deltas'],truth_deltas=f['truth_deltas'])
    keys=[k for k in SOURCE_ID_COLUMNS if k in arrays]
    order=align(source_ids(pool,keys),source_ids(arrays,keys))
    if not np.allclose(pool['truth'][order].numpy(),arrays['truth_deltas'],rtol=0,atol=1e-6):
        raise ValueError('Aligned truth targets changed')
    output=args.output.resolve()/('rescore-'+uuid.uuid4().hex[:10]);output.mkdir(parents=True)
    torch.save(dict(condition=pool['condition'][order],deltas=torch.from_numpy(arrays['deltas'])),output/'pool.pt')
    from train_neutrino_backend import read_yaml,read_overlay_yaml,deep_update,absolutize_default_paths
    raw=deep_update(read_yaml(ROOT/'config/train_diffusion_nersc.yaml'),read_overlay_yaml(ROOT/'config/dgpo_h4_kinematic_adaln_depth3.yaml'))
    raw=absolutize_default_paths(raw,ROOT/'config')
    raw.setdefault('compat',{}).update(backend='dgpo-evenet',repo_root=str(ROOT))
    runtime=output/'runtime.yaml';runtime.write_text(yaml.safe_dump(raw,sort_keys=False))
    cfg=dict(runtime=str(runtime),checkpoint=str(args.checkpoint),output=str(output),batch_size=args.batch_size,
             workers=args.workers,samples=str(args.samples),events=len(order),event_source=str(args.events.resolve()),candidate_regenerations=0,
             classifier_fits=0,weight_mode='joint',classifier_overlap='Not established for this saved stack; exploratory validation result',
             increment_count=len(stack['reward']['increments']),tempering=1,log_ratio_clip=None)
    cfg.update(channel_diagnostics=args.channel_diagnostics,
        allow_truth_mismatch=args.allow_truth_mismatch,analysis_version=ANALYSIS_VERSION,
               kappa_signs=args.kappa_signs,selection=args.selection,bootstrap=args.bootstrap)
    (output/'manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    import wandb
    import ray
    from ray.train import RunConfig,ScalingConfig,FailureConfig
    from ray.train.torch import TorchTrainer
    with wandb.init(entity='ytchou97-university-of-washington',project='nu2flow-RL',mode='online',
        name='Does FiLM reweighting improve spin correlation? | fixed step-1110 samples | raw ratio',
        group='H4 Cij reweighting',config=cfg,dir=str(output)) as run:
        print('WANDB:',run.url,flush=True);run.summary['phase']='rescoring'
        ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto',runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join([str(ROOT/'evenet_dgpo'),str(ROOT/'scripts'),str(ROOT)])}})
        if ray.cluster_resources().get('GPU',0)<args.workers:raise ValueError('Insufficient GPUs')
        TorchTrainer(train_loop_per_worker=worker,train_loop_config=cfg,
            scaling_config=ScalingConfig(num_workers=args.workers,use_gpu=True),
            run_config=RunConfig(name='film-rescore',storage_path=str(output/'ray_results'),failure_config=FailureConfig(max_failures=0))).fit()
        logits=np.empty(len(order));seen=[]
        for rank in range(args.workers):
            part=torch.load(output/f'rank-{rank:03d}.pt',weights_only=True,map_location='cpu')
            ids=part['positions'].numpy();seen.extend(ids.tolist());logits[ids]=part['logits'].numpy()
        if sorted(seen)!=list(range(len(order))):raise ValueError('Missing/duplicate scored events')
        run.log({f'raw_ratio/{k}':v for k,v in weight_summary(logits).items()})
        args.candidates=output/'candidates.npz'
        np.savez_compressed(args.candidates,**arrays,log_ratio=logits[:,None])
        run.summary['cij/computed']=False
        if args.kappa_signs is not None:
            args.output=output;args.weight_mode='joint';args.seed=42;args.tolerance=1e-4;args.tt2l_repo=None
            cij_run(args)
            log_cij_reports(run,output)
            run.summary['cij/computed']=True
        (output/'COMPLETE').write_text(ANALYSIS_VERSION+'\n')
        run.summary['phase']='complete';run.summary['output']=str(output)
        print('OUTPUT:',output,flush=True)


if __name__=='__main__':main()
