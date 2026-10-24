"""User-launched16-GPU classifier A/B on fixed step100 samples; NO DGPO or generation.

check: config only. verify: read-only source validation. prepare: snapshot cached
inputs on CPU. run: prepare, extract frozen features, fit two fresh heads, score
all saved K8 candidates and report. Never submits an allocation or policy update.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts.tau_physics_branch import physics_features
from scripts.train_diffusion_film_10pct import TRAIN,VAL


def settings(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if (cfg['workers']!=16 or cfg['policy_step']!=100 or cfg['validation_candidates']!=8
            or cfg['fit']['batch_size']!=1024 or cfg['bootstrap']<20 or cfg['seed']!=42):
        raise ValueError('Pinned experiment requires step100, K8,16 GPUs, batch1024, seed42')
    if cfg['fit']['min_steps']<1000 or cfg['fit']['min_selected_steps']<1000:
        raise ValueError('Both heads must meet the minimum fitting/selection budget')
    if cfg['fit']['epochs']*22<cfg['fit']['min_selected_steps']:
        raise ValueError('Insufficient classifier fit budget')
    if Path(cfg['output']).resolve().is_relative_to(Path(cfg['source_directory']).resolve()):
        raise ValueError('Experiment output must be separate from the DGPO run')
    if len(cfg['logger']['name'])>96 or len(cfg['logger']['name'].split(' | '))<3:
        raise ValueError('Use a readable W&B display name')
    return cfg


def verify_sources(cfg):
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from RL.DGPO_neutrino.conditional_tau_reward import validate_head
    root=Path(cfg['source_directory']); runtime=yaml.safe_load((root/'runtime.yaml').read_text())
    for path,expected in ((TRAIN,416701),(VAL,119002)):
        if validate_filtered_dataset(path)['rows']!=expected:raise ValueError('Filtered manifest changed')
    if runtime['platform']['data_parquet_dir']!=TRAIN or runtime['platform']['data_parquet_val_dir']!=VAL:
        raise ValueError('Wrong source populations')
    tau=runtime['dgpo']['tau_ratio']
    if tau['policy_conditioning_contract']!='supervised_process_labels_v2':raise ValueError('Requires actual labels')
    if tau['source_global_step']!=18655 or tau['source_completed_epochs']!=287:raise ValueError('Wrong frozen trunk source')
    checkpoint=torch.load(cfg['policy_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if checkpoint['global_step']!=cfg['policy_step']:raise ValueError('Wrong actor checkpoint provenance')
    del checkpoint
    step=f"{cfg['policy_step']:08d}"
    validation=root/f'tau_diagnostics/eval-step-{step}'
    training=root/f'tau_diagnostics/refit-step-{step}'
    report=json.loads((validation/'report.json').read_text())
    if report['policy_step']!=cfg['policy_step'] or report['events']!=119002 or report['candidates']!=8:
        raise ValueError('Incomplete/wrong fixed validation endpoint')
    fit=json.loads((training/'best.json').read_text())
    if fit['best_steps']<1000 or not fit['minimum_fit_steps_met']:raise ValueError('Source refit incomplete')
    head=torch.load(training/'best.pt',map_location='cpu',weights_only=True)
    validate_head(head)
    if head['source_policy_step']!=cfg['policy_step'] or head['candidate_dim']!=533 or head['ratio_bound']!=30:
        raise ValueError('Wrong current-policy fitting features/head')
    for key,value in cfg['fit'].items():
        if key in head and head[key]!=value:raise ValueError('Unmatched production fit setting: '+key)
    from scripts.train_conditional_spin_ratio import build_classifier
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed']);control=build_classifier(head)
        torch.manual_seed(cfg['seed']);physics=build_classifier(arm_config(head,'physics'))
    for key,value in control.state_dict().items():
        if not torch.equal(value,physics.base.state_dict()[key]):raise ValueError('Shared initialization differs')
    print('HEADS OK: control parameters='+str(sum(p.numel() for p in control.parameters()))+
          ', physics parameters='+str(sum(p.numel() for p in physics.parameters())),flush=True)
    for label,folder in [('train',training),('validation',validation)]:
        for rank in range(16):
            if not (folder/f'candidates/rank-{rank:02d}.npz').is_file():raise ValueError('Missing candidate shard')
    return root,training,validation,head,report


def load_panel(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def merge_candidates(folder,n,k,feature_dim=533):
    deltas=np.empty((n,k,2,2),np.float32);features=np.empty((n,feature_dim),np.float32);seen=np.zeros(n,int)
    for rank in range(16):
        with np.load(folder/f'rank-{rank:02d}.npz',allow_pickle=False) as f:
            pos=f['positions']
            if not np.array_equal(pos,np.arange(rank,n,16)):raise ValueError('Candidate rank/event order differs')
            if f['deltas'].shape!=(len(pos),k,2,2) or f['features'].shape!=(len(pos),feature_dim):
                raise ValueError('Wrong saved candidate shape')
            np.add.at(seen,pos,1);deltas[pos]=f['deltas'];features[pos]=f['features']
    if not np.all(seen==1) or not np.isfinite(deltas).all() or not np.isfinite(features).all():
        raise ValueError('Incomplete/nonfinite candidates')
    return deltas,features


def prepare(cfg):
    root,training,validation,head,source_report=verify_sources(cfg)
    out=Path(cfg['output'])
    if (out/'PREPARED').is_file():
        if json.loads((out/'settings.json').read_text())!=cfg:raise ValueError('Existing preparation has different settings')
        return out
    if out.exists() and any(out.iterdir()):raise ValueError('Partial output exists; choose another readable output name')
    out.mkdir(parents=True,exist_ok=True)
    (out/'settings.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copy2(root/'runtime.yaml',out/'source_actor_runtime.yaml')
    shutil.copy2(validation/'report.json',out/'source_validation_report.json')
    shutil.copy2(training/'best.pt',out/'source_fitted_head.pt')
    # Pin the independent epoch287 feature trunk and its schema/normalization.
    bundle=torch.load(root/'initial_tau_reward.pt',map_location='cpu',weights_only=True)
    for key,name in [('source_checkpoint','feature_source.ckpt'),('normalization_file','normalization.pt')]:
        shutil.copy2(bundle[key],out/name);bundle[key]=str(out/name)
    source=yaml.safe_load(Path(bundle['backbone_runtime']).read_text())
    source['options']['Dataset']['normalization_file']=str(out/'normalization.pt')
    (out/'feature_runtime.yaml').write_text(yaml.safe_dump(source,sort_keys=False))
    bundle['backbone_runtime']=str(out/'feature_runtime.yaml')
    torch.save(bundle,out/'feature_bundle.pt')
    panels={label:load_panel(root/(label+'_panel.npz')) for label in ('train','validation')}
    if np.intersect1d(panels['train']['source_ids'],panels['validation']['source_ids']).size:
        raise ValueError('Fitting and external events overlap')
    from RL.DGPO_neutrino.conditional_tau_reward import condition_inputs
    from scripts.conditional_tau_preprocessing import layout
    from scripts.tau_physics_branch_report import terms
    for label,n,folder,k in [('train',416701,training,1),('validation',119002,validation,8)]:
        p=panels[label]
        if len(p['source_ids'])!=n or len(np.unique(p['source_ids']))!=n or not bool(p['truth_features_ready']):
            raise ValueError('Incomplete/duplicate panel')
        if not np.isfinite(p['candidate_truth']).all() or p['candidate_truth'].shape!=(n,533):raise ValueError('Wrong truth features')
        for start in range(0,n,2048):
            sl=slice(start,start+2048)
            c=condition_inputs(p['raw_condition'][sl],p['category'][sl],bundle['condition_mean'],bundle['condition_scale'],head['packing_spec'])
            np.testing.assert_array_equal(c,p['condition'][sl])
        d,f=merge_candidates(folder/'candidates',n,k)
        geometry_truth=physics_features(p['visible_a'],p['visible_b'],p['truth_deltas'])
        if label=='train':
            geometry_gen=physics_features(p['visible_a'],p['visible_b'],d[:,0])
            arrays=dict(condition=p['condition'],candidate_truth=np.c_[p['candidate_truth'],geometry_truth],
                candidate_generated=np.c_[f,geometry_gen],event_weight=p['event_weight'],split=p['split'],source_ids=p['source_ids'])
            if (np.count_nonzero(p['split']==0),np.count_nonzero(p['split']==1))!=(354488,62213):
                raise ValueError('Training/selection split changed')
            for key,value in arrays.items():np.save(out/('train_'+key+'.npy'),value)
        else:
            # Replay the saved DGPO endpoint before comparing any new heads.
            t=terms(p,p['truth_deltas']);g=np.mean([terms(p,d[:,j]) for j in range(k)],axis=0)
            np.testing.assert_allclose(np.average(t,weights=p['event_weight'],axis=0).reshape(3,3),source_report['truth'],rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(np.average(g,weights=p['event_weight'],axis=0).reshape(3,3),source_report['generated'],rtol=1e-7,atol=1e-7)
            with np.load(validation/'measurements.npz',allow_pickle=False) as measurements:
                np.testing.assert_array_equal(measurements['source_ids'],p['source_ids'])
                np.testing.assert_array_equal(measurements['deltas'],d)
            np.save(out/'validation_deltas.npy',d)
            np.save(out/'validation_candidate0.npy',f)
            np.savez(out/'validation_panel.npz',**p)
        del d,f
    base=copy.deepcopy(head);base.pop('state_dict')
    for key in ('epoch','optimizer_steps','val_bce'):base.pop(key,None)
    base.update(cfg['fit']);base.update(seed=cfg['seed'],physics_branch=None)
    torch.save(base,out/'head_config.pt')
    (out/'PREPARED').write_text('Pinned step100 samples, original splits, source normalization, and physics endpoint verified.\n')
    return out


def arm_config(base,arm):
    cfg=copy.deepcopy(base)
    if arm=='physics':
        cfg.update(candidate_dim=base['candidate_dim']+15,
            physics_branch=dict(version=1,base_candidate_dim=base['candidate_dim'],width=64))
    elif arm!='control':raise ValueError('Unknown experiment arm')
    return cfg


def worker(cfg):
    import ray.train
    from ray.train.torch import get_device
    from RL.DGPO_neutrino.conditional_tau_cycle import fit_head,panel_batch,barrier
    from RL.DGPO_neutrino.conditional_tau_reward import load_reward,feature_precision
    from scripts.train_conditional_spin_ratio import build_classifier
    ctx=ray.train.get_context();rank=ctx.get_world_rank();world=ctx.get_world_size();device=get_device()
    if world!=16 or device.type!='cuda':raise ValueError('Real-case execution requires16 GPUs')
    torch.set_num_threads(1)
    out=Path(cfg['output']);base=torch.load(out/'head_config.pt',map_location='cpu',weights_only=True)
    train={key:np.load(out/('train_'+key+'.npy'),mmap_mode='r') for key in
           ('condition','candidate_truth','candidate_generated','event_weight','split','source_ids')}
    run=None
    if rank==0:
        import wandb
        run=wandb.init(entity=cfg['logger']['entity'],project=cfg['logger']['project'],name=cfg['logger']['name'],
            group=cfg['logger']['group'],dir=str(out),config={**cfg,'policy_updates':0,'candidate_generation':False,
            'arms':['control','physics'],'ratio':'single jointly fitted paired-BCE score, cap30'},
            tags=['classifier-only','fixed-candidates','physics-branch','16-gpu','process-labels'])
        (out/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url)))
        for arm in ('control','physics'):
            run.define_metric(arm+'/*',step_metric=arm+'/optimizer_steps')
    try:
        for arm in ('control','physics'):
            folder=out/arm;folder.mkdir(exist_ok=True)
            acfg=arm_config(base,arm)
            if rank==0:
                probe=build_classifier(acfg)
                run.summary[arm+'/parameters']=sum(p.numel() for p in probe.parameters())
                del probe
            arrays=dict(train)
            if arm=='control':
                for key in ('candidate_truth','candidate_generated'):arrays[key]=train[key][:,:base['candidate_dim']]
            def emit(row):
                if run is not None:run.log({arm+'/'+k:v for k,v in row.items()})
                print(f"[{arm}] epoch={row['epoch']} updates={row['optimizer_steps']} val_bce={row['val_bce']:.6f}",flush=True)
            with feature_precision():
                model,saved,status=fit_head(acfg,arrays,folder/'best.pt',device,rank,world,emit)
            if saved['optimizer_steps']<1000:raise ValueError('Undertrained selected head')
            del model,arrays
            if rank==0:run.summary.update({arm+'/fit/'+key:value for key,value in status.items()})
        # Both fits complete before any external scoring; never select using Cij/test.
        panel=load_panel(out/'validation_panel.npz');all_d=np.load(out/'validation_deltas.npy',mmap_mode='r')
        candidate0=np.load(out/'validation_candidate0.npy',mmap_mode='r')
        idx=np.arange(rank,len(all_d),world)
        reward=load_reward(out/'feature_bundle.pt',device)
        heads={}
        for arm in ('control','physics'):
            saved=torch.load(out/arm/'best.pt',map_location='cpu',weights_only=True)
            model=build_classifier(saved).to(device);model.load_state_dict(saved['state_dict'],strict=True)
            heads[arm]=model.eval().requires_grad_(False)
        scores={a:np.empty((len(idx),8),np.float32) for a in heads};positives={a:np.empty(len(idx),np.float32) for a in heads}
        with torch.no_grad(),feature_precision():
            for start in range(0,len(idx),256):
                take=idx[start:start+256];b=panel_batch(panel,take,reward.spec,device)
                c=panel['condition'][take];ct=torch.as_tensor(c,device=device)
                truth=np.c_[panel['candidate_truth'][take],physics_features(panel['visible_a'][take],panel['visible_b'][take],panel['truth_deltas'][take])]
                for arm,model in heads.items():
                    x=truth[:,:533] if arm=='control' else truth
                    positives[arm][start:start+len(take)]=model(ct,torch.as_tensor(x,device=device)).cpu().numpy()
                for j in range(8):
                    d=np.array(all_d[take,j]);cc,f=reward.features(b,torch.as_tensor(d,device=device))
                    np.testing.assert_allclose(cc,c,rtol=1e-6,atol=1e-6)
                    if j==0:np.testing.assert_allclose(f,candidate0[take],rtol=1e-4,atol=1e-3)
                    extended=np.c_[f,physics_features(panel['visible_a'][take],panel['visible_b'][take],d)]
                    for arm,model in heads.items():
                        x=f if arm=='control' else extended
                        scores[arm][start:start+len(take),j]=model(ct,torch.as_tensor(x,device=device)).cpu().numpy()
                if rank==0:run.log({'scoring/events_per_rank':start+len(take)})
        np.savez(out/f'scores-rank-{rank:02d}.npz',positions=idx,source_ids=panel['source_ids'][idx],
            **scores,**{arm+'_truth':v for arm,v in positives.items()})
        barrier(world)
        if rank==0:
            from scripts.tau_physics_branch_report import write_report
            from scripts.train_conditional_spin_ratio import pair_metrics
            full={a:np.empty((len(all_d),8),np.float32) for a in heads};truth={a:np.empty(len(all_d),np.float32) for a in heads}
            seen=np.zeros(len(all_d),int)
            for rr in range(world):
                with np.load(out/f'scores-rank-{rr:02d}.npz',allow_pickle=False) as f:
                    pos=f['positions'];np.add.at(seen,pos,1)
                    np.testing.assert_array_equal(f['source_ids'],panel['source_ids'][pos])
                    for a in heads:full[a][pos]=f[a];truth[a][pos]=f[a+'_truth']
            if not np.all(seen==1):raise ValueError('Missing or duplicate scores')
            np.savez(out/'scores.npz',source_ids=panel['source_ids'],**full,**{a+'_truth':v for a,v in truth.items()})
            report,plots=write_report(out,panel,all_d,full,cfg['bootstrap'],cfg['bootstrap_seed'])
            test={a:pair_metrics(truth[a],full[a][:,0],panel['event_weight']) for a in heads}
            (out/'heldout_classifier.json').write_text(json.dumps(test,indent=2))
            run.summary.update({'policy_updates':0,'phase':'complete','physics_minus_control_error':report['comparisons']['physics_minus_control']['error']['delta']})
            for a in heads:run.summary.update({a+'/test/'+key:value for key,value in test[a].items()})
            for a,values in report['arms'].items():
                run.summary.update({a+'/physics/'+key:value for key,value in values.items() if isinstance(value,(int,float))})
            for observable,arms in report['distributions'].items():
                for a,values in arms.items():
                    run.summary.update({a+'/distribution/'+observable+'/'+key:value for key,value in values.items()})
            for key,row in report['comparisons']['physics_minus_control'].items():
                run.summary.update({'comparison/'+key:row['delta'],'comparison/'+key+'_ci_low':row['ci95'][0],'comparison/'+key+'_ci_high':row['ci95'][1]})
            run.log({'plots/'+path.stem:wandb.Image(str(path)) for path in plots})
            for name in ('report.json','histograms.json','heldout_classifier.json','settings.json'):
                run.save(str(out/name),base_path=str(out),policy='now')
            (out/'COMPLETE').write_text('Both fresh classifier fits and held-out distribution/physics comparison completed; no policy update.\n')
        barrier(world)
        ray.train.report({'completed':1})
    finally:
        if run is not None:run.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('phase',choices=('check','verify','prepare','run'),default='check',nargs='?')
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto');args=p.parse_args()
    cfg=settings(args.config)
    if args.phase=='check':print('CONFIG OK: classifier-only A/B, fixed step100,16 GPUs,cap30');return
    if args.phase=='verify':verify_sources(cfg);print('SOURCE OK: completed step100 samples and filtered manifests');return
    out=prepare(cfg)
    if args.phase=='prepare':print('PREPARED:',out);return
    if (out/'STARTED').exists():raise ValueError('Experiment already started; choose a fresh output for a new comparison')
    import ray
    from ray.train import RunConfig,ScalingConfig,FailureConfig
    from ray.train.torch import TorchTrainer
    ray.init(address=args.ray_address,runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join((str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')))}})
    try:
        if ray.cluster_resources().get('GPU',0)<16:raise ValueError('Requires existing16-GPU Ray allocation')
        (out/'STARTED').write_text('User launched classifier-only experiment.\n')
        TorchTrainer(worker,train_loop_config=cfg,scaling_config=ScalingConfig(num_workers=16,use_gpu=True),
            run_config=RunConfig(name='classifier-physics-comparison',storage_path=str(out/'ray_results'),
                                failure_config=FailureConfig(max_failures=0))).fit()
    finally:ray.shutdown()
    print('REPORT:',out/'report.json')


if __name__=='__main__':main()
