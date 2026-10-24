"""User-launched adapter over native EveNet DGPO, with periodic General refits."""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
import yaml
from experiments.ttbar_general_classifier import run as classifier
from experiments.ttbar_dgpo_general.reward import GeneralOracleReward,load_feature_model


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def validate_settings(c):
    for key,val in dict(workers=16,batch_size=512,candidates=8,seed=42).items():
        if c[key]!=val:raise ValueError(f'This arm requires {key}={val}')
    if c['mixture']!=dict(oracle='reco_top_truthnu_p4',alpha=.3,temperature=1.):raise ValueError('Mixture contract differs')
    if c['refit']['every_steps']<=0 or not c['refit']['require_plateau']:raise ValueError('Plateau-selected periodic refits required')
    if c['refit']['batch_size']!=2048:raise ValueError('Classifier refit batch must be 2048 per GPU')
    if c['rollout_chunk_k']*c['batch_size']>1024:raise ValueError('Declared rollout forward cap is 1024 candidates')
    if len(c['wandb']['run_name'])>=96:raise ValueError('W&B display name too long')


def prepare(c,resume=None):
    validate_settings(c)
    src=Path(c['classifier_directory']);out=Path(c['output'])
    if src==out or src in out.parents or out in src.parents:raise ValueError('Keep source/output separate')
    names=['classifier-best.pt','normalization-classifier.pt','policy.ckpt','policy.yaml','normalization.pt','cache_manifest.json','CACHE_COMPLETE.json','FIT_COMPLETE.json']
    for name in names:
        if not (src/name).is_file():raise FileNotFoundError(src/name)
    fit=json.loads((src/'FIT_COMPLETE.json').read_text())
    if not fit['plateau']:raise ValueError('Initial classifier did not reach validation plateau')
    manifest=json.loads((src/'cache_manifest.json').read_text())
    if manifest['experiment']['cache']['batch_size']!=c['generation_batch_size']:
        raise ValueError('Frozen positive-cache reuse requires the original source row chunks')
    fitted=torch.load(src/'classifier-best.pt',map_location='cpu',weights_only=False)
    stats=torch.load(src/'normalization-classifier.pt',map_location='cpu',weights_only=True)
    for name in ('mean','scale'):
        if not torch.equal(fitted['stats'][name],stats[name]):raise ValueError('Classifier feature normalization differs')
    if fitted['stats']['oracle_scale']!=stats['oracle_scale'] or stats['oracle_reference']!='reco_top_truthnu_p4':
        raise ValueError('Oracle scale/reference differs from the successful classifier')
    policy=yaml.safe_load((src/'policy.yaml').read_text())
    p=policy['options']['Training'];components=p['Components']
    if components['Assignment']['include'] or p['EMA']['enable']:raise ValueError('Expected assignment/EMA off')
    if policy['options']['Dataset']['dataset_limit']!=1:raise ValueError('Require full MG5')
    for spec in manifest['inputs'].values():
        for path,size in zip(spec['files'],spec['file_sizes']):
            if Path(path).stat().st_size!=size:raise ValueError('Parquet population changed: '+path)
    # Reuse the cache's complete resolved architecture, not a mutable default YAML.
    runtime=copy.deepcopy(policy)
    runtime['platform'].update(number_of_workers=c['workers'],batch_size=c['batch_size'],
        resources_per_worker={'CPU':c['cpus_per_worker'],'GPU':1},prefetch_batches=2,use_gpu=True)
    runtime['rl']={'enabled':True}
    t=runtime['options']['Training'];t.update(epochs=c['epochs'],total_epochs=c['epochs'],seed=c['seed'],
        model_checkpoint_load_path=str(Path(resume).resolve()) if resume else None,
        pretrain_model_load_path=str(src/'policy.ckpt'),model_checkpoint_save_path=str(out/'checkpoints'),
        learning_rate=c['learning_rate'],learning_rate_body=c['backbone_learning_rate'],weight_decay=c['weight_decay'],
        learning_rate_warm_up_epochs=0,learning_rate_warm_up_factor=1.)
    # Every parameter still has a native optimizer group; explicitly retain body/head LR split.
    for name,settings in t['Components'].items():
        if not isinstance(settings,dict):continue
        settings['learning_rate']=c['backbone_learning_rate'] if name in ('GlobalEmbedding','PET','ObjectEncoder') else c['learning_rate']
        settings['weight_decay']=c['weight_decay'];settings['warm_up']=False
    t['EMA'].update(enable=False,use_for_generation=False,replace_model_after_load=False)
    runtime['options']['prediction']={'extra_save':True} # retain oracle/physics metadata; assignment loss stays disabled
    runtime['logger'].update(name=out.name,save_dir=str(out/'logs'),wandb=c['wandb'])
    runtime['reward_config']={'source':'ttbar_general_top_mixture','weight':1.}
    runtime['dgpo']=dict(beta=c['beta'],K=c['candidates'],validation_K=c['candidates'],
        num_ddim_steps=c['ddim_steps'],validation_num_ddim_steps=c['ddim_steps'],
        num_train_timesteps=c['policy_timesteps'],advantage_estimator='leave_one_out_unscaled',adv_clip_max=None,
        grad_clip_norm=1.,policy_eval_t_min=c['policy_t_min'],policy_eval_t_max=c['policy_t_max'],
        rollout_chunk_k=c['rollout_chunk_k'],extract_tokens_from_reference=False,
        reference_trust={'enabled':True,'coefficient':c['reference_coefficient']},
        projection_constraint={'enabled':False},adaptive_omnifold={'enabled':False},
        validation_every_n_epochs=c['validation_every_epochs'],validation_compute_winrate=False,
        checkpoint_every_n_steps=c['checkpoint_every_steps'],log_every=1,diagnostic_plot_every=50)
    runtime['ttbar_general']=copy.deepcopy(c)
    native=Path(c['evenet_root'])/'RL/DGPO_neutrino'
    trainer=(native/'dgpo_trainer.py').read_text()
    for marker in ('def _accumulate_dgpo_backward','def _dgpo_needs_parameter_snapshot','leave_one_out_unscaled'):
        if marker not in trainer:raise ValueError('Native DGPO efficiency/advantage update missing: '+marker)
    contract=dict(config=c,source={n:sha(src/n) for n in names},
        native={n:sha(native/n) for n in ('dgpo_trainer.py','model_utils.py','dgpo_utils.py')},
        adapter={p.name:sha(p) for p in Path(__file__).parent.glob('*.py') if not p.name.startswith('test_')},
        classifier_runner_sha256=sha(Path(classifier.__file__)),
        classifier_core_sha256=sha(Path(classifier.__file__).with_name('core.py')),
        evenet_sources={str(p.relative_to(Path(c['evenet_root']))):sha(p)
            for directory in ('evenet','preprocessing') for p in sorted((Path(c['evenet_root'])/directory).rglob('*.py'))})
    out.mkdir(parents=True,exist_ok=True)
    path=out/'contract.json'
    if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Run inputs/code changed; use a new output, do not silently resume')
    if not resume and (out/'checkpoints/last.ckpt').exists():raise ValueError('Existing DGPO checkpoint: explicitly --resume it')
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        stack=saved.get('dgpo_omnifold_reward_stack',{})
        if stack.get('kind')!='ttbar_general_top_mixture' or stack.get('contract')!=contract:
            raise ValueError('Resume does not match the classifier/mixture/feature-model contract')
    classifier.write_json(path,contract)
    rpath=out/('runtime-resume.yaml' if resume else 'runtime.yaml')
    rpath.write_text(yaml.safe_dump(runtime,sort_keys=False))
    classifier.eve_imports(c)
    from RL.DGPO_neutrino.model_utils import load_training_config
    resolved=load_training_config(rpath)
    if resolved.network.to_dict()!=policy['network']:
        raise ValueError('Resolved diffusion architecture differs from the pinned source')
    return rpath


def barrier():
    import torch.distributed as dist
    if dist.is_initialized():dist.barrier()


def rank_world():
    import ray.train
    c=ray.train.get_context();return c.get_world_rank(),c.get_world_size()


def start_wandb(c,*,disable=False,resume_run_id=None):
    if disable or os.environ.get('WANDB_DISABLED','').lower() in ('1','true','yes'):return False
    import wandb
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino import dgpo_trainer as native
    w=copy.deepcopy(c['wandb']);w['name']=w.pop('run_name')
    if resume_run_id is not None:
        if resume_run_id!=w['id']:raise ValueError('Checkpoint W&B ID differs from the prepared run')
        w['resume']='must'
    else:w['resume']='never'
    wandb.init(**w,config=global_config.to_logger())
    wandb.define_metric('epoch');wandb.define_metric('global_step')
    for prefix in ('val/*','train_dist/*','train_dist_k1/*'):wandb.define_metric(prefix,step_metric='epoch')
    for prefix in ('train/*','reward/*'):wandb.define_metric(prefix,step_metric='global_step')
    native._wandb_reset_step_tracker()
    return True


def make_pool(c,path,model,feature_model,*,validation_only=False,candidates=1):
    rank,world=rank_world();src=Path(c['classifier_directory']);path=Path(path)
    manifest=json.loads((src/'cache_manifest.json').read_text())
    if validation_only:manifest['inputs']={'validation':manifest['inputs']['validation']}
    cfg=copy.deepcopy(manifest['experiment']);cfg.update(output=str(path),workers=world)
    cfg['cache'].update(batch_size=c['generation_batch_size'],train_candidates=1,evaluation_candidates=candidates,ddim_steps=c['ddim_steps'])
    if rank==0:
        from RL.DGPO_neutrino.model_utils import state_dict_sha256
        path.mkdir(parents=True,exist_ok=True)
        identity=dict(policy_sha256=state_dict_sha256(model),candidates=candidates,validation_only=validation_only)
        identity_path=path/'pool_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
            raise ValueError('Existing cache belongs to different policy weights; refusing mixed validation/refit')
        classifier.write_json(identity_path,identity)
        manifest['source_checkpoint']='DGPO policy SHA256 '+identity['policy_sha256']
        manifest['feature_checkpoint']=str(src/'policy.ckpt')
        classifier.write_json(path/'cache_manifest.json',manifest)
    barrier()
    old=model.training;model.eval()
    try:classifier.cache_worker(cfg,generation_model=model,feature_model=feature_model,positive_cache_directory=src)
    finally:model.train(old)
    barrier()
    if rank==0:
        entries=[]
        for r in range(world):entries+=json.loads((path/f'cache-rank-{r}.json').read_text())['parts']
        if len(set(entries))!=len(entries):raise ValueError('Duplicate cache shards across workers')
        classifier.write_json(path/'CACHE_COMPLETE.json',{'parts':sorted(entries)})
    barrier()
    return cfg


class Session:
    def __init__(self,c,device,resume):
        self.c=c;self.device=device;self.rank,self.world=rank_world();self.step=0;self.refit_step=0;self.round_id=0
        src=Path(c['classifier_directory'])
        self.contract=json.loads((Path(c['output'])/'contract.json').read_text())
        payload=torch.load(src/'classifier-best.pt',map_location='cpu',weights_only=False)
        self.reward=GeneralOracleReward(load_feature_model(src,device),payload,alpha=c['mixture']['alpha'],
            temperature=c['mixture']['temperature'],feature_batch_size=c['feature_batch_size'])
        self.reference_sha=None
        if resume:
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            state=saved['dgpo_omnifold_reward_stack']
            if state['contract']!=self.contract:raise ValueError('Resume contract mismatch')
            self.reward.install(state['head']);self.refit_step=state['refit_step'];self.round_id=state['round_id']
            self.reference_sha=state['reference_sha'];self.step=int(saved['global_step'])
        self.agg=None

    def stack(self):
        return dict(kind='ttbar_general_top_mixture',contract=self.contract,head={k:v.cpu() for k,v in self.reward.head.state_dict().items()},
            stats=self.reward.stats,head_config=self.reward.head_config,alpha=self.reward.alpha,
            refit_step=self.refit_step,round_id=self.round_id,reference_sha=self.reference_sha)

    def maybe_refit(self,model,round_ref,step):
        if step-self.refit_step<self.c['refit']['every_steps']:return
        from RL.DGPO_neutrino.model_utils import state_dict_sha256
        path=Path(self.c['output'])/'refits'/f'step-{step:06d}'
        anchor=state_dict_sha256(model)
        if self.rank==0:
            path.mkdir(parents=True,exist_ok=True)
            identity={'policy_sha256':anchor,'source_contract':self.contract}
            record=path/'source.json'
            if record.exists() and json.loads(record.read_text())!=identity:
                raise ValueError('Incomplete refit belongs to another policy; preserve it and use its boundary checkpoint')
            classifier.write_json(record,identity)
        barrier()
        # Refit RNG must not advance the policy's optimizer/noise stream.
        with torch.random.fork_rng(devices=[self.device] if self.device.type=="cuda" else []):
            cfg=make_pool(self.c,path,model,self.reward.feature_model,candidates=1)
            cfg['fit']={k:v for k,v in self.c['refit'].items() if k not in ('every_steps','require_plateau')}
            if self.rank==0:classifier.atomic_torch(path/'normalization-classifier.pt',self.reward.stats)
            barrier()
            if not (path/'FIT_COMPLETE.json').exists():
                cfg['_resume']=(path/'classifier-last.pt').exists()
                def log(row):
                    import wandb
                    if wandb.run is not None:wandb.log({f'refit/{k}':v for k,v in row.items()}|{'refit/policy_step':step},step=step,commit=False)
                classifier.fit_worker(cfg,initial_state=self.reward.head.state_dict(),log_callback=log)
            status=json.loads((path/'FIT_COMPLETE.json').read_text())
            if not status['plateau']:raise RuntimeError('General refit hit cap without validation plateau; no new reward installed')
            payload=torch.load(path/'classifier-best.pt',map_location='cpu',weights_only=False)
            self.reward.install(payload['state_dict'])
        round_ref.load_state_dict(model.state_dict(),strict=True);round_ref.eval().requires_grad_(False)
        self.reference_sha=state_dict_sha256(round_ref);self.refit_step=step;self.round_id+=1
        self.agg._reference_sha=self.reference_sha
        barrier()
        if self.rank==0:print(f'Installed General round {self.round_id} at step {step}; alpha=0.3; validation BCE={status["best_validation_bce"]}',flush=True)


def train_worker(cfg):
    """Ray imports this function by module name; install hooks inside EVERY worker."""
    c=yaml.safe_load(Path(cfg['config_path']).read_text())['ttbar_general']
    classifier.eve_imports(c) # run.py imports can otherwise prepend ml_pipeline
    from RL.DGPO_neutrino import dgpo_trainer as native
    from evenet.control.global_config import global_config
    rank,_=rank_world()
    random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    torch.cuda.manual_seed_all(c['seed']+rank)
    state={}
    old_build=native.build_reward_aggregator;old_step=native.train_step;old_val=native.run_validation_epoch
    old_save=native.save_lightning_compatible_checkpoint
    old_generate=native.generate_neutrino_candidates
    old_wandb=native._start_wandb_run
    step_sig=inspect.signature(old_step);val_sig=inspect.signature(old_val)
    def build(model,device,normalization_dict=None):
        from RL.DGPO_neutrino.rewards import RewardAggregator
        class Aggregate(RewardAggregator):
            _reference_sha=None
            @property
            def reward_reference_sha256(self):return self._reference_sha
        s=Session(c,device,global_config.options.Training.get('model_checkpoint_load_path'))
        agg=Aggregate();agg.add(s.reward,1.);agg._reference_sha=s.reference_sha;s.agg=agg;state['session']=s
        return agg
    def save(*args,**kwargs):
        s=state['session'];kwargs['dgpo_omnifold_reward_stack']=s.stack();kwargs['reward_round_id']=s.round_id
        result=old_save(*args,**kwargs)
        path=Path(args[0] if args else kwargs['path']);step=int(kwargs['global_step'])
        if step%50==0 or (Path(c['output'])/'validation'/f'step-{step:06d}'/'metrics.json').exists():
            # Atomic native last.ckpt is a new inode each save; preserve named snapshots.
            named=path.parent/f'step-{step:06d}.ckpt'
            if not named.exists():os.link(path,named)
        return result
    def generate(*args,**kwargs):
        candidates=old_generate(*args,**kwargs)
        if not torch.isfinite(candidates).all():raise FloatingPointError('Nonfinite rollout; refusing native zero replacement')
        return candidates
    def step(*args,**kwargs):
        a=step_sig.bind(*args,**kwargs).arguments;s=state['session'];s.step=int(a['global_step'])
        core=native._unwrap_core_evenet(a['model'])
        s.maybe_refit(core,a['round_ref_model'],s.step)
        metrics=old_step(*args,**kwargs);metrics.update(s.reward.metrics());metrics['reward/refit_round']=s.round_id
        s.step+=1
        return metrics
    def validation(*args,**kwargs):
        a=val_sig.bind(*args,**kwargs).arguments;s=state['session']
        # Full validation from stable source file/row IDs (native synchronized loader
        # stops at the shortest shard). One saved K8 pass; no duplicate generation.
        core=native._unwrap_core_evenet(a['model'])
        path=Path(c['output'])/'validation'/f'step-{s.step:06d}'
        with torch.random.fork_rng(devices=[s.device] if s.device.type=="cuda" else []):
            pool_cfg=make_pool(c,path,core,s.reward.feature_model,validation_only=True,candidates=8)
        if s.rank==0:
            payload=dict(state_dict={k:v.cpu() for k,v in s.reward.head.state_dict().items()},stats=s.reward.stats,config=pool_cfg)
            classifier.atomic_torch(path/'classifier-best.pt',payload)
            classifier.atomic_torch(path/'normalization-classifier.pt',s.reward.stats)
            fit_source=Path(c['classifier_directory']) if s.round_id==0 else Path(c['output'])/'refits'/f'step-{s.refit_step:06d}'
            fit_status=json.loads((fit_source/'FIT_COMPLETE.json').read_text())
            classifier.write_json(path/'FIT_COMPLETE.json',{**fit_status,'installed_classifier_source':str(fit_source)})
            (path/'evaluation-config.yaml').write_text(yaml.safe_dump(pool_cfg,sort_keys=False))
            classifier.write_json(path/'validation_source.json',dict(step=s.step,round=s.round_id,contract=s.contract))
            from experiments.ttbar_dgpo_general.validation import summarize
            result=summarize(path,s.reward)
            classifier.write_json(path/'metrics.json',result)
        barrier()
        result=json.loads((path/'metrics.json').read_text())
        if s.rank==0:
            import wandb
            if wandb.run is not None:result['val/distributions']=wandb.Image(str(path/'distributions.png'))
        return result
    native.build_reward_aggregator=build;native.train_step=step;native.run_validation_epoch=validation
    native.save_lightning_compatible_checkpoint=save
    native.generate_neutrino_candidates=generate
    native._start_wandb_run=lambda **kw:start_wandb(c,**kw)
    try:native.dgpo_train_loop(cfg)
    finally:
        native.build_reward_aggregator=old_build;native.train_step=old_step;native.run_validation_epoch=old_val
        native.save_lightning_compatible_checkpoint=old_save
        native.generate_neutrino_candidates=old_generate
        native._start_wandb_run=old_wandb
