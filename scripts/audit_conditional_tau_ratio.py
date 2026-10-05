"""Tail attribution and fresh held-out weighted two-sample audits.

Only the source ratio's external test population enters the fresh audit.
All weights stay fixed. Nothing trains or modifies the diffusion policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from scripts.diagnose_conditional_tau_ratio_tail import analyze, normalized_weights, health
from scripts.train_conditional_spin_ratio import ConditionalSpinMLP, score_pair

ARMS = ('unweighted_joint', 'weighted_joint', 'weighted_condition')


def audit_split(ids, seed):
    """Independent identity split; input order and source scores cannot affect it."""
    if len(set(map(str, ids))) != len(ids):
        raise ValueError('Duplicate audit event identity')
    u = np.array([int.from_bytes(hashlib.blake2b(
        f'tau-weighted-audit:{seed}:{x}'.encode(), digest_size=8).digest(), 'big') / 2**64
        for x in ids])
    return np.where(u < .6, 0, np.where(u < .8, 1, 2)).astype('uint8')


def class_weights(base, logits, weighted):
    # Separate full-split normalization gives balanced class priors. Training
    # uses these fixed weights, never a random minibatch denominator.
    pos, _ = normalized_weights(base, np.zeros_like(base))
    neg, _ = normalized_weights(base, logits if weighted else np.zeros_like(logits))
    return pos * len(pos), neg * len(neg)


def weighted_bce(p, q, wp, wq):
    return .5 * (wp * F.softplus(-p) + wq * F.softplus(q)).mean()


def metrics(p, q, base, logits, weighted):
    from sklearn.metrics import roc_auc_score
    wp, wq = class_weights(base, logits, weighted)
    bce = .5 * (np.mean(wp*np.logaddexp(0., -p)) + np.mean(wq*np.logaddexp(0., q)))
    auc = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p,q],
                       sample_weight=np.r_[wp,wq])
    return dict(bce=float(bce), auc=float(auc), auc_gap=float(abs(auc-.5)))


class ConditionOnly(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden,64),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(64,1))

    def forward(self, condition, candidate):
        return self.net(condition).flatten()


def new_model(cfg, arrays, arm):
    torch.manual_seed(cfg['seed'])
    if arm == 'weighted_condition':
        return ConditionOnly(arrays['condition'].shape[1], cfg['hidden'], cfg['dropout'])
    # No source classifier parameters loaded. Same random initialization for
    # both joint arms; the source's frozen feature extractor is shared.
    return ConditionalSpinMLP(arrays['condition'].shape[1], cfg['hidden'], cfg['dropout'],
        arrays['candidate_truth'].shape[1], cfg['relative_dim'])


def attribution(truth, generated, base, logits, ids, categories):
    """Exact signed error-vector accounting plus exact leave-one-pair-out effects."""
    b, _ = normalized_weights(base, np.zeros_like(base))
    w, _ = normalized_weights(base, logits)
    target = np.sum(b[:,None]*truth,axis=0)
    estimate = np.sum(w[:,None]*generated,axis=0)
    terms = w[:,None]*generated - b[:,None]*truth
    error = estimate-target
    norm = float(np.linalg.norm(error))
    projection = np.sum(terms*(error / norm),axis=1) if norm > 0 else np.zeros(len(base))
    raw_error = np.sum(b[:,None]*generated,axis=0) - target
    raw_norm = float(np.linalg.norm(raw_error))
    # (||e1||-||e0||) = (e1-e0).(e1+e0)/(||e1||+||e0||).
    # Decompose exactly the *change* in error, not only the final error.
    direction = (error+raw_error)/(norm+raw_norm) if norm+raw_norm>0 else np.zeros_like(error)
    change_terms = np.sum((w-b)[:,None]*generated*direction,axis=1)
    if np.max(b) >= 1 or np.max(w) >= 1:
        raise ValueError('Single-event support: leave-one-out undefined')
    leave = (estimate-w[:,None]*generated)/(1-w[:,None]) - (target-b[:,None]*truth)/(1-b[:,None])
    loo = np.linalg.norm(leave, axis=1)-norm
    order = np.argsort(-projection, kind='stable')
    rows = [dict(source_id=str(ids[i]), category=int(categories[i]), mass=float(w[i]),
        error_direction_contribution=float(projection[i]),
        error_change_contribution=float(change_terms[i]),
        leave_one_pair_out_error_change=float(loo[i])) for i in order[:20]]
    damage_rows = [dict(source_id=str(ids[i]), category=int(categories[i]), mass=float(w[i]),
        error_direction_contribution=float(projection[i]), error_change_contribution=float(change_terms[i]),
        leave_one_pair_out_error_change=float(loo[i]))
        for i in np.argsort(-change_terms,kind='stable')[:20]]
    groups = []
    for cat in np.unique(categories):
        take = categories == cat
        groups.append(dict(category=int(cat), events=int(take.sum()), mass=float(w[take].sum()),
            error_direction_contribution=float(projection[take].sum()),
            error_change_contribution=float(change_terms[take].sum())))
    return dict(error_norm=norm, error_vector=error.tolist(),
        reconstruction_error=float(np.max(np.abs(terms.sum(0)-error))),
        projection_sum=float(projection.sum()), top_error_events=rows, top_damage_events=damage_rows,
        error_change=norm-raw_norm, change_contribution_sum=float(change_terms.sum()), categories=groups,
        positive_projection_mass=float(projection[projection>0].sum()),
        negative_projection_mass=float(projection[projection<0].sum()),
        interpretation='Signed contributions add to error norm; descriptive, not causal. '
        'Leave-one-pair-out renormalizes both populations and changes the target.')


def prepare(source, cfg):
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('representation') != 'tau' or manifest.get('candidate_count') != 1:
        raise ValueError('Requires completed tau K=1 ratio source')
    if manifest.get('relative_dim') != 6 or not manifest.get('backbone_cache'):
        raise ValueError('Requires the frozen-trunk relative-input source')
    with np.load(source/'prepared.npz', allow_pickle=False) as f:
        arrays = {k:f[k] for k in f.files}
    with np.load(source/'test_scores.npz', allow_pickle=False) as f:
        scores = {k:f[k] for k in f.files}
    tail = analyze(arrays, scores)  # checks ID alignment and raw-logit identity
    test = arrays['split'] == 2
    ids = arrays['source_ids'][test]
    if set(ids).intersection(arrays['source_ids'][~test]):
        raise ValueError('Source fit/validation overlap with fresh audit population')
    data = {key:arrays[key][test] for key in ('condition','candidate_truth',
        'candidate_generated','event_weight','source_ids')}
    data['log_ratio'] = scores['log_ratio'].astype('float64')
    data['split'] = audit_split(ids, cfg['seed'])
    for key in ('condition','candidate_truth','candidate_generated'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Nonfinite {key}')
    powers = np.prod(arrays['kappas'][test], axis=1)[:,None]
    t = np.einsum('ni,nj->nij', arrays['truth_a'][test], arrays['truth_b'][test]).reshape(-1,9)
    g = np.einsum('ni,nj->nij', arrays['sample_a'][test], arrays['sample_b'][test]).reshape(-1,9)
    families = {'tau':(arrays['tau_truth'][test],arrays['tau_generated'][test]),
                'angular':(t,g), 'cij':(9*t/powers,9*g/powers)}
    tail['error_attribution'] = {k:attribution(a,b,data['event_weight'],data['log_ratio'],
        ids,arrays['category'][test]) for k,(a,b) in families.items()}
    split_report = {}
    for value,name in enumerate(('fit','validation','test')):
        take = data['split']==value
        if take.sum() < max(20,cfg['workers']):
            raise ValueError(f'Too few audit {name} events')
        split_report[name] = health(data['event_weight'][take],data['log_ratio'][take])
    return data, tail, dict(relative_dim=6, source_manifest=manifest, split_health=split_report,
        audit_population='Source classifier external test only, split by event identity 60/20/20',
        normalization='Source fit-only transforms unchanged; audit class weights normalized per split',
        source_classifier_frozen=True, generator_updates=0,
        limitation='Fresh heads share frozen source representation, not an independent architecture. '
        'Tail-selected subsets change the target and are diagnostic only. '
        'Near-chance audit alone cannot certify closure, especially with low weighted ESS.')


def worker(cfg):
    import ray.train
    import ray.train.torch
    import torch.distributed as dist
    from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    device = ray.train.torch.get_device()
    with np.load(cfg['prepared'], allow_pickle=False) as f:
        a = {k:f[k] for k in f.files}
    arm = cfg['arm']; weighted = arm != 'unweighted_joint'
    fit, val = a['split']==0, a['split']==1
    wp,wq = class_weights(a['event_weight'][fit],a['log_ratio'][fit],weighted)
    ds = TensorDataset(*[torch.from_numpy(x.astype('float32')) for x in
        (a['condition'][fit],a['candidate_truth'][fit],a['candidate_generated'][fit],wp,wq)])
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, seed=cfg['seed'], shuffle=True)
    loader = DataLoader(ds, batch_size=cfg['batch_size'], sampler=sampler, pin_memory=True)
    model = ray.train.torch.prepare_model(new_model(cfg,a,arm).to(device))
    opt = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg['epochs'],eta_min=cfg['min_lr'])
    best, patience_best, stale, steps = float('inf'),float('inf'),0,0
    for epoch in range(cfg['epochs']):
        model.train(); sampler.set_epoch(epoch)
        running = torch.zeros(2,device=device)
        for batch in loader:
            c,t,g,pw,qw = [x.to(device,non_blocking=True) for x in batch]
            opt.zero_grad(set_to_none=True)
            loss = weighted_bce(model(c,t),model(c,g),pw,qw)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite weighted audit loss')
            loss.backward(); opt.step(); steps += 1
            running += torch.stack((loss.detach()*len(c),loss.new_tensor(len(c))))
        dist.all_reduce(running)
        scheduler.step()
        status = torch.zeros(5,device=device)
        if rank == 0:
            bare = model.module if hasattr(model,'module') else model
            p,q = score_pair(bare,a['condition'][val],a['candidate_truth'][val],
                a['candidate_generated'][val],device)
            m = metrics(p,q,a['event_weight'][val],a['log_ratio'][val],weighted)
            if m['bce'] < best:
                best = m['bce']
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in bare.state_dict().items()},
                    epoch=epoch+1,steps=steps,val_bce=best), cfg['checkpoint'])
            if m['bce'] < patience_best-cfg['min_delta']:
                patience_best=m['bce']; stale=0
            elif steps >= cfg['min_steps']: stale+=1
            status[:] = torch.tensor([m['bce'],m['auc'],best,
                float(steps>=cfg['min_steps'] and stale>=cfg['patience']),stale],device=device)
        dist.broadcast(status,src=0)
        ray.train.report(dict(epoch=epoch+1,optimizer_steps=steps,
            train_bce=float(running[0]/running[1]),val_bce=float(status[0]),val_auc=float(status[1]),
            best_val_bce=float(status[2]),stale_epochs=float(status[4]),
            lr=float(opt.param_groups[0]['lr'])))
        if bool(status[3]): break
    # Endpoint inference is sharded across every GPU, with no padded test IDs.
    dist.barrier()
    saved=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    bare=model.module if hasattr(model,'module') else model
    bare.load_state_dict(saved['state_dict'])
    test=np.flatnonzero(a['split']==2)
    positions=np.arange(rank,len(test),world)
    idx=test[positions]
    p,q=score_pair(bare,a['condition'][idx],a['candidate_truth'][idx],
        a['candidate_generated'][idx],device,batch_size=cfg['batch_size'])
    directory=Path(cfg['checkpoint']).parent
    np.savez(directory/f'test-rank-{rank:02d}.npz',positions=positions,
        source_ids=a['source_ids'][idx],truth_logits=p,generated_logits=q)
    if rank==0:
        (directory/'fit_status.json').write_text(json.dumps(dict(epochs=epoch+1,steps=steps,
            stopped_early=bool(status[3]),selected_steps=saved['steps']))+'\n')
    dist.barrier()


def merge_scores(directory, ids, workers):
    p,q=np.empty(len(ids)),np.empty(len(ids))
    seen=np.zeros(len(ids),dtype=int)
    for rank in range(workers):
        with np.load(directory/f'test-rank-{rank:02d}.npz',allow_pickle=False) as f:
            idx=f['positions']
            if np.any(idx<0) or np.any(idx>=len(ids)) or not np.array_equal(f['source_ids'],ids[idx]):
                raise ValueError('Audit score shard identities differ')
            np.add.at(seen,idx,1)
            p[idx],q[idx]=f['truth_logits'],f['generated_logits']
    if not np.all(seen==1) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Missing, duplicate or nonfinite audit scores')
    return p,q


def endpoint(cfg, a, arm):
    saved = torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    take = a['split']==2
    directory=Path(cfg['checkpoint']).parent
    p,q = merge_scores(directory,a['source_ids'][take],cfg['workers'])
    base,logits = a['event_weight'][take],a['log_ratio'][take]
    weighted = arm!='unweighted_joint'
    result = metrics(p,q,base,logits,weighted)
    rng = np.random.default_rng(cfg['seed']+912)
    draws = []
    for _ in range(cfg['bootstrap']):
        idx = rng.integers(0,len(p),len(p))
        draws.append(metrics(p[idx],q[idx],base[idx],logits[idx],weighted))
    for key in ('auc','bce','auc_gap'):
        result[key+'_ci95'] = np.quantile([r[key] for r in draws],[.025,.975]).tolist()
    result.update(best_epoch=saved['epoch'],best_steps=saved['steps'],
        fit_status=json.loads((directory/'fit_status.json').read_text()),
        minimum_fit_budget_met=saved['steps']>=cfg['min_steps'],
        weight_health=health(base,logits if weighted else np.zeros_like(logits)),
        uncertainty='Paired event bootstrap, fixed classifiers and samples; low ESS limits reliability')
    np.savez_compressed(Path(cfg['checkpoint']).parent/'audit_test_scores.npz',
        source_ids=a['source_ids'][take],truth_logits=p,generated_logits=q,
        source_log_ratio=logits,event_weight=base)
    return result


def compare_audits(output, bootstrap, seed):
    """Paired-event uncertainty of the fresh weighted-minus-raw audit gap."""
    loaded=[]
    for arm in ('unweighted_joint','weighted_joint'):
        with np.load(output/arm/'audit_test_scores.npz',allow_pickle=False) as f:
            loaded.append({k:f[k] for k in f.files})
    a,b=loaded
    for key in ('source_ids','source_log_ratio','event_weight'):
        if not np.array_equal(a[key],b[key]):
            raise ValueError('Fresh audit comparison populations differ')
    def difference(idx):
        values=[metrics(x['truth_logits'][idx],x['generated_logits'][idx],
            x['event_weight'][idx],x['source_log_ratio'][idx],weighted)
            for x,weighted in ((a,False),(b,True))]
        return {k:values[1][k]-values[0][k] for k in ('auc_gap','bce')}
    n=len(a['source_ids']);point=difference(np.arange(n))
    rng=np.random.default_rng(seed+912)
    draws=[difference(rng.integers(0,n,n)) for _ in range(bootstrap)]
    return dict(weighted_minus_unweighted=point,
        ci95={k:np.quantile([d[k] for d in draws],[.025,.975]).tolist() for k in point},
        interpretation='Negative AUC-gap change supports a smaller distinguishable residual; '
        'BCE toward log(2) is corroborating. Same events, separately trained heads and class measures. '
        'Fixed-model paired bootstrap excludes training uncertainty; not proof of full closure.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS','auto'))
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text())
    if not cfg['arms'] or any(x not in ARMS for x in cfg['arms']) or len(set(cfg['arms']))!=len(cfg['arms']):
        raise ValueError('Invalid audit arms')
    for key in ('workers','batch_size','epochs','min_steps','patience','bootstrap'):
        if cfg[key] <= 0: raise ValueError(f'{key} must be positive')
    a,tail,provenance=prepare(Path(cfg['source_directory']),cfg)
    cfg.update(provenance)
    if int(np.ceil(np.ceil(np.sum(a['split']==0)/cfg['workers'])/cfg['batch_size']))*cfg['epochs'] < cfg['min_steps']:
        raise ValueError('Configured training budget cannot reach minimum audit steps')
    print(json.dumps(provenance['split_health'],indent=2),flush=True)
    if args.prepare_only: return
    output=Path(cfg['output'])/('audit-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    cfg['prepared']=str(output/'prepared.npz')
    np.savez(output/'prepared.npz',**a)
    (output/'manifest.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (output/'tail_report.json').write_text(json.dumps(tail,indent=2,allow_nan=False)+'\n')
    import ray
    import wandb
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from ray.tune import Callback
    ray.init(address=args.ray_address,runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(
        (str(ROOT),str(ROOT/'evenet_dgpo'),os.environ.get('PYTHONPATH','')))}})
    if ray.cluster_resources().get('GPU',0)<cfg['workers']:
        raise ValueError('Insufficient GPUs for configured audit')
    class Progress(Callback):
        def on_trial_result(self,iteration,trials,trial,result,**info):
            keys=('epoch','optimizer_steps','train_bce','val_bce','val_auc','best_val_bce','lr','stale_epochs')
            run.log({f'{arm}/{k}':result[k] for k in keys if k in result})
    wb=cfg['wandb']
    with wandb.init(entity=wb['entity'],project=wb['project'],name=wb['name'],group=wb['group'],
        config=cfg,dir=str(output),tags=['fresh-weighted-audit','no-policy-update','16-gpu']) as run:
        (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
        for name in cfg['arms']:
            run.define_metric(f'{name}/epoch')
            run.define_metric(f'{name}/*',step_metric=f'{name}/epoch')
        for split,item in cfg['split_health'].items():
            for key in ('events','ess','ess_fraction','max_mass','top1pct_mass','log_mean_ratio'):
                run.summary[f'audit_population/{split}/{key}']=item[key]
        run.summary['phase']='tail_analysis'
        for f in ('manifest.json','tail_report.json'):
            run.save(str(output/f),base_path=str(output),policy='now')
        rows=[]
        for entry in tail['removal_arms']:
            rows.append([entry['removed'],entry['removed_original_mass'],entry['health']['ess'],
                *[entry['moments'][f]['error_change'] for f in ('tau','angular','cij')]])
            for family in ('tau','angular','cij'):
                run.summary[f'tail/drop_{entry["removed"]}/{family}_error_change']=entry['moments'][family]['error_change']
        run.log({'tail/removal':wandb.Table(columns=['removed','removed_mass','ess',
            'tau_error_change','angular_error_change','cij_error_change'],data=rows)})
        for name,item in tail['error_attribution'].items():
            columns=['source_id','category','mass','error_direction_contribution',
                'error_change_contribution','leave_one_pair_out_error_change']
            for kind in ('top_error_events','top_damage_events'):
                run.log({f'tail/{name}_{kind}':wandb.Table(columns=columns,
                    data=[[row[k] for k in columns] for row in item[kind]])})
        results={}
        for arm in cfg['arms']:
            directory=output/arm;directory.mkdir()
            armcfg=dict(cfg,arm=arm,checkpoint=str(directory/'best.pt'))
            run.summary['phase']=arm
            TorchTrainer(train_loop_per_worker=worker,train_loop_config=armcfg,
                scaling_config=ScalingConfig(num_workers=cfg['workers'],use_gpu=True),
                run_config=RunConfig(name=arm,storage_path=str(output/'ray_results'),
                    callbacks=[Progress()],failure_config=FailureConfig(max_failures=0))).fit()
            results[arm]=endpoint(armcfg,a,arm)
            for k,v in results[arm].items():
                run.summary[f'{arm}/test/{k}']=v
            (output/'audit_results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
            run.save(str(output/'audit_results.json'),base_path=str(output),policy='now')
            run.save(str(directory/'audit_test_scores.npz'),base_path=str(output),policy='now')
        if all(arm in cfg['arms'] for arm in ('unweighted_joint','weighted_joint')):
            comparison=compare_audits(output,cfg['bootstrap'],cfg['seed'])
            (output/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
            run.save(str(output/'comparison.json'),base_path=str(output),policy='now')
            for key,value in comparison['weighted_minus_unweighted'].items():
                run.summary[f'comparison/{key}_change']=value
                run.summary[f'comparison/{key}_change_ci95']=comparison['ci95'][key]
        run.summary['phase']='complete'
        print('WANDB:',run.url,flush=True)
    print('OUTPUT:',output,flush=True)


if __name__=='__main__': main()
