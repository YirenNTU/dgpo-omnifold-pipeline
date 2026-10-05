"""Matched FiLM paired BCE: fresh K1 per training epoch, fixed K1/K64 endpoints.

User launches on an existing 16-GPU Ray cluster. No generator/backbone updates.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml

from scripts.run_tau_conditioning import prepare as prepare_upstream, make_config as fusion_config
from scripts.run_tau_ratio_objectives import run_arm, finish_reports


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['workers'] != 16 or cfg['batch_size'] != 1024 or cfg['feature_batch_size'] < 1:
        raise ValueError('Requires 16 GPUs, batch1024 and positive feature microbatch')
    if cfg['bootstrap'] < 20 or cfg['fresh_seed'] < 0:
        raise ValueError('Invalid bootstrap or sampling seed')
    name = cfg['logger']['name']
    if len(name) > 96 or len(name.split(' | ')) < 3:
        raise ValueError('Invalid W&B display name')
    return cfg


def verify_matched_film(old, expected):
    keys = ('seed','workers','batch_size','epochs','lr','min_lr','hidden','dropout','weight_decay',
            'patience','min_delta','min_steps','head_kind','head_depth','relative_dim',
            'relative_preprocessing','packing_spec','condition_normalization','backbone_manifest',
            'train_source','test_source','train_events','test_events','split_counts',
            'condition_pt_edges','ratio_objective','mmd_coefficient')
    for key in keys:
        if old.get(key) != expected.get(key):
            raise ValueError('Historical FiLM differs from matched protocol: '+key)
    if old.get('fresh_negatives'):
        raise ValueError('Control must use fixed negatives')


def prepare(settings):
    upstream = yaml.safe_load((ROOT/settings['upstream_config']).read_text())
    upstream_source, base, arrays, _ = prepare_upstream(upstream)
    source = Path(settings['baseline_directory']).resolve()
    if not (source/'COMPLETE').is_file():
        raise ValueError('Historical FiLM must have completed')
    if json.loads((source/'wandb.json').read_text())['id'] != settings['baseline_run']:
        raise ValueError('Wrong historical FiLM run')
    baseline = json.loads((source/'manifest.json').read_text())
    expected = fusion_config(upstream, base, upstream_source, source, 'film')
    verify_matched_film(baseline, expected)
    if (source/'prepared.npz').resolve() != (upstream_source/'prepared.npz').resolve():
        with np.load(source/'prepared.npz',allow_pickle=False) as f:
            if set(f.files) != set(arrays) or any(not np.array_equal(f[k],v) for k,v in arrays.items()):
                raise ValueError('FiLM prepared inputs differ from original data')
    with np.load(source/'test_scores.npz',allow_pickle=False) as f:
        scores = {k:f[k] for k in f.files}
    if not np.array_equal(scores['source_ids'], arrays['source_ids'][arrays['split']==2]):
        raise ValueError('Historical scores have different test identities')
    if not np.array_equal(scores['log_ratio'], scores['generated_logits']):
        raise ValueError('Expected raw historical logits')
    panel = Path(settings['panel_directory'])
    if not (panel/'COMPLETE').is_file() or json.loads((panel/'wandb.json').read_text())['id'] != settings['panel_run']:
        raise ValueError('Fixed K64 panel is missing or differs from pinned run')
    manifest = json.loads((panel/'manifest.json').read_text())
    backbone = baseline['backbone_manifest']
    if (manifest['classifier_run'] != settings['baseline_run'] or
        Path(manifest['generator_checkpoint']).resolve() != Path(backbone['checkpoint']).resolve() or
        manifest['weights'] != 'raw_state_dict_only' or manifest['ddim_steps'] != 20):
        raise ValueError('Fixed panel and fit use different generator/classifier protocols')
    train_sampling = baseline['train_manifest']
    if (Path(train_sampling['checkpoint']).resolve() != Path(backbone['checkpoint']).resolve() or
        train_sampling['weights'] != 'raw_state_dict_only' or train_sampling['ddim_steps'] != 20 or
        train_sampling['candidates'] != 1):
        raise ValueError('Historical training negatives must come from raw1110 DDIM20 K1')
    with np.load(panel/'inputs.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],scores['source_ids']):
            raise ValueError('K64 panel has different test events/order')
    with np.load(panel/'samples_and_scores.npz',allow_pickle=False) as f:
        if f['logits'].shape != (119002,64) or not np.array_equal(f['source_ids'],scores['source_ids']):
            raise ValueError('Expected the complete matched 119002-event K64 panel')
    settings['fresh_runtime'] = backbone['runtime']
    settings['fresh_generator_checkpoint'] = backbone['checkpoint']
    print('READY: verified historical FiLM control, fixed K64 panel, raw1110, 16 GPUs',flush=True)
    return source, baseline, arrays, scores


def prepare_fit_inputs(cfg, output):
    """Align filtered parquet to prepared fit rows; do not refit preprocessing."""
    from scripts.sample_1110_cij import validation_pool
    from scripts.diagnose_ztautau_cij import source_ids, align, read_event_table, SOURCE_ID_COLUMNS
    from scripts.conditional_tau_preprocessing import apply_masked_feature
    with np.load(cfg['prepared'],allow_pickle=False) as f:
        take = f['split']==0
        wanted, condition, category = f['source_ids'][take], f['condition'][take], f['category'][take]
        mean, scale, weights = f['condition_mean'], f['condition_scale'], f['event_weight'][take]
    pool = validation_pool(cfg['train_events'], {'packing_spec':cfg['packing_spec']})
    if pool['rows_read'] != 416701 or pool['invalid_target_rows'] != 0:
        raise ValueError('Training parquet population changed')
    idx = align(source_ids(pool),wanted)
    raw = pool['condition'][idx].numpy()
    keys = [k for k in SOURCE_ID_COLUMNS if k in pool]
    required = keys+['event_category','event_weight']+[f'lead_{leg}_visible_{c}' for leg in ('a','b') for c in ('E','px','py','pz')]
    table = read_event_table(cfg['train_events'],required)
    cols = {k:np.asarray(table[k].to_numpy()) for k in required}
    order = align(source_ids(cols,keys),wanted)
    cols = {k:v[order] for k,v in cols.items()}
    if not np.array_equal(category,cols['event_category']):
        raise ValueError('Training category mismatch')
    if not np.allclose(weights,cols['event_weight'],rtol=1e-6,atol=1e-7):
        raise ValueError('Training event weights changed')
    onehot = np.eye(16,dtype=np.float32)[(category.astype(int)//10-1)*4+category.astype(int)%10-1]
    rebuilt = apply_masked_feature(np.concatenate((raw,onehot),1),mean,scale,cfg['packing_spec'])
    if not np.allclose(rebuilt,condition,rtol=1e-6,atol=1e-6):
        raise ValueError('Condition preprocessing changed')
    with np.load(Path(cfg['train_source'])/'candidates.npz',allow_pickle=False) as f:
        old_idx = align(source_ids({k:f[k] for k in keys},keys),wanted)
        if not np.array_equal(f['truth_deltas'][old_idx],pool['truth'][idx].numpy()):
            raise ValueError('Training truth pairing changed')
        old = f['deltas'][old_idx,0]
    visible = {f'visible_{leg}':np.stack([cols[f'lead_{leg}_visible_{c}'] for c in ('E','px','py','pz')],1)
               for leg in ('a','b')}
    path = output/'fresh_fit_inputs.npz'
    np.savez(path,raw_condition=raw,source_ids=wanted,old_deltas=old,**visible)
    return str(path)


def make_config(settings, baseline, source, output, arm):
    cfg = copy.deepcopy(baseline)
    cfg.update(output=str(output),prepared=str(output/'prepared.npz'),checkpoint=str(output/'best.pt'),
        baseline_directory=str(source),baseline_run=settings['baseline_run'],
        fresh_negatives=True,fresh_seed=settings['fresh_seed'],fresh_runtime=settings['fresh_runtime'],
        fresh_generator_checkpoint=settings['fresh_generator_checkpoint'],
        fresh_generation_precision='highest',
        feature_batch_size=settings['feature_batch_size'],panel_directory=settings['panel_directory'],
        panel_run=settings['panel_run'],run_name=settings['logger']['name'],
        wandb_group=settings['logger']['group'],bootstrap=settings['bootstrap'],
        no_bce_refit=False,historical_bce_reused=True,policy_updates=0,backbone_updates=0,
        initialization='Fresh head seed42, same FiLM initializer; historical trained head is comparison only.',
        budget_comparison='Same updates/epochs/stopping, NOT equal wallclock: DDIM and feature extraction each batch.',
        negative_sampling='Independent unconditional-on-truth K1 draw for each fit condition each epoch; DDP padding retained.',
        fresh_precision_provenance='Historical sample_conditional_spin_train.worker used torch default highest, not the DGPO precision setting.',
        ratio_transform='raw exp(logit); paired equal class weights; cap30 ONLY a fixed evaluation sensitivity.',
        tags=['tau','fresh-negatives','FiLM','BCE','16-gpu','raw-1110','no-policy-update'])
    cfg['fresh_inputs'] = prepare_fit_inputs(cfg,output)
    return cfg


@torch.no_grad()
def panel_worker(cfg, rank):
    from scripts.tau_fresh_negatives import load_policy, make_batch, candidate_features, isolated_rng
    from scripts.train_conditional_spin_ratio import build_classifier
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    device = torch.device('cuda:0')
    torch.set_num_threads(1)
    panel = Path(cfg['panel_directory'])
    pos = np.arange(rank,119002,16)
    with np.load(panel/'inputs.npz',allow_pickle=False) as f:
        a = {k:f[k][pos] for k in ('raw_condition','condition','visible_a','visible_b','source_ids')}
    with np.load(panel/'samples_and_scores.npz',allow_pickle=False) as f:
        delta, old_logits = f['deltas'][pos], f['logits'][pos]
    saved = torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    with isolated_rng(device,cfg['seed']):
        policy = load_policy(cfg,device)
        head = build_classifier(saved).to(device).eval()
        head.load_state_dict(saved['state_dict'],strict=True)
        old = torch.load(Path(cfg['baseline_directory'])/'best.pt',map_location='cpu',weights_only=True)
        old_head = build_classifier(old).to(device).eval()
        old_head.load_state_dict(old['state_dict'],strict=True)
        spec = EventPackingSpec.from_dict(saved['packing_spec'])
        logits = np.empty(delta.shape[:2],dtype=np.float32)
        for start in range(0,len(pos),cfg['batch_size']):
            stop = min(start+cfg['batch_size'],len(pos))
            batch = make_batch(a['raw_condition'][start:stop],spec,device)
            c = torch.as_tensor(a['condition'][start:stop],device=device)
            for j in range(64):
                features = candidate_features(policy,batch,torch.as_tensor(delta[start:stop,j],device=device),
                    a['visible_a'][start:stop],a['visible_b'][start:stop],saved['relative_preprocessing'],cfg['feature_batch_size'])
                features = torch.as_tensor(features,device=device)
                if start == 0 and j == 0:
                    replay = old_head(c,features).cpu().numpy()
                    if not np.allclose(replay,old_logits[start:stop,j],atol=1e-3,rtol=1e-4):
                        raise ValueError('Fixed K64 historical score replay failed')
                if saved.get('explicit_input'):
                    from scripts.tau_explicit_inputs import transform_panel_inputs
                    explicit_c, explicit_f = transform_panel_inputs(
                        a['condition'][start:stop], features.cpu().numpy(),
                        a['visible_a'][start:stop], a['visible_b'][start:stop],
                        delta[start:stop,j], saved['explicit_input'])
                    logits[start:stop,j] = head(torch.as_tensor(explicit_c,device=device),
                        torch.as_tensor(explicit_f,device=device)).cpu().numpy()
                else:
                    logits[start:stop,j] = head(c,features).cpu().numpy()
            print(f'[K64 rescore rank={rank}] {stop}/{len(pos)} conditions',flush=True)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite K64 logits')
    np.savez(Path(cfg['output'])/f'panel-{rank:02d}.npz',positions=pos,source_ids=a['source_ids'],logits=logits)


def panel_report(inputs, data, fresh_logits, bootstrap, seed):
    from scripts.tau_tail_attribution import candidate_weights, paired_bootstrap
    old_logits = data['logits']
    if fresh_logits.shape != old_logits.shape or not np.isfinite(fresh_logits).all():
        raise ValueError('Fresh scores do not match fixed candidate panel')
    target = np.average(inputs['truth_cij'],weights=inputs['weight'],axis=0)
    arms = dict(unweighted=np.zeros_like(old_logits),old_raw=old_logits,
        old_cap30=np.minimum(old_logits,np.log(30)),fresh_raw=fresh_logits,
        fresh_cap30=np.minimum(fresh_logits,np.log(30)))
    nums, dens, result = [], [], {}
    for name, logits in arms.items():
        w, log_mean = candidate_weights(inputs['weight'],logits)
        num, den = np.einsum('nk,nkd->nd',w,data['cij']),w.sum(1)
        mean = num.sum(0)
        nums.append(num);dens.append(den)
        result[name] = dict(C=mean.reshape(3,3).tolist(),error=float(np.linalg.norm(mean-target)),
            candidate_ess=float(1/np.square(w).sum()),event_ess=float(1/np.square(den).sum()),
            max_candidate_mass=float(w.max()),log_mean_ratio=log_mean)
    draws, truths = paired_bootstrap(inputs,nums,dens,bootstrap,seed)
    residuals = draws-truths[:,None,:]
    errors = np.linalg.norm(residuals,axis=2)
    labels = list(arms); comparisons = {}
    for new, old in [('fresh_raw','old_raw'),('fresh_cap30','old_cap30'),
                     ('fresh_raw','unweighted'),('fresh_cap30','unweighted'),('fresh_raw','old_cap30')]:
        i,j = labels.index(new),labels.index(old)
        delta = errors[:,i]-errors[:,j]
        component_delta = np.abs(residuals[:,i])-np.abs(residuals[:,j])
        comparisons[new+'_minus_'+old] = dict(error_change=result[new]['error']-result[old]['error'],
            error_change_ci95=np.quantile(delta,[.025,.975]).tolist(),
            absolute_component_error_change_ci95=np.quantile(component_delta,[.025,.975],axis=0).tolist())
    return dict(arms=result,comparisons=comparisons,truth_C=target.reshape(3,3).tolist(),
        events=len(old_logits),candidates=old_logits.shape[1],bootstrap=bootstrap,
        convention='9*mean(a_i*b_j/(kappa_a*kappa_b)); same saved matched truth and candidates',
        uncertainty='Paired event bootstrap, pointwise intervals, fixed heads/candidates; previously inspected test; no fit uncertainty.',
        primary='Fresh raw vs old raw Cij error; also must inspect all nine entries and support. Cap30 is sensitivity only.')


def finish(cfg, arrays, p, q, scores, run, settings):
    import ray
    import wandb
    # Retain original K1 endpoints, but name the baseline accurately.
    finish_reports(cfg,arrays,p,q,scores,run,settings)
    run.summary['bce_baseline_is'] = 'pzq0nl1i matched fixed-negative FiLM (not legacy head)'
    run.summary['phase'] = 'fixed_K64_rescore'
    task = ray.remote(num_gpus=1,num_cpus=1,max_calls=1)(panel_worker)
    ray.get([task.remote(cfg,rank) for rank in range(16)])
    panel = Path(cfg['panel_directory'])
    with np.load(panel/'inputs.npz',allow_pickle=False) as f:
        inputs = {k:f[k] for k in ('source_ids','weight','truth_cij')}
    with np.load(panel/'samples_and_scores.npz',allow_pickle=False) as f:
        data = {k:f[k] for k in ('logits','cij')}
    logits = np.empty_like(data['logits']);seen = np.zeros(len(logits),int)
    for rank in range(16):
        with np.load(Path(cfg['output'])/f'panel-{rank:02d}.npz',allow_pickle=False) as f:
            idx = f['positions']
            if not np.array_equal(f['source_ids'],inputs['source_ids'][idx]):
                raise ValueError('K64 score shard identity mismatch')
            logits[idx] = f['logits'];seen[idx] += 1
    if not np.all(seen==1):
        raise ValueError('Missing or duplicate K64 scores')
    np.savez(Path(cfg['output'])/'fixed_panel_scores.npz',source_ids=inputs['source_ids'],logits=logits)
    report = panel_report(inputs,data,logits,cfg['bootstrap'],cfg['seed'])
    path = Path(cfg['output'])/'fixed_K64_report.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    run.save(str(path),base_path=cfg['output'],policy='now')
    for arm,r in report['arms'].items():
        for key,value in r.items():
            if key != 'C': run.summary[f'K64/{arm}/{key}'] = value
    for label,r in report['comparisons'].items():
        run.summary[f'K64/{label}'] = r['error_change']
        run.summary[f'K64/{label}_ci95'] = r['error_change_ci95']
    rows = [[name,i,j,r['C'][i][j],report['truth_C'][i][j]]
            for name,r in report['arms'].items() for i in range(3) for j in range(3)]
    run.log({'K64/Cij':wandb.Table(columns=['arm','i','j','Cij','truth_Cij'],data=rows)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('phase',choices=('prepare','train'),nargs='?',default='train')
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = p.parse_args()
    settings = read_settings(args.config)
    source,baseline,arrays,scores = prepare(settings)
    if args.phase == 'train':
        run_arm(settings,'fresh',source,baseline,arrays,scores,args.ray_address,
                config_builder=make_config,reporter=finish)


if __name__ == '__main__':
    main()
