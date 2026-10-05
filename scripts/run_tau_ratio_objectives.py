"""User-launched MLC/nnUKL ablation; reuse BCE, its data, and frozen features.

One YAML, phases prepare / mlc / nnukl / all. Never fit BCE or the backbone,
generate candidates, submit allocations, or update DGPO. Each fit is a fresh
W&B run and output directory. 'all' sequentially runs only the two new arms.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]

import numpy as np
import torch
import yaml

from scripts.train_conditional_spin_ratio import (
    build_classifier, training_worker, score_pair, pair_metrics, tau_moment_report,
)
from scripts.tau_ratio_objectives import validation_objective
from scripts.diagnose_conditional_tau_ratio_tail import health, normalized_weights


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['workers'] != 16 or cfg['batch_size'] != 1024:
        raise ValueError('This matched ablation requires 16 GPUs and 1024 conditions/GPU')
    if not np.isfinite(cfg['nnukl_c']) or not 0 < cfg['nnukl_c'] < 1:
        raise ValueError('nnukl_c must be finite and between zero and one')
    if cfg['bootstrap'] < 20:
        raise ValueError('Require at least 20 paired bootstrap draws')
    stop = cfg['early_stopping']
    if (stop['monitor'] != 'val_bce' or stop['patience'] < 1 or stop['min_steps'] < 0
        or not np.isfinite(stop['min_delta']) or stop['min_delta'] < 0):
        raise ValueError('Invalid validation-BCE early stopping configuration')
    for arm in ('mlc','nnukl'):
        name = cfg['logger'][arm+'_name']
        if not isinstance(name,str) or len(name)>96 or len(name.split(' | '))<3:
            raise ValueError('Invalid W&B display name')
    return cfg


def check_protocol(settings, baseline, expected):
    """No tuning or architecture drift hidden in the loss comparison."""
    for key in ('seed','workers','batch_size','epochs','patience','min_delta','min_steps',
                'lr','min_lr','weight_decay','dropout','hidden','condition_normalization',
                'backbone_cache','mmd_coefficient'):
        if baseline.get(key) != expected['classifier'][key]:
            raise ValueError(f'BCE baseline differs from pinned config: {key}')
    for key in ('workers','batch_size'):
        if baseline[key] != settings[key]:
            raise ValueError(f'Loss ablation differs from BCE: {key}')
    if (baseline.get('ratio_objective','bce') != 'bce' or baseline.get('mmd_coefficient') != 0
        or baseline.get('relative_dim') != 6 or baseline.get('representation') != 'tau'
        or baseline.get('candidate_count') != 1 or baseline.get('weight_mode') != 'joint'):
        raise ValueError('Requires completed relative-input paired BCE tau K=1 baseline')
    if baseline['patience'] <= baseline['epochs']:
        raise ValueError('Requires the matched fixed-epoch baseline budget')
    source_pairs = ((baseline['train_events'], expected['platform']['data_parquet_dir']),
        (baseline['test_events'], expected['platform']['data_parquet_val_dir']),
        (baseline['train_source'], expected['experiment']['train_sample_root']),
        (baseline['test_source'], expected['experiment']['test_source']),
        (settings['baseline_directory'], expected['experiment']['output']))
    if any(Path(a).resolve() != Path(b).resolve() for a,b in source_pairs):
        raise ValueError('BCE source paths differ from the pinned filtered experiment')
    if baseline['train_manifest']['weights'] != 'raw_state_dict_only':
        raise ValueError('EMA is not permitted')
    if baseline['backbone_manifest'].get('global_step') != 1110:
        raise ValueError('Requires raw step1110 cached backbone')


def load_baseline(settings):
    source = Path(settings['baseline_directory']).resolve()
    baseline = json.loads((source/'manifest.json').read_text())
    expected = yaml.safe_load((ROOT/settings['baseline_config']).read_text())
    check_protocol(settings, baseline, expected)
    if json.loads((source/'wandb.json').read_text())['id'] != settings['baseline_run']:
        raise ValueError('BCE W&B run ID differs from configured control')
    for path in ('best.pt','preprocessing.json','candidate_ratio_closure.json',
                 'candidate_ratio_tau_moments.json','test_scores.npz','prepared.npz'):
        if not (source/path).is_file():
            raise ValueError(f'BCE baseline is incomplete: {path}')
    for key in ('train_events','test_events'):
        manifest = Path(baseline[key]).parent/'filter_manifest.json'
        if json.loads(manifest.read_text()).get('complete') is not True:
            raise ValueError(f'Incomplete filtered dataset: {manifest}')
    # Reuse the exact prepared file: no normalization, split, extraction or sampling rerun.
    with np.load(source/'prepared.npz', allow_pickle=False) as f:
        arrays = {key:f[key] for key in f.files}
    with np.load(source/'test_scores.npz', allow_pickle=False) as f:
        scores = {key:f[key] for key in f.files}
    split, ids = arrays['split'], arrays['source_ids']
    if len(np.unique(ids)) != len(ids) or not np.isin(split, [0,1,2]).all():
        raise ValueError('Duplicate identities or invalid split')
    counts = {name:int((split==i).sum()) for i,name in enumerate(('train','validation','test'))}
    if counts != baseline['split_counts']:
        raise ValueError('Prepared split counts differ from BCE manifest')
    if not np.array_equal(ids[split==2], scores['source_ids']):
        raise ValueError('BCE test scores do not align with saved input identities')
    if not np.array_equal(scores['generated_logits'], scores['log_ratio']):
        raise ValueError('BCE comparison must use raw log ratios')
    for key in ('condition','candidate_truth','candidate_generated','event_weight'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'Nonfinite cached input: {key}')
    if (arrays['event_weight'] < 0).any():
        raise ValueError('Signed event weights are unsupported')
    return source, baseline, arrays, scores


def make_arm_config(settings, baseline, source, output, arm):
    if arm not in ('mlc','nnukl'):
        raise ValueError('Only MLC and nnUKL are new training arms; BCE is reused')
    cfg = copy.deepcopy(baseline)
    stop = settings['early_stopping']
    differences = {k:dict(baseline=baseline[k],current=stop[k])
        for k in ('patience','min_delta','min_steps') if baseline[k] != stop[k]}
    cfg.update(output=str(output), prepared=str(output/'prepared.npz'), checkpoint=str(output/'best.pt'),
        baseline_directory=str(source), baseline_run=settings['baseline_run'], baseline_training_differences=differences,
        patience=stop['patience'], min_delta=stop['min_delta'], min_steps=stop['min_steps'],
        early_stopping=stop,
        ratio_objective=arm, nnukl_c=settings['nnukl_c'] if arm=='nnukl' else 0.,
        run_name=settings['logger'][arm+'_name'], wandb_group=settings['logger']['group'],
        bootstrap=settings['bootstrap'], policy_updates=0, backbone_updates=0,
        selector='minimum internal validation BCE; identical to reused BCE control',
        budget_comparison='BCE ran full 250 epochs; new arms may stop early. Not equal-compute vs BCE.',
        risk_reduction='global DDP batch expectations BEFORE non-negative correction',
        model_precision='float32; exponential risk arithmetic float64',
        ratio_transform='exp(logit), raw; no cap, tempering, per-event normalization or ESS penalty',
        bound_assumption='Unverified true sup(p/q) < 1/c for nnUKL; c is exploratory, not a cap',
        evaluation_status='Exploratory: external test pool was inspected in earlier rounds',
        no_bce_refit=True, mmd_coefficient=0.,
        train_mmd_diagnostic='skipped; coefficient was zero in BCE; validation MMD retained',
        limitation='No fresh weighted classifier fit in this command; existing audit launcher can use this output. '
                   'Moment and Cij improvement alone do not establish full conditional closure.')
    return cfg


def inference_worker(cfg, rank):
    """One GPU per shard; no DDP padding and no candidate regeneration."""
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    saved = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    model = build_classifier(saved).to(device)
    model.load_state_dict(saved['state_dict'])
    with np.load(cfg['prepared'], allow_pickle=False) as a:
        test_idx = np.flatnonzero(a['split']==2)
        positions = np.arange(rank,len(test_idx),cfg['workers'])
        selected = test_idx[positions]
        p,q = score_pair(model,a['condition'][selected],a['candidate_truth'][selected],
                         a['candidate_generated'][selected],device,cfg['batch_size'])
        np.savez(Path(cfg['output'])/f'score-{rank:02d}.npz', positions=positions,
                 source_ids=a['source_ids'][selected], truth_logits=p, generated_logits=q)


def merge_scores(output, ids, workers):
    p, q = np.empty(len(ids)), np.empty(len(ids))
    seen = np.zeros(len(ids),dtype=int)
    for rank in range(workers):
        with np.load(Path(output)/f'score-{rank:02d}.npz', allow_pickle=False) as a:
            idx = a['positions']
            if (idx<0).any() or (idx>=len(ids)).any() or len(np.unique(idx))!=len(idx):
                raise ValueError('Invalid inference shard positions')
            if not np.array_equal(a['source_ids'],ids[idx]):
                raise ValueError('Inference shard identities differ')
            seen[idx] += 1
            p[idx],q[idx] = a['truth_logits'],a['generated_logits']
    if not (seen==1).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Missing, duplicate or nonfinite inference scores')
    return p,q


def paired_cij_comparison(arrays, score_arms, bootstrap, seed):
    """Same inverse-kappa Cij estimator as existing analysis; paired bootstrap.

All arms keep ALL external events; no tail removal or channel rebalancing.
This preserves the adopted analysis convention, not a new certification of it.
"""
    from scripts.diagnose_reweighted_cij import features
    mask = arrays['split']==2
    base = arrays['event_weight'][mask]
    powers = arrays['kappas'][mask]
    if not np.isfinite(powers).all() or (powers==0).any():
        raise ValueError('Invalid analyzing powers')
    truth = features(arrays['truth_a'][mask].astype(float),arrays['truth_b'][mask].astype(float),powers)
    gen = features(arrays['sample_a'][mask].astype(float),arrays['sample_b'][mask].astype(float),powers)
    labels = ['unweighted', *score_arms]
    weights = np.stack([normalized_weights(base,np.zeros(len(base)))[0]]+
                       [normalized_weights(base,score_arms[k])[0] for k in score_arms],axis=1)
    def compute(idx):
        w = weights[idx]
        if (w.sum(0)<=0).any():
            raise ValueError('Insufficient bootstrap weight support')
        target = np.sum(truth[idx]*w[:,0,None],axis=0)/w[:,0].sum()
        matrices = (w.T @ gen[idx])/w.sum(0)[:,None]
        return target,matrices,np.linalg.norm(matrices-target,axis=1)
    target,matrices,errors = compute(np.arange(len(base)))
    rng = np.random.default_rng(seed)
    draws = np.stack([compute(rng.integers(0,len(base),len(base)))[2] for _ in range(bootstrap)])
    result = dict(events=len(base), truth_C=target.reshape(3,3).tolist(),
        arms={label:dict(C=matrices[i].reshape(3,3).tolist(),error=float(errors[i]),
            error_ci95=np.quantile(draws[:,i],[.025,.975]).tolist()) for i,label in enumerate(labels)},
        comparisons={}, bootstrap=bootstrap,
        convention='9*mean(a_i*b_j/(kappa_a*kappa_b)); same matched truth as BCE; all events',
        uncertainty='Paired event bootstrap, fixed models/candidates; no refit uncertainty; '
                    'pointwise intervals; previously inspected test is exploratory')
    current = labels.index('candidate_ratio')
    for other in labels:
        if other=='candidate_ratio':
            continue
        i = labels.index(other)
        ci = np.quantile(draws[:,current]-draws[:,i],[.025,.975])
        result['comparisons']['candidate_minus_'+other] = dict(value=float(errors[current]-errors[i]),
            ci95=ci.tolist(), supports_lower_error=bool(ci[1]<0))
    return result


def finish_reports(cfg, arrays, p, q, baseline_scores, run, settings, extra_score_arms=None):
    import wandb
    output = Path(cfg['output'])
    test = arrays['split']==2
    score_arms = dict(bce=baseline_scores['log_ratio'],candidate_ratio=q)
    if extra_score_arms:
        if set(extra_score_arms) & set(score_arms):
            raise ValueError('Duplicate score-arm label')
        score_arms.update(extra_score_arms)
    # A completed matched MLC from this family is an additional control for nnUKL.
    pointer = Path(settings['output_root'])/'mlc_latest_completed.json'
    if cfg['ratio_objective']=='nnukl' and pointer.is_file():
        mlc_dir = Path(json.loads(pointer.read_text())['output'])
        if not (mlc_dir/'COMPLETE').is_file():
            raise ValueError('MLC control did not complete its endpoint analysis')
        old = json.loads((mlc_dir/'manifest.json').read_text())
        keys = ('baseline_directory','baseline_run','seed','workers','batch_size','epochs',
                'lr','min_lr','hidden','dropout','weight_decay','selector','patience','min_delta','min_steps')
        if old.get('ratio_objective')!='mlc' or any(old[k]!=cfg[k] for k in keys):
            raise ValueError('Completed MLC pointer is not a matched comparison')
        with np.load(mlc_dir/'test_scores.npz',allow_pickle=False) as f:
            if not np.array_equal(f['source_ids'],arrays['source_ids'][test]):
                raise ValueError('MLC control identities differ')
            score_arms['mlc'] = f['log_ratio']
        run.summary['mlc_control_directory'] = str(mlc_dir)
    np.savez_compressed(output/'test_scores.npz',source_ids=arrays['source_ids'][test],
        truth_logits=p,generated_logits=q,log_ratio=q,saved_h4_log_ratio=arrays['base_log_ratio'][test])
    run.save(str(output/'test_scores.npz'),base_path=str(output),policy='now')
    metrics = pair_metrics(p,q,arrays['event_weight'][test])
    run.summary.update({f'test/{k}':v for k,v in metrics.items()})
    # The uncorrected UKL/MLC risk is also a common diagnostic for BCE fits.
    risk_kind = 'mlc' if cfg['ratio_objective']=='bce' else cfg['ratio_objective']
    risk = validation_objective(p,q,arrays['event_weight'][test],risk_kind,cfg['nnukl_c'])
    run.summary.update({f'test_risk/{k}':v for k,v in risk.items()})
    rows = []
    for arm,logits in score_arms.items():
        h = health(arrays['event_weight'][test],logits)
        for key in ('ess','ess_fraction','max_mass','top1pct_mass','log_mean_ratio'):
            run.summary[f'{arm}/{key}'] = h[key]
        tau = tau_moment_report(arrays,logits,test)
        path = output/f'{arm}_tau_moments.json'
        path.write_text(json.dumps(tau,indent=2,allow_nan=False)+'\n')
        run.save(str(path),base_path=str(output),policy='now')
        for group in tau['groups']:
            rows.append([arm,group['group'],group['events'],group['l2_error_unweighted'],
                         group['l2_error_reweighted'],group['l2_error_change']])
            if group['group']=='all':
                run.summary[f'{arm}/tau_error_change'] = group['l2_error_change']
    run.log({'tau/closure_by_condition':wandb.Table(columns=['arm','group','events',
        'unweighted_error','reweighted_error','error_change'],data=rows)})
    report = paired_cij_comparison(arrays,score_arms,cfg['bootstrap'],cfg['seed'])
    path = output/'cij_comparison.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    run.save(str(path),base_path=str(output),policy='now')
    for arm,values in report['arms'].items():
        run.summary[f'cij/{arm}/error'] = values['error']
    for label,value in report['comparisons'].items():
        run.summary[f'cij/{label}'] = value['value']
        run.summary[f'cij/{label}_ci95'] = value['ci95']
    rows = [[arm,i,j,values['C'][i][j],report['truth_C'][i][j]]
        for arm,values in report['arms'].items() for i in range(3) for j in range(3)]
    run.log({'cij/matrices':wandb.Table(columns=['arm','i','j','Cij','truth_Cij'],data=rows)})
    run.summary['cij/convention'] = report['convention']
    run.summary['evaluation_status'] = cfg['evaluation_status']


def run_arm(settings, arm, source, baseline, arrays, scores, ray_address,
            config_builder=make_arm_config, reporter=finish_reports, materialize_inputs=False):
    import ray
    import wandb
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from ray.tune import Callback
    root = Path(settings['output_root']).resolve()
    output = root/(arm+'-'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    if materialize_inputs:
        np.savez(output/'prepared.npz', **arrays)
    else:
        (output/'prepared.npz').symlink_to(source/'prepared.npz')
    cfg = config_builder(settings,baseline,source,output,arm)
    (output/'manifest.json').write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
    ray.init(address=ray_address,runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(
        (str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo'),os.environ.get('PYTHONPATH','')))}})
    if ray.cluster_resources().get('GPU',0)<cfg['workers']:
        raise ValueError('Requires 16 Ray GPUs')
    class Progress(Callback):
        def on_trial_result(self,iteration,trials,trial,result,**info):
            metrics = {k:v for k,v in result.items() if isinstance(v,(int,float)) and
                (k.startswith(('ratio_','train_','val_','best_val_','relative_','early_stop_','condition_','fresh_','explicit_')) or
                 k in ('epoch','optimizer_steps','lr'))}
            if metrics:
                run.log(metrics,step=int(result['optimizer_steps']))
    logger = settings['logger']
    with wandb.init(entity=logger['entity'],project=logger['project'],name=cfg['run_name'],
        group=logger['group'],tags=cfg.get('tags',['tau','ratio-objective',arm,'16-gpu','raw-1110','BCE-reused']),
        config=cfg,dir=str(output),mode='online') as run:
        (output/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
        run.summary['phase']='training'
        run.summary['baseline_run']=settings['baseline_run']
        run.save(str(output/'manifest.json'),base_path=str(output),policy='now')
        try:
            fit = TorchTrainer(train_loop_per_worker=training_worker,train_loop_config=cfg,
                scaling_config=ScalingConfig(num_workers=cfg['workers'],use_gpu=True),
                run_config=RunConfig(name=arm,storage_path=str(output/'ray_results'),
                    callbacks=[Progress()],failure_config=FailureConfig(max_failures=0))).fit()
            run.summary.update(dict(total_fit_optimizer_steps=fit.metrics['optimizer_steps'],
                total_fit_epochs=fit.metrics['epoch'],
                early_stopped=bool(fit.metrics.get('early_stop_triggered',0))))
            saved = torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
            run.summary.update(dict(best_epoch=saved['epoch']+1,fit_optimizer_steps=saved['optimizer_steps'],
                best_val_bce=saved['val_bce'],phase='inference'))
            worker = ray.remote(num_gpus=1,num_cpus=1,max_calls=1)(inference_worker)
            ray.get([worker.remote(cfg,rank) for rank in range(cfg['workers'])])
            p,q = merge_scores(output,arrays['source_ids'][arrays['split']==2],cfg['workers'])
            run.summary['phase']='closure'
            reporter(cfg,arrays,p,q,scores,run,settings)
            run.summary['phase']='complete'
            (output/'COMPLETE').write_text('Completed new head fit and paired endpoint analysis\n')
            (root/(arm+'_latest_completed.json')).write_text(json.dumps(dict(output=str(output),
                run_id=run.id,url=run.url))+'\n')
            print('OUTPUT:',output,'WANDB:',run.url,flush=True)
        except BaseException:
            run.summary['phase']='failed'
            raise
        finally:
            ray.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('phase',choices=('prepare','mlc','nnukl','all'))
    p.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = p.parse_args()
    settings = read_settings(args.config)
    if args.phase=='all':
        failed = []
        for arm in ('mlc','nnukl'):
            result = subprocess.run([sys.executable,str(Path(__file__).resolve()),str(args.config.resolve()),
                arm,'--ray-address',args.ray_address],cwd=ROOT,check=False)
            if result.returncode:
                failed.append(arm)
                print('FAILED ARM:',arm,'; the other ablation is still attempted.',flush=True)
        if failed:
            raise SystemExit('Failed ablation(s): '+', '.join(failed))
        return
    source,baseline,arrays,scores = load_baseline(settings)
    print('REUSING BCE:',settings['baseline_run'],'SPLITS:',baseline['split_counts'],flush=True)
    if args.phase=='prepare':
        print('READY: MLC and nnUKL only; no BCE refit, feature extraction or sampling.',flush=True)
        return
    run_arm(settings,args.phase,source,baseline,arrays,scores,args.ray_address)


if __name__=='__main__':
    main()
