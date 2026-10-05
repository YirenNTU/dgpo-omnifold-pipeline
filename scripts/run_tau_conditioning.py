"""User-launched matched concat / FiLM head ablation on cached raw 1110 samples."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]

import numpy as np
import torch
import yaml

from scripts.run_tau_ratio_objectives import load_baseline, run_arm, finish_reports
from scripts.train_conditional_spin_ratio import build_classifier
from scripts.tau_conditioning_heads import parameter_count
from scripts.tau_conditioning_diagnostics import fit_pt_edges, conditional_report


def read_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if cfg['workers'] != 16 or cfg['batch_size'] != 1024 or cfg['head_depth'] != 3:
        raise ValueError('Requires 16 GPUs, 1024 conditions/GPU, and 3 matched blocks')
    stop = cfg['early_stopping']
    if (stop['monitor'] != 'val_bce' or stop['patience'] < 1 or stop['min_steps'] < 0 or
        not np.isfinite(stop['min_delta']) or stop['min_delta'] < 0 or cfg['diagnostic_every'] < 1):
        raise ValueError('Invalid stopping or diagnostic cadence')
    if cfg['bootstrap'] < 20:
        raise ValueError('Require paired bootstrap')
    for arm in ('concat','film'):
        name = cfg['logger'][arm+'_name']
        if len(name)>96 or len(name.split(' | '))<3:
            raise ValueError('Invalid W&B display name')
    return cfg


def make_config(settings, baseline, source, output, arm):
    if arm not in ('concat','film'):
        raise ValueError('Only matched concat and FiLM heads are trained')
    cfg = copy.deepcopy(baseline)
    stop = settings['early_stopping']
    cfg.update(output=str(output),prepared=str(output/'prepared.npz'),checkpoint=str(output/'best.pt'),
        baseline_directory=str(source),baseline_run=settings['baseline_run'],
        ratio_objective='bce',nnukl_c=0.,mmd_coefficient=0.,
        head_kind=arm,head_depth=settings['head_depth'],
        condition_pt_edges=settings['condition_pt_edges'],
        parameter_counts=settings['parameter_counts'],
        patience=stop['patience'],min_delta=stop['min_delta'],min_steps=stop['min_steps'],
        early_stopping=stop,conditioning_diagnostics=True,diagnostic_every=settings['diagnostic_every'],
        skip_train_mmd_diagnostic=True,
        run_name=settings['logger'][arm+'_name'],wandb_group=settings['logger']['group'],
        bootstrap=settings['bootstrap'],policy_updates=0,backbone_updates=0,
        selector='minimum internal validation BCE; no selection on Cij or external test',
        budget_comparison='New concat/FiLM matched maximum budget and stopping rule; historical BCE is an anchor only.',
        ratio_transform='exp(logit), raw; no cap, tempering or per-event normalization',
        evaluation_status='Exploratory: external test was inspected in earlier rounds',
        no_bce_refit=False, historical_bce_reused=True,
        initialization='Identical encoders, blocks and outputs across arms; zero context projections in both.',
        conditioning_scope='Only new head fusion changes; cached candidate representation already depends on visible context.',
        limitation='Coarse conditional moments and fixed-model bootstrap do not certify full conditional closure.',
        tags=['tau','conditioning','capacity-matched',arm,'BCE','16-gpu','raw-1110','no-policy-update'])
    return cfg


def matched_control(settings, cfg, arrays):
    if cfg['head_kind'] != 'film':
        return {}
    pointer = Path(settings['output_root'])/'concat_latest_completed.json'
    if not pointer.is_file():
        return {}
    directory = Path(json.loads(pointer.read_text())['output'])
    if not (directory/'COMPLETE').is_file():
        raise ValueError('Concat endpoint did not complete')
    old = json.loads((directory/'manifest.json').read_text())
    keys = ('baseline_directory','baseline_run','seed','workers','batch_size','epochs','lr','min_lr',
            'hidden','dropout','weight_decay','patience','min_delta','min_steps','head_depth',
            'condition_pt_edges','ratio_objective','parameter_counts')
    if old.get('head_kind') != 'concat' or any(old[k] != cfg[k] for k in keys):
        raise ValueError('Concat control differs from the matched protocol')
    with np.load(directory/'test_scores.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],arrays['source_ids'][arrays['split']==2]):
            raise ValueError('Concat test identities differ')
        return {'matched_concat':f['log_ratio'].copy()}


def finish_conditioning(cfg, arrays, p, q, scores, run, settings):
    import wandb
    extra = matched_control(settings,cfg,arrays)
    finish_reports(cfg,arrays,p,q,scores,run,settings,extra_score_arms=extra)
    run.summary['matched_concat_available'] = bool(extra)
    if extra:
        pointer = json.loads((Path(settings['output_root'])/'concat_latest_completed.json').read_text())
        run.summary['matched_concat_directory'] = pointer['output']
        run.summary['matched_concat_run_id'] = pointer.get('run_id')
    test = arrays['split']==2
    reports, rows = {}, []
    for arm,logits in dict(bce=scores['log_ratio'],candidate_ratio=q,**extra).items():
        report = conditional_report(arrays['tau_truth'][test],arrays['tau_generated'][test],
            arrays['event_weight'][test],logits,arrays['category'][test],arrays['visible_pt_sum'][test],
            cfg['condition_pt_edges'])
        reports[arm] = report
        run.summary.update({f'conditional/{arm}/{k}':v for k,v in report['summary'].items()})
        rows.extend([[arm,*[r[k] for k in ('group','events','base_mass','weighted_mass',
            'log_mean_ratio','ess','tau_error_unweighted','tau_error_reweighted')]] for r in report['groups']])
    path = Path(cfg['output'])/'conditional_ratio_report.json'
    path.write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n')
    run.save(str(path),base_path=cfg['output'],policy='now')
    run.log({'conditional/groups':wandb.Table(columns=['arm','group','events','base_mass','weighted_mass',
        'log_mean_ratio','ess','tau_error_unweighted','tau_error_reweighted'],data=rows)})


def prepare(settings):
    source, baseline, arrays, scores = load_baseline(settings)
    settings['condition_pt_edges'] = fit_pt_edges(arrays['visible_pt_sum'],arrays['split'])
    counts = {}
    for kind in ('concat','film'):
        with torch.random.fork_rng():
            torch.manual_seed(baseline['seed'])
            model = build_classifier(dict(baseline,head_kind=kind,head_depth=settings['head_depth']),
                arrays['condition'].shape[1],arrays['candidate_truth'].shape[1])
        counts[kind] = parameter_count(model)
    if counts['concat'] != counts['film']:
        raise ValueError('Heads must have exactly equal trainable parameter counts')
    settings['parameter_counts'] = counts
    print('READY:',dict(splits=baseline['split_counts'],parameters=counts,
        source_step=1110,weights='raw',new_fits=['matched_concat','matched_film']),flush=True)
    return source,baseline,arrays,scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('phase',choices=('prepare','concat','film','all'))
    parser.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = parser.parse_args()
    settings = read_settings(args.config)
    if args.phase == 'all':
        failed = []
        for arm in ('concat','film'):
            result = subprocess.run([sys.executable,str(Path(__file__).resolve()),str(args.config.resolve()),
                arm,'--ray-address',args.ray_address],cwd=ROOT,check=False)
            if result.returncode:
                failed.append(arm)
        if failed:
            raise SystemExit('Failed arm(s): '+', '.join(failed))
        return
    source,baseline,arrays,scores = prepare(settings)
    if args.phase != 'prepare':
        run_arm(settings,args.phase,source,baseline,arrays,scores,args.ray_address,
                config_builder=make_config,reporter=finish_conditioning)


if __name__ == '__main__':
    main()
