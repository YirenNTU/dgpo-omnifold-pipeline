"""One new context256 head; reuse completed context64 bound30 scores (16 GPUs)."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
from scripts import run_tau_bounded_ratio as bounded
from scripts.run_tau_fresh_negatives import prepare as prepare_upstream, panel_report
from scripts.run_tau_ratio_objectives import run_arm
from scripts.train_conditional_spin_ratio import build_classifier
from scripts.tau_conditioning_heads import parameter_count


def read_settings(path):
    cfg=bounded.read_settings(path)
    if (cfg['condition_hidden'],cfg['condition_width'],cfg['control_run'])!=(256,256,'a7mczoed'):
        raise ValueError('Requires context256 versus saved a7mczoed context64')
    return cfg


def load_control(settings,baseline,arrays):
    directory=Path(settings['control_directory'])
    cfg=json.loads((directory/'manifest.json').read_text())
    if json.loads((directory/'wandb.json').read_text())['id']!=settings['control_run']:
        raise ValueError('Wrong bounded control run')
    # a7mczoed failed ONLY at final logging; validate saved endpoints instead of
    # requiring its nonexistent COMPLETE marker or retraining a valid head.
    expected=bounded.make_config(settings,baseline,Path(settings['baseline_directory']),directory,'bounded')
    keys=('seed','workers','batch_size','epochs','lr','min_lr','hidden','dropout','weight_decay',
          'patience','min_delta','min_steps','head_kind','head_depth','relative_dim','relative_preprocessing',
          'packing_spec','condition_normalization','backbone_manifest','train_source','test_source',
          'train_events','test_events','split_counts','condition_pt_edges','ratio_objective','mmd_coefficient',
          'ratio_bound','fresh_negatives','baseline_run','panel_run','selector','diagnostic_every',
          'conditioning_diagnostics','skip_train_mmd_diagnostic','bootstrap','feature_batch_size')
    for key in keys:
        if cfg.get(key)!=expected.get(key): raise ValueError('Control protocol changed: '+key)
    if cfg.get('condition_width',64)!=64 or cfg.get('condition_hidden',cfg['hidden'])!=128:
        raise ValueError('Expected original 128 -> 64 condition encoder')
    if Path(cfg['panel_directory']).resolve()!=Path(settings['panel_directory']).resolve():
        raise ValueError('Control K64 panel differs')
    if (directory/'prepared.npz').resolve()!=(Path(settings['baseline_directory'])/'prepared.npz').resolve():
        with np.load(directory/'prepared.npz',allow_pickle=False) as f:
            if set(f.files)!=set(arrays) or any(not np.array_equal(f[k],v) for k,v in arrays.items()):
                raise ValueError('Control prepared inputs changed')
    saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    if (saved.get('ratio_bound')!=30 or saved.get('head_kind')!='film'
            or saved.get('condition_width',64)!=64 or saved.get('condition_hidden',saved['hidden'])!=128
            or saved['condition_dim']!=arrays['condition'].shape[1]
            or saved['candidate_dim']!=arrays['candidate_truth'].shape[1]):
        raise ValueError('Control checkpoint is not bounded FiLM')
    model=build_classifier(saved);model.load_state_dict(saved['state_dict'],strict=True)
    test=arrays['split']==2
    with np.load(directory/'test_scores.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],arrays['source_ids'][test]):
            raise ValueError('Control K1 identities changed')
        scores=f['log_ratio'].copy()
        if not np.array_equal(scores,f['generated_logits']): raise ValueError('Control K1 score convention changed')
    with np.load(directory/'fixed_panel_scores.npz',allow_pickle=False) as f:
        if f['logits'].shape!=(119002,64) or not np.array_equal(f['source_ids'],arrays['source_ids'][test]):
            raise ValueError('Control K64 identities/shape changed')
        check_scores(f['logits'])
    check_scores(scores)
    report=json.loads((directory/'fixed_K64_report.json').read_text())
    if report['events']!=119002 or report['candidates']!=64 or report['ratio_bound']!=30:
        raise ValueError('Control endpoint report incomplete')
    bounded.verify_panel_report(report,Path(settings['panel_directory']))
    return cfg,scores


def check_scores(scores):
    if not np.isfinite(scores).all() or np.max(scores)>np.log(30)+1e-6:
        raise ValueError('Control scores are not finite bounded ratios')


def make_config(settings,baseline,source,output,arm):
    cfg=bounded.make_config(settings,baseline,source,output,arm)
    cfg.update(condition_hidden=256,condition_width=256,
        control_directory=settings['control_directory'],control_run=settings['control_run'],
        parameter_counts=settings['parameter_counts'],
        initialization='Fresh seed42; shared candidate/hidden/readout parameters and initial function match context64; context projections zero.',
        intervention='Condition input ->256->256 instead of ->128->64; context projections256->128; candidate width64/depth3 unchanged.',
        primary='K64 bounded(context256) minus condition64 Cij Frobenius error, paired event bootstrap; all nine components inspected.',
        limitation='Tests condition-branch capacity/routing, not pure information loss; more parameters; fixed previously inspected panel.',
        tags=['tau','FiLM','context256','bound30','16-gpu','raw-1110','no-policy-update'])
    return cfg


def width_panel_report(inputs,data,logits,bootstrap,seed,control_logits,control_report):
    report=bounded.bounded_panel_report(inputs,data,logits,bootstrap,seed)
    check_scores(control_logits)
    pair=panel_report(inputs,dict(data,logits=control_logits),logits,bootstrap,seed)
    row=pair['arms']['old_raw'];old=control_report['arms']['bounded']
    for key in ('C','error','event_ess','candidate_ess','max_candidate_mass','log_mean_ratio'):
        if not np.allclose(row[key],old[key],atol=1e-9,rtol=1e-8):
            raise ValueError('Saved context64 endpoint did not reproduce: '+key)
    row['absolute_component_error']=np.abs(np.asarray(row['C'])-np.asarray(report['truth_C'])).reshape(9).tolist()
    report['arms']['condition64']=row
    report['comparisons']['bounded_minus_condition64']=pair['comparisons']['fresh_raw_minus_old_raw']
    report.update(primary='bounded_minus_condition64',bounded_arm='new condition256; candidate64; depth3; bound30',
                  condition64_arm='saved a7mczoed; no refit',control_endpoints_verified=True)
    return report


def finish(cfg,arrays,p,q,scores,run,settings):
    directory=Path(settings['control_directory'])
    with np.load(directory/'test_scores.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],arrays['source_ids'][arrays['split']==2]):
            raise ValueError('Control K1 identities changed')
        old_q=f['log_ratio'].copy()
    with np.load(directory/'fixed_panel_scores.npz',allow_pickle=False) as f:
        old_ids=f['source_ids'].copy();old=f['logits'].copy()
    prior=json.loads((directory/'fixed_K64_report.json').read_text())
    def report(inputs,data,logits,bootstrap,seed):
        if not np.array_equal(old_ids,inputs['source_ids']): raise ValueError('Control panel IDs differ')
        return width_panel_report(inputs,data,logits,bootstrap,seed,old,prior)
    run.summary.update(dict(condition_width=256,condition_hidden=256,candidate_width=64,
        control_run=settings['control_run'],control_refits=0))
    bounded.finish(cfg,arrays,p,q,scores,run,settings,extra_score_arms={'condition64':old_q},panel_reporter=report)
    # Add the actual width control to condition-group diagnostics, not just old cap30.
    output=Path(cfg['output']);panel=Path(cfg['panel_directory'])
    with np.load(panel/'inputs.npz',allow_pickle=False) as f:
        inputs={k:f[k] for k in ('weight','truth_cij','category','visible_pt_sum')}
    with np.load(panel/'samples_and_scores.npz',allow_pickle=False) as f:
        data={k:f[k] for k in ('logits','cij')}
    groups=json.loads((output/'fixed_K64_groups.json').read_text())
    control_groups=bounded.group_report(inputs,data,old,cfg['condition_pt_edges'])
    for row in control_groups['groups']:
        if row['arm']=='bounded': groups['groups'].append(dict(row,arm='condition64'))
    (output/'fixed_K64_groups.json').write_text(json.dumps(groups,indent=2,allow_nan=False)+'\n')
    final=json.loads((output/'fixed_K64_report.json').read_text())
    bounded.publish_panel_report(final,groups,output,run)
    run.summary['control_endpoints_verified']=True


def prepare(settings):
    source,baseline,arrays,scores=prepare_upstream(settings)
    load_control(settings,baseline,arrays)
    counts={}
    for width in (64,256):
        cfg=dict(baseline,ratio_bound=30,condition_width=width,condition_hidden=128 if width==64 else 256)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg['seed'])
            model=build_classifier(cfg,arrays['condition'].shape[1],arrays['candidate_truth'].shape[1])
        counts[str(width)]=parameter_count(model)
    settings['parameter_counts']=counts
    print('READY:',dict(control=settings['control_run'],new_fits=['condition256'],parameters=counts,
                        workers=16,batch_size=1024,policy_updates=0),flush=True)
    return source,baseline,arrays,scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('phase',choices=('prepare','train','report'),nargs='?',default='train')
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args=parser.parse_args();settings=read_settings(args.config)
    if args.phase=='report':
        if args.directory is None: parser.error('report requires --directory')
        # The common publisher uploads all arms/comparisons; no fit/inference.
        cfg=json.loads((args.directory/'manifest.json').read_text())
        report=json.loads((args.directory/'fixed_K64_report.json').read_text())
        if (cfg.get('condition_width')!=256 or cfg.get('control_run')!=settings['control_run']
                or not report.get('control_endpoints_verified')):
            raise ValueError('Not a verified condition-width report')
        bounded.recover_report(settings,args.directory)
        return
    if args.directory is not None: parser.error('--directory only applies to report')
    source,baseline,arrays,scores=prepare(settings)
    if args.phase=='train':
        run_arm(settings,'condition256',source,baseline,arrays,scores,args.ray_address,
                config_builder=make_config,reporter=finish)


if __name__=='__main__': main()
