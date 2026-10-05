"""Two matched input ablations on saved raw1110 candidates; user-launched 16 GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts import run_tau_condition_width as width
from scripts import run_tau_bounded_ratio as bounded
from scripts.run_tau_ratio_objectives import run_arm
from scripts.run_tau_fresh_negatives import panel_report
from scripts.tau_explicit_inputs import prepare_arrays, geometry_features
from scripts.train_conditional_spin_ratio import build_classifier
from scripts.tau_conditioning_heads import parameter_count


def read_settings(path):
    settings=yaml.safe_load(Path(path).read_text())
    inherited=width.read_settings(ROOT/settings['width_config'])
    inherited.update(settings)
    if inherited['wide_run']!='zrv2yfgt' or (inherited['workers'],inherited['batch_size'])!=(16,1024):
        raise ValueError('Requires saved zrv2yfgt and 16 GPUs x1024')
    for arm in ('visible','geometry'):
        name=inherited['logger'][arm+'_name']
        if len(name)>96 or not 3<=len(name.split(' | '))<=5:
            raise ValueError('Invalid display name')
    return inherited


def load_saved_scores(directory, ids):
    directory=Path(directory)
    with np.load(directory/'test_scores.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],ids): raise ValueError('K1 comparator identities differ')
        q=f['log_ratio'].copy()
        if not np.array_equal(q,f['generated_logits']): raise ValueError('Comparator uses transformed scores')
    with np.load(directory/'fixed_panel_scores.npz',allow_pickle=False) as f:
        if not np.array_equal(f['source_ids'],ids) or f['logits'].shape!=(len(ids),64):
            raise ValueError('K64 comparator identities/shape differ')
        panel=f['logits'].copy()
    width.check_scores(q);width.check_scores(panel)
    report=json.loads((directory/'fixed_K64_report.json').read_text())
    if report['events']!=len(ids) or report['candidates']!=64 or report['ratio_bound']!=30:
        raise ValueError('Incomplete comparator report')
    return q,panel,report


def verify_wide(settings,baseline,source,arrays):
    directory=Path(settings['wide_directory'])
    if not (directory/'COMPLETE').is_file(): raise ValueError('Wide baseline must be complete')
    if json.loads((directory/'wandb.json').read_text())['id']!=settings['wide_run']:
        raise ValueError('Wrong wide baseline run')
    cfg=json.loads((directory/'manifest.json').read_text())
    expected=width.make_config(settings,baseline,source,directory,'condition256')
    keys=('seed','workers','batch_size','epochs','lr','min_lr','hidden','dropout','weight_decay',
        'patience','min_delta','min_steps','head_kind','head_depth','relative_dim','relative_preprocessing',
        'packing_spec','condition_normalization','backbone_manifest','train_source','test_source',
        'train_events','test_events','split_counts','condition_pt_edges','ratio_objective','mmd_coefficient',
        'ratio_bound','fresh_negatives','baseline_run','panel_run','condition_width','condition_hidden',
        'selector','diagnostic_every','conditioning_diagnostics','skip_train_mmd_diagnostic')
    for k in keys:
        if cfg.get(k)!=expected.get(k): raise ValueError('Wide baseline protocol differs: '+k)
    if cfg.get('explicit_input'): raise ValueError('Baseline already has explicit inputs')
    if (directory/'prepared.npz').resolve()!=(source/'prepared.npz').resolve():
        with np.load(directory/'prepared.npz',allow_pickle=False) as f:
            if set(f.files)!=set(arrays) or any(not np.array_equal(f[k],v) for k,v in arrays.items()):
                raise ValueError('Wide baseline prepared data differ')
    saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    if (saved.get('condition_width'),saved.get('condition_hidden'),saved.get('ratio_bound'))!=(256,256,30):
        raise ValueError('Wrong wide checkpoint')
    model=build_classifier(saved);model.load_state_dict(saved['state_dict'],strict=True)
    if (saved['condition_dim'],saved['candidate_dim'])!=(arrays['condition'].shape[1],arrays['candidate_truth'].shape[1]):
        raise ValueError('Wide checkpoint dimensions differ')
    _,_,report=load_saved_scores(directory,arrays['source_ids'][arrays['split']==2])
    bounded.verify_panel_report(report,Path(settings['panel_directory']))
    return cfg


def load_observables(baseline,arrays):
    """Read aligned p4, rebuild the original condition, verify each candidate's geometry."""
    from scripts.diagnose_ztautau_cij import read_event_table,source_ids,align,SOURCE_ID_COLUMNS
    from scripts.conditional_tau_preprocessing import apply_masked_feature
    n=len(arrays['split'])
    va=np.empty((n,4));vb=np.empty((n,4));gt=np.empty((n,6),np.float32);gg=np.empty_like(gt)
    for part in ('train','test'):
        take=arrays['split']!=2 if part=='train' else arrays['split']==2
        source=Path(baseline[part+'_source']);events=Path(baseline[part+'_events'])
        if json.loads((events.parent/'filter_manifest.json').read_text()).get('complete') is not True:
            raise ValueError('Incomplete filtered input')
        with np.load(source/'candidates.npz',allow_pickle=False) as f:
            keys=[k for k in SOURCE_ID_COLUMNS if k in f]
            ids=source_ids({k:f[k] for k in keys},keys)
            order=align(ids,arrays['source_ids'][take])
            truth=f['truth_deltas'][order];generated=f['deltas'][order,0]
        fields=keys+['event_category','event_weight']+[f'lead_{leg}_visible_{c}' for leg in ('a','b') for c in ('E','px','py','pz')]
        table=read_event_table(events,fields)
        expected_rows=416701 if part=='train' else 119002
        if len(table)!=expected_rows: raise ValueError('Filtered population changed')
        cols={key:np.asarray(table[key].to_numpy()) for key in fields}
        idx=align(source_ids(cols,keys),arrays['source_ids'][take]);cols={k:v[idx] for k,v in cols.items()}
        if not np.array_equal(cols['event_category'],arrays['category'][take]): raise ValueError('Category mismatch')
        if not np.allclose(cols['event_weight'],arrays['event_weight'][take],rtol=1e-6,atol=1e-7):
            raise ValueError('Base weight mismatch')
        a=np.stack([cols[f'lead_a_visible_{c}'] for c in ('E','px','py','pz')],1)
        b=np.stack([cols[f'lead_b_visible_{c}'] for c in ('E','px','py','pz')],1)
        pool=source/('pool.pt' if (source/'pool.pt').is_file() else 'validation_pool.pt')
        raw=torch.load(pool,map_location='cpu',weights_only=True)['condition'][order].numpy()
        cat=cols['event_category'].astype(int)
        onehot=np.eye(16,dtype=np.float32)[(cat//10-1)*4+cat%10-1]
        rebuilt=apply_masked_feature(np.concatenate((raw,onehot),1),arrays['condition_mean'],
                                    arrays['condition_scale'],baseline['packing_spec'])
        if not np.allclose(rebuilt,arrays['condition'][take],rtol=1e-6,atol=1e-6):
            raise ValueError('Saved packed condition pairing/preprocessing changed')
        va[take],vb[take]=a,b
        gt[take]=geometry_features(a,b,truth);gg[take]=geometry_features(a,b,generated)
    for value,left,right in ((gt,'truth_a','truth_b'),(gg,'sample_a','sample_b')):
        if not np.allclose(value,np.concatenate((arrays[left],arrays[right]),1),atol=2e-6,rtol=1e-6):
            raise ValueError('Recomputed analyzer geometry differs from saved analysis')
    shapes=baseline['packing_spec']['shapes']
    inventory=dict(direct_named_p4_fields={f'lead_{leg}_visible_{key}':f'lead_{leg}_visible_{key}' in shapes
        for leg in ('a','b') for key in ('E','px','py','pz')},
        interpretation='Named-field inventory only. Particle x may encode equivalent information; redundancy not certified.',
        effect='B tests explicit leg-associated p4 access; may also supply previously absent energy information.',
        original_condition_reproduced=True,candidate_geometry_reproduced=True)
    return va,vb,gt,gg,inventory


def prepare(settings):
    source,baseline,arrays,scores=width.prepare(settings)
    verify_wide(settings,baseline,source,arrays)
    obs=load_observables(baseline,arrays)
    # K64 uses the exact same visible p4 and saved original conditioning.
    with np.load(Path(settings['panel_directory'])/'inputs.npz',allow_pickle=False) as f:
        test=arrays['split']==2
        for key,value in [('visible_a',obs[0][test]),('visible_b',obs[1][test]),('condition',arrays['condition'][test])]:
            if not np.allclose(f[key],value,rtol=1e-6,atol=1e-6): raise ValueError('K64 input differs: '+key)
    return source,baseline,arrays,scores,obs


def make_config(settings,baseline,source,output,arm):
    cfg=width.make_config(settings,baseline,source,output,arm)
    cfg.update(explicit_input=settings['explicit_input'],input_inventory=settings['input_inventory'],
        condition_dim=settings['input_dimensions']['condition'],
        candidate_dim=settings['input_dimensions']['candidate'],
        wide_directory=settings['wide_directory'],wide_run=settings['wide_run'],
        visible_control=settings.get('visible_control'),
        parameter_counts=settings['explicit_parameter_counts'],
        run_name=settings['logger'][arm+'_name'],
        initialization='Fresh seed42; shared initial tensors/RNG unchanged; new input columns zero.',
        intervention='Append observed visible p4 (8) to condition'+('; insert candidate-owned rest-frame directions (6)' if arm=='geometry' else ''),
        primary='K64 Cij Frobenius error: '+('geometry minus explicit-visible' if arm=='geometry' else 'explicit-visible minus saved condition256'),
        limitation='Physics-aware observable representation; inherited analyzer convention; fixed inspected panel; no independent convention certification or full conditional closure proof.',
        tags=['tau','FiLM','context256','bound30',arm,'16-gpu','raw-1110','no-policy-update'])
    return cfg


def add_comparison(report,inputs,data,logits,bootstrap,seed,label,control,prior):
    width.check_scores(control)
    pair=panel_report(inputs,dict(data,logits=control),logits,bootstrap,seed)
    row=pair['arms']['old_raw']
    for key in ('C','error','event_ess','candidate_ess','max_candidate_mass','log_mean_ratio'):
        if not np.allclose(row[key],prior['arms']['bounded'][key],atol=1e-9,rtol=1e-8):
            raise ValueError('Comparator endpoint failed replay: '+label+'/'+key)
    row['absolute_component_error']=np.abs(np.asarray(row['C'])-report['truth_C']).reshape(9).tolist()
    report['arms'][label]=row
    report['comparisons']['bounded_minus_'+label]=pair['comparisons']['fresh_raw_minus_old_raw']


def load_visible_control(settings,arrays,expected):
    pointer=Path(settings['output_root'])/'visible_latest_completed.json'
    if not pointer.is_file(): raise ValueError('Run explicit-visible first; geometry needs its matched completed control')
    directory=Path(json.loads(pointer.read_text())['output'])
    if not (directory/'COMPLETE').is_file(): raise ValueError('Visible control incomplete')
    cfg=json.loads((directory/'manifest.json').read_text())
    for key in ('seed','workers','batch_size','epochs','lr','min_lr','hidden','dropout','weight_decay',
            'patience','min_delta','min_steps','condition_width','condition_hidden','head_kind','head_depth',
            'ratio_bound','ratio_objective','mmd_coefficient','relative_dim','relative_preprocessing',
            'packing_spec','condition_normalization','backbone_manifest','train_source','test_source',
            'train_events','test_events','split_counts','condition_pt_edges','wide_directory','wide_run',
            'panel_directory','panel_run','selector','diagnostic_every','skip_train_mmd_diagnostic'):
        if cfg.get(key)!=expected.get(key): raise ValueError('Visible control protocol changed: '+key)
    if (cfg.get('wide_run')!=settings['wide_run'] or cfg.get('explicit_input',{}).get('arm')!='visible'
        or cfg.get('explicit_input',{}).get('visible_mean')!=settings['explicit_input']['visible_mean']
        or cfg.get('explicit_input',{}).get('visible_scale')!=settings['explicit_input']['visible_scale']):
        raise ValueError('Wrong explicit-visible comparator')
    # Strict data match, excluding only the six newly inserted candidate coordinates.
    with np.load(directory/'prepared.npz',allow_pickle=False) as f:
        for key,value in arrays.items():
            if key.startswith('candidate_'):
                value=np.concatenate((value[:,:-27],value[:,-21:]),1)
            if key not in f or not np.array_equal(value,f[key]): raise ValueError('Visible control data changed: '+key)
    load_saved_scores(directory,arrays['source_ids'][arrays['split']==2])
    return str(directory)


def finish(cfg,arrays,p,q,scores,run,settings):
    controls={'condition256':settings['wide_directory']}
    if settings.get('visible_control'): controls['explicit_visible']=settings['visible_control']
    if settings.get('geometry_control'): controls['geometry']=settings['geometry_control']
    ids=arrays['source_ids'][arrays['split']==2]
    saved={label:load_saved_scores(path,ids) for label,path in controls.items()}
    def report(inputs,data,logits,bootstrap,seed):
        if not np.array_equal(inputs['source_ids'],ids): raise ValueError('Panel comparator IDs differ')
        result=bounded.bounded_panel_report(inputs,data,logits,bootstrap,seed)
        for label,(_,control,prior) in saved.items():
            add_comparison(result,inputs,data,logits,bootstrap,seed,label,control,prior)
        primary_control = 'geometry' if 'geometry' in saved else ('explicit_visible' if 'explicit_visible' in saved else 'condition256')
        result.update(primary='bounded_minus_'+primary_control,
                      bounded_arm=cfg['explicit_input']['arm'],control_endpoints_verified=True)
        return result
    bounded.finish(cfg,arrays,p,q,scores,run,settings,
        extra_score_arms={label:value[0] for label,value in saved.items()},panel_reporter=report)
    output=Path(cfg['output']);panel=Path(cfg['panel_directory'])
    with np.load(panel/'inputs.npz',allow_pickle=False) as f:
        inputs={k:f[k] for k in ('weight','truth_cij','category','visible_pt_sum')}
    with np.load(panel/'samples_and_scores.npz',allow_pickle=False) as f:
        data={k:f[k] for k in ('logits','cij')}
    groups=json.loads((output/'fixed_K64_groups.json').read_text())
    for label,(_,control,_) in saved.items():
        groups['groups'] += [dict(row,arm=label) for row in bounded.group_report(inputs,data,control,cfg['condition_pt_edges'])['groups'] if row['arm']=='bounded']
    (output/'fixed_K64_groups.json').write_text(json.dumps(groups,indent=2,allow_nan=False)+'\n')
    bounded.publish_panel_report(json.loads((output/'fixed_K64_report.json').read_text()),groups,output,run)
    run.summary.update(dict(control_refits=0,explicit_input_arm=cfg['explicit_input']['arm'],wide_control_run=settings['wide_run']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('phase',choices=('prepare','visible','geometry','all'),nargs='?',default='all')
    parser.add_argument('--ray-address',default=os.environ.get('RAY_ADDRESS') or 'auto')
    args=parser.parse_args();settings=read_settings(args.config)
    if args.phase=='all':
        for arm in ('visible','geometry'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),str(args.config.resolve()),arm,
                            '--ray-address',args.ray_address],cwd=ROOT,check=True)
        return
    source,baseline,arrays,scores,obs=prepare(settings)
    if args.phase=='prepare':
        print('READY: input provenance verified; no training or inference launched.',flush=True);return
    arrays,spec=prepare_arrays(arrays,*obs[:4],args.phase)
    settings.update(explicit_input=spec,input_inventory=obs[4],explicit_parameter_counts={},
        input_dimensions=dict(condition=arrays['condition'].shape[1],candidate=arrays['candidate_truth'].shape[1]))
    cfg=make_config(settings,baseline,source,source,args.phase)
    settings['explicit_parameter_counts'][args.phase]=parameter_count(build_classifier(cfg))
    if args.phase=='geometry': settings['visible_control']=load_visible_control(settings,arrays,cfg)
    run_arm(settings,args.phase,source,baseline,arrays,scores,args.ray_address,
        config_builder=make_config,reporter=finish,materialize_inputs=True)


if __name__=='__main__': main()
