"""Prepare/preflight user-launched General + configurable oracle DGPO."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml

LEGACY_FIT = dict(epochs=1000, lr_decay_epochs=250, require_plateau=True, weight_decay=.005, dropout=.1)
ADAPTIVE_FIT = dict(epochs=3000, scheduler='validation_plateau', lr=1e-4, min_lr=1e-5,
    lr_patience=8, lr_factor=.5, batch_size=1024, patience=25, min_delta=.0001,
    require_plateau=True, weight_decay=.005, dropout=.1,
    checkpoint_fit=True, allow_best_warm_start=True)

COSINE_FIT = dict(epochs=3000, scheduler='cosine', lr=1e-4, min_lr=1e-5,
    lr_decay_epochs=1000, batch_size=1024, patience=25, min_delta=.0001,
    require_plateau=True, weight_decay=.005, dropout=.1,
    checkpoint_fit=True, allow_best_warm_start=True)

COSINE1000_FIT = dict(COSINE_FIT)
COSINE_FIT = dict(COSINE1000_FIT, lr_decay_epochs=1500)


def validate_runtime_workers(r):
    """Check the actual refit group as well as Ray and launch metadata."""
    groups = dict(platform=r['platform']['number_of_workers'],
        tau_ratio=r['dgpo']['tau_ratio']['workers'],
        execution=r['nersc']['execution']['workers'])
    if len({int(n) for n in groups.values()}) != 1:
        raise ValueError(f'Inconsistent DGPO worker groups: {groups}')


def settings(path):
    c=yaml.safe_load(Path(path).read_text())
    from RL.DGPO_neutrino.tau_oracle_mixture import POLAR_GEOMETRY, MOMENTUM_GEOMETRY, RECO_MC_GEOMETRY
    polar = (c.get('joint_oracle') or {}).get('geometry') == POLAR_GEOMETRY
    momentum = (c.get('joint_oracle') or {}).get('geometry') == MOMENTUM_GEOMETRY
    reco_mc = (c.get('joint_oracle') or {}).get('geometry') == RECO_MC_GEOMETRY
    expected=dict(workers=4 if momentum else 16,seed=42,epochs=1500,refit_relative_to_install=True,validation_every_epochs=5)
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Pinned single-seed General oracle protocol/workers changed')
    if c.get('refit_every_epochs') not in (30,50):raise ValueError('Require refit interval 300 or 500 DGPO steps')
    allowed_fractions = (.5,) if momentum else (.1,.2,.3)
    if c.get('oracle_fraction') not in allowed_fractions:raise ValueError('Oracle fraction differs from this experiment')
    if c['oracle_fraction'] in (.2,.3) and c.get('continuation_checkpoint'):
        raise ValueError('20/30% ablations must start from epoch287 pretrain')
    if c.get('adaptive_oracle') is not None:
        from RL.DGPO_neutrino.tau_adaptive_oracle import validate_gate
        validate_gate(c['adaptive_oracle'])
        if c['oracle_fraction']!=.3 or c.get('continuation_checkpoint'):
            raise ValueError('Adaptive30 starts fresh from epoch287')
    if c.get('joint_oracle') is not None:
        from RL.DGPO_neutrino.tau_oracle_mixture import joint_contract
        joint_contract(dict(enabled=True,alpha=c['oracle_fraction'],**c['joint_oracle']))
        if c.get('continuation_checkpoint') or c.get('adaptive_oracle'):
            raise ValueError('Normalized joint10 starts fresh from epoch287, fixed alpha')
        if not c['joint_oracle'].get('scale_source') or not c.get('joint_monitor'):
            raise ValueError('Joint oracle needs normalization provenance and raw joint monitoring')
        filename = 'tau_dgpo_general_polar_oracle10_epoch287.yaml' if polar else 'tau_dgpo_general_joint_oracle10_epoch287.yaml'
        if momentum:
            filename = 'tau_dgpo_general_momentum_oracle50_epoch287.yaml'
            if (c['joint_oracle'].get('fixed_energy_GeV') != 45.625
                    or c['joint_oracle'].get('tau_mass_GeV') != 1.77
                    or c['joint_oracle'].get('term_order') != ['mean_two_leg_p3_error_squared_over_p0_squared']):
                raise ValueError('Momentum run must match the existing fixed-energy cache definition')
        if reco_mc:
            filename = 'tau_dgpo_general_reconstructed_tau_oracle30_epoch287.yaml'
            if (c['joint_oracle'].get('fixed_energy_GeV') != 45.6
                    or c['joint_oracle'].get('tau_mass_GeV') != 1.777
                    or c['joint_oracle'].get('term_order') != ['mean_two_leg_euclidean_p4_error_squared_over_p0_squared']):
                raise ValueError('Reconstructed MC oracle requires exporter E45.6/m1.777 and actual MC p4 distance')
        if c.get('command_config') != filename:
            raise ValueError('Use the dedicated oracle launch config: '+filename)
        if polar and (c.get('unfolding_kregs') != [5,6] or c['joint_oracle'].get('term_order') != ['signed_polar_sum_error_squared']):
            raise ValueError('Polar-only requires signed-polar ranking and declared common kreg5/6 evaluation')
        spec=json.loads((ROOT/c['joint_monitor']).read_text())
        if spec['method']!='RooUnfoldSvd' or spec['kreg']!=5 or any(len(e)!=9 for e in spec['edges'].values()):
            raise ValueError('Joint monitoring requires pinned physical8 bins and common kreg5')
    if c.get('classifier_fit') not in (LEGACY_FIT, ADAPTIVE_FIT, COSINE1000_FIT, COSINE_FIT):
        raise ValueError('Require extended classifier fitting with validation plateau selection')
    if c['unfolding']!=dict(method='RooUnfoldSvd',bins=8,kreg=5,response_bootstraps=300,seed=4203):raise ValueError('SVD protocol changed')
    if not c['run_name'].isascii() or len(c['run_name'])>=96:raise ValueError('Readable W&B name required')
    out=Path(c['output']).resolve()
    for key in ('source_directory','classifier_directory'):
        src=Path(c[key]).resolve()
        if out.is_relative_to(src) or src.is_relative_to(out):raise ValueError('Keep source separate from output')
    return c


def assemble(source,c):
    from scripts.train_dgpo_tau_process_labels import validate_settings
    # Reuse the audited original General DGPO runtime, not a spin/PET head.
    validate_settings(source)
    from scripts.train_tau_dgpo_spin_angles import assemble as base_assemble
    options=dict(c,cij_bootstrap=1000,classifier_run='epoch287-general-step0')
    r=base_assemble(source,options)
    dg=r['dgpo'];tau=dg['tau_ratio']
    tau.update(combined_reward=False,direct_oracle_monitor=True,
        refit_relative_to_install=c.get('refit_relative_to_install',False))
    # Missing override preserves historical runtime assembly for migration checks.
    tau['fit'].update(c.get('classifier_fit',{}))
    tau.pop('combined_classifier_directory',None)
    tau.pop('wandb_profile',None)
    alpha=float(c['oracle_fraction']);percent=f'{100*alpha:g}'
    dg['oracle_mixture']=dict(enabled=True,alpha=alpha,geometry='sum_two_leg_squared_geodesic',reward='log_convex_weights')
    if c.get('joint_oracle') is not None:
        dg['oracle_mixture'].update(copy.deepcopy(c['joint_oracle']))
        tau['joint_oracle_monitor']=json.loads((ROOT/c['joint_monitor']).read_text())
    if c.get('adaptive_oracle') is not None:
        dg['oracle_mixture']['adaptive']=copy.deepcopy(c['adaptive_oracle'])
        tau['adaptive_oracle']=copy.deepcopy(c['adaptive_oracle'])
    if dg.get('adv_clip_max') is not None:
        raise ValueError('Do not silently clip the new mixture advantage')
    r['logger']['wandb'].pop('profile',None)
    r['logger']['wandb'].update(group='Tau DGPO General oracle transfer',
        tags=['DGPO','General',f'Oracle{percent}pct','Epoch287','Filtered10pct','GPU16','SingleSeed',f"Refit{c['refit_every_epochs']}",'Validation5'])
    r['logger']['local']['name']=f'tau-dgpo-general-oracle{percent}'
    r['experiment']=dict(protocol=f'general-oracle{percent}-log-mixture-v1',seed=c['seed'],source_completed_epochs=287,
        source_global_step=18655,oracle_fraction=alpha,K=8,workers=16,
        reward=f'log({1-alpha:g} softmax(General logit) + {alpha:g} onehot(two-leg nearest truth))',
        advantage='Unchanged leave_one_out_unscaled; alpha0 equals original General advantage',
        reference='Original velocity MSE coefficient1; recentered after accepted General refits',
        validation='Unweighted K8 native policy samples; fixed validation noise seed and event order',
        unfolding=c['unfolding'],stop_policy_step=15000,
        limitation='Truth-assisted training; no inference oracle. Best-of-K transfer is not guaranteed and no hard anti-collapse bound.',
        evaluation_scope='Four direct offsets and conditional spread; no concurrence claim')
    if c.get('adaptive_oracle') is not None:
        r['experiment'].update(adaptive_oracle=c['adaptive_oracle'],
            reward='log((1-alpha_round) softmax(General logit) + alpha_round onehot(two-leg nearest truth))',
            alpha_update='0.3 * clip(max(AUC-0.5-1.96*paired_SE,0)/initial_resolved_gap,0,1); held fixed within each refit round',
            calibration='11903 fixed external-validation events, K1; uncapped logits. Exclude reserved IDs from final physics evaluation. Initial General reused; later heads require validation plateau.',
            resume='Persist baseline gap and round calibration in reward stack; never silently reset alpha')
        r['logger']['wandb']['tags'].append('AdaptiveOracle')
    if c.get('joint_oracle') is not None:
        r['experiment'].update(protocol='general-normalized-joint-oracle10-v1',
            reward='log(0.9 softmax(current General logit) + 0.1 onehot(argmin normalized joint error))',
            joint_oracle=copy.deepcopy(c['joint_oracle']),
            evaluation_scope='Four direct offsets, opening cosine, acoplanarity, polar deficit, signed polar, ordered pair-axis xyz and error; raw K8 conditional spread; local RooUnfoldSvd with closure and separate data/response MC errors',
            normalization='Three frozen mean-squared error scales from development response panel; not per batch/refit; reused development validation, not a final test')
        r['logger']['wandb']['tags'].append('NormalizedJointOracle')
        r['logger']['local']['name']='tau-dgpo-general-joint-oracle10'
        from RL.DGPO_neutrino.tau_oracle_mixture import POLAR_GEOMETRY
        if c['joint_oracle']['geometry'] == POLAR_GEOMETRY:
            r['platform']['number_of_workers']=c['workers']
            tau['workers']=c['workers']
            r['nersc'].update(nodes=1,gpus_per_node=4)
            r['nersc']['execution']['workers']=c['workers']
            r['nersc']['reproducibility'].update(source_checkpoint=str(Path(c['output'])/'source.ckpt'),
                source_completed_epochs=287,source_global_step=18655,source_dgpo_global_step=None,
                source_wandb_run=None,diagnostic_wandb_run=None,classifier_backbone_checkpoint=str(Path(c['output'])/'source.ckpt'),
                note='Fresh supervised10pct epoch287 weights-only source, verified epoch286/global18655; original matched General step0. One4GPU node, fixed signed-polar10 preference after every refit. No inherited DGPO optimizer or late-policy checkpoint; source network and pinned normalization preserved.')
            r['logger']['wandb']['tags']=[x for x in r['logger']['wandb']['tags'] if x not in ('GPU16','NormalizedJointOracle')]+['GPU4','SignedPolarOracle']
            r['logger']['local']['name']='tau-dgpo-general-polar-oracle10'
            r['experiment'].update(protocol='general-signed-polar-oracle10-v1',workers=c['workers'],
                reward='log(0.9 softmax(current General logit) +0.1 onehot(argmin signed polar-sum error squared / frozen scale))',
                normalization='One frozen response-only mean-squared signed-polar error scale. Positive scalar cannot change ranking; no batch/refit adaptation.',
                unfolding_kregs=c['unfolding_kregs'],primary='Raw unweighted polar-deficit RooUnfoldSvd at common8physicalbins,kreg6, with bias/closure and separate data/response-MC; preserve historical kreg5 and all10 guards.',
                resource_scope='One4-GPU node, source per-GPU batch retained. Historical16-GPU optimization runs are not matched global-batch controls; compare this raw policy with its paired epoch287 baseline.',
                policy_events_per_gpu=r['platform']['batch_size'],policy_global_events=r['platform']['batch_size']*c['workers'])
            if c['workers']==16:
                r['nersc']['nodes']=4
                r['logger']['wandb']['tags']=[x if x!='GPU4' else 'GPU16' for x in r['logger']['wandb']['tags']]
                r['nersc']['reproducibility']['note']=r['nersc']['reproducibility']['note'].replace('One4GPU node','Four4GPU nodes (16 workers), explicitly requested by user')
                r['experiment']['resource_scope']='User-requested four4-GPU nodes,16workers; source per-GPU policy batch512 retained,global8192. Compare raw policy with its paired epoch287 baseline.'
        from RL.DGPO_neutrino.tau_oracle_mixture import MOMENTUM_GEOMETRY
        if c['joint_oracle']['geometry'] == MOMENTUM_GEOMETRY:
            r['platform']['number_of_workers']=c['workers']
            tau['workers']=c['workers']
            r['nersc'].update(nodes=1,gpus_per_node=4)
            r['nersc']['execution']['workers']=c['workers']
            r['nersc']['reproducibility'].update(source_checkpoint=str(Path(c['output'])/'source.ckpt'),
                source_completed_epochs=287,source_global_step=18655,source_dgpo_global_step=None,
                source_wandb_run=None,diagnostic_wandb_run=None,classifier_backbone_checkpoint=str(Path(c['output'])/'source.ckpt'),
                note='Fresh supervised10pct epoch287 weights-only source, epoch286/global18655; original General step0. One4GPU node. Fixed General50/momentum50 after every refit; source architecture and normalization retained.')
            r['logger']['wandb']['tags']=[x for x in r['logger']['wandb']['tags'] if x not in ('GPU16','NormalizedJointOracle')]+['GPU4','FixedEnergyMomentumOracle']
            r['logger']['local']['name']='tau-dgpo-general-momentum-oracle50'
            r['experiment'].update(protocol='general-fixed-energy-momentum-oracle50-v1',workers=c['workers'],
                reward='log(0.5 softmax(current General logit) +0.5 onehot(argmin mean_leg ||p0*u-p0*u_truth||^2/p0^2))',
                normalization='Equal legs/Cartesian axes, common p0=sqrt(45.625^2-1.77^2) GeV; same fixed energy for candidate and truth anchor. No batch/refit scale changes.',
                primary='Raw unweighted theta targets and polar deficit RooUnfoldSvd versus paired epoch287; common8bins/kreg5, closure and separate data/response-MC uncertainty. Require distribution and conditional-spread guards alongside precision.',
                resource_scope='One4GPU node; source policy batch512/GPU retained, global2048. Historical16GPU runs are not matched global-batch controls.',
                policy_events_per_gpu=r['platform']['batch_size'],policy_global_events=r['platform']['batch_size']*c['workers'],
                limitation='Truth-assisted preference. Fixed-energy p3 ranking nearly duplicates old angular oracle on existing cache. This50% arm changes alpha as well as geometry versus old10%, and has no anti-collapse guarantee. No new energy output or changed diffusion target.')
        from RL.DGPO_neutrino.tau_oracle_mixture import RECO_MC_GEOMETRY
        if c['joint_oracle']['geometry'] == RECO_MC_GEOMETRY:
            r['platform']['number_of_workers']=c['workers']
            tau['workers']=c['workers']
            r['nersc'].update(nodes=4,gpus_per_node=4)
            r['nersc']['execution']['workers']=c['workers']
            r['nersc']['reproducibility'].update(source_checkpoint=str(Path(c['output'])/'source.ckpt'),
                source_completed_epochs=287,source_global_step=18655,source_dgpo_global_step=None,
                source_wandb_run=None,diagnostic_wandb_run=None,classifier_backbone_checkpoint=str(Path(c['output'])/'source.ckpt'),
                note='Fresh epoch287 weights-only, epoch286/global18655; original step0 General.16GPUs;fixed70/30 after every refit. Reconstructed E45.6/m1.777 raw directions versus actual MC p4, no post calibration; truth excluded from policy/General inputs.')
            r['logger']['wandb']['tags']=[x for x in r['logger']['wandb']['tags'] if x!='NormalizedJointOracle']+['ReconstructedTauOracle','ActualMCTruth','NoPostCalibration']
            r['logger']['local']['name']='tau-dgpo-general-reconstructed-tau-oracle30'
            r['experiment'].update(protocol='general-reconstructed-mc-tau-oracle30-v1',workers=c['workers'],
                reward='log(0.7 softmax(current General logit) +0.3 onehot(argmin mean_leg Euclidean(p4_reco-p4_MC)^2/p0^2))',
                normalization='Equal legs and Cartesian components; common p0=sqrt(45.6^2-1.777^2) GeV. Actual MC energies and momentum magnitudes unmodified. Fixed candidate energy contributes a K-independent term; no per-batch/refit rescaling.',
                primary='Raw unweighted theta and polar-geometry RooUnfoldSvd versus paired epoch287 and old-oracle30 where available; fixed8bins/kreg5, closure, separate data/response-MC errors and all10 distribution/spread guards.',
                resource_scope='Matched16GPU setup,4nodes; source batch512/GPU,global8192; seed42,K8; no inherited DGPO optimizer.',
                policy_events_per_gpu=r['platform']['batch_size'],policy_global_events=r['platform']['batch_size']*c['workers'],
                limitation='Truth-assisted training only. Candidate energy is fixed, not learned. Changes in ranking arise from spatial distance and actual MC leg momenta. No guaranteed precision gain or anti-collapse bound; lab geometry is not spin closure.')
    if c.get('continuation_checkpoint'):
        r['experiment'].update(continuation_checkpoint=c['continuation_checkpoint'],
            continuation_policy_step=290,continuation_mode='full_state_new_run',
            inherited_validation_baseline='parent step0 epoch287')
    config_name='tau_dgpo_general_oracle10_step290_regularized.yaml' if c.get('continuation_checkpoint') else f'tau_dgpo_general_oracle{percent}_epoch287.yaml'
    if c.get('adaptive_oracle') is not None:config_name='tau_dgpo_general_adaptive_oracle30_epoch287.yaml'
    if c.get('joint_oracle') is not None:config_name=c['command_config']
    r['nersc']['execution']['command']=f'shifter python3 -u scripts/train_tau_dgpo_general_oracle.py config/{config_name} train'
    validate_runtime_workers(r)
    return r


def update_workers(c):
    """Repair only the failed polar launch's inherited nested16 vs Ray4."""
    from RL.DGPO_neutrino.tau_oracle_mixture import POLAR_GEOMETRY
    if c['workers'] != 4 or (c.get('joint_oracle') or {}).get('geometry') != POLAR_GEOMETRY:
        raise ValueError('Worker repair is only for the four-worker signed-polar run')
    out=Path(c['output']); path=out/'runtime.yaml'
    if not (out/'PREPARED').exists() or json.loads((out/'settings.json').read_text()) != c:
        raise ValueError('Require the original prepared polar settings')
    source=yaml.safe_load((Path(c['source_directory'])/'runtime.yaml').read_text())
    expected=assemble(source,c); actual=yaml.safe_load(path.read_text())
    if actual == expected:
        print('WORKERS ALREADY ALIGNED: Ray4/refit4; no training started',flush=True)
        return path
    old=copy.deepcopy(expected);old['dgpo']['tau_ratio']['workers']=16
    if actual != old:
        raise ValueError('Worker repair permits ONLY tau_ratio.workers16->4; other runtime changes found')
    if any((out/'checkpoints').rglob('*.ckpt')):
        raise ValueError('Worker repair requires the failed-before-step-zero run without checkpoints')
    record=out/'worker-group-repair.json'
    audit=dict(previous_runtime=actual,new_runtime=expected,
        scope='Only dgpo.tau_ratio.workers16->4; settings, source, reward, optimizer and failed Ray logs preserved')
    if record.exists():
        if json.loads(record.read_text()) != audit:raise ValueError('Worker repair audit differs')
    else:
        with record.open('x') as f:json.dump(audit,f,indent=2)
    temporary=path.with_suffix('.workers-pending.yaml')
    temporary.write_text(yaml.safe_dump(expected,sort_keys=False));temporary.replace(path)
    print('WORKERS FIXED: Ray4/refit4; previous runtime saved in worker-group-repair.json. No training started.',flush=True)
    return path


def update_resources(c):
    """User-requested polar4->16 migration before any policy checkpoint."""
    from RL.DGPO_neutrino.tau_oracle_mixture import POLAR_GEOMETRY
    if c['workers']!=16 or (c.get('joint_oracle') or {}).get('geometry')!=POLAR_GEOMETRY:
        raise ValueError('Resource migration is only for user-requested signed-polar16')
    out=Path(c['output']);path=out/'runtime.yaml'
    if not (out/'PREPARED').exists():raise ValueError('Require prepared polar output')
    saved=json.loads((out/'settings.json').read_text())
    actual=yaml.safe_load(path.read_text())
    source=yaml.safe_load((Path(c['source_directory'])/'runtime.yaml').read_text())
    expected=assemble(source,c)
    if saved==c and actual==expected:
        print('RESOURCES ALREADY ALIGNED:16workers,4nodes; no training started',flush=True)
        return path
    record=out/'resources-four-to-sixteen.json'
    if record.exists():
        audit=json.loads(record.read_text());old=audit['previous_settings'];oldruntime=audit['previous_runtime']
        if audit['new_settings']!=c or audit['new_runtime']!=expected:raise ValueError('Resource migration audit differs')
    else:
        old=saved;oldruntime=actual
    changed=copy.deepcopy(old);changed['workers']=16
    if old.get('workers')!=4 or changed!=c or oldruntime!=assemble(source,old):
        raise ValueError('Resource migration permits ONLY the prepared polar4->16 configuration')
    if saved not in (old,c) or actual not in (oldruntime,expected):raise ValueError('Unexpected resource migration state')
    if any((out/'checkpoints').rglob('*.ckpt')):
        raise ValueError('Do not change this run allocation after policy checkpoint creation')
    if not record.exists():
        audit=dict(previous_settings=old,previous_runtime=oldruntime,new_settings=c,new_runtime=expected,
            scope='Explicit user-requested4->16 workers/1->4 nodes before policy updates; source/reward/per-GPU batches/fit cadence unchanged')
        with record.open('x') as f:json.dump(audit,f,indent=2)
    for name,value in [('runtime.yaml',yaml.safe_dump(expected,sort_keys=False)),('settings.json',json.dumps(c,indent=2))]:
        tmp=out/(name+'.resources-pending');tmp.write_text(value);tmp.replace(out/name)
    print('RESOURCES UPDATED:16workers,4nodes; policy512/GPU,General1024/GPU; reward90/10 unchanged. No training started.',flush=True)
    return path


def continuation_state(c):
    """Validate an explicit full-state step290 branch before creating its output."""
    if not c.get('continuation_checkpoint'):return None
    path=Path(c['continuation_checkpoint']).resolve(strict=True)
    if Path(c['output']).resolve() in path.parents:raise ValueError('Continuation source must be separate')
    p=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    for key in ('dgpo_optimizer_state_dict','dgpo_ref_state_dict','dgpo_omnifold_reward_stack','dgpo_next_epoch'):
        if key not in p:raise ValueError('Incomplete continuation checkpoint: '+key)
    h=p['dgpo_omnifold_reward_stack']
    if (p['global_step'],p['dgpo_next_epoch'],h['round_id'],h['last_refit_epoch'])!=(290,29,0,-1):
        raise ValueError('Require pre-refit step290 state with original General')
    if h['head']['candidate_dim']!=533 or h['head']['source_policy_step']!=h['denominator_step']:
        raise ValueError('Continuation General mismatch')
    baseline=path.parent.parent/'tau_diagnostics/eval-step-00000000'
    if not all((baseline/f).is_file() for f in ('DIRECT_COMPLETE','direct_metrics.json','direct_samples.npz')):
        raise ValueError('Missing parent step0 monitoring baseline')
    if not (baseline.parent/'direct_monitor_protocol.json').is_file():
        raise ValueError('Missing parent step-zero monitor protocol')
    return path,baseline


def verify(c):
    from scripts.train_conditional_spin_ratio import build_classifier
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from scripts.train_diffusion_film_10pct import TRAIN,VAL
    src=Path(c['source_directory']);cd=Path(c['classifier_directory'])
    source=yaml.safe_load((src/'runtime.yaml').read_text());assemble(source,c)
    for path,n in ((TRAIN,416701),(VAL,119002)):
        if validate_filtered_dataset(path)['rows']!=n:raise ValueError('Filtered manifest population changed')
    from RL.DGPO_neutrino.tau_oracle_mixture import RECO_MC_GEOMETRY
    if (c.get('joint_oracle') or {}).get('geometry') == RECO_MC_GEOMETRY:
        from scripts.verify_tau_mc_p4_inputs import verify_inputs
        verify_inputs([(TRAIN,416701),(VAL,119002)])
    bundle=torch.load(cd/'feature_bundle.pt',map_location='cpu',weights_only=True)
    head=torch.load(cd/'general.pt',map_location='cpu',weights_only=True)
    if (head.get('source_policy_step')!=0 or not head.get('fitting_current_policy')
        or head.get('candidate_dim')!=533 or head.get('condition_dim')!=531
        or head.get('ratio_bound')!=30 or head.get('optimizer_steps',0)<1000
        or any(head.get(k) for k in ('general_spin_angles','cross_attention','physics_branch','pair_interaction'))):
        raise ValueError('Require adequately fitted original step-zero General cap30')
    # Ensure this is precisely the source experiment's pre-update General.
    original=torch.load(src/'tau_diagnostics/refit-step-00000000/best.pt',map_location='cpu',weights_only=True)
    if set(head['state_dict'])!=set(original['state_dict']) or any(not torch.equal(v,original['state_dict'][k]) for k,v in head['state_dict'].items()):
        raise ValueError('General differs from original step-zero classifier')
    model=build_classifier(head);model.load_state_dict(head['state_dict'],strict=True)
    for key in ('packing_spec','relative_preprocessing','policy_conditioning_contract'):
        if head[key]!=bundle['head'][key]:raise ValueError('Feature bundle mismatch: '+key)
    payload=torch.load(bundle['source_checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    if payload.get('dgpo_checkpoint_version') or (payload['epoch'],payload['global_step'])!=(286,18655):
        raise ValueError('Require raw supervised epoch287')
    bundle['head']=head
    ids={}
    for name,n in [('train',416701),('validation',119002)]:
        with np.load(src/(name+'_panel.npz'),allow_pickle=False) as z:
            ids[name]=z['source_ids']
            if len(ids[name])!=n or len(np.unique(ids[name]))!=n or z['truth_deltas'].shape!=(n,2,2):raise ValueError('Invalid panel')
            if not np.isfinite(z['truth_deltas']).all() or not np.all(z['event_weight']==z['event_weight'][0]) or z['event_weight'][0]<=0:raise ValueError('Require finite truth and equal positive event mass')
    if np.intersect1d(ids['train'],ids['validation']).size:raise ValueError('Train/validation overlap')
    print(f'SOURCE OK: strict General head, epoch287 raw weights, verified filtered panels,{c["workers"]}GPUs; no training',flush=True)
    return source,bundle


def update_cadence(c):
    """Explicit authorized10->30 or30->50 migration, preserving all model/optimizer files.

    Running workers keep their in-memory configuration until user restarts them.
    A saved transaction allows retries after an interrupted metadata write.
    """
    out=Path(c['output'])
    if not (out/'PREPARED').exists():
        raise ValueError('Prepare a new experiment first; cadence migration is for an existing run')
    source=yaml.safe_load((Path(c['source_directory'])/'runtime.yaml').read_text())
    expected=assemble(source,c)
    saved=json.loads((out/'settings.json').read_text())
    actual=yaml.safe_load((out/'runtime.yaml').read_text())
    if saved==c and actual==expected:
        print(f'CADENCE ALREADY UPDATED:{c["refit_every_epochs"]*10} steps since last refit; val50 steps; resume required for running workers')
        return out/'runtime.yaml'
    target=int(c['refit_every_epochs'])
    previous=10 if target==30 else 30
    transaction=out/f'refit{previous}-to-refit{target}.json'
    if transaction.exists():
        record=json.loads(transaction.read_text())
        old=record['previous_settings'];oldruntime=record['previous_runtime']
        if record['new_settings']!=c or oldruntime!=assemble(source,old):
            raise ValueError('Cadence transaction differs from expected experiment')
    else:
        old=saved;oldruntime=actual
    validate_cadence_change(old,c)
    if oldruntime!=assemble(source,old) or saved not in (old,c) or actual not in (oldruntime,expected):
        raise ValueError('Only the requested refit cadence may change; other runtime settings differ')
    if not transaction.exists():
        record=dict(previous_settings=old,previous_runtime=oldruntime,new_settings=c,
            scope=f'Only{previous}->{target} logical epochs since last refit; no checkpoint, optimizer, General head or oracle mixture changes')
        tmp=transaction.with_suffix('.pending');tmp.write_text(json.dumps(record,indent=2));tmp.replace(transaction)
    for name,text in [('runtime.yaml',yaml.safe_dump(expected,sort_keys=False)),('settings.json',json.dumps(c,indent=2))]:
        tmp=out/(name+'.pending');tmp.write_text(text);tmp.replace(out/name)
    print(f'CADENCE UPDATED:{target*10} DGPO steps since last installed General; validation50 steps; checkpoints untouched. Restart/resume to apply.',flush=True)
    return out/'runtime.yaml'


def validate_cadence_change(old,new):
    expected=copy.deepcopy(old)
    transition=(old.get('refit_every_epochs'),new.get('refit_every_epochs'))
    if transition==(10,30) and not old.get('refit_relative_to_install',False):
        expected.update(refit_every_epochs=30,refit_relative_to_install=True)
    elif transition==(30,50) and old.get('refit_relative_to_install',False):
        expected.update(refit_every_epochs=50)
    else:
        raise ValueError('Only absolute10->relative30 or relative30->relative50 cadence migration is supported')
    if new!=expected:
        raise ValueError('Only refit cadence change is authorized in this migration')


def update_fit(c):
    """Migrate only the classifier budget; no process or checkpoint changes."""
    out=Path(c['output'])
    if not (out/'PREPARED').exists():raise ValueError('Prepare the experiment first')
    source=yaml.safe_load((Path(c['source_directory'])/'runtime.yaml').read_text())
    saved=json.loads((out/'settings.json').read_text())
    actual=yaml.safe_load((out/'runtime.yaml').read_text())
    expected=assemble(source,c)
    cosine = c['classifier_fit'] == COSINE_FIT
    adaptive = c['classifier_fit'] in (ADAPTIVE_FIT, COSINE_FIT)
    transaction=out/('classifier-fit-cosine1500.json' if cosine else
        'classifier-fit-adaptive-resume.json' if adaptive else 'classifier-fit-to-plateau.json')
    if saved==c and actual==expected:
        print('FIT ALREADY UPDATED: validation plateau required; running workers need resume',flush=True)
        return out/'runtime.yaml'
    if transaction.exists():
        record=json.loads(transaction.read_text())
        old=record['previous_settings'];oldruntime=record['previous_runtime']
        if record['new_settings']!=c:raise ValueError('Fit migration differs')
    else:
        old=saved;oldruntime=actual
    proposed=copy.deepcopy(old);proposed['classifier_fit']=c['classifier_fit']
    allowed_old = (old.get('classifier_fit') in (LEGACY_FIT, ADAPTIVE_FIT, COSINE1000_FIT) if cosine else
        old.get('classifier_fit') == LEGACY_FIT if adaptive else 'classifier_fit' not in old)
    if not allowed_old or proposed!=c or oldruntime!=assemble(source,old):
        raise ValueError('Only the declared classifier fit extension may change')
    if saved not in (old,c) or actual not in (oldruntime,expected):raise ValueError('Unexpected prepared runtime')
    if not transaction.exists():
        if adaptive:
            # Preserve the legacy best-only fit before its explicit weights warm start.
            for folder in (out/'tau_diagnostics').glob('refit-step-*'):
                for name in ('best.pt','best.json'):
                    src=folder/name; dst=folder/('before-adaptive-'+name)
                    if src.exists() and not dst.exists():shutil.copy2(src,dst)
        record=dict(previous_settings=old,previous_runtime=oldruntime,new_settings=c)
        tmp=transaction.with_suffix('.pending');tmp.write_text(json.dumps(record,indent=2));tmp.replace(transaction)
    for name,text in [('runtime.yaml',yaml.safe_dump(expected,sort_keys=False)),('settings.json',json.dumps(c,indent=2))]:
        tmp=out/(name+'.pending');tmp.write_text(text);tmp.replace(out/name)
    print('FIT UPDATED: '+ ('max3000, batch1024/GPU, cosine decay1500, full fit checkpoints; legacy best weights warm start with NEW optimizer.' if cosine else
        'max3000, batch1024/GPU, validation-driven LR, full fit checkpoints; legacy best weights warm start with NEW optimizer.' if adaptive else
        'max1000, original250-epoch LR decay then min_lr.')+' Patience25, weight_decay0.005/dropout0.1; require plateau before install. Training not started.',flush=True)
    return out/'runtime.yaml'


def prepare(c):
    source,bundle=verify(c);out=Path(c['output']);runtime=assemble(source,c)
    if (out/'PREPARED').exists():
        if json.loads((out/'settings.json').read_text())!=c or yaml.safe_load((out/'runtime.yaml').read_text())!=runtime:
            raise ValueError('Prepared settings differ; use the applicable explicit update-cadence, update-fit, update-workers or update-resources migration')
        if c.get('continuation_checkpoint'):
            from scripts.tau_oracle_dgpo_monitor import inherit_monitor_protocol
            _,baseline=continuation_state(c)
            inherit_monitor_protocol(baseline.parent,out/'tau_diagnostics')
        return out/'runtime.yaml'
    if out.exists() and any(out.iterdir()):raise ValueError('Partial output exists; preserve and choose a new run number')
    continuation=continuation_state(c)
    if continuation:
        # Full-state reward restoration checks immutable provenance paths too.
        bundle=torch.load(continuation[0].parent.parent/'initial_tau_reward.pt',map_location='cpu',weights_only=True)
    out.mkdir(parents=True)
    (out/'settings.json').write_text(json.dumps(c,indent=2))
    for key,name in [('source_checkpoint','source.ckpt'),('normalization_file','normalization.pt')]:
        shutil.copy2(bundle[key],out/name)
        if not continuation:bundle[key]=str(out/name)
    feature=yaml.safe_load(Path(bundle['backbone_runtime']).read_text())
    feature['options']['Dataset']['normalization_file']=str(out/'normalization.pt')
    (out/'feature_runtime.yaml').write_text(yaml.safe_dump(feature,sort_keys=False))
    if not continuation:bundle['backbone_runtime']=str(out/'feature_runtime.yaml')
    torch.save(bundle,out/'initial_tau_reward.pt')
    # Panels remain immutable shared inputs; avoid duplicating their large features.
    for name in ('train','validation'):
        (out/(name+'_panel.npz')).symlink_to(Path(c['source_directory'])/(name+'_panel.npz'))
    if continuation:
        checkpoint,baseline=continuation
        dest=out/'checkpoints';dest.mkdir()
        shutil.copy2(checkpoint,dest/checkpoint.name)
        (dest/'last.ckpt').symlink_to(checkpoint.name)
        from scripts.tau_oracle_dgpo_monitor import inherit_monitor_protocol
        inherit_monitor_protocol(baseline.parent,out/'tau_diagnostics')
        (out/'CONTINUATION.json').write_text(json.dumps(dict(source_checkpoint=str(checkpoint),
            global_step=290,next_epoch=29,mode='full_state_new_run',baseline=str(baseline)),indent=2))
    path=out/'runtime.yaml';path.write_text(yaml.safe_dump(runtime,sort_keys=False))
    (out/'PREPARED').write_text(f'General +{100*c["oracle_fraction"]:g}% oracle; no training started\n')
    return path


def preflight(path):
    validate_runtime_workers(yaml.safe_load(Path(path).read_text()))
    # Generic strict actor/reference/reward/optimizer check also works for General.
    from scripts.train_tau_dgpo_spin_angles import preflight as strict_preflight
    strict_preflight(path)
    from evenet.control.global_config import global_config as cfg
    from RL.DGPO_neutrino.model_utils import resolve_dgpo_auto_resume_checkpoint
    from RL.DGPO_neutrino.tau_oracle_mixture import validate_resume_contract
    resumed=resolve_dgpo_auto_resume_checkpoint(cfg.options.Training.model_checkpoint_save_path,enabled=True)
    if resumed:
        validate_resume_contract(cfg.dgpo['oracle_mixture'],torch.load(resumed,map_location='cpu',weights_only=False,mmap=True))
    from RL.DGPO_neutrino.conditional_tau_reward import load_reward
    from RL.DGPO_neutrino.conditional_tau_cycle import panel_batch
    from RL.DGPO_neutrino.tau_oracle_mixture import training_reward, geometry_options
    reward=load_reward(cfg.reward_config.conditional_tau.bundle_file,torch.device('cpu'))
    from RL.DGPO_neutrino.tau_oracle_mixture import RECO_MC_GEOMETRY
    if cfg.dgpo['oracle_mixture']['geometry'] == RECO_MC_GEOMETRY:
        # Panels have no actual MC p4: exercise the native parquet loader instead.
        from scripts.train_diffusion_film_10pct import first_batch
        from RL.DGPO_neutrino.tau_process_labels import canonical_policy_batch
        batch=canonical_policy_batch(first_batch(cfg,count=4))
    else:
        with np.load(cfg.dgpo['tau_ratio']['validation_panel'],allow_pickle=False) as z:
            panel={k:z[k] for k in z.files}
        idx=np.arange(4);batch=panel_batch(panel,idx,reward.spec,torch.device('cpu'))
        # panel_batch supplies observed inputs; training loader separately provides physical targets.
        batch['x_invisible']=torch.tensor(panel['truth_deltas'][idx])
        batch['x_invisible_mask']=torch.ones(4,2,1)
    candidates=batch['x_invisible'].unsqueeze(0).repeat(8,1,1,1)
    with torch.no_grad():
        score=reward.compute(candidates,batch)
        mixed,metrics=training_reward(score,candidates,batch,float(cfg.dgpo['oracle_mixture']['alpha']),
                                     **geometry_options(cfg.dgpo['oracle_mixture']))
    if not torch.isfinite(mixed).all():raise ValueError('Nonfinite mixture smoke test')
    print('ORACLE PREFLIGHT OK: physical truth, General scoring and mixture; no Ray/training started',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config')
    p.add_argument('phase',choices=['check','verify','prepare','preflight','update-cadence','update-fit','update-workers','update-resources','train'],default='check',nargs='?')
    a=p.parse_args();c=settings(a.config)
    if a.phase=='check':
        mode='adaptive cap' if c.get('adaptive_oracle') else 'fixed'
        print(f'CONFIG OK: epoch287,General+{mode} {100*c["oracle_fraction"]:g}% oracle,K8,seed42,{c["workers"]}GPUs,refit{c["refit_every_epochs"]*10} steps since last install,val50 steps; no training');return
    if a.phase=='verify':verify(c);return
    if a.phase=='update-cadence':update_cadence(c);return
    if a.phase=='update-fit':update_fit(c);return
    if a.phase=='update-workers':update_workers(c);return
    if a.phase=='update-resources':update_resources(c);return
    runtime=prepare(c)
    if a.phase in ('preflight','train'):preflight(runtime)
    if a.phase=='train':
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join((str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo'),env.get('PYTHONPATH','')))
        subprocess.run([sys.executable,'-u',str(ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'),str(runtime),
            '--max-steps',str(c['epochs']*10),'--ray-dir',str(Path(c['output'])/'ray_results')],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
