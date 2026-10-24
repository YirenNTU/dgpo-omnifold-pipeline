"""Prepare or USER-launch tau-DGPO with the exact supervised conditioning.

Preparation is CPU-only. The user's existing16-GPU worker group extracts
truth/generated features and fits the fresh paired classifier before update0.
No allocation, job submission or background training is performed here.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
import numpy as np
import torch
import yaml
from scripts.train_neutrino_backend import deep_update, read_overlay_yaml, read_yaml, absolutize_default_paths
from scripts.train_diffusion_film_10pct import TRAIN, VAL, first_batch
from RL.DGPO_neutrino.tau_process_labels import (
    SUPERVISED_POLICY_CONDITIONING_CONTRACT as CONTRACT, canonical_policy_batch,
)


def validate_settings(cfg):
    dg, p = cfg['dgpo'], cfg['platform']
    s = dg['tau_ratio']
    checks = {
        'supervised process labels': s.get('policy_conditioning_contract') == CONTRACT,
        '16 GPU workers': p['number_of_workers'] == s['workers'] == 16 and p['use_gpu'],
        'verified filtered population': p.get('require_filtered_data') and p['data_parquet_dir'] == TRAIN
            and p['data_parquet_val_dir'] == VAL,
        'full population': cfg['options']['Dataset']['dataset_limit'] == cfg['options']['Dataset']['val_dataset_limit'] == 1.,
        'fresh classifier before updates': s.get('bootstrap_fresh_classifier') is True
            and s['minimum_selected_steps'] >= 1000 and s['fit']['min_steps'] >= 1000,
        'pinned terminal source': s['source_completed_epochs'] == 287 and s['source_global_step'] == 18655,
        'raw weights': not cfg['options']['Training']['EMA']['enable'],
        'same DGPO objective': dg['K'] == 8 and dg['advantage_estimator'] == 'leave_one_out_unscaled'
            and dg['reference_trust']['objective'] == 'velocity_mse' and dg['reference_trust']['coefficient'] == 1.,
        'readable run name': len(cfg['logger']['wandb']['run_name']) < 96,
    }
    if not all(checks.values()):
        raise ValueError('Invalid process-conditioned tau DGPO: '+', '.join(k for k,v in checks.items() if not v))
    if s['fit']['epochs']*int(np.ceil(71401/(16*s['fit']['batch_size']))) < s['minimum_selected_steps']:
        raise ValueError('Audit fit budget cannot meet minimum selected steps')
    return s


POLICY_STOCHASTIC_FIELDS = {
    'GroupedSequentialEmbedding': ('dropout',),
    'GlobalEmbedding': ('dropout',),
    'PET': ('dropout', 'feature_drop', 'drop_probability'),
    'TruthGeneration': ('dropout', 'feature_drop', 'drop_probability'),
}


def deterministic_policy_network(source_network):
    """Keep the source parameter layout and eval function; disable train noise."""
    network = copy.deepcopy(source_network)
    for name, fields in POLICY_STOCHASTIC_FIELDS.items():
        section = network if name == 'TruthGeneration' else network.setdefault('Body', {})
        values = section.setdefault(name, {})
        for field in fields:
            values[field] = 0.
    return network


def align_conditioning_optimizer(cfg):
    """Remove inherited branch LR/schedule entries absent from the source model."""
    network, dg = cfg['network'], cfg['dgpo']
    enabled = {
        'angular_conditioning': network['Body']['PET'].get('visible_angular_fourier', {}).get('enabled', False),
        'visible_conditioning': network.get('VisibleConditioning', {}).get('diffusion_enabled', False),
    }
    absent = {name for name, active in enabled.items() if not active}
    rates = dg.get('conditioning_learning_rates')
    if rates is not None:
        dg['conditioning_learning_rates'] = {k: v for k, v in rates.items() if k not in absent}
    schedule = dg.get('lr_schedule', {})
    if schedule.get('groups') is not None:
        schedule['groups'] = [name for name in schedule['groups'] if name not in absent]


def assemble_runtime(source, overlay, output):
    """Preserve source weights/schema with a deterministic train-mode actor."""
    cfg = deep_update(copy.deepcopy(source), overlay)
    cfg = absolutize_default_paths(cfg, ROOT/'config')
    cfg['network'] = deterministic_policy_network(source['network'])
    if 'event_info' in source:
        cfg['event_info'] = copy.deepcopy(source['event_info'])
    cfg['options']['Dataset'] = copy.deepcopy(source['options']['Dataset'])
    dg = cfg['dgpo']
    dg['require_deterministic_policy'] = True
    dg['adaptive_omnifold'] = {'enabled': False}
    dg.pop('beta_kl', None)
    dg.update(checkpoint_load_mode='weights_only', auto_resume_from_last=True,
        auto_resume_fallback_checkpoint_path=None, auto_resume_best_source_checkpoint_dir=None,
        pinned_classifier_restart=False, best_source_start_new_experiment=False,
        step_zero_architecture_bootstrap=False)
    dg['lr_schedule']['resume_use_config'] = False
    align_conditioning_optimizer(cfg)
    s = dg['tau_ratio']
    s.pop('baseline_directory', None)
    s.update(train_panel=str(output/'train_panel.npz'), validation_panel=str(output/'validation_panel.npz'),
             output=str(output/'tau_diagnostics'))
    cfg['compat'] = dict(backend='dgpo-evenet', repo_root=str(ROOT))
    cfg['rl'] = dict(enabled=True)
    cfg['reward_config'] = dict(type='conditional_tau', weight=1.,
        conditional_tau=dict(bundle_file=str(output/'initial_tau_reward.pt'), feature_batch_size=256))
    cfg['options']['Training'].update(pretrain_model_load_path=None,
        model_checkpoint_load_path=str(output/'source.ckpt'), model_checkpoint_save_path=str(output/'checkpoints'))
    # The pinned source dataset normalizer is intentionally copied above.
    cfg['logger']['local'].update(save_dir=str(output/'logs'), name='tau-dgpo-process-labels-film')
    cfg['logger']['wandb'].update(id=None, resume='never', fresh_run=True)
    cfg['nersc']['ray']['results_dir'] = str(output/'ray_results')
    cfg['nersc']['submit'] = False
    cfg['experiment'] = dict(protocol='tau-dgpo-supervised-conditioning',
        source_run=s['source_run'], source_completed_epochs=287, source_global_step=18655,
        policy_conditioning_contract=CONTRACT,
        architecture='supervised source parameter layout; actor training dropout and stochastic depth disabled',
        classifier='fresh paired BCE; visible conditions and learned process-label projection added to FiLM context; frozen source FiLM trunk',
        first_update='after >=1000 selected classifier optimizer steps',
        optimization='unchanged native DGPO leave-one-out; velocity MSE coefficient1',
        physics_scope='raw matched Cij and tau topology; not complete spin/entanglement closure')
    return cfg


def preflight_runtime(path):
    """CPU startup validation before creating any Ray workers or policy updates."""
    from evenet.control.global_config import global_config as cfg
    from RL.DGPO_neutrino.model_utils import (
        build_evenet_on_device, load_normalization_dict, assert_dgpo_neutrino_policy_deterministic,
        apply_component_freezes,
    )
    from RL.DGPO_neutrino.dgpo_trainer import build_optimizer
    from scripts.evaluate_diffusion_pretrain_cij import strict_load
    from RL.DGPO_neutrino.conditional_tau_reward import validate_head
    from scripts.train_conditional_spin_ratio import build_classifier
    cfg.load_yaml(path)
    cfg.load_yaml(path)  # A Ray worker reloads an already populated config.
    validate_settings(read_yaml(path))
    model = build_evenet_on_device(cfg, load_normalization_dict(cfg), torch.device('cpu'))
    payload = torch.load(cfg.options.Training.model_checkpoint_load_path,
        map_location='cpu', weights_only=False, mmap=True)
    trainable = {name: p.requires_grad for name, p in model.named_parameters()}
    strict_load(model, payload)  # The evaluation helper freezes every parameter.
    for name, p in model.named_parameters():
        p.requires_grad_(trainable[name])
    apply_component_freezes(model, cfg)
    model.train()
    assert_dgpo_neutrino_policy_deterministic(model)
    # Use the same production factory and step counts as the 16-GPU workers.
    dg = cfg.dgpo
    steps_per_epoch = dg.get('steps_per_epoch')
    if steps_per_epoch is None:
        steps_per_epoch = max(1, math.ceil(416701 / (cfg.platform.batch_size * cfg.platform.number_of_workers)))
    steps_per_epoch = int(steps_per_epoch)
    warmup_steps = max(1, math.ceil(float(cfg.options.Training.get('learning_rate_warm_up_factor', 1.)) * steps_per_epoch))
    optimizer = build_optimizer(model, steps_per_epoch=steps_per_epoch, warmup_steps=warmup_steps,
        lr_schedule=dg.get('lr_schedule'), conditioning_learning_rates=dg.get('conditioning_learning_rates'))
    grouped = [id(p) for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if not expected or len(grouped) != len(set(grouped)) or set(grouped) != expected:
        raise ValueError('Preflight optimizer must cover every trainable parameter exactly once')
    print('PREFLIGHT optimizer: '+', '.join(
        f"{group['group_name']} (base_lr={group['initial_lr']:g}, parameters={sum(p.numel() for p in group['params'])})"
        for group in optimizer.param_groups), flush=True)
    bundle = torch.load(cfg.reward_config.conditional_tau.bundle_file,
        map_location='cpu', weights_only=True)
    validate_head(bundle['head'])
    if bundle['source_global_step'] != 18655 or bundle['head']['ratio_bound'] != 30:
        raise ValueError('Preflight requires the pinned epoch287 source and reward cap30')
    build_classifier(bundle['head']).load_state_dict(bundle['head']['state_dict'], strict=True)
    print('PREFLIGHT OK: source weights/normalizers, deterministic actor, optimizer/scheduler and parameter coverage, process-conditioned classifier,cap30', flush=True)


def read_panel(directory, spec, expected):
    from scripts.sample_1110_cij import validation_pool
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, SOURCE_ID_COLUMNS
    pool = validation_pool(directory, dict(packing_spec=spec.to_dict()))
    if pool['rows_read'] != expected or pool['invalid_target_rows']:
        raise ValueError('Filtered population changed or has invalid targets; no dropping/fallback')
    ids = source_ids(pool)
    keys = [k for k in SOURCE_ID_COLUMNS if k in pool]
    names = keys+['event_category','event_weight','analyzing_power_a','analyzing_power_b']
    names += [f'lead_{leg}_visible_{key}' for leg in ('a','b') for key in ('E','px','py','pz')]
    table = read_event_table(directory, names)
    cols = {k:np.asarray(table[k].to_numpy()) for k in names}
    order = align(source_ids(cols, keys), ids)
    cols = {k:v[order] for k,v in cols.items()}
    panel = dict(raw_condition=pool['condition'].numpy(), source_ids=ids,
        truth_deltas=pool['truth'].numpy(), category=cols['event_category'],
        event_weight=cols['event_weight'],
        kappas=np.stack([cols['analyzing_power_a'],cols['analyzing_power_b']], -1),
        truth_features_ready=np.array(False))
    for leg in ('a','b'):
        panel['visible_'+leg] = np.stack([cols[f'lead_{leg}_visible_{key}'] for key in ('E','px','py','pz')], -1)
    if (any(not np.isfinite(v).all() for k,v in panel.items() if k != 'source_ids')
            or len(np.unique(ids)) != expected or (panel['event_weight'] < 0).any()
            or panel['event_weight'].sum() <= 0 or (panel['kappas'] == 0).any()):
        raise ValueError('Invalid/duplicate filtered panel inputs')
    return panel


def snapshot_source_runtime(source_file, output):
    """Pin effective architecture, class mapping and normalization contents."""
    source = read_yaml(source_file)
    for name,section in list(source.items()):
        if isinstance(section,dict) and 'default' in section:
            include = (source_file.parent/section['default']).resolve(strict=True)
            source[name] = deep_update(read_yaml(include),{k:v for k,v in section.items() if k != 'default'})
    normalization = Path(source['options']['Dataset']['normalization_file']).resolve(strict=True)
    shutil.copy2(normalization,output/'normalization.pt')
    source['options']['Dataset']['normalization_file'] = str(output/'normalization.pt')
    (output/'source_runtime.yaml').write_text(yaml.safe_dump(source,sort_keys=False))
    return source


def prepare(config):
    from RL.DGPO_neutrino.conditional_tau_reward import KIND, condition_inputs, validate_head
    from evenet.control.global_config import Config
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    from scripts.train_conditional_spin_ratio import event_split, build_classifier
    from scripts.conditional_tau_preprocessing import fit_masked_feature
    from scripts.tau_relative_inputs import relative_angles
    from scripts.diagnose_ztautau_cij import tau_from_deltas
    overlay = read_overlay_yaml(config.resolve())
    s = validate_settings(overlay)
    source_file = Path(s['source_runtime']).resolve(strict=True)
    checkpoint = Path(s['source_checkpoint']).resolve(strict=True)
    source = read_yaml(source_file)
    network = source['network']
    v = network['VisibleConditioning']
    if (network['TruthGeneration']['num_layers'] != 1 or not v['diffusion_enabled']
            or v['diffusion_fourier_enabled'] or network['Body']['PET']['visible_angular_fourier']['enabled']):
        raise ValueError('Source is not the requested one-block supervised FiLM architecture')
    if source['platform']['data_parquet_dir'] != TRAIN or source['platform']['data_parquet_val_dir'] != VAL:
        raise ValueError('Source and new policy filtered populations differ')
    for directory,count in ((TRAIN,416701),(VAL,119002)):
        if validate_filtered_dataset(directory)['rows'] != count:
            raise ValueError('Wrong filtered manifest count')
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if (raw.get('dgpo_checkpoint_version') or raw['epoch']+1 != 287 or raw['global_step'] != 18655
            or not any('visible_conditioning.' in k for k in raw['state_dict'])):
        raise ValueError('Require raw supervised terminal epoch287, not best-val/old DGPO')
    label_weights = [v for k,v in raw['state_dict'].items() if k.endswith('TruthGeneration.label_dense.weight')]
    if len(label_weights) != 1:
        raise ValueError('Missing/ambiguous source process-label embedding')
    num_classes = int(label_weights[0].shape[1])
    del raw, label_weights
    output = Path(s['output_root']).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(source_checkpoint=str(checkpoint), source_runtime=str(source_file), source_run=s['source_run'],
                    completed_epochs=287, source_global_step=18655, policy_conditioning_contract=CONTRACT)
    marker = output/'prepared.json'
    if marker.exists():
        if json.loads(marker.read_text()) != contract:
            raise ValueError('Output belongs to another source; choose a fresh readable output_root')
        for file in ('source.ckpt','source_runtime.yaml','normalization.pt','initial_tau_reward.pt','train_panel.npz','validation_panel.npz'):
            if not (output/file).is_file():
                raise ValueError('Incomplete prepared artifact: '+file)
    else:
        if any((output/file).exists() for file in ('checkpoints','initial_tau_reward.pt','train_panel.npz')):
            raise ValueError('Incomplete preparation in this output; choose a fresh output_root')
        shutil.copy2(checkpoint, output/'source.ckpt')
        source = snapshot_source_runtime(source_file,output)
        resolved = Config(); resolved.load_yaml(output/'source_runtime.yaml')
        normalization = str(resolved.options.Dataset.normalization_file)
        batch = canonical_policy_batch(first_batch(source),num_classes=num_classes)
        spec = EventPackingSpec({k:tuple(batch[k].shape[1:]) for k in
            ('x','x_mask','conditions','conditions_mask','classification')})
        panels = [read_panel(TRAIN,spec,416701),read_panel(VAL,spec,119002)]
        if np.intersect1d(panels[0]['source_ids'],panels[1]['source_ids']).size:
            raise ValueError('Training and external validation source identities overlap')
        panels[0]['split'] = event_split(panels[0]['source_ids'],42)
        panels[1]['split'] = np.full(119002,2,np.uint8)
        # Preserve categorical indicators and fit only continuous conditions.
        dimension = spec.width+16+num_classes
        # Validate IDs, then construct unnormalized packed+one-hot inputs without clipping
        # the physical values used to estimate preprocessing statistics.
        condition_inputs(panels[0]['raw_condition'],panels[0]['category'],
                         np.zeros(dimension),np.ones(dimension),spec.to_dict())
        from scripts.conditional_tau_preprocessing import layout
        spans,_ = layout(spec.to_dict())
        labels = panels[0]['raw_condition'][:,spans['classification'][0]].reshape(-1).astype(int)
        categories = panels[0]['category']
        raw_condition = np.concatenate((panels[0]['raw_condition'],
            np.eye(16,dtype=np.float32)[(categories//10-1)*4+categories%10-1],
            np.eye(num_classes,dtype=np.float32)[labels]),1)
        mean,scale = fit_masked_feature(raw_condition,panels[0]['split']==0,spec.to_dict())
        fit = panels[0]['split']==0
        p = panels[0]
        relative = relative_angles(tau_from_deltas(p['visible_a'][fit],p['truth_deltas'][fit,0]),
            tau_from_deltas(p['visible_b'][fit],p['truth_deltas'][fit,1]),p['visible_a'][fit],p['visible_b'][fit])
        stats = dict(mean=relative.mean(0).tolist(),scale=np.maximum(relative.std(0),1e-6).tolist(),
                     fitted_on='filtered fitting events; truth relative coordinates only; shared mapping for both classes')
        head = dict(condition_dim=dimension,hidden=128,candidate_dim=2*resolved.network.TruthGeneration.hidden_dim+21,
            relative_dim=6,head_kind='film',head_depth=3,condition_hidden=256,condition_width=256,dropout=0.,
            ratio_bound=30,ratio_objective='bce',relative_preprocessing=stats,packing_spec=spec.to_dict(),
            policy_conditioning_contract=CONTRACT,fitting_current_policy=False,
            process_label_conditioning=True,process_label_embedding='one_hot_linear_added_to_film_context',
            num_process_classes=num_classes,
            condition_mean=torch.from_numpy(mean),condition_scale=torch.from_numpy(scale))
        with torch.random.fork_rng():
            torch.manual_seed(42)
            head['state_dict'] = build_classifier(head).state_dict()
        validate_head(head)
        for label,p in zip(('train','validation'),panels):
            p['num_process_classes'] = np.array(num_classes)
            p['condition'] = condition_inputs(p['raw_condition'],p['category'],mean,scale,spec.to_dict())
            np.savez(output/f'{label}_panel.npz',**p)
        torch.save(dict(kind=KIND,schema_version=1,source_run=s['source_run'],
            source_checkpoint=str(output/'source.ckpt'),source_global_step=18655,
            backbone_runtime=str(output/'source_runtime.yaml'),normalization_file=normalization,
            policy_conditioning_contract=CONTRACT,feature_trunk='frozen_supervised_epoch287_film',
            head=head,condition_mean=torch.from_numpy(mean),condition_scale=torch.from_numpy(scale)),
            output/'initial_tau_reward.pt')
        marker.write_text(json.dumps(contract,indent=2)+'\n')
    source = read_yaml(output/'source_runtime.yaml')
    runtime = assemble_runtime(source,overlay,output)
    path = output/'runtime.yaml'
    path.write_text(yaml.safe_dump(runtime,sort_keys=False))
    return path,output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    p.add_argument('phase',choices=('check','prepare','preflight','train'),nargs='?',default='check')
    args = p.parse_args()
    validate_settings(read_overlay_yaml(args.config.resolve()))
    if args.phase == 'check':
        print('CONFIG OK: exact source architecture; supervised labels; fresh FiLM classifier;16 GPUs')
        return
    runtime,output = prepare(args.config)
    print('READY:',runtime,flush=True)
    if args.phase in ('preflight', 'train'):
        preflight_runtime(runtime)
    if args.phase == 'train':
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(ROOT),str(ROOT/'scripts'),str(ROOT/'evenet_dgpo'),env.get('PYTHONPATH','')))
        subprocess.run([sys.executable,'-u',str(ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'),
                        str(runtime),'--ray-dir',str(output/'ray_results')],cwd=ROOT,env=env,check=True)


if __name__ == '__main__':
    main()
