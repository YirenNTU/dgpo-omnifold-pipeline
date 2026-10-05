"""Prepare and USER-launch native DGPO with the saved best conditional tau head.

No allocation/submission, baseline head refit, EMA, or toy upload. `prepare`
only builds a portable reward bundle and verified event panels on shared disk.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]
if __name__ == '__main__':
    print('[DGPO/tau] Loading Python dependencies...', flush=True)
import numpy as np
import torch
import yaml

from scripts.train_neutrino_backend import deep_update, read_overlay_yaml, read_yaml, absolutize_default_paths
from RL.DGPO_neutrino.conditional_tau_reward import (
    KIND, POLICY_CONDITIONING_CONTRACT, condition_inputs, validate_head)


def prepare_panel(directory, spec, wanted, arrays, take, destination):
    print(f'[DGPO/tau prepare] Reading {len(wanted)} events from {directory}', flush=True)
    from scripts.sample_1110_cij import validation_pool
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, SOURCE_ID_COLUMNS
    pool = validation_pool(directory, {'packing_spec': spec})
    if pool['invalid_target_rows'] or pool['rows_read'] != len(wanted):
        raise ValueError('Filtered panel population changed')
    order = align(source_ids(pool), wanted)
    keys = [k for k in SOURCE_ID_COLUMNS if k in pool]
    names = keys+['event_category', 'event_weight', 'analyzing_power_a', 'analyzing_power_b']
    names += [f'lead_{leg}_visible_{k}' for leg in ('a', 'b') for k in ('E', 'px', 'py', 'pz')]
    table = read_event_table(directory, names)
    cols = {k: np.asarray(table[k].to_numpy()) for k in names}
    idx = align(source_ids(cols, keys), wanted)
    cols = {k: v[idx] for k, v in cols.items()}
    raw = pool['condition'][order].numpy()
    c = condition_inputs(raw, cols['event_category'], arrays['condition_mean'], arrays['condition_scale'], spec)
    if not np.allclose(c, arrays['condition'][take], atol=1e-6, rtol=1e-6):
        raise ValueError('Rebuilt condition differs from baseline preprocessing')
    if not np.allclose(cols['event_weight'], arrays['event_weight'][take], atol=1e-7, rtol=1e-6):
        raise ValueError('Event weights differ from classifier training')
    if (not np.isfinite(cols['event_weight']).all() or (cols['event_weight'] < 0).any()
            or cols['event_weight'].sum() <= 0):
        raise ValueError('Paired BCE requires finite nonnegative event weights with positive total')
    payload = dict(raw_condition=raw, condition=c, truth_deltas=pool['truth'][order].numpy(),
        source_ids=wanted, category=cols['event_category'], event_weight=cols['event_weight'],
        split=arrays['split'][take], candidate_truth=arrays['candidate_truth'][take],
        kappas=np.stack([cols['analyzing_power_a'], cols['analyzing_power_b']], -1))
    for leg in ('a', 'b'):
        payload['visible_'+leg] = np.stack([cols[f'lead_{leg}_visible_{k}'] for k in ('E', 'px', 'py', 'pz')], -1)
    np.savez(destination, **payload)
    print(f'[DGPO/tau prepare] Wrote {destination}', flush=True)


def assemble_runtime(base, overlay, source, output, bundle_path):
    """Keep native solver/conditioning settings; replace only the reward path."""
    cfg = deep_update(copy.deepcopy(base), overlay)
    cfg = absolutize_default_paths(cfg, ROOT/'config')
    cfg['compat'] = dict(backend='dgpo-evenet', repo_root=str(ROOT))
    cfg['rl'] = dict(enabled=True)
    dg = cfg['dgpo']
    dg['adaptive_omnifold'] = {'enabled': False}
    dg.pop('beta_kl', None)
    dg.update(checkpoint_load_mode='weights_only', auto_resume_from_last=True,
              auto_resume_fallback_checkpoint_path=None, auto_resume_best_source_checkpoint_dir=None,
              pinned_classifier_restart=False, best_source_start_new_experiment=False,
              step_zero_architecture_bootstrap=False)
    cfg['reward_config'] = dict(type='conditional_tau', weight=1.,
        conditional_tau=dict(bundle_file=str(bundle_path), feature_batch_size=256))
    cfg['options']['Training'].update(pretrain_model_load_path=None,
        model_checkpoint_load_path=source['backbone_manifest']['checkpoint'],
        model_checkpoint_save_path=str(output/'checkpoints'))
    cfg['options']['Dataset']['normalization_file'] = source['normalization_file']
    dg['tau_ratio']['train_panel'] = str(output/'train_panel.npz')
    dg['tau_ratio']['validation_panel'] = str(output/'validation_panel.npz')
    dg['tau_ratio']['output'] = str(output/'tau_diagnostics')
    dg['lr_schedule']['resume_use_config'] = False
    cfg['logger']['local'].update(save_dir=str(output/'logs'), name='dgpo-conditional-tau')
    cfg['logger']['wandb'].update(id=None, resume='never', fresh_run=True)
    cfg['nersc']['ray']['results_dir'] = str(output/'ray_results')
    cfg['nersc']['reproducibility'] = dict(source_checkpoint=source['backbone_manifest']['checkpoint'],
        note='Raw1110 weights-only first launch; subsequent launches fully resume this experiment, including best-val tau reward and optimizer.')
    cfg['nersc']['execution']['command'] = 'shifter python3 -u scripts/train_dgpo_tau_ratio.py config/dgpo_tau_ratio_1110.yaml'
    cfg['experiment'] = dict(protocol='conditional-tau-ratio-to-native-dgpo',
        policy_conditioning_contract=POLICY_CONDITIONING_CONTRACT,
        source_run='zrv2yfgt', source_policy_step=1110, weights='raw_state_dict_only',
        first_head='inherited best-val; no bootstrap refit', primary='unweighted generated Cij vs matched step0',
        secondary='fresh unbounded audit held-out BCE/AUC; within-round reward and ESS',
        refit='fresh paired BCE against new current-policy K1; replaces old ratio; best-val; bound30',
        physics_scope='same tau_from_deltas fixed-energy reconstruction and TT2L convention; not full tau-energy unfolding')
    cfg['experiment']['policy_fourier'] = dict(
        pet=cfg['network']['Body']['PET']['visible_angular_fourier'].get('fourier_enabled', True),
        film=cfg['network']['VisibleConditioning'].get('diffusion_fourier_enabled', True),
        ablation='zero Fourier channels only; FiLM, raw numeric inputs, reward and timestep embeddings unchanged')
    return cfg


def validate_settings(cfg):
    s = cfg['dgpo']['tau_ratio']
    if s.get('policy_conditioning_contract') != POLICY_CONDITIONING_CONTRACT:
        raise ValueError('Tau DGPO requires the explicit saved-denominator label0 conditioning contract')
    if s['workers'] != 16 or cfg['platform']['number_of_workers'] != 16 or not cfg['platform']['use_gpu']:
        raise ValueError('This experiment requires 16 GPU workers')
    if s['fit']['batch_size'] != 1024:
        raise ValueError('Preserve the selected classifier batch: 1024 paired events/GPU')
    for key in ('validation_candidates', 'generation_batch_size', 'refit_every_epochs', 'validation_every_epochs'):
        if int(s[key]) < 1:
            raise ValueError('Invalid tau cycle setting: '+key)
    if int(s['fit']['epochs']) * int(np.ceil(71401/(16*1024))) < s['fit']['min_steps']:
        raise ValueError('Audit epoch budget cannot meet the minimum classifier steps')
    for key in ('train_events', 'test_events'):
        if s.get(key):
            raise ValueError('Use the baseline manifest populations, not panel overrides')
    if cfg['dgpo']['reference_trust'] != dict(enabled=True, coefficient=1., objective='velocity_mse', adaptive_boundary={'enabled': False}):
        raise ValueError('Preserve the coefficient-1 soft velocity-MSE reference')
    if cfg['options']['Training']['EMA'].get('enable'):
        raise ValueError('Raw policy only; EMA must stay disabled')
    return s


def prepare(overlay_path):
    from evenet.dataset.filtered_data import validate_filtered_dataset
    from evenet.control.global_config import Config
    cfg = read_overlay_yaml(overlay_path.resolve())
    settings = validate_settings(cfg)
    source_dir, output = Path(settings['baseline_directory']), Path(settings['output_root'])
    print(f'[DGPO/tau prepare] Checking saved classifier and filtered inputs: {source_dir}', flush=True)
    manifest = json.loads((source_dir/'manifest.json').read_text())
    if json.loads((source_dir/'wandb.json').read_text())['id'] != 'zrv2yfgt':
        raise ValueError('Expected completed context256 baseline zrv2yfgt')
    head = torch.load(source_dir/'best.pt', map_location='cpu', weights_only=True)
    validate_head(head)
    if manifest['condition_normalization'] != 'masked_feature':
        raise ValueError('Unsupported baseline condition normalization')
    for key, count in (('train_events', 416701), ('test_events', 119002)):
        if validate_filtered_dataset(manifest[key])['rows'] != count:
            raise ValueError('Unexpected filtered event count')
    if cfg['platform']['data_parquet_dir'] != manifest['train_events'] or cfg['platform']['data_parquet_val_dir'] != manifest['test_events']:
        raise ValueError('Policy and classifier filtered populations differ')
    backbone = manifest['backbone_manifest']
    if backbone['weights'] != 'raw_state_dict_only' or backbone['global_step'] != 1110:
        raise ValueError('Expected frozen raw step1110 backbone')
    original = Config(); original.load_yaml(backbone['runtime'])
    manifest['normalization_file'] = str(original.options.Dataset.normalization_file)
    source = torch.load(str(backbone['checkpoint']), map_location='cpu', weights_only=False, mmap=True)
    if source['global_step'] != 1110:
        raise ValueError('Initial ratio denominator is not raw1110')
    del source
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output/'initial_tau_reward.pt'
    contract = dict(source=str(source_dir.resolve()), checkpoint=str(Path(backbone['checkpoint']).resolve()),
                    normalization_file=manifest['normalization_file'],
                    best_epoch=int(head['epoch']), best_steps=int(head['optimizer_steps']), version=3,
                    policy_conditioning_contract=POLICY_CONDITIONING_CONTRACT)
    marker = output/'prepared.json'
    if marker.is_file():
        if json.loads(marker.read_text()) != contract:
            raise ValueError('Output already belongs to a different reward; choose another output_root')
        for path in (bundle_path, output/'train_panel.npz', output/'validation_panel.npz', output/'replay.npz'):
            if not path.is_file():
                raise ValueError('Prepared artifact missing: '+str(path))
    else:
        print('[DGPO/tau prepare] Loading saved feature arrays; GPU workers start after preparation.', flush=True)
        with np.load(source_dir/'prepared.npz', allow_pickle=False) as f:
            a = {k: f[k] for k in ('source_ids', 'condition', 'condition_mean', 'condition_scale',
                  'candidate_truth', 'candidate_generated', 'event_weight', 'split')}
        if len(np.unique(a['source_ids'])) != 535703 or tuple(np.bincount(a['split'])) != (354488, 62213, 119002):
            raise ValueError('Classifier split/identity contract changed')
        for key in ('condition_mean', 'condition_scale'):
            if not torch.equal(torch.as_tensor(head[key]), torch.as_tensor(a[key])):
                raise ValueError('Saved head and cached condition preprocessing differ: '+key)
        bundle = dict(kind=KIND, schema_version=1, source_run='zrv2yfgt',
            source_checkpoint=str(Path(backbone['checkpoint']).resolve()), backbone_runtime=backbone['runtime'],
            head=head, condition_mean=torch.from_numpy(a['condition_mean']), condition_scale=torch.from_numpy(a['condition_scale']),
            normalization_file=manifest['normalization_file'], source_directory=str(source_dir), replay=str(output/'replay.npz'))
        for label, key, take in [('train', 'train_events', a['split'] != 2), ('validation', 'test_events', a['split'] == 2)]:
            prepare_panel(manifest[key], head['packing_spec'], a['source_ids'][take], a, take, output/f'{label}_panel.npz')
        # Saved K1 reference: each worker will verify its own feature AND logit replay.
        with np.load(source_dir/'test_scores.npz', allow_pickle=False) as scores, np.load(Path(manifest['test_source'])/'candidates.npz', allow_pickle=False) as samples:
            from scripts.diagnose_ztautau_cij import source_ids, align, SOURCE_ID_COLUMNS
            wanted = a['source_ids'][a['split'] == 2]
            if not np.array_equal(scores['source_ids'], wanted):
                raise ValueError('Saved head scores do not match external identities')
            ids = source_ids({k: samples[k] for k in SOURCE_ID_COLUMNS if k in samples})
            idx = align(ids, wanted)
            np.savez(output/'replay.npz', deltas=samples['deltas'][idx[:1024], 0],
                features=a['candidate_generated'][a['split'] == 2][:1024],
                logits=scores['generated_logits'][:1024])
        torch.save(bundle, bundle_path)
        marker.write_text(json.dumps(contract, indent=2)+'\n')
    base = read_yaml(ROOT/'config/train_diffusion_nersc.yaml')
    runtime = assemble_runtime(base, cfg, manifest, output, bundle_path)
    runtime['nersc']['execution']['command'] = f'shifter python3 -u scripts/train_dgpo_tau_ratio.py {overlay_path}'
    path = output/'runtime.yaml'
    path.write_text(yaml.safe_dump(runtime, sort_keys=False))
    print('READY:', dict(runtime=str(path), raw_policy=backbone['checkpoint'], classifier=str(source_dir/'best.pt'),
        workers=16, classifier_batch_per_gpu=1024, classifier_bootstrap_fits=0,
        automatic_resume=str(output/'checkpoints/last.ckpt'), refit_every_epochs=settings['refit_every_epochs']), flush=True)
    return path, output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('phase', choices=('prepare', 'train'), nargs='?', default='train')
    args = p.parse_args()
    runtime, output = prepare(args.config)
    if args.phase == 'train':
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo'), env.get('PYTHONPATH', '')))
        subprocess.run([sys.executable, '-u', str(ROOT/'evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py'),
                        str(runtime), '--ray-dir', str(output/'ray_results')], cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
