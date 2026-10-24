#!/usr/bin/env python3
"""Prepare, CPU-check or USER-run foundation diffusion with visible FiLM.

Never submits jobs or requests an allocation. Default is the FiLM-on pilot;
an optional off arm prepares a future matched control with separate outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'evenet_dgpo')]
from train_neutrino_backend import absolutize_default_paths, deep_update, read_overlay_yaml, read_yaml

CONFIG = ROOT / 'config/train_diffusion_film_10pct_foundation.yaml'
BASE = ROOT / 'config/train_diffusion_nersc.yaml'
TRAIN = '/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train'
VAL = '/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val'


def create_run_directory(root, film):
    """Reserve a readable sequence name atomically, without overwriting runs."""
    if film not in ('on', 'off'):
        raise ValueError('Select FiLM on or off')
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        output = root / f'film-{film}-run-{index:02d}'
        try:
            output.mkdir()
        except FileExistsError:
            index += 1
        else:
            return output


def configuration(film='on'):
    cfg = deep_update(read_yaml(BASE), read_overlay_yaml(CONFIG))
    cfg = absolutize_default_paths(cfg, BASE.parent)
    cfg['rl'] = dict(enabled=False)
    cfg['compat'] = dict(backend='pure-evenet', repo_root=str(ROOT))
    if film not in ('on', 'off'):
        raise ValueError('Select FiLM on or off')
    if film == 'off':
        cfg['network']['VisibleConditioning']['diffusion_enabled'] = False
        cfg['logger']['wandb'].update(run_name='Does FiLM improve spin closure? | diffusion 10% | matched FiLM off')
        cfg['logger']['wandb']['tags'] = [t for t in cfg['logger']['wandb']['tags'] if t != 'GlobalFiLM'] + ['FiLMOff']
    cfg['experiment']['film'] = film
    return cfg


def validate_config(cfg):
    p, t = cfg['platform'], cfg['options']['Training']
    v = cfg['network']['VisibleConditioning']
    d = cfg['options']['Dataset']
    conditions = {
        '16-GPU filtered population': p['number_of_workers'] == 16 and p['use_gpu'] and p['require_filtered_data']
            and p['data_parquet_dir'] == TRAIN and p['data_parquet_val_dir'] == VAL,
        'no second subset': d['dataset_limit'] == d['val_dataset_limit'] == 1.0,
        'foundation weights only': t['pretrain_model_load_path'] is not None and t['model_checkpoint_load_path'] is None,
        'original depth and batch': cfg['network']['TruthGeneration']['num_layers'] == 1 and p['batch_size'] == 2048,
        'no Fourier or classifier conditioning': not v['diffusion_fourier_enabled'] and not v['classifier_enabled']
            and not cfg['network']['Body']['PET']['visible_angular_fourier']['enabled'],
        'supervised raw objective': not cfg['rl']['enabled'] and not t['EMA']['enable']
            and t['Components']['TruthGeneration']['low_noise_weight'] == 1.0
            and t['Components']['TruthGeneration']['low_noise_cutoff'] == 0.0,
        'original budget and full-panel physics': t['epochs'] == t['total_epochs'] == 1500
            and t['TauPhysics']['events'] == 119002 and t['TauPhysics']['K'] == 8,
        'paired batches and noise': p['ordered_data'] and type(t['paired_diffusion_seed']) is int,
        'fresh readable W&B run': cfg['logger']['wandb']['id'] is None and len(cfg['logger']['wandb']['run_name']) < 96,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if failed:
        raise ValueError('Invalid foundation experiment: ' + ', '.join(failed))


def first_batch(cfg, count=2):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from evenet.dataset.preprocess import unflatten_dict
    metadata = json.loads((Path(TRAIN) / 'shape_metadata.json').read_text())
    for file in sorted(Path(VAL).glob('*.parquet')):
        rb = next(pq.ParquetFile(file).iter_batches(batch_size=count), None)
        if rb is not None and len(rb):
            break
    else:
        raise ValueError('Empty filtered validation dataset')
    flat = {name: rb.column(i).to_numpy(zero_copy_only=False) for i, name in enumerate(rb.schema.names)}
    arrays = unflatten_dict(flat, metadata,
        drop_column_prefix=['EXTRA/', 'regression-', 'assignments-', 'segmentation-'])
    return {k: torch.as_tensor(np.array(v, copy=True)) for k, v in arrays.items()}


def validate_inputs(cfg):
    from evenet.dataset.filtered_data import validate_filtered_dataset
    inputs = {name: validate_filtered_dataset(cfg['platform'][key]) for name, key in
              [('train', 'data_parquet_dir'), ('validation', 'data_parquet_val_dir')]}
    if [inputs[k]['rows'] for k in ('train', 'validation')] != [416701, 119002]:
        raise ValueError('Unexpected filtered populations')
    return inputs


def prepare_panel(cfg, output):
    import numpy as np
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    from scripts.sample_1110_cij import validation_pool
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, SOURCE_ID_COLUMNS
    batch = first_batch(cfg)
    # Keep exactly the labels consumed by supervised training; no label0 override.
    spec = EventPackingSpec({k: tuple(batch[k].shape[1:])
        for k in ('x', 'x_mask', 'conditions', 'conditions_mask', 'classification')})
    if json.loads((Path(TRAIN) / 'shape_metadata.json').read_text()) != json.loads((Path(VAL) / 'shape_metadata.json').read_text()):
        raise ValueError('Training/validation packing schemas differ')
    pool = validation_pool(VAL, dict(packing_spec=spec.to_dict()))
    if pool['rows_read'] != 119002 or pool['invalid_target_rows']:
        raise ValueError('Full filtered validation population is required; no dropped targets')
    identities = source_ids(pool)
    keys = [k for k in SOURCE_ID_COLUMNS if k in pool]
    names = keys + ['event_category', 'event_weight', 'analyzing_power_a', 'analyzing_power_b']
    names += [f'lead_{leg}_visible_{key}' for leg in ('a', 'b') for key in ('E', 'px', 'py', 'pz')]
    table = read_event_table(VAL, names)
    columns = {k: np.asarray(table[k].to_numpy()) for k in names}
    order = align(source_ids(columns, keys), identities)
    columns = {k: v[order] for k, v in columns.items()}
    panel = dict(source_ids=identities, raw_condition=pool['condition'].numpy(),
        truth_deltas=pool['truth'].numpy(), event_weight=columns['event_weight'],
        category=columns['event_category'], kappas=np.stack([columns['analyzing_power_a'], columns['analyzing_power_b']], -1))
    for leg in ('a', 'b'):
        panel['visible_' + leg] = np.stack([columns[f'lead_{leg}_visible_{k}'] for k in ('E', 'px', 'py', 'pz')], -1)
    if (any(not np.isfinite(v).all() for k, v in panel.items() if k != 'source_ids')
            or (panel['event_weight'] < 0).any() or panel['event_weight'].sum() <= 0
            or (panel['kappas'] == 0).any()):
        raise ValueError('Invalid physics inputs; no silent event removal')
    np.savez(output / 'validation_panel.npz', **panel)
    metadata = dict(events=119002, validation_directory=VAL, packing_spec=spec.to_dict(),
                    label_contract='exact_supervised_parquet_classification', normalization='unchanged')
    (output / 'validation_panel.json').write_text(json.dumps(metadata, indent=2) + '\n')


def prepare(film, output_root):
    import torch
    cfg = configuration(film)
    validate_config(cfg)
    inputs = validate_inputs(cfg)
    source = Path(cfg['options']['Training']['pretrain_model_load_path']).resolve(strict=True)
    normalization = Path(cfg['options']['Dataset']['normalization_file']).resolve(strict=True)
    payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('dgpo_checkpoint_version') or any(k.startswith('dgpo_') for k in payload):
        raise ValueError('Use the original foundation before DGPO, not an RL policy')
    if any('visible_conditioning.' in k or 'angular_conditioning.' in k for k in payload['state_dict']):
        raise ValueError('Foundation already contains the intervention')
    output = create_run_directory(output_root, film)
    shutil.copy2(source, output / 'foundation.ckpt')
    shutil.copy2(normalization, output / 'normalization.pt')
    prepare_panel(cfg, output)
    cfg['options']['Training'].update(pretrain_model_load_path=str(output / 'foundation.ckpt'),
        model_checkpoint_save_path=str(output / 'checkpoints'))
    cfg['options']['Training']['TauPhysics'].update(panel_path=str(output / 'validation_panel.npz'),
        panel_metadata=str(output / 'validation_panel.json'))
    cfg['options']['Dataset']['normalization_file'] = str(output / 'normalization.pt')
    cfg['logger']['local'].update(save_dir=str(output / 'logs'))
    cfg['nersc']['ray']['results_dir'] = str(output / 'ray_results')
    cfg['experiment'].update(source_checkpoint=str(source), source_epoch=payload.get('epoch'),
        source_global_step=payload.get('global_step'), source_normalization_file=str(normalization))
    path = output / 'runtime.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest = dict(complete=True, mode='prepared_only', output=str(output), film=film,
        source=str(source), source_epoch=payload.get('epoch'), source_step=payload.get('global_step'),
        inputs=inputs, runtime=str(path), weights='foundation_weights_only',
        normalization='exact_copy_of_existing_Ztautau_normalization', training_started=False)
    (output / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)
    return output


def preflight(output):
    """Two-event real-data CPU forward/backward check; no optimizer or Ray."""
    worker_import_preflight()
    import torch
    from evenet.control.global_config import Config
    from evenet.engine import EveNetEngine
    from evenet.utilities.fourier_integration import paired_diffusion_rng
    cfg = read_yaml(output / 'runtime.yaml')
    batch = first_batch(cfg)
    reference = None
    report = dict(events=len(batch['x']), device='cpu', optimizer_updates=0, arms={},
                  worker_imports='fresh_interpreter_without_scripts_path')
    for film in ('off', 'on'):
        arm = deep_update(cfg, dict(network=dict(VisibleConditioning=dict(diffusion_enabled=film == 'on'))))
        path = output / f'preflight_{film}.yaml'
        path.write_text(yaml.safe_dump(arm, sort_keys=False))
        config = Config(); config.load_yaml(path)
        torch.manual_seed(config.options.Training.seed)
        engine = EveNetEngine(config, world_size=16, total_events=416701, total_val_events=119002)
        engine.configure_model()
        model = engine.model
        optimizers, _ = engine.configure_optimizers()
        owned = [id(p) for opt in optimizers for group in opt.param_groups for p in group['params']]
        # The historical trainer registers inactive FAMO logits even with
        # FAMO disabled. They are outside every loss and optimizer in this arm.
        trainable = {id(p) for name, p in model.named_parameters() if p.requires_grad and not name.startswith('famo.')}
        if len(owned) != len(set(owned)) or set(owned) != trainable:
            missing = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in set(owned)]
            raise ValueError(f'Optimizer ownership must cover each trainable parameter exactly once: '
                f'groups={engine.model_parts}, missing={missing}, duplicates={len(owned)-len(set(owned))}')
        model.eval()
        with paired_diffusion_rng(42, training=False, epoch=0, batch_idx=0, rank=0, device='cpu'):
            result = model.shared_step(batch, len(batch['x']), {}, schedules=[('neutrino_generation', True)])['generations']['neutrino']
        if not all(torch.isfinite(result[k]).all() for k in ('vector', 'truth', 'time')):
            raise FloatingPointError('Nonfinite initial diffusion forward')
        if reference is None:
            reference = {k: result[k].detach().clone() for k in ('vector', 'truth', 'time')}
        else:
            for key, value in reference.items():
                torch.testing.assert_close(result[key], value, rtol=0, atol=0)
        mask = result['mask'].to(result['vector'])
        loss = ((result['vector'] - result['truth']).square() * mask).sum() / (mask.sum() * result['vector'].shape[-1])
        row = dict(initial_velocity_mse=float(loss.detach()), parameters=sum(p.numel() for p in model.parameters()),
            optimizer_parameter_ownership='complete_and_unique', optimizer_groups=list(engine.model_parts))
        if film == 'on':
            loss.backward()
            heads = model.TruthGeneration.visible_conditioning.modulations
            gradients = [p.grad for head in heads for p in head.parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in gradients) or not any(torch.count_nonzero(g) for g in gradients):
                raise ValueError('FiLM modulation has no finite nonzero supervised gradient')
            row['film_gradient_nonzero'] = True
            # Exercise the observer's real packed-label DDIM/Cij interface on
            # two events. This is an engineering check, not a physics evaluation.
            import numpy as np
            from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
            from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
            from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
            from evenet.utilities.joint_coverage import event_noise
            from evenet.utilities.diffusion_sampler import make_replay_sampler
            with np.load(output / 'validation_panel.npz', allow_pickle=False) as f:
                panel = {k: f[k][:2].copy() for k in f.files}
            metadata = json.loads((output / 'validation_panel.json').read_text())
            spec = EventPackingSpec.from_dict(metadata['packing_spec'])
            packed_batch = unpack_event_inputs(torch.as_tensor(panel['raw_condition']), spec)
            packed_batch['classification'] = packed_batch['classification'].long()
            for key in spec.shapes:
                torch.testing.assert_close(packed_batch[key], batch[key].to(packed_batch[key]), rtol=0, atol=0)
            packed_batch['x_invisible'] = torch.zeros(2, 2, 2)
            packed_batch['x_invisible_mask'] = torch.ones(2, 2, dtype=torch.bool)
            noise = event_noise(torch.arange(2), 8, config.options.Training.TauPhysics.seed)
            sampler = make_replay_sampler(noise.permute(1, 0, 2, 3), 'legacy')
            with torch.no_grad():
                draws = generate_neutrino_candidates(model, packed_batch, sampler, K=8,
                    num_ddim_steps=20, device=torch.device('cpu'), parallel_chains=1)
            contributions = np.mean([cij_terms(panel, np.arange(2), draw.numpy()) for draw in draws], axis=0)
            if not torch.isfinite(draws).all() or not np.isfinite(contributions).all():
                raise ValueError('Nonfinite DDIM/physics engineering check')
            row.update(packed_labels_and_inputs_match=True, ddim20_K8_and_cij_finite=True)
        report['arms'][film] = row
        del model, result, loss, engine, optimizers
    report['initial_on_off_equal'] = True
    report['limitation'] = 'CPU engineering check only; distributed training and physics outcome require the user run'
    (output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


WORKER_IMPORT_PROBE = '''
from types import SimpleNamespace
from evenet.utilities.tau_physics_validation import TauPhysicsValidation
cfg = dict(events=119002, K=8, ddim_steps=20, every_n_epochs=50,
           batch_size=512, seed=20261006, bootstrap=300)
callback = TauPhysicsValidation(cfg, '/tmp/film-worker-import-check/checkpoints')
try:
    callback.evaluate(SimpleNamespace(global_rank=0, world_size=1), None, 0)
except ValueError as error:
    assert 'sixteen-GPU group' in str(error), error
else:
    raise AssertionError('Expected the production sixteen-GPU guard')
from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
import numpy as np
panel = dict(visible_a=np.array([[20.,18.,4.,2.]]),
             visible_b=np.array([[20.,-16.,3.,-1.]]), kappas=np.ones((1,2)))
assert np.isfinite(cij_terms(panel, np.arange(1), np.zeros((1,2,2)))).all()
print('Clean worker imports passed without scripts on PYTHONPATH.', flush=True)
'''


def worker_import_preflight():
    """Fresh worker-like interpreter: repo/packages only, no scripts path/cache."""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'evenet_dgpo'), str(ROOT)])
    subprocess.run([sys.executable, '-c', WORKER_IMPORT_PROBE], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('check', 'prepare', 'preflight', 'run'))
    parser.add_argument('--film', choices=('on', 'off'), default='on')
    parser.add_argument('--output-root', type=Path, default=Path('/pscratch/sd/y/yiren/Ztautau/diffusion_film_10pct_foundation'))
    parser.add_argument('--prepared', type=Path, help='Reuse an explicitly prepared fresh invocation directory')
    args = parser.parse_args()
    cfg = configuration(args.film); validate_config(cfg)
    if args.stage == 'check':
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    output = args.prepared.resolve(strict=True) if args.prepared else prepare(args.film, args.output_root)
    manifest = json.loads((output / 'prepared.json').read_text())
    if not manifest['complete'] or manifest['film'] != args.film:
        parser.error('Prepared invocation differs from selected arm')
    runtime = read_yaml(output / 'runtime.yaml')
    validate_config(runtime)
    if runtime['experiment']['film'] != args.film:
        parser.error('Runtime differs from selected arm')
    # Recheck completed filters even when reusing a previously prepared invocation.
    validate_inputs(runtime)
    if args.stage == 'prepare':
        return
    if args.stage == 'preflight':
        preflight(output)
        return
    if (output / 'training_started.json').exists() or (output / 'checkpoints').exists():
        parser.error('This invocation already started; prepare a fresh run or use the standard full-resume entrypoint')
    # Fail before any Ray workers/GPU reservation if callback imports are broken.
    worker_import_preflight()
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'evenet_dgpo'), str(ROOT), env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    env.pop('WANDB_RUN_ID', None); env.pop('WANDB_RESUME', None)
    (output / 'training_started.json').write_text(json.dumps(dict(user_launched=True, film=args.film)) + '\n')
    command = [sys.executable, '-u', str(ROOT / 'evenet_dgpo/evenet/train.py'), str(output / 'runtime.yaml'),
               '--ray_dir', str(output / 'ray_results')]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
