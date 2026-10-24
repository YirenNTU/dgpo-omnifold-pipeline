#!/usr/bin/env python3
"""USER-run historical no-FiLM inference with the supervised FiLM Cij protocol.

Preparation and CPU preflight never connect to Ray, fit a classifier, construct
an optimizer or launch a job. GPU execution requires an existing 16-GPU cluster.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'evenet_dgpo'), str(ROOT), str(ROOT / 'scripts')]
DEFAULT = ROOT / 'config/eval_diffusion_pretrain_no_film.yaml'
VAL = '/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val'
ERROR_KEYS = ('error', 'diagonal_error', 'offdiagonal_error')
AXES = ('k', 'r', 'n')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_spec(spec):
    if (spec['workers'] != 16 or spec['events'] != 119002 or spec['K'] != 8
            or spec['ddim_steps'] != 20 or spec['seed'] != 20261006
            or spec['batch_size'] != 512 or spec['bootstrap'] < 2):
        raise ValueError('Use the existing full-val 16-GPU K8/DDIM20 paired-noise protocol')
    if not spec['checkpoints'] or len(spec['wandb']['name']) >= 96:
        raise ValueError('Need explicit checkpoints and a readable W&B name')
    for name, source in spec['checkpoints'].items():
        if not name.startswith('pretrain-no-film-epoch') or '/' in name:
            raise ValueError('Use readable no-FiLM checkpoint names')
        if not Path(source['path']).is_absolute() or type(source['epoch']) is not int:
            raise ValueError('Pin an absolute named checkpoint and its stored epoch')
        if Path(source['path']).name == 'last.ckpt':
            raise ValueError('Pin the named checkpoint, not a mutable last.ckpt')


def reserve_directory(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        path = root / f'no-film-eval-{index:02d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            index += 1


def expanded_runtime(path):
    """Freeze the source runtime's top-level defaults with Config's merge rules."""
    from train_neutrino_backend import deep_update
    path = Path(path)
    result = yaml.safe_load(path.read_text())
    for section, value in list(result.items()):
        if isinstance(value, dict) and 'default' in value:
            overrides = dict(value)
            default = path.parent / overrides.pop('default')
            result[section] = deep_update(yaml.safe_load(default.read_text()) or {}, overrides)
    return result


def checkpoint_metadata(payload, expected):
    if payload.get('dgpo_checkpoint_version') or any(k.startswith('dgpo_') for k in payload):
        raise ValueError('Expected supervised pretrain, not an RL checkpoint')
    if any(payload.get(k) != expected[k] for k in ('epoch', 'global_step')):
        raise ValueError('Checkpoint epoch/global_step differs from the pinned source')
    keys = payload.get('state_dict', {})
    if not keys or any(any(part in k for part in ('visible_conditioning.', 'angular_conditioning.',
            'token_conditioning.', 'pair_conditioning.')) for k in keys):
        raise ValueError('Checkpoint is missing raw weights or contains conditioning intervention weights')
    return {k: payload[k] for k in ('epoch', 'global_step')}


def strict_load(model, payload):
    """Load every model tensor; retain/check the checkpoint's normalizers."""
    import torch
    source = {k.removeprefix('model.'): v for k, v in payload['state_dict'].items()}
    actual = model.state_dict()
    missing = set(actual) - set(source)
    extra = {k for k in source if k not in actual and not k.startswith('famo.w.')}
    if missing or extra:
        raise ValueError(f'Raw architecture mismatch: missing={sorted(missing)}, extra={sorted(extra)}')
    for key, value in actual.items():
        if 'normalizer.' in key and not torch.equal(value.cpu(), source[key].cpu()):
            raise ValueError('Pinned normalization file disagrees with checkpoint buffer: ' + key)
    model.load_state_dict({k: source[k] for k in actual}, strict=True)
    model.eval().requires_grad_(False)


def load_panel(output):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec
    with np.load(Path(output) / 'validation_panel.npz', allow_pickle=False) as f:
        panel = {k: f[k] for k in f.files}
    meta = json.loads((Path(output) / 'validation_panel.json').read_text())
    if (len(panel['source_ids']) != 119002 or len(np.unique(panel['source_ids'])) != 119002
            or meta['events'] != 119002 or meta['validation_directory'] != VAL
            or meta['label_contract'] != 'exact_supervised_parquet_classification'):
        raise ValueError('Need the complete unique filtered supervised validation panel')
    if any(not np.isfinite(a).all() for k, a in panel.items() if k != 'source_ids'):
        raise ValueError('Nonfinite panel; no silent event removal')
    return panel, EventPackingSpec.from_dict(meta['packing_spec'])


def align_measurements(measurements, panel):
    """Validate identity/order and weights rather than comparing aggregate means."""
    if not np.array_equal(measurements['source_ids'], panel['source_ids']):
        raise ValueError('Reference event identities/order changed')
    if not np.array_equal(measurements['weight'], panel['event_weight']):
        raise ValueError('Reference base event weights changed')
    shape = (len(panel['source_ids']), 8, 2, 2)
    if measurements['deltas'].shape != shape or not np.isfinite(measurements['deltas']).all():
        raise ValueError('Reference requires all eight finite candidates per event')
    if (measurements['generated_cij'].shape != (len(panel['source_ids']), 9)
            or not np.isfinite(measurements['generated_cij']).all()):
        raise ValueError('Reference requires nine finite Cij contributions per event')


def full_truth_contributions(panel):
    from scripts.diagnose_ztautau_cij import read_event_table, source_ids, align, angles, SOURCE_ID_COLUMNS
    from scripts.diagnose_reweighted_cij import features
    import pyarrow.parquet as pq
    names = pq.ParquetFile(next(Path(VAL).glob('*.parquet'))).schema_arrow.names
    ids = [k for k in SOURCE_ID_COLUMNS if k in names]
    prefixes = ('truth_tau_a', 'truth_tau_b', 'truth_a_visible', 'truth_b_visible')
    columns = ids + [f'{p}_{k}' for p in prefixes for k in ('E', 'px', 'py', 'pz')]
    table = read_event_table(VAL, columns)
    raw = {k: np.asarray(table[k].to_numpy()) for k in columns}
    order = align(source_ids(raw, ids), panel['source_ids'])
    p4 = [np.stack([raw[f'{p}_{k}'][order] for k in ('E', 'px', 'py', 'pz')], -1) for p in prefixes]
    a, b = angles(*p4)
    terms = features(a, b, panel['kappas'])
    if terms.shape != (119002, 9) or not np.isfinite(terms).all():
        raise ValueError('Incomplete/nonfinite full-truth polarimeter contributions')
    return terms


def prepare(spec):
    import torch
    from train_diffusion_film_10pct import validate_inputs
    validate_spec(spec)
    reference = Path(spec['reference_run']).resolve(strict=True)
    new = yaml.safe_load((reference / 'runtime.yaml').read_text())
    inputs = validate_inputs(new)
    physics = new['options']['Training']['TauPhysics']
    for key in ('events', 'K', 'ddim_steps', 'seed', 'batch_size'):
        if physics[key] != spec[key]:
            raise ValueError('FiLM reference generation protocol differs: ' + key)
    old = expanded_runtime(spec['source_runtime'])
    for key, value in old['network'].get('VisibleConditioning', {}).items():
        if key.endswith('_enabled') and value:
            raise ValueError('Historical runtime contains an enabled conditioning intervention')
    if old['network']['TruthGeneration']['num_layers'] != 1:
        raise ValueError('Historical baseline must retain its original one-block architecture')
    old['rl'] = dict(enabled=False)
    old['compat'] = dict(backend='pure-evenet', repo_root=str(ROOT))
    # Do not upgrade the old architecture or drop any source model tensors.
    old.setdefault('network', {}).setdefault('VisibleConditioning', {}).update(
        diffusion_enabled=False, classifier_enabled=False, diffusion_fourier_enabled=False)
    sources = {}
    for name, pin in spec['checkpoints'].items():
        source = Path(pin['path']).resolve(strict=True)
        payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        sources[name] = dict(source=str(source), **checkpoint_metadata(payload, pin), role=pin['role'])
        del payload
    output = reserve_directory(spec['output_root'])
    for suffix in ('npz', 'json'):
        shutil.copy2(reference / f'validation_panel.{suffix}', output / f'validation_panel.{suffix}')
    panel, _ = load_panel(output)
    np.save(output / 'full_truth_cij.npy', full_truth_contributions(panel))
    normalization = Path(old['options']['Dataset']['normalization_file']).resolve(strict=True)
    shutil.copy2(normalization, output / 'normalization.pt')
    old['options']['Dataset']['normalization_file'] = str(output / 'normalization.pt')
    old['options']['Training'].update(model_checkpoint_load_path=None, pretrain_model_load_path=None)
    (output / 'runtime.yaml').write_text(yaml.safe_dump(old, sort_keys=False))
    for name, pin in sources.items():
        target = output / f'{name}.ckpt'
        shutil.copy2(pin['source'], target)
        pin['snapshot'] = str(target)
    references = {}
    for path in sorted(reference.glob('tau_physics/epoch-*/report.json')):
        report = json.loads(path.read_text())
        epoch = report['completed_epochs']
        if epoch <= 0:
            continue
        for key in ('events', 'K', 'ddim_steps', 'seed', 'batch_size'):
            if report['config'][key] != spec[key]:
                raise ValueError('Reference report sampling protocol changed: ' + key)
        measurements = path.parent / 'measurements.npz'
        if not measurements.is_file():
            raise ValueError('Completed reference report lacks measurements')
        name = f'film-on-epoch{epoch:03d}'
        dest = output / 'references' / name
        dest.mkdir(parents=True)
        shutil.copy2(path, dest / 'report.json')
        shutil.copy2(measurements, dest / 'measurements.npz')
        with np.load(dest / 'measurements.npz', allow_pickle=False) as f:
            align_measurements(f, panel)
        references[name] = dict(completed_epochs=epoch, measurements=str(dest / 'measurements.npz'))
    if not references:
        raise ValueError('Need at least one completed FiLM-on physics evaluation')
    write_json(output / 'spec.json', spec)
    manifest = dict(complete=True, stage='prepared_only', output=str(output), inputs=inputs,
        checkpoints=sources, references=references, reference_run=str(reference),
        optimizer_updates=0, classifier_fits=0, weights='strict_raw_state_dict_no_EMA',
        normalization='checkpoint_buffers_must_equal_copied_source_normalization',
        comparison='paired inference; historical training setups/budgets are not matched',
        full_truth='complete truth tau/visible p4; common A helicity k/r/n basis',
        matched_truth='true deltas under the same fixed-energy/reconstructed-visible map')
    write_json(output / 'prepared.json', manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return output


def build_frozen_model(output, name, device):
    import torch
    from evenet.control.global_config import Config
    from evenet.network.evenet_model import build_evenet_model_from_training_config
    # Each arm gets fresh parsed EventInfo; the singleton cannot reload an
    # expanded runtime after replacing its event_info mapping with an object.
    config = Config()
    config.load_yaml(Path(output) / 'runtime.yaml')
    norm = torch.load(Path(output) / 'normalization.pt', map_location='cpu', weights_only=False)
    torch.manual_seed(42)
    model = build_evenet_model_from_training_config(config, norm, device).to(device)
    manifest = json.loads((Path(output) / 'prepared.json').read_text())
    source = manifest['checkpoints'][name]
    payload = torch.load(source['snapshot'], map_location='cpu', weights_only=False, mmap=True)
    checkpoint_metadata(payload, source)
    strict_load(model, payload)
    return model


def preflight(output):
    import torch
    from train_diffusion_film_10pct import worker_import_preflight
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
    from evenet.utilities.joint_coverage import event_noise
    from evenet.utilities.diffusion_sampler import make_replay_sampler
    worker_import_preflight()
    panel, packing = load_panel(output)
    spec = json.loads((Path(output) / 'spec.json').read_text())
    ids = np.arange(2)
    batch = unpack_event_inputs(torch.as_tensor(panel['raw_condition'][ids]), packing)
    batch['classification'] = batch['classification'].long()
    batch['x_invisible'] = torch.zeros(2, 2, 2)
    batch['x_invisible_mask'] = torch.ones(2, 2, dtype=torch.bool)
    report = dict(events=2, device='cpu', optimizer_updates=0, arms={})
    for name in spec['checkpoints']:
        model = build_frozen_model(output, name, torch.device('cpu'))
        bank = event_noise(ids, spec['K'], spec['seed']).permute(1, 0, 2, 3)
        with torch.no_grad():
            draws = generate_neutrino_candidates(model, batch, make_replay_sampler(bank, 'legacy'),
                K=spec['K'], num_ddim_steps=spec['ddim_steps'], device=torch.device('cpu'), parallel_chains=1)
        terms = np.mean([cij_terms(panel, ids, d.numpy()) for d in draws], axis=0)
        if not torch.isfinite(draws).all() or not np.isfinite(terms).all():
            raise ValueError('Nonfinite CPU real-data sampler/pretrain Cij interface')
        report['arms'][name] = dict(strict_checkpoint_load=True, frozen=True,
            parameters=sum(p.numel() for p in model.parameters()), finite_ddim20_K8_and_cij=True)
        del model
    write_json(Path(output) / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def summarize(output):
    from RL.DGPO_neutrino.tau_reward_transfer import matrix_errors, paired_statistics
    from RL.DGPO_neutrino.conditional_tau_cycle import cij_terms
    output = Path(output)
    spec, manifest = [json.loads((output / name).read_text()) for name in ('spec.json', 'prepared.json')]
    panel, _ = load_panel(output)
    w = panel['event_weight'] / panel['event_weight'].sum()
    truth = dict(matched=cij_terms(panel, np.arange(len(w)), panel['truth_deltas']),
                 full=np.load(output / 'full_truth_cij.npy', allow_pickle=False))
    zero = np.zeros((len(w), 1))
    generated, sources = {}, {}
    for name, pin in manifest['checkpoints'].items():
        folder = output / name / 'tau_physics' / f'epoch-{pin["epoch"]+1:04d}'
        with np.load(folder / 'measurements.npz', allow_pickle=False) as f:
            align_measurements(f, panel)
            generated[name] = f['generated_cij'].copy()
        sources[name] = pin
    for name, pin in manifest['references'].items():
        with np.load(pin['measurements'], allow_pickle=False) as f:
            align_measurements(f, panel)
            generated[name] = f['generated_cij'].copy()
            if not np.allclose(f['truth_cij'], truth['matched'], rtol=0, atol=1e-12):
                raise ValueError('Reference matched-truth observable changed')
        sources[name] = pin
    metrics = {}
    for name, g in generated.items():
        metrics[name] = {}
        for ref, t in truth.items():
            delta = (w @ (g - t)).reshape(3, 3)
            metrics[name][ref] = dict(truth=(w @ t).reshape(3, 3).tolist(),
                generated=(w @ g).reshape(3, 3).tolist(), delta=delta.tolist(), **matrix_errors(delta))
    paired = {}
    for old in manifest['checkpoints']:
        for new in manifest['references']:
            paired[new + '-minus-' + old] = {ref: paired_statistics(zero, zero,
                generated[old], generated[new], t, panel['event_weight'],
                replicates=spec['bootstrap'], seed=spec['seed'])['cij'] for ref, t in truth.items()}
    report = dict(complete=True, events=len(w), K=spec['K'], ddim_steps=spec['ddim_steps'],
        seed=spec['seed'], sources=sources, metrics=metrics, paired=paired,
        optimizer_updates=0, classifier_fits=0,
        uncertainty='Paired condition bootstrap with common generation noise; excludes training-seed uncertainty',
        limitation='Historical epochs/training setups are not matched; this does not isolate the causal FiLM effect. Fixed-energy prediction is not complete rest-frame/entanglement closure.')
    write_json(output / 'report.json', report)
    lines = ['# Historical no-FiLM versus supervised FiLM spin evaluation', '',
        '| Model | Reference | Cij error | Diagonal | Offdiagonal |',
        '| --- | --- | ---: | ---: | ---: |']
    for name, rows in metrics.items():
        for ref, m in rows.items():
            lines.append('| ' + name + ' | ' + ref + ' | ' + ' | '.join(f'{m[k]:.6f}' for k in ERROR_KEYS) + ' |')
    for ref in truth:
        lines += ['', f'## All nine signed Cij deviations from {ref} truth', '',
            '| Model | ' + ' | '.join(i+j for i in AXES for j in AXES) + ' |',
            '| --- | ' + ' | '.join(['---:'] * 9) + ' |']
        for name, rows in metrics.items():
            lines.append('| ' + name + ' | ' + ' | '.join(
                f'{x:+.6f}' for row in rows[ref]['delta'] for x in row) + ' |')
    lines += ['', report['uncertainty'], '', report['limitation'], '',
        'Negative new-minus-old error changes indicate improvement. Exact paired 95% intervals and all nine matrix entries are in report.json.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return report


def worker(config):
    import torch
    import ray.train
    from ray.train.torch import get_device
    from evenet.utilities.tau_physics_validation import TauPhysicsValidation
    output = Path(config['output'])
    spec = json.loads((output / 'spec.json').read_text())
    manifest = json.loads((output / 'prepared.json').read_text())
    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    if world != 16:
        raise ValueError('Use sixteen allocated GPUs')
    device = get_device()
    # Match the supervised engine's A100 matmul mode used by saved FiLM draws.
    torch.set_float32_matmul_precision('medium' if 'A100' in torch.cuda.get_device_name(device) else 'highest')
    observer_config = {k: spec[k] for k in ('events', 'K', 'ddim_steps', 'seed', 'bootstrap', 'batch_size')}
    observer_config.update(every_n_epochs=1, panel_path=str(output / 'validation_panel.npz'),
        panel_metadata=str(output / 'validation_panel.json'))
    for name, pin in manifest['checkpoints'].items():
        model = build_frozen_model(output, name, device)
        observer = TauPhysicsValidation(observer_config, output / name / 'checkpoints')
        trainer = SimpleNamespace(global_rank=rank, world_size=world, global_step=pin['global_step'], loggers=[])
        observer.evaluate(trainer, SimpleNamespace(model=model, device=device), pin['epoch'] + 1)
        del observer, model
        torch.cuda.empty_cache()
    ray.train.report(dict(evaluated_checkpoints=len(manifest['checkpoints']), events_per_checkpoint=spec['events']))


def execute(output):
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from train_diffusion_film_10pct import worker_import_preflight, validate_inputs
    output = Path(output)
    spec = json.loads((output / 'spec.json').read_text())
    validate_spec(spec)
    reference = json.loads((output / 'prepared.json').read_text())['reference_run']
    validate_inputs(yaml.safe_load((Path(reference) / 'runtime.yaml').read_text()))
    worker_import_preflight()
    if (output / 'started.json').exists():
        raise ValueError('This invocation already started; prepare a fresh numbered evaluation')
    ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto', runtime_env={'env_vars':{
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'evenet_dgpo'), str(ROOT)]),
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}})
    available = ray.available_resources()
    if available.get('GPU', 0) < 16 or available.get('CPU', 0) < 32:
        raise ValueError('Requires sixteen free GPUs and 32 CPUs in the existing Ray allocation')
    import wandb
    for key in ('WANDB_RUN_ID', 'WANDB_RESUME'):
        os.environ.pop(key, None)
    run = wandb.init(**spec['wandb'], config={**spec, 'provenance': json.loads((output / 'prepared.json').read_text())},
                     dir=str(output), job_type='inference-only-cij')
    write_json(output / 'started.json', dict(user_launched=True, wandb_id=run.id))
    try:
        TorchTrainer(worker, train_loop_config=dict(output=str(output)),
            scaling_config=ScalingConfig(num_workers=16, use_gpu=True, resources_per_worker={'CPU': 2, 'GPU': 1}),
            run_config=RunConfig(name=output.name, storage_path=str(output / 'ray_results'),
                                 failure_config=FailureConfig(max_failures=0))).fit()
        report = summarize(output)
        for name, rows in report['metrics'].items():
            values = {f'cij/{ref}/{name}/{key}': m[key] for ref, m in rows.items() for key in ERROR_KEYS}
            for ref, m in rows.items():
                for family in ('truth', 'generated', 'delta'):
                    values.update({f'cij/{ref}/{name}/{family}/{left}{right}': m[family][i][j]
                        for i, left in enumerate(AXES) for j, right in enumerate(AXES)})
            run.log(values)
        for pair, rows in report['paired'].items():
            run.log({f'paired/{ref}/{pair}/{key}/{bound}': value
                for ref, m in rows.items() for key, interval in m['error_changes'].items()
                for bound, value in interval.items()})
        run.save(str(output / 'report.json'), base_path=str(output))
        run.save(str(output / 'REPORT.md'), base_path=str(output))
        run.summary['complete'] = True
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()
    print('Full Cij report:', output / 'REPORT.md', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('inspect', 'prepare', 'preflight', 'run'))
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--prepared', type=Path)
    args = parser.parse_args(argv)
    spec = yaml.safe_load(args.config.read_text())
    validate_spec(spec)
    if args.stage == 'inspect':
        import torch
        for name, pin in spec['checkpoints'].items():
            path = Path(pin['path']).resolve(strict=True)
            payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            print(name, path, checkpoint_metadata(payload, pin), flush=True)
        return
    output = args.prepared.resolve(strict=True) if args.prepared else prepare(spec)
    if not json.loads((output / 'prepared.json').read_text())['complete']:
        raise ValueError('Incomplete prepared invocation')
    if args.stage == 'preflight':
        preflight(output)
    elif args.stage == 'run':
        execute(output)


if __name__ == '__main__':
    main()
