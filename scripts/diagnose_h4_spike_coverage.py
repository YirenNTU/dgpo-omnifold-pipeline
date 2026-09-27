#!/usr/bin/env python3
"""Fixed-threshold coverage audit; optional 16-GPU frozen-policy sampling replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from diagnose_h4_ratio_tail import unpack
from diagnose_h4_topology_resolution import analyze as validate_topology, reconstruct

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (1e-6, 1e-5, 1e-4)
PREFIXES = (1, 8, 32, 128)
PRETRAIN_CHECKPOINT = '/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt'
FULL_PRETRAIN_CHECKPOINT = '/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_v1/checkpoints/last.ckpt'
ARM_METADATA = {
    'step1110': ('h4cov1', 'Can the policy sample the sharp spike? | step-1110 replay | K=128'),
    'pretrained10pct': ('h4covpre1', 'Did DGPO lose spike coverage? | pretrained 10% raw | matched K=128'),
    'pretrainedfull': ('h4covfull1', 'Does full pretraining recover spike coverage? | full-pretrain raw | matched K=128'),
}


def load_raw_state(policy, source, arm):
    """Never inspect or load EMA; reject incompatible raw state before sampling."""
    if arm == 'step1110' and int(source.get('global_step', -1)) != 1110:
        raise ValueError('Requires old DGPO step-1110 checkpoint')
    if arm not in ARM_METADATA:
        raise ValueError('Unknown checkpoint arm')
    if arm.startswith('pretrained') and int(source.get('dgpo_checkpoint_version', 0)) > 0:
        raise ValueError('Pretrained arm cannot load a DGPO checkpoint')
    expected = {k.removeprefix('model.'): v for k, v in source['state_dict'].items()}
    actual = policy.state_dict()
    if set(actual) - set(expected) or any(k not in actual and not k.startswith('famo.w.') for k in expected):
        raise ValueError('Checkpoint/model keys differ')
    for key, value in actual.items():
        if value.dtype != expected[key].dtype or value.shape != expected[key].shape:
            raise ValueError(f'Raw checkpoint shape/dtype mismatch: {key}')
    policy.load_state_dict({k: expected[k] for k in actual}, strict=True)
    for key, value in policy.state_dict().items():
        if not torch.equal(value.detach().cpu(), expected[key].cpu()):
            raise ValueError(f'Raw checkpoint weight mismatch: {key}')


def matched_source(directory, args, bundle, ids):
    """Fail closed on a different panel or random-number/sampler protocol."""
    directory = Path(directory)
    if not (directory / 'COMPLETE').is_file():
        raise ValueError('Matched baseline is incomplete')
    manifest = json.loads((directory / 'manifest.json').read_text())
    for key in ('workers', 'events', 'batch_size', 'seed'):
        if manifest[key] != getattr(args, key):
            raise ValueError(f'Matched baseline {key} differs')
    if manifest['K'] != 128 or manifest['policy_updates'] != 0:
        raise ValueError('Incompatible baseline protocol')
    panel = torch.load(directory / 'panel.pt', map_location='cpu', weights_only=True)
    for key, value in dict(test_rows=ids, pool_rows=bundle['split_indices']['test'][ids],
                           truth=bundle['test_truth'][ids], condition=bundle['test_condition'][ids]).items():
        if not torch.equal(panel[key], value):
            raise ValueError(f'Matched baseline panel {key} differs')
    if panel['packing_spec'] != bundle['packing_spec']:
        raise ValueError('Matched packing spec differs')
    return manifest, panel


def paired_joint_comparison(before_angles, after_angles):
    if before_angles.shape != after_angles.shape or after_angles.ndim != 3 or after_angles.shape[1:] != (128, 2):
        raise ValueError('Baseline candidate shape differs')
    if not np.isfinite(before_angles).all() or not np.isfinite(after_angles).all():
        raise ValueError('Nonfinite comparison angles')
    rows = []
    for threshold in THRESHOLDS:
        for k in PREFIXES:
            before = (before_angles[:, :k] < threshold).all(-1).mean(1)
            after = (after_angles[:, :k] < threshold).all(-1).mean(1)
            delta = after - before
            rows.append(dict(K=k, threshold_radians=threshold,
                current_minus_baseline_joint_fraction=float(delta.mean()),
                paired_event_se=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None))
    return rows


def audit_ddim_steps(raw):
    value = raw['dgpo'].get('validation_num_ddim_steps')
    steps = int(value if value is not None else raw['dgpo']['num_ddim_steps'])
    if steps < 1:
        raise ValueError('DDIM steps must be positive')
    return steps


def select_panel(n, size, seed):
    if not 0 < size <= n:
        raise ValueError('Panel size must be positive and no larger than test set')
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:size]


def coverage(truth, generated):
    """Angles are (events, 2) and (events, candidates, 2), in radians."""
    t, g = np.asarray(truth), np.asarray(generated)
    if t.ndim != 2 or t.shape[1] != 2 or g.ndim != 3 or g.shape[0] != len(t) or g.shape[2] != 2 or not len(t) or not g.shape[1]:
        raise ValueError('Invalid aligned angle shapes')
    if not np.isfinite(t).all() or not np.isfinite(g).all() or (t < 0).any() or (g < 0).any():
        raise ValueError('Invalid angles')
    rows = []
    for threshold in THRESHOLDS:
        for name, axes in [('acoplanarity', [0]), ('acollinearity', [1]), ('joint', [0, 1])]:
            a = (t[:, axes] < threshold).all(-1)
            b = (g[:, :, axes] < threshold).all(-1)
            per_event = b.mean(1)
            hit = b.any(1)
            rows.append(dict(threshold_radians=threshold, region=name,
                truth_count=int(a.sum()), truth_fraction=float(a.mean()),
                generated_draw_count=int(b.sum()), generated_draw_fraction=float(b.mean()),
                generated_fraction_event_cluster_se=float(per_event.std(ddof=1) / np.sqrt(len(t))) if len(t)>1 else None,
                any_hit_event_count=int(hit.sum()), any_hit_event_fraction=float(hit.mean()),
                truth_spike_events_with_hit=int((a & hit).sum()),
                hit_fraction_given_truth_spike=float(hit[a].mean()) if a.any() else None))
    return dict(events=len(t), candidates_per_event=g.shape[1], regions=rows)


def angles(fields, candidates):
    r = reconstruct(fields, candidates)
    if any(r['invalid'].values()):
        raise ValueError(f"Invalid direction inputs: {r['invalid']}")
    return np.stack([r['acoplanarity'], r['acollinearity']], -1)


def load_artifact(directory):
    directory = Path(directory)
    if not (directory / 'COMPLETE').is_file():
        raise ValueError('Incomplete ratio artifact')
    b = torch.load(directory / 'best_classifier_and_test.pt', map_location='cpu', weights_only=True)
    checks = validate_topology(b)['checks']
    if not all(v['matches_saved'] for v in checks.values()):
        raise ValueError('Saved topology reconstruction mismatch')
    return b


def write_json(path, result):
    with Path(path).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


def worker(cfg):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg['runtime'])
    torch.set_float32_matmul_precision(str(global_config.dgpo.get('float32_matmul_precision', 'medium')))
    source = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device).eval()
    load_raw_state(policy, source, cfg['arm'])
    checkpoint_step = source.get('global_step')
    del source
    policy.requires_grad_(False)
    panel = torch.load(Path(cfg['output']) / 'panel.pt', map_location='cpu', weights_only=True)
    positions = torch.arange(rank, len(panel['truth']), world)
    spec = EventPackingSpec.from_dict(panel['packing_spec'])
    sampler = DDIMSampler(device=device)
    torch.manual_seed(cfg['seed'] + 10000 + rank)
    outputs = []
    run = None
    if rank == 0 and cfg['wandb']:
        import wandb
        run = wandb.init(project='nu2flow-RL', entity='ytchou97-university-of-washington',
            id=cfg['run_id'], resume='never',
            name=cfg['run_name'],
            group='H4 ratio coverage', tags=['no-policy-update', 'fixed-events', 'coverage', 'raw-only', cfg['arm']], config=cfg)
        run.summary.update({'checkpoint/global_step': checkpoint_step, 'checkpoint/raw_verified': True})
    success = False
    try:
        for start in range(0, len(positions), cfg['batch_size']):
            ids = positions[start:start + cfg['batch_size']]
            batch = unpack_event_inputs(panel['condition'][ids].to(device), spec)
            # Production materialize_pool selects events with BOTH slots valid.
            # Only the shape is consumed by sampling; do not expose truth values.
            batch['x_invisible'] = torch.zeros(len(ids), 2, 2, device=device)
            batch['x_invisible_mask'] = torch.ones(len(ids), 2, dtype=torch.bool, device=device)
            generated = generate_neutrino_candidates(policy, batch, sampler, K=128,
                num_ddim_steps=cfg['ddim_steps'], device=device, parallel_chains=1)
            if tuple(generated.shape) != (128, len(ids), 2, 2) or not torch.isfinite(generated).all():
                raise ValueError('Invalid sampler output')
            outputs.append(generated.permute(1, 0, 2, 3).reshape(len(ids), 128, 4).cpu())
            print(f'[coverage rank={rank}] events={min(start+len(ids),len(positions))}/{len(positions)} K=128', flush=True)
            if run:
                run.log({'sampling/rank0_events_done': start + len(ids), 'sampling/K': 128})
        torch.save(dict(positions=positions, generated=torch.cat(outputs)), Path(cfg['output']) / f'rank-{rank:03d}.pt')
        torch.distributed.barrier()
        if rank == 0:
            generated = torch.empty(len(panel['truth']), 128, 4)
            seen = []
            for i in range(world):
                part = torch.load(Path(cfg['output']) / f'rank-{i:03d}.pt', weights_only=True)
                generated[part['positions']] = part['generated']
                seen.extend(part['positions'].tolist())
            if sorted(seen) != list(range(len(generated))):
                raise ValueError('Missing/duplicate panel positions')
            fields = unpack(dict(test_condition=panel['condition'], packing_spec=panel['packing_spec']))
            truth_angles = angles(fields, panel['truth'])
            candidate_angles = np.stack([angles(fields, generated[:, k]) for k in range(128)], 1)
            results = {str(k): coverage(truth_angles, candidate_angles[:, :k]) for k in PREFIXES}
            torch.save(dict(generated=generated, angles=torch.from_numpy(candidate_angles)), Path(cfg['output']) / 'candidates.pt')
            write_json(Path(cfg['output']) / 'coverage_report.json', dict(schema='h4-spike-coverage-v1', config=cfg, prefixes=results,
                limitation='Reused test, exploratory. Any-hit is candidate availability, not distribution closure. Zero hits do not prove zero support.'))
            if cfg.get('matched_baseline'):
                old = torch.load(Path(cfg['matched_baseline']) / 'candidates.pt', map_location='cpu', weights_only=True)['angles'].numpy()
                if old.shape != candidate_angles.shape:
                    raise ValueError('Baseline candidate shape differs')
                comparisons = []
                for threshold in THRESHOLDS:
                    for k in PREFIXES:
                        before = (old[:, :k] < threshold).all(-1).mean(1)
                        after = (candidate_angles[:, :k] < threshold).all(-1).mean(1)
                        delta = after - before
                        comparisons.append(dict(K=k, threshold_radians=threshold,
                            pretrained_minus_step1110_joint_fraction=float(delta.mean()),
                            paired_event_se=float(delta.std(ddof=1)/np.sqrt(len(delta)))))
                write_json(Path(cfg['output']) / 'paired_comparison.json', comparisons)
                if run:
                    for row in comparisons:
                        if row['K'] == 128:
                            run.summary[f"comparison/joint/{row['threshold_radians']:g}/pretrained_minus_step1110"] = row['pretrained_minus_step1110_joint_fraction']
            if cfg.get('compare_pretrained10pct'):
                before = torch.load(Path(cfg['compare_pretrained10pct']) / 'candidates.pt', map_location='cpu', weights_only=True)['angles'].numpy()
                rows = paired_joint_comparison(before, candidate_angles)
                write_json(Path(cfg['output']) / 'paired_vs_pretrained10pct.json', dict(
                    current_arm=cfg['arm'], baseline=cfg['compare_pretrained10pct'], comparisons=rows))
                if run:
                    for row in rows:
                        if row['K'] == 128:
                            prefix = f"comparison_vs_pretrained10pct/joint/{row['threshold_radians']:g}"
                            run.summary[prefix + '/delta'] = row['current_minus_baseline_joint_fraction']
                            run.summary[prefix + '/paired_event_se'] = row['paired_event_se']
            if run:
                for k, result in results.items():
                    log = {'coverage/K': int(k)}
                    for row in result['regions']:
                        prefix = f"coverage/{row['region']}/{row['threshold_radians']:g}"
                        for key in ('truth_fraction', 'generated_draw_fraction', 'any_hit_event_fraction', 'hit_fraction_given_truth_spike'):
                            if row[key] is not None:
                                log[f'{prefix}/{key}'] = row[key]
                    run.log(log)
            (Path(cfg['output']) / 'COMPLETE').write_text('h4-spike-coverage-v1\n')
        success = True
    finally:
        if run:
            run.finish(exit_code=0 if success else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    p.add_argument('--sample', action='store_true')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--events', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42017)
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--run-id')
    p.add_argument('--arm', choices=list(ARM_METADATA), default='step1110')
    p.add_argument('--matched-baseline', type=Path, help='Completed h4cov1 directory; required for pretrained arm')
    p.add_argument('--compare-pretrained10pct', type=Path, help='Completed h4covpre1 directory; required for full-pretrained arm')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a fresh directory')
    if args.arm.startswith('pretrained') and (not args.sample or args.matched_baseline is None):
        p.error('Pretrained arm requires --sample and --matched-baseline')
    if args.matched_baseline and not args.arm.startswith('pretrained'):
        p.error('--matched-baseline is for pretrained arms')
    if (args.arm == 'pretrainedfull') != (args.compare_pretrained10pct is not None):
        p.error('Full-pretrained arm requires --compare-pretrained10pct; other arms must omit it')
    if min(args.workers, args.batch_size, args.events) < 1 or args.events < args.workers:
        p.error('Positive sizes required; events must be >= workers')
    b = load_artifact(args.directory)
    fields = unpack(b)
    baseline = coverage(angles(fields, b['test_truth']), angles(fields, b['test_generated'])[:, None])
    ids = select_panel(len(b['test_truth']), args.events, args.seed)
    matched = matched_source(args.matched_baseline, args, b, ids) if args.matched_baseline else None
    comparison = matched_source(args.compare_pretrained10pct, args, b, ids) if args.compare_pretrained10pct else None
    if comparison:
        meta = comparison[0]
        if meta.get('arm') != 'pretrained10pct' or meta.get('weights') != 'raw_state_dict_only':
            p.error('Comparison must be the raw pretrained10pct arm')
        if meta['ddim_steps'] != matched[0]['ddim_steps']:
            p.error('Comparison DDIM budget differs')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'artifact_coverage.json', baseline)
    print(json.dumps(baseline, indent=2), flush=True)
    if not args.sample:
        return
    import yaml
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    overlay_path = args.directory.parent / 'resolved_ratio_experiment.yaml'
    raw = deep_update(read_yaml(ROOT / 'config/train_diffusion_nersc.yaml'), read_yaml(overlay_path))
    raw = absolutize_default_paths(raw, ROOT / 'config')
    raw.setdefault('compat', {}).update(backend='dgpo-evenet', repo_root=str(ROOT))
    raw.setdefault('rl', {})['enabled'] = True
    if matched:
        raw = read_yaml(args.matched_baseline / 'runtime.yaml')
        if audit_ddim_steps(raw) != matched[0]['ddim_steps']:
            raise ValueError('Baseline runtime DDIM steps differ from manifest')
        if raw['options']['Training']['model_checkpoint_load_path'] != matched[0]['checkpoint']:
            raise ValueError('Baseline checkpoint provenance mismatch')
        raw['options']['Training']['model_checkpoint_load_path'] = FULL_PRETRAIN_CHECKPOINT if args.arm == 'pretrainedfull' else PRETRAIN_CHECKPOINT
        raw['options']['Training']['pretrain_model_load_path'] = None
    checkpoint = raw['options']['Training']['model_checkpoint_load_path']
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    runtime = args.output.resolve() / 'runtime.yaml'
    with runtime.open('x') as stream:
        yaml.safe_dump(raw, stream, sort_keys=False)
    panel = matched[1] if matched else dict(condition=b['test_condition'][ids], truth=b['test_truth'][ids],
        test_rows=ids, pool_rows=b['split_indices']['test'][ids], packing_spec=b['packing_spec'])
    torch.save(panel, args.output / 'panel.pt')
    # Online raw audit pool is generated with num_ddim_val in dgpo_trainer.
    ddim_steps = audit_ddim_steps(raw)
    cfg = dict(runtime=str(runtime), checkpoint=checkpoint, output=str(args.output.resolve()),
        artifact=str(args.directory.resolve()), ddim_steps=ddim_steps, seed=args.seed,
        batch_size=args.batch_size, workers=args.workers, events=args.events, K=128,
        wandb=not args.no_wandb, run_id=args.run_id or ARM_METADATA[args.arm][0],
        arm=args.arm, weights='raw_state_dict_only',
        matched_baseline=str(args.matched_baseline.resolve()) if matched else None,
        compare_pretrained10pct=str(args.compare_pretrained10pct.resolve()) if comparison else None,
        training_evaluation_overlap='unverified; exploratory checkpoint comparison, not held-out generalization',
        training_budget_comparability='unverified; not a pure dataset-size causal ablation',
        run_name=ARM_METADATA[args.arm][1],
        policy_updates=0, classifier_fits=0)
    write_json(args.output / 'manifest.json', cfg)
    ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto', runtime_env={'env_vars': {
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'evenet_dgpo'), str(ROOT / 'scripts'), os.environ.get('PYTHONPATH', '')])}})
    if ray.cluster_resources().get('GPU', 0) < args.workers:
        raise RuntimeError(f'Requires {args.workers} GPUs')
    TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=True),
        run_config=RunConfig(name='spike-coverage', storage_path=str(args.output.resolve() / 'ray_results'),
            failure_config=FailureConfig(max_failures=0))).fit()
    print('FULL REPORT:', args.output / 'coverage_report.json')


if __name__ == '__main__':
    main()
