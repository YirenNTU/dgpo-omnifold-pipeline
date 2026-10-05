#!/usr/bin/env python3
"""Paired inference-only matmul precision screen on an existing 16-GPU Ray cluster."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evenet_dgpo'))
ARMS = ('medium', 'high', 'highest')
NAME = 'Does precision limit coverage? | global FiLM | paired FP32 inference'
TIMES = (.02, .1, .3, .5, .7, .9, .98)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@contextmanager
def precision_scope(precision):
    """Apply AFTER model construction; restore even if inference fails."""
    if precision not in ARMS:
        raise ValueError('Unknown precision')
    previous = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision(precision)
        # Keep convolution settings identical in all arms.
        torch.backends.cudnn.allow_tf32 = False
        yield dict(matmul=torch.get_float32_matmul_precision(),
                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    finally:
        torch.set_float32_matmul_precision(previous)
        torch.backends.cudnn.allow_tf32 = cudnn


def geometry(fields, candidates, dtype):
    """Same stable physics formula, differing only in CPU arithmetic dtype.

    Cast BEFORE angle reconstruction, never after precision has been lost.
    Promoting archived inputs does not restore upstream data precision.
    """
    z = candidates.to(dtype).reshape(len(candidates), -1, 2, 2)
    directions, phis = [], []
    for j, leg in enumerate(('a', 'b')):
        p = torch.stack([fields[f'lead_{leg}_visible_{axis}'].reshape(-1).to(dtype)
                         for axis in ('px', 'py', 'pz')], -1)
        theta = torch.atan2(torch.hypot(p[:, 0], p[:, 1]), p[:, 2])[:, None] + z[:, :, j, 0]
        phi = torch.atan2(p[:, 1], p[:, 0])[:, None] + z[:, :, j, 1]
        directions.append(torch.stack((theta.sin()*phi.cos(), theta.sin()*phi.sin(), theta.cos()), -1))
        phis.append(phi)
    delta = phis[0]-phis[1]
    acop = torch.pi-torch.atan2(delta.sin(), delta.cos()).abs()
    dot = (directions[0]*directions[1]).sum(-1)
    cross = torch.linalg.cross(directions[0], directions[1], dim=-1).norm(dim=-1)
    acol = torch.pi-torch.atan2(cross, dot)
    return torch.stack((acop, acol), -1).double().numpy()


def difference(before, after):
    delta = (after.double()-before.double()).abs()
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite paired difference')
    return dict(rms=delta.square().mean().sqrt().item(), max=delta.max().item(),
                p99=torch.quantile(delta.flatten(), .99).item())


def mse_comparison(before, after, cfg):
    a, b = before.double(), after.double()
    delta = b-a
    indices = torch.randint(len(delta), (cfg['bootstrap'], len(delta)),
                            generator=torch.Generator().manual_seed(cfg['seed']))
    return dict(before=a.mean().item(), highest=b.mean().item(),
                delta_mse=delta.mean().item(),
                delta_ci95=torch.quantile(delta[indices].mean(1),
                    torch.tensor([.025, .975], dtype=torch.float64)).tolist())


def analyze(panel, values, cfg):
    from diagnose_h4_ddim_coverage import analyze_arm, paired_comparison
    from diagnose_h4_ratio_tail import unpack
    fields = unpack(dict(test_condition=panel['condition'], packing_spec=panel['packing_spec']))
    report = dict(complete=False, config=cfg, arms={}, comparisons={}, metric_precision={})
    angles = {}
    truth64 = geometry(fields, panel['truth'], torch.float64)[:, 0]
    for arm in ARMS:
        samples = values[f'samples/{arm}']
        result, truth_angles, candidate_angles = analyze_arm(panel, samples)
        angles[arm] = candidate_angles
        fp64 = geometry(fields, samples, torch.float64)
        if not np.allclose(fp64, candidate_angles, rtol=0, atol=1e-12) or not np.allclose(truth64, truth_angles, rtol=0, atol=1e-12):
            raise ValueError('Independent FP64 geometry disagrees with canonical coverage')
        fp32 = geometry(fields, samples, torch.float32)
        report['arms'][arm] = result
        report['metric_precision'][arm] = dict(
            angle_difference=difference(torch.from_numpy(fp32), torch.from_numpy(fp64)),
            # Truth held fixed at FP64 here, isolating generated metric arithmetic.
            fp64_minus_fp32=paired_comparison(truth64, fp32, fp64, cfg['bootstrap'], cfg['seed']))
    truth32 = geometry(fields, panel['truth'], torch.float32)[:, 0]
    report['truth_metric_precision'] = dict(
        angle_difference=difference(torch.from_numpy(truth32), torch.from_numpy(truth64)),
        joint_fraction={str(t): dict(fp32=float((truth32 < t).all(-1).mean()),
                                     fp64=float((truth64 < t).all(-1).mean())) for t in (1e-6, 1e-5, 1e-4)})
    for arm in ('medium', 'high'):
        report['comparisons'][f'highest_minus_{arm}'] = dict(
            target_coordinate_difference=difference(values[f'samples/{arm}'], values['samples/highest']),
            coverage=paired_comparison(truth64, angles[arm], angles['highest'], cfg['bootstrap'], cfg['seed']),
            velocity={str(t): difference(values[f'velocity/{arm}/{t}'], values[f'velocity/highest/{t}']) for t in TIMES},
            mse={str(t): mse_comparison(values[f'mse/{arm}/{t}'], values[f'mse/highest/{t}'], cfg) for t in TIMES})
    report['limitations'] = (
        'Inference only; no reward classifier, gradients, optimizer updates or training-precision conclusion. '
        'FP64 metrics promote saved FP32 inputs, not the network or upstream data. '
        'Pointwise event-bootstrap intervals on a reused panel and one checkpoint; no training-seed uncertainty. '
        'Identical outputs may mean the backend used the same kernel. Zero hits/zero CI do not prove zero support. '
        'Coordinate divergence alone is not distribution improvement; inspect truth-gap and W1 together.')
    return report


def evaluate_batch(policy, batch, truth, noise, cfg):
    from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma
    from diagnose_h4_ddim_coverage import make_replay_sampler
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    device = truth.device
    mask = torch.ones(len(truth), 2, 1, device=device)
    batch = dict(batch, x_invisible=torch.zeros_like(truth), x_invisible_mask=mask.squeeze(-1).bool())
    values, settings = {}, {}
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        # All probes use the exact same noised truth and target tensors.
        x0 = policy.invisible_normalizer(x=truth, mask=mask)
        probes = []
        for t in TIMES:
            time = torch.full((len(truth),), t, device=device)
            _, a, s = get_logsnr_alpha_sigma(time, (-1, 1, 1))
            probes.append((t, time, a*x0+s*noise[:, 0], a*noise[:, 0]-s*x0))
        for arm in ARMS:
            with precision_scope(arm) as effective:
                settings[arm] = effective
                for t, time, xt, target in probes:
                    v = policy.predict_diffusion_vector(xt, batch, time, 'neutrino', noise_mask=mask)
                    values[f'velocity/{arm}/{t}'] = v.cpu()
                    values[f'mse/{arm}/{t}'] = (v.double()-target.double()).square().mean((-1, -2)).cpu()
                sampler = make_replay_sampler(noise.permute(1, 0, 2, 3), cfg['x0_mode'])
                generated = generate_neutrino_candidates(policy, batch, sampler, K=cfg['K'],
                    num_ddim_steps=cfg['ddim_steps'], device=device, parallel_chains=1)
                if sampler.draws_used != cfg['K']:
                    raise ValueError('Noise replay chain count differs')
                values[f'samples/{arm}'] = generated.permute(1, 0, 2, 3).reshape(len(truth), cfg['K'], 4).cpu()
        # Same-mode replay detects state mutation/nondeterministic forward probes.
        with precision_scope('medium'):
            t, time, xt, _ = probes[0]
            replay = policy.predict_diffusion_vector(xt, batch, time, 'neutrino', noise_mask=mask).cpu()
            if not torch.equal(replay, values[f'velocity/medium/{t}']):
                raise ValueError('Same-precision velocity replay is not exact')
    if any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite diagnostic output')
    return values, settings


def worker(cfg):
    import ray.train
    import ray.train.torch
    import torch.distributed as dist
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from diagnose_h4_spike_coverage import load_raw_state
    from diagnose_film_strength import merge_parts
    from evenet.utilities.joint_coverage import event_noise

    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    if world != 16:
        raise ValueError('Exactly 16 GPU workers required')
    device, out = ray.train.torch.get_device(), Path(cfg['output'])
    global_config.load_yaml(str(out / 'runtime.yaml'))
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device)
    load_raw_state(policy, torch.load(out / 'raw_policy.pt', map_location='cpu', weights_only=True), 'pretrained10pct')
    policy.eval().requires_grad_(False)
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in policy.parameters()):
        raise ValueError('Expected FP32 model parameters')
    panel = torch.load(out / 'panel.pt', map_location='cpu', weights_only=True)
    spec = EventPackingSpec.from_dict(panel['packing_spec'])
    positions = torch.arange(rank, len(panel['truth']), world)
    run, success = None, False
    try:
        if rank == 0 and cfg['wandb']:
            import wandb
            run = wandb.init(project='EveNet', entity='ytchou97-university-of-washington',
                id=cfg['run_id'], resume='never', name=NAME, group='Numerical precision diagnostics', config=cfg)
        chunks = {}
        for start in range(0, len(positions), cfg['batch_size']):
            ids = positions[start:start+cfg['batch_size']]
            batch = unpack_event_inputs(panel['condition'][ids].to(device), spec)
            batch = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in batch.items()}
            noise = event_noise(ids, cfg['K'], cfg['seed']).to(device)
            values, settings = evaluate_batch(policy, batch, panel['truth'][ids].reshape(-1, 2, 2).to(device), noise, cfg)
            values['initial_noise'] = noise.cpu()
            for key, value in values.items():
                chunks.setdefault(key, []).append(value)
            if rank == 0:
                print(f'precision screen rank0 events {start+len(ids)}/{len(positions)}', flush=True)
                if run:
                    run.log({'progress/rank0_events': start+len(ids)})
        part = dict(positions=positions, values={k: torch.cat(v) for k, v in chunks.items()},
                    effective_settings=settings, torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    device=torch.cuda.get_device_name(device))
        torch.save(part, out / f'rank-{rank:02d}.pt')
        gathered = [None]*world if rank == 0 else None
        dist.gather_object(part, gathered, dst=0)
        if rank == 0:
            if any(p['effective_settings'] != settings for p in gathered):
                raise ValueError('Worker precision settings disagree')
            values = merge_parts(gathered, len(panel['truth']))
            torch.save(dict(test_rows=panel['test_rows'], pool_rows=panel['pool_rows'], values=values), out / 'paired.pt')
            report = analyze(panel, values, cfg)
            report['workers'] = [{k: p[k] for k in ('effective_settings', 'torch_version', 'cuda_version', 'device')} for p in gathered]
            report['complete'] = True
            save_json(out / 'report.json', report)
            lines = ['# Paired precision screen', '',
                     f"Checkpoint epoch {cfg['checkpoint']['epoch']}, step {cfg['checkpoint']['global_step']}; "
                     f"{cfg['events']} events, K={cfg['K']}, {cfg['ddim_steps']} DDIM steps, {cfg['x0_mode']}.", '',
                     '| Comparison | Target-coordinate RMS difference (rad) | Joint <1e-4 absolute truth-gap change [95% CI] |',
                     '|---|---:|---|']
            for key, row in report['comparisons'].items():
                joint = next(r for r in row['coverage'] if r['region']=='joint' and r['threshold_radians']==1e-4)
                lines.append(f"| {key} | {row['target_coordinate_difference']['rms']:.6g} | "
                             f"{joint['absolute_truth_gap_change']:.6g} {joint['absolute_truth_gap_change_ci95']} |")
            lines += ['', 'Negative truth-gap change is improvement. Coordinate divergence alone is not improvement.',
                      '', report['limitations'], '',
                      'See report.json for all time probes, W1, FP32/FP64 metric contrasts and actual worker settings.']
            (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
            if run:
                metrics = {'evaluation/complete': True}
                for key, row in report['comparisons'].items():
                    metrics[f'{key}/coordinate_rms'] = row['target_coordinate_difference']['rms']
                    joint = next(r for r in row['coverage'] if r['region']=='joint' and r['threshold_radians']==1e-4)
                    metrics[f'{key}/joint_gap_change'] = joint['absolute_truth_gap_change']
                run.log(metrics)
                run.summary.update(metrics)
                run.save(str(out / 'report.json'), base_path=str(out), policy='now')
            (out / 'COMPLETE').write_text(cfg['schema']+'\n')
        dist.barrier()
        success = True
    finally:
        if run:
            run.finish(exit_code=0 if success else 1)


def ray_worker_entry(cfg):
    """Keep the serialized entry point free of model/backend global objects.

    Executing this file as __main__ makes cloudpickle traverse worker helpers
    by value, including contextmanager closures and PyTorch backend wrappers.
    Import the implementation on the worker instead (scripts is on PYTHONPATH).
    """
    from diagnose_precision import worker as implementation
    return implementation(cfg)


def main(argv=None):
    from evaluate_global_film_coverage import inspect_checkpoint, runtime_configuration
    from diagnose_film_strength import validate_panel
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--panel', type=Path)
    p.add_argument('--panel-report', type=Path, default=Path('/pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_lr_10pct_seed42/checkpoints/joint_coverage/epoch-0100.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--ddim-steps', type=int, default=20)
    p.add_argument('--x0-mode', choices=('legacy', 'stable_v'), default='legacy')
    p.add_argument('--seed', type=int, default=42017)
    p.add_argument('--bootstrap', type=int, default=500)
    p.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    if min(args.batch_size, args.K, args.ddim_steps) < 1 or args.bootstrap < 20:
        p.error('Positive batch/K/steps and bootstrap >=20 required')
    cfg = dict(schema='precision-screen-v1', checkpoint_path=str(args.checkpoint), output=str(args.output.resolve()),
        workers=16, K=args.K, ddim_steps=args.ddim_steps, x0_mode=args.x0_mode, batch_size=args.batch_size,
        seed=args.seed, bootstrap=args.bootstrap, times=list(TIMES), arms=list(ARMS),
        wandb=not args.no_wandb, run_id=uuid.uuid4().hex[:8], run_name=NAME,
        policy_updates=0, classifier_fits=0, panel_path=str(args.panel) if args.panel else None,
        panel_report=str(args.panel_report), network_dtype='float32', autocast=False)
    runtime = runtime_configuration()
    runtime['logger']['wandb'].update(run_name=NAME, group='Numerical precision diagnostics', project='EveNet')
    runtime['experiment'] = dict(question=NAME, policy_updates=0, classifier_fits=0)
    runtime['nersc']['execution'] = dict(mode='inference_only_existing_ray_cluster')
    if args.dry_run:
        print(yaml.safe_dump(dict(diagnostic=cfg, runtime=runtime), sort_keys=False))
        return
    panel_path = args.panel or Path(json.loads(args.panel_report.read_text())['config']['panel_path'])
    panel = torch.load(panel_path, map_location='cpu', weights_only=True)
    validate_panel(panel)
    if len(panel['truth']) < 16:
        raise ValueError('At least 16 events required')
    metadata, raw = inspect_checkpoint(args.checkpoint)
    cfg.update(checkpoint=metadata, panel_path=str(panel_path.resolve()), events=len(panel['truth']))
    out = Path(cfg['output'])
    if out.exists():
        p.error('Use a fresh output directory')
    if args.check_only:
        print(json.dumps(cfg, indent=2))
        return
    if int(os.environ.get('SLURM_PROCID', '0')) != 0:
        p.error('Launch once only')
    import ray
    # Fail before creating output files if future edits break serialization.
    from ray import cloudpickle
    cloudpickle.dumps(ray_worker_entry)
    ray.init(address=args.ray_address, runtime_env={'env_vars': {
        'PYTHONPATH': os.pathsep.join([str(ROOT/'scripts'), str(ROOT/'evenet_dgpo'), os.environ.get('PYTHONPATH', '')])}})
    if ray.cluster_resources().get('GPU', 0) < 16:
        raise RuntimeError('Requires an existing 16-GPU cluster; no local fallback')
    out.mkdir(parents=True, exist_ok=False)
    torch.save(raw, out/'raw_policy.pt')
    torch.save(panel, out/'panel.pt')
    save_json(out/'manifest.json', cfg)
    (out/'runtime.yaml').write_text(yaml.safe_dump(runtime, sort_keys=False))
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    TorchTrainer(train_loop_per_worker=ray_worker_entry, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=16, use_gpu=True, resources_per_worker={'CPU': 2}),
        run_config=RunConfig(name='precision-screen', storage_path=str(out/'ray_results'),
                            failure_config=FailureConfig(max_failures=0))).fit()
    print(out/'report.json')


if __name__ == '__main__':
    main()
