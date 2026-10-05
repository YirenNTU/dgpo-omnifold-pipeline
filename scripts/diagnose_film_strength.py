#!/usr/bin/env python3
"""Inference-only real-event FiLM interventions on 16 GPUs of an existing Ray cluster."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import uuid
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evenet_dgpo'))
NAME = 'Does FiLM reach velocity? | real events | layer and noise sensitivity'
SWEEP_NAME = 'Does weaker scale improve velocity? | global FiLM | paired gain sweep'
GAINS = [0., .25, .5, 1.]
TIMES = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def noise_for(positions, seed):
    return torch.stack([torch.randn(2, 2, generator=torch.Generator().manual_seed(seed + int(i)))
                        for i in positions])


class FilmProbe:
    """Temporary hooks; never edit weights or the production forward implementation."""
    def __init__(self, head, intervention='baseline', record=False, scale_gain=None):
        self.head, self.intervention, self.record = head, intervention, record
        if scale_gain is not None and (not isinstance(scale_gain, (int, float)) or not 0 <= scale_gain <= 1):
            raise ValueError('scale_gain must be finite and between zero and one')
        self.scale_gain = scale_gain
        self.handles, self.values, self.active = [], {}, {}

    def add(self, key, value, mask):
        if not self.record:
            return
        # Preserve event and invisible-slot dimensions. Exactly two valid tau slots.
        selected = value[mask.squeeze(-1).bool()].reshape(value.shape[0], 2, -1)
        if not torch.isfinite(selected).all():
            raise ValueError(f'Nonfinite diagnostic: {key}')
        self.values[key] = selected.detach().square().mean(-1).cpu()

    def pre(self, index, module, args, kwargs):
        parts = kwargs.get('modulation')
        mask = kwargs.get('modulation_mask')
        if parts is None or mask is None or not (mask.squeeze(-1).sum(-1) == 2).all():
            raise ValueError('Expected global FiLM with two valid invisible slots')
        parts = list(parts)
        if self.scale_gain is not None:
            for j in (0, 2):
                parts[j] = parts[j] * self.scale_gain
        targeted = self.intervention in ('no_scale', 'no_shift', 'no_film') or self.intervention == f'no_block_{index}'
        if targeted:
            indices = (0, 2) if self.intervention == 'no_scale' else ((1, 3) if self.intervention == 'no_shift' else range(4))
            for j in indices:
                parts[j] = torch.zeros_like(parts[j])
        self.active[index] = (parts, mask)
        if self.record:
            for j, label in enumerate(('attn_scale', 'attn_shift', 'mlp_scale', 'mlp_shift')):
                expanded = parts[j][:, None, :].expand(-1, mask.shape[1], -1)
                self.add(f'block{index}/{label}', expanded, mask)
        return args, {**kwargs, 'modulation': tuple(parts)}

    def norm(self, index, branch, output):
        parts, mask = self.active[index]
        offset = 0 if branch == 'attn' else 2
        delta = output * parts[offset][:, None, :] + parts[offset + 1][:, None, :]
        self.add(f'block{index}/{branch}_normalized', output, mask)
        self.add(f'block{index}/{branch}_film_delta', delta, mask)

    def residual(self, index, label, output):
        mask = self.active[index][1]
        self.add(f'block{index}/{label}', output[0] if isinstance(output, tuple) else output, mask)

    def __enter__(self):
        for i, block in enumerate(self.head.gen_transformer_blocks):
            self.handles.append(block.register_forward_pre_hook(
                lambda m, a, k, i=i: self.pre(i, m, a, k), with_kwargs=True))
            if self.record:
                for name, branch in [('norm1', 'attn'), ('norm3', 'mlp')]:
                    self.handles.append(getattr(block, name).register_forward_hook(
                        lambda m, a, o, i=i, branch=branch: self.norm(i, branch, o)))
                for name, label in [('attn', 'attn_pre_layerscale'), ('mlp', 'mlp_pre_layerscale')]:
                    self.handles.append(getattr(block, name).register_forward_hook(
                        lambda m, a, o, i=i, label=label: self.residual(i, label, o)))
                if block.layer_scale_flag:
                    for name, label in [('layer_scale1', 'attn_post_layerscale'), ('layer_scale2', 'mlp_post_layerscale')]:
                        self.handles.append(getattr(block, name).register_forward_hook(
                            lambda m, a, o, i=i, label=label: self.residual(i, label, o)))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def interventions(cfg):
    if cfg.get('scale_sweep'):
        return [(f'gain_{gain:g}', 'baseline', gain) for gain in GAINS]
    return [(arm, arm, None) for arm in
            ['no_scale', 'no_shift', 'no_film', 'no_block_0', 'no_block_1', 'no_block_2']]


def paired_mse_stats(values, arms, seed, replicates=2000):
    """Pointwise event bootstrap; keep both slots together in every resample."""
    base = values['mse/baseline'].double()
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(base), (replicates, len(base)), generator=rng)
    result = {}
    for arm, _, _ in arms:
        current = values[f'mse/{arm}'].double()
        delta = (current-base).mean(-1)
        ci = torch.quantile(delta[indices].mean(1), torch.tensor([.025,.975], dtype=torch.float64))
        result[arm] = dict(mse=current.mean().item(), baseline_mse=base.mean().item(),
            delta_mse=delta.mean().item(), delta_ci95=ci.tolist(),
            relative_mse_change=delta.mean().item()/max(base.mean().item(),1e-12))
    return result


def summarize(values):
    """Means of per-event squared magnitudes, with separate tau slots."""
    return {k: {'rms_by_slot': v.double().mean(0).sqrt().tolist(),
                'event_rms_quantiles_by_slot': torch.quantile(v.double().sqrt(),
                    torch.tensor([.1, .5, .9, .99], dtype=torch.float64), dim=0).tolist()}
            for k, v in values.items()}


def merge_parts(parts, events):
    positions = torch.cat([p['positions'] for p in parts])
    if not torch.equal(positions.sort().values, torch.arange(events)):
        raise ValueError('Distributed panel has duplicate or missing events')
    order = positions.argsort()
    keys = parts[0]['values'].keys()
    if any(p['values'].keys() != keys for p in parts):
        raise ValueError('Distributed metric keys differ')
    return {key: torch.cat([p['values'][key] for p in parts])[order] for key in keys}


def validate_panel(panel):
    from evaluate_global_film_coverage import validate_panel as validate
    validate(panel, len(panel['truth']))
    if 'packing_spec' not in panel:
        raise ValueError('Missing packing spec')
    if panel['truth'].dtype != torch.float32:
        raise ValueError('Expected raw physical FP32 target corrections')


def worker(cfg):
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from evenet.utilities.diffusion_sampler import get_logsnr_alpha_sigma
    from diagnose_h4_spike_coverage import load_raw_state
    import ray.train
    import ray.train.torch
    import torch.distributed as dist
    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    if world != 16:
        raise ValueError('This diagnostic requires exactly 16 workers')
    out = Path(cfg['output'])
    global_config.load_yaml(str(out / 'runtime.yaml'))
    device = ray.train.torch.get_device()
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device)
    load_raw_state(policy, torch.load(out / 'raw_policy.pt', map_location='cpu', weights_only=True), 'pretrained10pct')
    policy.eval().requires_grad_(False)
    if policy.invisible_input_dim != 2 or len(policy.TruthGeneration.gen_transformer_blocks) != 3:
        raise ValueError('Expected three blocks and two correction coordinates')
    panel = torch.load(out / 'panel.pt', map_location='cpu', weights_only=True)
    spec = EventPackingSpec.from_dict(panel['packing_spec'])
    run = None
    if rank == 0 and cfg['wandb']:
        import wandb
        run = wandb.init(project='EveNet', entity='ytchou97-university-of-washington',
            id=cfg['run_id'], resume='never', name=cfg.get('run_name', NAME), group='FiLM effective strength',
            tags=['InferenceOnly', 'GlobalFiLM', 'RealEvents', 'PairedNoise'], config=cfg)
    arms = interventions(cfg)
    success = False
    report = dict(config=cfg, times=[], interpretation='Forward-noised truth sensitivity, not a sampling trajectory, retraining ablation or closure test.')
    try:
        with torch.inference_mode():
            for t in cfg['times']:
                chunks = {}
                positions = torch.arange(rank, len(panel['truth']), world)
                for start in range(0, len(positions), cfg['batch_size']):
                    ids = positions[start:start + cfg['batch_size']]
                    batch = unpack_event_inputs(panel['condition'][ids].to(device), spec)
                    batch = {k: v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in batch.items()}
                    truth = panel['truth'][ids].reshape(-1, 2, 2).to(device)
                    mask = torch.ones(len(ids), 2, 1, device=device)
                    x0 = policy.invisible_normalizer(x=truth, mask=mask)
                    eps = noise_for(ids, cfg['seed']).to(device)
                    time = torch.full((len(ids),), t, device=device)
                    _, alpha, sigma = get_logsnr_alpha_sigma(time, (-1, 1, 1))
                    xt, target = alpha*x0 + sigma*eps, alpha*eps - sigma*x0
                    def predict():
                        return policy.predict_diffusion_vector(xt, batch, time, 'neutrino', noise_mask=mask)
                    baseline = predict()
                    with FilmProbe(policy.TruthGeneration, record=True) as probe:
                        hooked = predict()
                    if not torch.equal(baseline, hooked):
                        raise ValueError('Recording hooks changed baseline output')
                    values = dict(probe.values)
                    values['velocity/baseline'] = baseline.square().mean(-1).cpu()
                    values['mse/baseline'] = (baseline-target).square().mean(-1).cpu()
                    for arm, intervention, gain in arms:
                        with FilmProbe(policy.TruthGeneration, intervention, scale_gain=gain):
                            v = predict()
                        if gain == 1. and not torch.equal(v, baseline):
                            raise ValueError('Gain one must reproduce baseline exactly')
                        values[f'velocity_delta/{arm}'] = (v-baseline).square().mean(-1).cpu()
                        values[f'mse/{arm}'] = (v-target).square().mean(-1).cpu()
                    if not torch.equal(baseline, predict()):
                        raise ValueError('Intervention failed to restore baseline')
                    for k, v in values.items():
                        if not torch.isfinite(v).all():
                            raise ValueError(f'Nonfinite {k}')
                        chunks.setdefault(k, []).append(v)
                values = {k: torch.cat(v) for k, v in chunks.items()}
                gathered = [None] * world if rank == 0 else None
                dist.gather_object({'positions': positions, 'values': values}, gathered, dst=0)
                if rank != 0:
                    continue
                values = merge_parts(gathered, len(panel['truth']))
                torch.save({'test_rows': panel['test_rows'], 'time': t, 'mean_squares': values}, out / f'time-{t:.3f}.pt')
                metrics = summarize(values)
                strength = {}
                for layer in range(3):
                    for branch in ['attn', 'mlp']:
                        prefix = f'block{layer}/{branch}'
                        strength[prefix] = (values[prefix+'_film_delta'].double().mean(0).sqrt() /
                            values[prefix+'_normalized'].double().mean(0).sqrt().clamp_min(1e-12)).tolist()
                row = {'time': t, 'film_relative_rms_by_slot': strength, 'metrics': metrics, 'mse_change_by_slot': {}, 'velocity_relative_rms_by_slot': {}}
                for arm, intervention, gain in arms:
                    row['mse_change_by_slot'][arm] = (values[f'mse/{arm}']-values['mse/baseline']).double().mean(0).tolist()
                    row['velocity_relative_rms_by_slot'][arm] = (values[f'velocity_delta/{arm}'].double().mean(0).sqrt() /
                        values['velocity/baseline'].double().mean(0).sqrt().clamp_min(1e-12)).tolist()
                row['paired_mse'] = paired_mse_stats(values, arms, cfg['seed'])
                row['interval_scope'] = 'Pointwise 95% event-bootstrap intervals; not corrected for gain/time selection'
                report['times'].append(row)
                save_json(out / 'report.json', report)
                print(json.dumps({'time': t, 'mse_change_by_slot': row['mse_change_by_slot']}), flush=True)
                if run:
                    log = {'noise_time': t}
                    for family in ['mse_change_by_slot', 'velocity_relative_rms_by_slot', 'film_relative_rms_by_slot']:
                        for arm, slots in row[family].items():
                            for slot, value in enumerate(slots):
                                log[f'{family}/{arm}/slot{slot}'] = value
                    for arm, stats in row['paired_mse'].items():
                        for key, value in stats.items():
                            if key == 'delta_ci95':
                                log[f'paired_mse/{arm}/ci95_low'], log[f'paired_mse/{arm}/ci95_high'] = value
                            else:
                                log[f'paired_mse/{arm}/{key}'] = value
                    run.log(log)
        if rank == 0:
            report['complete'] = True
            save_json(out / 'report.json', report)
            (out / 'COMPLETE').write_text(cfg['schema']+'\n')
            if run:
                run.summary['evaluation/complete'] = True
                run.save(str(out / 'report.json'), base_path=str(out), policy='now')
                for saved in out.glob('time-*.pt'):
                    run.save(str(saved), base_path=str(out), policy='now')
        dist.barrier()
        success = True
    finally:
        if run:
            run.finish(exit_code=0 if success else 1)
    return str(out / 'report.json')


def main(argv=None):
    from evaluate_global_film_coverage import inspect_checkpoint, runtime_configuration
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--panel', type=Path, help='Existing held-out joint-coverage panel.pt')
    p.add_argument('--panel-report', type=Path, default=Path('/pscratch/sd/y/yiren/Ztautau/diffusion_low_noise_lr_10pct_seed42/checkpoints/joint_coverage/epoch-0100.json'), help='Read only config.panel_path when --panel is omitted')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--scale-sweep', action='store_true', help='Scale gamma by 0, .25, .5, 1; leave beta unchanged')
    p.add_argument('--seed', type=int, default=42017)
    p.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    cfg = dict(schema='film-strength-v1', checkpoint_path=str(args.checkpoint), panel_path=str(args.panel) if args.panel else None, panel_report=str(args.panel_report),
        output=str(args.output.resolve()), batch_size=args.batch_size, seed=args.seed, times=TIMES,
        wandb=not args.no_wandb, run_id=uuid.uuid4().hex[:8], policy_updates=0, classifier_fits=0,
        panel_type='fixed held-out forward-noised truth', workers=16)
    cfg.update(scale_sweep=args.scale_sweep, gains=GAINS if args.scale_sweep else None,
               run_name=SWEEP_NAME if args.scale_sweep else NAME)
    if args.scale_sweep:
        cfg['schema'] = 'film-scale-sweep-v1'
    runtime = runtime_configuration()
    runtime['logger']['wandb'].update(project='EveNet', run_name=cfg['run_name'], group='FiLM effective strength')
    runtime['experiment'] = dict(question=cfg['run_name'], policy_updates=0, classifier_fits=0)
    runtime['nersc']['execution'] = dict(mode='inference_only_existing_ray_cluster')
    if args.dry_run:
        print(yaml.safe_dump({'diagnostic': cfg, 'runtime': runtime}, sort_keys=False))
        return
    panel_path = args.panel or Path(json.loads(args.panel_report.read_text())['config']['panel_path'])
    cfg['panel_path'] = str(panel_path.resolve(strict=True))
    panel = torch.load(panel_path, map_location='cpu', weights_only=True)
    validate_panel(panel)
    cfg['events'] = len(panel['truth'])
    if cfg['events'] < 16:
        raise ValueError('Panel must contain at least 16 events')
    metadata, raw = inspect_checkpoint(args.checkpoint)
    cfg['checkpoint'] = metadata
    out = Path(cfg['output'])
    if out.exists():
        p.error('Use a new output directory')
    if args.check_only:
        print(json.dumps(cfg, indent=2))
        return
    if int(os.environ.get('SLURM_PROCID', '0')) != 0:
        p.error('Launch once only')
    import ray
    ray.init(address=args.ray_address, runtime_env={'env_vars': {
        'PYTHONPATH': os.pathsep.join([str(ROOT / 'scripts'), str(ROOT / 'evenet_dgpo'), os.environ.get('PYTHONPATH', '')])}})
    if ray.cluster_resources().get('GPU', 0) < 16:
        raise RuntimeError('Requires an existing 16-GPU allocation; no local fallback')
    out.mkdir(parents=True, exist_ok=False)
    torch.save(raw, out / 'raw_policy.pt')
    torch.save(panel, out / 'panel.pt')
    save_json(out / 'manifest.json', cfg)
    (out / 'runtime.yaml').write_text(yaml.safe_dump(runtime, sort_keys=False))
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=16, use_gpu=True, resources_per_worker={'CPU': 2}),
        run_config=RunConfig(name='film-strength', storage_path=str(out / 'ray_results'),
                             failure_config=FailureConfig(max_failures=0))).fit()
    print(out / 'report.json')


if __name__ == '__main__':
    main()
