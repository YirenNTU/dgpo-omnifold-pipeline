#!/usr/bin/env python3
"""Read-only Fourier scale sensitivity on paired validation events and noise."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evenet_dgpo'))
from evenet.utilities.fourier_integration import paired_diffusion_rng


@contextmanager
def branch_scale(branch, scale):
    # Scale only the residual returned by the existing branch; no weight edits.
    handle = branch.register_forward_hook(lambda module, args, output: output * scale)
    try:
        yield
    finally:
        handle.remove()


def load_trained_weights(model, checkpoint):
    source = {k.removeprefix('model.'): v for k, v in checkpoint['state_dict'].items()}
    target = model.state_dict()
    allowed = {'famo', 'Classification', 'Regression', 'Assignment', 'Segmentation',
               'GlobalGeneration', 'ReconGeneration'}
    unexpected = [k for k in source if k not in target and k.split('.')[0] not in allowed]
    missing = [k for k in target if k not in source]
    if missing or unexpected:
        raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    # New readouts also persist their nonzero frequency bank; only the final
    # residual projection establishes that the branch has left zero init.
    projection = source.get('PET.angular_conditioning.projection.weight')
    if projection is None or not torch.count_nonzero(projection).item():
        raise ValueError('Requires a trained nonzero Fourier branch')
    model.load_state_dict({k: source[k] for k in target}, strict=True)


def paired_batch(model, branch, batch, scales, seed, batch_idx, device):
    results = {}
    for scale in [1.0, *(s for s in scales if s != 1.0)]:
        with torch.inference_mode(), branch_scale(branch, scale), paired_diffusion_rng(
            seed, training=False, epoch=0, batch_idx=batch_idx, rank=0, device=device
        ):
            r = model.shared_step(batch, len(batch['x']), {},
                schedules=[('neutrino_generation', True)])['generations']['neutrino']
        for key in ('vector', 'truth', 'time'):
            if not torch.isfinite(r[key]).all():
                raise ValueError(f'Nonfinite {key} at scale {scale}')
        if scale == 1.0:
            reference = {k: r[k].detach().clone() for k in ('vector', 'truth', 'time', 'mask')}
        else:
            for key in ('truth', 'time', 'mask'):
                torch.testing.assert_close(r[key], reference[key], rtol=0, atol=0)
        mask = r['mask'].bool()
        if mask.ndim == r['vector'].ndim - 1:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(r['vector']).clone()
        padding = getattr(model, 'invisible_padding', 0)
        if padding:
            mask[..., -padding:] = False
        dims = tuple(range(1, mask.ndim))
        error = (r['vector'].double() - r['truth'].double()).square()
        shift = (r['vector'].double() - reference['vector'].double()).square()
        # Store additive per-event sufficient statistics, including uneven masks.
        results[scale] = np.stack([
            torch.where(mask, error, 0).sum(dims).cpu().numpy(),
            torch.where(mask, shift, 0).sum(dims).cpu().numpy(),
            mask.sum(dims).cpu().numpy(),
        ], axis=-1)
    return results


def validation_batches(cfg, limit, batch_size, device, manifest):
    import pyarrow.parquet as pq
    from evenet.dataset.preprocess import unflatten_dict
    platform = cfg['platform']
    metadata = json.loads((Path(platform['data_parquet_dir']) / 'shape_metadata.json').read_text())
    seen = 0
    for path in sorted(Path(platform['data_parquet_val_dir']).glob('*.parquet')):
        offset = 0
        for record in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            count = min(record.num_rows, limit - seen)
            if not count:
                continue
            record = record.slice(0, count)
            flat = {name: record.column(i).to_numpy(zero_copy_only=False)
                    for i, name in enumerate(record.schema.names)}
            arrays = unflatten_dict(flat, metadata,
                drop_column_prefix=['EXTRA/', 'regression-', 'assignments-', 'segmentation-'])
            manifest.append(dict(file=str(path), start_row=offset, count=count))
            yield {k: torch.as_tensor(v.copy(), device=device) for k, v in arrays.items()}
            offset += count
            seen += count
            if seen == limit:
                return
    if seen != limit:
        raise ValueError(f'Requested {limit} events but only found {seen}')


def summarize(arrays):
    # Each array is [noise seed, event, (loss sum, velocity delta sum, count)].
    reference = arrays[1.0]
    report = {}
    for scale, a in arrays.items():
        den = a[..., 2].sum()
        if den <= 0:
            raise ValueError('No valid targets')
        delta = a[..., 0] - reference[..., 0]
        report[str(scale)] = dict(
            velocity_mse=float(a[..., 0].sum() / den),
            paired_loss_delta_vs_scale1=float(delta.sum() / den),
            velocity_delta_rms_vs_scale1=float(np.sqrt(a[..., 1].sum() / den)),
            loss_by_noise_seed=(a[..., 0].sum(1) / a[..., 2].sum(1)).tolist(),
            paired_delta_by_noise_seed=(delta.sum(1) / a[..., 2].sum(1)).tolist())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'config/fourier_scale_diagnostic.yaml')
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    scales = [float(s) for s in spec['scales']]
    seeds = spec['noise_seeds']
    if len(set(scales)) != len(scales) or set(scales) != {0., .5, 1., 2.}:
        raise ValueError('Expected unique scales 0, 0.5, 1, 2')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use distinct noise seeds')
    if spec['max_events'] < 1 or spec['batch_size'] < 1:
        raise ValueError('Positive event and batch counts required')
    if args.check_only:
        print(yaml.safe_dump(spec, sort_keys=False))
        return
    from evenet.control.global_config import Config
    from evenet.network.evenet_model import build_evenet_model_from_training_config
    device = torch.device(spec['device'])
    torch.manual_seed(42)
    runtime = Path(spec['runtime_config'])
    cfg = yaml.safe_load(runtime.read_text())
    training = cfg['options']['Training']
    truth = training['Components']['TruthGeneration']
    if training.get('apply_event_weight', False) or truth.get('low_noise_weight', 1.) != 1.:
        raise ValueError('This diagnostic requires the unweighted continuation objective')
    config = Config()
    config.load_yaml(runtime)
    checkpoint_path = Path(spec['checkpoint']).resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    normalization = torch.load(config.options.Dataset.normalization_file,
                               map_location='cpu', weights_only=False)
    model = build_evenet_model_from_training_config(config, normalization, device)
    load_trained_weights(model, checkpoint)
    model.to(device).eval()
    branch = model.PET.angular_conditioning
    output = Path(spec['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    chunks = {s: [[] for _ in seeds] for s in scales}
    for idx, batch in enumerate(validation_batches(cfg, spec['max_events'], spec['batch_size'], device, manifest)):
        for j, seed in enumerate(seeds):
            result = paired_batch(model, branch, batch, scales, seed, idx, device)
            for scale in scales:
                chunks[scale][j].append(result[scale])
        print(f'Evaluated {sum(m["count"] for m in manifest)}/{spec["max_events"]} events', flush=True)
    arrays = {s: np.stack([np.concatenate(c) for c in chunks[s]]) for s in scales}
    report = dict(spec=spec, resolved_checkpoint=str(checkpoint_path),
        checkpoint_epoch=checkpoint.get('epoch'), checkpoint_global_step=checkpoint.get('global_step'),
        events=spec['max_events'], results=summarize(arrays),
        interpretation='Fixed-checkpoint branch reliance, not retraining without Fourier. '
        'Masked velocity MSE on this event/noise panel is not the distributed W&B val/loss panel.')
    np.savez_compressed(output / 'paired_event_statistics.npz', **{str(s): a for s, a in arrays.items()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    (output / 'runtime.yaml').write_text(runtime.read_text())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
