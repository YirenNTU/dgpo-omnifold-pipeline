"""Generate one paired tau candidate on each filtered 10% OmniFold train event."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'evenet_dgpo')]
import numpy as np
import torch

from scripts.sample_1110_cij import validation_pool
from scripts.rescore_film_cij import validate_stack

DEFAULT_TEST_SOURCE = Path('/pscratch/sd/y/yiren/Ztautau/h4_film_step1110_cij/rescore-ca89b70725')
DEFAULT_EVENTS = Path('/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train')
DEFAULT_OUTPUT = Path('/pscratch/sd/y/yiren/Ztautau/h4_step1110_cij_omnifold_train')


@torch.no_grad()
def worker(cfg):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from RL.DGPO_neutrino.sampling import generate_neutrino_candidates
    from scripts.diagnose_h4_spike_coverage import load_raw_state

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    device = ray.train.torch.get_device()
    global_config.load_yaml(cfg['runtime'])
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device).eval()
    checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    load_raw_state(policy, checkpoint, 'step1110')
    del checkpoint
    policy.requires_grad_(False)
    spec = EventPackingSpec.from_dict(cfg['packing_spec'])
    pool = torch.load(Path(cfg['output']) / 'pool.pt', map_location='cpu', weights_only=True)
    positions = torch.arange(rank, len(pool['truth']), world)
    sampler = DDIMSampler(device=device)
    torch.manual_seed(cfg['seed'] + rank)
    draws = []
    for offset in range(0, len(positions), cfg['batch_size']):
        ids = positions[offset:offset + cfg['batch_size']]
        packed = pool['condition'][ids].to(device)
        batch = unpack_event_inputs(packed, spec)
        batch['x_invisible'] = torch.zeros(len(ids), 2, 2, device=device)
        batch['x_invisible_mask'] = torch.ones(len(ids), 2, dtype=torch.bool, device=device)
        generated = generate_neutrino_candidates(policy, batch, sampler, K=1,
            num_ddim_steps=cfg['ddim_steps'], device=device, parallel_chains=1)
        draws.append(generated.permute(1, 0, 2, 3).cpu())
        print(f'[tau sample rank={rank}] {offset+len(ids)}/{len(positions)}', flush=True)
    torch.save(dict(positions=positions, draws=torch.cat(draws)),
        Path(cfg['output']) / f'rank-{rank:03d}.pt')
    ray.train.report({'sampled_events_per_rank':len(positions)})


def merge_shards(output, events, workers):
    seen = []
    draws = torch.empty(events, 1, 2, 2)
    for rank in range(workers):
        shard = torch.load(output / f'rank-{rank:03d}.pt', map_location='cpu', weights_only=True)
        idx = shard['positions'].numpy()
        if shard['draws'].shape != (len(idx), 1, 2, 2):
            raise ValueError(f'Wrong generated shape on rank {rank}')
        seen.extend(idx.tolist())
        draws[idx] = shard['draws']
    if sorted(seen) != list(range(events)) or not torch.isfinite(draws).all():
        raise ValueError('Missing, duplicate or nonfinite generated samples')
    return draws.numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-source', type=Path, default=DEFAULT_TEST_SOURCE)
    p.add_argument('--events', type=Path, default=DEFAULT_EVENTS)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=42929)
    p.add_argument('--run-id')
    p.add_argument('--run-name', default='Generate paired tau samples | 10 percent OmniFold train | step 1110')
    p.add_argument('--wandb-group', default='Conditional ratio closure')
    p.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    args = p.parse_args()
    if min(args.workers, args.batch_size) < 1:
        p.error('Positive workers and batch size required')
    test_manifest = json.loads((args.test_source / 'manifest.json').read_text())
    test_sample = json.loads((Path(test_manifest['samples']) / 'manifest.json').read_text())
    if test_sample.get('weights') != 'raw_state_dict_only' or test_sample.get('candidates') != 1:
        raise ValueError('Test source must use one raw candidate from the step-1110 policy')
    classifier_checkpoint = torch.load(test_manifest['checkpoint'],
        map_location='cpu', weights_only=False)
    stack = validate_stack(classifier_checkpoint)
    spec = stack['reward']['increments'][0]['packing_spec']
    del classifier_checkpoint
    pool = validation_pool(args.events, {'packing_spec':spec})
    n = len(pool['truth'])
    if n < args.workers:
        raise ValueError('More GPU workers than OmniFold train events')
    output = args.output.resolve() / ('sample-' + uuid.uuid4().hex[:10])
    output.mkdir(parents=True)
    torch.save(dict(condition=pool['condition'], truth=pool['truth']), output / 'pool.pt')
    cfg = dict(output=str(output), event_source=str(args.events.resolve()),
        test_source=str(args.test_source.resolve()),
        classifier_checkpoint=str(Path(test_manifest['checkpoint']).resolve()),
        checkpoint=test_sample['checkpoint'], runtime=test_sample['runtime'],
        ddim_steps=int(test_sample['ddim_steps']), weights='raw_state_dict_only',
        candidates=1, weight_mode='joint', pool_mode='omnifold-train',
        packing_spec=spec, events=n, rows_read=pool['rows_read'],
        invalid_target_rows=pool['invalid_target_rows'], workers=args.workers,
        batch_size=args.batch_size, seed=args.seed, classifier_fits=0,
        policy_updates=0, run_name=args.run_name, wandb_group=args.wandb_group)
    (output / 'manifest.json').write_text(json.dumps(cfg, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in cfg.items() if k != 'packing_spec'}, indent=2), flush=True)
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    import wandb
    ray.init(address=args.ray_address, runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(
        (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'evenet_dgpo'), os.environ.get('PYTHONPATH', '')))}})
    if ray.cluster_resources().get('GPU', 0) < args.workers:
        raise ValueError(f'Requested {args.workers} workers but fewer GPUs available')
    with wandb.init(entity='ytchou97-university-of-washington', project='nu2flow-RL',
        id=args.run_id, resume='never', mode='online',
        name=args.run_name,
        group=args.wandb_group, tags=['tau-pair','step1110','16-gpu','K1','sampling'],
        config=cfg, dir=str(output)) as run:
        (output / 'wandb.json').write_text(json.dumps({'id':run.id,'url':run.url}) + '\n')
        run.summary['phase'] = 'sampling'
        try:
            TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg,
                scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=True),
                run_config=RunConfig(name='spin-train-samples',
                    storage_path=str(output / 'ray_results'),
                    failure_config=FailureConfig(max_failures=0))).fit()
            draws = merge_shards(output, n, args.workers)
            arrays = dict(source_sample_index=pool['source_sample_index'].numpy(),
                source_event_key=pool['source_event_key'].numpy(),
                **{key:pool[key].numpy() for key in ('source_file_index','source_event_index') if key in pool},
                truth_deltas=pool['truth'].numpy(), deltas=draws,
                log_ratio=np.zeros((n, 1), dtype=np.float32))
            np.savez_compressed(output / 'candidates.npz', **arrays)
            (output / 'COMPLETE').write_text('paired-tau-train-samples-v1\n')
            run.summary.update({'phase':'complete','events':n,'samples':n,
                'output':str(output)})
            print('READY:', output, 'WANDB:', run.url, flush=True)
        except BaseException:
            run.summary['phase'] = 'failed'
            raise


if __name__ == '__main__':
    main()
