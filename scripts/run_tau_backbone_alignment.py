"""User-launched 16-GPU frozen-trunk tau classifier loss ablation.

Phases: prepare (read-only provenance checks), features (one shared cache),
bce, mmd. Both training arms have a fixed, equal epoch budget; internal
validation BCE selects the checkpoint. No remote jobs are submitted here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')]

from scripts.run_conditional_tau_ratio import read_config, command


def arm_command(cfg, phase):
    import copy
    conf = copy.deepcopy(cfg)
    settings = cfg['alignment']
    conf['experiment']['output'] = str(Path(cfg['experiment']['output'])/phase)
    conf['logger']['wandb']['classifier_run_name'] = settings[f'{phase}_run_name']
    args = command(conf, 'train', os.environ.get('RAY_ADDRESS'))
    return args + ['--backbone-cache', settings['cache'], '--mmd-coefficient',
                   str(settings['mmd_coefficient'] if phase == 'mmd' else 0.)]


def extract(cfg):
    import numpy as np
    import ray
    from scripts.train_conditional_spin_ratio import resolve_sample_source, SOURCE_ID_COLUMNS, source_ids
    from scripts.tau_backbone_alignment import extract_worker, merge_features
    sources = dict(train=str(resolve_sample_source(cfg['experiment']['train_sample_root'])),
                   test=cfg['experiment']['test_source'])
    train = json.loads((Path(sources['train'])/'manifest.json').read_text())
    test = json.loads((Path(sources['test'])/'manifest.json').read_text())
    test_policy = json.loads((Path(test['samples'])/'manifest.json').read_text())
    for key in ('checkpoint', 'ddim_steps', 'weights', 'candidates'):
        if train[key] != test_policy[key]:
            raise ValueError(f'Sample lineage mismatch: {key}')
    if train['weights'] != 'raw_state_dict_only' or train['candidates'] != 1:
        raise ValueError('Requires raw K=1 saved step1110 samples')
    cache = Path(cfg['alignment']['cache'])
    settings = dict(cache=str(cache), sources=sources, checkpoint=train['checkpoint'],
        runtime=train['runtime'], packing_spec=train['packing_spec'],
        weights=train['weights'], workers=cfg['classifier']['workers'],
        feature_batch_size=cfg['alignment']['feature_batch_size'],
        extraction='native clean candidate t=0, pre-velocity Linear, last two tokens; frozen eval',
        policy_updates=0, global_step=1110)
    if (cache/'COMPLETE').exists():
        old = json.loads((cache/'manifest.json').read_text())
        if old != settings:
            raise ValueError('Existing cache uses another extraction setup')
        print('Reusing complete shared feature cache:', cache, flush=True)
        return
    cache.mkdir(parents=True, exist_ok=True)
    (cache/'manifest.json').write_text(json.dumps(settings, indent=2)+'\n')
    ray.init(address=os.environ.get('RAY_ADDRESS') or 'auto', runtime_env={'env_vars':
        {'PYTHONPATH':os.pathsep.join((str(ROOT), str(ROOT/'scripts'), str(ROOT/'evenet_dgpo')))}})
    if ray.cluster_resources().get('GPU', 0) < settings['workers']:
        raise ValueError('Insufficient GPUs for the configured feature workers')
    # Tasks reserve one GPU each. No gradients or collectives are needed for
    # extraction; downstream head fitting uses TorchTrainer/DDP on all GPUs.
    worker = ray.remote(num_gpus=1, num_cpus=1, max_calls=1)(extract_worker)
    ray.get([worker.remote(settings, rank) for rank in range(settings['workers'])])
    for label, source in sources.items():
        meta = json.loads((Path(source)/'manifest.json').read_text())
        a, b = merge_features(cache, label, int(meta['events']), settings['workers'])
        with np.load(Path(source)/'candidates.npz') as bundle:
            keys = [key for key in SOURCE_ID_COLUMNS if key in bundle]
            identities = source_ids(bundle, keys)
        np.savez(cache/f'{label}.npz', truth=a, generated=b, source_ids=identities)
    (cache/'COMPLETE').write_text('raw step1110 frozen features complete\n')
    print('FEATURE CACHE READY:', cache, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('phase', choices=('prepare','features','bce','mmd'))
    args = p.parse_args()
    cfg = read_config(args.config)
    for phase in ('bce', 'mmd'):
        name = cfg['alignment'][f'{phase}_run_name']
        if not isinstance(name, str) or len(name) > 96 or len(name.split(' | ')) < 3:
            raise ValueError('Use a readable three-part W&B arm name under 96 characters')
    if cfg['classifier']['workers'] != 16:
        raise ValueError('This matched production ablation is configured for 16 GPUs')
    if cfg['classifier']['condition_normalization'] != 'masked_feature':
        raise ValueError('Use fixed fit-only masked feature normalization for the joint kernel')
    if cfg['classifier']['patience'] <= cfg['classifier']['epochs']:
        raise ValueError('Use patience > epochs for a matched fixed-budget loss ablation')
    if args.phase in ('prepare','features'):
        # Includes full parquet/source pairing checks BEFORE any expensive work.
        subprocess.run(command(cfg, 'prepare'), cwd=ROOT, check=True)
        if args.phase == 'features':
            extract(cfg)
    else:
        subprocess.run(arm_command(cfg, args.phase), cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
