"""Saved real native training batches shared by the step-1920 trajectory arms.

Capture reads every local Ray shard once, without policy updates. Replay keeps
each rank's original batches and partial tail, cycling independently per rank.
The iterator is infinite: native MIN-exhaustion cannot drop longer shards.
This pairs training inputs; callers must separately pair candidate/timestep/
diffusion noise and dropout using their predeclared per-update RNG stream.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


KIND = 'tau_native_training_batch_replay'
SCHEMA_VERSION = 1
SOURCE_STEP = 1920
EXPECTED_EVENTS = 416701
TAIL_RECIPE = 'Preserve original partial tails; each rank cycles its complete saved batch list independently; no padding, truncation, or synchronized reset'


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return copy.deepcopy(value)
    raise ValueError(f'Native replay cannot preserve unsupported batch value {type(value).__name__}')


def _finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, float):
        return np.isfinite(value)
    return value is None or isinstance(value, (str, bool, int))


def _digest_update(digest, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b'dict')
        for key in sorted(value):
            digest.update(json.dumps(key).encode()); _digest_update(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode())
        digest.update(str(len(value)).encode())
        for item in value:
            _digest_update(digest, item)
    else:
        digest.update(json.dumps([type(value).__name__, value], allow_nan=False).encode())


def batch_fingerprint(batch):
    digest = hashlib.blake2b(digest_size=16)
    _digest_update(digest, batch)
    return digest.hexdigest()


def _sequence_fingerprint(fingerprints):
    digest = hashlib.blake2b(digest_size=16)
    for value in fingerprints:
        digest.update(value.encode())
    return digest.hexdigest()


@contextmanager
def _preserve_rng(device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python, numpy = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python); np.random.set_state(numpy)


def record_native_training_replay(train_shard, loader_cfg, folder, *, rank, world,
                                  device, seed, expected_events=EXPECTED_EVENTS,
                                  source_checkpoint=None, allow_cpu_fixture=False):
    """Save one complete native preprocessed pass; all ranks participate.

Local failures are gathered before any worker raises, so a failed capture does
not leave peers waiting at a success-only collective. A completed manifest is
written only after every shard has validated and global event count matches.
"""
    if world != 16 and not allow_cpu_fixture:
        raise ValueError('Real-case native batch capture requires sixteen workers')
    if not 0 <= rank < world or type(seed) is not int or seed < 0:
        raise ValueError('Declare a valid replay rank and nonnegative integer shuffle seed')
    if not allow_cpu_fixture and (expected_events != EXPECTED_EVENTS or loader_cfg.get('batch_size') != 512):
        raise ValueError('Native replay requires the complete filtered 416701-event population and batch512')
    if expected_events < 1 or int(loader_cfg.get('batch_size', 0)) < 1:
        raise ValueError('Native replay requires positive event count and batch size')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'manifest.json').exists():
        raise ValueError('Native replay manifest already exists; load it explicitly instead of overwriting')
    config = dict(loader_cfg, drop_last=False, local_shuffle_seed=seed+rank)
    json.dumps(config, allow_nan=False)
    batches, sizes, fingerprints, error = [], [], [], None
    try:
        with _preserve_rng(device):
            for raw in train_shard.iter_torch_batches(**config):
                batch = _cpu_copy(raw)
                if (not isinstance(batch, dict) or not torch.is_tensor(batch.get('x'))
                        or batch['x'].ndim < 1 or len(batch['x']) < 1 or not _finite(batch)):
                    raise ValueError('Native capture received an empty/nonfinite batch or missing native x tensor')
                if len(batch['x']) > int(config['batch_size']):
                    raise ValueError('Native capture batch exceeds the declared batch size')
                batches.append(batch); sizes.append(len(batch['x']))
                fingerprints.append(batch_fingerprint(batch))
        if not batches:
            raise ValueError('Native capture received an empty shard')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    local = dict(rank=rank, rows=sum(sizes), batches=len(batches), batch_sizes=sizes,
        partial_batch_indices=[i for i, size in enumerate(sizes) if size < config['batch_size']],
        local_shuffle_seed=seed+rank, replay_fingerprint=_sequence_fingerprint(fingerprints),
        batch_fingerprints=fingerprints, error=error)
    if world > 1:
        rows = torch.tensor(sum(sizes), device=device, dtype=torch.int64)
        dist.all_reduce(rows, op=dist.ReduceOp.SUM)
        global_rows = int(rows.item())
        ranks = [None]*world
        dist.all_gather_object(ranks, local)
    else:
        global_rows, ranks = sum(sizes), [local]
    failures = [row for row in ranks if row['error'] is not None]
    if failures:
        raise ValueError('Native replay capture failed: '+ '; '.join(f"rank{row['rank']}: {row['error']}" for row in failures))
    if (global_rows != expected_events or sum(row['rows'] for row in ranks) != global_rows
            or sorted(row['rank'] for row in ranks) != list(range(world))):
        raise ValueError(f'Native replay must contain every filtered row exactly once by Ray shard count: {global_rows} != {expected_events}')
    payload = dict(kind=KIND, schema_version=SCHEMA_VERSION, complete=True,
        source_policy_step=SOURCE_STEP, source_checkpoint=source_checkpoint, world=world,
        rank=rank, loader_config=config, tail_recipe=TAIL_RECIPE, metadata=local, batches=batches)
    path = folder/f'rank-{rank:02d}.pt'
    write_error = None
    try:
        pending = path.with_suffix('.pending.pt'); torch.save(payload, pending); pending.replace(path)
    except Exception as exc:
        write_error = f'rank{rank}: {type(exc).__name__}: {exc}'
    if world > 1:
        errors = [None]*world
        dist.all_gather_object(errors, write_error)
    else:
        errors = [write_error]
    if any(error is not None for error in errors):
        raise ValueError('Native replay shard write failed: '+ '; '.join(error for error in errors if error is not None))
    manifest = dict(kind=KIND, schema_version=SCHEMA_VERSION, complete=True,
        source_policy_step=SOURCE_STEP, source_checkpoint=source_checkpoint,
        world=world, expected_events=expected_events, global_rows=global_rows,
        batch_size=int(config['batch_size']), base_shuffle_seed=seed,
        ranks=ranks, minimum_rank_batches=min(row['batches'] for row in ranks),
        maximum_rank_batches=max(row['batches'] for row in ranks), tail_recipe=TAIL_RECIPE,
        actor_updates=0, all_native_tensor_fields_preserved=True,
        pairing='Exact saved per-rank batches/order; MC noise must be paired separately by the update observer',
        global_replay_fingerprint=_sequence_fingerprint([row['replay_fingerprint'] for row in ranks]))
    manifest_error = [None]
    if rank == 0:
        try:
            pending = folder/'manifest.pending.json'
            pending.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
            pending.replace(folder/'manifest.json')
        except Exception as exc:
            manifest_error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        dist.broadcast_object_list(manifest_error, src=0)
    if manifest_error[0] is not None:
        raise ValueError('Native replay manifest write failed: '+manifest_error[0])
    return NativeTrainingReplay.load(folder, rank=rank, world=world,
        expected_events=expected_events, source_checkpoint=source_checkpoint,
        allow_cpu_fixture=allow_cpu_fixture)


def ensure_native_training_replay(train_shard, loader_cfg, folder, *, rank, world,
                                  device, seed, expected_events=EXPECTED_EVENTS,
                                  source_checkpoint=None, allow_cpu_fixture=False):
    """Capture once without fitting a classifier, then reuse across method arms.

Rank zero atomically claims a new directory. A concurrent or interrupted
capture is rejected rather than overwritten or silently replaced with live data.
"""
    folder = Path(folder)
    decision = [None]
    if rank == 0:
        try:
            if (folder / 'manifest.json').is_file():
                decision[0] = 'load'
            else:
                folder.parent.mkdir(parents=True, exist_ok=True)
                try:
                    folder.mkdir(exist_ok=False)
                    decision[0] = 'capture'
                except FileExistsError:
                    raise ValueError(f'Replay capture is running or incomplete at {folder}; '
                                     'finish the first launch, or select a new native_training_directory in the YAML')
        except Exception as exc:
            decision[0] = f'error: {exc}'
    if world > 1:
        dist.broadcast_object_list(decision, src=0)
    if decision[0] not in ('load', 'capture'):
        raise ValueError(decision[0])
    kwargs = dict(rank=rank, world=world, expected_events=expected_events,
                  source_checkpoint=source_checkpoint, allow_cpu_fixture=allow_cpu_fixture)
    if decision[0] == 'load':
        replay = NativeTrainingReplay.load(folder, **kwargs)
        if replay.manifest['base_shuffle_seed'] != seed:
            raise ValueError('Shared replay shuffle seed changed')
        return replay
    return record_native_training_replay(train_shard, loader_cfg, folder,
        device=device, seed=seed, **kwargs)


class NativeTrainingReplay:
    def __init__(self, manifest, payload):
        self.manifest, self.payload = manifest, payload
        self.batches = tuple(payload['batches'])
        self.rank, self.world = int(payload['rank']), int(payload['world'])
        self.next_update = 0

    @classmethod
    def load(cls, folder, *, rank, world, expected_events=EXPECTED_EVENTS,
             source_checkpoint=None, allow_cpu_fixture=False):
        if world != 16 and not allow_cpu_fixture:
            raise ValueError('Real-case native batch replay requires sixteen workers')
        folder = Path(folder)
        manifest = json.loads((folder/'manifest.json').read_text())
        if (manifest.get('kind') != KIND or manifest.get('schema_version') != SCHEMA_VERSION
                or not manifest.get('complete') or manifest.get('source_policy_step') != SOURCE_STEP
                or manifest.get('world') != world or manifest.get('global_rows') != expected_events
                or manifest.get('expected_events') != expected_events
                or manifest.get('tail_recipe') != TAIL_RECIPE or not 0 <= rank < world
                or len(manifest.get('ranks', [])) != world):
            raise ValueError('Native replay manifest differs from the predeclared source/population/workers')
        if source_checkpoint is not None and manifest['source_checkpoint'] != source_checkpoint:
            raise ValueError('Native replay source checkpoint changed')
        if not allow_cpu_fixture and (expected_events != EXPECTED_EVENTS or manifest['batch_size'] != 512):
            raise ValueError('Native replay population/batch size differs from filtered train batch512')
        payload = torch.load(folder/f'rank-{rank:02d}.pt', map_location='cpu', weights_only=True)
        if (payload.get('kind') != KIND or payload.get('schema_version') != SCHEMA_VERSION
                or not payload.get('complete') or payload.get('rank') != rank
                or payload.get('world') != world or payload.get('source_policy_step') != SOURCE_STEP
                or payload.get('source_checkpoint') != manifest['source_checkpoint']
                or payload.get('tail_recipe') != TAIL_RECIPE):
            raise ValueError('Native replay shard differs from its manifest')
        batch_size = int(manifest['batch_size'])
        if (payload['loader_config'].get('drop_last') is not False
                or payload['loader_config'].get('batch_size') != batch_size
                or payload['loader_config'].get('local_shuffle_seed') != manifest['base_shuffle_seed']+rank):
            raise ValueError('Native replay loader dropped tails or changed declared order')
        batches = payload['batches']
        if not batches or any(not isinstance(batch, dict) or not _finite(batch)
                              or not torch.is_tensor(batch.get('x')) or batch['x'].ndim < 1
                              or not 0 < len(batch['x']) <= batch_size for batch in batches):
            raise ValueError('Native replay has empty/nonfinite batches')
        sizes = [len(batch['x']) for batch in batches]
        fingerprints = [batch_fingerprint(batch) for batch in batches]
        metadata = payload['metadata']
        if (metadata != manifest['ranks'][rank] or metadata['rank'] != rank
                or metadata['rows'] != sum(sizes) or metadata['batches'] != len(batches)
                or metadata['batch_sizes'] != sizes or metadata['batch_fingerprints'] != fingerprints
                or metadata['replay_fingerprint'] != _sequence_fingerprint(fingerprints)):
            raise ValueError('Native replay batch content/order/count changed')
        if (sum(row['rows'] for row in manifest['ranks']) != expected_events
                or any(row['rank'] != i or row['error'] is not None for i, row in enumerate(manifest['ranks']))
                or manifest['global_replay_fingerprint'] != _sequence_fingerprint(
                    [row['replay_fingerprint'] for row in manifest['ranks']])):
            raise ValueError('Native replay global shard inventory is inconsistent')
        return cls(manifest, payload)

    def fingerprint_for_update(self, relative_update):
        if type(relative_update) is not int or relative_update < 0:
            raise ValueError('Relative replay update must be a nonnegative integer')
        return self.payload['metadata']['batch_fingerprints'][relative_update % len(self.batches)]

    def iterator(self, *, start_update=None):
        """Yield independent copies forever; preserve cursor across renewals."""
        if start_update is not None:
            if type(start_update) is not int or start_update < 0:
                raise ValueError('Replay start_update must be a nonnegative integer')
            self.next_update = start_update
        while True:
            index = self.next_update % len(self.batches)
            self.next_update += 1
            yield _cpu_copy(self.batches[index])
