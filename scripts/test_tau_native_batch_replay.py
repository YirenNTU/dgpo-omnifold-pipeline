"""CPU checks for saved native inputs and partial-tail replay, no Ray/jobs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'evenet_dgpo'))
from RL.DGPO_neutrino.tau_native_batch_replay import (
    NativeTrainingReplay, batch_fingerprint, record_native_training_replay, ensure_native_training_replay,
    _sequence_fingerprint,
)


def native_batch(start, rows):
    return dict(x=torch.arange(start, start+rows*8, dtype=torch.float32).reshape(rows, 2, 4),
        x_mask=torch.ones(rows, 2, dtype=torch.bool),
        x_invisible=torch.full((rows, 2, 2), start+.5),
        x_invisible_mask=torch.ones(rows, 2, dtype=torch.bool),
        classification=torch.arange(rows, dtype=torch.int64)+3,
        event_category=torch.full((rows,), 11, dtype=torch.int64),
        event_valid=torch.ones(rows, dtype=torch.float32),
        event_weight=torch.arange(1., rows+1),
        lead_a_visible_E=torch.arange(rows, dtype=torch.float32)+100.,
        static=torch.tensor(7.), metadata={'population': 'filtered', 'shape': (2, 4)})


class Shard:
    def __init__(self, batches):
        self.batches, self.calls = batches, []
    def iter_torch_batches(self, **kwargs):
        self.calls.append(kwargs)
        random.random(); np.random.random(); torch.rand(1)
        yield from self.batches


def capture(tmp_path, *, batches=None, seed=101):
    batches = batches if batches is not None else [native_batch(0, 4), native_batch(32, 3)]
    shard = Shard(batches)
    replay = record_native_training_replay(shard, dict(batch_size=4, prefetch_batches=1,
        local_shuffle_buffer_size=4), tmp_path, rank=0, world=1, device='cpu', seed=seed,
        expected_events=sum(len(batch['x']) for batch in batches),
        source_checkpoint='/pinned/policy1920.ckpt', allow_cpu_fixture=True)
    return replay, shard


def test_capture_preserves_every_native_tensor_dtype_value_and_rng(tmp_path):
    before_torch, before_numpy, before_python = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    replay, shard = capture(tmp_path)
    assert shard.calls == [dict(batch_size=4, prefetch_batches=1, local_shuffle_buffer_size=4,
                                drop_last=False, local_shuffle_seed=101)]
    assert replay.manifest['global_rows'] == 7 and replay.manifest['actor_updates'] == 0
    assert replay.manifest['ranks'][0]['batch_sizes'] == [4, 3]
    assert replay.manifest['ranks'][0]['partial_batch_indices'] == [1]
    for saved, original in zip(replay.batches, shard.batches):
        assert saved.keys() == original.keys()
        for key in saved:
            if torch.is_tensor(saved[key]):
                assert saved[key].dtype == original[key].dtype
                torch.testing.assert_close(saved[key], original[key], rtol=0, atol=0)
        assert saved['metadata'] == original['metadata']
        assert batch_fingerprint(saved) == batch_fingerprint(original)
    torch.testing.assert_close(torch.get_rng_state(), before_torch)
    for first, second in zip(np.random.get_state(), before_numpy):
        np.testing.assert_equal(first, second)
    assert random.getstate() == before_python


def test_replay_is_infinite_clones_inputs_and_preserves_cursor_between_iterators(tmp_path):
    replay, _ = capture(tmp_path)
    stream = replay.iterator()
    first = next(stream); first['x'].fill_(-99)
    assert len(next(stream)['x']) == 3
    assert len(next(stream)['x']) == 4
    assert replay.batches[0]['x'][0, 0, 0] == 0
    renewed = replay.iterator()
    assert len(next(renewed)['x']) == 3
    reset = replay.iterator(start_update=100)
    assert batch_fingerprint(next(reset)) == replay.fingerprint_for_update(100)
    for step in range(101, 111):
        assert batch_fingerprint(next(reset)) == replay.fingerprint_for_update(step)
    with pytest.raises(ValueError):
        replay.fingerprint_for_update(-1)


def test_missing_population_nan_empty_and_overwrite_fail_without_complete_manifest(tmp_path):
    shard = Shard([native_batch(0, 3)])
    with pytest.raises(ValueError, match='every filtered row'):
        record_native_training_replay(shard, dict(batch_size=4), tmp_path/'wrong', rank=0,
            world=1, device='cpu', seed=1, expected_events=4, allow_cpu_fixture=True)
    assert not (tmp_path/'wrong/manifest.json').exists()
    with pytest.raises(ValueError, match='empty shard'):
        record_native_training_replay(Shard([]), dict(batch_size=4), tmp_path/'empty', rank=0,
            world=1, device='cpu', seed=1, expected_events=4, allow_cpu_fixture=True)
    bad = native_batch(0, 3); bad['x_invisible'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        capture(tmp_path/'nan', batches=[bad])
    assert not (tmp_path/'nan/manifest.json').exists()
    capture(tmp_path/'valid')
    with pytest.raises(ValueError, match='already exists'):
        capture(tmp_path/'valid')
    with pytest.raises(ValueError, match='sixteen'):
        record_native_training_replay(shard, dict(batch_size=512), tmp_path/'real', rank=0,
            world=1, device='cpu', seed=1)


def test_loaded_replay_rejects_mutated_native_truth_order_or_manifest(tmp_path):
    capture(tmp_path)
    path = tmp_path/'rank-00.pt'
    original = torch.load(path, map_location='cpu', weights_only=True)
    changed = copy.deepcopy(original)
    changed['batches'][1]['x_invisible'][0, 0, 0] += 1
    torch.save(changed, path)
    with pytest.raises(ValueError, match='content/order/count'):
        NativeTrainingReplay.load(tmp_path, rank=0, world=1, expected_events=7, allow_cpu_fixture=True)
    changed = copy.deepcopy(original); changed['batches'].reverse()
    torch.save(changed, path)
    with pytest.raises(ValueError, match='content/order/count'):
        NativeTrainingReplay.load(tmp_path, rank=0, world=1, expected_events=7, allow_cpu_fixture=True)
    torch.save(original, path)
    with pytest.raises(ValueError, match='checkpoint'):
        NativeTrainingReplay.load(tmp_path, rank=0, world=1, expected_events=7,
            source_checkpoint='/wrong.ckpt', allow_cpu_fixture=True)
    manifest = json.loads((tmp_path/'manifest.json').read_text()); manifest['complete'] = False
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        NativeTrainingReplay.load(tmp_path, rank=0, world=1, expected_events=7, allow_cpu_fixture=True)


def test_unequal_rank_counts_have_explicit_tail_recipe_and_never_drop_longer_rank(tmp_path):
    first, _ = capture(tmp_path/'rank0', batches=[native_batch(0, 4), native_batch(32, 3)], seed=101)
    second, _ = capture(tmp_path/'rank1', batches=[native_batch(100, 4), native_batch(132, 4), native_batch(164, 1)], seed=102)
    combined = tmp_path/'combined'; combined.mkdir()
    payloads = [copy.deepcopy(first.payload), copy.deepcopy(second.payload)]
    ranks = []
    for rank, payload in enumerate(payloads):
        payload['world'], payload['rank'] = 2, rank
        payload['metadata']['rank'] = rank
        ranks.append(payload['metadata'])
        torch.save(payload, combined/f'rank-{rank:02d}.pt')
    manifest = copy.deepcopy(first.manifest)
    manifest.update(world=2, expected_events=16, global_rows=16, ranks=ranks,
        minimum_rank_batches=2, maximum_rank_batches=3,
        global_replay_fingerprint=_sequence_fingerprint([row['replay_fingerprint'] for row in ranks]))
    (combined/'manifest.json').write_text(json.dumps(manifest))
    streams = [NativeTrainingReplay.load(combined, rank=rank, world=2, expected_events=16,
                                        allow_cpu_fixture=True).iterator() for rank in range(2)]
    assert [len(next(streams[0])['x']) for _ in range(6)] == [4, 3, 4, 3, 4, 3]
    assert [len(next(streams[1])['x']) for _ in range(6)] == [4, 4, 1, 4, 4, 1]
    assert 'independently' in manifest['tail_recipe'] and 'no padding' in manifest['tail_recipe']


def test_record_uses_global_count_allreduce_and_collects_rank_errors(tmp_path):
    # Exercise the distributed capture branch without starting a process group.
    shard = Shard([native_batch(0, 3)])
    reductions = []
    def reduce(value, op):
        reductions.append(value.item()); value.fill_(6)
    def gather(output, local):
        output[:] = [local, dict(local, rank=1, error='ValueError: invalid other shard')]
    with patch('RL.DGPO_neutrino.tau_native_batch_replay.dist.all_reduce', reduce), patch(
            'RL.DGPO_neutrino.tau_native_batch_replay.dist.all_gather_object', gather):
        with pytest.raises(ValueError, match='rank1: ValueError'):
            record_native_training_replay(shard, dict(batch_size=4), tmp_path, rank=0,
                world=2, device='cpu', seed=1, expected_events=6, allow_cpu_fixture=True)
    assert reductions == [3] and not (tmp_path/'manifest.json').exists()


def test_failed_shard_write_never_marks_the_replay_complete(tmp_path):
    with patch('RL.DGPO_neutrino.tau_native_batch_replay.torch.save', side_effect=OSError('fixture disk full')):
        with pytest.raises(ValueError, match='shard write failed'):
            capture(tmp_path)
    assert not (tmp_path/'manifest.json').exists()


def test_ensure_captures_once_reuses_and_rejects_incomplete_or_changed_seed(tmp_path):
    shard = Shard([native_batch(0, 4), native_batch(32, 3)])
    kwargs = dict(rank=0, world=1, device='cpu', seed=101, expected_events=7,
                  source_checkpoint='/policy1920.ckpt', allow_cpu_fixture=True)
    folder = tmp_path/'shared'
    first = ensure_native_training_replay(shard, dict(batch_size=4), folder, **kwargs)
    second = ensure_native_training_replay(shard, dict(batch_size=4), folder, **kwargs)
    assert len(shard.calls) == 1
    assert first.manifest == second.manifest
    with pytest.raises(ValueError, match='shuffle seed'):
        ensure_native_training_replay(shard, dict(batch_size=4), folder, **dict(kwargs, seed=102))
    with pytest.raises(ValueError, match='source'):
        ensure_native_training_replay(shard, dict(batch_size=4), folder, **dict(kwargs, source_checkpoint='/other.ckpt'))
    incomplete = tmp_path/'incomplete'; incomplete.mkdir()
    with pytest.raises(ValueError, match='running or incomplete'):
        ensure_native_training_replay(shard, dict(batch_size=4), incomplete, **kwargs)
    assert len(shard.calls) == 1
