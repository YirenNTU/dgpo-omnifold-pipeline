"""Frozen raw step1110 features and fixed-geometry joint tau alignment.

No new sampling, physics-label loss, or policy updates. The MMD is a biased,
self-normalized minibatch V-statistic, averaged over DDP ranks, not an exact
conditional-density guarantee. Its geometry cannot adapt with the classifier.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def joint_mmd(condition, truth, generated, logits, base_weight):
    """Multi-scale product RBF on fixed (c, tau); globally paired within batch.

    c is fit-only mask-normalized, tau is the explicit 15-vector (not learned
    embeddings). RMS distances keep condition width from setting the scale.
    There is no per-condition ratio normalization: K=1 would erase all signal.
    """
    c, a, b = (x.detach().float() for x in (condition, truth, generated))
    weight = base_weight.detach().float()
    if not bool(weight.sum() > 0):
        return logits.sum() * 0
    logw = torch.where(weight > 0, weight.log(), -torch.inf)
    p = torch.softmax(logw, 0)
    q = torch.softmax(logw + logits.float(), 0)
    dc = torch.cdist(c, c).square() / c.shape[1]
    aa = torch.cdist(a, a).square() / a.shape[1]
    bb = torch.cdist(b, b).square() / b.shape[1]
    ab = torch.cdist(a, b).square() / a.shape[1]
    loss = logits.new_zeros((), dtype=torch.float32)
    for scale in (0.25, 1., 4.):
        k = torch.exp(-dc / (2 * scale))
        loss = loss + (p @ (k * torch.exp(-aa / (2 * scale))) @ p
            + q @ (k * torch.exp(-bb / (2 * scale))) @ q
            - 2 * p @ (k * torch.exp(-ab / (2 * scale))) @ q) / 3
    return loss


@torch.no_grad()
def alignment_report(condition, truth, generated, logits, weight, batch_size=256):
    # A fixed event permutation, identical across arms. Report the statistic
    # at its training batch scale, plus the change under the actual raw ratio.
    order = np.random.default_rng(814).permutation(len(condition))
    values, sizes = [], []
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        if len(idx) < 2 or weight[idx].sum() <= 0:
            continue
        c, a, b, q, w = [torch.as_tensor(x[idx], dtype=torch.float32)
                         for x in (condition, truth, generated, logits, weight)]
        values.append([joint_mmd(c, a, b, torch.zeros_like(q), w).item(),
                       joint_mmd(c, a, b, q, w).item()])
        sizes.append(len(idx))
    raw, weighted = np.average(values, weights=sizes, axis=0)
    # Condition mean drift uses full-population weights, not batch-normalized.
    w = np.asarray(weight, dtype=np.float64)
    logw = np.full(len(w), -np.inf)
    np.log(w, out=logw, where=w > 0)
    z = logw + logits
    q = np.exp(z - z.max()); q /= q.sum()
    drift = (q - w/w.sum()) @ condition
    return dict(joint_mmd_unweighted=float(raw), joint_mmd_reweighted=float(weighted),
        joint_mmd_change=float(weighted-raw), condition_mean_drift_rms=float(np.sqrt(np.mean(drift**2))),
        scope='Fixed-geometry paired minibatch MMD; not full conditional closure', batch_size=batch_size)


@torch.no_grad()
def candidate_hidden(policy, batch, candidate):
    """Use native inference path, clean candidate at t=0; never add noise.

    Hook only observes the input of the velocity output Linear. No model
    parameters are replaced, and truth/generated labels never enter the trunk.
    """
    policy.eval()
    mask = torch.ones((*candidate.shape[:2], 1), device=candidate.device)
    normalizer = policy.invisible_coordinate_normalizer(batch)
    pad = policy.invisible_normalizer.mean.numel() - candidate.shape[-1]
    if pad < 0:
        raise ValueError('Candidate width exceeds native normalizer')
    normalized = normalizer(torch.nn.functional.pad(candidate, (0, pad)), mask=mask)
    captured = []
    handle = policy.TruthGeneration.generator.register_forward_pre_hook(
        lambda module, args: captured.append(args[0][:, -2:].detach().clone()))
    try:
        policy.predict_diffusion_vector(normalized, batch,
            torch.zeros(len(candidate), device=candidate.device), 'neutrino', noise_mask=mask)
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.isfinite(captured[0]).all():
        raise ValueError('Expected one finite pre-velocity hidden tensor')
    return captured[0].flatten(1).cpu().numpy()


def extract_worker(cfg, rank):
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import build_evenet_on_device, load_normalization_dict
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import EventPackingSpec, unpack_event_inputs
    from scripts.diagnose_h4_spike_coverage import load_raw_state
    device = torch.device('cuda:0')
    global_config.load_yaml(cfg['runtime'])
    policy = build_evenet_on_device(global_config, load_normalization_dict(global_config), device).eval()
    checkpoint = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)
    load_raw_state(policy, checkpoint, 'step1110')
    del checkpoint
    policy.requires_grad_(False)
    spec = EventPackingSpec.from_dict(cfg['packing_spec'])
    output = Path(cfg['cache'])
    for label, source in cfg['sources'].items():
        source = Path(source)
        pool_path = source/('pool.pt' if (source/'pool.pt').is_file() else 'validation_pool.pt')
        pool = torch.load(pool_path, map_location='cpu', weights_only=True)
        with np.load(source/'candidates.npz') as f:
            truth, generated = f['truth_deltas'], f['deltas'][:, 0]
        idx = np.arange(rank, len(truth), cfg['workers'])
        hidden = [[], []]
        for start in range(0, len(idx), cfg['feature_batch_size']):
            take = idx[start:start+cfg['feature_batch_size']]
            batch = unpack_event_inputs(pool['condition'][take].to(device), spec)
            # No truth/gen class identity is sent to the pretrained trunk.
            batch.pop('classification', None)
            for j, candidate in enumerate((truth, generated)):
                hidden[j].append(candidate_hidden(policy, batch,
                    torch.as_tensor(candidate[take], device=device, dtype=torch.float32)))
            print(f'[features {label} rank {rank}] {min(start+len(take),len(idx))}/{len(idx)}', flush=True)
        np.savez(output/f'{label}-{rank:02d}.npz', positions=idx,
                 truth=np.concatenate(hidden[0]), generated=np.concatenate(hidden[1]))
    return rank


def merge_features(directory, label, events, workers):
    output = None
    seen = np.zeros(events, dtype=np.int64)
    for rank in range(workers):
        with np.load(Path(directory)/f'{label}-{rank:02d}.npz') as shard:
            idx, a, b = shard['positions'], shard['truth'], shard['generated']
            if a.shape != b.shape or a.ndim != 2 or len(a) != len(idx):
                raise ValueError('Feature shard shape mismatch')
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite cached features')
            if output is None:
                output = [np.empty((events, a.shape[1]), dtype=np.float32) for _ in range(2)]
            np.add.at(seen, idx, 1)
            output[0][idx], output[1][idx] = a, b
    if not np.all(seen == 1):
        raise ValueError('Missing or duplicate feature rows')
    return output


def attach_features(arrays, cache, train_source, test_source):
    cache = Path(cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    if not (cache/'COMPLETE').is_file() or manifest['weights'] != 'raw_state_dict_only':
        raise ValueError('Incomplete or non-raw trunk cache')
    for label, source in (('train', train_source), ('test', test_source)):
        if Path(manifest['sources'][label]).resolve() != Path(source).resolve():
            raise ValueError('Cache sample source mismatch')
    for target in ('truth', 'generated'):
        pieces = []
        for label in ('train', 'test'):
            with np.load(cache/f'{label}.npz') as f:
                pieces.append(f[target])
        features = np.concatenate(pieces)
        if len(features) != len(arrays['split']) or not np.isfinite(features).all():
            raise ValueError('Invalid cache features')
        # Pre-velocity hidden tokens are already layer-normalized. Keep raw
        # explicit tau features LAST for the fixed MMD geometry.
        arrays[f'candidate_{target}'] = np.concatenate((features, arrays[f'tau_{target}']), axis=1)
    identities = []
    for label in ('train', 'test'):
        with np.load(cache/f'{label}.npz') as f:
            identities.append(f['source_ids'])
    if not np.array_equal(np.concatenate(identities), arrays['source_ids']):
        raise ValueError('Cached backbone features have different event identities/order')
    return manifest
