"""Train conditional tau- or spin-coordinate ratios on saved K=1 samples.

The user launches this on an existing 16-GPU Ray allocation. The classifier
head starts fresh; optional backbone-cache features come from a separately
extracted, frozen raw policy trunk. This script never updates the policy.
Candidate and truth features use the same observable mapping, and truth-only
information never enters the visible condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'evenet_dgpo')]

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.diagnose_ztautau_cij import (
    SOURCE_ID_COLUMNS, align, angles, read_event_table, source_ids,
    tau_from_deltas,
)

DEFAULT_TRAIN_SOURCE = Path('/pscratch/sd/y/yiren/Ztautau/h4_step1110_cij_omnifold_train')
DEFAULT_TEST_SOURCE = Path('/pscratch/sd/y/yiren/Ztautau/h4_film_step1110_cij/rescore-ca89b70725')
DEFAULT_TRAIN_EVENTS = Path('/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train')
DEFAULT_TEST_EVENTS = Path('/pscratch/sd/y/yiren/Ztautau/diffusion_val_20pct_seed42_stic_filtered_test1/val')
DEFAULT_OUTPUT = Path('/pscratch/sd/y/yiren/Ztautau/conditional_spin_ratio_1110')
DEFAULT_TAU_OUTPUT = Path('/pscratch/sd/y/yiren/Ztautau/conditional_tau_ratio_1110')
AXES = ('k', 'r', 'n')
SPIN_DIM = 24
TAU_DIM = 15


def spin_features(a, b, kappas):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Spin directions must be matching [N,3] arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite spin directions')
    if not np.allclose(np.square(a).sum(1), 1, atol=1e-5) or not np.allclose(np.square(b).sum(1), 1, atol=1e-5):
        raise ValueError('Spin directions must be unit vectors')
    kappas = np.asarray(kappas, dtype=np.float64)
    if kappas.shape != (len(a), 2) or not np.isfinite(kappas).all() or (kappas == 0).any():
        raise ValueError('Analyzing powers must be finite, nonzero [N,2]')
    products = np.einsum('ni,nj->nij', a, b).reshape(-1, 9)
    # These are the exact per-event terms whose weighted means form Cij / 9.
    # They are computed identically for truth and generated candidates.
    cij_terms = products / np.prod(kappas, axis=1)[:, None]
    return np.concatenate((a, b, products, cij_terms), axis=1).astype('float32')


def tau_features(tau_a, tau_b):
    """Two reconstructed tau directions plus their joint products.

    The generator's tau energy and momentum magnitude are fixed by
    tau_from_deltas, so these directions retain all candidate information
    without inserting decay-channel analyzing powers or Cij observables.
    """
    a, b = np.asarray(tau_a, dtype=np.float64), np.asarray(tau_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4:
        raise ValueError('Tau four-vectors must be matching [N,4] arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite tau four-vectors')
    a, b = a[:, 1:], b[:, 1:]
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if (norm_a < 1e-12).any() or (norm_b < 1e-12).any():
        raise ValueError('Undefined tau direction')
    a, b = a / norm_a[:, None], b / norm_b[:, None]
    return np.concatenate((a, b, np.einsum('ni,nj->nij', a, b).reshape(-1, 9)), 1).astype('float32')


def event_split(event_ids, seed):
    """Split OmniFold-train conditions into fitting and early-stop groups."""
    out = np.empty(len(event_ids), dtype=np.uint8)
    for i, key in enumerate(event_ids):
        digest = hashlib.blake2b(f'{seed}:{key}'.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, 'little') % 100
        out[i] = 0 if value < 85 else 1
    if any(np.count_nonzero(out == k) < 2 for k in (0, 1)):
        raise ValueError('Need at least two events in both train and early-stop splits')
    return out


def read_bundle(source, events, representation='spin', verify_packing_spec=None, relative_input=False):
    if representation not in ('spin', 'tau'):
        raise ValueError('Representation must be spin or tau')
    source, events = Path(source), Path(events)
    manifest = json.loads((source / 'manifest.json').read_text())
    if int(manifest['events']) < 3 or manifest.get('weight_mode') != 'joint':
        raise ValueError('Requires complete joint-weight K=1 saved score source')
    recorded_source = manifest.get('event_source')
    if recorded_source and Path(recorded_source).resolve() != events.resolve():
        raise ValueError(f'Saved sample source {recorded_source} differs from parquet {events}')
    with np.load(source / 'candidates.npz', allow_pickle=False) as f:
        bundle = {k: f[k] for k in f.files}
    n = len(bundle['truth_deltas'])
    if n != int(manifest['events']):
        raise ValueError(f'Saved candidate count {n} differs from manifest events {manifest["events"]}')
    if bundle['deltas'].shape != (n, 1, 2, 2) or bundle['log_ratio'].shape != (n, 1):
        raise ValueError('Requires one generated candidate per paired condition')
    if not np.isfinite(bundle['deltas']).all() or not np.isfinite(bundle['truth_deltas']).all():
        raise ValueError('Nonfinite candidate/target directions')
    pool_file = source / ('pool.pt' if (source / 'pool.pt').is_file() else 'validation_pool.pt')
    pool = torch.load(pool_file, map_location='cpu', weights_only=True)
    condition = pool['condition'].numpy()
    if condition.ndim != 2 or len(condition) != n or not np.isfinite(condition).all():
        raise ValueError('Misaligned or nonfinite packed visible conditions')
    if 'deltas' in pool:
        if not np.allclose(pool['deltas'].numpy(), bundle['deltas'], atol=1e-7, rtol=0):
            raise ValueError('pool.pt and candidate bundle have different event order')
    elif not np.allclose(pool['truth'].numpy(), bundle['truth_deltas'], atol=1e-6, rtol=0):
        raise ValueError('validation_pool.pt and candidate truth differ')
    if not np.isfinite(bundle['log_ratio']).all():
        raise ValueError('Nonfinite saved baseline logits')
    keys = [k for k in SOURCE_ID_COLUMNS if k in bundle]
    ids = source_ids(bundle, keys)
    if verify_packing_spec is not None:
        verify_paired_parquet(events, bundle, condition, ids, keys, verify_packing_spec)
    required = ['source_sample_index', 'source_event_key', 'event_weight',
                'event_category', 'analyzing_power_a', 'analyzing_power_b']
    required += [k for k in keys if k not in required]
    for prefix in ('lead_a_visible', 'lead_b_visible'):
        required += [f'{prefix}_{component}' for component in ('E', 'px', 'py', 'pz')]
    table = read_event_table(events, required)
    columns = {key: np.asarray(table[key].to_numpy()) for key in required}
    order = align(source_ids(columns, keys), ids)
    columns = {key: value[order] for key, value in columns.items()}
    if len(condition) != len(columns['event_category']):
        raise ValueError('Event count changed during parquet alignment')
    category = columns['event_category'].astype(np.int64)
    if not np.isin(category // 10, range(1, 5)).all() or not np.isin(category % 10, range(1, 5)).all():
        raise ValueError('Unexpected tau decay category')
    def p4(prefix):
        return np.stack([columns[f'{prefix}_{c}'] for c in ('E', 'px', 'py', 'pz')], -1)
    va, vb = p4('lead_a_visible'), p4('lead_b_visible')
    truth = bundle['truth_deltas']
    generated = bundle['deltas'][:, 0]
    tau_truth_a, tau_truth_b = tau_from_deltas(va, truth[:, 0]), tau_from_deltas(vb, truth[:, 1])
    tau_sample_a, tau_sample_b = tau_from_deltas(va, generated[:, 0]), tau_from_deltas(vb, generated[:, 1])
    truth_angles = angles(tau_truth_a, tau_truth_b, va, vb)
    sample_angles = angles(tau_sample_a, tau_sample_b, va, vb)
    kappas = np.stack((columns['analyzing_power_a'], columns['analyzing_power_b']), 1).astype('float64')
    truth_tau_features = tau_features(tau_truth_a, tau_truth_b)
    sample_tau_features = tau_features(tau_sample_a, tau_sample_b)
    if representation == 'spin':
        candidate_truth = spin_features(*truth_angles, kappas)
        candidate_generated = spin_features(*sample_angles, kappas)
    else:
        candidate_truth, candidate_generated = truth_tau_features, sample_tau_features
    # Decay category is shared by the paired examples and available from visible topology.
    one_hot = np.eye(16, dtype=np.float32)[(category // 10 - 1) * 4 + category % 10 - 1]
    condition = np.concatenate((condition.astype('float32'), one_hot), axis=1)
    event_weight = columns['event_weight'].astype('float64')
    if not np.isfinite(event_weight).all() or (event_weight < 0).any() or event_weight.sum() <= 0:
        raise ValueError('Invalid event weights')
    arrays = dict(condition=condition, candidate_truth=candidate_truth,
        candidate_generated=candidate_generated,
        tau_truth=truth_tau_features, tau_generated=sample_tau_features,
        visible_pt_sum=np.linalg.norm(va[:, 1:3], axis=1) + np.linalg.norm(vb[:, 1:3], axis=1),
        truth_a=truth_angles[0], truth_b=truth_angles[1],
        sample_a=sample_angles[0], sample_b=sample_angles[1],
        base_log_ratio=bundle['log_ratio'][:, 0],
        category=category, event_weight=event_weight, kappas=kappas,
        source_ids=ids)
    if relative_input:
        from scripts.tau_relative_inputs import relative_angles
        arrays['relative_truth'] = relative_angles(tau_truth_a, tau_truth_b, va, vb)
        arrays['relative_generated'] = relative_angles(tau_sample_a, tau_sample_b, va, vb)
    return arrays, manifest


def verify_paired_parquet(events, bundle, saved_condition, ids, keys, packing_spec):
    """Certify old bundles with missing path metadata against actual parquet.

    Rebuild the original packed visible condition and truth target, then align
    by complete source identity. A matching event ID alone is insufficient.
    """
    from scripts.sample_1110_cij import validation_pool
    rebuilt = validation_pool(events, {'packing_spec':packing_spec})
    if len(rebuilt['truth']) != len(ids):
        raise ValueError('Saved candidates and validation parquet contain different event counts')
    order = align(source_ids(rebuilt, keys), ids)
    actual_condition = rebuilt['condition'][order].numpy()
    actual_truth = rebuilt['truth'][order].numpy()
    if actual_condition.shape != saved_condition.shape or not np.allclose(
            actual_condition, saved_condition, rtol=1e-6, atol=1e-6):
        raise ValueError('Saved classifier condition does not match the validation parquet')
    if actual_truth.shape != bundle['truth_deltas'].shape or not np.allclose(
            actual_truth, bundle['truth_deltas'], rtol=0, atol=1e-6):
        raise ValueError('Saved truth tau target does not match the validation parquet')
    return {'events':len(ids), 'source_id_match':True,
        'packed_visible_match':True, 'truth_target_match':True}


def resolve_sample_source(path):
    path = Path(path)
    if (path / 'candidates.npz').is_file():
        return path
    complete = sorted(p for p in path.glob('sample-*')
        if (p / 'COMPLETE').is_file() and (p / 'candidates.npz').is_file())
    if len(complete) != 1:
        raise ValueError(f'Expected exactly one completed sample directory under {path}; found {len(complete)}. Pass --train-source explicitly.')
    return complete[0]


def resolve_test_events(source, explicit=None):
    manifest = json.loads((Path(source) / 'manifest.json').read_text())
    recorded = manifest.get('event_source')
    events = Path(explicit) if explicit is not None else (Path(recorded) if recorded else DEFAULT_TEST_EVENTS)
    if recorded and events.resolve() != Path(recorded).resolve():
        raise ValueError(f'Explicit test parquet {events} differs from saved event_source {recorded}')
    if not events.exists():
        raise ValueError(f'Test parquet event source does not exist: {events}')
    return events


def join_pools(train, test, seed, condition_normalization='legacy_slot', packing_spec=None):
    if train['condition'].shape[1] != test['condition'].shape[1]:
        raise ValueError('OmniFold train and independent test condition packing differ')
    if set(train['source_ids']).intersection(test['source_ids']):
        raise ValueError('OmniFold train and independent test pools overlap by source identity')
    fingerprints = {hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()
        for row in train['condition']}
    if any(hashlib.blake2b(np.ascontiguousarray(row).tobytes(), digest_size=16).digest()
           in fingerprints for row in test['condition']):
        raise ValueError('OmniFold train and independent test pools overlap by visible condition')
    split = np.r_[event_split(train['source_ids'], seed),
                  np.full(len(test['source_ids']), 2, dtype=np.uint8)]
    arrays = {key:np.concatenate((train[key], test[key]), axis=0) for key in train}
    arrays['split'] = split
    for group in range(3):
        if arrays['event_weight'][split == group].sum() <= 0:
            raise ValueError('Zero total event weight in one split')
    condition = arrays['condition']
    if condition_normalization == 'masked_feature':
        from scripts.conditional_tau_preprocessing import fit_masked_feature, apply_masked_feature
        mean, scale = fit_masked_feature(condition, split == 0, packing_spec)
        arrays['condition'] = apply_masked_feature(condition, mean, scale, packing_spec)
        arrays['condition_mean'], arrays['condition_scale'] = mean, scale
        return arrays
    if condition_normalization != 'legacy_slot':
        raise ValueError(f'Unknown condition normalization: {condition_normalization}')
    mean = condition[split == 0].mean(0)
    scale = condition[split == 0].std(0)
    # Keep masks and one-hot columns as 0/1, and standardize continuous values
    # from OmniFold fit events only. The test distribution never sets statistics.
    for j in range(condition.shape[1] - 16):
        if np.isin(condition[:, j], (0., 1.)).all():
            mean[j], scale[j] = 0., 1.
    mean[-16:], scale[-16:] = 0., 1.
    scale = np.where(scale > 1e-6, scale, 1.)
    arrays['condition'] = np.clip((condition - mean) / scale, -20, 20).astype('float32')
    arrays['condition_mean'] = mean.astype('float32')
    arrays['condition_scale'] = scale.astype('float32')
    return arrays


class ConditionalSpinMLP(nn.Module):
    def __init__(self, condition_dim, hidden=128, dropout=0.05, candidate_dim=SPIN_DIM, relative_dim=0):
        super().__init__()
        if relative_dim not in (0, 6) or candidate_dim-relative_dim < 15:
            raise ValueError('Invalid relative-input dimensions')
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 64), nn.SiLU())
        self.spin_encoder = nn.Sequential(nn.Linear(candidate_dim-relative_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(128, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, 1))
        if relative_dim:
            # Build exactly the baseline first, then extend its input Linear.
            # Preserve common parameters AND the RNG stream (e.g. dropout).
            old = self.spin_encoder[0]
            with torch.random.fork_rng(devices=[]):
                new = nn.Linear(candidate_dim, 64)
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:, :-21].copy_(old.weight[:, :-15])
                new.weight[:, -15:].copy_(old.weight[:, -15:])
                new.bias.copy_(old.bias)
            self.spin_encoder[0] = new

    def forward(self, condition, spin):
        return self.head(torch.cat((self.condition_encoder(condition),
                                    self.spin_encoder(spin)), 1)).flatten()


def build_classifier(cfg, condition_dim=None, candidate_dim=None):
    if cfg.get('explicit_input'):
        from scripts.tau_explicit_inputs import build_explicit_classifier
        return build_explicit_classifier(cfg, condition_dim, candidate_dim)
    args = (condition_dim if condition_dim is not None else cfg['condition_dim'],
            cfg['hidden'], cfg['dropout'],
            candidate_dim if candidate_dim is not None else cfg['candidate_dim'], cfg.get('relative_dim', 0))
    kind = cfg.get('head_kind', 'legacy')
    bound = cfg.get('ratio_bound')
    if bound is not None and (kind != 'film' or cfg.get('ratio_objective','bce') != 'bce'):
        raise ValueError('Bounded-ratio experiment requires FiLM paired BCE')
    if kind == 'legacy':
        return ConditionalSpinMLP(*args)
    from scripts.tau_conditioning_heads import ConditionedTauMLP
    cls, extra = ConditionedTauMLP, {}
    if cfg.get('cross_attention', False):
        from scripts.tau_conditioning_heads import AttentionConditionedTauMLP
        if kind != 'film':
            raise ValueError('Cross-attention ablation requires FiLM')
        cls = AttentionConditionedTauMLP
        extra = dict(packing_spec=cfg['packing_spec'], attention_heads=cfg.get('attention_heads', 4))
    return cls(*args, **extra, kind=kind, depth=cfg.get('head_depth', 3),ratio_bound=bound,
        condition_hidden=cfg.get('condition_hidden'),condition_width=cfg.get('condition_width',64))


def paired_loss(model, condition, truth, generated, weight):
    positive = model(condition, truth)
    negative = model(condition, generated)
    loss = (F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive), reduction='none') +
            F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative), reduction='none')) / 2
    # Fit weights are normalized once over the entire fit population. A
    # per-batch weight denominator would make the stochastic objective depend
    # on the random mix of event weights in each batch.
    return (loss * weight).mean(), positive, negative


@torch.no_grad()
def score_pair(model, condition, truth, generated, device, batch_size=2048):
    model.eval()
    positive, negative = [], []
    for start in range(0, len(condition), batch_size):
        stop = start + batch_size
        c = torch.from_numpy(condition[start:stop]).to(device)
        a = torch.from_numpy(truth[start:stop]).to(device)
        b = torch.from_numpy(generated[start:stop]).to(device)
        positive.append(model(c, a).float().cpu().numpy())
        negative.append(model(c, b).float().cpu().numpy())
    return np.concatenate(positive), np.concatenate(negative)


def pair_metrics(positive, negative, weight):
    from sklearn.metrics import roc_auc_score
    weight = np.asarray(weight, dtype=np.float64)
    labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
    logits = np.r_[positive, negative]
    weights = np.r_[weight, weight]
    bce = np.logaddexp(0., -positive) + np.logaddexp(0., negative)
    return dict(bce=float(np.average(bce / 2, weights=weight)),
        auc=float(roc_auc_score(labels, logits, sample_weight=weights)))


def generated_log_ratio(positive, negative):
    """Balanced BCE: generated logit = log p_truth(s,c)/q_generated(s,c).

    Positive-event logits are classification diagnostics, not a baseline to
    subtract from each generated event's log density ratio.
    """
    positive, negative = np.asarray(positive), np.asarray(negative)
    if positive.shape != negative.shape or not np.isfinite(negative).all():
        raise ValueError('Invalid paired classifier scores')
    return negative


def training_worker(cfg):
    import ray.train
    import ray.train.torch
    from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from torch import distributed as dist

    context = ray.train.get_context()
    rank, world = context.get_world_rank(), context.get_world_size()
    device = ray.train.torch.get_device()
    torch.manual_seed(cfg['seed'])
    with np.load(cfg['prepared'], allow_pickle=False) as f:
        c = f['condition']; s_true = f['candidate_truth']; s_gen = f['candidate_generated']
        weights = f['event_weight']; split = f['split']
        condition_mean, condition_scale = f['condition_mean'], f['condition_scale']
        if cfg.get('conditioning_diagnostics'):
            categories, visible_pt = f['category'], f['visible_pt_sum']
    train = split == 0
    valid = split == 1
    train_weight = (weights[train] / weights[train].mean()).astype('float32')
    dataset = TensorDataset(torch.from_numpy(c[train]), torch.from_numpy(s_true[train]),
        torch.from_numpy(s_gen[train]), torch.from_numpy(train_weight))
    if cfg.get('fresh_negatives'):
        # Fit-local indices only: validation/test candidates never enter the refresh path.
        dataset = TensorDataset(*dataset.tensors, torch.arange(int(train.sum())))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                 seed=cfg['seed'], drop_last=False)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], sampler=sampler,
                        num_workers=0, pin_memory=True)
    model = build_classifier(cfg, c.shape[1], s_true.shape[1]).to(device)
    model = ray.train.torch.prepare_model(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['min_lr'])
    refresh = None
    if cfg.get('fresh_negatives'):
        from scripts.tau_fresh_negatives import FreshNegatives
        refresh = FreshNegatives(cfg, device, rank, s_gen[train])
    best, stop_best, stale, steps = float('inf'), float('inf'), 0, 0
    ratio_kind = cfg.get('ratio_objective', 'bce')
    direct_ratio = ratio_kind != 'bce'
    compute_train_mmd = bool(cfg.get('backbone_cache') and not direct_ratio and
                             not cfg.get('skip_train_mmd_diagnostic', False))
    if cfg.get('skip_train_mmd_diagnostic') and cfg.get('mmd_coefficient', 0.) != 0:
        raise ValueError('Cannot skip a nonzero MMD loss')
    for epoch in range(cfg['epochs']):
        extra_metrics = {}
        model.train(); sampler.set_epoch(epoch)
        running = torch.zeros(5, device=device)
        ratio_running = torch.zeros(8, dtype=torch.float64, device=device)
        if refresh is not None:
            refresh.begin_epoch(epoch)
        for batch_index, items in enumerate(loader):
            condition, truth, generated, weight = items[:4]
            condition, truth, generated, weight = [x.to(device, non_blocking=True)
                for x in (condition, truth, generated, weight)]
            if refresh is not None:
                generated = refresh.draw(items[4].numpy(), epoch, batch_index)
            optimizer.zero_grad(set_to_none=True)
            bce, p, q = paired_loss(model, condition, truth, generated, weight)
            from scripts.tau_backbone_alignment import joint_mmd
            mmd = joint_mmd(condition, truth[:, -15:], generated[:, -15:], q, weight) if compute_train_mmd else bce.new_zeros(())
            loss = bce + cfg.get('mmd_coefficient', 0.) * mmd
            if direct_ratio:
                if cfg.get('mmd_coefficient', 0.) != 0:
                    raise ValueError('Ratio-loss ablation must not include MMD')
                from scripts.tau_ratio_objectives import ratio_objective
                loss, stats = ratio_objective(p, q, weight, ratio_kind, cfg['nnukl_c'])
                base_grad = torch.autograd.grad(stats['mlc'], (p,q), retain_graph=True)
                corr_grad = torch.autograd.grad(stats['correction'], (p,q), retain_graph=True)
                norms = torch.stack((sum(g.double().square().sum() for g in base_grad),
                                     sum(g.double().square().sum() for g in corr_grad)))
                dist.all_reduce(norms)
                grad_ratio = (norms[1]/norms[0].clamp_min(1e-30)).sqrt()
                ratio_running += torch.stack(tuple(stats[k].detach() for k in
                    ('mlc','correction','risk_component','correction_active',
                     'mean_ratio_truth','mean_ratio_generated')) +
                    (grad_ratio.detach(), loss.detach()))
            logit_grad_ratio = bce.new_zeros(())
            if compute_train_mmd:
                # Magnitude check at the reward-logit interface, not a claim
                # about full parameter-gradient alignment. Logged in BOTH arms.
                gb, = torch.autograd.grad(bce, q, retain_graph=True)
                gm, = torch.autograd.grad(mmd, q, retain_graph=True)
                logit_grad_ratio = gm.detach().norm() / gb.detach().norm().clamp_min(1e-12)
            loss.backward()
            if direct_ratio or cfg.get('conditioning_diagnostics'):
                finite = torch.tensor(int(all(param.grad is None or bool(torch.isfinite(param.grad).all())
                    for param in model.parameters())), device=device)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite):
                    raise FloatingPointError('Nonfinite ratio-objective gradient; optimizer not stepped')
            optimizer.step()
            running += torch.stack((bce.detach() * len(weight), torch.as_tensor(len(weight), device=device),
                mmd.detach() * len(weight), loss.detach() * len(weight), logit_grad_ratio * len(weight)))
            steps += 1
        dist.all_reduce(running)
        scheduler.step()
        status = torch.zeros(15, dtype=torch.float64, device=device)
        if rank == 0:
            # rank-zero validation avoids 16 copies of the complete validation pool.
            bare = model.module if isinstance(model, DistributedDataParallel) else model
            p, q = score_pair(bare, c[valid], s_true[valid], s_gen[valid], device)
            metrics = pair_metrics(p, q, weights[valid])
            from scripts.diagnose_conditional_tau_ratio_tail import health
            ratio_health = health(weights[valid], q)
            alignment = dict(joint_mmd_unweighted=0., joint_mmd_reweighted=0., condition_mean_drift_rms=0.)
            if cfg.get('backbone_cache'):
                from scripts.tau_backbone_alignment import alignment_report
                ids = np.random.default_rng(814).permutation(len(p))[:4096]
                # Keep the diagnostic kernel geometry matched across input arms.
                # New visible inputs alter the classifier, not this fixed metric.
                diagnostic_c = c[:,:-8] if cfg.get('explicit_input') else c
                alignment = alignment_report(diagnostic_c[valid][ids], s_true[valid][ids, -15:],
                    s_gen[valid][ids, -15:], q[ids], weights[valid][ids], cfg['batch_size'])
            # Absolute best selects the checkpoint; min_delta only controls
            # the patience clock. Small real improvements must not be lost.
            if metrics['bce'] < best:
                best = metrics['bce']
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in bare.state_dict().items()},
                    condition_dim=c.shape[1], hidden=cfg['hidden'], dropout=cfg['dropout'],
                    candidate_dim=s_true.shape[1], representation=cfg['representation'],
                    condition_mean=torch.from_numpy(condition_mean),
                    condition_scale=torch.from_numpy(condition_scale),
                    packing_spec=cfg['packing_spec'],
                    condition_normalization=cfg['condition_normalization'],
                    backbone_cache=cfg.get('backbone_cache'), mmd_coefficient=cfg.get('mmd_coefficient', 0.),
                    relative_dim=cfg.get('relative_dim', 0), relative_preprocessing=cfg.get('relative_preprocessing'),
                    ratio_objective=ratio_kind, nnukl_c=cfg.get('nnukl_c', 0.),
                    head_kind=cfg.get('head_kind','legacy'), head_depth=cfg.get('head_depth',3),
                    ratio_bound=cfg.get('ratio_bound'),
                    condition_hidden=cfg.get('condition_hidden',cfg['hidden']),
                    condition_width=cfg.get('condition_width',64),
                    explicit_input=cfg.get('explicit_input'),
                    epoch=epoch, optimizer_steps=steps, val_bce=best, val_auc=metrics['auc']),
                    cfg['checkpoint'])
            if metrics['bce'] < stop_best - cfg['min_delta']:
                stop_best, stale = metrics['bce'], 0
            elif steps >= cfg['min_steps']:
                stale += 1
            if cfg.get('ratio_bound') is not None:
                from scripts.tau_bounded_ratio import bound_metrics
                extra_metrics.update({'val_bound/'+k:v for k,v in
                    bound_metrics(p,q,weights[valid],cfg['ratio_bound']).items()})
            if cfg.get('conditioning_diagnostics'):
                from scripts.tau_ratio_objectives import validation_objective
                risk = validation_objective(p,q,weights[valid],'mlc',0.)
                extra_metrics['val_ratio_risk'] = risk['mlc']
                if (epoch == 0 or (epoch+1) % cfg['diagnostic_every'] == 0 or
                    (steps >= cfg['min_steps'] and stale >= cfg['patience']) or epoch+1 == cfg['epochs']):
                    from scripts.tau_conditioning_diagnostics import conditional_report
                    d = conditional_report(s_true[valid,-15:],s_gen[valid,-15:],weights[valid],q,
                        categories[valid],visible_pt[valid],cfg['condition_pt_edges'])
                    extra_metrics.update({'val_condition/'+k:v for k,v in d['summary'].items()})
                extra_metrics['condition_projection_weight_norm'] = float(torch.stack([
                    block.context.weight.detach().norm() for block in bare.blocks]).norm())
            status[:11] = status.new_tensor((metrics['bce'], metrics['auc'], best,
                float(steps >= cfg['min_steps'] and stale >= cfg['patience']),
                ratio_health['ess_fraction'], ratio_health['max_mass'],
                ratio_health['log_mean_ratio'], float(np.max(q)),
                alignment['joint_mmd_unweighted'], alignment['joint_mmd_reweighted'],
                alignment['condition_mean_drift_rms']))
            if direct_ratio:
                from scripts.tau_ratio_objectives import validation_objective
                v = validation_objective(p, q, weights[valid], ratio_kind, cfg['nnukl_c'])
                status[11:] = status.new_tensor([v['objective'],v['risk_component'],
                                                v['correction'],v['correction_active']])
        dist.broadcast(status, src=0)
        report = dict(epoch=epoch + 1, optimizer_steps=steps,
            train_bce=float((running[0] / running[1]).item()),
            val_bce=float(status[0].item()), val_auc=float(status[1].item()),
            best_val_bce=float(status[2].item()), lr=float(optimizer.param_groups[0]['lr']),
            val_ratio_ess_fraction=float(status[4].item()), val_ratio_max_mass=float(status[5].item()),
            val_log_mean_ratio=float(status[6].item()), val_log_ratio_max=float(status[7].item()))
        report.update(extra_metrics)
        if refresh is not None:
            refresh_stats = torch.tensor([refresh.events, refresh.seconds], device=device, dtype=torch.float64)
            dist.all_reduce(refresh_stats)
            parity = torch.tensor(refresh.parity,device=device,dtype=torch.float64)
            dist.all_reduce(parity,op=dist.ReduceOp.MAX)
            report.update(fresh_negative_events=float(refresh_stats[0]),
                fresh_generation_feature_seconds_mean_rank=float(refresh_stats[1]/world),
                fresh_replay_max_error=float(parity),
                fresh_generator_trainable_parameters=0,
                fresh_candidates_per_condition=1, fresh_validation_refreshed=0,
                fresh_sampler_padding_events=int(sampler.total_size-len(dataset)))
        if rank == 0:
            report.update(early_stop_stale_epochs=stale,
                early_stop_triggered=int(bool(status[3].item())))
        if cfg.get('backbone_cache'):
            report.update(train_joint_mmd=float((running[2] / running[1]).item()),
                train_objective=float((running[3] / running[1]).item()),
                train_mmd_to_bce_logit_grad_norm=float((running[4] / running[1]).item()),
                val_joint_mmd_unweighted=float(status[8].item()), val_joint_mmd_reweighted=float(status[9].item()),
                val_condition_mean_drift_rms=float(status[10].item()))
        if direct_ratio:
            # Risk stats are already identical on every rank after global reduction.
            for key,value in zip(('mlc','correction','risk_component','correction_active_fraction',
                                 'mean_ratio_truth','mean_ratio_generated',
                                 'correction_to_mlc_logit_grad_norm','objective'), ratio_running/len(loader)):
                report[f'ratio_train/{key}'] = float(value)
            for key,value in zip(('objective','risk_component','correction','correction_active'), status[11:]):
                report[f'ratio_val/{key}'] = float(value)
        if cfg.get('relative_dim', 0):
            bare = model.module if isinstance(model, DistributedDataParallel) else model
            layer = bare.spin_encoder[0]
            report.update(relative_input_weight_norm=float(layer.weight[:, -21:-15].detach().norm()),
                          relative_input_grad_norm=float(layer.weight.grad[:, -21:-15].detach().norm()))
        if cfg.get('explicit_input'):
            bare = model.module if isinstance(model, DistributedDataParallel) else model
            layer = bare.condition_encoder[0]
            report.update(explicit_visible_weight_norm=float(layer.weight[:,-8:].detach().norm()),
                          explicit_visible_grad_norm=float(layer.weight.grad[:,-8:].detach().norm()))
            if cfg['explicit_input']['arm'] in ('geometry','products'):
                layer = bare.spin_encoder[0]
                start,stop = (-36,-30) if cfg['explicit_input']['arm']=='products' else (-27,-21)
                report.update(explicit_geometry_weight_norm=float(layer.weight[:,start:stop].detach().norm()),
                              explicit_geometry_grad_norm=float(layer.weight.grad[:,start:stop].detach().norm()))
                if cfg['explicit_input']['arm']=='products':
                    report.update(explicit_products_weight_norm=float(layer.weight[:,-30:-21].detach().norm()),
                                  explicit_products_grad_norm=float(layer.weight.grad[:,-30:-21].detach().norm()))
        ray.train.report(report)
        if bool(status[3].item()):
            break


def closure_report(arrays, logits, mask, bootstrap, seed):
    from scripts.cij_channel_diagnostics import channel_diagnostics
    from scripts.diagnose_reweighted_cij import analyze
    selected = np.flatnonzero(mask)
    a = arrays['truth_a'][selected]; b = arrays['truth_b'][selected]
    data = dict(event_id=arrays['source_ids'][selected], truth_a=a, truth_b=b,
        sample_a=arrays['sample_a'][selected, None], sample_b=arrays['sample_b'][selected, None],
        log_ratio=np.asarray(logits, dtype=np.float64)[:, None],
        event_weight=arrays['event_weight'][selected])
    # Truth directions are reconstructed from the saved true offsets through
    # the same map as generated candidates: the matched-target comparison.
    angles_report = analyze(data, [1, 1], 'joint', bootstrap, seed)
    channels = channel_diagnostics(data, arrays['kappas'][selected],
        arrays['category'][selected], 'joint', bootstrap, seed)
    return dict(events=len(selected), angular_moments_9x=angles_report,
        channels=channels, interpretation='Held-out paired events; target is matched reconstruction. '
        'The angular report contains 9 * <a_i b_j>; divide displayed matrices and errors by 9 '
        'for direct moments. Channel angular_moments fields are direct <a_i b_j>.')


def tau_moment_report(arrays, logits, mask):
    """Matched tau-pair direction moments, overall and within decay channels.

    These 15 moments are diagnostics, not a proof of full conditional-density
    alignment. Cij is assessed separately by closure_report.
    """
    from scipy.special import logsumexp
    selected = np.flatnonzero(mask)
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (len(selected),) or not np.isfinite(scores).all():
        raise ValueError('Invalid held-out tau ratio scores')
    base = arrays['event_weight'][selected].astype(np.float64)
    weighted = np.exp(np.log(np.maximum(base, np.finfo(float).tiny)) + scores -
                      logsumexp(np.log(np.maximum(base, np.finfo(float).tiny)) + scores))
    weighted[base == 0] = 0
    truth = arrays['tau_truth'][selected].astype(np.float64)
    generated = arrays['tau_generated'][selected].astype(np.float64)
    categories = arrays['category'][selected]
    visible_pt = arrays['visible_pt_sum'][selected]
    def group(name, take):
        w0, w1 = base[take], weighted[take]
        if w0.sum() <= 0 or w1.sum() <= 0:
            raise ValueError(f'Zero weight in tau report group {name}')
        target = np.average(truth[take], weights=w0, axis=0)
        raw = np.average(generated[take], weights=w0, axis=0)
        fit = np.average(generated[take], weights=w1, axis=0)
        return dict(group=name, events=int(take.sum()), base_fraction=float(w0.sum()/base.sum()),
            reweighted_fraction=float(w1.sum()/weighted.sum()),
            target=target.tolist(), unweighted=raw.tolist(), reweighted=fit.tolist(),
            l2_error_unweighted=float(np.linalg.norm(raw-target)),
            l2_error_reweighted=float(np.linalg.norm(fit-target)),
            l2_error_change=float(np.linalg.norm(fit-target)-np.linalg.norm(raw-target)))
    groups = [group('all', np.ones(len(selected), dtype=bool))]
    groups += [group(f'category_{int(cat)}', categories == cat) for cat in np.unique(categories)]
    # Coarse visible-condition strata expose cancellations hidden by the
    # inclusive tau moments. Quantiles are analysis-only and never enter fitting.
    edges = np.quantile(visible_pt, [0., .25, .5, .75, 1.])
    bin_index = np.searchsorted(edges[1:-1], visible_pt, side='right')
    for q in range(4):
        take = bin_index == q
        if take.sum() >= 2:
            groups.append(group(f'visible_pt_q{q}', take))
        for cat in np.unique(categories):
            take_channel = take & (categories == cat)
            if take_channel.sum() >= 2:
                groups.append(group(f'category_{int(cat)}_visible_pt_q{q}', take_channel))
    return dict(feature_order='tau_a_unit_xyz, tau_b_unit_xyz, 9 tau_a_outer_tau_b terms',
        scope='Independent held-out matched reconstruction; paired truth and generated conditions',
        visible_pt_quartiles=edges.tolist(),
        limitation='First/second joint direction moments within coarse visible strata do not establish full conditional tau distribution closure',
        groups=groups)


def main(default_representation='spin'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representation', choices=('spin','tau'), default=default_representation)
    parser.add_argument('--backbone-cache', type=Path,
        help='Frozen raw step1110 full generation-trunk features, prepared once for both arms')
    parser.add_argument('--mmd-coefficient', type=float, default=0.,
        help='Fixed-geometry joint weighted MMD, only with tau backbone cache')
    parser.add_argument('--relative-angle-input', action='store_true',
        help='BCE-only input ablation: six scaled tau-visible relations; reuse frozen cache')
    parser.add_argument('--condition-normalization', choices=('legacy_slot','masked_feature'), default='legacy_slot')
    parser.add_argument('--baseline-directory', type=Path,
        help='Completed matched tau run: verify protocol and compare saved scores without refitting')
    parser.add_argument('--baseline-run', default=None)
    parser.add_argument('--allow-baseline-batch-size-mismatch', action='store_true',
        help='Allow only batch-size mismatch against historical baseline; record confounding.')
    parser.add_argument('--train-source', type=Path, default=DEFAULT_TRAIN_SOURCE)
    parser.add_argument('--test-source', type=Path, default=DEFAULT_TEST_SOURCE)
    parser.add_argument('--train-events', type=Path, default=DEFAULT_TRAIN_EVENTS)
    parser.add_argument('--test-events', type=Path,
        help='Validation parquet directory; default is the exact event_source in the saved test manifest')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256, help='paired events per GPU')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--min-delta', type=float, default=0.0,
        help='Minimum validation BCE decrease counted as improvement; 0 saves every new minimum')
    parser.add_argument('--min-steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--bootstrap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-id', type=str)
    parser.add_argument('--run-name', type=str)
    parser.add_argument('--wandb-group', default='Conditional ratio closure')
    parser.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS') or 'auto')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.relative_angle_input and (not args.backbone_cache or args.representation != 'tau' or args.mmd_coefficient != 0):
        parser.error('Relative-angle ablation requires frozen tau backbone and pure BCE')
    if not math.isfinite(args.mmd_coefficient) or args.mmd_coefficient < 0:
        parser.error('MMD coefficient must be finite and nonnegative')
    if (args.backbone_cache and args.representation != 'tau') or (args.mmd_coefficient and not args.backbone_cache):
        parser.error('Joint tau MMD requires the frozen tau backbone cache')
    if not math.isfinite(args.min_delta) or args.min_delta < 0 or args.patience < 1 or args.min_steps < 0:
        parser.error('Require finite min-delta >= 0, patience >= 1 and min-steps >= 0')
    if args.workers < 1 or args.batch_size < 1 or args.epochs < 1 or args.bootstrap < 20:
        parser.error('Workers, batch size and epochs must be positive; bootstrap >= 20')
    if not 0 <= args.dropout < 1 or not 0 < args.min_lr <= args.lr:
        parser.error('Invalid dropout or learning rate')
    output = (args.output or (DEFAULT_TAU_OUTPUT if args.representation == 'tau' else DEFAULT_OUTPUT)).resolve()
    if output.exists() and not args.prepare_only:
        parser.error('Use a fresh output directory')
    args.train_source = resolve_sample_source(args.train_source)
    for source in (args.train_source, args.test_source):
        if not source.is_dir() or not (source / 'candidates.npz').is_file():
            parser.error(f'Saved source with candidates.npz is required: {source}')
    was_auto = args.test_events is None
    args.test_events = resolve_test_events(args.test_source, args.test_events)
    test_meta = json.loads((args.test_source / 'manifest.json').read_text())
    print(f'TEST PARQUET SOURCE: {args.test_events}', flush=True)
    if not test_meta.get('event_source'):
        print('NOTE: old test manifest lacks event_source; verifying complete IDs, '
              'packed visible conditions, and truth targets against this parquet.', flush=True)
    elif was_auto and args.test_events.resolve() != DEFAULT_TEST_EVENTS.resolve():
        print(f'NOTE: saved test source differs from expected filtered default '
              f'{DEFAULT_TEST_EVENTS}; using the source that actually produced the saved candidates.',
              flush=True)
    train, train_manifest = read_bundle(args.train_source, args.train_events, args.representation,
        relative_input=args.relative_angle_input)
    test, test_manifest = read_bundle(args.test_source, args.test_events, args.representation,
        verify_packing_spec=train_manifest['packing_spec'], relative_input=args.relative_angle_input)
    if train_manifest.get('pool_mode') != 'omnifold-train':
        raise ValueError('Training source must be generated from the OmniFold 10pct train pool')
    if Path(train_manifest['classifier_checkpoint']).resolve() != Path(test_manifest['checkpoint']).resolve():
        raise ValueError('Train and test packed conditions must use the same classifier packing checkpoint')
    test_sample = Path(test_manifest['samples']) / 'manifest.json'
    test_sample_manifest = json.loads(test_sample.read_text())
    for field in ('checkpoint', 'ddim_steps', 'weights', 'candidates'):
        if train_manifest.get(field) != test_sample_manifest.get(field):
            raise ValueError(f'Train and test samples use different {field}')
    arrays = join_pools(train, test, args.seed, args.condition_normalization, train_manifest['packing_spec'])
    backbone_manifest = None
    if args.backbone_cache:
        from scripts.tau_backbone_alignment import attach_features
        backbone_manifest = attach_features(arrays, args.backbone_cache, args.train_source, args.test_source)
        for field in ('checkpoint', 'runtime', 'packing_spec'):
            if backbone_manifest[field] != train_manifest[field]:
                raise ValueError(f'Frozen backbone cache differs from generation source: {field}')
    from scripts.conditional_tau_preprocessing import preprocessing_report, particle_view, ratio_group_report
    input_report = preprocessing_report(arrays['condition'], arrays['split'], train_manifest['packing_spec'])
    relative_preprocessing = None
    input_contract = None
    if args.relative_angle_input:
        from scripts.tau_relative_inputs import attach_relative_inputs, paired_input_contract
        relative_preprocessing = attach_relative_inputs(arrays)
        input_report['relative_angles'] = relative_preprocessing
        input_contract = paired_input_contract(arrays, train_manifest['packing_spec'])
        input_report['paired_input_contract'] = input_contract
    previous_scores = None
    baseline_training_differences = {}
    if args.baseline_directory is not None:
        previous = json.loads((args.baseline_directory/'manifest.json').read_text())
        if args.relative_angle_input:
            if (previous.get('relative_dim', 0) != 0 or previous.get('mmd_coefficient', 0) != 0
                or previous.get('condition_normalization') != args.condition_normalization
                or previous.get('backbone_cache') != str(args.backbone_cache.resolve())):
                raise ValueError('Relative-input control must use the same frozen cache, pure BCE, and original inputs')
        for key in ('seed','workers','batch_size','epochs','patience','min_steps','min_delta',
                    'lr','min_lr','weight_decay','dropout','hidden','representation'):
            if previous[key] != getattr(args, key):
                if key == 'batch_size' and args.allow_baseline_batch_size_mismatch:
                    baseline_training_differences[key] = dict(
                        baseline=previous[key], current=args.batch_size)
                    print('NOTE: baseline batch size differs; comparison is not an input-only '
                          'causal ablation:', baseline_training_differences[key], flush=True)
                    continue
                raise ValueError(f'Matched baseline differs in {key}')
        for key in ('train_source','test_source','train_events','test_events'):
            if Path(previous[key]).resolve() != getattr(args, key).resolve():
                raise ValueError(f'Matched baseline differs in {key}')
        with np.load(args.baseline_directory/'prepared.npz', allow_pickle=False) as old:
            if not np.array_equal(old['source_ids'], arrays['source_ids']) or not np.array_equal(old['split'], arrays['split']):
                raise ValueError('Baseline event identities/splits differ')
        with np.load(args.baseline_directory/'test_scores.npz', allow_pickle=False) as f:
            previous_scores = {k:f[k] for k in f.files}
        if not np.array_equal(previous_scores['source_ids'], arrays['source_ids'][arrays['split'] == 2]):
            raise ValueError('Baseline test-score order differs')
    split_counts = {name:int(np.sum(arrays['split'] == index))
        for index,name in enumerate(('train','validation','test'))}
    if args.prepare_only:
        print(json.dumps({'prepared':True,'split_counts':split_counts,
            'condition_dim':int(arrays['condition'].shape[1]),
            'candidate_dim':int(arrays['candidate_truth'].shape[1]),
            'representation':args.representation,
            'condition_normalization':args.condition_normalization,
            'preprocessing':input_report,
            'train_source':str(args.train_source),'test_source':str(args.test_source)}, indent=2), flush=True)
        return
    output.mkdir(parents=True)
    np.savez(output / 'prepared.npz', **arrays)
    (output / 'preprocessing.json').write_text(json.dumps(input_report, indent=2)+'\n')
    run_name = args.run_name or ('Can tau-pair ratios close the gap? | conditional MLP | step 1110'
        if args.representation == 'tau' else 'Can spin coordinates close angular moments? | conditional MLP | step 1110')
    cfg = dict(train_source=str(args.train_source.resolve()), train_manifest=train_manifest,
        test_source=str(args.test_source.resolve()), test_manifest=test_manifest,
        test_sample_manifest=test_sample_manifest,
        train_events=str(args.train_events.resolve()), test_events=str(args.test_events.resolve()),
        output=str(output), prepared=str(output / 'prepared.npz'),
        checkpoint=str(output / 'best.pt'), seed=args.seed, workers=args.workers,
        representation=args.representation,
        condition_normalization=args.condition_normalization,
        baseline_directory=str(args.baseline_directory) if args.baseline_directory else None,
        baseline_run=args.baseline_run,
        baseline_training_differences=baseline_training_differences,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        min_delta=args.min_delta,
        min_steps=args.min_steps, lr=args.lr, min_lr=args.min_lr,
        weight_decay=args.weight_decay, dropout=args.dropout, hidden=args.hidden,
        bootstrap=args.bootstrap, split_counts=split_counts,
        candidate_count=1, classifier='conditional_candidate_mlp',
        backbone_cache=str(args.backbone_cache.resolve()) if args.backbone_cache else None,
        backbone_manifest=backbone_manifest, mmd_coefficient=args.mmd_coefficient,
        relative_dim=6 if args.relative_angle_input else 0, relative_preprocessing=relative_preprocessing,
        paired_input_contract=input_contract,
        selector='minimum internal validation BCE, identical in both arms',
        candidate_features=('a3+b3+outer9+outer9_over_signed_kappa_product'
            if args.representation == 'spin' else 'tau_a_unit3+tau_b_unit3+tau_pair_outer9'),
        run_name=run_name, wandb_group=args.wandb_group,
        packing_spec=train_manifest.get('packing_spec'),
        weight_mode='joint', evaluation_target='raw matched reconstruction',
        baseline_overlap='Original saved H4 classifier overlap with independent test pool unknown')
    if args.backbone_cache:
        cfg.update(classifier='frozen_full_generation_trunk_plus_conditional_tau_head',
            candidate_features='pre_velocity_hidden_tau_tokens+tau_unit_directions_and_outer_products',
            mmd_estimator='rank-local self-normalized biased V-statistic; DDP averages local objectives',
            mmd_geometry='fixed mask-normalized c times explicit tau product RBF; scales 0.25,1,4',
            val_alignment_events=4096,
            limitation='No fresh weighted two-sample audit; moments and MMD do not certify full conditional closure')
    if args.relative_angle_input:
        cfg['candidate_features'] += '+six_fit_scaled_tau_visible_relative_angles'
    (output / 'manifest.json').write_text(json.dumps(cfg, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in cfg.items() if k not in ('train_manifest','test_manifest','test_sample_manifest')}, indent=2), flush=True)
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    import wandb
    from ray.tune import Callback
    ray.init(address=args.ray_address, runtime_env={'env_vars':{'PYTHONPATH':os.pathsep.join(
        (str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'evenet_dgpo'), os.environ.get('PYTHONPATH', '')))}})
    if ray.cluster_resources().get('GPU', 0) < args.workers:
        raise ValueError(f'Requested {args.workers} workers but fewer GPUs available')
    class Progress(Callback):
        def on_trial_result(self, iteration, trials, trial, result, **info):
            metrics = {k:result[k] for k in ('epoch','optimizer_steps','train_bce',
                'val_bce','val_auc','best_val_bce','lr', 'val_ratio_ess_fraction',
                'val_ratio_max_mass','val_log_mean_ratio','val_log_ratio_max',
                'train_joint_mmd','train_objective','train_mmd_to_bce_logit_grad_norm',
                'val_joint_mmd_unweighted','val_joint_mmd_reweighted','val_condition_mean_drift_rms',
                'relative_input_weight_norm','relative_input_grad_norm') if k in result}
            if metrics:
                run.log(metrics, step=int(metrics.get('optimizer_steps', 0)))
    with wandb.init(entity='ytchou97-university-of-washington', project='nu2flow-RL',
        id=args.run_id, resume='never', mode='online',
        name=run_name,
        group=args.wandb_group, tags=[args.representation,'conditioned','16-gpu','step1110','K1'],
        config=cfg, dir=str(output)) as run:
        (output / 'wandb.json').write_text(json.dumps({'id':run.id,'url':run.url}) + '\n')
        run.summary['phase'] = 'training'
        run.save(str(output/'preprocessing.json'), base_path=str(output), policy='now')
        if input_contract:
            for row in input_contract['splits']:
                run.summary[f'input_contract/{row["split"]}/condition_only_auc'] = row['condition_only_auc']
            run.summary['input_contract/control_kind'] = 'structural paired check, not trained classifier'
        for row in input_report['splits']:
            for key in ('valid_feature_clip_fraction','events_with_clipped_valid_features','padding_nonzero_values'):
                run.summary[f'preprocessing/{row["split"]}/{key}'] = row[key]
        try:
            TorchTrainer(train_loop_per_worker=training_worker, train_loop_config=cfg,
                scaling_config=ScalingConfig(num_workers=args.workers, use_gpu=True),
                run_config=RunConfig(name='conditional-spin-ratio',
                    storage_path=str(output / 'ray_results'), callbacks=[Progress()],
                    failure_config=FailureConfig(max_failures=0))).fit()
            saved = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
            model = build_classifier(saved)
            model.load_state_dict(saved['state_dict'])
            mask = arrays['split'] == 2
            p, q = score_pair(model, arrays['condition'][mask], arrays['candidate_truth'][mask],
                arrays['candidate_generated'][mask], torch.device('cpu'))
            metrics = pair_metrics(p, q, arrays['event_weight'][mask])
            run.summary.update({'phase':'closure', 'test/bce':metrics['bce'],
                'test/auc':metrics['auc'], 'best_epoch':saved['epoch'] + 1,
                'fit_optimizer_steps':saved['optimizer_steps']})
            log_ratio = generated_log_ratio(p, q)
            # Save scores before closure so later diagnostics can recover even
            # if an uncertainty calculation has insufficient weight support.
            np.savez_compressed(output / 'test_scores.npz', source_ids=arrays['source_ids'][mask],
                truth_logits=p, generated_logits=q, log_ratio=log_ratio,
                saved_h4_log_ratio=arrays['base_log_ratio'][mask])
            run.save(str(output/'test_scores.npz'), base_path=str(output), policy='now')
            if args.backbone_cache:
                from scripts.tau_backbone_alignment import alignment_report
                alignment = alignment_report(arrays['condition'][mask], arrays['tau_truth'][mask],
                    arrays['tau_generated'][mask], log_ratio, arrays['event_weight'][mask], args.batch_size)
                (output/'joint_alignment.json').write_text(json.dumps(alignment, indent=2)+'\n')
                run.save(str(output/'joint_alignment.json'), base_path=str(output), policy='now')
                for key, value in alignment.items():
                    if isinstance(value, (int, float)):
                        run.summary[f'test_alignment/{key}'] = value
            _, particle_mask = particle_view(arrays['condition'], train_manifest['packing_spec'])
            multiplicity = particle_mask[mask].sum(axis=1)
            score_arms = [('candidate_ratio', log_ratio, p)]
            if previous_scores is not None:
                score_arms.append(('previous_tau', previous_scores['log_ratio'], previous_scores['truth_logits']))
            for name, scores, truth_scores in score_arms:
                groups = ratio_group_report(arrays['event_weight'][mask], scores, truth_scores,
                    arrays['category'][mask], multiplicity)
                group_path = output/f'{name}_ratio_groups.json'
                group_path.write_text(json.dumps(groups, indent=2, allow_nan=False)+'\n')
                run.save(str(group_path), base_path=str(output), policy='now')
                columns = ['group','events','log_mean_ratio','ess_fraction','max_mass',
                    'base_fraction','reweighted_fraction','truth_logit_mean','generated_logit_mean']
                run.log({f'{name}/ratio_groups':wandb.Table(columns=columns,
                    data=[[row[k] for k in columns] for row in groups['groups']])})
                for row in groups['groups']:
                    if row['group'] == 'all':
                        for key in ('ess','ess_fraction','max_mass','log_mean_ratio','top1pct_mass'):
                            run.summary[f'{name}/{key}'] = row[key]
            fresh = closure_report(arrays, log_ratio, mask, args.bootstrap, args.seed)
            baseline = closure_report(arrays, arrays['base_log_ratio'][mask],
                mask, args.bootstrap, args.seed)
            moment_arms = [(name, scores) for name, scores, _ in score_arms]
            moment_arms.append(('saved_h4', arrays['base_log_ratio'][mask]))
            for name, scores in moment_arms:
                tau_report = tau_moment_report(arrays, scores, mask)
                path = output / f'{name}_tau_moments.json'
                path.write_text(json.dumps(tau_report, indent=2, allow_nan=False) + '\n')
                run.save(str(path), base_path=str(output), policy='now')
                for group in tau_report['groups']:
                    if '_visible_pt_q' in group['group']:
                        continue
                    run.summary[f'{name}/tau/{group["group"]}/l2_error_change'] = group['l2_error_change']
            closure_arms = [('candidate_ratio', fresh), ('saved_h4', baseline)]
            if previous_scores is not None:
                closure_arms.append(('previous_tau', closure_report(arrays,
                    previous_scores['log_ratio'], mask, args.bootstrap, args.seed)))
            for name, report in closure_arms:
                path = output / f'{name}_closure.json'
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                run.save(str(path), base_path=str(output), policy='now')
                angular = report['angular_moments_9x']
                run.summary[f'{name}/angular_error_unweighted'] = angular['results']['unweighted']['frobenius_error'] / 9
                run.summary[f'{name}/angular_error_weighted'] = angular['results']['reweighted']['frobenius_error'] / 9
                run.summary[f'{name}/angular_error_change'] = angular['comparison']['C_frobenius_error_change']['value'] / 9
                run.summary[f'{name}/angular_error_change_ci95'] = (
                    np.asarray(angular['comparison']['C_frobenius_error_change']['ci95']) / 9).tolist()
                run.summary[f'{name}/ess_fraction'] = angular['weight_health']['event_ess_fraction']
                for arm in ('truth','unweighted','reweighted'):
                    run.summary[f'{name}/nn_{arm}'] = angular['results'][arm]['C'][2][2] / 9
                cij = report['channels']['decomposition']['errors']['stored_truth']
                run.summary[f'{name}/matched_cij_error_unweighted'] = cij['unweighted']
                run.summary[f'{name}/matched_cij_error_weighted'] = cij['reweighted']
                run.summary[f'{name}/matched_cij_error_change'] = cij['reweighted'] - cij['unweighted']
                channel_rows = []
                moment_rows = []
                for channel in report['channels']['channels']:
                    moments = channel['angular_moments']['references']['stored_truth']
                    change = moments['error_change']
                    channel_rows.append([channel['event_category'], channel['events'],
                        change['value'], *change['ci95']])
                    for i,axis_a in enumerate(AXES):
                        for j,axis_b in enumerate(AXES):
                            moment_rows.append([channel['event_category'],axis_a+axis_b,
                                moments['results']['truth']['moment'][i][j],
                                moments['results']['unweighted']['moment'][i][j],
                                moments['results']['reweighted']['moment'][i][j],
                                moments['absolute_error_change'][i][j],
                                moments['absolute_error_change_ci95'][0][i][j],
                                moments['absolute_error_change_ci95'][1][i][j]])
                run.log({f'{name}/channels':wandb.Table(columns=['category','events',
                    'angular_error_change','lo95','hi95'], data=channel_rows)})
                run.log({f'{name}/moments':wandb.Table(columns=['category','component',
                    'matched_truth','unweighted','reweighted','abs_error_change',
                    'change_lo95','change_hi95'], data=moment_rows)})
            run.summary['phase'] = 'complete'
            print('WANDB:', run.url, 'OUTPUT:', output, flush=True)
        except BaseException:
            run.summary['phase'] = 'failed'
            raise


if __name__ == '__main__':
    main()
