"""One residual density-ratio round; no physics moments enter training.

Balanced weighted BCE learns p/(w1*q/Z1). Thus the raw cumulative ratio
against q is exp(s1+s2-log(Z1)). Fix Z1 on the residual FIT split, not test.
Only the cumulative ratio is capped. Every fit restores the strict best val.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.audit_conditional_tau_ratio import class_weights, metrics
from scripts.train_conditional_spin_ratio import build_classifier, score_pair
from scripts.tau_tail_attribution import candidate_weights


def split_events(ids, seed, fractions=(.8, .2)):
    ids = np.asarray(ids).astype(str)
    fractions = np.asarray(fractions, float)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or fractions.ndim != 1
            or not np.isfinite(fractions).all() or (fractions <= 0).any()
            or not np.isclose(fractions.sum(), 1)):
        raise ValueError('Require unique event identities and positive split fractions')
    values = [int.from_bytes(hashlib.blake2b(f'tau-residual:{seed}:{x}'.encode(),
              digest_size=8).digest(), 'big') / 2**64 for x in ids]
    return np.searchsorted(np.cumsum(fractions)[:-1], values, side='right').astype('uint8')


def log_normalizer(base, log_ratio):
    scores = np.asarray(log_ratio, float)
    if scores.ndim == 1:
        scores = scores[:, None]
    return candidate_weights(base, scores)[1]


def compose(base_log_ratio, residual_logits, fit_log_z, cap=30.):
    """No truth/condition normalization/test-fit constant is accepted here."""
    a, b = np.asarray(base_log_ratio, float), np.asarray(residual_logits, float)
    if (a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not math.isfinite(fit_log_z) or not math.isfinite(cap) or cap <= 1):
        raise ValueError('Invalid residual composition')
    return np.minimum(a + b - fit_log_z, math.log(cap))


class ResidualRatioStack(torch.nn.Module):
    """Deployable log ratio on the same preprocessed condition/candidate inputs."""
    def __init__(self, base, residual, fit_log_z, cap):
        super().__init__()
        if not math.isfinite(fit_log_z) or not math.isfinite(cap) or cap <= 1:
            raise ValueError('Invalid stack normalization/cap')
        self.base, self.residual = base, residual
        self.register_buffer('fit_log_z', torch.tensor(fit_log_z, dtype=torch.float64))
        self.cap = float(cap)

    def forward(self, condition, candidate):
        total = self.base(condition, candidate).double() + self.residual(condition, candidate).double() - self.fit_log_z
        return total.clamp(max=math.log(self.cap))


def load_stack(path, device='cpu'):
    config = json.loads(Path(path).read_text())
    saved = [torch.load(config[key], map_location='cpu', weights_only=True)
             for key in ('base_checkpoint', 'residual_checkpoint')]
    for key in ('condition_dim', 'candidate_dim', 'head_kind', 'head_depth', 'condition_width', 'relative_preprocessing'):
        if saved[0].get(key) != saved[1].get(key):
            raise ValueError('Stack architectures/preprocessing differ: '+key)
    if saved[1]['fit_log_z'] != config['fit_log_z'] or saved[1]['cumulative_cap'] != config['cap']:
        raise ValueError('Stack constants differ from fitted checkpoint')
    heads = [build_classifier(s) for s in saved]
    for head, state in zip(heads, saved):
        head.load_state_dict(state['state_dict'], strict=True)
    return ResidualRatioStack(*heads, config['fit_log_z'], config['cap']).to(device).eval()


def make_data(arrays, logits, seed, audit=False):
    """Residual uses only old internal validation; fresh audits use external data.

The base checkpoint was selected on its old validation pool. That selection
dependence is disclosed; this is NOT an out-of-fold/untouched base holdout.
"""
    take = arrays['split'] == (2 if audit else 1)
    if np.shape(logits) != np.shape(arrays['split']):
        raise ValueError('Scores and prepared rows differ')
    keys = ('source_ids', 'condition', 'candidate_truth', 'candidate_generated', 'event_weight')
    data = {key: arrays[key][take] for key in keys}
    data['log_ratio'] = np.asarray(logits, float)[take]
    data['split'] = split_events(data['source_ids'], seed, (.6, .2, .2) if audit else (.8, .2))
    if len(np.unique(data['split'])) != (3 if audit else 2):
        raise ValueError('An empty split would invalidate this experiment')
    return data


def epoch_batches(size, world, batch_size, rank, seed, epoch):
    """Equal DDP forward counts; zero-weight padding, never duplicate event mass."""
    if min(size, world, batch_size) < 1 or not 0 <= rank < world:
        raise ValueError('Invalid distributed batch dimensions')
    order = np.random.default_rng(seed + epoch).permutation(size)
    width = world * batch_size
    for start in range(0, size, width):
        global_count = min(width, size-start)
        count = math.ceil(global_count/world)
        indices = np.full(count, -1, dtype=int)
        local = order[start:start+global_count][rank::world]
        indices[:len(local)] = local
        yield np.maximum(indices, 0), indices >= 0, global_count


class BestValidation:
    """Strict minimum selects weights; min_delta only controls patience."""
    def __init__(self, patience, min_delta, min_steps):
        self.patience, self.min_delta, self.min_steps = patience, min_delta, min_steps
        self.best = self.patience_best = float('inf')
        self.stale = 0

    def update(self, value, steps):
        if not math.isfinite(value):
            raise ValueError('Nonfinite validation BCE')
        improved = value < self.best
        if improved:
            self.best = value
        if value < self.patience_best-self.min_delta:
            self.patience_best, self.stale = value, 0
        elif steps >= self.min_steps:
            self.stale += 1
        return improved, steps >= self.min_steps and self.stale >= self.patience


def read_data(path):
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k] for k in f.files}
    n = len(a['source_ids'])
    if len(np.unique(a['source_ids'])) != n or any(len(a[k]) != n for k in a):
        raise ValueError('Invalid paired data identities/dimensions')
    for key in ('condition', 'candidate_truth', 'candidate_generated', 'event_weight', 'log_ratio'):
        if not np.isfinite(a[key]).all():
            raise ValueError('Nonfinite prepared data: '+key)
    if (a['event_weight'] < 0).any():
        raise ValueError('Signed weights unsupported')
    return a


def training_worker(cfg):
    """16-GPU DDP training and validation; full-split balanced class weights."""
    import ray.train
    import ray.train.torch
    import torch.distributed as dist

    ctx = ray.train.get_context()
    rank, world = ctx.get_world_rank(), ctx.get_world_size()
    if world != cfg['workers']:
        raise ValueError('Unexpected training world size')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    device = ray.train.torch.get_device()
    a = read_data(cfg['prepared'])
    fi, vi = np.flatnonzero(a['split'] == 0), np.flatnonzero(a['split'] == 1)
    wp, wq = class_weights(a['event_weight'][fi], a['log_ratio'][fi], True)
    model = ray.train.torch.prepare_model(build_classifier(cfg).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'], eta_min=cfg['min_lr'])
    selection = BestValidation(cfg['patience'], cfg['min_delta'], cfg['min_steps'])
    steps = 0
    status = torch.zeros(6, device=device, dtype=torch.float64)
    for epoch in range(cfg['epochs']):
        model.train()
        total = torch.zeros(2, device=device, dtype=torch.float64)
        lr_used = opt.param_groups[0]['lr']
        for ii, valid, global_count in epoch_batches(len(fi), world, cfg['batch_size'], rank, cfg['seed'], epoch):
            idx = fi[ii]
            c, t, g = [torch.as_tensor(a[k][idx], device=device) for k in
                       ('condition', 'candidate_truth', 'candidate_generated')]
            pw = torch.as_tensor(wp[ii]*valid, device=device, dtype=torch.float32)
            qw = torch.as_tensor(wq[ii]*valid, device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            p, q = model(torch.cat((c, c)), torch.cat((t, g))).chunk(2)
            loss_sum = .5*(pw*F.softplus(-p)+qw*F.softplus(q)).sum()
            loss = loss_sum*world/global_count
            finite = torch.isfinite(loss).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise ValueError('Nonfinite weighted training BCE')
            loss.backward()
            clip = cfg['grad_clip'] if cfg['grad_clip'] is not None else float('inf')
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            opt.step()
            steps += 1
            total[0] += loss_sum.detach().double()
            total[1] += valid.sum()
        dist.all_reduce(total)
        bare = model.module if hasattr(model, 'module') else model
        local = vi[rank::world]
        if len(local):
            p, q = score_pair(bare, a['condition'][local], a['candidate_truth'][local],
                             a['candidate_generated'][local], device, cfg['batch_size'])
        else:
            p = q = np.empty(0)
        gathered = [None]*world if rank == 0 else None
        dist.gather_object((local, p, q), gathered, dst=0)
        if rank == 0:
            ids = np.concatenate([v[0] for v in gathered])
            pp, qq = [np.concatenate([v[j] for v in gathered]) for j in (1, 2)]
            m = metrics(pp, qq, a['event_weight'][ids], a['log_ratio'][ids], True)
            improved, stop = selection.update(m['bce'], steps)
            if improved:
                # cfg contains the complete architecture/preprocessing/stack provenance.
                payload = dict(cfg, state_dict={k: v.detach().cpu() for k, v in bare.state_dict().items()},
                               epoch=epoch+1, optimizer_steps=steps, val_bce=m['bce'])
                temporary = Path(cfg['checkpoint']).with_suffix('.pending.pt')
                torch.save(payload, temporary)
                temporary.replace(cfg['checkpoint'])
            status[:] = torch.tensor([m['bce'], m['auc'], selection.best, stop,
                                       selection.stale, improved], device=device, dtype=torch.float64)
        dist.broadcast(status, src=0)
        scheduler.step()
        ray.train.report(dict(epoch=epoch+1, optimizer_steps=steps, train_bce=float(total[0]/total[1]),
            val_bce=float(status[0]), val_auc=float(status[1]), best_val_bce=float(status[2]),
            early_stopped=bool(status[3]), stale_epochs=int(status[4]), best_saved=bool(status[5]),
            lr=lr_used, next_lr=opt.param_groups[0]['lr'], grad_norm_last=float(grad)))
        if bool(status[3]):
            break
    dist.barrier()
    saved = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=True)
    bare.load_state_dict(saved['state_dict'], strict=True)
    # Endpoint always uses best validation, even if its step is before min_steps.
    ti = np.flatnonzero(a['split'] == 2)
    positions = np.arange(rank, len(ti), world)
    if len(positions):
        idx = ti[positions]
        p, q = score_pair(bare, a['condition'][idx], a['candidate_truth'][idx],
                         a['candidate_generated'][idx], device, cfg['batch_size'])
        np.savez(Path(cfg['checkpoint']).parent/f'test-rank-{rank:02d}.npz', positions=positions,
                 source_ids=a['source_ids'][idx], truth_logits=p, generated_logits=q)
    if rank == 0:
        Path(cfg['checkpoint']).with_name('fit_status.json').write_text(json.dumps(dict(
            epochs=epoch+1, optimizer_steps=steps, early_stopped=bool(status[3]),
            best_epoch=saved['epoch'], best_steps=saved['optimizer_steps'], best_val_bce=saved['val_bce'],
            minimum_fit_budget_met=steps >= cfg['min_steps'],
            selected_budget_met=saved['optimizer_steps'] >= cfg['min_steps']), indent=2)+'\n')
    dist.barrier()
