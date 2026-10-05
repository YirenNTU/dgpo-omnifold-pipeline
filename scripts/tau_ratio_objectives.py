"""Matched log-ratio objectives; no clipping, tempering or self-normalization.

MLC = UKL up to an additive constant. nnUKL is Eq. (5), UKL instance,
Kato & Teshima (ICML 2021). The common factor 1/2 matches paired BCE.
The optional distributed reduction happens BEFORE the nonlinear correction.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def global_means(terms, *, distributed=True):
    """Global event means with a local autograd surrogate for DDP averaging.

All ranks see identical forward values. Local derivatives are multiplied by
world size; subsequent DDP gradient averaging gives the global-batch gradient,
including unequal local counts. Do not apply another loss/world division.
"""
    local = torch.stack([value.sum() for value in terms])
    n = len(terms[0])
    if not n or any(value.shape != (n,) for value in terms):
        raise ValueError('Expected nonempty aligned one-dimensional terms')
    dist = torch.distributed
    if not distributed or not dist.is_available() or not dist.is_initialized():
        return local / n
    totals = torch.cat((local.detach(), local.new_tensor([n])))
    dist.all_reduce(totals)
    return totals[:-1] / totals[-1] + (local-local.detach()) * dist.get_world_size() / totals[-1]


def ratio_objective(positive, negative, weight, kind, c=0., *, distributed=True):
    """Truth positive, generated negative; both outputs are log p/q.

Weights have a fixed full-fit-population normalization, NOT minibatch
normalization. c is a risk-correction hyperparameter, not a ratio cap.
Population preservation requires c < 1/sup(p/q), unverified in this study.
"""
    if kind not in ('mlc', 'nnukl'):
        raise ValueError('Ratio objective must be mlc or nnukl')
    if not math.isfinite(c) or (kind == 'nnukl' and not 0 < c < 1) or (kind == 'mlc' and c != 0):
        raise ValueError('Require c=0 for MLC and 0<c<1 for nnUKL')
    if positive.ndim != 1 or positive.shape != negative.shape or weight.shape != positive.shape:
        raise ValueError('Expected aligned paired logits and weights')
    # Float64 for exponential risk arithmetic only; model/parameters remain FP32.
    p, q, w = positive.double(), negative.double(), weight.double()
    valid = torch.isfinite(p).all() & torch.isfinite(q).all() & torch.isfinite(w).all() & (w >= 0).all()
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        valid = valid.to(torch.int32)
        torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
    if not bool(valid):
        raise ValueError('Nonfinite logits or invalid event weights')
    p, q = torch.where(w > 0, p, 0.), torch.where(w > 0, q, 0.)
    ep, eq, mp, bce = global_means((w*p.exp(), w*q.exp(), w*p,
        .5*w*(F.softplus(-p)+F.softplus(q))), distributed=distributed)
    if not bool(torch.isfinite(torch.stack((ep, eq, mp, bce))).all()):
        raise FloatingPointError('Exponential ratio risk overflowed; no silent cap is applied')
    raw_risk = eq-c*ep
    mlc = .5*(eq-mp)
    correction = .5*F.relu(-raw_risk) if kind == 'nnukl' else raw_risk*0
    loss = mlc+correction
    # Equivalent to .5*(-E_p s + c E_p exp(s) + [E_q exp(s)-c E_p exp(s)]_+).
    return loss, dict(mlc=mlc, correction=correction, risk_component=raw_risk,
        correction_active=(raw_risk < 0).to(loss.dtype) if kind == 'nnukl' else raw_risk*0,
        mean_ratio_truth=ep, mean_ratio_generated=eq, paired_bce=bce)


def validation_objective(p, q, weight, kind, c):
    w = torch.as_tensor(weight, dtype=torch.float64)
    with torch.no_grad():
        loss, stats = ratio_objective(torch.as_tensor(p), torch.as_tensor(q), w/w.mean(),
                                     kind, c, distributed=False)
    return dict(objective=float(loss), **{key:float(value) for key,value in stats.items()})
