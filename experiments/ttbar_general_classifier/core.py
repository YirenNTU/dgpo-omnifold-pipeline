"""TTbar adaptation of the Ztautau General FiLM head; no policy optimization."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConditioningBlock(nn.Module):
    # Same operations/parameter layout as scripts.tau_conditioning_heads.
    def __init__(self, dropout=.1, context_width=256):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.context = nn.Linear(context_width, 128, bias=False)
        self.hidden = nn.Linear(64, 128)
        self.output = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.context.weight)

    def forward(self, h, c):
        gamma, beta = self.context(c).chunk(2, -1)
        return self.output(self.dropout(F.silu(self.hidden((1+gamma)*self.norm(h)+beta))))


class GeneralClassifier(nn.Module):
    def __init__(self, condition_dim, candidate_dim, dropout=.1, ratio_bound=30., classes=1):
        super().__init__()
        self.condition_encoder = nn.Sequential(nn.Linear(condition_dim, 256), nn.SiLU(),
                                                nn.Linear(256, 256), nn.SiLU())
        self.process_label_dense = nn.Linear(classes, 256, bias=False)
        self.spin_encoder = nn.Sequential(nn.Linear(candidate_dim, 64), nn.SiLU(),
                                           nn.Linear(64, 64), nn.SiLU())
        self.blocks = nn.ModuleList([ConditioningBlock(dropout) for _ in range(3)])
        self.readout = nn.Linear(64, 1)
        self.ratio_bound, self.classes = ratio_bound, classes

    def forward(self, condition, candidate, labels):
        c = self.condition_encoder(condition) + self.process_label_dense(
            F.one_hot(labels.long(), self.classes).to(condition.dtype))
        h = self.spin_encoder(candidate)
        for block in self.blocks:
            h = h + block(h, c)/math.sqrt(3)
        z = self.readout(h).flatten()
        if self.ratio_bound is not None:
            z = math.log(self.ratio_bound)-F.softplus(math.log(self.ratio_bound-1)-z)
        return z


def wrap(phi):
    return (phi+np.pi) % (2*np.pi)-np.pi


def nu_p4(x):
    """Native EveNet coordinates (log1p pt, eta, phi); same nu2flow convention."""
    x = np.asarray(x, dtype=np.float64)
    pt, eta, phi = np.expm1(x[..., 0]), x[..., 1], x[..., 2]
    return np.stack((pt*np.cosh(eta), pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)), -1)


def geometry(x):
    x = np.asarray(x, dtype=np.float64)
    direct = np.stack((x[..., 0], x[..., 1], np.sin(x[..., 2]), np.cos(x[..., 2])), -1)
    eta, phi = x[..., 1], x[..., 2]
    u = np.stack((np.cos(phi)/np.cosh(eta), np.sin(phi)/np.cosh(eta), np.tanh(eta)), -1)
    joint = np.einsum('...i,...j->...ij', u[..., 0, :], u[..., 1, :]).reshape(*x.shape[:-2], 9)
    return np.concatenate((direct.reshape(*x.shape[:-2], 8), u.reshape(*x.shape[:-2], 6), joint), -1).astype('float32')


def event_partition(ids, seed=42):
    # Selection never enters physics. Remaining events are split response/data/reference.
    vals = np.array([int.from_bytes(hashlib.blake2b(f'{seed}:{s}'.encode(), digest_size=8).digest(), 'little') % 100 for s in ids])
    return np.select((vals<20, vals<60, vals<80), (0, 1, 2), default=3).astype('int8')


def mixture(logits, candidate_top, truth_top, scale, alpha=.3, temperature=1.):
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError('Invalid mixture')
    logits = np.asarray(logits, float)/temperature
    score = np.exp(logits-logits.max(1, keepdims=True))
    general = score/score.sum(1, keepdims=True)
    scale = np.asarray(scale)
    if scale.shape != (4,) or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Require fixed training-only four-momentum scales')
    distance = (((candidate_top-truth_top[:, None])/scale)**2).mean(axis=(-1,-2))
    if not np.isfinite(distance).all():
        raise ValueError('Nonfinite oracle distance; no clipping/substitution')
    oracle = np.zeros_like(general)
    oracle[np.arange(len(oracle)), distance.argmin(1)] = 1
    return general, (1-alpha)*general+alpha*oracle, distance


def p4_from_columns(extra, prefix, mass=False):
    pt, eta, phi = [extra[f'{prefix}/{x}'] for x in ('pt','eta','phi')]
    px, py, pz = pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)
    e = np.sqrt((pt*np.cosh(eta))**2+extra[f'{prefix}/mass']**2) if mass else extra[f'{prefix}/energy']
    return np.stack((e, px, py, pz), -1)


def required_extra():
    return [f'EXTRA/t{i}/{obj}/{v}' for i in (1,2) for obj in ('b','l') for v in ('pt','eta','phi','energy')] + [f'EXTRA/truth_t{i}/{obj}/{v}' for i in (1,2) for obj in ('t','l') for v in ('pt','eta','phi','mass')]


def reconstructed_visible(extra):
    return sum(np.stack([p4_from_columns(extra,f'EXTRA/t{i}/{obj}') for i in (1,2)],1) for obj in ('b','l'))


def oracle_reference_p4(truth, extra, reference):
    if reference == 'reco_top_truthnu_p4':
        return reconstructed_visible(extra)+nu_p4(truth)
    if reference == 'top_p4':
        return np.stack([p4_from_columns(extra,f'EXTRA/truth_t{i}/t',True) for i in (1,2)],1)
    raise ValueError(f'Unknown oracle reference {reference}')


def top_oracle_inputs(candidates, extra, truth=None, reference='top_p4'):
    """Same ordered reconstructed visible on both sides for reco_top_truthnu_p4.

    Historical top_p4 retains MC top reference. Neither reference is supplied
    to the classifier. Source b/lepton assignment is oracle/evaluation only.
    """
    if reference=='reco_top_truthnu_p4' and truth is None:
        raise ValueError('MC neutrino target required; no MC top fallback')
    return reconstructed_visible(extra)[:,None]+nu_p4(candidates), oracle_reference_p4(truth,extra,reference)


def mass(p):
    return np.sqrt(np.maximum(p[..., 0]**2-(p[..., 1:]**2).sum(-1), 0))


def pair_quantities(tops, leptons):
    pair=tops.sum(-2)
    yarg=(pair[..., 0]+pair[..., 3])/(pair[..., 0]-pair[..., 3])
    if (yarg<=0).any() or not np.isfinite(yarg).all():
        raise ValueError('Nonphysical top-pair rapidity; no silent candidate cuts')
    return dict(m_tt=mass(pair), pt_t1=np.linalg.norm(tops[...,0,1:3],axis=-1),
                pt_t2=np.linalg.norm(tops[...,1,1:3],axis=-1),
                pt_tt=np.linalg.norm(pair[...,1:3],axis=-1), y_tt=.5*np.log(yarg),
                dphi_ll=np.abs(wrap(np.arctan2(leptons[...,0,2],leptons[...,0,1])-
                                    np.arctan2(leptons[...,1,2],leptons[...,1,1])))/np.pi)


NU2FLOW_EDGES = dict(m_tt=[0,400,500,800,np.inf], dphi_ll=[0,.25,.5,.75,1],
                    pt_t1=[0,75,125,175,np.inf], pt_t2=[0,75,125,175,np.inf],
                    pt_tt=[0,70,140,200,np.inf], y_tt=[-np.inf,-1,0,1,np.inf])


def bin_ids(x, edges):
    # pd.cut default right=True, as used by native nu2flow pre_bin.
    x=np.asarray(x); edges=np.asarray(edges)
    labels=np.searchsorted(edges,x,side='left')-1
    if not np.isfinite(x).all() or (labels<0).any() or (labels>=len(edges)-1).any():
        raise ValueError('Observable outside declared bins; no silent removal')
    return labels


def contribution(bins, weights, nbin):
    h=np.zeros((len(weights),nbin),float)
    np.add.at(h,(np.arange(len(weights))[:,None],bins),weights)
    np.testing.assert_allclose(h.sum(1),1,atol=1e-6)
    return h


def physics(truth, candidates, extra, *, include_mc_top_mass=True):
    """Outputs retain event/candidate axes; truth assignment is evaluation-only."""
    reco_l=np.stack([p4_from_columns(extra,f'EXTRA/t{i}/l') for i in (1,2)],1)
    reco_b=np.stack([p4_from_columns(extra,f'EXTRA/t{i}/b') for i in (1,2)],1)
    true_l=np.stack([p4_from_columns(extra,f'EXTRA/truth_t{i}/l',True) for i in (1,2)],1)
    true_t=np.stack([p4_from_columns(extra,f'EXTRA/truth_t{i}/t',True) for i in (1,2)],1)
    target_top=reco_l+reco_b+nu_p4(truth)
    reco_top=(reco_l+reco_b)[:,None]+nu_p4(candidates)
    results={}
    for leg,name in enumerate(('nu','nubar')):
        for j,key in enumerate(('pt','eta','phi')):
            t=truth[:,leg,j]; x=candidates[:,:,leg,j]
            if j==0: t,x=np.expm1(t),np.expm1(x)
            if j==2: t,x=wrap(t),wrap(x)
            # Same native interior bin edges, with explicit tail folding into edge bins.
            lo,hi,nb=(0,150,30) if j==0 else ((-4,4,20) if j==1 else (-np.pi,np.pi,16))
            edges=np.linspace(lo,hi,nb+1); edges[0],edges[-1]=-np.inf,np.inf
            results[f'{name}_{key}']=(t,x,edges,5)
        edges=np.linspace(100,240,29);edges[0],edges[-1]=-np.inf,np.inf
        results[f'top{leg+1}_mass_reco_truthnu']=(mass(target_top[:,leg]),mass(reco_top[:,:,leg]),edges,5)
        if include_mc_top_mass:
            results[f'top{leg+1}_mass_mc_top']=(mass(true_t[:,leg]),mass(reco_top[:,:,leg]),edges,5)
    tq=pair_quantities(true_t,true_l)
    pq=pair_quantities(reco_top,reco_l[:,None])
    for key,edges in NU2FLOW_EDGES.items():
        if key=='m_tt':continue
        t=bin_ids(tq[key],edges)+4*bin_ids(tq['m_tt'],NU2FLOW_EDGES['m_tt'])
        # dphi_ll is candidate independent, broadcast explicitly.
        v=np.broadcast_to(pq[key],pq['m_tt'].shape)
        x=bin_ids(v,edges)+4*bin_ids(pq['m_tt'],NU2FLOW_EDGES['m_tt'])
        results[f'mtt_x_{key}']=(t.astype(float),x.astype(float),np.arange(17)-.5,7)
    return results


def validate_cache_part(a):
    n=len(a['ids'])
    if a['truth'].shape!=(n,2,3) or a['candidates'].ndim!=4 or a['candidates'].shape[2:]!=(2,3):
        raise ValueError('Invalid TTbar candidate schema')
    if len(set(a['ids']))!=n:raise ValueError('Duplicate event identities')
    if a['generated_features'].shape[:2]!=a['candidates'].shape[:2]:raise ValueError('K axis lost')
    for k in ('truth','candidates','condition','truth_features','generated_features'):
        if len(a[k])!=n or not np.isfinite(a[k]).all():raise ValueError(f'Invalid cache {k}')
