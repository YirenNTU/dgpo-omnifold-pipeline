"""Symmetric candidate-owned geometry inputs; no labels or kappa division."""
import numpy as np
import torch
from torch import nn
from scripts.diagnose_ztautau_cij import angles, tau_from_deltas


def validate_spec(spec):
    if spec['arm'] not in ('visible', 'geometry', 'products') or spec.get('version') != 1:
        raise ValueError('Unknown explicit-input schema')
    if spec['arm'] == 'products' and spec.get('product_fields') != product_fields():
        raise ValueError('Unknown product ordering')
    mean, scale = np.asarray(spec['visible_mean']), np.asarray(spec['visible_scale'])
    if (mean.shape != (8,) or scale.shape != (8,) or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Invalid fit-only visible normalization')
    return mean, scale


def visible_features(va, vb):
    va, vb = np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64)
    if va.shape != vb.shape or va.ndim != 2 or va.shape[1] != 4:
        raise ValueError('Visible p4 must be aligned [N,4]')
    out = np.concatenate((va, vb), axis=1)
    if not np.isfinite(out).all() or (va[:,0] <= 0).any() or (vb[:,0] <= 0).any():
        raise ValueError('Nonfinite or nonpositive visible energy')
    return out


def geometry_features(va, vb, deltas):
    """Each candidate owns its tau; no truth argument exists here."""
    if np.shape(deltas) != (len(va),2,2):
        raise ValueError('Expected one delta pair per condition')
    a, b = angles(tau_from_deltas(va, deltas[:,0]), tau_from_deltas(vb, deltas[:,1]), va, vb)
    return np.concatenate((a,b),axis=1).astype('float32')


def product_fields():
    return [f'a_{i}*b_{j}' for i in ('k','r','n') for j in ('k','r','n')]


def analyzer_products(geometry):
    """Nine outer products, kk,kr,kn,rk,rr,rn,nk,nr,nn; NOT a vector cross product.

    No factor9, analyzing power, empirical target or truth partner is used.
    """
    x = np.asarray(geometry)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all() or (np.abs(x)>1+1e-6).any():
        raise ValueError('Invalid six-direction input')
    return (x[:,:3,None]*x[:,None,3:]).reshape(len(x),9).astype('float32')


def insert_geometry(candidate, geometry, products=False):
    # Keep relative6 immediately before tau15, preserving legacy diagnostics.
    candidate, geometry = np.asarray(candidate), np.asarray(geometry)
    if (candidate.ndim != 2 or candidate.shape[1] < 21
            or geometry.shape != (len(candidate),6) or not np.isfinite(geometry).all()
            or (np.abs(geometry)>1+1e-6).any()):
        raise ValueError('Invalid six-direction input')
    extra = np.concatenate((geometry,analyzer_products(geometry)),axis=1) if products else geometry
    return np.concatenate((candidate[:,:-21],extra,candidate[:,-21:]),axis=1).astype('float32')


def extend_condition(condition, va, vb, spec):
    mean, scale = validate_spec(spec)
    extra = (visible_features(va,vb)-mean)/scale
    out = np.concatenate((condition,extra),axis=1).astype('float32')
    if not np.isfinite(out).all(): raise ValueError('Nonfinite extended condition')
    return out


def transform_panel_inputs(condition, candidate, va, vb, deltas, spec):
    c = extend_condition(condition,va,vb,spec)
    f = (insert_geometry(candidate,geometry_features(va,vb,deltas),products=spec['arm']=='products')
         if spec['arm'] in ('geometry','products') else candidate)
    return c,f


def prepare_arrays(arrays, va, vb, truth_geometry, generated_geometry, arm):
    if arm not in ('visible','geometry','products'): raise ValueError('Unknown arm')
    fit = arrays['split']==0
    if not fit.any(): raise ValueError('Empty fitting split')
    raw = visible_features(va,vb)
    mean, scale = raw[fit].mean(0),raw[fit].std(0)
    scale = np.where(scale>1e-6,scale,1.)
    spec = dict(version=1,arm=arm,visible_mean=mean.tolist(),visible_scale=scale.tolist(),
        fields=[f'visible_{leg}_{key}' for leg in ('a','b') for key in ('E','px','py','pz')],
        geometry_fields=['a_k','a_r','a_n','b_k','b_r','b_n'] if arm in ('geometry','products') else [],
        normalization='visible: fit-only mean/std without clipping; geometry: unit-direction projections',
        convention='Inherited TT2L common A basis, n=r cross k; visible-direction analyzer, not optimal polarimetry')
    if arm=='products':
        spec['product_fields']=product_fields()
        spec['product_convention']='Unscaled outer products of candidate-owned analyzer directions; no kappa division or Cij labels'
    out = dict(arrays)
    out['condition'] = extend_condition(arrays['condition'],va,vb,spec)
    out['condition_mean'] = np.r_[arrays['condition_mean'],mean].astype('float32')
    out['condition_scale'] = np.r_[arrays['condition_scale'],scale].astype('float32')
    if arm in ('geometry','products'):
        for label,geometry in [('truth',truth_geometry),('generated',generated_geometry)]:
            out['candidate_'+label]=insert_geometry(arrays['candidate_'+label],geometry,products=arm=='products')
    return out,spec


def _extend_linear(old, extra, insertion):
    with torch.random.fork_rng(devices=[]):
        new=nn.Linear(old.in_features+extra,old.out_features,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:,:insertion].copy_(old.weight[:,:insertion])
        new.weight[:,insertion+extra:].copy_(old.weight[:,insertion:])
        new.bias.copy_(old.bias)
    return new


def build_explicit_classifier(cfg, condition_dim=None, candidate_dim=None):
    from scripts.train_conditional_spin_ratio import build_classifier
    spec=cfg['explicit_input'];validate_spec(spec)
    if cfg.get('relative_dim')!=6 or cfg.get('head_kind')!='film':
        raise ValueError('Explicit input requires relative6 FiLM')
    c=condition_dim if condition_dim is not None else cfg['condition_dim']
    f=candidate_dim if candidate_dim is not None else cfg['candidate_dim']
    extra={'visible':0,'geometry':6,'products':15}[spec['arm']]
    base=dict(cfg,condition_dim=c-8,candidate_dim=f-extra,explicit_input=None)
    model=build_classifier(base)
    model.condition_encoder[0]=_extend_linear(model.condition_encoder[0],8,c-8)
    if extra:
        model.spin_encoder[0]=_extend_linear(model.spin_encoder[0],extra,f-extra-21)
    return model
