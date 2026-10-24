"""Candidate-owned multiscale angular branch for classifier-only experiments."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FIELDS=['cos_opening','delta_phi_to_pi_over_pi','log_opening_core','log_phi_core',
        'sin_reco_delta_phi','cos_reco_delta_phi','cos_theta_a','cos_theta_b',
        'a_delta_theta_over_pi','a_sin_delta_phi','a_cos_delta_phi',
        'b_delta_theta_over_pi','b_sin_delta_phi','b_cos_delta_phi']


def topology_features(visible_a,visible_b,deltas):
    """No truth partner or class label; same deterministic map for both classes.

    Keep broad and smooth log-core coordinates, plus periodic angular encodings.
    Constants are fixed before fitting; no external-population normalization.
    """
    from evenet_dgpo.RL.DGPO_neutrino.diagnostics.tau_marginals import observables
    d=np.array(deltas,dtype=np.float64,copy=True)
    if d.shape!=(len(visible_a),2,2) or not np.isfinite(d).all():raise ValueError('Invalid candidate deltas')
    o=observables(dict(visible_a=visible_a,visible_b=visible_b),d)
    co=o['topology/cos_opening'];phi=o['topology/delta_phi_to_pi']
    dp=o['reco/tau_a_phi']-o['reco/tau_b_phi']
    features=[co,phi/np.pi,np.log10(1e-8+np.maximum(0.,1+co))/8.,
              np.log10(1e-8+np.maximum(0.,phi))/8.,np.sin(dp),np.cos(dp),
              np.cos(o['reco/tau_a_theta']),np.cos(o['reco/tau_b_theta'])]
    for leg in ('a','b'):
        features.extend([o[f'target/tau_{leg}_delta_theta']/np.pi,
                         np.sin(o[f'target/tau_{leg}_delta_phi']),np.cos(o[f'target/tau_{leg}_delta_phi'])])
    result=np.stack(features,axis=1).astype('float32')
    if not np.isfinite(result).all():raise ValueError('Nonfinite topology features')
    return result


def topology_spec(base_dim=533):
    return dict(version=1,base_candidate_dim=base_dim,width=64,fields=FIELDS,
                log_epsilon=1e-8,log_scale=8.)


class TopologyBranchClassifier(nn.Module):
    def __init__(self,base,spec,ratio_bound):
        super().__init__()
        if spec!=topology_spec(spec.get('base_candidate_dim')):raise ValueError('Unknown topology feature schema')
        if base.process_label_index is None or ratio_bound!=30:raise ValueError('Requires process labels and cap30')
        self.base,self.base_candidate_dim=base,int(spec['base_candidate_dim'])
        self.ratio_bound=ratio_bound
        from scripts.tau_conditioning_heads import ConditioningBlock
        with torch.random.fork_rng(devices=[]):
            self.topology_condition=nn.Sequential(nn.Linear(base.condition_encoder[0].in_features,64),nn.SiLU())
            self.topology_label=nn.Linear(base.num_process_classes,64,bias=False)
            self.topology_input=nn.Sequential(nn.Linear(len(FIELDS),64),nn.SiLU())
            self.topology_block=ConditioningBlock('film',64,0.,64)
            self.topology_readout=nn.Linear(64,1)
            nn.init.zeros_(self.topology_readout.weight);nn.init.zeros_(self.topology_readout.bias)

    def forward(self,condition,candidate):
        if candidate.shape[1]!=self.base_candidate_dim+len(FIELDS):raise ValueError('Wrong topology candidate layout')
        general=self.base(condition,candidate[:,:self.base_candidate_dim])
        # base.forward validates the actual process label before this lookup.
        label=condition[:,self.base.process_label_index].long()
        c=self.topology_condition(condition[:,self.base.continuous_condition_columns])
        c=c+self.topology_label(F.one_hot(label,self.base.num_process_classes).to(c.dtype))
        h=self.topology_input(candidate[:,self.base_candidate_dim:])
        h=h+self.topology_block(h,c)
        from scripts.tau_bounded_ratio import bounded_log_ratio
        return bounded_log_ratio(general+self.topology_readout(h).flatten(),self.ratio_bound)


def build_topology_classifier(cfg,condition_dim=None,candidate_dim=None):
    from scripts.train_conditional_spin_ratio import build_classifier
    if any(cfg.get(k) for k in ('physics_branch','explicit_input','cross_attention')):
        raise ValueError('Topology member must be independent of other branch interventions')
    spec=cfg['topology_branch']
    dim=cfg['candidate_dim'] if candidate_dim is None else candidate_dim
    if dim!=spec['base_candidate_dim']+len(FIELDS):raise ValueError('Topology member requires14 appended features')
    base=copy.deepcopy(cfg);base.update(topology_branch=None,candidate_dim=spec['base_candidate_dim'],ratio_bound=None)
    return TopologyBranchClassifier(build_classifier(base,condition_dim=condition_dim),spec,cfg['ratio_bound'])


def head_config(base):
    cfg=copy.deepcopy(base)
    if cfg.get('physics_branch') or cfg.get('topology_branch'):raise ValueError('Requires the general base configuration')
    cfg.update(candidate_dim=base['candidate_dim']+len(FIELDS),topology_branch=topology_spec(base['candidate_dim']))
    return cfg
