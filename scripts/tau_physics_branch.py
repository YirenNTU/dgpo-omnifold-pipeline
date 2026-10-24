"""Classifier-only additive physics branch; a single jointly trained bounded score."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def physics_features(visible_a, visible_b, deltas):
    """Candidate-owned analyzer projections and all nine products; no truth partner."""
    from scripts.tau_explicit_inputs import geometry_features, analyzer_products
    geometry = geometry_features(visible_a, visible_b, deltas)
    return np.concatenate((geometry, analyzer_products(geometry)), axis=1).astype('float32')


class PhysicsBranchClassifier(nn.Module):
    def __init__(self, base, spec, ratio_bound):
        super().__init__()
        if spec != dict(version=1, base_candidate_dim=spec.get('base_candidate_dim'), width=64):
            raise ValueError('Unknown physics branch specification')
        self.base, self.base_candidate_dim = base, int(spec['base_candidate_dim'])
        self.ratio_bound = ratio_bound
        if base.process_label_index is None or ratio_bound != 30:
            raise ValueError('Physics ablation requires process labels and bound30')
        from scripts.tau_conditioning_heads import ConditioningBlock
        # Do not perturb shared initialization or the RNG used for paired training.
        with torch.random.fork_rng(devices=[]):
            self.physics_condition = nn.Sequential(nn.Linear(base.condition_encoder[0].in_features,64), nn.SiLU())
            self.physics_label = nn.Linear(base.num_process_classes,64,bias=False)
            self.physics_input = nn.Sequential(nn.Linear(15,64),nn.SiLU())
            self.physics_block = ConditioningBlock('film',64,0.,64)
            self.physics_readout = nn.Linear(64,1)
            nn.init.zeros_(self.physics_readout.weight)
            nn.init.zeros_(self.physics_readout.bias)

    def forward(self, condition, candidate):
        if candidate.shape[1] != self.base_candidate_dim+15:
            raise ValueError('Physics candidate layout differs from saved schema')
        general = self.base(condition,candidate[:,:self.base_candidate_dim])
        label = condition[:,self.base.process_label_index]
        # base.forward already validates integer labels and their range.
        c = self.physics_condition(condition[:,self.base.continuous_condition_columns])
        c = c+self.physics_label(F.one_hot(label.long(),self.base.num_process_classes).to(c.dtype))
        h = self.physics_input(candidate[:,self.base_candidate_dim:])
        h = h+self.physics_block(h,c)
        from scripts.tau_bounded_ratio import bounded_log_ratio
        return bounded_log_ratio(general+self.physics_readout(h).flatten(),self.ratio_bound)


def build_physics_classifier(cfg, condition_dim=None, candidate_dim=None):
    from scripts.train_conditional_spin_ratio import build_classifier
    spec=cfg['physics_branch']
    if cfg.get('explicit_input') or cfg.get('cross_attention'):
        raise ValueError('Do not mix interventions in the physics branch ablation')
    dimension=cfg['candidate_dim'] if candidate_dim is None else candidate_dim
    if dimension != spec['base_candidate_dim']+15:
        raise ValueError('Physics branch requires 15 appended analyzer features')
    base_cfg=copy.deepcopy(cfg)
    base_cfg.update(physics_branch=None,candidate_dim=spec['base_candidate_dim'],ratio_bound=None)
    base=build_classifier(base_cfg,condition_dim=condition_dim)
    return PhysicsBranchClassifier(base,spec,cfg['ratio_bound'])
