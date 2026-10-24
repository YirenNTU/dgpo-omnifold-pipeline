"""Candidate-owned local angular features; no energy or truth-only inputs."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCALES = (.005, .02, .05)
CENTERS = (-3., -2., -1., 0., 1., 2., 3.)
PAIRS = ((0, 1), (2, 3), (0, 2), (0, 3), (1, 2), (1, 3))
FEATURE_DIM = 246  # 3 * (4 * 7 + 6 * 3 * 3)
PRIMARY = ('Mean absolute two-leg delta-phi core-mass error at |delta phi|<0.01 rad: '
           'residual must beat General, continued General and unweighted; all upper paired CI95 < 0')


def feature_spec():
    return dict(version=1, scales_rad=list(SCALES), centers_in_sigma=list(CENTERS),
                coordinate_order=['a_theta', 'a_phi', 'b_theta', 'b_phi'],
                periodic_phi='2*sin((phi-center)/2)', pairs=[list(p) for p in PAIRS],
                pair_centers_in_sigma=[-1., 0., 1.], width=64, dimension=FEATURE_DIM)


def fine_angle_features(visible_a, visible_b, deltas):
    # Signature matches other candidate feature maps. No visible/truth p4 is used.
    d = np.asarray(deltas, dtype=np.float64)
    if d.ndim != 3 or d.shape[1:] != (2, 2) or not np.isfinite(d).all():
        raise ValueError('Require finite candidate-owned angular offsets [N,2,2]')
    x = d.reshape(-1, 4)
    blocks = []
    for scale in SCALES:
        distance = x[:, :, None] - scale*np.array(CENTERS)[None, None, :]
        distance[:, (1, 3), :] = 2*np.sin(distance[:, (1, 3), :]/2)
        basis = np.exp(-.5*np.square(distance/scale))
        blocks.append(basis.reshape(len(x), -1))
        for i, j in PAIRS:
            blocks.append((basis[:, i, 2:5, None]*basis[:, j, None, 2:5]).reshape(len(x), -1))
    result = np.concatenate(blocks, axis=1).astype(np.float32)
    if result.shape != (len(x), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError('Invalid local angular features')
    return result


class FrozenGeneralFineAngles(nn.Module):
    """Frozen fitted General + zero-initialized conditioned angular correction."""
    def __init__(self, base, dimension):
        super().__init__()
        from scripts.tau_conditioning_heads import ConditioningBlock
        self.base = base.eval().requires_grad_(False)
        self.base_candidate_dim = dimension
        self.condition = nn.Sequential(nn.Linear(base.condition_encoder[0].in_features, 64), nn.SiLU())
        self.label = nn.Linear(base.num_process_classes, 64, bias=False)
        self.input = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.SiLU())
        self.block = ConditioningBlock('film', 64, 0., 64)
        self.readout = nn.Linear(64, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, condition, candidate):
        from scripts.tau_bounded_ratio import bounded_log_ratio
        if candidate.shape[1] != self.base_candidate_dim+FEATURE_DIM:
            raise ValueError('Wrong fine-angle candidate layout')
        general = self.base(condition, candidate[:, :self.base_candidate_dim])
        label = condition[:, self.base.process_label_index].long()
        context = self.condition(condition[:, self.base.continuous_condition_columns])
        context = context+self.label(F.one_hot(label, self.base.num_process_classes).to(context.dtype))
        h = self.input(candidate[:, self.base_candidate_dim:])
        return bounded_log_ratio(general+self.readout(h+self.block(h, context)).flatten(), 30)
