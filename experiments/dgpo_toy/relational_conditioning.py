"""Two visible directions encode the SAME conditional cube, not a new truth.

Only the frozen source can decode scalar c. Learned policy corrections and
classifiers receive four visible Cartesian coordinates, never c or truth g(c).
The relative-feature arm is an explicit structural-prior diagnostic control.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from . import conditional as native
from .film_conditioning import time_features

REPRESENTATIONS = ('raw', 'particle', 'relative')


def encode_visible(c, rotation):
    """Uniform common rotation makes either visible direction label-uninformative."""
    if c.shape[-1] != 1:
        raise ValueError('Expected scalar c')
    angle = torch.cat((rotation + math.pi*c/2, rotation - math.pi*c/2), -1)
    return torch.stack((angle.cos(), angle.sin()), -1).flatten(-2)


def rotate_visible(visible, angle):
    xy = visible.reshape(*visible.shape[:-1], 2, 2)
    angle = torch.as_tensor(angle, dtype=visible.dtype, device=visible.device)
    x, y = xy.unbind(-1)
    return torch.stack((x*angle.cos()-y*angle.sin(), x*angle.sin()+y*angle.cos()), -1).flatten(-2)


def relative_components(visible):
    if visible.shape[-1] != 4:
        raise ValueError('Learned models accept two visible (x,y) vectors only')
    x1, y1, x2, y2 = visible.unbind(-1)
    return x1*x2+y1*y2, y1*x2-x1*y2  # cos(delta), sin(delta)


def decode_condition(visible):
    """For data generation, diagnostics and the immutable source ONLY."""
    cosine, sine = relative_components(visible)
    return (torch.atan2(sine, cosine)/math.pi)[..., None]


def harmonics(cosine, sine):
    """k=1..4 without atan2, avoiding an artificial angle-wrap discontinuity."""
    cc, ss = cosine, sine
    parts = []
    for _ in range(4):
        parts.extend((ss, cc))
        cc, ss = cc*cosine-ss*sine, ss*cosine+cc*sine
    return torch.stack(parts, -1)


def condition_features(visible, representation):
    if representation not in REPRESENTATIONS or visible.shape[-1] != 4:
        raise ValueError('Unknown representation or non-visible condition')
    xy = visible.reshape(*visible.shape[:-1], 2, 2)
    if representation == 'particle':
        extra = harmonics(xy[..., 0], xy[..., 1]).flatten(-2)
    elif representation == 'relative':
        extra = harmonics(*relative_components(visible)).repeat_interleave(2, dim=-1)
    else:
        extra = visible.new_zeros(*visible.shape[:-1], 16)
    # B/C have equal extra-feature RMS, width and nominal parameter count.
    # Repetition in C matches amplitude, NOT active rank/effective capacity.
    return torch.cat((visible, extra), -1)


class RelationalCritic(nn.Module):
    def __init__(self, features='relative', width=32):
        super().__init__()
        if features not in REPRESENTATIONS:
            raise ValueError(features)
        self.feature_mode = features
        self.net = nn.Sequential(nn.Linear(47, width), nn.GELU(), nn.Linear(width, width),
                                 nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def features(self, y, c):
        if c.shape[-1] != 4:
            raise ValueError('Classifier cannot consume a scalar condition')
        c = c.expand(*y.shape[:-1], 4)
        cond = condition_features(c, self.feature_mode)
        phase = y * (math.pi/2)
        # Output Fourier is present and identical in ALL three arms.
        yf = torch.cat([f(k*phase) for k in range(1, 5) for f in (torch.sin, torch.cos)], -1)
        return torch.cat((c, y, yf, cond[..., 4:]), -1)

    def forward(self, y, c, data=None):
        return self.net(self.features(y, c)).squeeze(-1)


def lift_panels(panels, seed):
    result = {}
    for offset, split in enumerate(('train', 'validation', 'test')):
        p = panels[split]
        rotation = (2*torch.rand(p['c'].shape, generator=native.generator(seed+offset))-1)*math.pi
        result[split] = {**p, 'scalar_c': p['c'].clone(), 'c': encode_visible(p['c'], rotation)}
    return result


class RelationalData:
    def __init__(self, base):
        self.base, self.cfg, self.centers = base, base.cfg, base.centers

    def contexts(self, n, rng):
        c = self.base.contexts(n, rng)
        rotation = (2*torch.rand(n, 1, generator=rng)-1)*math.pi
        return encode_visible(c, rotation)

    def sample(self, c, rng, *, truth=False):
        return self.base.sample(decode_condition(c), rng, truth=truth)

    def joint_signal(self, y, c):
        return self.base.joint_signal(y, decode_condition(c))


def as_frozen_buffers(module):
    """Native policy_train calls requires_grad_(True); buffers cannot be unfrozen.

    State dicts preserve the original names, but AdamW and gradient diagnostics
    see ONLY the learned correction parameters, never either frozen anchor.
    """
    module = copy.deepcopy(module).eval()
    for child in module.modules():
        for name, parameter in list(child.named_parameters(recurse=False)):
            delattr(child, name)
            child.register_buffer(name, parameter.detach().clone())
    return module


class VisibleSource(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = as_frozen_buffers(source)

    def forward(self, x, t, c):
        return self.source(x, t, decode_condition(c))


class RelationalPolicy(nn.Module):
    """Fixed source + trainable correction; same initial function for every arm.

    v = source(x,t,decoded_visible) + F(x,t,visible) - F_initial(x,t,zero_c).
    F's old scalar-c input is always zero. Only its visible FiLM branch supplies
    trainable conditioning. This is a residual-transfer experiment, not a claim
    that a de-novo visible-conditioned pretraining architecture was tested.
    """
    def __init__(self, source, cfg, representation='raw', width=32):
        super().__init__()
        if representation not in REPRESENTATIONS:
            raise ValueError(representation)
        self.representation = representation
        self.base = VisibleSource(source)
        self.anchor = as_frozen_buffers(source)
        self.network = copy.deepcopy(source.network)
        self.condition_input = nn.Linear(20, width)
        self.condition_hidden = nn.Sequential(nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.condition_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition_heads = nn.ModuleList([nn.Linear(width, 2*cfg.hidden) for _ in range(3)])
        for head in self.condition_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def correction(self, x, t, c, diagnostics=False):
        if c.shape[-1] != 4:
            raise ValueError('Correction cannot consume a scalar condition')
        t = t.expand(x.shape[:-1])
        c = c.expand(*x.shape[:-1], 4)
        z = self.condition_norm(self.condition_hidden(self.condition_input(
            condition_features(c, self.representation))))
        hidden = torch.cat((x, torch.zeros_like(c[..., :1]), time_features(t)), -1)
        stats = {'encoder_rms': float(z.detach().square().mean().sqrt())} if diagnostics else {}
        for i, head in enumerate(self.condition_heads):
            gamma, beta = head(z).chunk(2, -1)
            raw = self.network[2*i](hidden)
            modified = raw*(1+gamma)+beta
            hidden = self.network[2*i+1](modified)
            if diagnostics:
                stats[f'layer{i}'] = {
                    'scale_rms': float(gamma.square().mean().sqrt()),
                    'shift_rms': float(beta.square().mean().sqrt()),
                    'residual_over_hidden_rms': float((modified-raw).square().mean().sqrt()/
                                                     raw.square().mean().sqrt().clamp_min(1e-12))}
        velocity = native.alpha_sigma(t)[1][..., None]*self.network[-1](hidden)
        zero = self.anchor(x, t, torch.zeros_like(c[..., :1]))
        return velocity-zero, hidden, stats

    def predict_features(self, x, t, c):
        correction, hidden, _ = self.correction(x, t, c)
        return self.base(x, t, c)+correction, hidden

    def forward(self, x, t, c):
        return self.predict_features(x, t, c)[0]


@torch.no_grad()
def verify_initial(source, cfg, models):
    rng = native.generator(81917)
    c = 1.98*torch.rand(128, 1, generator=rng)-.99
    visible = encode_visible(c, (2*torch.rand(128, 1, generator=rng)-1)*math.pi)
    x, t = torch.randn(128, 3, generator=rng), torch.rand(128, generator=rng)
    base = VisibleSource(source)
    reference_v = base(x, t, visible)
    reference_y = native.ddim(base, visible, x, cfg.ddim_steps)
    result = {'arms': {}, 'source_roundtrip_velocity_max': float((source(x,t,c)-reference_v).abs().max()),
              'source_roundtrip_samples_max': float((native.ddim(source,c,x,cfg.ddim_steps)-reference_y).abs().max())}
    for name, model in models.items():
        result['arms'][name] = {'velocity_bitwise': torch.equal(model(x,t,visible), reference_v),
            'samples_bitwise': torch.equal(native.ddim(model,visible,x,cfg.ddim_steps), reference_y)}
    if not all(all(item.values()) for item in result['arms'].values()):
        raise ValueError('Initial policy functions differ')
    if max(result['source_roundtrip_velocity_max'], result['source_roundtrip_samples_max']) > 1e-5:
        raise ValueError('Geometry round-trip changes the archived source materially')
    return result
