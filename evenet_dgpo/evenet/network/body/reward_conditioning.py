"""Matched, zero-output late-policy conditioning probes (not a new DGPO loss).

The two arms have identical parameters. Only a deterministic angle basis changes:
individual noisy reconstructed directions versus pair differences/sums. These are
noisy coordinate features, NOT estimates of the clean truth at large diffusion t.
"""
from __future__ import annotations

import math
import torch
from torch import nn


class RewardConditioningProbe(nn.Module):
    def __init__(self, token_dim: int, *, basis: str, width: int = 128,
                 seed: int = 20260930):
        super().__init__()
        if basis not in ("individual", "relative") or width < 2:
            raise ValueError("conditioning probe requires individual/relative basis and positive width")
        self.basis = basis
        self.disabled = False
        # Identical initialization across arms/ranks without moving rollout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.token_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
            self.token_encoder = nn.Sequential(nn.Linear(token_dim, width), nn.SiLU(),
                                               nn.Linear(width, width), nn.SiLU())
            # pooled visible context, normalized x_t, t bank, angles, slot mask
            self.encoder = nn.Sequential(nn.Linear(width + 4 + 5 + 32 + 2, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU(),
                                         nn.LayerNorm(width, elementwise_affine=False))
            self.output = nn.Linear(width, 4)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        self.diagnostics = {}

    def angle_features(self, physical_noisy_delta, batch, present=None):
        if physical_noisy_delta.shape[1:] != (2, 2):
            raise ValueError("probe is specific to two tau slots ordered [delta_theta, delta_phi]")
        directions = []
        for leg in ("a", "b"):
            # Deliberate allowlist: no x_invisible or truth keys are read.
            xyz = [batch[f"lead_{leg}_visible_{axis}"].reshape(-1).to(physical_noisy_delta)
                   for axis in ("px", "py", "pz")]
            if present is not None:
                xyz = [torch.where(present, value, 0.) for value in xyz]
            px, py, pz = xyz
            directions.append(torch.stack((torch.atan2((px.square() + py.square() + 1e-12).sqrt(), pz),
                                           torch.atan2(py, px)), -1))
        angles = torch.stack(directions, 1) + physical_noisy_delta
        if self.basis == "relative":
            # Same four angular channels / 32 bounded Fourier channels as B.
            angles = torch.stack((angles[:, 0] - angles[:, 1], angles[:, 0] + angles[:, 1]), 1)
        phase = angles.flatten(1)[..., None] * torch.arange(1, 5, device=angles.device, dtype=angles.dtype)
        return torch.cat((phase.sin(), phase.cos()), -1).flatten(1)

    def forward(self, *, tokens, visible_mask, noisy, physical_noisy_delta, time, invisible_mask, batch):
        if noisy.shape[1:] != (2, 2):
            raise ValueError("probe requires normalized noisy [B,2,2], not clean targets")
        valid = visible_mask.reshape(tokens.shape[:2]).bool()
        iv = invisible_mask.reshape(noisy.shape[:2]).bool()
        if self.disabled:
            return torch.zeros_like(noisy)
        safe_tokens = torch.where(valid[..., None], tokens, 0.)
        encoded = self.token_encoder(self.token_norm(safe_tokens))
        pooled = torch.where(valid[..., None], encoded, 0.).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        safe_noisy = torch.where(iv[..., None], noisy, 0.)
        safe_physical = torch.where(iv[..., None], physical_noisy_delta, 0.)
        present = iv.all(1) & valid.any(1)
        angles = self.angle_features(safe_physical, batch, present=present)
        t = time.reshape(len(noisy), -1)
        if t.shape[1] != 1:
            raise ValueError("probe expects a single diffusion time per event")
        tf = torch.cat((t, (math.pi*t).sin(), (math.pi*t).cos(),
                        (2*math.pi*t).sin(), (2*math.pi*t).cos()), -1)
        inputs = torch.cat((pooled, safe_noisy.flatten(1), tf, angles, iv.to(noisy)), -1)
        result = self.output(self.encoder(inputs)).reshape_as(noisy)
        # Two-leg relation is undefined if either leg is absent.
        result = torch.where(present[:, None, None], result, 0.)
        with torch.no_grad():
            self.diagnostics = {"probe_residual_rms": result.detach().float().square().mean().sqrt()}
        return result
