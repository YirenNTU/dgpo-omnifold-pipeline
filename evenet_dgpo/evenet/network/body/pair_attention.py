"""Observed pair geometry as additive logits in generation self-attention.

Keep each unordered pair, rather than pooling its features into event statistics.
Bias only visible-to-visible attention. Invisible queries consume the resulting
visible representations in later blocks; the final block needs no pair head.
"""
import torch
from torch import nn


class VisiblePairAttentionBias(nn.Module):
    def __init__(self, *, heads, layers, width=32, seed=20260930):
        super().__init__()
        if heads < 1 or layers < 2 or width < 4:
            raise ValueError("pair attention requires heads >= 1, layers >= 2 and width >= 4")
        self.heads = heads
        # Local RNG keeps every pre-existing parameter and the data/noise stream unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(4, width), nn.SiLU(),
                                         nn.Linear(width, width), nn.SiLU())
            self.outputs = nn.ModuleList(nn.Linear(width, heads) for _ in range(layers - 1))
            for head in self.outputs:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        self.diagnostics = {}

    def forward(self, theta, phi, energy, pt, valid):
        if valid.ndim != 2 or any(x.shape != valid.shape for x in (theta, phi, energy, pt)):
            raise ValueError("pair attention expects matching [batch, visible] inputs")
        valid = valid.bool()
        theta, phi, energy, pt = [torch.where(valid, x, 0.) for x in (theta, phi, energy, pt)]
        if any(not torch.isfinite(x).all() for x in (theta, phi, energy, pt)):
            raise ValueError("Nonfinite valid visible pair input")
        energy, pt = energy.clamp_min(0), pt.clamp_min(0)
        batch, n_visible = valid.shape
        i, j = torch.triu_indices(n_visible, n_visible, offset=1, device=theta.device)
        keep = valid[:, i] & valid[:, j]
        cos_phi = (phi[:, i] - phi[:, j]).cos()
        opening = (theta[:, i].cos() * theta[:, j].cos()
                   + theta[:, i].sin() * theta[:, j].sin() * cos_phi)
        def asymmetry(x):
            return ((x[:, i] - x[:, j]) / (x[:, i] + x[:, j]).clamp_min(1e-8)).square()
        features = torch.stack((cos_phi, opening.clamp(-1, 1),
                                asymmetry(energy), asymmetry(pt)), dim=-1)
        encoded = self.encoder(features)
        result = []
        diagnostics = {}
        for block, head in enumerate(self.outputs):
            values = torch.where(keep[..., None], head(encoded), 0.)
            bias = values.new_zeros(batch, self.heads, n_visible, n_visible)
            # Symmetric features, head-specific learned weights, zero diagonal/padding.
            bias[:, :, i, j] = values.transpose(1, 2)
            bias[:, :, j, i] = values.transpose(1, 2)
            result.append(bias)
            with torch.no_grad():
                denom = (keep.sum() * self.heads).clamp_min(1)
                diagnostics[f"block_{block}/pair_bias_rms"] = (
                    values.detach().float().square().sum() / denom).sqrt()
        self.diagnostics = diagnostics
        return result
