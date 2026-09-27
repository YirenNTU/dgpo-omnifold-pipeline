"""Observed-token angular conditioning, never clean generated coordinates."""
import math

import torch
from torch import nn


class VisibleAngularFourier(nn.Module):
    def __init__(self, feature_names, hidden_dim, theta_source="Part_eta", phi_source="Part_phi",
                 placement="input", projection_type="linear", mlp_dim=64,
                 log_diagnostics=False, harmonics=None, attention_heads=4):
        super().__init__()
        if placement not in ("input", "output"):
            raise ValueError("angular placement must be input or output")
        if projection_type not in ("linear", "mlp", "cross_attention") or mlp_dim < 1:
            raise ValueError("Unknown angular projection_type or nonpositive mlp_dim")
        if projection_type == "cross_attention" and (placement != "output" or
                attention_heads < 1 or mlp_dim % attention_heads):
            raise ValueError("Cross-attention requires output placement and mlp_dim divisible by attention_heads")
        self.placement = placement
        self.projection_type = projection_type
        self.log_diagnostics = log_diagnostics
        self.diagnostics = {}
        if theta_source not in ("eta", "theta", "Part_eta", "Part_theta") or phi_source not in ("phi", "Part_phi"):
            raise ValueError("Use physical eta/theta and phi (radians) from visible x only")
        self.theta_index = feature_names.index(theta_source)
        self.phi_index = feature_names.index(phi_source)
        self.theta_source = theta_source
        bank = [1, 2, 3, 4] if harmonics is None else list(harmonics)
        if not bank or len(set(bank)) != len(bank) or any(
                not math.isfinite(k) or k <= 0 or int(k) != k for k in bank):
            raise ValueError("Harmonics must be distinct positive integers to preserve phi periodicity")
        self.register_buffer("harmonics", torch.tensor(bank, dtype=torch.float32),
                             persistent=projection_type == "cross_attention")
        feature_dim = 4 * len(bank)
        # Keep the legacy linear state_dict key and exact default computation.
        self.encoder = (nn.Identity() if projection_type == "linear" else nn.Sequential(
            nn.Linear(feature_dim, mlp_dim), nn.GELU(), nn.LayerNorm(mlp_dim)))
        if projection_type == "cross_attention":
            self.query_norm = nn.LayerNorm(hidden_dim)
            self.query_projection = nn.Linear(hidden_dim, mlp_dim)
            self.time_encoder = nn.Sequential(nn.Linear(5, mlp_dim), nn.SiLU())
            self.attention = nn.MultiheadAttention(mlp_dim, attention_heads,
                                                   dropout=0., batch_first=True)
        self.projection = nn.Linear(feature_dim if projection_type == "linear" else mlp_dim,
                                    hidden_dim, bias=False)
        nn.init.zeros_(self.projection.weight)

    def features(self, visible_raw, visible_mask):
        mask = visible_mask.bool()
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        # Sanitize padding before trigonometry; cos(0) padding is masked below.
        raw = torch.where(mask, visible_raw, torch.zeros_like(visible_raw))
        theta = raw[..., self.theta_index]
        if self.theta_source in ("eta", "Part_eta"):
            small = 2 * torch.atan(torch.exp(-theta.abs()))
            theta = torch.where(theta >= 0, small, math.pi - small)
        phi = raw[..., self.phi_index]
        angles = torch.stack((theta, phi), dim=-1).unsqueeze(-1)
        phase = angles * self.harmonics.to(dtype=raw.dtype)
        features = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return torch.where(mask, features, torch.zeros_like(features))

    def forward(self, visible_raw, visible_mask, query=None, time=None):
        raw = visible_raw.to(dtype=self.projection.weight.dtype)
        mask = visible_mask.bool()
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        features = self.encoder(self.features(raw, visible_mask))
        if self.projection_type == "cross_attention":
            if query is None or time is None:
                raise ValueError("Cross-attention needs noisy-token queries and diffusion time")
            memory = torch.where(mask, features, 0)
            valid = mask.squeeze(-1)
            # A null key is available ONLY for events with no visible tokens.
            # Avoid all-masked softmax NaNs without attending to padded particles.
            memory = torch.cat((memory, memory.new_zeros(memory.shape[0], 1, memory.shape[-1])), 1)
            key_padding_mask = torch.cat((~valid, valid.any(1, keepdim=True)), 1)
            t = time.to(dtype=raw.dtype)
            tf = torch.cat((t, (math.pi*t).sin(), (math.pi*t).cos(),
                            (2*math.pi*t).sin(), (2*math.pi*t).cos()), -1)
            q = self.query_projection(self.query_norm(query)) + self.time_encoder(tf)
            readout, _ = self.attention(q, memory, memory, key_padding_mask=key_padding_mask,
                                        need_weights=False)
            residual = self.projection(readout)
            return torch.where(valid.any(1)[:, None, None], residual, 0)
        residual = self.projection(features)
        # An MLP can produce nonzero padding through its biases.
        return torch.where(mask, residual, torch.zeros_like(residual))

    def inject(self, encoded, visible_raw, visible_mask, target_mask=None, time=None):
        n_visible = visible_raw.shape[1]
        if self.projection_type == "cross_attention":
            if target_mask is None or time is None:
                raise ValueError("Cross-attention injection needs full token mask and diffusion time")
            if encoded.shape[1] == n_visible:
                return encoded
            mask = target_mask[:, n_visible:].bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            base = torch.where(mask, encoded[:, n_visible:], 0)
            residual = self(visible_raw, visible_mask, query=base, time=time[:, n_visible:])
            residual = torch.where(mask, residual, 0)
            padded_residual = torch.cat((torch.zeros_like(encoded[:, :n_visible]), residual), 1)
        else:
            mask = visible_mask.bool()
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            base = torch.where(mask, encoded[:, :n_visible], 0)
            residual = self(visible_raw, visible_mask)
            padded_residual = torch.cat((residual, torch.zeros_like(encoded[:, n_visible:])), 1)
        if self.log_diagnostics:
            with torch.no_grad():
                count = (mask.sum() * encoded.shape[-1]).clamp_min(1)
                base_rms = (base.float().square().sum() / count).sqrt()
                residual_rms = (residual.float().square().sum() / count).sqrt()
                self.diagnostics = {
                    "base_rms": base_rms.detach(),
                    "residual_rms": residual_rms.detach(),
                    "residual_to_base_rms": (residual_rms / base_rms.clamp_min(1e-12)).detach(),
                }
        return encoded + padded_residual
