"""Visible-only nonlinear conditioning, independent of candidates and truth.

Fourier features augment (not replace) the pre-PET visible projection. Only the
final per-block modulation projections are zero initialized. Existing network
normalizations, residual gates, and timestep conditioning remain untouched.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def visible_conditioning_spec(network_cfg, *, target, feature_names, token_dim,
                              hidden_dim, num_layers, n_branches, output_dim=None,
                              attention_heads=None):
    cfg = dict(getattr(network_cfg, "VisibleConditioning", {}) or {})
    if not cfg.get(f"{target}_enabled", False):
        return None
    spec = dict(feature_names=list(feature_names), token_dim=int(token_dim),
                hidden_dim=int(hidden_dim), num_layers=int(num_layers),
                n_branches=int(n_branches), width=int(cfg.get("width", 64)),
                harmonics=list(cfg.get("harmonics", [1, 2, 3, 4])),
                theta_source=str(cfg.get("theta_source", "Part_eta")),
                phi_source=str(cfg.get("phi_source", "Part_phi")),
                log_diagnostics=bool(cfg.get("log_diagnostics", True)))
    mode = cfg.get("feature_mode", "angles")
    if mode == "kinematics":
        spec.update(feature_mode=mode, numerical_features=list(cfg.get("numerical_features", ["Part_energy", "Part_pt"])),
                    numerical_frequencies=list(cfg.get("numerical_frequencies", [.25, .5, 1., 2.])))
    elif mode != "angles":
        raise ValueError("visible conditioning feature_mode must be angles or kinematics")
    # Never change classifier PEFT specs or payloads when enabling policy readout.
    token_cfg = dict(cfg.get("diffusion_token_readout", {}) or {})
    if target == "diffusion" and token_cfg.get("enabled", False):
        if token_cfg.get("mode", "film") == "velocity":
            if output_dim is None:
                raise ValueError("velocity token readout requires the diffusion output dimension")
            token_cfg["output_dim"] = int(output_dim)
        spec["token_readout"] = token_cfg
    probe_cfg = dict(cfg.get("diffusion_reward_probe", {}) or {})
    if target == "diffusion" and probe_cfg.get("enabled", False):
        if output_dim != 2:
            raise ValueError("reward conditioning probe is only for delta-theta/phi generation")
        spec["reward_probe"] = probe_cfg
    relation_cfg = dict(cfg.get("diffusion_relation_adapter", {}) or {})
    if target == "diffusion" and relation_cfg.get("enabled", False):
        spec["relation_adapter"] = relation_cfg
    if target == "diffusion":
        fourier_enabled = cfg.get("diffusion_fourier_enabled", True)
        if type(fourier_enabled) is not bool:
            raise ValueError("diffusion_fourier_enabled must be boolean")
        if not fourier_enabled:
            spec["fourier_enabled"] = False
        scale_enabled = cfg.get("diffusion_film_scale_enabled", True)
        if type(scale_enabled) is not bool:
            raise ValueError("diffusion_film_scale_enabled must be boolean")
        if not scale_enabled:
            spec["film_scale_enabled"] = False
        pair_cfg = dict(cfg.get("diffusion_pair_attention", {}) or {})
        if pair_cfg.get("enabled", False):
            if attention_heads is None:
                raise ValueError("pair attention requires the generation head count")
            spec["pair_attention"] = dict(pair_cfg, heads=int(attention_heads))
    return spec


class VisibleConditioning(nn.Module):
    def __init__(self, *, feature_names, token_dim, hidden_dim, num_layers,
                 n_branches, width=64, harmonics=(1, 2, 3, 4),
                 theta_source="Part_eta", phi_source="Part_phi",
                 log_diagnostics=True, feature_mode="angles", numerical_features=(),
                 numerical_frequencies=(.25, .5, 1., 2.), token_readout=None, reward_probe=None,
                 relation_adapter=None, film_scale_enabled=True, pair_attention=None,
                 fourier_enabled=True):
        super().__init__()
        if type(fourier_enabled) is not bool:
            raise ValueError("fourier_enabled must be boolean")
        self.fourier_enabled = fourier_enabled
        if min(token_dim, hidden_dim, num_layers, n_branches) < 1 or width < 2:
            raise ValueError("visible conditioning dimensions must be positive (width >= 2)")
        if not harmonics or any(int(k) != k or k < 1 for k in harmonics) or len(set(harmonics)) != len(harmonics):
            raise ValueError("harmonics must be distinct positive integers")
        names = list(feature_names)
        if theta_source not in names or phi_source not in names:
            raise ValueError("visible conditioning angle sources are missing from raw feature schema")
        if theta_source not in ("Part_eta", "Part_theta"):
            raise ValueError("theta_source must be Part_eta or Part_theta")
        self.spec = dict(feature_names=names, token_dim=int(token_dim), hidden_dim=int(hidden_dim),
                         num_layers=int(num_layers), n_branches=int(n_branches), width=int(width),
                         harmonics=list(harmonics), theta_source=theta_source, phi_source=phi_source,
                         log_diagnostics=bool(log_diagnostics))
        if type(film_scale_enabled) is not bool:
            raise ValueError("film_scale_enabled must be boolean")
        if not fourier_enabled:
            self.spec["fourier_enabled"] = False
        self.film_scale_enabled = film_scale_enabled
        if not film_scale_enabled:
            self.spec["film_scale_enabled"] = False
        self.theta_index, self.phi_index = names.index(theta_source), names.index(phi_source)
        self.feature_mode = feature_mode
        extra_token_dim = 0
        if feature_mode == "kinematics":
            # Only explicitly selected magnitudes: no ID, mask, shower or global expansion.
            if (not numerical_features or len(set(numerical_features)) != len(numerical_features)
                    or any(name not in names or name in (theta_source, phi_source) for name in numerical_features)):
                raise ValueError("kinematic numerical features must be distinct known non-angle fields")
            if (not numerical_frequencies or len(set(numerical_frequencies)) != len(numerical_frequencies)
                    or any(not math.isfinite(b) or b <= 0 for b in numerical_frequencies)):
                raise ValueError("numerical frequencies must be distinct finite positive values")
            self.spec.update(feature_mode=feature_mode, numerical_features=list(numerical_features),
                             numerical_frequencies=list(numerical_frequencies))
            self.numerical_indices = [names.index(name) for name in numerical_features]
            self.register_buffer("numerical_frequencies", torch.tensor(numerical_frequencies, dtype=torch.float32))
            extra_token_dim = len(numerical_features) * (1 + 2 * len(numerical_frequencies))
        elif feature_mode != "angles":
            raise ValueError("visible conditioning feature_mode must be angles or kinematics")
        self.register_buffer("harmonics", torch.tensor(harmonics, dtype=torch.float32))
        self.token_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.encoder = nn.Sequential(nn.Linear(token_dim + 4 * len(harmonics) + extra_token_dim, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU())
        self.pool_score = nn.Linear(width, 1, bias=False)
        self.event_encoder = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(),
                                           nn.Linear(width, width), nn.SiLU(),
                                           nn.LayerNorm(width, elementwise_affine=False))
        self.modulations = nn.ModuleList(nn.Linear(width, 2 * n_branches * hidden_dim)
                                         for _ in range(num_layers))
        for head in self.modulations:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.log_diagnostics = bool(log_diagnostics)
        self.diagnostics = {}
        self.token_readout = None
        if token_readout and token_readout.get("enabled", False):
            from .token_conditioning import TokenSpecificConditioning
            if n_branches != 2:
                raise ValueError("token readout is supported only for diffusion")
            unknown = set(token_readout) - {"enabled", "block", "width", "heads", "mode", "output_dim"}
            if unknown:
                raise ValueError(f"unknown token readout options: {sorted(unknown)}")
            self.spec["token_readout"] = dict(token_readout)
            # Adding parameters must not shift the shared rollout/data RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.token_readout = TokenSpecificConditioning(
                    hidden_dim, width, width=int(token_readout.get("width", 64)),
                    heads=int(token_readout.get("heads", 4)),
                    mode=token_readout.get("mode", "film"),
                    block=token_readout.get("block", "last"),
                    output_dim=token_readout.get("output_dim"))
        self.reward_probe = None
        if reward_probe and reward_probe.get("enabled", False):
            from .reward_conditioning import RewardConditioningProbe
            if n_branches != 2 or self.token_readout is not None:
                raise ValueError("reward probe requires global-FiLM diffusion without token readout")
            if set(reward_probe) - {"enabled", "basis", "width", "seed"}:
                raise ValueError("unknown reward conditioning probe option")
            self.spec["reward_probe"] = dict(reward_probe)
            self.reward_probe = RewardConditioningProbe(token_dim, basis=reward_probe["basis"],
                width=int(reward_probe.get("width", 128)), seed=int(reward_probe.get("seed", 20260930)))

        self.relation_adapter = None
        if relation_adapter and relation_adapter.get("enabled", False):
            from .relation_conditioning import RelationConditioning
            if n_branches != 2 or self.token_readout is not None or self.reward_probe is not None:
                raise ValueError("Relation adapter requires global-FiLM diffusion only")
            if set(relation_adapter) - {"enabled", "mode", "width", "seed"}:
                raise ValueError("Unknown relation adapter options")
            if "Part_energy" not in names or "Part_pt" not in names:
                raise ValueError("Relation adapter needs observed energy and pt")
            self.relation_indices = [names.index("Part_energy"), names.index("Part_pt")]
            self.spec["relation_adapter"] = dict(relation_adapter)
            self.relation_adapter = RelationConditioning(width, 2*n_branches*hidden_dim, num_layers,
                mode=relation_adapter.get("mode", "relations"), width=int(relation_adapter.get("width",64)),
                seed=int(relation_adapter.get("seed",20260930)))

        self.pair_attention = None
        if pair_attention and pair_attention.get("enabled", False):
            from .pair_attention import VisiblePairAttentionBias
            if n_branches != 2 or self.token_readout is not None or self.reward_probe is not None:
                raise ValueError("pair attention requires global-FiLM diffusion without token/reward readout")
            if set(pair_attention) - {"enabled", "heads", "width", "seed"}:
                raise ValueError("Unknown pair attention options")
            if "Part_energy" not in names or "Part_pt" not in names:
                raise ValueError("pair attention needs observed energy and pt")
            self.pair_indices = [names.index("Part_energy"), names.index("Part_pt")]
            self.spec["pair_attention"] = dict(pair_attention)
            self.pair_attention = VisiblePairAttentionBias(
                heads=int(pair_attention["heads"]), layers=num_layers,
                width=int(pair_attention.get("width", 32)),
                seed=int(pair_attention.get("seed", 20260930)))

    def forward(self, raw, tokens, mask, *, normalized_raw=None, return_tokens=False,
                return_attention_bias=False):
        if return_attention_bias and (self.pair_attention is None or return_tokens):
            raise ValueError("attention bias requires an enabled pair branch and no token readout")
        if raw.ndim != 3 or tokens.ndim != 3 or raw.shape[:2] != tokens.shape[:2]:
            raise ValueError("visible conditioning requires matching [batch, visible slots, features]")
        if raw.shape[-1] != len(self.spec["feature_names"]) or tokens.shape[-1] != self.spec["token_dim"]:
            raise ValueError("visible conditioning feature schema/width mismatch")
        valid = mask.squeeze(-1).bool() if mask.ndim == 3 else mask.bool()
        if valid.shape != raw.shape[:2] or raw.shape[1] == 0:
            raise ValueError("visible conditioning requires a mask and at least one padded slot")
        raw = torch.where(valid[..., None], raw, 0.)
        tokens = torch.where(valid[..., None], tokens, 0.)
        theta = raw[..., self.theta_index]
        if self.spec["theta_source"] == "Part_eta":
            small = 2. * torch.atan(torch.exp(-theta.abs()))
            theta = torch.where(theta >= 0., small, math.pi - small)
        phi = raw[..., self.phi_index]
        k = self.harmonics.to(raw)
        angles = torch.cat((theta[..., None] * k, phi[..., None] * k), dim=-1)
        fourier = torch.cat((angles.sin(), angles.cos()), dim=-1).to(tokens)
        if not self.fourier_enabled:
            fourier = torch.zeros_like(fourier)
        token_parts = [self.token_norm(tokens), fourier]
        if self.feature_mode == "kinematics":
            if normalized_raw is None or normalized_raw.shape != raw.shape:
                raise ValueError("kinematic conditioning requires pre-projection normalized observed inputs")
            numeric = torch.where(valid[..., None], normalized_raw, 0.)[..., self.numerical_indices].to(tokens)
            numeric_fourier = self._numerical_fourier(numeric)
            if not self.fourier_enabled:
                numeric_fourier = torch.zeros_like(numeric_fourier)
            token_parts.extend((numeric, numeric_fourier))
        encoded = self.encoder(torch.cat(token_parts, dim=-1))
        count = valid.sum(dim=1, keepdim=True)
        # Finite sentinel permits empty visible events without softmax NaNs.
        scores = self.pool_score(encoded).squeeze(-1).masked_fill(~valid, torch.finfo(encoded.dtype).min)
        weights = scores.softmax(dim=1) * valid.to(encoded.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.)
        pooled = (encoded * weights[..., None]).sum(dim=1)
        context = self.event_encoder(torch.cat((pooled, count.to(pooled).log1p()), dim=-1))
        present = (count > 0).to(context)
        outputs = [head(context) * present for head in self.modulations]
        if self.relation_adapter is not None:
            from .relation_conditioning import pair_summary
            relations, pair_present = pair_summary(theta, phi, raw[...,self.relation_indices[0]],
                                                   raw[...,self.relation_indices[1]], valid)
            additions = self.relation_adapter(context, relations, pair_present)
            outputs = [base + addition for base, addition in zip(outputs, additions)]
        if self.log_diagnostics:
            with torch.no_grad():
                packed = torch.stack(outputs).detach().float().reshape(
                    len(outputs), raw.shape[0], self.spec["n_branches"], 2, self.spec["hidden_dim"])
                self.diagnostics = {"context_rms": (context.detach().float() * present).square().mean().sqrt(),
                                    "scale_rms": packed[..., 0, :].square().mean().sqrt(),
                                    "shift_rms": packed[..., 1, :].square().mean().sqrt()}
                if self.relation_adapter is not None:
                    delta = torch.stack(additions).detach().float().reshape(
                        len(outputs), raw.shape[0], self.spec["n_branches"], 2, self.spec["hidden_dim"])
                    self.diagnostics["relation_scale_rms"] = delta[...,0,:].square().mean().sqrt()
                    self.diagnostics["relation_shift_rms"] = delta[...,1,:].square().mean().sqrt()
                if self.feature_mode == "kinematics":
                    self.diagnostics["numerical_input_rms"] = numeric.detach().float().square().sum().div(
                        (valid.sum() * max(1, len(self.numerical_indices))).clamp_min(1)).sqrt()
        modulations = [output.chunk(2 * self.spec["n_branches"], dim=-1) for output in outputs]
        modulations = [self.apply_scale_policy(parts) for parts in modulations]
        if self.log_diagnostics:
            self.diagnostics["effective_scale_rms"] = (
                self.diagnostics["scale_rms"] if self.film_scale_enabled
                else self.diagnostics["scale_rms"].new_zeros(()))
        if return_attention_bias:
            biases = self.pair_attention(theta, phi, raw[..., self.pair_indices[0]],
                                         raw[..., self.pair_indices[1]], valid)
            if self.log_diagnostics:
                self.diagnostics.update(self.pair_attention.diagnostics)
            return modulations, biases
        if return_tokens:
            return modulations, torch.where(valid[..., None], encoded, 0.), valid
        return modulations

    def apply_scale_policy(self, modulation):
        """Disable only FiLM gamma; keep beta, LayerNorm and residual LayerScale."""
        if self.film_scale_enabled:
            return modulation
        # Keep checkpoint tensor shapes intact; gamma rows are inactive in this arm.
        return tuple(torch.zeros_like(value) if i % 2 == 0 else value
                     for i, value in enumerate(modulation))

    def _numerical_fourier(self, values):
        phase = (values[..., None] * self.numerical_frequencies.to(values)).flatten(-2)
        return torch.cat((phase.sin(), phase.cos()), -1)


def modulate_visible_condition(normed, scale, shift, mask):
    """Only valid invisible queries are modulated; visible tokens stay unchanged."""
    # Legacy event-global FiLM [B,D] and additive token-specific FiLM [B,L,D].
    if scale.ndim == 2:
        scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
    delta = normed * scale + shift
    return normed + torch.where(mask.bool(), delta, torch.zeros_like(delta))
