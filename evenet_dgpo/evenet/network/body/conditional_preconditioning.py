"""Frozen, observed-event-dependent coordinates for two ordered angular slots.

The auxiliary Gaussian estimates coordinates, not the final posterior. Diffusion
learns the entire residual distribution in z = L(c)^-1 (normalize(y) - mu(c)).
"""
from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import nn


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ConditionalPreconditioner(nn.Module):
    def __init__(self, feature_names, sequential_normalizer, global_normalizer,
                 target_normalizer, *, width=128, scale_floor=0.01,
                 max_diagonal=5.0, off_diagonal_bound=2.0, seed=20261001):
        super().__init__()
        if not 0 < scale_floor < 1 < max_diagonal or off_diagonal_bound <= 0 or width < 8:
            raise ValueError("Invalid conditional preconditioning bounds/width")
        if target_normalizer.mean.numel() != 2 or target_normalizer.padding:
            raise ValueError("Conditional preconditioning requires exactly two features per tau, no padding")
        if "Part_phi" not in feature_names:
            raise ValueError("Observed Part_phi is required for periodic condition encoding")
        self.spec = dict(feature_names=list(feature_names), width=int(width),
                         scale_floor=float(scale_floor), max_diagonal=float(max_diagonal),
                         off_diagonal_bound=float(off_diagonal_bound), seed=int(seed),
                         target_shape=[2, 2],
                         inverse_cdf_indices=[list(n.inv_cdf_index) for n in
                                              (sequential_normalizer, global_normalizer, target_normalizer)])
        self.phi_index = list(feature_names).index("Part_phi")
        self.sequential_normalizer = deepcopy(sequential_normalizer)
        self.global_normalizer = deepcopy(global_normalizer)
        self.target_normalizer = deepcopy(target_normalizer)
        self.register_buffer("fitted", torch.tensor(False))
        # Fingerprint travels with weights and rejects mismatched construction on load.
        digest = hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).digest()
        self.register_buffer("contract", torch.tensor(list(digest), dtype=torch.uint8))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.particle_encoder = nn.Sequential(nn.Linear(len(feature_names) + 1, width), nn.SiLU(),
                                                  nn.Linear(width, width), nn.SiLU())
            self.event_encoder = nn.Sequential(
                nn.Linear(2 * width + global_normalizer.mean.numel() + 2, width), nn.SiLU(),
                nn.Linear(width, width), nn.SiLU())
            self.output = nn.Linear(width, 14)  # four means + ten lower-triangular entries
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
            # Sigma = A A^T + floor^2 I, initialized to identity.
            ratio = math.sqrt(1 - scale_floor ** 2) / max_diagonal
            initial = math.log(ratio / (1 - ratio))
            with torch.no_grad():
                self.output.bias[torch.tensor([4, 6, 9, 13])] = initial
        self.register_buffer("initial_bias", self.output.bias.detach().clone())
        self.requires_grad_(False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        saved = state_dict.get(prefix + "contract")
        if saved is not None and not torch.equal(saved.cpu(), self.contract.cpu()):
            raise ValueError("Conditional preconditioning checkpoint/config contract differs")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def identity_initialized(self):
        return (not bool(self.fitted) and not bool(torch.count_nonzero(self.output.weight))
                and torch.equal(self.output.bias, self.initial_bias))

    @staticmethod
    def validate_slots(value, mask):
        if value.ndim != 3 or value.shape[1:] != (2, 2):
            raise ValueError("Expected ordered [tau A, tau B] x [delta theta, delta phi]")
        if mask is not None and (mask.shape not in (value.shape[:2], (*value.shape[:2], 1))
                                 or not bool(mask.bool().all())):
            raise ValueError("Full covariance requires both tau slots valid; partial masks are unsupported")

    def forward(self, batch):
        # Intentionally reads only observed fields. No target, assignment or noisy state.
        raw, valid = batch["x"], batch["x_mask"].bool()
        raw = torch.where(valid[..., None], raw, 0.)
        if raw.shape[-1] != len(self.spec["feature_names"]) or not torch.isfinite(raw).all():
            raise ValueError("Invalid observed preconditioning inputs")
        for name in ("Part_sticNumTowers", "Part_sticChargedTag"):
            if name in self.spec["feature_names"]:
                index = self.spec["feature_names"].index(name)
                if bool((raw[..., index].abs() > 1.e6).any()):
                    raise ValueError(f"Extreme valid {name}: use the verified STIC-filtered dataset")
        normalized = self.sequential_normalizer(raw, valid[..., None])
        phi = raw[..., self.phi_index:self.phi_index + 1]
        features = torch.cat((normalized[..., :self.phi_index], phi.sin(), phi.cos(),
                              normalized[..., self.phi_index + 1:]), -1)
        tokens = self.particle_encoder(features)
        count = valid.sum(1, keepdim=True)
        mean = torch.where(valid[..., None], tokens, 0.).sum(1) / count.clamp_min(1)
        maximum = tokens.masked_fill(~valid[..., None], -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, 0.)
        global_mask = batch["conditions_mask"].reshape(len(raw), 1).bool()
        global_values = self.global_normalizer(batch["conditions"].reshape(len(raw), -1), global_mask)
        if not torch.isfinite(global_values).all():
            raise ValueError("Nonfinite observed global conditions")
        context = torch.cat((mean, maximum, global_values, global_mask.to(mean), count.to(mean).log1p()), -1)
        output = self.output(self.event_encoder(context))
        mu, packed = output[:, :4], output[:, 4:]
        rows, cols = torch.tril_indices(4, 4, device=output.device)
        diagonal = rows == cols
        values = torch.where(diagonal, self.spec["max_diagonal"] * packed.sigmoid(),
                             self.spec["off_diagonal_bound"] * packed.tanh())
        a = output.new_zeros(len(raw), 4, 4)
        a[:, rows, cols] = values
        # Tiny 4x4 factorization in float64 preserves the eigenvalue floor even
        # for strongly correlated events. All neural-network work stays FP32.
        a64 = a.double()
        covariance = a64 @ a64.mT + self.spec["scale_floor"] ** 2 * torch.eye(4, device=a.device, dtype=a64.dtype)
        factor = torch.linalg.cholesky(covariance).to(output.dtype)
        return mu, factor

    def coordinates(self, batch):
        if not bool(self.fitted):
            raise RuntimeError("Fit/load conditional preconditioning before diffusion training or sampling")
        with torch.no_grad():
            return self(batch)

    @staticmethod
    def encode(y, mu, factor):
        ConditionalPreconditioner.validate_slots(y, None)
        return torch.linalg.solve_triangular(factor, (y.flatten(1) - mu)[..., None],
                                            upper=False).squeeze(-1).reshape_as(y)

    @staticmethod
    def decode(z, mu, factor):
        ConditionalPreconditioner.validate_slots(z, None)
        return (mu + (factor @ z.flatten(1)[..., None]).squeeze(-1)).reshape_as(z)

    def gaussian_loss(self, batch, coordinates=None):
        y, mask = batch["x_invisible"], batch["x_invisible_mask"]
        self.validate_slots(y, mask)
        target = self.target_normalizer(y, mask[..., None])
        mu, factor = self(batch) if coordinates is None else coordinates
        residual = self.encode(target, mu, factor).flatten(1)
        # Proper Gaussian NLL in the fixed normalized target chart, per event.
        nll = .5 * residual.square().sum(-1) + factor.diagonal(dim1=-2, dim2=-1).log().sum(-1) + 2 * math.log(2 * math.pi)
        if not torch.isfinite(nll).all():
            raise ValueError("Nonfinite calibration loss/targets")
        return nll, residual, factor

    def load_calibration(self, path, source_path):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("schema") != "conditional-preconditioning-v1" or artifact["spec"] != self.spec:
            raise ValueError("Calibration architecture/normalization contract differs")
        if not artifact.get("fit_complete", False):
            raise ValueError("Calibration training has not completed")
        if artifact["source_sha256"] != file_sha256(source_path):
            raise ValueError("Calibration and diffusion must use the identical raw source checkpoint")
        state = artifact["state_dict"]
        for key, value in self.state_dict().items():
            if "normalizer." in key and (key not in state or not torch.equal(value.cpu(), state[key].cpu())):
                raise ValueError(f"Calibration normalizer mismatch: {key}")
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Nonfinite calibration weights")
        self.load_state_dict(state, strict=True)
        if not bool(self.fitted):
            raise ValueError("Calibration is incomplete")
        self.requires_grad_(False)


class ConditionalTargetNormalizer:
    """Per-call adapter; no mutable condition state is installed on the model."""
    def __init__(self, base, preconditioner, batch):
        self.base = base
        self.preconditioner = preconditioner
        self.mu, self.factor = preconditioner.coordinates(batch)

    def __call__(self, x, mask=None):
        self.preconditioner.validate_slots(x, mask)
        return self.preconditioner.encode(self.base(x, mask), self.mu, self.factor)

    def denormalize_grad(self, x, mask=None, remove_padding=False):
        self.preconditioner.validate_slots(x, mask)
        normalized = self.preconditioner.decode(x, self.mu, self.factor)
        return self.base.denormalize_grad(normalized, mask, remove_padding=remove_padding)

    @torch.no_grad()
    def denormalize(self, x, mask=None, remove_padding=False):
        return self.denormalize_grad(x, mask, remove_padding)


def preconditioning_spec(network):
    cfg = dict(network.get("ConditionalPreconditioning", {}))
    if not cfg.pop("enabled", False):
        return None
    cfg.pop("calibration_path", None)
    return cfg
