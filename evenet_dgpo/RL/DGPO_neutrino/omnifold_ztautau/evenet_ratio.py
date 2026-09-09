"""EveNet classifiers for the no-smearing OmniFold variant.

The production problem is already expressed in the target neutrino space: there is
no detector response to invert.  Each round therefore fits one residual conditional
density ratio and adds its logit to the cumulative log weight.  There is deliberately
no OmniFold Step-2 projection classifier in this module.

The adaptive path owns one reusable classifier architecture and creates separate
cross-fit fold instances for every residual. Selected iterations can initialize
from the previous round's weights using persistent event-based folds. It trains PET's registered internal
adapters and may also fine-tune the complete active EveNet body, saving held-out
improvements over the exact null classifier, and propagating only out-of-fold
logits into later training weights. A no-op/invalid classifier is not part of the reward. Runtime
staleness is decided by a fresh temporary audit classifier on the fully reweighted
population; it is independent from the reward classifier and discarded afterward.
"""
from __future__ import annotations

import hashlib
import logging
import math
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from evenet.network.body.embedding import PointCloudPositionalEmbedding


_log = logging.getLogger(__name__)

_EVENT_KEYS = ("x", "x_mask", "conditions", "conditions_mask")
_PAIRWISE_CONTEXT_KEYS = (
    "lead_a_visible_px",
    "lead_a_visible_py",
    "lead_a_visible_pz",
    "lead_b_visible_px",
    "lead_b_visible_py",
    "lead_b_visible_pz",
)


@dataclass(frozen=True)
class EventPackingSpec:
    """Lossless fixed-shape packing contract for EveNet event inputs."""

    shapes: dict[str, tuple[int, ...]]

    @property
    def width(self) -> int:
        return sum(int(np.prod(shape)) for shape in self.shapes.values())

    def to_dict(self) -> dict[str, Any]:
        return {"shapes": {key: list(shape) for key, shape in self.shapes.items()}}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventPackingSpec":
        return cls(
            shapes={key: tuple(int(v) for v in shape) for key, shape in payload["shapes"].items()}
        )


def pack_event_inputs(
    batch: Mapping[str, Any],
    spec: EventPackingSpec | None = None,
    *,
    include_pairwise_context: bool = False,
) -> tuple[Tensor, EventPackingSpec]:
    """Pack deterministic EveNet inputs into one ``(B, D)`` tensor.

    Legacy reward checkpoints contain only ``_EVENT_KEYS``.  New topology-aware
    classifiers additionally retain the two visible tau-leg four-vector
    directions required to construct exact periodic pair features.  Passing an
    existing ``spec`` always reproduces that checkpoint's original byte layout.
    """

    keys = (
        tuple(spec.shapes)
        if spec is not None
        else _EVENT_KEYS
        + (_PAIRWISE_CONTEXT_KEYS if include_pairwise_context else ())
    )
    missing = [key for key in keys if not isinstance(batch.get(key), Tensor)]
    if missing:
        raise KeyError(f"EveNet ratio classifier needs tensor inputs {missing}")
    tensors = {key: batch[key] for key in keys}
    batch_size = int(tensors["x"].shape[0])
    if any(int(value.shape[0]) != batch_size for value in tensors.values()):
        raise ValueError("all packed EveNet inputs must share their batch dimension")
    observed = EventPackingSpec(
        {
            key: tuple(int(v) for v in value.shape[1:])
            for key, value in tensors.items()
        }
    )
    if spec is not None and observed != spec:
        raise ValueError(f"event input shapes changed: expected {spec.shapes}, got {observed.shapes}")
    packed = torch.cat(
        [
            tensors[key].reshape(batch_size, -1).to(dtype=torch.float32)
            for key in keys
        ],
        dim=-1,
    )
    return packed, observed


def unpack_event_inputs(packed: Tensor, spec: EventPackingSpec) -> dict[str, Tensor]:
    """Inverse of :func:`pack_event_inputs`; masks are restored as booleans."""

    if packed.ndim != 2 or int(packed.shape[-1]) != spec.width:
        raise ValueError(f"packed events must be (B, {spec.width}), got {tuple(packed.shape)}")
    result: dict[str, Tensor] = {}
    offset = 0
    for key in spec.shapes:
        shape = spec.shapes[key]
        width = int(np.prod(shape))
        value = packed[:, offset : offset + width].reshape(len(packed), *shape)
        result[key] = value > 0.5 if key.endswith("mask") else value
        offset += width
    return result


def periodic_tau_pair_features(
    packed_event: Tensor,
    candidate_flat: Tensor,
    packing_spec: EventPackingSpec,
    *,
    max_harmonic: int = 1,
    include_theta_pair: bool = False,
) -> Tensor:
    """Exact smooth pair features derived from visible legs and candidate deltas.

    The returned channels are ``sin(n*delta_phi_tau)`` and
    ``cos(n*delta_phi_tau)`` for ``n=1..max_harmonic``, followed by the 3D
    tau-direction cosine. Optional normalized theta difference/sum channels
    complete the pair geometry. They are
    deterministic transformations of inputs already defining the conditional
    density ratio, so exposing them changes finite-capacity efficiency rather
    than the optimal OmniFold ratio.
    """

    if candidate_flat.ndim not in (2, 3) or int(candidate_flat.shape[-1]) != 4:
        raise ValueError(
            "periodic pair features require candidate shape (B,4) or (B,K,4), "
            f"got {tuple(candidate_flat.shape)}"
        )
    if int(max_harmonic) < 1:
        raise ValueError("max_harmonic must be at least one")
    batch = unpack_event_inputs(packed_event, packing_spec)
    missing = [key for key in _PAIRWISE_CONTEXT_KEYS if key not in batch]
    if missing:
        raise ValueError(
            "periodic pair features require packed visible tau-leg directions; "
            f"missing {missing}"
        )

    def _visible_angles(prefix: str) -> tuple[Tensor, Tensor]:
        px = batch[f"{prefix}_px"].reshape(len(packed_event))
        py = batch[f"{prefix}_py"].reshape(len(packed_event))
        pz = batch[f"{prefix}_pz"].reshape(len(packed_event))
        pt = torch.sqrt(px.square() + py.square() + 1.0e-12)
        return torch.atan2(pt, pz), torch.atan2(py, px)

    theta_vis_a, phi_vis_a = _visible_angles("lead_a_visible")
    theta_vis_b, phi_vis_b = _visible_angles("lead_b_visible")
    candidate = candidate_flat.reshape(*candidate_flat.shape[:-1], 2, 2)
    if candidate_flat.ndim == 3:
        theta_vis_a = theta_vis_a[:, None]
        phi_vis_a = phi_vis_a[:, None]
        theta_vis_b = theta_vis_b[:, None]
        phi_vis_b = phi_vis_b[:, None]
    theta_a = theta_vis_a + candidate[..., 0, 0]
    theta_b = theta_vis_b + candidate[..., 1, 0]
    phi_a = phi_vis_a + candidate[..., 0, 1]
    phi_b = phi_vis_b + candidate[..., 1, 1]
    delta_phi = phi_a - phi_b
    harmonics: list[Tensor] = []
    for frequency in range(1, int(max_harmonic) + 1):
        harmonics.extend(
            (
                torch.sin(float(frequency) * delta_phi),
                torch.cos(float(frequency) * delta_phi),
            )
        )
    cos_delta_phi = harmonics[1]
    cos_opening = (
        torch.sin(theta_a) * torch.sin(theta_b) * cos_delta_phi
        + torch.cos(theta_a) * torch.cos(theta_b)
    )
    features = [*harmonics, cos_opening]
    if include_theta_pair:
        features.extend(
            (
                (theta_a - theta_b) / math.pi,
                (theta_a + theta_b) / math.pi - 1.0,
            )
        )
    return torch.stack(features, dim=-1)


def periodic_tau_pair_feature_dim(
    max_harmonic: int = 1,
    *,
    include_theta_pair: bool = False,
) -> int:
    if int(max_harmonic) < 1:
        raise ValueError("max_harmonic must be at least one")
    return 2 * int(max_harmonic) + 1 + (2 if include_theta_pair else 0)


PEFT_SCHEMA_VERSION = 6
_BACKBONE_STATE_PREFIX = "_backbone."
_BANK_REQUIRED_PREFIXES = (
    "position_encoder.",
    "decoder.",
    "output.",
)


class _SharedModuleRef:
    """Hold an ``nn.Module`` without registering it as a child."""

    __slots__ = ("module",)

    def __init__(self, module: nn.Module) -> None:
        self.module = module


def freeze_shared_backbone(model: nn.Module) -> None:
    """Freeze the shared pretrained EveNet body. Trainable PEFT lives in the bank."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _is_norm_module(module: nn.Module) -> bool:
    name = module.__class__.__name__
    return name in {"LayerNorm", "RMSNorm", "T5LayerNorm"} or isinstance(
        module, nn.LayerNorm
    )


def configure_adapter_training(
    model: nn.Module,
    *,
    train_layernorm: bool = False,
    train_encoder: bool = False,
    train_grouped_sequential_embedding: bool = False,
    train_invisible_projector: bool = False,
    train_backbone: bool = False,
) -> None:
    """Freeze the shared body, then optionally reopen selected tensors.

    PET's registered internal adapters are always trainable. ``train_backbone``
    additionally opens every module on this classifier's forward path,
    including PET attention.
    ``train_encoder`` opens GlobalEmbedding (AdaLN event token).
    ``train_grouped_sequential_embedding`` opens the visible-object input
    embedding without unfreezing PET attention or the rest of the backbone.
    ``train_invisible_projector`` opens only the projector that maps normalized
    neutrino features into the PET input basis. ObjectEncoder is unused on this
    path. ``train_layernorm`` opens every affine norm in the body.
    """

    pet = getattr(model, "PET", None)
    if pet is None:
        raise ValueError("EveNet ratio classifier requires a PET body")
    freeze_shared_backbone(model)
    adapters = getattr(pet, "adapters", None)
    if adapters is None or len(adapters) == 0:
        raise ValueError("EveNet ratio classifier requires internal PET adapters")
    for parameter in adapters.parameters():
        parameter.requires_grad_(True)
    if train_backbone:
        # ObjectEncoder and the task heads are not called by the ratio forward.
        # Do not checkpoint/optimize unreachable parameters under the label
        # "full fine-tune"; open every module that actually produces logits.
        for name in (
            "GroupedSequentialEmbedding",
            "GlobalEmbedding",
            "InvisibleInputProjector",
            "PET",
        ):
            module = getattr(model, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        model.eval()
        return
    if train_grouped_sequential_embedding:
        grouped_embedding = getattr(model, "GroupedSequentialEmbedding", None)
        if grouped_embedding is None:
            raise ValueError(
                "train_grouped_sequential_embedding=true requires "
                "backbone.GroupedSequentialEmbedding"
            )
        for parameter in grouped_embedding.parameters():
            parameter.requires_grad_(True)
    if train_layernorm:
        for module in model.modules():
            if _is_norm_module(module):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(True)
    if train_invisible_projector:
        projector = getattr(model, "InvisibleInputProjector", None)
        if projector is None:
            raise ValueError(
                "train_invisible_projector=true requires "
                "backbone.InvisibleInputProjector"
            )
        for parameter in projector.parameters():
            parameter.requires_grad_(True)
    if train_encoder:
        encoder = getattr(model, "GlobalEmbedding", None)
        if encoder is not None:
            for parameter in encoder.parameters():
                parameter.requires_grad_(True)
    model.eval()


def _module_digest(module: nn.Module) -> str:
    """Hash the pretrained body, not randomly initialized PET adapters."""

    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        if "adapter" in key.lower():
            continue
        digest.update(key.encode())
        if isinstance(value, Tensor):
            digest.update(tuple(value.shape).__repr__().encode())
            digest.update(str(value.dtype).encode())
            digest.update(
                value.detach().cpu().contiguous().reshape(-1)[:64].to(torch.float32).numpy().tobytes()
            )
    return digest.hexdigest()


def _require_pet_adapter_flag(pet: nn.Module) -> None:
    if not bool(getattr(pet, "use_adapter", False)):
        raise ValueError("EveNet ratio classifier requires Body.PET.use_adapter=true")


class AdaLNZeroModulation(nn.Module):
    """Per-branch scale, shift, and residual gate from the event token."""

    def __init__(self, event_dim: int, hidden_dim: int, *, n_branches: int = 3) -> None:
        super().__init__()
        self.n_branches = int(n_branches)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.Linear(int(event_dim), self.n_branches * 3 * self.hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, event_token: Tensor) -> tuple[Tensor, ...]:
        return self.proj(event_token).chunk(self.n_branches * 3, dim=-1)


class AdaLNZeroDecoderBlock(nn.Module):
    """Candidate residual block: self-attn, event-memory cross-attn, FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        event_dim: int,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, int(num_heads), float(dropout), batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, int(num_heads), float(dropout), batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(approximate="none"),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.modulation = AdaLNZeroModulation(event_dim, hidden_dim, n_branches=3)

    @staticmethod
    def _adaln(normed: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        return normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        memory_padding_mask: Tensor,
        event_token: Tensor,
    ) -> Tensor:
        scale_self, shift_self, gate_self, scale_cross, shift_cross, gate_cross, scale_ffn, shift_ffn, gate_ffn = (
            self.modulation(event_token)
        )
        query = self._adaln(self.norm_self(hidden), scale_self, shift_self)
        self_update, _ = self.self_attn(query, query, query, need_weights=False)
        hidden = hidden + gate_self.unsqueeze(1) * self_update
        query = self._adaln(self.norm_cross(hidden), scale_cross, shift_cross)
        cross_update, _ = self.cross_attn(
            query,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        hidden = hidden + gate_cross.unsqueeze(1) * cross_update
        hidden = hidden + gate_ffn.unsqueeze(1) * self.ffn(
            self._adaln(self.norm_ffn(hidden), scale_ffn, shift_ffn)
        )
        return hidden


class AdaLNZeroCandidateDecoder(nn.Module):
    """Read the ratio off the two neutrino tokens, conditioned on PET visibles.

    The candidate tokens stay on the residual stream.  Visible PET tokens are
    read-only cross-attention memory.  The GlobalEmbedding event token only
    drives per-layer AdaLN-Zero scale/shift/gates.  ObjectEncoder is not used.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        event_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("candidate decoder needs at least one layer")
        self.hidden_dim = int(hidden_dim)
        self.num_slots = 2
        self.candidate_in = nn.Linear(int(token_dim), self.hidden_dim)
        self.memory_in = nn.Linear(int(token_dim), self.hidden_dim)
        self.blocks = nn.ModuleList(
            AdaLNZeroDecoderBlock(
                self.hidden_dim, int(num_heads), float(dropout), int(event_dim)
            )
            for _ in range(int(num_layers))
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        *,
        candidate_tokens: Tensor,
        event_token: Tensor,
        memory_tokens: Tensor | None = None,
        memory_mask: Tensor | None = None,
        context_tokens: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        if memory_tokens is None:
            memory_tokens = context_tokens
            memory_mask = context_mask if memory_mask is None else memory_mask
        if memory_tokens is None or memory_mask is None:
            raise ValueError("candidate decoder needs event memory tokens and mask")
        hidden = self.candidate_in(candidate_tokens)
        memory = self.memory_in(memory_tokens)
        padding_mask = ~memory_mask.squeeze(-1).bool()
        for block in self.blocks:
            hidden = block(hidden, memory, padding_mask, event_token)
        if int(hidden.shape[1]) != self.num_slots:
            raise ValueError(
                f"candidate decoder expects {self.num_slots} neutrino slots, "
                f"got {tuple(hidden.shape)}"
            )
        return self.output_norm(hidden)


class CandidateConditionedRatioHead(AdaLNZeroCandidateDecoder):
    """Standalone decoder plus scalar readout, used by decoder-only unit tests."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.output = nn.Linear(self.num_slots * self.hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, **kwargs: Tensor) -> Tensor:
        hidden = super().forward(**kwargs)
        return self.output(hidden.reshape(hidden.shape[0], -1)).squeeze(-1)


class PeriodicPairAuditClassifier(nn.Module):
    """Small independent judge for the two-tau angular relationship only."""

    def __init__(
        self,
        packing_spec: EventPackingSpec,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        max_harmonic: int = 1,
        include_theta_pair: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("topology audit hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("topology audit dropout must lie in [0, 1)")
        missing = [
            key for key in _PAIRWISE_CONTEXT_KEYS if key not in packing_spec.shapes
        ]
        if missing:
            raise ValueError(
                "topology audit requires visible tau-leg context; "
                f"missing {missing}"
            )
        self.packing_spec = packing_spec
        self.max_harmonic = int(max_harmonic)
        self.include_theta_pair = bool(include_theta_pair)
        feature_dim = periodic_tau_pair_feature_dim(
            self.max_harmonic,
            include_theta_pair=self.include_theta_pair,
        )
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, packed_event: Tensor, candidate_flat: Tensor) -> Tensor:
        output_shape = candidate_flat.shape[:-1]
        features = periodic_tau_pair_features(
            packed_event,
            candidate_flat,
            self.packing_spec,
            max_harmonic=self.max_harmonic,
            include_theta_pair=self.include_theta_pair,
        )
        logits = self.network(features.reshape(-1, features.shape[-1])).squeeze(-1)
        return logits.reshape(output_shape)


class EvenetRatioPEFTBank(nn.Module):
    """Slot identity, decoder, and scalar readout for an internal-adapter PET."""

    def __init__(
        self,
        *,
        token_dim: int,
        event_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_position_length: int,
        dropout: float,
        pairwise_feature_dim: int = 0,
        topology_fourier_embedding: bool = False,
        topology_conditioning: bool = False,
        topology_hidden_dim: int = 64,
        topology_embedding_dim: int = 32,
        topology_fusion_hidden_dim: int = 64,
        topology_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.position_encoder = PointCloudPositionalEmbedding(
            num_points=int(max_position_length),
            embed_dim=int(token_dim),
        )
        self.topology_conditioning = bool(topology_conditioning)
        if self.topology_conditioning and not topology_fourier_embedding:
            raise ValueError("topology_conditioning requires topology_fourier_embedding")
        self.decoder = AdaLNZeroCandidateDecoder(
            token_dim=int(token_dim),
            event_dim=int(event_dim) + (int(topology_embedding_dim) if self.topology_conditioning else 0),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.pairwise_feature_dim = int(pairwise_feature_dim)
        self.topology_fourier_embedding = bool(topology_fourier_embedding)
        decoder_width = self.decoder.num_slots * int(hidden_dim)
        if self.topology_fourier_embedding:
            if self.pairwise_feature_dim < 1:
                raise ValueError("Fourier topology embedding requires pair features")
            if min(
                int(topology_hidden_dim),
                int(topology_embedding_dim),
                int(topology_fusion_hidden_dim),
            ) < 1:
                raise ValueError("Fourier topology embedding dimensions must be positive")
            if not 0.0 <= float(topology_dropout) < 1.0:
                raise ValueError("topology_dropout must lie in [0, 1)")
            self.topology_encoder = nn.Sequential(
                nn.LayerNorm(self.pairwise_feature_dim),
                nn.Linear(self.pairwise_feature_dim, int(topology_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(topology_dropout)),
                nn.Linear(int(topology_hidden_dim), int(topology_embedding_dim)),
                nn.GELU(),
            )
            if self.topology_conditioning:
                # Fourier context controls every decoder block through AdaLN;
                # keep the ordinary two-token readout, with no late-fusion path.
                self.output = nn.Linear(decoder_width, 1)
            else:
                self.fusion = nn.Sequential(
                    nn.LayerNorm(decoder_width + int(topology_embedding_dim)),
                    nn.Linear(
                        decoder_width + int(topology_embedding_dim),
                        int(topology_fusion_hidden_dim),
                    ),
                    nn.GELU(),
                    nn.Dropout(float(topology_dropout)),
                )
                self.output = nn.Linear(int(topology_fusion_hidden_dim), 1)
        else:
            self.output = nn.Linear(
                decoder_width + self.pairwise_feature_dim,
                1,
            )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        *,
        dropout: float,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        periodic_pair_features: bool = False,
        topology_fourier_embedding: bool = False,
        topology_conditioning: bool = False,
        topology_max_harmonic: int = 1,
        topology_include_theta_pair: bool = False,
        topology_hidden_dim: int = 64,
        topology_embedding_dim: int = 32,
        topology_fusion_hidden_dim: int = 64,
        topology_dropout: float = 0.15,
        position_state: Mapping[str, Tensor] | None = None,
    ) -> "EvenetRatioPEFTBank":
        pet_cfg = backbone.network_cfg.Body.PET
        global_cfg = getattr(backbone.network_cfg.Body, "GlobalEmbedding", None)
        truth_cfg = getattr(backbone.network_cfg, "TruthGeneration", None)
        pet = backbone.PET
        _require_pet_adapter_flag(pet)
        token_dim = int(getattr(pet, "projection_dim", None) or getattr(pet_cfg, "hidden_dim", hidden_dim))
        event_dim = int(getattr(global_cfg, "hidden_dim", token_dim))
        bank = cls(
            token_dim=token_dim,
            event_dim=event_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            max_position_length=int(getattr(truth_cfg, "max_position_length", 36)),
            dropout=float(dropout),
            pairwise_feature_dim=(
                periodic_tau_pair_feature_dim(
                    topology_max_harmonic,
                    include_theta_pair=topology_include_theta_pair,
                )
                if periodic_pair_features
                else 0
            ),
            topology_fourier_embedding=topology_fourier_embedding,
            topology_conditioning=topology_conditioning,
            topology_hidden_dim=topology_hidden_dim,
            topology_embedding_dim=topology_embedding_dim,
            topology_fusion_hidden_dim=topology_fusion_hidden_dim,
            topology_dropout=topology_dropout,
        )
        if position_state is not None:
            try:
                bank.position_encoder.load_state_dict(position_state)
            except RuntimeError:
                _log.info("[DGPO/omnifold] pretrained slot position weights were shape-incompatible")
        return bank

    def assert_complete(self) -> None:
        keys = set(self.state_dict())
        missing = [
            prefix
            for prefix in _BANK_REQUIRED_PREFIXES
            if not any(key.startswith(prefix) for key in keys)
        ]
        if missing:
            raise ValueError(f"EveNet PEFT bank is missing required groups: {missing}")
        if self.topology_fourier_embedding:
            for prefix in (("topology_encoder.",) if self.topology_conditioning else ("topology_encoder.", "fusion.")):
                if not any(key.startswith(prefix) for key in keys):
                    raise ValueError(
                        f"Fourier topology PEFT bank is missing {prefix}"
                    )

    def score(self, decoder_readout: Tensor, pair_features: Tensor | None) -> Tensor:
        if self.topology_conditioning:
            return self.output(decoder_readout).squeeze(-1)
        if self.topology_fourier_embedding:
            if pair_features is None:
                raise ValueError("Fourier topology bank requires pair features")
            topology = self.topology_encoder(pair_features)
            fused = self.fusion(torch.cat((decoder_readout, topology), dim=-1))
            return self.output(fused).squeeze(-1)
        if pair_features is not None:
            decoder_readout = torch.cat((decoder_readout, pair_features), dim=-1)
        return self.output(decoder_readout).squeeze(-1)


class EvenetAdapterRatioClassifier(nn.Module):
    """Binary truth-vs-policy classifier for two Ztautau angular-delta slots."""

    input_kind = "packed_evenet_event_physical_invisible"

    def __init__(
        self,
        backbone: nn.Module,
        packing_spec: EventPackingSpec,
        *,
        bank: EvenetRatioPEFTBank | None = None,
        train_layernorm: bool = False,
        train_encoder: bool = False,
        train_grouped_sequential_embedding: bool = False,
        train_invisible_projector: bool = False,
        train_backbone: bool = False,
        asymmetric_attention: bool = False,
        periodic_pair_features: bool = False,
        topology_fourier_embedding: bool = False,
        topology_conditioning: bool = False,
        topology_max_harmonic: int = 1,
        topology_include_theta_pair: bool = False,
        topology_hidden_dim: int = 64,
        topology_embedding_dim: int = 32,
        topology_fusion_hidden_dim: int = 64,
        topology_dropout: float = 0.15,
        head_dropout: float | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_layers: int | None = None,
        decoder_heads: int | None = None,
        adapter_bottleneck: int | None = None,
        position_state: Mapping[str, Tensor] | None = None,
        base_digest: str | None = None,
        bank_name: str | None = None,
    ) -> None:
        super().__init__()
        required_backbone_methods = (
            "project_sequential_inputs",
            "project_invisible_inputs",
        )
        missing_methods = [
            name for name in required_backbone_methods if not hasattr(backbone, name)
        ]
        if missing_methods:
            raise ValueError(
                "Ztautau EveNet backbone is missing OmniFold input projectors: "
                f"{missing_methods}"
            )
        configure_adapter_training(
            backbone,
            train_layernorm=train_layernorm,
            train_encoder=train_encoder,
            train_grouped_sequential_embedding=(
                train_grouped_sequential_embedding
            ),
            train_invisible_projector=train_invisible_projector,
            train_backbone=train_backbone,
        )
        self._train_layernorm = bool(train_layernorm)
        self._train_encoder = bool(train_encoder)
        self._train_grouped_sequential_embedding = bool(
            train_grouped_sequential_embedding
        )
        self._include_train_grouped_sequential_embedding_in_payload = True
        self._train_invisible_projector = bool(train_invisible_projector)
        self._include_train_invisible_projector_in_payload = True
        self._train_backbone = bool(train_backbone)
        self._asymmetric_attention = bool(asymmetric_attention)
        self._include_asymmetric_attention_in_payload = True
        self._periodic_pair_features = bool(periodic_pair_features)
        self._include_periodic_pair_features_in_payload = True
        self._topology_fourier_embedding = bool(topology_fourier_embedding)
        self._topology_conditioning = bool(topology_conditioning)
        if self._topology_fourier_embedding and not self._periodic_pair_features:
            raise ValueError(
                "topology_fourier_embedding requires periodic_pair_features"
            )
        self._topology_max_harmonic = int(topology_max_harmonic)
        self._topology_include_theta_pair = bool(topology_include_theta_pair)
        self._topology_hidden_dim = int(topology_hidden_dim)
        self._topology_embedding_dim = int(topology_embedding_dim)
        self._topology_fusion_hidden_dim = int(topology_fusion_hidden_dim)
        self._topology_dropout = float(topology_dropout)
        self._backbone_state_keys = tuple(
            name
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        )
        self._shared = _SharedModuleRef(backbone)
        self.packing_spec = packing_spec
        if self._periodic_pair_features:
            missing_pair_context = [
                key for key in _PAIRWISE_CONTEXT_KEYS if key not in packing_spec.shapes
            ]
            if missing_pair_context:
                raise ValueError(
                    "periodic pair features require visible tau-leg context in the "
                    f"packing spec; missing {missing_pair_context}"
                )
        self.bank_name = bank_name
        self.base_digest = base_digest or _module_digest(backbone)
        obj_cfg = backbone.network_cfg.Body.ObjectEncoder
        cls_cfg = backbone.network_cfg.Classification
        dropout = float(cls_cfg.dropout) if head_dropout is None else float(head_dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"head_dropout must lie in [0, 1), got {dropout}")
        self.invisible_input_dim = int(getattr(backbone, "invisible_input_dim", 3))
        self.num_invisible_slots = 2
        self.candidate_width = self.num_invisible_slots * self.invisible_input_dim
        self.sequential_input_dim = int(
            getattr(backbone, "sequential_input_dim", 0)
            or self.invisible_input_dim
        )
        hidden_dim = int(
            cls_cfg.hidden_dim if decoder_hidden_dim is None else decoder_hidden_dim
        )
        num_layers = int(
            cls_cfg.num_classification_layers if decoder_layers is None else decoder_layers
        )
        num_heads = int(
            decoder_heads
            if decoder_heads is not None
            else getattr(
                cls_cfg,
                "num_attention_heads",
                getattr(obj_cfg, "num_attention_heads", 1),
            )
        )
        self._head_dropout = float(dropout)
        self._decoder_hidden_dim = int(hidden_dim)
        self._decoder_layers = int(num_layers)
        self._decoder_heads = int(num_heads)
        self._adapter_bottleneck = int(
            adapter_bottleneck
            if adapter_bottleneck is not None
            else getattr(backbone.PET, "adapter_bottleneck", 16)
        )
        self.bank = bank or EvenetRatioPEFTBank.from_backbone(
            backbone,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            periodic_pair_features=self._periodic_pair_features,
            topology_fourier_embedding=self._topology_fourier_embedding,
            topology_conditioning=self._topology_conditioning,
            topology_max_harmonic=self._topology_max_harmonic,
            topology_include_theta_pair=self._topology_include_theta_pair,
            topology_hidden_dim=self._topology_hidden_dim,
            topology_embedding_dim=self._topology_embedding_dim,
            topology_fusion_hidden_dim=self._topology_fusion_hidden_dim,
            topology_dropout=self._topology_dropout,
            position_state=position_state,
        )
        self.bank.assert_complete()
        internal_adapters = getattr(backbone.PET, "adapters", None)
        if internal_adapters is None or len(internal_adapters) == 0:
            raise RuntimeError(
                "OmniFold requires backbone.PET.adapters; external adapters "
                "are not supported by PEFT schema v6"
            )
        self.trainable_parameter_counts = {
            group: sum(
                int(parameter.numel())
                for name, parameter in self.named_parameters()
                if parameter.requires_grad and name.startswith(prefix)
            )
            for group, prefix in (
                ("head", "bank.decoder."),
                ("output", "bank.output."),
                ("topology_encoder", "bank.topology_encoder."),
                ("fusion", "bank.fusion."),
                ("position_encoder", "bank.position_encoder."),
                ("object_encoder", "backbone.ObjectEncoder."),
                (
                    "grouped_sequential_embedding",
                    "backbone.GroupedSequentialEmbedding.",
                ),
                ("global_embedding", "backbone.GlobalEmbedding."),
                ("invisible_projector", "backbone.InvisibleInputProjector."),
                ("pet", "backbone.PET."),
                ("internal_pet_adapters", "backbone.PET.adapters."),
            )
        }
        self.trainable_parameter_counts["total"] = sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        if (
            self._train_grouped_sequential_embedding
            and self.trainable_parameter_counts[
                "grouped_sequential_embedding"
            ]
            <= 0
        ):
            raise RuntimeError(
                "OmniFold requested a trainable GroupedSequentialEmbedding, "
                "but none of its parameters reached the classifier optimizer view"
            )
        if (
            self._train_invisible_projector
            and self.trainable_parameter_counts["invisible_projector"] <= 0
        ):
            raise RuntimeError(
                "OmniFold requested a trainable InvisibleInputProjector, but "
                "none of its parameters reached the classifier optimizer view"
            )
        _log.info(
            "[DGPO/omnifold] ratio classifier trainable parameters: %s "
            "(head_dropout=%.3g bank=%s)",
            self.trainable_parameter_counts,
            dropout,
            bank_name,
        )
        self._peft_state_keys = tuple(self.bank.state_dict())

    @property
    def backbone(self) -> nn.Module:
        return self._shared.module

    def train(self, mode: bool = True):
        """Keep the unregistered EveNet body in the correct stochastic mode."""

        super().train(mode)
        if self._train_backbone:
            self.backbone.train(mode)
        else:
            self.backbone.eval()
            if self._train_grouped_sequential_embedding:
                self.backbone.GroupedSequentialEmbedding.train(mode)
            # Internal adapters are trainable even when the pretrained body is
            # frozen; enable their dropout without enabling frozen PET dropout.
            self.backbone.PET.adapters.train(mode)
        return self

    def _trainable_backbone_parameters(
        self,
    ) -> list[tuple[str, Tensor]]:
        parameters = dict(self.backbone.named_parameters())
        return [(name, parameters[name]) for name in self._backbone_state_keys]

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        seen: set[int] = set()
        for name, parameter in super().named_parameters(
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
        ):
            seen.add(id(parameter))
            yield name, parameter
        if not recurse:
            return
        backbone_prefix = f"{prefix}backbone" if prefix else "backbone"
        for name, parameter in self.backbone.named_parameters(
            prefix=backbone_prefix, recurse=True, remove_duplicate=remove_duplicate
        ):
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                yield name, parameter

    def parameters(self, recurse: bool = True):
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def state_dict(self, *args: Any, **kwargs: Any):
        payload = super().state_dict(*args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        keep_vars = bool(kwargs.get("keep_vars", False))
        for name, parameter in self._trainable_backbone_parameters():
            key = f"{prefix}{_BACKBONE_STATE_PREFIX}{name}"
            payload[key] = parameter if keep_vars else parameter.detach().clone()
        return payload

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        incoming = dict(state_dict)
        body = {
            key[len(_BACKBONE_STATE_PREFIX) :]: value
            for key, value in incoming.items()
            if key.startswith(_BACKBONE_STATE_PREFIX)
        }
        rest = {
            key: value
            for key, value in incoming.items()
            if not key.startswith(_BACKBONE_STATE_PREFIX)
        }
        result = super().load_state_dict(rest, strict=strict)
        trainable = dict(self._trainable_backbone_parameters())
        missing = [name for name in trainable if name not in body]
        unexpected = [name for name in body if name not in trainable]
        for name, value in body.items():
            parameter = trainable.get(name)
            if parameter is None:
                continue
            parameter.data.copy_(
                value.to(device=parameter.device, dtype=parameter.dtype)
            )
        if strict and body and (missing or unexpected):
            raise RuntimeError(
                "backbone trainable-state mismatch: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return result

    @property
    def head(self) -> AdaLNZeroCandidateDecoder:
        return self.bank.decoder

    @property
    def position_encoder(self) -> PointCloudPositionalEmbedding:
        return self.bank.position_encoder

    @staticmethod
    def _event_token_from_global(global_embedding: Tensor) -> Tensor:
        """Squeeze GlobalEmbedding to the AdaLN ``(B, D)`` event token."""

        if global_embedding.ndim == 2:
            return global_embedding
        if global_embedding.ndim == 3:
            if int(global_embedding.shape[1]) == 1:
                return global_embedding.squeeze(1)
            return global_embedding.mean(dim=1)
        raise ValueError(
            "GlobalEmbedding must be (B, D) or (B, C, D), got "
            f"{tuple(global_embedding.shape)}"
        )

    def _physical_invisibles(self, candidate_flat: Tensor) -> Tensor:
        """Restore flattened candidates to EveNet's native invisible slots."""
        return candidate_flat.reshape(
            len(candidate_flat), self.num_invisible_slots, self.invisible_input_dim
        )

    def _normalize_invisible_physics(self, physical: Tensor) -> Tensor:
        """Normalize Ztautau angular deltas the way EveNet training does.

        If the backbone stores a padded invisible normalizer, pad first and
        then slice back to the physical width. The learned
        ``InvisibleInputProjector`` maps this normalized 2D Ztautau input into
        the visible PET feature basis later in :meth:`forward`.
        """
        pad = int(getattr(self.backbone, "invisible_padding", 0))
        inv_in = int(getattr(self.backbone, "invisible_input_dim", physical.shape[-1]))
        features = physical
        if pad > 0:
            features = F.pad(features, (0, pad), value=0.0)
        mask = torch.ones(
            *features.shape[:-1], 1, device=features.device, dtype=torch.bool
        )
        return self.backbone.invisible_normalizer(x=features, mask=mask)[..., :inv_in]

    def forward(self, packed_event: Tensor, candidate_flat: Tensor) -> Tensor:
        output_shape = candidate_flat.shape[:-1]
        if candidate_flat.shape[-1] != self.candidate_width:
            raise ValueError(
                f"OmniFold candidate must have width {self.candidate_width} "
                f"({self.num_invisible_slots} slots x {self.invisible_input_dim} features), "
                f"got {tuple(candidate_flat.shape)}"
            )
        pair_features = None
        if self._periodic_pair_features:
            pair_features = periodic_tau_pair_features(
                packed_event,
                candidate_flat,
                self.packing_spec,
                max_harmonic=self._topology_max_harmonic,
                include_theta_pair=self._topology_include_theta_pair,
            )
        if candidate_flat.ndim == 3:
            bsz, count = int(candidate_flat.shape[0]), int(candidate_flat.shape[1])
            if int(packed_event.shape[0]) != bsz:
                raise ValueError("event and candidate batch dimensions do not match")
            packed_event = packed_event[:, None, :].expand(-1, count, -1).reshape(bsz * count, -1)
            candidate_flat = candidate_flat.reshape(bsz * count, self.candidate_width)
            if pair_features is not None:
                pair_features = pair_features.reshape(bsz * count, -1)
        elif candidate_flat.ndim != 2:
            raise ValueError(
                "candidate sample must be (B,F) or (B,K,F), "
                f"got {tuple(candidate_flat.shape)}"
            )

        batch = unpack_event_inputs(packed_event, self.packing_spec)
        x = batch["x"]
        x_mask = batch["x_mask"]
        if x_mask.ndim == 2:
            x_mask = x_mask.unsqueeze(-1)
        conditions = batch["conditions"]
        if conditions.ndim == 2:
            conditions = conditions.unsqueeze(1)
        conditions_mask = batch["conditions_mask"].reshape(len(x), 1, 1)
        # A multiplication-based mask cannot remove NaN padding (NaN * 0 is
        # still NaN).  Clear only invalid slots before any learned projection;
        # non-finite values in valid slots deliberately remain visible to the
        # fit-time fail-fast diagnostics.
        x = torch.where(
            x_mask.bool(),
            x,
            torch.zeros((), device=x.device, dtype=x.dtype),
        )
        conditions = torch.where(
            conditions_mask.bool(),
            conditions,
            torch.zeros((), device=conditions.device, dtype=conditions.dtype),
        )
        expected = getattr(self.backbone.global_normalizer, "mean", None)
        if expected is not None and int(conditions.shape[-1]) != int(expected.shape[-1]):
            raise ValueError(
                f"packed conditions have width {int(conditions.shape[-1])}, "
                f"global_normalizer expects {int(expected.shape[-1])}"
            )

        visible_raw = self.backbone.sequential_normalizer(x=x, mask=x_mask)
        visible = self.backbone.project_sequential_inputs(x=visible_raw, mask=x_mask)
        global_values = self.backbone.global_normalizer(x=conditions, mask=conditions_mask)
        invisible = self._normalize_invisible_physics(
            self._physical_invisibles(candidate_flat)
        )
        invisible_mask = torch.ones(
            len(x), self.num_invisible_slots, 1, device=x.device, dtype=torch.bool
        )
        invisible_projected = self.backbone.project_invisible_inputs(
            x=invisible, mask=invisible_mask
        )
        visible_mask = x_mask.bool()
        if conditions_mask.ndim == 2:
            conditions_mask = conditions_mask.unsqueeze(-1)
        time = torch.zeros(len(x), device=x.device, dtype=x.dtype)
        n_visible = int(visible.shape[1])
        full = torch.cat((visible, invisible_projected), dim=1)
        full_mask = torch.cat((visible_mask, invisible_mask), dim=1)
        attention_mask = None
        if self._asymmetric_attention:
            # Legacy reward checkpoints were trained with visible queries
            # blocked from attending to invisible keys. Preserve that behavior
            # only while restoring those frozen stacks; new fits use full
            # bidirectional self-attention.
            is_invisible = torch.cat(
                (
                    torch.zeros(n_visible, dtype=torch.bool, device=x.device),
                    torch.ones(
                        self.num_invisible_slots,
                        dtype=torch.bool,
                        device=x.device,
                    ),
                )
            )
            attention_mask = (~is_invisible[:, None]) & is_invisible[None, :]
        time_masking = torch.cat(
            (torch.zeros_like(visible_mask), invisible_mask), dim=1
        ).float()
        global_embedding = self.backbone.GlobalEmbedding(
            x=global_values, mask=conditions_mask
        )
        encoded = self.backbone.PET(
            input_features=full,
            input_points=full[..., self.backbone.local_feature_indices],
            mask=full_mask,
            attn_mask=attention_mask,
            time=time,
            time_masking=time_masking,
            # ``None`` always selects PET's registered internal adapter stack.
            adapters=None,
        )
        memory_tokens = encoded[:, :n_visible]
        memory_mask = visible_mask
        # Multihead cross-attention is undefined for a row whose every memory
        # key is masked.  Some CUDA kernels return finite zeros in the forward
        # pass but NaN gradients in the backward pass.  Supply one neutral
        # sentinel only for truly empty visible events.  Ordinary events and
        # their physics content are unchanged.
        if n_visible < 1:
            raise ValueError("EveNet ratio classifier needs at least one visible slot")
        empty_visible = ~memory_mask.squeeze(-1).any(dim=1)
        memory_tokens = torch.cat(
            (memory_tokens, memory_tokens.new_zeros(len(x), 1, memory_tokens.shape[-1])),
            dim=1,
        )
        memory_mask = torch.cat(
            (memory_mask, empty_visible[:, None, None]),
            dim=1,
        )
        event_token = self._event_token_from_global(global_embedding)
        if self.bank.topology_conditioning:
            if pair_features is None:
                raise ValueError("Fourier decoder conditioning requires candidate pair features")
            topology = self.bank.topology_encoder(pair_features.to(event_token))
            event_token = torch.cat((event_token, topology), dim=-1)
        candidate_tokens = self.bank.position_encoder(
            x=encoded[:, n_visible:],
            time_mask=invisible_mask.to(encoded.dtype),
            x_mask=invisible_mask.to(encoded.dtype),
        )
        hidden = self.bank.decoder(
            candidate_tokens=candidate_tokens,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            event_token=event_token,
        )
        readout = hidden.reshape(hidden.shape[0], -1)
        if pair_features is not None:
            pair_features = pair_features.to(
                device=readout.device,
                dtype=readout.dtype,
            )
        logits = self.bank.score(readout, pair_features)
        return logits.reshape(output_shape)

    def peft_payload(self) -> dict[str, Any]:
        """Serialize the classifier bank and any fine-tuned EveNet body tensors."""

        self.bank.assert_complete()
        state = self.bank.state_dict()
        missing = sorted(set(self._peft_state_keys) - set(state))
        if missing:
            raise ValueError(f"EveNet ratio PEFT payload is missing keys: {missing[:5]}")
        payload: dict[str, Any] = {
            "schema_version": PEFT_SCHEMA_VERSION,
            "candidate_width": self.candidate_width,
            "num_invisible_slots": self.num_invisible_slots,
            "invisible_input_dim": self.invisible_input_dim,
            "packing_spec": self.packing_spec.to_dict(),
            "base_digest": self.base_digest,
            "bank_name": self.bank_name,
            "state": {
                key: state[key].detach().cpu().clone() for key in self._peft_state_keys
            },
            "body": {
                name: parameter.detach().cpu().clone()
                for name, parameter in self._trainable_backbone_parameters()
            },
        }
        if getattr(self, "_include_classifier_config_in_payload", True):
            classifier_config = {
                "head_dropout": self._head_dropout,
                "decoder_hidden_dim": self._decoder_hidden_dim,
                "decoder_layers": self._decoder_layers,
                "decoder_heads": self._decoder_heads,
                "adapter_bottleneck": self._adapter_bottleneck,
                "train_backbone": self._train_backbone,
            }
            if getattr(
                self,
                "_include_train_grouped_sequential_embedding_in_payload",
                True,
            ):
                classifier_config["train_grouped_sequential_embedding"] = (
                    self._train_grouped_sequential_embedding
                )
            if getattr(
                self, "_include_train_invisible_projector_in_payload", True
            ):
                classifier_config["train_invisible_projector"] = (
                    self._train_invisible_projector
                )
            if getattr(self, "_include_asymmetric_attention_in_payload", True):
                classifier_config["asymmetric_attention"] = (
                    self._asymmetric_attention
                )
            if getattr(self, "_include_periodic_pair_features_in_payload", True):
                classifier_config["periodic_pair_features"] = (
                    self._periodic_pair_features
                )
            if (
                self._topology_fourier_embedding
                or self._topology_max_harmonic != 1
                or self._topology_include_theta_pair
            ):
                classifier_config.update(
                    {
                        "topology_fourier_embedding": self._topology_fourier_embedding,
                        "topology_max_harmonic": self._topology_max_harmonic,
                        "topology_include_theta_pair": self._topology_include_theta_pair,
                        "topology_hidden_dim": self._topology_hidden_dim,
                        "topology_embedding_dim": self._topology_embedding_dim,
                        "topology_fusion_hidden_dim": self._topology_fusion_hidden_dim,
                        "topology_dropout": self._topology_dropout,
                    }
                )
            classifier_config["adapter_placement"] = "internal"
            # Omit the default to preserve legacy PEFT payload digests.
            if self._topology_conditioning:
                classifier_config["topology_conditioning"] = True
            payload["classifier_config"] = classifier_config
        return payload

    @classmethod
    def from_peft_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        model_builder: Callable[[EventPackingSpec], "EvenetAdapterRatioClassifier"],
        device: torch.device,
    ) -> "EvenetAdapterRatioClassifier":
        if int(payload.get("schema_version", -1)) != PEFT_SCHEMA_VERSION:
            raise ValueError("unsupported EveNet ratio PEFT schema")
        spec = EventPackingSpec.from_dict(payload["packing_spec"])
        model = model_builder(spec)
        # Preserve legacy payload bytes exactly so an architecture migration can
        # restore an older reward stack without changing its integrity digest.
        model._include_classifier_config_in_payload = (
            "classifier_config" in payload
        )
        model._include_train_invisible_projector_in_payload = (
            "train_invisible_projector"
            in dict(payload.get("classifier_config") or {})
        )
        model._include_train_grouped_sequential_embedding_in_payload = (
            "train_grouped_sequential_embedding"
            in dict(payload.get("classifier_config") or {})
        )
        model._include_asymmetric_attention_in_payload = (
            "asymmetric_attention"
            in dict(payload.get("classifier_config") or {})
        )
        model._include_periodic_pair_features_in_payload = (
            "periodic_pair_features"
            in dict(payload.get("classifier_config") or {})
        )
        expected_width = int(payload.get("candidate_width", model.candidate_width))
        if expected_width != model.candidate_width:
            raise ValueError(
                "EveNet ratio candidate width does not match the current Ztautau model: "
                f"{expected_width} vs {model.candidate_width}"
            )
        model.bank.to(device)
        current = model.bank.state_dict()
        delta = dict(payload["state"])
        unknown = sorted(set(delta) - set(current))
        if unknown:
            raise ValueError(f"EveNet ratio PEFT payload has unknown keys: {unknown[:5]}")
        missing = sorted(set(model._peft_state_keys) - set(delta))
        if missing:
            raise ValueError(f"EveNet ratio PEFT payload is missing keys: {missing[:5]}")
        expected_digest = payload.get("base_digest")
        body = dict(payload.get("body") or {})
        if expected_digest is not None and model.base_digest != expected_digest:
            # Full-FT payloads carry the trained body; the 20M.a4 digest is only
            # a cold-start tag. A drifted template hash must not discard a
            # resumable round-1 stack and leave the velocity anchor unpaired.
            if not body:
                raise ValueError(
                    "EveNet ratio PEFT payload backbone digest does not match the "
                    f"shared pretrained body ({str(expected_digest)[:12]} vs "
                    f"{model.base_digest[:12]})"
                )
            _log.warning(
                "[DGPO/omnifold] PEFT payload digest %s != current template %s; "
                "restoring from saved body weights",
                str(expected_digest)[:12],
                model.base_digest[:12],
            )
        for key, value in delta.items():
            if tuple(value.shape) != tuple(current[key].shape):
                raise ValueError(
                    f"EveNet ratio PEFT shape mismatch for {key}: "
                    f"{tuple(value.shape)} vs {tuple(current[key].shape)}"
                )
            current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        model.bank.load_state_dict(current, strict=True)
        trainable = dict(model._trainable_backbone_parameters())
        unknown_body = sorted(set(body) - set(trainable))
        if unknown_body:
            raise ValueError(
                f"EveNet ratio PEFT payload has unknown body keys: {unknown_body[:5]}"
            )
        for name, value in body.items():
            parameter = trainable[name]
            parameter.data.copy_(
                value.to(device=parameter.device, dtype=parameter.dtype)
            )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model


class _ConfigWithClonedNetwork:
    """Config view that overrides only the YAML network tree.

    The live DGPO ``Config`` cannot be deep-copied: ``event_info`` is a constructed
    object (not YAML), and ``Config.__getattr__`` must not forward ``__deepcopy__``.
    The ratio builder only needs to flip PET adapter flags.
    """

    def __init__(self, base: Any, network: Any) -> None:
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "network", network)

    def __getattr__(self, key: str) -> Any:
        return getattr(self._base, key)


def _config_with_pet_adapters(config: Any, adapter_bottleneck: int) -> Any:
    """Clone ``config.network`` and enable PET adapters without mutating the policy."""
    network = deepcopy(config.network)
    network.Body.PET.use_adapter = True
    network.Body.PET.adapter_bottleneck = int(adapter_bottleneck)
    return _ConfigWithClonedNetwork(config, network)


class EvenetAdapterModelBuilder:
    """Build independent full-FT classifiers or PEFT views of one frozen body."""

    def __init__(
        self,
        *,
        config: Any,
        normalization_dict: dict[str, Any],
        checkpoint_path: str | Path,
        device: torch.device,
        adapter_bottleneck: int = 16,
        train_layernorm: bool = False,
        train_encoder: bool = False,
        train_grouped_sequential_embedding: bool = False,
        train_invisible_projector: bool = False,
        train_backbone: bool = False,
        asymmetric_attention: bool = False,
        periodic_pair_features: bool = False,
        topology_fourier_embedding: bool = False,
        topology_conditioning: bool = False,
        topology_max_harmonic: int = 1,
        topology_include_theta_pair: bool = False,
        topology_hidden_dim: int = 64,
        topology_embedding_dim: int = 32,
        topology_fusion_hidden_dim: int = 64,
        topology_dropout: float = 0.15,
        head_dropout: float | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_layers: int | None = None,
        decoder_heads: int | None = None,
    ) -> None:
        from RL.DGPO_neutrino.model_utils import (
            build_evenet_on_device,
            load_weights_like_configure_model,
        )

        path = Path(str(checkpoint_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EveNet ratio pretrained checkpoint not found: {path}")
        ratio_config = _config_with_pet_adapters(config, adapter_bottleneck)
        template = build_evenet_on_device(ratio_config, normalization_dict, device)
        load_weights_like_configure_model(
            template, path, device, ratio_config, for_dgpo_training=False
        )
        position_state = None
        truth_head = getattr(template, "TruthGeneration", None)
        if truth_head is not None and hasattr(truth_head, "position_encoder"):
            position_state = {
                key: value.detach().cpu().clone()
                for key, value in truth_head.position_encoder.state_dict().items()
            }
        self._strip_task_heads(template)
        freeze_shared_backbone(template)
        self._pretrained_body = {
            key: value.detach().cpu().clone()
            for key, value in template.state_dict().items()
        }
        self._base_digest = _module_digest(template)
        self._backbone = template.to(device)
        self._config = ratio_config
        self._normalization_dict = normalization_dict
        self._device = device
        self._train_layernorm = bool(train_layernorm)
        self._train_encoder = bool(train_encoder)
        self._train_grouped_sequential_embedding = bool(
            train_grouped_sequential_embedding
        )
        self._train_invisible_projector = bool(train_invisible_projector)
        self._train_backbone = bool(train_backbone)
        self._asymmetric_attention = bool(asymmetric_attention)
        self._periodic_pair_features = bool(periodic_pair_features)
        self._topology_fourier_embedding = bool(topology_fourier_embedding)
        self._topology_conditioning = bool(topology_conditioning)
        self._topology_max_harmonic = int(topology_max_harmonic)
        self._topology_include_theta_pair = bool(topology_include_theta_pair)
        self._topology_hidden_dim = int(topology_hidden_dim)
        self._topology_embedding_dim = int(topology_embedding_dim)
        self._topology_fusion_hidden_dim = int(topology_fusion_hidden_dim)
        self._topology_dropout = float(topology_dropout)
        self._head_dropout = None if head_dropout is None else float(head_dropout)
        self._adapter_bottleneck = int(adapter_bottleneck)
        self._decoder_hidden_dim = decoder_hidden_dim
        self._decoder_layers = decoder_layers
        self._decoder_heads = decoder_heads
        self._position_state = position_state
        self._banks: dict[str, EvenetRatioPEFTBank] = {}
        self._classifiers: dict[str, EvenetAdapterRatioClassifier] = {}

    @staticmethod
    def _strip_task_heads(backbone: nn.Module) -> None:
        """The ratio uses EveNet's body only; remove unrelated pretrained heads."""
        for name in (
            "Classification",
            "Regression",
            "Assignment",
            "GlobalGeneration",
            "ReconGeneration",
            "TruthGeneration",
            "Segmentation",
        ):
            if hasattr(backbone, name):
                delattr(backbone, name)

    @property
    def backbone(self) -> nn.Module:
        return self._backbone

    @property
    def base_digest(self) -> str:
        return self._base_digest

    def _fresh_backbone(self) -> nn.Module:
        """Return an isolated pretrained body for a full/partial fine-tune."""

        # EveNet models contain runtime metadata such as ``dict_keys`` views
        # that cannot be pickled by ``copy.deepcopy``.  Rebuild the module from
        # the same config and restore the immutable pretrained body snapshot,
        # matching the rebuild + state_dict pattern used for DGPO references.
        from RL.DGPO_neutrino.model_utils import build_evenet_on_device

        backbone = build_evenet_on_device(
            self._config,
            self._normalization_dict,
            self._device,
        )
        self._strip_task_heads(backbone)
        backbone.load_state_dict(self._pretrained_body, strict=True)
        freeze_shared_backbone(backbone)
        return backbone

    def make_bank(
        self,
        backbone: nn.Module | None = None,
        *,
        head_dropout: float | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_layers: int | None = None,
        decoder_heads: int | None = None,
        periodic_pair_features: bool | None = None,
        topology_fourier_embedding: bool | None = None,
        topology_conditioning: bool | None = None,
        topology_max_harmonic: int | None = None,
        topology_include_theta_pair: bool | None = None,
        topology_hidden_dim: int | None = None,
        topology_embedding_dim: int | None = None,
        topology_fusion_hidden_dim: int | None = None,
        topology_dropout: float | None = None,
    ) -> EvenetRatioPEFTBank:
        source = self._backbone if backbone is None else backbone
        resolved_dropout = (
            self._head_dropout if head_dropout is None else float(head_dropout)
        )
        resolved_hidden = (
            self._decoder_hidden_dim
            if decoder_hidden_dim is None
            else int(decoder_hidden_dim)
        )
        resolved_layers = (
            self._decoder_layers if decoder_layers is None else int(decoder_layers)
        )
        resolved_heads = (
            self._decoder_heads if decoder_heads is None else int(decoder_heads)
        )
        resolved_periodic_pair_features = (
            self._periodic_pair_features
            if periodic_pair_features is None
            else bool(periodic_pair_features)
        )
        resolved_topology_fourier_embedding = (
            self._topology_fourier_embedding
            if topology_fourier_embedding is None
            else bool(topology_fourier_embedding)
        )
        resolved_topology_conditioning = (
            self._topology_conditioning if topology_conditioning is None else bool(topology_conditioning)
        )
        resolved_topology_max_harmonic = (
            self._topology_max_harmonic
            if topology_max_harmonic is None
            else int(topology_max_harmonic)
        )
        resolved_topology_include_theta_pair = (
            self._topology_include_theta_pair
            if topology_include_theta_pair is None
            else bool(topology_include_theta_pair)
        )
        resolved_topology_hidden_dim = (
            self._topology_hidden_dim
            if topology_hidden_dim is None
            else int(topology_hidden_dim)
        )
        resolved_topology_embedding_dim = (
            self._topology_embedding_dim
            if topology_embedding_dim is None
            else int(topology_embedding_dim)
        )
        resolved_topology_fusion_hidden_dim = (
            self._topology_fusion_hidden_dim
            if topology_fusion_hidden_dim is None
            else int(topology_fusion_hidden_dim)
        )
        resolved_topology_dropout = (
            self._topology_dropout
            if topology_dropout is None
            else float(topology_dropout)
        )
        return EvenetRatioPEFTBank.from_backbone(
            source,
            dropout=0.0 if resolved_dropout is None else float(resolved_dropout),
            hidden_dim=int(
                resolved_hidden
                if resolved_hidden is not None
                else source.network_cfg.Classification.hidden_dim
            ),
            num_layers=int(
                resolved_layers
                if resolved_layers is not None
                else source.network_cfg.Classification.num_classification_layers
            ),
            num_heads=int(
                resolved_heads
                if resolved_heads is not None
                else getattr(
                    source.network_cfg.Classification,
                    "num_attention_heads",
                    1,
                )
            ),
            periodic_pair_features=resolved_periodic_pair_features,
            topology_fourier_embedding=resolved_topology_fourier_embedding,
            topology_conditioning=resolved_topology_conditioning,
            topology_max_harmonic=resolved_topology_max_harmonic,
            topology_include_theta_pair=resolved_topology_include_theta_pair,
            topology_hidden_dim=resolved_topology_hidden_dim,
            topology_embedding_dim=resolved_topology_embedding_dim,
            topology_fusion_hidden_dim=resolved_topology_fusion_hidden_dim,
            topology_dropout=resolved_topology_dropout,
            position_state=self._position_state,
        ).to(self._device)

    def make_classifier(
        self,
        packing_spec: EventPackingSpec,
        name: str | None = None,
        *,
        reset: bool = True,
        bank: EvenetRatioPEFTBank | None = None,
        head_dropout: float | None = None,
        decoder_hidden_dim: int | None = None,
        decoder_layers: int | None = None,
        decoder_heads: int | None = None,
        adapter_bottleneck: int | None = None,
        train_grouped_sequential_embedding: bool | None = None,
        train_invisible_projector: bool | None = None,
        train_backbone: bool | None = None,
        asymmetric_attention: bool | None = None,
        periodic_pair_features: bool | None = None,
        topology_fourier_embedding: bool | None = None,
        topology_conditioning: bool | None = None,
        topology_max_harmonic: int | None = None,
        topology_include_theta_pair: bool | None = None,
        topology_hidden_dim: int | None = None,
        topology_embedding_dim: int | None = None,
        topology_fusion_hidden_dim: int | None = None,
        topology_dropout: float | None = None,
    ) -> EvenetAdapterRatioClassifier:
        if (
            bank is None
            and name is not None
            and not reset
            and name in self._classifiers
        ):
            return self._classifiers[name]
        resolved_train_invisible_projector = (
            self._train_invisible_projector
            if train_invisible_projector is None
            else bool(train_invisible_projector)
        )
        resolved_train_grouped_sequential_embedding = (
            self._train_grouped_sequential_embedding
            if train_grouped_sequential_embedding is None
            else bool(train_grouped_sequential_embedding)
        )
        resolved_train_backbone = (
            self._train_backbone
            if train_backbone is None
            else bool(train_backbone)
        )
        resolved_asymmetric_attention = (
            self._asymmetric_attention
            if asymmetric_attention is None
            else bool(asymmetric_attention)
        )
        resolved_periodic_pair_features = (
            self._periodic_pair_features
            if periodic_pair_features is None
            else bool(periodic_pair_features)
        )
        resolved_topology_fourier_embedding = (
            self._topology_fourier_embedding
            if topology_fourier_embedding is None
            else bool(topology_fourier_embedding)
        )
        resolved_topology_conditioning = (
            self._topology_conditioning if topology_conditioning is None else bool(topology_conditioning)
        )
        resolved_topology_max_harmonic = (
            self._topology_max_harmonic
            if topology_max_harmonic is None
            else int(topology_max_harmonic)
        )
        resolved_topology_include_theta_pair = (
            self._topology_include_theta_pair
            if topology_include_theta_pair is None
            else bool(topology_include_theta_pair)
        )
        resolved_topology_hidden_dim = (
            self._topology_hidden_dim
            if topology_hidden_dim is None
            else int(topology_hidden_dim)
        )
        resolved_topology_embedding_dim = (
            self._topology_embedding_dim
            if topology_embedding_dim is None
            else int(topology_embedding_dim)
        )
        resolved_topology_fusion_hidden_dim = (
            self._topology_fusion_hidden_dim
            if topology_fusion_hidden_dim is None
            else int(topology_fusion_hidden_dim)
        )
        resolved_topology_dropout = (
            self._topology_dropout
            if topology_dropout is None
            else float(topology_dropout)
        )
        # Internal PET adapters are always trainable, so every classifier/fold
        # must own an isolated body even when the rest of the backbone is frozen.
        backbone = self._fresh_backbone()
        if bank is None:
            bank = self.make_bank(
                backbone,
                head_dropout=head_dropout,
                decoder_hidden_dim=decoder_hidden_dim,
                decoder_layers=decoder_layers,
                decoder_heads=decoder_heads,
                periodic_pair_features=resolved_periodic_pair_features,
                topology_fourier_embedding=resolved_topology_fourier_embedding,
                topology_conditioning=resolved_topology_conditioning,
                topology_max_harmonic=resolved_topology_max_harmonic,
                topology_include_theta_pair=resolved_topology_include_theta_pair,
                topology_hidden_dim=resolved_topology_hidden_dim,
                topology_embedding_dim=resolved_topology_embedding_dim,
                topology_fusion_hidden_dim=resolved_topology_fusion_hidden_dim,
                topology_dropout=resolved_topology_dropout,
            )
            if name is not None:
                self._banks[name] = bank
        classifier = EvenetAdapterRatioClassifier(
            backbone,
            packing_spec,
            bank=bank,
            train_layernorm=self._train_layernorm,
            train_encoder=self._train_encoder,
            train_grouped_sequential_embedding=(
                resolved_train_grouped_sequential_embedding
            ),
            train_invisible_projector=resolved_train_invisible_projector,
            train_backbone=resolved_train_backbone,
            asymmetric_attention=resolved_asymmetric_attention,
            periodic_pair_features=resolved_periodic_pair_features,
            topology_fourier_embedding=resolved_topology_fourier_embedding,
            topology_conditioning=resolved_topology_conditioning,
            topology_max_harmonic=resolved_topology_max_harmonic,
            topology_include_theta_pair=resolved_topology_include_theta_pair,
            topology_hidden_dim=resolved_topology_hidden_dim,
            topology_embedding_dim=resolved_topology_embedding_dim,
            topology_fusion_hidden_dim=resolved_topology_fusion_hidden_dim,
            topology_dropout=resolved_topology_dropout,
            head_dropout=(
                self._head_dropout if head_dropout is None else head_dropout
            ),
            decoder_hidden_dim=(
                self._decoder_hidden_dim
                if decoder_hidden_dim is None
                else decoder_hidden_dim
            ),
            decoder_layers=(
                self._decoder_layers if decoder_layers is None else decoder_layers
            ),
            decoder_heads=(
                self._decoder_heads if decoder_heads is None else decoder_heads
            ),
            adapter_bottleneck=(
                self._adapter_bottleneck
                if adapter_bottleneck is None
                else adapter_bottleneck
            ),
            position_state=self._position_state,
            base_digest=self._base_digest,
            bank_name=name,
        )
        if name is not None:
            self._classifiers[name] = classifier
        return classifier

    def restore_pretrained_body(self) -> None:
        """Reload the frozen template used to construct new classifier bodies."""

        self._backbone.load_state_dict(self._pretrained_body, strict=True)
        freeze_shared_backbone(self._backbone)

    def reset_audit_bank(self, packing_spec: EventPackingSpec) -> EvenetAdapterRatioClassifier:
        return self.make_classifier(packing_spec, "audit", reset=True)

    def discard_bank(self, name: str) -> None:
        """Drop a temporary PEFT bank/classifier while retaining the shared body."""

        self._classifiers.pop(str(name), None)
        self._banks.pop(str(name), None)

    def __call__(self, packing_spec: EventPackingSpec) -> EvenetAdapterRatioClassifier:
        return self.make_classifier(packing_spec)


def peft_bank_factory(
    builder: Any,
    packing_spec: EventPackingSpec,
    name: str | None = None,
    *,
    reset: bool = True,
    classifier_overrides: Mapping[str, Any] | None = None,
) -> Callable[[], EvenetAdapterRatioClassifier]:
    """Build one classifier view, resetting the named bank when requested."""

    overrides = dict(classifier_overrides or {})
    if overrides:
        dropout_keys = {"head_dropout", "topology_dropout"}
        dimension_keys = {"decoder_hidden_dim", "decoder_layers", "decoder_heads"}
        flag_keys = {"periodic_pair_features", "topology_fourier_embedding", "topology_conditioning"}
        if set(overrides) - dropout_keys - dimension_keys - flag_keys:
            raise ValueError("Unsupported classifier architecture override")
        if any(not 0.0 <= float(overrides[key]) < 1.0 for key in dropout_keys & overrides.keys()):
            raise ValueError("Classifier dropout must be in [0, 1)")
        if any(type(overrides[key]) is not int or overrides[key] < 1 for key in dimension_keys & overrides.keys()):
            raise ValueError("Classifier dimensions must be positive integers")
        if any(type(overrides[key]) is not bool for key in flag_keys & overrides.keys()):
            raise ValueError("Classifier feature flags must be boolean")
        if not hasattr(builder, "make_classifier"):
            raise TypeError("Classifier overrides require make_classifier")
        return lambda: builder.make_classifier(packing_spec, name, reset=reset, **overrides)
    if name == "audit" and hasattr(builder, "reset_audit_bank"):
        return lambda: builder.reset_audit_bank(packing_spec)
    if hasattr(builder, "make_classifier"):
        return lambda: builder.make_classifier(packing_spec, name, reset=reset)
    return lambda: builder(packing_spec)


@dataclass(frozen=True)
class ResidualIterationDiagnostics:
    """Held-out decision for one cross-fitted residual proposal."""

    iteration: int
    fold_diagnostics: tuple[Any, ...]
    null_validation_loss: float
    validation_loss: float
    validation_balanced_accuracy: float
    validation_auc: float
    validation_loss_gain: float
    accepted: bool
    rejection_reason: str | None = None
    warm_started_folds: tuple[int, ...] = ()

    @property
    def saturated(self) -> bool:
        return bool(self.fold_diagnostics) and all(
            bool(getattr(item, "saturated", False))
            for item in self.fold_diagnostics
        )

    @property
    def final_loss(self) -> float:
        values = [
            float(getattr(item, "loss"))
            for item in self.fold_diagnostics
            if getattr(item, "loss", None) is not None
        ]
        return float(np.mean(values)) if values else float("nan")

    @property
    def final_accuracy(self) -> float:
        values = [
            float(getattr(item, "balanced_accuracy"))
            for item in self.fold_diagnostics
            if getattr(item, "balanced_accuracy", None) is not None
        ]
        return float(np.mean(values)) if values else float("nan")


@dataclass(frozen=True)
class ResidualRatioResult:
    classifier: nn.Module
    checkpoints: tuple[dict[str, Tensor], ...]
    checkpoint_coefficients: tuple[float, ...]
    checkpoint_iterations: tuple[int, ...]
    diagnostics: tuple[Any, ...]
    train_log_weight: Tensor
    validation_log_weight: Tensor | None
    warm_start_state: dict[str, Any] | None = None

    @property
    def iterations(self) -> int:
        return max(self.checkpoint_iterations, default=0)


@dataclass(frozen=True)
class EvenetAuditResult:
    """Final evaluation metrics from a fresh temporary EveNet judge."""

    auc: float
    auc_gap: float
    balanced_accuracy: float
    truth_positive_rate: float
    gen_negative_rate: float
    fit_diagnostics: Any
    fit_events: int
    early_stop_events: int
    audit_events: int
    warm_started: bool = False
    training_min_steps: int = 0
    training_steps_per_epoch: int = 0
    training_ready: bool = False


@torch.no_grad()
def _score_in_batches(
    model: nn.Module, condition: Tensor, sample: Tensor, batch_size: int
) -> Tensor:
    # ``batch_size`` is a row budget, matching evaluate_density_ratio where the
    # candidate axis is already flat. A candidate axis multiplies rows per event,
    # and PET's kNN local embedding expands one row to O(10^4) floats, so slicing
    # on events alone asks for tens of GiB in a single activation at K=32.
    rows_per_event = 1
    for dim in sample.shape[1:-1]:
        rows_per_event *= max(1, int(dim))
    event_batch = max(1, int(batch_size) // rows_per_event)
    pieces = []
    for start in range(0, len(condition), event_batch):
        pieces.append(
            model(
                condition[start : start + event_batch],
                sample[start : start + event_batch],
            )
        )
    score = torch.cat(pieces, dim=0)
    if not bool(torch.isfinite(score).all().item()):
        raise FloatingPointError("EveNet ratio classifier produced NaN/Inf logits")
    return score


def _leading_dim_shard(n: int, rank: int, world: int) -> tuple[int, int]:
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid distributed shard arguments")
    return (int(n) * rank) // world, (int(n) * (rank + 1)) // world


def _all_gather_leading_dim(local: Tensor, *, global_n: int) -> Tensor:
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import distributed_context

    rank, world = distributed_context()
    expected_n = int(global_n)
    if world <= 1:
        if int(local.shape[0]) != expected_n:
            raise RuntimeError(
                f"single-process score length {int(local.shape[0])} != {expected_n}"
            )
        return local
    sizes = [
        (expected_n * (other + 1)) // world - (expected_n * other) // world
        for other in range(world)
    ]
    if int(local.shape[0]) != int(sizes[rank]):
        raise RuntimeError(
            f"rank {rank} scored {int(local.shape[0])} rows, expected {sizes[rank]}"
        )
    max_n = max(sizes) if sizes else 0
    if max_n == 0:
        return local
    padded = local.new_zeros((max_n, *local.shape[1:]))
    if local.shape[0] > 0:
        padded[: local.shape[0]] = local
    gathered = [local.new_empty(padded.shape) for _ in range(world)]
    torch.distributed.all_gather(gathered, padded.contiguous())
    return torch.cat(
        [piece[: sizes[other]] for other, piece in enumerate(gathered)], dim=0
    )


def _score_population(
    model: nn.Module, condition: Tensor, sample: Tensor, batch_size: int
) -> Tensor:
    """Score a population, sharding the event axis when a process group is live."""
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import distributed_context

    rank, world = distributed_context()
    n = int(condition.shape[0])
    start, stop = _leading_dim_shard(n, rank, world)
    if start < stop:
        local = _score_in_batches(
            model, condition[start:stop], sample[start:stop], batch_size
        )
    else:
        local = sample.new_zeros((0, *sample.shape[1:-1]))
    return _all_gather_leading_dim(local, global_n=n)


def _seeded_model(
    model_factory: Callable[[], nn.Module], *, seed: int, device: torch.device
) -> nn.Module:
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        return model_factory()


def _scaled_crossfit_config(
    fit_config: Any, train_fraction: float, *, min_steps_per_fold: int = 0,
) -> Any:
    """Preserve epoch-based controls when a fold trains on fewer events."""

    if type(min_steps_per_fold) is not int or min_steps_per_fold < 0:
        raise ValueError("min_steps_per_fold must be a nonnegative integer")
    if not is_dataclass(fit_config):
        if min_steps_per_fold:
            raise TypeError("min_steps_per_fold requires a dataclass fit configuration")
        return fit_config
    fraction = float(train_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("cross-fit training fraction must lie in (0, 1]")

    def scaled(name: str, *, allow_zero: bool = False) -> int | None:
        raw_value = getattr(fit_config, name)
        if raw_value is None:
            return None
        value = int(raw_value)
        if allow_zero and value == 0:
            return 0
        return max(1, int(math.ceil(value * fraction)))

    steps = scaled("steps")
    scaled_min_steps = scaled("min_steps", allow_zero=True)
    assert scaled_min_steps is not None
    updates = {
        "steps": steps,
        "min_steps": (
            scaled_min_steps
            if steps is None
            else min(steps, scaled_min_steps)
        ),
        "validation_interval_steps": scaled("validation_interval_steps"),
        "progress_interval_steps": scaled(
            "progress_interval_steps", allow_zero=True
        ),
        "checkpoint_interval_steps": scaled(
            "checkpoint_interval_steps", allow_zero=True
        ),
    }
    # This is an absolute optimizer-update floor AFTER fold-size scaling.
    # A global min_steps=1000 would otherwise become roughly 500 per fold.
    if steps is not None and steps < min_steps_per_fold:
        raise ValueError("classifier step budget is smaller than min_steps_per_fold")
    updates["min_steps"] = max(updates["min_steps"], min_steps_per_fold)
    return replace(fit_config, **updates)


def _crossfit_splits(
    n_events: int, *, folds: int, seed: int, device: torch.device
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Return deterministic (fit, OOF-score) event-index pairs."""

    if int(folds) < 2:
        raise ValueError("residual cross-fitting needs at least two folds")
    if int(n_events) < 2 * int(folds):
        raise ValueError("residual cross-fitting needs at least two events per fold")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(int(n_events), generator=generator)
    holdouts = torch.tensor_split(order, int(folds))
    pairs: list[tuple[Tensor, Tensor]] = []
    for fold, holdout in enumerate(holdouts):
        fit_parts = [part for index, part in enumerate(holdouts) if index != fold]
        fit_index = torch.cat(fit_parts, dim=0)
        pairs.append((fit_index.to(device), holdout.to(device)))
    return tuple(pairs)


def _identity_crossfit_splits(
    condition: Tensor, *, folds: int, seed: int
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Keep an identity in the same fold across Ray reorderings and refits.

    Only visible conditions enter the hash, never generated samples or policy
    noise. Identical conditions always stay together, including duplicate rows.
    Hash on rank zero and broadcast labels to avoid repeating CPU work per GPU.
    """
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import (
        _broadcast_tensor, distributed_context,
    )

    rank, world = distributed_context()
    labels = torch.zeros(len(condition), dtype=torch.int64, device=condition.device)
    if rank == 0:
        assignments = np.empty(len(condition), dtype=np.int64)
        salt = f"residual-condition-v1:{int(seed)}:".encode()
        for start in range(0, len(condition), 8192):
            rows = condition[start:start + 8192].detach().cpu().numpy()
            rows = np.array(rows, dtype="<f4", order="C", copy=True)
            rows[rows == 0] = 0  # Canonicalize signed zero.
            for offset, row in enumerate(rows):
                digest = hashlib.sha256(salt + row.tobytes()).digest()
                assignments[start + offset] = int.from_bytes(digest[:8], "little") % folds
        labels.copy_(torch.from_numpy(assignments).to(labels.device))
    if world > 1:
        _broadcast_tensor(labels)
    pairs = []
    for fold in range(folds):
        fit_index = torch.nonzero(labels != fold, as_tuple=True)[0]
        holdout_index = torch.nonzero(labels == fold, as_tuple=True)[0]
        if min(len(fit_index), len(holdout_index)) < 2:
            raise ValueError("identity cross-fitting needs at least two events in each fold")
        pairs.append((fit_index, holdout_index))
    return tuple(pairs)


@torch.no_grad()
def _weighted_binary_score_metrics(
    data_score: Tensor,
    gen_score: Tensor,
    gen_weight: Tensor,
) -> tuple[float, float, float]:
    """Balanced BCE, tie-safe BA, and oriented weighted AUC from fixed scores."""

    from sklearn.metrics import roc_auc_score

    data = data_score.reshape(-1)
    gen = gen_score.reshape(-1)
    weight = gen_weight.reshape(-1).to(device=gen.device, dtype=gen.dtype)
    if int(weight.numel()) != int(gen.numel()):
        raise ValueError("Gen score and weight populations do not match")
    if min(int(data.numel()), int(gen.numel())) < 1:
        raise ValueError("residual validation populations must be non-empty")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (data, gen, weight)
    ):
        raise FloatingPointError("residual validation received NaN/Inf values")
    weight = weight / weight.mean().clamp_min(torch.finfo(weight.dtype).tiny)
    loss = 0.5 * (
        F.softplus(-data).mean()
        + (weight * F.softplus(gen)).mean()
    )
    truth_credit = (data > 0.0).to(data.dtype) + 0.5 * (
        data == 0.0
    ).to(data.dtype)
    gen_credit = (gen < 0.0).to(gen.dtype) + 0.5 * (
        gen == 0.0
    ).to(gen.dtype)
    balanced_accuracy = 0.5 * (
        truth_credit.mean() + (weight * gen_credit).mean()
    )
    data_np = data.detach().cpu().numpy()
    gen_np = gen.detach().cpu().numpy()
    gen_weight_np = weight.detach().cpu().numpy().astype(np.float64)
    labels = np.concatenate((np.ones(len(data_np)), np.zeros(len(gen_np))))
    scores = np.concatenate((data_np, gen_np))
    weights = np.concatenate((np.ones(len(data_np)), gen_weight_np))
    auc = float(roc_auc_score(labels, scores, sample_weight=weights))
    return float(loss.cpu()), float(balanced_accuracy.cpu()), auc


def fit_residual_ratio_stack(
    *,
    model_factory: Callable[[], nn.Module],
    data_condition: Tensor,
    data_sample: Tensor,
    gen_condition: Tensor,
    gen_sample: Tensor,
    iterations: int,
    fit_config: Any,
    tempering: float,
    seed: int,
    min_iterations: int = 1,
    stop_balanced_accuracy: float | None = None,
    crossfit_folds: int = 2,
    residual_min_auc_gain: float = 1.0e-3,
    validation_data_condition: Tensor | None = None,
    validation_data_sample: Tensor | None = None,
    validation_gen_condition: Tensor | None = None,
    validation_gen_sample: Tensor | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    device: torch.device | None = None,
    warm_start_iterations: tuple[int, ...] = (),
    warm_start_state: Mapping[str, Any] | None = None,
    crossfit_seed: int | None = None,
    crossfit_partition: str = "auto",
    min_steps_per_fold: int = 0,
) -> ResidualRatioResult:
    """Fit safe residual increments with event-level cross-fitting.

    Every event's propagated training weight is predicted out-of-fold by a model
    that did not fit that identity.  Each accepted residual is an equal-weight
    ensemble of the fold models on unseen validation and DGPO events. A proposal
    is committed only when its oriented held-out AUC clears the configured residual
    threshold. Held-out BCE remains diagnostic but is not an outer-iteration gate.
    The first no-op/invalid proposal is kept in diagnostics but never added to the
    reward stack.

    ``warm_start_iterations`` reuses only the corresponding previous fold's
    weights. Matching event-hash fold provenance is mandatory; legacy or changed
    protocols start fresh. Cumulative weights and classifier optimizers always
    start fresh. The selected fitted models (including closure-only fits) are
    returned as a CPU cache for the next refit.
    """

    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import (
        fit_density_ratio,
        global_mean_one,
    )

    if iterations < 1:
        raise ValueError("residual ratio stack needs at least one classifier")
    if int(min_iterations) < 1 or int(min_iterations) > int(iterations):
        raise ValueError("min_iterations must lie in [1, iterations]")
    if stop_balanced_accuracy is not None and not (
        0.5 < float(stop_balanced_accuracy) < 1.0
    ):
        raise ValueError("stop_balanced_accuracy must lie in (0.5, 1)")
    if not 0.0 < float(tempering) <= 1.0:
        raise ValueError("tempering must lie in (0, 1]")
    if int(crossfit_folds) < 2:
        raise ValueError("crossfit_folds must be at least two")
    if not 0.0 <= float(residual_min_auc_gain) < 0.5:
        raise ValueError("residual_min_auc_gain must lie in [0, 0.5)")
    have_validation = validation_data_condition is not None
    if have_validation != all(
        value is not None
        for value in (
            validation_data_condition,
            validation_data_sample,
            validation_gen_condition,
            validation_gen_sample,
        )
    ):
        raise ValueError("residual validation requires all four populations")
    if not have_validation:
        raise ValueError("safe residual fitting requires held-out validation")

    fit_device = gen_sample.device if device is None else torch.device(device)
    n_events = int(gen_condition.shape[0])
    if not all(
        int(value.shape[0]) == n_events
        for value in (data_condition, data_sample, gen_sample)
    ):
        raise ValueError(
            "cross-fitted residual populations must share event identities"
        )
    selected_iterations = tuple(int(value) for value in warm_start_iterations)
    if crossfit_partition not in ("auto", "identity"):
        raise ValueError("crossfit_partition must be 'auto' or 'identity'")
    if type(min_steps_per_fold) is not int or min_steps_per_fold < 0:
        raise ValueError("min_steps_per_fold must be a nonnegative integer")
    if len(set(selected_iterations)) != len(selected_iterations) or any(
        value < 1 or value > int(iterations) for value in selected_iterations
    ):
        raise ValueError("warm_start_iterations must be unique ids in [1, iterations]")
    protocol = None
    initial_states: dict[tuple[int, int], Mapping[str, Tensor]] = {}
    if selected_iterations or crossfit_partition == "identity":
        if not torch.equal(data_condition, gen_condition):
            raise ValueError("warm-start cross-fitting requires paired event conditions")
        protocol = {
            "scheme": "condition_sha256_v1",
            "folds": int(crossfit_folds),
            "seed": int(seed if crossfit_seed is None else crossfit_seed),
            "condition_width": int(gen_condition.shape[-1]),
        }
        fold_pairs = _identity_crossfit_splits(
            gen_condition, folds=int(crossfit_folds), seed=protocol["seed"]
        )
        if warm_start_state is not None and warm_start_state.get("protocol") == protocol:
            for entry in warm_start_state.get("models", ()):
                iteration_id, fold_id = int(entry["iteration"]), int(entry["fold"])
                if iteration_id in selected_iterations:
                    key = (iteration_id, fold_id)
                    if key in initial_states or not 1 <= fold_id <= int(crossfit_folds):
                        raise ValueError("invalid or duplicate warm-start fold")
                    initial_states[key] = entry["state"]
        elif selected_iterations:
            _log.info(
                "[DGPO/omnifold] warm start unavailable: no matching saved fold "
                "protocol; fit fresh classifiers and save identity-stable folds."
            )
    else:
        fold_pairs = _crossfit_splits(
            n_events, folds=int(crossfit_folds), seed=int(seed) + 313,
            device=gen_sample.device,
        )
    next_warm_models: list[dict[str, Any]] = []

    train_logw = torch.zeros(
        gen_sample.shape[:-1],
        device=gen_sample.device,
        dtype=gen_sample.dtype,
    )
    assert validation_gen_sample is not None
    assert validation_gen_condition is not None
    assert validation_data_condition is not None
    assert validation_data_sample is not None
    validation_gen_sample = validation_gen_sample.to(fit_device)
    validation_gen_condition = validation_gen_condition.to(fit_device)
    validation_data_condition = validation_data_condition.to(fit_device)
    validation_data_sample = validation_data_sample.to(fit_device)
    val_logw = torch.zeros(
        validation_gen_sample.shape[:-1],
        device=validation_gen_sample.device,
        dtype=validation_gen_sample.dtype,
    )
    snapshots: list[dict[str, Tensor]] = []
    checkpoint_coefficients: list[float] = []
    checkpoint_iterations: list[int] = []
    diagnostics: list[Any] = []
    converged = False
    model: nn.Module | None = None
    for iteration in range(1, int(iterations) + 1):
        negative_weight = global_mean_one(
            torch.exp(train_logw - train_logw.max())
        )
        validation_negative_weight = global_mean_one(
            torch.exp(val_logw - val_logw.max())
        )
        oof_logit = torch.empty_like(train_logw)
        validation_data_logit: Tensor | None = None
        validation_gen_logit: Tensor | None = None
        fold_snapshots: list[dict[str, Tensor]] = []
        fold_diagnostics: list[Any] = []
        warm_started_folds: list[int] = []
        for fold_index, (fit_index, holdout_index) in enumerate(
            fold_pairs, start=1
        ):
            fold_seed = int(seed) + 1000 * iteration + 37 * fold_index
            model = _seeded_model(
                model_factory,
                seed=fold_seed,
                device=fit_device,
            ).to(fit_device)
            initial_state = initial_states.get((iteration, fold_index))
            if initial_state is not None:
                # Copy weights into a fresh trainable instance. The installed
                # reward stays frozen; no previous Adam moments or log weights
                # are reused. Validate before a potentially partial load.
                current_state = model.state_dict()
                if set(initial_state) != set(current_state) or any(
                    initial_state[key].shape != current_state[key].shape
                    for key in current_state
                ):
                    raise ValueError("warm-start classifier architecture does not match")
                if any(not torch.isfinite(value).all() for value in initial_state.values()):
                    raise FloatingPointError("warm-start classifier has non-finite weights")
                model.load_state_dict(initial_state, strict=True)
                warm_started_folds.append(fold_index)
            _log.info(
                "[DGPO/omnifold] residual iteration=%s fold=%s warm_started=%s; "
                "new optimizer and fresh policy samples",
                iteration, fold_index, initial_state is not None,
            )
            fold_config = _scaled_crossfit_config(
                fit_config,
                float(len(fit_index)) / float(n_events),
                min_steps_per_fold=min_steps_per_fold,
            )
            _log.info("[DGPO/omnifold] residual iteration=%s fold=%s minimum_updates=%s partition=%s",
                      iteration, fold_index, getattr(fold_config, "min_steps", None),
                      "identity" if selected_iterations or crossfit_partition == "identity" else "index")
            validation = (
                validation_data_condition,
                validation_data_sample,
                torch.ones_like(validation_data_sample[..., 0]),
                validation_gen_condition,
                validation_gen_sample,
                validation_negative_weight,
            )
            diag = fit_density_ratio(
                model,
                data_condition[fit_index],
                data_sample[fit_index],
                torch.ones_like(data_sample[fit_index, ..., 0]),
                gen_condition[fit_index],
                gen_sample[fit_index],
                negative_weight[fit_index],
                fold_config,
                fold_seed,
                validation,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda row, iteration=iteration, fold_index=fold_index,
                    warm_started=initial_state is not None: (
                        progress_callback(
                            {
                                "iteration": float(iteration),
                                "fold": float(fold_index),
                                "warm_started": float(warm_started),
                                **row,
                            }
                        )
                    )
                ),
            )
            if fit_config.require_saturation and not bool(diag.saturated):
                raise RuntimeError(
                    f"residual classifier {iteration} fold {fold_index} "
                    "did not saturate"
                )
            fold_diagnostics.append(diag)
            fold_snapshots.append(
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )
            if iteration in selected_iterations:
                next_warm_models.append({
                    "iteration": iteration, "fold": fold_index,
                    "state": fold_snapshots[-1],
                })
            holdout_condition = gen_condition[holdout_index].to(fit_device)
            holdout_sample = gen_sample[holdout_index].to(fit_device)
            holdout_logit = _score_population(
                model,
                holdout_condition,
                holdout_sample,
                fit_config.validation_batch_size,
            )
            oof_logit[holdout_index] = holdout_logit.to(oof_logit.device)
            del holdout_condition, holdout_sample, holdout_logit
            fold_data_logit = _score_population(
                model,
                validation_data_condition,
                validation_data_sample,
                fit_config.validation_batch_size,
            )
            fold_gen_logit = _score_population(
                model,
                validation_gen_condition,
                validation_gen_sample,
                fit_config.validation_batch_size,
            )
            coefficient = 1.0 / float(crossfit_folds)
            validation_data_logit = (
                coefficient * fold_data_logit
                if validation_data_logit is None
                else validation_data_logit + coefficient * fold_data_logit
            )
            validation_gen_logit = (
                coefficient * fold_gen_logit
                if validation_gen_logit is None
                else validation_gen_logit + coefficient * fold_gen_logit
            )

        assert model is not None
        assert validation_data_logit is not None
        assert validation_gen_logit is not None
        validation_loss, validation_accuracy, validation_auc = (
            _weighted_binary_score_metrics(
                validation_data_logit,
                validation_gen_logit,
                validation_negative_weight,
            )
        )
        null_loss = math.log(2.0)
        loss_gain = null_loss - float(validation_loss)
        useful = bool(
            math.isfinite(validation_auc)
            and validation_auc > 0.5 + float(residual_min_auc_gain)
        )
        rejection_reason = None
        if not useful:
            rejection_reason = (
                "held-out residual did not clear the oriented AUC gate: "
                f"loss={validation_loss:.6g}, null={null_loss:.6g}, "
                f"auc={validation_auc:.6g}, "
                f"required_auc>{0.5 + float(residual_min_auc_gain):.6g}"
            )
        iteration_diag = ResidualIterationDiagnostics(
            iteration=int(iteration),
            fold_diagnostics=tuple(fold_diagnostics),
            null_validation_loss=float(null_loss),
            validation_loss=float(validation_loss),
            validation_balanced_accuracy=float(validation_accuracy),
            validation_auc=float(validation_auc),
            validation_loss_gain=float(loss_gain),
            accepted=bool(useful),
            rejection_reason=rejection_reason,
            warm_started_folds=tuple(warm_started_folds),
        )
        diagnostics.append(iteration_diag)
        if progress_callback is not None:
            progress_callback({
                "iteration": float(iteration),
                "fold": 0.0,
                "step": float(
                    max(
                        int(getattr(item, "steps_completed", 0) or 0)
                        for item in fold_diagnostics
                    )
                ),
                "validation_loss": float(validation_loss),
                "validation_balanced_accuracy": float(validation_accuracy),
                "validation_auc": float(validation_auc),
                "null_validation_loss": float(null_loss),
                "validation_loss_gain": float(loss_gain),
                "accepted": float(useful),
                "saturated": float(iteration_diag.saturated),
                "warm_started_folds": float(len(warm_started_folds)),
            })
        if not useful:
            if not snapshots:
                raise RuntimeError(
                    "first residual classifier failed the AUC gate: "
                    f"{rejection_reason}"
                )
            if iteration < int(min_iterations):
                raise RuntimeError(
                    "residual classifier stopped before min_iterations: "
                    f"iteration={iteration}, required={int(min_iterations)}; "
                    f"{rejection_reason}"
                )
            converged = True
            break

        snapshots.extend(fold_snapshots)
        checkpoint_coefficients.extend(
            [1.0 / float(crossfit_folds)] * int(crossfit_folds)
        )
        checkpoint_iterations.extend([int(iteration)] * int(crossfit_folds))
        train_logw = train_logw + float(tempering) * oof_logit
        val_logw = val_logw + float(tempering) * validation_gen_logit

    if not converged:
        raise RuntimeError(
            "residual classifier did not produce a held-out no-op before the "
            f"{int(iterations)}-iteration safety cap"
        )
    assert model is not None
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return ResidualRatioResult(
        classifier=model,
        checkpoints=tuple(snapshots),
        checkpoint_coefficients=tuple(checkpoint_coefficients),
        checkpoint_iterations=tuple(checkpoint_iterations),
        diagnostics=tuple(diagnostics),
        train_log_weight=train_logw.detach(),
        validation_log_weight=val_logw.detach(),
        warm_start_state=(
            {"protocol": protocol, "models": next_warm_models}
            if protocol is not None else None
        ),
    )


def validate_monitor_training_readiness(value: Any) -> dict[str, float] | None:
    """Validate opt-in, event-epoch training floors without an AUC target."""
    if value is None:
        return None
    keys = {"cold_start_min_epochs", "warm_start_min_epochs"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("monitor training_readiness requires cold_start_min_epochs and warm_start_min_epochs")
    result = {}
    for key in keys:
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw < 1:
            raise ValueError(f"monitor {key} must be finite and >= 1")
        result[key] = float(raw)
    if result["cold_start_min_epochs"] < result["warm_start_min_epochs"]:
        raise ValueError("monitor cold-start floor must be >= warm-start floor")
    return result


def fit_independent_evenet_audit(
    *,
    model_factory: Callable[[], nn.Module],
    data_condition: Tensor,
    data_sample: Tensor,
    gen_condition: Tensor,
    gen_sample: Tensor,
    gen_weight: Tensor,
    fit_config: Any,
    seed: int,
    audit_fraction: float = 0.20,
    early_stop_fraction: float = 0.20,
    reuse_early_stop_for_audit: bool = False,
    early_stop_auc_gap: float | None = None,
    early_stop_balanced_accuracy_lcb: float | None = None,
    early_stop_balanced_accuracy_confidence_z: float = 1.96,
    early_stop_balanced_accuracy_required_consecutive: int = 1,
    warm_start_cache: dict[str, Any] | None = None,
    identity_split_seed: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    training_readiness: Mapping[str, Any] | None = None,
) -> EvenetAuditResult:
    """Fit a temporary EveNet judge and score evaluation identities.

    Splitting happens before the candidate axis is flattened, so every candidate from
    one event stays in exactly one split. By default fit, early-stop, and final-audit
    are disjoint. ``reuse_early_stop_for_audit`` instead uses an 80/20 train/validation
    protocol and reports final metrics on the restored-best model's validation split.
    The caller must provide a factory distinct from all reward-classifier factories.
    Optional raw-monitor warm starts use condition-hashed 80/20 identities and
    transfer only weights, never optimizer or early-stopping state. Without a
    cache (including trust monitors), initialization remains fresh.
    Optional training readiness gives cold and certified warm fits separate
    event-epoch floors. A completed fit certifies the training budget, not the
    statistical power of this architecture or equality of the distributions.
    """

    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig, fit_density_ratio

    readiness = validate_monitor_training_readiness(training_readiness)
    if readiness is not None:
        if not isinstance(fit_config, RatioFitConfig) or fit_config.sampling != "independent_epoch_shuffle":
            raise ValueError("monitor training_readiness requires epoch-shuffle RatioFitConfig")
        if not reuse_early_stop_for_audit or early_stop_fraction != .20 or identity_split_seed is None:
            raise ValueError("monitor training_readiness requires identity-stable 80/20 train/validation")
        if early_stop_auc_gap is not None or early_stop_balanced_accuracy_lcb is not None:
            raise ValueError("monitor training_readiness cannot use classifier-trust threshold early stopping")

    n_events = int(data_condition.shape[0])
    if not (
        int(data_sample.shape[0])
        == int(gen_condition.shape[0])
        == int(gen_sample.shape[0])
        == int(gen_weight.shape[0])
        == n_events
    ):
        raise ValueError("audit populations must share event identities")
    if n_events < 30:
        raise ValueError("independent EveNet audit needs at least 30 events")
    if not 0.0 < early_stop_fraction < 0.5:
        raise ValueError("early-stop fraction must lie in (0, 0.5)")
    if not reuse_early_stop_for_audit and not 0.0 < audit_fraction < 0.5:
        raise ValueError("audit fraction must lie in (0, 0.5)")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(n_events, generator=generator)
    n_audit = (
        0
        if reuse_early_stop_for_audit
        else max(1, int(round(n_events * audit_fraction)))
    )
    n_early = max(1, int(round(n_events * early_stop_fraction)))
    if n_audit + n_early >= n_events:
        raise ValueError("audit split leaves no fit events")
    audit_idx = order[:n_audit]
    early_idx = order[n_audit : n_audit + n_early]
    fit_idx = order[n_audit + n_early :]
    device = gen_sample.device
    fit_idx, early_idx, audit_idx = (
        fit_idx.to(device), early_idx.to(device), audit_idx.to(device)
    )
    split_seed = int(seed if identity_split_seed is None else identity_split_seed)
    protocol = {"schema": "raw-monitor-condition-split-v1", "seed": split_seed, "folds": 5}
    if warm_start_cache is not None or identity_split_seed is not None:
        if not reuse_early_stop_for_audit or early_stop_fraction != 0.20:
            raise ValueError("warm-start raw monitor requires identity-stable 80/20 train/validation")
        # Never let an old training identity become validation when Ray reorders
        # rows or the pool grows. The visible condition, not generated x, owns
        # the assignment. Duplicate conditions therefore stay together too.
        fit_idx, early_idx = _identity_crossfit_splits(
            data_condition, folds=5, seed=split_seed,
        )[0]

    model = _seeded_model(model_factory, seed=int(seed) + 71, device=device).to(device)
    initial_state = (
        warm_start_cache.get("state")
        if warm_start_cache is not None and warm_start_cache.get("protocol") == protocol
        else None
    )
    training_policy = None
    if readiness is not None:
        training_policy = {
            "schema": "raw-monitor-training-readiness-v1",
            "epoch_floors": readiness,
            "fit_config": asdict(fit_config),
            "condition_width": int(data_condition.shape[-1]),
        }
        if initial_state is not None and warm_start_cache.get("training_policy") != training_policy:
            # An old 60-step null fit is not a certified warm start. This also
            # rejects changed training/packing protocols without mutating cache.
            initial_state = None
            _log.info("[DGPO/omnifold] raw monitor training protocol changed or uncertified; cold-start fitting required")
    if initial_state is not None:
        current_state = model.state_dict()
        if set(initial_state) != set(current_state) or any(
            initial_state[key].shape != current_state[key].shape for key in current_state
        ):
            raise ValueError("raw monitor warm-start architecture mismatch")
        if any(not torch.isfinite(value).all() for value in initial_state.values()):
            raise FloatingPointError("raw monitor warm-start weights are non-finite")
        model.load_state_dict(initial_state, strict=True)
    training_steps_per_epoch = 0
    if readiness is not None:
        training_steps_per_epoch = (
            len(fit_idx) // fit_config.batch_size if fit_config.drop_last_batch
            else max(1, math.ceil(len(fit_idx) / fit_config.batch_size))
        )
        if training_steps_per_epoch < 1:
            raise ValueError("monitor training fold is smaller than its drop-last batch")
        min_epochs = readiness["warm_start_min_epochs" if initial_state is not None else "cold_start_min_epochs"]
        fit_config = replace(
            fit_config,
            min_steps=max(fit_config.min_steps, math.ceil(min_epochs * training_steps_per_epoch)),
            require_saturation=True,
        )
        fit_config.validate()  # Fail before training if a finite budget is too short.
        _log.info(
            "[DGPO/omnifold] raw monitor training budget: warm_started=%s fit_events=%s "
            "steps_per_epoch=%s min_epochs=%s min_steps=%s patience_evaluations=%s",
            initial_state is not None, len(fit_idx), training_steps_per_epoch,
            min_epochs, fit_config.min_steps, fit_config.validation_patience_evaluations,
        )
    _log.info("[DGPO/omnifold] raw/audit warm_started=%s; optimizer and early stopping reset",
              initial_state is not None)
    n_params = sum(int(parameter.numel()) for parameter in model.parameters())
    _log.info(
        "[DGPO/omnifold] audit classifier ready on %s (%s params); "
        "NCCL-syncing weights, then fitting fit=%s early_stop=%s heldout=%s",
        device,
        n_params,
        int(len(fit_idx)),
        int(len(early_idx)),
        int(len(audit_idx)),
    )
    score_batch = max(1, int(getattr(fit_config, "validation_batch_size", 8192)))

    def evaluate_early_stop(fitted_model: nn.Module) -> tuple[float, float, float]:
        data_score = _score_population(
            fitted_model,
            data_condition[early_idx],
            data_sample[early_idx],
            score_batch,
        )
        gen_score = _score_population(
            fitted_model,
            gen_condition[early_idx],
            gen_sample[early_idx],
            score_batch,
        )
        return _weighted_binary_score_metrics(
            data_score,
            gen_score,
            gen_weight[early_idx],
        )

    diagnostics = fit_density_ratio(
        model,
        data_condition[fit_idx],
        data_sample[fit_idx],
        torch.ones_like(data_sample[fit_idx, 0]),
        gen_condition[fit_idx],
        gen_sample[fit_idx],
        gen_weight[fit_idx],
        fit_config,
        int(seed) + 71,
        (
            data_condition[early_idx],
            data_sample[early_idx],
            torch.ones_like(data_sample[early_idx, 0]),
            gen_condition[early_idx],
            gen_sample[early_idx],
            gen_weight[early_idx],
        ),
        progress_callback=(
            None
            if progress_callback is None
            else lambda row: progress_callback({
                "iteration": 1.0, "warm_started": float(initial_state is not None), **row,
            })
        ),
        validation_evaluator=evaluate_early_stop,
        stop_when_validation_auc_gap_exceeds=early_stop_auc_gap,
        stop_when_validation_balanced_accuracy_lcb_exceeds=(
            early_stop_balanced_accuracy_lcb
        ),
        validation_balanced_accuracy_lcb_confidence_z=(
            early_stop_balanced_accuracy_confidence_z
        ),
        validation_balanced_accuracy_lcb_events_per_class=int(len(early_idx)),
        validation_balanced_accuracy_lcb_required_consecutive=(
            early_stop_balanced_accuracy_required_consecutive
        ),
    )
    if bool(getattr(fit_config, "require_saturation", False)) and not bool(
        getattr(diagnostics, "saturated", False)
    ):
        raise RuntimeError("independent EveNet audit did not saturate")
    training_ready = bool(
        readiness is not None and diagnostics.saturated
        and int(diagnostics.steps_completed or 0) >= fit_config.min_steps
    )
    if readiness is not None and not training_ready:
        raise RuntimeError("raw monitor did not complete its required training budget; no cache or baseline committed")
    model.eval()
    score_idx = early_idx if reuse_early_stop_for_audit else audit_idx
    # Score the selected evaluation split in batches and shard it across ranks. A single
    # full-population forward is both an OOM risk and 16x redundant work when
    # every rank runs the same audit.
    data_score = _score_population(
        model, data_condition[score_idx], data_sample[score_idx], score_batch
    ).reshape(-1)
    gen_score = _score_population(
        model, gen_condition[score_idx], gen_sample[score_idx], score_batch
    ).reshape(-1)
    if not bool(torch.isfinite(data_score).all().item()) or not bool(
        torch.isfinite(gen_score).all().item()
    ):
        raise FloatingPointError("independent EveNet audit produced NaN/Inf logits")
    if not bool(torch.isfinite(gen_weight[score_idx]).all().item()):
        raise FloatingPointError("independent EveNet audit received NaN/Inf weights")
    _, balanced_accuracy, auc = _weighted_binary_score_metrics(
        data_score,
        gen_score,
        gen_weight[score_idx],
    )
    data_np = data_score.detach().cpu().numpy()
    gen_np = gen_score.detach().cpu().numpy()
    gen_w = gen_weight[score_idx].reshape(-1).detach().cpu().numpy().astype(np.float64)
    gen_w = gen_w / max(float(np.mean(gen_w)), 1e-12)
    truth_tpr = float(np.mean(data_np > 0.0))
    gen_negative = (gen_np < 0.0).astype(np.float64)
    gen_tnr = float(np.sum(gen_w * gen_negative) / max(float(np.sum(gen_w)), 1e-12))
    if warm_start_cache is not None:
        # Commit only after fit + scoring succeed. No optimizer or generated
        # data are inherited, and the reward/trust classifier banks stay separate.
        next_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if any(not torch.isfinite(value).all() for value in next_state.values()):
            raise FloatingPointError("raw monitor fitted weights are non-finite")
        warm_start_cache.clear()
        warm_start_cache.update(protocol=protocol, state=next_state)
        if training_policy is not None:
            warm_start_cache["training_policy"] = training_policy
    return EvenetAuditResult(
        auc=auc,
        auc_gap=float(abs(auc - 0.5)),
        balanced_accuracy=balanced_accuracy,
        truth_positive_rate=truth_tpr,
        gen_negative_rate=gen_tnr,
        fit_diagnostics=diagnostics,
        fit_events=int(len(fit_idx)),
        early_stop_events=int(len(early_idx)),
        audit_events=int(len(score_idx)),
        warm_started=initial_state is not None,
        training_min_steps=int(fit_config.min_steps) if readiness is not None else 0,
        training_steps_per_epoch=training_steps_per_epoch,
        training_ready=training_ready,
    )


class FrozenResidualRatioReward(nn.Module):
    """Cumulative cross-fitted log ratio replayed through one module.

    The architecture exists once. ``checkpoints`` contains only its saved weights;
    :meth:`forward` loads each snapshot in order and combines fold logits using
    ``checkpoint_coefficients`` before summing residual iterations.
    For checkpoint compatibility the constructor also accepts the old tuple of
    classifier modules and immediately compacts them into one module plus snapshots.
    """

    input_kind = "packed_evenet_event_physical_invisible"

    def __init__(
        self,
        classifier: nn.Module | tuple[nn.Module, ...],
        checkpoints: tuple[Mapping[str, Tensor], ...] | None = None,
        *,
        tempering: float = 1.0,
        checkpoint_coefficients: tuple[float, ...] | None = None,
        checkpoint_iterations: tuple[int, ...] | None = None,
        warm_start_state: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if checkpoints is None:
            if not isinstance(classifier, tuple) or not classifier:
                raise ValueError("frozen residual reward needs at least one checkpoint")
            legacy = classifier
            classifier = legacy[0]
            checkpoints = tuple(
                {
                    name: value.detach().clone()
                    for name, value in module.state_dict().items()
                }
                for module in legacy
            )
        if not checkpoints:
            raise ValueError("frozen residual reward needs at least one checkpoint")
        if checkpoint_coefficients is None:
            checkpoint_coefficients = tuple(1.0 for _ in checkpoints)
        if checkpoint_iterations is None:
            checkpoint_iterations = tuple(range(1, len(checkpoints) + 1))
        if not (
            len(checkpoints)
            == len(checkpoint_coefficients)
            == len(checkpoint_iterations)
        ):
            raise ValueError("residual checkpoint metadata lengths do not match")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in checkpoint_coefficients
        ):
            raise ValueError("residual checkpoint coefficients must be positive")
        observed_iterations = tuple(
            sorted(set(int(value) for value in checkpoint_iterations))
        )
        expected_iterations = tuple(
            range(1, max(int(value) for value in checkpoint_iterations) + 1)
        )
        if observed_iterations != expected_iterations:
            raise ValueError("residual checkpoint iteration ids must be contiguous")
        for iteration in sorted(set(int(value) for value in checkpoint_iterations)):
            total = sum(
                float(coefficient)
                for coefficient, group in zip(
                    checkpoint_coefficients, checkpoint_iterations
                )
                if int(group) == iteration
            )
            if not math.isclose(total, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
                raise ValueError(
                    f"residual iteration {iteration} ensemble weights sum to {total}"
                )
        self.classifier = classifier
        self._checkpoints = [
            {name: value.detach().clone() for name, value in checkpoint.items()}
            for checkpoint in checkpoints
        ]
        self._checkpoint_coefficients = tuple(
            float(value) for value in checkpoint_coefficients
        )
        self._checkpoint_iterations = tuple(
            int(value) for value in checkpoint_iterations
        )
        self.tempering = float(tempering)
        # Training-only CPU cache; never replayed in the cumulative reward and
        # never moved onto each GPU by _apply. Includes a selected closure-only
        # iteration if it was fitted but not installed as a ratio increment.
        from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import _clone_to_cpu
        self.warm_start_state = _clone_to_cpu(warm_start_state)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_fit_result(
        cls, result: ResidualRatioResult, *, tempering: float
    ) -> "FrozenResidualRatioReward":
        return cls(
            result.classifier,
            result.checkpoints,
            tempering=float(tempering),
            checkpoint_coefficients=result.checkpoint_coefficients,
            checkpoint_iterations=result.checkpoint_iterations,
            warm_start_state=result.warm_start_state,
        )

    @property
    def num_iterations(self) -> int:
        return max(self._checkpoint_iterations, default=0)

    @property
    def num_checkpoints(self) -> int:
        return len(self._checkpoints)

    @property
    def packing_spec(self) -> EventPackingSpec:
        return self.classifier.packing_spec

    def _apply(self, fn: Callable[[Tensor], Tensor]):
        super()._apply(fn)
        self._checkpoints = [
            {name: fn(value) for name, value in checkpoint.items()}
            for checkpoint in self._checkpoints
        ]
        return self

    def _load_checkpoint(self, index: int) -> None:
        self.classifier.load_state_dict(self._checkpoints[int(index)], strict=True)

    @torch.no_grad()
    def checkpoint_logits(
        self,
        index: int,
        condition: Tensor,
        candidate: Tensor,
        *,
        batch_size: int | None = None,
    ) -> Tensor:
        self._load_checkpoint(index)
        if batch_size is None:
            return self.classifier(condition, candidate)
        return _score_population(
            self.classifier, condition, candidate, int(batch_size)
        )

    @torch.no_grad()
    def forward(self, condition: Tensor, candidate: Tensor) -> Tensor:
        total: Tensor | None = None
        for index, coefficient in enumerate(self._checkpoint_coefficients):
            logit = float(coefficient) * self.checkpoint_logits(
                index, condition, candidate
            )
            total = logit if total is None else total + logit
        assert total is not None
        return self.tempering * total

    def assert_frozen(self) -> None:
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise RuntimeError("residual OmniFold reward must remain frozen")

    def serializable_payload(self) -> dict[str, Any]:
        increments: list[dict[str, Any]] = []
        if not isinstance(self.classifier, EvenetAdapterRatioClassifier):
            raise TypeError(
                "EveNet residual reward can serialize only an adapter ratio classifier"
            )
        for index in range(self.num_checkpoints):
            self._load_checkpoint(index)
            increments.append(self.classifier.peft_payload())
        digests = {item.get("base_digest") for item in increments}
        if len(digests) != 1:
            raise ValueError("residual PEFT banks must share one frozen backbone digest")
        payload = {
            "schema_version": PEFT_SCHEMA_VERSION,
            "kind": "evenet_adapter_residual_crossfit",
            "tempering": float(self.tempering),
            "base_digest": next(iter(digests)),
            "increments": increments,
            "increment_coefficients": list(self._checkpoint_coefficients),
            "increment_iterations": list(self._checkpoint_iterations),
        }
        # Omit for legacy stacks so checkpoint digests still round-trip exactly.
        if self.warm_start_state is not None:
            payload["warm_start_state"] = deepcopy(self.warm_start_state)
        return payload

    @classmethod
    def from_serializable_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        model_builder: Callable[[EventPackingSpec], EvenetAdapterRatioClassifier],
        device: torch.device,
    ) -> "FrozenResidualRatioReward":
        if int(payload.get("schema_version", -1)) != PEFT_SCHEMA_VERSION:
            raise ValueError("unsupported EveNet residual reward schema")
        if str(payload.get("kind", "")) not in {
            "evenet_adapter_residual",
            "evenet_adapter_residual_sequential",
            "evenet_adapter_residual_crossfit",
        }:
            raise ValueError("payload is not an EveNet adapter residual reward")
        increments = tuple(payload.get("increments", ()))
        if not increments:
            raise ValueError("EveNet residual reward payload has no checkpoints")
        classifier = EvenetAdapterRatioClassifier.from_peft_payload(
            increments[0], model_builder=model_builder, device=device
        )
        checkpoints: list[dict[str, Tensor]] = []
        expected_spec = increments[0].get("packing_spec")
        expected_digest = increments[0].get("base_digest")
        for item in increments:
            if item.get("packing_spec") != expected_spec:
                raise ValueError("residual checkpoints use different event packing specs")
            if item.get("base_digest") != expected_digest:
                raise ValueError("residual checkpoints use different backbone digests")
            checkpoint = {
                f"bank.{name}": value.detach().clone()
                for name, value in dict(item.get("state") or {}).items()
            }
            checkpoint.update(
                {
                    f"{_BACKBONE_STATE_PREFIX}{name}": value.detach().clone()
                    for name, value in dict(item.get("body") or {}).items()
                }
            )
            checkpoints.append(checkpoint)
        reward = cls(
            classifier,
            tuple(checkpoints),
            tempering=float(payload.get("tempering", 1.0)),
            checkpoint_coefficients=tuple(
                float(value)
                for value in payload.get(
                    "increment_coefficients",
                    [1.0] * len(checkpoints),
                )
            ),
            checkpoint_iterations=tuple(
                int(value)
                for value in payload.get(
                    "increment_iterations",
                    range(1, len(checkpoints) + 1),
                )
            ),
            warm_start_state=payload.get("warm_start_state"),
        )
        reward.to(device).eval()
        reward.assert_frozen()
        return reward


__all__ = [
    "EventPackingSpec",
    "EvenetAuditResult",
    "AdaLNZeroCandidateDecoder",
    "CandidateConditionedRatioHead",
    "EvenetAdapterModelBuilder",
    "EvenetAdapterRatioClassifier",
    "EvenetRatioPEFTBank",
    "PeriodicPairAuditClassifier",
    "FrozenResidualRatioReward",
    "peft_bank_factory",
    "ResidualIterationDiagnostics",
    "ResidualRatioResult",
    "configure_adapter_training",
    "fit_residual_ratio_stack",
    "fit_independent_evenet_audit",
    "pack_event_inputs",
    "periodic_tau_pair_feature_dim",
    "periodic_tau_pair_features",
    "unpack_event_inputs",
]
