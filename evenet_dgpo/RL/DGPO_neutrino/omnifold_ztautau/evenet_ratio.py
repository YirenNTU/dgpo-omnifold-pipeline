"""EveNet classifiers for the no-smearing OmniFold variant.

The production problem is already expressed in the target neutrino space: there is
no detector response to invert.  Each round therefore fits one residual conditional
density ratio and adds its logit to the cumulative log weight.  There is deliberately
no OmniFold Step-2 projection classifier in this module.

The adaptive path owns one reusable classifier architecture and creates fresh
cross-fit fold instances for every residual. It trains PET's registered internal
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
from dataclasses import dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from evenet.network.body.embedding import PointCloudPositionalEmbedding


_log = logging.getLogger(__name__)

_EVENT_KEYS = ("x", "x_mask", "conditions", "conditions_mask")


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
    batch: Mapping[str, Any], spec: EventPackingSpec | None = None
) -> tuple[Tensor, EventPackingSpec]:
    """Pack the four deterministic EveNet inputs into one ``(B, D)`` tensor."""

    missing = [key for key in _EVENT_KEYS if not isinstance(batch.get(key), Tensor)]
    if missing:
        raise KeyError(f"EveNet ratio classifier needs tensor inputs {missing}")
    tensors = {key: batch[key] for key in _EVENT_KEYS}
    batch_size = int(tensors["x"].shape[0])
    if any(int(value.shape[0]) != batch_size for value in tensors.values()):
        raise ValueError("all packed EveNet inputs must share their batch dimension")
    observed = EventPackingSpec(
        {key: tuple(int(v) for v in value.shape[1:]) for key, value in tensors.items()}
    )
    if spec is not None and observed != spec:
        raise ValueError(f"event input shapes changed: expected {spec.shapes}, got {observed.shapes}")
    packed = torch.cat(
        [tensors[key].reshape(batch_size, -1).to(dtype=torch.float32) for key in _EVENT_KEYS],
        dim=-1,
    )
    return packed, observed


def unpack_event_inputs(packed: Tensor, spec: EventPackingSpec) -> dict[str, Tensor]:
    """Inverse of :func:`pack_event_inputs`; masks are restored as booleans."""

    if packed.ndim != 2 or int(packed.shape[-1]) != spec.width:
        raise ValueError(f"packed events must be (B, {spec.width}), got {tuple(packed.shape)}")
    result: dict[str, Tensor] = {}
    offset = 0
    for key in _EVENT_KEYS:
        shape = spec.shapes[key]
        width = int(np.prod(shape))
        value = packed[:, offset : offset + width].reshape(len(packed), *shape)
        result[key] = value > 0.5 if key.endswith("mask") else value
        offset += width
    return result


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
    train_invisible_projector: bool = False,
    train_backbone: bool = False,
) -> None:
    """Freeze the shared body, then optionally reopen selected tensors.

    PET's registered internal adapters are always trainable. ``train_backbone``
    additionally opens every module on this classifier's forward path,
    including PET attention.
    ``train_encoder`` opens GlobalEmbedding (AdaLN event token).
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
    ) -> None:
        super().__init__()
        self.position_encoder = PointCloudPositionalEmbedding(
            num_points=int(max_position_length),
            embed_dim=int(token_dim),
        )
        self.decoder = AdaLNZeroCandidateDecoder(
            token_dim=int(token_dim),
            event_dim=int(event_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.output = nn.Linear(self.decoder.num_slots * int(hidden_dim), 1)
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
        train_invisible_projector: bool = False,
        train_backbone: bool = False,
        asymmetric_attention: bool = False,
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
            train_invisible_projector=train_invisible_projector,
            train_backbone=train_backbone,
        )
        self._train_layernorm = bool(train_layernorm)
        self._train_encoder = bool(train_encoder)
        self._train_invisible_projector = bool(train_invisible_projector)
        self._include_train_invisible_projector_in_payload = True
        self._train_backbone = bool(train_backbone)
        self._asymmetric_attention = bool(asymmetric_attention)
        self._include_asymmetric_attention_in_payload = True
        self._backbone_state_keys = tuple(
            name
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        )
        self._shared = _SharedModuleRef(backbone)
        self.packing_spec = packing_spec
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
                ("position_encoder", "bank.position_encoder."),
                ("object_encoder", "backbone.ObjectEncoder."),
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
        if candidate_flat.ndim == 3:
            bsz, count = int(candidate_flat.shape[0]), int(candidate_flat.shape[1])
            if int(packed_event.shape[0]) != bsz:
                raise ValueError("event and candidate batch dimensions do not match")
            packed_event = packed_event[:, None, :].expand(-1, count, -1).reshape(bsz * count, -1)
            candidate_flat = candidate_flat.reshape(bsz * count, self.candidate_width)
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
        event_token = self._event_token_from_global(global_embedding)
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
        logits = self.bank.output(hidden.reshape(hidden.shape[0], -1)).squeeze(-1)
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
                self, "_include_train_invisible_projector_in_payload", True
            ):
                classifier_config["train_invisible_projector"] = (
                    self._train_invisible_projector
                )
            if getattr(self, "_include_asymmetric_attention_in_payload", True):
                classifier_config["asymmetric_attention"] = (
                    self._asymmetric_attention
                )
            classifier_config["adapter_placement"] = "internal"
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
        model._include_asymmetric_attention_in_payload = (
            "asymmetric_attention"
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
        train_invisible_projector: bool = False,
        train_backbone: bool = False,
        asymmetric_attention: bool = False,
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
        self._train_invisible_projector = bool(train_invisible_projector)
        self._train_backbone = bool(train_backbone)
        self._asymmetric_attention = bool(asymmetric_attention)
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
        train_invisible_projector: bool | None = None,
        train_backbone: bool | None = None,
        asymmetric_attention: bool | None = None,
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
            )
            if name is not None:
                self._banks[name] = bank
        classifier = EvenetAdapterRatioClassifier(
            backbone,
            packing_spec,
            bank=bank,
            train_layernorm=self._train_layernorm,
            train_encoder=self._train_encoder,
            train_invisible_projector=resolved_train_invisible_projector,
            train_backbone=resolved_train_backbone,
            asymmetric_attention=resolved_asymmetric_attention,
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
) -> Callable[[], EvenetAdapterRatioClassifier]:
    """Build one classifier view, resetting the named bank when requested."""

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

    @property
    def iterations(self) -> int:
        return max(self.checkpoint_iterations, default=0)


@dataclass(frozen=True)
class EvenetAuditResult:
    """Final event-held-out metrics from a fresh temporary EveNet judge."""

    auc: float
    auc_gap: float
    balanced_accuracy: float
    truth_positive_rate: float
    gen_negative_rate: float
    fit_diagnostics: Any
    fit_events: int
    early_stop_events: int
    audit_events: int


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


def _scaled_crossfit_config(fit_config: Any, train_fraction: float) -> Any:
    """Preserve epoch-based controls when a fold trains on fewer events."""

    if not is_dataclass(fit_config):
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
) -> ResidualRatioResult:
    """Fit safe residual increments with event-level cross-fitting.

    Every event's propagated training weight is predicted out-of-fold by a model
    that did not fit that identity.  Each accepted residual is an equal-weight
    ensemble of the fold models on unseen validation and DGPO events. A proposal
    is committed only when its oriented held-out AUC clears the configured residual
    threshold. Held-out BCE remains diagnostic but is not an outer-iteration gate.
    The first no-op/invalid proposal is kept in diagnostics but never added to the
    reward stack.
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
    fold_pairs = _crossfit_splits(
        n_events,
        folds=int(crossfit_folds),
        seed=int(seed) + 313,
        device=gen_sample.device,
    )

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
        for fold_index, (fit_index, holdout_index) in enumerate(
            fold_pairs, start=1
        ):
            fold_seed = int(seed) + 1000 * iteration + 37 * fold_index
            model = _seeded_model(
                model_factory,
                seed=fold_seed,
                device=fit_device,
            ).to(fit_device)
            fold_config = _scaled_crossfit_config(
                fit_config,
                float(len(fit_index)) / float(n_events),
            )
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
                    else lambda row, iteration=iteration, fold_index=fold_index: (
                        progress_callback(
                            {
                                "iteration": float(iteration),
                                "fold": float(fold_index),
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
    )


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
    early_stop_auc_gap: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EvenetAuditResult:
    """Fit a fresh temporary EveNet judge and score untouched event identities.

    Splitting happens before the candidate axis is flattened, so every candidate from
    one event stays in exactly one of fit, early-stop, or final-audit.  The caller must
    provide a factory that is distinct from all reward-classifier factories.
    """

    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio

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
    if not 0.0 < audit_fraction < 0.5 or not 0.0 < early_stop_fraction < 0.5:
        raise ValueError("audit and early-stop fractions must lie in (0, 0.5)")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(n_events, generator=generator)
    n_audit = max(1, int(round(n_events * audit_fraction)))
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

    model = _seeded_model(model_factory, seed=int(seed) + 71, device=device).to(device)
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
            else lambda row: progress_callback({"iteration": 1.0, **row})
        ),
        validation_evaluator=evaluate_early_stop,
        stop_when_validation_auc_gap_exceeds=early_stop_auc_gap,
    )
    if bool(getattr(fit_config, "require_saturation", False)) and not bool(
        getattr(diagnostics, "saturated", False)
    ):
        raise RuntimeError("independent EveNet audit did not saturate")
    model.eval()
    # Score the held-out split in batches and shard it across ranks. A single
    # full-population forward is both an OOM risk and 16x redundant work when
    # every rank runs the same audit.
    data_score = _score_population(
        model, data_condition[audit_idx], data_sample[audit_idx], score_batch
    ).reshape(-1)
    gen_score = _score_population(
        model, gen_condition[audit_idx], gen_sample[audit_idx], score_batch
    ).reshape(-1)
    if not bool(torch.isfinite(data_score).all().item()) or not bool(
        torch.isfinite(gen_score).all().item()
    ):
        raise FloatingPointError("independent EveNet audit produced NaN/Inf logits")
    if not bool(torch.isfinite(gen_weight[audit_idx]).all().item()):
        raise FloatingPointError("independent EveNet audit received NaN/Inf weights")
    _, balanced_accuracy, auc = _weighted_binary_score_metrics(
        data_score,
        gen_score,
        gen_weight[audit_idx],
    )
    data_np = data_score.detach().cpu().numpy()
    gen_np = gen_score.detach().cpu().numpy()
    gen_w = gen_weight[audit_idx].reshape(-1).detach().cpu().numpy().astype(np.float64)
    gen_w = gen_w / max(float(np.mean(gen_w)), 1e-12)
    truth_tpr = float(np.mean(data_np > 0.0))
    gen_negative = (gen_np < 0.0).astype(np.float64)
    gen_tnr = float(np.sum(gen_w * gen_negative) / max(float(np.sum(gen_w)), 1e-12))
    return EvenetAuditResult(
        auc=auc,
        auc_gap=float(abs(auc - 0.5)),
        balanced_accuracy=balanced_accuracy,
        truth_positive_rate=truth_tpr,
        gen_negative_rate=gen_tnr,
        fit_diagnostics=diagnostics,
        fit_events=int(len(fit_idx)),
        early_stop_events=int(len(early_idx)),
        audit_events=int(len(audit_idx)),
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
        return {
            "schema_version": PEFT_SCHEMA_VERSION,
            "kind": "evenet_adapter_residual_crossfit",
            "tempering": float(self.tempering),
            "base_digest": next(iter(digests)),
            "increments": increments,
            "increment_coefficients": list(self._checkpoint_coefficients),
            "increment_iterations": list(self._checkpoint_iterations),
        }

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
    "FrozenResidualRatioReward",
    "peft_bank_factory",
    "ResidualIterationDiagnostics",
    "ResidualRatioResult",
    "configure_adapter_training",
    "fit_residual_ratio_stack",
    "fit_independent_evenet_audit",
    "pack_event_inputs",
    "unpack_event_inputs",
]
