"""Frozen Ztautau OmniFold reward adapter for the DGPO candidate interface."""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EvenetAdapterModelBuilder,
    EventPackingSpec,
    FrozenResidualRatioReward,
    pack_event_inputs,
)
from RL.DGPO_neutrino.rewards import BaseReward, apply_event_valid_to_rewards


REWARD_CHECKPOINT_KEY = "dgpo_omnifold_reward_metadata"
REWARD_STACK_CHECKPOINT_KEY = "dgpo_omnifold_reward_stack"

_log = logging.getLogger(__name__)


def _update_payload_digest(digest: Any, value: Any) -> None:
    """Hash nested metadata and tensors without relying on pickle byte layout."""
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        byte_view = tensor.reshape(1) if tensor.ndim == 0 else tensor
        digest.update(byte_view.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_payload_digest(digest, str(key))
            _update_payload_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        for item in value:
            _update_payload_digest(digest, item)
        return
    digest.update(type(value).__name__.encode())
    digest.update(b"\0")
    digest.update(repr(value).encode())


def payload_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_payload_digest(digest, value)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint or reward artifact not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ZtautauOmniFoldReward(BaseReward):
    """Score all ``K`` angular-delta candidates with one frozen ratio stack.

    Event truth is intentionally absent from the classifier condition. The only
    per-event inputs are the same visible objects and global conditions packed
    during the K=1 OmniFold fit.
    """

    def __init__(
        self,
        frozen_reward: FrozenResidualRatioReward | None,
        *,
        bundle_sha256: str,
        policy_reference_sha256: str,
        base_digest: str,
        stack_sha256: str,
        bundle_schema_version: int,
        device: torch.device,
        artifact_path: Path | None = None,
        model_builder: EvenetAdapterModelBuilder | None = None,
        reward_round_id: int = 0,
        reference_kind: str = "checkpoint_sha256",
    ) -> None:
        if frozen_reward is not None:
            frozen_reward.to(device).eval()
            frozen_reward.assert_frozen()
        self._reward = frozen_reward
        self._packing_spec = (
            None if frozen_reward is None else frozen_reward.packing_spec
        )
        self._bundle_sha256 = str(bundle_sha256)
        self._policy_reference_sha256 = str(policy_reference_sha256)
        self._base_digest = str(base_digest)
        self._stack_sha256 = str(stack_sha256)
        self._bundle_schema_version = int(bundle_schema_version)
        self._device = device
        self._artifact_path = artifact_path
        # Keep the shared frozen EveNet body alive for the classifier's module ref.
        self._model_builder = model_builder
        self._reward_round_id = int(reward_round_id)
        self._reference_kind = str(reference_kind)

    @property
    def name(self) -> str:
        return "omnifold"

    @property
    def iterations(self) -> int:
        return 0 if self._reward is None else int(self._reward.num_iterations)

    @property
    def is_installed(self) -> bool:
        return self._reward is not None

    @property
    def policy_reference_sha256(self) -> str:
        return self._policy_reference_sha256

    @property
    def artifact_path(self) -> Path | None:
        return self._artifact_path

    @property
    def frozen_reward(self) -> FrozenResidualRatioReward:
        if self._reward is None:
            raise RuntimeError("OmniFold reward stack has not been bootstrapped")
        return self._reward

    @property
    def model_builder(self) -> EvenetAdapterModelBuilder:
        if self._model_builder is None:
            raise RuntimeError("OmniFold reward has no EveNet adapter model builder")
        return self._model_builder

    @property
    def reward_round_id(self) -> int:
        return self._reward_round_id

    @property
    def reference_kind(self) -> str:
        return self._reference_kind

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "kind": "ztautau_evenet_omnifold_reward",
            "bundle_schema_version": self._bundle_schema_version,
            "bundle_sha256": self._bundle_sha256,
            "policy_reference_sha256": self._policy_reference_sha256,
            "base_digest": self._base_digest,
            "stack_sha256": self._stack_sha256,
            "iterations": self.iterations,
            "candidates_per_event_for_fit": 1,
            "reward_round_id": self._reward_round_id,
            "reference_kind": self._reference_kind,
        }

    def stack_payload(self) -> dict[str, Any]:
        if self._reward is None:
            raise RuntimeError("cannot serialize an uninstalled OmniFold reward")
        reward_payload = self._reward.serializable_payload()
        observed = payload_sha256(reward_payload)
        if observed != self._stack_sha256:
            raise RuntimeError("installed OmniFold stack changed after installation")
        return {
            "schema_version": 1,
            "kind": "ztautau_adaptive_omnifold_stack",
            "source_bundle_sha256": self._bundle_sha256,
            "reward_round_id": self._reward_round_id,
            "reference_kind": self._reference_kind,
            "policy_reference_sha256": self._policy_reference_sha256,
            "base_digest": self._base_digest,
            "stack_sha256": self._stack_sha256,
            "reward": reward_payload,
        }

    def replace_stack(
        self,
        new_stack: FrozenResidualRatioReward,
        *,
        round_id: int,
        reference_sha256: str,
        reference_kind: str = "state_dict_sha256",
    ) -> None:
        if int(round_id) <= self._reward_round_id:
            raise ValueError("adaptive OmniFold round ids must increase monotonically")
        if len(str(reference_sha256)) != 64:
            raise ValueError("adaptive OmniFold reference needs a SHA256 digest")
        if str(reference_kind) != "state_dict_sha256":
            raise ValueError("dynamic OmniFold rounds require a state-dict reference")
        new_stack.to(self._device).eval()
        new_stack.assert_frozen()
        serialized = new_stack.serializable_payload()
        base_digest = str(serialized.get("base_digest", ""))
        if base_digest != self._base_digest:
            raise ValueError("adaptive OmniFold stack uses a different EveNet backbone")
        self._reward = new_stack
        self._packing_spec = new_stack.packing_spec
        self._stack_sha256 = payload_sha256(serialized)
        self._policy_reference_sha256 = str(reference_sha256)
        self._reference_kind = str(reference_kind)
        self._reward_round_id = int(round_id)

    def load_stack_payload(
        self,
        payload: Mapping[str, Any],
        *,
        allow_source_bundle_migration: bool = False,
    ) -> None:
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported adaptive OmniFold stack checkpoint schema")
        if str(payload.get("kind", "")) != "ztautau_adaptive_omnifold_stack":
            raise ValueError("checkpoint does not contain a Ztautau adaptive stack")
        saved_source_bundle = str(payload.get("source_bundle_sha256", ""))
        source_bundle_changed = saved_source_bundle != self._bundle_sha256
        if source_bundle_changed:
            # A dynamic in-process stack is self-describing and is protected by
            # its own SHA256 plus the frozen EveNet base digest below.  Permit a
            # configuration-fingerprint migration only when the caller has
            # explicitly scheduled a versioned one-shot resume refit.  Never
            # apply this escape hatch to a standalone reward artifact.
            if not allow_source_bundle_migration or self._artifact_path is not None:
                raise ValueError(
                    "adaptive stack was initialized from a different reward bundle"
                )
            _log.warning(
                "[DGPO/omnifold] migrating in-process reward source fingerprint "
                "%s -> %s; saved stack/base digests remain fail-closed",
                saved_source_bundle[:12] or "<missing>",
                self._bundle_sha256[:12],
            )
        reward_payload = payload.get("reward")
        if not isinstance(reward_payload, Mapping):
            raise ValueError("adaptive OmniFold checkpoint is missing its ratio stack")
        observed = payload_sha256(reward_payload)
        if observed != str(payload.get("stack_sha256", "")):
            raise ValueError("adaptive OmniFold checkpoint stack digest is invalid")
        if str(reward_payload.get("base_digest", "")) != self._base_digest:
            raise ValueError("adaptive OmniFold checkpoint uses a different EveNet body")

        increments = reward_payload.get("increments")
        if not isinstance(increments, (list, tuple)) or not increments:
            raise ValueError("adaptive OmniFold checkpoint has no ratio increments")
        bank_names: list[str | None] = []
        for increment in increments:
            if not isinstance(increment, Mapping):
                raise ValueError("adaptive OmniFold checkpoint has an invalid increment")
            raw_name = increment.get("bank_name")
            if raw_name is not None and not isinstance(raw_name, str):
                raise ValueError("adaptive OmniFold checkpoint has an invalid bank name")
            bank_names.append(raw_name)
        if any(name != bank_names[0] for name in bank_names[1:]):
            raise ValueError(
                "adaptive OmniFold checkpoint increments use different bank names"
            )
        restored_bank_name = bank_names[0]
        first_increment = dict(increments[0])
        saved_classifier_config = dict(
            first_increment.get("classifier_config") or {}
        )
        if not saved_classifier_config:
            # Checkpoints written before classifier architecture metadata can
            # still be resumed after shrinking the new audit/refit head. Infer
            # tensor-determined dimensions and use the pretrained EveNet's
            # attention-head default, which matches the legacy 256x2x8 head.
            saved_state = dict(first_increment.get("state") or {})
            norm_weight = saved_state.get("decoder.output_norm.weight")
            if norm_weight is None:
                raise ValueError(
                    "legacy adaptive OmniFold checkpoint has no decoder shape"
                )
            layer_ids = {
                int(str(key).split(".")[2])
                for key in saved_state
                if str(key).startswith("decoder.blocks.")
                and len(str(key).split(".")) > 3
                and str(key).split(".")[2].isdigit()
            }
            adapter_down = next(
                (
                    value
                    for key, value in saved_state.items()
                    if str(key).startswith("pet_adapters.")
                    and str(key).endswith(".down.weight")
                ),
                None,
            )
            saved_classifier_config = {
                "decoder_hidden_dim": int(norm_weight.numel()),
                "decoder_layers": max(layer_ids, default=-1) + 1,
                "decoder_heads": int(
                    getattr(
                        self.model_builder.backbone.network_cfg.Classification,
                        "num_attention_heads",
                        1,
                    )
                ),
                "adapter_bottleneck": (
                    self.model_builder._adapter_bottleneck
                    if adapter_down is None
                    else int(adapter_down.shape[0])
                ),
            }

        def _classifier_factory(spec: EventPackingSpec):
            return self.model_builder.make_classifier(
                spec,
                # ``bank_name`` is provenance metadata included in the stack
                # digest. Preserve it exactly across resume; changing only this
                # label must not masquerade as a frozen-weight mutation.
                name=restored_bank_name,
                reset=True,
                head_dropout=saved_classifier_config.get("head_dropout"),
                decoder_hidden_dim=int(
                    saved_classifier_config["decoder_hidden_dim"]
                ),
                decoder_layers=int(saved_classifier_config["decoder_layers"]),
                decoder_heads=int(saved_classifier_config["decoder_heads"]),
                adapter_bottleneck=int(
                    saved_classifier_config["adapter_bottleneck"]
                ),
            )

        restored = FrozenResidualRatioReward.from_serializable_payload(
            reward_payload,
            model_builder=_classifier_factory,
            device=self._device,
        )
        restored_digest = payload_sha256(restored.serializable_payload())
        if restored_digest != observed:
            raise ValueError(
                "adaptive OmniFold checkpoint stack changed during restore"
            )
        round_id = int(payload.get("reward_round_id", -1))
        reference = str(payload.get("policy_reference_sha256", ""))
        reference_kind = str(payload.get("reference_kind", ""))
        if round_id < 0 or len(reference) != 64:
            raise ValueError("adaptive OmniFold checkpoint has invalid round provenance")
        if round_id > 0 and reference_kind != "state_dict_sha256":
            raise ValueError("dynamic OmniFold checkpoint lacks a state-dict reference")
        self._reward = restored
        self._packing_spec = restored.packing_spec
        self._stack_sha256 = observed
        self._policy_reference_sha256 = reference
        self._reference_kind = reference_kind
        self._reward_round_id = round_id

    @torch.no_grad()
    def compute(
        self,
        candidates: Tensor,
        batch: dict[str, Any],
        mask: Tensor | None = None,
    ) -> Tensor:
        if self._reward is None or self._packing_spec is None:
            raise RuntimeError(
                "DGPO attempted to score before the in-process OmniFold bootstrap"
            )
        if candidates.ndim != 4:
            raise ValueError(
                "OmniFold DGPO candidates must be (K,B,N,F), got "
                f"{tuple(candidates.shape)}"
            )
        k, batch_size, num_slots, feature_dim = map(int, candidates.shape)
        if num_slots != 2 or feature_dim != 2:
            raise ValueError(
                "Ztautau OmniFold expects two (delta_theta, delta_phi) slots; "
                f"got {tuple(candidates.shape)}"
            )
        packed_event, observed = pack_event_inputs(batch, self._packing_spec)
        if observed != self._packing_spec:
            raise RuntimeError("OmniFold event packing changed after validation")
        candidate_bk4 = (
            candidates.permute(1, 0, 2, 3)
            .reshape(batch_size, k, num_slots * feature_dim)
            .to(device=self._device, dtype=torch.float32)
        )
        scores_bk = self._reward(
            packed_event.to(device=self._device, dtype=torch.float32),
            candidate_bk4,
        )
        if tuple(scores_bk.shape) != (batch_size, k):
            raise RuntimeError(
                f"OmniFold ratio stack returned {tuple(scores_bk.shape)}, "
                f"expected {(batch_size, k)}"
            )
        scores_kb = scores_bk.transpose(0, 1).contiguous()
        if not bool(torch.isfinite(scores_kb).all().item()):
            raise FloatingPointError("OmniFold reward produced NaN or Inf")
        if mask is not None:
            scores_kb = scores_kb * mask.to(
                device=scores_kb.device,
                dtype=scores_kb.dtype,
            )
        return apply_event_valid_to_rewards(scores_kb, batch)


def build_uninstalled_ztautau_omnifold_reward(
    *,
    backbone_checkpoint: str | Path,
    training_config: Any,
    normalization_dict: dict[str, Any],
    device: torch.device,
    classifier_config: Mapping[str, Any],
) -> ZtautauOmniFoldReward:
    """Construct the classifier factory; DGPO installs the first stack itself."""

    backbone_path = Path(backbone_checkpoint).expanduser().resolve()
    if not backbone_path.is_file():
        raise FileNotFoundError(
            f"OmniFold bootstrap backbone checkpoint not found: {backbone_path}"
        )
    train_backbone = bool(classifier_config.get("train_backbone", False))
    builder = EvenetAdapterModelBuilder(
        config=training_config,
        normalization_dict=normalization_dict,
        checkpoint_path=backbone_path,
        device=device,
        adapter_bottleneck=int(classifier_config.get("adapter_bottleneck", 16)),
        train_layernorm=bool(classifier_config.get("train_layernorm", False)),
        train_encoder=bool(classifier_config.get("train_encoder", False)),
        train_backbone=train_backbone,
        head_dropout=float(classifier_config.get("head_dropout", 0.1)),
        decoder_hidden_dim=int(classifier_config.get("decoder_hidden_dim", 256)),
        decoder_layers=int(classifier_config.get("decoder_layers", 2)),
        decoder_heads=int(classifier_config.get("decoder_heads", 8)),
    )
    policy_reference = sha256_file(backbone_path)
    # This identifies the immutable denominator/body pair.  Individual ratio
    # increments already serialize their exact classifier architecture and
    # weights, so optimizer/head choices do not belong in the bundle identity.
    # A changed classifier is installed through a versioned adaptive refit.
    source_identity = {
        "schema_version": 2,
        "kind": "ztautau_in_dgpo_omnifold_bootstrap",
        "base_digest": builder.base_digest,
        "policy_reference_sha256": policy_reference,
    }
    return ZtautauOmniFoldReward(
        None,
        bundle_sha256=payload_sha256(source_identity),
        policy_reference_sha256=policy_reference,
        base_digest=builder.base_digest,
        stack_sha256="",
        bundle_schema_version=1,
        device=device,
        artifact_path=None,
        model_builder=builder,
        reward_round_id=0,
        reference_kind="checkpoint_sha256",
    )


def load_ztautau_omnifold_reward(
    *,
    bundle_file: str | Path,
    backbone_checkpoint: str | Path,
    training_config: Any,
    normalization_dict: dict[str, Any],
    device: torch.device,
    expected_iterations: int | None = None,
) -> ZtautauOmniFoldReward:
    """Restore the standalone fit artifact on a frozen EveNet PEFT backbone."""

    bundle_path = Path(bundle_file).expanduser().resolve()
    backbone_path = Path(backbone_checkpoint).expanduser().resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"OmniFold reward bundle not found: {bundle_path}")
    if not backbone_path.is_file():
        raise FileNotFoundError(
            f"OmniFold frozen backbone checkpoint not found: {backbone_path}"
        )
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"unsupported Ztautau OmniFold bundle schema: {bundle_path}")
    if str(payload.get("kind", "")) != "ztautau_evenet_omnifold_reward":
        raise ValueError(f"artifact is not a Ztautau OmniFold reward: {bundle_path}")
    if int(payload.get("candidates_per_event_for_fit", -1)) != 1:
        raise ValueError("DGPO may consume only an OmniFold reward fitted with K=1")

    classifier_cfg = payload.get("classifier")
    if not isinstance(classifier_cfg, Mapping):
        raise ValueError(
            "OmniFold bundle is missing classifier architecture metadata; refit it "
            "with the current standalone stage"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("OmniFold bundle is missing provenance")
    policy_reference = str(provenance.get("policy_reference_sha256", ""))
    if len(policy_reference) != 64:
        raise ValueError(
            "OmniFold bundle has no valid denominator-policy SHA256; rebuild its K=1 pools"
        )

    reward_payload = payload.get("reward")
    if not isinstance(reward_payload, Mapping):
        raise ValueError("OmniFold bundle is missing the serialized ratio stack")
    base_digest = str(reward_payload.get("base_digest", ""))
    if not base_digest:
        raise ValueError("OmniFold ratio stack is missing its frozen-backbone digest")

    builder = EvenetAdapterModelBuilder(
        config=training_config,
        normalization_dict=normalization_dict,
        checkpoint_path=backbone_path,
        device=device,
        adapter_bottleneck=int(classifier_cfg["adapter_bottleneck"]),
        train_layernorm=bool(classifier_cfg.get("train_layernorm", False)),
        train_encoder=bool(classifier_cfg.get("train_encoder", False)),
        train_backbone=bool(classifier_cfg.get("train_backbone", False)),
        head_dropout=float(classifier_cfg["head_dropout"]),
        decoder_hidden_dim=int(classifier_cfg["decoder_hidden_dim"]),
        decoder_layers=int(classifier_cfg["decoder_layers"]),
        decoder_heads=int(classifier_cfg["decoder_heads"]),
    )

    def _classifier_factory(spec: EventPackingSpec):
        return builder.make_classifier(
            spec,
            name="installed_dgpo_reward",
            reset=True,
        )

    frozen = FrozenResidualRatioReward.from_serializable_payload(
        reward_payload,
        model_builder=_classifier_factory,
        device=device,
    )
    if expected_iterations is not None and frozen.num_iterations != int(
        expected_iterations
    ):
        raise ValueError(
            f"OmniFold bundle has {frozen.num_iterations} iterations, expected "
            f"{int(expected_iterations)}"
        )
    return ZtautauOmniFoldReward(
        frozen,
        bundle_sha256=sha256_file(bundle_path),
        policy_reference_sha256=policy_reference,
        base_digest=base_digest,
        stack_sha256=payload_sha256(reward_payload),
        bundle_schema_version=int(payload["schema_version"]),
        device=device,
        artifact_path=bundle_path,
        model_builder=builder,
    )


def validate_omnifold_reward_startup(
    *,
    checkpoint: Mapping[str, Any] | None,
    current_metadata: Mapping[str, Any] | None,
    policy_checkpoint: str | Path | None,
    allow_source_bundle_migration: bool = False,
) -> None:
    """Fail closed on a reward/reference mismatch at cold start or resume."""

    if current_metadata is None:
        return
    is_resume = bool(
        checkpoint is not None
        and int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1
    )
    if is_resume:
        saved = checkpoint.get(REWARD_CHECKPOINT_KEY) if checkpoint is not None else None
        if saved is None:
            raise ValueError(
                "DGPO resume checkpoint predates OmniFold reward provenance; refusing "
                "to pair it with a newly supplied density-ratio reward"
            )
        metadata_match = dict(saved) == dict(current_metadata)
        if not metadata_match and allow_source_bundle_migration:
            # The one-shot migration may change only the in-process source
            # fingerprint.  Weight, stack digest, base digest, round, reference,
            # and every other reward source must remain byte-for-byte equal.
            normalized_current = copy.deepcopy(dict(current_metadata))
            saved_sources = list(dict(saved).get("sources", []))
            current_sources = list(normalized_current.get("sources", []))
            if len(saved_sources) == len(current_sources):
                for saved_source, current_source in zip(
                    saved_sources, current_sources, strict=True
                ):
                    saved_meta = (
                        saved_source.get("metadata")
                        if isinstance(saved_source, Mapping)
                        else None
                    )
                    current_meta = (
                        current_source.get("metadata")
                        if isinstance(current_source, Mapping)
                        else None
                    )
                    if (
                        isinstance(saved_meta, Mapping)
                        and isinstance(current_meta, dict)
                        and saved_meta.get("kind")
                        == "ztautau_evenet_omnifold_reward"
                        and current_meta.get("kind")
                        == "ztautau_evenet_omnifold_reward"
                    ):
                        current_meta["bundle_sha256"] = saved_meta.get(
                            "bundle_sha256"
                        )
            metadata_match = dict(saved) == normalized_current
        if not metadata_match:
            raise ValueError(
                "DGPO resume checkpoint was trained with a different OmniFold reward "
                "bundle, weight, or source configuration"
            )
        return

    omnifold_metadata = [
        source["metadata"]
        for source in current_metadata.get("sources", [])
        if isinstance(source, Mapping)
        and isinstance(source.get("metadata"), Mapping)
        and source["metadata"].get("kind") == "ztautau_evenet_omnifold_reward"
    ]
    # An in-process bootstrap source has only a frozen classifier backbone at
    # this point; it has no ratio stack or denominator policy yet. The fresh
    # current policy snapshot becomes the denominator when the initial stack is
    # successfully installed. Therefore its backbone checkpoint hash must not
    # be compared with a weights-only warm-start policy checkpoint.
    uninstalled_bootstrap = bool(omnifold_metadata) and all(
        int(metadata.get("reward_round_id", 0)) == 0
        and int(metadata.get("iterations", 0)) == 0
        and not str(metadata.get("stack_sha256", ""))
        for metadata in omnifold_metadata
    )
    if uninstalled_bootstrap:
        return

    references = {
        str(metadata.get("policy_reference_sha256", ""))
        for metadata in omnifold_metadata
    }
    references.discard("")
    if len(references) != 1:
        raise ValueError("OmniFold reward metadata needs exactly one denominator policy")
    if policy_checkpoint is None:
        raise ValueError("OmniFold-guided DGPO requires a cold-start policy checkpoint")
    actual = sha256_file(policy_checkpoint)
    expected = next(iter(references))
    if actual != expected:
        raise ValueError(
            "OmniFold denominator policy does not match the DGPO cold-start checkpoint "
            f"({expected[:12]} != {actual[:12]})"
        )


__all__ = [
    "REWARD_CHECKPOINT_KEY",
    "REWARD_STACK_CHECKPOINT_KEY",
    "ZtautauOmniFoldReward",
    "build_uninstalled_ztautau_omnifold_reward",
    "load_ztautau_omnifold_reward",
    "payload_sha256",
    "sha256_file",
    "validate_omnifold_reward_startup",
]
