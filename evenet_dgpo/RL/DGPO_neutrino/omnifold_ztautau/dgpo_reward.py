"""Frozen Ztautau OmniFold reward adapter for the DGPO candidate interface."""

from __future__ import annotations

import copy
import hashlib
import logging
import math
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
from RL.DGPO_neutrino.reward_interface import (
    consensus_gated_advantage,
    member_ordering_metrics,
    paired_ordering_metrics,
)


REWARD_CHECKPOINT_KEY = "dgpo_omnifold_reward_metadata"
REWARD_STACK_CHECKPOINT_KEY = "dgpo_omnifold_reward_stack"

_log = logging.getLogger(__name__)


_CONSENSUS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "apply_to_reward": False,
    "temporal_history_rounds": 0,
    "minimum_sign_agreement": 0.75,
    "uncertainty_scale": 1.0,
    "epsilon": 1.0e-8,
    "fixed_panel_events_per_rank": 128,
    "fixed_panel_candidates": 8,
}


def _candidate_consensus_config(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    config = dict(_CONSENSUS_DEFAULTS)
    if raw is not None:
        if not isinstance(raw, Mapping) and hasattr(raw, "__dict__"):
            raw = vars(raw)
        if not isinstance(raw, Mapping):
            raise TypeError("candidate_consensus must be a mapping")
        unknown = sorted(set(raw) - set(config))
        if unknown:
            raise ValueError(
                "unknown candidate_consensus settings: " + ", ".join(unknown)
            )
        config.update(dict(raw))
    for key in ("enabled", "apply_to_reward"):
        if type(config[key]) is not bool:
            raise TypeError(f"candidate_consensus.{key} must be boolean")
    history = config["temporal_history_rounds"]
    panel = config["fixed_panel_events_per_rank"]
    panel_candidates = config["fixed_panel_candidates"]
    if type(history) is not int or not 0 <= history <= 4:
        raise ValueError(
            "candidate_consensus.temporal_history_rounds must be in [0,4]"
        )
    if type(panel) is not int or panel < 1:
        raise ValueError(
            "candidate_consensus.fixed_panel_events_per_rank must be positive"
        )
    if type(panel_candidates) is not int or panel_candidates < 2:
        raise ValueError(
            "candidate_consensus.fixed_panel_candidates must be at least two"
        )
    agreement = float(config["minimum_sign_agreement"])
    scale = float(config["uncertainty_scale"])
    epsilon = float(config["epsilon"])
    if not 0.5 <= agreement <= 1.0:
        raise ValueError(
            "candidate_consensus.minimum_sign_agreement must lie in [0.5,1]"
        )
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError(
            "candidate_consensus.uncertainty_scale must be finite and nonnegative"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("candidate_consensus.epsilon must be finite and positive")
    if config["apply_to_reward"] and not config["enabled"]:
        raise ValueError("candidate consensus cannot be applied when disabled")
    config.update(
        minimum_sign_agreement=agreement,
        uncertainty_scale=scale,
        epsilon=epsilon,
    )
    return config


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
        candidate_consensus: Mapping[str, Any] | None = None,
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
        self._candidate_consensus = _candidate_consensus_config(
            candidate_consensus
        )
        self._consensus_history: list[
            tuple[int, FrozenResidualRatioReward]
        ] = []
        self._rank_audit_panel: tuple[Tensor, Tensor] | None = None
        self._rank_audit_history: list[dict[str, float]] = []
        self._last_rank_audit_round: int | None = None
        self._last_interface_metrics: dict[str, float] = {}
        if frozen_reward is not None:
            self._assert_candidate_consensus_compatible(frozen_reward)

    def make_endpoint_kl_classifier(self):
        """Independent H4 critic initialized from fold 1, with a neutral logit.

        The new labels will be current=1/reference=0. Never train the frozen
        truth-vs-reference reward or share its trainable tensors.
        """
        if self._model_builder is None or self._reward is None:
            raise RuntimeError("endpoint KL requires an installed EveNet OmniFold reward")
        classifier = self._model_builder.make_classifier(self._packing_spec)
        classifier.load_state_dict(self._reward._checkpoints[0], strict=True)
        with torch.no_grad():
            classifier.bank.output.weight.zero_()
            classifier.bank.output.bias.zero_()
            if hasattr(classifier.bank, "topology_output"):
                classifier.bank.topology_output.weight.zero_()
                classifier.bank.topology_output.bias.zero_()
        classifier.endpoint_candidate_grad = True
        return classifier

    def _assert_candidate_consensus_compatible(
        self, reward: FrozenResidualRatioReward
    ) -> None:
        if not self._candidate_consensus["enabled"]:
            return
        if int(reward.num_iterations) != 1:
            raise ValueError(
                "candidate consensus requires an iteration-one-only reward"
            )
        if int(reward.num_checkpoints) < 2:
            raise ValueError(
                "candidate consensus requires at least two reward members"
            )
        if getattr(reward, "log_ratio_clip", None) is not None:
            raise ValueError("candidate consensus requires unclipped reward logits")

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
        metadata = {
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
        if self._candidate_consensus["enabled"]:
            metadata["candidate_consensus"] = copy.deepcopy(
                self._candidate_consensus
            )
        return metadata

    def stack_payload(self) -> dict[str, Any]:
        if self._reward is None:
            raise RuntimeError("cannot serialize an uninstalled OmniFold reward")
        reward_payload = self._reward.serializable_payload()
        observed = payload_sha256(reward_payload)
        if observed != self._stack_sha256:
            raise RuntimeError("installed OmniFold stack changed after installation")
        payload = {
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
        if self._candidate_consensus["enabled"]:
            payload["candidate_consensus"] = copy.deepcopy(
                self._candidate_consensus
            )
            history_payloads: list[dict[str, Any]] = []
            for round_id, stack in self._consensus_history:
                serialized_history = stack.serializable_payload()
                history_payloads.append(
                    {
                        "reward_round_id": int(round_id),
                        "stack_sha256": payload_sha256(serialized_history),
                        "reward": serialized_history,
                    }
                )
            payload["consensus_history"] = history_payloads
            if self._rank_audit_panel is not None:
                condition, candidates = self._rank_audit_panel
                payload["rank_audit_panel"] = {
                    "condition": condition.detach().cpu().clone(),
                    "candidates": candidates.detach().cpu().clone(),
                }
            payload["rank_audit_history"] = copy.deepcopy(
                self._rank_audit_history
            )
        return payload

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
        self._assert_candidate_consensus_compatible(new_stack)
        serialized = new_stack.serializable_payload()
        base_digest = str(serialized.get("base_digest", ""))
        if base_digest != self._base_digest:
            raise ValueError("adaptive OmniFold stack uses a different EveNet backbone")
        if (
            self._candidate_consensus["enabled"]
            and self._reward is not None
            and int(self._candidate_consensus["temporal_history_rounds"]) > 0
        ):
            # Only the frozen member checkpoints are needed for temporal
            # scoring. The CPU warm-start cache belongs to classifier fitting
            # and would otherwise inflate every recovery checkpoint.
            if self._reward is not new_stack:
                self._reward.warm_start_state = None
            self._consensus_history.append(
                (int(self._reward_round_id), self._reward)
            )
            keep = int(self._candidate_consensus["temporal_history_rounds"])
            self._consensus_history = self._consensus_history[-keep:]
        self._reward = new_stack
        self._packing_spec = new_stack.packing_spec
        self._stack_sha256 = payload_sha256(serialized)
        self._policy_reference_sha256 = str(reference_sha256)
        self._reference_kind = str(reference_kind)
        self._reward_round_id = int(round_id)
        self._last_rank_audit_round = None

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
            raise ValueError(
                "adaptive OmniFold checkpoint predates the internal-adapter "
                "classifier schema; restart from the diffusion checkpoint"
            )
        if saved_classifier_config.get("adapter_placement") != "internal":
            raise ValueError(
                "external-adapter OmniFold rewards are no longer supported; "
                "restart from the diffusion checkpoint"
            )

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
                train_grouped_sequential_embedding=bool(
                    saved_classifier_config.get(
                        "train_grouped_sequential_embedding", False
                    )
                ),
                # Legacy reward stacks did not train/save this projector. Keep
                # their exact frozen-body semantics during restore so their
                # integrity digest remains stable; newly fitted stacks persist
                # the flag and the projector tensors in ``body``.
                train_invisible_projector=bool(
                    saved_classifier_config.get(
                        "train_invisible_projector", False
                    )
                ),
                train_angular_conditioning=bool(
                    saved_classifier_config.get("train_angular_conditioning", False)
                ),
                # Same for LN / GlobalEmbedding: a fork may open them for the
                # upcoming refit_once, but restore must keep the parent body
                # key set or ``load_state_dict`` fails on missing body tensors.
                train_layernorm=bool(
                    saved_classifier_config.get("train_layernorm", False)
                ),
                train_encoder=bool(
                    saved_classifier_config.get("train_encoder", False)
                ),
                train_backbone=bool(
                    saved_classifier_config.get("train_backbone", False)
                ),
                train_last_pet_block=bool(
                    saved_classifier_config.get("train_last_pet_block", False)
                ),
                # Payloads written before this field used the asymmetric mask.
                # New fits persist ``False`` and use full self-attention.
                asymmetric_attention=bool(
                    saved_classifier_config.get("asymmetric_attention", True)
                ),
                periodic_pair_features=bool(
                    saved_classifier_config.get("periodic_pair_features", False)
                ),
                topology_fourier_embedding=bool(
                    saved_classifier_config.get("topology_fourier_embedding", False)
                ),
                topology_direct_logit=bool(
                    saved_classifier_config.get("topology_direct_logit", False)
                ),
                topology_context_residual_scale=float(
                    saved_classifier_config.get(
                        "topology_context_residual_scale", 1.0
                    )
                ),
                conditional_residual_rank=int(
                    saved_classifier_config.get("conditional_residual_rank", 0)
                ),
                topology_conditioning=bool(saved_classifier_config.get("topology_conditioning", False)),
                topology_pair_token=bool(saved_classifier_config.get("topology_pair_token", False)),
                relation_token_count=int(saved_classifier_config.get("relation_token_count", 0)),
                visible_pair_rest_frame=bool(saved_classifier_config.get("visible_pair_rest_frame", False)),
                topology_max_harmonic=int(
                    saved_classifier_config.get("topology_max_harmonic", 1)
                ),
                topology_include_theta_pair=bool(
                    saved_classifier_config.get("topology_include_theta_pair", False)
                ),
                topology_theta_fourier=bool(
                    saved_classifier_config.get("topology_theta_fourier", False)
                ),
                topology_hidden_dim=int(
                    saved_classifier_config.get("topology_hidden_dim", 64)
                ),
                topology_embedding_dim=int(
                    saved_classifier_config.get("topology_embedding_dim", 32)
                ),
                topology_fusion_hidden_dim=int(
                    saved_classifier_config.get("topology_fusion_hidden_dim", 64)
                ),
                topology_dropout=float(
                    saved_classifier_config.get("topology_dropout", 0.15)
                ),
            )

        restored = FrozenResidualRatioReward.from_serializable_payload(
            reward_payload,
            model_builder=_classifier_factory,
            device=self._device,
        )
        self._assert_candidate_consensus_compatible(restored)
        restored_digest = payload_sha256(restored.serializable_payload())
        if restored_digest != observed:
            raise ValueError(
                "adaptive OmniFold checkpoint stack changed during restore"
            )

        saved_consensus_raw = payload.get("candidate_consensus")
        saved_consensus = _candidate_consensus_config(saved_consensus_raw)
        if saved_consensus != self._candidate_consensus:
            raise ValueError(
                "adaptive OmniFold checkpoint candidate-consensus protocol changed"
            )
        restored_history: list[tuple[int, FrozenResidualRatioReward]] = []
        if self._candidate_consensus["enabled"]:
            history_rows = payload.get("consensus_history", [])
            if not isinstance(history_rows, (list, tuple)):
                raise ValueError("candidate-consensus history must be a sequence")
            keep = int(self._candidate_consensus["temporal_history_rounds"])
            if len(history_rows) > keep:
                raise ValueError("candidate-consensus history exceeds configured length")
            for row in history_rows:
                if not isinstance(row, Mapping) or not isinstance(
                    row.get("reward"), Mapping
                ):
                    raise ValueError("candidate-consensus history row is invalid")
                history_payload = row["reward"]
                history_digest = payload_sha256(history_payload)
                if history_digest != str(row.get("stack_sha256", "")):
                    raise ValueError(
                        "candidate-consensus history stack digest is invalid"
                    )
                if str(history_payload.get("base_digest", "")) != self._base_digest:
                    raise ValueError(
                        "candidate-consensus history uses a different EveNet body"
                    )
                history_stack = FrozenResidualRatioReward.from_serializable_payload(
                    history_payload,
                    model_builder=_classifier_factory,
                    device=self._device,
                )
                history_stack.assert_frozen()
                restored_history.append(
                    (int(row.get("reward_round_id", -1)), history_stack)
                )
            history_rounds = [round_id for round_id, _stack in restored_history]
            if history_rounds != sorted(history_rounds) or any(
                round_id < 1 for round_id in history_rounds
            ):
                raise ValueError(
                    "candidate-consensus history round ids are invalid"
                )

            panel = payload.get("rank_audit_panel")
            if panel is not None:
                if not isinstance(panel, Mapping):
                    raise ValueError("rank-audit panel is invalid")
                condition = panel.get("condition")
                candidates = panel.get("candidates")
                if not isinstance(condition, Tensor) or not isinstance(
                    candidates, Tensor
                ):
                    raise ValueError("rank-audit panel tensors are missing")
                if condition.ndim != 2 or candidates.ndim != 3:
                    raise ValueError("rank-audit panel tensor shapes are invalid")
                if int(condition.shape[0]) != int(candidates.shape[0]):
                    raise ValueError("rank-audit panel event counts differ")
                self._rank_audit_panel = (
                    condition.detach().cpu().clone(),
                    candidates.detach().cpu().clone(),
                )
            audit_history = payload.get("rank_audit_history", [])
            if not isinstance(audit_history, (list, tuple)):
                raise ValueError("rank-audit metric history must be a sequence")
            self._rank_audit_history = [
                {str(key): float(value) for key, value in dict(row).items()}
                for row in audit_history
            ]
            self._last_rank_audit_round = (
                int(self._rank_audit_history[-1]["reward_round_id"])
                if self._rank_audit_history
                else None
            )
        round_id = int(payload.get("reward_round_id", -1))
        reference = str(payload.get("policy_reference_sha256", ""))
        reference_kind = str(payload.get("reference_kind", ""))
        if round_id < 0 or len(reference) != 64:
            raise ValueError("adaptive OmniFold checkpoint has invalid round provenance")
        if round_id > 0 and reference_kind != "state_dict_sha256":
            raise ValueError("dynamic OmniFold checkpoint lacks a state-dict reference")
        self._reward = restored
        self._consensus_history = restored_history
        self._packing_spec = restored.packing_spec
        self._stack_sha256 = observed
        self._policy_reference_sha256 = reference
        self._reference_kind = reference_kind
        self._reward_round_id = round_id

    @property
    def candidate_consensus_config(self) -> dict[str, Any]:
        return copy.deepcopy(self._candidate_consensus)

    def last_interface_metrics(self) -> dict[str, float]:
        return dict(self._last_interface_metrics)

    @property
    def rank_audit_history(self) -> list[dict[str, float]]:
        return copy.deepcopy(self._rank_audit_history)

    @torch.no_grad()
    def _score_members(
        self,
        reward: FrozenResidualRatioReward,
        condition: Tensor,
        candidates_bk4: Tensor,
    ) -> tuple[Tensor, Tensor]:
        members_mbk = reward.member_logits(condition, candidates_bk4)
        ensemble_bk = reward.ensemble_from_member_logits(members_mbk)
        return members_mbk.permute(0, 2, 1).contiguous(), ensemble_bk.transpose(
            0, 1
        ).contiguous()

    @torch.no_grad()
    def _maybe_run_fixed_rank_audit(
        self,
        packed_event: Tensor,
        candidates_bk4: Tensor,
        *,
        policy_update: bool,
    ) -> dict[str, float]:
        if not self._candidate_consensus["enabled"] or not policy_update:
            return {}
        if int(candidates_bk4.shape[1]) != int(
            self._candidate_consensus["fixed_panel_candidates"]
        ):
            # validation_K may differ from the DGPO training K. The diagnostic
            # panel must be captured from, and compared on, the exact candidate
            # set that supplies the policy gradient.
            return {}
        if self._rank_audit_panel is None:
            count = min(
                int(packed_event.shape[0]),
                int(self._candidate_consensus["fixed_panel_events_per_rank"]),
            )
            self._rank_audit_panel = (
                packed_event[:count].detach().cpu().clone(),
                candidates_bk4[:count].detach().cpu().clone(),
            )
        if self._last_rank_audit_round == self._reward_round_id:
            if not self._rank_audit_history:
                return {}
            latest = self._rank_audit_history[-1]
            return {
                f"reward_rank_audit/fixed_panel/{key}": float(value)
                for key, value in latest.items()
            }

        assert self._reward is not None
        panel_condition_cpu, panel_candidates_cpu = self._rank_audit_panel
        panel_condition = panel_condition_cpu.to(
            device=self._device, dtype=torch.float32
        )
        panel_candidates = panel_candidates_cpu.to(
            device=self._device, dtype=torch.float32
        )
        current_members, current_ensemble = self._score_members(
            self._reward, panel_condition, panel_candidates
        )
        current_metrics = member_ordering_metrics(current_members)
        row: dict[str, float] = {
            "reward_round_id": float(self._reward_round_id),
            "previous_reward_round_id": 0.0,
            "temporal_available": 0.0,
            "panel_events_per_rank": float(panel_condition.shape[0]),
            "candidates_per_event": float(panel_candidates.shape[1]),
            **{
                f"current_member/{key}": float(value)
                for key, value in current_metrics.items()
            },
        }
        if self._consensus_history:
            previous_round, previous_reward = self._consensus_history[-1]
            previous_members, previous_ensemble = self._score_members(
                previous_reward, panel_condition, panel_candidates
            )
            temporal = paired_ordering_metrics(
                current_ensemble, previous_ensemble
            )
            pooled = member_ordering_metrics(
                torch.cat((previous_members, current_members), dim=0)
            )
            row.update(
                previous_reward_round_id=float(previous_round),
                temporal_available=1.0,
                **{
                    f"temporal/{key}": float(value)
                    for key, value in temporal.items()
                },
                **{
                    f"pooled_member/{key}": float(value)
                    for key, value in pooled.items()
                },
            )
        self._rank_audit_history.append(row)
        self._last_rank_audit_round = int(self._reward_round_id)
        return {
            f"reward_rank_audit/fixed_panel/{key}": float(value)
            for key, value in row.items()
        }

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
        packed_event_device = packed_event.to(
            device=self._device, dtype=torch.float32
        )
        consensus_metrics = self._maybe_run_fixed_rank_audit(
            packed_event_device,
            candidate_bk4,
            policy_update=(
                batch.get("_dgpo_reward_context") == "policy_update"
            ),
        )
        if self._candidate_consensus["apply_to_reward"]:
            current_members, scores_kb = self._score_members(
                self._reward, packed_event_device, candidate_bk4
            )
            pooled_members = [current_members]
            for _round_id, previous_reward in self._consensus_history:
                previous_members, _previous_ensemble = self._score_members(
                    previous_reward, packed_event_device, candidate_bk4
                )
                pooled_members.insert(0, previous_members)
            gated_advantage, live_metrics = consensus_gated_advantage(
                scores_kb,
                torch.cat(pooled_members, dim=0),
                minimum_sign_agreement=float(
                    self._candidate_consensus["minimum_sign_agreement"]
                ),
                uncertainty_scale=float(
                    self._candidate_consensus["uncertainty_scale"]
                ),
                epsilon=float(self._candidate_consensus["epsilon"]),
            )
            # For centered scores s, leave-one-out(s) = K/(K-1) s. Encode the
            # desired consensus advantage so the trainer's existing estimator
            # recovers it exactly.
            scores_kb = ((k - 1.0) / k) * gated_advantage
            scores_bk = scores_kb.transpose(0, 1).contiguous()
            consensus_metrics.update(
                {
                    "reward_consensus/applied": 1.0,
                    "reward_consensus/current_members": float(
                        current_members.shape[0]
                    ),
                    "reward_consensus/temporal_rounds": float(
                        len(self._consensus_history)
                    ),
                    **{
                        f"reward_consensus/live/{key}": float(value)
                        for key, value in live_metrics.items()
                    },
                }
            )
        else:
            scores_bk = self._reward(packed_event_device, candidate_bk4)
            if self._candidate_consensus["enabled"]:
                consensus_metrics.update(
                    {
                        "reward_consensus/applied": 0.0,
                        "reward_consensus/current_members": float(
                            self._reward.num_checkpoints
                        ),
                        "reward_consensus/temporal_rounds": float(
                            len(self._consensus_history)
                        ),
                    }
                )
        self._last_interface_metrics = consensus_metrics
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
    candidate_consensus: Mapping[str, Any] | None = None,
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
        body_only_checkpoint=bool(
            classifier_config.get("body_only_checkpoint", False)
        ),
        train_layernorm=bool(classifier_config.get("train_layernorm", False)),
        train_encoder=bool(classifier_config.get("train_encoder", False)),
        train_grouped_sequential_embedding=bool(
            classifier_config.get(
                "train_grouped_sequential_embedding", False
            )
        ),
        train_invisible_projector=bool(
            classifier_config.get("train_invisible_projector", False)
        ),
        train_angular_conditioning=bool(classifier_config.get("train_angular_conditioning", False)),
        train_backbone=train_backbone,
        train_last_pet_block=bool(classifier_config.get("train_last_pet_block", False)),
        asymmetric_attention=bool(
            classifier_config.get("asymmetric_attention", False)
        ),
        periodic_pair_features=bool(
            classifier_config.get("periodic_pair_features", False)
        ),
        topology_fourier_embedding=bool(
            classifier_config.get("topology_fourier_embedding", False)
        ),
        topology_direct_logit=bool(
            classifier_config.get("topology_direct_logit", False)
        ),
        topology_context_residual_scale=float(
            classifier_config.get("topology_context_residual_scale", 1.0)
        ),
        conditional_residual_rank=int(
            classifier_config.get("conditional_residual_rank", 0)
        ),
        topology_conditioning=bool(classifier_config.get("topology_conditioning", False)),
        topology_pair_token=bool(classifier_config.get("topology_pair_token", False)),
        relation_token_count=int(classifier_config.get("relation_token_count", 0)),
        visible_pair_rest_frame=bool(classifier_config.get("visible_pair_rest_frame", False)),
        topology_max_harmonic=int(
            classifier_config.get("topology_max_harmonic", 1)
        ),
        topology_include_theta_pair=bool(
            classifier_config.get("topology_include_theta_pair", False)
        ),
        topology_theta_fourier=bool(
            classifier_config.get("topology_theta_fourier", False)
        ),
        topology_hidden_dim=int(classifier_config.get("topology_hidden_dim", 64)),
        topology_embedding_dim=int(
            classifier_config.get("topology_embedding_dim", 32)
        ),
        topology_fusion_hidden_dim=int(
            classifier_config.get("topology_fusion_hidden_dim", 64)
        ),
        topology_dropout=float(classifier_config.get("topology_dropout", 0.15)),
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
    if builder.body_only_checkpoint:
        source_identity["checkpoint_scope"] = "body_only"
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
        candidate_consensus=candidate_consensus,
    )


def load_ztautau_omnifold_reward(
    *,
    bundle_file: str | Path,
    backbone_checkpoint: str | Path,
    training_config: Any,
    normalization_dict: dict[str, Any],
    device: torch.device,
    expected_iterations: int | None = None,
    candidate_consensus: Mapping[str, Any] | None = None,
) -> ZtautauOmniFoldReward:
    """Restore a standalone internal-adapter EveNet ratio artifact."""

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
    if classifier_cfg.get("adapter_placement") != "internal":
        raise ValueError(
            "standalone OmniFold artifact does not use the required internal "
            "PET adapters; refit it with the current stage"
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
        train_grouped_sequential_embedding=bool(
            classifier_cfg.get(
                "train_grouped_sequential_embedding", False
            )
        ),
        train_invisible_projector=bool(
            classifier_cfg.get("train_invisible_projector", False)
        ),
        train_angular_conditioning=bool(classifier_cfg.get("train_angular_conditioning", False)),
        train_backbone=bool(classifier_cfg.get("train_backbone", False)),
        train_last_pet_block=bool(classifier_cfg.get("train_last_pet_block", False)),
        # Standalone artifacts predating this field used the asymmetric mask.
        asymmetric_attention=bool(
            classifier_cfg.get("asymmetric_attention", True)
        ),
        periodic_pair_features=bool(
            classifier_cfg.get("periodic_pair_features", False)
        ),
        topology_fourier_embedding=bool(
            classifier_cfg.get("topology_fourier_embedding", False)
        ),
        topology_direct_logit=bool(
            classifier_cfg.get("topology_direct_logit", False)
        ),
        topology_context_residual_scale=float(
            classifier_cfg.get("topology_context_residual_scale", 1.0)
        ),
        conditional_residual_rank=int(
            classifier_cfg.get("conditional_residual_rank", 0)
        ),
        topology_conditioning=bool(classifier_cfg.get("topology_conditioning", False)),
        topology_pair_token=bool(classifier_cfg.get("topology_pair_token", False)),
        relation_token_count=int(classifier_cfg.get("relation_token_count", 0)),
        visible_pair_rest_frame=bool(classifier_cfg.get("visible_pair_rest_frame", False)),
        topology_max_harmonic=int(classifier_cfg.get("topology_max_harmonic", 1)),
        topology_include_theta_pair=bool(
            classifier_cfg.get("topology_include_theta_pair", False)
        ),
        topology_theta_fourier=bool(
            classifier_cfg.get("topology_theta_fourier", False)
        ),
        topology_hidden_dim=int(classifier_cfg.get("topology_hidden_dim", 64)),
        topology_embedding_dim=int(
            classifier_cfg.get("topology_embedding_dim", 32)
        ),
        topology_fusion_hidden_dim=int(
            classifier_cfg.get("topology_fusion_hidden_dim", 64)
        ),
        topology_dropout=float(classifier_cfg.get("topology_dropout", 0.15)),
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
        candidate_consensus=candidate_consensus,
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

    tau_metadata = [source["metadata"] for source in current_metadata.get("sources", [])
                    if isinstance(source, Mapping) and isinstance(source.get("metadata"), Mapping)
                    and source["metadata"].get("kind") == "conditional_tau_bound30"]
    if tau_metadata:
        if len(tau_metadata) != 1 or len(current_metadata["sources"]) != 1:
            raise ValueError("Conditional tau requires exactly one unmixed ratio reward")
        expected = tau_metadata[0]["source_checkpoint"]
        if policy_checkpoint is None or Path(policy_checkpoint).resolve() != Path(expected).resolve():
            raise ValueError("Conditional tau denominator must match the raw1110 cold-start policy")
        if int(tau_metadata[0]["reward_round_id"]) != 0:
            raise ValueError("Later tau rounds require full-state resume")
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
