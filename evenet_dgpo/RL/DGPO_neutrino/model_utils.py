"""Load EveNet for DGPO neutrino RL: checkpoint, frozen reference clone, EMA."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from evenet.control.global_config import Config
from evenet.network.evenet_model import EveNetModel, build_evenet_model_from_training_config
from evenet.utilities.ema import EMA
from evenet.utilities.tool import safe_load_state

_log = logging.getLogger(__name__)

# EveNetEngine.configure_model registers ``model.famo.w.<task>`` even when FAMO is off.
# DGPO saves bare EveNetModel weights; predict/load with strict=True needs these keys.
FAMO_STATE_DICT_TASKS = (
    "classification",
    "regression",
    "assignment",
    "generation",
    "segmentation",
)


def inject_default_famo_state_dict_keys(state_dict: dict[str, Any]) -> int:
    """Ensure Lightning ``state_dict`` contains default ``model.famo.w.*`` tensors.

    Returns the number of keys injected (0 when already present).
    """
    missing = [
        task
        for task in FAMO_STATE_DICT_TASKS
        if f"model.famo.w.{task}" not in state_dict
    ]
    if not missing:
        return 0
    for task in missing:
        state_dict[f"model.famo.w.{task}"] = torch.tensor([0.0])
    return len(missing)


@dataclass(frozen=True)
class EvenetForDGPO:
    """Artifacts returned by :func:`load_evenet_model_for_dgpo`."""

    model: EveNetModel
    config: Config
    normalization_dict: dict[str, Any]
    checkpoint_path: Path | None


def load_training_config(config_path: str | Path) -> Config:
    """Load merged EveNet YAML (including ``dgpo`` / ``reward_config``) into a fresh :class:`Config`."""
    path = Path(config_path).resolve()
    cfg = Config()
    cfg.load_yaml(path)
    return cfg


def load_normalization_dict(config: Config) -> dict[str, Any]:
    """Load ``options.Dataset.normalization_file``."""
    path = config.options.Dataset.normalization_file
    normalization_dict: dict[str, Any] = torch.load(path, weights_only=False)
    _log.info("[DGPO/model] normalization_file=%s", path)
    return normalization_dict


def resolve_checkpoint_path(
    config: Config,
    checkpoint_path: str | Path | None,
) -> Path | None:
    """Prefer explicit path; fail if a configured checkpoint does not exist."""
    if checkpoint_path is not None:
        p = Path(checkpoint_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"configured DGPO checkpoint does not exist: {p}")
        return p
    tr = config.options.Training
    for key in ("model_checkpoint_load_path", "pretrain_model_load_path"):
        raw = getattr(tr, key, None)
        if not raw:
            continue
        p = Path(str(raw)).expanduser().resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"options.Training.{key} does not exist or is not a file: {p}"
        )
    return None


def select_dgpo_training_state(
    checkpoint: dict[str, Any] | None,
    *,
    load_mode: str,
) -> dict[str, Any] | None:
    """Select whether a loaded checkpoint also restores DGPO training state.

    ``weights_only`` keeps the policy weights already loaded by
    :func:`load_evenet_model_for_dgpo`, but presents no checkpoint state to the
    reference/EMA/optimizer/controller builders.  This creates a true fresh
    DGPO run initialized from an existing policy.
    """
    mode = str(load_mode).strip().lower()
    if mode == "resume":
        return checkpoint
    if mode == "weights_only":
        return None
    raise ValueError(
        "dgpo.checkpoint_load_mode must be 'resume' or 'weights_only', "
        f"got {load_mode!r}"
    )


def dgpo_snapshot_checkpoint_name(
    *,
    last_completed_epoch: int,
    dgpo_next_epoch: int,
    global_step: int,
) -> str:
    """Stable filename for an unpruned DGPO recovery snapshot."""
    return (
        f"dgpo-epoch={int(last_completed_epoch)}-"
        f"next_ep={int(dgpo_next_epoch)}-step={int(global_step)}.ckpt"
    )


def update_last_checkpoint_pointer(snapshot_path: Path | str) -> Path:
    """Atomically point ``last.ckpt`` at a saved snapshot without losing legacy last."""
    snapshot = Path(snapshot_path).expanduser().resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(f"DGPO snapshot does not exist: {snapshot}")
    last = snapshot.parent / "last.ckpt"
    preserved = snapshot.parent / "dgpo-preserved-last-before-snapshot-mode.ckpt"
    if last.exists() and not last.is_symlink() and not preserved.exists():
        os.replace(last, preserved)
        _log.info("[DGPO/model] Preserved previous regular last.ckpt as %s", preserved)
    temporary = snapshot.parent / f".last.ckpt.tmp-{os.getpid()}"
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(snapshot.name)
        os.replace(temporary, last)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    _log.info("[DGPO/model] Updated last.ckpt → %s", snapshot.name)
    return last


def load_weights_like_configure_model(
    model: EveNetModel,
    ckpt_path: Path,
    device: torch.device,
    config: Config,
    *,
    for_dgpo_training: bool = False,
) -> dict[str, Any]:
    """Load Lightning checkpoint: respect EMA replace flags like ``EveNetEngine.configure_model``.

    When ``for_dgpo_training`` is True the EMA ``replace_model_after_load`` flag is **ignored**:
    DGPO resumes from ``state_dict`` and restores the EMA shadow separately via :func:`make_ema`.
    For DGPO checkpoints saved by current code, ``state_dict`` is the live trainable model.
    """
    ema_cfg = config.options.Training.get("EMA", None) or {}
    ema_enable = bool(ema_cfg.get("enable", False))
    ema_replace = bool(ema_cfg.get("replace_model_after_load", False))

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    is_dgpo_ckpt = int(ckpt.get("dgpo_checkpoint_version", 0)) >= 1

    if for_dgpo_training and is_dgpo_ckpt:
        safe_load_state(model, ckpt["state_dict"])
    elif ema_enable and "ema_state_dict" in ckpt and ema_replace:
        safe_load_state(model, ckpt["ema_state_dict"])
    else:
        safe_load_state(model, ckpt["state_dict"])
    return ckpt


def freeze_reference_model(model: nn.Module) -> None:
    """``eval()`` and disable gradients (reference policy in DGPO)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def unwrap_for_state_dict(model: nn.Module) -> nn.Module:
    """Return the repository-native EveNet module behind DDP/trainer wrappers."""
    current = model
    if isinstance(current, nn.parallel.DistributedDataParallel):
        current = current.module
    inner = getattr(current, "eve_net", None)
    if isinstance(inner, nn.Module):
        current = inner
    if hasattr(current, "_orig_mod"):
        current = current._orig_mod
    return current


def state_dict_sha256(model: nn.Module) -> str:
    """Stable digest of model tensor names, dtypes, shapes, and bytes."""
    digest = hashlib.sha256()
    state = unwrap_for_state_dict(model).state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        byte_view = tensor.reshape(1) if tensor.ndim == 0 else tensor
        digest.update(byte_view.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _debug_verify_component_freeze(
    model: EveNetModel, logical_name: str, freeze_cfg: Any
) -> None:
    """Emit debug logs when YAML freeze did not take effect as expected."""
    ftype = freeze_cfg.get("type", "none")
    if ftype == "none":
        return

    head = getattr(model, logical_name, None)
    if head is None:
        _log.debug(
            "[DGPO/model] freeze: component %r missing on model (YAML type=%r); no parameters frozen.",
            logical_name,
            ftype,
        )
        return

    if ftype == "full":
        n_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
        if n_train:
            _log.debug(
                "[DGPO/model] freeze: %r declared type=full but %s parameters still require_grad.",
                logical_name,
                f"{n_train:,}",
            )
        return

    if ftype == "partial":
        components = freeze_cfg.get("partial_freeze_components", None) or []
        if not components:
            _log.debug(
                "[DGPO/model] freeze: %r type=partial but partial_freeze_components is empty.",
                logical_name,
            )
            return
        named = dict(head.named_modules())
        unknown = [c for c in components if c not in named]
        if unknown:
            _log.debug(
                "[DGPO/model] freeze: %r partial_freeze_components not found on module: %s",
                logical_name,
                unknown,
            )
        for c in components:
            if c not in named:
                continue
            sub = named[c]
            n_sub = sum(p.numel() for p in sub.parameters() if p.requires_grad)
            if n_sub:
                _log.debug(
                    "[DGPO/model] freeze: %r partial subtree %r still has %s trainable parameters.",
                    logical_name,
                    c,
                    f"{n_sub:,}",
                )
        return

    if ftype == "random":
        params = list(head.parameters())
        if not params:
            _log.debug(
                "[DGPO/model] freeze: %r type=random but submodule has no parameters.",
                logical_name,
            )


def apply_component_freezes(model: EveNetModel, config: Config) -> None:
    """
    Apply ``options.Training.Components.<Name>.freeze`` via :meth:`EveNetModel.freeze_module`
    (same contract as Lightning ``EveNetEngine.configure_model``).

    Component names in YAML (``GlobalEmbedding``, ``PET``, ``TruthGeneration``, etc.) match
    :class:`EveNetModel` attributes used by ``freeze_module``.
    """
    cc = config.options.Training.Components
    applied: list[str] = []
    for name in cc:
        freeze_cfg = cc[name].get("freeze", None)
        if freeze_cfg is None:
            continue
        model.freeze_module(name, freeze_cfg)
        _debug_verify_component_freeze(model, name, freeze_cfg)
        ftype = freeze_cfg.get("type", "none")
        if ftype != "none":
            applied.append(f"{name}({ftype})")

    n_train = count_trainable_params(model)
    _log.info(
        "[DGPO/model] Component freeze from YAML: %s | trainable params: %s",
        applied if applied else "(none)",
        f"{n_train:,}",
    )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_reference_model_frozen(model_ref: EveNetModel, *, where: str) -> None:
    """Hard guard: reference must have no trainable parameters."""
    if count_trainable_params(model_ref) != 0:
        raise RuntimeError(f"[{where}] model_ref expected 0 trainable params.")
    n_grad = sum(1 for p in model_ref.parameters() if p.grad is not None)
    if n_grad != 0:
        raise RuntimeError(f"[{where}] model_ref has gradients on {n_grad} tensors.")


def build_evenet_on_device(
    config: Config,
    normalization_dict: dict[str, Any],
    device: torch.device,
) -> EveNetModel:
    """Instantiate ``EveNetModel`` and move modules/buffers to ``device``."""
    model = build_evenet_model_from_training_config(config, normalization_dict, device)
    return model.to(device)


def load_evenet_model_for_dgpo(
    config_path: str | Path | None = None,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    *,
    config: Config | None = None,
) -> EvenetForDGPO:
    """Load config, normalization, build ``EveNetModel``, optionally load checkpoint weights.

    Pass either ``config_path`` or a pre-populated ``config`` (e.g. ``global_config`` after
    ``load_yaml``) so dataset prep and model loading share the same merged YAML.

    Checkpoint selection uses EMA swap when configured in the training YAML.
    """
    if config is None:
        if config_path is None:
            raise ValueError("load_evenet_model_for_dgpo requires config_path or config=")
        config = load_training_config(config_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalization_dict = load_normalization_dict(config)
    model = build_evenet_on_device(config, normalization_dict, device)

    resolved = resolve_checkpoint_path(config, checkpoint_path)
    if resolved is not None:
        _log.info("[DGPO/model] Loading weights from %s", resolved)
        load_weights_like_configure_model(model, resolved, device, config, for_dgpo_training=True)
    else:
        _log.warning(
            "[DGPO/model] No checkpoint (pass checkpoint_path= or set model_checkpoint_load_path / "
            "pretrain_model_load_path); model is randomly initialized."
        )

    return EvenetForDGPO(
        model=model,
        config=config,
        normalization_dict=normalization_dict,
        checkpoint_path=resolved,
    )


def make_reference_model(
    current_model: EveNetModel,
    config: Config,
    normalization_dict: dict[str, Any],
    device: torch.device,
    checkpoint: dict[str, Any] | None = None,
) -> EveNetModel:
    """Frozen policy reference for the DGPO objective.

    On **first run** (no DGPO checkpoint, or supervised-only ckpt): ``ref_model`` gets the same
    weights as ``current_model`` (= the pretrained init). This is correct because both start equal.

    On **resume** from a DGPO checkpoint that contains ``dgpo_ref_state_dict``: the original
    reference weights are restored so the anchor stays fixed across sessions.

    Uses rebuild + :meth:`load_state_dict` because ``EveNetModel`` is not ``deepcopy``-safe.
    """
    model_ref = build_evenet_on_device(config, normalization_dict, device)
    if (
        checkpoint is not None
        and int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1
        and "dgpo_ref_state_dict" in checkpoint
    ):
        safe_load_state(model_ref, checkpoint["dgpo_ref_state_dict"])
        _log.info("[DGPO/model] Loaded ref_model from dgpo_ref_state_dict (fixed anchor).")
    else:
        model_ref.load_state_dict(current_model.state_dict())
        _log.info("[DGPO/model] Initialized ref_model from current model weights (first run).")
    freeze_reference_model(model_ref)
    assert_reference_model_frozen(model_ref, where="make_reference_model")
    return model_ref


def make_round_reference_model(
    fallback_reference: EveNetModel,
    config: Config,
    normalization_dict: dict[str, Any],
    device: torch.device,
    checkpoint: dict[str, Any] | None = None,
) -> EveNetModel:
    """Frozen policy paired with the currently installed adaptive ratio reward."""
    model_ref = build_evenet_on_device(config, normalization_dict, device)
    if checkpoint is not None and "dgpo_round_ref_state_dict" in checkpoint:
        safe_load_state(model_ref, checkpoint["dgpo_round_ref_state_dict"])
        _log.info("[DGPO/model] Restored adaptive OmniFold round reference.")
    else:
        model_ref.load_state_dict(unwrap_for_state_dict(fallback_reference).state_dict())
        _log.info("[DGPO/model] Initialized round reference from the fixed reference.")
    freeze_reference_model(model_ref)
    assert_reference_model_frozen(model_ref, where="make_round_reference_model")
    return model_ref


def make_ema(
    model: EveNetModel,
    config: Config,
    checkpoint: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> EMA | None:
    """Build :class:`~evenet.utilities.ema.EMA` and optionally load ``ema_state_dict`` from a Lightning ckpt.

    Returns ``None`` when ``options.Training.EMA.enable`` is false (same gating as :class:`~evenet.engine.EveNetEngine`).
    """
    ema_cfg = config.options.Training.get("EMA", None) or {}
    if not bool(ema_cfg.get("enable", False)):
        return None
    decay = float(ema_cfg.get("decay", 0.999))
    ema = EMA(model, decay=decay)
    if checkpoint is not None and "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"], device=device)
    return ema


def make_ema_rollout(
    model: EveNetModel,
    config: Config,
    checkpoint: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> EMA | None:
    """Build the optional EMA used only for Phase-1 rollout.

    Live-policy generation (``EMA.use_for_generation: false``) returns ``None`` so
    no unused rollout shadow is allocated, updated, restored, or checkpointed.
    When enabled, decay is overridden per step via :meth:`EMA.update`; the
    constructor value is unused.
    On resume, the smoothed rollout shadow is restored from ``dgpo_ema_rollout_state_dict`` when
    present so the Phase-1 rollout policy is continuous across sessions (avoids the transient
    constraint fluctuation caused by re-seeding from raw trainable weights). Falls back to the
    current trainable weights when the key is absent (first run or pre-this-change checkpoints).
    Returns ``None`` when EMA is disabled or generation explicitly uses the live policy.
    """
    ema_cfg = config.options.Training.get("EMA", None) or {}
    if not generation_uses_ema_shadow(ema_cfg):
        if bool(ema_cfg.get("enable", False)):
            _log.info(
                "[DGPO/model] Live-policy generation selected; rollout EMA shadow disabled."
            )
        return None
    ema = EMA(model, decay=0.0)
    if checkpoint is not None and "dgpo_ema_rollout_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["dgpo_ema_rollout_state_dict"], device=device)
        _log.info("[DGPO/model] Restored rollout EMA shadow from dgpo_ema_rollout_state_dict.")
    return ema


def build_lightning_compatible_checkpoint(
    model: nn.Module,
    ema: EMA | None,
    config: Config,
    ema_rollout: EMA | None = None,
) -> dict[str, Any]:
    """Build a Lightning-style DGPO checkpoint payload.

    ``state_dict`` uses the live trainable model weights so DGPO resume matches the optimizer
    state. ``ema_state_dict`` separately holds the save-EMA shadow when EMA is enabled.
    ``dgpo_ema_rollout_state_dict`` holds the fast Phase-1 rollout EMA shadow so the rollout
    policy is continuous across resume (see :func:`make_ema_rollout`).
    """
    orig_model = model
    if isinstance(orig_model, nn.parallel.DistributedDataParallel):
        orig_model = orig_model.module
    _inner = getattr(orig_model, "eve_net", None)
    if isinstance(_inner, nn.Module):
        orig_model = _inner
    if hasattr(orig_model, "_orig_mod"):
        orig_model = orig_model._orig_mod
    ema_cfg = config.options.Training.get("EMA", None) or {}
    ema_enabled = bool(ema_cfg.get("enable", False))

    checkpoint: dict[str, Any] = {}
    checkpoint["state_dict"] = {f"model.{k}": v for k, v in orig_model.state_dict().items()}
    n_famo = inject_default_famo_state_dict_keys(checkpoint["state_dict"])
    if n_famo > 0:
        _log.info(
            "[DGPO/model] Injected %s default FAMO key(s) into state_dict for predict/load.",
            n_famo,
        )

    if ema_enabled and ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
        _log.info("[DGPO/model] Saved live state_dict plus separate ema_state_dict.")

    if ema_enabled and ema_rollout is not None:
        checkpoint["dgpo_ema_rollout_state_dict"] = ema_rollout.state_dict()

    return checkpoint


def save_lightning_compatible_checkpoint(
    path: Path | str,
    model: nn.Module,
    ema: EMA | None,
    config: Config,
    *,
    last_completed_epoch: int,
    dgpo_next_epoch: int,
    global_step: int,
    optimizer: Optimizer | None = None,
    ref_model: nn.Module | None = None,
    round_ref_model: nn.Module | None = None,
    reward_round_id: int = 0,
    ema_rollout: EMA | None = None,
    dgpo_projection_constraint_state: dict[str, Any] | None = None,
    dgpo_omnifold_reward_metadata: dict[str, Any] | None = None,
    dgpo_adaptive_omnifold_state: dict[str, Any] | None = None,
    dgpo_omnifold_reward_stack: dict[str, Any] | None = None,
) -> None:
    """Write a ``.ckpt`` file using the same tensor layout as Lightning + EveNetEngine.

    ``last_completed_epoch`` is the last fully finished training epoch index (0-based).
    ``dgpo_next_epoch`` is the next epoch index the loop should run (equals
    ``last_completed_epoch + 1`` after a full epoch; can equal ``last_completed_epoch`` when
    saving mid-epoch interrupt).

    ``ref_model`` — frozen reference policy. Its ``state_dict`` is saved as
    ``dgpo_ref_state_dict`` so the anchor survives across resume sessions.
    """
    out_path = Path(path).expanduser().resolve()
    payload = build_lightning_compatible_checkpoint(model, ema, config, ema_rollout=ema_rollout)
    payload["epoch"] = int(last_completed_epoch)
    payload["global_step"] = int(global_step)
    payload["dgpo_checkpoint_version"] = 1
    try:
        import lightning
        payload["pytorch-lightning_version"] = lightning.__version__
    except Exception:
        payload["pytorch-lightning_version"] = "2.0.0"
    payload["dgpo_next_epoch"] = int(dgpo_next_epoch)
    if optimizer is not None:
        payload["dgpo_optimizer_state_dict"] = optimizer.state_dict()
    if ref_model is not None:
        orig_ref = unwrap_for_state_dict(ref_model)
        payload["dgpo_ref_state_dict"] = {
            f"model.{k}": v for k, v in orig_ref.state_dict().items()
        }
    if round_ref_model is not None:
        orig_round_ref = unwrap_for_state_dict(round_ref_model)
        payload["dgpo_round_ref_state_dict"] = {
            f"model.{k}": v for k, v in orig_round_ref.state_dict().items()
        }
        payload["dgpo_reward_round_id"] = int(reward_round_id)
        payload["dgpo_round_ref_sha256"] = state_dict_sha256(round_ref_model)
    if dgpo_projection_constraint_state is not None:
        payload["dgpo_projection_constraint_state"] = dgpo_projection_constraint_state
    if dgpo_omnifold_reward_metadata is not None:
        payload["dgpo_omnifold_reward_metadata"] = dgpo_omnifold_reward_metadata
    if dgpo_adaptive_omnifold_state is not None:
        payload["dgpo_adaptive_omnifold_state"] = dgpo_adaptive_omnifold_state
    if dgpo_omnifold_reward_stack is not None:
        payload["dgpo_omnifold_reward_stack"] = dgpo_omnifold_reward_stack
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The adaptive stack makes checkpoints large. Preserve the previous resume
    # point if a wall-time kill interrupts serialization.
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _log.info(
        "[DGPO/model] Wrote checkpoint %s (next_epoch=%s step=%s)",
        out_path,
        dgpo_next_epoch,
        global_step,
    )


def is_lightning_trainer_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """Return True if ``checkpoint`` looks like a full PyTorch Lightning trainer state (not DGPO-only)."""
    if int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1:
        return False
    keys = set(checkpoint.keys())
    return bool(
        "pytorch-lightning_version" in keys
        or "optimizer_states" in keys
        or "lr_schedulers" in keys
    )


def parse_dgpo_resume_from_checkpoint(checkpoint: dict[str, Any] | None) -> tuple[int, int]:
    """Return ``(start_epoch, global_step)`` for the DGPO training loop.

    Loads weights separately via :func:`load_evenet_model_for_dgpo`. Here we only interpret
    scheduling counters so supervised Lightning ckpts do not accidentally set a huge start epoch:
    those are detected via :func:`is_lightning_trainer_checkpoint` and reset to ``(0, 0)``.
    """
    if not checkpoint:
        return 0, 0
    gs = int(checkpoint.get("global_step", 0))
    if int(checkpoint.get("dgpo_checkpoint_version", 0)) >= 1:
        if "dgpo_next_epoch" not in checkpoint:
            _log.warning(
                "[DGPO/model] Checkpoint has dgpo_checkpoint_version but no dgpo_next_epoch; "
                "starting from epoch 0."
            )
            return 0, gs
        return int(checkpoint["dgpo_next_epoch"]), gs
    if is_lightning_trainer_checkpoint(checkpoint):
        return 0, 0
    if "dgpo_next_epoch" in checkpoint:
        return int(checkpoint["dgpo_next_epoch"]), gs
    ep = int(checkpoint.get("epoch", -1))
    return ep + 1, gs


def generation_uses_ema_shadow(ema_cfg: Mapping[str, Any] | None) -> bool:
    """Whether train/validation DDIM should temporarily install EMA weights.

    EMA remains updated and checkpointed independently.  An explicit
    ``use_for_generation: false`` keeps candidate generation on the live policy,
    matching the policy snapshot paired with adaptive OmniFold rewards.
    """
    cfg = dict(ema_cfg or {})
    if not bool(cfg.get("enable", False)):
        return False
    if "use_for_generation" in cfg:
        return bool(cfg["use_for_generation"])
    return True
