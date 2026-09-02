"""Materialize K=1 Ztautau diffusion pools and fit the OmniFold reward.

This is intentionally an independent pipeline stage.  It does not modify the
diffusion policy and it does not run DGPO.  The saved PEFT reward bundle is the
artifact consumed by the later OmniFold-guided DGPO integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch import Tensor

from evenet.dataset.preprocess import unflatten_dict
from evenet.utilities.diffusion_sampler import DDIMSampler
from RL.DGPO_neutrino.model_utils import (
    load_evenet_model_for_dgpo,
    load_normalization_dict,
    load_training_config,
)
from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
    EventPackingSpec,
    EvenetAdapterModelBuilder,
    FrozenResidualRatioReward,
    fit_residual_ratio_stack,
    pack_event_inputs,
    peft_bank_factory,
)
from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import RatioFitConfig
from RL.DGPO_neutrino.sampling import generate_neutrino_candidates

_log = logging.getLogger("ztautau_omnifold")
POOL_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}, got {type(value)!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _require_parquet_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not any(path.glob("*.parquet")):
        raise FileNotFoundError(f"{label} contains no parquet files: {path}")
    return path


def _shape_metadata(input_dir: Path, explicit: Path | None) -> dict[str, list[int]]:
    if explicit is not None:
        path = explicit
    else:
        candidates = (
            input_dir / "shape_metadata.json",
            input_dir.parent / "shape_metadata.json",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    _require_file(path, "shape metadata")
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"shape metadata must be a mapping: {path}")
    return value


def _record_batch_to_torch(
    record_batch: pa.RecordBatch,
    shape_metadata: Mapping[str, list[int]],
    device: torch.device,
) -> dict[str, Tensor]:
    flat = {
        key: np.asarray(value)
        for key, value in pa.Table.from_batches([record_batch]).to_pydict().items()
    }
    nested = unflatten_dict(flat, dict(shape_metadata), drop_column_prefix=None)
    output: dict[str, Tensor] = {}
    for key, value in nested.items():
        array = np.asarray(value)
        tensor = torch.from_numpy(array)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        output[key] = tensor.to(device=device)
    return output


def _parquet_batches(
    input_dir: Path,
    shape_metadata: Mapping[str, list[int]],
    *,
    batch_size: int,
    device: torch.device,
) -> Iterable[dict[str, Tensor]]:
    for parquet_path in sorted(input_dir.glob("*.parquet")):
        source = pq.ParquetFile(parquet_path)
        for record_batch in source.iter_batches(batch_size=int(batch_size)):
            yield _record_batch_to_torch(record_batch, shape_metadata, device)


def _event_valid_mask(batch: Mapping[str, Tensor]) -> Tensor:
    if "x_invisible" not in batch or "x_invisible_mask" not in batch:
        raise KeyError("OmniFold pool needs x_invisible and x_invisible_mask")
    mask = batch["x_invisible_mask"]
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    if mask.ndim != 2 or int(mask.shape[1]) < 2:
        raise ValueError(f"x_invisible_mask must be (B,>=2), got {tuple(mask.shape)}")
    return (mask[:, :2] > 0.5).all(dim=1)


@torch.no_grad()
def materialize_pool(
    *,
    train_config: Path,
    policy_checkpoint: Path,
    input_dir: Path,
    shape_metadata_path: Path | None,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    num_ddim_steps: int,
    max_events: int | None,
    seed: int,
) -> dict[str, Any]:
    """Generate exactly one independent diffusion candidate per valid event."""

    torch.manual_seed(int(seed))
    bundle = load_evenet_model_for_dgpo(
        config_path=train_config,
        checkpoint_path=policy_checkpoint,
        device=device,
    )
    policy = bundle.model.eval()
    sampler = DDIMSampler(device)
    shapes = _shape_metadata(input_dir, shape_metadata_path)
    packed_events: list[Tensor] = []
    truths: list[Tensor] = []
    candidates: list[Tensor] = []
    spec: EventPackingSpec | None = None
    collected = 0

    for batch in _parquet_batches(
        input_dir, shapes, batch_size=batch_size, device=device
    ):
        valid = _event_valid_mask(batch)
        if not bool(valid.any().item()):
            continue
        x_invisible = batch["x_invisible"]
        if x_invisible.ndim != 3 or int(x_invisible.shape[1]) < 2:
            raise ValueError(
                f"x_invisible must be (B,>=2,F), got {tuple(x_invisible.shape)}"
            )
        feature_dim = int(getattr(policy, "invisible_input_dim", x_invisible.shape[-1]))
        if int(x_invisible.shape[-1]) < feature_dim:
            raise ValueError(
                f"x_invisible has {x_invisible.shape[-1]} features; model expects {feature_dim}"
            )
        generated = generate_neutrino_candidates(
            policy,
            batch,
            sampler,
            K=1,
            num_ddim_steps=int(num_ddim_steps),
            device=device,
            parallel_chains=1,
        )
        if tuple(generated.shape[:3]) != (1, int(x_invisible.shape[0]), 2):
            raise RuntimeError(
                "unexpected K=1 diffusion output shape: "
                f"{tuple(generated.shape)}"
            )
        packed, spec = pack_event_inputs(batch, spec)
        keep = valid.nonzero(as_tuple=True)[0]
        packed_events.append(packed[keep].cpu())
        truths.append(
            x_invisible[keep, :2, :feature_dim].reshape(len(keep), -1).float().cpu()
        )
        candidates.append(
            generated[:, keep, :2, :feature_dim]
            .permute(1, 0, 2, 3)
            .reshape(len(keep), 1, -1)
            .float()
            .cpu()
        )
        collected += int(keep.numel())
        if max_events is not None and collected >= int(max_events):
            break

    if not packed_events or spec is None:
        raise RuntimeError(f"OmniFold pool collected no valid events from {input_dir}")
    stop = collected if max_events is None else min(collected, int(max_events))
    payload = {
        "schema_version": POOL_SCHEMA_VERSION,
        "kind": "ztautau_omnifold_k1_pool",
        "packing_spec": spec.to_dict(),
        "candidate_features": ["delta_theta", "delta_phi"],
        "num_invisible_slots": 2,
        "candidates_per_event": 1,
        "packed_event": torch.cat(packed_events, dim=0)[:stop],
        "truth": torch.cat(truths, dim=0)[:stop],
        "candidate": torch.cat(candidates, dim=0)[:stop],
        "source": {
            "input_dir": str(input_dir),
            "train_config": str(train_config),
            "policy_checkpoint": str(policy_checkpoint),
            "policy_checkpoint_sha256": _sha256_file(policy_checkpoint),
            "num_ddim_steps": int(num_ddim_steps),
            "seed": int(seed),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    _log.info("wrote K=1 OmniFold pool %s with %s events", output_path, stop)
    return payload


def load_pool(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", -1)) != POOL_SCHEMA_VERSION:
        raise ValueError(f"unsupported Ztautau OmniFold pool schema in {path}")
    if int(payload.get("candidates_per_event", -1)) != 1:
        raise ValueError("current Ztautau OmniFold fit requires one candidate per event")
    condition = payload.get("packed_event")
    truth = payload.get("truth")
    candidate = payload.get("candidate")
    if not all(isinstance(value, Tensor) for value in (condition, truth, candidate)):
        raise TypeError(f"pool {path} is missing tensor populations")
    if condition.ndim != 2 or truth.ndim != 2 or candidate.ndim != 3:
        raise ValueError(
            f"pool {path} needs condition/truth rank 2 and candidate rank 3"
        )
    if len(condition) != len(truth) or len(condition) != len(candidate):
        raise ValueError(f"pool {path} event axes do not match")
    if tuple(candidate.shape[1:]) != (1, int(truth.shape[-1])) or int(truth.shape[-1]) != 4:
        raise ValueError(
            f"Ztautau pool samples must be truth (N,4), candidate (N,1,4); "
            f"got {truth.shape} and {candidate.shape}"
        )
    return payload


def build_fit_config(
    block: Mapping[str, Any], *, n_train: int, n_validation: int
) -> RatioFitConfig:
    global_batch = int(block.get("batch_size", 8192))
    drop_last_batch = bool(block.get("drop_last_batch", False))
    steps_per_epoch = (
        int(n_train) // global_batch
        if drop_last_batch
        else max(1, math.ceil(int(n_train) / global_batch))
    )
    if steps_per_epoch < 1:
        raise ValueError(
            "drop_last_batch requires n_train >= the configured global batch"
        )
    interval_epochs = float(block.get("validation_interval_epochs", 0.2))
    patience_epochs = float(block.get("validation_patience_epochs", 5.0))
    if interval_epochs <= 0.0 or patience_epochs <= 0.0:
        raise ValueError("validation interval and patience epochs must be positive")
    if "steps" in block:
        explicit_steps = block["steps"]
        steps: int | None = (
            None if explicit_steps is None else int(explicit_steps)
        )
    else:
        safety_max_epochs = block.get("safety_max_epochs", 300)
        steps = (
            None
            if safety_max_epochs is None
            else max(1, math.ceil(float(safety_max_epochs) * steps_per_epoch))
        )
    min_steps = int(
        block.get(
            "min_steps",
            max(1, math.ceil(float(block.get("min_epochs", 1)) * steps_per_epoch)),
        )
    )
    return RatioFitConfig(
        steps=steps,
        batch_size=global_batch,
        train_microbatch_size_per_rank=(
            None
            if block.get("train_microbatch_size_per_rank") is None
            else int(block["train_microbatch_size_per_rank"])
        ),
        drop_last_batch=drop_last_batch,
        learning_rate=float(block.get("learning_rate", 1.0e-3)),
        backbone_learning_rate=(
            None
            if block.get("backbone_learning_rate") is None
            else float(block["backbone_learning_rate"])
        ),
        weight_decay=float(block.get("weight_decay", 1.0e-4)),
        sampling=str(block.get("sampling", "independent_epoch_shuffle")),
        min_steps=min_steps,
        validation_interval_steps=int(
            block.get(
                "validation_interval_steps",
                max(1, math.ceil(interval_epochs * steps_per_epoch)),
            )
        ),
        validation_patience_evaluations=int(
            block.get(
                "validation_patience_evaluations",
                max(1, round(patience_epochs / interval_epochs)),
            )
        ),
        validation_min_delta=float(block.get("validation_min_delta", 5.0e-4)),
        validation_batch_size=max(
            1, min(int(block.get("validation_batch_size", 8192)), int(n_validation))
        ),
        restore_best=bool(block.get("restore_best", True)),
        progress_interval_steps=max(
            0, int(block.get("progress_every_n_steps", 0))
        ),
        require_saturation=bool(block.get("require_saturation", True)),
        train_candidates_per_event=1,
    )


def _diagnostic_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _diagnostic_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_payload(item) for item in value]
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, np.generic):
        return value.item()
    return value


def fit_omnifold(
    *,
    train_config: Path,
    backbone_checkpoint: Path,
    train_pool_path: Path,
    validation_pool_path: Path,
    output_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    train_pool = load_pool(train_pool_path)
    validation_pool = load_pool(validation_pool_path)
    if train_pool["packing_spec"] != validation_pool["packing_spec"]:
        raise ValueError("train and validation OmniFold pools have different event shapes")
    train_reference = str(
        (train_pool.get("source") or {}).get("policy_checkpoint_sha256", "")
    )
    validation_reference = str(
        (validation_pool.get("source") or {}).get("policy_checkpoint_sha256", "")
    )
    if not train_reference or train_reference != validation_reference:
        raise ValueError(
            "train and validation OmniFold pools must carry the same policy "
            "checkpoint SHA256; rebuild both K=1 pools"
        )
    spec = EventPackingSpec.from_dict(train_pool["packing_spec"])
    training_config = load_training_config(train_config)
    normalization_dict = load_normalization_dict(training_config)
    model_builder = EvenetAdapterModelBuilder(
        config=training_config,
        normalization_dict=normalization_dict,
        checkpoint_path=backbone_checkpoint,
        device=device,
        adapter_bottleneck=int(config.get("adapter_bottleneck", 16)),
        train_layernorm=bool(config.get("train_layernorm", False)),
        train_encoder=bool(config.get("train_encoder", False)),
        train_invisible_projector=bool(
            config.get("train_invisible_projector", False)
        ),
        train_backbone=bool(config.get("train_backbone", False)),
        asymmetric_attention=bool(config.get("asymmetric_attention", False)),
        head_dropout=float(config.get("head_dropout", 0.1)),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 256)),
        decoder_layers=int(config.get("decoder_layers", 2)),
        decoder_heads=int(config.get("decoder_heads", 8)),
    )
    fit_cfg = build_fit_config(
        dict(config.get("fit") or {}),
        n_train=len(train_pool["truth"]),
        n_validation=len(validation_pool["truth"]),
    )
    train_condition = train_pool["packed_event"].to(device=device, dtype=torch.float32)
    train_truth = train_pool["truth"].to(device=device, dtype=torch.float32)
    train_candidate = train_pool["candidate"].to(device=device, dtype=torch.float32)
    val_condition = validation_pool["packed_event"].to(device=device, dtype=torch.float32)
    val_truth = validation_pool["truth"].to(device=device, dtype=torch.float32)
    val_candidate = validation_pool["candidate"].to(device=device, dtype=torch.float32)
    result = fit_residual_ratio_stack(
        model_factory=peft_bank_factory(model_builder, spec, "reward", reset=True),
        data_condition=train_condition,
        data_sample=train_truth,
        gen_condition=train_condition,
        gen_sample=train_candidate,
        iterations=int(config.get("max_iterations", 12)),
        min_iterations=int(config.get("min_iterations", 2)),
        stop_balanced_accuracy=float(config.get("stop_balanced_accuracy", 0.55)),
        fit_config=fit_cfg,
        tempering=float(config.get("tempering", 1.0)),
        crossfit_folds=int(config.get("crossfit_folds", 2)),
        residual_min_auc_gain=float(
            config.get("residual_min_auc_gain", 1.0e-3)
        ),
        seed=int(config.get("seed", 20260819)),
        validation_data_condition=val_condition,
        validation_data_sample=val_truth,
        validation_gen_condition=val_condition,
        validation_gen_sample=val_candidate,
    )
    frozen = FrozenResidualRatioReward.from_fit_result(
        result, tempering=float(config.get("tempering", 1.0))
    )
    payload = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": "ztautau_evenet_omnifold_reward",
        "candidate_features": ["delta_theta", "delta_phi"],
        "num_invisible_slots": 2,
        "candidates_per_event_for_fit": 1,
        "reward": frozen.serializable_payload(),
        "fit": {
            "iterations": int(result.iterations),
            "crossfit_folds": int(config.get("crossfit_folds", 2)),
            "residual_min_auc_gain": float(
                config.get("residual_min_auc_gain", 1.0e-3)
            ),
            "diagnostics": _diagnostic_payload(result.diagnostics),
            "config": asdict(fit_cfg),
        },
        "classifier": {
            "adapter_bottleneck": int(config.get("adapter_bottleneck", 16)),
            "train_layernorm": bool(config.get("train_layernorm", False)),
            "train_encoder": bool(config.get("train_encoder", False)),
            "train_invisible_projector": bool(
                config.get("train_invisible_projector", False)
            ),
            "train_backbone": bool(config.get("train_backbone", False)),
            "asymmetric_attention": bool(
                config.get("asymmetric_attention", False)
            ),
            "adapter_placement": "internal",
            "decoder_hidden_dim": int(config.get("decoder_hidden_dim", 256)),
            "decoder_layers": int(config.get("decoder_layers", 2)),
            "decoder_heads": int(config.get("decoder_heads", 8)),
            "head_dropout": float(config.get("head_dropout", 0.1)),
        },
        "provenance": {
            "train_config": str(train_config),
            "backbone_checkpoint": str(backbone_checkpoint),
            "train_pool": str(train_pool_path),
            "validation_pool": str(validation_pool_path),
            "policy_reference_sha256": train_reference,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    _log.info("wrote Ztautau OmniFold reward bundle %s", output_path)
    return payload


def _resolved_config(config_path: Path) -> dict[str, Any]:
    raw = _read_yaml(config_path)
    block = raw.get("omnifold")
    if not isinstance(block, dict):
        raise ValueError(f"{config_path} needs a top-level omnifold mapping")
    if int(block.get("candidates_per_event", 1)) != 1:
        raise ValueError("current OmniFold stage requires omnifold.candidates_per_event: 1")
    base = config_path.parent
    output_dir = _resolve_path(block.get("output_dir"), base=base)
    if output_dir is None:
        raise ValueError("omnifold.output_dir is required")
    return {
        **block,
        "train_parquet_dir": _resolve_path(block.get("train_parquet_dir"), base=base),
        "validation_parquet_dir": _resolve_path(block.get("validation_parquet_dir"), base=base),
        "shape_metadata": _resolve_path(block.get("shape_metadata"), base=base),
        "policy_checkpoint": _resolve_path(block.get("policy_checkpoint"), base=base),
        "backbone_checkpoint": _resolve_path(
            block.get("backbone_checkpoint") or block.get("policy_checkpoint"), base=base
        ),
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "pool", "fit", "check"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-pools", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_config = _require_file(args.train_config.expanduser().resolve(), "train config")
    config_path = _require_file(args.config.expanduser().resolve(), "OmniFold config")
    cfg = _resolved_config(config_path)
    policy_checkpoint = _require_file(cfg["policy_checkpoint"], "policy checkpoint")
    backbone_checkpoint = _require_file(cfg["backbone_checkpoint"], "backbone checkpoint")
    train_dir = _require_parquet_dir(cfg["train_parquet_dir"], "train parquet directory")
    validation_dir = _require_parquet_dir(
        cfg["validation_parquet_dir"], "validation parquet directory"
    )
    output_dir = Path(cfg["output_dir"])
    train_pool_path = output_dir / "train_k1_pool.pt"
    validation_pool_path = output_dir / "validation_k1_pool.pt"
    reward_path = output_dir / "omnifold_reward.pt"

    if args.stage == "check":
        print(
            json.dumps(
                {
                    "train_config": str(train_config),
                    "policy_checkpoint": str(policy_checkpoint),
                    "backbone_checkpoint": str(backbone_checkpoint),
                    "train_parquet_dir": str(train_dir),
                    "validation_parquet_dir": str(validation_dir),
                    "output_dir": str(output_dir),
                    "candidates_per_event": 1,
                },
                indent=2,
            )
        )
        return

    device = torch.device(args.device)
    if args.stage in {"all", "pool"}:
        common = {
            "train_config": train_config,
            "policy_checkpoint": policy_checkpoint,
            "shape_metadata_path": cfg.get("shape_metadata"),
            "device": device,
            "batch_size": int(cfg.get("pool_batch_size", 256)),
            "num_ddim_steps": int(cfg.get("num_ddim_steps", 20)),
        }
        pool_seed = int(cfg.get("seed", 20260819))
        if args.rebuild_pools or not train_pool_path.is_file():
            materialize_pool(
                input_dir=train_dir,
                output_path=train_pool_path,
                max_events=(
                    None if cfg.get("max_train_events") is None else int(cfg["max_train_events"])
                ),
                seed=pool_seed,
                **common,
            )
        if args.rebuild_pools or not validation_pool_path.is_file():
            materialize_pool(
                input_dir=validation_dir,
                output_path=validation_pool_path,
                max_events=(
                    None
                    if cfg.get("max_validation_events") is None
                    else int(cfg["max_validation_events"])
                ),
                seed=pool_seed + 1,
                **common,
            )
    if args.stage in {"all", "fit"}:
        fit_omnifold(
            train_config=train_config,
            backbone_checkpoint=backbone_checkpoint,
            train_pool_path=_require_file(train_pool_path, "train OmniFold pool"),
            validation_pool_path=_require_file(
                validation_pool_path, "validation OmniFold pool"
            ),
            output_path=reward_path,
            config=cfg,
            device=device,
        )


if __name__ == "__main__":
    main()
