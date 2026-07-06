#!/usr/bin/env python3
"""Diagnostic: compare DGPO checkpoint weights loaded two ways and run a quick DDIM test.

Usage:
    python RL/DGPO_neutrino/diagnose_checkpoint.py \
        --ckpt /path/to/DGPO/last.ckpt \
        --config RL/DGPO_neutrino/config.yaml

This script:
1. Loads the checkpoint and prints key statistics (EMA vs state_dict difference).
2. Loads the model via DGPO's own code path and via EveNet's predict code path.
3. Compares weight tensors between the two loading paths.
4. Runs a single DDIM rollout from each and compares output distributions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("diagnose")


def _stat(t: torch.Tensor) -> str:
    t = t.float()
    return f"mean={t.mean():.6f} std={t.std():.6f} min={t.min():.6f} max={t.max():.6f} nan={t.isnan().sum()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="DGPO checkpoint path")
    parser.add_argument("--config", default=str(_REPO / "RL/DGPO_neutrino/config.yaml"))
    parser.add_argument("--predict_config", default=str(_REPO / "reasoning_module/predict_TT2L.yaml"))
    args = parser.parse_args()

    # ── Step 1: Raw checkpoint inspection ─────────────────────────────────
    log.info("=== Step 1: Checkpoint inspection ===")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    log.info("Keys: %s", list(ckpt.keys()))
    log.info("dgpo_checkpoint_version: %s", ckpt.get("dgpo_checkpoint_version"))
    log.info("epoch: %s  global_step: %s", ckpt.get("epoch"), ckpt.get("global_step"))
    log.info("Has ema_state_dict: %s", "ema_state_dict" in ckpt)
    log.info("Has dgpo_ref_state_dict: %s", "dgpo_ref_state_dict" in ckpt)

    sd = ckpt["state_dict"]
    ema_sd = ckpt.get("ema_state_dict", {})

    if ema_sd:
        n_match = 0
        n_diff = 0
        max_diff = 0.0
        for k, v in ema_sd.items():
            sd_key = f"model.{k}"
            if sd_key in sd:
                diff = (sd[sd_key].float() - v.float()).abs().max().item()
                if diff < 1e-7:
                    n_match += 1
                else:
                    n_diff += 1
                    max_diff = max(max_diff, diff)
        log.info(
            "EMA vs state_dict: %d exact matches, %d differ (max diff %.6g)",
            n_match, n_diff, max_diff,
        )
    else:
        log.warning("No ema_state_dict in checkpoint!")

    # Print stats for a few key tensors
    for prefix in ["model.TruthGeneration.", "model.PET.transformer_blocks.0."]:
        for k, v in sd.items():
            if k.startswith(prefix) and v.dim() >= 1:
                log.info("  sd[%s]: shape=%s %s", k, tuple(v.shape), _stat(v))
                break

    # ── Step 2: Load via DGPO code path ───────────────────────────────────
    log.info("\n=== Step 2: Load via DGPO code path ===")
    from evenet.control.global_config import global_config
    from RL.DGPO_neutrino.model_utils import (
        load_evenet_model_for_dgpo,
        load_training_config,
        make_ema,
    )

    dgpo_cfg = load_training_config(args.config)
    global_config.load_yaml(Path(args.config).resolve())
    device = torch.device("cpu")

    loaded = load_evenet_model_for_dgpo(config=dgpo_cfg, device=device, checkpoint_path=args.ckpt)
    model_dgpo = loaded.model
    model_dgpo.eval()

    ema_dgpo = make_ema(model_dgpo, dgpo_cfg, checkpoint=ckpt, device=device)
    if ema_dgpo is not None:
        ema_dgpo.copy_to(model_dgpo)
        log.info("Applied DGPO EMA shadow to model")

    dgpo_sd = {k: v.clone() for k, v in model_dgpo.state_dict().items()}
    log.info("DGPO model: %d parameters loaded", len(dgpo_sd))

    # ── Step 3: Load via EveNet predict code path ─────────────────────────
    log.info("\n=== Step 3: Load via EveNet predict code path ===")
    from evenet.engine import EveNetEngine

    predict_cfg_obj = load_training_config(args.predict_config)
    global_config.load_yaml(Path(args.predict_config).resolve())

    norm_file = global_config.options.Dataset.normalization_file
    norm_dict = torch.load(norm_file, weights_only=False, map_location="cpu")

    engine = EveNetEngine(
        global_config=global_config,
        world_size=1,
        total_events=1000,
        total_val_events=100,
        normalization_dict=norm_dict,
    )
    engine.configure_model()

    # Simulate _load_predict_checkpoint
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = engine.load_state_dict(raw["state_dict"], strict=False)
    if missing:
        log.warning("Missing keys: %s", missing[:10])
        log.warning("  ... total %d missing", len(missing))
    if unexpected:
        log.warning("Unexpected keys: %s", unexpected[:10])
        log.warning("  ... total %d unexpected", len(unexpected))
    engine.on_load_checkpoint(raw)
    engine.eval()

    predict_sd = {k: v.clone() for k, v in engine.model.state_dict().items()}
    log.info("Predict model: %d parameters loaded", len(predict_sd))

    # ── Step 4: Compare weights ───────────────────────────────────────────
    log.info("\n=== Step 4: Weight comparison (DGPO vs Predict loading) ===")
    all_keys = set(dgpo_sd.keys()) | set(predict_sd.keys())
    n_same = 0
    n_diff = 0
    n_missing_dgpo = 0
    n_missing_predict = 0
    diffs = []
    for k in sorted(all_keys):
        if k not in dgpo_sd:
            n_missing_dgpo += 1
            continue
        if k not in predict_sd:
            n_missing_predict += 1
            continue
        d = (dgpo_sd[k].float() - predict_sd[k].float()).abs().max().item()
        if d < 1e-6:
            n_same += 1
        else:
            n_diff += 1
            diffs.append((k, d))

    log.info("Same: %d | Different: %d | Only in DGPO: %d | Only in Predict: %d",
             n_same, n_diff, n_missing_dgpo, n_missing_predict)
    if diffs:
        log.warning("Top differing keys:")
        for k, d in sorted(diffs, key=lambda x: -x[1])[:20]:
            log.warning("  %s: max_diff=%.6g", k, d)

    # ── Step 5: Quick DDIM rollout comparison ─────────────────────────────
    log.info("\n=== Step 5: DDIM rollout comparison ===")
    from evenet.utilities.diffusion_sampler import DDIMSampler

    B, N_nu, F = 8, 2, 3
    sampler = DDIMSampler(device=device)

    # Create a synthetic batch
    torch.manual_seed(42)
    fake_batch = {
        "x": torch.randn(B, 20, 7),
        "x_mask": torch.ones(B, 20),
        "conditions": torch.randn(B, 10),
        "conditions_mask": torch.ones(B, 1),
        "x_invisible": torch.randn(B, N_nu, F),
        "x_invisible_mask": torch.ones(B, N_nu),
    }

    noise_mask = fake_batch["x_invisible_mask"].unsqueeze(-1)
    data_shape = (B, N_nu, F)

    from functools import partial

    with torch.no_grad():
        # DGPO-style rollout
        torch.manual_seed(123)
        pred_fn_dgpo = partial(
            model_dgpo.predict_diffusion_vector,
            mode="neutrino",
            cond_x=fake_batch,
            noise_mask=noise_mask,
        )
        out_dgpo = sampler.sample(
            data_shape=data_shape,
            pred_fn=pred_fn_dgpo,
            normalize_fn=model_dgpo.invisible_normalizer,
            num_steps=30,
            remove_padding=True,
            noise_mask=noise_mask,
        )

        # Predict-style rollout (same model, same seed)
        torch.manual_seed(123)
        pred_fn_predict = partial(
            engine.model.predict_diffusion_vector,
            mode="neutrino",
            cond_x=fake_batch,
            noise_mask=noise_mask,
        )
        out_predict = sampler.sample(
            data_shape=data_shape,
            pred_fn=pred_fn_predict,
            normalize_fn=engine.model.invisible_normalizer,
            num_steps=30,
            remove_padding=True,
            noise_mask=noise_mask,
        )

    diff = (out_dgpo - out_predict).abs()
    log.info("DDIM output diff: max=%.6g mean=%.6g", diff.max().item(), diff.mean().item())
    for i, name in enumerate(["log_pt", "eta", "phi"]):
        log.info("  %s DGPO: %s", name, _stat(out_dgpo[..., i]))
        log.info("  %s Pred: %s", name, _stat(out_predict[..., i]))

    # Also test predict_step-style (no noise_mask in sampler)
    torch.manual_seed(123)
    out_predict_no_mask = sampler.sample(
        data_shape=data_shape,
        pred_fn=pred_fn_predict,
        normalize_fn=engine.model.invisible_normalizer,
        num_steps=100,  # predict uses 100 steps
        remove_padding=True,
        # NO noise_mask here — this is how predict_step calls it!
    )
    log.info("\nPredict-style (no noise_mask in sampler, 100 steps):")
    for i, name in enumerate(["log_pt", "eta", "phi"]):
        log.info("  %s: %s", name, _stat(out_predict_no_mask[..., i]))

    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()
