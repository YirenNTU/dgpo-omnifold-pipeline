#!/usr/bin/env python3
"""Pre-flight check: latent-SWD encoder checkpoint + DGPO config alignment.

Run on NERSC (login node is fine) before launching DGPO:

    python3 RL/DGPO_neutrino/latent_constraint/verify_latent_swd_checkpoint.py \
        RL/DGPO_neutrino/config.yaml

Exits 0 when the frozen encoder loads, normalizes, and encodes a synthetic batch.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evenet.control.global_config import global_config  # noqa: E402
from RL.DGPO_neutrino.latent_constraint.dgpo_constraint import (  # noqa: E402
    init_latent_swd_state,
    latent_swd_constraint_from_kin,
    resolve_latent_swd_config,
)
from RL.DGPO_neutrino.projection_cpo import resolve_projection_constraint_config  # noqa: E402

_log = logging.getLogger("verify_latent_swd")


def _synthetic_batch(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    n_vis, n_cond, n_part = 7, 10, 12
    x_mask = torch.ones(batch_size, n_part, device=device)
    x_mask[:, -3:] = 0.0
    return {
        "x": torch.randn(batch_size, n_part, n_vis, device=device),
        "x_mask": x_mask,
        "conditions": torch.randn(batch_size, n_cond, device=device),
        "conditions_mask": torch.ones(batch_size, 1, device=device),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Verify latent-SWD checkpoint for DGPO")
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=_REPO / "RL/DGPO_neutrino/config.yaml",
        help="DGPO YAML config (reads dgpo.projection_constraint.latent_swd)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the smoke test (default: cpu)",
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config_path = args.config.resolve()
    global_config.load_yaml(config_path)
    proj = resolve_projection_constraint_config(global_config.dgpo)
    latent_cfg = proj.latent_swd
    if latent_cfg is None:
        raise SystemExit("latent_swd block missing from projection_constraint.")

    ckpt_path = Path(latent_cfg.checkpoint_file).expanduser()
    policy_norm = Path(str(global_config.options.Dataset.normalization_file)).expanduser()
    latent_norm = Path(latent_cfg.normalization_file).expanduser() if latent_cfg.normalization_file else None

    _log.info("=== Config ===")
    _log.info("  DGPO config:        %s", config_path)
    _log.info("  checkpoint_file:    %s", ckpt_path)
    _log.info("  checkpoint exists:  %s", ckpt_path.is_file())
    _log.info("  policy norm:        %s (exists=%s)", policy_norm, policy_norm.is_file())
    if latent_norm is not None:
        _log.info("  latent_swd norm:    %s (exists=%s)", latent_norm, latent_norm.is_file())
        if policy_norm.resolve() != latent_norm.resolve():
            _log.warning(
                "  policy normalization_file differs from latent_swd.normalization_file "
                "— latent and policy neutrino spaces may diverge."
            )

    device = torch.device(args.device)
    state = init_latent_swd_state(latent_cfg, device=device, resume_payload=None)

    _log.info("=== Smoke encode + SWD ===")
    bs = 32
    batch = _synthetic_batch(bs, device)
    # The object-token bottleneck encoder conditions on the event CLS token and the
    # per-object token set; synthesize both. P must match x_mask's object count.
    n_part = int(batch["x_mask"].shape[1])
    batch["event_token"] = torch.randn(bs, int(state.model.token_dim), device=device)
    batch["object_token"] = torch.randn(
        bs, n_part, int(state.model.token_dim), device=device
    )
    _log.info(
        "  encoder is object_token_bottleneck_ae (token_dim=%s, P=%s); DGPO data "
        "must be the *_evttok augmented parquet (event_token + object_token).",
        int(state.model.token_dim), n_part,
    )
    pred_kin = torch.randn(bs, 2, state.model.nu_kin_dim, device=device)
    truth_kin = torch.randn(bs, 2, state.model.nu_kin_dim, device=device)
    c_norm, diag = latent_swd_constraint_from_kin(state, batch, pred_kin, truth_kin)
    _log.info("  C_norm:             %.6f", float(c_norm))
    for key in (
        "latent_constraint/swd_pred_truth",
        "latent_constraint/swd_truth_truth",
        "latent_constraint/swd_ratio",
        "latent_constraint/mask_count",
    ):
        val = diag.get(key)
        if val is not None:
            _log.info("  %s: %s", key, float(val.reshape(-1)[0]))

    frozen_ok = all(not p.requires_grad for p in state.model.parameters())
    eval_ok = not state.model.training
    if not frozen_ok or not eval_ok:
        raise SystemExit("Encoder is not frozen/eval after init_latent_swd_state.")

    _log.info("=== OK ===")
    _log.info("Checkpoint loads correctly; safe to launch DGPO with this config.")


if __name__ == "__main__":
    main()
