#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from run_ad_dgpo_pipeline import AD_TEMPLATE, REPO_ROOT, run_ad_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 (AD): export frozen EveNet tokens and train the latent constraint autoencoder."
    )
    parser.add_argument("--stage-root", type=Path, required=True, help="Shared output directory for the two-stage pipeline.")
    parser.add_argument("--ad-base-config", type=Path, default=REPO_ROOT / "config" / "train_pretrain_cls.yaml")
    parser.add_argument("--ad-overlay-config", type=Path, default=AD_TEMPLATE)
    parser.add_argument(
        "--ad-backbone-checkpoint",
        type=Path,
        default=None,
        help="Frozen classification/backbone checkpoint used only for token export.",
    )
    parser.add_argument(
        "--latent-checkpoint",
        type=Path,
        default=None,
        help="Skip latent training and reuse an existing latent constraint checkpoint.",
    )
    parser.add_argument("--augmented-dirname", default="ad_augmented")
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--token-workers", type=int, default=None, help="Parallel workers for token export. Defaults to one worker per selected device.")
    parser.add_argument("--token-devices", default=None, help="Comma-separated GPU device list for token export, or 'auto'.")
    args = parser.parse_args()

    run_ad_stage(
        ad_base_config=args.ad_base_config.resolve(),
        ad_overlay_config=args.ad_overlay_config.resolve() if args.ad_overlay_config else None,
        stage_root=args.stage_root.expanduser().resolve(),
        augmented_dirname=args.augmented_dirname,
        ad_checkpoint=args.ad_backbone_checkpoint.expanduser().resolve() if args.ad_backbone_checkpoint else None,
        latent_checkpoint=args.latent_checkpoint.expanduser().resolve() if args.latent_checkpoint else None,
        token_batch_size=args.token_batch_size,
        token_workers=args.token_workers,
        token_devices=args.token_devices,
    )


if __name__ == "__main__":
    main()
