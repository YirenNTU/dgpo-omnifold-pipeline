#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from run_ad_dgpo_pipeline import DGPO_TEMPLATE, REPO_ROOT, run_diffusion_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 (DGPO): run diffusion/DGPO post-training on the AD-augmented parquet."
    )
    parser.add_argument("--stage-root", type=Path, required=True, help="Shared output directory used by the AD stage.")
    parser.add_argument("--diffusion-base-config", type=Path, default=REPO_ROOT / "config" / "train_pretrain.yaml")
    parser.add_argument("--diffusion-overlay-config", type=Path, default=DGPO_TEMPLATE)
    parser.add_argument(
        "--latent-checkpoint",
        type=Path,
        default=None,
        help="Override the latent constraint checkpoint. Defaults to <stage-root>/latent_constraint/checkpoints/last.ckpt.",
    )
    parser.add_argument(
        "--dgpo-init-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint used to initialize DGPO stage 2. Independent from the AD frozen backbone checkpoint.",
    )
    parser.add_argument("--augmented-dirname", default="ad_augmented")
    args = parser.parse_args()

    run_diffusion_stage(
        diffusion_base_config=args.diffusion_base_config.resolve(),
        diffusion_overlay_config=args.diffusion_overlay_config.resolve() if args.diffusion_overlay_config else None,
        stage_root=args.stage_root.expanduser().resolve(),
        augmented_dirname=args.augmented_dirname,
        latent_checkpoint=args.latent_checkpoint.expanduser().resolve() if args.latent_checkpoint else None,
        dgpo_init_checkpoint=args.dgpo_init_checkpoint.expanduser().resolve() if args.dgpo_init_checkpoint else None,
    )


if __name__ == "__main__":
    main()
