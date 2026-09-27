#!/usr/bin/env python3
"""Cold H4 audit on the original OmniFold fold; no policy or reward updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from train_dgpo_token_film import BASE, ROOT, configuration as token_configuration
from train_neutrino_backend import deep_update

DEFAULT_CHECKPOINT = Path(
    "/pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110_step0_lr5e5/checkpoints/"
    "dgpo-epoch=277-next_ep=278-step=2780.ckpt"
)
DEFAULT_OUTPUT = Path("/pscratch/sd/y/yiren/Ztautau/h4_latest_checkpoint_matched_fold_audit")


def inspect_checkpoint(path: Path) -> dict:
    """Pin the last symlink and read actual policy progress, never infer it from W&B."""
    import torch

    resolved = path.expanduser().resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
        raise ValueError("Audit source has no raw policy state_dict")
    step = payload.get("global_step")
    epoch = payload.get("epoch")
    if type(step) is not int or step < 0 or type(epoch) is not int or epoch < -1:
        raise ValueError("Audit source must record its actual global_step and epoch")
    if any("TruthGeneration.visible_conditioning.token_readout." in key
           for key in payload["state_dict"]):
        raise ValueError("This audit uses the original global-FiLM policy architecture, not the token-readout ablation")
    return {"path": str(resolved), "global_step": step, "epoch": epoch}


def configuration(checkpoint=DEFAULT_CHECKPOINT, output=DEFAULT_OUTPUT, *, metadata=None):
    step = None if metadata is None else metadata["global_step"]
    path = str(checkpoint) if metadata is None else metadata["path"]
    # Reuse the already-tested standalone cold-audit path and fit budget.
    cfg = token_configuration("control", audit_checkpoint=Path(path),
                              expected_step=0 if step is None else step)
    destination = str(output)
    cfg = deep_update(cfg, {
        "options": {"Training": {"model_checkpoint_save_path": destination + "/checkpoints"}},
        "dgpo": {"adaptive_omnifold": {
            "audit_fit": {"training_population": "omnifold_fold", "training_fold": 1},
        }},
        "logger": {
            "local": {"name": "h4-latest-matched-fold-audit", "save_dir": destination + "/logs"},
            "wandb": {
                "id": None, "resume": "never", "fresh_run": True,
                "run_name": "Does H4 improvement survive matched data? | OmniFold fold 1 | raw DGPO step 2780",
                "group": "H4 fresh checkpoint audits",
                "tags": ["FreshAudit", "MatchedFold", "H4", "RawWeights", "NoPolicyUpdate", "NoRewardRefit", "SixteenGPU"],
            },
        },
        "nersc": {
            "ray": {"results_dir": destination + "/ray_results"},
            "reproducibility": {
                "source_checkpoint": path,
                "note": "Raw policy weights only, pinned to the same step-2780 source as 052997c9 by default. Cold classifier from the original pretrained backbone, not saved reward banks. Train on the original repeat-1/fold-1 OmniFold identities, with fresh K=1 samples. External validation is identity-split 50/50 for early stop and test, matching 227e4975. One fit, zero policy/reward updates.",
            },
            "execution": {"command": "shifter python3 -u scripts/train_h4_checkpoint_audit.py"},
        },
    })
    # Do not carry obsolete reward-fit/epsilon-ablation metadata into this run.
    cfg["experiment"] = {
        "protocol": "h4-matched-fold-fresh-audit-v1",
        "arm": "fresh_h4_on_frozen_policy_matched_fold",
        "classifier_only": True,
        "classifier_fit_count": 1,
        "reward_fit_count": 0,
        "rounds": 0,
        "policy_updates_per_round": 0,
        "source_policy_step": step,
        "source_policy_epoch": None if metadata is None else metadata["epoch"],
        "source_stage": "pinned_raw_dgpo_policy",
        "checkpoint_metadata_verified": metadata is not None,
        "comparison_run": "227e4975",
        "supersedes_unmatched_audit": "052997c9",
        "historical_population_events": {"fit": 208355, "early_stop": 59465, "test": 59527},
        "startup_mode": "raw_policy_weights_only_cold_classifier",
        "intervention": "restore_original_omnifold_fold_audit_population",
        "primary_endpoint": "staleness/raw_auc",
        "endpoint_selection": "disjoint_test_at_best_validation_bce",
        "single_question": "Does the lower fresh-H4 AUC persist when the original OmniFold training fold and external validation split are restored?",
        "stop_rule": "One cold fit: no max steps, 25-epoch validation-BCE patience, restore best before disjoint test.",
        "primary_decision_rule": "Report disjoint test AUC/gap and best-validation BCE with fit convergence. Compare only to a fresh audit with matched data, architecture and fit budget; installed reward AUC is not a matched baseline.",
    }
    # A custom source must never inherit the default checkpoint's display label.
    label = f"raw DGPO step {step}" if step is not None else (
        "raw DGPO step 2780" if Path(checkpoint) == DEFAULT_CHECKPOINT else "raw checkpoint"
    )
    cfg["logger"]["wandb"]["run_name"] = (
        "Does H4 improvement survive matched data? | OmniFold fold 1 | " + label
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without loading files or launching Ray")
    args = parser.parse_args()
    metadata = None if args.dry_run else inspect_checkpoint(args.checkpoint)
    cfg = configuration(args.checkpoint, args.output, metadata=metadata)
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    print(json.dumps({"source_policy": metadata, "classifier": "fresh pretrained-backbone initialization",
                      "classifier_fits": 1, "policy_updates": 0, "reward_fits": 0,
                      "training_population": "omnifold_fold", "training_fold": 1,
                      "historical_population_events": cfg["experiment"]["historical_population_events"],
                      "workers": cfg["platform"]["number_of_workers"]}, indent=2), flush=True)
    with tempfile.TemporaryDirectory(prefix="h4-checkpoint-audit-") as directory:
        overlay = Path(directory) / "audit.yaml"
        overlay.write_text(yaml.safe_dump(cfg, sort_keys=False))
        subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts/train_neutrino_backend.py"),
            "--backend", "dgpo-evenet", "--base-config", str(BASE),
            "--overlay-config", str(overlay), "--", "--ray-dir", cfg["nersc"]["ray"]["results_dir"],
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
