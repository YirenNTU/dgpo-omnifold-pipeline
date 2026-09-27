#!/usr/bin/env python3
"""Prepare matched token-FiLM A/B or an independent fresh audit; user launches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml
from train_neutrino_backend import deep_update, read_overlay_yaml, read_yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config/train_diffusion_nersc.yaml"


def configuration(arm, *, audit_checkpoint=None, expected_step=None):
    cfg = deep_update(read_yaml(BASE), read_overlay_yaml(ROOT / f"config/dgpo_token_film_{arm}.yaml"))
    if audit_checkpoint is None:
        if expected_step is not None:
            raise ValueError("--expected-step requires --audit-checkpoint")
        return cfg
    if expected_step is None or expected_step < 0:
        raise ValueError("fresh audit requires --expected-step; compare equal-step endpoints, not selected winners")
    # This is a different measurement from the old, smaller probe-split audit.
    # Keep its outputs and W&B identity separate rather than mixing populations.
    output = f"/pscratch/sd/y/yiren/Ztautau/h4_token_film_{arm}_matched_fold_audit_step{expected_step}"
    label = "global FiLM" if arm == "control" else "token FiLM"
    cfg = deep_update(cfg, {
        "options": {"Training": {"model_checkpoint_load_path": str(audit_checkpoint),
                                  "model_checkpoint_save_path": output + "/checkpoints"}},
        "dgpo": {"checkpoint_load_mode": "weights_only", "step_zero_architecture_bootstrap": False,
                 "lr_schedule": None,
                 "adaptive_omnifold": {"trigger": {"warm_start_classifier": False},
                                        "audit_fit": {"training_population": "omnifold_fold",
                                                      "training_fold": 1}}},
        "logger": {
            "local": {"name": f"token-film-{arm}-matched-audit-{expected_step}", "save_dir": output + "/logs"},
            "wandb": {"id": f"h4tok-{arm}-matched-{expected_step}", "resume": "never", "fresh_run": False,
                      "run_name": f"Does token routing reduce fresh H4 separation? | {label} | fold 1 | step {expected_step}",
                      "classifier_loss_curves": True, "classifier_loss_curves_raw": True,
                      "tags": ["ClassifierOnly", "FreshAudit", "MatchedFold", "NoPolicyUpdate",
                               "NoRewardRefit", "RawWeights", "TokenFiLM", "SixteenGPU"]}},
        "nersc": {"ray": {"results_dir": output + "/ray_results"},
                  "reproducibility": {"source_checkpoint": str(audit_checkpoint),
                    "note": "Frozen raw policy, cold H4 from the original pretrained classifier backbone. Same repeat-1/fold-1 OmniFold training identities and external validation identity split as 9592bbca/227e4975, with fresh K=1/DDIM20 samples. No saved classifier fit is reused or installed into RL."},
                  "execution": {"command": f"shifter python3 -u scripts/train_dgpo_token_film.py --arm {arm} --audit-checkpoint {audit_checkpoint} --expected-step {expected_step}"}},
    })
    # Replace inherited training metadata: this job makes no policy/reward update.
    cfg["experiment"] = {
        "protocol": "h4-matched-fold-fresh-audit-v1",
        "arm": f"fresh_h4_on_{arm}_matched_fold",
        "classifier_only": True,
        "source_policy_step": expected_step,
        "source_policy_epoch": None,
        "checkpoint_metadata_verified": False,
        "rounds": 0, "policy_updates_per_round": 0, "policy_update_budget": 0,
        "classifier_fit_count": 1, "reward_fit_count": 0, "cold_h4_audit_policy_steps": [0],
        "population_reference_runs": ["9592bbca", "227e4975"],
        "historical_population_events": {"fit": 208355, "early_stop": 59465, "test": 59527},
        "primary_endpoint": "staleness/raw_auc_gap",
        "endpoint_selection": "disjoint_test_at_best_validation_bce",
        "single_question": "Does token-specific diffusion conditioning reduce fresh-H4 distinguishability at matched policy steps and audit data?",
        "stop_rule": "One cold best-validation-BCE fit: no max steps, minimum 0, 25-epoch patience, zero policy/reward updates.",
        "primary_decision_rule": "Compare equal-step control/treatment disjoint-test AUC gaps and BCE after verifying matched populations and adequate fit convergence. Fixed reward alone is not closure; historical runs at other policy steps are context, not the matched architecture control.",
    }
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("control", "last"))
    parser.add_argument("--audit-checkpoint", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print configuration only; no checkpoint access or launch")
    args = parser.parse_args()
    cfg = configuration(args.arm, audit_checkpoint=args.audit_checkpoint, expected_step=args.expected_step)
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    if args.audit_checkpoint is not None:
        import torch
        path = args.audit_checkpoint.expanduser().resolve(strict=True)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
            raise ValueError("audit source has no raw policy state_dict")
        if int(payload.get("global_step", -1)) != args.expected_step:
            raise ValueError("audit source global_step does not match --expected-step")
        has_readout = any("TruthGeneration.visible_conditioning.token_readout." in key
                          for key in payload["state_dict"])
        if has_readout != (args.arm == "last"):
            raise ValueError("audit checkpoint architecture does not match --arm")
        cfg["options"]["Training"]["model_checkpoint_load_path"] = str(path)
        cfg["nersc"]["reproducibility"]["source_checkpoint"] = str(path)
        cfg["experiment"]["source_policy_epoch"] = payload.get("epoch")
        cfg["experiment"]["checkpoint_metadata_verified"] = True
        print(json.dumps({
            "source_checkpoint": str(path), "source_policy_step": args.expected_step,
            "source_policy_epoch": payload.get("epoch"), "policy_arm": args.arm,
            "classifier": "cold H4 from pretrained backbone; same architecture in both arms",
            "training_population": "omnifold_fold", "training_fold": 1,
            "historical_population_events": cfg["experiment"]["historical_population_events"],
            "workers": cfg["platform"]["number_of_workers"],
            "classifier_fits": 1, "policy_updates": 0, "reward_fits": 0,
        }, indent=2), flush=True)
        del payload
    with tempfile.TemporaryDirectory(prefix="h4-token-film-") as directory:
        overlay = Path(directory) / "overlay.yaml"
        overlay.write_text(yaml.safe_dump(cfg, sort_keys=False))
        command = [sys.executable, "-u", str(ROOT / "scripts/train_neutrino_backend.py"),
                   "--backend", "dgpo-evenet", "--base-config", str(BASE),
                   "--overlay-config", str(overlay), "--", "--ray-dir", cfg["nersc"]["ray"]["results_dir"]]
        if not cfg["experiment"]["classifier_only"]:
            # Stop the first screen without compressing/restarting the LR decay.
            command.extend(["--max-steps", str(cfg["experiment"]["policy_update_budget"])])
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
