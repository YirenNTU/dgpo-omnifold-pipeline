#!/usr/bin/env python3
"""Launch the 16-GPU, fixed-H4 DGPO pilot without additive reference trust."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from train_neutrino_backend import read_overlay_yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_lastblock_no_ref_100step.yaml"


def assert_contract(c: dict) -> None:
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    d = c["dgpo"]
    require(d["log_every"] == 1 and d["log_parameter_update_rms"], "log policy gradient and actual updates every step")
    require(d["gradient_conflict"]["enabled"] and d["gradient_conflict"]["every_n_steps"] == 50, "gradient direction probes at 50/100 required")
    a = d["adaptive_omnifold"]
    r = a["recalibration"]
    t = c["options"]["Training"]
    e = c["experiment"]
    require(e.get("classifier_only") is False, "must execute policy training")
    require(d["checkpoint_load_mode"] == "weights_only" and not d["auto_resume_from_last"], "fresh policy state required")
    require(t["model_checkpoint_load_path"].endswith("dgpo-epoch=110-next_ep=111-step=1110.ckpt"), "pin step-1110 policy")
    require("10pct" in c["platform"]["data_parquet_dir"], "matched 10pct data required")
    require(d["beta"] == 1 and float(d.get("beta_kl", 0)) == 0, "beta=1; no legacy KL anchor")
    trust = d["reference_trust"]
    require(not trust["enabled"] and trust["coefficient"] == 0 and not trust["adaptive_boundary"]["enabled"], "no reference trust or hard boundary")
    require(not d["tarp"]["enabled"] and d["projection_constraint"]["type"] == "none", "no extra controllers")
    require(d["ztautau_metrics"]["enabled"] and d["ztautau_metrics"]["candidate_index"] == 0, "unbiased joint-topology diagnostics required")
    require(d["validation_every_n_epochs"] == d["validation_full_every_n_epochs"] == 1, "physics diagnostics every policy epoch required")
    require(t["epochs"] == t["total_epochs"] == 10 and d["steps_per_epoch"] == 10, "exactly 100 policy steps")
    require(a["log_only"] and not a["baseline_probe_on_start"] and a["monitor_mode"] == "raw_only", "no step-zero audit; read-only later audits")
    require(a["fixed_audit_panel"] and a["cache_event_inputs"], "fixed event/noise panel required")
    require(r["score_pool_events"] == a["trigger"]["probe_max_events"], "match bootstrap and later frozen-classifier populations")
    require(e["primary_endpoint"] == "frozen_classifier/ensemble/auc_gap", "use the original frozen classifier as the primary endpoint")
    require(e["cold_h4_audit_policy_steps"] == [50, 100], "audit only at 50/100")
    require(e["retain_installed_classifier"] is True, "retain installed classifier for later ablations")
    require(e["retained_classifier_payload_key"] == "dgpo_omnifold_reward_stack", "record reward-stack payload key")
    require(e["retained_classifier_checkpoint"].endswith("dgpo-epoch=-1-next_ep=0-step=0.ckpt"), "pin pre-policy classifier checkpoint")
    require(d["checkpoint_every_n_epochs"] == 1, "preserve unpruned recovery checkpoints")
    require(a["staleness_every_n_epochs"] == 5 and a["staleness_every_n_steps"] is None, "audit every five epochs")
    require(not a["trigger"]["warm_start_classifier"] and not a["trigger"]["rollback_to_best_on_plateau"], "no warm audit or rollback")
    require(r["bootstrap_on_start"] and r["max_reward_rounds"] == 1 and not r["refit_once_on_resume"], "bootstrap once, no reward refits")
    require((r["crossfit_folds"], r["crossfit_repeats"], r["min_iterations"], r["max_iterations"]) == (2, 1, 1, 1), "two folds, one repeat, one iteration")
    require(r["iteration_one_only"] and e["residual_iterations_max"] == 1, "install one iteration without requiring residual closure")
    require(r["tempering"] == 1 and not r["adaptive_tempering"]["enabled"], "fixed alpha=1")
    require(not r["reset_optimizer_state_on_install"] and not r["reset_adam_first_moment_on_install"], "do not reset AdamW")
    for architecture in (r, a["audit_fit"]):
        for key, value in {
            "train_last_pet_block": True, "train_backbone": False,
            "train_encoder": False, "train_layernorm": False,
            "train_grouped_sequential_embedding": False, "train_invisible_projector": False,
            "head_dropout": 0.25, "topology_dropout": 0.25,
            "periodic_pair_features": True, "topology_fourier_embedding": True,
            "topology_max_harmonic": 4, "topology_direct_logit": False,
            "topology_conditioning": False, "visible_pair_rest_frame": False,
            "topology_include_theta_pair": False,
        }.items():
            require(architecture.get(key) == value, f"classifier requires {key}={value}")
    for fit in (r["fit"], a["audit_fit"]):
        require(fit["steps"] is None and fit["min_steps"] == 1000 and fit["safety_max_epochs"] is None, "minimum 1000 updates with no maximum")
        require(fit["validation_interval_steps"] is None and fit["validation_interval_epochs"] == 1 and fit["validation_patience_epochs"] == 10, "ten classifier epochs of patience")
        require(fit["checkpoint_selection_metric"] == "loss" and fit["restore_best"], "select by validation BCE")
        require(fit["learning_rate"] == 2e-4 and fit["backbone_learning_rate"] == 1e-5 and fit["weight_decay"] == 0.001, "preserve classifier optimizer")
    p = c["platform"]
    require(p["number_of_workers"] == 16 and p["resources_per_worker"]["GPU"] == 1 and p["use_gpu"], "16 one-GPU workers required")
    require(c["nersc"]["nodes"] * c["nersc"]["gpus_per_node"] == 16, "16-GPU allocation required")
    require(c["logger"]["wandb"]["id"] == "h4lbnr01", "separate W&B ID required")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--ray-dir", type=Path)
    args = parser.parse_args()
    config = read_overlay_yaml(CONFIG)
    assert_contract(config)
    print("Contract OK: 16 GPUs; one accepted reward iteration; 100 DGPO updates; no reference trust; frozen AUC at 0/50/100; secondary cold audits at 50/100; fits >=1000 updates, no max, patience 10 classifier epochs.", flush=True)
    if args.validate_only:
        return 0
    return subprocess.run([
        sys.executable, str(ROOT / "scripts/train_neutrino_backend.py"),
        "--backend", "dgpo-evenet", "--base-config", str(ROOT / "config/train_diffusion_nersc.yaml"),
        "--overlay-config", str(CONFIG), "--", "--ray-dir",
        str(args.ray_dir or config["nersc"]["ray"]["results_dir"]),
    ], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
