#!/usr/bin/env python3
"""Launch controlled H4 fresh-ensemble pilots after the epsilon sweep."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_dgpo_old_method_10pct import assert_live_ray_16gpu
from train_neutrino_backend import read_overlay_yaml

DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble_5round.yaml"
ESS15_DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble_ess15_5round.yaml"
LONG_REWARD_DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble_ess15_single_reward_50step.yaml"
LONG_REWARD_6_DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble6_ess15_single_reward_50step.yaml"
LONG_REWARD_8_DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble8_ess15_single_reward_50step.yaml"
REFIT20_6_DEFAULT = ROOT / "config/dgpo_omnifold_ztautau_10pct_h4_fresh_ensemble6_ess15_refit20_keepadam_50step.yaml"
BASE = ROOT / "config/train_diffusion_nersc.yaml"
EXPECTED_SOURCE_RUN = "ytchou97-university-of-washington/nu2flow-RL/c4a91e07"
EXPECTED_SOURCE_STEP = 1110
CONTROL_PROTOCOL = "h4-fresh-ensemble-5round-v1"
ESS15_PROTOCOL = "h4-fresh-ensemble-ess15-5round-v1"
LONG_REWARD_PROTOCOL = "h4-fresh-ensemble-ess15-single-reward-50step-v1"
LONG_REWARD_6_PROTOCOL = "h4-fresh-ensemble6-ess15-single-reward-50step-v1"
LONG_REWARD_8_PROTOCOL = "h4-fresh-ensemble8-ess15-single-reward-50step-v1"
REFIT20_6_PROTOCOL = "h4-fresh-ensemble6-ess15-refit20-keepadam-50step-v1"
ESS15_CONTROL_RUN = "ytchou97-university-of-washington/nu2flow-RL/h4fr5r01"
LONG_REWARD_CONTROL_RUN = "ytchou97-university-of-washington/nu2flow-RL/h4e15r01"
LONG_REWARD_6_CONTROL_RUN = "ytchou97-university-of-washington/nu2flow-RL/h4e50r01"
LONG_REWARD_8_CONTROL_RUN = "ytchou97-university-of-washington/nu2flow-RL/h4e50r01"
REFIT20_6_CONTROL_RUN = "ytchou97-university-of-washington/nu2flow-RL/h4e50m6"
H4E50M6_BOOTSTRAP = Path(
    "/pscratch/sd/y/yiren/Ztautau/"
    "c4a91e07_h4_ensemble6_ess15_single_reward_50step_v1/checkpoints/"
    "dgpo-epoch=-1-next_ep=0-step=0.ckpt"
)
RECOVERY_KEYS = {
    "dgpo_checkpoint_version", "dgpo_next_epoch", "dgpo_optimizer_state_dict",
    "dgpo_round_ref_state_dict", "dgpo_adaptive_omnifold_state",
    "dgpo_omnifold_reward_stack", "state_dict",
}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used" not in str(exc):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def read_epsilon_result(path: Path, experiment: Mapping[str, Any]) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != experiment["expected_epsilon_schema"]:
        raise ValueError("epsilon report schema mismatch")
    if int(report.get("policy_global_step", -1)) != EXPECTED_SOURCE_STEP:
        raise ValueError("epsilon report does not use c4a91e07 step 1110")
    if int(report.get("classifier_fits", -1)) != 0 or int(report.get("policy_updates", -1)) != 0:
        raise ValueError("epsilon report was not a read-only v5 artifact replay")
    selection = dict(experiment.get("epsilon_selection") or {})
    if selection.get("mode") != "stable_distribution_signal":
        raise ValueError("pilot requires the stable_distribution_signal epsilon rule")
    minimum_cosine = float(selection.get("minimum_gradient_cosine", 0.99))
    if not 0.0 < minimum_cosine <= 1.0:
        raise ValueError("minimum_gradient_cosine must lie in (0, 1]")
    candidates = []
    for row in report.get("sweep", []):
        decision = row.get("decision", {})
        plus = row.get("plus", {})
        cosine = float(plus.get("anchor_gradient_cosine", float("nan")))
        if (
            bool(decision.get("pass_judge"))
            and bool(decision.get("pass_physics_jsd"))
            and math.isfinite(cosine)
            and cosine >= minimum_cosine
        ):
            candidates.append(row)
    selected = max(
        candidates, key=lambda row: float(row["epsilon_rms"]), default=None
    )
    if selected is None:
        raise ValueError(
            "epsilon sweep found no locally linear radius with improving "
            "classifier and distribution signal"
        )
    epsilon = float(selected["epsilon_rms"])
    temperature = float(report["temperature"])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("selected epsilon must be finite and positive")
    if not math.isfinite(temperature) or temperature < 1.0:
        raise ValueError("v5 temperature must be finite and at least one for conservative scaling")
    decision = selected["decision"]
    return {
        "epsilon_rms": epsilon,
        "temperature": temperature,
        "tempering": 1.0 / temperature,
        "selection_mode": "stable_distribution_signal",
        "minimum_gradient_cosine": minimum_cosine,
        "anchor_gradient_cosine": float(
            selected["plus"]["anchor_gradient_cosine"]
        ),
        "local_judge_auc_gap_delta": float(decision["judge_auc_gap_delta"]),
        "local_physics_mean_jsd_delta": float(decision["physics_mean_jsd_delta"]),
        "local_response_offset_delta": float(
            decision["response_mean_abs_bin_offset_delta"]
        ),
        "strict_joint_eligible": bool(decision.get("eligible", False)),
    }


def apply_epsilon_result(config: dict[str, Any], result: Mapping[str, float]) -> None:
    """Map the local RMS radius to AdamW LR and use the measured v5 logit scale."""

    epsilon = float(result["epsilon_rms"])
    training = config["options"]["Training"]
    training["learning_rate"] = epsilon
    training["learning_rate_body"] = 0.1 * epsilon
    for name in (
        "InvisibleInputProjector", "GroupedSequentialEmbedding",
        "GlobalEmbedding", "TruthGeneration",
    ):
        training["Components"][name]["learning_rate"] = epsilon
    training["Components"]["PET"]["learning_rate"] = 0.1 * epsilon
    config["dgpo"]["adaptive_omnifold"]["recalibration"]["tempering"] = float(
        result["tempering"]
    )
    config["experiment"]["resolved_epsilon_rms"] = epsilon
    config["experiment"]["resolved_v5_temperature"] = float(result["temperature"])
    config["experiment"]["resolved_reward_tempering"] = float(result["tempering"])


def assert_ess15_pairing(config: Mapping[str, Any]) -> None:
    """Prove the ESS pilot differs from h4fr5r01 only in reward tempering.

    Output locations and experiment/logger metadata are intentionally unique.
    Everything that can affect training must remain identical to the control,
    except for the adaptive-tempering mapping itself.
    """

    reference = read_overlay_yaml(DEFAULT)
    result = {
        "epsilon_rms": float(config["experiment"]["resolved_epsilon_rms"]),
        "temperature": float(config["experiment"]["resolved_v5_temperature"]),
        "tempering": float(config["experiment"]["resolved_reward_tempering"]),
    }
    apply_epsilon_result(reference, result)
    candidate = copy.deepcopy(dict(config))
    expected = copy.deepcopy(reference)

    if config["options"]["Training"]["model_checkpoint_save_path"] == expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]:
        raise ValueError("ESS15 requires a checkpoint directory distinct from its control")

    # These fields carry experiment identity or write to isolated destinations.
    candidate["experiment"] = copy.deepcopy(expected["experiment"])
    candidate["nersc"] = copy.deepcopy(expected["nersc"])
    candidate["logger"] = copy.deepcopy(expected["logger"])
    candidate["options"]["Training"]["model_checkpoint_save_path"] = expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]
    candidate["dgpo"]["adaptive_omnifold"]["recalibration"][
        "adaptive_tempering"
    ] = copy.deepcopy(
        expected["dgpo"]["adaptive_omnifold"]["recalibration"][
            "adaptive_tempering"
        ]
    )
    if candidate != expected:
        raise ValueError(
            "ESS15 pilot must differ from h4fr5r01 only in adaptive tempering "
            "and isolated experiment/output metadata"
        )


def assert_long_reward_pairing(config: Mapping[str, Any]) -> None:
    """Prove reward lifetime is the only change from h4e15r01.

    The paired run keeps the initial classifier/reward construction and every
    policy setting fixed.  Capping installed reward rounds at one removes the
    four intermediate reward and Adam resets while retaining ten-step audits.
    """

    reference = read_overlay_yaml(ESS15_DEFAULT)
    result = {
        "epsilon_rms": float(config["experiment"]["resolved_epsilon_rms"]),
        "temperature": float(config["experiment"]["resolved_v5_temperature"]),
        "tempering": float(config["experiment"]["resolved_reward_tempering"]),
    }
    apply_epsilon_result(reference, result)
    candidate = copy.deepcopy(dict(config))
    expected = copy.deepcopy(reference)

    if config["options"]["Training"]["model_checkpoint_save_path"] == expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]:
        raise ValueError(
            "long-reward pilot requires a checkpoint directory distinct from h4e15r01"
        )

    candidate["experiment"] = copy.deepcopy(expected["experiment"])
    candidate["nersc"] = copy.deepcopy(expected["nersc"])
    candidate["logger"] = copy.deepcopy(expected["logger"])
    candidate["options"]["Training"]["model_checkpoint_save_path"] = expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]
    candidate["dgpo"]["adaptive_omnifold"]["recalibration"][
        "max_reward_rounds"
    ] = expected["dgpo"]["adaptive_omnifold"]["recalibration"][
        "max_reward_rounds"
    ]
    if candidate != expected:
        raise ValueError(
            "long-reward pilot must differ from h4e15r01 only in "
            "max_reward_rounds and isolated experiment/output metadata"
        )


def assert_long_reward_ensemble_pairing(
    config: Mapping[str, Any], members: int
) -> None:
    """Prove ensemble size is the only change from h4e50r01."""

    reference = read_overlay_yaml(LONG_REWARD_DEFAULT)
    result = {
        "epsilon_rms": float(config["experiment"]["resolved_epsilon_rms"]),
        "temperature": float(config["experiment"]["resolved_v5_temperature"]),
        "tempering": float(config["experiment"]["resolved_reward_tempering"]),
    }
    apply_epsilon_result(reference, result)
    candidate = copy.deepcopy(dict(config))
    expected = copy.deepcopy(reference)

    if config["options"]["Training"]["model_checkpoint_save_path"] == expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]:
        raise ValueError(
            f"ensemble-{members} pilot requires a checkpoint directory distinct "
            "from h4e50r01"
        )

    candidate["experiment"] = copy.deepcopy(expected["experiment"])
    candidate["nersc"] = copy.deepcopy(expected["nersc"])
    candidate["logger"] = copy.deepcopy(expected["logger"])
    candidate["options"]["Training"]["model_checkpoint_save_path"] = expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]
    candidate["dgpo"]["adaptive_omnifold"]["recalibration"][
        "crossfit_repeats"
    ] = expected["dgpo"]["adaptive_omnifold"]["recalibration"][
        "crossfit_repeats"
    ]
    if candidate != expected:
        raise ValueError(
            f"ensemble-{members} pilot must differ from h4e50r01 only in "
            "crossfit_repeats and isolated experiment/output metadata"
        )


def assert_refit20_6_pairing(config: Mapping[str, Any]) -> None:
    """Prove non-disruptive 20-step refresh is the change from h4e50m6."""

    reference = read_overlay_yaml(LONG_REWARD_6_DEFAULT)
    result = {
        "epsilon_rms": float(config["experiment"]["resolved_epsilon_rms"]),
        "temperature": float(config["experiment"]["resolved_v5_temperature"]),
        "tempering": float(config["experiment"]["resolved_reward_tempering"]),
    }
    apply_epsilon_result(reference, result)
    candidate = copy.deepcopy(dict(config))
    expected = copy.deepcopy(reference)

    if config["options"]["Training"]["model_checkpoint_save_path"] == expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]:
        raise ValueError(
            "refit20 pilot requires a checkpoint directory distinct from h4e50m6"
        )

    candidate["experiment"] = copy.deepcopy(expected["experiment"])
    candidate["nersc"] = copy.deepcopy(expected["nersc"])
    candidate["logger"] = copy.deepcopy(expected["logger"])
    candidate["options"]["Training"]["model_checkpoint_save_path"] = expected[
        "options"
    ]["Training"]["model_checkpoint_save_path"]
    candidate["options"]["Training"]["model_checkpoint_load_path"] = expected[
        "options"
    ]["Training"]["model_checkpoint_load_path"]
    candidate["dgpo"]["checkpoint_load_mode"] = expected["dgpo"][
        "checkpoint_load_mode"
    ]
    candidate["dgpo"]["pinned_classifier_restart"] = expected["dgpo"].get(
        "pinned_classifier_restart", False
    )
    candidate["dgpo"]["adaptive_omnifold"]["recalibration"][
        "bootstrap_on_start"
    ] = expected["dgpo"]["adaptive_omnifold"]["recalibration"][
        "bootstrap_on_start"
    ]
    candidate["dgpo"]["adaptive_omnifold"]["trigger"][
        "max_reward_age_epochs"
    ] = expected["dgpo"]["adaptive_omnifold"]["trigger"][
        "max_reward_age_epochs"
    ]
    candidate["dgpo"]["adaptive_omnifold"]["recalibration"][
        "max_reward_rounds"
    ] = expected["dgpo"]["adaptive_omnifold"]["recalibration"][
        "max_reward_rounds"
    ]
    candidate["dgpo"]["adaptive_omnifold"]["recalibration"][
        "reset_optimizer_state_on_install"
    ] = expected["dgpo"]["adaptive_omnifold"]["recalibration"][
        "reset_optimizer_state_on_install"
    ]
    if candidate != expected:
        raise ValueError(
            "refit20 pilot must differ from h4e50m6 only in the "
            "artifact-equivalent round-1 reuse, the non-disruptive 20-step "
            "reward-refresh intervention, and isolated experiment/output metadata"
        )


def assert_inherited_six_member_checkpoint(payload: Mapping[str, Any]) -> bool:
    """Fail closed unless the source is h4e50m6's complete step-0 round 1."""

    required = {
        "state_dict", "dgpo_checkpoint_version", "dgpo_optimizer_state_dict",
        "dgpo_ref_state_dict", "dgpo_round_ref_state_dict",
        "dgpo_round_ref_sha256", "dgpo_omnifold_reward_stack",
        "dgpo_omnifold_reward_metadata", "dgpo_adaptive_omnifold_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "inherited h4e50m6 checkpoint is incomplete: " + ", ".join(missing)
        )
    if (
        int(payload.get("global_step", -1)),
        int(payload.get("epoch", -2)),
        int(payload.get("dgpo_next_epoch", -1)),
        int(payload.get("dgpo_reward_round_id", -1)),
    ) != (0, -1, 0, 1):
        raise ValueError(
            "initial reward reuse requires h4e50m6 epoch=-1/step=0/round=1"
        )
    state = payload["dgpo_adaptive_omnifold_state"]
    if int(state.get("reward_round_id", -1)) != 1:
        raise ValueError("inherited adaptive state is not reward round 1")
    monitor = state.get("raw_monitor_state") or {}
    monitor_complete = bool(monitor.get("state") and monitor.get("protocol"))
    if monitor and not monitor_complete:
        raise ValueError("inherited checkpoint has a partial raw monitor")
    reward_payload = (
        (payload.get("dgpo_omnifold_reward_stack") or {}).get("reward", {})
    )
    cache = reward_payload.get("warm_start_state") or {}
    protocol = cache.get("protocol") or {}
    if (
        cache.get("outer_partition") is not None
        or
        protocol.get("scheme") != "condition_sha256_v1"
        or int(protocol.get("folds", -1)) != 2
        or int(protocol.get("repeats", -1)) != 3
    ):
        raise ValueError(
            "inherited reward does not use the pinned 3x2 split without an "
            "outer single-pool partition"
        )
    increments = list(reward_payload.get("increments") or [])
    iterations = list(reward_payload.get("increment_iterations") or [])
    coefficients = list(reward_payload.get("increment_coefficients") or [])
    if (
        len(increments) != 6
        or iterations != [1] * 6
        or len(coefficients) != 6
        or any(
            not math.isclose(
                float(value), 1.0 / 6.0, rel_tol=1.0e-6, abs_tol=1.0e-6
            )
            for value in coefficients
        )
        or any(not item.get("state") for item in increments)
    ):
        raise ValueError("inherited reward is missing one or more of its six members")
    base_digests = {item.get("base_digest") for item in increments}
    packing_specs = [item.get("packing_spec") for item in increments]
    if (
        len(base_digests) != 1
        or None in base_digests
        or any(spec != packing_specs[0] for spec in packing_specs)
    ):
        raise ValueError("inherited reward members have inconsistent metadata")
    return monitor_complete


def assert_contract(config: Mapping[str, Any]) -> None:
    experiment = config["experiment"]
    dgpo = config["dgpo"]
    adaptive = dgpo["adaptive_omnifold"]
    trigger = adaptive["trigger"]
    recal = adaptive["recalibration"]
    training = config["options"]["Training"]
    protocol = experiment.get("protocol")
    if protocol not in {
        CONTROL_PROTOCOL, ESS15_PROTOCOL, LONG_REWARD_PROTOCOL,
        LONG_REWARD_6_PROTOCOL, LONG_REWARD_8_PROTOCOL, REFIT20_6_PROTOCOL,
    }:
        raise ValueError("wrong pilot protocol")
    expected_round_shape = (
        (3, 20)
        if protocol == REFIT20_6_PROTOCOL
        else (1, 50)
        if protocol in {
            LONG_REWARD_PROTOCOL, LONG_REWARD_6_PROTOCOL,
            LONG_REWARD_8_PROTOCOL,
        }
        else (5, 10)
    )
    if (
        experiment.get("rounds"), experiment.get("policy_updates_per_round")
    ) != expected_round_shape:
        raise ValueError(
            "pilot reward rounds and updates per round do not match its protocol"
        )
    if (training.get("epochs"), training.get("total_epochs"), dgpo.get("steps_per_epoch")) != (5, 5, 10):
        raise ValueError("fixed endpoint must be epoch 5 / global step 50")
    expected_load_mode = "resume" if protocol == REFIT20_6_PROTOCOL else "weights_only"
    if (
        dgpo.get("checkpoint_load_mode") != expected_load_mode
        or not dgpo.get("auto_resume_from_last")
    ):
        raise ValueError("checkpoint startup mode does not match the pilot protocol")
    if protocol == REFIT20_6_PROTOCOL:
        if not dgpo.get("pinned_classifier_restart"):
            raise ValueError("refit20 must pinned-inherit the h4e50m6 round-1 reward")
        if Path(training["model_checkpoint_load_path"]) != H4E50M6_BOOTSTRAP:
            raise ValueError("refit20 must load the exact h4e50m6 step-0 snapshot")
        if recal.get("bootstrap_on_start") is not False:
            raise ValueError("refit20 must not retrain the inherited initial reward")
    elif dgpo.get("pinned_classifier_restart") or recal.get("bootstrap_on_start") is not True:
        raise ValueError("fresh-start pilots must train their bootstrap reward")
    if dgpo.get("advantage_estimator") != "leave_one_out_unscaled" or float(dgpo.get("beta")) != 1.0:
        raise ValueError("pilot requires unscaled LOO and DGPO beta=1")
    if not dgpo.get("log_parameter_update_rms"):
        raise ValueError("pilot must measure actual parameter update RMS")
    if not dgpo.get("fail_on_skipped_optimizer_step"):
        raise ValueError("pilot must count ten committed optimizer updates per round")
    if dgpo["reference_trust"]["adaptive_boundary"].get("enabled"):
        raise ValueError("hard trust boundary must remain disabled")
    if dgpo.get("projection_constraint", {}).get("type") != "none":
        raise ValueError("pilot cannot add a projection constraint")
    if adaptive.get("monitor_mode") != "raw_plateau_refit":
        raise ValueError("iteration-one scheduled rounds require raw_plateau_refit")
    if adaptive.get("staleness_every_n_epochs") != 1 or adaptive.get("staleness_every_n_steps") is not None:
        raise ValueError("one adaptive boundary must occur after each ten-step epoch")
    expected_reward_age = 2 if protocol == REFIT20_6_PROTOCOL else 1
    if (
        trigger.get("max_reward_age_epochs") != expected_reward_age
        or trigger.get("retrain_cooldown_epochs") != 0
    ):
        raise ValueError("reward-age schedule does not match the pilot protocol")
    if trigger.get("warm_start_classifier") or trigger.get("rollback_to_best_on_plateau"):
        raise ValueError("trajectory judge must start fresh and cannot roll back the policy")
    if recal.get("warm_start_iterations") or recal.get("warm_start_from_iteration_one"):
        raise ValueError("reward classifiers must start fresh every round")
    if (recal.get("min_iterations"), recal.get("max_iterations"), recal.get("iteration_one_only")) != (1, 1, True):
        raise ValueError("pilot installs exactly one H4 residual increment per round")
    expected_crossfit = {
        REFIT20_6_PROTOCOL: (3, 2),
        LONG_REWARD_6_PROTOCOL: (3, 2),
        LONG_REWARD_8_PROTOCOL: (4, 2),
    }.get(protocol, (2, 2))
    if (
        recal.get("crossfit_repeats"), recal.get("crossfit_folds")
    ) != expected_crossfit:
        raise ValueError("reward ensemble size does not match the pilot protocol")
    expected_reward_rounds = (
        3
        if protocol == REFIT20_6_PROTOCOL
        else 1
        if protocol in {
            LONG_REWARD_PROTOCOL, LONG_REWARD_6_PROTOCOL,
            LONG_REWARD_8_PROTOCOL,
        }
        else 5
    )
    if recal.get("max_reward_rounds") != expected_reward_rounds:
        raise ValueError("reward-round budget does not match the pilot protocol")
    if recal.get("scheduled_refit_fail_closed") is not True:
        raise ValueError("a rejected scheduled classifier must stop the fixed-round pilot")
    if recal.get("acceptance_audit_enabled") or recal.get("topology_acceptance_audit_enabled"):
        raise ValueError("no metric gate may alter the fixed 50-step endpoint")
    h4 = {
        "periodic_pair_features": True, "topology_fourier_embedding": True,
        "topology_conditioning": False, "visible_pair_rest_frame": False,
        "topology_max_harmonic": 4, "topology_include_theta_pair": False,
        "topology_direct_logit": False,
    }
    for block_name, block in (("reward", recal), ("judge", adaptive["audit_fit"])):
        mismatch = {key: (block.get(key), value) for key, value in h4.items() if block.get(key) != value}
        if mismatch:
            raise ValueError(f"{block_name} classifier is not the pinned nonlinear H4 model: {mismatch}")
    if recal["fit"].get("checkpoint_selection_metric") != "balanced_accuracy":
        raise ValueError("H4 members must select ordering by held-out balanced accuracy")
    if adaptive["audit_fit"].get("checkpoint_selection_metric") != "balanced_accuracy":
        raise ValueError("trajectory judges must select ordering by held-out balanced accuracy")
    if not adaptive["audit_fit"].get("disjoint_final_audit"):
        raise ValueError("trajectory judge requires a disjoint final audit partition")
    for label, fit in (
        ("reward", recal["fit"]),
        ("trajectory judge", adaptive["audit_fit"]),
    ):
        if (
            int(fit.get("topology_warmup_steps", 0)) != 0
            or int(fit.get("topology_body_unfreeze_step", 0)) != 0
            or fit.get("topology_warmup_learning_rate") is not None
        ):
            raise ValueError(
                f"{label} must use joint H4 fitting with topology staging disabled"
            )
    if not (0.0 < float(recal.get("tempering", 0.0)) <= 1.0):
        raise ValueError("resolved reward tempering must lie in (0, 1]")
    adaptive_tempering = recal.get("adaptive_tempering") or {}
    if protocol == CONTROL_PROTOCOL:
        if adaptive_tempering != {"enabled": False}:
            raise ValueError("control pilot requires fixed v5 tempering")
    else:
        expected_adaptive_tempering = {
            "enabled": True,
            "target_ess_fraction": 0.15,
            "minimum": 0.10,
            "grid_steps": 14,
            "inherit_previous": False,
        }
        if adaptive_tempering != expected_adaptive_tempering:
            raise ValueError(
                "ESS15 pilot requires the pinned soft ESS tempering search"
            )
        if float(recal.get("minimum_ess_fraction", 0.0)) != 0.0:
            raise ValueError("ESS15 is a soft scaling experiment, not a hard ESS gate")
        if (recal.get("ess_aware_checkpoint_selection") or {}).get("enabled"):
            raise ValueError("ESS15 cannot also change classifier checkpoint selection")
        expected_control_run = (
            REFIT20_6_CONTROL_RUN
            if protocol == REFIT20_6_PROTOCOL
            else (
                LONG_REWARD_6_CONTROL_RUN
                if protocol == LONG_REWARD_6_PROTOCOL
                else (
                    LONG_REWARD_8_CONTROL_RUN
                    if protocol == LONG_REWARD_8_PROTOCOL
                    else (
                        LONG_REWARD_CONTROL_RUN
                        if protocol == LONG_REWARD_PROTOCOL
                        else ESS15_CONTROL_RUN
                    )
                )
            )
        )
        if experiment.get("control_wandb_run") != expected_control_run:
            raise ValueError("adaptive-tempering pilot has the wrong paired control")
        expected_control_id = expected_control_run.rsplit("/", 1)[-1]
        if config["logger"]["wandb"].get("id") == expected_control_id:
            raise ValueError(
                "adaptive-tempering pilot requires a W&B run distinct from its control"
            )
        paired_overlay = (
            LONG_REWARD_6_DEFAULT
            if protocol == REFIT20_6_PROTOCOL
            else LONG_REWARD_DEFAULT
            if protocol in {LONG_REWARD_6_PROTOCOL, LONG_REWARD_8_PROTOCOL}
            else (ESS15_DEFAULT if protocol == LONG_REWARD_PROTOCOL else DEFAULT)
        )
        paired_ray_dir = read_overlay_yaml(paired_overlay)["nersc"]["ray"][
            "results_dir"
        ]
        if config["nersc"]["ray"].get("results_dir") == paired_ray_dir:
            raise ValueError(
                "adaptive-tempering pilot requires a Ray directory distinct from its control"
            )
        if protocol == REFIT20_6_PROTOCOL:
            if int(experiment.get("reward_lifetime_updates", -1)) != 20:
                raise ValueError("refit20 pilot requires a twenty-update reward lifetime")
            if experiment.get("reward_refresh_steps") != [0, 20, 40]:
                raise ValueError("refit20 pilot requires reward installs at steps 0, 20, 40")
            if int(experiment.get("final_reward_updates", -1)) != 10:
                raise ValueError("refit20 pilot final reward must run for ten updates")
            if (
                experiment.get("initial_reward_training_reused") is not True
                or experiment.get("initial_reward_source_run")
                != REFIT20_6_CONTROL_RUN
                or int(experiment.get("initial_reward_source_global_step", -1)) != 0
            ):
                raise ValueError("refit20 pilot must declare h4e50m6 step-0 reward reuse")
            if (
                int(experiment.get("ensemble_seeds", -1)),
                int(experiment.get("crossfit_folds", -1)),
            ) != (3, 2):
                raise ValueError("refit20 pilot requires three seeds by two folds")
            if recal.get("reset_optimizer_state_on_install") is not False:
                raise ValueError("refit20 pilot must preserve policy Adam state")
            assert_refit20_6_pairing(config)
        elif protocol in {
            LONG_REWARD_PROTOCOL, LONG_REWARD_6_PROTOCOL,
            LONG_REWARD_8_PROTOCOL,
        }:
            if int(experiment.get("reward_lifetime_updates", -1)) != 50:
                raise ValueError("long-reward pilot requires a fifty-update lifetime")
            if protocol in {LONG_REWARD_6_PROTOCOL, LONG_REWARD_8_PROTOCOL}:
                members = 6 if protocol == LONG_REWARD_6_PROTOCOL else 8
                repeats = members // 2
                if (
                    int(experiment.get("ensemble_seeds", -1)),
                    int(experiment.get("crossfit_folds", -1)),
                ) != (repeats, 2):
                    raise ValueError(
                        f"ensemble-{members} pilot requires {repeats} seeds by two folds"
                    )
                assert_long_reward_ensemble_pairing(config, members)
            else:
                assert_long_reward_pairing(config)
        else:
            assert_ess15_pairing(config)
    expected_lr = float(experiment["resolved_epsilon_rms"])
    if not math.isclose(float(training["learning_rate"]), expected_lr, abs_tol=0.0):
        raise ValueError("policy LR does not match the selected epsilon")
    wb = config["logger"]["wandb"]
    if wb.get("fresh_run") or wb.get("resume") != "allow" or not wb.get("id"):
        raise ValueError("fixed W&B id with resume=allow is required for live/preempt-safe logging")
    if not experiment.get("wandb_required"):
        raise ValueError("pilot must fail before training when live W&B is unavailable")
    selection = experiment.get("epsilon_selection", {})
    if selection.get("mode") != "stable_distribution_signal":
        raise ValueError("pilot epsilon selection rule changed")
    targets = experiment.get("primary_physics_targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError("pilot requires a predeclared primary physics target panel")
    if wb.get("profile") == "critical":
        raise ValueError("critical W&B profile suppresses response matrices")
    platform = config["platform"]
    if int(platform.get("number_of_workers", 0)) != 16 or int(platform["resources_per_worker"].get("GPU", 0)) != 1:
        raise ValueError("pilot requires 16 Ray workers with one GPU each")


def reference_ray_dir() -> str:
    return str(read_overlay_yaml(DEFAULT)["nersc"]["ray"]["results_dir"])


def assert_wandb_runtime_ready() -> None:
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("WANDB_DISABLED conflicts with this live-logging experiment")
    if os.environ.get("WANDB_MODE", "").lower() in {
        "offline", "disabled", "dryrun",
    }:
        raise RuntimeError("WANDB_MODE must allow online logging for this experiment")
    if importlib.util.find_spec("wandb") is None:
        raise RuntimeError("wandb is not installed in the launcher's Python environment")


def verify_paths(config: Mapping[str, Any]) -> str:
    training = config["options"]["Training"]
    for value in (
        training["model_checkpoint_load_path"],
        config["reward_config"]["omnifold"]["backbone_checkpoint"],
        config["platform"]["data_parquet_dir"],
        config["platform"]["data_parquet_val_dir"],
    ):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    source = Path(training["model_checkpoint_load_path"])
    payload = _load_checkpoint(source)
    if config["experiment"].get("protocol") == REFIT20_6_PROTOCOL:
        inherited_monitor = assert_inherited_six_member_checkpoint(payload)
        source_mode = (
            "pinned h4e50m6 step-0 policy/reward inheritance; "
            + (
                "raw monitor inherited"
                if inherited_monitor
                else "fresh diagnostic raw baseline required"
            )
        )
    else:
        if int(payload.get("global_step", -1)) != EXPECTED_SOURCE_STEP:
            raise ValueError(
                f"source checkpoint is not c4a91e07 step {EXPECTED_SOURCE_STEP}"
            )
        source_mode = "fresh weights-only start from c4a91e07 step 1110"
    output = Path(training["model_checkpoint_save_path"])
    if output.resolve() == source.resolve().parent:
        raise ValueError("pilot output cannot overwrite c4a91e07")
    last = output / "last.ckpt"
    if last.is_file():
        resumed = _load_checkpoint(last)
        missing = sorted(RECOVERY_KEYS - set(resumed))
        if missing:
            raise ValueError(f"incomplete pilot last.ckpt: missing {missing}")
        step = int(resumed.get("global_step", -1))
        if not 0 <= step <= 50:
            raise ValueError(f"pilot last.ckpt step is outside [0, 50]: {step}")
        return f"resume step={step} round={resumed.get('dgpo_reward_round_id')}"
    if output.exists() and any(output.glob("*.ckpt")):
        raise FileExistsError("pilot output has checkpoints but no complete last.ckpt")
    return source_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--epsilon-report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--skip-filesystem-checks", action="store_true")
    args = parser.parse_args(argv)
    config = read_overlay_yaml(args.config.resolve())
    experiment = config["experiment"]
    recal = config["dgpo"]["adaptive_omnifold"]["recalibration"]
    report_path = (
        args.epsilon_report.expanduser().resolve()
        if args.epsilon_report is not None
        else Path(experiment["epsilon_sweep_report"]).expanduser().resolve()
    )
    result = read_epsilon_result(report_path, experiment)
    apply_epsilon_result(config, result)
    assert_contract(config)
    mode = "filesystem checks skipped"
    if not args.skip_filesystem_checks:
        mode = verify_paths(config)
    ensemble_description = (
        f"inherited H4 {recal['crossfit_repeats']}x{recal['crossfit_folds']} "
        "round 1 plus fresh matching refits"
        if experiment["protocol"] == REFIT20_6_PROTOCOL
        else f"fresh H4 {recal['crossfit_repeats']}x{recal['crossfit_folds']} ensemble"
    )
    print(
        f"Pilot preflight passed: {mode}\n"
        f"epsilon/LR={result['epsilon_rms']:.6g}, v5 T={result['temperature']:.6g}, "
        f"reward tempering={result['tempering']:.6g}\n"
        f"Contract: 16 GPUs, {experiment['rounds']} reward round(s), "
        f"{experiment['policy_updates_per_round']} updates/reward, "
        f"{ensemble_description}, "
        "step-50 endpoint, live W&B, no hard boundary; "
        + (
            "soft ESS>=15% adaptive tempering."
            if experiment["protocol"] in {
                ESS15_PROTOCOL, LONG_REWARD_PROTOCOL, LONG_REWARD_6_PROTOCOL,
                LONG_REWARD_8_PROTOCOL, REFIT20_6_PROTOCOL,
            }
            else "fixed v5 tempering."
        ),
        flush=True,
    )
    if args.check_only:
        return 0
    assert_wandb_runtime_ready()
    assert_live_ray_16gpu()
    output_root = Path(config["options"]["Training"]["model_checkpoint_save_path"]).parent
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["options"]["Training"]["model_checkpoint_save_path"])
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.ckpt"))
    resolved_path = output_root / "resolved_overlay.yaml"
    if resolved_path.exists():
        existing = yaml.safe_load(resolved_path.read_text())
        if existing != config:
            if has_checkpoint:
                raise ValueError("resolved pilot overlay changed after a checkpoint was written")
            with resolved_path.open("w") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)
    else:
        with resolved_path.open("x") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
    manifest = output_root / "experiment_manifest.json"
    manifest_payload = {
        "protocol": experiment["protocol"], "source_wandb_run": EXPECTED_SOURCE_RUN,
        "source_policy_step": EXPECTED_SOURCE_STEP, "epsilon_report": str(report_path),
        **result, "rounds": int(experiment["rounds"]),
        "policy_updates_per_round": int(experiment["policy_updates_per_round"]),
        "endpoint_global_step": 50,
        "classifier_members_per_round": int(
            config["dgpo"]["adaptive_omnifold"]["recalibration"][
                "crossfit_repeats"
            ]
        ) * int(
            config["dgpo"]["adaptive_omnifold"]["recalibration"][
                "crossfit_folds"
            ]
        ),
    }
    if experiment["protocol"] == REFIT20_6_PROTOCOL:
        manifest_payload.update({
            "initial_reward_training_reused": True,
            "initial_reward_source_run": REFIT20_6_CONTROL_RUN,
            "initial_reward_source_global_step": 0,
            "initial_raw_monitor_reused": False,
            "initial_raw_monitor_rebuilt_at_step_zero": True,
            "reward_refresh_steps": [0, 20, 40],
            "reset_policy_optimizer_on_reward_install": False,
        })
    if experiment["protocol"] in {
        ESS15_PROTOCOL, LONG_REWARD_PROTOCOL, LONG_REWARD_6_PROTOCOL,
        LONG_REWARD_8_PROTOCOL, REFIT20_6_PROTOCOL,
    }:
        adaptive = config["dgpo"]["adaptive_omnifold"]["recalibration"][
            "adaptive_tempering"
        ]
        manifest_payload.update({
            "control_wandb_run": experiment["control_wandb_run"],
            "adaptive_tempering": True,
            "target_ess_fraction": float(adaptive["target_ess_fraction"]),
            "minimum_tempering": float(adaptive["minimum"]),
            "tempering_grid_steps": int(adaptive["grid_steps"]),
        })
    if manifest.exists():
        if json.loads(manifest.read_text()) != manifest_payload:
            if has_checkpoint:
                raise ValueError("pilot manifest changed after a checkpoint was written")
            with manifest.open("w") as handle:
                json.dump(manifest_payload, handle, indent=2)
                handle.write("\n")
    else:
        with manifest.open("x") as handle:
            json.dump(manifest_payload, handle, indent=2)
            handle.write("\n")
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/train_neutrino_backend.py"),
            "--backend", "dgpo-evenet", "--base-config", str(BASE),
            "--overlay-config", str(resolved_path), "--", "--ray-dir",
            str(config["nersc"]["ray"]["results_dir"]),
        ],
        cwd=ROOT, check=True,
    )
    final = _load_checkpoint(Path(config["options"]["Training"]["model_checkpoint_save_path"]) / "last.ckpt")
    if int(final.get("global_step", -1)) != 50:
        raise RuntimeError("pilot returned without reaching the fixed step-50 endpoint")
    expected_rounds = int(experiment["rounds"])
    if int(final.get("dgpo_reward_round_id", -1)) != expected_rounds:
        raise RuntimeError(
            f"pilot did not finish with exactly {expected_rounds} installed reward rounds"
        )
    print(
        f"Pilot complete: fixed endpoint step=50, installed reward rounds={expected_rounds}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
