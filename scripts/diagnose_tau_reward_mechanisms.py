#!/usr/bin/env python3
"""Prepare or run isolated step1920 tau reward mechanism experiments.

The user runs stages in an existing sixteen-GPU allocation. Preparation only
pins and validates inputs; it does not connect to Ray or start a trainer.
All native stages resume the original actor, reference, AdamW and scheduler.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml


REPO = Path(__file__).resolve().parents[1]
EVENET = REPO / "evenet_dgpo"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.diagnose_tau_reward_transfer import (  # noqa: E402
    POLICY_CONTRACT, SOURCE_STEP, TRAIN_DIRECTORY, VALIDATION_DIRECTORY,
    read_mapping, source_metadata, validate_source_runtime,
)
from scripts.tau_gradient_balance import validate_reference_balance
from scripts.tau_trajectory_reward_refit import validate_reward_refit
from scripts.tau_classifier_kl import validate_classifier_kl, validate_classifier_trust


STAGES = ("prepare", "diagnose", "repeat_diagnose", "ensemble", "trajectory", "summarize")
NATIVE_STAGES = ("diagnose", "repeat_diagnose", "ensemble", "trajectory")
REPEAT_PROTOCOL = "tau_reward_direction_repeats"
TRAJECTORY_PROTOCOL = "tau_full_trajectory"
REWARD_ARMS = ("inherited", "member0", "ensemble")
TRAJECTORY_UPDATES = (50, 200, 500)
RELATIVE_STEPS = (0, 1, 5, 10, 20, 35, 50, 100, 150, 200, 300, 400, 500)
AUDIT_STEPS = (0, 50, 200, 500)
MANIFEST = "invocation_manifest.json"


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def trajectory_reference_coefficient(block):
    value = block.get("reference_coefficient", 1.0)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("full_trajectory.reference_coefficient must be finite and positive")
    return float(value)


def validate_settings(settings):
    if settings.get("source_step") != SOURCE_STEP or type(settings.get("source_step")) is not int:
        raise ValueError("This series requires the exact step1920 source")
    if settings.get("workers") != 16 or type(settings.get("workers")) is not int:
        raise ValueError("All real-case stages require sixteen GPUs")
    for key in ("checkpoint", "source_runtime", "series_root"):
        if not isinstance(settings.get(key), str) or not Path(settings[key]).is_absolute():
            raise ValueError(f"{key} must be an explicit absolute path")
    repeated = settings.get("protocol") == REPEAT_PROTOCOL
    if settings.get("protocol") == TRAJECTORY_PROTOCOL:
        block = settings.get("full_trajectory", {})
        trajectory_reference_coefficient(block)
        balance = validate_reference_balance(block.get("reference_balance"))
        refit = validate_reward_refit(block.get("reward_refit"))
        classifier_kl = validate_classifier_kl(block.get('classifier_kl'))
        hard_trust = validate_classifier_trust(block.get('hard_trust_region'))
        if hard_trust is not None and classifier_kl is None:
            raise ValueError('Hard classifier trust requires coefficient-one classifier KL')
        if classifier_kl is not None and (refit is None or refit['ratio_bound'] is not None
                or balance is not None or trajectory_reference_coefficient(block) != 1.0):
            raise ValueError('Classifier KL requires a fresh unbounded reward and coefficient 1 without dynamic balancing')
        if balance is not None and trajectory_reference_coefficient(block) != 1.0:
            raise ValueError("Dynamic reference balance requires base reference_coefficient 1")
        if refit is not None and (balance is not None or trajectory_reference_coefficient(block) != 1.0):
            raise ValueError("Fresh ratio-bound ablation preserves reference coefficient 1 without dynamic balance")
        _integer(block.get("event_microbatch"), "full_trajectory.event_microbatch")
        for key in ("parity_atol", "parity_rtol"):
            value = block.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Full trajectory requires positive finite parity tolerances")
        if not Path(settings.get("native_training_directory", "")).is_absolute():
            raise ValueError("Declare the shared native_training_directory")
        if settings.get("ensemble_directory") is not None:
            raise ValueError("Full trajectory uses the inherited single head; remove ensemble_directory")
    if settings.get("protocol") not in (None, REPEAT_PROTOCOL, TRAJECTORY_PROTOCOL):
        raise ValueError("Unknown tau reward diagnostic protocol")
    if not repeated:
        if settings.get("trajectory_updates") != list(TRAJECTORY_UPDATES):
            raise ValueError("Preserve the predeclared 50/200/500 update budgets")
        if settings.get("relative_steps") != list(RELATIVE_STEPS):
            raise ValueError("Preserve the predeclared trajectory measurement points")
        if settings.get("audit_relative_steps") != list(AUDIT_STEPS):
            raise ValueError("Preserve the baseline and +50/+200/+500 audit endpoints")
    if settings.get("gradient_events_per_rank") != 512:
        raise ValueError("The gradient panel must match native 512 events per GPU")
    if settings.get("condition_groups") != 3 or settings.get("condition_edges") != [-1.0, -0.5, 0.5, 1.0]:
        raise ValueError("Use the three predeclared observed visible-opening-cosine strata")
    _integer(settings.get("gradient_repeats"), "gradient_repeats", 2)
    _integer(settings.get("bootstrap_replicates"), "bootstrap_replicates", 2)
    _integer(settings.get("bootstrap_seed"), "bootstrap_seed", 0)
    fractions = settings.get("direction_rms_fractions")
    if (not isinstance(fractions, list) or not fractions
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in fractions)):
        raise ValueError("direction_rms_fractions must contain finite positive numbers")
    if fractions != ([0.01, 0.03, 0.1] if repeated else [1.0]):
        raise ValueError("Preserve the predeclared signed parameter-RMS radii")
    edges = settings.get("noise_edges")
    if not isinstance(edges, list) or len(edges) != 4 or edges[0] != 0 or edges[-1] != 0.7:
        raise ValueError("Three noise bands must partition the native [0,0.7] range")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in edges):
        raise ValueError("noise_edges must be finite numbers")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise ValueError("noise_edges must be strictly increasing")
    seed_fields = (("evaluation_seeds", 2),) if repeated else (() if settings.get("protocol") == TRAJECTORY_PROTOCOL else (("ensemble_fit_seeds", 4),))
    for key, length in seed_fields:
        values = settings.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"{key} requires {length} distinct seeds")
        for value in values:
            _integer(value, key, 0)
        if len(set(values)) != length:
            raise ValueError(f"{key} requires {length} distinct seeds")
    if repeated:
        if type(settings.get("draw_repeats")) is not int or settings["draw_repeats"] != 4:
            raise ValueError("Repeated local-direction screening requires four native draws")
        if type(settings.get("gradient_repeats")) is not int or settings["gradient_repeats"] != 2:
            raise ValueError("Use two gradient sampling replicas with fixed candidates inside each draw")
        if settings.get("ensemble_directory") is not None:
            raise ValueError("Repeated local-direction screening uses only the inherited frozen head")
    elif settings.get("protocol") != TRAJECTORY_PROTOCOL:
        if settings.get("ensemble_members") != 4 or settings.get("ensemble_aggregation") != "mean_bounded_log_ratio":
            raise ValueError("Use four heads and average their bounded log-ratios")
        _integer(settings.get("ensemble_judge_seed"), "ensemble_judge_seed", 0)
        if settings["ensemble_judge_seed"] in settings["ensemble_fit_seeds"]:
            raise ValueError("The common judge requires an initialization independent of all four members")
    _integer(settings.get("diagnostic_evaluation_events"), "diagnostic_evaluation_events", 16)
    if settings["diagnostic_evaluation_events"] > 119002:
        raise ValueError("Diagnostic panel cannot exceed the complete filtered validation population")
    if repeated and settings["diagnostic_evaluation_events"] != 32768:
        raise ValueError("Repeated screening uses the same 32768 held-out identities")
    for key in ("reconstruction_tolerance",):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    directory = settings.get("ensemble_directory")
    if directory is not None and (not isinstance(directory, str) or not Path(directory).is_absolute()):
        raise ValueError("ensemble_directory must be null or an explicit absolute path")


def validate_native_runtime(cfg):
    validate_source_runtime(cfg)
    if cfg["platform"].get("batch_size") != 512:
        raise ValueError("Preserve the native 512 policy events per GPU")
    if cfg["dgpo"].get("normalize_advantages", False):
        raise ValueError("Preserve unscaled leave-one-out advantages")


def display_name(stage, *, arm="inherited", updates=50):
    names = {
        "diagnose": f"Why does reward uptake stall? | tau {arm} reward | condition and noise | step 1920",
        "repeat_diagnose": "Do local reward directions repeat? | frozen tau head | four draws | step 1920",
        "ensemble": "Is classifier noise limiting uptake? | tau attention | four fresh heads | step 1920",
        "trajectory": f"Can reward gains accumulate? | tau {arm} reward | {updates} native updates | step 1920",
    }
    if stage not in names:
        raise ValueError(f"No native run for stage {stage}")
    name = names[stage]
    if len(name) >= 96:
        raise ValueError("W&B display name must remain below 96 characters")
    return name


def configure(cfg, settings, metadata, *, pinned, output, stage="diagnose", arm="inherited",
              updates=50, ensemble_directory=None, method="native"):
    validate_settings(settings)
    validate_native_runtime(cfg)
    if stage not in NATIVE_STAGES:
        raise ValueError("Unknown native tau reward diagnostic stage")
    if (settings.get("protocol") == REPEAT_PROTOCOL) != (stage == "repeat_diagnose"):
        raise ValueError("repeat_diagnose requires its explicit opt-in configuration; preserve the legacy protocol")
    if arm not in REWARD_ARMS or updates not in TRAJECTORY_UPDATES:
        raise ValueError("Unknown reward arm or trajectory budget")
    if stage == "ensemble" and arm != "inherited":
        raise ValueError("The ensemble fitting stage starts with the inherited reward")
    if stage in ("diagnose", "trajectory") and arm != "inherited" and ensemble_directory is None:
        raise ValueError("Fresh-head arms require the saved ensemble directory")
    if stage == "repeat_diagnose" and (arm != "inherited" or ensemble_directory is not None):
        raise ValueError("repeat_diagnose preserves the inherited fixed head and does not load an ensemble")
    full_trajectory = settings.get("protocol") == TRAJECTORY_PROTOCOL
    if full_trajectory:
        if stage != "trajectory" or arm != "inherited" or ensemble_directory is not None or method not in ("native", "pathwise"):
            raise ValueError("Full trajectory requires trajectory stage, inherited reward and native/pathwise method, without ensemble artifacts")
        reference_coefficient = trajectory_reference_coefficient(settings["full_trajectory"])
        reference_balance = validate_reference_balance(settings["full_trajectory"].get("reference_balance"))
        reward_refit = validate_reward_refit(settings["full_trajectory"].get("reward_refit"))
        classifier_kl = validate_classifier_kl(settings['full_trajectory'].get('classifier_kl'))
        hard_trust = validate_classifier_trust(settings['full_trajectory'].get('hard_trust_region'))
        if method != "pathwise" and (reference_coefficient != 1.0 or reference_balance is not None or reward_refit is not None):
            raise ValueError("Reference scale ablation requires --method pathwise; native preserves coefficient 1")
    elif method != "native":
        raise ValueError("Pathwise method requires the explicit full-trajectory protocol")
    cfg = copy.deepcopy(cfg)
    dg, tau = cfg["dgpo"], cfg["dgpo"]["tau_ratio"]
    dg.update(checkpoint_load_mode="resume", auto_resume_from_last=False,
              auto_resume_fallback_checkpoint_path=None, auto_resume_best_source_checkpoint_dir=None,
              unbounded_training=False, checkpoint_transfer={"enabled": False})
    for name in ("gradient_conflict", "tarp", "gradient_transfer_trace"):
        if name in dg:
            dg[name]["enabled"] = False
    tau.update(classifier_only=False, classifier_attention_ablation=False,
               production_cross_attention=True, reward_cij_probe=False,
               output=str(output / "tau_diagnostics"), output_root=str(output))
    tau["transfer_probe"] = {"enabled": False}
    steps = [step for step in RELATIVE_STEPS if step <= updates] if stage == "trajectory" else [0]
    audits = ([step for step in AUDIT_STEPS if step <= updates] if stage == "trajectory"
              else [] if stage == "repeat_diagnose" else [0])
    artifacts = Path(ensemble_directory) if ensemble_directory is not None else output / "ensemble"
    tau["mechanism_probe"] = {
        "enabled": True, "mode": stage, "source_step": SOURCE_STEP,
        "reward_arm": arm, "updates": updates if stage == "trajectory" else 0,
        "relative_steps": steps, "audit_relative_steps": audits,
        "expected_source_reward_round": metadata["inherited_reward_round"],
        "expected_source_denominator_step": metadata["inherited_denominator_step"],
        "output_directory": str(output / "mechanisms"),
        "ensemble_directory": str(artifacts),
        "evaluation_events": settings["diagnostic_evaluation_events"] if stage in ("diagnose", "repeat_diagnose") else 119002,
        "reference_recenter": False, "periodic_refits": False,
        "source_checkpoint": metadata.get("source_checkpoint", settings["checkpoint"]),
        **{key: copy.deepcopy(settings[key]) for key in (
            "gradient_events_per_rank", "condition_groups", "condition_edges", "noise_edges", "gradient_repeats",
            "bootstrap_replicates", "bootstrap_seed", "reconstruction_tolerance",
            "direction_rms_fractions")},
    }
    probe = tau["mechanism_probe"]
    if stage == "repeat_diagnose":
        probe.update(draw_repeats=settings["draw_repeats"], evaluation_seeds=list(settings["evaluation_seeds"]),
                     evaluator="installed_training_head", persistent_updates=0, ensemble_directory=None)
    elif not full_trajectory:
        probe.update({key: copy.deepcopy(settings[key]) for key in (
            "ensemble_members", "ensemble_fit_seeds", "ensemble_judge_seed", "ensemble_aggregation")})
    cfg["options"]["Training"].update(model_checkpoint_load_path=str(pinned), pretrain_model_load_path=None,
                                        model_checkpoint_save_path=str(output / "checkpoints"))
    cfg.setdefault("compat", {}).update(backend="dgpo-evenet", repo_root=str(REPO))
    cfg.setdefault("rl", {})["enabled"] = True
    cfg.setdefault("logger", {}).setdefault("local", {})["save_dir"] = str(output / "logs")
    wb = cfg["logger"]["wandb"] if cfg["logger"].get("wandb", {}).get("project") else cfg["wandb"]
    name = display_name(stage, arm=arm, updates=updates)
    wb.update(id=None, resume="never", fresh_run=True, run_name=name, name=name,
              group=settings.get("wandb_group", "Tau reward mechanisms"))
    wb["tags"] = list(dict.fromkeys([
        *[tag for tag in wb.get("tags", []) if not str(tag).lower().startswith("refit")],
        "TauRewardMechanisms", "Step1920", "InheritedReference", "InheritedAdamW", "RawWeights",
        "SixteenGPU", stage, arm,
    ]))
    if "wandb" in cfg and cfg["wandb"] is not wb:
        cfg["wandb"].update(id=None, resume="never", fresh_run=True, run_name=name, name=name, group=wb["group"])
    nersc = cfg.setdefault("nersc", {})
    nersc.setdefault("ray", {})["results_dir"] = str(output / "ray_results")
    nersc["submit"] = False
    nersc.setdefault("execution", {})["command"] = (
        "shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py "
        + ("config/tau_reward_direction_repeats_1920.yaml" if stage == "repeat_diagnose" else "config/tau_reward_mechanisms_1920.yaml")
        + f" {stage}"
    )
    nersc["reproducibility"] = {"source_checkpoint": str(pinned), "source_policy_step": SOURCE_STEP,
        "note": "Full-state counterfactual continuation; inherited reference, AdamW and scheduler; original RNG/data iterator absent."}
    experiment = cfg.setdefault("experiment", {})
    experiment.pop("actor_updates", None)
    experiment.update(classifier_only=False, tau_classifier_only=False, protocol="tau-reward-mechanisms",
        stage=stage, reward_arm=arm, relative_updates=updates if stage == "trajectory" else 0,
        source_policy_step=SOURCE_STEP, source_metadata=copy.deepcopy(metadata),
        first_head="inherited" if arm == "inherited" else "saved fresh member0" if arm == "member0" else "saved four-head ensemble",
        reference_mode="inherited source reference without recentering",
        continuation_scope="Counterfactual continuation; original RNG/data iterator were not checkpointed",
        primary="paired held-out all-K reward change under the common independent judge, with event-bootstrap intervals",
        physics="raw polarimeter Cij total/diagonal/offdiagonal/component closure; fixed-energy reconstruction scope; entanglement not directly inferred",
        predeclared_hypotheses=["reference conflict", "condition gradient cancellation", "noise gradient conflict",
                               "longer accumulation", "classifier estimator noise"])
    experiment["native_training_inputs"] = (
        "Exact shared recorded native per-rank tensor batches and order; partial tails preserved"
        if ensemble_directory is not None else "Live Ray batches; training inputs not paired across separate launches"
    )
    experiment["native_mc_seed_recipe"] = "bootstrap_seed + rank + relative_update_index * 100003"
    if stage == "repeat_diagnose":
        experiment.update(protocol=REPEAT_PROTOCOL,
            primary="paired held-out all-K mean reward gain under the inherited frozen installed head",
            evaluator="installed_training_head; no independent judge or fresh classifier fit",
            draws=settings["draw_repeats"], evaluation_seeds=list(settings["evaluation_seeds"]),
            native_training_inputs="Four separately sampled nonoverlapping native shuffled chunks; fresh candidates for each draw",
            native_mc_seed_recipe="bootstrap_seed + draw_index * 1000003 + rank + gradient_replica_index * 100003",
            direction_gradient_replica=0,
            gradient_repeat_role="All three directions use replica 0; replica 1 is a fixed-candidate gradient stability check, not averaging",
            persistent_updates=0,
            uncertainty="Separate native update draws, two evaluation MC seeds, and paired event-bootstrap intervals; four draws are screening, not proof",
            purpose="Check whether small signed raw reward/total/native AdamW directions reproduce local reward uptake",
            predeclared_hypotheses=["local signed direction actionability and finite-radius response"],
            untested="Condition/noise capacity, classifier fitting noise, independent distribution closure and longer accumulation")
    if full_trajectory:
        dg["reference_trust"]["coefficient"] = reference_coefficient
        if classifier_kl is not None:
            dg['reference_trust'].update(enabled=False, coefficient=0.0)
        probe["full_trajectory"] = dict(settings["full_trajectory"], method=method,
                                        reference_coefficient=reference_coefficient)
        probe["ensemble_directory"] = None
        probe["native_training_directory"] = settings["native_training_directory"]
        dg["log_parameter_update_rms"] = True
        dg["fail_on_skipped_optimizer_step"] = True
        name = f"Can trajectory gradients retain reward? | {method} | DDIM20 | step 1920"
        if reference_coefficient != 1.0:
            name = f"Does stronger reference control drift? | pathwise | ref={reference_coefficient:g} DDIM20 | step 1920"
        if reference_balance is not None:
            name = "Does dynamic reference control drift? | pathwise | EMA norms DDIM20 | step 1920"
        if reward_refit is not None:
            bound_label = "unbounded" if reward_refit['ratio_bound'] is None else "bound30 control"
            name = f"Does removing the ratio bound help? | pathwise | {bound_label} | anchor 1920"
        if classifier_kl is not None:
            name = 'Does classifier KL retain truth alignment? | pathwise | coef1 | anchor 1920'
        if hard_trust is not None:
            name = f"Does a hard KL gate control drift? | pathwise | KL<={hard_trust['max_kl']:g} | anchor 1920"
        if len(name) >= 96:
            raise ValueError("W&B display name must remain below 96 characters")
        for logger in (cfg["logger"].get("wandb", {}), cfg.get("wandb", {})):
            if logger:
                logger.update(name=name, run_name=name, group="Tau full trajectory", id=None, resume="never", fresh_run=True)
        experiment.update(protocol=TRAJECTORY_PROTOCOL, method=method,
            full_trajectory=dict(probe["full_trajectory"]),
            primary=f"Same inherited-head paired all-K mean reward gain at +{updates}; fresh audit and spin closure assess independent transfer",
            native_training_inputs="Shared recorded native per-rank batches; capture once at startup if absent; no classifier fit",
            evaluator="same inherited training classifier, not an independent judge",
            predeclared_hypotheses=["Full-DDIM pathwise reward uptake versus native DGPO"],
            rollout="Same K8 chunked training-mode rollout in both arms; exact noise/dropout replay for pathwise backward",
            objective=(f"negative mean bounded tau log-ratio plus coefficient-{reference_coefficient:g} velocity-MSE" if method == "pathwise"
                       else "native DGPO plus inherited coefficient-1 velocity-MSE"),
            reference_coefficient=reference_coefficient,
            reference_balance=reference_balance,
            reference_scale_scope="Fixed coefficient ablation; no gradient normalization or projection; not exact distribution KL",
            reference_gradient="Native eight time draws on detached candidates; no derivative through reference sampling",
            gradient_scope="Full 20-step sampler; not DPO or exact distribution KL; inherited AdamW state retained",
            compute_scope="Equal updates and events, not equal compute; local component norms are rank-local")
        nersc["execution"]["command"] = ("shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py "
            f"config/tau_full_trajectory_1920.yaml trajectory --method {method} --updates {updates}")
        if reference_balance is not None:
            experiment.update(
                objective="negative mean bounded tau log-ratio plus detached EMA-global-norm-weighted velocity-MSE gradient",
                reference_scale_scope="Current-step global component norms; EMA, bounds and rate limit; not exact distribution KL",
                predeclared_hypotheses=["Dynamic gradient scale balance controls pathwise drift while retaining reward uptake"])
        if reward_refit is not None:
            bound_label = "unbounded" if reward_refit['ratio_bound'] is None else "bounded"
            experiment.update(
                first_head=f"fresh {bound_label} paired-BCE head against source policy step1920",
                reference_mode="One startup recenter to step1920, then frozen",
                objective=f"negative mean {bound_label} tau log-ratio plus coefficient-1 velocity-MSE",
                evaluator="own fresh frozen training classifier; raw reward incomparable across ratio-bound arms",
                primary=f"Fresh independent audit and full Cij closure at +{updates}; own-head reward gain is supporting evidence",
                native_training_inputs="Shared recorded native batches; one balanced startup classifier fit on full filtered train panel",
                reward_refit=reward_refit,
                predeclared_hypotheses=["Removing the ratio bound changes final-finetuning transfer at matched step1920 anchor"],
                gradient_scope="Full 20-step pathwise sampler; one startup refit; velocity-MSE remains a proxy, not exact KL")
            for logger in (cfg["logger"].get("wandb", {}), cfg.get("wandb", {})):
                if logger:
                    logger['tags'] = [tag for tag in logger.get('tags', []) if tag != 'InheritedReference']
                    logger['tags'] += ['FreshStep1920Reward', 'Step1920Reference', bound_label]
            nersc['reproducibility']['note'] = 'Inherited actor/AdamW/scheduler; explicit fresh truth/step1920 reward and matching step1920 velocity reference.'
        if classifier_kl is not None:
            experiment.update(objective='mean(current/anchor raw logit - truth/anchor raw logit); coefficient 1',
                classifier_kl=classifier_kl, velocity_penalty_enabled=False,
                reference_gradient='Current/anchor KL classifier input VJP through the same full DDIM sampler as reward',
                reference_scale_scope='Fixed coefficient1, same all-K/event normalization; no gradient balancing',
                gradient_scope='Feature-space ratio approximation; not exact full conditional KL or a hard trust region',
                native_training_inputs='Shared native batches; full filtered paired train panel for independent per-update KL fits',
                predeclared_hypotheses=['Classifier KL controls reward-driven drift with the truth-target coefficient1 objective'])
            nersc['reproducibility']['note'] = 'Frozen step1920 reward denominator/anchor; fresh current-vs-anchor KL head before every update; native velocity penalty disabled.'
        if hard_trust is not None:
            experiment.update(hard_trust_region=hard_trust,
                gradient_scope='Same coefficient1 feature-space truth-target objective; additional hard estimated-KL feasibility gate',
                trust_estimator='Complete held-out filtered validation; unclamped current/anchor logit mean + event-SE bound; excludes classifier bias',
                trust_rejection=f"Scale the exact AdamW displacement by {hard_trust['backtrack_factor']:g}, up to {hard_trust['max_backtracks']} backtracks; restore parameters/moments on failure and stop without advancing clocks",
                classifier_fit_reuse='Accepted candidate head reused for the next policy gradient; rejected heads never installed',
                primary=f'Fresh independent audit and Cij closure at +{updates} accepted updates; early boundary stop is an incomplete endpoint',
                predeclared_hypotheses=['Hard classifier-KL acceptance gate controls cumulative anchor drift with coefficient1 truth-target objective'])
    return cfg


def command_for_runtime(runtime, output, *, stage="diagnose", updates=50):
    if stage not in NATIVE_STAGES:
        raise ValueError("No trainer command for this stage")
    if updates not in TRAJECTORY_UPDATES:
        raise ValueError("Preserve the predeclared trajectory budgets")
    # Diagnosis intercepts the first batch before the optimizer. Entering that
    # callback requires a cap above the source clock, even though no step applies.
    stop = SOURCE_STEP + updates if stage == "trajectory" else SOURCE_STEP + 1 if stage in ("diagnose", "repeat_diagnose") else SOURCE_STEP
    return [sys.executable, "-u", str(EVENET / "RL/DGPO_neutrino/dgpo_trainer.py"), str(runtime),
            "--max-steps", str(stop), "--ray-dir", str(output / "ray_results")]


def resolve_ensemble_directory(settings, override, *, arm, stage):
    directory = override or settings.get("ensemble_directory")
    if settings.get("protocol") == TRAJECTORY_PROTOCOL:
        if directory is not None or arm != "inherited":
            raise ValueError("Full trajectory uses the inherited single head; remove --ensemble-directory and use --arm inherited")
        return None
    if stage == "repeat_diagnose" and (directory is not None or arm != "inherited"):
        raise ValueError("repeat_diagnose uses the inherited head without ensemble artifacts")
    if stage not in ("diagnose", "trajectory"):
        if override is not None:
            raise ValueError("--ensemble-directory belongs to diagnose or trajectory")
        return None
    if directory is None:
        if arm == "inherited":
            return None
        raise ValueError("Pass --ensemble-directory from the completed ensemble stage")
    if str(directory) == "unique":
        matches = []
        for invocation in Path(settings.get("ensemble_series_root", settings["series_root"])).glob("ensemble-*"):
            report = invocation / "mechanisms" / "ensemble_diagnostic.json"
            candidate = invocation / "ensemble"
            if not report.is_file() or json.loads(report.read_text()).get("complete") is not True:
                continue
            try:
                validate_native_training_manifest(candidate, source_checkpoint=str(Path(settings["checkpoint"]).resolve()))
            except (ValueError, OSError):
                continue
            if (candidate / "ensemble.pt").is_file():
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(f"Found {len(matches)} completed ensembles; pass the explicit ensemble directory to select one")
        directory = matches[0]
    directory = Path(directory)
    if not directory.is_absolute():
        raise ValueError("The ensemble artifact directory must be absolute")
    directory = directory.resolve(strict=True)
    root = Path(settings.get("ensemble_series_root", settings["series_root"])).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Ensemble artifacts must belong to this declared series_root")
    if not (directory / "ensemble.pt").is_file():
        raise ValueError("The selected ensemble stage has no completed ensemble.pt artifact")
    validate_native_training_manifest(directory, source_checkpoint=str(Path(settings["checkpoint"]).resolve()))
    return directory


def validate_native_training_manifest(ensemble_directory, *, source_checkpoint=None):
    """Validate the complete shared native batch inventory without Torch/Ray.

GPU workers additionally load and fingerprint-check every preserved native
tensor field. This preflight catches unfinished capture before starting them.
"""
    return validate_native_training_directory(Path(ensemble_directory) / "native_training", source_checkpoint=source_checkpoint)


def validate_native_training_directory(folder, *, source_checkpoint=None):
    folder = Path(folder)
    path = folder / "manifest.json"
    if not path.is_file():
        raise ValueError("The ensemble stage has no completed native_training/manifest.json")
    manifest = json.loads(path.read_text())
    expected = {"kind": "tau_native_training_batch_replay", "schema_version": 1,
        "complete": True, "source_policy_step": SOURCE_STEP, "world": 16,
        "expected_events": 416701, "global_rows": 416701, "batch_size": 512,
        "actor_updates": 0, "all_native_tensor_fields_preserved": True}
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Native training capture must be complete step1920/416701-event/16-rank/batch512")
    if manifest.get("complete") is not True or manifest.get("all_native_tensor_fields_preserved") is not True:
        raise ValueError("Native training replay must explicitly preserve all tensor fields")
    if not isinstance(manifest.get("source_checkpoint"), str) or not Path(manifest["source_checkpoint"]).is_absolute():
        raise ValueError("Native training replay is missing its exact original source checkpoint path")
    if source_checkpoint is not None and manifest.get("source_checkpoint") != str(source_checkpoint):
        raise ValueError("Native training replay must preserve the exact original source checkpoint path")
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 16:
        raise ValueError("Native training manifest requires all sixteen rank inventories")
    seed = manifest.get("base_shuffle_seed")
    _integer(seed, "native training base_shuffle_seed", 0)
    for rank, row in enumerate(ranks):
        if not isinstance(row, dict) or row.get("rank") != rank or row.get("error") is not None:
            raise ValueError("Native training replay rank inventory is inconsistent")
        sizes = row.get("batch_sizes")
        if not isinstance(sizes, list) or not sizes or any(type(size) is not int or not 0 < size <= 512 for size in sizes):
            raise ValueError("Native replay batches must preserve positive full or partial native sizes")
        if (row.get("batches") != len(sizes) or row.get("rows") != sum(sizes)
                or row.get("partial_batch_indices") != [index for index, size in enumerate(sizes) if size < 512]
                or row.get("local_shuffle_seed") != seed + rank
                or len(row.get("batch_fingerprints", [])) != len(sizes)):
            raise ValueError("Native training replay changed native batch counts, tails or order metadata")
        if not (folder / f"rank-{rank:02d}.pt").is_file():
            raise ValueError("Native training replay is missing a recorded rank shard")
    if sum(row["rows"] for row in ranks) != 416701:
        raise ValueError("Native replay rank rows do not cover the complete filtered population")
    if (manifest.get("minimum_rank_batches") != min(row["batches"] for row in ranks)
            or manifest.get("maximum_rank_batches") != max(row["batches"] for row in ranks)
            or not isinstance(manifest.get("global_replay_fingerprint"), str)
            or not manifest["global_replay_fingerprint"]
            or not isinstance(manifest.get("tail_recipe"), str) or not manifest["tail_recipe"]):
        raise ValueError("Native replay manifest is missing its order/tail provenance")
    return {key: copy.deepcopy(manifest[key]) for key in (
        "kind", "schema_version", "source_policy_step", "source_checkpoint", "world", "global_rows", "batch_size",
        "base_shuffle_seed", "global_replay_fingerprint", "tail_recipe",
    )}


def validate_ensemble_metadata(payload, metadata):
    """Cheap artifact preflight before the trainer creates its GPU workers.

The worker loader additionally compares every preprocessing tensor and builds
the classifiers with strict state loading against the live saved reward.
"""
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in (
        ("kind", "tau_classifier_ensemble_diagnostic"), ("schema_version", 1),
        ("complete", True), ("source_policy_step", SOURCE_STEP), ("fresh_denominator_step", SOURCE_STEP),
        ("single_control_member", 0), ("aggregation", "mean_training_time_bounded_log_ratio"),
        ("training_population", 416701), ("training_candidates", 1),
    )):
        raise ValueError("Select a complete matched step1920 four-head artifact")
    if payload.get("inherited_denominator_step") != metadata["inherited_denominator_step"]:
        raise ValueError("Ensemble was fitted against a different inherited denominator")
    clocks = payload.get("inherited_clocks", {})
    for key, expected in (("round_id", metadata["inherited_reward_round"]),
                          ("denominator_step", metadata["inherited_denominator_step"]),
                          ("last_refit_epoch", metadata["inherited_last_refit_epoch"])):
        if clocks.get(key) != expected:
            raise ValueError("Ensemble changed the source reward clock: " + key)
    contract = payload.get("reward_contract", {})
    expected = {"normalization_file": metadata["normalization_file"],
                "source_checkpoint": metadata["classifier_trunk_source_checkpoint"],
                "policy_conditioning_contract": POLICY_CONTRACT}
    if any(str(contract.get(key)) != str(value) for key, value in expected.items()):
        raise ValueError("Ensemble normalization, feature trunk or policy label0 contract changed")
    members = payload.get("members")
    if not isinstance(members, list) or len(members) != 4 or not isinstance(payload.get("judge"), dict):
        raise ValueError("Require four reward members and one separately fitted common judge")
    fitted = [*members, payload["judge"]]
    seeds = [member.get("seed") for member in fitted]
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 5:
        raise ValueError("Members and judge must have five distinct seeds")
    for member in fitted:
        fit = member.get("fit", {})
        if fit.get("minimum_fit_steps_met") is not True or type(fit.get("total_steps")) is not int or fit["total_steps"] < 1000:
            raise ValueError("An ensemble member or common judge is undertrained")


def _report_summary(path):
    report = json.loads(path.read_text())
    result = {key: copy.deepcopy(report[key]) for key in (
        "complete", "source_step", "policy_step", "actor_updates", "scope", "conclusion", "full_trajectory",
        "primary_endpoint", "judge_scope", "reward_arm", "reference_recentered",
        "centered_member_variance", "ensemble_to_single_advantage_rms",
        "actual_updates", "evaluator", "comparable_across_arms", "inherited_reference", "inherited_adamw",
        "native_training_inputs",
        "native_training_directory",
        "draw_repeats", "completed_draws", "evaluation_seeds", "gradient_repeats", "direction_rms_fractions",
        "uncertainty", "screening", "aggregate", "summary", "perturbation_validity", "condition_scope",
    ) if key in report}
    measurements = report.get("measurements")
    if isinstance(measurements, dict) and measurements:
        numeric = sorted((int(key), key) for key in measurements if str(key).isdigit())
        if numeric:
            relative, key = numeric[-1]
            point = measurements[key]
            result["last_relative_step"] = relative
            result["measured_relative_steps"] = [relative for relative, _ in numeric]
            result["endpoint"] = {key: copy.deepcopy(point[key]) for key in (
                "reward", "training_reward", "judge_reward", "fresh_audit", "events", "candidates",
            ) if key in point}
            cij = point.get("cij", {})
            result["endpoint"]["cij"] = {key: copy.deepcopy(cij[key]) for key in (
                "error", "diagonal_error", "offdiagonal_error", "nn_error", "error_changes",
            ) if key in cij}
    arms = report.get("arms")
    if isinstance(arms, dict):
        result["arms"] = {}
        for name, arm in arms.items():
            brief = {key: copy.deepcopy(arm[key]) for key in ("zero_replay_max_abs_error", "native_parameter_update_rms") if key in arm}
            brief["gradient_replicas"] = [{
                "reconstruction": replica.get("reconstruction"),
                "reward_reference": replica.get("reward_reference"),
                "reward_conditions": replica.get("components", {}).get("reward", {}).get("conditions"),
                "reward_noise_bands": replica.get("components", {}).get("reward", {}).get("noise_bands"),
            } for replica in arm.get("gradient_replicas", [])]
            stability = arm.get("sampling_stability", {})
            brief["sampling_stability"] = {"replicas": stability.get("replicas"),
                "interpretation": stability.get("interpretation"),
                "totals": {component: values.get("total") for component, values in stability.get("components", {}).items()}}
            brief["interventions"] = {key: {
                "parameter_update_rms": effect.get("parameter_update_rms"),
                "judge_reward": effect.get("reward"),
                "cij_error_changes": effect.get("cij", {}).get("error_changes"),
                "condition_reward": {condition: value.get("reward")
                                     for condition, value in effect.get("condition_transfer", {}).items()},
            } for key, effect in arm.get("interventions", {}).items()}
            result["arms"][name] = brief
    return result


def _paired_cross_arm(settings, invocations):
    """Reuse event-paired statistics only when saved identities/judge agree."""
    comparisons = []
    eligible = [record for record in invocations if record["stage"] == "trajectory"
                and (record.get("ensemble_directory") or (record.get("method") and record.get("native_training_directory")))
                and record.get("endpoint_measurements")
                and record.get("source_checkpoint") and type(record.get("validation_seed")) is int
                and record.get("candidates") == 8 and record.get("native_replay_valid") is True]
    for index, before in enumerate(eligible):
        for after in eligible[index+1:]:
            if bool(before.get("method")) != bool(after.get("method")):
                continue
            before_coefficient = after_coefficient = 1.0
            before_balance = after_balance = None
            if before.get("method"):
                left_contract = dict(before.get("trajectory_contract") or {})
                right_contract = dict(after.get("trajectory_contract") or {})
                before_coefficient = trajectory_reference_coefficient(left_contract)
                after_coefficient = trajectory_reference_coefficient(right_contract)
                left_contract.pop("reference_coefficient", None)
                right_contract.pop("reference_coefficient", None)
                before_balance = left_contract.pop("reference_balance", None)
                after_balance = right_contract.pop("reference_balance", None)
                if left_contract != right_contract:
                    continue
                if before.get("method") != after.get("method") and (before_coefficient != after_coefficient or before_balance != after_balance):
                    continue  # Do not combine method and coefficient interventions.
            if ((before["reward_arm"], before.get("method")) == (after["reward_arm"], after.get("method"))
                    and before_coefficient == after_coefficient and before_balance == after_balance):
                continue
            compatible = all(before.get(key) == after.get(key) for key in (
                "source_checkpoint", "ensemble_directory", "last_relative_step", "validation_seed", "candidates",
                "native_replay_fingerprint", "native_training_directory",
            ))
            if not compatible:
                continue
            # Import numerical dependencies only when executed result artifacts
            # actually make a paired comparison possible.
            import numpy as np
            if str(EVENET) not in sys.path:
                sys.path.insert(0, str(EVENET))
            from RL.DGPO_neutrino.tau_reward_transfer import paired_statistics
            shared_head = bool(before.get("method")) and not before.get("ensemble_directory")
            reward_key = "reward" if shared_head else "judge_reward"
            required = ("source_ids", "weight", "truth_cij", "generated_cij", reward_key)
            paths = (before["endpoint_measurements"], after["endpoint_measurements"])
            with np.load(paths[0], allow_pickle=False) as left, np.load(paths[1], allow_pickle=False) as right:
                if any(key not in file for file in (left, right) for key in required):
                    comparisons.append({"before": before["output"], "after": after["output"],
                        "status": "unavailable", "reason": f"Shared evaluator measurement arrays ({reward_key}) were not saved"})
                    continue
                if any(not np.array_equal(left[key], right[key]) for key in ("source_ids", "weight", "truth_cij")):
                    raise ValueError("Cannot pair trajectory arms with different validation identities/weights/truth")
                result = paired_statistics(left[reward_key], right[reward_key],
                    left["generated_cij"], right["generated_cij"], left["truth_cij"], left["weight"],
                    replicates=settings["bootstrap_replicates"], seed=settings["bootstrap_seed"])
            comparisons.append({"before": before["output"], "after": after["output"],
                "before_arm": before["reward_arm"], "after_arm": after["reward_arm"],
                "before_method": before.get("method"), "after_method": after.get("method"),
                "before_reference_coefficient": before_coefficient,
                "after_reference_coefficient": after_coefficient,
                "before_reference_balance": before_balance, "after_reference_balance": after_balance,
                "intervention": ("reference_balance" if before_balance != after_balance else
                                 "reference_coefficient" if before_coefficient != after_coefficient else "reward_update"),
                "relative_step": before["last_relative_step"], "status": "paired",
                ("inherited_head_reward" if shared_head else "judge_reward"): result["reward"],
                "reward_evaluator": "same_inherited_training_head" if shared_head else "common_independent_judge",
                "cij_error_changes": result["cij"]["error_changes"],
                "scope": "After minus before, explicitly labeled shared evaluator, exact shared native training batches/order and paired MC recipe; CRN validation/event-bootstrap uncertainty, not training-seed uncertainty"})
    return comparisons


def summarize(settings):
    """Read recorded local artifacts only; never infer results from configuration."""
    root = Path(settings["series_root"])
    invocations = []
    if root.is_dir():
        for manifest in sorted(root.glob(f"*/{MANIFEST}")):
            record = json.loads(manifest.read_text())
            folder = manifest.parent
            names = ("gradient_diagnosis.json", "local_direction_repeats.json", "ensemble_diagnostic.json", "transfer_report.json", "report.json")
            summaries = [folder / "mechanisms" / name for name in names if (folder / "mechanisms" / name).is_file()]
            item = {"output": str(folder), "stage": record["stage"],
                "reward_arm": record["reward_arm"], "relative_updates": record["relative_updates"],
                "method": record.get("method"), "trajectory_contract": record.get("trajectory_contract"),
                "native_training_directory": record.get("native_training_directory"),
                "prepared_only": record["prepared_only"],
                "result_files": [str(path) for path in sorted(summaries)],
                "execution_evidence": "result artifact present" if summaries else "no result artifact recorded",
                **{key: record.get(key) for key in ("source_checkpoint", "ensemble_directory", "validation_seed", "candidates")},
                "reports": {path.name: _report_summary(path) for path in summaries}}
            replay = record.get("native_training_replay")
            item["native_replay_valid"] = False
            if isinstance(replay, dict) and (record.get("ensemble_directory") or record.get("native_training_directory")):
                try:
                    replay_directory = record.get("native_training_directory") or str(Path(record["ensemble_directory"]) / "native_training")
                    current_replay = validate_native_training_directory(replay_directory,
                        source_checkpoint=record.get("source_checkpoint"))
                    item["native_replay_valid"] = current_replay == replay
                    item["native_replay_fingerprint"] = replay.get("global_replay_fingerprint")
                    item["native_replay_status"] = "validated" if item["native_replay_valid"] else "saved replay metadata changed"
                except (ValueError, OSError) as exc:
                    item["native_replay_status"] = str(exc)
            transfer = next((item["reports"][name] for name in ("transfer_report.json", "report.json")
                             if name in item["reports"] and "last_relative_step" in item["reports"][name]), None)
            if transfer is not None:
                item["last_relative_step"] = transfer["last_relative_step"]
                measurements = folder / "mechanisms" / f"step-{item['last_relative_step']:02d}" / "measurements.npz"
                if measurements.is_file():
                    item["endpoint_measurements"] = str(measurements)
            invocations.append(item)
    return {"series_root": str(root), "source_step": SOURCE_STEP, "invocations": invocations,
        "paired_cross_arm": _paired_cross_arm(settings, invocations),
        "scope": "Artifact inventory; prepared configurations are not executed evidence and do not establish any hypothesis."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("stage", choices=STAGES, nargs="?", default="prepare")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and pin only; no trainer/Ray")
    parser.add_argument("--target-stage", choices=NATIVE_STAGES, default=None,
                        help="Native runtime prepared by stage prepare")
    parser.add_argument("--arm", choices=REWARD_ARMS, default="inherited")
    parser.add_argument("--method", choices=("native", "pathwise"), default="native",
                        help="Full-trajectory protocol only: matched native control or complete DDIM reward backward")
    parser.add_argument("--updates", type=int, choices=TRAJECTORY_UPDATES, default=50)
    parser.add_argument("--ensemble-directory", type=Path,
                        help="Exact completed artifact directory, or 'unique' when this series has one completed ensemble")
    args = parser.parse_args(argv)
    settings = read_mapping(args.config)
    validate_settings(settings)
    if args.stage == "summarize":
        print(json.dumps(summarize(settings), indent=2), flush=True)
        return
    stage = (args.target_stage or ("repeat_diagnose" if settings.get("protocol") == REPEAT_PROTOCOL else "trajectory" if settings.get("protocol") == TRAJECTORY_PROTOCOL else "diagnose")) if args.stage == "prepare" else args.stage
    if settings.get("protocol") == TRAJECTORY_PROTOCOL and args.ensemble_directory is not None:
        raise ValueError("Full trajectory needs no ensemble: remove --ensemble-directory")
    ensemble_directory = resolve_ensemble_directory(settings, args.ensemble_directory, arm=args.arm, stage=stage)
    if settings.get("protocol") == TRAJECTORY_PROTOCOL:
        if stage != "trajectory" or args.arm != "inherited":
            raise ValueError("Full trajectory needs trajectory/inherited")
    elif args.method != "native":
        raise ValueError("Use the full-trajectory configuration for --method pathwise")
    source = Path(settings["checkpoint"]).resolve(strict=True)
    source_runtime = Path(settings["source_runtime"]).resolve(strict=True)
    cfg = read_mapping(source_runtime)
    validate_native_runtime(cfg)
    sys.path[:0] = [str(REPO), str(EVENET)]
    from evenet.dataset.filtered_data import validate_filtered_dataset
    manifests = {}
    for key, rows in (("data_parquet_dir", 416701), ("data_parquet_val_dir", 119002)):
        manifests[key] = validate_filtered_dataset(cfg["platform"][key])
        if manifests[key]["rows"] != rows:
            raise ValueError("Filtered dataset population changed: " + key)
    replay_metadata = None
    if settings.get("protocol") == TRAJECTORY_PROTOCOL:
        folder = Path(settings["native_training_directory"])
        if folder.exists():
            replay_metadata = validate_native_training_directory(folder, source_checkpoint=str(source))
            if replay_metadata['base_shuffle_seed'] != settings['bootstrap_seed']:
                raise ValueError("Shared replay shuffle seed differs from this configuration")
    root = Path(settings["series_root"])
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{stage}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-"), dir=root.resolve()))
    pinned = output / "source.ckpt"
    try:
        os.link(source, pinned)
        pin_method = "hard_link"
    except OSError:
        shutil.copy2(source, pinned)
        pin_method = "copy"
    shutil.copy2(source_runtime, output / "source_runtime.yaml")
    import torch
    checkpoint = torch.load(pinned, map_location="cpu", weights_only=False, mmap=True)
    metadata = source_metadata(checkpoint)
    del checkpoint
    if ensemble_directory is not None:
        ensemble_payload = torch.load(ensemble_directory / "ensemble.pt", map_location="cpu", weights_only=True)
        validate_ensemble_metadata(ensemble_payload, metadata)
        del ensemble_payload
        metadata["native_training_replay"] = validate_native_training_manifest(ensemble_directory, source_checkpoint=str(source))
    if replay_metadata is not None:
        metadata["native_training_replay"] = replay_metadata
    metadata.update(source_checkpoint=str(source), pinned_checkpoint=str(pinned), source_runtime=str(source_runtime),
                    pinned_runtime=str(output / "source_runtime.yaml"), pin_method=pin_method, filtered_manifests=manifests)
    configured = configure(cfg, settings, metadata, pinned=pinned, output=output, stage=stage,
                           arm=args.arm, updates=args.updates, ensemble_directory=ensemble_directory, method=args.method)
    # Persist the actual opt-in config path in reviewable launch metadata.
    configured['nersc']['execution']['command'] = (
        f"shifter python3 -u scripts/diagnose_tau_reward_mechanisms.py {args.config} {stage}"
        f" --method {args.method} --updates {args.updates}"
        + (f" --arm {args.arm} --ensemble-directory {ensemble_directory}" if ensemble_directory is not None else ""))
    runtime = output / "runtime.yaml"
    runtime.write_text(yaml.safe_dump(configured, sort_keys=False))
    (output / "source_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    command = command_for_runtime(runtime, output, stage=stage, updates=args.updates)
    prepared_only = args.prepare_only or args.stage == "prepare"
    invocation = {"schema_version": 1, "stage": stage, "source_step": SOURCE_STEP,
        "reward_arm": args.arm, "relative_updates": args.updates if stage == "trajectory" else 0,
        "prepared_only": prepared_only, "source_checkpoint": str(source), "runtime": str(runtime),
        "validation_seed": configured["dgpo"]["tau_ratio"].get("validation_seed"), "candidates": 8,
        "native_training_replay": metadata.get("native_training_replay"),
        "wandb_name": display_name(stage, arm=args.arm, updates=args.updates),
        "ensemble_directory": configured["dgpo"]["tau_ratio"]["mechanism_probe"]["ensemble_directory"]}
    if settings.get("protocol") == TRAJECTORY_PROTOCOL:
        invocation.update(method=args.method, native_training_directory=settings["native_training_directory"],
            trajectory_contract=dict(settings["full_trajectory"],
            bootstrap_seed=settings["bootstrap_seed"]),
            wandb_name=configured["experiment"]["method"])
        wb_config = configured["logger"].get("wandb", {}) or configured.get("wandb", {})
        invocation["wandb_name"] = wb_config["run_name"]
    (output / MANIFEST).write_text(json.dumps(invocation, indent=2) + "\n")
    print(json.dumps({**invocation, "output": str(output), "source": metadata, "workers": 16,
        "heldout_conditions": configured["dgpo"]["tau_ratio"]["mechanism_probe"]["evaluation_events"], "candidates_per_condition": 8,
        "relative_steps": configured["dgpo"]["tau_ratio"]["mechanism_probe"]["relative_steps"],
        "absolute_stop_step": int(command[command.index("--max-steps") + 1]),
        "startup": ("restore actor and full AdamW; freshly fit balanced truth/step1920 head, recenter reference once, then freeze"
                    if (settings.get('full_trajectory') or {}).get('reward_refit') is not None else
                    "restore inherited head, reference and full AdamW; select saved fresh head only in matched trajectory arms"),
        "periodic_refits": False,
        "classifier_kl_refit_every_updates": 1 if (settings.get('full_trajectory') or {}).get('classifier_kl') else None,
        "classifier_hard_trust": (settings.get('full_trajectory') or {}).get('hard_trust_region'),
        "command": command}, indent=2), flush=True)
    if prepared_only:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "scripts"), str(EVENET), env.get("PYTHONPATH", "")])
    subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
