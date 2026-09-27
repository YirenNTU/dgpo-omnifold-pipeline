#!/usr/bin/env python3
"""Standalone fixed-policy residual-weight investigation; never starts DGPO.

Reproduce the production stack, then replay weight interventions from identical
saved initial classifiers. Optional independent W&B run; no reward installation
or policy updates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import inspect
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_raw_monitor_replay as replay
from ablate_raw_monitor_initialization import clone_state, file_digest, frozen_state, seeded
from residual_diagnostic_wandb import DiagnosticWandb, validate_wandb

ARMS = ("original", "unit_raw", "capped", "truth_null", "condition_only_weighted")


def frozen_digest(model):
    state = frozen_state(model) if hasattr(model, "backbone") else {}
    return replay.state_digest(state) if state else None


def score_on_device(model, condition, sample, batch_size):
    """Keep large pools on CPU; stage only the local scoring microbatch."""
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _score_population
    device = next(model.parameters()).device

    class Staged(torch.nn.Module):
        def forward(self, c, z):
            return model(c.to(device), z.to(device))

    return _score_population(Staged(), condition, sample, batch_size)


def mean_one(logw):
    x = logw.detach().double().reshape(-1)
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError("Empty/nonfinite log weights")
    return torch.softmax(x, dim=0) * len(x)


def weight_stats(logw):
    w = mean_one(logw).cpu()
    mass = w / w.sum()
    n = len(w)
    ess = float(1 / mass.square().sum())
    return {"events": n, "ess": ess, "ess_fraction": ess / n,
            "max_mean_one_weight": float(w.max()),
            "p99_weight": float(torch.quantile(w, .99)),
            "top_1pct_mass": float(mass.topk(max(1, math.ceil(n * .01))).values.sum()),
            "top_0_1pct_mass": float(mass.topk(max(1, math.ceil(n * .001))).values.sum())}


def score_metrics(data_logit, gen_logit, logw):
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import _weighted_binary_score_metrics
    d, g = data_logit.reshape(-1).double(), gen_logit.reshape(-1).double()
    if len(d) != len(g) or len(g) != logw.numel():
        raise ValueError("Paired score/weight sizes differ")
    if not torch.isfinite(d).all() or not torch.isfinite(g).all():
        raise ValueError("Nonfinite classifier scores")
    w = mean_one(logw).to(g.device)
    bce, acc, auc = _weighted_binary_score_metrics(d, g, w)
    negative = w * torch.nn.functional.softplus(g)
    tail = w.topk(max(1, math.ceil(len(w) * .01))).indices
    return {"bce": bce, "auc": auc, "balanced_accuracy": acc,
            "bce_gain_vs_null": math.log(2) - bce,
            "negative_bce_top_weight_1pct_fraction": float(negative[tail].sum() / negative.sum().clamp_min(1e-300)),
            "logit_abs_max": float(torch.cat((d, g)).abs().max())}


def _weighted_histogram_jsd(truth, generated, generated_weight, edges):
    """Square-root JSD with unit truth mass and weighted generated mass."""

    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    generated = np.asarray(generated, dtype=np.float64).reshape(-1)
    generated_weight = np.asarray(generated_weight, dtype=np.float64).reshape(-1)
    if generated.size != generated_weight.size:
        raise ValueError("Generated observables and weights differ in length")
    truth = truth[np.isfinite(truth)]
    keep = np.isfinite(generated) & np.isfinite(generated_weight) & (generated_weight >= 0)
    generated, generated_weight = generated[keep], generated_weight[keep]
    if not truth.size or not generated.size:
        return float("nan")
    p = np.histogram(truth, bins=edges)[0].astype(np.float64)
    q = np.histogram(generated, bins=edges, weights=generated_weight)[0].astype(np.float64)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left, right):
        positive = left > 0
        return float(np.sum(left[positive] * np.log2(left[positive] / right[positive])))

    return float(math.sqrt(max(0.0, 0.5 * (kl(p, midpoint) + kl(q, midpoint)))))


def weighted_physics_jsd(pool, logw, *, bins=40):
    """Target/reconstruction/topology closure under mean-one generated weights."""

    from RL.DGPO_neutrino.diagnostics.ztautau_validation import _observable_edges

    if type(bins) is not int or bins < 2:
        raise ValueError("physics_bins must be an integer >= 2")
    truth = pool.truth.detach().double().reshape(-1, 2, 2).clone()
    generated = pool.candidates[:, 0].detach().double().reshape(-1, 2, 2).clone()
    for values in (truth, generated):
        values[..., 1] = torch.atan2(torch.sin(values[..., 1]), torch.cos(values[..., 1]))
    observables = {
        "target/tau_a_delta_theta": (truth[:, 0, 0], generated[:, 0, 0]),
        "target/tau_a_delta_phi": (truth[:, 0, 1], generated[:, 0, 1]),
        "target/tau_b_delta_theta": (truth[:, 1, 0], generated[:, 1, 0]),
        "target/tau_b_delta_phi": (truth[:, 1, 1], generated[:, 1, 1]),
    }
    packing_spec = getattr(pool, "packing_spec", None)
    required = {
        "lead_a_visible_px", "lead_a_visible_py", "lead_a_visible_pz",
        "lead_b_visible_px", "lead_b_visible_py", "lead_b_visible_pz",
    }
    if packing_spec is not None and required.issubset(packing_spec.shapes):
        from RL.DGPO_neutrino.domains.ztautau import (
            reconstruct_tau_angles_from_deltas,
            tau_back_to_back_metrics,
            tau_calibration_magnitude_metrics,
        )
        from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import unpack_event_inputs

        batch = unpack_event_inputs(pool.packed_event, packing_spec)
        truth_k, generated_k = truth.unsqueeze(0), generated.unsqueeze(0)
        truth_a, truth_b = reconstruct_tau_angles_from_deltas(truth_k, batch)
        generated_a, generated_b = reconstruct_tau_angles_from_deltas(generated_k, batch)
        for leg, truth_leg, generated_leg in (
            ("a", truth_a, generated_a),
            ("b", truth_b, generated_b),
        ):
            observables[f"reco/tau_{leg}_theta"] = (
                truth_leg[0, :, 0], generated_leg[0, :, 0]
            )
            observables[f"reco/tau_{leg}_phi"] = (
                truth_leg[0, :, 1], generated_leg[0, :, 1]
            )
        truth_topology = {
            **tau_back_to_back_metrics(truth_k, batch),
            **tau_calibration_magnitude_metrics(truth_k, batch),
        }
        generated_topology = {
            **tau_back_to_back_metrics(generated_k, batch),
            **tau_calibration_magnitude_metrics(generated_k, batch),
        }
        for name in truth_topology:
            observables[f"topology/{name}"] = (
                truth_topology[name][0], generated_topology[name][0]
            )
    weights = mean_one(logw).cpu().numpy()
    result = {}
    for name, (truth_values, generated_values) in observables.items():
        truth_np = truth_values.detach().cpu().numpy()
        generated_np = generated_values.detach().cpu().numpy()
        edges = _observable_edges(name, (truth_np, generated_np), bins)
        unweighted = _weighted_histogram_jsd(
            truth_np, generated_np, np.ones_like(weights), edges
        )
        weighted = _weighted_histogram_jsd(
            truth_np, generated_np, weights, edges
        )
        result[name] = {
            "unweighted_jsd": unweighted,
            "weighted_jsd": weighted,
            "improvement": unweighted - weighted,
        }
    return result


def intervention(arm, train_logw, val_logw, quantile):
    """Cap is estimated on training weights only, in mean-one units."""
    if arm not in ARMS:
        raise ValueError(arm)
    if arm in ("unit_raw", "truth_null"):
        return torch.zeros_like(train_logw), torch.zeros_like(val_logw), None
    if arm in ("original", "condition_only_weighted"):
        return train_logw.clone(), val_logw.clone(), None
    train_w, val_w = mean_one(train_logw), mean_one(val_logw)
    cap = torch.quantile(train_w, quantile)
    tiny = torch.finfo(torch.float64).tiny
    return (train_w.clamp(max=cap).clamp_min(tiny).log().reshape(train_logw.shape),
            val_w.clamp(max=cap.to(val_w.device)).clamp_min(tiny).log().reshape(val_logw.shape), float(cap))


def resolved_protocol_arms(cfg, default_folds=2):
    """Return (repeats, folds) pairs. Fold arms may broadcast a single repeat."""

    repeats = [int(value) for value in cfg["crossfit_repeat_arms"]]
    fold_arms = cfg.get("crossfit_fold_arms")
    if fold_arms is None:
        return [(repeat, int(default_folds)) for repeat in repeats]
    folds = [int(value) for value in fold_arms]
    if len(repeats) == 1 and len(folds) > 1:
        repeats = repeats * len(folds)
    if len(repeats) != len(folds):
        raise ValueError(
            "crossfit_fold_arms must match crossfit_repeat_arms, or use a single repeat"
        )
    return list(zip(repeats, folds))


def validate_settings(settings):
    cfg = dict(settings)
    cfg["wandb"] = validate_wandb(cfg.get("wandb"))
    cfg["_legacy_single_arm_layout"] = (
        "crossfit_repeat_arms" not in cfg and "crossfit_fold_arms" not in cfg
    )
    cfg.setdefault("crossfit_repeat_arms", [1])
    cfg.setdefault("run_controls", True)
    cfg.setdefault("physics_bins", 40)
    for k in ("workers", "generation_batch_size", "score_batch_size", "diagnostic_iterations",
              "control_steps", "control_evaluate_every", "expected_policy_step",
              "generation_seed", "physics_bins"):
        if type(cfg[k]) is not int or cfg[k] < (0 if k == "expected_policy_step" else 1):
            raise ValueError(f"Invalid {k}")
    if not 2 <= cfg["diagnostic_iterations"] <= 3:
        raise ValueError("diagnostic_iterations must be two or three")
    if cfg["physics_bins"] < 2:
        raise ValueError("physics_bins must be at least two")
    if not 0 < cfg["cap_quantile"] < 1:
        raise ValueError("cap_quantile must lie in (0,1)")
    seeds = cfg["training_seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("training_seeds must be distinct nonnegative integers")
    repeats = cfg["crossfit_repeat_arms"]
    if (
        not isinstance(repeats, list)
        or not repeats
        or any(type(value) is not int or value < 1 for value in repeats)
    ):
        raise ValueError(
            "crossfit_repeat_arms must contain positive integers"
        )
    fold_arms = cfg.get("crossfit_fold_arms")
    if fold_arms is None:
        if len(set(repeats)) != len(repeats):
            raise ValueError(
                "crossfit_repeat_arms must contain distinct positive integers"
            )
    else:
        if (
            not isinstance(fold_arms, list)
            or not fold_arms
            or any(type(value) is not int or value < 2 for value in fold_arms)
        ):
            raise ValueError(
                "crossfit_fold_arms must contain integers >= 2 so every "
                "training increment stays held-out"
            )
        if len(repeats) not in (1, len(fold_arms)):
            raise ValueError(
                "crossfit_fold_arms must match crossfit_repeat_arms, or use a single repeat"
            )
        resolved_repeats = (
            repeats * len(fold_arms) if len(repeats) == 1 else repeats
        )
        if len(set(zip(resolved_repeats, fold_arms))) != len(fold_arms):
            raise ValueError("crossfit protocol arms must be unique")
    n_arms = len(resolved_protocol_arms(cfg))
    if n_arms > 1 and len(seeds) != 2:
        raise ValueError(
            "paired residual-protocol screens require exactly two training seeds"
        )
    if type(cfg["run_controls"]) is not bool:
        raise ValueError("run_controls must be boolean")
    if cfg["run_controls"] and n_arms > 1:
        raise ValueError(
            "intervention controls are supported only for a single protocol arm; "
            "use run_controls: false for a focused 2-fold vs 5-fold screen"
        )
    if cfg.get("pool_events") is not None and (type(cfg["pool_events"]) is not int or cfg["pool_events"] < 1):
        raise ValueError("pool_events must be null (full data) or positive")
    return cfg


def prepare(settings):
    """Read-only preflight. Pin the policy, data inputs, code and merged config."""
    from train_neutrino_backend import read_yaml, deep_update, absolutize_default_paths
    cfg = validate_settings(settings)
    base = (ROOT / cfg["base_config"]).resolve(strict=True)
    overlay = (ROOT / cfg["overlay_config"]).resolve(strict=True)
    runtime = absolutize_default_paths(deep_update(read_yaml(base), read_yaml(overlay)), base.parent)
    recal = runtime["dgpo"]["adaptive_omnifold"]["recalibration"]
    if recal["fit"].get("later_iteration_train_mode", "full") != "full":
        raise ValueError("This diagnostic requires the fullfit control; do not silently change staged fitting")
    if int(recal.get("candidates_per_event", 1)) != 1:
        raise ValueError("The paired residual diagnostic requires K=1")
    if (
        not recal.get("warm_start_from_iteration_one")
        or recal["crossfit_folds"] != 2
        or int(recal.get("crossfit_repeats", 1)) != 1
    ):
        raise ValueError(
            "Expected current iteration-1 inheritance and the 1x2-fold control"
        )
    if not runtime["dgpo"]["adaptive_omnifold"].get("single_pool_train_validation"):
        raise ValueError("Expected fixed single-pool train/validation split")
    runtime["options"]["Training"].setdefault("EMA", {}).update(
        replace_model_after_load=False, use_for_generation=False, use_ema_during_training_eval=False)
    path = Path(runtime["options"]["Training"]["model_checkpoint_load_path"]).resolve(strict=True)
    source = replay.read_checkpoint(path)
    if source.get("global_step") != cfg["expected_policy_step"]:
        raise ValueError("Wrong source policy step")
    digest = replay.state_digest(source["state_dict"])
    if cfg.get("expected_policy_sha256") and digest != cfg["expected_policy_sha256"]:
        raise ValueError("Policy tensor digest differs")
    del source
    backbone = Path(runtime["reward_config"]["omnifold"]["backbone_checkpoint"]).resolve(strict=True)
    normalization = Path(runtime["options"]["Dataset"]["normalization_file"]).resolve(strict=True)
    data_dir = Path(runtime["platform"]["data_parquet_dir"]).resolve(strict=True)
    files = sorted(str(p.resolve()) for p in data_dir.glob("*.parquet"))
    if not files:
        raise ValueError("No processed parquet files")
    output = Path(cfg["output_dir"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {output}")
    for protected in (path.parent, backbone.parent, normalization.parent, data_dir):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output overlaps a protected source directory")
    code_paths = [Path(__file__), ROOT / "scripts/diagnose_raw_monitor_replay.py",
                  ROOT / "scripts/residual_diagnostic_wandb.py",
                  ROOT / "scripts/ablate_raw_monitor_initialization.py",
                  ROOT / "scripts/train_neutrino_backend.py",
                  ROOT / "evenet_dgpo/RL/DGPO_neutrino/model_utils.py",
                  ROOT / "evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py"]
    code_paths += sorted((ROOT / "evenet_dgpo/RL/DGPO_neutrino/omnifold_ztautau").glob("*.py"))
    fingerprints = {str(p): file_digest(p) for p in [base, overlay, path, backbone, normalization, *code_paths]}
    cfg.update(output_dir=str(output), policy_checkpoint=str(path), policy_sha256=digest,
               backbone=str(backbone), runtime=runtime, files=files,
               source_fingerprints=fingerprints,
               data_stats={p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in files})
    return cfg


def verify_sources(cfg):
    for p, expected in cfg["source_fingerprints"].items():
        if file_digest(p) != expected:
            raise RuntimeError(f"Source changed: {p}")
    for p, expected in cfg["data_stats"].items():
        if [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] != expected:
            raise RuntimeError(f"Dataset metadata changed: {p}")


class DiagnosticLimit(Exception):
    """Intentional standalone stop after the requested scored iterations."""


class Capture:
    def __init__(self, cfg, train, val, root, rank, tracker=None, seed=None):
        self.cfg, self.train, self.val, self.root, self.rank = cfg, train, val, root, rank
        self.tracker, self.seed = tracker, seed
        self.repeats = int(cfg.get("crossfit_repeats", 1))
        self.folds = int(cfg.get("crossfit_folds", 2))
        self.protocol = cfg.get("protocol")
        self.inputs, self.fold_scores, self.iterations = {}, {}, []
        self.restored_reports, self.fit_started = {}, {}
        self.prior_val_fold_logw = {
            (repeat, fold): torch.zeros(val.n_events, dtype=torch.float64)
            for repeat in range(1, self.repeats + 1)
            for fold in range(1, self.folds + 1)
        }

    def member_stem(self, iteration, repeat, fold):
        if self.repeats == 1 and self.protocol is None:
            return f"i{iteration}_f{fold}"
        return f"i{iteration}_r{repeat}_f{fold}"

    def training_weight_path(self, repeat, fold):
        """Validation analogue of the OOF paths seen by one member's fit rows."""

        result = torch.zeros(self.val.n_events, dtype=torch.float64)
        for other_repeat in range(1, self.repeats + 1):
            if other_repeat == repeat:
                source_folds = [
                    value for value in range(1, self.folds + 1)
                    if value != fold
                ]
            else:
                source_folds = list(range(1, self.folds + 1))
            coefficient = 1.0 / float(self.repeats * len(source_folds))
            for source_fold in source_folds:
                result += coefficient * self.prior_val_fold_logw[
                    (other_repeat, source_fold)
                ]
        return result

    def save_json(self, name, data):
        if self.rank == 0:
            replay._exclusive_json(self.root / name, data)

    def save_tensor(self, name, data):
        if self.rank == 0:
            replay._exclusive_torch_save(self.root / name, data)

    def __call__(self, event, row):
        i = row["iteration"]
        repeat, fold = int(row.get("repeat", 1)), int(row.get("fold", 0))
        if event == "before_fit":
            key = (i, repeat, fold)
            self.fit_started[key] = time.perf_counter()
            self.inputs[key] = {"initial": clone_state(row["model"].state_dict()),
                                "fit_index": row["fit_index"].cpu().clone(),
                                "holdout_index": row["holdout_index"].cpu().clone(),
                                "fit_config": row["fit_config"], "seed": row["seed"],
                                "frozen_sha256": frozen_digest(row["model"]),
                                "train_logw": row["train_log_weight"].cpu().clone(),
                                "val_logw": row["validation_log_weight"].cpu().clone()}
            self.save_tensor(f"{self.member_stem(i, repeat, fold)}_input.pt", {
                **self.inputs[key], "fit_config": asdict(row["fit_config"])})
        elif event == "fold_scored":
            key = (i, repeat, fold)
            model = row["model"]
            before = replay.state_digest(model.state_dict())
            was_training = model.training
            model.eval()
            d = score_on_device(model, self.train.packed_event, self.train.truth, self.cfg["score_batch_size"]).cpu()
            g = score_on_device(model, self.train.packed_event, self.train.candidates, self.cfg["score_batch_size"]).cpu()
            model.train(was_training)
            if replay.state_digest(model.state_dict()) != before:
                raise RuntimeError("Diagnostic scoring mutated classifier")
            vd, vg = row["validation_data_logit"].cpu(), row["validation_gen_logit"].cpu()
            self.fold_scores[key] = {
                "train_data": d,
                "train_gen": g,
                "validation_data": vd,
                "validation_gen": vg,
            }
            inp = self.inputs[key]
            if frozen_digest(model) != inp["frozen_sha256"]:
                raise RuntimeError("Frozen classifier backbone changed during production fit")
            report = {
                      "repeat": repeat, "fold": fold,
                      "steps": row["diagnostics"].steps_completed,
                      "wall_time_seconds": time.perf_counter() - self.fit_started[key],
                      "state_sha256": before,
                      "validation_production": score_metrics(vd, vg, inp["val_logw"]),
                      "validation_training_weight_path": score_metrics(
                          vd, vg, self.training_weight_path(repeat, fold)
                      )}
            for label, indices in (("fit", inp["fit_index"]), ("oof", inp["holdout_index"])):
                report[label + "_eval"] = score_metrics(d[indices], g[indices], inp["train_logw"][indices])
                report[label + "_weights"] = weight_stats(inp["train_logw"][indices])
            self.restored_reports[key] = report
            stem = self.member_stem(i, repeat, fold)
            self.save_json(f"{stem}_restored_best.json", report)
            if self.tracker is not None:
                self.tracker.restored(
                    self.seed, i, fold, report,
                    repeat=repeat, protocol=self.protocol,
                )
            self.save_tensor(f"{stem}_restored_best.pt", {
                "state": clone_state(model.state_dict()), "train_truth_logits": d, "train_gen_logits": g,
                "validation_truth_logits": vd, "validation_gen_logits": vg})
        elif event == "iteration_scored":
            repeat_oof = []
            validation_repeat = []
            fold_differences = []
            validation_member_logits = []
            for repeat_index in range(1, self.repeats + 1):
                oof = torch.empty_like(row["oof_logit"].cpu()).reshape(-1).double()
                covered = torch.zeros(len(oof), dtype=torch.bool)
                validation_members = []
                for fold_index in range(1, self.folds + 1):
                    key = (i, repeat_index, fold_index)
                    scores = self.fold_scores[key]
                    indices = self.inputs[key]["holdout_index"]
                    oof[indices] = scores["train_gen"].reshape(-1).double()[indices]
                    covered[indices] = True
                    validation_members.append(
                        scores["validation_gen"].reshape(-1).double()
                    )
                    validation_member_logits.append(
                        scores["validation_gen"].reshape(-1).double()
                    )
                if not bool(covered.all()):
                    raise RuntimeError("Diagnostic repeat left OOF rows unscored")
                repeat_oof.append(oof)
                members = torch.stack(validation_members)
                validation_repeat.append(members.mean(0))
                for left in range(self.folds):
                    for right in range(left + 1, self.folds):
                        fold_differences.append(members[left] - members[right])
            repeat_oof = torch.stack(repeat_oof)
            validation_repeat = torch.stack(validation_repeat)
            expected_oof = repeat_oof.mean(0)
            observed_oof = row["oof_logit"].detach().cpu().reshape(-1).double()
            if not torch.allclose(expected_oof, observed_oof, rtol=1e-6, atol=1e-7):
                raise RuntimeError(
                    "Production OOF increment is not the mean of repeat logits"
                )
            fold_delta = torch.stack(fold_differences)
            oof_centered = repeat_oof - repeat_oof.mean(0, keepdim=True)
            validation_centered = (
                validation_repeat - validation_repeat.mean(0, keepdim=True)
            )
            validation_member_logits = torch.stack(validation_member_logits)
            oof_range = repeat_oof.max(0).values - repeat_oof.min(0).values
            validation_range = (
                validation_repeat.max(0).values
                - validation_repeat.min(0).values
            )
            member_reports = [
                self.restored_reports[(i, repeat_index, fold_index)]
                for repeat_index in range(1, self.repeats + 1)
                for fold_index in range(1, self.folds + 1)
            ]
            fit_bce = float(np.mean([item["fit_eval"]["bce"] for item in member_reports]))
            oof_bce = float(np.mean([item["oof_eval"]["bce"] for item in member_reports]))
            diag = row["diagnostics"]
            record = {"iteration": i, "crossfit_repeats": self.repeats,
                      "crossfit_folds": self.folds,
                      "accepted_by_production_auc_gate": diag.accepted,
                      "ensemble_validation": score_metrics(row["validation_data_logit"], row["validation_gen_logit"], row["validation_log_weight"]),
                      "train_weights_before": weight_stats(row["train_log_weight"]),
                      "validation_weights_before": weight_stats(row["validation_log_weight"]),
                      "fold_logit_rms_difference": float(fold_delta.square().mean().sqrt()),
                      "fold_logit_abs_difference_p99": float(torch.quantile(fold_delta.abs(), .99)),
                      "fold_sign_disagreement": float(
                          (
                              (validation_member_logits > 0).any(0)
                              & (validation_member_logits < 0).any(0)
                          ).double().mean()
                      ),
                      "oof_repeat_rms_dispersion": float(oof_centered.square().mean().sqrt()),
                      "oof_ensemble_vs_first_repeat_rms": float(
                          (repeat_oof.mean(0) - repeat_oof[0]).square().mean().sqrt()
                      ),
                      "oof_repeat_abs_range_p99": float(torch.quantile(oof_range, .99)),
                      "oof_repeat_sign_disagreement": float(
                          ((repeat_oof > 0).any(0) & (repeat_oof < 0).any(0)).double().mean()
                      ),
                      "validation_repeat_rms_dispersion": float(validation_centered.square().mean().sqrt()),
                      "validation_ensemble_vs_first_repeat_rms": float(
                          (
                              validation_repeat.mean(0)
                              - validation_repeat[0]
                          ).square().mean().sqrt()
                      ),
                      "validation_repeat_abs_range_p99": float(torch.quantile(validation_range, .99)),
                      "member_fit_bce_mean": fit_bce,
                      "member_oof_bce_mean": oof_bce,
                      "member_fit_minus_oof_bce": fit_bce - oof_bce,
                      "classifier_updates": int(sum(item["steps"] for item in member_reports)),
                      "classifier_wall_time_seconds": float(sum(item["wall_time_seconds"] for item in member_reports)),
                      "classifier_gpu_hours": float(
                          sum(item["wall_time_seconds"] for item in member_reports)
                          * int(self.cfg.get("workers", 1)) / 3600.0
                      ),
                      "train_weights_if_applied": weight_stats(row["proposed_train_log_weight"].cpu()),
                      "validation_weights_if_applied": weight_stats(row["proposed_validation_log_weight"].cpu()),
                      "train_physics_before": weighted_physics_jsd(
                          self.train, row["train_log_weight"].cpu(),
                          bins=int(self.cfg.get("physics_bins", 40)),
                      ),
                      "train_physics_if_applied": weighted_physics_jsd(
                          self.train, row["proposed_train_log_weight"].cpu(),
                          bins=int(self.cfg.get("physics_bins", 40)),
                      ),
                      "validation_physics_before": weighted_physics_jsd(
                          self.val, row["validation_log_weight"].cpu(),
                          bins=int(self.cfg.get("physics_bins", 40)),
                      ),
                      "validation_physics_if_applied": weighted_physics_jsd(
                          self.val, row["proposed_validation_log_weight"].cpu(),
                          bins=int(self.cfg.get("physics_bins", 40)),
                      )}
            self.iterations.append(record)
            self.save_json(f"iteration_{i}.json", record)
            if self.tracker is not None:
                self.tracker.iteration(
                    self.seed, i, record, protocol=self.protocol
                )
            if self.rank == 0:
                print(f"[residual-diagnostic] iteration={i} {json.dumps(record)}", flush=True)
            if diag.accepted:
                alpha = float(row.get("applied_tempering", self.cfg["tempering"]))
                for repeat_index in range(1, self.repeats + 1):
                    for fold_index in range(1, self.folds + 1):
                        self.prior_val_fold_logw[(repeat_index, fold_index)] += (
                            alpha
                            * self.fold_scores[
                                (i, repeat_index, fold_index)
                            ]["validation_gen"].reshape(-1).double()
                        )
            if i >= self.cfg["diagnostic_iterations"]:
                raise DiagnosticLimit()


def run_controls(capture, factory, device):
    from RL.DGPO_neutrino.omnifold_ztautau.ratio_fit import fit_density_ratio, _broadcast_module
    results = []
    for (i, repeat, f), inp in capture.inputs.items():
        if i < 2:
            continue
        tr = capture.train.select(inp["fit_index"])
        va = capture.val
        base_train_logw = inp["train_logw"][inp["fit_index"]].to(device)
        base_val_logw = inp["val_logw"].to(device)
        fc = replace(inp["fit_config"], steps=capture.cfg["control_steps"], min_steps=0,
                     validation_interval_steps=0, validation_patience_evaluations=0,
                     restore_best=False, require_saturation=False, progress_interval_steps=1)
        for arm in ARMS:
            with seeded(inp["seed"], device):
                model = factory().to(device)
            model.load_state_dict(inp["initial"], strict=True)
            _broadcast_module(model)
            tw, vw, cap = intervention(arm, base_train_logw, base_val_logw, capture.cfg["cap_quantile"])
            td, vd = tr.truth, va.truth
            tg = tr.truth[:, None] if arm == "truth_null" else tr.candidates
            vg = va.truth[:, None] if arm == "truth_null" else va.candidates
            if arm == "condition_only_weighted":
                # Fix candidate angles identically for both populations. Any
                # discrimination comes from weighted event context, not the
                # truth/generated candidate difference. This is NOT a null test.
                td, vd = torch.zeros_like(tr.truth), torch.zeros_like(va.truth)
                tg, vg = td[:, None], vd[:, None]
            rows = []

            def evaluate(step, progress=None):
                was_training = model.training
                model.eval()
                ds = score_on_device(model, va.packed_event, vd, capture.cfg["score_batch_size"])
                gs = score_on_device(model, va.packed_event, vg, capture.cfg["score_batch_size"])
                model.train(was_training)
                row = {"step": step, "own_objective": score_metrics(ds, gs, vw),
                       "original_weights_same_samples": score_metrics(ds, gs, base_val_logw),
                       "progress": progress or {}}
                rows.append(row)
                if capture.tracker is not None:
                    capture.tracker.control_evaluation(
                        capture.seed, i, f, arm, row,
                        repeat=repeat, protocol=capture.protocol,
                    )
                stem = capture.member_stem(i, repeat, f)
                capture.save_tensor(f"control_{stem}_{arm}_s{step}_scores.pt", {
                    "truth_logits": ds.cpu(), "gen_logits": gs.cpu()})
                if capture.rank == 0:
                    print(f"[residual-control] i={i} f={f} arm={arm} step={step} BCE={row['own_objective']['bce']:.6f}", flush=True)

            def progress(row):
                step = int(row["step"])
                if capture.tracker is not None:
                    capture.tracker.control_progress(
                        capture.seed, i, f, arm, row,
                        repeat=repeat, protocol=capture.protocol,
                    )
                if step % capture.cfg["control_evaluate_every"] == 0 or step == fc.steps:
                    evaluate(step, row)

            initial_hash = replay.state_digest(model.state_dict())
            frozen_before = frozen_digest(model)
            if initial_hash != replay.state_digest(inp["initial"]):
                raise RuntimeError("Control initialization changed after synchronization")
            evaluate(0)
            with seeded(inp["seed"], device):
                diag = fit_density_ratio(model, tr.packed_event, td, torch.ones(tr.n_events),
                    tr.packed_event, tg, mean_one(tw).reshape(tg.shape[:-1]).float().cpu(), fc, inp["seed"],
                    progress_callback=progress)
            if diag.steps_completed != fc.steps:
                raise RuntimeError("Control stopped before its matched update budget")
            if frozen_digest(model) != frozen_before:
                raise RuntimeError("Frozen classifier backbone changed during control fit")
            result = {"iteration": i, "repeat": repeat, "fold": f,
                      "arm": arm, "cap_from_training": cap,
                      "initial_sha256": initial_hash, "frozen_backbone_unchanged": True, "fit_config": asdict(fc),
                      "evaluations": rows, "train_weights": weight_stats(tw), "validation_weights": weight_stats(vw)}
            stem = capture.member_stem(i, repeat, f)
            capture.save_json(f"control_{stem}_{arm}.json", result)
            capture.save_tensor(
                f"control_{stem}_{arm}_final.pt", clone_state(model.state_dict())
            )
            results.append(result)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return results


def compact_results(results):
    out = []
    for seed in results:
        controls = []
        for arm in seed["controls"]:
            rows = arm["evaluations"]
            best = min(rows, key=lambda r: r["own_objective"]["bce"])
            controls.append({"iteration": arm["iteration"],
                             "repeat": arm.get("repeat", 1),
                             "fold": arm["fold"], "arm": arm["arm"],
                             "initial": rows[0]["own_objective"], "final": rows[-1]["own_objective"],
                             "best_sampled_step": best["step"], "best_sampled": best["own_objective"]})
        out.append({"seed": seed["seed"], "protocol": seed.get("protocol"),
                    "crossfit_repeats": seed.get("crossfit_repeats", 1),
                    "crossfit_folds": seed.get("crossfit_folds", 2),
                    "status": seed["status"], "iterations": seed["iterations"],
                    "controls": controls})
    return out


def paired_protocol_comparisons(results):
    """Compact ensemble-minus-r1f2 deltas for each shared seed/iteration."""

    by_key = {
        (
            int(result.get("crossfit_repeats", 1)),
            int(result.get("crossfit_folds", 2)),
            int(result["seed"]),
        ): result
        for result in results
    }
    if not any(repeats == 1 and folds == 2 for repeats, folds, _ in by_key):
        return []

    def physics_mean(record, field, group):
        values = [
            item["weighted_jsd"]
            for name, item in record[field].items()
            if name.startswith(group + "/")
            and math.isfinite(float(item["weighted_jsd"]))
        ]
        return float(np.mean(values)) if values else float("nan")

    comparisons = []
    ensemble_protocols = sorted({
        (repeats, folds)
        for repeats, folds, _ in by_key
        if (repeats, folds) != (1, 2)
    })
    for repeats, folds in ensemble_protocols:
        shared_seeds = sorted(
            seed
            for arm_repeats, arm_folds, seed in by_key
            if (arm_repeats, arm_folds) == (repeats, folds)
            and (1, 2, seed) in by_key
        )
        for seed in shared_seeds:
            control = by_key[(1, 2, seed)]
            ensemble = by_key[(repeats, folds, seed)]
            for control_iteration, ensemble_iteration in zip(
                control["iterations"], ensemble["iterations"]
            ):
                if control_iteration["iteration"] != ensemble_iteration["iteration"]:
                    raise ValueError("Paired protocol iteration ids differ")
                row = {
                    "seed": seed,
                    "control": "r1f2",
                    "ensemble": f"r{repeats}f{folds}",
                    "iteration": control_iteration["iteration"],
                    "validation_bce_delta": (
                        ensemble_iteration["ensemble_validation"]["bce"]
                        - control_iteration["ensemble_validation"]["bce"]
                    ),
                    "validation_auc_delta": (
                        ensemble_iteration["ensemble_validation"]["auc"]
                        - control_iteration["ensemble_validation"]["auc"]
                    ),
                    "member_fit_minus_oof_bce_delta": (
                        ensemble_iteration["member_fit_minus_oof_bce"]
                        - control_iteration["member_fit_minus_oof_bce"]
                    ),
                    "train_ess_fraction_delta": (
                        ensemble_iteration["train_weights_if_applied"]["ess_fraction"]
                        - control_iteration["train_weights_if_applied"]["ess_fraction"]
                    ),
                    "validation_ess_fraction_delta": (
                        ensemble_iteration["validation_weights_if_applied"]["ess_fraction"]
                        - control_iteration["validation_weights_if_applied"]["ess_fraction"]
                    ),
                    "classifier_updates_ratio": (
                        ensemble_iteration["classifier_updates"]
                        / max(1, control_iteration["classifier_updates"])
                    ),
                    "classifier_wall_time_ratio": (
                        ensemble_iteration["classifier_wall_time_seconds"]
                        / max(
                            1e-12,
                            control_iteration["classifier_wall_time_seconds"],
                        )
                    ),
                }
                for population in ("train", "validation"):
                    field = f"{population}_physics_if_applied"
                    for group in ("target", "reco", "topology"):
                        row[f"{population}_{group}_weighted_jsd_delta"] = (
                            physics_mean(ensemble_iteration, field, group)
                            - physics_mean(control_iteration, field, group)
                        )
                comparisons.append(row)
    return comparisons


def worker(cfg):
    import ray.train
    import ray.train.torch
    tracker = DiagnosticWandb(cfg, rank=ray.train.get_context().get_world_rank(),
                              device=ray.train.torch.get_device())
    tracker.start()
    exit_code = 1
    try:
        _worker_impl(cfg, tracker)
        exit_code = 0
    finally:
        tracker.finish(exit_code)


def _worker_impl(cfg, tracker):
    import ray.train
    import ray.train.torch
    from evenet.control.global_config import global_config
    from evenet.utilities.diffusion_sampler import DDIMSampler
    from RL.DGPO_neutrino.model_utils import load_evenet_model_for_dgpo, load_normalization_dict
    from RL.DGPO_neutrino.dgpo_trainer import _materialize_adaptive_omnifold_pool
    from RL.DGPO_neutrino.omnifold_ztautau.adaptive import resolve_adaptive_config, residual_closure_auc_limit
    from RL.DGPO_neutrino.omnifold_ztautau.evenet_ratio import (
        EvenetAdapterModelBuilder, _crossfit_repeat_seed,
        _identity_crossfit_splits, fit_residual_ratio_stack, peft_bank_factory)
    from RL.DGPO_neutrino.omnifold_ztautau.stage import build_fit_config

    logging.basicConfig(level=logging.INFO)
    ctx = ray.train.get_context()
    rank, world, device = ctx.get_world_rank(), ctx.get_world_size(), ray.train.torch.get_device()
    if world != cfg["workers"]:
        raise ValueError("Unexpected worker count")
    global_config.load_yaml(cfg["runtime_path"])
    torch.set_float32_matmul_precision(str(global_config.dgpo.get("float32_matmul_precision", "medium")))
    ac = resolve_adaptive_config(global_config.dgpo)
    source = replay.read_checkpoint(cfg["policy_checkpoint"])
    if replay.state_digest(source["state_dict"]) != cfg["policy_sha256"]:
        raise ValueError("Policy changed since preflight")
    bundle = load_evenet_model_for_dgpo(config=global_config, device=device, checkpoint_path=cfg["policy_checkpoint"])
    policy = bundle.model.eval().requires_grad_(False)
    before = replay.verify_policy_loaded(policy, source["state_dict"])
    del source
    pool = _materialize_adaptive_omnifold_pool(ray.train.get_dataset_shard("pool"),
        {"batch_size": cfg["generation_batch_size"], "prefetch_batches": 1},
        model=policy, sampler=DDIMSampler(device=device), device=device, world_size=world, rank=rank,
        quota_events=cfg["pool_events"], num_ddim_steps=int(global_config.dgpo.validation_num_ddim_steps),
        seed=cfg["generation_seed"], include_pairwise_context=ac.periodic_pair_features_enabled,
        include_visible_pair_rest_frame=ac.visible_pair_rest_frame_enabled)
    if cfg["pool_events"] is not None and pool.n_events != cfg["pool_events"]:
        raise ValueError("Incomplete generated pool")
    if replay.state_digest(policy.state_dict()) != before:
        raise RuntimeError("Policy changed during fixed-pool generation")
    del policy, bundle
    gc.collect()
    torch.cuda.empty_cache()
    root = Path(cfg["output_dir"])
    ti, vi = _identity_crossfit_splits(pool.identity_inputs, folds=5, seed=ac.single_pool_split_seed)[0]
    if torch.isin(ti, vi).any():
        raise RuntimeError("Train/validation overlap")
    train, val = pool.select(ti), pool.select(vi)
    tracker.emit("diagnostic/population", 0, {"pool_events": pool.n_events,
                 "train_events": train.n_events, "validation_events": val.n_events}, axis="event")
    if rank == 0:
        replay._exclusive_torch_save(root / "pool.pt", {"packing_spec": pool.packing_spec.to_dict(),
            "packed_event": pool.packed_event.cpu(), "truth": pool.truth.cpu(), "raw": pool.candidates.cpu(),
            "policy_sha256": before, "train_indices": ti.cpu(), "validation_indices": vi.cpu()})
    del pool
    rec = dict(global_config.dgpo.adaptive_omnifold.recalibration)
    kwargs = {k: v for k, v in rec.items() if k in inspect.signature(EvenetAdapterModelBuilder).parameters}
    with seeded(cfg["training_seeds"][0], device):
        builder = EvenetAdapterModelBuilder(config=global_config, normalization_dict=load_normalization_dict(global_config),
            checkpoint_path=cfg["backbone"], device=device, **kwargs)
    factory = peft_bank_factory(builder, train.packing_spec, "residual_diagnostic", reset=True)
    protocol_arms = resolved_protocol_arms(cfg, ac.crossfit_folds)
    folds = [
        pair
        for repeats, n_folds in protocol_arms
        for repeat in range(1, repeats + 1)
        for pair in _identity_crossfit_splits(
            train.identity_inputs,
            folds=n_folds,
            seed=_crossfit_repeat_seed(ac.seed, repeat),
        )
    ]
    fc = build_fit_config(ac.fit, n_train=train.n_events, n_validation=val.n_events,
                          max_batch_population=min(len(t) for t, _ in folds))
    if fc.batch_size % world:
        raise ValueError("Classifier global batch must be divisible by workers")
    all_results = []
    for repeats, n_folds in protocol_arms:
        protocol = (
            None
            if cfg.get("_legacy_single_arm_layout") and repeats == 1
            else f"r{repeats}f{n_folds}"
        )
        protocol_root = root if protocol is None else root / protocol
        if rank == 0 and protocol is not None:
            protocol_root.mkdir(exist_ok=False)
        for seed in cfg["training_seeds"]:
            folder = protocol_root / f"seed_{seed}"
            if rank == 0:
                folder.mkdir(exist_ok=False)
            capture = Capture(
                {
                    **cfg,
                    "tempering": ac.tempering,
                    "crossfit_folds": n_folds,
                    "crossfit_repeats": repeats,
                    "protocol": protocol,
                },
                train,
                val,
                folder,
                rank,
                tracker,
                seed,
            )

            def progress(row):
                tracker.production(seed, row, protocol=protocol)
                if rank == 0:
                    with (folder / "production_fit.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        f"[residual-fit] protocol={protocol} seed={seed} "
                        f"i={row.get('iteration')} r={row.get('repeat')} "
                        f"f={row.get('fold')} step={row.get('step')} "
                        f"loss={row.get('training_loss')} "
                        f"val={row.get('validation_loss')}",
                        flush=True,
                    )

            status = "production_closure"
            try:
                with seeded(seed, device):
                    result = fit_residual_ratio_stack(model_factory=factory,
                        data_condition=train.packed_event, data_sample=train.truth,
                        gen_condition=train.packed_event, gen_sample=train.candidates,
                        iterations=ac.max_iterations, min_iterations=ac.min_iterations, fit_config=fc,
                        tempering=ac.tempering, seed=seed, crossfit_folds=n_folds,
                        crossfit_repeats=repeats,
                        residual_min_auc_gain=residual_closure_auc_limit(ac, global_step=0) - .5,
                        validation_data_condition=val.packed_event.to(device), validation_data_sample=val.truth.to(device),
                        validation_gen_condition=val.packed_event.to(device), validation_gen_sample=val.candidates.to(device),
                        device=device, warm_start_iterations=ac.warm_start_iterations,
                        warm_start_from_iteration_one=ac.warm_start_from_iteration_one,
                        crossfit_seed=ac.seed, crossfit_partition=ac.crossfit_partition,
                        min_steps_per_fold=ac.fit.get("min_steps_per_fold", 0),
                        warm_start_min_epochs_per_fold=ac.fit.get("warm_start_min_epochs_per_fold"),
                        validation_interval_epochs=ac.fit.get("validation_interval_epochs"),
                        validation_patience_epochs=ac.fit.get("validation_patience_epochs"),
                        identity_condition=train.identity_override, diagnostic_callback=capture, progress_callback=progress)
                    del result
            except DiagnosticLimit:
                status = "diagnostic_iteration_limit_not_a_closure_claim"
            controls = (
                run_controls(capture, factory, device)
                if cfg["run_controls"]
                else []
            )
            record = {
                "protocol": protocol or f"r{repeats}f{n_folds}",
                "crossfit_folds": n_folds,
                "crossfit_repeats": repeats,
                "seed": seed,
                "status": status,
                "iterations": capture.iterations,
                "controls": controls,
                "tested_residual_members": (
                    len(controls) // len(ARMS) if controls else 0
                ),
            }
            capture.save_json("report.json", record)
            all_results.append(record)
            del capture
    if rank == 0:
        verify_sources(cfg)
        paired_comparisons = paired_protocol_comparisons(all_results)
        for row in paired_comparisons:
            tracker.emit(
                f"paired/{row['ensemble']}/s{row['seed']}",
                int(row["iteration"]),
                row,
                axis="iteration",
            )
        report = {"policy_updates": 0, "reward_installs": 0, "policy_loaded_verified": True,
                  "policy_unchanged": True, "source_files_unchanged": True,
                  "pool_events": train.n_events + val.n_events, "train_events": train.n_events,
                  "validation_events": val.n_events,
                  "crossfit_repeat_arms": cfg["crossfit_repeat_arms"],
                  "crossfit_fold_arms": cfg.get("crossfit_fold_arms"),
                  "results": all_results,
                  "paired_comparisons": paired_comparisons,
                  "limitations": [
                      "Fresh fixed-noise pool, not a bitwise replay of historical Ray order/noise.",
                      "Validation is used for model selection; not an untouched test set.",
                      "Cross-fold recursive dependence is not eliminated by this diagnostic.",
                      "Member-specific validation weight paths are evaluation-only counterfactuals, not proof of a corrected pipeline.",
                      "Weight clipping changes the target; improvement is diagnostic, not a recommended production fix.",
                      "Fixed-budget controls are not convergence guarantees; compare curves across seeds.",
                      "In-sample fit scores and OOF scores must not be confused.",
                  ]}
        replay._exclusive_json(root / "report.json", report)
        replay._exclusive_json(root / "summary.json", {
            "policy_updates": 0, "results": compact_results(all_results),
            "paired_comparisons": paired_comparisons,
            "interpretation": "AUC refers to the listed classifier objective, not DGPO improvement; compare objectives separately.",
            "limitations": report["limitations"]})
        print(f"Report: {root / 'report.json'}", flush=True)
    ray.train.report({
        "policy_updates": 0,
        "diagnostic_protocol_seed_fits_completed": len(all_results),
    })


def upload_existing(settings):
    """Upload completed JSON records without loading tensors, Ray or a policy."""
    root = Path(settings["output_dir"]).expanduser().resolve(strict=True)
    cfg = json.loads((root / "manifest.json").read_text())
    cfg.update(output_dir=str(root), wandb=validate_wandb(settings.get("wandb")))
    if not cfg["wandb"]["enabled"]:
        raise ValueError("--upload-existing requires wandb.enabled: true")
    tracker = DiagnosticWandb(cfg)
    tracker.start()
    exit_code = 1
    try:
        protocols = (
            [(None, 1, 2)]
            if cfg.get("crossfit_repeat_arms") is None
            or cfg.get("_legacy_single_arm_layout")
            else [
                (f"r{repeats}f{folds}", repeats, folds)
                for repeats, folds in resolved_protocol_arms(cfg)
            ]
        )
        for protocol, repeats, n_folds in protocols:
            protocol_root = root if protocol is None else root / protocol
            for seed in cfg["training_seeds"]:
                folder = protocol_root / f"seed_{seed}"
                curve = folder / "production_fit.jsonl"
                if curve.is_file():
                    lines = curve.read_text().splitlines()
                    for number, line in enumerate(lines):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            if number == len(lines) - 1:
                                logging.warning("Skipping incomplete final production log line: %s", curve)
                                break
                            raise
                        tracker.production(seed, row, protocol=protocol)
                for iteration in range(1, cfg["diagnostic_iterations"] + 1):
                    for repeat in range(1, repeats + 1):
                        for fold in range(1, n_folds + 1):
                            stem = (
                                f"i{iteration}_f{fold}"
                                if protocol is None
                                else f"i{iteration}_r{repeat}_f{fold}"
                            )
                            path = folder / f"{stem}_restored_best.json"
                            if path.is_file():
                                tracker.restored(
                                    seed, iteration, fold,
                                    json.loads(path.read_text()),
                                    repeat=repeat, protocol=protocol,
                                )
                    path = folder / f"iteration_{iteration}.json"
                    if path.is_file():
                        tracker.iteration(
                            seed, iteration, json.loads(path.read_text()),
                            protocol=protocol,
                        )
                    for repeat in range(1, repeats + 1):
                        for fold in range(1, n_folds + 1):
                            stem = (
                                f"i{iteration}_f{fold}"
                                if protocol is None
                                else f"i{iteration}_r{repeat}_f{fold}"
                            )
                            for arm in ARMS:
                                path = folder / f"control_{stem}_{arm}.json"
                                if not path.is_file():
                                    continue
                                result = json.loads(path.read_text())
                                for row in result["evaluations"]:
                                    if row.get("progress"):
                                        tracker.control_progress(
                                            seed, iteration, fold, arm,
                                            row["progress"], repeat=repeat,
                                            protocol=protocol,
                                        )
                                    tracker.control_evaluation(
                                        seed, iteration, fold, arm, row,
                                        repeat=repeat, protocol=protocol,
                                    )
        report_path = root / "report.json"
        if report_path.is_file():
            for row in json.loads(report_path.read_text()).get(
                "paired_comparisons", ()
            ):
                tracker.emit(
                    f"paired/{row['ensemble']}/s{row['seed']}",
                    int(row["iteration"]),
                    row,
                    axis="iteration",
                )
        tracker.emit("diagnostic/upload", 0, {"existing_files_only": 1,
                     "complete_report_present": int((root / "report.json").is_file())}, axis="event")
        exit_code = 0
    finally:
        tracker.finish(exit_code)
    return 0


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--upload-existing", action="store_true",
                      help="Upload saved diagnostic JSON records to a NEW W&B run, no GPU or retraining")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if args.output_dir:
        settings["output_dir"] = str(args.output_dir)
    if args.upload_existing:
        return upload_existing(settings)
    cfg = prepare(settings)
    print(f"Verified policy step={cfg['expected_policy_step']} SHA256={cfg['policy_sha256']}; output={cfg['output_dir']}", flush=True)
    if args.check_only:
        return 0
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=False)
    cfg["runtime_path"] = str(root / "runtime.yaml")
    with Path(cfg["runtime_path"]).open("x") as f:
        yaml.safe_dump(cfg["runtime"], f, sort_keys=False)
    replay._exclusive_json(root / "manifest.json", cfg)
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchTrainer
    from evenet.control.global_config import global_config
    from evenet.shared import make_process_fn, register_dataset
    global_config.load_yaml(cfg["runtime_path"])
    ray.init(address=os.environ.get("RAY_ADDRESS") or "auto", runtime_env={"env_vars": {
        "PYTHONPATH": os.pathsep.join([str(ROOT / "evenet_dgpo"), str(ROOT / "scripts"), os.environ.get("PYTHONPATH", "")]),
        "WANDB_MODE": "online" if cfg["wandb"]["enabled"] else "disabled"}})
    dataset, _ = register_dataset(cfg["files"], make_process_fn(Path(global_config.platform.data_parquet_dir)),
                                  global_config.platform, dataset_limit=1.0, file_shuffling=False)
    TorchTrainer(train_loop_per_worker=worker, train_loop_config=cfg, datasets={"pool": dataset},
        scaling_config=ScalingConfig(num_workers=cfg["workers"], use_gpu=True, resources_per_worker={"CPU": 2, "GPU": 1}),
        run_config=RunConfig(name="residual-weight-diagnostic", storage_path=str(root / "ray_results"),
                             failure_config=FailureConfig(max_failures=0))).fit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
