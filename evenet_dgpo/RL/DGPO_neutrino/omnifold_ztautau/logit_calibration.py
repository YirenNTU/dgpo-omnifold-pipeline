"""CPU-only affine calibration of a frozen classifier, not physics projection.

Truth is label one; every event contributes one truth and one generated score.
Only calibration rows may determine a, b or optimization conditioning. The
positive slope preserves ranking, but does not guarantee a full-space ratio.
"""
from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.metrics import roc_auc_score


SCHEMA = "h4-logit-calibration-v1"
MIN_SLOPE = 1e-6


def paired_logits(truth, generated):
    t, g = (np.asarray(x, dtype=np.float64).reshape(-1) for x in (truth, generated))
    if len(t) != len(g) or len(t) < 2:
        raise ValueError("Requires at least two aligned K=1 truth/generated events")
    if not np.isfinite(t).all() or not np.isfinite(g).all():
        raise ValueError("Nonfinite logits")
    return t, g


def identity_groups(condition, chunk_size=8192):
    """Group identical saved identity inputs; never hash candidates or scores.

    This is event grouping, not checkpoint-SHA bookkeeping. Canonical float32
    and signed-zero handling match the production identity partition convention.
    Chunking also works with the memory-mapped CPU tensors in the source export.
    """
    if len(condition.shape) != 2:
        raise ValueError("identity_condition must be a two-dimensional event matrix")
    result = np.empty(len(condition), dtype="U64")
    for start in range(0, len(condition), chunk_size):
        rows = np.array(np.asarray(condition[start:start + chunk_size]), dtype="<f4", order="C", copy=True)
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite identity inputs")
        rows[rows == 0] = 0
        for offset, row in enumerate(rows):
            result[start + offset] = hashlib.sha256(row.tobytes()).hexdigest()
    return result


def validate_source(bundle):
    if bundle.get("schema") != "h4-ratio-health-v1":
        raise ValueError("Requires the original h4-ratio-health-v1 export, not projected samples")
    t, g = paired_logits(bundle["truth_logits"], bundle["gen_logits"])
    identity = bundle["identity_condition"]
    indices = {}
    for name in ("fit", "early_stop", "test"):
        raw = np.asarray(bundle["split_indices"][name])
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError("Split indices must be one-dimensional integers")
        idx = raw.astype(np.int64)
        if not len(idx) or len(np.unique(idx)) != len(idx) or idx.min() < 0 or idx.max() >= len(identity):
            raise ValueError("Invalid or duplicate source split indices")
        indices[name] = idx
    if len(indices["test"]) != len(t):
        raise ValueError("Saved logits do not align with source test indices; K=1 required")
    all_indices = np.concatenate(list(indices.values()))
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Source test overlaps classifier fitting/early stopping")
    groups = identity_groups(identity)
    test_groups = groups[indices["test"]]
    for name in ("fit", "early_stop"):
        if np.intersect1d(test_groups, groups[indices[name]]).size:
            raise ValueError("Source test identity overlaps classifier fitting/early stopping")
    return t, g, test_groups, indices["test"]


def split_groups(groups, *, seed=20260920, calibration_fraction=0.5):
    """Deterministic group split independent of labels, logits and topology."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if not 0 < calibration_fraction < 1 or len(unique) < 4:
        raise ValueError("Need at least four identities and a calibration fraction in (0,1)")
    count = int(math.floor(len(unique) * calibration_fraction))
    if count < 2 or len(unique) - count < 2:
        raise ValueError("Need at least two identities in calibration and evaluation")
    order = np.random.default_rng(seed).permutation(len(unique))
    mask = np.isin(groups, unique[order[:count]])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def event_bce(truth, generated):
    t, g = paired_logits(truth, generated)
    return 0.5 * (np.logaddexp(0.0, -t) + np.logaddexp(0.0, g))


def fit_affine(truth, generated, *, max_iterations=1000,
               progress: Callable[[dict], None] | None = None):
    """Constrained convex two-parameter logistic fit; never accesses test data.

    Optimize in centered/scaled coordinates for conditioning, then convert
    back to logit_cal = a * logit + b. No penalty, clipping, smoothing, or
    ESS-driven temperature selection is used. Identity (a=1,b=0) is the start.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    t, g = paired_logits(truth, generated)
    scores = np.concatenate((t, g))
    labels = np.concatenate((np.ones(len(t)), np.zeros(len(g))))
    center, scale = float(scores.mean()), float(scores.std())
    if not math.isfinite(center) or not math.isfinite(scale):
        raise ValueError("Calibration-score moments overflowed")
    curve = []

    def record(a, b):
        row = dict(iteration=len(curve), a=float(a), b=float(b),
                   bce=float(event_bce(a * t + b, a * g + b).mean()))
        curve.append(row)
        if progress is not None:
            progress(dict(row))

    record(1.0, 0.0)
    constant = scale <= 1e-12
    separable = bool(not constant and float(t.min()) >= float(g.max()))
    if constant:
        a, b = 1.0, -center
        record(a, b)
        success, message, iterations = True, "Constant score: intercept-only balanced-null fit", 0
    else:
        x = (scores - center) / scale

        def objective(theta):
            z = theta[0] * x + theta[1]
            # This form avoids cancellation for confidently correct labels.
            value = np.where(labels > 0, np.logaddexp(0.0, -z), np.logaddexp(0.0, z)).mean()
            residual = expit(z) - labels
            gradient = np.array([(residual * x).mean(), residual.mean()])
            return float(value), gradient

        def callback(theta):
            a = theta[0] / scale
            record(a, theta[1] - a * center)

        result = minimize(objective, np.array([scale, center]), jac=True, method="L-BFGS-B",
                          bounds=[(MIN_SLOPE * scale, None), (None, None)], callback=callback,
                          options=dict(maxiter=max_iterations, ftol=1e-13, gtol=1e-10, maxls=50))
        a = float(result.x[0] / scale)
        b = float(result.x[1] - a * center)
        success, message, iterations = bool(result.success), str(result.message), int(result.nit)
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0:
        raise ValueError("Optimizer returned invalid affine parameters")
    if curve[-1]["a"] != a or curve[-1]["b"] != b:
        record(a, b)
    at_floor = a <= MIN_SLOPE * (1 + 1e-5)
    not_worse = curve[-1]["bce"] <= curve[0]["bce"] + 1e-10
    eligible = success and not constant and not separable and not at_floor and not_worse
    return dict(a=a, b=b, success=success, message=message, optimizer_iterations=iterations,
                constant_scores=constant, separated_or_quasiseparated_calibration=separable,
                slope_at_lower_bound=at_floor, calibration_bce_not_worse=not_worse,
                eligible=eligible, minimum_slope=MIN_SLOPE, curve=curve,
                calibration_center=center, calibration_scale=scale)


def reliability(truth, generated, *, bins=15):
    t, g = paired_logits(truth, generated)
    if bins < 2:
        raise ValueError("Need at least two reliability bins")
    probabilities = expit(np.concatenate((t, g)))
    labels = np.concatenate((np.ones(len(t)), np.zeros(len(g))))
    assignment = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        prediction = float(probabilities[mask].mean()) if count else None
        observed = float(labels[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(prediction - observed)
        rows.append(dict(bin=index, lower=index / bins, upper=(index + 1) / bins,
                         count=count, mean_probability=prediction, truth_fraction=observed))
    return dict(ece=float(ece), bins=rows)


def ratio_metrics(logits, groups):
    s = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(s) != len(groups) or len(s) < 2 or not np.isfinite(s).all():
        raise ValueError("Invalid ratio scores/groups")
    log_total = float(logsumexp(s))
    w = np.exp(s - log_total)
    log_mean = log_total - math.log(len(s))
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    if len(counts) < 2:
        raise ValueError("Need at least two event identities for ratio uncertainty")
    # Cluster-relative SE of the unnormalized mean; avoid exponent overflow.
    group_mass = np.bincount(inverse, weights=w)
    relative_se = math.sqrt(len(counts) / (len(counts) - 1) *
                            float(np.square(group_mass - counts / len(s)).sum()))
    return dict(log_mean_ratio=log_mean,
                mean_ratio=math.exp(log_mean) if -700 < log_mean < 700 else None,
                mean_ratio_relative_se=relative_se,
                ess=float(1 / np.square(w).sum()),
                ess_fraction=float(1 / (len(s) * np.square(w).sum())),
                top1pct_mass=float(np.sort(w)[-max(1, math.ceil(0.01 * len(s))):].sum()),
                max_weight_mass=float(w.max()), log_ratio_max=float(s.max()),
                **{f"log_ratio_q{q:g}": float(np.quantile(s, q)) for q in (0.01, 0.5, 0.95, 0.99, 0.999)})


def measure(truth, generated, groups):
    t, g = paired_logits(truth, generated)
    rel = reliability(t, g)
    return dict(events=len(t), identities=len(np.unique(groups)),
                bce=float(event_bce(t, g).mean()),
                bce_truth=float(np.logaddexp(0.0, -t).mean()),
                bce_generated=float(np.logaddexp(0.0, g).mean()),
                brier=float(0.5 * (np.square(expit(-t)).mean() + np.square(expit(g)).mean())),
                auc=float(roc_auc_score(np.r_[np.ones(len(t)), np.zeros(len(g))], np.r_[t, g])),
                ece=rel["ece"], reliability=rel["bins"], ratio=ratio_metrics(g, groups))


def compare(truth, generated, groups, a, b, *, bootstrap=500, seed=20260921):
    """Paired identity-cluster bootstrap conditional on the fixed fitted a,b."""
    if bootstrap < 20 or not math.isfinite(a) or not math.isfinite(b) or a <= 0:
        raise ValueError("Need positive finite slope and at least 20 bootstrap replicates")
    t, g = paired_logits(truth, generated)
    if len(groups) != len(t):
        raise ValueError("Groups do not align with paired logits")
    tc, gc = a * t + b, a * g + b
    raw, calibrated = measure(t, g, groups), measure(tc, gc, groups)
    delta = event_bce(tc, gc) - event_bce(t, g)
    unique, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique)
    counts = np.bincount(inverse)
    changes = np.bincount(inverse, weights=delta)
    group_log_weights = []
    for scores in (g, gc):
        totals = np.full(n_groups, -np.inf)
        np.logaddexp.at(totals, inverse, scores)
        group_log_weights.append(totals)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        selected = rng.integers(0, n_groups, n_groups)
        n = counts[selected].sum()
        samples.append([float(changes[selected].sum() / n)] + [
            float(logsumexp(weights[selected]) - math.log(n))
            for weights in group_log_weights])
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    cluster_residual = changes - float(delta.mean()) * counts
    se = math.sqrt(n_groups / (n_groups - 1) * float(np.square(cluster_residual).sum())) / len(t)
    return dict(raw=raw, calibrated=calibrated,
                delta=dict(bce=float(delta.mean()), brier=calibrated["brier"] - raw["brier"],
                           auc=calibrated["auc"] - raw["auc"], ece=calibrated["ece"] - raw["ece"]),
                uncertainty=dict(paired_bce_se=se, bce_delta_ci95=intervals[:, 0].tolist(),
                                 raw_log_mean_ratio_ci95=intervals[:, 1].tolist(),
                                 calibrated_log_mean_ratio_ci95=intervals[:, 2].tolist(),
                                 bootstrap_replicates=bootstrap,
                                 scope="Conditional on frozen classifier and fitted a,b; identity-cluster bootstrap. "
                                       "Does not cover calibration-fit/training uncertainty or unseen rare tails."),
                auc_preserved=abs(calibrated["auc"] - raw["auc"]) <= 1e-12)


def run_experiment(truth, generated, groups, *, seed=20260920, calibration_fraction=0.5,
                   bootstrap=500, max_iterations=1000, progress=None):
    t, g = paired_logits(truth, generated)
    groups = np.asarray(groups)
    if len(groups) != len(t):
        raise ValueError("Event groups do not align with logits")
    cal, test = split_groups(groups, seed=seed, calibration_fraction=calibration_fraction)
    fit = fit_affine(t[cal], g[cal], max_iterations=max_iterations, progress=progress)
    a, b = fit["a"], fit["b"]
    # No evaluation scores are inspected until the calibrator is finalized.
    evaluation = compare(t[test], g[test], groups[test], a, b, bootstrap=bootstrap, seed=seed + 1)
    if not fit["eligible"] or not evaluation["auc_preserved"]:
        decision = "inconclusive_fit_or_rank_check"
    elif evaluation["uncertainty"]["bce_delta_ci95"][1] < 0:
        decision = "exploratory_calibration_improvement"
    elif evaluation["uncertainty"]["bce_delta_ci95"][0] > 0:
        decision = "held_out_bce_worsened"
    else:
        decision = "no_clear_held_out_bce_improvement"
    result = dict(schema=SCHEMA, fit=fit,
                  calibration=dict(raw=measure(t[cal], g[cal], groups[cal]),
                                   calibrated=measure(a * t[cal] + b, a * g[cal] + b, groups[cal])),
                  evaluation=evaluation, decision=decision,
                  dgpo_interface=dict(measured=False, reason="Source artifact has K=1; no K>=2 policy/gate replay.",
                                      advantage_scale=a, additive_offset_cancels=True,
                                      identity="For log-ratio leave-one-out rewards, A_cal = a * A_raw.",
                                      gate_identity="For fixed candidates and L_cur-L_ref, M_cal = a * M_raw; "
                                                    "w_cal = sigmoid(a * M_raw). Not merely a learning-rate change."),
                  protocol=dict(seed=seed, calibration_fraction=calibration_fraction,
                                calibration_events=len(cal), evaluation_events=len(test),
                                calibration_identities=len(np.unique(groups[cal])),
                                evaluation_identities=len(np.unique(groups[test])),
                                balanced_class_prior=0.5, classifier_fits=0, policy_updates=0,
                                calibration_parameters=2, physics_projection=False,
                                deployed_to_dgpo=False, held_out_from_calibration=True,
                                source_pool_previously_examined=True, confirmatory=False,
                                primary_endpoint="Evaluation paired balanced BCE: calibrated minus raw.",
                                selection="Fit a,b only on calibration BCE. No ESS/AUC/test-driven selection."),
                  limitations=["Exploratory re-split of the previously inspected classifier-test pool, not a fresh confirmation set.",
                               "Calibration preserves ranking and cannot repair missed features or missing generator coverage.",
                               "Score calibration alone does not certify full-dimensional or conditional density-ratio accuracy.",
                               "Low ESS may reflect a real overlap deficit; ESS improvement is not a selection criterion.",
                               "This experiment does not measure policy gradients, DGPO gate values or physics closure."])
    return result, cal, test
