#!/usr/bin/env python3
"""Fit positive affine score calibration on frozen original H4 logits, CPU only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evenet_dgpo"))
from RL.DGPO_neutrino.omnifold_ztautau.logit_calibration import (  # noqa: E402
    SCHEMA, run_experiment, split_groups, validate_source,
)

DEFAULT_SOURCE = Path("/pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit")
DEFAULT_OUTPUT = Path("/pscratch/sd/y/yiren/Ztautau/h4_logit_calibration")
RUN_ID = "h4logcal1"
RUN_NAME = "Can score calibration improve ratios? | frozen H4 | positive affine | CPU replay"
RUN_GROUP = "H4 ratio calibration"
OWNED_FILES = ("manifest.json", "report.json", "calibrator.json", "scores.npz", "COMPLETE",
               "reliability.png", "log_ratio.png", "calibration_fit.png")


def prepare_output(output, source):
    """Allow reruns; replace this diagnostic's files only, never source artifacts."""
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Output must be separate from the source artifact directory")
    if output.exists() and not output.is_dir():
        raise ValueError("Output must be a directory")
    # Check all targets before removing anything; unknown user files survive.
    if any((output / name).is_dir() for name in OWNED_FILES):
        raise ValueError("A diagnostic output filename is occupied by a directory")
    output.mkdir(parents=True, exist_ok=True)
    replaced = []
    for name in OWNED_FILES:
        target = output / name
        if target.is_file() or target.is_symlink():
            target.unlink()
            replaced.append(name)
    return output, replaced


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def flat_scalars(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flat_scalars(item, name))
        elif isinstance(item, (int, float, bool)) and not isinstance(item, np.generic):
            result[name] = item
    return result


def make_plots(report, generated_test, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evaluation, fit = report["evaluation"], report["fit"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for arm, color in (("raw", "#2878b5"), ("calibrated", "#df7938")):
        rows = [row for row in evaluation[arm]["reliability"] if row["count"]]
        axes[0].plot([r["mean_probability"] for r in rows], [r["truth_fraction"] for r in rows],
                     "o-", label=arm, color=color)
        axes[1].plot([(r["lower"] + r["upper"]) / 2 for r in rows], [r["count"] for r in rows],
                     "o-", label=arm, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Predicted truth probability", ylabel="Observed truth fraction",
                title="Evaluation reliability (15 fixed bins)", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Probability bin", ylabel="Examples", yscale="log", title="Bin occupancy")
    for axis in axes:
        axis.legend()
    fig.savefig(output / "reliability.png", dpi=150)
    plt.close(fig)

    g = np.asarray(generated_test)
    calibrated = fit["a"] * g + fit["b"]
    low, high = min(g.min(), calibrated.min()), max(g.max(), calibrated.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, 61)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, scores in (("raw", g), ("calibrated", calibrated)):
        axis.hist(scores, bins=edges, histtype="step", label=label, linewidth=1.5)
    axis.set(xlabel="Generated log ratio", ylabel="Events", yscale="log",
             title="Evaluation scores: no clipping or event removal")
    axis.legend()
    fig.savefig(output / "log_ratio.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot([r["iteration"] for r in fit["curve"]], [r["bce"] for r in fit["curve"]], "o-")
    axis.axhline(fit["curve"][0]["bce"], linestyle="--", color="gray", label="raw")
    axis.set(xlabel="Calibration optimizer iteration", ylabel="Balanced BCE",
             title="Calibration rows only; evaluation not used during fitting")
    axis.legend()
    fig.savefig(output / "calibration_fit.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0 or not 0 < args.calibration_fraction < 1 or args.bootstrap < 20 or args.max_iterations < 1:
        parser.error("Need a nonnegative seed, fraction in (0,1), bootstrap>=20 and max-iterations>=1")
    if int(os.environ.get("SLURM_PROCID", "0")) != 0:
        parser.error("Run this CPU diagnostic once, not once per GPU/Slurm task")
    source = args.directory.expanduser().resolve()
    if not (source / "COMPLETE").is_file():
        parser.error("Source ratio export is incomplete")
    torch.set_num_threads(1)
    # No model restoration, inference, sampling, Ray connection or GPU required.
    bundle = torch.load(source / "best_classifier_and_test.pt", map_location="cpu", weights_only=True, mmap=True)
    t, g, groups, pool_rows = validate_source(bundle)
    split_groups(groups, seed=args.seed, calibration_fraction=args.calibration_fraction)
    source_fit = bundle.get("fit_diagnostics", {})
    # Do not duplicate optional checkpoint tensors or long fitting histories.
    source_fit = {key: source_fit.get(key) for key in
                  ("steps_completed", "best_step", "validation_loss", "validation_auc", "saturated")}
    source_fit = {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                  for key, value in source_fit.items()}
    provenance = dict(source_directory=str(source), source_schema=bundle["schema"],
                      source_seed=bundle.get("seed"), source_split_protocol=bundle.get("split_protocol"),
                      source_fit_diagnostics=source_fit,
                      score_origin="Original frozen best-BCE classifier export; not physics-projected scores.")
    # Avoid carrying large model/condition tensors throughout the scalar fit.
    t, g, pool_rows = t.copy(), g.copy(), pool_rows.copy()
    del bundle
    config = dict(schema=SCHEMA, **provenance, seed=args.seed,
                  calibration_fraction=args.calibration_fraction, bootstrap=args.bootstrap,
                  max_iterations=args.max_iterations, run_id=args.run_id, run_name=RUN_NAME,
                  classifier_fits=0, policy_updates=0, calibration_parameters=2, device="cpu",
                  physics_projection=False, source_pool_previously_examined=True, confirmatory=False)
    output, replaced = prepare_output(args.output, source)
    if replaced:
        print("Replaced prior diagnostic files (not source data): " + ", ".join(replaced), flush=True)
    print(f"Loaded {len(t)} original paired scores. No classifier fit, policy update, Ray or GPU.", flush=True)
    print("Exploratory re-split of previously examined test events; not fresh confirmatory evidence.", flush=True)
    write_json(output / "manifest.json", dict(config, started_utc=datetime.now(timezone.utc).isoformat()))
    run = None
    try:
        if not args.no_wandb:
            import wandb
            run = wandb.init(entity="ytchou97-university-of-washington", project="nu2flow-RL",
                             id=args.run_id, resume="allow", name=RUN_NAME, group=RUN_GROUP,
                             tags=["H4", "logit-calibration", "frozen-classifier", "CPU-replay", "no-policy-update"],
                             config=config)
            run.summary.update(dict(complete=False, phase="fit_calibrator", confirmatory=False))
            run.define_metric("calibration_fit/iteration")
            run.define_metric("calibration_fit/*", step_metric="calibration_fit/iteration")

        def progress(row):
            print(f"Calibration iter={row['iteration']} BCE={row['bce']:.8f} a={row['a']:.6g} b={row['b']:.6g}", flush=True)
            if run:
                run.log({"calibration_fit/" + key: value for key, value in row.items()})

        report, cal, test = run_experiment(t, g, groups, seed=args.seed,
                                         calibration_fraction=args.calibration_fraction,
                                         bootstrap=args.bootstrap, max_iterations=args.max_iterations, progress=progress)
        report["source"] = provenance
        a, b = report["fit"]["a"], report["fit"]["b"]
        write_json(output / "report.json", report)
        write_json(output / "calibrator.json", dict(schema=SCHEMA, a=a, b=b, logit_transform="a * logit + b",
                                                    positive_class="truth", balanced_class_prior=0.5,
                                                    source=provenance, eligible=report["fit"]["eligible"],
                                                    decision=report["decision"], deployed_to_dgpo=False))
        cal_mask = np.zeros(len(t), dtype=bool)
        cal_mask[cal] = True
        np.savez_compressed(output / "scores.npz", pool_rows=pool_rows, identity_group=groups,
                            calibration_mask=cal_mask, raw_truth=t, raw_generated=g,
                            calibrated_truth=a * t + b, calibrated_generated=a * g + b)
        make_plots(report, g[test], output)
        if run:
            metrics = flat_scalars({key: report[key] for key in ("fit", "calibration", "evaluation", "protocol", "dgpo_interface")})
            for index, side in enumerate(("lo95", "hi95")):
                for key in ("bce_delta_ci95", "raw_log_mean_ratio_ci95", "calibrated_log_mean_ratio_ci95"):
                    metrics[f"evaluation/{key}/{side}"] = report["evaluation"]["uncertainty"][key][index]
            run.log(metrics)
            run.summary.update(metrics)
            run.summary.update(dict(complete=True, phase="complete", decision=report["decision"]))
            run.log({f"diagnostics/{name}": wandb.Image(str(output / f"{name}.png"))
                     for name in ("reliability", "log_ratio", "calibration_fit")})
            run.save(str(output / "report.json"), base_path=str(output))
            run.save(str(output / "calibrator.json"), base_path=str(output))
            # Per-event scores/identities stay local; never upload scores.npz.
        (output / "COMPLETE").write_text(SCHEMA + "\n")
        print(json.dumps(dict(a=a, b=b, decision=report["decision"],
                              calibration_events=len(cal), evaluation_events=len(test),
                              raw_test_bce=report["evaluation"]["raw"]["bce"],
                              calibrated_test_bce=report["evaluation"]["calibrated"]["bce"],
                              paired_delta_ci95=report["evaluation"]["uncertainty"]["bce_delta_ci95"],
                              auc_preserved=report["evaluation"]["auc_preserved"]), indent=2, allow_nan=False), flush=True)
        print(f"REPORT: {output / 'report.json'}", flush=True)
    except BaseException:
        if run:
            run.summary.update(dict(complete=False, phase="failed"))
            run.finish(exit_code=1)
        raise
    else:
        if run:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
