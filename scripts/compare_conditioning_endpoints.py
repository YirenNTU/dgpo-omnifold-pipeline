#!/usr/bin/env python3
"""Compare saved epoch-50 coverage samples; no model execution or training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def paired_joint_w1(truth_angles, control_angles, candidate_angles, *, bootstrap=1000, seed=42017):
    truth, control, candidate = [np.asarray(x, dtype=np.float64)
                                 for x in (truth_angles, control_angles, candidate_angles)]
    if (truth.ndim != 2 or truth.shape[1] != 2 or len(truth) < 2
            or control.ndim != 3 or control.shape[0] != len(truth)
            or control.shape[2] != 2 or control.shape[1] < 1 or candidate.shape != control.shape
            or bootstrap < 20 or not all(np.isfinite(x).all() for x in (truth, control, candidate))):
        raise ValueError("Expected finite aligned N x 2 truth and N x K x 2 candidate angles")
    t, a, b = truth.max(-1), control.max(-1), candidate.max(-1)
    def contrast(ids):
        reference = t[ids]
        return (wasserstein_distance(reference, b[ids].reshape(-1))
                - wasserstein_distance(reference, a[ids].reshape(-1)))
    rng = np.random.default_rng(seed)
    draws = np.array([contrast(rng.integers(len(t), size=len(t))) for _ in range(bootstrap)])
    return dict(control_w1=wasserstein_distance(t, a.reshape(-1)),
                candidate_w1=wasserstein_distance(t, b.reshape(-1)),
                candidate_minus_control_w1=contrast(np.arange(len(t))),
                paired_event_ci95=np.quantile(draws, [.025, .975]).tolist(),
                bootstrap=bootstrap, bootstrap_seed=seed, events=len(t), draws_per_event=a.shape[1],
                resampling_unit="whole event with all K draws and its truth",
                interpretation="Negative favors candidate; exploratory pointwise interval, not training-seed uncertainty")


def load_endpoint(path, panel):
    from diagnose_h4_ddim_coverage import analyze_arm
    report = json.loads(path.read_text())
    if report["completed_epochs"] != 50:
        raise ValueError("Compare the predeclared completed-epoch-50 endpoints")
    saved = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    cfg = report["config"]
    if saved["generated"].shape != (cfg["events"], cfg["K"], 4):
        raise ValueError("Saved generated sample count or candidate order is incompatible")
    result, truth_angles, generated_angles = analyze_arm(panel, saved["generated"])
    if not np.allclose(saved["angles"].numpy(), generated_angles, rtol=0, atol=1e-12):
        raise ValueError("Saved angles do not reconstruct from this panel")
    for family in ("topology", "target_marginals"):
        for name, entry in result[family].items():
            if not np.isclose(entry["w1_radians"], report["result"][family][name]["w1_radians"],
                              rtol=1e-10, atol=1e-12):
                raise ValueError(f"Saved report does not reproduce: {family}/{name}")
    return report, result, truth_angles, generated_angles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True, help="relcontext01 epoch-0050.json")
    parser.add_argument("--candidate", type=Path, required=True, help="New arm epoch-0050.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file already exists; choose a new filename")
    metadata = [json.loads(p.read_text()) for p in (args.control, args.candidate)]
    if metadata[0]["config"] != metadata[1]["config"]:
        raise ValueError("Different coverage protocols/panel paths; cannot claim paired comparison")
    panel = torch.load(metadata[0]["config"]["panel_path"], map_location="cpu", weights_only=True)
    for key in ("test_rows", "pool_rows"):
        if len(panel[key].unique()) != len(panel["truth"]):
            raise ValueError("Duplicate or missing panel identities")
    before, result_a, truth_a, angles_a = load_endpoint(args.control, panel)
    after, result_b, truth_b, angles_b = load_endpoint(args.candidate, panel)
    if not np.array_equal(truth_a, truth_b):
        raise ValueError("Truth alignment changed")
    from diagnose_h4_ddim_coverage import paired_comparison
    report = dict(control=str(args.control), candidate=str(args.candidate), completed_epochs=50,
                  control_global_step=before["global_step"], candidate_global_step=after["global_step"],
                  primary=paired_joint_w1(truth_a, angles_a, angles_b, bootstrap=args.bootstrap),
                  coverage=paired_comparison(truth_a, angles_a, angles_b, bootstrap=args.bootstrap, seed=42017),
                  control_result=result_a, candidate_result=result_b,
                  notes="Absolute endpoint comparison. Two candidate intervals are not familywise adjusted. "
                        "Panel was reused; confirm a selected improvement independently. No H4 closure claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["primary"], indent=2))


if __name__ == "__main__":
    main()
