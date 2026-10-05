# Held-out spin-moment calibration

## Question and status

**Can a small physics-targeted correction of the existing context256/bound30
weights improve cross terms on disjoint events without sacrificing diagonal
closure, condition mass or ESS?** Implemented; not yet run on NERSC.

This is a calibration feasibility ablation, not a DGPO change or a replacement
for model-agnostic unfolding. No classifier/backbone/generator is retrained.
It explicitly uses simulated truth spin moments on calibration/selection events.
It cannot measure unknown real-data spin correlations by imposing simulation's
answer. A pass would support generalization of this targeted correction on this
simulation, not unbiased unfolding or full conditional tau-distribution closure.

## Evidence motivating the one intervention

Same 119002-event K64 panel: context256/bound30 total Cij error was 0.25648;
lower cap5 gave 0.39830 despite reducing cross-term norm from 0.21112 to 0.18284.
Explicit products (`gp8z7sjn`) improved BCE but did not establish better Cij
than its Geometry control (`n3jjqcn7`): total-error delta +0.00861,
pointwise95% CI [-0.09099, 0.09996]. No individual component difference resolved.
Do not infer that all cross terms improved or that coverage is established.

Sources: [cap/ESS](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/76w48b7i),
[products](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/gp8z7sjn).

## Fixed source and split

- Raw step1110, never EMA. Reuse frozen `zrv2yfgt` context256/bound30 scores.
- Reuse `fzekzrmr` completed **16-GPU** K64 panel: all119002 filtered validation
  events,64 saved candidates each. No inference on a single GPU, no new samples.
- Verify source run IDs, event ordering, complete outputs, raw1110 metadata,
  filtered train416701 and validation119002 manifests, and saved full-panel
  baseline endpoint. Preserve the source normalization and spin convention.
- BLAKE2 event-ID split, fixed seed100167: approximately40% calibration,
  20% selection,40% test. All candidates and the truth of an event stay together.
  No input event is dropped. Calibration normalization uses calibration only;
  pT group edges are the upstream classifier's fit-only edges.
- This panel has been inspected previously. The final partition is held out
  from **this calibration and selection**, but is NOT a pristine new blind test.
  Results are explicitly exploratory; previous full-panel errors are not the
  numerical baseline for the smaller final-test subset.

## Correction and objective

Let `f_ij = 9*a_i*b_j/(kappa_a*kappa_b)` be the **candidate's own** nine saved
Cij contributions, in order kk,kr,kn,rk,rr,rn,nk,nr,nn. No paired truth feature
is passed to the correction at inference. Scale/center these using unweighted
generated calibration candidates and inherited base event weights.

Only nine coefficients are learned:

```
delta = log(2) * tanh(dot(theta, standardized_candidate_f) / log(2))
corrected_log_ratio = min(log(30), baseline_log_ratio + delta)
w = global_normalize(base_event_weight / K * exp(corrected_log_ratio))
```

Zero theta is the identity. The raw ratio stays at most30 and differs from
baseline by at most a factor2 up/down. These bounds do not imply a cap30 on
globally normalized weights, nor do they guarantee adequate ESS.

Fit on the calibration split:

```
L = KL(w || w_baseline)
    + strength/2 * sum_all_9 (Cij(w) - Cij_truth_calibration)^2
    + 1e-4/2 * ||theta||^2
```

All nine moments get equal weight, not just whichever cross terms looked bad.
The weight KL limits disruption of the existing candidate distribution. This
KL is NOT the DGPO velocity/reference term. Raw classifier BCE is not optimized.
Condition mass is guarded in selection/test, not added as a second fit objective.
Three strengths0.1,1,10 start independently from theta0, with analytic-gradient
L-BFGS-B (coefficient bounds +/-5, maximum300 iterations). Nonconverged fits are
ineligible. The bounded nonlinear parameterization makes this a regularized
moment-calibration heuristic, **not an exact convex entropy-balancing solver**.
See [Hainmueller's entropy balancing](https://www.mit.edu/~jhainm/Paper/eb.pdf)
for the underlying moment-balancing idea, not a guarantee for this adaptation.

## Selection, endpoint and interpretation

On selection events only, candidates must improve cross-term norm and pass:

- candidate and event ESS fractions at least90% of matched baseline;
- category x visible-pT massTV increase at most0.005;
- diagonal error increase at most0.02;
- total Cij error increase at most0.01.

Among eligible converged fits choose lowest offdiagonal error; tie favors lower
strength. If none qualify, retain baseline. **Never relax gates after observing
the final test.** Save coefficients, IDs, split, and selection before test analysis.
No full-panel/application checkpoint is installed automatically.

Primary: selected-minus-baseline offdiagonal Frobenius error on final-test events.
2000 paired whole-event bootstrap resamples include all candidates and paired
truth. Negative95% interval supports cross-term improvement. A qualified pass
also needs observed ESS/mass guards and upper95% diagonal/total error changes
below the declared tolerances. Report all nine component changes with approximate
simultaneous family9 bands; do not cherry-pick a near-zero entry's percentage.
Intervals condition on the learned/selected correction; they do not include
calibration-fit/selection uncertainty. Moment/group closure is not full
conditional distribution closure. Failure is not proof of missing support:
the restricted nine-coefficient family, radius, cap, statistical noise, and
nonconvex optimization are alternative limitations.

## Run and outputs

From the existing NERSC ml_pipeline checkout, with Shifter configured:

```bash
shifter python3 -u scripts/diagnose_tau_moment_balance.py \
  config/conditional_tau_moment_balance.yaml
```

The source is16-GPU; this step uses16 CPU threads for saved-array arithmetic
and nine-parameter calibration. No Ray cluster, allocation or new GPU training
is needed. The user submits the command personally. Do not upload toy outputs.

W&B creates a new run named
`Can moment calibration repair cross terms? | frozen context256 | bound30 | held-out K64`.
Logs: live objective/gradient norm, calibration and selection errors, convergence,
selection guards, final all-nine Cij/component intervals, ESS/massTV, group errors,
and two figures. Large per-event arrays stay local.

Unique output under `/pscratch/sd/y/yiren/Ztautau/conditional_tau_moment_balance/`:
`calibration_model.json` (sealed before test), `split_ids.npz`,
`moment_balance_report.json`, `heldout_event_bootstrap.npz`, two figures, W&B link,
and COMPLETE only after success. Baseline/source files are never overwritten.

Local tests: `python3 -m pytest -q scripts/test_tau_moment_balance.py`.

Validation completed locally:10 new tests and130 tests in the combined ratio /
conditioning / saved-panel regression suite passed (RuntimeWarnings treated as
errors). Covers analytic gradients, identity/cap/factor bounds, coefficient
round-trip, source replay, event-ID split stability, test-truth noninterference,
fallback selection, plots and W&B publication. No NERSC experiment was submitted.
