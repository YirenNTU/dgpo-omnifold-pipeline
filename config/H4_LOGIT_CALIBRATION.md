# Frozen H4: positive affine logit calibration

Implemented locally; the real NERSC artifact has not been evaluated by this
implementation yet. One question: **does calibrating the numeric scale of the
existing H4 score improve held-out balanced BCE without changing its ranking?**

This is not the `h4calclf4` physics-projection test. We do not transform truth
or generated candidates, add features, reconstruct a model, refit the
classifier, update DGPO, or use EMA. We read the original saved logits from
the `h4ratio1` best-BCE classifier artifact. This is not the two-fold reward
classifier from `h4lbnr01`.

## Run once on CPU inside Shifter

From the updated repository on NERSC:

```bash
shifter python3 -u scripts/diagnose_h4_logit_calibration.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_logit_calibration \
  --run-id h4logcal1
```

No Ray cluster, runtime YAML, checkpoint reload, batch size, or GPU allocation
is needed. Do not launch one copy per GPU. The existing Shifter environment
needs PyTorch, NumPy, SciPy, scikit-learn, Matplotlib and W&B. `--no-wandb`
provides an offline run. The source must have its `COMPLETE` marker.

Reruns overwrite this diagnostic's eight named output files only and preserve
unrelated files. The source artifact is never overwritten. W&B resumes the
stable ID with `resume=allow`: latest summaries/files are replaced, but earlier
history points remain. Use a different ID if intentionally testing a different
protocol; do not choose a better-looking split seed after inspecting results.

## Data split and fitting

The original artifact contains 23,927 paired K=1 truth/generated test scores,
with identities held out of classifier fitting and early stopping. Validate
that separation against the saved split indices and full identity inputs.
Then split **event identities**, not score rows, 50:50 into calibration and
evaluation, seed `20260920`. Keep each truth/generated pair and duplicate
identity on the same side. No event is clipped or removed. Grouping uses the
production canonical float32 identity convention, not checkpoint hashing.

This pool has already been examined in previous experiments. The evaluation
partition is independent of fitting the new calibrator, but **is not a fresh
confirmatory dataset**. Both metadata and the decision label say exploratory.

Fit only two parameters on calibration rows:

```text
ell_cal = a * ell_raw + b,  a > 0
r_cal = exp(ell_cal)
BCE = mean_events[0.5 * (softplus(-ell_truth) + softplus(ell_generated))]
```

Truth is class 1; the balanced prior is 0.5, so no class-prior correction is
required for score odds. L-BFGS-B optimizes the convex affine-logit BCE with
analytic gradients, starting from identity `a=1,b=0`. Score centering/scaling
uses calibration rows only and is undone in the reported `a,b`. A numerical
slope floor of `1e-6` preserves ranking; hitting it is flagged inconclusive.
The 1,000-iteration numerical cap is not classifier training. Constant scores,
complete/quasi separation, failed convergence or increased calibration BCE
also prevent a positive decision. There is no ESS-based tuning, forced
mean-ratio normalization, regularization, clipping or temperature sweep.

## Readout and decision rule

Primary: evaluation `BCE_calibrated - BCE_raw`, using paired
identity-cluster bootstrap (500 replicates, seed `20260921`).

- Upper endpoint of the 95% interval below zero: exploratory evidence that
  score calibration improves BCE, provided the fit and ranking checks pass.
- Lower endpoint above zero: held-out BCE worsened.
- Interval crosses zero: no clear improvement at this evaluation precision.
- Degenerate/failed fit or changed AUC: inconclusive; investigate before use.

The uncertainty is conditional on this fixed classifier and fitted `a,b`.
It does not include classifier-training or calibrator-fitting uncertainty and
cannot discover rare tails absent from the saved sample. Do not interpret
resplitting the same pool as independent replication.

Secondary diagnostics: separate truth/generated BCE, Brier score, 15-bin
reliability/ECE and bin occupancy; AUC invariance; generated log-ratio
quantiles/max, ESS and ESS/N, top-1%/largest weight mass, unnormalized
log-mean ratio, identity-cluster relative SE and bootstrap intervals. Mean
ratio is omitted if exponentiating would be numerically unsafe; log mean
remains available. Every ratio statistic uses the same evaluation events.

AUC should be unchanged because `a>0`. A better BCE/reliability does not
certify full-dimensional or conditional density ratios. Score calibration
cannot recover information discarded by the classifier or repair missing
generator coverage. A mean ratio near one is a diagnostic under adequate
support, not an enforced objective or proof of correctness. ESS measures
concentration, not accuracy; a smaller ESS is not automatically failure.

## W&B and local outputs

ID `h4logcal1`; group `H4 ratio calibration`; display name:
`Can score calibration improve ratios? | frozen H4 | positive affine | CPU replay`.

Key summaries:

| Key | Meaning |
| --- | --- |
| `fit/a`, `fit/b`, `fit/eligible` | Final affine fit and numerical eligibility |
| `calibration_fit/bce` | Live optimizer trace using calibration data only |
| `evaluation/raw/bce`, `evaluation/calibrated/bce` | Independent evaluation BCE |
| `evaluation/delta/bce` | Primary paired change; negative is better |
| `evaluation/bce_delta_ci95/lo95`, `.../hi95` | Conditional paired 95% interval |
| `evaluation/auc_preserved` | Ranking sanity check |
| `evaluation/raw/ratio/*`, `evaluation/calibrated/ratio/*` | Tail/normalization diagnostics |
| `decision`, `complete`, `phase` | Interpretation and execution status |

`report.json` contains all metrics, bootstrap intervals, fit trajectory,
provenance, protocol and limitations. `calibrator.json` saves `a,b` without
deploying them. `scores.npz` keeps original/calibrated paired scores, partition
mask, identity groups and source-pool rows locally; pool rows are not original
Parquet event IDs. W&B receives aggregate metrics, three plots and the two
report/calibrator JSON files, never event-level scores or identities.
`manifest.json` records execution settings; `COMPLETE` is written last.

## DGPO boundary

Production training code and rewards are unchanged. For the current
log-ratio reward and `leave_one_out_unscaled` advantage:

```text
A_cal = a * A_raw                   (b cancels within each event)
M_cal = a * M_raw                   (same candidates and loss deltas)
w_cal = sigmoid(a * M_raw)          (not simply an LR rescaling)
```

These identities are regression-tested against the production advantage/gate
helpers. The artifact has K=1, so it cannot measure actual K=8 advantages,
policy gradients or gate changes. The report marks these implications
`measured=false`; it does not invent cross-event candidate groups. A positive
result here would justify a separate fixed-candidate DGPO replay, not automatic
deployment or a claim of improved physics closure.

References: [Guo et al., calibration of neural networks](https://proceedings.mlr.press/v70/guo17a.html)
and [Cranmer et al., likelihood ratios with calibrated classifiers](https://arxiv.org/abs/1506.02169).
