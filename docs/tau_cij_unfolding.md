# Saved-sample Cij unfolding uncertainty study

Run personally on NERSC in the environment containing ROOT, RooUnfold,
NumPy, PyArrow, Torch, PyYAML, Matplotlib and W&B:

```bash
python3 -u scripts/run_tau_cij_unfolding.py config/tau_cij_unfolding.yaml
```

Use `shifter python3` instead only if that container has RooUnfold installed.
If not already discoverable, set `roounfold_library` to the actual shared
library path. No GPU inference, Ray cluster or new training is required.
The source is the saved raw pretrain epoch214 and DGPO step890 panel, not
the latest checkpoint. Input filter manifests are verified.

## Fixed protocol

- Deterministic, disjoint 50/50 response/test event split, stratified by the
  pair of signed parquet analyzing powers. Save IDs for reproducibility.
- Separate SVD responses for every kappa pair, component, and model. Start
  with ten angular-product bins; merge adjacent sparse bins using ONLY the
  response split, across truth and both models and all kappa pairs. Require
  at least 20 positive-weight entries and effective entries per bin. Both
  models use identical final edges for each component. Use k=min(5, bins).
  Save initial/final occupancy, removed edges, kappa pairs and actual k in
  binning.json and W&B. Nonuniform bin centers enter the moment estimator;
  discretization shift is still reported, not silently corrected.
- Source/test are within the existing development validation population;
  this is not a new blinded final test sample.
- Use original nonnegative MC weights to build responses and to define
  pseudo-data sampling probabilities. No truth-based response retuning.
- Full-truth event moment exponential tilts target -0.5 to +0.5. Each
  component has its own nominal case and seven targets. Other components
  and channel fractions are not held fixed.
- 100 independent Poisson event-count experiments per point (expected
  count equals the test split size), paired between models. Unlike the
  earlier fixed-N resampling, total count fluctuates. This matches the
  Poisson measured-bin covariance passed to RooUnfold.
- Unfold each group's angular histogram, then combine counts with the
  appropriate 9/(kappa_a*kappa_b). Preserve full within-group bin covariance
  and propagate the derivative of the normalized moment. Group blocks are
  independent under Poisson sampling. No 9-component joint covariance claim.
- 30 Poisson bootstraps of the independent response events estimate finite
  response-MC uncertainty. These responses are reused across targets; they
  are statistical replicas, not target-trained responses.

## Reading results

`sigma_stat`: covariance-propagated data statistical uncertainty.
`sigma_response_mc`: response bootstrap standard deviation on expected data.
`sigma_stat_mc_quadrature`: approximate quadrature combination of those two,
NOT total uncertainty including model or regularization systematics.
`coverage68_stat_fixed_response`: empirical statistical interval coverage
against the fixed test-population exact truth moment, conditional on the
nominal response. It is NOT coverage of the combined uncertainty.
`truth_binned - truth`: separately reported bin-center discretization shift.
`bias`: expected-data unfolded moment minus exact truth moment.

W&B logs a full results table and nine-panel plots of actual C, bias,
combined statistical/MC uncertainty and statistical-only coverage. Raw
pseudoexperiment estimates, response-bootstrap estimates, split IDs and a
JSON report remain in a unique NERSC output directory.

If support remains insufficient at two bins, or a bootstrap has empty truth
bins, fail explicitly; do not silently remove decay groups or replicas.
Unsupported targets also fail. SVD k=5 is a first fixed diagnostic, not an optimized
regularization choice. ROOT's full covariance API supports both old
Hreco/Ereco and newer Hunfold/Eunfold spellings.

This is selected-population angular-moment closure, not inclusive
efficiency/acceptance unfolding or a certified physical spin-state injection.
The reference implementation is TT2L-QC-Study; uncertainty propagation and
independent splitting deliberately improve on its demonstration script.
RooUnfold documentation: https://github.com/roofit-dev/RooUnfold

## Validate the tool before ranking models

```bash
python3 -u scripts/validate_tau_unfolding.py config/tau_cij_unfolding.yaml
```

Uses the same ROOT library and saved source, no model updates. New output and
W&B run. Exact-inversion identity and asymmetric migration controls check bin
counts, normalization, response orientation and complete Poisson covariance
against NumPy analytic answers. Signed mixed-kappa normalization checks use a
finite-difference Jacobian. If these hard controls fail, stop before physics.
300 Poisson pseudo-experiments per synthetic case give descriptive uncertainty
checks. They are not hard pass/fail coverage thresholds.

SVD identity/migration results are separately reported at two k values:
regularized SVD is not assumed to equal an exact inverse even at k=nbins.
Real-data diagnostics use both complementary folds of the deterministic split:
each of the 119002 events is tested once per model, while its response is built
only from the other half. Each fold chooses bins only on its own response split.
The folds share the source population in opposite roles, not independent
replicas; no naive independent averaging of uncertainties is performed.
Check response-self closure vs independent nominal closure at
configured k and k=nbins. No tuning on shifted targets and no truth injection.
Real Cij comparisons separate exact-event truth from bin-center truth, report
histogram residuals, total-normalization ratio, covariance eigenvalue and
response condition number. This does not certify all unfolding systematics.
Use `--synthetic-only` to test ROOT without loading saved event data.

## Response statistics convergence (fixed test)

```bash
python3 -u scripts/diagnose_tau_response_statistics.py config/tau_cij_unfolding.yaml
```

Reuses the original fold-0 split, no new inference. The independent test set
stays fixed. Nested, kappa-stratified response subsets use 25%, 50%, 100% of
the available response half (approximately 14900, 29700, 59500 events).
The bin edges are selected once from that full response half, identically to
the prior test, and held fixed across sizes and models. No target-based tuning.

At each size, compare the fixed-test expected-data closure residual with the
response-only spread from 30 Poisson bootstraps within that nested subset.
The same bootstrap event multipliers are paired across models/components.
This is not 30 fresh independent datasets, nor an unconditional physical bias
estimate. Residuals need not shrink monotonically for one nested realization.
Full-pool bootstrap spread need not go to zero.

Plots show closure residual (response-only percentile bands), response-MC
standard deviation, and conditional statistical sigma. Any failed bootstrap
invalidates that case's uncertainty summary; failures and counts are saved,
not silently discarded. This is particularly informative at the smallest
scale where empty response bins can recur. No automatic rebinning per size.
All event identities, replica values and binning are saved locally; W&B gets
the report, plots and case table. More than 59500 independent response events
requires additional verified filtered events and corresponding predictions.
