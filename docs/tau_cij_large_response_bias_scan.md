# Large-response Cij bias scan

Run in the NERSC ROOT/RooUnfold environment already used for the tool-validation
and response-statistics tests. No new inference, training, or GPU allocation is
needed: both raw models' saved samples are reused. Existing configuration and
previous output directories are unchanged. Every invocation creates a new W&B
run and output directory.

```bash
python3 -u scripts/diagnose_tau_unfolded_bias_scan.py \
  config/tau_cij_unfolding.yaml \
  --response-source /pscratch/sd/y/yiren/Ztautau/tau_cij_fullval_response
```

## Fixed protocol

- Original fold-0 test: 59,505 events. Expanded, disjoint nominal response:
  476,010 events. Verify matching raw checkpoints, sampling settings, kappa
  groups and event identities. The completed response root resolves through
  `ready_samples.json`.
- Retain the original response-only binning and SVD k from the prior diagnostics.
  New response data must not change the measurement definition. No test-target
  tuning, retraining, bias subtraction or failed-replica dropping.
- Scan each of nine Cij separately at seven absolute targets from −0.5 to +0.5,
  plus nominal (144 cases including two models). Other moments can change along
  with the targeted component; this is not independent physical control of all
  entries of a spin density matrix.
- Truth-only exponential tilt defines event selection probabilities. Generate
  100 Poisson pseudo-data sets, expected size 59,505, with unit event counts.
  Thus the actual count fluctuates; this is not exactly fixed-N sampling.
  Use the same sampled events for pretrain and DGPO. Repeated selections of the
  same saved event reuse its saved prediction, not a fresh diffusion draw.
- Build a nominal response and 30 Poisson response-MC bootstraps per component
  and model, once, reused for every target. Identical event multipliers are
  used across models. Response construction may take substantially longer than
  the previous fixed-response-only test.

## Read the output

`bias_scan/plots/actual_cij`: injected truth vs unfolding of the expected measured
histogram. Dashed diagonal is ideal. Thick inner error bars are data statistics;
thin outer error bars add response-MC variance in quadrature. These are one-sigma
measurement uncertainties, **not** standard errors of the 100-trial mean.

`bias_scan/plots/bias`: pseudoexperiment mean minus the exact injected moment.
The report also retains the binned-truth value and bin-centre approximation shift.

`bias_scan/plots/coverage68`: two distinct experiments:

1. `stat_*`: pseudo-data fluctuated, nominal response fixed; statistical errors only.
2. `joint_bootstrap_*`: both pseudo-data and response fluctuated, with combined
   errors. Each trial chooses one of the 30 response replicas; replicas are reused.
   This is a finite-bootstrap approximation, not independent fresh-MC coverage.

Response-MC sigma is the spread of bootstrap estimates on the expected measured
histogram for that scenario. Combined errors are the quadrature of this spread
and the trial's propagated statistical error. The approximation neglects higher
order interactions and does not include detector/physics systematics or residual
regularization bias. No bias correction is applied.

Rows include expected Cij/bias, statistical and MC sigmas, empirical spread,
pull mean/width, 68%/95% coverage and sampling ESS. `mean_mc_se` only measures
finite-trial precision of an ensemble mean. A nominal binomial coverage error
does not include uncertainty from the finite bank of response bootstraps.

`bias_scan_report.json`, identities, per-case trial arrays (including sampled
truth, response-replica IDs, sampling probabilities and bootstrap estimates), and
plots are saved locally. The report, plots and case table are uploaded to W&B;
large trial arrays remain on NERSC.

This tests closure within the available empirical selected population. It does
not establish real-data robustness, inclusive acceptance correction, or certify
the physical polarimeter/kappa convention. Resampling the finite test panel
cannot separate its underlying population fluctuation from physical model bias.

## Local checks

```bash
python3 -m unittest scripts.test_tau_unfold_core scripts.test_tau_unfolded_bias_scan
```

These checks exercise sampling, pairing, identity moments, variance accounting
and failure handling without ROOT. The production RooUnfold path requires the
NERSC environment; successful local tests do not certify a completed scan.
