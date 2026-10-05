# Saved-score channel diagnostics

Add `--channel-diagnostics` to `scripts/rescore_film_cij.py --analysis-only DIR`.
This reads the existing candidate bundle, aligns parquet event categories by source
identity, and starts a new W&B analysis run. It does not launch Ray, regenerate
samples, refit classifiers, or change weights. Existing analysis reports in DIR
are refreshed; candidate scores and checkpoints remain unchanged.

Outputs: `raw_channels.json` and `calibrated_channels.json`, uploaded to W&B,
plus W&B channel tables. Each channel includes full-truth and matched-target
closure, base/reweighted population fractions, ESS and paired-bootstrap intervals.
Zero-weight channels and channels with fewer than two events are explicitly marked.

Let f0 be each channel's original base-weight fraction, f1 its reweighted fraction,
C0 its unweighted matrix, and C1 its within-channel reweighted matrix. Define
the diagnostic fixed-mixture matrix as sum(f0*C1). Then the signed matrix shift is

    global C1 - global C0
      = [sum(f0*C1) - sum(f0*C0)] + [sum(f1*C1) - sum(f0*C1)].

This is exact descriptive accounting, not causal attribution. Matrix norms do
not add. The fixed-mixture comparison is not an alternative deployed ratio.
Decomposition quantities are point estimates; per-channel intervals condition
on the fitted classifier and use paired event bootstrap, not refit uncertainty.
Intervals are pointwise, without multiple-comparison correction. No claim of
fully acceptance-corrected physical Cij follows from these diagnostics alone.

## Direct angular moments (no analyzing power)

Each computed channel now also contains `angular_moments`: the actual nine
means M_ij = <a_i b_j>, with neither division by kappa nor multiplication by 9.
The same event weights, candidate ratios, references and paired bootstrap are
used. W&B `angular_channels/{raw,calibrated}/matched_target` tables list the
target, unweighted and reweighted moments with intervals, and paired changes
in absolute error for all 16 x 9 components. Negative change means improvement.
Full numerical reports also retain alternative truth references. These are
pointwise intervals, not simultaneous tests; no refit uncertainty is included.

Run the same `--channel-diagnostics` command to include these additional results.
No new samples or classifier fitting are needed. Reports are refreshed in place
and uploaded to a new W&B run. For a constant kappa product within a channel,
removing it only rescales the norm and cannot change the improvement sign;
the new value is localizing which angular components move, not an independent
test of the same per-channel norm conclusion.
