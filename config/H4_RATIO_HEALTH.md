# Fixed best-BCE classifier: independent ratio health audit

Implemented; not run on NERSC. The prior h4scllong1 run did not persist its
best classifier. This authorized same-seed replay preserves the successful
standardized architecture and 1000-minimum / 3000-maximum fit, constant learning
rates, ten-epoch BCE patience, step-1110 policy, clean data, and 16 workers.
It adds export and post-fit measurement. The user subsequently changed the
patience improvement threshold from 1e-4 to **1e-3** for this arm only.
The 1000-update minimum, ten-epoch patience and 3000-update cap are unchanged.
Patience resets on a BCE drop greater than 0.001 relative to its last qualifying
reference; smaller improvements can accumulate toward that threshold.
Best-checkpoint selection still saves every strict validation-BCE minimum.
This is no longer an exact stopping-rule replay of h4scllong1. Historical
quick/long arms retain their original threshold. Zero policy updates/reward fits.

## Run in the existing 16-GPU Ray allocation

```bash
shifter python3 scripts/train_h4_output_scaling.py --arm ratio-health --validate-only && \
shifter python3 scripts/train_h4_output_scaling.py --arm ratio-health
```

W&B ID `h4ratio1`; name `Are the learned ratios usable? | H4 held-out reweighting | best BCE`.
All settings are saved to `resolved_ratio_experiment.yaml` in
`/pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/`.
Sync modified trainer, fit config/stage, evenet_ratio, new ratio_health module,
launcher and report script together. Do not reuse an existing output directory.

## Artifacts and leakage controls

`ratio_audit/best_classifier_and_test.pt` includes full model weights (including
fixed standardizer buffers), packing specification, fit config, selected best
step, fit/early-stop/test indices, complete split identity conditions, and test
conditions, physical candidates, logits and topology values. These are local
research data, not uploaded to W&B. This is a weight/score artifact, **not** an
optimizer-state resume checkpoint. The sibling resolved YAML supplies complete
architecture/source provenance needed to reconstruct the model.

Export happens after fit_density_ratio restores best BCE, before the temporary
classifier is discarded. Requires independent test (no early-stop reuse), K=1,
unit generator weights and best-BCE selection. Scoring remains rank-sharded;
only rank zero exports the globally gathered evaluation scores. Failures are
synchronized and fail the run rather than silently losing artifacts. COMPLETE
is written last; reject incomplete directories. No test-driven temperature or
checkpoint selection. Context-hash splits keep event identities together.

## Prespecified endpoints

Primary: raw-ratio weighted vs unweighted joint histogram JSD of tau
`cos(delta_phi)` and physical opening-angle cosine, using fixed 16-by-16 bins
over [-1,1]. Negative change supports better agreement on this projection,
not complete high-dimensional closure. Report a paired event-bootstrap 95%
interval (200 resamples); interval below zero supports improvement conditional
on this fixed model. Bootstrap does not cover training uncertainty or unseen
rare tails. Marginal projections are secondary diagnostics.

Report separately for raw r=exp(logit) and fixed r^0.75 (sensitivity only):
ESS/N, top 1% and maximum normalized weight mass, log-ratio quantiles/max,
log of the UNNORMALIZED mean ratio with bootstrap interval and relative SE.
For true p/q with adequate support E_q[r]=1; zero should be compatible with
raw log-mean within sampling uncertainty. This identity is NOT expected for
r^0.75 and is not guaranteed by artificially normalized plotting weights.
ESS is concentration, not accuracy; no universal cutoff is imposed.

Held-out balanced BCE is logged independently from early-stop BCE. AUC is not
the primary criterion. All audit scalars go under
`omnifold_live/raw_staleness_audit/ratio_health/` in W&B and to local report.json.
The classifier used related physics features, so topology improvement is a
held-out reweighting check, not an independent fresh-classifier closure test.

Recompute after the run without fitting or GPU work:

```bash
shifter python3 scripts/report_h4_ratio_health.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit
```

## Post-hoc tail attribution (no training, no W&B mutation)

The h4ratio1 local artifact reported raw ESS about 9/23927, top-1% mass 82.6%
and one event mass 21.7%, despite test BCE 0.36855. The next question is whether
these influential events show data anomalies or a sparsely sampled physical
region. No classification, clipping threshold or temperature is fitted here.

```bash
shifter python3 scripts/diagnose_h4_ratio_tail.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_tail_report.json
```

The report lists top 20 generated-weight events with test/pool row indices,
paired truth/generated logits, physical angular deltas, topology, valid-input
ranges, individual BCE contributions and cumulative weight mass. Pool indices
refer to saved identity_condition, not original Parquet event IDs. State-dict
normalizer buffers yield only affine/pre-ICDF values; missing transformations
are labeled, not invented. Valid masks exclude padded values from input scans.

Sensitivity at removal counts 0/1/5/20 recomputes raw and fixed-0.75 ESS, mean
ratio and joint JSD. It distinguishes fixed-truth comparisons (only generated
points removed) from paired removal (both truth/generated events removed,
changing the target population). Neither is permission to deploy clipping.
Top-event fixed 16x16 topology cells report truth/generated counts and raw
weight mass. These are coarse marginal coverage diagnostics, NOT evidence of
conditional high-dimensional support or a calibration proof. Regions are
selected post hoc; no significance claims or parameter choices use this test.
Output creation is exclusive; source artifacts and model weights are unchanged.

## Coordinate and resolution audit after tail localization

Tail attribution found 23922 truth and 23913 generated events (of 23927)
in the same coarse topology cell. The earlier relative JSD improvement is
therefore weak evidence about joint structure. Next run, without training:

```bash
shifter python3 scripts/diagnose_h4_topology_resolution.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/topology_resolution_report.json
```

Reconstruct visible angles from momentum, add the physical candidate angular
deltas under the production radians contract, and calculate opening angle via
vector cross/dot atan2. Compare independently reconstructed sin(delta_phi),
cos(delta_phi), and opening cosine against saved topology. Report zero-momentum
events, noncanonical theta counts, and coordinate quantiles. Matching saved
features verifies implementation consistency, NOT original Parquet units.
No degree-to-radian conversion is guessed from numeric ranges.

Report every prespecified resolution, not the best-looking one: cosine 16/64/128
bins, linear-angle 64 bins, and fixed logarithmic angle edges spanning 1e-6 to
pi plus zero. Angular axes are acoplanarity = pi-|wrapped delta_phi| and
acollinearity = pi-opening. All histogram counts must cover every event.
Occupancy and sparse-cell fractions expose unreliable fine-binned JSD.
Exact one-dimensional weighted Wasserstein distances in radians are included
as bin-independent projections, not high-dimensional closure measures.
Top20 fine-cell truth/generated counts remain marginal coverage diagnostics.
This post-hoc reused-test analysis never changes weights, classifier, bins,
temperature or calibration based on its outcome. No automatic physics/unit fix.
