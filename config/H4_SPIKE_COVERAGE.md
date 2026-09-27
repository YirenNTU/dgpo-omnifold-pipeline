# Frozen step-1110 spike coverage

## Full-pretrained raw checkpoint comparison

Use the full-pretrain source configured in
`dgpo_omnifold_ztautau_fullpretrain_40pct_raw_plateau_vpkl.yaml`:
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_v1/checkpoints/last.ckpt`.
This is NOT the unrelated `checkpoints.20M.a4.last.ckpt` base-config fallback.
Only raw weights are loaded; incompatible keys/shapes/dtypes stop the replay.
No new training, EMA, event cuts, smoothing, or sampler change.

```bash
shifter python3 -u scripts/diagnose_h4_spike_coverage.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_fullpretrain_raw \
  --sample --arm pretrainedfull \
  --matched-baseline /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110 \
  --compare-pretrained10pct /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_pretrain10pct_raw \
  --workers 16 --events 1024 --batch-size 16 --seed 42017 \
  --run-id h4covfull1
```

Both completed comparison directories must pass the fixed-panel/settings
checks. The runtime architecture and normalization-file configuration are
inherited from h4cov1, while checkpoint-stored buffers are loaded raw. This
does not establish that the original pretraining normalization or training
budget was identical; source provenance still needs inspection before a
dataset-size causal claim. Full-pretraining/evaluation overlap is unverified
and explicitly recorded in W&B config. Treat this as exploratory checkpoint
coverage, not an independent held-out generalization result.

W&B: `Does full pretraining recover spike coverage? | full-pretrain raw | matched K=128`.
Primary comparison is K=128 joint draw fraction at 1e-4 rad versus 10% raw;
report all thresholds and both projections. `paired_vs_pretrained10pct.json`
contains full-minus-10% paired differences and event-level SE; W&B summary
includes K=128 differences and SE. `paired_comparison.json` retains comparison
to step1110. Small hit-count changes alone do not establish improvement.
If full pretraining remains similarly deficient, data quantity alone is not
demonstrated sufficient by these checkpoints; sampler/truth-definition checks
remain open. If it improves, training exposure/budget becomes a stronger lead,
subject to normalization, architecture and event-overlap confounding.

## Matched pretrained 10% raw arm

Run after completed `h4cov1`. Only the checkpoint changes, to
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`,
the source path recorded in `dgpo_omnifold_ztautau_10pct_old_method_hard_ablation.yaml`.
This is a mutable `last.ckpt` path: the repo establishes the configured lineage,
not that its remote contents have never been replaced. The loaded global step
is recorded in W&B; no hash requirement is imposed.

The arm reuses the completed baseline's saved runtime and panel. It rejects
different event identities/values, seeds, GPU count, batch size or DDIM budget.
Both arms now instantiate the model directly and strictly load only raw
`state_dict`, verify every tensor, and never invoke the EMA-aware loader.
EMA data in the checkpoint is ignored. No classifier fitting or policy updates.
Normalization, architecture, eval mode and 20-step sampler remain matched.

```bash
shifter python3 -u scripts/diagnose_h4_spike_coverage.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_pretrain10pct_raw \
  --sample --arm pretrained10pct \
  --matched-baseline /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110 \
  --workers 16 --events 1024 --batch-size 16 --seed 42017 \
  --run-id h4covpre1
```

W&B display name: `Did DGPO lose spike coverage? | pretrained 10% raw | matched K=128`.
Adds `paired_comparison.json`: pretrained minus step1110 joint draw fraction
for each fixed threshold/K, with paired event-level standard errors. Positive
means pretrained has more spike mass, not automatically overall better physics.
Existing `artifact_coverage.json` still describes the historical step1110 ratio
artifact; the new pretrained results are in `coverage_report.json`.
The test is exploratory, not a new held-out model-selection test. A better
pretrained joint fraction supports deterioration during fine-tuning; equally
poor coverage shows the gap predates DGPO. Neither settles truth provenance.

Question: can unchanged policy samples reach the narrow back-to-back truth
region? No classifier fit, policy update, ratio clipping, calibration or
architecture change. This is an exploratory reused-test diagnostic.

Motivation: the supplied topology-resolution report finds truth median
acoplanarity 3.74e-5 rad versus generated 7.13e-3, and truth median
acollinearity 2.21e-4 versus generated 8.40e-3. Raw reweighting improves
azimuthal W1 but worsens opening-angle W1. The old cosine16 JSD hides
endpoint structure. These observations suggest a finite-sample coverage gap;
they do not prove absent mathematical support or a truth-data bug.

## Protocol

- Full saved test: count truth/generated fractions below 1e-6, 1e-5, 1e-4 rad
  separately for acoplanarity, acollinearity, and their intersection.
- Select 1,024 test events using seed 42017, independent of truth and logits.
- Restore the source checkpoint from `resolved_ratio_experiment.yaml`;
  require global_step=1110 and exact policy tensor equality (no SHA, no EMA
  substitution). Normalization and DDIM validation steps come from the merged
  original base plus saved overlay, not current classifier-experiment defaults.
- 16 GPUs, 64 events/rank, batches of 16, 128 sequential candidate chains.
  Rank seeds = 52017 + rank. Keep worker count and batch size fixed between
  comparisons; changing them changes the random-number stream.
- Compare nested K=1/8/32/128 from the same 128 draws. The K=1 replay is a
  fresh draw, not a bitwise reproduction of the historical candidate.
- Primary: joint draw fraction at 1e-4 rad versus truth fraction. All three
  thresholds and both projections must be reported, not selected afterward.
- Secondary: per-event any-hit fraction, and any-hit conditional on the
  observed truth being in the region. Draw-fraction SE clusters by event;
  zero observed hits yield zero empirical SE, NOT certainty or a support bound.

Interpretation: increasing any-hit alone demonstrates candidate availability,
not improved policy distribution. If K=128 still rarely reaches the region,
investigate generator resolution and upstream truth construction before further
ratio tuning. If hits are frequent but raw weights remain concentrated, coverage
alone is insufficient and ratio estimation/conditional ranking remains open.
Do not tune the classifier against this repeatedly inspected test set.

## Run on the existing NERSC Shifter/Ray 16-GPU allocation

```bash
shifter python3 -u scripts/diagnose_h4_spike_coverage.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110 \
  --sample --workers 16 --events 1024 --batch-size 16 --run-id h4cov1
```

Requires running Ray cluster (`RAY_ADDRESS` or Ray auto-discovery), the source
checkpoint, original normalization and base-config dependencies. Refuses an
existing output directory; a retry must use a new directory and W&B ID.
W&B: `Can the policy sample the sharp spike? | step-1110 replay | K=128`.
Sampling progress is rank-0 local; final coverage metrics aggregate all ranks.

CPU-only exact fractions (does not require GPUs or W&B):

```bash
shifter python3 -u scripts/diagnose_h4_spike_coverage.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_spike_counts_1110
```

Outputs: `artifact_coverage.json`, and for sampling `panel.pt`, `candidates.pt`,
rank shards, `runtime.yaml`, `manifest.json`, `coverage_report.json`, `COMPLETE`.
No source artifact or checkpoint is overwritten. `COMPLETE` is written last.

## Truth provenance limitation

The production reconstruction adds target delta-theta/delta-phi to visible
directions and assumes radians. Independent reconstruction agrees with the
saved topology, but that does not validate the upstream truth definition.
The ratio artifact exports pool/test row indices, not original Parquet event
IDs or generator-process labels. Do not interpret these rows as source event
indices. A source-level investigation requires mapping back to the filtered
Parquet and its upstream target-construction code; whether this spike is
physical, constrained by construction, or a preprocessing artifact remains
unresolved. Prediction-export post-calibration is not evidence that truth was
post-calibrated.
