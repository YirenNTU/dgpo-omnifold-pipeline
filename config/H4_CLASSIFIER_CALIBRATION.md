# Frozen classifier calibration diagnostic

Question: are extreme classifier weights associated with the narrow angular
structures that production calibration removes? This is a descriptive
diagnostic, not a classifier training experiment or physics closure test.

Uses the exported best H4 classifier and its original held-out K=1 candidates.
No new features, fits, policy updates, clipping, normalization fitting, or EMA
checkpoint loading. Both truth and generated candidates undergo the same
`common.post_calibrate_tau_tau` projection. Context is unchanged; physical
delta encoding and candidate-dependent Fourier/decoder features are recomputed.

## Run on an existing 16-GPU NERSC Ray allocation

From the repository directory:

```bash
source scripts/nersc/start_interactive_ray.sh
shifter ray status --address="$RAY_ADDRESS"

shifter python3 -u scripts/diagnose_h4_classifier_calibration.py \
  /pscratch/sd/y/yiren/Ztautau/h4_scale_standardized_long-ratio-health/ratio_audit \
  --output /pscratch/sd/y/yiren/Ztautau/h4_classifier_calibration \
  --score \
  --runtime /pscratch/sd/y/yiren/Ztautau/h4_spike_coverage_1110/runtime.yaml \
  --workers 16 --batch-size 128 --run-id h4calclf1
```

Requires an active Ray cluster exported as `RAY_ADDRESS`, shared output
storage, and the same dependencies/normalization files as classifier training.
Ray connectivity and the requested GPU count are now checked before creating
the output directory or W&B run. `--ray-address host:port` may be used instead
of the environment variable.
All Ray Train modules are also imported before the output directory is
reserved. A cold Shifter import can take time; interrupting it is not a model
failure, but the process should be allowed to finish unless it emits an actual
exception.
The runtime supplies architecture only; all trained tensors come from
`best_classifier_and_test.pt`. No policy checkpoint is loaded.
For the strict replay guard, the script deliberately overrides the requested
batch size with the artifact's saved `fit_config.validation_batch_size` and
uses the original contiguous 16-rank shard boundaries. The requested value and
effective saved value are both recorded in the manifest. The same exact layout
is then used after projection. This is required for comparable GPU kernels;
it does not change event membership or classifier weights.
Reruns may reuse the same output and W&B run ID. The script overwrites only its
owned files (`tail_attribution.json`, manifest/report/tensor outputs, rank shards,
`COMPLETE`, and `ray_results`) and preserves unknown files. W&B uses
`resume=allow`. The source ratio artifact can never be selected as the output.
Omit `--score` and `--runtime` for CPU-only tail attribution; use a separate
output directory. `--no-wandb` disables W&B.

## Safeguards and outputs

- `tail_attribution.json`: top-20/top-1% weight mass and feature summaries,
  Spearman associations, narrow-angle event fractions versus weight mass.
- Before transformed inference, each worker must reproduce saved original
  logits with `atol=rtol=1e-4`. A mismatch stops the experiment; do not loosen
  the tolerance without investigating runtime/architecture/numerical differences.
- Replay first loads the exact frozen EveNet body from the runtime's
  `reward_config.omnifold.backbone_checkpoint`, through the same production
  builder used during classifier training. The ratio artifact intentionally
  stores only the classifier bank and trainable backbone tensors; constructing
  a fresh random body and loading `model_state` is not a valid reconstruction.
- `frozen_classifier_report.json`: balanced BCE, AUC, logit shifts, score-weight
  ESS/concentration, paired event-level BCE change and standard error.
- `scores_and_projected_candidates.pt`: aligned event-level scores/candidates.
  Rows match the input artifact test rows, not original parquet row numbers.
- `manifest.json`: exact input/runtime paths and execution settings.
- `COMPLETE` is written only after successful distributed scoring/validation.
  Tail-only runs intentionally do not receive this full-experiment marker.
- W&B `h4calclf1`, display name
  `What does calibration remove? | frozen H4 classifier | paired test`:
  phase plus final before/after metrics. No training curves are expected.

## Interpretation declared before running

Primary readout: paired change in balanced BCE, with separate truth/generated
logit shifts. AUC and weight concentration are complementary diagnostics.
Large changes show sensitivity of this frozen classifier to the projection;
they do not prove calibration fixes the original distribution mismatch.
Small changes mean this classifier retains separation after projection, not
that narrow topology is irrelevant to every possible classifier.

The projection is many-to-one and may produce out-of-training-distribution
inputs. Therefore `exp(post-projection logit)` is only a score-derived weight,
not a validated density ratio for the projected distributions. Chance AUC or
better ESS is not closure. Calibration shift and opening deficit are linked
geometrically, not independent explanatory variables. The reused test set is
for diagnosis, not tuning followed by a claim of independent validation.

No Cij, analyzer vectors, decay-channel conventions, or physics weights are
invented or required. Local tests validate geometry, label direction, guards,
batch scoring, and offline output; actual checkpoint replay requires NERSC.
