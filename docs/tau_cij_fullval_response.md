# Expand response MC using the independent full-validation remainder

No job is submitted by the assistant. Start the usual 16-GPU Ray cluster in
the original EveNet image, then from the existing NERSC ml_pipeline checkout:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/sample_tau_fullval_response.py \
  config/tau_cij_fullval_samples.yaml
```

Optional CPU preparation only: append `prepare`. It does not need a Ray
cluster. This scans the canonical full val-diffusion and excludes all 119002
original filtered validation IDs. Those IDs must all be recovered in the
source; missing or duplicate IDs are errors, not silently deduplicated.
STIC filtering uses raw feature names from the pinned runtime, preserves all
retained columns, writes removed-event reasons and a completed filter manifest.
No raw dataset is used for model inference. The original data remain unchanged.

The original comparison's immutable raw pretrain epoch214 and DGPO step890
checkpoint snapshots and runtimes are reused. No last.ckpt selection, EMA,
classifier scoring, training or normalization recomputation. K=1, 16 workers,
batch size 1024/GPU; original DDIM steps and conditioning settings retained.
The completed event panel is cached; failed partial filtering is not certified.

W&B records the source/checkpoint settings, phase, generated event counts and
completed sample directory. Large parquet/prediction data remain on scratch,
not uploaded to W&B. Completion validates all 16 shard positions and finite
predictions. `ready_samples.json` points only to a fully completed generation.

Then in the ROOT/RooUnfold evnet_env (not necessarily the inference image):

```bash
python3 -u scripts/diagnose_tau_response_statistics.py \
  config/tau_cij_unfolding.yaml \
  --response-source /pscratch/sd/y/yiren/Ztautau/tau_cij_fullval_response
```

This keeps the original fold-0 test events and original response-derived bins
and k unchanged. Response subsets now come from the disjoint additional
population. Test/model/packing/DDIM consistency checks must pass. No mixing
of old training samples into the test; source population remains development
validation, not a new blinded dataset. Actual retained row count is measured,
not assumed from the approximate 20% fraction. Roughly 476k events are expected
before new filtering if the canonical source is unchanged.

Sync only real-case scripts/configs; repository-root uploads must retain
`--exclude-from=NERSC/upload-excludes.txt`. Do not overwrite the NERSC YAML's
configured RooUnfold library path when updating inference code.

## Fixed large-response pseudo-experiments

In the RooUnfold environment:

```bash
python3 -u scripts/diagnose_tau_fixed_response_pseudo.py \
  config/tau_cij_unfolding.yaml \
  --response-source /pscratch/sd/y/yiren/Ztautau/tau_cij_fullval_response
```

Freeze the complete additional response population and use the original
fold-0 test and original binning/k. No additional inference or response
bootstrapping. Config `repeats: 100` controls paired Poisson pseudo-data draws;
the same random event counts are used across models and components. Nominal
only: no reweighted-target scan or parameter tuning.

Compare empirical spread with reported statistical sigma, pull mean/width,
68%/95% coverage against exact and binned empirical truth. `centered_coverage68`
instead centers at the expected-data unfolded estimate: it tests statistical
calibration without requiring an unbiased estimator, NOT physical closure.
Mean-residual MC error describes precision of the pseudo-experiment mean, not
the precision of a real measurement. These fixed-response intervals exclude
response-MC and model/regularization systematics. Repeated resampling cannot
remove or identify the origin of an offset already present in the finite test
population. All raw trial estimates, sigmas and paired sampled-truth moments
are saved. A new W&B run logs tables and plots; no jobs are submitted automatically.
