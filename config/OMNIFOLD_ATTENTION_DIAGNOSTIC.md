# 1% versus 10% classifier initialization / attention test

This is a diagnostic, not a new optimization ablation. It compares the **v18
scaling controls** on both sides, not standalone OmniFold or the v19 epoch-refit
experiment. Base+overlay comparison currently finds matching classifier/model
settings apart from 10%'s anomaly detection; data, checkpoint and run paths differ.
Real fold batch sizes can still differ because the fitter caps small populations.

## 1. Inspect effective configuration differences

```bash
shifter python3 scripts/diagnose_omnifold_attention.py compare
```

`--one`, `--ten`, and `--base` accept other YAML paths. Comparison covers the
launcher merge, not checkpoint contents or subsequent data-dependent batch caps.

## 2. Capture on an allocated Ray cluster

Sync the updated trainer, ratio fitter, diagnostic module and script first.
Run these separately unless you have two independent 16-GPU allocations.

```bash
shifter python3 scripts/diagnose_omnifold_attention.py capture \
  --fraction 10 \
  --output-dir /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1
```

```bash
shifter python3 scripts/diagnose_omnifold_attention.py capture \
  --fraction 1 \
  --output-dir /pscratch/sd/y/yiren/Ztautau/omnifold_attention_1pct_test1
```

Output directories must be new; existing directories are never overwritten.
The launcher cold-starts from the selected pretraining checkpoint, resolves its
symlink and records its SHA256 in `runtime.yaml`. Policy and classifier both use
the same live `state_dict`, never EMA. It disables automatic DGPO resume and W&B,
and places Ray results/checkpoints/logs in the new directory. `--checkpoint PATH`
tests another existing pretraining epoch without editing production YAMLs.
`--prepare-only` writes the runtime config without starting training.

Both captures enable anomaly detection for the first 10 steps of each classifier
fit. One healthy first-step positive microbatch is saved by rank 0 per worker
process. Any failed forward/backward within this window is also saved, separately
by failing rank. The original exception still aborts training before an optimizer
update; no bad batch is skipped and no backend is changed during training.

The launcher passes `--max-steps 1` to bound DGPO policy updates, **not** OmniFold
bootstrap duration. Normal saturation/closure still apply. A healthy 1% capture
can be obtained even if its later bootstrap reaches the closure safety cap.
Artifacts do not automatically stop a healthy fit; terminate the job manually
after collecting the desired capture if further bootstrap work is unnecessary.

Artifact paths appear as `[DGPO/attention-diagnostic] saved .../failure.pt` under
`OUTPUT_DIR/attention_artifacts/`. These are debugging records, NOT resumable
training checkpoints. They contain event data, classifier and frozen-backbone
weights, batch and per-attention RNG, masks, exact inputs, and incoming output
gradients. Keep them in trusted storage. Saving can consume hundreds of MB and
diagnostic graph retention/anomaly detection has runtime and memory overhead.

## 3. Replay the attention operation on one GPU

Use a captured path from the log, in the same Shifter/PyTorch/CUDA environment:

```bash
shifter python3 scripts/diagnose_omnifold_attention.py replay \
  /absolute/path/from/log/failure.pt \
  --device cuda:0 \
  --output /absolute/path/to/new-report.json
```

This replays classifier attention modules only, **not** full classifier fitting.
Each backend receives the captured module weights, inputs/mask, module mode,
input aliases, incoming gradient, dtype, precision settings and restored RNG.
No optimizer runs. Reports include raw batch, attention input and projected Q/K/V
finite/range summaries, masked-row counts, outputs and gradients. Operators not
reached in backward are explicitly marked as skipped. Unsupported kernels are
reported as errors, never silently replaced by another backend. Exit status 1
means at least one replay errored or produced nonfinite tensors.

Identical RNG states do **not** guarantee identical dropout masks across SDPA
backends. Repeat with `--dropout-zero` as a separately labelled secondary test;
this changes only replay dropout, never training configuration. Use `--backends
math --device cpu` for a portability check, not CUDA verification.

Interpretation:

- Finite inputs/upstream gradient + efficient fails + math passes: evidence of a
  backend-sensitive numerical failure, not proof the pretrained model is bad.
- Nonfinite attention inputs/projected QKV: trace preprocessing/backbone values.
- Nonfinite incoming gradient: the error can originate downstream of attention.
- Both pass: exact failing conditions may not have been reproduced; compare
  environment, masks/dropout, and the captured error before drawing conclusions.
- Both fail with finite inputs: investigate ranges/masks/gradients, not just LR.

The capture hooks do not cover upstream PET attention; its output is captured at
the decoder boundary. Full saved batch/backbone state supports deeper follow-up.
No remote CUDA result is implied by the local CPU regression tests.

## 4. Trace the extreme memory scale using existing artifacts

No new capture or OmniFold training is needed. `inspect` reports each packed
field and feature index, separates valid values from padding, prints saved
normalizer means/stds, and compares saved weights. It runs on CPU:

```bash
shifter python3 scripts/diagnose_omnifold_attention.py inspect \
  /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/attention_artifacts/rank0-step5-9ko2ld7z/failure.pt \
  --compare /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/attention_artifacts/rank0-step1-_a509nqe/failure.pt \
  --output /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/inspect_step1_step5.json
```

`trace` reconstructs the classifier with the capture run's config, then strictly
restores the saved classifier AND complete backbone weights/buffers. It uses the
captured batch and RNG for one forward. It reports normalization, grouped input
embedding, invisible projector, PET transformer blocks/adapters, and decoder
projection/attention ranges in execution order. Raw sequential/global feature
names map the feature indices in `inspect` to the dataset fields.

```bash
shifter python3 scripts/diagnose_omnifold_attention.py trace \
  /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/attention_artifacts/rank0-step5-9ko2ld7z/failure.pt \
  --runtime /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/runtime.yaml \
  --device cuda:0 \
  --output /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/trace_step5.json
```

The original pretraining checkpoint and normalization/config resources must
remain accessible for constructing the model, but the final model weights and
normalizer buffers come from the artifact. No optimizer or backward is run.
Autograd remains enabled during forward to preserve the captured training-time
attention dispatch. This needs a GPU allocation but not a Ray cluster. Original
backend selection is the default; `--backend math` explicitly changes forward
dispatch and should be reported as a separate test. Comparing captured decoder
inputs against reconstructed inputs checks whether the original forward was
reproduced; inspect differences before attributing a particular layer.

Step 1 and step 5 use different batches, so changes in their observed scales alone
cannot establish that optimizer updates caused the problem. Repeat the trace
with the SAME step-5 artifact and add:

```text
--weights-from /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/attention_artifacts/rank0-step1-_a509nqe/failure.pt
--output /pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1/trace_step5_with_step1_weights.json
```

This holds the failed batch and RNG fixed while substituting earlier weights.
It skips agreement checks against the original step-5 forward, since different
weights intentionally change that output. Apply `inspect`/`trace` to the healthy
1% artifact as a separate comparison when available; cross-run differences can
also reflect checkpoint/data differences, not solely optimizer updates.
# Standalone pretrained diffusion inference check

`scripts/diagnose_diffusion_policy.py` bypasses OmniFold fitting and loads the
pretrained policy through the DGPO loader with EMA replacement disabled. It
tests every row of each captured batch, using the runtime DDIM step count,
K=1, eval/no-grad, and the existing attention dispatch. It checks each velocity
prediction before sampler postprocessing and the final physical samples.
No clipping, field replacement, classifier weights, or policy updates are used.
The two invisible slots are shape/mask placeholders for Ztautau generation,
not truth conditioning. Defaults to 256 events per inference microbatch.

Run on one NERSC GPU (no Ray launch needed), with the healthy and failed batches:

```bash
diagnostic_root=/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1
shifter python3 scripts/diagnose_diffusion_policy.py \
  "$diagnostic_root/attention_artifacts/rank0-step1-_a509nqe/failure.pt" \
  "$diagnostic_root/attention_artifacts/rank0-step5-9ko2ld7z/failure.pt" \
  --runtime "$diagnostic_root/runtime.yaml" \
  --device cuda:0 \
  --output "$diagnostic_root/diffusion_policy_step1_step5.json"
```

The report includes the resolved checkpoint path/SHA256, raw feature names,
input ranges, per-microbatch velocity/sample ranges, and numerical status.
It refuses to overwrite a report. An optional `--checkpoint` selects a different
pretrained checkpoint; otherwise the original capture runtime's policy path is
used. This is NOT a physics-quality validation or backward/optimizer test.
Finite results cannot establish that parquet is bad: the inputs are post-loader
captures, and the classifier uses a different forward/backward path.

## Follow-up: stored STIC fields and fixed-batch classifier comparison

First sync `scripts/diagnose_stic_parquet.py` and the updated
`scripts/diagnose_omnifold_attention.py`. The parquet scan needs CPU only;
classifier traces need one GPU. No training or data mutation is performed.

```bash
diagnostic_root=/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1
shifter python3 scripts/diagnose_stic_parquet.py \
  --runtime "$diagnostic_root/runtime.yaml" \
  --policy-report "$diagnostic_root/diffusion_policy_step1_step5.json" \
  --output "$diagnostic_root/stic_parquet_scan.json"

shifter python3 scripts/diagnose_omnifold_attention.py trace \
  "$diagnostic_root/attention_artifacts/rank0-step5-9ko2ld7z/failure.pt" \
  --runtime "$diagnostic_root/runtime.yaml" \
  --output "$diagnostic_root/trace_step5_own_weights.json" --summary-only

shifter python3 scripts/diagnose_omnifold_attention.py trace \
  "$diagnostic_root/attention_artifacts/rank0-step5-9ko2ld7z/failure.pt" \
  --runtime "$diagnostic_root/runtime.yaml" \
  --weights-from "$diagnostic_root/attention_artifacts/rank0-step1-_a509nqe/failure.pt" \
  --output "$diagnostic_root/trace_step5_step1_weights.json" --summary-only
```

The scanner projects `x:slot:feature` and `x_mask:slot` directly from every
training parquet file. It counts valid-particle large (>1e6 magnitude),
nonfinite, subnormal and nonbinary-tag values, excluding padding. Reports
include bounded examples, file-local row/particle positions and available
source identifiers. File/schema errors and partial scans are explicit, not
reported as clean. Mapping comes from the policy's recorded feature ordering;
this tests the preprocessed training parquet, not the upstream detector files.
Unusual-value flags are evidence to investigate, not universal physics cuts.

Both traces use the exact same failed batch and initial RNG state, with either
failed-step weights or initial-step weights. Different parameter-dependent
branches may still affect dropout RNG consumption. The original-state trace
also reports attention input differences from capture: check these before
claiming an exact reproduction. `--summary-only` keeps terminal output compact
while saving the full trace. These traces test forward scales, not backward;
the existing attention replay tests captured local backward separately.

## Exclude the known huge-STIC events in an isolated rerun

`scripts/prepare_stic_filtered_test.py` reads the original capture runtime's
training dataset and writes a new copy, excluding a whole event if any valid
particle has nonfinite or absolute value >1e6 in `Part_sticNumTowers` or
`Part_sticChargedTag`. This deliberately targets the observed huge values;
subnormal-only events and moderate nonbinary tags are not silently repaired.
Original parquet is untouched. Source IDs and reasons are recorded in
`removed_events.jsonl`; counts and file provenance are in `filter_manifest.json`.

All surviving columns, including truth targets, are preserved. Both policy
and classifier populations use the new pool with the same internal split
protocol. The original normalization is preserved (including any existing
contamination); recomputing it would confound this specific exclusion test.
The checkpoint is pinned and SHA256-checked against the diffusion report.
The generated runtime disables resume, starts with live pretrained weights,
fits fresh OmniFold, and uses new output paths. Other fit/backend settings
come from the original captured runtime, not later 1% ablation changes.

```bash
diagnostic_root=/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_test1
filtered_root=/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1
shifter python3 scripts/prepare_stic_filtered_test.py \
  --runtime "$diagnostic_root/runtime.yaml" \
  --policy-report "$diagnostic_root/diffusion_policy_step1_step5.json" \
  --output-dir "$filtered_root"
```

Only after `READY` appears, run the isolated classifier/bootstrap test on the
allocated Ray cluster:

```bash
shifter python3 evenet_dgpo/RL/DGPO_neutrino/dgpo_trainer.py \
  "$filtered_root/runtime.yaml" --no-wandb --max-steps 1 \
  --ray-dir "$filtered_root/ray_results"
```

`--max-steps 1` caps DGPO updates, not initial OmniFold fitting. The new output
directory must not already exist. An interrupted preparation can leave partial
files but no runnable runtime is written until filtering completes. Nothing
is auto-deleted. Removing events changes the sampled population and batch
sequence, so a successful rerun supports (but alone does not prove) causality.
