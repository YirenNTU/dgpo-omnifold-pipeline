# Fixed-policy raw-monitor replay

This is a separate, read-only diagnostic, not another DGPO training run. It
compares the saved clean raw monitor from `f6b4ec46` step 260 with the clean
monitor from `f7401d77` **after its initial monitor fit but before any DGPO update**.
Neither classifier is refitted. There are no optimizer steps, EMA swaps, reward
installation, rollback, production checkpoint writes, or W&B runs.

## Run on NERSC

Sync the new script and diagnostic YAML along with the current repository code.
Use an available 16-GPU Ray allocation; do not compete with a training job using
all 16 GPUs. The configured new monitor checkpoint must already exist and contain
its completed raw-monitor state. Do not replace it with `last.ckpt` after policy
training has advanced.

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/diagnose_raw_monitor_replay.py \
  config/dgpo_10pct_raw_monitor_replay.yaml
```

Default output:

```text
/pscratch/sd/y/yiren/Ztautau/raw_monitor_replay_step260_old_vs_cold100_v1/
```

Existing output directories are refused. For a new attempt, pass
`--output-dir /pscratch/sd/y/yiren/Ztautau/raw_monitor_replay_step260_old_vs_cold100_v2`.
`--prepare-only` performs CPU checkpoint checks and writes a manifest without
launching Ray; use a separate output directory for that preflight because the
full run also requires a new directory.

## What is held fixed

- The policy is the verified `/global/.../f6b4ec46_step260/` backup, with full
  `state_dict` SHA256 `dc66b775e474070c12aa9f3bc5f67436921fa1dd36d78202ac4f9ed75d376378`.
- Both monitor checkpoints must contain exactly that policy state. Missing raw
  monitor caches, a changed source file, or a later policy checkpoint fail closed.
- The real loaded model is compared tensor by tensor against the source, allowing
  only the synthetic `famo.w.*` entries absent from the inference model.
- Production processed-Parquet loading and K=1 / 20-step DDIM generate one common
  250k-event pool. Both saved classifiers see the same truth and raw samples.
- Each monitor receives its original packing layout. The old layout excludes
  Fourier context; the new monitor's split was hashed with that extra context,
  even though its classifier architecture does not use Fourier features.
- Only the **intersection** of the two saved validation folds is scored. With
  different approximately 20% folds this will often be about 10k of 250k events,
  not 50k. This avoids using rows in either monitor's training fold. At least
  2,000 common validation events are required; the split is never relaxed.
- Classifier restoration checks exact keys, shapes, dtypes, finite values and
  post-load tensor equality, including saved adapters and input projectors.
- Frozen-backbone reconstruction checks its legacy fingerprint and the saved
  bootstrap source-bundle digest against the current backbone file SHA256. An
  unsupported legacy source identity or a mismatch stops interpretation.
- All metric weights are one. Reports retain signed raw AUC, AUC gap, balanced
  accuracy and BCE; neither direction nor thresholds are optimized on this test.

## Artifacts and interpretation

`report.json` contains both results, sample counts and the measured difference.
`pool.pt` preserves the shared generated pool; `scores.pt` stores validation indices
and paired truth/raw logits for both monitors. `manifest.json` and the two merged
runtime YAMLs preserve checkpoint/split/architecture provenance. Per-monitor JSON
files are written as each evaluation completes.

- Old AUC high and new AUC low on the same panel: prioritize the new monitor's
  initialization/training history, not a policy improvement interpretation.
- Both low: check loading, processing and generation against historical inputs;
  this alone does not establish that the policy matches truth.
- Both high: historical/current panel differences or the earlier evaluation path
  need investigation; the two monitors can distinguish this shared sample.

This is **not an exact numerical reproduction of historical W&B AUC**: the Ray
row order and DDIM noise are newly generated, and the common validation subset
differs. Both validation folds were previously used for model selection, so this
is a diagnostic comparison, not an untouched generalization test. Legacy raw
monitor caches do not contain a complete runtime manifest; the supplied old/new
overlays and reward packing provenance are recorded, and strict tensor loading
does not by itself prove historical code equivalence. This first replay does not
train a fresh classifier or test the Fourier reward classifier.
