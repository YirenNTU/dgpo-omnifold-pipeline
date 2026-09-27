# Matched-fold fresh H4 classifier on the raw DGPO endpoint

This trains a **new classifier**, not a fixed-classifier score replay. The frozen
generator is pinned to the same high-LR global-FiLM checkpoint evaluated in
`052997c9`, so this rerun changes the audit population, not the generator:

`/pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110_step0_lr5e5/checkpoints/dgpo-epoch=277-next_ep=278-step=2780.ckpt`

The launcher resolves the supplied path and prints/records its actual epoch/global step.
It loads raw policy weights only, ignoring saved optimizer, EMA, fitted classifier,
reward/reference and monitor state. It does not alter the source checkpoint.

The classifier uses the same current three-block H4/nonlinear visible-kinematic
FiLM architecture. Its backbone starts from the original pre-DGPO classifier
foundation (`diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`), not the current
generator backbone. Fresh trainable classifier banks are initialized with
`reset=True`; no saved classifier fit is resumed. Trainable scope is unchanged:
last PET block, invisible projector and classifier/Fourier/conditioning branches;
the rest of the backbone remains frozen.

One cold fit, one seed, 16 GPUs. Restore the original `227e4975` audit protocol:

- Training: all repeat-1/fold-1 OmniFold training identities (`omnifold_fold`),
  not 60% of the validation pool. The original log recorded **208,355 events**.
- Evaluation: the cleaned external validation pool, **118,992 events**, split
  by the same visible-identity hash into **59,465 early-stop / 59,527 test** events.
- Generate fresh K=1/DDIM20 raw samples from this frozen endpoint for both pools.
  Preserve the original fold, classifier, generation and panel seeds. These are
  the same input events/split rule, not reused samples from the older generator.
- Global classifier batch 16,384, generation batch 2,048 per GPU; the historical
  fold gives 12 optimizer updates per classifier epoch.

These are historical reference counts, not a new subsampling quota. Actual
counts are logged, including if the source parquet or data-sharding behavior
changes. The generator has already used the external data for validation, so
this is not a newly untouched generator test set.

The prior `052997c9` result (AUC 0.849875) used only about 71,400 training
events, versus 208,355 in `227e4975` (last recorded cold audit at policy step300,
AUC 0.909955). That 2.9-fold data difference prevents treating the old comparison
as matched. The W&B config in `227e4975` was overwritten on resume; its original
console logs, rather than its latest config, establish the population used.

Classifier LR2e-4, backbone LR1e-5, constant schedule. No min/max-step override:
minimum0, no maximum, 25 classifier epochs of validation-BCE patience, min_delta
1e-3. Restore the lowest validation-BCE checkpoint before final test. Interpret
near-chance AUC only after inspecting fit curves and convergence; early termination
alone does not prove distribution agreement.

## Launch (user only)

After updating the usual remote repository, inside an existing 16-GPU Ray
allocation:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_h4_checkpoint_audit.py
```

To use the copied CFS checkpoint explicitly, add:

```bash
  --checkpoint /global/cfs/cdirs/m5019/yiren/Ztautau/dgpo_omnifold_ztautau/condition_dgpo/last.ckpt
```

`--dry-run` prints the resolved configuration without loading files or launching.
`--output` changes the audit-only output root. Default:
`/pscratch/sd/y/yiren/Ztautau/h4_latest_checkpoint_matched_fold_audit`.
This separate output preserves the previous smaller-population audit.

New W&B run in `nu2flow-RL` with an automatically generated ID:
**Does H4 improvement survive matched data? | OmniFold fold 1 | raw DGPO step 2780**.
It logs live classifier fit curves, `staleness/raw_auc`, `staleness/raw_auc_gap`,
validation BCE, fit steps/saturation and zero policy/reward updates. Both logging
profiles retain `staleness/raw_audit_fit_events`, `raw_audit_early_stop_events`,
`raw_audit_test_events`, `raw_audit_probe_events`, `raw_audit_steps_per_epoch`,
`raw_audit_uses_omnifold_fold` and `raw_audit_training_fold` (each under
`staleness/`). `classifier_only/source_policy_step` distinguishes step2780 of
the generator from step0 of the measurement-only job.

## Comparison

A smaller final-test AUC gap is evidence of reduced distinguishability for this
fresh classifier family, not full distribution closure. Compare against the
original cold audits of `227e4975`: step0 AUC 0.906178 (1,452 classifier updates),
last recorded step300 AUC 0.909955 (2,040 updates). Verify actual population
counts and fit convergence first. Do not mistake the run's later training endpoint
for an audited checkpoint, or compare against its installed reward classifier.

The default test is one endpoint audit only. An optional independently rerun
bootstrap baseline can use the same launcher and a separate output:

```bash
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/train_h4_checkpoint_audit.py \
  --checkpoint /pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110/checkpoints/dgpo-epoch=-1-next_ep=0-step=0.ckpt \
  --output /pscratch/sd/y/yiren/Ztautau/h4_step0_matched_fresh_audit
```

The script reads each policy's true saved step; it does not take that value from
W&B or guess it from the filename. It does not install the trained judge as a
reward or proceed to a DGPO update. No jobs are submitted by implementation/tests.
