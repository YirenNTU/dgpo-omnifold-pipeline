# Production DGPO: iteration 1 only

Overlay: `dgpo_omnifold_ztautau_10pct_iteration1_only.yaml`.
Control: `dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_fullfit.yaml`.

This is real DGPO training, not the fixed-policy diagnostic.

## Initialization and intervention

- Same source as diagnostic run `c3b347fe4eea`: `f6b4ec46` step 320 live `state_dict`.
- The launch guard checks source step and complete tensor SHA256
  `befb8f9fc6e28787c491cab855411b645dc180b6d848f04b1fafc0594b687513`.
- New policy clock: step 0 / epoch 0; first update is step 1. No imported reward,
  monitor, optimizer, EMA, reference or global-best state. Train initial classifiers fresh.
- New output root: `/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_iteration1_only_step320_seed42`.
  New W&B run (`resume: never`). Launcher refuses a checkpoint directory containing checkpoints.
- Fit exactly iteration 1 with two identity-stable cross-fit classifiers each round.
  Require saturation and held-out oriented AUC > 0.51. No iteration 2 / closure fit.
  Failure does not install an invalid reward. Initial bootstrap fails closed;
  a rejected later refit retains the last valid reward under the existing forward-refit policy.
- **Tempering = 1.0**, per user request, for both propagated weights and installed reward.
  Two folds still have ensemble coefficients 0.5 each; this is averaging, not tempering.
- Refit uses latest policy, no rollback. Each iteration-1 fold inherits its corresponding
  previous-round weights, including output; new classifier optimizer and unit initial weights.
- Architecture unchanged: Fourier + visible-pair rest context, 2 x 128 decoder, dropout 0.15.
  Raw staleness shares architecture and inputs but uses its own weights and fitting cache.
- Full filtered 10% pool for OmniFold, fixed 80/20 split; staleness 250k pool
  (approximately 200k train / 50k validation). Only training batches shuffle/drop tails.
- Existing cosine LR, weight decay, trust region, warm-up and patience schedule remain unchanged.
  This comparison changes BOTH iteration count and tempering relative to the 0.75 control;
  it is not a one-factor causal test of iteration count alone.

## Monitors

| When | Metrics | Interpretation |
|---|---|---|
| Every reward refit | `omnifold/iteration1_monitor/raw_auc`, train/validation BCE, warm-started folds | Classifier strength on that refit's current policy |
| Every reward refit | `omnifold/iteration1_monitor/train_oof/*`, `validation_ensemble/*` | ESS, ESS fraction, maximum mean-one weight, top-1% mass, log-weight std; actual unclipped weights |
| Every 5 DGPO steps | Existing raw staleness AUC, balanced accuracy, training-readiness and best-point records | Independent classifier weights; raw generated vs truth, NOT weighted closure |
| Every 10 steps and refit lifecycle | `gradient_conflict/*` | Reward / raw-monitor / trust gradient alignment and uncertainty |
| Every 5 epochs | Existing diffusion validation and physics-observable scalars | Raw policy quality, not just the reward classifier's score |
| Every epoch / initial installation | Existing DGPO checkpoints including reward and monitor states | Recovery artifacts; this fresh-start launcher intentionally does not resume them |

New monitor curves use DGPO `global_step`. Physics validation uses `epoch`.
Classifier loss curves retain separate per-fit local update axes.
`omnifold/closure_evaluated=0`: no false claim that iteration 1 reached closure.
No new monitor changes gradients, triggers clipping, installs weights or adds a classifier fit.
The weight metrics reuse already-computed OOF and held-out log weights.
They are descriptive, not an independent calibration or closure test.

Success means sustained improvement in adequately trained raw monitors AND physics
observables, with reproducible behavior, not simply low ESS or a declining weak-judge AUC.

## NERSC

In an interactive allocation with the existing 16-GPU Ray cluster running:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/train_dgpo_iteration_one.py --check-only
shifter python3 scripts/train_dgpo_iteration_one.py
```

The second command repeats the read-only source checks and then invokes
`train_neutrino_backend.py` with the base YAML, this overlay, and its new Ray directory.
No shell variables are required. `--check-only` neither starts Ray training nor connects W&B.
NERSC files and GPU execution cannot be verified from the local Mac; run this guard there.

If the residual diagnostic is still running, do not update its checkout in place:
it verifies source-file hashes at completion. Use a separate checkout for this ablation,
or wait for the diagnostic to finish before syncing the changed Python files.
