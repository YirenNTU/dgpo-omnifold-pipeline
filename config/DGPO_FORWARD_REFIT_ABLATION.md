# Forward-only refit ablation

Overlay: `dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit.yaml`.
Control: `dgpo_omnifold_ztautau_10pct_arch_fourier_visible_rest.yaml` (4267caeb).

Same source step-320 live policy, **weights-only**, new optimizer and step/epoch 0.
Initial OmniFold and raw monitor are trained fresh. This does not resume step 215.
Both classifiers, data splits, dropout, tempering=1, velocity-MSE coefficient=1,
hard trust boundary, global-best radius decay, cosine LR and warm-up are unchanged.
Output root is separate from the control:
`/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_gradlife_step320_seed42`.

## Controller

- Raw check every 5 policy updates; patience 4/8/16/24 at steps 0/100/300/600.
- Warm-up checks do not consume patience, as in the control.
- A plateau refits OmniFold on the **latest policy**, then installs that policy
  as the new round reference. No policy rollback to the recorded best.
- Global best remains recorded/saved and drives the existing trust-radius decay.
- `global_max_failed_rounds: 0` disables global-stagnation early stopping; failed
  rounds remain logged. The configured training budget (150 epochs) still applies.
- AdamW moments reset on accepted reward/reference installs; LR schedule clock,
  weight decay and policy step/epoch never rewind.
- No change to residual classifier training or closure semantics in this control.
  The separately configured staged-training variant is described below.

## Gradient monitoring

Periodic probes remain every 10 policy steps. Lifecycle probes are additional,
diagnostic only, and never trigger a refit or alter the loss:

1. `gradient_conflict/pre_refit/*`: before refitting the current policy.
2. `gradient_conflict/post_install/*`: after accepted reward/reference installation,
   before the next optimizer update. Trust norm should be zero (up to numerics).
3. `gradient_conflict/post_warmup/*`: first raw check after the 10-update warm-up.

Pre/post use the same event panel, noise seed and current raw judge. Post-warm-up
uses a checkpointed copy of the install-time judge, NOT the newly fine-tuned raw
monitor; routine periodic probes continue using the latest raw monitor. Compare
panel hashes to verify identical event identities. Exact panel ordering after a
new Ray launch is not guaranteed. `raw_auc` in lifecycle probes is the judge's
source-fit AUC, NOT a new current-policy AUC; `judge_fit_global_step` records its age.
Lifecycle markers and pending judge state are checkpointed for full resume.
Pre/post series are separate so they cannot overwrite each other at the same step.

`total_on_omnifold/projection_ratio` measures the component of g_reward+g_trust
along g_reward, divided by the original reward component (1=no change, 0=canceled,
negative=reversed). The corresponding `total_on_staleness/*` uses the raw judge
surrogate direction. Cross-block dot-product intervals and `opposed` are logged.
These are **raw probe loss gradients**, not AdamW/weight-decay-adjusted updates;
they do not certify AUC improvement or classifier calibration.

## Launch on an allocated NERSC interactive Ray cluster

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_gradlife_step320_seed42/ray_results
```

Sync the modified Python files and the new YAML before launch. Repeating this
command still starts from step 0 and reuses this ablation's output root; do not
run simultaneous writers there. The old control directory is not modified.

## Staged residual-training variant

Overlay: `dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_staged.yaml`.
The forward-only overlay above remains unchanged as the control. The new overlay
changes only iteration 2+ fitting and its output/run names; it does not change
iteration 1, staleness, architecture, inputs, splits, DGPO loss or controller.

Each iteration 2+ starts from the current round's iteration-1 same-fold weights:

1. Stage A zeros the residual output and trains only the 256-to-1 linear layer
   (257 parameters), LR `5e-5`. All inherited feature extractors stay frozen/eval.
2. Stage B starts from restored-best A, without resetting the output. Only the
   last decoder block is additionally unfrozen, LR `1e-5`; output stays at `5e-5`.
   Earlier decoder blocks, conditioning encoders and backbone remain frozen/eval.
3. Both stages have fresh AdamW moments, the same train/validation examples and
   weights, and retain existing classifier weight decay, gradient clipping,
   shuffled/drop-last training, validation cadence and early stopping. Each
   inherited stage gets at least 10 fold-data epochs and validation patience of
   10 data epochs. This is additional training, not a speed optimization.
4. Stage B is selected only when its restored-best validation BCE is lower than
   stage A by more than `validation_min_delta=0.0001`. Otherwise the entire A
   state is restored. Existing residual acceptance and closure checks then run.

Only iteration 1 is cached across refits; later iterations still inherit the
current iteration 1, not the previous residual. Staleness retains its independent
same-architecture warm-start cache. The stage selection is a validation-based
regularization measure, not proof of unbiased ratios or physical closure.

W&B loss curves have separate `_stage1` / `_stage2` local-step axes. Live metrics
include `fit_stage`, `selected_stage`, `stage_a_validation_loss` and
`stage_b_validation_loss`. Saved diagnostics count both stages' training steps.

This variant still loads **live policy weights only from source step 320**, then
starts at step/epoch 0 with fresh initial OmniFold/staleness and a new W&B run.
It is not a full training-state resume. Use an allocated interactive Ray cluster:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_staged.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_staged_gradlife_step320_seed42/ray_results
```

Sync the modified Python files and YAML before launch. This is a third, separate
output root; neither control is overwritten. Repeated launches reuse this new
root, so do not run simultaneous writers there. Local unit tests and the runtime
configuration/launcher check do not verify remote checkpoint availability or
replace a NERSC GPU execution test.

## Current variant: all iterations use iteration-1 training

Overlay: `dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_fullfit.yaml`.
The staged overlay is preserved as the d3e6a8ed control, with a separate output
directory. The fullfit variant sets `later_iteration_train_mode: full` and removes
both later-iteration learning-rate overrides. It uses the already-supported full
fit path; it does not add layers or change any input features.

The current fullfit overlay additionally sets residual `tempering: 0.75`.
Each accepted iteration adds `0.75 * residual_log_ratio` to the train/validation
log weights; the frozen installed reward uses the same factor on its sum of
increments (not a second factor on already-tempered weights). Later classifiers
fit the resulting tempered distribution. This is a more conservative correction,
not merely a learning-rate change, and can require more residual iterations.
The closure schedule, safety cap, DGPO trust coefficient and learning rates stay
unchanged. Previous control overlays retain tempering 1.0.

- Every iteration trains the same modules as iteration 1: both decoder layers,
  output, conditioning encoders, enabled position parameters, grouped sequential
  embedding, invisible projector and PET adapters. The pretrained main backbone
  remains frozen exactly as for iteration 1; `full` does not mean full-backbone
  fine-tuning.
- All iterations use the same optimizer grouping: ordinary classifier/adapters
  LR `2e-4`, conservative backbone group LR `1e-5`, weight decay `0.0005`, clip 5.
- Iterations 2+ inherit the COMPLETE current iteration-1 same-fold state,
  INCLUDING the output head; no zero-head reset and no stage A/B. Each classifier
  fit creates fresh AdamW state. Only iteration 1 is cached across refits.
- Training budget remains initialization-dependent: cold iteration 1 at least
  1000 updates per fold; inherited fits at least 10 actual fold-data epochs.
  Validation patience, best-BCE restoration, shuffling and drop-last are unchanged.
- Closure now follows DGPO global step: 0--99 uses AUC <=0.55, 100--299 <=0.53,
  300--599 <=0.52, 600 onward <=0.51. The safety cap stays 10. DGPO loss/trust,
  staleness and patience 4/8/16/24 are unchanged. This does not guarantee closure
  or less overfit; the early target is approximate closure, not exact matching.

`recalibration.residual_closure_schedule` takes strictly increasing `start_step`
entries (starting at zero) with nonincreasing `max_auc`. It overrides the legacy
`residual_min_auc_gain` scalar when present. It is resolved ONCE at refit entry
and used for BOTH residual acceptance/stopping and final stack installation.
It never uses classifier training updates, OmniFold iteration count, epoch or
refit count. A boundary crossing applies at the NEXT normal refit, not as a new
forced refit. A first classifier below the threshold still fails closed rather
than installing an empty stack. W&B records `omnifold/residual_closure_auc_limit`
and `omnifold/refit_global_step` on accepted and rejected refits.

Full resume uses the restored DGPO global step; this overlay's weights-only
restart resets that clock to zero and therefore begins at 0.55 regardless of
the imported policy's source step 320. Configurations without this schedule
retain their old static threshold. No extra mutable schedule state is required.

The trainer now distinguishes an actual policy rollback from global-best
bookkeeping on refit rejection. Without rollback it keeps the installed pair and
optimizer and logs `forward_recenter_deferred`; after actual rollback it still
stops safely. No failed candidate is installed. Existing retry cadence is not
changed: while plateau persists, later raw checks can retry (potentially costly).
Critical W&B logging now retains refit failure reasons and stage-selection metrics
for the staged control as well.

Like the control, this is source step-320 live-policy WEIGHTS ONLY, fresh clocks
at step/epoch 0, fresh initial classifiers and a new W&B run, not full resume:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_visible_rest_forward_refit_fullfit.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_visible_rest_l2_forward_refit_p4_8_16_24_fullfit_gradlife_step320_seed42/ray_results
```

Sync Python changes AND the new YAML before launch. Previous control directories
are untouched; do not launch concurrent writers into this new output directory.
