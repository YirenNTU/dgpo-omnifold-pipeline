# 10% OmniFold architecture ablations

An additional [visible-pair rest-frame arm](DGPO_VISIBLE_REST_ABLATION.md) adds
visible-only Lorentz conditioning and a two-layer decoder. It now starts from
the final step-320 DGPO policy of W&B run `f6b4ec46`, with fresh clocks/classifiers.
It uses its own YAML/output directory; the three configurations below remain unchanged.
Unlike those three arms, the rest-frame arm uses the same Fourier + rest-frame
architecture and inputs for OmniFold and staleness, with separate trained weights.

These are independent full-DGPO experiments, not continuations of the existing
resume run. The latest Fourier arm starts from the live `state_dict` of the
10% diffusion pretrain checkpoint at epoch 214. The control and larger/deeper
YAMLs remain unchanged and start from DGPO step 260. They are therefore not
matched-start architecture-only controls for this new Fourier run. Checkpoint
presence/content on NERSC must be checked there before launching; local tests
do not access remote checkpoints.

| Arm | Reward decoder | Dropout | Fourier path |
| --- | --- | --- | --- |
| `control` | 128 wide, 1 layer, 4 heads | 0.15 | None |
| `fourier_conditioning` | 128 wide, 1 layer, 4 heads | 0.15 | Harmonics 1–4 plus theta-pair context, 64→32 encoder, concatenated with the event context feeding each decoder block's AdaLN |
| `large_deep_dropout` | 256 wide, 2 layers, 4 heads | 0.25 | None |

Fourier conditioning uses candidate-dependent pair features computed identically
for truth and generated candidates. It does not read truth targets as additional
conditioning for generated candidates. Unlike the legacy late-fusion variant,
the Fourier arm has no fusion classifier after the decoder: its two-token
readout remains 256→1. Legacy Fourier late fusion remains available unchanged.

All three arms use the same clean staleness classifier: width 128, one layer,
four heads, dropout 0.15, no Fourier features. `audit_fit` explicitly overrides
the reward builder's architecture. The latest Fourier arm strengthens monitor
cold-start training (below); its training protocol is no longer identical to
older runs, even though the architecture is unchanged. Keep evaluation event
identities, generation noise, seeds and training protocol fixed for comparisons.
Classifier fitting stochasticity and repeated validation selection still mean
that a raw-AUC change alone is not proof of an improvement in generalization.

## Initialization and unchanged settings

### Fourier diffusion-pretrain + warm-start rerun

The Fourier YAML now writes to
`dgpo_omnifold_10pct_arch_fourier_conditioning_frompretrain214_minfold1000_warm12_monitorcold100_seed42`,
preserving all earlier Fourier directories. Its policy source is explicitly
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/epoch=214_train=0.1519_val=0.1278.ckpt`.
Only the live policy `state_dict` is loaded, not EMA or the supervised training
epoch/step. DGPO starts at epoch/step zero with a fresh initial reward stack,
fresh clean raw monitor, optimizer, global best and reference. The fixed
classifier backbone stays on its existing 10% pretrain `last.ckpt`; changing
the policy start does not change the classifier backbone source.
No classifier state from the step-260 experiment is loaded. Subsequent reward
refits still warm-start iterations 1/2, and raw checks still warm-start the
complete compatible monitor (bank plus trainable backbone modules). These
inherited modules remain trainable, not frozen. The first raw monitor retains
the cold training budget; later certified fits use the warm budget below.
Already running jobs do not pick up this YAML change automatically.

`recalibration.fit.min_steps_per_fold: 1000` is an absolute minimum applied
after fold-size scaling, on every fold of every OmniFold iteration/refit. It is
not a fixed 1000-step budget: validation and best-checkpoint tracking continue
throughout; early-stop patience only counts at/after the floor, and fitting may
continue longer. Both cold and warm fits use `validation_patience_epochs: 10`
and `safety_max_epochs: null`. `require_saturation: true` prevents installing an
unfinished fit. `restore_best: true` still selects the best validation loss,
even if that best checkpoint occurred before the minimum-training floor.
Staleness fitting does not receive this 1000-update minimum; it has its own
cold/warm event-epoch policy below.

`recalibration.crossfit_partition: identity` preserves the original Fourier
run's condition-hash folds with the same seed, independently of weight inheritance.
`warm_start_iterations: [1, 2]` now caches the best weights from both folds of
iterations 1 and 2 for the next refit. Only the corresponding iteration/fold is
reused, after matching packing and split-protocol checks; iteration 3+ stays fresh.
Every fold gets a new AdamW optimizer and new policy samples, and each refit
recomputes cumulative log weights from zero. No older architecture's classifier
weights or optimizer moments are imported at startup.
Identity hashing does not yet solve cross-architecture split
comparability: adding packed context columns still changes the identity hash.

### Clean raw monitor training readiness

Only this Fourier YAML opts into `audit_fit.training_readiness`:

- Cold start: at least 100 training-fold epochs. Certified warm start: at least
  5 epochs. The actual fold size and global batch determine the update floor,
  including `drop_last_batch`; about 200k training events / 32768 global batch
  means 6 updates per epoch, so roughly 600 cold or 30 warm minimum updates.
- Both use validation-loss early stopping with 10 epochs patience, counted only
  at/after their floor. There is no fixed total-step cap; the best validation-loss
  weights are restored, even if the best occurred before the floor.
- The readiness path enforces saturation and the completed minimum before
  committing the cache or reporting an eligible baseline. It never requires
  AUC to exceed a target. Budget completion is not proof of statistical power.
- Only weights from a successful fit with matching split, packing width and
  training-policy provenance can warm-start. Legacy short-trained caches or
  changed policies use cold initialization. Optimizer and patience always reset.
- Readiness applies to raw truth-vs-policy monitoring, not classifier trust or
  the OmniFold reward folds. The monitor architecture, data pool, batch, LR,
  dropout and every-5-policy-updates cadence are unchanged.
- `staleness/raw_classifier_warm_started`, `raw_audit_training_ready`,
  `raw_audit_training_min_steps`, `raw_audit_training_steps` and
  `raw_audit_training_epochs` remain searchable in critical W&B logging.

The new run starts with a fresh baseline/global-best record. Do not full-resume
an older run's near-chance baseline under this changed monitor protocol:
the existing global-best audit-signature guard rejects that mismatch. Updating
local files does not alter the already running job.

- Policy `state_dict` only; no inherited EMA, optimizer, reward stack, monitor,
  global best, reference, or training clocks. Epoch/step restart at zero, with
  the current policy as the new reference. Rebuild the initial OmniFold stack
  in every arm, including control; warmup is 10 updates.
- Same filtered 10% event pool, classifier backbone, internal train/validation
  split rule, policy optimizer, cosine schedule and weight decay as before.
- Staleness every 5 updates; patience 8 valid post-warmup misses; global-best
  rollback/refit; no separate candidate confirmation.
- Coefficient-one velocity-MSE penalty and best-decay hard trust boundary.
  The boundary starts from the configured 0.1, without inherited shrinkage.
- Diffusion validation every 10 epochs, at most 15 batches per rank; no TARP or
  image logging. These in-pool validation panels are not an independent test.
- The Fourier and control arms warm-start iterations 1/2 within their own
  experiment after the fresh initial stack. The larger/deeper arm still has
  `warm_start_iterations: []`: every refit there starts from the pretrained
  backbone with fresh classifier-specific parameters. The staleness monitor
  still warm-starts within its own arm. Training budgets and initialization
  differ between arms (the 1000-update floor is Fourier-only), and the latest
  Fourier policy now starts from diffusion pretrain rather than step 260.
  These are not strictly architecture-only comparisons.

The larger/deeper/dropout arm is a bundled capacity-plus-regularization test;
it cannot identify their separate causal contributions. A causal Fourier
comparison now needs a clean control from the same diffusion-pretrain source
with matched training and evaluation protocols; the existing step-260 control
is historical context only. Evaluate both raw-monitor AUC and physics
discrepancies at comparable update/sample budgets. The initialization A/B/C
test supports full monitor warm-start on the tested fixed policy, but does not
by itself prove a Fourier-reward or DGPO-performance improvement.

## Launch

Synchronize the changed Python code as well as these YAMLs to NERSC. In a shell
with the existing 16-GPU Ray cluster, choose **one** arm:

```bash
cd /global/u2/y/yiren/ml_pipeline
ablation_arm=fourier_conditioning
# Alternatives: large_deep_dropout, control
ablation_run="dgpo_omnifold_10pct_arch_${ablation_arm}_from260_seed42"
if [ "$ablation_arm" = fourier_conditioning ]; then
  ablation_run=dgpo_omnifold_10pct_arch_fourier_conditioning_frompretrain214_minfold1000_warm12_monitorcold100_seed42
fi

shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config "config/dgpo_omnifold_ztautau_10pct_arch_${ablation_arm}.yaml" \
  -- \
  --ray-dir "/pscratch/sd/y/yiren/Ztautau/${ablation_run}/ray_results"
```

Each arm writes to its own directory (the latest Fourier rerun uses `frompretrain214`) and starts a
fresh W&B run. The original resume directory is only read. Each job requires
16 GPUs; concurrent jobs need separate adequate allocations/Ray clusters.
These launch YAMLs always restart from their configured source; do not rerun the
same arm in-place expecting automatic resume. Make a separate full-resume
overlay when continuing a completed/paused arm.
