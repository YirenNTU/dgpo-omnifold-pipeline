# Three conditioned blocks in diffusion and classifier

Prepared, not launched. Both requested feature variants now have a separate
three-block configuration. The policy source is now the original raw step1110
checkpoint. Older one-block configs remain unchanged but have a different
policy source, so they are not matched depth controls for this revision.

| Variant | Overlay | W&B ID |
| --- | --- | --- |
| Observed theta/phi | `dgpo_h4_visible_adaln_depth3.yaml` | `h4vad3s1110` |
| Observed theta/phi + energy/pT | `dgpo_h4_kinematic_adaln_depth3.yaml` | `h4kfd3s1110` |

## Depth and nonlinear conditioning

- TruthGeneration: **3 transformer blocks**, width256, four attention heads.
  Scale/shift modulation before attention and FFN in every block: **6 sites**.
- Fresh OmniFold and cold-audit classifier: **3 decoder blocks**, width128,
  four heads. Self-attention, cross-attention and FFN modulation in every block:
  **9 sites**. Existing AdaLN-Zero event gates are retained.
- Each model retains its own shared conditioning encoder: two particle-level
  Linear+SiLU layers width64, masked learned pooling, two event-level
  Linear+SiLU layers width64, then encoder-only LayerNorm. Three independent
  zero-initialized output heads supply the per-block scale/shift values.
- PET depth, candidate H4 late fusion, feature selection, Fourier frequencies,
  raw input normalization and policy optimizer/LR settings are unchanged.
  Classifier trainable scope is specified below.

These are transformer blocks with nonlinear FFNs, not three extra linear
projections. Diffusion and classifiers do not share learned encoder weights.

## Classifier trainable scope (OmniFold and cold audit)

The user permits last-block fine-tuning while freezing the earlier pretrained
backbone. Both feature variants set the same explicit flags in `recalibration`
and `audit_fit`:

| Component | Train? | Learning rate |
| --- | --- | --- |
| Final PET transformer block, including its own norms | Yes | 1e-5 |
| Earlier PET blocks, input/time embeddings | No | — |
| GlobalEmbedding, GroupedSequentialEmbedding | No | — |
| PET internal adapters | Yes | 5e-5 |
| Added PET angular Fourier projection | Yes | 2e-4 |
| New visible nonlinear encoder and per-layer FiLM heads | Yes | 2e-4 |
| InvisibleInputProjector | Yes | 2e-4 |
| Three-block candidate decoder | Yes | 5e-5 |
| H4 Fourier encoder, fusion/output heads | Yes | 2e-4 |

The existing slot-position encoder retains its conservative1e-5 rate. No blanket
LayerNorm unfreeze: norms in the final PET block are trainable as part of that
block, while norms in the frozen body stay frozen. Fourier frequencies themselves
are fixed; training Fourier means training the projections/encoders that use them.
The new `train_angular_conditioning: true` opens only PET's added angular branch;
`train_backbone: false` and `train_last_pet_block: true` remain separate controls.
The new angular group logs `optimizer_group_lr_angular_conditioning`; nonlinear
FiLM retains `optimizer_group_lr_visible_conditioning` and its cheap RMS/gradients.

Each fold owns an isolated body. The Fourier branch's trained weights and flag
are included in the classifier payload; old payloads without the flag retain
their old frozen-Fourier scope when restored. The new fast angular LR group is
opt-in, so legacy full-backbone fits retain their previous LR assignment.
These classifier settings do **not** freeze or otherwise change the DGPO policy.

## Preserving the pretrained policy

Load this **raw**, weights-only policy source:
`/pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_old_method_hard_nc4shnpg_t075_trust1_nohardtrust_seed42/checkpoints/dgpo-epoch=110-next_ep=111-step=1110.ckpt`.
This is the original c4a91e07 lineage, not the later angular diffusion finetune
or its epoch0 classifier-bootstrap snapshot. No EMA, old classifier payload,
reference state or optimizer state is reused. New policy epoch/step start at0;
references start from the newly attached policy, not the source run's old anchor.
Two iterations of fresh two-fold OmniFold fitting precede policy updates.

The user confirmed the classifier backbone must retain its separate **pre-DGPO**
raw 10% diffusion pretrain source, explicitly pinned in `reward_config.omnifold`:
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`.
Both OmniFold and cold audit nevertheless use the new three-block decoder and
fresh nonlinear visible conditioning; no old classifier head is restored.
This is not the policy step1110 checkpoint or the earlier EveNet foundation-only
checkpoint. Both feature variants inherit the same classifier source.

`network.TruthGeneration.identity_init_from_layer: 1` initializes the appended
zero-based blocks1 and2 with zero attention output and final FFN projections.
Their internal projections are normally initialized. Their LayerScale starts
at1, not1e-5, since the zero outputs already ensure identity and an additional
tiny scale would unnecessarily damp the first gradients. The loaded block0
and all other source weights remain untouched.
Step1110 has no PET angular branch, so its added angular output projection also
starts at zero. The new nonlinear modulation heads start at zero as before.

EveNet accumulates `cond_token` through these blocks: the new blocks initially
return the same `cond_token`. Thus increasing depth does not change the initial
velocity function. Initialization occurs only during construction, before
checkpoint loading; a future matching three-block resume restores its learned
projections, not zeros. The frozen reference receives the whole current state.

On the first backward pass, new residual output projections receive gradients;
their inner layers and modulation pathways begin receiving gradients after
those outputs move. Do not mistake this short zero-init effect for a dead branch.
Fresh classifiers retain their existing AdaLN-Zero initialization; they do not
load a one-block classifier into three blocks.

## Policy-only conditioning learning rates

Both depth-3 variants inherit the following explicit DGPO override:

```yaml
options:
  Training:
    epochs: 1500
    total_epochs: 1500
dgpo:
  unbounded_training: false
  conditioning_learning_rates:
    angular_conditioning: 1.0e-4  # PET.angular_conditioning
    visible_conditioning: 1.0e-4  # TruthGeneration.visible_conditioning
  lr_schedule:
    type: cosine
    total_epochs: 1500
    total_steps: null
    min_lr_ratio: 0.1
    groups: [angular_conditioning, visible_conditioning]
```

This changes **only the diffusion's Fourier and nonlinear FiLM branches**.
The angular projection no longer shares the pretrained PET group's LR, and
the whole visible-conditioning encoder/pooling/modulation branch no longer
shares the generation head's 1e-6 LR. Their parameters are excluded from the
parent groups, so AdamW updates each parameter exactly once. All other policy
parameters retain their existing LR, including the appended generation blocks.
Weight decay and warmup are inherited from each parent. **Only these two new
branches decay**, from 1e-4 to 1e-5. The other policy groups retain their existing
constant rates. The horizon is 1500 **logical policy epochs**, not classifier
epochs or complete dataset passes: with `steps_per_epoch: 10`, it resolves to
15,000 policy updates. Explicit `total_steps: null` removes the inherited
1500-step horizon. The epoch and step horizons cannot both be specified.
Training is capped at 1500 epochs; refitting/clearing Adam moments does not
restart the scheduler. No additional policy early-stop criterion is introduced.
OmniFold and audit classifier LR/configuration are unchanged (their new
conditioning groups already use the head LR, 2e-4).

Console output lists each group's modules, parameter count and LR. W&B logs
`train/lr/scheduled/angular_conditioning` and
`train/lr/scheduled/visible_conditioning`, plus the parent-group rates, against
policy `global_step`. These are the rates **before** round warmup/trust scaling.

The existing commands below use these rates on the next launch. This does not
change an already-running process or submit any job. Keep the configured raw
step1110 **weights-only** start: an older full optimizer checkpoint has a
different group layout and is not directly resumable with the new groups.
Checkpoints created with these groups can resume with the same configuration.
The higher starting LR and cosine schedule replace the preceding 1e-5 constant
proposal. This is an unexecuted LR/schedule change, not an observed improvement.

Previous constant-branch-LR validation: 102 tests passed across `test_dgpo_trainer.py`,
`test_visible_conditioning.py`, `test_kinematic_conditioning.py` and
`test_conditioning_depth.py` (plus 20 subtests). Coverage includes exact
parameter ownership, actual AdamW update magnitudes, unchanged default/parent
groups, cosine scheduling, matching-checkpoint continuation, invalid overrides,
compact W&B logging, and resolved settings for both sixteen-worker variants.
These are local CPU tests, not a new NERSC/GPU training run.

Current 1e-4/1500-epoch cosine validation: **108 tests and 33 subtests passed**
across the same four modules. Added checks cover epoch-to-update conversion
(1500 x 10 = 15000), rejection of ambiguous horizons, branch-only decay with
unchanged parent LRs, checkpoint/refit clock preservation, legacy all-group
cosine compatibility, and resolved epoch limits for both feature variants.

## Experimental interpretation

The user requested **both** diffusion and classifier depth increases. This
tests the combined deeper-conditioning package, not diffusion depth alone.
It is also not a capacity-matched test of FiLM. More blocks cost compute/memory
and can overfit; they are not guaranteed to help.

Keep coefficient1 velocity-MSE anchoring, AdamW, two folds x one repeat x two
iterations, OmniFold refit every20 policy epochs, cold audit every5 and validation
every10. Classifier best-validation-BCE selection, patience25 epochs fromstep0,
no forced minimum or maximum fit steps, and sixteen workers are inherited.

### Raw audit data (matched training fold)

Both depth-3 variants now inherit `audit_fit.training_population: omnifold_fold`
and `training_fold: 1`. This is a **cold** judge, not a reused reward classifier:

- Training: exactly the repeat-1/fold-1 **training identities** from the full
  OmniFold train parquet (the complement of that fold's out-of-fold identities).
  Reuse the reward's condition hash and fixed crossfit seed, not its weights.
- Each audit generates new K=1 samples from the **current raw policy**. Only
  event inputs are cached; candidates are never inherited from an earlier fit.
  Select the fold before DDIM generation to avoid generating the unused half.
- Evaluation: the external validation parquet is split by a fixed condition
  hash into 50% early-stop and 50% final-test. Both are disjoint from this
  audit's training rows; final-test is used only after restoring the best BCE
  checkpoint, never for audit fitting or early stopping.
- With the previously measured files, this is approximately **208k train,
  59.5k early-stop and 59.5k final-test**, instead of 71.4k/23.8k/23.8k.
  Actual counts are logged; caps are maxima, not instructions to duplicate data.
- Batch16384 remains global per class. Audit epoch length uses its actual
  training population: about12 updates/epoch and25 patience checks, or about
  300 updates without a qualifying BCE improvement. This is not a minimum or
  a maximum fit budget. OmniFold keeps its existing rounded crossfit interval
  (about13 updates/check); its refit logic and fitting setup are unchanged.
- Audit every5 policy epochs remains measurement-only; refit stays every20.
  An optional step-zero audit also uses this same protocol. No audit weights
  are installed as reward and no AUC threshold terminates these audit fits.

The final-test is held out **within the audit**; this external parquet is also
used for reward-classifier validation, so it is not an untouched project-level
test set. Old small-pool audit results are not a matched ruler for this new
larger-data protocol. The audit will cost more training/generation compute.
Legacy configs retain `probe_split` and their previous behavior.

Console output names the source fold and all population counts. W&B retains
existing raw-audit AUC/BCE/fit-step metrics and additionally logs
`staleness/raw_audit_uses_omnifold_fold`, `raw_audit_early_stop_events`,
`raw_audit_steps_per_epoch`, `raw_audit_validation_interval_steps` and
`raw_audit_patience_evaluations` under the same `staleness/` prefix.
`raw_audit_training_fold` records the selected one-based fold. Trajectory
summaries exclude older small-pool points after a protocol change; an old
small-pool step-zero audit cannot satisfy the new step-zero audit check.

Inspect fixed-reward improvement **within** a reward round, velocity MSE, ESS,
fresh audit BCE/AUC, train-validation gap and cheap conditioning RMS/gradients.
Audit capacity changes relative to a one-block run: direct AUC comparisons are
not capacity-matched. Absolute rewards from independently fitted classifiers
are not a common ruler. Do not call an underfit near-chance audit closure.

## User launch commands

With the user's existing sixteen-GPU Ray cluster, run the desired arm:

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_h4_visible_adaln_depth3.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_visible_adaln_depth3_1110/ray_results
```

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_h4_kinematic_adaln_depth3.yaml \
  -- --ray-dir /pscratch/sd/y/yiren/Ztautau/h4_kinematic_adaln_depth3_1110/ray_results
```

No jobs, allocations, uploads or W&B runs are started by this implementation.
Local tests do not establish sixteen-GPU throughput or memory fit; the remote
source checkpoint must exist with its expected one-block generation head.

## Local validation (2026-09-25)

279 related CPU tests passed, 2 skipped, 29 subtests passed. The22 focused
depth/source tests include production raw checkpoint-loader checks with
conflicting EMA and old DGPO state. These tests
cover both feature variants; one-to-three migration with/without learned
conditioning and LayerScale; exact initial velocity/input-gradient equivalence;
frozen-reference numerical agreement; gradient flow through all three blocks
after updates; optimizer ownership; learned three-block checkpoint reload;
three-block classifier gradients and two-fold/two-iteration frozen reward
serialization; and resolved architecture/training controls. W&B display names
passed the naming validator. No training-performance improvement is claimed.
Six additional trainable-scope tests use real PET blocks and actual classifier
AdamW fitting: frozen weights unchanged, permitted weights updated, no duplicate
optimizer ownership, exact LR groups, independent fold/audit builders, new
four-member reward payload reload, and legacy frozen-Fourier payload behavior.

### Matched-fold audit validation (2026-09-25)

The20 focused audit/pool tests pass, including an actual CPU classifier fit,
exact reward-fold membership across16 simulated unequal input shards, current
policy regeneration with fixed input caching, an empty local selected fold
still participating in gather, early-stop/final-test isolation, actual nested
trainer dispatch at intermediate audit boundaries, and legacy-config behavior.
This is not a sixteen-GPU execution or throughput test.

Broader regression: **331 passed, 79 subtests passed, 1 failed**. The failure is
`TestBestDecayTrustRadius.test_best_point_restart_opts_inherited_reference_in_at_age_zero`
in the untouched best-decay restart path (`best_decay requires a fresh reference
install`); it also fails when run alone. This round's fixed schedule has that
controller disabled. No unrelated trust-controller code was changed to mask it.
