# Fourier + visible-pair rest-frame classifier

Launch overlay: `dgpo_omnifold_ztautau_10pct_arch_fourier_visible_rest.yaml`.
The existing Fourier-only YAML and its outputs are unchanged.

## Physics definition

The frame is **visible_a + visible_b**, using measured/reconstructed visible
`E, px, py, pz` in GeV. It is not the tau-pair frame or either tau's rest frame.
There is no assumed tau energy, tau mass, beam energy, invisible mass, or
Lorentz boost of a direction-only predicted tau. In particular, no truth tau
four-vector or truth label enters the new feature calculation.

The two lab +/-z null reference directions are boosted into this same frame.
Their normalized directions define Collins-Soper-style axes: z bisects beam+
and reversed beam-, x bisects beam+ and beam-, and y=z cross x. Leg a retains
the dataset's ordering; this is not an inferred tau charge or a measured spin
polarimetric vector. Beam references identify lab directions, not beam charge.

The versioned `visible_pair_rest_v3` block has eight channels (no validity flags):

- `log1p(M_vis / GeV)`, beta_z and beta_transverse;
- asinh(mass_a^2 / M_vis^2), asinh(mass_b^2 / M_vis^2);
- leg-a unit-vector components along the z/x/y axes.

Seven redundant v2 channels were removed: log(gamma), E_a*/M_vis,
E_b*/M_vis, |p_a*|/M_vis, cos(a*,beam+*), cos(a*,beam-*), and cos(beam+*,beam-*).
For valid nondegenerate frames these follow from the retained boost, mass-ratio
and direction channels. This is not a guarantee of better classification:
redundant representations can still help a finite neural network learn.

These are event-only conditions shared identically by truth and generated
candidates. They cannot distinguish paired classes by themselves; their purpose
is to help the decoder combine candidate geometry with visible event kinematics.
Do not interpret them as true tau decay/helicity angles. The visible legs are
necessarily back-to-back in their own rest frame, so that trivial opening-angle
feature is deliberately omitted.

Boost arithmetic is float64, with the stable p4 formula and the **into-CM** sign.
No beta^2 division, clipping a superluminal boost, or arbitrary azimuthal axis
is used. A null/spacelike/ill-conditioned finite visible-pair frame produces
neutral features; the event is retained. Non-finite inputs
raise a diagnostic error instead of contaminating optimizer state. At zero
transverse recoil, undefined x/y channels are zero. Validity checks remain
internal numerical safeguards, not learned feature channels. Zero filling is
an explicit convention for undefined quantities, not a measured zero angle.

### What "directly calculated" does and does not mean

For valid inputs, all eight values are deterministic functions of the supplied
visible four-vectors and the fixed lab +/-z directions. Beam-reference energies
are arbitrary normalization units that cancel from their boosted unit directions.
No assumed E_tau=45.6 GeV, on-shell tau-mass constraint, label, truth tau p4,
fitted calibration, or target-dependent constant enters this feature calculation.
The existing 11 Fourier channels likewise use each class's own candidate angles
with the same formula; they do not force an opening angle or back-to-back target.

This is not a claim of zero inductive or statistical bias. Feature selection,
four harmonics, log1p/asinh transforms, GeV scaling, leg ordering and axis choice
are fixed representation choices; encoder LayerNorm and learned weights affect
finite-capacity training. The boosts use float64, cosine roundoff is clamped to
[-1,1], direction norms use epsilon=1e-10, and nearly-null total frames use the
documented numerical threshold. These guards are label-independent and do not
drop events, but zero-filled invalid frames/undefined axes can still be inferred
by a model even without explicit flags. Removing flags does not prove zero bias.
No new correction is made to upstream reconstructed visible four-vectors; their
detector/preprocessing systematics and the classifier's generalization error
are not ruled out by a code audit.

References for conventions and motivation (not evidence this ablation wins):
[Vector Lorentz boosts](https://vector.readthedocs.io/en/latest/src/momentum4d.html#vector._methods.MomentumProtocolLorentz.boostCM_of_p4),
[Particle Transformer](https://arxiv.org/abs/2202.03772).

## Architecture and fitting

Keep the original 11-channel lab-frame Fourier branch (11 -> 64 -> 32).
Add an independent rest-context encoder (LayerNorm, 8 -> 64 -> 32, GELU,
dropout 0.15). Concatenate both embeddings with the existing global event token
to condition the two-layer, width-128 decoder through AdaLN (four heads,
dropout 0.15 in each layer). The two candidate
tokens still use a 256 -> 1 output; there is no event-only output shortcut.

Rest features are computed when the event is packed, once per event batch,
not recomputed for every classifier update/candidate. Raw visible energies are
also retained so a cached pool can be safely repacked. No new Parquet production
or preprocessing pass is required if the existing visible E/px/py/pz columns
are present.

The new branch is included in trainable parameter counts, AdamW's head-rate
group (0.0002), classifier state_dict, PEFT payload, and saved reward architecture.
The full classifier, including this branch, follows this initialization policy:

- Only **iteration 1** inherits the previous refit's iteration-1 same-fold weights.
- **Iterations 2 through max_iterations** each clone THIS round's fitted
  iteration-1 same-fold best-validation checkpoint. Iteration 3 does not inherit
  iteration 2. The source snapshot is unchanged by later fine tuning.
- Older iteration-2+ warm-start cache entries are ignored and omitted from the
  next cache. Fitted residuals still remain in the active reward stack normally.
- Each fit uses fresh AdamW and early-stopping counters. Cold iteration-1
  folds retain minimum 1000 updates;
  both cross-round and within-round warm starts require at least **10 actual
  fold training-data epochs**, then validation-based saturation.
- Iteration 1 retains its full trainable-module selection and LR **2e-4**
  (pretrained position encoder **1e-5**) even when warm-started across refits.
- Iterations 2+ use `later_iteration_train_mode: last_decoder_and_output`:
  only the LAST decoder block and final two-token linear readout train at
  **5e-5**. The output weight/bias reset to zero after inheritance, so the
  initial residual ratio is one and validation BCE starts at log(2).
  The last block retains iteration-1 weights. All earlier decoder blocks,
  decoder input projections/output norm, position encoder, Fourier/rest encoders,
  GroupedSequentialEmbedding, invisible projector and PET adapters are frozen.
  Their dropout stays off even after the fitter calls `train()` following a
  validation. Only the unfrozen last block retains training-mode dropout.
  Frozen inherited body tensors remain in snapshots and PEFT exports.
  This is not a full cold restart: only the residual output head is reset.

These are initial weights, never automatically accepted residual increments.
Every new stack starts cumulative log weights from zero; within the stack each
classifier fits the CURRENT cumulative sample weights. Fold identities and outer
train/validation protocols must match; incompatible cross-round caches are not
reused. Other overlays retain their legacy initialization policy by default.

This change applies at the next process launch, not to an already running job.
The current launch remains weights-only from the step-320 policy checkpoint, so
its initial iteration 1 starts fresh; initial iterations 2+ use the newly fitted
iteration 1. Do not interpret a weights-only restart as resuming the current
classifier fit.

OmniFold and raw staleness use the **same architecture and input schema**:
the same model builder, fixed pretrained backbone, 11 Fourier channels, eight
rest-context channels and decoder. Iteration 1 and staleness have the same
trainable-module selection (including
GroupedSequentialEmbedding, invisible projector and internal PET adapters).
Iteration 2+ intentionally trains a restricted subset without changing architecture.
Architecture is defined only in `recalibration`; `audit_fit` has no architecture
overrides. Their trainable weights, optimizer states and warm-start caches remain
separate: the raw monitor never takes fitted reward-classifier weights. The raw
monitor compares unweighted generated samples with truth; OmniFold fits the
residual weighted problem. Equal architecture does not make these objectives equal.

Training budgets remain separate too: OmniFold uses the full filtered 10% pool,
while staleness uses a 250k-event pool (approximately 200k train/50k validation).
Batch sizes and the raw monitor's optimizer settings are unchanged. The monitor retains its
complete within-run warm-start, cold250/warm5 event-epoch floors, and
every-5-policy-updates cadence. The identity projection excludes ONLY the
new rest/energy columns, preserving the Fourier-only control's existing 80/20
and two-fold assignments for the same events. Legacy pool hashing is unchanged.
This also keeps round-to-round validation identities out of inherited training.

The raw monitor's first cold fit now requires **250 training-data epochs**
(1500 updates for approximately 200k training events and global batch 32768).
After a successful fit, later checks inherit that monitor's complete weights
and require only **5 epochs** (approximately 30 updates), then validation-based
early stopping. AdamW and patience counters reset; the reward stack is never
used as raw-monitor initialization. A saved cold100 cache is not certified for
this new training policy, so it will not bypass the longer initial fit.

Both reward and staleness classifiers now validate every **2 training-data
epochs**, retaining **10 epochs of early-stopping patience** (five validation
checks). Reward-fold epochs use the actual fold event count and global per-class
batch, with the dropped training tail excluded; they are not DGPO epochs or
per-GPU microbatch counts. Best-validation checkpoint restoration, full final
scoring and closure criteria are unchanged. Diffusion validation remains every
five DGPO epochs. Validation-cadence changes do not alter dropout, batch size or
the controller schedule; the later-iteration LR change is documented above.

### Fixed inputs, split provenance and batch shuffling

`cache_event_inputs: true` copies each rank's OmniFold and raw-monitor source
event batches once into CPU memory. Later pool materializations replay those
inputs in the same order, fixing the capped 250k monitor population as well as
its train/validation membership within the run. Candidates are **not cached**:
each check/refit regenerates them with the current live policy. The raw monitor
uses the same rollout seed/noise sequence on its fixed input order.

The cache is process-local, consumes host memory and initially reads the full
source shard (not just the 250k prefix). It is not written into policy checkpoints.
On a process restart Ray can repartition/reorder the source pool, so bitwise
panel replay across separate jobs is not promised. The identity-hash split
provenance still keeps a given event in its original train/validation role.

Outer train/validation and staleness splits use seed **42**. OmniFold's two-fold
split uses fixed seed **20260906**, separate from changing refit initialization
seeds. Identical visible conditions stay together; generated values never enter
the split hash. Incompatible warm-start split provenance is not reused. Keeping
the existing identity splitter, instead of shuffling and slicing anew each refit,
also protects inherited classifiers if Ray changes row order.

Both classifier fitters use `independent_epoch_shuffle` and
`drop_last_batch: true`: only the training fold is reshuffled each epoch, and its
short final batch is discarded. Validation retains **all** its rows, including
the final short batch. Dropped training events vary with the epoch shuffle;
they are not permanently removed from the dataset. Diffusion validation now runs
every five completed epochs (both tiers have cadence 5, so only one full pass
runs, not a duplicate cheap pass). Its existing batch limit remains unchanged.

### Loss curves

`logger.wandb.classifier_loss_curves: true` adds fixed custom charts under
`Classifier training/Reward/*` and `Classifier training/Fresh audit/*`.
`training_loss` is minibatch BCE plotted against the selected fit's local
optimizer step, not DGPO global step, epoch, or the W&B transport row. Iterations
are overlaid as separate lines; only the first repeat/fold is displayed for each
iteration. The chart title identifies the DGPO step. Progress remains every 10
classifier updates, plus final/stopping records; no extra classifier evaluations
are added. Validation metrics are appended only when validation was actually
evaluated. Raw per-member `omnifold_live/*` history remains queryable but hidden.
Set `logger.wandb.classifier_loss_curves_raw: true` only to restore the legacy
per-fit `classifier_fit/*` panels for a debugging run. Existing DGPO/physics
curves retain their original clocks.

## Gradient-conflict monitor (diagnostic only)

`dgpo.gradient_conflict.enabled: true` runs every **10 DGPO updates**, after the
coincident raw staleness fit, before any rollback, refit or reference replacement.
Staleness itself stays at every **5** updates. The gradient monitor compares
three gradients in the **same trainable diffusion-policy parameter space**:

- `omnifold`: the existing DGPO surrogate with the installed reward stack;
- `staleness`: that same surrogate with detached truth-positive raw-classifier
  logits substituted as reward (never installed as the training reward);
- `trust`: the configured coefficient times the existing half velocity-MSE
  penalty against the active round reference.

The actual `build_dgpo_loss` detached gate, leave-one-out advantage, optional
advantage clip, K=8, eight timestep draws and t range [0,0.7] are retained.
No AUC gradient is claimed. This does not compare classifier parameter gradients,
nor does it measure AdamW-preconditioned displacement or weight-decay direction.

Each check uses **8 disjoint blocks x 512 global events = 4096 events**, not
4096 per GPU. On 16 GPUs this is 32 events per GPU per block. Events come only
from the fitted raw monitor's identity-hash validation split; duplicate identities
are removed from the probe. This is a previously selected validation set, not an
untouched test set. A policy-only mask sidecar preserves the original invisible
slot masks without adding inputs to either classifier or changing identity hashes.
The panel/seed stays fixed within a job; its SHA256 is logged to reveal changes
after a Ray repartition/restart. Classifier architecture/complete cached weights
are restored into a separate frozen judge. No classifier is trained by this probe.

All three objectives share the same candidates and `(t, eps)` within each block;
different blocks use independent draws at one fixed policy snapshot. Generation
and rewards are no-grad. Gradients are obtained with `autograd.grad`, never
`backward` or `optimizer.step`; existing `.grad`, RNG state, policy/reference
buffers and module modes are preserved. Global reductions weight by event count
(trust by active mask mass), including unequal rank/microbatch sizes. No gradient
averages span policy updates or reward rounds. No new controller action is added.

W&B records `gradient_conflict/*` against **DGPO global_step**, not classifier
steps or W&B row numbers. Main visible curves are O/S cosine, conclusive/conflict,
O/trust cosine, and O/S split-half self-cosines. Additional searchable metrics
include all three norms, cross-block dot estimates and lower/upper intervals,
alignment, readiness/skip flags, panel hash, reward round, sample counts and time.

- `cosine` is the cosine of the block-averaged gradients, not the mean of cosines.
- `cross_dot` excludes same-block diagonal terms, avoiding shared-noise covariance
  as evidence of directional agreement. Its interval is an approximate 95%
  delete-one-block jackknife Student-t interval (7 degrees of freedom with 8 blocks).
- `conclusive=1` requires both gradients' positive lower cross-self-dot bounds,
  split-half self-cosine >=0.1, non-negligible norms and a cross-dot interval that
  excludes zero. `conflict=1` additionally requires the upper bound <0.
- `conflict=0, conclusive=0` means **inconclusive**, not agreement. Zero/weak/noisy
  gradients cannot produce a conflict verdict. Negative O/trust cosine can simply
  be the intended regularizer pullback, not an implementation error.

The confidence and reliability thresholds are diagnostic heuristics, not calibrated
sequential tests. Eight blocks can be insufficient, especially near zero signal;
increase `blocks` manually to 16 or 32 if needed. There is deliberately no adaptive
optional-stopping escalation or automatic gradient surgery. Classifier error,
reused validation, fixed panel and repeated monitoring limit statistical claims.

This adds real compute: K-candidate generation and three backward-gradient reads
per timestep/block, but no optimizer updates. GPU work is event-microbatched.
Only rank 0 stores all exact vectors in host RAM (approximately 96 bytes per
trainable policy parameter with 8 blocks); a chunked float64 Gram matrix avoids a
second full-vector stack. Other ranks retain only one block. No random projection
is used. Disabled by default in other YAMLs. VP-path-KL/EMA rollout configurations
are rejected rather than silently monitoring a different objective/policy.

## DGPO raw-staleness patience schedule

This schedule controls only the **raw monitor's refit/rollback trigger**.
Reward-classifier and raw-monitor fitting use their separate minimum budgets
documented above and 10 event-epoch early-stopping patience, so shorter early
DGPO patience does not truncate classifier training. The schedule is a
configurable heuristic, not a convergence
guarantee or a statistically calibrated threshold.

| Persisted DGPO global_step | Required consecutive eligible non-improving checks | Steps at 5-step cadence |
| --- | --- | --- |
| 0-99 | 6 | 30 |
| 100-299 | 10 | 50 |
| 300-599 | 16 | 80 |
| 600+ | 24 | 120 |

The row applies at the current monitor step (step 100 immediately uses 10).
Existing consecutive misses carry across a stage boundary. A new best resets
the streak; invalid/unready monitors break it as before. Warmup intervals pause
and reset the streak under the existing warmup rule. Patience counts qualified
failures to improve the configured best AUC gap, not only strictly rising AUC.

The clock is the training global_step, not classifier updates, W&B's internal
step, accepted-only updates, or the checkpoint step of the best policy. It
continues across in-run policy rollback and reward installation. Full checkpoint
resume uses the restored training clock/streak; this weights-only fresh-start
YAML begins at step 0. New reward installation resets the streak, not the
schedule. Global-best rollback, refit, optimizer reset and trust settings remain
unchanged. Other stop conditions (including two failed global-best refit rounds)
are not disabled by the schedule, so later stages are not guaranteed to be reached.

The effective patience is recorded in probe history, terminal plateau logs and
W&B `staleness/raw_no_improvement_patience` (kept in the critical profile).
With no `trigger.patience_schedule`, all other configs keep fixed patience.

## Start and outputs

- Live policy `state_dict` from the final step-320 checkpoint of
  [W&B run f6b4ec46](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/f6b4ec46),
  not EMA, the earlier step 260 or a mutable `last.ckpt` alias. Its W&B output log
  records `dgpo-epoch=31-next_ep=32-step=320.ckpt` in the v26_resume3 checkpoint
  directory. This is the last saved checkpoint of that run, not a best-AUC selection.
- Fresh DGPO clocks, initial reward stack, raw monitor and global-best record.
  Baseline is step 0; the first policy update is step **1**, not 321. AdamW,
  cosine schedule, warmup, reference and trust schedule restart as a new experiment.
- Both classifiers load the same fixed 10% backbone from
  `diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`. This classifier
  initialization is distinct from the policy's pinned DGPO step-320 checkpoint.
- Same filtered 10% pool, LR cosine decay, weight decay, warmup and trust region.
- New `...arch_fourier_visible_rest8_samearch_l2_psched_from_f6b4ec46_step320_seed42`
  directory and new W&B run. No existing experiment is overwritten by this
  directory change. This YAML is a fresh launch, not an auto-resume overlay.
- The v3/8-channel encoder cannot inherit v1/18-channel or v2/15-channel rest
  classifiers or the previous one-layer monitor/reward stack. Startup remains
  fresh; later same-v3, two-layer refits retain full warm-starts.

Synchronize the Python changes and YAML to NERSC. With an existing 16-GPU Ray
cluster, launch:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_arch_fourier_visible_rest.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_arch_fourier_visible_rest8_samearch_l2_psched_from_f6b4ec46_step320_seed42/ray_results
```

Do not regenerate NERSC's `config/generated_event_info.yaml`: use the schema
that produced the data and pretrained model. Local tests use an isolated schema
fixture and cannot verify NERSC checkpoint/data contents or GPU execution.
The W&B log proves the step-320 file was saved then, not that the NERSC file
still exists or has not since been overwritten in the reused source directory.

## What would demonstrate an improvement?

This requested run changes the policy starting checkpoint, rest-frame features, the patience schedule and the
monitor architecture, including decoder depth from one to two layers. It is
therefore a combined ablation. To isolate depth, compare against one layer with
the same rest features, initialization and fitting protocol. The existing
Fourier-only control still uses its clean monitor. Historical raw AUC values
from that monitor are not directly comparable to this run's new baseline.
To isolate feature effects in an end-to-end comparison, match the patience and
monitor protocols in the control; for a fixed-policy classifier comparison the
DGPO patience schedule is inactive.

On the **same fixed policy**, compare held-out classifier AUC and BCE against
Fourier-only with matched event IDs, noise, seeds and training budgets. Inspect
train/validation gaps and ratio-weight tails/ESS, not AUC alone. Across trained
policies, use an equally trained common monitor and physics observables.
Repeated validation selection is not an untouched-test guarantee.

This implements a physically defined candidate for a stronger classifier; it
does not establish that it is the strongest architecture or improves DGPO
without running these comparisons.
