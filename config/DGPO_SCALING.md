# DGPO: matched 1% and 10% data budgets

The first section documents the retained v18 raw-plateau/VP-KL controls.
For the latest v26 velocity-MSE/global-best-decay/round-warmup runs, see the sections below.
Both initialize directly from their own diffusion pretraining checkpoints:
1% uses epoch 172, while 10% uses `last.ckpt`. Neither loads a DGPO-trained policy.
These overlays must be merged
with `config/train_diffusion_nersc.yaml` through the launcher below; they are
not standalone trainer configs.

| Budget | Training pool | Pretrained diffusion and classifier backbone |
| --- | --- | --- |
| 1% | `diffusion_train_1pct_seed42/train` | `diffusion_pretrain_1pct_seed42/checkpoints/last.ckpt` |
| 10% | `diffusion_train_10pct_seed42/train` | `diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt` |

All paths above are under `/pscratch/sd/y/yiren/Ztautau/`. The 10% checkpoint
path follows `train_diffusion_10pct_nersc.yaml`; its existence must be verified
on NERSC before launch.

## Data and checkpoint protocol

- No additional fractional subsampling or event-count cap: each run's policy,
  OmniFold, raw-staleness classifier, and reference-trust classifier use only
  its own materialized training pool. No external validation directory is read.
- Classifiers share a fixed condition-hash split (seed 42), approximately
  80% fit / 20% validation. Duplicate conditions stay together. The two
  OmniFold reward folds are inside the fit 80%; closure uses the outer 20%.
  Warm-start provenance protects this split across refits and pool reordering.
- Classifier global batches are capped to the smallest actual fit population
  when necessary, aligned to the worker count. Otherwise the configured
  global batches remain 32,768 for reward fits and 8,192 for monitors.
  Normal shuffled drop-last classifier training still applies.
- Both policy and classifier backbone explicitly load `state_dict`, **not
  `ema_state_dict`** (`Training.EMA.replace_model_after_load: false`). Startup
  logs the chosen weight source. DGPO generation uses the live policy too.
- On first launch: weights-only pretrained start, fresh optimizer, epoch/step
  zero, fresh OmniFold bootstrap, then a pre-DGPO raw-classifier baseline.
  The installed reward is checkpointed before that baseline fit.
- Subsequent launches resume only the same scaling run's `last.ckpt`, including
  its installed OmniFold stack, optimizer, reference, controller, and progress.
  An existing output directory therefore means **resume**, not a new experiment.
- Output directories and W&B names differ between budgets and from older runs.
- `logger.wandb.profile: critical` keeps only the decision/physics plots.
  W&B `resume: never` starts a new tracking run, while DGPO still resumes its
  own checkpoint. The new run records the checkpoint's epoch/step in Summary
  `resume/*`; it does not reset the training clock.

Plot axes are explicit: DGPO/staleness/trust use completed `global_step`,
physics uses zero-based `epoch`, and the pretraining baseline is epoch -1 at
step 0. The W&B internal `_step` counts logging events, not DGPO updates.
Classifier progress is retained as hidden diagnostic history with separate
fold/iteration/local-step metadata, not connected across folds into one curve.
Each event is committed separately to avoid baseline or monitor coordinates
being overwritten. Mid-epoch resume does not rerun a false step-zero baseline.

The retained controls are DGPO AdamW weight decay 0.001; independent classifier
weight decay 0.0005; warm starts for OmniFold iterations 1 and 2 and the raw
monitor; raw staleness every 5 policy steps; rollback/refit after 5 consecutive
eligible checks without a new best. The trust classifier runs each epoch and
can independently trigger refitting. VP-KL coefficient remains 1; the radius
starts at 0.1, shrinks by 0.9 after successful reference updates, and stops
shrinking at 0.02. Acceptance/topology audits remain off.

## Launch on an allocated 16-GPU Ray cluster

From the repository root, choose one command. Parallel runs need sufficient
separately allocated resources; each config requests 16 GPUs.

1%:

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_1pct_scaling_raw_plateau_vpkl.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_1pct_rawplateau_vpkl_v18_scaling_seed42/ray_results
```

10%:

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_scaling_raw_plateau_vpkl.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_rawplateau_vpkl_v18_scaling_seed42/ray_results
```

The base event-info, resonance and normalization files are still required.
Pretrained artifacts and distributed GPU execution cannot be validated from
the local CPU-only tests.

## Interpreting this comparison

### 1% every-epoch refit / no hard trust ablation

`dgpo_omnifold_ztautau_1pct_epoch_refit_nohardtrust_ablation.yaml` starts from
the same live epoch-172 pretrained checkpoint used by control `o6z0z5ch`,
not EMA or the control's latest DGPO policy. Outputs and W&B runs use a separate
`v19_epoch_refit_nohardtrust` directory/name. Automatic resume is disabled:
each launch resets DGPO epoch/step and optimizer state and fits the initial
OmniFold again. The existing output directory is deliberately reused, so
same-name checkpoints can be overwritten; old unmatched files are not removed.
W&B still starts a new run (`resume: never`).

VP-path-KL remains enabled with coefficient 1; hard adaptive boundaries,
backtracking and classifier-trust monitoring are disabled. Gradient clipping
and DGPO weight decay remain unchanged. At each epoch end (10 policy updates),
the raw monitor runs and OmniFold refitting is attempted on newly generated
samples from the **current** policy, even when raw AUC improves. No best-policy
rollback occurs. Iterations 1 and 2 warm-start the previous round's corresponding
fold classifiers; later iterations initialize fresh. The existing residual
closure threshold of 0.55 and maximum of 10 iterations govern installation.
A candidate that fails closure is not installed. Both classifiers use the full
1% pool with the existing internal train/validation split. Each epoch saves a
checkpoint after the adaptive cycle, including successfully installed rewards.

The OmniFold reward fit uses a global batch of 4,096 (256 per rank on 16 GPUs),
approximately one quarter of the observed small-pool-capped batch of 16,576, and an
early-stopping patience of 25 classifier epochs rather than 10. This is not a
25-epoch training cap. Epoch-based validation intervals are rounded to whole
updates per fold. Monitor batches/patience, learning rates, and DGPO batches
are unchanged. The residual closure threshold was subsequently relaxed from
0.51 to 0.55 (`residual_min_auc_gain: 0.05`); this permits more residual
distinguishability and is not evidence of better distribution matching.
Restart with this overlay to apply
these settings; an already-running process does not reload YAML changes.

The current 1% ablation further raises only the OmniFold reward classifier's
`head_dropout` and `topology_dropout` from 0.25 to 0.35. Explicit overrides in
`adaptive_omnifold.audit_fit` retain 0.25 for both raw-monitor dropouts; this
requires the updated `adaptive.py` and `evenet_ratio.py`, not just the YAML.
Policy dropout remains zero. Other configs without monitor dropout overrides
keep their existing shared-builder defaults. This tests stronger classifier
regularization; faster closure alone does not demonstrate better raw matching.

This requires the updated trainer as well as the YAML: the raw-plateau branch
now honors `max_reward_age_epochs: 1`. Existing configs with a null age limit
keep their original trigger behavior.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_1pct_epoch_refit_nohardtrust_ablation.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_1pct_v19_epoch_refit_nohardtrust_seed42/ray_results
```

### 1% clean 128-wide classifier / velocity-MSE / hard-boundary experiment

`dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml`
now runs the v26 step-monitor/round-best-refit experiment (the YAML filename is retained).
It uses reward dropout 0.15, monitor dropout 0.15, policy dropout zero,
closure AUC 0.52, 256 reward events/GPU and classifier early-stopping patience 25.
Raw staleness checks run every five optimizer steps. Five consecutive valid,
saturated checks without a new round-best raw AUC trigger OmniFold refitting
after restoring the current reward round's best raw-AUC policy checkpoint.
Samples are regenerated from that restored policy before reward fitting.
Optimizer state is cleared while the global training clock continues forward.
Local hard-trust step backtracking remains enabled.
An improvement resets this outer patience; invalid/unsaturated checks break
the consecutive streak. Epoch-age forced refits are disabled. OmniFold reuses
the last installed iteration-1/2 fold weights (later iterations start fresh);
the raw monitor reuses its own last classifier weights. Optimizers are fresh.
Every successfully installed reward round (including bootstrap) also clears
the DGPO AdamW state: first/second moments, AMSGrad state if present, and
per-parameter step counters. Policy weights, optimizer groups, weight decay,
learning rate, scheduler and global training clock are preserved. Rejected
refits and ordinary same-round resume do not invoke this install reset.
This uses `recalibration.reset_optimizer_state_on_install: true`, not the
legacy first-moment-only reset. W&B logs `reference_trust/adam_full_state_reset`
and `reference_trust/adam_states_cleared` at the install boundary.
After each accepted installation (including bootstrap), DGPO also uses a
20-accepted-update LR warmup: update `i=1..20` multiplies the current scheduled
group LRs by `0.1 + 0.9*(i-1)/19`, then uses 1.0 thereafter. This multiplier
combines with trust scaling/backtracking; classifier LRs and weight-decay
coefficients are unchanged. The original scheduler and global clocks are not
reset. Rejected/nonfinite policy updates do not consume warmup, rejected refits
do not restart it, and ordinary resume restores the saved warmup progress.
Changing the warmup protocol mid-round requires a new weights-only run/refit,
not silently resetting a resumed run. Critical W&B charts include
`train/round_warmup/lr_scale`, `train/round_warmup/completed_updates`, and
`reference_trust/policy_lr/effective_step_scale` (including backtracking).
Classifier trust remains disabled. It loads only the live policy weights from
`diffusion_pretrain_1pct_seed42/checkpoints/epoch=172_train=0.2060_val=0.1993.ckpt`.
Do not load the old OmniFold stack, optimizer, EMA or controller. Start a new
epoch/step-zero clock with fresh one-layer, 128-wide, no-topology reward and staleness
classifiers. The initial OmniFold stack MUST be fitted again for this architecture.
The classifier backbone still uses the original epoch-172 pretrained checkpoint.
Verify the source diffusion checkpoint exists on NERSC; it cannot be verified locally.
Automatic resume is disabled; outputs use a new v26 directory.

The coefficient-one penalty is the existing legacy `velocity_mse` objective:
`0.5 * masked_mean((v_policy - v_round_reference)^2)`, not VP-path KL. The hard
boundary uses `velocity_mse_ratio`: velocity MSE divided by reference denoising
loss on a fixed probe. With `radius_mode: best_decay`, the initial radius is 0.1.
Each saturated GLOBAL-best raw-AUC improvement advances the target schedule
`max(0.02, 0.1 * 0.9**new_best_count)`. First baseline registration, ties, misses,
unsaturated/invalid checks and repeated same-step calls do not advance it.
The global minimum `abs(raw_AUC - 0.5)` is separate from the round-best used
for patience/rollback and never resets on OmniFold refits. Local recovery that
does not beat the historical global record does not shrink the radius.
The effective radius cannot expand or exclude the live policy: it is
`min(previous_radius, max(target, fixed_probe_distance / warning_fraction))`,
with `warning_fraction: 0.9`. If this prevents the full contraction, the target
is retained and becomes feasible after the next reference recenter.
Same-round resume preserves that effective radius; successful reference installs
apply the target without advancing the count. No extra per-round multiplier is used.
The count survives refits and policy-only rollback and is saved in checkpoints.
W&B logs `best_decay/count`, `best_decay/target`, `best_decay/global_best_auc_gap`,
and `best_decay/feasibility_limited`
under `reference_trust/`, alongside the effective `reference_trust/delta`.
This ratio is not VP-KL and is not a proven physical-distribution bound.
Post-step backtracking halves an excessive update, up to eight backtracks,
using 128 probe events per rank (16 ranks). Empirical radius calibration,
trajectory search, and signed-direction probes remain disabled. The reference
recenters after each successful reward install. The critical W&B profile now
retains velocity MSE and its ratio along with boundary/accepted-step metrics.

The original 5% baseline `dgpo_omnifold_ztautau.yaml` uses the same default
velocity-MSE loss, coefficient 1, unscaled leave-one-out advantages, K=8,
beta=1, beta_kl=0, 20 DDIM steps, eight training timesteps in [0, 0.7],
and policy gradient clip 1. The v26 config matches these core settings and
disables `sequential_vp_trust_backward` (that option requires VP-path KL).
This is not a full reproduction of the historical 5% protocol: v26 preserves
the four-head candidate decoder at hidden dimension 128, now with one layer,
with both `periodic_pair_features` and `topology_fourier_embedding` disabled.
Two 128-wide candidate tokens feed a 256-to-1 output directly: no topology MLP,
no fusion branch, and no raw pair-feature concatenation. Classifier dropout 0.15,
closure 0.52, warm starts and five-step raw monitoring remain
with patience-based round-best rollback/refitting, without acceptance audits. Data, initialization, batch sizes,
weight decay and validation protocol also differ from that historical run.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_1pct_epoch_refit_velocity_mse_trust_ablation.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_1pct_v26_rawplateau5_patience5_returnbest_notopology_d128_l1_velocity_mse_trust10_globalbestdecay09_warm20from10_dropout15_seed42/ray_results
```

This jointly changes regularization and the trust objective/boundary relative
to the running dropout-0.25 no-hard-trust run; a difference cannot be attributed
to hard trust alone. Sync the trainer, adaptive/model-builder code and YAML.

### 10% v26 clean-classifier / velocity-MSE / global-best-decay / warmup experiment

`dgpo_omnifold_ztautau_10pct_velocity_mse_trust_ablation.yaml` applies the same
latest v26 training settings as the 1% velocity-MSE overlay, except closure is 0.51
(1% remains 0.52), and the 10% reward batch is 16384 / 1024 per GPU,
four times the 1% reward batch of 4096 / 256 per GPU. The 10% staleness batch
is also increased fourfold to 32768 / 2048 per GPU; scoring chunks remain
2048 per rank. Its staleness pool is capped at 250,000 event identities total,
targeting approximately 200k fit / 50k validation. The stable condition-hash
split determines actual counts; they are not guaranteed to be exactly 200k/50k.
Shared settings include reward dropout 0.15 (monitor 0.15), reward
fit patience 25 epochs, five-step raw checks and five-check round-best rollback/refit,
warm starts, DGPO weight decay 0.001, full AdamW reset on accepted rounds,
and the feasibility-protected best-decay target `max(0.02, 0.1 * 0.9**new_best_count)`.
Policy and two-fold OmniFold use
all surviving events in the STIC-filtered 10% pool; the capped raw monitor
draws only from this same source. Both platform paths point to
`/pscratch/sd/y/yiren/Ztautau/omnifold_attention_10pct_stic_filtered_test1/train`,
the output of the earlier `prepare_stic_filtered_test.py` command, rather
than the unfiltered `diffusion_train_10pct_seed42/train`. Original normalization
is retained. This intentionally changes the event set relative to the raw 10% run.
Both retain the stable internal 80/20 partition protocol.
It loads live policy `state_dict` directly from
`/pscratch/sd/y/yiren/Ztautau/diffusion_pretrain_10pct_seed42/checkpoints/last.ckpt`,
not EMA or the unavailable v21 bootstrap checkpoint. DGPO starts at epoch 0,
step 0 with fresh optimizer/controller state. The classifier backbone keeps
its separate initialization from the same 10% pretrained `last.ckpt`.
It never loads 1% weights or data. The initial classifier stack is retrained
without topology at width 128, just as for 1%. Cold start and
automatic-resume-disabled behavior match the 1% v26 overlay. Both policies
start from diffusion pretraining, not existing DGPO checkpoints. Outputs are new
and the old 10% v18 YAML is left unchanged. Expensive autograd anomaly tracing
is disabled explicitly (`anomaly_detection_steps: 0`), matching the 1% default.
Normal nonfinite guards, staleness monitoring and trust checks remain enabled.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_velocity_mse_trust_ablation.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_v26_rawplateau5_patience5_returnbest_notopology_d128_l1_velocity_mse_trust10_globalbestdecay09_warm20from10_dropout15_seed42/ray_results
```

Sync YAML plus the updated trainer, adaptive controller and classifier builder.
These settings do not establish that the earlier 10% STIC/attention numerical
failure is resolved. Checkpoint existence and distributed GPU behavior must
still be checked on NERSC. Confirm the filtered directory's preparation finished
(`filter_manifest.json` has `complete: true`) before launch; local tests cannot
verify that remote artifact. The diagnostic runtime's pinned checkpoint is not
copied; verify that the configured diffusion pretrain `last.ckpt` exists on NERSC.

### Full v26 resume with a new W&B run

Use `dgpo_omnifold_ztautau_1pct_resume_v26.yaml` or
`dgpo_omnifold_ztautau_10pct_resume_v26.yaml` to resume explicitly. The 1% config
loads original v26 `checkpoints/last.ckpt`; the 10% config now loads
**v26_resume2 (W&B run 8317316b)** `checkpoints/last.ckpt`, not the original
step-35 source or the exported step-120 evaluation best. These are full `checkpoint_load_mode: resume`
configs, not weights-only starts. The policy, optimizer/scheduler, reference,
OmniFold stack, monitor cache, epoch/global-step clocks, global-best trust
schedule and partially completed round warmup are restored. Initial bootstrap
and forced resume refits are disabled. Normal scheduled plateau refits still run.
An unfinished raw-baseline check may need completion; this does not refit the
already installed reward stack.

W&B uses `fresh_run: true` and `resume: never`: a new explicit ID is generated
even if the shell has `WANDB_RUN_ID` set. W&B transport rows start fresh, but
charts retain the checkpoint's real epoch/global-step coordinates. Outputs go
to separate `v26_resume1` (1%) and `v26_resume3` (10%) directories.
The 10% continuation fixes the previously unregistered `global_best_candidate`
and `global_best_incumbent` progress phases. The in-flight comparison was not
committed, so it is rerun from the last complete checkpoint; saved OmniFold and
cosine/global-best state are restored, without an initial-stack refit.
The 10% continuation uses patience=6 and diffusion validation every 3 epochs
(both cheap and full tiers set to 3, so only one full pass runs when due).
Staleness remains every 5 steps, independently of diffusion validation; six valid
misses correspond to 30 additional updates when every check is eligible.
Classifier fitting/early-stopping settings, group base LRs,
round warmup and trust settings are unchanged; it does not reset the inherited trust
radius to 0.1. It now enables **cosine LR decay** for DGPO only:
`dgpo.lr_schedule: {type: cosine, total_steps: 1500, min_lr_ratio: 0.1}`.
On the first resume from a constant-LR checkpoint, the curve anchors at the saved
group LRs and scheduler step, so there is no immediate LR jump. It decays smoothly
to 10% of those LRs at absolute scheduler step 1500 and stays at that floor.
The endpoint matches 150 logical epochs x 10 steps when no boundary steps are
rejected; trust-rejected steps do not advance the scheduler, so the floor may
not be reached before the epoch budget. W&B/global steps are not substituted for
this saved scheduler clock. The anchor and cosine protocol are checkpointed;
subsequent resumes preserve the same curve, not a new decay cycle. Protocol
mismatches or missing optimizer/scheduler state fail closed. Refits reset Adam
moments and the separate 20-update round warmup, not the cosine clock. The warmup
and trust backtracking multiply the scheduled LR; classifier LRs are unchanged.
Critical W&B plots `train/lr/scheduled_max` and `train/lr/scheduled_min` expose the
pre-update scheduled group LRs before those extra multipliers.

The 10% resume overlay now also selects **global-best rollback** (the 1% configs
remain unchanged). The ordinary monitor still warm-starts and runs every five
DGPO steps. A candidate must improve the recorded global raw AUC gap by more
than `raw_improvement_min_delta: 0.001`. Before replacing an existing best, it
is compared with the incumbent on freshly materialized **paired** pools: one
event pass, common DDIM noise, matching train/validation splits and identical
classifier initialization seeds/training settings. These two confirmation
classifiers start fresh and do not modify the routine monitor's warm-start
cache or the installed reward. Both must saturate and the candidate must beat
the incumbent by more than 0.001. This is an engineering effect-size/confirmation
gate, not a confidence bound or a guarantee against adaptive-validation bias.
It runs only for candidate bests, not at every routine check; it requires two
additional classifier fits per candidate and no third data split.

Six valid checks without a confirmed global improvement trigger rollback to
the retained best policy. Invalid/unsaturated checks reset the streak. Refits
never clear the global-best record. After rollback, samples are regenerated
from that policy and OmniFold is refit with the existing iteration-1/2 warm-start
protocol. Only a successful reward/reference installation resets AdamW moments
and starts the 20-update warmup; cosine progress, global step/epoch, and the trust
decay history are never rewound. A rejected global refit is fail-closed rather
than training with mismatched old reward/reference state. The normal boundary
checkpoint immediately records the successful installation.

Legacy resume checkpoints migrate the best pointer from their saturated raw
history across **all** recorded rounds. Startup checks the exact epoch, step,
next-epoch and raw score in the selected file; it searches the source checkpoint
directory and the explicit `global_best_checkpoint_search_dirs`, never a
runner-up or a different same-epoch snapshot. Missing/truncated best history or
incompatible raw-audit protocols fail closed. The imported historical best is
not claimed to be re-confirmed at migration; subsequent replacements require
the paired confirmation above.

`global_max_failed_rounds: 2` counts complete plateau windows **after refitting
the same best**, not the first plateau that initiated that refit. A confirmed
new best resets the counter. After two failed refit rounds the trainer saves
the latest consistent policy/reward/reference and terminal status, preserves
the separate best checkpoint, and stops without another refit. Mid-epoch stops
save the correct within-epoch position. This is stagnation, not convergence.
Resuming that terminal checkpoint requires an explicit increase of the failed-
round limit after reviewing results; it does not silently restart the loop.
W&B exposes `staleness/global_best/{improved,failed_rounds,stop_requested}` and
confirmation delta/validity/acceptance alongside `staleness/raw_best_auc_gap`.
Keep original v26, v26_resume1, and v26_resume2 checkpoint directories:
the restored round-best rollback pointer may still reference a checkpoint there.
Missing/incompatible source checkpoints are errors, not a pretrain fallback.
Both source files must be checked on NERSC; local tests do not verify them.

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_1pct_resume_v26.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_1pct_v26_resume1_rawplateau5_patience5_returnbest_notopology_d128_l1_velocity_mse_trust10_globalbestdecay09_warm20from10_dropout15_seed42/ray_results
```

```bash
shifter python3 scripts/train_neutrino_backend.py \
  --backend dgpo-evenet \
  --base-config config/train_diffusion_nersc.yaml \
  --overlay-config config/dgpo_omnifold_ztautau_10pct_resume_v26.yaml \
  -- \
  --ray-dir /pscratch/sd/y/yiren/Ztautau/dgpo_omnifold_10pct_v26_resume3_rawplateau5_patience6_val3_returnbest_notopology_d128_l1_velocity_mse_trust10_globalbestdecay09_warm20from10_dropout15_seed42/ray_results
```

### September 8 continuation: step 205, stronger confirmation, patience 8

The current 10% interactive overlay pins
`dgpo-epoch=20-next_ep=20-step=205.ckpt` from run `85f392a1` in the
existing `v26_resume3` checkpoint directory. It uses full resume with
`auto_resume_from_last: false`; the first new update is 206. The saved
epoch/within-epoch position, optimizer/cosine schedule, reward stack, raw
monitor and global-best state are retained. W&B starts a new run. The same
output directory is reused as requested; historical names still contain
`patience6` although the active YAML sets 8. Load 205 rather than the terminal
295 checkpoint, whose two-failed-round stop flag is intentionally preserved.

Confirmation fits take independent copies of the monitor weights captured
before the current routine monitor fit. Each gets a fresh optimizer and
early-stopping state, common event/noise samples and identity-stable train/val
splits, and at least both 512 updates and 25 classifier epochs before early
stopping is eligible. Only confirmation uses these larger minimum budgets;
the routine raw monitor and its protocol fingerprint are unchanged. These
checks reduce unequal fitting as a confounder; they do not prove statistical
significance or guarantee improved policy AUC.

`pause_patience_during_warmup: true` keeps monitoring and accepting confirmed
improvements during warmup but resets the miss streak on any interval that
contains warmup updates. A serialized interval marker prevents counting the
last warmup interval after a mid-round resume. Eight valid misses at five-step
cadence allow 40 further updates after the 20-update warmup, instead of the
previous 30-update total allowance. The limit remains two failed refit rounds.
DGPO LR, velocity-MSE coefficient and trust boundary are unchanged.

For the current continuation, optional TARP coverage/null trials and physics
figure rendering are disabled. Scalar physics validation still runs every
three epochs, raw staleness every five updates, and confirmation only on
eligible potential bests. This reduces diagnostic work without changing
policy updates, classifier fit budgets or checkpoint validation.

### Evaluation limitations

Physics validation panels reuse the training pool: they are diagnostics, not
independent generalization measurements. The classifier validation partition
is held out from classifier gradient fitting, but is reused for early stopping,
monitoring and policy selection, and the diffusion model can train on these
identities. It is not an untouched test set.

The existing diffusion pretraining configs also use an external validation
subset; these DGPO overlays do not retroactively change that pretraining
protocol. A clean final scaling comparison needs a common independent test
protocol and matched post-training budgets. Do not compare a cold pretrained
start directly with the old v18 best-DGPO-point restart as equal-budget trials.
