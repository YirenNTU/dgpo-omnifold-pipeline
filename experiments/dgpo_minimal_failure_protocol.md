# Current DGPO plateau: minimal diagnostic protocol

Status: completed as `6704e92e`; see the executed result below. The first user-launched NERSC attempt
stopped before policy updates during config validation: it inherited
`refit_once_fail_closed=true` after disabling `refit_once_on_resume`. The probe
overlay now disables both, and the launcher runs the native adaptive and
gradient-trace parsers before checkpoint I/O or Ray launch. The corrected rerun finished.
Production configuration is unchanged; measurement hooks are inactive unless
explicitly enabled.

## Executed result — 2026-09-29 review

[6704e92e](https://wandb.ai/ytchou97-university-of-washington/nu2flow-RL/runs/6704e92e)
finished the native step1130 -> 1150 continuation, with reward round7 fixed.
The 32768-event paired held-out mean reward deltas at +1/+5/+20 were
+0.06022225 / +0.06615362 / +0.00993407. The predeclared +20 CI
[+0.00399729, +0.01564535] passes local transfer; about85% of the gain measured
at +5 is lost by +20. No-update replay error is zero on both panels.
Raw total-on-H4 projections are 0.8092 / 0.9911 / 1.0117 at the three traced
updates: near-total reference cancellation is not observed at those points.
No fresh classifier was trained, so there is no new AUC/closure result.
Full evidence and limitations: `artifacts/checkpoint_transfer_6704e92e/REPORT.md`.

## Question

At the latest saved plateau checkpoint in the supervised-global-FiLM DGPO
lineage, can one native production update improve expected installed-classifier
reward on independent event identities? Do not assume conditioning, ESS, or
velocity-MSE is the cause.

## Evidence and scope

Previously inspected histories: 31bc2755, b07cbcc0, ad827b1f, f93f9788,
99de15bc. Fresh audit AUC improved from 0.896864 at step 0 to 0.870426 at
600, then returned to 0.886787 at 1100. Latest run ended with recorded policy
updates through 1137; this does not establish that step 1137 was checkpointed.
At round 7, mean raw reward was -1.413404 over updates 1051–1060 and
-1.396461 over 1128–1137. These are unpaired training batches, not a precise
estimate of the effect of an individual update.

Historical h4grad01/h4opt01 established useful local directions for older
snapshots. They do not identify the current plateau mechanism. Their scripts
hard-code old artifacts, member counts or source steps; do not launch them
unchanged against the new model.

## Stage 0: measurement and state contract

- Resolve last.ckpt once and pin its inode (hard link; copy across filesystems); report actual saved
  epoch, global step, reward round and next epoch. Do not infer them from W&B.
- Restore raw policy, AdamW moments and counters, scheduler, installed reward
  and its paired reference. Never use EMA or reset the reference.
- Verify a no-update replay produces identical samples/scores with fixed
  initial DDIM noise; policy and classifier evaluation modes must match the
  production deterministic sampling path.
- Fix 32,768 held-out event identities, disjoint from update batches, with K=8
  paired candidate noises. Keep identity matching across all comparisons.
  A single training seed is used; bootstrap resamples events, not candidates.
  Process 2,048 events/rank as 16 batches of 128 on each of 16 GPUs. Selection
  happens once from the external validation dataset, not from classifier scores.
  Policy training stays at 512 events/rank. Visible-input fingerprints check
  duplicates across validation shards and overlap with every policy update batch.
  This is held out from policy updates, not a never-used final classifier test.
- Preserve production loss, detached nonlinear gate, batch size, K, timestep
  sampling, all trainable parameter groups, optimizer and velocity-MSE
  coefficient 1. Use 16 GPUs; no multi-seed training sweep.
- Record unscaled reward and reference gradient norms and dot product from
  the same update batch. Recombined gradients must match production gradients
  within dtype-appropriate numerical tolerance before interpreting them.

## Stage 1: native one-update replay

Compare the unchanged checkpoint with one native AdamW update in a disposable
state copy. Freeze reward and reference, and regenerate policy samples using
the same held-out identities and initial noises. No classifier fits, refits,
checkpoint overwrites or production training-state mutations.

Primary endpoint: mean per-event reward change, paired across candidates.
Report its event-bootstrap 95% interval, median event change and fraction of
events improving. Retain native per-update velocity-MSE and realized AdamW
parameter-displacement diagnostics (not a new held-out velocity-MSE estimate).
Also evaluate the same paired change on the first update batch to distinguish
immediate fit from held-out transfer. At +5/+20 this remains the first update
batch, not all subsequent training data.

- Interval wholly above zero: supports local reward transfer, not distribution
  closure and not proof that the effect is sufficient for sustained progress.
- Interval wholly below zero: reproduces a local failure; inspect actual
  displacement versus reward/reference gradients before changing architecture.
- Interval crossing zero: unresolved; first assess measurement precision.
  Do not declare failure merely because a tiny effect is not significant.

Run the same unchanged trajectory to updates 5 and 20 regardless of the +1
effect (unless validity checks fail). This avoids outcome-dependent stopping.
Update 20 is the predeclared primary endpoint; intermediate points are diagnostics,
not selections. Pointwise bootstrap intervals quantify event uncertainty on
this one realized trajectory, not optimizer-seed uncertainty or simultaneous
coverage across all inspection points.
No budget extension unless its result changes the next decision.

## Implementation and launch

The launcher `scripts/diagnose_dgpo_checkpoint_transfer.py` reads
`config/dgpo_checkpoint_transfer.yaml`, inheriting the current production
global-FiLM architecture/data/optimizer settings. It reads actual saved counters
from the latest production `last.ckpt`, pins it under a new output directory,
and calls the native DGPO trainer with an absolute cap of saved step +20.
It never reconstructs or substitutes the training objective. Startup/scheduled
refits and cold audits cannot run in probe mode. Ordinary validation is replaced
by these paired panels; production training without probe mode is unchanged.

Source:
`/pscratch/sd/y/yiren/Ztautau/dgpo_global_film_diffusion_one_iteration/checkpoints/last.ckpt`

Run inside the existing 16-GPU Ray allocation:

```bash
cd /global/u2/y/yiren/ml_pipeline
shifter --image=registry.nersc.gov/m2616/avencast/evenet:1.3 \
  python3 -u scripts/diagnose_dgpo_checkpoint_transfer.py
```

Optional `--dry-run` validates/pins and prepares configuration without starting
training. Optional `--events-per-rank N` increases/decreases validation only;
the global panel contains 16*N distinct events. The default is 2,048 per rank.

Every launch creates a new directory under
`/pscratch/sd/y/yiren/Ztautau/dgpo_checkpoint_transfer/` and a fresh W&B run:
**Does late reward still transfer? | native DGPO | paired 20-update probe**.
The original source/checkpoint directory and run are never written.

Artifacts: `source_metadata.json`, resolved `runtime.yaml`, frozen CPU input
panels, per-rank per-endpoint NPZ candidate/reward arrays, and
`measurements/report.json`. The report has `status=complete` only after both
panels finish at +20. Readable W&B axes use `checkpoint_transfer/relative_step`;
the native gradient-transfer metrics retain the original absolute policy clock.
Report gradient reconstruction error must be <=1e-3 before interpretation.

Original checkpoints do not save RNG/data-iterator state. Policy, installed
reward, paired reference, AdamW and scheduler resume; the resumed trajectory is
not a bitwise continuation of the interrupted run. Evaluations restore Python,
NumPy and Torch RNGs and all per-module train/eval modes. No-update replay must
match samples and scores to max absolute error <=1e-6 on both panels.

## Conditional next rounds (not automatic arms)

- Native update fails but a smaller displacement along that exact direction
  helps: investigate finite-step scale. Do not reset AdamW at the same time.
- Reward and reference conflict: test one matched constraint intervention only
  after this conflict is measured; a small velocity-MSE scalar alone is not
  evidence of a small opposing gradient.
- Installed reward improves reliably through 20 updates: move to independent
  classifier validation. A frozen judge supplies a local diagnostic, not fresh
  best-response closure. Adequately train matched fresh endpoint audits before
  claiming distribution improvement; keep a disjoint final test set.

## Round boundary

Summarize observation, validity, hypothesis changed and next single question
before another experiment. Do not add architecture, new features, a new toy,
ESS tempering or refit schedules in this first round. User submits NERSC jobs.
