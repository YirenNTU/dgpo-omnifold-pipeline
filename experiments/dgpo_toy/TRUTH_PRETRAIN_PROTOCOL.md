# Full-truth target diffusion, fixed dataset + early stopping

2026-09-21. User corrected the scientific setup: the generator must see full
truth targets and discover the joint structure from ordinary diffusion training.
The old Gaussian-teacher baseline experiments are preserved, NOT relabelled.
The proposed fresh-sample20k budget was superseded BEFORE launch by the user's
fixed-dataset/early-stopping instruction. No20k truth run was launched.

## Round contract

- Objective: train the original small conditional denoiser on complete truth
  targets, without supplying the analytic joint relationships to its loss/input.
- Success measure: valid standard-training run, loss-selected raw checkpoint,
  held-out velocity MSE and independent generation diagnostics. Joint learning
  is measured, not forced to fail or required to leave residuals.
- Limiting factor: learning from sample-based supervision in the original model.
- First-principles claim: the missing joint structure is present in targets;
  the learner should be given the opportunity to discover it.
- Deliverable: reusable fixed dataset, early-stopped raw model, resumable state.
- Time box: no fixed update/epoch cap; validation patience20 is the stop rule.
  Expect minutes locally, report progress rather than promise a convergence time.
- Delete: Gaussian analytic teacher, old checkpoint initialization, classifier,
  DGPO, reward/KL, truth-dependent input features, architecture/LR sweeps and EMA.
- Authority/risk: local toy only; new paths preserve all earlier artifacts.
- Evidence debt: small architecture and finite data can limit learning. An
  early-stopped loss plateau does not establish optimality or a production failure.

## Dataset and information boundary

Prepare ONCE and save `truth_dataset_seed17_v1/dataset.pt` plus JSON manifest:
train32768,validation8192,test16384, independent seeds40017/50017/60017.
Each row contains only3-D condition and12-D COMPLETE truth target. Truth uses
the same90% structured/10% baseline mixture, kappa8, four context-dependent
triple-phase copulas. No clean-structure labels, group identities or phase sums
are passed to the learner. Training reloads/reuses these fixed rows.

Only dataset preparation and evaluation know the truth construction. Network
gets raw noisy target, condition and its existing time embedding. No CDF or
coordinate/joint Fourier transformation enters this denoiser. Time sin/cos
embedding remains unchanged. No truth statistic is added to training loss.

## Standard sample-target training

Fresh random initialization (seed17); original3-hidden-layer width64 MLP,
original velocity parameterization and cosine VP schedule. For each target y:

    t ~ Uniform[0,1], epsilon ~ Normal(0,I)
    x_t = alpha(t)*y + sigma(t)*epsilon
    target_v = alpha(t)*epsilon - sigma(t)*y
    loss = mean((model(x_t,t,c)-target_v)^2)

Shuffle the finite train set each epoch, resample t/epsilon every visit, batch512,
AdamW lr1e-3/decay.001, clip1, constant LR; no scheduling or warmup changes.
64 updates/epoch. Raw weights only. Dataset targets are never resampled during fit.

## Selection, stopping and monitoring

- Every epoch: event-weighted train MSE; fixed8192-event validation MSE using
  fixed time/noise seed21017;10 time-bin MSE/counts; max gradient norm/clipping;
  epoch/update clocks, LR, best epoch, stale epoch counter.
- Stop after20 consecutive epochs without an absolute validation-MSE improvement
  greater than1e-4 from the last significant-improvement anchor. Smaller gains
  can accumulate against that anchor. No forced minimum/hard maximum steps.
- Select best raw checkpoint by MINIMUM observed validation MSE, independently
  of the significant-improvement counter. Do not select by joint diagnostics.
- Every10 epochs (and stop): DDIM20 generation on fixed4096 conditions x8,
 56 higher-order moments and marginal mean/variance/covariance. These are
  observer diagnostics and cannot alter training, checkpoint choice or stopping.
- Only AFTER stopping/selection: evaluate the untouched16384-event test split
  with noise22017. Evaluate initial/last/selected generation on fresh92017,
 4096 conditions x8;500 paired context bootstrap draws for structural MSE change.
- Lower-order drift and residual joint structure are reported honestly. Neither
  gets hidden by checkpoint selection; learning the full truth is a welcome result.

Every complete epoch saves atomic model/AdamW/RNG/stopping/history state.
Interruptions resume from last completed saved epoch, not a half-applied update.
best_model.pt is the selected RAW model; last_state.pt is full resume state.
New truth checkpoints are NOT disguised as old reward.pt classifier bundles.

## Launch

```bash
python -u -m experiments.dgpo_toy.truth_pretrain \
  --dataset artifacts/dgpo_toy/truth_dataset_seed17_v1/dataset.pt \
  --output artifacts/dgpo_toy/truth_diffusion_earlystop_v1 \
  --patience 20 --min-delta 0.0001 --structure-every 10
```

To continue an INTERRUPTED run, use the same dataset/settings and add
`--resume-from <old-output>/last_state.pt` with a new output path. This preserves
the early-stop counter and optimizer rather than reopening a completed run.
100 tests passed before launch: fixed truth data, ordinary v-target identity,
no teacher/oracle reward/resampling, deterministic validation, early stop,
and exact uninterrupted-vs-resumed model/AdamW/RNG/history/endpoints.

## Completed first run

Output `artifacts/dgpo_toy/truth_diffusion_earlystop_v1`. Process exited0;
early stopping at epoch30/update1920; selected raw best epoch25/update1600.
The final significant improvement anchor was epoch10, MSE1.001978951;
epoch25's absolute best1.001921082 improves it by less than1e-4, so it does
not reset patience. This is intentional, tested, and distinct from best selection.
There is no hard step cap. Recorded training/evaluation runtime2.98s, excluding
dataset setup/initial monitoring/Python startup.10444 model parameters,1 CPU thread.

Fixed dataset stored at `truth_dataset_seed17_v1/dataset.pt` and its JSON manifest.
Verified finite targets in all three splits, truth joint means.842818/.839006/
.837389 (analytic expectation~.841712). The structure IS present in targets;
this is not the old deliberately independent Gaussian dataset.

| Metric | Random initial | Selected raw model |
| --- | ---: | ---: |
|Validation velocity MSE|1.081358950|1.001921082|
|Untouched test velocity MSE|1.076511355|1.000658282|
|Independent56-moment RMSE|.491956290|.491903278|
|Max conditional-mean residual magnitude|.115377461|.034369594|
|Max marginal variance error|.182041580|.132766517|
|Max residual pair covariance|.212266047|.023120084|

Selected-vs-initial structural MSE change-.0000521564,
95%paired-bootstrapCI[-.00134371,.00135929]: unresolved, not proof of an exact
zero change. Lower-order summaries improve but variance error remains~13.3%;
do not call generated marginals exact. Joint signal was not a training/stop criterion.
The original DDIM20 schedule was not adjusted to conceal sampling error.

Verified last model/AdamW clock1920, selected weights match stored best state,
raw/no EMA, finite parameters. Focused10 tests also pass after manifest addition.

- Outcome: complete fixed-dataset, ordinary full-target diffusion training with
  the requested early stop; best raw checkpoint and full resume state retained.
- Evidence: lower held-out MSE, dataset signal checks, exact state clocks and tests.
- Decision: supports correct sample-based learning setup; meaningful joint
  learning unresolved at the chosen validation-based stopping point.
- Learning index: one setup distinction settled (truth was actually supplied);
 2.98s recorded fit/evaluation. No mechanism-of-residual claim established.
- Deleted: teacher pretraining, streaming targets, fixed20k budget, classifier/RL.
- Next limiting factor: residual joint learning under this standard trained model,
  with architecture/data/stopping/sampling explanations not yet separated.
- Better next round: use this actual truth-trained checkpoint when testing
  residual correction; do not relabel old Gaussian-teacher results as this model.
