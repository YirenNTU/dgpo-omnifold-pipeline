# Does condition Fourier help diffusion pretraining?

Status: COMPLETED locally on 2026-09-22 after explicit user authorization.
Offline W&B run `mr15pebb`; results in `artifacts/dgpo_toy/cube_truth_fourier_v1`.
All six fits early-stopped; 96000 total updates, 114.01 seconds. All three seeds
pass the prespecified joint-gain and corner-shape rules at matched budget and
final selection. See the output RESULTS.md for numbers and inference limits.
Local CPU toy only. Do not upload to NERSC.

## One question

Does exposing a multiscale condition basis FROM THE FIRST PRETRAINING UPDATE
improve held-out conditional generation under ordinary velocity-MSE training?
This is not the earlier zero-adapter DGPO rescue experiment. No classifier,
oracle reward, KL penalty, transport, or policy optimization is executed here.

## Fixed truth and model

- One saved dataset, shared by ALL arms and training seeds: 131072 train,
  8192 validation, 8192 test observed `(c,y)` rows. Dataset seed 17; splits have
  independent random streams. No streaming replacement of training targets.
- Continuous `c ~ Uniform[-1,1]`; eight Gaussian cube corners `(+-1,+-1,+-1)`
  with coordinate width .15. Positive-parity probability is
  `p(+|c)=.5+.4*sin(8*pi*c)`, uniform among the four corners in either parity.
  The first two signs are uniform; only their joint relation with the third
  depends on c. Marginal means are zero, variances 1.0225, pair covariance zero.
- Dataset construction explicitly calls `sample(..., truth=True)`. No biased
  reference mixture or old pretrained checkpoint is used.
- 3-output, scalar-condition, three-hidden-layer width-128 SiLU velocity model;
  existing time embedding and DDIM50 unchanged, raw weights, never EMA.
- Raw c remains present in BOTH arms. Eight added channels are zeros in raw,
  or unit-variance sin/cos at frequencies 1,2,4,8 in Fourier. A zero-initialized
  8x128 projection adds to the first hidden preactivation. Both variants exist
  before the first pretraining update and all network parameters can train.
- Identical initial velocity and a two-step DDIM probe are checked. Both have
  35715 stored parameters; raw's zero-feature adapter is inert. This is NOT an
  equal-effective-capacity control. The prior polynomial DGPO control does not
  replace a pretraining capacity ablation, and no such causal claim is made.
- The basis deliberately includes known k8. No parity/product feature of y,
  target mode label, analytic score, or truth velocity enters the model/loss.
  Frequency-aligned inductive bias is still present; this is not a proof of a
  physics-prior-free solution for EveNet.

## Training and stopping

- Three paired training seeds 17,23,41 (six fits), all on the same fixed dataset.
  Within each pair, network initialization, shuffled minibatch identities,
  diffusion times and Gaussian noise are matched exactly. Generate each batch
  once, then update both arms. Seeds vary initialization AND training streams.
- AdamW lr=.001, weight_decay=.001, batch512, gradient clipping1, constant LR.
  Ordinary target `alpha(t)*epsilon - sigma(t)*observed_y`; uniform t in [0,1].
  Do not use a known conditional probability or oracle in this loss.
- Fixed validation-noise panel every epoch; select the actual minimum validation
  velocity MSE, independent of the early-stop significance threshold.
- Early stop after 20 epochs without a cumulative absolute 1e-4 improvement.
  No total-step or epoch cap. Each arm stops independently.
- At the FIRST arm's stop, evaluate both validation-selected checkpoints from
  that same available update budget. Then finish the other arm. This reports
  matched-budget efficiency separately from final early-stop quality.
- Resume with `--resume` in the SAME output directory. Restore both models,
  AdamW, RNG, best models and stopping counters from the last complete epoch.
  A mid-epoch interruption replays that epoch; JSONL/W&B may contain duplicate
  points for an interrupted epoch. Completed seed pairs are not retrained.

## Evaluation and predeclared decision

Primary: raw minus Fourier **conditional eight-mode total variation** at the
validation-selected, independently early-stopped checkpoints. Positive is better.
Use 8192 ordered midpoint conditions, K=8 independent samples each, fixed latent
noise seed880017 common to both arms and all evaluation checkpoints. There are
256 equal-width condition bins. Compare empirical eight-mode probabilities
with analytic truth averaged over the EXACT condition grid in each bin.
Endpoint samples are NOT used for stopping or model selection.

Bootstrap paired CONDITION rows within each bin, retaining all K candidates
together, 256 replicates, percentile95% interval. This estimates sampling
uncertainty conditional on fitted models, not training-seed or dataset uncertainty.
Report an independently sampled analytic-truth floor, not an expectation of zero
empirical TV. No training dataset is resampled across the three seed pairs.

Support this bounded hypothesis only if EVERY training seed has:

1. Raw-minus-Fourier TV > .05, with bootstrap lower bound > 0.
2. Fourier near-corner fraction at most .02 below raw.

Report the matched-budget comparison independently. A one-seed run is a pilot,
not a replicated conclusion. A failed gate is unresolved, not proof Fourier
cannot work. No claim of exact distribution closure even after a pass.

Secondary diagnostics: conditional parity RMSE, E[sin(8*pi*c)*parity] (truth .4),
marginal mean/variance errors, off-diagonal covariance, corner residual RMS,
near-corner fraction, maximum coordinate, held-out velocity MSE by noise decile.
Joint generation diagnostics every10 epochs on a separate1024-condition panel;
training/validation, clipping, LR, epoch and update clocks every epoch. Record
generation-diagnostic wall time separately from paired training time.

W&B default is offline, project `dgpo-toy`, group `Conditional cube truth pretraining`.
Display name: `Can Fourier improve pretraining? | truth cube k=8 | raw vs Fourier`.
Each seed/basis has its own epoch axis. Local report, data, selected raw models,
full resume state, and endpoint sample arrays are saved for independent inspection.

## Launch (user submits)

```bash
cd /Users/yirenwu/Ztautau/ml_pipeline
/opt/miniconda3/envs/MyEve/bin/python -u -m experiments.dgpo_toy.cube_truth_fourier \
  --output artifacts/dgpo_toy/cube_truth_fourier_v1 \
  --seeds 17 23 41 \
  --patience 20 --min-delta 1e-4 \
  --monitor-every 10 --threads 2 \
  --wandb-mode offline
```

Repeat the same command with `--resume` after an interruption. To log live to
the already authenticated W&B account, explicitly choose `--wandb-mode online`.
Do not run two processes against the same output. No remote job is submitted.

## What the result can change

If joint error improves before any RL, condition representation can limit
supervised denoising for this toy. If only matched-budget quality improves,
it is a learning-speed result rather than a better final fit. If neither
improves, the DGPO-stage rescue does not transfer to ordinary pretraining.
Production architecture, new classifier fits, and subsequent DGPO are separate
experiments and are not automatically launched by this runner.

## Implementation handoff (before the subsequently authorized launch)

- Outcome: runnable local A/B and restartable paired fits prepared; no formal job launched.
- Evidence: 45 relevant tests passed (including15 new tests); CLI and W&B name validated.
- Decision: unresolved scientifically; the proposed pretraining intervention has not run.
- Learning index: implementation/test cost divided by zero scientific decisions is
  still infinite; do not count test success as research evidence.
- Deleted: classifier, DGPO, KL, EMA, teacher-score supervision and remote submission.
- Next limiting factor: measured held-out conditional generation after pretraining.
- Better next round: retain matched-budget and final-fit comparisons as separate claims.
